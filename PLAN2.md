# One Layer Deeper — Non-Transformer Exploration Space (Agent Handoff v2)

**Repo:** `github.com/tilde-research/one-layer-deeper`
**Context:** the full transformer-oriented plan (v1) was run to completion. Depth sweeps, depth curricula, optimizer swaps, batch size, loss design, adaptive computation, throughput work, parameter allocation — no result.
**Working conclusion:** that is not a tuning failure. It is the signature of an **expressivity** failure, and there is a specific body of theory that predicts it.

---

## 0. Why v1 failed — the reframe

Iterated function composition is the canonical hard case for parallel computation.

Two facts do most of the work here:

1. **Composition is associative.** `f_n ∘ … ∘ f_1` can be computed by a parallel prefix scan in `O(log n)` sequential depth instead of `O(n)` — *if* your architecture represents each `f_i` as an element of an associative operator you can scan over.
2. **The algebraic structure of the function set determines the minimum expressivity of that operator.** If the functions generate a **non-solvable** monoid (any transformation monoid on ≥5 elements contains `S₅`), then composition is `NC¹`-complete by Barrington's theorem, and constant-depth architectures provably cannot do it in one forward pass.

Merrill et al. showed that both transformers and diagonal SSMs sit in `TC⁰` and <cite index="16-1">are theoretically incapable of emulating FSAs with non-solvable transformation semigroups</cite>. That is a proof, not a tuning gap. A fixed-depth transformer needs depth growing with sequence length to compose, which under a wall-clock budget is exactly the tradeoff v1 kept losing.

So the v1 result is informative: it is weak evidence that the task's composition monoid is non-solvable, or at minimum that it requires serial state tracking beyond `TC⁰`. **That is the hypothesis this plan is built to exploit.**

### The design target

> Learn a representation in which each input element maps to a **matrix**, so composition becomes **matrix product**, then compute the composition with a **parallel associative scan**. One layer. `log T` depth. Full expressivity.

Everything below is either an instance of this, a cheaper approximation of it, or a control.

### Compliance note

Your "square modulo" characterization needs to have come from the public task description, not from looking at `data/generated/`. Rule 10 forbids data inspection and task-specific solvers, and a design whose justification traces to observed data is at risk even if the code looks generic. **The architecture sweep below is a legal probe for the same information** — which expressivity class works is inferred from your own training curves, not from the data. Use that route.

---

## 1. The organizing axes

Classify every candidate on two axes. This is the whole space; the named architectures are just points in it.

### Axis A — transition operator expressivity (ordered, cheapest first)

| Parameterization | Expressivity | Cost per step |
|---|---|---|
| Scaled identity | trivial | ~0 |
| Real diagonal, eigenvalues `[0,1]` (Mamba, GLA, RWKV-ish) | cannot even do parity | `O(d)` |
| Diagonal, eigenvalues `[-1,1]` | parity, abelian/solvable groups | `O(d)` |
| Complex diagonal | + oscillatory / cyclic structure | `O(d)` |
| DPLR / single Householder `I − βkkᵀ` (DeltaNet) | non-commutative, partial state tracking | `O(d)` |
| Product of `n_h` Householders (DeltaProduct) | tunable — approaches dense as `n_h→d` | `O(n_h·d)` |
| Permutation × diagonal (PD-SSM) | **any N-state FSA, 1 layer, dim N** | ~diagonal |
| Dense `d×d` | maximal | `O(d²)` or `O(d³)` scan |
| Non-linear RNN (LSTM/GRU) | maximal, but no scan | `O(T)` serial |

Two hard results to design against:
- Eigenvalues restricted to `[0,1]` **cannot solve parity**; extending to `[-1,1]` fixes it at no extra cost — for DeltaNet this is just `I − 2βkkᵀ` instead of `I − βkkᵀ`. <cite index="6-1">Non-triangular matrices are additionally needed to count modulo 3, and LRNNs whose transitions are products of identity-minus-outer-product matrices with eigenvalues in [−1,1] can learn any regular language.</cite>
- <cite index="13-1">PD-SSM parametrizes the transition as a column one-hot matrix times a complex diagonal matrix, is BIBO-stable, and can emulate any N-state FSA with one layer of dimension N plus an N×N linear readout — the strongest guarantee among structured SSMs, at parallel-scan cost comparable to diagonal.</cite>

### Axis B — how serial depth is realized

- **Fully sequential RNN** — `O(T)` depth. Max expressivity, worst wall clock. Viable only if `max_seq_len` is small.
- **Associative scan** — `O(log T)` depth. Requires the operator be associative. Matrix product is.
- **Chunkwise parallel** — sequential across chunks, parallel within. Best hardware utilization; the DeltaNet/Flash-PD-SSM approach.
- **Fixed unrolled tied depth** — Neural GPU / Universal Transformer style. Depth is a hyperparameter, not `log T`.

### Axis C — where state lives

Fixed vector · fixed matrix (linear-attention family) · addressable external memory (NTM/DNC) · tied gated workspace.

---

## 2. Phase 0 — Diagnosis before construction

Do not build until these are answered. All are legal (they read your own training curves, never the data).

1. **Expressivity vs optimization.** Train a small **non-linear LSTM** — maximally expressive, slow, guaranteed to be the wrong wall-clock answer — at whatever tiny scale fits the Easy budget. If it gets signal where every v1 model got none, the binding constraint is expressivity and this plan is correct. If it also gets nothing, the problem is elsewhere (data scale, loss, or a harness bug) and stop.
2. **Length scaling.** Accuracy vs sequence length for a fixed model. A clean cliff at some length is the state-tracking signature. Flat-and-low means something else is wrong.
3. **Train-vs-eval split.** Training loss → 0 with low eval accuracy is a length-generalization problem. Training loss plateauing high is expressivity or optimization.
4. **Solvability probe.** Diagonal-`[0,1]` vs diagonal-`[-1,1]` vs DeltaNet-`[-1,1]` at matched compute. The ordering of these three tells you which algebraic class you are in — this is the legal substitute for inspecting the data.

**Record the answer to (4) before doing anything else.** It determines whether half of §3 is even necessary.

### Environment questions to resolve at the same time

- Is `triton` importable? It ships with torch, so probably yes — which puts custom chunkwise scan kernels on the table and makes your kernel background load-bearing rather than incidental.
- Is `torch._higher_order_ops.associative_scan` available in the pinned torch version? If not, the log-depth doubling implementation in §5 works in plain PyTorch.
- What are `vocab_size` and `max_seq_len` per tier? `max_seq_len` decides whether the sequential-RNN branch (§3.6) is alive or dead.

---

## 3. Candidates, ranked

Each: mechanism, why it fits, cost, falsifier.

### 3.1 Learned monoid + matrix associative scan — **build this first**

**Mechanism.** Map each token to a small `d×d` matrix `M_i` (via a low-rank or structured parameterization). Compute all prefix products `P_k = M_k ⋯ M_1` with a parallel scan. Read out from `P_k`. Optionally constrain `M_i` toward permutations, orthogonal matrices, or doubly-stochastic matrices.

**Why.** This is the direct implementation of the target. If the task is composition over a group of order ≤ `d`, the model can in principle learn the regular representation exactly, and the scan gives it in `log T` depth in **one layer**.

**Cost.** Log-depth doubling: `log₂(T)` batched matmuls over a `(B, T, d, d)` tensor. `d=16, T=512, B=256` → ~33M elements per level, trivially affordable. `d` is the critical knob — memory goes as `d²` and dense-scan compute as `d³`. Start at `d ∈ {8, 16, 32}`.

**Falsifier.** No signal at any `d` up to memory limit, with matrix-valued and permutation-constrained variants both tried. That would mean composition is not the bottleneck and §0's whole reframe is wrong.

### 3.2 PD-SSM (permutation × diagonal)

**Mechanism.** Transition `A_t = P_t D_t` where `P_t` is column one-hot (a permutation/function on states, produced via straight-through argmax over a softmax) and `D_t` is complex diagonal. Scan cost is linear in state size, not cubic.

**Why.** Best expressivity-per-FLOP in the literature: exact FSA emulation with minimal state size at near-diagonal cost. If the task is a finite-state composition, this is the theoretically optimal shape. The one-hot `P` is also a strong inductive bias toward *exactly* the "each token is a function on states" structure that a composition task has.

**Cost.** Comparable to diagonal SSM. Forward uses hard sparsity; backward uses the softmax gradient.

**Falsifier.** Underperforms §3.1 at matched wall clock — which would mean the sparsity bias is wrong for this task.

### 3.3 DeltaNet / DeltaProduct with extended eigenvalue range

**Mechanism.** Transitions as products of generalized Householder matrices `I − 2β_t k_t k_tᵀ`, `β ∈ [0,1]` giving eigenvalues in `[-1,1]`. DeltaProduct applies `n_h` such factors per step, making expressivity a tunable knob.

**Why.** Mature chunkwise-parallel training algorithms exist, so it is the most likely to be *fast* rather than merely correct. `n_h` gives you a clean expressivity/compute sweep — exactly the kind of single-axis experiment §4 of v1 was built around. <cite index="8-1">DeltaProduct was evaluated directly on the S3, S4, A5, and S5 permutation-group word problems with the extended eigenvalue range.</cite>

**Cost.** Chunkwise, near-linear-attention throughput.

**Falsifier.** Accuracy flat in `n_h`. If more Householders don't help, non-commutativity isn't what's missing.

**Do not skip the eigenvalue detail.** Running DeltaNet with `β ∈ [0,1]` and concluding "DeltaNet doesn't work" would be the single most likely wasted week in this plan.

### 3.4 Tied gated recurrent workspace (your proposal — formalized)

**Mechanism.** A workspace `W ∈ ℝ^{m×d}` (matrix-valued state, `m` slots). One tied block, applied `k` times, with gated read/write:
```
r_t = softmax(q_t Wᵀ) W          # content-addressed read
W  ← (1 − g_t ⊙ a_t) ⊙ W + g_t ⊙ a_t ⊙ v_t   # gated slot write
```
Weight tying keeps you far under the 500M ceiling, which frees the budget for step count.

**Why.** This is the general form that DeltaNet (`m=d`, rank-1 write), NTM (soft addressing), and fast-weight programmers are all special cases of. Worth building as a *parameterized family* so the ablations move between named architectures continuously rather than as separate rewrites.

**Cost.** `O(k·m·d)` per token. `k` and `m` are the knobs; both trade directly against step count.

**Falsifier.** Best configuration reduces to `m=1` or `k=1` — meaning neither the workspace nor the recurrence is earning its wall clock.

### 3.5 Neural GPU / tied convolutional recurrence

**Mechanism.** A convolutional GRU applied `k` times over a 2D state grid with fully tied weights.

**Why.** Purpose-built for algorithmic tasks with tied weights and length generalization; "one layer deeper" is literally its design axis. Cheap in parameters, high in compute.

**Cost.** `k` conv-GRU applications. Highly parallel, good H100 utilization.

**Falsifier / known risk.** Notoriously hard to train — needs curriculum learning, gradient noise, and careful init, and generalization is brittle. Budget for the possibility that it fails to optimize rather than fails to express. Timebox it.

### 3.6 Sequential non-linear RNN, aggressively fused

**Mechanism.** A plain LSTM/GRU, but with the recurrence fused into as few kernels as possible.

**Why.** Non-linear RNNs remain the top performers on state-tracking benchmarks — the constraint has always been scaling, not capability. In this competition **sequences are short and the model is small**, which is exactly the regime where the usual objection is weakest. If `max_seq_len` is a few hundred, `O(T)` serial steps at small `d` may be entirely affordable, and you get maximal expressivity with no theoretical caveats.

**Cost.** `O(T)` serial. Dominated by kernel launch overhead — which is precisely what a fused Triton implementation fixes.

**Falsifier.** Step time makes the achievable update count non-competitive. Measure this in Phase 0 rather than assuming it.

**This is the highest-variance entry in the plan and the one most likely to be under-explored by other competitors.**

### 3.7 Hybrid: chunked divide-and-conquer

**Mechanism.** Compose exactly within fixed chunks (sequential or scanned), emit a chunk summary as a composed operator, then compose summaries across chunks with a second-level scan or a small transformer.

**Why.** Associativity means chunk summaries are exact, not approximate — unlike the usual hierarchical-attention story. Gives an explicit knob on the serial/parallel split for hardware tuning.

---

## 4. Deprioritized — with reasons

**JEPA — do not pursue as a primary approach.** JEPA is a self-supervised *objective* for learning representations by predicting latent embeddings of masked targets. This competition supplies labels, scores exact accuracy, and fixes the loss signature to `(logits, labels, aux)`. There is no representation-transfer phase to exploit and no unlabeled corpus. The one legitimate use is as an **auxiliary consistency term routed through `aux`** — e.g. penalizing disagreement between the state reached by composing `k` steps and the state predicted directly. That is a regularizer worth one experiment, not an architecture.

**NTM / DNC — low priority.** Everything the differentiable-memory line offers is present in §3.4 in a cheaper form. The full apparatus (content + location addressing, temporal link matrix, allocation weighting) is slow per step, famously unstable to train, and inherently sequential. Under a hard wall clock this is close to a worst case. If the workspace in §3.4 shows that addressable memory is what's carrying performance, *then* consider adding DNC mechanisms one at a time.

**Fast / FFT convolutions (Hyena, H3, long convs) — useful, but not for the core.** An FFT convolution is linear time-invariant. LTI operators commute, so they sit at the bottom of Axis A and cannot express non-solvable state tracking, for the same reason diagonal SSMs cannot. Use them as **cheap token mixing around** an expressive recurrent core, never as the composition mechanism itself. If Phase 0 probe (4) says the structure is abelian, this judgment changes and long convs become genuinely attractive.

**Vanilla Mamba / Mamba-2 — control only.** Real diagonal transitions with non-negative eigenvalues. Include it to confirm it fails, and to calibrate the probe in Phase 0. Do not tune it.

---

## 5. Implementation notes

**Matrix associative scan in plain PyTorch.** No custom kernel needed for a first result. Log-depth doubling:
```python
# M: (B, T, d, d)
n = 1
while n < T:
    shifted = pad_identity_left(M, n)     # shift by n, fill with I
    M = torch.matmul(M, shifted)          # composition order matters — verify against a serial loop
    n *= 2
```
`log₂(T)` batched matmuls, fully parallel, `torch.compile`-friendly. **Write a serial reference implementation and assert equality first** — off-by-one and left/right composition-order bugs here are silent and will look like "the architecture doesn't work."

**Parameter budget.** Every architecture here is small. The 500M ceiling is not binding; weight tying makes it irrelevant. Spend the freed budget on update count and on `d`.

**Numerics.** Long products of matrices explode or vanish. Constrain: orthogonal/Householder parameterization, spectral normalization, or explicit renormalization each scan level. Run the scan in fp32 even if the rest is bf16 — the prefix products are the one place where bf16 will silently destroy a correct model.

**Straight-through estimators (PD-SSM).** Hard one-hot forward, softmax gradient backward. Sweep the temperature; this is the most likely source of a "correct architecture that won't train."

**Promotion protocol.** Unchanged from v1. Local screen → Easy (60/day, must hold across e1–e5 and exceed the variance floor) → Medium (6/day, expect overhead-driven reversals) → Hard (1/day, ranked queue, never an untested change).

---

## 6. Kill criteria

- Phase 0 step 1: if a non-linear LSTM also gets nothing, this entire plan is built on a wrong diagnosis. Stop and re-examine the harness, the loss, and the label alignment before writing another architecture.
- Any candidate that fails to beat the §3.1 scan baseline at matched wall clock after one honest tuning pass gets archived with its curves. Do not re-litigate.
- If probe (4) shows diagonal-`[0,1]` performs the same as DeltaNet-`[-1,1]`, the task is not a state-tracking problem, §0 is wrong, and the space to explore is a different one entirely.

---

## 7. Reading list (in priority order)

1. Merrill, Petty & Sabharwal — *The Illusion of State in State-Space Models* (2024). Why diagonal SSMs and transformers both fail here.
2. Grazzi et al. — *Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues* (arXiv:2411.12537). The eigenvalue-range fix; free expressivity.
3. Terzić et al. — *Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models* (PD-SSM, arXiv:2509.22284), and the Flash PD-SSM follow-up (arXiv:2605.19150) for the memory-optimized kernel.
4. Siems et al. — *DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products* (arXiv:2502.10297). Group word-problem results.
5. Yang et al. — *Parallelizing Linear Transformers with the Delta Rule over Sequence Length* (2024). The chunkwise algorithm you'd actually implement.
6. Barrington (1986) — bounded-width branching programs and `NC¹`. The source of the non-solvability argument.
7. Kaiser & Sutskever — *Neural GPUs Learn Algorithms* (2015). For §3.5, including its training difficulties.
