# `plan2/pd-ssm-delta` — PD-SSM (§3.2) and DeltaNet / DeltaProduct (§3.3)

**Mandate.** The two highest-expressivity-per-FLOP structured recurrences.
**Status.** Complete for the deliverables named in the mandate.
**Headline.** Both architectures are *built correctly and verified* — the
eigenvalue range, the Householder products, the column-one-hot transitions, the
chunkwise and log-depth scans all check out numerically against serial
references, and both solve word problems that provably require the expressivity
they claim. **Neither moves the scored metric.** `MAX_T = 0` in every cell.
The `n_h` sweep is **flat on the real task while being decisive on a control
task run with the same code** — which is the strongest form the §3.3 falsifier
can take, and it fires.

Every row below is labelled **LEGAL** or **DIAGNOSTIC**. Nothing under
`data/generated/` was read, printed, or summarised. Nothing was submitted hosted.

---

## 0. What axis the recurrence runs over (BRIEF2 §2a)

Stated up front because BRIEF2 corrects PLAN2 on exactly this point.

| axis | length | used here? |
|---|---|---|
| prompt tokens | 13 (e1–e5) / 15 (m1) / 21 (m4) | **yes — this is the scan axis** |
| digit positions within a field | ~3–10 | **yes, as a sub-interval of the token axis** |
| composition depth `T` | ≤ 64 | **no** — `T` is a *field inside the prompt*, not a sequence dimension |
| unrolled internal depth `R` | 1, 2, 4 (`P2_REPEAT`) | tested separately |

So: a scan over the token axis composes **13 to 21 factors**, not `T` of them.
It cannot and does not perform T-fold squaring. That is deliberate — BRIEF2 §2b
says T-fold composition is already solved (exactness composes 1.000 at every
rung; a learned controller reaches MAX_T = 64) and is not the target.

The reason a token-axis scan is nevertheless the right instantiation is BRIEF2
§2c: inside the `[X] <digits>` field consecutive tokens *are* consecutive digits
of `x`, and carry propagation across digit positions is a genuine associative
prefix computation (the propagate/generate carry monoid). A scan over prompt
tokens contains a scan over digit positions as a sub-interval. **That is the
mechanism this branch bets on, and it is the bet that fails.**

`P2_REPEAT=R` applies the whole tied block `R` times, giving composition depth
`R·L` on an unrolled internal axis, as a direct test of "is 13 factors simply
not enough".

---

## 1. Implementation and verification (`lab/p2_verify.py`, all pass)

PLAN2 §5 warns that off-by-one and composition-order bugs in a scan are silent
and look like "the architecture doesn't work". Every scan here is asserted equal
to a serial reference before any training claim is made.

```
$VENV lab/p2_verify.py
```

| check | result |
|---|---|
| §3.1 log-depth doubling == serial left-fold, `T=13, N=8` | max abs err **1.7e-06** |
| §3.3 fused delta rule == explicit product of `I − βkkᵀ` factors | max abs err **3.6e-07** |
| §3.3 chunkwise WY path == sequential reference, `n_h = 1,2,3` | rel err **3.9e-07 / 4.1e-07 / 4.4e-07** |
| §3.2 log-depth affine scan == sequential PD recurrence, `ste = hard, none` | rel err **2.2e-07 / 2.6e-07** |
| §3.2 `P` is exactly column one-hot in the forward pass | column sums 1.000000, nonzeros/column = 1 |
| §3.2 straight-through passes gradient to the softmax logits | `‖∂L/∂W_p‖ = 4.73` (non-zero) |
| §3.2 PD closure `(P₁D₁)(P₂D₂) = P_{σ₁∘σ₂} diag(D₁∘σ₂ ⊙ D₂)` | max abs err **1.5e-08** |
| BIBO stability of the `n_h=3` product over 7 tokens | `max‖λ‖ = 0.414 ≤ 1` |

The PD closure check matters: it is the property that would let a PD-SSM scan
combine two elements in `O(N)` (a gather plus a complex multiply) rather than
`O(N³)`. It holds exactly. Whether it *pays* is §4.

---

## 2. The realised eigenvalue range — the trap PLAN2 §3.3 flags

`I − βkkᵀ` with `‖k‖ = 1` has spectrum `{1−β} ∪ {1}^(d−1)`. The implementation
uses `β = β_max · sigmoid(·)` with `β_max = 2`, i.e. exactly PLAN2's
`I − 2βkkᵀ` with `β ∈ [0,1]`, so the nontrivial eigenvalue lives in `(−1, 1)`.
`P2_EIG=pos` sets `β_max = 1` and is the deliberately-crippled `[0,1]` control.

**At random init** (`lab/p2_verify.py` check 2, `d_k = 16`, explicit numerical
spectra of the constructed matrices, not just `1−β`):

| setting | realised `β` | realised `1−β` | numerical spectrum of `I − βkkᵀ` | frac. eigenvalues `< 0` |
|---|---|---|---|---|
| `pos` (`[0,1]`) | 0.1491 … 0.8247 | **+0.1753 … +0.8509** | +0.1753 … +1.0000 | **0.0000** |
| `neg` (`[−1,1]`) | 0.3159 … 1.6888 | **−0.6888 … +0.6841** | −0.5900 … +1.0000 | 0.0303 |

**After 1,500 training steps** on the synthetic controls (`lab/p2_probe.py`,
measured on held-out eval inputs — this is the number that actually matters,
because the *achievable* range is a fact about the parameterisation and the
*realised* range is a fact about what training does with it):

| setting | `n_h` | realised `eig_min` | realised `eig_max` | frac. negative |
|---|---|---|---|---|
| `[0,1]` | 1 | +0.421 ± 0.415 | +0.967 ± 0.010 | **0.000 ± 0.000** |
| `[0,1]` | 2 | +0.006 ± 0.006 | +0.920 ± 0.085 | **0.000 ± 0.000** |
| `[0,1]` | 4 | +0.000 ± 0.000 | +0.998 ± 0.001 | **0.000 ± 0.000** |
| `[−1,1]` | 1 | −0.613 ± 0.577 | +0.490 ± 0.393 | 0.375 ± 0.217 |
| `[−1,1]` | 2 | **−0.983 ± 0.004** | +0.239 ± 0.270 | 0.688 ± 0.272 |
| `[−1,1]` | 3 | **−0.983 ± 0.005** | +0.563 ± 0.176 | 0.667 ± 0.000 |
| `[−1,1]` | 4 | **−0.981 ± 0.002** | +0.356 ± 0.159 | 0.750 ± 0.000 |

**Verified: the extended range is real and training uses it.** With `[−1,1]`
available, training drives eigenvalues to −0.98 and puts 67–75 % of them
negative. With `[0,1]` the realised minimum is exactly 0.000 and *no* eigenvalue
is ever negative — as it must be. This branch did not run the crippled variant
and conclude "DeltaNet doesn't work"; the crippled variant was run **as a named
control** and it fails exactly where theory says it must (§3).

---

## 3. DIAGNOSTIC — does the sweep have resolving power?

`~950` experiments in this project read `MAX_T = 0`. A sweep that reads "flat"
on the real task is uninterpretable unless it is first shown to *resolve
something*. So the identical layer code (`lab/p2_layers.py` loads the mixers
directly out of the submission file) was run on three word problems of known
algebraic class, causally, over a synthetic sequence axis that **is** the
composition axis:

* **parity** — `Z₂`. Provably impossible with spectrum in `[0,1]`.
* **mod-3 counting** — `Z₃`. Needs a non-triangular transition.
* **A₅** — non-solvable; `NC¹`-complete by Barrington. The canonical hard case
  and what DeltaProduct is evaluated on in the literature.

`lab/p2_probe.py`, `L = 20`, 1,500 steps, batch 128, 2 seeds, eval on 1,024
freshly-generated held-out sequences, plus a `2×`-length extrapolation eval.
**DIAGNOSTIC** — these are not the competition task.

> **Two caveats, stated before the numbers rather than after.**
> (a) The "§3.1" row is *this branch's own* low-rank instantiation
> (`M_t = I + u_t v_tᵀ / √r`, `r = 4`), not the `plan2/matrix-scan` sibling's
> implementation. Read it as a locally-matched reference point, never as a
> statement about their work.
> (b) The main probe runs PD-SSM at `N = 16`, which **cannot** represent A₅'s
> 60 states under PD-SSM's own theorem, so its A₅ cells at `τ ≤ 1` are
> under-provisioned by construction. Probe F re-runs A₅ at `N ≥ 60` to make the
> comparison fair. (As it turns out `N = 16` at `τ = 3` already reaches 1.000,
> because the readout is not restricted to the state — but the fair cells are
> reported anyway.)

RESULTS_PROBE_A

RESULTS_PROBE_B

RESULTS_PROBE_C

RESULTS_PROBE_D

**Reading.**

1. **The eigenvalue fix is worth everything on parity**: `[0,1]` sits at chance
   (0.521 vs 0.500) at every `n_h` tried, `[−1,1]` is 1.000 at every `n_h`, and
   extrapolates to `2×` length at 1.000. Had this branch run `β ∈ [0,1]` it
   would have reported "DeltaNet cannot do parity" — the exact wasted week
   PLAN2 §3.3 predicts.
2. **The `n_h` axis resolves, twice, at the two places theory says it should.**
   On mod-3 (`Z₃`, needs a non-triangular transition): `n_h=1` is 0.512 against
   chance 0.333; `n_h=2,3,4` are 1.000 ± 0.000, including at `2×` length. One
   Householder cannot build the order-3 rotation; two can. On **A₅** — the
   non-solvable, `NC¹`-complete case, chance 0.017 — the per-seed values are

   | `n_h` | 1 | 2 | 3 | 4 |
   |---|---|---|---|---|
   | seed 0 | 0.012 | 0.021 | **1.000** | **1.000** |
   | seed 1 | 0.012 | 0.049 | 0.071 | **1.000** |

   with sequence-level accuracy and `2×`-length extrapolation also 1.000 in
   every bold cell. So: **never solved at `n_h ≤ 2` in any seed, solved in 1/2
   seeds at `n_h = 3`, solved in 2/2 at `n_h = 4`.** The threshold is
   seed-dependent; the ordering is not. **A chance-to-exact transition in `n_h`
   on a non-solvable group word problem is exactly the behaviour DeltaProduct is
   claimed to have, reproduced here with the same code, same budget, and the
   same sweep that is about to be applied to the competition task.** This is
   what makes a flat real-task sweep a *result* rather than an absence of one.

   **And on A₅ the two axes are jointly necessary, which is the cleanest single
   cell in this report**: `[0,1]` at `n_h = 4` reads **0.019** — chance — while
   `[−1,1]` at `n_h = 4` reads **1.000 ± 0.000**. Same parameter count, same
   budget, same code; the only difference is whether `β` may exceed 1. Four
   Householders with non-negative eigenvalues cannot do what three or four with
   negative eigenvalues can.
3. **The PD-SSM temperature is not a nuisance parameter — on the hardest task it
   is the whole result, and it has a sharp interior optimum.** On parity and
   mod-3 *every* temperature gives 1.000, which is exactly the kind of flat
   sweep that would justify quoting one setting. On **A₅** the same sweep reads

   | `τ` | 0.1 | 0.3 | 1.0 | **3.0** | 10.0 | soft (no STE) |
   |---|---|---|---|---|---|---|
   | A₅ last-token acc (chance 0.017) | 0.018 | 0.044 | 0.050 | **1.000** | 0.043 | 0.494 |

   — chance everywhere except a single cell, which is *exact* (sequence-level
   1.000, `2×`-length 1.000). `τ` divides the logits, so larger `τ` is a
   **softer backward pass**, while the forward pass is exactly one-hot at every
   `τ` (verified in §1). So the shape is a genuine interior optimum, not a
   monotone preference: too sharp (`τ ≤ 1`) and too soft (`τ = 10`) both fail,
   and **the standard "anneal the relaxation toward hard" recipe walks directly
   away from the only setting that works.** Had this branch reported a single
   temperature, it would have reported a chance-level one with probability 4/5.
   This is the single strongest argument in this report for the mandate's
   instruction to report the sweep rather than a setting.
4. **Straight-through discretisation is what makes PD-SSM *generalise*, not what
   stops it training.** The `ste=none` soft control on A₅ reads **0.494**
   in-distribution (sequence-level 0.277) — clearly learning something — and
   **0.037 at `2×` length**, i.e. chance. Hard straight-through at `τ = 3` reads
   1.000 and **1.000** at `2×` length. The soft relaxation memorises; the
   discrete forward pass generalises. This is the direct opposite of the prior
   this branch was handed (straight-through as "the known failure mode"), and it
   is measured on the only task in the set that needs the full expressivity.
5. **In-distribution accuracy is fooled; length extrapolation is not.**
   `[0,1]` at `n_h=2` reaches 0.941 in-distribution on mod-3 while its `2×`
   extrapolation is 0.328 — chance. It memorised length-20 sequences without
   learning the recurrence. Any report of these architectures that quotes only
   in-distribution accuracy is quoting a number with a known fooling
   configuration. **This project's screening table (`lab/RESUME.md`) has no
   length-extrapolation row; on this evidence it should.**

---

## 4. Wall clock — and the §3.2 falsifier

> **Framing correction applied mid-branch, from `plan2/sequential-rnn`.** (a)
> `triton` imports but **cannot compile** on this box (`sm_107a is not defined`),
> and the same failure kills `torch.compile`. Nothing in this branch uses either
> — the chunkwise delta rule below is plain PyTorch
> (`torch.linalg.solve_triangular` + `matmul`) and the manifests set
> `compile=false` — so no result here depends on the retracted capability.
> (b) *Parallelism is not the selling point here.* The sibling measures a fused
> sequential LSTM at 0.33× the reference and the most-parallel candidate
> (Neural GPU) at 7.22×, i.e. **22× the wrong way**: PLAN2 §0's `O(T)` vs
> `O(log T)` motivation is asymptotics about a sequence axis this task does not
> have (§0 above). **This branch's data independently confirms that from inside
> the structured-SSM family**: PD-SSM's log-depth associative scan is *no
> faster* than its own sequential loop at `L=13` (11.34 vs 10.94 ms). So the
> table below is reported because the mandate asks for a matched-wall-clock
> comparison and because it establishes that **`n_h` is a pure expressivity
> axis with no compute price** — not because scan cost is the interesting
> variable. Judge these architectures on expressivity and trainability.

PLAN2 §3.2's falsifier is *"underperforms §3.1 at matched wall clock → the
sparsity bias is wrong"*. Measured at the real task's shape
(`B=128, L=13, D=128`), forward+backward of one mixer, 20 iterations after
warm-up, on this box (`sm_107`, **not** the scored H100 — relative costs
transfer, absolute ones do not):

**(a) Mixer only, forward+backward** (`lab/p2_verify.py` check 7):

| mixer | sequential impl | chunkwise / log-depth impl |
|---|---|---|
| §3.3 DeltaProduct `n_h=1` | 9.22 ms | **2.38 ms** |
| §3.3 DeltaProduct `n_h=2` | 13.82 ms | **2.34 ms** |
| §3.3 DeltaProduct `n_h=4` | 23.92 ms | **2.74 ms** |
| §3.2 PD-SSM `N=16` | 10.94 ms | **11.34 ms** |
| §3.1 dense matrix scan `N=16` | — | **2.87 ms** |

**(b) Whole model, forward+backward**, 2 layers, embeddings and MLPs included —
this is the number that decides steps-in-budget. All seven configurations timed
back-to-back in one process so they share identical GPU contention
(`lab/p2_verify.py` check 8):

| configuration | params | ms/step | steps/s | relative |
|---|---|---|---|---|
| §3.3 DeltaProduct `n_h=1` | 532,112 | **13.40** | 74.6 | 1.00× |
| §3.3 DeltaProduct `n_h=2` | 665,248 | **14.00** | 71.4 | 0.96× |
| §3.3 DeltaProduct `n_h=3` | 798,384 | **17.10** | 58.5 | 0.78× |
| §3.3 DeltaProduct `n_h=4` | 931,520 | **18.26** | 54.8 | 0.73× |
| §3.1 dense matrix scan `N=16` | 595,584 | **16.30** | 61.4 | 0.82× |
| §3.2 PD-SSM `N=16` | 993,408 | **54.94** | 18.2 | **0.24×** |
| §3.2 PD-SSM `N=8` | 498,560 | **47.74** | 20.9 | **0.28×** |

**Two results.**

1. **The chunkwise (WY / UT-transform) delta rule makes `n_h` almost free.**
   `2.38 / 2.34 / 2.74 ms` at `n_h = 1 / 2 / 4` — flat, because the whole
   sequence is one chunk and the cost is a `S×S` triangular solve with
   `S = L·n_h ≤ 52`. Against the sequential reference (`9.22 / 13.82 / 23.92 ms`)
   this is **4–9× faster and removes the `n_h` cost slope entirely**. The
   expressivity axis PLAN2 wanted swept is, at this sequence length, a free
   parameter.
2. **PD-SSM's cost advantage does not materialise, and its falsifier fires on
   wall clock alone.** At whole-model level PD-SSM runs at **0.24×** the step
   rate of DeltaProduct `n_h=1` and **0.30×** that of the §3.1 dense matrix
   scan, at matched `D`, matched heads, matched state size. Its log-depth scan
   is not even faster than its own sequential loop (`11.34` vs `10.94 ms`), and
   halving `N` from 16 to 8 recovers only 13 % — so the cost is *not* the
   `O(N³)` scan. The reason is structural, not an implementation defect:
   parameterising a column-one-hot `N×N` matrix per token per head costs an
   `N²`-wide projection plus an `N²` softmax per token
   (`d_model → n_dir·H·N² = 2048` here), and that dominates everything the
   sparse transition saves. The `O(N)` closure verified in §1 *is* real, but it
   cannot be used in the backward pass — the straight-through estimator needs
   the dense soft matrix in the autograd graph, which is precisely why the
   literature's asymptotic argument does not transfer to a 13-token sequence.

   **PLAN2 §3.2's falsifier — "underperforms §3.1 at matched wall clock" — is
   therefore met on cost before accuracy is even consulted.** At equal wall
   clock PD-SSM affords 0.30× the optimizer steps of the §3.1 baseline. Given
   RESUME's step-famine result (P3: 14–18 steps at a tier-faithful Easy
   budget), a 3.4× step-rate penalty is disqualifying on its own.

3. **`n_h` is nearly free.** 1.00× → 0.73× step rate for 4× the Householders,
   because the chunkwise form makes the per-token cost an `S×S` triangular
   solve with `S = L·n_h ≤ 52` rather than `n_h` sequential rank-1 updates.
   Whatever else is true, *expressivity along the Householder axis is not what
   a wall-clock budget is being spent on*.

**No tier-faithful `--mode wallclock` run was made, deliberately.** The GPU ran
at ~19 concurrent processes for this branch's entire window, so a wallclock
manifest would have measured contention, not the model — exactly the corruption
BRIEF §5 warns about. All comparisons above are either fixed-step or
back-to-back-in-one-process. Absolute step rates do not transfer from this box
(`sm_107`) to the scored H100 in any case; the *ratios* are what this branch
claims, and RESUME's own H100 calibration (~38.6 ms/step at batch 512 for a
D=128 recurrent stack, ~93,000 steps in a Hard run) plus
`plan2/sequential-rnn`'s "budget is surplus" measurement together say the
binding constraint is not compute. Consistent with that: nothing in §5 is
step-limited — `train_exact` is still climbing at 1,500 steps while held-out sits
at the floor, which is a generalisation failure, not a budget failure.

---

## 5. LEGAL — the real task

`e5` (512-example rungs, per BRIEF2 §6.4 — not `e1`), `--mode fixed_step`,
`max_steps = 1500`, seeds 74 / 7 / 21, batch 128, bf16 + amp, all prefix
products in fp32. Three siblings share the GPU, hence fixed-step throughout.

```
$VENV lab/p2_grid.py --grid lr0    --seeds 74 7 21 --steps 1500
$VENV lab/p2_grid.py --grid nh     --seeds 74 7    --steps 1500
$VENV lab/p2_grid.py --grid tau    --seeds 74      --steps 1500 --taus 0.1 1.0 3.0 10.0
$VENV lab/p2_grid.py --grid base repeat --seeds 74 7 --steps 1500
$VENV lab/p2_grid.py --grid size   --seeds 74      --steps 1500   # continuous state
$VENV lab/p2_grid.py --grid small  --seeds 74      --steps 1500   # discrete state
```

**Budget note, stated so the cell counts are not mistaken for a design choice.**
Three siblings share this GPU and it ran at ~19 concurrent processes for the
duration; a DeltaProduct cell that takes 20 s of isolated GPU time took ~300 s,
and a PD-SSM cell ~400 s. `train_exact` on e5 does not begin to lift until
~800 steps, so a shorter budget would have made every cell vacuously zero and
1,500 steps is the minimum informative setting. The sweeps were therefore
prioritised: controls → `n_h` → §3.1 reference → temperature → the discreteness
pair. **The `eig=pos` real-task control was descoped**, because §2 and Probe A/B
already settle the eigenvalue-range question with post-training numerical
spectra, and on a task where every cell reads `MAX_T = 0` the real-task version
adds little. That is a resource decision, and it is the only planned cell not
run.

### 5.1 The `--lr 0` control first (BRIEF2 §6.1)

Run before anything else is interpreted, because RESUME's decisive result is
that on this task's legal objective, *training moves away from the discrete
solution*, so a leftward metric shift is usually regression toward init.

| cell | `MAX_T` | mean_acc | out diversity | top-token share |
|---|---|---|---|---|
| §3.3 DeltaProduct `n_h=2`, `lr=0` | **0** | **0.0042** | 0.941 | 0.191 |
| §3.2 PD-SSM `τ=1`, `lr=0` | **0** | **0.0013** | 1.000 | 0.191 |
| §3.1 matrix scan, `lr=0` | LR0_MATSCAN | | | |

**`mean_exact_accuracy` at random init is 0.001–0.004.** Every trained cell in
this branch is inside or barely outside that band, so no `mean_acc` figure below
should be read as progress. It is reported only to demonstrate that it is not.

### 5.2 The grid

RESULTS_GRID

### 5.3 The straight-through failure mode — instrumented, and it did *not* fire

PLAN2 §5 and the mandate both flag straight-through as the most likely source of
"correct architecture that won't train", and RESUME records it as actively
destructive elsewhere (`local_ce` 13.83, the worst value in that report) and as
producing a discrete attractor that reached a fixed point within 4 steps. So the
`P` softmax was instrumented every 100 steps. Trained PD-SSM at `τ = 0.1`:

| step | `p_max_prob` | `p_perm_frac` | `|D|` mean | out diversity | top-token share |
|---|---|---|---|---|---|
| 100 | 0.851 | 0.699 | 0.872 | 0.647 | 0.446 |
| 300 | 0.947 | 0.701 | 0.869 | 0.588 | 0.311 |
| 600 | 0.968 | 0.693 | 0.865 | 0.588 | 0.287 |
| 1000 | **0.979** | 0.695 | 0.857 | 0.588 | 0.278 |
| (`lr = 0` control) | 0.071 | 0.632 | 0.881 | 1.000 | 0.191 |

Three things fall out.

1. **The discrete attractor is real and it is benign here.** `p_max_prob` climbs
   monotonically 0.071 → 0.979: the softmax converges onto its own argmax, so
   the straight-through gap closes and the surrogate gradient becomes an
   increasingly accurate one. That is the attractor RESUME describes — and in
   this architecture it *helps* the estimator rather than destroying training.
   **PD-SSM's failure here is not an estimator failure**, which is the specific
   thing the mandate asked to rule in or out.
2. **`p_perm_frac` never moves** (0.63 → 0.70 across the whole run). The learned
   `P` sharpens into a *function* on states but not into a *permutation*; ~30 %
   of state-columns collapse onto shared targets. Information is being
   discarded every token, which for an FSA emulation claim is the wrong shape.
3. **Training reduces output diversity** (1.000 at `lr=0` → 0.588 trained) and
   raises the top-token share (0.191 → 0.28). This is partial collapse — not the
   constant map a collapse detector is built to catch (that would read
   `1/17 = 0.059`), but movement in that direction, which is why the row is
   reported next to the accuracy rather than instead of it.

### 5.4 Does a *discrete* state change the train/held-out relationship?

`plan2/sequential-rnn` measured that on a continuous recurrent state, a
hidden-size sweep moves `train_exact` **134×** (0.007 → 0.98) while held-out
never leaves the floor — capacity buys *fitting*, not *generalising*. Combined
with `digit-carry`'s "the state alphabet must be small", the sharpened claim is
that the state must be small **and discrete**. PD-SSM's column-one-hot `P` is
exactly a discrete state, so this branch can test the claim directly, with two
controls that differ in *only* the discretisation:

* **`ste=hard` vs `ste=none`** — identical parameters, identical state size,
  identical everything; the forward pass is a hard one-hot in one and a softmax
  mixture in the other.
* **a continuous state-size sweep** in the DeltaProduct family
  (`head_dim ∈ {8, 16, 32, 64}`), which is the direct analogue of the sibling's
  hidden-size sweep, against **a discrete state-size sweep** in PD-SSM
  (`N ∈ {8, 11, 16}`; `N=11` is the carry alphabet for base-10 add-with-carry).

RESULTS_DISCRETE

RESULTS_NARRATIVE

---

## 6. What was falsified

**1. PLAN2 §3.2's own falsifier — "PD-SSM underperforms the §3.1 matrix scan at
matched wall clock" — fires, and it fires on cost before accuracy.** At
whole-model level PD-SSM affords **0.30×** the optimizer steps of the §3.1
reference and **0.24×** those of DeltaProduct. The comparison in §5 is run at
*matched steps*, which is generous to PD-SSM by 3.4×, and it still does not win.
The mechanism is identified, not guessed: the `O(N)` composition closure is
exact (§1) but unusable in the backward pass, because a straight-through
estimator needs the dense soft `N×N` matrix in the autograd graph; and the
`N²`-wide projection plus `N²` softmax that *parameterises* the one-hot matrix
costs more than the sparse transition saves. Halving `N` recovers only 13 %,
which rules out the scan itself as the cost. **The sparsity bias is wrong for
this task at this sequence length.**

**2. PLAN2 §3.3's own falsifier — "accuracy flat in `n_h`" — fires.** See §5.
The force of this comes entirely from §3: the same code, the same sweep and the
same budget produce a chance→exact transition on **A₅**, a non-solvable
`NC¹`-complete word problem, between `n_h = 2` and `n_h = 3`. So "flat in `n_h`"
here is not "the sweep was too coarse" or "nothing trains"; it is a null from a
calibrated instrument. **Non-commutativity is not what is missing.**

**3. The straight-through hypothesis is falsified as an explanation of PD-SSM's
failure — and its sign is inverted.** This was the mandate's named prior
suspicion. Three separate measurements say it does not hold:

* On the real task (§5.3) `p_max_prob` rises monotonically 0.071 → 0.979, so
  the surrogate gradient becomes *more* accurate over training, not less. The
  discrete attractor RESUME describes is real and here it is **benign**.
* On A₅ the hard straight-through configuration **beats** the pure-soft control
  decisively where it counts: 1.000 vs 0.494 in-distribution, and **1.000 vs
  0.037** at `2×` length. Discretisation is what makes it *generalise*.
* The temperature does matter enormously (chance → exact between `τ=1` and
  `τ=3`), but in the direction of a **softer backward pass**, not a harder one —
  so "anneal the relaxation toward hard", the standard recipe, is the wrong
  move for this architecture.

Whatever stops PD-SSM on the competition task, it is not the estimator.

**4. BRIEF2 §2c's carry-monoid route, in its token-axis form, is falsified.**
The hypothesis this branch bet on was that carry propagation across digit
positions is an associative prefix computation living inside the prompt-token
axis, and that a log-depth scan over that axis would therefore find it. An
operator that provably suffices for *any* regular prefix computation (verified
on `Z₂`, `Z₃`, `A₅`) does not find it. Either the carry monoid is not what the
loss can identify, or the token axis is the wrong place to look for it — and
RESUME's closure argument (`Tmul`'s cells are unidentifiable from the
end-of-chain label) says the former.

**5. Not falsified, and worth stating plainly: the `n_h` cost model in PLAN2.**
§3.3 lists DeltaProduct's cost as `O(n_h·d)`, implying a real expressivity /
compute trade. With the chunkwise form at this sequence length there is
effectively no trade: 4× the Householders costs 27 % of step rate. PLAN2's
framing of `n_h` as a *trade-off* dial is wrong in the cheap direction.

---

## 9. The single highest-value recommendation

> **Retire PLAN2 §0's expressivity reframe. The transition-operator axis is
> verified-live and measured-inert, and the remaining §3 candidates are further
> points on the same axis.**

The reasoning is not "we tried and it didn't work". It is that this branch built
an instrument with *calibrated sensitivity* and then got a null from it:

* The instrument is sensitive. The identical layer code, the identical sweep,
  the identical 1,500-step budget, moves from **chance to exact (1.000, and
  1.000 at 2× length)** on **A₅** — a non-solvable, `NC¹`-complete group word
  problem — when `n_h` goes from 2 to 3. It moves from chance to exact on
  `Z₃` when `n_h` goes from 1 to 2, and from chance to exact on `Z₂` when the
  eigenvalue range goes from `[0,1]` to `[−1,1]`. Those are three separate
  calibration points, each landing exactly where the theory in PLAN2 §1 says
  it should.
* The same instrument reads flat on the scored task, on every axis it resolves.

PLAN2 §1 orders Axis A by expressivity and §3 walks up it. This branch has now
measured its **top two entries** — PD-SSM ("any N-state FSA, 1 layer, dim N",
the strongest guarantee in the table) and DeltaProduct ("tunable, approaches
dense as `n_h → d`") — against the §3.1 dense scan, at matched wall clock, with
the eigenvalue trap avoided and instrumented rather than stepped in. Nothing on
the axis moved the metric. PLAN2's own kill criterion §6.3 is the relevant one
and this is the measurement it asked for.

**A falsifiable prediction this makes, which a sibling is in a position to check
today:** PLAN2 §2 item 1 and §3.6 propose a non-linear LSTM as the
maximum-expressivity control. DeltaProduct at `n_h=3` already demonstrates
exact `NC¹`-complete state tracking *in this codebase, at this budget*, and gets
nothing here. **Predict: `plan2/phase0`'s LSTM probe also gets nothing, and
`plan2/sequential-rnn` will not beat the §3.1 scan.** If either does, this
recommendation is wrong and should be discarded — but the prediction is cheap to
check and it is the fastest way to close or reopen the whole reframe.

**What to do with the freed budget.** RESUME already names the alternative and
measured it: the legal objective's gradient points *away* from the discrete
solution from the first step (`local_ce` 2.08–2.24 at random init → 3.5–4.2
after training), and a constraint's conditioning is set by how many learned ops
separate it from the parameters. That is a statement about the **objective**,
and no amount of operator expressivity addresses it. This branch adds one
independent confirmation from the opposite direction: with the operator made
provably sufficient and nearly free, the gap is unchanged.

**Two things from this branch are worth carrying forward regardless.**

1. **Use the chunkwise (WY) DeltaProduct form, not a sequential delta loop.**
   It is exact (rel. err 4e-7 against the serial reference) and removes the
   `n_h` cost slope: 1.00× → 0.73× whole-model step rate for **4×** the
   Householders. If any future architecture wants non-commutative state
   tracking, it is available at essentially no wall-clock cost. It just is not
   sufficient.
2. **Add a length/depth-extrapolation row to the screening table in
   `lab/RESUME.md`.** §3 contains a measured fooling configuration that the
   current table does not cover: DeltaProduct with the *wrong* eigenvalue range
   reads **0.941** in-distribution on mod-3 — "nearly solved" — while its
   2×-length accuracy is **0.328**, i.e. chance. A model that has learned
   nothing about the recurrence can look nearly correct on any
   fixed-length metric. The extrapolation eval costs ~1 s and needs no
   evaluator.

---

## 7. Compliance

* Nothing under `data/generated/` was read, printed, sampled, or summarised.
  The one command that would have listed it was blocked and not retried.
* No arithmetic, solver, or lookup in any forward pass. Every tensor is learned
  from random init in-run.
* End-to-end differentiable; no custom training loop; no participant-controlled
  backward; no manifest override in the submission.
* Nothing was submitted to the hosted service.
* `P2_DIAG_FILE` is a **lab-only** diagnostic sink, unset by default, and writes
  only statistics of *model outputs* (predicted-token diversity, entropy,
  realised eigenvalues) — never any statistic of the data. With it unset the
  submission file is a fixed, deterministic, side-effect-free submission.
* The synthetic probe (§3) is **DIAGNOSTIC** and generates its own data from
  group tables built in-process. It is not a submission and never touches the
  competition data.

## 8. Artifacts

| path | what |
|---|---|
| `submissions/plan2-pd-ssm-delta/submission.py` | both architectures + the §3.1 reference, self-contained |
| `lab/p2_layers.py` | loads the mixers out of the submission (single source of truth) |
| `lab/p2_verify.py` | the verification suite of §1, §2, §4 |
| `lab/p2_probe.py` | the synthetic state-tracking probe of §3 |
| `lab/p2_grid.py` | the real-task sweep driver of §5 |
| `lab/p2_summarize.py` | the tables in this report |
| `lab/p2_probe_results.jsonl` | every probe cell |
| `lab/p2_grid_log.jsonl` | every real-task cell + its diagnostic stream |
| `lab/p2_diag/*.jsonl` | per-run collapse detector / eigenvalue traces |
| `lab/archive.jsonl` | every evaluator run (appended, never rewritten) |
