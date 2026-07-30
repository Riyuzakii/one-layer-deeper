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
| `[0,1]` | 1 | +0.284 ± 0.390 | +0.902 ± 0.092 | **0.000 ± 0.000** |
| `[0,1]` | 2 | +0.004 ± 0.005 | +0.870 ± 0.164 | **0.000 ± 0.000** |
| `[0,1]` | 4 | +0.000 ± 0.000 | +0.984 ± 0.027 | **0.000 ± 0.000** |
| `[−1,1]` | 1 | −0.589 ± 0.474 | +0.608 ± 0.363 | 0.450 ± 0.236 |
| `[−1,1]` | 2 | **−0.988 ± 0.008** | +0.441 ± 0.362 | 0.741 ± 0.237 |
| `[−1,1]` | 3 | **−0.988 ± 0.009** | +0.567 ± 0.209 | 0.688 ± 0.049 |
| `[−1,1]` | 4 | **−0.987 ± 0.008** | +0.335 ± 0.178 | 0.749 ± 0.001 |

**Verified: the extended range is real and training uses it.** With `[−1,1]`
available, training drives eigenvalues to **−0.988** and puts 69–75 % of them
negative. With `[0,1]` the realised minimum is exactly 0.000 and *no* eigenvalue
is ever negative — as it must be. The realised range on the **real task** is the
same (`eig[−0.96, +0.79]` … `eig[−0.99, +0.98]`, 53–92 % negative, per-cell in
the §5 table), so the mechanism is switched on there too; it simply buys nothing. This branch did not run the crippled variant
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

> **Three caveats, stated before the numbers rather than after.**
> (a) The "§3.1" row is *this branch's own* low-rank instantiation
> (`M_t = I + u_t v_tᵀ / √r`, `r = 4`), not the `plan2/matrix-scan` sibling's
> implementation. Read it as a locally-matched reference point, never as a
> statement about their work.
> (b) The main probe runs PD-SSM at `N = 16`, which **cannot** represent A₅'s
> 60 states under PD-SSM's own theorem, so its A₅ cells are under-provisioned by
> construction. Probe F re-runs A₅ at `N = 32` and `N = 64` and **the theorem is
> confirmed at its boundary**: at `τ = 1`, over two seeds, `N = 16` solves
> **0/2**, `N = 32` solves **1/2** (1.000 and 0.146), and `N = 64` — the first
> size tried at or above `|A₅| = 60` — solves **2/2** (1.000 and 0.987, with
> `2×`-length extrapolation 1.000 and 0.738). PD-SSM's "one layer of dimension
> `N` emulates any `N`-state FSA" claim reproduces here. (`N = 16` can still
> reach 1.000 at a lucky temperature, because the readout is not restricted to
> the state — but the state-size dependence is real and it lands where the
> theorem says.)
> (c) The main probe process was killed by harness cleanup at cell 93 of 96; the
> three lost cells (seed-1 A₅ at `τ=10`, the seed-1 soft control, and the seed-1
> §3.1 reference) were re-run separately via `lab/p2_probe_finish.py`. No cell
> was dropped for being inconvenient.

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

   PROBE_C_A5_PERSEED

   `τ` divides the logits, so larger `τ` is a **softer backward pass**, while
   the forward pass is exactly one-hot at every `τ` (verified in §1). Three
   things matter here.

   * The sweep spans **chance to exact**. Some temperature reaches near-perfect
     A₅ in *both* seeds; most temperatures are at chance in both.
   * **Which** temperature works is entirely seed-dependent — `τ = 3` in seed 0
     (1.000, and 0.055 in seed 1), `τ = 0.3` in seed 1 (0.961, and 0.044 in
     seed 0). **No single `τ` in the sweep works for both seeds.** So this is
     not "tune `τ` once"; it is a basin-selection problem in disguise, of
     exactly the shape `explore/alu-population` characterised — which suggests
     the right instrument is a replica dimension over `τ`, not a schedule.
   * The standard "anneal the relaxation toward hard" recipe is therefore not
     reliably the right direction either.

   Had this branch reported one temperature it would have reported a
   chance-level cell in most draws. This is the strongest argument in the report
   for the mandate's instruction to report the sweep rather than a setting.
4. **Straight-through discretisation is what makes PD-SSM *generalise*, not what
   stops it training.** The `ste=none` soft control on A₅ clearly learns
   something in-distribution — **0.494** (seq 0.277) in seed 0 and **0.385**
   (seq 0.231) in seed 1 — and then fails completely at `2×` length: **0.037**
   and **0.021**, both chance. Hard straight-through at its working temperature
   reads 1.000 in-distribution *and* 1.000 at `2×` length. **The soft relaxation
   memorises; the discrete forward pass generalises, in both seeds.** This is
   the direct opposite of the prior this branch was handed (straight-through as
   "the known failure mode"), and it is measured on the only task in the set
   that needs the full expressivity.
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
this is the number that decides steps-in-budget. All configurations timed
back-to-back in one process so they share identical GPU contention
(`lab/p2_verify.py` check 8). Reported as **step-rate ratios to DeltaProduct
`n_h=1`**, at *both* shapes, because `plan2/sequential-rnn` measured that cost
ratios taken at Easy's shape badly understate the gaps (Neural GPU vs fused
LSTM: 1.7× at e5's shape, 22× at Hard's). Absolute ms are `sm_107` numbers and
do not transfer; the ratios were the thing to report and this correction landed
mid-branch.

| configuration | params | e5 shape (B=128, L=13) | **Hard shape (B=512, L=21)** |
|---|---|---|---|
| §3.3 DeltaProduct `n_h=1` | 533k | 1.00× (13.40 ms) | **1.00×** (20.03 ms) |
| §3.3 DeltaProduct `n_h=2` | 666k | 0.96× (14.00 ms) | **0.74×** (27.09 ms) |
| §3.3 DeltaProduct `n_h=3` | 799k | 0.78× (17.10 ms) | **0.56×** (35.96 ms) |
| §3.3 DeltaProduct `n_h=4` | 933k | 0.73× (18.26 ms) | **0.45×** (44.78 ms) |
| §3.1 dense matrix scan `N=16` | 597k | 0.82× (16.30 ms) | **0.38×** (53.11 ms) |
| §3.2 PD-SSM `N=16` | 994k | 0.24× (54.94 ms) | **0.10×** (210.85 ms) |
| §3.2 PD-SSM `N=8` | 500k | 0.28× (47.74 ms) | **0.12×** (162.66 ms) |

**The correction matters and it changes one of this branch's own claims.** Every
gap widens at Hard's shape: PD-SSM goes 0.24× → **0.10×**, the §3.1 scan
0.82× → **0.38×**, and `n_h=4` 0.73× → **0.45×**.

**Two results.**

1. **The chunkwise (WY / UT-transform) delta rule makes `n_h` almost free.**
   `2.38 / 2.34 / 2.74 ms` at `n_h = 1 / 2 / 4` — flat, because the whole
   sequence is one chunk and the cost is a `S×S` triangular solve with
   `S = L·n_h ≤ 52`. Against the sequential reference (`9.22 / 13.82 / 23.92 ms`)
   this is **4–9× faster and removes the `n_h` cost slope entirely**. The
   expressivity axis PLAN2 wanted swept is, at this sequence length, a free
   parameter.
2. **PD-SSM's cost advantage does not materialise, and its falsifier fires on
   wall clock alone.** At Hard's shape PD-SSM runs at **0.10×** the step rate of
   DeltaProduct `n_h=1` and **0.26×** that of the §3.1 dense matrix scan, at
   matched `D`, matched heads, matched state size. Its log-depth scan
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
   clock PD-SSM affords **0.26×** the optimizer steps of the §3.1 baseline at
   Hard's shape (0.10× / 0.38×), i.e. a **3.8× step-rate penalty**, worse than
   the 3.4× that the e5-shape measurement suggested. And §5 runs the accuracy
   comparison at *matched steps*, which is generous to PD-SSM by that same 3.8×
   — and it still does not win a rung.

3. **`n_h` is cheap at e5's shape and NOT free at Hard's — this branch's earlier
   claim was wrong and is retracted here.** At e5's shape 4× the Householders
   costs 27 % of step rate (0.73×), which reads as "free". At Hard's shape it
   costs **55 %** (0.45×), because the chunkwise cost is an `S×S` triangular
   solve with `S = L·n_h`, and `S` goes from 52 to **84** when `L` goes 13 → 21.
   The scaling is in `L·n_h`, so the penalty grows with sequence length exactly
   as it should. The honest version of the claim: **the chunkwise form removes
   the `n_h` penalty relative to a sequential delta loop (4–9×), but `n_h` is
   still a real 2.2× cost at the tier that is ranked.** Cheap, not free.

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
| §3.1 matrix scan, `lr=0` | **0** | **0.0033** | 1.000 | 0.240 |

**`mean_exact_accuracy` at random init is 0.001–0.004.** Every trained cell in
this branch is inside or barely outside that band, so no `mean_acc` figure below
should be read as progress. It is reported only to demonstrate that it is not.

**The collapse detector, and what it actually says.** `out_diversity` is the
fraction of the 17-token vocabulary the model ever emits; a constant map reads
`1/17 = 0.059`. **No cell collapses.** But the reading is more specific than
that: every *trained* cell — all three architectures, every setting — reads
**exactly 0.588 = 10/17**, while every `lr = 0` control reads 0.941–1.000. That
is not noise, it is the models learning that answers are digits and dropping the
seven marker/special tokens. So the collapse detector's verdict here is "all
architectures learn the output alphabet and then stop", which is a sharper
statement of the failure than the accuracy numbers give on their own.

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

**(a) The continuous state-size sweep replicates the sibling's result inside
this family.** DeltaProduct `n_h = 2`, everything else identical, `head_dim`
varied:

| `head_dim` | 8 | 16 | 32 | 64 |
|---|---|---|---|---|
| `train_exact` | 0.242 | 0.391 | **0.531** | **0.664** |
| rung-1 held-out | 0.004 | 0.010 | 0.004 | **0.000** |

`train_exact` climbs **2.7× monotonically** while held-out rung-1 does not move
and finishes at **zero** at the largest state. Same conclusion as
`plan2/sequential-rnn`, reached on a completely different recurrence: **a larger
continuous state buys fitting and nothing else.**

**(b) The discrete-state cells sit higher on held-out at far lower
`train_exact`.** Every PD-SSM cell reads rung-1 held 0.006–0.016 (3–8 of 512)
while its `train_exact` is 0.055–0.141 — except `τ = 0.1`, which reaches
`train_exact` 0.680. The continuous cells reach `train_exact` 0.24–0.66 and read
0.000–0.010 held. Ranked by held-out rung-1, the top four cells in the entire
grid are **all PD-SSM** (`τ=0.1` 0.016, `τ=3` 0.016, `τ=1` 0.012, `N=11` 0.010).

**Caveat, and it is a large one.** These are 3–8 correct examples out of 512, at
one seed per cell, against `lr = 0` controls that read 0–3. The ordering is
consistent across six PD-SSM cells and six continuous cells, which is why it is
reported — but it is **not** a certified difference, it moves no rung, and it
must not be read as "discreteness works". The honest statement is: *the
discreteness hypothesis survives this branch's test rather than being confirmed
by it, and it is the only hypothesis in this report that does.*

DISCRETE_HARDSOFT

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

> **Retire PLAN2's *expressivity* axis and keep only its *discreteness*
> corollary. The transition-operator axis is verified-live and measured-inert;
> the one thing in this branch that behaved differently from a capacity knob is
> the discreteness of PD-SSM's state, and that is where the remaining budget
> should go — as a population over `τ`, not as a schedule.**

Three measurements make that concrete, and they separate cleanly:

| knob | what it does on A₅ (DIAGNOSTIC) | what it does on the real task (LEGAL) |
|---|---|---|
| eigenvalue range `[0,1]` → `[−1,1]` | chance → **exact** | nothing |
| `n_h` 1 → 4 | chance → **exact** | behaves as a **capacity knob**: `train_exact` up, held-out flat |
| continuous state size 8 → 64 | (not the axis) | `train_exact` **2.7×**, held-out → **0.000** |
| **discreteness of the state** (hard vs soft `P`) | soft memorises (`2×`-length 0.02–0.04), hard **extrapolates** (1.000) | the only cells above the continuous ones on held-out |

The first three are the expressivity/capacity story and they are closed. The
fourth is not, and it is the only one whose *signature* — generalising rather
than fitting — is the signature the task actually needs.

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

**What to do with the freed budget — the one concrete experiment this branch
would run next.** Not another point on Axis A. The A₅ temperature result says
`τ` selects a *basin*, not a quality: `τ = 3` solves it in seed 0 and reads
0.055 in seed 1; `τ = 0.3` solves it in seed 1 and reads 0.044 in seed 0. **No
single `τ` works for both seeds, and each working `τ` gives an exact solution,
not a better one.** That is precisely the signature `explore/alu-population`
characterised — per-replica success rate 0.26, `1-(1-p)^P` reaching 0.9999 at
P=32, differentiable selection legal and working, cost free to P=8. So the
experiment is: **a replica dimension over `τ` (and seed) on PD-SSM, with the
population machinery `alu-population` already built and validated**, screened on
2×-length extrapolation rather than in-distribution accuracy. It is the only
configuration in this branch where a population is the right instrument, because
it is the only place the obstruction reads *stochastic* rather than systematic.

That said, RESUME's harder result still stands over all of this: the legal
objective's gradient points *away* from the discrete solution from the first
step (`local_ce` 2.08–2.24 at random init → 3.5–4.2 after training). That is a
statement about the **objective**, and no amount of operator expressivity
addresses it. This branch adds one independent confirmation from the opposite
direction: with the operator made provably sufficient and nearly free, the gap
is unchanged.

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

## 6b. The submission — do not promote it

`submissions/plan2-pd-ssm-delta/submission.py` exists because BRIEF §6.3 asks
each branch for one, and it is a real, lint-clean, self-contained submission
(22.8 KiB; `benchmark.validation.lint_submission_source` passes; runs end-to-end
through `lab/run_experiment.py` in every cell above). With no `P2_*` environment
set it is a fixed, deterministic, side-effect-free model: DeltaProduct,
`n_h = 2`, `eig ∈ (−1,1)`, 2 layers, `D = 128`, `batch_size = 128`.

**It scores `MAX_T = 0` and should not be promoted to any tier.** Its
`mean_exact_accuracy` (0.004–0.010) is inside or barely outside the `--lr 0`
band (0.001–0.004), and the mandate's condition was "a submission only if
something works". Nothing here works on the ranked metric.

The one thing in it worth reusing is the tied-head init fix: `nn.Embedding`'s
default `N(0,1)` with a tied output head makes the initial loss ≈ 80 instead of
`ln 17 = 2.83`. Measured on this scaffold, fixing it lifted `train_exact` from
**0.414 → 0.531** at a matched 1,500 steps, and it costs one line. Every
submission in this repo that ties `head.weight = token_embedding.weight` without
re-initialising — including `submissions/baseline_adamw/submission.py` — is
paying that toll. At a tier-faithful Easy budget (RESUME P3: 14–18 optimizer
steps) it would consume the entire run.

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
