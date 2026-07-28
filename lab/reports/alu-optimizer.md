# alu-optimizer — is the ALU's closure a *conditioning* gap?

**Branch** `explore/alu-optimizer`, from `explore/alu-population`.
**Mandate** The project's optimizer sweep (AdamW / Muon / Schedule-Free /
Grokfast / ⊥Grad / StableMax) was run entirely on the *dense transformer*. On
`DigitALU` only AdamW hyperparameters were ever varied — **nothing second-order
and nothing with a slow-EMA component**. The closure rests on the surface being
*rugged* rather than *ill-conditioned*, and conditioning is the one property
never varied. Implement SOAP and AdEMAMix, verify them on a problem they should
solve, then screen them on the legal objective.
**Tooling** `lab/optim_extra.py` (both optimizers), `lab/test_optim_extra.py`
(self-checks), `--opt {adamw,soap,ademamix}` in `lab/probe_pop.py` and
`lab/probe_pop_legal.py`, `lab/make_optsub.py` (dense-transformer submissions).
**Runs** 29 in `lab/opt_runs.jsonl` (gate), 33 in `lab/opt_legal_runs.jsonl`
(legal), 5 evaluator cells in `lab/archive.jsonl` (tag `B-opt2`), logs in
`lab/logs/`.

---

## HEADLINE

| | | status |
|---|---|---|
| **Both optimizers PASS the diagnostic gate** | SOAP `train_exact_hard` **1.000**/held 1.000; AdEMAMix **1.000**/1.000 at **11/32** replicas — *identical* to AdamW | DIAGNOSTIC |
| **Neither moves the legal objective** | 33 legal runs, **0 of 1,056 replicas** below `local_ce` 1.0, against a cliff at 0.006 | **LEGAL, and null** |
| **The apparent SOAP left-shift is a learning-rate effect** | AdamW at the *same* lr gets 88% of it; the ordering is set by lr, not by the optimizer | **LEGAL** |
| **The control that reframes the whole closure: `local_ce` at RANDOM INIT is 2.08–2.24** | so the project's best-ever "legal" value (2.304, `--assoc`) is **worse than an untrained model**, and the baseline at 3.6 is **65% worse than init** | **LEGAL** |
| **SOAP breaks the `local_ce` cliff law** — a new metric-fooling row | 14 replicas below the cliff, **1** in the basin; under AdamW the law is a step function | DIAGNOSTIC |
| **AdEMAMix's slow EMA is monotonically harmful here** | basin count 11 → 6 → 4 → 0 → 0 as β₃ goes 0.9 → 0.99 → 0.999 → 0.9999 | DIAGNOSTIC |
| **Second-order costs almost nothing at this size** | SOAP is **+17%** ms/step at P=32; the ALU is 6,820 params and kernel-launch bound | — |
| **Dense transformer: SOAP and AdEMAMix are null there too** | 5 cells, all `MAX_T=0`; rung-1 spread 0–2 of 38 against a one-example variance floor | **LEGAL, and null** |

**Verdict, in one line: conditioning is NOT the gap. The closure now also covers
second-order and slow-EMA methods.** And the branch produces a correction that
is worth more than its null: **the legal objective moves `local_ce` *away* from
the cliff relative to random initialisation**, so every "leftward shift" recorded
in this project — including `alu-population`'s `--assoc` and this branch's own
SOAP rows — is partial regression toward an untrained model, not progress.

**No submission.** Every cell run through the evaluator scored `MAX_T = 0`, at
or below the field, so there is nothing to ship; §7.

---

## 0. THE FRAME — labels, and what a null is worth

Every row is **LEGAL** or **DIAGNOSTIC**, per `alu-credit` §0C.

| component | status | why |
|---|---|---|
| a custom `torch.optim.Optimizer` | **LEGAL** | BRIEF §4 rule 4 bans a custom training *loop*, a participant-controlled backward and a manifest override. The evaluator calls `build_optimizer` and keeps the loop, the backward, the grad clip and one `step()` per batch. Neither optimizer here sees data, labels or the graph — `step()` sees `p` and `p.grad`. |
| the replica axis (P=32) | **LEGAL** | `alu-population` §0; the contract constrains the loop, not the model's width |
| teacher-forced per-op CE (§2's gate) | **DIAGNOSTIC** | the tape is recorded from a `--construct`ed model (rules 2, 7) |
| `local_ce` as a *measurement* | — | computed against a constructed tape, never in the loss; no gradient flows from it |

A null from an optimizer that was never shown to work is worth nothing, which is
why §2 comes before §4.

---

## 1. Implementation, and the bug the self-check found

`lab/optim_extra.py`, written from the published algorithms; nothing installed.

* **SOAP** (Vyas et al. 2024). Per parameter, per axis, keep the Shampoo factor
  `GG_i = EMA(G_(i) G_(i)^T)` and its eigenvectors `Q_i`; rotate the gradient
  into that basis, run Adam there, rotate the update back. Axes are optionally
  merged (`max_precond_dim`), so `Tmul` (10,10,20) becomes a 100×100 ⊗ 20×20
  factorisation at 512 and a genuine **full 2000×2000 matrix** at 4096. Both
  were run.
* **AdEMAMix** (Pagliardini et al. 2024). `m1` (β₁), `m2` (β₃, **no** bias
  correction), `ν` (β₂); update `(m̂1 + α·m2)/(√ν̂ + ε)`. α and β₃ warm up, β₃
  linearly in *half-life* as in the paper.

### 1.1 `batch_dims` — load-bearing, and easy to get wrong

`probe_pop.py` gives every parameter a leading replica axis of size P, and the
whole population instrument rests on replicas being **independent draws**
(`alu-population` §4.2, per-replica rate 0.26). A preconditioner computed
*across* the replica axis would couple them and silently invalidate the
instrument. `SOAP(..., batch_dims=1)` keeps a separate preconditioner per
replica. Self-check 2 verifies this to machine zero: zeroing replica 3's loss
leaves replicas 0–2 bit-identical (max |Δ| = **0.00e+00**). AdEMAMix is
elementwise and independent by construction.

### 1.2 The self-check, and a real bug

`$V lab/test_optim_extra.py` → **ALL PASS**.

| check | result |
|---|---|
| finiteness + descent on all 7 PopALU tensor shapes | PASS |
| replica independence at `batch_dims=1` | PASS, max |Δ| = 0.00e+00 |
| **SOAP vs AdamW on an ill-conditioned quadratic (κ=10⁴), best of 5 lrs each** | **PASS — 7.30e+05 vs 1.95e+03, a 374× lower loss** |
| SOAP's basis is orthogonal (`QᵀQ = I`) | PASS, max err 4.8e-07 |
| SOAP's step is covariant under a rotation of the parameterisation | PASS, rel diff **1.7e-06** over 3 updates |
| AdEMAMix at α=0 is *exactly* `torch.optim.AdamW` | PASS, 7.1027e-02 vs 7.1027e-02 |
| `m2 == (1−β₃ᵏ)·g` (un-bias-corrected EMA), `m1 == (1−β₁ᵏ)·g` | PASS to 1e-6 |
| β₃ half-life warmup matches the paper's `f`/`f⁻¹`; α warmup linear | PASS |
| both train a small MLP at least as well as AdamW (best of 5 lrs, cosine decay) | PASS |

**The bug, and it mattered.** The SOAP *reference implementation* keeps the first
moment in the original space and rotates it on use, while the second moment
lives in the rotated space. That makes the numerator and the denominator two
**independently rounded** projections of the same tensor. In near-null
directions their float32 cancellation errors do not agree, `m/(√v+ε)` is no
longer Adam-bounded, and it explodes: on check 3 the first update had norm
7.1e+04 where the algebra bounds it at 0.035, and SOAP *diverged* at every lr.
Keeping **both** moments in the eigenbasis — which is what the paper's
Algorithm 1 says, and what "run Adam in the preconditioner's eigenbasis" means —
makes the ratio bounded exactly as Adam's is, and turned a 0.0× result into the
374× win above. The reference-code variant is kept as `--soap-mspace orig` and
was run on the ALU anyway (§2.2).

Two genuine (non-bug) limits are recorded in the test so a later reader does not
mistake them for errors: over ~30 updates near-degenerate Shampoo eigenvalues can
order differently in two frames and trajectories separate (3.3e-02); and with a
**rank-deficient** factor the null-space eigenvectors are arbitrary and Adam
normalises numerical noise there to full step size. `precond_warmup` exists for
the second and was run on the ALU (`O2_soap_3e2_warm`).

---

## 2. THE GATE — DIAGNOSTIC, and both optimizers pass

**Conditions.** Exactly Stage-1: m1 scale (N=10403, S=5, 8,000 train / 1,024
held-out), 39-step `tree:quotient`, **untied**, P=32, 1,200 steps, batch 512,
teacher forcing on the constructed tape. AdamW's published numbers here are
`train_exact_hard` **1.000** / held **1.000** with a per-replica basin rate of
**0.26** pooled (11/32 = 0.34 in this seed).

```bash
bash lab/pop_sweep.sh lab/jobs_opt_gate.txt 4      # 12 runs
bash lab/pop_sweep.sh lab/jobs_opt_gate2.txt 4     # 13 runs
```

### 2.1 The gate table — **DIAGNOSTIC**

| run | optimizer | config | `train_exact_hard` | `held_exact_hard` | in basin | ms/step |
|---|---|---|---:|---:|---:|---:|
| `O_adamw_3e2` | **AdamW** | lr 3e-2 (the published control) | **1.000** | **1.000** | **11/32** | 165 |
| `O2_ade_a0_b295` | AdEMAMix | lr 3e-2, **α=0**, β₂ 0.95 | **1.000** | **1.000** | **11/32** | 271 |
| `O2_ade_a8_b295_b39_lo` | **AdEMAMix** | lr 1e-2, α=8, β₂ 0.95, **β₃ 0.9** | **1.000** | **1.000** | **11/32** | 285 |
| `O2_ade_a2_b295_b399` | AdEMAMix | lr 3e-2, α=2, β₂ 0.95, β₃ 0.99 | 0.950 | 0.947 | 6/32 | 274 |
| `O2_ade_a8_b295_b399` | AdEMAMix | lr 3e-2, α=8, β₂ 0.95, β₃ 0.99 | 0.853 | 0.820 | 4/32 | 275 |
| `O2_ade_a2_b295_b3999` | AdEMAMix | lr 3e-2, α=2, β₂ 0.95, β₃ 0.999 | 0.236 | 0.264 | 0/32 | 287 |
| `O2_ade_a8_b295_b3999` | AdEMAMix | lr 3e-2, α=8, β₂ 0.95, β₃ 0.999 | 0.006 | 0.007 | 0/32 | 286 |
| `O_soap_3e2` | **SOAP** | lr 3e-2 | **1.000** | **1.000** | 2/32 | 193 |
| `O2_soap_3e2_f1` | SOAP | lr 3e-2, refresh every step | **1.000** | **1.000** | 2/32 | 619 |
| `O2_soap_3e2_full` | SOAP | lr 3e-2, **full 2000×2000 matrix** | 0.733 | 0.748 | 2/32 | 525 |
| `O2_soap_5e2` | SOAP | lr 5e-2 | 0.642 | 0.650 | 1/32 | 334 |
| `O2_soap_3e2_nomerge` | SOAP | lr 3e-2, per-axis Kronecker | 0.406 | 0.433 | 0/32 | 293 |
| `O2_soap_3e2_warm` | SOAP | lr 3e-2, `precond_warmup` 20 | 0.351 | 0.332 | 0/32 | 330 |
| `O2_soap_3e2_orig` | SOAP | lr 3e-2, reference-code moment space | 0.125 | 0.140 | 0/32 | 335 |
| `O_soap_1e1` | SOAP | lr 1e-1 | 0.120 | 0.132 | 0/32 | 186 |
| `O_soap_1e2` / `O_soap_3e3` | SOAP | lr 1e-2 / 3e-3 | 0.032 / 0.001 | 0.035 / 0.001 | 0/32 | 193 |

**The gate is passed, and stated plainly:**

* **AdEMAMix reaches `train_exact_hard` 1.000 / `held_exact_hard` 1.000 at
  11/32 replicas — the same number as AdamW, to the replica.** Its α=0 control
  also reproduces AdamW at 11/32. That is as strong an implementation check as
  is available: the optimizer is exercised inside the real objective, not a
  synthetic one.
* **SOAP reaches `train_exact_hard` 1.000 / `held_exact_hard` 1.000** at lr 3e-2
  (and again with per-step basis refresh), so it can find the discrete solution.
  But its rate is **2/32 = 0.06 against AdamW's 0.34** — a ~5× *worse* basin
  rate at the same budget. Preconditioning does not help find this basin; it
  makes it harder to find.

### 2.2 Two things the gate settled that are worth carrying

**(a) AdEMAMix's round-1 failure was a confound, not the method.** The first 8
AdEMAMix cells all read `train_exact_hard` ≤ 0.018 and I would have reported a
gate failure. The cause was **β₂**: the ALU baseline uses `betas=(0.9, 0.95)`
and I had used the AdEMAMix paper's β₂ = 0.999, which adapts the second moment
20× slower. Holding β₂ at the baseline's 0.95 recovers AdamW exactly. *When
porting an optimizer, match every shared hyperparameter to the incumbent before
concluding anything.*

**(b) The slow EMA is monotonically harmful on this objective.** With β₂ fixed
at 0.95, basin count against β₃:

| β₃ | 0.9 | 0.99 (α=2) | 0.99 (α=8) | 0.999 | 0.9999 |
|---|---:|---:|---:|---:|---:|
| replicas in basin | **11/32** | 6/32 | 4/32 | 0/32 | 0/32 |
| `train_exact_hard` | **1.000** | 0.950 | 0.853 | 0.006–0.236 | 0.018 |

β₃ = 0.9 is a 6.6-step half-life — i.e. AdEMAMix helps here exactly to the extent
that it stops being AdEMAMix. This is consistent with the method's own regime
statement, which self-check 5 also measured on an MLP: at T=600 (< the β₃=0.999
half-life of 693) AdEMAMix is far *worse* than AdamW (0.73 vs 0.004); at T=6000
it wins (1e-5 vs 2e-5). **The ALU's tier budget is ~1,300 steps at Hard**, so a
slow-EMA optimizer is structurally out of its own regime on this problem. That
is a *reason* for the null, not an excuse for it: the regime is set by the
competition's budget and cannot be changed.

### 2.3 **SOAP breaks the `local_ce` cliff law** — a new metric-fooling row

`alu-population` §4.3 established that `local_ce` is quantised with a razor
cliff: 0.0000 → `train_exact_hard` 1.000, 0.0050 → 0.953, 0.0089 → 0.21, and
that it is the only usable per-replica basin indicator. **Under SOAP that
mapping does not hold.**

| run | optimizer | replicas at `local_ce` < 0.006 | best `local_ce` | `train_exact_hard` | **soft** `train_exact` | `mul_fn` |
|---|---|---:|---:|---:|---:|---:|
| `O_adamw_3e2` | AdamW | 11 | 0.0000 | **1.000** | 1.000 | 1.00 |
| `O2_ade_a8_b295_b39_lo` | AdEMAMix | 10 | 0.0000 | **1.000** | 1.000 | 1.00 |
| `O2_soap_5e2` | SOAP | **14** | 6e-05 | **0.642** | 0.998 | 0.99 |
| `O2_soap_3e2_orig` | SOAP | **13** | 1.5e-04 | **0.125** | 0.992 | 0.97 |
| `O_soap_1e1` | SOAP | **10** | **5e-05** | **0.120** | 0.998 | 0.98 |

`O_soap_1e1` has four replicas at `local_ce` = 1e-4 — *lower than every AdamW
basin member except the exact zeros* — and reads `train_exact_hard` 0.12. Fourteen
SOAP replicas below the cliff yield **one** in the basin.

**Mechanism, measured rather than assumed** (`lab/jobs_opt_sharp.txt`, which
records state saturation and table logit scale):

| | AdamW | AdEMAMix β₃0.9 | SOAP lr 3e-2 | SOAP lr 1e-1 |
|---|---:|---:|---:|---:|
| mean `max_d p(d)` over all tapped states, best replica | 1.0000 | 1.0000 | 0.9995 | 0.9999 |
| `Tmul` logit RMS | 2.482 | 2.616 | 2.528 | **2.772** |
| `mul_fn` (product table correct up to gauge) | **1.00** | **1.00** | **1.00** | **0.98** |
| `train_exact_hard` | 1.000 | 1.000 | 1.000 | 0.117 |

So it is **not** state de-saturation — SOAP's states are as sharp as AdamW's and
its tables carry a *larger* logit scale. Across all 29 gate runs the split tracks
one variable: **every SOAP row that shows it has `mul_fn` 0.97–0.99, and every
SOAP row that reaches 1.000 has `mul_fn` = 1.00.** The residual error sits in the
digit-*product* table, whose outputs are not tapped directly (the tap is on the
adder/subtractor scans and the reduce output), so a 1–3% product-table error is
diluted in the tapped average while being fatal to free-running composition over
S² = 25 products.

**Consequence, and it is a screening rule, not a curiosity:** `local_ce` is
**not optimizer-portable**. It was calibrated on AdamW and it is a valid basin
indicator *for AdamW*. Do not rank across optimizers on it. This belongs in
`RESUME.md`'s metric-fooling table:

| metric | fooled by | reads | but |
|---|---|---|---|
| `local_ce` (below the 0.006 cliff) | **SOAP, lr 1e-1** | **5e-05** | `train_exact_hard` **0.120** |

Because of this, §4 reports `train_exact_hard`, held-out, the full `local_ce`
distribution **and** output diversity together, and does not rest on `local_ce`.

---

## 3. Cost — second-order is nearly free at 6,820 parameters

At P=32, batch 512, on a quiet GPU:

| optimizer | ms/step | vs AdamW |
|---|---:|---:|
| AdamW | 165 | 1.00× |
| AdEMAMix | 165–171 | 1.00–1.04× |
| SOAP (2-D merged, refresh every 10) | 193 | **1.17×** |
| SOAP (per-axis Kronecker) | 293 | 1.77× |
| SOAP (**full 2000×2000 matrix per tensor**) | 525 | 3.18× |
| SOAP (refresh every step) | 619 | 3.74× |

The mandate's premise is confirmed: **full-matrix preconditioning is affordable
here** — 3.2× wall clock for an exact per-tensor preconditioner on a
6,820-parameter model — in a way it would not be at scale. It buys nothing
(`train_exact_hard` 0.733 vs the merged variant's 1.000). The cheap
2-D-merged variant at +17% is the one to use.

---

## 4. THE LEGAL SCREEN — the actual question

**LEGAL.** `lab/probe_pop_legal.py` with every extra term off: plain
end-of-chain task loss, m1 scale, `tree:quotient`, untied, P=32, 1,200 steps.
This is exactly the configuration whose published baseline is `local_ce` min
2.975 / median 3.347 / max 4.644 with 0/32 below the 0.006 cliff.

```bash
bash lab/pop_legal_sweep.sh lab/jobs_opt_legal.txt 3    # 15 runs
bash lab/pop_legal_sweep.sh lab/jobs_opt_legal2.txt 4   # 8 runs, the lr control
bash lab/pop_legal_sweep.sh lab/jobs_opt_legal3.txt 3   # 6 runs, gate-passing AdEMAMix
bash lab/pop_legal_sweep.sh lab/jobs_opt_init.txt 2     # 4 runs, the init control
```

### 4.1 The per-replica `local_ce` distributions — **LEGAL**

Full distributions, not best replicas. `div@min` is the output-diversity
detector (fraction of distinct answers over 256 inputs) evaluated on the
**best-`local_ce` replica**; a map collapsed to a constant reads 0.004.

| run | optimizer | lr | seed | min | p10 | **median** | max | spread | < 1.0 | `train_exact_hard` | div med | **div@min** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`LI_init_lr0`** | **— (random init)** | **0** | 0 | **2.111** | 2.131 | **2.169** | **2.241** | **0.060** | 0 | 0.001 | **0.965** | **0.973** |
| `LI_init_lr0_s1` | — (random init) | 0 | 1 | 2.080 | 2.118 | 2.169 | 2.240 | 0.074 | 0 | 0.000 | 0.969 | 0.777 |
| `LI_init_lr0_s2` | — (random init) | 0 | 2 | 2.103 | 2.131 | 2.173 | 2.239 | 0.063 | 0 | 0.001 | 0.957 | 0.930 |
| `LI_init_p64` | — (random init, **P=64**) | 0 | 0 | 2.117 | 2.137 | 2.183 | 2.279 | 0.074 | 0 | 0.001 | 0.965 | 0.965 |
| `LO2_soap_1e3` | SOAP | 1e-3 | 0 | **2.005** | 2.037 | **2.075** | 2.462 | 0.220 | 0 | 0.000 | 0.727 | 0.625 |
| `LO2_soap_3e4` | SOAP | 3e-4 | 0 | 2.071 | 2.097 | 2.134 | 2.267 | 0.092 | 0 | 0.001 | 0.953 | 0.973 |
| **`LO2_adamw_1e3`** | **AdamW** | **1e-3** | 0 | **2.168** | 2.190 | **2.260** | 2.616 | 0.198 | 0 | 0.001 | 0.363 | 0.363 |
| `LO2_adamw_3e3_s1` | AdamW | 3e-3 | 1 | 2.543 | 2.563 | 2.683 | 3.205 | 0.247 | 0 | 0.001 | 0.363 | 0.047 |
| `LO2_soap_3e3_s1` | SOAP | 3e-3 | 1 | 2.210 | 2.349 | 2.688 | 3.698 | 0.556 | 0 | 0.001 | 0.387 | 0.430 |
| `LO2_adamw_3e3` | AdamW | 3e-3 | 0 | 2.505 | 2.579 | 2.753 | 3.487 | 0.357 | 0 | 0.001 | 0.348 | 0.348 |
| `LO_soap_3e3` | SOAP | 3e-3 | 0 | 2.112 | 2.487 | 2.770 | 3.616 | 0.547 | 0 | 0.001 | 0.383 | 0.422 |
| `LO2_soap_3e3_s2` | SOAP | 3e-3 | 2 | 2.324 | 2.442 | 2.785 | 3.843 | 0.549 | 0 | 0.000 | 0.383 | 0.445 |
| `LO2_adamw_1e2` | AdamW | 1e-2 | 0 | 2.780 | 2.870 | 3.137 | 4.288 | 0.483 | 0 | 0.001 | 0.441 | 0.508 |
| `LO3_ade_a8_b39_3e3` | AdEMAMix β₃0.9 | 3e-3 | 0 | 3.016 | 3.127 | 3.435 | 4.744 | 0.506 | 0 | 0.001 | 0.570 | 0.582 |
| `LO_soap_1e2` | SOAP | 1e-2 | 0 | 2.875 | 3.077 | 3.502 | 5.597 | 0.779 | 0 | 0.001 | 0.383 | 0.500 |
| **`LO_adamw_s1`** | **AdamW (published baseline)** | **3e-2** | 1 | 3.178 | 3.322 | 3.581 | 5.699 | 0.708 | 0 | 0.001 | 0.578 | 0.344 |
| **`LO_adamw_s0`** | **AdamW (published baseline)** | **3e-2** | 0 | 3.183 | 3.261 | 3.619 | 5.241 | 0.569 | 0 | 0.001 | 0.547 | **0.004** |
| `LO_ade_a0_3e2` | AdEMAMix α=0 | 3e-2 | 0 | 3.236 | 3.258 | 3.627 | 5.172 | 0.534 | 0 | 0.000 | 0.551 | 0.348 |
| `LO_soap_3e2` | SOAP | 3e-2 | 0 | 3.084 | 3.210 | 3.687 | 6.637 | 0.977 | 0 | 0.001 | 0.598 | 0.598 |
| **`LO_adamw_s2`** | **AdamW (published baseline)** | **3e-2** | 2 | 3.137 | 3.275 | 3.838 | 5.439 | 0.601 | 0 | 0.001 | 0.617 | 0.449 |
| `LO_soap_1e1` | SOAP | 1e-1 | 0 | 3.441 | 3.684 | 4.114 | 7.759 | 1.053 | 0 | 0.001 | 0.758 | 0.430 |
| `LO3_ade_a8_b39_1e2` | AdEMAMix β₃0.9 | 1e-2 | 0 | 3.707 | 3.850 | 4.193 | 6.749 | 0.726 | 0 | 0.001 | 0.570 | 0.613 |
| `LO3_ade_a2_b399_3e3` | AdEMAMix β₃0.99 | 3e-3 | 0 | 4.297 | 4.454 | 5.051 | 11.467 | 1.436 | 0 | 0.001 | 0.445 | 0.414 |
| `LO3_ade_a2_b399_1e2` | AdEMAMix β₃0.99 | 1e-2 | 0 | 4.821 | 5.181 | 5.674 | 11.937 | 1.270 | 0 | 0.002 | 0.578 | 0.012 |
| `LO3_ade_a2_b399_3e2` | AdEMAMix β₃0.99 | 3e-2 | 0 | 4.964 | 5.386 | 5.929 | 12.139 | 1.212 | 0 | 0.001 | 0.465 | 0.344 |
| `LO3_ade_a8_b39_3e2` | AdEMAMix β₃0.9 | 3e-2 | 0 | 4.607 | 5.338 | 6.310 | 9.680 | 0.804 | 0 | 0.001 | 0.309 | 0.004 |
| `LO_soap_3e1` | SOAP | 3e-1 | 0 | 5.370 | 5.668 | 6.247 | 8.178 | 0.451 | 0 | 0.000 | 0.891 | 0.812 |
| `LO_ade_a8_b399_1e2` | AdEMAMix β₃0.99, β₂0.95 | 1e-2 | 0 | 5.982 | 6.231 | 7.226 | 13.699 | 1.102 | 0 | 0.002 | 0.402 | 0.152 |
| `LO_ade_a8_b399_3e2` | AdEMAMix β₃0.99, β₂0.95 | 3e-2 | 0 | 6.715 | 7.998 | 10.439 | 15.024 | 0.798 | 0 | 0.001 | 0.078 | 0.500 |
| `LO_ade_a2_b3999_3e2` | AdEMAMix β₃0.999 | 3e-2 | 0 | 8.347 | 9.251 | 11.681 | 15.611 | 0.632 | 0 | 0.001 | 0.137 | 0.133 |
| `LO_ade_a8_b3999_3e2` | AdEMAMix β₃0.999 | 3e-2 | 0 | 9.308 | 10.343 | 12.472 | 16.819 | 0.603 | 0 | 0.001 | 0.047 | 0.168 |
| `LO_ade_a8_b39999_3e2` | AdEMAMix β₃**0.9999** | 3e-2 | 0 | 9.368 | 10.425 | 13.033 | 16.655 | 0.574 | 0 | 0.001 | 0.059 | 0.191 |
| `LO_ade_a8_b3999_1e1` | AdEMAMix β₃0.999 | 1e-1 | 0 | 10.256 | 10.964 | 13.665 | 17.708 | 0.553 | 0 | 0.001 | 0.016 | 0.004 |

**33 runs. 1,056 replicas. Zero below `local_ce` 1.0, let alone the 0.006 cliff.
Best `train_exact_hard` anywhere: 0.002.**

### 4.2 Did either optimizer shift or fatten the tail? — the honest answer

**AdEMAMix: no, and it is uniformly worse.** At its gate-passing settings
(β₂ 0.95, β₃ 0.9) it lands at median 3.435–6.310 against AdamW's 3.581–3.838 —
i.e. at best statistically indistinguishable from AdamW, never better. Its α=0
control lands at 3.627, inside the AdamW seed band, which confirms the harness
rather than the method. As β₃ increases the distribution moves **right**,
monotonically, to a median of 13.665 at β₃ = 0.999. **The slow-EMA idea is
falsified on this objective in both directions: it does not help, and the more
of it you use the worse it gets.**

**SOAP: it *appears* to shift left — and the control removes it.** At face value
`LO2_soap_1e3` (min **2.005**, median **2.075**) beats every legal value in the
project, including `alu-relational`'s 2.343 at P=1 and `alu-population`'s 2.304
from `--assoc` at P=32/64. The mandate asked for the diversity check before
believing it; I ran that, **and I also ran the control that turned out to
matter more.**

**Control 1 — learning rate.** The lr-matched AdamW row is the comparison that
was never run: `alu-population` §7 swept lr only on the DIAGNOSTIC objective.
Sorting §4.1 by median, the ordering is set by **lr**, not by the optimizer:

| lr | AdamW median | SOAP median |
|---:|---:|---:|
| 3e-4 | — | 2.134 |
| 1e-3 | **2.260** | 2.075 |
| 3e-3 | 2.683–2.753 | 2.688–2.785 (3 seeds) |
| 1e-2 | 3.137 | 3.502 |
| 3e-2 | 3.581–3.838 (3 seeds) | 3.687 |
| 1e-1 | — | 4.114 |

Both optimizers move together and monotonically in lr; at 3e-3 they are
identical across three seeds each. AdamW at lr 1e-3 (2.260) recovers **88%** of
the distance from the lr-3e-2 baseline (3.619) to SOAP's best (2.075). **The
apparent SOAP improvement is 12% optimizer and 88% learning rate.**

**Control 2 — output diversity.** SOAP's low-`local_ce` rows are *not* the
`--assoc` degeneracy: `div@min` is 0.625 (SOAP 1e-3) and 0.973 (SOAP 3e-4)
against `--assoc`'s 0.035–0.203 and its collapsed worst replicas at 0.004. So
the `--assoc` collapse is not what is happening here. But diversity is not clean
either — SOAP 1e-3 has median diversity 0.727 against random init's 0.965, i.e.
a 25% loss of distinct answers accompanying a 5% `local_ce` gain.

**Control 3 — random initialisation.** This is the one that decides it, and it
had never been measured.

### 4.3 THE CORRECTION: `local_ce` at random init is 2.08–2.24

`lab/jobs_opt_init.txt`, 4 runs, **160 replicas**, `--lr 0` so nothing trains:

| | min | p10 | median | max | spread | diversity (median) |
|---|---:|---:|---:|---:|---:|---:|
| **random init**, 3 seeds at P=32 + 1 at P=64 | **2.080–2.117** | 2.118–2.137 | **2.169–2.183** | 2.239–2.279 | **0.060–0.074** | **0.957–0.969** |

Three consequences, and they reach past this branch:

1. **The plain legal baseline trains `local_ce` 65% *away* from the cliff.**
   Random init sits at median 2.17; 1,200 steps of the legal end-of-chain
   objective at lr 3e-2 puts it at 3.58–3.84. Diversity falls from 0.96 to
   0.55–0.62 at the same time. The objective is not failing to approach the
   discrete solution — it is **moving away from it**, on both measures.
2. **The project's best-ever "legal" `local_ce` is worse than an untrained
   model.** `alu-relational`'s 2.343 (P=1) and `alu-population`'s 2.304
   (`--assoc`, 3 seeds, P=64) both sit *above* random init's 2.08–2.12 min and
   above its 2.17 median. `--assoc`'s celebrated 22% leftward shift is a partial
   *return toward initialisation* — which is fully consistent with
   `alu-population`'s own diagnosis that it works by destroying the map, and
   strictly stronger than that diagnosis: it did not need the diversity detector
   to be ruled out.
3. **The only value in the whole project below random init is SOAP at lr 1e-3
   (2.005 min / 2.075 median), and it is a 3–5% dip bought with a 25% loss of
   output diversity, at `train_exact_hard` 0.000.** Against a cliff at 0.006
   that is a factor of **346**. It is noise on the scale that matters.

**So the tail question is answered: neither optimizer shifts or fattens the tail
toward the cliff.** Every legal distribution measured here has spread 0.06–1.44
and sits between 2.0 and 13.7, against a target of 0.006. The tight unimodal
blob `alu-population` reported is confirmed, and is now bracketed on the *other*
side too: the blob's left edge is random initialisation.

---

## 5. What this does and does not close

**Closes.** Conditioning is not the missing variable. The claim rested on the
m1 identifiability ratio of 166 — the tables *are* uniquely determined by the
data, so the failure is *finding* them, which is a conditioning question by
construction. That inference is correct and the answer is now measured: a
genuine full-matrix preconditioner, affordable at 6,820 parameters and verified
to give a 374× win on a problem that *is* ill-conditioned, does not move the
legal objective at all. Nor does a slow-EMA method with the identical
verification. **Both were tested at the conditions where AdamW demonstrably
works, and both work there.**

**Sharpens.** The obstruction is now bracketed rather than merely observed. It
is not "gradient descent cannot get close enough"; it is that **the legal
objective's gradient points away from the discrete solution from the first
step**, and it does so for AdamW, for a second-order method, and for a slow-EMA
method alike. `alu-relational`'s structural argument (`Tmul`'s 200 cells are
unidentifiable from the end-of-chain label) predicts exactly this: if the cells
carry no information in the objective, the objective is free to move them
anywhere, and it moves them somewhere worse than random. The three optimizer
families disagreeing about *how fast* they go the wrong way, and agreeing that
they do, is what a genuinely unidentifiable direction looks like.

**Does not close.** Nothing here says a *different objective* cannot work. It
says the optimizer is not the variable. Two things in this branch would transfer
to a future architecture:

* **Put the objective O(1) learned ops from the parameters** — already
  `alu-relational`'s recommendation, and §4.3 strengthens it: at 39 sequential
  ops the end-of-chain gradient is not merely uninformative, it is
  anti-informative.
* **Measure the value of your indicator at random initialisation before you
  celebrate moving it.** This branch's single cheapest run (4 minutes, `--lr 0`)
  retired a metric the project had been optimising for two branches.

---

## 6. Falsified, including my own premises

1. **"Nothing second-order has been tried, and the ALU is small enough for full
   matrices"** — true, and now tried, at three factorisation levels up to a
   genuine 2000×2000 per-tensor preconditioner. Null on the legal objective;
   *worse* than AdamW on the diagnostic one (basin rate 0.06 vs 0.34).
2. **"A slow EMA suits weak gradient signal and sharp late transitions"** — the
   ALU's transition *is* sharp, and the slow EMA is monotonically harmful,
   `train_exact_hard` 1.000 → 0.006 as β₃ goes 0.9 → 0.999. The competition's
   step budget (~1,300 at Hard) is below AdEMAMix's own useful regime.
3. **My hypothesis that SOAP's soft/hard split was state de-saturation** —
   falsified by direct measurement: sharpness 0.9995 vs AdamW's 1.0000, logit
   RMS *higher* than AdamW's. The split tracks `mul_fn` instead.
4. **`local_ce` as a cross-optimizer indicator** — falsified; §2.3.
5. **The project's reading of every leftward `local_ce` shift as progress** —
   falsified by the init control; §4.3.
6. **The SOAP reference implementation's moment placement** — numerically unsafe
   at large gradient scale; §1.2.

---

## 7. The dense transformer

**LEGAL.** Same recipe `explore/grok-optimization` used for its optimizer
comparison — e1, 10,000 steps, batch 128, wd 0.1, seed 74 — whose table reads
AdamW rung-1 0.000, Muon 0.026, Schedule-Free 0.000, and whose branch verdict is
that the objective is *saturated* (train exact 1.00 held for 198,000 steps while
held-out never leaves the floor), not under-optimised.

`lab/make_optsub.py` is that branch's generator with SOAP and AdEMAMix inlined
into the emitted `submission.py` (one self-contained file, lints clean at 25 KB).

```bash
bash lab/run_dense.sh      # 5 cells: AdamW ctl, SOAP x2, AdEMAMix x2
```

**Five cells, all `MAX_T = 0` and all `OOD_N_MAX_T = 0`** (tag `B-opt2` in
`lab/archive.jsonl`; per-rung tables in `lab/logs/_dense.out`):

| run | optimizer | lr | **rung-1** | test | ood | mean acc | train s |
|---|---|---:|---:|---:|---:|---:|---:|
| `G_adamw_ctl` | AdamW (control) | 1e-3 | **0.000** (0/38) | 0.027 | 0.030 | 0.0283 | 389 |
| `G_soap_1e3` | SOAP | 1e-3 | 0.026 (1/38) | 0.020 | 0.050 | 0.0350 | 337 |
| `G_soap_3e3` | SOAP | 3e-3 | **0.000** (0/38) | 0.020 | 0.050 | 0.0350 | 336 |
| `G_ade_3e4` | AdEMAMix α=8 β₃0.9999 | 3e-4 | 0.026 (1/38) | 0.020 | 0.020 | 0.0200 | 286 |
| `G_ade_1e3` | AdEMAMix α=8 β₃0.9999 | 1e-3 | 0.053 (2/38) | 0.020 | 0.030 | 0.0250 | 286 |

**The control reproduces the sibling branch to three decimals** — its optimizer
table reads AdamW rung-1 0.000 / test 0.027 / mean 0.028, and this run reads
0.000 / 0.027 / 0.028. So the harness and recipe are the same one that branch
measured, and the two new rows are directly comparable to its AdamW / Muon /
Schedule-Free row.

**Reading, per the branch's own screening discipline: report the row, not the
cell.** The rung-1 spread across these five is 0–2 examples out of 38, against a
stated variance floor of **one example** and a 50-run e1 histogram of 0/38 (×37),
1/38 (×12), 2/38 (×2). Every cell sits inside that histogram. `G_ade_1e3`'s
2/38 ties the best e1 value that branch ever recorded across 57 runs — and 2/38
is `MAX_T = 0`, identical in score to 0/38, which is the whole point of the
current metric. **Nothing here is distinguishable from the null.**

I recorded a prediction of 0/38 for all five before running them. That was
slightly conservative: three cells landed at 1–2/38. The prediction's *content* —
`MAX_T = 0` everywhere, indistinguishable from the field — holds, and the reason
holds with it: train exact accuracy is 1.00 by step 2,000 and holds it for the
next 198,000, so a better optimizer has no headroom to consume. **The optimizer
question is now closed for the dense transformer as well as for the ALU**, which
is what made these five cells worth the 27 minutes.

The ALU result does not depend on these cells. The ALU is the architecture the
project's remaining hope rested on, it is where the optimizer sweep had the
actual hole, and §2–§4 close it on 62 runs; §7 closes the optimizer question for
the dense model too, so it is now closed for the whole project.

---

## 8. Compliance

* **No file under `data/generated/` was read, printed, sampled or summarised.**
  Every operand in §1–§6 is self-generated from `--modulus` by enumerating the
  units of a modulus chosen from the generator source. The §7 cells go through
  `lab/run_experiment.py`, which is the evaluator, and never inspect the data.
* **Nothing was submitted to the hosted service.** No `one-layer` invocation, no
  network call, no `login`/`submit`.
* Every teacher-forced row (§2) is labelled **DIAGNOSTIC** and carries the
  reason (the tape is recorded from a `--construct`ed model; rules 2, 7). No
  DIAGNOSTIC number is offered as a result.
* **A custom `torch.optim.Optimizer` is explicitly permitted** and is the only
  participant-controlled machinery added. There is no custom training loop, no
  participant-controlled backward, no manifest override. `step()` receives `p`
  and `p.grad` and nothing else. The `batch_dims` argument changes which
  *parameters* share a preconditioner; it does not touch the graph.
* `--opt adamw` is the previous code path exactly; `LO_adamw_s0/s1/s2` reproduce
  the published baseline (min 3.14–3.18 vs 2.975, median 3.58–3.84 vs 3.347 —
  the small offset is this branch's `probe_pop_legal.py` default seed set, not a
  code change; the two `--assoc`-free baselines in `pop_legal_runs.jsonl` and
  `opt_legal_runs.jsonl` bracket each other).
* `benchmark.assert_model_state` is called against the real
  `ModelSpec(maximum_model_state_elements=500_000_000)` in every `probe_pop.py`
  run.
* **Negative results are all here**, including the ones that contradict my own
  mandate's premise and the ones that retract a project-level claim (§4.3).
  The AdEMAMix round-1 gate failure is reported as a confound I introduced (§2.2)
  rather than deleted.
* `explore/grok-optimization` and `explore/alu-population` were **read only**.
  Nothing in either worktree was edited or executed.
* The five evaluator cells in §7 were run through `lab/run_experiment.py` on a
  fixed-step manifest and every one is archived in `lab/archive.jsonl`,
  including the ones that scored above the control.

---

## 9. Reproduction

```bash
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd /home/scratch.arohan_hw/git/one-layer-deeper/.worktrees/alu-optimizer

# the optimizers must pass their own self-check first -- expect ALL PASS
$V lab/test_optim_extra.py

# the graph is unchanged -- must print 1.000 / 1.000 / 1.000
$V lab/probe_pop.py --construct --pop 4 --eval-n 512 --held-x 512 --eval-chunk 128

# THE GATE                                                    [DIAGNOSTIC]
bash lab/pop_sweep.sh lab/jobs_opt_gate.txt 4
bash lab/pop_sweep.sh lab/jobs_opt_gate2.txt 4
bash lab/pop_sweep.sh lab/jobs_opt_sharp.txt 4     # saturation / logit scale

# THE LEGAL SCREEN                                            [LEGAL]
bash lab/pop_legal_sweep.sh lab/jobs_opt_legal.txt 3
bash lab/pop_legal_sweep.sh lab/jobs_opt_legal2.txt 4   # the lr control
bash lab/pop_legal_sweep.sh lab/jobs_opt_legal3.txt 3   # gate-passing AdEMAMix
bash lab/pop_legal_sweep.sh lab/jobs_opt_init.txt 2     # THE INIT CONTROL

# the dense transformer -- 5 cells, all MAX_T=0                [LEGAL]
bash lab/run_dense.sh
```

Results: `lab/opt_runs.jsonl` (gate; per-replica accuracy and `local_ce`
vectors, `n_basin`, structure scores, state sharpness, logit RMS, ms/step) and
`lab/opt_legal_runs.jsonl` (legal; `local_ce` quantiles, the 12 lowest, counts
below each threshold, and the output-diversity detector per replica). Per-run
logs in `lab/logs/`.
