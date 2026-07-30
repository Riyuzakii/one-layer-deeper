# Resumption log — exploration session, 2026-07-25

Written at the end of a GPU session with work still in flight. Read this plus
`lab/BRIEF.md` and you can restart everything without reconstructing context.

Nothing was merged to `main`. No dataset file under `data/generated/` was ever read.
**One hosted Hard submission has been made** (see "Round 6" below) — it returned h1's
split structure and the first real H100 timing calibration.

---

## 1. State of the world

**The metric changed on 2026-07-24** (commit `79f0a09`). Hard ranks by **Max T** —
the largest `T ∈ {1,2,4,8,16,32,64}` whose rung and every lower rung are **100%
exact** — then by OOD-N Max T, then by earliest submission. No partial credit.
`mean_exact_accuracy` is a diagnostic with no ranking value. `lab/findings.md` from
the *previous* session optimises that obsolete metric; read it for what was ruled
out, not for what to do.

**~900 experiments across thirteen branches. Every single one scored MAX_T = 0.** Best
rung-1 anywhere is 3/38 on e1, against a trivial-predictor floor of 1/38.

**FINAL STATE: four of five components are solved; the fifth is closed.** Parsing,
composition in T, the depth controller and the readout representation all work
end-to-end. Training the arithmetic tables has a *provably exact, reachable* solution
(`train_exact_hard` **1.000** / held **1.000** by step 600, inside a Hard budget) and
**no legal way to reach it**. See "Round 4" below — the closure survives 544-replica
screening and is structural, not empirical. **My recommendation is to stop the
architecture search.**

### Settled — do not re-derive

- The failure is a **memorisation/generalisation gap**, not capacity, depth, or
  representation. Models hit ~100% train exact accuracy by step 300 with held-out at
  chance.
- **Falsified:** eight input/output representations; depth and iteration count
  including an ideal-halting diagnostic that hands the model the true T (fails on
  four datasets); on-manifold state via straight-through re-embedding; error
  compounding as the constraint (rung profile is flat, not decaying); capacity down
  14×; 12× more data; wd=1.0 at 20k steps; the Fourier/rotation route; composition
  self-consistency (semigroup law constrains `M(N,·,T) = h^T` but says nothing about
  `h`'s value, so it adds zero information at unlabelled operands).
- **Iteration/depth in T is SOLVED.** A weight-tied block gives perfect
  T-extrapolation (trained T≤3 → 1.000 at T=8 on seen x). An **ordered** depth
  selector (learned scalar location + annealed window) plus randomising the loop
  budget per step reaches the oracle ceiling. An entropy penalty is the *wrong* fix —
  it makes the selector sharp but T-independent.
- **Known trap:** `h ← core(h + base)` with a pre-norm readout makes iteration count a
  gauge freedom the loss cannot see. Every depth sweep in this repo before this
  session, including the prior session's "Axis A confirmed", measured a no-op
  recurrence.

### The three bottlenecks, cleanly separated

| # | bottleneck | status |
|---|---|---|
| 1 | applying the step T times, exactly | **solved** — exactness composes 1.000 at every rung, five regimes, bf16+amp, no drift after 64 compositions |
| 2 | per-step arithmetic on unseen operands | representation **solved and uncapped** (digit readout, below); *training* it is open |
| 3 | prompt → digit-slot parsing | **solved**, verified at multi-digit T |
| 4 | **training the discrete transducer** | open — credit assignment through ~280 sequential soft steps (`alu-depth`, `alu-credit`) |
| 5 | depth controller extrapolating in T | **SOLVED** — 12 learned scalars reach 1.000 at every rung T=1..64, hard-committed and under `--eval-hard`, at Easy and Medium settings and at 22-bit hp1 |
| 6 | eval budget | **manageable, but only on the cheap graph** — early halting is necessary and NOT sufficient on the 257-step ALU |

**Bottleneck 5, solved (`explore/depth-controller`).** Per-rung 1.000 at T = 1,2,4,8,16,32,64;
seeds at MAX_T=64: Easy/N=329 5/5, Easy/N=10403 4/5, Medium/N=10403 2/3, Medium/hp1 22-bit
2/3. Baseline in the same setting was Easy 4/4/0, Medium 0/0/0. All moduli non-degenerate
(7/7 distinct rungs), unlike e1/e2.

Attribution — four changes, ablated, and **not** the ones the mandate predicted:
1. **Do not dump unspent halting mass on the last candidate** (highest leverage, one line):
   otherwise "never halt" is exactly correct for the deepest training T.
2. **Commit to the mode, not the blend** — Medium 0 -> 4 with no retraining. A good part of
   alu-compose's "Medium fails downward" was a **readout artifact**; the decision at T=1,2
   was already right.
3. Straight-through one-hot register.
4. **Detector margin scored in log space** — the ladder rode on one scalar; widening the
   margin 1 -> 14 took Easy 3/5 -> 5/5.

Self-consistency `step(k+1)=step(step(k),1)` is **sufficient but not necessary**, and is
**actively harmful on `(place,digit)`-indexed heads** (2 -> 0): the orbit's increment
collapses to identity because nothing anchors it. The law has content only when its orbit
operator is anchored by the task loss — true in a counter, false in a pointer head. Cause
(a), coverage in the T field, remains exactly true for the pointer family.

**Eval budget, measured (contended GPU, pessimistic):** Easy 7.3s/30s (4.1x), Medium
60.8s/300s (4.9x), all 16 splits, both full ladders. But `fixed64` loses both ladders on
Easy, and `fixed16` — the proxy for a correctly trained controller on the 257-step
`DigitALU` — costs **31.3s of 30s** and forfeits 2 OOD-N rungs. **"ACT fits the Easy budget"
is FALSE on the 257-step graph**; with `tree:quotient` the projection is ~15s (2.0x). This
is an independent reason the cheap graph is load-bearing.

Residual: the learned unit digit lands on the wrong digit in 1 of 5 seeds (a *deterministic*
attractor on Medium; more steps and higher lr both measured flat). Diagnosed fix, one loss
term, not yet run: the countdown step and the consistency orbit's increment are the same
vector, so requiring `dec(inc(r)) = r` identifies it with no labels. Command in
`lab/reports/depth-controller.md` §12.

### Second round — the endgame, measured (`explore/alu-compose`, complete)

**Given a *perfect* single squaring step, the pipeline certifies MAX_T = 2 on Easy and
MAX_T = 0 on Medium — not 64.** The arithmetic was never the last bottleneck.

*What works:* parser → constructed `DigitALU` × T → digit readout gives exact-example
accuracy **1.000 at every rung T = 1,2,4,8,16,32,64** at five regimes (N=323 S=3;
12 sampled 10/11-bit; N=10403; N=4028033 22-bit; 4 sampled 30/32-bit), identical in
fp32, in **bf16+amp** (the manifests' dtype), and with discrete states. No drift: after
64 compositions the minimum over examples of `max_d p(d)` is 1.00000 — saturated tables
make `softmax(log p)` a fixed point. **Downward generalisation to T=1,2 works by
construction in the step** (1.000 with no T=1/T=2 supervision anywhere).

**P1 — the depth controller cannot extrapolate in T.** Parser and ALU held at
construction, only the controller trained: **28 cells** (6 parameterisations, 3–1409
params × 2 seeds × 2 tiers × 2 moduli) → **every Easy cell MAX_T=2, every Medium cell
MAX_T=0**; constructed controllers reach 64. Causes: (a) **coverage in the T field** —
Easy never shows the tens place of T, Medium never shows digits 2 or 3; structurally
the same argument as the residue coverage bound, applied to a different field; (b) the
soft window makes the loss flat inside the correct bin, so `loc` is pinned to an
interval, not a value (Easy ordinal fit `loc ≈ 0.83T − 0.92`, wrong by 1.4 iterations
by T=8).

**P2 — the eval budget binds on Easy, and fails hard.** A fixed 64-iteration readout
throws `TimeoutError` in the `test` split on e1 → **run status *failed***, which is
*below* the leaderboard floor (`service/db.py` counts only `status='succeeded'`) and
would waste a 1/day Hard attempt. Same on m1: 305.7s vs a 300s budget, OOD-N ladder
truncated to 3/7 rungs — tie-break silently forfeited. ACT/PonderNet early exit fits
(e1 24.8s/30s, m1 111.5s/300s, hp1 274.5s/1800s, all 16 splits) but **margin is ~1.1×,
not the ~5× measured for a cheap model**, and 1 of 3 tier-faithful Easy runs still lost
the whole OOD-N ladder. **Any candidate must halt early, never run to a fixed depth.**

**P3 — step famine.** Tier-faithful Easy affords **14–18 optimizer steps** with this
model (~150 on an idle GPU); Medium ~22 measured. `DigitALU` needs ≳4,000 steps and
still plateaus at 0.78. **No tier affords the step count at which the architecture is
already known to fail.** This makes chain-shortening a hard requirement, not a
nicety.

### Round 2 results — the ALU family, measured to its edge

**`explore/alu-depth` (complete) — depth is not the binding variable, and it produced
the graph everything else should use.**

`tree:quotient`: same hypothesis class, same discrete alphabet (10/2/2/11), 6,820
params, constructed ceiling **1.000 soft AND argmax-hard** at N = 323/899/2021/10403
and on 12 unseen sampled moduli — at **39 sequential soft steps instead of 257**,
103 ms/step instead of 390. Throughput: **4.6x (Easy) / 7.9x (Medium) / 12.2x (Hard)**
more optimizer steps; add `prefix` above Easy (loses at Easy, 43 vs 39; wins at
Medium 71 vs 83 and Hard 101 vs 155). **Adopt unconditionally.**

But removing 85% of the chain did not move the discrete solution: `train_exact_hard`
across the ladder 257/195/117/83/62/39 reads 0.008/0.008/0.012/0.000/0.000/0.004 -
zero everywhere; `train_exact` is not monotone in depth. Also falsified: **larger
internal radix cannot be stacked** (coverage 0.947 -> 0.026 -> 0.000 at radix
10/100/1000, because at radix 1000 with S=1 the pair table is indexed by `(x,x)` and
*is* a residue-indexed readout - the radix is the dial between shallow-and-memorising
and deep-and-compositional); reduce-less-often (quotient range grows as `10^p`);
straight-through at every chain length (loss 15-17 at depths 257, 39, 21).

**`explore/alu-credit` (complete) - ~160 training procedures, none works.**

Best legal `train_exact_hard` **0.020**; best of any kind 0.183 (short chain, held
0.000). The state-pressure direction is falsified monotonically at 12,000 steps:
sharpness 0.762 -> 0.967 while `train_exact` collapses 0.616 -> 0.004 and
`train_exact_hard` stays pinned in 0.000-0.020. **The soft channel is not a crutch
over a nearly-found discrete solution; it is the entirety of what the model has.**

Null: annealing, Gumbel-to-discrete, straight-through, staged unfreezing, truncated
BPTT, magnitude curriculum, entropy/commutativity/`Tsub.Tadd=id` regularisers,
bounded losses, every init family, every optimiser setting, target propagation.
Not legal (need the true register trace, rule 7) and the only things that moved
anything: `--construct`, `--deep-sup`, `--teacher-force`, cross-modulus curricula.

**The one positive, and it is parameter-level:** teacher forcing drives gauge-invariant
structure scores to `mul_lo` 0.90 / `add_shift` 0.78 / `sub_shift` 0.817 (random
baseline 0.23-0.28), **78% of it within 20 optimizer steps** - inside the Easy budget.
Snapping cannot touch these. So per-step inputs *do* teach the tables and the tables
*are* identifiable; the residual 10-20% of wrong cells is what the soft channel
absorbs. **The open problem is table identification, not credit assignment and not
relaxation tightness.**

### Round 3 — the ALU's legal training is CLOSED

**`explore/alu-credit` Stage 1 (DIAGNOSTIC ceiling).** With m1-scale operands on the
39-step `tree:quotient` graph, **untied**, teacher forcing on the true register trace
reaches `train_exact_hard` **0.951** / `held_exact_hard` **0.946** — soft and hard
agreeing to 0.001, the first genuinely discrete transducer in the repo, generalising at
a scale with no room to memorise, plateauing by **step 1,000** (inside a Hard budget).
Scale is the active ingredient: same graph at e1 scale gives 0.028. **Teacher forcing is
illegal (rules 2, 7) — this is a ceiling, not a result.**

- The control variable is `local_ce`, with a razor cliff: **0.0050 -> 0.951**,
  **0.0073 -> 0.103**, 0.070 -> 0.000. That is the `(1-eps)^50` law measured. Use
  `local_ce` as the early indicator; it separates basins long before exact-match moves.
- Basin rate **1 of 7 untied seeds**, deterministic per seed. **Weight ties are harmful
  for learning from random init** (the one successful seed fails when tied,
  `local_ce` 0.0050 -> 0.1187) even though ties help *repair* near a solution. Two
  different problems — do not carry a tie result across them.
- **Target propagation collapses at every mixing weight** (0.3/1.0/3.0, 3 seeds, +/-ties):
  `mul_gauge` 0.100 = a constant map, unanimously. A low weight does *not* recover the
  baseline. It is an attractor, not a tunable balance.

**`explore/alu-relational` (complete) — no legal signal with intra-squaring content.**
37 training runs; highest `train_exact_hard` anywhere **0.001**. Three families, all null:
the relational increment law, dual-path agreement, and the algebraic re-screen
(`--sym`, `--inv`, `--assoc`, `--cancel`).

- **The missing control, now measured:** the plain LEGAL baseline at Stage-1 conditions
  is a hard zero — `local_ce` **3.5-4.2** against the 0.005-0.0073 cliff, i.e. ~500-800x
  the per-op error rate that separates 0.951 from 0.000.
- **The relational family is closed by its own ILLEGAL ceiling.** Writing the true
  identity `(x+1)^2 = x^2 + 2x + 1` into the loss lands inside the baseline band on every
  metric. No legal version with learned operators can beat the version handed the true
  coefficients.
- **A Stage-1 re-screen failed to overturn an e1-scale negative** (`--sym`, `--inv` null
  at both scales). "e1 negatives do not survive" is NOT universal — check, don't assume.
- `--assoc` collapses the map to a constant (`out_div` 0.002); `--cancel` blocks the
  collapse and buys nothing.
- **The agreement-law dichotomy (closes the class, not just the instances):** any
  agreement-style law either **tolerates** the maximum-entropy solution (KL form —
  already satisfied at init, so no gradient) or **prices** it (CE form — starts at
  `2 ln 10` and does not descend in 4,000 steps even with no competing loss). There is
  no third framing. Straight-through, the direct test of "graded discrete landscape,
  bad estimator", is actively destructive (`local_ce` 13.83, the worst in the report).
- **Structural closure argument:** `Tmul`'s 200 cells are unidentifiable — not from the
  end-of-chain label, not from any generic algebraic law, and the only law that *would*
  identify them (multiplication as repeated addition) is the definition of the
  arithmetic, i.e. rule 2 moved into the loss.

**The one result worth carrying to a future architecture (a measurement, not a recipe):**
**a constraint's conditioning is set by how many learned ops separate it from the tables,
not by its information content.** Laws evaluated O(1) ops from a table (adder
associativity/commutativity/cancellativity) exactly repair **50 of 400 corrupted cells,
3/3 seeds**, against the end-of-chain label's **0/5 at k=20 of 337** — a **~14x wider**
exact-repair basin, and the only objective still climbing past k=100. Laws evaluated
*through the chain* inherit the chain's ruggedness exactly. Both still fail from random
init, so this buys no submission — but it says the next architecture should put its
objective O(1) ops from its parameters.

**Compliance line, adopted and applied consistently (alu-relational §7, agreeing with
discrete-search §9):** a loss term is **legal** if it asserts *a generic algebraic
property of an operation the model already performs*; **not legal** if it asserts *the
specific relationship that constitutes the definition of the arithmetic* — that is rule 2
moved from the forward pass into the loss. Under this rule `--rel affine` ("the increment
is affine, i.e. the map is quadratic") was flagged and resolved **against**, since
"degree 2" is a fact about this task rather than a generic property. It measured null
anyway, so the ruling cost nothing.

**Two methodological rules earned the hard way:**
1. **Measure a signal's ILLEGAL ceiling before building its legal version.** Four runs
   closed the strongest family in `alu-relational`.
2. **Measure a signal's BASIN before optimising it.** Costs no training and predicts
   whether optimisation can go anywhere.

### Round 4 — the closure, made robust (`explore/alu-population`, complete)

**The conditional question, answered positively.** Replicas inside one run are as
independent as separate seeds — the basin is selected by *initialisation*, not data
order. Per-replica rate **0.26** under the working (diagnostic) signal: P=1 gives 1/7,
P=8 gives 3/3 runs containing a solved replica, P=32 gives 2/2; `1-(1-0.26)^P` puts
P=8 at 0.91 and P=32 at 0.9999. **The hit is `train_exact_hard` 1.000 / held 1.000 by
step 600** — higher and cheaper than the 0.951 quoted earlier in this log.

- **Differentiable selection works completely and is LEGAL**: mixture = argmax =
  CE-argmin = population best, in all six runs, held-out as well as train, under five
  selector variants, with no sharpening schedule (`alpha` saturates on its own).
- **Cost:** free to P=8 (1.29x), cheap to P=16 (1.55x), knee at P≈16-32; past 64 the
  marginal cost is flat at 3.4 ms/replica. State ceiling is *not* binding (P=256 =
  0.349%; ceiling P=73,313). Eval can slice the argmax replica and run at **P=1 cost**.
- **The evaluator's global grad clip gives the same result as per-replica clipping** —
  Adam absorbs the common scale, so replicas stay independent under the evaluator's
  own loop.
- **Cheaper alternatives do not suffice**: batch, LR decay, lr up/down, scale-spread,
  reinit-losers all null; only init scale 0.5->0.25 moved the rate (0.34->0.53), not
  enough alone.

**THE DECISIVE RE-SCREEN.** Every legal-signal screen in this project had been run at
P=1, and the basin is initialisation-selected — so the closure could have been an
artifact of screening design. It is not. All six terms `alu-relational` rules LEGAL,
ported to a replica dimension: **16 configurations, 544 replicas, 1,200 steps each.**

- **0 of 544 replicas reached `local_ce` 1.0**, against a cliff at 0.006. Best anywhere
  **2.304** vs 2.343 at P=1 — **32x the draws bought 1.7%**.
- **The one apparent exception is a degeneracy.** `--assoc` shifts the distribution left
  22% (median 3.35 -> 2.62), reproducibly over 3 seeds and at P=64 — but its best
  replicas predict the same answer for 80-96% of inputs and its worst collapse to a
  **literal constant** (diversity 0.004 = 1 answer in 256). It lowers `local_ce` by
  destroying the map. **`--cancel` does not prevent this**: it prices the *adder*, while
  the collapse is in the *composed map*.
- **Shape is the story.** Same P, graph and budget: the working diagnostic signal gives
  a **bimodal** population with a spike at zero (11/32 below the cliff); every legal
  term gives a **tight unimodal blob ~400x away** (relative spread 0.39-0.89). A
  population is the right instrument for a *stochastic* obstruction and the wrong one
  for a *systematic* one — and it reads systematic across all 544 replicas.

**Three corrections this branch makes to earlier entries in this log:**
1. The ceiling is **1.000/1.000 by step 600**, not 0.951.
2. **"Always commit to the mode" is not general** — at P=64 step 400 the blend *beat*
   the commit because `alpha` had not concentrated. Reported alongside
   `depth-controller`'s opposite finding; the accurate rule is **check `wmax` first**.
3. **Parameter-space averaging/ensembling is meaningless for every learned-table
   architecture here**, for a gauge reason: `Tmul`'s output code is a gauge, two solved
   replicas solve in *different* gauges, and their average is uniform. Stated as a
   general result, not a sweep null.

### Round 5 — the optimizer question closed, and the control that reframes it

**`explore/alu-optimizer` (complete).** SOAP and AdEMAMix had never been tried anywhere
in the project (the sweep covered AdamW / Muon / Schedule-Free / Grokfast / perp-Grad /
StableMax, all on the dense transformer; only AdamW hyperparameters on the ALU). They
target the one property the closure assumes but never measured: **conditioning**.

**Both pass a diagnostic gate**, so the nulls are interpretable — teacher-forced,
Stage-1, P=32: AdamW 1.000/1.000 with 11/32 replicas in basin; **AdEMAMix identical
(11/32)**; SOAP reaches the ceiling at 2/32 (5x worse basin rate). Full-matrix
preconditioning costs 3.2x wall clock at 6,820 params and buys nothing.

**Legal objective: 29 runs, 928 replicas, 0 below `local_ce` 1.0**, best
`train_exact_hard` 0.002. SOAP's apparent record (2.005 vs baseline 3.14) is **88%
learning rate** — an lr-matched AdamW control at lr 1e-3, never previously run on the
legal objective, recovers most of the shift. AdEMAMix is monotonically *worse* as beta3
rises (median 13.7 at beta3=0.999).

**THE CONTROL THAT REFRAMES EVERYTHING — `local_ce` at RANDOM INIT is 2.08-2.24**
(160 replicas, `--lr 0`, 4 minutes). The plain legal objective *trains* `local_ce` to
3.5-4.2, i.e. **training moves it AWAY from the cliff**, and the project's best-ever
legal value (`--assoc`'s 2.304) is **worse than an untrained model**. Every leftward
shift ever recorded on the legal objective, including SOAP's, is **partial regression
toward initialisation**, not progress.

So the obstruction is sharper than "cannot get close enough": **the legal objective's
gradient points away from the discrete solution from the first step, identically for
all three optimizer families.** Run `--lr 0` as a control before interpreting any
future "improvement" on this objective.

**Dense transformer:** 5 evaluator cells, all MAX_T=0/OOD_N 0; the AdamW control
reproduces `grok-optimization` to three decimals.

**Verdict: conditioning is NOT a gap. The closure now covers second-order and slow-EMA
methods**, tested at conditions where both demonstrably work.

**Also found: a real bug in the published SOAP reference code** — keeping the first
moment outside the eigenbasis makes numerator and denominator independently-rounded
projections that blow up in near-null directions. Moving both moments into the basin
(paper Algorithm 1) turned divergence into a 374x win on an ill-conditioned quadratic.

### Round 6 — the first hosted Hard run: what h1 actually looks like

One Hard submission was made (an `exp_axis` screening file that hard-codes
`_MAX_STEPS = 400`, so it used **19.3s of the 3600s budget** — 0.5%). Metrics saved at
`lab/hosted/hard_h1_exp_axis_400steps.jsonl`. Two structural facts fall out.

**1. Hard uses `split_group="modulus"` with `separate_ood_splits=True`.** The scoring
splits are `test`, `ood_t`, `ood_n_t`, and in `data/squaring_mod.py` that exact triple
is produced only by `_generate_modulus_grouped_records`. Therefore:

- `_partition_factor_pairs` gives **disjoint train and test modulus pools** — Hard's
  `test` split already requires generalising to **moduli never seen in training**, not
  merely to unseen `x`.
- `ood_t` = training-pool moduli at unseen T; `ood_n_t` = held-out moduli at unseen T.

**This invalidates our Hard proxies as models of h1.** `hp1`/`hp2`/`hp3` were all
generated with `--split_group prompt`, where train and test share moduli. Hard is
strictly harder. Anyone regenerating proxies should use `--split_group modulus
--separate_ood_splits true`.

It also sharpens the coverage result: on Hard a residue-indexed readout cannot work
*at all*, because test moduli never appear in training. The modulus-independent digit
readout (`DigitALU`, same ~6.8k parameters at every N) is the right shape for this
tier; the training closure, not the representation, is what blocks it.

**2. First real H100 calibration** (all previous timings were local, on sm_107):

| quantity | measured |
|---|---|
| startup: import + construct + `.to(device)` + optimizer + first step | **3.9 s** |
| steady state, batch 512, D=128 recurrent stack | **~38.6 ms/step** |
| implied steps in a full 3600s Hard run | **~93,000** |

**Results:** train exact accuracy 0.000 at every logged step; `test` 0.001,
`ood_t` 0.000, `ood_n_t` 0.001; loss 2.22-2.25 against `ln(17) = 2.833`. Digit
marginals learned, nothing more.

**Reading the metrics file:** `_evaluate_depth_profile` prints per-rung results but
never calls the metric recorder, so `one-layer metrics` structurally **cannot** contain
depth rungs. Their absence is not a failure; Max T is visible only on the
leaderboard/status page.

### Corrections to earlier entries in this log

- **`batch_size` 512→32 is 1.2× for the ALU model, not 5.8×.** The DataLoader lever
  evaporates once the model step dominates. The 5.8× figure holds only for the cheap
  dense model it was measured on.
- The model is **kernel-launch bound**: 8× the eval batch costs 5%, so a large
  `eval_batch_size` is nearly free. Discrete eval states cost +42% and buy nothing.
- **e1 and e2 are degenerate for any depth experiment.** λ(323) = lcm(16,18) = **144**,
  so applying the step 4, 16 or 64 times is *the same function*, and 8 and 32 are the
  same function — that ladder has **four** distinct maps, not seven. A controller was
  observed scoring 1.000 at four rungs while applying the step 15 times when asked for
  4. This is finer than the mod-φ(=288) analysis used by two earlier branches and
  supersedes it. Run depth work on m1/hp1/hp3 or a purpose-built non-degenerate λ, and
  state the λ of every modulus used.

**Bottleneck 3, solved (`explore/digit-carry`).** Anchor each digit on the marker that
*terminates its own field*: `d(x)` ends at `[T]`, so `x`'s slots key off the `[T]`
marker position, which does not move when `T` gains a digit. Implemented as a learned
differentiable pointer (`MarkerPointer`) plus a second learned table scoring
cumulative anchor mass to supply the field's opening boundary. Verified 1.000 on
x/N/T at **T=16,32,64** for both fixed and sampled N; the distance-from-end control
scores 0.000 there, exactly as predicted.

**Note this falsified my own brief's premise** that parsing was "worth ~0.25 on its
own". With the parse exact, held-out T=1 is 0.026 — indistinguishable from the broken
control across 3 seeds, both optimiser settings, and the evaluator. **Parsing is
necessary, not sufficient.**

**The coverage ceiling — and its escape (measured, `explore/digit-carry`).** With a
*perfect* representation and the generator's 250 training x, held-out accuracy for a
residue-indexed readout equals `P[x² already seen]`, a closed-form combinatorial
quantity. A **digit-compositional** readout is not subject to it, and this was
measured, not assumed:

| modulus | residue readout | **digit readout** |
|---|---|---|
| 323 (e1) | 1.000 | **1.000** |
| 899 (e2) | 0.614 | **1.000** |
| 2021 | 0.337 | **1.000** |
| 10403 (m1) | 0.031 | **1.000** |
| 12 *unseen* 10/11-bit moduli (e5) | needs a new `Z_N` table per N | **1.000** (4800 operands, one parameter vector) |
| 6 *unseen* 20/24-bit moduli (hp3) | 0.001 / 0.000 (closed form) | **1.000** (1200 operands, same vector) |

The parameter count is identical at every modulus — that is the mechanism. A
closed-form sweep over 7 moduli from 9 to 30 bits shows the digit quantity does not
degrade with N (0.95–0.99) while `P[x² already seen]` reaches exactly 0.000 by 24
bits. **The scale argument that made Medium/Hard look hopeless is removed.** Residue-
indexed readouts (Fourier, softmax over `Z_N`, embedding table, learned permutation)
remain capped, and for those e1 is still the only public dataset where T=1 is
certifiable.

> **Compliance boundary — important.** The ceiling above was measured with
> `probe_alu.py --construct`, which *sets* the digit tables to the truth. That is a
> diagnostic oracle and **is not a legal submission** (rule 7: no hard-coded algorithm
> in the forward pass). A submission may use the same *structure* — digit-indexed
> product/add/subtract tables and a gate — but every tensor must be learned from
> random init. Keep this distinction explicit in any write-up.

### Two protocol corrections (both cost real runs to learn)

1. **Held-out CE is not a progress signal.** A label-smoothing control with zero
   algebraic content moves held-out CE from 7.61 to 2.93 — below the uniform
   `ln(17)=2.833` reference — while rung-1 stays at 1/38. It tracks confidence
   calibration, not correctness. Rank on **train-vs-held-out exact accuracy** and the
   **rung profile** only.
2. **Screen on e5, not e1.** On e1, 96% of the `test` split's operands are training
   operands seen at a different T (all of T∈{1,2,3} draw from the same 250
   non-reserved units), so `test` and `mean_exact_accuracy` on e1/e2 measure
   *T-transfer*, not operand generalisation. e1 rungs are 38 examples — variance floor
   is one example. e5 has 512-example rungs. Keep e1 only for a final MAX_T=1 attempt.

### Screening discipline

**THE HEADLINE METRIC IS `train_exact_hard`** (every inter-step state snapped to
argmax). `train_exact` is fooled by mixtures, and graph shapes that mix *more* score
*higher* on it - ranking on it actively selects for the failure mode. A trained ALU
goes from `train_exact` 0.556 to **0.004** under hard states while the *constructed*
solution survives at 1.000.

**No single metric is safe — report the row, not the cell.** Every metric used this
session has a measured configuration that fools it:

| metric | fooled by | reads | but |
|---|---|---|---|
| `train_exact` | baseline, 12k steps | 0.616 | `train_exact_hard` 0.000 |
| state sharpness | `--tau-final 0.01` | 0.975 | `train_exact_hard` 0.000 |
| `sub_shift` (structure) | target propagation | 0.567 | `train_exact_hard` 0.004 |
| `held_exact` | teacher forcing, 3k | 0.816 | `held_exact_hard` 0.026 |
| `train_exact_hard` alone | short chain | 0.183 | `held_exact_hard` 0.000 |
| `mul_fn` (structure) | a constant map | 1.000 | `mul_gauge` 0.100 |
| algebraic objective | basin hopping, 38x compute | 0.786 -> 0.211 | `add_shift` driven *below* chance, 0.285 -> 0.205 |
| `local_ce` | SOAP on the legal objective | 14/32 replicas below the 0.006 cliff | only **1** actually in basin; one cell reads 5e-05 with `train_exact_hard` 0.120 |
| any leftward `local_ce` shift | the legal objective itself | 3.5 -> 2.3 looks like progress | **random init is 2.08-2.24** — it is regression toward init |

Note two rows especially. **Parameter-level progress and discrete correctness are close
to independent** (target propagation). And in the last row **objective and structure are
anticorrelated** over the range a stronger searcher explores — which closes the "just try
a better optimiser" objection with a measurement rather than an expectation. Target propagation has the best structure score of any legal
procedure and the worst hard correctness in the table. So "per-step signal teaches the
tables" should be read as *it moves the parameters*, not as *it nearly solves it*.

Two screening premises were retracted this session, both from trusting a single
number: "`train_exact` implies `held_exact`" and "the soft channel is a crutch over a
nearly-found discrete solution". Report `train_exact_hard`, `held_exact_hard`, the
structure scores and sharpness together.


Two-stage, validated: parsed-input probes (`probe_step.py`, `probe_sel.py`, ~90s)
**over-state a submission by ~0.25** — use them only to *kill* candidates.
`probe_learnability.py` (real prompts, real submission) called the evaluator to
within one example on four consecutive candidates — believe only that. Do **not** use
evaluator runs as a primary screen while parsing is unfixed: a lever that moved
held-out 0.079 → 0.158 offline showed as a one-example difference in the evaluator,
and four selector variants spanning 0.000–0.263 offline were mutually
indistinguishable in the evaluator.

---

## 2. Branches

All branch from `lab/base`. Worktrees live in `.worktrees/<name>` (gitignored).

| branch | commits | runs | report | verdict |
|---|---|---|---|---|
| `explore/exact-arithmetic` | 15 | 33 | `lab/reports/exact-arithmetic.md` | closed — representation is not the bottleneck |
| `explore/group-rotation` | 30 | 49 | `lab/reports/group-rotation.md` | closed — rotation route falsified; produced the coverage ceiling, the parsing bottleneck, and the screening discipline |
| `explore/tied-recurrence` | 34 | 47 | `lab/reports/tied-recurrence.md` | closed — per-step exactness binds, not propagation; found the gauge-freedom trap |
| `explore/algebraic-closure` | 4 | 47 | `lab/reports/algebraic-closure.md` | closed — semigroup law adds no information at unlabelled operands |
| `explore/grok-optimization` | 5 | 57 | `lab/reports/grok-optimization.md` | closed — no transition at any recipe up to 2e5 steps; Hard's ceiling is below that |
| `explore/digit-carry` | 8 | 9 | `lab/reports/digit-carry.md` | closed — digit readout escapes the coverage ceiling; parsing solved |
| `explore/alu-compose` | 4 | 8 | `lab/reports/alu-compose.md` | closed — endgame measured: constructed pipeline certifies MAX_T=2 Easy / 0 Medium; found P1, P2, P3 |
| `explore/alu-depth` | — | — | `lab/reports/alu-depth.md` | **running** — shorten the ALU chain (owns graph depth/shape) |
| `explore/alu-credit` | — | — | `lab/reports/alu-credit.md` | **running** — training procedure for the transducer (architecture fixed) |
| `explore/depth-controller` | — | — | `lab/reports/depth-controller.md` | **running** — make the controller extrapolate in T (owns P1) |

Each branch also carries `submissions/<branch>/submission.py`. All lint clean; all
score MAX_T = 0. None beats the baseline on the metric.

---

## 3. The two branches that ran to the cutoff

### `explore/grok-optimization` — steps-to-exactness

**Mandate.** Hold architecture roughly fixed and ask whether exact generalisation is
reachable by *any* training recipe, and at what step count. Sweeps: weight decay, LR
and schedule, optimizer (AdamW / Muon / schedule-free / Grokfast / perp-grad /
StableMax), batch size, and a metric-aligned loss. This is the last untested
mechanism — four sibling branches independently concluded the only remaining route is
an inductive bias making the true solution the shortest description.

**Why it matters most.** The number decides strategy: if rung-1 exactness needs ~10k
steps, Hard is winnable and the team should optimise step throughput; if it needs
10M, no architecture search fits in 3600s.

**COMPLETE — 57 runs, all MAX_T = 0. THE NUMBER: steps-to-exactness on rung 1 is
`> 2 × 10⁵` optimizer steps — a lower bound, not an observed transition.** Three
200,000-step runs on e1 (wd = 0.01 / 0.1 / 1.0, bs 128, lr 1e-3, constant, AdamW) gave
rung-1 = 2/38, 1/38, 0/38 — statistically identical to the same recipe at 2,000 steps.

**Which knobs moved it: none.** Weight decay 0→3.0 (3.0 destroys training), lr
3e-4→1e-2, schedule const/cosine/linear, optimizer AdamW/Muon/Schedule-Free, Grokfast
(λ=2,5 at 50k), ⊥Grad, StableMax, ⊥Grad+StableMax, small/orthogonal init,
embedding-norm projection, batch 16→512 (512 ≈ full batch on 600 rows), width 32→256,
loss CE/focal/hardest-token/label-smoothing. **Every one moves rung-1 by less than one
example out of 38.** Seed spread is zero: 50,000 steps × 3 seeds → 0/38, 0/38, 0/38.
Across all 50 successful e1 runs the rung-1 histogram is 0/38 (37×), 1/38 (12×), 2/38
(2×).

**Mechanism — why it is not a pre-grok plateau.** Train exact accuracy hits 1.00 by
~2,000 steps and *holds it for the next 198,000* while held-out never leaves the
floor. A grokking plateau creeps before it jumps; this does not. Two purpose-built
control datasets (N=77, 40 facts, 2-digit; N=1147, 800 facts, 4-digit) fail
identically, ruling out both "too few facts" and "arithmetic too wide". Rung 1 on e1
is a *unary* map with only 288 facts in the universe presented as ≤3 shared decimal
digit tokens — there is no shared-embedding structure for weight decay to reorganise
into the Fourier solution grokking normally finds.

**Per-tier feasibility (throughput assumption stated: host-bound workload, ~0.8M
params, seq ≤10, so H100 ≈ this box within ±2×).** At the improved bs=32 rate
(52 steps/s, quiet machine): Easy ≈ 2,900 steps, Medium ≈ 31,000, **Hard ≈ 187,000 —
below the 2×10⁵ already shown empty.** Conclusion: **rung-1 certification is not
reachable at any tier by a recipe change.**

> **Scope caveat, important.** This was measured on a *dense* architecture that
> **can** memorise, and the mechanism above is a lookup table. `digit-carry`'s
> `DigitALU` cannot memorise (6,817 digit-indexed parameters, no `Z_N` index), so this
> conclusion does **not** automatically transfer to it. Re-test rather than inherit.

**Transferable win — free steps at every tier.** The evaluator's DataLoader uses
`num_workers=2` with **no `persistent_workers`**, and e1's 600 rows at the manifest's
`batch_size=512` is *one batch per epoch* — a worker respawn every step. Measured on a
quiet GPU: **111 ms/step at bs 512 → 19 ms/step at bs 32 (5.8×)**, still 1.5× on
Medium-sized data. Setting `SUBMISSION.batch_size = 128` is free steps everywhere.

**Abandoned at cutoff** (killed deliberately when GPU time ran out; ~2h15m elapsed,
no checkpointing, nothing salvageable): two 5×10⁵-step runs. Report §15 has exact
resume commands. Single next command, with the branch's recorded prediction of
rung-1 = 0/38 or 1/38:
```
TMO=25200 TAG=B-steps lab/grok.sh L_500k_bs16_wd0.1 500000 74 "500k step probe" \
  --batch-size 16 --wd 0.1 --wd-emb 0.1 --lr 0.001
```

### `explore/digit-carry` — COMPLETE (HEAD `ed804e0`), and it changes the target

Owned bottlenecks 2 and 3. **Both resolved**; see the two tables above. Report:
`lab/reports/digit-carry.md` (543 lines), 7 evaluator runs archived.

`DigitALU` is the artifact: a readout in which every learned tensor is indexed by a
digit tuple — a 10×10 product table, a `[digit, addend, carry]` add table, a
`[digit, n_digit, borrow]` subtract table, and a gate — **6,817 parameters, no index
ranging over `Z_N`**. N enters only as input digits.

**What it falsified, including its own mandate's premises:**
1. "Parsing is worth ~0.25 on its own" — false; necessary but not sufficient.
2. "Compositional structure prevents memorisation" — **false, and the sharpest
   result.** A place-shared product table with a *continuous* carry vector memorises
   as fast as anything (train 1.000 / held 0.000 by step 2000). **The state alphabet
   must be small**; a 32-dim carry just re-encodes the value.
3. `DigitALU` does not train (train_exact 0.196, held 0.000). Freezing sub-modules at
   the truth makes it monotonically *worse* (loss 2.3 → 18.7); straight-through and
   identity-init do not help.

Best **evaluator** rung-1: **0/38 on e1**, 4/512 on e5, `MAX_T = 0` — at the floor and
below the 3/38 field best. (An earlier draft reported 1/38 for e1; that was the
offline probe figure, not the evaluator's. The one-example gap is the variance floor
on a 38-example rung, so the honest reading is "indistinguishable from every other
null result" — and it is the fourth consecutive confirmation that
`probe_learnability` tracks the evaluator to within one example.)

All three front ends measured on e5: marker 0.008 / abs 0.006 / rev 0.004 rung-1 — a
one-to-two example spread with controls not consistently ordered, so **the evaluator
cannot resolve the parsing fix**. Parsing is verified by the direct slot-accuracy
measurement above, not by any evaluator score.

**The diagnostic inversion — cheap screening from here on.** `DigitALU` *cannot*
memorise 250 residues in 6,817 digit-indexed parameters, so for this architecture
**train_exact → 1.000 implies held-out → 1.000**. Train accuracy, previously the
signal that told you a lever was useless, is now the whole game. Screen on it.

**The measured optimisation direction** (not a guess): shortening the soft chain from
~280 to ~117 sequential steps (S=3→S=2) took train_exact from **0.20 to 0.78**, with
the constructed ceiling 1.000 in both cases. The `R=11` tied conditional subtractions
are ~80% of the depth and 10 of 11 are no-ops; replacing them with one learned
quotient digit plus a single subtraction is exact and gives ~60 steps.

**Next command on resume:**
```
$VENV lab/probe_alu.py --modulus 323 --slots 3 --reduce 4 --steps 4000
```

Abandoned at cutoff, all confirmatory and none load-bearing (exact commands in report
§8): the 30/32-bit construction, two evaluator cells (`dcp_abs` on e5, `dcp_marker`
on e1), and a second seed on the short-chain result.

---

## 4. How to restart

Everything below is on the shared filesystem and should survive the GPU session.

```bash
cd /home/scratch.arohan_hw/git/one-layer-deeper
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python   # Python 3.13.5, torch 2.12.1+cu130
git worktree list          # the six worktrees should still be present
$VENV -m unittest discover -s tests    # 117 tests, expect OK
```

If `data/generated/` is missing (it is gitignored and not in any commit), regenerate —
this takes a few minutes and must use the venv on PATH:

```bash
PATH=$VENV_DIR:$PATH bash scripts/generate_datasets.sh   # the 10 public datasets
PATH=$VENV_DIR:$PATH bash lab/gen_hard_proxy.sh          # hp1/hp2/hp3 Hard proxies
```

Each worktree needs `data/generated` symlinked, since it is gitignored:

```bash
ln -sfn /home/scratch.arohan_hw/git/one-layer-deeper/data/generated .worktrees/<name>/data/generated
```

Sanity check the harness end to end (should print `MAX_T=0` and a rung table):

```bash
$VENV lab/make_manifest.py --dataset e1 --mode fixed_step --max-steps 20 --name smoke
$VENV lab/run_experiment.py --submission submissions/baseline_adamw/submission.py \
  --manifest lab/manifests/smoke.json --tag smoke --note "post-restart plumbing check"
```

Hardware note: this box is `sm_107`, **not** the H100 the competition scores on.
Kernels JIT from PTX on first use; `run_experiment.py` already points
`CUDA_CACHE_PATH` at scratch so the cache persists. Accuracy at fixed step count and
certified T transfer; absolute wall clock, steps-in-budget and compile payoff do not.

---

## 5. Ranked next actions

1. **Shorten the ALU chain** (`alu-depth`, running). Now justified three independent
   ways: trainability (2.4x shorter took train_exact 0.20 -> 0.78), the eval-budget
   failure (P2), and the step famine (P3). Target is quantitative: the chain must
   shrink until the tier budget affords the steps training needs. The `R=11` ->
   learned-quotient-digit fix is ~4.6x and probably not sufficient alone; stack it
   with parallel-prefix carries and a larger internal radix.
2. **Make the depth controller extrapolate in T** (`depth-controller`, running, owns
   P1). Two untried directions: parameterise the controller so nothing is indexed by
   the digits of T (counted halting with a detector shared with the ALU), and
   supervise at unprovided depths via `step(state, k+1) = step(step(state, k), 1)`.
   Note this is *not* the falsified algebraic-closure idea: there the step map `h` was
   unknown and the semigroup law said nothing about its value; here `h` is fixed and
   the unknown is the iteration count, which consistency across k does pin down.
3. **Find a training procedure that converges in tier-affordable steps**
   (`alu-credit`, running). Convergence *speed* now matters more than final value:
   ~15-150 steps on Easy, and any curriculum must be expressible under the evaluator's
   fixed one-step-per-batch loop.
4. **Every candidate must halt early.** A fixed-depth readout fails the run outright
   (status `failed`, no leaderboard row), which is strictly worse than scoring 0. Set
   a large `eval_batch_size` (kernel-launch bound, 8x costs 5%); do not use discrete
   eval states (+42%, no benefit).
5. **Consider one early Hard submission as ranking insurance.** `service/db.py:585-618`
   orders by `max_certified_time_steps DESC, ood_n_... DESC, created_at ASC` over
   `status='succeeded'` runs only. Every branch is MAX_T=0, so `created_at` is
   plausibly the live tiebreaker. Use a known-good cheap model — **not** an ALU
   candidate, which can time out and fail. **User's call; Hard is 1/day.** Deadline
   2026-08-31 22:00 PT.

## 6. Honest summary

Six hypothesis families, ~240 experiments, **zero certified rungs — every branch
scored MAX_T = 0**, and the best rung-1 anywhere is 3/38. No submission here beats the
official baseline on the metric.

The value is the map. A vague "1-5% accuracy plateau" became four separately
diagnosed bottlenecks: iteration in T (solved), prompt parsing (solved), readout
representation (solved, and measured to be uncapped by modulus size), and training a
small discrete transducer (open, and now the only one left). Two branches concluded
MAX_T ≥ 1 was unreachable by architecture search; `digit-carry` then removed the
ceiling those conclusions rested on, so they should be re-tested rather than
inherited — its architecture cannot memorise, which inverts the diagnostic those
branches used.

The honest position: the path is narrower and better lit than at the start, and it is
still unproven. Nothing here has certified a single rung.
