# Resumption log — exploration session, 2026-07-25

Written at the end of a GPU session with work still in flight. Read this plus
`lab/BRIEF.md` and you can restart everything without reconstructing context.

Nothing was merged to `main`. Nothing was submitted to the hosted service. No
dataset file under `data/generated/` was ever read.

---

## 1. State of the world

**The metric changed on 2026-07-24** (commit `79f0a09`). Hard ranks by **Max T** —
the largest `T ∈ {1,2,4,8,16,32,64}` whose rung and every lower rung are **100%
exact** — then by OOD-N Max T, then by earliest submission. No partial credit.
`mean_exact_accuracy` is a diagnostic with no ranking value. `lab/findings.md` from
the *previous* session optimises that obsolete metric; read it for what was ruled
out, not for what to do.

**~230 experiments across six branches. Every single one scored MAX_T = 0.** Best
rung-1 anywhere is 3/38 on e1, against a trivial-predictor floor of 1/38.

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
| 1 | iteration / depth in T | **solved** — stop spending runs |
| 2 | per-step arithmetic on unseen operands | representation **solved and uncapped** (digit readout, above); *training* it is now the open problem |
| 3 | prompt → digit-slot parsing | **solved**, verified at multi-digit T |
| 4 | **optimising a small discrete transducer** | **the one remaining bottleneck** — and no longer capped by anything |

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

1. **Make `DigitALU` train.** This is now the entire problem, and it is no longer
   capped by representation, scale, or parsing. Start from the measured direction:
   shorten the soft chain (S=3→S=2 already took train_exact 0.20→0.78), then replace
   the `R=11` tied conditional subtractions with one learned quotient digit plus a
   single subtraction (~60 steps). Screen on **train_exact**, which for this
   architecture implies held-out. Resume command in §3.
2. **Keep the state alphabet small.** The falsification that matters: a continuous
   carry vector re-encodes the value and restores memorisation. Any variant must keep
   the inter-step state discrete or near-discrete.
3. **Set `SUBMISSION.batch_size = 128`.** Free steps at every tier (5.8x on Easy,
   1.5x on Medium) from the evaluator's non-persistent DataLoader workers. Costs
   nothing, applies to every candidate.
4. **Then combine**: weight-tied recurrent step (bottleneck 1, solved) + trained
   digit readout (2) + marker-relative slots (3). Target **MAX_T = 1 on e1** — rung 1
   at 38/38, against a field best of 3/38 — then check e5/m1, which the digit readout
   makes legitimate targets for the first time.
5. **Consider one early Hard submission as ranking insurance.** `service/db.py:585-618`
   orders the leaderboard by `max_certified_time_steps DESC, ood_n_... DESC,
   created_at ASC` over `status='succeeded'` runs only. Every branch is MAX_T=0, so
   `created_at` is plausibly the live tiebreaker across the field. A *successful*
   MAX_T=0 entry submitted early outranks an identical one submitted later.
   `submissions/baseline_adamw/submission.py` is a known-good minimal entry, and the
   measured eval margin is ~5× on Easy and ~200× on Hard, so it cannot blow the
   deadline. **This is the user's call — Hard attempts are 1/day and belong to them.**
   Deadline is 2026-08-31 22:00 PT.

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
