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
| 2 | per-step arithmetic on unseen operands | open, **capped by coverage** unless the readout is digit-compositional |
| 3 | prompt → digit-slot parsing | open, worth ~0.25, previously unrecognised |

**The coverage ceiling.** With a *perfect* representation and the generator's 250
training x, held-out accuracy equals `P[x² already seen]` — a closed-form
combinatorial quantity, verified against measurement at three moduli:

| dataset | modulus | oracle held-out |
|---|---|---|
| e1 | 323 | 1.000 |
| e2 | 899 | 0.614 |
| — | 2021 | 0.337 |
| m1 | 10403 | 0.031 |

This caps **any readout that is an arbitrary function of the residue** (Fourier
readout, softmax over `Z_N`, embedding table, learned permutation), so **e1 is the
only public dataset where such a readout can certify even T=1**. The only escape is a
readout compositional in the digits, with carries.

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
| `explore/grok-optimization` | 3+ | 101 | `lab/reports/grok-optimization.md` | **IN FLIGHT** |
| `explore/digit-carry` | 5+ | 50 | `lab/reports/digit-carry.md` | **IN FLIGHT** |

Each branch also carries `submissions/<branch>/submission.py`. All lint clean; all
score MAX_T = 0. None beats the baseline on the metric.

---

## 3. What was running when the session ended

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

**Committed state at cutoff.** Its last commit message reads *"no transition at any
recipe or step budget up to 2e5"* — treat as the branch's provisional headline, not a
confirmed final result; the report and archive had uncommitted edits in flight.

**In flight at cutoff** (from `lab/manifests/`, all fixed-step unless noted):
```
lab/run_experiment.py --submission submissions/grok-optimization/L_500k_bs16_wd0.1/submission.py \
  --manifest lab/manifests/lab_e1_fs500000_s74.json
lab/run_experiment.py --submission submissions/grok-optimization/L_500k_bs16_wd1.0/submission.py \
  --manifest lab/manifests/lab_e1_fs500000_s74.json
lab/run_experiment.py --submission submissions/grok-optimization/wc_bs32/submission.py \
  --manifest lab/manifests/lab_e1_wc60_s74.json          # wall-clock throughput check
```
The two 500k-step runs will not have finished. **Resume by relaunching them**; they
are the tail of the steps-to-exactness curve.

### `explore/digit-carry` — digit-compositional readout + prompt parsing

**Mandate.** Owns bottlenecks 2 and 3. Highest-value single experiment: build the
oracle-equivalent for a **digit-compositional** readout and measure held-out accuracy
at N = 323 / 899 / 2021 / 10403. If it stays high where the residue readout collapses
to 0.031, there is a path at Medium/Hard scale; if it collapses too, the task is
unlearnable from the evaluator's data above e1 — a far stronger negative than
anything measured so far.

**Committed state at cutoff.** `dc87b01 WIP: all digit-carry probes, submissions and
report draft (time-limit checkpoint)`. Earlier commit `609903d` states *"digit readout
is modulus-transferable; parser alone does not move held-out"* — provisional, see the
branch report for the numbers and caveats.

**In flight at cutoff:**
```
lab/run_experiment.py --submission submissions/digit-carry/dcp_abs/submission.py \
  --manifest lab/manifests/lab_e5_fs2000_s74.json
lab/run_experiment.py --submission submissions/digit-carry/dcp_marker/submission.py \
  --manifest lab/manifests/lab_e1_fs2000_s74.json
```

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

1. **Finish the digit-readout coverage-ceiling measurement** (`digit-carry`). One
   number at four moduli; decides whether anything above e1 is reachable.
2. **Finish the steps-to-exactness curve** (`grok-optimization`). Relaunch the two
   500k-step runs. Even a lower bound is actionable.
3. **Solve prompt→digit-slot parsing properly.** Worth ~0.25 and blocks everything
   downstream from registering in the evaluator. Must be verified at multi-digit T —
   distance-from-the-end indexing breaks on exactly the T=16/32/64 rungs.
4. **Combine**: a weight-tied recurrent step (bottleneck 1, solved) with a
   digit-compositional readout (2) reading place-valued slots (3). Target **MAX_T = 1
   on e1**, i.e. rung 1 at 38/38, against a field best of 3/38.
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

Six hypothesis families, ~230 experiments, zero certified rungs. The session's value
is the map, not a score: it converted a vague "1-5% accuracy plateau" into three
separately-diagnosed bottlenecks, one of which is solved, one of which has a
closed-form ceiling that rules out most of the search space, and one of which was
previously invisible. Two independent branches concluded that MAX_T ≥ 1 is probably
not reachable by architecture or loss search in this budget. That conclusion is worth
testing against the two in-flight results before accepting it.
