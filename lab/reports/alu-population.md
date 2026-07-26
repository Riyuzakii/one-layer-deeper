# alu-population — converting a 1-in-N basin into ~1 inside a single evaluator run

**Branch** `explore/alu-population`, from `explore/alu-credit`.
**Mandate** `alu-credit` Stage 1 reaches `train_exact_hard` 0.951 /
`held_exact_hard` 0.946 on the 39-step `tree:quotient` graph at m1 scale — but
in only 1 of 7 untied seeds. The evaluator runs **one** seed (`seeds: [74]`).
Make the restarts happen *inside* the run.
**Tooling** `lab/probe_pop.py` (population ALU + differentiable mixture),
`lab/pop_time.py` (contention-robust ms/step vs replica count),
`lab/pop_sweep.sh`, `lab/pop_table.py`; runs in `lab/pop_runs.jsonl`, per-run
logs in `lab/logs/`.

---

## 0. THE FRAME — read this before any number

The signal that reaches the 0.951 basin **at all** is teacher forcing on a tape
recorded from a *constructed* copy of the model. That is a lab diagnostic and
**illegal in a submission** (BRIEF §4 rules 2 and 7). Everything in this report
that carries a `train_exact_hard` above the 0.00–0.02 floor is therefore
**conditional**: it says that *if* a legal per-step signal is found
(`explore/alu-relational` is searching for one), the seed-fragility of the
resulting optimisation is **not** the blocker.

**A population result is not a scoring candidate.** Every row below is labelled
**LEGAL** or **DIAGNOSTIC**. No row labelled DIAGNOSTIC can appear in a
submission, no matter how good the number is.

The two components are separable, and only one of them is illegal:

| component | status | why |
|---|---|---|
| replica dimension + one `optimizer.step()` for all of them | **LEGAL** | the contract constrains the loop, not the model's internal width |
| learned mixture / argmax over replicas, trained by the ordinary task loss | **LEGAL** | unbroken gradient path; no labels at eval |
| teacher-forced per-op CE that gets any replica into the basin | **DIAGNOSTIC** | the tape is computed arithmetic (rule 2), and needs the true register trace (rule 7) |

---

## 1. The machinery, and that it is the same hypothesis class

`lab/probe_pop.py` gives `probe_alu_depth.DigitALU` (`mul_mode=tree`,
`reduce_mode=quotient`, `scan_mode=serial`, untied — see §1.1) a leading
replica axis. Every state tensor is `(P, n, ...)`; every table read carries a
`p` index (`einsum("pnu,pnv,pnc,puvco->pno", ...)`), so replica `p`'s forward
touches only replica `p`'s parameters and the gradients are disjoint. All
replicas see the same minibatch.

Verification that the graph is unchanged:

```
$V lab/probe_pop.py --construct --pop 4 --eval-n 512 --held-x 512 --eval-chunk 128
  -> CONSTRUCTED train_exact_hard=1.000 held_exact_hard=1.000 mix=1.000
```

— the constructed ceiling is 1.000 **soft and hard**, at P=4, for both the naive
and the reassociated quotient reduction (§4.1). Parameter count is
**6,820 per replica**, matching `alu-credit`.

### 1.1 Ties are off, per the coordinator's mid-flight correction

Weight tying is harmful from random init (`local_ce` 0.0050 → 0.1187 on the one
seed that succeeds untied). All runs here are **untied**; `--tie/--tie-sub`
exist in the probe but were not used for any reported row.

### 1.2 State ceiling — verified, not estimated

`benchmark.assert_model_state` is called in every run against the real
`ModelSpec(maximum_model_state_elements=500_000_000)`:

| P | model state elements | % of ceiling |
|---|---|---|
| 1 | 6,821 | 0.0014% |
| 4 | 27,284 | 0.0055% |
| 16 | 109,136 | 0.0218% |
| 64 | 436,544 | 0.0873% |
| 256 | 1,746,176 | 0.349% |
| **73,313** (ceiling) | 500,000,000 | 100% |

**The state ceiling is not the binding constraint on population size — compute
is** (§4). At P=256 the model uses one third of one percent of the budget.

---

## 2. Cost: ms/step vs replica count, and where replicas stop being free

Measured with `lab/pop_time.py`, which times a **complete training step**
(reference tape under `no_grad` + forced forward + `backward()` + `AdamW.step()`)
for every replica count **inside one process**, interleaved over four rounds,
and reports the **minimum**. The GPU is shared with `explore/alu-relational`,
so the minimum over interleaved rounds is the only contention-robust statistic
available; the four rounds agree to <1% and two independent sweeps 40 minutes
apart agree on the *ratios* to within 0.1x.

**Batch 512, `tree:quotient`, m1 scale, teacher-forced step:**

| P | ms/step | vs P=1 | ms per replica | peak GiB | model state |
|---|---|---|---|---|---|
| 1 | 94.0 | 1.00x | 94.0 | 0.20 | 6,821 |
| 2 | 102.7 | 1.09x | 51.3 | 0.33 | 13,642 |
| 4 | 108.0 | **1.15x** | 27.0 | 0.59 | 27,284 |
| 8 | 121.4 | **1.29x** | 15.2 | 1.12 | 54,568 |
| 16 | 145.8 | **1.55x** | 9.1 | 2.14 | 109,136 |
| 32 | 191.1 | 2.03x | 6.0 | 4.25 | 218,272 |
| 64 | 284.1 | 3.02x | 4.4 | 8.30 | 436,544 |
| 128 | 483.9 | 5.15x | 3.8 | 16.50 | 873,088 |
| 256 | 873.1 | 9.28x | 3.4 | 32.77 | 1,746,176 |

**Where it stops being free: P ≈ 16.** Up to 8 replicas the model is still
kernel-launch bound and 8 independent runs cost **29% more wall clock than
one**. At 16 the premium is 55%; at 32 it is 2.0x (you pay one extra step per
step); past 64 the marginal cost per replica is flat at ~3.4 ms, i.e. the
population has become FLOP/bandwidth bound and scaling is linear.

The decision-relevant form: at a *fixed* wall-clock budget, P=8 costs you
**22%** of your optimizer steps and buys **8** independent basin draws; P=32
costs you **half** your steps and buys 32.

The sibling's "kernel-launch bound" finding is confirmed and is exactly what
makes this work — 27x more arithmetic (P=1 → 32) costs 2.0x.

### 2.1 The straightforward implementation is 1.6x slower and 2.9x fatter

The quotient reduction broadcasts each of the M=12 candidate multiples of N to
every example, so the naive scan runs over `b*M` = 6,144 rows. But the
subtrahend digit does not depend on the example, so it can be contracted into
`Tsub` once per forward (`Tm[p,m,w,u,c,o] = sum_v mults*Tsub`, ~17k elements per
replica) instead of being materialised per row. This is an algebraic
reassociation, not a different graph — both give the constructed ceiling 1.000.

| P | naive ms | reassociated ms | naive GiB | reassociated GiB |
|---|---|---|---|---|
| 1 | 92.4 | 115.8 | 0.45 | 0.20 |
| 4 | 134.9 | 122.2 | 1.56 | 0.59 |
| 16 | 250.5 | 164.7 | 6.10 | 2.14 |
| 32 | 343.0 | 211.1 | 12.07 | 4.25 |
| 64 | 581.2 | 360.2 | 23.91 | 8.30 |

(This block was taken under heavier contention than the table above; compare
columns, not rows across tables.) At P=1 the reassociation is a small *loss*
(one more kernel launch, nothing to amortise); from P=4 up it wins, and by P=64
it is 1.6x faster on 2.9x less memory. **Anyone building a population should do
this**; without it P=64 is 6.3x a single model instead of 3.0x.

### 2.2 The selection forward, and the eval-time cost

The mixture needs a *free-running* forward (the teacher-forced pass has its
end-of-chain output replaced by the tape, so it carries no signal for `alpha`).
Interleaved in one process:

| P | teacher-forced step | + mixture fwd, replicas detached | + mixture fwd, full backward |
|---|---|---|---|
| 1 | 48.5 | 93.1 (1.92x) | 175.8 (3.6x) |
| 16 | 146.1 | 180.9 (1.24x) | 279.7 (1.91x) |
| 64 | 275.1 | 362.1 (**1.32x**) | 569.3 (2.07x) |

So the selector costs **+32% at P=64** if the replica outputs are detached
(forward only, no second backward) and **+107%** if not. In a legal submission
there is no teacher-forced pass, so this cost disappears — the free-running
forward *is* the training forward.

**Eval-time cost does not have to scale with P at all.** If the submission
commits to the argmax replica, `forward()` at eval can index that one replica's
tables and run at **P=1 cost** — the population is a training-time device only.
Only the *blend* requires all P replicas at eval. Given `alu-compose` P2 (a
fixed-depth readout made the run *fail*, which is below the leaderboard floor),
this is a second, independent reason to prefer the argmax over the mixture.

---

## 3. The single-model control — the 1-in-7 basin, reproduced in this code

**DIAGNOSTIC** (teacher forcing). Seven seeds, P=1, 1,200 optimizer steps
(the `tree:quotient` Hard budget), m1 scale, untied, lr 3e-2, batch 512:

```
bash lab/pop_sweep.sh lab/jobs_pop_ctl.txt 3
```

| seed | `local_ce` | `train_exact_hard` | `held_exact_hard` | in basin? |
|---|---|---|---|---|
| 3 | **0.0000** | **1.000** | **1.000** | **yes** |
| 0 | 0.0241 | 0.468 | 0.469 | no |
| 1 | 0.0285 | 0.077 | 0.068 | no |
| 5 | 0.0457 | 0.000 | 0.002 | no |
| 4 | 0.0500 | 0.000 | 0.000 | no |
| 2 | 0.0891 | 0.000 | 0.000 | no |
| 6 | (see §3.1) | | | |

**1 of 7 — the coordinator's rate, reproduced independently.** Two things are
worth recording beyond the rate:

1. **The successful seed is exact, not 0.95.** `train_exact_hard` = 1.000 and
   `held_exact_hard` = 1.000, agreeing to 0.000 on a 1,024-example held-out
   cohort at m1 scale. `alu-credit`'s best was 0.951/0.946. The difference is
   the initialisation of the 5-parameter quotient scorer (`randn*0.5` here vs
   `nn.Linear` default there) and the reassociated reduction; I did not chase
   it further, but it means the ceiling of this diagnostic is **1.000, not
   0.951**.
2. **It converges in under 600 steps.** Seed 3 reads `train_exact_hard` 1.000
   and `local_ce` 0.0000 at step 600 and holds it. That is inside the Medium
   budget (~170 steps on this graph is the coordinator's figure — so still 3.5x
   short) and comfortably inside Hard's ~1,300.

**`local_ce` is the only usable per-replica basin indicator, and the structure
scores are worthless here.** `mul_fn`, `mul_gauge`, `add_shift` and `sub_shift`
read **1.000 for all six seeds**, including the four at `train_exact_hard`
0.000. This is `alu-credit`'s "parameter-level progress and discrete
correctness are close to independent" in its sharpest form: every seed learns a
relabelled multiplication table and a correct shift structure, and only one of
them composes. Rank on `local_ce`; do not rank on structure.

---

## 4. THE RESULT — basin hits vs replica count

**DIAGNOSTIC** (teacher forcing). All runs: 1,200 optimizer steps, m1 scale
(N=10403, S=5, 8,000 train operands / 1,024 held-out), untied, `tree:quotient`,
lr 3e-2, batch 512, mixture head with replicas detached (`--sel 1
--sel-detach`).

```
bash lab/pop_sweep.sh lab/jobs_pop_ctl.txt 3      # P=1, seeds 0-6
bash lab/pop_sweep.sh lab/jobs_pop_basin.txt 3    # P=8/32/64
```

### 4.1 Run-level: does the run contain a solution at all?

| P | runs | runs containing a basin hit | wall-clock premium per run |
|---|---|---|---|
| **1** | 7 | **1 / 7** (14%) | 1.00x |
| **8** | 3 | **3 / 3** (100%) | 1.29x |
| **32** | 2 | **2 / 2** (100%) | 2.03x |
| **64** | 1 | **1 / 1** (100%) | 3.02x |

**Six of six population runs contained a solved replica; one of seven single
models did.** At P=8 the expected number of hits per run is ~1.3 and the
observed run-level success is 3/3; at P=32 and P=64 every replica-level
distribution has 10–16 solved members, so a miss would be a ~10-sigma event.

### 4.2 Replica-level: are replicas inside one run as independent as seeds?

Fraction of replicas reaching the basin (`local_ce` < 0.006, equivalently
`train_exact_hard` >= 0.95):

| P | runs | replicas | in basin | rate | 95% CI |
|---|---|---|---|---|---|
| 1 | 7 | 7 | 1 | 0.14 | 0.00–0.58 |
| 8 | 3 | 24 | 4 | 0.17 | 0.05–0.37 |
| 32 | 2 | 64 | 21 | 0.33 | 0.22–0.46 |
| 64 | 1 | 64 | 16 | 0.25 | 0.15–0.37 |
| **pooled** | 13 | **159** | **42** | **0.26** | 0.20–0.34 |

**Yes — replicas within one run behave like independent seeds.** All four rates
are mutually compatible; the pooled per-replica rate is 0.26, and the P=1 rate
of 1/7 is inside its own interval and inside the pooled one. The replicas share
the minibatch *sequence* and differ only in initialisation, and that is enough:
**the basin is selected by initialisation, not by data order.** That is the
load-bearing fact — it is what makes a population inside one run equivalent to
N restarts across runs, which is the thing the evaluator's single seed
otherwise forbids.

The run-level success probability is therefore `1 - (1 - 0.26)^P`:

| P | predicted | observed |
|---|---|---|
| 1 | 0.26 | 1/7 |
| 4 | 0.70 | — |
| 8 | **0.91** | 3/3 |
| 16 | 0.99 | — |
| 32 | **0.9999** | 2/2 |
| 64 | 1 - 1e-8 | 1/1 |

### 4.3 `local_ce` is quantised, and the cliff is exactly where the coordinator put it

Every replica in all six population runs lands on one of four values. Pooling
159 replicas:

| `local_ce` | `train_exact_hard` | n |
|---|---|---|
| 0.0000 | **1.000** | 34 |
| 0.0050 | **0.953** | 8 |
| 0.0089–0.0100 | 0.10 – 0.23 | 11 |
| >= 0.014 | 0.000 – 0.070 | 106 |

The gap between `local_ce` 0.0050 (0.953) and 0.0089 (0.206) is the whole
result: **there is no continuum.** This reproduces `alu-credit`'s
0.0050 -> 0.951 / 0.0076 -> 0.202 exactly and tightens it. `(1-eps)^50` with
eps = 0.001 gives 0.951 and with eps = 0.003 gives 0.86 — the composition over
~50 ops turns a factor-2 difference in per-op error into all-or-nothing.

**Use `local_ce` and nothing else as the per-replica basin indicator.** The
gauge-invariant structure scores (`mul_fn`, `mul_gauge`, `add_shift`,
`sub_shift`) read 1.000 for essentially every replica, solved or not.

---

## 5. Does differentiable selection recover the good replica?

**Yes, completely, and by three different selectors.** In every one of the six
population runs:

| run | P | best replica | **mixture** | **argmax replica** | CE-argmin replica | held (argmax) | rank of argmax replica |
|---|---|---|---|---|---|---|---|
| `b8_s0` | 8 | 1.000 | **1.000** | **1.000** | 1.000 | 1.000 | 0 |
| `b8_s1` | 8 | 1.000 | **1.000** | **1.000** | 1.000 | 1.000 | 0 |
| `b8_s2` | 8 | 0.953 | **0.953** | **0.953** | 0.953 | 0.946 | 0 |
| `b32_s0` | 32 | 1.000 | **1.000** | **1.000** | 1.000 | 1.000 | 2 |
| `b32_s1` | 32 | 1.000 | **1.000** | **1.000** | 1.000 | 1.000 | 1 |
| `b64_s0` | 64 | 1.000 | **1.000** | **1.000** | 1.000 | 1.000 | 11 |

`alpha` is P scalars trained by the **ordinary end-of-chain task loss** on the
training labels — no per-step signal, no held-out data, no labels at eval.
The gradient path from the loss to every replica's parameters is unbroken. The
selector is **LEGAL**; only the thing that fills the population with a solved
replica is not.

Three separate points, because they do not all point the same way:

1. **The mixture does not dilute a correct replica — because it stops being a
   mixture.** `alpha` concentrates fast: `wmax` goes 0.016 -> 0.229 -> 0.998 ->
   1.000 over 1,200 steps at P=64 (0.032 -> 0.746 -> 0.990 at P=32). By the end
   the "blend" *is* the argmax replica, which is why the two columns agree
   everywhere.
2. **But there is a window where the blend is better than the commit, and it is
   the opposite of `depth-controller`'s finding.** At P=64, step 400:
   `best` 1.000, **`mix` 0.999, `argmax` 0.943** — `alpha` had not yet
   concentrated (`wmax` 0.229) and the argmax pointed at a `local_ce`=0.005
   replica rather than a `local_ce`=0.000 one. The blend of 15 solved replicas
   was better than one arbitrary solved replica. So: **at a tight step budget,
   report the blend; at a comfortable one, commit.** The mode is only safe once
   `wmax` says it is. This does not contradict `depth-controller` — there the
   blend mixed *distinguishable decisions*, here it mixes near-identical correct
   ones — but it does mean "always commit to the mode" is not a general rule.
3. **A cheaper selector works from step 1: pick the replica with the lowest
   per-replica training CE.** `ce_argmin` in the table is exactly that, and it
   is 1.000 at step 400 at P=64 when the argmax is 0.943. It needs no learned
   parameter and cannot get stuck in a rich-get-richer lock-in. It is legal in a
   submission (`training_loss` receives the labels and can accumulate a running
   per-replica CE in a **non-persistent** buffer, which does not count against
   the state ceiling), but it commits by an argmin over a buffer rather than by
   a trained parameter, so I would ship `alpha` and keep this as a fallback.

**Eval cost.** Because the argmax replica reproduces the population's best
result exactly, a submission can slice that replica's tables at eval and run
the readout at **P=1 cost**. The population is a training-time device. Given
`alu-compose` P2 — a model that runs to a fixed depth throws `TimeoutError` and
the run status becomes *failed*, which is strictly below a leaderboard score of
0 — a training-time-only cost is the difference between "affordable" and "does
not fit".

---

## 9. Compliance

* **No file under `data/generated/` was read, printed, sampled or summarised.**
  Every operand in this branch is self-generated from `--modulus` by
  enumerating the units of a modulus chosen from the generator source; nothing
  in `lab/probe_pop.py` opens a dataset.
* **Nothing was submitted to the hosted service.** No `one-layer` invocation,
  no network call.
* Every teacher-forced row is labelled **DIAGNOSTIC** and carries the reason
  (the tape is recorded from a `--construct`ed copy of the model; rules 2, 7).
* **The reinitialise-losers optimizer is flagged, not shipped** — see §7.
* `benchmark.assert_model_state` is called against the real
  `ModelSpec(maximum_model_state_elements=500_000_000)` in every run of
  `lab/probe_pop.py`, and the count is printed (§1.2).
* Negative results are all here, including the ones that contradict the
  premise of my own brief.
* No evaluator cells were run (`lab/archive.jsonl` is untouched by this
  branch): there is no legal end-to-end candidate to run, and running the
  evaluator on a DIAGNOSTIC model would produce a number that could be
  mistaken for a score.

---

## 10. Reproduction

```bash
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python

# the ceiling of the population graph -- must print 1.000 / 1.000 / 1.000
$V lab/probe_pop.py --construct --pop 4 --eval-n 512 --held-x 512 --eval-chunk 128

# cost vs replica count (one process, interleaved, contention-robust minimum)
$V lab/pop_time.py --pops 1,2,4,8,16,32,64,128,256 --rounds 4 --iters 10
$V lab/pop_time.py --pops 1,4,16,32,64 --rounds 3 --iters 8 --slow-quot
$V lab/pop_time.py --pops 1,4,16,32,64,128 --rounds 3 --iters 8 --sel

# the single-model control: 7 seeds, 1 basin hit          [DIAGNOSTIC]
bash lab/pop_sweep.sh lab/jobs_pop_ctl.txt 3

# the population, P = 8 / 32 / 64                          [DIAGNOSTIC]
bash lab/pop_sweep.sh lab/jobs_pop_basin.txt 3

# the legal control (no teacher forcing at all)            [LEGAL]
bash lab/pop_sweep.sh lab/jobs_pop_legal.txt 2

# cheaper-than-a-population alternatives, all measured at P=32 so that ONE run
# is 32 basin draws                                        [DIAGNOSTIC]
bash lab/pop_sweep.sh lab/jobs_pop_var.txt 3

# selection variants                                       [DIAGNOSTIC]
bash lab/pop_sweep.sh lab/jobs_pop_sel.txt 3

# render the tables
$V lab/pop_table.py lab/pop_runs.jsonl
```

Job files: `lab/jobs_pop_{ctl,basin,legal,var,sel}.txt`. One JSON line per run
in `lab/pop_runs.jsonl` with the full argv, the per-replica accuracy and
`local_ce` vectors, the mixture / argmax / CE-argmin results, `n_basin`,
`ms_per_step` and peak memory. Per-run logs in `lab/logs/<tag>.log`.
