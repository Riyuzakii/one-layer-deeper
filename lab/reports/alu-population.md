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

## HEADLINE

| | | status |
|---|---|---|
| **A population of 8 replicas turns a 1-in-7 basin into 6-of-6 runs containing a solution** | run-level: P=1 **1/7**, P=8 **3/3**, P=32 **2/2**, P=64 **1/1** | LEGAL machinery / DIAGNOSTIC signal |
| **It costs 29% of wall clock at P=8**, 2.0x at P=32 | replicas are nearly free because the model is kernel-launch bound | LEGAL |
| **Differentiable selection recovers the good replica every time** | mixture = argmax = CE-argmin = best replica, in all 6 runs, held-out too | **LEGAL** |
| **Replicas inside one run are as independent as separate seeds** | per-replica rate 0.26 over 159 replicas, flat in P | — |
| **The state ceiling is not the constraint** | P=256 uses 0.349% of 5e8; the ceiling is P = 73,313 | — |
| **Cheaper alternatives do not work** | batch, LR, LR schedule all flat; weight averaging is **catastrophic** (0.001) | — |
| **A population does NOT rescue the legal objective** | `--tf 0`: 0/64 replicas, `local_ce` 3.0-4.1 vs a cliff at 0.006 | **LEGAL, and null** |
| **`alu-relational`'s closure survives 32-64 draws per configuration** | 16 configs, **544 replicas, 0 below `local_ce` 1.0**; best 2.304 vs its 2.343 at P=1 | **LEGAL, and null** |
| **The one legal term that shifts the distribution (`--assoc`) does it by collapsing the map** | best-`local_ce` replica predicts 3-20% distinct answers; `--cancel` does not prevent it | — |

**Read the last three rows with the first.** The 1.000s in this report are
reached with teacher forcing, which is a lab diagnostic and illegal in a
submission. This branch shows that **if** a legal per-step signal is found, the
1-in-7 basin is not a blocker — and, having built the instrument, that no legal
signal currently known produces one, at 32-64 draws per configuration. It does
not supply one, and no submission was produced.

**Three findings that correct or qualify sibling results, flagged for
`RESUME.md`:**

1. **The Stage-1 ceiling is `train_exact_hard` 1.000 / `held_exact_hard` 1.000,
   not 0.951**, and it converges by **step 600** (§3). The project has been
   quoting a number that is both lower and slower than the truth.
2. **"Always commit to the mode" is not general** (§5). `depth-controller`
   measured that committing took its Medium result 0 -> 4; at P=64 step 400 the
   *blend* beat the commit 0.999 vs 0.943. The accurate rule is: commit once the
   selector has concentrated, and check `wmax` before assuming it has. Both
   results should be reported together.
3. **Parameter-space averaging or ensembling is meaningless for every
   learned-table architecture in this repo** (§7.3) — not merely a null in one
   sweep. `Tmul`'s output code is a **gauge**; two replicas that have both
   solved the task in general solve it in different gauges, and the average of
   two one-hot tables in different gauges is uniform. Measured: 0.001 from a
   population containing a 1.000 replica. Only output-space mixing or selection
   is coherent.

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
| 6 | 0.0438 | 0.000 | 0.000 | no |
| 5 | 0.0457 | 0.000 | 0.002 | no |
| 4 | 0.0500 | 0.000 | 0.000 | no |
| 2 | 0.0891 | 0.000 | 0.000 | no |

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
read **1.000 for all seven seeds**, including the five at `train_exact_hard`
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

### 5.1 Every selection variant I tried is the same — the selector is not delicate

P=32, seed 0, 1,200 steps, against the `b32_s0` baseline (mixture 1.000,
argmax 1.000, 11/32 in basin):

| selection variant | mixture | argmax | CE-argmin | replicas in basin |
|---|---|---|---|---|
| baseline (soft mixture, `alpha` lr = 3e-2, replicas detached) | 1.000 | 1.000 | 1.000 | 11/32 |
| **straight-through commit to the mode** (`--sel-hard`) | 1.000 | 1.000 | 1.000 | 11/32 |
| `sel_tau` annealed 5 -> 0.2 | 1.000 | 1.000 | 1.000 | 11/32 |
| `alpha` lr 3e-2 -> 3e-1 | 1.000 | 1.000 | 1.000 | 11/32 |
| mixture loss **not** detached (backprops into the replicas) | 1.000 | 1.000 | 1.000 | 11/32 (but `local_ce`<0.006 count 11 -> 8) |

Nothing is needed. I expected the rich-get-richer failure — `alpha`
concentrating on a bad replica before a good one emerges — and it does not
happen, because the good replicas separate in likelihood *long* before `alpha`
saturates. **No sharpening schedule, no annealing, no learning-rate tuning on
the selector.**

One negative worth recording: letting the end-of-chain mixture loss backprop
into the replicas (rather than detaching) leaves the basin count unchanged but
drops the number of replicas at `local_ce` < 0.006 from 11 to 8. That is
consistent with `alu-credit`'s finding that the end-of-chain objective pulls
toward the degenerate solution. **Detach the replicas in the selector loss**
when a per-step signal is doing the real training. In a legal submission there
is no per-step signal, so the question does not arise.

**Eval cost.** Because the argmax replica reproduces the population's best
result exactly, a submission can slice that replica's tables at eval and run
the readout at **P=1 cost**. The population is a training-time device. Given
`alu-compose` P2 — a model that runs to a fixed depth throws `TimeoutError` and
the run status becomes *failed*, which is strictly below a leaderboard score of
0 — a training-time-only cost is the difference between "affordable" and "does
not fit".

---

## 6. The LEGAL control — a population does not rescue the legal objective

**LEGAL.** Same graph, same scale, same 1,200 steps, same mixture head, and
**no teacher forcing at all** (`--tf 0`): each replica gets its own
end-of-chain cross-entropy plus the mixture CE, which is everything a
submission can express under the evaluator's fixed loop.

| run | P | `local_ce` (best replica) | `train_exact_hard` best | mixture | argmax | held (best) | replicas in basin |
|---|---|---|---|---|---|---|---|
| `L_p1` | 1 | **3.204** | 0.000 | 0.000 | 0.000 | 0.000 | 0/1 |
| `L_p64` | 64 | **3.041** | 0.001 | 0.000 | 0.000 | 0.001 | **0/64** |

The 64 replicas' `local_ce` values span **3.04–4.13**. The basin cliff is at
**0.006**. That is a factor of **~600**, and it does not move when you take the
best of 64 independent draws. This independently reproduces
`explore/alu-relational`'s measurement of the plain legal baseline at Stage-1
conditions (`local_ce` 3.5–4.2).

**This is the honest boundary of the whole branch.** A population multiplies
the number of draws from a distribution; it cannot move a distribution whose
entire support is 600x away from the target. The population converts *1-in-7
into ~1* — it does not convert *0-in-infinity into anything*. **A legal
per-step signal is still missing, and nothing here supplies one.**

Note also that the 64 replicas of `L_p64` are *not* diverse in any useful way:
they all sit in the same failure mode with `local_ce` within 30% of each other.
That is the signature of a systematic obstruction, not a basin-hunting problem,
and it is a second reason not to expect population methods to help the legal
objective.

### 6.1 The re-screen: `alu-relational`'s LEGAL candidates at P = 32

`alu-relational` closed the legal-signal search on **44 training runs, each at
P = 1** with one or two seeds. Section 4.2 of this report shows the basin is
selected by *initialisation* with a per-replica rate of 0.26 under a working
signal — so a P=1 screen sees **one draw from a distribution**. If any legal
term had even a small tail toward the cliff, that design could not see it, and
the "closed" verdict would be an artifact of the screening rather than a fact
about the signals. This tests the closure itself.

`lab/probe_pop_legal.py` ports the loss terms from
`explore/alu-relational/lab/probe_rel.py` (read-only to this branch; nothing
there was edited or run) with a replica index added to every table read. Terms
are exactly the six that report §7 rules **LEGAL**; `--rel affine` (which §7
flags as on the wrong side of its own line) and `--rel true` (illegal) are
deliberately **not** implemented. `local_ce` is *measured* with a constructed
tape and never enters the loss, so the training is fully **LEGAL**.

**16 runs, 544 replicas, 1,200 steps each, m1 scale, weights 1.0 as in
`alu-relational`'s job files.**

| term | `local_ce` min | p10 | median | max | spread | replicas < 0.006 | < 1.0 | < 2.34 | best `train_exact_hard` | ms/step |
|---|---|---|---|---|---|---|---|---|---|---|
| **baseline** (label only) | 2.975 | 3.047 | 3.347 | 4.644 | 0.50 | **0/32** | 0 | 0 | 0.001 | 95 |
| baseline, seed 1 | 2.974 | 3.121 | 3.575 | 5.154 | 0.61 | **0/32** | 0 | 0 | 0.001 | 95 |
| `--dual fold` | 2.750 | 2.965 | 3.304 | 4.261 | 0.46 | **0/32** | 0 | 0 | 0.000 | 196 |
| `--dual fold`, seed 1 | 2.899 | 2.942 | 3.268 | 5.802 | 0.89 | **0/32** | 0 | 0 | 0.001 | 200 |
| `--dual redall` | 3.007 | 3.063 | 3.420 | 5.003 | 0.59 | **0/32** | 0 | 0 | 0.001 | 206 |
| `--dual horner` | 2.673 | 2.765 | 3.209 | 4.150 | 0.47 | **0/32** | 0 | 0 | 0.001 | 254 |
| `--sym` | 2.531 | 2.668 | 2.924 | 3.838 | 0.45 | **0/32** | 0 | 0 | 0.001 | 97 |
| `--inv` | 4.845 | 5.121 | 5.674 | 7.040 | 0.39 | **0/32** | 0 | 0 | 0.000 | 96 |
| `--assoc` | **2.304** | 2.397 | 2.622 | 3.534 | 0.47 | **0/32** | 0 | 1 | 0.000 | 102 |
| `--assoc`, seed 1 | 2.304 | 2.363 | 2.614 | 4.188 | 0.73 | **0/32** | 0 | 2 | 0.002 | 103 |
| `--assoc`, seed 2 | 2.320 | 2.370 | 2.604 | 3.973 | 0.64 | **0/32** | 0 | 3 | 0.000 | 103 |
| `--assoc`, **P=64** | 2.328 | 2.354 | 2.817 | 3.930 | 0.57 | **0/64** | 0 | 1 | 0.001 | 190 |
| `--cancel` | 2.492 | 2.596 | 2.907 | 4.615 | 0.73 | **0/32** | 0 | 0 | 0.002 | 96 |
| `--assoc --cancel` | 2.309 | 2.390 | 2.617 | 3.362 | 0.40 | **0/32** | 0 | 1 | 0.001 | 100 |
| `--sym --inv --assoc --cancel` | 3.803 | 4.266 | 4.915 | 7.532 | 0.76 | **0/32** | 0 | 0 | 0.000 | 92 |
| `--rel free` | 2.795 | 2.984 | 3.353 | 4.628 | 0.55 | **0/32** | 0 | 0 | 0.001 | 95 |

**0 of 544 replicas reached `local_ce` 1.0, let alone the 0.006 cliff.** The
best value anywhere is **2.304**, against `alu-relational`'s best-ever legal
value of 2.343 at P=1 — **32x more draws bought 1.7%.** No `train_exact_hard`
exceeded 0.002.

**Nothing was too expensive for P=32.** The most costly term, `--dual horner`,
runs at 254 ms/step against the baseline's 95 (2.7x, because horner reduces at
K=9 places instead of 6 and the agreement needs a second full square);
`--dual fold`/`redall` are ~2.1x; every algebraic law is within 8% of the
baseline. A 1,200-step P=32 run is 2–5 minutes. I ran everything at P=32 and
`--assoc` additionally at P=64; nothing had to be substituted.

### 6.2 The tail question, answered directly

The coordinator asked specifically whether any term *shifts or fattens the
tail*, since that is what P=1 screening cannot see. Two terms do move the
distribution, and neither is progress:

* **`--assoc` shifts the whole distribution left by ~22%** (median 3.35 -> 2.62,
  min 2.98 -> 2.30) and does so **reproducibly across 3 seeds and at P=64**.
  This is the largest distributional effect of any legal term in this repo.
* **`--cancel` and the 4-law stack fatten the spread** (0.50 -> 0.73 and 0.76)
  without moving the minimum usefully.

**But the shift is the constant-map collapse, and I can now show it per
replica.** I added `alu-relational`'s output-diversity collapse detector at
replica granularity — the fraction of *distinct* predicted answers over 256
inputs; a map collapsed to a constant reads 1/256 = 0.004, the truth reads
1.000:

| run | diversity of the **best-`local_ce`** replica | worst replica's diversity |
|---|---|---|
| baseline, seed 1 | **0.391** | 0.234 |
| `--dual fold`, seed 1 | **0.516** | 0.387 |
| `--assoc`, seed 1 | **0.203** | **0.004** |
| `--assoc`, seed 2 | **0.035** | **0.004** |
| `--assoc`, P=64 | **0.184** | **0.004** |
| `--assoc --cancel` | **0.176** | **0.004** |

The replicas that `--assoc` drives to the best `local_ce` are predicting the
same answer for 80–96% of inputs, and its worst replicas have collapsed to a
**literal constant** (0.004 = 1 distinct answer in 256). The baseline and the
dual-path runs, which have *worse* `local_ce`, keep 2–4x the diversity.
**`--assoc` lowers `local_ce` by destroying the map, so its leftward shift is
in the wrong direction.** This confirms `alu-relational`'s reading of its own
best number and upgrades it from an aggregate observation to a per-replica one.

**And `--cancel` does not prevent it** — which is new and sharper than anything
in the sibling report. `--cancel` exists precisely to exclude the constant
adder (§7 of that report: "its only content is *the adder is not constant*").
Run together with `--assoc` it leaves the best replica at diversity 0.176 and
still admits fully collapsed replicas at 0.004. The non-degeneracy law is
evaluated on the *adder*, while the collapse happens in the *composed map*, so
it prices the wrong object.

### 6.3 The shape of the distribution is the whole story

Put the two regimes side by side at the same P=32, same graph, same budget:

| | working (teacher-forced, **DIAGNOSTIC**) | every legal term (**LEGAL**) |
|---|---|---|
| `local_ce` distribution | **bimodal**, with a spike *at zero* | **unimodal blob**, 2.3–7.5 |
| min | **0.0000** | 2.304 |
| replicas below the cliff | **11/32** | **0/544** |
| relative spread (max-min)/median | ~3 (multi-scale) | 0.39–0.89 |
| best `train_exact_hard` | **1.000** | 0.002 |

A working signal produces a population with *mass on the solution*. Every legal
term produces a tight unimodal blob **~400x** away from the cliff whose width
is a factor of two, not a factor of a thousand. The "replicas sit within 30% of
each other" signature I flagged in §6 holds across all 544: **the obstruction is
systematic, not stochastic**, and a population is the wrong instrument for a
systematic obstruction — which is exactly what it *should* show if the closure
is real.

**Verdict on the closure: it stands, and it is now robust to the
screening-design objection.** `alu-relational`'s conclusion was reached with one
draw per configuration; it survives 32–64 draws per configuration across 16
configurations and 544 replicas, with the one apparent exception diagnosed as a
degeneracy at replica granularity. This is worth as much as a positive result:
the closure is a fact about the signals, not an artifact of how they were
screened.

---

## 7. Cheaper alternatives, tested first — only one moves the rate, and not enough

The mandate said to test variance reduction, weight averaging, restarts and
init before concluding a population is needed. **The population is the cheapest
instrument for testing them**: one P=32 run *is* 32 basin draws, so a single
600 s run resolves a rate that would take 32 separate runs to measure. Every
row below is one run at P=32, seed 0, 1,200 steps, against the two baseline
runs at **11/32 and 10/32**. **DIAGNOSTIC** (teacher forcing throughout).

| lever | replicas in basin | best `train_exact_hard` | verdict |
|---|---|---|---|
| **baseline** (lr 3e-2, batch 512, init 0.5, per-replica clip) | **11/32, 10/32** | 1.000 | — |
| batch 512 -> **2048** (gradient variance /4) | 10/32 | 1.000 | **no effect**, and 2.8x the cost/step |
| lr decayed 3e-2 -> 3e-3 over the run | 11/32 | 1.000 | **no effect** |
| lr 3e-2 -> **1e-1** | 11/32 | 0.953 | **no effect** on the rate |
| lr 3e-2 -> **3e-3** | **0/32** | 0.281 | **destroys it** — too slow to arrive in 1,200 steps |
| **init scale 0.5 -> 0.25** | **17/32** | 1.000 | **the only lever that helps: 0.34 -> 0.53** |
| init scale spread over a 16x log range across replicas | 9/32 | 1.000 | no effect (dilutes with bad scales) |
| **global** grad clip (as the evaluator does) instead of per-replica | 11/32 | 1.000 | **no effect — see §7.1** |
| reinitialise the worst 50% every 200 steps (80 reinits) | 9/32 | 1.000 | **no effect** — see §7.2 |
| **weight averaging** across replicas (P=8) | — | **0.001** | **catastrophic — see §7.3** |

**Conclusion: ordinary variance reduction does not move the basin rate.** Batch
size, learning rate and learning-rate schedule are all flat; the only knob that
moves it is the *initialisation scale*, and halving it takes the per-replica
rate from 0.34 to 0.53 — useful, worth stacking, and **not remotely enough on
its own**: a single model at 0.53 still fails once in two runs, and the
evaluator gives you one run. A population is not merely the best option here,
it is the only one that reaches ~1.

### 7.1 Global gradient clipping does not couple the replicas — this matters for legality

The evaluator clips with `clip_grad_norm_(raw_model.parameters(), 1.0)`
(`benchmark/runner.py:331-334`), which is a **global** norm over the whole
population — so one exploding replica rescales everybody's update. That is a
real coupling and it would break the "P independent runs" claim if it mattered.
It does not: global clipping gives **11/32**, identical to per-replica clipping.
The reason is that AdamW normalises per parameter, so a common scale factor on
the gradient is almost entirely absorbed. **A submission can use the
evaluator's own clipping and still get independent replicas.**

### 7.2 Reinitialising losers inside `optimizer.step()` — flagged, and it does not help anyway

`AdamWReinit` (in `lab/probe_pop.py`) re-randomises the lowest-`alpha` half of
the population every 200 steps and zeroes their Adam moments. 80 reinitialisations
over the run gave **9/32**, indistinguishable from the 10–11/32 baseline: a
replica that is going to find the basin does so in the first few hundred steps,
so recycling losers just restarts a clock that has already run out.

**Compliance, as the mandate asked me to judge it.** I do not think this breaks
rule 8 — the gradient path from the loss to every parameter is rebuilt and
intact on every step, and a custom `torch.optim.Optimizer` is explicitly
permitted — but it is a discrete, non-gradient jump applied to the parameters
that produce the prediction, and which replica survives to eval depends on when
the jumps happened. It is a gray area, and since it **buys nothing** there is no
reason to spend the argument. **Flagged, not shipped.** The mixture head is
unambiguous and does the job.

### 7.3 Weight averaging is catastrophic, and the reason is instructive

Averaging the replicas' parameters (P=8, one replica at 1.000 and another at
0.896) gives `train_exact_hard` = **0.001**. This is not a near miss, it is the
floor.

The mechanism is the **gauge**. `Tmul`'s output code is only identified up to a
relabelling of the 10 digit classes — `probe_tf2`'s `mul_gauge` score exists
precisely because a solved table can use any bijection of the output alphabet.
Two replicas that have both solved the task will in general have solved it in
*different gauges*, and the average of two one-hot tables in different gauges is
a uniform table. **Any parameter-space averaging or ensembling of these models
is meaningless**; only output-space mixing (which is what `alpha` does) or
selection is coherent. That generalises beyond this branch — it applies to every
learned-table architecture in this repo.

---

## 8. Verdict

**A population of independent replicas converts the 1-in-7 basin into ~1 inside
a single evaluator run, at P >= 8, for a 29% wall-clock premium. Selection is
free and reliable. Both halves are LEGAL. And it does not currently matter,
because the signal that reaches the basin at all is not.**

The three numbers, in the conditional frame of §0:

1. **Basin hits vs P** — run-level 1/7 at P=1, **3/3 at P=8, 2/2 at P=32, 1/1
   at P=64**; per-replica rate 0.26 pooled over 159 replicas, statistically
   identical at every P. Replicas inside one run are as independent as separate
   seeds because **the basin is chosen by initialisation, not by data order**.
   `1 - (1-0.26)^P` puts P=8 at 0.91 and P=32 at 0.9999.
2. **Selection recovers the good replica** — mixture, argmax replica and
   CE-argmin all return the population's best replica in all six runs, on
   held-out data as well as train, under five selector variants. The blend does
   *not* hide a correct component here, because `alpha` saturates; but there is
   a real window (P=64, step 400: blend 0.999 vs commit 0.943) where the blend
   is *better*, so **report both** rather than assuming the mode always wins.
3. **Cost** — replicas are nearly free to P=8 (1.29x) and cheap to P=16
   (1.55x); the knee is P~16-32 and past P=64 the marginal cost is a flat
   3.4 ms/replica. **P=8 is the sweet spot**: it costs 22% of your optimizer
   steps and takes the failure probability from 6/7 to 1/11.

**What this does not do.** The legal end-of-chain objective is unmoved: 64
replicas all sit at `local_ce` 3.0-4.1 against a cliff at 0.006, and 0/64 reach
the basin (§6). A population multiplies draws from a distribution; it cannot
move a distribution whose entire support is 600x from the target. **This result
is conditional on someone finding a legal per-step signal.** That has not
happened, and §6.1-6.3 is my attempt to break the closure with the instrument
this branch built: **16 legal configurations, 544 replicas, 0 below `local_ce`
1.0**, best 2.304 against `alu-relational`'s 2.343 at P=1. The one term that
moves the distribution (`--assoc`, reproducibly, across 3 seeds and at P=64)
does so by collapsing the map, which I can now show at replica granularity, and
`--cancel` does not prevent it. **The closure stands and is now robust to the
screening-design objection** — which was the point of testing it.

**No submission.** There is no legal end-to-end candidate on this branch, so
`submissions/explore/alu-population/` was not created and no evaluator cell was
run. Shipping a population trained by the legal objective would score MAX_T = 0
like everything else, and shipping the teacher-forced one is a compliance
violation. Building either would have produced a number that a reader could
mistake for a result.

### 8.1 The single highest-value recommendation

**Stop discounting a legal per-step signal by its seed-fragility, and make the
replica dimension standard equipment for every seed-fragile procedure in this
repo.**

Before this branch the honest summary of `alu-credit` Stage 1 was "there is a
solution, but you only find it 1 time in 10, and the evaluator gives you one
seed" — which reads as *two* blockers, and makes the search for a legal signal
look less worth funding. **It is one blocker.** The seed-fragility half is
solved, cheaply, legally, and with no tuning: 8 replicas, a softmax over them,
one `optimizer.step()`, +29% wall clock, and the selector needs no schedule. If
a legal per-step signal is found and it lands anywhere near the same basin
structure, the path from it to a certified rung is short.

The corollary is general and costs almost nothing: **any procedure in this repo
whose outcome depends on the seed should be run with a replica axis rather than
with more seeds.** One P=32 run is 32 draws for 2x the cost of one — a 16x
improvement in draws per GPU-second — and it is the instrument that made §7's
alternatives measurable at all. Three concrete places it applies today:
`depth-controller`'s residual "1 of 5 seeds lands on the wrong unit digit",
`alu-relational`'s search over relational laws, and any future grokking-style
recipe search.

### 8.2 What I would run next, in order

1. **~~Re-run `alu-relational`'s best legal candidates at P=32.~~ DONE — §6.1.**
   16 configurations, 544 replicas, null. The closure holds. Do not re-open the
   legal-signal search inside `DigitALU` on the grounds that it was
   under-screened; it was not.
2. **Stack init scale 0.25 with the population.** It is the only cheap lever
   that moved the rate (0.34 -> 0.53) and it composes with P for free; at P=8 it
   takes the run-level success from 0.91 to 0.99. Sweep the scale properly —
   I tested two values.
3. **Measure the population's step-budget curve.** Everything here is at 1,200
   steps (the `tree:quotient` Hard budget). The basin is entered by step 400-800
   at P>=32, so a population may fit Medium's ~170 steps once the per-step
   signal exists; that is worth knowing before anyone assumes Hard is the only
   viable tier.
4. **Do not spend more time on weight averaging, restarts, batch size or LR
   schedules** for this failure mode. All four are measured flat or negative
   (§7).

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
