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
