# grok-optimization — optimization & regularization for *exact* generalization

Branch `explore/grok-optimization`. Metric is the post-2026-07-24 one: **Max T**, the
largest rung whose exact accuracy is 100% with every lower rung also 100%. On `e1` a rung
is 38 held-out examples, so one wrong example ⇒ `MAX_T = 0`.

Every comparison used `--mode fixed_step` (100 000 s budget, hard `max_steps`), so step
counts are contention-immune. The GPU was shared with three other agents throughout;
**wall-clock seconds inside the fixed-step tables are lower bounds on throughput**. The
two deliberate timing measurements are called out in §10 with their contention state.

---

## 0. TL;DR

* **No grokking transition was observed anywhere.** 57 archived runs (52 on `e1`),
  ~30 distinct configurations, step budgets 2 × 10² → 5 × 10⁵, 3 seeds at the key point.
  Over the 50 successful `e1` runs the rung-1 exact accuracy histogram is
  **0/38 (37 runs), 1/38 (12 runs), 2/38 (2 runs)** — i.e. the trivial floor, always.
  **`MAX_T = 0` in every single run.**
* **Steps-to-exactness on rung 1 is > 2 × 10⁵ optimizer steps** (measured, three weight
  decays) — a lower bound, and the shape of the evidence says the true answer is "never
  for this input representation", not "somewhere past 10⁶".
* The model **memorises perfectly** — train exact-accuracy reaches 1.00 by ≈2 000 steps
  for every recipe with wd ≤ 0.1 and stays there — and **transfers nothing** to unseen
  `x`. That is the signature of a lookup table, not of a pre-grok plateau. In real
  grokking, test accuracy creeps above chance before it jumps; here it never leaves the
  trivial floor.
* Weight decay (0 → 3.0), learning rate (3e-4 → 1e-2), schedule (const / warmup+cosine /
  warmup+linear), optimizer (AdamW / Muon / Schedule-Free AdamW), Grokfast, ⊥Grad,
  StableMax, small init, orthogonal init, embedding-norm projection, batch size
  (16 → 512), width (32 → 256) and loss shape (CE / focal / hardest-token / label
  smoothing) **all move rung-1 by less than one example out of 38**.
* Two diagnostic control datasets (§8) — the same `e1` recipe at `N = 77` (2-digit,
  40 training facts) and `N = 1147` (4-digit, 800 training facts) — fail identically. So
  the wall is **not** "too few facts" and **not** "the arithmetic is too wide".
* **The one real, transferable win here is throughput, not accuracy.** The evaluator's
  DataLoader is `num_workers=2` with no `persistent_workers`, so it respawns workers
  every epoch. `e1` has ~600 train rows, so the manifest default `batch_size = 512` is
  *one batch per epoch* → a worker respawn on **every step**. Measured on a quiet GPU:
  111 ms/step at bs=512 vs **19 ms/step at bs=32** — a **5.8× step multiplier for free**,
  and 1.5× even on a Medium-sized dataset where the respawn effect is absent.
* **Feasibility verdict (throughput assumption stated in §10): rung-1 certification is
  not reachable at Easy, Medium or Hard by any training-recipe change.** Hard's ~1.7 × 10⁵
  achievable steps is *below* the 2 × 10⁵ step budget already shown empty.

---

## 1. Hypothesis, and its status

`e1` looks like textbook grokking territory: a tiny training set (~600 rows) against an
exact discrete structure, with a model that trivially memorises it. The literature says
that in this regime exact generalization arrives suddenly, and that the arrival time is
governed by weight decay, learning rate and step count far more than by architecture.

**H:** holding the architecture roughly fixed (small bidirectional transformer, d = 128,
2 blocks, 4 heads, tied head), some training recipe drives rung-1 to 38/38, and the
decision-relevant number is how many optimizer steps it needs.

**Status: falsified**, with the specific and useful failure mode identified in §9.

## 2. What rung 1 actually is (from the generator source; the data was never read)

`data/squaring_mod.py` + `scripts/generate_datasets.sh` fix `e1` as `p = 17, q = 19`,
`N = 323`, `phi = 288`, `time_steps = [1,2,3]`, `examples_per_setting = 250`,
`depth_evaluation_exhaustive_x = true`, `train_fraction = 0.8`.

`_generate_prompt_grouped_records` enumerates the `phi(323) = 288` units and reserves
`288 − max(250, 100) = 38` of them **entirely** — those 38 `x` values appear in no train,
test or OOD prompt, and are exactly the depth-ladder cohort at every rung.

| quantity | value |
|---|---|
| units of `N = 323` | 288 |
| `x` values available to train/test/ood | 250 |
| train rows | 250 × 0.8 × 3 T-settings = **600** |
| rung cohort (every rung) | the **same 38 never-seen `x`** |
| rung-1 target | `x² mod 323` — **T = 1 is in the training distribution** |

Two consequences that shape the whole result:

1. **Rung 1 is not an extrapolation in T.** It is the easiest possible ask: reproduce the
   training-time map `x ↦ x² mod 323` on 38 fresh `x`. If rung 1 is unreachable, every
   deeper rung is too, and no amount of depth/recurrence work matters.
2. **The exponent ladder collapses.** `pow(2, T, 288)` for `T = 1,2,4,8,16,32,64` is
   `2, 4, 16, 256, 160, 256, 160`. Rungs 8 and 32 are the *same function*; so are 16 and
   64. Training only ever exhibits exponents 2, 4, 8. Useful for whoever attacks deeper
   rungs; it never became relevant here.

Prompt/target layout (`tokenize_squaring_mod_with_result`, `separate_input_output=True`):
`[N] 3 2 3 [X] <digits of x> [T] <digit of T>`, with the answer digits supervised at the
**last `len(answer)` positions of the prompt** (right-aligned). Answers are 1–3 decimal
digits with no leading zeros, so the number of supervised tokens per row varies. That
matters for §7.

## 3. Method and tooling

`lab/make_grok.py` emits a self-contained `submission.py` from a `CFG` dict. Every knob is
an optimizer / schedule / loss / batch knob; the model is fixed.

All participant logic lives where the rules allow: `build_model`, `build_optimizer` (one
custom `torch.optim.Optimizer` — explicitly permitted) and `training_loss`. The evaluator
keeps the loop, the backward, `grad_clip = 1` and one step per batch. `GrokOpt`
implements, selectable per param group:

* **AdamW** (decoupled decay),
* **Muon** — quintic Newton–Schulz orthogonalised momentum on hidden 2-D matrices, AdamW
  on embeddings/norms; one optimizer covering every trainable tensor exactly once (the
  evaluator validates coverage),
* **Schedule-Free AdamW** (Defazio et al. 2024),
* **Grokfast** (Lee et al. 2024) — `g ← g + λ·EMA_α(g)`,
* **⊥Grad** (Prieto et al. 2025, *Grokking at the Edge of Numerical Stability*) — project
  the gradient orthogonal to the weights so the loss cannot be reduced by scaling up the
  memorising solution,
* embedding row-norm projection; orthogonal / small-scale init.

`training_loss` implements CE, label smoothing, focal, a **hardest-token** reweighting
(softmax over per-token losses — the all-or-nothing metric is decided by the worst token
of a sequence) and **StableMax** CE.

Runner: `lab/grok.sh <name> <steps> <seed> "<note>" <knobs…>` → `lab/run_experiment.py`
→ one row in `lab/archive.jsonl`. Tables: `lab/grok_table.py --tag <axis>`.

```bash
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
$V lab/make_manifest.py --dataset e1 --mode fixed_step --max-steps 200000 --seeds 74
TAG=B-steps lab/grok.sh B_long200k_wd0.1_bs128 200000 74 "steps-to-exactness" \
    --wd 0.1 --wd-emb 0.1 --batch-size 128 --lr 0.001
$V lab/grok_table.py --tag B-steps --curve
```

Baseline held constant unless stated: d = 128, 2 blocks, 4 heads, mlp×4, tied head,
AdamW β = (0.9, 0.98), lr = 1e-3, wd = 0.1 (also on embeddings), constant LR,
`batch_size = 128`, CE loss, seed 74, dataset `e1`.

---

## 4. The steps-to-exactness curve — **there is no knee**

Matched recipe (bs = 128, lr = 1e-3, wd = 0.1, constant, AdamW), `e1`, seed 74:

| steps | rung-1 (38 ex) | rung-2 | rung-4 | test | mean_exact | final train loss | train exact-acc |
|------:|---:|---:|---:|---:|---:|---:|---:|
| 2 000 | 0.000 | 0.000 | 0.053 | 0.020 | 0.025 | 0.038 | ~1.00 |
| 10 000 | 0.000 | 0.026 | 0.026 | 0.027 | 0.028 | 0.000 | 1.00 |
| 50 000 | 0.000 | 0.000 | 0.053 | 0.020 | 0.025 | 0.010 | 0.99 |
| 200 000 | 0.026 | 0.000 | 0.000 | 0.027 | 0.038 | 0.011 | 1.00 |

Same sweep at two other weight decays, 200 000 steps:

| wd | steps | rung-1 | test | mean_exact |
|---:|---:|---:|---:|---:|
| 0.01 | 200 000 | 0.053 | 0.040 | 0.045 |
| 0.10 | 200 000 | 0.026 | 0.027 | 0.038 |
| 1.00 | 200 000 | 0.000 | 0.020 | 0.020 |

A 100× increase in step count changes rung-1 by at most 2 examples out of 38, which is
inside the run-to-run spread of everything else in this report. **`MAX_T = 0` at every
budget.** There is no transition, no creep before a transition, and no dependence on step
count at all.

The train curve is the important companion fact: **train exact-accuracy is 1.00 from
about step 2 000 onwards in every wd ≤ 0.1 run and stays there for the next 198 000
steps** while held-out accuracy does not move. The optimizer is working perfectly; there
is simply nothing for weight decay to select between, because the memorising solution is
the only solution the model ever finds.

## 5. Weight decay — the primary grokking knob does nothing here

10 000 steps, bs = 128, lr = 1e-3, seed 74:

| wd | rung-1 | rung-2 | test | mean | final train loss | train exact-acc @10k |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000 | 0.000 | 0.033 | 0.042 | 0.0000 | 1.00 |
| 0.01 | 0.053 | 0.000 | 0.047 | 0.053 | 0.0000 | 1.00 |
| 0.1 | 0.000 | 0.026 | 0.027 | 0.028 | 0.0000 | 1.00 |
| 1.0 | 0.000 | 0.000 | 0.027 | 0.058 | 0.1071 | 0.95 |
| 3.0 | 0.000 | 0.000 | 0.027 | 0.063 | 1.8704 | **0.08** |

Reading: wd = 1.0 is already at the edge — it starts eroding the memorised solution
(train acc 0.95) without buying any generalization. wd = 3.0 destroys training outright.
So the entire *usable* weight-decay range has been covered and the transition is not
inside it. Combining the strongest usable decay with the strongest accelerator
(wd = 1.0 + Grokfast λ = 5) also gives rung-1 = 0.000.

## 6. Learning rate, schedule and optimizer

**Learning rate** (10 000 steps, bs = 128, wd = 0.1):

| lr | rung-1 | test | mean |
|---:|---:|---:|---:|
| 3e-4 | 0.000 | 0.040 | 0.035 |
| 1e-3 | 0.000 | 0.027 | 0.028 |
| 3e-3 | 0.000 | 0.040 | 0.035 |
| 1e-2 | 0.000 | 0.040 | 0.035 |

**Schedule** (10 000 steps, bs = 128, wd = 0.1, peak lr = 2e-3 for the decaying ones):

| schedule | rung-1 | test | mean |
|---|---:|---:|---:|
| constant (lr 1e-3) | 0.000 | 0.027 | 0.028 |
| warmup 500 + cosine→0 | 0.000 | 0.033 | 0.042 |
| warmup 500 + linear→0 | 0.000 | 0.027 | 0.023 |

No benefit — and there is a real *hazard* the evaluator makes worse. The evaluator scores
the **final** checkpoint, and a submission cannot know how many steps its wall-clock
budget will buy. A decaying schedule must guess a horizon: guess too long and the final
checkpoint sits at near-peak LR; guess too short and the remaining budget runs at lr ≈ 0.
Since decay buys nothing measurable here, **constant LR is the correct default for a
submission** and is what `submissions/grok-optimization/submission.py` uses.

**Optimizer** (10 000 steps, bs = 128, wd = 0.1; Muon lr_hidden = 0.02, AdamW lr = 1e-3
on embeddings/norms; Schedule-Free with 200-step warmup):

| optimizer | rung-1 | test | mean | train exact-acc @10k | s / 10k steps (contended) |
|---|---:|---:|---:|---:|---:|
| AdamW | 0.000 | 0.027 | 0.028 | 1.00 | 248 |
| Muon + AdamW-aux | 0.026 | 0.027 | 0.033 | 0.94 | 538 |
| Schedule-Free AdamW | 0.000 | 0.027 | 0.043 | 1.00 | 514 |

At 50 000 steps: Muon rung-1 = 0.000, Schedule-Free not re-run. **Better updates: none of
them buys anything at equal step count.** **Cheaper updates:** Muon is the *most*
expensive per step (five Newton–Schulz iterations per 2-D matrix on a tiny model where
the optimizer is a meaningful fraction of a 20 ms step); Schedule-Free costs the same
state as AdamW and roughly the same time; plain AdamW is the cheapest. Since accuracy is
identical, **AdamW wins on throughput alone**.

## 7. Batch size and loss

**Batch size** (10 000 fixed steps, wd = 0.1, lr = 1e-3, seed 74). Note that at 600 train
rows, bs = 512 is 85 % of the dataset per step, i.e. essentially full-batch:

| bs | batches / epoch | epochs @10k steps | rung-1 | test | mean | s / 10k steps (contended) |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 37 | 270 | 0.000 | 0.013 | 0.022 | 397 |
| 32 | 18 | 555 | 0.026 | 0.020 | 0.035 | 276 |
| 128 | 4 | 2 133 | 0.000 | 0.027 | 0.028 | 248 |
| 512 | 1 | 8 533 | 0.000 | 0.027 | 0.023 | 1 311 |

Near-full-batch at 50 000 steps (bs = 512): rung-1 = 0.026, test = 0.020. The classic
grokking setup (full batch, long training, weight decay) is therefore covered and also
produces nothing. Batch size does not affect accuracy here; it affects **only**
throughput (§10), where it matters a great deal.

**Loss** (10 000 steps, bs = 128, wd = 0.1). The score is all-or-nothing per sequence, so
token-averaged CE is misaligned. The `training_loss` signature the evaluator provides is
`(logits[n_valid_tokens, V], labels[n_valid_tokens], aux)` — **already flattened across
the batch, with no sequence boundaries**, and because `e1` answers are 1–3 digits the
per-row valid-token count varies, so true per-sequence weighting is *not* expressible in
this API without the model smuggling row lengths through `aux` (it cannot: `forward` never
sees the labels). The closest expressible surrogate is to concentrate the loss on the
worst tokens, since one bad token kills a sequence — that is the `hard` row below.

| loss | rung-1 | test | mean |
|---|---:|---:|---:|
| CE (no smoothing) | 0.000 | 0.027 | 0.028 |
| CE + label smoothing 0.1 | 0.000 | 0.027 | 0.018 |
| focal, γ = 2 | 0.000 | 0.033 | 0.037 |
| hardest-token softmax reweighting, τ = 0.5 | 0.000 | 0.013 | 0.047 |
| StableMax CE | 0.000 | 0.027 | 0.028 |

Label smoothing is confirmed to be pointless-to-mildly-harmful under an exact-match
metric (mean 0.018 vs 0.028), and it is *not* in the default path — plain
`F.cross_entropy` is unsmoothed, so "remove label smoothing" is a no-op unless a
submission adds it. The hardest-token loss destabilises training (train exact-accuracy
oscillates down to 0.39 mid-run) without helping.

## 8. Grokking accelerators, init, capacity — and the two control datasets

10 000 steps, bs = 128, lr = 1e-3, seed 74:

| variant | wd | rung-1 | test | mean |
|---|---:|---:|---:|---:|
| baseline | 0.1 | 0.000 | 0.027 | 0.028 |
| Grokfast λ = 2, α = 0.98 (**50k steps**) | 0.1 | 0.000 | 0.053 | 0.040 |
| Grokfast λ = 5, α = 0.98 (**50k steps**) | 0.1 | 0.000 | 0.053 | 0.057 |
| Grokfast λ = 5 + wd = 1.0 | 1.0 | 0.000 | 0.007 | 0.013 |
| ⊥Grad | 0 | 0.000 | 0.013 | 0.022 |
| ⊥Grad | 0.1 | 0.026 | 0.020 | 0.025 |
| StableMax CE | 0.1 | 0.000 | 0.027 | 0.028 |
| ⊥Grad + StableMax | 0 | 0.026 | 0.040 | 0.035 |
| ⊥Grad + StableMax (**50k steps**) | 0 | 0.000 | 0.053 | 0.047 |
| small init (gain 0.2, emb std 0.004) | 0.1 | 0.000 | 0.027 | 0.018 |
| orthogonal init | 0.1 | 0.000 | 0.033 | 0.037 |
| embedding row-norm → 1.0 | 0.1 | 0.000 | 0.027 | 0.023 |
| width d = 32 | 0.1 | 0.000 | 0.033 | 0.032 |
| width d = 64 | 1.0 | 0.026 | 0.040 | 0.030 |
| width d = 256 (**50k steps**) | 0.1 | 0.026 | 0.020 | 0.035 |

Grokfast and ⊥Grad are the two published interventions that specifically claim to remove
the delay before grokking on modular arithmetic. Both are implemented inside the custom
optimizer (rules-compliant: the evaluator still owns the loop and the backward). Neither
moves rung-1.

**Seed spread** — 50 000 steps, baseline recipe, three seeds:

| seed | rung-1 | rung-2 | rung-4 | test | mean |
|---:|---:|---:|---:|---:|---:|
| 74 | 0.000 | 0.000 | 0.053 | 0.020 | 0.025 |
| 75 | 0.000 | 0.000 | 0.000 | 0.040 | 0.030 |
| 76 | 0.000 | 0.000 | 0.053 | 0.027 | 0.038 |

**Zero spread on the thing that matters**: rung-1 is exactly 0/38 in all three seeds.
This is the answer to "did a transition happen in *some* seed": no, not in any seed, at
any budget, under any recipe. Across all 50 successful `e1` runs the rung-1 values observed are
{0.000 (37×), 0.026 (12×), 0.053 (2×)} — i.e. 0, 1 or 2 correct out of 38, which is the
trivial-predictor floor (a model that emits a plausible digit string occasionally hits a
short answer).

**Positive control for the metric machinery.** A rung is `certified` only when
`correct_examples == example_count`, computed by the same `_loss_and_accuracy` path
(`exact_rows` = every supervised token in the row correct) that reports training
accuracy. That path *does* return 1.0 in these runs — on the training batches, from
~step 2 000 onwards. So exactness is reachable and observable through this code; it is
specifically the held-out-`x` cohort that never gets there.

### Control datasets — it is not a fact-count or arithmetic-width problem

`lab/gen_grok_controls.sh` (public generator only; the rows were never read) reproduces
the exact `e1` recipe at two other fixed moduli:

```
gsmall : --fixed_p 7  --fixed_q 11  -> N=77   (2-digit), 60 units,  ~40 train facts / T, 10 reserved x
e1     : --fixed_p 17 --fixed_q 19  -> N=323  (3-digit), 288 units, ~200 train facts / T, 38 reserved x
gbig   : --fixed_p 31 --fixed_q 37  -> N=1147 (4-digit), 1080 units,~800 train facts / T, 80 reserved x
```

10 000 steps, wd = 0.1, lr = 1e-3 (gsmall uses bs = 32 because it only has 120 train rows
and bs = 128 leaves no complete batch — the evaluator raises on that, which is itself
worth knowing):

| dataset | train facts / T | held-out x | rung-1 | test | train exact-acc |
|---|---:|---:|---:|---:|---:|
| gsmall (N = 77) | ~40 | 10 | 0.100 = **1/10** | 0.033 | 1.00 |
| e1 (N = 323) | ~200 | 38 | 0.000–0.053 | 0.027 | 1.00 |
| gbig (N = 1147) | ~800 | 80 | 0.013 = **1/80** | 0.008 | 0.98 |

Across a 20× range in the number of training facts and 2→4 decimal digits of arithmetic,
held-out-`x` accuracy stays at the floor while training accuracy stays at 1.00. **The
blocker is neither too little data nor too much arithmetic.** It is that nothing the
model learns extends off the training support at all.

## 9. What this means — the failure mode, stated precisely

The model learns a **lookup table keyed on the digit tokens of `x`**, and learns it fast
(≈2 000 steps), completely (train exact = 1.00) and stably (still 1.00 at 200 000 steps
under wd = 0.1). No optimizer, regularizer or step budget converts that table into an
algorithm.

The mechanism is a representation mismatch, and it explains why the grokking literature
does not transfer:

* Classical grokking on modular arithmetic uses **one token per operand**. The operand's
  embedding vector is a free parameter, and the p² training pairs all share the same p
  embeddings, so weight decay has something to do: it reorganises that shared table into a
  Fourier basis, and the reorganisation generalises.
* Here `x` arrives as **≤3 shared decimal-digit tokens**. The model's representation of
  `x` is forced to be a sum of position-tagged digit embeddings, from a pool of only 10
  digits. To generalise it would have to (a) reconstruct the integer value from the digit
  code, (b) square it, and (c) reduce mod N — i.e. learn multi-digit multiplication plus
  modular reduction inside one forward pass, which small transformers are famously bad at
  without chain-of-thought. There is no shared-embedding structure for weight decay to
  reorganise, so the "grokking" pressure has no lever.
* And crucially, at T = 1 the target is a **unary** map: the entire universe is 288
  `(x, x² mod 323)` facts. There is no combinatorial constraint (as there is with p²
  pairs over p embeddings) forcing an algorithm rather than a table. The control datasets
  confirm this survives a 20× change in fact count.

## 10. Throughput, and the per-tier feasibility verdict

### The DataLoader finding

`data/factory.py` builds every loader with `num_workers = manifest.data.num_workers` (2 in
every official manifest) and **no `persistent_workers`**, so PyTorch forks fresh workers at
the start of each epoch. `e1` has ~600 train rows and the official manifest sets
`batch_size = 512` with `drop_last = True`, so **one batch = one epoch = a worker respawn
every single step**. A submission's only lever on this is `SUBMISSION.batch_size`.

Measured on `e1`, 200 fixed steps, d = 128 / 2 blocks, machine quiet (1-min load average
0.8–1.5, no other agent jobs running):

| batch_size | batches/epoch | ms / step | steps / s | vs bs=512 |
|---:|---:|---:|---:|---:|
| 512 | 1 | 111.3 | 9.0 | 1.0× |
| 200 | 3 | 47.9 | 20.9 | 2.3× |
| 128 | 4 | 38.4 | 26.0 | 2.9× |
| 64 | 9 | 28.0 | 35.7 | 4.0× |
| 32 | 18 | 19.1 | 52.3 | **5.8×** |
| 16 | 37 | 15.5 | 64.6 | 7.2× |
| 8 | 75 | 16.6 | 60.2 | 6.7× |

The fit is `ms/step ≈ 14 + 97 / (batches per epoch)`: ~97 ms per epoch restart plus a
~14 ms floor of Python/launch overhead. This is **not** an artifact of a tiny dataset
only: on `m1` (Medium, large train set, so no respawn effect) 500 fixed steps took
31.9 ms/step at bs = 512 vs **21.7 ms/step at bs = 128** — still 1.5× — because at this
model size the step is dominated by fixed per-step overhead, not by batch compute.

### Steps achievable per tier

Direct wall-clock check, `--mode wallclock`, `e1`, 60 s, **GPU deliberately shared** with
six of my own long fixed-step runs plus other agents — so these are *pessimistic*:

| batch_size | steps completed in 60 s (contended) |
|---:|---:|
| 512 | 349 |
| 128 | 673 |
| 32 | 1 254 |

**Throughput assumption for the H100 extrapolation, stated explicitly:** this workload is
overhead-bound, not GPU-bound — GPU utilisation was ~4 % in the prior session's
measurement, the step time fits a pure CPU-side model (worker respawn + Python), and the
model is ~0.8 M parameters on sequences of length ≤10. I therefore assume the H100 runs
this at **the same steps/second, ±2×**, dominated by host-side cost rather than by device
FLOPs. With that assumption, and taking the quiet-machine bs = 32 rate of ~52 steps/s
(and discounting a few seconds of import + build, which are charged to the budget):

| tier | budget | steps achievable @ bs=512 | steps achievable @ bs=32 | steps needed for rung-1 |
|---|---:|---:|---:|---|
| Easy | 60 s | ~500 | **~2 900** | > 200 000 (unmeasured; no transition seen) |
| Medium | 600 s | ~5 400 | **~31 000** | > 200 000 |
| Hard | 3 600 s | ~32 000 | **~187 000** | > 200 000 |

### Verdict

**Not reachable at any tier.** Even the most favourable reading — Hard, with the batch-size
throughput fix applied, ~1.9 × 10⁵ steps — lands *below* the 2 × 10⁵-step budget that was
directly measured to produce rung-1 = 0.026 (1 example out of 38). And the step axis is
flat: 2 × 10³ and 2 × 10⁵ steps give statistically identical results, so buying more steps
is not buying progress. The gap is not a factor of 2 or 10 in throughput; it is that the
target quantity does not respond to steps at all.

## 11. What was falsified

1. **"`e1` is a grokking problem and the answer is more steps."** False. 100× more steps
   (2 × 10³ → 2 × 10⁵) changes rung-1 by ≤2/38 and `MAX_T` not at all.
2. **"Weight decay is the knob."** False. The whole usable range 0 → 1.0 is covered
   (3.0 breaks training entirely) with no effect on held-out accuracy.
3. **"A grokking accelerator will surface it."** False for both published candidates:
   Grokfast (λ = 2 and 5, at 50 000 steps) and ⊥Grad + StableMax (at 50 000 steps).
4. **"It is seed luck."** False. 0/38 in all three seeds at 50 000 steps; 0–2/38 across
   all 50 successful `e1` runs.
5. **"Muon / Schedule-Free give better updates here."** False at equal step count; and
   Muon is the *most* expensive per step in this regime, so it is a net loss.
6. **"An LR schedule helps."** False, and it adds a real final-checkpoint hazard because
   the submission cannot know its own step horizon.
7. **"The training set is too small / the arithmetic is too wide."** False — the two
   control datasets (40 facts / 2-digit, and 800 facts / 4-digit) fail identically.
8. **"A metric-aligned loss will help."** Not falsified so much as *not expressible*: the
   evaluator flattens away sequence boundaries before `training_loss` sees the tensors,
   and `e1`'s variable answer length makes them unrecoverable. The best expressible
   surrogate (hardest-token reweighting) destabilised training and did not help.

## 12. What I would run next, and why

1. **Nothing further on this axis.** The optimizer/regularizer surface on `e1` is mapped
   and it is flat. Additional recipe search has negative expected value.
2. **Make the operand's *value* representable** (this is architecture work, and belongs to
   whoever owns representation): give the model a learned, differentiable way to turn the
   digit sequence of `x` into a value code — e.g. a learned place-value pooling into a
   scalar, then a learned bank of periodic features of that scalar — so that squaring can
   be a *linear* operation in the code (phase doubling) and therefore extends off the
   training support by construction. That is the only class of change that addresses the
   measured failure mode. Nothing may be hard-coded; the point is to make the Fourier-style
   solution *reachable*, which the digit-sum representation currently is not.
3. **Cheap validation protocol for anyone who tries that:** rung-1 on `e1` at 10 000 fixed
   steps is a 4-minute experiment with this tooling and is a perfect binary detector — if
   held-out-`x` accuracy is still at the floor, the representation has not changed
   anything, and no amount of optimization will rescue it.
4. **Apply the batch-size fix everywhere** (§10) — free steps at every tier, and the
   larger the deficit between required and achievable steps, the more it matters.

## 13. Submission

`submissions/grok-optimization/submission.py` — d = 128, 2 blocks, 4 heads, mlp×4, tied
head; custom `GrokOpt` in AdamW mode, lr = 1e-3, β = (0.9, 0.98), wd = 0.1, **constant
LR**, `batch_size = 128`, `eval_batch_size = 512`, `max_steps = 1_000_000` (so wall clock
binds). Verified through `lab/run_experiment.py` (which exercises
`benchmark/validation.py:lint_submission_source`); 14.7 KiB, one UTF-8 file, imports only
`torch` and `benchmark`.

It scores `MAX_T = 0`, like everything else in this report. Its only defensible claim over
the provided baseline is throughput: `batch_size = 128` is 2.9× more steps per second on
Easy-sized data and 1.5× on Medium-sized data, and it is the setting I could justify with
a measurement on both. It carries no schedule, because a schedule cannot know its horizon
and buys nothing measurable. **It should not be treated as a competitive entry** — it is a
matched baseline for the next architecture idea to beat.

## 14. Compliance

No file under `data/generated/` was read, printed or summarized at any point; the two
control datasets were produced by running the public generator (`lab/gen_grok_controls.sh`)
and only its "wrote N examples" line was observed. All task structure above is derived
from `data/squaring_mod.py`, `scripts/generate_datasets.sh` and elementary number theory.
No arithmetic or solver is hard-coded; every forward pass is learned parameters from
random init. No custom training loop and no participant-controlled backward — all custom
behaviour is inside a `torch.optim.Optimizer` and a `training_loss`, both explicitly
permitted. Nothing was submitted to the hosted service; `one-layer login`/`submit` were
never run. Every run, including the two failures, is in `lab/archive.jsonl`.
