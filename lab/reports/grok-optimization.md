# grok-optimization — optimization & regularization for *exact* generalization

Branch `explore/grok-optimization`. Metric is the post-2026-07-24 one: **Max T**, the
largest rung whose exact accuracy is 100% with every lower rung also 100%. On `e1` a
rung is 38 held-out examples, so one wrong example ⇒ `MAX_T = 0`.

Everything below was run with `--mode fixed_step` (100,000 s budget, hard `max_steps`),
so step counts are contention-immune. The GPU was shared with three other agents
throughout; **wall-clock numbers in the fixed-step tables are therefore lower bounds on
throughput**, and the dedicated throughput measurements are called out separately.

---

## 0. TL;DR

* **No transition was observed.** Across 30 runs and ~20 distinct recipes on `e1`,
  rung-1 exact accuracy never exceeded **2/38 (0.053)** and was usually 0/38. **Max T = 0
  everywhere.** Steps-to-exactness on rung 1 is therefore **> 5 × 10⁵ optimizer steps**,
  or infinite for this architecture — a lower bound, not a measurement.
* The model **memorises perfectly** (train exact-accuracy = 1.00 by ~step 2 000 for every
  recipe with wd ≤ 0.1) and **transfers nothing** to unseen `x`. That is the diagnostic
  signature of a lookup table, not of a pre-grok plateau.
* Weight decay, learning rate, schedule, optimizer (AdamW / Muon / Schedule-Free),
  Grokfast, ⊥Grad, StableMax, small init, batch size and loss shape **all move rung-1 by
  less than one example out of 38**. Nothing is a knob because nothing is moving.
* **The one real, transferable win of this session is throughput, not accuracy**: the
  evaluator's DataLoader is rebuilt every epoch, and `e1`'s training set is ~600 rows, so
  `batch_size = 512` means *one batch per epoch* and pays a worker respawn on **every
  step**. Dropping `batch_size` to 32 gives **5.8× more optimizer steps per second**
  (111 ms/step → 19 ms/step) at zero accuracy cost.
* Feasibility verdict: since the required step count is unbounded from below at 5 × 10⁵,
  **rung-1 certification is not reachable at any tier by a training-recipe change alone.**
  Easy (60 s) buys ~3 000 steps, Medium ~30 000, Hard ~180 000 at the improved
  throughput — all far inside the region already shown to be empty.

---

## 1. Hypothesis

`e1` is textbook grokking territory: a tiny training set (~600 rows) against an exact
discrete structure, with a model that can trivially memorise it. The literature says that
in this regime exact generalization arrives suddenly, and that the arrival time is
governed by weight decay, learning rate and step count far more than by architecture.

**H:** with the architecture held roughly fixed (small bidirectional transformer,
d = 128, 2 blocks, 4 heads, tied head), there exists a training recipe that drives rung-1
to 38/38, and the decision-relevant quantity is the number of optimizer steps it needs.

**Falsified.** See §4–§9.

## 2. What rung 1 actually is (derived from the generator source, never from data)

`data/squaring_mod.py` + `scripts/generate_datasets.sh` fix `e1` as `p=17, q=19`,
`N = 323`, `phi = 288`, `time_steps = [1,2,3]`, `examples_per_setting = 250`,
`depth_evaluation_exhaustive_x = true`, `train_fraction = 0.8`.

`_generate_prompt_grouped_records` enumerates the `phi(323) = 288` units and reserves
`288 − max(250, 100) = 38` of them **entirely** — those 38 `x` values appear in no train,
test or OOD prompt and are exactly the depth-ladder cohort. So:

| quantity | value |
|---|---|
| units of `N = 323` | 288 |
| `x` values usable by train/test/ood | 250 |
| train rows | 250 × 0.8 × 3 T-settings = **600** |
| rung cohort (every rung) | the **same 38 unseen `x`** |
| rung-1 target | `x² mod 323` — **T = 1 is in the training distribution** |

Two consequences that shape the whole result:

1. **Rung 1 is not an extrapolation in T.** It is the *easiest possible* thing the model
   could be asked: reproduce the training-time map `x ↦ x² mod 323` on 38 fresh `x`.
   If rung 1 is unreachable, every deeper rung is too.
2. **The exponent ladder collapses.** `pow(2, T, 288)` for `T = 1,2,4,8,16,32,64` is
   `2, 4, 16, 256, 160, 256, 160`. Rungs 8 and 32 are the *same function*; so are 16 and
   64. Training only ever exhibits exponents 2, 4, 8. This is worth knowing for anyone
   attacking deeper rungs, but it did not become relevant here.

Prompt/target layout (`tokenize_squaring_mod_with_result`, `separate_input_output=True`):
`[N] 3 2 3 [X] <digits of x> [T] <digit of T>`, and the answer digits are supervised at
the **last `len(answer)` positions of the prompt**, i.e. right-aligned. The answer is 1–3
decimal digits with no leading zeros, so the number of supervised tokens per row varies
(1, 2 or 3). That matters for §9.

## 3. Method

Generator: `lab/make_grok.py` emits a self-contained `submission.py` from a `CFG` dict.
Every knob is an optimizer / schedule / loss / batch knob; the model is fixed.

All participant logic lives where the rules allow it: `build_model`, `build_optimizer`
(one custom `torch.optim.Optimizer` — explicitly permitted) and `training_loss`. The
evaluator keeps the loop, the backward, `grad_clip = 1` and the one-step-per-batch
cadence. `GrokOpt` implements, selectable per param group:

* **AdamW** (decoupled decay),
* **Muon** — quintic Newton–Schulz orthogonalised momentum on hidden 2-D matrices, AdamW
  on embeddings / norms, one optimizer covering every parameter exactly once,
* **Schedule-Free AdamW** (Defazio et al.),
* **Grokfast** — EMA of the gradient added back to the update, `g ← g + λ·EMA_α(g)`,
* **⊥Grad** — projecting the gradient orthogonal to the weights (Prieto et al. 2025,
  "Grokking at the Edge of Numerical Stability"),
* **row-norm projection** on the embedding matrix,
* orthogonal / small-scale init.

`training_loss` implements CE, label smoothing, focal, a *hardest-token* reweighting
(softmax over per-token losses — the all-or-nothing metric is decided by the worst token
in a sequence) and **StableMax** cross-entropy.

Runner: `lab/grok.sh <name> <steps> <seed> "<note>" <knobs…>` → `lab/run_experiment.py`
→ one row in `lab/archive.jsonl`. Tables: `lab/grok_table.py --tag <axis>`.

```bash
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
$V lab/make_manifest.py --dataset e1 --mode fixed_step --max-steps 10000 --seeds 74
TAG=A-wd lab/grok.sh A_wd1.0_bs128_lr1e-3_10k 10000 74 "wd sweep" \
    --wd 1.0 --wd-emb 1.0 --batch-size 128 --lr 0.001
```

---

<!-- RESULTS SECTIONS FILLED BELOW -->
