# One Layer Deeper — Exploration Plan (Agent Handoff)

**Repo:** `github.com/Riyuzakii/one-layer-deeper`
**Deliverable:** a single `submission.py` (≤256 KiB) that maximizes mean exact accuracy on the Hard tier.
**Audience:** an autonomous agent with the repo checked out and GPU access.

---

## 0. The one framing that matters

This is **not** an architecture competition. It is a **wall-clock optimization problem** with an architecture-shaped search space.

```
score = accuracy( steps × per-step-progress )
steps = (T_budget − T_overhead) / T_step
```

Three consequences, all of which should drive every decision below:

1. **Parameters are not the binding constraint.** The ceiling is 500M scalars, but you cannot push enough data through 500M params in 60s (Easy) or even 3600s (Hard) to train them. Spend parameters only where they buy per-step progress. Expect the winning model to be far under the ceiling.
2. **Overhead is charged to you.** Model construction, submission import, and compilation come out of the training budget. A 30s `torch.compile` is *half the Easy budget* and *free on Hard*. The optimal submission is therefore **tier-dependent** — do not assume an Easy-tier result transfers to Hard.
3. **Depth trades directly against updates.** Doubling depth halves your step count. The competition name is the hypothesis; your job is to find where the tradeoff actually breaks even.

---

## 1. Hard constraints (from the official rules)

| Constraint | Value |
|---|---|
| Model state ceiling | 500,000,000 scalar params + persistent buffers (shared counts once, frozen still counts) |
| Training budget | Easy 60s / Medium 600s / Hard 3600s of H100 time |
| Eval budget | Half the training allowance, one pass, `model.eval()` |
| Submission | one self-contained UTF-8 `submission.py`, ≤256 KiB |
| Imports | public `benchmark` API + pinned evaluator deps only — no repo `model`/`optim` modules, no extra files, no pip installs, no network |
| You control | model, depth, optimizer, LR schedule, training loss, batch size, max_steps |
| Evaluator controls | data, sampling, the one-fwd/one-bwd loop, grad clipping, optimizer cadence, seeds, deadline, final eval, aggregation |
| Forbidden | data inspection, task-specific solvers, custom training loops, participant-controlled backward, manifest overrides |
| Rate limits | Easy 60/day (e1–e5), Medium 6/day (m1–m5), Hard 1/day (h1) |
| Ranking | best successful **Hard** submission only |

**Explicitly allowed and under-exploited:** recurrence, weight tying, adaptive computation, depth curricula, memory tokens, routing, parameter-free work, custom differentiable loss over `(logits, labels, aux)`.

### Compliance rules for the agent — non-negotiable

- **Do not read, print, or statistically summarize any generated dataset.** `scripts/generate_datasets.sh` writes to `data/generated/`. Treat that directory as write-only. Every design choice must be justifiable from the public task description and general reasoning, never from observed data.
- **Do not write a solver.** No hand-coded composition logic, symbolic evaluator, or lookup structure targeting the task.
- If an idea's justification requires knowing what the data looks like, discard the idea.

---

## 2. Phase 0 — Environment and baseline (do first, do not skip)

1. Clone and install. Note: the README's clone URL (`one-layer-benchmark`) differs from the repo name (`one-layer-deeper`). **Verify which is correct before assuming a broken link.** Requires exactly Python 3.13.5.
   ```
   uv sync --extra benchmark
   uv run python -m unittest discover -s tests
   ```
2. Run the CPU smoke test with `submissions/baseline_adamw/submission.py` against `benchmark/manifests/smoke_cpu.json`. Confirm the loop end-to-end before touching a GPU.
3. **Read the evaluator source, not just the README.** Produce a written summary of:
   - `ModelSpec` / `OptimizerSpec` / `OptimizerBundle` / `Submission` / `assert_model_state` exact signatures
   - the pinned dependency set in `pyproject.toml` (this defines what you're allowed to import)
   - evaluator default `batch_size` and the `max_steps` ceiling
   - **when the clock starts** — at import, at `build_model`, or at step 1
   - the exact tensor signature the evaluator passes to `forward`, and the required type of `aux_value` when unused
   - how eval batch size is chosen and whether the eval deadline is enforced per-batch or in aggregate
   - what `vocab_size` and `max_seq_len` are for each tier
4. Reproduce the AdamW baseline on `h100_easy_e1` locally and submit it once to the hosted Easy tier. **Record both numbers.** The gap between them is your local-vs-hosted calibration error and you need it before trusting any local result.

> **Hardware warning:** if local GPUs are not H100, wall-clock results do not transfer. All timing-sensitive conclusions must be confirmed on the hosted Easy tier, which is H100 and rate-limited generously (60/day) precisely for this.

---

## 3. Phase 1 — Instrumentation

Build this before running experiments. Everything downstream depends on it.

- **Overhead probe.** A submission that does nothing but construct the model and run trivial steps. Measures import + construction + first-step compile cost as a function of model size and compile mode. Output: an overhead table per tier.
- **Step-time model.** Measure `T_step` vs (depth, width, batch size, seq len). You want a predictive formula so you can estimate step counts without a run.
- **Archive.** One row per experiment: config hash, full config, tier, dataset, wall clock, steps completed, final loss, exact accuracy, and a free-text note on what was being tested. Persist as JSONL alongside the `one-layer metrics` artifacts. **Negative results are recorded, not discarded** — the shape of the failure surface is the main asset you're building.
- **Variance floor.** Run one fixed config across all of e1–e5. The spread across datasets is your noise floor. Any effect smaller than it is not an effect. Do this before believing a single improvement.

---

## 4. Phase 2 — Exploration axes

Change **one axis at a time**. Each entry below gives a hypothesis, a test, and what would falsify it. Ordered by expected value.

### A. Minimum viable depth (highest priority)

**Hypothesis.** Function composition over *k* compositions requires roughly *k* steps of serial computation. Below a threshold depth, accuracy is near zero regardless of training; above it, returns flatten sharply. If true, the optimal strategy is: find the threshold, then spend every remaining second on update count, not depth.

**Test.** Weight-tied recurrent block, parameters held fixed, sweep iteration count over a wide range (e.g. 2, 4, 8, 16, 32, 64). Plot accuracy vs depth at fixed wall clock. Look for a knee.

**Falsified if** accuracy rises smoothly and monotonically with depth — in which case depth is genuinely the resource and the plan inverts toward maximal depth with minimal steps.

### B. Depth curricula

**Hypothesis.** Early training does not need full depth. Running shallow early (cheap, many updates) and deepening later beats fixed depth at equal wall clock.

**Test.** Weight-tied recurrence makes this a single integer that varies over training. The model can hold a scalar step-counter buffer (negligible against the state ceiling), incremented on training forwards, and set its own iteration count from it. Sweep schedules: constant, linear ramp, step ramp, and reverse (deep→shallow) as a control.

**Hazard.** The eval budget is half the training budget. A model that ends training at depth 64 must still *evaluate* at depth 64 within that budget. Measure eval cost explicitly; blowing the deadline fails the whole run.

### C. Optimizer

**Hypothesis.** AdamW is not the wall-clock optimum. Muon (or a Muon/Adam hybrid: orthogonalized updates on hidden 2D matrices, Adam on embeddings, biases, and scalars) is the current standard in fixed-clock speedrun settings and should be the first thing tried after the baseline.

**Test.** Baseline AdamW → Muon hybrid → one second-order option (SOAP/Shampoo-style) → a schedule-free variant. Score each on accuracy at fixed wall clock, and separately report per-step overhead so you can tell "better updates" from "cheaper updates."

**Note.** The evaluator fixes optimizer *cadence* and gradient clipping. Confirm from source whether that forecloses anything you're planning (e.g. multi-step inner updates).

### D. Batch size and the step-count/noise tradeoff

**Hypothesis.** There is an interior optimum. Small batches give more updates but noisier gradients and poor H100 utilization; large batches give clean gradients and few updates.

**Test.** Sweep batch size over ~4 octaves at fixed wall clock. Cheap on the Easy tier — do this early and reuse the answer. Re-check it after any change that alters step time significantly.

### E. Loss design

**Hypothesis.** The scoring metric is **exact** accuracy — all-or-nothing per sequence. Token-averaged cross-entropy is misaligned: it rewards getting most tokens right on sequences that will score zero anyway.

**Test.** Against a CE baseline, try: per-sequence weighting that concentrates on nearly-correct sequences, focal-style weighting on hard tokens, position weighting, deep supervision on intermediate recurrent states, and removal of any label smoothing. Custom loss receives `(logits, labels, aux)` and must return one differentiable finite scalar — the evaluator performs backward, so all of this must live in the loss function, not a custom loop.

### F. Adaptive computation and halting

**Hypothesis.** Not all sequences need the same depth. Per-sequence halting frees compute for more updates during training and cuts eval cost.

**Test.** ACT/PonderNet-style halting head; route the ponder cost through `aux_value` into the custom loss as a compute penalty; sweep the penalty weight. Confirm behavior differs correctly between `self.training` and eval.

**Watch for.** Halting that collapses to always-minimum or always-maximum depth. If it collapses, that is itself an informative result about axis A.

### G. Throughput engineering

**Hypothesis.** A meaningful fraction of the budget is recoverable at the kernel level, which converts directly into more updates.

**Test, in order of expected payoff:**
1. `torch.compile` mode sweep (`none` / `default` / `max-autotune`) **per tier** — the answer will differ between 60s and 3600s.
2. bf16 autocast inside the model's forward.
3. fp8 GEMMs via `torch._scaled_mm` for the large matmuls, if the pinned dep set permits and numerics hold. Transformer Engine will not be importable; plain torch only.
4. Fusion of the recurrent block's elementwise work, kernel launch reduction, memory layout.

Every change here is measured as **steps completed in the budget**, not as microbenchmark speedup.

### H. Parameter allocation

**Hypothesis.** Given the time constraint, wide-and-shallow, narrow-and-deep, and tied-recurrent models at equal step time have materially different accuracy.

**Test.** Hold `T_step` constant, vary the width/depth/tying split. This is a clean controlled comparison because the confound (step count) is pinned.

---

## 5. Phase 3 — Promotion protocol

Rate limits make discipline the difference between 30 useful Hard attempts and 30 wasted ones.

1. **Local** — unlimited. Screening only. Never trust local wall clock unless the GPU is an H100.
2. **Easy (60/day, e1–e5)** — the workhorse. An effect must hold across all five datasets and exceed the variance floor from Phase 1 before it counts.
3. **Medium (6/day, m1–m5)** — promotion gate. Confirm the effect survives a 10× budget increase. **Expect reversals here**, specifically anything overhead-related: compile modes and large models that lose on Easy will start winning.
4. **Hard (1/day, h1)** — maintain a ranked queue of Hard candidates. Submit the top of the queue daily. Never spend a Hard attempt on an untested change, and never bundle two untested changes into one attempt.

Budget the calendar backwards from the deadline: the number of Hard attempts you get is exactly the number of days remaining.

---

## 6. Known failure modes

- **Tier-transfer failure.** Tuning on Easy and assuming Hard follows. Overhead amortization inverts the ranking. Re-derive at each tier.
- **Eval-deadline failure.** Deep or adaptive models that train fine and then exceed the half-budget eval window. Always measure eval cost as a first-class metric.
- **OOM.** Optimizer state, activations, and workspace share remaining VRAM. Deep recurrence with stored activations is the obvious risk; check whether gradient checkpointing is a net win under wall clock (it trades compute for memory, which here costs steps).
- **Winner's curse.** Argmax over many noisy Easy runs selects for luck. Require effects to exceed the measured variance floor and to replicate across e1–e5.
- **Silent rule violation.** Anything that touches data, imports a repo internal, or reaches outside the file. Run `one-layer validate` before every submit; validation rejections are free, accepted-then-failed runs are not.

---

## 7. Reporting

Produce, and keep current:

- `archive.jsonl` — every run, including failures.
- `findings.md` — one section per axis in §4: hypothesis, what was run, result, whether it held at Medium, current belief. Keep the negative results with the same weight as the positive ones.
- `queue.md` — ranked Hard candidates with justification and the Medium evidence behind each.

---

## 8. Open questions to resolve in Phase 0

Answer these from the source before designing anything:

1. Does the clock start at import or at the first optimizer step?
2. What is the evaluator's `max_steps` ceiling per tier, and can it bind before the deadline does?
3. Is the eval batch size participant-controlled, and is the eval deadline per-batch or aggregate?
4. What exactly does the evaluator pass to `forward`, and what is the required form of `aux_value` when unused?
5. Does "fixed optimizer cadence" mean strictly one step per batch?
6. What is in the pinned dependency set — specifically, is anything beyond core torch available?
7. Do persistent buffers include autograd-free state used for a depth curriculum counter? (Assume yes; keep it scalar.)
8. Is `torch.compile` invoked inside `build_model` charged to construction or to the first step?