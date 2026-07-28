# One Layer Deeper — Iterative Architecture Exploration Plan

## 1. Objective

Build an autonomous research agent that searches for architectures and training strategies that solve the **One Layer Deeper** repeated modular-squaring task under the competition constraints.

The agent must optimize for:

1. Exact task accuracy.
2. Generalization to larger task depth `T` than seen during training.
3. Generalization to unseen modulus identities `N`.
4. Stable training across random seeds.
5. Efficient use of the fixed H100 wall-clock budget.
6. Simplicity and interpretability of discovered mechanisms.
7. Reproducible evidence for why each approach succeeds or fails.

The target function is:

\[
y = x^{2^T} \bmod N
\]

which can be written as the iterative transition:

\[
z_0 = x,\qquad z_{t+1} = z_t^2 \bmod N
\]

The search should favor models that learn reusable internal computation rather than memorize dataset-specific mappings.

---

## 2. Competition Constraints

The agent must preserve the evaluator contract.

- Submission consists of one self-contained `submission.py`.
- The evaluator owns:
  - data,
  - sampling,
  - training loop,
  - backward pass,
  - gradient clipping,
  - seeds,
  - final evaluation,
  - wall-clock cutoff.
- The submission may control:
  - architecture,
  - recurrence,
  - adaptive computation,
  - optimizer,
  - learning-rate schedule,
  - optional loss,
  - batch sizes,
  - maximum training steps.
- Maximum model state: 500,000,000 elements.
- Model state and computation must remain on GPU.
- No CPU offloading.
- No hard-coded solver in the forward pass.
- End-to-end differentiability is required.
- Easy budget: 60 seconds.
- Medium budget: 600 seconds.
- Hard budget: 3,600 seconds.
- Hard submissions are limited, so almost all search must happen locally on Easy and Medium.
- Depth is unconstrained, but deeper forwards reduce optimizer-update count.

The agent must treat wall-clock time, compile overhead, memory use, and optimizer-step count as first-class objectives.

---

## 3. Research Questions

The search should answer these questions, not just optimize leaderboard score.

### 3.1 Iterative computation

- Can a tied transition block learn a reusable state update?
- Does applying the block more times at inference improve accuracy?
- Does performance extend to task depths not seen in training?
- Do intermediate latent states correspond to partial task progress?

### 3.2 Adaptive computation

- Can the model infer how much internal compute is needed from `T`, `N`, and `x`?
- Can it halt early on easy inputs without harming hard inputs?
- Does adaptive depth improve wall-clock efficiency enough to increase training updates?

### 3.3 State representation

- What state format best supports modular arithmetic?
- Does the model benefit from explicit digit-local state?
- Are global memory tokens useful?
- Does a single latent vector lose important carry structure?
- Does hierarchical state improve generalization across modulus sizes?

### 3.4 Generalization

- Does a method improve:
  - seen-`N` depth generalization,
  - unseen-`N` depth generalization,
  - both,
  - neither?
- Is the model learning an algorithm, a family-specific shortcut, or interpolation?

### 3.5 Stability

- How does hidden-state norm change with recurrent depth?
- Does the recurrent Jacobian become unstable?
- Does performance degrade after too many iterations?
- Does training remain stable across seeds?

### 3.6 Search transferability

- Which discovered motifs are specific to modular arithmetic?
- Which motifs may transfer to other compositional or algorithmic tasks?

---

## 4. Core Search Axes

The agent must search along all axes below. It should vary one axis at a time in early experiments and combine axes only after isolating their effects.

---

### 4.1 State Structure

Candidate state representations:

1. **Standard token sequence**
   - Token embedding per input token.
   - Positional embeddings.
   - Shared recurrent block over the full sequence.

2. **Input tokens plus memory tokens**
   - Add 1, 2, 4, 8, or 16 learned memory slots.
   - Test whether memory slots store:
     - current value,
     - modulus,
     - iteration count,
     - global carry or summary state.

3. **Digit-grid state**
   - One latent vector per digit.
   - Separate streams for:
     - `N`,
     - `x`,
     - `T`,
     - working value.
   - Local neighbor communication plus periodic global communication.

4. **Global accumulator**
   - A small set of latent vectors summarizes all digits.
   - Digit tokens read from and write to the accumulator.

5. **Hierarchical state**
   - Digit-level tokens.
   - Chunk-level summary tokens.
   - Global state token.

6. **Two-stream state**
   - Static input stream.
   - Dynamic recurrent workspace.
   - Static stream remains unchanged.
   - Workspace is updated at each recurrent step.

7. **Slot-based state**
   - Specialized learned slots for:
     - current residue,
     - modulus,
     - task depth,
     - iteration counter,
     - temporary workspace.

8. **Compressed recurrent state**
   - Recurrent latent bottleneck smaller than token embedding width.
   - Decoder maps recurrent state back to output digits.

For every state design, record:

- parameter count,
- persistent-state size,
- activation memory,
- forward time,
- training steps achieved,
- hidden-state norms,
- exact accuracy by task depth,
- exact accuracy on unseen modulus identities.

---

### 4.2 Transition Operator

Candidate recurrent transition blocks:

1. Transformer attention + MLP.
2. Gated attention + gated MLP.
3. MLP-only token mixer.
4. Convolutional token mixer.
5. Local attention with periodic global attention.
6. Linear attention.
7. GRU-like recurrent update.
8. LSTM-like recurrent update.
9. SSM-inspired mixer.
10. Residual gated MLP.
11. Multiplicative interaction block.
12. Bilinear layer.
13. Tensor-product feature interaction.
14. Mixture-of-experts transition.
15. Routed transition with specialized arithmetic submodules.
16. Alternating transition types:
    - local update,
    - global reduction,
    - broadcast.
17. Parameter-free state transforms combined with learned residuals.
18. Learned update plus explicit normalization or projection.

The agent should compare operators under matched conditions:

- same state width,
- same number of recurrent iterations,
- similar parameter count,
- similar forward-time budget where possible.

---

### 4.3 Weight Sharing

Search:

1. Fully untied stack.
2. Fully tied recurrent block.
3. Period-2 tied blocks.
4. Period-4 tied blocks.
5. Input block + tied core + output block.
6. Tied attention with untied MLP.
7. Untied attention with tied MLP.
8. Layer groups with shared parameters.
9. Hypernetwork-generated iteration parameters.
10. Shared base weights plus iteration-specific low-rank adapters.
11. Shared block with learned iteration embeddings.
12. Shared block without explicit iteration identity.

Key comparison:

- parameter efficiency,
- training stability,
- depth extrapolation,
- unseen-`N` generalization,
- runtime.

Reject approaches that improve only by increasing parameters without improving generalization or wall-clock efficiency.

---

### 4.4 Internal Depth

Search:

1. Fixed depth.
2. Depth proportional to encoded `T`.
3. Depth proportional to `log2(T + 1)`.
4. Random depth during training.
5. Progressive depth curriculum.
6. Variable depth sampled per batch.
7. Variable depth sampled per example.
8. Test-time depth larger than train-time depth.
9. Learned adaptive halting.
10. Confidence-based halting.
11. Fixed-point convergence halting.
12. Multi-scale depth:
    - coarse iterations,
    - fine iterations.
13. Early-exit heads at every iteration.
14. Iterative refinement after a direct prediction.

The agent must produce a grid:

\[
\text{accuracy}(T_{\text{task}}, K_{\text{internal}})
\]

where:

- `T_task` is the requested modular-squaring depth,
- `K_internal` is the number of model iterations.

This grid is required for every promising recurrent model.

---

### 4.5 Communication Pattern

Search:

1. Full bidirectional attention.
2. Local digit attention.
3. Dilated local attention.
4. Ring communication.
5. Prefix-scan-style communication.
6. Hierarchical reduction and broadcast.
7. Cross-attention from workspace to immutable input.
8. Cross-attention from digits to global memory.
9. Sparse routing.
10. Alternating local and global phases.
11. Fixed communication graph.
12. Learned communication graph.
13. Shared memory bus.
14. Message passing on a digit graph.

The agent should test whether full attention is necessary or whether structured communication provides better arithmetic inductive bias.

---

### 4.6 Stability Mechanisms

Search:

1. Pre-norm.
2. Post-norm.
3. RMSNorm.
4. LayerNorm.
5. No normalization.
6. ReZero.
7. LayerScale.
8. Gated residual update:

\[
h_{k+1}=h_k+\alpha_k F_\theta(h_k)
\]

9. Input-dependent residual gate.
10. Fixed small residual coefficient.
11. Spectral normalization.
12. Weight normalization.
13. Hidden-state clipping.
14. Hidden-state renormalization.
15. Orthogonal initialization.
16. Identity-biased initialization.
17. Zero-initialized output projection.
18. Gradient checkpointing only if it improves feasible depth.
19. Auxiliary norm penalty.
20. Jacobian or contraction regularization if affordable.

Record:

- loss spikes,
- NaNs or Infs,
- gradient norm,
- activation norm per iteration,
- output entropy,
- accuracy as internal depth increases,
- whether extra iterations help or destroy the answer.

---

### 4.7 Training Curriculum

Search:

1. Train on one task depth.
2. Train on several small depths.
3. Uniform depth sampling.
4. Geometric depth sampling.
5. Curriculum from small to large `T`.
6. Anti-curriculum from large to small `T`.
7. Mixed curriculum.
8. Random recurrent unroll depth.
9. Progressive recurrent unroll depth.
10. Joint curriculum over:
    - modulus size,
    - task depth,
    - internal depth.
11. Hard-example replay.
12. Failure-depth oversampling.
13. Balanced sampling across `N`, `x`, and `T`.
14. Curriculum based on model confidence.
15. Curriculum based on exact-accuracy threshold.

The agent must distinguish:

- curriculum that improves optimization,
- curriculum that harms extrapolation by overfitting to a fixed schedule.

---

### 4.8 Objective and Auxiliary Losses

Search:

1. Standard final-token cross-entropy.
2. Label smoothing.
3. Digit-position weighted loss.
4. Confidence penalty.
5. Intermediate prediction loss.
6. Deep supervision at each recurrent step.
7. Consistency between consecutive recurrent states.
8. Monotonic improvement loss.
9. Early-exit loss.
10. Halting penalty.
11. State-norm regularization.
12. Contrastive separation of different residues.
13. Auxiliary reconstruction of:
    - `N`,
    - `x`,
    - `T`.
14. Auxiliary prediction of task-depth bucket.
15. Auxiliary carry-state prediction if a useful unsupervised proxy can be defined.
16. Teacher-student distillation from a deeper model.
17. Self-distillation across recurrent steps.
18. Multi-head objective:
    - answer head,
    - confidence head,
    - halt head.

The agent must reject auxiliary losses that improve training metrics but not exact generalization.

---

### 4.9 Optimizer

Search:

1. AdamW.
2. Adam.
3. SGD with momentum.
4. Adafactor.
5. Lion.
6. Schedule-free optimizer if available in allowed dependencies.
7. Custom optimizer only if submission rules allow it.
8. Parameter-group-specific learning rates:
   - embeddings,
   - recurrent core,
   - output head,
   - halting head.
9. Different weight decay for:
   - normalization,
   - biases,
   - recurrent core.
10. Gradient accumulation only if evaluator contract permits it indirectly.
11. Lookahead or EMA only if allowed and worth state cost.

Measure optimizer value in terms of:

- accuracy per wall-clock second,
- optimizer-state memory,
- step count,
- seed stability,
- final generalization.

---

### 4.10 Learning-Rate Schedule

Search:

1. Constant learning rate.
2. Linear warmup + cosine decay.
3. Linear warmup + constant.
4. One-cycle.
5. Exponential decay.
6. Step decay.
7. Inverse square root.
8. Warm restart.
9. Depth-aware schedule.
10. Curriculum-stage-aware schedule.

Every learning-rate experiment must record:

- effective number of steps,
- learning rate at final step,
- best validation point,
- whether the final checkpoint is worse than an earlier one.

Since final evaluation uses the last checkpoint, avoid schedules that peak early and then regress.

---

### 4.11 Width, Capacity, and Parameter Allocation

Search:

1. Model width.
2. MLP expansion ratio.
3. Head count.
4. Head dimension.
5. Number of memory tokens.
6. Recurrent-state width.
7. Static-input width versus dynamic-workspace width.
8. Number of experts.
9. Embedding sharing.
10. Output-head tying.
11. Low-rank projections.
12. Narrow recurrent core with wide input/output adapters.
13. Wide recurrent core with minimal adapters.
14. State-space size versus iteration count.

Plot Pareto curves for:

- accuracy,
- `T_max`,
- OOD-`N T_max`,
- runtime,
- parameters,
- optimizer steps.

---

### 4.12 Input Encoding

Search:

1. Existing decimal digit tokenization.
2. Learned positional embeddings.
3. Sinusoidal positions.
4. Rotary positions.
5. Relative positions.
6. Field-specific embeddings for `N`, `x`, `T`.
7. Digit-position embeddings within each field.
8. Least-significant-digit-first internal reordering.
9. Most-significant-digit-first.
10. Bidirectional dual order.
11. Chunked digits.
12. Binary or mixed-radix latent encoding derived inside the model.
13. Separate encoders for `N`, `x`, and `T`.
14. Explicit length tokens.
15. Learned numeric magnitude embeddings.

Do not alter evaluator inputs outside the model. Any re-encoding must remain differentiable and occur in the submitted model.

---

### 4.13 Output Decoding

Search:

1. Independent digit prediction.
2. Autoregressive digit prediction if allowed by the one-forward interface.
3. Iterative output refinement.
4. Shared output head.
5. Position-specific output heads.
6. Output confidence head.
7. Error-correcting output representation.
8. Least-significant-digit-first internal decoding with remapping to evaluator order.
9. Multi-pass decoding using recurrent state.

Track exact-example accuracy, not only per-digit accuracy.

---

### 4.14 Routing and Modularity

Search:

1. Input-conditioned expert routing.
2. Task-depth-conditioned routing.
3. Modulus-size-conditioned routing.
4. Iteration-conditioned routing.
5. Separate local and global experts.
6. Separate arithmetic and communication experts.
7. Top-1 versus top-2 routing.
8. Soft routing.
9. Hard routing with straight-through estimator.
10. Shared expert plus specialist experts.
11. Recurrent expert selection.
12. Learned phase machine.

Reject modular systems that only memorize dataset partitions.

---

### 4.15 Adaptive Halting

Search:

1. Fixed number of iterations.
2. Learned scalar halt probability.
3. Per-token halt probability.
4. Global memory-token halt.
5. Confidence-threshold halt.
6. State-change threshold halt.
7. Entropy-threshold halt.
8. Maximum-depth cap with halt penalty.
9. ACT-style ponder cost.
10. Halt decision conditioned on `T`.
11. Halt decision not directly conditioned on `T`.
12. Training-time stochastic halting.
13. Deterministic evaluation halting.

Record:

- average iterations,
- iteration distribution by task depth,
- accuracy versus compute,
- whether the model learns trivial always-max-depth behavior,
- whether halt decisions generalize to unseen `T`.

---

## 5. Required Baselines

The agent must establish these baselines before broad search.

### B0. Official one-block Transformer

Use the repository baseline unchanged.

### B1. Deeper untied Transformer

Match the official block structure with 2, 4, 8, and 16 blocks.

### B2. Tied recurrent Transformer

Reuse one block for 2, 4, 8, 16, and 32 iterations.

### B3. Tied recurrent Transformer with residual gate

Use one shared scalar or vector gate initialized near zero.

### B4. Tied recurrent Transformer with random unroll depth

Train with a sampled internal depth.

### B5. Tied recurrent Transformer with memory tokens

Use 2, 4, and 8 working-memory tokens.

### B6. MLP-only recurrent model

Use token mixing plus channel mixing without attention.

### B7. GRU-style recurrent workspace

Use static input encoding and a recurrent global workspace.

All later experiments should compare against the strongest baseline under the same dataset and budget.

---

## 6. Experimental Stages

---

### Stage 0: Infrastructure Validation

Goal: prove the experiment loop is correct.

Tasks:

1. Clone and install the repository.
2. Run unit tests.
3. Generate public datasets.
4. Run CPU smoke test.
5. Run the official baseline.
6. Confirm metrics parsing.
7. Confirm wall-clock measurement.
8. Confirm model-state validation.
9. Confirm reproducibility for fixed seed.
10. Confirm automatic artifact collection.

Exit criteria:

- Two identical baseline runs produce consistent metrics.
- All logs and artifacts are stored correctly.
- Failed runs are classified correctly.
- The agent can regenerate a prior experiment from its record.

---

### Stage 1: Baseline Characterization

Goal: understand the benchmark before searching.

Run the official baseline on:

- E1,
- E2,
- E3,
- E4,
- E5.

Then run on M1–M5 as resources permit.

Record:

- train loss,
- test exact accuracy,
- `T_max`,
- OOD-`N T_max`,
- optimizer steps,
- examples per second,
- forward time,
- backward time,
- memory use,
- compile time,
- parameter count.

Exit criteria:

- Dataset difficulty is ranked.
- Failure modes are known.
- A primary development dataset is selected.
- A secondary transfer dataset is selected.

Recommended initial pair:

- E1 or E2 for fast debugging.
- E5 for joint `N/T` conditioning.
- M5 for serious comparison.

---

### Stage 2: Depth and Weight Sharing

Goal: isolate the effect of recurrent depth.

Run:

- untied depth sweep,
- tied depth sweep,
- periodic-sharing sweep,
- gated recurrent sweep.

Control:

- width,
- optimizer,
- batch size,
- training budget.

Required outputs:

- accuracy versus internal depth,
- optimizer steps versus internal depth,
- state norm versus internal depth,
- seen-`N` and unseen-`N` depth profile.

Reject a recurrent design if:

- it achieves fewer useful optimizer updates without improving `T_max`,
- extra iterations reduce accuracy,
- seed variance is high,
- OOD-`N` does not improve.

---

### Stage 3: State Representation

Goal: identify the best workspace structure.

Compare:

- token-only,
- token + memory,
- static input + dynamic workspace,
- digit-grid,
- hierarchical state.

Use the strongest recurrent transition from Stage 2.

Exit criteria:

- one state representation dominates on a Pareto basis,
- or multiple representations are retained for different regimes.

---

### Stage 4: Training Strategy

Goal: determine whether depth extrapolation requires curriculum.

Search:

- task-depth curriculum,
- recurrent-depth curriculum,
- mixed depth sampling,
- intermediate losses,
- optimizer and schedule.

Required ablations:

- same architecture with and without curriculum,
- same architecture with and without deep supervision,
- same architecture with fixed and random unroll.

Reject training tricks that do not survive architecture transfer.

---

### Stage 5: Adaptive Computation

Goal: improve compute allocation.

Starting from the strongest fixed-depth recurrent model:

1. Add halt head.
2. Add confidence estimate.
3. Add ponder cost.
4. Compare fixed maximum depth against learned average depth.
5. Evaluate at unseen task depths.

Retain adaptive halting only if it improves:

- accuracy at fixed wall-clock budget,
- or wall-clock cost at fixed accuracy.

---

### Stage 6: Transition Search

Goal: move beyond standard Transformer blocks.

Search:

- GRU-like,
- convolutional,
- MLP-only,
- local/global alternating,
- multiplicative,
- routed modular blocks.

Do not search all operators at full scale. Use a tournament:

1. Small-width, short-budget screening.
2. Medium-width confirmation.
3. Multi-seed validation.
4. Transfer to second dataset.
5. Promote only robust candidates.

---

### Stage 7: Architecture Composition

Goal: combine independently validated gains.

Possible composition:

- static input encoder,
- digit-local transition,
- global memory bus,
- tied recurrent core,
- gated residual,
- random unroll curriculum,
- adaptive halt head,
- intermediate supervision.

Combine one validated change at a time.

Do not merge several untested changes in one experiment.

---

### Stage 8: Hard-Tier Preparation

Goal: minimize wasted Hard submissions.

Before a Hard submission, require:

- strong Medium result,
- multi-seed stability,
- no regression on unseen modulus identities,
- valid runtime margin,
- no OOM risk,
- no dependence on public-dataset quirks,
- final-checkpoint stability,
- clean standalone `submission.py`,
- local validation pass.

Hard submissions should test major hypotheses, not hyperparameter noise.

---

## 7. Autonomous Agent Loop

The agent must execute this loop continuously.

### Step 1: Read current research state

Load:

- experiment registry,
- current Pareto frontier,
- unresolved hypotheses,
- prior failures,
- remaining compute budget,
- competition deadline,
- current best submission.

### Step 2: Select one hypothesis

Examples:

- “A small residual gate will stabilize tied recurrence at 32 iterations.”
- “Static input plus dynamic memory will improve OOD modulus generalization.”
- “Random unroll depth will reduce overfitting to a fixed iteration count.”

The hypothesis must be falsifiable.

### Step 3: Propose a minimal experiment

The experiment should change as few variables as possible.

State:

- independent variable,
- controlled variables,
- expected outcome,
- rejection threshold,
- compute cost,
- dataset,
- seed count.

### Step 4: Static validation

Before running:

- validate syntax,
- validate model-state limit,
- estimate activation memory,
- estimate optimizer-state memory,
- estimate forward time,
- check submission contract,
- check unsupported imports,
- check deterministic seeding assumptions.

### Step 5: Run pilot

Use:

- Easy dataset,
- one seed,
- small model,
- short budget.

Stop and reject immediately for:

- invalid submission,
- OOM,
- NaN,
- severe slowdown,
- obvious non-learning,
- worse result than baseline by a large margin.

### Step 6: Run controlled comparison

If pilot passes:

- run baseline and candidate under matched conditions,
- use at least three seeds for serious candidates,
- collect depth profiles.

### Step 7: Analyze

The agent must answer:

1. Did the intervention improve the target metric?
2. Was the gain due to more parameters?
3. Was the gain due to more optimizer steps?
4. Did it improve seen-`N` only?
5. Did it improve unseen-`N`?
6. Did it improve task-depth extrapolation?
7. Did it improve internal-depth extrapolation?
8. Did it remain stable across seeds?
9. Did it harm runtime?
10. Did it add complexity worth keeping?

### Step 8: Decide

One of:

- **Accept**: clear robust gain.
- **Reject**: clear failure.
- **Inconclusive**: noisy or underpowered.
- **Refine**: promising but flawed.
- **Archive**: useful only as a diagnostic.
- **Combine later**: isolated gain that should not yet be merged.

### Step 9: Record full account

The agent must write a detailed experiment report before starting the next experiment.

### Step 10: Update research state

Update:

- best model,
- Pareto frontier,
- hypothesis list,
- failure taxonomy,
- next recommended experiment,
- compute ledger.

---

## 8. Experiment Record Format

Each experiment must have its own directory:

```text
experiments/
  EXP-0001-baseline-e1/
    config.yaml
    submission.py
    command.sh
    environment.txt
    metrics.json
    metrics.jsonl
    stdout.log
    stderr.log
    report.md
    plots/
    checkpoints/
```

Not all evaluator runs will expose checkpoints. Store them only when local infrastructure permits.

---

### 8.1 Experiment Metadata

Each `report.md` must include:

```markdown
# EXP-XXXX: Experiment title

## Status
accepted | rejected | inconclusive | refined | archived

## Date

## Parent experiment

## Hypothesis

## Motivation

## Independent variable

## Controlled variables

## Dataset

## Training budget

## Seeds

## Architecture summary

## Parameter count

## Persistent-state elements

## Optimizer

## Learning-rate schedule

## Batch size

## Internal depth

## Expected result

## Rejection criteria

## Raw results

## Normalized comparison

## Seen-N depth profile

## OOD-N depth profile

## Runtime profile

## Stability profile

## Interpretation

## Confounders

## Why accepted or rejected

## What this teaches us

## Follow-up experiment
```

---

### 8.2 Required Numeric Fields

Store in `metrics.json`:

```json
{
  "experiment_id": "EXP-XXXX",
  "parent_id": "EXP-XXXX",
  "git_commit": "",
  "dataset": "",
  "tier": "",
  "seed": 0,
  "status": "",
  "parameter_count": 0,
  "model_state_elements": 0,
  "optimizer_state_bytes": 0,
  "peak_vram_bytes": 0,
  "training_time_seconds": 0.0,
  "evaluation_time_seconds": 0.0,
  "optimizer_steps": 0,
  "examples_seen": 0,
  "examples_per_second": 0.0,
  "train_loss_final": 0.0,
  "train_loss_best": 0.0,
  "test_exact_accuracy": 0.0,
  "max_t": 0,
  "ood_n_max_t": 0,
  "grad_norm_max": 0.0,
  "activation_norm_max": 0.0,
  "nan_count": 0,
  "mean_internal_iterations": 0.0,
  "max_internal_iterations": 0,
  "notes": ""
}
```

---

## 9. Failure Taxonomy

Every rejection must use one or more tags.

### F01: Invalid submission

Contract, import, shape, or API violation.

### F02: OOM

Model state, optimizer state, activations, or temporary workspace exceed VRAM.

### F03: Timeout

Compilation or execution exceeds evaluator budget.

### F04: No learning

Loss and accuracy remain near random.

### F05: Optimization instability

NaNs, Infs, exploding gradients, severe oscillation.

### F06: Recurrent state instability

Hidden states explode, vanish, collapse, or oversmooth over iterations.

### F07: Extra-depth degradation

More internal iterations reduce accuracy.

### F08: Seen-`N` overfit

Seen modulus identities improve, unseen identities do not.

### F09: Task-depth overfit

Training depths improve, larger depths do not.

### F10: Iteration-count overfit

Model works only at the trained internal depth.

### F11: Runtime regression

Accuracy gain does not justify lost optimizer steps.

### F12: Parameter inefficiency

Gain comes only from a large capacity increase.

### F13: Seed fragility

Results vary too much across seeds.

### F14: Final-checkpoint regression

Best performance occurs earlier, but the final model is worse.

### F15: Dataset-specific shortcut

Gain does not transfer to a second dataset.

### F16: Auxiliary-loss mismatch

Auxiliary metrics improve while exact accuracy does not.

### F17: Trivial halting

Adaptive model always halts immediately or always uses maximum depth.

### F18: Routing collapse

One expert dominates or experts map to dataset partitions.

### F19: Measurement failure

Instrumentation or metric collection is unreliable.

### F20: Inconclusive

Effect size is smaller than run-to-run noise.

---

## 10. Rejection Rules

The agent must reject an approach when any of these holds, unless there is a clear diagnostic reason to retain it.

1. It fails to beat the matched baseline.
2. It improves only per-digit accuracy, not exact accuracy.
3. It improves seen-`N` but harms OOD-`N`.
4. It improves accuracy but reduces optimizer steps enough to lose under fixed wall-clock budget.
5. It requires much larger parameter count for a marginal gain.
6. It is unstable across seeds.
7. It works only on one Easy dataset.
8. It fails when internal depth changes.
9. It fails when task depth changes.
10. It gains from a suspected public-data shortcut.
11. Its complexity prevents reliable standalone submission.
12. It approaches memory or runtime limits without sufficient margin.

Rejected approaches must remain documented. Do not delete failed code or results.

---

## 11. Acceptance Rules

An approach can enter the active search branch only if:

1. It beats its parent under matched conditions.
2. It survives at least three seeds for medium-confidence claims.
3. It improves or preserves OOD-`N`.
4. It improves the target depth metric or wall-clock efficiency.
5. It does not introduce severe instability.
6. It reproduces on a second dataset.
7. The mechanism has a plausible explanation.
8. The code remains competition-valid.

---

## 12. Statistical Discipline

For screening:

- one seed is acceptable.

For promotion:

- at least three seeds.

For final claims:

- five or more seeds where feasible.

Report:

- mean,
- median,
- standard deviation,
- minimum,
- maximum,
- paired difference versus baseline.

Do not promote a model based on a single lucky seed.

When the metric is discrete, such as `T_max`, also compare:

- exact accuracy at each rung,
- number of examples failed,
- consistency of the consecutive prefix.

---

## 13. Pareto Frontier

Maintain a frontier over:

- `T_max`,
- OOD-`N T_max`,
- exact accuracy,
- wall-clock time,
- optimizer steps,
- parameter count,
- peak VRAM,
- mean internal iterations,
- seed variance.

A model is dominated if another model is at least as good on all selected objectives and better on one.

Do not keep only the single highest-scoring model. Preserve diverse frontier candidates:

- fastest,
- smallest,
- most stable,
- best OOD,
- deepest,
- most interpretable.

---

## 14. Search Policy

### 14.1 Exploration versus exploitation

Allocate search budget roughly as:

- 50% local improvements around the current frontier.
- 30% new architecture families.
- 10% optimizer and schedule sweeps.
- 10% diagnostics and replication.

Change this allocation if progress stalls.

### 14.2 Novelty pressure

The agent should avoid repeating near-identical experiments.

Before proposing an experiment, compare it against the registry.

Reject duplicate proposals unless:

- prior run was invalid,
- prior result was noisy,
- a new control resolves a confounder.

### 14.3 Complexity penalty

Prefer simpler models when performance is close.

A useful internal score is:

\[
S =
w_1 T_{\max}
+
w_2 T_{\max}^{OOD-N}
+
w_3 A_{\text{exact}}
-
w_4 \log(\text{runtime})
-
w_5 \log(\text{parameters})
-
w_6 \sigma_{\text{seed}}
\]

Do not use this scalar as the only decision rule. Keep the full Pareto view.

---

## 15. Diagnostics for Learned Iteration

For promising recurrent models, run the following diagnostics.

### 15.1 Intermediate decoding

Attach a shared or probe-only head to every recurrent state.

Measure whether iteration `k` predicts a meaningful partial result.

### 15.2 Iteration truncation

Run the same trained model with fewer recurrent steps.

Plot exact accuracy against internal depth.

### 15.3 Iteration extension

Run with more recurrent steps than used during training.

Check:

- continued improvement,
- saturation,
- degradation,
- divergence.

### 15.4 State trajectory

Record:

- state norm,
- cosine similarity between consecutive states,
- update norm,
- attention entropy,
- memory-token specialization.

### 15.5 Input intervention

Change one field at a time:

- `x`,
- `N`,
- `T`.

Observe which parts of the latent state change.

### 15.6 Task-depth intervention

Hold `N` and `x` fixed while varying `T`.

Determine whether the model:

- executes more computation,
- changes routing,
- changes halting,
- or directly maps from `T` to an output shortcut.

### 15.7 Modulus transfer

Hold bit width constant but use unseen modulus identities.

This is required to detect family memorization.

### 15.8 Iteration permutation

For periodic or modular blocks, alter the order of substeps.

If performance collapses, infer phase specialization.

---

## 16. Search Database

Maintain a machine-readable registry:

```text
research/
  registry.jsonl
  frontier.json
  hypotheses.md
  failures.md
  current_best.md
  compute_ledger.csv
  dataset_notes.md
  architecture_catalog.md
```

Each registry row should include:

```json
{
  "experiment_id": "EXP-XXXX",
  "parent_id": "EXP-XXXX",
  "status": "accepted",
  "hypothesis": "",
  "changes": [],
  "datasets": [],
  "seeds": [],
  "metrics_path": "",
  "report_path": "",
  "failure_tags": [],
  "next_actions": []
}
```

---

## 17. Agent Output After Every Iteration

The agent must print a compact end-of-run summary:

```text
Experiment: EXP-XXXX
Hypothesis: ...
Result: accepted | rejected | inconclusive
Main metric change: ...
OOD-N change: ...
Runtime change: ...
Failure tags: ...
Reason: ...
Next experiment: ...
```

It must also write the full report to disk.

---

## 18. Initial Experiment Queue

Run in this order unless evidence forces a change.

1. Official baseline on E1.
2. Official baseline on E5.
3. Untied 2-block Transformer.
4. Untied 4-block Transformer.
5. Tied 2-iteration Transformer.
6. Tied 4-iteration Transformer.
7. Tied 8-iteration Transformer.
8. Tied 16-iteration Transformer.
9. Tied 8-iteration Transformer with ReZero gate.
10. Tied 16-iteration Transformer with ReZero gate.
11. Random unroll depth `K ∈ [1, 8]`.
12. Random unroll depth `K ∈ [1, 16]`.
13. Add 4 memory tokens.
14. Split static input and recurrent workspace.
15. Add intermediate shared prediction head.
16. Add progressive unroll curriculum.
17. Compare AdamW schedules.
18. Compare MLP-only recurrent transition.
19. Compare GRU-style recurrent workspace.
20. Transfer top three candidates from E1/E5 to M5.

After this queue, choose experiments based on recorded evidence.

---

## 19. Suggested Repository Layout

```text
one-layer-agent/
  plan.md
  agent/
    propose.py
    validate.py
    run.py
    analyze.py
    decide.py
    report.py
    registry.py
  templates/
    submission_base.py
    report_template.md
    config_template.yaml
  architectures/
    transformer.py
    recurrent_transformer.py
    recurrent_mlp.py
    recurrent_gru.py
    memory_workspace.py
  experiments/
  research/
  scripts/
    run_local.sh
    run_easy.sh
    run_medium.sh
    compare.py
    plot_depth_grid.py
    plot_frontier.py
  tests/
```

---

## 20. Agent Guardrails

The agent must not:

1. Hard-code modular arithmetic in the forward pass.
2. Use hidden evaluator information.
3. Exploit metric recording.
4. Inspect prohibited data.
5. alter the evaluator-owned training loop.
6. hide failed runs.
7. discard negative results.
8. claim algorithmic generalization from seen-`N` results alone.
9. claim success from per-digit accuracy.
10. use Hard submissions for routine tuning.
11. exceed the model-state ceiling.
12. rely on CPU offloading.
13. optimize only for one public dataset.
14. merge several changes without ablation.
15. overwrite experiment artifacts.

---

## 21. Stopping Conditions

Stop a branch when:

1. Five consecutive refinements fail to improve the frontier.
2. The branch repeatedly fails OOD-`N`.
3. Runtime scaling makes larger depth infeasible.
4. Seed variance remains high after stabilization attempts.
5. The approach depends on dataset-specific behavior.
6. The architecture cannot fit the standalone submission contract.
7. A competing branch dominates it across all relevant objectives.

Stop the overall search when:

1. Competition deadline is too close for another full validation cycle.
2. Compute budget is exhausted.
3. The best model has been reproduced and packaged.
4. Remaining hypotheses have low expected value.
5. Hard-tier submission slots must be reserved for final candidates.

---

## 22. Final Deliverables

The agent must produce:

1. Best standalone `submission.py`.
2. Reproducible experiment registry.
3. Full record of accepted and rejected approaches.
4. Pareto frontier plots.
5. Depth-generalization plots.
6. Seen-`N` versus OOD-`N` comparison.
7. Runtime and optimizer-step analysis.
8. Stability analysis.
9. Architecture family comparison.
10. Final research report explaining:
    - what worked,
    - what failed,
    - why,
    - what appears task-specific,
    - what may transfer to broader neural architecture discovery.

---

## 23. Definition of Success

The project succeeds only if it produces more than a leaderboard entry.

A strong result should show:

1. A model that performs iterative latent computation.
2. Better generalization to larger `T`.
3. Better generalization to unseen `N`.
4. A measured compute-versus-accuracy tradeoff.
5. Stable behavior when recurrent depth is extended.
6. Evidence for which architectural motifs caused the gain.
7. A complete negative-results record that prevents repeated dead ends.
8. A search procedure reusable for broader open-ended architecture discovery.

