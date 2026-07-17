# One Layer Deeper

An architecture-and-optimizer competition from **Core Automation × Tilde Research**.

Build the best function-composition model under a fixed persistent-state ceiling and H100 training-time budget. Participants control architecture, depth, optimizer, learning-rate schedule, and training loss. The evaluator controls data, the outer loop, and final evaluation.

For competition updates, join [discord.gg/gpumode](https://discord.gg/gpumode) and follow the `#one-layer-deeper` channel.

## Acknowledgements

We are grateful to [Modal](https://modal.com/) for supporting the GPU evaluation infrastructure and to [Northflank](https://northflank.com/) for supporting the competition service and leaderboard.

## Install the CLI

Note, this is the submission CLI only go to [Local development](#local-development) to develop locally.
Install [uv](https://docs.astral.sh/uv/) and then install the command directly from GitHub:

```bash
uv tool install git+https://github.com/tilde-research/one-layer-benchmark.git
one-layer --help
```

The CLI installation is lightweight and does not install PyTorch or the local evaluator. Upgrade it with:

```bash
uv tool upgrade one-layer-benchmark
```

## Participant flow

```bash
one-layer login
one-layer validate submission.py
one-layer submit submission.py --tier easy --dataset e1 --wait
one-layer jobs
one-layer status <submission-id>
one-layer leaderboard
```

`one-layer login` opens GitHub authentication, receives a generated `old_…` API key through a temporary localhost callback, and saves it to `~/.config/one-layer/config.json` with user-only permissions. Signing in again rotates a lost key. The service stores the GitHub identity plus only the key's SHA-256 digest and short support prefix.

The CLI defaults to the [hosted leaderboard](https://http--one-layer-deeper--7v28wph27ynb.code.run). Set `ONE_LAYER_URL` or pass `--server` to use another compatible endpoint. Set `ONE_LAYER_API_KEY` or pass `--api-key` instead of saving a key locally.

## Official rules

1. Submit exactly one UTF-8 file named `submission.py`. It exports one `benchmark.Submission` with model and optimizer factories and an optional training loss.
2. The submission must be self-contained. It may import the public `benchmark` API and pinned evaluator dependencies, but it may not depend on repository `model` or `optim` modules, extra files, package installation, or external services.
3. Participant code defines the model, optimizer bundle, optional learning-rate scheduler, optional loss, batch size, and maximum training steps. Recurrence, adaptive computation, and depth curricula are allowed.
4. The evaluator fixes data, sampling, the one-forward/one-backward loop, gradient clipping, optimizer cadence, seeds, deadline, final evaluation, and aggregation. Participants may choose the training and evaluation batch size and a lower maximum step count; evaluator ceilings still apply.
5. The model may contain at most 500,000,000 scalar parameters and persistent buffers. Shared state counts once; frozen state still counts.
6. Optimizer state, activations, and temporary workspace may use remaining VRAM. OOM or timeout fails the run.
7. Easy provides 60 H100 training seconds, Medium 600 seconds, and Hard 3,600 seconds. Model construction, submission import, and compilation consume the budget.
8. A custom training loss receives final logits, labels, and the model's auxiliary output and returns one differentiable finite scalar. The evaluator performs backward.
9. Each final checkpoint is evaluated once with a separate time budget equal to half its training allowance. The evaluator uses fixed loss and exact accuracy, and the score is mean exact accuracy across fixed evaluation splits and seeds.
10. Data inspection, task-specific solvers, custom training loops, participant-controlled backward passes, and manifest overrides are not allowed.

Depth is deliberately unconstrained. Fixed stacks, tied recurrence, iterative refinement, routing, adaptive halting, memory tokens, and parameter-free work are all valid if the model-state ceiling is respected. A deeper forward completes fewer optimizer updates under the same clock.

## Submission contract

The file is limited to 256 KiB. `build_model(spec)` receives `vocab_size`, `max_seq_len`, and `maximum_model_state_elements`. It returns a `torch.nn.Module` whose `config` exposes the first two matching fields. The model accepts evaluator tensor arguments and returns `(logits, auxiliary_value)`.

The evaluator calls `model.train()` for optimization and `model.eval()` for final evaluation. Use PyTorch's inherited `self.training` flag if the model should behave differently during evaluation.

```python
from benchmark import ModelSpec, OptimizerBundle, OptimizerSpec, Submission, assert_model_state

def build_model(spec: ModelSpec):
    model = MyModel(spec)
    assert_model_state(model, spec)
    return model

def build_optimizer(model, spec: OptimizerSpec) -> OptimizerBundle:
    return OptimizerBundle(MyOptimizer(model.parameters()))

SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=512,       # optional; applies to training and evaluation
    max_steps=20_000,     # optional; cannot exceed the evaluator ceiling
)
```

If omitted, `batch_size` and `max_steps` use evaluator defaults. An optional scheduler returned in `OptimizerBundle` is stepped after every completed optimizer update. A standalone AdamW example is available under `submissions/`.

## Compute tiers

- **Easy:** datasets `e1`–`e5`, 60 training seconds, 60 accepted attempts per UTC day.
- **Medium:** datasets `m1`–`m5`, 600 training seconds, 6 accepted attempts per UTC day.
- **Hard:** dataset `h1`, 3,600 training seconds, 1 accepted attempt per UTC day.

Easy and Medium are practice tiers. The public leaderboard ranks only each participant's best successful Hard submission. Failed evaluations count after acceptance; authentication and validation rejections do not. Source and detailed results remain private.

## Local development

Clone the repository and install the benchmark extra:

```bash
git clone https://github.com/tilde-research/one-layer-benchmark.git
cd one-layer-benchmark
uv sync --extra benchmark
uv run python -m unittest discover -s tests
```

The package requires exactly Python 3.13.5 to match the hosted evaluator.

Run the short CPU smoke test without generating any datasets:

```bash
uv run python -m benchmark.runner \
  --manifest benchmark/manifests/smoke_cpu.json \
  --submission-file submissions/baseline_adamw/submission.py
```

Generate the tier datasets before running an H100 manifest:

```bash
uv run bash scripts/generate_datasets.sh
```

The generated data lives under `data/generated/` and is intentionally ignored by Git. To run on a local H100, identify an idle GPU and expose only that device:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m benchmark.runner \
  --manifest benchmark/manifests/h100_easy_e1.json \
  --submission-file submissions/baseline_adamw/submission.py
```

Hard evaluation is available only through hosted submission. The final `RESULT_JSON=...` line contains aggregate and split metrics. Local results never update the hosted leaderboard.

## License

Licensed under the [Apache License 2.0](LICENSE).
