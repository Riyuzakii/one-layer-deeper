# Exploration brief — read this before anything else

You are one of several agents searching for a competition-winning `submission.py`
for **One Layer Deeper**. Each agent owns one hypothesis family and one branch.
This file is the shared ground truth. `plan_old.md` (repo root) is the long-form
search methodology; this file is what supersedes and specialises it.

---

## 1. The metric changed on 2026-07-24. Most of `lab/findings.md` targets the old one.

Commit `79f0a09` ("Updating scoring to be based on extrapolation in T") replaced the
ranking metric. The **Hard leaderboard now ranks by**:

1. **Max T** — the largest `T ∈ {1,2,4,8,16,32,64}` such that that rung *and every
   lower rung* have **100% exact-example accuracy** on fresh prompts using modulus
   identities seen in training.
2. tie-break: **OOD N Max T** — same ladder over *unseen* modulus identities.
3. tie-break: earlier submission time.

Read `benchmark/runner.py:429-497` (`_evaluate_depth_profile`) for the exact
semantics: a rung is `certified` only if `correct_examples == example_count`, and
certification must form a consecutive prefix from T=1 upward.

**Consequences that reframe everything:**

- `mean_exact_accuracy` is now a *diagnostic only*. A model at 5% mean accuracy
  scores **Max T = 0**, identical to an untrained model. There is no partial credit.
- The entire prior lab session (`lab/findings.md`, `lab/queue.md`) optimised mean
  accuracy and concluded there is a "robust ~1-5% plateau". That plateau is a **score
  of zero** under the current rules. Its *diagnostic* content is still valuable — read
  it for what has already been ruled out — but do not treat its recommendations as
  the current strategy.
- The first real milestone is **certifying T=1 at all**: 100% exact on every held-out
  prompt at T=1. Nobody has done that yet. On `e1` a rung is 38 examples, so one
  wrong example means `Max T = 0`.
- This is now an **exact algorithmic generalization** problem, not an accuracy-shaving
  problem. Wall-clock efficiency only matters once something generalizes exactly.

## 2. The task, from the public generator source (compliant to read; the data is not)

`data/squaring_mod.py` generates repeated modular squaring. The prompt encodes
`(N, x, T)` and the target is

```
y = x^(2^T) mod N        computed as pow(x, pow(2, T, phi), N),  phi = (p-1)(q-1)
```

i.e. apply `s(y) = y² mod N` exactly `T` times. `N = p·q` is a semiprime.

Structural facts worth designing around (all derived from the generator source and
elementary number theory, never from data):

- **Composition depth = T.** A single learned squaring step, applied T times, solves
  every rung — this is the literal reading of the competition name.
- **The exponent is periodic in T.** `2^T mod phi` is eventually periodic with a small
  period, so high-T rungs collapse onto a handful of distinct exponents. A model that
  conditions its computation on T (rather than iterating T times) can in principle
  certify T=64 without 64 serial steps.
- **The units group is abelian.** `Z*_N ≅ Z_(p-1) × Z_(q-1)`; squaring is doubling in
  the exponent. Representations in which squaring becomes a *rotation* (Fourier /
  complex features — the classical grokking solution to modular arithmetic) give any T
  at constant depth. Nothing here may be hard-coded; the point is to choose
  architectures whose inductive bias makes such a solution *learnable*.
- **Training sets are tiny relative to the space.** `e1` has ~600 training rows total
  (250 prompts/setting × 0.8 × 3 values of T) against N=323. This is the classic
  grokking regime: small data, exact structure, delayed generalization under weight
  decay. Optimisation strategy should be considered in that light.
- **Depth rungs use held-out prompts.** For fixed-N Easy datasets the profile cohort
  is the *complement* of every (N,x) pair used in train/test/ood (`e1`: 38 x values).
  For sampled-N datasets it is fresh (N,x) pairs from training moduli.
- **Prompt format** (`tokenize_squaring_mod_with_result`, `separate_input_output=True`):
  `[N] <digits of N> [X] <digits of x> [T] <digits of T>`, most-significant digit
  first, vocab 17 (PAD/BOS/N/X/T/ANS/EOS + 10 digits). Targets are the digits of the
  answer, supervised at the *last* `len(answer)` positions of the prompt via
  `target_positions`. The evaluator supplies a padding mask, so attention is
  **bidirectional** over the whole prompt.

## 3. Evaluator contract (source-verified; re-verify anything you depend on)

- **Clock starts at import** and is backdated before `build_model`
  (`runner.py:539-542`): import + construction + `.to(device)` + `build_optimizer` +
  any `torch.compile` all come out of the training budget.
- **Exactly one `optimizer.step()` per batch**, evaluator-owned (`runner.py:316-338`).
  Arbitrary recurrent compute *inside one forward* is allowed and is where depth lives.
- `forward(input_ids: int64[B,L], attention_mask: bool[B,L]) -> (logits[B,L,17], aux)`.
  Return `aux=None` when unused. A custom `training_loss(logits[valid], labels[valid],
  aux)` must return one finite differentiable scalar; the evaluator calls `.backward()`.
- **Eval budget = half the training budget**, aggregate, deadline checked per batch.
  It must cover `test` + `ood` + 7 seen-N rungs + 7 OOD-N rungs. A rung that times out
  is `not_completed` and ends certification there — so a very deep model can be fast
  enough to train and still score 0 by running out of eval clock.
- Model state ceiling: 500,000,000 elements = parameters + **persistent** buffers
  (tied tensors counted once, frozen state counted). **Non-persistent buffers are
  excluded** (`api.py:26-42`) — that is the free place for a step counter.
- Importable: `torch==2.12.1`, `numpy==2.5.0`, and the public `benchmark` API. Nothing
  else. One UTF-8 file, ≤256 KiB.
- Manifests: Easy 60s / Medium 600s / Hard 3600s, bf16 + amp, `compile=false`,
  `grad_clip=1`, batch 512, one seed `[74]`, `max_steps` ceiling 1e6. A submission may
  set a *lower* `max_steps`, its own `batch_size` and `eval_batch_size`.

## 4. Compliance — non-negotiable, a violation costs the whole competition

1. **Never read, print, sample, or statistically summarize anything under
   `data/generated/`.** It is write-only. Every design decision must be justifiable
   from the generator source and general reasoning. `benchmark/runner.py:75-90`
   installs an audit hook that blocks a submission from reopening dataset files —
   do not go looking for ways around it, and do not inspect the files yourself
   outside the runner either.
2. **No hard-coded algorithm in the forward pass.** No modular-exponentiation
   routine, no digit-wise multiplication written by you, no lookup table of answers,
   no `torch.load`. Outputs must be produced by learned parameters trained from
   random init in this run. Choosing an architecture whose *structure* suits modular
   arithmetic is allowed and encouraged; implementing the arithmetic is not.
3. **End-to-end differentiable.** An unbroken gradient path from the loss to the
   parameters responsible for the prediction; all input-dependent computation inside
   the autograd graph. Learned/soft halting is allowed (ACT/PonderNet are explicitly
   permitted). Control flow that decodes `T` out of `input_ids` to set a Python loop
   count is a **gray area** — it is data-dependent control flow rather than a solver,
   but it also reads the prompt format directly. If you test it, flag it clearly in
   your report as *diagnostic only, compliance-uncertain*, and always have a learned
   alternative.
4. **No custom training loop, no participant-controlled backward, no manifest
   override** in a submission. (Lab manifests for local screening are fine — they are
   not part of a submission.)
5. Everything stays on GPU; no CPU offload.
6. **Do not submit anything to the hosted service.** No `one-layer login`, `submit`,
   or any network call. Hard attempts are 1/day and belong to the user.
7. Keep negative results. Do not delete failed experiments or quietly drop a run that
   contradicts your hypothesis.

## 5. How to run an experiment

Environment: `/home/scratch.arohan_hw/git/one-layer-deeper/.venv` (Python 3.13.5,
torch 2.12.1+cu130). Use its absolute path; do not create your own venv.

```bash
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python

# build a screening manifest (writes into lab/manifests/)
$VENV lab/make_manifest.py --dataset e1 --mode fixed_step --max-steps 400 --seeds 74

# run a submission against it; appends one row to lab/archive.jsonl
$VENV lab/run_experiment.py \
  --submission path/to/submission.py \
  --manifest lab/manifests/lab_e1_fs400_s74.json \
  --tag <axis-label> --note "what is being tested"
```

`run_experiment.py` prints and archives `MAX_T`, `OOD_N_MAX_T`, the per-rung exact
accuracies, mean accuracy, step count and wall clock. **`MAX_T` is the score.**

### Use fixed-step manifests for every comparison

Several agents share **one GPU**. Wall-clock manifests (`--mode wallclock`) produce
step counts that depend on who else is running, which silently corrupts any
comparison. `--mode fixed_step` sets a 100,000s budget and a hard `max_steps`, so the
clock never binds and results are contention-immune and reproducible. Only use
`--mode wallclock` for a deliberate, final timing/feasibility check, and say so.

### Datasets available locally

Public: `e1`–`e5` (Easy, 60s), `m1`–`m5` (Medium, 600s). Hard (`h1`) is a private
hidden evaluator — unavailable locally by design.

Hard proxies generated by `lab/gen_hard_proxy.sh` (the real Hard is bigger than
Medium on the same two knobs; these extend past Medium so an approach can be stress
tested before a 1/day Hard attempt is spent):

| name  | modulus            | train T   | purpose |
|-------|--------------------|-----------|---------|
| `hp1` | 22-bit fixed N     | 4,8,16    | per-step arithmetic one step above M2 |
| `hp2` | sampled 30/32-bit  | 4,8,16    | arithmetic at scale |
| `hp3` | sampled 20/24/28-bit | 8       | OOD-N (tie-break) stress test |

All carry the full `T=1,2,4,8,16,32,64` ladder plus an OOD-N ladder. You may generate
further proxies with `data.squaring_mod` if your hypothesis needs a specific regime —
document the exact command in your report. Note the generator enumerates every unit
of a fixed modulus, so `--fixed_p/--fixed_q` is intractable past ~24 bits; use
`--modulus_bits` above that.

### Hardware

One GPU, ~280 GB, `sm_107` — **not** the H100 (`sm_90`) the competition scores on.
Kernels JIT from PTX on first use; `run_experiment.py` already points
`CUDA_CACHE_PATH` at scratch so the JIT cache persists (local workaround only).
**What transfers:** accuracy at fixed step count, certified T, loss design,
per-step optimizer efficiency, parameter allocation at equal steps. **What does not:**
absolute wall clock, steps-in-budget, compile payoff, overhead fractions.

## 6. What to produce

Work only on your own branch. Do not merge to `main`, do not touch another agent's
branch, do not submit anything hosted.

1. `lab/reports/<your-branch>.md` — the deliverable. Structure per `plan_old.md` §8.1,
   but keep it readable: hypothesis, what you ran (exact commands), what happened
   (numbers, including the per-rung table), what it means, what was falsified, what
   you would run next and why. Negative results stated as plainly as positive ones.
2. Every run archived in `lab/archive.jsonl` via `run_experiment.py` (it appends;
   never rewrite it).
3. Your best `submission.py` under `submissions/<your-branch>/`, self-contained. The
   evaluator lints it through `benchmark/validation.py:lint_submission_source` (which
   calls `submission_validation.validate_submission_source`) — the file must be named
   `submission.py`, be valid Python under 256 KiB, and import no `data`/`model`/`optim`
   module. Running it once through `lab/run_experiment.py` exercises that lint.
4. Commit to your branch as you go, with messages that say what was learned.

Report honestly. "The hypothesis is falsified, here is the evidence" is a
first-class result and more useful than a marginal win that will not replicate.
Do not claim a result you have not run. Do not report a number you did not measure.
