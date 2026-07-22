# One Layer Deeper — Findings (B300 exploration session)

_Working log. Negative results kept with equal weight (per PLAN.md §7)._
_Hardware: NVIDIA B300 SXM6 (sm_103), 275 GB. Competition scores on H100 (sm_90)._

## The task (from the public generator spec — compliant to read; data itself never inspected)

`data/squaring_mod.py` generates **repeated modular squaring**: given a prompt encoding
`(N, x, T)`, the target is `x^(2^T) mod N` — i.e. apply the squaring map `s(y)=y² mod N`
**T times**. So **composition depth = T**, and the competition name is literal: "one layer
deeper" = generalizing to more compositions.

- Separate prompt/output tensors → **bidirectional** attention over the prompt (padding mask,
  not causal) on Easy/Medium. Hard is a hidden evaluator.
- Tiers scale two knobs: **modulus size** (bits) and **composition depth T**.
- **Scored splits include OOD at deeper T than trained** (E1 trains T∈{1,2,3}, scores OOD T=6).
  The final metric is an **unweighted mean of exact-match accuracy across all splits incl. OOD**
  → OOD-depth generalization is a large fraction of the score. A model that can iterate to
  arbitrary depth (weight-tied recurrence) is the natural fit.

## Evaluator contract (the parts that drive design) — source-verified

- **Clock starts at import**, backdated before `build_model`; construction + `.to(device)` +
  `build_optimizer` + any `torch.compile` are all charged (`runner.py:444-448`). On Easy (60s)
  a compile can eat the whole budget → 0 steps; on Hard (3600s) it amortizes. **Tier-dependent.**
- **Exactly one `optimizer.step()` per batch**, evaluator-owned (`runner.py:297-319`). BUT
  arbitrary **recurrent compute inside a single `forward` is allowed**. → Depth lives in forward:
  more compute/update ⇄ fewer updates. **This is the whole game (Axis A).**
- `forward(input_ids:int64[B,L], attention_mask:bool[B,L])` → returns `(logits[B,L,17], aux)`.
  Return `aux=None` when unused. Custom loss gets `(logits[valid], labels[valid], aux)` → must
  return one scalar differentiable finite tensor; evaluator runs `.backward()`.
- **vocab_size = 17** (10 digits + 7 special) for every tier. Embeddings are negligible; the
  **500M state ceiling is enormous** for this task — spend params on composition, not surface.
- Metric = `mean_exact_accuracy` = unweighted `fmean` over every (seed × split). **Exact match
  per row** (all target positions correct). Scored splits: `test`, `ood`, `ood_t`, `ood_n_t`.
- Eval budget = **half** the training budget, aggregate across splits, per-batch deadline. Deep
  models must also *evaluate* deep within half-budget (Axis B hazard).
- Submission may import only **torch==2.12.1, numpy==2.5.0, benchmark**. No einops/TE.
- State ceiling counts params + persistent buffers; tied tensors once; frozen counts;
  **non-persistent buffers are EXCLUDED** → a depth-curriculum step counter should be a
  non-persistent buffer (free), better than PLAN's "keep it scalar".
- Manifests: Easy 60s / Medium 600s / Hard 3600s; all bf16+amp, compile=false, grad_clip=1,
  batch 512, one seed [74], max_steps ceiling 1e6 (wall-clock always binds). Participant may set
  a LOWER max_steps to stop early — useful for fixed-step local experiments.
- Hosted submit = `one-layer login` (GitHub OAuth) + network; rate-limited (Easy 60/day,
  Medium 6/day, Hard 1/day). **Not touched this session** — needs the user's identity and burns
  attempts. Deferred pending explicit OK.

## Hardware reality: B300 ≠ H100 (critical for what transfers)

- torch 2.12.1+cu130 compiled arch list = `[75,80,86,90,100,120]` — **no sm_103**. B300 kernels
  **JIT from sm_100 PTX on first use**. Cold first GPU run paid **~88s** of JIT (baseline did
  **1 step in 88s**); after warming the JIT cache the same run did **365 steps in 60s**.
  - Fix (local-only): `CUDA_CACHE_PATH=/home/scratch.arohan_hw/.nv_cache` (home is a 5 GB quota;
    the JIT cache is ~1.1 GB). Baked into `lab/run_experiment.py`. **H100 never JITs — this is a
    pure B300 workaround and does not affect real submissions.**
- Warmed B300 matmul: **bf16 ~1747 TFLOP/s**, fp32/TF32 ~823. Roughly **2–2.5× an H100's
  achieved throughput**. So a 60s wall-clock run here completes ~2× the steps H100 would.

**What transfers to H100:** accuracy-vs-depth at fixed step count; loss design; per-step
optimizer efficiency; parameter allocation at equal steps. **What does NOT:** absolute wall-clock,
steps-in-budget, compile-payoff, overhead fractions. Strategy: spend B300 time on the
hardware-independent axes; treat every timing number as B300-local and H100-pending.

## Phase 0 — DONE
- Env on scratch (uv 0.11.31, Python 3.13.5, torch 2.12.1+cu130). 105/105 unit tests pass.
- CPU smoke: harness failure-path validated (smoke's 0.05s eval budget is too tight on this CPU;
  structurally fine — training ran, eval started).
- **Baseline (AdamW, D=128, 1 block) on e1 (B300, 60s): 365 steps, mean_acc=0.0067
  (test 0.013, OOD 0.000).** One layer ≈ cannot compose. This is the gap to beat.

## Phase 1 — instrumentation
- `lab/run_experiment.py` — runs a submission×manifest as a subprocess, parses `RESULT_JSON`
  (+ a downsampled train curve), appends to `lab/archive.jsonl` (successes AND failures). Never
  reads generated data. `lab/summarize.py` tabulates. `lab/make_submission.py` generates
  standalone recurrent submissions (width/opt=adamw|muon/loss=ce|focal/cosine-sched/steps).
- **Variance floor (L8 d128 AdamW):**
  - e1 ×3 repeats (same bytes): mean {0.005,0.027,0.017}, OOD {0.01,0.04,0.02} → **OOD std ≈0.015**.
  - Cross-dataset e1–e5: mean {e1~0.016, e2 0.008, e3 0.005, e4 0.001, e5 0.006}. All near floor.
  - **Fixed N (e1=323) is the only mildly-learnable one; sampled-N (e3–e5) collapses to ~0** — the
    model can memorize one modulus a little but cannot learn general x²mod N. e4 got 0.0014 at
    3249 steps (more steps, less accuracy → confirms it's difficulty, not undertraining).
- Step-time: regime is overhead/dataloader-bound (GPU ~4%); step count varies with dataset
  (e1 ~460 vs e4 ~3250 in 60s, from seq-len/throughput differences), which is itself a noise source.

## ★ AXIS A — the rigorous verdict (this supersedes the single-run sweep below)
Two-step story. (1) The single-run 60s depth sweep (table below) showed OOD 0→0.06 with L, BUT
the variance floor exposed that as mostly noise: the SAME submission across 4 runs gave OOD
{0.06,0.01,0.04,0.02} (std ≈0.02) — winner's-curse from tiny OOD eval (~100 ex) + wall-clock
step-count jitter (453–496 → different final models). (2) A CONTROLLED grid — **fixed 400 steps
(kills jitter) × 3 seeds** — recovers a CLEAN effect:

| L | OOD mean±std | raw OOD (3 seeds) | test | 
|---|--------------|-------------------|------|
| 2 | 0.003±0.005 | 0.0, 0.01, 0.0 | 0.020 |
| 8 | **0.023±0.005** | 0.02, 0.02, 0.03 | 0.020 |
| 32| **0.027±0.009** | 0.02, 0.02, 0.04 | 0.013 |

**Axis A CONFIRMED (with error bars): depth improves OOD, gap L2→L8 = 0.020 ≈ 4× pooled std.**
Mechanism is coherent: ID `test` needs ≤3 iterations so even L=2 suffices → test flat across
depth; OOD T=6 needs ≥6 iterations so L=2 fails (≈0) and L≥8 succeeds (~0.025). Knee ≤8
(L8≈L32). **Methodology lesson: single 60s runs are too noisy here (step-jitter dominates);
fixed-step + multi-seed is required to see real effects.** Absolute OOD is still only ~2.5% —
depth fixes the *iteration* bottleneck, NOT the *arithmetic* one (test stuck ~2%).
Next: confirm on Medium (OOD T=32, 3000-example eval → far better SNR).

## Phase 2A — minimum viable depth: [SINGLE-RUN, see correction above]
Weight-tied recurrence (one shared Block, identical params), e1 @60s on B300:

| L | steps/60s | test(ID) | OOD(T=6) | mean_acc |
|---|-----------|----------|----------|----------|
| 1 | 429 | 0.027 | 0.00 | 0.013 |
| 2 | 524 | 0.013 | 0.02 | 0.017 |
| 4 | 486 | 0.013 | 0.03 | 0.022 |
| 8 | 453 | 0.013 | **0.06** | 0.037 |
| 16| 472 | 0.013 | 0.06 | 0.037 |

**What the single run APPEARED to show (SUPERSEDED by the variance floor — see correction above):**
1. Depth *appeared* to improve OOD monotonically to a knee at L≈8 (OOD 0→0.06), plateauing at
   L=16. The mechanism is plausible (e1 OOD T=6 needs ≥6 iterations) and worth re-testing where
   the signal is bigger — but the OOD std (~0.02) equals this whole trend, so on e1 it is **NOT
   established**. Kept here as the hypothesis to test properly on Medium (bigger eval, OOD T=32).
2. **Depth is nearly free here** — step count stays ~450–520 across L=1..16. At D=128 the model
   is overhead/dataloader-bound, not compute-bound (Phase-1 step-time insight). So "go deep"
   costs almost nothing until the block compute exceeds the ~130ms/step overhead floor.
3. **Absolute accuracy is low (~4% mean)** — ~450 steps in 60s is little training. The knee says
   depth beyond ~8 is wasted; remaining gains must come from **more effective steps** (train
   longer / faster convergence) and **better optimization (Muon)** at depth ≈8, NOT more depth.

Caveat: B300 does ~2× H100 steps in 60s, so hosted would see ~230 steps, even lower accuracy —
but the *knee location* (a task property) transfers; the absolute numbers do not.

Watch: residual stream grows across un-normalized loops (init loss ~48 on random input at L=8)
— a stability risk at high L; may need inter-loop scaling/normalization.

## Phase 2B — learnability: PLATEAU (more steps don't help)
- L=8 AdamW: 450 steps(60s) mean 0.037 vs 2000 steps(243s) mean 0.032 — **flat**. The model
  converges to a low plateau (~1.3% ID test, ~5% OOD); the bottleneck is **capacity/optimization/
  architecture, NOT step count**. Modular squaring x^(2^T) mod N is an algorithmic-generalization
  task (test prompts have unseen x) — small models struggle to learn the arithmetic itself.
- ID `test` stuck ~1.3% across all depths/steps; only small-count OOD moved. **Variance floor
  needed** before trusting the depth->OOD trend (may be partly noise).

## Phase-1 insight — the regime is OVERHEAD-BOUND, not GPU-bound
- During training, **GPU util ~4%**, 1.2 GB used. Step time (~130ms) is dataloader + Python +
  hand-rolled-optimizer overhead, not block compute — which is why L=1..32 all did ~460 steps/60s.
- Consequences: (a) **B300≈H100 wall-clock for small models** (2x only applies GPU-bound) — better
  transfer than feared; (b) **massive free headroom**: with GPU 96% idle and a 500M ceiling, going
  wider / bigger-batch is ~free on wall-clock until compute-bound. Capability push is cheap.

## KEY REFRAME — two independent bottlenecks (the map that matters)
The depth sweep + plateau together reveal WHY it's stuck:
1. **Iteration count.** OOD T=6 needs ≥6 serial squarings, so needs L≥6 — this is exactly why
   OOD accuracy rose with depth to L≈8 (Axis A is real and about *having enough iterations*).
2. **Per-step arithmetic accuracy.** ID `test` (T∈{1,2,3}) is stuck ~1.3% at every depth/step/
   optimizer — the model cannot accurately compute even ONE squaring step x²mod N and generalize
   to unseen x from ~250 examples. This is a capacity/architecture wall; depth & steps don't touch it.

**Structural mismatch:** T is in the input and varies per example, but the model loops a FIXED L.
So on T=1 it over-iterates and must learn L−1 no-op steps; on OOD T it must have L≥T. A fixed-L
model can't natively match a variable-T task → this is a strong argument for **adaptive computation
/ halting (Axis F)** or explicit T-conditioning (loop count read from the prompt), not just brute
depth. This is the highest-value idea to try next session.

## Capability push (bottleneck #2) — width does NOT help; plateau is robust
L=8 AdamW, e1@60s:

| D | params | steps | test | OOD | mean |
|---|--------|-------|------|-----|------|
| 128 | 0.25M | 453 | 0.013 | 0.06 | 0.037 |
| 256 | 0.80M | 467 | 0.013 | 0.01 | 0.012 |
| 512 | 3.0M  | 512 | 0.013 | 0.00 | 0.007 |
| 768 | 6.6M  | 282 | 0.000 | 0.00 | 0.000 |

Wider is strictly WORSE at this budget — more params, same ~few-hundred (undertrained) steps.
**D=128 is the sweet spot** for the overhead-bound / few-step regime. `test` pinned at exactly
0.013 (128–512) = trivial-predictor floor.

## Medium (m1, N=10403, OOD T=32) — arithmetic wall dominates
Fixed-800-step × 2-seed grid on m1: **~0 everywhere** (L=4: mean 0.0010±0.0002; L=16: 0.0006;
test ≈0.001). With a bigger modulus the model can't do even ONE squaring, so there's no signal
for depth to act on — Axis A is untestable here because bottleneck #2 (arithmetic) swamps it.
Key strategic takeaway: **the arithmetic-generalization wall gets worse with modulus size and
is THE blocker for larger tiers (incl. Hard).** Depth/adaptive-computation only pays once per-step
arithmetic works (true only on e1's tiny N=323). (Aside: m1 runs fast — 800 steps in ~16s.)

## ROBUST PLATEAU — the headline negative result
The ~1–5% exact-accuracy plateau on e1 is **unbroken by every brute-force lever tried**:
more steps (450→2000: flat), Muon vs AdamW (both ~plateau), width (128→768: worse). The provided
competition baseline itself scores ~0.007 — near-zero is the *designed* baseline; recurrence L=8
(0.037) is ~5× it but still low. **The wall is arithmetic generalization** (learn x²mod N and
generalize to unseen x from ~250 examples) — an architecture problem, not scale. This is the
main asset of the session: the failure surface is mapped.

## Axis F — adaptive computation (PonderNet-lite) prototype: BUILT + tested
`lab/make_adaptive.py`: weight-tied recurrence (up to MAX_LOOPS, input-injected) + per-example
halt head → halting distribution p_t; readout from the halt-WEIGHTED hidden state; E[steps]
exposed via `aux`, penalized by a custom loss (ponder cost β). Validated on CPU (halting learns).
e1, fixed 400 steps × 3 seeds:

| config | OOD mean±std | test | note |
|--------|-------------|------|------|
| fixed L=8 | 0.023±0.005 | 0.020 | reference |
| fixed L=32 | 0.027±0.009 | 0.013 | reference |
| ACT β=0.01 | **0.000±0.000** | 0.027 | **halting COLLAPSED to min depth** |
| ACT β=0 | 0.023±0.026 | 0.029 | ≈ fixed deep-L, higher variance, no win |

Findings: (1) with a ponder penalty the halt head **collapses to always-halt-early** (the exact
Axis F failure PLAN.md flagged) — because the accuracy signal (~2%) is too weak to outweigh even
β=0.01 ponder cost, so the model minimizes steps. (2) With β=0 it recovers fixed-deep-L OOD but
with no gain and MORE variance. Expected: **e1's OOD is a single T=6, so per-example adaptivity
has nothing to adapt to.** Adaptive computation's payoff is gated behind (a) mixed-T datasets AND
(b) working per-step arithmetic — and the arithmetic wall denies (b) everywhere but tiny-N e1.
The prototype is correct and ready; it just can't win until the arithmetic bottleneck is broken.
Next: needs ponder-schedule/warmup (delay β until accuracy exists) or KL-to-prior to avoid collapse.

## Highest-value next-session directions (see queue.md)
1. **Adaptive / T-conditioned iteration (Axis F).** The task needs data-dependent depth (T in
   input); fixed-L is a structural mismatch. Learned halting (PonderNet/ACT) or loop count
   conditioned on the prompt's T. Top idea.
2. **Arithmetic-friendly architecture** — digit-wise processing, or exploit that N is fixed per
   Easy dataset (learn the squaring permutation as a table + compose).
3. **Variance floor** (running) to certify which small effects (depth→OOD) are real vs noise.
