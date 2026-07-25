# group-rotation — make squaring a rotation

**Branch:** `explore/group-rotation` · **Verdict: falsified as a learning route, with two
quantitative bounds that outlive the hypothesis.** The group/Fourier representation is *exactly right* and *fully
identifiable from e1-sized data* — but gradient descent cannot reach it or stay in it,
for a reason that is measurable and that does not exist in classic modular-arithmetic
grokking. Every architecture in this family scores `MAX_T = 0`, identical to the baseline.

---

## 1. Hypothesis

`Z*_N` is abelian and squaring is doubling in the exponent, so in a representation where
the input is carried by phase-like features, `x -> x^(2^T)` should be a rotation and every
`T` should be reachable at constant depth. The plan was: multiplicative/bilinear
interaction blocks, complex (cos, sin) latents, a learned `T`-conditioned angle multiplier,
a phase-sensitive readout — then check whether the group solution is what actually gets
learned.

## 2. The structural analysis that drove every design

Two facts, both derived from the public generator source and elementary number theory
(never from data), turn out to determine the whole result:

1. **A place-valued decimal string is a sum; squaring is a product.**
   `x = sum_i 10^i d_i`, so `x^2 = sum_{i,j} 10^(i+j) d_i d_j`. If a value `v` is carried by
   *additive characters* `cos(w v), sin(w v)`, then reduction mod `N` is **free** (a phase
   is periodic) and the phase of `x^2` is a **sum over ordered pairs of digit slots** of a
   term depending only on `(slot_i, slot_j, d_i, d_j)`. That is an outer-product /
   pairwise feature map — compositional in the digits, so it generalises to unseen `x`
   *by construction*.

2. **The rotation picture needs the *other* character basis, and that one is not
   compositional.** Squaring is a rotation in *multiplicative* characters
   `chi_j(x) = e^{2 pi i j dlog(x)/ord}`, because `chi(x^e) = chi(x)^e`. But `dlog` of a
   decimal string has no decomposition over digits — the only way to compute it from
   digits is a degree-`(#digits)` tensor, i.e. a lookup on the *value*, which is exactly
   what cannot generalise to a held-out value.

**These two requirements are in direct conflict, and that conflict is the headline
structural result of this branch:**

| | encoder compositional in digits? | squaring is...? | any `T` at constant depth? |
|---|---|---|---|
| additive characters `e^{2 pi i k v/N}` | **yes** | a degree-2 map | no — needs degree `2^T` |
| multiplicative characters `e^{2 pi i j dlog v}` | **no** (needs a value lookup) | **a rotation** | yes |

So "any `T` for free by rotating a phase" is unreachable *by construction* on this task's
input format. The only survivor inside the family is: additive characters + **iterate** a
shared degree-2 block, decoding and re-encoding digits between steps. That is literally
"one layer deeper", and it is the one thing in this report that measurably helped.

## 3. What was built

Three generators, all emitting self-contained submissions, all parameters learned from
random init, no modulus / `phi` / discrete log / modular exponentiation anywhere in a
forward pass, and no Python control flow that reads `input_ids`:

| file | architecture | knobs |
|---|---|---|
| `lab/make_gr.py` | transformer (L8, d128) | `--mixer std/geglu/bilin`, `--rot` complex-rotation sublayer, `--gbp` outer-product pooling, `--fhead` Fourier readout, `--tcond` learned T-conditioned angle multiplier, `--tie` |
| `lab/make_grnet.py` | minimal exactly-expressive model: pooled per-(position,token) value, explicit product, `cos/sin`, tail-aligned readout | `--width`, `--freqs`, `--harm`, `--quad`, `--posmode abs/rev/both`, `--tcond` |
| `lab/make_grpair.py` | phase table over (position-pair, token-pair) — represents the squaring map with **no continuous frequency search** | `--freqs`, `--harm`, `--posmode`, `--order 1/2`, `--tcond` |

Diagnostics (never open `data/generated/`):
`lab/mkdiag.py` (lab-only weight-dumping copy of a submission) + `lab/diagnose_gr.py`
(probes a trained model on synthetic prompts built from the public tokenizer spec),
`lab/probe_step.py`, `lab/probe_iter.py`, `lab/probe_learnability.py` (offline studies on
self-generated data).

## 4. Evaluator results — every run `MAX_T = 0`

All comparisons `--mode fixed_step` (contention-immune). e1 rung = 38 examples, so
`0.026 = 1/38` is the one-example floor.

### 4.1 Transformer ablation, one variable at a time (e1, 2000 fixed steps, seeds 74/75/76)

```
$VENV lab/make_manifest.py --dataset e1 --mode fixed_step --max-steps 2000 --seeds 74 75 76
bash lab/sweep_gr_ablate.sh
```

Matched control = the same generator with every toggle off.  3 seeds, so the
single-example floor is 1/114 = 0.009.

| variant | MAX_T | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | T=64 | mean acc |
|---|---|---|---|---|---|---|---|---|---|
| GR control: std MLP, tied head, L8 d128 | **0** | 0.000 | 0.000 | 0.000 | 0.035 | 0.000 | 0.018 | 0.026 | 0.0278 |
| H1 bilinear mixer (pure product of 2 projections) | **0** | 0.000 | 0.009 | 0.000 | 0.000 | 0.018 | 0.018 | 0.026 | 0.0222 |
| H1 control: GEGLU (gated, not pure bilinear) | **0** | 0.009 | 0.000 | 0.000 | 0.009 | 0.009 | 0.026 | 0.018 | 0.0389 |
| H2 complex-rotation sublayer (unit-modulus) | **0** | 0.018 | 0.035 | 0.000 | 0.018 | 0.035 | 0.044 | 0.009 | 0.0306 |
| outer-product global bilinear pooling | **0** | 0.009 | 0.009 | 0.009 | 0.026 | 0.009 | 0.026 | 0.009 | 0.0261 |
| H4 Fourier (phase) readout + linear head | **0** | 0.009 | 0.018 | 0.009 | 0.035 | 0.009 | 0.035 | 0.026 | 0.0361 |
| H3 T-conditioned per-frequency angle multiplier | **0** | 0.009 | 0.000 | 0.000 | 0.000 | 0.009 | 0.026 | 0.009 | 0.0350 |
| H4b untied input embedding / output head | **0** | 0.009 | 0.009 | 0.044 | 0.018 | 0.035 | 0.018 | 0.026 | 0.0378 |

Matched recurrent baseline (`submissions/exp_recur/L8_d128`, 2000 steps, seed 74):

| variant | MAX_T | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | T=64 | mean acc |
|---|---|---|---|---|---|---|---|---|---|
| matched baseline: recurrent L8 d128, e1 fs2000 s74 | **0** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.026 | 0.0150 |

**Every toggle is null.** No variant certifies any rung; rung-1 is within one
example of the control everywhere.  The T-conditioned angle multiplier (H3), the
headline idea of this branch, is indistinguishable from the control.

### 4.2 GRNet, long runs (e1, 20 000 fixed steps, seed 74)

```
bash lab/sweep_gr_long.sh
```

| variant | MAX_T | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | T=64 | mean acc |
|---|---|---|---|---|---|---|---|---|---|
| GRNet w2 K256 wd0.1 lr0.01 20k | **0** | 0.026 | 0.000 | 0.053 | 0.079 | 0.026 | 0.000 | 0.079 | 0.0300 |
| GRNet w2 K256 wd1.0 lr0.01 20k | **0** | 0.026 | 0.000 | 0.053 | 0.105 | 0.053 | 0.026 | 0.079 | 0.0383 |
| GRNet w2 K256 wd3.0 lr0.01 20k | **0** | 0.026 | 0.026 | 0.026 | 0.026 | 0.026 | 0.000 | 0.000 | 0.0333 |
| GRNet w2 K256 wd1.0 QUAD=0 (control) | **0** | 0.000 | 0.000 | 0.053 | 0.026 | 0.079 | 0.079 | 0.079 | 0.0633 |
| GRNet w2 K256 wd1.0 posmode=rev | **0** | 0.026 | 0.026 | 0.000 | 0.000 | 0.000 | 0.053 | 0.000 | 0.0383 |
| GRNet w2 K256 wd1.0 posmode=abs | **0** | 0.000 | 0.026 | 0.053 | 0.026 | 0.000 | 0.000 | 0.053 | 0.0200 |

`quad=0` is the control that removes the multiplicative interaction; it is not worse.
`posmode=rev` indexes slots by distance from the end of the prompt, which is the only
indexing under which a decimal place value is well defined when x has variable length --
also null.  Train accuracy reaches 1.000 within ~100-1000 steps in every one of these
runs (see `train_curve` in `lab/archive.jsonl`), so all 20 000 steps are post-memorisation.

### 4.3 GRIter — iterating one shared phase block (e1, 20 000 fixed steps, seed 74)

The offline result of §5.6 ported to the real prompt: learned slot queries read the
prompt into S soft digit slots, ONE shared pairwise-phase block is applied `LOOPS` times
with a soft-digit round trip between steps, and a learned selector pooled from the prompt
(which contains the `T` field) mixes the step outputs — so composition depth is *learned
from the prompt*, not set by Python control flow over `input_ids`. `LOOPS=1` is the
control that removes the round-trip constraint.

| variant | MAX_T | T=1 | T=2 | T=4 | T=8 | T=16 | T=32 | T=64 | mean acc |
|---|---|---|---|---|---|---|---|---|---|
| GRIter LOOPS=1 (control: no round-trip constraint) | **0** | 0.000 | 0.079 | 0.053 | 0.053 | 0.026 | 0.053 | 0.026 | 0.0533 |
| GRIter LOOPS=2 | **0** | 0.026 | 0.026 | 0.026 | 0.132 | 0.000 | 0.026 | 0.079 | 0.0433 |
| GRIter LOOPS=2 slots=4 e1 20k | **0** | 0.026 | 0.026 | 0.026 | 0.132 | 0.000 | 0.026 | 0.079 | 0.0433 |
| GRIter LOOPS=4 | **0** | 0.026 | 0.000 | 0.026 | 0.079 | 0.026 | 0.053 | 0.000 | 0.0333 |
| GRIter LOOPS=4 slots=4 e1 20k | **0** | 0.026 | 0.000 | 0.026 | 0.079 | 0.026 | 0.053 | 0.000 | 0.0333 |
| GRIter LOOPS=8 | **0** | 0.000 | 0.026 | 0.053 | 0.053 | 0.053 | 0.053 | 0.026 | 0.0400 |

(`LOOPS=2` and `LOOPS=4` each appear twice: the first sweep was interrupted and re-run.
The duplicate rows are *identical to three decimals* — a free reproducibility check on the
fixed-step protocol.)

**It does not reproduce the offline gain, and the selector diagnostic says why.** Dumping
the weights of a GRIter LOOPS=4 run and reading the selector's output on synthetic prompts
(`lab/diagnose_gr.py` probe [6]):

```
T= 1  weights=[0.357, 0.497, 0.074, 0.072]  entropy=1.097 (max 1.386)
T= 2  weights=[0.409, 0.022, 0.422, 0.147]  entropy=1.097
T= 4  weights=[0.850, 0.150, 0.000, 0.000]  entropy=0.424
T=64  weights=[0.039, 0.878, 0.074, 0.009]  entropy=0.475
```

The selector never learns `T -> step count`: it is diffuse, it puts almost all mass on
steps 1–2 regardless of `T`, and steps 3–4 are effectively dead. A *soft blend* of
composition depths lets the model satisfy every training row without ever performing a
clean round trip — so the constraint that produced the offline gain (§5.6, where each
depth is supervised separately) is dissolved by the very mechanism that was supposed to
make `T` learnable. **Concrete fix for whoever picks this up: anneal the selector towards
one-hot (or add an entropy penalty through `training_loss`/`aux`, which the evaluator
allows), so that a row with time-step `T` is actually forced through `T` applications of
the block.**

### 4.4 Screens (archived, tags `smoke` / `lr-screen` / `wd-screen`)

GRPair (the pair-phase table) memorises e1's training set to `loss=0.001, acc=1.000` in
**under 100 steps** (tag `smoke`); GRNet's quadratic variant does it by ~500 steps while
its `quad=0` control only reaches 0.45. On GRNet an lr screen (1e-3 / 1e-2 / 3e-2, tag
`lr-screen`) and a weight-decay screen (0.3 / 1 / 3, tag `wd-screen`) were run to find a
setting that does not instantly memorise: wd=1 holds train accuracy at 0.87 and wd=3 at
0.69 after 2000 steps, and neither moves rung-1 off the floor.

At that point the evaluator stopped being the efficient instrument. A 20 000-step e1 run
costs 5–14 minutes and returns a single number that is 0; the offline crux in §5 asks a
sharper question in 90 seconds. The written-but-unrun `lab/sweep_grpair_long.sh` (an 8-cell
`(K, H)` grid) was superseded by §5.1, which sweeps the same axes offline and shows they
are all at the memorisation floor — running it would have spent an hour to re-derive that.

## 5. Why — the offline probes (this is the real content)

The evaluator runs only say "0". To find out *why*, the task was stripped to its crux on
**self-generated** data (`lab/probe_step.py`; no dataset access):

> learn `digits(v) -> digits(v^2 mod 323)` from 250 of the 288 units of `Z*_323` and get
> the other 38 right — exactly e1's rung-1 problem with the prompt removed.

### 5.1 Learned end-to-end: memorisation, always

Every setting tried (`K` in {1,2,4,8,32,161}, `H` in {1,4,8,16,32,80,161}, weight decay in
{0.1,1,3,10}, lr in {3e-4 … 3e-2}, up to 20 000 steps):

| | train exact | held-out exact |
|---|---|---|
| pair-phase table, learned | **1.000** (by step ~2000) | **0.000 – 0.053** |

Train loss reaches `0.00000` and stays there. This is the pre-grok state, and it never
groks.

### 5.2 Oracle: the representation is right *and* identifiable

Freeze the pair table at the exact additive-character phases of `v^2`
(`theta_k = 2 pi k v^2 / N`) and train **only** the readout on the same 250 examples:

```
$VENV lab/probe_step.py --oracle --freqs 161 --harm 1 --steps 1500 --wd 0.01 --lr 0.01
  step=500  loss=0.00000  train_exact=1.000  held_exact=1.000
```

**100% on the 38 held-out values within 500 steps.** So the hypothesis is *correct as a
representation claim*: the group/Fourier code solves e1 rung-1 exactly, and 250 examples
are enough to pin the readout. (It works because squaring is 4-to-1 on `Z*_N`, so the ~72
distinct squares are all covered by the training `x`.)

### 5.3 The basin around the group solution is ~1e-5 wide, and training loss is flat in it

Perturb the oracle frequencies by a relative `eps` and retrain the readout:

| relative frequency error `eps` | train exact | **held-out exact** |
|---|---|---|
| 0 | 1.000 | **1.000** |
| 1e-6 | 1.000 | **1.000** |
| 1e-5 | 1.000 | **0.868** |
| 1e-4 | 0.996 | **0.053** |
| 1e-3 | 0.988 | **0.000** |
| 1e-2 | 0.992 | **0.000** |

**Train accuracy is ~1.00 at every `eps`.** The training objective cannot distinguish the
group solution from a solution that is wrong by one part in 10^3. There is no gradient
pointing at it.

### 5.4 And an optimizer step of ~1e-5 relative destroys it

Take the model *initialised at* the group solution (a scale-free construction, §5.5) and
vary the learning rate on the phase-code parameters, readout lr fixed at 1e-2:

| phase-code lr | train exact | held-out exact |
|---|---|---|
| 0 (frozen) | 1.000 | **1.000** |
| 1e-7 | 1.000 | **1.000** |
| 1e-5 | 0.736 (falling) | 0.526 (falling) |
| 3e-3 (uniform) | 0.02 | 0.000 |

So the group solution is not just unreachable — it is *not even stable* under a normal
learning rate. AdamW's per-parameter step is ~`lr`, and the phases need relative precision
~1e-5 on parameters of size ~10, i.e. absolute precision ~1e-4.

**This is the mechanistic difference from classic grokking.** In modular-arithmetic
grokking the input is a one-hot over `Z_p`, so the "frequency" is a *discrete* choice
inside an embedding table — every entry is free and periodicity is automatic. Here the
input is a decimal string, so the frequency is a *continuous scale* that must be
commensurate with `N` to ~1e-5. That is a measure-zero target in a flat loss landscape.

### 5.5 Two attempted fixes, both falsified

**(a) Remove the degeneracy by construction.** A learned linear digit code gives
`code(v) = s v` for an unknown scale `s`; reading the modulus out of the prompt with the
*same* code makes the scale cancel exactly:

```
code(x)^2 / (code(N) * code(1)) = (s x)^2 / ((s N)(s)) = x^2 / N
```

Wrapping that in `cos/sin` of *integer* multiples of `2 pi` makes mod-`N` periodicity
**structural** — there is no frequency left to search for, and it should transfer across
moduli. Verified correct: with the code frozen at a value-like init it gives held-out
**1.000** (§5.4, row 1). But the code cannot be *learned*: at any usable lr the code
parameters move ~1e-3 relative per step and the representation is gone by step 10
(train 0.02). Making it work requires initialising the digit ramp at `d` and the place
code at `10^i` and then effectively freezing them — which is hand-writing the decimal
value decoder, not learning it. Flagged as **compliance-uncertain and not pursued**.

**(b) Force the rank-1 `place x digit x frequency` factorisation** (which *cannot*
memorise — 13 numbers in the whole phase pathway). Result: it cannot fit the training set
either (train exact 0.13 – 0.58). The hypothesis class that excludes memorisation also
excludes anything the optimizer can descend into.

### 5.6 The one thing that worked: **iterate a shared block**

`lab/probe_iter.py` — one shared "soft digits -> pairwise phase -> soft digits" block,
applied `T` times, trained on `T in {1,2,3}` exactly as e1 supplies:

| training `T` | seen-x T=1 | **held-out T=1** | seen-x T=4 (never trained) | held-out T=4 |
|---|---|---|---|---|
| `{1}` | 1.000 | **0.000** | 0.872 | 0.000 |
| `{1,2}` | 1.000 | **0.237** | 1.000 | 0.263 |
| `{1,2,3}` | 1.000 | **0.237 – 0.289** | 1.000 | 0.263 |
| `{1,2,3}`, K=64 | 1.000 | **0.237 – 0.342** | 1.000 | 0.342 |

**5 seeds**, held-out T=1 at 4000 steps:

| | s0 | s1 | s2 | s3 | s4 | mean ± sd |
|---|---|---|---|---|---|---|
| single step (`T={1}`) | 0.000 | 0.000 | 0.000 | — | — | **0.000 ± 0.000** |
| iterated (`T={1,2,3}`) | 0.237 | 0.263 | 0.263 | 0.289 | 0.316 | **0.274 ± 0.028** |

and the label-free mod-`N` symmetry check moves with it (0.750 / 0.750 / 0.764 for the
single step — exactly the memorisation baseline — vs 0.826 / 0.806 / 0.812 iterated). The
effect is far outside seed noise, which is 0 for the control.

Two things happen the moment there is more than one composed step:

* **Held-out accuracy goes from 0.000 to ~0.24–0.34** — an 8-10x jump over the best
  single-step model. Composing the block forces the decode to round-trip
  (`decode(phase(v^2))` must re-encode consistently), and a round-trip is only consistent
  when the phase is genuinely periodic mod `N`. **The T=1,2,3 rows of e1 are not extra
  data; they are the constraint that partially pins the frequency.**
* **`T` extrapolation on seen `x` is perfect** — trained on `T<=3`, the model is at 1.000
  on `T=4`. Composition depth transfers exactly. This is the competition's premise working.

It plateaus at ~0.24–0.32 though, invariant to `K` (32/64), `H` (8/16), softmax temperature
(1.0/0.25), weight decay (0.1/1.0) and step count (4k/20k) — the round trip constrains the
frequency but does not pin it to the 1e-5 that certification needs. At 20 000 steps with
`K=64 H=16`:

```
step=20000 loss=0.00000  T1:seen=1.000,held=0.237  T2:seen=1.000,held=0.237
                         T4:seen=1.000,held=0.237  T8:seen=1.000,held=0.316
```

**Read the last column carefully: trained only on `T<=3`, the model is at 1.000 on `T=8`
for values it can handle.** Composition depth extrapolates *perfectly*; value
generalisation does not move at all. That is the two-bottleneck picture of `findings.md`
reproduced with a mechanism — the iteration bottleneck is solved outright by weight tying,
and the per-step arithmetic bottleneck is the entire remaining problem.

**Deeper `T` training makes it worse, not better.** Training on `T in {4,8,16}` (m1's
ladder, i.e. up to 16 unrolled soft steps) **diverges** from random init — loss climbs from
2.6 to 8.9 and every rung sits at 0.000. So the extra round-trip constraint that deeper `T`
would supply is not collectible without a curriculum, and the evaluator forbids a custom
training loop. This independently reproduces `findings.md`'s "m1 is ~0 everywhere".

## 6. Is the learned solution the group one? (weight/activation diagnostics)

Two independent diagnostics, both on the model's **own weights and activations**
(synthetic prompts built from the public tokenizer spec; no dataset file is opened).

### 6.1 On a real trained submission (GRPair K=8 H=16 wd=1.0, e1, 20 000 steps)

```
$VENV lab/mkdiag.py --submission submissions/group-rotation/gp_k8h16/submission.py
GR_DIAG_SAVE=/tmp/gp_k8h16.pt $VENV lab/run_experiment.py \
    --submission lab/diag/gp_k8h16/submission.py \
    --manifest lab/manifests/lab_e1_nw0_fs20000_s74.json --tag diag
$VENV lab/diagnose_gr.py --submission submissions/group-rotation/gp_k8h16/submission.py \
    --checkpoint /tmp/gp_k8h16.pt --modulus 323 --p 17
```

| probe | result | reading |
|---|---|---|
| exact accuracy over **all 288 units**, T=1 | 0.656 (189/288) | ~ the 250 training x memorised; the rest fail |
| same, T=2 | 0.667 | trained T, same story |
| same, T=4 / 8 / 16 / 32 / 64 | 0.031 / 0.049 / 0.028 / 0.021 / 0.028 | chance — **no** T-extrapolation |
| channels with `R^2(theta_k ~ x^2) > 0.99` | **0 / 8** | the phases are *not* a quadratic form in the value |
| best channel `R^2` | 0.570 | and its implied `w*N/2pi` is **0.005 cycles** — the phase barely wraps at all |
| spectral participation ratio of `cos(theta_k)` over the units | median 15.7 of 162 | not sparse; a Fourier solution would be ~1-3 |
| `rank@90%` of the pair-phase table | 7 of 8 | no low-rank "place x digit x frequency" structure |

**The learned solution is not the group one, on every measure.**  Note also the rung
profile: rungs that *share an exponent* (`2^T mod phi` gives T=8 and T=32 the same
exponent, and T=16 and T=64 the same) do **not** succeed or fail together — they are all
independently at chance, which is what a value lookup with no `T` structure looks like.

### 6.2 Mod-N symmetry of the learned map (label-free)

`(N-v)^2 = v^2 mod N`, so a block that genuinely squares mod `N` must give the same answer
for `v` and `N-v`.  Measured over all 288 units:

| model | agreement `pred(v) == pred(N-v)` |
|---|---|
| memorisation baseline `(250/288)^2` | 0.754 |
| single step (trained on T={1}) | **0.764** |
| iterated block (trained on T={1,2,3}) | **0.826** |

The single-step model has essentially *zero* mod-`N` structure beyond what memorising both
members of a pair gives you for free.  Iterating buys a real but small excess — the same
ordering as the held-out accuracy (0.000 -> 0.237).

### 6.3 Cross-modulus check (e2's N=899) — generalised in §7

The same offline crux, `digits(v) -> digits(v^2 mod 899)` from 250 of 840 units:

| | train exact | held-out exact |
|---|---|---|
| learned end-to-end | 1.000 | **0.003** |
| oracle phases, learned readout (K=161) | 1.000 | 0.614 |
| oracle phases, full frequency basis (K=449) | 1.000 | 0.614 |

Two things worsen with `N`: the learned model is *even more* purely memorising, and the
oracle ceiling itself drops to 0.61, because 250 training `x` no longer cover all the
distinct squares, so the readout is genuinely underdetermined.  **On e2 and above, the
group representation alone would not be enough even if it were found.**


## 7. THE COVERAGE CEILING — a hard bound on any value-indexed readout

This is the most transferable number in the report and it is not specific to my
architecture. Take the **oracle** setup of §5.2 (phases frozen at the exact additive
characters of `v^2`, so the representation is perfect) and vary only the modulus, keeping
the generator's `examples_per_setting = 250` training `x`:

| dataset | modulus | digits | units | distinct squares | **oracle held-out exact** | predicted `P[x^2 already seen]` |
|---|---|---|---|---|---|---|
| e1 | 323 = 17·19 | 3 | 288 | 72 | **1.000** | 1.000 |
| e2 | 899 = 29·31 | 3 | 840 | 210 | **0.614** | 0.614 |
| — | 2021 = 43·47 | 4 | 1932 | 483 | **0.337** | 0.337 |
| m1 | 10403 = 101·103 | 5 | 10200 | 2550 | **0.031** | 0.072 |

The last column is pure combinatorics — the fraction of held-out `x` whose square
`x^2 mod N` already occurs as the square of some training `x` — computed with no model at
all. **It predicts the measured oracle ceiling exactly at 323, 899 and 2021.**

Why: the readout maps a complete encoding of `v^2 mod N` to the digits of `v^2 mod N`, so
it is an *arbitrary function on `Z_N`*. Trained on the residues that appear in training, it
is unconstrained on any residue that does not. e1 is saved only by squaring being 4-to-1 on
`Z*_N` — 250 `x` cover all 72 squares. Nothing bigger is.

**Consequences the team should plan around:**

1. **e1 is the only public dataset where a value-indexed readout can certify even T=1.**
   At e2 the ceiling is 0.614 and certification needs 1.000, so *no* amount of
   architecture search or training on a Fourier/lookup readout can certify e2, let alone
   m1 or Hard.
2. The bound applies to **any** method whose final step is "learn a function of the
   residue" — a Fourier readout, a softmax over `Z_N`, an embedding table, a learned
   permutation. It is not a statement about my family only.
3. The escape is a readout that is **compositional in the digits** (per-digit prediction
   with carries), because that generalises across residues it has never seen. This is an
   independent, quantitative argument for the exact-arithmetic route — and it is the
   reason the ceiling problem gets *worse*, not better, at Medium and Hard.
4. It also explains the shape of §6.3: at N=899 even a perfect group representation is at
   0.61, so a 0.003 learned result there is not "the architecture is close and needs
   tuning" — the whole approach is capped well below certification.

Reproduce:

```bash
for cfg in "323 3 161" "899 3 449" "2021 4 1010" "10403 5 5201"; do
  set -- $cfg
  $VENV lab/probe_step.py --oracle --modulus $1 --slots $2 --freqs $3 \
      --harm 1 --train-x 250 --steps 1200 --wd 0.01 --lr 0.01
done
```

## 8. Making `T` applications actually happen (the selector fix)

§4.3 showed GRIter's learned soft selector never learns `T -> step count`. `lab/probe_sel.py`
reproduces GRIter's *training setting* offline (one target per row, a selector that must
read `T`) so variants cost ~2 minutes instead of a 10-minute evaluator run. e1 modulus,
250 training `x`, `T in {1,2,3}`, LOOPS=4, held-out exact at T=1 unless noted:

| selector | fits train? | **held-out T=1** | learned `T -> step` mapping |
|---|---|---|---|
| `soft` (what GRIter shipped) | no (0.72) | **0.000** | collapses onto the last step for every `T` |
| `entropy` penalty | no (0.08) | **0.000** | sharp but **`T`-independent** — same step for all `T` |
| `gumbel` straight-through | no (0.26) | 0.053 | sharp, wrong mapping (T=1→1, T=2→1, T=3→2) |
| `pointer` (ordered location) | partly (0.73) | 0.079 | **monotone and correct**, but blurred |
| `pointer` + `rand-loops` | yes (0.98) | 0.158 | monotone, sharper |
| `pointer` + `rand-loops` + annealed window | yes (1.00) | **0.237 – 0.263** | monotone, sharp |
| `oracle` one-hot at `T-1` *(diagnostic)* | yes (1.00) | **0.237 – 0.289** | — the ceiling |

**Findings:**

* **Sharpness is not the problem; `T`-dependence is.** An entropy penalty — the obvious
  fix, and the one I proposed in the first draft of this report — makes the selector
  one-hot on *the same step for every `T`*, and makes fitting worse. Recording that
  because it is the intuitive fix and it is wrong.
* Composition depth is an **ordered** quantity. Parameterising the selector as a learned
  scalar *location* on the step axis plus a window (instead of an unstructured L-way
  classifier) is what makes the correct mapping findable.
* **Randomising the depth budget per training step helps materially** (0.079 → 0.158),
  which is the lever `explore/tied-recurrence` had identified but not tested. Combined
  with an annealed window it closes the gap to the oracle entirely
  (**0.000 → 0.237–0.263 vs an oracle ceiling of 0.237–0.289**).
* Residual defect: across seeds the `T=1` cell is unstable (0.000 / 0.132 / 0.263) while
  `T=2,3` are consistently 0.21–0.26. `softplus` biases the pointer location away from
  index 0, so the *first* step is the hardest to point at — and T=1 is precisely the rung
  that `MAX_T=1` needs. An unconstrained (sign-free) location parameterisation is the
  obvious next fix.

**In the evaluator.** The selector fix was ported into GRIter (`--selector pointer
--rand-loops 1`, `lab/make_griter.py`) and queued on e1 at 20 000 fixed steps against two
controls (`pointer` without `rand-loops`, and `soft` with it), tag `griter-sel`. Those runs
had not returned when this was written — the GPU is shared with two other branches — so no
number is claimed for them here. **What is already measured is that they cannot help:** §9
shows the same architecture loses the entire gain in prompt parsing (0.25 -> 0.00) *before*
the selector is reached, and that `probe_learnability` — which does parse the prompt —
tracked the evaluator to within one example on every candidate tried.

**But note what this does and does not buy.** A perfect selector recovers exactly the
§5.6 number and no more: **0.254, against the 1.000 that certification requires.** The
selector was never the ceiling — the per-step arithmetic is. This is a mechanism repaired,
not a score moved.


## 9. Probe validity — where the offline gain is actually lost

The coordinator asked which candidates the probe passed and then failed in the evaluator.
The answer is sharp, and it identifies a bottleneck nobody had isolated.

Same architecture (GRIter, pointer selector, `rand-loops`), same modulus, same 250/38
split, same optimiser — only the **input representation** differs:

| what the model is given | train exact | **held-out T=1** | source |
|---|---|---|---|
| clean one-hot digit slots | 1.000 | **0.237 – 0.263** | `lab/probe_sel.py` |
| the actual prompt, absolute-position slots | 0.975 | **0.000** | `lab/probe_learnability.py` |
| the actual prompt, absolute + reverse-position slots | 1.000 | **0.000 – 0.026** | `lab/probe_learnability.py` |
| the actual prompt, real evaluator | 1.000 | **0.000 – 0.026** | `lab/archive.jsonl`, tag `griter` |

**The entire offline gain is lost in prompt parsing, not in the evaluator.** The probe and
the evaluator agree with each other (0.00–0.03 both); what they disagree with is the probe
that hands the model *parsed* input. So:

* **The offline probes are valid about the evaluator, but only if they parse the prompt.**
  `probe_step` / `probe_sel` measure the arithmetic core and systematically over-state what
  a submission will do; `probe_learnability` (which loads the real submission and feeds it
  real prompts) tracked the evaluator to within one example on every candidate tried. Use
  `probe_step`/`probe_sel` to *kill* candidates cheaply, and `probe_learnability` to
  *believe* one.
* **There is a third bottleneck.** `findings.md` identified two (iteration count;
  per-step arithmetic). This is a third, and it is worth ~0.25 of exact accuracy on its
  own: **recovering place-valued digit slots from a left-aligned prompt whose fields have
  variable length.** `x` occupies positions 5..4+len(x), so absolute position does not
  determine place value; distance from the end does, but only while the `T` field has a
  fixed digit count — and the ladder's T=16/32/64 are two digits where training's T=1,2,3
  are one. Adding a reverse-position-indexed table recovers only 0.00 → 0.026.

That last point is independently actionable and cheap to attack, and it is not specific to
this hypothesis family: **any** recurrent digit-arithmetic model has to solve it first.

## 10. What is falsified

1. **"Squaring is a rotation, so any `T` is free at constant depth."** Falsified for this
   input format. The rotation lives in multiplicative characters, whose encoder is a
   discrete-log lookup on the value, which provably cannot generalise to a held-out `x`.
   The compositional (additive-character) encoder makes squaring a degree-2 map, so `T`
   costs `2^T` degree — depth, not a phase multiplier.
2. **A learned `T`-conditioned angle multiplier buys nothing** (`a6_tcond`, 3 seeds):
   rung-1 0.009 vs control 0.000, mean 0.0350 vs 0.0278 — indistinguishable. It has
   nothing to multiply, because the state's phase is not a group element's phase (§6.1:
   0/8 channels are quadratic in `x`).
3. **Bilinear / multiplicative mixers, complex-rotation sublayers, outer-product pooling,
   Fourier readouts and untying the head are all null** on the metric: every variant is at
   `MAX_T = 0` and rung-1 within one example of the control.
4. **Grokking does not happen here, and "run it longer" will not fix it.** 20 000 steps is
   ~200x past memorisation and rung-1 is flat. §5.3 shows why: the objective is flat in the
   direction that matters.
5. **Small `K` / large `H` / strong weight decay do not force the Fourier solution** — the
   bottleneck that would prevent memorisation also prevents fitting (§5.5b).
6. **The rung profile shows no exponent structure.** `2^T mod phi` makes T=8 and T=32 the
   same exponent, and T=16 and T=64 the same, so a model with any `T` structure would
   succeed or fail on those rungs together. Across all 22 evaluator runs they move
   independently and all sit at chance (§6.1).

## 11. What survives, and the single highest-value recommendation

**Recommendation: stop trying to make `T` free. Put the budget on a weight-tied recurrent
block whose single step is *discrete* digit arithmetic (per-digit classification with
carries, not a continuous phase), make a row with time-step `T` actually pass through `T`
applications rather than a soft blend, and screen every candidate architecture with
`lab/probe_step.py` — 90 seconds, self-generated data — before spending an evaluator run
on it.**

Reasons, in order of evidential weight:

1. The only lever that moved held-out accuracy at all (0.000 -> 0.254 over 3 seeds) was
   **composing one shared block**, and composition extrapolates in `T` perfectly on values
   the model can handle (`T<=3` training -> `T=8` at 1.000 on seen `x`). The iteration
   half of "one layer deeper" is *solved*; the bottleneck is entirely the per-step map,
   exactly as `findings.md` concluded under the old metric — this branch now supplies the
   mechanism.
   *Immediate, cheap follow-up:* GRIter already implements the iterated block but its soft
   depth selector dissolves the constraint (§4.3). Annealing that selector to one-hot, or
   penalising its entropy via `training_loss`/`aux`, is a few lines and is the single most
   likely way to port the 0.254 offline gain into the evaluator.
2. Certification needs 38/38. Anything that carries a value in a *continuous* phase needs
   the phase commensurate with `N` to 1e-5, and nothing in the training signal supplies
   that. A representation that is **discrete in the value** (per-digit classification with
   carries, i.e. an exact-arithmetic step) has no such needle. The `exact-arithmetic`
   branch is where the remaining probability mass is.
3. **Screen in two stages** (§9). `lab/probe_step.py` / `probe_sel.py` cost ~90 s and
   kill a candidate's arithmetic core cheaply, but they hand the model parsed input and
   so over-state it by ~0.25. `lab/probe_learnability.py` loads the *real submission* and
   feeds it real prompts; it tracked the evaluator to within one example on every
   candidate here. Kill with the first, believe only the second.
4. **Solve prompt parsing as its own subproblem.** It costs 0.25 → 0.00 all by itself
   (§9) and it is architecture-independent, so it is worth someone's whole session
   regardless of which arithmetic core wins.
5. **Above e1, the representation alone stops being sufficient.** On e2's `N=899` even the
   *oracle* group representation tops out at 0.61 held-out from 250 training `x`, because
   the training set no longer covers the distinct squares (§6.3). Certification on the
   larger tiers therefore needs a readout that is itself structured in the value (digits +
   carries), not a table over `Z_N` — another argument for the exact-arithmetic route.
6. If anyone does revisit Fourier features, the scale-free `code(x)^2/(code(N) code(1))`
   construction in §5.5a is the right shape — it removes the flat direction and would
   transfer across sampled-`N` — but it needs the digit/place code supplied or trained
   with a ~1e-7 learning rate, and supplying it is hand-writing the value decoder.

## 12. Not attempted / out of scope

* Sampled-`N` datasets (`e5`, `hp3`): the plan was to test transfer only if something
  worked on fixed `N`. Nothing did, so these were not run — a fixed-`N` failure is
  strictly easier than a sampled-`N` one.
* `e2` / `m1` evaluator runs: §7 now gives a stronger reason than "not justified" — the
  *oracle* ceiling at e2 is 0.614 and at m1 is 0.031, so certification there is impossible
  for this family regardless of training. `probe_iter` separately shows the deeper-`T`
  regime diverges from random init.
* Nothing was submitted to the hosted service. No `one-layer login` / `submit` was run.

## 13. Reproduction

```bash
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
$VENV lab/make_manifest.py --dataset e1 --mode fixed_step --max-steps 2000  --seeds 74 75 76
$VENV lab/make_manifest.py --dataset e1 --mode fixed_step --max-steps 20000 --seeds 74
bash lab/sweep_gr_ablate.sh          # transformer ablation (8 x 3 seeds)
bash lab/sweep_gr_long.sh            # GRNet long runs
bash lab/sweep_griter.sh             # GRIter LOOPS sweep
bash lab/sweep_grpair_long.sh        # GRPair (K,H) grid -- WRITTEN BUT NOT RUN, see 4.4

# the offline crux (no dataset access, ~90s each)
$VENV lab/probe_step.py --freqs 32 --harm 8 --steps 20000          # learned  -> held-out 0.00
$VENV lab/probe_step.py --oracle --freqs 161 --harm 1 --steps 1500 # oracle   -> held-out 1.00
for e in 0 1e-6 1e-5 1e-4 1e-3 1e-2; do
  $VENV lab/probe_step.py --oracle --freq-jitter $e --freqs 161 --harm 1 --steps 1500
done                                                                # basin width
$VENV lab/probe_iter.py --time-steps 1 2 3 --freqs 32 --harm 8      # iteration -> held-out 0.25
$VENV lab/probe_iter.py --time-steps 4 8 16                         # deep T    -> diverges

# the coverage ceiling (6A) -- oracle representation vs modulus size
for cfg in "323 3 161" "899 3 449" "2021 4 1010" "10403 5 5201"; do set -- $cfg
  $VENV lab/probe_step.py --oracle --modulus $1 --slots $2 --freqs $3 \
      --harm 1 --train-x 250 --steps 1200 --wd 0.01 --lr 0.01
done

# the selector fix (6B)
$VENV lab/probe_sel.py --selector soft                              # -> 0.000
$VENV lab/probe_sel.py --selector entropy                           # -> 0.000 (sharp, T-independent)
$VENV lab/probe_sel.py --selector pointer --rand-loops --ptr-anneal  # -> 0.237-0.263
$VENV lab/probe_sel.py --selector oracle                            # -> 0.237-0.289 (ceiling)

# probe validity (6C): the SAME architecture, but parsing a real prompt
$VENV lab/probe_learnability.py --submission submissions/group-rotation/gi_ptr/submission.py
```

Every evaluator run is archived in `lab/archive.jsonl` (tags `baseline`, `gr-ablate`,
`grnet-long`, `griter`, `diag`, plus `smoke` / `lr-screen` / `wd-screen` / `speed`
screens) — 40+ runs this session, none deleted, including the ones that contradict the
hypothesis. The offline studies are ~25 further runs, reproducible from the commands
above.

## 14. Deliverable submission

`submissions/group-rotation/submission.py` is GRIter (§4.3). It is delivered because it
is the architecture that embodies this branch's one positive finding — iterating a single
shared block, with composition depth learned from the prompt — **not** because it beats
anything: like every other variant here it scores `MAX_T = 0` on e1, the same as the
baseline. Per-variant copies live in `submissions/group-rotation/<name>/submission.py`.
