# group-rotation — make squaring a rotation

**Branch:** `explore/group-rotation` · **Verdict: falsified as a learning route, with a
precise mechanism.** The group/Fourier representation is *exactly right* and *fully
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

<!--ABLATE_TABLE-->

### 4.2 GRNet, long runs (e1, 20 000 fixed steps, seed 74)

```
bash lab/sweep_gr_long.sh
```

<!--GRNET_TABLE-->

### 4.3 GRPair, (K, H) bottleneck grid (e1, 20 000 fixed steps, seed 74)

<!--GRPAIR_TABLE-->

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

Two things happen the moment there is more than one composed step:

* **Held-out accuracy goes from 0.000 to ~0.24–0.34** — an 8-10x jump over the best
  single-step model. Composing the block forces the decode to round-trip
  (`decode(phase(v^2))` must re-encode consistently), and a round-trip is only consistent
  when the phase is genuinely periodic mod `N`. **The T=1,2,3 rows of e1 are not extra
  data; they are the constraint that partially pins the frequency.**
* **`T` extrapolation on seen `x` is perfect** — trained on `T<=3`, the model is at 1.000
  on `T=4`. Composition depth transfers exactly. This is the competition's premise working.

It plateaus at ~0.3 though, invariant to `K`, `H`, softmax temperature, and weight decay —
the round-trip constrains the frequency but does not pin it to the 1e-5 needed.

**Deeper `T` training makes it worse, not better.** Training on `T in {4,8,16}` (m1's
ladder, i.e. up to 16 unrolled soft steps) **diverges** from random init — loss climbs from
2.6 to 8.9 and every rung sits at 0.000. So the extra round-trip constraint that deeper `T`
would supply is not collectible without a curriculum, and the evaluator forbids a custom
training loop. This independently reproduces `findings.md`'s "m1 is ~0 everywhere".

## 6. Is the learned solution the group one? (weight/activation diagnostics)

<!--DIAG_BLOCK-->

## 7. What is falsified

1. **"Squaring is a rotation, so any `T` is free at constant depth."** Falsified for this
   input format. The rotation lives in multiplicative characters, whose encoder is a
   discrete-log lookup on the value, which provably cannot generalise to a held-out `x`.
   The compositional (additive-character) encoder makes squaring a degree-2 map, so `T`
   costs `2^T` degree — depth, not a phase multiplier.
2. **A learned `T`-conditioned angle multiplier buys nothing** (`a6_tcond`, `gp_tcond`):
   at every step count it is indistinguishable from the control. It has nothing to
   multiply, because the state's phase is not a group element's phase.
3. **Bilinear / multiplicative mixers, complex-rotation sublayers, outer-product pooling,
   Fourier readouts and untying the head are all null** on the metric: every variant is at
   `MAX_T = 0` and rung-1 within one example of the control.
4. **Grokking does not happen here, and "run it longer" will not fix it.** 20 000 steps is
   ~200x past memorisation and rung-1 is flat. §5.3 shows why: the objective is flat in the
   direction that matters.
5. **Small `K` / large `H` / strong weight decay do not force the Fourier solution** — the
   bottleneck that would prevent memorisation also prevents fitting (§5.5b).

## 8. What survives, and the single highest-value recommendation

**Recommendation: stop trying to make `T` free, and put the whole budget on a
weight-tied recurrent block whose single step is exact digit arithmetic. Judge candidate
architectures by the offline crux test in `lab/probe_step.py` before spending an evaluator
run on them.**

Reasons, in order of evidential weight:

1. The only lever that moved held-out accuracy at all (0.00 -> 0.30) was **composing one
   shared block**, and composition extrapolates in `T` perfectly on values the model can
   handle. The bottleneck is entirely the per-step map, exactly as `findings.md` concluded
   under the old metric — this branch now supplies the mechanism.
2. Certification needs 38/38. Anything that carries a value in a *continuous* phase needs
   the phase commensurate with `N` to 1e-5, and nothing in the training signal supplies
   that. A representation that is **discrete in the value** (per-digit classification with
   carries, i.e. an exact-arithmetic step) has no such needle. The `exact-arithmetic`
   branch is where the remaining probability mass is.
3. `lab/probe_step.py` costs ~90 seconds and answers "can this architecture generalise to
   held-out `x` at all" far more sharply than a 20 000-step evaluator run. Every
   architecture should be screened there first. `--oracle` gives the achievable ceiling.
4. If anyone does revisit Fourier features, the scale-free `code(x)^2/(code(N) code(1))`
   construction in §5.5a is the right shape — it removes the flat direction and would
   transfer across sampled-`N` — but it needs the digit/place code supplied or trained
   with a ~1e-7 learning rate, and supplying it is hand-writing the value decoder.

## 9. Not attempted / out of scope

* Sampled-`N` datasets (`e5`, `hp3`): the plan was to test transfer only if something
  worked on fixed `N`. Nothing did, so these were not run — a fixed-`N` failure is
  strictly easier than a sampled-`N` one.
* `e2` / `m1` evaluator runs: `probe_iter` shows the deeper-`T` regime diverges from random
  init, and `findings.md` already records m1 at ~0; spending runs there was not justified.
* Nothing was submitted to the hosted service. No `one-layer login` / `submit` was run.

## 10. Reproduction

```bash
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
$VENV lab/make_manifest.py --dataset e1 --mode fixed_step --max-steps 2000  --seeds 74 75 76
$VENV lab/make_manifest.py --dataset e1 --mode fixed_step --max-steps 20000 --seeds 74
bash lab/sweep_gr_ablate.sh          # transformer ablation
bash lab/sweep_gr_long.sh            # GRNet long runs
bash lab/sweep_grpair_long.sh        # GRPair (K,H) grid

# the offline crux (no dataset access, ~90s each)
$VENV lab/probe_step.py --freqs 32 --harm 8 --steps 20000          # learned  -> held-out 0.00
$VENV lab/probe_step.py --oracle --freqs 161 --harm 1 --steps 1500 # oracle   -> held-out 1.00
for e in 0 1e-6 1e-5 1e-4 1e-3 1e-2; do
  $VENV lab/probe_step.py --oracle --freq-jitter $e --freqs 161 --harm 1 --steps 1500
done                                                                # basin width
$VENV lab/probe_iter.py --time-steps 1 2 3 --freqs 32 --harm 8      # iteration -> held-out 0.24
$VENV lab/probe_iter.py --time-steps 4 8 16                         # deep T    -> diverges
```

Every evaluator run is archived in `lab/archive.jsonl` (tags `baseline`, `gr-ablate`,
`grnet-long`, `grpair-grid`, `diag`, plus `smoke`/`lr-screen`/`wd-screen` screens).
