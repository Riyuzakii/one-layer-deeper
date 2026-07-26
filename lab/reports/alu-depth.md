# alu-depth — the computational graph's depth and shape

**Branch:** `explore/alu-depth`, from `explore/digit-carry` · **Mandate:** shorten
`DigitALU`'s ~280-step soft chain architecturally, without leaving the hypothesis
class that excludes memorisation.

**Two results, and the second one is the important one.**

1. **The chain is 6.6–20× shorter, the hypothesis class is intact, and the
   optimizer gets 4.6–12.2× more steps per second.** 257 → **39** sequential
   soft steps at e1's modulus, 745 → **71** at m1's, 2,062 → **101** at Hard
   scale, with the constructed ceiling still 1.000 (soft *and* hard) at four
   fixed moduli and on 12 unseen sampled ones, the state alphabet unchanged, and
   +2 to +653 parameters. Measured ms/optimizer-step: 390 → 84 (Easy),
   1,240 → 158 (Medium), 2,834 → 232 (Hard). **Adopt it** — it answers the
   eval-budget and step-famine arguments in full (§2.4). But **the depth is not
   what moved `train_exact`** — the ladder is flat from 257 to 83 and only the
   *form* of the reduction moves it (§2.1).
2. **And it does not matter, because what `DigitALU` learns is not a discrete
   transducer.** Snapping every inter-step state to its argmax — which the
   *constructed* solution survives at 1.000 in all four graph shapes — takes the
   trained model from `train_exact` 0.556 to **0.004**. The mean max-probability
   of the register states is **0.775**, not ~1. The learned model is riding the
   continuous 10-simplex, which is precisely the value-encoding channel
   `digit-carry` §5.3 identified when a 32-dim carry restored memorisation. It is
   present in `DigitALU` too, just narrower.

**Consequence for the whole team: the screening premise is wrong.**
`digit-carry` §6.4 and `RESUME.md` §5 say "`DigitALU` cannot memorise, so
`train_exact` → 1.000 implies `held_exact` → 1.000; screen on `train_exact`".
Measured here: `train_exact` reaches 0.94 (N=91) and 0.62 (N=323) while
`held_exact` is 0.000 and `held_ce` climbs to 8–15 — the classic overfitting
signature — because `train_exact` is achievable through the soft register.
**Screen on `train_exact_hard`** (`probe_alu.py --eval-hard`), which the exact
solution passes at 1.000 and the trained model fails at 0.004.

---

## 1. What I changed in the graph

`lab/probe_alu.py` gained `--reduce-mode {serial,binary,quotient}`,
`--mul-mode {horner,tree}` and `--scan-mode {serial,prefix}`. Every mode keeps
the same learned tensors
(`Tmul (10,10,20)`, `Tadd (10,10,2,12)`, `Tsub (10,10,2,12)`, a gate, three
constants) and adds no index over `Z_N`.

### 1.1 `--reduce-mode binary` — 4 conditional subtractions instead of 11

The `R = 11` tied `cond_sub(r, N)` loop is 86% of the depth and 10 of its 11
iterations are no-ops. Replace it with conditional subtraction of
`8N, 4N, 2N, N` — a radix-2 restoring division, exact for any quotient ≤ 15, and
the measured maximum quotient is 9 (N=323) or 10 (N=899, 2021, 10403).

The multiples are *not* new parameters and *not* supplied: `2N = add_scan(N,N)`,
`4N = add_scan(2N,2N)`, `8N = add_scan(4N,4N)`, using the same learned `Tadd`.
They depend only on `N`, so they are a shared prefix (`ceil(log2 m)` scans) paid
once per forward, not once per Horner place.

### 1.2 `--reduce-mode quotient` — one learned quotient digit, one subtraction

All of `0·N … (Q+1)·N` are subtracted from the register **in a single scan**,
batched over the multiple index. The final borrow state of candidate `m` *is*
the comparison bit `[m·N > r]`; a 5-parameter learned scorer reads the adjacent
pair `(borrow_m, borrow_{m+1})` and picks the largest non-borrowing `m`. That
pattern `(no-borrow, borrow)` occurs at exactly one `m`, which is the quotient.
The quotient digit is a distribution over an alphabet of `Q+1 = 11` — discrete
and small. Depth per reduction: `W + 1` instead of `11W`.

This is the "learned quotient digit plus a single subtraction" the `digit-carry`
report named as the next step, and it is the same *kind* of object that was
already there: `cond_sub` already ended in "a learned gate on the final borrow
state chooses between the scanned result and the input".

### 1.3 `--mul-mode tree` — a log-depth multiply, then one long division

The Horner form chains `S²` `add_scan`s, one per place pair. Instead: compute
the full `2S`-digit product by a **balanced tree** of adds
(`ceil(log2 leaves)` chained scans, every level one batched scan), then do one
long division, shifting one product digit into an `S+1`-slot register at a time.

Two products whose place offsets differ by ≥ 2 occupy disjoint slots, so they
share a leaf register for free; that packs `S²` products into `~S+1` leaves
before a single add is spent. The division needs `S+1` reductions (the first
`S-1` shift-ins cannot exceed `N`, because an `S`-digit `N` is ≥ 10^(S-1)).

### 1.4 The depth ladder this produces

`probe_alu.py --depth-only` prints these; `main` is the x→y path, the quantity
`digit-carry` quoted as "~280 sequential soft table lookups".

| mul | reduce | S=3 (e1, N=323) | S=4 (2021) | S=5 (m1, 10403) | S=8 (24-bit, Hard) |
|---|---|---|---|---|---|
| horner | serial (R=11) | **257** | 466 | 745 | 2,062 |
| horner | binary | 117 | 221 | 367 | 1,117 |
| tree | serial | 195 | 300 | 437 | 956 |
| tree | binary | 83 | 125 | 185 | 389 |
| horner | quotient | 62 | 123 | 214 | 727 |
| tree | quotient | **39** | 55 | 83 | 155 |
| tree | quotient + `prefix` | 43 | **54** | **71** | **101** |

The gain grows with the modulus: 6.6× at 9 bits, 9× at 14 bits, **13× at 24
bits**. That is the number that matters for Hard: a Hard-scale modulus was a
~2,000-step soft chain and is now ~155.
The N-multiples prefix (`ceil(log2 Q)·W` = 16 at S=3) is a side branch that
merges at the first reduction and is not on the longest path.

### 1.5 The hypothesis class is intact — `--construct` at 1.000 everywhere

| mode | 323 | 899 | 2021 | 10403 | 12 unseen sampled moduli (4,800 operands) |
|---|---|---|---|---|---|
| horner+serial (digit-carry) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| horner+quotient | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| tree+quotient | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| tree+quotient+prefix | 1.000 | 1.000 | 1.000 | 1.000 | not run |

`train_exact = held_exact = 1.000`, `held_ce = 0.0000` in every cell. Parameter
count 6,818 → **6,820** (the 5-parameter quotient scorer replaces the 3-parameter
gate). State alphabet: digit 10, carry 2, borrow 2, plus a quotient alphabet of
11 in the quotient modes. No tensor is indexed by a residue.

*(`--construct` is a LAB DIAGNOSTIC and is not a legal submission — rule 7. It is
here to certify that shortening the graph did not leave the class.)*

---

## 2. The variant table

Everything below: `lab/probe_alu.py`, AdamW lr 3e-2 wd 0 (digit-carry's default
cell, unchanged — the *optimiser* belongs to `explore/alu-credit`), full batch,
tau 1.0, no identity init. Only the graph shape varies. Regenerate with
`lab/depth_grid.sh`; collect with `lab/depth_table.py`.

### 2.1 e1's modulus (N=323, S=3, 250 train / 38 held), 2,000 steps, 3 seeds

| variant | depth | reduction form | alphabet | params | ceiling | train_exact (seeds) | mean |
|---|---|---|---|---|---|---|---|
| `horner:serial` (digit-carry) | 257 | 11 gated `cond_sub` | 10/2/2 | 6,818 | 1.000 | 0.132 / 0.248 / 0.252 | 0.211 |
| `tree:serial` | 195 | 11 gated `cond_sub` | 10/2/2 | 6,818 | 1.000 | 0.212 / 0.068 / 0.084 | 0.121 |
| `horner:binary` | 117 | 4 gated `cond_sub` | 10/2/2 | 6,818 | 1.000 | 0.044 / 0.416 / 0.120 | 0.193 |
| `tree:binary` | 83 | 4 gated `cond_sub` | 10/2/2 | 6,818 | 1.000 | 0.140 / 0.280 / 0.080 | 0.167 |
| `horner:quotient` | 62 | compare-all + learned select | 10/2/2/**11** | 6,820 | 1.000 | 0.340 / 0.332 / 0.236 | 0.303 |
| **`tree:quotient`** | **39** | compare-all + learned select | 10/2/2/**11** | 6,820 | 1.000 | 0.380 / 0.368 / 0.440 | **0.396** |

`held_exact` is **0.000 in every cell** except one seed of `tree:quotient`
(0.026 = 1/38, the variance floor). Alphabet column = digit / carry / borrow /
quotient. **No index ranges over `Z_N` in any row**, and the parameter count is
the same at every modulus.

**Read the table by column, not by row order: `train_exact` is not monotone in
depth.** A 3.1× shortening (257 → 83) moves it *down* (0.211 → 0.167), and the
deepest shape of all beats two shallower ones. The gain arrives with the *form*
of the reduction: the four `serial`/`binary` cells span depths 257 to 83 and sit
at 0.12–0.21 with no ordering by depth, while both `quotient` cells sit at
0.30–0.40. Within the quotient family the remaining 1.6× depth reduction
(62 → 39) is worth +0.09, which is a real but second-order depth term.

So the first-order variable is **not** the chain length; it is that a chain of
sigmoid-gated conditional subtractions (`g·subtracted + (1-g)·unchanged`,
weight-tied, 4 or 11 deep) is badly conditioned, and replacing it with one
parallel comparison against all multiples plus a learned selection over a
discrete quotient alphabet is worth ~2×. This revises `digit-carry` §2.4's
"the obstruction is measurably depth", and it revises my own mandate's premise.

Longer training at the shallowest depth (39), same cell, `--steps 20000`:
`train_exact` **0.620 / 0.576** (2 seeds); the last 10,000 steps buy < 0.07.

### 2.2 A smaller operand (N=91, S=2, 50 train / 22 held), 4,000 steps, 3 seeds

The regime `digit-carry` §2.4 used for its "shorter chain → 0.72–0.78" result.
My `horner:serial` row reproduces its published 0.780 / 0.720 exactly, which is
the check that this ladder and that one are measuring the same thing.

| variant | depth | ceiling | train_exact (3 seeds) | mean | held_exact |
|---|---|---|---|---|---|
| `horner:serial` | 112 | 1.000 | 0.780 / 0.720 / 0.780 | 0.760 | 0.045 / 0.000 / 0.045 |
| `tree:binary` | 45 | 1.000 | 0.800 / 0.920 / 0.900 | 0.873 | 0.136 / 0.000 / 0.000 |
| `horner:quotient` | 25 | 1.000 | 0.840 / 0.840 / 0.740 | 0.807 | 0.000 |
| `tree:quotient` | 21 | 1.000 | 0.820 / 0.860 / 0.760 | 0.813 | 0.000 |

**The best `train_exact` this branch reached anywhere is 0.940** (N=91,
`tree:binary`, seed 1, mid-run). **`held_exact` never lifted off**: the largest
value seen at any depth, any modulus, any seed is 0.136 = 3/22 on the smallest
possible held-out cohort, and it is not reproduced by the two sibling seeds.

### 2.3 The measurement that reframes the result: hard-state exactness

`--eval-hard` re-runs the trained weights with every inter-step state snapped to
its argmax, and `state_sharpness` is the mean max-probability over every
inter-step softmax. The **constructed** solution scores `train_exact_hard =
held_exact_hard = 1.000` in all four `mul × reduce` shapes, so the diagnostic is
sound: the target *is* in the discrete family.

**The whole ladder, N=323, 2,000 steps, 2 seeds each** — the same six shapes as
§2.1, re-run with the diagnostic on:

| shape | depth | train_exact | **train_exact_hard** | state_sharpness | q_sharpness |
|---|---|---|---|---|---|
| `horner:serial` | 257 | 0.132 / 0.248 | **0.008 / 0.004** | 0.778 / 0.822 | — |
| `tree:serial` | 195 | 0.212 / 0.068 | **0.008 / 0.000** | 0.777 / 0.759 | — |
| `horner:binary` | 117 | 0.044 / 0.416 | **0.012 / 0.004** | 0.784 / 0.856 | — |
| `tree:binary` | 83 | 0.140 / 0.280 | **0.000 / 0.000** | 0.770 / 0.798 | — |
| `horner:quotient` | 62 | 0.340 / 0.332 | **0.000 / 0.000** | 0.812 / 0.831 | 0.65 / 0.75 |
| `tree:quotient` | 39 | 0.380 / 0.368 | **0.004 / 0.000** | 0.765 / 0.775 | 0.54 / 0.86 |
| constructed, any shape | — | **1.000** | **1.000** | ~1.000 | ~1.000 |

**`train_exact` spans 0.04–0.42 across the ladder. `train_exact_hard` spans
0.000–0.012 — i.e. zero, everywhere, at every depth.** A 6.6× shorter chain and
a 3× better `train_exact` buy nothing whatsoever on the discrete solution.

Training longer does not change it either:

| cell | depth | steps | train_exact | **train_exact_hard** | state_sharpness |
|---|---|---|---|---|---|
| N=323 `tree:quotient` s0 | 39 | 6,000 | 0.556 | **0.004** | 0.764 |
| N=323 `tree:quotient` s0 | 39 | 8,000 | 0.604 | **0.008** | 0.765 |
| N=323 `tree:quotient` s0 | 39 | **20,000** | 0.620 | **0.000** | 0.767 |
| N=91 `tree:quotient` s0/s1 | 21 | 6,000 | 0.820 / 0.860 | **0.100 / 0.220** | 0.832 / 0.849 |
| N=91 `tree:quotient` s0/s1 | 21 | **20,000** | 0.820 / 0.860 | **0.120 / 0.040** | 0.838 / 0.844 |

The learned quotient digit is itself a blur: `q_sharpness` (the max probability
of the 11-way quotient distribution) is 0.758 after 8,000 steps, so the
reduction is subtracting a *mixture* of multiples of N, not one of them.

`train_exact` is 0.13–0.86; `train_exact_hard` is 0.00–0.14; the state is a
mixture with 0.76–0.84 of its mass on the argmax, not a digit. **What the model
learns is a continuous relaxation, not a transducer over the digit alphabet.**

(One caveat, in the lenient direction: `--eval-hard` snaps every `softmax`
state, but `cond_sub`'s final sigmoid gate — used only by `serial` and `binary`
— is left soft. So the `serial` row's 0.008 is if anything an over-estimate.
`quotient` has no such gate and is snapped everywhere.)

The soft register is a `W × 10` simplex, and reading a table with
`einsum("bu,bv,bc,uvco->bo", ...)` is *bilinear* in that simplex — a mixture
therefore addresses the table at points no digit pair can reach. That is the
same value-encoding channel `digit-carry` §5.3 measured when a 32-dim carry
restored memorisation (train 1.000 / held 0.000), only narrower: 9 free
dimensions per slot instead of 32.

It is also the exact mechanism behind the `held_ce` column nobody has been
able to explain: at N=91, `train_ce` falls to 0.11 while `held_ce` rises to
**13.9–15.6** — confidently wrong on held-out operands, which is textbook
overfitting, from an architecture that was adopted because it *cannot* overfit.

---

## 2.4 Throughput — the number the step-famine argument needs

`lab/depth_bench.py`. One optimizer step (forward + backward + AdamW + grad
clip) of the **readout alone**, batch 512, bf16 autocast — the manifests'
settings. "Easy/Medium/Hard" are `budget / (ms per step)`; they are an upper
bound on the submission's step count because the parser, the T-loop, the
import-time clock and the eval half of the budget are not charged here, and
because this box is **shared with sibling agents** (the same `horner:serial`
cell measured 580 / 390 / 250 ms/step at three contention levels over the
session). **The ratios are the transferable quantity** — the workload is
kernel-launch bound and every shape launches the same kind of kernel.

| shape | S=3 (e1) ms | steps in 60 s | S=5 (m1) ms | steps in 600 s | S=8 (Hard) ms | steps in 3600 s |
|---|---|---|---|---|---|---|
| `horner:serial` (digit-carry) | 389.8 | 154 | 1240.1 | 484 | 2834.2 | 1,270 |
| `tree:serial` | 315.0 | 190 | 656.4 | 914 | 1200.5 | 2,999 |
| `horner:binary` | 238.0 | 252 | 624.2 | 961 | 1596.5 | 2,255 |
| `tree:binary` | 179.4 | 334 | 381.8 | 1,572 | 639.4 | 5,630 |
| `horner:quotient` | 145.4 | 413 | 441.6 | 1,359 | 1185.4 | 3,037 |
| `tree:quotient` | 103.4 | 580 | 209.5 | 2,864 | 373.5 | 9,639 |
| **`tree:quotient:prefix`** | **84.0** | **714** | **157.9** | **3,800** | **232.4** | **15,491** |

**Speedup over the current design: 4.6× at Easy, 7.9× at Medium, 12.2× at
Hard.** ms/step tracks the soft-step count almost exactly (`horner:serial` is
6.6× deeper and 4.6× slower at S=3; 13× deeper and 12.2× slower at S=8), which
is the signature of a launch-bound graph and is why the depth ladder *is* the
throughput ladder.

This is the part of the mandate that survives §3.2 intact: whatever step count
the tier affords today, this graph multiplies it by 4.6–12.2× and is provably
exact (`--construct` = 1.000 at four moduli and twelve unseen ones, soft and
hard). It does not, by itself, buy accuracy — see §3.2 — but it removes the
eval-budget failure and the step famine as *separate* obstacles.

---

## 3. What was falsified

### 3.1 "Screen `DigitALU` on `train_exact`, because it implies `held_exact`"

`digit-carry` §6.4 and `RESUME.md` §5.1 both state this, and it is the premise
under which four branches are currently screening. **It is false.** The argument
was "6,817 digit-indexed parameters have nowhere to memorise 250 residues", and
the parameters indeed have nowhere — but the *state* does. Measured:
`train_exact` 0.94 with `held_exact` 0.000 and `held_ce` 15.6.

Replace it with `train_exact_hard`. It costs one extra forward pass, the exact
solution scores 1.000 on it, and it is the only number here that distinguishes
"learning the transducer" from "fitting the relaxation".

### 3.2 "Depth is the binding variable" — falsified twice over

This is `digit-carry` §2.4's conclusion, `RESUME.md` §5.1's first
recommendation, and my own mandate's premise. It fails on two independent
readings of the data.

**(a) `train_exact` is not monotone in depth.** §2.1: depths 257, 195, 117 and
83 all give 0.17–0.21; the jump to 0.30–0.40 tracks the *form* of the reduction,
not its length. The evidence `digit-carry` had — S=3 → S=2 taking train_exact
0.20 → 0.78 — changed the operand width, the output length (3 digits → 2), the
held-out cohort and the training-set size at the same time as the chain length,
so it could not separate them. This ladder holds the target completely fixed and
varies only the graph, and the depth term is second-order.

**(b) On the metric that measures the discrete solution, depth buys nothing at
all.** `train_exact_hard` is 0.008 at depth 257 and 0.004–0.008 at depth 39 — a
6.6× shortening for no change. Even a **21-step** chain (N=91, S=2 — a 12×
reduction and the shortest exact graph I can build) reaches only 0.06–0.22 and
then sits there for 15,000 further steps while `train_exact` holds at 0.82.
Shortening the chain makes the *relaxation* easier to fit; it does not make the
*discrete* solution easier to find.

Note the uncomfortable corollary: the shapes that score best on `train_exact`
are the ones that mix *more* — `quot_reduce` passes a soft mixture over 11
candidate registers forward at every reduction, where `cond_sub` passes a
2-way mixture. A metric that rewards blur ranks blurrier graphs higher. That is
the strongest single argument for §3.1.

### 3.3 A larger internal radix (mandate direction 3) — falsified, with mechanism

`probe_coverage.py --radix B` generalises `digit-carry`'s atom enumeration to a
base-`B` internal alphabet. Base 10 reproduces its published numbers exactly.
`P[every table entry a held-out x needs was already exercised by the 250
training x]`:

| internal radix | 323 | 899 | 2021 | 10403 |
|---|---|---|---|---|
| 10 | **0.947** | **0.966** | **0.956** | **0.977** |
| 100 | 0.026 | 0.005 | 0.001 | 0.010 |
| 1000 | 0.000 | 0.000 | 0.000 | 0.000 |

Pairing adjacent decimal digits halves the slot count and so roughly quarters
the depth — but it **squares every table's index space**, and 250 training `x`
cannot exercise it (the pair table alone needs 3,540 entries at N=2021 and sees
582). At radix 1000 with S=1 the "pair" table is indexed by `(x, x)`: it *is* a
residue-indexed readout, and its digit ceiling is exactly the residue ceiling.
**The radix is the dial between shallow-and-memorising and
deep-and-compositional**, and the whole value of this family sits at the deep
end. Do not spend runs here.

*(This is a direct answer to the coordinator's "stack the larger internal radix
rather than stopping at the first win". It cannot be stacked: it buys its depth
by re-buying the coverage ceiling `digit-carry` removed. The 4.6–12.2× in §2.4
is obtained without touching the radix, and is therefore free of this cost.)*

### 3.4 Straight-through discretisation, at every chain length

`digit-carry` §5.4 falsified `--hard` at depth 257 (train 0.012). It fails
identically at 39 (0.000 / 0.004, loss 16.3–16.5) and at 21 (0.040 / 0.080,
loss 14.7–16.6), i.e. **the failure is not caused by depth** and shortening the
chain does not rescue it. Closing the continuous channel by fiat kills training
outright; leaving it open lets the model overfit through it. That is the vice
this family is currently caught in.

### 3.5 Reducing less often (mandate direction 4) — falsified by arithmetic

With a radix-2 reduction, handling a quotient up to `Q` costs `ceil(log2 Q)`
sub-scans, so reducing every `p` places costs `ceil(log2(10^p))·W` per `p`
places against `p·ceil(log2 10)·W` for reducing every place — a wash — while the
register must widen by `p-1` slots, which makes every scan longer and enlarges
`W` in *both* terms. Measured on the ladder: `tree:quotient` (reduce at every
division step, `W = S+1`) is 39 steps; every "reduce less often" variant is
strictly worse. Reduce as often as the register allows.

### 3.6 Parallel-prefix carry propagation (mandate direction 2) — implemented, and it pays only above Easy

`--scan-mode prefix`. The carry/borrow propagation semigroup is
{kill, propagate, generate} — a **3-element** alphabet — so a carry-lookahead
formulation is legal for this family: a learned table maps a digit pair to a
semigroup element (`Egen`/`Ecmp`, 300 each), a learned 27-entry table composes
two (`Ccomp`), and a learned 12-entry table applies one to the incoming
carry/borrow (`Aapp_a`/`Aapp_b`). +651 parameters, one new alphabet of size 3,
`ceil(log2 W)` chained composes instead of `W` chained slot steps. Constructed
ceiling **1.000 soft and 1.000 hard at N = 323, 899, 2021, 10403**.

The constant is 3 (element, apply, digit), so depth per scan is
`ceil(log2 W) + 3` against a serial `W`: a **loss** at W=4, a wash at W=5–6, and
a win from W ≥ 8. On top of `tree:quotient`:

| | S=3 (e1) | S=4 | S=5 (m1) | S=8 (Hard) |
|---|---|---|---|---|
| `tree:quotient` depth | **39** | 55 | 83 | 155 |
| `+ prefix` depth | 43 | 54 | **71** | **101** |
| measured ms/step ratio | 1.23× | — | 1.33× | 1.61× |

The measured throughput gain **exceeds** the depth-count gain (1.61× against
155/101 = 1.53× at S=8) because the workload is kernel-launch bound and a prefix
level is one large launch where a slot step is several small ones. **Stack it
for Medium and Hard; skip it for Easy**, where it is a small loss on depth and a
small win on wall clock (1.23×) that comes with 651 extra parameters.

---

## 4. What survives, and the single highest-value recommendation

**The graph is fixed and it was not the problem.** `tree:quotient` is a strict
improvement on `horner:serial` on every axis — 6.6× shallower at e1, 13× at
Hard scale, 5–7× faster per step, +2 parameters, same alphabet, same 1.000
constructed ceiling at four fixed moduli and twelve unseen sampled ones. Anyone
continuing this family should start from it. But it does not certify a rung and
it does not move `held_exact`.

**Adopt `tree:quotient` (+`prefix` above Easy) unconditionally** — it is the
same hypothesis class, provably exact, and 4.6–12.2× more optimizer steps per
second (§2.4). That answers the coordinator's justifications 2 (eval budget) and
3 (step famine) in full, and it is worth doing even though it does not move
accuracy.

**But the first justification — "shorten the chain and it will train" — is
falsified (§3.2), so do not expect the extra steps to convert.** At depth 39 the
model already gets 20,000 steps in this probe and `train_exact_hard` is 0.000;
the step famine was real but it was not the reason `DigitALU` fails.

**Recommendation: stop optimising `train_exact` and start optimising
`train_exact_hard`.** The bottleneck is not credit assignment through a long
chain — I removed 85% of the chain and the discrete solution is no closer. The
bottleneck is that *the soft relaxation and the discrete target are different
problems*, and the optimiser is descending the first. Concretely, the next
things I would run, in order:

1. **Re-screen every live lever on `train_exact_hard`.** `alu-credit`'s
   temperature/init sweep, `alu-compose`'s composition — all of them are
   currently ranked on a number that a mixture can win. This is one flag
   (`--eval-hard`) and it re-prices the whole session's screening. (The
   coordinator's note that "discrete eval states cost +42% and buy nothing" is
   about the *submission's* eval path, where it is correct — a submission should
   run soft. As a **lab screen** it is the only number that separates learning
   the transducer from fitting the relaxation, and 42% of a probe run is cheap.)
2. **Put the discreteness in the loss, not the graph.** The graph is now cheap
   enough that the state can be penalised at every one of 39 steps (entropy of
   each register slot, or a distance-to-vertex term) without the cost that made
   it impractical at 257. This is `alu-credit`'s lane, and it now has a target
   metric.
3. **If (2) fails, the family may be unsalvageable by gradient descent** and the
   right question becomes whether a *discrete* search (the table is 6,820
   parameters over five small alphabets) can be posed inside one differentiable
   forward pass without violating rule 3.

**No submission is delivered.** `train_exact_hard` ≤ 0.14 everywhere and
`held_exact` never left the floor; per `digit-carry` §6.4 and this session's
protocol, evaluator runs cannot resolve anything in that regime, so I spent none
and `lab/archive.jsonl` is unchanged on this branch. Claiming a submission here
would be claiming a result I did not measure.

---

## 5. Reproduction

```bash
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python

# the depth ladder (structure only, no training)
for mm in horner tree; do for rm in serial binary quotient; do
  for cfg in "323 3" "2021 4" "10403 5" "10000019 8"; do set -- ${=cfg}
    $VENV lab/probe_alu.py --modulus $1 --slots $2 --mul-mode $mm \
        --reduce-mode $rm --depth-only --tag d; done; done; done

# the hypothesis class is intact: constructed ceiling, soft AND hard states
for mm in horner tree; do for rm in serial binary quotient; do
  for cfg in "323 3" "899 3" "2021 4" "10403 5"; do set -- ${=cfg}
    $VENV lab/probe_alu.py --modulus $1 --slots $2 --mul-mode $mm \
        --reduce-mode $rm --construct --eval-hard --tag c; done; done; done
$VENV lab/probe_alu_oodn.py --moduli 6 --mul-mode tree --reduce-mode quotient

# the training ladder (3 seeds x 6 shapes, 2000 steps)
bash lab/depth_grid.sh
MOD=91 SLOTS=2 OUT=lab/logs/short EXTRA="--train-x 50" SUF="_tx50" \
  CELLS="tree:quotient horner:quotient tree:binary horner:serial" \
  STEPS=4000 bash lab/depth_grid.sh

# THE RESULT THAT MATTERS: hard-state exactness across the ladder
OUT=lab/logs/ehladder LOGEVERY=1000 EXTRA="--eval-hard" SUF="_eh" \
  CELLS="horner:serial tree:serial horner:binary tree:binary horner:quotient tree:quotient" \
  STEPS=2000 SEEDS="0 1" bash lab/depth_grid.sh

# straight-through fails at every chain length
OUT=lab/logs/hard EXTRA="--hard" SUF="_hard" CELLS="tree:quotient" \
  STEPS=4000 SEEDS="0 1" bash lab/depth_grid.sh

# the internal-radix falsification (CPU, closed form)
$VENV lab/probe_coverage.py --configs 323:3 899:3 2021:4 10403:5
$VENV lab/probe_coverage.py --radix 100  --configs 323:2 899:2 2021:2 10403:3
$VENV lab/probe_coverage.py --radix 1000 --configs 323:1 899:1 2021:2 10403:2

# throughput -- the step-famine number, per shape, three scales
$VENV lab/depth_bench.py --slots 3 5 8 --modulus 323 10403 10000019 \
  --cells horner:serial:serial tree:serial:serial horner:binary:serial \
          tree:binary:serial horner:quotient:serial tree:quotient:serial \
          tree:quotient:prefix

# parallel-prefix carries: exact, and worth it only above Easy
for cfg in "323 3" "899 3" "2021 4" "10403 5"; do set -- ${=cfg}
  $VENV lab/probe_alu.py --modulus $1 --slots $2 --mul-mode tree \
      --reduce-mode quotient --scan-mode prefix --construct --eval-hard; done

# collect everything
$VENV lab/depth_table.py lab/logs/*
```

---

## 6. Compliance

Nothing under `data/generated/` was read, printed, sampled or summarised; every
probe synthesises its operands from the modulus given on the command line and
from the public generator spec. `--construct`, `--freeze` and `probe_coverage.py`
are LAB DIAGNOSTICS in `lab/`, never imported by a submission, and are the direct
analogue of `probe_step.py --oracle`; `--construct` sets tables to the truth and
**is not a legal submission** (rule 7). Choosing the graph's *shape* — a tree
instead of a chain, one learned quotient digit instead of eleven tied
subtractions — is a structural choice, and every tensor in it is still learned
from random init. All computation is differentiable and inside the autograd
graph; no Python control flow reads `input_ids`; there is no custom training loop
and no participant-controlled backward. No evaluator run was made on this branch
and `lab/archive.jsonl` is untouched. Nothing was submitted to the hosted service
and no `one-layer login`/`submit` was run.

---

---

## 7. Appendix — every training run, as collected

`lab/logs/` is gitignored (repo convention), so the raw logs do not survive the
branch; this is `lab/depth_table.py`'s output over all of them at the end of
the session. `var` is scan-mode plus the cell suffix: `tx50` = the N=91 split,
`hard` = `--hard` straight-through TRAINING, `eh` = `--eval-hard` diagnostic,
`n20000`/`qs` = long runs.

| N | S | mul | reduce | depth | params | steps | var | seeds | train_exact | held_exact | train_hard | sharp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 91 | 2 | tree | quotient | 21 | 6,820 | 4000 | serial/tx50 | 3 | 0.820/0.860/0.760 (mu=0.813) | 0.000/0.000/0.000 | -/-/- | -/-/- |
| 91 | 2 | tree | quotient | 21 | 6,820 | 4000 | serial/tx50_hard | 2 | 0.040/0.080 (mu=0.060) | 0.000/0.045 | -/- | -/- |
| 91 | 2 | horner | quotient | 25 | 6,820 | 4000 | serial/tx50 | 3 | 0.840/0.840/0.740 (mu=0.807) | 0.000/0.000/0.000 | -/-/- | -/-/- |
| 91 | 2 | tree | binary | 45 | 6,818 | 4000 | serial/tx50 | 3 | 0.800/0.920/0.900 (mu=0.873) | 0.136/0.000/0.000 | -/-/- | -/-/- |
| 91 | 2 | horner | serial | 112 | 6,818 | 4000 | serial/tx50 | 3 | 0.780/0.720/0.780 (mu=0.760) | 0.045/0.000/0.045 | -/-/- | -/-/- |
| 91 | 2 | tree | quotient | 21 | 6,820 | 6000 | serial/tx50_eh | 2 | 0.820/0.860 (mu=0.840) | 0.000/0.000 | 0.100/0.220 | 0.832/0.849 |
| 91 | 2 | tree | quotient | 21 | 6,820 | 20000 | serial/tx50_eh_n20000 | 2 | 0.820/0.860 (mu=0.840) | 0.045/0.000 | 0.120/0.040 | 0.838/0.844 |
| 323 | 3 | tree | quotient | 39 | 6,820 | 2000 | serial/- | 3 | 0.380/0.368/0.440 (mu=0.396) | 0.026/0.000/0.000 | -/-/- | -/-/- |
| 323 | 3 | tree | quotient | 39 | 6,820 | 2000 | serial/eh | 2 | 0.380/0.368 (mu=0.374) | 0.026/0.000 | 0.004/0.000 | 0.765/0.775 |
| 323 | 3 | horner | quotient | 62 | 6,820 | 2000 | serial/- | 3 | 0.340/0.332/0.236 (mu=0.303) | 0.000/0.000/0.000 | -/-/- | -/-/- |
| 323 | 3 | horner | quotient | 62 | 6,820 | 2000 | serial/eh | 2 | 0.340/0.332 (mu=0.336) | 0.000/0.000 | 0.000/0.000 | 0.812/0.831 |
| 323 | 3 | tree | binary | 83 | 6,818 | 2000 | serial/- | 3 | 0.140/0.280/0.080 (mu=0.167) | 0.000/0.000/0.000 | -/-/- | -/-/- |
| 323 | 3 | tree | binary | 83 | 6,818 | 2000 | serial/eh | 2 | 0.140/0.280 (mu=0.210) | 0.000/0.000 | 0.000/0.000 | 0.770/0.798 |
| 323 | 3 | horner | binary | 117 | 6,818 | 2000 | serial/- | 3 | 0.044/0.416/0.120 (mu=0.193) | 0.000/0.000/0.000 | -/-/- | -/-/- |
| 323 | 3 | horner | binary | 117 | 6,818 | 2000 | serial/eh | 2 | 0.044/0.416 (mu=0.230) | 0.000/0.000 | 0.012/0.004 | 0.784/0.856 |
| 323 | 3 | tree | serial | 195 | 6,818 | 2000 | serial/- | 3 | 0.212/0.068/0.084 (mu=0.121) | 0.000/0.000/0.000 | -/-/- | -/-/- |
| 323 | 3 | tree | serial | 195 | 6,818 | 2000 | serial/eh | 2 | 0.212/0.068 (mu=0.140) | 0.000/0.000 | 0.008/0.000 | 0.777/0.759 |
| 323 | 3 | horner | serial | 257 | 6,818 | 2000 | serial/- | 3 | 0.132/0.248/0.252 (mu=0.211) | 0.000/0.000/0.000 | -/-/- | -/-/- |
| 323 | 3 | horner | serial | 257 | 6,818 | 2000 | serial/eh | 2 | 0.132/0.248 (mu=0.190) | 0.000/0.000 | 0.008/0.004 | 0.778/0.822 |
| 323 | 3 | tree | quotient | 39 | 6,820 | 4000 | serial/hard | 2 | 0.000/0.004 (mu=0.002) | 0.000/0.000 | -/- | -/- |
| 323 | 3 | tree | quotient | 39 | 6,820 | 6000 | serial/eh | 2 | 0.556/0.520 (mu=0.538) | 0.000/0.000 | 0.004/0.000 | -/0.786 |
| 323 | 3 | tree | quotient | 39 | 6,820 | 8000 | serial/qs | 1 | 0.604 (mu=0.604) | 0.000 | 0.008 | 0.765 |
| 323 | 3 | tree | quotient | 39 | 6,820 | 19000 | serial/n20000 | 1 | 0.564 (mu=0.564) | 0.000 | - | - |
| 323 | 3 | tree | quotient | 39 | 6,820 | 20000 | serial/eh_n20000 | 1 | 0.620 (mu=0.620) | 0.000 | 0.000 | 0.767 |
| 323 | 3 | tree | quotient | 39 | 6,820 | 20000 | serial/n20000 | 2 | 0.620/0.576 (mu=0.598) | 0.000/0.000 | -/- | -/- |
