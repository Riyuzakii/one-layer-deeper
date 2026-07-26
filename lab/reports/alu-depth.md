# alu-depth — the computational graph's depth and shape

**Branch:** `explore/alu-depth`, from `explore/digit-carry` · **Mandate:** shorten
`DigitALU`'s ~280-step soft chain architecturally, without leaving the hypothesis
class that excludes memorisation.

**Two results, and the second one is the important one.**

1. **The chain is now 6.6× shorter and the hypothesis class is intact.**
   257 → **39** sequential soft steps at e1's modulus, 745 → **83** at m1's,
   with the constructed ceiling still 1.000 at every modulus and on 12 unseen
   sampled moduli, the state alphabet unchanged, and two extra parameters.
   `train_exact` at a fixed 2000 steps goes **0.132 → 0.396** (3 seeds), and
   0.620 at 20,000 steps.
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

`lab/probe_alu.py` gained `--reduce-mode {serial,binary,quotient}` and
`--mul-mode {horner,tree}`. Every mode keeps the same learned tensors
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
| tree | quotient | **39** | **55** | **83** | **155** |

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

`train_exact = held_exact = 1.000`, `held_ce = 0.0000` in every cell. Parameter
count 6,818 → **6,820** (the 5-parameter quotient scorer replaces the 3-parameter
gate). State alphabet: digit 10, carry 2, borrow 2, plus a quotient alphabet of
11 in the quotient modes. No tensor is indexed by a residue.

*(`--construct` is a LAB DIAGNOSTIC and is not a legal submission — rule 7. It is
here to certify that shortening the graph did not leave the class.)*

---

## 2. The variant table

(filled in below from `lab/logs/`)

---

## 3. What was falsified

---

## 4. Compliance
