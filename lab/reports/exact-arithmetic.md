# exact-arithmetic — input/output representation for exact digit arithmetic

Branch `explore/exact-arithmetic`. Metric: **Max T** (largest rung whose own and
every lower rung's exact accuracy is 100%). Everything below reports **per-rung
exact accuracy**, because MAX_T is 0 almost everywhere and rung-1 is the real
progress signal.

## 1. Hypothesis

A plain transformer over the flat digit string cannot align digit *places*
between `N`, `x` and the answer, which is what carrying and modular reduction
require. The claim under test is that the ~1–5 % plateau found by the previous
session is a **representation** failure, not a capacity/optimisation failure.

### 1.1 The structural fact the family rests on (verified, not assumed)

`lab/test_repr.py::check_target_alignment` builds prompts with the **public
tokenizer** (`data/squaring_mod.tokenize_squaring_mod_with_result`, no generated
data touched) and checks the evaluator's own `target_positions`:

```
answer digit of place value 10^j  is read at sequence position  (input_len - 1 - j)
```

i.e. *distance-from-end j == answer place value j*, exactly. Meanwhile the
prompt is `[N] d(N) [X] d(x) [T] d(T)` with **no leading zeros**, so
`input_len` varies with the digit count of `x` (on `e1`: 8, 9 or 10 tokens) and
of `T` (1 or 2 digits — the ladder goes to T=64). Consequences for a model with
a single absolute-from-left position embedding:

* the position that carries answer place 0 is 9 for a 3-digit `x` and 8 for a
  2-digit `x` — the readout mapping is **not a function of absolute position**;
* `x`'s own place values sit at different absolute positions per example;
* place k of `x` and place k of the answer share no index at all.

That is a concrete, verifiable misalignment, and it is what the axes below fix.

## 2. What was built

`lab/repr_template.py` + `lab/make_repr_submission.py` generate one submission
per configuration, differing in exactly one flag. Shared across every run:
weight-tied recurrent block, `D_MODEL=128`, 4 heads, `NUM_LOOPS=8`, FF mult 4,
AdamW lr 1e-3 wd 0.1, plain CE, manifest batch 512 — i.e. the previous
session's reference model, changed **only** in how tokens are embedded and
where the answer is read from.

| flag | meaning |
|------|---------|
| `USE_ABS` | learned absolute-from-left position embedding (baseline) |
| `USE_FIELD` | learned field id: digit-of-N / digit-of-x / digit-of-T / the marker token itself |
| `USE_PLACE` | learned place-value index **within its own field**, LSD = 0, one table shared by all fields (abacus / index-hint) |
| `USE_RPOS` | learned distance-from-end index — by §1.1 this *is* the answer's place value |
| `LAYOUT=slots_sep` | digits re-laid-out inside `forward` into place-aligned slots (N places, x places, T digits), LSD-first, answer read off the x-aligned slot of the same place |
| `LAYOUT=slots_sum` | one token per place value; N-digit and x-digit embeddings summed into that slot |

Slot layouts do their re-ordering with gathers inside `forward` and scatter the
per-place logits back onto the evaluator's own positions, so the
`(logits[B,L,17], aux)` contract is unchanged and the path stays differentiable
(`lab/test_repr.py` asserts every scored position is written by the scatter and
that the loss backprops to every parameter).

**Compliance note.** The field segmentation is a `cumsum` over the public marker
tokens and the place index is arithmetic on positions; no arithmetic on the
*numbers* is performed anywhere, there is no lookup table of answers, no Python
control flow is switched on data values (loop count is a constant), and every
prediction comes from parameters trained from random init in the run.

## 3. Results

(filled in below as runs land)

