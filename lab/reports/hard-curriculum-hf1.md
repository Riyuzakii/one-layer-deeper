# `hard/curriculum-hf1` — a loss-side curriculum over modulus size and operand magnitude

**Verdict: NULL, and closed for a mechanical reason rather than a tuning one.**
Draft — numbers are filled in as runs land. See §9 for the state of play.

Method #4 in `lab/RANKING.md`. Ranked below the architectural entries, and the
brief asked for a clean negative if that is what the evidence says. It is.

---

## 1. The hypothesis, and the one thing that makes it worth testing

`hf1` contains **three modulus sizes — 16, 18 and 20 bits — in one training
set**, and a `DigitALU`'s tables are modulus-independent by construction, so
*every* example trains the *same* ~6.8k parameters whatever its modulus. That
makes "upweight the 16-bit examples early and anneal toward 20-bit" a genuine
curriculum over difficulty on shared parameters — not a data-order trick, which
the evaluator's fixed loop would forbid. Operand magnitude within a modulus is a
second, finer axis.

The honest prior, stated in the brief: `explore/alu-credit` measured a magnitude
curriculum as null (`--xcurr 0.5`, `train_exact_hard` 0.012 against a 0.008
baseline), but at **e1 scale on a single 3-digit modulus**, where the
modulus-size axis does not exist at all. Whether the axis behaves differently on
a dataset that actually contains three sizes was untested.

---

## 2. What I built, and why it had to be new

`explore/alu-depth`'s `DigitALU` and `explore/alu-population`'s `PopALU` both
take **one modulus per forward** (`ndig` is a single `(W,10)` tensor). A
curriculum over modulus size needs several moduli in the same batch, so
`lab/probe_curric.py` carries a **per-example modulus**: `ndig` is `(b, W, 10)`,
`multiples` is `(b, M, W, 10)`, and the multiples-contracted subtract table is
formed once per forward. The graph is otherwise exactly
`probe_alu_depth.DigitALU(mul_mode='tree', reduce_mode='quotient',
scan_mode='serial')` — the corrected `tree:quotient` the ranking names — at
`S = 7` slots, which is what a 20-bit (and a 21-bit OOD-N) modulus needs.

**Base architecture, and why.** The corrected `DigitALU`: `tree:quotient`,
untied, 6,820 parameters, no index ranging over `Z_N`. It is the only
architecture in the project whose *representation* survives `hf1`'s
modulus split — test moduli never appear in training, so residue-indexed
readouts are dead by construction (`RESUME.md` Round 6). A curriculum over
modulus size is only meaningful on a readout whose parameters are shared across
moduli, which singles this family out.

`EMB_INIT = 0.02` is an embedding-init correction and has no embedding to act on
in the probe; it is applied in the shipped submission, which does have one.

**A correctness consequence of mixing sizes, and it is load-bearing.** With a
*fixed* slot count shared across modulus sizes, the published reduction schedule
("reduce only for `t <= S`") is **wrong for the small moduli**: after `2S-1-S`
unreduced places the register can exceed `10·N` for a 16-bit `N`, so the true
quotient leaves the `Q+1` alphabet. `probe_curric.py` therefore reduces at
**every** place, which keeps `r < 10·N` at every modulus size and costs 14
reductions instead of 8 (graph depth **183** sequential softmax steps instead of
~129, plus a 32-step shared `N`-multiples prefix). Anyone else running a
fixed-`S` ALU on a mixed-modulus dataset needs this.

---

## 3. Gates, run before any curriculum run

| gate | result |
|---|---|
| **constructed ceiling** (DIAGNOSTIC) | `train`/`held_x`/`held_n` **1.000 soft AND hard** at 16/18/20 bits, and **1.000** at OOD-N 17/19/21 |
| `assert_model_state` | 6,820 / 500,000,000 |
| true max quotient | 9, alphabet covers 0..10 |
| **`--lr 0` control** (LEGAL) | exact **0.000** everywhere; `dacc` 0.105; `local_ce` **2.201** |
| **`local_ce` cross-check** | 2.201 lands inside `alu-optimizer`'s measured random-init band **2.08–2.24**, at a different `S` and a different modulus mix |
| **measured collapse reference** | constructed-solution output diversity **0.992–0.997** on these cohorts (not 1.0, and not `matrix-scan`'s 0.221 — that is the right quantity for a fixed-modulus cohort, not this one) |
| **measured `dacc` trivial floors** | constant-**zero** predictor: **0.379 / 0.303 / 0.235** at 16/18/20 bits; per-slot majority 0.390 / 0.323 / 0.247 |
| **`--grad-equiv`** | per-row loss weight vs per-row gradient scale: relative gradient difference **6.9e-08**, cosine **1.00000000** |

The trivial-floor row is the important one. A 16-bit answer leaves the top two of
seven slots identically zero, so *predicting zero everywhere* already scores
0.379 on 16-bit examples and 0.235 on 20-bit ones. **A model that learns nothing
but "leading digits are zero" produces exactly the profile a successful
size-curriculum would produce.** Only *within-bucket, across-configuration*
comparisons of `dacc` mean anything; the cross-bucket ordering is an artifact.

---

## 4. A mechanical fact about `training_loss` on this dataset

`training_loss(logits, labels, aux)` receives `token_logits[valid]` and
`token_targets[valid]` — already flattened. On `squaring_mod`,
`tokenize_squaring_mod_with_result` emits `number_tokens(result)` with **no zero
padding**, so the number of supervised positions is the **decimal digit count of
the answer** and varies row by row (`data/squaring_mod.py:434`,
`collate_squaring_mod:90-106`). Nothing in `(logits, labels, aux)` recovers the
row boundaries, and the model cannot predict them — that count is a property of
the answer.

**So a per-example weight cannot be applied inside `training_loss` here.** The
equivalent operation is a per-row **gradient scale** in the forward, applied
*after* the head:

```python
logits = w.view(-1, 1, 1) * logits + (1 - w.view(-1, 1, 1)) * logits.detach()
```

forward value exactly `logits`, gradient to every upstream parameter scaled by
`w`. It is ordinary arithmetic — no custom autograd `Function`, no
participant-controlled backward — and `--grad-equiv` verifies it agrees with a
weighted loss to 6.9e-08. A useful side effect: the logged training loss stays
**unweighted**, so the curriculum cannot flatter the number the runner prints.

Applying it before the head is a bug: the head (and, with a tied embedding, the
embedding) would then receive *unweighted* gradients.

---

## 5. Calibration — the fitting curve, and what the plateau is

*(filled in from `lab/logs/CAL-lr*.log`)*

---

## 6. The sweep

*(filled in)*

---

## 7. The failure mode, checked explicitly

*(filled in)*

---

## 8. Why it cannot work here — the two measurements that close it

*(filled in)*

---

## 9. State of play

*(filled in)*
