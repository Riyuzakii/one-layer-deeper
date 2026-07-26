# alu-credit — making `DigitALU` trainable: what the optimiser can and cannot do

**Branch** `explore/alu-credit` (from `explore/digit-carry`).
**Mandate** hold the architecture fixed, vary everything about how it is
optimised, and get `train_exact` to 1.000.
**Status** *(draft — filled in as runs land)*

---

## 0. Headline

*(to be written)*

## 1. What I inherited and what I re-verified

`DigitALU` (`lab/probe_alu.py`), 6,817 parameters, ~280 sequential soft table
lookups, constructed ceiling 1.000 at N=323/S=3/R=11, plateau at
`train_exact ≈ 0.2` from random init.

`lab/probe_credit.py` subclasses it. The forward pass is byte-for-byte the same
computation — verified by `--construct`, which still reports
`train_exact = held_exact = 1.000` after the refactor into an explicit op list.

## 2. Diagnostics

## 3. What was falsified

## 4. What worked, and whether it is legal

## 5. Compliance

## 6. Recommendation
