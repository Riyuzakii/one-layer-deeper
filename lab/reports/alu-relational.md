# alu-relational — legal training signals with intra-squaring content

**Branch** `explore/alu-relational`, from `explore/alu-credit`.
**Mandate** find a *legal* training signal with intra-squaring content for
`DigitALU`'s discrete tables. Three families: (1) relational constraints on the
learned map at self-generated operands, (2) dual-path agreement, (3) re-screen
the algebraic regularisers under Stage-1 conditions.
**Tooling** `lab/probe_rel.py` (new), `lab/rel_sweep.sh`, `lab/jobs_*.txt`,
runs in `lab/rel_runs.jsonl`, per-run logs in `lab/logs/`.

Every table below is labelled **LEGAL** or **DIAGNOSTIC**, as `alu-credit` §0C
does. Nothing under `data/generated/` was read; all operands are generated from
`math.gcd` over `range(1, N)`.

---

## 0. Verdict

*(filled in at the end — see §9)*

---

## 1. Conditions, and the correctness gate

All runs: **m1 scale** (N = 10403, S = 5, 10,200 units, 8,000 training operands,
1,024 held-out), `alu-depth`'s **`tree:quotient`** graph (39 sequential soft
steps), **untied** (the coordinator's mid-branch correction: ties help *repair*
from a near-solution and hurt *learning* from random init), AdamW lr 3e-2,
betas (0.9, 0.95), batch 512, grad-clip 1.0.

`probe_rel.py` adds to `alu-depth`'s `DigitALU`, using **the same tables and no
new arithmetic**:

* a general two-operand modular multiply `mulmod(s, t)` — the diagonal `s == t`
  is exactly the task, so every table it uses is anchored by the task loss;
* a modular add `addmod(s, t)` — `add_scan` then one learned `quot_reduce`;
* three alternate graph shapes for the same squaring: `fold` (leaf sum by a
  chained left fold instead of a balanced tree), `redall` (reduce after every
  product digit instead of only the last S+1), `horner` (the 257-step Horner
  graph).

**Correctness gate** (`--check-law`, DIAGNOSTIC — uses `--construct`):

```
CONSTRUCTED tree   soft=1.000 hard=1.000 held_hard=1.000
CONSTRUCTED fold   soft=1.000 hard=1.000
CONSTRUCTED redall soft=1.000 hard=1.000
CONSTRUCTED horner soft=1.000 hard=1.000
LAW CHECK (constructed, true constants): lhs==rhs exactly on 1.000 of 512 operands; sym_ce=0.0000
DUAL CHECK tree vs fold  : agree=1.000 sym_ce=0.0000
DUAL CHECK tree vs redall: agree=1.000 sym_ce=0.0000
DUAL CHECK tree vs horner: agree=1.000 sym_ce=0.0000
```

So the target is in the class for every path, and the relational law is exactly
satisfied at the solution. Any failure below is a failure of the *signal*, not
of the implementation.

## 2. The control that was missing: the LEGAL baseline at Stage-1 conditions

`alu-credit` reported the *teacher-forced* number at these conditions (0.951)
and the target-propagation number (0.000), but never the plain legal baseline.
It is a hard zero.

**LEGAL**

| run | `train_exact` | **`train_exact_hard`** | **`held_exact_hard`** | `local_ce` | `add_shift` | `sub_shift` |
|---|---|---|---|---|---|---|
| baseline seed 0 | 0.000 | **0.000** | 0.000 | 3.495 | 0.270 | 0.300 |
| baseline seed 1 | 0.001 | **0.000** | 0.000 | 3.674 | 0.315 | 0.362 |
| baseline seed 2 | 0.000 | **0.000** | 0.000 | 4.177 | 0.280 | 0.237 |
| tied seed 0 / 1 / 2 | 0.000 / 0.000 / 0.001 | 0.000 | 0.000 | 3.26 / 3.12 / 4.03 | ~0.26 | ~0.26 |

Random-init reference for the structure scores is 0.23–0.28, so the tables are
at chance. `local_ce` 3.5–4.2 against a cliff at **0.005–0.0073**: the baseline
is ~500× the per-op error rate that separates 0.951 from 0.000. Ties change
nothing at this end either.

Two conventions used throughout:

* **`out_div`** — the fraction of *distinct* predicted answers over 512
  operands with states snapped. The truth reads 1.000; a map collapsed to a
  constant reads ~0.002. This is a sharper collapse detector than `mul_gauge`
  because it looks at the composed map, not one table. The plain baseline
  oscillates in 0.1–0.9 (measured control, `t_basediv`), so a *low* `out_div`
  is only meaningful against that band.
* **`local_ce`** is now logged during training, not only at the end.
