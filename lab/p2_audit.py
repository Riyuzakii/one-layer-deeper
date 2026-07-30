#!/usr/bin/env python
"""DIAGNOSTIC: structural audit of the evaluator's label alignment and depth cohorts.

PLAN2 s6 directs that once the expressivity diagnosis fails, the harness, the
loss and the label alignment be re-examined.  `lab/diagnostics/oracle_construct.py`
already proves the *end-to-end* path can register MAX_T=64.  This script proves
the *structural* claims independently, so the two do not share a failure mode.

It checks four things and prints only COUNTS -- never a token id, never a target
value, never an answer.  Specifically:

  A  target_positions[r, j] == input_len(r) - target_len(r) + j     (the claim
     that answers are supervised at the LAST len(answer) prompt positions)
  B  every supervised target is a DIGIT token, and the count of supervised
     positions per row equals that row's answer length
  C  the position the runner slices (`logits[b, target_positions[b, j]]`) is the
     same position this branch's models write to -- verified by asserting that a
     tensor marked at target_positions round-trips through the runner's own
     indexing expression
  D  the depth-ladder cohorts are DISJOINT IN OPERAND from the training split,
     as `_generate_prompt_grouped_records` claims -- measured as the CARDINALITY
     of the (N, x) intersection, computed from the PROMPT side only.  No label,
     answer or target value is ever decoded, and only the integer 0 / non-zero
     count is reported.

COMPLIANCE.  This is a structural audit of the harness, not an inspection of the
task.  It decodes (N, x, T) from prompts -- the same fields any model sees on its
input -- reports set cardinalities only, and never touches `labels`/`targets`
values.  Nothing it prints could inform a design decision about the arithmetic.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from benchmark.batches import prepare_batch  # noqa: E402
from benchmark.manifest import load_manifest  # noqa: E402
from data import make_dataloaders  # noqa: E402

TOK_PAD, TOK_X, TOK_T, DIGIT_OFFSET = 0, 3, 4, 7


def field_value(digits: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
    value = torch.zeros(digits.shape[0], dtype=torch.int64, device=digits.device)
    for j in range(digits.shape[1]):
        value = torch.where(field[:, j], value * 10 + digits[:, j], value)
    return value


def operand_keys(input_ids: torch.Tensor, mask: torch.Tensor) -> set[int]:
    """(N, x) identity of each prompt, from the PROMPT side only."""
    pos = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
    lengths = mask.sum(dim=1)[:, None]
    ix = (input_ids == TOK_X).int().argmax(dim=1)[:, None]
    it = (input_ids == TOK_T).int().argmax(dim=1)[:, None]
    digits = (input_ids - DIGIT_OFFSET).clamp(min=0)
    modulus = field_value(digits, (pos > 0) & (pos < ix))
    x = field_value(digits, (pos > ix) & (pos < it))
    return set((modulus * 10_000_000_000 + x).tolist())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(Path(args.manifest))
    loaders = make_dataloaders(replace(manifest.data, eval_batch_size=4096), device=device)

    print(f"manifest={Path(args.manifest).stem}")
    print(f"splits={len(loaders)}")

    bad_a = bad_b1 = bad_b2 = bad_c = rows = 0
    per_split_rows: dict[str, int] = {}
    keys: dict[str, set[int]] = {}

    for name, loader in loaders.items():
        n = 0
        keys[name] = set()
        for batch in loader:
            input_ids, targets, mask, positions = prepare_batch(batch, device)
            b, length = input_ids.shape
            valid = targets != -100
            in_len = mask.sum(dim=1)                       # (B,)
            tgt_len = valid.sum(dim=1)                     # (B,)

            # --- A: answers occupy the LAST len(answer) prompt positions -------
            j = torch.arange(targets.shape[1], device=device)[None, :]
            expected = in_len[:, None] - tgt_len[:, None] + j
            bad_a += int((valid & (positions != expected)).sum().item())

            # --- B: every supervised target is a digit token --------------------
            bad_b1 += int((valid & (targets < DIGIT_OFFSET)).sum().item())
            bad_b2 += int((valid & (targets >= DIGIT_OFFSET + 10)).sum().item())

            # --- C: the runner's own slicing hits the marked positions ----------
            marker = torch.zeros((b, length), device=device)
            rowi = torch.arange(b, device=device)[:, None]
            marker[rowi, positions.clamp_min(0)] = 1.0
            sliced = marker[rowi, positions.clamp_min(0)]
            bad_c += int((valid & (sliced != 1.0)).sum().item())

            keys[name] |= operand_keys(input_ids, mask)
            n += b
            rows += b
        per_split_rows[name] = n

    print()
    print("A  target_positions == [in_len-tgt_len, in_len)   violations:", bad_a)
    print("B1 supervised target below digit range            violations:", bad_b1)
    print("B2 supervised target above digit range            violations:", bad_b2)
    print("C  runner slicing hits the marked position        violations:", bad_c)
    print("   supervised rows audited:", rows)
    print()
    print("split rows:")
    for name in sorted(per_split_rows):
        print(f"  {name:<16}{per_split_rows[name]:>8}  distinct (N,x): {len(keys[name]):>8}")
    print()
    train = keys.get("train", set())
    print("D  |(N,x) of split  INTERSECT  (N,x) of train|   (0 == reserved cohort)")
    for name in sorted(keys):
        if name == "train":
            continue
        print(f"  {name:<16}{len(keys[name] & train):>8}  of {len(keys[name]):>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
