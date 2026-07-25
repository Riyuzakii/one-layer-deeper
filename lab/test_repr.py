#!/usr/bin/env python
"""Self-checks for the representation template.

Prompts are constructed here with the PUBLIC tokenizer from data/squaring_mod.py
(compliant: generator source, not generated data).  Nothing under
data/generated/ is opened.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data.squaring_mod import (  # noqa: E402
    collate_squaring_mod,
    tokenize_squaring_mod_with_result,
)


def make_batch(rows):
    items = []
    for modulus, x, t, result in rows:
        ids, labels = tokenize_squaring_mod_with_result(
            modulus, x, t, result, separate_input_output=True
        )
        items.append({"input_ids": ids, "labels": labels})
    return collate_squaring_mod(items)


def check_target_alignment():
    """The evaluator reads answer place j off position (input_len - 1 - j)."""
    rows = [(323, 7, 1, 49), (323, 300, 2, 5), (323, 45, 3, 123)]
    batch = make_batch(rows)
    tp, labels = batch["target_positions"], batch["labels"]
    lens = batch["attention_mask"].sum(1)
    for r, (_, _, _, result) in enumerate(rows):
        digits = [int(c) for c in str(result)]
        n = len(digits)
        for j in range(n):  # j = place value (0 = units)
            pos = int(tp[r, n - 1 - j])
            lab = int(labels[r, n - 1 - j]) - 7
            assert pos == int(lens[r]) - 1 - j, (r, j, pos, lens[r])
            assert lab == digits[n - 1 - j], (r, j, lab)
    print("ok: answer place j is read at distance-from-end j")


def check_geometry(mod):
    rows = [(323, 7, 1, 49), (323, 300, 2, 5), (323, 45, 16, 123)]
    batch = make_batch(rows)
    ids, mask = batch["input_ids"], batch["attention_mask"]
    _, _, _, field, fstart, fend, flen = mod._field_geometry(ids, mask)
    expect_len = [
        [3, 1, 1],  # N=323, x=7,  T=1
        [3, 3, 1],  # N=323, x=300,T=2
        [3, 2, 2],  # N=323, x=45, T=16
    ]
    for r, exp in enumerate(expect_len):
        got = [int(flen[r, f]) for f in (1, 2, 3)]
        assert got == exp, (r, got, exp)
    print("ok: field segmentation", flen[:, 1:].tolist())


def check_slot_gather(mod):
    rows = [(323, 7, 1, 49), (323, 300, 2, 5), (323, 45, 16, 123)]
    batch = make_batch(rows)
    ids, mask = batch["input_ids"], batch["attention_mask"]
    _, _, _, _, _, fend, flen = mod._field_geometry(ids, mask)
    places = torch.arange(3)
    d_n = mod.Model._gather_places(ids, fend, flen, 1, places)
    d_x = mod.Model._gather_places(ids, fend, flen, 2, places)
    assert d_n.tolist() == [[3, 2, 3]] * 3, d_n.tolist()  # 323 LSD-first
    assert d_x.tolist() == [[7, 10, 10], [0, 0, 3], [5, 4, 10]], d_x.tolist()
    print("ok: place-aligned gather (LSD first, 10 = absent)", d_x.tolist())


def check_scatter_roundtrip(mod):
    """Slot j logits must land exactly on the evaluator's target position."""
    rows = [(323, 7, 1, 49), (323, 300, 2, 5), (323, 45, 16, 123)]
    batch = make_batch(rows)
    ids, mask = batch["input_ids"], batch["attention_mask"]
    spec = mod.ModelSpec(vocab_size=17, max_seq_len=int(ids.shape[1]) + 4,
                         maximum_model_state_elements=500_000_000)
    model = mod.Model(spec)
    logits, _ = model(ids, mask)
    tp, labels = batch["target_positions"], batch["labels"]
    bi = torch.arange(ids.shape[0])[:, None]
    picked = logits[bi, tp.clamp_min(0)]
    valid = labels != -100
    assert torch.isfinite(picked[valid]).all()
    # every scored position must have been written (non-zero) by the scatter
    assert (picked[valid].abs().sum(-1) > 0).all(), "scored position not written"
    loss = torch.nn.functional.cross_entropy(picked[valid], labels[valid])
    loss.backward()
    grads = [p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()]
    assert all(grads), "missing/non-finite gradient"
    print("ok: scatter covers every scored position; loss backprops")


def main() -> int:
    check_target_alignment()
    import importlib.util

    candidates = [
        ("flat", REPO / "submissions" / "exp_repr" / "r4_all" / "submission.py"),
        ("sep", REPO / "submissions" / "exact-arithmetic" / "submission.py"),
        ("sum", REPO / "submissions" / "exp_repr" / "r7_slotsum" / "submission.py"),
    ]
    for tag, path in candidates:
        if not path.exists():
            print(f"skip {tag}: {path} missing (generate it first)")
            continue
        layout = "flat" if tag == "flat" else f"slots_{tag}"
        spec = importlib.util.spec_from_file_location(f"_repr_{tag}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print(f"-- {layout}")
        check_geometry(mod)
        if layout != "flat":
            check_slot_gather(mod)
            check_scatter_roundtrip(mod)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
