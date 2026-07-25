#!/usr/bin/env python
"""Structural checks for the algebraic-closure submission.

Everything here builds its prompts with the PUBLIC tokenizer
(`data.squaring_mod.tokenize_squaring_mod_with_result`).  No file under
`data/generated/` is opened.

Run:  .venv/bin/python lab/test_closure.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from data.squaring_mod import (  # noqa: E402
    TOKEN_IDS,
    collate_squaring_mod,
    tokenize_squaring_mod_with_result,
)
from benchmark.api import ModelSpec  # noqa: E402


def load(path: Path):
    spec = importlib.util.spec_from_file_location("closure_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_batch(rows):
    items = []
    for modulus, x, t, y in rows:
        ids, labels = tokenize_squaring_mod_with_result(
            modulus, x, t, y, separate_input_output=True
        )
        items.append({"input_ids": ids, "labels": labels})
    return collate_squaring_mod(items)


def main() -> int:
    path = REPO / "submissions" / "exp_closure" / "_selftest" / "submission.py"
    if not path.is_file():
        raise SystemExit(f"generate it first: {path}")
    mod = load(path)

    # a mixed batch: 1-, 2- and 3-digit x, 1- and 2-digit T, answers of
    # differing digit counts.  Values are arbitrary integers chosen here, not
    # read from any dataset.
    rows = [
        (323, 5, 1, 25),
        (323, 300, 2, 118),
        (323, 47, 64, 9),
        (323, 7, 16, 123),
        (323, 12, 3, 300),
    ]
    batch = make_batch(rows)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    target_positions = batch["target_positions"]
    length = input_ids.shape[1]

    spec = ModelSpec(vocab_size=17, max_seq_len=11, maximum_model_state_elements=5 * 10**8)
    model = mod.build_model(spec)
    print(f"places={model.places} (expected 4 for max_seq_len=11)")

    # --- 1. slot layout is the LSD-first decimal expansion -----------------
    n_slots, x_slots, t_slots = mod._slots(input_ids, attention_mask, model.places)
    for row, (modulus, x, t, _y) in enumerate(rows):
        want_n = [(modulus // 10**k) % 10 for k in range(model.places)]
        want_x = [(x // 10**k) % 10 for k in range(model.places)]
        want_t = [(t // 10**k) % 10 for k in range(mod.T_SLOTS)]
        got_n = n_slots[row].tolist()
        got_x = x_slots[row].tolist()
        got_t = t_slots[row].tolist()
        assert got_n == want_n, (row, got_n, want_n)
        assert got_x == want_x, (row, got_x, want_x)
        assert got_t == want_t, (row, got_t, want_t)
    print("OK  slot layout == LSD-first decimal expansion for N, x and T")

    # --- 2. every scored position is written by the scatter ----------------
    model.eval()
    logits, aux = model(input_ids, attention_mask)
    assert aux is None, "eval mode must not build the auxiliary term"
    assert logits.shape == (len(rows), length, 17), logits.shape
    marker = torch.zeros(len(rows), length, dtype=torch.bool)
    places = model.places
    lens = attention_mask.sum(1)
    for row in range(len(rows)):
        for j in range(places):
            pos = int(lens[row]) - 1 - j
            if pos >= 0:
                marker[row, pos] = True
    valid = labels != -100
    scored = target_positions[valid]
    rows_idx = torch.arange(len(rows)).unsqueeze(1).expand_as(labels)[valid]
    assert bool(marker[rows_idx, scored].all()), "a scored position is unwritten"
    print("OK  every evaluator-scored position is produced by the place scatter")

    # --- 3. answer place j is read at distance j from the end --------------
    for row, (_m, _x, _t, y) in enumerate(rows):
        n_digits = len(str(y))
        for j in range(n_digits):
            pos = int(lens[row]) - 1 - j
            col = (labels[row] != -100).sum().item() - 1 - j
            assert int(target_positions[row, col]) == pos
            assert int(labels[row, col]) - TOKEN_IDS["ANS"] - 2 == (y // 10**j) % 10
    print("OK  answer place j is scored at sequence position input_len-1-j")

    # --- 4. training mode produces a differentiable auxiliary scalar -------
    model.train()
    logits, aux = model(input_ids, attention_mask)
    assert aux is not None and aux.ndim == 0, aux
    loss = mod.training_loss(
        logits[torch.arange(len(rows)).unsqueeze(1).expand_as(labels)[valid], scored],
        labels[valid],
        aux,
    )
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"
    print(f"OK  loss={float(loss):.4f} backprops to all {len(list(model.parameters()))} params")

    # --- 5. the auxiliary term really depends on the composition route -----
    model.zero_grad()
    torch.manual_seed(0)
    _, aux_a = model(input_ids, attention_mask)
    torch.manual_seed(0)
    _, aux_b = model(input_ids, attention_mask)
    assert torch.allclose(aux_a, aux_b), "auxiliary term is not deterministic given the RNG"
    print(f"OK  auxiliary term reproducible: {float(aux_a):.6f}")

    # --- 6. source lint ----------------------------------------------------
    sys.path.insert(0, str(REPO))
    from submission_validation import validate_submission_source

    validate_submission_source(
        str(path), path.read_text(), 256 * 1024, required_filename="submission.py"
    )
    print("OK  submission_validation.validate_submission_source passes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
