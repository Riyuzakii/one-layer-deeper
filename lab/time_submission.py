#!/usr/bin/env python
"""Time an actual submission file's train step, in the evaluator's shape and dtype.

Reports local ms/step and converts to an H100 estimate through the calibration model
(BRIEF2 §4: exp_axis D=128 / 8 loops = 38.6 ms/step at batch 512 on the hosted H100).

COMPLIANCE: synthetic integer inputs; nothing under data/generated/ is opened.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from benchmark.api import ModelSpec, OptimizerSpec  # noqa: E402

sys.path.insert(0, str(REPO / "lab"))
from rnn_bench import (  # noqa: E402
    H100_REF_MS,
    H100_STARTUP_S,
    HARD_BUDGET_S,
    RefRecurrentTransformer,
    time_model,
)


class _Wrap(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, ids, mask):
        return self.inner(ids, attention_mask=mask)[0]


def load(path: Path):
    spec = importlib.util.spec_from_file_location("_timed_submission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", nargs="+", required=True)
    ap.add_argument("--seq-len", type=int, default=21)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--d-h", type=int, nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    L, B = args.seq_len, args.batch
    ids = torch.randint(1, 17, (B, L), device="cuda")
    mask = torch.ones(B, L, dtype=torch.bool, device="cuda")
    tgt = torch.randint(0, 17, (B, L), device="cuda")

    ref_ms, _ = time_model(RefRecurrentTransformer(L), ids, mask, tgt, steps=args.steps)
    print(f"calibration REF = {ref_ms:.2f} ms local  ==  {H100_REF_MS} ms on H100")

    rows = []
    for path in args.submission:
        module = load(Path(path))
        widths = args.d_h if args.d_h else [module.D_H]
        for d_h in widths:
            module.D_H = d_h
            module.D_EMB = d_h
            spec = ModelSpec(17, L, 500_000_000)
            model = _Wrap(module.build_model(spec))
            ms, par = time_model(model, ids, mask, tgt, steps=args.steps)
            h100 = ms / ref_ms * H100_REF_MS
            steps = (HARD_BUDGET_S - H100_STARTUP_S) / (h100 / 1e3)
            row = dict(
                submission=Path(path).parent.name,
                d_h=d_h,
                loops=module.LOOPS,
                align=module.ALIGN,
                params=par,
                local_ms=round(ms, 3),
                ratio=round(ms / ref_ms, 4),
                h100_ms_est=round(h100, 2),
                hard_steps_est=int(steps),
            )
            rows.append(row)
            print(
                f"  {row['submission']:<28} d_h={d_h:<4} loops={module.LOOPS} "
                f"align={int(module.ALIGN)} params={par:<9} "
                f"{ms:7.2f} ms  ratio {row['ratio']:6.3f}  "
                f"H100~{h100:6.2f} ms  steps~{int(steps):>7}"
            )
    if args.out:
        Path(args.out).write_text(
            json.dumps(dict(ref_local_ms=ref_ms, seq_len=L, batch=B, rows=rows), indent=2)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
