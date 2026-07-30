#!/usr/bin/env python
"""Emit `submissions/p2-sequential-rnn/<variant>/submission.py` by rewriting the
module-level knobs of the canonical file.  Same pattern as `lab/make_submission.py`.

  make_rnn.py --d-h 8 --name d8
  make_rnn.py --d-h 64 --loops 2 --name d64_x2
  make_rnn.py --lr 0 --name lr0            # the mandatory --lr 0 control
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "submissions" / "p2-sequential-rnn" / "submission.py"


def rewrite(source: str, **knobs) -> str:
    for key, value in knobs.items():
        if value is None:
            continue
        pattern = rf"^{key} = .*$"
        if not re.search(pattern, source, flags=re.M):
            raise KeyError(f"knob {key} not found in the base submission")
        source = re.sub(pattern, f"{key} = {value!r}", source, count=1, flags=re.M)
    return source


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(BASE))
    ap.add_argument("--name", required=True)
    ap.add_argument("--d-h", type=int, default=None)
    ap.add_argument("--loops", type=int, default=None)
    ap.add_argument("--align", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    base = Path(args.base)
    source = base.read_text()
    source = rewrite(
        source,
        D_H=args.d_h,
        D_EMB=args.d_h,
        LOOPS=args.loops,
        ALIGN=None if args.align is None else bool(args.align),
        LR=args.lr,
        WD=args.wd,
        BATCH_SIZE=args.batch_size,
        MAX_STEPS=args.max_steps,
    )
    out = base.parent / args.name / "submission.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(source)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
