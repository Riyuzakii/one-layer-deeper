#!/usr/bin/env python
"""Generate a representation-axis submission from lab/repr_template.py.

One variable at a time: every flag below toggles exactly one input/output
representation choice.  The model, optimiser, width, depth and step budget are
identical across configs so the comparison is clean.

  python lab/make_repr_submission.py --tag base
  python lab/make_repr_submission.py --tag field --field
  python lab/make_repr_submission.py --tag slots --layout slots_sep --no-abs
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "lab" / "repr_template.py"
BEGIN = "# --- CONFIG BEGIN ---"
END = "# --- CONFIG END ---"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", default="submissions/exp_repr")
    ap.add_argument("--dmodel", type=int, default=128)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--loops", type=int, default=8)
    ap.add_argument("--ff-mult", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--layout", choices=["flat", "slots_sep", "slots_sum"], default="flat")
    ap.add_argument("--abs", dest="use_abs", action="store_true", default=True)
    ap.add_argument("--no-abs", dest="use_abs", action="store_false")
    ap.add_argument("--field", action="store_true")
    ap.add_argument("--place", action="store_true")
    ap.add_argument("--rpos", action="store_true")
    ap.add_argument("--ans-slots", action="store_true")
    ap.add_argument("--head-per-place", action="store_true")
    ap.add_argument("--scratch", type=int, default=0,
                    help="scratch slots as a multiple of the place count")
    args = ap.parse_args()

    config = "\n".join(
        [
            BEGIN,
            f"D_MODEL = {args.dmodel}",
            f"NUM_HEADS = {args.num_heads}",
            f"NUM_LOOPS = {args.loops}",
            f"FF_MULT = {args.ff_mult}",
            f"_LR = {args.lr!r}",
            f"_WD = {args.wd!r}",
            f"_MAX_STEPS = {args.max_steps!r}",
            f"_BATCH_SIZE = {args.batch_size!r}",
            f"LAYOUT = {args.layout!r}",
            f"USE_ABS = {args.use_abs!r}",
            f"USE_FIELD = {args.field!r}",
            f"USE_PLACE = {args.place!r}",
            f"USE_RPOS = {args.rpos!r}",
            f"ANS_SLOTS = {args.ans_slots!r}",
            f"HEAD_PER_PLACE = {args.head_per_place!r}",
            f"SCRATCH_MULT = {args.scratch!r}",
            END,
        ]
    )
    source = TEMPLATE.read_text()
    out_source = re.sub(
        re.escape(BEGIN) + r".*?" + re.escape(END), config, source, flags=re.S
    )
    if out_source == source:
        raise SystemExit("config block not substituted")
    out = REPO / args.outdir / args.tag / "submission.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(out_source)
    print(out.relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
