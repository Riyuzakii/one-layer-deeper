#!/usr/bin/env python
"""Make a *lab-only* copy of a submission that saves its trained weights.

The real submission stays untouched; this appends an `atexit` hook that dumps
`state_dict()` to `$GR_DIAG_SAVE` after the runner finishes.  Nothing here reads
any dataset -- the point is to get the *model's own weights* out so they can be
probed on synthetic prompts (weight/activation analysis, which is allowed;
dataset inspection, which is not, never happens).

  python lab/mkdiag.py --submission submissions/group-rotation/gl_wd1/submission.py
  GR_DIAG_SAVE=/tmp/gl_wd1.pt python lab/run_experiment.py --submission lab/diag/gl_wd1/submission.py ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

HOOK = '''

# --- lab-only diagnostic hook (appended by lab/mkdiag.py; never submitted) ---
import atexit as _atexit
import os as _os

_orig_build_model = build_model


def build_model(spec):  # noqa: F811
    model = _orig_build_model(spec)
    path = _os.environ.get("GR_DIAG_SAVE")
    if path:
        _atexit.register(
            lambda: torch.save(
                {k: v.detach().float().cpu() for k, v in model.state_dict().items()},
                path,
            )
        )
    return model


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    batch_size=_BATCH,
    max_steps=_MAX_STEPS,
)
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--outdir", default="lab/diag")
    args = ap.parse_args()
    src = Path(args.submission).resolve()
    name = src.parent.name
    out = REPO / args.outdir / name / "submission.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(src.read_text() + HOOK)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
