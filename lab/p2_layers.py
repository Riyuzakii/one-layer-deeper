"""Single source of truth for the §3.2 / §3.3 mixers.

The layers live in ``submissions/plan2-pd-ssm-delta/submission.py`` (they have to
be there anyway -- a submission must be one self-contained file).  This module
loads that file so the synthetic probe and the evaluator runs are guaranteed to
exercise *identical* code.  Nothing here is imported by a submission.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SUB = (
    Path(__file__).resolve().parent.parent
    / "submissions"
    / "plan2-pd-ssm-delta"
    / "submission.py"
)

_spec = importlib.util.spec_from_file_location("_p2_submission", _SUB)
sub = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sub)

DeltaProductMixer = sub.DeltaProductMixer
PDSSMMixer = sub.PDSSMMixer
MatrixScanMixer = sub.MatrixScanMixer
RMSNorm = sub.RMSNorm

MIXERS = {
    "delta": DeltaProductMixer,
    "pdssm": PDSSMMixer,
    "matscan": MatrixScanMixer,
}


def set_impl(impl: str) -> None:
    """Switch between the chunkwise/log-depth path and the sequential reference."""
    assert impl in ("fast", "seq")
    sub.IMPL = impl


def get_impl() -> str:
    return sub.IMPL
