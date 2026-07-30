#!/usr/bin/env python
"""Splice the generated tables into lab/reports/p2-pd-ssm-delta.md placeholders."""
from __future__ import annotations
import io, re, subprocess, sys
from contextlib import redirect_stdout
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lab.p2_summarize as S  # noqa: E402

buf = io.StringIO()
with redirect_stdout(buf):
    S.probe_tables()
    S.grid_table()
text = buf.getvalue()

def section(name: str) -> str:
    m = re.search(rf"^### {re.escape(name)}.*?(?=^###|\Z)", text, re.S | re.M)
    return m.group(0).rstrip() if m else f"(no rows for {name})"

REPORT = Path(__file__).resolve().parent / "reports" / "p2-pd-ssm-delta.md"
r = REPORT.read_text()
mapping = {
    "RESULTS_PROBE_A": section("PROBE A"),
    "RESULTS_PROBE_B": section("PROBE B"),
    "RESULTS_PROBE_C": section("PROBE C --") + "\n\n" + section("PROBE F"),
    "RESULTS_PROBE_D": section("PROBE D"),
    "PROBE_C_A5_PERSEED": section("PROBE C'"),
    "RESULTS_GRID": section("REAL TASK"),
}
for k, v in mapping.items():
    r = r.replace(k, v)
REPORT.write_text(r)
print("spliced:", ", ".join(k for k in mapping if k not in r))
remaining = [k for k in ("RESULTS_PROBE_A","RESULTS_PROBE_B","RESULTS_PROBE_C",
                         "RESULTS_PROBE_D","PROBE_C_A5_PERSEED","RESULTS_GRID",
                         "RESULTS_LR0","RESULTS_NARRATIVE","RESULTS_DISCRETE",
                         "LR0_MATSCAN","FALSIFIED") if k in r]
print("still open:", remaining)
