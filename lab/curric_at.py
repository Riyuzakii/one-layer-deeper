#!/usr/bin/env python
"""Pull every cell's row at a MATCHED step out of lab/logs/*.log.

Cells were run at different total step counts (the baselines to 3,000, the
curriculum cells to 1,500), so the comparison has to be made at a step both
sides logged, not at each run's own endpoint.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAT = re.compile(
    r"step=\s*(\d+) loss=([\d.]+) beta=\(([-\d.nae]+),([-\d.nae]+)\) "
    r"local_ce=([\d.nae-]+) "
    r"train_soft=([\d.]+) dacc=([\d.]+) \((.*?)\) ce=([\d.]+) \| "
    r"train_hard=([\d.]+) dacc=([\d.]+) \((.*?)\) ce=([\d.]+) \| "
    r"heldN_hard=([\d.]+) dacc=([\d.]+) \((.*?)\) ce=([\d.]+)"
)


def buckets(s):
    return {k: v for k, v in re.findall(r"([bd]\d+)=([\d.]+)", s)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=1500)
    ap.add_argument("--glob", default="*.log")
    args = ap.parse_args()
    rows = []
    for f in sorted((HERE / "logs").glob(args.glob)):
        for ln in f.read_text(errors="ignore").splitlines():
            m = PAT.search(ln)
            if m and int(m.group(1)) == args.step:
                ts, th, hh = (buckets(m.group(8)), buckets(m.group(12)),
                              buckets(m.group(16)))
                rows.append((f.stem, m.group(2), m.group(5), m.group(6),
                             m.group(7), ts, m.group(10), m.group(14),
                             m.group(15), hh))
    print(f"| cell | loss | local_ce | train_exact | train_exact_hard | "
          f"heldN_exact_hard | train dacc soft | train dacc soft 16/18/20 | "
          f"heldN dacc HARD 16/18/20 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (tag, loss, lce, te, td, ts, thd, hhe, hhd, hh) in rows:
        def trip(d, p):
            return "/".join(d.get(f"{p}{b}", "-") for b in (16, 18, 20, 10, 12)
                            if f"{p}{b}" in d) or "-"
        print(f"| {tag} | {loss} | {lce} | {te} | {thd} | {hhe} | {td} | "
              f"{trip(ts,'d')} | {trip(hh,'d')} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
