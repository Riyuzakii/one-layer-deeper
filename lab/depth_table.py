#!/usr/bin/env python
"""Collect lab/logs/**/ *.log from probe_alu into the variant table."""
from __future__ import annotations

import argparse
import pathlib
import re
import statistics
import sys

# `mul=` was added mid-session; logs written before it are read as horner.
HEAD = re.compile(r"modulus=(\d+) S=(\d+) K=\d+ W=(\d+) (?:mul=(\w+) )?"
                  r"(?:scan=(\w+) )?mode=(\w+) R=(\d+) Q=(\d+) "
                  r".*params=([\d,]+)")
DEPTH = re.compile(r"DEPTH main=(\d+) \(adds=(\d+) reduce=(\d+)\) "
                   r"N-prefix=(\d+) alphabet=(\{[^}]*\})")
STEP = re.compile(r"step=\s*(\d+) loss=([\d.naif-]+) train_exact=([\d.]+) "
                  r"held_exact=([\d.]+)")
HARD = re.compile(r"train_exact_hard=([\d.]+)")
SHARP = re.compile(r"state_sharpness=([\d.]+)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", default=["lab/logs/depth", "lab/logs/long"])
    ap.add_argument("--md", action="store_true")
    args = ap.parse_args()
    rows = {}
    for d in args.dirs:
        for f in sorted(pathlib.Path(d).glob("*.log")):
            txt = f.read_text()
            h, dp = HEAD.search(txt), DEPTH.search(txt)
            steps = STEP.findall(txt)
            if not (h and dp and steps):
                continue
            last = steps[-1]
            tail = txt.rsplit("step=", 1)[-1]
            hh = HARD.search(tail)
            ss = SHARP.search(tail)
            var = re.sub(r"^\d+_S\d+_\w+?_(serial|binary|quotient)", "",
                         f.stem)
            var = re.sub(r"_s\d+$", "", re.sub(r"_n\d+$", "", var)) or "-"
            key = (int(h.group(1)), int(h.group(2)), h.group(4) or "horner",
                   h.group(6), int(dp.group(1)), h.group(9), int(last[0]),
                   ((h.group(5) or "serial") + "/" + var.lstrip("_")).rstrip("/"))
            rows.setdefault(key, []).append(
                (float(last[2]), float(last[3]),
                 float(hh.group(1)) if hh else None,
                 float(ss.group(1)) if ss else None))
    if not rows:
        print("no logs parsed", file=sys.stderr)
        return 1
    hdr = ("N", "S", "mul", "reduce", "depth", "params", "steps", "var", "seeds",
           "train_exact", "held_exact", "train_hard", "sharp")
    out = []
    for k in sorted(rows, key=lambda z: (z[0], z[1], z[6], z[4])):
        v = rows[k]
        tr = "/".join(f"{a:.3f}" for a, _, _, _ in v)
        he = "/".join(f"{b:.3f}" for _, b, _, _ in v)
        hd = "/".join("-" if c is None else f"{c:.3f}" for _, _, c, _ in v)
        sh = "/".join("-" if d is None else f"{d:.3f}" for _, _, _, d in v)
        mean = statistics.mean(a for a, _, _, _ in v)
        out.append((str(k[0]), str(k[1]), k[2], k[3], str(k[4]), k[5],
                    str(k[6]), k[7] or "-", str(len(v)),
                    f"{tr} (mu={mean:.3f})", he, hd, sh))
    if args.md:
        print("| " + " | ".join(hdr) + " |")
        print("|" + "---|" * len(hdr))
        for r in out:
            print("| " + " | ".join(r) + " |")
    else:
        w = [max(len(hdr[i]), max(len(r[i]) for r in out)) for i in range(len(hdr))]
        print("  ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
        for r in out:
            print("  ".join(c.ljust(w[i]) for i, c in enumerate(r)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
