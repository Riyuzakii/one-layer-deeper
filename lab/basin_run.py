#!/usr/bin/env python
"""Drive the whole repair-basin grid inside ONE process.

Per-process CUDA context creation costs ~4 minutes on this shared GPU -- more
than a 2,000-step cell itself (77 s).  Running the grid in one process turns a
3.3-hour sweep into a ~50-minute one and changes nothing about the measurement.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_o1  # noqa: E402

OUT = "lab/runs/basin.jsonl"
HF = ["--bits-list", "16,18,20", "--n-mod-train", "8", "--n-mod-held", "4",
      "--train-x", "250", "--held-x", "256"]


def cells(which: str):
    if which == "basin":            # uniform corruption -- the comparable protocol
        for k in (5, 20, 50, 100, 400):
            for s in (0, 1, 2):
                yield f"o1-k{k}-s{s}", ["--modulus", "323", "--recip", "oracle",
                                        "--corrupt", str(k), "--seed", str(s)]
    elif which == "strat":          # k cells of EVERY table -> repair by depth
        for k in (4, 16):
            for s in (0, 1, 2):
                yield f"o1-strat{k}-s{s}", ["--modulus", "323", "--recip", "oracle",
                                            "--corrupt", str(k), "--corrupt-mode",
                                            "per_table", "--seed", str(s)]
    elif which == "lr0":            # BRIEF2 6.1 -- the control every claim needs
        for k in (5, 20, 400):
            yield f"o1-k{k}-lr0", ["--modulus", "323", "--recip", "oracle",
                                   "--corrupt", str(k), "--seed", "0", "--lr", "0"]
        for k in (4, 16):
            yield f"o1-strat{k}-lr0", ["--modulus", "323", "--recip", "oracle",
                                       "--corrupt", str(k), "--corrupt-mode",
                                       "per_table", "--seed", "0", "--lr", "0"]
    elif which == "sharp":          # one and two cells from a solution that scores 1.000
        for k in (1, 2):
            for s in (0, 1, 2):
                yield f"o1-k{k}-s{s}", ["--modulus", "323", "--recip", "oracle",
                                        "--corrupt", str(k), "--seed", str(s)]
        # step-count calibration for the basin protocol itself: 10x the budget
        for k in (5, 20):
            yield f"o1-k{k}-20k", ["--modulus", "323", "--recip", "oracle",
                                   "--corrupt", str(k), "--seed", "0",
                                   "--steps", "20000", "--log-every", "5000"]
    elif which == "k0":             # start AT the exact solution and train
        for s_ in (0, 1, 2):
            yield f"o1-k0-s{s_}", ["--modulus", "323", "--recip", "oracle",
                                   "--from-construct", "--seed", str(s_),
                                   "--steps", "2000", "--log-every", "250"]
        yield "o1-k0-lr0", ["--modulus", "323", "--recip", "oracle", "--from-construct",
                            "--seed", "0", "--lr", "0", "--steps", "500",
                            "--log-every", "250"]
        yield "hf1-k0-s0", HF + ["--recip", "oracle", "--from-construct", "--seed", "0",
                                 "--steps", "2000", "--log-every", "500"]
        yield "hf1-k0-lr0", HF + ["--recip", "oracle", "--from-construct", "--seed", "0",
                                  "--lr", "0", "--steps", "500", "--log-every", "500"]
    elif which == "div":            # nothing oracular anywhere
        for k in (20, 400):
            for s in (0, 1, 2):
                yield f"div-k{k}-s{s}", ["--modulus", "323", "--recip", "div",
                                         "--corrupt", str(k), "--seed", str(s)]
    elif which == "hf1basin":       # the same, at hf1's scale and split shape
        for k in (20, 400):
            for s in (0, 1, 2):
                yield f"hf1-k{k}-s{s}", HF + ["--recip", "oracle", "--corrupt", str(k),
                                              "--seed", str(s)]
        for s in (0, 1, 2):
            yield f"hf1-strat4-s{s}", HF + ["--recip", "oracle", "--corrupt", "4",
                                            "--corrupt-mode", "per_table",
                                            "--seed", str(s)]
        for k in (20, 400):
            yield f"hf1-k{k}-lr0", HF + ["--recip", "oracle", "--corrupt", str(k),
                                         "--seed", "0", "--lr", "0"]
    elif which == "hf1fit":         # LEGAL from random init, with the fitting curve
        yield "hf1-legal-orac-s0", HF + ["--recip", "oracle", "--seed", "0",
                                         "--steps", "8000", "--log-every", "1000"]
        yield "hf1-legal-orac-lr0", HF + ["--recip", "oracle", "--seed", "0", "--lr", "0",
                                          "--steps", "1000", "--log-every", "1000"]
        yield "hf1-legal-head-s0", HF + ["--recip", "head", "--seed", "0",
                                         "--steps", "8000", "--log-every", "1000"]
        yield "hf1-legal-div-s0", HF + ["--recip", "div", "--seed", "0",
                                        "--steps", "4000", "--log-every", "500"]
        yield "hf1-legal-div-lr0", HF + ["--recip", "div", "--seed", "0", "--lr", "0",
                                         "--steps", "500", "--log-every", "500"]
        yield "hf1-ceiling", HF + ["--recip", "div", "--construct"]
        yield "hf1-ceiling-orac", HF + ["--recip", "oracle", "--construct"]
    elif which == "sfit":           # N=323 calibration: does this class fit at all?
        yield "s3-legal-orac-s0", ["--modulus", "323", "--recip", "oracle", "--seed", "0",
                                   "--steps", "8000", "--log-every", "1000"]
        yield "s3-legal-div-s0", ["--modulus", "323", "--recip", "div", "--seed", "0",
                                  "--steps", "8000", "--log-every", "1000"]
        yield "s3-legal-orac-lr0", ["--modulus", "323", "--recip", "oracle", "--seed", "0",
                                    "--lr", "0", "--steps", "1000", "--log-every", "1000"]
        yield "s3-ceiling", ["--modulus", "323", "--recip", "div", "--construct"]


def main() -> int:
    for which in sys.argv[1:]:
        for tag, argv in cells(which):
            t = time.time()
            print(f"=== {tag} ===", flush=True)
            base = ["--jsonl", OUT, "--tag", tag]
            if not any(a == "--steps" for a in argv) and "--construct" not in argv:
                base += ["--steps", "2000", "--log-every", "2000"]
            probe_o1.main(base + argv)
            print(f"--- {tag} done in {time.time() - t:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
