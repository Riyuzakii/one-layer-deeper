#!/usr/bin/env bash
# Dense-transformer cells: the SAME recipe explore/grok-optimization used for
# its optimizer comparison (e1, 10,000 steps, bs 128, wd 0.1, seed 74), with
# SOAP and AdEMAMix added.  Its table: AdamW rung-1 0.000, Muon 0.026,
# Schedule-Free 0.000.
set -u
cd /home/scratch.arohan_hw/git/one-layer-deeper/.worktrees/alu-optimizer
export TAG=B-opt2
lab/optsub.sh "$@"
