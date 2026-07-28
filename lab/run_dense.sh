#!/usr/bin/env bash
# Dense-transformer cells.  Same recipe explore/grok-optimization used for its
# optimizer comparison (e1, 10,000 steps, bs 128, wd 0.1, lr 1e-3, seed 74);
# its table reads AdamW rung-1 0.000 / Muon 0.026 / Schedule-Free 0.000.
set -u
export TAG=B-opt2
C="--batch-size 128 --wd 0.1 --wd-emb 0.1"
bash lab/optsub.sh G_adamw_ctl 10000 74 "dense ctl: AdamW lr1e-3"          --opt adamw    --lr 1e-3 $C
bash lab/optsub.sh G_soap_1e3  10000 74 "dense: SOAP lr1e-3"               --opt soap     --lr 1e-3 $C
bash lab/optsub.sh G_soap_3e3  10000 74 "dense: SOAP lr3e-3"               --opt soap     --lr 3e-3 $C
bash lab/optsub.sh G_ade_3e4   10000 74 "dense: AdEMAMix a8 b3=.9999 lr3e-4" --opt ademamix --lr 3e-4 --ade-alpha 8 --ade-beta3 0.9999 --ade-warmup 3000 $C
bash lab/optsub.sh G_ade_1e3   10000 74 "dense: AdEMAMix a8 b3=.9999 lr1e-3" --opt ademamix --lr 1e-3 --ade-alpha 8 --ade-beta3 0.9999 --ade-warmup 3000 $C
