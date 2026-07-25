#!/usr/bin/env bash
# Transformer-family ablation: one variable at a time, matched 2000 fixed steps.
# Isolates hypotheses 1 (bilinear), 2 (complex rotation), 4 (Fourier readout /
# tying) and 3 (T-conditioned angle multiplier) against a matched control.
set -u
cd "$(dirname "$0")/.."
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
MAN=lab/manifests/lab_e1_nw0_fs2000_s74_75_76.json

run() {
  local name="$1"; shift
  local note="$1"; shift
  $VENV lab/make_gr.py --name "$name" "$@" >/dev/null || return 1
  $VENV lab/run_experiment.py \
    --submission "submissions/group-rotation/$name/submission.py" \
    --manifest "$MAN" --tag gr-ablate --note "$note" --timeout 3600
}

run c0_ctrl   "GR control: std MLP, tied head, L8 d128"      --mixer std
run a1_bilin  "H1 bilinear mixer (pure product of 2 projections)" --mixer bilin
run a2_geglu  "H1 control: GEGLU (gated, not pure bilinear)" --mixer geglu
run a3_rot    "H2 complex-rotation sublayer (unit-modulus)"  --rot 1
run a4_gbp    "outer-product global bilinear pooling"        --gbp 1
run a5_fhead  "H4 Fourier (phase) readout + linear head"     --fhead 1
run a6_tcond  "H3 T-conditioned per-frequency angle multiplier" --fhead 1 --tcond 1
run a7_untied "H4b untied input embedding / output head"     --tie 0
echo "ABLATE DONE"
