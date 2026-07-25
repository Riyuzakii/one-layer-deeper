#!/usr/bin/env bash
# GRPair fixed-step grid on e1.  The pair-phase table memorises the ~600 training
# rows in <100 steps, so every run here is >=200x past the memorisation point:
# what is being measured is whether the *delayed generalisation* (grokking)
# transition happens, and under what bottleneck / regularisation.
#
# The (K, H) axis is the important one.  With K learned base angles and H
# harmonics the readout sees K*H distinct Fourier components but the model has
# only K degrees of freedom in phase space -- small K, large H is the classic
# "few key frequencies" grokking parameterisation and is the configuration in
# which memorisation through a linear readout is hardest.
set -u
cd "$(dirname "$0")/.."
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
MAN=lab/manifests/lab_e1_nw0_fs20000_s74.json

run() {
  local name="$1"; shift
  local note="$1"; shift
  $VENV lab/make_grpair.py --name "$name" "$@" >/dev/null || return 1
  $VENV lab/run_experiment.py \
    --submission "submissions/group-rotation/$name/submission.py" \
    --manifest "$MAN" --tag grpair-grid --note "$note" --timeout 5400
}

run gp_k128h1 "GRPair rev K128 H1 wd1.0 20k"                  --freqs 128 --harm 1 --wd 1.0
run gp_k16h8  "GRPair rev K16  H8  wd1.0 20k"                 --freqs 16  --harm 8 --wd 1.0
run gp_k8h16  "GRPair rev K8   H16 wd1.0 20k"                 --freqs 8   --harm 16 --wd 1.0
run gp_k4h32  "GRPair rev K4   H32 wd1.0 20k"                 --freqs 4   --harm 32 --wd 1.0
run gp_k8h16w3 "GRPair rev K8  H16 wd3.0 20k"                 --freqs 8   --harm 16 --wd 3.0
run gp_ord1   "GRPair rev K8 H16 ORDER=1 (control: additive phase, no product)" \
                                                              --freqs 8   --harm 16 --wd 1.0 --order 1
run gp_abs    "GRPair abs K8 H16 wd1.0 (absolute slots, not tail-aligned)" \
                                                              --freqs 8   --harm 16 --wd 1.0 --posmode abs
run gp_tcond  "GRPair rev K8 H16 wd1.0 + T-conditioned angle multiplier" \
                                                              --freqs 8   --harm 16 --wd 1.0 --tcond 1
echo "GRPAIR DONE"
