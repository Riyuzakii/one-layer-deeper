#!/usr/bin/env bash
# Phase T6 -- with the iteration mechanism chosen, push per-step exactness.
#
# Why these knobs:
#  * `tsoft` -- a LEARNED scalar read-out pointer mu regressed from the prompt,
#    p_k = softmax(-(k-mu)^2/2sigma^2).  Because mu is a scalar it can leave the
#    trained range, which is what m1/hp1 certification needs (rungs T=1,2 are
#    BELOW the trained T in {4,8,16}).  Nothing about T is parsed.
#  * `swiglu` -- x^2 mod N written out in decimal digits a,b,c is
#    (310a^2+100b^2+c^2+62ab+200ac+20bc) mod 323, i.e. a BILINEAR form in the
#    digits followed by a modular fold.  A gated (multiplicative) MLP can express
#    digit products directly; a GELU MLP has to approximate them.  This is an
#    inductive-bias choice, not an implementation of the arithmetic.
#  * `reembed_st` -- the intermediate after k squarings is a real residue, so the
#    inter-iteration state is forced back onto the token-embedding manifold with
#    a straight-through hard argmax: exact forward, unbroken gradient.
set -u
cd "$(dirname "$0")/.."
VENV=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
MAN=${MAN:-lab/manifests/lab_e1_fs8000_s3.json}
S=${S:-8000}
BS=${BS:-64}
WD=${WD:-1.0}

run () {
  local tag=$1; shift
  $VENV lab/make_tied.py --tag "$tag" --max-steps $S --batch-size $BS --wd $WD "$@" >/dev/null
  echo "##### $tag"
  $VENV lab/run_experiment.py --submission "submissions/exp_tied/$tag/submission.py" \
    --manifest "$MAN" --tag T6-arch --note "per-step exactness axis: $tag" --timeout 12000
}

run a_base      --loops 4 --state-mode res        --pos-mode rev
run a_swiglu    --loops 4 --state-mode res        --pos-mode rev --act swiglu
run a_reest     --loops 4 --state-mode reembed_st --pos-mode rev --act swiglu
run a_tsoft     --loops 4 --eval-loops 8 --iter-mode tsoft --state-mode res --pos-mode rev \
                --act swiglu --beta 0.02 --warmup 2000 --ramp 2000
