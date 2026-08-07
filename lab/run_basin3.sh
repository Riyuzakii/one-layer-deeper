#!/usr/bin/env bash
# (a) the objective PROFILE (no search: how informative is the legal label at
#     each corruption level?), (b) discrete search from RANDOM init -- the
#     "can it reach the basin at all" question -- and (c) module-restricted
#     searches: with every other table constructed, does the end-of-chain label
#     identify the ADDER on its own?  In DigitALU the same question about Tmul
#     answered no (alu-relational); ADD-ONLY has nothing but the adder, so this
#     is the sharpest form of the branch's hypothesis.
set -u
V=/home/scratch.arohan_hw/git/one-layer-deeper/.venv/bin/python
cd "$(dirname "$0")/.."
mkdir -p lab/logs

# (a) profile
$V lab/probe_addsearch.py --modulus 323 --slots 3 --train-x 250 --basin \
  --basin-reps 8 --tie sym,inv --tag prof_addA_tied \
  > lab/logs/prof_addA_tied.log 2>&1
$V lab/probe_search.py --modulus 323 --slots 3 --train-x 250 --basin \
  --basin-reps 8 --tie sym,inv --tag prof_dalu_tied \
  > lab/logs/prof_dalu_tied.log 2>&1

# (b) from RANDOM init, 7 seeds, both tie settings
: > lab/logs/rand_addA.log
for s in 0 1 2 3 4 5 6; do
  $V lab/probe_addsearch.py --modulus 323 --slots 3 --train-x 250 \
    --tie sym,inv --repair-ks 0 --seed $s --tag rand_addA_tied_s$s \
    --jsonl lab/rand_runs.jsonl >> lab/logs/rand_addA.log 2>&1
done
: > lab/logs/rand_addA_untied.log
for s in 0 1 2; do
  $V lab/probe_addsearch.py --modulus 323 --slots 3 --train-x 250 \
    --repair-ks 0 --seed $s --tag rand_addA_untied_s$s \
    --jsonl lab/rand_runs.jsonl >> lab/logs/rand_addA_untied.log 2>&1
done

# (c) module-restricted: only `add` random and searched, everything else
#     CONSTRUCTED (LAB DIAGNOSTIC).  Then only `sub`, then only `pick`.
: > lab/logs/mod_addA.log
for m in add sub pick add,sub; do
  for s in 0 1 2; do
    $V lab/probe_addsearch.py --modulus 323 --slots 3 --train-x 250 \
      --modules $m --repair-ks 0 --seed $s --tag mod_${m//,/+}_s$s \
      --jsonl lab/mod_runs.jsonl >> lab/logs/mod_addA.log 2>&1
  done
done
# the same question for DigitALU's Tmul, as the reference point
: > lab/logs/mod_dalu.log
for m in mul add; do
  for s in 0 1 2; do
    $V lab/probe_search.py --modulus 323 --slots 3 --train-x 250 \
      --modules $m --seed $s --tag modD_${m}_s$s \
      --jsonl lab/mod_runs_dalu.jsonl >> lab/logs/mod_dalu.log 2>&1
  done
done
echo DONE
