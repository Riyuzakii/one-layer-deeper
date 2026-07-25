#!/usr/bin/env bash
# Local Hard-tier proxies.
#
# The real Hard dataset (h1) is a private hidden evaluator: we know only that it
# sits above Medium on the same two knobs (modulus size, composition depth T) and
# is scored on the same T=1,2,4,8,16,32,64 certification ladder.  These proxies
# extend the public generator past Medium so an approach can be stress-tested on
# larger N and deeper T *before* spending one of the 1/day Hard attempts.
#
# COMPLIANCE: this only runs the public generator.  Generated rows are never read,
# printed, or summarized — data/generated/ is write-only for us.
#
# Usage:  PATH=.venv/bin:$PATH bash lab/gen_hard_proxy.sh
set -euo pipefail

# HP1: 22-bit fixed N (p=2003, q=2011 -> N=4,028,033), Medium-like T schedule.
# One step above M2 (16-bit) in modulus size; ladder to T=64.
python -m data.squaring_mod \
  --output_dir data/generated/proxy_hard_fixed_n_p2003_q2011_t4816 \
  --fixed_p 2003 --fixed_q 2011 \
  --time_steps '[4,8,16]' --ood_time_steps '[32]' \
  --examples_per_setting 30000 --ood_examples_per_setting 3000 \
  --depth_evaluation_time_steps '[1,2,4,8,16,32,64]' \
  --depth_evaluation_examples_per_setting 256 \
  --ood_n_depth_evaluation_modulus_bits '[24,26]' \
  --ood_n_depth_evaluation_examples_per_setting 256 \
  --train_fraction 0.9 --test_fraction 0.1 \
  --split_group prompt --seed 45 --separate_input_output true

# HP2: sampled 30/32-bit N.  Digit count roughly doubles vs HP1, so this is the
# stress test for per-step arithmetic at scale.  Sampled rather than fixed: the
# generator enumerates every unit of a fixed modulus, which is intractable past
# ~24 bits.
python -m data.squaring_mod \
  --output_dir data/generated/proxy_hard_sampled_b3032_t4816 \
  --modulus_bits '[30,32]' \
  --time_steps '[4,8,16]' --ood_time_steps '[32]' \
  --examples_per_setting 30000 --ood_examples_per_setting 3000 \
  --depth_evaluation_time_steps '[1,2,4,8,16,32,64]' \
  --depth_evaluation_examples_per_setting 256 \
  --ood_n_depth_evaluation_modulus_bits '[31,34]' \
  --ood_n_depth_evaluation_examples_per_setting 256 \
  --train_fraction 0.9 --test_fraction 0.1 \
  --split_group prompt --seed 45 --separate_input_output true

# HP3: sampled N spanning 20-28 bits at fixed T=8, with unseen 22/26/30-bit moduli
# in the OOD-N ladder.  This is the OOD-N (tie-break) stress test.
python -m data.squaring_mod \
  --output_dir data/generated/proxy_hard_sampled_b202428_t8 \
  --modulus_bits '[20,24,28]' --fixed_time_steps 8 \
  --ood_time_steps '[16]' \
  --examples_per_setting 30000 --ood_examples_per_setting 3000 \
  --depth_evaluation_time_steps '[1,2,4,8,16,32,64]' \
  --depth_evaluation_examples_per_setting 256 \
  --ood_n_depth_evaluation_modulus_bits '[22,26,30]' \
  --ood_n_depth_evaluation_examples_per_setting 256 \
  --train_fraction 0.9 --test_fraction 0.1 \
  --split_group prompt --seed 45 --separate_input_output true
