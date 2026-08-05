#!/usr/bin/env bash
# Hard-FAITHFUL local proxies.
#
# The earlier proxies (lab/gen_hard_proxy.sh, hp1/hp2/hp3) used --split_group prompt,
# where train and test share moduli. The one hosted Hard run showed h1's scoring splits
# are `test`, `ood_t`, `ood_n_t`, which in data/squaring_mod.py is produced ONLY by
# _generate_modulus_grouped_records — i.e. split_group=modulus with separate_ood_splits.
# So hp1/hp2/hp3 model the wrong thing; these replace them.
#
# WHAT THAT STRUCTURE BUYS, AND WHY IT MATTERS
#   * train and test draw from DISJOINT modulus pools, so even `test` requires
#     unseen-modulus generalisation. Per-modulus lookups are dead by construction.
#   * ood_t  = training-pool moduli at an unseen T   (seen N, unseen T)
#   * ood_n_t = held-out moduli at an unseen T       (unseen N, unseen T)
#   * depth_t_*        rungs use SEEN (training-pool) moduli  -> Max T
#   * depth_ood_n_t_*  rungs use unseen modulus SIZES         -> OOD-N Max T
#
# CONSTRAINTS THIS CONFIG IS BUILT AROUND (all from the generator source)
#   1. split_group=modulus requires SAMPLED moduli (no --fixed_p/--fixed_q).
#   2. HARD CAP: _enumerate_sampled_factor_pairs returns None once
#      max(p_bits, q_bits) > 10, and the modulus-grouped path raises on that. So
#      in-distribution modulus_bits <= 20. 21+ bits is IMPOSSIBLE for the ID side.
#      (m4 reaches 22 bits only because it uses split_group=prompt.)
#   3. ood_n_depth_evaluation_modulus_bits must NOT overlap modulus_bits. Those moduli
#      are rejection-sampled by _sample_rsa_factors, NOT enumerated, so they may exceed
#      20 bits — which is why 21 is legal here and 21 is not legal above.
#   4. depth_evaluation_exhaustive_x requires fixed_p/fixed_q, so it is unavailable;
#      depth_evaluation_examples_per_setting is required instead.
#   5. ood_time_steps must be disjoint from time_steps.
#   6. _partition_factor_pairs splits the enumerated pairs by COUNT and then checks
#      sum (p-1)(q-1) >= rows per split. Verified with >=100x margin at every bit size.
#
# LADDER DEGENERACY (the e1 trap: lambda(323)=144 collapses T=4/16/64 into one map, so
# e1 has four distinct rungs, not seven). Fraction of moduli with all seven rungs
# distinct, by bit size: 12b 43%, 14b 52%, 16b 82%, 18b 91%, 20b 91%. The chosen
# [16,18,20] is the non-degenerate end of the enumerable range.
#
# ID [16,18,20] with OOD-N [17,19,21] interleaves the sizes, matching the README's
# "unseen identities at nearby dataset-scale modulus sizes".
#
# COMPLIANCE: runs the public generator only. Generated rows are never read, printed or
# summarised — data/generated/ stays write-only.
#
# Usage:  PATH=.venv/bin:$PATH bash lab/gen_hard_faithful.sh
set -euo pipefail

COMMON=(
  --modulus_bits '[16,18,20]'
  --time_steps '[4,8,16]' --ood_time_steps '[32]'
  --depth_evaluation_time_steps '[1,2,4,8,16,32,64]'
  --depth_evaluation_examples_per_setting 256
  --ood_n_depth_evaluation_modulus_bits '[17,19,21]'
  --ood_n_depth_evaluation_examples_per_setting 256
  --train_fraction 0.9 --test_fraction 0.1
  --split_group modulus --separate_ood_splits true
  --seed 45 --separate_input_output true
)

# HF1 — the primary Hard-faithful proxy. ~299k rows (~81k train rows per T setting).
python -m data.squaring_mod \
  --output_dir data/generated/proxy_hard_modulus_b161820_t4816 \
  --examples_per_setting 30000 --ood_examples_per_setting 3000 \
  "${COMMON[@]}"

# HF1-small — identical structure, ~49k rows. For calibrating fitting curves cheaply
# before committing to a long run. NOTE: fewer rows makes memorisation EASIER, so the
# generalisation gap is not directly comparable to the primary — use it for step-count
# calibration and smoke tests, not for accuracy claims.
python -m data.squaring_mod \
  --output_dir data/generated/proxy_hard_modulus_b161820_t4816_small \
  --examples_per_setting 4000 --ood_examples_per_setting 400 \
  "${COMMON[@]}"
