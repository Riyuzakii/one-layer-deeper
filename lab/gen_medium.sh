#!/usr/bin/env bash
# Generate Medium m1-m5 (deeper T: m1/m2 train T{4,8,16} OOD T=32 -> strong depth test).
set -u
cd /home/scratch.arohan_hw/git/one-layer-deeper
PY=.venv/bin/python
G="squaring_mod_new11_medium_bidirectional"
$PY -m data.squaring_mod --output_dir data/generated/${G}_fixed_n_10403_t4816 --fixed_p 101 --fixed_q 103 --time_steps '[4,8,16]' --ood_time_steps '[32]' --examples_per_setting 10000 --ood_examples_per_setting 3000 --train_fraction 0.9 --test_fraction 0.1 --split_group prompt --seed 45 --separate_input_output true && echo M1_done
$PY -m data.squaring_mod --output_dir data/generated/${G}_fixed_n_38021_t4816 --fixed_p 193 --fixed_q 197 --time_steps '[4,8,16]' --ood_time_steps '[32]' --examples_per_setting 30000 --ood_examples_per_setting 5000 --train_fraction 0.9 --test_fraction 0.1 --split_group prompt --seed 45 --separate_input_output true && echo M2_done
$PY -m data.squaring_mod --output_dir data/generated/${G}_fixed_t_b111315_t2 --modulus_bits '[11,13,15]' --fixed_time_steps 2 --ood_time_steps '[4]' --examples_per_setting 8000 --ood_examples_per_setting 1000 --train_fraction 0.9 --test_fraction 0.1 --split_group prompt --seed 45 --separate_input_output true && echo M3_done
$PY -m data.squaring_mod --output_dir data/generated/${G}_fixed_t_b141822_t8 --modulus_bits '[14,18,22]' --fixed_time_steps 8 --ood_time_steps '[16]' --examples_per_setting 30000 --ood_examples_per_setting 3000 --train_fraction 0.9 --test_fraction 0.1 --split_group prompt --seed 45 --separate_input_output true && echo M4_done
$PY -m data.squaring_mod --output_dir data/generated/${G}_variable_b121416_t248 --modulus_bits '[12,14,16]' --time_steps '[2,4,8]' --ood_time_steps '[16]' --examples_per_setting 10000 --ood_examples_per_setting 1000 --train_fraction 0.9 --test_fraction 0.1 --split_group prompt --seed 45 --separate_input_output true && echo M5_done
echo MEDIUM_ALL_DONE
