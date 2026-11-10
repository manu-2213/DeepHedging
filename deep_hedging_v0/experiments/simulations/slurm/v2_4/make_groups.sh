#!/bin/bash
# v2.4 GRPO group size on the fast transformer single net (2026-11-09). Group 8 was far better than group 4 at the same rate,
# so test larger groups at about the same rollout size. One change each from tfpgrposingle, seed 1, tracking selection:
#   tgsg16 : 16 copies x 6 paths      tgsg32 : 32 copies x 3 paths
set -e
cd "$(dirname "$0")"
for fee in 01 1; do
  b=run_tfpgrposingle_fee${fee}_s1.slurm; o=tfpgrposingle_fee${fee}_s1
  sed -e "s/$o/tgsg16_fee${fee}_s1/g" -e "s/--grpo-group 4 --paths-per-iter 25/--grpo-group 16 --paths-per-iter 6/" $b > run_tgsg16_fee${fee}_s1.slurm
  sed -e "s/$o/tgsg32_fee${fee}_s1/g" -e "s/--grpo-group 4 --paths-per-iter 25/--grpo-group 32 --paths-per-iter 3/" $b > run_tgsg32_fee${fee}_s1.slurm
done
