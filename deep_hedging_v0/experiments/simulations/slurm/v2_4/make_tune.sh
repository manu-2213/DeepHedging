#!/bin/bash
# v2.4 transformer + GRPO tuning on the single net (2026-11-08). Until now these used the LSTM's settings unchanged.
# Base: run_tfpgrposingle_fee<f>_s1 (fast transformer, GRPO, rate 1e-4, group 4). One change each, seed 1, tracking selection:
#   tgslr3e5 / tgslr3e4 : actor rate 3e-5 / 3e-4        tgsg8 : GRPO group 8 (12 paths x 8, about the same rollout size)
#   tgsstd02 (0.1% fee) : initial noise 0.2, as the 1% runs already use
#   wtgs                : window transformer + GRPO instead of the fast transformer
set -e
cd "$(dirname "$0")"
for fee in 01 1; do
  b=run_tfpgrposingle_fee${fee}_s1.slurm; o=tfpgrposingle_fee${fee}_s1
  sed -e "s/$o/tgslr3e5_fee${fee}_s1/g" -e "s/--actor-lr-range 1e-4 1e-4/--actor-lr-range 3e-5 3e-5/" $b > run_tgslr3e5_fee${fee}_s1.slurm
  sed -e "s/$o/tgslr3e4_fee${fee}_s1/g" -e "s/--actor-lr-range 1e-4 1e-4/--actor-lr-range 3e-4 3e-4/" $b > run_tgslr3e4_fee${fee}_s1.slurm
  sed -e "s/$o/tgsg8_fee${fee}_s1/g" -e "s/--grpo-group 4 --paths-per-iter 25/--grpo-group 8 --paths-per-iter 12/" $b > run_tgsg8_fee${fee}_s1.slurm
  sed -e "s/$o/wtgs_fee${fee}_s1/g" -e "s/--arch tfpath/--arch transformer/" $b > run_wtgs_fee${fee}_s1.slurm
done
sed -e "s/tfpgrposingle_fee01_s1/tgsstd02_fee01_s1/g" -e "s/ --metric / --init-std 0.2 --metric /" run_tfpgrposingle_fee01_s1.slurm > run_tgsstd02_fee01_s1.slurm
