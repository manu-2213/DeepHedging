#!/bin/bash
# v2.4 probe 3 (2026-11-07 late).
#  (a) The two diagnostics of probe 2, on LSTM + GRPO (the most consistent method):
#      grpocvar  : --reward mean_cvar       grpostate : agent state on
#  (b) Fast-transformer learning-rate sweep (it has only ever used the LSTM's rate):
#      tfplr3e5 / tfplr3e4 : actor rate 3e-5 / 3e-4, seeds 1-3 ; tfp seed 3 at the base rate 1e-4
set -e
cd "$(dirname "$0")"
for fee in 01 1; do
  g=run_grpo_fee${fee}_s1.slurm
  sed -e "s/grpo_fee${fee}_s1/grpocvar_fee${fee}_s1/g" -e "s/--reward tracking/--reward mean_cvar/" $g > run_grpocvar_fee${fee}_s1.slurm
  sed -e "s/grpo_fee${fee}_s1/grpostate_fee${fee}_s1/g" -e "s/ --no-agent-state//" $g > run_grpostate_fee${fee}_s1.slurm
  src=run_tfp_fee${fee}_s1.slurm
  sed -e "s/tfp_fee${fee}_s1/tfp_fee${fee}_s3/g" -e "s/--tuning-seeds 1 /--tuning-seeds 3 /" $src > run_tfp_fee${fee}_s3.slurm
  for s in 1 2 3; do
    for lr in 3e-5 3e-4; do
      tag=tfplr${lr/-/}
      sed -e "s/tfp_fee${fee}_s1/${tag}_fee${fee}_s${s}/g" -e "s/--tuning-seeds 1 /--tuning-seeds $s /" \
          -e "s/--actor-lr-range 1e-4 1e-4/--actor-lr-range $lr $lr/" $src > run_${tag}_fee${fee}_s${s}.slurm
    done
  done
done
# Selection on tracking error (2026-11-07: tracking error is the measure; costs are reported alongside).
for f in run_tfpstate_fee*_s1.slurm run_grpostate_fee*_s1.slurm run_tfplr*.slurm run_tfp_fee*_s3.slurm; do [ -f "$f" ] && ! grep -q -- "--metric" "$f" && sed -i "s/ --name / --metric mean_abs_tracking_error --name /" "$f"; done; true
