#!/bin/bash
# v2.4 probe 2: does the reward or the missing agent state explain the gap to Whalley-Wilmott?
# From run_tfp_fee{01,1}_s1.slurm (fast transformer, PPO, tracking reward, no agent state), change one thing:
#   tfpcvar  : --reward mean_cvar (E[PnL] - CVaR95, each cost counted once; gamma 1)
#   tfpstate : agent state on (holdings, tracking error, costs, time since trade fed to the net)
set -e
cd "$(dirname "$0")"
for fee in 01 1; do
  src=run_tfp_fee${fee}_s1.slurm
  sed -e "s/tfp_fee${fee}_s1/tfpcvar_fee${fee}_s1/g" -e "s/--reward tracking/--reward mean_cvar/" $src > run_tfpcvar_fee${fee}_s1.slurm
  sed -e "s/tfp_fee${fee}_s1/tfpstate_fee${fee}_s1/g" -e "s/ --no-agent-state//" $src > run_tfpstate_fee${fee}_s1.slurm
done
grep -c "mean_cvar" run_tfpcvar_*.slurm; grep -c "no-agent-state" run_tfpstate_*.slurm || true
# Selection on tracking error (2026-11-07: tracking error is the measure; costs are reported alongside).
for f in run_tfpstate_fee*_s1.slurm run_grpostate_fee*_s1.slurm run_tfplr*.slurm run_tfp_fee*_s3.slurm; do [ -f "$f" ] && ! grep -q -- "--metric" "$f" && sed -i "s/ --name / --metric mean_abs_tracking_error --name /" "$f"; done; true
