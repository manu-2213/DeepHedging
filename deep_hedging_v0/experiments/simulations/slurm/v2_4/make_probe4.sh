#!/bin/bash
# v2.4 probe 4 (2026-11-08): discount factor. The measure is the undiscounted mean over all N=250 steps of
# |PV_t - L_t|; the tracking reward uses gamma 0.99 (horizon ~100 steps). Try gamma 0.995 and 1.0 at the 0.1% fee,
# where the nets under-trade (about 30 trades per path against 127 for Whalley-Wilmott). Selection on tracking error.
set -e
cd "$(dirname "$0")"
for base in tfp grpo; do
  for g in 995 1; do
    gv=$([ $g = 1 ] && echo 1.0 || echo 0.$g)
    sed -e "s/${base}_fee01_s1/${base}g${g}_fee01_s1/g" -e "s/ --name / --gamma $gv --metric mean_abs_tracking_error --name /" run_${base}_fee01_s1.slurm > run_${base}g${g}_fee01_s1.slurm
  done
done
grep -ho -- "--gamma [0-9.]* --metric [a-z_]* --name [a-z0-9_]*" run_tfpg*_fee01_s1.slurm run_grpog*_fee01_s1.slurm
