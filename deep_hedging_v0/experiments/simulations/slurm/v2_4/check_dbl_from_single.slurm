#!/bin/bash
#SBATCH --job-name=v24_check_dbl
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=computeshort
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
# Tests the path the double-net grid depends on: --actor-init from-single, inactor + joint rates, chained resume (exit 75).
# The source single net (tfpsingle_fee01_s1) is a poor actor; this checks the machinery, not the result.
set -eo pipefail
cd /data/home/acw796/DeepHedging/deep_hedging_v0
module load python/3.11.7-gcc-12.2.0
source ../.venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export FAST_HEDGING_CACHE=$PWD/../.fast_hedging_cache
OUT=experiments/results/v2_4_checks/dbl_from_single
rm -rf $OUT; mkdir -p $OUT
for k in 1 2 3 4 5 6 7 8; do
  rc=0
  python -u -m fast_hedging.experiment --dynamics bsm --product call --net double --n-trials 1 --no-pruning --final-seeds --actor-lr-range 1e-4 1e-4 --inactor-lr-range 1e-4 1e-4 --joint-lr-range 1e-5 1e-5 --actor-init from-single --single-dir experiments/results/v2_4/tfpsingle_fee01_s1 --tuning-seeds 1 --arch tfpath --rnn-mode window --paths-per-iter 20 --ppo-epochs 1 --minibatch 25000 --target-kl 0 --reward tracking --reward-norm none --no-obs-norm --no-adv-norm --value-loss huber --value-coef 1.0 --window 10 --pad-mode zero --no-agent-state --call-action-low -1 --underlying-fee 0.001 --option-fee 0.002 --train-paths 200 --val-paths 100 --test-paths 200 --max-iters 40 --min-iters 10 --patience 200 --lr-patience 50 --lr-decays 3 --lr-decay-factor 0.5 --time-budget-min 1 --metric mean_abs_tracking_error --device cpu --name dbl_from_single --out $OUT || rc=$?
  echo "pass $k exit $rc"
  [ "$rc" -eq 75 ] || break
done
[ "$rc" -eq 0 ] && [ -f $OUT/summary.json ] && echo "CHECK PASSED" || { echo "CHECK FAILED rc=$rc"; exit 1; }
