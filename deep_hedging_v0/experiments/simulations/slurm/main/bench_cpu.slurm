#!/bin/bash
#SBATCH --job-name=mn_bench_cpu
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=computeshort
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:20:00
# CPU-speed benchmark (2026-11-10): the main-table single net on 8 CPU cores for 5 minutes; reports s/it.
set -eo pipefail
cd ${DEEP_HEDGING_REPO:-/data/home/acw796/DeepHedging}/deep_hedging_v0
module load python/3.11.7-gcc-12.2.0
source ../.venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export FAST_HEDGING_CACHE=${FAST_HEDGING_CACHE:-$PWD/../.fast_hedging_cache}
C="--dynamics bsm --product call --net single --actor-lr-range 1e-4 1e-4 --inactor-lr-range 1e-3 1e-3 --tuning-seeds 1 --underlying-fee 0.01 --option-fee 0.05 --init-std 0.2 --n-trials 1 --no-pruning --final-seeds --arch tfpath --rnn-mode window --ppo-epochs 1 --minibatch 125000 --target-kl 0 --reward tracking --reward-norm none --no-obs-norm --no-adv-norm --value-loss huber --value-coef 1.0 --window 10 --pad-mode zero --no-agent-state --call-action-low -1 --train-paths 2000 --val-paths 500 --test-paths 2000 --max-iters 1000000000 --patience 200 --lr-patience 50 --lr-decays 3 --lr-decay-factor 0.5 --advantage grpo --grpo-group 8 --paths-per-iter 12 --metric mean_abs_tracking_error --time-budget-min 5 --device cpu"
B=experiments/results/main_checks/bench_cpu; rm -rf $B; mkdir -p $B
python -u -m fast_hedging.experiment $C --name bench_cpu --out $B > $B/stdout.txt 2>&1 || true
grep -E "\] it " $B/log.txt | tail -2
echo "BENCH DONE"
