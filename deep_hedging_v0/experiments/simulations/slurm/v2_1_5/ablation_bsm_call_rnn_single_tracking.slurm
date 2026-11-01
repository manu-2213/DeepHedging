#!/bin/bash
#SBATCH --job-name=bsm_call_rnn_single_tracking
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --account=pilot_sae_gpu
#SBATCH --partition=sae
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=240:00:00

set -eo pipefail
cd ${DEEP_HEDGING_REPO:-/data/home/acw796/DeepHedging}/deep_hedging_v0
module load cuda/12.4.0-gcc-12.2.0
module load python/3.11.7-gcc-12.2.0
source ../.venv/bin/activate
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export FAST_HEDGING_CACHE=${FAST_HEDGING_CACHE:-$PWD/../.fast_hedging_cache}
nvidia-smi --query-gpu=name,memory.total --format=csv || true

# One run: Optuna LR search (2 tuning seeds per trial, validation CVaR95, early stopping,
# median pruning), then 10 final seeds at the best LRs, evaluated on the held-out 2021 year.
# Resumable: resubmitting this file continues the study and skips finished seeds.
python -u -m fast_hedging.experiment \
    --dynamics bsm --product call --net single --reward tracking \
    --arch rnn --rnn-mode window --window 8 --underlying-fee 0.01 --option-fee 0.05 --digital-payout-frac 0.1 \
    --n-trials 30 --tuning-hours 150 --tuning-seeds 0 1000 --final-seeds 1 2 3 4 5 6 7 8 9 10 \
    --name bsm_call_rnn_single_tracking --out experiments/results/v2_1_5/bsm_call_rnn_single_tracking
