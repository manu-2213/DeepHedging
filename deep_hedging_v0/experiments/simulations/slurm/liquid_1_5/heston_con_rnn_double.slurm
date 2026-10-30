#!/bin/bash
#SBATCH --job-name=heston_con_rnn_double
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --account=pilot_sae_gpu
#SBATCH --partition=sae
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=96:00:00

set -eo pipefail
cd ${DEEP_HEDGING_REPO:-/data/home/acw796/DeepHedging}/deep_hedging_v0
module load cuda/12.4.0-gcc-12.2.0
module load python/3.11.7-gcc-12.2.0
source ../.venv/bin/activate
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export FAST_HEDGING_CACHE=${FAST_HEDGING_CACHE:-$PWD/../.fast_hedging_cache}
mkdir -p experiments/simulations/slurm/liquid_1_5/logs

# One run: Optuna learning-rate search (validation CVaR95, early stopping with patience),
# then 5 final seeds with the best rates, evaluated on the held-out test year.
# Resumable: resubmitting this same file continues the study and skips finished seeds.
python -u -m fast_hedging.experiment \
    --dynamics heston --product con --arch rnn --net double \
    --underlying-fee 0.01 --option-fee 0.05 \
    --n-trials 30 --tuning-hours 80 --final-seeds 1 2 3 4 5 \
    --out experiments/results/liquid_1_5/heston_con_rnn_double
