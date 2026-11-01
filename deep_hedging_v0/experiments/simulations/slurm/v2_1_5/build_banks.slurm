#!/bin/bash
#SBATCH --job-name=build_banks
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --account=pilot_sae_gpu
#SBATCH --partition=sae
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00

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

# Build the cached market banks (paths + all prices + model deltas) once per (dynamics, product),
# then run the martingale / sanity checks on each. Read the WARNING lines in the log.
for dyn in bsm heston; do
  for prod in call doc con; do
    python -u -m fast_hedging.experiment --dynamics $dyn --product $prod --digital-payout-frac 0.1 \
        --bank-only --out experiments/results/v2_1_5/_banks/${dyn}_${prod}
  done
done
grep -h "WARNING" experiments/results/v2_1_5/_banks/*/log.txt || echo "no bank warnings"
