#!/bin/bash
#SBATCH --job-name=build_banks
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --account=pilot_sae_gpu
#SBATCH --partition=sae
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

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

# Build the cached market banks (paths + all option prices) once for every
# (dynamics, product). Run this first; the 24 runs then only load .npz files.
for dyn in bsm heston; do
  for prod in call doc con; do
    python -u -m fast_hedging.experiment --dynamics $dyn --product $prod \
        --bank-only --out experiments/results/liquid_1_5/_banks/${dyn}_${prod}
  done
done
