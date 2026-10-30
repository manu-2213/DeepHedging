#!/bin/bash
#SBATCH --job-name=fast_smoke_test
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --account=pilot_sae_gpu
#SBATCH --partition=sae
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# Fast pipeline, single friction regime: 1% underlying, 5% options (experiments/fast/config.py).
set -eo pipefail
cd "${DEEP_HEDGING_REPO:-/data/home/acw796/DeepHedging}"
module load cuda/12.4.0-gcc-12.2.0
module load python/3.11.7-gcc-12.2.0
source .venv/bin/activate
set -u

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export WANDB_MODE=${WANDB_MODE:-offline}
# BSM banks reuse hedging/simulators.py, which imports JAX; keep JAX off the GPU.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cpu

nvidia-smi --query-gpu=name,memory.total --format=csv || true

# Tiny end-to-end check (small banks, 2 trials, 1 seed). Run this first; results go to results/smoke.
for exp in bsm_call_rnn bsm_doc_rnn heston_con_rnn heston_doc_rnn; do
    python -u deep_hedging_v0/experiments/fast/run.py --experiment $exp --mode single --smoke --final-seeds 1
    python -u deep_hedging_v0/experiments/fast/run.py --experiment $exp --mode double --smoke --final-seeds 1
done
