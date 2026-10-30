#!/bin/bash
#SBATCH --job-name=inactive_heston_hedge_call_mlp
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

#SBATCH --account=pilot_sae_gpu
#SBATCH --partition=sae          # GPU partition
#SBATCH --gres=gpu:1             # request 1 GPU
#SBATCH --cpus-per-task=12       # CPU cores
#SBATCH --mem=240G               # total RAM
#SBATCH --time=240:00:00         # walltime

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL

# ----------------------------
# Environment
# ----------------------------
cd /data/home/acw796/DeepHedging || exit 1

module load cuda/12.4.0-gcc-12.2.0
module load python/3.11.7-gcc-12.2.0
source .venv/bin/activate

# JAX / PyTorch GPU memory fix
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Simulated paths are cached on disk (see hedging/simulators.py); run
# pregenerate_paths.slurm once beforehand (same seeds) so these 10 runs
# reuse cached paths instead of re-simulating them.
export DEEP_HEDGING_PATH_CACHE=1

export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

exec python -u deep_hedging_v0/experiments/hp_tuning/run_experiment.py \
    --script inactive_heston_hedge_call_mlp.py \
    --output "deep_hedging_v0/experiments/hp_tuning/results/inactive_heston_hedge_call_mlp_${SLURM_JOB_ID}" \
    --epochs 60 --policy-epochs 40 --inaction-epochs 20 --episodes 400
