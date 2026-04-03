#!/bin/bash
#SBATCH --job-name=hp_c16
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

#SBATCH --account=pilot_sae_gpu
#SBATCH --partition=sae
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --time=240:00:00

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL

# Config 16 — inactor LR sweep  |  actor: lr=2e-5 cosine (fixed baseline)
# inactor_lr = 5e-6  (previous implicit value — PPOConfig.learning_rate_ex)

cd /data/home/acw796/DeepHedging || exit 1

module load cuda/12.4.0-gcc-12.2.0
module load python/3.11.7-gcc-12.2.0
source .venv/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

for seed in {1..10}; do
    python deep_hedging_v0/experiments/hp_tuning/inactive_bsm_hedge_call_mlp_hp.py \
        --seed $seed --lr 2e-5 --scheduler cosine --inactor_lr 5e-6
done
