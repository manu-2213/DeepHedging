#!/bin/bash
#SBATCH --job-name=illiquid_all
#SBATCH --output=%x.%A_%a.out
#SBATCH --error=%x.%A_%a.err
#SBATCH --account=pilot_sae_gpu
#SBATCH --partition=sae
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --time=240:00:00
#SBATCH --array=0-23%4

set -eo pipefail

cd "${DEEP_HEDGING_REPO:-/data/home/acw796/DeepHedging}"
module load cuda/12.4.0-gcc-12.2.0
module load python/3.11.7-gcc-12.2.0
source .venv/bin/activate
set -u

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Illiquid hedging configuration
# transaction_fee_rate sets the underlying cost; options cost = rate * 2 (hardcoded in conc/doc envs)
# rate=0.01 => underlying 5%, options 10%
export DEEP_HEDGING_TRANSACTION_FEE_RATE=0.05
export WANDB_PROJECT=SEVERLY_ILLIQUID_HEDGING_5_10
export WANDB_MODE=online
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export DEEP_HEDGING_PATH_CACHE=1
export DEEP_HEDGING_NUM_PATHS=50
export DEEP_HEDGING_NUM_PATHS_HESTON=10
export DEEP_HEDGING_SUB_BATCH_NUM=4

# All simulation experiments (24 total):
# 0-5: bsm (6 baseline + inactive variants)
# 6-11: heston (6 baseline + inactive variants)
# etc.

experiments=(
    # BSM experiments
    "bsm_hedge_call_mlp"
    "bsm_hedge_call_rnn"
    "bsm_hedge_con_mlp"
    "bsm_hedge_con_rnn"
    "bsm_hedge_doc_mlp"
    "bsm_hedge_doc_rnn"
    # Heston experiments
    "heston_hedge_call_mlp"
    "heston_hedge_call_rnn"
    "heston_hedge_con_mlp"
    "heston_hedge_con_rnn"
    "heston_hedge_doc_mlp"
    "heston_hedge_doc_rnn"
    # Inactive BSM experiments
    "inactive_bsm_hedge_call_mlp"
    "inactive_bsm_hedge_call_rnn"
    "inactive_bsm_hedge_con_mlp"
    "inactive_bsm_hedge_con_rnn"
    "inactive_bsm_hedge_doc_mlp"
    "inactive_bsm_hedge_doc_rnn"
    # Inactive Heston experiments
    "inactive_heston_hedge_call_mlp"
    "inactive_heston_hedge_call_rnn"
    "inactive_heston_hedge_con_mlp"
    "inactive_heston_hedge_con_rnn"
    "inactive_heston_hedge_doc_mlp"
    "inactive_heston_hedge_doc_rnn"
)

script_name="${experiments[${SLURM_ARRAY_TASK_ID}]}"
output_dir="deep_hedging_v0/experiments/hp_tuning/results/illiquid_hedging_${SLURM_ARRAY_JOB_ID}/${script_name}"
mkdir -p "$output_dir"

echo "=========================================="
echo "Running experiment: $script_name"
echo "Job ID: $SLURM_ARRAY_JOB_ID Task ID: $SLURM_ARRAY_TASK_ID"
echo "Transaction fee rate: $DEEP_HEDGING_TRANSACTION_FEE_RATE (5% for illiquid options)"
echo "W&B Project: $WANDB_PROJECT"
echo "Output dir: $output_dir"
echo "=========================================="

python -u deep_hedging_v0/experiments/hp_tuning/run_experiment.py \
    --script "${script_name}.py" \
    --output "$output_dir" \
    --epochs 60 --policy-epochs 40 --inaction-epochs 20 --episodes 400
