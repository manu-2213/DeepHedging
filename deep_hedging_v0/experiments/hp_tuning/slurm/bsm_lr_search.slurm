#!/bin/bash
#SBATCH --job-name=bsm_lr_search
#SBATCH --output=%x.%A_%a.out
#SBATCH --error=%x.%A_%a.err
#SBATCH --account=pilot_sae_gpu
#SBATCH --partition=sae
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240G
#SBATCH --time=240:00:00
#SBATCH --array=0-5%2

set -eo pipefail

cd "${DEEP_HEDGING_REPO:-/data/home/acw796/DeepHedging}"
module load cuda/12.4.0-gcc-12.2.0
module load python/3.11.7-gcc-12.2.0
source .venv/bin/activate
set -u

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export DEEP_HEDGING_PATH_CACHE=1
export DEEP_HEDGING_NUM_PATHS=50
export DEEP_HEDGING_NUM_PATHS_HESTON=10
export DEEP_HEDGING_SUB_BATCH_NUM=4

experiments=(call_mlp call_rnn con_mlp con_rnn doc_mlp doc_rnn)
experiment="bsm_hedge_${experiments[${SLURM_ARRAY_TASK_ID}]}"
output_dir="deep_hedging_v0/experiments/hp_tuning/results/lr_search_${SLURM_ARRAY_JOB_ID}"
mkdir -p "$output_dir"

python -u deep_hedging_v0/experiments/hp_tuning/sweep_learning_rates.py \
    --scripts "${experiment}.py" "inactive_${experiment}.py" \
    --epochs 60 --policy-epochs 40 --inaction-epochs 20 --episodes 400 \
    --seed "${SWEEP_SEED:-0}" \
    --output "${output_dir}/${experiment}.csv" "$@"