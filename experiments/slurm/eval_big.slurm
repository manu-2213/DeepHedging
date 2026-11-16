#!/bin/bash
#SBATCH --job-name=mn_big
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=gpushort,sae
#SBATCH --account=pilot_sae_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
# Large-sample test evaluation (50,000 fresh paths) of one cell's selected seed-1 nets, for the P&L histograms (2026-11-12).
# args: dyn prod fee single_dir double_dir. The bank is cached, so a resubmission after the time limit resumes.
set -eo pipefail
cd ${DEEP_HEDGING_REPO:-/data/home/acw796/DeepHedging}
module load cuda/12.4.0-gcc-12.2.0
module load python/3.11.7-gcc-12.2.0
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q V100; then source .venv_v100/bin/activate; else source .venv/bin/activate; fi
export PYTHONUNBUFFERED=1 HEDGING_ENGINE_CACHE=${HEDGING_ENGINE_CACHE:-$PWD/.market_cache}
python -u -m experiments.evaluation.eval_big "$@"
