#!/bin/bash
#SBATCH --job-name=mn_years
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=gpushort,sae
#SBATCH --account=pilot_sae_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
# Out-of-sample years 2021/2022/2023 (market sp500x): per-year evaluation of one cell's selected nets (5 seeds x single/double)
# plus the model band for that cell (2026-11-13). args: dyn prod fee run_dir...
set -eo pipefail
cd ${DEEP_HEDGING_REPO:-/data/home/acw796/DeepHedging}
module load cuda/12.4.0-gcc-12.2.0
module load python/3.11.7-gcc-12.2.0
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q V100; then source .venv_v100/bin/activate; else source .venv/bin/activate; fi
export PYTHONUNBUFFERED=1 HEDGING_ENGINE_CACHE=${HEDGING_ENGINE_CACHE:-$PWD/.market_cache}
dyn=$1; prod=$2; fee=$3; shift 3
python -u -m experiments.evaluation.eval_years $dyn $prod $fee "$@"
extra=""; [ "$dyn$prod" = hestondoc ] && extra="--bump 0.05 --mc-paths 10000"
python -u -m experiments.baselines --dynamics $dyn --product $prod --fees $fee --market sp500x --device cuda --out experiments/results/main_years_baselines $extra
echo "YEARS DONE"
