#!/bin/bash
#SBATCH --job-name=mn_band_paths
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=gpushort,sae
#SBATCH --account=pilot_sae_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:30:00
# Per-path arrays of the model band / target for the paper figures (2026-11-12); greeks are cached.
set -eo pipefail
cd ${DEEP_HEDGING_REPO:-/data/home/acw796/DeepHedging}
module load cuda/12.4.0-gcc-12.2.0
module load python/3.11.7-gcc-12.2.0
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q V100; then source .venv_v100/bin/activate; else source .venv/bin/activate; fi
export PYTHONUNBUFFERED=1 HEDGING_ENGINE_CACHE=${HEDGING_ENGINE_CACHE:-$PWD/.market_cache}
PYTHONPATH=$PWD python -u -m experiments.evaluation.band_paths
