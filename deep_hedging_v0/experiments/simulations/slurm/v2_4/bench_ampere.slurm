#!/bin/bash
#SBATCH --job-name=v24_bench_ampere
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=gpushort
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:25:00
set -eo pipefail
cd ${DEEP_HEDGING_REPO:-/data/home/acw796/DeepHedging}/deep_hedging_v0
module load cuda/12.4.0-gcc-12.2.0
module load python/3.11.7-gcc-12.2.0
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q V100; then source ../.venv_v100/bin/activate; else source ../.venv/bin/activate; fi
set -u
python -u -m fast_hedging.tests.bench_transformer
