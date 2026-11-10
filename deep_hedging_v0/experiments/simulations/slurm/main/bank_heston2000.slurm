#!/bin/bash
#SBATCH --job-name=mn_bank_heston
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:40:00
# Heston banks at the BSM path counts (2,000 / 500 / 2,000) for the main table (2026-11-10). The 1,000-path banks took
# 28-96 s each set on a GPU on 2026-11-02, so this should be a few minutes.
set -eo pipefail
cd /data/home/acw796/DeepHedging/deep_hedging_v0
module load cuda/12.4.0-gcc-12.2.0
module load python/3.11.7-gcc-12.2.0
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q V100; then source ../.venv_v100/bin/activate; else source ../.venv/bin/activate; fi
export PYTHONUNBUFFERED=1
export FAST_HEDGING_CACHE=${FAST_HEDGING_CACHE:-$PWD/../.fast_hedging_cache}
for prod in call doc con; do
  python -u -m fast_hedging.experiment --dynamics heston --product $prod --underlying-fee 0.01 --option-fee 0.05 --train-paths 2000 --val-paths 500 --test-paths 2000 --bank-only --out experiments/results/main_checks/bank_heston_$prod
done
echo "BANK DONE"
