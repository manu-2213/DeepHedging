#!/bin/bash
#SBATCH --job-name=v24_bank_val2000
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=computeshort
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
# Builds the 2,000-path BSM call validation bank once, so the probe-5 GPU jobs do not build it at the same time.
set -eo pipefail
cd /data/home/acw796/DeepHedging/deep_hedging_v0
module load python/3.11.7-gcc-12.2.0
source ../.venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export FAST_HEDGING_CACHE=$PWD/../.fast_hedging_cache
python -u -m fast_hedging.experiment --dynamics bsm --product call --underlying-fee 0.01 --option-fee 0.05 --val-paths 2000 --bank-only --device cpu --out experiments/results/v2_4_checks/bank_val2000
echo "BANK DONE"
