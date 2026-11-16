#!/bin/bash
#SBATCH --job-name=mn_baselines
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=gpushort,sae
#SBATCH --account=pilot_sae_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
# the Heston barrier (nested MC, 5 bumps, 10,000 inner paths) may need a second submission: greeks are cached per split
# Model-based reference strategies (delta / delta-gamma target with a Whalley-Wilmott-style band) for every cell of the
# main table, at both fees (2026-11-11). Greeks are cached next to the banks, so a rerun after the time limit resumes.
set -eo pipefail
cd ${DEEP_HEDGING_REPO:-/data/home/acw796/DeepHedging}/deep_hedging_v0
module load cuda/12.4.0-gcc-12.2.0
module load python/3.11.7-gcc-12.2.0
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q V100; then source ../.venv_v100/bin/activate; else source ../.venv/bin/activate; fi
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export FAST_HEDGING_CACHE=${FAST_HEDGING_CACHE:-$PWD/../.fast_hedging_cache}
for cell in "bsm call" "bsm doc" "bsm con" "heston call" "heston con" "heston doc --bump 0.05 --mc-paths 10000"; do
  set -- $cell; dyn=$1; prod=$2; shift 2
  if [ -f experiments/results/main_baselines/${dyn}_${prod}_fee1.json ] && [ -f experiments/results/main_baselines/${dyn}_${prod}_fee01.json ]; then echo "skip $dyn $prod (done)"; continue; fi
  python -u -m fast_hedging.model_band --dynamics $dyn --product $prod --fees 01 1 --device cuda "$@"
done
echo "BASELINES DONE"
