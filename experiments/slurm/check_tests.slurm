#!/bin/bash
#SBATCH --job-name=mn_check_tests
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=sae
#SBATCH --account=pilot_sae_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
# The test suite on a GPU node: pricing checks, exact resume (a run split over several jobs equals the run done in one
# go) and the transformer / GRPO checks. Run it after pushing new engine code and before launching runs.
# Each of the three must print ALL PASSED.
set -eo pipefail
cd ${DEEP_HEDGING_REPO:-/data/home/acw796/DeepHedging}
module load cuda/12.4.0-gcc-12.2.0
module load python/3.11.7-gcc-12.2.0
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q V100; then source .venv_v100/bin/activate; else source .venv/bin/activate; fi
set -u
export PYTHONUNBUFFERED=1
export HEDGING_ENGINE_CACHE=${HEDGING_ENGINE_CACHE:-$PWD/.market_cache}
python -u -m tests.test_pricing
python -u -m tests.test_resume
python -u -m tests.test_new_modes
echo "TESTS DONE"
