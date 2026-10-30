#!/bin/bash
#SBATCH --job-name=smoke_test
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --account=pilot_sae_gpu
#SBATCH --partition=sae
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

set -eo pipefail
cd ${DEEP_HEDGING_REPO:-/data/home/acw796/DeepHedging}/deep_hedging_v0
module load cuda/12.4.0-gcc-12.2.0
module load python/3.11.7-gcc-12.2.0
source ../.venv/bin/activate
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export FAST_HEDGING_CACHE=${FAST_HEDGING_CACHE:-$PWD/../.fast_hedging_cache}
mkdir -p experiments/simulations/slurm/liquid_1_5/logs

# 1) parity of the fast engine against the original Gym environments
python -u -m fast_hedging.tests.test_parity
# 2) tiny end-to-end runs: recurrent RNN single + double, old sliding-window RNN, and a
#    Heston DOC double net (exercises Heston pricing, the barrier MC and two instruments)
for spec in "bsm call recurrent single" "bsm call recurrent double" "bsm call window single"             "heston doc recurrent double"; do
  set -- $spec
  python -u -m fast_hedging.experiment --dynamics $1 --product $2 --arch rnn --rnn-mode $3 --net $4 \
      --underlying-fee 0.01 --option-fee 0.05 --train-paths 200 --val-paths 100 --test-paths 200 \
      --n-trials 3 --final-seeds 1 --max-iters 30 --min-iters 10 --patience 3 \
      --out experiments/results/liquid_1_5/_smoke/$1_$2_$3_$4_$SLURM_JOB_ID
done
