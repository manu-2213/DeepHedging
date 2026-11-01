#!/bin/bash
#SBATCH --job-name=smoke_test
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --account=pilot_sae_gpu
#SBATCH --partition=sae
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -eo pipefail
cd ${DEEP_HEDGING_REPO:-/data/home/acw796/DeepHedging}/deep_hedging_v0
module load cuda/12.4.0-gcc-12.2.0
module load python/3.11.7-gcc-12.2.0
source ../.venv/bin/activate
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export FAST_HEDGING_CACHE=${FAST_HEDGING_CACHE:-$PWD/../.fast_hedging_cache}
nvidia-smi --query-gpu=name,memory.total --format=csv || true

SMOKE=experiments/results/v2_1_5/_smoke_$SLURM_JOB_ID
SMALL="--train-paths 200 --val-paths 100 --test-paths 200 --n-trials 3 --tuning-seeds 0 1 --final-seeds 1 2 --max-iters 30 --min-iters 10 --patience 3"
# 1) parity of the fast engine against the original Gym environments
python -u -m fast_hedging.tests.test_parity
# 2) tiny end-to-end runs covering every net, both rewards, the rescaled digital, Heston pricing with the
#    adaptive integration limit (+ model and minimum-variance delta baselines) and the barrier MC
for spec in "bsm call single mean_cvar" "bsm call band mean_cvar" "bsm call double mean_cvar" \
            "bsm call single tracking" "bsm con band mean_cvar" "heston call single mean_cvar" \
            "heston doc double mean_cvar"; do
  set -- $spec
  name=$1_$2_rnn_$3_$4
  python -u -m fast_hedging.experiment --dynamics $1 --product $2 --net $3 --reward $4 \
      --arch rnn --rnn-mode window --window 8 --underlying-fee 0.01 --option-fee 0.05 --digital-payout-frac 0.1 $SMALL --name $name --out $SMOKE/$name
done
# 3) every artefact loads and the report runs end to end
python -m fast_hedging.report $SMOKE --paired --n-boot 200
python - <<PY
from fast_hedging import analysis as fa
root = "$SMOKE"
print(len(fa.load_trials(root)), "trial rows;", len(fa.load_curves(root, "train")), "train-log rows")
p = fa.load_paths(root + "/bsm_call_rnn_band_mean_cvar", seed=1)
print({k: v.shape for k, v in p.items() if k.startswith("trace_")})
print(sorted(fa.load_baselines(root + "/bsm_call_rnn_single_mean_cvar")))
PY
echo "SMOKE TEST FINISHED"
