#!/bin/bash
#SBATCH --job-name=mn_check
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=computeshort
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
# Machinery check for the main table (2026-11-10): for every (dynamics, product) other than BSM call, a tiny single net
# (tfpath + GRPO group 8, chained resume) and then a tiny double net from it (from-single). CPU, tiny banks.
set -eo pipefail
cd /data/home/acw796/DeepHedging/deep_hedging_v0
module load python/3.11.7-gcc-12.2.0
source ../.venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
C="--n-trials 1 --no-pruning --final-seeds --actor-lr-range 1e-4 1e-4 --tuning-seeds 1 --arch tfpath --rnn-mode window --ppo-epochs 1 --minibatch 25000 --target-kl 0 --reward tracking --reward-norm none --no-obs-norm --no-adv-norm --value-loss huber --value-coef 1.0 --window 10 --pad-mode zero --no-agent-state --call-action-low -1 --underlying-fee 0.01 --option-fee 0.05 --init-std 0.2 --train-paths 200 --val-paths 100 --test-paths 200 --max-iters 40 --min-iters 10 --patience 200 --lr-patience 50 --lr-decays 3 --lr-decay-factor 0.5 --time-budget-min 1 --metric mean_abs_tracking_error --device cpu --advantage grpo --grpo-group 8 --paths-per-iter 3"
for dp in "bsm doc" "bsm con" "heston call" "heston doc" "heston con"; do
  set -- $dp; dyn=$1; prod=$2
  S=experiments/results/main_checks/${dyn}_${prod}_single; D=experiments/results/main_checks/${dyn}_${prod}_double
  rm -rf $S $D; mkdir -p $S $D
  for k in 1 2 3 4 5 6 7 8; do
    rc=0
    python -u -m fast_hedging.experiment --dynamics $dyn --product $prod $C --net single --inactor-lr-range 1e-3 1e-3 --name ${dyn}_${prod}_single --out $S || rc=$?
    echo "$dyn $prod single pass $k exit $rc"; [ "$rc" -eq 75 ] || break
  done
  [ "$rc" -eq 0 ] && [ -f $S/summary.json ] || { echo "CHECK FAILED $dyn $prod single rc=$rc"; exit 1; }
  for k in 1 2 3 4 5 6 7 8; do
    rc=0
    python -u -m fast_hedging.experiment --dynamics $dyn --product $prod $C --net double --inactor-lr-range 1e-3 1e-3 --joint-lr-range 1e-4 1e-4 --actor-init from-single --single-dir $S --name ${dyn}_${prod}_double --out $D || rc=$?
    echo "$dyn $prod double pass $k exit $rc"; [ "$rc" -eq 75 ] || break
  done
  [ "$rc" -eq 0 ] && [ -f $D/summary.json ] || { echo "CHECK FAILED $dyn $prod double rc=$rc"; exit 1; }
  grep -c "from-single" $D/protocol.json >/dev/null || { echo "CHECK FAILED $dyn $prod: double did not load the single actor"; exit 1; }
  python - "$D" <<'PY'
import json, sys
s = json.load(open(sys.argv[1] + "/summary.json"))
d = s["best_trial_test"]["deterministic"]; b = s["baselines_test"]
print(sys.argv[1], "test track=%.2f cost=%.2f hold=%.2f" % (d["mean_abs_tracking_error"], d["mean_cost"], d["hold_rate"]), "baselines:", sorted(b))
PY
done
echo "CHECK PASSED"
