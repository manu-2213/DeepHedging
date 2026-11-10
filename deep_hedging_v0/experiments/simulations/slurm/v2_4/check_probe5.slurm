#!/bin/bash
#SBATCH --job-name=v24_check_p5
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=computeshort
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
# Checks the probe-5 path (2026-11-09): single net with agent state + forced holds (--actor-hold-dropout), GRPO group 8,
# chained resume; then a double net from it with --inactor-init-hold-prob 0.05. Machinery only, tiny sizes.
set -eo pipefail
cd /data/home/acw796/DeepHedging/deep_hedging_v0
module load python/3.11.7-gcc-12.2.0
source ../.venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export FAST_HEDGING_CACHE=$PWD/../.fast_hedging_cache
C="--dynamics bsm --product call --n-trials 1 --no-pruning --final-seeds --actor-lr-range 1e-4 1e-4 --tuning-seeds 1 --arch tfpath --rnn-mode window --ppo-epochs 1 --minibatch 25000 --target-kl 0 --reward tracking --reward-norm none --no-obs-norm --no-adv-norm --value-loss huber --value-coef 1.0 --window 10 --pad-mode zero --call-action-low -1 --underlying-fee 0.01 --option-fee 0.05 --init-std 0.2 --train-paths 200 --val-paths 100 --test-paths 200 --max-iters 40 --min-iters 10 --patience 200 --lr-patience 50 --lr-decays 3 --lr-decay-factor 0.5 --time-budget-min 1 --metric mean_abs_tracking_error --device cpu --advantage grpo --grpo-group 8 --paths-per-iter 3"
S=experiments/results/v2_4_checks/p5_single; D=experiments/results/v2_4_checks/p5_double
rm -rf $S $D; mkdir -p $S $D
for k in 1 2 3 4 5 6 7 8; do
  rc=0
  python -u -m fast_hedging.experiment $C --net single --inactor-lr-range 1e-3 1e-3 --actor-hold-dropout 0.95 --name p5_single --out $S || rc=$?
  echo "single pass $k exit $rc"; [ "$rc" -eq 75 ] || break
done
[ "$rc" -eq 0 ] && [ -f $S/summary.json ] || { echo "CHECK FAILED single rc=$rc"; exit 1; }
for k in 1 2 3 4 5 6 7 8; do
  rc=0
  python -u -m fast_hedging.experiment $C --net double --inactor-lr-range 1e-4 1e-4 --joint-lr-range 1e-4 1e-4 --actor-init from-single --single-dir $S --inactor-init-hold-prob 0.05 --name p5_double --out $D || rc=$?
  echo "double pass $k exit $rc"; [ "$rc" -eq 75 ] || break
done
[ "$rc" -eq 0 ] && [ -f $D/summary.json ] || { echo "CHECK FAILED double rc=$rc"; exit 1; }
python - <<'PY'
import json, glob, re
hr = [float(x) for f in glob.glob('experiments/results/v2_4_checks/p5_single/**/*.json', recursive=True) for x in re.findall(r'"train_hold_rate": ([0-9.]+)', open(f).read())]
print("single train hold rates n=%d mean=%.3f" % (len(hr), sum(hr) / max(1, len(hr))))
first = [l for l in open('experiments/results/v2_4_checks/p5_double/log.txt') if '[joint] it' in l and ' val ' in l][0]
print("double first check:", first.strip()[:150])
assert hr and 0.3 < sum(hr) / len(hr) < 0.65, "forced holds not active"
assert float(re.search(r'hold=([0-9.]+)', first).group(1)) < 0.2, "inactor init hold prob not applied"
PY
echo "CHECK PASSED"
