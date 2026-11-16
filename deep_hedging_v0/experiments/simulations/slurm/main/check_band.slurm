#!/bin/bash
#SBATCH --job-name=mn_check_band
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=computeshort
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
# Smoke test of fast_hedging.model_band on tiny banks (2026-11-11): every cell, CPU. The BSM call must reproduce the
# analytic Whalley-Wilmott band (model_band == ww_band_analytic up to finite-difference error).
set -eo pipefail
cd /data/home/acw796/DeepHedging/deep_hedging_v0
module load python/3.11.7-gcc-12.2.0
source ../.venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export FAST_HEDGING_CACHE=${FAST_HEDGING_CACHE:-$PWD/../.fast_hedging_cache}
for dyn in bsm heston; do for prod in call doc con; do
  python -u -m fast_hedging.model_band --dynamics $dyn --product $prod --fees 1 --val-paths 20 --test-paths 20 --device cpu --out experiments/results/main_checks/band_${dyn}_${prod}
done; done
python - <<'PY'
import json
r = json.load(open("experiments/results/main_checks/band_bsm_call/bsm_call_fee1.json"))["test"]
a, b = r["model_band"]["mean_abs_tracking_error"], r["ww_band_analytic"]["mean_abs_tracking_error"]
print("bsm call: model_band", a, "ww analytic", b)
assert abs(a - b) / b < 0.05, "finite-difference band does not reproduce the analytic Whalley-Wilmott band"
PY
echo "CHECK PASSED"
