#!/bin/bash
#SBATCH --job-name=mn_calib2223
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
# Heston calibration of the S&P 500 surface for 2022 and 2023 (OptionsDX end-of-day chains converted to the pipeline's
# yearly format by experiments/calibration/convert_spx.py), with the same script and settings as 2017-2021. Years already
# in hedging_engine/data/heston_params_QL_v2_sp500x.json are skipped. (The original job, 2026-11-13, appended the two
# years to the sp500 file, which was then split into the seven-year market sp500x.)
set -eo pipefail
cd ${DEEP_HEDGING_REPO:-/data/home/acw796/DeepHedging}/experiments/calibration
module load python/3.11.7-gcc-12.2.0
source ../../.venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
python -u - <<'PY'
import calibrate_heston as c
c.MARKETS = ["sp500x"]; c.YEARS = ["2017", "2018", "2019", "2020", "2021", "2022", "2023"]
c.main()
PY
python -u fit_bsm.py sp500x 2017 2018 2019 2020 2021 2022 2023
echo "CALIB DONE"
