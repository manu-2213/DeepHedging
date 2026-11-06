#!/bin/bash
# Usage, from this folder:
#   bash submit_all.sh pilot    -> resume checks, then the 9 BSM call pilot runs if they pass (start here)
#   bash submit_all.sh first    -> the 3 BSM call main runs (check these before the rest)
#   bash submit_all.sh rest     -> the other 15 main runs
# Each run then resubmits itself until it prints RUN FINISHED. Stop one with: scancel -n <name>
# Banks are not built here: 1-hour jobs reuse the cache already on the HPC (BANK_VERSION 4).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
case "${1:-pilot}" in
  pilot) files=(pilot_*.slurm) ;;
  first) files=(main_bsm_call_*.slurm) ;;
  rest) files=($(ls main_*.slurm | grep -v '^main_bsm_call_')) ;;
  *) echo "unknown group $1"; exit 1 ;;
esac
dep=()
if [ "${1:-pilot}" = pilot ]; then
  check=$(sbatch --parsable check_resume.slurm)
  echo "check_resume: $check"
  dep=(--dependency=afterok:$check)
fi
for f in "${files[@]}"; do
  echo "$f: $(sbatch --parsable ${dep[@]+"${dep[@]}"} "$f")"
done
