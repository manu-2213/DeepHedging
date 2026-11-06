#!/bin/bash
# Usage, from this folder: bash submit_all.sh
# Submits the resume checks, then the 18 runs, which start only if the checks pass. Each run resubmits itself
# until its log prints RUN FINISHED. Stop one with: scancel -n <name>
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
check=$(sbatch --parsable check_resume.slurm)
echo "check_resume: $check"
for f in run_*.slurm; do
  echo "$f: $(sbatch --parsable --dependency=afterok:$check "$f")"
done
