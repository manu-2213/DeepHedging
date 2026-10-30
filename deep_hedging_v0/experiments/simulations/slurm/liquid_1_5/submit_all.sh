#!/bin/bash
# Run from this folder after smoke_test.slurm passed: submits the bank build, then the
# 12 RNN runs (one job each) that start once the banks exist.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
banks=$(sbatch --parsable build_banks.slurm)
for f in bsm_*.slurm heston_*.slurm; do
  sbatch --dependency=afterok:$banks "$f"
done
