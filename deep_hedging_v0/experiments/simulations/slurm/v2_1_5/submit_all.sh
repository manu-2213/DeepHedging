#!/bin/bash
# Usage (from this folder, after smoke_test.slurm printed SMOKE TEST FINISHED with ALL PASSED):
#   bash submit_all.sh            -> bank build, then the 18 main runs
#   bash submit_all.sh ablation   -> bank build, then the 6 tracking-reward ablations
#   bash submit_all.sh lambda     -> bank build, then the 3 GAE-lambda-1 ablations
#   bash submit_all.sh all        -> everything
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
case "${1:-main}" in
  main) files=(main_*.slurm) ;;
  ablation) files=(ablation_*.slurm) ;;
  lambda) files=(lambda_*.slurm) ;;
  all) files=(main_*.slurm ablation_*.slurm lambda_*.slurm) ;;
  *) echo "unknown group $1"; exit 1 ;;
esac
banks=$(sbatch --parsable build_banks.slurm)
echo "bank build: $banks"
for f in "${files[@]}"; do
  echo "$f: $(sbatch --parsable --dependency=afterok:$banks "$f")"
done
