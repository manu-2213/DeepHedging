#!/bin/bash
# Usage, from this folder: bash submit_fast.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
ids=$(squeue -u "$USER" -h -t PD -o "%i %j %r" | awk '$2 ~ /^v24_/ && $2 !~ /bench/ && $3 !~ /JobHeldUser/ {print $1}' | paste -sd,)
if [ -n "$ids" ]; then scontrol hold "$ids"; echo "$ids" > held_for_fast.txt; echo "held: $ids"; fi
check=$(sbatch --parsable check_fast.slurm); echo "check_fast: $check"
while read -r run; do
  echo "$run: $(sbatch --parsable --dependency=afterok:$check run_$run.slurm)"
done < fast_order.txt
