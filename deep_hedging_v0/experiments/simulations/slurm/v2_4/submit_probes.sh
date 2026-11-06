#!/bin/bash
# Usage, from this folder: bash submit_probes.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
check=$(sbatch --parsable check_new_modes.slurm)
echo "check_new_modes: $check"
for f in pack_arch_*.slurm pack_scale_*.slurm; do
  echo "$f: $(sbatch --parsable --dependency=afterok:$check "$f")"
done
