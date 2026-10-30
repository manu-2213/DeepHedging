#!/bin/bash
# Submit the whole grid with dependencies:  bank -> single (rnn) -> double (rnn).
# Each run is its own slurm job. Usage (from this folder):
#   bash submit_all.sh                       # all 6 products (rnn only)
#   bash submit_all.sh bsm_call heston_doc   # only some <dynamics>_<product>
set -euo pipefail
cd "$(dirname "$0")"
targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then
    targets=(bsm_call bsm_doc bsm_con heston_call heston_doc heston_con)
fi
for target in "${targets[@]}"; do
    bank=$(sbatch --parsable "bank_${target}.slurm")
    echo "bank_${target}: $bank"
    for arch in rnn; do
        single=$(sbatch --parsable --dependency=afterok:$bank "${target}_${arch}_single.slurm")
        double=$(sbatch --parsable --dependency=afterok:$single "${target}_${arch}_double.slurm")
        echo "  ${target}_${arch}: single $single -> double $double"
    done
done
