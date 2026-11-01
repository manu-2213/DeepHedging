#!/bin/bash
# Regenerates the v2 slurm files in this folder. One file = one run (one Optuna study + final seeds).
# Fees: underlying 1%, options 5%.  Results: deep_hedging_v0/experiments/results/v2_1_5/<name>/
#   main     : {bsm, heston} x {call, doc, con} x {single, band, double}, reward mean_cvar   (18 runs)
#   ablation : bsm x {call, doc} x {single, band, double}, reward tracking (v1 objective)   (6 runs)
#   lambda   : bsm x call x {single, band, double}, mean_cvar with GAE lambda = 1.0              (3 runs)
#              (does the terminal CVaR penalty reach early decisions only through the critic?)
set -euo pipefail
cd "$(dirname "$0")"
REPO=/data/home/acw796/DeepHedging
RESULTS=experiments/results/v2_1_5

header() {  # $1 job name, $2 walltime, $3 memory
cat <<EOF
#!/bin/bash
#SBATCH --job-name=$1
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --account=pilot_sae_gpu
#SBATCH --partition=sae
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=$3
#SBATCH --time=$2

set -eo pipefail
cd \${DEEP_HEDGING_REPO:-$REPO}/deep_hedging_v0
module load cuda/12.4.0-gcc-12.2.0
module load python/3.11.7-gcc-12.2.0
source ../.venv/bin/activate
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export FAST_HEDGING_CACHE=\${FAST_HEDGING_CACHE:-\$PWD/../.fast_hedging_cache}
nvidia-smi --query-gpu=name,memory.total --format=csv || true
EOF
}

COMMON="--arch rnn --rnn-mode window --window 8 --underlying-fee 0.01 --option-fee 0.05 --digital-payout-frac 0.1"
TUNE="--n-trials 30 --tuning-hours 150 --tuning-seeds 0 1000 --final-seeds 1 2 3 4 5 6 7 8 9 10"
rm -f main_*.slurm ablation_*.slurm lambda_*.slurm

emit() {  # $1 group(main|ablation|lambda) $2 dynamics $3 product $4 net $5 reward [$6 name suffix] [$7 extra args]
  local name=${2}_${3}_rnn_${4}_${5}${6:-}
  { header "$name" 240:00:00 32G
    cat <<EOF

# One run: Optuna LR search (2 tuning seeds per trial, validation CVaR95, early stopping,
# median pruning), then 10 final seeds at the best LRs, evaluated on the held-out 2021 year.
# Resumable: resubmitting this file continues the study and skips finished seeds.
python -u -m fast_hedging.experiment \\
    --dynamics $2 --product $3 --net $4 --reward $5 ${7:+$7 }\\
    $COMMON \\
    $TUNE \\
    --name $name --out $RESULTS/$name
EOF
  } > "${1}_${name}.slurm"
}

for dyn in bsm heston; do
  for prod in call doc con; do
    for net in single band double; do emit main $dyn $prod $net mean_cvar; done
  done
done
for prod in call doc; do
  for net in single band double; do emit ablation bsm $prod $net tracking; done
done
for net in single band double; do emit lambda bsm call $net mean_cvar _lmbda1 "--lmbda 1.0"; done

{ header build_banks 08:00:00 64G
  cat <<EOF

# Build the cached market banks (paths + all prices + model deltas) once per (dynamics, product),
# then run the martingale / sanity checks on each. Read the WARNING lines in the log.
for dyn in bsm heston; do
  for prod in call doc con; do
    python -u -m fast_hedging.experiment --dynamics \$dyn --product \$prod --digital-payout-frac 0.1 \\
        --bank-only --out $RESULTS/_banks/\${dyn}_\${prod}
  done
done
grep -h "WARNING" $RESULTS/_banks/*/log.txt || echo "no bank warnings"
EOF
} > build_banks.slurm

{ header smoke_test 06:00:00 32G
  cat <<EOF

SMOKE=$RESULTS/_smoke_\$SLURM_JOB_ID
SMALL="--train-paths 200 --val-paths 100 --test-paths 200 --n-trials 3 --tuning-seeds 0 1 --final-seeds 1 2 --max-iters 30 --min-iters 10 --patience 3"
# 1) parity of the fast engine against the original Gym environments
python -u -m fast_hedging.tests.test_parity
# 2) tiny end-to-end runs covering every net, both rewards, the rescaled digital, Heston pricing with the
#    adaptive integration limit (+ model and minimum-variance delta baselines) and the barrier MC
for spec in "bsm call single mean_cvar" "bsm call band mean_cvar" "bsm call double mean_cvar" \\
            "bsm call single tracking" "bsm con band mean_cvar" "heston call single mean_cvar" \\
            "heston doc double mean_cvar"; do
  set -- \$spec
  name=\$1_\$2_rnn_\$3_\$4
  python -u -m fast_hedging.experiment --dynamics \$1 --product \$2 --net \$3 --reward \$4 \\
      $COMMON \$SMALL --name \$name --out \$SMOKE/\$name
done
# 3) every artefact loads and the report runs end to end
python -m fast_hedging.report \$SMOKE --paired --n-boot 200
python - <<PY
from fast_hedging import analysis as fa
root = "\$SMOKE"
print(len(fa.load_trials(root)), "trial rows;", len(fa.load_curves(root, "train")), "train-log rows")
p = fa.load_paths(root + "/bsm_call_rnn_band_mean_cvar", seed=1)
print({k: v.shape for k, v in p.items() if k.startswith("trace_")})
print(sorted(fa.load_baselines(root + "/bsm_call_rnn_single_mean_cvar")))
PY
echo "SMOKE TEST FINISHED"
EOF
} > smoke_test.slurm

cat > submit_all.sh <<'EOF'
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
EOF
echo "generated $(ls *.slurm | wc -l) slurm files"
