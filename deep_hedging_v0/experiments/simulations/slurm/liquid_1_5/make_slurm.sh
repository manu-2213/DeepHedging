#!/bin/bash
# Regenerates the per-run slurm files in this folder (one file = one run).
# Fees: underlying 1%, options 5%. Edit HEADER / the loops and rerun to change the grid.
set -euo pipefail
cd "$(dirname "$0")"
REPO=/data/home/acw796/DeepHedging

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
mkdir -p experiments/simulations/slurm/liquid_1_5/logs
EOF
}

FEES="--underlying-fee 0.01 --option-fee 0.05"
for dyn in bsm heston; do
  for prod in call doc con; do
    for arch in rnn; do   # RNN only: LSTM over the last 8 observations (history_len 8)
      for net in single double; do
        name=${dyn}_${prod}_${arch}_${net}
        wall=240:00:00; hours=180   # 180 h tuning cap leaves ~60 h for the 5 final seeds
        { header "$name" "$wall" 32G
          cat <<EOF

# One run: Optuna learning-rate search (validation CVaR95, early stopping with patience),
# then 5 final seeds with the best rates, evaluated on the held-out test year.
# Resumable: resubmitting this same file continues the study and skips finished seeds.
python -u -m fast_hedging.experiment \\
    --dynamics $dyn --product $prod --arch $arch --rnn-mode window --window 8 --net $net \\
    $FEES \\
    --n-trials 30 --tuning-hours $hours --final-seeds 1 2 3 4 5 \\
    --out experiments/results/liquid_1_5/$name
EOF
        } > "$name.slurm"
      done
    done
  done
done

{ header build_banks 06:00:00 32G
  cat <<EOF

# Build the cached market banks (paths + all option prices) once for every
# (dynamics, product). Run this first; the 24 runs then only load .npz files.
for dyn in bsm heston; do
  for prod in call doc con; do
    python -u -m fast_hedging.experiment --dynamics \$dyn --product \$prod \\
        --bank-only --out experiments/results/liquid_1_5/_banks/\${dyn}_\${prod}
  done
done
EOF
} > build_banks.slurm

{ header smoke_test 04:00:00 16G
  cat <<EOF

# 1) parity of the fast engine against the original Gym environments
python -u -m fast_hedging.tests.test_parity
# 2) tiny end-to-end runs: window-8 RNN single + double, the recurrent RNN, and a
#    Heston DOC double net (exercises Heston pricing, the barrier MC and two instruments)
for spec in "bsm call window single" "bsm call window double" "bsm call recurrent single" \
            "heston doc window double"; do
  set -- \$spec
  python -u -m fast_hedging.experiment --dynamics \$1 --product \$2 --arch rnn --rnn-mode \$3 --net \$4 \\
      $FEES --train-paths 200 --val-paths 100 --test-paths 200 \\
      --n-trials 3 --final-seeds 1 --max-iters 30 --min-iters 10 --patience 3 \\
      --out experiments/results/liquid_1_5/_smoke/\$1_\$2_\$3_\$4_\$SLURM_JOB_ID
done
# 3) every saved artefact loads with the analysis helpers
python - <<'PY'
import glob
from fast_hedging import analysis as fa
root = "experiments/results/liquid_1_5/_smoke"
print(fa.load_summaries(root).filter(regex="run|seed|cvar_95").to_string())
print(len(fa.load_trials(root)), "trial rows;", len(fa.load_curves(root, "train")), "train-log rows;",
      len(fa.load_curves(root, "eval", source="trials")), "trial eval rows")
run = sorted(glob.glob(root + "/*_double_*"))[-1]
print({k: v.shape for k, v in fa.load_paths(run, seed=1).items()})
print(sorted(fa.load_baselines(run)))
PY
EOF
} > smoke_test.slurm

{ cat <<'EOF'
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
EOF
} > submit_all.sh
chmod +x submit_all.sh
echo "generated $(ls *.slurm | wc -l) slurm files"
