#!/bin/bash
# v2.3: BSM call trained to convergence. 3 nets x 2 fees x 3 seeds = 18 runs, one seed per run.
# Old-equivalent setup (see v2_2_5/probes.sh): it is the configuration that learns in the probes.
# Each run trains until 2000 iterations pass without improvement; before that it halves the learning rates (and restores
# the best weights) after each 500 iterations without improvement, three times at most. Each phase of the double net
# does this separately. Each Slurm job works for 50 minutes, saves its state and resubmits itself.
# Results: deep_hedging_v0/experiments/results/v2_3/<name>/   (test metrics: summary.json -> best_trial_test)
# Partition: gpushort with the default account (sae queued about 6 months on 2026-11-05). Override with PARTITION=... ACCOUNT=...
set -euo pipefail
cd "$(dirname "$0")"
REPO=/data/home/acw796/DeepHedging
RESULTS=experiments/results/v2_3
PARTITION=${PARTITION-gpushort}
ACCOUNT=${ACCOUNT-}
MAX_JOBS=300

OLD="--arch rnn --rnn-mode window --paths-per-iter 100 --ppo-epochs 1 --minibatch 125000 --target-kl 0 --reward tracking --reward-norm none --no-obs-norm --no-adv-norm --value-loss huber --value-coef 1.0 --window 10 --pad-mode zero --no-agent-state --call-action-low -1"
RUN="--dynamics bsm --product call --n-trials 1 --no-pruning --final-seeds --actor-lr-range 1e-4 1e-4 --inactor-lr-range 1e-3 1e-3"
STOP="--max-iters 1000000000 --patience 200 --lr-patience 50 --lr-decays 3 --lr-decay-factor 0.5 --time-budget-min 50"
FEE01="--underlying-fee 0.001 --option-fee 0.002"
FEE1="--underlying-fee 0.01 --option-fee 0.05 --init-std 0.2"     # low exploration noise: full noise blocks learning at 1%
rm -f run_*.slurm check_resume.slurm

setup() {  # $1 job name
  cat <<EOF
#!/bin/bash
#SBATCH --job-name=$1
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
${ACCOUNT:+#SBATCH --account=$ACCOUNT
}#SBATCH --partition=$PARTITION
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00

set -eo pipefail
cd \${DEEP_HEDGING_REPO:-$REPO}/deep_hedging_v0
module load cuda/12.4.0-gcc-12.2.0
module load python/3.11.7-gcc-12.2.0
# V100 cards need the CUDA 12.6 build (.venv_v100); every other card uses the main environment
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q V100; then source ../.venv_v100/bin/activate; else source ../.venv/bin/activate; fi
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export FAST_HEDGING_CACHE=\${FAST_HEDGING_CACHE:-\$PWD/../.fast_hedging_cache}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
python -c "import sys, torch; ok = torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', ok); sys.exit(3) if not ok else print('gpu operation', torch.ones(2, device='cuda').mean().item(), torch.cuda.get_device_name(0))"
EOF
}

emit() {  # $1 run name, $2 experiment arguments
  { setup "$1"
    cat <<EOF

OUT=$RESULTS/$1
mkdir -p \$OUT
count=\$(( \$(cat \$OUT/job_count 2>/dev/null || echo 0) + 1 ))
echo \$count > \$OUT/job_count
echo "job \$count of this run on \$(hostname)"

rc=0
python -u -m fast_hedging.experiment $2 \\
    --name $1 --out \$OUT || rc=\$?

# 75: time budget used and state saved, so continue in a new job. 0: finished. Anything else: a real failure.
if [ "\$rc" -eq 75 ]; then
  if [ "\$count" -ge $MAX_JOBS ]; then echo "CHAIN STOPPED: $MAX_JOBS jobs reached"; exit 1; fi
  cd "\$SLURM_SUBMIT_DIR" && sbatch "run_$1.slurm"
elif [ "\$rc" -eq 0 ]; then
  echo "RUN FINISHED after \$count jobs"
else
  echo "RUN FAILED with exit code \$rc"; exit "\$rc"
fi
EOF
  } > "run_$1.slurm"
}

for net in single band double; do
  for fee in 01 1; do
    F=FEE$fee
    for s in 1 2 3; do
      emit "bsm_call_${net}_fee${fee}_s$s" "$RUN --net $net --tuning-seeds $s $OLD ${!F} $STOP"
    done
  done
done

{ setup check_resume
  echo "python -u -m fast_hedging.tests.test_resume"
} > check_resume.slurm

cat > submit_all.sh <<'EOF'
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
EOF
echo "generated $(ls run_*.slurm | wc -l) run files"
