#!/bin/bash
# Regenerates the chained 1-hour slurm files in this folder. One file = one run.
# Each job works for 50 minutes on gpushort, saves its state and resubmits the same file until the run is finished.
# Results: deep_hedging_v0/experiments/results/v2_2_5/<name>/
# Stopping, everywhere: no iteration cap, patience 2000 iterations = 200 validation checks. That is 5 epochs of
# the old pipeline (1 epoch = 400 episodes, 1 episode = 1 rollout = 1 iteration here).
#   pilot : BSM call, single net, one seed per setting: which learning rate lets the LSTM learn when it may train
#           this long? The old pipeline used 5e-6 to 2e-5 with 4 gradient steps per rollout; this engine makes about 76.
#   main  : {bsm, heston} x {call, doc, con} x {single, band, double}, reward mean_cvar (18 runs), fixed rates
# Fill in other partitions or an account with:  PARTITION=sae ACCOUNT=pilot_sae_gpu bash make_slurm.sh
set -euo pipefail
cd "$(dirname "$0")"
REPO=/data/home/acw796/DeepHedging
RESULTS=experiments/results/v2_2_5
PARTITION=${PARTITION:-gpushort}
ACCOUNT=${ACCOUNT:-}
CONSTRAINT=${CONSTRAINT-ampere|hopper}   # A100, H100/H200. The installed PyTorch has no code for the V100 on gpushort
BUDGET_MIN=50          # of the 60 allowed; the rest covers the final test evaluation and saving
MAX_JOBS=1000          # a run that resubmits this often has a problem: stop it

COMMON="--arch rnn --rnn-mode window --underlying-fee 0.01 --option-fee 0.05 --digital-payout-frac 0.1 --time-budget-min $BUDGET_MIN"
STOP="--max-iters 1000000000 --patience 200"
# Main runs: no tuning. Actor and joint rate 1e-5, inaction net 1e-3 (the old default). Seed 0 is the single "trial".
# More seeds later: FINAL_SEEDS="1 2 3 4 5 6 7 8 9 10" bash make_slurm.sh, then resubmit. Finished seeds are skipped.
FINAL_SEEDS=${FINAL_SEEDS:-1 2 3}
TUNE="--window 8 --actor-lr-range 1e-5 1e-5 --inactor-lr-range 1e-3 1e-3 --no-pruning --n-trials 1 --tuning-seeds 0 --final-seeds $FINAL_SEEDS"
PILOT="--dynamics bsm --product call --net single --reward mean_cvar --n-trials 1 --no-pruning --tuning-seeds 1 --final-seeds"
PILOT="$PILOT --window 8"
rm -f main_*.slurm pilot_*.slurm check_resume.slurm

setup() {  # $1 job name
  cat <<EOF
#!/bin/bash
#SBATCH --job-name=$1
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
${ACCOUNT:+#SBATCH --account=$ACCOUNT
}#SBATCH --partition=$PARTITION
${CONSTRAINT:+#SBATCH --constraint="$CONSTRAINT"
}#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=11G
#SBATCH --time=01:00:00

set -eo pipefail
cd \${DEEP_HEDGING_REPO:-$REPO}/deep_hedging_v0
module load cuda/12.4.0-gcc-12.2.0
module load python/3.11.7-gcc-12.2.0
source ../.venv/bin/activate
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export FAST_HEDGING_CACHE=\${FAST_HEDGING_CACHE:-\$PWD/../.fast_hedging_cache}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
# PyTorch falls back to the CPU without an error when it cannot use the GPU, and may see a GPU it has no code for:
# run a real operation there and fail now instead of crawling or crashing later
python -c "import sys, torch; ok = torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', ok); sys.exit(3) if not ok else print('gpu operation', torch.ones(2, device='cuda').mean().item(), torch.cuda.get_device_name(0))"
EOF
}

emit() {  # $1 slurm file stem, $2 run name, $3 experiment arguments
  { setup "$2"
    cat <<EOF

OUT=$RESULTS/$2
mkdir -p \$OUT
count=\$(( \$(cat \$OUT/job_count 2>/dev/null || echo 0) + 1 ))
echo \$count > \$OUT/job_count
echo "job \$count of this run"

rc=0
python -u -m fast_hedging.experiment $3 \\
    $COMMON $STOP \\
    --name $2 --out \$OUT || rc=\$?

# 75: time budget used and state saved, so continue in a new job. 0: finished. Anything else: a real failure.
if [ "\$rc" -eq 75 ]; then
  if [ "\$count" -ge $MAX_JOBS ]; then echo "CHAIN STOPPED: $MAX_JOBS jobs reached"; exit 1; fi
  cd "\$SLURM_SUBMIT_DIR" && sbatch "$1.slurm"
elif [ "\$rc" -eq 0 ]; then
  echo "RUN FINISHED after \$count jobs"
else
  echo "RUN FAILED with exit code \$rc"; exit "\$rc"
fi
EOF
  } > "$1.slurm"
}

# Short learning-rate probes sharing one GPU: sbatch -J NAME probe_pack.slurm FILE. Each line of FILE is: name <arguments>
rm -f probe.slurm
{ setup probe_pack
  cat <<EOF
export OMP_NUM_THREADS=1
mkdir -p $RESULTS/probes
while read -r name args; do
  mkdir -p $RESULTS/probes/\$name
  python -u -m fast_hedging.experiment --dynamics bsm --product call --net single --n-trials 1 --no-pruning \\
      --tuning-seeds 1 --final-seeds --max-iters 800 --patience 1000000 --time-budget-min 52 \$args \\
      --name \$name --out $RESULTS/probes/\$name > $RESULTS/probes/\$name/console.txt 2>&1 &
done < "\$1"
wait
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv || true
EOF
} > probe_pack.slurm

# Resume checks on a real GPU; the pilot starts only if they pass.
{ setup check_resume
  echo "python -u -m fast_hedging.tests.test_resume"
} > check_resume.slurm

# Pilot. lr LR: fixes the learning rate. OLD: 4 gradient steps per rollout, critic weight 1, window 10, as the old pipeline.
lr() { echo "--actor-lr-range $1 $1"; }
OLD="--ppo-epochs 1 --minibatch 80000 --value-coef 1.0 --window 10"
emit pilot_lr_1e-6        pilot_lr_1e-6        "$PILOT $(lr 1e-6)"
emit pilot_lr_3e-6        pilot_lr_3e-6        "$PILOT $(lr 3e-6)"
emit pilot_lr_1e-5        pilot_lr_1e-5        "$PILOT $(lr 1e-5)"
emit pilot_lr_3e-5        pilot_lr_3e-5        "$PILOT $(lr 3e-5)"
emit pilot_lr_1e-4        pilot_lr_1e-4        "$PILOT $(lr 1e-4)"
emit pilot_control_v2_1   pilot_control_v2_1   "$PILOT $(lr 5.85e-4)"     # the rate v2.1 selected, as a control
emit pilot_noise_lr_1e-5  pilot_noise_lr_1e-5  "$PILOT $(lr 1e-5) --init-std 0.2"
emit pilot_old_lr_1e-5    pilot_old_lr_1e-5    "$PILOT $(lr 1e-5) $OLD"
emit pilot_old_lr_5e-6    pilot_old_lr_5e-6    "$PILOT $(lr 5e-6) $OLD"   # the old script's default rate

for dyn in bsm heston; do
  for prod in call doc con; do
    for net in single band double; do
      name=${dyn}_${prod}_rnn_${net}_mean_cvar
      emit "main_$name" "$name" "--dynamics $dyn --product $prod --net $net --reward mean_cvar $TUNE"
    done
  done
done

cat > submit_all.sh <<'EOF'
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
EOF
echo "generated $(ls *.slurm | wc -l) slurm files"
