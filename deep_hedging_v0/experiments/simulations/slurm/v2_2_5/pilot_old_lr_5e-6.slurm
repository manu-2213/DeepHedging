#!/bin/bash
#SBATCH --job-name=pilot_old_lr_5e-6
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=gpushort
#SBATCH --constraint=ampere|hopper
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=11G
#SBATCH --time=01:00:00

set -eo pipefail
cd ${DEEP_HEDGING_REPO:-/data/home/acw796/DeepHedging}/deep_hedging_v0
module load cuda/12.4.0-gcc-12.2.0
module load python/3.11.7-gcc-12.2.0
source ../.venv/bin/activate
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export FAST_HEDGING_CACHE=${FAST_HEDGING_CACHE:-$PWD/../.fast_hedging_cache}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
# PyTorch falls back to the CPU without an error when it cannot use the GPU, and may see a GPU it has no code for:
# run a real operation there and fail now instead of crawling or crashing later
python -c "import sys, torch; ok = torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', ok); sys.exit(3) if not ok else print('gpu operation', torch.ones(2, device='cuda').mean().item(), torch.cuda.get_device_name(0))"

OUT=experiments/results/v2_2_5/pilot_old_lr_5e-6
mkdir -p $OUT
count=$(( $(cat $OUT/job_count 2>/dev/null || echo 0) + 1 ))
echo $count > $OUT/job_count
echo "job $count of this run"

rc=0
python -u -m fast_hedging.experiment --dynamics bsm --product call --net single --reward mean_cvar --n-trials 1 --no-pruning --tuning-seeds 1 --final-seeds --window 8 --actor-lr-range 5e-6 5e-6 --ppo-epochs 1 --minibatch 80000 --value-coef 1.0 --window 10 \
    --arch rnn --rnn-mode window --underlying-fee 0.01 --option-fee 0.05 --digital-payout-frac 0.1 --time-budget-min 50 --max-iters 1000000000 --patience 200 \
    --name pilot_old_lr_5e-6 --out $OUT || rc=$?

# 75: time budget used and state saved, so continue in a new job. 0: finished. Anything else: a real failure.
if [ "$rc" -eq 75 ]; then
  if [ "$count" -ge 1000 ]; then echo "CHAIN STOPPED: 1000 jobs reached"; exit 1; fi
  cd "$SLURM_SUBMIT_DIR" && sbatch "pilot_old_lr_5e-6.slurm"
elif [ "$rc" -eq 0 ]; then
  echo "RUN FINISHED after $count jobs"
else
  echo "RUN FAILED with exit code $rc"; exit "$rc"
fi
