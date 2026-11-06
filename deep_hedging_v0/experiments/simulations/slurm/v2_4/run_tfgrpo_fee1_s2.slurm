#!/bin/bash
#SBATCH --job-name=v24_tfgrpo_fee1_s2
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00

set -eo pipefail
cd ${DEEP_HEDGING_REPO:-/data/home/acw796/DeepHedging}/deep_hedging_v0
module load cuda/12.4.0-gcc-12.2.0
module load python/3.11.7-gcc-12.2.0
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q V100; then source ../.venv_v100/bin/activate; else source ../.venv/bin/activate; fi
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export FAST_HEDGING_CACHE=${FAST_HEDGING_CACHE:-$PWD/../.fast_hedging_cache}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
python -c "import sys, torch; ok = torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', ok); sys.exit(3) if not ok else print('gpu operation', torch.ones(2, device='cuda').mean().item(), torch.cuda.get_device_name(0))"

OUT=experiments/results/v2_4/tfgrpo_fee1_s2
mkdir -p $OUT
count=$(( $(cat $OUT/job_count 2>/dev/null || echo 0) + 1 ))
echo $count > $OUT/job_count
echo "job $count of this run on $(hostname)"
rc=0
python -u -m fast_hedging.experiment --dynamics bsm --product call --net band --n-trials 1 --no-pruning --final-seeds --actor-lr-range 1e-4 1e-4 --inactor-lr-range 1e-3 1e-3 --tuning-seeds 2 --arch rnn --rnn-mode window --paths-per-iter 100 --ppo-epochs 1 --minibatch 125000 --target-kl 0 --reward tracking --reward-norm none --no-obs-norm --no-adv-norm --value-loss huber --value-coef 1.0 --window 10 --pad-mode zero --no-agent-state --call-action-low -1 --underlying-fee 0.01 --option-fee 0.05 --init-std 0.2 --max-iters 1000000000 --patience 200 --lr-patience 50 --lr-decays 3 --lr-decay-factor 0.5 --time-budget-min 50 --arch transformer --advantage grpo --grpo-group 4 --paths-per-iter 25 --name tfgrpo_fee1_s2 --out $OUT || rc=$?
if [ "$rc" -eq 75 ]; then
  if [ "$count" -ge 300 ]; then echo "CHAIN STOPPED: 300 jobs reached"; exit 1; fi
  cd "$SLURM_SUBMIT_DIR" && sbatch "run_tfgrpo_fee1_s2.slurm"
elif [ "$rc" -eq 0 ]; then
  echo "RUN FINISHED after $count jobs"
else
  echo "RUN FAILED with exit code $rc"; exit "$rc"
fi
