#!/bin/bash
#SBATCH --job-name=probe_pack
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=gpushort
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00

set -eo pipefail
cd ${DEEP_HEDGING_REPO:-/data/home/acw796/DeepHedging}/deep_hedging_v0
module load cuda/12.4.0-gcc-12.2.0
module load python/3.11.7-gcc-12.2.0
source ../.venv_v100/bin/activate
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export FAST_HEDGING_CACHE=${FAST_HEDGING_CACHE:-$PWD/../.fast_hedging_cache}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
# PyTorch falls back to the CPU without an error when it cannot use the GPU, and may see a GPU it has no code for:
# run a real operation there and fail now instead of crawling or crashing later
python -c "import torch; print(torch.cuda.get_arch_list())"; python -c "import sys, torch; ok = torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', ok); sys.exit(3) if not ok else print('gpu operation', torch.ones(2, device='cuda').mean().item(), torch.cuda.get_device_name(0))"
export OMP_NUM_THREADS=1
mkdir -p experiments/results/v2_2_5/probes
while read -r name args; do
  mkdir -p experiments/results/v2_2_5/probes/$name
  python -u -m fast_hedging.experiment --dynamics bsm --product call --net single --n-trials 1 --no-pruning \
      --tuning-seeds 1 --final-seeds --max-iters 800 --patience 1000000 --time-budget-min 52 $args \
      --name $name --out experiments/results/v2_2_5/probes/$name > experiments/results/v2_2_5/probes/$name/console.txt 2>&1 &
done < "$1"
wait
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv || true
