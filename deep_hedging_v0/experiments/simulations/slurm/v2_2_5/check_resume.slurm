#!/bin/bash
#SBATCH --job-name=check_resume
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
python -u -m fast_hedging.tests.test_resume
