#!/bin/bash
#SBATCH --job-name=v24_arch_fee01_s2
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=gpushort
#SBATCH --constraint=80G
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
export OMP_NUM_THREADS=2
export FAST_HEDGING_CACHE=${FAST_HEDGING_CACHE:-$PWD/../.fast_hedging_cache}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
python -c "import sys, torch; ok = torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', ok); sys.exit(3) if not ok else print('gpu operation', torch.ones(2, device='cuda').mean().item(), torch.cuda.get_device_name(0))"
while read -r probe args; do
  [ -z "$probe" ] && continue
  OUT=experiments/results/v2_4/$probe; mkdir -p $OUT
  python -u -m fast_hedging.experiment $args --name $probe --out $OUT > $OUT/console.txt 2>&1 &
done < "$SLURM_SUBMIT_DIR/pack_arch_fee01_s2.txt"
sleep 600; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv || true
wait || true
