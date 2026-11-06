#!/bin/bash
# v2.4 convergence: the 12 transformer / GRPO probes (pack_arch_*.txt) continued from their saved state until
# they stop improving, one run per GPU, as chained 50-minute jobs (exit 75 -> the script resubmits itself).
# Same arguments as the probe, so the run resumes in the same folder: experiments/results/v2_4/<probe>/.
# Usage: bash make_runs.sh && bash submit_runs.sh
set -euo pipefail
cd "$(dirname "$0")"
REPO=/data/home/acw796/DeepHedging
RESULTS=experiments/results/v2_4
MAX_JOBS=300
rm -f run_*.slurm submit_runs.sh
# GRPO at 1% and the transformer at 0.1% first (the clearest gains in the probes)
ORDER="grpo_fee1 tf_fee01 tfgrpo_fee01 tf_fee1 grpo_fee01 tfgrpo_fee1"
for key in $ORDER; do
  for s in 1 2; do
    probe=${key}_s$s
    args=$(grep -h "^$probe " pack_arch_*.txt | cut -d' ' -f2- | sed 's/--time-budget-min 52/--time-budget-min 50/')
    [ -n "$args" ] || { echo "no arguments for $probe"; exit 1; }
    cat > "run_$probe.slurm" <<EOS
#!/bin/bash
#SBATCH --job-name=v24_$probe
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00

set -eo pipefail
cd \${DEEP_HEDGING_REPO:-$REPO}/deep_hedging_v0
module load cuda/12.4.0-gcc-12.2.0
module load python/3.11.7-gcc-12.2.0
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q V100; then source ../.venv_v100/bin/activate; else source ../.venv/bin/activate; fi
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export FAST_HEDGING_CACHE=\${FAST_HEDGING_CACHE:-\$PWD/../.fast_hedging_cache}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
python -c "import sys, torch; ok = torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', ok); sys.exit(3) if not ok else print('gpu operation', torch.ones(2, device='cuda').mean().item(), torch.cuda.get_device_name(0))"

OUT=$RESULTS/$probe
mkdir -p \$OUT
count=\$(( \$(cat \$OUT/job_count 2>/dev/null || echo 0) + 1 ))
echo \$count > \$OUT/job_count
echo "job \$count of this run on \$(hostname)"
rc=0
python -u -m fast_hedging.experiment $args --name $probe --out \$OUT || rc=\$?
if [ "\$rc" -eq 75 ]; then
  if [ "\$count" -ge $MAX_JOBS ]; then echo "CHAIN STOPPED: $MAX_JOBS jobs reached"; exit 1; fi
  cd "\$SLURM_SUBMIT_DIR" && sbatch "run_$probe.slurm"
elif [ "\$rc" -eq 0 ]; then
  echo "RUN FINISHED after \$count jobs"
else
  echo "RUN FAILED with exit code \$rc"; exit "\$rc"
fi
EOS
    echo "sbatch run_$probe.slurm" >> submit_runs.sh
  done
done
sed -i.bak '1i #!/bin/bash\nset -euo pipefail\ncd "$(dirname "$0")"\nmkdir -p logs' submit_runs.sh 2>/dev/null || true
echo "generated $(ls run_*.slurm | wc -l) runs"
