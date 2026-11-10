#!/bin/bash
#SBATCH --job-name=mn_bench_share
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:40:00
# GPU-sharing benchmark (2026-11-10): the main-table single net (BSM call, 1%, group 8) run 1, 2 and 4 times at once on
# one GPU, 4 minutes each phase. Reports s/it per process and the GPU utilisation of a single process.
set -eo pipefail
scontrol release 35712553 35712554 35712555 35712556 35712557 35712558 35712559 35712560 || true
cd ${DEEP_HEDGING_REPO:-/data/home/acw796/DeepHedging}/deep_hedging_v0
module load cuda/12.4.0-gcc-12.2.0
module load python/3.11.7-gcc-12.2.0
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q V100; then source ../.venv_v100/bin/activate; else source ../.venv/bin/activate; fi
set -u
export PYTHONUNBUFFERED=1
export FAST_HEDGING_CACHE=${FAST_HEDGING_CACHE:-$PWD/../.fast_hedging_cache}
nvidia-smi --query-gpu=name,memory.total --format=csv
C="--dynamics bsm --product call --net single --actor-lr-range 1e-4 1e-4 --inactor-lr-range 1e-3 1e-3 --tuning-seeds 1 --underlying-fee 0.01 --option-fee 0.05 --init-std 0.2 --n-trials 1 --no-pruning --final-seeds --arch tfpath --rnn-mode window --ppo-epochs 1 --minibatch 125000 --target-kl 0 --reward tracking --reward-norm none --no-obs-norm --no-adv-norm --value-loss huber --value-coef 1.0 --window 10 --pad-mode zero --no-agent-state --call-action-low -1 --train-paths 2000 --val-paths 500 --test-paths 2000 --max-iters 1000000000 --patience 200 --lr-patience 50 --lr-decays 3 --lr-decay-factor 0.5 --advantage grpo --grpo-group 8 --paths-per-iter 12 --metric mean_abs_tracking_error --time-budget-min 4"
B=experiments/results/main_checks/bench_share
rm -rf $B; mkdir -p $B
for n in 1 2 4; do
  echo "=== $n concurrent process(es) ==="
  export OMP_NUM_THREADS=$(( 8 / n ))
  if [ $n -eq 1 ]; then ( for i in $(seq 1 20); do sleep 12; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; done ) > $B/util_1.txt & UT=$!; fi
  for k in $(seq 1 $n); do
    mkdir -p $B/n${n}_k$k
    python -u -m fast_hedging.experiment $C --name bench_n${n}_k$k --out $B/n${n}_k$k > $B/n${n}_k$k/stdout.txt 2>&1 || true &
  done
  wait $(jobs -p | grep -v "^${UT:-x}$") 2>/dev/null || wait
  for k in $(seq 1 $n); do
    last=$(grep -E "\] it " $B/n${n}_k$k/log.txt | tail -1)
    echo "n=$n k=$k: $(echo "$last" | grep -oE "it +[0-9]+" | head -1) $(echo "$last" | grep -oE "[0-9.]+s/it")"
  done
done
echo "=== GPU utilisation samples, single process (util %, memory MiB) ==="; cat $B/util_1.txt
echo "BENCH DONE"
