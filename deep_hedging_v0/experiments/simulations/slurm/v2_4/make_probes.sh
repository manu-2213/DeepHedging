#!/bin/bash
# v2.4 probes: can a transformer encoder, critic-free GRPO advantages, a bigger batch, more training data
# or a wider net make the band net (BSM call) learn faster or better than the v2.3 setup?
# Every probe is the v2.3 band run (same seed, same fees, same old-equivalent setup) with ONE change, so the
# matching v2.3 run (experiments/results/v2_3/bsm_call_band_fee*_s*) is its control, compared at equal
# iterations (validation CVaR95 and tracking error in log.txt). Each pack runs 3 probes side by side on one
# 80 GB GPU for 52 minutes and saves a resume state at the end, so a promising probe can be continued.
# Results: deep_hedging_v0/experiments/results/v2_4/<probe>/log.txt
# Usage: bash make_probes.sh && bash submit_probes.sh   (the tests run first; the packs start only if they pass)
set -euo pipefail
cd "$(dirname "$0")"
REPO=/data/home/acw796/DeepHedging
RESULTS=experiments/results/v2_4

OLD="--arch rnn --rnn-mode window --paths-per-iter 100 --ppo-epochs 1 --minibatch 125000 --target-kl 0 --reward tracking --reward-norm none --no-obs-norm --no-adv-norm --value-loss huber --value-coef 1.0 --window 10 --pad-mode zero --no-agent-state --call-action-low -1"
RUN="--dynamics bsm --product call --net band --n-trials 1 --no-pruning --final-seeds --actor-lr-range 1e-4 1e-4 --inactor-lr-range 1e-3 1e-3"
STOP="--max-iters 1000000000 --patience 200 --lr-patience 50 --lr-decays 3 --lr-decay-factor 0.5 --time-budget-min 52"
FEE01="--underlying-fee 0.001 --option-fee 0.002"
FEE1="--underlying-fee 0.01 --option-fee 0.05 --init-std 0.2"
# each change is appended after the v2.3 arguments; argparse keeps the last value of a repeated flag
TF="--arch transformer"
GRPO="--advantage grpo --grpo-group 4 --paths-per-iter 25"      # 25 paths x 4 copies = the 100 rollouts of v2.3
BATCH="--paths-per-iter 400"                                    # 4x the rollouts, 16 gradient steps per iteration
DATA="--train-paths 20000"                                      # 10x the training paths
WIDE="--hidden 128"
rm -f pack_*.txt pack_*.slurm submit_probes.sh

pack() {  # $1 pack name; then "probe_name|extra args" items
  local name=$1; shift
  : > "pack_$name.txt"
  for item in "$@"; do
    local probe=${item%%|*} extra=${item#*|}
    local fee=${probe#*_fee}; fee=FEE${fee%%_*}; local seed=${probe##*_s}
    echo "$probe $RUN --tuning-seeds $seed $OLD ${!fee} $STOP $extra" >> "pack_$name.txt"
  done
  cat > "pack_$name.slurm" <<EOS
#!/bin/bash
#SBATCH --job-name=v24_$name
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
cd \${DEEP_HEDGING_REPO:-$REPO}/deep_hedging_v0
module load cuda/12.4.0-gcc-12.2.0
module load python/3.11.7-gcc-12.2.0
source ../.venv/bin/activate
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export FAST_HEDGING_CACHE=\${FAST_HEDGING_CACHE:-\$PWD/../.fast_hedging_cache}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
python -c "import sys, torch; ok = torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', ok); sys.exit(3) if not ok else print('gpu operation', torch.ones(2, device='cuda').mean().item(), torch.cuda.get_device_name(0))"
while read -r probe args; do
  [ -z "\$probe" ] && continue
  OUT=$RESULTS/\$probe; mkdir -p \$OUT
  python -u -m fast_hedging.experiment \$args --name \$probe --out \$OUT > \$OUT/console.txt 2>&1 &
done < "\$SLURM_SUBMIT_DIR/pack_$name.txt"
sleep 600; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv || true
wait || true
EOS
}

# transformer and GRPO first (the user's priority), then batch / data / width
for s in 1 2; do
  for fee in 01 1; do
    pack "arch_fee${fee}_s$s" "tf_fee${fee}_s$s|$TF" "grpo_fee${fee}_s$s|$GRPO" "tfgrpo_fee${fee}_s$s|$TF $GRPO"
  done
done
for fee in 01 1; do
  pack "scale_fee${fee}_s1" "batch_fee${fee}_s1|$BATCH" "data_fee${fee}_s1|$DATA" "wide_fee${fee}_s1|$WIDE"
done

cat > check_new_modes.slurm <<EOS
#!/bin/bash
#SBATCH --job-name=v24_check
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:30:00

set -eo pipefail
cd \${DEEP_HEDGING_REPO:-$REPO}/deep_hedging_v0
module load cuda/12.4.0-gcc-12.2.0
module load python/3.11.7-gcc-12.2.0
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q V100; then source ../.venv_v100/bin/activate; else source ../.venv/bin/activate; fi
set -u
export PYTHONUNBUFFERED=1
export FAST_HEDGING_CACHE=\${FAST_HEDGING_CACHE:-\$PWD/../.fast_hedging_cache}
python -u -m fast_hedging.tests.test_new_modes
EOS

cat > submit_probes.sh <<'EOS'
#!/bin/bash
# Usage, from this folder: bash submit_probes.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
check=$(sbatch --parsable check_new_modes.slurm)
echo "check_new_modes: $check"
for f in pack_arch_*.slurm pack_scale_*.slurm; do
  echo "$f: $(sbatch --parsable --dependency=afterok:$check "$f")"
done
EOS
echo "generated $(ls pack_*.slurm | wc -l) packs"
