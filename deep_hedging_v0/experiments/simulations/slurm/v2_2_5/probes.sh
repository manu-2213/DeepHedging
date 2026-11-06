#!/bin/bash
# Learning-rate probes on the BSM call, packed on the 2 GPUs gpushort allows. One old epoch = 400 iterations.
# Usage, from this folder: bash probes.sh
# OLD reproduces the old pipeline as far as the engine allows, and is faster because it is vectorised:
# tracking reward in dollars, no observation or advantage normalisation, Huber critic with weight 1, 100 paths,
# 4 gradient steps per rollout, no KL stop, window 10 with zero padding, old inputs, positions -1 to 1, fee 0.1%.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
lr() { echo "--actor-lr-range $1 $1"; }
SHAPE="--paths-per-iter 100 --ppo-epochs 1 --minibatch 125000 --target-kl 0"
OLD="$SHAPE --reward tracking --reward-norm none --no-obs-norm --no-adv-norm --value-loss huber --value-coef 1.0 --window 10 --pad-mode zero --no-agent-state --call-action-low -1"
FEE01="--underlying-fee 0.001 --option-fee 0.002"
FEE1="--underlying-fee 0.01 --option-fee 0.05"
{ for x in 1e-6 5e-6 1e-5 2e-5 1e-4; do echo "old_lr_$x $OLD $FEE01 $(lr $x)"; done; } > logs/pack_old.txt
{ for x in 5e-6 1e-5; do echo "oldfee1_lr_$x $OLD $FEE1 $(lr $x)"; done
  for x in 1e-5 1e-4 1e-3; do echo "cvar4_lr_$x $SHAPE --reward mean_cvar --window 8 $FEE1 $(lr $x)"; done; } > logs/pack_other.txt
for pack in old other; do
  echo "$pack: $(sbatch --parsable -J pack_$pack probe_pack.slurm "$PWD/logs/pack_$pack.txt")"
done
