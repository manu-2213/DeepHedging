#!/bin/bash
# Main table (2026-11-10): (call, doc, con) x (bsm, heston) x (0.1%, 1%) x (single, double), 5 seeds each.
# Method fixed from the BSM-call work: fast transformer (tfpath) + GRPO group 8, tracking-error reward, selection on
# validation tracking error, chained 1-hour jobs on gpushort OR sae (account pilot_sae_gpu; short 1-GPU jobs backfill
# into sae within minutes, quota 12 GPUs there). Learning rates are tuned per cell on seed 1, then seeds 2-5 rerun the
# chosen rates. Heston uses the same path counts as BSM (2,000 / 500 / 2,000).
#
#   make_main.sh single <dyn> <prod> <fee:01|1> <seed> <actor_lr> [tune]
#       -> run_<dyn>_<prod>_fee<fee>_single_lr<lr>_s<seed>.slurm        results in experiments/results/main/<same name>
#          "tune": on finish it calls "pick", which launches the double grid once all three tuning singles are done
#   make_main.sh tune <dyn> <prod> <fee>                 the three tuning singles (actor lr 3e-5, 1e-4, 3e-4), seed 1
#   make_main.sh pick <dyn> <prod> <fee>                 if the 3 tuning singles have finished: pick the validation-best,
#                                                        build its 4-cell double grid and submit it (once; marker file)
#   make_main.sh double <dyn> <prod> <fee> <seed> <inactor_lr> <joint_lr> <single_run_name>
#       -> run_<dyn>_<prod>_fee<fee>_double_i<il>_j<jl>_s<seed>.slurm
#   make_main.sh grid <dyn> <prod> <fee> <single_run_name>         4 doubles: inactor {1e-3,1e-4} x joint {1e-4,1e-5}
#   make_main.sh seedchain <dyn> <prod> <fee> <seed> <actor_lr> <inactor_lr> <joint_lr>
#       -> the seed's single net, which submits the seed's double net (chosen rates) when it finishes
set -e
cd "$(dirname "$0")"
mkdir -p logs
RES=experiments/results/main
RESABS=${DEEP_HEDGING_REPO:-/data/home/acw796/DeepHedging}/deep_hedging_v0/$RES
COMMON="--n-trials 1 --no-pruning --final-seeds --arch tfpath --rnn-mode window --ppo-epochs 1 --minibatch 125000 --target-kl 0 --reward tracking --reward-norm none --no-obs-norm --no-adv-norm --value-loss huber --value-coef 1.0 --window 10 --pad-mode zero --no-agent-state --call-action-low -1 --train-paths 2000 --val-paths 500 --test-paths 2000 --max-iters 1000000000 --patience 200 --lr-patience 50 --lr-decays 3 --lr-decay-factor 0.5 --time-budget-min 50 --advantage grpo --grpo-group 8 --paths-per-iter 12 --metric mean_abs_tracking_error"
TUNE_LRS="3e-5 1e-4 3e-4"
fee_args() { case "$1" in 01) echo "--underlying-fee 0.001 --option-fee 0.002";; 1) echo "--underlying-fee 0.01 --option-fee 0.05 --init-std 0.2";; *) echo "bad fee $1" >&2; exit 2;; esac; }
tag() { echo "${1/-/}"; }     # 1e-4 -> 1e4

emit() {   # name, python args, on-finish shell (may be empty)
  local name=$1 args=$2 finish=$3
  cat > run_$name.slurm <<EOF
#!/bin/bash
#SBATCH --job-name=mn_$name
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --partition=gpushort,sae
#SBATCH --account=pilot_sae_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00

set -eo pipefail
cd \${DEEP_HEDGING_REPO:-/data/home/acw796/DeepHedging}/deep_hedging_v0
module load cuda/12.4.0-gcc-12.2.0
module load python/3.11.7-gcc-12.2.0
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q V100; then source ../.venv_v100/bin/activate; else source ../.venv/bin/activate; fi
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export FAST_HEDGING_CACHE=\${FAST_HEDGING_CACHE:-\$PWD/../.fast_hedging_cache}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
python -c "import sys, torch; ok = torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', ok); sys.exit(3) if not ok else print('gpu operation', torch.ones(2, device='cuda').mean().item(), torch.cuda.get_device_name(0))"

OUT=$RES/$name
mkdir -p \$OUT
count=\$(( \$(cat \$OUT/job_count 2>/dev/null || echo 0) + 1 ))
echo \$count > \$OUT/job_count
echo "job \$count of this run on \$(hostname)"
rc=0
python -u -m fast_hedging.experiment $args --name $name --out \$OUT || rc=\$?
if [ "\$rc" -eq 75 ]; then
  if [ "\$count" -ge 300 ]; then echo "CHAIN STOPPED: 300 jobs reached"; exit 1; fi
  cd "\$SLURM_SUBMIT_DIR" && sbatch "run_$name.slurm"
elif [ "\$rc" -eq 0 ]; then
  echo "RUN FINISHED after \$count jobs"; cd "\$SLURM_SUBMIT_DIR"; $finish
else
  echo "RUN FAILED with exit code \$rc"; exit "\$rc"
fi
EOF
  echo run_$name.slurm
}

single() {   # dyn prod fee seed lr [tune]
  local dyn=$1 prod=$2 fee=$3 seed=$4 lr=$5 name finish=""
  name=${dyn}_${prod}_fee${fee}_single_lr$(tag $lr)_s$seed
  [ "${6:-}" = tune ] && finish="bash make_main.sh pick $dyn $prod $fee"
  emit $name "--dynamics $dyn --product $prod --net single --actor-lr-range $lr $lr --inactor-lr-range 1e-3 1e-3 --tuning-seeds $seed $(fee_args $fee) $COMMON" "$finish"
}
double() {   # dyn prod fee seed il jl single_run_name
  local dyn=$1 prod=$2 fee=$3 seed=$4 il=$5 jl=$6 src=$7 name
  name=${dyn}_${prod}_fee${fee}_double_i$(tag $il)_j$(tag $jl)_s$seed
  [ -n "$src" ] || { echo "double needs the single run name" >&2; exit 2; }
  emit $name "--dynamics $dyn --product $prod --net double --actor-lr-range 1e-4 1e-4 --inactor-lr-range $il $il --joint-lr-range $jl $jl --actor-init from-single --single-dir $RES/$src --tuning-seeds $seed $(fee_args $fee) $COMMON" ""
}
grid() {     # dyn prod fee single_run_name
  local seed=${4##*_s}
  for il in 1e-3 1e-4; do for jl in 1e-4 1e-5; do double $1 $2 $3 $seed $il $jl $4; done; done
}
case "$1" in
  single) single $2 $3 $4 $5 $6 ${7:-} ;;
  tune)   for lr in $TUNE_LRS; do single $2 $3 $4 1 $lr tune; done ;;
  double) double $2 $3 $4 $5 $6 $7 $8 ;;
  grid)   grid $2 $3 $4 $5 ;;
  pick)
    dyn=$2; prod=$3; fee=$4; best=""; bestv=""
    for lr in $TUNE_LRS; do
      n=${dyn}_${prod}_fee${fee}_single_lr$(tag $lr)_s1; f=$RESABS/$n/summary.json
      [ -f "$f" ] || { echo "pick: $n not finished yet"; exit 0; }
      v=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['best_val_metric'])" "$f")
      echo "pick: $n validation $v"
      if [ -z "$best" ] || python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) < float(sys.argv[2]) else 1)" "$v" "$bestv"; then best=$n; bestv=$v; fi
    done
    mkdir "picked_${dyn}_${prod}_fee${fee}" 2>/dev/null || { echo "pick: grid for $dyn $prod $fee already submitted"; exit 0; }
    echo "pick: $dyn $prod fee$fee -> $best ($bestv); submitting the double grid"
    echo "$best $bestv $(date)" > "picked_${dyn}_${prod}_fee${fee}/choice.txt"
    for f in $(grid $dyn $prod $fee $best); do sbatch "$f"; done ;;
  seedchain)
    dyn=$2; prod=$3; fee=$4; seed=$5; lr=$6; il=$7; jl=$8
    sname=${dyn}_${prod}_fee${fee}_single_lr$(tag $lr)_s$seed
    dname=$(double $dyn $prod $fee $seed $il $jl $sname)
    emit $sname "--dynamics $dyn --product $prod --net single --actor-lr-range $lr $lr --inactor-lr-range 1e-3 1e-3 --tuning-seeds $seed $(fee_args $fee) $COMMON" "sbatch $dname" ;;
  *) echo "usage: see header"; exit 2 ;;
esac
