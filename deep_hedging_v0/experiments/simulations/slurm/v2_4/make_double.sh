#!/bin/bash
# v2.4 double net (2026-11-08): the main architecture (actor + inaction net, costs modelled by the hold decision).
# Learning rates first. All runs select on validation tracking error.
#  Stage A  single nets, the double net's stage-1 actor and the reference it must beat:
#           lstmsingle (LSTM + PPO, the best actor so far), grposingle (LSTM + GRPO), tfpsingle, tfpgrposingle (fast transformer)
#  Stage B  joint phase from a finished single net (--actor-init from-single), grid
#           inactor lr {1e-5, 1e-4, 1e-3} x joint actor lr {1e-5, 1e-4}:  dbl<src>_i<lr>_j<lr>_fee<f>_s1
#           Lower than the first draft because in the v2.3 runs an inactor rate of 1e-3 collapsed to always holding.
#  usage: make_double.sh A                 -> stage A scripts
#         make_double.sh B lstm|grpo|tfp|tfpgrpo|tg8 01|1  -> stage B scripts for that fee from that single net
#  run_lstmsingle_fee<f>_s1.slurm launches its own stage B grid when it finishes (HPC side, no local connection needed).
set -e
cd "$(dirname "$0")"
M="--metric mean_abs_tracking_error"
NN='/^#SBATCH --nice/d'      # band templates carry a low priority; the double-net runs must not inherit it
template() {                 # LSTM + PPO band script, used only as a template (never submitted)
  sed -e "$NN" -e "s/grpo_fee${1}_s1/lstm_fee${1}_s1/g" -e "s/ --advantage grpo --grpo-group 4 --paths-per-iter 25//" run_grpo_fee${1}_s1.slurm > run_lstm_fee${1}_s1.slurm
}
if [ "$1" = A ]; then
  for fee in 01 1; do
    template $fee
    sed -e "$NN" -e "s/tfp_fee${fee}_s1/tfpsingle_fee${fee}_s1/g" -e "s/--net band/--net single/" -e "s/ --name / $M --name /" run_tfp_fee${fee}_s1.slurm > run_tfpsingle_fee${fee}_s1.slurm
    sed -e "$NN" -e "s/tfpgrpo_fee${fee}_s1/tfpgrposingle_fee${fee}_s1/g" -e "s/--net band/--net single/" -e "s/ --name / $M --name /" run_tfpgrpo_fee${fee}_s1.slurm > run_tfpgrposingle_fee${fee}_s1.slurm
    sed -e "$NN" -e "s/grpo_fee${fee}_s1/grposingle_fee${fee}_s1/g" -e "s/--net band/--net single/" -e "s/ --name / $M --name /" run_grpo_fee${fee}_s1.slurm > run_grposingle_fee${fee}_s1.slurm
    sed -e "s/lstm_fee${fee}_s1/lstmsingle_fee${fee}_s1/g" -e "s/--net band/--net single/" -e "s/ --name / $M --name /" \
        -e "s|echo \"RUN FINISHED after \$count jobs\"|echo \"RUN FINISHED after \$count jobs\"; cd \"\$SLURM_SUBMIT_DIR\" \&\& bash make_double.sh B lstm ${fee} \&\& for f in run_dbllstm_*_fee${fee}_s1.slurm; do sbatch \"\$f\"; done|" \
        run_lstm_fee${fee}_s1.slurm > run_lstmsingle_fee${fee}_s1.slurm
  done
elif [ "$1" = B ]; then
  src=$2; fee=$3
  if [ "$src" = lstm ] && [ -z "$ALLOW_LSTM_GRID" ]; then   # 2026-11-09: double-net grids from the fast transformer + GRPO actor only
    echo "LSTM grid disabled: set ALLOW_LSTM_GRID=1 to make it"; exit 3
  fi
  [ "$src" = lstm ] && template $fee
  sdir=${src}single
  if [ "$src" = tg8 ]; then    # fast transformer + GRPO with group 8 (2026-11-09): actor from run tgsg8
    sdir=tgsg8
    sed -e "s/tfpgrpo_fee${fee}_s1/tg8_fee${fee}_s1/g" -e "s/--grpo-group 4 --paths-per-iter 25/--grpo-group 8 --paths-per-iter 12/" run_tfpgrpo_fee${fee}_s1.slurm > run_tg8_fee${fee}_s1.slurm
  fi
  for il in 1e-5 1e-4 1e-3; do for jl in 1e-5 1e-4; do
    n=dbl${src}_i${il/-/}_j${jl/-/}_fee${fee}_s1
    sed -e "$NN" -e "s/${src}_fee${fee}_s1/$n/g" -e "s/--net band/--net double/" \
        -e "s/--inactor-lr-range 1e-3 1e-3/--inactor-lr-range $il $il --joint-lr-range $jl $jl --actor-init from-single --single-dir experiments\/results\/v2_4\/${sdir}_fee${fee}_s1/" \
        -e "s/ --name / $M --name /" run_${src}_fee${fee}_s1.slurm > run_$n.slurm
  done; done
fi
