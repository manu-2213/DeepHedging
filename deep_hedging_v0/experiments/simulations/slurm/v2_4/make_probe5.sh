#!/bin/bash
# v2.4 probe 5 (2026-11-09): double-net training ideas, 1% fee, seed 1, fast transformer + GRPO group 8.
# Every run here selects on a 2,000-path validation set (--val-paths 2000) instead of 500.
#   p5ctl       double from the existing actor tgsg8_fee1_s1, inactor 1e-4 x joint 1e-4   (control: only the validation size changes)
#   p5init      p5ctl + inactor starts at 5% holding (--inactor-init-hold-prob 0.05)
#   p5sstate    single net WITH agent state (holdings, tracking error, costs, steps since trade)   -> p5dstate (double from it)
#   p5shold     p5sstate + forced holds in training (--actor-hold-dropout 0.95)   -> p5dhold, p5dholdinit (+ inactor at 5% holding)
# The two single nets submit their double nets when they finish (HPC side).
set -e
cd "$(dirname "$0")"
V="--val-paths 2000"
D=run_dbltg8_i1e4_j1e4_fee1_s1.slurm; d=dbltg8_i1e4_j1e4_fee1_s1
S=run_tgsg8_fee1_s1.slurm; s=tgsg8_fee1_s1
HOOK='; cd "$SLURM_SUBMIT_DIR" \&\& bash make_double.sh B tg8 1 \&\& for f in run_dbltg8_\*_fee1_s1.slurm; do sbatch "$f"; done'
dbl() {      # name, single dir name, extra flags, "state" to keep agent state
  sed -e "s/$d/$1_fee1_s1/g" -e "s|--single-dir experiments/results/v2_4/tgsg8_fee1_s1|--single-dir experiments/results/v2_4/$2|" \
      -e "s/ --metric / $V $3 --metric /" $D > run_$1_fee1_s1.slurm
  [ "$4" = state ] && sed -i -e "s/ --no-agent-state//" run_$1_fee1_s1.slurm
  return 0
}
single() {   # name, extra flags, double nets to submit on finish
  sub=""; for n in $3; do sub="$sub sbatch run_${n}_fee1_s1.slurm;"; done
  sed -e "s/$s/$1_fee1_s1/g" -e "s/ --no-agent-state//" -e "s/ --metric / $V $2 --metric /" \
      -e "s|; cd \"\$SLURM_SUBMIT_DIR\" && bash make_double.sh B tg8 1 && for f in run_dbltg8_\*_fee1_s1.slurm; do sbatch \"\$f\"; done|; cd \"\$SLURM_SUBMIT_DIR\" \&\&$sub|" $S > run_$1_fee1_s1.slurm
}
dbl p5ctl  tgsg8_fee1_s1 ""
dbl p5init tgsg8_fee1_s1 "--inactor-init-hold-prob 0.05"
single p5sstate "" "p5dstate"
single p5shold "--actor-hold-dropout 0.95" "p5dhold p5dholdinit"
dbl p5dstate    p5sstate_fee1_s1 "" state
dbl p5dhold     p5shold_fee1_s1 "" state
dbl p5dholdinit p5shold_fee1_s1 "--inactor-init-hold-prob 0.05" state
