#!/bin/bash
#SBATCH --job-name=pregen_paths
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

#SBATCH --account=pilot_sae_gpu
#SBATCH --partition=sae          # same partition as the training jobs
#SBATCH --gres=gpu:1             # this account's QOS requires a GPU request (QOSMinGRES), even though this job is CPU-only work
#SBATCH --cpus-per-task=4        # lightweight: path generation itself doesn't use the GPU
#SBATCH --mem=16G
#SBATCH --time=00:30:00

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL

# ----------------------------
# Environment
# ----------------------------
cd /data/home/acw796/DeepHedging || exit 1

module load python/3.11.7-gcc-12.2.0
source .venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export DEEP_HEDGING_PATH_CACHE=1

# ----------------------------
# Prime the on-disk simulation-path cache for seeds 1-10 (train split) and
# seed 0 (test split, always used by testing.py:test_model). Run this job
# to completion BEFORE submitting the 24 per-script jobs in this folder, e.g.:
#
#   pregen_id=$(sbatch --parsable pregenerate_paths.slurm)
#   for f in *.slurm; do
#       [ "$f" = "pregenerate_paths.slurm" ] && continue
#       sbatch --dependency=afterok:$pregen_id "$f"
#   done
# ----------------------------
python deep_hedging_v0/experiments/utils/pregenerate_simulation_paths.py --seed 1 2 3 4 5 6 7 8 9 10
