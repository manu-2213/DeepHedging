#!/bin/bash
# Usage, from this folder: bash submit_runs.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
sbatch run_grpo_fee1_s1.slurm
sbatch run_grpo_fee1_s2.slurm
sbatch run_tf_fee01_s1.slurm
sbatch run_tf_fee01_s2.slurm
sbatch run_tfgrpo_fee01_s1.slurm
sbatch run_tfgrpo_fee01_s2.slurm
sbatch run_tf_fee1_s1.slurm
sbatch run_tf_fee1_s2.slurm
sbatch run_grpo_fee01_s1.slurm
sbatch run_grpo_fee01_s2.slurm
sbatch run_tfgrpo_fee1_s1.slurm
sbatch run_tfgrpo_fee1_s2.slurm
