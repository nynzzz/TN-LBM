#!/bin/bash
# Pre-generate all vanilla baselines as a SLURM array (parallel).
# Run BEFORE submitting the MPS array (submit.sh) so MPS jobs don't race
# on baseline writes.
#
# This is a SLURM array — one task per unique baseline. The number of
# tasks must match `python -m experiments.stage3.build_baselines --count`.
# As of the current master_jobs.csv that's 80; adjust --array if you change
# the master CSV.
#
# Usage:
#     # Find the right array size:
#     python -m experiments.stage3.build_baselines --count   # e.g. 68
#     # Submit:
#     sbatch experiments/stage3/jobs/submit_baselines.sh
#     # Or override array size from command line:
#     sbatch --array=0-67 experiments/stage3/jobs/submit_baselines.sh

#SBATCH --job-name=stage3-baselines
#SBATCH --partition=rome
#SBATCH --array=0-79
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --output=experiments/stage3/jobs/logs/baselines_%A_%a.out
#SBATCH --error=experiments/stage3/jobs/logs/baselines_%A_%a.err

set -e

echo "=== Baseline task ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}] ==="
echo "Started at $(date)"
echo "Running on: $(hostname)"

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
source ~/tn-lbm-venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd ~/master_thesis_code
mkdir -p experiments/stage3/jobs/logs

python -u -m experiments.stage3.build_baselines \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --verbose

echo "=== Done at $(date) ==="
