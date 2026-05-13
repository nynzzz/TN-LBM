#!/bin/bash
# Stage 3 MPS-native LBM — SLURM array submission for Snellius.
#
# Default: submits all 460 jobs in the master CSV.
# Use --array override to submit a subset:
#     sbatch --array=0-191      jobs/submit.sh     # Exp 1 only
#     sbatch --array=192-327    jobs/submit.sh     # Exp 2 only
#     sbatch --array=99,150,200 jobs/submit.sh     # specific jobs
#
# Each task runs one job (single MPS simulation). Idempotent: results that
# already exist are skipped, so resubmits are safe.
#
# Exit 99 = wall-clock limit hit, checkpoint saved — resubmit to resume.

#SBATCH --job-name=stage3-mps
#SBATCH --partition=rome
#SBATCH --array=0-459
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5-00:00:00
#SBATCH --mem=8G
#SBATCH --output=experiments/stage3/jobs/logs/job_%A_%a.out
#SBATCH --error=experiments/stage3/jobs/logs/job_%A_%a.err

set -e

echo "=== SLURM array task ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}] ==="
echo "Started at $(date)"
echo "Running on node: $(hostname)"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK}"
echo "Memory allocated: ${SLURM_MEM_PER_NODE} MB"

# --- Environment (matches stage 2 setup) ---
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
source ~/tn-lbm-venv/bin/activate

# Force single-threaded numpy/quimb so cores aren't oversubscribed
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd ~/master_thesis_code
mkdir -p experiments/stage3/jobs/logs

# --- Wall-clock budget ---
# 5 days (matches #SBATCH --time=5-00:00:00) minus 5 min slack for checkpoint save.
WALL_LIMIT_S=$((5 * 24 * 3600 - 300))
echo "Wall-clock limit passed to runner: ${WALL_LIMIT_S}s"

# --- Run the job ---
python -u -m experiments.stage3.run_job \
    --job-id "${SLURM_ARRAY_TASK_ID}" \
    --wall-clock-limit-s "${WALL_LIMIT_S}" \
    --verbose

EXIT_CODE=$?
echo "=== Task exit code: ${EXIT_CODE} ==="
echo "Finished at $(date)"

if [ ${EXIT_CODE} -eq 99 ]; then
    echo "Wall-clock limit hit — checkpoint saved. Resubmit to resume."
fi

exit ${EXIT_CODE}
