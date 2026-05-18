#!/bin/bash
# Packed submission — runs N tasks in parallel inside ONE Snellius slot.
# Each SLURM array index maps to one "pack" of task IDs (a line in packs.txt).
# This circumvents Snellius's per-slot billing minimum (16 cores) by packing
# many single-CPU tasks into one slot.
#
# Usage:
#     1. Generate packs.txt:  python -m experiments.stage3.jobs.generate_packs
#     2. Submit: sbatch --array=0-$(($(wc -l < experiments/stage3/jobs/packs.txt) - 1)) \
#                       experiments/stage3/jobs/submit_packed.sh

#SBATCH --job-name=stage3-packed
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=5-00:00:00
#SBATCH --output=experiments/stage3/jobs/logs/packed_%A_%a.out
#SBATCH --error=experiments/stage3/jobs/logs/packed_%A_%a.err

set -u   # error on undefined vars; do NOT set -e (so one failing task doesn't kill the pack)

echo "=== Packed task ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}] ==="
echo "Started at $(date)"
echo "Running on: $(hostname)  cpus=${SLURM_CPUS_PER_TASK}  mem=${SLURM_MEM_PER_NODE}MB"

# --- Environment (matches submit.sh) ---
module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
source ~/tn-lbm-venv/bin/activate

# Force single-threaded numpy/quimb per task; tasks share the 16 cores via OS scheduling
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd ~/master_thesis_code
mkdir -p experiments/stage3/jobs/logs

# --- Read this pack's task IDs from packs.txt ---
PACKS_FILE="experiments/stage3/jobs/packs.txt"
if [ ! -f "$PACKS_FILE" ]; then
    echo "ERROR: $PACKS_FILE not found. Run generate_packs.py first."
    exit 1
fi

LINE_NUM=$((SLURM_ARRAY_TASK_ID + 1))
TASKS=$(sed -n "${LINE_NUM}p" "$PACKS_FILE")
if [ -z "$TASKS" ]; then
    echo "ERROR: no pack at line ${LINE_NUM} of $PACKS_FILE"
    exit 1
fi

echo "Pack contains tasks: $TASKS"

# --- Wall-clock budget (5 days minus 5 min slack for clean checkpoint) ---
WALL_LIMIT_S=$((5 * 24 * 3600 - 300))

# --- Spawn one python process per task, all in parallel ---
PIDS=()
for tid in ${TASKS//,/ }; do
    LOG="experiments/stage3/jobs/logs/packed_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_task${tid}.log"
    echo "  spawning task ${tid} (log: $(basename $LOG))"
    python -u -m experiments.stage3.run_job \
        --job-id "${tid}" \
        --wall-clock-limit-s "${WALL_LIMIT_S}" \
        --verbose \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo "Spawned ${#PIDS[@]} parallel tasks. Waiting for completion..."

# --- Wait for all to finish, collect exit codes ---
EXIT_COUNT_OK=0
EXIT_COUNT_INCOMPLETE=0
EXIT_COUNT_ERROR=0
for pid in "${PIDS[@]}"; do
    if wait "$pid"; then
        EXIT_COUNT_OK=$((EXIT_COUNT_OK + 1))
    else
        rc=$?
        if [ $rc -eq 99 ]; then
            EXIT_COUNT_INCOMPLETE=$((EXIT_COUNT_INCOMPLETE + 1))
        else
            EXIT_COUNT_ERROR=$((EXIT_COUNT_ERROR + 1))
        fi
    fi
done

echo "=== Pack done at $(date) ==="
echo "  complete: $EXIT_COUNT_OK"
echo "  incomplete (wall-clock): $EXIT_COUNT_INCOMPLETE"
echo "  error: $EXIT_COUNT_ERROR"

# Exit 0 if at least one task succeeded; SLURM treats the pack as successful
exit 0
