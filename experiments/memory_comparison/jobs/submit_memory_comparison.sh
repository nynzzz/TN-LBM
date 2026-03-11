#!/bin/bash
#SBATCH --job-name=tn-lbm-memory
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=logs/memory_comparison_%j.out
#SBATCH --error=logs/memory_comparison_%j.err

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"

# Load modules
module purge
module load 2024
module load Python/3.11.5-GCCcore-13.2.0

# Activate virtual environment
source ~/tn-lbm-venv/bin/activate

# Navigate to project directory
cd ~/master_thesis_code

# Disable numpy threading 
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Run the experiment (48 cores for 48 tasks)
python -u experiments/memory_comparison/run_memory_comparison.py \
    --sim all \
    --n-cores 48 \
    --verbose

echo "Job completed at $(date)"
