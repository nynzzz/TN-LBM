#!/bin/bash
# Submit one SV-capture run to Snellius.
#
# Usage:
#     sbatch experiments/stage3/exp_sv_spectrum/submit_sv.sh <chi> <run_name>
# Example:
#     sbatch experiments/stage3/exp_sv_spectrum/submit_sv.sh 17 cluster_chi17
#     sbatch experiments/stage3/exp_sv_spectrum/submit_sv.sh 12 cluster_chi12_repro
#
# Runs TG N=256 Re=1000 (the wrong-attractor test physics) with the standard
# 11-snapshot set from t=0 to t=64845. Outputs snapshot pickles into
# experiments/stage3/exp_sv_spectrum/data/<run_name>/.

#SBATCH --job-name=sv-spectrum
#SBATCH --partition=fat_rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=experiments/stage3/exp_sv_spectrum/data/%x_%j.out
#SBATCH --error=experiments/stage3/exp_sv_spectrum/data/%x_%j.err

set -u

if [ $# -lt 2 ]; then
    echo "Usage: sbatch $0 <chi> <run_name>"
    exit 1
fi
CHI="$1"
RUN_NAME="$2"

echo "=== SV-capture job ${SLURM_JOB_ID} ==="
echo "Started at $(date)"
echo "Running on: $(hostname)  cpus=${SLURM_CPUS_PER_TASK}  mem=${SLURM_MEM_PER_NODE}MB"
echo "chi=${CHI}  run_name=${RUN_NAME}"

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
source ~/tn-lbm-venv/bin/activate

# Force single-threaded numpy/quimb for reproducibility on the cluster
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd ~/master_thesis_code
mkdir -p experiments/stage3/exp_sv_spectrum/data

python -u -m experiments.stage3.exp_sv_spectrum.run_sv_capture \
    --chi "${CHI}" --run-name "${RUN_NAME}" \
    --snapshot-times 0 100 500 2000 7000 15000 22000 30000 40000 50000 64845 \
    --n 256 --re 1000 \
    --cutoff 1e-10 --taylor-order 2

echo "=== done at $(date) ==="
