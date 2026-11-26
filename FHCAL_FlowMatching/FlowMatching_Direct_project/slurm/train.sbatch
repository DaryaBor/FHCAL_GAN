#!/bin/bash
#SBATCH --job-name=fhcal-direct-fm
#SBATCH --partition=gpu-ef-quick
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=02:50:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# Submit from THIS project directory: sbatch slurm/<job>.sbatch
test -f run_pipeline.py || { echo "Wrong Slurm submit directory: $PWD" >&2; exit 2; }
source "${FHCAL_VENV:-$HOME/.venvs/fhcal-flow}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
: "${FHCAL_DATA_FILE:=$HOME/FHCAL_GAN/datasets/raw/fhcal_data_muons_81619_large.npz}"
python -u run_pipeline.py train \
    --data "$FHCAL_DATA_FILE" \
    --max-per-module "${FHCAL_MAX_PER_MODULE:-600}" \
    --out "${FHCAL_RUN_DIR:-experiments/experiment}" \
    --epochs "${FHCAL_EPOCHS:-100}" --resume
