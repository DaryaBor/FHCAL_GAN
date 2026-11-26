#!/bin/bash
#SBATCH --job-name=fhcal-gmm-fm
#SBATCH --partition=gpu-ef-quick
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# Submit from THIS project directory: sbatch slurm/<job>.sbatch
test -f run_pipeline.py || { echo "Wrong Slurm submit directory: $PWD" >&2; exit 2; }
source "${FHCAL_VENV:-$HOME/.venvs/fhcal-flow}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
: "${FHCAL_DATA_DIR:=$PWD/data}"
python -u run_pipeline.py train \
    --data8 "$FHCAL_DATA_DIR/fhcal_data_muons_8.npz" \
    --data16 "$FHCAL_DATA_DIR/fhcal_data_muons_16.npz" \
    --data19 "$FHCAL_DATA_DIR/fhcal_data_muons_19.npz" \
    --max-per-module "${FHCAL_MAX_PER_MODULE:-600}" \
    --fits "${FHCAL_FITS:-experiments/gmm_fits/fitbank.npz}" \
    --out "${FHCAL_RUN_DIR:-experiments/experiment}" \
    --epochs "${FHCAL_EPOCHS:-100}" --resume
