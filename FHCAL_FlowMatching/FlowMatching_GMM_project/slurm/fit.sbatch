#!/bin/bash
#SBATCH --job-name=fhcal-gmm-fit
#SBATCH --partition=cpu-e-quick
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Using cpu-e-quick for CPU fitting; override partition if needed.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# Submit from THIS project directory: sbatch slurm/<job>.sbatch
test -f run_pipeline.py || { echo "Wrong Slurm submit directory: $PWD" >&2; exit 2; }
source "${FHCAL_VENV:-$HOME/.venvs/fhcal-flow}/bin/activate"
export OMP_NUM_THREADS=1
: "${FHCAL_DATA_FILE:=$HOME/FHCAL_GAN/datasets/raw/fhcal_data_muons_81619_large.npz}"
python -u run_pipeline.py fit \
    --data "$FHCAL_DATA_FILE" \
    --max-per-module "${FHCAL_MAX_PER_MODULE:-600}" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    --out "${FHCAL_FITS:-experiments/gmm_fits/fitbank.npz}"
