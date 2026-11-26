#!/bin/bash
#SBATCH --job-name=fhcal-check-data
#SBATCH --partition=cpu-e-quick
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -f run_pipeline.py || { echo "Submit from inside FlowMatching project" >&2; exit 2; }
source "${FHCAL_VENV:-$HOME/.venvs/fhcal-flow}/bin/activate"
: "${FHCAL_DATA_FILE:=$HOME/FHCAL_GAN/datasets/raw/fhcal_data_muons_81619_large.npz}"
python -u run_pipeline.py check_data \
    --data "$FHCAL_DATA_FILE" \
    --max-per-module "${FHCAL_MAX_PER_MODULE:-60}"
