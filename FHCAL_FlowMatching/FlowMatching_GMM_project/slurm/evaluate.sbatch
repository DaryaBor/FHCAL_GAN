#!/bin/bash
#SBATCH --job-name=fhcal-gmm-eval
#SBATCH --partition=gpu-ef-quick
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# Submit from THIS project directory: sbatch slurm/<job>.sbatch
test -f run_pipeline.py || { echo "Wrong Slurm submit directory: $PWD" >&2; exit 2; }
source "${FHCAL_VENV:-$HOME/.venvs/fhcal-flow}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
: "${FHCAL_DATA_DIR:=$PWD/data}"
python -u run_pipeline.py evaluate \
    --data8 "$FHCAL_DATA_DIR/fhcal_data_muons_8.npz" \
    --data16 "$FHCAL_DATA_DIR/fhcal_data_muons_16.npz" \
    --data19 "$FHCAL_DATA_DIR/fhcal_data_muons_19.npz" \
    --checkpoint "${FHCAL_RUN_DIR:-experiments/experiment}/best.pt" \
    --out "${FHCAL_RESULT_DIR:-results/experiment}" \
    --num-per-module "${FHCAL_EVAL_NUM:-2048}" --steps 20 --solver euler --packets "${FHCAL_GMM_PACKETS:-128}"
