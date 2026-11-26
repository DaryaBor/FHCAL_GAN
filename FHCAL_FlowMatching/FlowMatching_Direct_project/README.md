# Direct CNN Flow Matching — FHCAL 8/16/19

**Location in shared repository:** `FHCAL_GAN/FlowMatching_Direct_project/`.
The existing `GAN_project/` stays untouched. This project has its own code,
configuration, data paths, checkpoints and Slurm scripts.

## Task and representation

- Three **muon** beam scenarios incident in central modules **8, 16, 19**.
- Output shape `(N,7,7,5)` = 7 longitudinal sections × 7 rows × 5 central columns (display: 7×5×7).
- The central beam hole is fixed to zero. No 8th–10th longitudinal sections are generated.
- Input NPZ must include `EnergyDeposit`, `ParticlePDG`, `ParticlePoint`, `ParticleMomentum`.
- If a file is `(N,7,9,10)`, it is cropped with `[:, :, 2:7, :7]` and transposed to `(N,7,7,5)`.
- Three-module stratified split 80/10/10 with seed 42, train-only normalization.
- *One conditional network*, module one-hot plus normalized ParticlePoint (x,y).
- Source `GAN_project` has independent `config/`, `pipeline/`, `run_pipeline.py`. We use an analogous layout,
  but name our own Python package **`flow_pipeline/`** to avoid collisions with the **upstream FHCal_GMM**
  Python module also named `pipeline` (and with the adjacent `GAN_project/pipeline`).

## Architecture

4-block 64-channel CNN/ResNet predicting 7-channel velocities, trained with independent conditional Flow Matching from N(0,I). No GAN discriminator.

## Project files

```text
FlowMatching_Direct_project/
├── run_pipeline.py            # unified entry point: train / evaluate / check_data / prepare_combined
├── flow_pipeline/             # data, neural network, fitting, metrics, evaluation
├── config/                    # example Charisma environment variables
├── data/                      # small README only; use FHCAL_DATA_FILE for external NPZ
├── experiments/               # uncommitted checkpoints and fit banks
├── results/                   # uncommitted plots and CSV/JSON metrics
├── slurm/                     # SBATCH jobs
├── tests/                     # CPU smoke tests
├── docs/                      # comparison protocol
└── README.md
```

## Local checks (inside THIS folder; small files only)

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
# For the real 3.5 GB data use `sbatch slurm/check_data.sbatch` instead of login node.
```

**The combined 8/16/19 NPZ is supported directly via `--data`: no splitting needed.**
Legacy `--data8/--data16/--data19` remains supported if needed. `prepare_combined` is optional.
You must check beam directions against those encoded in `flow_pipeline/core.py`; do not silently relabel beams.
Don't substitute earlier proton-only files for 16/19.

## Slurm on Charisma (start from the shared repository root)

```bash
cd FlowMatching_Direct_project
mkdir -p logs
# Optional: cp config/charisma.env.example config/charisma.env && edit config/charisma.env
# Optional: set -a; source config/charisma.env; set +a
export FHCAL_DATA_FILE=$HOME/FHCAL_GAN/datasets/raw/fhcal_data_muons_81619_large.npz
export FHCAL_VENV=$HOME/.venvs/fhcal-flow
export FHCAL_MAX_PER_MODULE=600  # pipeline pilot, not full dataset
# No upstream reference required
sbatch slurm/train.sbatch
# After training finishes: sbatch slurm/evaluate.sbatch

squeue -u "$USER"
```

**Important:** These Slurm scripts intentionally omit `#SBATCH --mem` because this
Charisma partition reports `RealMemory=1` MB to Slurm. GPU jobs are limited to 2h50m by default;
verify actual resources/limits with cluster administrators before long jobs.

**Important:** GPU partition `gpu-ef-quick` is PROVISIONAL; first inspect `sinfo -o '%P %l %G'`
and adjust `#SBATCH --partition`, resource requests, and virtual environment on Charisma.
Training is not run on the login node.

Pilot defaults: 600 selected examples per module, 100 epochs and 20-step Euler during evaluation.
Full data: set `FHCAL_MAX_PER_MODULE=0`, use NEW experiment output paths, and match the setting
and split seed across both projects. Evaluation produces per-module and pooled distributions,
KS, p-value (with zero-mass caveat), JS, occupancy, profile and throughput.


## Scientific caveat

Both projects have been smoke-tested on synthetic inputs only; no full three-module GPU training
is claimed. Comparison is valid only on the SAME central Geant4 data selection and split seed.
Upstream revision/dataset hashes and selected checkpoint epoch should be recorded in the report.

See `docs/EXPERIMENT_PROTOCOL.md`.

## Combined dataset (recommended)

The shared file `fhcal_data_muons_81619_large.npz` contains the three beam
scenarios. One conditional model is trained across all three, not one model per
module. `flow_pipeline/core.py` classifies module IDs from `ParticleMomentum`
against explicitly configured reference directions and stops if unmatched.
The split is seeded 80/10/10 per module. Normalization uses train only.

On Charisma: `export FHCAL_DATA_FILE=$HOME/FHCAL_GAN/datasets/raw/fhcal_data_muons_81619_large.npz`.
After changing scripts, commit/push and `git pull` on the cluster.
The scripts also accept the legacy three-file arguments for older experiments.

## First real-file validation on Charisma (CPU, NOT login node)

```bash
mkdir -p logs
export FHCAL_MAX_PER_MODULE=60
sbatch slurm/check_data.sbatch
# Once completed inspect logs/fhcal-check-data-JOBID.out
```

Even with a 60-event pilot the compressed NPZ `EnergyDeposit` array is
decompressed once to read selected events, so the check can use several GB
of actual RAM. Do not execute it interactively on the login node.
