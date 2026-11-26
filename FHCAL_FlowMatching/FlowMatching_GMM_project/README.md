# FHCal_GMM parameter Flow Matching — FHCAL 8/16/19

**Location in shared repository:** `FHCAL_GAN/FlowMatching_GMM_project/`.
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

4-block conditional residual MLP trained with independent Flow Matching on 27 fitted parameters of the ORIGINAL four-component FHCal_GMM plus log1p(total central energy). Original upstream projector and packet sampler are imported at runtime.

## Project files

```text
FlowMatching_GMM_project/
├── run_pipeline.py            # unified entry point: train / evaluate / check_data / fit / prepare_combined
├── flow_pipeline/             # data, neural network, fitting, metrics, evaluation
├── config/                    # example Charisma environment variables
├── data/                      # uncommitted NPZ data (or FHCAL_DATA_DIR)
├── experiments/               # uncommitted checkpoints and fit banks
├── results/                   # uncommitted plots and CSV/JSON metrics
├── slurm/                     # SBATCH jobs
├── tests/                     # CPU smoke tests
├── docs/                      # comparison protocol
└── README.md
```

## Local checks (inside THIS folder)

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
python run_pipeline.py check_data --data8 /path/to/muons_8.npz \
  --data16 /path/to/muons_16.npz --data19 /path/to/muons_19.npz \
  --max-per-module 600
```

If data files contain all three directions in one NPZ, use `python run_pipeline.py prepare_combined --help`.
You must check beam directions against those encoded in `flow_pipeline/core.py`; do not silently relabel beams.
Don't substitute earlier proton-only files for 16/19.

## Slurm on Charisma (start from the shared repository root)

```bash
cd FlowMatching_GMM_project
mkdir -p logs
# Optional: cp config/charisma.env.example config/charisma.env && edit config/charisma.env
# Optional: set -a; source config/charisma.env; set +a
export FHCAL_DATA_DIR=/absolute/path/to/three_npz
export FHCAL_VENV=$HOME/.venvs/fhcal-flow
export FHCAL_MAX_PER_MODULE=600  # pipeline pilot, not full dataset
bash scripts/get_reference.sh
fit_job=$(sbatch --parsable slurm/fit.sbatch)
train_job=$(sbatch --parsable --dependency=afterok:$fit_job slurm/train.sbatch)
sbatch --dependency=afterok:$train_job slurm/evaluate.sbatch
squeue -u "$USER"
```

**Important:** GPU partition `gpu-ef-quick` is PROVISIONAL; first inspect `sinfo -o '%P %l %G'`
and adjust `#SBATCH --partition`, resource requests, and virtual environment on Charisma.
Training is not run on the login node.

Pilot defaults: 600 selected examples per module, 100 epochs and 20-step Euler during evaluation.
Full data: set `FHCAL_MAX_PER_MODULE=0`, use NEW experiment output paths, and match the setting
and split seed across both projects. Evaluation produces per-module and pooled distributions,
KS, p-value (with zero-mass caveat), JS, occupancy, profile and throughput.

## Original GMM dependency

For the ORIGINAL author code, `scripts/get_reference.sh` fetches upstream `voboch/FHCal_GMM` pinned at `84fd872c03de9ed4bd03010c894150e4d8042bbd`. External files are not included in this repository. If GitHub access fails on Charisma, unpack your user-provided upstream ZIP at `reference_FHCal_GMM/` and set `FHCAL_GMM_REFERENCE` to its root. `fit_codes.py` fits 4-component mixtures to TRAIN and VALIDATION only; test events are held out. The CPU fit step can be very expensive at full statistics. `--packets 128` uses the original stochastic packet decoder.


## Scientific caveat

Both projects have been smoke-tested on synthetic inputs only; no full three-module GPU training
is claimed. Comparison is valid only on the SAME central Geant4 data selection and split seed.
Upstream revision/dataset hashes and selected checkpoint epoch should be recorded in the report.

See `docs/EXPERIMENT_PROTOCOL.md`.
