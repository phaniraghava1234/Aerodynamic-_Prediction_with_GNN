# Aerodynamic Prediction with Graph Neural Networks

A Graph Neural Network surrogate for the two-dimensional incompressible
Reynolds-Averaged Navier-Stokes flow around airfoils, trained on the AirfRANS
`scarce` regime (200 training simulations, 200 held-out test simulations). The
same network provides Monte Carlo Dropout predictive uncertainty at inference
time, without requiring a separate model or retraining.

The project targets a portfolio deliverable rather than a state-of-the-art
result. The scientific interest is the behaviour of a small message-passing
model in the low-data regime and the reliability of MC Dropout as an
uncertainty signal.

## Overview

- **Data.** AirfRANS `scarce` split (Bonnet et al., arXiv:2212.07564).
  Each simulation is a point cloud of roughly 180 000 mesh nodes with per-node
  position, freestream velocity, wall distance, surface normals and RANS
  solution fields. Chord length is 1 m for every airfoil.
- **Model.** Encoder / processor / decoder. The encoder is a two-layer MLP;
  the processor is a stack of three GraphSAGE convolutions with LayerNorm on
  a graph whose edges come from a 2D Delaunay triangulation of the mesh
  nodes; the decoder is a two-layer MLP that produces four node-level
  regression outputs. Around 126 000 parameters. Dropout layers in the
  encoder and decoder MLPs supply the stochastic source used later for MC
  Dropout, so no architectural change is needed for uncertainty estimation.
- **Targets, per mesh node.** Velocity components u and v (m/s), kinematic
  pressure p/ρ (m²/s²), turbulent kinematic viscosity ν_t (m²/s).
- **Reported metrics.** Field-wise mean squared and mean absolute error on
  denormalised predictions; lift and pressure-integrated drag coefficients
  with R² against the CFD ground truth; mean predictive standard deviation
  per field and Pearson correlation between predictive standard deviation and
  absolute error.

## Repository layout

```
src/
  download_data.py       one-time AirfRANS download (~9 GB)
  inspect_data.py        quick shape and dtype check on the raw dataset
  eda.py                 baseline exploratory data analysis
  eda_extended.py        per-simulation summary statistics on the scarce split
  eda_all_1000.py        the same over the full 1000-simulation set
  eda_flow_fields.py     velocity, pressure and turbulence field images
  eda_metrics.py         metric EDA: audit, aero coefficients, geometry, correlations
  data_processing.py     converts raw simulations to PyTorch Geometric graphs
  models.py              baseline GNN definition (SAGEConv + LayerNorm)
  train.py               training loop, checkpointing, TensorBoard logging
  evaluate.py            deterministic evaluation on the test split
  evaluate_uq.py         Monte Carlo Dropout uncertainty quantification
  postprocess.py         shared utilities: de-normalisation, Cp, CL, CD, R²
tests/
  test_pipeline.py       graph construction and forward/backward pass
  test_full_pipeline.py  synthetic training to evaluation to UQ smoke test
data/
  Dataset/               raw AirfRANS (created by download_data.py; gitignored)
  processed/             PyG graph datasets and normalisation statistics
models/                  trained model checkpoints
results/
  eda/                   baseline EDA outputs
  flow_fields/           physics field images
  eda_metrics/           master metrics table and correlation figures
  runs/                  TensorBoard logs
  evaluation/            deterministic evaluation metrics and figures
  uq/                    MC Dropout uncertainty figures
docs/                    reference materials and review notes
plan.md                  execution plan for the training campaign
SUMMARY.md               one-page project summary
```

## Environment

The project uses the `gnn_surrogate` conda environment, which contains
Python 3.12, PyTorch 2.8.0 with CUDA 12.8 support, PyTorch Geometric 2.6.1
with the full C++ extension stack, `airfrans`, and the standard scientific
Python packages. TensorBoard is the only remaining requirement:

```bash
conda activate gnn_surrogate
pip install tensorboard
```

Verify CUDA is exposed to PyTorch:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `True NVIDIA GeForce RTX 2060`.

## Pipeline

Executed sequentially, from the repository root, with `gnn_surrogate`
activated:

```bash
python src/download_data.py     # one-time download and extraction (~9 GB)
python src/eda.py               # baseline EDA outputs
python src/data_processing.py   # writes train/val/test.pt and norm_stats.pt
python src/train.py             # writes models/baseline_best.pt
python src/evaluate.py          # writes results/evaluation/
python src/evaluate_uq.py       # writes results/uq/
```

Smoke tests using synthetic data (do not require the AirfRANS download):

```bash
python tests/test_pipeline.py
python tests/test_full_pipeline.py
```

## Exploratory data analysis

The EDA modules together produce per-simulation summary statistics, node-level
target and input distributions, mesh and Delaunay graph statistics, images of
the velocity, kinematic pressure and turbulent viscosity fields, aerodynamic
coefficient distributions computed from surface pressure, boundary-layer and
wake analyses, and a correlation matrix over flight parameters, integrated
aerodynamic coefficients, mesh statistics and target-field summaries. Outputs
are grouped under `results/eda/`, `results/flow_fields/` and
`results/eda_metrics/`. A review of the metric EDA and the fixes that
followed is recorded in `docs/eda_metrics_issues.md`.

## Data schema

Each raw AirfRANS simulation is delivered as a numpy array of shape (N, 12):

| Columns | Content | Unit |
|---|---|---|
| 0-1 | position x, y | m |
| 2-3 | freestream velocity U_x, U_y | m/s |
| 4 | wall distance | m |
| 5-6 | surface normals n_x, n_y (zero off the surface) | - |
| 7-8 | velocity u, v (target) | m/s |
| 9 | kinematic pressure p/ρ (target) | m²/s² |
| 10 | turbulent kinematic viscosity ν_t (target) | m²/s |
| 11 | on-airfoil flag | - |

After `data_processing.py`, each simulation becomes a PyTorch Geometric `Data`
object with:

- `x` (N, 8): min-max scaled position (2), z-scored freestream velocity and
  wall distance (3), surface normals (2), on-surface flag (1).
- `edge_index` (2, E): undirected Delaunay edges.
- `edge_attr` (E, 3): edge displacement components and length, expressed in
  the scaled position frame.
- `y` (N, 4): z-scored targets.
- `pos`, `surf`, `name`: raw position, surface mask and simulation identifier,
  retained for evaluation and plotting.

Normalisation statistics are fitted on the training split alone; the resulting
`norm_stats.pt` supplies the inverse transform needed to report predictions in
physical units.

## Assumptions and known limitations

These are deliberate modelling choices, documented so they are not treated as
defects.

1. Edges come from a 2D Delaunay triangulation of the mesh nodes. A small
   number of edges cross the airfoil interior; this is acceptable for a
   baseline and can be filtered if error maps indicate artefacts along the
   surface.
2. The boundary-condition input is a single on-surface bit. Wall distance
   carries the near-field against far-field distinction; no separate
   inlet / outlet / far-field one-hot is used.
3. Drag is integrated from surface pressure only. Viscous shear is neglected;
   the reported CD is a pressure drag and should be interpreted as such when
   compared to a viscous-inclusive CFD reference.
4. Surface points are ordered by angle around the surface centroid. This is
   valid for star-shaped NACA 4- and 5-digit contours as generated in
   AirfRANS; a strongly reflexed geometry would require a nearest-neighbour
   ordering instead.
5. Twenty simulations from the end of the 200-sim training split are held
   out as validation. The slice is fixed rather than randomised, to keep
   runs reproducible.
6. No physics-informed loss, hierarchical GNN or multi-fidelity extension is
   included. These are outside the scope of this deliverable.

## Expected outputs after a training run

- `models/baseline_best.pt`: checkpoint with the lowest validation loss.
- `results/runs/<timestamp>/`: TensorBoard scalars for training loss,
  validation loss and learning rate.
- `results/evaluation/metrics.json`: field-wise MSE and MAE, and R² with mean
  absolute and mean relative error for CL and pressure-based CD.
- `results/evaluation/*.png`: lift and drag scatter plots, and field, Cp and
  streamline comparisons for the best, median and worst test cases by
  pressure MSE.
- `results/uq/uq_metrics.json`: mean predictive standard deviation per field
  and Pearson correlation between predictive standard deviation and absolute
  error.
- `results/uq/*.png`: predictive standard deviation against absolute error
  scatter, spatial uncertainty maps, and Cp with ±2σ bands for the least and
  most uncertain test cases.
