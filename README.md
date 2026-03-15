# Graph-ODE Traffic Speed Imputation

Imputes traffic speeds at unobserved (blind) sensor nodes on the PEMS04 road
network using a Graph Neural ODE with Kalman-style assimilation.

## Overview

- **Dataset**: PEMS04 — 307 sensors, California highways, 5-min intervals
- **Task**: Recover speeds at 80% of nodes that have no sensor (blind nodes)
- **Model**: Graph-ODE propagates information along road adjacency; an
  assimilation module gates observed readings into the hidden state at each
  step (similar to a learned Kalman filter)

### Key design choices

| Component | Description |
|---|---|
| `GraphConv` | Symmetric-normalised adjacency × linear projection |
| `GraphODEFunc` | Two-layer GCN with Tanh + LayerNorm on the *delta* |
| `AssimilationUpdate` | GRU-style gate blends ODE state with fresh sensor data |
| Euler integration | Single dt=1 step — avoids gradient vanishing of adaptive solvers |
| Curriculum masking | Randomly drops 10 % of observed nodes from the loss so blind-node code paths receive gradients |

## Requirements

```
pip install -r requirements.txt
```

PyTorch with CUDA is optional but recommended for faster training.

## Usage

```bash
python cth_node_complete.py
```

The script will:
1. Download `PEMS04.npz` and `PEMS04.csv` from Zenodo if not present
2. Build the normalised road-graph adjacency matrix
3. Train for 300 epochs with curriculum masking, saving the best checkpoint
4. Print evaluation metrics (MAE on blind nodes, jam-period MAE vs. baseline)
5. Save a thesis-quality figure to `thesis_graph_ode.png`

## Results

The model is evaluated on the held-out validation window (timesteps 4500–4950).
Metrics are reported for **blind nodes only** to measure true imputation quality.

- **Baseline** — global mean speed (no model)
- **Model MAE** — all blind-node samples
- **Jam MAE** — blind nodes during congested periods (speed < 40 km/h)

## Bugs fixed vs. previous version

1. **dopri5 inside the step loop** — adaptive 6-stage RK caused gradient
   vanishing over 24 chained steps. Replaced with a single Euler step.
2. **Zero gradients on blind nodes** — loss masked out blind nodes entirely,
   so weights for imputation never learned. Fixed with curriculum masking.
3. **LayerNorm on skip connection** — normalising `(delta + x)` suppressed
   the skip signal. Fixed by norming only `delta` before the residual add.
