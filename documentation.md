# Code Documentation: cth_node_complete.py
### Line-by-line explanation of every decision

---

## CELL 1 — Imports

```python
import torch
import torch.nn as nn
```
PyTorch is the deep learning framework. `torch.nn` provides building blocks
(Linear, LayerNorm, LeakyReLU, etc.) and the base class `nn.Module` that all
model components inherit from.

```python
import numpy as np
```
NumPy is used for pre-processing data (adjacency matrix construction, hypergraph
incidence matrix) before converting to PyTorch tensors. CPU-only operations that
don't need autograd.

```python
import os, copy, urllib.request
```
- `os`: check if data files already exist before downloading.
- `copy`: `copy.deepcopy(state_dict)` saves the best model weights without
  reference aliasing (if we just assigned `best_state = m.state_dict()`, the
  dict would update in-place as training continues).
- `urllib.request`: download PEMS04 files from Zenodo without needing `wget`.

```python
import pandas as pd
```
Read the PEMS04 edge list CSV (columns: from_node, to_node, distance). Pandas
`iterrows()` is used to convert the edge list into a distance matrix.

```python
import matplotlib.pyplot as plt
```
Plot training curves, evaluation figures, and save PNGs for the thesis.

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
Automatically use GPU if available, otherwise CPU. All tensors and model
parameters are moved to this device with `.to(device)`.

---

## CELL 2 — Load Data + Road-Network Adjacency

### Data loading

```python
url_npz = "https://zenodo.org/records/7816008/files/PEMS04.npz?download=1"
```
Public Zenodo mirror of PEMS04. The `.npz` file contains speed, flow, and
occupancy readings for 307 sensors at 5-minute intervals.

```python
if not os.path.exists(filename_npz):
    urllib.request.urlretrieve(url_npz, filename_npz)
```
Only download if the file isn't already on disk — avoids re-downloading on every
notebook restart.

```python
raw_data = data['data'][:, :307, 2]
```
The NPZ `'data'` array has shape `[T, N, 3]` where the last dimension is
`[speed, flow, occupancy]`. Index 2 = speed channel. We take all 307 nodes
(`[:307]` is defensive in case the file has extra padding).

```python
raw_data = np.nan_to_num(raw_data)
```
Replace any NaN sensor dropouts with 0 before normalisation. If left as NaN,
they would propagate through the normalisation and poison the entire dataset.

```python
mean, std = raw_data.mean(), raw_data.std()
data_norm = (raw_data - mean) / (std + 1e-8)
```
Z-score normalisation. The 1e-8 prevents division by zero if std is somehow 0.
All loss, MAE, and threshold comparisons operate in normalised space; results are
de-normalised for display with `* std + mean`.

### Adjacency matrix

```python
dist_mat = np.full((NUM_NODES, NUM_NODES), np.inf)
np.fill_diagonal(dist_mat, 0.0)
for _, row in df.iterrows():
    i, j, d = int(row.iloc[0]), int(row.iloc[1]), float(row.iloc[2])
    dist_mat[i, j] = d
    dist_mat[j, i] = d
```
Build a symmetric distance matrix from the edge list. `np.inf` for non-connected
pairs means "no road between these sensors". The diagonal is 0 (a sensor has
distance 0 to itself).

```python
sigma = dist_mat[dist_mat < np.inf].std()
adj   = np.where(dist_mat < np.inf, np.exp(-(dist_mat**2) / (sigma**2)), 0.0)
```
**Gaussian kernel affinity**: connected pairs get weight `exp(-d²/σ²)` where σ
is the standard deviation of all finite distances. This maps distance to [0,1]
similarity — close sensors have high weight, distant sensors have low weight.
The choice of σ² (instead of σ) is standard in the DCRNN/WaveNet literature.

```python
np.fill_diagonal(adj, 1.0)
```
Self-loops: every node is its own strongest neighbour. Needed so the GAT
attention doesn't ignore the node's own hidden state when aggregating.

```python
deg = adj.sum(axis=1)
d_inv = np.where(deg > 0, deg**(-0.5), 0.0)
adj_norm = (adj * d_inv[:, None]) * d_inv[None, :]
```
**Symmetric normalisation** D^{-1/2} A D^{-1/2}. This ensures that the
aggregation is scale-invariant: high-degree nodes (many neighbours) don't
dominate the message passing. The `np.where` guard handles isolated nodes
with degree 0.

```python
A_road = torch.tensor(adj_norm, ...).unsqueeze(0).to(device)
```
Shape `[1, N, N]` — the leading batch dimension of 1 will be broadcast to
the actual batch size B inside the GAT `forward()`.

### Graph Laplacian

```python
L_sym_np = np.eye(NUM_NODES, dtype=np.float32) - adj_norm
L_graph  = torch.tensor(L_sym_np, ...).to(device)
```
**Symmetric normalised Laplacian**: L = I − D^{-1/2} A D^{-1/2}.
Used in the physics loss: `||L · v||²` = sum of squared differences between
each node's predicted speed and its adjacency-weighted neighbour mean. This
encodes the LWR principle that speed should vary smoothly along roads.

### Hypergraph construction

```python
adj_binary = (adj > 1e-6).astype(np.float32)
np.fill_diagonal(adj_binary, 0.0)
```
Binarise adjacency (1 = edge exists, 0 = no edge). Remove self-loops before
computing 2-hop reachability (otherwise every node trivially reaches itself).

```python
adj2 = adj_binary @ adj_binary
adj2 = (adj2 > 0).astype(np.float32)
adj2 = np.clip(adj2 + adj_binary, 0, 1)
np.fill_diagonal(adj2, 1.0)
```
`adj_binary @ adj_binary` produces the 2-hop reachability matrix: entry [i,j]
counts paths of length 2 between i and j. We binarise it (>0), union with the
1-hop adjacency, and add self-loops. Result: for node i, the row `adj2[i,:]`
flags all nodes within 2 hops including i itself — this is the hyperedge for i.

```python
H_np = adj2.T
```
Transpose so columns are hyperedges: `H_np[:, e]` = membership vector of
hyperedge e. Each column is the 2-hop neighbourhood of one sensor.

```python
d_v = H_np.sum(axis=1)   # node degree = how many hyperedges contain node i
d_e = H_np.sum(axis=0)   # hyperedge degree = how many nodes in hyperedge e
```
These are the diagonal entries of D_v and D_e in the HGNN formula.

```python
H_conv_np = (d_v_inv_sqrt[:, None] * H_np) * d_e_inv[None, :]
H_conv_np = H_conv_np @ (H_np.T * d_v_inv_sqrt[None, :])
```
Pre-compute the full normalised propagation operator:
```
H_conv = D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2}
```
Breaking it down:
- `d_v_inv_sqrt[:, None] * H_np` = D_v^{-1/2} H  (scale each row by 1/√deg_v)
- `* d_e_inv[None, :]` = · D_e^{-1}              (scale each col by 1/deg_e)
- `@ (H_np.T * d_v_inv_sqrt[None, :])` = · H^T D_v^{-1/2}

This is done **once** at construction time so the runtime convolution is just
`H_conv @ (X · Θ)` — a single matmul, no re-computing the normalisation.

---

## CELL 3 — Sensor Mask + Input Features

```python
sparsity_ratio = 0.80
torch.manual_seed(42)
node_mask = (torch.rand(1, NUM_NODES, 1, 1) > sparsity_ratio).float().to(device)
```
Randomly mark 80% of nodes as blind (mask = 0). Shape `[1, N, 1, 1]` — the 1s
on dims 2 and 3 allow broadcasting over time T and feature F dimensions.
`seed=42` makes the blind node selection reproducible.

```python
data_tensor = torch.tensor(data_norm, ...).T.unsqueeze(0).unsqueeze(-1).to(device)
```
`data_norm` has shape `[T, N]` (time × nodes). `.T` → `[N, T]`, then two unsqueezes
give `[1, N, T, 1]` matching the `[B, N, T, F]` convention.

```python
obs_data = data_tensor * node_mask
```
Zeros out blind nodes. `node_mask` broadcasts: shape `[1,N,1,1]` × `[1,N,T,1]`
= `[1,N,T,1]`. Observed nodes keep their true speed; blind nodes become 0.

### Global context feature

```python
num_obs = node_mask.sum(dim=1, keepdim=True)
network_context = obs_data.sum(dim=1, keepdim=True) / (num_obs + 1e-6)
```
Sum over the N dimension to get the total observed speed at each timestep, divide
by the number of observed nodes. Result shape `[1,1,T,1]` — one scalar per timestep.
`keepdim=True` preserves the dimension for broadcasting. 1e-6 prevents divide-by-zero.

### Neighbourhood context feature

```python
A_t    = torch.tensor(adj_norm, ...).to(device)   # [N, N]
obs_2d = obs_data[0, :, :, 0]                     # [N, T]
mask_1d = node_mask[0, :, 0, 0]                   # [N]
mask_2d = mask_1d.unsqueeze(1).expand_as(obs_2d)  # [N, T]
nbr_sum = torch.mm(A_t, obs_2d)                   # [N, T]
nbr_cnt = torch.mm(A_t, mask_2d)                  # [N, T]
nbr_ctx = (nbr_sum / (nbr_cnt + 1e-6)).unsqueeze(0).unsqueeze(-1)
```
For each node i at time t:
`nbr_ctx[i,t] = Σ_j A[i,j]*obs_j(t) / Σ_j A[i,j]*mask_j`
= adjacency-weighted mean of *observed* neighbour speeds.

`mask_2d` broadcasts `mask_1d` over time (all timesteps have the same sensor mask).
`A_t @ mask_2d` counts the total observed-node adjacency weight for each node.
No GT leakage: only `obs_data` (with blind nodes zeroed) enters the sum.

### Temporal features

```python
STEPS_PER_DAY = 288   # 24h × 60min / 5min per sample = 288
t_idx = torch.arange(T_full, ...)
t_sin = torch.sin(2 * np.pi * (t_idx % STEPS_PER_DAY) / STEPS_PER_DAY)
t_cos = torch.cos(2 * np.pi * (t_idx % STEPS_PER_DAY) / STEPS_PER_DAY)
```
Cyclic encoding of time-of-day. Using both sin and cos avoids the ambiguity of
using only one (sin alone can't distinguish morning rush from evening rush).
`% STEPS_PER_DAY` extracts intra-day position.

```python
TIME_SCALE = 0.25
```
Scale down temporal features. At full amplitude (scale = 1.0), the model
learned to predict congestion at 7:40am and 7pm from time-of-day alone, generating
false jams at rush hours even for free-flowing blind nodes. Scaling to 0.25
retains the daily periodicity signal while preventing it from dominating over
the spatial evidence.

### Leakage checks

```python
assert (input_features[0, node_mask[0,:,0,0]==0, :, 0] == 0).all()
assert (input_features[0, node_mask[0,:,0,0]==0, :, 3] == 0).all()
```
These assertions will crash immediately if:
1. Feature 0 (obs_speed) is non-zero for blind nodes — ground truth leaked.
2. Feature 3 (is_observed) is 1 for blind nodes — observation flag leaked.
This acts as a guard against any future code change that accidentally introduces leakage.

---

## CELL 4 — Model Definition

### GraphAttention

```python
self.W     = nn.Linear(in_dim, out_dim, bias=False)
self.a_src = nn.Linear(out_dim, 1, bias=False)
self.a_dst = nn.Linear(out_dim, 1, bias=False)
```
`W` projects node features into the attention space.
`a_src` and `a_dst` produce scalar attention logits from source and destination
node features separately. Their sum (before the non-linearity) gives the edge
score — this additive decomposition avoids the O(N² × H) matmul that the full
concatenation form would require.

```python
self.leaky = nn.LeakyReLU(0.2)
```
LeakyReLU with negative slope 0.2 (as in the original GAT paper). Regular ReLU
would kill negative attention scores; leaky allows small gradients for scores
below zero so inactive edges can become active during training.

```python
self.temperature = temperature   # default 2.0
```
Dividing logits by τ > 1 before softmax flattens the attention distribution.
Without temperature, the model often places ~100% weight on one neighbour (the
closest congested node), causing the ODE to over-amplify that node's signal
at every step — leading to oscillation after jams clear.

```python
e = self.leaky(e_src + e_dst.transpose(1, 2))
```
`e_src` shape `[B, N, 1]`, `e_dst.transpose(1,2)` shape `[B, 1, N]`.
Addition broadcasts to `[B, N, N]` where `[b, i, j]` = edge score from i to j.

```python
e = e.masked_fill(A < 1e-9, float('-inf'))
```
Non-edges (A[i,j] ≈ 0) are masked to −∞ so they contribute zero after softmax.
This enforces the road topology: only actual road connections carry information.

```python
alpha = torch.softmax(e / self.temperature, dim=-1)
alpha = torch.nan_to_num(alpha, 0.0)
```
`softmax(..., dim=-1)` normalises attention weights over all neighbours of node i.
`nan_to_num` handles isolated nodes (all neighbours masked → all −∞ → nan after
softmax). Such nodes get zero weight — they propagate nothing and receive nothing.

### HypergraphConv

```python
def forward(self, x, H_conv):
    h = self.theta(x)              # [B, N, out_dim]
    return torch.matmul(H_conv, h) # [B, N, out_dim]
```
`H_conv` is `[N, N]` (pre-computed at construction time).
`torch.matmul` broadcasts over the batch dimension: `[N,N] @ [B,N,H]` gives
`[B,N,H]`. Each output node i receives the normalised sum of all hyperedge
members' projected features — a group-level aggregation.

### GraphODEFunc

```python
self.hyper_gate = nn.Parameter(torch.tensor(-2.0))
```
A single learnable scalar. `sigmoid(-2) ≈ 0.12` — the gate starts nearly closed.
The model learns to open it on hyperedge paths that improve prediction.
Starting at 0 (sigmoid(0) = 0.5) gave equal initial weight to GAT and HGNN,
causing over-smoothing at jam nodes before any useful gradient had propagated.

```python
def forward(self, t, x):
    A   = self.A.expand(x.size(0), -1, -1)   # [1,N,N] → [B,N,N]
    h   = self.act(self.gat1(x, A))
    h   = self.act(self.gat2(h, A))
    if self.hconv is not None:
        h_hyp = self.act(self.hconv(x, self.H_conv))
        h     = h + torch.sigmoid(self.hyper_gate) * h_hyp
    delta = self.norm(h)
    return delta    # derivative only — Euler adds residual
```
The function returns the *derivative* dz/dt, not the next state z.
The Euler step `z + 0.3 * f(z)` adds the residual. Returning z + delta here
(the old bug) would mean Euler computed `z + (z+delta) = 2z + delta`, doubling
the hidden state every step until it exploded (~10^24 loss).

`LayerNorm` normalises the *delta* only (not delta + z). Normalising the sum
would suppress the skip signal, losing the ODE residual structure.

### AssimilationUpdate

```python
z_obs  = self.obs_encoder(x_new)               # encode new sensor reading
gate   = self.gate(torch.cat([z, z_obs], -1))  # how much to update
update = gate * (z_obs - z) * obs_mask         # correction term
return z + update
```
This is a GRU-style update rule. The gate is a sigmoid network that takes both
the current hidden state `z` and the new encoded observation `z_obs` as input,
learning how much to trust the sensor reading vs the ODE prediction.

`* obs_mask` zeros the update for blind nodes (mask = 0). Without this, blind
nodes would "assimilate" their own zero speed observations, pulling their hidden
state toward 0 at every step (the old leakage bug).

### GraphCTH_NODE

```python
def _euler_step(self, z):
    return z + 0.3 * self.ode_func(None, z)
```
`dt = 0.3` was chosen empirically. At `dt = 1.0`, the hidden state accumulated
too much momentum: after a jam event cleared, the ODE continued predicting low
speeds for many timesteps, then over-corrected and oscillated at 78 km/h.
Smaller dt = smaller per-step change = smoother post-jam recovery.

```python
def forward(self, x_seq, obs_mask):
    mask = obs_mask[:, :, 0, :]   # [B, N, 1]
    z    = self.encoder(x_seq[:, :, 0, :])
    for i in range(T):
        preds.append(self.decoder(z))   # current prediction
        if i < T - 1:
            z = self._euler_step(z)     # advance
            z = self.assimilate(z, x_seq[:, :, i+1, :], mask)  # correct
```
The loop alternates: decode → step → assimilate → decode → ...
We decode *before* the Euler step so the first prediction uses the initial
encoded state (not a stepped state). This matches the teacher-forcing regime
where we inject actual observations at `i+1`.

### ObservedMSELoss

```python
w = torch.where(obs < self.jam_thresh,
                torch.full_like(obs, self.jam_weight),  # 4.0 for jams
                torch.ones_like(obs))
loss_obs = torch.mean(((pred - obs) * sup_mask) ** 2 * w)
```
`sup_mask` selects which nodes contribute to the loss: observed nodes + any
pseudo-blind nodes created by curriculum masking. Blind nodes are excluded unless
explicitly included by the curriculum.

The jam weight 4.0 compensates for class imbalance: jams are ~8% of timesteps.
Without this weight, the model minimises total MSE by ignoring congestion entirely.

```python
loss_smooth = torch.mean((pred[:, :, 1:] - pred[:, :, :-1]) ** 2)
```
Mean squared difference between consecutive predicted timesteps. λ=0.60 is
relatively strong — it's needed to suppress the oscillation artefact where the
model alternates between over- and under-predicting jam speeds.

```python
v = pred[0, :, :, 0]   # [N, T]
loss_phy = torch.mean(torch.mm(self.L, v) ** 2)
```
`L @ v`: for each column (timestep), the Laplacian times the speed vector gives
`(v_i − mean_nbr(v_i))` for each node i. Squaring and summing penalises predictions
that violate spatial continuity. λ=0.02 keeps this as a soft regulariser — the
model can still predict localised jams, just pays a small cost for sharp spatial
boundaries.

---

## CELL 5 — Training

```python
CURRICULUM_DROP = 0.15
```
At each batch, 15% of observed nodes are pseudo-blinded. This was tuned: lower
values (5%) provide insufficient gradient through the blind path; higher values
(30%) degrade observed-node performance as the model receives too little direct
supervision.

```python
jam_t_valid = has_jam.nonzero(as_tuple=True)[0]
jam_t_valid = jam_t_valid[jam_t_valid < TRAIN_END - BATCH_TIME]
```
Pre-compute all timestep starts where at least one blind node has speed < 40 km/h.
The second line removes windows that would run past the training boundary.

```python
if len(jam_t_valid) > 0 and np.random.rand() < 0.5:
    t0 = int(jam_t_valid[torch.randint(len(jam_t_valid), (1,)).item()])
else:
    t0 = np.random.randint(0, TRAIN_END - BATCH_TIME)
```
50% jam-biased sampling. Random uniform sampling would see jams in ~8% of batches.
Even with 4× loss weight, the gradient variance between jam and free-flow batches
is too high. Forcing 50% jam windows approximately balances the loss contributions.

```python
ACCUM_STEPS = 4
for _ in range(ACCUM_STEPS):
    ...
    step_loss = criterion(...) / ACCUM_STEPS
    step_loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```
Gradient accumulation over 4 windows. Dividing by ACCUM_STEPS is critical —
without it the effective learning rate would be 4× too large. Clipping at norm 1.0
prevents a single catastrophic gradient update from destroying trained weights.

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400)
```
T_max=400 with 800 epochs = two cosine cycles. The second cycle starts from the
initial LR again, which can help the model escape local minima found in the first
cycle. This is sometimes called "warm restarts" (SGDR, Loshchilov & Hutter 2017).

---

## CELL 6 — Evaluation

```python
EVAL_START = 4500
EVAL_LEN   = 450
EVAL_WIN   = BATCH_TIME   # 48
```
The evaluation window starts 500 timesteps after training ends (buffer), avoiding
any indirect temporal leakage. Windows of size 48 match the training window —
if evaluation used a single 450-step rollout, ODE state drift would accumulate
and predictions would degrade towards the end of the window. Using 48-step
chunks and concatenating prevents this.

---

## CELL 8 — Sensor Sparsity Sweep

```python
_SP_EPOCHS = 150
_SP_HIDDEN = 32
_SP_ACCUM  = 2
```
Reduced from the main training (800 epochs, hidden=64, accum=4) for speed.
The sweep is intended to show relative performance ranking across sparsity
levels, not absolute MAE — smaller models converge to approximately the same
ranking as full-size models in fewer epochs.

```python
def _idw_eval(feats_sp, mask_sp):
    p_chunks.append(feats_sp[:,:,t:t+EVAL_WIN,2:3].cpu() * std + mean)
```
IDW prediction uses `feature index 2 = nbr_ctx`, which was pre-computed as the
adjacency-weighted mean of observed neighbours. This is exactly the IDW formula —
no model, no training, pure spatial interpolation.

---

## CELL 9 — Ablation Study

```python
class _GraphCTH_NoAssim(GraphCTH_NODE):
    def forward(self, x_seq, obs_mask):
        z = self.encoder(x_seq[:, :, 0, :])
        for i in range(T):
            preds.append(self.decoder(z))
            if i < T - 1:
                z = self._euler_step(z)   # no assimilation call
```
The no-assimilation variant keeps all weights and the ODE dynamics but removes
the `self.assimilate(...)` call. The ODE runs freely without any sensor correction
after the initial encoding step.

```python
ablation_configs = [
    ("Full model",          GraphCTH_NODE,     input_features,   0.02, True),
    ("− Hypergraph",        GraphCTH_NODE,     input_features,   0.02, False),
    ("− Assimilation",      _GraphCTH_NoAssim, input_features,   0.02, True),
    ("− Physics loss",      GraphCTH_NODE,     input_features,   0.00, True),
    ("− Neighbour context", GraphCTH_NODE,     feats_no_nbr,     0.02, True),
    ("− Temporal encoding", GraphCTH_NODE,     feats_no_time,    0.02, True),
]
```
Each variant changes exactly one thing:
- `"− Hypergraph"`: passes `H_conv=None` → `GraphODEFunc` skips HGNN branch.
- `"− Physics loss"`: sets `lambda_physics=0.0` → Laplacian term multiplied by 0.
- `"− Neighbour context"`: sets feature 2 (nbr_ctx) to zero in the input tensor.
- `"− Temporal encoding"`: sets features 4–5 (t_sin, t_cos) to zero.

```python
delta = f"{full_jv - jv:>+.2f}"
```
Δ jam = full_jv − variant_jv.
**Positive**: removing this component raises jam MAE → the component HELPS.
**Negative**: removing this component lowers jam MAE → the component HURTS
(investigate over-smoothing or feature conflict).
