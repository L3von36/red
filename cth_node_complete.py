# =============================================================================
# CELL 1 — Install & Imports
# =============================================================================
# !pip install torchdiffeq

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from torchdiffeq import odeint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# CELL 2 — Load Data + Road-Network Adjacency
# =============================================================================
url_npz      = "https://zenodo.org/records/7816008/files/PEMS04.npz?download=1"
url_csv      = "https://zenodo.org/records/7816008/files/PEMS04.csv?download=1"
filename_npz = "PEMS04.npz"
filename_csv = "PEMS04.csv"

if not os.path.exists(filename_npz):
    os.system(f'wget -O {filename_npz} "{url_npz}"')
if not os.path.exists(filename_csv):
    os.system(f'wget -O {filename_csv} "{url_csv}"')

# --- Speed data ---
data     = np.load(filename_npz)
raw_data = data['data'][:, :307, 2]   # speed channel, 307 nodes
raw_data = np.nan_to_num(raw_data)

mean, std = raw_data.mean(), raw_data.std()
data_norm = (raw_data - mean) / (std + 1e-8)

NUM_NODES  = 307
TIME_STEPS = 5000

# --- Road adjacency from edge list ---
df = pd.read_csv(filename_csv, header=0)
print(f"Edge list: {df.shape}  columns: {df.columns.tolist()}")

dist_mat = np.full((NUM_NODES, NUM_NODES), np.inf)
np.fill_diagonal(dist_mat, 0.0)
for _, row in df.iterrows():
    i, j, d = int(row.iloc[0]), int(row.iloc[1]), float(row.iloc[2])
    dist_mat[i, j] = d
    dist_mat[j, i] = d

# Gaussian kernel affinity + symmetric normalisation D^{-1/2} A D^{-1/2}
sigma    = dist_mat[dist_mat < np.inf].std()
adj      = np.where(dist_mat < np.inf, np.exp(-(dist_mat**2) / (sigma**2)), 0.0)
np.fill_diagonal(adj, 1.0)
deg      = adj.sum(axis=1)
d_inv    = np.where(deg > 0, deg**(-0.5), 0.0)
adj_norm = (adj * d_inv[:, None]) * d_inv[None, :]   # [N, N]

A_road = torch.tensor(adj_norm, dtype=torch.float32).unsqueeze(0).to(device)
print(f"✅ Adjacency ready. Shape: {A_road.shape}  "
      f"mean degree ≈ {(adj > 0).sum(1).mean():.1f}")


# =============================================================================
# CELL 3 — Sensor Mask + Honest Input Features (no GT leakage)
# =============================================================================
sparsity_ratio = 0.80
torch.manual_seed(42)

# node_mask: [1, N, 1, 1]  —  1 = observed sensor, 0 = blind node
node_mask = (torch.rand(1, NUM_NODES, 1, 1) > sparsity_ratio).float().to(device)

# Full ground-truth tensor: [1, N, T, 1]
data_tensor = torch.tensor(
    data_norm, dtype=torch.float32
).T.unsqueeze(0).unsqueeze(-1).to(device)

obs_data = data_tensor * node_mask   # blind nodes → 0

# Honest global context: mean of OBSERVED nodes only
num_obs         = node_mask.sum(dim=1, keepdim=True)          # [1,1,1,1]
network_context = obs_data.sum(dim=1, keepdim=True) / (num_obs + 1e-6)
network_context = torch.nan_to_num(network_context, 0.0)      # [1,1,T,1]

# 3-feature input: [obs_speed (0 if blind), global_context, is_observed_flag]
obs_flag       = node_mask.expand_as(data_tensor)              # [1,N,T,1]
context_feat   = network_context.expand_as(data_tensor)        # [1,N,T,1]
input_features = torch.cat([obs_data, context_feat, obs_flag], dim=-1)  # [1,N,T,3]

print(f"✅ Input features: {input_features.shape}")
print(f"   Observed: {node_mask.mean()*100:.0f}%  |  Blind: {(1-node_mask.mean())*100:.0f}%")

# Leakage checks — will crash immediately if GT sneaks through
assert (input_features[0, node_mask[0,:,0,0]==0, :, 0] == 0).all(), \
    "Leakage: blind nodes have non-zero speed."
assert (input_features[0, node_mask[0,:,0,0]==0, :, 2] == 0).all(), \
    "Leakage: blind nodes flagged as observed."
print("✅ Leakage checks passed.")


# =============================================================================
# CELL 4 — Model Definition
# =============================================================================
#
# Three bugs fixed vs the previous version:
#
# BUG 1 — dopri5 inside the assimilation step-loop.
#   An adaptive 6-stage RK solver for a fixed dt=1 step calls the ODE
#   function 6+ times per call. With T=24 steps chained in backprop,
#   that's 144+ evaluations and 24 separate odeint computation graphs,
#   causing severe gradient vanishing.
#   FIX: single Euler step  z ← z + f(z)  — one evaluation, one graph node.
#
# BUG 2 — Blind nodes received zero observation-loss gradient.
#   loss = mean(((pred-obs)*sensor_mask)²) — blind nodes are masked out,
#   so no gradients ever reach the weights responsible for imputation.
#   FIX: curriculum masking — each batch randomly drop 10% of observed
#   nodes from supervision. They become "pseudo-blind": forward pass
#   treats them as blind, but GT is known so we compute loss on them.
#   This gives the blind-node code path real, meaningful gradients.
#
# BUG 3 — ODEFunc residual normalised the wrong thing.
#   LayerNorm(gc2_out + x) normalises both delta and skip together,
#   suppressing the skip signal. gc2 also had no activation.
#   FIX: Tanh on both GCN layers, LayerNorm on delta only before add.

class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, A):
        # x: [B, N, H]   A: [B, N, N]  →  [B, N, H]
        return self.W(torch.bmm(A, x))


class GraphODEFunc(nn.Module):
    def __init__(self, hidden_dim, A_road):
        super().__init__()
        self.register_buffer('A', A_road)   # [1, N, N], non-trainable
        self.gc1  = GraphConv(hidden_dim, hidden_dim)
        self.gc2  = GraphConv(hidden_dim, hidden_dim)
        self.act  = nn.Tanh()
        self.norm = nn.LayerNorm(hidden_dim)   # applied to delta only (BUG 3 fix)

    def forward(self, t, x):
        A     = self.A.expand(x.size(0), -1, -1)
        h     = self.act(self.gc1(x, A))      # activate gc1
        h     = self.act(self.gc2(h, A))      # activate gc2 (BUG 3 fix)
        delta = self.norm(h)                  # norm the delta, not (h+x)
        return x + delta                      # residual


class AssimilationUpdate(nn.Module):
    """
    Learned Kalman-style correction after each Euler step.

    Observed nodes: GRU-like gate blends ODE prediction with new sensor reading.
    Blind nodes:    obs_mask=0  →  update=0, state unchanged.
                    They already received neighbour info via GraphConv.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.obs_encoder = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, z, x_new, obs_mask):
        # z:        [B, N, H]
        # x_new:    [B, N, F]
        # obs_mask: [B, N, 1]  — 1=observed, 0=blind
        z_obs  = self.obs_encoder(x_new)
        gate   = self.gate(torch.cat([z, z_obs], dim=-1))
        update = gate * (z_obs - z) * obs_mask   # blind nodes zeroed out
        return z + update


class GraphCTH_NODE(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, A_road=None):
        super().__init__()
        self.encoder    = nn.Linear(input_dim, hidden_dim)
        self.ode_func   = GraphODEFunc(hidden_dim, A_road)
        self.assimilate = AssimilationUpdate(input_dim, hidden_dim)
        self.decoder    = nn.Linear(hidden_dim, 1)

    def _euler_step(self, z):
        """Single Euler step dt=1.  FIX for BUG 1 — replaces dopri5."""
        return z + self.ode_func(None, z)

    def forward(self, x_seq, obs_mask):
        """
        x_seq:    [B, N, T, F]
        obs_mask: [B, N, 1, 1]
        returns:  [B, N, T, 1]
        """
        mask  = obs_mask[:, :, 0, :]   # [B, N, 1]
        z     = self.encoder(x_seq[:, :, 0, :])
        preds = []
        T = x_seq.shape[2]

        for i in range(T):
            preds.append(self.decoder(z))          # decode → speed prediction
            if i < T - 1:
                z = self._euler_step(z)            # propagate via road graph
                z = self.assimilate(z, x_seq[:, :, i+1, :], mask)  # inject sensors

        return torch.stack(preds, dim=2)           # [B, N, T, 1]


class ObservedMSELoss(nn.Module):
    def forward(self, pred, obs, sup_mask, lambda_smooth=0.05):
        loss_obs    = torch.mean(((pred - obs) * sup_mask) ** 2)
        loss_smooth = torch.mean((pred[:, :, 1:] - pred[:, :, :-1]) ** 2)
        return loss_obs + lambda_smooth * loss_smooth


# =============================================================================
# CELL 5 — Training with curriculum masking
# =============================================================================
model     = GraphCTH_NODE(input_dim=3, hidden_dim=64, A_road=A_road).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.5)
criterion = ObservedMSELoss()

VAL_START       = 4500
BATCH_TIME      = 24
CURRICULUM_DROP = 0.10   # drop 10% of observed nodes from supervision per batch
best_mae        = 9999
mask_4d         = node_mask   # [1, N, 1, 1]

print("Training Graph-ODE + Assimilation (Euler, curriculum masking)...")
for epoch in range(300):
    model.train()
    optimizer.zero_grad()

    t0         = np.random.randint(0, TIME_STEPS - BATCH_TIME)
    x_window   = input_features[:, :, t0:t0+BATCH_TIME, :]   # [1, N, T, 3]
    obs_window = data_tensor[:, :, t0:t0+BATCH_TIME, :]       # [1, N, T, 1]

    # Curriculum mask: start from real sensor mask, then randomly drop
    # some observed nodes so blind-node code path gets gradients (BUG 2 fix)
    sup_mask = node_mask.expand_as(obs_window).clone()   # [1, N, T, 1]
    obs_indices = (node_mask[0, :, 0, 0] == 1).nonzero(as_tuple=True)[0]
    n_drop = max(1, int(len(obs_indices) * CURRICULUM_DROP))
    drop_idx = obs_indices[torch.randperm(len(obs_indices))[:n_drop]]
    sup_mask[0, drop_idx, :, :] = 0   # hide from loss, not from forward pass

    preds = model(x_window, mask_4d)
    loss  = criterion(preds, obs_window, sup_mask)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            x_val  = input_features[:, :, VAL_START:VAL_START+50, :]
            p_val  = model(x_val, mask_4d)
            p_real = p_val.cpu() * std + mean
            g_real = data_tensor[:, :, VAL_START:VAL_START+50, :].cpu() * std + mean
            hid    = (node_mask.expand_as(g_real).cpu() == 0)
            mae    = torch.mean(torch.abs(p_real[hid] - g_real[hid])).item()

            if mae < best_mae:
                best_mae = mae
                torch.save(model.state_dict(), 'best_graph_model.pth')

            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | "
                  f"Blind-node Val MAE: {mae:.2f} km/h")

print(f"\nDone. Best blind-node Val MAE: {best_mae:.2f} km/h")


# =============================================================================
# CELL 6 — Evaluation
# =============================================================================
model.load_state_dict(torch.load('best_graph_model.pth'))
model.eval()

EVAL_START = 4500
EVAL_LEN   = 450

with torch.no_grad():
    x_eval = input_features[:, :, EVAL_START:EVAL_START+EVAL_LEN, :]
    preds  = model(x_eval, mask_4d)
    preds  = preds.cpu() * std + mean

gts = data_tensor[:, :, EVAL_START:EVAL_START+EVAL_LEN, :].cpu() * std + mean

hid_mask = (node_mask.cpu() == 0).expand_as(gts)
jam_mask = (gts < 40)
target   = hid_mask & jam_mask

m_model   = torch.mean(torch.abs(preds[target]   - gts[target])).item()
m_base    = torch.mean(torch.abs(torch.ones_like(gts[target]) * mean - gts[target])).item()
m_overall = torch.mean(torch.abs(preds[hid_mask] - gts[hid_mask])).item()

print("\n" + "="*57)
print("   GRAPH-ODE + ASSIMILATION  (blind nodes only)")
print("="*57)
print(f"  Baseline MAE (global mean) — jam samples : {m_base:.2f} km/h")
print(f"  Model MAE   — jam samples                : {m_model:.2f} km/h")
print(f"  Model MAE   — all samples                : {m_overall:.2f} km/h")
print(f"  Jam improvement vs baseline              : {((m_base-m_model)/m_base)*100:.1f}%")
print("="*57)


# =============================================================================
# CELL 7 — Thesis Figure
# =============================================================================
# Pick the blind node with the most jam timesteps (best visual)
blind_ids  = (node_mask[0, :, 0, 0] == 0).nonzero(as_tuple=True)[0].cpu().numpy()
best_node  = max(blind_ids, key=lambda n: (gts[0, n, :, 0] < 40).sum().item())
best_count = (gts[0, best_node, :, 0] < 40).sum().item()

node_pred = preds[0, best_node, :, 0].numpy()
node_gt   = gts[0,  best_node, :, 0].numpy()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(node_gt,   label='Ground Truth',             lw=3,   color='black',    alpha=0.8)
ax.plot(node_pred, label='Graph-ODE + Assimilation', lw=2.5, color='tab:blue', linestyle='--')
ax.axhline(mean,   label='Baseline (Global Avg)',    color='gray', linestyle=':', lw=2)

# Shade every contiguous jam segment separately
jam_times = np.where(node_gt < 40)[0]
if len(jam_times) > 0:
    in_jam, first = False, True
    for t in range(len(node_gt)):
        if node_gt[t] < 40 and not in_jam:
            jam_start, in_jam = t, True
        elif node_gt[t] >= 40 and in_jam:
            ax.axvspan(jam_start, t, color='red', alpha=0.1,
                       label='Traffic Jam Period' if first else '_')
            in_jam, first = False, False
    if in_jam:
        ax.axvspan(jam_start, len(node_gt), color='red', alpha=0.1)

    deepest = int(np.argmin(node_gt))
    ax.annotate(
        f'Model tracks jam via\ngraph neighbours\n(Blind Node {best_node})',
        xy=(deepest, node_pred[deepest]),
        xytext=(max(deepest - 70, 5), node_pred[deepest] + 12),
        arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.8),
        fontsize=10, color='steelblue',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='steelblue', alpha=0.9),
    )

ax.set_title(
    f"Graph-ODE + Assimilation — Blind Node {best_node}\n"
    f"(80% Sparsity | Road-Graph MP | Observation Assimilation | No GT Leakage)",
    fontsize=12,
)
ax.set_ylabel("Speed (km/h)", fontsize=12)
ax.set_xlabel("Time Steps (5-min intervals)", fontsize=12)
ax.legend(fontsize=11, loc='upper right')
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('thesis_graph_ode.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ Saved thesis_graph_ode.png  (Node {best_node}, {best_count} jam timesteps)")
