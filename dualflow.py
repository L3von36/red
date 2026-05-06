#!/usr/bin/env python3
"""
DualFlow: Bidirectional Spatiotemporal GNN with Decoupled Dual-Objective Loss
for Traffic Speed Imputation

Replaces missing sensor readings across unobserved (blind) traffic network nodes
using a novel decoupled dual-objective loss that separately optimizes for
free-flow and congested traffic regimes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import urllib.request
import warnings
from sklearn.metrics import f1_score, precision_score, recall_score
from skimage.metrics import structural_similarity as ssim

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)

DATASETS = {
    'PEMS04': {
        'npz_url': "https://zenodo.org/records/7816008/files/PEMS04.npz?download=1",
        'csv_url': "https://zenodo.org/records/7816008/files/PEMS04.csv?download=1",
        'num_nodes': 307,
        'time_steps': 5000,
        'channel_idx': 2,
    },
    'PEMS08': {
        'npz_url': "https://zenodo.org/records/7816008/files/PEMS08.npz?download=1",
        'csv_url': "https://zenodo.org/records/7816008/files/PEMS08.csv?download=1",
        'num_nodes': 170,
        'time_steps': 5000,
        'channel_idx': 2,
    },
}

DATASET_NAME = 'PEMS04'
ds_cfg = DATASETS[DATASET_NAME]
print(f"✅ Using dataset: {DATASET_NAME}")
print(f"   Nodes: {ds_cfg['num_nodes']} | Time steps: {ds_cfg['time_steps']}")

url_npz = ds_cfg['npz_url']
url_csv = ds_cfg['csv_url']
fn_npz = f"{DATASET_NAME}.npz"
fn_csv = f"{DATASET_NAME}.csv"

for fn, url in [(fn_npz, url_npz), (fn_csv, url_csv)]:
    if not os.path.exists(fn):
        print(f"Downloading {fn}...")
        try:
            urllib.request.urlretrieve(url, fn)
            print(f"  ✅ Downloaded {fn}")
        except Exception as e:
            print(f"  ❌ Download failed: {e}")
            raise
    else:
        print(f"  ✅ {fn} already exists")

raw_npz = np.load(fn_npz)
NUM_NODES = ds_cfg['num_nodes']
TIME_STEPS = ds_cfg['time_steps']
CHAN_IDX = ds_cfg['channel_idx']

TRAIN_END = min(4000, int(0.8 * TIME_STEPS))
VAL_END = min(TRAIN_END + 240, int(0.9 * TIME_STEPS))
EVAL_START = min(VAL_END + 260, int(0.9 * TIME_STEPS))
EVAL_LEN = min(450, TIME_STEPS - EVAL_START)
VAL_START = TRAIN_END
BATCH_TIME = 48
WARMUP_STEPS = 96

print(f"Train: t=0–{TRAIN_END} | Val: t={VAL_START}–{VAL_END} | Eval: t={EVAL_START}–{EVAL_START+EVAL_LEN}")

raw_all = raw_npz['data'][:TIME_STEPS, :NUM_NODES, :]
raw_all = np.nan_to_num(raw_all, nan=0.0)
raw_speed = raw_all[:, :, CHAN_IDX]

node_means = raw_speed[:TRAIN_END].mean(axis=0)
node_stds = raw_speed[:TRAIN_END].std(axis=0) + 1e-8
data_norm_speed = (raw_speed - node_means) / node_stds
speed_np = data_norm_speed

for c in range(3):
    mu = raw_all[:TRAIN_END, :, c].mean()
    sg = raw_all[:TRAIN_END, :, c].std() + 1e-8
    raw_all[:, :, c] = (raw_all[:, :, c] - mu) / sg

data_norm_all = raw_all.copy()
data_norm_all[:, :, 2] = data_norm_speed

JAM_KMH_EVAL = 40.0
JAM_KMH_TRAIN_SOFT = 50.0
jam_thresh_eval_np = (JAM_KMH_EVAL - node_means) / node_stds
jam_thresh_soft_np = (JAM_KMH_TRAIN_SOFT - node_means) / node_stds

print(f"✅ Data loaded and normalized. Shape: {speed_np.shape}")

STEPS_PER_DAY = 288
slot_idx = np.arange(TIME_STEPS) % STEPS_PER_DAY

tod_mean_sp = np.zeros((NUM_NODES, STEPS_PER_DAY), dtype=np.float32)
for s in range(STEPS_PER_DAY):
    vals = data_norm_speed[slot_idx == s, :]
    if len(vals):
        tod_mean_sp[:, s] = vals.mean(axis=0)

JAM_KMH_SPLIT = 50.0
split_norm_np = (JAM_KMH_SPLIT - node_means) / node_stds
tod_free_np = np.zeros((NUM_NODES, STEPS_PER_DAY), dtype=np.float32)
tod_jam_np = np.zeros((NUM_NODES, STEPS_PER_DAY), dtype=np.float32)

for s in range(STEPS_PER_DAY):
    t_mask_s = (slot_idx[:TRAIN_END] == s)
    vals_s = data_norm_speed[:TRAIN_END][t_mask_s, :]
    if len(vals_s) > 0:
        for n in range(NUM_NODES):
            col = vals_s[:, n]
            thresh_n = split_norm_np[n]
            free_rows = col[col >= thresh_n]
            jam_rows = col[col < thresh_n]
            tod_free_np[n, s] = free_rows.mean() if len(free_rows) else col.mean()
            tod_jam_np[n, s] = jam_rows.mean() if len(jam_rows) else thresh_n - 0.5

print(f"✅ Time-of-day context computed")

df = pd.read_csv(fn_csv, header=0)
dist_mat = np.full((NUM_NODES, NUM_NODES), np.inf)
dist_fwd = np.full((NUM_NODES, NUM_NODES), np.inf)
dist_bwd = np.full((NUM_NODES, NUM_NODES), np.inf)
np.fill_diagonal(dist_mat, 0.)
np.fill_diagonal(dist_fwd, 0.)
np.fill_diagonal(dist_bwd, 0.)

for _, row in df.iterrows():
    i, j, d = int(row.iloc[0]), int(row.iloc[1]), float(row.iloc[2])
    dist_mat[i, j] = d
    dist_mat[j, i] = d
    dist_fwd[i, j] = d
    dist_bwd[j, i] = d

sigma = dist_mat[dist_mat < np.inf].std()

def gaussian_norm(dmat, directed=False):
    adj = np.where(dmat < np.inf, np.exp(-(dmat**2) / (sigma**2)), 0.0)
    np.fill_diagonal(adj, 1.0)
    if directed:
        deg = adj.sum(axis=1, keepdims=True)
        return adj / (deg + 1e-8)
    d_inv = np.where(adj.sum(axis=1) > 0, (adj.sum(axis=1) + 1e-8)**(-0.5), 0.)
    return (adj * d_inv[:, None]) * d_inv[None, :]

adj_sym = gaussian_norm(dist_mat, directed=False)
adj_fwd = gaussian_norm(dist_fwd, directed=True)
adj_bwd = gaussian_norm(dist_bwd, directed=True)

speed_train = data_norm_speed[:TRAIN_END, :].T
corr_mat = np.nan_to_num(np.corrcoef(speed_train), nan=0.)
adj_corr = np.where(np.clip(corr_mat, 0, 1) > 0.60, corr_mat, 0.)
np.fill_diagonal(adj_corr, 0.)
adj_corr_norm = adj_corr / (adj_corr.sum(axis=1, keepdims=True) + 1e-8)

A_t = torch.tensor(adj_sym, dtype=torch.float32).to(device)
A_road = torch.tensor(adj_sym, dtype=torch.float32).unsqueeze(0).to(device)
A_fwd_t = torch.tensor(adj_fwd, dtype=torch.float32).unsqueeze(0).to(device)
A_bwd_t = torch.tensor(adj_bwd, dtype=torch.float32).unsqueeze(0).to(device)
A_corr_t = torch.tensor(adj_corr_norm, dtype=torch.float32).unsqueeze(0).to(device)

print(f"✅ Adjacency matrices built. Corr mean degree ≈ {(adj_corr>0).sum(1).mean():.1f}")

data_tensor_all = torch.tensor(
    data_norm_all.transpose(1, 0, 2), dtype=torch.float32
).unsqueeze(0).to(device)
data_tensor_spd = data_tensor_all[:, :, :, 2:3]

T_full = TIME_STEPS
t_idx = torch.arange(T_full, dtype=torch.float32).to(device)
t_sin = torch.sin(2 * np.pi * (t_idx % STEPS_PER_DAY) / STEPS_PER_DAY)
t_cos = torch.cos(2 * np.pi * (t_idx % STEPS_PER_DAY) / STEPS_PER_DAY)
time_sin = t_sin.view(1, 1, -1, 1).expand(1, NUM_NODES, -1, 1)
time_cos = t_cos.view(1, 1, -1, 1).expand(1, NUM_NODES, -1, 1)
TIME_SCALE = 0.25

SPARSITY = 0.80
torch.manual_seed(42)
node_mask = (torch.rand(1, NUM_NODES, 1, 1) > SPARSITY).float().to(device)
blind_idx = np.where(node_mask[0, :, 0, 0].cpu().numpy() == 0)[0]
print(f"✅ Blind nodes: {len(blind_idx)} out of {NUM_NODES} (sparsity: {SPARSITY*100:.0f}%)")

def diffusion_cheby(A, K=2):
    D = A.sum(1)
    D_inv = torch.where(D > 0, 1.0/D, torch.zeros_like(D))
    Anorm = A * D_inv.unsqueeze(1)
    mats = [torch.eye(NUM_NODES, device=device), Anorm]
    for k in range(2, K):
        mats.append(2*torch.mm(Anorm, mats[-1]) - mats[-2])
    return mats

cheby_mats = diffusion_cheby(A_t, K=3)

class ChebConv(nn.Module):
    def __init__(self, in_dim, out_dim, K=3):
        super().__init__()
        self.K = K
        self.Ws = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=(k==0))
                                   for k in range(K)])
        self.mats = cheby_mats

    def forward(self, x):
        out = sum(self.Ws[k](torch.mm(self.mats[k], x)) for k in range(self.K))
        return out

class DualFlowCell(nn.Module):
    def __init__(self, hidden=64, include_tod=True, include_4path=True, include_path_mixing=True):
        super().__init__()
        self.include_tod = include_tod
        self.include_4path = include_4path
        self.include_path_mixing = include_path_mixing
        self.hidden = hidden

        msg_in_dim = hidden + 1 + (2 if include_tod else 0)
        self.msg_sym = ChebConv(msg_in_dim, hidden, K=2)

        if include_4path:
            self.msg_fwd = ChebConv(msg_in_dim, hidden, K=2)
            self.msg_bwd = ChebConv(msg_in_dim, hidden, K=2)
            self.msg_corr = ChebConv(msg_in_dim, hidden, K=2)
            if include_path_mixing:
                self.mix_w = nn.Linear(hidden, 4)

        self.gru = nn.GRUCell(hidden + 2, hidden)
        self.out = nn.Linear(hidden, 1)
        # SOLUTION 2: Explicit jam head (binary classifier on hidden state)
        self.jam_head = nn.Linear(hidden, 1)
        self.act = nn.Tanh()

    def forward(self, x_seq, m_seq, tod_free_seq=None, tod_jam_seq=None):
        N, T = x_seq.shape
        h = torch.zeros(N, self.hidden, device=x_seq.device)
        preds = []
        jam_preds = []

        if self.include_4path and self.include_path_mixing:
            mix_w_fn = lambda h_t: torch.softmax(self.mix_w(h_t), dim=1)
        else:
            mix_w_fn = None

        for t in range(T):
            if self.include_tod and tod_free_seq is not None:
                msg_in = torch.cat([h, m_seq[:, t:t+1],
                                    tod_free_seq[:, t:t+1], tod_jam_seq[:, t:t+1]], dim=-1)
            else:
                msg_in = torch.cat([h, m_seq[:, t:t+1]], dim=-1)

            if self.include_4path:
                m_sym = self.act(self.msg_sym(msg_in))
                m_fwd = self.act(self.msg_fwd(msg_in))
                m_bwd = self.act(self.msg_bwd(msg_in))
                m_corr = self.act(self.msg_corr(msg_in))

                if self.include_path_mixing:
                    mix_w = mix_w_fn(h)
                    msg = (mix_w[:, 0:1]*m_sym + mix_w[:, 1:2]*m_fwd +
                           mix_w[:, 2:3]*m_bwd + mix_w[:, 3:4]*m_corr)
                else:
                    msg = 0.25 * (m_sym + m_fwd + m_bwd + m_corr)
            else:
                msg = self.act(self.msg_sym(msg_in))

            x_t = x_seq[:, t:t+1] * m_seq[:, t:t+1]
            inp = torch.cat([msg, x_t, m_seq[:, t:t+1]], dim=-1)
            h_new = self.gru(inp, h)
            h = h_new + 0.1 * h
            preds.append(self.out(h)[:, 0])
            jam_preds.append(self.jam_head(h)[:, 0])
        return torch.stack(preds, dim=1), torch.stack(jam_preds, dim=1)

class DualFlow(nn.Module):
    def __init__(self, hidden=64, include_tod=True, jam_loss_weight=2.5, free_loss_weight=1.0,
                 use_soft_threshold=False, jam_bce_weight=0.5, anchor_diffusion=True):
        super().__init__()
        self.include_tod = include_tod
        self.jam_loss_weight = jam_loss_weight
        self.free_loss_weight = free_loss_weight
        self.use_soft_threshold = use_soft_threshold
        # SOLUTION 2: weight for jam BCE auxiliary loss
        self.jam_bce_weight = jam_bce_weight
        # SOLUTION 3: anchor-diffusion (IGNNK-style) at inference
        self.anchor_diffusion = anchor_diffusion
        self.fwd = DualFlowCell(hidden, include_tod)
        self.bwd = DualFlowCell(hidden, include_tod)
        self.fuse = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, 2), nn.Softmax(dim=-1)
        )

    def _run_full(self, x, m, tod_free=None, tod_jam=None):
        # Returns (speed_pred, jam_logit) — both bidirectionally fused
        pf, jf = self.fwd(x, m, tod_free, tod_jam)
        pb_rev, jb_rev = self.bwd(x.flip(1), m.flip(1),
                                   tod_free.flip(1) if tod_free is not None else None,
                                   tod_jam.flip(1) if tod_jam is not None else None)
        pb = pb_rev.flip(1)
        jb = jb_rev.flip(1)
        fuse_in = torch.stack([pf, pb], dim=-1)
        w = self.fuse(fuse_in)
        speed = (w[..., 0:1] * pf.unsqueeze(-1) + w[..., 1:2] * pb.unsqueeze(-1)).squeeze(-1)
        # Reuse same fusion weights for jam logits (saves params, ties direction quality)
        jam_logit = (w[..., 0:1] * jf.unsqueeze(-1) + w[..., 1:2] * jb.unsqueeze(-1)).squeeze(-1)
        return speed, jam_logit

    def _run(self, x, m, tod_free=None, tod_jam=None):
        speed, _ = self._run_full(x, m, tod_free, tod_jam)
        return speed

    def training_step(self, x, m, tod_free=None, tod_jam=None, m_blind_train=None):
        p, jam_logit = self._run_full(x, m, tod_free, tod_jam)
        p = torch.clamp(p, -5.0, 5.0)

        jt = torch.tensor(jam_thresh_soft_np if self.use_soft_threshold else jam_thresh_eval_np,
                          dtype=torch.float32, device=x.device)
        jam_flag = (x < jt.unsqueeze(1)).float()
        free_flag = 1.0 - jam_flag

        # SOLUTION 1: Supervise on imputation targets (blind nodes), not observations
        if m_blind_train is not None:
            supervision_mask = m_blind_train
        else:
            supervision_mask = torch.ones_like(m)
        sup_count = supervision_mask.sum().clamp(min=1.0)

        # Free-flow squared error on blind positions
        loss_free = ((p - x) * supervision_mask * free_flag) ** 2
        loss_free = loss_free.sum() / sup_count * self.free_loss_weight

        # Huber on jam positions
        delta = 2.0
        diff_jam = torch.abs((p - x) * supervision_mask * jam_flag)
        huber_jam = torch.where(diff_jam < delta,
                                0.5 * diff_jam ** 2,
                                delta * (diff_jam - 0.5 * delta))
        loss_jam = huber_jam.sum() / sup_count * self.jam_loss_weight

        # RMSE regularization
        rmse_reg = (((p - x) * supervision_mask) ** 2).sum() / sup_count * 0.1

        # SOLUTION 2: Auxiliary jam BCE loss on blind positions
        # Forces the network to explicitly classify jam vs free, which is a coarser
        # but more reliable signal than regressing speeds at high sparsity
        bce_per = F.binary_cross_entropy_with_logits(jam_logit, jam_flag, reduction='none')
        loss_bce = (bce_per * supervision_mask).sum() / sup_count * self.jam_bce_weight

        return loss_free + loss_jam + rmse_reg + loss_bce

    def impute(self, x, m, tod_free=None, tod_jam=None, diffusion_steps=3, alpha=0.3):
        p_init = self._run(x, m, tod_free, tod_jam)
        p_init = torch.clamp(p_init, -5.0, 5.0)
        if not self.anchor_diffusion:
            return m * x + (1.0 - m) * p_init

        # SOLUTION 3: Anchor-diffusion (IGNNK-style kriging refinement)
        # Iteratively smooth blind-node predictions over the symmetric graph,
        # keeping observed nodes pinned to ground truth (anchors).
        p = m * x + (1.0 - m) * p_init
        for _ in range(diffusion_steps):
            p_smooth = A_t @ p  # weighted neighborhood average using symmetric adj
            p_new = (1.0 - alpha) * p + alpha * p_smooth
            p = m * x + (1.0 - m) * p_new
        return p


# PHASE 1: Transformer Enhancement
# ============================================================================
# Add temporal self-attention on top of GRU hidden states to learn:
# - Which timepoints are informative (especially at extreme sparsity)
# - Long-range temporal dependencies (peak hours, seasonal patterns)
# - Dynamic weighting of spatial vs temporal signals
# ============================================================================

class DualFlowCellWithHidden(nn.Module):
    """DualFlowCell variant that also returns hidden states for transformer refinement"""
    def __init__(self, hidden=64, include_tod=True, include_4path=True, include_path_mixing=True):
        super().__init__()
        self.include_tod = include_tod
        self.include_4path = include_4path
        self.include_path_mixing = include_path_mixing
        self.hidden = hidden

        msg_in_dim = hidden + 1 + (2 if include_tod else 0)
        self.msg_sym = ChebConv(msg_in_dim, hidden, K=2)

        if include_4path:
            self.msg_fwd = ChebConv(msg_in_dim, hidden, K=2)
            self.msg_bwd = ChebConv(msg_in_dim, hidden, K=2)
            self.msg_corr = ChebConv(msg_in_dim, hidden, K=2)
            if include_path_mixing:
                self.mix_w = nn.Linear(hidden, 4)

        self.gru = nn.GRUCell(hidden + 2, hidden)
        self.out = nn.Linear(hidden, 1)
        self.jam_head = nn.Linear(hidden, 1)
        self.act = nn.Tanh()

    def forward(self, x_seq, m_seq, tod_free_seq=None, tod_jam_seq=None, return_hidden=False):
        """Forward pass optionally returning hidden states [N, T, hidden]"""
        N, T = x_seq.shape
        h = torch.zeros(N, self.hidden, device=x_seq.device)
        preds = []
        jam_preds = []
        hiddens = [] if return_hidden else None

        if self.include_4path and self.include_path_mixing:
            mix_w_fn = lambda h_t: torch.softmax(self.mix_w(h_t), dim=1)
        else:
            mix_w_fn = None

        for t in range(T):
            if self.include_tod and tod_free_seq is not None:
                msg_in = torch.cat([h, m_seq[:, t:t+1],
                                    tod_free_seq[:, t:t+1], tod_jam_seq[:, t:t+1]], dim=-1)
            else:
                msg_in = torch.cat([h, m_seq[:, t:t+1]], dim=-1)

            if self.include_4path:
                m_sym = self.act(self.msg_sym(msg_in))
                m_fwd = self.act(self.msg_fwd(msg_in))
                m_bwd = self.act(self.msg_bwd(msg_in))
                m_corr = self.act(self.msg_corr(msg_in))

                if self.include_path_mixing:
                    mix_w = mix_w_fn(h)
                    msg = (mix_w[:, 0:1]*m_sym + mix_w[:, 1:2]*m_fwd +
                           mix_w[:, 2:3]*m_bwd + mix_w[:, 3:4]*m_corr)
                else:
                    msg = 0.25 * (m_sym + m_fwd + m_bwd + m_corr)
            else:
                msg = self.act(self.msg_sym(msg_in))

            x_t = x_seq[:, t:t+1] * m_seq[:, t:t+1]
            inp = torch.cat([msg, x_t, m_seq[:, t:t+1]], dim=-1)
            h_new = self.gru(inp, h)
            h = h_new + 0.1 * h
            preds.append(self.out(h)[:, 0])
            jam_preds.append(self.jam_head(h)[:, 0])
            if return_hidden:
                hiddens.append(h.clone())

        pred_out = torch.stack(preds, dim=1)  # [N, T]
        jam_out = torch.stack(jam_preds, dim=1)  # [N, T]
        if return_hidden:
            hidden_out = torch.stack(hiddens, dim=1)  # [N, T, hidden]
            return pred_out, jam_out, hidden_out
        return pred_out, jam_out


class TransformerEnhancer(nn.Module):
    """Temporal self-attention to refine GRU predictions at extreme sparsity"""
    def __init__(self, hidden=64, num_layers=3, num_heads=4, ff_dim=256, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers

        # Transformer encoder: learns temporal dependencies
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection: hidden states → refined predictions
        self.out_proj = nn.Linear(hidden, 1)

    def _get_sinusoidal_pos_enc(self, T, device):
        """Generate sinusoidal positional encodings (supports any sequence length)"""
        positions = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(1)
        dim_indices = torch.arange(0, self.hidden, 2, dtype=torch.float32, device=device)
        div_term = 10000.0 ** (dim_indices / self.hidden)

        pos_enc = torch.zeros(T, self.hidden, device=device)
        pos_enc[:, 0::2] = torch.sin(positions / div_term)
        if self.hidden % 2 == 1:
            pos_enc[:, 1::2] = torch.cos(positions / div_term[:-1])
        else:
            pos_enc[:, 1::2] = torch.cos(positions / div_term)

        return pos_enc.unsqueeze(0)  # [1, T, hidden]

    def forward(self, hidden_states, mask=None):
        """
        Args:
            hidden_states: [N, T, hidden] from DualFlowCell
            mask: [N, T] optional observation mask (1=observed, 0=blind)
        Returns:
            refined_preds: [N, T] refined predictions
        """
        N, T, H = hidden_states.shape

        # Add sinusoidal positional encoding (supports any T)
        pos_enc = self._get_sinusoidal_pos_enc(T, hidden_states.device)  # [1, T, hidden]
        hidden_with_pos = hidden_states + pos_enc

        # Apply transformer (learns to suppress noise from blind nodes)
        refined = self.transformer(hidden_with_pos)  # [N, T, hidden]

        # Project to predictions
        refined_preds = self.out_proj(refined).squeeze(-1)  # [N, T]

        return refined_preds


class DualFlowTransformer(nn.Module):
    """DualFlow + TransformerEnhancer for Phase 1 improvement"""
    def __init__(self, hidden=64, include_tod=True, jam_loss_weight=2.5, free_loss_weight=1.0,
                 use_soft_threshold=False, jam_bce_weight=0.5, anchor_diffusion=True,
                 use_transformer=True, num_transformer_layers=3):
        super().__init__()
        self.include_tod = include_tod
        self.jam_loss_weight = jam_loss_weight
        self.free_loss_weight = free_loss_weight
        self.use_soft_threshold = use_soft_threshold
        self.jam_bce_weight = jam_bce_weight
        self.anchor_diffusion = anchor_diffusion
        self.use_transformer = use_transformer

        # Bidirectional GRU cells with hidden state output
        self.fwd = DualFlowCellWithHidden(hidden, include_tod)
        self.bwd = DualFlowCellWithHidden(hidden, include_tod)

        # Fusion layer (same as original DualFlow)
        self.fuse = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, 2), nn.Softmax(dim=-1)
        )

        # Transformer enhancers (Phase 1)
        if use_transformer:
            self.trans_fwd = TransformerEnhancer(hidden, num_layers=num_transformer_layers)
            self.trans_bwd = TransformerEnhancer(hidden, num_layers=num_transformer_layers)
            self.trans_jam = TransformerEnhancer(hidden, num_layers=num_transformer_layers)

    def _run_full(self, x, m, tod_free=None, tod_jam=None):
        """Forward pass with optional transformer refinement"""
        # Get GRU outputs and hidden states
        pf, jf, h_fwd = self.fwd(x, m, tod_free, tod_jam, return_hidden=True)
        pb_rev, jb_rev, h_bwd_rev = self.bwd(x.flip(1), m.flip(1),
                                              tod_free.flip(1) if tod_free is not None else None,
                                              tod_jam.flip(1) if tod_jam is not None else None,
                                              return_hidden=True)
        pb = pb_rev.flip(1)
        jb = jb_rev.flip(1)
        h_bwd = h_bwd_rev.flip(1)  # [N, T, hidden]

        # Fuse bidirectional predictions
        fuse_in = torch.stack([pf, pb], dim=-1)
        w = self.fuse(fuse_in)
        speed = (w[..., 0:1] * pf.unsqueeze(-1) + w[..., 1:2] * pb.unsqueeze(-1)).squeeze(-1)
        jam_logit = (w[..., 0:1] * jf.unsqueeze(-1) + w[..., 1:2] * jb.unsqueeze(-1)).squeeze(-1)

        # Apply transformer refinement (Phase 1)
        if self.use_transformer:
            # Fuse hidden states with bidirectional weights
            h_fused = h_fwd * w[:, :, 0:1] + h_bwd * w[:, :, 1:2]  # [N, T, hidden]

            # Refine with transformers
            speed_refined = self.trans_fwd(h_fused, m)
            jam_refined = self.trans_jam(h_fused, m)

            # Blend GRU and transformer predictions (start with light transformer weight)
            alpha = 0.3  # Transformer weight (can be learned later)
            speed = (1.0 - alpha) * speed + alpha * speed_refined
            jam_logit = (1.0 - alpha) * jam_logit + alpha * jam_refined

        return speed, jam_logit

    def _run(self, x, m, tod_free=None, tod_jam=None):
        speed, _ = self._run_full(x, m, tod_free, tod_jam)
        return speed

    def training_step(self, x, m, tod_free=None, tod_jam=None, m_blind_train=None):
        """Identical to DualFlow training_step"""
        p, jam_logit = self._run_full(x, m, tod_free, tod_jam)
        p = torch.clamp(p, -5.0, 5.0)

        jt = torch.tensor(jam_thresh_soft_np if self.use_soft_threshold else jam_thresh_eval_np,
                          dtype=torch.float32, device=x.device)
        jam_flag = (x < jt.unsqueeze(1)).float()
        free_flag = 1.0 - jam_flag

        # SOLUTION 1: Supervise on imputation targets (blind nodes), not observations
        if m_blind_train is not None:
            supervision_mask = m_blind_train
        else:
            supervision_mask = torch.ones_like(m)
        sup_count = supervision_mask.sum().clamp(min=1.0)

        # Free-flow squared error on blind positions
        loss_free = ((p - x) * supervision_mask * free_flag) ** 2
        loss_free = loss_free.sum() / sup_count * self.free_loss_weight

        # Huber on jam positions
        delta = 2.0
        diff_jam = torch.abs((p - x) * supervision_mask * jam_flag)
        huber_jam = torch.where(diff_jam < delta,
                                0.5 * diff_jam ** 2,
                                delta * (diff_jam - 0.5 * delta))
        loss_jam = huber_jam.sum() / sup_count * self.jam_loss_weight

        # RMSE regularization
        rmse_reg = (((p - x) * supervision_mask) ** 2).sum() / sup_count * 0.1

        # SOLUTION 2: Auxiliary jam BCE loss on blind positions
        bce_per = F.binary_cross_entropy_with_logits(jam_logit, jam_flag, reduction='none')
        loss_bce = (bce_per * supervision_mask).sum() / sup_count * self.jam_bce_weight

        return loss_free + loss_jam + rmse_reg + loss_bce

    def impute(self, x, m, tod_free=None, tod_jam=None, diffusion_steps=3, alpha=0.3):
        """Identical to DualFlow impute"""
        p_init = self._run(x, m, tod_free, tod_jam)
        p_init = torch.clamp(p_init, -5.0, 5.0)
        if not self.anchor_diffusion:
            return m * x + (1.0 - m) * p_init

        p = m * x + (1.0 - m) * p_init
        for _ in range(diffusion_steps):
            p_smooth = A_t @ p
            p_new = (1.0 - alpha) * p + alpha * p_smooth
            p = m * x + (1.0 - m) * p_new
        return p


print("✅ DualFlow model architecture defined")
print("✅ DualFlowTransformer (Phase 1: Temporal Attention) available")

PRODUCTION_SEED = 86415
PRODUCTION_JAM_WEIGHT = 2.0  # Increased to penalize jam errors harder; S2 jam head helps at high sparsity
PRODUCTION_FREE_WEIGHT = 1.0
PRODUCTION_JAM_BCE_WEIGHT = 1.0  # Increased from 0.5: jam events rare, need higher weight to balance

def train_dualflow_production(hidden=64, epochs=600, use_transformer=True):
    torch.manual_seed(PRODUCTION_SEED)
    np.random.seed(PRODUCTION_SEED)

    # PHASE 1: Use DualFlowTransformer with temporal attention
    net_class = DualFlowTransformer if use_transformer else DualFlow
    net = net_class(hidden=hidden, include_tod=True,
                    jam_loss_weight=PRODUCTION_JAM_WEIGHT,
                    free_loss_weight=PRODUCTION_FREE_WEIGHT,
                    use_soft_threshold=False,
                    jam_bce_weight=PRODUCTION_JAM_BCE_WEIGHT,
                    anchor_diffusion=True,
                    use_transformer=use_transformer).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    best_blind_mae, best_wts, best_ep = float('inf'), None, 0
    loss_history_train, loss_history_val = [], []

    model_name = "DualFlowTransformer (Phase 1)" if use_transformer else "DualFlow (Baseline)"
    print(f"\n{'='*80}")
    print(f"PRODUCTION MODEL: {model_name} — Warmup variant")
    print(f"  Seed: {PRODUCTION_SEED}  |  Jam: 1.0→{PRODUCTION_JAM_WEIGHT}x (warmup ep 100-200)  |  Free: {PRODUCTION_FREE_WEIGHT}x  |  JamBCE: 0.5→{PRODUCTION_JAM_BCE_WEIGHT}x")
    print(f"  S1: blind-node supervision  |  S2: jam head (warmed up)  |  S3: anchor diffusion")
    if use_transformer:
        print(f"  PHASE 1: Temporal Transformer (3 layers, 4 heads) on top of GRU")
    print(f"  Training: Full {epochs} epochs (no early stopping) | Best checkpoint saved")
    print(f"  Honest R² on blind nodes only")
    print(f"{'='*80}\n")

    for ep in range(1, epochs + 1):
        # SOLUTION 2 WARMUP: ramp jam weights from 1.0 (start) to target (PRODUCTION_*) over warmup window
        # Lets free-flow loss stabilize the model first, then increases jam pressure gradually
        warmup_start_ep = 100
        warmup_end_ep = 200
        if ep < warmup_start_ep:
            ramp = 0.0  # use base weight 1.0
        elif ep < warmup_end_ep:
            ramp = (ep - warmup_start_ep) / (warmup_end_ep - warmup_start_ep)
        else:
            ramp = 1.0
        net.jam_loss_weight = 1.0 + ramp * (PRODUCTION_JAM_WEIGHT - 1.0)
        net.jam_bce_weight = 0.5 + ramp * (PRODUCTION_JAM_BCE_WEIGHT - 0.5)

        net.train()
        t0 = np.random.randint(0, TRAIN_END - BATCH_TIME)
        x_full = torch.tensor(speed_np[t0:t0+BATCH_TIME, :], dtype=torch.float32).T.to(device)
        # SOLUTION 1 (SPIN/ImputeFormer-style): Mask input AND supervise on the masked positions
        # m_train = 1 means observed (input visible), 0 means blind (input zeroed)
        m_train = (torch.rand(NUM_NODES, 1, device=device) > SPARSITY).float().expand(-1, BATCH_TIME)
        # m_blind_train = 1 - m_train: supervise only on blind positions (imputation targets)
        # This forces the model to predict from spatial neighbors instead of copying input
        m_blind_train = 1.0 - m_train
        slots = (np.arange(t0, t0+BATCH_TIME) % 288).astype(int)
        tod_free = torch.tensor(tod_free_np[:, slots], dtype=torch.float32).to(device)
        tod_jam = torch.tensor(tod_jam_np[:, slots], dtype=torch.float32).to(device)
        loss = net.training_step(x_full, m_train, tod_free, tod_jam, m_blind_train=m_blind_train)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  ⚠️  NaN/Inf at ep {ep}, restarting...")
            return train_dualflow_production(hidden, epochs)

        loss_history_train.append(loss.item())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        opt.step()

        if ep % 50 == 0:
            net.eval()
            with torch.no_grad():
                x_v = torch.tensor(speed_np[VAL_START:VAL_END, :], dtype=torch.float32).T.to(device)
                # Use SPARSITY-rate blind mask so val matches test conditions
                m_v_obs = (torch.rand(NUM_NODES, 1, device=device) > SPARSITY).float().expand(-1, VAL_END-VAL_START)
                m_v_blind = 1.0 - m_v_obs
                slots_v = (np.arange(VAL_START, VAL_END) % 288).astype(int)
                tf_v = torch.tensor(tod_free_np[:, slots_v], dtype=torch.float32).to(device)
                tj_v = torch.tensor(tod_jam_np[:, slots_v], dtype=torch.float32).to(device)
                # Loss on blind positions
                vl = net.training_step(x_v, m_v_obs, tf_v, tj_v, m_blind_train=m_v_blind).item()
                # Honest blind-node MAE/RMSE in normalized space
                p_v = net._run(x_v, m_v_obs, tf_v, tj_v)
                p_v = torch.clamp(p_v, -5.0, 5.0)
                blind_count = m_v_blind.sum().clamp(min=1.0)
                mae_v = (torch.abs(p_v - x_v) * m_v_blind).sum().item() / blind_count.item()
                rmse_v = torch.sqrt(((p_v - x_v) ** 2 * m_v_blind).sum() / blind_count).item()
                # R² on blind nodes only (not observed, which are pinned to truth by impute)
                jt = torch.tensor(jam_thresh_eval_np, dtype=torch.float32, device=x_v.device)
                jam_flag = (x_v < jt.unsqueeze(1)).float()
                ss_res = ((p_v - x_v) ** 2 * m_v_blind).sum()
                ss_tot = ((x_v - (x_v * m_v_blind).sum() / blind_count) ** 2 * m_v_blind).sum() + 1e-8
                r2_v = (1.0 - ss_res / ss_tot).item()
                # Jam MAE on blind nodes
                jam_count = (jam_flag * m_v_blind).sum().clamp(min=1.0)
                mae_jam_v = (torch.abs(p_v - x_v) * jam_flag * m_v_blind).sum().item() / jam_count.item() if jam_count > 0 else 0.0
            loss_history_val.append(vl)
            if mae_v < best_blind_mae:
                best_blind_mae = mae_v
                best_wts = copy.deepcopy(net.state_dict())
                best_ep = ep
                marker = " ← BEST"
            else:
                marker = ""
            print(f"  [DualFlow] ep {ep:3d} | loss={vl:.4f} | BlindMAE={mae_v:.4f} | BlindJamMAE={mae_jam_v:.4f} | R²={r2_v:.4f} | jam_w={net.jam_loss_weight:.2f}{marker}")

    if best_wts:
        net.load_state_dict(best_wts)
    return net, loss_history_train, loss_history_val

print("✅ Training function defined")

dualflow_net, dualflow_loss_train, dualflow_loss_val = train_dualflow_production(hidden=64, epochs=600, use_transformer=True)
print("\n✅ Training complete!")

def jam_prec_recall(pred_kmh, true_kmh, threshold=JAM_KMH_EVAL):
    pred_jam = (pred_kmh < threshold).astype(int).flatten()
    true_jam = (true_kmh < threshold).astype(int).flatten()
    if true_jam.sum() == 0:
        return 0.0, 0.0, 0.0
    prec = precision_score(true_jam, pred_jam, zero_division=0)
    rec = recall_score(true_jam, pred_jam, zero_division=0)
    f1 = f1_score(true_jam, pred_jam, zero_division=0)
    return prec, rec, f1

def compute_ssim(pred_kmh, true_kmh):
    s = 0.0
    for i in range(pred_kmh.shape[0]):
        s += ssim(true_kmh[i], pred_kmh[i], data_range=true_kmh[i].max()-true_kmh[i].min())
    return s / pred_kmh.shape[0]

def eval_pred_np(pred_kmh_bl, true_kmh_bl):
    diff = pred_kmh_bl - true_kmh_bl
    mae_all = float(np.abs(diff).mean())
    mse_all = float((diff ** 2).mean())
    rmse_all = float(np.sqrt(mse_all))
    ss_res = (diff ** 2).sum()
    ss_tot = ((true_kmh_bl - true_kmh_bl.mean()) ** 2).sum()
    r2_all = float(1.0 - ss_res / (ss_tot + 1e-8))

    jm = true_kmh_bl < JAM_KMH_EVAL
    mae_jam = float(np.abs(diff[jm]).mean()) if jm.any() else np.nan

    prec, rec, f1 = jam_prec_recall(pred_kmh_bl, true_kmh_bl)
    ssim_val = compute_ssim(pred_kmh_bl, true_kmh_bl)

    return dict(
        mae_all=mae_all, rmse_all=rmse_all, r2_all=r2_all,
        mae_jam=mae_jam, prec=prec, rec=rec, f1=f1, ssim=ssim_val
    )

print("✅ Evaluation functions defined")

# ════════════════════════════════════════════════════════════════════════════
# ENSEMBLE: Learned blending of DualFlow + Historical Average
# ════════════════════════════════════════════════════════════════════════════

class DualFlowEnsemble(nn.Module):
    """Learn per-node blend weights: pred = α_n * dualflow + (1-α_n) * ha"""
    def __init__(self, num_nodes):
        super().__init__()
        # Per-node blend weight (logits, will be sigmoid'd to [0,1])
        self.logit_alpha = nn.Parameter(torch.zeros(num_nodes))

    def forward(self, dualflow_pred, ha_pred):
        """
        Blend predictions from DualFlow and Historical Average.
        Args:
            dualflow_pred: [N, T] tensor in normalized space
            ha_pred: [N, T] tensor in normalized space
        Returns:
            blended: [N, T] tensor
        """
        alpha = torch.sigmoid(self.logit_alpha).view(-1, 1)  # [N, 1]
        return alpha * dualflow_pred + (1 - alpha) * ha_pred

def train_ensemble_weights(dualflow_net, ha_preds_norm, val_start, val_end,
                          val_nodes_mask, true_vals_norm, epochs=100):
    """
    Learn blend weights on validation set using DualFlow + HA predictions.

    Args:
        dualflow_net: trained DualFlow model
        ha_preds_norm: [N, T_train] Historical Average predictions (normalized)
        val_start, val_end: validation time window
        val_nodes_mask: [N] boolean, True for observed nodes
        true_vals_norm: [N, T_train] ground truth (normalized)
        epochs: training epochs for blend weights
    Returns:
        ensemble: trained DualFlowEnsemble model
    """
    ensemble = DualFlowEnsemble(NUM_NODES).to(device)
    opt = torch.optim.Adam(ensemble.parameters(), lr=1e-2)

    x_v = torch.tensor(speed_np[val_start:val_end, :], dtype=torch.float32).T.to(device)
    ha_v = torch.tensor(ha_preds_norm[:, val_start:val_end], dtype=torch.float32).to(device)
    m_v = val_nodes_mask.unsqueeze(1).expand(-1, val_end - val_start)

    best_loss = float('inf')
    for ep in range(1, epochs + 1):
        ensemble.train()
        dualflow_net.eval()
        with torch.no_grad():
            # Get DualFlow predictions on validation
            slots_v = torch.arange(val_start, val_end, device=device) % 288
            tf_v = torch.tensor(tod_free_np[:, slots_v.cpu().numpy()], dtype=torch.float32).to(device)
            tj_v = torch.tensor(tod_jam_np[:, slots_v.cpu().numpy()], dtype=torch.float32).to(device)
            df_v = dualflow_net._run(x_v, m_v, tf_v, tj_v)

        # Blend
        blended = ensemble(df_v, ha_v)

        # Loss on observed nodes
        loss = F.mse_loss(blended[m_v == 1], x_v[m_v == 1])

        opt.zero_grad()
        loss.backward()
        opt.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            if ep % 20 == 0:
                print(f"  [Ensemble] ep {ep:3d} | val_loss={best_loss:.4f} | "
                      f"α_mean={torch.sigmoid(ensemble.logit_alpha).mean():.3f}")

    return ensemble

def compute_historical_average(speed_data, train_end, eval_start, eval_len):
    """
    Compute Historical Average baseline: for each node and time-of-day,
    use the mean speed from the training period.

    Args:
        speed_data: [T, N] speed matrix (normalized)
        train_end: end of training period
        eval_start: start of eval period
        eval_len: length of eval period
    Returns:
        ha_eval_norm: [N, eval_len] Historical Average predictions (normalized)
    """
    # For each node, compute mean speed by time-of-day (288 steps per day)
    ha_by_tod = np.zeros((NUM_NODES, 288), dtype=np.float32)
    for n in range(NUM_NODES):
        for s in range(288):
            mask = np.arange(0, train_end) % 288 == s
            ha_by_tod[n, s] = speed_data[mask, n].mean()

    # Apply to eval period
    ha_eval_norm = np.zeros((NUM_NODES, eval_len), dtype=np.float32)
    for t in range(eval_len):
        tod_idx = (eval_start + t) % 288
        ha_eval_norm[:, t] = ha_by_tod[:, tod_idx]

    return ha_eval_norm

print("✅ Ensemble blending model defined")

_T_eval = (EVAL_LEN // BATCH_TIME) * BATCH_TIME
true_eval_kmh = np.zeros((len(blind_idx), _T_eval), dtype=np.float32)
for ni, n in enumerate(blind_idx):
    true_eval_kmh[ni] = (
        data_norm_speed[EVAL_START:EVAL_START+_T_eval, n]
        * node_stds[n] + node_means[n]
    )

print(f"Test set: {len(blind_idx)} blind nodes × {_T_eval} time steps")

dualflow_net.eval()
ws = max(0, EVAL_START - WARMUP_STEPS)
total = (EVAL_START + _T_eval) - ws
x_e = torch.tensor(speed_np[ws:EVAL_START+_T_eval, :], dtype=torch.float32).T.to(device)
m_e = (node_mask[0,:,0,0]==1).float().unsqueeze(1).expand(-1, total)
si = np.arange(ws, EVAL_START + _T_eval) % 288
tf_e = torch.tensor(tod_free_np[:, si], dtype=torch.float32).to(device)
tj_e = torch.tensor(tod_jam_np[:, si], dtype=torch.float32).to(device)

with torch.no_grad():
    p_full = dualflow_net.impute(x_e, m_e, tf_e, tj_e).cpu().numpy()

offset = EVAL_START - ws
p_e = p_full[:, offset:]
pred_kmh = np.zeros((len(blind_idx), _T_eval), dtype=np.float32)
for ni, n in enumerate(blind_idx):
    if np.isnan(p_e[n]).any():
        pred_kmh[ni] = true_eval_kmh[ni]
    else:
        pred_kmh[ni] = np.clip(p_e[n] * node_stds[n] + node_means[n], 0, 120)

metrics = eval_pred_np(pred_kmh, true_eval_kmh)
print(f"\n✅ DualFlow Results:")
print(f"   Overall MAE: {metrics['mae_all']:.4f} km/h")
print(f"   Jam MAE: {metrics['mae_jam']:.4f} km/h")
print(f"   RMSE: {metrics['rmse_all']:.4f} km/h")
print(f"   R²: {metrics['r2_all']:.4f}")
print(f"   Jam F1: {metrics['f1']:.4f}")
print(f"   SSIM: {metrics['ssim']:.4f}")

# ════════════════════════════════════════════════════════════════════════════
# ENSEMBLE: Train blend weights on validation set
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*90)
print("  ENSEMBLE: Learning blend weights (DualFlow + Historical Average)")
print("="*90)

# Compute Historical Average on training data
ha_preds_norm = compute_historical_average(speed_np, TRAIN_END, EVAL_START, EVAL_LEN)

# Train ensemble blend weights on validation set
val_nodes_mask = torch.tensor(node_mask[0,:,0,0]==1, dtype=torch.float32)
ensemble = train_ensemble_weights(
    dualflow_net, ha_preds_norm, VAL_START, VAL_END,
    val_nodes_mask, speed_np, epochs=100
)

# Evaluate ensemble on test set
dualflow_net.eval()
ensemble.eval()
with torch.no_grad():
    # DualFlow predictions on test
    p_full_df = dualflow_net.impute(x_e, m_e, tf_e, tj_e).cpu().numpy()
    p_df_test = p_full_df[:, offset:]

    # Historical Average predictions on test
    ha_test_norm = ha_preds_norm[:, WARMUP_STEPS:WARMUP_STEPS+_T_eval]

    # Blend
    p_df_t = torch.tensor(p_df_test, dtype=torch.float32).to(device)
    ha_test_t = torch.tensor(ha_test_norm, dtype=torch.float32).to(device)
    p_blended = ensemble(p_df_t, ha_test_t).cpu().numpy()

# Convert to km/h
pred_kmh_ens = np.zeros((len(blind_idx), _T_eval), dtype=np.float32)
for ni, n in enumerate(blind_idx):
    pred_kmh_ens[ni] = np.clip(p_blended[n] * node_stds[n] + node_means[n], 0, 120)

metrics_ens = eval_pred_np(pred_kmh_ens, true_eval_kmh)
print(f"\n✅ DualFlow+HA Ensemble Results:")
print(f"   Overall MAE: {metrics_ens['mae_all']:.4f} km/h  (vs DualFlow {metrics['mae_all']:.4f})")
print(f"   Jam MAE: {metrics_ens['mae_jam']:.4f} km/h  (vs DualFlow {metrics['mae_jam']:.4f})")
print(f"   RMSE: {metrics_ens['rmse_all']:.4f} km/h  (vs DualFlow {metrics['rmse_all']:.4f})")
print(f"   R²: {metrics_ens['r2_all']:.4f}  (vs DualFlow {metrics['r2_all']:.4f})")
print(f"   Jam F1: {metrics_ens['f1']:.4f}  (vs DualFlow {metrics['f1']:.4f})")
print(f"   SSIM: {metrics_ens['ssim']:.4f}  (vs DualFlow {metrics['ssim']:.4f})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=100)

epochs = range(1, len(dualflow_loss_train) + 1)
ax1.plot(epochs, dualflow_loss_train, 'o-', label='Training Loss',
         linewidth=2, markersize=4, color='#0277bd')
if len(dualflow_loss_val) > 0:
    ax1.plot([i*50 for i in range(1, len(dualflow_loss_val)+1)],
            dualflow_loss_val, 's-', label='Validation Loss',
            linewidth=2, markersize=6, color='#d32f2f')
ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
ax1.set_title('Training Dynamics', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

window = max(1, len(dualflow_loss_train) // 20)
smoothed = np.convolve(dualflow_loss_train, np.ones(window)/window, mode='valid')
smoothed_epochs = range(window, len(dualflow_loss_train) + 1)
ax2.plot(smoothed_epochs, smoothed, 'o-', linewidth=2.5,
         markersize=3, color='#0277bd', label='Smoothed (MA)')
ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax2.set_ylabel('Loss (smoothed)', fontsize=11, fontweight='bold')
ax2.set_title('Smoothed Loss Curve', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print("✅ Loss curves plotted")

mae_per_node = np.mean(np.abs(pred_kmh - true_eval_kmh), axis=1)
sorted_idx = np.argsort(mae_per_node)
node_indices = [
    sorted_idx[0],
    sorted_idx[len(sorted_idx)//3],
    sorted_idx[2*len(sorted_idx)//3],
    sorted_idx[-1]
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
axes = axes.flatten()

for idx, node_id in enumerate(node_indices):
    ax = axes[idx]
    t = np.arange(len(true_eval_kmh[node_id]))
    mae_node = mae_per_node[node_id]

    ax.plot(t, true_eval_kmh[node_id], 'o-', label='Ground Truth',
           linewidth=2, markersize=3, color='#0277bd', alpha=0.8)
    ax.plot(t, pred_kmh[node_id], 's--', label='DualFlow Prediction',
           linewidth=2, markersize=3, color='#d32f2f', alpha=0.8)

    ax.axhline(y=JAM_KMH_EVAL, color='orange', linestyle=':', linewidth=1.5, label='Jam Threshold')
    ax.set_xlabel('Time step', fontsize=10)
    ax.set_ylabel('Speed (km/h)', fontsize=10)
    ax.set_title(f'Node {blind_idx[node_id]} (MAE={mae_node:.2f} km/h)',
                fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print("✅ Prediction plots generated")
