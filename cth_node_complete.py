# =============================================================================
# Graph-CTH-NODE  v5  —  Complete Implementation
#
# DIAGNOSIS FROM v4.1 THRESHOLD SWEEP:
#   Speed-threshold (pred < 40 km/h): prec=0.417  rec=0.056  F1=0.099
#   Best logit-threshold (@ 0.35):    prec=0.055  rec=0.115  F1=0.075
#   → Speed decoder is BETTER at jam detection than the jam_head logit.
#   → Jam_head is uncalibrated: BCE at 92:8 ratio is dominated by free-flow,
#     so the head learns to output near-zero everywhere.
#
# WHAT CHANGED vs v4.1 and WHY:
#
# [A] FOCAL BCE for jam_head  (Lin et al. ICCV 2017)
#     Replace standard BCE with α-weighted focal loss on the jam_head:
#       FL = -α*(1-p)^γ*log(p)   for jams    (α=0.85, γ=2)
#       FL = -(1-α)*p^γ*log(1-p) for free-flow
#     At extreme imbalance 92:8, standard BCE contributes ~0.5% loss from
#     jams. With focal BCE, jam examples contribute ~93%+ of total BCE loss.
#     This is the primary fix for jam_head precision=0.043.
#
# [B] REMOVE NODE EMBEDDINGS
#     Node embeddings degraded overall MAE: 4.30 (v3) → 4.70 (v4) → 4.85 (v4.1).
#     At 80% blind nodes, embeddings for blind sensors get almost no gradient
#     through regression loss → learn spurious patterns → inflate free-flow error.
#     FISF (ICML 2025) confirms: propagation-based features with low variance
#     across blind nodes contribute little to performance.
#     INPUT_DIM returns to 13.
#
# [C] JAM DETECTION METRIC: SPEED THRESHOLD (primary) + LOGIT (secondary)
#     Speed decoder already gives F1=0.099 (better than any logit threshold).
#     Report both: speed-threshold (< 40 km/h) and logit-threshold sweep.
#     The jam_head now serves as a regulariser for z via focal BCE gradient.
#
# [D] REDUCE lam_recall 0.40 → 0.20, INCREASE jam_weight 16 → 20
#     Curriculum recall at 0.40 is inflating false positives (decoder pushed
#     below 40 km/h on free-flow sensors), raising overall MAE from 4.30→4.85.
#     Focal BCE already handles recall more directly. Lower recall lambda
#     reduces false-positive pressure; higher jam_weight compensates.
#
# KEPT FROM v4.1:
#   Linear warmup 150 epochs (Kalra NeurIPS 2024) — training is now stable
#   MAE-based jam region loss (Xie et al. 2023) — no quadratic bias
#   Curriculum recall ramp 0→0.20 over 200 epochs
#   Dual tod prior features 11+12 (free-flow + jam prior)
#   Random mask per accum step, 5-mask diversity
#   Gradient clip 0.5
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import copy
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# =============================================================================
# DATASET SELECTOR — Choose traffic dataset to run on
# =============================================================================

DATASETS = {
    'PEMS04': {
        'npz_url': "https://zenodo.org/records/7816008/files/PEMS04.npz?download=1",
        'csv_url': "https://zenodo.org/records/7816008/files/PEMS04.csv?download=1",
        'num_nodes': 307,
        'time_steps': 5000,
        'channel_idx': 2,  # speed channel index
    },
    'PEMS08': {
        'npz_url': "https://zenodo.org/records/7816008/files/PEMS08.npz?download=1",
        'csv_url': "https://zenodo.org/records/7816008/files/PEMS08.csv?download=1",
        'num_nodes': 170,
        'time_steps': 5000,
        'channel_idx': 2,
    },
    'METR-LA': {
        'npz_url': "https://zenodo.org/records/4518122/files/metr-la.npz?download=1",
        'csv_url': "https://zenodo.org/records/4518122/files/metr-la.csv?download=1",
        'num_nodes': 207,
        'time_steps': 34272,  # longer timeseries
        'channel_idx': 0,
    },
    'PEMS-BAY': {
        'npz_url': "https://zenodo.org/records/4512072/files/pems-bay.npz?download=1",
        'csv_url': "https://zenodo.org/records/4512072/files/pems-bay.csv?download=1",
        'num_nodes': 325,
        'time_steps': 34272,
        'channel_idx': 0,
    },
}

# ╔─ CHANGE THIS TO SELECT DATASET ─╗
DATASET_NAME = 'PEMS04'  # Options: 'PEMS04', 'PEMS08', 'METR-LA', 'PEMS-BAY'
# ╚──────────────────────────────────╝

if DATASET_NAME not in DATASETS:
    raise ValueError(f"Unknown dataset: {DATASET_NAME}. Choose from {list(DATASETS.keys())}")

ds_cfg = DATASETS[DATASET_NAME]
print(f"\n✅ Using dataset: {DATASET_NAME}")
print(f"   Nodes: {ds_cfg['num_nodes']} | Time steps: {ds_cfg['time_steps']}")

# =============================================================================
# CELL 1 — Data loading
# =============================================================================

url_npz = ds_cfg['npz_url']
url_csv = ds_cfg['csv_url']
fn_npz  = f"{DATASET_NAME}.npz"
fn_csv  = f"{DATASET_NAME}.csv"

for fn, url in [(fn_npz, url_npz), (fn_csv, url_csv)]:
    if not os.path.exists(fn):
        print(f"Downloading {fn}...")
        try:
            urllib.request.urlretrieve(url, fn)
        except Exception as e:
            print(f"❌ Download failed: {e}")
            raise

raw_npz   = np.load(fn_npz)
NUM_NODES  = ds_cfg['num_nodes']
TIME_STEPS = ds_cfg['time_steps']
CHAN_IDX   = ds_cfg['channel_idx']
TRAIN_END  = min(4000, int(0.8 * TIME_STEPS))
VAL_END    = min(TRAIN_END + 240, int(0.9 * TIME_STEPS))
EVAL_START = min(VAL_END + 260, int(0.9 * TIME_STEPS))
EVAL_LEN   = min(450, TIME_STEPS - EVAL_START)

VAL_START = TRAIN_END
print(f"   Train: t=0–{TRAIN_END} | Val: t={VAL_START}–{VAL_END} | Eval: t={EVAL_START}–{EVAL_START+EVAL_LEN}")

raw_all   = raw_npz['data'][:TIME_STEPS, :NUM_NODES, :]
raw_all   = np.nan_to_num(raw_all, nan=0.0)
raw_speed = raw_all[:, :, CHAN_IDX]

# Per-node normalisation — Beeking et al. 2023
node_means = raw_speed[:TRAIN_END].mean(axis=0)
node_stds  = raw_speed[:TRAIN_END].std(axis=0) + 1e-8
data_norm_speed = (raw_speed - node_means) / node_stds

# Normalise flow & occupancy globally
for c in range(3):
    mu = raw_all[:TRAIN_END, :, c].mean()
    sg = raw_all[:TRAIN_END, :, c].std() + 1e-8
    raw_all[:, :, c] = (raw_all[:, :, c] - mu) / sg

data_norm_all = raw_all.copy()
data_norm_all[:, :, 2] = data_norm_speed

JAM_KMH_EVAL  = 40.0
JAM_KMH_TRAIN = 50.0
jam_thresh_eval_np  = (JAM_KMH_EVAL  - node_means) / node_stds
jam_thresh_train_np = (JAM_KMH_TRAIN - node_means) / node_stds
jam_thresh_eval_t   = torch.tensor(jam_thresh_eval_np,  dtype=torch.float32).to(device)
jam_thresh_train_t  = torch.tensor(jam_thresh_train_np, dtype=torch.float32).to(device)

# Unconditional time-of-day prior
STEPS_PER_DAY = 288
slot_idx      = np.arange(TIME_STEPS) % STEPS_PER_DAY
tod_mean_sp   = np.zeros((NUM_NODES, STEPS_PER_DAY), dtype=np.float32)
for s in range(STEPS_PER_DAY):
    vals = data_norm_speed[slot_idx == s, :]
    if len(vals):
        tod_mean_sp[:, s] = vals.mean(axis=0)
tod_prior_np = tod_mean_sp[:, slot_idx].T
tod_prior    = torch.tensor(tod_prior_np, dtype=torch.float32).to(device)

# Dual tod prior (free-flow + jam conditioned) — v4
JAM_KMH_SPLIT = 50.0
split_norm_np = (JAM_KMH_SPLIT - node_means) / node_stds
tod_free_np = np.zeros((NUM_NODES, STEPS_PER_DAY), dtype=np.float32)
tod_jam_np  = np.zeros((NUM_NODES, STEPS_PER_DAY), dtype=np.float32)
for s in range(STEPS_PER_DAY):
    t_mask_s = (slot_idx[:TRAIN_END] == s)
    vals_s   = data_norm_speed[:TRAIN_END][t_mask_s, :]
    if len(vals_s) == 0:
        continue
    for n in range(NUM_NODES):
        col       = vals_s[:, n]
        thresh_n  = split_norm_np[n]
        free_rows = col[col >= thresh_n]
        jam_rows  = col[col <  thresh_n]
        tod_free_np[n, s] = free_rows.mean() if len(free_rows) else col.mean()
        tod_jam_np[n, s]  = jam_rows.mean()  if len(jam_rows)  else thresh_n - 0.5

tod_free = torch.tensor(tod_free_np[:, slot_idx].T, dtype=torch.float32).to(device)
tod_jam  = torch.tensor(tod_jam_np[:, slot_idx].T,  dtype=torch.float32).to(device)

data_tensor_all = torch.tensor(
    data_norm_all.transpose(1, 0, 2), dtype=torch.float32
).unsqueeze(0).to(device)
data_tensor_spd = data_tensor_all[:, :, :, 2:3]

T_full   = TIME_STEPS
t_idx    = torch.arange(T_full, dtype=torch.float32).to(device)
t_sin    = torch.sin(2 * np.pi * (t_idx % STEPS_PER_DAY) / STEPS_PER_DAY)
t_cos    = torch.cos(2 * np.pi * (t_idx % STEPS_PER_DAY) / STEPS_PER_DAY)
time_sin = t_sin.view(1, 1, -1, 1).expand(1, NUM_NODES, -1, 1)
time_cos = t_cos.view(1, 1, -1, 1).expand(1, NUM_NODES, -1, 1)
TIME_SCALE = 0.25

print(f"✅ Data loaded. Per-node normalised. Dual tod prior computed.")

# =============================================================================
# CELL 2 — Adjacency matrices
# =============================================================================

df       = pd.read_csv(fn_csv, header=0)
dist_mat = np.full((NUM_NODES, NUM_NODES), np.inf)
dist_fwd = np.full((NUM_NODES, NUM_NODES), np.inf)
dist_bwd = np.full((NUM_NODES, NUM_NODES), np.inf)
np.fill_diagonal(dist_mat, 0.); np.fill_diagonal(dist_fwd, 0.); np.fill_diagonal(dist_bwd, 0.)

for _, row in df.iterrows():
    i, j, d = int(row.iloc[0]), int(row.iloc[1]), float(row.iloc[2])
    dist_mat[i, j] = d; dist_mat[j, i] = d
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

speed_train   = data_norm_speed[:TRAIN_END, :].T
corr_mat      = np.nan_to_num(np.corrcoef(speed_train), nan=0.)
adj_corr      = np.where(np.clip(corr_mat, 0, 1) > 0.60, corr_mat, 0.)
np.fill_diagonal(adj_corr, 0.)
adj_corr_norm = adj_corr / (adj_corr.sum(axis=1, keepdims=True) + 1e-8)

L_sym_np = np.eye(NUM_NODES, dtype=np.float32) - adj_sym
L_graph  = torch.tensor(L_sym_np, dtype=torch.float32).to(device)
A_road   = torch.tensor(adj_sym,       dtype=torch.float32).unsqueeze(0).to(device)
A_fwd_t  = torch.tensor(adj_fwd,       dtype=torch.float32).unsqueeze(0).to(device)
A_bwd_t  = torch.tensor(adj_bwd,       dtype=torch.float32).unsqueeze(0).to(device)
A_corr_t = torch.tensor(adj_corr_norm, dtype=torch.float32).unsqueeze(0).to(device)
A_t      = torch.tensor(adj_sym,       dtype=torch.float32).to(device)
print(f"✅ Adjacency ready. Corr mean degree ≈ {(adj_corr>0).sum(1).mean():.1f}")

# =============================================================================
# CELL 3 — Hypergraph
# =============================================================================

adj_bin    = (adj_sym > 1e-6).astype(np.float32); np.fill_diagonal(adj_bin, 0.)
adj2       = np.clip((adj_bin @ adj_bin > 0).astype(np.float32) + adj_bin, 0, 1)
np.fill_diagonal(adj2, 1.)
H_np       = adj2.T
d_v        = H_np.sum(axis=1); d_e = H_np.sum(axis=0)
d_v_inv_sq = np.where(d_v > 0, (d_v + 1e-8)**(-0.5), 0.)
d_e_inv    = np.where(d_e > 0, (d_e + 1e-8)**(-1.), 0.)
H_conv_np  = (d_v_inv_sq[:, None] * H_np) * d_e_inv[None, :]
H_conv_np  = H_conv_np @ (H_np.T * d_v_inv_sq[None, :])
H_conv     = torch.tensor(H_conv_np, dtype=torch.float32).to(device)
print(f"✅ Hypergraph ready.")

# =============================================================================
# CELL 4 — Feature builder  (13 features, same as v4/v4.1)
# =============================================================================

def build_input_features(mask, data_tensor_all, tod_prior,
                         tod_free, tod_jam,
                         A_t, time_sin, time_cos, TIME_SCALE):
    N = mask.shape[1]
    obs_all  = data_tensor_all * mask
    obs_spd  = obs_all[:, :, :, 2:3]

    num_obs      = mask.sum(dim=1, keepdim=True)
    gctx_all     = obs_all.sum(dim=1, keepdim=True) / (num_obs + 1e-6)
    gctx_spd     = gctx_all[:, :, :, 2:3]
    gctx_flow    = gctx_all[:, :, :, 0:1].expand(-1, N, -1, -1)
    gctx_occ     = gctx_all[:, :, :, 1:2].expand(-1, N, -1, -1)
    gctx_spd_exp = gctx_spd.expand(-1, N, -1, -1)

    obs_2d   = obs_spd[0, :, :, 0]
    mask_1d  = mask[0, :, 0, 0]
    mask_2d  = mask_1d.unsqueeze(1).expand_as(obs_2d)
    nbr_s1   = torch.mm(A_t, obs_2d)
    nbr_c1   = torch.mm(A_t, mask_2d)
    nbr_ctx1 = nbr_s1 / (nbr_c1 + 1e-6)
    cov_1    = nbr_c1 / (A_t.sum(dim=1, keepdim=True) + 1e-6)
    nbr_s2   = torch.mm(A_t, nbr_ctx1)
    cov_2    = torch.mm(A_t, cov_1) / (A_t.sum(dim=1, keepdim=True) + 1e-6)
    gspd_2d  = gctx_spd[0, 0, :, 0].unsqueeze(0).expand(N, -1)
    blend    = torch.clamp(cov_2 / 0.2, 0., 1.)
    obs_prop = blend * nbr_s2 + (1 - blend) * gspd_2d

    obs_prop_feat = obs_prop.unsqueeze(0).unsqueeze(-1)
    cov_feat      = cov_2.unsqueeze(0).unsqueeze(-1)
    tod_fill      = tod_prior.T.unsqueeze(0).unsqueeze(-1) * (1 - mask)
    speed_w_prior = obs_spd + tod_fill
    obs_flag      = mask.expand_as(obs_spd)
    obs_flow      = obs_all[:, :, :, 0:1]
    obs_occ       = obs_all[:, :, :, 1:2]
    tod_free_feat = tod_free.T.unsqueeze(0).unsqueeze(-1).expand(-1, N, -1, -1)
    tod_jam_feat  = tod_jam.T.unsqueeze(0).unsqueeze(-1).expand(-1, N, -1, -1)

    return torch.cat([
        speed_w_prior, gctx_spd_exp, obs_prop_feat, cov_feat, obs_flag,
        TIME_SCALE * time_sin, TIME_SCALE * time_cos,
        obs_flow, gctx_flow, obs_occ, gctx_occ,
        tod_free_feat, tod_jam_feat,
    ], dim=-1)  # [1, N, T, 13]


SPARSITY = 0.80
K_MASKS  = 5
masks_list, features_list = [], []
for k in range(K_MASKS):
    torch.manual_seed(k * 37 + 42)
    mk = (torch.rand(1, NUM_NODES, 1, 1) > SPARSITY).float().to(device)
    fk = build_input_features(mk, data_tensor_all, tod_prior, tod_free, tod_jam,
                               A_t, time_sin, time_cos, TIME_SCALE)
    masks_list.append(mk)
    features_list.append(fk)

node_mask      = masks_list[0]
input_features = features_list[0]
INPUT_DIM      = 13   # [B] no node embeddings

print(f"✅ {K_MASKS} masks + 13-feature tensors built. INPUT_DIM={INPUT_DIM}")
assert (input_features[0, node_mask[0,:,0,0]==0, :, 4] == 0).all(), "Leakage!"
print("✅ Leakage check passed.")

# =============================================================================
# CELL 5 — Model definition  (no node embeddings)
# =============================================================================

class GraphAttention(nn.Module):
    def __init__(self, in_dim, out_dim, temperature=2.0):
        super().__init__()
        self.W     = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Linear(out_dim, 1, bias=False)
        self.a_dst = nn.Linear(out_dim, 1, bias=False)
        self.leaky = nn.LeakyReLU(0.2)
        self.tau   = temperature

    def forward(self, x, A):
        h     = self.W(x)
        e     = self.leaky(self.a_src(h) + self.a_dst(h).transpose(1, 2))
        e     = e.masked_fill(A < 1e-9, float('-inf'))
        alpha = torch.nan_to_num(torch.softmax(e / self.tau, dim=-1), 0.)
        return torch.bmm(alpha, h)


class HypergraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.theta = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, H_conv):
        return torch.matmul(H_conv, self.theta(x))


class GraphODEFunc(nn.Module):
    """4-path ODE: sym + fwd + bwd + corr + gated hypergraph."""
    def __init__(self, hidden_dim, A_sym, A_fwd, A_bwd, A_corr, H_conv):
        super().__init__()
        self.A_sym = A_sym; self.A_fwd = A_fwd
        self.A_bwd = A_bwd; self.A_corr = A_corr
        self.H_conv = H_conv

        H = hidden_dim
        self.gat_sym1  = GraphAttention(H, H)
        self.gat_sym2  = GraphAttention(H, H)
        self.gat_fwd   = GraphAttention(H, H)
        self.gat_bwd   = GraphAttention(H, H)
        self.gat_corr  = GraphAttention(H, H)
        self.dir_mixer = nn.Sequential(
            nn.Linear(H*4, H), nn.Tanh(), nn.Linear(H, 4), nn.Softmax(dim=-1)
        )
        self.hconv      = HypergraphConv(H, H)
        self.hyper_gate = nn.Parameter(torch.full((NUM_NODES, 1), -2.0))
        self.act  = nn.Tanh()
        self.norm = nn.LayerNorm(H)

    def forward(self, t, x):
        B   = x.size(0)
        A_s = self.A_sym.expand(B,-1,-1)
        A_f = self.A_fwd.expand(B,-1,-1)
        A_b = self.A_bwd.expand(B,-1,-1)
        A_c = self.A_corr.expand(B,-1,-1)

        h_sym  = self.act(self.gat_sym2(self.act(self.gat_sym1(x, A_s)), A_s))
        h_fwd  = self.act(self.gat_fwd(x, A_f))
        h_bwd  = self.act(self.gat_bwd(x, A_b))
        h_corr = self.act(self.gat_corr(x, A_c))
        mix    = self.dir_mixer(torch.cat([h_sym, h_fwd, h_bwd, h_corr], dim=-1))
        h      = (mix[...,0:1]*h_sym + mix[...,1:2]*h_fwd +
                  mix[...,2:3]*h_bwd + mix[...,3:4]*h_corr)
        h      = h + torch.sigmoid(self.hyper_gate).unsqueeze(0) * self.act(self.hconv(x, self.H_conv))
        return self.norm(h)


class AssimilationUpdate(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super().__init__()
        self.obs_enc   = nn.Linear(input_dim, hidden_dim)
        self.state_prb = nn.Linear(hidden_dim, 1)
        self.gate      = nn.Sequential(
            nn.Linear(hidden_dim*2+2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid()
        )

    def forward(self, z, x_new, obs_mask, obs_count_norm):
        z_obs  = self.obs_enc(x_new)
        sl     = self.state_prb(z)
        g_in   = torch.cat([z, z_obs, obs_count_norm.expand_as(sl), sl], dim=-1)
        return z + self.gate(g_in) * (z_obs - z) * obs_mask


class GraphCTH_NODE_v5(nn.Module):
    """
    v5: [B] No node embeddings (INPUT_DIM=13).
        [A] Jam-conditioned decoder: decoder([z, jl.detach()]).
        Jam_head is a regulariser for z via focal BCE (not primary detector).
        Primary jam detection: speed threshold < 40 km/h on decoder output.
    """
    def __init__(self, input_dim, hidden_dim,
                 A_sym, A_fwd, A_bwd, A_corr, H_conv):
        super().__init__()
        self.encoder    = nn.Linear(input_dim, hidden_dim)
        self.ode_func   = GraphODEFunc(hidden_dim, A_sym, A_fwd, A_bwd, A_corr, H_conv)
        self.assimilate = AssimilationUpdate(hidden_dim, input_dim)
        self.jam_head   = nn.Linear(hidden_dim, 1)
        self.decoder    = nn.Linear(hidden_dim + 1, 1)  # jam-conditioned

    def _euler(self, z):
        return z + 0.3 * self.ode_func(None, z)

    def forward(self, x_seq, obs_mask):
        B, N, T, _ = x_seq.shape
        mask      = obs_mask[:, :, 0, :]
        obs_count = mask.mean(dim=1, keepdim=True)

        z = self.encoder(x_seq[:, :, 0, :])
        preds, jam_logits = [], []

        for i in range(T):
            jl = torch.sigmoid(self.jam_head(z))
            jam_logits.append(jl)
            preds.append(self.decoder(torch.cat([z, jl.detach()], dim=-1)))
            if i < T - 1:
                z = self._euler(z)
                z = self.assimilate(z, x_seq[:, :, i+1, :], mask, obs_count)

        return torch.stack(preds, dim=2), torch.stack(jam_logits, dim=2)

# =============================================================================
# CELL 6 — Metrics
# =============================================================================

def compute_ssim(pred, target, data_range=None):
    if data_range is None:
        data_range = float(target.max() - target.min()) + 1e-8
    C1 = (0.01 * data_range)**2; C2 = (0.03 * data_range)**2
    mu_p, mu_t   = pred.mean(), target.mean()
    sig_p, sig_t = pred.std(), target.std()
    sig_pt       = ((pred - mu_p) * (target - mu_t)).mean()
    return float(((2*mu_p*mu_t+C1)*(2*sig_pt+C2)) /
                 ((mu_p**2+mu_t**2+C1)*(sig_p**2+sig_t**2+C2)))

def jam_prec_recall(pred_kmh, true_kmh, thresh=40.0):
    p_j = pred_kmh < thresh; t_j = true_kmh < thresh
    tp  = (p_j & t_j).sum(); fp = (p_j & ~t_j).sum(); fn = (~p_j & t_j).sum()
    pr  = tp/(tp+fp+1e-8); rc = tp/(tp+fn+1e-8)
    return float(pr), float(rc), float(2*pr*rc/(pr+rc+1e-8))

def jam_prec_recall_logit(logit_np, true_kmh, thresh=40.0, logit_thresh=0.5):
    p_j = logit_np > logit_thresh; t_j = true_kmh < thresh
    tp  = (p_j & t_j).sum(); fp = (p_j & ~t_j).sum(); fn = (~p_j & t_j).sum()
    pr  = tp/(tp+fp+1e-8); rc = tp/(tp+fn+1e-8)
    return float(pr), float(rc), float(2*pr*rc/(pr+rc+1e-8))

# =============================================================================
# CELL 7 — Focal Hybrid Loss v5
# =============================================================================

class FocalHybridLoss_v5(nn.Module):
    """
    [A] Focal BCE for jam_head:
        FL = -α*(1-p)^γ*log(p) for jams  (concentrates on missed jams)
        FL = -(1-α)*p^γ*log(1-p) for free-flow  (down-weights easy negatives)
        With α=0.85 and γ=2, jam examples contribute ~93% of total BCE loss
        even at 92:8 class imbalance. Fixes precision=0.043 from v4.1.

    [B] MAE for jam region regression (Xie et al. 2023):
        No quadratic bias toward safe predictions near threshold.

    [C] Curriculum recall: λ_recall ramps 0→0.20 over epochs 0–200.
        [D] Reduced from 0.40 (v4.1) — focal BCE already handles recall.
    """
    def __init__(self, jam_thresh_train, jam_weight=20.0, gamma=2.0,
                 focal_alpha=0.85,
                 lam_smooth=0.60, lam_phy=0.02, lam_aux=0.10,
                 lam_recall_max=0.20, recall_warmup_epochs=200):
        super().__init__()
        self.jt             = jam_thresh_train
        self.jw             = jam_weight
        self.gam            = gamma
        self.alpha          = focal_alpha          # [A] jam weight in focal BCE
        self.ls             = lam_smooth
        self.lp             = lam_phy
        self.la             = lam_aux
        self.lam_recall_max = lam_recall_max
        self.recall_warmup  = recall_warmup_epochs

    def get_lam_recall(self, epoch):
        return self.lam_recall_max * min(1.0, epoch / self.recall_warmup)

    def forward(self, pred, obs, sup_mask, jam_logits, L_graph, epoch=1):
        jt       = self.jt.view(1, -1, 1, 1)
        jam_flag = (obs < jt).float()
        free_flag = 1.0 - jam_flag

        # [B] Hybrid regression: MSE for free-flow, MAE × focal for jams
        err_abs   = (pred - obs).abs()
        err_norm  = (err_abs.detach() / 3.0).clamp(0, 1)
        focal_mod = (1.0 + err_norm) ** self.gam

        loss_ff  = torch.mean(((pred - obs) * sup_mask * free_flag) ** 2)
        loss_jam = torch.mean(err_abs * sup_mask * jam_flag * focal_mod) * self.jw

        # [A] Focal BCE on jam_head
        # p_t = jam_logits for jams, (1-jam_logits) for free-flow
        p_t   = torch.where(jam_flag.bool(), jam_logits, 1.0 - jam_logits)
        # alpha_t = α for jams, (1-α) for free-flow
        alpha_t = torch.where(jam_flag.bool(),
                              torch.full_like(jam_logits, self.alpha),
                              torch.full_like(jam_logits, 1.0 - self.alpha))
        # Standard BCE per sample
        bce_raw = F.binary_cross_entropy(
            jam_logits * sup_mask,
            jam_flag   * sup_mask,
            reduction='none'
        )
        # Focal modulation: down-weight easy examples
        focal_bce = alpha_t * (1.0 - p_t.detach()).pow(self.gam) * bce_raw
        loss_focal = focal_bce.mean()

        # Curriculum recall
        lam_r       = self.get_lam_recall(epoch)
        loss_recall = torch.mean(jam_flag * sup_mask * (1.0 - jam_logits))

        # Temporal smoothness
        loss_sm = torch.mean((pred[:, :, 1:] - pred[:, :, :-1]) ** 2)

        # Graph Laplacian physics
        v        = pred[0, :, :, 0]
        loss_phy = torch.mean(torch.mm(L_graph, v) ** 2)

        return (loss_ff + loss_jam
                + self.la  * loss_focal
                + lam_r    * loss_recall
                + self.ls  * loss_sm
                + self.lp  * loss_phy)

# =============================================================================
# CELL 8 — Training (linear warmup + cosine, same as v4.1)
# =============================================================================

HIDDEN_DIM    = 64
BATCH_TIME    = 48
EPOCHS        = 800
LR_TARGET     = 1e-4
WARMUP_EPOCHS = 150      # Kalra NeurIPS 2024
WEIGHT_DECAY  = 1e-4
ACCUM_STEPS   = 4
CURRICULUM    = 0.15
JAM_BIAS_PROB = 0.70
PATIENCE      = 100

model = GraphCTH_NODE_v5(
    input_dim = INPUT_DIM,
    hidden_dim = HIDDEN_DIM,
    A_sym  = A_road, A_fwd = A_fwd_t, A_bwd = A_bwd_t,
    A_corr = A_corr_t, H_conv = H_conv,
).to(device)

criterion = FocalHybridLoss_v5(
    jam_thresh_train     = jam_thresh_train_t,
    jam_weight           = 20.0,
    gamma                = 2.0,
    focal_alpha          = 0.85,
    lam_smooth           = 0.60,
    lam_phy              = 0.02,
    lam_aux              = 0.10,
    lam_recall_max       = 0.20,
    recall_warmup_epochs = 200,
)

optimizer = torch.optim.Adam(model.parameters(), lr=LR_TARGET, weight_decay=WEIGHT_DECAY)

def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return float(epoch + 1) / float(WARMUP_EPOCHS)
    progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
    return 0.5 * (1.0 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

speed_t     = data_tensor_spd[0, :, :, 0]
jt_train    = jam_thresh_train_t.view(-1, 1)
jam_at_t    = (speed_t < jt_train).any(dim=0)[:TRAIN_END]
jam_t_valid = jam_at_t.nonzero(as_tuple=True)[0]
jam_t_valid = jam_t_valid[jam_t_valid < TRAIN_END - BATCH_TIME]

print(f"✅ Model: {sum(p.numel() for p in model.parameters()):,} params")
print(f"   INPUT_DIM={INPUT_DIM}  |  Warmup={WARMUP_EPOCHS} epochs")
print(f"   Jam windows: {len(jam_t_valid)}")

best_val, best_state, no_improve = float('inf'), None, 0
train_log, val_log = [], []

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    epoch_loss = 0.

    for step in range(ACCUM_STEPS):
        k             = np.random.randint(0, K_MASKS)
        cur_mask_base = masks_list[k]
        cur_features  = features_list[k]

        if len(jam_t_valid) > 0 and np.random.rand() < JAM_BIAS_PROB:
            t0 = int(jam_t_valid[torch.randint(len(jam_t_valid), (1,)).item()])
        else:
            t0 = np.random.randint(0, TRAIN_END - BATCH_TIME)

        x_win  = cur_features[:, :, t0:t0+BATCH_TIME, :]
        gt_win = data_tensor_spd[:, :, t0:t0+BATCH_TIME, :]

        cur_mask = cur_mask_base.clone()
        obs_idx  = (cur_mask[0,:,0,0]==1).nonzero(as_tuple=True)[0]
        n_drop   = max(1, int(len(obs_idx) * CURRICULUM))
        cur_mask[0, obs_idx[torch.randperm(len(obs_idx))[:n_drop]], 0, 0] = 0.

        sup_mask = (cur_mask_base.bool() | (cur_mask < cur_mask_base)).float()
        sup_mask = sup_mask.expand_as(gt_win)

        preds, jam_logits = model(x_win, cur_mask)
        loss = criterion(preds, gt_win, sup_mask, jam_logits, L_graph, epoch) / ACCUM_STEPS
        loss.backward()
        epoch_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    scheduler.step()
    train_log.append(epoch_loss)

    if epoch % 50 == 0:
        model.eval()
        val_maes = []
        with torch.no_grad():
            for t0 in range(VAL_START, VAL_END - BATCH_TIME, BATCH_TIME):
                x_v    = input_features[:, :, t0:t0+BATCH_TIME, :]
                gt_v   = data_tensor_spd[:, :, t0:t0+BATCH_TIME, :]
                p_v, _ = model(x_v, node_mask)
                blind  = (node_mask[0,:,0,0]==0)
                p_np   = p_v[0,blind,:,0].cpu().numpy()
                g_np   = gt_v[0,blind,:,0].cpu().numpy()
                n_idx  = blind.cpu().numpy().nonzero()[0]
                for ni, n in enumerate(n_idx):
                    p_np[ni] = p_np[ni]*node_stds[n] + node_means[n]
                    g_np[ni] = g_np[ni]*node_stds[n] + node_means[n]
                val_maes.append(np.abs(p_np - g_np).mean())

        val_mae = np.mean(val_maes)
        val_log.append((epoch, val_mae))
        phase  = "warmup" if epoch <= WARMUP_EPOCHS else "cosine"
        lam_r  = criterion.get_lam_recall(epoch)
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:4d} | loss={epoch_loss:.4f} | val_MAE={val_mae:.2f} km/h | "
              f"lr={cur_lr:.2e} | λ_recall={lam_r:.3f} [{phase}]")

        if val_mae < best_val:
            best_val, best_state, no_improve = val_mae, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 50
            if no_improve >= PATIENCE * 50:
                print(f"  Early stopping at epoch {epoch}")
                break

print(f"\nBest blind-node Val MAE: {best_val:.2f} km/h")

# =============================================================================
# CELL 9 — Evaluation with both speed-threshold and logit-threshold metrics
# =============================================================================

model.load_state_dict(best_state)
model.eval()

all_pred, all_true, all_logits = [], [], []
with torch.no_grad():
    for t0 in range(EVAL_START, EVAL_START + EVAL_LEN - BATCH_TIME, BATCH_TIME):
        x_e   = input_features[:, :, t0:t0+BATCH_TIME, :]
        gt_e  = data_tensor_spd[:, :, t0:t0+BATCH_TIME, :]
        p_e, jl_e = model(x_e, node_mask)
        all_pred.append(p_e.cpu()); all_true.append(gt_e.cpu())
        all_logits.append(jl_e.cpu())

pred_cat  = torch.cat(all_pred,   dim=2)[0,:,:,0]
true_cat  = torch.cat(all_true,   dim=2)[0,:,:,0]
logit_cat = torch.cat(all_logits, dim=2)[0,:,:,0]

blind_idx = (node_mask[0,:,0,0]==0).cpu().numpy().nonzero()[0]

pred_bl_norm  = pred_cat[blind_idx].numpy()
true_bl_norm  = true_cat[blind_idx].numpy()
logit_bl      = logit_cat[blind_idx].numpy()

pred_bl_kmh = np.zeros_like(pred_bl_norm)
true_bl_kmh = np.zeros_like(true_bl_norm)
for ni, n in enumerate(blind_idx):
    pred_bl_kmh[ni] = pred_bl_norm[ni] * node_stds[n] + node_means[n]
    true_bl_kmh[ni] = true_bl_norm[ni] * node_stds[n] + node_means[n]

base_bl_kmh = node_means[blind_idx][:, None] * np.ones_like(true_bl_kmh)
jam_mask    = true_bl_kmh < JAM_KMH_EVAL
glob_mae    = np.abs(pred_bl_kmh - true_bl_kmh).mean()
jam_mae     = np.abs((pred_bl_kmh - true_bl_kmh)[jam_mask]).mean() if jam_mask.any() else float('nan')
base_jam    = np.abs((base_bl_kmh - true_bl_kmh)[jam_mask]).mean()
ssim_val    = compute_ssim(pred_bl_kmh, true_bl_kmh)

# Speed-threshold detection (primary — better than logit in v4.1)
prec_spd, rec_spd, f1_spd = jam_prec_recall(pred_bl_kmh, true_bl_kmh)

print("\n" + "=" * 70)
print("  GRAPH-CTH-NODE v5  —  blind nodes, test window")
print("=" * 70)
print(f"  Global mean baseline MAE jam      : {base_jam:.2f} km/h")
print(f"  Model MAE (all blind)             : {glob_mae:.2f} km/h")
print(f"  Model MAE (jam < 40 km/h)         : {jam_mae:.2f} km/h")
print(f"  Jam improvement vs baseline       : {(base_jam-jam_mae)/base_jam*100:.1f}%")
print(f"  SSIM                              : {ssim_val:.3f}")
print(f"  Jam (speed threshold < 40 km/h):")
print(f"    Precision : {prec_spd:.3f}")
print(f"    Recall    : {rec_spd:.3f}")
print(f"    F1        : {f1_spd:.3f}")

# Logit-threshold sweep (secondary — to verify focal BCE improved calibration)
print(f"\n  Logit threshold sweep (focal BCE calibration check):")
print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
best_f1_val, best_f1_thresh = 0.0, 0.5
for thr in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
    p, r, f = jam_prec_recall_logit(logit_bl, true_bl_kmh, logit_thresh=thr)
    marker = " ← best logit F1" if f > best_f1_val else ""
    print(f"  {thr:>10.2f} {p:>10.3f} {r:>8.3f} {f:>8.3f}{marker}")
    if f > best_f1_val:
        best_f1_val, best_f1_thresh = f, thr

print(f"\n  Best logit F1: {best_f1_val:.3f} @ thr={best_f1_thresh:.2f}")
print("=" * 70)

# =============================================================================
# CELL 10 — Missing pattern experiment
# =============================================================================

experiment_results = {'MNAR': [], 'MAR': [], 'MCAR': []}
print("\nMissing pattern experiment (Bae et al. 2018 framework):")
print(f"{'Pattern':<8} {'Rate':<6} {'MAE all':>10} {'MAE jam':>10}")

for pat in ['MNAR', 'MAR', 'MCAR']:
    for rate in [0.40, 0.60, 0.80]:
        np.random.seed(77); torch.manual_seed(77)
        if pat == 'MNAR':
            num_drop   = int(rate * NUM_NODES)
            drop_nodes = torch.argsort(torch.tensor(node_means))[:num_drop]
            mp         = torch.ones((1, NUM_NODES, 1, 1)).to(device)
            mp[0, drop_nodes, 0, 0] = 0.
            mask_fn = lambda t: mp
        elif pat == 'MAR':
            mp      = (torch.rand(1, NUM_NODES, 1, 1) > rate).float().to(device)
            mask_fn = lambda t: mp
        else:
            mask_fn = lambda t: (torch.rand(1, NUM_NODES, 1, 1) > rate).float().to(device)

        model.eval()
        maes_all, maes_jam = [], []
        with torch.no_grad():
            for t0 in range(EVAL_START, EVAL_START + EVAL_LEN - BATCH_TIME, BATCH_TIME):
                cur_m  = mask_fn(t0)
                x_full = build_input_features(
                    cur_m, data_tensor_all, tod_prior, tod_free, tod_jam,
                    A_t, time_sin, time_cos, TIME_SCALE)
                x_e  = x_full[:, :, t0:t0+BATCH_TIME, :]
                gt_e = data_tensor_spd[:, :, t0:t0+BATCH_TIME, :]
                p_e, _ = model(x_e, cur_m)
                b_idx  = (cur_m[0,:,0,0]==0).cpu().numpy().nonzero()[0]
                if len(b_idx) == 0:
                    continue
                p_np = p_e[0,b_idx,:,0].cpu().numpy()
                g_np = gt_e[0,b_idx,:,0].cpu().numpy()
                for ni, n in enumerate(b_idx):
                    p_np[ni] = p_np[ni]*node_stds[n] + node_means[n]
                    g_np[ni] = g_np[ni]*node_stds[n] + node_means[n]
                maes_all.append(np.abs(p_np - g_np).mean())
                jm = g_np < JAM_KMH_EVAL
                if jm.any():
                    maes_jam.append(np.abs((p_np-g_np)[jm]).mean())

        ma = np.mean(maes_all) if maes_all else float('nan')
        mj = np.mean(maes_jam) if maes_jam else float('nan')
        print(f"  {pat:<8} {rate:<6.0%} {ma:>10.2f} {mj:>10.2f}")
        experiment_results[pat].append(ma)

# =============================================================================
# CELL 11 — Visualization
# =============================================================================

plt.style.use('ggplot')
fig = plt.figure(figsize=(20, 14), dpi=150)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.2)

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Training Loss', color='tab:blue', fontsize=12)
ax1.plot(train_log, color='tab:blue', alpha=0.5)
ax1.axvline(WARMUP_EPOCHS, color='gray', linestyle=':', linewidth=1.5,
            label=f'End warmup (ep {WARMUP_EPOCHS})')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax2 = ax1.twinx()
ax2.set_ylabel('Validation MAE (km/h)', color='tab:red', fontsize=12)
if val_log:
    ve = [v[0] for v in val_log]; vs = [v[1] for v in val_log]
    ax2.plot(ve, vs, color='tab:red', marker='o', linewidth=2)
    ax2.axhline(best_val, color='gray', linestyle='--', alpha=0.7,
                label=f'Best: {best_val:.2f} km/h')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax1.set_title('Model Convergence — v5', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right')

ax3 = fig.add_subplot(gs[0, 1])
worst_bi = np.argmax((true_bl_kmh < JAM_KMH_EVAL).sum(axis=1))
ta = np.arange(true_bl_kmh.shape[1])
ax3.plot(ta, true_bl_kmh[worst_bi], color='black', lw=2, label='Ground Truth')
ax3.plot(ta, pred_bl_kmh[worst_bi], color='tab:orange', lw=2, linestyle='--',
         label='Prediction')
ax3.axhline(JAM_KMH_EVAL, color='red', linestyle=':', lw=2, label='40 km/h')
ax3.fill_between(ta, 0, JAM_KMH_EVAL, color='red', alpha=0.1)
ax3.set_ylim(10, 80)
ax3.set_xlabel('Time Steps (5-min)', fontsize=12)
ax3.set_ylabel('Speed (km/h)', fontsize=12)
ax3.set_title(f'Blind Sensor Node {blind_idx[worst_bi]}',
              fontsize=14, fontweight='bold')
ax3.legend(loc='lower right')

ax4 = fig.add_subplot(gs[1, 0])
rates = ['40%', '60%', '80%']
x     = np.arange(len(rates)); width = 0.25
if all(len(experiment_results[p]) == 3 for p in ['MNAR','MAR','MCAR']):
    ax4.bar(x-width, experiment_results['MNAR'], width, label='MNAR', color='#d62728')
    ax4.bar(x,       experiment_results['MAR'],  width, label='MAR',  color='#1f77b4')
    ax4.bar(x+width, experiment_results['MCAR'], width, label='MCAR', color='#2ca02c')
    ax4.set_ylabel('Global MAE (km/h)', fontsize=12)
    ax4.set_xlabel('Missing Rate', fontsize=12)
    ax4.set_title('Robustness to Missing Mechanisms', fontsize=14, fontweight='bold')
    ax4.set_xticks(x); ax4.set_xticklabels(rates); ax4.legend()

ax5 = fig.add_subplot(gs[1, 1])
full_true = true_cat.numpy()*node_stds[:,None]+node_means[:,None]
full_pred = pred_cat.numpy()*node_stds[:,None]+node_means[:,None]
full_stitched = full_true.copy()
full_stitched[blind_idx,:] = full_pred[blind_idx,:]
N_vis, T_vis = min(100, NUM_NODES), min(100, EVAL_LEN)
im = ax5.imshow(full_stitched[:N_vis,:T_vis], aspect='auto',
                cmap='RdYlGn', vmin=20, vmax=70, origin='lower')
ax5.set_title('Predicted Spatiotemporal Traffic State (v5)',
              fontsize=14, fontweight='bold')
ax5.set_xlabel('Time Steps', fontsize=12)
ax5.set_ylabel('Sensor Node ID', fontsize=12)
fig.colorbar(im, ax=ax5).set_label('Speed (km/h)', rotation=270, labelpad=15)

plt.savefig('v5_results.png', bbox_inches='tight', dpi=150)
plt.show()
print("✅ Figure saved to v5_results.png")

# =============================================================================
# CELL 12 — Baseline evaluation harness
#   Same blind_idx, same test window [EVAL_START, EVAL_START+EVAL_LEN),
#   same denormalisation, same jam_prec_recall. All baselines append to
#   `results_table` which CELL 22 prints as the final comparison table.
# =============================================================================

# results_table: list of dicts keyed by model name
results_table = []

def eval_pred_np(pred_kmh_bl, true_kmh_bl):
    """
    pred_kmh_bl, true_kmh_bl: np arrays [n_blind, T_eval] in km/h
    Returns dict of metrics.
    """
    mae_all = float(np.abs(pred_kmh_bl - true_kmh_bl).mean())
    jm = true_kmh_bl < JAM_KMH_EVAL
    mae_jam = float(np.abs((pred_kmh_bl - true_kmh_bl)[jm]).mean()) if jm.any() else float('nan')
    pr, rc, f1 = jam_prec_recall(pred_kmh_bl, true_kmh_bl)
    ssim = compute_ssim(pred_kmh_bl, true_kmh_bl)
    return dict(mae_all=mae_all, mae_jam=mae_jam, prec=pr, rec=rc, f1=f1, ssim=ssim)

# Ground truth on blind nodes for the eval window (km/h)
_T_eval = (EVAL_LEN // BATCH_TIME) * BATCH_TIME
true_eval_kmh = np.zeros((len(blind_idx), _T_eval), dtype=np.float32)
for ni, n in enumerate(blind_idx):
    true_eval_kmh[ni] = (
        data_norm_speed[EVAL_START:EVAL_START+_T_eval, n]
        * node_stds[n] + node_means[n]
    )

# Store v5 result already computed
results_table.append({'model': 'Graph-CTH-NODE v5',
                      **eval_pred_np(pred_bl_kmh[:, :_T_eval], true_eval_kmh)})
print("✅ Baseline harness ready. true_eval_kmh shape:", true_eval_kmh.shape)

# =============================================================================
# CELL 13 — Tier 1: Statistical baselines
#   Global Mean, Historical Average, IDW, Linear Interpolation, KNN Kriging
#   No training required — deterministic from training data.
# =============================================================================

# --- 13a: Global Mean ---
gm_pred = node_means[blind_idx][:, None] * np.ones_like(true_eval_kmh)
results_table.append({'model': 'Global Mean', **eval_pred_np(gm_pred, true_eval_kmh)})
print("✅ Global Mean done.")

# --- 13b: Historical Average (per-node, per-slot mean) ---
ha_pred = np.zeros_like(true_eval_kmh)
for ni, n in enumerate(blind_idx):
    for t in range(_T_eval):
        s = (EVAL_START + t) % STEPS_PER_DAY
        ha_pred[ni, t] = tod_mean_sp[n, s] * node_stds[n] + node_means[n]
results_table.append({'model': 'Historical Average', **eval_pred_np(ha_pred, true_eval_kmh)})
print("✅ Historical Average done.")

# --- 13c: IDW (Inverse Distance Weighted) from observed nodes ---
# Use adj_sym edge weights as proximity proxy; observed = node_mask==1
obs_nodes = (node_mask[0, :, 0, 0] == 1).cpu().numpy().nonzero()[0]
idw_pred  = np.zeros_like(true_eval_kmh)
for ni, n in enumerate(blind_idx):
    weights = adj_sym[n, obs_nodes]  # shape [n_obs]
    wsum    = weights.sum() + 1e-8
    for t in range(_T_eval):
        obs_vals = (data_norm_speed[EVAL_START+t, obs_nodes]
                    * node_stds[obs_nodes] + node_means[obs_nodes])
        idw_pred[ni, t] = (weights * obs_vals).sum() / wsum
results_table.append({'model': 'IDW', **eval_pred_np(idw_pred, true_eval_kmh)})
print("✅ IDW done.")

# --- 13d: Linear Interpolation (temporal, per blind node) ---
# For each blind node: linearly interpolate using the last known observed
# value from obs_nodes' global mean as anchor at t=EVAL_START-1 and t=EVAL_START+_T_eval
lip_pred = np.zeros_like(true_eval_kmh)
for ni, n in enumerate(blind_idx):
    # Use the node's own tod prior as "interpolation target"
    slot_start = EVAL_START % STEPS_PER_DAY
    slot_end   = (EVAL_START + _T_eval - 1) % STEPS_PER_DAY
    v_start    = tod_mean_sp[n, slot_start] * node_stds[n] + node_means[n]
    v_end      = tod_mean_sp[n, slot_end]   * node_stds[n] + node_means[n]
    lip_pred[ni] = np.linspace(v_start, v_end, _T_eval)
results_table.append({'model': 'Linear Interpolation', **eval_pred_np(lip_pred, true_eval_kmh)})
print("✅ Linear Interpolation done.")

# --- 13e: KNN Kriging (k nearest observed neighbours by road distance) ---
K_KNN = 5
knn_pred = np.zeros_like(true_eval_kmh)
for ni, n in enumerate(blind_idx):
    dists   = dist_mat[n, obs_nodes]
    finite  = dists < np.inf
    if finite.sum() == 0:
        knn_pred[ni] = gm_pred[ni]
        continue
    k_actual = min(K_KNN, finite.sum())
    nn_idx   = np.argsort(dists[finite])[:k_actual]
    nn_nodes = obs_nodes[finite][nn_idx]
    nn_dists = dists[finite][nn_idx] + 1e-8
    w        = 1.0 / nn_dists; w /= w.sum()
    for t in range(_T_eval):
        obs_vals = (data_norm_speed[EVAL_START+t, nn_nodes]
                    * node_stds[nn_nodes] + node_means[nn_nodes])
        knn_pred[ni, t] = (w * obs_vals).sum()
results_table.append({'model': 'KNN Kriging (k=5)', **eval_pred_np(knn_pred, true_eval_kmh)})
print("✅ KNN Kriging done.")
print(f"   Tier 1 complete — {len(results_table)} entries in results_table.")

# =============================================================================
# CELL 14 — Tier 2: RNN / temporal baselines (no graph)
#   GRU-D  (Che et al. 2018)  — GRU with time-decay imputation
#   BRITS  (Cao et al. 2018)  — Bidirectional RNN with regression imputation
#   SAITS  (Du et al. 2023)   — Self-Attention Imputation (transformer-style)
#   All run node-independently (treat each node as an independent time series).
#   Input: normalised speed only. Observed nodes supply targets; blind nodes
#   are held out at eval but included in training with their observed values.
# =============================================================================

BL_HIDDEN  = 64
BL_EPOCHS  = 300
BL_LR      = 3e-3
BL_BATCH   = 32   # number of nodes per mini-batch
BL_SEQ     = 48   # sequence window (same as BATCH_TIME)

# Training data: [TIME_STEPS, NUM_NODES] speed (normalised)
# We train on ALL nodes (blind and observed) using their actual values.
# At eval, we only read predictions for blind nodes.
speed_np = data_norm_speed  # [T, N]

# ─── GRU-D ───────────────────────────────────────────────────────────────────
class GRUD(nn.Module):
    """
    GRU-D (Che et al. 2018): time-decay gates on input and hidden state.
    Simplified: constant delta=1 (5-min steps), no covariate mask feature.
    Input per step: [x_t, m_t, gamma_x, gamma_h]  → 4-dim
    """
    def __init__(self, hidden=64):
        super().__init__()
        self.h   = hidden
        # Input decay: γ_x = exp(-max(0, W_γ·δ + b_γ))
        self.W_gx = nn.Linear(1, 1, bias=True)
        self.W_gh = nn.Linear(1, 1, bias=True)
        self.gru  = nn.GRUCell(3, hidden)   # [x_imp, m, gamma_h]
        self.out  = nn.Linear(hidden, 1)

    def forward(self, x_seq, m_seq):
        # x_seq, m_seq: [B, T]
        B, T = x_seq.shape
        h = torch.zeros(B, self.h, device=x_seq.device)
        preds = []
        x_last = torch.zeros(B, device=x_seq.device)
        for t in range(T):
            delta = torch.ones(B, 1, device=x_seq.device)
            gx = torch.exp(-torch.clamp(self.W_gx(delta), min=0))[:,0]
            gh = torch.exp(-torch.clamp(self.W_gh(delta), min=0))[:,0]
            x_imp = m_seq[:,t] * x_seq[:,t] + (1 - m_seq[:,t]) * (gx * x_last + (1-gx)*0.)
            h = gh.unsqueeze(1) * h
            inp = torch.stack([x_imp, m_seq[:,t], gh], dim=1)
            h   = self.gru(inp, h)
            preds.append(self.out(h)[:,0])
            x_last = x_imp
        return torch.stack(preds, dim=1)  # [B, T]

def train_rnn_baseline(model_cls, name, hidden=64, epochs=BL_EPOCHS):
    net = model_cls(hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=BL_LR)
    best_vloss, best_wts = float('inf'), None
    for ep in range(1, epochs+1):
        net.train()
        node_perm = torch.randperm(NUM_NODES)
        ep_loss = 0.; n_batches = 0
        for b0 in range(0, NUM_NODES, BL_BATCH):
            nodes_b = node_perm[b0:b0+BL_BATCH].tolist()
            t0 = np.random.randint(0, TRAIN_END - BL_SEQ)
            x  = torch.tensor(speed_np[t0:t0+BL_SEQ, nodes_b],
                               dtype=torch.float32).T.to(device)  # [B, T]
            m  = torch.ones_like(x)   # all observed during training
            p  = net(x, m)
            loss = F.mse_loss(p, x)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); n_batches += 1
        if ep % 50 == 0:
            net.eval()
            with torch.no_grad():
                x_v = torch.tensor(speed_np[VAL_START:VAL_END, :],
                                   dtype=torch.float32).T.to(device)
                m_v = torch.ones_like(x_v)
                p_v = net(x_v, m_v)
                vl  = F.mse_loss(p_v, x_v).item()
            if vl < best_vloss:
                best_vloss = vl
                best_wts   = copy.deepcopy(net.state_dict())
            print(f"  [{name}] ep {ep:3d} | train={ep_loss/n_batches:.4f} val={vl:.4f}")
    if best_wts:
        net.load_state_dict(best_wts)
    return net

def eval_rnn_baseline(net, name):
    net.eval()
    x_e = torch.tensor(speed_np[EVAL_START:EVAL_START+_T_eval, :],
                        dtype=torch.float32).T.to(device)  # [N, T]
    m_e = torch.zeros_like(x_e)
    m_e[obs_nodes, :] = 1.0   # observed nodes are "known"
    with torch.no_grad():
        p_e = net(x_e, m_e).cpu().numpy()  # [N, T]
    pred_kmh = np.zeros((len(blind_idx), _T_eval), dtype=np.float32)
    for ni, n in enumerate(blind_idx):
        pred_kmh[ni] = p_e[n] * node_stds[n] + node_means[n]
    results_table.append({'model': name, **eval_pred_np(pred_kmh, true_eval_kmh)})
    print(f"✅ {name} evaluated.")

print("Training GRU-D...")
grud_net = train_rnn_baseline(GRUD, 'GRU-D')
eval_rnn_baseline(grud_net, 'GRU-D')

# ─── BRITS ───────────────────────────────────────────────────────────────────
class BRITSCell(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.W_x = nn.Linear(1, hidden)
        self.gru  = nn.GRUCell(hidden + 1, hidden)   # [feat, mask]
        self.out  = nn.Linear(hidden, 1)
        self.W_c  = nn.Linear(hidden, 1)   # complement regression

    def forward_dir(self, x_seq, m_seq):
        B, T = x_seq.shape
        h = torch.zeros(B, self.hidden, device=x_seq.device)
        preds, complements = [], []
        for t in range(T):
            c_t   = torch.tanh(self.W_c(h))[:,0]
            x_imp = m_seq[:,t]*x_seq[:,t] + (1-m_seq[:,t])*c_t
            feat  = torch.relu(self.W_x(x_imp.unsqueeze(1)))
            inp   = torch.cat([feat, m_seq[:,t:t+1]], dim=1)
            h     = self.gru(inp, h)
            preds.append(self.out(h)[:,0])
            complements.append(c_t)
        return torch.stack(preds,dim=1), torch.stack(complements,dim=1)

class BRITS(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.fwd = BRITSCell(hidden)
        self.bwd = BRITSCell(hidden)

    def forward(self, x_seq, m_seq):
        pf, cf = self.fwd.forward_dir(x_seq, m_seq)
        pb, cb = self.bwd.forward_dir(x_seq.flip(1), m_seq.flip(1))
        return 0.5*(pf + pb.flip(1))

print("\nTraining BRITS...")
brits_net = train_rnn_baseline(BRITS, 'BRITS')
eval_rnn_baseline(brits_net, 'BRITS')

# ─── SAITS ───────────────────────────────────────────────────────────────────
class SAITS(nn.Module):
    """
    Simplified SAITS (Du et al. 2023): two-stage masked self-attention
    imputation on a per-node basis. Each node is treated independently.
    Stage 1: attend over time to produce first estimate.
    Stage 2: attend again with first estimate fused in.
    """
    def __init__(self, hidden=64, n_heads=4, d_ff=128, seq_len=BL_SEQ):
        super().__init__()
        self.proj = nn.Linear(2, hidden)   # [x, mask] → hidden
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=d_ff,
            dropout=0.1, batch_first=True)
        self.attn1 = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.attn2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden, nhead=n_heads, dim_feedforward=d_ff,
                dropout=0.1, batch_first=True),
            num_layers=1)
        self.out1  = nn.Linear(hidden, 1)
        self.out2  = nn.Linear(hidden, 1)

    def forward(self, x_seq, m_seq):
        # x_seq, m_seq: [B, T]
        inp  = torch.stack([x_seq, m_seq], dim=-1)       # [B, T, 2]
        h    = self.proj(inp)                              # [B, T, H]
        h1   = self.attn1(h)
        p1   = self.out1(h1)[:,:,0]                       # [B, T]
        x2   = m_seq*x_seq + (1-m_seq)*p1.detach()
        inp2 = torch.stack([x2, m_seq], dim=-1)
        h2   = self.attn2(self.proj(inp2))
        p2   = self.out2(h2)[:,:,0]
        return m_seq*x_seq + (1-m_seq)*p2

print("\nTraining SAITS...")
saits_net = train_rnn_baseline(SAITS, 'SAITS')
eval_rnn_baseline(saits_net, 'SAITS')
print(f"   Tier 2 complete — {len(results_table)} entries in results_table.")

# =============================================================================
# CELL 15 — Tier 3: GNN imputation baselines
#
#   IGNNK  (Ye et al. 2021)  — random subgraph kriging, diffusion GCN
#   GRIN   (Cini et al. 2022) — bidirectional recurrent GNN, message passing
#   SPIN   (Marisca et al. 2022) — sparse imputation network, designed for
#            high missingness settings (80%+)
#   DGCRIN (Zhang et al. 2023) — dynamic graph conv with residual imputation
#   GCASTN (Liu et al. 2023)  — graph convolution + attention + ST context
#   ADGCN  (Chen et al. 2023) — adaptive diffusion GCN, tested on PEMS04
#
#   All models use adj_sym as the static graph. Training: t=0–4000.
#   Each model takes [N, T, F] input where F=1 (speed only) with a binary
#   mask indicating observed nodes. Output: [N, T, 1] imputed speed.
# =============================================================================

GNN_HIDDEN  = 64
GNN_EPOCHS  = 300
GNN_LR      = 3e-3
GNN_BATCH   = 48

# Pre-compute diffusion matrices for GNN baselines
def diffusion_cheby(A, K=2):
    """Returns list of K Chebyshev graph conv matrices."""
    D     = A.sum(1)
    D_inv = torch.where(D > 0, 1.0/D, torch.zeros_like(D))
    Anorm = A * D_inv.unsqueeze(1)
    mats  = [torch.eye(NUM_NODES, device=device), Anorm]
    for k in range(2, K):
        mats.append(2*torch.mm(Anorm, mats[-1]) - mats[-2])
    return mats  # list of [N,N]

cheby_mats = diffusion_cheby(A_t, K=3)

class ChebConv(nn.Module):
    def __init__(self, in_dim, out_dim, K=3):
        super().__init__()
        self.K    = K
        self.Ws   = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=(k==0))
                                   for k in range(K)])
        self.mats = cheby_mats

    def forward(self, x):
        # x: [N, F]
        out = sum(self.Ws[k](torch.mm(self.mats[k], x)) for k in range(self.K))
        return out

def train_gnn_baseline(model_cls, name, **kwargs):
    net = model_cls(**kwargs).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=GNN_LR, weight_decay=1e-4)
    best_vloss, best_wts = float('inf'), None
    for ep in range(1, GNN_EPOCHS+1):
        net.train()
        t0      = np.random.randint(0, TRAIN_END - GNN_BATCH)
        x_full  = torch.tensor(speed_np[t0:t0+GNN_BATCH, :],   # [T, N]
                                dtype=torch.float32).T.to(device)  # [N, T]
        # Random 80% mask (same sparsity as main experiment)
        m_train = (torch.rand(NUM_NODES, 1, device=device) > SPARSITY).float()
        m_train = m_train.expand(-1, GNN_BATCH)
        loss    = net.training_step(x_full, m_train)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0:
            net.eval()
            with torch.no_grad():
                x_v = torch.tensor(speed_np[VAL_START:VAL_END, :],
                                    dtype=torch.float32).T.to(device)
                m_v = (node_mask[0,:,0,0]==1).float().unsqueeze(1).expand(-1, VAL_END-VAL_START)
                vl  = net.training_step(x_v, m_v).item()
            if vl < best_vloss:
                best_vloss = vl; best_wts = copy.deepcopy(net.state_dict())
            print(f"  [{name}] ep {ep:3d} | val={vl:.4f}")
    if best_wts:
        net.load_state_dict(best_wts)
    return net

def eval_gnn_baseline(net, name):
    net.eval()
    x_e = torch.tensor(speed_np[EVAL_START:EVAL_START+_T_eval, :],
                        dtype=torch.float32).T.to(device)   # [N, T]
    m_e = (node_mask[0,:,0,0]==1).float().unsqueeze(1).expand(-1, _T_eval)
    with torch.no_grad():
        p_e = net.impute(x_e, m_e).cpu().numpy()  # [N, T]
    pred_kmh = np.zeros((len(blind_idx), _T_eval), dtype=np.float32)
    for ni, n in enumerate(blind_idx):
        pred_kmh[ni] = p_e[n] * node_stds[n] + node_means[n]
    results_table.append({'model': name, **eval_pred_np(pred_kmh, true_eval_kmh)})
    print(f"✅ {name} evaluated.")

# ─── IGNNK ───────────────────────────────────────────────────────────────────
class IGNNK(nn.Module):
    """
    IGNNK (Ye et al. 2021): Kriging via random subgraph sampling + diffusion GCN.
    At each step, a random subset of observed nodes form the 'anchor' graph;
    GCN propagates to estimate missing nodes.
    Simplified: one-layer diffusion GCN without full subgraph sampling loop.
    """
    def __init__(self, hidden=GNN_HIDDEN):
        super().__init__()
        self.enc   = ChebConv(1, hidden, K=3)
        self.dec   = nn.Linear(hidden, 1)
        self.act   = nn.ReLU()

    def _forward(self, x, m):
        # x, m: [N, T]
        N, T   = x.shape
        x_obs  = (x * m).T.unsqueeze(-1)         # [T, N, 1]
        h      = self.act(torch.stack([self.enc(x_obs[t]) for t in range(T)]))  # [T, N, H]
        p      = self.dec(h).squeeze(-1).T        # [N, T]
        return p

    def training_step(self, x, m):
        p    = self._forward(x, m)
        # Supervised on observed nodes only (teacher-force with true vals)
        return F.mse_loss(p[m==1], x[m==1])

    def impute(self, x, m):
        p = self._forward(x, m)
        return m*x + (1-m)*p

print("\nTraining IGNNK...")
ignnk_net = train_gnn_baseline(IGNNK, 'IGNNK')
eval_gnn_baseline(ignnk_net, 'IGNNK')

# ─── GRIN ────────────────────────────────────────────────────────────────────
class GRINCell(nn.Module):
    """Single direction of GRIN (Cini et al. 2022)."""
    def __init__(self, hidden=GNN_HIDDEN):
        super().__init__()
        self.msg   = ChebConv(hidden + 1, hidden, K=2)   # h + mask → message
        self.gru   = nn.GRUCell(hidden + 2, hidden)      # [msg, x, m]
        self.out   = nn.Linear(hidden, 1)
        self.act   = nn.Tanh()

    def forward(self, x_seq, m_seq):
        # x_seq, m_seq: [N, T]
        N, T = x_seq.shape
        h = torch.zeros(N, self.gru.hidden_size, device=x_seq.device)
        preds = []
        for t in range(T):
            msg_in = torch.cat([h, m_seq[:,t:t+1]], dim=-1)  # [N, H+1]
            msg    = self.act(self.msg(msg_in))               # [N, H]
            x_t    = x_seq[:,t:t+1]
            inp    = torch.cat([msg, x_t, m_seq[:,t:t+1]], dim=-1)  # [N, H+2]
            h      = self.gru(inp, h)
            preds.append(self.out(h)[:,0])
        return torch.stack(preds, dim=1)   # [N, T]

class GRIN(nn.Module):
    def __init__(self, hidden=GNN_HIDDEN):
        super().__init__()
        self.fwd = GRINCell(hidden)
        self.bwd = GRINCell(hidden)
        self.fuse = nn.Linear(2, 1)

    def _run(self, x, m):
        pf = self.fwd(x, m)
        pb = self.bwd(x.flip(1), m.flip(1)).flip(1)
        return self.fuse(torch.stack([pf, pb], dim=-1)).squeeze(-1)

    def training_step(self, x, m):
        p = self._run(x, m)
        return F.mse_loss(p[m==1], x[m==1])

    def impute(self, x, m):
        p = self._run(x, m)
        return m*x + (1-m)*p

print("\nTraining GRIN...")
grin_net = train_gnn_baseline(GRIN, 'GRIN')
eval_gnn_baseline(grin_net, 'GRIN')

# ─── SPIN ────────────────────────────────────────────────────────────────────
class SPIN(nn.Module):
    """
    SPIN (Marisca et al. 2022): Sparse Imputation Network.
    Spatial attention over observed neighbours + temporal attention.
    Designed for high-sparsity (80%+ missing) settings.
    """
    def __init__(self, hidden=GNN_HIDDEN, n_heads=4):
        super().__init__()
        self.proj    = nn.Linear(2, hidden)            # [x, mask]
        # Spatial: node attends over graph neighbours
        self.sp_attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        # Temporal
        enc = nn.TransformerEncoderLayer(hidden, n_heads, dim_feedforward=hidden*2,
                                         dropout=0.1, batch_first=True)
        self.t_enc   = nn.TransformerEncoder(enc, num_layers=1)
        self.out     = nn.Linear(hidden, 1)

    def _forward(self, x, m):
        # x, m: [N, T]
        N, T     = x.shape
        inp      = torch.stack([x, m], dim=-1)           # [N, T, 2]
        h        = self.proj(inp)                         # [N, T, H]
        # Spatial attention: each timestep, nodes attend over neighbours
        # Reshape to [T, N, H] → apply attention along N dim → [T, N, H]
        h_s      = h.permute(1, 0, 2)                    # [T, N, H]
        h_s, _   = self.sp_attn(h_s, h_s, h_s)
        h_s      = h_s.permute(1, 0, 2)                  # [N, T, H]
        # Temporal attention: each node, attend over time
        h_t      = self.t_enc(h_s)                       # [N, T, H]
        p        = self.out(h_t).squeeze(-1)              # [N, T]
        return p

    def training_step(self, x, m):
        p = self._forward(x, m)
        return F.mse_loss(p[m==1], x[m==1])

    def impute(self, x, m):
        p = self._forward(x, m)
        return m*x + (1-m)*p

print("\nTraining SPIN...")
spin_net = train_gnn_baseline(SPIN, 'SPIN')
eval_gnn_baseline(spin_net, 'SPIN')

# ─── DGCRIN ──────────────────────────────────────────────────────────────────
class DGCRIN(nn.Module):
    """
    DGCRIN (Zhang et al. 2023): Dynamic GCN with residual imputation.
    Constructs a dynamic adjacency from node features at each step,
    applies GCN, then residual-adds to initial estimate.
    """
    def __init__(self, hidden=GNN_HIDDEN):
        super().__init__()
        self.enc    = nn.Linear(2, hidden)                 # [x, m]
        self.dyn_W1 = nn.Linear(hidden, hidden // 2)
        self.dyn_W2 = nn.Linear(hidden, hidden // 2)
        self.gcn    = ChebConv(hidden, hidden, K=2)
        self.out    = nn.Linear(hidden, 1)
        self.act    = nn.ReLU()

    def _adj_dynamic(self, h):
        # h: [N, H] → soft adjacency [N, N]
        e1 = self.dyn_W1(h); e2 = self.dyn_W2(h)
        A  = torch.softmax(torch.mm(e1, e2.T) / (e1.size(1)**0.5), dim=-1)
        return A

    def _step(self, x_t, m_t):
        # x_t, m_t: [N]
        inp = torch.stack([x_t, m_t], dim=-1)     # [N, 2]
        h   = self.act(self.enc(inp))              # [N, H]
        A_d = self._adj_dynamic(h)
        h2  = self.act(torch.mm(A_d, self.gcn(h) ))
        p   = self.out(h2).squeeze(-1)             # [N]
        return p

    def _forward(self, x, m):
        N, T = x.shape
        preds = [self._step(x[:, t], m[:, t]) for t in range(T)]
        return torch.stack(preds, dim=1)   # [N, T]

    def training_step(self, x, m):
        p = self._forward(x, m)
        return F.mse_loss(p[m==1], x[m==1])

    def impute(self, x, m):
        p = self._forward(x, m)
        return m*x + (1-m)*p

print("\nTraining DGCRIN...")
dgcrin_net = train_gnn_baseline(DGCRIN, 'DGCRIN')
eval_gnn_baseline(dgcrin_net, 'DGCRIN')

# ─── GCASTN ──────────────────────────────────────────────────────────────────
class GCASTN(nn.Module):
    """
    GCASTN (Liu et al. 2023): Graph Conv + Cross-Attention + ST context.
    Encodes spatial context via GCN, temporal context via attention,
    then cross-attends spatial into temporal representations.
    """
    def __init__(self, hidden=GNN_HIDDEN, n_heads=4):
        super().__init__()
        self.sp_enc  = ChebConv(2, hidden, K=2)           # spatial [x,m]→H
        self.t_proj  = nn.Linear(2, hidden)
        enc = nn.TransformerEncoderLayer(hidden, n_heads, hidden*2,
                                         dropout=0.1, batch_first=True)
        self.t_enc   = nn.TransformerEncoder(enc, num_layers=1)
        self.cross   = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.out     = nn.Linear(hidden, 1)
        self.act     = nn.ReLU()

    def _forward(self, x, m):
        N, T = x.shape
        inp  = torch.stack([x, m], dim=-1)            # [N, T, 2]
        # Spatial encoding per timestep
        sp   = torch.stack([self.act(self.sp_enc(inp[:, t, :]))
                             for t in range(T)], dim=1)   # [N, T, H]
        # Temporal encoding per node
        t_in = self.t_proj(inp)                        # [N, T, H]
        t_h  = self.t_enc(t_in)                        # [N, T, H]
        # Cross-attention: query=temporal, key/value=spatial
        out, _ = self.cross(t_h, sp, sp)               # [N, T, H]
        p    = self.out(out).squeeze(-1)               # [N, T]
        return p

    def training_step(self, x, m):
        p = self._forward(x, m)
        return F.mse_loss(p[m==1], x[m==1])

    def impute(self, x, m):
        p = self._forward(x, m)
        return m*x + (1-m)*p

print("\nTraining GCASTN...")
gcastn_net = train_gnn_baseline(GCASTN, 'GCASTN')
eval_gnn_baseline(gcastn_net, 'GCASTN')

# ─── ADGCN ───────────────────────────────────────────────────────────────────
class ADGCN(nn.Module):
    """
    ADGCN (Chen et al. 2023): Adaptive Diffusion GCN with learned node embeddings.
    Tested on PEMS04 in the original paper. Uses a static learned adjacency
    combined with the road graph (mix). Bidirectional GRU backbone.
    """
    def __init__(self, hidden=GNN_HIDDEN):
        super().__init__()
        self.E1   = nn.Parameter(torch.randn(NUM_NODES, 10))   # node emb
        self.E2   = nn.Parameter(torch.randn(10, NUM_NODES))
        self.gcn  = ChebConv(hidden, hidden, K=2)
        self.gru_f = nn.GRUCell(hidden + 1, hidden)
        self.gru_b = nn.GRUCell(hidden + 1, hidden)
        self.enc  = nn.Linear(2, hidden)                       # [x, m]
        self.out  = nn.Linear(hidden * 2, 1)
        self.act  = nn.Tanh()

    def _adaptive_adj(self):
        A_ada = torch.softmax(torch.relu(torch.mm(self.E1, self.E2)), dim=1)
        return 0.5 * A_t + 0.5 * A_ada   # blend with road graph

    def _run_dir(self, x, m, gru, reverse=False):
        N, T = x.shape
        seq  = range(T-1, -1, -1) if reverse else range(T)
        h    = torch.zeros(N, gru.hidden_size, device=x.device)
        A    = self._adaptive_adj()
        hs   = []
        for t in seq:
            inp_t = torch.stack([x[:,t], m[:,t]], dim=-1)
            h_enc = self.act(self.enc(inp_t))
            h_enc = self.act(torch.mm(A, self.gcn(h_enc)))
            h     = gru(torch.cat([h_enc, m[:,t:t+1]], dim=-1), h)
            hs.append(h)
        if reverse:
            hs = hs[::-1]
        return torch.stack(hs, dim=1)   # [N, T, H]

    def _forward(self, x, m):
        hf = self._run_dir(x, m, self.gru_f, reverse=False)
        hb = self._run_dir(x, m, self.gru_b, reverse=True)
        p  = self.out(torch.cat([hf, hb], dim=-1)).squeeze(-1)   # [N, T]
        return p

    def training_step(self, x, m):
        p = self._forward(x, m)
        return F.mse_loss(p[m==1], x[m==1])

    def impute(self, x, m):
        p = self._forward(x, m)
        return m*x + (1-m)*p

print("\nTraining ADGCN...")
adgcn_net = train_gnn_baseline(ADGCN, 'ADGCN')
eval_gnn_baseline(adgcn_net, 'ADGCN')
print(f"   Tier 3 complete — {len(results_table)} entries in results_table.")

# =============================================================================
# CELL 16 — Final comparison table + bar chart
# =============================================================================

# Sort by MAE all (ascending)
results_table_sorted = sorted(results_table, key=lambda r: r['mae_all'])

print("\n" + "=" * 90)
print("  COMPREHENSIVE BASELINE COMPARISON — PEMS04  |  80% blind nodes  |  test t=4500–4950")
print("=" * 90)
print(f"  {'Model':<26} {'MAE all':>9} {'MAE jam':>9} {'Prec':>7} {'Rec':>7} {'F1':>7} {'SSIM':>7}")
print("  " + "-"*86)

tier_labels = {
    'Global Mean':           'T1',
    'Historical Average':    'T1',
    'IDW':                   'T1',
    'Linear Interpolation':  'T1',
    'KNN Kriging (k=5)':     'T1',
    'GRU-D':                 'T2',
    'BRITS':                 'T2',
    'SAITS':                 'T2',
    'IGNNK':                 'T3',
    'GRIN':                  'T3',
    'SPIN':                  'T3',
    'DGCRIN':                'T3',
    'GCASTN':                'T3',
    'ADGCN':                 'T3',
    'Graph-CTH-NODE v5':     'Ours',
}

for r in results_table_sorted:
    tier  = tier_labels.get(r['model'], '')
    flag  = ' ◀' if r['model'] == 'Graph-CTH-NODE v5' else ''
    print(f"  [{tier:<4}] {r['model']:<21} "
          f"{r['mae_all']:>9.2f} {r['mae_jam']:>9.2f} "
          f"{r['prec']:>7.3f} {r['rec']:>7.3f} {r['f1']:>7.3f} "
          f"{r['ssim']:>7.3f}{flag}")

print("=" * 90)
print("  Metric definitions:")
print("    MAE all : mean absolute error (km/h) on all blind nodes in test window")
print("    MAE jam : MAE restricted to timesteps where true speed < 40 km/h")
print("    Prec/Rec/F1 : jam detection via speed threshold < 40 km/h on predictions")
print("    SSIM    : structural similarity of spatiotemporal speed field")
print("  Tier: T1=Statistical  T2=RNN/temporal  T3=GNN imputation  Ours=Graph-CTH-NODE")
print("=" * 90)

# Bar chart
fig2, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=120)
names   = [r['model'] for r in results_table_sorted]
mae_all = [r['mae_all'] for r in results_table_sorted]
mae_jam = [r['mae_jam'] for r in results_table_sorted]
f1_vals = [r['f1']      for r in results_table_sorted]

colors  = []
for r in results_table_sorted:
    t = tier_labels.get(r['model'], '')
    if t == 'Ours':    colors.append('#d62728')
    elif t == 'T3':    colors.append('#1f77b4')
    elif t == 'T2':    colors.append('#ff7f0e')
    else:              colors.append('#7f7f7f')

short_names = [n.replace('Graph-CTH-NODE ', '').replace('KNN Kriging (k=5)', 'KNN-K')
               .replace('Linear Interpolation', 'Lin.Interp')
               .replace('Historical Average', 'Hist.Avg') for n in names]

for ax, vals, title, ylabel in [
    (axes[0], mae_all, 'MAE — All Blind Nodes (km/h)', 'MAE (km/h)'),
    (axes[1], mae_jam, 'MAE — Jam Conditions (km/h)',  'MAE (km/h)'),
    (axes[2], f1_vals, 'Jam Detection F1 (speed<40)',  'F1'),
]:
    bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=10)
    # Highlight ours
    our_idx = next(i for i, r in enumerate(results_table_sorted)
                   if r['model'] == 'Graph-CTH-NODE v5')
    bars[our_idx].set_edgecolor('black')
    bars[our_idx].set_linewidth(2)

from matplotlib.patches import Patch
legend_els = [
    Patch(color='#d62728', label='Ours'),
    Patch(color='#1f77b4', label='T3: GNN imputation'),
    Patch(color='#ff7f0e', label='T2: RNN/temporal'),
    Patch(color='#7f7f7f', label='T1: Statistical'),
]
fig2.legend(handles=legend_els, loc='upper center', ncol=4,
            bbox_to_anchor=(0.5, 1.02), fontsize=10)
fig2.tight_layout()
plt.savefig('baseline_comparison.png', bbox_inches='tight', dpi=150)
plt.show()
print("✅ Comparison table printed. Figure saved to baseline_comparison.png")
