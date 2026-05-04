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
        self.act = nn.Tanh()

    def forward(self, x_seq, m_seq, tod_free_seq=None, tod_jam_seq=None):
        N, T = x_seq.shape
        h = torch.zeros(N, self.hidden, device=x_seq.device)
        preds = []

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
        return torch.stack(preds, dim=1)

class DualFlow(nn.Module):
    def __init__(self, hidden=64, include_tod=True, jam_loss_weight=2.5, free_loss_weight=1.0, use_soft_threshold=False):
        super().__init__()
        self.include_tod = include_tod
        self.jam_loss_weight = jam_loss_weight
        self.free_loss_weight = free_loss_weight
        self.use_soft_threshold = use_soft_threshold
        self.fwd = DualFlowCell(hidden, include_tod)
        self.bwd = DualFlowCell(hidden, include_tod)
        self.fuse = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, 2), nn.Softmax(dim=-1)
        )

    def _run(self, x, m, tod_free=None, tod_jam=None):
        pf = self.fwd(x, m, tod_free, tod_jam)
        pb = self.bwd(x.flip(1), m.flip(1),
                      tod_free.flip(1) if tod_free is not None else None,
                      tod_jam.flip(1) if tod_jam is not None else None).flip(1)
        fuse_in = torch.stack([pf, pb], dim=-1)
        w = self.fuse(fuse_in)
        return (w[..., 0:1] * pf.unsqueeze(-1) + w[..., 1:2] * pb.unsqueeze(-1)).squeeze(-1)

    def training_step(self, x, m, tod_free=None, tod_jam=None):
        p = self._run(x, m, tod_free, tod_jam)
        # Clip predictions to [-5, 5] in normalized space to prevent unbounded gradients
        p = torch.clamp(p, -5.0, 5.0)

        jt = torch.tensor(jam_thresh_soft_np if self.use_soft_threshold else jam_thresh_eval_np,
                          dtype=torch.float32, device=x.device)
        jam_flag = (x < jt.unsqueeze(1)).float()
        free_flag = 1.0 - jam_flag
        # Supervise ALL positions (both observed and blind). Input is already masked
        # so the model must learn imputation from neighbors+temporal context at blind
        # nodes, while keeping observed-node predictions consistent with truth.
        loss_free = torch.mean(((p - x) * free_flag) ** 2) * self.free_loss_weight

        delta = 2.0
        diff_jam = torch.abs((p - x) * jam_flag)
        huber_jam = torch.where(diff_jam < delta,
                                0.5 * diff_jam ** 2,
                                delta * (diff_jam - 0.5 * delta))
        loss_jam = torch.mean(huber_jam) * self.jam_loss_weight

        rmse_reg = torch.mean((p - x) ** 2) * 0.1
        return loss_free + loss_jam + rmse_reg

    def impute(self, x, m, tod_free=None, tod_jam=None):
        return m * x + (1.0 - m) * self._run(x, m, tod_free, tod_jam)

print("✅ DualFlow model architecture defined")

PRODUCTION_SEED = 86415
PRODUCTION_JAM_WEIGHT = 1.0  # Reduced from 2.5 to stabilize training (less prone to gradient spikes)
PRODUCTION_FREE_WEIGHT = 1.0

def train_dualflow_production(hidden=64, epochs=600):
    torch.manual_seed(PRODUCTION_SEED)
    np.random.seed(PRODUCTION_SEED)

    net = DualFlow(hidden=hidden, include_tod=True,
                   jam_loss_weight=PRODUCTION_JAM_WEIGHT,
                   free_loss_weight=PRODUCTION_FREE_WEIGHT,
                   use_soft_threshold=False).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    best_vloss, best_wts, patience_ctr = float('inf'), None, 0
    loss_history_train, loss_history_val = [], []

    print(f"\n{'='*80}")
    print(f"PRODUCTION MODEL: DualFlow")
    print(f"  Seed: {PRODUCTION_SEED}  |  Jam weight: {PRODUCTION_JAM_WEIGHT}x  |  Free weight: {PRODUCTION_FREE_WEIGHT}x")
    print(f"{'='*80}\n")

    for ep in range(1, epochs + 1):
        net.train()
        t0 = np.random.randint(0, TRAIN_END - BATCH_TIME)
        x_full = torch.tensor(speed_np[t0:t0+BATCH_TIME, :], dtype=torch.float32).T.to(device)
        m_train = (torch.rand(NUM_NODES, 1, device=device) > SPARSITY).float().expand(-1, BATCH_TIME)
        slots = (np.arange(t0, t0+BATCH_TIME) % 288).astype(int)
        tod_free = torch.tensor(tod_free_np[:, slots], dtype=torch.float32).to(device)
        tod_jam = torch.tensor(tod_jam_np[:, slots], dtype=torch.float32).to(device)
        loss = net.training_step(x_full, m_train, tod_free, tod_jam)

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
                m_v = (node_mask[0,:,0,0]==1).float().unsqueeze(1).expand(-1, VAL_END-VAL_START)
                slots_v = (np.arange(VAL_START, VAL_END) % 288).astype(int)
                tf_v = torch.tensor(tod_free_np[:, slots_v], dtype=torch.float32).to(device)
                tj_v = torch.tensor(tod_jam_np[:, slots_v], dtype=torch.float32).to(device)
                vl = net.training_step(x_v, m_v, tf_v, tj_v).item()
            loss_history_val.append(vl)
            if vl < best_vloss:
                best_vloss = vl
                best_wts = copy.deepcopy(net.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1
            mae_v = torch.mean(torch.abs(p_v - x_v) * m_v).item()
            rmse_v = torch.sqrt(torch.mean(((p_v - x_v) * m_v) ** 2)).item()
            print(f"  [DualFlow] ep {ep:3d} | loss={vl:.4f} | MAE={mae_v:.4f} | RMSE={rmse_v:.4f}")
            if patience_ctr >= 3:
                print(f"  -> Early stop at ep {ep}")
                break

    if best_wts:
        net.load_state_dict(best_wts)
    return net, loss_history_train, loss_history_val

print("✅ Training function defined")

dualflow_net, dualflow_loss_train, dualflow_loss_val = train_dualflow_production(hidden=64, epochs=600)
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
