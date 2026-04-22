"""
Build a comprehensive, professional code documentation PDF.
Covers every significant line of cth_node_complete.py with detailed explanations.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor, black, white, grey
from reportlab.lib.units import cm, mm, inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    Preformatted, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

# ─ Setup ─────────────────────────────────────────────────────────────────────
PW, PH = A4
ML, MR, MT, MB = 25*mm, 25*mm, 25*mm, 25*mm
BODY_W = PW - ML - MR

def create_styles():
    """Create professional document styles"""
    styles = {}
    styles['title'] = ParagraphStyle(
        'title', fontName='Times-Bold', fontSize=20, textColor=black,
        leading=24, spaceAfter=12, alignment=TA_CENTER)
    styles['chapter'] = ParagraphStyle(
        'chapter', fontName='Times-Bold', fontSize=16, textColor=black,
        leading=19, spaceBefore=12, spaceAfter=8)
    styles['section'] = ParagraphStyle(
        'section', fontName='Times-Bold', fontSize=13, textColor=HexColor("#1B2A4A"),
        leading=16, spaceBefore=10, spaceAfter=6)
    styles['subsection'] = ParagraphStyle(
        'subsection', fontName='Times-BoldItalic', fontSize=11, textColor=black,
        leading=14, spaceBefore=8, spaceAfter=4)
    styles['body'] = ParagraphStyle(
        'body', fontName='Times-Roman', fontSize=10, textColor=black,
        leading=13, spaceAfter=5, alignment=TA_JUSTIFY)
    styles['code'] = ParagraphStyle(
        'code', fontName='Courier', fontSize=8.5, textColor=black,
        leading=10, leftIndent=12, spaceBefore=3, spaceAfter=3,
        backColor=HexColor("#F5F5F5"))
    styles['code_block'] = ParagraphStyle(
        'code_block', fontName='Courier', fontSize=8, textColor=HexColor("#333333"),
        leading=9.5, leftIndent=16, rightIndent=8,
        spaceBefore=4, spaceAfter=4, backColor=HexColor("#FAFAFA"))
    styles['bullet'] = ParagraphStyle(
        'bullet', fontName='Times-Roman', fontSize=10, textColor=black,
        leading=13, spaceAfter=4, leftIndent=24, firstLineIndent=0)
    styles['note'] = ParagraphStyle(
        'note', fontName='Times-Italic', fontSize=9, textColor=HexColor("#555555"),
        leading=12, spaceAfter=6, leftIndent=12, rightIndent=12)
    return styles

# ─ Build document ────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    "code_documentation.pdf",
    pagesize=A4,
    leftMargin=ML, rightMargin=MR, topMargin=MT, bottomMargin=MB
)
styles = create_styles()
story = []

# ─ Title page ────────────────────────────────────────────────────────────────
story.append(Spacer(1, 20*mm))
story.append(Paragraph("Code Documentation", styles['title']))
story.append(Paragraph("Graph-CTH-NODE v7 FreqDGT", styles['title']))
story.append(Spacer(1, 10*mm))
story.append(Paragraph(
    "Frequency-Decomposed Dynamic Graph Transformer for Sparse Traffic Speed Imputation",
    ParagraphStyle('subtitle', fontName='Times-Italic', fontSize=11, textColor=HexColor("#333333"),
                   alignment=TA_CENTER, leading=14, spaceAfter=20)))
story.append(Paragraph(
    "A comprehensive line-by-line walkthrough of the complete Python implementation",
    ParagraphStyle('desc', fontName='Times-Roman', fontSize=10, textColor=HexColor("#555555"),
                   alignment=TA_CENTER, leading=13)))
story.append(Spacer(1, 30*mm))

# ─ Table of contents ─────────────────────────────────────────────────────────
story.append(Paragraph("Contents", styles['chapter']))
toc_items = [
    "1. File Structure & Imports",
    "2. Global Configuration & Dataset Setup",
    "3. Data Loading & Normalization (CELL 1)",
    "4. Graph Construction (CELLS 2-3)",
    "5. Input Feature Engineering (CELL 4)",
    "6. Baseline Architectures (v6, v6-Jam, v6-Next)",
    "7. Metrics & Evaluation Harness (CELLS 6-8)",
    "8. v7 FreqDGT Architecture (Thesis Contribution)",
    "9. Training Loop (train_freqdgt)",
    "10. Baseline Models (CELLS 9-11)",
    "11. Results Table & Visualization (CELL 12)",
    "12. Analysis & Conclusions (CELL 13)",
]
for item in toc_items:
    story.append(Paragraph(item, styles['bullet']))

story.append(PageBreak())

# ─ Main content ──────────────────────────────────────────────────────────────

# SECTION 1: File Structure & Imports
story.append(Paragraph("1. File Structure & Imports", styles['chapter']))
story.append(Paragraph(
    "The codebase is organized as a single comprehensive Jupyter notebook converted to Python. "
    "It contains 13 cells, each addressing a specific aspect of the traffic imputation pipeline.",
    styles['body']))
story.append(Spacer(1, 3*mm))

story.append(Paragraph("1.1 Header Comment (Lines 1-29)", styles['subsection']))
story.append(Paragraph(
    "The file begins with a detailed header documenting the v7 FreqDGT architecture:",
    styles['body']))
story.append(Preformatted(
    "# Graph-CTH-NODE v7 \"FreqDGT\" — Frequency-Decomposed Dynamic Graph Transformer\n"
    "# THESIS CONTRIBUTION: Novel combination of wavelet-style decomposition +\n"
    "#                      dynamic graphs + expert gating",
    styles['code_block']))
story.append(Paragraph(
    "<b>Purpose:</b> Documents the key innovations: frequency decomposition (separating speed into "
    "trends and spikes), dynamic graph construction (per-timestep attention-based adjacency), "
    "and expert gating (MLP routing between specialized branches).",
    styles['note']))
story.append(Spacer(1, 4*mm))

story.append(Paragraph("1.2 Imports (Lines 31-42)", styles['subsection']))
story.append(Preformatted(
    "import torch                        # Deep learning framework\n"
    "import torch.nn as nn              # Neural network modules\n"
    "import torch.nn.functional as F    # Activation functions (relu, sigmoid, etc.)\n"
    "import numpy as np                 # Numerical computing\n"
    "import os, copy, urllib.request    # File I/O, object copying, URL downloads\n"
    "import pandas as pd                # Data frame operations (reading CSV)\n"
    "import matplotlib.pyplot as plt    # Visualization\n"
    "import warnings                    # Suppress RuntimeWarnings",
    styles['code_block']))
story.append(Paragraph(
    "<b>Notes:</b> <br/>"
    "• torch: GPU-accelerated tensor operations; automatic differentiation (autograd)<br/>"
    "• torch.nn: Base classes (Module, Linear, GRUCell, etc.)<br/>"
    "• torch.nn.functional: Stateless functions (ReLU, sigmoid, conv1d)<br/>"
    "• numpy: CPU-based preprocessing (before converting to tensors)<br/>"
    "• pandas: Read road distance matrix from CSV file<br/>"
    "• matplotlib: Generate comparison charts and loss curves",
    styles['note']))
story.append(Spacer(1, 4*mm))

story.append(Paragraph("1.3 Global Seed & Device (Lines 45-54)", styles['subsection']))
story.append(Preformatted(
    "GLOBAL_SEED = 42\n"
    "torch.manual_seed(GLOBAL_SEED)           # CPU randomness\n"
    "np.random.seed(GLOBAL_SEED)              # NumPy randomness\n"
    "torch.cuda.manual_seed_all(GLOBAL_SEED)  # GPU randomness (all devices)\n"
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
    styles['code_block']))
story.append(Paragraph(
    "<b>Purpose:</b> Ensures reproducibility across runs. All random operations "
    "(tensor initialization, mask generation, etc.) use the same seed. "
    "<b>Device:</b> Automatically selects GPU if CUDA is available; falls back to CPU otherwise.",
    styles['note']))
story.append(Spacer(1, 5*mm))

# SECTION 2: Dataset Configuration
story.append(Paragraph("2. Global Configuration & Dataset Setup", styles['chapter']))
story.append(Paragraph("2.1 Dataset Selector (Lines 60-100)", styles['subsection']))
story.append(Paragraph(
    "The DATASETS dictionary defines URLs and metadata for four traffic benchmarks:",
    styles['body']))
story.append(Preformatted(
    "DATASETS = {\n"
    "    'PEMS04': {\n"
    "        'npz_url': 'https://zenodo.org/...',\n"
    "        'num_nodes': 307,        # sensors in SF Bay Area\n"
    "        'time_steps': 5000,      # ~17 days of 5-min data\n"
    "        'channel_idx': 2,        # speed is channel 2 (vs flow, occupancy)\n"
    "    },\n"
    "    'PEMS08': {...},  # 170 sensors\n"
    "    'METR-LA': {...}, # 207 sensors, longer timeseries\n"
    "    'PEMS-BAY': {...} # 325 sensors\n"
    "}",
    styles['code_block']))
story.append(Paragraph(
    "The code selects PEMS04 by default (line 92: <i>DATASET_NAME = 'PEMS04'</i>). "
    "This can be changed to experiment on other datasets.",
    styles['body']))
story.append(Spacer(1, 5*mm))

# SECTION 3: Data Loading
story.append(Paragraph("3. Data Loading & Normalization (CELL 1)", styles['chapter']))
story.append(Paragraph("3.1 File Download & Loading (Lines 106-122)", styles['subsection']))
story.append(Preformatted(
    "for fn, url in [(fn_npz, url_npz), (fn_csv, url_csv)]:\n"
    "    if not os.path.exists(fn):\n"
    "        urllib.request.urlretrieve(url, fn)\n"
    "raw_all = raw_npz['data'][:TIME_STEPS, :NUM_NODES, :]\n"
    "raw_speed = raw_all[:, :, CHAN_IDX]  # [T, N] extract speed channel",
    styles['code_block']))
story.append(Paragraph(
    "<b>Logic:</b> Download NPZ (data) and CSV (road network distances) if not cached. "
    "Extract speed channel (index 2) from the raw data shape [T=5000, N=307, C=3].",
    styles['note']))
story.append(Spacer(1, 4*mm))

story.append(Paragraph("3.2 Per-Node Normalization (Lines 139-142)", styles['subsection']))
story.append(Preformatted(
    "node_means = raw_speed[:TRAIN_END].mean(axis=0)  # [N] mean per sensor\n"
    "node_stds  = raw_speed[:TRAIN_END].std(axis=0)   # [N] std per sensor\n"
    "data_norm_speed = (raw_speed - node_means) / node_stds  # [T, N] z-normalized",
    styles['code_block']))
story.append(Paragraph(
    "<b>Design:</b> Each sensor is normalized by its own mean and std, computed on training data only. "
    "This ensures sensors on fast freeways (mean=70 km/h) and slow city streets (mean=30 km/h) "
    "are treated equitably by the model. The denominator adds 1e-8 to avoid division by zero.",
    styles['note']))
story.append(Spacer(1, 4*mm))

story.append(Paragraph("3.3 Jam Threshold per Node (Lines 154-169)", styles['subsection']))
story.append(Preformatted(
    "JAM_KMH_EVAL = 40.0  # Global threshold for evaluation metrics\n"
    "jam_thresh_eval_np = (JAM_KMH_EVAL - node_means) / node_stds  # [N] normalized\n"
    "# Per-node heterogeneous threshold:\n"
    "node_jam_thresh_kmh = node_means - 0.5 * node_stds  # [N] in km/h\n"
    "node_jam_thresh_norm = (node_jam_thresh_kmh - node_means) / node_stds",
    styles['code_block']))
story.append(Paragraph(
    "<b>Rationale:</b> Evaluation uses a global 40 km/h threshold for metrics. "
    "Training uses per-node thresholds (mean - 0.5×std) to capture heterogeneous jam conditions: "
    "freeways jam at ~70 km/h, city streets jam at ~30 km/h. This avoids noisy percentile-based outlier detection.",
    styles['note']))
story.append(Spacer(1, 4*mm))

story.append(Paragraph("3.4 Time-of-Day (ToD) Priors (Lines 171-201)", styles['subsection']))
story.append(Preformatted(
    "STEPS_PER_DAY = 288  # 24h ÷ 5min = 288 timesteps\n"
    "slot_idx = np.arange(TIME_STEPS) % STEPS_PER_DAY  # [T] which daily slot?\n"
    "# Compute mean speed for each time slot:\n"
    "tod_mean_sp = np.zeros((NUM_NODES, STEPS_PER_DAY))\n"
    "for s in range(STEPS_PER_DAY):\n"
    "    vals = data_norm_speed[slot_idx == s, :]\n"
    "    tod_mean_sp[:, s] = vals.mean(axis=0)  # [N] average speed at slot s\n"
    "tod_prior = tod_mean_sp[:, slot_idx].T  # [T, N] reindex to timesteps",
    styles['code_block']))
story.append(Paragraph(
    "<b>Purpose:</b> Create contextual priors: \"at 7am, freeway speed is typically 45 km/h; "
    "at 2pm, it's 70 km/h\". Also compute conditional priors (free-flow vs jam speeds at each time).",
    styles['note']))
story.append(Spacer(1, 5*mm))

# SECTION 4: Graph Construction
story.append(Paragraph("4. Graph Construction (CELLS 2-3)", styles['chapter']))
story.append(Paragraph("4.1 Distance Matrix & Gaussian Kernel (Lines 222-244)", styles['subsection']))
story.append(Preformatted(
    "# Read distances from CSV\n"
    "dist_mat = np.full((NUM_NODES, NUM_NODES), np.inf)\n"
    "for _, row in df.iterrows():\n"
    "    i, j, d = int(row[0]), int(row[1]), float(row[2])\n"
    "    dist_mat[i, j] = d\n"
    "    dist_mat[j, i] = d  # symmetric\n"
    "# Gaussian kernel: A[i,j] = exp(-(d_ij^2) / sigma^2)\n"
    "sigma = dist_mat[dist_mat < np.inf].std()\n"
    "adj_sym = np.exp(-(dist_mat**2) / (sigma**2))",
    styles['code_block']))
story.append(Paragraph(
    "<b>Design:</b> Build symmetric adjacency from road distances. "
    "Gaussian kernel maps physical distance to edge weight: nearby sensors → weight ~1, "
    "distant sensors → weight ~0. Sigma (bandwidth) is the std of all pairwise distances.",
    styles['note']))
story.append(Spacer(1, 4*mm))

story.append(Paragraph("4.2 Directed Adjacencies (Lines 245-252)", styles['subsection']))
story.append(Preformatted(
    "# Forward/backward capture traffic flow direction\n"
    "adj_fwd = gaussian_norm(dist_fwd, directed=True)   # upstream → downstream\n"
    "adj_bwd = gaussian_norm(dist_bwd, directed=True)   # downstream → upstream\n"
    "# Correlation adjacency: high co-varying speeds\n"
    "corr_mat = np.corrcoef(speed_train)  # [N, N] correlation matrix\n"
    "adj_corr = np.where(corr_mat > 0.60, corr_mat, 0.)  # threshold at 0.60",
    styles['code_block']))
story.append(Paragraph(
    "<b>Insight:</b> Four adjacency matrices capture different relationships: "
    "symmetric (geometry), forward/backward (flow direction), correlation (hidden hotspots). "
    "The model learns to mix them adaptively per node per timestep.",
    styles['note']))
story.append(Spacer(1, 4*mm))

story.append(Paragraph("4.3 Hypergraph Construction (Lines 268-277)", styles['subsection']))
story.append(Preformatted(
    "adj_bin = (adj_sym > 1e-6).astype(float)  # thresholded to 0/1\n"
    "adj2 = (adj_bin @ adj_bin > 0).astype(float) + adj_bin  # 1-hop + 2-hop union\n"
    "H = adj2.T  # [N, N] incidence matrix: H[i,j] = 1 if i in j's hyperedge\n"
    "# Normalize: D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2}\n"
    "H_conv = (d_v_inv_sq[:, None] * H) * d_e_inv[None, :] @ (H.T * d_v_inv_sq[None, :])",
    styles['code_block']))
story.append(Paragraph(
    "<b>Purpose:</b> Group sensors into hyperedges (multi-sensor corridor clusters). "
    "Each sensor's hyperedge = {itself} ∪ {1-hop neighbors} ∪ {2-hop neighbors}. "
    "Pre-compute the normalised operator offline for O(1) runtime cost.",
    styles['note']))
story.append(Spacer(1, 5*mm))

# SECTION 5: Features
story.append(Paragraph("5. Input Feature Engineering (CELL 4)", styles['chapter']))
story.append(Paragraph("5.1 13-Dimensional Feature Vector (Lines 321-326)", styles['subsection']))
story.append(Paragraph(
    "For each node at each timestep, the model receives 13 features:",
    styles['body']))
features = [
    "1. <b>speed_w_prior</b>: Observed speed + ToD prior for blind nodes",
    "2. <b>global_ctx_speed</b>: Mean speed across all observed nodes",
    "3. <b>nbr_prop_speed</b>: Propagated neighbor mean + global blending",
    "4. <b>cov_feat</b>: % of neighbors observed (coverage)",
    "5. <b>is_observed</b>: Binary flag (1=observed, 0=blind)",
    "6-7. <b>time_sin, time_cos</b>: Daily periodicity (scaled by 0.25)",
    "8. <b>obs_flow</b>: Observed vehicle flow (channel 0)",
    "9. <b>global_ctx_flow</b>: Mean flow across all nodes",
    "10. <b>obs_occupancy</b>: Observed road occupancy (channel 1)",
    "11. <b>global_ctx_occ</b>: Mean occupancy",
    "12-13. <b>tod_free, tod_jam</b>: Conditional ToD priors (free-flow and jam speeds at this time slot)"
]
for feat in features:
    story.append(Paragraph(feat, styles['bullet']))
story.append(Spacer(1, 4*mm))

story.append(Paragraph("5.2 No Ground Truth Leakage (Lines 345-346)", styles['subsection']))
story.append(Preformatted(
    "assert (input_features[0, node_mask[0,:,0,0]==0, :, 0] == 0).all()\n"
    "# Verifies: blind node speeds (feature 0) are exactly zero",
    styles['code_block']))
story.append(Paragraph(
    "<b>Safety Check:</b> Crashes immediately if blind node ground truth appears in features. "
    "This is the most critical invariant—leakage would make the model useless.",
    styles['note']))
story.append(Spacer(1, 5*mm))

# SECTION 6: Baseline Architectures (Summary)
story.append(Paragraph("6. Baseline Architectures (v6, v6-Jam, v6-Next)", styles['chapter']))
story.append(Paragraph(
    "The codebase includes three baseline models to validate improvements: <br/>"
    "• <b>v6:</b> Bidirectional RNN + 4-path graphs + ToD priors (MAE 0.81) <br/>"
    "• <b>v6-Jam:</b> v6 with Mamba-inspired selective updates (MAE 0.89) <br/>"
    "• <b>v6-Next:</b> v6 with dynamic graph constructor (MAE 1.51)",
    styles['body']))
story.append(Spacer(1, 3*mm))
story.append(Paragraph(
    "These are <b>not trained</b> in the notebook (they're cached); their code is included for reference. "
    "The notebook trains <b>only v7 FreqDGT</b>.",
    styles['note']))
story.append(Spacer(1, 5*mm))

# SECTION 7: Metrics & Evaluation
story.append(Paragraph("7. Metrics & Evaluation Harness (CELLS 6-8)", styles['chapter']))
story.append(Paragraph("7.1 Metrics Functions (Lines 626-648)", styles['subsection']))
story.append(Preformatted(
    "def eval_pred_np(pred_kmh_bl, true_kmh_bl):\n"
    "    mae_all = np.abs(pred_kmh_bl - true_kmh_bl).mean()\n"
    "    jam_mask = true_kmh_bl < 40  # jam = speed < 40 km/h\n"
    "    mae_jam = np.abs((pred - true)[jam_mask]).mean()\n"
    "    prec, rec, f1 = jam_prec_recall(pred_kmh_bl, true_kmh_bl)\n"
    "    ssim = compute_ssim(pred_kmh_bl, true_kmh_bl)\n"
    "    return {'mae_all': mae_all, 'mae_jam': mae_jam, 'prec': prec, ...}",
    styles['code_block']))
story.append(Paragraph(
    "<b>Metrics:</b> <br/>"
    "• <b>MAE all</b>: Mean absolute error on all blind nodes <br/>"
    "• <b>MAE jam</b>: MAE restricted to congested timesteps (speed < 40 km/h) <br/>"
    "• <b>Precision/Recall/F1</b>: Jam detection quality (binary: pred < 40 vs true < 40) <br/>"
    "• <b>SSIM</b>: Structural Similarity Index (spatial pattern preservation)",
    styles['note']))
story.append(Spacer(1, 5*mm))

# SECTION 8: v7 FreqDGT (THE MAIN CONTRIBUTION)
story.append(Paragraph("8. v7 FreqDGT Architecture (Thesis Contribution)", styles['chapter']))
story.append(Paragraph(
    "The core innovation: decompose speed into frequency bands and route via expert gating.",
    styles['body']))
story.append(Spacer(1, 3*mm))

story.append(Paragraph("8.1 Frequency Decomposer (Lines 1206-1221)", styles['subsection']))
story.append(Preformatted(
    "class FreqDecomposer(nn.Module):\n"
    "    def __init__(self, window=5):\n"
    "        self.kernel = nn.Parameter(torch.ones(1, 1, window) / window)\n"
    "    def forward(self, x):  # x: [N, T]\n"
    "        low = F.conv1d(x.unsqueeze(1), self.kernel).squeeze(1)  # trend\n"
    "        high = x - low  # residual (spikes)\n"
    "        return low, high",
    styles['code_block']))
story.append(Paragraph(
    "<b>Design:</b> Learnable 1D convolution acts as a moving-average filter. "
    "Separates input into smooth trend (low-freq) and spiky residual (high-freq). "
    "Unlike fixed wavelets, the kernel learns the optimal frequency cutoff from data.",
    styles['note']))
story.append(Spacer(1, 4*mm))

story.append(Paragraph("8.2 Low-Frequency Branch (Lines 1224-1289)", styles['subsection']))
story.append(Preformatted(
    "class LowFreqBranch(nn.Module):\n"
    "    def forward(self, x_seq, m_seq, tod_free, tod_jam):\n"
    "        # Two-direction RNN on 4-path graph convolution\n"
    "        h_fwd = self._forward_dir(gru_fwd, x_seq, m_seq, ...)  # [N, T, H]\n"
    "        h_bwd = self._forward_dir(gru_bwd, x_seq.flip(1), ...)  # reversed, flip back\n"
    "        h = cat([h_fwd, h_bwd], dim=-1)  # [N, T, 2H]\n"
    "        pred = self.out(h).squeeze(-1)  # [N, T]",
    styles['code_block']))
story.append(Paragraph(
    "<b>Logic:</b> Process low-freq signal (trend) with bidirectional GRU. "
    "Forward GRU reads left-to-right; backward GRU reads right-to-left. "
    "Concatenate and project to scalar predictions. Captures sustained congestion patterns.",
    styles['note']))
story.append(Spacer(1, 4*mm))

story.append(Paragraph("8.3 High-Frequency Branch (Lines 1292-1322)", styles['subsection']))
story.append(Preformatted(
    "class HighFreqBranch(nn.Module):\n"
    "    def forward(self, x_high, m_seq):  # x_high: [N, T] residual\n"
    "        h_seq = self.in_proj(cat([x_high, m_seq], dim=-1))  # [N, T, H]\n"
    "        for t in range(T):\n"
    "            A_dyn = self.dynamic_graph(h_seq[:, t, :])  # [N, N] attention\n"
    "            h_gcn = A_dyn @ ReLU(self.gcn_w(h_seq[:, t, :]))  # message pass\n"
    "            h_seq[:, t, :] = h_t + h_gcn  # residual\n"
    "        h_seq = self.transformer(h_seq)  # temporal attention\n"
    "        h_seq = self.norm(h_seq)  # LayerNorm stabilization\n"
    "        pred = self.out(h_seq).squeeze(-1)",
    styles['code_block']))
story.append(Paragraph(
    "<b>Design:</b> Process high-freq signal (spikes/jams) with dynamic graphs + transformer. "
    "Per-timestep adjacency A_t discovered via attention on hidden state. "
    "Transformer aggregates temporal patterns. LayerNorm prevents explosion.",
    styles['note']))
story.append(Spacer(1, 4*mm))

story.append(Paragraph("8.4 Expert Gate (Lines 1325-1341)", styles['subsection']))
story.append(Preformatted(
    "class ExpertGate(nn.Module):\n"
    "    def forward(self, x, m, tod_free, tod_jam):\n"
    "        feat = stack([x, m, tod_free, tod_jam], dim=-1)  # [N, T, 4]\n"
    "        gate = sigmoid(self.mlp(feat)).squeeze(-1)  # [N, T] in [0,1]\n"
    "        return gate",
    styles['code_block']))
story.append(Paragraph(
    "<b>Gating Logic:</b> MLP routes per-node per-timestep. "
    "Input: observed speed, mask, and ToD context (free-flow/jam priors). "
    "Output: gate ∈ [0,1] blends high-freq (gate=1) vs low-freq (gate=0) predictions.",
    styles['note']))
story.append(Spacer(1, 4*mm))

story.append(Paragraph("8.5 Full FreqDGT Model (Lines 1344-1376)", styles['subsection']))
story.append(Preformatted(
    "class FreqDGT(nn.Module):\n"
    "    def forward(self, x_seq, m_seq, tod_free, tod_jam):\n"
    "        low, high = self.decomposer(x_seq)  # [N, T] each\n"
    "        pred_low = self.low_branch(low, m_seq, ...)  # [N, T]\n"
    "        pred_high = self.high_branch(high, m_seq)  # [N, T]\n"
    "        gate = self.gate(x_seq, m_seq, tod_free, tod_jam)  # [N, T]\n"
    "        # Expert mixture: gate·high + (1-gate)·low\n"
    "        final = pred_low + gate * pred_high\n"
    "        final = clamp(final, -5, 5)  # numerical stability\n"
    "        return final",
    styles['code_block']))
story.append(Paragraph(
    "<b>Forward Pass:</b> Decompose → two branches → gate → mixture. "
    "The gate learns to emphasize high-freq branch during jam periods and low-freq during free-flow. "
    "Clamping prevents exploding predictions early in training.",
    styles['note']))
story.append(Spacer(1, 5*mm))

# SECTION 9: Training
story.append(Paragraph("9. Training Loop (train_freqdgt, Lines 1382-1479)", styles['chapter']))
story.append(Paragraph("9.1 Initialization (Lines 1382-1402)", styles['subsection']))
story.append(Preformatted(
    "def train_freqdgt(hidden=64, epochs=400):\n"
    "    net = FreqDGT(hidden=hidden, include_tod=True).to(device)\n"
    "    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)\n"
    "    scheduler = ReduceLROnPlateau(opt, patience=3, factor=0.5, min_lr=1e-5)",
    styles['code_block']))
story.append(Paragraph(
    "<b>Optimizer:</b> Adam with low LR (1e-3, down from 3e-3 in baseline) and scheduler. "
    "ReduceLROnPlateau decays LR if validation loss stagnates, preventing divergence from aggressive jam weighting.",
    styles['note']))
story.append(Spacer(1, 4*mm))

story.append(Paragraph("9.2 Per-Epoch Training (Lines 1405-1439)", styles['subsection']))
story.append(Preformatted(
    "for ep in range(1, epochs + 1):\n"
    "    t0 = np.random.randint(0, TRAIN_END - BATCH_TIME)  # random window\n"
    "    x_t = speed_np[t0:t0+BATCH_TIME, :].T  # [N, T]\n"
    "    m_t = (torch.rand(NUM_NODES, 1) > 0.8).expand(-1, BATCH_TIME)  # 20% obs\n"
    "    p_t = net(x_t, m_t, tod_free_t, tod_jam_t)  # forward\n"
    "    # Loss: jam-weighted MSE + spatial smoothness\n"
    "    loss_free = ((p_t - x_t) ** 2)[free_mask].mean()\n"
    "    jam_weight = clamp((1-freq_jam) / freq_jam, 1, 10)\n"
    "    loss_jam = abs(p_t - x_t)[jam_mask].mean() * jam_weight * 3.5\n"
    "    loss = loss_free + loss_jam + loss_smooth\n"
    "    opt.zero_grad(); loss.backward(); opt.step()",
    styles['code_block']))
story.append(Paragraph(
    "<b>Per-Epoch:</b> Sample random window, forward pass, compute loss. "
    "Jam weight capped at 10 (prevents 100× multiplier from degenerate jam frequencies). "
    "Multiplier 3.5 is empirically tuned for stable training.",
    styles['note']))
story.append(Spacer(1, 4*mm))

story.append(Paragraph("9.3 Validation & Early Stopping (Lines 1441-1475)", styles['subsection']))
story.append(Preformatted(
    "if ep % 10 == 0:\n"
    "    with torch.no_grad():\n"
    "        p_v = net(x_val, m_val, tod_free_v, tod_jam_v)  # validation\n"
    "        vl = loss_free_v + loss_jam_v + loss_smooth_v\n"
    "    scheduler.step(vl)  # decay LR if no improvement\n"
    "    if vl < best_vloss:\n"
    "        best_vloss = vl\n"
    "        best_wts = copy.deepcopy(net.state_dict())\n"
    "        patience_ctr = 0\n"
    "    else:\n"
    "        patience_ctr += 1\n"
    "    if patience_ctr >= 6:\n"
    "        print('Early stop'); break",
    styles['code_block']))
story.append(Paragraph(
    "<b>Validation:</b> Every 10 epochs, evaluate on validation set. "
    "If loss improves, save checkpoint. If no improvement for 6 evaluations (~60 epochs), stop. "
    "This prevents overfitting and saves training time.",
    styles['note']))
story.append(Spacer(1, 5*mm))

# SECTION 10: Baseline Models
story.append(Paragraph("10. Baseline Models (CELLS 9-11)", styles['chapter']))
story.append(Paragraph(
    "The notebook includes 23+ baseline models across four tiers: <br/>"
    "• <b>T1 (Statistical):</b> Global Mean, IDW, Linear Interpolation, KNN Kriging <br/>"
    "• <b>T2 (RNN/Temporal):</b> GRU-D, BRITS, SAITS <br/>"
    "• <b>T3 (GNN Imputation):</b> IGNNK, GRIN, GRIN++, SPIN, DGCRIN, GCASTN, GCASTN+, ADGCN <br/>"
    "• <b>SOTA References:</b> T-DGCN, Improved T-DGCN",
    styles['body']))
story.append(Paragraph(
    "All baselines use <b>cached results</b> (lines 2036-2100+); they are not retrained. "
    "Results are pre-computed and stored as dicts in results_table.",
    styles['note']))
story.append(Spacer(1, 5*mm))

# SECTION 11: Results Table
story.append(Paragraph("11. Results Table & Visualization (CELL 12)", styles['chapter']))
story.append(Paragraph("11.1 Deduplication & Sorting (Lines 3147-3158)", styles['subsection']))
story.append(Preformatted(
    "results_dedup = {}\n"
    "for r in results_table:\n"
    "    model_name = r['model']\n"
    "    if model_name not in results_dedup or r['mae_all'] < results_dedup[...]['mae_all']:\n"
    "        results_dedup[model_name] = r  # keep best run per model\n"
    "results_table_sorted = sorted(results_table, key=lambda r: r['mae_all'])",
    styles['code_block']))
story.append(Paragraph(
    "<b>Process:</b> Remove duplicate runs (if a model was trained multiple times). "
    "Sort by MAE (ascending) so best models appear first.",
    styles['note']))
story.append(Spacer(1, 4*mm))

story.append(Paragraph("11.2 Table Printing (Lines 3160-3209)", styles['subsection']))
story.append(Paragraph(
    "Print formatted table with model name, MAE all, MAE jam, Precision, Recall, F1, SSIM. "
    "Tier labels ([SOTA], [Ours], [T1], [T2], [T3]) indicate model category. "
    "v7 FreqDGT marked with ◀◀ THESIS annotation.",
    styles['body']))
story.append(Spacer(1, 3*mm))

story.append(Paragraph("11.3 Bar Chart (Lines 3211-3268)", styles['subsection']))
story.append(Preformatted(
    "plot_rows = [r for r in results_table_sorted\n"
    "             if tier_labels.get(r['model'], '') not in ('T1', '')]\n"
    "# Exclude T1 statistical baselines (MAE 2.6–43 km/h) for readability\n"
    "# Plot three charts: MAE all, MAE jam, F1 (one chart per subplot)",
    styles['code_block']))
story.append(Paragraph(
    "<b>Visualization:</b> Three subplots showing MAE all, MAE jam, F1 across all models. "
    "v7 FreqDGT highlighted with black outline and star. T1 excluded because their MAE "
    "dominates y-axis scale, making competitive models indistinguishable.",
    styles['note']))
story.append(Spacer(1, 5*mm))

# SECTION 12: Analysis
story.append(Paragraph("12. Analysis & Conclusions (CELL 13)", styles['chapter']))
story.append(Paragraph("12.1 Metric-by-Metric Comparison (Lines 3275-3314)", styles['subsection']))
story.append(Preformatted(
    "v7_result = next((r for r in results_table if 'v7 FreqDGT' in r['model']), None)\n"
    "next_best = next((r for r in results_table_sorted if 'v7' not in r['model']), None)\n"
    "# Compare each metric; mark wins with ✅\n"
    "for metric, key, higher_is_better in metrics:\n"
    "    win = '✅ WIN' if (v7_val > nb_val) == higher_is_better else '  ---'",
    styles['code_block']))
story.append(Paragraph(
    "<b>Summary:</b> v7 FreqDGT achieves: <br/>"
    "• <b>MAE all: 0.40 km/h</b> (best among 23 models; 31% below prior SOTA 0.58) <br/>"
    "• <b>Precision: 0.972</b> (when v7 predicts jam, it's correct 97.2% of time) <br/>"
    "• <b>F1: 0.938</b> (balanced detection quality) <br/>"
    "• <b>SSIM: 0.975</b> (spatial structure preservation)",
    styles['note']))
story.append(Spacer(1, 5*mm))

story.append(Paragraph("12.2 Thesis Contribution Summary (Lines 3316-3333)", styles['subsection']))
story.append(Paragraph(
    "<b>Novel Architecture Innovations:</b> <br/>"
    "1. <b>Learnable frequency decomposition</b> (moving-avg filter, not hand-tuned wavelets) <br/>"
    "2. <b>Dual-branch expert routing</b> (low-freq for trends, high-freq for spikes) <br/>"
    "3. <b>4-path Chebyshev convolution</b> (symmetric, forward, backward, correlation adjacencies) <br/>"
    "4. <b>Per-timestep dynamic graphs</b> (attention-based A_t discovers jam clusters) <br/>"
    "5. <b>Time-of-day gating</b> (expert gate conditioned on free-flow/jam priors)",
    styles['bullet']))
story.append(Spacer(1, 5*mm))

story.append(Paragraph("Conclusion", styles['chapter']))
story.append(Paragraph(
    "This codebase implements a complete traffic speed imputation pipeline on PEMS04 benchmark. "
    "The v7 FreqDGT architecture achieves state-of-the-art performance by decomposing the problem "
    "into frequency-specialized branches and routing them via learned expertise gating. "
    "All code is thoroughly documented to enable reproduction and extension.",
    styles['body']))

# Build PDF
doc.build(story)
print("✅ Code documentation PDF built: code_documentation.pdf")
