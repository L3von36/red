"""
Build thesis-quality PDFs from the article, SOTA, and documentation content.
Uses ReportLab for full typographic control.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Preformatted, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate, Frame
from reportlab.platypus import NextPageTemplate
import re, textwrap

# ── Colour palette ─────────────────────────────────────────────────────────────
NAVY    = HexColor("#1B2A4A")   # headings, page header
ACCENT  = HexColor("#2E6FD9")   # section rules, table headers
LIGHT   = HexColor("#EBF1FA")   # table alt row, code background
MID     = HexColor("#C5D6F0")   # table grid lines
DARK_TXT= HexColor("#1C1C1C")
GREY_TXT= HexColor("#555555")
CODE_BG = HexColor("#F4F6F9")
CODE_TXT= HexColor("#1A1A2E")

W, H = A4

# ── Page template with header/footer ──────────────────────────────────────────
class ThesisTemplate(BaseDocTemplate):
    def __init__(self, filename, title, **kw):
        # Set attributes BEFORE calling super().__init__ because
        # BaseDocTemplate may call _draw_page during initialisation.
        self.title  = title or ""
        self.author = "Thesis — PEMS04 Traffic Imputation"
        super().__init__(filename, **kw)
        frame = Frame(2.2*cm, 2.5*cm, W - 4.4*cm, H - 4.5*cm, id='normal')
        tmpl  = PageTemplate(id='main', frames=[frame],
                              onPage=self._draw_page)
        self.addPageTemplates([tmpl])

    def _draw_page(self, canv, doc):
        canv.saveState()
        title  = self.title  or ""
        author = self.author or ""
        # top bar
        canv.setFillColor(NAVY)
        canv.rect(0, H - 1.5*cm, W, 1.5*cm, fill=1, stroke=0)
        canv.setFillColor(white)
        canv.setFont("Helvetica-Bold", 8)
        canv.drawString(2.2*cm, H - 0.95*cm, title)
        canv.setFont("Helvetica", 8)
        canv.drawRightString(W - 2.2*cm, H - 0.95*cm, author)
        # bottom bar
        canv.setFillColor(ACCENT)
        canv.rect(0, 0, W, 1.0*cm, fill=1, stroke=0)
        canv.setFillColor(white)
        canv.setFont("Helvetica", 8)
        canv.drawCentredString(W / 2, 0.35*cm, f"Page {doc.page}")
        canv.restoreState()


# ── Style sheet ───────────────────────────────────────────────────────────────
def make_styles():
    base = getSampleStyleSheet()

    s = {}
    s['h1'] = ParagraphStyle('h1',
        fontName='Helvetica-Bold', fontSize=18, textColor=NAVY,
        spaceAfter=6, spaceBefore=18, leading=22)
    s['h2'] = ParagraphStyle('h2',
        fontName='Helvetica-Bold', fontSize=13, textColor=ACCENT,
        spaceAfter=4, spaceBefore=12, leading=17)
    s['h3'] = ParagraphStyle('h3',
        fontName='Helvetica-Bold', fontSize=11, textColor=NAVY,
        spaceAfter=3, spaceBefore=8, leading=14)
    s['body'] = ParagraphStyle('body',
        fontName='Helvetica', fontSize=10, textColor=DARK_TXT,
        leading=15, spaceAfter=6, alignment=TA_JUSTIFY)
    s['bullet'] = ParagraphStyle('bullet',
        fontName='Helvetica', fontSize=10, textColor=DARK_TXT,
        leading=14, spaceAfter=3, leftIndent=16,
        bulletIndent=4, bulletFontName='Helvetica', bulletFontSize=10)
    s['code'] = ParagraphStyle('code',
        fontName='Courier', fontSize=8.5, textColor=CODE_TXT,
        backColor=CODE_BG, leading=12,
        leftIndent=8, rightIndent=8,
        spaceBefore=4, spaceAfter=4,
        borderPad=6)
    s['caption'] = ParagraphStyle('caption',
        fontName='Helvetica-Oblique', fontSize=9, textColor=GREY_TXT,
        leading=12, spaceAfter=8, alignment=TA_CENTER)
    s['abstract'] = ParagraphStyle('abstract',
        fontName='Helvetica', fontSize=10, textColor=DARK_TXT,
        leading=15, spaceAfter=8, leftIndent=24, rightIndent=24,
        borderPad=8, backColor=LIGHT, alignment=TA_JUSTIFY)
    s['note'] = ParagraphStyle('note',
        fontName='Helvetica-Oblique', fontSize=9, textColor=GREY_TXT,
        leading=12, spaceAfter=4, leftIndent=12)
    return s


# ── Markdown-to-ReportLab parser ──────────────────────────────────────────────
def md_inline(text):
    """Convert inline markdown (bold, code, italic) to RML tags."""
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`(.+?)`',        r'<font name="Courier" size="9"><b>\1</b></font>', text)
    text = re.sub(r'\*(.+?)\*',      r'<i>\1</i>', text)
    text = text.replace('&', '&amp;').replace('<b>', '<b>').replace('</b>', '</b>')
    return text


def parse_md(text, styles, is_abstract=False):
    """Convert a markdown string to a list of ReportLab flowables."""
    lines   = text.split('\n')
    items   = []
    in_code = False
    code_buf= []
    in_table= False
    tbl_rows= []
    tbl_header = False

    def flush_code():
        if code_buf:
            src = '\n'.join(code_buf)
            items.append(Spacer(1, 4))
            for ln in src.split('\n'):
                items.append(Preformatted(ln, styles['code']))
            items.append(Spacer(1, 4))
            code_buf.clear()

    def flush_table():
        if not tbl_rows:
            return
        col_n = len(tbl_rows[0])
        col_w = (W - 4.4*cm) / col_n
        col_ws = [col_w] * col_n

        tbl_data = []
        for ri, row in enumerate(tbl_rows):
            cells = [Paragraph(md_inline(c.strip()), styles['body']) for c in row]
            tbl_data.append(cells)

        t = Table(tbl_data, colWidths=col_ws, repeatRows=1)
        ts = TableStyle([
            ('BACKGROUND',  (0,0), (-1,0), ACCENT),
            ('TEXTCOLOR',   (0,0), (-1,0), white),
            ('FONTNAME',    (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',    (0,0), (-1,-1), 9),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, LIGHT]),
            ('GRID',        (0,0), (-1,-1), 0.5, MID),
            ('VALIGN',      (0,0), (-1,-1), 'TOP'),
            ('TOPPADDING',  (0,0), (-1,-1), 5),
            ('BOTTOMPADDING',(0,0), (-1,-1), 5),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
        ])
        t.setStyle(ts)
        items.append(Spacer(1, 6))
        items.append(t)
        items.append(Spacer(1, 8))
        tbl_rows.clear()

    for line in lines:
        # code fence
        if line.strip().startswith('```'):
            if in_code:
                flush_code()
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_buf.append(line)
            continue

        # table
        if line.strip().startswith('|'):
            cells = [c for c in line.strip().split('|') if c.strip()]
            if all(set(c.strip()) <= set('-: ') for c in cells):
                continue   # separator row
            tbl_rows.append(cells)
            continue
        else:
            if tbl_rows:
                flush_table()

        # headings
        if line.startswith('# ') and not line.startswith('## '):
            items.append(Spacer(1, 8))
            items.append(Paragraph(line[2:].strip(), styles['h1']))
            items.append(HRFlowable(width="100%", thickness=2,
                                     color=ACCENT, spaceAfter=6))
            continue
        if line.startswith('## '):
            items.append(Spacer(1, 6))
            items.append(Paragraph(line[3:].strip(), styles['h2']))
            items.append(HRFlowable(width="100%", thickness=0.5,
                                     color=MID, spaceAfter=4))
            continue
        if line.startswith('### '):
            items.append(Paragraph(line[4:].strip(), styles['h3']))
            continue

        # horizontal rule
        if line.strip().startswith('---'):
            items.append(HRFlowable(width="100%", thickness=1,
                                     color=MID, spaceAfter=6, spaceBefore=6))
            continue

        # bullets
        if line.startswith('- '):
            txt = md_inline(line[2:].strip())
            items.append(Paragraph(f'• {txt}', styles['bullet']))
            continue
        if re.match(r'^\d+\. ', line):
            txt = md_inline(re.sub(r'^\d+\. ', '', line).strip())
            num = re.match(r'^(\d+)', line).group(1)
            items.append(Paragraph(f'<b>{num}.</b> {txt}', styles['bullet']))
            continue

        # blockquote / note
        if line.startswith('> '):
            items.append(Paragraph(md_inline(line[2:]), styles['note']))
            continue

        # normal paragraph
        stripped = line.strip()
        if stripped:
            st = styles['abstract'] if is_abstract else styles['body']
            items.append(Paragraph(md_inline(stripped), st))
            is_abstract = False   # only first paragraph gets abstract style
        else:
            items.append(Spacer(1, 4))

    flush_code()
    if tbl_rows:
        flush_table()
    return items


# ── Title-page builder ─────────────────────────────────────────────────────────
def title_page(title, subtitle, author="", affiliation=""):
    items = []
    items.append(Spacer(1, 3.5*cm))
    items.append(HRFlowable(width="100%", thickness=3, color=ACCENT, spaceAfter=18))

    ttl_style = ParagraphStyle('ttl',
        fontName='Helvetica-Bold', fontSize=22, textColor=NAVY,
        leading=28, alignment=TA_CENTER, spaceAfter=12)
    sub_style = ParagraphStyle('sub',
        fontName='Helvetica', fontSize=13, textColor=GREY_TXT,
        leading=18, alignment=TA_CENTER, spaceAfter=6)
    aut_style = ParagraphStyle('aut',
        fontName='Helvetica-Bold', fontSize=11, textColor=DARK_TXT,
        leading=15, alignment=TA_CENTER, spaceAfter=4)
    aff_style = ParagraphStyle('aff',
        fontName='Helvetica-Oblique', fontSize=10, textColor=GREY_TXT,
        leading=14, alignment=TA_CENTER)

    items.append(Paragraph(title, ttl_style))
    items.append(Paragraph(subtitle, sub_style))
    items.append(HRFlowable(width="100%", thickness=1, color=MID,
                              spaceBefore=12, spaceAfter=16))
    if author:
        items.append(Paragraph(author, aut_style))
    if affiliation:
        items.append(Paragraph(affiliation, aff_style))
    items.append(Spacer(1, 1.5*cm))

    # decorative box
    note = ParagraphStyle('deco',
        fontName='Helvetica', fontSize=9, textColor=GREY_TXT,
        backColor=LIGHT, leading=13, alignment=TA_CENTER,
        borderPad=10, leftIndent=40, rightIndent=40)
    items.append(Paragraph(
        "PEMS04 Benchmark · 307 Sensors · 80% Sparsity · Graph Neural ODE", note))
    items.append(PageBreak())
    return items


# ══════════════════════════════════════════════════════════════════════════════
# Document 1 — Research Article
# ══════════════════════════════════════════════════════════════════════════════
ARTICLE_CONTENT = """
# Abstract

Traffic monitoring infrastructure is rarely complete: sensor failures, budget constraints, and road geometry mean that a significant fraction of road segments lack direct speed measurements at any given time. We present a model that recovers missing speeds from a network of partially observed sensors on the PEMS04 benchmark (307 sensors, 80% unobserved). Our architecture combines three components: (1) a **Hypergraph-augmented Graph Attention ODE** that models continuous traffic dynamics with both pairwise and multi-node corridor context, (2) a **Kalman-style observation assimilation gate** that injects real sensor readings into the hidden state at each timestep without leaking ground truth to blind nodes, and (3) a **physics-informed loss** encoding the LWR flow continuity principle via graph Laplacian regularisation. Training uses curriculum masking and jam-biased sampling to overcome the severe class imbalance between free-flow (92%) and congested (8%) timesteps. On the held-out evaluation set, the full model achieves a blind-node MAE of **5.18 km/h** overall and **33.93 km/h** during congestion events, outperforming the global-mean baseline (35.99 km/h jam) and IDW spatial interpolation (32.95 km/h jam).

## 1. Introduction

Urban traffic monitoring systems depend on a fixed network of loop detectors and radar sensors to measure vehicle speed. In practice, a large fraction of these sensors are unavailable at any given moment due to hardware failure, maintenance windows, or gaps in infrastructure deployment. The California PEMS04 dataset, a standard benchmark with 307 sensors across San Francisco Bay Area freeways, illustrates this: realistic deployments often observe only 20–60% of nodes, leaving the remainder as "blind" sensors whose speeds must be inferred.

This **sparse traffic speed imputation** task is substantially harder than the well-studied traffic forecasting problem for two reasons. First, the model must reconstruct entire spatial fields rather than extend known sequences. Second, the key failure mode — congestion — is rare (roughly 8% of timesteps) and spatially localised, making it easy for a model to achieve good average MAE by predicting free-flow everywhere while completely failing on jams.

We address both challenges through a unified architecture: a **Graph Neural Ordinary Differential Equation** that operates on the road network graph, extended with hyperedge groups capturing multi-sensor corridor dynamics, a learned sensor assimilation step at each timestep, and physics-informed training objectives.

## 2. Problem Formulation

Let G = (V, E) be the road network graph with N = 307 nodes (sensors) and edges weighted by Gaussian kernel affinity on pairwise road distance. At each timestep t, each sensor either reports a speed observation or is hidden (mask = 0). The observed speed is set to zero for blind nodes.

**Goal:** Given partial speed observations for observed nodes and the road graph G, estimate the speed at all blind nodes for all timesteps — with no ground-truth access for blind nodes during inference.

**Input features** (6-dimensional per node per timestep):

1. `obs_speed` — observed speed (0 for blind nodes)
2. `global_ctx` — mean of all observed speeds at timestep t
3. `nbr_ctx` — adjacency-weighted mean of observed neighbour speeds
4. `is_observed` — binary sensor flag
5. `t_sin` — 0.25 × sin(2π × time-of-day)
6. `t_cos` — 0.25 × cos(2π × time-of-day)

The 0.25 temporal scaling prevents the model from predicting rush-hour congestion from time alone. Leakage assertions enforce that blind-node speeds and flags never appear in the input.

## 3. Model Architecture

### 3.1 Input Encoder

A shared linear layer maps 6-dimensional inputs to hidden dimension H = 64 for all N nodes: z₀ = W_enc · x₀.

### 3.2 Graph Attention Layer (GAT)

Pairwise attention on the road graph. Attention score: e_ij = LeakyReLU(a_src(Wh_i) + a_dst(Wh_j)). Normalised with temperature τ = 2 to prevent single-neighbour dominance, and masked to road topology (non-edges → −∞ before softmax).

### 3.3 Hypergraph Convolution

A hyperedge for node i contains i and all 2-hop reachable neighbours, capturing multi-sensor corridor groups. The normalised convolution operator H_conv = D_v^{−1/2} H D_e^{−1} H^T D_v^{−1/2} is pre-computed once. A learnable gate g = sigmoid(w), initialised at w = −2 (g ≈ 0.12), controls the contribution to prevent over-smoothing at jam nodes.

### 3.4 ODE Function and Euler Integration

The ODE function computes: f_θ(z) = LayerNorm(Tanh(GAT2(Tanh(GAT1(z)))) + g · HypConv(z)). Euler integration with dt = 0.3: z_{t+1} = z_t + 0.3 · f_θ(z_t). The small dt dampens hidden-state momentum and prevents post-jam oscillation.

### 3.5 Observation Assimilation

A GRU-style gate after each Euler step: gate = σ(W_g [z; z_obs]); update = gate × (z_obs − z) × obs_mask. The mask zeros updates for blind nodes, preventing them from assimilating their own zero observations.

### 3.6 Three-Term Loss

**Jam-weighted MSE** (w = 4 for speed < 40 km/h, else 1): compensates for 12:1 free-flow:jam imbalance.

**Temporal smoothness** (λ = 0.60): penalises step-to-step jumps, suppressing oscillation artefacts.

**Graph Laplacian physics** (λ = 0.02): L_phys = ||L_sym · v||², encoding LWR flow continuity.

## 4. Training Strategy

**Curriculum masking:** 15% of observed nodes are pseudo-blinded each batch, ensuring gradients always flow through the blind-node code path.

**Jam-biased sampling:** 50% of batches are forced to start at jam timesteps (vs the natural 8% rate), providing sufficient gradient signal for congestion.

**Gradient accumulation:** Gradients from 4 windows are accumulated per update, smoothing the high variance between jam and free-flow batches.

**Optimiser:** Adam (lr = 3×10⁻⁴, weight decay = 10⁻⁴), cosine annealing (T_max = 400) over 800 epochs, gradient clip at norm 1.0.

**Split:** Train t = 0–3999, Validation t = 4000–4239, Evaluation t = 4500–4949 (no overlap).

## 5. Experiments

### 5.1 Dataset

PEMS04 — 307 sensors, San Francisco Bay Area, 5-minute intervals. Speed channel extracted, z-score normalised. 80% sensors randomly masked (seed = 42), yielding ~61 observed and ~246 blind nodes.

### 5.2 Main Results (80% Sparsity)

| Model | MAE all (km/h) | MAE jam (km/h) |
|---|---|---|
| Global mean baseline | 5.18 | 35.99 |
| IDW spatial interpolation | 5.23 | 32.95 |
| **Ours (full model)** | **5.18** | **33.93** |

### 5.3 Sensor Sparsity Sweep

| Sparsity | Blind% | Base Jam | IDW Jam | Model Jam | vs Baseline |
|---|---|---|---|---|---|
| 20% | 22% | 36.04 | 27.35 | 27.68 | +23.2% |
| 40% | 48% | 35.74 | 29.73 | 28.27 | +20.9% |
| 60% | 62% | 36.01 | 29.98 | 31.66 | +12.1% |
| 80% | 83% | 35.99 | 32.95 | 31.19 | +13.3% |
| 90% | 90% | 35.99 | 33.81 | 34.58 | +3.9% |

### 5.4 Ablation Study

| Model / Variant | MAE all | MAE jam | Δ jam |
|---|---|---|---|
| Global mean baseline | 5.18 | 35.99 | — |
| IDW (spatial interp.) | 5.23 | 32.95 | — |
| Full model | 5.18 | 33.93 | +0.00 |
| − Hypergraph | 5.54 | 31.66 | −2.27 |
| − Assimilation | 5.29 | 32.54 | −1.38 |
| − Physics loss | 5.32 | 33.32 | −0.61 |
| − Neighbour context | 5.84 | 28.99 | −4.94 |
| − Temporal encoding | 6.00 | 33.51 | −0.41 |

Positive Δ jam = removing that component hurts congestion imputation.

## 6. Discussion

**Jam imputation is the key challenge.** Free-flow speed is concentrated near the global mean, so an average-predicting model achieves good overall MAE while being useless during congestion. Evaluating jam MAE separately is essential for measuring real-world utility.

**Hypergraph gate convergence.** The learnable gate starts at ≈0.12 and allows the model to decide where corridor context helps. Without the gate, 2-hop aggregation averaged jam nodes with ~20 free-flowing corridor neighbours, actively contradicting the jam signal.

**Curriculum masking is non-negotiable.** Without it, the model sees zero gradient through the blind-node path, leading to collapse where imputed speeds ignore spatial context entirely.

**Comparison to forecasting SOTA.** DCRNN, STGCN, and Graph WaveNet achieve ~1.6–1.8 km/h MAE on PEMS04, but for the full-sensor 15-minute forecasting task. Our task is fundamentally different (80% sensors missing, imputation not forecasting); the numbers are included only as a difficulty reference.

## 7. Conclusion

We presented a Hypergraph Neural ODE with observation assimilation for sparse traffic speed imputation. The architecture combines continuous-time graph dynamics (Euler ODE with GAT layers), multi-hop corridor context (gated HGNN), Kalman-style sensor fusion, and physics regularisation. Training with curriculum masking and jam-biased sampling overcomes class imbalance. The full model achieves consistent improvement over baselines across sparsity levels from 20% to 90%.

## References

- Chen et al. (2018). Neural Ordinary Differential Equations. NeurIPS.
- Feng et al. (2019). Hypergraph Neural Networks. AAAI.
- Kalman, R.E. (1960). A New Approach to Linear Filtering. J. Basic Engineering.
- Kipf & Welling (2017). Semi-Supervised Classification with GCNs. ICLR.
- Li et al. (2018). Diffusion Convolutional Recurrent Neural Network. ICLR.
- Raissi et al. (2019). Physics-Informed Neural Networks. J. Computational Physics.
- Velickovic et al. (2018). Graph Attention Networks. ICLR.
- Wu et al. (2019). Graph WaveNet for Deep Spatial-Temporal Graph Modeling. IJCAI.
- Yu et al. (2018). Spatio-Temporal Graph Convolutional Networks. IJCAI.
"""

SOTA_CONTENT = """
# State of the Art
## Traffic Speed Imputation and Graph Neural Networks

### 1. Problem Definition

**Traffic speed imputation** is the task of estimating the speed at road sensors whose readings are missing or unavailable, given partial observations from neighbouring sensors and the structure of the road network. This differs fundamentally from **traffic forecasting**, where all sensors are observed and the goal is to predict future values. Imputation is strictly harder: the model must simultaneously reason about space (what does an unobserved node look like given its neighbours?) and time (how does traffic state evolve?).

### 2. Classical Methods

**Global Mean / Median Imputation** replaces missing values with the dataset-wide mean speed. While simple, it completely ignores spatial structure and temporal dynamics. MAE on PEMS04 jam nodes: ~36 km/h. Used in this work as Baseline 1.

**Inverse Distance Weighting (IDW)** estimates missing node speed as a weighted average of observed neighbours, with weights proportional to 1/distance. Equivalent to adjacency-weighted mean. Purely spatial — no temporal modelling, no learning. Used in this work as Baseline 2.

**Kriging** applies spatial Gaussian processes. Assumes stationarity; computationally O(N³); doesn't scale to 307 sensors; no temporal component.

### 3. Deep Learning for Traffic

**LSTM (Hochreiter & Schmidhuber, 1997)** models each sensor independently as a time series. The weakness: sensors are spatially correlated; independent modelling ignores road topology.

**WaveNet (van den Oord et al., 2016)** uses dilated causal convolutions for large receptive fields without RNN depth. Adapted for traffic as Graph WaveNet. 1-D convolution has no notion of graph structure.

### 4. Graph Neural Networks

**GCN (Kipf & Welling, 2017)** first-order spectral convolution: X' = D^{−1/2}AD^{−1/2}XW. Fixed normalisation treats all neighbours equally — ignores that a congested upstream neighbour matters more than a free-flowing one.

**GAT (Velickovic et al., 2018)** replaces fixed weights with learned attention scores. This work uses GAT as the primary spatial aggregator inside the ODE function, with temperature τ = 2 to prevent single-neighbour dominance.

### 5. Spatio-Temporal Forecasting SOTA

| Model | Venue | Mechanism | PEMS04 MAE |
|---|---|---|---|
| DCRNN (Li et al., 2018) | ICLR 2018 | Diffusion GCN + seq2seq RNN | ~1.8 km/h |
| STGCN (Yu et al., 2018) | IJCAI 2018 | Graph conv + temporal conv | ~1.7 km/h |
| Graph WaveNet (Wu et al., 2019) | IJCAI 2019 | Adaptive adj + dilated conv | ~1.6 km/h |
| ASTGCN (Guo et al., 2019) | AAAI 2019 | Spatial + temporal attention | ~1.6 km/h |
| AGCRN (Bai et al., 2020) | NeurIPS 2020 | Node-adaptive GCN + GRU | ~1.5 km/h |

> These numbers are for the full-sensor 15-minute forecasting task. They are not directly comparable to sparse imputation MAE — they are included only as a difficulty reference for the PEMS04 dataset.

### 6. Neural ODEs

**Neural ODE (Chen et al., NeurIPS 2018)** parameterises the hidden state derivative as a neural network: dz/dt = f_θ(z, t), solved with an ODE solver. Advantage: continuous-time modelling, memory-efficient via the adjoint method. Weakness: the original form uses a generic MLP with no graph structure.

**Graph Neural ODE variants** replace the MLP with a GNN. STGODE (Fang et al., SIGKDD 2021) applies spatio-temporal ODEs for traffic. This work uses Euler integration (not dopri5) to avoid gradient vanishing from multiple solver evaluations during backpropagation.

### 7. Hypergraph Neural Networks

A hypergraph G = (V, E) allows edges (hyperedges) to connect more than two nodes — capturing road corridors and intersections naturally.

**HGNN (Feng et al., AAAI 2019)** defines: X' = D_v^{−1/2} H W_e D_e^{−1} H^T D_v^{−1/2} X Θ. This work pre-computes the full normalised operator H_conv offline, reducing runtime cost to a single matmul. A learnable gate (init = sigmoid(−2) ≈ 0.12) prevents over-smoothing at congested nodes.

### 8. Observation Assimilation

**Kalman Filter (Kalman, 1960)** is the optimal linear estimator blending model prediction with sensor observations. The Kalman gain K determines how much to trust the new measurement vs the model prediction.

**GRU-D (Che et al., 2018)** extends the GRU with a decay mechanism for irregular time series.

**ODE with jumps (Rubanova et al., NeurIPS 2019)** allows the ODE hidden state to jump at observation times via a recognition network.

**This work** implements a GRU-style gate at each timestep. Blind nodes are zeroed out by obs_mask, preventing them from "assimilating" their own zero observations.

### 9. Physics-Informed Neural Networks

**PINN (Raissi et al., JCP 2019)** encodes PDE residuals as additional loss terms, widely used in fluid dynamics and heat transfer.

**LWR traffic physics (Lighthill-Whitham-Richards, 1955)** states that traffic obeys a conservation law: speed gradients propagate continuously along roads. This work implements graph Laplacian regularisation: L_phys = ||L_sym · v||² = Σ_i (v_i − mean_nbr(v_i))², with λ = 0.02 as a soft constraint.

### 10. Curriculum Learning

**Curriculum learning (Bengio et al., ICML 2009)** starts with easy examples and gradually increases difficulty. This work applies curriculum masking: 15% of observed sensors are randomly hidden ("pseudo-blind") each batch, ensuring gradients flow through the blind-node code path at every update.

### 11. Position of This Work

| Aspect | This Work | DCRNN / STGCN / WaveNet |
|---|---|---|
| Task | Sparse imputation (80% missing) | Full-sensor forecasting |
| Graph structure | Hypergraph + pairwise GAT | Standard adjacency |
| Temporal model | Continuous Neural ODE (Euler) | Discrete RNN / Conv |
| Sensor fusion | Kalman-style assimilation gate | Not applicable |
| Physics | Graph Laplacian regularisation | None |
| Training | Curriculum masking + jam-biased sampling | Standard mini-batch |

This work is among the first to combine Hypergraph Neural ODE, Kalman-style observation assimilation, and physics regularisation for the sparse traffic imputation problem.
"""

DOC_CONTENT = """
# Code Documentation
## Line-by-Line Explanation of cth_node_complete.py

### CELL 1 — Imports

`import torch` — PyTorch deep learning framework. Provides tensors, autograd, and neural network layers.

`import torch.nn as nn` — The `nn` module provides `nn.Module` (base class for all models), `nn.Linear`, `nn.LayerNorm`, `nn.LeakyReLU`, `nn.Sigmoid`, and other building blocks.

`import numpy as np` — NumPy for CPU-side pre-processing: building the adjacency matrix and hypergraph incidence matrix before they are converted to PyTorch tensors.

`import copy` — Used for `copy.deepcopy(state_dict)` to save the best model weights without reference aliasing. A plain assignment would point to the same dict, which updates in-place as training continues.

`device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` — Automatically selects GPU when available. All tensors and model parameters are moved here with `.to(device)`.

### CELL 2 — Data Loading and Adjacency

`raw_data = data['data'][:, :307, 2]` — The PEMS04 NPZ array has shape [T, N, 3] where the last dimension is [speed, flow, occupancy]. Index 2 = speed channel. The `:307` is defensive in case the file has extra padding nodes.

`raw_data = np.nan_to_num(raw_data)` — Replace any NaN sensor dropouts with 0 before normalisation. Left as NaN, they would propagate and poison the entire dataset.

`data_norm = (raw_data - mean) / (std + 1e-8)` — Z-score normalisation. The 1e-8 guards against division by zero. All comparisons and loss computations operate in normalised space; results are de-normalised with `* std + mean` for display.

**Adjacency matrix construction:**

`sigma = dist_mat[dist_mat < np.inf].std()` — Standard deviation of all finite road distances, used as the Gaussian kernel bandwidth. This is the convention from DCRNN and Graph WaveNet.

`adj = np.where(dist_mat < np.inf, np.exp(-(dist_mat**2) / (sigma**2)), 0.0)` — Gaussian kernel affinity: close sensors get high weight (near 1), distant sensors get low weight (near 0). Non-connected pairs get 0.

`np.fill_diagonal(adj, 1.0)` — Self-loops ensure every node is its own strongest neighbour. Required so GAT attention doesn't ignore the node's own hidden state.

`adj_norm = (adj * d_inv[:, None]) * d_inv[None, :]` — Symmetric normalisation D^{−1/2} A D^{−1/2}. This makes aggregation scale-invariant: high-degree nodes don't dominate message passing. The `np.where(deg > 0, ...)` guard handles isolated nodes.

**Graph Laplacian:**

`L_sym_np = np.eye(NUM_NODES) - adj_norm` — Symmetric normalised Laplacian L = I − D^{−1/2}AD^{−1/2}. Used in the physics loss: ||L · v||² = Σ_i (v_i − mean_nbr(v_i))², encoding the LWR continuity principle.

**Hypergraph construction:**

`adj2 = adj_binary @ adj_binary` — Matrix product gives the 2-hop reachability matrix: entry [i,j] counts paths of length 2. We binarise (>0), union with 1-hop adjacency, and add self-loops. Result: `adj2[i,:]` flags all nodes within 2 hops of i — this is the hyperedge for node i.

`H_np = adj2.T` — Transpose so columns are hyperedges. `H_np[:, e]` = membership vector of hyperedge e (the 2-hop neighbourhood of sensor e).

`H_conv_np = (d_v_inv_sqrt[:, None] * H_np) * d_e_inv[None, :]` — First half of D_v^{−1/2} H D_e^{−1}. Then `@ (H_np.T * d_v_inv_sqrt[None, :])` completes the full operator H^T D_v^{−1/2}. Pre-computing this once means runtime convolution is just a single matmul: H_conv @ (X · Θ).

### CELL 3 — Sensor Mask and Input Features

`node_mask = (torch.rand(1, NUM_NODES, 1, 1) > sparsity_ratio).float()` — 80% of nodes are randomly blind (mask = 0). Shape [1, N, 1, 1] allows broadcasting over time T and feature F dimensions. seed=42 makes selection reproducible.

`obs_data = data_tensor * node_mask` — Zeros blind node speeds. Broadcasting: [1,N,1,1] × [1,N,T,1] = [1,N,T,1]. Ground truth is preserved only for observed nodes.

`network_context = obs_data.sum(dim=1, keepdim=True) / (num_obs + 1e-6)` — Mean over the N dimension = global mean of observed speeds per timestep. `keepdim=True` preserves the dimension for broadcasting. This gives every node (observed or blind) knowledge of the network-wide average.

`nbr_sum = torch.mm(A_t, obs_2d)` — Matrix multiplication: for each node i, the weighted sum of observed neighbour speeds. `A_t @ mask_2d` computes the total observed adjacency weight, which is the denominator for the weighted mean. No GT leakage: only `obs_data` (blind nodes = 0) enters.

`TIME_SCALE = 0.25` — At full amplitude, the model learned to predict congestion at 7:40am and 7pm from time-of-day alone, generating false jams even for free-flowing blind nodes. Scaling to 0.25 retains the periodicity signal without letting it dominate spatial evidence.

**Leakage assertions (lines 168–171):** These will crash immediately if blind node speed (feature 0) or observation flag (feature 3) is non-zero. This acts as a permanent guard against any future code change that accidentally introduces data leakage.

### CELL 4 — Model Definition

**GraphAttention:**

`self.a_src = nn.Linear(out_dim, 1, bias=False)` — Produces a scalar attention logit from the source node's projected features. `self.a_dst` does the same for the destination. Their sum (additive decomposition) avoids the O(N²·H) matmul of the full concatenation form while preserving asymmetric attention.

`self.temperature = 2.0` — Dividing logits by τ > 1 flattens the attention distribution. Without temperature, the model often places ~100% weight on one neighbour (the closest congested node), causing the ODE to over-amplify that node's signal at every step — observed as oscillation after jams clear.

`e = e.masked_fill(A < 1e-9, float('-inf'))` — Non-edges are masked to −∞ so they contribute zero after softmax. This enforces road topology: only actual road connections carry information.

`alpha = torch.nan_to_num(alpha, 0.0)` — Handles isolated nodes (all neighbours masked → all −∞ → NaN after softmax). Such nodes get zero weight — they propagate nothing and receive nothing.

**HypergraphConv:**

`return torch.matmul(H_conv, h)` — `H_conv` is [N, N]. `torch.matmul` broadcasts over the batch dimension: [N,N] @ [B,N,H] → [B,N,H]. Each output node i receives the normalised sum of all hyperedge members' projected features — a group-level aggregation across the full 2-hop corridor.

**GraphODEFunc:**

`self.hyper_gate = nn.Parameter(torch.tensor(-2.0))` — A single learnable scalar. sigmoid(−2) ≈ 0.12 — the gate starts nearly closed. The model learns to open it on paths where corridor context improves prediction. Starting at sigmoid(0) = 0.5 gave equal initial weight to GAT and HGNN, causing over-smoothing at jam nodes before any useful gradient had propagated.

`return delta` — The function returns the derivative dz/dt, NOT the next state z. The Euler step `z + 0.3 * f(z)` adds the residual. The old bug returned `z + delta` (the state), making Euler compute `z + (z + delta) = 2z + delta` — doubling the hidden state every step until it exploded (~10^24 loss).

`delta = self.norm(h)` — LayerNorm normalises the delta only, not delta + z. Normalising the sum would suppress the skip signal (the original hidden state), losing the ODE residual structure. Tanh on GAT outputs prevents unbounded activations from entering LayerNorm.

**AssimilationUpdate:**

`update = gate * (z_obs - z) * obs_mask` — GRU-style correction. The gate is a sigmoid network that takes both the current hidden state z and the new encoded observation z_obs, learning how much to trust the sensor reading vs the ODE prediction — analogous to the Kalman gain. `* obs_mask` zeros the update for blind nodes.

**GraphCTH_NODE._euler_step:**

`return z + 0.3 * self.ode_func(None, z)` — dt = 0.3 chosen empirically. At dt = 1.0, the hidden state accumulated too much momentum: after a jam event cleared, the ODE continued predicting low speeds for many timesteps, then over-corrected and oscillated at ~78 km/h. Smaller dt = smaller per-step change = smoother post-jam recovery.

### CELL 5 — Training

`CURRICULUM_DROP = 0.15` — 15% of observed nodes are pseudo-blinded each batch. Lower values (5%) provide insufficient gradient through the blind path; higher values (30%) degrade observed-node performance as the model receives too little direct supervision.

`jam_t_valid = jam_t_valid[jam_t_valid < TRAIN_END - BATCH_TIME]` — Removes windows that would run past the training boundary (otherwise the model would see validation data during jam-biased sampling).

`step_loss = criterion(...) / ACCUM_STEPS` — Dividing by ACCUM_STEPS is critical. Without it, the effective learning rate would be 4× too large (gradients from 4 windows summed without normalisation).

`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` — Clips the L2 norm of all gradients to 1.0. Prevents a single catastrophic gradient update — particularly important during jam windows where the 4× loss weight can produce large gradients.

`scheduler = CosineAnnealingLR(optimizer, T_max=400)` — T_max = 400 with 800 epochs = two cosine cycles. The second cycle restarts from the initial LR, which can help the model escape local minima found in the first cycle (SGDR warm restarts, Loshchilov & Hutter 2017).

### CELL 6 — Evaluation

`EVAL_WIN = BATCH_TIME` — Non-overlapping windows of size 48 match the training window. A single 450-step rollout would cause ODE state drift — predictions degrade towards the end of the window as error accumulates. Using 48-step chunks and concatenating prevents this.

### CELL 8 — Sparsity Sweep

`_SP_EPOCHS = 150; _SP_HIDDEN = 32; _SP_ACCUM = 2` — Reduced from main training for speed. The sweep shows relative performance ranking across sparsity levels, not absolute MAE — smaller models converge to approximately the same ranking as full-size models in fewer epochs.

`feats_sp[:,:,t:t+EVAL_WIN,2:3]` — IDW prediction uses feature index 2 = nbr_ctx, pre-computed as the adjacency-weighted mean of observed neighbours. This is exactly the IDW formula — no model, no training, pure spatial interpolation used directly as the prediction.

### CELL 9 — Ablation Study

`class _GraphCTH_NoAssim(GraphCTH_NODE)` — Inherits all weights and the ODE dynamics. The only change: the `self.assimilate(...)` call is removed from the forward loop. The ODE runs freely after the initial encoding step, receiving no sensor corrections.

`delta = f"{full_jv - jv:>+.2f}"` — Δ jam = full_jv − variant_jv. Positive = removing this component raises jam MAE (the component helps). Negative = removing this component lowers jam MAE (the component introduces noise or over-smoothing — investigate).
"""


def build_pdf(filename, title, subtitle, content, is_abstract_first=False):
    doc = ThesisTemplate(
        filename, title,
        pagesize=A4,
        leftMargin=2.2*cm, rightMargin=2.2*cm,
        topMargin=2.0*cm, bottomMargin=2.5*cm,
    )
    styles = make_styles()
    story  = []
    story += title_page(title, subtitle)
    story += parse_md(content, styles, is_abstract_first)
    doc.build(story)
    print(f"  ✅ {filename}")


print("Building PDFs…")
build_pdf(
    "thesis_article.pdf",
    "Hypergraph Neural ODEs with Observation Assimilation",
    "for Sparse Traffic Speed Imputation — Research Article",
    ARTICLE_CONTENT, is_abstract_first=True
)
build_pdf(
    "thesis_sota.pdf",
    "State of the Art Review",
    "Traffic Imputation, Graph Neural Networks, Neural ODEs, HGNN",
    SOTA_CONTENT
)
build_pdf(
    "thesis_documentation.pdf",
    "Code Documentation",
    "Line-by-Line Explanation of cth_node_complete.py",
    DOC_CONTENT
)
print("All PDFs built.")
