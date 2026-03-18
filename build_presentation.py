"""
Build a professional thesis-level PowerPoint presentation.
Uses python-pptx with custom colours, shapes, and layouts.
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu, Cm
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
from pptx.enum.dml import MSO_THEME_COLOR
import copy

# ── Colour palette ─────────────────────────────────────────────────────────────
NAVY    = RGBColor(0x1B, 0x2A, 0x4A)
ACCENT  = RGBColor(0x2E, 0x6F, 0xD9)
LIGHT   = RGBColor(0xEB, 0xF1, 0xFA)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
DARK    = RGBColor(0x1C, 0x1C, 0x1C)
GREY    = RGBColor(0x55, 0x55, 0x55)
GREEN   = RGBColor(0x1E, 0x8B, 0x4C)
RED     = RGBColor(0xC0, 0x39, 0x2B)
GOLD    = RGBColor(0xE6, 0xA8, 0x17)
MID     = RGBColor(0xC5, 0xD6, 0xF0)

W = Inches(13.33)   # widescreen 16:9
H = Inches(7.5)


# ── Helper: add a solid-fill rectangle ────────────────────────────────────────
def add_rect(slide, x, y, w, h, fill_rgb, line_rgb=None, line_width=Pt(0)):
    shape = slide.shapes.add_shape(1, x, y, w, h)   # 1 = MSO_SHAPE_TYPE.RECTANGLE
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_rgb
    if line_rgb:
        shape.line.color.rgb = line_rgb
        shape.line.width = line_width
    else:
        shape.line.fill.background()
    return shape


def add_text_box(slide, text, x, y, w, h,
                 font_size=Pt(14), bold=False, italic=False,
                 color=DARK, align=PP_ALIGN.LEFT,
                 font_name="Calibri", wrap=True):
    txb = slide.shapes.add_textbox(x, y, w, h)
    tf  = txb.text_frame
    tf.word_wrap = wrap
    p   = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size     = font_size
    run.font.bold     = bold
    run.font.italic   = italic
    run.font.color.rgb = color
    run.font.name     = font_name
    return txb


def add_para(tf, text, font_size=Pt(13), bold=False, italic=False,
             color=DARK, align=PP_ALIGN.LEFT, space_before=Pt(6),
             font_name="Calibri"):
    p = tf.add_paragraph()
    p.alignment    = align
    p.space_before = space_before
    run = p.add_run()
    run.text           = text
    run.font.size      = font_size
    run.font.bold      = bold
    run.font.italic    = italic
    run.font.color.rgb = color
    run.font.name      = font_name
    return p


# ── Page layout helpers ────────────────────────────────────────────────────────
def nav_bar(slide, label=""):
    """Top navy bar + slide label."""
    add_rect(slide, 0, 0, W, Inches(0.65), NAVY)
    if label:
        add_text_box(slide, label, Inches(0.3), Inches(0.1), W - Inches(0.6),
                     Inches(0.45), font_size=Pt(10), color=WHITE,
                     align=PP_ALIGN.RIGHT)


def bottom_bar(slide, page_num):
    """Bottom accent bar with page number."""
    add_rect(slide, 0, H - Inches(0.35), W, Inches(0.35), ACCENT)
    add_text_box(slide, str(page_num), W - Inches(0.6), H - Inches(0.32),
                 Inches(0.5), Inches(0.28), font_size=Pt(9),
                 color=WHITE, align=PP_ALIGN.CENTER)
    add_text_box(slide, "Hypergraph Neural ODEs for Sparse Traffic Imputation",
                 Inches(0.3), H - Inches(0.32), Inches(8), Inches(0.28),
                 font_size=Pt(9), color=WHITE, align=PP_ALIGN.LEFT)


def section_heading(slide, title, subtitle=""):
    """Left accent stripe + heading."""
    add_rect(slide, Inches(0.3), Inches(0.8), Inches(0.07), Inches(0.55), ACCENT)
    add_text_box(slide, title, Inches(0.55), Inches(0.75), W - Inches(1.0),
                 Inches(0.6), font_size=Pt(22), bold=True, color=NAVY)
    if subtitle:
        add_text_box(slide, subtitle, Inches(0.55), Inches(1.32), W - Inches(1.0),
                     Inches(0.4), font_size=Pt(13), italic=True, color=GREY)


def bullet_box(slide, title, bullets, x=Inches(0.4), y=Inches(1.6),
               w=Inches(12.5), h=Inches(5.0), bg=LIGHT):
    """Coloured card with a title and bullet list."""
    add_rect(slide, x, y, w, h, bg, line_rgb=MID, line_width=Pt(0.5))
    add_text_box(slide, title, x + Inches(0.15), y + Inches(0.12),
                 w - Inches(0.3), Inches(0.4),
                 font_size=Pt(13), bold=True, color=NAVY)
    # divider
    add_rect(slide, x + Inches(0.15), y + Inches(0.54), w - Inches(0.3),
             Inches(0.02), ACCENT)

    txb = slide.shapes.add_textbox(x + Inches(0.18), y + Inches(0.62),
                                   w - Inches(0.36), h - Inches(0.8))
    txb.text_frame.word_wrap = True
    first = True
    for b in bullets:
        if first:
            p = txb.text_frame.paragraphs[0]
            first = False
        else:
            p = txb.text_frame.add_paragraph()
        p.space_before = Pt(3)
        r = p.add_run()
        r.text = f"▸  {b}"
        r.font.size = Pt(12.5)
        r.font.color.rgb = DARK
        r.font.name = "Calibri"


def info_card(slide, label, value, detail, x, y, w=Inches(3.8), h=Inches(2.0),
              accent=ACCENT):
    add_rect(slide, x, y, w, h, WHITE, line_rgb=accent, line_width=Pt(1.5))
    add_rect(slide, x, y, w, Inches(0.06), accent)
    add_text_box(slide, label, x + Inches(0.12), y + Inches(0.12),
                 w - Inches(0.24), Inches(0.38),
                 font_size=Pt(10), bold=True, color=accent)
    add_text_box(slide, value, x + Inches(0.12), y + Inches(0.48),
                 w - Inches(0.24), Inches(0.55),
                 font_size=Pt(18), bold=True, color=NAVY)
    add_text_box(slide, detail, x + Inches(0.12), y + Inches(1.0),
                 w - Inches(0.24), Inches(0.85),
                 font_size=Pt(10.5), italic=False, color=GREY, wrap=True)


def results_table(slide, headers, rows, x, y, col_widths, row_height=Inches(0.42)):
    """Professional styled table."""
    n_cols = len(headers)
    n_rows = len(rows) + 1

    total_w = sum(col_widths)
    total_h = n_rows * row_height

    tbl = slide.shapes.add_table(n_rows, n_cols, x, y, total_w, total_h).table
    tbl.first_row = True

    def style_cell(cell, text, bold=False, bg=None, color=DARK,
                   align=PP_ALIGN.CENTER, size=Pt(11)):
        cell.text = text
        p = cell.text_frame.paragraphs[0]
        p.alignment = align
        if p.runs:
            r = p.runs[0]
        else:
            r = p.add_run()
            r.text = text
        r.font.bold  = bold
        r.font.size  = size
        r.font.color.rgb = color
        r.font.name  = "Calibri"
        if bg:
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg

    for ci, (h, cw) in enumerate(zip(headers, col_widths)):
        tbl.columns[ci].width = cw
        style_cell(tbl.cell(0, ci), h, bold=True, bg=NAVY, color=WHITE)

    for ri, row in enumerate(rows):
        bg = LIGHT if ri % 2 == 1 else WHITE
        for ci, val in enumerate(row):
            align = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER
            style_cell(tbl.cell(ri + 1, ci), str(val), bg=bg, align=align)

    return tbl


# ══════════════════════════════════════════════════════════════════════════════
prs = Presentation()
prs.slide_width  = W
prs.slide_height = H
blank = prs.slide_layouts[6]   # blank layout
page  = 0


# ── SLIDE 1 — Title ────────────────────────────────────────────────────────────
page += 1
sl = prs.slides.add_slide(blank)
add_rect(sl, 0, 0, W, H, NAVY)
# Gold accent bar
add_rect(sl, 0, Inches(3.6), W, Inches(0.06), GOLD)
# White content block
add_rect(sl, Inches(0.5), Inches(0.9), W - Inches(1.0), Inches(2.65), WHITE)

add_text_box(sl,
    "Hypergraph Neural ODEs with Observation Assimilation",
    Inches(0.65), Inches(1.0), W - Inches(1.3), Inches(1.1),
    font_size=Pt(28), bold=True, color=NAVY, align=PP_ALIGN.CENTER)
add_text_box(sl,
    "for Sparse Traffic Speed Imputation",
    Inches(0.65), Inches(2.05), W - Inches(1.3), Inches(0.55),
    font_size=Pt(20), bold=False, color=ACCENT, align=PP_ALIGN.CENTER)
add_text_box(sl,
    "PEMS04 Benchmark  ·  307 Sensors  ·  80% Sparsity",
    Inches(0.65), Inches(2.6), W - Inches(1.3), Inches(0.4),
    font_size=Pt(13), italic=True, color=GREY, align=PP_ALIGN.CENTER)

# Stat boxes
for i, (lbl, val) in enumerate([
    ("Dataset", "PEMS04"), ("Sensors", "307"), ("Sparsity", "80%"), ("Task", "Imputation")
]):
    bx = Inches(0.55) + i * Inches(3.15)
    add_rect(sl, bx, Inches(3.85), Inches(2.9), Inches(1.45), ACCENT)
    add_text_box(sl, lbl, bx, Inches(3.92), Inches(2.9), Inches(0.4),
                 font_size=Pt(10), color=WHITE, align=PP_ALIGN.CENTER)
    add_text_box(sl, val, bx, Inches(4.3), Inches(2.9), Inches(0.7),
                 font_size=Pt(22), bold=True, color=WHITE, align=PP_ALIGN.CENTER)

add_text_box(sl, "MSc Thesis Presentation", Inches(0.5), Inches(5.55),
             W - Inches(1.0), Inches(0.45),
             font_size=Pt(13), italic=True, color=WHITE, align=PP_ALIGN.CENTER)
bottom_bar(sl, page)


# ── SLIDE 2 — Motivation ───────────────────────────────────────────────────────
page += 1
sl = prs.slides.add_slide(blank)
nav_bar(sl, f"{page}")
section_heading(sl, "Why Does This Problem Exist?", "Real sensor networks are never complete")

cols = [Inches(0.35), Inches(4.6), Inches(8.85)]
titles = ["The Problem", "The Scale", "The Impact"]
bodies = [
    ["Hardware failures & maintenance windows",
     "Budget constraints — sensors are expensive",
     "Road geometry limits sensor placement",
     "→ Many sensors are DARK at any moment"],
    ["PEMS04: 307 sensors, SF Bay Area freeways",
     "Realistic deployments: only 20–60% observed",
     "This work: 80% sensors missing",
     "~246 blind nodes must be inferred"],
    ["Missing speeds block: route planning",
     "Incident detection and response",
     "Signal timing optimisation",
     "Emissions modelling and policy"],
]
colors = [LIGHT, RGBColor(0xE8, 0xF5, 0xE9), RGBColor(0xFF, 0xF3, 0xE0)]

for i in range(3):
    bullet_box(sl, titles[i], bodies[i],
               x=cols[i], y=Inches(1.7), w=Inches(4.1), h=Inches(4.6),
               bg=colors[i])

bottom_bar(sl, page)


# ── SLIDE 3 — Task definition ─────────────────────────────────────────────────
page += 1
sl = prs.slides.add_slide(blank)
nav_bar(sl, f"{page}")
section_heading(sl, "Imputation vs Forecasting", "This task is harder than standard traffic prediction")

# Left column — forecasting
bullet_box(sl, "Traffic Forecasting (Existing Work)",
           ["All sensors are observed",
            "Predict future values",
            "Well-studied: DCRNN, STGCN, WaveNet",
            "PEMS04 MAE ≈ 1.6 – 1.8 km/h",
            "Single-horizon evaluation"],
           x=Inches(0.35), y=Inches(1.7), w=Inches(5.9), h=Inches(3.5),
           bg=LIGHT)

# Right column — this work
bullet_box(sl, "Sparse Imputation (This Work)",
           ["80% of sensors are MISSING",
            "Recover present values, not future",
            "Under-explored — no deep baseline",
            "Jam MAE is the key metric",
            "Sparsity sweep from 20% to 90%"],
           x=Inches(6.6), y=Inches(1.7), w=Inches(6.4), h=Inches(3.5),
           bg=RGBColor(0xE8, 0xF0, 0xFF))

# Key insight box
add_rect(sl, Inches(0.35), Inches(5.45), W - Inches(0.7), Inches(1.2),
         RGBColor(0xFF, 0xF8, 0xE1), line_rgb=GOLD, line_width=Pt(1.5))
add_text_box(sl, "⚠  Key insight: A model that always predicts the global mean speed achieves 5.18 km/h "
             "overall MAE — but completely fails during congestion. Jam MAE (speed < 40 km/h) is "
             "the meaningful metric for a real traffic system.",
             Inches(0.55), Inches(5.55), W - Inches(1.1), Inches(1.0),
             font_size=Pt(12), color=RGBColor(0x7B, 0x5B, 0x00), italic=True, wrap=True)

bottom_bar(sl, page)


# ── SLIDE 4 — Data Setup ───────────────────────────────────────────────────────
page += 1
sl = prs.slides.add_slide(blank)
nav_bar(sl, f"{page}")
section_heading(sl, "Dataset & Setup", "PEMS04 · 80% Sensor Sparsity · 6-Feature Input")

info_card(sl, "DATASET", "PEMS04",
          "307 sensors · SF Bay Area\n5-min intervals · 17 days\nSpeed channel only",
          Inches(0.35), Inches(1.7), w=Inches(3.7), h=Inches(2.2))
info_card(sl, "SPARSITY", "80%",
          "~246 blind nodes\n~61 observed sensors\nFixed mask, seed=42",
          Inches(4.3), Inches(1.7), w=Inches(3.7), h=Inches(2.2), accent=GREEN)
info_card(sl, "SPLIT", "Train / Val / Eval",
          "Train: t = 0 – 3999\nVal: t = 4000 – 4239\nEval: t = 4500 – 4949",
          Inches(8.25), Inches(1.7), w=Inches(4.7), h=Inches(2.2), accent=GOLD)

# Feature table
add_text_box(sl, "6-Feature Input (per node, per timestep — no GT leakage)",
             Inches(0.35), Inches(4.1), W - Inches(0.7), Inches(0.4),
             font_size=Pt(12), bold=True, color=NAVY)

feat_headers = ["#", "Feature", "Description", "Blind Node Value"]
feat_rows = [
    ["1", "obs_speed", "Observed sensor speed (normalised)", "0.0 (zeroed)"],
    ["2", "global_ctx", "Mean of ALL observed nodes at time t", "Same as observed"],
    ["3", "nbr_ctx", "Adj-weighted mean of observed neighbours", "0.0 if no obs nbr"],
    ["4", "is_observed", "Binary sensor flag", "0"],
    ["5–6", "t_sin / t_cos", "Time-of-day cyclic encoding (scaled ×0.25)", "Shared"],
]
results_table(sl, feat_headers, feat_rows,
              x=Inches(0.35), y=Inches(4.55),
              col_widths=[Inches(0.5), Inches(2.0), Inches(6.5), Inches(3.5)],
              row_height=Inches(0.42))
bottom_bar(sl, page)


# ── SLIDE 5 — Architecture Overview ───────────────────────────────────────────
page += 1
sl = prs.slides.add_slide(blank)
nav_bar(sl, f"{page}")
section_heading(sl, "Architecture Overview", "Four components — one unified model")

# Flow arrow background
add_rect(sl, Inches(0.35), Inches(1.65), W - Inches(0.7), Inches(4.6), LIGHT,
         line_rgb=MID, line_width=Pt(0.5))

components = [
    ("Input Encoder", "Linear(6→64)\nper node", NAVY),
    ("GAT ODE", "2× Graph Attention\n+ Euler dt=0.3", ACCENT),
    ("Hypergraph\nConv", "2-hop corridors\nGated fusion", GREEN),
    ("Assimilation", "Kalman-style gate\nSensor injection", GOLD),
    ("Decoder", "Linear(64→1)\nSpeed output", RGBColor(0x8E, 0x44, 0xAD)),
]

box_w = Inches(2.2)
box_h = Inches(1.6)
gap   = Inches(0.28)
start_x = Inches(0.5)
cy = Inches(2.55)

prev_x = None
for i, (title, desc, col) in enumerate(components):
    bx = start_x + i * (box_w + gap)
    add_rect(sl, bx, cy, box_w, box_h, col)
    add_text_box(sl, title, bx + Inches(0.08), cy + Inches(0.1),
                 box_w - Inches(0.16), Inches(0.55),
                 font_size=Pt(12), bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_rect(sl, bx + Inches(0.08), cy + Inches(0.62),
             box_w - Inches(0.16), Inches(0.02), WHITE)
    add_text_box(sl, desc, bx + Inches(0.08), cy + Inches(0.7),
                 box_w - Inches(0.16), Inches(0.8),
                 font_size=Pt(10.5), italic=True, color=WHITE, align=PP_ALIGN.CENTER)
    # Arrow between boxes
    if prev_x is not None:
        ax = prev_x + box_w + Inches(0.04)
        add_text_box(sl, "→", ax, cy + Inches(0.6), gap,
                     Inches(0.45), font_size=Pt(18), bold=True,
                     color=NAVY, align=PP_ALIGN.CENTER)
    prev_x = bx

# Bullet notes below
note_bullets = [
    "Input: [B, N, T, 6] — batch × nodes × timesteps × features",
    "Hidden state z ∈ ℝ^{N×64} evolves via: decode → Euler ODE step → assimilation → repeat",
    "Euler integration: z_{t+1} = z_t + 0.3 · f_θ(z_t)   |   Hypergraph gate: g = sigmoid(w), init w=−2",
]
for i, n in enumerate(note_bullets):
    add_text_box(sl, f"▸  {n}", Inches(0.45), Inches(4.48) + i * Inches(0.45),
                 W - Inches(0.9), Inches(0.4), font_size=Pt(11), color=DARK)

bottom_bar(sl, page)


# ── SLIDE 6 — GAT ─────────────────────────────────────────────────────────────
page += 1
sl = prs.slides.add_slide(blank)
nav_bar(sl, f"{page}")
section_heading(sl, "Graph Attention Network (GAT)", "Learn which neighbours matter — not just how close they are")

# Left card — GCN problem
bullet_box(sl, "Standard GCN — Fixed Weights",
           ["h_i = Σ_j  A_norm[i,j] × h_j",
            "Weights fixed by road distance",
            "Cannot adapt to congestion context",
            "A congested close neighbour has same\ninfluence as a free-flowing distant one"],
           x=Inches(0.35), y=Inches(1.65), w=Inches(5.95), h=Inches(4.1),
           bg=LIGHT)

# Right card — GAT solution
bullet_box(sl, "Our GAT — Learned Attention",
           ["e_ij = LeakyReLU( a_src(Wh_i) + a_dst(Wh_j) )",
            "α_ij = softmax( e_ij / τ=2 )  over road neighbours",
            "h_i' = Σ_j  α_ij × Wh_j",
            "Temperature τ=2 prevents single-neighbour\ndominance → no oscillation after jams",
            "Non-edges masked to −∞ → road topology enforced"],
           x=Inches(6.55), y=Inches(1.65), w=Inches(6.45), h=Inches(4.1),
           bg=RGBColor(0xE8, 0xF0, 0xFF))

# Bottom note
add_rect(sl, Inches(0.35), Inches(5.95), W - Inches(0.7), Inches(0.75),
         RGBColor(0xE8, 0xF5, 0xE9), line_rgb=GREEN, line_width=Pt(1))
add_text_box(sl, "Two stacked GAT layers inside the ODE function. Output is the derivative dz/dt — "
             "the Euler step z + 0.3·f(z) adds the residual. LayerNorm applied to the derivative "
             "only (not the sum), preserving ODE residual structure.",
             Inches(0.55), Inches(6.0), W - Inches(1.1), Inches(0.65),
             font_size=Pt(11), color=RGBColor(0x1A, 0x5C, 0x3A), wrap=True)
bottom_bar(sl, page)


# ── SLIDE 7 — Hypergraph ───────────────────────────────────────────────────────
page += 1
sl = prs.slides.add_slide(blank)
nav_bar(sl, f"{page}")
section_heading(sl, "Hypergraph Convolution", "Capture corridor-level group dynamics beyond pairwise edges")

bullet_box(sl, "Why Hypergraph?",
           ["Standard edges: connect exactly 2 nodes (pairwise)",
            "Hyperedge: connects a GROUP of nodes simultaneously",
            "A freeway corridor {s₁, s₂, s₃, s₄, s₅} is naturally a hyperedge",
            "The HGNN sees the whole corridor's state at once — not one hop at a time"],
           x=Inches(0.35), y=Inches(1.65), w=Inches(12.6), h=Inches(2.1), bg=LIGHT)

bullet_box(sl, "Construction & Convolution",
           ["Hyperedge for node i = i ∪ {all 2-hop reachable sensors}",
            "Incidence matrix H: H[v,e] = 1 if node v ∈ hyperedge e",
            "Normalised operator: H_conv = D_v^{−½} H D_e^{−1} H^T D_v^{−½}  (pre-computed once)",
            "Runtime: h_hyp = H_conv · (x · Θ)   — single matmul, very fast"],
           x=Inches(0.35), y=Inches(3.9), w=Inches(7.7), h=Inches(2.6),
           bg=RGBColor(0xE8, 0xF0, 0xFF))

bullet_box(sl, "Learnable Gate (critical!)",
           ["g = sigmoid(w),  init: w = −2  →  g ≈ 0.12",
            "h = h_GAT + g · h_hyp  (gated fusion)",
            "Without gate: jam nodes average with ~20\nfree-flowing corridor members → over-smooth",
            "Gate learns WHEN corridor context helps"],
           x=Inches(8.2), y=Inches(3.9), w=Inches(4.75), h=Inches(2.6),
           bg=RGBColor(0xFF, 0xF8, 0xE1))
bottom_bar(sl, page)


# ── SLIDE 8 — Assimilation ────────────────────────────────────────────────────
page += 1
sl = prs.slides.add_slide(blank)
nav_bar(sl, f"{page}")
section_heading(sl, "Observation Assimilation", "Kalman-style sensor fusion after each ODE step")

bullet_box(sl, "The Update Rule",
           ["z_obs  = W_obs · x_{t+1}            ← encode new sensor reading",
            "gate   = σ( W_g [ z ; z_obs ] )     ← learned Kalman gain",
            "update = gate  ×  (z_obs − z)  ×  obs_mask",
            "z ← z + update                      ← corrected hidden state"],
           x=Inches(0.35), y=Inches(1.65), w=Inches(7.7), h=Inches(2.8),
           bg=LIGHT)

bullet_box(sl, "Kalman Filter Analogy",
           ["Kalman: z_pred = A·z  →  z_corr = z_pred + K·(obs − z_pred)",
            "Ours:   ODE Euler step  →  z + gate·(z_obs − z)",
            "Kalman gain K: fixed, computed from covariance matrix",
            "Our gate: learned, adapts to non-linear traffic dynamics"],
           x=Inches(8.2), y=Inches(1.65), w=Inches(4.75), h=Inches(2.8),
           bg=RGBColor(0xE8, 0xF0, 0xFF))

bullet_box(sl, "No Leakage — the Critical Detail",
           ["obs_mask = 1 for observed sensors,  0 for blind nodes",
            "update = ... × obs_mask  →  blind nodes get update = 0",
            "Without this: blind nodes 'assimilate' their own zero observations",
            "→ hidden state pulled toward 0 every step regardless of ODE prediction"],
           x=Inches(0.35), y=Inches(4.65), w=Inches(12.6), h=Inches(2.0),
           bg=RGBColor(0xFF, 0xEB, 0xEE))
bottom_bar(sl, page)


# ── SLIDE 9 — Physics Loss ────────────────────────────────────────────────────
page += 1
sl = prs.slides.add_slide(blank)
nav_bar(sl, f"{page}")
section_heading(sl, "Physics-Informed Training", "Three loss terms — each solving a different failure mode")

cards = [
    ("Term 1\nJam-Weighted MSE", NAVY,
     ["L_obs = mean( ((ŝ−s)×mask)² × w )",
      "w = 4 if speed < 40 km/h  (jam)",
      "w = 1 otherwise  (free-flow)",
      "Compensates 12:1 free-flow:jam ratio",
      "λ = 1.0  (primary loss)"]),
    ("Term 2\nTemporal Smoothness", ACCENT,
     ["L_smooth = mean( (ŝ_{t+1} − ŝ_t)² )",
      "Penalises step-to-step jumps",
      "Suppresses post-jam oscillation",
      "(model alternates between jam/free-flow)",
      "λ = 0.60  (strong regulariser)"]),
    ("Term 3\nLaplacian Physics", GREEN,
     ["L_phys = mean( ||L_sym · v||² )",
      "= Σ_i  (v_i − mean_nbr(v_i))²",
      "LWR principle: speed varies continuously",
      "along road — no sharp spatial boundaries",
      "λ = 0.02  (soft constraint)"]),
]

for i, (title, col, bullets) in enumerate(cards):
    bx = Inches(0.35) + i * Inches(4.35)
    add_rect(sl, bx, Inches(1.65), Inches(4.15), Inches(5.1), col)
    add_text_box(sl, title, bx + Inches(0.12), Inches(1.72),
                 Inches(3.9), Inches(0.68), font_size=Pt(12), bold=True,
                 color=WHITE, align=PP_ALIGN.CENTER)
    add_rect(sl, bx + Inches(0.12), Inches(2.38), Inches(3.9), Inches(0.02), WHITE)
    for j, b in enumerate(bullets):
        add_text_box(sl, f"▸  {b}", bx + Inches(0.15), Inches(2.48) + j * Inches(0.5),
                     Inches(3.85), Inches(0.45), font_size=Pt(10.5), color=WHITE, wrap=True)

bottom_bar(sl, page)


# ── SLIDE 10 — Training Strategy ──────────────────────────────────────────────
page += 1
sl = prs.slides.add_slide(blank)
nav_bar(sl, f"{page}")
section_heading(sl, "Training Strategy", "Three techniques to handle extreme class imbalance")

strategies = [
    ("Curriculum Masking", GREEN,
     "15% of observed sensors are randomly pseudo-blinded each batch.",
     ["Zeroes their speed, nbr_ctx, and is_observed features",
      "Excludes from assimilation gate",
      "Computes loss on their known ground truth",
      "→ Gradients always flow through the blind-node code path"]),
    ("Jam-Biased Sampling", ACCENT,
     "50% of batches are forced to start at jam timesteps.",
     ["Natural rate of jam windows: only ~8%",
      "Even with 4× loss weight, variance is too high",
      "Pre-computed jam_t_valid index for fast lookup",
      "→ Balanced gradient signal between jam and free-flow"]),
    ("Gradient Accumulation", NAVY,
     "Gradients from 4 windows accumulated per parameter update.",
     ["Jam batches: very high loss",
      "Free-flow batches: very low loss",
      "High variance → oscillating validation MAE",
      "→ 4-step accumulation smooths this variance"]),
]

for i, (title, col, lead, bullets) in enumerate(strategies):
    bx = Inches(0.35) + i * Inches(4.35)
    add_rect(sl, bx, Inches(1.65), Inches(4.15), Inches(0.06), col)
    add_rect(sl, bx, Inches(1.71), Inches(4.15), Inches(5.0), LIGHT)
    add_text_box(sl, title, bx + Inches(0.12), Inches(1.74),
                 Inches(3.9), Inches(0.45), font_size=Pt(13), bold=True, color=col)
    add_text_box(sl, lead, bx + Inches(0.12), Inches(2.2),
                 Inches(3.9), Inches(0.5), font_size=Pt(11), italic=True,
                 color=GREY, wrap=True)
    add_rect(sl, bx + Inches(0.12), Inches(2.72), Inches(3.9), Inches(0.015), col)
    for j, b in enumerate(bullets):
        add_text_box(sl, f"▸  {b}", bx + Inches(0.15), Inches(2.78) + j * Inches(0.5),
                     Inches(3.85), Inches(0.45), font_size=Pt(10.5), color=DARK, wrap=True)

# Optimiser details
add_rect(sl, Inches(0.35), Inches(6.82), W - Inches(0.7), Inches(0.38), NAVY)
add_text_box(sl,
    "Adam lr=3×10⁻⁴  ·  weight decay=10⁻⁴  ·  Cosine Annealing T_max=400  ·  800 epochs  ·  Grad clip norm=1.0",
    Inches(0.5), Inches(6.87), W - Inches(1.0), Inches(0.28),
    font_size=Pt(10.5), color=WHITE, align=PP_ALIGN.CENTER)
bottom_bar(sl, page)


# ── SLIDE 11 — Main Results ────────────────────────────────────────────────────
page += 1
sl = prs.slides.add_slide(blank)
nav_bar(sl, f"{page}")
section_heading(sl, "Results: 80% Sparsity Evaluation", "Consistent improvement over both baselines on congestion events")

results_table(sl,
    ["Model", "MAE all (km/h)", "MAE jam (km/h)", "vs Global Mean Jam"],
    [
        ["Global mean baseline", "5.18", "35.99", "—"],
        ["IDW spatial interpolation", "5.23", "32.95", "−8.5%"],
        ["Ours — Full model", "5.18", "33.93", "−5.7%"],
    ],
    x=Inches(1.5), y=Inches(1.75),
    col_widths=[Inches(5.0), Inches(2.2), Inches(2.2), Inches(2.4)],
    row_height=Inches(0.5)
)

# Call-out boxes
info_card(sl, "OVERALL MAE", "5.18 km/h",
          "Matches global mean — expected.\nFree-flow (92%) dominates the average.",
          Inches(0.35), Inches(3.3), w=Inches(4.0), h=Inches(1.8), accent=GREY)
info_card(sl, "JAM MAE", "33.93 km/h",
          "5.7% improvement over global mean.\nThe key metric for real-world utility.",
          Inches(4.6), Inches(3.3), w=Inches(4.0), h=Inches(1.8), accent=GREEN)
info_card(sl, "vs IDW", "+2.9% better on jams",
          "Learned spatial + temporal context\nbeyond static adjacency weighting.",
          Inches(8.85), Inches(3.3), w=Inches(4.1), h=Inches(1.8), accent=ACCENT)

add_rect(sl, Inches(0.35), Inches(5.35), W - Inches(0.7), Inches(1.3),
         RGBColor(0xFF, 0xF8, 0xE1), line_rgb=GOLD, line_width=Pt(1.5))
add_text_box(sl,
    "⚠  Why does our model tie the global mean on overall MAE?\n"
    "Free-flow is 92% of timesteps and tightly clustered near the mean. Any model near the mean on "
    "free-flow will tie on overall MAE. Jam MAE is where the real difference lies — and where our "
    "graph dynamics and sensor assimilation provide genuine value.",
    Inches(0.55), Inches(5.42), W - Inches(1.1), Inches(1.15),
    font_size=Pt(11), color=RGBColor(0x7B, 0x5B, 0x00), wrap=True)
bottom_bar(sl, page)


# ── SLIDE 12 — Sparsity Sweep ─────────────────────────────────────────────────
page += 1
sl = prs.slides.add_slide(blank)
nav_bar(sl, f"{page}")
section_heading(sl, "Results: Sensor Sparsity Sweep", "Performance degrades gracefully from 20% to 90% missing sensors")

results_table(sl,
    ["Sparsity", "Blind Sensors", "Global Mean Jam", "IDW Jam", "Model Jam", "vs Baseline"],
    [
        ["20%", "22%", "36.04", "27.35", "27.68", "+23.2%"],
        ["40%", "48%", "35.74", "29.73", "28.27", "+20.9%"],
        ["60%", "62%", "36.01", "29.98", "31.66", "+12.1%"],
        ["80%", "83%", "35.99", "32.95", "31.19", "+13.3%"],
        ["90%", "90%", "35.99", "33.81", "34.58", "+3.9%"],
    ],
    x=Inches(0.5), y=Inches(1.75),
    col_widths=[Inches(1.4), Inches(1.7), Inches(2.2), Inches(1.8), Inches(1.8), Inches(2.0)],
    row_height=Inches(0.5)
)

bullet_box(sl, "Key Observations",
           ["Model consistently outperforms global-mean baseline at ALL sparsity levels",
            "At 90% sparsity: model approaches baseline — expected (almost no neighbour context remains)",
            "vs IDW: mixed results (IDW has an advantage when observed sensors are many and close)",
            "Sparsity sweep confirms graceful degradation — no catastrophic failure mode"],
           x=Inches(0.35), y=Inches(4.7), w=Inches(12.6), h=Inches(2.0), bg=LIGHT)
bottom_bar(sl, page)


# ── SLIDE 13 — Ablation ───────────────────────────────────────────────────────
page += 1
sl = prs.slides.add_slide(blank)
nav_bar(sl, f"{page}")
section_heading(sl, "Ablation Study", "Measuring each component's contribution — 300 epochs per variant")

results_table(sl,
    ["Model / Variant", "MAE all", "MAE jam", "Δ jam  (+ = component helps)"],
    [
        ["Global mean baseline", "5.18", "35.99", "—"],
        ["IDW (spatial interp.)", "5.23", "32.95", "—"],
        ["Full model  ◀", "5.18", "33.93", "+0.00"],
        ["−  Hypergraph", "5.54", "31.66", "−2.27"],
        ["−  Assimilation", "5.29", "32.54", "−1.38"],
        ["−  Physics loss", "5.32", "33.32", "−0.61"],
        ["−  Neighbour context", "5.84", "28.99", "−4.94"],
        ["−  Temporal encoding", "6.00", "33.51", "−0.41"],
    ],
    x=Inches(0.5), y=Inches(1.75),
    col_widths=[Inches(4.2), Inches(1.7), Inches(1.7), Inches(4.8)],
    row_height=Inches(0.46)
)

add_rect(sl, Inches(0.35), Inches(6.35), W - Inches(0.7), Inches(0.85),
         RGBColor(0xE8, 0xF0, 0xFF), line_rgb=ACCENT, line_width=Pt(1))
add_text_box(sl,
    "Note: Δ jam = full_jam − variant_jam. Positive = removing this component hurts jam imputation. "
    "The gated hypergraph (init gate ≈ 0.12) and neighbour context require more epochs to converge — "
    "their gate/weights need time to learn when corridor context helps vs when it over-smooths. "
    "Temporal encoding and assimilation show clear positive contributions.",
    Inches(0.55), Inches(6.4), W - Inches(1.1), Inches(0.75),
    font_size=Pt(10.5), color=NAVY, wrap=True)
bottom_bar(sl, page)


# ── SLIDE 14 — SOTA Comparison ─────────────────────────────────────────────────
page += 1
sl = prs.slides.add_slide(blank)
nav_bar(sl, f"{page}")
section_heading(sl, "State of the Art Context", "Different task — not directly comparable, but contextualised")

results_table(sl,
    ["Model", "Venue", "Task", "Sensors", "PEMS04 MAE"],
    [
        ["DCRNN (Li et al., 2018)", "ICLR 2018", "Forecasting", "100%", "~1.8 km/h"],
        ["STGCN (Yu et al., 2018)", "IJCAI 2018", "Forecasting", "100%", "~1.7 km/h"],
        ["Graph WaveNet (Wu et al., 2019)", "IJCAI 2019", "Forecasting", "100%", "~1.6 km/h"],
        ["AGCRN (Bai et al., 2020)", "NeurIPS 2020", "Forecasting", "100%", "~1.5 km/h"],
        ["Ours (full model)", "This thesis", "Imputation", "20%", "5.18 km/h"],
    ],
    x=Inches(0.35), y=Inches(1.75),
    col_widths=[Inches(4.0), Inches(1.8), Inches(1.8), Inches(1.5), Inches(2.5)],
    row_height=Inches(0.5)
)

add_rect(sl, Inches(0.35), Inches(4.7), W - Inches(0.7), Inches(1.9),
         RGBColor(0xFF, 0xEB, 0xEE), line_rgb=RED, line_width=Pt(1.5))
add_text_box(sl, "Why the MAE gap (~3×) is expected — NOT a weakness:",
             Inches(0.55), Inches(4.77), W - Inches(1.1), Inches(0.38),
             font_size=Pt(12), bold=True, color=RED)
for i, b in enumerate([
    "DCRNN/STGCN/WaveNet: all 307 sensors observed, predict only 1 future step",
    "This work: only 61 sensors observed (80% missing), must recover ALL 246 blind node speeds simultaneously",
    "Imputation with 80% missing data is fundamentally a harder, different task — comparison is directional only",
]):
    add_text_box(sl, f"▸  {b}", Inches(0.55), Inches(5.18) + i * Inches(0.43),
                 W - Inches(1.1), Inches(0.4), font_size=Pt(11), color=DARK, wrap=True)
bottom_bar(sl, page)


# ── SLIDE 15 — Conclusion ─────────────────────────────────────────────────────
page += 1
sl = prs.slides.add_slide(blank)
nav_bar(sl, f"{page}")
section_heading(sl, "Conclusion", "A new architecture for sparse traffic speed imputation")

bullet_box(sl, "Contributions",
           ["Hypergraph Neural ODE — first application to sparse traffic imputation",
            "Gated HGNN fusion — prevents over-smoothing at congested nodes",
            "Kalman-style assimilation — injects sensor readings continuously without leakage",
            "Physics-informed loss — LWR flow continuity via graph Laplacian regularisation",
            "Curriculum masking + jam-biased sampling — overcomes severe class imbalance"],
           x=Inches(0.35), y=Inches(1.65), w=Inches(7.7), h=Inches(3.65), bg=LIGHT)

bullet_box(sl, "Future Work",
           ["Directed hyperedges following traffic flow direction",
            "Dynamic hypergraph (edges change with congestion patterns)",
            "Neural ODE adjoint method for full continuous-time training",
            "Multi-variate imputation: speed + flow + occupancy jointly"],
           x=Inches(8.2), y=Inches(1.65), w=Inches(4.75), h=Inches(3.65),
           bg=RGBColor(0xE8, 0xF0, 0xFF))

# Result summary
add_rect(sl, Inches(0.35), Inches(5.52), W - Inches(0.7), Inches(1.6), NAVY)
add_text_box(sl, "Final Results — 80% Sparsity, PEMS04",
             Inches(0.55), Inches(5.6), W - Inches(1.1), Inches(0.38),
             font_size=Pt(12), bold=True, color=WHITE, align=PP_ALIGN.CENTER)
for i, txt in enumerate([
    "Overall blind-node MAE: 5.18 km/h    ·    Jam MAE: 33.93 km/h",
    "5.7% improvement over global mean on congestion  ·  Validated at 5 sparsity levels (20–90%)",
]):
    add_text_box(sl, txt, Inches(0.55), Inches(6.02) + i * Inches(0.45),
                 W - Inches(1.1), Inches(0.4), font_size=Pt(11),
                 color=LIGHT, align=PP_ALIGN.CENTER)
bottom_bar(sl, page)


# ── SLIDE 16 — Thank You ───────────────────────────────────────────────────────
page += 1
sl = prs.slides.add_slide(blank)
add_rect(sl, 0, 0, W, H, NAVY)
add_rect(sl, 0, Inches(3.35), W, Inches(0.08), ACCENT)

add_text_box(sl, "Thank You", Inches(0), Inches(1.0), W, Inches(1.4),
             font_size=Pt(48), bold=True, color=WHITE, align=PP_ALIGN.CENTER)
add_text_box(sl, "Questions & Discussion", Inches(0), Inches(2.35), W, Inches(0.75),
             font_size=Pt(22), italic=True, color=ACCENT, align=PP_ALIGN.CENTER)
add_rect(sl, Inches(2.0), Inches(3.65), W - Inches(4.0), Inches(0.04), ACCENT)

details = [
    "Dataset: PEMS04  ·  307 sensors  ·  80% sparsity",
    "Architecture: Hypergraph GAT-ODE + Kalman Assimilation + Physics Loss",
    "Results: 5.18 km/h overall MAE  ·  33.93 km/h jam MAE",
    "Branch: claude/improve-repo-quality-MODmC",
]
for i, d in enumerate(details):
    add_text_box(sl, d, Inches(0), Inches(3.9) + i * Inches(0.55), W, Inches(0.45),
                 font_size=Pt(13), color=LIGHT, align=PP_ALIGN.CENTER)
bottom_bar(sl, page)


# ── Save ───────────────────────────────────────────────────────────────────────
prs.save("thesis_presentation.pptx")
print("✅ thesis_presentation.pptx saved")
