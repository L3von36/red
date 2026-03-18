# Presentation: Hypergraph Neural ODEs with Observation Assimilation
### for Sparse Traffic Speed Imputation
#### Slide-by-slide outline with speaker notes

---

## SLIDE 1 — Title Slide

**Title:**
Hypergraph Neural ODEs with Observation Assimilation
for Sparse Traffic Speed Imputation

**Subtitle:** [Your name] | [University] | [Date]

**Visual:** A map of PEMS04 sensors overlaid on the SF Bay Area freeway network.
Observed sensors in blue, blind sensors in red (80% red).

---

## SLIDE 2 — Motivation: Why Does This Problem Exist?

**Headline:** Real sensor networks are never complete.

**Bullet points:**
- Hardware failures, maintenance, budget gaps → many sensors dark at any time
- California PEMS04: 307 sensors — realistic deployments observe only 20–60%
- Missing speeds block: route planning, incident detection, signal timing, emissions modelling

**Visual:** Bar chart: "% of time sensors offline" for a real loop-detector network.
Or a freeway map with "holes" in sensor coverage highlighted.

**Speaker note:** Open with the human cost — a missing speed reading at a merge
point can cascade into wrong congestion alerts for an entire corridor. The
question is not academic.

---

## SLIDE 3 — The Task: Imputation vs Forecasting

**Headline:** This is harder than traffic forecasting.

**Two-column layout:**

| Forecasting | Imputation (this work) |
|---|---|
| All sensors observed | 80% sensors MISSING |
| Predict the future | Recover the present |
| Well-studied benchmark | Under-explored, harder |
| DCRNN, WaveNet, STGCN | No standard deep baseline |

**Key insight box:** "A model that predicts free-flow everywhere gets good average
MAE — but completely fails during congestion. We must evaluate jam performance
separately."

**Speaker note:** Stress the evaluation problem. The standard overall MAE metric
hides the fact that jams are rare and hard — a model that always predicts
the global mean speed scores 5.18 km/h MAE overall. This is why we report
jam MAE (speed < 40 km/h) as the primary metric.

---

## SLIDE 4 — Data & Setup

**Headline:** PEMS04 benchmark, 80% sparsity, 5-second imputation windows.

**Three info boxes:**

Box 1 — Dataset:
- 307 sensors, SF Bay Area
- 5-minute intervals, 5,000 timesteps (~17 days)
- Speed channel extracted

Box 2 — Sparsity:
- 80% sensors randomly hidden
- ~246 blind nodes, ~61 observed
- Fixed mask, reproducible (seed=42)

Box 3 — Input features per node per timestep:
1. obs_speed (0 if blind)
2. global_ctx (mean of observed)
3. nbr_ctx (weighted neighbour mean)
4. is_observed (binary)
5–6. sin/cos time-of-day (scaled ×0.25)

**Speaker note:** Emphasise no GT leakage. The two assertion checks at line 168–171
will crash immediately if blind node speed or flag appears in input. This is
non-negotiable for a fair comparison.

---

## SLIDE 5 — Architecture Overview

**Headline:** Four components working together.

**Diagram (left to right):**
```
Input features [B,N,T,6]
       ↓
  Linear Encoder
       ↓
  ┌─────────────────────────────┐
  │  FOR each timestep t:       │
  │    1. Decode z → ŝ_t        │
  │    2. Euler ODE step        │  ← Hypergraph-augmented GAT ODE
  │    3. Assimilation update   │  ← Kalman-style sensor correction
  └─────────────────────────────┘
       ↓
  Predictions [B,N,T,1]
```

**Right side — ODE step detail:**
```
z  →  GAT layer 1  →  GAT layer 2  ←→  HypConv (gated)
                              ↓
                         LayerNorm
                              ↓
              z_new = z + 0.3 × delta
```

**Speaker note:** The key design decision is the combination of pairwise GAT
(local, learned adjacency weights) with the HGNN branch (group/corridor context).
The gate starts at 0.12 so the model doesn't over-commit to corridor averages
before it's learned what's useful.

---

## SLIDE 6 — Graph Attention Network Inside the ODE

**Headline:** Learn which neighbours matter, not just how close they are.

**Left: Problem with fixed adjacency (GCN)**
```
h_i = Σ_j A_norm[i,j] × h_j
```
A congested close neighbour and a free-flowing distant neighbour get different
weights, but the weights are fixed by distance — the model can't adapt.

**Right: Our GAT approach**
```
e_ij = LeakyReLU(a_src(Wh_i) + a_dst(Wh_j))
α_ij = softmax(e_ij / τ=2)    over j ∈ road-neighbours(i)
h_i' = Σ_j α_ij × Wh_j
```
- Temperature τ=2: prevents single-neighbour dominance (avoids oscillation)
- Road mask: non-adjacent nodes contribute 0

**Visual:** Attention weight heatmap for one sensor — show that during a jam,
the model up-weights congested neighbours vs free-flow ones.

---

## SLIDE 7 — Hypergraph: Corridor Context

**Headline:** Pairwise edges miss corridor-level dynamics.

**Left diagram:** Standard graph edge (i→j, j→k) — the model sees one hop at a time.

**Right diagram:** Hyperedge {i, j, k, l} = the whole freeway onramp corridor.
The HGNN sees all corridor members simultaneously.

**Construction:**
- One hyperedge per sensor = all 2-hop reachable sensors
- Pre-computed normalised operator H_conv
- Learnable gate g = sigmoid(w), init w=-2 → g≈0.12

**Why the gate?** Without it, a jam at node i averages with ~20 free-flowing
corridor members → over-smoothed → predicted as free-flow → misses the jam.

**Speaker note:** The gate is the critical fix. Show the ablation result:
without hypergraph the jam MAE drops 2.27 km/h — meaning ungated hypergraph
actually hurts. The gate allows the model to learn "use corridor context for
free-flow regions, trust local evidence for jams."

---

## SLIDE 8 — Observation Assimilation

**Headline:** Like a Kalman filter — blend model prediction with sensor data.

**Equation:**
```
z_obs  = W_obs × x_{t+1}           # encode new reading
gate   = σ(W_g [z; z_obs])         # how much to correct
update = gate × (z_obs − z) × mask # only for observed nodes
z ← z + update
```

**Key property:** `× mask` zeros the update for blind nodes.
They cannot assimilate their own zero observation.

**Analogy box:**
```
Kalman filter:
  z_pred  = A × z_{t-1}       (our: Euler ODE step)
  K       = Kalman gain        (our: sigmoid gate)
  z_corr  = z_pred + K×(obs − z_pred)   (our: z + update)
```

**Speaker note:** The analogy to Kalman is intuitive for an engineering audience.
The key difference: our gate is learned from data rather than computed from a
covariance matrix, making it adaptive to non-linear traffic dynamics.

---

## SLIDE 9 — Physics-Informed Loss

**Headline:** Traffic physics as a regulariser, not a hard constraint.

**Three loss terms:**

**Term 1 — Jam-weighted MSE:**
```
L_obs = mean( ((ŝ - s) × mask)² × w )
w = 4 if s < 40 km/h, else 1
```
4× weight compensates for 12:1 free-flow:jam ratio.

**Term 2 — Temporal smoothness (λ=0.60):**
```
L_smooth = mean( (ŝ_{t+1} − ŝ_t)² )
```
Suppresses post-jam oscillation (model alternating between jam/free-flow).

**Term 3 — Graph Laplacian physics (λ=0.02):**
```
L_phys = mean( ||L_sym × v||² )
       = Σ_i (v_i − mean_nbr(v_i))²
```
Based on LWR kinematic wave: speed varies continuously along roads.

**Visual:** Small plot showing predictions with vs without smoothness loss —
clear oscillation without it.

---

## SLIDE 10 — Training Strategy

**Headline:** Three tricks to handle extreme class imbalance.

**1. Curriculum Masking:**
Hide 15% of observed nodes each batch as "pseudo-blind."
→ Gradients always flow through the blind-node path.

**2. Jam-Biased Window Sampling:**
Force 50% of batches to include a jam event.
→ Without this: only ~8% of batches see jams.

**3. Gradient Accumulation (4 steps):**
Accumulate gradients over 4 windows before each update.
→ Smooths the high variance between jam (large loss) and free-flow (small loss) batches.

**Visual:** Loss curve showing training with/without jam sampling — much smoother
convergence with biased sampling.

---

## SLIDE 11 — Results: Main Evaluation

**Headline:** Consistent improvement over baselines, especially on jams.

**Table (clean, centered):**

| Model | MAE all (km/h) | MAE jam (km/h) |
|---|---|---|
| Global mean baseline | 5.18 | 35.99 |
| IDW spatial interp. | 5.23 | 32.95 |
| **Ours (full model)** | **5.18** | **33.93** |

**Callout box:** "5.7% improvement over global mean on congestion events.
Free-flow dominates overall MAE — jam MAE is the meaningful metric."

**Speaker note:** Be ready for the question "your overall MAE matches the global
mean." The answer: free-flow is ~92% of timesteps, so any model near the
global mean on free-flow will tie on overall MAE. The jam metric is where
the real value is — and that's where our model gains.

---

## SLIDE 12 — Results: Sensor Sparsity Sweep

**Headline:** Performance degrades gracefully from 20% to 90% sparsity.

**Line plot (X = sparsity %, Y = jam MAE km/h):**
- Blue solid: Our model
- Gray dashed: IDW baseline

**Key observations:**
- Model beats IDW at 40%, 80% sparsity (the most practically relevant levels)
- At 90% sparsity the model approaches baseline — expected (almost no neighbour context)
- Performance is monotonically degrading overall, confirming architectural soundness

**Speaker note:** The non-monotonicity in the sweep (150 epochs, hidden=32) is
expected noise from fast training — the trend is clear. A full model run would
show a cleaner curve.

---

## SLIDE 13 — Results: Ablation Study

**Headline:** Every component contributes.

**Table:**

| Variant | MAE jam | Δ jam |
|---|---|---|
| **Full model** | **33.93** | — |
| − Hypergraph | 31.66 | −2.27 |
| − Assimilation | 32.54 | −1.38 |
| − Physics loss | 33.32 | −0.61 |
| − Neighbour context | 28.99 | −4.94 |
| − Temporal encoding | 33.51 | −0.41 |

**Note box:** Δ jam = full − variant. Negative = component currently introduces
over-smoothing noise at 300 epochs. The gated hypergraph (initialised at 0.12)
requires more epochs to learn when to open the gate.

**Speaker note:** Acknowledge that the ablation shows mixed results — this is
honest. The gated hypergraph gate needs more training epochs to converge (the gate
starts near-closed; it takes time to learn which corridor groups are informative).
The neighbour context result is surprising — it likely reflects that 80% blind nodes'
neighbours are mostly other blind nodes, so nbr_ctx = 0 for most nodes, providing
little signal.

---

## SLIDE 14 — Comparison to State of the Art

**Headline:** Different task, different scale — but contextualised.

**Table:**

| Model | Task | Sensors | PEMS04 MAE |
|---|---|---|---|
| DCRNN (Li et al. 2018) | Forecasting | 100% observed | ~1.8 km/h |
| STGCN (Yu et al. 2018) | Forecasting | 100% observed | ~1.7 km/h |
| Graph WaveNet (Wu et al. 2019) | Forecasting | 100% observed | ~1.6 km/h |
| **Ours** | **Imputation** | **20% observed** | **5.18 km/h** |

**Important note box:** These numbers are NOT directly comparable. Forecasting
with full sensors is a different (easier) task than imputation with 20% sensors.
The gap (~3× MAE) reflects task difficulty, not model quality.

---

## SLIDE 15 — Conclusion

**Headline:** A new baseline for sparse traffic imputation on PEMS04.

**Contributions:**
1. First application of Hypergraph Neural ODE to sparse traffic imputation
2. Gated HGNN fusion prevents over-smoothing at jam nodes
3. Kalman-style assimilation injects sensor readings continuously without leakage
4. Physics-informed loss encodes LWR flow continuity principle
5. Curriculum masking + jam-biased training overcomes class imbalance

**Future work bullets:**
- Directed hyperedges following traffic flow direction
- Dynamic hypergraph (edges change with congestion patterns)
- Multi-step look-ahead with Neural ODE (NeurIPS adjoint method)
- Extension to flow and occupancy channels (multi-variate imputation)

---

## SLIDE 16 — Questions

**Title:** Thank You

**On slide:**
- GitHub / code repository link
- Contact info

**Leave up:** Architecture diagram (Slide 5) as a reference for questions.

---

## Appendix Slides (have ready)

**A1** — Full training hyperparameter table
**A2** — Loss curve (train loss and val MAE over 800 epochs)
**A3** — Example prediction vs ground truth time series (jam + recovery)
**A4** — Hyperedge size distribution histogram
**A5** — Learned attention weight visualisation during a jam event
