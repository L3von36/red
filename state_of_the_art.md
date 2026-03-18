# State of the Art: Traffic Speed Imputation and Graph Neural Networks

---

## 1. Problem Definition

**Traffic speed imputation** is the task of estimating the speed at road sensors
whose readings are missing or unavailable, given partial observations from
neighbouring sensors and the structure of the road network.

This differs from **traffic forecasting**, where all sensors are observed and
the goal is to predict future values. Imputation is strictly harder: the model
must simultaneously reason about *space* (what does an unobserved node's speed
look like given its neighbours?) and *time* (how does traffic state evolve
between observations?).

---

## 2. Classical Methods

### 2.1 Global Mean / Median Imputation
Replace missing values with the dataset-wide mean speed.
- **MAE on PEMS04 (jam nodes)**: ~36 km/h
- **Weakness**: completely ignores spatial structure and temporal dynamics.
- **Used in this work as Baseline 1.**

### 2.2 Inverse Distance Weighting (IDW)
Estimate missing node speed as a weighted average of observed neighbours,
weights proportional to 1/distance.
- Equivalent to: `pred_i = Σ_j A[i,j] * obs_j / Σ_j A[i,j] * mask_j`
- **Weakness**: purely spatial, no temporal modelling, no learning.
- **Used in this work as Baseline 2 (IDW).**

### 2.3 Kriging
Spatial interpolation via Gaussian processes.
- **Weakness**: assumes stationarity; computationally O(N³); doesn't scale to
  307 sensors; no temporal component.

---

## 3. Deep Learning for Traffic

### 3.1 Recurrent Neural Networks (RNN / LSTM)
Early deep approaches model each sensor independently as a time series.
- **LSTM (Hochreiter & Schmidhuber, 1997)**: long-short term memory cells
  capture temporal dependencies.
- **Weakness for traffic**: sensors are spatially correlated; modelling them
  independently ignores road topology entirely.

### 3.2 Convolutional Sequence Models
- **WaveNet (van den Oord et al., 2016)**: dilated causal convolutions achieve
  large receptive fields without RNN depth. Adapted for traffic in Graph WaveNet.
- **Weakness**: 1-D convolution has no notion of graph structure.

---

## 4. Graph Neural Networks for Traffic

Graph Neural Networks (GNNs) treat sensors as nodes and roads as edges,
naturally encoding spatial dependency.

### 4.1 Spectral GNNs
- **ChebNet (Defferrard et al., 2016)**: polynomial approximations of spectral
  graph convolution.
- **GCN (Kipf & Welling, 2017)**: first-order approximation: `X' = D^{-1/2}AD^{-1/2}XW`.
- **Weakness**: symmetric normalisation treats all neighbours equally — ignores
  that a congested upstream neighbour matters more than a free-flowing one.

### 4.2 Graph Attention Networks (GAT)
- **GAT (Veličković et al., 2018)**: replaces fixed weights with learned
  attention scores `α_ij = softmax(a(Wh_i, Wh_j))`.
- **This work uses GAT** as the primary spatial aggregator inside the ODE
  function. Additive decomposition (`a_src + a_dst`) used for efficiency.
  Temperature τ=2 softens the distribution to prevent single-neighbour dominance.

### 4.3 Spatio-Temporal Models for Forecasting

These are the most relevant published benchmarks. Note: they are designed for
**forecasting** (all sensors observed, predict future), not sparse imputation.

| Model | Venue | Mechanism | PEMS04 MAE |
|---|---|---|---|
| DCRNN (Li et al., 2018) | ICLR 2018 | Diffusion GCN + seq2seq RNN | ~1.8 km/h |
| STGCN (Yu et al., 2018) | IJCAI 2018 | Graph conv + temporal conv | ~1.7 km/h |
| Graph WaveNet (Wu et al., 2019) | IJCAI 2019 | Adaptive adj + dilated conv | ~1.6 km/h |
| ASTGCN (Guo et al., 2019) | AAAI 2019 | Spatial + temporal attention | ~1.6 km/h |
| AGCRN (Bai et al., 2020) | NeurIPS 2020 | Node-adaptive GCN + GRU | ~1.5 km/h |

> **Important**: These MAE values are for the full-sensor, 15-minute-ahead
> forecasting task on PEMS04. They are **not directly comparable** to this
> work's sparse imputation MAE (~5 km/h overall, ~34 km/h jam) because:
> (a) 80% of sensors are hidden here, (b) the task is imputation not forecasting,
> (c) jam performance is reported separately.

---

## 5. Neural ODEs

### 5.1 Neural ODE (Chen et al., NeurIPS 2018)
Parameterise the hidden state derivative as a neural network:
```
dz/dt = f_θ(z, t)
z(t1) = z(t0) + ∫[t0,t1] f_θ(z,t) dt
```
Solved with an ODE solver (e.g. RK4, dopri5).
- **Advantage**: continuous-time modelling, memory-efficient backprop via adjoint.
- **Weakness for traffic**: original form uses a generic MLP, no graph structure.

### 5.2 Graph Neural ODE (GNDE variants)
- **GODE (Poli et al., 2019)**: replace MLP with GNN in the ODE function.
- **STGODE (Fang et al., SIGKDD 2021)**: spatio-temporal ODE for traffic with
  semantic and spatial graphs.
- **This work**: uses Euler integration (not dopri5) inside the ODE function
  to avoid gradient vanishing from multiple solver evaluations per backprop step.
  GAT layers replace the fixed-adjacency GCN for adaptive spatial weighting.

---

## 6. Hypergraph Neural Networks

### 6.1 Standard Hypergraph
A hypergraph G = (V, E) allows edges (hyperedges) to connect more than two
nodes. A road corridor of 5 sensors is naturally a hyperedge.

### 6.2 HGNN (Feng et al., AAAI 2019)
Hypergraph convolution:
```
X' = D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2} X Θ
```
where H is the incidence matrix, D_v and D_e are node/hyperedge degree matrices.

### 6.3 HyperGCN (Yadati et al., NeurIPS 2019)
Approximates hypergraph Laplacian with a standard graph Laplacian via mediator
nodes, reducing computation.

### 6.4 This Work
- Hyperedges are **2-hop corridor clusters**: each node i defines a hyperedge
  containing i and all nodes reachable in 1 or 2 hops.
- H_conv is pre-computed offline so runtime cost is a single dense matmul.
- A **learnable sigmoid gate** (init = sigmoid(-2) ≈ 0.12) controls how much
  the hypergraph branch contributes to the ODE derivative, preventing
  over-smoothing at jam nodes.

---

## 7. Observation Assimilation / Data Assimilation in ML

### 7.1 Classical Data Assimilation
Kalman Filter (Kalman, 1960): optimal linear estimator blending model prediction
with sensor observations. Extended Kalman Filter (EKF) and Ensemble Kalman
Filter (EnKF) extend this to nonlinear systems.

### 7.2 Learned Assimilation in Neural Networks
- **GRU-D (Che et al., 2018)**: GRU with decay mechanism for irregular time series.
- **ODE with jumps (Rubanova et al., NeurIPS 2019)**: ODE hidden state jumps at
  observation times via a recognition network.
- **This work**: GRU-style gate at each timestep:
  ```
  update = σ(W[z; z_obs]) * (z_obs - z) * obs_mask
  z ← z + update
  ```
  Blind nodes are zeroed out by `* obs_mask`, preventing leakage.
  The gate learns *how much* to trust each new sensor reading vs the ODE prediction.

---

## 8. Physics-Informed Neural Networks (PINNs)

### 8.1 PINN (Raissi et al., JCP 2019)
Encode PDE residuals as additional loss terms. Widely used in fluid dynamics,
heat transfer, and traffic flow.

### 8.2 Traffic Flow Physics: LWR Model
Lighthill-Whitham-Richards (LWR) model: traffic obeys a conservation law.
Speed gradients propagate continuously — a sharp speed boundary cannot appear
instantaneously. Discretised on a graph: speed at node i should not differ
drastically from the mean speed of its road neighbours.

### 8.3 This Work
Graph Laplacian regularisation:
```
L_phys = ||L_sym · v||² = Σ_i (v_i − mean_nbr(v_i))²
```
where L_sym = I − D^{-1/2} A D^{-1/2}.
This penalises predicted speeds that violate the continuous-flow principle.
λ_physics = 0.02 keeps it as a soft constraint, not overriding data evidence.

---

## 9. Curriculum Learning

### 9.1 Curriculum Learning (Bengio et al., ICML 2009)
Start with easy examples, gradually increase difficulty. Improves convergence
and generalisation for tasks with imbalanced difficulty.

### 9.2 This Work: Curriculum Masking
At each training batch, 15% of observed sensors are randomly hidden
("pseudo-blind"). The model forward pass treats them as blind nodes, but the
loss is computed on their known ground truth. This ensures:
- Gradients flow through the blind-node code path every batch.
- The model cannot memorise "observed = good, blind = ignore".

---

## 10. Position of This Work

| Aspect | This Work | DCRNN/STGCN/WaveNet |
|---|---|---|
| Task | Sparse imputation (80% sensors missing) | Full-sensor forecasting |
| Graph structure | Hypergraph + pairwise GAT | Standard adjacency |
| Temporal model | Continuous Neural ODE (Euler) | Discrete RNN / Conv |
| Sensor fusion | Kalman-style assimilation gate | Not applicable |
| Physics | Graph Laplacian regularisation | None |
| Training | Curriculum masking + jam-biased sampling | Standard mini-batch |

This work is the first to combine **Hypergraph Neural ODE + Kalman-style
observation assimilation + physics regularisation** for the sparse traffic
imputation problem.
