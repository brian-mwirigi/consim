# Findings

Empirical results from consim — a multi-agent simulation testing the Machine Consciousness Hypothesis (MCH).

**What consim is:** Brian Munene Mwirigi's original simulation architecture, inspired by the MCH (Fitz, 2025). The theory is Fitz's. The simulation, metrics, experiments, and all results are Brian's original work. Fitz proposed that consciousness arises from lossy predictive communication between observers capable of second-order perception; consim is the first empirical implementation.

**Total compute:** 5,960 simulation runs across 10 experiments.

| Experiment | Runs | Key variable | CSV |
|---|:-:|---|---|
| Primary (size 24) | 1,250 | 5 topologies × 5 noises × 50 seeds | sweep_1250.csv |
| Replication (size 12) | 1,250 | same design, smaller grid | sweep_size12.csv |
| Size 48 + MCH metrics | 250 | R, T, E measured first time | sweep_size48.csv |
| Scaling (sizes 12–36) | 1,000 | 5 sizes × 5 topos × 5 noises × 10 seeds | sweep_size{12,18,24,36}_full.csv |
| Null model (lr=0) | 250 | learning disabled | sweep_null.csv |
| Activation universality | 360 | linear, ReLU | sweep_linear.csv, sweep_relu.csv |
| K* transition | 1,000 | K = 3–12 on random graphs | sweep_k_transition.csv |
| Second-order perception | 500 | γ × noise full sweep | sweep_second_order.csv |
| C. elegans connectome | 100 | real biological wiring | sweep_celegans.csv |

---

## Part I: The Foundation

### 1. Emergent Self-Prediction

Agents learn to predict their neighbors' states via gradient descent. Self-prediction — how well an agent's broadcast matches its own next state — is never in the gradient. It is never trained. If it rises above chance, the agent has built an implicit self-model as a side effect of modeling others.

**Result:** Self-prediction emerges in every topology, every noise level, every grid size. Mean self-prediction ranges from 0.18 to 0.21 (well above chance). Individual agents reach 0.90+. This confirms Fitz's core intuition: lossy predictive communication between observers generates emergent self-prediction.

### 2. Three Robust Structural Effects

These three findings replicate across all grid sizes (12, 18, 24, 36, 48), all activation functions (tanh, linear, ReLU), and with learning disabled (lr=0).

#### 2a. Moore noise amplification

On high-connectivity grids (Moore K=8, hex K=6), noise *helps* self-prediction. More noise → better self-models.

| Noise | Moore self | Hex self | Random self |
|-------|:-:|:-:|:-:|
| 0.04 | 0.188 | 0.189 | 0.182 |
| 0.12 | 0.198 | 0.198 | 0.183 |
| 0.30 | 0.206 | 0.204 | 0.182 |

**Mechanism (see THEORY.md §2):** On high-K grids, adjacent agents share many neighbors and converge to similar states, creating redundancy. Noise decorrelates them. With 8 neighbors averaging out the noise, the signal survives while the redundancy breaks. More diverse neighbor states → richer learning gradients → better self-models.

#### 2b. Random topology is a fixed point

Random graphs are immune to noise. Every metric — self-prediction, Phi, R, T — is flat within 0.002 across a 7.5× noise range (0.04 → 0.30).

| Metric | noise=0.04 | noise=0.30 | Δ |
|--------|:-:|:-:|:-:|
| mean_self | 0.182 | 0.182 | <0.001 |
| mean_phi | 0.347 | 0.343 | 0.004 |
| mean_T | 0.659 | 0.658 | 0.001 |
| mean_R | 0.181 | 0.181 | <0.001 |

Exception: E still declines with noise on random graphs (noise increases external perturbation on any topology). But all prediction-related metrics are invariant.

**Mechanism (THEORY.md §1):** On random graphs, neighbors are independent samples from the population. By the law of large numbers, the received signal converges to the population mean regardless of noise level. This creates a self-stabilizing loop.

#### 2c. Small-world Phi dissociation

Small-world shortcuts (10% edge rewiring from von Neumann) boost integration (Phi) by 20% at high noise while leaving self-prediction unchanged.

| Noise | Phi boost (sw − vn) | Self diff (sw − vn) |
|-------|:-:|:-:|
| 0.04 | +0.015 | −0.001 |
| 0.12 | +0.037 | −0.001 |
| 0.30 | +0.044 | +0.000 |

This is a clean dissociation. The shortcuts add uncorrelated long-range neighbors that are poor individual predictors (raising the parts residual in Phi) while barely affecting the received signal. See THEORY.md §3.

---

## Part II: The Autonomy-Predictability Constraint

### 3. How Noise Differentially Affects T and E on Structured Networks

On structured networks, increasing noise pushes causal efficacy (E) down while pushing temporal persistence (T) slightly up. This creates a strong negative correlation when pooled across noise levels.

**Moore topology, size 24 (baseline, 10 seeds per noise level):**

| Noise | E | T | Within-noise r(T,E) |
|-------|:-:|:-:|:-:|
| 0.04 | 0.923 | 0.659 | −0.33 |
| 0.08 | 0.802 | 0.661 | −0.29 |
| 0.12 | 0.718 | 0.661 | −0.16 |
| 0.20 | 0.634 | 0.662 | +0.05 |
| 0.30 | 0.593 | 0.662 | +0.12 |
| **Pooled** | **0.734** | **0.661** | **−0.70** |

The pooled r(T,E) = −0.70 is driven by noise acting as a confound: higher noise systematically lowers E (more external perturbation → less self-determination) while slightly raising T (averaging smooths trajectories → more temporal stability). Within any single noise level (n = 10 seeds), r(T,E) is substantially weaker than the pooled value — none reaching statistical significance — though a trend from moderately negative at low noise to weakly positive at high noise is visible.

**What this means:** The T-E anti-correlation is a between-condition effect. It describes how the system responds to noise, not how individual agents trade off T against E within a single environment. On structured networks, conditions that produce high causal autonomy are conditions that produce low temporal stability, and vice versa.

**Topology dependence (all pooled across noise levels):**

| Topology | K | Pooled r(T,E) |
|----------|:-:|:-:|
| moore | 8 | −0.70 |
| hex | 6 | −0.44 |
| C. elegans | ~21 | −0.16 |
| random | 4 | −0.08 |
| small_world | 4+ | +0.08 |
| von_neumann | 4 | +0.21 |

The differential response to noise is strongest on high-K structured grids (Moore, hex) and weakest on random and low-K graphs. Von Neumann (K=4) shows a mild *positive* r(T,E), suggesting that on sparse regular grids the pooled noise effect can reverse sign.

### 4. E ≡ Phi on Structured Graphs

Causal efficacy and our integration proxy measure the same thing on spatially structured networks and completely decouple on random graphs.

| Topology | r(E, Φ) |
|----------|:-:|
| von_neumann | +0.970 |
| moore | +0.971 |
| hex | +0.970 |
| small_world | +0.919 |
| C. elegans | +0.889 |
| random | +0.210 |

**Important caveat:** Our Phi is a simplified partition-free proxy based on joint vs. partitioned neighborhood prediction residuals. It is NOT Tononi's full IIT integrated information, which requires exponential partition search. The E ≡ Phi identity holds for our proxy on structured networks. We cannot claim this extends to the full IIT measure.

### 5. Null Model: The Constraint Is Structural, Not Learned

A null model (250 runs, lr=0, weights frozen at random initialization) produces the same T-E pattern:

| Correlation | Learning (lr=0.01) | Null (lr=0) | Δ |
|-------------|:-:|:-:|:-:|
| moore r(T,E) pooled | −0.82 | −0.79 | 0.03 |
| hex r(T,E) pooled | −0.74 | −0.70 | 0.04 |
| von_neumann r(T,E) | +0.07 | +0.11 | 0.04 |

Learning at lr=0.01 for 1,000 ticks improves mean_self by 0.0008 — the entire metric structure is dominated by the state dynamics equation and graph topology, not by learned weights.

**Interpretation for the MCH:** The constraint on consciousness correlates is a geometric property of the network, not a product of learning. This means the structural conditions that allow or constrain consciousness are present *before* any learning occurs. Learning operates within this pre-existing geometric structure.

### 6. Activation Universality

The effects persist across all three activation functions:

| Activation | Moore r(T,E) pooled | Moore r(E,self) pooled |
|---|:-:|:-:|
| tanh | −0.70 | −0.46 |
| linear | −0.77 | −0.53 |
| ReLU | −0.76 | −0.61 |

All three measured at size 24 (N = 576 agents). The T-E pattern does not require nonlinear dynamics. Even purely linear activations (with clipping) produce the same structure — and in fact show slightly stronger coupling than tanh at the same grid size. The constraint is topological, not dynamical.

---

## Part III: Scaling Law

### 7. Dimensionality Collapse as a Phase Transition

PCA on the five metrics {self, Φ, R, T, E} reveals that on Moore grids, a single principal component captures most of the variance — and the fraction scales logarithmically with system size.

| Size | N agents | PC1 (moore) | PC1 (random) | Pooled r(T,E) moore |
|------|:-:|:-:|:-:|:-:|
| 12 | 144 | 48.8% | 49.5% | −0.11 |
| 18 | 324 | 55.9% | 38.7% | −0.25 |
| 24 | 576 | 67.4% | 36.3% | −0.70 |
| 36 | 1,296 | 75.1% | 47.3% | −0.78 |
| 48 | 2,304 | 82.5% | 39.6% | −0.82 |

Random stays flat at ~40%. Moore rises from 49% to 83%.

**Scaling law fit:** PC1 = −0.136 + 0.124 × ln(N), R² = 0.987

**The phase transition:** The largest gain in both PC1 and r(T,E) occurs between N=324 and N=576 agents. Below ~300 agents, the five metrics are nearly independent. Above ~600, they lock into a single axis. This resembles a phase transition — below a critical system size, the five MCH correlates behave as independent quantities; above it, they collapse into one.

**PC1 loadings (the autonomy-predictability axis):**

| Metric | Loading | Pole |
|--------|:-:|---|
| Φ | +0.48 | autonomy |
| E | +0.49 | autonomy |
| R | +0.45 | autonomy |
| T | −0.44 | predictability |
| self | −0.38 | predictability |

Five consciousness correlates reduce to one dimension on large structured networks: the balance between autonomy and predictability. The system's environment (noise level) determines WHERE on this axis each agent sits.

---

## Part IV: Follow-Up Experiments

### 8. K* Transition (1,000 runs)

**Question:** Is there a critical connectivity K* where the T-E relationship sharply changes?

**Design:** Random graphs, K = 3 to 12, 100 runs per K, size 24.

| K | Mean E | Mean T | Mean self | Pooled r(T,E) |
|:-:|:-:|:-:|:-:|:-:|
| 3 | 0.672 | 0.658 | 0.188 | +0.10 |
| 4 | 0.686 | 0.658 | 0.188 | −0.06 |
| 5 | 0.703 | 0.658 | 0.188 | −0.01 |
| 6 | 0.711 | 0.658 | 0.187 | +0.09 |
| 7 | 0.725 | 0.658 | 0.188 | −0.10 |
| 8 | 0.737 | 0.659 | 0.189 | +0.04 |
| 9 | 0.746 | 0.658 | 0.181 | +0.01 |
| 10 | 0.755 | 0.658 | 0.185 | −0.17 |
| 11 | 0.763 | 0.658 | 0.183 | +0.04 |
| 12 | 0.768 | 0.659 | 0.183 | −0.02 |

**Findings:**
- E rises monotonically with K (0.672 → 0.768). More neighbors → more information about the agent's causal contribution.
- T is invariant across K. Self-prediction is flat.
- **No sharp critical K.** The pooled r(T,E) fluctuates near zero at all K values on random graphs. The T-E differential noise response requires spatially correlated neighbors, not just high connectivity. This distinguishes between two possible mechanisms: the constraint could arise from connectivity alone, or from the spatial correlation structure. The K* experiment rules out connectivity — spatial structure is essential.

### 9. Second-Order Perception (500 runs)

**Question:** Does self-modeling capacity — the MCH's proposed mechanism for consciousness — change the structural constraint on consciousness correlates?

**Design:** Each agent gets a second-order weight matrix V that learns to predict the agent's own next broadcast. The state update becomes:

$$s_i' = \tanh(\alpha s_i + (1-\alpha) r_i + \gamma \cdot V_i \cdot m_i + \text{drive})$$

where γ controls feedback strength. Full factorial: 20 seeds × 5 γ values × 5 noise levels = 500 runs on Moore topology, size 24.

**This is the most important experiment in the project.** The MCH predicts that second-order perception — the ability to model one's own modeling process — is essential for consciousness. The question is: does adding this ability change the fundamental structural constraint, or does the system operate within it?

#### 9a. Self-prediction improves dose-dependently

| γ | Self | E | T | Phi | R | so_err |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0.00 | 0.1995 | 0.7296 | 0.6612 | 0.2894 | 0.1217 | 0.0340 |
| 0.01 | 0.2004 | 0.7293 | 0.6614 | 0.2897 | 0.1223 | 0.0340 |
| 0.05 | 0.2039 | 0.7280 | 0.6620 | 0.2910 | 0.1248 | 0.0340 |
| 0.10 | 0.2083 | 0.7263 | 0.6628 | 0.2926 | 0.1279 | 0.0339 |
| 0.20 | 0.2174 | 0.7232 | 0.6645 | 0.2960 | 0.1344 | 0.0339 |

All values pooled across 5 noise levels (100 runs per γ).

Self-prediction improves +9.0% from γ=0 to γ=0.20. Reflexivity improves +10.4%. Phi improves +2.3%. This is dose-dependent and monotonic. The V layer's self-prediction feedback makes the agent's trajectory more self-predictable, more reflexive, and more integrated.

γ also shifts the system: **T rises, E falls.** Self-modeling feedback makes agents slightly less self-determined (E: 0.730 → 0.723, −0.9%) and more temporally stable (T: 0.661 → 0.665, +0.5%). Second-order perception trades a small amount of autonomy for better self-knowledge.

#### 9b. The pooled T-E constraint *strengthens* with self-modeling

This is the central finding. Comparing pooled r(T,E) across all 5 noise levels:

| Condition | Pooled r(T,E) |
|-----------|:-:|
| **Baseline (no V layer)** | **−0.7037** |
| γ=0.00 (V present, no feedback) | −0.6113 |
| γ=0.01 | −0.6135 |
| γ=0.05 | −0.6225 |
| γ=0.10 | −0.6332 |
| **γ=0.20** | **−0.6531** |

**Note on the baseline comparison:** The baseline (sweep_size24_full.csv, seeds 1–10) and the second-order runs (seeds 1–20) use different random number generator streams because SecondOrderWorld consumes extra random draws for V initialization. The γ=0.00 vs baseline difference (−0.61 vs −0.70) may partly reflect different RNG streams, not just the presence of the V layer. The proper comparison is **within-SecondOrderWorld**: γ=0.00 vs γ=0.20.

**Within-SecondOrderWorld, increasing γ makes the pooled r(T,E) MORE negative:**

$$r(T,E)_{\text{pooled}}: \;\; -0.611 \xrightarrow{\gamma=0 \to 0.20} -0.653$$

This is a 6.8% strengthening. The structural constraint **does not break with self-modeling — it deepens.**

#### 9c. But within-noise, the constraint weakens

| Noise | r(T,E) at γ=0.00 | r(T,E) at γ=0.20 | Direction |
|:-:|:-:|:-:|---|
| 0.04 | −0.08 | −0.08 | stable |
| 0.08 | −0.06 | −0.07 | → slightly more negative |
| 0.12 | +0.07 | +0.06 | → slightly toward zero |
| 0.20 | +0.09 | +0.11 | → away from zero |
| 0.30 | +0.05 | +0.08 | → away from zero |

Within any single noise level, the T-E correlation is near zero and either stays near zero or moves slightly away from it. Self-modeling provides **local independence** — agents in a given environment are not constrained in their T-E relationship.

#### 9d. Self-modeling amplifies noise sensitivity

The key to the paradox: why does the pooled constraint strengthen if the within-noise constraint weakens?

| γ | T spread (across noise) | E spread (across noise) |
|:-:|:-:|:-:|
| 0.00 | 0.0028 | 0.3325 |
| 0.05 | 0.0029 | 0.3347 |
| 0.10 | 0.0030 | 0.3369 |
| 0.20 | 0.0032 | 0.3411 |

Self-modeling increases the E spread by 2.6% — it makes the system *more* sensitive to noise. At low noise, E is pushed higher; at high noise, E is pushed lower. This wider spread between conditions strengthens the pooled anti-correlation even as the within-condition correlation weakens.

**Interpretation:** Self-modeling does not escape the structural constraint. It **optimizes within it**, trading autonomy for predictability. The system becomes more coupled to its environment (more noise-sensitive), while gaining local independence within each environment.

**Note on V learning:** The so_err (self-prediction error of the V matrix) barely changes with γ (0.0340 → 0.0339 pooled). At 1,000 ticks with a low learning rate, V does not meaningfully diverge from its identity-biased initialization. The improvements in self-prediction and reflexivity are driven primarily by the feedback mechanics (the γ·V·s term in the state update), not by V's learned representations. Longer simulations (10k+ ticks) would test whether trained V weights qualitatively change the picture.

#### 9e. Dimensionality collapse increases with γ

| γ | PC1 (%) | r(E,Φ) |
|:-:|:-:|:-:|
| 0.00 | 63.7% | +0.972 |
| 0.01 | 63.8% | +0.972 |
| 0.05 | 64.2% | +0.972 |
| 0.10 | 64.7% | +0.972 |
| 0.20 | 65.7% | +0.972 |

PC1 rises from 63.7% to 65.7% with γ. Self-modeling makes the five consciousness correlates MORE correlated, not less. The system becomes MORE dimensionally collapsed. r(E,Φ) is unchanged — the E-Phi coupling is unaffected by the V layer.

**PC1 loadings at γ=0.20:** self=−0.12, Phi=+0.54, R=+0.48, T=−0.41, E=+0.54

The same autonomy-predictability axis. Self-modeling doesn't change the axis — it moves the system further along it.

### 10. C. elegans Connectome (100 runs)

**Question:** Where does a real biological nervous system sit on the topology spectrum?

**Data:** The C. elegans hermaphrodite connectome from OpenWorm (c302 project). 448 neurons, 7,379 synaptic connections (4,681 chemical synapses + 2,698 electrical gap junctions). Mean degree 21.3, range 1–100. This is one of only two complete nervous systems ever mapped at single-synapse resolution.

**Design:** 20 seeds × 5 noise levels (0.04–0.30), 1,000 ticks. Each agent corresponds to a real neuron. Adjacency matrix built from actual synaptic partners, weighted by synapse count.

| Noise | E | T | Self | Phi |
|-------|:-:|:-:|:-:|:-:|
| 0.04 | 0.924 | 0.659 | 0.196 | 0.374 |
| 0.08 | 0.819 | 0.660 | 0.195 | 0.352 |
| 0.12 | 0.742 | 0.660 | 0.195 | 0.339 |
| 0.20 | 0.656 | 0.660 | 0.195 | 0.326 |
| 0.30 | 0.611 | 0.660 | 0.194 | 0.320 |

**Pooled:** E = 0.750, T = 0.660, self = 0.195, Phi = 0.342, R = 0.155

**Findings:**

1. **Highest causal efficacy of any topology tested.** E = 0.750 (vs. moore 0.734, hex 0.713, random 0.685). The biological wiring preserves self-determination better than any synthetic topology.

2. **Near-best self-prediction.** Self = 0.195, comparable to moore (0.196) and hex (0.199).

3. **Nearly noise-immune self-prediction.** Self ranges from 0.196 to 0.194 across a 7.5× noise range — nearly as flat as random. Degree heterogeneity may provide noise robustness similar to random's independent-neighbor mechanism.

4. **Moderate pooled T-E anti-correlation.** r(T,E) = −0.16, between hex (−0.44) and random (−0.08). Biology partially relaxes the noise-driven T-E pattern seen on regular grids.

5. **Strong E-Phi coupling.** r(Phi,E) = +0.889, weaker than pure grids (0.97+) but far stronger than random (0.21). The biological network behaves more like a structured grid than a random graph for integration-autonomy coupling.

---

## Part V: The Full Topology Spectrum

Six systems ordered by the strength of the noise-driven T-E response:

| System | K | Pooled r(T,E) | E | T | Self | Phi | r(Phi,E) |
|--------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| moore | 8 | −0.70 | 0.734 | 0.661 | 0.196 | 0.287 | +0.971 |
| hex | 6 | −0.44 | 0.713 | 0.661 | 0.199 | 0.279 | +0.970 |
| C. elegans | ~21 | −0.16 | 0.750 | 0.660 | 0.195 | 0.342 | +0.889 |
| random | 4 | −0.08 | 0.685 | 0.658 | 0.180 | 0.343 | +0.210 |
| small_world | 4+ | +0.08 | 0.692 | 0.659 | 0.190 | 0.283 | +0.919 |
| von_neumann | 4 | +0.21 | 0.692 | 0.658 | 0.190 | 0.248 | +0.970 |

**Key observations:**

1. **C. elegans does NOT sit between small-world and random** as small-world theory might predict. It sits between hex and random on the T-E spectrum. Its mean degree of 21.3 and hub-rich degree distribution place it closer to the random end for T-E coupling, while its spatial wiring gives it structured-grid-like E and Phi values.

2. **Biology achieves high E + high Phi simultaneously.** C. elegans is the only topology that ranks in the top 3 for both causal efficacy AND integration. Heterogeneous degree distributions may allow this combination.

3. **Random leads in Phi.** Random topology has the highest integration (0.343) despite the weakest E-Phi coupling. Phi on random graphs measures population-level variance structure rather than individual self-determination.

---

## Part VI: What This Means for Testing the MCH

The MCH (Fitz, 2025) proposes that consciousness arises from lossy predictive communication between observers capable of second-order perception. Consim tests this by measuring four MCH correlates (Φ, R, T, E) in networks of predictive agents, asking: what constraints does the network structure impose on these correlates, and does second-order perception change those constraints?

### What the simulation demonstrates

1. **Lossy predictive communication generates emergent self-prediction.** Fitz's core claim is computationally confirmed. Agents that only learn to predict neighbors develop self-models as a side effect. This emerges in every topology, every noise level, every grid size.

2. **The four MCH correlates collapse to one dimension on structured networks.** On Moore grids above ~576 agents, a single principal component captures 82.5% of the variance in {Φ, R, T, E, self-prediction}. This is a predicted consequence of structured communication: the information geometry of the network determines a single axis (autonomy vs. predictability) along which all correlates covary.

3. **The collapse follows a scaling law with a phase transition.** PC1 = −0.136 + 0.124 × ln(N), R² = 0.987. Below ~300 agents, correlates are independent; above ~600, they lock together. This transition is topologically governed — it requires spatial structure, not just connectivity (K* experiment).

4. **The constraint is structural, not learned.** It persists without learning (lr=0), across activation functions (tanh, linear, ReLU), and across system sizes. It is a geometric property of the network, not a product of optimization.

5. **Second-order perception does not break the structural constraint.** Adding self-modeling capacity (γ > 0) makes the system trade autonomy for predictability, improves self-prediction by 9.0%, and **strengthens** the pooled T-E anti-correlation by 6.8%. The dimensionality collapse also increases (PC1: 63.7% → 65.7%).

6. **Second-order perception provides local independence within the global constraint.** Within any single noise condition, the T-E correlation is near zero regardless of γ. Self-modeling decouples T and E locally while amplifying the system's sensitivity to environmental conditions globally.

7. **Real biological wiring occupies a specific position on the spectrum.** C. elegans achieves the highest causal efficacy (E = 0.750) and strong integration (Phi = 0.342) while operating at a moderate pooled r(T,E) = −0.16. Biology sits at a point where the structural constraint is present but not rigid.

### The second-order paradox

The MCH predicts that second-order perception is essential for consciousness. Consim shows that adding it doesn't free the system from structural constraints — it deepens them. Self-modeling agents become:
- Better self-predictors (+9.0% self-prediction, +10.4% reflexivity)
- More integrated (+2.3% Phi)
- More temporally stable (+0.5% T)
- Less causally autonomous (−0.9% E)
- More environmentally coupled (+2.6% E spread across noise levels)
- More dimensionally collapsed (+3.1% PC1)

This is not a failure of the MCH — it may be exactly what the theory predicts. Second-order perception doesn't create independence from physical constraints. It creates a system that is *better at navigating within them*. The agent optimizes its position on the autonomy-predictability axis, sliding toward the predictability pole where self-knowledge is maximized.

The biological operating point (C. elegans at r(T,E) = −0.16) may represent the evolutionary compromise: enough structure for self-modeling to work, but not so much that the constraint locks agents into a single behavioral mode.

### Known limitations

1. **Phi is a proxy.** Our partition-free Phi proxy is not Tononi's full IIT integrated information. The E ≡ Phi identity (r = +0.97 on Moore) holds for our proxy; we cannot claim it extends to the full IIT measure.

2. **1,000 ticks is short.** Learning barely contributes at this timescale (Δmean_self ≈ 0.0008 vs null model). Longer simulations (10k, 50k, 100k ticks) would test whether learning eventually differentiates the system from the structural null.

3. **The pooled r(T,E) is a between-condition effect.** It is driven by noise acting as a confounding variable, not by individual agents trading T against E. Within any single environment, r(T,E) is substantially weaker than the pooled value and not statistically significant at the sample sizes used (n = 10–20 seeds). This is not a flaw — it characterizes how the system's information geometry responds to its environment — but it must be stated clearly.

4. **One organism.** The C. elegans result is from 448 neurons. Cortical data (Allen Mouse Brain Atlas, Human Connectome Project) would test whether the biological operating point is conserved across scales.

5. **No proof of consciousness.** Consim demonstrates structural constraints on *correlates* of consciousness as defined by the MCH. Elevated metrics are necessary but not sufficient conditions. The simulation is evidence for the MCH's structural predictions, not proof that these agents are conscious.

---

## Part VII: Open Questions

### Highest priority

1. **Why does C. elegans relax the T-E pattern?** Degree heterogeneity (range 1–100), hub neurons, and asymmetric synaptic weights may break the averaging argument that drives the constraint on regular grids. Formal analysis needed.

2. **Longer learning.** Does 10,000 or 100,000 ticks eventually let learning differentiate from the null model? If learning never matters, the MCH's emphasis on "predictive" communication may need qualification — the communication structure matters, but the prediction quality does not.

3. **More biological connectomes.** Mouse cortex (Allen Institute), Drosophila (FlyWire), human diffusion MRI. Does the moderate r(T,E) ≈ −0.16 appear in all nervous systems?

### Deeper investigations

4. **Mathematical derivation of the phase transition.** The N=324 → N=576 scaling transition should be derivable from the effective sample size and correlation structure. See THEORY.md for partial results.

5. **The scaling law saturation.** The log fit predicts PC1 → 1.0 at ~8,000 agents. Simulations at N=4,096+ would test whether the law saturates or breaks.

6. **Time evolution of the constraint.** Does the pooled r(T,E) change over time within a single run? Early ticks vs. late ticks could reveal whether the constraint emerges gradually or is present from initialization.

7. **γ beyond 0.20.** At what point does self-modeling feedback destabilize the system? Is there a critical γ*?

---

## Reproduce

```bash
pip install numpy matplotlib Pillow

# Primary sweep (size 24, 1,250 runs)
python run.py --sweep --sweep-seeds 1-50 \
  --sweep-topos von_neumann,moore,hex,random,small_world \
  --sweep-noises 0.04,0.08,0.12,0.20,0.30 \
  --ticks 1000 --size 24 --sweep-csv sweep_1250.csv

# Replication (size 12)
python run.py --sweep --sweep-seeds 1-50 \
  --sweep-topos von_neumann,moore,hex,random,small_world \
  --sweep-noises 0.04,0.08,0.12,0.20,0.30 \
  --ticks 1000 --size 12 --sweep-csv sweep_size12.csv

# Size 48 + MCH metrics
python run.py --sweep --sweep-seeds 1-10 \
  --sweep-topos von_neumann,moore,hex,random,small_world \
  --sweep-noises 0.04,0.08,0.12,0.20,0.30 \
  --ticks 1000 --size 48 --sweep-csv sweep_size48.csv

# Null model
python run.py --sweep --sweep-seeds 1-10 \
  --sweep-topos von_neumann,moore,hex,random,small_world \
  --sweep-noises 0.04,0.08,0.12,0.20,0.30 \
  --ticks 1000 --size 48 --lr 0 --sweep-csv sweep_null.csv

# Scaling sweeps
python run.py --sweep --sweep-seeds 1-10 \
  --sweep-topos von_neumann,moore,hex,random,small_world \
  --sweep-noises 0.04,0.08,0.12,0.20,0.30 \
  --ticks 1000 --size 18 --sweep-csv sweep_size18_full.csv

python run.py --sweep --sweep-seeds 1-10 \
  --sweep-topos von_neumann,moore,hex,random,small_world \
  --sweep-noises 0.04,0.08,0.12,0.20,0.30 \
  --ticks 1000 --size 36 --sweep-csv sweep_size36_full.csv

# Activation universality
python run.py --sweep --sweep-seeds 1-10 \
  --sweep-topos von_neumann,moore,random \
  --sweep-noises 0.04,0.12,0.30 \
  --ticks 1000 --size 24 --activation relu --sweep-csv sweep_relu.csv

python run.py --sweep --sweep-seeds 1-10 \
  --sweep-topos von_neumann,moore,random \
  --sweep-noises 0.04,0.12,0.30 \
  --ticks 1000 --size 24 --activation linear --sweep-csv sweep_linear.csv

# K* transition
python k_transition.py

# Second-order perception (500 runs)
python second_order.py

# C. elegans connectome
python connectome_analysis.py
```
