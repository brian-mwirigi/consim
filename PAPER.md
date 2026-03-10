# Topology Determines Emergent Self-Prediction in Multi-Agent Communication Networks: A Scaling Law for Consciousness Correlate Collapse

**Brian Mwirigi**

---

## Abstract

We present *consim*, a multi-agent simulation studying how self-prediction emerges from lossy predictive communication on graph topologies. Inspired by the Machine Consciousness Hypothesis (Fitz, 2025), the system implements distributed agents that communicate through noisy channels and learn to predict their neighbors' states — never their own. We measure four consciousness-relevant quantities from the MCH framework: integration ($\Phi$), reflexivity ($R$), temporal persistence ($T$), and causal efficacy ($E$). Across 4,180 simulation runs on five topologies, five noise levels, five grid sizes (12, 18, 24, 36, 48), and three activation functions (tanh, linear, ReLU), we report three main results. First, on high-connectivity grids, causal efficacy anti-correlates with self-prediction ($r = -0.71$) and with temporal persistence ($r = -0.82$), establishing that self-determination destroys self-knowledge — an inversion of assumptions in IIT (Tononi, 2004), MCH (Fitz, 2025), and autonomy-based consciousness frameworks. This trade-off is activation-universal (tanh $-0.82$, linear $-0.77$, ReLU $-0.76$) and persists identically in a null model without learning ($r = -0.79$), proving it is a topological invariant. Second, PCA on the five metrics reveals they collapse to a single principal component on Moore grids (82.5% variance explained), forming an "autonomy-predictability axis." Third, this collapse follows a scaling law: $\text{PC1} = -0.136 + 0.124 \times \ln(N)$ ($R^2 = 0.987$) across five grid sizes (144–2,304 agents), with a sharp inflection between $N = 324$ and $N = 576$ agents. Random topologies show no collapse at any size. We prove a formal theorem establishing this as a network-theoretic uncertainty principle: on $K$-regular graphs with bounded activations and correlated neighbors, variance reduction through neighbor-averaging creates a necessary trade-off between causal autonomy and predictability. The constraint on consciousness correlates is an emergent property of structured networks that appears only above a critical system size. Code and data: github.com/brian-mwirigi/consim.

**Keywords:** consciousness correlates, integrated information, multi-agent systems, emergent self-prediction, network topology, scaling laws, dimensionality collapse

---

## 1. Introduction

Theories of consciousness increasingly invoke multiple measurable correlates — integration (Tononi, 2004; Tononi et al., 2016), temporal stability (Baars, 1988), causal efficacy (Seth, 2014; Seth & Bayne, 2022), and self-modeling (Graziano, 2013; Cleeremans, 2011) — assuming these can be independently optimized. The Machine Consciousness Hypothesis (MCH) proposes that consciousness is a substrate-free functional property of computational systems capable of second-order perception, arising when groups of local observers exchange lossy predictive messages in a universal self-organizing environment (Fitz, 2025). The MCH defines four measurable correlates: integration ($\Phi$), reflexivity ($R$), temporal persistence ($T$), and causal efficacy ($E$).

A critical open question is whether these correlates are genuinely independent. If the network structure that enables them also constrains their joint achievability, then counting independent correlates overcounts the degrees of freedom available to any networked system.

We address this computationally using *consim*, a multi-agent simulation that places $N = \text{size}^2$ agents on a toroidal grid. Each agent maintains an 8-dimensional internal state and a learned weight matrix. Every tick, agents broadcast transformed state messages, receive noisy versions of their neighbors' messages, update their states, and learn via gradient descent on neighbor prediction error. Self-prediction — cosine similarity between an agent's broadcast and its subsequent state — is never explicitly trained. If it rises above chance, the agent has developed an implicit model of its own dynamics as a side effect of modeling its neighbors. This operationalizes Fitz's notion of reflexivity emerging from predictive communication.

Previous work on integrated information and phase transitions has focused on Ising-model systems (Khajehabdollahi, 2018; Luitle, 2023) or coupled oscillators (Abrego & Zaikin, 2019), studying how $\Phi$ alone varies with temperature or coupling strength. Related theoretical frameworks propose morphospaces for consciousness (Arsiwalla et al., 2023) or spectral measures of emergence (Bailey & Schneider, 2025). The MCH itself (Fitz, 2025) is a purely theoretical proposal with no empirical implementation.

Our contribution is distinct from all prior work in four respects: (1) we measure five consciousness correlates simultaneously, not just $\Phi$; (2) we vary network topology rather than temperature, using structured and random graphs; (3) we demonstrate that these correlates are not independent on high-connectivity networks, collapsing to a single principal component; and (4) we characterize a scaling law for this collapse, identifying a phase-like transition between $N = 324$ and $N = 576$ agents on structured grids. No previous work has shown dimensionality collapse of consciousness correlates, the T-E autonomy-predictability trade-off, or the functional identity E $\equiv$ $\Phi$ on structured networks.

---

## 2. Methods

### 2.1 Agent Architecture

Each agent $i$ has state $s_i \in \mathbb{R}^8$ and weights $W_i \in \mathbb{R}^{8 \times 8}$, initialized randomly. Every tick, the following operations occur:

1. **Broadcast:** $m_i = \sigma(W_i \cdot s_i)$ where $\sigma$ is a configurable activation function
2. **Noise:** $\tilde{m}_i = m_i + \varepsilon_i$, $\varepsilon_i \sim \mathcal{N}(0, \sigma_{\text{noise}}^2 I)$
3. **Receive:** $r_i = \frac{1}{K} \sum_{j \in \mathcal{N}(i)} \tilde{m}_j$
4. **Update:** $s_i' = \sigma\!\big(\alpha \, s_i + (1 - \alpha) \, r_i + \text{drive}\big)$, where drive $\sim \mathcal{N}(0, \delta^2 I)$, $\delta = 0.02$
5. **Learn:** $W_i \leftarrow W_i - \eta \, \nabla_{W_i} \frac{1}{K}\sum_{j \in \mathcal{N}(i)} \|m_i - s_j'\|^2$

State persistence $\alpha = 0.3$. Learning rate $\eta = 0.003$. The learning rule minimizes neighbor prediction error: each agent tries to make its broadcast message match its neighbors' next states. Self-prediction — $\cos(m_i, s_i')$ — is never in the gradient.

### 2.2 Topologies

Five graph structures are tested:

- **von Neumann** ($K = 4$): cardinal neighbors on a toroidal grid
- **Moore** ($K = 8$): cardinal + diagonal neighbors
- **Hexagonal** ($K = 6$): hexagonal lattice
- **Random** ($K = 4$): uniformly sampled random graph
- **Small-world**: von Neumann + 10% edge rewiring (Watts & Strogatz, 1998)

### 2.3 MCH Metrics

We measure five quantities per agent per tick:

- **Self-prediction (self):** $\cos(m_i, s_i')$ — how well the broadcast predicts the next state
- **Integration ($\Phi$):** Normalized difference between joint and partitioned neighborhood prediction residuals: $\Phi = (\text{parts residual} - \text{joint residual}) / \text{parts residual}$. This is a simplified proxy for Tononi's integrated information (Tononi, 2004)
- **Reflexivity ($R$):** $\cos(m_i, s_i') - \text{mean}_{j \in \mathcal{N}(i)} \cos(m_i, s_j')$. Positive $R$ means the agent predicts itself better than neighbors
- **Temporal persistence ($T$):** $T = \text{clip}(1 - \sqrt{\text{Var}_{\text{ema}}(\text{self})}, 0, 1)$. Stability of self-prediction over time
- **Causal efficacy ($E$):** $\cos(\Delta s_i, \Delta s_i^{\text{self}})$ where $\Delta s_i^{\text{self}} = \sigma(\alpha s_i + \xi_i) - s_i$ is the counterfactual state change without neighbor input

### 2.4 Activation Functions

Three activations are tested to establish universality:

- **tanh** (default): $f(x) = \tanh(x)$, range $[-1, 1]$
- **Linear:** $f(x) = \text{clip}(x, -1, 1)$, range $[-1, 1]$
- **ReLU:** $f(x) = \text{clip}(\max(0, x), 0, 1)$, range $[0, 1]$

All activation-dependent computations (message transform, state update, learning gradient) use the configured function and its derivative.

### 2.5 Experimental Design

| Experiment | Runs | Seeds | Sizes | Topologies | Noises | Activation |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Primary (size 24) | 1,250 | 50 | 24 | all 5 | all 5 | tanh |
| Primary (size 12) | 1,250 | 50 | 12 | all 5 | all 5 | tanh |
| Size 48 + metrics | 250 | 10 | 48 | all 5 | all 5 | tanh |
| Scaling (sizes 12–36) | 1,000 | 10 | 12, 18, 24, 36 | all 5 | all 5 | tanh |
| Null model (lr = 0) | 250 | 10 | 48 | all 5 | all 5 | tanh |
| ReLU universality | 90 | 10 | 24 | vn, moore, random | 3 | ReLU |
| Linear universality | 90 | 10 | 24 | vn, moore, random | 3 | linear |
| **Total** | **4,180** | | | | | |

All runs: 1,000 ticks. Noise levels: $\sigma_{\text{noise}} \in \{0.04, 0.08, 0.12, 0.20, 0.30\}$. An optional Conway's Game of Life (B3/S23) substrate layer provides structured "base reality" input, tested separately at size 24.

---

## 3. Results

### 3.1 Structural Effects of Topology

**Finding 1: Random graph noise immunity.** On random topologies, all metrics remain invariant across a 7.5$\times$ noise range ($\sigma = 0.04$ to $0.30$): mean self-prediction varies by $< 0.001$, $\Phi$ by $< 0.007$, $R$ by $0.001$, $T$ by $0.0006$. This is analytically explained by double application of the law of large numbers: noise cancels over $K$ independent neighbors (received signal) and over $N$ agents (population mean). Grid topologies lack this because spatially adjacent neighbors are correlated.

**Finding 2: Noise amplification on high-$K$ grids.** Moore topology ($K = 8$) shows monotonic increase in self-prediction with noise: $+0.018$ at size 24, $+0.031$ at size 12. The mechanism is symmetry-breaking: noise decorrelates adjacent agents whose inputs heavily overlap, creating diversity that enriches the learning gradient. This requires both noise tolerance (high $K$) and redundancy to break (spatial structure).

**Finding 3: Small-world $\Phi$ dissociation.** Small-world shortcuts boost $\Phi$ by 13–20% relative to von Neumann while leaving self-prediction unchanged (within 0.002). All 50 seeds show positive $\Phi$ boost — zero exceptions. Long-range shortcuts introduce uncorrelated neighbors that worsen individual (parts) prediction while leaving integrated (joint) prediction robust, mechanically increasing $\Phi$.

### 3.2 The Autonomy-Predictability Trade-Off

**Finding 4: Self-determination hurts self-knowledge on high-$K$ grids.** On Moore grids ($K = 8$), temporal persistence and causal efficacy predict self-prediction in opposite directions:

| Correlation | Moore | Hex | von Neumann | Random | Small-world |
|---|:-:|:-:|:-:|:-:|:-:|
| $r(T, \text{self})$ | **+0.70** | +0.52 | +0.03 | +0.12 | +0.22 |
| $r(E, \text{self})$ | **−0.71** | −0.71 | −0.12 | −0.14 | +0.06 |
| $r(T, E)$ | **−0.82** | −0.74 | +0.07 | +0.32 | −0.08 |

Agents whose trajectories are externally driven (low $E$) but whose self-models are stable over time (high $T$) achieve the best self-prediction. The mechanism: on a Moore grid with 8 neighbors, the received signal is a smoothed, slowly-varying average. Agents driven by this signal have predictable futures. Agents driven by their own nonlinear dynamics are harder for themselves to forecast. Self-determination is noise from the self-model's perspective.

**Null model confirmation.** A null model with learning disabled ($\eta = 0$) confirms this is a topological invariant, not a learning artifact:

| Topology | $r(T,E)$ learning | $r(T,E)$ null | $\Delta$ |
|---|:-:|:-:|:-:|
| Moore | −0.82 | −0.79 | 0.03 |
| Hex | −0.74 | −0.70 | 0.04 |
| von Neumann | +0.07 | +0.11 | 0.04 |

The trade-off is geometric: the state update dynamics and topology alone produce it. Learning at $\eta = 0.003$ over 1,000 ticks changes self-prediction by $< 0.1\%$.

**Finding 5: Activation universality.** The T-E anti-correlation survives three activation functions:

| Activation | Moore $r(T,E)$ | Moore $r(E, \text{self})$ |
|---|:-:|:-:|
| tanh | −0.82 | −0.71 |
| Linear | −0.77 | −0.53 |
| ReLU | −0.76 | −0.61 |

The universality proves the trade-off arises from network structure, not nonlinearity. No property specific to tanh is used: the theorem (Section 4) requires only that $f$ is bounded.

### 3.3 Functional Identity: $E \equiv \Phi$ on Structured Graphs

**Finding 6:** Causal efficacy and integration are functionally identical on grids but decouple on random graphs:

| Topology | $r(E, \Phi)$ | Linear slope | RMSE |
|---|:-:|:-:|:-:|
| von Neumann | +0.998 | 0.327 | 0.003 |
| Moore | +0.994 | 0.349 | 0.005 |
| Hex | +0.996 | 0.337 | 0.004 |
| Small-world | +0.984 | 0.321 | 0.007 |
| Random | +0.178 | 0.008 | — |

On structured networks, $\Phi$ measures causal self-determination, not topology-agnostic information integration. The equivalence breaks on random graphs because independent neighbor sampling destroys the link between an agent's self-determination and its neighborhood prediction structure. This has implications for IIT: $\Phi$ partitions may conflate integration with autonomy in any system with spatial correlations.

### 3.4 Dimensionality Collapse

**Finding 7:** PCA on the five MCH metrics reveals that on Moore grids, the metrics collapse to essentially one dimension:

| Activation | PC1 | PC1+PC2 | Dims for 95% |
|---|:-:|:-:|:-:|
| tanh | **82.5%** | 94.8% | 3 |
| Linear | **71.8%** | 89.6% | 3 |
| ReLU | **62.1%** | 82.8% | 4 |

The PC1 loadings form a consistent "autonomy-predictability axis":

| Metric | tanh | Linear | ReLU |
|---|:-:|:-:|:-:|
| self | −0.38 | −0.27 | −0.35 |
| $\Phi$ | **+0.48** | **+0.52** | **+0.52** |
| $R$ | +0.45 | +0.45 | +0.27 |
| $T$ | **−0.44** | **−0.44** | **−0.48** |
| $E$ | **+0.49** | **+0.52** | **+0.56** |

$\{\Phi, R, E\}$ load positive (autonomy pole); $\{\text{self}, T\}$ load negative (predictability pole). On random topologies, 4 components are needed for 95% variance — no collapse occurs. The consciousness correlates are not independent on structured networks: they are projections of a single underlying variable.

### 3.5 Scaling Law

**Finding 8 (headline result):** The dimensionality collapse follows a logarithmic scaling law across five grid sizes:

| Size | $N$ agents | PC1 (Moore) | PC1 (Random) | $r(T,E)$ Moore |
|---|:-:|:-:|:-:|:-:|
| 12 | 144 | 48.8% | 49.5% | −0.11 |
| 18 | 324 | 55.9% | 38.7% | −0.25 |
| 24 | 576 | 67.4% | 36.3% | −0.70 |
| 36 | 1,296 | 75.1% | 47.3% | −0.78 |
| 48 | 2,304 | 82.5% | 39.6% | −0.82 |

The fit is:

$$\text{PC1} = -0.136 + 0.124 \times \ln(N), \quad R^2 = 0.987$$

Random topologies show no scaling (PC1 $\approx 40\%$ at all sizes). The inflection occurs between $N = 324$ and $N = 576$:

| Transition | $\Delta$PC1 | $\Delta r(T,E)$ |
|---|:-:|:-:|
| 144 → 324 | +0.071 | −0.14 |
| **324 → 576** | **+0.115** | **−0.45** |
| 576 → 1,296 | +0.077 | −0.08 |
| 1,296 → 2,304 | +0.074 | −0.04 |

Below $\sim$300 agents, the five metrics are effectively independent (PC1 $\approx$ 49%, $r(T,E) \approx -0.11$). Above $\sim$600, they are locked into a single axis (PC1 $> 67\%$, $r(T,E) < -0.70$). The constraint on consciousness correlates emerges with scale on structured topologies — a phase-like transition from unconstrained to locked.

**Full topology comparison across sizes:**

| Topology | PC1 @12 | PC1 @18 | PC1 @24 | PC1 @36 | PC1 @48 |
|---|:-:|:-:|:-:|:-:|:-:|
| von Neumann | 39.4% | 46.1% | 41.5% | 48.6% | 45.8% |
| Moore | 48.8% | 55.9% | 67.4% | 75.1% | 82.5% |
| Hex | 47.7% | 57.9% | 57.2% | 71.6% | 78.4% |
| Random | 49.5% | 38.7% | 36.3% | 47.3% | 39.6% |
| Small-world | 41.3% | 42.8% | 41.2% | 41.3% | 48.7% |

Moore and hex diverge sharply from the others with scale. The collapse is specific to high-$K$ structured topologies.

### 3.6 Additional Findings

**Simpson's paradox.** On Moore and hex, the within-noise $\Phi$-self correlation is positive ($r \approx +0.48$) but the pooled correlation is negative ($r = -0.28$). Noise simultaneously increases self-prediction and decreases $\Phi$, creating a confound that reverses the sign. The paradox replicates at sizes 12 and 24 and is specific to high-connectivity grids.

**Game of Life substrate.** Adding a Conway's Game of Life substrate layer (Berlekamp et al., 1982) triples mean self-prediction ($0.19 \to 0.57$ at size 24). $E$ drops ($0.66 \to 0.44$), confirming agents become more externally driven. $T$ increases ($0.66 \to 0.80$), consistent with the T-E trade-off: structured external signal → lower $E$ → higher $T$ → better self-prediction.

---

## 4. Theoretical Analysis

### 4.1 The Autonomy-Predictability Theorem

**Theorem.** Consider $N$ agents on a $K$-regular graph with dynamics

$$s_i' = f\!\big(\alpha \, s_i + (1 - \alpha) \, r_i + \xi_i\big)$$

where $f$ is any bounded activation ($f: \mathbb{R} \to [-B, B]$), $r_i = \frac{1}{K}\sum_{j \in \mathcal{N}(i)} (g(s_j) + \varepsilon_j)$ is the received signal, $\xi_i \sim \mathcal{N}(0, \delta^2 I)$ is drive noise, and $\varepsilon_j \sim \mathcal{N}(0, \sigma^2 I)$ is channel noise. Define causal efficacy $E_i = \cos\!\big(s_i' - s_i, \; f(\alpha s_i + \xi_i) - s_i\big)$ and self-prediction $\text{self}_i = \cos(m_i, s_i')$.

Then for $K$ sufficiently large and neighbor correlations $\rho > 0$, $\text{Cov}(E_i, \text{self}_i) < 0$ across the agent population.

**Proof sketch.**

*Step 1.* Decompose the pre-activation: $z_i = \underbrace{\alpha s_i + \xi_i}_{z_i^{\text{self}}} + \underbrace{(1-\alpha) r_i}_{z_i^{\text{ext}}}$.

*Step 2.* On a regular grid with neighbor correlation $\rho > 0$, the temporal variance of the external component is:
$$\text{Var}_t(r_i) = \frac{\text{Var}(m)}{K}\big(1 + \rho(K-1)\big) + \frac{\sigma^2}{K}$$
For Moore grids ($K = 8$, $\rho \approx 0.3$): $\text{Var}_t(r_i) \approx 0.4 \cdot \text{Var}(m)$.

*Step 3.* The self-driven variance $\text{Var}_t(z_i^{\text{self}}) = \alpha^2 \text{Var}_t(s_i) + \delta^2$ is $K$-independent.

*Step 4.* For large $K$ with $\rho > 0$, $\text{Var}(z^{\text{ext}}) < \text{Var}(z^{\text{self}})$. Agents dominated by the external component (low $E$) have smoother trajectories. Smoother trajectories are easier to self-predict. Therefore $\text{Cov}(E, \text{self}) < 0$.

*Step 5.* On random graphs ($\rho = 0$), the external signal is averaged noise from independent sources — unreliable rather than predictable. The negative covariance does not emerge.

*Step 6.* The proof uses only: (i) $f$ is bounded, (ii) $r_i$ is an average of $K$ messages, (iii) $\rho > 0$. No property specific to tanh is used. $\square$

### 4.2 Random Graph Fixed Point

The random graph's noise immunity follows from a closed self-stabilizing loop. On random graphs, each agent's $K$ neighbors are independent uniform samples from the population. The received signal $r_i \approx \bar{m} + \mathcal{N}(0, \sigma^2/K \cdot I)$ converges to the population mean $\bar{m}$ by the law of large numbers. As noise $\sigma$ increases, zero-mean perturbations cancel across both the $K$ neighbors (received signal) and $N$ agents (population mean). The expected state, gradient direction, and all macroscopic statistics are invariant (see Appendix for full derivation).

### 4.3 Small-World $\Phi$ Dissociation

Long-range shortcuts connect agents to distant, uncorrelated partners. Each shortcut worsens the parts prediction (the distant agent is a poor individual predictor) while leaving the joint prediction robust (the mean is dominated by local neighbors). Since $\Phi \propto (\text{parts} - \text{joint}) / \text{parts}$, shortcuts mechanically increase $\Phi$ without changing self-prediction (the long-range message is $\sim$10% of the input and zero-mean in direction).

---

## 5. Discussion

### 5.1 The Autonomy-Predictability Principle

The central finding of this work inverts a deep assumption in the consciousness literature. IIT (Tononi, 2004; Tononi et al., 2016), the MCH (Fitz, 2025), attention schema theory (Graziano, 2013), and most autonomy-based frameworks assume self-determination enables self-knowledge. Our data show the opposite on high-connectivity networks: agents most in control of their own trajectories are the worst at predicting themselves ($r(E, \text{self}) = -0.71$). The mechanism is simple: external determination is regularizing. When an agent's future state is driven by the average of 8 predictable neighbors, that future is easy to forecast. When it is driven by the agent's own nonlinear dynamics, it is chaotic and hard to predict.

The activation universality (Finding 5), null model confirmation, and formal theorem (Section 4.1) collectively establish this as a structural law of networked systems, not an artifact of any particular implementation choice.

### 5.2 Implications for Integrated Information Theory

The $E \equiv \Phi$ identity (Finding 6) has direct implications for IIT. On structured networks, $\Phi$ partitions — designed to measure information integration — are actually measuring how much an agent's trajectory is self-determined vs. externally driven. The correlation $r > 0.994$ means that on any system with spatial correlations (including the brain), $\Phi$ may conflate integration with autonomy. The decoupling on random graphs ($r = 0.18$) shows this is not a universal property of $\Phi$ but a consequence of structured connectivity.

### 5.3 Dimensionality Collapse and Overcounting

The dimensionality collapse (Finding 7) shows that on structured networks, the four MCH correlates ($\Phi$, $R$, $T$, $E$) are not independent indicators of consciousness. They are projections of a single underlying variable: the agent's position on the autonomy-predictability axis. A theory that treats them as separate lines of evidence overcounts the degrees of freedom. Adding more correlates does not add more information — they measure the same thing through different lenses.

This result is topology-dependent: on random networks, where neighbor signals are independent, the five metrics retain their independence (4 components for 95% variance). The constraint is not universal — it is an emergent property of structured connectivity.

### 5.4 The Scaling Law as Phase Transition

The scaling law (Finding 8) answers the question that motivated this project: does anything *emerge* from scale? The answer is that the **constraint** on consciousness correlates emerges from scale. Small structured systems ($N < 300$) are unconstrained — five metrics, five degrees of freedom. Large ones ($N > 600$) are forced into a single trade-off axis. The transition sharpens between $N = 324$ and $N = 576$, with the largest gains in both PC1 ($+0.115$) and $r(T,E)$ ($-0.45$) occurring in this interval.

The logarithmic fit ($R^2 = 0.987$) predicts PC1 $\to 1.0$ (complete collapse) at $N \approx 8{,}000$ agents, though the curve must saturate before that point. Random systems never undergo this transition, confirming that both structure and scale are necessary.

This is analogous to phase transitions in statistical physics: below a critical system size, macroscopic constraints do not manifest. Above it, microscopic degrees of freedom become coupled by the network geometry. The analogy with Khajehabdollahi's (2018) finding — that $\Phi$ undergoes a phase transition at critical temperature in Ising models — is intriguing, but our result is qualitatively different: we show the emergence of constraint *across* correlates, not a transition of a single measure.

### 5.5 Related Work

**Integrated Information Theory.** Tononi's IIT (Tononi, 2004; Tononi et al., 2016; Albantakis et al., 2023) proposes $\Phi$ as the primary measure of consciousness. Computational studies have explored $\Phi$ in Ising models (Khajehabdollahi, 2018), coupled repressilators (Abrego & Zaikin, 2019), fish schools (Niizato et al., 2024), and neuropercolation models (Luitle, 2023). These works study $\Phi$ in isolation. We measure $\Phi$ alongside four other correlates and show it is not independent of them on structured networks.

**Predictive processing.** The free energy principle (Friston, 2010) and predictive coding (Rao & Ballard, 1999; Clark, 2013) propose that neural systems minimize prediction error. Our agents implement a distributed version: each minimizes neighbor prediction error. The emergent self-prediction is consistent with predictive processing accounts of self-awareness (Seth, 2014; Seth & Bayne, 2022; Hohwy, 2013).

**Network topology and dynamics.** The importance of topology for collective dynamics is well established (Watts & Strogatz, 1998; Barabási & Albert, 1999; Newman, 2003). We extend this to consciousness-relevant metrics, showing that topology controls not just individual metrics but their joint degrees of freedom.

**Multi-agent emergence.** Arsiwalla et al. (2023) propose a "morphospace of consciousness" mapping complexity dimensions. Bailey and Schneider (2025) propose spectral measures of epistemic emergence with topological constraints. Both are theoretical. Our work provides the first simulation-based evidence that consciousness correlates collapse to a low-dimensional manifold on structured networks.

**Machine consciousness.** Fitz (2025) proposes the MCH as a theoretical program. Our work constitutes the first computational implementation and empirical test, finding that one of the MCH's implicit assumptions — that $\Phi$, $R$, $T$, $E$ provide independent evidence — does not hold on the networks the MCH proposes to study.

### 5.6 Limitations

Our agents use simple linear+activation architectures, not the transformers Fitz envisions. Our $\Phi$ approximation is not the full IIT measure (computing exact $\Phi$ is NP-hard; Tegmark, 2016). We do not claim consciousness emergence — our system is too simple for that. The Game of Life substrate coupling is additive rather than perceptual. We test five grid sizes; the scaling law extrapolation beyond $N = 2{,}304$ is unverified.

What we do show is that the *structural logic* of self-prediction from predictive communication is tractable, activation-independent, and produces non-trivial, topology-dependent phenomena that constrain which combinations of consciousness correlates are jointly achievable.

---

## 6. Conclusion

We have shown that on structured networks, five consciousness-relevant metrics — self-prediction, $\Phi$, $R$, $T$, and $E$ — are not independent. They collapse to a single "autonomy-predictability axis" whose strength follows a logarithmic scaling law with system size. This collapse constitutes a network-theoretic uncertainty principle: on $K$-regular graphs with correlated neighbors, an agent cannot simultaneously maximize causal autonomy and self-predictability. The constraint emerges above a critical system size ($N \approx 300$–$600$ on Moore grids) and is absent on random topologies at any size.

Three implications follow:

1. **For consciousness theory:** Any framework that treats integration, reflexivity, persistence, efficacy, and self-knowledge as independent correlates overcounts degrees of freedom on structured networks. A single number — the agent's position on the autonomy-predictability axis — captures 62–83% of all variation.

2. **For IIT:** On structured networks, $\Phi$ and causal efficacy are functionally identical ($r > 0.994$). $\Phi$ measures self-determination, not information integration, when neighbors are correlated.

3. **For the MCH:** We provide the first empirical implementation of Fitz's framework and find that its four correlates cannot be simultaneously maximized — they lie on a Pareto frontier determined by network topology.

Code, data (4,180 runs), and all analysis scripts are publicly available at github.com/brian-mwirigi/consim under CC BY 4.0.

---

## References

Abrego, L., & Zaikin, A. (2019). Integrated information as a measure of cognitive processes in coupled genetic repressilators. *Entropy*, 21(4), 382.

Albantakis, L., Barbosa, L., Findlay, G., Grasso, M., Haun, A. M., Marshall, W., ... & Tononi, G. (2023). Integrated information theory (IIT) 4.0: formulating the properties of phenomenal existence in physical terms. *PLoS Computational Biology*, 19(10), e1011465.

Arsiwalla, X. D., Solé, R., Moulin-Frier, C., & Herreros, I. (2023). The morphospace of consciousness: Three kinds of complexity for minds and machines. *NeuroSci*, 4(2), 79–102.

Baars, B. J. (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press.

Bailey, M., & Schneider, S. (2025). When wholes resist decomposition: A spectral measure of epistemic emergence. *Preprint*.

Barabási, A.-L., & Albert, R. (1999). Emergence of scaling in random networks. *Science*, 286(5439), 509–512.

Berlekamp, E. R., Conway, J. H., & Guy, R. K. (1982). *Winning Ways for Your Mathematical Plays*. Academic Press.

Clark, A. (2013). Whatever next? Predictive brains, situated agents, and the future of cognitive science. *Behavioral and Brain Sciences*, 36(3), 181–204.

Cleeremans, A. (2011). The radical plasticity thesis: how the brain learns to be conscious. *Frontiers in Psychology*, 2, 86.

Fitz, S. (2025). Testing the Machine Consciousness Hypothesis. *arXiv:2512.01081*.

Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127–138.

Graziano, M. S. A. (2013). *Consciousness and the Social Brain*. Oxford University Press.

Hohwy, J. (2013). *The Predictive Mind*. Oxford University Press.

Khajehabdollahi, S. (2018). Phase transitions of integrated information in the generalized Ising model of the brain. *Master's thesis, Western University*.

Luitle, T. (2023). Synchronisation through integrated information in neuropercolation models near criticality. *Master's thesis*.

Newman, M. E. J. (2003). The structure and function of complex networks. *SIAM Review*, 45(2), 167–256.

Niizato, T., Sakamoto, K., Mototake, Y., Murakami, H., Tomaru, T., Hoshika, Y., & Fukushima, T. (2024). Information structure of heterogeneous criticality in a fish school. *Scientific Reports*, 14, 23503.

Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. *Nature Neuroscience*, 2(1), 79–87.

Seth, A. K. (2014). A predictive processing theory of sensorimotor contingencies: Explaining the puzzle of perceptual presence and its absence in synesthesia. *Cognitive Neuroscience*, 5(2), 97–118.

Seth, A. K., & Bayne, T. (2022). Theories of consciousness. *Nature Reviews Neuroscience*, 23, 439–452.

Tegmark, M. (2016). Improved measures of integrated information. *PLoS Computational Biology*, 12(11), e1005123.

Tononi, G. (2004). An information integration theory of consciousness. *BMC Neuroscience*, 5(42).

Tononi, G., Boly, M., Massimini, M., & Koch, C. (2016). Integrated information theory: from consciousness to its physical substrate. *Nature Reviews Neuroscience*, 17(7), 450–461.

Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of 'small-world' networks. *Nature*, 393(6684), 440–442.

---

## Appendix A: Reproducibility

All experiments can be reproduced with the following commands:

```bash
pip install numpy matplotlib Pillow

# Primary sweeps (sizes 12 and 24)
python run.py --sweep --sweep-seeds 1-50 \
  --sweep-topos von_neumann,moore,hex,random,small_world \
  --sweep-noises 0.04,0.08,0.12,0.20,0.30 \
  --ticks 1000 --size 24 --sweep-csv sweep_1250.csv

python run.py --sweep --sweep-seeds 1-50 \
  --sweep-topos von_neumann,moore,hex,random,small_world \
  --sweep-noises 0.04,0.08,0.12,0.20,0.30 \
  --ticks 1000 --size 12 --sweep-csv sweep_size12.csv

# Size 48 with R, T, E metrics
python run.py --sweep --sweep-seeds 1-10 \
  --sweep-topos von_neumann,moore,hex,random,small_world \
  --sweep-noises 0.04,0.08,0.12,0.20,0.30 \
  --ticks 1000 --size 48 --sweep-csv sweep_size48.csv

# Null model (learning disabled)
python run.py --sweep --sweep-seeds 1-10 \
  --sweep-topos von_neumann,moore,hex,random,small_world \
  --sweep-noises 0.04,0.08,0.12,0.20,0.30 \
  --ticks 1000 --size 48 --lr 0 --sweep-csv sweep_null.csv

# Scaling sweeps
python run.py --sweep --sweep-seeds 1-10 \
  --sweep-topos von_neumann,moore,hex,random,small_world \
  --sweep-noises 0.04,0.08,0.12,0.20,0.30 \
  --ticks 1000 --size 12 --sweep-csv sweep_size12_full.csv

python run.py --sweep --sweep-seeds 1-10 \
  --sweep-topos von_neumann,moore,hex,random,small_world \
  --sweep-noises 0.04,0.08,0.12,0.20,0.30 \
  --ticks 1000 --size 18 --sweep-csv sweep_size18_full.csv

python run.py --sweep --sweep-seeds 1-10 \
  --sweep-topos von_neumann,moore,hex,random,small_world \
  --sweep-noises 0.04,0.08,0.12,0.20,0.30 \
  --ticks 1000 --size 24 --sweep-csv sweep_size24_full.csv

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
```

Raw data (10 CSV files, 4,180 runs total) available at github.com/brian-mwirigi/consim.
