# Topology Determines Emergent Self-Prediction in Multi-Agent Communication Networks

**Brian Mwirigi**

---

## Abstract

We present *consim*, a multi-agent simulation studying how self-prediction emerges from lossy predictive communication on graph topologies. Inspired by the Machine Consciousness Hypothesis (Fitz, 2025), the system implements distributed agents that communicate through noisy channels and learn to predict their neighbors' states — never their own. We measure four consciousness-relevant quantities from the MCH framework: integration (Φ), reflexivity (R), temporal persistence (T), and causal efficacy (E). Across 5,500 simulations on five topologies, five noise levels, and three grid sizes, we find three robust structural effects and one striking dissociation: (1) random graphs are noise-immune fixed points, analytically explained by the law of large numbers; (2) high-connectivity grids exhibit noise amplification of self-prediction, driven by symmetry-breaking decorrelation; (3) small-world shortcuts boost integration (Φ) by 20% without affecting self-prediction; and (4) on high-connectivity grids, self-determination (E) anti-correlates with self-prediction at r = −0.71 while temporal stability (T) correlates at r = +0.70, with T and E anti-correlated at r = −0.82 — agents most in control of their own trajectories are the worst at predicting themselves. A null model with learning disabled (lr=0) confirms this dissociation is a topological invariant, not a learning artifact. We additionally find that E and Φ are functionally identical on structured graphs (r > 0.994) but decouple on random topologies (r = 0.18), and identify a Simpson's paradox where Φ-self correlations reverse sign between within-noise and pooled analyses. Adding a Conway's Game of Life substrate layer as base reality triples self-prediction scores while reducing causal efficacy. All structural findings replicate across grid sizes and are supported by closed-form analytical derivations. Code and data are publicly available at github.com/brian-mwirigi/consim.

---

## 1. Introduction

The Machine Consciousness Hypothesis (MCH) proposes that consciousness is a substrate-free functional property of computational systems capable of second-order perception, arising when groups of local observers exchange lossy predictive messages in a universal self-organizing environment (Fitz, 2025). The MCH defines four measurable correlates: integration (Φ), reflexivity (R), temporal persistence (T), and causal efficacy (E).

We test a simplified version of this proposal computationally. Our simulation, *consim*, places N = size² agents on a toroidal grid. Each agent maintains an 8-dimensional internal state and a learned weight matrix. Every tick, agents: (1) transform their state into a broadcast message via $m_i = \tanh(W_i \cdot s_i)$; (2) transmit with additive Gaussian noise; (3) receive the mean of neighbors' noisy messages; (4) update their state as $s_i' = \tanh(\alpha s_i + (1-\alpha) r_i + \text{drive})$; and (5) learn via gradient descent on neighbor prediction error.

Self-prediction — cosine similarity between an agent's broadcast and its subsequent state — is never explicitly trained. If it rises above chance, the agent has developed an implicit model of its own dynamics as a side effect of modeling its neighbors. This operationalizes Fitz's notion of reflexivity emerging from predictive communication.

We extend the base model with two features drawn from the MCH: (a) a Game of Life substrate layer providing structured "base reality" input, and (b) measurements of all four MCH correlates (Φ, R, T, E) alongside self-prediction scores.

## 2. Methods

**Architecture.** Each agent $i$ has state $s_i \in \mathbb{R}^8$ and weights $W_i \in \mathbb{R}^{8 \times 8}$. The learning rule minimizes $\frac{1}{K}\sum_{j \in \mathcal{N}(i)} \|m_i - s_j'\|^2$ via gradient descent ($\eta = 0.003$). State persistence $\alpha = 0.3$.

**Topologies.** Five graph structures: von Neumann (K=4), Moore (K=8), hexagonal (K=6), random (K=4, uniformly sampled), and small-world (von Neumann + 10% rewiring).

**MCH Metrics.** Integration (Φ): normalized difference between joint and partitioned neighborhood prediction residuals. Reflexivity (R): self-prediction score minus mean neighbor-prediction score — positive R means the agent is better at predicting itself than others. Temporal persistence (T): stability of self-prediction over time, computed via exponential moving average variance: $T = \text{clip}(1 - \sqrt{\text{Var}_{\text{ema}}}, 0, 1)$. Causal efficacy (E): cosine similarity between the actual state change and the counterfactual self-only state change (no neighbor input).

**Game of Life Substrate.** An optional Conway's Game of Life (B3/S23) grid evolves at each tick. Agents observe their local GoL cell state and neighborhood density, receiving this as an additive signal scaled by coupling parameter $\gamma = 0.1$.

**Experiments.** 5,500 primary runs: 50 seeds × 5 topologies × 5 noise levels ($\sigma \in \{0.04, 0.08, 0.12, 0.20, 0.30\}$), at sizes 12 and 24 (1,250 each). Separate size-48 sweep (250 runs) with all metrics. Null model sweep (250 runs, lr=0) to test whether findings require learning. GoL experiments at size 24.

## 3. Results

**Finding 1: Random graph noise immunity.** On random topologies, all metrics remain invariant across a 7.5× noise range ($\sigma = 0.04$ to $0.30$): mean self-prediction varies by <0.001, Φ by <0.007. This is analytically explained by double application of the law of large numbers: noise cancels over K independent neighbors (received signal) and over N agents (population mean). Grid topologies lack this because spatially adjacent neighbors are correlated.

**Finding 2: Noise amplifies self-prediction on high-K grids.** Moore topology (K=8) shows monotonic increase in self-prediction with noise: +0.018 at size 24, +0.031 at size 12. The mechanism is symmetry-breaking: noise decorrelates adjacent agents whose inputs heavily overlap, creating diversity that enriches the learning gradient. This requires both noise tolerance (high K) and redundancy to break (spatial structure), explaining why random graphs (no redundancy) and von Neumann (K too low) show no effect.

**Finding 3: Small-world Φ dissociation.** Small-world shortcuts boost Φ by 13–20% relative to von Neumann while leaving self-prediction unchanged (within 0.002). All 50 seeds show positive Φ boost — zero exceptions. Long-range shortcuts introduce uncorrelated neighbors that worsen individual (parts) prediction while leaving integrated (joint) prediction robust, mechanically increasing $\Phi = (\text{parts} - \text{joint}) / \text{parts}$.

**Finding 4: Self-determination hurts self-knowledge on high-K grids.** On Moore grids (K=8), temporal persistence (T) correlates with self-prediction at r = +0.70, while causal efficacy (E) anti-correlates at r = −0.71. The two metrics are anti-correlated with each other: r(T, E) = −0.82. Agents whose trajectories are externally driven (low E) but whose self-models are stable over time (high T) achieve the best self-prediction. A null model with learning disabled (lr=0) confirms this is a **topological invariant**: the correlations persist at nearly identical strength (r(T,E) = −0.79) without any weight updates. The mechanism is structural: 8 neighbors average into a predictable input signal. Agents driven by that signal have predictable futures. Agents driven by their own nonlinear dynamics are harder for themselves to forecast. Self-determination is noise from the self-model's perspective. Hex (K=6) shows the same pattern more weakly. Von Neumann (K=4) and random show no effect.

**Finding 5: E ≡ Φ on structured graphs.** Causal efficacy and integration correlate at r > 0.994 on all grid topologies (von Neumann, Moore, hex, small-world) but decouple on random graphs (r = 0.18). Φ is not measuring topology-agnostic information integration; on structured networks, it measures causal self-determination. The equivalence breaks on random graphs because independent neighbor sampling destroys the link between an agent's self-determination and its neighborhood prediction structure.

**Finding 6: Simpson's paradox in Φ-self relationship.** On Moore and hex, the within-noise Φ-self correlation is positive (r ≈ +0.48 on Moore) but the pooled correlation is negative (r = −0.28). Noise simultaneously increases self-prediction and decreases Φ, creating a confound that reverses the sign. The paradox replicates at sizes 12 and 24 and is specific to high-connectivity grids.

**GoL substrate effect.** Adding a Game of Life layer triples mean self-prediction (0.19 → 0.57 at size 24, averaged over 10 seeds). Causal efficacy (E) drops (0.66 → 0.44), confirming agents become more externally driven. Temporal persistence (T) increases (0.66 → 0.80), indicating the structured external signal stabilizes self-models. Reflexivity (R) remains positive (+0.06), indicating agents still predict themselves better than neighbors even with structured external input — though the margin narrows from +0.13 without GoL.

**Replication.** Findings 1–3 replicate at sizes 12 (144 agents), 24 (576 agents), and 48 (2,304 agents). Finding 4 is measured at size 48 (the only sweep with R/T/E metrics). Findings 5 and 6 are measured at size 48 and size 24/12 respectively. A previously reported crossover effect — specific seeds benefiting from noise universally — does not replicate across sizes and has been retracted.

## 4. Discussion

The structural effects demonstrate that communication topology alone determines qualitative properties of emergent self-prediction — including whether noise helps, hurts, or has no effect. The Φ dissociation (Finding 3) is relevant to consciousness research: it shows that information integration and self-prediction are mechanistically separable, suggesting that integration alone is insufficient as a consciousness metric.

The T-E dissociation (Finding 4) is the most provocative result. It directly challenges the assumption — central to IIT, MCH, and most autonomy-based frameworks — that self-determination enables self-knowledge. On high-connectivity grids, the relationship inverts: agents most in control of their own trajectories are the worst at predicting themselves. The mechanism is simple: external determination is regularizing. When an agent's future state is driven by the average of 8 predictable neighbors, that future is easy to forecast. When it's driven by the agent's own nonlinear dynamics, it's chaotic and hard to predict. This suggests that in any system where agents are embedded in dense communication networks, self-knowledge may require *surrendering* self-determination — a trade-off between autonomy and self-awareness.

The E ≡ Φ finding (Finding 5) reframes what Φ is measuring. On structured networks, Φ and causal efficacy are functionally identical (r > 0.994). This means Φ partitions — designed to measure information integration — are actually measuring how much an agent's trajectory is self-determined vs. externally driven. The equivalence breaks on random graphs, where independent neighbor sampling decouples integration from self-determination. This has direct implications for IIT: the Φ measure may be conflating integration with autonomy in any system that has spatial or structural correlations.

The GoL substrate results show that Fitz's proposal for a "base reality" layer fundamentally changes the dynamics: agents have structured external signal to learn from, which boosts self-prediction but reduces self-determination. Combined with Finding 4, this suggests a general principle: structured external signal → lower E → higher T → better self-prediction. The price of self-knowledge is dependence on the environment.

**Limitations.** Our agents use simple linear+tanh architectures rather than the transformers Fitz proposes. The GoL coupling is additive rather than perceptual. Our Φ approximation is not the full IIT measure. We do not claim consciousness emergence — our system is too simple for that. What we show is that the *structural logic* of self-prediction from predictive communication is tractable and produces non-trivial, topology-dependent phenomena.

**Reproducibility.** All code and data: github.com/brian-mwirigi/consim. CC BY 4.0.

## References

Fitz, S. (2025). Testing the Machine Consciousness Hypothesis. *arXiv:2512.01081*.

Tononi, G. (2004). An information integration theory of consciousness. *BMC Neuroscience*, 5(42).

Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of 'small-world' networks. *Nature*, 393(6684), 440–442.
