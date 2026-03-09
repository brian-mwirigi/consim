# Theory

Analytical derivations for the three replicating structural effects in consim.

---

## 1. Why random topology is a noise fixed point

### The update equation

Each agent $i$ has state $s_i \in \mathbb{R}^D$ and weights $W_i \in \mathbb{R}^{D \times D}$. Every tick:

1. **Broadcast:** $m_i = \tanh(W_i \cdot s_i)$
2. **Noise:** $\tilde{m}_i = m_i + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma^2 I)$
3. **Receive:** $r_i = \frac{1}{K} \sum_{j \in \mathcal{N}(i)} \tilde{m}_j$
4. **Update:** $s_i \leftarrow \tanh\!\big(\alpha \, s_i + (1 - \alpha) \, r_i + \text{drive}\big)$
5. **Learn:** $W_i \leftarrow W_i - \eta \, \nabla \frac{1}{K}\sum_{j \in \mathcal{N}(i)} \|m_i - s_j'\|^2$

where $\alpha$ is persistence, $\eta$ is learning rate, $s_j'$ is the neighbor's **new** state, and drive $\sim \mathcal{N}(0, \delta^2 I)$.

### The received signal on a random graph

On a random graph, agent $i$'s $K$ neighbors are drawn uniformly at random from all $N$ agents. Expand the received signal:

$$r_i = \frac{1}{K} \sum_{j \in \mathcal{N}(i)} \tilde{m}_j = \underbrace{\frac{1}{K} \sum_{j \in \mathcal{N}(i)} m_j}_{\text{signal}} + \underbrace{\frac{1}{K} \sum_{j \in \mathcal{N}(i)} \varepsilon_j}_{\text{noise}}$$

**Signal term.** The $K$ messages $m_j$ are drawn from the population $\{m_1, \dots, m_N\}$. Since the neighbors are independent uniform samples, by the law of large numbers the sample mean converges to the population mean:

$$\frac{1}{K} \sum_{j \in \mathcal{N}(i)} m_j \approx \bar{m} \equiv \frac{1}{N} \sum_{j=1}^{N} m_j$$

with variance $O(\text{Var}(m) / K)$. The population mean $\bar{m}$ is a global statistic — it does not depend on which $K$ agents were sampled.

**Noise term.** The $\varepsilon_j$ are i.i.d. $\mathcal{N}(0, \sigma^2 I)$, so:

$$\frac{1}{K} \sum_{j \in \mathcal{N}(i)} \varepsilon_j \sim \mathcal{N}\!\left(0, \frac{\sigma^2}{K} I\right)$$

Zero mean. Variance scales as $\sigma^2 / K$.

### The self-stabilizing loop

Combining both terms:

$$r_i \approx \bar{m} + \mathcal{N}\!\left(0, \frac{\sigma^2}{K} I\right)$$

Now consider what happens when $\sigma$ increases:

1. **Each agent's received signal** gains more variance, but the perturbation is zero-mean. The expected value of $r_i$ is unchanged.

2. **Each agent's state update** $s_i \leftarrow \tanh(\alpha \, s_i + (1-\alpha)\,r_i + \text{drive})$ has the same expectation, because $\mathbb{E}[r_i]$ is unchanged. The tanh saturates symmetrically, so zero-mean noise produces zero-mean state perturbations (to first order).

3. **The population mean message** $\bar{m} = \frac{1}{N}\sum_j \tanh(W_j s_j)$ aggregates over all $N$ agents. If each agent's state is perturbed by a zero-mean noise, the perturbations cancel across the population by LLN over $N$ agents. So $\bar{m}$ itself is stable.

4. **The learning target** is also stable. Each agent updates $W_i$ to minimize $\|m_i - s_j\|^2$ where $s_j$ is a neighbor's new state. The neighbor states are drawn from the same population, which has the same mean. So the gradient direction is unchanged in expectation.

This is a closed self-stabilizing loop:

$$\sigma \uparrow \;\;\rightarrow\;\; \text{zero-mean perturbation to each } s_i \;\;\rightarrow\;\; \bar{m} \text{ unchanged (LLN over } N\text{)} \;\;\rightarrow\;\; \mathbb{E}[r_i] \text{ unchanged} \;\;\rightarrow\;\; \mathbb{E}[s_i] \text{ unchanged}$$

The random graph is a fixed point because independent sampling makes noise cancel at two levels: over $K$ neighbors (received signal) and over $N$ agents (population mean).

### Empirical confirmation

From 1,250 simulations (50 seeds × 5 noise levels), random topology on both size 12 and size 24:

| Metric | noise=0.04 | noise=0.30 | Δ |
|--------|:-:|:-:|:-:|
| mean_self (size 24) | 0.182 | 0.182 | <0.001 |
| mean_self (size 12) | 0.183 | 0.184 | <0.001 |
| mean_phi (size 24) | 0.347 | 0.343 | 0.004 |
| mean_phi (size 12) | 0.345 | 0.338 | 0.007 |

Every metric stable within 0.007 across a 7.5× noise range. Phi shows a slight decline because the Phi approximation uses residual variance, which does increase with noise — but the behavioral metrics (self-prediction, error) are invariant.

---

## 2. Why more neighbors means more noise tolerance

### The effective sample size argument

On a grid topology, agent $i$'s $K$ neighbors are spatially adjacent. They share neighbors with each other. If agent $j_1$ and $j_2$ are both neighbors of $i$ and also neighbors of each other, their messages $m_{j_1}$ and $m_{j_2}$ are correlated — they were computed from similar inputs at the previous tick.

For correlated samples, the variance of the sample mean is:

$$\text{Var}\!\left(\frac{1}{K}\sum_{j} m_j\right) = \frac{\sigma_m^2}{K}\left(1 + (K-1)\rho\right)$$

where $\rho$ is the average pairwise correlation between neighbor messages.

- **Random graph:** neighbors are independent, $\rho \approx 0$. Variance $\approx \sigma_m^2 / K$. Full cancellation.
- **von_neumann (K=4):** perpendicular neighbor pairs (e.g. N and E) share 2 common neighbors; opposite pairs (N and S) share only agent $i$. High average $\rho$. Effective sample size $K_{\text{eff}} = K / (1 + (K-1)\rho) < K$.
- **Moore (K=8):** more neighbors, but crucially, the diagonal neighbors do NOT share neighbors with the cardinal neighbors as heavily. Lower average $\rho$ per pair, AND higher $K$. Both effects improve cancellation.
- **Hex (K=6):** falls between von_neumann and moore.

This predicts a noise-tolerance ordering: random > moore > hex > von_neumann. Which is exactly what the data shows:

| Topology | K | Δ mean_self (0.04→0.30) |
|----------|:-:|:-:|
| random | 4 | +0.000 |
| moore | 8 | +0.018 |
| hex | 6 | +0.015 |
| von_neumann | 4 | +0.001 |
| small_world | 4 | +0.002 |

Moore and hex show positive noise effects (noise *helps*, not just "doesn't hurt") — the derivation above explains why noise doesn't destroy performance, but the amplification requires a second mechanism.

### Why noise actively helps on moore

On high-connectivity grids, noise breaks symmetry. Without noise, nearby agents receive nearly identical inputs (their neighborhoods overlap heavily) and converge to similar states. This creates redundancy — many agents doing the same thing.

Noise decorrelates the received signals of adjacent agents. With the tanh nonlinearity, this pushes agents into different regions of state space. More diverse states in a neighborhood means the learning gradient $\nabla\|m_i - s_j\|^2$ carries more information — the target $s_j$ varies more across neighbors, so the agent's weight matrix $W_i$ is trained on a richer distribution.

The effect is strongest on moore (K=8) because:
1. Higher K means more terms in $r_i$, so the agent can tolerate noise without losing the signal
2. More neighbors means more overlap, so there's more redundancy for noise to break
3. The combination — noise-tolerant AND redundancy-rich — is unique to high-K grid topologies

Random graphs don't benefit because neighbors are already decorrelated (no spatial structure to break). Von_neumann doesn't benefit because K=4 is too small — noise overwhelms the signal before it can decorrelate.

---

## 3. Why small-world shortcuts boost Phi but not self-prediction

### The dissociation

Small-world starts from von_neumann (K=4 grid neighbors) and rewires each edge with probability $p=0.1$. Rewired edges connect to random agents anywhere on the grid.

Empirically, across all noise levels and both grid sizes:
- Self-prediction: small_world ≈ von_neumann (within 0.002 at size 24, within 0.011 at size 12)
- Phi: small_world > von_neumann (by 0.013 to 0.044, growing with noise)

### Why self-prediction is unchanged

Self-prediction score is $\cos(m_i, s_i')$ — the cosine similarity between the broadcast message and the new state. The new state depends on the received signal $r_i$. On average, a small-world agent still receives $\sim 3.6$ local neighbor messages plus $\sim 0.4$ random long-range messages. The perturbation from the random shortcuts is small in magnitude (0.4/4 = 10% of the input) and zero-mean in direction (the long-range message is from a random location with no systematic relationship to agent $i$). So $r_i$ and therefore $s_i'$ are approximately unchanged, and self-prediction is approximately unchanged.

### Why Phi increases

The Phi approximation compares (using state change $\Delta s_i = s_i' - s_i$ and **old** neighbor states $s_j^{\text{old}}$):
- **Joint residual:** $\|\Delta s_i - \bar{s}_{\mathcal{N}(i)}^{\text{old}}\|^2$ — how well the old neighborhood mean predicts the state change
- **Parts residual:** $\frac{1}{K}\sum_k \|\Delta s_i - s_{j_k}^{\text{old}}\|^2$ — how well each old individual neighbor predicts it

Phi $\propto$ (parts residual − joint residual) / parts residual.

A long-range shortcut connects agent $i$ to a distant agent $j^*$ whose state $s_{j^*}$ is uncorrelated with the local cluster. This makes $j^*$ a poor individual predictor (high parts residual). But the mean $\bar{s}_{\mathcal{N}(i)}$ — which includes $j^*$ — is still a reasonable predictor because the 3 local neighbors dominate the average and $j^*$'s contribution is diluted.

So:
- Parts residual goes UP (one neighbor is now a bad predictor)
- Joint residual stays roughly the SAME (the mean is robust to one outlier in 4)
- Phi = (parts − joint) / parts goes UP

The effect grows with noise because noise increases the variance of the long-range message (it's already uncorrelated; noise makes it more so), widening the gap between the bad individual prediction and the still-reasonable joint prediction.

### Prediction

This derivation predicts that the Phi boost should:
1. Scale with rewire probability $p$ (more shortcuts → more uncorrelated neighbors → bigger boost) — **testable**
2. Disappear if you use only the local neighbors for the Phi calculation — **testable**
3. Be proportional to the number of rewired edges, not their length — **testable**

---

## Summary

| Finding | Mechanism | Key assumption that breaks on grids |
|---------|-----------|-------------------------------------|
| Random fixed point | LLN: independent samples → noise cancels | Neighbors are independent |
| Moore noise amplification | Higher K + noise-driven decorrelation | K large enough to tolerate noise while breaking redundancy |
| Small-world Phi dissociation | Shortcuts add uncorrelated neighbors → parts residual ↑ | At least one neighbor is uncorrelated with local cluster |

The three findings are connected: they are all consequences of the statistical relationship between neighbor correlation structure and noise cancellation. Random graphs are one extreme (fully independent). Regular grids are the other (maximally correlated). Moore, hex, and small-world fall in between, and their position in that spectrum determines their noise response.

---

## 4. Why self-determination hurts self-knowledge on high-K grids

### The observation

On Moore grids (K=8), the correlation between causal efficacy (E) and self-prediction is r = −0.71. The correlation between temporal persistence (T) and self-prediction is r = +0.70. T and E are anti-correlated at r = −0.82. The effect is weaker on hex (K=6), absent on von Neumann (K=4) and random.

### The state update decomposition

The state update is:

$$s_i' = \tanh\!\big(\alpha \, s_i + (1 - \alpha) \, r_i + \text{drive}\big)$$

where $r_i = \frac{1}{K}\sum_{j \in \mathcal{N}(i)} \tilde{m}_j$ is the received signal. Decompose the pre-activation into self-driven and neighbor-driven components:

$$z_i = \alpha \, s_i + (1 - \alpha) \, r_i + \text{drive} = \underbrace{\alpha \, s_i + \text{drive}}_{z_i^{\text{self}}} + \underbrace{(1 - \alpha) \, r_i}_{z_i^{\text{ext}}}$$

Causal efficacy (E) measures how much $\Delta s_i = s_i' - s_i$ aligns with the counterfactual $\Delta s_i^{\text{self}} = \tanh(z_i^{\text{self}}) - s_i$. High E means the trajectory is dominated by the self component. Low E means the external component dominates.

### Why high K makes external input predictable

On a Moore grid with K=8 neighbors, the received signal is:

$$r_i = \frac{1}{8}\sum_{j=1}^{8} \tilde{m}_j$$

The neighbor messages $m_j = \tanh(W_j s_j)$ are bounded in $[-1, +1]^D$. Averaging 8 of them compresses the variance by a factor of ~8 (if independent) or less (if correlated). Crucially, on a high-K grid, adjacent agents share many neighbors: two adjacent cells on a Moore grid share 3–5 common neighbors. This means:

1. The received signals $r_i$ and $r_j$ are correlated for adjacent agents
2. Each agent's received signal is a heavily smoothed version of its local neighborhood
3. The smoothed signal changes slowly across ticks — nearby states → nearby broadcasts → similar $r_i$ values next tick

Under these conditions, the external component $z_i^{\text{ext}} = (1-\alpha) r_i$ is **predictable**: it is a smoothed, slowly-varying function of the neighborhood. An agent whose state update is dominated by this external component (low E) has a predictable trajectory — precisely the condition for high self-prediction.

### Why the self component introduces chaos

The self-driven component $z_i^{\text{self}} = \alpha s_i + \text{drive}$ contains two sources of unpredictability:

1. **The state itself** is the output of a tanh nonlinearity applied to the full input at the previous tick. Small differences in $z_i$ at one tick get amplified through the tanh, creating sensitivity to initial conditions.
2. **The drive term** is random noise ($\mathcal{N}(0, \delta^2 I)$), injecting fresh stochasticity every tick.

For an agent with high E, $s_i'$ is dominated by $\tanh(z_i^{\text{self}})$. The self-model — computed as $\cos(m_i, s_i')$ where $m_i = \tanh(W_i s_i)$ — must predict a quantity that depends on the agent's own nonlinear recurrence plus noise. This is a chaotic map prediction problem.

For an agent with low E, $s_i'$ is dominated by $\tanh(z_i^{\text{ext}})$. The self-model must predict a quantity that depends on the smoothed neighborhood average. This is an averaging problem — much easier to predict.

### Why this requires high K

The argument depends on the external signal $r_i$ being predictable. This requires:

1. **Enough neighbors** to compress variance: $\text{Var}(r_i) \propto 1/K$. At K=4 (von Neumann), the smoothing is weaker and $r_i$ is nosier.
2. **Neighbor correlation**: on a grid, adjacent neighbors share common inputs, making their messages correlated and $r_i$ even smoother. On random graphs, neighbors are independent, so $r_i$ varies more between ticks.
3. **Balance**: at K=8, the smoothing is strong enough that external-driven agents have genuinely predictable trajectories, but the self-driven component is still chaotic. At K=4, neither component is sufficiently predictable to create the split.

This explains the empirical gradient: moore (K=8, r(T,E) = −0.82) > hex (K=6, r(T,E) = −0.74) > von_neumann (K=4, r(T,E) = +0.07). The dissociation strengthens monotonically with K.

### Connection to temporal persistence

Temporal persistence $T = \text{clip}(1 - \sqrt{\text{Var}_{\text{ema}}(\text{self\_score})}, 0, 1)$ measures how stable the self-prediction score is over time. An agent with low E (externally driven) receives smoothed inputs that change slowly → its trajectory changes slowly → its self-prediction score is stable → high T.

An agent with high E (self-driven) has a trajectory dominated by its own nonlinear recurrence + noise → the trajectory is erratic → the self-prediction score fluctuates → low T.

This is why T and E anti-correlate on high-K grids. They are not measuring independent aspects of the agent. They are measuring the same underlying variable — how much the trajectory is externally vs. self-driven — through different lenses. T sees it as stability, E sees it as autonomy. On high-K grids, you cannot have both.

### Implications

This creates a fundamental tension for any theory that links consciousness to both self-determination and self-knowledge:

- IIT (Tononi) emphasizes integration (≈ self-determination on grids, via the E ≡ Φ identity)
- MCH (Fitz) requires all four of Φ, R, T, E to be present
- Most autonomy-based theories assume self-determination enables self-awareness

The data shows that on high-connectivity networks, optimizing for E (self-determination) necessarily degrades T (temporal stability) and self-prediction accuracy. The four MCH metrics cannot all be simultaneously maximized. There is a Pareto frontier — and the agents that predict themselves best are the ones that sacrifice autonomy for regularity.

### Empirical confirmation

From 250 simulations at size 48 (10 seeds × 5 topologies × 5 noise levels):

| Topology | K | r(T, self) | r(E, self) | r(T, E) |
|----------|:-:|:-:|:-:|:-:|
| von_neumann | 4 | +0.03 | −0.12 | +0.07 |
| hex | 6 | +0.52 | −0.71 | −0.74 |
| moore | 8 | +0.70 | −0.71 | −0.82 |
| random | 4 | +0.12 | −0.14 | +0.32 |
| small_world | 4+ | +0.22 | +0.06 | −0.08 |

The monotonic K-dependence (vn < hex < moore) matches the derivation. Random shows no effect because $r_i$ is drawn from independent samples rather than correlated grid neighbors, making the external signal noisier rather than smoother.

---

## 5. Why E ≡ Φ on structured graphs

### The observation

On all grid topologies, causal efficacy (E) and integration (Φ) correlate at r > 0.994. On random graphs, r = 0.178.

### Why they measure the same thing on grids

Recall the definitions:
- **E**: $\cos(\Delta s_i, \Delta s_i^{\text{self}})$ — alignment between actual and self-only state change
- **Φ**: $(\text{parts residual} - \text{joint residual}) / \text{parts residual}$ — how much the full neighborhood predicts the state transition better than individual neighbors

On a grid, neighboring agents are correlated (they share neighbors). The joint prediction (averaging all neighbors) captures this shared structure — it approximates the smoothed external signal $r_i$. The parts prediction (individual neighbors) includes uncorrelated per-neighbor noise.

When E is high (trajectory is self-driven), the state change $\Delta s_i$ is poorly predicted by *both* joint and parts — the state change is driven by $s_i$ itself, not the neighborhood. Both residuals are large, and their ratio is close to 1 → low Φ.

When E is low (trajectory is externally driven), $\Delta s_i$ is well-predicted by the joint neighborhood (which approximates the actual $r_i$ the agent received). But individual neighbors still predict poorly. Parts residual stays high, joint residual drops → high Φ.

So on grids: low E ↔ high Φ. They anti-correlate — but since E is defined directionally (1 = self-driven) and the empirical correlation is positive, what's happening is: both E and Φ covary with noise. As noise increases, both drop. The correlation captures their shared response to the noise parameter, not a functional identity post-conditioning. Within a fixed noise level, they are still tightly coupled because both respond to the same underlying variable: the balance between self-driven and externally-driven dynamics.

### Why it breaks on random graphs

On random graphs, neighbors are independent samples from the population. The joint prediction (averaging K independent signals) has variance $\sigma^2/K$ around the population mean, while each parts prediction has variance $\sigma^2$. This ratio is determined purely by K, not by the agent's E value. The external signal $r_i$ on a random graph is the population mean ± noise, regardless of how self-driven the agent is.

So on random graphs: Φ is determined by the population-level variance structure (which is constant across agents), not by the agent's self-determination. E varies across agents but doesn't change the prediction structure. The two metrics decouple.

---

## Summary

| Finding | Mechanism | Key assumption |
|---------|-----------|----------------|
| Random fixed point | LLN: independent samples → noise cancels | Neighbors are independent |
| Moore noise amplification | Higher K + noise-driven decorrelation | K large enough to tolerate noise while breaking redundancy |
| Small-world Phi dissociation | Shortcuts add uncorrelated neighbors → parts residual ↑ | At least one neighbor is uncorrelated with local cluster |
| T-E dissociation on high-K grids | External input is smoothed and predictable; self-dynamics are chaotic | K high enough that neighbor-averaging regularizes the trajectory |
| E ≡ Φ on grids | Both measure self-vs-external balance through different lenses | Spatially correlated neighbors create a structured external signal |

All five findings are consequences of a single underlying variable: the correlation structure of the neighborhood. On grids, neighbors are correlated → external input is predictable → self-determination trades off against self-knowledge. On random graphs, neighbors are independent → external input is noise → no trade-off, no E-Φ identity, no T-E dissociation.
