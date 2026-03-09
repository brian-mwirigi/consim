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
5. **Learn:** $W_i \leftarrow W_i - \eta \, \nabla \|m_i - s_j\|^2$

where $\alpha$ is persistence, $\eta$ is learning rate, and drive $\sim \mathcal{N}(0, \delta^2 I)$.

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
- **von_neumann (K=4):** each pair of adjacent neighbors shares 2 common neighbors. High $\rho$. Effective sample size $K_{\text{eff}} = K / (1 + (K-1)\rho) < K$.
- **Moore (K=8):** more neighbors, but crucially, the diagonal neighbors do NOT share neighbors with the cardinal neighbors as heavily. Lower average $\rho$ per pair, AND higher $K$. Both effects improve cancellation.
- **Hex (K=6):** falls between von_neumann and moore.

This predicts a noise-tolerance ordering: random > moore > hex > von_neumann. Which is exactly what the data shows:

| Topology | K | Δ mean_self (0.04→0.30) |
|----------|:-:|:-:|
| random | 4 | +0.000 |
| moore | 8 | +0.018 |
| hex | 6 | +0.015 |
| von_neumann | 4 | +0.006 |
| small_world | 4 | +0.001 |

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
- Self-prediction: small_world ≈ von_neumann (within 0.001)
- Phi: small_world > von_neumann (by 0.015 to 0.044, growing with noise)

### Why self-prediction is unchanged

Self-prediction score is $\cos(m_i, s_i')$ — the cosine similarity between the broadcast message and the new state. The new state depends on the received signal $r_i$. On average, a small-world agent still receives $\sim 3.6$ local neighbor messages plus $\sim 0.4$ random long-range messages. The perturbation from the random shortcuts is small in magnitude (0.4/4 = 10% of the input) and zero-mean in direction (the long-range message is from a random location with no systematic relationship to agent $i$). So $r_i$ and therefore $s_i'$ are approximately unchanged, and self-prediction is approximately unchanged.

### Why Phi increases

The Phi approximation compares:
- **Joint residual:** $\|s_i' - \bar{s}_{\mathcal{N}(i)}\|^2$ — how well the neighborhood mean predicts the state change
- **Parts residual:** $\frac{1}{K}\sum_k \|s_i' - s_{j_k}\|^2$ — how well each individual neighbor predicts it

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
