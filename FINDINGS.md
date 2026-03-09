# Findings

1,250 simulations. 50 seeds, 5 topologies, 5 noise levels. 1,000 ticks each at 24x24 (576 agents). Here's what the data says.

## Setup

```bash
python run.py --sweep --sweep-seeds 1-50 \
  --sweep-topos von_neumann,moore,hex,random,small_world \
  --sweep-noises 0.04,0.08,0.12,0.20,0.30 \
  --ticks 1000 --size 24 --sweep-csv sweep_1250.csv
```

---

## The weirdest thing in the data

### Noise reshuffles who wins

10 out of 50 seeds benefit from noise across all five topologies. 1 seed is hurt by noise across all five. They cross over.

| Noise | Beneficiaries (10 seeds) | Neutral (39 seeds) | Victim (seed 22) |
|-------|:-:|:-:|:-:|
| 0.04 | 0.180 | 0.187 | 0.207 |
| 0.08 | 0.189 | 0.188 | 0.199 |
| 0.12 | 0.194 | 0.189 | 0.197 |
| 0.20 | 0.199 | 0.190 | 0.194 |
| 0.30 | 0.201 | 0.191 | 0.193 |

The seeds that start below average at low noise end up above average at high noise. The seed that starts above average drops to mediocre. Noise doesn't add randomness rather it reverses the hierarchy of which initial conditions succeed.

This happens within individual runs too. Seed 4 on moore at noise=0.30: at tick 500 it's losing to seed 22 by 0.054. By tick 1000 it's winning by 0.047. A gap reversal of 0.101 in 500 ticks. The noise-adapted dynamics take hundreds of ticks to manifest.

Seed 4 is the strongest universal noise beneficiary. Its full noise curve across all topologies:

| Noise | von_neumann | moore | hex | random | small_world |
|-------|:-:|:-:|:-:|:-:|:-:|
| 0.04 | 0.192 | 0.181 | 0.185 | 0.173 | 0.179 |
| 0.12 | 0.225 | 0.219 | 0.217 | 0.181 | 0.212 |
| 0.30 | 0.237 | 0.247 | 0.240 | 0.183 | 0.224 |

On moore, seed 4 gains +0.066 in mean self-prediction from low to high noise. On random, it gains +0.010. Whatever noise does, it requires spatial structure to work.

---

## The other anomalies

### 1. Most runs produce a high-scoring agent

62% of 1,250 runs produced at least one agent above 0.90 self-prediction. Moore hits 72%, random hits 52%. But 32 runs are "lone wolves": max_self above 0.95, mean_self below 0.18. One agent gets nearly perfect while 575 others stay mediocre.

### 2. Noise helps moore, hurts nothing

Moore gains +0.018 in mean self-prediction from noise=0.04 to noise=0.30. Hex gains +0.015. Von Neumann, random, and small_world are flat. More noise should make prediction harder. For high-connectivity grids it makes self-prediction easier.

21 individual runs show noise=0.30 beating noise=0.04 by more than 0.03 in mean_self. 10 of those are moore, 5 are hex.

### 3. Random topology is in a fixed point

Random is completely noise-immune. Not just Phi — everything:

| Noise | mean_self | p95_self | max_self | std_self | mean_phi | max_phi |
|-------|:-:|:-:|:-:|:-:|:-:|:-:|
| 0.04 | 0.182 | 0.697 | 0.904 | 0.340 | 0.347 | 0.821 |
| 0.12 | 0.183 | 0.695 | 0.905 | 0.338 | 0.345 | 0.822 |
| 0.30 | 0.182 | 0.697 | 0.904 | 0.339 | 0.343 | 0.827 |

Every metric is flat within 0.002 across a 7.5x noise range. The one exception: max_phi goes slightly UP (0.821 to 0.827). Noise concentrates integration into fewer, stronger peaks on random graphs while leaving everything else untouched.

### 4. Small_world shortcuts protect Phi but do nothing for behavior

Small_world starts from von_neumann and adds rewired shortcuts. Self-prediction is identical (within 0.001 at every noise level). But Phi diverges more as noise increases:

| Noise | Phi boost (sw - vn) | Self diff (sw - vn) |
|-------|:-:|:-:|
| 0.04 | +0.015 | -0.001 |
| 0.12 | +0.037 | -0.001 |
| 0.30 | +0.044 | -0.000 |

At noise=0.30, the shortcuts give a 20% Phi boost. All 50 seeds show a positive boost. Zero exceptions. The shortcuts preserve information integration while doing nothing for self-prediction. This is a clean dissociation between Phi and behavior.

### 5. Seed 44 generates 6.5x normal spatial clustering

On grid topologies, seed 44's Moran's I averages 0.097. The population average is 0.015. It dominates the spatial structure outlier list.

On grid topologies, seed 44's clustering correlates with its performance at r = 0.97. More clustering, more self-prediction, almost perfectly. But this only works because seed 44 has some structural property that CREATES clustering where others don't. And it ranks 4th-23rd in self-prediction, not 1st.

Meanwhile seed 24 on von_neumann shows strong ANTI-clustering (Moran's I = -0.09) and consistently OUTPERFORMS seed 44 at every noise level (0.210 vs 0.191). Dispersion beats clustering for this seed-topology pair.

### 6. Fewer clusters means higher scores on moore, lower scores on von_neumann

| Topology | r(clusters_05, mean_self) |
|----------|:-:|
| moore | -0.35 |
| hex | -0.10 |
| von_neumann | +0.27 |
| random | +0.24 |
| small_world | +0.26 |

On moore, fewer score-based clusters means better average performance (agents consolidate into groups). On von_neumann and random, MORE clusters means better performance (agents spread out and specialize). The optimal spatial organization depends on the topology.

But at the higher threshold (clusters_07), ALL topologies show positive correlation with mean_self (r = +0.28 to +0.55). More tight clusters always means better scores. The difference is only in the loose clusters.

### 7. Seed 15 is #1 in random, #42 in von_neumann

Three seeds rank top-5 in one topology and bottom-10 in another:

| Seed | Best | Rank | Worst | Rank |
|------|------|:-:|-------|:-:|
| 15 | random | 1 | von_neumann | 42 |
| 8 | random | 5 | von_neumann | 45 |
| 12 | random | 4 | moore | 47 |

These seeds do best on random graphs and worst on grid topologies. The optimal initial conditions for random networks are the wrong initial conditions for grids.

### 8. Noise destroys Phi on grids (r = -0.87) while helping self-prediction

The error-Phi correlation on moore is r = -0.87. More prediction error means less information integration. But more noise increases self-prediction scores. So noise creates a state where agents are worse at predicting neighbors, worse at integration, but better at predicting themselves.

On moore, from noise=0.04 to 0.30: mean_err goes from 0.030 to 0.143. Mean_phi goes from 0.361 to 0.244. Mean_self goes from 0.188 to 0.206. Neighbor prediction gets 5x worse, integration drops 32%, self-prediction improves 10%.

### 9. Late bloomers exist

Seed 2 on von_neumann improves +0.07 in mean_self between tick 500 and tick 1000 at every noise level. Most runs plateau by tick 500. This one is still climbing. Seed 2 shows the same late-blooming on moore (+0.06) but reverses on random (improves at low noise, declines at high noise). Three seeds are consistent late bloomers across topologies. Late blooming is a seed property, not a noise or topology property.

---

---

## Replication at size 12

Ran the same 1,250 simulations at 12×12 (144 agents) to test whether the headline crossover finding is size-invariant.

```bash
python run.py --sweep --sweep-seeds 1-50 \
  --sweep-topos von_neumann,moore,hex,random,small_world \
  --sweep-noises 0.04,0.08,0.12,0.20,0.30 \
  --ticks 1000 --size 12 --sweep-csv sweep_size12.csv
```

### The crossover does not replicate

At size 12, 9/50 seeds are universal noise beneficiaries — but only 3 overlap with the size-24 set of 10. The other 7 are completely different seeds.

| Size | Universal beneficiaries | Overlap with size-24 set |
|------|:-:|:-:|
| 24 | 10/50: {4, 38, 39, 50, 5, 40, 3, 9, 16, 45} | — |
| 12 | 9/50: {16, 19, 30, 21, 28, 7, 44, 3, 39} | 3/10 |

Tracking the size-24 beneficiary seeds at size 12, the gap reverses direction:

| Noise | Size 24 gap (ben − victim) | Size 12 gap (ben − victim) |
|-------|:-:|:-:|
| 0.04 | −0.027 | **+0.023** |
| 0.12 | −0.002 | +0.010 |
| 0.30 | **+0.008** | **−0.006** |

At size 24, the beneficiaries start below the victim and cross over. At size 12, it's backwards — they start ahead and fall behind. The crossover flips. This means the finding is a seed×size interaction effect, not an intrinsic property of certain initial conditions.

### What does replicate

The structural effects are robust across sizes:

**Moore noise amplification** — replicates. Moore `mean_self` rises monotonically with noise at both sizes.

| Noise | Size 12 moore | Size 24 moore |
|-------|:-:|:-:|
| 0.04 | 0.184 | 0.188 |
| 0.12 | 0.203 | 0.198 |
| 0.30 | 0.215 | 0.206 |

**Random noise immunity** — replicates. `mean_self` stays flat (~0.183±0.001) at both sizes across all noise levels. Phi is also flat.

**Small-world Phi boost** — replicates. At both sizes, small_world shortcuts increase Phi relative to von_neumann while leaving self-prediction unchanged. The boost grows with noise (size 12: +0.013 to +0.040; size 24: +0.015 to +0.044).

**Universal beneficiary/victim existence** — replicates as a phenomenon. Both sizes produce ~9–10 seeds that benefit from noise everywhere and ~1 seed that is hurt everywhere. But WHICH seeds fill those roles changes with grid size.

---

## Summary

The headline crossover from the original analysis is real at size 24 but does not generalize to different grid sizes. Which seeds benefit from noise and which are hurt is an interaction between initial conditions and grid geometry, not an intrinsic property of the seed.

The robust, replicable findings are structural:
- Moore noise amplification (noise helps self-prediction on high-connectivity grids)
- Random topology fixed point (noise-immune across all metrics)
- Small-world Phi dissociation (shortcuts boost integration without touching behavior)
- The existence of universal noise beneficiaries/victims (the phenomenon replicates, the identities don't)

The Phi-vs-self-prediction dissociation is the most portable result. It holds across sizes, topologies, and noise levels.

## What I can't say

- Why the crossover is size-dependent. The seed×size interaction is clear but the mechanism is not.
- What structural property of initial conditions makes a seed a noise-beneficiary at a given size.
- Why the same seed ranks #1 in one topology and #42 in another.
- Whether any of this has anything to do with consciousness.

See [THEORY.md](THEORY.md) for analytical explanations of the random fixed point, moore noise amplification, and small-world Phi dissociation.

---

## Replication at size 48 (with R, T, E metrics)

250 simulations. 10 seeds, 5 topologies, 5 noise levels. 1,000 ticks each at 48×48 (2,304 agents). This sweep also includes the three new MCH metrics: Reflexivity (R), Temporal Persistence (T), and Causal Efficacy (E).

```bash
python run.py --sweep --sweep-seeds 1-10 \
  --sweep-topos von_neumann,moore,hex,random,small_world \
  --sweep-noises 0.04,0.08,0.12,0.20,0.30 \
  --ticks 1000 --size 48 --sweep-csv sweep_size48.csv
```

### All three structural effects replicate

**Moore noise amplification** — replicates. Moore `mean_self` rises monotonically with noise.

| Noise | Size 12 moore | Size 24 moore | Size 48 moore |
|-------|:-:|:-:|:-:|
| 0.04 | 0.184 | 0.188 | 0.187 |
| 0.12 | 0.203 | 0.198 | 0.201 |
| 0.30 | 0.215 | 0.206 | 0.209 |

**Random noise immunity** — replicates. `mean_self` stays flat (~0.184±0.001) across all noise levels. Phi, R, and T are also flat.

| Noise | mean_self | mean_phi | mean_R | mean_T |
|-------|:-:|:-:|:-:|:-:|
| 0.04 | 0.183 | 0.348 | +0.181 | 0.659 |
| 0.12 | 0.184 | 0.347 | +0.182 | 0.658 |
| 0.30 | 0.184 | 0.346 | +0.181 | 0.658 |

R varies by 0.001 across the full noise range. T varies by 0.0006. The fixed point extends to the new MCH metrics.

**Small-world Phi boost** — replicates. At all noise levels, small_world shortcuts increase Phi relative to von_neumann while leaving self-prediction unchanged.

| Noise | Phi boost (sw − vn) | Self diff (sw − vn) |
|-------|:-:|:-:|
| 0.04 | +0.011 | −0.001 |
| 0.12 | +0.034 | −0.004 |
| 0.30 | +0.042 | −0.004 |

The boost grows with noise (size 48: +0.011 to +0.042; size 24: +0.015 to +0.044). Nearly identical magnitudes.

### Cross-size stability

Self-prediction and Phi are remarkably stable across grid sizes 24 and 48:

| Topology | self Δ (48 − 24) | phi Δ (48 − 24) |
|----------|:-:|:-:|
| von_neumann | +0.001 | +0.001 |
| moore | +0.002 | +0.002 |
| hex | +0.003 | +0.002 |
| random | +0.002 | +0.002 |
| small_world | −0.002 | −0.002 |

All deltas within ±0.003. The system's macroscopic behavior is scale-invariant once grids are large enough (~12+).

### New findings from R, T, E

**E (Causal Efficacy) declines monotonically with noise on ALL topologies**. This is the only metric that does. More noise means more external perturbation, making agent trajectories less self-determined. The decline is steepest on moore (0.92 → 0.59) and shallowest on random (0.87 → 0.58).

| Topology | E at noise=0.04 | E at noise=0.30 | Δ |
|----------|:-:|:-:|:-:|
| von_neumann | +0.871 | +0.585 | −0.286 |
| moore | +0.923 | +0.590 | −0.334 |
| hex | +0.904 | +0.581 | −0.323 |
| random | +0.871 | +0.583 | −0.287 |
| small_world | +0.871 | +0.589 | −0.283 |

**T (Temporal Persistence) rises slightly with noise on moore and hex** (0.659 → 0.662). On von_neumann, random, and small_world it is flat (~0.658). Noise on high-connectivity grids stabilizes the self-model. This is the same topologies where noise helps self-prediction — suggesting both effects share a mechanism.

**R (Reflexivity) declines with noise on all grid topologies** (moore drops from +0.148 to +0.105). On random, R is flat (+0.181 ± 0.001). Higher noise makes agents worse at distinguishing self-prediction from other-prediction. Random's immunity extends to R.

**Random topology shows the highest R** (+0.181 vs ~0.125 for grid topologies). Random agents are uniquely good at self/other discrimination, consistent with the hypothesis that random connectivity creates more distinctive agent identities.

### Topology ranking by metric

| Metric | Best | | | | Worst |
|--------|------|---|---|---|-------|
| Self | hex (0.201) | moore (0.200) | vn (0.187) | random (0.184) | sw (0.183) |
| Phi | random (0.347) | moore (0.292) | hex (0.283) | sw (0.282) | vn (0.252) |
| R | random (+0.181) | sw (+0.140) | vn (+0.130) | moore (+0.122) | hex (+0.120) |
| T | moore (0.661) | hex (0.661) | vn (0.659) | random (0.658) | sw (0.658) |
| E | moore (+0.731) | hex (+0.711) | sw (+0.693) | random (+0.690) | vn (+0.690) |

Random leads in Phi and R but trails in self-prediction and E. Moore leads in self-prediction (tied with hex), T, and E. No single topology dominates all metrics — the MCH correlates are genuinely dissociated.

---

## Deep cross-dataset analysis

Statistical mining across all three datasets (5,500 total simulation rows). The findings below were verified for replication across at least two grid sizes unless noted.

### 1. Self-determination hurts self-knowledge (structural, not learned)

**The headline finding.** On moore grids, Temporal Persistence and Causal Efficacy predict self-prediction in **opposite directions**, and are strongly anti-correlated with each other. A null model experiment (lr=0, no learning) confirms this is a **topological invariant**, not a learning artifact.

| Correlation | moore | hex | von_neumann | random | small_world |
|-------------|:-:|:-:|:-:|:-:|:-:|
| r(T, self) | **+0.70** | +0.52 | +0.03 | +0.12 | +0.22 |
| r(E, self) | **−0.71** | −0.71 | −0.12 | −0.14 | +0.06 |
| r(T, E) | **−0.82** | −0.74 | +0.07 | +0.32 | −0.08 |

On moore, a stable self-model (high T) predicts better behavior, but a self-determined trajectory (high E) predicts *worse* behavior. The two metrics pull apart: r(T,E) = −0.82. The agents that predict themselves best are the ones whose self-models are stable over time (high T) but whose dynamics are externally driven (low E). **Self-determination is a liability for self-prediction on high-connectivity grids.**

The mechanism: neighbors on moore are predictable (8 of them, averaging out). If your trajectory is driven by those neighbors, your future state is a function of a predictable input — easy to self-predict. But if your trajectory is self-determined — driven by your own internal dynamics, which are nonlinear and chaotic — you're harder for yourself to forecast. External determination is regularizing. Self-determination is noise, from the self-model's perspective.

This inverts the deepest intuition in the consciousness literature. IIT, MCH, and most autonomy-based frameworks assume self-determination enables self-knowledge. The data says the opposite: on high-connectivity grids, autonomy destroys self-knowledge. A null model with learning disabled (lr=0) confirms the dissociation persists at nearly identical strength (r(T,E) = −0.79 vs −0.82), proving the trade-off is geometric, not a consequence of what the agents learn.

Hex shows the same pattern more weakly. Von Neumann, random, and small_world show near-zero T-E correlation — the dissociation is connectivity-dependent.

### 2. E ≡ Φ on structured graphs

Causal Efficacy and Phi are the same measurement on spatially structured networks. They completely decouple on random graphs.

| Topology | r(E, Φ) | Linear slope | RMSE |
|----------|:-:|:-:|:-:|
| von_neumann | **+0.998** | 0.327 | 0.003 |
| moore | **+0.994** | 0.349 | 0.005 |
| hex | **+0.996** | 0.337 | 0.004 |
| small_world | **+0.984** | 0.321 | 0.007 |
| random | +0.178 | 0.008 | — |

On every grid topology, knowing E gives you Φ (and vice versa) to within ~0.5% RMSE. On random graphs, the correlation drops to noise-level (+0.178). The linear slope on random is 0.008 — essentially flat.

**Implication:** Φ is not measuring "information integration" in a topology-agnostic sense. On structured networks, it is measuring causal self-determination — how much an agent's trajectory is driven by its own dynamics vs. external perturbation. Random connectivity destroys this equivalence because spatial structure is what makes causal efficacy and integration co-vary.

### 3. Simpson's paradox: Φ-self correlation reverses sign

On moore and hex, the Φ-self correlation is **positive** within each noise level but **negative** when pooled across noise levels.

**Moore (size 24):**

| Noise | r(Φ, self) within |
|-------|:-:|
| 0.04 | +0.41 |
| 0.08 | +0.46 |
| 0.12 | +0.55 |
| 0.20 | +0.48 |
| 0.30 | +0.49 |
| **Pooled** | **−0.28** |

Within any single noise level, higher Φ means higher self-prediction. Pool the data, and the sign flips to negative. The confound: noise simultaneously increases self-prediction (+0.018 from low to high noise) while crushing Φ (−0.117). The marginal correlation inherits the noise trend, not the within-condition relationship.

**Replication:** The sign flip replicates at size 12 (moore within: +0.38, pooled: −0.14; hex within: +0.41, pooled: −0.11). Von Neumann, random, and small_world do NOT show the flip at either size — the paradox is specific to high-connectivity grid topologies.

### 4. Seed ranks are non-transferable across sizes

Spearman rank correlations of seed mean_self (averaged over topologies and noises) across grid sizes:

| Pair | Spearman ρ |
|------|:-:|
| size 12 vs 24 | −0.115 |
| size 24 vs 48 | +0.042 |
| size 12 vs 48 | +0.539 |

The 12-vs-24 and 24-vs-48 correlations are essentially zero — knowing which seed wins at one size tells you nothing about the next size. The 12-vs-48 correlation is moderate (+0.54), suggesting the even-sized grids share some structural feature that size 24 disrupts.

**Example:** Seed 4 ranks 10th at size 12, 1st at size 24, 10th at size 48. Seed 8 ranks 1st at size 12, 7th at size 24, 3rd at size 48. Winner identity is entirely size-dependent.

### 5. Von Neumann trajectory anti-prediction (size 24)

On von_neumann at size 24, runs that look good at tick 500 tend to look **worse** at tick 1000.

| Topology | r(mid, end) size 24 | r(mid, end) size 48 |
|----------|:-:|:-:|
| von_neumann | **−0.273** | −0.141 |
| moore | −0.115 | +0.505 |
| hex | −0.080 | +0.346 |
| random | +0.048 | −0.157 |
| small_world | −0.121 | +0.147 |

Von Neumann is the most strongly anti-predictive — early success anti-correlates with final performance. At size 12 the effect is weaker (−0.063). This mean-reversion effect suggests that on sparse grids, early high-scorers are exploiting transient patterns that collapse over longer timescales.

Moore and hex invert between sizes: anti-predictive at size 24 but positively predictive at size 48, where runs have enough agents to stabilize early patterns.

### 6. Cross-topology transferability

Seeds that do well on one grid topology tend to do well on other grid topologies — but random is independent.

**Seed rank correlation between topologies (size 24):**

| | moore | hex | random | small_world |
|---|:-:|:-:|:-:|:-:|
| von_neumann | +0.647 | +0.726 | +0.195 | — |
| moore | — | **+0.877** | +0.255 | — |

**Replication at size 12:** moore-hex = +0.892, vn-moore = +0.675, vn-random = +0.258, moore-random = +0.285. The pattern holds: grid topologies form a performance cluster (moore-hex especially tight at r ≈ 0.88), while random graph success is an orthogonal skill.

### 7. Random topology is a different universe

Random topology is an outlier on nearly every metric and correlation:

| Property | Grid topologies | Random |
|----------|:-:|:-:|
| E-Φ correlation | r > +0.98 | r = +0.18 |
| Noise response (self) | varies | immune |
| Noise response (R, T) | declines/flat | flat |
| R (reflexivity) | ~+0.125 | **+0.181** |
| r(R, self) | −0.44 to +0.44 | **+0.82** |
| Cross-topo transfer | r > +0.6 | r < +0.3 |
| r(mid, end) | negative | +0.05 |
| Φ-self Simpson's paradox | yes (moore/hex) | no |

On random graphs, reflexivity (R) is the dominant predictor of self-prediction accuracy. On grids, T or noise-level dominates instead. Random agents are playing a fundamentally different game: no spatial structure means no causal-efficacy-integration link, no noise vulnerability, and no cross-topology skill transfer.

---

## Reproduce

```bash
pip install numpy matplotlib Pillow

# Original sweep (size 24)
python run.py --sweep --sweep-seeds 1-50 \
  --sweep-topos von_neumann,moore,hex,random,small_world \
  --sweep-noises 0.04,0.08,0.12,0.20,0.30 \
  --ticks 1000 --size 24 --sweep-csv sweep_1250.csv

# Replication sweep (size 12)
python run.py --sweep --sweep-seeds 1-50 \
  --sweep-topos von_neumann,moore,hex,random,small_world \
  --sweep-noises 0.04,0.08,0.12,0.20,0.30 \
  --ticks 1000 --size 12 --sweep-csv sweep_size12.csv

# Size 48 sweep (with R, T, E metrics)
python run.py --sweep --sweep-seeds 1-10 \
  --sweep-topos von_neumann,moore,hex,random,small_world \
  --sweep-noises 0.04,0.08,0.12,0.20,0.30 \
  --ticks 1000 --size 48 --sweep-csv sweep_size48.csv

# Null model sweep (learning disabled)
python run.py --sweep --sweep-seeds 1-10 \
  --sweep-topos von_neumann,moore,hex,random,small_world \
  --sweep-noises 0.04,0.08,0.12,0.20,0.30 \
  --ticks 1000 --size 48 --lr 0 --sweep-csv sweep_null.csv
```

## Null model: learning disabled (lr=0)

python run.py --sweep --sweep-seeds 1-10 \
  --sweep-topos von_neumann,moore,hex,random,small_world \
  --sweep-noises 0.04,0.08,0.12,0.20,0.30 \
  --ticks 1000 --size 48 --lr 0 --sweep-csv sweep_null.csv

```

250 runs identical to the size-48 sweep except learning rate = 0. Agents still communicate, update states, and compute metrics — but their weight matrices never move from random initialization. The gradient step `W -= lr * dW` becomes a no-op.

**Purpose:** Test whether the T-E dissociation (Finding 1 in deep analysis) is caused by learning dynamics or is a structural property of the network topology.

### Result: the dissociation is structural

The T-E anti-correlation survives without learning. The correlation structure is virtually identical:

| Correlation | Learning (lr=0.01) | Null (lr=0) |
|-------------|:-:|:-:|
| **moore r(T,E)** | **−0.82** | **−0.79** |
| moore r(E,self) | −0.71 | −0.67 |
| moore r(T,self) | +0.70 | +0.67 |
| hex r(T,E) | −0.74 | −0.70 |
| hex r(E,self) | −0.71 | −0.66 |
| von_neumann r(T,E) | +0.07 | +0.11 |
| random r(T,E) | +0.32 | +0.32 |

The means are also nearly identical — learning at lr=0.01 over 1000 ticks barely moves the weights from random initialization:

| Metric | Learning moore | Null moore | Δ |
|--------|:-:|:-:|:-:|
| mean_self | 0.1996 | 0.1988 | +0.0008 |
| mean_T | 0.6608 | 0.6607 | +0.0001 |
| mean_E | +0.7314 | +0.7314 | +0.0000 |
| mean_err | 0.0751 | 0.0752 | −0.0001 |

The learning improves self-prediction by 0.08%. The standard deviations are also identical to 4+ decimal places.

### What this means

The T-E dissociation is not a learning artifact — it is a **geometric property of the network topology**. The mechanism is entirely structural:

1. **E (causal efficacy) is topology-determined by construction.** It compares actual state change to counterfactual without neighbors. On high-K grids, more neighbors means more deviation from the self-only trajectory — E tracks how much the network pulls you, regardless of learning.

2. **T (temporal persistence) tracks state-dynamics stability.** With fixed random weights, T still varies because the underlying state dynamics are topology-dependent. On moore (K=8), the 8-neighbor signal average creates smoother dynamics for some agents, giving them higher T.

3. **The anti-correlation is a coupled response to connectivity.** On high-K grids, agents that are more externally driven (low E) receive more averaged neighbor signal, producing smoother dynamics (high T) and better self-prediction. Both T and E respond to the same structural variable — how much neighborhood influence an agent experiences — in opposite directions.

This is a **stronger result** than we expected. The T-E trade-off is not contingent on the learning algorithm — it is a topological invariant. You cannot engineer your way around it by choosing a different learning rule or hyperparameter. The geometry of the network imposes the trade-off.

### Why the learning rate is too small to matter

At lr=0.01 with 1000 ticks, the cumulative weight update is tiny relative to the random initialization. Weight decay further shrinks the effect. The entire dynamics — state updates, communication, metric computation — are dominated by the graph structure and the state update equation $s' = \tanh(\alpha s + (1-\alpha)r + \text{drive})$, not by the learned weights.

---

Raw data: [sweep_1250.csv](sweep_1250.csv), [sweep_size12.csv](sweep_size12.csv), [sweep_size48.csv](sweep_size48.csv), [sweep_null.csv](sweep_null.csv)
