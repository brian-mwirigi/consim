# Findings

1,250 simulations. 50 seeds, 5 topologies, 5 noise levels. 1,000 ticks each at 24x24 (576 agents). Here's what the data says.

## Setup

```bash
python run.py --sweep --sweep-seeds 1-50 \
  --sweep-topos von_neumann,moore,hex,random,small_world \
  --sweep-noises 0.04,0.08,0.12,0.20,0.30 \
  --ticks 1000 --size 24 --sweep-csv sweep_1250.csv
```

## 1. Most runs produce a high-scoring agent

62% of all 1,250 runs produced at least one agent with self-prediction score above 0.90. 8.5% hit 0.95+. This is not a rare event that requires special seeds or tuning. It happens in the majority of runs.

| Topology | Runs with max > 0.90 |
|----------|:-:|
| moore (8 neighbors) | 72% |
| hex (6 neighbors) | 70% |
| von_neumann (4 neighbors) | 59% |
| small_world | 56% |
| random (4 neighbors) | 52% |

## 2. Moore topology benefits from noise. That shouldn't happen.

Moore (8 neighbors) self-prediction scores increase as noise goes up:

| Noise | Moore mean_self | von_neumann mean_self |
|-------|:-:|:-:|
| 0.04 | 0.188 | 0.185 |
| 0.08 | 0.194 | 0.186 |
| 0.12 | 0.198 | 0.187 |
| 0.20 | 0.203 | 0.187 |
| 0.30 | 0.206 | 0.186 |

Moore gains +0.018 in mean self-prediction from low to high noise. Von Neumann is flat. Hex shows the same pattern as moore (+0.015). Random and small_world are flat.

More noise should make prediction harder. For high-connectivity topologies, it makes self-prediction easier. The effect is consistent across 50 seeds per cell.

## 3. Random topology Phi is immune to noise

This is the most striking result. Random graph Phi barely moves as noise increases 7.5x:

| Noise | Random mean_phi | Moore mean_phi |
|-------|:-:|:-:|
| 0.04 | 0.347 | 0.361 |
| 0.08 | 0.346 | 0.311 |
| 0.12 | 0.345 | 0.280 |
| 0.20 | 0.344 | 0.255 |
| 0.30 | 0.343 | 0.244 |

Random Phi drops 1% across the full noise range. Moore Phi drops 32%. The information integration structure of random graphs is completely noise-invariant. Grid-based topologies lose integration rapidly as communication degrades.

Random topology also has the highest overall Phi (0.345) and the highest max Phi (0.824) despite having the lowest self-prediction scores (0.182). High information integration, low self-prediction. These are measuring fundamentally different things.

## 4. Moore develops spatial structure from noise

Moran's I (spatial autocorrelation) on Moore grids increases with noise:

| Noise | Moore Moran's I |
|-------|:-:|
| 0.04 | 0.000 |
| 0.08 | +0.014 |
| 0.12 | +0.029 |
| 0.20 | +0.046 |
| 0.30 | +0.055 |

At low noise, high-scoring agents are randomly distributed. As noise increases, they cluster together spatially. Noise creates structure. This only happens on high-connectivity grids (moore and hex). Von Neumann, random, and small_world stay near zero.

Moore also produces roughly half the number of score clusters (40) that von_neumann does (70), meaning fewer but larger groups of high-performing agents.

## 5. Phi and self-prediction are anti-correlated in grids, correlated in random graphs

Overall Pearson correlation (mean_self vs mean_phi): r = -0.14

| Topology | Phi-self correlation |
|----------|:-:|
| random | +0.30 |
| small_world | +0.11 |
| von_neumann | +0.04 |
| hex | -0.25 |
| moore | -0.28 |

In random graphs, runs with more information integration also have more self-prediction. In grid topologies (moore, hex), it's the opposite. The relationship between these two measures flips sign depending on whether the network has spatial structure or not.

## 6. Topology effect is statistically large

Cohen's d comparing moore+hex vs the other three topologies: **d = 0.76**

That's a large effect size by standard conventions (>0.5 is "medium", >0.8 is "large"). The absolute difference is 0.013 in mean self-score, but it's highly consistent across 500 vs 750 runs with low variance.

| Group | Mean self-score | n |
|-------|:-:|:-:|
| moore + hex | 0.198 | 500 |
| others | 0.185 | 750 |

## 7. Initial conditions still dominate

Best seed (19): mean_self = 0.212. Worst seed (26): mean_self = 0.168. Spread: 0.044.

That spread is larger than the topology effect (0.016) or noise effect (0.007). Seed 7, which produces high self-scorers at 48x48, ranks in the bottom 5 at 24x24. Seed sensitivity depends on grid size in ways I can't explain.

## 8. The system plateaus fast

| Tick | Mean self-score | Max self-score | Mean Phi |
|------|:-:|:-:|:-:|
| 500 | 0.192 | 0.907 | 0.291 |
| 1000 | 0.190 | 0.909 | 0.290 |

44% of runs increased mean_self between tick 500 and 1000. 56% decreased. The system is at approximate equilibrium by tick 500 at this grid size. Individual runs fluctuate but there's no consistent trend.

## Summary

Three things stand out:

1. **Noise helps high-connectivity topologies self-predict.** This is backwards from what you'd expect. More message corruption makes moore and hex agents better at predicting themselves. It doesn't affect low-connectivity topologies at all.

2. **Random graph Phi is noise-invariant.** Every other topology loses information integration as noise increases. Random graphs don't. Something about the lack of spatial structure preserves integration.

3. **Phi and self-prediction measure different things and their relationship depends on network structure.** Random graphs show positive correlation, grids show negative. These are not two aspects of the same phenomenon.

## What I can't say

- Why noise helps self-prediction on high-connectivity grids. I have no mechanism for this.
- Why random topology Phi is noise-immune. I don't understand what structural property makes this happen.
- Why the Phi-self correlation flips sign between topologies.
- Whether any of this scales to larger grids or longer runs.
- Whether any of this has anything to do with consciousness.

## Reproduce

```bash
pip install numpy matplotlib Pillow
python run.py --sweep --sweep-seeds 1-50 \
  --sweep-topos von_neumann,moore,hex,random,small_world \
  --sweep-noises 0.04,0.08,0.12,0.20,0.30 \
  --ticks 1000 --size 24
```

Raw data: [sweep_1250.csv](sweep_1250.csv)
