# Findings

1,250 simulations. 50 seeds, 5 topologies, 5 noise levels. 1,000 ticks each at 24x24 (576 agents). Here's what the data says.

## Setup

```bash
python run.py --sweep --sweep-seeds 1-50 \
  --sweep-topos von_neumann,moore,hex,random,small_world \
  --sweep-noises 0.04,0.08,0.12,0.20,0.30 \
  --ticks 1000 --size 24 --sweep-csv sweep_1250.csv
```

## 1. Topology affects self-prediction, but not by much

| Topology | Mean self-score | Max self-score | Clusters (>0.5) |
|----------|:-:|:-:|:-:|
| moore (8 neighbors) | 0.198 | 0.916 | 39.9 |
| hex (6 neighbors) | 0.198 | 0.913 | 54.2 |
| von_neumann (4 neighbors) | 0.186 | 0.909 | 69.5 |
| small_world | 0.186 | 0.905 | 69.2 |
| random (4 neighbors) | 0.182 | 0.905 | 68.3 |

More neighbors correlates with slightly higher mean self-prediction. The effect is consistent across 250 runs per topology but the magnitude is small (0.016 difference between best and worst).

Moore produces roughly half the number of clusters that von Neumann does, meaning the high-scoring agents form fewer, larger groups when they have more neighbors. This is the clearest topological signal in the data.

## 2. Noise barely affects self-prediction

| Noise | Mean self-score | Max self-score | Mean Phi | Pred error |
|-------|:-:|:-:|:-:|:-:|
| 0.04 | 0.186 | 0.908 | 0.335 | 0.037 |
| 0.08 | 0.189 | 0.908 | 0.301 | 0.056 |
| 0.12 | 0.190 | 0.910 | 0.283 | 0.078 |
| 0.20 | 0.192 | 0.911 | 0.269 | 0.124 |
| 0.30 | 0.193 | 0.911 | 0.263 | 0.183 |

Self-model scores are almost identical across a 7.5x range of noise levels. A noise level of 0.30 adds substantial message corruption but agents still reach the same self-prediction scores as at 0.04.

Phi drops 21% as noise increases (0.335 to 0.263). Prediction error scales proportionally with noise, as expected.

This dissociation is interesting: self-prediction is noise-robust but information integration is not.

## 3. Phi and self-prediction are not consistently related

Overall correlation (Pearson r): **-0.14**

By topology:

| Topology | Phi-self correlation |
|----------|:-:|
| random | +0.30 |
| small_world | +0.11 |
| von_neumann | +0.04 |
| hex | -0.25 |
| moore | -0.28 |

The correlation between Phi (integrated information proxy) and self-model score actually flips sign depending on topology. In random graphs, agents with higher Phi tend to have higher self-prediction. In moore and hex grids, it's the opposite.

I don't have an explanation for this. It seems like the relationship between information integration and self-prediction depends on network structure in a way I can't characterize from this data alone.

## 4. Initial conditions matter more than parameters

Best seed (19): mean_self = 0.212
Worst seed (26): mean_self = 0.168
Spread: 0.044

That spread is larger than the effect of changing topology (0.016) or noise level (0.007). The initial random weights create basins of attraction that dominate the outcome. This is a problem if you want to claim any parameter effect is robust. It could also just mean 1,000 ticks isn't enough for the system to escape its initial basin.

Seed 7, which I highlighted in the README for producing high scorers at 48x48, ranks in the bottom 5 at 24x24 (mean_self = 0.174). Seed sensitivity appears to depend on grid size. I have no theory for why.

## 5. The system converges fast

| Tick | Mean self-score | Max self-score | Mean Phi |
|------|:-:|:-:|:-:|
| 500 | 0.192 | 0.907 | 0.291 |
| 1000 | 0.190 | 0.909 | 0.290 |

Barely any change between tick 500 and tick 1000. Mean self-score actually drops slightly (-0.002). Max self-score rises slightly (+0.002). The system reaches approximate equilibrium by tick 500 at this grid size.

Longer runs at larger grids may show different dynamics. I haven't tested that systematically yet.

## 6. Spatial clustering is weak

Moran's I (spatial autocorrelation) across all conditions:

| Topology | Moran's I |
|----------|:-:|
| moore | +0.029 |
| hex | +0.023 |
| von_neumann | -0.002 |
| small_world | -0.002 |
| random | -0.006 |

Values near zero mean scores are essentially randomly distributed in space. Moore and hex show slight positive autocorrelation (high-scorers tend to be near other high-scorers) but the effect is very small. At 24x24, there's no strong spatial structure.

## What I can say

- More neighbors produces slightly higher self-prediction and fewer, larger clusters. Consistent across 250 runs.
- Self-prediction is robust to noise. Phi is not.
- Phi and self-prediction are not measuring the same thing. Their relationship depends on topology.
- Initial conditions dominate over parameter choices at this scale.

## What I can't say

- Whether these effects hold at larger grid sizes or longer runs.
- Why the Phi-self correlation flips sign between topologies.
- Why certain seeds produce consistently different outcomes.
- Whether any of this has anything to do with consciousness. The math produces patterns. The patterns are interesting. That's as far as the data goes.

## Reproduce

```bash
pip install numpy matplotlib Pillow
python run.py --sweep --sweep-seeds 1-50 \
  --sweep-topos von_neumann,moore,hex,random,small_world \
  --sweep-noises 0.04,0.08,0.12,0.20,0.30 \
  --ticks 1000 --size 24
```

Raw data: [sweep_1250.csv](sweep_1250.csv)
