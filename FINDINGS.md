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
```

Raw data: [sweep_1250.csv](sweep_1250.csv), [sweep_size12.csv](sweep_size12.csv)
