# consim

I gave 2,304 agents one job: predict your neighbors. They started predicting themselves.

![emergence](emergence.gif)

I made 2,304 small agents on a grid. Each one sends a message to its four neighbors every tick. The messages have noise. Each agent learns to predict what its neighbors will do next. That is the only training objective.

I also measured something that is not part of the training. I check how well each agent's outgoing message predicts its own next state. This number is never optimized. Never rewarded. The agents have no reason to be good at it.

It goes up.

Not always. Not every agent. But run this with seed 7 and wait. Around tick 800, one agent hits 0.94 on self-prediction. I have run this dozens of times. Seed 7 does it reliably. Seed 12 never does. Same code, same parameters, different initial random weights. I don't know why.

I also approximate four MCH (Micro-Coherence Hypothesis) metrics per agent:

- **Φ (Integration)** — how much an agent's state transition depends on its full neighborhood versus each neighbor independently
- **R (Reflexivity)** — how much better an agent predicts itself versus predicting its neighbors
- **T (Temporal Persistence)** — how stable the self-model is over time (low variance = high T)
- **E (Causal Efficacy)** — how much the agent's own dynamics (vs. neighbor input) determine its trajectory

None of these are optimized. They are all emergent measurements.

I'm putting this out because I want other people to look at it.

## Setup

```bash
pip install numpy matplotlib Pillow
python run.py
```

## Topologies

The grid topology changes everything. Five are implemented:

```bash
python run.py --topology von_neumann   # 4 neighbors (default)
python run.py --topology moore         # 8 neighbors (diagonals)
python run.py --topology hex           # 6 neighbors (hexagonal)
python run.py --topology random        # k random neighbors
python run.py --topology small_world   # von_neumann + random rewiring
```

Moore and hex produce different clustering patterns. Random graphs produce different dynamics entirely. small_world with high rewire probability sometimes produces long-range self-model correlations that grid topologies never show. These are observations, not explanations.

## Game of Life substrate

Agents can sit on top of a Conway's Game of Life layer. The GoL runs B3/S23 rules on its own Moore grid, independent of the agent topology. Each agent receives an additive signal encoding its GoL cell state and local density.

```bash
python run.py --gol                           # GoL ON, default coupling 0.1
python run.py --gol --gol-coupling 0.3         # stronger GoL influence
python run.py --gol --gol-density 0.3          # sparser initial GoL state
```

GoL triples mean self-prediction (0.19 → 0.57 at size 24) because agents track the structured external signal. Causal efficacy drops (0.66 → 0.44) because the GoL signal is externally imposed. Temporal persistence rises (0.66 → 0.80) because the GoL provides a stable reference. Reflexivity narrows (+0.13 → +0.06) because the GoL signal helps predict neighbors too.

## Visualization

Left panel is a heatmap of self-model scores. Dark cells have low self-prediction, bright cells have high. Top right shows self-model curves over time. Bottom right shows prediction error and Phi curves.

Press `P` to toggle the heatmap between self-model score and Phi. The spatial distribution of Phi looks different from self-model score in ways I have not been able to characterize.

## Interventions

You can interact with the grid during live runs:

| Key | Mode | Effect |
|-----|------|--------|
| `K` | Kill | Click to permanently zero an agent |
| `I` | Isolate | Click to cut communication (toggles) |
| `J` | Inject | Click to copy the best agent into a cell |
| `P` | View | Toggle heatmap between self-model and Phi |
| `Esc` | Off | Stop intervening |

Killing an agent collapses it. Some neighbors compensate, some degrade. Isolating a high-scorer sometimes preserves its self-model for hundreds of ticks and sometimes collapses it within ten. I have no explanation for the difference.

## Parameter sweep

Run systematic experiments across seeds and topologies:

```bash
# Sweep seeds 1-20 across three topologies, 2000 ticks each
python run.py --sweep --sweep-seeds 1-20 --sweep-topos von_neumann,moore,hex --ticks 2000

# Smaller sweep
python run.py --sweep --sweep-seeds 1-5 --sweep-topos von_neumann --ticks 1000 --size 24
```

Outputs a CSV with per-tick samples of: mean/max/p95 self-model, mean/max Phi, prediction error, Moran's I (spatial autocorrelation), Shannon entropy, cluster counts at two thresholds, and the three MCH metrics (mean R, mean T, mean E).

## Recording

```bash
python run.py --record emergence.gif --seed 7
python run.py --record long_run.gif --record-ticks 5000 --fps 30
```

## All flags

| Flag | Default | |
|------|---------|--|
| `--size` | 48 | Grid side length |
| `--dim` | 8 | State dimensions |
| `--noise` | 0.12 | Message noise |
| `--lr` | 0.003 | Learning rate |
| `--persistence` | 0.3 | How much old state survives each tick |
| `--drive` | 0.02 | Random perturbation strength |
| `--topology` | von_neumann | von_neumann, moore, hex, random, small_world |
| `--num-neighbors` | 4 | Neighbor count for random topology |
| `--rewire-prob` | 0.1 | Rewiring probability for small_world |
| `--seed` | None | Random seed |
| `--headless` | off | No window |
| `--ticks` | 5000 | Headless tick count |
| `--output` | None | Save state to .npz |
| `--record` | None | Save grid GIF |
| `--record-ticks` | 2000 | Ticks to record |
| `--fps` | 24 | GIF speed |
| `--sweep` | off | Run parameter sweep |
| `--sweep-seeds` | 1-10 | Seed range for sweep |
| `--sweep-topos` | von_neumann,moore,hex | Topologies for sweep |
| `--sweep-noises` | 0.12 | Noise levels, comma-separated |
| `--sweep-csv` | sweep_results.csv | Output CSV path |
| `--gol` | off | Enable Game of Life substrate |
| `--gol-coupling` | 0.1 | GoL signal strength |
| `--gol-density` | 0.5 | Initial GoL alive fraction |

## What we found

5,500 simulations across grid sizes 12, 24, and 48. Three structural effects replicate at all sizes (analytical derivations in [THEORY.md](THEORY.md)):

1. **Random graphs are noise-immune fixed points.** Every metric flat within 0.002 across a 7.5× noise range. LLN over independent neighbors.
2. **Noise amplifies self-prediction on high-K grids.** Moore gains +0.018 in self-prediction from low to high noise. Symmetry-breaking decorrelation.
3. **Small-world shortcuts boost Φ 20% without touching behavior.** Uncorrelated long-range neighbors inflate parts residual.

And three findings from the deep cross-dataset analysis:

4. **E ≡ Φ on structured graphs.** Causal efficacy and Phi correlate at r = 0.994–0.998 on every grid topology but decouple on random (r = 0.18). Phi is measuring causal self-determination, not topology-agnostic integration.
5. **Simpson's paradox.** Φ-self correlation is positive within each noise level but negative when pooled, on moore and hex only.
6. **Self-determination hurts self-knowledge on moore.** r(T, self) = +0.70, r(E, self) = −0.71, r(T, E) = −0.82. Agents that control their own trajectories are the worst at predicting themselves. Externally driven agents with stable self-models predict best.

Full data and analysis: [FINDINGS.md](FINDINGS.md). Analytical derivations: [THEORY.md](THEORY.md).

## Loading saved runs

```python
import numpy as np
data = np.load("results.npz")
print(data["history_mean_self"][-1])
print(data["history_mean_phi"][-1])
print(data["self_scores"].reshape(48, 48))
print(data["phi_scores"].reshape(48, 48))
```

## The math

Every agent has state `s` in R^8 and weights `W` in R^(8x8).

Each tick:
1. Broadcast `m = tanh(W * s)` to neighbors
2. Messages get Gaussian noise added
3. Average incoming messages
4. New state = tanh(persistence * old_state + (1 - persistence) * received + small_noise)
5. Update W by gradient descent on prediction error against neighbor states

Self-model score = cosine similarity between broadcast message and new state. Not in the gradient.

Phi is approximated per agent by comparing the prediction residual of the full neighborhood (joint) against the average residual of each single neighbor (parts). When the whole neighborhood predicts the agent's transition better than the average individual neighbor, Phi is positive. This is a simplified proxy for Tononi's integrated information.

Reflexivity (R) = cos(message, own_new_state) − mean(cos(message, neighbor_new_states)). Positive R means the agent is better at self-prediction than other-prediction.

Temporal persistence (T) = 1 − √(EMA variance of self-scores). An agent with a stable self-model across time has T near 1.

Causal efficacy (E) = cos(Δs_actual, Δs_self_only), where Δs_self_only is the counterfactual state change without neighbor input. E near 1 means the agent's trajectory is self-determined.

## Background

Loosely inspired by [*Consciousness as a Coherence-Inducing Operator in a Landscape of Autonomous Micro-Agents*](https://arxiv.org/abs/2512.01081) (2025), which proposes that consciousness-like properties could emerge from lossy predictive communication between local observers. This project is not a faithful implementation of that paper. It borrows the core idea and builds a sandbox around it.

Also draws on predictive processing (Friston), integrated information theory (Tononi), and cellular automata (Conway, Wolfram). The update rule is a predictive coding network where your neighbors replace the layer hierarchy.

## Analysis tools

The `analysis.py` module provides:

- `morans_i(world)` - Moran's I spatial autocorrelation of self-model scores (+1 = clustered, 0 = random, -1 = dispersed)
- `state_entropy(world)` - Shannon entropy of score distribution
- `phi_entropy(world)` - Shannon entropy of Phi distribution
- `cluster_count(world, threshold)` - Connected components above a threshold
- `run_sweep(...)` - Batch runner with CSV output

```python
from world import World, Config
from analysis import morans_i, state_entropy, cluster_count

w = World(Config(size=24, seed=7))
for _ in range(1000): w.step()

print(f"Moran's I: {morans_i(w):.4f}")
print(f"Entropy:   {state_entropy(w):.4f}")
print(f"Clusters:  {cluster_count(w, threshold=0.5)}")
```

## Things I can't figure out

Seed 7 produces a 0.93+ agent every time. Seed 12 does not. The weight initialization creates basins of attraction I can't characterize.

High-scoring agents cluster on the grid. I don't know if one agent being good at self-prediction makes its neighbor better, or if they just happen to share favorable noise. I have tried killing a high-scorer to see if the cluster degrades. Sometimes it does. Sometimes the neighbors get better. I don't understand that at all.

Moore topology (8 neighbors) produces more uniform self-model distributions than von Neumann (4 neighbors). Hex (6 neighbors) produces tighter clusters. Random graphs sometimes produce isolated high-scorers with no spatial pattern. I don't have a theory for why neighbor count relates to spatial structure this way.

Past 10,000 ticks I see three behaviors: plateau, oscillation, and slow divergence. I have no idea what decides which one happens.

The isolation question: cut communication to a high-scorer, and sometimes the self-model holds without any input for a long time. I would like to know why. If you figure it out, open an issue.

Phi and self-model score don't always correlate. Some agents show high information integration with low self-prediction. The reverse also occurs. This bothers me because the theories predict they should be related.

## Results

I ran 3,750+ simulations across multiple grid sizes (12, 24, 48), 50 seeds, 5 topologies, and 5 noise levels. Full writeup with tables: [FINDINGS.md](FINDINGS.md). Analytical derivations for the three replicating effects: [THEORY.md](THEORY.md). Extended abstract for MC0001 conference submission: [ABSTRACT.md](ABSTRACT.md).

Short version: more neighbors produces slightly higher self-prediction. Self-prediction is noise-robust but Phi is not (except on random graphs, where both are noise-immune). Phi and self-model score are not measuring the same thing. Initial conditions matter more than parameter choices. Game of Life substrate triples self-prediction but reduces causal efficacy.

## About

~800 lines of NumPy. Runs on a laptop. Size-48 grids run at ~4ms/tick. MIT license.
