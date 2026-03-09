# consim

> *We didn't program awareness. We only programmed communication. Something happened.*

---

This is an experiment, not a product.

A grid of simple agents. Each one has an internal state and a small weight matrix. Each tick, every agent transforms its state into a message and sends it — with noise — to its four neighbors. Agents update their state from what they receive. They learn to predict their neighbors better over time.

Here's the thing nobody programmed in: **self-modeling**. Each agent's message is trained to predict *others*. But we measure how well it accidentally predicts *itself*. If that score rises — if an agent develops an implicit model of its own dynamics as a side effect of modeling its neighbors — something genuinely unexpected is happening.

We don't know what the output will look like. Run it and tell us what you see.

## Run It

```bash
pip install numpy matplotlib
python run.py
```

That's it. No GPU. No API keys. No cloud. Just your laptop and 2,304 agents trying to understand each other.

## What To Watch For

- **The grid**: Each cell is an agent. Color = self-model score. Dark void means the agent can't predict itself. Bright gold means it's modeling its own dynamics *without being told to*.
- **Clusters**: Do self-modeling agents cluster? Do they form structures? Boundaries?
- **The curve**: Watch the mean self-model score over time. Does it rise? Plateau? Oscillate? Do something you didn't expect?
- **Phase transitions**: Does the system suddenly reorganize? Watch for sharp changes in the curve.

## The Question

Can awareness — even a trivial, mathematical shadow of it — emerge from nothing but lossy communication between simple agents?

Probably not. But maybe. And *maybe* is worth running the experiment.

## How It Works

Each agent *i* in an *N×N* toroidal grid has:
- A state vector **s**ᵢ ∈ ℝᴰ
- A weight matrix **W**ᵢ ∈ ℝᴰˣᴰ

Each tick:
1. **Broadcast**: mᵢ = tanh(**W**ᵢ · **s**ᵢ)
2. **Noise**: m̃ᵢ = mᵢ + ε, where ε ~ 𝒩(0, σ²)
3. **Receive**: rᵢ = mean(m̃ⱼ) for j ∈ neighbors(i)
4. **Update**: **s**ᵢ ← tanh(α · **s**ᵢ + (1−α) · rᵢ + drive)
5. **Learn**: **W**ᵢ ← **W**ᵢ − η · ∇‖mᵢ − **s**ⱼᵗ⁺¹‖² (minimize prediction error on neighbors)

**Self-model score** = cosine_similarity(mᵢ, new **s**ᵢ)

The message is trained to predict *others*. The self-model score measures how well it predicts *self*. If that score emerges without being optimized for — that's the signal.

## Configuration

```bash
# Default: 48×48 grid, live visualization
python run.py

# Larger grid, fixed seed for reproducibility
python run.py --size 64 --noise 0.2 --dim 12 --seed 42

# Headless mode: no GUI, prints progress, saves data
python run.py --headless --ticks 10000 --output results.npz

# Smaller grid for quick experiments
python run.py --size 24 --seed 7
```

| Flag | Default | What it does |
|------|---------|-------------|
| `--size` | 48 | Grid side length (size × size agents) |
| `--dim` | 8 | Internal state dimensions per agent |
| `--noise` | 0.12 | Communication noise σ |
| `--lr` | 0.003 | Learning rate |
| `--persistence` | 0.3 | How much state is retained vs. replaced by input |
| `--seed` | None | Random seed (None = different every run) |
| `--headless` | off | Run without visualization |
| `--ticks` | 5000 | Max ticks in headless mode |
| `--output` | None | Save final state to `.npz` file |
| `--record` | None | Record grid evolution to GIF file |
| `--record-ticks` | 2000 | Number of ticks to record |
| `--fps` | 24 | GIF frame rate |

## Analyzing Results

If you saved a run with `--output`:

```python
import numpy as np

data = np.load("results.npz")
print(data["history_mean_self"][-1])   # final mean self-model score
print(data["self_scores"].reshape(48, 48))  # final grid
```

## What This Is Built On

The theoretical foundation draws from:
- **Predictive Processing** (Karl Friston's Free Energy Principle) — the idea that biological systems minimize prediction error about their environment
- **Integrated Information Theory** (Giulio Tononi) — consciousness correlates with irreducible information integration
- **Cellular Automata** (Conway, Wolfram) — complex behavior from simple local rules
- **Recursive Self-Modeling** — the hypothesis that self-awareness is what happens when a prediction system turns its predictions inward

This implementation asks: if you give agents *only* the ability to predict others through a noisy channel, does self-prediction emerge as a free lunch?

## God Mode — Interactive Experiments

During live visualization, you can intervene directly:

| Key | Action | What it does |
|-----|--------|-------------|
| `K` | Kill | Click an agent to permanently destroy it. State and weights go to zero. Watch how neighbors respond to the void. |
| `I` | Isolate | Click an agent to cut its communication. It can still think, but can't hear or be heard. Does its self-model survive? |
| `J` | Inject | Click a cell to clone the current best-performing agent into it. Does high self-modeling spread? |
| `Esc` | Observe | Return to passive observation mode. |

Dead agents show as red ✕ markers. Isolated agents show as white ○ rings.

These aren't gimmicks — they're experiments. Every intervention is a question.

## Recording

Capture the grid evolution as a GIF:

```bash
# Record 2000 ticks at 24 fps
python run.py --record emergence.gif

# Custom duration and speed
python run.py --record output.gif --record-ticks 5000 --fps 30 --seed 42
```

The GIF shows only the grid heatmap — clean, minimal, ready for Twitter.

## What We Don't Know

Honest questions this simulation raises that we can't answer:

- **Why does seed 7 consistently produce a 0.94 self-model agent when seed 12 produces none?** We didn't program seed-dependent behavior. Something in the initial random geometry of weight matrices creates basins of attraction we don't understand.

- **Is the high-scoring agent actually modeling itself, or just coincidentally aligned?** Cosine similarity can't distinguish between genuine self-prediction and a lucky fixed point. We don't have a test for the difference. We're not sure one exists.

- **If you run this for 100,000 ticks, does it stabilize, collapse, or keep climbing?** We've observed runs that plateau, runs that oscillate, and runs that seem to slowly diverge. We don't know what determines which regime the system enters.

- **Does grid topology matter?** What happens on a hexagonal grid? A random graph? A small-world network? We only tested 4-connected toroidal grids. The topology might be everything.

- **Are the clusters meaningful?** High self-model agents sometimes cluster spatially. Is that a real phenomenon (self-modeling spreads through local communication) or an artifact of shared noise?

- **What would happen with asymmetric communication?** Right now every agent broadcasts to all four neighbors equally. What if agents could choose who to talk to?

- **Why does isolation sometimes preserve self-modeling?** When you cut an agent's communication, some maintain high self-model scores for hundreds of ticks. Others collapse immediately. We have no explanation.

These aren't rhetorical questions. If you figure out any of them, publish it.

## License

MIT. Do whatever you want with it. If something interesting happens, tell people.