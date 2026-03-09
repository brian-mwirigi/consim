# consim

I gave 2,304 agents one job: predict your neighbors. They started predicting themselves.

![emergence](emergence.gif)

I made 2,304 small agents on a grid. Each one sends a message to its four neighbors every tick. The messages have noise. Each agent learns to predict what its neighbors will do next. That is the only training objective.

I also measured something that is not part of the training. I check how well each agent's outgoing message predicts its own next state. This number is never optimized. Never rewarded. The agents have no reason to be good at it.

It goes up.

Not always. Not every agent. But run this with seed 7 and wait. Around tick 800, one agent hits 0.94 on self-prediction. I have run this dozens of times. Seed 7 does it reliably. Seed 12 never does. Same code, same parameters, different initial random weights. I don't know why.

I'm putting this out because I want other people to look at it.

## Run it

```bash
pip install numpy matplotlib Pillow
python run.py
```

The grid shows up. Dark cells are agents with no self-prediction. When a cell goes bright, that agent learned to predict itself as a side effect of predicting others. Watch where the bright spots show up and whether they form clusters.

## Interventions

You can mess with the grid while it runs:

| Key | What happens |
|-----|-------------|
| `K` then click | Kill that agent permanently |
| `I` then click | Cut that agent's communication (toggle on/off) |
| `J` then click | Copy the best agent into that cell |
| `Esc` | Go back to watching |

The isolation one is what keeps me up. Some agents hold their self-model score for 300+ ticks after you cut their input. Others fall apart in under 10. I looked at the weights of both types. They look the same to me. I don't know what separates them.

## Record a GIF

```bash
python run.py --record emergence.gif --seed 7
python run.py --record long_run.gif --record-ticks 5000 --fps 30
```

## Flags

| Flag | Default | |
|------|---------|--|
| `--size` | 48 | Grid side length |
| `--dim` | 8 | State dimensions |
| `--noise` | 0.12 | Message noise |
| `--lr` | 0.003 | Learning rate |
| `--persistence` | 0.3 | How much old state survives each tick |
| `--seed` | None | Random seed |
| `--headless` | off | No window |
| `--ticks` | 5000 | Headless tick count |
| `--output` | None | Save state to .npz |
| `--record` | None | Save grid GIF |
| `--record-ticks` | 2000 | Ticks to record |
| `--fps` | 24 | GIF speed |

## Load a saved run

```python
import numpy as np
data = np.load("results.npz")
print(data["history_mean_self"][-1])
print(data["self_scores"].reshape(48, 48))
```

## The math

Every agent has state `s` in R^8 and weights `W` in R^(8x8).

Each tick:
1. Broadcast `m = tanh(W * s)` to four neighbors
2. Noise gets added to every message
3. Average what you receive
4. New state = tanh(persistence * old + (1 - persistence) * received + small perturbation)
5. Update W by gradient descent on how well m predicted neighbor states

Self-model score = cosine similarity between m and the new state. Measured, not trained.

Built on ideas from predictive processing (Friston), integrated information (Tononi), and cellular automata (Conway, Wolfram). Think of it as a predictive coding network where your layer hierarchy is replaced by whoever sits next to you.

## Things I can't figure out

Seed 7 produces a 0.93+ agent every time. Seed 12 never does. The weight initialization creates basins of attraction I can't characterize.

High-scoring agents cluster on the grid. I don't know if one agent being good at self-prediction makes its neighbor better, or if they just happen to share favorable noise. I have tried killing a high-scorer to see if the cluster degrades. Sometimes it does. Sometimes the neighbors get better. I don't understand that at all.

Past 10,000 ticks I see three behaviors: plateau, oscillation, and slow divergence. I have no idea what decides which one happens.

The isolation question again: cut communication to a high-scorer, and sometimes the self-model holds without any input for a long time. I would like to know why. If you figure it out, open an issue.

## About

~400 lines of NumPy. Runs on a laptop. MIT license.
