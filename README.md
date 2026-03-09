# consim

I made 2,304 tiny agents on a grid. Each one can only do two things: send a message to its neighbors, and learn from the messages it receives. The messages are noisy. The learning objective is simple: get better at predicting what your neighbors will do next.

That's it. That's the whole program.

But I also measured something else. I checked how well each agent's outgoing message predicts its own next state. This is not trained. Not rewarded. Not part of the loss function at all. It's just a number I computed on the side.

It goes up.

Not for every agent. Not every run. But seed 7 at tick 1000, there is an agent scoring 0.94 on self-prediction. I have run this maybe 200 times. I cannot explain why seed 7 does this and seed 12 does not. The initial random weights must create some geometry that I do not understand yet.

I am sharing this because I don't know what it means and I think other people should look at it.

## Run it

```bash
pip install numpy matplotlib Pillow
python run.py
```

You will see a grid. Dark cells are agents that cannot predict themselves. If a cell turns bright, that agent has developed self-prediction from nothing but neighbor communication. Watch where the bright spots appear and whether they cluster.

## What you can do during a run

I added the ability to interfere:

| Key | What happens |
|-----|-------------|
| `K` then click | Kill that agent. State and weights go to zero permanently. |
| `I` then click | Cut that agent's communication. It can still update internally but receives nothing. |
| `J` then click | Copy the best-performing agent into that cell. |
| `Esc` | Stop interfering. |

The isolation experiment is the one that bothers me most. Some agents keep their self-model score high for hundreds of ticks after you cut their communication. Others collapse in less than ten. I have looked at the weight matrices of both types and I cannot find a pattern.

## Record a GIF

```bash
python run.py --record emergence.gif --seed 7
python run.py --record long_run.gif --record-ticks 5000 --fps 30
```

## All flags

| Flag | Default | What it does |
|------|---------|-------------|
| `--size` | 48 | Grid side length |
| `--dim` | 8 | State dimensions |
| `--noise` | 0.12 | How much noise on messages |
| `--lr` | 0.003 | Learning rate |
| `--persistence` | 0.3 | How much old state is kept vs replaced |
| `--seed` | None | Random seed |
| `--headless` | off | No window, just printed numbers |
| `--ticks` | 5000 | How many ticks in headless mode |
| `--output` | None | Save final state to .npz |
| `--record` | None | Save grid GIF |
| `--record-ticks` | 2000 | How many ticks to record |
| `--fps` | 24 | GIF speed |

## Load a saved run

```python
import numpy as np
data = np.load("results.npz")
print(data["history_mean_self"][-1])
print(data["self_scores"].reshape(48, 48))
```

## How the math works

Each agent has state `s` in R^8 and weights `W` in R^(8x8). Every tick:

1. Broadcast `m = tanh(W * s)` to four neighbors
2. Messages get Gaussian noise added
3. Average incoming messages
4. New state = tanh(persistence * old_state + (1 - persistence) * received + small_noise)
5. Update W by gradient descent on prediction error against neighbor states

Self-model score = cosine similarity between broadcast message and new state. Not in the gradient. Just measured.

The theoretical background is predictive processing (Friston), integrated information (Tononi), and cellular automata (Conway, Wolfram). The update rule is basically a simplified predictive coding network where your neighbors replace the usual layer hierarchy.

## Things I cannot explain

Seed 7 produces a 0.93+ agent reliably. Seed 12 never does. The initial random weight matrices create basins of attraction that I cannot characterize.

High self-model agents cluster together on the grid. I don't know if agent A being good at self-prediction causes neighbor B to become good at it, or if they just share a favorable noise history. I haven't found a way to tell.

After 10,000 ticks some runs plateau, some oscillate, some slowly diverge. I do not know what determines which.

When I isolate an agent, sometimes the self-model score holds for 300+ ticks with no input. Sometimes it falls apart in 5 ticks. The weight matrices look similar in both cases. I don't know what I'm missing.

If you figure out any of this please open an issue. I really want to know.

## About

400 lines of NumPy. No frameworks. Runs on your laptop.

MIT license.
