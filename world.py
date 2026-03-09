"""
consim — Core Simulation

A toroidal grid of simple agents. Each agent has an internal state vector
and a weight matrix. Every tick, each agent transforms its state into a
message and broadcasts it (with noise) to its four neighbors. Agents
update their state from what they receive. They learn — via gradient
descent on prediction error — to anticipate what their neighbors will
become.

The self-model score measures how well each agent's outgoing message
(trained to predict *others*) accidentally predicts its OWN next state.
This is never explicitly trained for. If it rises, something interesting
is happening: the agent is developing an implicit model of its own
dynamics as a side effect of modeling its environment.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List


@dataclass
class Config:
    """Simulation parameters."""

    size: int = 48  # grid side length (size × size agents)
    dim: int = 8  # internal state dimensionality per agent
    noise: float = 0.12  # communication noise (Gaussian σ)
    lr: float = 0.003  # learning rate for prediction weights
    persistence: float = 0.3  # how much state is retained vs replaced by input
    drive: float = 0.02  # small random perturbation to prevent collapse
    weight_decay: float = 0.0001  # prevents weight explosion
    seed: Optional[int] = None  # random seed (None = unseeded)


class World:
    """
    A grid of communicating agents on a torus.

    Each agent i has:
      - state  sᵢ ∈ ℝᴰ   (internal representation)
      - weights Wᵢ ∈ ℝᴰˣᴰ (communication transform)

    Each tick:
      1. Broadcast:  mᵢ = tanh(Wᵢ · sᵢ)
      2. Noise:      m̃ᵢ = mᵢ + ε,  ε ~ 𝒩(0, σ²)
      3. Receive:    rᵢ = mean(m̃ⱼ)  for j ∈ neighbors(i)
      4. Update:     sᵢ ← tanh(α·sᵢ + (1−α)·rᵢ + drive)
      5. Learn:      Wᵢ ← Wᵢ − η · ∇‖mᵢ − sⱼ‖²

    Self-model score = cosine_similarity(mᵢ, new sᵢ)
    """

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        c = self.cfg
        self.rng = np.random.default_rng(c.seed)

        N = c.size ** 2
        D = c.dim
        self.N, self.D = N, D

        # ── agent internals ───────────────────────────────────
        self.states = self.rng.standard_normal((N, D)).astype(np.float32) * 0.3

        # Communication weights: small random + slight identity bias
        # so agents initially "echo" — broadcast something related to self
        self.W = self.rng.standard_normal((N, D, D)).astype(np.float32) * (0.5 / D)
        eye = np.eye(D, dtype=np.float32)
        self.W += eye[None, :, :] * 0.15

        # ── topology ──────────────────────────────────────────
        self._nbr = self._build_neighbors()

        # ── live metrics ──────────────────────────────────────
        self.self_scores = np.zeros(N, dtype=np.float32)
        self.pred_errors = np.zeros(N, dtype=np.float32)
        self.tick = 0

        # ── God Mode masks ────────────────────────────────────
        self.dead = np.zeros(N, dtype=bool)
        self.isolated = np.zeros(N, dtype=bool)

        # ── time-series history ───────────────────────────────
        self.history: Dict[str, List[float]] = {
            "tick": [],
            "mean_self": [],
            "max_self": [],
            "p95_self": [],
            "std_self": [],
            "mean_err": [],
        }

    # ── topology ──────────────────────────────────────────────

    def _build_neighbors(self) -> np.ndarray:
        """4-connected neighbor indices on a toroidal grid."""
        s = self.cfg.size
        g = np.arange(self.N, dtype=np.int32).reshape(s, s)
        return np.stack(
            [
                np.roll(g, 1, axis=0),  # north
                np.roll(g, -1, axis=1),  # east
                np.roll(g, -1, axis=0),  # south
                np.roll(g, 1, axis=1),  # west
            ],
            axis=-1,
        ).reshape(self.N, 4)

    # ── simulation ────────────────────────────────────────────

    def step(self) -> None:
        """Advance the world by one tick."""
        c = self.cfg
        old = self.states.copy()

        # 1  BROADCAST: each agent transforms its state into a message
        #    mᵢ = tanh(Wᵢ · sᵢ)
        msg = np.tanh(np.einsum("nij,nj->ni", self.W, self.states))

        # 2  LOSSY CHANNEL: add Gaussian noise
        noisy = msg + self.rng.standard_normal(msg.shape).astype(np.float32) * c.noise

        # 3  RECEIVE: each agent averages messages from its 4 neighbors
        recv = noisy[self._nbr].mean(axis=1)  # (N, D)

        # ── God Mode: isolated agents receive no neighbor signals ──
        if np.any(self.isolated):
            recv[self.isolated] = 0.0

        # 4  UPDATE: blend persistence, incoming signals, and random drive
        #    The drive prevents state collapse to zero
        drive = (
            self.rng.standard_normal(self.states.shape).astype(np.float32) * c.drive
        )
        self.states = np.tanh(c.persistence * self.states + (1 - c.persistence) * recv + drive)

        # 5  SELF-MODEL SCORE: cosine similarity between what the agent
        #    broadcast (trained for predicting others) and what it became.
        #    This is NEVER explicitly optimized — emergence only.
        dot = np.einsum("ni,ni->n", msg, self.states)
        n_msg = np.linalg.norm(msg, axis=1) + 1e-8
        n_st = np.linalg.norm(self.states, axis=1) + 1e-8
        self.self_scores = dot / (n_msg * n_st)

        # ── God Mode: dead agents stay dead ────────────────────
        if np.any(self.dead):
            self.states[self.dead] = 0.0
            self.self_scores[self.dead] = 0.0

        # 6  LEARN: update W to reduce prediction error on neighbors
        #    Loss = ‖mᵢ − sⱼᵗ⁺¹‖² averaged over the 4 neighbors j
        nbr_new = self.states[self._nbr]  # (N, 4, D)
        err = msg[:, None, :] - nbr_new  # (N, 4, D)
        avg_err = err.mean(axis=1)  # (N, D)
        self.pred_errors = np.linalg.norm(avg_err, axis=1)

        #    Gradient through tanh: d/dz tanh(z) = 1 − tanh²(z)
        grad_act = 1.0 - msg ** 2  # (N, D)
        scaled = avg_err * grad_act  # (N, D)
        dW = np.einsum("ni,nj->nij", scaled, old)  # (N, D, D)

        self.W -= c.lr * dW
        self.W *= 1.0 - c.weight_decay  # prevent divergence
        np.clip(self.W, -3, 3, out=self.W)

        # ── God Mode: dead agents don't learn ──────────────────
        if np.any(self.dead):
            self.W[self.dead] = 0.0
            self.pred_errors[self.dead] = 0.0

        # 7  RECORD
        self.tick += 1
        h = self.history
        h["tick"].append(self.tick)
        h["mean_self"].append(float(self.self_scores.mean()))
        h["max_self"].append(float(self.self_scores.max()))
        h["p95_self"].append(float(np.percentile(self.self_scores, 95)))
        h["std_self"].append(float(self.self_scores.std()))
        h["mean_err"].append(float(self.pred_errors.mean()))

    # ── God Mode interventions ────────────────────────────────

    def kill_agent(self, row: int, col: int) -> None:
        """Kill an agent — zero its state and weights permanently."""
        idx = row * self.cfg.size + col
        self.dead[idx] = True
        self.isolated[idx] = False
        self.states[idx] = 0.0
        self.W[idx] = 0.0
        self.self_scores[idx] = 0.0
        self.pred_errors[idx] = 0.0

    def isolate_agent(self, row: int, col: int) -> None:
        """Toggle isolation — cut/restore communication."""
        idx = row * self.cfg.size + col
        if self.dead[idx]:
            return
        self.isolated[idx] = not self.isolated[idx]

    def inject_agent(self, row: int, col: int) -> None:
        """Clone the current best-performing agent into this cell."""
        idx = row * self.cfg.size + col
        alive_mask = ~self.dead
        if not np.any(alive_mask):
            return
        scores = self.self_scores.copy()
        scores[~alive_mask] = -2.0
        best = int(np.argmax(scores))
        if best == idx:
            return
        self.states[idx] = self.states[best].copy()
        self.W[idx] = self.W[best].copy()
        self.dead[idx] = False
        self.isolated[idx] = False

    # ── utilities ─────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Capture full state for offline analysis."""
        return {
            "tick": self.tick,
            "states": self.states.copy(),
            "W": self.W.copy(),
            "self_scores": self.self_scores.copy(),
            "pred_errors": self.pred_errors.copy(),
            "config": vars(self.cfg),
        }

    def grid_scores(self) -> np.ndarray:
        """Self-model scores reshaped to the 2D grid."""
        return self.self_scores.reshape(self.cfg.size, self.cfg.size)

    def grid_errors(self) -> np.ndarray:
        """Prediction errors reshaped to the 2D grid."""
        return self.pred_errors.reshape(self.cfg.size, self.cfg.size)
