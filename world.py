"""
consim — Core Simulation

A toroidal grid of simple agents. Each agent has an internal state vector
and a weight matrix. Every tick, each agent transforms its state into a
message and broadcasts it (with noise) to its neighbors. Agents update
their state from what they receive. They learn — via gradient descent on
prediction error — to anticipate what their neighbors will become.

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
    topology: str = "von_neumann"  # von_neumann, moore, hex, random, small_world
    num_neighbors: int = 4  # neighbor count for random topology
    rewire_prob: float = 0.1  # rewiring probability for small_world
    seed: Optional[int] = None  # random seed (None = unseeded)
    gol_enabled: bool = False  # Game of Life substrate layer
    gol_coupling: float = 0.1  # strength of GoL signal injected into agents
    gol_density: float = 0.5  # initial fraction of alive GoL cells
    activation: str = "tanh"  # activation function: tanh, sigmoid, relu, linear


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

        # ── activation function ────────────────────────────────
        self._act, self._act_grad = self._build_activation(c.activation)

        # ── topology ──────────────────────────────────────────
        self._nbr = self._build_neighbors()

        # ── live metrics ──────────────────────────────────────
        self.self_scores = np.zeros(N, dtype=np.float32)
        self.pred_errors = np.zeros(N, dtype=np.float32)
        self.phi_scores = np.zeros(N, dtype=np.float32)
        self.reflexivity = np.zeros(N, dtype=np.float32)
        self.temporal_persistence = np.ones(N, dtype=np.float32)
        self.causal_efficacy = np.zeros(N, dtype=np.float32)
        self.tick = 0

        # ── EMA tracking for temporal persistence ─────────────
        self._self_score_ema = np.zeros(N, dtype=np.float32)
        self._self_score_var = np.zeros(N, dtype=np.float32)

        # ── Game of Life substrate ────────────────────────────
        if c.gol_enabled:
            self.gol = (self.rng.random(N) < c.gol_density).astype(np.float32)
            self._gol_nbr = self._build_moore_grid()  # GoL always uses Moore
        else:
            self.gol = None
            self._gol_nbr = None

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
            "mean_phi": [],
            "max_phi": [],
            "mean_R": [],
            "mean_T": [],
            "mean_E": [],
        }

    # ── topology ──────────────────────────────────────────────

    def _build_moore_grid(self) -> np.ndarray:
        """8-connected Moore neighbors on the raw grid (for GoL, not agent comms)."""
        s = self.cfg.size
        g = np.arange(self.N, dtype=np.int32).reshape(s, s)
        n = np.roll(g, 1, axis=0)
        so = np.roll(g, -1, axis=0)
        e = np.roll(g, -1, axis=1)
        w = np.roll(g, 1, axis=1)
        return np.stack([
            n, so, e, w,
            np.roll(n, -1, axis=1), np.roll(n, 1, axis=1),
            np.roll(so, -1, axis=1), np.roll(so, 1, axis=1),
        ], axis=-1).reshape(self.N, 8)

    @staticmethod
    def _build_activation(name):
        """Return (activation_fn, gradient_fn) pair for the given name."""
        if name == "tanh":
            def act(x):
                return np.tanh(x)
            def grad(output):
                return 1.0 - output ** 2
        elif name == "sigmoid":
            def act(x):
                return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))
            def grad(output):
                return output * (1.0 - output)
        elif name == "relu":
            def act(x):
                return np.clip(x, 0, 1)
            def grad(output):
                return ((output > 0) & (output < 1)).astype(np.float32)
        elif name == "linear":
            def act(x):
                return np.clip(x, -1, 1)
            def grad(output):
                return ((output > -1) & (output < 1)).astype(np.float32)
        else:
            raise ValueError(f"Unknown activation: {name}")
        return act, grad

    def _build_neighbors(self) -> np.ndarray:
        """Build neighbor index array based on configured topology."""
        builders = {
            "von_neumann": self._nbr_von_neumann,
            "moore": self._nbr_moore,
            "hex": self._nbr_hex,
            "random": self._nbr_random,
            "small_world": self._nbr_small_world,
        }
        builder = builders.get(self.cfg.topology)
        if builder is None:
            raise ValueError(f"Unknown topology: {self.cfg.topology}")
        return builder()

    def _nbr_von_neumann(self) -> np.ndarray:
        """4-connected neighbors on a toroidal grid."""
        s = self.cfg.size
        g = np.arange(self.N, dtype=np.int32).reshape(s, s)
        return np.stack([
            np.roll(g, 1, axis=0),
            np.roll(g, -1, axis=0),
            np.roll(g, -1, axis=1),
            np.roll(g, 1, axis=1),
        ], axis=-1).reshape(self.N, 4)

    def _nbr_moore(self) -> np.ndarray:
        """8-connected neighbors (includes diagonals)."""
        s = self.cfg.size
        g = np.arange(self.N, dtype=np.int32).reshape(s, s)
        n = np.roll(g, 1, axis=0)
        so = np.roll(g, -1, axis=0)
        e = np.roll(g, -1, axis=1)
        w = np.roll(g, 1, axis=1)
        return np.stack([
            n, so, e, w,
            np.roll(n, -1, axis=1),   # NE
            np.roll(n, 1, axis=1),    # NW
            np.roll(so, -1, axis=1),  # SE
            np.roll(so, 1, axis=1),   # SW
        ], axis=-1).reshape(self.N, 8)

    def _nbr_hex(self) -> np.ndarray:
        """6-connected hexagonal neighbors (offset coordinates)."""
        s = self.cfg.size
        g = np.arange(self.N, dtype=np.int32).reshape(s, s)
        n = np.roll(g, 1, axis=0)
        so = np.roll(g, -1, axis=0)
        e = np.roll(g, -1, axis=1)
        w = np.roll(g, 1, axis=1)
        even = np.zeros((s, s), dtype=bool)
        even[0::2, :] = True
        d1 = np.where(even, np.roll(n, 1, axis=1), np.roll(n, -1, axis=1))
        d2 = np.where(even, np.roll(so, 1, axis=1), np.roll(so, -1, axis=1))
        return np.stack([n, so, e, w, d1, d2], axis=-1).reshape(self.N, 6)

    def _nbr_random(self) -> np.ndarray:
        """k random neighbors per agent."""
        k = self.cfg.num_neighbors
        nbrs = np.zeros((self.N, k), dtype=np.int32)
        for i in range(self.N):
            # Sample k from [0, N-1] excluding i
            candidates = self.rng.choice(self.N - 1, size=k, replace=False)
            candidates[candidates >= i] += 1
            nbrs[i] = candidates
        return nbrs

    def _nbr_small_world(self) -> np.ndarray:
        """Von Neumann base with random rewiring (Watts-Strogatz style)."""
        base = self._nbr_von_neumann()
        p = self.cfg.rewire_prob
        for i in range(self.N):
            for j in range(base.shape[1]):
                if self.rng.random() < p:
                    new = self.rng.integers(0, self.N)
                    while new == i:
                        new = self.rng.integers(0, self.N)
                    base[i, j] = int(new)
        return base

    # ── simulation ────────────────────────────────────────────

    def step(self) -> None:
        """Advance the world by one tick."""
        c = self.cfg
        old = self.states.copy()

        # 1  BROADCAST: each agent transforms its state into a message
        #    mᵢ = act(Wᵢ · sᵢ)
        msg = self._act(np.einsum("nij,nj->ni", self.W, self.states))

        # 2  LOSSY CHANNEL: add Gaussian noise
        noisy = msg + self.rng.standard_normal(msg.shape).astype(np.float32) * c.noise

        # 3  RECEIVE: each agent averages messages from its neighbors
        recv = noisy[self._nbr].mean(axis=1)  # (N, D)

        # ── God Mode: isolated agents receive no neighbor signals ──
        if np.any(self.isolated):
            recv[self.isolated] = 0.0

        # 3b GAME OF LIFE: inject GoL substrate signal into received input
        if self.gol is not None:
            self._step_gol()
            gol_signal = self._gol_observe()
            recv = recv + c.gol_coupling * gol_signal

        # 4  UPDATE: blend persistence, incoming signals, and random drive
        #    The drive prevents state collapse to zero
        drive = (
            self.rng.standard_normal(self.states.shape).astype(np.float32) * c.drive
        )
        self.states = self._act(c.persistence * self.states + (1 - c.persistence) * recv + drive)

        # 5  SELF-MODEL SCORE: cosine similarity between what the agent
        #    broadcast (trained for predicting others) and what it became.
        #    This is NEVER explicitly optimized — emergence only.
        n_msg = np.linalg.norm(msg, axis=1) + 1e-8
        n_st = np.linalg.norm(self.states, axis=1) + 1e-8
        dot = np.einsum("ni,ni->n", msg, self.states)
        self.self_scores = dot / (n_msg * n_st)

        # ── God Mode: dead agents stay dead ────────────────────
        if np.any(self.dead):
            self.states[self.dead] = 0.0
            self.self_scores[self.dead] = 0.0

        # 6  LEARN: update W to reduce prediction error on neighbors
        nbr_new = self.states[self._nbr]  # (N, K, D)
        err = msg[:, None, :] - nbr_new  # (N, K, D)
        avg_err = err.mean(axis=1)  # (N, D)
        self.pred_errors = np.linalg.norm(avg_err, axis=1)

        #    Gradient through activation function
        grad_act = self._act_grad(msg)  # (N, D)
        scaled = avg_err * grad_act  # (N, D)
        dW = np.einsum("ni,nj->nij", scaled, old)  # (N, D, D)

        self.W -= c.lr * dW
        self.W *= 1.0 - c.weight_decay  # prevent divergence
        np.clip(self.W, -3, 3, out=self.W)

        # ── God Mode: dead agents don't learn ──────────────────
        if np.any(self.dead):
            self.W[self.dead] = 0.0
            self.pred_errors[self.dead] = 0.0

        # 7  PHI: approximate integrated information per agent
        self._compute_phi(old, msg)

        # 8  R (REFLEXIVITY): self-prediction minus neighbor-prediction
        #    Positive R = agent is better at predicting itself than others
        dot_nbr = np.einsum("ni,nki->nk", msg, nbr_new)  # (N, K)
        n_nbr = np.linalg.norm(nbr_new, axis=2) + 1e-8  # (N, K)
        cos_nbr = dot_nbr / (n_msg[:, None] * n_nbr)  # (N, K)
        self.reflexivity = (self.self_scores - cos_nbr.mean(axis=1)).astype(np.float32)

        # 9  T (TEMPORAL PERSISTENCE): stability of self-model over time
        beta = 0.95
        delta_score = self.self_scores - self._self_score_ema
        self._self_score_ema += (1.0 - beta) * delta_score
        self._self_score_var = beta * self._self_score_var + (1.0 - beta) * delta_score ** 2
        self.temporal_persistence = np.clip(
            1.0 - np.sqrt(self._self_score_var + 1e-8), 0.0, 1.0
        ).astype(np.float32)

        # 10 E (CAUSAL EFFICACY): how self-determined is the trajectory?
        #    Compare actual state change to counterfactual without neighbors
        self_only = self._act(c.persistence * old + drive)
        delta_actual = self.states - old  # (N, D)
        delta_self = self_only - old  # (N, D)
        dot_e = np.einsum("ni,ni->n", delta_actual, delta_self)
        n_da = np.linalg.norm(delta_actual, axis=1) + 1e-8
        n_ds = np.linalg.norm(delta_self, axis=1) + 1e-8
        self.causal_efficacy = (dot_e / (n_da * n_ds)).astype(np.float32)

        # ── God Mode: zero metrics for dead agents ──────────────
        if np.any(self.dead):
            self.reflexivity[self.dead] = 0.0
            self.temporal_persistence[self.dead] = 0.0
            self.causal_efficacy[self.dead] = 0.0

        # 11 RECORD
        self.tick += 1
        h = self.history
        h["tick"].append(self.tick)
        h["mean_self"].append(float(self.self_scores.mean()))
        h["max_self"].append(float(self.self_scores.max()))
        h["p95_self"].append(float(np.percentile(self.self_scores, 95)))
        h["std_self"].append(float(self.self_scores.std()))
        h["mean_err"].append(float(self.pred_errors.mean()))
        h["mean_phi"].append(float(self.phi_scores.mean()))
        h["max_phi"].append(float(self.phi_scores.max()))
        h["mean_R"].append(float(self.reflexivity.mean()))
        h["mean_T"].append(float(self.temporal_persistence.mean()))
        h["mean_E"].append(float(self.causal_efficacy.mean()))

    def _compute_phi(self, old_states: np.ndarray, messages: np.ndarray) -> None:
        """
        Approximate integrated information (Phi) per agent.

        For each agent, measure how much its state transition depends on
        the integrated whole of its neighborhood vs. the parts independently.
        """
        nbr_states_old = old_states[self._nbr]  # (N, K, D)

        # joint: how well does the full neighborhood predict the agent's change?
        delta = self.states - old_states  # (N, D)
        nbr_mean = nbr_states_old.mean(axis=1)  # (N, D)
        joint_resid = delta - nbr_mean  # (N, D)
        joint_var = np.sum(joint_resid ** 2, axis=1)  # (N,)

        # parts: average of each single-neighbor prediction residuals (vectorized)
        parts_resid = delta[:, None, :] - nbr_states_old  # (N, K, D)
        parts_var = np.sum(parts_resid ** 2, axis=2).mean(axis=1)  # (N,)

        # phi = how much better the whole predicts than the average part
        raw_phi = parts_var - joint_var
        self.phi_scores = np.maximum(raw_phi / (parts_var + 1e-8), 0.0).astype(np.float32)

        if np.any(self.dead):
            self.phi_scores[self.dead] = 0.0

    # ── Game of Life substrate ────────────────────────────────

    def _step_gol(self) -> None:
        """Advance the Game of Life grid by one generation (B3/S23)."""
        alive = self.gol[self._gol_nbr]  # (N, 8)
        count = alive.sum(axis=1)  # (N,) live neighbor count
        # B3/S23: birth if 3 neighbors, survive if 2 or 3
        birth = (self.gol == 0) & (count == 3)
        survive = (self.gol == 1) & ((count == 2) | (count == 3))
        self.gol = (birth | survive).astype(np.float32)

    def _gol_observe(self) -> np.ndarray:
        """Each agent observes its GoL cell and local GoL neighborhood."""
        cell = self.gol  # (N,)
        nbr_alive = self.gol[self._gol_nbr].mean(axis=1)  # (N,) fraction alive
        signal = np.zeros((self.N, self.D), dtype=np.float32)
        signal[:, 0::2] = (cell[:, None] - 0.5)  # center: [0,1] -> [-0.5, 0.5]
        signal[:, 1::2] = (nbr_alive[:, None] - 0.5)
        return signal

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
            "phi_scores": self.phi_scores.copy(),
            "reflexivity": self.reflexivity.copy(),
            "temporal_persistence": self.temporal_persistence.copy(),
            "causal_efficacy": self.causal_efficacy.copy(),
            "config": vars(self.cfg),
        }

    def grid_scores(self) -> np.ndarray:
        """Self-model scores reshaped to the 2D grid."""
        return self.self_scores.reshape(self.cfg.size, self.cfg.size)

    def grid_errors(self) -> np.ndarray:
        """Prediction errors reshaped to the 2D grid."""
        return self.pred_errors.reshape(self.cfg.size, self.cfg.size)

    def grid_phi(self) -> np.ndarray:
        """Phi scores reshaped to the 2D grid."""
        return self.phi_scores.reshape(self.cfg.size, self.cfg.size)

    def grid_reflexivity(self) -> np.ndarray:
        """Reflexivity scores reshaped to the 2D grid."""
        return self.reflexivity.reshape(self.cfg.size, self.cfg.size)

    def grid_persistence(self) -> np.ndarray:
        """Temporal persistence reshaped to the 2D grid."""
        return self.temporal_persistence.reshape(self.cfg.size, self.cfg.size)

    def grid_efficacy(self) -> np.ndarray:
        """Causal efficacy reshaped to the 2D grid."""
        return self.causal_efficacy.reshape(self.cfg.size, self.cfg.size)
