"""
consim — Second-Order Perception

Implements Fitz's MCH second-order prediction: agents predict their OWN
next broadcast, not just their neighbors' states. Tests whether this
changes the structural constraints on T and E.

Each agent gets a second weight matrix V_i that learns to predict its
own next broadcast, and that prediction feeds back into the state update:

  First-order:  m_i = act(W_i · s_i)           — predict neighbors
  Second-order: m̂_i^next = act(V_i · s_i)      — predict own next output
  Feedback:     s_i ← act(α·s_i + (1-α)·r_i + γ·m̂_i^next + drive)
  Second-order loss: ||m_i^{t+1} - m̂_i^next||²

The self-prediction has causal influence on the agent's dynamics.
γ controls how strongly the self-model shapes behavior.
"""

import csv
import sys
import numpy as np
from world import World, Config
from analysis import morans_i, state_entropy, phi_entropy


class SecondOrderWorld(World):
    """
    World with second-order perception: agents predict their own next broadcast.

    Inherits all first-order dynamics from World, adds V_i weight matrix
    and second-order learning.
    """

    def __init__(self, cfg=None, lr2=0.003, gamma=0.05):
        """
        Parameters
        ----------
        cfg : Config
            Standard consim config.
        lr2 : float
            Learning rate for second-order prediction weights V_i.
        gamma : float
            Feedback strength of V's prediction into state update.
        """
        super().__init__(cfg)
        self.lr2 = lr2
        self.gamma = gamma

        # Second-order weight matrix: predicts own next message from current state
        self.V = self.rng.standard_normal(
            (self.N, self.D, self.D)
        ).astype(np.float32) * (0.5 / self.D)
        # Slight identity bias
        eye = np.eye(self.D, dtype=np.float32)
        self.V += eye[None, :, :] * 0.15

        # Store last message and states for second-order learning
        self._last_msg = np.zeros((self.N, self.D), dtype=np.float32)
        self._last_predicted_msg = np.zeros((self.N, self.D), dtype=np.float32)
        self._prev_states = np.zeros((self.N, self.D), dtype=np.float32)  # states_t for V gradient

        # Second-order specific metrics
        self.self_pred_error = np.zeros(self.N, dtype=np.float32)

        # Extend history
        self.history["mean_self_pred_err"] = []
        self.history["mean_second_order_score"] = []

    def step(self):
        """Advance world by one tick with second-order perception."""
        c = self.cfg
        old = self.states.copy()

        # ── FIRST ORDER (same as base World) ──────────────────

        # 1. Broadcast
        msg = self._act(np.einsum("nij,nj->ni", self.W, self.states))

        # ── SECOND ORDER: compare prediction from last tick ───
        if self.tick > 0:
            # How well did V predict this tick's message?
            so_err = msg - self._last_predicted_msg  # (N, D)
            self.self_pred_error = np.linalg.norm(so_err, axis=1)

            # Second-order self-model score: cosine sim between predicted and actual
            n_pred = np.linalg.norm(self._last_predicted_msg, axis=1) + 1e-8
            n_actual = np.linalg.norm(msg, axis=1) + 1e-8
            dot_so = np.einsum("ni,ni->n", self._last_predicted_msg, msg)
            so_score = dot_so / (n_pred * n_actual)
        else:
            so_score = np.zeros(self.N, dtype=np.float32)

        # ── SECOND ORDER PREDICTION for next tick ─────────────
        # V_i predicts what m_i will be NEXT tick, using current states
        predicted_next_msg = self._act(np.einsum("nij,nj->ni", self.V, self.states))
        self._last_predicted_msg = predicted_next_msg.copy()
        self._last_msg = msg.copy()
        self._prev_states = old.copy()  # Bug 1 fix: save states_t for V gradient next tick

        # 2. Noise
        noisy = msg + self.rng.standard_normal(msg.shape).astype(np.float32) * c.noise

        # 3. Receive
        recv = noisy[self._nbr].mean(axis=1)

        if np.any(self.isolated):
            recv[self.isolated] = 0.0

        # 3b. GoL
        if self.gol is not None:
            self._step_gol()
            gol_signal = self._gol_observe()
            recv = recv + c.gol_coupling * gol_signal

        # 4. Update — Bug 2 fix: inject V's prediction with strength γ
        drive = self.rng.standard_normal(self.states.shape).astype(np.float32) * c.drive
        self.states = self._act(
            c.persistence * self.states
            + (1 - c.persistence) * recv
            + self.gamma * predicted_next_msg
            + drive
        )

        # 5. Self-model score (first order)
        n_msg = np.linalg.norm(msg, axis=1) + 1e-8
        n_st = np.linalg.norm(self.states, axis=1) + 1e-8
        dot = np.einsum("ni,ni->n", msg, self.states)
        self.self_scores = dot / (n_msg * n_st)

        if np.any(self.dead):
            self.states[self.dead] = 0.0
            self.self_scores[self.dead] = 0.0

        # 6. Learn W (first-order: predict neighbors)
        nbr_new = self.states[self._nbr]
        err = msg[:, None, :] - nbr_new
        avg_err = err.mean(axis=1)
        self.pred_errors = np.linalg.norm(avg_err, axis=1)

        grad_act = self._act_grad(msg)
        scaled = avg_err * grad_act
        dW = np.einsum("ni,nj->nij", scaled, old)
        self.W -= c.lr * dW
        self.W *= 1.0 - self.cfg.weight_decay
        np.clip(self.W, -3, 3, out=self.W)

        # ── Learn V (second-order: predict own next message) ──
        if self.tick > 0:
            # Gradient of ||msg - predicted_last||² w.r.t. V
            grad_v_act = self._act_grad(self._last_predicted_msg)
            # Bug 1 fix: use _prev_states (states at tick t-1 when prediction was made)
            so_scaled = so_err * grad_v_act
            dV = np.einsum("ni,nj->nij", so_scaled, self._prev_states)
            self.V += self.lr2 * dV
            self.V *= 1.0 - self.cfg.weight_decay
            np.clip(self.V, -3, 3, out=self.V)

        if np.any(self.dead):
            self.W[self.dead] = 0.0
            self.V[self.dead] = 0.0
            self.pred_errors[self.dead] = 0.0

        # 7. Phi
        self._compute_phi(old, msg)

        # 8. R (reflexivity)
        dot_nbr = np.einsum("ni,nki->nk", msg, nbr_new)
        n_nbr = np.linalg.norm(nbr_new, axis=2) + 1e-8
        cos_nbr = dot_nbr / (n_msg[:, None] * n_nbr)
        self.reflexivity = (self.self_scores - cos_nbr.mean(axis=1)).astype(np.float32)

        # 9. T (temporal persistence)
        beta = 0.95
        delta_score = self.self_scores - self._self_score_ema
        self._self_score_ema += (1.0 - beta) * delta_score
        self._self_score_var = beta * self._self_score_var + (1.0 - beta) * delta_score ** 2
        self.temporal_persistence = np.clip(
            1.0 - np.sqrt(self._self_score_var + 1e-8), 0.0, 1.0
        ).astype(np.float32)

        # 10. E (causal efficacy)
        self_only = self._act(c.persistence * old + self.gamma * predicted_next_msg + drive)
        delta_actual = self.states - old
        delta_self = self_only - old
        dot_e = np.einsum("ni,ni->n", delta_actual, delta_self)
        n_da = np.linalg.norm(delta_actual, axis=1) + 1e-8
        n_ds = np.linalg.norm(delta_self, axis=1) + 1e-8
        self.causal_efficacy = (dot_e / (n_da * n_ds)).astype(np.float32)

        if np.any(self.dead):
            self.reflexivity[self.dead] = 0.0
            self.temporal_persistence[self.dead] = 0.0
            self.causal_efficacy[self.dead] = 0.0

        # 11. Record
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
        h["mean_self_pred_err"].append(float(self.self_pred_error.mean()))
        h["mean_second_order_score"].append(float(so_score.mean()))


def run_gamma_sweep(seeds, gammas, noises, ticks=1000, size=24,
                    lr2=0.003, output_csv="sweep_second_order.csv"):
    """
    Run γ × noise dose-response sweep on Moore topology.

    Tests: does second-order perception change the pooled r(T,E)?
    """
    results = []
    total = len(seeds) * len(gammas) * len(noises)
    done = 0

    for gamma in gammas:
        for noise in noises:
            for seed in seeds:
                cfg = Config(
                    size=size, dim=8, noise=noise, lr=0.003,
                    persistence=0.3, drive=0.02, topology="moore", seed=seed,
                )
                world = SecondOrderWorld(cfg, lr2=lr2, gamma=gamma)

                for t in range(1, ticks + 1):
                    world.step()

                ss = world.self_scores
                row = {
                    "topology": "moore",
                    "seed": seed,
                    "noise": noise,
                    "gamma": gamma,
                    "tick": ticks,
                    "size": size,
                    "lr2": lr2,
                    "mean_self": round(float(ss.mean()), 6),
                    "max_self": round(float(ss.max()), 6),
                    "p95_self": round(float(np.percentile(ss, 95)), 6),
                    "std_self": round(float(ss.std()), 6),
                    "mean_phi": round(float(world.phi_scores.mean()), 6),
                    "max_phi": round(float(world.phi_scores.max()), 6),
                    "mean_R": round(float(world.reflexivity.mean()), 6),
                    "mean_T": round(float(world.temporal_persistence.mean()), 6),
                    "mean_E": round(float(world.causal_efficacy.mean()), 6),
                    "mean_err": round(float(world.pred_errors.mean()), 6),
                    "mean_self_pred_err": round(float(world.self_pred_error.mean()), 6),
                }
                results.append(row)

                done += 1
                sys.stdout.write(
                    f"\r  [{done}/{total}] γ={gamma:.2f} noise={noise:.2f} seed={seed:<4d} "
                    f"T={world.temporal_persistence.mean():.4f} "
                    f"E={world.causal_efficacy.mean():+.4f} "
                    f"so_err={world.self_pred_error.mean():.4f}"
                )
                sys.stdout.flush()

    print()

    if output_csv:
        fieldnames = list(results[0].keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"  Saved {len(results)} rows to {output_csv}")

    return results


def analyze_gamma_response(results):
    """Analyze dose-response: how does γ affect the T-E constraint?"""
    from numpy.linalg import eigh

    gammas = sorted(set(r["gamma"] for r in results))
    metrics = ["mean_self", "mean_phi", "mean_R", "mean_T", "mean_E"]

    # Load first-order Moore baseline for comparison
    baseline_rTE = float("nan")
    baseline_T = float("nan")
    baseline_E = float("nan")
    try:
        with open("sweep_size24_full.csv") as f:
            fo_rows = [r for r in csv.DictReader(f)
                       if r["tick"] == "1000" and r["topology"] == "moore"]
        if len(fo_rows) >= 3:
            T_fo = [float(r["mean_T"]) for r in fo_rows]
            E_fo = [float(r["mean_E"]) for r in fo_rows]
            baseline_rTE = np.corrcoef(T_fo, E_fo)[0, 1]
            baseline_T = np.mean(T_fo)
            baseline_E = np.mean(E_fo)
    except FileNotFoundError:
        pass

    print()
    print("=" * 70)
    print("γ DOSE-RESPONSE: Second-Order Feedback on Moore (size 24)")
    print("=" * 70)
    print(f"\n  First-order baseline (pooled): T={baseline_T:.4f}  E={baseline_E:+.4f}  r(T,E)={baseline_rTE:+.4f}")

    # KEY ANALYSIS: pooled r(T,E) across all noise levels per gamma
    print(f"\n  {'γ':>6s}  {'T':>8s}  {'E':>8s}  {'r(T,E)':>8s}  {'self':>8s}  {'Phi':>8s}  {'so_err':>8s}  {'n':>4s}")
    print("  " + "-" * 72)

    for gamma in gammas:
        sub = [r for r in results if r["gamma"] == gamma]
        T_vals = np.array([r["mean_T"] for r in sub])
        E_vals = np.array([r["mean_E"] for r in sub])
        phi_vals = np.array([r["mean_phi"] for r in sub])
        self_vals = np.array([r["mean_self"] for r in sub])
        so_err = np.array([r["mean_self_pred_err"] for r in sub])

        r_TE = np.corrcoef(T_vals, E_vals)[0, 1] if len(sub) >= 3 else float("nan")

        print(f"  {gamma:6.2f}  {T_vals.mean():8.4f}  {E_vals.mean():+8.4f}  "
              f"{r_TE:+8.4f}  {self_vals.mean():8.4f}  "
              f"{phi_vals.mean():8.4f}  {so_err.mean():8.4f}  {len(sub):4d}")

    # Within-noise breakdown
    noises_seen = sorted(set(r["noise"] for r in results))
    print(f"\n  Within-noise r(T,E) by γ:")
    print(f"  {'noise':>6s}", end="")
    for gamma in gammas:
        print(f"  γ={gamma:.2f}", end="")
    print()
    for noise in noises_seen:
        print(f"  {noise:6.2f}", end="")
        for gamma in gammas:
            sub = [r for r in results if r["gamma"] == gamma and r["noise"] == noise]
            T_vals = [r["mean_T"] for r in sub]
            E_vals = [r["mean_E"] for r in sub]
            c = np.corrcoef(T_vals, E_vals)[0, 1] if len(sub) >= 3 else float("nan")
            print(f"  {c:+.4f}", end="")
        print()

    # Transition detection
    print(f"\n  Does POOLED r(T,E) weaken with γ?")
    rte_by_gamma = []
    for gamma in gammas:
        sub = [r for r in results if r["gamma"] == gamma]
        T_vals = [r["mean_T"] for r in sub]
        E_vals = [r["mean_E"] for r in sub]
        rte_by_gamma.append(np.corrcoef(T_vals, E_vals)[0, 1])

    if not np.isnan(baseline_rTE):
        print(f"    Baseline (no V): r(T,E) = {baseline_rTE:+.4f}")
    for i, gamma in enumerate(gammas):
        delta = rte_by_gamma[i] - baseline_rTE if not np.isnan(baseline_rTE) else float("nan")
        print(f"    γ={gamma:.2f}:          r(T,E) = {rte_by_gamma[i]:+.4f}  Δ = {delta:+.4f}")


if __name__ == "__main__":
    seeds = list(range(1, 21))  # 20 seeds
    gammas = [0.00, 0.01, 0.05, 0.10, 0.20]
    noises = [0.04, 0.08, 0.12, 0.20, 0.30]

    print("\n" + "=" * 70)
    print("SECOND-ORDER PERCEPTION: γ × NOISE SWEEP")
    print(f"  Moore topology, size 24")
    print(f"  {len(seeds)} seeds × {len(gammas)} γ values × {len(noises)} noise levels = {len(seeds) * len(gammas) * len(noises)} runs")
    print("=" * 70)

    results = run_gamma_sweep(
        seeds, gammas, noises, ticks=1000, size=24,
        output_csv="sweep_second_order.csv"
    )

    analyze_gamma_response(results)

    print("\nDone.")
