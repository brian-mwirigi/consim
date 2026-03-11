"""
consim — K* Phase Transition Finder

Sweep connectivity K from 3 to 12 using random topology (which allows
exact control of K) to find the critical K* where the T-E constraint
structure undergoes a phase transition.

From existing data:
  - K=4 (von_neumann): strong T-E anti-correlation, high dimensionality constraint
  - K=6 (hex): intermediate
  - K=8 (moore): weaker constraint

The question: where exactly does the transition happen?
"""

import csv
import sys
import numpy as np
from numpy.linalg import eigh
from world import World, Config
from analysis import morans_i, state_entropy


def run_k_sweep(seeds, k_values, ticks=1000, size=24,
                noises=None, output_csv="sweep_k_transition.csv"):
    """
    Run sweep across K values using random topology.

    Random topology is key because it lets us vary K independently of
    spatial structure. Grid topologies confound K with spatial correlation.
    """
    if noises is None:
        noises = [0.04, 0.08, 0.12, 0.20, 0.30]

    results = []
    total = len(seeds) * len(k_values) * len(noises)
    done = 0

    for noise in noises:
        for K in k_values:
            for seed in seeds:
                cfg = Config(
                    size=size, dim=8, noise=noise, lr=0.003,
                    persistence=0.3, drive=0.02,
                    topology="random", num_neighbors=K,
                    seed=seed,
                )
                world = World(cfg)

                for t in range(1, ticks + 1):
                    world.step()

                ss = world.self_scores
                row = {
                    "K": K,
                    "seed": seed,
                    "noise": noise,
                    "tick": ticks,
                    "size": size,
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
                }
                results.append(row)

                done += 1
                sys.stdout.write(
                    f"\r  [{done}/{total}] K={K:<3d} noise={noise:.2f} seed={seed:<4d} "
                    f"T={world.temporal_persistence.mean():.4f} "
                    f"E={world.causal_efficacy.mean():+.4f} "
                    f"phi={world.phi_scores.mean():.4f}"
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


def analyze_k_transition(results):
    """Find K* — the critical connectivity for the T-E phase transition."""
    print()
    print("=" * 70)
    print("K* PHASE TRANSITION ANALYSIS")
    print("=" * 70)

    k_values = sorted(set(r["K"] for r in results))
    metrics = ["mean_self", "mean_phi", "mean_R", "mean_T", "mean_E"]

    print(f"\n  {'K':>3s}  {'T':>8s}  {'E':>8s}  {'r(T,E)':>8s}  {'PC1':>8s}  {'dims95':>6s}  {'Phi':>8s}  {'n':>4s}")
    print("  " + "-" * 66)

    k_stats = {}
    for K in k_values:
        subset = [r for r in results if r["K"] == K]
        T_vals = np.array([r["mean_T"] for r in subset])
        E_vals = np.array([r["mean_E"] for r in subset])
        phi_vals = np.array([r["mean_phi"] for r in subset])

        r_TE = np.corrcoef(T_vals, E_vals)[0, 1] if len(subset) >= 3 else float("nan")

        # PCA
        X = np.array([[r[m] for m in metrics] for r in subset])
        Z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
        C = np.cov(Z.T)
        eigvals, _ = eigh(C)
        eigvals = eigvals[::-1]
        ve = eigvals / eigvals.sum()
        cum = np.cumsum(ve)
        n95 = int(np.searchsorted(cum, 0.95)) + 1

        k_stats[K] = {
            "T": T_vals.mean(), "E": E_vals.mean(),
            "r_TE": r_TE, "PC1": ve[0], "n95": n95,
            "phi": phi_vals.mean(), "n": len(subset)
        }

        print(f"  {K:3d}  {T_vals.mean():8.4f}  {E_vals.mean():+8.4f}  "
              f"{r_TE:+8.4f}  {ve[0]:8.4f}  {n95:6d}  "
              f"{phi_vals.mean():8.4f}  {len(subset):4d}")

    # ── Find K*: maximum change in r(T,E) ──────────────────────
    print()
    print("=" * 70)
    print("PHASE TRANSITION DETECTION")
    print("=" * 70)

    ks = sorted(k_stats.keys())
    rte_values = [k_stats[k]["r_TE"] for k in ks]

    # First derivative of r(T,E) with respect to K
    print(f"\n  {'K→K+1':>8s}  {'Δr(T,E)':>10s}  {'|Δ|':>8s}")
    print("  " + "-" * 30)

    max_delta = 0
    k_star = None
    for i in range(len(ks) - 1):
        delta = rte_values[i + 1] - rte_values[i]
        abs_delta = abs(delta)
        marker = " ◀ MAX" if abs_delta > max_delta else ""
        if abs_delta > max_delta:
            max_delta = abs_delta
            k_star = (ks[i], ks[i + 1])
        print(f"  {ks[i]:>3d}→{ks[i+1]:<3d}  {delta:+10.4f}  {abs_delta:8.4f}{marker}")

    if k_star:
        print(f"\n  ★ K* ≈ {k_star[0]}–{k_star[1]}")
        print(f"    Maximum change in r(T,E) occurs between K={k_star[0]} and K={k_star[1]}")
        print(f"    r(T,E) at K={k_star[0]}: {k_stats[k_star[0]]['r_TE']:+.4f}")
        print(f"    r(T,E) at K={k_star[1]}: {k_stats[k_star[1]]['r_TE']:+.4f}")

    # ── Noise sensitivity by K ─────────────────────────────────
    print()
    print("=" * 70)
    print("NOISE SENSITIVITY BY K")
    print("=" * 70)

    noises = sorted(set(r["noise"] for r in results))
    noise_lo, noise_hi = min(noises), max(noises)

    print(f"\n  {'K':>3s}  {'T(lo)':>8s}  {'T(hi)':>8s}  {'ΔT':>8s}  "
          f"{'E(lo)':>8s}  {'E(hi)':>8s}  {'ΔE':>8s}")
    print("  " + "-" * 60)

    for K in k_values:
        lo = [r for r in results if r["K"] == K and r["noise"] == noise_lo]
        hi = [r for r in results if r["K"] == K and r["noise"] == noise_hi]
        if lo and hi:
            T_lo = np.mean([r["mean_T"] for r in lo])
            T_hi = np.mean([r["mean_T"] for r in hi])
            E_lo = np.mean([r["mean_E"] for r in lo])
            E_hi = np.mean([r["mean_E"] for r in hi])
            print(f"  {K:3d}  {T_lo:8.4f}  {T_hi:8.4f}  {T_hi - T_lo:+8.4f}  "
                  f"{E_lo:+8.4f}  {E_hi:+8.4f}  {E_hi - E_lo:+8.4f}")

    return k_stats


if __name__ == "__main__":
    seeds = list(range(1, 21))  # 20 seeds
    k_values = list(range(3, 13))  # K=3 through K=12

    print("\n" + "=" * 70)
    print("K* PHASE TRANSITION SWEEP")
    print(f"  K = {k_values[0]}..{k_values[-1]}")
    print(f"  {len(seeds)} seeds × {len(k_values)} K values × 5 noise levels")
    print(f"  = {len(seeds) * len(k_values) * 5} total runs")
    print("=" * 70)

    results = run_k_sweep(
        seeds, k_values, ticks=1000, size=24,
        output_csv="sweep_k_transition.csv"
    )

    k_stats = analyze_k_transition(results)

    # Also compare with grid topologies at matching K
    print()
    print("=" * 70)
    print("RANDOM K vs GRID TOPOLOGIES (at matching K)")
    print("=" * 70)

    try:
        with open("sweep_size24_full.csv") as f:
            grid_rows = [r for r in csv.DictReader(f) if r["tick"] == "1000"]

        grid_compare = {
            "von_neumann": 4,
            "hex": 6,
            "moore": 8,
        }

        print(f"\n  {'System':>20s}  {'K':>3s}  {'T':>8s}  {'E':>8s}  {'r(T,E)':>8s}")
        print("  " + "-" * 50)

        for topo, K in grid_compare.items():
            sub = [r for r in grid_rows if r["topology"] == topo]
            if sub:
                T = np.mean([float(r["mean_T"]) for r in sub])
                E = np.mean([float(r["mean_E"]) for r in sub])
                r_TE = np.corrcoef(
                    [float(r["mean_T"]) for r in sub],
                    [float(r["mean_E"]) for r in sub]
                )[0, 1] if len(sub) >= 3 else float("nan")
                print(f"  {'grid/' + topo:>20s}  {K:3d}  {T:8.4f}  {E:+8.4f}  {r_TE:+8.4f}")

            # Matching random K
            if K in k_stats:
                s = k_stats[K]
                print(f"  {'random K=' + str(K):>20s}  {K:3d}  {s['T']:8.4f}  {s['E']:+8.4f}  {s['r_TE']:+8.4f}")

    except FileNotFoundError:
        print("  (sweep_size24_full.csv not found)")

    print("\nDone.")
