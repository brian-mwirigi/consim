"""
consim — Analysis Tools

Spatial statistics, information metrics, and batch sweep runner
for systematic experimentation.
"""

import csv
import sys
import numpy as np
from typing import List, Dict, Any
from world import World, Config


def morans_i(world: World) -> float:
    """
    Moran's I spatial autocorrelation of self-model scores.

    Returns a value in [-1, 1]:
      +1 = perfect positive spatial autocorrelation (clusters)
       0 = random spatial distribution
      -1 = perfect dispersion (checkerboard)
    """
    x = world.self_scores
    n = world.N
    x_bar = x.mean()
    z = x - x_bar
    denom = np.sum(z ** 2)
    if denom < 1e-12:
        return 0.0

    nbr = world._nbr
    K = nbr.shape[1]
    W_total = n * K  # total weight (binary adjacency)

    # sum of z_i * z_j for all neighbor pairs
    nbr_z = z[nbr]  # (N, K)
    numerator = np.sum(z[:, None] * nbr_z)

    return float((n / W_total) * (numerator / denom))


def state_entropy(world: World, bins: int = 20) -> float:
    """
    Shannon entropy of the self-model score distribution.

    Higher entropy = more diverse scores. Lower = more uniform (all similar).
    """
    counts, _ = np.histogram(world.self_scores, bins=bins, range=(-1, 1))
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def phi_entropy(world: World, bins: int = 20) -> float:
    """Shannon entropy of the Phi score distribution."""
    mx = max(world.phi_scores.max(), 0.01)
    counts, _ = np.histogram(world.phi_scores, bins=bins, range=(0, mx))
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def cluster_count(world: World, threshold: float = 0.5) -> int:
    """
    Count connected components of agents with self-model score above threshold.

    Uses BFS on the neighbor graph. Only meaningful for grid topologies.
    """
    above = world.self_scores > threshold
    visited = np.zeros(world.N, dtype=bool)
    count = 0

    for start in range(world.N):
        if not above[start] or visited[start]:
            continue
        count += 1
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            for nb in world._nbr[node]:
                if above[nb] and not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
    return count


def run_sweep(
    seeds: List[int],
    topologies: List[str],
    ticks: int = 2000,
    size: int = 24,
    dim: int = 8,
    noises: List[float] = None,
    lr: float = 0.003,
    persistence: float = 0.3,
    output_csv: str = "sweep_results.csv",
    sample_interval: int = 500,
) -> List[Dict[str, Any]]:
    """
    Run a parameter sweep across seeds, topologies, and noise levels.

    Saves results to CSV and returns them as a list of dicts.
    """
    if noises is None:
        noises = [0.12]
    results = []
    total = len(seeds) * len(topologies) * len(noises)
    done = 0

    for noise in noises:
        for topo in topologies:
            for seed in seeds:
                cfg = Config(
                    size=size, dim=dim, noise=noise, lr=lr,
                    persistence=persistence, topology=topo, seed=seed,
                )
                world = World(cfg)

                for t in range(1, ticks + 1):
                    world.step()

                    if t % sample_interval == 0 or t == ticks:
                        ss = world.self_scores
                        row = {
                            "topology": topo,
                            "seed": seed,
                            "noise": noise,
                            "tick": t,
                            "size": size,
                            "mean_self": round(float(ss.mean()), 6),
                            "max_self": round(float(ss.max()), 6),
                            "p95_self": round(float(np.percentile(ss, 95)), 6),
                            "std_self": round(float(ss.std()), 6),
                            "mean_phi": round(float(world.phi_scores.mean()), 6),
                            "max_phi": round(float(world.phi_scores.max()), 6),
                            "mean_err": round(float(world.pred_errors.mean()), 6),
                            "morans_i": round(morans_i(world), 6),
                            "entropy": round(state_entropy(world), 4),
                            "phi_entropy": round(phi_entropy(world), 4),
                            "clusters_05": cluster_count(world, 0.5),
                            "clusters_07": cluster_count(world, 0.7),
                        }
                        results.append(row)

                done += 1
                ss = world.self_scores
                sys.stdout.write(
                    f"\r  [{done}/{total}] noise={noise:.2f} {topo:15s} seed={seed:<4d} "
                    f"mean_self={ss.mean():+.4f}  max_self={ss.max():+.4f}  "
                    f"phi={world.phi_scores.mean():.4f}  moran={morans_i(world):+.4f}"
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
