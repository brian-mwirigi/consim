"""
consim — Connectome Analysis

Run T and E metrics on real biological neural architectures.
Tests whether the structural constraints found in grid topologies
also hold for the C. elegans connectome.

Data source:
  OpenWorm c302 — herm_full_edgelist.csv
  Full hermaphrodite connectome with chemical + electrical synapses.
  Original data from Varshney et al. 2011 and Cook et al. 2019.
  https://github.com/openworm/c302/tree/master/c302/data
"""

import csv
import os
import sys
import numpy as np
from world import World, Config
from analysis import morans_i, state_entropy, phi_entropy, cluster_count


# ═══════════════════════════════════════════════════════════════
#  C. ELEGANS CONNECTOME — REAL DATA LOADER
#
#  Reads the OpenWorm herm_full_edgelist.csv (7,379 edges).
#  Columns: Source, Target, Weight, Type (chemical / electrical)
#  We build a weighted, directed adjacency from this.
# ═══════════════════════════════════════════════════════════════

_DATA_PATH = os.path.join(os.path.dirname(__file__), "celegans_connectome.csv")


def load_celegans_edgelist(path=None):
    """
    Load the real C. elegans connectome from the OpenWorm CSV.

    Returns
    -------
    neurons : list[str]
        Sorted list of unique neuron names.
    edges : list[tuple]
        (source_idx, target_idx, weight, synapse_type) for each edge.
    """
    if path is None:
        path = _DATA_PATH

    neurons_set = set()
    raw_edges = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row["Source"].strip()
            tgt = row["Target"].strip()
            weight = int(row["Weight"].strip())
            syn_type = row["Type"].strip()
            neurons_set.add(src)
            neurons_set.add(tgt)
            raw_edges.append((src, tgt, weight, syn_type))

    neurons = sorted(neurons_set)
    idx = {name: i for i, name in enumerate(neurons)}
    edges = [(idx[s], idx[t], w, tp) for s, t, w, tp in raw_edges]
    return neurons, edges


def build_celegans_adjacency(K=14, rng=None):
    """
    Build a fixed-width adjacency array from the real C. elegans connectome.

    Each neuron's neighbors are chosen from its actual synaptic partners,
    weighted by synapse count. For neurons with more than K partners, we
    sample K neighbors with probability proportional to weight. For neurons
    with fewer, we pad by resampling.

    Returns (N, K, adj_array, neuron_names, degree_info)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    neurons, edges = load_celegans_edgelist()
    N = len(neurons)

    # Build weighted adjacency: adj_weights[i] = {j: total_weight}
    adj_weights = [dict() for _ in range(N)]
    for src, tgt, w, _ in edges:
        adj_weights[src][tgt] = adj_weights[src].get(tgt, 0) + w
        # Electrical synapses are bidirectional; chemical are directed.
        # For consim we treat all connections as bidirectional communication
        adj_weights[tgt][src] = adj_weights[tgt].get(src, 0) + w

    adj_array = np.zeros((N, K), dtype=np.int32)
    for i in range(N):
        partners = adj_weights[i]
        if not partners:
            # Isolated node — self-loop
            adj_array[i] = i
            continue
        nbr_ids = np.array(list(partners.keys()), dtype=np.int32)
        nbr_wts = np.array(list(partners.values()), dtype=np.float64)
        nbr_wts /= nbr_wts.sum()

        if len(nbr_ids) >= K:
            adj_array[i] = rng.choice(nbr_ids, size=K, replace=False, p=nbr_wts)
        else:
            adj_array[i] = rng.choice(nbr_ids, size=K, replace=True, p=nbr_wts)

    degrees = [len(adj_weights[i]) for i in range(N)]
    return N, K, adj_array, neurons, degrees


class ConnectomeWorld(World):
    """
    World that uses an arbitrary adjacency list instead of a grid topology.

    This allows running consim's metrics on real neural architectures.
    """

    def __init__(self, n_neurons, adj_array, dim=8, noise=0.12, lr=0.003,
                 persistence=0.3, drive=0.02, seed=None, activation="tanh"):
        # Build a config with size=1 (we override N manually)
        cfg = Config(
            size=1,  # placeholder — we override N
            dim=dim,
            noise=noise,
            lr=lr,
            persistence=persistence,
            drive=drive,
            seed=seed,
            activation=activation,
        )
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        N = n_neurons
        D = dim
        self.N, self.D = N, D

        # Agent internals
        self.states = self.rng.standard_normal((N, D)).astype(np.float32) * 0.3
        self.W = self.rng.standard_normal((N, D, D)).astype(np.float32) * (0.5 / D)
        eye = np.eye(D, dtype=np.float32)
        self.W += eye[None, :, :] * 0.15

        # Activation
        self._act, self._act_grad = self._build_activation(activation)

        # Use the provided adjacency
        self._nbr = adj_array

        # Metrics
        self.self_scores = np.zeros(N, dtype=np.float32)
        self.pred_errors = np.zeros(N, dtype=np.float32)
        self.phi_scores = np.zeros(N, dtype=np.float32)
        self.reflexivity = np.zeros(N, dtype=np.float32)
        self.temporal_persistence = np.ones(N, dtype=np.float32)
        self.causal_efficacy = np.zeros(N, dtype=np.float32)
        self.tick = 0

        self._self_score_ema = np.zeros(N, dtype=np.float32)
        self._self_score_var = np.zeros(N, dtype=np.float32)

        self.dead = np.zeros(N, dtype=bool)
        self.isolated = np.zeros(N, dtype=bool)
        self.gol = None
        self._gol_nbr = None

        self.history = {
            "tick": [], "mean_self": [], "max_self": [], "p95_self": [],
            "std_self": [], "mean_err": [], "mean_phi": [], "max_phi": [],
            "mean_R": [], "mean_T": [], "mean_E": [],
        }


def run_connectome_sweep(seeds, ticks=1000, K=14,
                         dim=8, noises=None, output_csv=None):
    """
    Run T-E analysis on the real C. elegans connectome across seeds and noise levels.
    """
    if noises is None:
        noises = [0.04, 0.08, 0.12, 0.20, 0.30]

    results = []
    total = len(seeds) * len(noises)
    done = 0

    for noise in noises:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            N, K_actual, adj_array, neuron_names, degrees = build_celegans_adjacency(K=K, rng=rng)

            world = ConnectomeWorld(
                n_neurons=N, adj_array=adj_array, dim=dim,
                noise=noise, seed=seed
            )

            for t in range(1, ticks + 1):
                world.step()

            ss = world.self_scores
            row = {
                "connectome": "c_elegans",
                "seed": seed,
                "noise": noise,
                "tick": ticks,
                "N": N,
                "K": K_actual,
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
                f"\r  [{done}/{total}] c_elegans noise={noise:.2f} seed={seed:<4d} "
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


def analyze_connectome_results(results, connectome_name):
    """Print analysis of T-E metrics from connectome runs."""
    print()
    print("=" * 70)
    print(f"CONNECTOME ANALYSIS: {connectome_name}")
    print("=" * 70)

    T_vals = np.array([r["mean_T"] for r in results])
    E_vals = np.array([r["mean_E"] for r in results])
    phi_vals = np.array([r["mean_phi"] for r in results])
    self_vals = np.array([r["mean_self"] for r in results])

    print(f"\n  N = {results[0]['N']} neurons, K = {results[0]['K']} neighbors")
    print(f"  Runs: {len(results)}")
    print(f"\n  T (temporal persistence): {T_vals.mean():.4f} ± {T_vals.std():.4f}")
    print(f"  E (causal efficacy):      {E_vals.mean():+.4f} ± {E_vals.std():.4f}")
    print(f"  Phi (integration):        {phi_vals.mean():.4f} ± {phi_vals.std():.4f}")
    print(f"  Self-model:               {self_vals.mean():+.4f} ± {self_vals.std():.4f}")

    # T-E correlation
    if len(results) >= 3:
        r_TE = np.corrcoef(T_vals, E_vals)[0, 1]
        r_phiE = np.corrcoef(phi_vals, E_vals)[0, 1]
        print(f"\n  r(T, E)   = {r_TE:+.4f}")
        print(f"  r(Phi, E) = {r_phiE:+.4f}")

    # Noise effect
    noises = sorted(set(r["noise"] for r in results))
    if len(noises) > 1:
        print(f"\n  Noise sensitivity:")
        for noise in noises:
            subset = [r for r in results if r["noise"] == noise]
            T = np.mean([r["mean_T"] for r in subset])
            E = np.mean([r["mean_E"] for r in subset])
            print(f"    σ={noise:.2f}: T={T:.4f}  E={E:+.4f}")


def compare_to_grid_topologies():
    """Load grid topology results and compare with connectome results."""
    print()
    print("=" * 70)
    print("COMPARISON: Connectomes vs Grid Topologies")
    print("=" * 70)

    # Load grid data
    grid_data = {}
    try:
        with open("sweep_size24_full.csv") as f:
            rows = [r for r in csv.DictReader(f) if r["tick"] == "1000"]
        for topo in ["von_neumann", "moore", "hex", "random", "small_world"]:
            subset = [r for r in rows if r["topology"] == topo]
            if subset:
                T = np.mean([float(r["mean_T"]) for r in subset])
                E = np.mean([float(r["mean_E"]) for r in subset])
                r_TE = np.corrcoef(
                    [float(r["mean_T"]) for r in subset],
                    [float(r["mean_E"]) for r in subset]
                )[0, 1] if len(subset) >= 3 else float("nan")
                grid_data[topo] = {"T": T, "E": E, "r_TE": r_TE, "n": len(subset)}
    except FileNotFoundError:
        print("  (sweep_size24_full.csv not found — skipping grid comparison)")
        return grid_data

    print(f"\n  {'System':>20s}  {'T':>8s}  {'E':>8s}  {'r(T,E)':>8s}  {'n':>4s}")
    print("  " + "-" * 56)
    for topo, d in grid_data.items():
        K_map = {"von_neumann": 4, "moore": 8, "hex": 6, "random": 4, "small_world": 4}
        print(f"  {'grid/' + topo:>20s}  {d['T']:8.4f}  {d['E']:+8.4f}  {d['r_TE']:+8.4f}  {d['n']:4d}")

    return grid_data


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    seeds = list(range(1, 21))  # 20 seeds

    # Print data summary
    neurons, edges = load_celegans_edgelist()
    n_chem = sum(1 for _, _, _, t in edges if t == "chemical")
    n_elec = sum(1 for _, _, _, t in edges if t == "electrical")
    print(f"\nC. elegans connectome loaded:")
    print(f"  {len(neurons)} neurons, {len(edges)} edges")
    print(f"  Chemical synapses: {n_chem}")
    print(f"  Electrical (gap junctions): {n_elec}")

    # ── C. elegans ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"Running C. elegans connectome ({len(neurons)} neurons, K=14)...")
    print("=" * 70)

    ce_results = run_connectome_sweep(
        seeds, ticks=1000, output_csv="sweep_celegans.csv"
    )
    analyze_connectome_results(ce_results, f"C. elegans ({len(neurons)} neurons)")

    # ── Compare to grid topologies ────────────────────────────
    grid_data = compare_to_grid_topologies()

    # Print connectome results in same format
    T = np.mean([r["mean_T"] for r in ce_results])
    E = np.mean([r["mean_E"] for r in ce_results])
    r_TE = np.corrcoef(
        [r["mean_T"] for r in ce_results],
        [r["mean_E"] for r in ce_results]
    )[0, 1] if len(ce_results) >= 3 else float("nan")
    print(f"  {'C. elegans':>20s}  {T:8.4f}  {E:+8.4f}  {r_TE:+8.4f}  {len(ce_results):4d}")

    print()
    print("Done. Connectome CSV saved.")
