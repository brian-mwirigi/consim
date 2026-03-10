#!/usr/bin/env python3
"""
consim — consciousness simulation

We didn't program awareness. We only programmed communication.
Something happened.
"""

import argparse
import sys
import numpy as np

from world import World, Config


def main():
    p = argparse.ArgumentParser(
        description="consim — can awareness emerge from communication?",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python run.py                                    # default 48x48, live view\n"
            "  python run.py --size 64 --seed 42                # larger grid, reproducible\n"
            "  python run.py --topology moore --size 32         # 8-connected neighbors\n"
            "  python run.py --headless --ticks 10000           # no GUI, just numbers\n"
            "  python run.py --headless --ticks 5000 --output results.npz\n"
            "  python run.py --sweep --sweep-seeds 1-20         # batch sweep across seeds\n"
        ),
    )
    p.add_argument("--size", type=int, default=48, help="grid side length (default: 48)")
    p.add_argument("--dim", type=int, default=8, help="state dimensions per agent (default: 8)")
    p.add_argument("--noise", type=float, default=0.12, help="communication noise sigma (default: 0.12)")
    p.add_argument("--lr", type=float, default=0.003, help="learning rate (default: 0.003)")
    p.add_argument("--persistence", type=float, default=0.3, help="state persistence (default: 0.3)")
    p.add_argument("--drive", type=float, default=0.02, help="random perturbation strength (default: 0.02)")
    p.add_argument("--seed", type=int, default=None, help="random seed (default: None)")
    p.add_argument("--topology", type=str, default="von_neumann",
                   choices=["von_neumann", "moore", "hex", "random", "small_world"],
                   help="neighbor topology (default: von_neumann)")
    p.add_argument("--num-neighbors", type=int, default=4, help="neighbor count for random topology (default: 4)")
    p.add_argument("--rewire-prob", type=float, default=0.1, help="rewiring probability for small_world (default: 0.1)")
    p.add_argument("--headless", action="store_true", help="run without visualization")
    p.add_argument("--ticks", type=int, default=5000, help="max ticks in headless mode (default: 5000)")
    p.add_argument("--output", type=str, default=None, help="save final state to .npz file")
    p.add_argument("--record", type=str, default=None, help="record grid evolution to GIF file")
    p.add_argument("--record-ticks", type=int, default=2000, help="ticks to record (default: 2000)")
    p.add_argument("--fps", type=int, default=24, help="GIF frame rate (default: 24)")
    p.add_argument("--sweep", action="store_true", help="run parameter sweep")
    p.add_argument("--sweep-seeds", type=str, default="1-10", help="seed range for sweep, e.g. 1-20 (default: 1-10)")
    p.add_argument("--sweep-topos", type=str, default="von_neumann,moore,hex",
                   help="topologies for sweep, comma-separated (default: von_neumann,moore,hex)")
    p.add_argument("--sweep-noises", type=str, default="0.12",
                   help="noise levels for sweep, comma-separated (default: 0.12)")
    p.add_argument("--sweep-csv", type=str, default="sweep_results.csv", help="output CSV for sweep (default: sweep_results.csv)")
    p.add_argument("--gol", action="store_true", help="enable Game of Life substrate layer")
    p.add_argument("--activation", type=str, default="tanh",
                   help="activation function: tanh, sigmoid, relu, linear (default: tanh)")
    p.add_argument("--gol-coupling", type=float, default=0.1, help="GoL signal coupling strength (default: 0.1)")
    p.add_argument("--gol-density", type=float, default=0.5, help="initial GoL cell density (default: 0.5)")
    args = p.parse_args()

    cfg = Config(
        size=args.size,
        dim=args.dim,
        noise=args.noise,
        lr=args.lr,
        persistence=args.persistence,
        drive=args.drive,
        topology=args.topology,
        num_neighbors=args.num_neighbors,
        rewire_prob=args.rewire_prob,
        seed=args.seed,
        gol_enabled=args.gol,
        gol_coupling=args.gol_coupling,
        gol_density=args.gol_density,
        activation=args.activation,
    )

    world = World(cfg)
    n_agents = cfg.size ** 2

    print()
    print("  consim")
    print("  " + "\u2500" * 40)
    print(f"  {cfg.size}\u00d7{cfg.size} grid  \u00b7  {n_agents:,} agents  \u00b7  {cfg.dim}D state")
    print(f"  noise={cfg.noise}  lr={cfg.lr}  persistence={cfg.persistence}  drive={cfg.drive}")
    print(f"  topology={cfg.topology}  activation={cfg.activation}")
    if cfg.gol_enabled:
        print(f"  GoL substrate: coupling={cfg.gol_coupling}  density={cfg.gol_density}")
    if cfg.seed is not None:
        print(f"  seed={cfg.seed}")
    print()

    if args.sweep:
        # parse seed range
        parts = args.sweep_seeds.split("-")
        seed_start, seed_end = int(parts[0]), int(parts[1])
        seeds = list(range(seed_start, seed_end + 1))
        topos = [t.strip() for t in args.sweep_topos.split(",")]

        noises = [float(n.strip()) for n in args.sweep_noises.split(",")]

        from analysis import run_sweep
        total_runs = len(seeds) * len(topos) * len(noises)
        print(f"  Sweep: {len(seeds)} seeds x {len(topos)} topologies x {len(noises)} noise levels = {total_runs} runs")
        print(f"  {args.ticks} ticks each, output to {args.sweep_csv}")
        print()
        run_sweep(
            seeds=seeds,
            topologies=topos,
            ticks=args.ticks,
            size=args.size,
            dim=args.dim,
            noises=noises,
            lr=args.lr,
            persistence=args.persistence,
            drive=args.drive,
            num_neighbors=args.num_neighbors,
            rewire_prob=args.rewire_prob,
            output_csv=args.sweep_csv,
            gol_enabled=args.gol,
            gol_coupling=args.gol_coupling,
            gol_density=args.gol_density,
            activation=args.activation,
        )

    elif args.record:
        # ── record mode: GIF output, no window ─────────────
        try:
            from visualizer import record_gif
        except ImportError as e:
            print(f"  Error: matplotlib and Pillow are required for recording.")
            print(f"  Install them:  pip install matplotlib Pillow")
            sys.exit(1)

        record_gif(
            world,
            path=args.record,
            ticks=args.record_ticks,
            fps=args.fps,
        )

    elif args.headless:
        # ── headless mode: print progress, optionally save ────
        print(f"  Running {args.ticks:,} ticks (headless) ...")
        print(f"  {'tick':>8s}  \u2502  {'mean_self':>10s}  {'max_self':>9s}  {'p95_self':>9s}  \u2502  {'pred_err':>9s}  {'mean_phi':>9s}  \u2502  {'R':>7s}  {'T':>7s}  {'E':>7s}")
        print("  " + "\u2500" * 92)

        for t in range(1, args.ticks + 1):
            world.step()
            if t % 500 == 0 or t == 1:
                ss = world.self_scores
                print(
                    f"  {t:>8,}  \u2502  {ss.mean():>+10.5f}  {ss.max():>+9.4f}  "
                    f"{np.percentile(ss, 95):>+9.4f}  \u2502  {world.pred_errors.mean():>9.5f}"
                    f"  {world.phi_scores.mean():>9.5f}  \u2502  {world.reflexivity.mean():>+7.4f}"
                    f"  {world.temporal_persistence.mean():>7.4f}  {world.causal_efficacy.mean():>+7.4f}"
                )

        ss = world.self_scores
        print("  " + "\u2500" * 92)
        print(
            f"  Done.  mean_self={ss.mean():+.5f}  max_self={ss.max():+.4f}  "
            f"phi={world.phi_scores.mean():.5f}  "
            f"R={world.reflexivity.mean():+.4f}  T={world.temporal_persistence.mean():.4f}  "
            f"E={world.causal_efficacy.mean():+.4f}"
        )

        if args.output:
            h = world.history
            np.savez_compressed(
                args.output,
                states=world.states,
                W=world.W,
                self_scores=world.self_scores,
                pred_errors=world.pred_errors,
                phi_scores=world.phi_scores,
                history_tick=np.array(h["tick"]),
                history_mean_self=np.array(h["mean_self"]),
                history_max_self=np.array(h["max_self"]),
                history_p95_self=np.array(h["p95_self"]),
                history_std_self=np.array(h["std_self"]),
                history_mean_err=np.array(h["mean_err"]),
                history_mean_phi=np.array(h["mean_phi"]),
                history_max_phi=np.array(h["max_phi"]),
                history_mean_R=np.array(h["mean_R"]),
                history_mean_T=np.array(h["mean_T"]),
                history_mean_E=np.array(h["mean_E"]),
                reflexivity=world.reflexivity,
                temporal_persistence=world.temporal_persistence,
                causal_efficacy=world.causal_efficacy,
                cfg_size=np.int32(cfg.size),
                cfg_dim=np.int32(cfg.dim),
                cfg_noise=np.float32(cfg.noise),
                cfg_lr=np.float32(cfg.lr),
                cfg_persistence=np.float32(cfg.persistence),
            )
            print(f"\n  Saved to {args.output}")
        print()

    else:
        # ── live visualization ────────────────────────────────
        try:
            from visualizer import run_live
        except ImportError as e:
            print(f"  Error: matplotlib is required for live mode.")
            print(f"  Install it:  pip install matplotlib")
            print(f"  Or run headless:  python run.py --headless")
            sys.exit(1)

        print("  Launching visualization...")
        print("  Close the window to stop.\n")
        run_live(world)


if __name__ == "__main__":
    main()
