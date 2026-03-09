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
            "  python run.py                          # default 48x48, live view\n"
            "  python run.py --size 64 --seed 42      # larger grid, reproducible\n"
            "  python run.py --headless --ticks 10000  # no GUI, just numbers\n"
            "  python run.py --headless --ticks 5000 --output results.npz\n"
        ),
    )
    p.add_argument("--size", type=int, default=48, help="grid side length (default: 48)")
    p.add_argument("--dim", type=int, default=8, help="state dimensions per agent (default: 8)")
    p.add_argument("--noise", type=float, default=0.12, help="communication noise sigma (default: 0.12)")
    p.add_argument("--lr", type=float, default=0.003, help="learning rate (default: 0.003)")
    p.add_argument("--persistence", type=float, default=0.3, help="state persistence (default: 0.3)")
    p.add_argument("--seed", type=int, default=None, help="random seed (default: None)")
    p.add_argument("--headless", action="store_true", help="run without visualization")
    p.add_argument("--ticks", type=int, default=5000, help="max ticks in headless mode (default: 5000)")
    p.add_argument("--output", type=str, default=None, help="save final state to .npz file")
    args = p.parse_args()

    cfg = Config(
        size=args.size,
        dim=args.dim,
        noise=args.noise,
        lr=args.lr,
        persistence=args.persistence,
        seed=args.seed,
    )

    world = World(cfg)
    n_agents = cfg.size ** 2

    print()
    print("  consim")
    print("  " + "\u2500" * 40)
    print(f"  {cfg.size}\u00d7{cfg.size} grid  \u00b7  {n_agents:,} agents  \u00b7  {cfg.dim}D state")
    print(f"  noise={cfg.noise}  lr={cfg.lr}  persistence={cfg.persistence}")
    if cfg.seed is not None:
        print(f"  seed={cfg.seed}")
    print()

    if args.headless:
        # ── headless mode: print progress, optionally save ────
        print(f"  Running {args.ticks:,} ticks (headless) ...")
        print(f"  {'tick':>8s}  \u2502  {'mean_self':>10s}  {'max_self':>9s}  {'p95_self':>9s}  \u2502  {'pred_err':>9s}")
        print("  " + "\u2500" * 60)

        for t in range(1, args.ticks + 1):
            world.step()
            if t % 500 == 0 or t == 1:
                ss = world.self_scores
                print(
                    f"  {t:>8,}  \u2502  {ss.mean():>+10.5f}  {ss.max():>+9.4f}  "
                    f"{np.percentile(ss, 95):>+9.4f}  \u2502  {world.pred_errors.mean():>9.5f}"
                )

        ss = world.self_scores
        print("  " + "\u2500" * 60)
        print(f"  Done.  final mean_self = {ss.mean():+.5f}   max_self = {ss.max():+.4f}")

        if args.output:
            h = world.history
            np.savez_compressed(
                args.output,
                states=world.states,
                W=world.W,
                self_scores=world.self_scores,
                pred_errors=world.pred_errors,
                history_tick=np.array(h["tick"]),
                history_mean_self=np.array(h["mean_self"]),
                history_max_self=np.array(h["max_self"]),
                history_p95_self=np.array(h["p95_self"]),
                history_std_self=np.array(h["std_self"]),
                history_mean_err=np.array(h["mean_err"]),
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
