"""
consim — Real-time Visualization

Renders the simulation as a live matplotlib figure:
  - Left:  NxN heatmap of self-model scores (dark = no self-model, bright = emergent)
  - Top-right:  Self-modeling score over time (mean / p95 / max)
  - Bottom-right:  Prediction error over time
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

from world import World

# ── color palette ─────────────────────────────────────────────
# Deep void → ocean → teal → gold → fire
_NODES = [
    (0.00, "#05050f"),
    (0.15, "#0b1128"),
    (0.30, "#122a5e"),
    (0.45, "#1a5276"),
    (0.60, "#1abc9c"),
    (0.78, "#f1c40f"),
    (0.90, "#e67e22"),
    (1.00, "#e74c3c"),
]
CMAP = LinearSegmentedColormap.from_list(
    "emerge", [(p, c) for p, c in _NODES], N=256
)

BG = "#08080c"
FG_DIM = "#555"
FG_MID = "#999"
FG = "#ccc"
TEAL = "#1abc9c"
GOLD = "#f1c40f"
ORANGE = "#f39c12"
RED = "#e74c3c"


def run_live(
    world: World,
    interval: int = 40,
    steps_per_frame: int = 4,
) -> None:
    """
    Launch a real-time matplotlib window.

    Args:
        world:           Initialized World instance.
        interval:        Milliseconds between frames.
        steps_per_frame: Simulation ticks per rendered frame.
    """
    s = world.cfg.size

    # ── figure layout ─────────────────────────────────────────
    fig = plt.figure(figsize=(15, 7.5), facecolor=BG)
    fig.canvas.manager.set_window_title("consim")
    fig.suptitle(
        "consim",
        color=FG,
        fontsize=18,
        fontweight="bold",
        fontfamily="monospace",
        y=0.97,
    )

    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.3, 1],
        hspace=0.38,
        wspace=0.28,
        left=0.05,
        right=0.97,
        top=0.90,
        bottom=0.07,
    )

    # ── grid panel (left) ─────────────────────────────────────
    ax_g = fig.add_subplot(gs[:, 0])
    ax_g.set_title(
        "self-model score per agent",
        color=FG_MID,
        fontsize=10,
        fontfamily="monospace",
        pad=8,
    )
    ax_g.set_xticks([])
    ax_g.set_yticks([])

    blank = np.zeros((s, s))
    im = ax_g.imshow(
        blank, cmap=CMAP, vmin=-1, vmax=1, interpolation="nearest", aspect="equal"
    )
    cb = fig.colorbar(im, ax=ax_g, fraction=0.046, pad=0.03, shrink=0.85)
    cb.ax.tick_params(colors=FG_DIM, labelsize=7)
    cb.outline.set_edgecolor("#222")

    tick_lbl = ax_g.text(
        0.02,
        0.97,
        "",
        transform=ax_g.transAxes,
        color=TEAL,
        fontsize=10,
        va="top",
        fontfamily="monospace",
    )
    stats_lbl = ax_g.text(
        0.98,
        0.97,
        "",
        transform=ax_g.transAxes,
        color=FG_DIM,
        fontsize=8,
        va="top",
        ha="right",
        fontfamily="monospace",
    )

    # ── self-model curve (top-right) ──────────────────────────
    ax_s = fig.add_subplot(gs[0, 1])
    ax_s.set_title(
        "self-modeling emergence",
        color=FG_MID,
        fontsize=10,
        fontfamily="monospace",
    )
    ax_s.set_ylabel("score", color=FG_DIM, fontsize=9)

    (ln_mean,) = ax_s.plot([], [], color=TEAL, lw=1.6, label="mean")
    (ln_p95,) = ax_s.plot([], [], color=ORANGE, lw=1.0, alpha=0.8, label="p95")
    (ln_max,) = ax_s.plot([], [], color=RED, lw=0.8, alpha=0.6, label="max")
    ax_s.axhline(0, color="#333", lw=0.5, ls="--")
    ax_s.legend(
        fontsize=7,
        loc="upper left",
        facecolor=BG,
        edgecolor="#333",
        labelcolor=FG_DIM,
    )
    ax_s.set_xlim(0, 300)
    ax_s.set_ylim(-0.5, 1.0)

    # ── prediction error curve (bottom-right) ─────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    ax_e.set_title(
        "prediction error",
        color=FG_MID,
        fontsize=10,
        fontfamily="monospace",
    )
    ax_e.set_xlabel("tick", color=FG_DIM, fontsize=9)
    ax_e.set_ylabel("\u2016error\u2016", color=FG_DIM, fontsize=9)
    (ln_err,) = ax_e.plot([], [], color=RED, lw=1.4)
    ax_e.set_xlim(0, 300)
    ax_e.set_ylim(0, 1.0)

    # ── style all axes ────────────────────────────────────────
    for ax in (ax_g, ax_s, ax_e):
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG_DIM, labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#222")

    # ── animation loop ────────────────────────────────────────
    def _update(_frame):
        for _ in range(steps_per_frame):
            world.step()

        # grid
        im.set_data(world.grid_scores())

        # curves
        h = world.history
        t = h["tick"]
        ln_mean.set_data(t, h["mean_self"])
        ln_p95.set_data(t, h["p95_self"])
        ln_max.set_data(t, h["max_self"])
        ln_err.set_data(t, h["mean_err"])

        # auto-scroll x
        if t and t[-1] > ax_s.get_xlim()[1] * 0.85:
            xlim = t[-1] * 1.4
            ax_s.set_xlim(0, xlim)
            ax_e.set_xlim(0, xlim)

        # auto-scale error y
        if h["mean_err"]:
            recent = h["mean_err"][-300:]
            mx = max(recent) * 1.3 if recent else 1.0
            ax_e.set_ylim(0, max(mx, 0.05))

        # labels
        tick_lbl.set_text(f"tick {world.tick:,}")
        ss = world.self_scores
        stats_lbl.set_text(f"mean {ss.mean():+.3f}  max {ss.max():+.3f}")

        return [im, ln_mean, ln_p95, ln_max, ln_err, tick_lbl, stats_lbl]

    _ani = animation.FuncAnimation(
        fig,
        _update,
        interval=interval,
        blit=False,
        cache_frame_data=False,
    )

    plt.show()
