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

    # ── God Mode overlays ─────────────────────────────────
    mode_text = ax_g.text(
        0.5, 0.02, "", transform=ax_g.transAxes,
        color=GOLD, fontsize=10, ha="center", va="bottom",
        fontfamily="monospace", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a",
                  edgecolor="#333", alpha=0.9),
    )
    dead_scat = ax_g.scatter(
        [], [], marker="x", c=RED, s=50, linewidths=1.5, zorder=5,
    )
    iso_scat = ax_g.scatter(
        [], [], marker="o", facecolors="none", edgecolors="white",
        s=60, linewidths=1.2, zorder=5,
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

    # ── prediction error / phi curves (bottom-right) ───────────
    ax_e = fig.add_subplot(gs[1, 1])
    ax_e.set_title(
        "prediction error / phi",
        color=FG_MID,
        fontsize=10,
        fontfamily="monospace",
    )
    ax_e.set_xlabel("tick", color=FG_DIM, fontsize=9)
    ax_e.set_ylabel("\u2016error\u2016 / \u03a6", color=FG_DIM, fontsize=9)
    (ln_err,) = ax_e.plot([], [], color=RED, lw=1.4, label="error")
    (ln_phi_mean,) = ax_e.plot([], [], color="#9b59b6", lw=1.4, label="mean \u03a6")
    (ln_phi_max,) = ax_e.plot([], [], color="#8e44ad", lw=0.8, alpha=0.6, label="max \u03a6")
    ax_e.legend(
        fontsize=7,
        loc="upper right",
        facecolor=BG,
        edgecolor="#333",
        labelcolor=FG_DIM,
    )
    ax_e.set_xlim(0, 300)
    ax_e.set_ylim(0, 1.0)

    # ── style all axes ────────────────────────────────────────
    for ax in (ax_g, ax_s, ax_e):
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG_DIM, labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#222")

    # ── animation loop ────────────────────────────────────

    # ── God Mode event handlers ───────────────────────────
    _mode = ["observe"]  # mutable for closure
    _view = ["self"]  # "self" or "phi" heatmap toggle

    def _on_key(event):
        key = event.key
        if key == "k":
            _mode[0] = "kill"
        elif key == "i":
            _mode[0] = "isolate"
        elif key == "j":
            _mode[0] = "inject"
        elif key == "escape":
            _mode[0] = "observe"
        elif key == "p":
            _view[0] = "phi" if _view[0] == "self" else "self"
            title = "self-model score" if _view[0] == "self" else "\u03a6 integrated information"
            ax_g.set_title(title, color=FG_MID, fontsize=10, fontfamily="monospace", pad=8)
        labels = {"kill": "[K]ILL", "isolate": "[I]SOLATE", "inject": "IN[J]ECT"}
        if _mode[0] == "observe":
            mode_text.set_text("")
        else:
            mode_text.set_text(f"GOD MODE: {labels[_mode[0]]}  \u2014  click an agent")

    def _on_click(event):
        if event.inaxes != ax_g or _mode[0] == "observe":
            return
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        if not (0 <= row < s and 0 <= col < s):
            return
        if _mode[0] == "kill":
            world.kill_agent(row, col)
        elif _mode[0] == "isolate":
            world.isolate_agent(row, col)
        elif _mode[0] == "inject":
            world.inject_agent(row, col)

    fig.canvas.mpl_connect("key_press_event", _on_key)
    fig.canvas.mpl_connect("button_press_event", _on_click)

    def _update(_frame):
        for _ in range(steps_per_frame):
            world.step()

        # grid (toggle between self-model and phi)
        if _view[0] == "self":
            im.set_data(world.grid_scores())
        else:
            im.set_data(world.grid_phi())

        # curves
        h = world.history
        t = h["tick"]
        ln_mean.set_data(t, h["mean_self"])
        ln_p95.set_data(t, h["p95_self"])
        ln_max.set_data(t, h["max_self"])
        ln_err.set_data(t, h["mean_err"])
        ln_phi_mean.set_data(t, h["mean_phi"])
        ln_phi_max.set_data(t, h["max_phi"])

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

        # God Mode overlays
        dead_grid = world.dead.reshape(s, s)
        dr, dc = np.where(dead_grid)
        if len(dr):
            dead_scat.set_offsets(np.column_stack([dc, dr]))
        else:
            dead_scat.set_offsets(np.empty((0, 2)))

        iso_grid = world.isolated.reshape(s, s)
        ir, ic = np.where(iso_grid)
        if len(ir):
            iso_scat.set_offsets(np.column_stack([ic, ir]))
        else:
            iso_scat.set_offsets(np.empty((0, 2)))

        return [
            im, ln_mean, ln_p95, ln_max, ln_err, ln_phi_mean, ln_phi_max,
            tick_lbl, stats_lbl, dead_scat, iso_scat, mode_text,
        ]

    _ani = animation.FuncAnimation(
        fig,
        _update,
        interval=interval,
        blit=False,
        cache_frame_data=False,
    )

    plt.show()


def record_gif(
    world: World,
    path: str,
    ticks: int = 2000,
    steps_per_frame: int = 4,
    fps: int = 24,
    dpi: int = 120,
) -> None:
    """
    Record the grid evolution as a GIF file.

    Renders only the heatmap — clean, minimal, ready for social media.
    """
    s = world.cfg.size
    total_frames = ticks // steps_per_frame

    fig, ax = plt.subplots(figsize=(6, 6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#222")

    im_obj = ax.imshow(
        np.zeros((s, s)),
        cmap=CMAP,
        vmin=-1,
        vmax=1,
        interpolation="nearest",
        aspect="equal",
    )

    tick_text = ax.text(
        0.03, 0.97, "", transform=ax.transAxes,
        color=TEAL, fontsize=14, va="top", fontfamily="monospace",
        fontweight="bold",
    )
    score_text = ax.text(
        0.97, 0.97, "", transform=ax.transAxes,
        color=FG_DIM, fontsize=10, va="top", ha="right", fontfamily="monospace",
    )

    fig.tight_layout(pad=0.5)

    def _init():
        im_obj.set_data(np.zeros((s, s)))
        tick_text.set_text("")
        score_text.set_text("")
        return [im_obj, tick_text, score_text]

    def _update(frame):
        for _ in range(steps_per_frame):
            world.step()
        im_obj.set_data(world.grid_scores())
        tick_text.set_text(f"tick {world.tick:,}")
        ss = world.self_scores
        score_text.set_text(f"max {ss.max():+.3f}")
        return [im_obj, tick_text, score_text]

    anim = animation.FuncAnimation(
        fig, _update, init_func=_init,
        frames=total_frames, blit=True,
    )

    print(f"  Recording {total_frames} frames ({ticks} ticks) ...")
    anim.save(path, writer="pillow", fps=fps, dpi=dpi)
    plt.close(fig)
    print(f"  Saved to {path}")
