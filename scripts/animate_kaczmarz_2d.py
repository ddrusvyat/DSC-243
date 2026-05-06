"""
Randomized Kaczmarz on a 2D consistent linear system: animate the iterates
projecting onto a randomly chosen line at each step.

Each row d_i in R^2 is unit-norm and y_i = <d_i, w_*> so the line
ell_i = { w in R^2 : <d_i, w> = y_i } passes through the common point w_*.
At each step the Kaczmarz iterate w_{t+1} is the orthogonal projection of
w_t onto the randomly drawn line ell_{i_t}.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def main() -> None:
    rng = np.random.default_rng(3)

    w_star = np.array([2.0, 1.0])

    # Hand-picked angles to give a visually varied set of five lines through w_*.
    angles = np.array([0.45, 1.40, 2.30, 3.55, 4.70])
    D = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    y = D @ w_star
    n = D.shape[0]

    w0 = np.array([-1.6, -1.8])

    n_iters = 14
    iterates = np.empty((n_iters + 1, 2))
    iterates[0] = w0
    chosen = np.empty(n_iters, dtype=int)

    probs = np.einsum("ij,ij->i", D, D)
    probs /= probs.sum()

    w = w0.copy()
    for t in range(n_iters):
        i = int(rng.choice(n, p=probs))
        chosen[t] = i
        residual = y[i] - float(D[i] @ w)
        w = w + (residual / float(D[i] @ D[i])) * D[i]
        iterates[t + 1] = w

    pad = 1.0
    xmin = float(min(iterates[:, 0].min(), w_star[0]) - pad)
    xmax = float(max(iterates[:, 0].max(), w_star[0]) + pad)
    ymin = float(min(iterates[:, 1].min(), w_star[1]) - pad)
    ymax = float(max(iterates[:, 1].max(), w_star[1]) + pad)
    span = max(xmax - xmin, ymax - ymin)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    xmin, xmax = cx - 0.5 * span, cx + 0.5 * span
    ymin, ymax = cy - 0.5 * span, cy + 0.5 * span

    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=110)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")

    line_colors = plt.cm.tab10(np.arange(n) % 10)
    long_t = np.array([-100.0, 100.0])
    for i in range(n):
        normal = D[i]
        tangent = np.array([-normal[1], normal[0]])
        anchor = y[i] * normal
        pts = anchor[None, :] + long_t[:, None] * tangent[None, :]
        ax.plot(
            pts[:, 0], pts[:, 1], "-", color=line_colors[i],
            linewidth=1.1, alpha=0.35,
        )
        # Label the line at a chosen offset along its length.
        label_pt = anchor + 1.6 * tangent
        if xmin + 0.1 < label_pt[0] < xmax - 0.1 and ymin + 0.1 < label_pt[1] < ymax - 0.1:
            ax.text(
                label_pt[0], label_pt[1], rf"$\ell_{{{i+1}}}$",
                color=line_colors[i], fontsize=10, alpha=0.9,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.7),
            )

    ax.plot(
        w_star[0], w_star[1], marker="*", markersize=18, color="black",
        markerfacecolor="gold", markeredgewidth=1.0, zorder=5,
    )
    ax.annotate(
        r"$w_\ast$", w_star, xytext=(9, 9), textcoords="offset points",
        fontsize=13,
    )

    (trail_line,) = ax.plot([], [], "-", color="#2c7bb6", linewidth=1.0, alpha=0.55)
    (trail_pts,) = ax.plot([], [], "o", color="#2c7bb6", markersize=5)
    (chosen_line,) = ax.plot([], [], "-", color="#d7191c", linewidth=2.5, zorder=4)
    (proj_arrow,) = ax.plot([], [], "--", color="#d7191c", linewidth=1.4, zorder=4)
    (current_pt,) = ax.plot([], [], "o", color="#d7191c", markersize=9, zorder=6)

    title = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="lightgray", alpha=0.95),
    )

    ax.grid(True, alpha=0.2)
    ax.set_xlabel(r"$w^{(1)}$")
    ax.set_ylabel(r"$w^{(2)}$")

    frames_per_step = 2
    n_pause_frames = 6
    total_frames = n_iters * frames_per_step + n_pause_frames

    def update(frame):
        if frame >= n_iters * frames_per_step:
            trail_line.set_data(iterates[:, 0], iterates[:, 1])
            trail_pts.set_data(iterates[:, 0], iterates[:, 1])
            chosen_line.set_data([], [])
            proj_arrow.set_data([], [])
            current_pt.set_data([iterates[-1, 0]], [iterates[-1, 1]])
            title.set_text(rf"trajectory after $t={n_iters}$ steps")
            return trail_line, trail_pts, chosen_line, proj_arrow, current_pt, title

        step = frame // frames_per_step
        sub = frame % frames_per_step
        i = chosen[step]

        trail_line.set_data(iterates[: step + 1, 0], iterates[: step + 1, 1])
        trail_pts.set_data(iterates[: step + 1, 0], iterates[: step + 1, 1])

        normal = D[i]
        tangent = np.array([-normal[1], normal[0]])
        anchor = y[i] * normal
        pts = anchor[None, :] + long_t[:, None] * tangent[None, :]
        chosen_line.set_data(pts[:, 0], pts[:, 1])

        if sub == 0:
            w_t = iterates[step]
            w_next = iterates[step + 1]
            proj_arrow.set_data([w_t[0], w_next[0]], [w_t[1], w_next[1]])
            current_pt.set_data([w_t[0]], [w_t[1]])
            title.set_text(
                rf"$t={step}$: pick $\ell_{{{i+1}}}$, project $w_{{{step}}}$"
            )
        else:
            proj_arrow.set_data([], [])
            current_pt.set_data([iterates[step + 1, 0]], [iterates[step + 1, 1]])
            title.set_text(rf"$t={step}$: land at $w_{{{step+1}}}$")

        return trail_line, trail_pts, chosen_line, proj_arrow, current_pt, title

    anim = FuncAnimation(fig, update, frames=total_frames, interval=1000, blit=False)

    out = FIGURES_DIR / "kaczmarz_2d.gif"
    anim.save(out, writer=PillowWriter(fps=1.0))
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
