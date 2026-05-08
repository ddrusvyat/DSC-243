"""
Animate the chain property of the tridiagonal "hard" quadratic
    bar f(z) = (1/2) z^T T z - e_1^T z,
where T is the d x d tridiagonal matrix with 2 on the diagonal and -1 on the
sub/super-diagonals. Each gradient evaluation extends the support of the
iterate by exactly one new coordinate -- this is the engine behind Lemma 9.1.

Two side-by-side horizontal bar charts:
    left  -- coordinate values (x_t)_i
    right -- coordinate values of the gradient nabla bar f(x_t) = T x_t - e_1
Coordinates inside the active support are colored; coordinates that no
gradient query has touched yet are grayed out.

We illustrate by running gradient descent on bar f from x_0 = 0. The chain
property gives x_t in E_t and nabla bar f(x_t) in E_{t+1} at every step.
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
    d = 16
    n_steps = 12

    # Tridiagonal T: 2 on diagonal, -1 off.
    T = 2 * np.eye(d) - np.eye(d, k=1) - np.eye(d, k=-1)
    e1 = np.zeros(d)
    e1[0] = 1.0

    beta = 4.0  # upper bound on lambda_max(T)
    eta = 1.0 / beta

    iterates = np.zeros((n_steps + 1, d))
    grads = np.zeros((n_steps + 1, d))
    for t in range(n_steps + 1):
        grads[t] = T @ iterates[t] - e1
        if t < n_steps:
            iterates[t + 1] = iterates[t] - eta * grads[t]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 6.5), dpi=110, sharey=True)
    ax_x, ax_g = axes

    coords = np.arange(1, d + 1)
    bar_height = 0.72

    max_x = max(np.abs(iterates).max(), 0.05) * 1.15
    max_g = max(np.abs(grads).max(), 0.05) * 1.15

    color_active = "#2c7bb6"
    color_inactive = "#e6e6e6"
    edge_active = "#114878"
    edge_inactive = "#bbbbbb"

    bars_x = ax_x.barh(
        coords, np.zeros(d), height=bar_height,
        color=color_inactive, edgecolor=edge_inactive, linewidth=0.7,
    )
    bars_g = ax_g.barh(
        coords, np.zeros(d), height=bar_height,
        color=color_inactive, edgecolor=edge_inactive, linewidth=0.7,
    )

    for ax, lim in ((ax_x, max_x), (ax_g, max_g)):
        ax.set_xlim(-lim, lim)
        ax.invert_yaxis()
        ax.set_ylim(d + 0.5, 0.5)
        ax.set_yticks(np.arange(1, d + 1))
        ax.axvline(0.0, color="black", linewidth=0.6, alpha=0.6)
        ax.grid(True, axis="x", alpha=0.18)

    ax_x.set_xlabel(r"$\,(x_t)_i$", fontsize=12)
    ax_g.set_xlabel(r"$\,(\nabla \bar f(x_t))_i$", fontsize=12)
    ax_x.set_ylabel(r"coordinate $i$", fontsize=12)

    title_x = ax_x.set_title("", fontsize=12)
    title_g = ax_g.set_title("", fontsize=12)
    suptitle = fig.suptitle("", fontsize=14)

    def update(frame_index):
        t = min(frame_index, n_steps)

        # Chain property: x_t in E_t, nabla bar f(x_t) in E_{t+1}.
        x_support = t
        g_support = t + 1

        for i, bar in enumerate(bars_x):
            bar.set_width(iterates[t, i])
            if i < x_support:
                bar.set_color(color_active)
                bar.set_edgecolor(edge_active)
            else:
                bar.set_color(color_inactive)
                bar.set_edgecolor(edge_inactive)
        for i, bar in enumerate(bars_g):
            bar.set_width(grads[t, i])
            if i < g_support:
                bar.set_color(color_active)
                bar.set_edgecolor(edge_active)
            else:
                bar.set_color(color_inactive)
                bar.set_edgecolor(edge_inactive)

        title_x.set_text(rf"$x_t \in E_{{{x_support}}}$")
        title_g.set_text(rf"$\nabla \bar f(x_t) \in E_{{{g_support}}}$")
        suptitle.set_text(rf"After $t = {t}$ gradient queries on $\bar f$")

        return list(bars_x) + list(bars_g) + [title_x, title_g, suptitle]

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    n_pause = 4
    total_frames = n_steps + 1 + n_pause

    anim = FuncAnimation(fig, update, frames=total_frames, interval=1100, blit=False)

    out = FIGURES_DIR / "chain_property.gif"
    anim.save(out, writer=PillowWriter(fps=1.0))
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
