"""
Static counterpart to animate_chain_property.py: a 2 x 3 grid of frames
showing the support of x_t and nabla bar f(x_t) at six representative
iterations, illustrating how the chain property advances the support of
the iterate by exactly one coordinate per gradient call.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def main() -> None:
    d = 16
    snapshot_ts = [0, 2, 4, 6, 8, 10]

    T = 2 * np.eye(d) - np.eye(d, k=1) - np.eye(d, k=-1)
    e1 = np.zeros(d)
    e1[0] = 1.0
    eta = 1.0 / 4.0  # 1/beta upper bound

    n_steps = max(snapshot_ts)
    iterates = np.zeros((n_steps + 1, d))
    grads = np.zeros((n_steps + 1, d))
    for t in range(n_steps + 1):
        grads[t] = T @ iterates[t] - e1
        if t < n_steps:
            iterates[t + 1] = iterates[t] - eta * grads[t]

    max_x = max(np.abs(iterates).max(), 0.05) * 1.1
    max_g = max(np.abs(grads).max(), 0.05) * 1.1

    color_active = "#2c7bb6"
    color_inactive = "#e6e6e6"
    edge_active = "#114878"
    edge_inactive = "#bbbbbb"

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), dpi=130)
    axes = axes.flatten()

    coords = np.arange(1, d + 1)
    bar_height = 0.7

    for ax, t in zip(axes, snapshot_ts):
        x_support = t
        g_support = t + 1

        # Draw x_t bars (filled wide) and gradient bars (slightly narrower, offset)
        # using a single axis: x_t in solid, gradient as outlined hatching.
        # To keep the panel readable, plot two stacked sub-bars per coordinate.
        offset = 0.2
        for i in range(d):
            color_x = color_active if i < x_support else color_inactive
            edge_x = edge_active if i < x_support else edge_inactive
            color_g = color_active if i < g_support else color_inactive
            edge_g = edge_active if i < g_support else edge_inactive

            # Iterate bar
            ax.barh(
                coords[i] - offset, iterates[t, i] / max_x,
                height=bar_height * 0.45, color=color_x, edgecolor=edge_x,
                linewidth=0.6,
            )
            # Gradient bar
            ax.barh(
                coords[i] + offset, grads[t, i] / max_g,
                height=bar_height * 0.45, color=color_g, edgecolor=edge_g,
                linewidth=0.6, hatch="///", alpha=0.85,
            )

        ax.set_xlim(-1.1, 1.1)
        ax.invert_yaxis()
        ax.set_ylim(d + 0.5, 0.5)
        ax.set_yticks(np.arange(1, d + 1, 2))
        ax.axvline(0.0, color="black", linewidth=0.5, alpha=0.6)
        ax.grid(True, axis="x", alpha=0.18)
        ax.set_title(
            rf"$t = {t}$:  $x_t \in E_{{{x_support}}}$,  $\nabla\bar f(x_t)\in E_{{{g_support}}}$",
            fontsize=11,
        )
        ax.set_xlabel(r"value (rescaled)", fontsize=10)
        ax.set_ylabel(r"coordinate $i$", fontsize=10)

    # Legend in figure coordinates
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=color_active, edgecolor=edge_active, label=r"$x_t$ (solid)"),
        Patch(facecolor=color_active, edgecolor=edge_active,
              hatch="///", label=r"$\nabla\bar f(x_t)$ (hatched)"),
        Patch(facecolor=color_inactive, edgecolor=edge_inactive,
              label=r"untouched coordinates"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, 1.01), framealpha=0.95)

    fig.suptitle(
        r"Chain property of $\bar f$: every gradient call advances the support by one coordinate",
        fontsize=13, y=1.04,
    )
    fig.tight_layout()
    out = FIGURES_DIR / "chain_property_snapshots.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
