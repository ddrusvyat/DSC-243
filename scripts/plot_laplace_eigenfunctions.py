"""
Generate a GIF animation of the first eigenfunctions of the Laplace kernel
integral operator on [0,1] with uniform measure.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def compute_eigenfunctions():
    n = 2000
    x = np.linspace(0.0, 1.0, n)

    sigma = 0.1
    D = np.abs(x[:, None] - x[None, :])
    K = np.exp(-D / sigma)
    K_norm = K / n

    eigvals, eigvecs = np.linalg.eigh(K_norm)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    return x, eigvals, eigvecs, n, sigma


def normalize_sign(phi, x, i):
    """Choose a stable sign convention for display."""
    if i == 0 and np.mean(phi) < 0:
        return -phi
    if i == 1 and phi[-1] < phi[0]:
        return -phi
    if i >= 2 and phi[len(x) // 4] < 0:
        return -phi
    return phi


def make_gif():
    x, eigvals, eigvecs, n, sigma = compute_eigenfunctions()

    k_max = 12
    hold = 8  # frames to hold each eigenfunction

    # precompute all eigenfunctions
    phis = []
    for i in range(k_max):
        phi = normalize_sign(np.sqrt(n) * eigvecs[:, i], x, i)
        phis.append(phi)

    ymin = min(phi.min() for phi in phis) * 1.15
    ymax = max(phi.max() for phi in phis) * 1.15

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("$x$", fontsize=13)
    ax.set_ylabel(r"$\phi_i(x)$", fontsize=13)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.grid(True, alpha=0.2)
    title = ax.set_title("", fontsize=13, pad=12)

    cmap = plt.cm.viridis
    curve, = ax.plot([], [], linewidth=2.0)

    def update(frame):
        i = frame // hold
        if i >= k_max:
            i = k_max - 1

        color = cmap(i / (k_max - 1))
        curve.set_data(x, phis[i])
        curve.set_color(color)

        title.set_text(
            rf"$\phi_{{{i+1}}}(x)$,    "
            rf"$\mu_{{{i+1}}} = {eigvals[i]:.4f}$"
        )
        return [curve, title]

    total_frames = k_max * hold
    anim = FuncAnimation(fig, update, frames=total_frames, interval=80, blit=False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    anim.save(FIGURES_DIR / "laplace_eigenfunctions.gif",
              writer=PillowWriter(fps=10), dpi=120)
    plt.close(fig)
    print("  ✓ laplace_eigenfunctions.gif")


if __name__ == "__main__":
    make_gif()
