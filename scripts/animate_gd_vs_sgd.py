"""
Matplotlib animation for GD, SGD, and tail-averaged SGD (2000 iterations).

The trajectories are computed at every iteration but the GIF only displays
iterates at multiples of `stride` (default 100), and the viewport is zoomed
in around w_* so the asymptotic behaviour is clearly visible.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Force a local writable matplotlib cache to avoid repeated startup hangs.
ROOT = Path(__file__).resolve().parent.parent
MPL_CACHE = ROOT / ".matplotlib_cache"
MPL_CACHE.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_CACHE)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def gd_iterates(w0: np.ndarray, w_star: np.ndarray, H: np.ndarray, gamma: float, n_iters: int) -> np.ndarray:
    ws = [w0.copy()]
    w = w0.copy()
    for _ in range(n_iters):
        w = w - gamma * (H @ (w - w_star))
        ws.append(w.copy())
    return np.array(ws)


def sgd_iterates(
    w0: np.ndarray, w_star: np.ndarray, H: np.ndarray, sigma: float, gamma: float, n_iters: int, rng: np.random.Generator
) -> np.ndarray:
    L = np.linalg.cholesky(H)
    d = w0.shape[0]
    X = rng.standard_normal((n_iters, d)) @ L.T
    eta = sigma * rng.standard_normal(n_iters)
    Y = X @ w_star + eta
    ws = [w0.copy()]
    w = w0.copy()
    for t in range(n_iters):
        residual = float(X[t] @ w) - float(Y[t])
        w = w - gamma * residual * X[t]
        ws.append(w.copy())
    return np.array(ws)


def tail_average_iterates(ws: np.ndarray) -> np.ndarray:
    n_iters = ws.shape[0] - 1
    pref = np.concatenate([np.zeros((1, ws.shape[1])), np.cumsum(ws, axis=0)], axis=0)
    out = np.empty_like(ws)
    out[0] = ws[0]
    for t in range(1, n_iters + 1):
        a = t // 2
        out[t] = (pref[t] - pref[a]) / (t - a)
    return out


def main() -> None:
    H = np.diag([1.0, 0.25])
    w_star = np.array([2.0, -1.0])
    w0 = np.array([-2.5, 2.0])
    sigma = 0.6
    n_iters = 2000
    stride = 100

    R2 = np.trace(H) + 2.0 * np.max(np.linalg.eigvalsh(H))
    gamma = 0.5 / R2

    rng = np.random.default_rng(3)
    w_gd = gd_iterates(w0, w_star, H, gamma, n_iters)
    w_sgd = sgd_iterates(w0, w_star, H, sigma, gamma, n_iters, rng)
    w_tail = tail_average_iterates(w_sgd)

    # Frames to render: every `stride`-th iterate (and the final iterate).
    frame_indices = list(range(0, n_iters + 1, stride))
    if frame_indices[-1] != n_iters:
        frame_indices.append(n_iters)

    # Fixed zoomed viewport centred on w_*: tight enough for w_* to be the
    # focal point but wide enough to contain the SGD noise floor.
    half_x, half_y = 0.85, 0.7
    x_lo, x_hi = w_star[0] - half_x, w_star[0] + half_x
    y_lo, y_hi = w_star[1] - half_y, w_star[1] + half_y

    gx = np.linspace(x_lo, x_hi, 260)
    gy = np.linspace(y_lo, y_hi, 260)
    GX, GY = np.meshgrid(gx, gy)
    Z = 0.5 * (H[0, 0] * (GX - w_star[0]) ** 2 + H[1, 1] * (GY - w_star[1]) ** 2)
    levels = np.geomspace(1e-4, max(float(Z.max()), 1.0), 28)

    fig, ax = plt.subplots(figsize=(7.0, 5.8), dpi=100)
    ax.contour(GX, GY, Z, levels=levels, colors="0.75", linewidths=0.6)
    ax.set_xlim(float(x_lo), float(x_hi))
    ax.set_ylim(float(y_lo), float(y_hi))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$w^{(1)}$")
    ax.set_ylabel(r"$w^{(2)}$")
    ax.set_title(r"GD, SGD, and tail-averaged SGD ($T=2000,\ \gamma R^2=0.5,\ \sigma=0.6$)")
    ax.plot(w_star[0], w_star[1], marker="*", color="black", markersize=13, label=r"$w_\ast$", zorder=5)

    line_gd, = ax.plot([], [], "-o", color="#1f77b4", linewidth=1.8, markersize=4,
                       label="GD (population)", animated=True)
    line_sgd, = ax.plot([], [], "-o", color="#d62728", linewidth=0.9, markersize=3, alpha=0.55,
                        label="SGD (last iterate)", animated=True)
    line_tail, = ax.plot([], [], "-o", color="#2ca02c", linewidth=1.8, markersize=4,
                         label=r"tail-avg SGD $\overline{w}_{t/2:t}$", animated=True)
    head_gd, = ax.plot([], [], "o", color="#1f77b4", markersize=8,
                       markeredgecolor="white", markeredgewidth=1.0, animated=True)
    head_sgd, = ax.plot([], [], "o", color="#d62728", markersize=7,
                        markeredgecolor="white", markeredgewidth=1.0, animated=True)
    head_tail, = ax.plot([], [], "o", color="#2ca02c", markersize=8,
                         markeredgecolor="white", markeredgewidth=1.0, animated=True)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    fig.tight_layout()

    def init():
        for a in (line_gd, line_sgd, line_tail, head_gd, head_sgd, head_tail):
            a.set_data([], [])
        return line_gd, line_sgd, line_tail, head_gd, head_sgd, head_tail

    def update(t: int):
        keep = np.array([s for s in frame_indices if s <= t], dtype=int)
        line_gd.set_data(w_gd[keep, 0], w_gd[keep, 1])
        line_sgd.set_data(w_sgd[keep, 0], w_sgd[keep, 1])
        line_tail.set_data(w_tail[keep, 0], w_tail[keep, 1])
        head_gd.set_data([w_gd[t, 0]], [w_gd[t, 1]])
        head_sgd.set_data([w_sgd[t, 0]], [w_sgd[t, 1]])
        head_tail.set_data([w_tail[t, 0]], [w_tail[t, 1]])
        return line_gd, line_sgd, line_tail, head_gd, head_sgd, head_tail

    anim = FuncAnimation(fig, update, init_func=init, frames=frame_indices, interval=400, blit=True)
    out = FIGURES_DIR / "gd_vs_sgd.gif"
    anim.save(out, writer=PillowWriter(fps=2.5))
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
