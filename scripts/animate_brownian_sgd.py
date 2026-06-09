"""
Animate two ideas side by side, to illustrate the moment-matching picture of
Section 10:

Left panel  : a pure planar Brownian motion B_t in R^2 -- the driving noise of
              an SDE, with no drift. The path is colored by time.

Right panel : one trajectory of the moment-matched diffusion approximation of
              SGD on a simple anisotropic least-squares quadratic in R^2 (the SDE
              dX = b(X) dt + Sigma(X)^{1/2} dB). The path drifts toward the
              minimizer w_* while fluctuating -- drift plus Brownian noise.

The SDE is the exact moment match of the SGD step:
    u = w - w_*,    L(w) = (1/2)(sigma^2 + u^T H u)
    drift      b(w)     = -eta * H u
    diffusion  Sigma(w) = eta^2 * (2 L(w) H + (H u)(H u)^T)
so that b(w) and Sigma(w) reproduce the per-step conditional mean and covariance
of streaming SGD with features x ~ N(0, H) and label noise sigma.

Output: figures/brownian_and_sgd_diffusion.gif
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter


def segments(points):
    """Turn an (n, 2) path into (n-1, 2, 2) segments for a LineCollection."""
    pts = points.reshape(-1, 1, 2)
    return np.concatenate([pts[:-1], pts[1:]], axis=1)


def psd_sqrt_2x2(S):
    """Symmetric PSD square root of a 2x2 matrix via eigendecomposition."""
    vals, vecs = np.linalg.eigh(S)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def simulate_brownian(rng, n_steps=900, dt=0.02):
    incr = rng.normal(size=(n_steps, 2)) * np.sqrt(dt)
    path = np.vstack([[0.0, 0.0], np.cumsum(incr, axis=0)])
    return path


def simulate_sde(rng, H, w0, wstar, sigma, eta, n_steps, n_sub=8):
    """Euler-Maruyama for the moment-matched SDE; 1 time unit = 1 SGD step."""
    h = 1.0 / n_sub
    X = w0.astype(float).copy()
    traj = [X.copy()]
    for _ in range(n_steps):
        for _ in range(n_sub):
            u = X - wstar
            L = 0.5 * (sigma**2 + u @ H @ u)
            Hu = H @ u
            Sigma = eta**2 * (2.0 * L * H + np.outer(Hu, Hu))
            b = -eta * Hu
            Sig_half = psd_sqrt_2x2(Sigma)
            X = X + b * h + Sig_half @ (np.sqrt(h) * rng.normal(size=2))
        traj.append(X.copy())
    return np.array(traj)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "brownian_and_sgd_diffusion.gif")

    rng = np.random.default_rng(2)

    # --- left: pure Brownian motion ---
    n_bm = 900
    B = simulate_brownian(rng, n_steps=n_bm, dt=0.02)

    # --- right: SGD on an anisotropic quadratic + its diffusion approximation ---
    H = np.diag([2.0, 0.6])
    wstar = np.array([0.0, 0.0])
    w0 = np.array([2.6, 2.1])
    sigma = 0.45
    eta = 0.16
    n_steps = 380

    sde = simulate_sde(rng, H, w0, wstar, sigma, eta, n_steps, n_sub=8)

    # ---- figure ----
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5.2))
    fig.suptitle(
        "An SDE is drift + Brownian noise: pure Brownian motion (left) "
        "and the diffusion approximation of SGD (right)",
        fontsize=12,
    )

    # left axis limits
    padB = 0.1 * (B.max() - B.min())
    axL.set_xlim(B[:, 0].min() - padB, B[:, 0].max() + padB)
    axL.set_ylim(B[:, 1].min() - padB, B[:, 1].max() + padB)
    axL.set_aspect("equal")
    axL.set_title(r"Brownian motion $B_t$ in $\mathbb{R}^2$", fontsize=11)
    axL.set_xticks([])
    axL.set_yticks([])

    lc_bm = LineCollection([], cmap="viridis", linewidths=1.8)
    axL.add_collection(lc_bm)
    (dot_bm,) = axL.plot([], [], "o", color="#440154", ms=7, zorder=5)
    axL.plot(B[0, 0], B[0, 1], "o", color="black", ms=5, zorder=4)

    # right axis: contours of L
    xmin, ymin = sde.min(axis=0) - 0.6
    xmax, ymax = sde.max(axis=0) + 0.6
    gx = np.linspace(xmin, xmax, 220)
    gy = np.linspace(ymin, ymax, 220)
    GX, GY = np.meshgrid(gx, gy)
    U0 = GX - wstar[0]
    U1 = GY - wstar[1]
    Lgrid = 0.5 * (H[0, 0] * U0**2 + H[1, 1] * U1**2)
    axR.contour(GX, GY, Lgrid, levels=12, colors="0.8", linewidths=0.8, zorder=0)
    axR.set_xlim(xmin, xmax)
    axR.set_ylim(ymin, ymax)
    axR.set_aspect("equal")
    axR.set_title("Diffusion approximation of SGD", fontsize=11)
    axR.set_xticks([])
    axR.set_yticks([])
    axR.plot(wstar[0], wstar[1], "*", color="black", ms=15, zorder=6,
             label=r"minimizer $w_\ast$")

    lc_sde = LineCollection([], cmap="autumn", linewidths=2.0, zorder=4)
    axR.add_collection(lc_sde)
    (sde_dot,) = axR.plot([], [], "o", color="#d62728", ms=7, zorder=6)
    # legend proxy for the SDE line
    axR.plot([], [], "-", color="#d62728", lw=2.0, label="diffusion SDE $X_t$")
    axR.plot(w0[0], w0[1], "o", color="black", ms=5, zorder=5, label="start $w_0$")
    axR.legend(loc="upper right", fontsize=8, framealpha=0.9)

    n_frames = 200

    def update(f):
        frac = (f + 1) / n_frames

        # Brownian motion
        m = max(1, int(round(frac * n_bm)))
        segs = segments(B[: m + 1])
        lc_bm.set_segments(segs)
        lc_bm.set_array(np.linspace(0, 1, len(segs)))
        dot_bm.set_data([B[m, 0]], [B[m, 1]])

        # SDE
        k = max(1, int(round(frac * n_steps)))
        segs2 = segments(sde[: k + 1])
        lc_sde.set_segments(segs2)
        lc_sde.set_array(np.linspace(0, 1, len(segs2)))
        sde_dot.set_data([sde[k, 0]], [sde[k, 1]])

        return lc_bm, dot_bm, lc_sde, sde_dot

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    writer = PillowWriter(fps=25)
    anim.save(out_path, writer=writer, dpi=90)
    print(f"wrote {os.path.normpath(out_path)}")


if __name__ == "__main__":
    main()
