"""
Animate the Bayesian decoder underlying the lower bound of Theorem 9.4
(Minimax lower bound for streaming least squares).

As the prior precision lambda decreases, the prior N(0, sigma^2/(T lambda) I)
becomes uninformative and the posterior over w_* contracts onto the OLS
estimator. The optimal Bayes point estimator -- the posterior mean, which
coincides with the ridge regression solution at parameter lambda -- traces
out the ridge regularization path from the origin (heavy regularization,
lambda large) to the OLS minimizer (no regularization, lambda -> 0). This is
exactly the limit lambda -> 0 taken in the proof.

Setup: d = 2, T = 10 design points with Phi^T Phi / T = H non-isotropic, a
single fixed noise realization. At each lambda, draw:
- prior 1- and 2-sigma covariance contours (faint blue),
- posterior 1- and 2-sigma covariance contours (orange),
- the true parameter w_* (gold star),
- the OLS estimator (green diamond),
- the ridge / posterior-mean estimator at this lambda (red dot),
  with a trail showing the regularization path swept so far.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MPL_CACHE = ROOT / ".matplotlib_cache"
MPL_CACHE.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_CACHE)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse


FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def cov_ellipse(ax, mean, cov, n_std, **kwargs):
    """Draw the n_std covariance ellipse of a 2D Gaussian on `ax`."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
    width = 2.0 * n_std * float(np.sqrt(eigvals[0]))
    height = 2.0 * n_std * float(np.sqrt(eigvals[1]))
    e = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(e)
    return e


def main() -> None:
    rng = np.random.default_rng(11)
    d = 2
    T = 10
    sigma = 0.40
    w_star = np.array([1.0, 0.7])

    # Non-isotropic feature covariance H, axes rotated 25 degrees.
    theta = np.deg2rad(25.0)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    H = R @ np.diag([2.2, 0.45]) @ R.T

    # Design with Phi^T Phi / T = H (whiten random Gaussian to match H exactly).
    Phi = rng.standard_normal((T, d))
    S = Phi.T @ Phi / T
    eigS, vecS = np.linalg.eigh(S)
    S_inv_sqrt = vecS @ np.diag(1.0 / np.sqrt(eigS)) @ vecS.T
    eigH, vecH = np.linalg.eigh(H)
    H_sqrt = vecH @ np.diag(np.sqrt(eigH)) @ vecH.T
    Phi = Phi @ S_inv_sqrt @ H_sqrt
    PhiTPhi = Phi.T @ Phi

    eta = sigma * rng.standard_normal(T)
    y = Phi @ w_star + eta

    Phity = Phi.T @ y
    w_ols = np.linalg.solve(PhiTPhi, Phity)
    cov_ols = sigma ** 2 * np.linalg.inv(PhiTPhi)

    # Sweep lambda log-uniformly from large (heavy regularization) to small.
    n_main = 56
    lambdas_main = np.logspace(1.0, -3.0, n_main)
    n_pause = 6
    lambdas = np.concatenate([lambdas_main, np.full(n_pause, lambdas_main[-1])])
    n_frames = lambdas.size

    ridges = np.empty((n_frames, d))
    posterior_covs = np.empty((n_frames, d, d))
    prior_covs = np.empty((n_frames, d, d))
    for i, lam in enumerate(lambdas):
        Tlam_I = T * lam * np.eye(d)
        K = PhiTPhi + Tlam_I
        ridges[i] = np.linalg.solve(K, Phity)
        posterior_covs[i] = sigma ** 2 * np.linalg.inv(K)
        prior_covs[i] = (sigma ** 2 / (T * lam)) * np.eye(d)

    # Plot window: include origin, w_*, OLS, with extra room for the OLS
    # uncertainty ellipse (which the posterior contracts to as lambda -> 0).
    pts = np.vstack([np.zeros((1, d)), w_star[None, :], w_ols[None, :], ridges])
    pad = 0.6
    xmin, xmax = float(pts[:, 0].min()) - pad, float(pts[:, 0].max()) + pad
    ymin, ymax = float(pts[:, 1].min()) - pad, float(pts[:, 1].max()) + pad
    span = max(xmax - xmin, ymax - ymin) * 1.05
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    xmin, xmax = cx - 0.5 * span, cx + 0.5 * span
    ymin, ymax = cy - 0.5 * span, cy + 0.5 * span

    fig, ax = plt.subplots(figsize=(7.4, 7.0), dpi=120)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$w^{(1)}$", fontsize=12)
    ax.set_ylabel(r"$w^{(2)}$", fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.set_title(
        r"Bayesian decoder for Theorem 9.4: as $\lambda \downarrow 0$ the posterior contracts onto OLS",
        fontsize=11,
    )

    ax.plot(0, 0, marker="+", color="#444444", markersize=10, mew=2, zorder=4)
    ax.annotate(
        "0 (prior mean)", (0, 0), xytext=(8, 4), textcoords="offset points",
        fontsize=10, color="#444444",
    )
    ax.plot(
        w_star[0], w_star[1], marker="*", markersize=18,
        color="black", markerfacecolor="gold", markeredgewidth=1.0, zorder=5,
    )
    ax.annotate(
        r"$w_\ast$ (truth)", w_star, xytext=(10, 8), textcoords="offset points",
        fontsize=12, color="black",
    )
    ax.plot(
        w_ols[0], w_ols[1], marker="D", markersize=10,
        color="#1a8a3a", markerfacecolor="#9fdfa8", markeredgewidth=1.2, zorder=5,
    )
    ax.annotate(
        "OLS", w_ols, xytext=(10, -4), textcoords="offset points",
        fontsize=12, color="#1a8a3a",
    )

    # Static OLS uncertainty contour (1-sigma) as a dashed reference for the
    # lambda -> 0 limit of the posterior.
    cov_ellipse(
        ax, mean=tuple(w_ols), cov=cov_ols, n_std=1.0,
        edgecolor="#1a8a3a", facecolor="none", linewidth=1.3, linestyle=(0, (4, 3)),
        alpha=0.7, zorder=3,
    )

    prior_artists: list = []
    posterior_artists: list = []

    (trail_line,) = ax.plot([], [], "-", color="#d7191c", linewidth=1.6, alpha=0.85, zorder=4)
    (ridge_pt,) = ax.plot([], [], "o", color="#d7191c", markersize=11,
                          markeredgecolor="#7a0d12", markeredgewidth=1.0, zorder=6)

    title_box = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="lightgray", alpha=0.95),
    )

    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="-", color="#d7191c",
                   markersize=8, label=r"ridge $A_\ast(y) = \mathbb{E}[w_\ast\mid y]$"),
        Ellipse((0, 0), 0, 0, edgecolor="#2c7bb6", facecolor="#abd9e9",
                label=r"prior $\mathcal{N}(0,\,\frac{\sigma^2}{T\lambda}I)$", alpha=0.5),
        Ellipse((0, 0), 0, 0, edgecolor="#d7191c", facecolor="#fdae61",
                label=r"posterior $\mathcal{N}(A_\ast(y), \sigma^2(\Phi^\top\Phi+T\lambda I)^{-1})$", alpha=0.6),
        plt.Line2D([0], [0], linestyle=(0, (4, 3)), color="#1a8a3a",
                   label=r"OLS uncertainty ($1\sigma$)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9.5,
              framealpha=0.95)

    def clear_artists(artist_list: list) -> None:
        for a in artist_list:
            try:
                a.remove()
            except Exception:
                pass
        artist_list.clear()

    def update(frame):
        clear_artists(prior_artists)
        clear_artists(posterior_artists)

        for n_std, alpha in [(1.0, 0.22), (2.0, 0.10)]:
            e = cov_ellipse(
                ax, mean=(0.0, 0.0), cov=prior_covs[frame], n_std=n_std,
                edgecolor="#2c7bb6", facecolor="#abd9e9",
                alpha=alpha, linewidth=1.0, zorder=1,
            )
            prior_artists.append(e)

        for n_std, alpha in [(1.0, 0.40), (2.0, 0.18)]:
            e = cov_ellipse(
                ax, mean=tuple(ridges[frame]), cov=posterior_covs[frame], n_std=n_std,
                edgecolor="#d7191c", facecolor="#fdae61",
                alpha=alpha, linewidth=1.0, zorder=2,
            )
            posterior_artists.append(e)

        trail_line.set_data(ridges[: frame + 1, 0], ridges[: frame + 1, 1])
        ridge_pt.set_data([ridges[frame, 0]], [ridges[frame, 1]])

        title_box.set_text(rf"$\lambda = {lambdas[frame]:.3g}$")

        return prior_artists + posterior_artists + [trail_line, ridge_pt, title_box]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=90, blit=False)

    out = FIGURES_DIR / "mourtada_posterior_shrinkage.gif"
    anim.save(out, writer=PillowWriter(fps=11))
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
