"""
Compare the excess-risk curve of streaming SGD with that of its diffusion
approximation (the full moment-matched SDE of Section 10, including the rank-one
correction) on a correlated-feature least-squares model whose covariance H has a
*random eigenbasis*, in high dimension and for varying stepsizes.

Model:
    H = Q diag(lambda_1, ..., lambda_d) Q^T,  lambda_i = i/d on (0, 1],
        Q a Haar-random orthogonal matrix (so H is NOT diagonal -- the features
        are correlated across all coordinates, not axis-aligned).
    x ~ N(0, H),   y = <w_*, x> + sigma * eta,  eta ~ N(0, 1)
    L(w) - L(w_*) = (1/2) (w - w_*)^T H (w - w_*)
    w_* with w_{*,i} = 1/sqrt(d) (so ||w_*|| = 1), w_0 = 0.

Why we can simulate cheaply despite the rotation. The model is orthogonally
invariant: under the change of variables v = Q^T w, the rotated-H problem with
signal w_* becomes EXACTLY the diagonal problem with covariance diag(lambda) and
signal s = Q^T w_*. Streaming SGD, the loss, and the SDE all transform
covariantly, so the law of the excess-risk trajectory is identical. We therefore
generate Q, rotate the signal once (s = Q^T w_*, a generic unit vector since Q is
Haar-random and ||w_*|| = 1), and run the fast diagonal dynamics in the
eigenbasis -- this is the rotated-H experiment, computed in the smart basis, not
an approximation. (A literal dense-H simulation gives the same curves but costs a
d-by-d matmul per step.)

Diagonal dynamics in the eigenbasis (state v, signal s, v_0 = Q^T w_0 = 0):
    streaming SGD, stepsize gamma/d:
        x = sqrt(lambda) * z,  z ~ N(0, I_d)
        v <- v - (gamma/d) (<v, x> - y) x
    diffusion SDE (full, including rank-one), Euler-Maruyama, step h = 1/d,
    u = v - s, increment covariance (gamma^2/d^2)(2 L H + H u u^T H):
        v <- v - (gamma/d) (lambda*u)
               + (gamma/d) [ sqrt(2 L) sqrt(lambda)*z + (lambda*u) * g ],
        z ~ N(0, I_d),  g ~ N(0, 1) scalar,
    since lambda*u = H u so (lambda*u)(lambda*u)^T = H u u^T H is the rank-one part.

Left panel  : fix d, vary gamma (one shared random rotation). The SDE reproduces
              the SGD curve across stepsizes.
Right panel : fix gamma, vary d (a fresh random rotation per dimension). SGD and
              the SDE lock together more tightly and the 10-90% band narrows like
              1/sqrt(d).

The Brownian motion driving the SDE is independent of the SGD samples, so the
match is distributional (median/band), not pathwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def ramp_spectrum(d: int) -> np.ndarray:
    """lambda_i = i/d for i = 1, ..., d (eigenvalues of H)."""
    return np.arange(1, d + 1, dtype=float) / d


def rotated_signal(d: int, rng: np.random.Generator) -> np.ndarray:
    """Signal in H's eigenbasis, s = Q^T w_* with Q Haar-random and
    w_{*,i} = 1/sqrt(d). For Haar Q this is uniform on the unit sphere; we draw
    it directly as a normalized Gaussian (||s|| = ||w_*|| = 1)."""
    g = rng.standard_normal(d)
    return g / np.linalg.norm(g)


def run_sgd_batch(d, lam, sigma, gamma, s, v0, n_epochs, n_trials, rng):
    """n_trials streaming-SGD trajectories in the eigenbasis (vectorized over
    trials). Returns epochs t_k = k/d and excess risks R[trial, k]."""
    n_steps = int(round(n_epochs * d))
    sqrt_lam = np.sqrt(lam)
    V = np.tile(v0, (n_trials, 1)).astype(float)
    R = np.empty((n_trials, n_steps + 1))
    diff = V - s
    R[:, 0] = 0.5 * ((diff * diff) @ lam)
    for k in range(n_steps):
        Z = rng.standard_normal((n_trials, d))
        X = Z * sqrt_lam
        eta = sigma * rng.standard_normal(n_trials)
        y = X @ s + eta
        residual = np.sum(V * X, axis=1) - y
        V -= (gamma / d) * residual[:, None] * X
        diff = V - s
        R[:, k + 1] = 0.5 * ((diff * diff) @ lam)
    epochs = np.arange(n_steps + 1) / d
    return epochs, R


def run_sde_batch(d, lam, sigma, gamma, s, v0, n_epochs, n_trials, rng):
    """n_trials trajectories of the full diffusion SDE (drift plus the bulk
    diffusion 2 L H AND the rank-one correction H u u^T H) via Euler-Maruyama
    with step h = 1/d. Returns epochs and excess risks R[trial, k]."""
    n_steps = int(round(n_epochs * d))
    sqrt_lam = np.sqrt(lam)
    V = np.tile(v0, (n_trials, 1)).astype(float)
    R = np.empty((n_trials, n_steps + 1))
    diff = V - s
    R[:, 0] = 0.5 * ((diff * diff) @ lam)
    for k in range(n_steps):
        diff = V - s
        two_L = np.maximum(sigma ** 2 + (diff * diff) @ lam, 0.0)
        Z = rng.standard_normal((n_trials, d))
        g = rng.standard_normal((n_trials, 1))     # rank-one scalar
        Hu = lam * diff                            # H u
        drift = -(gamma / d) * Hu
        bulk = (gamma / d) * np.sqrt(two_L)[:, None] * (sqrt_lam * Z)
        rank_one = (gamma / d) * Hu * g            # along direction H u
        V = V + drift + bulk + rank_one
        diff = V - s
        R[:, k + 1] = 0.5 * ((diff * diff) @ lam)
    epochs = np.arange(n_steps + 1) / d
    return epochs, R


def panel_vary_gamma(ax, sigma, n_epochs, d, gammas, n_trials, rng_master):
    colors = plt.cm.viridis(np.linspace(0.15, 0.80, len(gammas)))
    lam = ramp_spectrum(d)
    s = rotated_signal(d, np.random.default_rng(rng_master.integers(0, 2 ** 31)))
    v0 = np.zeros(d)
    for gamma, color in zip(gammas, colors):
        rng = np.random.default_rng(rng_master.integers(0, 2 ** 31))
        epochs, R_sgd = run_sgd_batch(d, lam, sigma, gamma, s, v0,
                                      n_epochs, n_trials, rng)
        rng = np.random.default_rng(rng_master.integers(0, 2 ** 31))
        _, R_sde = run_sde_batch(d, lam, sigma, gamma, s, v0,
                                 n_epochs, n_trials, rng)
        med_sgd = np.median(R_sgd, axis=0)
        lo, hi = np.quantile(R_sgd, [0.1, 0.9], axis=0)
        med_sde = np.median(R_sde, axis=0)
        ax.fill_between(epochs, lo, hi, color=color, alpha=0.25, linewidth=0)
        ax.plot(epochs, med_sgd, "-", color=color, linewidth=1.6,
                label=rf"$\gamma={gamma}$")
        ax.plot(epochs, med_sde, "--", color="black", linewidth=1.3, alpha=0.9)
        gap = float(np.max(np.abs(med_sgd - med_sde)))
        print(f"  gamma={gamma}: max |median SGD - median SDE| = {gap:.3e}")
    ax.set_yscale("log")
    ax.set_xlim(0, n_epochs)
    ax.set_xlabel(r"epoch $t = k/d$", fontsize=12)
    ax.set_ylabel(r"excess risk $L(w_{[td]}) - L(w_\ast)$", fontsize=12)
    ax.set_title(rf"Vary $\gamma$ at fixed $d={d}$ "
                 rf"(solid: SGD, dashed: diffusion SDE)", fontsize=11.5)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9.5, loc="upper right", framealpha=0.95)


def panel_vary_d(ax, sigma, n_epochs, gamma, dims, n_trials, rng_master):
    colors = plt.cm.viridis(np.linspace(0.15, 0.80, len(dims)))
    for d, color in zip(dims, colors):
        lam = ramp_spectrum(d)
        s = rotated_signal(d, np.random.default_rng(rng_master.integers(0, 2 ** 31)))
        v0 = np.zeros(d)
        rng = np.random.default_rng(rng_master.integers(0, 2 ** 31))
        epochs, R_sgd = run_sgd_batch(d, lam, sigma, gamma, s, v0,
                                      n_epochs, n_trials, rng)
        rng = np.random.default_rng(rng_master.integers(0, 2 ** 31))
        _, R_sde = run_sde_batch(d, lam, sigma, gamma, s, v0,
                                 n_epochs, n_trials, rng)
        med_sgd = np.median(R_sgd, axis=0)
        lo, hi = np.quantile(R_sgd, [0.1, 0.9], axis=0)
        med_sde = np.median(R_sde, axis=0)
        ax.fill_between(epochs, lo, hi, color=color, alpha=0.22, linewidth=0)
        ax.plot(epochs, med_sgd, "-", color=color, linewidth=1.6,
                label=rf"$d={d}$")
        ax.plot(epochs, med_sde, "--", color="black", linewidth=1.2, alpha=0.85)
        gap = float(np.max(np.abs(med_sgd - med_sde)))
        band = float(np.max((hi - lo)[epochs >= 2.0]))
        print(f"  d={d}: max |median gap| = {gap:.3e}, max band (t>=2) = {band:.3e}")
    ax.set_yscale("log")
    ax.set_xlim(0, n_epochs)
    ax.set_xlabel(r"epoch $t = k/d$", fontsize=12)
    ax.set_ylabel(r"excess risk $L(w_{[td]}) - L(w_\ast)$", fontsize=12)
    ax.set_title(rf"Fix $\gamma={gamma}$, vary $d$ "
                 rf"(solid: SGD, dashed: SDE)", fontsize=11.5)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9.5, loc="upper right", framealpha=0.95)


def main():
    sigma = 0.1
    n_epochs = 8.0

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.0, 5.4))

    print("left panel (vary gamma, d=512):")
    panel_vary_gamma(ax_left, sigma, n_epochs, d=512, gammas=[0.5, 1.0, 1.5],
                     n_trials=40, rng_master=np.random.default_rng(0))
    print("right panel (fix gamma=1, vary d):")
    panel_vary_d(ax_right, sigma, n_epochs, gamma=1.0, dims=[64, 256, 1024],
                 n_trials=40, rng_master=np.random.default_rng(1))

    fig.suptitle(
        rf"Streaming SGD vs. its diffusion approximation, $H=Q\Lambda Q^\top$ with "
        rf"$\lambda_i=i/d$ and Haar-random $Q$ ($\sigma={sigma}$, $w_0=0$): the SDE tracks the SGD risk curve",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIGURES_DIR / "sgd_vs_sde_highd.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
