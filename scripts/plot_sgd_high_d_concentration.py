"""
Companion to plot_sgd_high_d_ode_limit.py.

Here we *fix* the stepsize gamma and *vary* the dimension d, to visualize the
concentration statement of Theorem 10.2: as d grows, the random excess-risk
trajectory L(w_{[td]}) - L_* concentrates around the (dimension-independent)
ODE limit psi(t), and the trial-to-trial fluctuation band shrinks.

Setup (same isotropic Gaussian model as the companion script):
    x ~ N(0, I_d),   eta ~ N(0, sigma^2),   y = <w_*, x> + eta
    L(w) - L(w_*) = (1/2) ||w - w_*||^2

Streaming SGD with stepsize gamma/d:
    w_{k+1} = w_k - (gamma/d) * (<w_k, x_{k+1}> - y_{k+1}) * x_{k+1}.

The ODE limit psi(t) does not depend on d, so the same dashed curve is drawn in
every panel; only the SGD band width changes (shrinking like 1/sqrt(d)).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def run_streaming_sgd_batch(
    d: int,
    sigma: float,
    gamma: float,
    w_star: np.ndarray,
    w0: np.ndarray,
    n_epochs: float,
    n_trials: int,
    rng: np.random.Generator,
):
    """Run n_trials independent streaming-SGD trajectories simultaneously
    (vectorized over trials). Returns epochs t_k = k/d and an array of excess
    risks R[trial, k] = (1/2) ||w_k - w_*||^2."""
    n_steps = int(round(n_epochs * d))
    W = np.tile(w0, (n_trials, 1)).astype(float)  # (n_trials, d)
    R = np.empty((n_trials, n_steps + 1))
    diff = W - w_star
    R[:, 0] = 0.5 * np.sum(diff * diff, axis=1)
    for k in range(n_steps):
        X = rng.standard_normal((n_trials, d))
        eta = sigma * rng.standard_normal(n_trials)
        y = X @ w_star + eta  # (n_trials,)
        pred = np.sum(W * X, axis=1)  # (n_trials,)
        residual = pred - y
        W -= (gamma / d) * residual[:, None] * X
        diff = W - w_star
        R[:, k + 1] = 0.5 * np.sum(diff * diff, axis=1)
    epochs = np.arange(n_steps + 1) / d
    return epochs, R


def ode_solution(t: np.ndarray, gamma: float, sigma: float, R0: float):
    """Closed-form solution of dot psi = (gamma^2 - 2 gamma) psi + gamma^2 sigma^2/2."""
    a = gamma ** 2 - 2 * gamma
    if abs(a) < 1e-12:
        return R0 + 0.5 * gamma ** 2 * sigma ** 2 * t
    psi_inf = -0.5 * gamma ** 2 * sigma ** 2 / a
    return psi_inf + (R0 - psi_inf) * np.exp(a * t)


def main():
    sigma = 0.1
    n_epochs = 8.0
    gamma = 1.0
    dims = [50, 200, 800]
    n_trials = 200

    rng_master = np.random.default_rng(1)
    sgd_color = plt.cm.viridis(0.35)

    fig, axes = plt.subplots(
        1, len(dims), figsize=(13.5, 4.6), sharey=True
    )

    psi_inf = 0.5 * gamma * sigma ** 2 / (2 - gamma)
    t_grid = np.linspace(0, n_epochs, 600)
    psi = ode_solution(t_grid, gamma, sigma, 0.5)

    for ax, d in zip(axes, dims):
        w_star = np.zeros(d)
        w_star[0] = 1.0
        w0 = np.zeros(d)

        rng = np.random.default_rng(rng_master.integers(0, 2 ** 31))
        epochs, R = run_streaming_sgd_batch(
            d, sigma, gamma, w_star, w0, n_epochs, n_trials, rng
        )

        med = np.median(R, axis=0)
        lo, hi = np.quantile(R, [0.1, 0.9], axis=0)

        ax.fill_between(
            epochs, lo, hi, color=sgd_color, alpha=0.30, linewidth=0,
            label="streaming SGD\n10-90% band",
        )
        ax.plot(epochs, med, "-", color=sgd_color, linewidth=1.6,
                label="SGD median")
        ax.plot(t_grid, psi, "--", color="black", linewidth=1.6,
                label=r"ODE limit $\psi(t)$")
        ax.axhline(psi_inf, color="black", linewidth=0.9, linestyle=":",
                   alpha=0.7, label=r"$\psi_\infty$")

        ax.set_yscale("log")
        ax.set_xlim(0, n_epochs)
        ax.set_xlabel(r"epoch $t = k/d$", fontsize=12)
        ax.set_title(rf"$d = {d}$", fontsize=13)
        ax.grid(True, which="both", alpha=0.25)

        band_width = float(np.max((hi - lo)[epochs >= 2.0]))
        print(
            f"  gamma={gamma}, d={d}: {n_trials} trials, "
            f"R_final median = {med[-1]:.3e}, "
            f"max band width (t>=2) = {band_width:.3e}"
        )

    axes[0].set_ylabel(r"excess risk $L(w_{[td]}) - L(w_\ast)$", fontsize=12)
    axes[-1].legend(fontsize=9.5, loc="upper right", framealpha=0.95)
    fig.suptitle(
        rf"Concentration of streaming SGD as $d$ grows "
        rf"(fixed $\gamma={gamma}$, $\sigma={sigma}$, $w_0=0$; "
        rf"median and 10-90% band over {n_trials} trials)",
        fontsize=12.5,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIGURES_DIR / "sgd_high_d_concentration.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
