"""
Illustrate Theorem 8.1 (last-iterate constant-stepsize SGD for least squares).

Well-specified Gaussian model:
    x ~ N(0, I_d),   y = <w_*, x> + eta,   eta ~ N(0, sigma^2),
so H = I, mu = 1, R^2 = d + 2, Sigma = sigma^2 H, Tr(Sigma) = d sigma^2.
The population excess risk is the exact quadratic
    L(w) - L(w_*) = (1/2) ||w - w_*||^2,
which we compute exactly from each iterate (no Monte-Carlo noise in the loss).

For several stepsizes gamma (chosen via gamma * R^2 in a range below 1) we run
constant-stepsize SGD from w_0 = 0 and plot the median over n_trials of the
single-iterate risk L(w_t) - L(w_*), together with a 10--90% interquantile band.
Overlaid is the horizontal noise floor gamma * Tr(Sigma) / (2 * (2 - gamma R^2))
predicted by Theorem 8.1.

The curves exhibit the two phases predicted by (27): an initial exponential
contraction at rate e^{-gamma mu t} followed by stationary oscillation around
a stepsize-dependent noise floor. Smaller gamma lowers the floor at the cost of
slower contraction; larger gamma contracts faster to a higher floor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def run_sgd_trial(
    d: int,
    sigma: float,
    gamma: float,
    w_star: np.ndarray,
    T_max: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Run a single trial of constant-stepsize SGD on the well-specified
    isotropic Gaussian model. Return the array of length T_max + 1 with
        gaps[t] = L(w_t) - L(w_*) = (1/2) ||w_t - w_*||^2.
    """
    d = int(d)
    w = np.zeros(d)
    gaps = np.empty(T_max + 1)
    gaps[0] = 0.5 * float((w - w_star) @ (w - w_star))
    for t in range(T_max):
        x = rng.standard_normal(d)
        eta = sigma * rng.standard_normal()
        y = float(w_star @ x) + eta
        residual = y - float(w @ x)
        w = w + gamma * residual * x
        gaps[t + 1] = 0.5 * float((w - w_star) @ (w - w_star))
    return gaps


def noise_floor(gamma: float, R2: float, tr_Sigma: float) -> float:
    """Theorem 8.1 noise floor gamma Tr(Sigma) / (2 (2 - gamma R^2))."""
    return gamma * tr_Sigma / (2.0 * (2.0 - gamma * R2))


def main() -> None:
    d = 20
    sigma = 0.3
    T_max = 2_000
    n_trials = 80

    mu = 1.0
    R2 = d + 2.0
    tr_Sigma = d * sigma ** 2

    # Three stepsizes spanning the stable range gamma R^2 in (0, 1).
    gamma_rates = [0.2, 0.5, 0.8]
    colors = ["#1b9e77", "#d95f02", "#7570b3"]

    rng_master = np.random.default_rng(11)

    # Fix the true parameter so ||w_0 - w_*||^2 = 1 across all gammas.
    w_star_base = rng_master.standard_normal(d)
    w_star = w_star_base / np.linalg.norm(w_star_base)
    w0_diff_sq = 1.0

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    iters = np.arange(T_max + 1)

    for gamma_rate, color in zip(gamma_rates, colors):
        gamma = gamma_rate / R2

        runs = np.empty((n_trials, T_max + 1))
        for trial in range(n_trials):
            rng = np.random.default_rng(rng_master.integers(0, 2 ** 31))
            runs[trial] = run_sgd_trial(d, sigma, gamma, w_star, T_max, rng)

        med = np.median(runs, axis=0)
        lo, hi = np.quantile(runs, [0.1, 0.9], axis=0)
        floor = noise_floor(gamma, R2, tr_Sigma)

        ax.fill_between(
            iters, lo, hi, color=color, alpha=0.18, linewidth=0,
        )
        ax.plot(
            iters, med, "-", color=color, linewidth=1.6,
            label=rf"SGD median, $\gamma R^2 = {gamma_rate}$",
        )
        ax.axhline(
            floor, color=color, linestyle=":", linewidth=1.1, alpha=0.85,
        )

        print(
            f"  gamma R^2 = {gamma_rate:.2f}: gamma = {gamma:.4g}, "
            f"floor = {floor:.3e}, empirical median at T = {med[-1]:.3e}"
        )

    ax.plot([], [], ":", color="0.25", linewidth=1.1, label="noise floor $\\gamma\\,\\mathrm{Tr}(\\Sigma)/(2(2-\\gamma R^2))$")

    ax.set_xlabel(r"iteration $t$", fontsize=12)
    ax.set_ylabel(r"excess risk $L(w_t) - L(w_\ast)$", fontsize=12)
    ax.set_yscale("log")
    ax.set_xlim(0, T_max)
    ax.grid(True, alpha=0.25, which="both")
    ax.set_title(
        rf"Last-iterate constant-stepsize SGD  ($d={d}$, $\sigma={sigma}$, "
        rf"median and 10--90% band over {n_trials} trials)",
        fontsize=11.5,
    )
    ax.legend(fontsize=9.5, loc="upper right", ncol=2, framealpha=0.95)

    fig.tight_layout()
    out = FIGURES_DIR / "sgd_last_iterate.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
