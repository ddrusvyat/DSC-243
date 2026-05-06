"""
Illustrate Theorem 8.1 (tail-averaged constant-stepsize SGD for least squares).

Well-specified Gaussian model:
    x ~ N(0, I_d),   y = <w_*, x> + eta,   eta ~ N(0, sigma^2),
so H = I, mu = 1, R^2 = d + 2, Sigma = sigma^2 H, sigma_MLE^2 = d sigma^2 / 2
and rho_misspec = 1. The population excess risk is the exact quadratic
    L(w) - L(w_*) = (1/2) ||w - w_*||^2,
which we compute exactly from each iterate (no Monte-Carlo noise in the loss).

We run constant-stepsize SGD from w_0 = 0 and compare at each horizon t:
  (a) Single iterate      L(w_t)            - L(w_*)
  (b) Tail average        L(bar w_{t/2:t})  - L(w_*)
against
  (c) Cramer-Rao rate     sigma_MLE^2 / t = d sigma^2 / (2 t)
  (d) Theorem 8.2 bound   (with burn-in s = t/2).

The single iterate decays exponentially at first and then plateaus at a noise
floor proportional to gamma * d sigma^2. The tail average, by contrast, keeps
decaying at the statistically optimal 1/t rate and matches Cramer-Rao up to a
constant.
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
):
    """
    Run a single trial of constant-stepsize SGD on the well-specified
    isotropic Gaussian model and return arrays of length T_max + 1 with:
      * gaps_single[t]  = L(w_t)            - L(w_*)
      * gaps_tail[t]    = L(bar w_{t/2:t})  - L(w_*)     (gaps_tail[0] = gaps_tail[1] = nan)
    """
    w = np.zeros(d)
    gaps_single = np.empty(T_max + 1)
    gaps_tail = np.full(T_max + 1, np.nan)

    # Prefix sums S[t] = sum_{s=0}^{t-1} w_s in R^d, so that
    # bar w_{a:b} = (S[b] - S[a]) / (b - a).
    S = np.zeros((T_max + 1, d))

    gaps_single[0] = 0.5 * float((w - w_star) @ (w - w_star))
    for t in range(T_max):
        S[t + 1] = S[t] + w

        x = rng.standard_normal(d)
        eta = sigma * rng.standard_normal()
        y = float(w_star @ x) + eta
        residual = y - float(w @ x)
        w = w + gamma * residual * x

        gaps_single[t + 1] = 0.5 * float((w - w_star) @ (w - w_star))

    # Tail averages over [t/2, t). Need t >= 2 so that t - t//2 >= 1.
    for t in range(2, T_max + 1):
        a = t // 2
        bar_w = (S[t] - S[a]) / (t - a)
        gaps_tail[t] = 0.5 * float((bar_w - w_star) @ (bar_w - w_star))

    return gaps_single, gaps_tail


def theorem_bound(
    t_grid: np.ndarray,
    gamma: float,
    mu: float,
    R2: float,
    sigma_MLE_sq: float,
    rho_misspec: float,
    w0_minus_wstar_sq: float,
):
    """
    Evaluate the right-hand side of (32) with burn-in s = t/2 at each t in t_grid.
    """
    bias = np.exp(-gamma * mu * t_grid / 2.0) * R2 * w0_minus_wstar_sq
    var_mult = 1.0 + (gamma * R2) / (1.0 - gamma * R2) * rho_misspec
    variance = 2.0 * var_mult * sigma_MLE_sq / (t_grid - t_grid // 2)
    return bias + variance


def main():
    d = 20
    sigma = 0.3
    T_max = 5_000
    n_trials = 60
    rng = np.random.default_rng(7)

    # Model constants.
    mu = 1.0
    R2 = d + 2.0
    Sigma_trace_times_Hinv = d * sigma ** 2          # Tr(H^{-1} Sigma)
    sigma_MLE_sq = 0.5 * Sigma_trace_times_Hinv       # = d sigma^2 / 2
    rho_misspec = 1.0

    # Pick the stepsize so that gamma R^2 = 1/2; then the variance multiplier is 2.
    gamma_rate = 0.5
    gamma = gamma_rate / R2

    # True parameter w_*: fixed unit vector (so ||w_0 - w_*||^2 = 1).
    w_star = rng.standard_normal(d)
    w_star /= np.linalg.norm(w_star)
    w0_diff_sq = 1.0

    single_runs = np.zeros((n_trials, T_max + 1))
    tail_runs = np.full((n_trials, T_max + 1), np.nan)
    for trial in range(n_trials):
        gs, gt = run_sgd_trial(d, sigma, gamma, w_star, T_max, rng)
        single_runs[trial] = gs
        tail_runs[trial] = gt
        if (trial + 1) % 10 == 0:
            print(f"  trial {trial + 1}/{n_trials} done")

    # Median and 10-90 interquantile bands.
    si_med = np.median(single_runs, axis=0)
    si_lo, si_hi = np.quantile(single_runs, [0.1, 0.9], axis=0)
    ta_med = np.nanmedian(tail_runs, axis=0)
    ta_lo = np.nanquantile(tail_runs, 0.1, axis=0)
    ta_hi = np.nanquantile(tail_runs, 0.9, axis=0)

    # --- Figure -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    iters = np.arange(T_max + 1)
    lo = 2  # skip t = 0, 1 for log axes
    mask = iters >= lo

    color_single = "#d7191c"
    color_tail = "#2c7bb6"

    ax.fill_between(
        iters[mask], si_lo[mask], si_hi[mask],
        color=color_single, alpha=0.15, linewidth=0,
    )
    ax.semilogy(
        iters[mask], si_med[mask], "-", color=color_single, linewidth=1.8,
        label=r"last iterate $L(w_t)-L(w_\ast)$",
    )

    ax.fill_between(
        iters[mask], ta_lo[mask], ta_hi[mask],
        color=color_tail, alpha=0.15, linewidth=0,
    )
    ax.semilogy(
        iters[mask], ta_med[mask], "-", color=color_tail, linewidth=1.8,
        label=r"tail average $L(\overline{w}_{t/2:t})-L(w_\ast)$",
    )

    # Reference line over a smooth grid.
    t_ref = np.linspace(lo, T_max, 400)
    ref_thm = theorem_bound(
        t_ref, gamma, mu, R2, sigma_MLE_sq, rho_misspec, w0_diff_sq,
    )

    ax.semilogy(
        t_ref, ref_thm, "--", color="0.35", linewidth=1.3,
        label=r"Theorem 8.2 bound (burn-in $=t/2$)",
    )

    ax.set_xlabel(r"iteration / sample size $t$", fontsize=12)
    ax.set_ylabel(r"excess risk", fontsize=12)
    ax.set_title(
        rf"Tail-averaged constant-stepsize SGD  ($d={d}$, $\sigma={sigma}$, "
        rf"$\gamma R^2={gamma_rate}$, median over {n_trials} trials)",
        fontsize=11.5,
    )
    ax.grid(True, alpha=0.25, which="both")
    ax.set_xlim(lo, T_max)
    ax.legend(fontsize=10, loc="lower left")

    fig.tight_layout()
    out = FIGURES_DIR / "sgd_tail_averaging.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
