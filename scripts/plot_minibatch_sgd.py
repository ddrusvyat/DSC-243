"""
Illustrate mini-batch tail-averaged SGD and batch saturation.

Model:
    x ~ N(0, I_d),   y = <w_*, x> + eta,   eta ~ N(0, sigma^2).

The experiment compares two resource models:
  1. fixed number of parameter updates T, where increasing B reduces the
     gradient noise until the optimization bias dominates;
  2. fixed total sample budget N = BT, where the variance term depends mainly
     on the total number of samples and large batches eventually leave too few
     updates to remove the bias.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def tail_gd_bias(gamma: float, t0: int, t1: int, w0_diff_sq: float = 1.0) -> float:
    """Population-GD contribution to L(bar w_{t0:t1}) - L(w_*), for H=I."""
    if t1 <= t0:
        return np.nan
    powers = (1.0 - gamma) ** np.arange(t0, t1)
    coeff = float(np.mean(powers))
    return 0.5 * coeff * coeff * w0_diff_sq


def run_minibatch_trial(
    d: int,
    sigma: float,
    gamma: float,
    batch_size: int,
    n_updates: int,
    w_star: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """Return L(bar w_{T/2:T}) - L(w_*) for one mini-batch SGD run."""
    w = np.zeros(d)
    tail_start = n_updates // 2
    tail_sum = np.zeros(d)
    tail_count = 0

    for t in range(n_updates):
        if t >= tail_start:
            tail_sum += w
            tail_count += 1

        X = rng.standard_normal((batch_size, d))
        eta = sigma * rng.standard_normal(batch_size)
        y = X @ w_star + eta
        residual = y - X @ w
        w = w + gamma * (X.T @ residual) / batch_size

    bar_w = tail_sum / tail_count
    err = bar_w - w_star
    return 0.5 * float(err @ err)


def median_risks(
    *,
    d: int,
    sigma: float,
    gamma: float,
    batch_sizes: np.ndarray,
    n_updates_for_batch,
    w_star: np.ndarray,
    n_trials: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    med = []
    lo = []
    hi = []
    for batch_size in batch_sizes:
        n_updates = int(n_updates_for_batch(int(batch_size)))
        vals = np.empty(n_trials)
        for trial in range(n_trials):
            vals[trial] = run_minibatch_trial(
                d, sigma, gamma, int(batch_size), n_updates, w_star, rng
            )
        med.append(np.median(vals))
        lo.append(np.quantile(vals, 0.1))
        hi.append(np.quantile(vals, 0.9))
        print(f"  B={batch_size:4d}, T={n_updates:4d}: median={med[-1]:.3e}")
    return np.array(med), np.array(lo), np.array(hi)


def first_batch_above_threshold(batch_sizes: np.ndarray, values: np.ndarray, threshold: float):
    idx = np.flatnonzero(values >= threshold)
    if len(idx) == 0:
        return None
    return int(batch_sizes[idx[0]])


def main() -> None:
    d = 20
    sigma = 0.3
    n_trials = 45
    batch_sizes = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])

    R2 = d + 2.0
    gamma = 0.5 / R2
    sigma_mle_sq = 0.5 * d * sigma**2

    rng = np.random.default_rng(11)
    w_star = rng.standard_normal(d)
    w_star /= np.linalg.norm(w_star)

    # Fixed-update experiment: the parallel-time horizon is held fixed.
    T_fixed = 250
    t_fixed = T_fixed // 2
    fixed_med, fixed_lo, fixed_hi = median_risks(
        d=d,
        sigma=sigma,
        gamma=gamma,
        batch_sizes=batch_sizes,
        n_updates_for_batch=lambda B: T_fixed,
        w_star=w_star,
        n_trials=n_trials,
        seed=101,
    )
    fixed_bias = tail_gd_bias(gamma, t_fixed, T_fixed)
    fixed_variance = sigma_mle_sq / (batch_sizes * (T_fixed - t_fixed))
    fixed_model = fixed_bias + fixed_variance
    fixed_bcrit = sigma_mle_sq / ((T_fixed - t_fixed) * fixed_bias)

    # Fixed-sample experiment: increasing B reduces the number of updates.
    N_total = 8192
    sample_med, sample_lo, sample_hi = median_risks(
        d=d,
        sigma=sigma,
        gamma=gamma,
        batch_sizes=batch_sizes,
        n_updates_for_batch=lambda B: max(4, N_total // B),
        w_star=w_star,
        n_trials=n_trials,
        seed=202,
    )
    sample_T = np.array([max(4, N_total // int(B)) for B in batch_sizes])
    sample_t = sample_T // 2
    sample_bias = np.array(
        [tail_gd_bias(gamma, int(t0), int(t1)) for t0, t1 in zip(sample_t, sample_T)]
    )
    sample_variance = np.array(
        [sigma_mle_sq / (int(B) * (int(T) - int(t0))) for B, T, t0 in zip(batch_sizes, sample_T, sample_t)]
    )
    sample_model = sample_bias + sample_variance
    sample_bcrit = first_batch_above_threshold(batch_sizes, sample_bias, sample_variance[0])

    # Figure.
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=False)
    color = "#2c7bb6"
    model_color = "0.25"
    bias_color = "#d7191c"

    ax = axes[0]
    ax.fill_between(batch_sizes, fixed_lo, fixed_hi, color=color, alpha=0.16, linewidth=0)
    ax.loglog(batch_sizes, fixed_med, "o-", color=color, linewidth=1.8, label="median over trials")
    ax.loglog(batch_sizes, fixed_model, "--", color=model_color, linewidth=1.4, label="bias + variance model")
    ax.axhline(fixed_bias, color=bias_color, linestyle=":", linewidth=1.4, label="bias level")
    ax.axvline(fixed_bcrit, color="0.45", linestyle=":", linewidth=1.1)
    ax.text(
        fixed_bcrit * 1.05,
        fixed_bias * 1.6,
        rf"$B_{{\rm crit}}\approx {fixed_bcrit:.0f}$",
        fontsize=10,
        color="0.25",
    )
    ax.set_title(r"Fixed updates ($T=250$)")
    ax.set_xlabel(r"batch size $B$")
    ax.set_ylabel(r"tail-averaged excess risk")
    ax.set_ylim(1e-4, 1.5e-2)
    ax.grid(True, alpha=0.25, which="both")

    ax = axes[1]
    ax.fill_between(batch_sizes, sample_lo, sample_hi, color=color, alpha=0.16, linewidth=0)
    ax.loglog(batch_sizes, sample_med, "o-", color=color, linewidth=1.8, label="median over trials")
    ax.loglog(batch_sizes, sample_model, "--", color=model_color, linewidth=1.4, label="bias + variance model")
    ax.loglog(batch_sizes, sample_bias, ":", color=bias_color, linewidth=1.4, label="bias")
    if sample_bcrit is not None:
        ax.axvline(sample_bcrit, color="0.45", linestyle=":", linewidth=1.1)
        ax.text(
            sample_bcrit * 1.08,
            max(sample_variance[0] * 1.4, 2e-4),
            rf"$B_{{\rm crit}}\approx {sample_bcrit}$",
            fontsize=10,
            color="0.25",
        )
    ax.set_title(r"Fixed samples ($N=8192$)")
    ax.set_xlabel(r"batch size $B$")
    ax.set_ylim(1e-4, 5e-1)
    ax.grid(True, alpha=0.25, which="both")

    for ax in axes:
        ax.set_xlim(batch_sizes[0], batch_sizes[-1])
        ax.legend(fontsize=9, loc="best")

    fig.suptitle(
        rf"Mini-batch tail-averaged SGD ($d={d}$, $\sigma={sigma}$, $\gamma R^2=0.5$)",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()

    out = FIGURES_DIR / "sgd_minibatch_saturation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
