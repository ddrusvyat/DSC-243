"""
Illustrate the matrix-level source condition on [0,1] with uniform data.

We plot the initial-error coefficients |c_i| = lambda_i^{-1} |v_i^T y| in the
kernel eigenbasis.  For a smooth target, these decay like lambda_i^{s'} where
s' = s - 1 is the matrix-level source exponent; for a rough target they do not.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from pathlib import Path

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def laplace_kernel(X, sigma):
    D = np.abs(X[:, None] - X[None, :])
    return np.exp(-D / sigma)


def matern32_kernel(X, sigma):
    D = np.abs(X[:, None] - X[None, :])
    r = np.sqrt(3) * D / sigma
    return (1 + r) * np.exp(-r)


def matern52_kernel(X, sigma):
    D = np.abs(X[:, None] - X[None, :])
    r = np.sqrt(5) * D / sigma
    return (1 + r + r ** 2 / 3) * np.exp(-r)


def gaussian_kernel(X, sigma):
    D = np.abs(X[:, None] - X[None, :])
    return np.exp(-D ** 2 / (2 * sigma ** 2))


def fit_power_law(eigvals, coeffs):
    log_lam = np.log(eigvals)
    log_c = np.log(coeffs + 1e-15)
    mask = coeffs > 1e-12
    A = np.column_stack([log_lam[mask], np.ones(mask.sum())])
    fit = np.linalg.lstsq(A, log_c[mask], rcond=None)[0]
    return fit[0], fit[1]


def _plot_source_condition(ax, eigvals, eigvecs, y, color, title):
    proj = np.abs(eigvecs.T @ y)
    c = proj / eigvals

    keep = eigvals > 1e-6 * eigvals[0]
    ev = eigvals[keep]
    cc = c[keep]

    slope, intercept = fit_power_law(ev, cc)

    ax.scatter(ev, cc, s=8, alpha=0.3, color=color, edgecolors="none",
               label=r"$|c_i|$")

    lam_range = np.array([ev[-1] * 0.5, ev[0] * 2])
    ax.loglog(lam_range, np.exp(intercept) * lam_range ** slope,
              "k--", linewidth=1.5,
              label=(rf"$\hat\mu^{{{slope:.1f}}}$ fit "
                     rf"$\Rightarrow\; s' \approx {slope:.1f}$"))

    ax.set_title(title, fontsize=10.5)
    ax.legend(fontsize=8.5, framealpha=0.9, loc="lower right")
    ax.grid(True, alpha=0.25, which="both")
    return slope


def make_single_kernel_plot():
    """Original single-row Laplace-only figure."""
    np.random.seed(0)
    n = 200
    x = np.sort(np.random.rand(n))
    sigma = 0.15

    K = laplace_kernel(x, sigma)
    eigvals, eigvecs = eigh(K / n)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    y_smooth = np.sin(2 * np.pi * x) + 0.5 * np.cos(4 * np.pi * x)
    y_smooth -= y_smooth.mean()

    y_rough = np.sign(np.random.randn(n)).astype(float)
    y_rough -= y_rough.mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    _plot_source_condition(axes[0], eigvals, eigvecs, y_smooth, "#2c7bb6",
                           r"Smooth: $\sin(2\pi x)+\frac{1}{2}\cos(4\pi x)$")
    _plot_source_condition(axes[1], eigvals, eigvecs, y_rough, "#d7191c",
                           "Rough: random signs")

    axes[0].set_ylabel(r"$|c_i| = \hat\mu_i^{-1}|v_i^\top y|$", fontsize=12)
    for ax in axes:
        ax.set_xlabel(r"Eigenvalue $\hat\mu_i$ of $K/n$", fontsize=12)

    fig.suptitle(
        rf"Matrix-level source condition "
        rf"(Laplace kernel, $n={n}$, $\sigma={sigma}$)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "source_condition_kernel.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ source_condition_kernel.png")


def make_all_kernels_plot():
    """Two-panel figure showing that eigenfunctions are shared but eigenvalue
    decay rates differ, which is the sole driver of the source exponent."""
    np.random.seed(0)
    n = 200
    x = np.sort(np.random.rand(n))
    sigma = 0.15

    y_smooth = np.sin(2 * np.pi * x) + 0.5 * np.cos(4 * np.pi * x)
    y_smooth -= y_smooth.mean()

    kernel_list = [
        (r"Laplace ($m=\frac{1}{2}$)", laplace_kernel(x, sigma), "#d7191c"),
        (r"Matérn 3/2 ($m=\frac{3}{2}$)", matern32_kernel(x, sigma), "#e66101"),
        (r"Matérn 5/2 ($m=\frac{5}{2}$)", matern52_kernel(x, sigma), "#5e3c99"),
        (r"Gaussian ($m=\infty$)", gaussian_kernel(x, sigma), "#2c7bb6"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    for kname, K, color in kernel_list:
        eigvals, eigvecs = eigh(K / n)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        b = np.abs(eigvecs.T @ (y_smooth / n))
        c = b / eigvals

        keep = eigvals > 1e-8 * eigvals[0]
        indices = np.arange(1, keep.sum() + 1)
        ev = eigvals[keep]
        bb = b[keep]
        cc = c[keep]

        ax1.scatter(indices, bb, s=8, alpha=0.4, color=color,
                    edgecolors="none", label=kname)

        slope, intercept = fit_power_law(ev, cc)
        ax2.scatter(ev, cc, s=8, alpha=0.4, color=color,
                    edgecolors="none", label=kname + rf" ($s'\approx{slope:.1f}$)")

        lam_range = np.array([ev[-1] * 0.5, ev[0] * 2])
        ax2.loglog(lam_range, np.exp(intercept) * lam_range ** slope,
                   "--", color=color, linewidth=1.5, alpha=0.7)

        print(f"  {kname:30s}  s'={slope:.2f}")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Mode index $i$", fontsize=12)
    ax1.set_ylabel(r"$|b_i| = |v_i^\top (y/n)|$", fontsize=12)
    ax1.set_title(
        r"Projections $b_i$ onto eigenbasis (kernel-independent)",
        fontsize=11)
    ax1.legend(fontsize=8.5, framealpha=0.9, loc="lower left")
    ax1.grid(True, alpha=0.25, which="both")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"Eigenvalue $\hat\mu_i$ of $K/n$", fontsize=12)
    ax2.set_ylabel(r"$|c_i| = |b_i|/\hat\mu_i$", fontsize=12)
    ax2.set_title(
        r"Initial error coefficients $c_i = b_i/\hat\mu_i$",
        fontsize=11)
    ax2.legend(fontsize=8.5, framealpha=0.9, loc="lower right")
    ax2.grid(True, alpha=0.25, which="both")

    fig.suptitle(
        r"Same eigenfunctions, different eigenvalue decay $\Rightarrow$ "
        r"different source exponents"
        f"\n($n={n}$, $\\sigma={sigma}$, "
        r"$h(x)=\sin(2\pi x)+\frac{1}{2}\cos(4\pi x)$)",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "source_condition_all_kernels.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ source_condition_all_kernels.png")


def main():
    make_single_kernel_plot()
    make_all_kernels_plot()


if __name__ == "__main__":
    main()
