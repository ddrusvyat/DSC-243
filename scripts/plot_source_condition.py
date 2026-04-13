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


def fit_power_law(eigvals, coeffs):
    log_lam = np.log(eigvals)
    log_c = np.log(coeffs + 1e-15)
    mask = coeffs > 1e-12
    A = np.column_stack([log_lam[mask], np.ones(mask.sum())])
    fit = np.linalg.lstsq(A, log_c[mask], rcond=None)[0]
    return fit[0], fit[1]


def main():
    np.random.seed(0)
    n = 1000
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

    proj_s = np.abs(eigvecs.T @ y_smooth)
    proj_r = np.abs(eigvecs.T @ y_rough)

    # Matrix-level coefficients: |c_i| = lambda_i^{-1} |v_i^T y|
    c_s = proj_s / eigvals
    c_r = proj_r / eigvals

    keep = eigvals > 1e-6 * eigvals[0]
    ev = eigvals[keep]
    cs = c_s[keep]
    cr = c_r[keep]

    s_smooth, b_s = fit_power_law(ev, cs)
    s_rough, b_r = fit_power_law(ev, cr)
    print(f"  smooth target: matrix-level slope s' ≈ {s_smooth:.2f}")
    print(f"  rough target:  matrix-level slope s' ≈ {s_rough:.2f}")

    make_plot(ev, cs, cr, s_smooth, s_rough, b_s, b_r, sigma, n)


def make_plot(ev, cs, cr, s_smooth, s_rough, b_s, b_r, sigma, n):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, coeffs, slope, intercept, s_val, label, color in [
        (axes[0], cs, s_smooth, b_s, s_smooth,
         r"Smooth: $\sin(2\pi x)+\frac{1}{2}\cos(4\pi x)$", "#2c7bb6"),
        (axes[1], cr, s_rough, b_r, s_rough, "Rough: random signs", "#d7191c"),
    ]:
        ax.scatter(ev, coeffs, s=10, alpha=0.35, color=color, edgecolors="none",
                   label=r"$|c_i| = \lambda_i^{-1}|v_i^\top y|$")

        lam_range = np.array([ev[-1] * 0.5, ev[0] * 2])
        ax.loglog(lam_range, np.exp(intercept) * lam_range ** slope,
                  "k--", linewidth=1.5,
                  label=(rf"$\lambda^{{{slope:.1f}}}$ fit "
                         rf"$\Rightarrow\; s' \approx {s_val:.1f}$"))

        ax.set_xlabel(r"Eigenvalue $\hat\mu_i$ of $K/n$", fontsize=12)
        ax.set_title(label, fontsize=12)
        ax.legend(fontsize=11, framealpha=0.9, loc="lower right")
        ax.grid(True, alpha=0.25, which="both")

    axes[0].set_ylabel(r"$|c_i| = \lambda_i^{-1}|v_i^\top y|$", fontsize=12)
    fig.suptitle(
        rf"Matrix-level source condition in kernel regression "
        rf"(Laplace kernel on $[0,1]$, $n={n}$, $\sigma={sigma}$)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "source_condition_kernel.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ source_condition_kernel.png  (n={n}, sigma={sigma:.3f})")


if __name__ == "__main__":
    main()
