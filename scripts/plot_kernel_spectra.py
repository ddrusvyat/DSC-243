"""
Plot eigenvalue spectra of Gaussian, Laplace, and Matérn-5/2 kernel matrices
built from random data in R^100 with a large number of samples.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.linalg import eigvalsh
from pathlib import Path

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

np.random.seed(42)


def pairwise_distances(X):
    sq = np.sum(X ** 2, axis=1)
    return np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0)


def gaussian_kernel(X, sigma):
    return np.exp(-pairwise_distances(X) / (2 * sigma ** 2))


def laplace_kernel(X, sigma):
    D = np.sqrt(pairwise_distances(X))
    return np.exp(-D / sigma)


def matern52_kernel(X, sigma):
    D = np.sqrt(pairwise_distances(X))
    r = np.sqrt(5) * D / sigma
    return (1.0 + r + r ** 2 / 3.0) * np.exp(-r)


def make_kernel_spectra_plot():
    n = 5000
    d = 100
    X = np.random.randn(n, d)

    med_dist = np.median(np.sqrt(pairwise_distances(X[:500]).ravel()))
    sigma = med_dist

    kernels = [
        ("Gaussian (RBF)", gaussian_kernel(X, sigma)),
        ("Laplace", laplace_kernel(X, sigma)),
        (r"Matérn 5/2", matern52_kernel(X, sigma)),
    ]

    colors = ["#2c7bb6", "#d7191c", "#1a9641"]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for (name, K), color in zip(kernels, colors):
        eigs = eigvalsh(K)[::-1]
        eigs = np.maximum(eigs, 1e-16)
        idx = np.arange(1, len(eigs) + 1)
        ax.loglog(idx, eigs / eigs[0], linewidth=1.8, label=name, color=color)

    ax.set_xlabel("Eigenvalue index $i$", fontsize=13)
    ax.set_ylabel(r"$\lambda_i\,/\,\lambda_1$", fontsize=13)
    ax.set_title(
        rf"Kernel matrix eigenvalue spectra ($n={n},\ d={d}$)",
        fontsize=14,
    )
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.25, which="both")
    ax.set_xlim(1, n)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "kernel_spectra.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ kernel_spectra.png")


if __name__ == "__main__":
    make_kernel_spectra_plot()
