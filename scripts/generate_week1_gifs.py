"""
Generate GIF animations for Week 1 lecture notes:
  1. GD on well-conditioned vs ill-conditioned quadratic
  2. Fixed stepsize GD vs Chebyshev stepsizes
  3. Convergence rate comparison (GD vs Chebyshev vs CG)
  4. Conjugate gradient directions on a 2D quadratic
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# ── Quadratic helpers ──────────────────────────────────────────────

def make_quadratic(eigenvalues):
    """Return (A, f, grad) for f(x) = 0.5 * x^T A x with given eigenvalues."""
    theta = 0.4
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    A = R @ np.diag(eigenvalues) @ R.T
    f = lambda x: 0.5 * x @ A @ x
    grad = lambda x: A @ x
    return A, f, grad


def gd_iterates(grad, x0, stepsize, n_iters):
    xs = [x0.copy()]
    x = x0.copy()
    for _ in range(n_iters):
        x = x - stepsize * grad(x)
        xs.append(x.copy())
    return np.array(xs)


def chebyshev_stepsizes(L, mu, n):
    """Chebyshev stepsize schedule for a quadratic with eigenvalues in [mu, L]."""
    kappa = L / mu
    stepsizes = []
    for k in range(1, n + 1):
        cos_val = np.cos((2 * k - 1) * np.pi / (2 * n))
        lam_k = mu + 0.5 * (L - mu) * (1 + cos_val)
        stepsizes.append(1.0 / lam_k)
    return stepsizes


def gd_chebyshev_iterates(grad, x0, L, mu, n_cycles, cycle_len):
    xs = [x0.copy()]
    x = x0.copy()
    for _ in range(n_cycles):
        steps = chebyshev_stepsizes(L, mu, cycle_len)
        for s in steps:
            x = x - s * grad(x)
            xs.append(x.copy())
    return np.array(xs)


def cg_iterates(A, x0, n_iters):
    """CG on f(x) = 0.5 x^T A x (i.e., solving Ax = 0 from x0)."""
    xs = [x0.copy()]
    x = x0.copy()
    r = -A @ x  # r = b - Ax, with b = 0
    p = r.copy()
    for _ in range(n_iters):
        Ap = A @ p
        rr = r @ r
        if rr < 1e-30:
            break
        alpha = rr / (p @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        beta = (r_new @ r_new) / rr
        p = r_new + beta * p
        r = r_new
        xs.append(x.copy())
    return np.array(xs)


def contour_grid(A, xs_list, pad=1.5):
    all_pts = np.vstack(xs_list)
    xmin, xmax = all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad
    ymin, ymax = all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 200),
                         np.linspace(ymin, ymax, 200))
    pts = np.stack([xx, yy], axis=-1)
    zz = 0.5 * np.einsum("...i,ij,...j", pts, A, pts)
    return xx, yy, zz


# ── GIF 1: Well-conditioned vs ill-conditioned ────────────────────

def make_gif_condition():
    eigs_good = [1.0, 1.5]
    eigs_bad = [0.2, 10.0]
    A_good, _, grad_good = make_quadratic(eigs_good)
    A_bad, _, grad_bad = make_quadratic(eigs_bad)

    x0 = np.array([4.0, 4.0])
    n_iters = 40

    step_good = 2.0 / (eigs_good[0] + eigs_good[1])
    step_bad = 2.0 / (eigs_bad[0] + eigs_bad[1])

    xs_good = gd_iterates(grad_good, x0, step_good, n_iters)
    xs_bad = gd_iterates(grad_bad, x0, step_bad, n_iters)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    xx1, yy1, zz1 = contour_grid(A_good, [xs_good])
    xx2, yy2, zz2 = contour_grid(A_bad, [xs_bad])

    levels1 = np.linspace(0, zz1.max() * 0.8, 20)
    levels2 = np.linspace(0, zz2.max() * 0.8, 20)

    ax1.contour(xx1, yy1, zz1, levels=levels1, cmap="Blues", alpha=0.6)
    ax2.contour(xx2, yy2, zz2, levels=levels2, cmap="Reds", alpha=0.6)

    kappa_good = max(eigs_good) / min(eigs_good)
    kappa_bad = max(eigs_bad) / min(eigs_bad)
    ax1.set_title(f"Well-conditioned ($\\kappa = {kappa_good:.1f}$)", fontsize=13)
    ax2.set_title(f"Ill-conditioned ($\\kappa = {kappa_bad:.0f}$)", fontsize=13)

    line1, = ax1.plot([], [], "o-", color="royalblue", markersize=4, linewidth=1.5)
    dot1, = ax1.plot([], [], "o", color="navy", markersize=8)
    line2, = ax2.plot([], [], "o-", color="indianred", markersize=4, linewidth=1.5)
    dot2, = ax2.plot([], [], "o", color="darkred", markersize=8)

    ax1.plot(0, 0, "*", color="gold", markersize=15, zorder=5)
    ax2.plot(0, 0, "*", color="gold", markersize=15, zorder=5)

    for ax in (ax1, ax2):
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_aspect("equal")

    fig.suptitle("Gradient Descent: Effect of Condition Number", fontsize=14, y=1.02)
    fig.tight_layout()

    def update(i):
        k = min(i, len(xs_good) - 1)
        line1.set_data(xs_good[:k+1, 0], xs_good[:k+1, 1])
        dot1.set_data([xs_good[k, 0]], [xs_good[k, 1]])
        k2 = min(i, len(xs_bad) - 1)
        line2.set_data(xs_bad[:k2+1, 0], xs_bad[:k2+1, 1])
        dot2.set_data([xs_bad[k2, 0]], [xs_bad[k2, 1]])
        return line1, dot1, line2, dot2

    anim = FuncAnimation(fig, update, frames=n_iters + 5, interval=120, blit=True)
    anim.save(FIGURES_DIR / "gd_condition.gif", writer=PillowWriter(fps=8))
    plt.close(fig)
    print("  ✓ gd_condition.gif")


# ── GIF 2: GD vs Chebyshev ────────────────────────────────────────

def make_gif_chebyshev():
    eigs = [0.3, 8.0]
    A, _, grad = make_quadratic(eigs)
    L, mu = max(eigs), min(eigs)

    x0 = np.array([5.0, 4.0])
    n_iters = 60
    cycle_len = 8
    n_cycles = n_iters // cycle_len

    step_fixed = 2.0 / (L + mu)
    xs_gd = gd_iterates(grad, x0, step_fixed, n_iters)
    xs_cheb = gd_chebyshev_iterates(grad, x0, L, mu, n_cycles, cycle_len)

    fig, ax = plt.subplots(figsize=(8, 6))
    xx, yy, zz = contour_grid(A, [xs_gd, xs_cheb])
    levels = np.linspace(0, zz.max() * 0.7, 25)
    ax.contour(xx, yy, zz, levels=levels, cmap="Greys", alpha=0.5)

    line_gd, = ax.plot([], [], "o-", color="royalblue", markersize=3, linewidth=1.2, label="GD (fixed step)")
    dot_gd, = ax.plot([], [], "o", color="navy", markersize=7)
    line_ch, = ax.plot([], [], "s-", color="orangered", markersize=3, linewidth=1.2, label="Chebyshev steps")
    dot_ch, = ax.plot([], [], "s", color="darkred", markersize=7)

    ax.plot(0, 0, "*", color="gold", markersize=15, zorder=5)
    ax.set_title("GD with Fixed Stepsize vs. Chebyshev Stepsizes", fontsize=13)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(fontsize=11, loc="upper right")
    ax.set_aspect("equal")
    fig.tight_layout()

    n_frames = max(len(xs_gd), len(xs_cheb))

    def update(i):
        k1 = min(i, len(xs_gd) - 1)
        line_gd.set_data(xs_gd[:k1+1, 0], xs_gd[:k1+1, 1])
        dot_gd.set_data([xs_gd[k1, 0]], [xs_gd[k1, 1]])
        k2 = min(i, len(xs_cheb) - 1)
        line_ch.set_data(xs_cheb[:k2+1, 0], xs_cheb[:k2+1, 1])
        dot_ch.set_data([xs_cheb[k2, 0]], [xs_cheb[k2, 1]])
        return line_gd, dot_gd, line_ch, dot_ch

    anim = FuncAnimation(fig, update, frames=n_frames + 5, interval=120, blit=True)
    anim.save(FIGURES_DIR / "gd_vs_chebyshev.gif", writer=PillowWriter(fps=8))
    plt.close(fig)
    print("  ✓ gd_vs_chebyshev.gif")


# ── GIF 3: Convergence comparison ─────────────────────────────────

def make_gif_convergence():
    eigs = [0.3, 8.0]
    A, f, grad = make_quadratic(eigs)
    L, mu = max(eigs), min(eigs)

    x0 = np.array([5.0, 4.0])
    n_iters = 60
    cycle_len = 8

    xs_gd = gd_iterates(grad, x0, 2.0 / (L + mu), n_iters)
    xs_cheb = gd_chebyshev_iterates(grad, x0, L, mu, n_iters // cycle_len, cycle_len)
    xs_cg = cg_iterates(A, x0, n_iters)

    f0 = f(x0)
    err_gd = np.array([f(x) / f0 for x in xs_gd])
    err_cheb = np.array([f(x) / f0 for x in xs_cheb])
    err_cg = np.array([f(x) / f0 for x in xs_cg])

    # Clamp to avoid log(0)
    floor = 1e-16
    err_gd = np.maximum(err_gd, floor)
    err_cheb = np.maximum(err_cheb, floor)
    err_cg = np.maximum(err_cg, floor)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, n_iters)
    ax.set_ylim(1e-15, 2)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("$f(x_k) / f(x_0)$", fontsize=12)
    ax.set_title("Convergence Comparison", fontsize=13)
    ax.grid(True, alpha=0.3)

    line_gd, = ax.plot([], [], "-", color="royalblue", linewidth=2, label="Gradient Descent")
    line_ch, = ax.plot([], [], "-", color="orangered", linewidth=2, label="Chebyshev GD")
    line_cg, = ax.plot([], [], "-", color="seagreen", linewidth=2, label="Conjugate Gradients")
    ax.legend(fontsize=11)
    fig.tight_layout()

    n_frames = n_iters + 10

    def update(i):
        k = min(i + 1, n_iters + 1)
        k_gd = min(k, len(err_gd))
        k_ch = min(k, len(err_cheb))
        k_cg = min(k, len(err_cg))
        line_gd.set_data(np.arange(k_gd), err_gd[:k_gd])
        line_ch.set_data(np.arange(k_ch), err_cheb[:k_ch])
        line_cg.set_data(np.arange(k_cg), err_cg[:k_cg])
        return line_gd, line_ch, line_cg

    anim = FuncAnimation(fig, update, frames=n_frames, interval=100, blit=True)
    anim.save(FIGURES_DIR / "convergence_comparison.gif", writer=PillowWriter(fps=10))
    plt.close(fig)
    print("  ✓ convergence_comparison.gif")


# ── GIF 4: Conjugate Gradient directions ──────────────────────────

def make_gif_cg():
    eigs = [0.5, 6.0]
    A, _, _ = make_quadratic(eigs)

    x0 = np.array([5.0, 4.0])
    n_iters = 3  # CG on 2D converges in 2 steps; show a few

    # Run CG manually to capture search directions
    xs = [x0.copy()]
    directions = []
    x = x0.copy()
    r = -A @ x
    p = r.copy()
    for _ in range(n_iters):
        Ap = A @ p
        rr = r @ r
        if rr < 1e-30:
            break
        alpha = rr / (p @ Ap)
        directions.append((x.copy(), alpha * p))
        x = x + alpha * p
        r_new = r - alpha * Ap
        beta = (r_new @ r_new) / rr
        p = r_new + beta * p
        r = r_new
        xs.append(x.copy())

    xs = np.array(xs)

    fig, ax = plt.subplots(figsize=(8, 6))
    xx, yy, zz = contour_grid(A, [xs], pad=2.0)
    levels = np.linspace(0, zz.max() * 0.7, 25)
    ax.contour(xx, yy, zz, levels=levels, cmap="Greys", alpha=0.5)
    ax.plot(0, 0, "*", color="gold", markersize=15, zorder=5)
    ax.set_title("Conjugate Gradient Method", fontsize=13)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    fig.tight_layout()

    line, = ax.plot([], [], "o-", color="seagreen", markersize=6, linewidth=2)
    dot, = ax.plot([], [], "o", color="darkgreen", markersize=10)
    arrows = []

    total_steps = len(xs) - 1
    frames_per_step = 15
    pause_frames = 8
    total_frames = total_steps * (frames_per_step + pause_frames) + pause_frames

    def update(frame):
        for arr in arrows:
            arr.remove()
        arrows.clear()

        step = min(frame // (frames_per_step + pause_frames), total_steps)
        sub = frame % (frames_per_step + pause_frames)
        t = min(sub / frames_per_step, 1.0)

        show_pts = list(xs[:step])
        if step < total_steps:
            interp = xs[step] + t * (xs[step + 1] - xs[step])
            show_pts.append(interp)
        else:
            show_pts.append(xs[step])

        pts = np.array(show_pts)
        line.set_data(pts[:, 0], pts[:, 1])
        dot.set_data([pts[-1, 0]], [pts[-1, 1]])

        for i in range(min(step + 1, len(directions))):
            if i < step or (i == step and t >= 0.3):
                origin = xs[i]
                d = directions[i]
                scale = min(t, 1.0) if i == step else 1.0
                arr = ax.annotate("", xy=origin + scale * d[1],
                                  xytext=origin,
                                  arrowprops=dict(arrowstyle="->", color="tomato",
                                                  lw=2.0))
                arrows.append(arr)

        return line, dot

    anim = FuncAnimation(fig, update, frames=total_frames, interval=80, blit=False)
    anim.save(FIGURES_DIR / "conjugate_gradient.gif", writer=PillowWriter(fps=12))
    plt.close(fig)
    print("  ✓ conjugate_gradient.gif")


# ── Plot 5: Chebyshev polynomials of various degrees ──────────────

def make_chebyshev_plot():
    x = np.linspace(-1, 1, 500)
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#2980b9", "#e74c3c", "#27ae60", "#8e44ad", "#e67e22"]
    for k, c in zip(range(1, 6), colors):
        y = np.cos(k * np.arccos(x))
        ax.plot(x, y, color=c, linewidth=2, label=f"$T_{k}$")

    ax.axhline(1, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(-1, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(0, color="grey", linewidth=0.5, alpha=0.4)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel("$x$", fontsize=13)
    ax.set_ylabel("$T_k(x)$", fontsize=13)
    ax.set_title("Chebyshev Polynomials of the First Kind", fontsize=14)
    ax.legend(fontsize=12, loc="lower right", ncol=3)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "chebyshev_polynomials.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ chebyshev_polynomials.png")


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating Week 1 figures...")
    make_gif_condition()
    make_gif_chebyshev()
    make_gif_convergence()
    make_gif_cg()
    make_chebyshev_plot()
    print("Done! Figures saved to", FIGURES_DIR)
