"""
Animate Lemma 9.1 (the rotation lemma). For any deterministic first-order
algorithm A run on f(x) = bar f(Q^T x), an orthogonal Q can be built
adaptively -- two columns per round -- so that Q^T x_t lies in E_{2t+1}
for every t. The crux of the proof is that the new columns q_{2t+3},
q_{2t+4} committed at round t+1 lie entirely in the orthogonal complement
of the span S_t = span{q_1, ..., q_{2t+2}} that already contains every
past iterate x_0, ..., x_t. Therefore Q^T x_s for s <= t is unaffected:
adding more columns never disturbs the rotated representation of
already-queried points.

The animation visualizes this consistency directly:
  - Left panel: a d x (k+1) heatmap whose column t shows Q^T x_t.
    As rounds progress, the staircase support pattern E_{2t+1} extends
    downward, but columns for s < t never change.
  - Right panel: a status strip for the columns of Q, with newly
    committed columns highlighted at each round.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def adaptive_Q_simulation(d: int, k: int, eta: float, delta: float, rng: np.random.Generator):
    """Run a perturbed-GD 'algorithm' on f(x) = bar f(Q^T x) while building Q
    column by column as prescribed by the proof of Lemma 9.1.
    """
    T = 2 * np.eye(d) - np.eye(d, k=1) - np.eye(d, k=-1)
    e1 = np.zeros(d)
    e1[0] = 1.0

    Q = np.zeros((d, d))
    n = [0]  # number of columns of Q committed so far

    def commit_along(r: np.ndarray) -> None:
        r = r.copy()
        if n[0] > 0:
            r = r - Q[:, : n[0]] @ (Q[:, : n[0]].T @ r)
        nr = float(np.linalg.norm(r))
        if nr > 1e-12:
            Q[:, n[0]] = r / nr
            n[0] += 1
        else:
            commit_arbitrary()

    def commit_arbitrary() -> None:
        for j in range(d):
            v = np.zeros(d)
            v[j] = 1.0
            if n[0] > 0:
                v = v - Q[:, : n[0]] @ (Q[:, : n[0]].T @ v)
            nv = float(np.linalg.norm(v))
            if nv > 1e-12:
                Q[:, n[0]] = v / nv
                n[0] += 1
                return

    # Pre-pick fixed perturbation directions used by the (deterministic)
    # algorithm. They do not depend on Q.
    perturbations = rng.standard_normal((k + 1, d))
    perturbations /= np.linalg.norm(perturbations, axis=1, keepdims=True)

    x0 = rng.standard_normal(d)
    x0 = x0 / np.linalg.norm(x0) * 0.8

    iterates = [x0]
    grads = []

    commit_along(x0)
    commit_arbitrary()
    g0 = Q @ (T @ (Q.T @ x0) - e1)
    grads.append(g0)

    for t in range(k):
        # Perturbed gradient descent: a deterministic first-order method that
        # adds a fixed off-Krylov "exploration" direction at every step.
        x_next = iterates[-1] - eta * grads[-1] + delta * perturbations[t]
        iterates.append(x_next)
        commit_along(x_next)
        commit_arbitrary()
        g_next = Q @ (T @ (Q.T @ x_next) - e1)
        grads.append(g_next)

    iterates_arr = np.asarray(iterates)
    grads_arr = np.asarray(grads)
    Qx = iterates_arr @ Q  # rows are (Q^T x_t)
    Qg = grads_arr @ Q
    return Q, Qx, Qg


def main() -> None:
    rng = np.random.default_rng(13)
    d = 14
    k = 5
    eta = 0.25
    delta = 0.45

    Q, Qx, Qg = adaptive_Q_simulation(d, k, eta, delta, rng)
    n_rounds = k + 1

    fig = plt.figure(figsize=(13.0, 6.4), dpi=110)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.6, 1.7], wspace=0.35)
    ax_grid = fig.add_subplot(gs[0])
    ax_q = fig.add_subplot(gs[1])

    abs_max = float(np.abs(Qx).max()) * 1.05

    # Heatmap canvas: rows are coordinates 1..d, columns are rounds 0..k.
    blank = np.full((d, n_rounds), np.nan)
    img = ax_grid.imshow(
        blank,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-abs_max,
        vmax=abs_max,
        interpolation="none",
        origin="upper",
    )
    ax_grid.set_xticks(range(n_rounds))
    ax_grid.set_xticklabels([f"$t={j}$" for j in range(n_rounds)])
    ax_grid.set_yticks(range(d))
    ax_grid.set_yticklabels([str(i) for i in range(1, d + 1)])
    ax_grid.set_xlabel("round")
    ax_grid.set_ylabel(r"rotated coordinate $i$")
    ax_grid.set_title(
        r"$(Q^\top x_t)_i$  —  past columns are frozen once written",
        fontsize=12,
    )
    for j in np.arange(0.5, n_rounds - 0.5, 1):
        ax_grid.axvline(j, color="white", linewidth=0.6)
    for i in np.arange(0.5, d - 0.5, 1):
        ax_grid.axhline(i, color="white", linewidth=0.4)

    cbar = fig.colorbar(img, ax=ax_grid, fraction=0.045, pad=0.02)
    cbar.set_label(r"$(Q^\top x_t)_i$")

    # Region highlighting E_{2t+1} on the current column.
    region = patches.Rectangle(
        (0, 0), 0, 0, fill=False, edgecolor="black", linewidth=1.8, linestyle="--"
    )
    ax_grid.add_patch(region)

    # Right panel: bar strip for the columns of Q.
    coords = np.arange(1, d + 1)
    bars_q = ax_q.barh(
        coords,
        np.ones(d),
        height=0.72,
        color="#e6e6e6",
        edgecolor="#bbbbbb",
        linewidth=0.6,
    )
    ax_q.set_xlim(0, 1.0)
    ax_q.invert_yaxis()
    ax_q.set_ylim(d + 0.5, 0.5)
    ax_q.set_yticks(np.arange(1, d + 1))
    ax_q.set_yticklabels([f"$q_{{{j}}}$" for j in range(1, d + 1)])
    ax_q.set_xticks([])
    ax_q.set_title(
        "columns of $Q$\nfree (gray) · committed (blue) · newly added (orange)",
        fontsize=11,
    )
    for spine in ("top", "right", "bottom"):
        ax_q.spines[spine].set_visible(False)

    suptitle = fig.suptitle("", fontsize=14)

    color_committed = "#2c7bb6"
    edge_committed = "#114878"
    color_new = "#fdae61"
    edge_new = "#8a4a0a"
    color_free = "#e6e6e6"
    edge_free = "#bbbbbb"

    n_pause = 3
    total_frames = n_rounds + n_pause

    def update(frame_index: int):
        t = min(frame_index, n_rounds - 1)
        data_now = np.full((d, n_rounds), np.nan)
        for s in range(t + 1):
            data_now[:, s] = Qx[s]
        img.set_data(data_now)

        # Outline E_{2t+1} support on column t:
        # column t lives at x in [t-0.5, t+0.5]; rows 1..2t+1 are y in [-0.5, 2t+0.5].
        region.set_bounds(t - 0.5, -0.5, 1.0, 2 * t + 1)

        n_committed = 2 * (t + 1)
        new_start = n_committed - 2
        for i, bar in enumerate(bars_q):
            if new_start <= i < n_committed:
                bar.set_color(color_new)
                bar.set_edgecolor(edge_new)
            elif i < new_start:
                bar.set_color(color_committed)
                bar.set_edgecolor(edge_committed)
            else:
                bar.set_color(color_free)
                bar.set_edgecolor(edge_free)

        suptitle.set_text(
            rf"Round $t = {t}$: algorithm queries $x_{{{t}}}$, "
            rf"we commit $q_{{{new_start + 1}}}, q_{{{new_start + 2}}}$ (orange) → "
            rf"$Q^\top x_{{{t}}} \in E_{{{2 * t + 1}}}$"
        )
        return [img, region, suptitle] + list(bars_q)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout(rect=[0, 0, 1, 0.93])

    anim = FuncAnimation(fig, update, frames=total_frames, interval=1500, blit=False)
    out = FIGURES_DIR / "rotation_lemma.gif"
    anim.save(out, writer=PillowWriter(fps=1.0))
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
