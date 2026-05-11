"""
Visualization: terrain contours, per-class particle trajectories,
loss curve with growth markers, convergence diagnostics, phase-volume R^2.

All functions accept plain numpy arrays so they can be reused from
notebooks; no JAX is required at viz time.
"""
from __future__ import annotations

from typing import Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm as mpl_cm

from .params import assemble_full
from .terrain import rbf_potential


def _class_colors(n: int):
    """Distinct colors for class trajectories / markers."""
    base = plt.get_cmap("tab10")
    return [base(i % 10) for i in range(n)]


def plot_terrain_2d(ax, state, cfg, grid_n: int = 100, extent=None):
    """Render V(x, y, z=0.5, ...) as filled contours."""
    spec = cfg.dataset_spec
    H, W = spec.image_size
    # Auto extent: from data domain to beyond every attractor
    if extent is None:
        xs_attr = [p[0] for p in spec.attractor_positions.values()]
        ys_attr = [p[1] for p in spec.attractor_positions.values()]
        xmin = min(0.0, min(xs_attr)) - 2.0
        xmax = max(W - 1, max(xs_attr)) + 2.0
        ymin = min(0.0, min(ys_attr)) - 2.0
        ymax = max(H - 1, max(ys_attr)) + 2.0
        extent = (xmin, xmax, ymin, ymax)
    xs = np.linspace(extent[0], extent[1], grid_n)
    ys = np.linspace(extent[2], extent[3], grid_n)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=-1)
    if state.D > 2:
        extra = np.zeros((pts.shape[0], state.D - 2), dtype=pts.dtype)
        if state.D >= 3:
            extra[:, 0] = 0.5
        if state.D >= 4:
            extra[:, 1] = 0.5
        pts_D = np.concatenate([pts, extra], axis=-1)
    else:
        pts_D = pts

    import jax.numpy as jnp
    w, mu, sigma = assemble_full(state.params, *state.frozen)
    V = np.asarray(rbf_potential(jnp.asarray(pts_D), w, mu, sigma))
    V = V.reshape(grid_n, grid_n)
    cs = ax.contourf(XX, YY, V, levels=25, cmap="RdBu_r")
    ax.set_aspect("equal")
    return cs, extent


def overlay_trajectory(ax, traj, mask, color, alpha=0.55):
    real = mask.astype(bool)
    if real.sum() == 0:
        return
    traj_xy = traj[:, real, :2]
    for i in range(traj_xy.shape[1]):
        ax.plot(traj_xy[:, i, 0], traj_xy[:, i, 1], color=color,
                alpha=alpha, lw=0.6)


def summary_figure(state, cfg, history: Dict, growth_log,
                   per_class_traj: Dict = None,
                   out_path: str = "kanzen_summary.png") -> None:
    """Multi-panel training summary.

    per_class_traj : optional dict {label: (traj_array, mask_array)} for
                     overlaying canonical trajectories on the terrain.
    """
    spec = cfg.dataset_spec
    labels = state.class_labels
    colors = _class_colors(len(labels))

    n_class_panels = len(labels)
    fig = plt.figure(figsize=(5 * max(n_class_panels, 3), 9))
    gs = fig.add_gridspec(2, max(n_class_panels, 3), hspace=0.32, wspace=0.30)

    # ---- top row: terrain + trajectory per class ----------------------------
    cs_last = None
    for c, lab in enumerate(labels):
        ax = fig.add_subplot(gs[0, c])
        cs, extent = plot_terrain_2d(ax, state, cfg)
        cs_last = cs
        if per_class_traj is not None and lab in per_class_traj:
            traj, mask = per_class_traj[lab]
            overlay_trajectory(ax, np.asarray(traj), np.asarray(mask),
                               color=colors[c])
        for c2, lab2 in enumerate(labels):
            ax_x, ax_y = spec.attractor_positions[lab2]
            ax.scatter(ax_x, ax_y, marker="*", s=180,
                       c=[colors[c2]], edgecolors="white", linewidths=1.0,
                       zorder=5)
        ax.set_title(f"Terrain + {lab} trajectory")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    # ---- bottom-left: loss + growth markers --------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(history["epoch"], history["loss"], color="black", lw=0.8)
    for ev in growth_log:
        col = "red" if ev["event"] == "grow_K" else "blue"
        ax.axvline(ev["epoch"], color=col, lw=1.0, ls="--", alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss (log)")
    ax.set_title("Loss + growth events")

    # ---- bottom-middle: per-class eps_q over time --------------------------
    ax = fig.add_subplot(gs[1, 1])
    if history["diag"]:
        eps_epochs = [d["epoch"] for d in history["diag"]]
        for c, lab in enumerate(labels):
            ax.plot(eps_epochs,
                    [d[f"eps_q_{lab}"] for d in history["diag"]],
                    label=f"eps_q {lab}", color=colors[c])
        ax.axhline(cfg.eps_q_thresh, color="gray", ls=":", lw=1)
    ax.set_yscale("log"); ax.set_xlabel("epoch")
    ax.set_title("Convergence (eps_q per class)")
    ax.legend(fontsize=8)

    # ---- bottom-right: per-class R^2 --------------------------------------
    if n_class_panels >= 3:
        ax = fig.add_subplot(gs[1, 2])
        if history["diag"]:
            eps_epochs = [d["epoch"] for d in history["diag"]]
            for c, lab in enumerate(labels):
                ax.plot(eps_epochs,
                        [d[f"R2_{lab}"] for d in history["diag"]],
                        label=f"R2 {lab}", color=colors[c])
            ax.axhline(cfg.phase_R2_thresh, color="gray", ls=":", lw=1)
        ax.set_xlabel("epoch"); ax.set_ylabel("phase-volume R^2")
        ax.set_title("R^2 vs exp(-D*gamma*t)")
        ax.set_ylim(-1.1, 1.1)
        ax.legend(fontsize=8)

    fig.suptitle(f"Contact Hamiltonian Machine -- {cfg.dataset} "
                 f"(D={state.D}, K_learn={state.K_learn})", fontsize=12)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] summary saved -> {out_path}")
