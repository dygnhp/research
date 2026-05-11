"""
Visualization: terrain contours, particle trajectories, loss curves,
phase-volume contraction, and an integrated summary figure.

These functions are deliberately matplotlib-only (no JAX) and accept
plain numpy arrays so they can be reused from notebooks.
"""
from __future__ import annotations

from typing import Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .params import assemble_full
from .dynamics import split_traj


def plot_terrain_2d(ax, state, cfg, grid_n: int = 80,
                    extent=(-10.0, 10.0)):
    """Render V(x, y, z=fixed) as filled contours.  When D >= 3, the
    z-slice is taken at z = 0.5; further dims are set to 0."""
    xs = np.linspace(extent[0], extent[1], grid_n)
    ys = np.linspace(extent[0], extent[1], grid_n)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=-1)  # (grid^2, 2)
    if state.D > 2:
        extra = np.zeros((pts.shape[0], state.D - 2), dtype=pts.dtype)
        if state.D >= 3:
            extra[:, 0] = 0.5
        pts_D = np.concatenate([pts, extra], axis=-1)
    else:
        pts_D = pts

    import jax.numpy as jnp
    from .terrain import rbf_potential
    w, mu, sigma = assemble_full(state.params, *state.frozen)
    V = np.asarray(rbf_potential(jnp.asarray(pts_D), w, mu, sigma))
    V = V.reshape(grid_n, grid_n)

    cs = ax.contourf(XX, YY, V, levels=25, cmap="RdBu_r")
    ax.set_aspect("equal")
    return cs


def overlay_trajectory(ax, traj, mask, color="black", alpha=0.5):
    real = mask.astype(bool)
    traj = traj[:, real, :2]  # (T+1, n_real, 2)
    for i in range(traj.shape[1]):
        ax.plot(traj[:, i, 0], traj[:, i, 1], color=color,
                alpha=alpha, lw=0.6)


def summary_figure(state, cfg, history: Dict, growth_log,
                   trajectory_O=None, trajectory_X=None,
                   mask_O=None, mask_X=None,
                   out_path: str = "kanzen_summary.png") -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # (0,0) terrain + O traj
    cs = plot_terrain_2d(axes[0, 0], state, cfg)
    if trajectory_O is not None:
        overlay_trajectory(axes[0, 0], np.asarray(trajectory_O),
                           np.asarray(mask_O), color="navy", alpha=0.6)
    axes[0, 0].scatter(*cfg.q_star_O_xy, marker="*", s=200, c="navy",
                       edgecolors="white", label="q*_O")
    axes[0, 0].scatter(*cfg.q_star_X_xy, marker="*", s=200, c="firebrick",
                       edgecolors="white", label="q*_X")
    axes[0, 0].set_title("Terrain + O trajectory")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.colorbar(cs, ax=axes[0, 0], fraction=0.045)

    # (0,1) terrain + X traj
    cs2 = plot_terrain_2d(axes[0, 1], state, cfg)
    if trajectory_X is not None:
        overlay_trajectory(axes[0, 1], np.asarray(trajectory_X),
                           np.asarray(mask_X), color="firebrick", alpha=0.6)
    axes[0, 1].scatter(*cfg.q_star_O_xy, marker="*", s=200, c="navy",
                       edgecolors="white")
    axes[0, 1].scatter(*cfg.q_star_X_xy, marker="*", s=200, c="firebrick",
                       edgecolors="white")
    axes[0, 1].set_title("Terrain + X trajectory")
    fig.colorbar(cs2, ax=axes[0, 1], fraction=0.045)

    # (0,2) loss curve with growth events
    ax = axes[0, 2]
    ax.plot(history["epoch"], history["loss"], color="black", lw=0.8)
    for ev in growth_log:
        c = "red" if ev["event"] == "grow_K" else "blue"
        ax.axvline(ev["epoch"], color=c, lw=1.0, ls="--", alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (log)")
    ax.set_title("Loss + growth events")

    # (1,0) terrain RBF centers as scatter
    ax = axes[1, 0]
    mu = np.asarray(state.params["mu"])
    w = np.asarray(state.params["w"])
    sizes = np.clip(np.abs(w) * 400, 20, 400)
    colors = ["firebrick" if wk > 0 else "navy" for wk in w]
    fmu = np.asarray(state.frozen[1])
    fw = np.asarray(state.frozen[0])
    ax.scatter(fmu[:, 0], fmu[:, 1], s=400, c="black", marker="*",
               edgecolors="white", label="frozen (attractor)")
    ax.scatter(mu[:, 0], mu[:, 1], s=sizes, c=colors, alpha=0.7,
               edgecolors="white",
               label="learnable (red=+,blue=-)")
    ax.set_xlim(-12, 12); ax.set_ylim(-12, 12)
    ax.set_aspect("equal")
    ax.set_title(f"RBF centers (D={state.D}, K_learn={state.K_learn})")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(alpha=0.3)

    # (1,1) eps_q / eps_p over epochs
    ax = axes[1, 1]
    if history["diag"]:
        eps_epochs = [d["epoch"] for d in history["diag"]]
        eqO = [d["eps_q_O"] for d in history["diag"]]
        eqX = [d["eps_q_X"] for d in history["diag"]]
        epO = [d["eps_p_O"] for d in history["diag"]]
        epX = [d["eps_p_X"] for d in history["diag"]]
        ax.plot(eps_epochs, eqO, label="eps_q_O", color="navy")
        ax.plot(eps_epochs, eqX, label="eps_q_X", color="firebrick")
        ax.plot(eps_epochs, epO, label="eps_p_O", color="navy", ls="--")
        ax.plot(eps_epochs, epX, label="eps_p_X", color="firebrick", ls="--")
        ax.axhline(cfg.eps_q_thresh, color="gray", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_title("Convergence diagnostics")
    ax.legend(fontsize=8)

    # (1,2) phase-volume R^2 over epochs
    ax = axes[1, 2]
    if history["diag"]:
        eps_epochs = [d["epoch"] for d in history["diag"]]
        R2O = [d["R2_O"] for d in history["diag"]]
        R2X = [d["R2_X"] for d in history["diag"]]
        ax.plot(eps_epochs, R2O, label="R2 O", color="navy")
        ax.plot(eps_epochs, R2X, label="R2 X", color="firebrick")
        ax.axhline(cfg.phase_R2_thresh, color="gray", ls=":", lw=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("phase volume R^2")
    ax.set_title("Phase-volume fit to exp(-D*gamma*t)")
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=8)

    fig.suptitle("Contact Hamiltonian Machine -- training summary", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] summary saved -> {out_path}")
