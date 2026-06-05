"""
Per-particle phase-space gallery.

After a model is trained, re-simulate a class reference image and, for every
particle, render a gallery with one (q_d, p_d) phase-plane per dimension --
the position-vs-momentum trajectory over time, drawn as a line with
equal-time-interval sample points.  The contact dynamics is dissipative
(dp/dt = -grad V - gamma p), so each (q_d, p_d) trajectory should settle
toward (q*_d, 0): a direct, per-particle view of the convergence.

Standalone CLI (loads a saved experiment) :
    python -m FINAL.phase_gallery --exp research/main_exp_1/exp_001 \
           --class O --max-particles 12

The runner also calls make_phase_galleries() when --phase-gallery is passed.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import fields as _dc_fields
from typing import List, Optional

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import Config
from .params import assemble_full
from .preprocess import make_S0
from .dynamics import split_traj


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _render_particle_gallery(q, p, i, D, label, num, qstar, out_dir, n_marks):
    """One particle -> a figure with D (q_d, p_d) phase-plane panels."""
    T1 = q.shape[0]
    samp = np.unique(np.linspace(0, T1 - 1, min(n_marks, T1)).astype(int))
    ncols = min(D, 4)
    nrows = int(np.ceil(D / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.2 * nrows),
                             squeeze=False)
    for d in range(D):
        ax = axes[d // ncols][d % ncols]
        qd = q[:, i, d]
        pd = p[:, i, d]
        ax.plot(qd, pd, lw=0.8, color="0.6", zorder=1)
        ax.scatter(qd[samp], pd[samp], c=samp, cmap="viridis", s=20, zorder=2)
        ax.scatter([qd[0]], [pd[0]], marker="o", c="green", s=60, zorder=3,
                   label="t=0")
        ax.scatter([qd[-1]], [pd[-1]], marker="*", c="red", s=120, zorder=3,
                   label="t=T")
        ax.axhline(0, color="k", lw=0.4, ls=":")          # p = 0 (settled)
        if qstar is not None and d < len(qstar):
            ax.axvline(float(qstar[d]), color="purple", lw=0.8, ls="--",
                       alpha=0.6)                           # q*_d target
        ax.set_xlabel(f"q_{d + 1}", fontsize=9)
        ax.set_ylabel(f"p_{d + 1}", fontsize=9)
        ax.set_title(f"dim {d + 1}", fontsize=9)
        ax.tick_params(labelsize=7)
        if d == 0:
            ax.legend(fontsize=7, loc="best")
    for k in range(D, nrows * ncols):                      # hide unused cells
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(f"Phase-space  '{label}'  particle #{num}  (D={D})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(out_dir, f"{label}_p{num:02d}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def make_phase_space_gallery(state, cfg, label, out_dir, image=None,
                             max_particles: int = 12, n_marks: int = 20,
                             show_target: bool = True) -> List[str]:
    """Render every particle's phase-space gallery for one class reference.

    Args:
        state         : trained TrainerState (params, frozen, simulate_eval, D).
        cfg           : the experiment Config.
        label         : class label whose reference image is simulated.
        out_dir       : directory the per-particle PNGs are written to.
        image         : reference image; defaults to the class canonical.
        max_particles : cap on particles per class (first N).  0 = all.
        n_marks       : number of equal-time sample points drawn per panel.
        show_target   : draw the attractor target q*_d as a vertical line.

    Returns the list of written PNG paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    if image is None:
        image = cfg.dataset_spec.canonicals[label]
    D = state.D
    w, mu, sigma = assemble_full(state.params, *state.frozen)
    S0, mask = make_S0(image, D=D, tau=cfg.tau, n_max=cfg.n_max)
    traj = np.asarray(state.simulate_eval(S0, w, mu, sigma))   # (T+1,N,2D+1)
    q, p, _ = split_traj(traj, D)
    real_idx = np.where(np.asarray(mask).astype(bool))[0]
    if max_particles and len(real_idx) > max_particles:
        real_idx = real_idx[:max_particles]
    qstar = cfg.q_star(label, D) if show_target else None
    paths = []
    for num, i in enumerate(real_idx, 1):
        paths.append(_render_particle_gallery(
            q, p, int(i), D, label, num, qstar, out_dir, n_marks))
    return paths


def make_phase_galleries(state, cfg, out_dir, labels=None,
                         max_particles: int = 12, n_marks: int = 20,
                         show_target: bool = True) -> str:
    """Render phase galleries for several classes into out_dir/phase_space/.

    Returns the phase_space directory path.
    """
    labels = list(labels) if labels else list(cfg.class_labels)
    ps_dir = os.path.join(out_dir, "phase_space")
    total = 0
    for lab in labels:
        paths = make_phase_space_gallery(
            state, cfg, lab, ps_dir, max_particles=max_particles,
            n_marks=n_marks, show_target=show_target)
        total += len(paths)
        print(f"[phase] '{lab}': 입자 갤러리 {len(paths)}장")
    print(f"[phase] 총 {total}장 (클래스당 최대 {max_particles}) -> {ps_dir}")
    return ps_dir


# ---------------------------------------------------------------------------
# Standalone loader + CLI
# ---------------------------------------------------------------------------
def load_state_from_exp(exp_dir: str):
    """Reconstruct (state, cfg) from a saved experiment dir (config.json +
    params.npz) so phase galleries can be rendered without retraining."""
    from .train import _build_state
    with open(os.path.join(exp_dir, "config.json"), encoding="utf-8") as f:
        cfg_dict = json.load(f)
    valid = {fld.name for fld in _dc_fields(Config)}
    kw = {k: v for k, v in cfg_dict.items() if k in valid}
    cfg = Config(**kw, _explicit_keys=set(kw.keys()))   # keep saved values exactly

    npz = np.load(os.path.join(exp_dir, "params.npz"))
    D = int(npz["D"])
    state = _build_state(cfg, D)
    params = {
        "w": jnp.asarray(npz["w"]),
        "mu": jnp.asarray(npz["mu"]),
        "sigma_raw": jnp.asarray(npz["sigma_raw"]),
    }
    if "attractor_sigma_raw" in npz.files:
        params["attractor_sigma_raw"] = jnp.asarray(npz["attractor_sigma_raw"])
    else:
        params["attractor_sigma_raw"] = state.params["attractor_sigma_raw"]
    state.params = params
    state.K_learn = int(npz["K_learn"])
    return state, cfg


def main(argv=None):
    parser = argparse.ArgumentParser(prog="FINAL.phase_gallery")
    parser.add_argument("--exp", required=True,
                        help="experiment dir containing params.npz + config.json")
    parser.add_argument("--class", dest="classes", default=None,
                        help="comma-separated class labels (default: all)")
    parser.add_argument("--max-particles", dest="max_particles", type=int,
                        default=12, help="cap per class (0 = all)")
    parser.add_argument("--n-marks", dest="n_marks", type=int, default=20)
    parser.add_argument("--no-target", dest="no_target", action="store_true",
                        help="do not draw the q* target line")
    parser.add_argument("--out", default=None,
                        help="output dir (default: the experiment dir)")
    args = parser.parse_args(argv)

    state, cfg = load_state_from_exp(args.exp)
    labels = args.classes.split(",") if args.classes else None
    out = args.out or args.exp
    make_phase_galleries(state, cfg, out, labels=labels,
                         max_particles=args.max_particles,
                         n_marks=args.n_marks, show_target=not args.no_target)


if __name__ == "__main__":
    main()
