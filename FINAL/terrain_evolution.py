"""
Terrain time-evolution visualizer (PART 2) -- 3D surfaces.

Given the per-100-epoch terrain snapshots collected by train() (in
history["terrain_snapshots"]), render a single PNG that lays the landscapes
out in time order.  Each panel is a MATLAB-surf-style 3D surface

        z = V(x, y)            (height axis = potential, NOT a particle coord)

so attractors (w_k < 0) appear as basins dipping down and barriers
(w_k > 0) as peaks rising up.

Two distinct 'z's, kept strictly separate:
  * the HEIGHT axis of every panel is the potential value V(x, y);
  * the particle's own 3rd coordinate (z_connectivity) is NOT plotted -- when
    D >= 3 it is held fixed at 0.5 to take a single (x, y) slice of V.

For D >= 4 a single (x, y) slice no longer represents the high-dimensional
landscape, so those snapshots are drawn as a plain text panel announcing the
dimension instead of a surface.

The height (V) axis range and the (x, y) range are shared across every D<=3
panel so the surfaces are visually comparable over time.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from .params import assemble_full
from .terrain import rbf_potential


# ---------------------------------------------------------------------------
# Grid / range helpers
# ---------------------------------------------------------------------------
def _xy_extent(cfg) -> Tuple[float, float, float, float]:
    """(xmin, xmax, ymin, ymax) covering the data domain and every attractor,
    matching viz.plot_terrain_2d so the two figures agree."""
    spec = cfg.dataset_spec
    H, W = spec.image_size
    xs_attr = [p[0] for p in spec.attractor_positions.values()]
    ys_attr = [p[1] for p in spec.attractor_positions.values()]
    xmin = min(0.0, min(xs_attr)) - 2.0
    xmax = max(W - 1, max(xs_attr)) + 2.0
    ymin = min(0.0, min(ys_attr)) - 2.0
    ymax = max(H - 1, max(ys_attr)) + 2.0
    return xmin, xmax, ymin, ymax


def _eval_surface(snap: dict, XX: np.ndarray, YY: np.ndarray) -> np.ndarray:
    """Evaluate V on the (x, y) grid for one snapshot, slicing all extra
    coordinates (dim >= 2) at 0.5.  Returns ZZ with XX/YY's shape."""
    D = snap["D"]
    pts = np.stack([XX.ravel(), YY.ravel()], axis=-1).astype(np.float32)
    if D > 2:
        extra = np.full((pts.shape[0], D - 2), 0.5, dtype=np.float32)
        pts_D = np.concatenate([pts, extra], axis=-1)
    else:
        pts_D = pts
    w_f, mu_f = snap["frozen"]
    w, mu, sigma = assemble_full(snap["params"], jnp.asarray(w_f),
                                 jnp.asarray(mu_f))
    V = np.asarray(rbf_potential(jnp.asarray(pts_D), w, mu, sigma))
    return V.reshape(XX.shape)


def _attractor_heights(snap: dict, cfg) -> List[Tuple[float, float, float]]:
    """(x*, y*, V*) for each class attractor, with V* taken on the SAME 0.5
    slice as the surface so the marker sits on the rendered sheet."""
    spec = cfg.dataset_spec
    D = snap["D"]
    out = []
    pts = []
    xy = []
    for lab in spec.class_labels:
        ax, ay = spec.attractor_positions[lab]
        p = np.zeros(D, dtype=np.float32)
        p[0], p[1] = ax, ay
        if D > 2:
            p[2:] = 0.5
        pts.append(p)
        xy.append((ax, ay))
    pts = np.stack(pts, axis=0)
    w_f, mu_f = snap["frozen"]
    w, mu, sigma = assemble_full(snap["params"], jnp.asarray(w_f),
                                 jnp.asarray(mu_f))
    Vs = np.asarray(rbf_potential(jnp.asarray(pts), w, mu, sigma))
    for (ax, ay), v in zip(xy, Vs):
        out.append((ax, ay, float(v)))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def make_terrain_evolution_3d(snapshots: List[dict], cfg,
                              out_path: Optional[str] = None,
                              grid_n: int = 80) -> str:
    """Render the terrain snapshots as a time-ordered grid of 3D surfaces.

    Args:
        snapshots : list of dicts from history["terrain_snapshots"], each
                    {epoch, D, K_learn, marker, params, frozen}.
        cfg       : the experiment Config (for attractor layout & labels).
        out_path  : destination PNG; defaults under research/experiment/.
        grid_n    : surface resolution (80 default; lower to 60 if heavy).

    Returns:
        The path the PNG was written to.
    """
    if not snapshots:
        raise ValueError("No terrain snapshots to render.")

    snaps = sorted(snapshots, key=lambda s: (s["epoch"], s.get("marker", "")))
    xmin, xmax, ymin, ymax = _xy_extent(cfg)
    xs = np.linspace(xmin, xmax, grid_n)
    ys = np.linspace(ymin, ymax, grid_n)
    XX, YY = np.meshgrid(xs, ys)

    # Precompute surfaces for every D<=3 snapshot so we can fix a common
    # height range; D>=4 snapshots are flagged for a text panel.
    surfaces: List[Optional[np.ndarray]] = []
    n_surface = 0
    n_text = 0
    for s in snaps:
        if s["D"] <= 3:
            ZZ = _eval_surface(s, XX, YY)
            surfaces.append(ZZ)
            n_surface += 1
        else:
            surfaces.append(None)
            n_text += 1

    finite_vals = np.concatenate(
        [z.ravel() for z in surfaces if z is not None]) if n_surface else None
    if finite_vals is not None and finite_vals.size:
        vmin = float(np.min(finite_vals))
        vmax = float(np.max(finite_vals))
        if vmin == vmax:
            vmax = vmin + 1e-6
    else:
        vmin, vmax = -1.0, 1.0

    # ---- layout ----------------------------------------------------------
    n = len(snaps)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))
    panel_in = 4.2
    fig = plt.figure(figsize=(n_cols * panel_in, n_rows * panel_in))

    grow_K_count = sum(1 for s in snaps if s.get("marker") == "[+K]")
    grow_D_count = sum(1 for s in snaps if s.get("marker") == "[+D]")

    for i, (s, ZZ) in enumerate(zip(snaps, surfaces)):
        r, c = divmod(i, n_cols)
        ep, D, K = s["epoch"], s["D"], s["K_learn"]
        marker = s.get("marker", "")
        mtag = f" {marker}" if marker else ""

        if ZZ is not None:
            # ---- 3D surface panel (D <= 3) -------------------------------
            ax = fig.add_subplot(n_rows, n_cols, i + 1, projection="3d")
            ax.plot_surface(XX, YY, ZZ, cmap="RdBu_r",
                            vmin=vmin, vmax=vmax,
                            linewidth=0, antialiased=True)
            for (sx, sy, sv) in _attractor_heights(s, cfg):
                ax.scatter([sx], [sy], [sv], marker="*", s=80, c="black",
                           depthshade=False)
            ax.set_zlim(vmin, vmax)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.view_init(elev=35, azim=-60)
            ax.set_xlabel("x", fontsize=8, labelpad=-4)
            ax.set_ylabel("y", fontsize=8, labelpad=-4)
            ax.set_zlabel("V", fontsize=8, labelpad=-4)
            ax.tick_params(labelsize=6, pad=-2)
            ax.set_title(f"epoch {ep} (D={D}, K={K}){mtag}", fontsize=10)
        else:
            # ---- text panel (D >= 4) -------------------------------------
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            ax.axis("off")
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                       facecolor="0.92", edgecolor="0.7"))
            ax.text(0.5, 0.62, f"D = {D}", ha="center", va="center",
                    fontsize=26, fontweight="bold", transform=ax.transAxes)
            ax.text(0.5, 0.40, f"epoch {ep}, K={K}{mtag}", ha="center",
                    va="center", fontsize=12, transform=ax.transAxes)
            # In-figure text stays ASCII: the default matplotlib font has no
            # Hangul glyphs, so a Korean caption would render as tofu boxes.
            ax.text(0.5, 0.26, "(3D surface omitted: D >= 4)",
                    ha="center", va="center", fontsize=9, color="0.4",
                    transform=ax.transAxes)

    final = snaps[-1]
    fig.suptitle(
        f"Terrain evolution -- {cfg.dataset}  "
        f"(final D={final['D']}, K_learn={final['K_learn']}, "
        f"epoch 0~{final['epoch']})",
        fontsize=14, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    # ---- save ------------------------------------------------------------
    if out_path is None:
        out_path = os.path.join("research", "experiment",
                                "terrain_evolution_3d.png")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- report ----------------------------------------------------------
    ep_lo, ep_hi = snaps[0]["epoch"], snaps[-1]["epoch"]
    print(f"[terrain-3d] 스냅샷 수: {n} (epoch {ep_lo}~{ep_hi})")
    print(f"[terrain-3d] 3D 곡면 패널(D<=3): {n_surface}개 / "
          f"텍스트 패널(D>=4): {n_text}개")
    print(f"[terrain-3d] 공통 z범위(V): [{vmin:.2f}, {vmax:.2f}]")
    print(f"[terrain-3d] 성장 이벤트: grow_K x{grow_K_count}, "
          f"grow_D x{grow_D_count}")
    print(f"[terrain-3d] 저장 완료: {out_path}")
    return out_path
