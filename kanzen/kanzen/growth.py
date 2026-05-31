"""
Autonomous landscape growth (Sections 7-8).

Two growth events are supported:

  grow_K : add K_grow new RBFs to the terrain.  The new centers are placed
           along the trajectories of mis-classified samples, where the
           current potential carries the wrong sign for the class.  Each
           new RBF's weight sign is chosen so that the local landscape
           steers the failing class toward its attractor:
              - if an X-class particle visits a location where V is too
                attractive toward the O side, plant a repulsive RBF (+w);
              - if it visits a location where the X attractor's basin
                vanishes, plant an attracting RBF (-w).

  grow_D : extend every learnable mu by one zero-initialized dimension,
           rescale every learnable sigma by sqrt(D_new/D_old) (Section 8:
           sigma must scale with sqrt(D) so the gradient does not vanish in
           higher-dimensional space), and signal the caller to rebuild the
           simulator / preprocess at D+1.

A simple moving-window plateau detector decides when growth is warranted.
"""
from __future__ import annotations

from typing import Dict, Tuple, List
import numpy as np
import jax.numpy as jnp

from .config import Config
from .params import _softplus_inv


# ---------------------------------------------------------------------------
# Plateau detector
# ---------------------------------------------------------------------------
class PlateauDetector:
    """Moving-window relative-improvement plateau detector."""

    def __init__(self, window: int = 100, threshold: float = 0.01):
        self.window = window
        self.threshold = threshold
        self.losses: List[float] = []

    def update(self, loss: float) -> None:
        self.losses.append(float(loss))

    def is_plateau(self) -> bool:
        n = len(self.losses)
        if n < 2 * self.window:
            return False
        old = float(np.mean(self.losses[-2 * self.window:-self.window]))
        new = float(np.mean(self.losses[-self.window:]))
        if abs(old) < 1e-12:
            return False
        improvement = (old - new) / abs(old)
        return improvement < self.threshold

    def reset(self) -> None:
        """Reset after a growth event so we observe a fresh window."""
        self.losses = []


# ---------------------------------------------------------------------------
# Helpers for sign decision on new K
# ---------------------------------------------------------------------------
def _choose_new_rbf_sign_and_loc(traj_q: np.ndarray, mask: np.ndarray,
                                 q_star: np.ndarray) -> Tuple[float, np.ndarray]:
    """Decide where and with what sign to plant a new RBF for a failing class.

    Heuristic:
      - location: the midpoint of the average particle's trajectory between
        time 0 and time T (mid-trajectory).  This is empirically a good
        place to drop a stepping stone for classes that fail to reach q*.
      - sign:    -1 (attractor) -- a class that did not converge needs MORE
        attraction toward its target.  This matches the MD's stepping-stone
        idea.  A separate diagnostic could refine this to repulsive when the
        failure mode is overshoot, but for the canonical "X gets stuck near
        the origin" failure the attracting choice is correct.

    traj_q : (T+1, N, D)   q-trajectory of a single class's forward pass
    mask   : (N,)          real-particle mask
    q_star : (D,)          target attractor for this class
    """
    T1 = traj_q.shape[0]
    mid_t = T1 // 2
    real = mask.astype(bool)
    if real.sum() == 0:
        return -1.0, q_star.copy()
    mid_com = traj_q[mid_t][real].mean(axis=0)
    # Pull toward the half-point between current mid-trajectory CoM and target.
    new_mu = 0.5 * (mid_com + q_star)
    return -1.0, new_mu.astype(np.float32)


# ---------------------------------------------------------------------------
# grow_K
# ---------------------------------------------------------------------------
def grow_K(params: Dict[str, jnp.ndarray],
           D: int,
           K_grow: int,
           diagnostic_per_class: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
           default_sigma: float = 2.0,
           image_size: Tuple[int, int] = (8, 8)
           ) -> Tuple[Dict[str, jnp.ndarray], int]:
    """Add K_grow new RBFs to the learnable set.

    diagnostic_per_class : list of (traj_q, mask, q_star) tuples for each
                           failing class.  The list length determines how
                           many "guided" placements we make; the rest are
                           placed by alternating signs in the data domain.
    image_size           : (H, W) of the source image.  Random fill RBFs
                           are placed uniformly inside [0, W-1] x [0, H-1]
                           so the growth step is dataset-agnostic (8x8,
                           16x16, 32x32, ...).  Default (8, 8) preserves
                           legacy behaviour for any caller that does not
                           pass this argument.

    Returns the new params dict (same keys, longer first axis) and the
    number of RBFs actually added (always == K_grow).
    """
    w = np.asarray(params["w"]).copy()
    mu = np.asarray(params["mu"]).copy()
    sigma_raw = np.asarray(params["sigma_raw"]).copy()
    sigma = np.log1p(np.exp(sigma_raw)) + 0.1  # softplus

    H, W = image_size

    new_w_list: List[np.ndarray] = []
    new_mu_list: List[np.ndarray] = []
    new_sigma_list: List[np.ndarray] = []

    # First: as many guided placements as we have diagnostic data for.
    n_guided = min(K_grow, len(diagnostic_per_class))
    for i in range(n_guided):
        traj_q, mask, q_star = diagnostic_per_class[i]
        sign, loc = _choose_new_rbf_sign_and_loc(traj_q, mask, q_star)
        new_w_list.append(np.array([sign * 0.5], dtype=np.float32))
        new_mu_list.append(loc.reshape(1, D).astype(np.float32))
        new_sigma_list.append(np.array([float(np.mean(sigma))], dtype=np.float32))

    # Fill the rest with alternating-sign random placements in the data domain.
    rng = np.random.RandomState(int(np.sum(np.abs(w) * 1000)) % (2 ** 31 - 1))
    fill = K_grow - n_guided
    for j in range(fill):
        sign = -1.0 if (j % 2 == 0) else 1.0
        loc = np.zeros(D, dtype=np.float32)
        loc[0] = rng.uniform(0.0, max(W - 1, 1e-3))
        loc[1] = rng.uniform(0.0, max(H - 1, 1e-3))
        if D >= 3:
            loc[2] = 0.5
        if D >= 4:
            loc[3] = 0.5
        new_w_list.append(np.array([sign * 0.3], dtype=np.float32))
        new_mu_list.append(loc.reshape(1, D).astype(np.float32))
        new_sigma_list.append(np.array([default_sigma], dtype=np.float32))

    new_w = np.concatenate([w, np.concatenate(new_w_list)])
    new_mu = np.concatenate([mu, np.concatenate(new_mu_list, axis=0)])
    new_sigma = np.concatenate([sigma, np.concatenate(new_sigma_list)])
    new_sigma_raw = _softplus_inv(new_sigma - 0.1)

    out = {
        "w": jnp.asarray(new_w),
        "mu": jnp.asarray(new_mu),
        "sigma_raw": jnp.asarray(new_sigma_raw),
    }
    # grow_K does not change the attractor count (fixed at C classes); carry
    # the learnable attractor sigma through unchanged.
    if "attractor_sigma_raw" in params:
        out["attractor_sigma_raw"] = params["attractor_sigma_raw"]
    return out, K_grow


# ---------------------------------------------------------------------------
# grow_D
# ---------------------------------------------------------------------------
def grow_D(params: Dict[str, jnp.ndarray],
           D_old: int,
           D_new: int) -> Dict[str, jnp.ndarray]:
    """Extend every learnable RBF center by (D_new - D_old) zero-padded dims
    and rescale every learnable sigma by sqrt(D_new / D_old).

    Section 8: as D grows, pairwise distances in R^D scale like sqrt(D),
    so to keep the per-RBF support comparable (and the gradient non-vanishing)
    we must scale sigma by the same factor.
    """
    if D_new <= D_old:
        return params
    mu = np.asarray(params["mu"])      # (K_learn, D_old)
    K_learn = mu.shape[0]
    pad = np.zeros((K_learn, D_new - D_old), dtype=mu.dtype)
    new_mu = np.concatenate([mu, pad], axis=-1)

    # Rescale sigma.  sigma is stored as sigma_raw with sigma = softplus(raw) + 0.1.
    scale = float(np.sqrt(D_new / D_old))

    def _scale_sigma_raw(raw):
        sigma = np.log1p(np.exp(np.asarray(raw))) + 0.1
        return _softplus_inv(sigma * scale - 0.1)

    sigma_raw_new = _scale_sigma_raw(params["sigma_raw"])

    out = {
        "w": params["w"],
        "mu": jnp.asarray(new_mu),
        "sigma_raw": jnp.asarray(sigma_raw_new),
    }
    # The learnable attractor sigma must scale by sqrt(D) too, exactly like the
    # free-RBF sigma, so attractor basins do not collapse in higher dimension.
    if "attractor_sigma_raw" in params:
        out["attractor_sigma_raw"] = jnp.asarray(
            _scale_sigma_raw(params["attractor_sigma_raw"]))
    return out


def grow_frozen_D(frozen_mu: jnp.ndarray,
                  frozen_sigma: jnp.ndarray,
                  D_old: int, D_new: int):
    """Apply the same dimension-padding rule to the frozen attractor block.

    The z-component (index 2) is already class-specific and should not be
    touched; new dimensions beyond D_old are appended as zeros.  Sigma is
    rescaled by sqrt(D_new/D_old) to match the learnable side.
    """
    if D_new <= D_old:
        return frozen_mu, frozen_sigma
    mu = np.asarray(frozen_mu)
    K_f = mu.shape[0]
    pad = np.zeros((K_f, D_new - D_old), dtype=mu.dtype)
    new_mu = np.concatenate([mu, pad], axis=-1)
    scale = float(np.sqrt(D_new / D_old))
    new_sigma = np.asarray(frozen_sigma) * scale
    return jnp.asarray(new_mu), jnp.asarray(new_sigma)
