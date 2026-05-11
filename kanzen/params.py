"""
Parameter layout for the RBF terrain.

The terrain has three roles for its K gaussians:

  k = 0 .. n_frozen-1                       FROZEN attractors (O / X)
        - centers placed at the class targets q* (Section 5)
        - never updated by the optimizer
  k = n_frozen .. n_frozen+n_stones-1       STEPPING STONES
        - centers placed near the data domain so particles see
          a finite gradient at the start of training
        - learnable
  k = n_frozen+n_stones .. K-1              FREE basis functions
        - distributed across the data plane
        - learnable

The 'learnable' object stored in the optimizer holds (w, mu, sigma_raw)
for *learnable* RBFs only.  Frozen RBFs live as constants alongside.

sigma is reparameterized via softplus to guarantee positivity:
    sigma = softplus(sigma_raw) + 0.1
"""
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import jax.numpy as jnp
import jax.nn as jnn

from .config import Config


def _softplus_inv(y: np.ndarray) -> np.ndarray:
    """Inverse of softplus, with a tiny floor to avoid log(0)."""
    y_clipped = np.clip(y, 1e-3, None)
    return np.log(np.expm1(y_clipped)).astype(np.float32)


def make_frozen(cfg: Config, D: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Frozen attractor RBFs (one per class)."""
    qO = cfg.q_star("O", D)
    qX = cfg.q_star("X", D)
    w = jnp.asarray(np.array([-2.0, -2.0], dtype=np.float32))
    mu = jnp.asarray(np.stack([qO, qX]).astype(np.float32))
    sigma = jnp.asarray(np.array([2.0, 2.0], dtype=np.float32))
    return w, mu, sigma


def make_learnable(cfg: Config, D: int, rng_seed: int = 0) -> Dict[str, jnp.ndarray]:
    """Initial learnable RBFs: stepping stones + free basis."""
    rng = np.random.RandomState(rng_seed)

    # ---------- stepping stones (one per class, in or near the data domain) ----------
    stone_O = np.zeros(D, dtype=np.float32)
    stone_O[0], stone_O[1] = 6.0, 6.0
    if D >= 3:
        stone_O[2] = 0.88

    stone_X = np.zeros(D, dtype=np.float32)
    stone_X[0], stone_X[1] = 0.0, 0.0
    if D >= 3:
        stone_X[2] = 0.12

    w_stones = np.array([-1.0, -1.0], dtype=np.float32)
    mu_stones = np.stack([stone_O, stone_X]).astype(np.float32)
    sig_stones = np.array([3.0, 3.0], dtype=np.float32)

    # ---------- free basis (initialized on a coarse grid over the image plane) -----
    n_free = cfg.K_init - cfg.n_frozen - cfg.n_stones
    if n_free < 0:
        raise ValueError("K_init must be >= n_frozen + n_stones")

    # 4 x 3 grid spanning the 8x8 image plane (12 free RBFs by default)
    xs = np.linspace(0.5, 6.5, 4, dtype=np.float32)
    ys = np.linspace(1.0, 6.0, 3, dtype=np.float32)
    grid_xy = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)
    if n_free > len(grid_xy):
        extra = rng.uniform(0.0, 7.0, size=(n_free - len(grid_xy), 2)).astype(np.float32)
        grid_xy = np.concatenate([grid_xy, extra], axis=0)
    grid_xy = grid_xy[:n_free]

    free_mu = np.zeros((n_free, D), dtype=np.float32)
    free_mu[:, 0] = grid_xy[:, 0]
    free_mu[:, 1] = grid_xy[:, 1]
    if D >= 3:
        free_mu[:, 2] = 0.5
    if D >= 4:
        free_mu[:, 3] = 0.5

    # Alternating signs for the free RBFs so the initial landscape has both
    # attracting hollows and repelling bumps.  Magnitudes are small so they
    # do not overpower the stepping stones.
    free_w_pattern = np.array([-0.20, 0.15, -0.10, 0.20,
                               -0.15, 0.10, -0.20, 0.15,
                               -0.10, 0.20, -0.15, 0.10], dtype=np.float32)
    if n_free <= len(free_w_pattern):
        free_w = free_w_pattern[:n_free]
    else:
        reps = (n_free // len(free_w_pattern)) + 1
        free_w = np.tile(free_w_pattern, reps)[:n_free]

    free_sig = np.full(n_free, 2.0, dtype=np.float32)

    # ---------- concatenate stones + free, build raw sigma ----------
    w = np.concatenate([w_stones, free_w])
    mu = np.concatenate([mu_stones, free_mu])
    sigma = np.concatenate([sig_stones, free_sig])
    sigma_raw = _softplus_inv(sigma - 0.1)

    return {
        "w": jnp.asarray(w),
        "mu": jnp.asarray(mu),
        "sigma_raw": jnp.asarray(sigma_raw),
    }


def assemble_full(params: Dict[str, jnp.ndarray],
                  frozen_w: jnp.ndarray,
                  frozen_mu: jnp.ndarray,
                  frozen_sigma: jnp.ndarray):
    """Stack frozen + learnable into the (w, mu, sigma) the dynamics expects."""
    sigma_learn = jnn.softplus(params["sigma_raw"]) + 0.1
    w = jnp.concatenate([frozen_w, params["w"]])
    mu = jnp.concatenate([frozen_mu, params["mu"]])
    sigma = jnp.concatenate([frozen_sigma, sigma_learn])
    return w, mu, sigma
