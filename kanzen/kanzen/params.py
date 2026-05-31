"""
Parameter layout for the RBF terrain (N-class generalization).

Three blocks for K total Gaussians:

  k = 0 .. C-1            FROZEN attractors (one per class)
        - centers at cfg.q_star(label, D)
        - never updated by the optimizer
  k = C .. 2C-1           STEPPING STONES (one per class)
        - placed along the line from the data center toward each
          class attractor, at a fixed fraction of the distance
        - learnable
  k = 2C .. K-1           FREE basis functions
        - on a grid covering the image plane
        - learnable

where C is the number of classes (cfg.n_classes).

sigma is reparameterized via softplus: sigma = softplus(sigma_raw) + 0.1
"""
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import jax.numpy as jnp
import jax.nn as jnn

from .config import Config


def _softplus_inv(y: np.ndarray) -> np.ndarray:
    y_c = np.clip(y, 1e-3, None)
    return np.log(np.expm1(y_c)).astype(np.float32)


def _image_scale(cfg: Config) -> float:
    """Image-size dependent length scale, normalized so 8x8 maps to 1.0."""
    H, W = cfg.dataset_spec.image_size
    return float(W) / 8.0


def make_frozen(cfg: Config, D: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Frozen attractor block: one per class, only (w, mu) are frozen.

    The attractor sigma is NO LONGER part of the frozen block -- it now lives
    in the learnable parameter pytree (see make_learnable's
    'attractor_sigma_raw') so the influence radius is learned rather than
    hand-set.  Only w (depth) and mu (position / label anchor) stay frozen.
    """
    qs = np.stack([cfg.q_star(lab, D) for lab in cfg.class_labels]).astype(np.float32)
    C = qs.shape[0]
    w = jnp.full((C,), float(cfg.frozen_w), dtype=jnp.float32)
    mu = jnp.asarray(qs)
    return w, mu


def make_learnable(cfg: Config, D: int, rng_seed: int = 0) -> Dict[str, jnp.ndarray]:
    """Stepping stones (one per class) + free basis on a grid.

    Stepping-stone placement: between the data center and each attractor,
    at 35% of the way out (clipped to inside the image plane on the (x,y)
    coordinates so the gradient is non-zero at the start of training).
    """
    rng = np.random.RandomState(rng_seed)
    spec = cfg.dataset_spec
    H, W = spec.image_size
    s = _image_scale(cfg)
    data_cx = (W - 1) / 2.0
    data_cy = (H - 1) / 2.0
    n_classes = spec.n_classes

    # ---- stepping stones -------------------------------------------------
    stone_w = []
    stone_mu = []
    stone_sigma = []
    for label in cfg.class_labels:
        # Honor an attractor override so stones aim at the active layout.
        qs = cfg.q_star(label, max(D, 3))
        ax, ay = float(qs[0]), float(qs[1])
        z_target = float(qs[2])
        dx, dy = ax - data_cx, ay - data_cy
        norm = max(np.hypot(dx, dy), 1e-6)
        dist = cfg.stepping_stone_frac * norm
        sx = data_cx + (dx / norm) * dist
        sy = data_cy + (dy / norm) * dist
        # Clip stone position into the image plane so the gradient acts
        # on the initial particle cloud (which lives in [0, W-1] x [0, H-1]).
        sx = float(np.clip(sx, 0.0, W - 1))
        sy = float(np.clip(sy, 0.0, H - 1))
        mu = np.zeros(D, dtype=np.float32)
        mu[0], mu[1] = sx, sy
        if D >= 3:
            mu[2] = z_target
        if D >= 4:
            mu[3] = 0.5
        stone_mu.append(mu)
        stone_w.append(-1.0)
        stone_sigma.append(3.0 * s)

    # ---- free basis on a grid -------------------------------------------
    n_free = cfg.K_init - 2 * n_classes
    if n_free < 0:
        raise ValueError(
            f"K_init={cfg.K_init} too small for {n_classes} classes "
            f"(need at least {2 * n_classes})")
    if n_free > 0:
        nx = max(1, int(np.ceil(np.sqrt(n_free * W / H))))
        ny = max(1, int(np.ceil(n_free / nx)))
        xs = np.linspace(1.0, W - 2.0, nx) if nx > 1 else np.array([W / 2.0])
        ys = np.linspace(1.0, H - 2.0, ny) if ny > 1 else np.array([H / 2.0])
        grid = [(float(x), float(y)) for y in ys for x in xs]
        grid = grid[:n_free]
        while len(grid) < n_free:
            grid.append((float(rng.uniform(0, W - 1)),
                         float(rng.uniform(0, H - 1))))
        free_mu = np.zeros((n_free, D), dtype=np.float32)
        for i, (x, y) in enumerate(grid):
            free_mu[i, 0] = x
            free_mu[i, 1] = y
            if D >= 3:
                free_mu[i, 2] = 0.5
            if D >= 4:
                free_mu[i, 3] = 0.5
        free_w = np.array([0.18 * ((-1.0) ** i) for i in range(n_free)],
                          dtype=np.float32)
        free_sigma = np.full(n_free, 2.0 * s, dtype=np.float32)
    else:
        free_mu = np.zeros((0, D), dtype=np.float32)
        free_w = np.zeros((0,), dtype=np.float32)
        free_sigma = np.zeros((0,), dtype=np.float32)

    # ---- concatenate stones + free --------------------------------------
    w = np.concatenate([np.array(stone_w, dtype=np.float32), free_w])
    mu = np.concatenate([np.stack(stone_mu).astype(np.float32), free_mu], axis=0)
    sigma = np.concatenate([np.array(stone_sigma, dtype=np.float32), free_sigma])
    sigma_raw = _softplus_inv(sigma - 0.1)

    # ---- attractor sigma (now a learnable parameter, shape (C,)) ---------
    # Initialized to the historical fixed value (attractor_sigma_init * s) so
    # the model's behaviour at step 0 is identical to the frozen-sigma version.
    n_classes = spec.n_classes
    attractor_sigma_init = float(cfg.attractor_sigma_init) * s
    attractor_sigma_raw = _softplus_inv(
        np.full((n_classes,), attractor_sigma_init - 0.1, dtype=np.float32))

    return {
        "w":                  jnp.asarray(w),
        "mu":                 jnp.asarray(mu),
        "sigma_raw":          jnp.asarray(sigma_raw),
        "attractor_sigma_raw": jnp.asarray(attractor_sigma_raw),
    }


def assemble_full(params: Dict[str, jnp.ndarray],
                  frozen_w: jnp.ndarray,
                  frozen_mu: jnp.ndarray):
    """Stack frozen + learnable into the (w, mu, sigma) the dynamics expects.

    The frozen block now supplies only (w, mu); the attractor sigma is
    recovered from the learnable pytree key 'attractor_sigma_raw' and placed
    at the FRONT of the sigma array so it stays index-aligned with the frozen
    (w, mu) -- attractors occupy the first C slots of every (w, mu, sigma).
    """
    attractor_sigma = jnn.softplus(params["attractor_sigma_raw"]) + 0.1
    sigma_learn = jnn.softplus(params["sigma_raw"]) + 0.1
    w = jnp.concatenate([frozen_w, params["w"]])
    mu = jnp.concatenate([frozen_mu, params["mu"]])
    sigma = jnp.concatenate([attractor_sigma, sigma_learn])
    return w, mu, sigma
