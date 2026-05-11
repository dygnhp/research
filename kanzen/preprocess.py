"""
Image -> particle ensemble (Section 2 of the algorithm spec).

Pixels with intensity above tau become particles.  Each particle is lifted
into R^D using a fixed feature stack:

    d = 0   : x  = column index (float)
    d = 1   : y  = (rows - 1 - row) so y increases upward
    d = 2   : z_connectivity   in (0, 1)     -- separates O-like vs X-like local structure
    d = 3   : local_density    in [0, 1]     -- 3x3 window count / 9
    d >= 4  : 0 (placeholder for learned features)

The lifted coordinate is the particle's initial position q(0).  Initial
momentum p(0) and the contact variable z(0) are both zero.  Real particles
are placed in the first slots; remaining slots are filled with dummy
particles whose 'mask' value is False so the loss and dynamics ignore them.
The padded array keeps JIT shapes stable.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp


def _axis_diag_features(image: np.ndarray, r: int, c: int):
    rows, cols = image.shape

    def px(rr, cc):
        if 0 <= rr < rows and 0 <= cc < cols:
            return float(image[rr, cc])
        return 0.0

    d_axis = px(r - 1, c) + px(r + 1, c) + px(r, c - 1) + px(r, c + 1)
    d_diag_signed = (px(r - 1, c - 1) + px(r + 1, c + 1)
                     - px(r - 1, c + 1) - px(r + 1, c - 1))
    return d_axis, d_diag_signed


def _local_density(image: np.ndarray, r: int, c: int) -> float:
    rows, cols = image.shape
    total = 0.0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols:
                total += float(image[rr, cc])
    return total / 9.0


def preprocess(image: np.ndarray, D: int, tau: float = 0.5,
               n_max: int = 64, beta: float = 1.0):
    """Lift an image into a padded particle state.

    Returns:
        q0   : (n_max, D) float32 -- initial position
        p0   : (n_max, D) float32 -- initial momentum (zeros)
        z0   : (n_max,)   float32 -- initial contact variable (zeros)
        mask : (n_max,)   bool    -- True for real particles
    """
    rows, cols = image.shape
    q_list = []
    for r in range(rows):
        for c in range(cols):
            if image[r, c] > tau:
                feat = np.zeros(D, dtype=np.float32)
                feat[0] = float(c)
                feat[1] = float(rows - 1 - r)
                if D >= 3:
                    d_axis, d_diag_signed = _axis_diag_features(image, r, c)
                    score = d_axis - abs(d_diag_signed)
                    feat[2] = 1.0 / (1.0 + np.exp(-beta * score))
                if D >= 4:
                    feat[3] = _local_density(image, r, c)
                q_list.append(feat)

    n_real = len(q_list)
    if n_real == 0:
        raise ValueError("Image has no particles above tau.")
    if n_real > n_max:
        raise ValueError(f"Too many particles ({n_real} > n_max={n_max}).")

    q0 = np.zeros((n_max, D), dtype=np.float32)
    p0 = np.zeros((n_max, D), dtype=np.float32)
    z0 = np.zeros((n_max,), dtype=np.float32)
    mask = np.zeros((n_max,), dtype=bool)
    for i, pos in enumerate(q_list):
        q0[i] = pos
        mask[i] = True

    return (jnp.asarray(q0), jnp.asarray(p0),
            jnp.asarray(z0), jnp.asarray(mask))


def make_S0(image: np.ndarray, D: int, tau: float = 0.5, n_max: int = 64):
    """Pack particle state into a single (n_max, 2D+1) array S0 and mask.

    Layout:  S0[:, 0:D]      = q
             S0[:, D:2D]     = p
             S0[:, 2D]       = z
    """
    q0, p0, z0, mask = preprocess(image, D=D, tau=tau, n_max=n_max)
    S0 = jnp.concatenate([q0, p0, z0[:, None]], axis=-1)
    return S0, mask
