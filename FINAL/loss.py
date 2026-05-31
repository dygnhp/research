"""
N-class phase-space classification loss.

For each class c the same forward simulator is run with that class's
initial state.  The position term penalizes the distance from the
masked center-of-mass at time T to the class's attractor q*_c; the
momentum term penalizes residual speed at T.

    L = sum_c  ||CoM_c(T) - q*_c||^2
      + lambda_p * sum_c mean_i ||p_{c,i}(T)||^2

CoM and momentum statistics are mask-aware so dummy (padding) particles
never contribute.
"""
from __future__ import annotations

import jax.numpy as jnp
import jax.nn as jnn

from .params import assemble_full
from .dynamics import split_traj


def _masked_mean(x, mask):
    m = mask.astype(x.dtype)[:, None]
    return jnp.sum(x * m, axis=0) / jnp.maximum(jnp.sum(m), 1.0)


def _masked_mean_scalar(x, mask):
    m = mask.astype(x.dtype)
    return jnp.sum(x * m) / jnp.maximum(jnp.sum(m), 1.0)


def forward_and_metrics(simulate_diff, params, frozen, D, S0, mask):
    """Run one class's forward pass and return (CoM, mean_p_sq, q_T, p_T)."""
    w_f, mu_f = frozen
    w, mu, sigma = assemble_full(params, w_f, mu_f)
    traj = simulate_diff(S0, w, mu, sigma)
    q_traj, p_traj, _ = split_traj(traj, D)
    q_T = q_traj[-1]
    p_T = p_traj[-1]
    com = _masked_mean(q_T, mask)
    mean_p_sq = _masked_mean_scalar(jnp.sum(p_T ** 2, axis=-1), mask)
    return com, mean_p_sq, q_T, p_T


def loss_batch(simulate_diff, params, frozen, D,
               S0_batch, mask_batch, q_stars, lambda_p: float,
               lambda_attractor_sigma: float = 0.0,
               attractor_sigma_target: float = 0.0):
    """N-class loss.

    S0_batch   : (C, N, 2D+1)
    mask_batch : (C, N)
    q_stars    : (C, D)

    The Python for-loop unrolls into the traced graph at trace time, so
    the compiled function contains one inlined forward pass per class.
    For C=4 with the default n_steps=200 this still fits comfortably in
    a single jitted region.

    lambda_attractor_sigma / attractor_sigma_target add a weak L2 pull of the
    learnable attractor sigma toward its initial value, preventing the
    influence radius from running away (a soft constraint, not a hard cap).
    Set lambda to 0 to disable (e.g. when attractor sigma is frozen).
    """
    pos = 0.0
    mom = 0.0
    coms = []
    n_classes = S0_batch.shape[0]
    for c in range(n_classes):
        com, mp, _, _ = forward_and_metrics(
            simulate_diff, params, frozen, D, S0_batch[c], mask_batch[c])
        pos = pos + jnp.sum((com - q_stars[c]) ** 2)
        mom = mom + mp
        coms.append(com)

    # Weak L2 regularization on the learnable attractor sigma (computed once;
    # attractor sigma is shared across classes, not per forward pass).
    attractor_sigma = jnn.softplus(params["attractor_sigma_raw"]) + 0.1
    sigma_reg = lambda_attractor_sigma * jnp.sum(
        (attractor_sigma - attractor_sigma_target) ** 2)

    total = pos + lambda_p * mom + sigma_reg
    return total, {"pos": pos, "mom": mom, "sigma_reg": sigma_reg,
                   "coms": jnp.stack(coms)}
