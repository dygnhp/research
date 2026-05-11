"""
Phase-space classification loss (Section 6).

For each class c in {O, X} we run a forward simulation with the same
parameters and compare the masked center-of-mass at time T to the
attractor target q*_c.  The momentum penalty discourages drive-through
trajectories where particles arrive at the target with residual speed.

    L = ||CoM_O(T) - q*_O||^2
      + ||CoM_X(T) - q*_X||^2
      + lambda_p * ( mean ||p_O(T)||^2 + mean ||p_X(T)||^2 )

CoM and momentum statistics are computed with the mask so dummy
(padding) particles never contribute.
"""
from __future__ import annotations

import jax.numpy as jnp

from .params import assemble_full
from .dynamics import split_traj


def _masked_mean(x, mask):
    """Mean along axis 0 (particles), excluding masked-out slots.

    x    : (N, D)
    mask : (N,) bool
    """
    m = mask.astype(x.dtype)[:, None]
    return jnp.sum(x * m, axis=0) / jnp.maximum(jnp.sum(m), 1.0)


def _masked_mean_scalar(x, mask):
    m = mask.astype(x.dtype)
    return jnp.sum(x * m) / jnp.maximum(jnp.sum(m), 1.0)


def forward_and_metrics(simulate_diff, params, frozen, D,
                        S0, mask, q_star):
    """Run a forward pass and return (CoM, mean_p_sq, q_T, p_T).

    Used as a building block by 'loss_two_class' and during gating.
    """
    w_f, mu_f, sigma_f = frozen
    w, mu, sigma = assemble_full(params, w_f, mu_f, sigma_f)
    traj = simulate_diff(S0, w, mu, sigma)
    q_traj, p_traj, _ = split_traj(traj, D)
    q_T = q_traj[-1]   # (N, D)
    p_T = p_traj[-1]   # (N, D)
    com = _masked_mean(q_T, mask)                       # (D,)
    mean_p_sq = _masked_mean_scalar(jnp.sum(p_T ** 2, axis=-1), mask)
    return com, mean_p_sq, q_T, p_T


def loss_two_class(simulate_diff, params, frozen, D,
                   S0_O, mask_O, S0_X, mask_X,
                   q_star_O, q_star_X, lambda_p: float):
    """Combined loss across the two-class batch."""
    com_O, mp_O, _, _ = forward_and_metrics(
        simulate_diff, params, frozen, D, S0_O, mask_O, q_star_O)
    com_X, mp_X, _, _ = forward_and_metrics(
        simulate_diff, params, frozen, D, S0_X, mask_X, q_star_X)
    pos_term = jnp.sum((com_O - q_star_O) ** 2) + jnp.sum((com_X - q_star_X) ** 2)
    mom_term = mp_O + mp_X
    return pos_term + lambda_p * mom_term, {
        "pos": pos_term,
        "mom": mom_term,
        "com_O": com_O,
        "com_X": com_X,
    }
