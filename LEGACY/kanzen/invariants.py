"""
Physics invariants and convergence diagnostics (Sections 4-5, 9).

Three quantities, computed from a forward trajectory, decide whether a
landscape is good enough to keep training without further growth:

    eps_q  = || CoM(T) - q* ||                  (position error)
    eps_p  = mean_i || p_i(T) ||                (momentum residue, real particles only)
    R^2    = how closely log(phase-volume(t)) tracks the theoretical
             slope -D * gamma * t   (Liouville-violation rate from contact dynamics)

The R^2 score follows the protocol of Block I: per particle we form a
small phase-space cloud by perturbing (q, p) by epsilon and propagating
both forward; the determinant of the empirical covariance gives the
volume.  Here we use a simpler approximation that works particle-cloud-wise:
we treat the (q, p) ensemble across real particles as the cloud and track
log det Cov(t) versus the prediction -D*gamma*t over the trajectory.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .dynamics import split_traj


def epsilon_q(com_T: jnp.ndarray, q_star: jnp.ndarray) -> float:
    return float(jnp.linalg.norm(com_T - q_star))


def epsilon_p(p_T: jnp.ndarray, mask: jnp.ndarray) -> float:
    norms = jnp.linalg.norm(p_T, axis=-1)
    m = mask.astype(p_T.dtype)
    return float(jnp.sum(norms * m) / jnp.maximum(jnp.sum(m), 1.0))


def phase_volume_R2(traj: jnp.ndarray, mask: jnp.ndarray,
                    D: int, gamma: float, dt: float) -> float:
    """Empirical phase-space volume R^2 vs the predicted exp(-D*gamma*t) slope.

    The trajectory shape is (T+1, N, 2D+1).  We extract (q, p), drop padded
    particles, and compute log(det(Cov_{q,p}(t))) at each step.  A small
    diagonal jitter is added for numerical stability before the determinant.
    """
    q, p, _ = split_traj(traj, D)
    qp = jnp.concatenate([q, p], axis=-1)       # (T+1, N, 2D)
    qp_np = np.asarray(qp)
    mask_np = np.asarray(mask).astype(bool)
    if mask_np.sum() < 2 * D + 2:
        return float("nan")
    qp_np = qp_np[:, mask_np, :]                # (T+1, n_real, 2D)
    T1 = qp_np.shape[0]

    log_vols = np.zeros(T1)
    jitter = 1e-6 * np.eye(2 * D)
    for t in range(T1):
        cov = np.cov(qp_np[t].T) + jitter
        sign, logdet = np.linalg.slogdet(cov)
        log_vols[t] = logdet if sign > 0 else np.nan

    finite = np.isfinite(log_vols)
    if finite.sum() < 5:
        return float("nan")
    ts = np.arange(T1, dtype=np.float64) * dt
    # Predicted slope from theory:  log V(t) = log V(0) - D * gamma * t
    pred = log_vols[0] - D * gamma * ts
    pred_f, obs_f = pred[finite], log_vols[finite]
    ss_res = np.sum((obs_f - pred_f) ** 2)
    ss_tot = np.sum((obs_f - obs_f.mean()) ** 2)
    if ss_tot <= 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)
