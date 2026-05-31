"""
RBF potential V(q) and its analytic gradient (Section 3).

    V(q) = sum_k  w_k * exp( -||q - mu_k||^2 / (2 sigma_k^2) )

The gradient is computed in closed form (no autograd dependence here) so
that the contact RHS stays explicit and inspectable.

Both functions are dimension-agnostic: q in R^{N, D}, mu in R^{K, D}, and
sigma in R^{K}.  An isotropic sigma per RBF keeps the parameter count
small while supporting the sqrt(D) rescaling rule from Section 8.
"""
from __future__ import annotations

import jax.numpy as jnp


def rbf_potential(q, w, mu, sigma):
    """V(q_i) for every particle i.

    q     : (N, D)
    w     : (K,)
    mu    : (K, D)
    sigma : (K,)
    returns: (N,) potential values
    """
    diff = q[:, None, :] - mu[None, :, :]                # (N, K, D)
    sq_dist = jnp.sum(diff ** 2, axis=-1)                # (N, K)
    gauss = jnp.exp(-sq_dist / (2.0 * sigma ** 2))       # (N, K)
    return jnp.sum(w * gauss, axis=-1)                   # (N,)


def rbf_gradient(q, w, mu, sigma):
    """grad_q V at every particle.

    Closed-form derivative:
        d/dq exp(-||q - mu_k||^2 / 2 sigma_k^2)
            = - (q - mu_k) / sigma_k^2 * exp(...)
    """
    diff = q[:, None, :] - mu[None, :, :]                # (N, K, D)
    sq_dist = jnp.sum(diff ** 2, axis=-1)                # (N, K)
    gauss = jnp.exp(-sq_dist / (2.0 * sigma ** 2))       # (N, K)
    factor = w * gauss / (sigma ** 2)                    # (N, K)
    return jnp.sum(-factor[:, :, None] * diff, axis=1)   # (N, D)
