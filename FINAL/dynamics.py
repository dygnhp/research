"""
Contact-Hamiltonian dynamics and its RK4 integrator (Section 4).

Per-particle equations:
    dq/dt = p
    dp/dt = -grad V(q) - gamma * p
    dz/dt = ||p||^2 - H,        with H = ||p||^2 / 2 + V(q)

The integrator is a fixed-step RK4 wrapped by jax.lax.scan for the forward
trajectory.  Two flavors are exposed:
    simulate_diff : memory-efficient (jax.checkpoint on each RK4 step) for
                    backprop through training.
    simulate_eval : no checkpoint; faster for forward-only evaluation.

Both return the full trajectory of S, shape (T+1, n_max, 2D+1).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import jit
from jax import checkpoint as jax_checkpoint
from functools import partial

from .terrain import rbf_potential, rbf_gradient


def contact_rhs(S, w, mu, sigma, gamma: float, D: int):
    """Compute dS/dt for the packed state.

    S : (N, 2D+1)  layout [q | p | z]
    """
    q = S[:, :D]
    p = S[:, D:2 * D]
    V = rbf_potential(q, w, mu, sigma)
    gV = rbf_gradient(q, w, mu, sigma)
    p_sq = jnp.sum(p ** 2, axis=-1)
    H = p_sq / 2.0 + V
    dq = p
    dp = -gV - gamma * p
    dz = p_sq - H
    return jnp.concatenate([dq, dp, dz[:, None]], axis=-1)


def make_rk4_step(D: int, gamma: float, dt: float):
    """Return a jitted single RK4 step bound to (D, gamma, dt)."""
    @jit
    def rk4_step(S, w, mu, sigma):
        f = lambda s: contact_rhs(s, w, mu, sigma, gamma, D)
        k1 = f(S)
        k2 = f(S + 0.5 * dt * k1)
        k3 = f(S + 0.5 * dt * k2)
        k4 = f(S + dt * k3)
        return S + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return rk4_step


def _sigma_safe(sigma, sigma_min: float = 0.1, sigma_max: float = 20.0):
    """Clip sigma into a safe range so the Gaussians never become singular.

    The default bounds match the historical behaviour; both are exposed via
    Config (sigma_min, sigma_max) and threaded through the simulator
    constructors below.  Kept as a separate helper so callers/tests can
    inspect or reuse it without touching the physics core.
    """
    return jnp.clip(sigma, sigma_min, sigma_max)


def make_simulate_diff(D: int, gamma: float, dt: float, n_steps: int,
                       sigma_min: float = 0.1, sigma_max: float = 20.0):
    """Differentiable simulator with checkpointing for O(sqrt(T)) memory."""
    rk4 = make_rk4_step(D, gamma, dt)

    def simulate_diff(S0, w, mu, sigma):
        sigma_s = _sigma_safe(sigma, sigma_min, sigma_max)

        @jax_checkpoint
        def step(S, _):
            S_next = rk4(S, w, mu, sigma_s)
            return S_next, S_next

        _, traj = jax.lax.scan(step, S0, None, length=n_steps)
        return jnp.concatenate([S0[None], traj], axis=0)

    return simulate_diff


def make_simulate_eval(D: int, gamma: float, dt: float, n_steps: int,
                       sigma_min: float = 0.1, sigma_max: float = 20.0):
    """Forward-only simulator (no checkpoint)."""
    rk4 = make_rk4_step(D, gamma, dt)

    def simulate_eval(S0, w, mu, sigma):
        sigma_s = _sigma_safe(sigma, sigma_min, sigma_max)

        def step(S, _):
            S_next = rk4(S, w, mu, sigma_s)
            return S_next, S_next

        _, traj = jax.lax.scan(step, S0, None, length=n_steps)
        return jnp.concatenate([S0[None], traj], axis=0)

    return simulate_eval


def split_traj(traj, D: int):
    """Slice a packed trajectory into (q_traj, p_traj, z_traj)."""
    q = traj[..., :D]
    p = traj[..., D:2 * D]
    z = traj[..., 2 * D]
    return q, p, z
