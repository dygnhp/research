"""
=============================================================================
Contact Hamiltonian Fluid Neural Network -- Block II: Learning Loop
=============================================================================
Physics:   Contact Hamiltonian dynamics with learnable RBF potential
Goal:      Learn theta = {w_k, mu_k, sigma_k} so that
               O-image CoM -> q*_O = (8, 8, 0)
               X-image CoM -> q*_X = (-8, -8, 0)
Gradient:  jax.value_and_grad through lax.scan (full BPTT)
           + jax.checkpoint for O(sqrt(N)) memory
Optimizer: optax Adam + warmup-cosine schedule + gradient clipping
Backend:   JAX (CUDA) -- JIT-compiled
Target HW: Windows 11 + NVIDIA GPU (RTX 4060) via WSL2  OR  PyCharm + WSL

Install:
    pip install --upgrade "jax[cuda12]" optax matplotlib scipy

Key design decisions:
  1. block_i.py is NOT imported. All physics functions are copied here
     to avoid JAX JIT / module import conflicts.
  2. diffrax is NOT used. Adjoint = automatic diff through lax.scan.
  3. Attractor parameters (k=0, k=1) are FROZEN throughout training.
  4. sigma is constrained positive via softplus reparameterization:
         sigma = softplus(sigma_raw) + 0.1,  min=0.1
  5. Data-proximal quasi-flat initialization:
         free RBF centers placed inside [0,7]^2 (data domain)
         so that RBF forces are nonzero from the first epoch.
=============================================================================
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax import checkpoint as jax_checkpoint
from functools import partial
import numpy as np
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time

print("=" * 60)
print("JAX version   :", jax.__version__)
print("optax version :", optax.__version__)
print("Devices       :", jax.devices())
print("Default backend:", jax.default_backend())
print("=" * 60)


# ===========================================================================
# SECTION 1: Global constants (identical to block_i.py)
# ===========================================================================
GAMMA    = 1.5
T_FINAL  = 10.0
DT       = 0.05
N_STEPS  = int(T_FINAL / DT)   # 200
N_MAX    = 64
TAU      = 0.5
LAMBDA_P = 0.1                  # momentum penalty coefficient

Q_STAR_O = jnp.array([ 8.0,  8.0, 0.0])
Q_STAR_X = jnp.array([-8.0, -8.0, 0.0])

# Convergence thresholds (identical to Block I)
CONV_Q_THR = 2.0
CONV_P_THR = 0.5


# ===========================================================================
# SECTION 2: Test images (identical to block_i.py)
# ===========================================================================
O_IMAGE = np.array([
    [0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,0,0],
    [0,1,0,0,0,0,1,0],
    [0,1,0,0,0,0,1,0],
    [0,1,0,0,0,0,1,0],
    [0,1,0,0,0,0,1,0],
    [0,0,1,1,1,1,0,0],
    [0,0,0,0,0,0,0,0],
], dtype=float)

X_IMAGE = np.array([
    [1,0,0,0,0,0,0,1],
    [0,1,0,0,0,0,1,0],
    [0,0,1,0,0,1,0,0],
    [0,0,0,1,1,0,0,0],
    [0,0,0,1,1,0,0,0],
    [0,0,1,0,0,1,0,0],
    [0,1,0,0,0,0,1,0],
    [1,0,0,0,0,0,0,1],
], dtype=float)


# ===========================================================================
# SECTION 3: Preprocessing -- Contextual 3D Lifting (copied from block_i.py)
# ===========================================================================
def preprocess(image, tau=TAU, n_max=N_MAX, beta=1.0):
    """
    Convert 8x8 binary image to JAX state arrays with contextual 3D lifting.

    Coordinate convention: row r, col c -> x=c,  y=7-r  (y increases upward)

    3D Lifting:
        d_axis  = sum of ON pixels among 4-connected neighbors
        |d_diag| = |NW+SE - NE-SW|
        z_i(0) = sigmoid(beta * (d_axis - |d_diag|))

        O-arc pixel  : d_axis=2, |d_diag|=0  ->  z ~ 0.88
        X-diag pixel : d_axis=0, |d_diag|=2  ->  z ~ 0.12
    """
    rows, cols = image.shape

    def safe_pixel(r, c):
        if 0 <= r < rows and 0 <= c < cols:
            return float(image[r, c])
        return 0.0

    def compute_z_init(r, c, b):
        d_axis = (safe_pixel(r-1, c) + safe_pixel(r+1, c) +
                  safe_pixel(r, c-1) + safe_pixel(r, c+1))
        d_diag_signed = (safe_pixel(r-1, c-1) + safe_pixel(r+1, c+1) -
                         safe_pixel(r-1, c+1) - safe_pixel(r+1, c-1))
        d_diag = abs(d_diag_signed)
        score  = d_axis - d_diag
        return 1.0 / (1.0 + np.exp(-b * score))

    q_list = []
    for r in range(rows):
        for c in range(cols):
            if image[r, c] > tau:
                x = float(c)
                y = float(rows - 1 - r)
                z = compute_z_init(r, c, beta)
                q_list.append([x, y, z])

    n_real = len(q_list)
    assert n_real > 0,          "No real particles -- check tau"
    assert n_real <= n_max,     f"Too many particles: {n_real} > {n_max}"

    q0_np   = np.zeros((n_max, 3), dtype=np.float32)
    p0_np   = np.zeros((n_max, 3), dtype=np.float32)
    z0_np   = np.zeros((n_max,),   dtype=np.float32)
    mask_np = np.zeros((n_max,),   dtype=bool)

    for i, pos in enumerate(q_list):
        q0_np[i]   = pos
        mask_np[i] = True

    return (jnp.array(q0_np), jnp.array(p0_np),
            jnp.array(z0_np), jnp.array(mask_np))


# Precompute masks and initial states (depend only on images, not params)
_, _, _, MASK_O = preprocess(O_IMAGE)
_, _, _, MASK_X = preprocess(X_IMAGE)


def _make_S0(image):
    q0, p0, z0, _ = preprocess(image)
    return jnp.concatenate([q0, p0, z0[:, None]], axis=-1)


S0_O = _make_S0(O_IMAGE)   # (N_MAX, 7)  -- device constant
S0_X = _make_S0(X_IMAGE)   # (N_MAX, 7)


# ===========================================================================
# SECTION 4: RBF Potential and Gradient (copied from block_i.py)
# ===========================================================================
def rbf_potential(q, w, mu, sigma):
    """
    V(q_i) = sum_k w_k * exp(-||q_i - mu_k||^2 / (2*sigma_k^2))
    q:(N,3)  w:(K,)  mu:(K,3)  sigma:(K,)  ->  V:(N,)
    """
    diff    = q[:, None, :] - mu[None, :, :]           # (N, K, 3)
    sq_dist = jnp.sum(diff ** 2, axis=-1)               # (N, K)
    gauss   = jnp.exp(-sq_dist / (2.0 * sigma ** 2))   # (N, K)
    return jnp.sum(w * gauss, axis=-1)                  # (N,)


def rbf_gradient(q, w, mu, sigma):
    """
    grad_{q_i} V = sum_k w_k * exp(...) * (-(q_i - mu_k) / sigma_k^2)
    q:(N,3)  ->  grad:(N,3)
    """
    diff    = q[:, None, :] - mu[None, :, :]
    sq_dist = jnp.sum(diff ** 2, axis=-1)
    gauss   = jnp.exp(-sq_dist / (2.0 * sigma ** 2))
    factor  = w * gauss / (sigma ** 2)                  # (N, K)
    return jnp.sum(-factor[:, :, None] * diff, axis=1)  # (N, 3)


# ===========================================================================
# SECTION 5: Contact Hamiltonian RHS (copied from block_i.py)
# ===========================================================================
def contact_rhs(S, w, mu, sigma, gamma):
    """
    dq/dt = p
    dp/dt = -grad_q V - gamma * p
    dz/dt = ||p||^2 - H_i    where H_i = ||p||^2/2 + V(q)
    => dH/dt = -gamma * ||p||^2 <= 0  (energy monotone decrease)
    """
    q    = S[:, :3]
    p    = S[:, 3:6]
    V    = rbf_potential(q, w, mu, sigma)
    gV   = rbf_gradient(q, w, mu, sigma)
    p_sq = jnp.sum(p ** 2, axis=-1)
    H_i  = p_sq / 2.0 + V

    dq_dt = p
    dp_dt = -gV - gamma * p
    dz_dt = p_sq - H_i
    return jnp.concatenate([dq_dt, dp_dt, dz_dt[:, None]], axis=-1)


# ===========================================================================
# SECTION 6: RK4 Step (copied from block_i.py)
# ===========================================================================
@partial(jit, static_argnums=(4, 5))
def rk4_step(S, w, mu, sigma, gamma, dt):
    """
    Classical 4-stage Runge-Kutta.
    gamma, dt declared static: prevents re-tracing inside lax.scan.
    """
    f  = partial(contact_rhs, w=w, mu=mu, sigma=sigma, gamma=gamma)
    k1 = f(S)
    k2 = f(S + 0.5 * dt * k1)
    k3 = f(S + 0.5 * dt * k2)
    k4 = f(S +       dt * k3)
    return S + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ===========================================================================
# SECTION 7: Analysis utilities (copied / adapted from block_i.py)
# ===========================================================================
def com_single(q_t, mask):
    """Centre-of-mass of real particles at one timestep. Returns (3,)."""
    mf = mask.astype(jnp.float32)
    return jnp.sum(q_t * mf[:, None], axis=0) / jnp.sum(mf)


def classify_traj(trajectory, mask, qO=Q_STAR_O, qX=Q_STAR_X):
    """Classify by final CoM distance to each attractor."""
    qf = com_single(trajectory[-1, :, :3], mask)
    dO = float(jnp.linalg.norm(qf - qO))
    dX = float(jnp.linalg.norm(qf - qX))
    return ('O' if dO < dX else 'X'), dO, dX, qf


# ===========================================================================
# SECTION 8: simulate_diff -- Pure JAX (grad-compatible)
# ===========================================================================
def simulate_diff(S0, w, mu, sigma, gamma=GAMMA, dt=DT, n_steps=N_STEPS):
    """
    Forward simulation compatible with jax.value_and_grad.

    Design choices:
    - No print statements or numpy conversions (would break grad tracing).
    - jax.checkpoint applied to scan body: reduces backward memory
      from O(N_STEPS) to O(sqrt(N_STEPS)) with ~2x FLOPs overhead.
    - sigma clipped to [0.1, 10.0] before use (NaN guard).

    Args:
        S0      : (N_MAX, 7) initial state, precomputed device constant
        w, mu, sigma : RBF parameters -- traced by jax.grad
        gamma, dt, n_steps : Python scalars/int (static)

    Returns:
        trajectory : (n_steps+1, N_MAX, 7)
    """
    sigma_safe = jnp.clip(sigma, 0.1, 10.0)

    @jax_checkpoint
    def step(S, _):
        return rk4_step(S, w, mu, sigma_safe, gamma, dt), \
               rk4_step(S, w, mu, sigma_safe, gamma, dt)

    _, steps   = jax.lax.scan(step, S0, None, length=n_steps)
    trajectory = jnp.concatenate([S0[None], steps], axis=0)
    return trajectory


# ===========================================================================
# SECTION 9: Parameter initialization -- Data-proximal quasi-flat
# ===========================================================================
#
# Physical motivation (from Block I failure analysis):
#
#   X-attractor at (-8,-8) is ~17 units from initial particles in [0,7]^2.
#   Gaussian force at that distance:
#       F ~ exp(-17^2 / (2*2^2)) = exp(-18.1) ~ 10^{-8}   <<  machine eps
#   => gradient dL/d(mu_1) ~ 10^{-8} * (q-mu_1) ~ 0
#   => mu_1 never moves;  learning is stuck from epoch 0.
#
#   Solution: Add STEPPING STONES (k=2,3) at intermediate positions, and
#   distribute FREE RBFs (k=4..15) INSIDE [0,7]^2 so that they experience
#   O(1) forces from real particles and can pull them toward the attractors.
#
# Frozen (never updated):
#   k=0: O-attractor (8, 8, 0.88)   w=-2.0  sigma=2.0
#   k=1: X-attractor (-8,-8, 0.12)  w=-2.0  sigma=2.0
#
# Learnable (k=2..15, 14 RBFs):
#   k=2 : O stepping stone  (6, 6, 0.5)   w=-0.5  sigma=2.0
#   k=3 : X stepping stone  (-2,-2, 0.5)  w=-0.5  sigma=2.0
#   k=4..15 : 4x3 grid in [0.5,6.5]x[1.0,6.0]x{0.5}
#             w ~ alternating small values in [-0.2, +0.2]
#             sigma = 2.0

W_FROZEN     = jnp.array([-2.0, -2.0])
MU_FROZEN    = jnp.array([[ 8.0,  8.0,  0.88],
                            [-8.0, -8.0,  0.12]], dtype=jnp.float32)
SIGMA_FROZEN = jnp.array([2.0, 2.0])

# Stepping stones
_w_stones   = np.array([-0.5, -0.5], dtype=np.float32)
_mu_stones  = np.array([[ 6.0,  6.0, 0.5],
                         [-2.0, -2.0, 0.5]], dtype=np.float32)
_sig_stones = np.array([2.0, 2.0], dtype=np.float32)

# Free RBFs: 4x3 grid  (x in [0.5,2.5,4.5,6.5], y in [6.0,3.5,1.0])
_free_mu = np.array([
    [0.5, 6.0, 0.5], [2.5, 6.0, 0.5], [4.5, 6.0, 0.5], [6.5, 6.0, 0.5],
    [0.5, 3.5, 0.5], [2.5, 3.5, 0.5], [4.5, 3.5, 0.5], [6.5, 3.5, 0.5],
    [0.5, 1.0, 0.5], [2.5, 1.0, 0.5], [4.5, 1.0, 0.5], [6.5, 1.0, 0.5],
], dtype=np.float32)

_free_w = np.array([-0.20,  0.15, -0.10,  0.20,
                    -0.15,  0.10, -0.20,  0.15,
                    -0.10,  0.20, -0.15,  0.10], dtype=np.float32)
_free_sig = np.full(12, 2.0, dtype=np.float32)

# Concatenate learnable initial values
_w_learn_np   = np.concatenate([_w_stones,  _free_w],   axis=0)   # (14,)
_mu_learn_np  = np.concatenate([_mu_stones, _free_mu],  axis=0)   # (14,3)
_sig_learn_np = np.concatenate([_sig_stones, _free_sig], axis=0)  # (14,)

# Reparameterize sigma: sigma = softplus(sigma_raw) + 0.1
# => sigma_raw = softplus_inverse(sigma - 0.1)
# softplus_inverse(y) = log(exp(y) - 1)  for y > 0
_sig_adj = np.clip(_sig_learn_np - 0.1, 1e-3, None)
_sigraw_learn_np = np.log(np.expm1(_sig_adj)).astype(np.float32)

params_init = {
    'w'        : jnp.array(_w_learn_np),
    'mu'       : jnp.array(_mu_learn_np),
    'sigma_raw': jnp.array(_sigraw_learn_np),
}


# ===========================================================================
# SECTION 10: Parameter utilities
# ===========================================================================
def full_params(params):
    """
    Combine frozen (k=0,1) with learnable (k=2..15) into K=16 arrays.

    sigma reparameterization:
        sigma_k = softplus(sigma_raw_k) + 0.1
        minimum value = 0.1  (prevents division by zero in RBF gradient)

    Returns: w (16,), mu (16,3), sigma (16,)
    """
    sigma_learn = jax.nn.softplus(params['sigma_raw']) + 0.1
    w     = jnp.concatenate([W_FROZEN,     params['w']],     axis=0)
    mu    = jnp.concatenate([MU_FROZEN,    params['mu']],    axis=0)
    sigma = jnp.concatenate([SIGMA_FROZEN, sigma_learn],     axis=0)
    return w, mu, sigma


# ===========================================================================
# SECTION 11: Loss function
# ===========================================================================
def loss_fn(params):
    """
    L(theta) = ||CoM_O(T) - q*_O||^2  +  ||CoM_X(T) - q*_X||^2
             + lambda_p * (mean_i ||p_i^O(T)||^2 + mean_i ||p_i^X(T)||^2)

    Gradient flows via automatic differentiation through lax.scan.

    Returns: (total_loss, (loss_q, loss_p, com_O_T, com_X_T, traj_O, traj_X))
    """
    w, mu, sigma = full_params(params)

    # O-image forward
    traj_O  = simulate_diff(S0_O, w, mu, sigma)
    com_O_T = com_single(traj_O[-1, :, :3], MASK_O)

    # X-image forward
    traj_X  = simulate_diff(S0_X, w, mu, sigma)
    com_X_T = com_single(traj_X[-1, :, :3], MASK_X)

    # Position loss: L2 distance of final CoM to target attractor
    loss_q = (jnp.sum((com_O_T - Q_STAR_O) ** 2) +
              jnp.sum((com_X_T - Q_STAR_X) ** 2))

    # Momentum penalty: encourages particles to come to rest at T
    mO      = MASK_O.astype(jnp.float32)
    mX      = MASK_X.astype(jnp.float32)
    p_O     = traj_O[-1, :, 3:6]
    p_X     = traj_X[-1, :, 3:6]
    pnorm_O = jnp.sum(jnp.sum(p_O ** 2, axis=-1) * mO) / jnp.sum(mO)
    pnorm_X = jnp.sum(jnp.sum(p_X ** 2, axis=-1) * mX) / jnp.sum(mX)
    loss_p  = pnorm_O + pnorm_X

    total = loss_q + LAMBDA_P * loss_p
    return total, (loss_q, loss_p, com_O_T, com_X_T, traj_O, traj_X)


# ===========================================================================
# SECTION 12: Optimizer setup
# ===========================================================================
N_EPOCHS     = 1000
LOG_EVERY    = 10
SAVE_EVERY   = 100
WARMUP_STEPS = 50
PEAK_LR      = 1e-3
END_LR       = 1e-5

schedule = optax.warmup_cosine_decay_schedule(
    init_value   = 0.0,
    peak_value   = PEAK_LR,
    warmup_steps = WARMUP_STEPS,
    decay_steps  = N_EPOCHS,
    end_value    = END_LR,
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),     # prevents gradient explosion
    optax.adam(learning_rate=schedule),
)


@jit
def train_step(params, opt_state):
    """
    Single training step (JIT-compiled):
      1. Forward pass + loss
      2. Backward pass (grad through lax.scan via autodiff)
      3. Optimizer update (Adam + gradient clipping)
      4. Parameter update

    Note: optax.chain requires params for scale-by-adam,
    so optimizer.update receives all three arguments.
    """
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss, aux


# ===========================================================================
# SECTION 13: Verification figure (6-panel)
# ===========================================================================
def make_verification_figure(history, params_final,
                              traj_O_final, traj_X_final,
                              epoch_converged,
                              output_path="block2_verification.png"):
    """
    6-panel verification figure (Block I style):

    [1,1] Learned RBF potential landscape (z=0.5 slice)
    [1,2] Final-epoch particle trajectories (xy-projection)
    [1,3] Training loss curves (total, q-term, p-term) vs epoch
    [2,1] CoM migration in xy-plane during training
    [2,2] Convergence metrics eps_q, eps_p (final trajectory)
    [2,3] Block II verification summary text
    """
    w, mu, sigma = full_params(params_final)
    mu_np = np.array(mu)

    mO = np.array(MASK_O, dtype=bool)
    mX = np.array(MASK_X, dtype=bool)

    # Final classification
    pred_O, dOO, dOX, cfO = classify_traj(traj_O_final, MASK_O)
    pred_X, dXO, dXX, cfX = classify_traj(traj_X_final, MASK_X)

    # Convergence metrics over final-epoch trajectory
    t_arr = np.linspace(0.0, T_FINAL, N_STEPS + 1)
    com_O_traj = np.array(
        vmap(lambda qt: com_single(qt, MASK_O))(traj_O_final[:, :, :3]))
    com_X_traj = np.array(
        vmap(lambda qt: com_single(qt, MASK_X))(traj_X_final[:, :, :3]))

    eps_q_O = np.linalg.norm(com_O_traj - np.array(Q_STAR_O), axis=-1)
    eps_q_X = np.linalg.norm(com_X_traj - np.array(Q_STAR_X), axis=-1)
    eps_p_O = np.mean(
        np.linalg.norm(np.array(traj_O_final[:, mO, 3:6]), axis=-1), axis=-1)
    eps_p_X = np.mean(
        np.linalg.norm(np.array(traj_X_final[:, mX, 3:6]), axis=-1), axis=-1)

    fq_O = eps_q_O[-1];  fq_X = eps_q_X[-1]
    fp_O = eps_p_O[-1];  fp_X = eps_p_X[-1]

    cls_pass = (pred_O == 'O') and (pred_X == 'X')
    conv_q   = (fq_O < CONV_Q_THR) and (fq_X < CONV_Q_THR)
    conv_p   = (fp_O < CONV_P_THR) and (fp_X < CONV_P_THR)
    all_pass = cls_pass and conv_q and conv_p

    # ── Layout ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.38)
    a   = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]
    a11, a12, a13 = a[0]
    a21, a22, a23 = a[1]

    # ── [1,1] Learned RBF potential landscape ──────────────────────────────
    xy_r = np.linspace(-12, 12, 300)
    gx, gy = np.meshgrid(xy_r, xy_r)
    q_grid = jnp.array(np.stack(
        [gx.ravel(), gy.ravel(), np.full(gx.size, 0.5)],
        axis=1).astype(np.float32))
    Vg  = np.array(rbf_potential(q_grid, w, mu, sigma)).reshape(gx.shape)
    cf  = a11.contourf(gx, gy, Vg, levels=40, cmap='RdYlBu_r', alpha=0.85)
    a11.contour(gx, gy, Vg, levels=20, colors='k', linewidths=0.3, alpha=0.4)
    plt.colorbar(cf, ax=a11, shrink=0.85, label='V(q)')

    special = [
        (0, 'blue',    '*', 220, 'O-attractor (frozen)'),
        (1, 'red',     '*', 220, 'X-attractor (frozen)'),
        (2, 'cyan',    'D', 130, 'O step stone'),
        (3, 'magenta', 'D', 130, 'X step stone'),
    ]
    for k, col, mrk, sz, lab in special:
        a11.scatter(*mu_np[k, :2], s=sz, c=col, marker=mrk, zorder=6, label=lab)
    w_np = np.array(w)
    for k in range(4, 16):
        col = 'limegreen' if w_np[k] < 0 else 'tomato'
        lbl = 'free w<0' if k==4 else ('free w>0' if k==5 else '')
        a11.scatter(*mu_np[k, :2], s=35, c=col, marker='o', zorder=5, label=lbl)
    a11.scatter(*np.array(Q_STAR_O)[:2], s=300, c='blue', marker='P', zorder=7, label='q*_O')
    a11.scatter(*np.array(Q_STAR_X)[:2], s=300, c='red',  marker='P', zorder=7, label='q*_X')
    a11.set(xlim=(-12,12), ylim=(-12,12), xlabel='x', ylabel='y',
            title='Learned RBF Potential (z=0.5 slice)')
    a11.legend(fontsize=6.2, loc='upper right', framealpha=0.8)

    # ── [1,2] Final particle trajectories ──────────────────────────────────
    tOnp = np.array(traj_O_final[:, mO, :])
    tXnp = np.array(traj_X_final[:, mX, :])
    dec  = max(1, N_STEPS // 50)
    for i in range(tOnp.shape[1]):
        a12.plot(tOnp[::dec, i, 0], tOnp[::dec, i, 1], 'b-', alpha=0.18, lw=0.6)
    for i in range(tXnp.shape[1]):
        a12.plot(tXnp[::dec, i, 0], tXnp[::dec, i, 1], 'r-', alpha=0.18, lw=0.6)
    a12.scatter(tOnp[ 0,:,0], tOnp[ 0,:,1], c='blue', s=14, marker='o', zorder=4)
    a12.scatter(tOnp[-1,:,0], tOnp[-1,:,1], c='blue', s=28, marker='x', zorder=5)
    a12.scatter(tXnp[ 0,:,0], tXnp[ 0,:,1], c='red',  s=14, marker='o', zorder=4)
    a12.scatter(tXnp[-1,:,0], tXnp[-1,:,1], c='red',  s=28, marker='x', zorder=5)
    a12.plot(com_O_traj[::dec, 0], com_O_traj[::dec, 1], 'b-', lw=2.5, label='CoM O')
    a12.plot(com_X_traj[::dec, 0], com_X_traj[::dec, 1], 'r-', lw=2.5, label='CoM X')
    a12.scatter(*np.array(Q_STAR_O)[:2], s=300, c='blue', marker='*', zorder=6, label='q*_O')
    a12.scatter(*np.array(Q_STAR_X)[:2], s=300, c='red',  marker='*', zorder=6, label='q*_X')
    a12.set(xlabel='q_x', ylabel='q_y',
            title='Final-Epoch Particle Trajectories (xy)')
    a12.legend(fontsize=8)

    # ── [1,3] Loss curves ──────────────────────────────────────────────────
    ep  = history['epoch']
    a13.semilogy(ep, history['loss_total'], 'k-',  lw=2,   label='Total loss')
    a13.semilogy(ep, history['loss_q'],     'b--', lw=1.5, label='Loss_q (position)')
    a13.semilogy(ep, history['loss_p'],     'r:',  lw=1.5, label='Loss_p (momentum)')
    if epoch_converged is not None:
        a13.axvline(epoch_converged, color='green', ls='--', lw=1.5,
                    label=f'Converged @ {epoch_converged}')
    a13.set(xlabel='Epoch', ylabel='Loss (log scale)', title='Training Loss Curves')
    a13.legend(fontsize=8)

    # ── [2,1] CoM migration during training ────────────────────────────────
    com_O_hist = np.array(history['com_O_final'])  # (n_log, 3)
    com_X_hist = np.array(history['com_X_final'])
    ep_arr     = np.array(ep, dtype=float)
    for i in range(len(ep) - 1):
        a21.plot(com_O_hist[i:i+2, 0], com_O_hist[i:i+2, 1], 'b-', alpha=0.5, lw=0.8)
        a21.plot(com_X_hist[i:i+2, 0], com_X_hist[i:i+2, 1], 'r-', alpha=0.5, lw=0.8)
    sc1 = a21.scatter(com_O_hist[:, 0], com_O_hist[:, 1],
                       c=ep_arr, cmap='Blues', s=18, zorder=4)
    a21.scatter(com_X_hist[:, 0], com_X_hist[:, 1],
                c=ep_arr, cmap='Reds',  s=18, zorder=4)
    plt.colorbar(sc1, ax=a21, label='Epoch')
    a21.scatter(*np.array(Q_STAR_O)[:2], s=300, c='blue', marker='*', zorder=6)
    a21.scatter(*np.array(Q_STAR_X)[:2], s=300, c='red',  marker='*', zorder=6)
    a21.set(xlabel='q_x', ylabel='q_y', title='CoM Migration During Training')

    # ── [2,2] Convergence metrics (final epoch) ─────────────────────────────
    a22.semilogy(t_arr, eps_q_O+1e-6, 'b-',  lw=2,   label=r'$\varepsilon_q$ O')
    a22.semilogy(t_arr, eps_q_X+1e-6, 'r-',  lw=2,   label=r'$\varepsilon_q$ X')
    a22.semilogy(t_arr, eps_p_O+1e-6, 'b--', lw=1.5, label=r'$\varepsilon_p$ O')
    a22.semilogy(t_arr, eps_p_X+1e-6, 'r--', lw=1.5, label=r'$\varepsilon_p$ X')
    a22.axhline(CONV_Q_THR, color='gray', ls=':',  lw=1, label=f'eps_q thr={CONV_Q_THR}')
    a22.axhline(CONV_P_THR, color='gray', ls='-.', lw=1, label=f'eps_p thr={CONV_P_THR}')
    a22.set(xlabel='t', ylabel='Error (log scale)',
            title=r'Convergence: $\varepsilon_q$ (solid), $\varepsilon_p$ (dashed)')
    a22.legend(fontsize=8, ncol=2)
    pass_str  = 'PASS' if (conv_q and conv_p) else 'FAIL'
    pass_col  = 'green' if (conv_q and conv_p) else 'red'
    a22.text(0.97, 0.97, pass_str, transform=a22.transAxes,
             ha='right', va='top', fontsize=13, fontweight='bold', color=pass_col)

    # ── [2,3] Summary ───────────────────────────────────────────────────────
    bg = '#d4edda' if all_pass else '#fff3cd'
    a23.set_facecolor(bg)
    a23.axis('off')
    a23.set(xlim=(0, 1), ylim=(0, 1))
    final_ep = ep[-1] if ep else N_EPOCHS
    lns = [
        "BLOCK II VERIFICATION SUMMARY",
        "=" * 34,
        "",
        f"Epochs trained  : {final_ep}",
        f"Converged at    : {epoch_converged if epoch_converged is not None else 'N/A'}",
        "",
        "O-image Final CoM:",
        f"  q=({float(cfO[0]):.3f},{float(cfO[1]):.3f},{float(cfO[2]):.3f})",
        f"  dist(q*_O)={dOO:.4f}  dist(q*_X)={dOX:.4f}",
        f"  pred={pred_O}  true=O  ({'OK' if pred_O=='O' else 'FAIL'})",
        "",
        "X-image Final CoM:",
        f"  q=({float(cfX[0]):.3f},{float(cfX[1]):.3f},{float(cfX[2]):.3f})",
        f"  dist(q*_O)={dXO:.4f}  dist(q*_X)={dXX:.4f}",
        f"  pred={pred_X}  true=X  ({'OK' if pred_X=='X' else 'FAIL'})",
        "",
        "Verification (final epoch):",
        f"  eps_q O={fq_O:.4f}  thr={CONV_Q_THR} -> {'PASS' if fq_O<CONV_Q_THR else 'FAIL'}",
        f"  eps_q X={fq_X:.4f}  thr={CONV_Q_THR} -> {'PASS' if fq_X<CONV_Q_THR else 'FAIL'}",
        f"  eps_p O={fp_O:.4f}  thr={CONV_P_THR} -> {'PASS' if fp_O<CONV_P_THR else 'FAIL'}",
        f"  eps_p X={fp_X:.4f}  thr={CONV_P_THR} -> {'PASS' if fp_X<CONV_P_THR else 'FAIL'}",
        f"  Classification : {'PASS' if cls_pass else 'FAIL'}",
        "",
        f"  OVERALL: {'*** ALL PASS ***' if all_pass else '--- PARTIAL ---'}",
        "",
        f"  gamma={GAMMA}  T={T_FINAL}  dt={DT}  K=16",
        f"  lambda_p={LAMBDA_P}  peak_lr={PEAK_LR}",
    ]
    a23.text(0.05, 0.97, "\n".join(lns), transform=a23.transAxes,
             fontsize=7.4, fontfamily='monospace', va='top')
    title_col = 'darkgreen' if all_pass else 'darkorange'
    a23.set_title('Block II Verification Summary',
                  fontweight='bold', color=title_col)

    fig.suptitle(
        "Contact Hamiltonian Fluid NN -- Block II: Learning Loop\n"
        f"gamma={GAMMA}  T={T_FINAL}  dt={DT}  K=16 RBF  "
        f"[JAX BPTT+checkpoint / CUDA]",
        fontsize=10.5, y=1.01)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  [Figure] Saved -> {output_path}")
    plt.close(fig)
    return all_pass


# ===========================================================================
# SECTION 14: Training loop
# ===========================================================================
def run_training():
    """
    Main training loop.

    Logging every LOG_EVERY epochs:
        loss total, loss_q, loss_p, CoM positions, predictions, eps_q

    Checkpointing every SAVE_EVERY epochs:
        numpy .npy files with learnable parameter arrays

    Early stopping:
        Triggered when pred_O=='O' AND pred_X=='X'
        AND eps_q_O < CONV_Q_THR AND eps_q_X < CONV_Q_THR
    """
    print("\n" + "=" * 62)
    print("CONTACT HAMILTONIAN FLUID NN -- BLOCK II (JAX/CUDA)")
    print("=" * 62)
    print(f"  Epochs={N_EPOCHS}  log_every={LOG_EVERY}  save_every={SAVE_EVERY}")
    print(f"  peak_lr={PEAK_LR}  end_lr={END_LR}  warmup={WARMUP_STEPS}")
    print(f"  lambda_p={LAMBDA_P}  conv_q_thr={CONV_Q_THR}  conv_p_thr={CONV_P_THR}")
    print(f"  T={T_FINAL}  dt={DT}  N_steps={N_STEPS}  N_max={N_MAX}")
    print(f"  Frozen: k=0 (O-attractor), k=1 (X-attractor)")
    print(f"  Learnable: k=2..15  (14 RBFs including 2 stepping stones)")
    print(f"  Gradient: full BPTT through lax.scan + jax.checkpoint")
    print("=" * 62 + "\n")

    params    = params_init
    opt_state = optimizer.init(params)

    history = {
        'epoch'      : [],
        'loss_total' : [],
        'loss_q'     : [],
        'loss_p'     : [],
        'com_O_final': [],
        'com_X_final': [],
        'pred_O'     : [],
        'pred_X'     : [],
    }

    epoch_converged = None
    t_start         = time.time()

    for epoch in range(N_EPOCHS):
        t0 = time.time()
        params, opt_state, loss_val, aux = train_step(params, opt_state)
        loss_q_val, loss_p_val, com_O_T, com_X_T, traj_O, traj_X = aux

        if epoch % LOG_EVERY == 0:
            pred_O_s, dOO, _, cfO = classify_traj(traj_O, MASK_O)
            pred_X_s, _,   dXX, cfX = classify_traj(traj_X, MASK_X)
            com_O_np = np.array(com_O_T)
            com_X_np = np.array(com_X_T)
            eps_q_O  = float(np.linalg.norm(com_O_np - np.array(Q_STAR_O)))
            eps_q_X  = float(np.linalg.norm(com_X_np - np.array(Q_STAR_X)))

            history['epoch'].append(epoch)
            history['loss_total'].append(float(loss_val))
            history['loss_q'].append(float(loss_q_val))
            history['loss_p'].append(float(loss_p_val))
            history['com_O_final'].append(com_O_np.tolist())
            history['com_X_final'].append(com_X_np.tolist())
            history['pred_O'].append(pred_O_s)
            history['pred_X'].append(pred_X_s)

            elapsed = time.time() - t0
            print(f"Ep {epoch:4d} | "
                  f"L={float(loss_val):.5f} "
                  f"(q={float(loss_q_val):.4f} p={float(loss_p_val):.4f}) | "
                  f"CoM_O=({com_O_np[0]:.2f},{com_O_np[1]:.2f}) "
                  f"CoM_X=({com_X_np[0]:.2f},{com_X_np[1]:.2f}) | "
                  f"pred={pred_O_s},{pred_X_s} | "
                  f"eq=({eps_q_O:.3f},{eps_q_X:.3f}) | "
                  f"{elapsed:.2f}s")

            # Early stopping check
            if epoch > 0:
                if (pred_O_s == 'O' and pred_X_s == 'X'
                        and eps_q_O < CONV_Q_THR and eps_q_X < CONV_Q_THR):
                    epoch_converged = epoch
                    print(f"\n  *** Converged at epoch {epoch}! ***")
                    print(f"  eps_q: O={eps_q_O:.4f}  X={eps_q_X:.4f}  "
                          f"(threshold={CONV_Q_THR})")
                    break

        if epoch % SAVE_EVERY == 0 and epoch > 0:
            save_path = f"block2_params_ep{epoch:04d}.npy"
            np.save(save_path, {
                'w'        : np.array(params['w']),
                'mu'       : np.array(params['mu']),
                'sigma_raw': np.array(params['sigma_raw']),
                'epoch'    : epoch,
                'loss'     : float(loss_val),
            }, allow_pickle=True)
            print(f"  [Checkpoint] -> {save_path}")

    t_total = time.time() - t_start
    print(f"\n[Training done] {t_total:.1f}s total")

    # ── Final forward pass (detached, for analysis) ───────────────────────
    print("Running final verification forward pass...")
    w_f, mu_f, sig_f = full_params(params)
    traj_O_fin = simulate_diff(S0_O, w_f, mu_f, sig_f)
    traj_X_fin = simulate_diff(S0_X, w_f, mu_f, sig_f)

    # Save trained parameters
    trained_path = "block2_trained_params.npy"
    np.save(trained_path, {
        'w'         : np.array(params['w']),
        'mu'        : np.array(params['mu']),
        'sigma_raw' : np.array(params['sigma_raw']),
        'w_full'    : np.array(w_f),
        'mu_full'   : np.array(mu_f),
        'sigma_full': np.array(sig_f),
        'epoch_converged': epoch_converged,
        'history'   : history,
    }, allow_pickle=True)
    print(f"  [Params] Saved -> {trained_path}")

    # Final report
    pred_O, dOO, dOX, cfO = classify_traj(traj_O_fin, MASK_O)
    pred_X, dXO, dXX, cfX = classify_traj(traj_X_fin, MASK_X)
    fq_O = float(np.linalg.norm(np.array(cfO) - np.array(Q_STAR_O)))
    fq_X = float(np.linalg.norm(np.array(cfX) - np.array(Q_STAR_X)))
    S = "=" * 62
    print(f"\n{S}\nBLOCK II FINAL REPORT\n{S}")
    print(f"  O: CoM=({float(cfO[0]):.4f},{float(cfO[1]):.4f},{float(cfO[2]):.4f})"
          f"  dO={dOO:.4f}  dX={dOX:.4f}  pred={pred_O}")
    print(f"  X: CoM=({float(cfX[0]):.4f},{float(cfX[1]):.4f},{float(cfX[2]):.4f})"
          f"  dO={dXO:.4f}  dX={dXX:.4f}  pred={pred_X}")
    cls_ok = (pred_O == 'O') and (pred_X == 'X')
    print(f"  CLASS : {'PASS' if cls_ok else 'FAIL'}")
    print(f"  eps_q : O={fq_O:.4f}  X={fq_X:.4f}  "
          f"-> {'PASS' if (fq_O<CONV_Q_THR and fq_X<CONV_Q_THR) else 'FAIL'}")
    print(S)

    # Build verification figure
    print("\nBuilding verification figure...")
    all_pass = make_verification_figure(
        history, params, traj_O_fin, traj_X_fin,
        epoch_converged,
        output_path="block2_verification.png")

    print(f"\n  OVERALL: {'*** ALL PASS ***' if all_pass else '--- PARTIAL (Block II needs more epochs or tuning) ---'}")
    print("  Figure : block2_verification.png")
    print("  Params : block2_trained_params.npy")
    print(S)

    return params, history, all_pass


# ===========================================================================
# SECTION 15: Entry point
# ===========================================================================
if __name__ == "__main__":
    # Shape and positivity sanity check
    w0, mu0, sig0 = full_params(params_init)
    assert w0.shape   == (16,),   f"w shape: {w0.shape}"
    assert mu0.shape  == (16,3),  f"mu shape: {mu0.shape}"
    assert sig0.shape == (16,),   f"sigma shape: {sig0.shape}"
    assert float(jnp.min(sig0)) > 0.0, "sigma must be positive"
    print(f"[Init check] w:{w0.shape} mu:{mu0.shape} sigma:{sig0.shape}  "
          f"sigma_min={float(jnp.min(sig0)):.4f}  sigma_max={float(jnp.max(sig0)):.4f}")
    print(f"  O particles: {int(MASK_O.sum())}  X particles: {int(MASK_X.sum())}")

    # Print initial parameter table (k=0..15)
    print("\nInitial parameter overview:")
    print(f"  {'k':>3}  {'role':<20}  {'w':>7}  {'mu_x':>6}  {'mu_y':>6}  {'sigma':>6}  {'trainable'}")
    roles = ['O-attractor(frozen)', 'X-attractor(frozen)',
             'O step stone', 'X step stone'] + [f'free k={k}' for k in range(4, 16)]
    w0np = np.array(w0); mu0np = np.array(mu0); sig0np = np.array(sig0)
    for k in range(16):
        tr = 'NO' if k < 2 else 'YES'
        print(f"  {k:>3}  {roles[k]:<20}  {w0np[k]:>7.3f}  "
              f"{mu0np[k,0]:>6.2f}  {mu0np[k,1]:>6.2f}  {sig0np[k]:>6.3f}  {tr}")

    # JIT warmup (triggers full XLA compilation)
    print("\n[JIT warmup] Compiling train_step (first run: 30-120s)...")
    opt_state_warmup = optimizer.init(params_init)
    t_c = time.time()
    _p, _o, _l, _a = train_step(params_init, opt_state_warmup)
    _l.block_until_ready()
    print(f"  Compilation done in {time.time()-t_c:.1f}s")
    print(f"  Initial loss = {float(_l):.6f}")
    _lq, _lp, _co, _cx, _, _ = _a
    print(f"  Initial CoM_O = ({float(_co[0]):.3f}, {float(_co[1]):.3f})")
    print(f"  Initial CoM_X = ({float(_cx[0]):.3f}, {float(_cx[1]):.3f})")

    # Run training
    trained_params, hist, passed = run_training()