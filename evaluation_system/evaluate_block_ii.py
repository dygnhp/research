"""
=============================================================================
Contact Hamiltonian Fluid Neural Network  --  Evaluation System: Block II
=============================================================================
Evaluates Block II trained parameters using the shared evaluation module.
This is the Block II-specific evaluation runner.

Evaluation pipeline:
  1. Load trained params from Block II (or fallback training)
  2. Baseline classification (canonical O/X)
  3. Dataset-variant accuracy (20 generated O/X variants)
  4. Noise robustness sweep (pixel-flip)
  5. Shift robustness sweep (translation)
  6. Novel pattern evaluation (T, L, +, Square)
  7. Ablation study (component removal)
  8. Gamma sensitivity analysis
  9. 8-panel paper-ready figure

Backend:   JAX (CUDA) -- JIT-compiled, same as Block II
=============================================================================
"""

from pathlib import Path
import os
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"

import jax
import jax.numpy as jnp
from jax import jit
from jax import checkpoint as jax_checkpoint
from functools import partial
import numpy as np
import optax
import time

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_BLOCK_II_DIR = _ROOT / "block_ii"

sys.path.insert(0, str(_ROOT))
from data_generator import (generate_dataset, O_CANONICAL, X_CANONICAL)
from evaluation import run_standard_eval

print("=" * 62)
print("EVALUATION SYSTEM: Block II Evaluation")
print("=" * 62)
print(f"JAX version    : {jax.__version__}")
print(f"Devices        : {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")
print("=" * 62)


# ===========================================================================
# SECTION 1: Constants (identical to Block II)
# ===========================================================================
GAMMA     = 1.5
T_FINAL   = 10.0
DT        = 0.05
N_STEPS   = int(T_FINAL / DT)
N_MAX     = 64
TAU       = 0.5
LAMBDA_P  = 0.1
D         = 3
K         = 16

Q_STAR_O  = jnp.array([ 8.0,  8.0, 0.0])
Q_STAR_X  = jnp.array([-8.0, -8.0, 0.0])
CONV_Q_THR = 2.0
CONV_P_THR = 0.5

O_IMAGE = O_CANONICAL
X_IMAGE = X_CANONICAL


# ===========================================================================
# SECTION 2: Novel Test Patterns (unseen by training)
# ===========================================================================
T_IMAGE = np.array([
    [1,1,1,1,1,1,1,1],
    [0,0,0,1,1,0,0,0],
    [0,0,0,1,1,0,0,0],
    [0,0,0,1,1,0,0,0],
    [0,0,0,1,1,0,0,0],
    [0,0,0,1,1,0,0,0],
    [0,0,0,1,1,0,0,0],
    [0,0,0,1,1,0,0,0],
], dtype=float)

L_IMAGE = np.array([
    [1,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,0,0],
    [0,0,0,0,0,0,0,0],
], dtype=float)

PLUS_IMAGE = np.array([
    [0,0,0,1,1,0,0,0],
    [0,0,0,1,1,0,0,0],
    [0,0,0,1,1,0,0,0],
    [1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1],
    [0,0,0,1,1,0,0,0],
    [0,0,0,1,1,0,0,0],
    [0,0,0,1,1,0,0,0],
], dtype=float)

SQUARE_IMAGE = np.array([
    [1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1],
], dtype=float)

NOVEL_IMAGES = {
    'T': T_IMAGE, 'L': L_IMAGE, '+': PLUS_IMAGE, 'Square': SQUARE_IMAGE,
}


# ===========================================================================
# SECTION 3: Preprocessing (3D lifting, identical to Block II)
# ===========================================================================
def preprocess(image, tau=TAU, n_max=N_MAX, beta=1.0):
    """Convert 8x8 binary image to JAX state arrays with contextual 3D lifting."""
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
        score = d_axis - abs(d_diag_signed)
        return 1.0 / (1.0 + np.exp(-b * score))

    q_list = []
    for r in range(rows):
        for c in range(cols):
            if image[r, c] > tau:
                q_list.append([float(c), float(rows - 1 - r),
                               compute_z_init(r, c, beta)])

    n_real = len(q_list)
    assert n_real > 0, "No real particles"
    assert n_real <= n_max, f"Too many particles: {n_real} > {n_max}"

    q0_np   = np.zeros((n_max, D), dtype=np.float32)
    p0_np   = np.zeros((n_max, D), dtype=np.float32)
    z0_np   = np.zeros((n_max,),   dtype=np.float32)
    mask_np = np.zeros((n_max,),   dtype=bool)

    for i, pos in enumerate(q_list):
        q0_np[i]   = pos
        mask_np[i] = True

    return (jnp.array(q0_np), jnp.array(p0_np),
            jnp.array(z0_np), jnp.array(mask_np))


def make_S0(image):
    """Build (S0, mask) for the evaluation module interface."""
    q0, p0, z0, mask = preprocess(image)
    S0 = jnp.concatenate([q0, p0, z0[:, None]], axis=-1)
    return S0, mask


# Precompute canonical masks
_, _, _, MASK_O = preprocess(O_IMAGE)
_, _, _, MASK_X = preprocess(X_IMAGE)
S0_O, _ = make_S0(O_IMAGE)
S0_X, _ = make_S0(X_IMAGE)


# ===========================================================================
# SECTION 4: RBF Potential and Gradient
# ===========================================================================
def rbf_potential(q, w, mu, sigma):
    diff    = q[:, None, :] - mu[None, :, :]
    sq_dist = jnp.sum(diff ** 2, axis=-1)
    gauss   = jnp.exp(-sq_dist / (2.0 * sigma ** 2))
    return jnp.sum(w * gauss, axis=-1)


def rbf_gradient(q, w, mu, sigma):
    diff    = q[:, None, :] - mu[None, :, :]
    sq_dist = jnp.sum(diff ** 2, axis=-1)
    gauss   = jnp.exp(-sq_dist / (2.0 * sigma ** 2))
    factor  = w * gauss / (sigma ** 2)
    return jnp.sum(-factor[:, :, None] * diff, axis=1)

_rbf_potential = rbf_potential
_rbf_gradient  = rbf_gradient


# ===========================================================================
# SECTION 5: Contact Hamiltonian RHS + RK4
# ===========================================================================
def contact_rhs(S, w, mu, sigma, gamma):
    q    = S[:, :D]
    p    = S[:, D:2*D]
    V    = _rbf_potential(q, w, mu, sigma)
    gV   = _rbf_gradient(q, w, mu, sigma)
    p_sq = jnp.sum(p ** 2, axis=-1)
    H_i  = p_sq / 2.0 + V
    dq_dt = p
    dp_dt = -gV - gamma * p
    dz_dt = p_sq - H_i
    return jnp.concatenate([dq_dt, dp_dt, dz_dt[:, None]], axis=-1)


@partial(jit, static_argnums=(4, 5))
def rk4_step(S, w, mu, sigma, gamma, dt):
    f  = partial(contact_rhs, w=w, mu=mu, sigma=sigma, gamma=gamma)
    k1 = f(S)
    k2 = f(S + 0.5 * dt * k1)
    k3 = f(S + 0.5 * dt * k2)
    k4 = f(S +       dt * k3)
    return S + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ===========================================================================
# SECTION 6: Simulators
# ===========================================================================
def simulate_eval(S0, w, mu, sigma, gamma=GAMMA, dt=DT, n_steps=N_STEPS):
    """Forward-only simulation (no gradient tracking)."""
    sigma_safe = jnp.clip(sigma, 0.1, 10.0)
    def step(S, _):
        S_next = rk4_step(S, w, mu, sigma_safe, gamma, dt)
        return S_next, S_next
    _, steps   = jax.lax.scan(step, S0, None, length=n_steps)
    return jnp.concatenate([S0[None], steps], axis=0)


def simulate_diff(S0, w, mu, sigma, gamma=GAMMA, dt=DT, n_steps=N_STEPS):
    """Differentiable simulation (with checkpoint for BPTT)."""
    sigma_safe = jnp.clip(sigma, 0.1, 10.0)
    @jax_checkpoint
    def step(S, _):
        S_next = rk4_step(S, w, mu, sigma_safe, gamma, dt)
        return S_next, S_next
    _, steps   = jax.lax.scan(step, S0, None, length=n_steps)
    return jnp.concatenate([S0[None], steps], axis=0)


# ===========================================================================
# SECTION 7: Parameter Init + Loading
# ===========================================================================
W_FROZEN     = jnp.array([-2.0, -2.0])
MU_FROZEN    = jnp.array([[ 8.0,  8.0,  0.88],
                            [-8.0, -8.0,  0.12]], dtype=jnp.float32)
SIGMA_FROZEN = jnp.array([2.0, 2.0])

_w_stones   = np.array([-1.0, -1.0], dtype=np.float32)
_mu_stones  = np.array([[ 6.0,  6.0, 0.88],
                         [ 0.0,  0.0, 0.12]], dtype=np.float32)
_sig_stones = np.array([3.0, 3.0], dtype=np.float32)

_free_mu = np.array([
    [0.5, 6.0, 0.5], [2.5, 6.0, 0.5], [4.5, 6.0, 0.5], [6.5, 6.0, 0.5],
    [0.5, 3.5, 0.5], [2.5, 3.5, 0.5], [4.5, 3.5, 0.5], [6.5, 3.5, 0.5],
    [0.5, 1.0, 0.5], [2.5, 1.0, 0.5], [4.5, 1.0, 0.5], [6.5, 1.0, 0.5],
], dtype=np.float32)
_free_w = np.array([-0.20,  0.15, -0.10,  0.20,
                    -0.15,  0.10, -0.20,  0.15,
                    -0.10,  0.20, -0.15,  0.10], dtype=np.float32)
_free_sig = np.full(12, 2.0, dtype=np.float32)

_w_learn_np   = np.concatenate([_w_stones, _free_w])
_mu_learn_np  = np.concatenate([_mu_stones, _free_mu])
_sig_learn_np = np.concatenate([_sig_stones, _free_sig])
_sig_adj = np.clip(_sig_learn_np - 0.1, 1e-3, None)
_sigraw_learn_np = np.log(np.expm1(_sig_adj)).astype(np.float32)

PARAMS_INIT = {
    'w'        : jnp.array(_w_learn_np),
    'mu'       : jnp.array(_mu_learn_np),
    'sigma_raw': jnp.array(_sigraw_learn_np),
}


def full_params(params):
    """Combine frozen (k=0,1) with learnable (k=2..K-1)."""
    sigma_learn = jax.nn.softplus(params['sigma_raw']) + 0.1
    w     = jnp.concatenate([W_FROZEN,     params['w']])
    mu    = jnp.concatenate([MU_FROZEN,    params['mu']])
    sigma = jnp.concatenate([SIGMA_FROZEN, sigma_learn])
    return w, mu, sigma


def load_trained_params(path=None):
    """Load trained params from Block II."""
    if path is None:
        path = _BLOCK_II_DIR / "block2_trained_params.npy"
    path = Path(path)
    if path.exists():
        data = np.load(str(path), allow_pickle=True).item()
        params = {
            'w'        : jnp.array(data['w']),
            'mu'       : jnp.array(data['mu']),
            'sigma_raw': jnp.array(data['sigma_raw']),
        }
        epoch = data.get('epoch_converged', 'unknown')
        print(f"  [Loaded] Block II params from {path} (epoch={epoch})")
        return params, True
    else:
        print(f"  [WARN] Not found: {path} -- using initial params")
        return PARAMS_INIT, False


# ===========================================================================
# SECTION 8: Fallback Training
# ===========================================================================
def fallback_training(n_epochs=200, peak_lr=2e-3):
    """Lightweight training if Block II params unavailable."""
    from evaluation import com_single, classify_traj
    print(f"\n  Fallback training: {n_epochs} epochs, lr={peak_lr}")

    def loss_fn(params):
        w, mu, sigma = full_params(params)
        traj_O  = simulate_diff(S0_O, w, mu, sigma)
        com_O_T = com_single(traj_O[-1, :, :D], MASK_O)
        traj_X  = simulate_diff(S0_X, w, mu, sigma)
        com_X_T = com_single(traj_X[-1, :, :D], MASK_X)
        loss_q = (jnp.sum((com_O_T - Q_STAR_O) ** 2) +
                  jnp.sum((com_X_T - Q_STAR_X) ** 2))
        mO = MASK_O.astype(jnp.float32)
        mX = MASK_X.astype(jnp.float32)
        pnorm_O = jnp.sum(jnp.sum(traj_O[-1, :, D:2*D] ** 2, axis=-1) * mO) / jnp.sum(mO)
        pnorm_X = jnp.sum(jnp.sum(traj_X[-1, :, D:2*D] ** 2, axis=-1) * mX) / jnp.sum(mX)
        return loss_q + LAMBDA_P * (pnorm_O + pnorm_X)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=peak_lr,
        warmup_steps=20, decay_steps=n_epochs, end_value=1e-5)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=schedule))

    params = PARAMS_INIT
    opt_state = opt.init(params)

    @jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt, loss

    for epoch in range(n_epochs):
        params, opt_state, loss = step(params, opt_state)
        if epoch % 50 == 0:
            print(f"    Ep {epoch:4d} | loss={float(loss):.4f}")

    print(f"    Final loss = {float(loss):.4f}")
    return params


# ===========================================================================
# SECTION 9: Main Evaluation Pipeline
# ===========================================================================
def run_evaluation():
    """Run Block II evaluation using shared evaluation module."""
    print("\n" + "=" * 62)
    print("EVALUATION SYSTEM: BLOCK II EVALUATION")
    print("=" * 62)

    # Load params
    print("\n[Step 1] Loading trained parameters...")
    params, loaded = load_trained_params()
    if not loaded:
        print("[Step 1b] Running fallback training...")
        params = fallback_training()

    # Dataset variant generator
    def dataset_fn():
        return generate_dataset(n_per_class=20, seed=99)

    # Run standard evaluation
    _, all_pass, results = run_standard_eval(
        params=params,
        full_params_fn=full_params,
        rbf_potential_fn=rbf_potential,
        preprocess_fn=make_S0,
        simulate_fn=simulate_eval,
        O_image=O_IMAGE,
        X_image=X_IMAGE,
        qO=Q_STAR_O,
        qX=Q_STAR_X,
        mask_O=MASK_O,
        mask_X=MASK_X,
        novel_images=NOVEL_IMAGES,
        dataset_fn=dataset_fn,
        title_prefix="Block III: Block II Eval",
        output_path=str(_HERE / "block3_verification.png"),
        gamma_default=GAMMA,
        T_final=T_FINAL,
        dt=DT,
        K=K,
        N_steps=N_STEPS,
    )

    print(f"\n  OVERALL: {'*** ALL PASS ***' if all_pass else '--- PARTIAL ---'}")
    return params, all_pass, results


# ===========================================================================
# SECTION 10: Entry Point
# ===========================================================================
if __name__ == "__main__":
    run_evaluation()
