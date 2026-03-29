"""
=============================================================================
Contact Hamiltonian Fluid Neural Network  --  Block III: Generalization &
                                               Robustness Evaluation
=============================================================================
Physics:   Same contact Hamiltonian dynamics as Block I/II
Goal:      Evaluate the learned RBF potential (from Block II) on:
             (A) Original O/X images  (sanity baseline)
             (B) Noisy variants       (pixel-flip noise sweep)
             (C) Shifted variants     (translation robustness)
             (D) Novel patterns       (T, L, +, square -- unseen classes)
           Then perform ablation studies:
             (E) Remove stepping stones (k=2,3)
             (F) Remove free RBFs (k=4..15)
             (G) Vary damping gamma
             (H) Vary integration time T
           Generate paper-ready figures and a quantitative summary.

Dependency: Trained parameters from Block II (block2_trained_params.npy).
            If not found, runs a lightweight training loop to produce them.

Backend:   JAX (CUDA) -- JIT-compiled, same as Block II
Target HW: Windows 11 + NVIDIA RTX 4060 via WSL2 / PyCharm+WSL

Install:
    pip install --upgrade "jax[cuda12]" optax matplotlib scipy
=============================================================================
"""

from pathlib import Path
import os

# JAX memory control -- must be set before importing jax
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

_HERE = Path(__file__).resolve().parent
_BLOCK_II_DIR = _HERE.parent / "block_ii"

print("=" * 62)
print("BLOCK III: Generalization & Robustness Evaluation")
print("=" * 62)
print(f"JAX version    : {jax.__version__}")
print(f"Devices        : {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")
print(f"Output dir     : {_HERE}")
print(f"Block II dir   : {_BLOCK_II_DIR}")
print("=" * 62)


# ===========================================================================
# SECTION 1: Global Constants  (identical to Block I/II)
# ===========================================================================
GAMMA     = 1.5
T_FINAL   = 10.0
DT        = 0.05
N_STEPS   = int(T_FINAL / DT)      # 200
N_MAX     = 64
TAU       = 0.5
LAMBDA_P  = 0.1

Q_STAR_O  = jnp.array([ 8.0,  8.0, 0.0])
Q_STAR_X  = jnp.array([-8.0, -8.0, 0.0])

CONV_Q_THR = 2.0
CONV_P_THR = 0.5


# ===========================================================================
# SECTION 2: Test Images  (original O/X from Block I/II)
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
# SECTION 3: Novel Test Patterns  (unseen by training)
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
# SECTION 4: Preprocessing  (identical to Block II)
# ===========================================================================
def preprocess(image, tau=TAU, n_max=N_MAX, beta=1.0):
    """
    Convert 8x8 binary image to JAX state arrays with contextual 3D lifting.
    Coordinate convention: row r, col c  -->  x = c,  y = 7 - r
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
        score = d_axis - abs(d_diag_signed)
        return 1.0 / (1.0 + np.exp(-b * score))

    q_list = []
    for r in range(rows):
        for c in range(cols):
            if image[r, c] > tau:
                q_list.append([float(c), float(rows - 1 - r),
                               compute_z_init(r, c, beta)])

    n_real = len(q_list)
    assert n_real > 0,      "No real particles -- check tau"
    assert n_real <= n_max,  f"Too many particles: {n_real} > {n_max}"

    q0_np   = np.zeros((n_max, 3), dtype=np.float32)
    p0_np   = np.zeros((n_max, 3), dtype=np.float32)
    z0_np   = np.zeros((n_max,),   dtype=np.float32)
    mask_np = np.zeros((n_max,),   dtype=bool)

    for i, pos in enumerate(q_list):
        q0_np[i]   = pos
        mask_np[i] = True

    return (jnp.array(q0_np), jnp.array(p0_np),
            jnp.array(z0_np), jnp.array(mask_np))


def make_S0(image):
    """Build initial state tensor S0 (N_MAX, 7) for a given image."""
    q0, p0, z0, mask = preprocess(image)
    S0 = jnp.concatenate([q0, p0, z0[:, None]], axis=-1)
    return S0, mask


# Precompute original test data
MASK_O = preprocess(O_IMAGE)[3]
MASK_X = preprocess(X_IMAGE)[3]
S0_O, _ = make_S0(O_IMAGE)
S0_X, _ = make_S0(X_IMAGE)


# ===========================================================================
# SECTION 5: RBF Potential and Gradient  (identical to Block II)
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
# SECTION 6: Contact Hamiltonian RHS + RK4  (identical to Block II)
# ===========================================================================
def contact_rhs(S, w, mu, sigma, gamma):
    q    = S[:, :3]
    p    = S[:, 3:6]
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
# SECTION 7: Analysis Utilities
# ===========================================================================
def com_single(q_t, mask):
    mf = mask.astype(jnp.float32)
    return jnp.sum(q_t * mf[:, None], axis=0) / jnp.sum(mf)


def classify_traj(trajectory, mask, qO=Q_STAR_O, qX=Q_STAR_X):
    qf = com_single(trajectory[-1, :, :3], mask)
    dO = float(jnp.linalg.norm(qf - qO))
    dX = float(jnp.linalg.norm(qf - qX))
    return ('O' if dO < dX else 'X'), dO, dX, qf


# ===========================================================================
# SECTION 8: simulate_eval  --  Forward-only (no grad, no checkpoint)
# ===========================================================================
def simulate_eval(S0, w, mu, sigma, gamma=GAMMA, dt=DT, n_steps=N_STEPS):
    """
    Forward-only simulation for evaluation (no gradient tracking needed).
    Faster than simulate_diff because no jax.checkpoint overhead.
    """
    sigma_safe = jnp.clip(sigma, 0.1, 10.0)

    def step(S, _):
        S_next = rk4_step(S, w, mu, sigma_safe, gamma, dt)
        return S_next, S_next

    _, steps   = jax.lax.scan(step, S0, None, length=n_steps)
    trajectory = jnp.concatenate([S0[None], steps], axis=0)
    return trajectory


# Also keep a differentiable version for the lightweight training fallback
def simulate_diff(S0, w, mu, sigma, gamma=GAMMA, dt=DT, n_steps=N_STEPS):
    """Differentiable forward simulation (with checkpoint for BPTT)."""
    sigma_safe = jnp.clip(sigma, 0.1, 10.0)

    @jax_checkpoint
    def step(S, _):
        S_next = rk4_step(S, w, mu, sigma_safe, gamma, dt)
        return S_next, S_next

    _, steps   = jax.lax.scan(step, S0, None, length=n_steps)
    trajectory = jnp.concatenate([S0[None], steps], axis=0)
    return trajectory


# ===========================================================================
# SECTION 9: Parameter Initialization  (identical to Block II)
# ===========================================================================
W_FROZEN     = jnp.array([-2.0, -2.0])
MU_FROZEN    = jnp.array([[ 8.0,  8.0,  0.88],
                            [-8.0, -8.0,  0.12]], dtype=jnp.float32)
SIGMA_FROZEN = jnp.array([2.0, 2.0])

_w_stones   = np.array([-0.5, -0.5], dtype=np.float32)
_mu_stones  = np.array([[ 6.0,  6.0, 0.5],
                         [-2.0, -2.0, 0.5]], dtype=np.float32)
_sig_stones = np.array([2.0, 2.0], dtype=np.float32)

_free_mu = np.array([
    [0.5, 6.0, 0.5], [2.5, 6.0, 0.5], [4.5, 6.0, 0.5], [6.5, 6.0, 0.5],
    [0.5, 3.5, 0.5], [2.5, 3.5, 0.5], [4.5, 3.5, 0.5], [6.5, 3.5, 0.5],
    [0.5, 1.0, 0.5], [2.5, 1.0, 0.5], [4.5, 1.0, 0.5], [6.5, 1.0, 0.5],
], dtype=np.float32)

_free_w = np.array([-0.20,  0.15, -0.10,  0.20,
                    -0.15,  0.10, -0.20,  0.15,
                    -0.10,  0.20, -0.15,  0.10], dtype=np.float32)
_free_sig = np.full(12, 2.0, dtype=np.float32)

_w_learn_np   = np.concatenate([_w_stones,  _free_w],    axis=0)
_mu_learn_np  = np.concatenate([_mu_stones, _free_mu],   axis=0)
_sig_learn_np = np.concatenate([_sig_stones, _free_sig], axis=0)

_sig_adj = np.clip(_sig_learn_np - 0.1, 1e-3, None)
_sigraw_learn_np = np.log(np.expm1(_sig_adj)).astype(np.float32)

PARAMS_INIT = {
    'w'        : jnp.array(_w_learn_np),
    'mu'       : jnp.array(_mu_learn_np),
    'sigma_raw': jnp.array(_sigraw_learn_np),
}


def full_params(params):
    """Combine frozen (k=0,1) with learnable (k=2..15) into K=16 arrays."""
    sigma_learn = jax.nn.softplus(params['sigma_raw']) + 0.1
    w     = jnp.concatenate([W_FROZEN,     params['w']],     axis=0)
    mu    = jnp.concatenate([MU_FROZEN,    params['mu']],    axis=0)
    sigma = jnp.concatenate([SIGMA_FROZEN, sigma_learn],     axis=0)
    return w, mu, sigma


# ===========================================================================
# SECTION 10: Parameter Loading from Block II
# ===========================================================================
def load_trained_params(path=None):
    """
    Load trained params from Block II checkpoint.
    Returns learnable params dict {w, mu, sigma_raw} and full arrays.
    Falls back to PARAMS_INIT if file not found.
    """
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
        print(f"  [Loaded] Block II trained params from {path}")
        print(f"  [Loaded] Converged at epoch: {epoch}")
        return params, True
    else:
        print(f"  [WARN] Block II params not found at {path}")
        print(f"  [WARN] Using initial params (untrained) -- results will be poor")
        return PARAMS_INIT, False


# ===========================================================================
# SECTION 11: Lightweight Fallback Training
# ===========================================================================
def fallback_training(n_epochs=200, peak_lr=2e-3):
    """
    Lightweight training loop (subset of Block II) so Block III can be
    tested standalone even when Block II hasn't been run yet.
    Produces usable (but not fully optimized) params.
    """
    print("\n" + "=" * 62)
    print("BLOCK III FALLBACK TRAINING (lightweight)")
    print("=" * 62)
    print(f"  epochs={n_epochs}  peak_lr={peak_lr}")

    def loss_fn(params):
        w, mu, sigma = full_params(params)
        traj_O  = simulate_diff(S0_O, w, mu, sigma)
        com_O_T = com_single(traj_O[-1, :, :3], MASK_O)
        traj_X  = simulate_diff(S0_X, w, mu, sigma)
        com_X_T = com_single(traj_X[-1, :, :3], MASK_X)
        loss_q = (jnp.sum((com_O_T - Q_STAR_O) ** 2) +
                  jnp.sum((com_X_T - Q_STAR_X) ** 2))
        mO = MASK_O.astype(jnp.float32)
        mX = MASK_X.astype(jnp.float32)
        pnorm_O = jnp.sum(jnp.sum(traj_O[-1, :, 3:6] ** 2, axis=-1) * mO) / jnp.sum(mO)
        pnorm_X = jnp.sum(jnp.sum(traj_X[-1, :, 3:6] ** 2, axis=-1) * mX) / jnp.sum(mX)
        loss_p = pnorm_O + pnorm_X
        total = loss_q + LAMBDA_P * loss_p
        return total, (loss_q, loss_p, com_O_T, com_X_T, traj_O, traj_X)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=peak_lr,
        warmup_steps=20, decay_steps=n_epochs, end_value=1e-5)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule))

    @jit
    def train_step(params, opt_state):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt, loss, aux

    params    = PARAMS_INIT
    opt_state = optimizer.init(params)

    print("  [JIT warmup] Compiling...")
    t0 = time.time()
    params, opt_state, loss_val, aux = train_step(params, opt_state)
    loss_val.block_until_ready()
    print(f"  Compilation: {time.time() - t0:.1f}s  initial loss={float(loss_val):.4f}")

    t_start = time.time()
    for epoch in range(1, n_epochs):
        params, opt_state, loss_val, aux = train_step(params, opt_state)
        if epoch % 20 == 0:
            loss_q, loss_p, com_O, com_X, _, _ = aux
            pred_O, _, _, _ = classify_traj(aux[4], MASK_O)
            pred_X, _, _, _ = classify_traj(aux[5], MASK_X)
            print(f"  Ep {epoch:4d} | L={float(loss_val):.4f} "
                  f"(q={float(loss_q):.3f} p={float(loss_p):.3f}) | "
                  f"pred={pred_O},{pred_X}")

    print(f"  Fallback training done in {time.time() - t_start:.1f}s")
    print(f"  Final loss = {float(loss_val):.6f}")
    return params


# ===========================================================================
# SECTION 12: Image Augmentation Utilities
# ===========================================================================
def add_noise(image, n_flips, rng_key):
    """Flip n_flips random pixels (0->1 or 1->0)."""
    flat = image.flatten().copy()
    indices = np.random.RandomState(int(rng_key)).choice(
        len(flat), size=min(n_flips, len(flat)), replace=False)
    flat[indices] = 1.0 - flat[indices]
    return flat.reshape(image.shape)


def shift_image(image, dx, dy):
    """Shift image by (dx, dy) pixels, zero-padding."""
    result = np.zeros_like(image)
    rows, cols = image.shape
    for r in range(rows):
        for c in range(cols):
            sr, sc = r - dy, c - dx
            if 0 <= sr < rows and 0 <= sc < cols:
                result[r, c] = image[sr, sc]
    return result


# ===========================================================================
# SECTION 13: Evaluation Pipeline
# ===========================================================================
def evaluate_single(image, w, mu, sigma, label="?",
                    gamma=GAMMA, dt=DT, n_steps=N_STEPS):
    """
    Run forward simulation on a single image and return classification result.
    Returns dict with: label, pred, dist_O, dist_X, final_com, n_particles
    """
    S0, mask = make_S0(image)
    n_particles = int(mask.sum())
    if n_particles == 0:
        return {'label': label, 'pred': '?', 'dist_O': float('inf'),
                'dist_X': float('inf'), 'final_com': [0, 0, 0],
                'n_particles': 0, 'converged_q': False, 'converged_p': False}

    traj = simulate_eval(S0, w, mu, sigma, gamma, dt, n_steps)
    pred, dO, dX, qf = classify_traj(traj, mask)

    mf = mask.astype(jnp.float32)
    p_final = traj[-1, :, 3:6]
    eps_p = float(jnp.sum(jnp.sum(p_final ** 2, axis=-1) * mf) / jnp.sum(mf))
    eps_q_O = float(jnp.linalg.norm(qf - Q_STAR_O))
    eps_q_X = float(jnp.linalg.norm(qf - Q_STAR_X))
    eps_q = min(eps_q_O, eps_q_X)

    return {
        'label':       label,
        'pred':        pred,
        'dist_O':      dO,
        'dist_X':      dX,
        'final_com':   [float(qf[0]), float(qf[1]), float(qf[2])],
        'n_particles': n_particles,
        'eps_q':       eps_q,
        'eps_p':       eps_p,
        'converged_q': eps_q < CONV_Q_THR,
        'converged_p': eps_p < CONV_P_THR,
    }


def evaluate_suite(images_dict, w, mu, sigma, gamma=GAMMA):
    """Evaluate a dict of {name: image} and return list of result dicts."""
    results = []
    for name, img in images_dict.items():
        r = evaluate_single(img, w, mu, sigma, label=name, gamma=gamma)
        results.append(r)
        print(f"    {name:>10s}: pred={r['pred']}  "
              f"dO={r['dist_O']:.3f}  dX={r['dist_X']:.3f}  "
              f"com=({r['final_com'][0]:.2f},{r['final_com'][1]:.2f})  "
              f"n={r['n_particles']}")
    return results


# ===========================================================================
# SECTION 14: Robustness Sweep -- Noise
# ===========================================================================
def noise_sweep(base_image, true_label, w, mu, sigma,
                flip_counts=(0, 1, 2, 3, 4, 5, 6, 8, 10),
                n_trials=5):
    """
    For each noise level (number of pixel flips), run n_trials and compute
    classification accuracy.
    Returns: list of (n_flips, accuracy, results)
    """
    print(f"\n  Noise sweep on {true_label}-image ({n_trials} trials/level):")
    sweep = []
    for n_flips in flip_counts:
        correct = 0
        trial_results = []
        for trial in range(n_trials):
            if n_flips == 0:
                img = base_image
            else:
                img = add_noise(base_image, n_flips, rng_key=trial * 100 + n_flips)
            r = evaluate_single(img, w, mu, sigma, label=f"{true_label}_n{n_flips}")
            trial_results.append(r)
            if r['pred'] == true_label:
                correct += 1
        acc = correct / n_trials
        sweep.append((n_flips, acc, trial_results))
        print(f"    flips={n_flips:2d}: acc={acc:.0%} ({correct}/{n_trials})")
    return sweep


# ===========================================================================
# SECTION 15: Robustness Sweep -- Spatial Shift
# ===========================================================================
def shift_sweep(base_image, true_label, w, mu, sigma,
                shifts=(-2, -1, 0, 1, 2)):
    """
    Test classification under (dx, dy) translations.
    Returns: dict of (dx,dy) -> result
    """
    print(f"\n  Shift sweep on {true_label}-image:")
    results = {}
    for dx in shifts:
        for dy in shifts:
            img = shift_image(base_image, dx, dy)
            n_on = int(img.sum())
            if n_on == 0:
                continue
            r = evaluate_single(img, w, mu, sigma,
                                label=f"{true_label}_s({dx},{dy})")
            results[(dx, dy)] = r
            status = 'OK' if r['pred'] == true_label else 'MISS'
            print(f"    shift=({dx:+d},{dy:+d}): pred={r['pred']} "
                  f"dO={r['dist_O']:.2f} dX={r['dist_X']:.2f}  [{status}]")
    total = len(results)
    correct = sum(1 for r in results.values() if r['pred'] == true_label)
    print(f"    Shift accuracy: {correct}/{total} = {correct/max(total,1):.0%}")
    return results


# ===========================================================================
# SECTION 16: Ablation Studies
# ===========================================================================
def ablation_study(params, images_dict):
    """
    Run evaluation with systematically removed components:
      (a) Full model (baseline)
      (b) No stepping stones (zero w_2, w_3)
      (c) No free RBFs (zero w_4..w_15)
      (d) Attractors only (zero all except w_0, w_1)
    """
    print("\n" + "-" * 50)
    print("ABLATION STUDY")
    print("-" * 50)
    ablations = {}

    # (a) Full model
    print("\n  [A] Full model (all K=16 RBFs):")
    w, mu, sigma = full_params(params)
    ablations['full'] = evaluate_suite(images_dict, w, mu, sigma)

    # (b) No stepping stones
    print("\n  [B] No stepping stones (w_2=w_3=0):")
    w_no_stones = w.at[2].set(0.0).at[3].set(0.0)
    ablations['no_stones'] = evaluate_suite(images_dict, w_no_stones, mu, sigma)

    # (c) No free RBFs
    print("\n  [C] No free RBFs (w_4..w_15=0):")
    w_no_free = w
    for k in range(4, 16):
        w_no_free = w_no_free.at[k].set(0.0)
    ablations['no_free'] = evaluate_suite(images_dict, w_no_free, mu, sigma)

    # (d) Attractors only
    print("\n  [D] Attractors only (w_0, w_1 only):")
    w_attract = jnp.zeros_like(w)
    w_attract = w_attract.at[0].set(w[0]).at[1].set(w[1])
    ablations['attract_only'] = evaluate_suite(images_dict, w_attract, mu, sigma)

    return ablations


# ===========================================================================
# SECTION 17: Gamma Sensitivity Analysis
# ===========================================================================
def gamma_sweep(params, gamma_values=(0.5, 1.0, 1.5, 2.0, 3.0)):
    """Evaluate classification with different damping coefficients."""
    print("\n" + "-" * 50)
    print("GAMMA SENSITIVITY ANALYSIS")
    print("-" * 50)
    w, mu, sigma = full_params(params)
    results = {}
    for g in gamma_values:
        r_O = evaluate_single(O_IMAGE, w, mu, sigma, label='O', gamma=g)
        r_X = evaluate_single(X_IMAGE, w, mu, sigma, label='X', gamma=g)
        ok_O = (r_O['pred'] == 'O')
        ok_X = (r_X['pred'] == 'X')
        results[g] = {'O': r_O, 'X': r_X, 'both_correct': ok_O and ok_X}
        print(f"  gamma={g:.1f}: O->{r_O['pred']}({'OK' if ok_O else 'FAIL'}) "
              f"X->{r_X['pred']}({'OK' if ok_X else 'FAIL'})  "
              f"dO={r_O['dist_O']:.3f}  dX={r_X['dist_X']:.3f}")
    return results


# ===========================================================================
# SECTION 18: Verification Figure  (8-panel, paper-ready)
# ===========================================================================
def make_block3_figure(params, noise_O, noise_X, shift_O, shift_X,
                       ablation_results, gamma_results, novel_results,
                       output_path=None):
    """
    8-panel Block III verification figure:
      [1,1] Learned potential landscape (z=0.5 slice)
      [1,2] Noise robustness curves (O and X)
      [1,3] Shift robustness heatmaps
      [1,4] Novel pattern classification
      [2,1] Ablation bar chart
      [2,2] Gamma sensitivity
      [2,3] Final particle trajectories (trained model)
      [2,4] Block III summary text
    """
    if output_path is None:
        output_path = str(_HERE / "block3_verification.png")

    w, mu, sigma = full_params(params)
    mu_np = np.array(mu)
    w_np  = np.array(w)

    fig = plt.figure(figsize=(22, 10))
    gs  = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.40)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(2)]

    # ── [1,1] Potential landscape ────────────────────────────────────────
    ax = axes[0][0]
    xy_r = np.linspace(-12, 12, 200)
    gx, gy = np.meshgrid(xy_r, xy_r)
    q_grid = jnp.array(np.stack(
        [gx.ravel(), gy.ravel(), np.full(gx.size, 0.5)],
        axis=1).astype(np.float32))
    Vg = np.array(rbf_potential(q_grid, w, mu, sigma)).reshape(gx.shape)
    cf = ax.contourf(gx, gy, Vg, levels=40, cmap='RdYlBu_r', alpha=0.85)
    ax.contour(gx, gy, Vg, levels=20, colors='k', linewidths=0.3, alpha=0.4)
    plt.colorbar(cf, ax=ax, shrink=0.85)
    ax.scatter(*np.array(Q_STAR_O)[:2], s=200, c='blue', marker='*', zorder=6)
    ax.scatter(*np.array(Q_STAR_X)[:2], s=200, c='red',  marker='*', zorder=6)
    for k in range(16):
        col = 'white' if k < 2 else ('cyan' if k < 4 else 'lime')
        ax.scatter(*mu_np[k, :2], s=15, c=col, edgecolors='k',
                   linewidths=0.3, zorder=5)
    ax.set(xlim=(-12, 12), ylim=(-12, 12), xlabel='x', ylabel='y',
           title='Learned RBF Potential (z=0.5)')

    # ── [1,2] Noise robustness ──────────────────────────────────────────
    ax = axes[0][1]
    flips_O = [s[0] for s in noise_O]
    acc_O   = [s[1] for s in noise_O]
    flips_X = [s[0] for s in noise_X]
    acc_X   = [s[1] for s in noise_X]
    ax.plot(flips_O, acc_O, 'bo-', lw=2, label='O-image')
    ax.plot(flips_X, acc_X, 'rs-', lw=2, label='X-image')
    ax.set(xlabel='# Pixel Flips', ylabel='Classification Accuracy',
           title='Noise Robustness', ylim=(-0.05, 1.15))
    ax.axhline(1.0, color='gray', ls=':', lw=0.8)
    ax.legend(fontsize=9)

    # ── [1,3] Shift robustness heatmap ──────────────────────────────────
    ax = axes[0][2]
    shifts = sorted(set(k[0] for k in shift_O.keys()))
    n_s = len(shifts)
    heat = np.zeros((n_s, n_s))
    for i, dy in enumerate(shifts):
        for j, dx in enumerate(shifts):
            key = (dx, dy)
            ok_O = shift_O.get(key, {}).get('pred', '?') == 'O'
            ok_X = shift_X.get(key, {}).get('pred', '?') == 'X'
            heat[i, j] = int(ok_O) + int(ok_X)
    im = ax.imshow(heat, cmap='RdYlGn', vmin=0, vmax=2,
                   extent=[shifts[0]-0.5, shifts[-1]+0.5,
                           shifts[0]-0.5, shifts[-1]+0.5],
                   origin='lower')
    plt.colorbar(im, ax=ax, ticks=[0, 1, 2],
                 label='Correct (0=none, 2=both)')
    ax.set(xlabel='dx shift', ylabel='dy shift',
           title='Shift Robustness (O+X)')

    # ── [1,4] Novel patterns ────────────────────────────────────────────
    ax = axes[0][3]
    ax.axis('off')
    lines = ["Novel Pattern Classification", "=" * 30, ""]
    for r in novel_results:
        lines.append(f"  {r['label']:>6s} -> {r['pred']}  "
                     f"dO={r['dist_O']:.2f}  dX={r['dist_X']:.2f}")
    lines.append("")
    lines.append("(Patterns unseen during training)")
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=8.5, fontfamily='monospace', va='top')
    ax.set_title('Novel Patterns', fontweight='bold')

    # ── [2,1] Ablation bar chart ────────────────────────────────────────
    ax = axes[1][0]
    configs = ['full', 'no_stones', 'no_free', 'attract_only']
    labels  = ['Full\nmodel', 'No step\nstones', 'No free\nRBFs', 'Attractors\nonly']
    accuracies = []
    for cfg in configs:
        res_list = ablation_results.get(cfg, [])
        n_correct = sum(1 for r in res_list
                        if (r['label'] == 'O' and r['pred'] == 'O') or
                           (r['label'] == 'X' and r['pred'] == 'X'))
        n_total = sum(1 for r in res_list if r['label'] in ('O', 'X'))
        accuracies.append(n_correct / max(n_total, 1))
    colors = ['#2196F3', '#FF9800', '#F44336', '#9E9E9E']
    ax.bar(range(len(configs)), accuracies, color=colors, edgecolor='k', lw=0.5)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set(ylabel='Accuracy (O+X)', ylim=(0, 1.15),
           title='Ablation Study')
    for i, v in enumerate(accuracies):
        ax.text(i, v + 0.03, f'{v:.0%}', ha='center', fontsize=9,
                fontweight='bold')

    # ── [2,2] Gamma sensitivity ─────────────────────────────────────────
    ax = axes[1][1]
    gammas = sorted(gamma_results.keys())
    dO_vals = [gamma_results[g]['O']['dist_O'] for g in gammas]
    dX_vals = [gamma_results[g]['X']['dist_X'] for g in gammas]
    correct = [gamma_results[g]['both_correct'] for g in gammas]
    ax.plot(gammas, dO_vals, 'bo-', lw=2, label='O: dist to q*_O')
    ax.plot(gammas, dX_vals, 'rs-', lw=2, label='X: dist to q*_X')
    ax.axhline(CONV_Q_THR, color='gray', ls=':', label=f'thr={CONV_Q_THR}')
    for i, g in enumerate(gammas):
        if correct[i]:
            ax.scatter(g, 0, marker='^', c='green', s=80, zorder=5)
    ax.set(xlabel='gamma', ylabel='Distance to target',
           title='Gamma Sensitivity')
    ax.legend(fontsize=8)

    # ── [2,3] Final trajectories ────────────────────────────────────────
    ax = axes[1][2]
    mO = np.array(MASK_O, dtype=bool)
    mX = np.array(MASK_X, dtype=bool)
    traj_O = simulate_eval(S0_O, w, mu, sigma)
    traj_X = simulate_eval(S0_X, w, mu, sigma)
    tOnp = np.array(traj_O[:, mO, :])
    tXnp = np.array(traj_X[:, mX, :])
    dec = max(1, N_STEPS // 50)
    for i in range(tOnp.shape[1]):
        ax.plot(tOnp[::dec, i, 0], tOnp[::dec, i, 1],
                'b-', alpha=0.15, lw=0.5)
    for i in range(tXnp.shape[1]):
        ax.plot(tXnp[::dec, i, 0], tXnp[::dec, i, 1],
                'r-', alpha=0.15, lw=0.5)
    com_O = np.array(vmap(lambda qt: com_single(qt, MASK_O))(traj_O[:, :, :3]))
    com_X = np.array(vmap(lambda qt: com_single(qt, MASK_X))(traj_X[:, :, :3]))
    ax.plot(com_O[::dec, 0], com_O[::dec, 1], 'b-', lw=2.5, label='CoM O')
    ax.plot(com_X[::dec, 0], com_X[::dec, 1], 'r-', lw=2.5, label='CoM X')
    ax.scatter(*np.array(Q_STAR_O)[:2], s=200, c='blue', marker='*', zorder=6)
    ax.scatter(*np.array(Q_STAR_X)[:2], s=200, c='red',  marker='*', zorder=6)
    ax.set(xlabel='q_x', ylabel='q_y', title='Particle Trajectories (trained)')
    ax.legend(fontsize=8)

    # ── [2,4] Summary ───────────────────────────────────────────────────
    ax = axes[1][3]
    ax.axis('off')

    # Compute summary stats
    full_res = ablation_results.get('full', [])
    o_ok = any(r['label'] == 'O' and r['pred'] == 'O' for r in full_res)
    x_ok = any(r['label'] == 'X' and r['pred'] == 'X' for r in full_res)
    cls_pass = o_ok and x_ok
    noise_robust = (all(s[1] >= 0.8 for s in noise_O[:4]) and
                    all(s[1] >= 0.8 for s in noise_X[:4]))
    n_gamma_ok = sum(1 for g in gamma_results.values() if g['both_correct'])

    all_pass = cls_pass and noise_robust
    bg = '#d4edda' if all_pass else '#fff3cd'
    ax.set_facecolor(bg)
    ax.set(xlim=(0, 1), ylim=(0, 1))

    lns = [
        "BLOCK III EVALUATION SUMMARY",
        "=" * 32, "",
        f"Classification (O/X) : {'PASS' if cls_pass else 'FAIL'}",
        f"  O->{'O' if o_ok else '?'}  X->{'X' if x_ok else '?'}",
        "",
        f"Noise robustness (<=3 flips):",
        f"  {'PASS' if noise_robust else 'FAIL'}",
        "",
        f"Gamma sensitivity:",
        f"  {n_gamma_ok}/{len(gamma_results)} settings correct",
        "",
        f"Novel patterns tested: {len(novel_results)}",
        "",
        f"OVERALL: {'*** PASS ***' if all_pass else '--- PARTIAL ---'}",
        "",
        f"gamma={GAMMA}  T={T_FINAL}  dt={DT}  K=16",
    ]
    ax.text(0.05, 0.95, "\n".join(lns), transform=ax.transAxes,
            fontsize=8, fontfamily='monospace', va='top')
    title_col = 'darkgreen' if all_pass else 'darkorange'
    ax.set_title('Block III Summary', fontweight='bold', color=title_col)

    fig.suptitle(
        "Contact Hamiltonian Fluid NN  --  Block III: "
        "Generalization & Robustness\n"
        f"gamma={GAMMA}  T={T_FINAL}  dt={DT}  K=16 RBF  [JAX / CUDA]",
        fontsize=10.5, y=1.01)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  [Figure] Saved -> {output_path}")
    plt.close(fig)
    return all_pass


# ===========================================================================
# SECTION 19: Main Evaluation Pipeline
# ===========================================================================
def run_evaluation():
    """
    Main Block III pipeline:
      1. Load (or train) parameters
      2. Baseline evaluation (O, X)
      3. Noise robustness sweep
      4. Shift robustness sweep
      5. Novel pattern evaluation
      6. Ablation studies
      7. Gamma sensitivity
      8. Generate verification figure
    """
    print("\n" + "=" * 62)
    print("BLOCK III: GENERALIZATION & ROBUSTNESS EVALUATION")
    print("=" * 62)

    # ── Step 1: Load parameters ──────────────────────────────────────────
    print("\n[Step 1] Loading trained parameters...")
    params, loaded = load_trained_params()
    if not loaded:
        print("[Step 1b] Running fallback training...")
        params = fallback_training(n_epochs=200)

    w, mu, sigma = full_params(params)
    print(f"  K=16 params ready.  sigma range: "
          f"[{float(jnp.min(sigma)):.3f}, {float(jnp.max(sigma)):.3f}]")

    # ── Step 2: Baseline evaluation ──────────────────────────────────────
    print("\n[Step 2] Baseline evaluation (original O/X):")
    baseline = {'O': O_IMAGE, 'X': X_IMAGE}
    baseline_results = evaluate_suite(baseline, w, mu, sigma)

    # ── Step 3: Noise robustness ─────────────────────────────────────────
    print("\n[Step 3] Noise robustness sweep:")
    noise_O = noise_sweep(O_IMAGE, 'O', w, mu, sigma)
    noise_X = noise_sweep(X_IMAGE, 'X', w, mu, sigma)

    # ── Step 4: Shift robustness ─────────────────────────────────────────
    print("\n[Step 4] Shift robustness sweep:")
    shift_O = shift_sweep(O_IMAGE, 'O', w, mu, sigma)
    shift_X = shift_sweep(X_IMAGE, 'X', w, mu, sigma)

    # ── Step 5: Novel patterns ───────────────────────────────────────────
    print("\n[Step 5] Novel pattern evaluation:")
    novel_results = evaluate_suite(NOVEL_IMAGES, w, mu, sigma)

    # ── Step 6: Ablation studies ─────────────────────────────────────────
    print("\n[Step 6] Ablation studies:")
    ablation_results = ablation_study(params, baseline)

    # ── Step 7: Gamma sensitivity ────────────────────────────────────────
    print("\n[Step 7] Gamma sensitivity:")
    gamma_results = gamma_sweep(params)

    # ── Step 8: Generate figure ──────────────────────────────────────────
    print("\n[Step 8] Generating verification figure...")
    all_pass = make_block3_figure(
        params, noise_O, noise_X, shift_O, shift_X,
        ablation_results, gamma_results, novel_results)

    # ── Final report ─────────────────────────────────────────────────────
    S = "=" * 62
    print(f"\n{S}")
    print("BLOCK III FINAL REPORT")
    print(S)
    for r in baseline_results:
        print(f"  {r['label']}: pred={r['pred']}  "
              f"com=({r['final_com'][0]:.3f},{r['final_com'][1]:.3f},"
              f"{r['final_com'][2]:.3f})  "
              f"dO={r['dist_O']:.4f}  dX={r['dist_X']:.4f}")
    o_ok = any(r['label'] == 'O' and r['pred'] == 'O' for r in baseline_results)
    x_ok = any(r['label'] == 'X' and r['pred'] == 'X' for r in baseline_results)
    print(f"  Classification: {'PASS' if (o_ok and x_ok) else 'FAIL'}")
    print(f"  OVERALL: {'*** ALL PASS ***' if all_pass else '--- PARTIAL ---'}")
    print(f"  Figure: {_HERE / 'block3_verification.png'}")
    print(S)

    return params, all_pass, {
        'baseline': baseline_results,
        'noise_O': noise_O, 'noise_X': noise_X,
        'shift_O': shift_O, 'shift_X': shift_X,
        'novel': novel_results,
        'ablation': ablation_results,
        'gamma': gamma_results,
    }


# ===========================================================================
# SECTION 20: Entry Point
# ===========================================================================
if __name__ == "__main__":

    # Sanity check on initial params
    w0, mu0, sig0 = full_params(PARAMS_INIT)
    assert w0.shape   == (16,),   f"w shape: {w0.shape}"
    assert mu0.shape  == (16, 3), f"mu shape: {mu0.shape}"
    assert sig0.shape == (16,),   f"sigma shape: {sig0.shape}"
    print(f"\n[Init check] w:{w0.shape} mu:{mu0.shape} sigma:{sig0.shape}")
    print(f"  O particles: {int(MASK_O.sum())}  X particles: {int(MASK_X.sum())}")

    # Novel image particle counts
    for name, img in NOVEL_IMAGES.items():
        n = int(img.sum())
        print(f"  {name} pixels: {n}")

    # Run full evaluation
    t_total = time.time()
    trained_params, passed, all_results = run_evaluation()
    print(f"\n[Total time] {time.time() - t_total:.1f}s")
