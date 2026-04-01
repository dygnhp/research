"""
=============================================================================
Contact Hamiltonian Fluid Neural Network  --  Block III: N-Dimensional
                                               Adaptive Processing
=============================================================================
Extends Block II with:
  (1) Variable dimension D (starting at D=3, grows to D_max)
  (2) Variable RBF count K (starting at K_init, grows to K_max)
  (3) Plateau-triggered growth: if loss stagnates, add K or D
  (4) Per-trial dataset saving to reference/dataset_used/

Growth strategy:
  Phase 1: Train with (D_init, K_init) until plateau
  Phase 2: On plateau → add K_GROW new RBFs at residual locations
  Phase 3: If K growth fails → expand dimension D → D+1
  Repeat until D_max or K_max or convergence

Backend:   JAX (CUDA) -- JIT-compiled
=============================================================================
"""

from pathlib import Path
import os
import sys
import json
import time

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"

import jax
import jax.numpy as jnp
from jax import jit
from jax import checkpoint as jax_checkpoint
from functools import partial
import numpy as np
import optax

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

sys.path.insert(0, str(_ROOT))
from data_generator import (generate_dataset, O_CANONICAL, X_CANONICAL)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("=" * 62)
print("BLOCK III: N-Dimensional Adaptive Processing")
print("=" * 62)
print(f"JAX version    : {jax.__version__}")
print(f"Devices        : {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")
print("=" * 62)


# ===========================================================================
# SECTION 1: Configuration
# ===========================================================================
class Config:
    """Block III hyperparameters. Immutable after construction."""
    def __init__(self, **overrides):
        # Physics
        self.GAMMA     = 1.5
        self.T_FINAL   = 10.0
        self.DT        = 0.05
        self.N_STEPS   = int(self.T_FINAL / self.DT)
        self.N_MAX     = 64
        self.TAU       = 0.5
        self.LAMBDA_P  = 0.1

        # Dimensions
        self.D_INIT    = 3       # starting dimension
        self.D_MAX     = 6       # maximum dimension
        self.K_INIT    = 16      # starting RBF count (2 frozen + 2 stones + 12 free)
        self.K_GROW    = 4       # RBFs added per growth event
        self.K_MAX     = 32      # maximum total RBF count
        self.N_FROZEN  = 2       # frozen attractors (k=0, k=1)
        self.N_STONES  = 2       # stepping stones (k=2, k=3)

        # Training
        self.N_EPOCHS     = 3000
        self.PEAK_LR      = 5e-3
        self.END_LR       = 1e-5
        self.WARMUP_STEPS  = 100
        self.LOG_EVERY     = 20
        self.SAVE_EVERY    = 500
        self.CONV_Q_THR    = 2.0
        self.CONV_P_THR    = 0.5

        # Growth triggers
        self.PLATEAU_WINDOW   = 100    # epochs to check for plateau
        self.PLATEAU_THRESHOLD = 0.01  # relative improvement threshold
        self.MIN_EPOCHS_BEFORE_GROW = 200  # minimum epochs before first growth
        self.COOLDOWN_AFTER_GROW    = 100  # epochs to wait after growth

        # Dataset
        self.N_TRAIN_PER_CLASS = 50
        self.DATASET_SEED      = 42

        # Attractors
        self.Q_STAR_O_2D = np.array([ 8.0,  8.0], dtype=np.float32)
        self.Q_STAR_X_2D = np.array([-8.0, -8.0], dtype=np.float32)

        # Apply overrides
        for k, v in overrides.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError(f"Unknown config key: {k}")

    def q_star(self, label, D):
        """Get D-dimensional attractor target."""
        base = self.Q_STAR_O_2D if label == 'O' else self.Q_STAR_X_2D
        return np.concatenate([base, np.zeros(D - 2, dtype=np.float32)])

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_') and not callable(v)}


# ===========================================================================
# SECTION 2: N-Dimensional Preprocessing
# ===========================================================================

def _axis_diag_features(image, r, c):
    """Compute axis-connectivity and diagonal-connectivity scores."""
    rows, cols = image.shape
    def px(rr, cc):
        return float(image[rr, cc]) if 0 <= rr < rows and 0 <= cc < cols else 0.0

    d_axis = px(r-1, c) + px(r+1, c) + px(r, c-1) + px(r, c+1)
    d_diag_signed = (px(r-1, c-1) + px(r+1, c+1) -
                     px(r-1, c+1) - px(r+1, c-1))
    return d_axis, d_diag_signed


def _local_density(image, r, c):
    """3x3 window density."""
    rows, cols = image.shape
    count = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols:
                count += image[rr, cc]
    return count / 9.0


def preprocess_nd(image, D, tau=0.5, n_max=64, beta=1.0):
    """
    N-dimensional preprocessing: image -> (q0, p0, z0, mask).

    Dimension features:
      d=0: x = col
      d=1: y = 7 - row
      d=2: z_connectivity = sigmoid(beta * (d_axis - |d_diag|))
      d=3: local_density = 3x3 neighbor count / 9
      d=4+: zeros (available for learned features in future)
    """
    rows, cols = image.shape
    q_list = []
    for r in range(rows):
        for c in range(cols):
            if image[r, c] > tau:
                features = np.zeros(D, dtype=np.float32)
                features[0] = float(c)
                features[1] = float(rows - 1 - r)
                if D >= 3:
                    d_axis, d_diag_signed = _axis_diag_features(image, r, c)
                    score = d_axis - abs(d_diag_signed)
                    features[2] = 1.0 / (1.0 + np.exp(-beta * score))
                if D >= 4:
                    features[3] = _local_density(image, r, c)
                # D>=5: features[4:] remain 0 (available for extensions)
                q_list.append(features)

    n_real = len(q_list)
    assert n_real > 0, "No particles"
    assert n_real <= n_max, f"Too many particles: {n_real} > {n_max}"

    q0   = np.zeros((n_max, D), dtype=np.float32)
    p0   = np.zeros((n_max, D), dtype=np.float32)
    z0   = np.zeros((n_max,),   dtype=np.float32)
    mask = np.zeros((n_max,),   dtype=bool)

    for i, pos in enumerate(q_list):
        q0[i]   = pos
        mask[i] = True

    return (jnp.array(q0), jnp.array(p0), jnp.array(z0), jnp.array(mask))


def make_S0_nd(image, D, n_max=64):
    """Build (S0, mask) for N-dimensional state."""
    q0, p0, z0, mask = preprocess_nd(image, D, n_max=n_max)
    S0 = jnp.concatenate([q0, p0, z0[:, None]], axis=-1)  # (N_MAX, 2D+1)
    return S0, mask


# ===========================================================================
# SECTION 3: N-Dimensional RBF Potential and Gradient
# ===========================================================================
def rbf_potential_nd(q, w, mu, sigma):
    """V(q) = sum_k w_k * exp(-||q - mu_k||^2 / (2*sigma_k^2)). D-agnostic."""
    diff    = q[:, None, :] - mu[None, :, :]
    sq_dist = jnp.sum(diff ** 2, axis=-1)
    gauss   = jnp.exp(-sq_dist / (2.0 * sigma ** 2))
    return jnp.sum(w * gauss, axis=-1)


def rbf_gradient_nd(q, w, mu, sigma):
    """grad_q V. D-agnostic."""
    diff    = q[:, None, :] - mu[None, :, :]
    sq_dist = jnp.sum(diff ** 2, axis=-1)
    gauss   = jnp.exp(-sq_dist / (2.0 * sigma ** 2))
    factor  = w * gauss / (sigma ** 2)
    return jnp.sum(-factor[:, :, None] * diff, axis=1)


# ===========================================================================
# SECTION 4: N-Dimensional Contact Hamiltonian
# ===========================================================================
def contact_rhs_nd(S, w, mu, sigma, gamma, D):
    """Contact Hamiltonian RHS for D-dimensional state."""
    q    = S[:, :D]
    p    = S[:, D:2*D]
    V    = rbf_potential_nd(q, w, mu, sigma)
    gV   = rbf_gradient_nd(q, w, mu, sigma)
    p_sq = jnp.sum(p ** 2, axis=-1)
    H_i  = p_sq / 2.0 + V
    dq_dt = p
    dp_dt = -gV - gamma * p
    dz_dt = p_sq - H_i
    return jnp.concatenate([dq_dt, dp_dt, dz_dt[:, None]], axis=-1)


def make_rk4_step(D, gamma, dt):
    """Create a JIT-compiled RK4 step for given D, gamma, dt."""
    @jit
    def rk4_step(S, w, mu, sigma):
        f = lambda s: contact_rhs_nd(s, w, mu, sigma, gamma, D)
        k1 = f(S)
        k2 = f(S + 0.5 * dt * k1)
        k3 = f(S + 0.5 * dt * k2)
        k4 = f(S +       dt * k3)
        return S + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return rk4_step


def make_simulate_diff(D, gamma, dt, n_steps):
    """Create differentiable simulator for given (D, gamma, dt, n_steps)."""
    rk4 = make_rk4_step(D, gamma, dt)

    def simulate_diff(S0, w, mu, sigma, **_ignored):
        sigma_safe = jnp.clip(sigma, 0.1, 10.0)
        @jax_checkpoint
        def step(S, _):
            S_next = rk4(S, w, mu, sigma_safe)
            return S_next, S_next
        _, steps   = jax.lax.scan(step, S0, None, length=n_steps)
        return jnp.concatenate([S0[None], steps], axis=0)

    return simulate_diff


def make_simulate_eval(D, gamma, dt, n_steps):
    """Create eval-only simulator (no checkpoint)."""
    rk4 = make_rk4_step(D, gamma, dt)

    def simulate_eval(S0, w, mu, sigma, **_ignored):
        sigma_safe = jnp.clip(sigma, 0.1, 10.0)
        def step(S, _):
            S_next = rk4(S, w, mu, sigma_safe)
            return S_next, S_next
        _, steps = jax.lax.scan(step, S0, None, length=n_steps)
        return jnp.concatenate([S0[None], steps], axis=0)

    return simulate_eval


# ===========================================================================
# SECTION 5: Parameter Initialization (N-Dimensional)
# ===========================================================================

def make_frozen_params(D, cfg):
    """Create frozen attractor params in D dimensions."""
    q_O = cfg.q_star('O', D)
    q_X = cfg.q_star('X', D)
    # z-channel: O-attractor at z=0.88, X-attractor at z=0.12
    if D >= 3:
        q_O[2] = 0.88
        q_X[2] = 0.12
    w_frozen     = jnp.array([-2.0, -2.0])
    mu_frozen    = jnp.array(np.stack([q_O, q_X]), dtype=jnp.float32)
    sigma_frozen = jnp.array([2.0, 2.0])
    return w_frozen, mu_frozen, sigma_frozen


def make_initial_learnable(D, cfg):
    """Create initial learnable parameters (stones + free) in D dimensions."""
    # Stepping stones
    mu_stone_O = np.zeros(D, dtype=np.float32)
    mu_stone_O[0], mu_stone_O[1] = 6.0, 6.0
    if D >= 3: mu_stone_O[2] = 0.88

    mu_stone_X = np.zeros(D, dtype=np.float32)
    mu_stone_X[0], mu_stone_X[1] = 0.0, 0.0
    if D >= 3: mu_stone_X[2] = 0.12

    w_stones   = np.array([-1.0, -1.0], dtype=np.float32)
    mu_stones  = np.stack([mu_stone_O, mu_stone_X])
    sig_stones = np.array([3.0, 3.0], dtype=np.float32)

    # Free RBFs: 4x3 grid in (x,y), mid-values in higher dims
    n_free = cfg.K_INIT - cfg.N_FROZEN - cfg.N_STONES  # 12
    xs = [0.5, 2.5, 4.5, 6.5]
    ys = [6.0, 3.5, 1.0]
    free_mu = np.zeros((n_free, D), dtype=np.float32)
    idx = 0
    for y_val in ys:
        for x_val in xs:
            free_mu[idx, 0] = x_val
            free_mu[idx, 1] = y_val
            if D >= 3: free_mu[idx, 2] = 0.5
            if D >= 4: free_mu[idx, 3] = 0.5
            idx += 1

    free_w = np.array([-0.20,  0.15, -0.10,  0.20,
                       -0.15,  0.10, -0.20,  0.15,
                       -0.10,  0.20, -0.15,  0.10], dtype=np.float32)[:n_free]
    free_sig = np.full(n_free, 2.0, dtype=np.float32)

    # Concatenate learnable
    w   = np.concatenate([w_stones,  free_w])
    mu  = np.concatenate([mu_stones, free_mu])
    sig = np.concatenate([sig_stones, free_sig])

    # Reparameterize sigma
    sig_adj = np.clip(sig - 0.1, 1e-3, None)
    sig_raw = np.log(np.expm1(sig_adj)).astype(np.float32)

    return {
        'w':         jnp.array(w),
        'mu':        jnp.array(mu),
        'sigma_raw': jnp.array(sig_raw),
    }


def full_params_fn(params, w_frozen, mu_frozen, sigma_frozen):
    """Combine frozen + learnable into full arrays."""
    sigma_learn = jax.nn.softplus(params['sigma_raw']) + 0.1
    w     = jnp.concatenate([w_frozen,     params['w']])
    mu    = jnp.concatenate([mu_frozen,    params['mu']])
    sigma = jnp.concatenate([sigma_frozen, sigma_learn])
    return w, mu, sigma


# ===========================================================================
# SECTION 6: Growth Mechanisms
# ===========================================================================

class PlateauDetector:
    """Detect loss plateau using moving average."""
    def __init__(self, window=100, threshold=0.01):
        self.window = window
        self.threshold = threshold
        self.losses = []

    def update(self, loss):
        self.losses.append(float(loss))

    def is_plateau(self):
        if len(self.losses) < 2 * self.window:
            return False
        old = np.mean(self.losses[-2*self.window:-self.window])
        new = np.mean(self.losses[-self.window:])
        if old == 0:
            return False
        improvement = (old - new) / abs(old)
        return improvement < self.threshold


def grow_K(params, w_frozen, mu_frozen, sigma_frozen, D, K_current, K_grow, cfg):
    """
    Add K_grow new RBFs. Place them at midpoints between current CoM and target.
    Returns: new_params, new_K
    """
    new_K = min(K_current + K_grow, cfg.K_MAX)
    n_add = new_K - K_current
    if n_add <= 0:
        return params, K_current

    # Compute current full params
    w, mu, sigma = full_params_fn(params, w_frozen, mu_frozen, sigma_frozen)

    # Place new RBFs in the data domain with slight random offsets
    rng = np.random.RandomState(K_current * 137)
    new_mu = np.zeros((n_add, D), dtype=np.float32)
    for i in range(n_add):
        # Random location in data domain [0,7]^2 x [0,1] x ...
        new_mu[i, 0] = rng.uniform(0, 7)
        new_mu[i, 1] = rng.uniform(0, 7)
        if D >= 3: new_mu[i, 2] = rng.uniform(0.1, 0.9)
        if D >= 4: new_mu[i, 3] = rng.uniform(0.2, 0.8)
        for d in range(4, D):
            new_mu[i, d] = 0.0

    new_w = rng.uniform(-0.3, 0.3, size=n_add).astype(np.float32)
    new_sig = np.full(n_add, 2.0, dtype=np.float32)
    new_sig_adj = np.clip(new_sig - 0.1, 1e-3, None)
    new_sig_raw = np.log(np.expm1(new_sig_adj)).astype(np.float32)

    # Extend learnable params
    new_params = {
        'w':         jnp.concatenate([params['w'],         jnp.array(new_w)]),
        'mu':        jnp.concatenate([params['mu'],        jnp.array(new_mu)]),
        'sigma_raw': jnp.concatenate([params['sigma_raw'], jnp.array(new_sig_raw)]),
    }

    print(f"  [GROW K] {K_current} -> {new_K} (+{n_add} RBFs)")
    return new_params, new_K


def grow_D(params, w_frozen, mu_frozen, sigma_frozen, D_old, D_new, cfg):
    """
    Extend dimension from D_old to D_new.
    Existing params are warm-started; new dimensions get zero position.
    Returns: new_params, new_w_frozen, new_mu_frozen, new_sigma_frozen
    """
    d_add = D_new - D_old
    if d_add <= 0:
        return params, w_frozen, mu_frozen, sigma_frozen

    # Extend frozen mu
    new_mu_frozen = jnp.concatenate(
        [mu_frozen, jnp.zeros((mu_frozen.shape[0], d_add))], axis=1)

    # Extend learnable mu
    new_mu_learn = jnp.concatenate(
        [params['mu'], jnp.zeros((params['mu'].shape[0], d_add))], axis=1)

    new_params = {
        'w':         params['w'],
        'mu':        new_mu_learn,
        'sigma_raw': params['sigma_raw'],
    }

    print(f"  [GROW D] {D_old} -> {D_new} (state vector: {2*D_old+1} -> {2*D_new+1})")
    return new_params, w_frozen, new_mu_frozen, sigma_frozen


# ===========================================================================
# SECTION 7: Trial Manager (dataset saving)
# ===========================================================================

class TrialManager:
    """Manage per-trial dataset, config, and result saving."""

    def __init__(self, trial_id, cfg, output_dir=None, dataset_dir=None):
        self.trial_id = trial_id
        self.cfg = cfg
        self.output_dir = Path(output_dir or _HERE)
        self.dataset_dir = Path(dataset_dir or _ROOT / "reference" / "dataset_used")
        self.growth_log = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    def tag(self):
        return f"block3_trial{self.trial_id:02d}"

    def save_config(self):
        """Save trial configuration as JSON."""
        config_path = self.dataset_dir / f"{self.tag()}_config.json"
        data = {
            'trial_id': self.trial_id,
            'tag': self.tag(),
            **self.cfg.to_dict(),
        }
        # Convert numpy arrays to lists for JSON
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                data[k] = v.tolist()
        with open(config_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  [Config] {config_path}")
        return config_path

    def save_dataset(self, n_per_class=None, seed=None):
        """Save the training dataset for this trial."""
        n = n_per_class or self.cfg.N_TRAIN_PER_CLASS
        s = seed or self.cfg.DATASET_SEED

        # Use save_dataset from the existing saver
        sys.path.insert(0, str(self.dataset_dir))
        try:
            from save_dataset import save_dataset as _save_ds
            return _save_ds(self.dataset_dir, n_per_class=n, seed=s,
                            tag=self.tag())
        except ImportError:
            # Fallback: save directly
            ds = generate_dataset(n_per_class=n, seed=s)
            npz_path = self.dataset_dir / f"{self.tag()}_dataset.npz"
            np.savez_compressed(
                npz_path,
                O_images=np.stack(ds['O_images']),
                X_images=np.stack(ds['X_images']),
                labels=np.array(ds['labels']),
                n_per_class=np.array(n),
                seed=np.array(s),
            )
            print(f"  [Dataset] {npz_path}")
            return npz_path

    def log_growth(self, epoch, event, before, after):
        """Record a growth event."""
        entry = {
            'epoch': epoch,
            'event': event,
            **before,
            **{f'{k}_after': v for k, v in after.items()},
        }
        self.growth_log.append(entry)

    def save_growth_log(self):
        """Save growth log as JSON."""
        path = self.output_dir / f"{self.tag()}_growth_log.json"
        with open(path, 'w') as f:
            json.dump(self.growth_log, f, indent=2)
        print(f"  [Growth log] {path}")
        return path

    def save_params(self, params, D, K, epoch, loss):
        """Save trained parameters."""
        path = self.output_dir / f"{self.tag()}_params_D{D}_K{K}.npy"
        np.save(path, {
            'w':         np.array(params['w']),
            'mu':        np.array(params['mu']),
            'sigma_raw': np.array(params['sigma_raw']),
            'D': D, 'K': K, 'epoch': epoch, 'loss': float(loss),
        }, allow_pickle=True)
        print(f"  [Params] {path}")
        return path


# ===========================================================================
# SECTION 8: Adaptive Training Loop
# ===========================================================================

def com_single(q_t, mask):
    mf = mask.astype(jnp.float32)
    return jnp.sum(q_t * mf[:, None], axis=0) / jnp.sum(mf)


def prepare_training_data(D, cfg):
    """Generate and preprocess training dataset for dimension D."""
    ds = generate_dataset(cfg.N_TRAIN_PER_CLASS, cfg.DATASET_SEED)

    S0_O_list, mask_O_list = [], []
    for img in ds['O_images']:
        s0, mask = make_S0_nd(img, D, cfg.N_MAX)
        S0_O_list.append(s0)
        mask_O_list.append(mask)

    S0_X_list, mask_X_list = [], []
    for img in ds['X_images']:
        s0, mask = make_S0_nd(img, D, cfg.N_MAX)
        S0_X_list.append(s0)
        mask_X_list.append(mask)

    return (jnp.stack(S0_O_list), jnp.stack(mask_O_list),
            jnp.stack(S0_X_list), jnp.stack(mask_X_list))


def run_adaptive_training(trial_id=1, cfg=None, **cfg_overrides):
    """
    Main Block III adaptive training loop.

    Returns: (final_params, D_final, K_final, trial_manager)
    """
    if cfg is None:
        cfg = Config(**cfg_overrides)

    trial = TrialManager(trial_id, cfg)
    trial.save_config()
    trial.save_dataset()

    # Current state
    D = cfg.D_INIT
    K = cfg.K_INIT

    # Initialize params
    w_frozen, mu_frozen, sigma_frozen = make_frozen_params(D, cfg)
    params = make_initial_learnable(D, cfg)

    # Build simulators and preprocessors for current D
    simulate_diff = make_simulate_diff(D, cfg.GAMMA, cfg.DT, cfg.N_STEPS)
    simulate_eval = make_simulate_eval(D, cfg.GAMMA, cfg.DT, cfg.N_STEPS)

    # Training data
    print(f"\n[Dataset] Generating D={D} training data...")
    train_S0_O, train_mask_O, train_S0_X, train_mask_X = \
        prepare_training_data(D, cfg)
    n_train = train_S0_O.shape[0]

    # Canonical data for validation
    S0_O_canon, mask_O = make_S0_nd(O_CANONICAL, D, cfg.N_MAX)
    S0_X_canon, mask_X = make_S0_nd(X_CANONICAL, D, cfg.N_MAX)
    qO = jnp.array(cfg.q_star('O', D))
    qX = jnp.array(cfg.q_star('X', D))

    # Closures for current D
    def get_full_params(p):
        return full_params_fn(p, w_frozen, mu_frozen, sigma_frozen)

    def loss_fn(p, s0_O, mask_O_arg, s0_X, mask_X_arg):
        w, mu, sigma = get_full_params(p)
        traj_O  = simulate_diff(s0_O, w, mu, sigma)
        com_O_T = com_single(traj_O[-1, :, :D], mask_O_arg)
        traj_X  = simulate_diff(s0_X, w, mu, sigma)
        com_X_T = com_single(traj_X[-1, :, :D], mask_X_arg)
        loss_q = (jnp.sum((com_O_T - qO) ** 2) +
                  jnp.sum((com_X_T - qX) ** 2))
        mO_f = mask_O_arg.astype(jnp.float32)
        mX_f = mask_X_arg.astype(jnp.float32)
        pnorm_O = jnp.sum(jnp.sum(traj_O[-1, :, D:2*D] ** 2, axis=-1) * mO_f) / jnp.sum(mO_f)
        pnorm_X = jnp.sum(jnp.sum(traj_X[-1, :, D:2*D] ** 2, axis=-1) * mX_f) / jnp.sum(mX_f)
        return loss_q + cfg.LAMBDA_P * (pnorm_O + pnorm_X), (loss_q, pnorm_O + pnorm_X, com_O_T, com_X_T)

    # Print header
    print(f"\n{'='*62}")
    print(f"BLOCK III ADAPTIVE TRAINING  (trial={trial_id})")
    print(f"{'='*62}")
    print(f"  D={D}  K={K}  D_max={cfg.D_MAX}  K_max={cfg.K_MAX}")
    print(f"  Epochs={cfg.N_EPOCHS}  peak_lr={cfg.PEAK_LR}")
    print(f"  Plateau: window={cfg.PLATEAU_WINDOW}  threshold={cfg.PLATEAU_THRESHOLD}")
    print(f"  Dataset: {n_train} variants/class")
    print(f"{'='*62}\n")

    # Optimizer + training step (rebuilt on growth)
    def make_optimizer_and_step():
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=cfg.PEAK_LR,
            warmup_steps=cfg.WARMUP_STEPS, decay_steps=cfg.N_EPOCHS,
            end_value=cfg.END_LR)
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=schedule))

        @jit
        def train_step(p, opt_state, s0_O, mask_O_arg, s0_X, mask_X_arg):
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                p, s0_O, mask_O_arg, s0_X, mask_X_arg)
            updates, new_opt = optimizer.update(grads, opt_state, p)
            new_p = optax.apply_updates(p, updates)
            return new_p, new_opt, loss, aux

        return optimizer, train_step

    optimizer, train_step = make_optimizer_and_step()
    opt_state = optimizer.init(params)
    plateau = PlateauDetector(cfg.PLATEAU_WINDOW, cfg.PLATEAU_THRESHOLD)
    rng = np.random.RandomState(123)

    epoch_converged = None
    last_growth_epoch = -cfg.COOLDOWN_AFTER_GROW  # allow immediate first check
    t_start = time.time()

    history = {
        'epoch': [], 'loss': [], 'D': [], 'K': [],
        'com_O': [], 'com_X': [], 'pred_O': [], 'pred_X': [],
    }

    for epoch in range(cfg.N_EPOCHS):
        # Sample random training pair
        idx_O = rng.randint(n_train)
        idx_X = rng.randint(n_train)

        params, opt_state, loss_val, aux = train_step(
            params, opt_state,
            train_S0_O[idx_O], train_mask_O[idx_O],
            train_S0_X[idx_X], train_mask_X[idx_X])
        loss_q, loss_p, com_O_T, com_X_T = aux

        plateau.update(float(loss_val))

        # Logging
        if epoch % cfg.LOG_EVERY == 0:
            # Validate on canonical
            w_v, mu_v, sig_v = get_full_params(params)
            traj_O_v = simulate_eval(S0_O_canon, w_v, mu_v, sig_v)
            traj_X_v = simulate_eval(S0_X_canon, w_v, mu_v, sig_v)
            com_O_v = com_single(traj_O_v[-1, :, :D], mask_O)
            com_X_v = com_single(traj_X_v[-1, :, :D], mask_X)
            dOO = float(jnp.linalg.norm(com_O_v - qO))
            dXX = float(jnp.linalg.norm(com_X_v - qX))
            dOX = float(jnp.linalg.norm(com_O_v - qX))
            dXO = float(jnp.linalg.norm(com_X_v - qO))
            pred_O = 'O' if dOO < dOX else 'X'
            pred_X = 'O' if dXO < dXX else 'X'

            history['epoch'].append(epoch)
            history['loss'].append(float(loss_val))
            history['D'].append(D)
            history['K'].append(K)
            history['com_O'].append([float(com_O_v[i]) for i in range(D)])
            history['com_X'].append([float(com_X_v[i]) for i in range(D)])
            history['pred_O'].append(pred_O)
            history['pred_X'].append(pred_X)

            print(f"Ep {epoch:4d} [D={D},K={K}] | "
                  f"L={float(loss_val):.4f} "
                  f"(q={float(loss_q):.3f} p={float(loss_p):.4f}) | "
                  f"pred={pred_O},{pred_X} | "
                  f"eq=({dOO:.2f},{dXX:.2f})")

            # Early stopping
            if epoch > 0 and pred_O == 'O' and pred_X == 'X' \
                    and dOO < cfg.CONV_Q_THR and dXX < cfg.CONV_Q_THR:
                epoch_converged = epoch
                print(f"\n  *** Converged at epoch {epoch}! ***")
                break

        # Growth check
        if (epoch >= cfg.MIN_EPOCHS_BEFORE_GROW and
                epoch - last_growth_epoch >= cfg.COOLDOWN_AFTER_GROW and
                plateau.is_plateau()):

            before = {'D': D, 'K': K}

            if K < cfg.K_MAX:
                # Try K growth first
                params, K = grow_K(params, w_frozen, mu_frozen, sigma_frozen,
                                   D, K, cfg.K_GROW, cfg)
                trial.log_growth(epoch, 'K_grow', before, {'D': D, 'K': K})
            elif D < cfg.D_MAX:
                # K maxed out, grow D
                D_new = D + 1
                params, w_frozen, mu_frozen, sigma_frozen = grow_D(
                    params, w_frozen, mu_frozen, sigma_frozen, D, D_new, cfg)
                D = D_new

                # Rebuild everything for new D
                simulate_diff = make_simulate_diff(D, cfg.GAMMA, cfg.DT, cfg.N_STEPS)
                simulate_eval = make_simulate_eval(D, cfg.GAMMA, cfg.DT, cfg.N_STEPS)

                print(f"  [Rebuilding] Training data for D={D}...")
                train_S0_O, train_mask_O, train_S0_X, train_mask_X = \
                    prepare_training_data(D, cfg)
                S0_O_canon, mask_O = make_S0_nd(O_CANONICAL, D, cfg.N_MAX)
                S0_X_canon, mask_X = make_S0_nd(X_CANONICAL, D, cfg.N_MAX)
                qO = jnp.array(cfg.q_star('O', D))
                qX = jnp.array(cfg.q_star('X', D))

                trial.log_growth(epoch, 'D_grow', before, {'D': D, 'K': K})
            else:
                print(f"  [GROW] D={D} and K={K} both at max. No growth possible.")
                continue

            # Rebuild optimizer (fresh state for new params)
            optimizer, train_step = make_optimizer_and_step()
            opt_state = optimizer.init(params)
            plateau = PlateauDetector(cfg.PLATEAU_WINDOW, cfg.PLATEAU_THRESHOLD)
            last_growth_epoch = epoch

        # Checkpoint
        if epoch % cfg.SAVE_EVERY == 0 and epoch > 0:
            trial.save_params(params, D, K, epoch, float(loss_val))

    t_total = time.time() - t_start
    print(f"\n[Training done] {t_total:.1f}s  D={D}  K={K}")

    # Save final
    trial.save_params(params, D, K, epoch, float(loss_val))
    trial.save_growth_log()

    # Save history
    hist_path = trial.output_dir / f"{trial.tag()}_history.json"
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  [History] {hist_path}")

    return params, D, K, w_frozen, mu_frozen, sigma_frozen, trial, history


# ===========================================================================
# SECTION 9: Block III Evaluation
# ===========================================================================

def run_block3_evaluation(params, D, K, w_frozen, mu_frozen, sigma_frozen,
                          trial, cfg=None):
    """Run the standard evaluation suite on Block III trained params."""
    if cfg is None:
        cfg = Config()

    from evaluation import run_standard_eval

    qO = jnp.array(cfg.q_star('O', D))
    qX = jnp.array(cfg.q_star('X', D))

    # Preprocessor and simulator for this D
    preprocess_fn = lambda img: make_S0_nd(img, D, cfg.N_MAX)
    simulate_fn = make_simulate_eval(D, cfg.GAMMA, cfg.DT, cfg.N_STEPS)

    _, mask_O = make_S0_nd(O_CANONICAL, D, cfg.N_MAX)
    _, mask_X = make_S0_nd(X_CANONICAL, D, cfg.N_MAX)

    def get_full_params(p):
        return full_params_fn(p, w_frozen, mu_frozen, sigma_frozen)

    # Novel patterns
    from evaluation_system.evaluate_block_ii import NOVEL_IMAGES

    def dataset_fn():
        return generate_dataset(n_per_class=20, seed=99)

    _, all_pass, results = run_standard_eval(
        params=params,
        full_params_fn=get_full_params,
        rbf_potential_fn=rbf_potential_nd,
        preprocess_fn=preprocess_fn,
        simulate_fn=simulate_fn,
        O_image=O_CANONICAL,
        X_image=X_CANONICAL,
        qO=qO,
        qX=qX,
        mask_O=mask_O,
        mask_X=mask_X,
        novel_images=NOVEL_IMAGES,
        dataset_fn=dataset_fn,
        title_prefix=f"Block III ({trial.tag()}) D={D} K={K}",
        output_path=str(trial.output_dir / f"{trial.tag()}_eval_D{D}_K{K}.png"),
        gamma_default=cfg.GAMMA,
        T_final=cfg.T_FINAL,
        dt=cfg.DT,
        K=K,
        N_steps=cfg.N_STEPS,
    )

    return all_pass, results


# ===========================================================================
# SECTION 10: Entry Point
# ===========================================================================

if __name__ == "__main__":
    cfg = Config(
        N_EPOCHS=3000,
        PEAK_LR=5e-3,
        D_INIT=3,
        D_MAX=5,
        K_INIT=16,
        K_MAX=32,
        K_GROW=4,
        PLATEAU_WINDOW=100,
        PLATEAU_THRESHOLD=0.01,
        MIN_EPOCHS_BEFORE_GROW=200,
        COOLDOWN_AFTER_GROW=100,
    )

    # Adaptive training
    params, D, K, w_frozen, mu_frozen, sigma_frozen, trial, history = \
        run_adaptive_training(trial_id=1, cfg=cfg)

    # Evaluation
    print("\n" + "=" * 62)
    print("BLOCK III EVALUATION")
    print("=" * 62)
    all_pass, results = run_block3_evaluation(
        params, D, K, w_frozen, mu_frozen, sigma_frozen, trial, cfg)

    print(f"\n  OVERALL: {'*** ALL PASS ***' if all_pass else '--- PARTIAL ---'}")
    print(f"  Final: D={D}  K={K}")
