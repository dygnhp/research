"""
=============================================================================
Contact Hamiltonian Fluid Neural Network -- Block I Forward Simulator
=============================================================================
Physics:   Contact Hamiltonian dynamics with RBF potential landscape
Math:      dq/dt = p,  dp/dt = -grad V - gamma*p,  dz/dt = ||p||^2 - H
Backend:   JAX (CUDA) -- JIT-compiled RK4 via jax.lax.scan
Target HW: Windows 11 + NVIDIA GPU (RTX 4060) + Intel CPU via WSL2

Install (WSL2 with CUDA 12):
    pip install --upgrade "jax[cuda12]" matplotlib scipy

NOTE on physics parameters (K=16):
  K=16 structured initialization addresses the core failure of K=4:
  Both O and X images share the same global CoM ≈ (3.5, 3.5), so a
  global attractor cannot distinguish them. Discrimination requires
  per-quadrant sensitivity to intra-quadrant pixel distribution:
    O-image: arc pixels at quadrant corners → compact covariance
    X-image: diagonal pixels across quadrant → elongated covariance
  k=3..6  (O-discriminators): attract O arc pixels, widening O-X gap
  k=7..10 (X-discriminators): repel O arc pixels from X-diagonal regions
  k=13..15 (reserve): free parameters for Block II to reshape.

NOTE on phase-volume computation:
  At t=0, all p_i=0 -> p-subspace has zero variance -> 6D covariance is
  rank-deficient. We find the first "active" timestep t_ref and normalize
  from there, comparing to exp(-3*gamma*(t - t_ref)).
=============================================================================
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"

import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time

print("=" * 60)
print("JAX version  :", jax.__version__)
print("Devices      :", jax.devices())
print("Default backend:", jax.default_backend())
print("=" * 60)

# ---------------------------------------------------------------------------
# 1. Global constants
# ---------------------------------------------------------------------------
GAMMA    = 1.5
T_FINAL  = 10.0
DT       = 0.05
N_STEPS  = int(T_FINAL / DT)   # 200
N_MAX    = 64
TAU      = 0.5

Q_STAR_O = jnp.array([ 8.0,  8.0, 0.0])
Q_STAR_X = jnp.array([-8.0, -8.0, 0.0])

# ── K=16 structured initialization ─────────────────────────────────────
# Role assignment:
#   k=0      : O-class attractor  (deep well at O target)
#   k=1      : X-class attractor  (deep well at X target)
#   k=2      : Central barrier    (repulsor separating two classes)
#   k=3..6   : O-class discriminator RBFs, one per quadrant
#              Placed at O arc pixel centroids — attract O particles
#   k=7..10  : X-class discriminator RBFs, one per quadrant
#              Placed at X diagonal pixel centroids — repel O, attract X
#   k=11     : O-class path guide (pulls O particles toward O attractor)
#   k=12     : X-class path guide (pulls X particles toward X attractor)
#   k=13..15 : Reserve RBFs (wide sigma, small w — let Block II reshape these)
# ────────────────────────────────────────────────────────────────────────

W_INIT = jnp.array([
    # Attractors
    -2.0,   # k=0  O-attractor
    -2.0,   # k=1  X-attractor
    # Barrier
    +2.5,   # k=2  central barrier (strengthened from +1.5)
    # O-class discriminators (attract O-arc pixels)
    -1.0,   # k=3  Q1 O-arc  (upper-left quadrant)
    -1.0,   # k=4  Q2 O-arc  (upper-right quadrant)
    -1.0,   # k=5  Q3 O-arc  (lower-left quadrant)
    -1.0,   # k=6  Q4 O-arc  (lower-right quadrant)
    # X-class discriminators (repel O, attract X diagonal pixels)
    +1.2,   # k=7  Q1 X-diag repulsor
    +1.2,   # k=8  Q2 X-diag repulsor
    +1.2,   # k=9  Q3 X-diag repulsor
    +1.2,   # k=10 Q4 X-diag repulsor
    # Path guides
    -0.6,   # k=11 O path guide
    -0.6,   # k=12 X path guide
    # Reserve (small weight, wide spread)
    -0.3,   # k=13
    -0.3,   # k=14
    -0.3,   # k=15
])

MU_INIT = jnp.array([
    # k=0  O-attractor
    [ 8.0,  8.0,  0.5],
    # k=1  X-attractor
    [-8.0, -8.0,  0.5],
    # k=2  Central barrier
    [ 0.0,  0.0,  0.5],
    # k=3..6  O-arc discriminators
    # Placed at centroid of O arc pixels in each quadrant
    # Q1(col 0-3, row 4-7): O arc at (2,6),(3,6),(1,5),(1,4) -> centroid~=(1.75,5.25)
    [ 1.75,  5.25,  0.5],   # k=3  Q1
    # Q2(col 4-7, row 4-7): O arc at (4,6),(5,6),(6,5),(6,4) -> centroid~=(5.25,5.25)
    [ 5.25,  5.25,  0.5],   # k=4  Q2
    # Q3(col 0-3, row 0-3): O arc at (1,2),(1,3),(2,1),(3,1) -> centroid~=(1.75,1.75)
    [ 1.75,  1.75,  0.5],   # k=5  Q3
    # Q4(col 4-7, row 0-3): O arc at (6,3),(6,2),(5,1),(4,1) -> centroid~=(5.25,1.75)
    [ 5.25,  1.75,  0.5],   # k=6  Q4
    # k=7..10  X-diagonal repulsors
    # Placed at centroid of X diagonal pixels in each quadrant
    # Q1 X-diag: (0,7),(1,6),(2,5),(3,4) -> centroid=(1.5,5.5)
    [ 1.5,   5.5,   0.5],   # k=7  Q1
    # Q2 X-diag: (7,7),(6,6),(5,5),(4,4) -> centroid=(5.5,5.5)
    [ 5.5,   5.5,   0.5],   # k=8  Q2
    # Q3 X-diag: (3,3),(2,2),(1,1),(0,0) -> centroid=(1.5,1.5)
    [ 1.5,   1.5,   0.5],   # k=9  Q3
    # Q4 X-diag: (4,3),(5,2),(6,1),(7,0) -> centroid=(5.5,1.5)
    [ 5.5,   1.5,   0.5],   # k=10 Q4
    # k=11  O path guide: midpoint O-start->O-attractor
    [ 5.0,   5.0,   0.5],   # k=11
    # k=12  X path guide: midpoint X-start->X-attractor (note: start~=(3.5,3.5))
    [-2.0,  -2.0,   0.5],   # k=12
    # k=13..15  Reserve RBFs (evenly spread, large sigma)
    [ 0.0,   6.0,   0.5],   # k=13
    [ 6.0,   0.0,   0.5],   # k=14
    [-4.0,   4.0,   0.5],   # k=15
], dtype=jnp.float32)

SIGMA_INIT = jnp.array([
    2.0,   # k=0  O-attractor
    2.0,   # k=1  X-attractor
    2.5,   # k=2  barrier (slightly wider)
    1.5,   # k=3  Q1 O-discriminator (narrow: must be precise)
    1.5,   # k=4  Q2 O-discriminator
    1.5,   # k=5  Q3 O-discriminator
    1.5,   # k=6  Q4 O-discriminator
    1.5,   # k=7  Q1 X-discriminator (narrow: must be precise)
    1.5,   # k=8  Q2 X-discriminator
    1.5,   # k=9  Q3 X-discriminator
    1.5,   # k=10 Q4 X-discriminator
    2.5,   # k=11 O path guide
    2.5,   # k=12 X path guide
    4.0,   # k=13 reserve (wide)
    4.0,   # k=14 reserve (wide)
    4.0,   # k=15 reserve (wide)
])

# ---------------------------------------------------------------------------
# 2. Test images
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------
def preprocess(image, tau=TAU, n_max=N_MAX):
    """
    Convert 8x8 image to JAX state arrays.
    Coordinate convention: row r, col c -> x=c, y=7-r (y increases upward)
    State per particle: [q_x, q_y, q_z(intensity), p_x, p_y, p_z, z_contact]
    Note: q[2] = pixel intensity (initial z-position lift)
          S[6]  = contact variable z_i(t) (dissipated energy), starts at 0
    """
    rows, cols = image.shape
    q_list = []
    for r in range(rows):
        for c in range(cols):
            if image[r, c] > tau:
                q_list.append([float(c), float(rows-1-r), float(image[r,c])])

    n_real = len(q_list)
    assert n_real > 0, "No real particles -- check tau"
    assert n_real <= n_max

    q0_np   = np.zeros((n_max, 3), dtype=np.float32)
    p0_np   = np.zeros((n_max, 3), dtype=np.float32)
    z0_np   = np.zeros((n_max,),   dtype=np.float32)
    mask_np = np.zeros((n_max,),   dtype=bool)

    for i, pos in enumerate(q_list):
        q0_np[i] = pos
        mask_np[i] = True

    q0   = jnp.array(q0_np)
    p0   = jnp.array(p0_np)
    z0   = jnp.array(z0_np)
    mask = jnp.array(mask_np)
    assert q0.shape == (n_max, 3)
    assert p0.shape == (n_max, 3)
    assert int(mask.sum()) > 0

    print(f"  Preprocessed: {n_real} real / {n_max} total")
    return q0, p0, z0, mask

# ---------------------------------------------------------------------------
# 4. RBF Potential and Gradient
# ---------------------------------------------------------------------------
@jit
def rbf_potential(q, w, mu, sigma):
    """
    V(q_i) = sum_k w_k * exp(-||q_i - mu_k||^2 / (2*sigma_k^2))
    q:(N,3) w:(K,) mu:(K,3) sigma:(K,) -> V:(N,)
    """
    diff    = q[:, None, :] - mu[None, :, :]           # (N, K, 3)
    sq_dist = jnp.sum(diff**2, axis=-1)                 # (N, K)
    gauss   = jnp.exp(-sq_dist / (2.0 * sigma**2))     # (N, K)
    return jnp.sum(w * gauss, axis=-1)                  # (N,)


@jit
def rbf_gradient(q, w, mu, sigma):
    """
    grad_{q_i} V = sum_k w_k * exp(...) * (-(q_i - mu_k) / sigma_k^2)
    q:(N,3) -> grad:(N,3)
    """
    diff    = q[:, None, :] - mu[None, :, :]
    sq_dist = jnp.sum(diff**2, axis=-1)
    gauss   = jnp.exp(-sq_dist / (2.0 * sigma**2))
    factor  = w * gauss / (sigma**2)                   # (N, K)
    return jnp.sum(-factor[:, :, None] * diff, axis=1) # (N, 3)

# ---------------------------------------------------------------------------
# 5. Contact Hamiltonian RHS
# ---------------------------------------------------------------------------
@jit
def contact_rhs(S, w, mu, sigma, gamma):
    """
    Contact Hamilton's equations:
        dq/dt = p
        dp/dt = -grad_q V(q) - gamma * p
        dz/dt = ||p||^2 - H_i    where H_i = ||p||^2/2 + V(q)

    Physical interpretation:
        dH_i/dt = -gamma * ||p_i||^2 <= 0  (energy monotone decrease)
        div(X_contact) = -gamma per DoF  --> V_phase ~ exp(-3*gamma*t)
        This intentionally BREAKS Liouville's theorem (Contact geometry).

    S:(N_max, 7)  gamma: Python float
    S layout: [:, 0:3]=q  [:, 3:6]=p  [:, 6]=z_contact
    """
    q      = S[:, :3]
    p      = S[:, 3:6]
    V      = rbf_potential(q, w, mu, sigma)          # (N,)
    gV     = rbf_gradient(q, w, mu, sigma)           # (N, 3)
    p_sq   = jnp.sum(p**2, axis=-1)                 # (N,)
    H_i    = p_sq / 2.0 + V                          # (N,)

    dq_dt  = p
    dp_dt  = -gV - gamma * p
    dz_dt  = p_sq - H_i

    return jnp.concatenate([dq_dt, dp_dt, dz_dt[:, None]], axis=-1)

# ---------------------------------------------------------------------------
# 6. RK4 Step
# ---------------------------------------------------------------------------
@partial(jit, static_argnums=(4, 5))
def rk4_step(S, w, mu, sigma, gamma, dt):
    """
    Classical RK4: S_new = S + (dt/6)(k1 + 2k2 + 2k3 + k4)
    gamma, dt are Python scalars (static) -- avoids re-tracing in lax.scan.
    """
    f  = partial(contact_rhs, w=w, mu=mu, sigma=sigma, gamma=gamma)
    k1 = f(S)
    k2 = f(S + 0.5*dt*k1)
    k3 = f(S + 0.5*dt*k2)
    k4 = f(S +     dt*k3)
    return S + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)

# ---------------------------------------------------------------------------
# 7. Full Simulation via jax.lax.scan
# ---------------------------------------------------------------------------
def simulate(image, w, mu, sigma, gamma=GAMMA, T=T_FINAL, dt=DT, tau=TAU):
    """
    Full simulation using lax.scan for GPU-compiled time loop.
    Returns trajectory (N_steps+1, N_max, 7) and mask (N_max,).
    """
    print("  Preprocessing...")
    q0, p0, z0, mask = preprocess(image, tau=tau)
    S0      = jnp.concatenate([q0, p0, z0[:, None]], axis=-1)  # (N_max, 7)
    n_steps = int(round(T / dt))

    @jit
    def step(S, _):
        S_new = rk4_step(S, w, mu, sigma, gamma, dt)
        return S_new, S_new

    print(f"  lax.scan: {n_steps} RK4 steps (JIT-compiled for GPU)...")
    t0 = time.time()
    _, steps = jax.lax.scan(step, S0, None, length=n_steps)
    steps.block_until_ready()
    print(f"  Done in {time.time()-t0:.3f}s")

    trajectory = jnp.concatenate([S0[None], steps], axis=0)  # (N_steps+1, N_max, 7)

    if bool(jnp.any(~jnp.isfinite(trajectory))):
        bad = jnp.where(
            ~jnp.all(jnp.isfinite(trajectory.reshape(n_steps+1,-1)), axis=-1))[0]
        raise ValueError(f"NaN/Inf at steps: {bad[:5]}")
    print("  Sanity: all finite.")
    return trajectory, mask

# ---------------------------------------------------------------------------
# 8. Analysis Functions
# ---------------------------------------------------------------------------
@jit
def com_single(q_t, mask):
    """CoM of real particles at one timestep. Returns shape (3,)."""
    mf = mask.astype(jnp.float32)
    return jnp.sum(q_t * mf[:, None], axis=0) / jnp.sum(mf)


def compute_com(trajectory, mask):
    """Returns (N_steps+1, 3) CoM trajectory."""
    fn = vmap(lambda qt: com_single(qt, mask))
    return fn(trajectory[:, :, :3])


def compute_hamiltonian(trajectory, mask, w, mu, sigma):
    """
    H_i(t) = ||p_i||^2/2 + V(q_i) for all real particles.
    Theory: dH_i/dt = -gamma * ||p_i||^2 <= 0.
    Returns (N_steps+1, N_real).
    """
    m_np = np.array(mask, dtype=bool)
    H_list = []
    for t in range(trajectory.shape[0]):
        q_t = trajectory[t, :, :3]
        p_t = trajectory[t, :, 3:6]
        H_t = jnp.sum(p_t**2, axis=-1)/2.0 + rbf_potential(q_t, w, mu, sigma)
        H_list.append(np.array(H_t[m_np]))
    return np.stack(H_list, axis=0)  # (N_steps+1, N_real)


def compute_phase_volume(trajectory, mask):
    """
    Phase-space volume via 6D covariance determinant:
        V_phase(t) = sqrt(det(Cov([q(t), p(t)])))

    Handles p(0)=0 singularity:
        Finds first "active" t_ref where det > 0.1% of max,
        returns ratio normalized from t_ref.

    Returns: (vol_ratio, t_ref_idx)
    """
    m_np  = np.array(mask, dtype=bool)
    n_t   = trajectory.shape[0]
    eps   = 1e-10
    v_arr = np.zeros(n_t, dtype=np.float64)

    for t in range(n_t):
        q_r = np.array(trajectory[t, m_np, :3])
        p_r = np.array(trajectory[t, m_np, 3:6])
        jnt = np.concatenate([q_r, p_r], axis=1)
        if jnt.shape[0] < 2:
            continue
        cov     = np.cov(jnt.T) + eps*np.eye(6)
        v_arr[t] = np.sqrt(max(np.linalg.det(cov), 0.0))

    v_max = v_arr.max()
    if v_max < 1e-30:
        return np.ones(n_t), 0

    active    = np.where(v_arr > 0.001 * v_max)[0]
    t_ref_idx = int(active[0]) if len(active) else 0
    v_ref     = v_arr[t_ref_idx]
    ratio     = np.where(v_ref > 0, v_arr / v_ref, 0.0)
    return ratio, t_ref_idx


def classify(trajectory, mask, qO=Q_STAR_O, qX=Q_STAR_X):
    """Classify by final CoM distance to each attractor."""
    qf  = com_single(trajectory[-1, :, :3], mask)
    dO  = float(jnp.linalg.norm(qf - qO))
    dX  = float(jnp.linalg.norm(qf - qX))
    return ('O' if dO < dX else 'X'), dO, dX, qf

# ---------------------------------------------------------------------------
# 9. Verification and 6-Panel Figure
# ---------------------------------------------------------------------------
def verify_and_plot(traj_O, mask_O, traj_X, mask_X,
                    w, mu, sigma, t_arr,
                    output_path="block1_verification.png"):

    print("\nComputing verification quantities...")
    com_O = np.array(compute_com(traj_O, mask_O))
    com_X = np.array(compute_com(traj_X, mask_X))

    print("  H(t)...")
    H_O = compute_hamiltonian(traj_O, mask_O, w, mu, sigma)
    H_X = compute_hamiltonian(traj_X, mask_X, w, mu, sigma)

    print("  V_phase(t)...")
    vol_O, t_ref_O = compute_phase_volume(traj_O, mask_O)
    vol_X, t_ref_X = compute_phase_volume(traj_X, mask_X)

    # Theory curves from t_ref (where p-distribution first has spread)
    thr_O = np.exp(-3.0*GAMMA*(t_arr - t_arr[t_ref_O]))
    thr_X = np.exp(-3.0*GAMMA*(t_arr - t_arr[t_ref_X]))
    thr_O[:t_ref_O] = 1.0
    thr_X[:t_ref_X] = 1.0

    # Convergence
    mO = np.array(mask_O, dtype=bool)
    mX = np.array(mask_X, dtype=bool)
    eps_q_O = np.linalg.norm(com_O - np.array(Q_STAR_O), axis=-1)
    eps_q_X = np.linalg.norm(com_X - np.array(Q_STAR_X), axis=-1)
    eps_p_O = np.mean(np.linalg.norm(np.array(traj_O[:, mO, 3:6]), axis=-1), axis=-1)
    eps_p_X = np.mean(np.linalg.norm(np.array(traj_X[:, mX, 3:6]), axis=-1), axis=-1)

    pred_O, dOO, dOX, cfO = classify(traj_O, mask_O)
    pred_X, dXO, dXX, cfX = classify(traj_X, mask_X)

    # --- PASS/FAIL ---
    def mono(H):
        dH = np.diff(H, axis=0)
        f  = 1.0 - np.sum(dH > 1e-6) / dH.size
        return f >= 0.95, f

    emO, fO = mono(H_O)
    emX, fX = mono(H_X)
    energy_pass = emO and emX

    def r2(yt, yp, s=0):
        yt = yt[s:]; yp = yp[s:]
        return 1.0 - np.sum((yt-yp)**2) / (np.sum((yt-yt.mean())**2) + 1e-12)

    r2O = r2(vol_O, thr_O, t_ref_O)
    r2X = r2(vol_X, thr_X, t_ref_X)
    vol_pass  = (r2O > 0.90) and (r2X > 0.90)
    conv_q    = (eps_q_O[-1] < 2.0) and (eps_q_X[-1] < 2.0)
    conv_p    = (eps_p_O[-1] < 0.5) and (eps_p_X[-1] < 0.5)
    conv_pass = conv_q and conv_p
    cls_pass  = (pred_O == 'O') and (pred_X == 'X')
    all_pass  = energy_pass and vol_pass and conv_pass and cls_pass

    # --- Report ---
    S = "=" * 62
    print(f"\n{S}\nBLOCK I VERIFICATION REPORT\n{S}")
    print(f"  [ENERGY]  Monotone: {'PASS' if energy_pass else 'FAIL'}  O={fO:.3%} X={fX:.3%}")
    print(f"  [VOLUME]  R^2:      {'PASS' if vol_pass else 'FAIL'}  O={r2O:.4f} X={r2X:.4f}")
    print(f"  [CONV q]  eps_q:    {'PASS' if conv_q else 'FAIL'}  O={eps_q_O[-1]:.4f} X={eps_q_X[-1]:.4f}")
    print(f"  [CONV p]  eps_p:    {'PASS' if conv_p else 'FAIL'}  O={eps_p_O[-1]:.4f} X={eps_p_X[-1]:.4f}")
    print(f"  [CLASS]   correct:  {'PASS' if cls_pass else 'FAIL'}  O->{pred_O}  X->{pred_X}")
    print(f"{S}")
    print(f"  OVERALL: {'*** ALL PASS ***' if all_pass else '--- PARTIAL (see notes) ---'}")
    if not cls_pass:
        print("\n  NOTE: Classification FAIL persists with K=16 structured init.")
        print("  Diagnostic — final CoM positions:")
        print(f"    O-image CoM: ({float(cfO[0]):.4f}, {float(cfO[1]):.4f}, {float(cfO[2]):.4f})")
        print(f"    X-image CoM: ({float(cfX[0]):.4f}, {float(cfX[1]):.4f}, {float(cfX[2]):.4f})")
        print(f"    O: dist(q*_O)={dOO:.4f}  dist(q*_X)={dOX:.4f}")
        print(f"    X: dist(q*_O)={dXO:.4f}  dist(q*_X)={dXX:.4f}")
        print("  Root cause: X-attractor at (-8,-8) is ~16 units from init particles.")
        print("  Gaussian support 3*sigma=6 -> force ~ exp(-16^2/8) ~ 0.")
        print("  Block II adjoint gradient descent will relocate mu_k to resolve this.")
    print(S)

    # --- Figure ---
    print("\nBuilding figure...")
    mu_np = np.array(mu)

    fig = plt.figure(figsize=(18, 10))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.35)
    a   = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]
    a11, a12, a13 = a[0]
    a21, a22, a23 = a[1]

    # [1,1] Potential contour
    xy = np.linspace(-12, 12, 300)
    gx, gy = np.meshgrid(xy, xy)
    q_g = jnp.array(np.stack([gx.ravel(), gy.ravel(),
                                np.full(gx.size, 0.5)], axis=1).astype(np.float32))
    Vg  = np.array(rbf_potential(q_g, w, mu, sigma)).reshape(gx.shape)
    cf  = a11.contourf(gx, gy, Vg, levels=40, cmap='RdYlBu_r', alpha=0.85)
    a11.contour(gx, gy, Vg, levels=20, colors='k', linewidths=0.3, alpha=0.4)
    plt.colorbar(cf, ax=a11, shrink=0.85, label='V(q)')
    # Plot special markers for named RBFs only (avoid legend clutter)
    special = [
        (0,  'blue',       '*', 220, 'O-attractor'),
        (1,  'red',        '*', 220, 'X-attractor'),
        (2,  'darkorange', '^', 160, 'Barrier'),
        (11, 'green',      'D', 110, 'O path guide'),
        (12, 'purple',     'D', 110, 'X path guide'),
    ]
    for k, col, mrk, sz, lab in special:
        a11.scatter(*mu_np[k, :2], s=sz, c=col, marker=mrk, zorder=6, label=lab)
    # Plot discriminator RBFs as small dots grouped by class
    for k in range(3, 7):
        a11.scatter(*mu_np[k, :2], s=55, c='cyan', marker='o',
                    zorder=5, label='O-discrim' if k==3 else '')
    for k in range(7, 11):
        a11.scatter(*mu_np[k, :2], s=55, c='orange', marker='s',
                    zorder=5, label='X-discrim' if k==7 else '')
    # Reserve RBFs: tiny grey dots, no label
    for k in range(13, 16):
        a11.scatter(*mu_np[k, :2], s=25, c='grey', marker='.', zorder=4, alpha=0.5)
    q0O = np.array(traj_O[0, mO, :2])
    q0X = np.array(traj_X[0, mX, :2])
    a11.scatter(q0O[:,0], q0O[:,1], s=18, c='blue', alpha=0.7, label='O init')
    a11.scatter(q0X[:,0], q0X[:,1], s=18, c='red',  alpha=0.7, label='X init')
    a11.set(xlim=(-12,12), ylim=(-12,12), xlabel='x', ylabel='y',
            title='RBF Potential Landscape (z=0.5 slice)')
    a11.legend(fontsize=6.5, loc='upper right', framealpha=0.8)

    # [1,2] Trajectories
    tOnp = np.array(traj_O[:, mO, :])
    tXnp = np.array(traj_X[:, mX, :])
    dec  = max(1, N_STEPS//50)
    for i in range(tOnp.shape[1]):
        a12.plot(tOnp[::dec,i,0], tOnp[::dec,i,1], 'b-', alpha=0.2, lw=0.6)
    for i in range(tXnp.shape[1]):
        a12.plot(tXnp[::dec,i,0], tXnp[::dec,i,1], 'r-', alpha=0.2, lw=0.6)
    a12.scatter(tOnp[0,:,0],  tOnp[0,:,1],  c='blue', s=12, marker='o', zorder=4)
    a12.scatter(tOnp[-1,:,0], tOnp[-1,:,1], c='blue', s=25, marker='x', zorder=5)
    a12.scatter(tXnp[0,:,0],  tXnp[0,:,1],  c='red',  s=12, marker='o', zorder=4)
    a12.scatter(tXnp[-1,:,0], tXnp[-1,:,1], c='red',  s=25, marker='x', zorder=5)
    a12.scatter(*np.array(Q_STAR_O)[:2], s=280, c='blue', marker='*', zorder=6, label='q*_O')
    a12.scatter(*np.array(Q_STAR_X)[:2], s=280, c='red',  marker='*', zorder=6, label='q*_X')
    a12.plot(com_O[::dec,0], com_O[::dec,1], 'b-', lw=2.5, label='CoM O')
    a12.plot(com_X[::dec,0], com_X[::dec,1], 'r-', lw=2.5, label='CoM X')
    a12.set(xlabel='q_x', ylabel='q_y', title='Particle Trajectories (xy-projection)')
    a12.legend(fontsize=8)

    # [1,3] H(t)
    for i in range(H_O.shape[1]):
        a13.plot(t_arr, H_O[:,i], 'b-', alpha=0.07, lw=0.5)
    for i in range(H_X.shape[1]):
        a13.plot(t_arr, H_X[:,i], 'r-', alpha=0.07, lw=0.5)
    a13.plot(t_arr, H_O.mean(1), 'b-', lw=2.2, label=f'<H> O  ({fO:.2%})')
    a13.plot(t_arr, H_X.mean(1), 'r-', lw=2.2, label=f'<H> X  ({fX:.2%})')
    a13.set(xlabel='t', ylabel='H(t)', title='Hamiltonian H(t) -- Monotone Decrease')
    a13.legend(fontsize=9)
    a13.text(0.97, 0.97, 'PASS' if energy_pass else 'FAIL',
             transform=a13.transAxes, ha='right', va='top', fontsize=13,
             fontweight='bold', color='green' if energy_pass else 'red')

    # [2,1] Phase-space volume
    a21.fill_between(t_arr[:t_ref_O+1], 0, vol_O[:t_ref_O+1], alpha=0.1, color='blue')
    a21.fill_between(t_arr[:t_ref_X+1], 0, vol_X[:t_ref_X+1], alpha=0.1, color='red')
    a21.plot(t_arr, vol_O, 'b-',  lw=2,   label='Empirical O')
    a21.plot(t_arr, vol_X, 'r-',  lw=2,   label='Empirical X')
    a21.plot(t_arr, thr_O, 'b--', lw=1.5, alpha=0.7, label='Theory O')
    a21.plot(t_arr, thr_X, 'r--', lw=1.5, alpha=0.7, label='Theory X')
    a21.set(xlabel='t', ylabel=r'$V_{phase}(t)/V_{phase}(t_{ref})$',
            title=r'Phase-Space Volume: $e^{-3\gamma(t-t_{ref})}$')
    a21.legend(fontsize=8, ncol=2)
    a21.text(0.05, 0.15, f"R2 O={r2O:.3f}\nR2 X={r2X:.3f}",
             transform=a21.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    a21.text(0.97, 0.97, 'PASS' if vol_pass else 'FAIL',
             transform=a21.transAxes, ha='right', va='top', fontsize=13,
             fontweight='bold', color='green' if vol_pass else 'red')

    # [2,2] Convergence
    a22.semilogy(t_arr, eps_q_O+1e-6, 'b-',  lw=2,   label=r'$\varepsilon_q$ O')
    a22.semilogy(t_arr, eps_q_X+1e-6, 'r-',  lw=2,   label=r'$\varepsilon_q$ X')
    a22.semilogy(t_arr, eps_p_O+1e-6, 'b--', lw=1.5, label=r'$\varepsilon_p$ O')
    a22.semilogy(t_arr, eps_p_X+1e-6, 'r--', lw=1.5, label=r'$\varepsilon_p$ X')
    a22.axhline(2.0, color='gray', ls=':',  lw=1, label='eps_q thr')
    a22.axhline(0.5, color='gray', ls='-.', lw=1, label='eps_p thr')
    a22.set(xlabel='t', ylabel='Error (log scale)',
            title=r'Convergence: $\varepsilon_q$ (solid), $\varepsilon_p$ (dashed)')
    a22.legend(fontsize=8, ncol=2)
    a22.text(0.97, 0.97, 'PASS' if conv_pass else 'FAIL',
             transform=a22.transAxes, ha='right', va='top', fontsize=13,
             fontweight='bold', color='green' if conv_pass else 'red')

    # [2,3] Summary
    bg = '#d4edda' if all_pass else '#fff3cd'
    a23.set_facecolor(bg); a23.axis('off')
    a23.set(xlim=(0,1), ylim=(0,1))
    lns = [
        "BLOCK I VERIFICATION SUMMARY",
        "=" * 34,
        "",
        "O-image Final CoM:",
        f"  q = ({float(cfO[0]):.3f}, {float(cfO[1]):.3f}, {float(cfO[2]):.3f})",
        f"  dist(q*_O) = {dOO:.4f}",
        f"  dist(q*_X) = {dOX:.4f}",
        f"  pred = {pred_O}  true = O  ({'OK' if pred_O=='O' else 'FAIL'})",
        "",
        "X-image Final CoM:",
        f"  q = ({float(cfX[0]):.3f}, {float(cfX[1]):.3f}, {float(cfX[2]):.3f})",
        f"  dist(q*_O) = {dXO:.4f}",
        f"  dist(q*_X) = {dXX:.4f}",
        f"  pred = {pred_X}  true = X  ({'OK' if pred_X=='X' else 'FAIL'})",
        "",
        "Verification:",
        f"  Energy mono : {'PASS' if energy_pass else 'FAIL'}",
        f"  Phase vol R2: {'PASS' if vol_pass else 'FAIL'}  O={r2O:.3f} X={r2X:.3f}",
        f"  Conv eps_q  : {'PASS' if conv_q else 'FAIL'}",
        f"  Conv eps_p  : {'PASS' if conv_p else 'FAIL'}",
        f"  Classification: {'PASS' if cls_pass else 'FAIL'}",
        "",
        f"  OVERALL: {'ALL PASS' if all_pass else 'PARTIAL (Block II needed)'}",
        "",
        f"  gamma={GAMMA}  T={T_FINAL}  dt={DT}  K=16",
        f"  N_real O={int(mask_O.sum())}  X={int(mask_X.sum())}",
    ]
    a23.text(0.05, 0.97, "\n".join(lns), transform=a23.transAxes,
             fontsize=7.8, fontfamily='monospace', va='top')
    a23.set_title('Block I Verification Summary', fontweight='bold',
                  color='darkgreen' if all_pass else 'darkorange')

    fig.suptitle(
        "Contact Hamiltonian Fluid NN -- Block I Forward Simulator\n"
        f"gamma={GAMMA}  T={T_FINAL}  dt={DT}  K=16 RBF  [JAX lax.scan / CUDA]",
        fontsize=10.5, y=1.01)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Figure saved -> {output_path}")
    plt.close(fig)
    return all_pass

# ---------------------------------------------------------------------------
# 10. Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*62)
    print("CONTACT HAMILTONIAN FLUID NN -- BLOCK I (JAX/CUDA)")
    print("="*62)

    w     = W_INIT
    mu    = MU_INIT
    sigma = SIGMA_INIT
    t_arr = np.linspace(0.0, T_FINAL, N_STEPS+1)

    print("\n[1/2] O-image simulation")
    traj_O, mask_O = simulate(O_IMAGE, w, mu, sigma)

    print("\n[2/2] X-image simulation")
    traj_X, mask_X = simulate(X_IMAGE, w, mu, sigma)

    passed = verify_and_plot(traj_O, mask_O, traj_X, mask_X,
                             w, mu, sigma, t_arr,
                             output_path="block1_verification.png")

    print(f"\nFinal: {'PASS' if passed else 'PARTIAL'}")
    print("Figure: block1_verification.png")
    print("="*62)