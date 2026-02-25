"""
Block I: Contact Hamiltonian Fluid Neural Network — Forward Simulator
=====================================================================
Physics: Particles flow over an RBF potential landscape under contact
         Hamiltonian dynamics with damping γ. Phase-space volume contracts
         as exp(-3γt), and particles converge to class-specific attractors.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# TEST DATA
# ─────────────────────────────────────────────────────────────────

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

# ─────────────────────────────────────────────────────────────────
# RBF PARAMETERS (K=4, fixed for Block I)
# ─────────────────────────────────────────────────────────────────

W   = np.array([-2.0, -2.0,  1.5, -0.5])          # weights
MU  = np.array([
    [ 8.0,  8.0, 0.5],   # k=0: O-class attractor
    [-8.0, -8.0, 0.5],   # k=1: X-class attractor
    [ 0.0,  0.0, 0.5],   # k=2: barrier
    [ 4.0,  4.0, 0.5],   # k=3: path guide
])
SIGMA = np.array([2.0, 2.0, 2.0, 3.0])             # widths

# Target convergence points
Q_STAR_O = np.array([ 8.0,  8.0, 0.0])
Q_STAR_X = np.array([-8.0, -8.0, 0.0])

# Simulation parameters
GAMMA  = 1.5
T_END  = 10.0
DT     = 0.05
N_STEP = int(T_END / DT)   # 200
N_MAX  = 64


# ═══════════════════════════════════════════════════════════════════
# PART A: PREPROCESSING
# ═══════════════════════════════════════════════════════════════════

def preprocess(image, tau=0.5, N_max=64):
    """
    Convert an 8×8 binary image to particle initial conditions.

    Coordinate convention:
        row r, col c  →  x = c,  y = 7 - r   (y increases upward)
    This ensures O-image pixels cluster near upper-right → attractor (8,8,0).

    Returns:
        q0   : (N_max, 3)  initial positions  [x, y, intensity]
        p0   : (N_max, 3)  initial momenta    [0, 0, 0]
        z0   : (N_max,)    contact variable   [0, ...]
        mask : (N_max,)    bool, True for real particles
    """
    rows, cols = np.where(image > tau)
    xs = cols.astype(float)           # x = column index
    ys = (7 - rows).astype(float)     # y = flipped row  (y↑)
    intensities = image[rows, cols]   # pixel value (1.0 for binary)

    N_real = len(xs)
    assert N_real > 0, "No real particles found — check threshold"

    q0   = np.zeros((N_max, 3))
    p0   = np.zeros((N_max, 3))
    z0   = np.zeros(N_max)
    mask = np.zeros(N_max, dtype=bool)

    # Fill real particles
    q0[:N_real, 0] = xs
    q0[:N_real, 1] = ys
    q0[:N_real, 2] = intensities      # 3rd q-component = pixel intensity
    mask[:N_real]  = True

    # Dummy particles remain at origin with mask=False

    # Shape assertions
    assert q0.shape == (64, 3), "Initial position shape error"
    assert p0.shape == (64, 3), "Initial momentum shape error"
    assert mask.sum() > 0,       "No real particles found — check threshold"

    return q0, p0, z0, mask


# ═══════════════════════════════════════════════════════════════════
# PART B: RBF POTENTIAL AND GRADIENT
# ═══════════════════════════════════════════════════════════════════

def rbf_potential(q, w, mu, sigma):
    """
    V(q; θ) = Σ_k  w_k · exp( -‖q - μ_k‖² / (2σ_k²) )

    Args:
        q     : (N, 3)  particle positions
        w     : (K,)    RBF weights
        mu    : (K, 3)  RBF centers
        sigma : (K,)    RBF widths
    Returns:
        V     : (N,)    potential energy per particle
    """
    # diff[k,i,:] = q[i] - mu[k]   →  broadcast: (K,1,3) vs (1,N,3)
    diff = q[np.newaxis, :, :] - mu[:, np.newaxis, :]   # (K, N, 3)
    r2   = np.sum(diff**2, axis=2)                       # (K, N)  squared distances
    # Gaussian kernel per RBF center
    gauss = np.exp(-r2 / (2.0 * sigma[:, np.newaxis]**2))  # (K, N)
    # Weighted sum over K centers
    V = np.einsum('k,kn->n', w, gauss)                   # (N,)
    return V


def rbf_gradient(q, w, mu, sigma):
    """
    ∇_q V(q; θ) = Σ_k  w_k · exp(-‖q-μ_k‖²/2σ_k²) · (-(q - μ_k)/σ_k²)

    Returns:
        grad  : (N, 3)  gradient of V w.r.t. q for each particle
    """
    diff  = q[np.newaxis, :, :] - mu[:, np.newaxis, :]    # (K, N, 3)
    r2    = np.sum(diff**2, axis=2)                         # (K, N)
    gauss = np.exp(-r2 / (2.0 * sigma[:, np.newaxis]**2))  # (K, N)

    # Factor: w_k * gauss_k / sigma_k²   shape (K, N)
    factor = (w[:, np.newaxis] * gauss
              / sigma[:, np.newaxis]**2)                    # (K, N)

    # grad = Σ_k  factor[k,i] * (-diff[k,i,:])
    grad = -np.einsum('kn,knd->nd', factor, diff)          # (N, 3)
    return grad


# ═══════════════════════════════════════════════════════════════════
# PART C: CONTACT HAMILTONIAN ODE (RHS)
# ═══════════════════════════════════════════════════════════════════

def contact_rhs(S, w, mu, sigma, gamma):
    """
    Contact Hamilton's equations:

        dq/dt = p
        dp/dt = -∇V(q) - γ·p          ← damped Newton (contact correction)
        dz/dt = ‖p‖² - H              ← contact variable tracks energy dissipation

    where H = ‖p‖²/2 + V(q)  (mechanical Hamiltonian)

    Args:
        S     : (N_max, 7)  state [q(3), p(3), z(1)]
        w, mu, sigma        RBF parameters
        gamma               damping coefficient
    Returns:
        dS/dt : (N_max, 7)
    """
    q = S[:, :3]    # (N_max, 3)  positions
    p = S[:, 3:6]   # (N_max, 3)  momenta
    # z = S[:, 6]   # (N_max,)    contact variable (not needed for rhs directly)

    V    = rbf_potential(q, w, mu, sigma)       # (N_max,)
    gradV = rbf_gradient(q, w, mu, sigma)       # (N_max, 3)

    p2   = np.sum(p**2, axis=1)                 # (N_max,)  ‖p‖²
    H    = 0.5 * p2 + V                         # (N_max,)  Hamiltonian

    dq = p                                      # (N_max, 3)
    dp = -gradV - gamma * p                     # (N_max, 3)  damped force
    dz = p2 - H                                 # (N_max,)    contact eq.

    dS = np.concatenate([dq, dp, dz[:, np.newaxis]], axis=1)  # (N_max, 7)
    return dS


# ═══════════════════════════════════════════════════════════════════
# PART D: RK4 INTEGRATOR
# ═══════════════════════════════════════════════════════════════════

def rk4_step(S, dt, w, mu, sigma, gamma):
    """
    Single 4th-order Runge-Kutta step.

    k1 = f(S)
    k2 = f(S + dt/2 · k1)
    k3 = f(S + dt/2 · k2)
    k4 = f(S + dt   · k3)
    S_new = S + dt/6 · (k1 + 2k2 + 2k3 + k4)
    """
    f = lambda s: contact_rhs(s, w, mu, sigma, gamma)

    k1 = f(S)
    k2 = f(S + 0.5 * dt * k1)
    k3 = f(S + 0.5 * dt * k2)
    k4 = f(S + dt * k3)

    return S + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ═══════════════════════════════════════════════════════════════════
# PART E: FULL SIMULATION
# ═══════════════════════════════════════════════════════════════════

def simulate(image, w, mu, sigma, gamma=1.5, T=10.0, dt=0.05, tau=0.5):
    """
    Run full contact Hamiltonian simulation.

    Returns:
        trajectory : (N_step+1, N_max, 7)  full state history
        mask       : (N_max,) bool          real particle mask
    """
    q0, p0, z0, mask = preprocess(image, tau=tau, N_max=N_MAX)

    N_step = int(round(T / dt))

    # Pack initial state: S[i] = [q_i(3), p_i(3), z_i(1)]
    S = np.concatenate([q0, p0, z0[:, np.newaxis]], axis=1)  # (64, 7)

    trajectory = np.zeros((N_step + 1, N_MAX, 7))
    trajectory[0] = S

    for step in range(N_step):
        S = rk4_step(S, dt, w, mu, sigma, gamma)
        trajectory[step + 1] = S

        # NaN/Inf guard
        if not np.all(np.isfinite(S)):
            first_bad = np.argwhere(~np.isfinite(S))
            print(f"[ERROR] NaN/Inf detected at timestep {step+1}, "
                  f"first occurrence: particle {first_bad[0,0]}, "
                  f"state index {first_bad[0,1]}")
            trajectory = trajectory[:step+2]
            break

    assert np.all(np.isfinite(trajectory)), "NaN/Inf detected in trajectory"

    return trajectory, mask


# ═══════════════════════════════════════════════════════════════════
# PART F: ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def compute_com(trajectory, mask):
    """
    Center of Mass over real particles at each timestep.

    Returns:
        q_com : (N_step+1, 3)
    """
    # trajectory: (T, N_max, 7), mask: (N_max,)
    q = trajectory[:, mask, :3]          # (T, N_real, 3)
    q_com = q.mean(axis=1)               # (T, 3)
    return q_com


def compute_hamiltonian(trajectory, mask, w, mu, sigma):
    """
    Mechanical Hamiltonian H_i(t) = ‖p_i‖²/2 + V(q_i) for real particles.

    Returns:
        H_real : (N_step+1, N_real)
    """
    T_steps = trajectory.shape[0]
    N_real  = mask.sum()
    H_all   = np.zeros((T_steps, N_real))

    q_all = trajectory[:, mask, :3]   # (T, N_real, 3)
    p_all = trajectory[:, mask, 3:6]  # (T, N_real, 3)

    for t in range(T_steps):
        V  = rbf_potential(q_all[t], w, mu, sigma)   # (N_real,)
        p2 = np.sum(p_all[t]**2, axis=1)             # (N_real,)
        H_all[t] = 0.5 * p2 + V

    return H_all


def compute_phase_volume(trajectory, mask):
    """
    Approximate phase-space volume via covariance matrix determinant.

    joint(t) = [q_real(t), p_real(t)]  shape (N_real, 6)
    V_phase(t) = sqrt(det(Cov(joint) + ε·I))

    Returns:
        vol_ratio : (N_step+1,)   V_phase(t) / V_phase(0)
    """
    T_steps = trajectory.shape[0]
    vols    = np.zeros(T_steps)

    q_all = trajectory[:, mask, :3]   # (T, N_real, 3)
    p_all = trajectory[:, mask, 3:6]  # (T, N_real, 3)

    for t in range(T_steps):
        joint = np.concatenate([q_all[t], p_all[t]], axis=1)  # (N_real, 6)
        cov   = np.cov(joint.T)                                # (6, 6)
        # Regularize to avoid singular matrix
        det   = np.linalg.det(cov + 1e-10 * np.eye(6))
        vols[t] = np.sqrt(np.abs(det))

    # Normalize by t=0
    vol0 = vols[0] if vols[0] > 0 else 1.0
    return vols / vol0


def classify(trajectory, mask, q_star_O, q_star_X):
    """
    Classify based on final CoM proximity to attractors.

    Returns:
        pred    : 'O' or 'X'
        dist_O  : distance to O-attractor
        dist_X  : distance to X-attractor
        q_com_f : final CoM position (3,)
    """
    q_com   = compute_com(trajectory, mask)    # (T, 3)
    q_com_f = q_com[-1]                        # final CoM

    dist_O = np.linalg.norm(q_com_f - q_star_O)
    dist_X = np.linalg.norm(q_com_f - q_star_X)

    pred = 'O' if dist_O < dist_X else 'X'
    return pred, dist_O, dist_X, q_com_f


# ═══════════════════════════════════════════════════════════════════
# PART G: VERIFICATION AND PLOTTING
# ═══════════════════════════════════════════════════════════════════

def verify_and_plot(traj_O, mask_O, traj_X, mask_X, w, mu, sigma, t_array):
    """
    Generate 2×3 verification figure.
    """
    gamma = GAMMA

    # ── Pre-compute all quantities ──────────────────────────────

    q_com_O = compute_com(traj_O, mask_O)     # (T, 3)
    q_com_X = compute_com(traj_X, mask_X)

    H_O = compute_hamiltonian(traj_O, mask_O, w, mu, sigma)  # (T, N_real_O)
    H_X = compute_hamiltonian(traj_X, mask_X, w, mu, sigma)

    vol_O = compute_phase_volume(traj_O, mask_O)  # (T,)
    vol_X = compute_phase_volume(traj_X, mask_X)

    # Convergence metrics
    eps_q_O = np.linalg.norm(q_com_O - Q_STAR_O, axis=1)   # (T,)
    eps_q_X = np.linalg.norm(q_com_X - Q_STAR_X, axis=1)

    p_O = traj_O[:, mask_O, 3:6]    # (T, N_real, 3)
    p_X = traj_X[:, mask_X, 3:6]
    eps_p_O = np.mean(np.linalg.norm(p_O, axis=2), axis=1)  # (T,)
    eps_p_X = np.mean(np.linalg.norm(p_X, axis=2), axis=1)

    # Theoretical volume contraction
    vol_theory = np.exp(-3 * gamma * t_array)

    # Classification
    pred_O, dO_O, dX_O, qf_O = classify(traj_O, mask_O, Q_STAR_O, Q_STAR_X)
    pred_X, dO_X, dX_X, qf_X = classify(traj_X, mask_X, Q_STAR_O, Q_STAR_X)

    # ── PASS/FAIL checks ────────────────────────────────────────

    # Energy monotone: dH/dt ≤ 0 (allow +1e-6 noise)
    def check_energy_monotone(H):
        diffs = np.diff(H, axis=0)    # (T-1, N_real)
        bad   = np.sum(diffs > 1e-6)
        total = diffs.size
        return (bad / total) < 0.05   # >95% steps monotone

    pass_energy_O = check_energy_monotone(H_O)
    pass_energy_X = check_energy_monotone(H_X)

    # Volume contraction R²
    from numpy.polynomial import polynomial as P
    def r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - y_true.mean())**2)
        return 1 - ss_res / (ss_tot + 1e-30)

    r2_O = r2(vol_O, vol_theory)
    r2_X = r2(vol_X, vol_theory)
    pass_vol_O = r2_O > 0.90
    pass_vol_X = r2_X > 0.90

    # Convergence
    pass_conv_O = (eps_q_O[-1] < 2.0) and (eps_p_O[-1] < 0.5)
    pass_conv_X = (eps_q_X[-1] < 2.0) and (eps_p_X[-1] < 0.5)

    pass_cls_O = (pred_O == 'O')
    pass_cls_X = (pred_X == 'X')

    all_pass = all([pass_energy_O, pass_energy_X,
                    pass_vol_O,    pass_vol_X,
                    pass_conv_O,   pass_conv_X,
                    pass_cls_O,    pass_cls_X])

    # ── Figure ──────────────────────────────────────────────────

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Block I — Contact Hamiltonian Simulator Verification",
                 fontsize=14, fontweight='bold')

    # ── [1,1] RBF Potential Contour ─────────────────────────────
    ax = axes[0, 0]
    gx = np.linspace(-12, 12, 300)
    gy = np.linspace(-12, 12, 300)
    GX, GY = np.meshgrid(gx, gy)
    grid_q = np.stack([GX.ravel(), GY.ravel(),
                       0.5 * np.ones(GX.size)], axis=1)   # (N_grid, 3)
    GV = rbf_potential(grid_q, w, mu, sigma).reshape(GX.shape)

    norm = TwoSlopeNorm(vmin=GV.min(), vcenter=0, vmax=GV.max())
    cf = ax.contourf(GX, GY, GV, levels=40, cmap='RdBu_r', norm=norm, alpha=0.85)
    ax.contour(GX, GY, GV, levels=15, colors='k', linewidths=0.4, alpha=0.4)
    plt.colorbar(cf, ax=ax, label='V(q)')

    # Attractor stars
    ax.plot(mu[0,0], mu[0,1], '*', ms=16, color='cyan',   zorder=5, label='μ_O (att)')
    ax.plot(mu[1,0], mu[1,1], '*', ms=16, color='lime',   zorder=5, label='μ_X (att)')
    ax.plot(mu[2,0], mu[2,1], '^', ms=12, color='orange', zorder=5, label='μ barrier')
    ax.plot(mu[3,0], mu[3,1], 'D', ms=10, color='yellow', zorder=5, label='μ guide')

    # Initial particle positions
    q0_O = traj_O[0, mask_O, :2]
    q0_X = traj_X[0, mask_X, :2]
    ax.scatter(q0_O[:,0], q0_O[:,1], c='blue',  s=15, zorder=6, label='O init')
    ax.scatter(q0_X[:,0], q0_X[:,1], c='red',   s=15, zorder=6, label='X init')

    ax.set_xlim(-12, 12); ax.set_ylim(-12, 12)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title("RBF Potential Landscape (z=0.5 slice)")
    ax.legend(fontsize=6, loc='upper left')

    # ── [1,2] Particle Trajectories ─────────────────────────────
    ax = axes[0, 1]
    q_traj_O = traj_O[:, mask_O, :2]   # (T, N_real_O, 2)
    q_traj_X = traj_X[:, mask_X, :2]

    for i in range(q_traj_O.shape[1]):
        ax.plot(q_traj_O[:,i,0], q_traj_O[:,i,1],
                'b-', lw=0.5, alpha=0.3)
        ax.plot(q_traj_O[0,i,0], q_traj_O[0,i,1], 'bo', ms=3, alpha=0.5)
        ax.plot(q_traj_O[-1,i,0], q_traj_O[-1,i,1], 'bx', ms=4, alpha=0.8)

    for i in range(q_traj_X.shape[1]):
        ax.plot(q_traj_X[:,i,0], q_traj_X[:,i,1],
                'r-', lw=0.5, alpha=0.3)
        ax.plot(q_traj_X[0,i,0], q_traj_X[0,i,1], 'ro', ms=3, alpha=0.5)
        ax.plot(q_traj_X[-1,i,0], q_traj_X[-1,i,1], 'rx', ms=4, alpha=0.8)

    ax.plot(*Q_STAR_O[:2], '*', ms=18, color='cyan',
            zorder=5, label='q* O attractor')
    ax.plot(*Q_STAR_X[:2], '*', ms=18, color='lime',
            zorder=5, label='q* X attractor')

    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title("Particle Trajectories (xy-projection)")
    ax.legend(fontsize=8)
    ax.set_xlim(-12, 12); ax.set_ylim(-12, 12)

    # ── [1,3] Energy Decrease ────────────────────────────────────
    ax = axes[0, 2]
    # Individual particle H (thin)
    for i in range(H_O.shape[1]):
        ax.plot(t_array, H_O[:, i], 'b-', lw=0.3, alpha=0.15)
    for i in range(H_X.shape[1]):
        ax.plot(t_array, H_X[:, i], 'r-', lw=0.3, alpha=0.15)

    ax.plot(t_array, H_O.mean(axis=1), 'b-', lw=2,
            label=f'O mean H(t)  [mono={pass_energy_O}]')
    ax.plot(t_array, H_X.mean(axis=1), 'r-', lw=2,
            label=f'X mean H(t)  [mono={pass_energy_X}]')

    ax.set_xlabel('t'); ax.set_ylabel('H(t)')
    ax.set_title("Hamiltonian H(t) — Must Be Monotone Decreasing")
    ax.legend(fontsize=9)

    # ── [2,1] Phase-Space Volume ──────────────────────────────────
    ax = axes[1, 0]
    ax.plot(t_array, vol_O,      'b-',  lw=2, label=f'O empirical (R²={r2_O:.3f})')
    ax.plot(t_array, vol_X,      'r-',  lw=2, label=f'X empirical (R²={r2_X:.3f})')
    ax.plot(t_array, vol_theory, 'k--', lw=2, label='Theory  exp(-3γt)')

    ax.set_xlabel('t'); ax.set_ylabel('V_phase(t) / V_phase(0)')
    ax.set_title("Phase-Space Volume: Empirical vs Theory exp(-3γt)")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    # ── [2,2] Convergence Metrics ─────────────────────────────────
    ax = axes[1, 1]
    ax.plot(t_array, eps_q_O, 'b-',  lw=2, label='ε_q O  (pos error)')
    ax.plot(t_array, eps_q_X, 'r-',  lw=2, label='ε_q X  (pos error)')
    ax.plot(t_array, eps_p_O, 'b--', lw=1.5, label='ε_p O  (mom error)')
    ax.plot(t_array, eps_p_X, 'r--', lw=1.5, label='ε_p X  (mom error)')
    ax.axhline(0, color='k', lw=0.7, ls=':')

    ax.set_xlabel('t'); ax.set_ylabel('Error')
    ax.set_title("Convergence: ε_q(t) and ε_p(t)")
    ax.legend(fontsize=9)

    # ── [2,3] Classification Result Summary ──────────────────────
    ax = axes[1, 2]
    ax.axis('off')
    bg = 'honeydew' if all_pass else 'mistyrose'
    ax.set_facecolor(bg)
    fig.patch.set_facecolor('white')

    def pf(b):
        return '✓ PASS' if b else '✗ FAIL'

    lines = [
        "═══  Block I Verification Summary  ═══",
        "",
        "── O-image ──",
        f"  Final CoM:  ({qf_O[0]:.2f}, {qf_O[1]:.2f}, {qf_O[2]:.2f})",
        f"  dist(q*_O): {dO_O:.3f}   dist(q*_X): {dX_O:.3f}",
        f"  Prediction: {pred_O}  (true: O)   {pf(pass_cls_O)}",
        f"  Energy mono:       {pf(pass_energy_O)}",
        f"  Volume R²={r2_O:.3f}:  {pf(pass_vol_O)}",
        f"  Conv ε_q={eps_q_O[-1]:.3f}, ε_p={eps_p_O[-1]:.3f}:  {pf(pass_conv_O)}",
        "",
        "── X-image ──",
        f"  Final CoM:  ({qf_X[0]:.2f}, {qf_X[1]:.2f}, {qf_X[2]:.2f})",
        f"  dist(q*_O): {dO_X:.3f}   dist(q*_X): {dX_X:.3f}",
        f"  Prediction: {pred_X}  (true: X)   {pf(pass_cls_X)}",
        f"  Energy mono:       {pf(pass_energy_X)}",
        f"  Volume R²={r2_X:.3f}:  {pf(pass_vol_X)}",
        f"  Conv ε_q={eps_q_X[-1]:.3f}, ε_p={eps_p_X[-1]:.3f}:  {pf(pass_conv_X)}",
        "",
        "══════════════════════════════════════",
        f"  OVERALL: {'✓ ALL PASS' if all_pass else '✗ SOME FAILED'}",
    ]

    ax.text(0.05, 0.97, '\n'.join(lines),
            transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor=bg, alpha=0.9))
    ax.set_title("Block I Verification Summary")

    plt.tight_layout()
    plt.savefig('block1_verification.png', dpi=150, bbox_inches='tight')
    print("Figure saved: block1_verification.png")
    plt.show()

    return all_pass, {
        'pass_energy_O': pass_energy_O, 'pass_energy_X': pass_energy_X,
        'pass_vol_O':    pass_vol_O,    'pass_vol_X':    pass_vol_X,
        'pass_conv_O':   pass_conv_O,   'pass_conv_X':   pass_conv_X,
        'pass_cls_O':    pass_cls_O,    'pass_cls_X':    pass_cls_X,
        'r2_O': r2_O,   'r2_X': r2_X,
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 60)
    print("  Block I: Contact Hamiltonian Fluid Simulator")
    print("=" * 60)

    # RBF parameters
    w     = W
    mu    = MU
    sigma = SIGMA

    t_array = np.linspace(0, T_END, N_STEP + 1)  # (201,)

    # ── Simulate O-image ──────────────────────────────────────
    print("\n[1/2] Simulating O-image...")
    traj_O, mask_O = simulate(O_IMAGE, w, mu, sigma,
                              gamma=GAMMA, T=T_END, dt=DT)
    N_real_O = mask_O.sum()
    print(f"      Real particles: {N_real_O},  Trajectory shape: {traj_O.shape}")

    # ── Simulate X-image ──────────────────────────────────────
    print("[2/2] Simulating X-image...")
    traj_X, mask_X = simulate(X_IMAGE, w, mu, sigma,
                              gamma=GAMMA, T=T_END, dt=DT)
    N_real_X = mask_X.sum()
    print(f"      Real particles: {N_real_X},  Trajectory shape: {traj_X.shape}")

    # ── Classification ────────────────────────────────────────
    pred_O, dO_O, dX_O, qf_O = classify(traj_O, mask_O, Q_STAR_O, Q_STAR_X)
    pred_X, dO_X, dX_X, qf_X = classify(traj_X, mask_X, Q_STAR_O, Q_STAR_X)

    print("\n── Classification Results ──")
    print(f"  O-image: pred={pred_O}  dist_O={dO_O:.3f}  dist_X={dX_O:.3f}")
    print(f"  X-image: pred={pred_X}  dist_O={dO_X:.3f}  dist_X={dX_X:.3f}")

    # ── Verify & Plot ─────────────────────────────────────────
    print("\n── Generating Verification Figure ──")
    all_pass, results = verify_and_plot(
        traj_O, mask_O, traj_X, mask_X,
        w, mu, sigma, t_array
    )

    # ── Stdout Pass/Fail Report ───────────────────────────────
    print("\n" + "=" * 60)
    print("  VERIFICATION PASS/FAIL REPORT")
    print("=" * 60)
    checks = [
        ("Energy monotone (O)",       results['pass_energy_O']),
        ("Energy monotone (X)",       results['pass_energy_X']),
        (f"Phase vol R²={results['r2_O']:.3f} ≥ 0.90 (O)", results['pass_vol_O']),
        (f"Phase vol R²={results['r2_X']:.3f} ≥ 0.90 (X)", results['pass_vol_X']),
        ("Convergence ε_q<2, ε_p<0.5 (O)", results['pass_conv_O']),
        ("Convergence ε_q<2, ε_p<0.5 (X)", results['pass_conv_X']),
        ("Classification O→O",        results['pass_cls_O']),
        ("Classification X→X",        results['pass_cls_X']),
    ]
    for name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {name}")

    print("=" * 60)
    overall = "✓ ALL PASS" if all_pass else "✗ SOME CHECKS FAILED"
    print(f"  OVERALL RESULT: {overall}")
    print("=" * 60)