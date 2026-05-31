[SYSTEM CONTEXT]
You are an expert in Hamiltonian mechanics, contact geometry, and scientific computing.
Build a complete, working Python artifact for the Block I simulator described below.
Use extended thinking to carefully verify every equation before implementation.

══════════════════════════════════════════════════════════════
PART 0: PROJECT BACKGROUND (READ BEFORE CODING)
══════════════════════════════════════════════════════════════

This is Block I of a "Contact Hamiltonian Fluid Neural Network" — a physically
interpretable image classifier where pixels are treated as fluid particles flowing
over a learnable RBF potential landscape under contact Hamiltonian dynamics.

Block I is the FORWARD SIMULATOR ONLY. No training, no backpropagation.
Goal: verify that particles actually converge to target attractors, and that
phase-space volume contracts as theory predicts.

══════════════════════════════════════════════════════════════
PART 1: MATHEMATICS (IMPLEMENT EXACTLY AS SPECIFIED)
══════════════════════════════════════════════════════════════

─── 1-A. PREPROCESSING ───────────────────────────────────────

Input: 8×8 binary image I(x,y) ∈ {0,1} (O or X pattern)

Step 1 — Threshold filtering:
    Particle set P = {(x_i, y_i) : I(x_i, y_i) > τ},   τ = 0.5
    x_i, y_i ∈ {0,1,...,7}  (pixel grid coordinates)

Step 2 — 3D Lifting (initial conditions):
    q_i(0) = (x_i,  y_i,  I(x_i, y_i))   ∈ ℝ³
    p_i(0) = (0,    0,    0)               ∈ ℝ³

Step 3 — Fixed-size padding:
    N_max = 64 (total slots)
    Real particles fill first N slots, dummy particles fill remaining slots.
    Dummy particles: set a boolean mask mask_i = False.
    Dummy particles do NOT contribute to CoM calculation.

─── 1-B. RBF POTENTIAL ───────────────────────────────────────

V(q; θ) = Σ_{k=1}^{K}  w_k · exp( -‖q - μ_k‖² / (2σ_k²) )

Gradient (needed for ODE):
∇_q V(q; θ) = Σ_{k=1}^{K}  w_k · exp(-‖q-μ_k‖²/2σ_k²) · (-(q - μ_k)/σ_k²)

Parameters for Block I (K=4, FIXED — no learning):

    k=0  ATTRACTOR for O-class:
         w_0 = -2.0,  μ_0 = (8.0,  8.0,  0.5),  σ_0 = 2.0

    k=1  ATTRACTOR for X-class:
         w_1 = -2.0,  μ_1 = (-8.0, -8.0, 0.5),  σ_1 = 2.0

    k=2  BARRIER (separates two attractors):
         w_2 = +1.5,  μ_2 = (0.0,  0.0,  0.5),  σ_2 = 2.0

    k=3  PATH GUIDE (draws particles toward attractor region):
         w_3 = -0.5,  μ_3 = (4.0,  4.0,  0.5),  σ_3 = 3.0

Target coordinates:
    q*_O = (8.0,  8.0,  0.0)   ← O-class convergence target
    q*_X = (-8.0, -8.0, 0.0)   ← X-class convergence target

─── 1-C. CONTACT HAMILTONIAN ODE ────────────────────────────

Mechanical Hamiltonian:
    H_i(t) = ‖p_i‖² / 2  +  V(q_i; θ)

Contact Hamilton's equations (for each particle i):
    dq_i/dt  =  p_i
    dp_i/dt  =  -∇_q V(q_i; θ)  -  γ · p_i
    dz_i/dt  =  ‖p_i‖²  -  H_i

Fixed parameters:
    γ = 1.5    (damping coefficient; slightly overdamped, γ_crit ≈ 1.41)
    T = 10.0   (total simulation time)
    Δt = 0.05  (fixed integration step → N_step = 200)

IMPORTANT: z_i(t) is the contact geometry variable tracking dissipated energy.
           Do NOT confuse with q_i's z-component (pixel intensity).
           z_i(0) = 0 for all particles.

─── 1-D. INTEGRATION METHOD ─────────────────────────────────

Use 4th-order Runge-Kutta (RK4) with fixed step Δt = 0.05.
Do NOT use adaptive step solvers for Block I (need reproducible trajectories).

State vector per particle: s_i = (q_i, p_i, z_i) ∈ ℝ^7
Full system state: S = stack of s_i for i=0,...,63  → shape (64, 7)

Vectorize over all 64 particles simultaneously.

─── 1-E. CENTER OF MASS (CoM) ───────────────────────────────

At final time T, compute CoM over REAL particles only (mask_i = True):

    q_CoM = (1/N) Σ_{mask_i=True} q_i(T)

where N = number of real particles.

Classification rule:
    dist_O = ‖q_CoM - q*_O‖
    dist_X = ‖q_CoM - q*_X‖
    prediction = 'O' if dist_O < dist_X else 'X'

─── 1-F. THEORETICAL PREDICTIONS TO VERIFY ──────────────────

(A) Energy monotone decrease:
    dH_i/dt = -γ ‖p_i‖² ≤ 0   (must hold at every timestep)

(B) Phase-space volume contraction:
    V_phase(t) / V_phase(0) = exp(-3γt)
    Approximate V_phase using covariance matrix det of (q, p) joint distribution
    across all real particles: V_phase(t) ≈ det(Cov([q(t), p(t)]))^(1/2)
    Plot this vs theoretical exp(-3γt) curve.

(C) Convergence metric:
    ε_q(t) = ‖q_CoM(t) - q*‖
    ε_p(t) = (1/N) Σ ‖p_i(t)‖
    Both should → 0 as t → T.

══════════════════════════════════════════════════════════════
PART 2: TEST DATA (HARDCODE THESE EXACTLY)
══════════════════════════════════════════════════════════════

Use numpy arrays. Pixels are (row, col) → map to (x=col, y=row) for coordinates.

O_IMAGE (8×8, 1=pixel on):
    O = np.array([
        [0,0,0,0,0,0,0,0],
        [0,0,1,1,1,1,0,0],
        [0,1,0,0,0,0,1,0],
        [0,1,0,0,0,0,1,0],
        [0,1,0,0,0,0,1,0],
        [0,1,0,0,0,0,1,0],
        [0,0,1,1,1,1,0,0],
        [0,0,0,0,0,0,0,0],
    ], dtype=float)

X_IMAGE (8×8, 1=pixel on):
    X = np.array([
        [1,0,0,0,0,0,0,1],
        [0,1,0,0,0,0,1,0],
        [0,0,1,0,0,1,0,0],
        [0,0,0,1,1,0,0,0],
        [0,0,0,1,1,0,0,0],
        [0,0,1,0,0,1,0,0],
        [0,1,0,0,0,0,1,0],
        [1,0,0,0,0,0,0,1],
    ], dtype=float)

══════════════════════════════════════════════════════════════
PART 3: LIBRARIES AND IMPLEMENTATION CONSTRAINTS
══════════════════════════════════════════════════════════════

ALLOWED LIBRARIES (standard Claude artifact environment):
    numpy, scipy, matplotlib, itertools, time

DO NOT USE: jax, diffrax, torch, tensorflow
(Those are for Block II; this is a pure numpy prototype.)

Implementation requirements:
  1. ALL particles vectorized — no Python for-loops over particles.
     Use numpy broadcasting for RBF gradient computation.
  2. RK4 step function takes full state S of shape (64,7) and returns dS/dt.
  3. Store full trajectory: shape (N_step+1, 64, 7).
  4. Mask must be applied BEFORE CoM computation at every timestep.

══════════════════════════════════════════════════════════════
PART 4: VISUALIZATION REQUIREMENTS (6 subplots, 2×3 grid)
══════════════════════════════════════════════════════════════

Run simulation for BOTH O and X images. Generate the following figure:

[1,1] RBF Potential Contour (xy-plane at z=0.5):
      - 2D contour plot of V(q_x, q_y, 0.5; θ) over grid x,y ∈ [-12,12]
      - Mark attractor positions μ_0, μ_1 with stars
      - Mark barrier μ_2 with triangle
      - Overlay initial particle positions for O (blue dots) and X (red dots)
      - Title: "RBF Potential Landscape (z=0.5 slice)"

[1,2] Particle Trajectories (xy-plane):
      - Plot q_x(t) vs q_y(t) for each real particle
      - O-class: blue lines; X-class: red lines
      - Mark start (circle) and end (cross) positions
      - Mark target attractors q*_O, q*_X with large stars
      - Title: "Particle Trajectories (xy-projection)"

[1,3] Energy Decrease Verification:
      - Plot H_i(t) averaged over real particles vs time, for both O and X
      - Must be monotonically decreasing
      - Also plot individual particle H_i(t) as thin transparent lines
      - Title: "Hamiltonian H(t) — Must Be Monotone Decreasing"

[2,1] Phase-Space Volume Contraction:
      - Plot empirical V_phase(t)/V_phase(0) vs time
      - Overlay theoretical exp(-3γt) as dashed line
      - Both O and X on same plot
      - Title: "Phase-Space Volume: Empirical vs Theory exp(-3γt)"

[2,2] Convergence Metrics:
      - Plot ε_q(t) = ‖q_CoM(t) - q*‖ for both O and X (solid lines)
      - Plot ε_p(t) = mean‖p_i(t)‖ for both (dashed lines)
      - Both should converge to 0
      - Title: "Convergence: ε_q(t) and ε_p(t)"

[2,3] Classification Result Summary:
      - Text panel showing:
        * Final CoM position for O and X
        * Distance to each attractor
        * Predicted class vs true class
        * PASS/FAIL for: energy monotone, volume contraction (R²>0.95), convergence (ε_q<1.0)
      - Background green if all PASS, red if any FAIL
      - Title: "Block I Verification Summary"

Figure size: (18, 10), tight_layout.
Save as: block1_verification.png  (also plt.show())

══════════════════════════════════════════════════════════════
PART 5: CODE STRUCTURE REQUIREMENTS
══════════════════════════════════════════════════════════════

Organize as the following functions (do NOT merge them):

    def preprocess(image, tau=0.5, N_max=64):
        """Returns q0 (N_max,3), p0 (N_max,3), z0 (N_max,), mask (N_max,bool)"""

    def rbf_potential(q, w, mu, sigma):
        """q: (N,3), returns V: (N,) scalar per particle"""

    def rbf_gradient(q, w, mu, sigma):
        """q: (N,3), returns ∇V: (N,3)"""

    def contact_rhs(S, w, mu, sigma, gamma):
        """S: (N_max, 7), returns dS/dt: (N_max, 7)"""
        # Unpack: q=S[:,:3], p=S[:,3:6], z=S[:,6]
        # Compute dq/dt, dp/dt, dz/dt
        # Dummy particles have zero force (use mask or just let them drift)

    def rk4_step(S, dt, w, mu, sigma, gamma):
        """Single RK4 step"""

    def simulate(image, w, mu, sigma, gamma=1.5, T=10.0, dt=0.05, tau=0.5):
        """Full simulation. Returns trajectory (N_step+1, N_max, 7), mask (N_max,)"""

    def compute_com(trajectory, mask):
        """Returns q_CoM(t) of shape (N_step+1, 3)"""

    def compute_hamiltonian(trajectory, mask, w, mu, sigma):
        """Returns H_i(t) of shape (N_step+1, N_real)"""

    def compute_phase_volume(trajectory, mask):
        """Returns V_phase(t)/V_phase(0) of shape (N_step+1,)"""

    def classify(trajectory, mask, q_star_O, q_star_X):
        """Returns predicted label 'O' or 'X' and distances"""

    def verify_and_plot(traj_O, mask_O, traj_X, mask_X, w, mu, sigma, t_array):
        """Generates the 2×3 verification figure"""

    # Main execution block:
    if __name__ == "__main__":
        # Define RBF parameters (K=4)
        # Run simulation for O and X
        # Print verification results
        # Generate figure

══════════════════════════════════════════════════════════════
PART 6: VERIFICATION PASS CRITERIA
══════════════════════════════════════════════════════════════

Block I is considered PASSING if ALL of the following hold:

  ✓ ENERGY:     H_i(t) is monotonically non-increasing for >95% of particles
                (allow floating point noise ≤ 1e-6 between steps)

  ✓ VOLUME:     R² between empirical V_phase(t)/V_phase(0) and exp(-3γt) > 0.90

  ✓ CONVERGENCE: ε_q(T) < 2.0 for both O and X
                 ε_p(T) < 0.5 for both O and X

  ✓ CLASSIFICATION: O-image → predicted 'O', X-image → predicted 'X'

Print a clear PASS/FAIL report to stdout after simulation.

══════════════════════════════════════════════════════════════
PART 7: PHYSICAL SANITY CHECKS (ADD AS ASSERTIONS)
══════════════════════════════════════════════════════════════

Add the following assertions during simulation:

  assert q0.shape == (64, 3),  "Initial position shape error"
  assert p0.shape == (64, 3),  "Initial momentum shape error"
  assert mask.sum() > 0,       "No real particles found — check threshold"
  assert np.all(np.isfinite(trajectory)), "NaN/Inf detected in trajectory"

If NaN appears, print the timestep at which it first occurred and halt.

══════════════════════════════════════════════════════════════
PART 8: ADDITIONAL NOTES
══════════════════════════════════════════════════════════════

1. COORDINATE CONVENTION:
   Image row r, column c → x = c, y = 7-r (flip y so y increases upward)
   This ensures O-image particles cluster in the upper-right (→ attractor at (8,8,0))
   and X-image particles cluster symmetrically.

2. DUMMY PARTICLES:
   Initialize dummy particles at q=(0,0,0), p=(0,0,0), z=0.
   They will drift under the potential. This is fine for Block I.
   Just exclude them from CoM and energy plots using the mask.

3. PHASE VOLUME APPROXIMATION:
   joint = np.concatenate([q_real(t), p_real(t)], axis=1)  # shape (N_real, 6)
   cov = np.cov(joint.T)                                    # shape (6,6)
   V_phase(t) = np.sqrt(np.linalg.det(cov))
   Handle singular matrix with: np.linalg.det(cov + 1e-10*np.eye(6))

4. This is a PROTOTYPE. Prioritize correctness over efficiency.
   Add comments explaining each equation and its physical meaning.

══════════════════════════════════════════════════════════════
FINAL INSTRUCTION
══════════════════════════════════════════════════════════════

Build the complete, runnable Python artifact now.
All 8 parts above must be satisfied.
Use extended thinking to verify equations before coding each function.
The artifact must run end-to-end without errors and produce the 6-panel figure.