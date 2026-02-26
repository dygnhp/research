[CONTEXT]
You are modifying an existing, working JAX-based physics simulator.
The code is block_i.py from github.com/dygnhp/research (block_i_prototype/).
The physics logic (RK4, contact_rhs, rbf_potential/gradient, lax.scan) is
CORRECT and must NOT be touched. Only the RBF parameter initialization and
a few hardcoded K=4 strings need to change.

══════════════════════════════════════════════════════════════
WHY K=4 FAILS (DO NOT FIX THE PHYSICS — FIX THE PARAMETERS)
══════════════════════════════════════════════════════════════

Root cause (mathematically verified):

    q_CoM_O(0) ≈ (3.5, 3.5, 1.0)
    q_CoM_X(0) ≈ (3.5, 3.5, 1.0)   ← IDENTICAL

Both classes start at nearly the same CoM. The distinguishing information
is NOT in the global CoM but in the INTRA-QUADRANT spatial distribution:

    O-image: pixels form arc shapes at each quadrant corner
             Cov(Q1_O) has small off-diagonal (compact arc)

    X-image: pixels form diagonal lines through each quadrant
             Cov(Q1_X) has large negative off-diagonal (diagonal spread)

K=4 cannot detect this per-quadrant distributional difference.
K=16 with discriminator RBFs placed near O-arc and X-diagonal positions
gives the potential landscape enough spatial resolution.

══════════════════════════════════════════════════════════════
MODIFICATION 1 — Replace the three parameter arrays (ONLY CHANGE)
══════════════════════════════════════════════════════════════

FIND this block in the Global constants section (lines ~50-57):

    W_INIT     = jnp.array([-2.0, -2.0,  1.5, -0.5])
    MU_INIT    = jnp.array([[ 8.0,  8.0, 0.5],
                              [-8.0, -8.0, 0.5],
                              [ 0.0,  0.0, 0.5],
                              [ 4.0,  4.0, 0.5]], dtype=jnp.float32)
    SIGMA_INIT = jnp.array([2.0, 2.0, 2.0, 3.0])

REPLACE with exactly the following (K=16):

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
    #   k=13..15 : Reserve RBFs (wide σ, small w — let Block II reshape these)
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
        # Q1(col 0-3, row 4-7): O arc at (2,6),(3,6),(1,5),(1,4) → centroid≈(1.75,5.25)
        [ 1.75,  5.25,  0.5],   # k=3  Q1
        # Q2(col 4-7, row 4-7): O arc at (4,6),(5,6),(6,5),(6,4) → centroid≈(5.25,5.25)
        [ 5.25,  5.25,  0.5],   # k=4  Q2
        # Q3(col 0-3, row 0-3): O arc at (1,2),(1,3),(2,1),(3,1) → centroid≈(1.75,1.75)
        [ 1.75,  1.75,  0.5],   # k=5  Q3
        # Q4(col 4-7, row 0-3): O arc at (6,3),(6,2),(5,1),(4,1) → centroid≈(5.25,1.75)
        [ 5.25,  1.75,  0.5],   # k=6  Q4
        # k=7..10  X-diagonal repulsors
        # Placed at centroid of X diagonal pixels in each quadrant
        # Q1 X-diag: (0,7),(1,6),(2,5),(3,4) → centroid=(1.5,5.5)
        [ 1.5,   5.5,   0.5],   # k=7  Q1
        # Q2 X-diag: (7,7),(6,6),(5,5),(4,4) → centroid=(5.5,5.5)
        [ 5.5,   5.5,   0.5],   # k=8  Q2
        # Q3 X-diag: (3,3),(2,2),(1,1),(0,0) → centroid=(1.5,1.5)
        [ 1.5,   1.5,   0.5],   # k=9  Q3
        # Q4 X-diag: (4,3),(5,2),(6,1),(7,0) → centroid=(5.5,1.5)
        [ 5.5,   1.5,   0.5],   # k=10 Q4
        # k=11  O path guide: midpoint O-start→O-attractor
        [ 5.0,   5.0,   0.5],   # k=11
        # k=12  X path guide: midpoint X-start→X-attractor (note: start≈(3.5,3.5))
        [-2.0,  -2.0,   0.5],   # k=12
        # k=13..15  Reserve RBFs (evenly spread, large σ)
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

══════════════════════════════════════════════════════════════
MODIFICATION 2 — Update hardcoded "K=4" strings (cosmetic only)
══════════════════════════════════════════════════════════════

FIND and REPLACE the following three occurrences of "K=4":

(A) In the verify_and_plot function, the summary text line:
    FIND:    f"  gamma={GAMMA}  T={T_FINAL}  dt={DT}  K=4\",
    REPLACE: f"  gamma={GAMMA}  T={T_FINAL}  dt={DT}  K=16\",

(B) In fig.suptitle at the bottom of verify_and_plot:
    FIND:    f"gamma={GAMMA}  T={T_FINAL}  dt={DT}  K=4 RBF  [JAX lax.scan / CUDA]\",
    REPLACE: f"gamma={GAMMA}  T={T_FINAL}  dt={DT}  K=16 RBF  [JAX lax.scan / CUDA]\",

(C) In the legend/label of the potential contour plot [1,1],
    the zip list has exactly 4 entries (one per RBF):

    FIND this block:
        for k, (col, mrk, sz, lab) in enumerate([
                ('blue', '*', 220, 'O-attractor'),
                ('red',  '*', 220, 'X-attractor'),
                ('darkorange', '^', 160, 'Barrier'),
                ('green', 'D', 110, 'Path guide')]):
            a11.scatter(*mu_np[k, :2], s=sz, c=col, marker=mrk, zorder=6, label=lab)

    REPLACE with:
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

══════════════════════════════════════════════════════════════
MODIFICATION 3 — Update the docstring NOTE (informational only)
══════════════════════════════════════════════════════════════

FIND in the module docstring:
    NOTE on physics parameters:
      The K=4 fixed parameters have limited convergence because both O- and X-
      image particles start in [0,7]^2, while the X-attractor is at (-8,-8) --
      distance ~16, far outside the Gaussian support (3*sigma=6).
      This is intentional: Block I tests the INFRASTRUCTURE (RK4, energy
      decrease, contact geometry). Block II will LEARN optimal parameters.

REPLACE with:
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

══════════════════════════════════════════════════════════════
WHAT NOT TO CHANGE
══════════════════════════════════════════════════════════════

Do NOT modify any of the following — they are verified correct:
  - contact_rhs()          (physics equations)
  - rbf_potential()        (RBF formula)
  - rbf_gradient()         (gradient formula)
  - rk4_step()             (RK4 integrator)
  - simulate() / lax.scan  (time loop)
  - preprocess()           (coordinate convention x=col, y=7-row)
  - compute_com()
  - compute_hamiltonian()
  - compute_phase_volume()  (t_ref normalization fix already in code)
  - classify()
  - GAMMA, T_FINAL, DT, N_STEPS, N_MAX, TAU
  - Q_STAR_O, Q_STAR_X
  - All verify_and_plot() logic EXCEPT the three string replacements above

══════════════════════════════════════════════════════════════
PASS CRITERIA (unchanged from original)
══════════════════════════════════════════════════════════════

Block I is FULL PASS when ALL hold:
  ✓ ENERGY:     H(t) monotone non-increasing ≥ 95% of particles
  ✓ VOLUME:     R² (empirical vs exp(-3γt)) > 0.90
  ✓ CONV q:     ε_q(T) < 2.0 for both O and X
  ✓ CONV p:     ε_p(T) < 0.5 for both O and X
  ✓ CLASS:      O → 'O',  X → 'X'

If FULL PASS is not achieved with this initialization, print the
final CoM positions and distances so parameter tuning can proceed.
Do not attempt to auto-tune parameters — output diagnostics only.

══════════════════════════════════════════════════════════════
DELIVERABLE
══════════════════════════════════════════════════════════════

Output the complete modified block_i.py as a single code block.
No explanation needed — just the full file.