# Contact Hamiltonian Fluid Neural Network -- Research Progress

## Project Overview

A physically-interpretable image classifier where 8x8 binary images (O/X patterns)
are treated as particle ensembles flowing under contact Hamiltonian dynamics over a
learnable RBF potential landscape. Classification emerges from the attractor basin
each ensemble converges to.

**Core equation**: dq/dt = p, dp/dt = -grad V - gamma*p, dz/dt = ||p||^2 - H

**Key insight**: Contact Hamiltonian dynamics guarantees dH/dt = -gamma||p||^2 <= 0
(energy monotone decrease) and phase-volume contraction V_phase ~ exp(-3*gamma*t),
providing built-in convergence guarantees that standard neural networks lack.

---

## Block I: Forward Simulator (COMPLETE)

**File**: `block_i_prototype/block_i.py`
**Status**: Verified, all physics checks pass.

### What was built
- Contact Hamiltonian dynamics integrator (RK4, dt=0.05, T=10.0)
- K=16 structured RBF potential with frozen parameters
- 6-panel verification figure

### Results
- Energy monotone decrease: PASS (>95% of particles)
- Phase-volume contraction R^2 > 0.90: PASS
- Convergence eps_q, eps_p: PASS
- Classification O->O, X->X: PASS with K=16 structured initialization

### Key finding
K=4 initialization FAILED classification because both O and X images start in [0,7]^2
while X-attractor is at (-8,-8). Gaussian force at distance ~17: exp(-18) ~ 10^{-8}.
Resolved by K=16 with per-quadrant discriminator RBFs.

---

## Block II: Learning Loop (REFACTORED)

**File**: `block_ii/block_ii.py`
**Status**: Code complete, refactored for dataset-driven training.

### Architecture
- K=16 RBFs: k=0,1 frozen attractors; k=2,3 stepping stones; k=4..15 free
- Optimizer: Adam + warmup-cosine LR + gradient clipping
- Gradient: Full BPTT via jax.value_and_grad through lax.scan + jax.checkpoint
- Loss: ||CoM_O(T) - q*_O||^2 + ||CoM_X(T) - q*_X||^2 + lambda_p * momentum_penalty

### Refactoring (2026-03-29)
- **Dataset-driven training**: Each epoch samples a random O/X variant from
  a pre-generated dataset (50 images/class) instead of using fixed images.
- **Data generator**: `data_generator.py` provides parametric O (ring/ellipse)
  and X (diagonal cross) generators with controlled variation.
- **Validation on canonical images**: Early stopping evaluated on the original
  fixed O/X images, ensuring backward compatibility.
- **Bug fix from prototype**: scan body calls rk4_step once (prototype called twice).

### Training run results (CPU, 2026-03-29)
- **Run 1** (N_EPOCHS=1000, PEAK_LR=1e-3): FAILED — both images predicted as O.
  Root cause: X stepping stone at (-2,-2) too far from data domain; LR too low.
- **Fix applied**: X stone → (0,0,0.12), sigma 2.0→3.0, weights -0.5→-1.0,
  z-values matched to attractor z, LR→5e-3, epochs→3000.
- **Run 2** (N_EPOCHS=3000, PEAK_LR=5e-3): Classification PASS (O→O, X→X).
  - Final loss: ~150-350 (fluctuates due to stochastic dataset sampling)
  - O CoM converges toward O-attractor direction, X toward X-attractor direction
  - eps_q still >2.0 (O: 9.03, X: 10.56) — particles correctly separated but
    don't fully reach (±8,±8) targets within CPU training budget
  - Full convergence requires GPU training with more epochs

---

## Block III: Generalization & Robustness Evaluation (NEW)

**File**: `block_iii/block_iii.py`
**Status**: Code complete, pending first test run.

### Evaluation pipeline
1. Load trained params from Block II (or fallback training)
2. Baseline evaluation on canonical O/X
3. Dataset-variant evaluation (20 generated O/X variants)
4. Noise robustness sweep (0-10 pixel flips, 5 trials/level)
5. Shift robustness sweep (dx,dy in {-2,-1,0,1,2})
6. Novel pattern evaluation (T, L, +, Square -- unseen classes)
7. Ablation studies:
   - Full model (all K=16)
   - No stepping stones (w_2=w_3=0)
   - No free RBFs (w_4..w_15=0)
   - Attractors only (w_0, w_1 only)
8. Gamma sensitivity analysis (gamma = 0.5, 1.0, 1.5, 2.0, 3.0)
9. 8-panel paper-ready verification figure

### Test run results (CPU-trained params, 2026-03-29)
- **Baseline**: Classification PASS (O→O, X→X)
- **Dataset variants** (20 O + 20 X): 88% accuracy (O: 80%, X: 95%)
- **Noise robustness** (0-10 pixel flips, 5 trials/level):
  - O: excellent — 100% accuracy up to 8 flips
  - X: fragile — accuracy drops rapidly with noise
- **Shift robustness** (dx,dy ∈ {-2,-1,0,1,2}): O 72%, X 76%
- **Novel patterns**: T→O, L→X, +→O, Square→O (all unseen during training)
- **Ablation**: Full model (all K=16) best; attractors-only fails X;
  stepping stones critical for X-class separation
- **Gamma sensitivity**: Works for γ ∈ [1.0, 2.0]; fails at γ=0.5, 3.0
- **Figures**: `block_iii/block3_verification.png` (8-panel)

---

## Data Generator

**File**: `data_generator.py`
**Status**: Complete, tested.

### Design
- **O generator**: Parametric ring/ellipse on 8x8 grid
  - Parameters: center (cx,cy), inner/outer radius, aspect ratio, noise
  - Default params produce exact match with canonical O_IMAGE
- **X generator**: Parametric diagonal cross on 8x8 grid
  - Parameters: center (cx,cy), thickness, arm scale, noise
  - Default params produce exact match with canonical X_IMAGE
- **Dataset**: Balanced O/X with canonical as first entry per class

### Verification
- O canonical match: exact (16 pixels)
- X canonical match: exact (16 pixels)
- Dataset (50/class, seed=42): O mean=16.1px, X mean=20.9px, all >= 8px

---

## File Structure

```
research_main/
  data_generator.py           # Shared O/X image generator (pure numpy)
  block_ii_prototype.py       # Original prototype (reference only)
  block_i_prototype/
    block_i.py                # Block I forward simulator (verified)
    prompt_block_i_initial.md # Original design prompt
    prompt_block_i_edit.md    # K=16 modification prompt
  block_ii/
    block_ii.py               # Block II learning loop (dataset-driven)
  block_iii/
    block_iii.py              # Block III evaluation pipeline
  reference/
    progress.md               # This file
```

---

## Timeline

| Date       | Milestone                                              |
|------------|--------------------------------------------------------|
| (earlier)  | Block I prototype built, K=4 -> K=16 fix, verified    |
| (earlier)  | Block II prototype, then production version created    |
| 2026-03-29 | Block III created, data_generator built                |
| 2026-03-29 | Block II refactored for dataset-driven training        |
| 2026-03-29 | Block II Run 1 failed (stepping stone too far)         |
| 2026-03-29 | Block II Run 2 PASS (classification correct, CPU)      |
| 2026-03-29 | Block III evaluation complete (8-panel figure)          |

---

## Paper-Relevant Observations

### Physics validation (Block I)
- Contact Hamiltonian guarantees monotone energy decrease -- verified empirically
- Phase-volume contraction matches theory exp(-3*gamma*t) with R^2 > 0.90
- These are not hyperparameter-dependent; they hold for any gamma > 0

### Learning challenges (Block II)
- Gradient vanishing at distance: RBF force ~ exp(-d^2/2sigma^2), unusable beyond 3*sigma
- Solution: stepping stones + data-proximal initialization
- Dataset diversity prevents overfitting to single canonical images

### Generalization findings (Block III)
- **Pixel noise**: O highly robust (100% to 8 flips); X fragile. Asymmetry likely
  due to O's ring structure being more resilient to local perturbation than X's
  thin diagonal arms.
- **Spatial shift**: Moderate robustness (~72-76%). Shifts move particles away from
  stepping stone effective range, reducing force coupling.
- **Novel patterns**: All map to one of {O, X}. T→O and Square→O (ring-like
  connectivity); L→X (diagonal-like). The z-channel (local connectivity) is the
  primary discriminator for unseen patterns.
- **Ablation**: Stepping stones are critical for X-class — without them, X particles
  cannot escape the data domain toward the X-attractor. Free RBFs provide modest
  improvement but are not essential for basic classification.
- **Gamma sensitivity**: γ ∈ [1.0, 2.0] works; γ=0.5 is underdamped (oscillations
  prevent convergence); γ=3.0 is overdamped (particles freeze before reaching attractor).
  Optimal range matches theory: γ_crit ≈ 1.41.
