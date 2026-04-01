# Contact Hamiltonian Fluid Neural Network -- Block Features

## Overview

A physically-interpretable image classifier where 8x8 binary images are
treated as particle ensembles flowing under contact Hamiltonian dynamics
over a learnable RBF potential landscape. Classification emerges from the
attractor basin each ensemble converges to.

**Core dynamics**: dq/dt = p, dp/dt = -grad V - gamma*p, dz/dt = ||p||^2 - H

---

## Block I: Forward Simulator

**Location**: `block_i_prototype/block_i.py`

### Purpose
Verify that contact Hamiltonian physics produces the desired behavior:
monotone energy decrease, phase-volume contraction, and attractor convergence.

### Features
| Feature | Description |
|---------|-------------|
| Contact Hamiltonian integrator | RK4, dt=0.05, T=10.0, 200 steps |
| K=16 structured RBF potential | Frozen attractors + stepping stones + free |
| Physics verification | dH/dt <= 0, V_phase ~ exp(-3*gamma*t) |
| 6-panel verification figure | Potential landscape, trajectories, energy curves |

### Input / Output
- **Input**: Hardcoded O/X canonical images (8x8 binary)
- **Processing**: Forward simulation with fixed (non-learnable) params
- **Output**: 6-panel figure, PASS/FAIL for each physics check

### Key Finding
K=4 FAILED: X-attractor at (-8,-8) is ~17 units from data in [0,7]^2.
RBF force at that distance: exp(-18) ~ 10^-8 (gradient vanishes).
Resolved by K=16 with per-quadrant discriminator RBFs.

---

## Block II: Learning Loop

**Location**: `block_ii/block_ii.py`

### Purpose
Learn the RBF potential landscape parameters via gradient descent (BPTT)
so that particle ensembles converge to the correct attractor.

### Features
| Feature | Description |
|---------|-------------|
| Full BPTT | `jax.value_and_grad` through `jax.lax.scan` |
| Memory-efficient | `jax.checkpoint` on scan body -> O(sqrt(N)) memory |
| Dataset-driven training | Random O/X variant sampled per epoch (50/class) |
| Canonical validation | Every LOG_EVERY epochs on fixed O/X images |
| Warmup-cosine LR | Peak=5e-3, warmup=100, cosine decay to 1e-5 |
| Gradient clipping | Global norm = 1.0 |
| Softplus sigma | sigma = softplus(sigma_raw) + 0.1 (always positive) |
| Structured K=16 | k=0,1 frozen attractors; k=2,3 stepping stones; k=4..15 free |
| Checkpoint saving | Every SAVE_EVERY epochs (.npy) |
| 6-panel verification figure | Potential, trajectories, loss curves, convergence |

### Input -> Processing -> Output
```
Input:  data_generator.py -> 50 O + 50 X variants (8x8 binary)
        Initial params theta_0 = {w, mu, sigma_raw} for K=16 RBFs

Processing (per epoch):
  1. Sample random O/X variant from dataset
  2. Preprocess -> 3D lifting (x, y, z_connectivity)
  3. Simulate: lax.scan x 200 RK4 steps
  4. Loss = ||CoM_O(T) - q*_O||^2 + ||CoM_X(T) - q*_X||^2 + lambda_p * momentum
  5. BPTT gradient -> Adam update

Output: block2_trained_params.npy, block2_verification.png
```

---

## Evaluation System

**Location**: `evaluation_system/evaluate_block_ii.py` + `evaluation.py`

### Purpose
Comprehensive evaluation of any block's trained parameters. Block-agnostic
shared module (`evaluation.py`) + block-specific runners.

### Features
| Feature | Description |
|---------|-------------|
| Shared evaluation module | `evaluation.py` -- any block can use it |
| Baseline classification | Canonical O/X forward pass |
| Dataset variant accuracy | 20 generated O/X variants |
| Noise robustness sweep | 0-10 pixel flips, 5 trials/level |
| Shift robustness sweep | dx,dy in {-2,-1,0,1,2} = 25 combos |
| Novel pattern evaluation | T, L, +, Square (unseen during training) |
| Ablation study | Full / no stones / no free / attractors only |
| Gamma sensitivity | gamma = {0.5, 1.0, 1.5, 2.0, 3.0} |
| 8-panel paper-ready figure | Potential, noise, shift, novel, ablation, gamma, trajectories, summary |
| Fallback training | Lightweight training if Block II params unavailable |

### Architecture
```
evaluation.py (shared, block-agnostic)
  -> evaluate_single(image, w, mu, sigma, preprocess_fn, simulate_fn, ...)
  -> evaluate_suite(), noise_sweep(), shift_sweep()
  -> ablation_study(), gamma_sweep()
  -> make_eval_figure(), run_standard_eval()

evaluation_system/evaluate_block_ii.py (Block II specific)
  -> Defines D=3 preprocessing, K=16 params, simulate_eval
  -> Loads Block II trained params
  -> Calls run_standard_eval() with Block II functions
```

---

## Block III: N-Dimensional Adaptive Processing

**Location**: `block_iii/block_iii.py`

### Purpose
Extend the fixed (D=3, K=16) architecture with adaptive growth:
dimensions and RBF count increase during training when loss plateaus.

### Features
| Feature | Description |
|---------|-------------|
| Variable dimension D | D_INIT=3 up to D_MAX (configurable, default 6) |
| Variable K | K_INIT=16 up to K_MAX (configurable, default 32) |
| Plateau detection | Moving-average loss, triggers growth on stagnation |
| K growth | Add K_GROW (default 4) new RBFs at data-domain locations |
| D growth | Extend state vector q,p from R^D to R^(D+1), warm-start |
| N-dim preprocessing | D=3: (x,y,z_conn), D=4: +density, D=5+: extensible |
| N-dim RBF potential | Fully D-agnostic V(q) and grad V(q) |
| N-dim Contact Hamiltonian | q,p in R^D, dz/dt = ||p||^2 - H |
| Config class | All hyperparams in one object, overridable |
| TrialManager | Per-trial dataset/config/params/growth-log saving |
| Per-trial dataset saving | reference/dataset_used/block3_trial{NN}_dataset.npz |
| Growth logging | JSON log of (epoch, event, D_before, D_after, K_before, K_after) |
| Evaluation integration | Uses shared evaluation.py for 8-panel figure |

### Growth Strategy
```
Phase 1: Train with (D_init, K_init) until plateau
Phase 2: Plateau detected -> add K_GROW RBFs (if K < K_MAX)
Phase 3: K maxed + still plateau -> grow D (D -> D+1, up to D_MAX)
  - Existing params warm-started (new dims initialized to 0)
  - Optimizer state rebuilt for new param shapes
  - Training data re-preprocessed for new D
Repeat until convergence or D_MAX + K_MAX reached
```

### Input -> Processing -> Output
```
Input:  data_generator.py -> 50 O + 50 X variants
        Config(D_INIT, D_MAX, K_INIT, K_MAX, K_GROW, ...)

Processing (adaptive loop):
  1. Train with current (D, K) until plateau
  2. On plateau: grow K or D
  3. Rebuild simulator, data, optimizer for new shape
  4. Resume training with warm-started params
  5. Repeat until convergence

Output (per trial):
  block3_trial{NN}_params_D{D}_K{K}.npy    -- trained params
  block3_trial{NN}_growth_log.json          -- growth events
  block3_trial{NN}_history.json             -- training history
  block3_trial{NN}_eval_D{D}_K{K}.png      -- evaluation figure
  reference/dataset_used/block3_trial{NN}_dataset.npz
  reference/dataset_used/block3_trial{NN}_config.json
```

### Dimension Features
| Dim | Feature | Description |
|-----|---------|-------------|
| d=0 | x | Image column coordinate |
| d=1 | y | Image row (inverted: 7-r) |
| d=2 | z_connectivity | sigmoid(axis_neighbors - diag_neighbors) |
| d=3 | local_density | 3x3 window pixel count / 9 |
| d=4+ | (reserved) | Zeros; available for learned features |

---

## Shared Components

### data_generator.py
Parametric O/X image generator with controlled variation.
- `generate_O()` / `generate_X()`: parametric shapes
- `generate_random_O()` / `generate_random_X()`: randomized variants
- `generate_dataset(n_per_class, seed)`: balanced dataset
- `O_CANONICAL` / `X_CANONICAL`: exact canonical images

### evaluation.py
Block-agnostic evaluation module (see Evaluation System above).

### reference/dataset_used/save_dataset.py
Reusable dataset saver. Saves `.npz`, summary `.txt`, and grid `.png`.
Called with `tag="block3_trial01"` for per-trial isolation.

---

## Dependency Graph

```
data_generator.py
  |
  +-- block_ii/block_ii.py          (imports generate_dataset)
  +-- evaluation_system/             (imports O/X_CANONICAL)
  +-- block_iii/block_iii.py         (imports generate_dataset)
  +-- reference/dataset_used/        (imports for saving)

evaluation.py
  |
  +-- evaluation_system/evaluate_block_ii.py  (imports run_standard_eval)
  +-- block_iii/block_iii.py                  (imports run_standard_eval)

evaluation_system/evaluate_block_ii.py
  |
  +-- block_iii/block_iii.py   (imports NOVEL_IMAGES)
```
