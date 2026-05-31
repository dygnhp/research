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
  evaluation.py               # Shared evaluation module (block-agnostic)
  block_ii_prototype.py       # Original prototype (reference only)
  block_i_prototype/
    block_i.py                # Block I forward simulator (verified)
  block_ii/
    block_ii.py               # Block II learning loop (dataset-driven)
    block2_trained_params.npy # Final trained parameters
    block2_verification.png   # 6-panel verification figure
  evaluation_system/
    evaluate_block_ii.py      # Block II evaluation (uses evaluation.py)
    block3_verification.png   # 8-panel paper-ready figure
  block_iii/
    block_iii.py              # Block III: N-dim adaptive processing
    block3_trial{NN}_*.npy    # Per-trial trained params
    block3_trial{NN}_*.json   # Per-trial history + growth log
    block3_trial{NN}_*.png    # Per-trial evaluation figure
  reference/
    progress.md               # This file
    block_features.md         # Block features overview
    dataset_used/
      save_dataset.py         # Dataset saver (reusable for all blocks)
      block2_dataset.npz      # Block II training dataset
      block3_trial{NN}_*.npz  # Per-trial datasets (Block III)
      block3_trial{NN}_config.json  # Per-trial configs
```

---

## Program Structure

### Block II: Learning Loop

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT                                                       │
│  ├── data_generator.py → generate_dataset(n=50, seed=42)    │
│  │     O_images[50] (8x8 binary), X_images[50] (8x8 binary)│
│  ├── Canonical O/X (index 0) for validation                 │
│  └── Initial params θ₀ = {w_k, μ_k, σ_raw_k} (K=16)       │
│        k=0,1: frozen attractors  (±8,±8)                    │
│        k=2,3: stepping stones    (data-proximal)            │
│        k=4..15: free RBFs        (inside [0,7]²)            │
├─────────────────────────────────────────────────────────────┤
│  PROCESSING (per epoch)                                      │
│  1. Sample random O/X variant from dataset                  │
│  2. Preprocess → contextual 3D lifting (x, y, z_conn)      │
│     - z ≈ 0.88 for axis-connected (O-like)                 │
│     - z ≈ 0.12 for diag-connected (X-like)                 │
│  3. Pack to S0 ∈ ℝ^{N_MAX × 7} = [q(3), p(3), z(1)]      │
│  4. Simulate: lax.scan × 200 RK4 steps (T=10.0, dt=0.05)  │
│     - dq/dt = p                                             │
│     - dp/dt = -∇V_RBF(q; θ) - γp                           │
│     - dz/dt = ‖p‖² - H                                     │
│  5. Compute loss:                                            │
│     L = ‖CoM_O(T) - q*_O‖² + ‖CoM_X(T) - q*_X‖²          │
│         + λ_p (‖p_O(T)‖² + ‖p_X(T)‖²)                     │
├─────────────────────────────────────────────────────────────┤
│  LEARNING                                                    │
│  1. ∇L via jax.value_and_grad (full BPTT through scan)     │
│  2. jax.checkpoint on scan body → O(√N) memory             │
│  3. Adam optimizer + warmup-cosine LR (peak=5e-3)          │
│  4. Gradient clipping (max_norm=1.0)                        │
│  5. σ positivity: softplus(σ_raw) + 0.1                    │
│  6. Validation on canonical images every 20 epochs          │
├─────────────────────────────────────────────────────────────┤
│  OUTPUT                                                      │
│  ├── block2_trained_params.npy    (final θ)                 │
│  ├── block2_params_ep{N}.npy     (checkpoints)             │
│  ├── block2_verification.png      (6-panel figure)          │
│  └── Console: loss curve, CoM trajectory, classification    │
└─────────────────────────────────────────────────────────────┘
```

### Block III: Generalization & Robustness Evaluation

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT                                                       │
│  ├── Trained params θ from Block II (.npy)                  │
│  │   (or fallback lightweight training if unavailable)      │
│  ├── Canonical O/X images                                   │
│  ├── Generated variants (20 O + 20 X from data_generator)  │
│  └── Novel patterns: T, L, +, Square (hardcoded, unseen)   │
├─────────────────────────────────────────────────────────────┤
│  PROCESSING (forward-only, no gradient)                      │
│  Step 1. Baseline: canonical O/X → simulate → classify      │
│  Step 2. Dataset variants: 20+20 images → accuracy          │
│  Step 3. Noise sweep: 0-10 pixel flips × 5 trials/level    │
│          → accuracy vs. corruption curve                    │
│  Step 4. Shift sweep: dx,dy ∈ {-2,-1,0,1,2} (25 combos)   │
│          → spatial robustness grid                          │
│  Step 5. Novel patterns: T/L/+/□ → which attractor?        │
│  Step 6. Ablation:                                           │
│          (a) Full model (all K=16)                          │
│          (b) No stepping stones (w₂=w₃=0)                  │
│          (c) No free RBFs (w₄..w₁₅=0)                      │
│          (d) Attractors only (w₀,w₁ only)                  │
│  Step 7. Gamma sensitivity: γ ∈ {0.5, 1.0, 1.5, 2.0, 3.0} │
│                                                              │
│  Physics: same contact Hamiltonian as Block II               │
│  Simulation: simulate_eval (JIT, no checkpoint needed)       │
├─────────────────────────────────────────────────────────────┤
│  OUTPUT                                                      │
│  ├── block3_verification.png   (8-panel paper-ready figure) │
│  │     Panel 1: O trajectory        Panel 2: X trajectory  │
│  │     Panel 3: noise sweep (O)     Panel 4: noise sweep (X)│
│  │     Panel 5: shift heatmap (O)   Panel 6: shift heatmap (X)│
│  │     Panel 7: ablation bar chart  Panel 8: gamma curve    │
│  └── Console: full quantitative summary per step            │
│                                                              │
│  (No LEARNING in Block III -- evaluation only)               │
└─────────────────────────────────────────────────────────────┘
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
| 2026-03-30 | Dataset saved to reference/dataset_used/                |
| 2026-03-30 | Block II/III structure documented, Block IV TODO drafted |
| 2026-03-31 | Block III restructured -> shared evaluation.py           |
| 2026-03-31 | Block IV built: N-dim adaptive processing (D/K growth)   |
| 2026-03-31 | Block IV tested: component + 200ep training (K:16->24)   |

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

---

## Shared Evaluation Module (NEW)

**File**: `evaluation.py`
**Status**: Complete, tested.

### Purpose
Extracted from Block III into a block-agnostic evaluation module.
Any block passes its own `preprocess_fn`, `simulate_fn`, `full_params_fn`
to get the full evaluation suite: noise sweep, shift sweep, ablation,
gamma sensitivity, and 8-panel figure generation.

### Functions
- `evaluate_single()` / `evaluate_suite()` -- classify images
- `noise_sweep()` / `shift_sweep()` -- robustness
- `ablation_study()` / `gamma_sweep()` -- component analysis
- `make_eval_figure()` -- 8-panel paper-ready figure
- `run_standard_eval()` -- full pipeline in one call

---

## Block IV: N-Dimensional Adaptive Processing (BUILT)

**File**: `block_iv/block_iv.py`
**Status**: Code complete, component-tested, 200-epoch validation passed.

### Architecture
- **Config class**: all hyperparameters in one place, overridable
- **N-dimensional preprocessing**: D-agnostic feature extraction
  - D=3: (x, y, z_connectivity) -- same as Block II
  - D=4: + local_density (3x3 window count / 9)
  - D=5+: zeros (available for learned features)
- **N-dimensional RBF**: potential and gradient work for any D
- **N-dimensional Contact Hamiltonian**: q,p in R^D
- **Adaptive growth**: PlateauDetector + grow_K + grow_D
- **TrialManager**: per-trial dataset/config/params/log saving

### Growth strategy
1. Train with (D_init=3, K_init=16) until loss plateau
2. On plateau: add K_GROW=4 new RBFs (up to K_max=32)
3. If K maxed and still plateau: expand D -> D+1 (up to D_max)
4. Each growth rebuilds optimizer state (warm-start params)
5. New RBFs placed randomly in data domain [0,7]^2
6. New dimensions initialized to 0 (preserves existing landscape)

### Validation (200-epoch CPU test)
- Training loop runs correctly, loss decreases
- K growth triggered twice: 16 -> 20 -> 24 at epochs ~95 and ~182
- Plateau detection works (window=30, threshold=0.02)
- Dataset + config saved to reference/dataset_used/ per trial
- Growth log saved as JSON
- Full evaluation pipeline runs via shared evaluation.py

### Pending
- Full GPU training run (3000 epochs, D_max=5, K_max=32)
- D growth validation (requires longer training to hit K_max plateau)
- Dimension contribution analysis (D=3 vs D=4 vs D=5)

---

## Block IV: N-Dimensional Adaptive Processing (TODO)

### Concept

Block II/III는 고정 3D (x, y, z) + 고정 K=16 RBF로 동작한다.
Block IV는 **학습 과정에서 차원 D와 RBF 개수 K를 동적으로 증가**시키는
적응적 프로세싱을 도입한다. 핵심 아이디어:

- **차원 확장 (Dimension Growth)**: 분류 성능이 정체(plateau)되면
  상태 벡터에 새로운 차원을 추가하여 더 풍부한 표현 공간을 확보.
  q ∈ ℝ³ → ℝ⁴ → ℝᴰ, 새 차원은 0으로 초기화 (warm-start).
- **K 확장 (Basis Growth)**: 기존 K개의 RBF가 수렴한 후
  새로운 RBF를 잔차(residual)가 큰 영역에 추가. K=16 → K=24 → K=K_max.
- **시행(trial)별 데이터셋 분리 저장**: 각 실험 설정(D, K, seed)에
  대응하는 데이터셋과 결과를 독립적으로 기록하여 재현성 보장.

### Architecture Sketch

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT                                                       │
│  ├── data_generator.py (확장: 더 큰 이미지, 더 많은 클래스) │
│  ├── Lifting: image → q ∈ ℝ^D (D는 학습 중 증가)           │
│  │   - D=3: (x, y, z_conn)           ← Block II와 동일     │
│  │   - D=4: + z_density (local density feature)             │
│  │   - D=5+: + learned features via autoencoder embedding   │
│  └── Config: D_init, D_max, K_init, K_max, growth schedule │
├─────────────────────────────────────────────────────────────┤
│  PROCESSING                                                  │
│  Phase 1 (D=D_init, K=K_init):                              │
│    Standard Block II training until plateau detection        │
│  Phase 2 (growth trigger):                                   │
│    IF loss plateau for P epochs:                             │
│      → Option A: Add dimension (D → D+1)                    │
│        - Extend q, p, μ_k with 0-init new components        │
│        - Extend σ_k to D+1 (or use isotropic σ)             │
│      → Option B: Add K_new RBFs at high-residual locations  │
│        - μ_new = argmax_q ‖CoM(T) - q*‖ along trajectory   │
│        - w_new = small, σ_new = local scale                 │
│    Resume training with expanded model                       │
│  Phase 3: Repeat until D_max or K_max or convergence        │
├─────────────────────────────────────────────────────────────┤
│  LEARNING                                                    │
│  - Same BPTT + Adam, but params shape changes at growth     │
│  - Optimizer state re-initialized for new parameters only   │
│  - Existing params warm-started (no reset)                  │
│  - Growth decision: moving-average loss stagnation detector │
├─────────────────────────────────────────────────────────────┤
│  OUTPUT (per trial)                                          │
│  ├── reference/dataset_used/block4_trial{NN}_dataset.npz   │
│  ├── block_iv/trial{NN}_params_D{D}_K{K}.npy               │
│  ├── block_iv/trial{NN}_growth_log.json                     │
│  │     (epoch, event, D_before→D_after, K_before→K_after)   │
│  └── block_iv/trial{NN}_figure.png                          │
└─────────────────────────────────────────────────────────────┘
```

### TODO Checklist

#### Phase 0: Infrastructure
- [ ] `block_iv/block_iv.py` 스켈레톤 생성 (Block II 기반 fork)
- [ ] **시행별 데이터셋 저장 코드 작성**
  - `save_dataset.py`에 `tag="block4_trial{NN}"` 호출 추가
  - 각 trial의 config (D, K, seed, hyperparams)를 JSON으로 함께 저장
  - `reference/dataset_used/block4_trial{NN}_dataset.npz` 자동 생성
  - `reference/dataset_used/block4_trial{NN}_config.json` 자동 생성
- [ ] Trial manager 클래스 작성: config → dataset 생성 → 학습 → 결과 저장

#### Phase 1: N-Dimensional Lifting
- [ ] `preprocess()` 일반화: `D` 파라미터 도입 (D=3이면 기존과 동일)
- [ ] D=4 feature: local density (3x3 window sum / 9)
- [ ] D=5+ features: learnable embedding (small MLP or fixed features)
- [ ] RBF potential 일반화: `μ_k ∈ ℝ^D`, `σ_k ∈ ℝ^D` (anisotropic 옵션)
- [ ] Contact Hamiltonian RHS 일반화: q,p ∈ ℝ^D → dq/dt, dp/dt ∈ ℝ^D
- [ ] Attractor 좌표 D차원 확장: q*_O, q*_X ∈ ℝ^D

#### Phase 2: Adaptive K Growth
- [ ] Plateau detector: moving-average loss (window=100), trigger if Δ < ε
- [ ] RBF insertion strategy: residual-guided placement
  - 최종 시간 T에서 CoM과 target 사이 중간점에 신규 RBF 배치
- [ ] Parameter vector resize: w, μ, σ_raw 배열 확장 + optimizer state 재구성
- [ ] Growth cap: K_max (e.g., 64) 초과 시 성장 중단

#### Phase 3: Adaptive Dimension Growth
- [ ] Dimension growth trigger: K growth로도 plateau 해소 실패 시
- [ ] State vector resize: S0 ∈ ℝ^{N×(2D+1)} 차원 확장
  - 기존 차원 보존, 신규 차원 q=0, p=0으로 초기화
- [ ] μ_k 확장: 신규 차원 성분 = 0 (기존 landscape 보존)
- [ ] σ_k 확장: 신규 차원 성분 = 기존 σ 평균 (isotropic start)
- [ ] JIT 재컴파일 처리: shape 변경 시 `jit` 캐시 무효화 전략

#### Phase 4: Evaluation & Comparison
- [ ] Block III 평가 파이프라인을 D-agnostic하게 확장
- [ ] Trial별 비교표 자동 생성: (D, K) → accuracy, eps_q, loss
- [ ] 차원별 기여도 분석: D=3 vs D=4 vs D=5 성능 비교
- [ ] Growth trajectory 시각화: epoch vs (D, K, loss) 3축 plot

#### Phase 5: Paper Integration
- [ ] `reference/progress.md` Block IV 섹션 추가
- [ ] 논문용 figure: adaptive growth curve, dimension ablation
- [ ] 계산 비용 분석: D×K 증가에 따른 wall-clock time scaling
