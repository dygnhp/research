# Block II: 구현된 피처 정리

**파일**: `block_ii/block_ii.py`  
**기반**: `block_ii_prototype.py` (GitHub `dygnhp/research`)  
**작성일**: 2026-03-29

---

## 개요

Block II는 Contact Hamiltonian Machine의 학습 루프다. Block I에서 검증한 물리 엔진(에너지 단조감소, 위상부피 수축)을 그대로 유지하면서, O/X 이진분류를 달성하도록 RBF 파라미터 θ = {w_k, μ_k, σ_k}를 경사하강으로 학습한다.

---

## 1. 핵심 설계 결정 — 논의 → 반영 현황

### 1-1. Block I 미임포트 (독립 실행)

**논의**: Block I과 모듈 공유 시 JAX JIT/module import 충돌 발생.  
**반영**: block_i.py를 `import`하지 않고, 모든 물리 함수(RBF, contact_rhs, rk4_step 등)를 block_ii.py 내에 자기완결적으로 복사.

---

### 1-2. Block I 물리 검증 결과 승계

**논의**: Block I에서 확인한 물리적 정합성(dH/dt = -γ‖p‖² ≤ 0)을 Block II에서도 유지해야 한다.  
**반영**:
- `contact_rhs()`, `rk4_step()`, `rbf_potential()`, `rbf_gradient()` 함수 시그니처 및 로직 동일 유지
- `rk4_step`의 `static_argnums=(4, 5)` 규약 (gamma, dt를 Python 스칼라로 고정) 그대로 승계
- `preprocess()`의 3D Lifting 로직 (contextual z = sigmoid(β × (d_axis - d_diag))) 동일

---

### 1-3. Block I 분류 실패 원인 분석 및 해결

**논의**: Block I에서 K=16 structured init으로도 X-클래스 분류 실패.  
**분석**:
```
X-attractor at (-8,-8): ~17 units from initial particles in [0,7]²
RBF force: exp(-17² / (2×4.0)) = exp(-18.1) ≈ 10⁻⁸  <<  machine eps
=> dL/d(μ₁) ≈ 0  =>  μ₁ never moves  =>  gradient-starved
```

**반영**: Data-proximal quasi-flat initialization 채택:
```
k=0,1  FROZEN: O/X attractor (올바른 앵커, 기울기 필요 없음)
k=2,3  STEPPING STONES: (6,6) → O, (-2,-2) → X  (브릿지 역할)
k=4..15 FREE RBFs: 4×3 grid inside [0.5,6.5]×[1.0,6.0]
         → 데이터 도메인 내 배치 → 첫 epoch부터 O(1) force 보장
```

---

### 1-4. Frozen/Learnable 파라미터 분리

**논의**: Attractor 위치는 학습 중 변경되면 안 된다.  
**반영**:
```python
W_FROZEN, MU_FROZEN, SIGMA_FROZEN  # k=0,1 고정값 (JAX 상수)
params_init = {'w': ..., 'mu': ..., 'sigma_raw': ...}  # k=2..15만 학습
full_params(params)  # 매 step마다 concat하여 K=16 완성
```
- `optax`에 `params_init`만 전달 → optimizer가 frozen 파라미터를 건드리지 않음

---

### 1-5. σ softplus 재파라미터화

**논의**: σ_k > 0 제약이 필요하며, rbf_gradient의 분모 σ²에서 division-by-zero 방지.  
**반영**:
```python
sigma_k = softplus(sigma_raw_k) + 0.1    # minimum = 0.1
sigma_raw는 unconstrained → 자유롭게 학습
```
역변환 초기화:
```python
sigma_raw_init = log(exp(sigma_init - 0.1) - 1)  # softplus_inverse
```

---

### 1-6. diffrax 미사용 — lax.scan BPTT

**논의**: adjoint 방법(diffrax)과 full BPTT 중 어느 것을 사용할지.  
**반영**: `jax.lax.scan` + `jax.value_and_grad`를 통한 full BPTT 채택.
- 이유: adjoint는 메모리 효율적이지만 중간 trajectory 접근 불가 → 분류/수렴 확인 불편
- BPTT는 전체 trajectory를 자동미분으로 추적 → 물리 해석과 직접 연결

---

### 1-7. jax.checkpoint 메모리 최적화

**논의**: N_steps=200 step trajectory 전체를 메모리에 유지하면 GPU OOM 위험.  
**반영**: `@jax_checkpoint` (gradient checkpointing)을 scan body에 적용:
- 메모리: O(N_steps) → O(√N_steps)로 감소 (~14 remat checkpoints)
- FLOPs: 역전파 시 전방 재계산 (~2× overhead)
- RTX 4060 (8GB)에서 안전하게 동작

---

### 1-8. simulate_diff 이중 rk4_step 버그 수정

**논의**: 프로토타입 코드 검토 중 발견된 버그.  
**프로토타입 (버그)**:
```python
@jax_checkpoint
def step(S, _):
    return rk4_step(S, w, mu, sigma_safe, gamma, dt), \
           rk4_step(S, w, mu, sigma_safe, gamma, dt)   # 동일 연산 2회 호출!
```
**Production (수정)**:
```python
@jax_checkpoint
def step(S, _):
    S_next = rk4_step(S, w, mu, sigma_safe, gamma, dt)  # 1회 계산
    return S_next, S_next                                 # carry, output 공유
```
**효과**: scan 200 step × rk4 4 stage = 800 `contact_rhs` 호출 → 1600→800으로 절반 감소.

---

### 1-9. 손실함수 — 위치항 + 모멘텀 패널티

**논의**: 최종 CoM 위치만 최소화하면 입자들이 attractor 근처에서 진동 가능.  
**반영**:
```
L(θ) = ‖CoM_O(T) - q*_O‖² + ‖CoM_X(T) - q*_X‖²
      + λ_p × (mean_i ‖p_i^O(T)‖² + mean_i ‖p_i^X(T)‖²)
```
- λ_p = 0.1: 모멘텀 패널티 계수
- 역할: T 시점에서 입자들이 정지 상태(p→0)에 가깝게 → 분류 경계 선명화

---

### 1-10. Optimizer — Adam + Warmup-Cosine Schedule

**논의**: 단순 Adam vs. 학습률 스케줄링.  
**반영**: `optax.chain` 구성:
```
1. clip_by_global_norm(1.0)           # 기울기 폭발 방지
2. adam(lr=warmup_cosine_schedule)     # 적응형 업데이트
   - warmup 50 steps (0 → 1e-3)
   - cosine decay (1e-3 → 1e-5) over 1000 epochs
```

---

### 1-11. train_step JIT 컴파일

**논의**: 매 epoch마다 Python 오버헤드 없이 GPU 실행.  
**반영**: `@jit` 데코레이터로 전체 forward+backward+optimizer 단일 XLA kernel화.
- 첫 실행: 30~120s (XLA compilation)
- 이후 실행: GPU bound (~수초/epoch)

---

### 1-12. PyCharm Unresolved Reference 방지

**논의**: PyCharm에서 JAX-traced nested function 내 외부 함수 참조 시 IDE 경고 발생.  
**반영**: block_i.py 규약 그대로 적용:
```python
_rbf_potential = rbf_potential   # 모듈 수준 명시적 참조
_rbf_gradient  = rbf_gradient
# contact_rhs 내부에서 _rbf_potential, _rbf_gradient 사용
```

---

### 1-13. 출력 경로 — `__file__` 기반

**논의**: PyCharm에서 실행 시 CWD가 project root로 설정되므로, 출력 파일이 의도치 않은 위치에 저장될 수 있음.  
**반영**: `pathlib.Path(__file__).resolve().parent`를 `_HERE`로 정의:
```python
_HERE = Path(__file__).resolve().parent
# 모든 출력 파일: _HERE / "block2_verification.png" 등
```
→ `block_ii/` 디렉토리 내에 자동 저장.

---

### 1-14. 조기 종료 (Early Stopping)

**논의**: 1000 epoch 전에 수렴하면 불필요한 학습 방지.  
**반영**: 매 LOG_EVERY(10) epoch마다 조건 확인:
```python
if pred_O == 'O' and pred_X == 'X'
   and eps_q_O < CONV_Q_THR and eps_q_X < CONV_Q_THR:
    break
```
- 수렴 정의: 올바른 분류 + eps_q < 2.0

---

## 2. 검증 항목 (Verification Checklist)

| 항목 | 기준 | 코드 위치 |
|------|------|----------|
| 파라미터 형상 | w(16,), mu(16,3), σ(16,) | `__main__` sanity check |
| σ > 0 보장 | jnp.min(sigma) > 0 | `full_params()` |
| 에너지 단조감소 | dH/dt = -γ‖p‖² ≤ 0 | `contact_rhs()` |
| 분류 정확도 | pred_O='O', pred_X='X' | `classify_traj()` |
| eps_q 수렴 | < 2.0 | 학습 루프 early stop |
| eps_p 수렴 | < 0.5 | `make_verification_figure()` |
| 기울기 건전성 | NaN/Inf 없음 | clip_by_global_norm |
| 출력 파일 | block2_verification.png, .npy | `_HERE` 경로 |

---

## 3. 프로토타입 vs. Production 차이 요약

| 항목 | 프로토타입 (`block_ii_prototype.py`) | Production (`block_ii/block_ii.py`) |
|------|--------------------------------------|--------------------------------------|
| rk4_step 이중 호출 | ❌ 버그 있음 (2× FLOPs 낭비) | ✅ 수정 (단일 계산, carry/output 공유) |
| PyCharm 참조 | 직접 호출 (IDE 경고) | `_rbf_potential`, `_rbf_gradient` 명시 |
| 출력 경로 | CWD 상대 경로 | `__file__` 기반 절대 경로 |
| 코드 구조 | 15섹션 (동일) | 15섹션 + 상세 docstring 강화 |
| Block I 관계 설명 | 간략 | 실패 원인 분석 + 해결책 상세 기술 |

---

## 4. 미구현 / Block III로 이관된 항목

| 항목 | 이유 |
|------|------|
| N차원 일반화 (D > 3) | Block III Phase 0 |
| γ 학습 파라미터 | Block III Phase 1 |
| JL 차원 축소 RBF | Block III Phase 2 |
| TurboQuant/QJL 압축 | Block III Phase 3 |
| N-클래스 일반화 | Block III Phase 4 |
| LaTeX 논문 통합 | Block III Phase 5 후 |
