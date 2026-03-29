# Block III: N차원 일반화 + TurboQuant 통합 TODO

**프로젝트**: Contact Hamiltonian Machine (CHM)  
**상태**: 계획 단계  
**선행 조건**: Block II 학습 루프 수렴 확인 (분류 PASS + eps_q < 2.0)

---

## 배경 및 동기

### Block I → II → III 서사

| 블록 | 핵심 기여 | 상태 |
|------|-----------|------|
| I | 물리 엔진 검증 (dH/dt ≤ 0, 위상부피 수축) | ✅ 완료 |
| II | BPTT 경사하강 O/X 이진분류 학습 루프 | ✅ 완료 |
| III | N차원 일반화 + γ 학습 + TurboQuant 압축 | 🔲 예정 |

### Block III의 핵심 문제

Block II는 3차원 위치공간(q ∈ ℝ³)에서만 작동한다. 실제 이미지 분류에서는:

1. **차원 폭발**: N 클래스, 고차원 입력 → 입자 상태 차원 D ≫ 3
2. **메모리 병목**: N_MAX × (2D+1) 상태 텐서가 GPU 메모리를 초과
3. **계산 병목**: RBF 포텐셜 계산 O(N_MAX × K × D)가 D 증가에 따라 선형 증가
4. **γ 고정의 한계**: 단일 γ=1.5는 전체 앙상블에 동일 소산률 → 클래스별 수렴 속도 차이 무시

TurboQuant + QJL (Johnson-Lindenstrauss 보조정리 기반)이 이 세 문제를 동시에 해결한다.

---

## Johnson-Lindenstrauss 보조정리와 CHM의 연결

### JL 보조정리 (원형)

임의의 m개의 점 {x₁, …, xₘ} ⊂ ℝᴰ에 대해, ε ∈ (0,1)이 주어지면

$$k = O\!\left(\frac{\log m}{\varepsilon^2}\right)$$

차원의 선형 사상 f: ℝᴰ → ℝᵏ가 존재하여 모든 쌍 (i,j)에 대해

$$(1-\varepsilon)\|x_i - x_j\|^2 \;\leq\; \|f(x_i) - f(x_j)\|^2 \;\leq\; (1+\varepsilon)\|x_i - x_j\|^2$$

이 성립한다.

### CHM에서의 함의

RBF 포텐셜의 핵심 연산은 **이차거리(squared distance)**다:

$$V(q_i) = \sum_k w_k \exp\!\left(-\frac{\|q_i - \mu_k\|^2}{2\sigma_k^2}\right)$$

JL 보조정리에 의해, ε 오차 범위 내에서 D차원 입자 위치 q_i와 K개의 RBF 센터 μ_k를 모두 포함하는 m = N_MAX + K개의 점은 **k = O(log(N_MAX+K) / ε²)** 차원으로 이사적(isometric) 매핑 가능하다. D = 128일 때 k ≈ 20~30으로 압축 가능 (ε = 0.1 기준).

### QJL 연결: 내적 보존 1-bit 스케치

QJL (Quantized Johnson-Lindenstrauss) [Zandieh et al., AAAI 2025]은 다음을 달성한다:

1. **JL 변환** Φ ∈ ℝ^{k×D} (랜덤 가우시안 행렬): v ↦ Φv ∈ ℝᵏ
2. **1-bit 부호 양자화**: sign(Φv) ∈ {±1}ᵏ
3. **비편향 내적 추정량** (비대칭 추정기):
   $$\widehat{\langle q, \mu \rangle} = \frac{\pi}{2k}\,\mathrm{sign}(\Phi q)^\top (\Phi \mu)$$
   → 분산 O(1/k), overhead 0 (양자화 상수 저장 불필요)

CHM에서 q_i · μ_k 내적이 RBF 지수 안 제곱거리와 동치이므로, QJL로 **RBF 평가를 1-bit 벡터 연산으로 근사**할 수 있다.

---

## TurboQuant 2-Stage Pipeline (ICLR 2026)

TurboQuant [Zandieh & Mirrokni, ICLR 2026]은 두 단계로 구성된다:

### Stage 1 — PolarQuant (주 압축)

- 입력 벡터 v ∈ ℝᴰ를 랜덤 회전 행렬 R로 전처리: ṽ = Rv
- 전처리 후 각 성분의 분포가 균일해지므로 → 스칼라 양자화 최적 조건 달성
- **극좌표 분해**: 크기(norm) r = ‖v‖를 별도 저장, 방향(unit vector) v/r를 양자화
  - 방향 성분: 각도 θ_j = arctan(ṽ_{2j+1}/ṽ_{2j})를 균일 격자로 양자화 (B-1 bits)
  - 크기: r를 1 bit로 부호화 (양자화 상수 overhead 0)
- **이론적 보장**: MSE 최적 왜곡률 달성

### Stage 2 — QJL 잔차 보정 (1-bit)

- Stage 1 잔차 e = v - v̂에 QJL 적용
- 내적 추정량을 **비편향화**: E[⟨q, v̂_TurboQuant⟩] = ⟨q, v⟩
- 전체 비트폭: (B-1) + 1 = B bits (B=3이면 6× 압축)

### CHM Block III 적용

| TurboQuant 구성요소 | CHM 대응 |
|---------------------|----------|
| KV 벡터 | 입자 위치 q_i ∈ ℝᴰ |
| 쿼리 벡터 | RBF 센터 μ_k ∈ ℝᴰ |
| 내적 보존 | exp(-‖q_i - μ_k‖²/2σ²) 근사 |
| 1-bit QJL 잔차 보정 | BPTT 기울기 비편향화 |
| PolarQuant 회전 전처리 | 입자 위치 등방화 (데이터 불균형 보정) |

---

## Phase 계획

### Phase 0: 상태공간 D차원 일반화
**파일**: `block_iii/phase0_nd_state.py`

- [ ] `preprocess()` 확장: 8×8 이미지 → D차원 특징 벡터 (D=64 or 128)
  - 방법: flatten + positional embedding (푸리에 기반)
  - 또는: Conv feature map → D차원 입자 위치
- [ ] `contact_rhs()` 일반화: 3D 하드코딩 제거, 임의 D에 대해 동작
  - `dq_dt = p`  (D-dim)
  - `dp_dt = -grad_q V - γ * p`  (D-dim)
  - `dz_dt = ‖p‖² - H`  (scalar)
  - 상태 차원: S ∈ ℝ^{N_MAX × (2D+1)}
- [ ] `rbf_potential()` / `rbf_gradient()` D-dim 버전 검증
  - 현재 코드는 임의 D에 대해 이미 정확 (shape 의존성 없음)
  - 단, XLA kernel fusion이 고차원에서 유지되는지 확인 필요
- [ ] `simulate_diff()`: n_steps, gamma, dt static 유지 / D만 변경

**검증 기준**: D=3에서 Block II와 동일한 결과 재현 (regression test)

---

### Phase 1: γ 학습 가능 파라미터 확장
**파일**: `block_iii/phase1_learnable_gamma.py`

- [ ] **세 가지 γ 변형 구현 및 비교**:

  a. `scalar_gamma`: 현재 방식, 전체 앙상블 동일 γ (고정 or 학습)
  
  b. `class_gamma`: O-클래스 / X-클래스 별 γ₀, γ₁ 독립 학습
     - 물리적 해석: 두 클래스의 에너지 소산 속도 차별화
     - 해석가능성: γ비 = γ_O/γ_X가 클래스 분리도 지표
  
  c. `particle_gamma`: 입자별 γ_i (i = 1..N_MAX) 학습
     - 주의: `rk4_step`의 static_argnums 위반 → lax.scan 내 동적 처리 필요

- [ ] softplus 재파라미터화: `γ = softplus(γ_raw) + ε_γ` (γ > 0 보장)
- [ ] 학습된 γ 해석: 접촉 다양체의 소산 구조를 데이터에서 학습
- [ ] 이론 검증: `dH/dt = -γ(t) ‖p‖²` 여전히 ≤ 0 (각 입자별)
- [ ] `full_params()` 확장: γ 포함한 frozen/learnable 분리

**검증 기준**: γ_O ≠ γ_X 수렴 → 물리적 의미 논의

---

### Phase 2: JL 차원 축소 기반 RBF 근사
**파일**: `block_iii/phase2_jl_rbf.py`

- [ ] **랜덤 JL 행렬 Φ ∈ ℝ^{k×D}** 생성 (PRNG key 고정)
  - 구현: `Φ = jax.random.normal(key, (k, D)) / sqrt(k)`
  - JL 보조정리: k = ceil(C * log(N_MAX + K) / ε²), 권장 ε = 0.05~0.1

- [ ] **JL-RBF 포텐셜 근사**:
  ```
  q̃_i = Φ @ q_i   # (k,)
  μ̃_j = Φ @ μ_j   # (k,)
  ‖q_i - μ_j‖²_D ≈ (D/k) * ‖q̃_i - μ̃_j‖²_k
  ```
  - 이론 오차 bound: |V_approx - V_exact| ≤ ε * max_k |w_k|

- [ ] **JL-RBF vs. exact RBF** 수치 비교 실험
  - D = 8, 16, 32, 64, 128 각각에서 ε 실측
  - RBF 값 상대 오차 < 5% 달성하는 최소 k 탐색

- [ ] **`jit`-compatible 구현**: Φ는 static constant로 처리
  - `partial(jit, static_argnums=...)` 또는 module-level constant

- [ ] BPTT 기울기 전파: JL 투영은 선형 → grad 자동 통과

**이론적 핵심**: JL 보조정리가 RBF 지수 안의 거리를 보존하므로 gradient 왜곡도 O(ε) 이내

---

### Phase 3: TurboQuant/QJL 입자 앙상블 압축
**파일**: `block_iii/phase3_turboquant_compress.py`

- [ ] **PolarQuant 입자 위치 압축기 구현**:
  ```python
  def polar_quant(q, B=3, key=None):
      # 1. 랜덤 회전 전처리
      R = random_rotation(D, key)
      q_rot = R @ q          # (D,) 등방화
      # 2. norm 추출
      r = jnp.linalg.norm(q_rot)
      q_unit = q_rot / r
      # 3. 극좌표 각도 양자화 (B-1 bits)
      thetas = jnp.arctan2(q_unit[1::2], q_unit[0::2])  # (D//2,)
      theta_q = uniform_quantize(thetas, bits=B-1)
      return r, theta_q, R
  ```

- [ ] **QJL 잔차 보정기 구현**:
  ```python
  def qjl_residual(e, Phi):
      # e: 잔차 벡터 (D,)
      # Phi: JL 행렬 (k, D)
      sketch = jnp.sign(Phi @ e)          # (k,) 1-bit
      return sketch   # inner product 추정에 사용
  
  def qjl_inner_product(q_sketch, mu_full, Phi):
      # 비편향 내적 추정량
      return (jnp.pi / (2 * k)) * q_sketch @ (Phi @ mu_full)
  ```

- [ ] **압축-해제 루프** (jax.lax.scan 호환):
  - 압축: 학습 완료 후 trajectory 저장 시
  - 해제: 분류/검증 시 unpack → inner product 추정
  - 학습 중 압축 (quantization-aware training): optional Phase 3b

- [ ] **비트폭별 정밀도 실험**: B = 2, 3, 4 bits
  - 분류 정확도 vs. 메모리 사용량 Pareto 곡선
  - 목표: B=3에서 6× 압축, 분류 정확도 유지

- [ ] **이론 검증**: 
  - E[⟨q̂_TQ, μ⟩] = ⟨q, μ⟩ (비편향성 수치 확인)
  - Var[추정량] ≤ C/k 수치 측정

---

### Phase 4: N-클래스 일반화 (O/X → {0,…,N-1})
**파일**: `block_iii/phase4_multiclass.py`

- [ ] **N개의 클래스 attractor** 정의:
  - q*_c ∈ ℝᴰ, c = 0..N-1
  - 균일 분포: 단위 D-구(hypersphere)의 정점 (정다포체 정점) 권장
  - N=2: ±(8,8,0) (Block II 호환)
  - N=4: 4D 단체(simplex) 정점
  - N=8: 8D 하이퍼큐브 정점

- [ ] **손실함수 일반화**:
  ```
  L(θ) = Σ_c ‖CoM_c(T) - q*_c‖² + λ_p * Σ_c mean_i ‖p_i^c(T)‖²
  ```

- [ ] **frozen attractor 설계**:
  - k=0..N-1: N개의 attractor (frozen)
  - k=N..K-1: K-N개의 learnable RBF (stepping stones + free)

- [ ] **분류 규칙**: argmin_c ‖CoM(T) - q*_c‖

- [ ] **JL 보조정리 활용**: N개의 attractor는 m = N_MAX + K에 포함 → JL-reduced RBF 자동 확장

---

### Phase 5: 검증 및 논문 준비
**파일**: `block_iii/phase5_verification.py`

- [ ] **에너지 단조감소 검증**: Block I 기준 그대로 적용 (D-dim 확장)
- [ ] **위상부피 수축 검증**: 6D → (2D+2)D Cov matrix, JL 근사 포함
- [ ] **TurboQuant 압축률 vs. 분류 오차** 실험 표
- [ ] **γ 해석가능성 보고서**: 학습된 γ_c 패턴 분석
- [ ] **verification figure** (8-panel 확장):
  - [추가] JL 차원 k vs. RBF 오차 곡선
  - [추가] TurboQuant 비트폭 vs. 분류 정확도
  - [추가] 학습된 γ 히트맵 (N-클래스)
- [ ] **LaTeX 논문 초안**: Block I~III 통합 서술

---

## 구현 우선순위

```
Phase 0 (ND 일반화)   ← 최우선: Block II 코드 검증 후 즉시 착수
    ↓
Phase 1 (γ 학습)      ← 해석가능성 핵심, 논문 차별점
    ↓
Phase 2 (JL RBF)      ← 수렴 속도 향상 + 고차원 대응
    ↓
Phase 3 (TurboQuant)  ← 메모리 효율화, 실용화 단계
    ↓
Phase 4 (N-class)     ← 일반성 증명
    ↓
Phase 5 (검증+논문)   ← 최종 산출물
```

---

## 참고 문헌

- Zandieh, A., Daliri, M., Han, I. "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead." *AAAI 2025*. [[arXiv:2406.03482]](https://arxiv.org/abs/2406.03482)
- Zandieh, A., Mirrokni, V. "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate." *ICLR 2026*. [[arXiv:2504.19874]](https://arxiv.org/html/2504.19874v1)
- Johnson, W., Lindenstrauss, J. "Extensions of Lipschitz mappings into a Hilbert space." *Contemporary Mathematics*, 1984.
- Block I/II 코드: `block_i_prototype/block_i.py`, `block_ii/block_ii.py`

---

## 메모

- **Block III 시작 조건**: Block II 학습 완료 후 `block2_trained_params.npy` 존재 확인
- **하드웨어**: RTX 4060 (8GB VRAM) → D=64까지 Phase 0~2 가능, D=128은 Phase 3 이후
- **γ 해석가능성 가치**: 복잡도가 낮아서가 아니라 물리 구조(접촉 해밀토니안)에 의해 보장됨 — 핵심 차별점 유지
- **TurboQuant JAX 구현 참고**: [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) (PyTorch → JAX 이식 필요)
