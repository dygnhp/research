# 실험 가이드라인 — kanzen

> 데이터셋 선택부터 하이퍼파라미터, 에폭 설정, 정착 조건 해석, 그리고
> 실패 시나리오의 진단까지를 모은 단일 참고 문서.

CHM의 메커니즘 자체는 [`ALGORITHM.md`](ALGORITHM.md)를 참고하세요.
이 문서는 "어떤 데이터로 어떻게 학습할지"의 가이드입니다.

---

## 1. 데이터셋 카드

| 데이터셋     | 이미지     | 클래스        | n_max | 끌개 배치 (Phase A) | 캐노니컬 픽셀 수            |
|--------------|-----------|---------------|-------|--------------------|-----------------------------|
| `OX_8`       | 8 × 8     | O, X          | 64    | (10,10) & (-3,-3) (데이터 중심 대칭) | O=16, X=16    |
| `ABC_16`     | 16 × 16   | A, B, C       | 128   | 반지름 14의 정삼각형 (Phase 0: 20) | A=54, B=45, C=38 |
| `abcd_32`    | 32 × 32   | a, b, c, d    | 400   | 반지름 26의 정사각형 (Phase 0: 36) | a=113, b=126, c=95, d=126 |

데이터는 모두 합성 파라메트릭 이미지로, 각 클래스는

- 결정론적 **canonical** (디폴트 인자) 한 장,
- 무작위 인자 변형(`generate_random_*`) `n_train_per_class - 1`장,

을 가집니다. 첫 장은 항상 캐노니컬이라 검증/시각화에 사용됩니다.

각 클래스의 끌개 좌표 `q*_c`는 (x, y) 평면에서 **데이터 도메인 바깥**에
배치되어 있어, 학습 전에는 끌개가 어떤 입자에도 거의 힘을 가하지 못합니다.
이 문제는 클래스마다 한 개씩 있는 **stepping stone**(데이터 도메인 안쪽에
배치된 학습 가능 끌개)이 해결합니다.

---

## 2. 데이터셋 선택 방법

세 가지 데이터셋 모두 **동일한 학습 코드**로 처리됩니다. 선택은 Config 한 곳에서:

```python
from LEGACY.kanzen import Config, train

cfg = Config.with_dataset("ABC_16")  # 디폴트 하이퍼파라미터 자동 적용
cfg = Config.with_dataset("ABC_16", n_epochs=2000)  # 일부만 오버라이드
run = train(cfg)
```

CLI에서는:

```bash
python -m kanzen.main train --dataset ABC_16
python -m kanzen.main train --dataset abcd_32 --epochs 4000
python -m kanzen.main demo  --dataset OX_8
```

`Config.with_dataset(name)`을 호출하면 `_DATASET_DEFAULTS[name]`에 정의된
에폭 수·학습률·plateau 윈도우·정착 임계값이 **자동 적용**됩니다.
필요한 인자만 명시적으로 덮어쓰면 됩니다.

---

## 3. 권장 하이퍼파라미터 (Phase A 갱신본, 디폴트로 적용됨)

| 항목                    | OX_8   | ABC_16 | abcd_32 |
|-------------------------|-------:|-------:|--------:|
| `K_init`                | 16     | 21     | 28      |
| 그 중 frozen            | 2      | 3      | 4       |
| 그 중 stepping stones   | 2      | 3      | 4       |
| 그 중 free RBF          | 12     | 15     | 20      |
| `n_epochs`              | 3 000  | 5 000  | 8 000   |
| `peak_lr`               | 5e-3   | 3e-3   | 2e-3    |
| `warmup_steps`          | 100    | 200    | 300     |
| `plateau_window`        | 100    | 150    | 200     |
| `min_epochs_before_grow`| 200    | 400    | 600     |
| `eps_q_thresh` (정착)   | 2.0    | 3.0    | 5.0     |
| `eps_p_thresh` (정착)   | 0.5    | 0.6    | 0.8     |
| `phase_R2_thresh`       | 0.90   | 0.85   | 0.80    |
| **`frozen_sigma` (Phase A)** | **5.0** | **8.0** | **15.0** |
| **`K_grows_before_D`**  | 3      | 3      | **2** (Phase B) |
| **`cooldown_after_grow`** | 100  | 100    | **200** (Phase B) |

### Phase A 핵심 변화점

초기 구현(Phase 0)에서 frozen attractor의 σ가 2.0 (= 2·image_scale) 였는데,
이는 σ가 데이터 도메인의 거의 절반 정도에서만 의미 있는 force를 가하는
좁은 값이었다. 결과적으로 **frozen attractor가 inert** 가 되어 학습된
지형이 데이터 도메인 *내부*에 분류용 basin을 형성하는 \"soft basin
routing\" 현상이 일어났다.

Phase A 에서는 다음 두 가지를 수정:

1. **σ_frozen 데이터셋별 확장**: $d/2.14$ 룰 (여기서 d는 데이터 중심과
   끌개 사이 거리)을 적용하여 σ_frozen 이 데이터 도메인에 *닿게* 함.
2. **OX_8 끌개 좌표 대칭화**: 기존 `(±8, ±8)`은 데이터 중심 (3.5, 3.5)
   기준 비대칭이었다(O는 거리 6.36, X는 16.26). 이를 `(10, 10)` 과
   `(-3, -3)` 으로 변경하여 둘 다 데이터 중심에서 거리 9.19 의 대칭
   배치로 만들었다.

### 왜 이렇게 차이가 나는가

**K_init**: 클래스가 늘어나면 frozen 끌개와 stepping stone이 클래스마다
한 개씩 늘어납니다. free RBF 수는 이미지 면적이 커지므로 조금씩 증가.

**n_epochs**: 큰 이미지일수록 (a) 입자 수가 많아 한 epoch의 forward 비용이
커지고, (b) 끌개까지 거리가 멀어 수렴이 느립니다. 32×32는 8×8 대비 입자
수 약 8배, 거리 약 3배라 ~3× 에폭이 필요.

**peak_lr**: 이미지가 클수록 손실값과 그래디언트의 크기가 커집니다. LR을
낮추지 않으면 step 크기가 과도해져 발산합니다.

**eps_q_thresh / phase_R2_thresh**: 이미지가 커지면 모든 거리 스케일이
함께 커지므로 정착 임계값도 비례하여 완화합니다.

**`frozen_sigma`** (Phase A 신규): 끌개의 Gaussian이 데이터 도메인까지 충
분히 닿도록 데이터셋 크기에 비례하여 키웁니다. 핵심 룰: $\sigma \ge d /
2.14$ — 여기서 $d$는 데이터 중심에서 끌개까지의 거리. 이 룰이 깨지면
끌개 force가 사실상 0이 되어 학습된 지형이 데이터 도메인 *내부*에 \"soft
basin\" 만 만들고, 결과적으로 진짜 끌개에 도달하지 못합니다(Phase 0에서
관찰됨).

---

## 4. 실험 시나리오별 권장 설정

### 시나리오 A — 빠른 검증 (smoke test)

새 환경에서 모듈이 동작하는지 확인할 때.

```python
cfg = Config.with_dataset("OX_8", n_epochs=200, log_every=50,
                          min_epochs_before_grow=100)
```

기대치: 끝날 때까지 손실이 일관되게 감소. 분류가 맞지 않아도 상관 없음.

### 시나리오 B — 한 데이터셋 전체 학습

논문/리포트용 결과를 만들 때. 디폴트가 곧 권장값입니다.

```bash
python -m kanzen.main train --dataset ABC_16
```

기대치 (CPU 기준):
- OX_8: ~15분, 분류 정확도 100% (캐노니컬), eps_q < 3.
- ABC_16: ~1.5시간, 정확도 ≥ 85%, eps_q < 5.
- abcd_32: ~6시간, 정확도 ≥ 70%, eps_q < 8.

GPU에서는 약 5–10× 빠릅니다.

### 시나리오 C — 자율 성장 데모

K_init을 의도적으로 낮춰 학습 중 K가 자라는 모습을 보이고 싶을 때.

```python
cfg = Config.with_dataset(
    "ABC_16",
    K_init=9,                # 3 frozen + 3 stones + 3 free  (보통의 절반 이하)
    K_grow=4, K_max=32,
    min_epochs_before_grow=300, plateau_window=100,
)
```

기대치: 학습 중 K_learn이 6 → 10 → 14 식으로 단계적으로 증가. 손실 곡선에
세로 빨간 점선이 나타남.

### 시나리오 D — 차원 확장 데모

K가 K_max에 도달한 뒤에도 plateau가 풀리지 않으면 D가 증가합니다.

```python
cfg = Config.with_dataset(
    "abcd_32",
    K_init=12, K_max=20,    # 빠르게 K_max 도달
    D_init=3, D_max=5,
    K_grows_before_D=2,
    min_epochs_before_grow=400,
)
```

기대치: 학습 중 D가 3 → 4 → 5로 증가. 차원 확장 시점에서 sigma가
`sqrt(D_new/D_old)` 배 자동 재스케일링됨 (콘솔 출력 + summary 파란 점선).

### 시나리오 E — 잡음/시프트 강건성

학습된 모델의 robustness 평가.

```python
from LEGACY.kanzen import noise_sweep, shift_sweep

acc_noise = noise_sweep(canonical, "A", state, cfg, levels=range(0, 30, 3))
acc_shift = shift_sweep(canonical, "A", state, cfg, max_shift=3)
```

`evaluate.py`의 모든 sweep은 forward only라 학습된 파라미터를 보존합니다.

---

## 5. 에폭 설정의 직관

### 5.1 손실의 단위와 스케일

손실의 위치 항은 `||CoM - q*||²`이고 끌개까지의 거리는 데이터셋마다
다릅니다.

| 데이터셋  | 데이터 → 끌개 평균 거리 | 초기 손실 (대략) |
|-----------|------------------------:|------------------:|
| OX_8      | ~14                     | ~380              |
| ABC_16    | ~22                     | ~1200             |
| abcd_32   | ~36                     | ~5000             |

**손실 감소가 정상인지의 기준**:
- 초반 200 epoch에서 손실이 10–30% 감소해야 함.
- 그렇지 않으면 LR이 너무 낮거나 stepping stone 위치가 잘못된 것.

### 5.2 plateau 감지 윈도우

`plateau_window`는 *움직이는 평균*을 계산할 폭입니다. 너무 작으면 잡음에
의한 일시적 정체를 plateau로 오인하여 grow_K가 너무 자주 호출됩니다.
너무 크면 진짜 plateau를 늦게 감지합니다.

경험칙: `plateau_window ≈ n_epochs / 30`. 디폴트가 이 비율로 맞춰져 있습니다.

### 5.3 정착 게이트

정착 조건은 **3중**입니다:
1. 모든 클래스의 `eps_q < eps_q_thresh`
2. 모든 클래스의 `eps_p < eps_p_thresh`
3. `R^2_phase ≥ phase_R2_thresh`

조건 (1) 만으로는 부족합니다 — 입자가 끌개를 빠른 속도로 *통과*하는
경로를 학습할 수 있고, 다음 에폭에서 이탈하기 때문입니다. (2)가 이를
막습니다. (3)은 위상공간 부피 수축이 이론과 맞는지의 sanity check 입니다.

학습 도중 `R^2`가 음수에 머무는 경우, 이는 위상 부피가 이론적 예측보다
훨씬 빠르거나 느리게 변하는 상태(보통은 시스템이 진동 중)임을 의미합니다.
끝까지 음수라면 보통 `gamma`가 너무 작거나 LR이 너무 커서 발산하는
경우입니다.

---

## 6. 실패 모드와 진단

| 증상                                  | 진단                                            | 처방                                                          |
|---------------------------------------|------------------------------------------------|--------------------------------------------------------------|
| 손실이 감소하지 않음                  | LR이 너무 작거나 stepping stone이 데이터 도메인에 닿지 않음 | `peak_lr` 2× 또는 K_init 증가 (free RBF 추가) |
| 한 클래스만 끌개에 도달, 다른 클래스는 데이터 도메인에 머무름 | stepping stone의 부호/위치 비대칭             | `Config.with_dataset(..., dataset_seed=다른 값)`             |
| 손실은 줄지만 eps_q 정체              | 정착 임계값 이하로 가지 못함                   | `min_epochs_before_grow` 줄이고 `K_grow` 증가                 |
| eps_q는 작은데 eps_p가 큼             | 입자가 끌개를 통과 중                          | `lambda_p` 2–5×로 증가                                       |
| R^2가 계속 음수                        | gamma 미스매치 또는 발산                       | `gamma` 1.0→1.5, peak_lr 절반으로 줄임                       |
| grow_D가 절대 발화하지 않음           | K_max에 도달하지 못함                          | K_max 낮춤 또는 `K_grows_before_D` 줄임                       |
| JIT 재컴파일이 매 에폭 발생           | n_max 가변 또는 K가 매번 변함                  | 보고된 적 없음 — 발생하면 issue                              |

---

## 7. 산출물 구조

```
kanzen_runs/run_<dataset>_<timestamp>/
  params.npz          # 최종 w, mu, sigma_raw, D, K_learn
  config.json         # 사용된 Config 전체
  history.json        # 손실, per-class diagnostics, 이벤트
  growth_log.json     # grow_K / grow_D 이벤트 (epoch, kind, after-K / after-D)
  summary.png         # 클래스별 terrain + trajectory + 손실 + diagnostics
```

평가는 같은 데이터셋의 가장 최근 run을 자동으로 로드합니다:

```bash
python -m kanzen.main evaluate --dataset ABC_16
```

---

## 8. 빠른 참조 — 한 줄로 실험하기

| 목적                                              | 명령                                                          |
|--------------------------------------------------|---------------------------------------------------------------|
| OX_8 풀 학습                                     | `python -m kanzen.main train`                                |
| ABC_16 풀 학습                                   | `python -m kanzen.main train --dataset ABC_16`               |
| abcd_32 풀 학습 (4 시간 이상)                    | `python -m kanzen.main train --dataset abcd_32`              |
| 빠른 데모 (100 epoch)                            | `python -m kanzen.main demo --dataset ABC_16`                |
| 시드 변경 재현 실험                              | `... train --dataset ABC_16 --seed 7`                         |
| 짧은 에폭으로 끊어서 테스트                      | `... train --dataset ABC_16 --epochs 500`                     |
| 학습된 모델 평가                                 | `python -m kanzen.main evaluate --dataset ABC_16`             |

---

## 9. 새 데이터셋 추가하는 법

`kanzen/data.py`의 `DATASETS` 딕셔너리에 `DatasetSpec` 항목 하나만
추가하면 됩니다. 필요한 것:

1. `image_size`, `class_labels`
2. 클래스마다 캐노니컬 이미지(numpy 2D)와 random variant 생성기
3. `attractor_positions`: 클래스 → (x, y) — `_polygon_attractors(...)` 유틸 사용 권장
4. `attractor_z`: 클래스 → 평균 z (자동 계산: `_mean_z(canonical)`)
5. `n_max`: 캐노니컬 픽셀 수의 2~3배

그 다음 `kanzen/config.py`의 `_DATASET_DEFAULTS`에 권장 하이퍼파라미터를
추가하면, `Config.with_dataset("새이름")`이 즉시 동작합니다.

다른 어떤 코드도 손댈 필요가 없습니다 — params, loss, train, evaluate,
viz 모두 클래스 개수에 무관하게 동작합니다.

---

## 10. 알아두면 좋은 점

- **클래스별 입자 수가 달라도 됩니다.** 패딩(`mask`)으로 처리되므로 A가
  54개 픽셀, B가 45개여도 같은 (128, …) 텐서로 들어갑니다. CoM과
  momentum penalty는 모두 mask-aware입니다.

- **z 채널은 자동 추정됩니다.** 각 클래스의 attractor `z` 좌표는 그 클래스
  캐노니컬의 평균 axis/diagonal connectivity로 계산됩니다. O는 0.88, X는
  0.12, A는 0.79, c는 0.90 등 — 이 값들이 손실의 D=3 차원 target이 됩니다.

- **dataset_seed는 데이터셋 변형과 옵티마이저 양쪽에 적용됩니다.** 단일
  시드만 바꾸면 학습 데이터의 변형 인자와 free RBF 초기 배치가 함께
  바뀌어 새로운 trial을 만듭니다.

- **CPU에서도 학습 가능합니다.** OX_8과 ABC_16은 노트북 CPU로도 합리적인
  시간 안에 끝납니다. abcd_32는 GPU 사용을 권장합니다.
