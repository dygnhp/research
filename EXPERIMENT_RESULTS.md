# CHM 실험 결과 — 2x2 ablation (CPU) + GPU 테스트

환경: i7-13700F (8P+8E, 24 logical) / 16GB RAM / RTX 4060 8GB.
Windows-CPU = jax 0.4.30, WSL2(Ubuntu)-CPU/GPU = jax 0.10.1 (+cuda12).
모든 실험 seed=42. 축소 예산: OX_8 1000 / ABC_16 1500 / abcd_32 1000 epoch.

---

## 1. CPU 2x2 ablation — 지표 (seed 고정, 결정적)

데이터셋당 {default, improved 끌개} x {sigma 동결, 학습} = 4셀.

| dataset | cell | eps_q_max | R2_min | canonical | variant |
|---|---|---|---|---|---|
| OX_8 | default/frozen | 11.77 | -273 | 1/2 | 0.71 |
| OX_8 | default/learned | 11.76 | -277 | 1/2 | 0.73 |
| OX_8 | **improved/frozen** | **5.34** | -326 | **2/2** | **0.82** |
| OX_8 | **improved/learned** | **5.20** | -326 | **2/2** | **0.82** |
| ABC_16 | default/frozen | 20.38 | — | 2/3 | 0.69 |
| ABC_16 | default/learned | 20.27 | — | 2/3 | 0.69 |
| ABC_16 | **improved/frozen** | **9.76** | — | 2/3 | **0.77** |
| ABC_16 | **improved/learned** | **9.77** | — | 2/3 | **0.77** |
| abcd_32 | default/frozen | 36.03 | — | 2/4 | 0.41 |
| abcd_32 | default/learned | 36.03 | — | 2/4 | 0.41 |
| abcd_32 | **improved/frozen** | **18.25** | — | 2/4 | 0.41 |
| abcd_32 | improved/learned | 18.25 | — | 1/4 | 0.40 |

**결론**
- **improved 끌개 레이아웃이 eps_q를 모든 데이터셋에서 ~절반으로**: OX 11.8->5.2, ABC 20.3->9.8, abcd 36->18.3. mu 재배치(데이터중심 대칭)의 효과.
- **sigma 동결 vs 학습은 eps_q에 거의 영향 없음** (예: OX 5.34 vs 5.20, abcd 18.25 동일). mu 재배치가 주효, sigma 학습은 미세 보정.
- **정확도**: OX variant 0.71->0.82 / canonical 1/2->2/2; ABC variant 0.69->0.77. abcd는 eps_q 절반에도 variant 0.40 정체 — a/d 글자가 거의 동일(데이터 본질적 난점) + 축소 예산.

---

## 2. 실험 소요 시간 (CPU, ms/epoch)

오염 셀(초반, GPU 설치와 겹침)은 깨끗한 환경에서 재측정. OX는 재측정값, ABC/abcd는 본실행값(설치 종료 후 클린 구간).

| dataset (n_max) | default/frozen | default/learned | improved/frozen | improved/learned |
|---|---|---|---|---|
| OX_8 (64) | 62.9 | 61.1 | 68.1 | 68.2 |
| ABC_16 (128) | ~232 | 178.9 | 172.6 | 201.6 |
| abcd_32 (400) | 791.8 | 810.2 | 754.8 | 746.6 |

- per-epoch는 **n_max(입자수)에 비례** (OX~63, ABC~180, abcd~770 ms/ep).
- sigma 학습 오버헤드는 무시 가능(multi_transform 저렴).
- 절대 wall-time은 16GB 공유 데스크톱에서 변산 큼; 지표는 정확.

---

## 3. GPU 테스트 (WSL2, RTX 4060)

WSL2에 jax[cuda12] 설치 -> `jax.devices() = [CudaDevice(id=0)]`. FINAL이 jax 0.10.1+GPU에서 정상 동작.
동일 WSL 환경에서 device만 바꿔 정상구간 ms/epoch 측정 (improved/learned, 성장 off, 워밍업+2점법).

| dataset (n_max) | WSL-CPU | WSL-GPU | GPU 대 CPU | (참고) Win-CPU |
|---|---|---|---|---|
| OX_8 (64) | 20.5 | 62.9 | **0.33x (GPU 3.1x 느림)** | ~63 |
| ABC_16 (128) | 119.1 | 82.5 | **1.44x 빠름** | ~180 |
| abcd_32 (400) | 724.3 | 129.7 | **5.58x 빠름** | ~770 |

**결론**
- **GPU 이득은 문제 크기에 비례.** 작은 OX(n_max 64)는 RK4 200스텝 `lax.scan`의 작은 per-step 연산이 GPU 커널 런치 오버헤드에 압도돼 **GPU가 3x 느림**. 큰 abcd(400)는 **GPU 5.6x 빠름**. 교차점은 대략 ABC_16.
- **워밍업 필수**: 워밍업 없이는 GPU OX가 197.8ms로 측정(첫 CUDA 초기화+컴파일 오염) -> 워밍업 후 62.9ms.
- 부가: **WSL-CPU(jax 0.10.1)가 Win-CPU(0.4.30)보다 빠름** (OX 20.5 vs 63) — 최신 XLA + Linux 백엔드.

### 실용 함의 (전체 예산 투영)
- abcd_32 전체 8000 epoch: CPU ~770ms x 8000 = **~100분** vs GPU ~130ms x 8000 = **~17분** (약 6x 단축).
- OX/ABC 같은 작은 실험: CPU(특히 WSL-CPU)로 충분/더 빠름. **abcd 및 고-D/고-K/전체예산 실험엔 GPU 권장.**
- 16GB RAM 제약: WSL-GPU와 Windows-CPU를 **동시 실행하지 말 것**(HDD 페이지파일 스왑 발생).

---

## 4. 제1차 본실험 (간소화, improved + sigma 학습, 전체 예산)

seed 42/1 (각 2회), OX/ABC=CPU, abcd=GPU. 원본 출력은 research/main_exp_1/ (gitignore; 로컬 보존).

| dataset | canonical | variant (seed 평균) | 클래스별 (평균) | 최종 D/K | 시간 |
|---|---|---|---|---|---|
| OX_8 | **2/2 (100%)** | **86.0%** | O 78%, X 94% | 6 / 54 | CPU ~271s |
| ABC_16 | **3/3 (100%)** | **93.4%** | A 100%, B 100%, C 80% | 5 / 37 | CPU ~580s |
| abcd_32 | **4/4 (100%)** | **71.0%** | a 55%, b 95%, c 58%, d 76% | 6 / 56-60 | GPU ~1390s |

- **canonical 3종 모두 100%** — CHM 분류 메커니즘 작동 입증.
- **전체 예산 + D-growth가 결정적**: 축소예산 ablation 대비 abcd 0.40->0.71, ABC 0.77->0.93, OX 0.82->0.86.
- seed 간 거의 동일(최종 D/K 일치) -> 높은 재현성.
- **열린 이슈**: 위상부피 R^2 게이트는 여전히 미달(별개 깊은 진단 이슈). abcd a/c 낮음.

## 5. abcd 'a' 글자 실험 (탐색, 이후 원복)

기존 'a'/'d'가 둘 다 "원+세로획"(O|)이라 a/d 혼동. 'a'를 2층(Helvetica) 글자로 바꿔 재실험:
- d 76->100%, b 95->100%, c 58->64%, **전체 variant 71->75.5%** (d가 a를 흉내내던 혼동 해소).
- 그러나 **새 'a' 자체는 55->38%로 악화** (2층 a가 CoM 라우팅엔 더 어려움).
- 결론: a/d 혼동의 진짜 레버는 데이터/특징/지형이지 끌개 배치가 아님. **'a'는 원본으로 원복**(데이터셋 표준 유지). 끌개 위치(mu) 학습은 자명해 위험 + a/d 동일입력 미해결로 보류.

## 6. 제2차 본실험 (3-seed 재현성, improved + sigma 학습, 전체 예산)

seed 42/1/2. main_exp_1과 동일 설정. 원본 출력은 research/main_exp_2/ (gitignore).

| dataset | canonical | variant 평균±std | D/K | device |
|---|---|---|---|---|
| OX_8 | 2/2 (전부 100%) | **86.3 ± 3.1%** (83/89/87) | 6/54 | CPU (+phase gallery) |
| ABC_16 | 3/3 (전부 100%) | **93.3 ± 0.7%** (94/92.7/93.3) | 5/37 | CPU |
| abcd_32 | 4/4 (전부 100%) | **70.8 ± 1.5%** (69.5/72.5/70.5) | 6/56-60 | GPU |

- canonical 3종 모두 100% (3 seed 전부), 변산 작음 -> **높은 재현성** (1차 결과와 일치).
- OX는 입자별 phase-space 갤러리 동반 생성 (12입자 x 2클래스 x 3 seed).

## 7. ANN 대조군 (TensorFlow Sequential, control/ann_baseline.py)

동일 데이터, Flatten -> Dense(32, relu) -> Dense(C, softmax). **5 seed(42,1,2,3,4)** 평균±std.
FLOPs 비교는 추후. (원본 research/ann_control.json, gitignore)

| dataset | ANN params | ANN canonical | ANN held-out (평균±std) | CHM params | CHM variant |
|---|---|---|---|---|---|
| OX_8 | 2,146 | 100% | **99.4 ± 0.5%** | ~434 | 86.3% |
| ABC_16 | 8,323 | 100% | **100 ± 0%** | ~262 | 93.3% |
| abcd_32 | 32,932 | 100% | **100 ± 0%** | ~470 | 70.8% |

- **표준 MLP는 3종 모두 100%** (미학습 held-out 포함). 합성 글자는 black-box ANN엔 사소.
- **ANN이 abcd a/d도 100% 분리** -> a/d 구분 정보는 **픽셀에 분명히 존재**; CHM의 어려움은 데이터가 아니라 **"거의 동일한 입자구름의 CoM 라우팅"이라는 표현방식의 한계**.
- ANN은 파라미터 많음(입력->은닉 행렬). CHM은 lifted 입자공간에서 동작해 **파라미터 효율적**.
- **ANN=black-box vs CHM=white-box** -> CHM은 정확도/연산을 **해석가능성과 맞바꿈**. (FLOPs 측정이 그 trade-off를 정량화할 예정)
