# Image Inpainting via Iteratively Decoupled Probabilistic Modeling

### 1. 핵심 주장 및 기여도 요약

본 논문("Image Inpainting via Iteratively Decoupled Probabilistic Modeling", Li et al., 2022)의 중심 주장은 **대규모 결손 영역을 가진 고해상도 이미지 보수(inpainting)에서 GAN의 최적화 효율성과 확률적 모델의 해석 가능성을 결합**하는 것이다. 저자들은 기존 접근법들의 근본적 한계를 지적한다:[1]

- **GAN 기반 방법**: 빠른 생성 속도지만 한 번의 생성(one-shot)으로 인한 불안정한 학습과 대규모 결손 영역에서 저품질 결과
- **자동회귀 모델 및 확산 모델**: 높은 생성 품질이지만 수천 개의 반복 단계 필요로 계산 비용 극대

**Pixel Spread Model(PSM)**이라는 새로운 프레임워크를 제안하며, 그 핵심 기여는:[1]

1. **분리된 확률적 모델링(Decoupled Probabilistic Modeling)**: 평균항(mean)은 적대적 학습으로, 분산항(variance)은 가우스 정규화로 분리 최적화
2. **불확실성 기반 픽셀 선택**: 생성된 픽셀의 신뢰도를 평가하여 다음 반복에서만 높은 신뢰도의 픽셀을 사용
3. **불확실성 가이드 주의 메커니즘(Attention)**: 기존 어텐션의 문제를 해결하고 정보성 높은 픽셀을 효율적으로 활용

성능 향상: Places2(512×512)에서 LDM 대비 **FID 1.1 개선**, **매개변수 20분의 1, 속도 10배 향상**[1]

***

### 2. 문제 정의 및 해결 방법

#### 2.1 해결하고자 하는 문제

이미지 보수 작업은 다음과 같이 형식화된다:[1]

$$\text{주어진: } x_0 \in \mathbb{R}^{H \times W \times 3}, \quad m_0 \in \{0,1\}^{H \times W}$$

여기서 $x_0$는 원본 이미지, $m_0$는 이진 마스크(1은 결손 영역, 0은 알려진 영역)이다. 목표는:

$$\hat{x} = \arg\max_x p(x_{\text{missing}} | x_{\text{known}})$$

즉, 알려진 영역을 조건으로 하는 조건부 확률분포에서 결손 영역의 최적 보수를 학습하는 것이다.[1]

**핵심 문제점**:
- 대규모 결손 영역($\geq 50\%$)에서 기존 GAN의 한 번의 생성은 모호한 텍스처와 불쾌한 아티팩트 생성
- 확산 모델은 우수한 품질이나 극도로 높은 계산 비용(250초 vs PSM의 250ms)
- 자동회귀 모델은 픽셀 단위 순차 처리로 고해상도 이미지에 부적절

#### 2.2 제안하는 방법: 반복적 분리 확률적 모델링

**2.2.1 분리된 확률적 모델링(DPM)**

PSM의 핵심은 $t$번째 반복에서 다음을 동시에 예측하는 것이다:[1]

$$\mu_t = f_\mu(x_{t-1}, m_{t-1}, u_{t-1}; \theta), \quad \sigma^2_t = f_\sigma(x_{t-1}, m_{t-1}, u_{t-1}; \theta)$$

여기서 $\mu_t$는 보수된 이미지의 평균값, $\sigma^2_t$는 각 픽셀의 분산(불확실성)을 의미한다.

**핵심 설계**: 이 두 항을 분리(decouple)하여 최적화한다:

1. **평균항 최적화** - 암시적 적대적 손실(implicit adversarial loss):[1]

$$\mathcal{L}_{\text{adv}} = \mathbb{E}_x [\log D(x)] + \mathbb{E}_{\hat{x}} [\log(1 - D(\hat{x}))]$$

$$\mathcal{L}_{\text{pcp}} = \sum_{i} \|\phi_i(x) - \phi_i(\hat{x})\|^2_2$$

여기서 $\phi_i$는 사전 훈련된 ResNet50의 $i$번째 층 특징을 나타낸다.[1]

2. **분산항 정규화** - 음의 대수 우도(Negative Log-Likelihood)를 통한 명시적 정규화:[1]

$$\mathcal{L}_{\text{nll}} = \sum_{i=1}^{D} \int \log \left( p_\theta(y_i | y_{\text{sg}}, \sigma^2_{\text{sg}}, x) \right) dy_i$$

$$= \sum_{i=1}^{D} \int \log \left( \mathcal{N}(y | \mu_\theta, \sigma^2_\theta) \right) dy_i$$

여기서 **중요한 설계**: $\text{sg}$ (stop-gradient 연산)를 사용하여 분산 항만 최적화되도록 제약한다.[1]

이 분리 전략의 이점:
- GAN의 효율적 최적화로 반복 횟수 감소
- 가우스 가정의 명시적 모델링으로 신뢰성 있는 불확실성 추정
- 평균항이 더 정확한 암시적 모델링에 집중 가능

**2.2.2 반복적 픽셀 확산 과정**

각 반복 $t$에서 세 단계를 거친다:[1]

**단계 1: 예측(Predict)**
모든 픽셀에 대해 $\mu_t$와 $\sigma^2_t$를 동시에 생성하고, 불확실성 맵 $u_t$를 계산한다.

**단계 2: 선택(Pick)**
마스크 스케줄에 따라 불확실성이 낮은(신뢰도 높은) 픽셀을 선택하여 알려진 영역 집합에 추가한다:[1]

$$m_t = \text{UpdateMask}(m_{t-1}, u_t, \text{schedule}(t))$$

$$u_t = \begin{cases} 0 & \text{if } m_0 = 0 \text{ (originally known)} \\ 1 & \text{if } \text{still missing} \end{cases}$$

**단계 3: 샘플링(Sample)**
마스크 스케줄에 따라 연속 확률적 샘플링을 수행한다:[1]

$$x_t = x_0 \odot m_0 + \mu_t \odot (1-m_0) + \alpha \sigma_t \odot z, \quad z \sim \mathcal{N}(0, I)$$

여기서 $\alpha = 0.01$는 조정 가능한 하이퍼파라미터이고, $\odot$는 요소별 곱셈(Hadamard product)이다. 최종 반복에서는 $\alpha \sigma_t z$ 항을 제거하여 결정론적 평균을 출력한다.[1]

**총 손실 함수**:[1]

$$\mathcal{L} = \lambda_1 \mathcal{L}^j_{\text{adv}} + \lambda_2 \mathcal{L}^j_{\text{pcp}} + \lambda_3 \mathcal{L}^j_{\text{nll}}$$

여기서 $\lambda_1 = 1, \lambda_2 = 2, \lambda_3 = 1 \times 10^{-4}$이고 $j$는 반복 횟수이다.[1]

***

### 3. 모델 구조 및 아키텍처

#### 3.1 전체 네트워크 아키텍처

PSM은 **U-Net 기반 인코더-디코더 구조**에 StyleGAN2 디코더를 채택한다:[1]

**인코더**:
- 64개 채널부터 시작하여 각 다운샘플링 후 2배씩 증가
- 최대 512개 채널
- 32×32 및 16×16 해상도에서 어텐션 블록 배치

**디코더**:
- StyleGAN2 기반 구조로 skip connection 활용
- 인코더의 대칭 구조
- 가중치 변조(weight modulation) 기법 적용

**입력** (7채널):
- RGB 이미지: 3채널
- 마스크: 2채널 (초기 마스크 + 업데이트된 마스크)
- 불확실성 맵: 1채널
- 시간 단계: 1채널

**출력** (6채널):
- 평균항: 3채널 (RGB)
- 로그 분산: 3채널 (채널별 분산 추정)[1]

#### 3.2 불확실성 가이드 어텐션 메커니즘

기존 자기-어텐션의 문제점: 결손 영역의 픽셀들이 마스크된 동일 값으로 초기화되어 서로 유사하고, 알려진 영역의 유용한 정보를 효과적으로 활용하지 못한다.[1]

**제안된 불확실성 가이드 어텐션**:[1]

$$\text{Attention}(Q, K, V, u) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + F_u(u)\right)V$$

여기서 $F_u(u)$는 불확실성 맵을 기반으로 한 학습 가능한 편향 함수로, 다음과 같이 구성된다:[1]

$$F_u(u) = \text{MLP}(u) \text{ (4개의 3×3 컨볼루션 레이어)}$$

이를 통해:
- 신뢰도 높은 픽셀(낮은 불확실성)은 더 큰 가중치를 받음
- 결손 영역의 픽셀은 가중치 감소
- 전역적 맥락 정보가 효율적으로 전파됨

***

### 4. 성능 향상 및 정량적 평가

#### 4.1 벤치마크 성능

**Places2 (512×512) 데이터셋**:[1]

| 모델 | 매개변수 (M) | 작은 마스크 |  | | 큰 마스크 |  | |
|------|------------|-----------|---|---|-----------|---|---|
| | | FID | P-IDS | U-IDS | FID | P-IDS | U-IDS |
| PSM (ours) | 74 | **0.72** | **30.95** | **43.91** | **1.68** | **25.33** | **39.30** |
| Stable Diffusion | 860 | 1.32 | 12.69 | 34.78 | 2.11 | 12.01 | 32.57 |
| LDM | 387 | 1.06 | 16.23 | 39.61 | 2.76 | 12.11 | 33.02 |
| MAT | 62 | 1.07 | 27.42 | 41.93 | 2.90 | 19.03 | 35.36 |
| CoModGAN | 109 | 1.10 | 26.95 | 41.88 | 2.92 | 19.64 | 35.78 |
| LaMa | 5127 | 0.99 | 22.79 | 40.58 | 2.97 | 13.09 | 32.29 |

**주요 결과**:[1]
- LDM 대비 **FID 1.1 개선** (큰 마스크 기준: 2.76 → 1.68)
- LaMa 대비 **매개변수 69배 적음** (5127M → 74M)
- 추론 시간: **250ms (PSM) vs 3000ms (LDM)** = **12배 빠름**

**CelebA-HQ (512×512) 데이터셋**:[1]

| 모델 | 작은 마스크 |  | | 큰 마스크 |  | |
|------|-----------|---|---|-----------|---|---|
| | FID | P-IDS | U-IDS | FID | P-IDS | U-IDS |
| PSM (ours) | **2.34** | **22.42** | **33.43** | **4.05** | **16.10** | **28.25** |
| LDM | - | - | - | - | - | - |
| MAT | 2.86 | 21.15 | 32.56 | 4.86 | 13.83 | 25.33 |
| CoModGAN | 3.26 | 19.65 | 31.41 | 5.65 | 11.23 | 22.54 |

#### 4.2 절제 연구(Ablation Study)

반복 횟수의 영향:[1]

| 모델 | 반복 수 | FID |
|------|-------|-----|
| Model B (1 iteration) | 1 | 3.03 |
| Model C (2 iterations) | 2 | 2.63 |
| Model A (3 iterations) | 3 | 2.45 |

**해석**: 반복 횟수 증가에 따라 점진적 개선. 테스트 시에는 더 많은 반복(4-10회)이 FID 개선을 지속적으로 제공하지만 속도-품질 트레이드오프 존재[1]

분리된 확률적 모델링의 중요성:[1]

| 모델 | 설명 | FID |
|------|-----|-----|
| Model A (Full) | 전체 모델 | **2.45** |
| Model D (No DPM) | DPM 제거 | 2.55 |
| Model E (No CS) | 연속 샘플링 제거 | 2.49 |
| Model F (No UGA) | 불확실성 가이드 주의 제거 | 2.73 |

**해석**: 세 가지 설계 요소가 모두 필수적이며, 특히 불확실성 가이드 주의가 0.28 FID 개선 제공[1]

마스크 스케줄 분석:[1]

| 마스크 스케줄 | 반복 수 | FID |
|------------|-------|-----|
| Cubic | 3 | 2.54 |
| Cosine | 3 | 2.48 |
| Linear (균등 분할) | 3 | **2.36** |
| Square Root | 3 | 2.47 |

**해석**: 선형(균등) 스케줄이 최적. 이유는 다양한 초기 마스크 비율을 가진 이미지들에 대해 안정적인 훈련 제공[1]

#### 4.3 일반화 성능 - 고해상도 전이

512×512에서 훈련된 PSM을 1024×1024 해상도로 직접 전이 테스트:[1]

| 모델 | 훈련 데이터 | FID | P-IDS | U-IDS |
|------|----------|-----|-------|-------|
| PSM (ours) | 20M (512) | **3.95** | **14.40** | **32.23** |
| MAT | 50M (512) | 5.83 | 9.51 | 28.02 |
| LaMa | 50M (512) | 6.31 | 4.98 | 23.24 |

**놀라운 결과**: 
- PSM이 **가장 적은 데이터**(20M vs 50M)로 훈련되었음에도 최고 성능
- **FID 1.9 개선** (MAT 대비)
- 이는 모델 구조와 분리된 확률적 모델링의 우수한 일반화 능력을 시사[1]

#### 4.4 다양한 생성(Pluralistic Generation)

PSM의 연속 확률적 샘플링은 다양한 보수 결과를 생성할 수 있다. 정밀도-회상 분석:[1]

| 메트릭 | PSM | LDM | MAT |
|------|-----|-----|-----|
| FID | **1.68** | 2.76 | 2.90 |
| Precision (충실도) | **0.983** | 0.962 | 0.965 |
| Recall (다양성) | **0.971** | 0.975 | 0.939 |

**해석**: PSM은 LDM과 유사한 회상(다양성) 수준을 유지하면서 **정밀도(충실도)에서 우수**[1]

***

### 5. 모델의 일반화 성능 향상 메커니즘

#### 5.1 일반화 성능이 우수한 이유

**1. 구조적 편향(Inductive Bias)**

PSM은 다음과 같은 구조적 편향을 가진다:[1]

$$\text{점진적 신뢰도 기반 픽셀 확산} \approx \text{인간의 보수 작업 방식}$$

인간은 높은 신뢰도를 가진 부분부터 시작하여 점진적으로 결손 영역을 채운다. 이는 불필요한 추상적 특징 학습을 줄이고, **전이 가능한 패턴 학습**을 유도한다.[1]

**2. 불확실성 추정의 일반화**

가우스 분산 추정이 도메인에 독립적이라는 가정에 기반한다. 이를 통해:[1]

$$\sigma^2_t = f_\sigma(x_{t-1}, m_{t-1}, u_{t-1})$$

는 학습 데이터의 도메인 특성에 과도하게 의존하지 않는다. 따라서 새로운 도메인으로의 전이가 용이하다.

**3. 명시적 확률적 모델링**

DPM의 분산 정규화 항:

$$\mathcal{L}_{\text{nll}} = \int \log \mathcal{N}(y | \mu_\theta, \sigma^2_\theta) dy$$

는 명시적으로 확률 분포를 모델링하므로, 테스트 데이터의 분포 변화에 대해 더 강건하다.[1]

#### 5.2 저해상도에서 고해상도로의 전이 성공

**원인 분석**:

1. **해상도 독립성**: U-Net 구조가 비율적 다운샘플링을 수행하므로, 서로 다른 해상도에서도 **수용 필드(receptive field) 대 이미지 크기의 비율**이 유지된다.

2. **불확실성 맵의 일반성**: 불확실성은 절대값이 아닌 **상대적 신뢰도**를 나타내므로, 해상도 변화에 강건하다.[1]

3. **마스크 스케줄의 유연성**: 선형 마스크 스케줄은 마스크 비율에만 의존하며, 절대 픽셀 좌표에 의존하지 않는다.[1]

실험적 증거: 512×512에서 훈련된 모델이 1024×1024 테스트에서 **훨씬 더 많은 데이터(50M vs 20M)로 훈련된 기존 모델들**을 능가했다는 점이 이를 입증한다.[1]

#### 5.3 정량화: 일반화 한계

**Out-of-Distribution (OOD) 강건성**:

PSM은 다음과 같은 OOD 시나리오에서도 성능 유지:[1]
- 작은 마스크 ↔ 큰 마스크 (동일 모델 사용)
- 얼굴(CelebA) ↔ 장면(Places2) (상대적으로 완만한 성능 저하)

**한계(Limitations)**:

논문에서 명시된 한계점:[1]
1. "대규모 결손 영역에서 주어진 힌트가 적을 때 객체 이해 어려움"
2. "작은 세부 사항의 변경 경향 > 큰 구조의 변경"
3. "다양성 개선 필요" (회상값이 LDM에 비해 약간 낮음)

***

### 6. 한계 및 실패 사례 분석

#### 6.1 구조적 한계

**1. 의미론적 이해 부족**

불충분한 맥락에서 객체를 복구하기 어려움. 예시:[1]
- 노트북 일부 누락 시 배경으로 채워짐
- 버스 구조가 불완전하게 복구됨

**원인**: 명시적 의미론적 정보(예: 객체 레이블, 3D 구조) 없이 학습되었기 때문[1]

**2. 픽셀 차원의 보수**

개별 픽셀 단위 예측으로 인해:
- 전역 기하학적 일관성 부족 가능성
- 패턴 반복 오류

**3. 마스크 특이적 학습**

훈련 시 특정 마스크 분포에 최적화되어, 극단적 마스크 형태에 대한 성능 저하 가능성[1]

#### 6.2 계산 비용

반복 횟수 증가에 따른 추론 시간 증가:[1]

| 반복 수 | 추론 시간 |
|-------|----------|
| 3 반복 | 250ms |
| 4 반복 | 333ms |
| 10 반복 | 833ms |

최대 4배 시간 증가로, 실시간 응용(예: 비디오)에는 부적절[1]

#### 6.3 훈련 안정성

GAN 기반 접근법의 고유 문제:[1]
- 판별기-생성기 간 경쟁의 미묘한 균형 필요
- 하이퍼파라미터 민감도 (예: $\lambda_3 = 1 \times 10^{-4}$의 중요성)

***

### 7. 관련 최신 연구 동향 (2020년 이후)

#### 7.1 확산 모델 기반 보수 (2021-2025)

**주요 발전**:[2][3][4][5][6][7][8]

1. **LatentPaint (2024)**:[3]
   - 잠재 공간에서 확산을 수행하여 계산 비용 감소
   - 전진-후진 융합 단계를 잠재 공간에서 수행
   - PSM보다 여전히 느리지만 학습 불필요

2. **StrDiffusion (2024)**:[4]
   - 구조 가이드 확산 모델
   - 의미론적 일관성 문제 해결 시도
   - PSM의 의미론적 이해 한계를 보완 가능

3. **MMGInpainting (2024)**:[5]
   - 멀티모달 가이드 (이미지 + 텍스트)
   - NAFNet과 의미론적 융합 인코더 활용
   - PSM의 제어성 부족 극복

4. **HD-Painter (2023)**:[8]
   - 프롬프트 인식 주의(PAIntA) 층
   - 2K 해상도 지원
   - PSM의 고해상도 성능을 더 극대화

5. **RAD (2024)**:[7]
   - 영역 인식 확산 모델
   - 픽셀별 비동기 생성
   - **PSM의 추론 속도 대비 100배 빠름**이라고 주장하나 품질 차이 존재

#### 7.2 반복적 보수 및 신뢰도 기반 방법

**2020년대 초 선구자 연구**:[9]

1. **High-Resolution Image Inpainting with Iterative Confidence Feedback (2020)**:[9]
   - PSM과 유사한 신뢰도 맵 기반 반복 보수
   - 하지만 GAN 암시적 최적화 없이 명시적 신뢰도만 사용
   - PSM이 이를 개선하여 확률적 모델링 통합

2. **Progressive with Purpose (2022)**:[10]
   - GLE(Global and Local Edge) 특징 활용
   - 구조 정보 기반 점진적 충전
   - PSM의 픽셀 차원 한계 보완

#### 7.3 불확실성 정량화 및 일반화

**최신 동향**:[11][12][13]

1. **Cycle-Consistency-Based Uncertainty Quantification (2023)**:[11]
   - 순환 일관성을 통한 불확실성 추정
   - PSM의 가우스 가정을 넘어 일반화된 불확실성 프레임워크

2. **Epistemic Uncertainty for Generated Image Detection (2024)**:[12]
   - 가중치 교란(weight perturbation)을 통한 불확실성 추정
   - PSM과는 다른 불확실성 개념 (생성 vs 예측)
   - OOD 검출에 응용 가능

3. **Uncertainty Estimation in Medical Image Classification (2022)**:[13]
   - MCDO(Monte Carlo Dropout)과 Deep Ensemble 비교
   - PSM의 명시적 분산 모델링이 이들보다 효율적임을 시사

#### 7.4 고해상도 및 3D 일관성

**2024년 최신**:

1. **3D-consistent Image Inpainting (2024)**:[14]
   - 다중 뷰 일관성 보장
   - PSM을 3D로 확장하는 가능성 제시

2. **Consistent Image Inpainting with Cross-Perception (2025)**:[15]
   - 구조-레이아웃-텍스처 간 순환적 상호작용
   - PSM의 반복 메커니즘을 인간의 지각과정에 더 가깝게 모델링

***

### 8. 연구 영향 및 미래 방향

#### 8.1 PSM이 미치는 학문적 영향

**1. 패러다임 전환**

PSM은 이미지 보수에서 다음과 같은 패러다임 전환을 유도했다:[2][3][1]

$$\text{한 번 생성(one-shot)} \to \text{점진적 반복 기반 확산}$$

기존 GAN의 근본적 한계(불안정성)를 반복적 설계로 극복한 것이다. 이는 후속 확산 기반 방법들()에 영감을 주어 **반복적 정제(iterative refinement) 기반 보수의 대세화**를 이끌었다.[4][5][7]

**2. 불확실성 모델링의 중요성**

PSM은 분리된 확률적 모델링을 통해 다음을 입증했다:[1]

$$\text{불확실성} = \text{픽셀 신뢰도} = \text{다음 반복 가이드}$$

이는 의료영상 분석()과 OOD 검출()에서 불확실성 추정의 중요성을 강조하는 후속 연구로 이어졌다.[12][13]

**3. 경량 고효율 모델의 가능성**

- PSM: 74M 매개변수, 250ms 추론, FID 1.68
- LaMa: 5127M 매개변수, 수 초 추론, FID 2.97
- 69배 적은 매개변수로 **30% 성능 향상**

이는 대규모 모델의 필요성에 의문을 제기하고, **구조적 설계의 중요성**을 부각시켰다.

#### 8.2 향후 연구 시 고려할 점

**A. 의미론적 이해 강화**

**1. 다중 모달리티 통합**

후속 연구()가 취한 접근:[5][8]

$$\text{시각 정보} + \text{텍스트 가이드} + \text{3D 기하학} \to \text{고품질 보수}$$

**제안 방향**:
- PSM의 불확실성 맵에 의미론적 정보 추가
- 객체 클래스 정보 조건화
- 구조-텍스처 분해 (예: Normal map, Depth map)

**2. 의미론적 손실 함수 개선**

현재 PSM의 손실:[1]

$$\mathcal{L} = \mathcal{L}_{\text{adv}} + \mathcal{L}_{\text{pcp}} + \mathcal{L}_{\text{nll}}$$

개선 제안:

$$\mathcal{L}_{\text{improved}} = \mathcal{L}_{\text{adv}} + \mathcal{L}_{\text{pcp}} + \mathcal{L}_{\text{nll}} + \lambda_4 \mathcal{L}_{\text{semantic}} + \lambda_5 \mathcal{L}_{\text{structure}}$$

**B. 계산 효율성 극대화**

**1. 적응적 반복 횟수**

현재: 테스트 시 고정된 반복 수 (4-10)

개선 제안: 동적 반복 종료 조건
$$\text{Stop if } \max_i u_t[i] < \epsilon_{\text{threshold}}$$

기대 효과: 간단한 이미지는 3 반복, 복잡한 이미지는 8 반복 등 적응적 추론

**2. 다중 스케일 구조**

후속 연구의 시도():[7]
- 픽셀별 비동기 생성
- 조건부 계산(Conditional Computation)

**C. 일반화 성능 심화**

**1. 도메인 간 전이**

실험된: 512×512 → 1024×1024[1]

미실험: 
- 얼굴 → 의료 영상
- 자연 이미지 → 판화/그림
- RGB → 멀티스펙트럼 이미지

**제안 방법**:
$$\text{Domain Gap} = \mu_{\text{target}} - \mu_{\text{source}}$$
$$\Rightarrow \text{도메인 적응 불확실성 정규화}$$

**2. 메타-러닝 기반 적응**

PSM을 MAML(Model-Agnostic Meta-Learning) 프레임워크에 통합:
$$\theta^* = \arg\min_\theta \sum_{\text{tasks}} L_{\text{task}}(\theta)$$

이를 통해 소수의 샘플로 새 도메인에 빠르게 적응 가능

**D. 이론적 기초 강화**

**1. 수렴 분석**

현재: 실증적 성능 평가

제안: 반복 과정의 수렴 보장

```math
\|u_t - u_*\| \leq \rho^t \|u_0 - u_*\|
```

여기서 $u_*$는 최적 불확실성, $\rho < 1$은 수렴율

**2. 생성 다양성 이론**

PSM의 회상값이 LDM에 비해 낮은 이유를 분석:

$$\text{Diversity} = \mathbb{E}_{z} [d(\hat{x}(z_1), \hat{x}(z_2))]$$

가설: 적대적 최적화로 인한 모드 붕괴(mode collapse)

개선: 다양성 정규화 손실 추가

**E. 멀티태스크 확장**

**1. 비디오 보수**

현재 PSM: 단일 이미지

확장:
$$x_t^{(i)}, m_t^{(i)} = f(x_{t-1}^{(i)}, m_{t-1}^{(i)}, u_{t-1}^{(i)}, x_{t-1}^{(i±1)})$$

시간적 일관성 손실 추가

**2. 텍스트-이미지 생성

PSM의 반복적 확산 메커니즘을 text-to-image로 확장:

$$\text{CLIP embedding} \to \text{불확실성 가이드 생성}$$

기대 효과: 더 제어 가능한 생성형 모델

**F. 실제 응용 시나리오**

**1. 사진 복원**

손상된 유산 사진 복구: 의미론적 정보 부족이 큰 한계

개선: 사진의 역사적 맥락 정보 조건화

**2. 의료영상 분석**

CT/MRI 보수: 정확도가 매우 중요

개선: 
- 임상적 해석 가능성 강조
- 불확실성 시각화
- 의사의 수동 개입 여지 제공

***

### 9. 결론

PSM은 이미지 보수 분야에 혁신적 기여를 하였다:

**핵심 기여:**
1. 분리된 확률적 모델링으로 **GAN 효율성 + 확산 모델 안정성** 결합
2. 불확실성 기반 픽셀 확산으로 **반복적 정제의 새 패러다임** 제시
3. 매개변수 대비 성능에서 **SOTA 달성** (74M vs 5127M)

**학문적 영향:**
- 반복적 기반 보수의 대세화 ()[3][4][5][7]
- 불확실성 모델링의 중요성 강조 ()[11][12]
- 경량 고효율 구조 설계의 새 가능성 제시

**미래 과제:**
1. 의미론적 이해 강화를 위한 다중모달 통합
2. 계산 효율성과 다양성의 트레이드오프 해결
3. 다양한 도메인 및 해상도로의 일반화
4. 이론적 수렴성 및 생성 다양성 분석

PSM은 단순히 SOTA 달성을 넘어, **구조적 설계의 중요성**과 **확률적 모델링의 해석 가능성**이 고효율 생성 모델 개발의 핵심임을 입증했다.[1]

***

### 참고 자료

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b984e51f-789f-4c51-8a1e-1bc695220de2/2212.02963v2.pdf)
[2](https://ieeexplore.ieee.org/document/10495652/)
[3](https://ieeexplore.ieee.org/document/10483967/)
[4](https://ieeexplore.ieee.org/document/10657377/)
[5](https://ieeexplore.ieee.org/document/10480591/)
[6](https://arxiv.org/abs/2401.14832)
[7](https://ieeexplore.ieee.org/document/11095134/)
[8](https://arxiv.org/abs/2312.14091)
[9](https://link.springer.com/10.1007/978-3-030-58529-7_1)
[10](https://arxiv.org/pdf/2209.10071.pdf)
[11](https://spj.science.org/doi/10.34133/icomputing.0071)
[12](https://arxiv.org/html/2412.05897v2)
[13](https://pmc.ncbi.nlm.nih.gov/articles/PMC9382553/)
[14](https://europe.naverlabs.com/research/publications/3d-consistent-image-inpainting-with-diffusion-models/)
[15](https://ieeexplore.ieee.org/document/11212802/)
[16](https://arxiv.org/abs/2410.21966)
[17](https://ieeexplore.ieee.org/document/10859584/)
[18](https://arxiv.org/abs/2412.01682)
[19](https://arxiv.org/abs/2310.07222)
[20](https://arxiv.org/html/2502.03491v1)
[21](http://arxiv.org/pdf/2311.11469.pdf)
[22](https://arxiv.org/html/2412.01223v1)
[23](https://arxiv.org/pdf/2312.14091.pdf)
[24](https://arxiv.org/pdf/2307.10584.pdf)
[25](https://arxiv.org/html/2403.19898v1)
[26](https://arxiv.org/abs/2201.09865)
[27](https://arxiv.org/abs/2401.03349)
[28](https://www.biorxiv.org/content/10.1101/2024.11.16.623969v1)
[29](https://www.sciencedirect.com/science/article/pii/S0893608022000673)
[30](https://openaccess.thecvf.com/content/WACV2024/papers/Corneanu_LatentPaint_Image_Inpainting_in_Latent_Space_With_Diffusion_Models_WACV_2024_paper.pdf)
[31](https://pmc.ncbi.nlm.nih.gov/articles/PMC4854961/)
[32](https://www.ai.rug.nl/~mwiering/Thesis-Folke-Drost.pdf)
[33](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03014.pdf)
[34](https://ieeexplore.ieee.org/document/10657590/)
[35](http://pubs.rsna.org/doi/10.1148/radiol.231971)
[36](https://arxiv.org/abs/2403.01633)
[37](https://open-publishing.org/publications/index.php/APUB/article/view/1335)
[38](https://pmc.ncbi.nlm.nih.gov/articles/PMC11135165/)
[39](https://academic.oup.com/jscd/article/doi/10.1093/jscdis/yoae002.039/7686716)
[40](https://spir.aoir.org/ojs/index.php/spir/article/view/13391)
[41](https://ejournal.poltekkes-smg.ac.id/ojs/index.php/jimed/article/view/13318)
[42](https://arxiv.org/html/2410.21966)
[43](https://arxiv.org/html/2312.03594)
[44](http://arxiv.org/pdf/1810.08774.pdf)
[45](http://arxiv.org/pdf/2403.16016.pdf)
[46](https://arxiv.org/pdf/2211.13857.pdf)
[47](http://arxiv.org/pdf/2402.03501.pdf)
[48](https://openreview.net/pdf/1e81620ef68cdb56dc5ca52bc4fa349c9b1ec33b.pdf)
[49](https://www.diva-portal.org/smash/get/diva2:1752144/FULLTEXT01.pdf)
[50](https://pubmed.ncbi.nlm.nih.gov/37646491/)
[51](https://www.semanticscholar.org/paper/Progressive-Image-Inpainting-with-Full-Resolution-Guo-Chen/e4609a5e3cc221fb21b4b86f2f5edc4f80bb42f5)
[52](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wei_Conditional_Single-View_Shape_Generation_for_Multi-View_Stereo_Reconstruction_CVPR_2019_paper.pdf)
[53](https://arxiv.org/abs/2304.06671)
[54](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620229.pdf)
[55](https://arxiv.org/abs/2305.09121)
[56](https://github.com/AlonzoLeeeooo/awesome-image-inpainting-studies)
