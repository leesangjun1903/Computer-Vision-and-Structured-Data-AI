# EGC: Image Generation and Classification via a Diffusion Energy-Based Model

### 1. 논문의 핵심 주장 및 주요 기여

EGC (Energy-based classifier and Generator)는 **단일 신경망으로 이미지 분류와 생성을 동시에 우수한 성능으로 수행**하는 획기적인 모델입니다. 기존 접근 방식들은 한 가지 작업에서 뛰어난 성능을 보이면 다른 작업에서는 성능이 저하되는 트레이드오프 문제를 겪었습니다. EGC의 핵심 혁신은 다음과 같습니다:[1]

**핵심 기여:**

1. **판별학습과 생성학습의 통합**: 정방향 패스에서는 분류 모델로, 역방향 패스에서는 생성 모델로 작동하는 에너지 기반 모델을 제시합니다. 표준 분류기가 조건부 분부 분포 $p(y|x)$를 출력하는 것과 달리, EGC는 결합 분포 $p(x,y)$를 모델링합니다.

2. **우수한 이중 성능**: CIFAR-10에서 95.9%의 분류 정확도와 FID 3.30의 생성 품질을 동시에 달성합니다. 이는 비슷한 아키텍처의 Wide ResNet-28-12 (95.6%)을 능가합니다.[1]

3. **확산 과정과 EBM의 결합**: EBM의 유연성과 확산 모델의 안정성을 결합하여 저밀도 영역에서의 점수 추정 정확도를 개선합니다.[1]

4. **응용 다양성**: 인페인팅, 의미론적 보간, 초고해상도 생성(∼1024²), 적대적 공격에 대한 견고성 향상 등 다양한 응용을 시연합니다.[1]

***

### 2. 해결하고자 하는 문제, 제안하는 방법 및 모델 구조

#### 2.1 핵심 문제 정의

**문제 배경:**
- 이미지 분류: 조건부 확률 분포 $p(y|x)$ 모델링
- 이미지 생성: 간단한 사전 분포 $p(z)$를 목표 분포 $p(x)$로 변환
- 기존 한계: 두 작업을 동시에 잘 수행하는 단일 모델이 부재[1]

에너지 기반 모델은 유연성이 있지만 다음의 문제점이 있습니다:[1]
- 정규화 상수 $Z(\theta)$의 계산 불가능성
- Langevin 동역학을 통한 불안정한 샘플링
- 저밀도 영역에서의 부정확한 점수 추정

#### 2.2 제안하는 방법론

**에너지 기반 모델의 기본 정의:**

$$p_\theta(\mathbf{x}) = \frac{\exp(-E_\theta(\mathbf{x}))}{Z(\theta)}$$

여기서 $E_\theta(\mathbf{x}): \mathbb{R}^D \to \mathbb{R}$는 에너지 함수이고, $Z(\theta) = \int \exp(-E_\theta(\mathbf{x}))d\mathbf{x}$는 분할 함수입니다.[1]

**확산 과정의 통합:**

확산 과정은 단계적으로 데이터에 노이즈를 추가합니다:[1]

$$q(\mathbf{x}_{1:T}|\mathbf{x}_0) = \prod_{t=1}^{T} q(\mathbf{x}_t|\mathbf{x}_{t-1}), \quad q(\mathbf{x}_t|\mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{\alpha_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})$$

임의의 타임스텝 $t$에서 직접 샘플링:[1]

$$q(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$$

Tweedie의 공식을 적용하면:[1]

$$\sqrt{\bar{\alpha}_t}\mathbf{x}_0 = \mathbf{x}_t + (1-\bar{\alpha}_t)\nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t|\mathbf{x}_0)$$

점수 함수는:[1]

$$\nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t|\mathbf{x}_0) = -\frac{\epsilon_t}{\sqrt{1-\bar{\alpha}_t}}$$

**Fisher 발산을 통한 최적화:**

$$D_F = \mathbb{E}_q\left[\frac{1}{2}\|\nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t|\mathbf{x}_0) - \nabla_{\mathbf{x}_t}\log p_\theta(\mathbf{x}_t)\|^2\right]$$

이는 EBM의 정규화 상수 $Z(\theta)$ 최적화를 회피합니다.[1]

#### 2.3 EGC의 핵심 구조

**결합 분포 모델링:**

표준 분류기의 로짓 $f(\mathbf{x})[y]$를 이용한 결합 확률:[1]

$$p_\theta(\mathbf{x}, y) = \frac{\exp(f_\theta(\mathbf{x})[y])}{Z(\theta)}$$

이를 통해 조건부 확률을 도출:[1]

$$p(\mathbf{y}|\mathbf{x}) = \frac{\exp(f(\mathbf{x})[y])}{\sum_{y'} \exp(f(\mathbf{x})[y'])}$$

**자유 에너지 함수:**

$$E_\theta(\mathbf{x}) = -\log \sum_y \exp(f_\theta(\mathbf{x})[y])$$

따라서 점수는:[1]

$$\nabla_\mathbf{x} \log p_\theta(\mathbf{x}) = \nabla_\mathbf{x} \log \sum_y \exp(f_\theta(\mathbf{x})[y])$$

**조건부 점수를 통한 가이드:**

Bayes 정리에 의해:[1]

$$\nabla \log p_\theta(\mathbf{x}_t|y) = \nabla \log p_\theta(\mathbf{x}_t) + \nabla \log p_\theta(y|\mathbf{x}_t)$$

**통합 학습 목표:**

$$\mathcal{L} = \mathbb{E}_q\left[\frac{1}{2}\|\nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t|\mathbf{x}_0) - \nabla_{\mathbf{x}_t}\log p_\theta(\mathbf{x}_t)\|^2 - \gamma\sum_{i=1}^{C} q(y_i|\mathbf{x}_t, \mathbf{x}_0)\log p_\theta(y_i|\mathbf{x}_t)\right]$$

첫 번째 항은 재구성 손실, 두 번째 항은 분류 손실입니다. 하이퍼파라미터 $\gamma$는 두 손실 항의 균형을 맞춥니다.[1]

#### 2.4 모델 아키텍처

**네트워크 설계:**[1]
- U-Net 아키텍처 기반 (ADM과 동일)
- 어텐션 풀링 모듈 추가로 로짓 예측
- 스펙트럼 정규화는 사용하지 않음

**훈련 알고리즘 (Algorithm 1):**[1]
```
repeat:
  t ~ Uniform({1, ..., T})
  (x₀, y) 샘플링, ε ~ N(0, I)
  xₜ = √ᾱₜx₀ + √(1-ᾱₜ)ε
  경사 하강 스텝:
    ∇θ(||∇ₓₜlog pθ(xₜ) + ε||² - γ Σᶜᵢ₌₁ q(yᵢ|xₜ)log pθ(yᵢ|xₜ))
until 수렴
```

**샘플링 알고리즘 (Algorithm 2):**[1]
```
xₜ ~ N(0, I)
for t = T, ..., 1:
  ε ~ N(0, I) (t > 1이면), else ε = 0
  xₜ₋₁ = 1/√αₜ(xₜ + (1-αₜ)∇ₓₜlog pθ(xₜ)) + √βₜε
return x₀
```

***

### 3. 성능 향상 및 한계

#### 3.1 성능 향상 결과

**CIFAR-10 하이브리드 모델링:**[1]

| 메트릭 | EGC | Wide ResNet-28-12 | 최고 생성 모델 |
|--------|------|------------------|-------------|
| 정확도 (%) | **95.9** | 95.6 | N/A |
| IS (↑) | **9.43** | N/A | DDPM: 9.46 |
| FID (↓) | **3.30** | N/A | DDPM: 3.17 |

EGC는 분류에서 표준 분류기를 능가하고, 생성 품질에서도 최고 수준의 점수 기반 모델과 경쟁합니다.[1]

**ImageNet-1k 결과:**[1]

| 설정 | 정확도 (%) | IS (↑) | FID (↓) |
|------|----------|--------|---------|
| EGC (기본) | 70.4 | 79.9 | 17.5 |
| EGC† (조건+무조건) | 72.5 | 189.5 | 6.77 |
| EGC‡ (RandResizeCrop) | **78.9** | **231.3** | **6.05** |

RandResizeCrop 데이터 증강으로 78.9% 정확도 달성 (ADM-Classifier의 64.3%을 크게 상회).[1]

**CIFAR-100 결과:**[1]
- 정확도: 77.9%
- IS: 11.50
- FID: 4.88
- JEM (72.2%)을 능가하며 GAN 기반 메소드와 경쟁[1]

**무조건 생성 모델 (CelebA-HQ, LSUN Church):**[1]

| 데이터셋 | EGC FID | ATEEBM | DDPM |
|----------|---------|--------|------|
| CelebA-HQ | **7.75** | 17.31 | - |
| LSUN Church | **8.97** | 14.87 | 7.89 |

EBM 방법 중 최고 성능을 달성하면서 DDPM과 경쟁합니다.[1]

#### 3.2 절제 연구 (Ablation Study)

**CIFAR-10에서의 각 구성 요소 기여도:**[1]

| EBM | 분류기 | 가이드 | 네트워크 | 정확도 (%) | FID (↓) |
|-----|--------|--------|---------|----------|---------|
| ✓ | - | - | U-Net | - | 5.36 |
| ✓ | ✓ | - | U-Net | 95.9 | 3.49 |
| ✓ | ✓ | ✓ | U-Net | 95.9 | **3.30** |
| ✓ | ✓ | ✓ | ResNet | 95.9 | 7.15 |

- **무조건 EBM**: FID 5.36
- **EGC (분류기 추가)**: FID 1.87 개선 (3.49로)
- **분류기 가이드 추가**: FID 0.19 추가 개선 (3.30으로)[1]

U-Net의 스킵 연결이 세밀한 세부사항 전파에 중요함을 시각화합니다.[1]

#### 3.3 적대적 견고성

**CIFAR-10에서의 FGSM/PGD 공격 저항성:**[1]

- EGC는 Wide ResNet-28-12와 JEM (92.9%)을 모두 능가합니다
- 적대적 훈련을 통해 학습된 결합 확률 분포의 견고성 이점을 확인[1]

#### 3.4 주요 한계

**1. 생성 품질의 한계:**
- ImageNet 기본 설정에서 FID 17.5로, DDPM (12.3)과 LDM (7.77)에 비해 여전히 격차 존재[1]
- 저용량 데이터 증강 (랜덤 플립만 사용)으로 인한 한계를 저자들이 인정[1]

**2. 분류 정확도의 약간의 저하:**
- ImageNet 기본 설정에서 70.4%로 최고 수준 분류기 대비 약간 낮음[1]
- 더 강력한 증강 전략이 필요함[1]

**3. 일반화 성능:**
- 다른 아키텍처 (ResNet)에서 U-Net보다 성능 저하 (7.15 vs 3.30 FID)[1]
- 이는 아키텍처 선택이 중요함을 시사

**4. 계산 복잡도:**
- 확산 과정의 반복 역전파로 인한 상대적으로 높은 학습 비용
- 샘플링 단계 수가 많음 (1000 스텝)[1]

**5. 하이퍼파라미터 민감성:**
- 가중치 인자 $\gamma$ 조정 필요 (CIFAR: 0.001, ImageNet: 0.005)[1]
- 노이즈 스케줄 선택에 따른 성능 변화

***

### 4. 모델의 일반화 성능 향상 가능성

#### 4.1 현재 일반화 성능

**에너지 랜드스케이프 분석:**

논문은 신경망이 Fisher 발산 최적화에도 불구하고 목표 가우시안 분포를 효과적으로 모델링함을 시각화합니다.[1]

- Figure 6(a): 노이즈 수준에 따른 비정규화 확률 밀도가 폴드 정규분포 형태 (가우시안 유사)
- Figure 6(b), (c): 2D 확률 밀도가 가우시안 분포와 유사한 형태[1]

이는 **에너지 기반 모델이 점수 함수만 최적화해도 전체 분포를 잘 학습**함을 의미합니다.

#### 4.2 일반화 향상 메커니즘

**조건부 점수와 무조건부 점수의 시너지:**

$$\nabla_{\mathbf{x}_t}\log p_\theta(\mathbf{x}_t|y) = \underbrace{\nabla_{\mathbf{x}_t}\log p_\theta(\mathbf{x}_t)}_{\text{무조건부 점수}} + \underbrace{\nabla_{\mathbf{x}_t}\log p_\theta(y|\mathbf{x}_t)}_{\text{분류기 가이드}}$$

이 두 항의 결합은:[1]

1. **클래스 일관성**: 분류기 가이드가 생성 과정을 특정 클래스로 유도
2. **다양성 보존**: 무조건부 점수가 자연스러운 데이터 분포 유지
3. **적대적 견고성**: 결합 분포 학습이 판별 정경계를 더 강건하게 형성[1]

#### 4.3 Future 일반화 개선 방향

**1. 아키텍처 특화:**
- U-Net 특화 설계 (스킵 연결 강화)
- Vision Transformer 기반 아키텍처 탐색[1]

**2. 확산 프로세스 개선:**
- 동적 노이즈 스케줄 적응 (현재: 고정 선형/코사인)
- 중요도 샘플링을 통한 타임스텝 가중치[1]

**3. 데이터 증강 최적화:**
- ImageNet에서 RandResizeCrop 적용 시 78.9% 달성 (기본 70.4% → +8.5%)[1]
- 의미론적으로 일관된 증강 전략 개발[1]

**4. 손실 함수 재설계:**
- $\gamma$ 값의 적응적 조정 (현재: 고정값)
- 다중 스케일 손실 항 추가[1]

**5. 준지도 학습 확장:**
- 라벨 없는 데이터 활용으로 점수 함수 학습 안정화
- 반자동 라벨링과의 결합[1]

***

### 5. 연구에 미치는 영향 및 향후 고려사항

#### 5.1 이론적 영향

**에너지 기반 모델 이론의 진전:**

EGC는 JEM (Grathwohl et al., 2020)의 근본적인 한계를 극복합니다:[1]

| 측면 | JEM | EGC |
|------|-----|-----|
| 훈련 안정성 | Langevin MCMC 필요 | Fisher 발산만 사용 |
| 점수 추정 | 확률 밀도 직접 최적화 | 점수 함수만 최적화 |
| 정규화 상수 | 근사 필요 | 회피 가능 |
| 분류 정확도 (CIFAR-10) | 92.9% | **95.9%** |
| FID (CIFAR-10) | 38.4 | **3.30** |

**확산 모델의 판별 활용:**

기존 관점: 확산 모델 = 생성 전문
새로운 통찰: 확산 모델의 중간 표현이 판별 정보 포함[1]

이는 Diffusion Models Beat GANs on Image Classification (2023)과 일맥상통하며, 생성 모델의 판별 능력을 체계적으로 활용합니다.[2]

#### 5.2 실제 응용의 영향

**1. 데이터 증강과 정규화:**
- 생성 모델로 데이터 부족 시나리오 해결[1]
- 합성 데이터가 분류 성능을 직접 개선[1]

**2. 강건성과 보안:**
- 적대적 견고성 개선으로 안전-중시 응용 (자율주행, 의료진단) 적합[1]
- Out-of-Distribution 탐지 향상[1]

**3. 멀티태스크 학습:**
- 단일 모델로 분류와 생성 동시 처리
- 메모리 및 계산 효율성 향상[1]

#### 5.3 향후 연구 시 고려할 점

**1. 모델 확장성:**

$$\text{확장 가능성} = \frac{\text{대규모 데이터 (ImageNet) 성능}}{\text{소규모 데이터 (CIFAR-10) 성능}}$$

- ImageNet에서 추가 증강 없이 기본 70.4% (개선 필요)[1]
- 고해상도 이미지 (1024×1024)에서의 성능 안정화[1]

**2. 계산 효율성:**

현재 EGC의 주요 제약:
- 1000 단계 확산 프로세스 (추론 시간 증가)[1]
- 매 단계마다 역전파 필요 (메모리 오버헤드)[1]

개선 방향:
- 가속화된 샘플러 (DDIM 형태)
- 지식 증류 (한 단계 생성 모델)[1]

**3. 아키텍처 일반화:**

ResNet 기반에서의 성능 저하 (U-Net 3.30 vs ResNet 7.15 FID):[1]
- 모든 아키텍처에서 효과적인 설계 원칙 도출[1]
- Vision Transformer 호환성 확인[1]

**4. 이론적 심화:**

$$\text{Fisher 발산}\ D_F = \mathbb{E}_q\left[\frac{1}{2}\|\nabla_\mathbf{x}\log q(\mathbf{x}|\mathbf{x}_0) - \nabla_\mathbf{x}\log p_\theta(\mathbf{x})\|^2\right]$$

에 의존하는 이론적 정당성:[1]
- 정규화 상수 회피의 정보 이론적 근거
- 최적성 조건 분석
- 수렴 보장 증명[1]

**5. 응용별 최적화:**

| 응용 | 우선순위 | 고려사항 |
|------|---------|---------|
| 의료 영상 | 높음 | 클래스 불균형, 라벨 부족 |
| 자율주행 | 높음 | 실시간성, 견고성 |
| 콘텐츠 생성 | 중간 | 다양성 vs 충실성 트레이드오프 |
| 데이터 증강 | 중간 | 합성 데이터 신뢰성 |

***

### 6. 2020년 이후 관련 최신 연구 비교 분석

#### 6.1 에너지 기반 모델 (EBM) 진화

**JEM (Your Classifier is Secretly an Energy Based Model, 2020):**[3]

- **핵심 아이디어**: 표준 소프트맥스 분류기를 결합 분포 $p(x,y)$의 EBM으로 해석
- **성과**: CIFAR-10에서 분류 92.9%, FID 38.4
- **한계**: SGLD 샘플링 불안정, Lipschitz 제약 필요
- **EGC와의 비교**:
  - JEM: 확률 밀도 직접 최적화 → 불안정
  - EGC: Fisher 발산으로 점수만 최적화 → **95.9% 정확도, FID 3.30**[1]

**SADA-JEM (Towards Bridging the Performance Gaps of JEM, 2023):**[4]

- **혁신**: Sharpness-Aware Minimization (SAM) + selective data augmentation
- **성과**: 
  - CIFAR-10: 95.5% 정확도, FID 9.41 (vs JEM 37.1)
  - CIFAR-100: FID 14.4 (vs JEM 33.7)
- **EGC와의 비교**:
  - 비슷한 정확도 (95.9% vs 95.5%)
  - EGC의 생성 품질이 더 우수 (FID 3.30 vs 9.41)[4][1]

**M-EBM (Towards Understanding the Manifolds of EBMs, 2023):**[5]

- **핵심**: 다양체 기반 EBM으로 훈련 안정성/속도 개선
- **성과**: CIFAR-10 FID 개선, ImageNet 32×32에서 경쟁력
- **차별점**: EGC는 확산 프로세스로 유사 효과 달성[1]

#### 6.2 확산 모델의 판별 응용

**Diffusion Models Beat GANs on Image Classification (2023):**[6]

- **발견**: 확산 모델의 중간 표현이 우수한 분류 정보 함유
- **성과**: ImageNet에서 BigBiGAN 능가
- **EGC와의 차이**:
  - 이 연구: 사전 학습된 생성 모델 → 분류 특성 추출
  - EGC: 단일 모델로 두 작업 동시 최적화 (end-to-end)[1]

**Synthetic Data from Diffusion Models Improves ImageNet Classification (2023):**[7]

- **방법**: 확산 모델로 합성 데이터 생성 → 분류기 학습에 사용
- **성과**: Classification Accuracy Score 64.96 (256×256), 69.24 (1024×1024)
- **EGC와의 차이**:
  - 이 연구: 두 단계 (생성 → 분류)
  - EGC: 한 단계 (통합 훈련)[1]

**Semantic-Guided Generative Image Augmentation (SGID, 2023):**[8]

- **방법**: 확산 모델 + 의미론적 가이드로 데이터 증강
- **특징**: 의미 일관성 보존하며 다양성 증가
- **EGC와의 관계**: EGC의 조건부 점수가 유사한 의미론적 제어 달성[1]

#### 6.3 통합 판별-생성 학습

**Vermouth (Bridging Generative and Discriminative Models, 2024):**[9]

- **접근**: Stable Diffusion + 통합 헤드로 다양한 시각 인식 작업
- **특징**: 계층적 특성 활용, 다양한 의미 세분성 지원
- **EGC와의 비교**:
  - Vermouth: 사전 학습 모델 + 어댑터 (무겁고 복잡)
  - EGC: 단일 신경망 (경량)[1]

**EB-CLIP (Text-to-Image Generation via Energy-Based CLIP, 2024):**[10]

- **혁신**: JEM을 CLIP 공간으로 확장하여 멀티모달 생성-판별 통합
- **성과**: 고해상도 텍스트-이미지 생성
- **EGC와의 차이**:
  - EB-CLIP: 멀티모달 (텍스트-이미지)
  - EGC: 단일모달 (이미지 중심)[1]

**Mediffusion (Semi-Supervised Classification and Medical Image Generation, 2024):**[11]

- **방법**: 확산 기반 결합 모델로 반지도학습 + 설명 가능한 분류
- **응용**: 의료 영상 (데이터 부족 해결)
- **EGC와의 공통점**: 단일 네트워크로 분류+생성, 반지도학습 가능성[1]

#### 6.4 점수 기반 생성 모델 (Score-Based Generative Models)

**Score-Based Generative Modeling (Yang Song et al., 2021):**[12]

- **기본 개념**: 점수 함수 (데이터 분포의 로그 기울기) 학습
- **장점**: GAN의 대적 훈련 불필요, 유연한 구조, 정확한 로그 우도 계산 가능
- **기여**: EGC의 이론적 기초[12][1]

**Score-Based Generative Modeling with Critically-Damped Langevin Diffusion (2021):**[13]

- **혁신**: 분자 동역학의 속도 변수 도입으로 더 효율적인 확산
- **성과**: 더 복잡한 점수 함수 학습 피함
- **EGC와의 관계**: EGC의 Fisher 발산도 유사한 정보 이론적 이점 제공[13][1]

#### 6.5 확산 모델의 고속화

**PIXART-δ (Latent Consistency Models, 2024):**[14]

- **방법**: LCM + ControlNet으로 확산 가속화
- **성과**: 1024×1024 생성을 0.5초에 달성 (7배 개선)
- **EGC와의 고려사항**: EGC도 고속 샘플링을 위한 유사 기법 적용 가능[14][1]

**Optical Diffusion Models (2024):**[15]

- **혁신**: 광학 컴퓨팅으로 확산 모델 가속화
- **EGC와의 관계**: 하드웨어 가속으로 EGC의 계산 오버헤드 완화 가능[1]

#### 6.6 응용 맥락에서의 최신 진화

**Medical Image Generation:**

- **Mediffusion (2024)**: 반지도 + 의료 영상 생성[11]
- **RadGazeGen (2024)**: 방사선 전문의 시선 + 확산 모델[16]
- **EGC와의 차이**: 이들은 특화된 응용, EGC는 범용 기반 모델[1]

**Out-of-Distribution Detection:**

- **Energy-Based Out-of-Distribution Detection (Liu et al., 2021)**: EBM의 에너지를 OOD 스코어로 사용
- **EGC와의 시너지**: EGC의 에너지 함수가 더 강건한 OOD 탐지 가능[1]

***

### 7. 비교 종합 표

| 항목 | JEM (2020) | SADA-JEM (2023) | EGC (2023) | Mediffusion (2024) |
|------|----------|--------------|---------|-----------------|
| **접근** | EBM 재해석 | JEM + SAM | EBM + 확산 | 확산 + 반지도 |
| **분류 정확도 (CIFAR-10)** | 92.9% | 95.5% | **95.9%** | N/A (의료) |
| **FID (CIFAR-10)** | 38.4 | 9.41 | **3.30** | N/A |
| **훈련 안정성** | 낮음 | 중간 | 높음 | 높음 |
| **적대적 견고성** | ✓ | ✓ | **✓✓** | ✓ |
| **멀티태스크** | ✓ | ✓ | **✓✓** | ✓ (특화) |
| **이론적 신향성** | 높음 | 중간 | 높음 | 중간 |
| **실무 적용** | 보편적 | 보편적 | **보편적** | 의료 특화 |

***

### 결론

**EGC의 위치 및 의의:**

EGC는 에너지 기반 모델과 확산 프로세스를 결합하여 **단일 신경망으로 이미지 분류와 생성을 동시에 우수한 성능으로 달성**한 획기적 모델입니다. JEM의 훈련 불안정성을 Fisher 발산으로 극복하고, 확산 모델의 안정성을 에너지 기반 접근의 유연성과 결합했습니다.

**향후 개선 방향:**

1. **확장성**: 더 강력한 데이터 증강 및 아키텍처 특화로 ImageNet 성능 향상
2. **효율성**: 고속 샘플러 도입으로 계산 복잡도 감소
3. **응용화**: 의료, 자율주행 등 도메인별 최적화
4. **이론화**: Fisher 발산의 최적성 증명 및 수렴 보장

EGC는 생성적 사전 학습과 판별적 파인튜닝의 전통적 경계를 허물며, **통합 학습 패러다임의 새로운 가능성**을 열었습니다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/098282fd-0733-4fb1-8cec-8422ffe522fd/2304.02012v3.pdf)
[2](https://link.springer.com/10.1007/978-3-031-53302-0_5)
[3](https://arxiv.org/abs/1912.03263)
[4](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Towards_Bridging_the_Performance_Gaps_of_Joint_Energy-Based_Models_CVPR_2023_paper.pdf)
[5](https://arxiv.org/pdf/2303.04343.pdf)
[6](https://arxiv.org/abs/2307.08702)
[7](https://arxiv.org/abs/2304.08466)
[8](http://arxiv.org/pdf/2302.02070.pdf)
[9](https://www.ijcai.org/proceedings/2024/0082.pdf)
[10](https://arxiv.org/html/2408.17046)
[11](https://arxiv.org/abs/2411.09434)
[12](https://yang-song.net/blog/2021/score/)
[13](https://research.nvidia.com/labs/toronto-ai/CLD-SGM/)
[14](https://arxiv.org/abs/2401.05252)
[15](https://proceedings.neurips.cc/paper_files/paper/2024/file/6cb81234ab47027e991728ed7dd76735-Paper-Conference.pdf)
[16](https://arxiv.org/abs/2410.00307)
[17](https://arxiv.org/abs/2410.00712)
[18](https://arxiv.org/abs/2303.11477)
[19](https://ieeexplore.ieee.org/document/10234379/)
[20](https://ieeexplore.ieee.org/document/10657216/)
[21](https://arxiv.org/abs/2412.05694)
[22](https://arxiv.org/pdf/2305.15316.pdf)
[23](https://arxiv.org/pdf/2412.09063.pdf)
[24](http://arxiv.org/pdf/2403.16379.pdf)
[25](http://arxiv.org/pdf/2408.08502.pdf)
[26](https://arxiv.org/pdf/2208.08664.pdf)
[27](https://arxiv.org/pdf/2211.01324.pdf)
[28](https://proceedings.neurips.cc/paper_files/paper/2024/file/90812824c8b36622e6f61803d03b2926-Paper-Conference.pdf)
[29](https://openaccess.thecvf.com/content/CVPR2024/papers/Graikos_Learned_Representation-Guided_Diffusion_Models_for_Large-Image_Generation_CVPR_2024_paper.pdf)
[30](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1605706/full)
[31](https://scorebasedgenerativemodeling.github.io)
[32](https://academic.oup.com/nsr/article/11/12/nwae348/7810289)
[33](https://atcold.github.io/NYU-DLSP20/en/week07/07-1/)
[34](https://www.lgresearch.ai/blog/view?seq=405)
[35](https://arxiv.org/html/2505.06890v1)
[36](https://arxiv.org/abs/2302.02070)
[37](https://arxiv.org/html/2504.16262v1)
[38](https://arxiv.org/html/2510.21887v1)
[39](https://arxiv.org/html/2510.12311v1)
[40](https://arxiv.org/html/2505.03432v2)
[41](https://arxiv.org/abs/2312.07330)
[42](https://arxiv.org/html/2504.10612v5)
[43](https://arxiv.org/abs/2310.07051)
[44](https://arxiv.org/html/2502.19716v1)
[45](https://www.sciencedirect.com/science/article/abs/pii/S1361841523001068)
[46](https://onlinelibrary.wiley.com/doi/10.1002/suco.202000213)
[47](https://ieeexplore.ieee.org/document/9200457/)
[48](https://www.semanticscholar.org/paper/96dbf4694f62feb411397bea8809229337389516)
[49](https://ashpublications.org/blood/article/136/Supplement%201/29/471938/A-Radiomic-Machine-Learning-Model-to-Predict)
[50](https://academic.oup.com/jbmr/article/35/11/2091/7516780)
[51](https://publikationen.bibliothek.kit.edu/1000122364)
[52](https://www.semanticscholar.org/paper/cbef2149379f0be31c8d8d7afe0456aa9a4dd76b)
[53](https://www.mdpi.com/1996-1073/13/7/1798)
[54](https://dl.acm.org/doi/10.1145/3390525.3390537)
[55](https://www.semanticscholar.org/paper/f7fc98ae45b90043ec55fca305031c9299663ed7)
[56](http://arxiv.org/pdf/2209.07959.pdf)
[57](https://arxiv.org/pdf/2303.04187.pdf)
[58](https://arxiv.org/abs/2108.04227)
[59](https://arxiv.org/html/2407.06315)
[60](http://arxiv.org/pdf/2406.12391.pdf)
[61](https://www.iieta.org/download/file/fid/122782)
[62](https://openreview.net/pdf/018bc1b4985ae60f98e1a7ab4de18e66d2830ad5.pdf)
[63](https://ieeexplore.ieee.org/document/10140940/)
[64](https://velog.io/@ma-kjh/ICLR-2020Your-Classifier-is-Secretly-An-Energy-Based-Model-And-You-Should-Treat-It-Like-One)
[65](https://openaccess.thecvf.com/content/WACV2025/papers/Asakura_Diffusion-Based_Generative_Regularization_for_Supervised_Discriminative_Learning_WACV_2025_paper.pdf)
[66](https://www.sciencedirect.com/science/article/pii/S2215098625000242)
[67](https://github.com/divymurli/ML_Reprod_HybridEnergyModels)
[68](https://arxiv.org/html/2503.20853v1)
[69](https://arxiv.org/pdf/2105.03826.pdf)
[70](https://arxiv.org/pdf/2209.07959.pdf)
[71](https://arxiv.org/html/2511.08416v1)
[72](https://openaccess.thecvf.com/content/ICCV2023/papers/Guo_EGC_Image_Generation_and_Classification_via_a_Diffusion_Energy-Based_Model_ICCV_2023_paper.pdf)
[73](https://arxiv.org/html/2505.10999v1)
[74](https://arxiv.org/html/2509.13353v1)
[75](https://arxiv.org/html/2505.22486v1)
[76](http://www.gatsby.ucl.ac.uk/~balaji/udl2020/accepted-papers/UDL2020-paper-105.pdf)
[77](https://www.sabrepc.com/blog/deep-learning-and-ai/image-classification-models-transformers-cnns-and-hybrid)
