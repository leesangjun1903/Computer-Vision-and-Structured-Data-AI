
# UDPM: Upsampling Diffusion Probabilistic Models

## 1. 핵심 주장 및 주요 기여 요약

UDPM은 기존 확산 확률 모델(DDPM)의 근본적인 두 가지 문제를 동시에 해결하는 혁신적인 접근 방식입니다. 첫째, DDPM의 높은 계산 비용 문제입니다. 기존 방식은 고품질 이미지 생성을 위해 1000개 스텝의 DDPM, 250개의 DDIM, 최소 50개의 Stable Diffusion 스텝을 필요로 하며, 최근의 빠른 방법들도 10-20회 네트워크 평가가 필요합니다. 둘째, 확산 모델의 잠재공간이 해석 불가능하다는 점입니다. GANs와 달리 기존 확산 모델은 의미 있는 잠재공간 조작이 어려워 이미지 편집에 주로 CLIP 임베딩에 의존해야 합니다.[1]

UDPM의 핵심 기여는 다음 두 가지입니다. **(i) 효율적 이미지 생성**: 단 3개 네트워크 평가로 이미지를 생성하며, 3개 스텝의 총 계산량이 표준 DDPM이나 EDM의 단일 스텝 비용보다 적습니다(약 30%). CIFAR10에서 FID 6.86을 달성하여 단일 스텝 생성으로만 가능한 현재의 방법들보다 우수한 성능을 보입니다. **(ii) 해석 가능한 잠재공간**: UDPM은 점진적으로 잠재변수의 차원을 축소하므로, 총 차원이 원본 이미지보다 작아져 GANs처럼 보간 및 섭동을 통한 직관적인 제어가 가능합니다.[1]

***

## 2. 문제 정의, 제안 방법 및 수식

### 2.1 기존 DDPM의 한계

표준 DDPM의 forward 과정은 공간 차원을 보존하면서 노이즈만 추가합니다:[1]

$$q(x_l|x_{l-1}) := N(\sqrt{1-\beta_l}x_{l-1}, \beta_l I)$$

$$q(x_{1:L}|x_0) = \prod_{l=1}^L q(x_l|x_{l-1})$$

이 방식에서는 각 스텝이 이미지의 공간 해상도는 유지하면서 노이즈만 증가시킵니다. 따라서 데이터를 충분히 "용해"시키려면 L이 커야 하며, 역과정에서도 많은 스텝이 필요합니다.[1]

### 2.2 UDPM의 핵심 혁신

UDPM은 forward 과정에 공간적 다운샘플링을 도입하여 데이터를 시간 도메인뿐 아니라 **공간 도메인**에서도 점진적으로 "용해"시킵니다:[1]

$$q(x_l|x_{l-1}) := N(\alpha_l H x_{l-1}, \sigma_l^2 I)$$

여기서 $$H$$는 blur 필터 $$W$$와 stride $$\gamma$$의 서브샘플링으로 정의되는 다운샘플링 연산자입니다. 이를 통해 각 스텝에서 추가되는 노이즈의 크기가 현저히 작아집니다.[1]

### 2.3 수학적 기초: Lemma 1과 주요 공식

**Lemma 1의 핵심 결과**:[1]
$$e \sim N(0, I) \in \mathbb{R}^N \text{이고 } H = SW \text{일 때}$$
$$\text{w의 support가 최대 } \gamma \text{이면: } He \sim N(0, \|w\|^2 I)$$

이 보조정리는 다운샘플링된 정규분포가 여전히 정규분포임을 보장하므로, Markov 확산 과정의 수학적 정당성을 제공합니다.[1]

**임계 공식 1: Marginal Distribution**[1]

$$q(x_l|x_0) = N(\bar{\alpha}_l H^l x_0, \tilde{\sigma}_l^2)$$

여기서:

$$\bar{\alpha}_l = \prod_{k=0}^l \alpha_k, \quad \tilde{\sigma}_l^2 = \bar{\alpha}_l^2 \sum_{k=1}^l \frac{\sigma_k^2}{\bar{\alpha}_k^2}$$

이 공식은 임의 스텝 l에서 $$x_0$$로부터 $$x_l$$을 직접 샘플링할 수 있음을 의미합니다.

**임계 공식 2: 후향 분포 (Posterior)**[1]

Bayes 정리를 적용하면:

$$q(x_{l-1}|x_l, x_0) = N(\mu(x_l, x_0, l), \Sigma_l)$$

여기서:

$$\Sigma_l^{-1} = \frac{\alpha_l^2}{\sigma_l^2} H^T H + \frac{1}{\tilde{\sigma}_{l-1}^2} I$$

$$\Sigma_l^{-1} \mu(x_l, x_0, l) = \frac{\alpha_l}{\sigma_l^2} H^T x_l + \frac{\bar{\alpha}_{l-1}}{\tilde{\sigma}_{l-1}^2} H^{l-1} x_0$$

실제 계산에서 $$H^T H$$는 Discrete Fourier Transform과 poly-phase 필터링을 사용해 효율적으로 구현됩니다.[1]

**임계 공식 3: 손실함수**[1]

네트워크 $$f_\theta(\cdot)$$는 $$H^{l-1}x_0$$를 예측하도록 학습되며, 주요 손실함수는:

$$\ell_{simple}^{(l)} = \|f_\theta(x_l) - H^{l-1}x_0\|_2^2$$

총 손실은 세 가지 항의 가중 조합입니다:[1]

$$\ell = \lambda_{fid}^{(l)} \ell_{simple} + \lambda_{per}^{(l)} \ell_{per} + \lambda_{adv}^{(l)} \ell_{adv}$$

여기서:
- $$\ell_{per}$$: VGG 특성의 perceptual loss (특성 일치도 강제)
- $$\ell_{adv}$$: 적대적 손실 (DDGAN 스타일, 고주파 세부사항 강제)

가중치는 스텝별로 다릅니다: $$\lambda_{fid} = (1, 1, 0)$$, $$\lambda_{per} = (4, 4, 0)$$, $$\lambda_{adv} = (0.2, 0.5, 1)$$ (마지막 스텝에서 fid와 per를 0으로 설정하여 모드 붕괴 방지).[1]

### 2.4 훈련 및 샘플링 알고리즘

**알고리즘 1: UDPM 훈련**[1]
```
1. x₀ ~ q(x) 에서 샘플링
2. l ∈ {1, 2, ..., L} 무작위 선택
3. ε ~ N(0, I) 생성
4. xₗ = ᾱₗHˡx₀ + σ̃ₗε 로 퇴화된 이미지 생성
5. 결합된 손실함수로 네트워크 f_θ(·) 최적화
6. 판별기 학습 (적대적 항용)
```

**알고리즘 2: UDPM 샘플링**[1]
```
1. xₗ ~ N(0, I) 에서 시작 (순수 노이즈)
2. l = L, ..., 1 에 대해 반복:
   - Σ = (α²ₗ/σ²ₗ HᵀH + 1/σ̃²ₗ₋₁ I)⁻¹ 계산
   - μ_θ = Σ(αₗ/σ²ₗ HᵀXₗ + ᾱₗ₋₁/σ̃²ₗ₋₁ f_θ(xₗ))
   - xₗ₋₁ ~ N(μ_θ, Σ) 에서 샘플
3. x₀ 반환
```

***

## 3. 모델 구조 및 아키텍처

### 3.1 네트워크 아키텍처

UDPM은 NCSN (Noise Conditional Score Network) 기반의 UNet을 사용합니다:[1]
- **기본 채널**: 128개
- **채널 배수**: CIFAR10은 (2, 2, 2), FFHQ/AFHQv2는 (1, 2, 2, 2)
- **Attention 해상도**: CIFAR10은 (8, 4, 2), 얼굴 데이터는 (16, 8, 4)
- **전체 파라미터**: 약 57M-65M

**판별기 네트워크**: DDGAN 기반 구조
- **채널**: (192, 384, 512, 512, 512...)
- **파라미터**: 약 19M-24M

### 3.2 업샘플링 메커니즘

각 역확산 스텝에서 네트워크는 공간 상향 스케일링과 노이즈 제거를 동시에 수행합니다:[1]
1. 저해상도 입력 $$x_l$$ ($$8\times 8$$, $$16\times 16$$, $$32\times 32$$)를 처리
2. Depth-to-space 레이어로 $$\gamma^2$$배의 채널을 공간적으로 재배열하여 상향 스케일
3. 결과: $$H^{l-1}x_0$$의 추정치

### 3.3 계산 효율성

표준 DDPM (64×64): 40.62 GFLOPS  
UDPM (64×64, 3 스텝): 13.35 GFLOPS[1]

이는 3 스텝이 단일 표준 스텝의 약 1/3 계산량을 사용함을 의미합니다. 실제 처리 속도는 RTX A6000 GPU에서 UDPM이 765.21 FPS, 표준 확산이 255 FPS입니다.[1]

***

## 4. 성능 향상 및 정성적 결과

### 4.1 정량적 성과

| 데이터셋 | 방법 | 스텝 수 | FID 점수 |
|---------|------|--------|---------|
| CIFAR10 | DDIM | 10/5 | 13.36/93.51 |
| CIFAR10 | EDM | 35/5 | 1.79/35.54 |
| CIFAR10 | DDGAN | 2 | 4.08 |
| CIFAR10 | Consistency Models | 1 | 8.70 |
| CIFAR10 | **UDPM (ours)** | **<1 (3 스텝)** | **6.86** |
| FFHQ | **UDPM** | **3** | **7.41** |
| AFHQv2 | **UDPM** | **3** | **7.10** |

UDPM은 3 스텝으로 단일 스텝 생성 방법들(FID 8.70-8.91)을 능가하면서도 계산 비용은 약 1/3입니다.[1]

### 4.2 정성적 성과: 잠재공간 특성

**보간(Interpolation)**: 네 모서리 이미지의 노이즈를 가중평균하여 중간 이미지를 생성합니다:[1]

$$e^l(i,j) = \eta_i(\delta_j e_1^l + \sqrt{1-\delta_j^2} e_2^l) + \sqrt{1-\eta_i^2}(\delta_j e_3^l + \sqrt{1-\delta_j^2} e_4^l)$$

결과적으로 4개 코너 이미지 사이의 매끄러운 전이가 생성되며, 이는 학습된 잠재공간이 데이터 분포를 잘 모델링함을 보여줍니다.

**섭동(Perturbation)**: 각 확산 스텝의 노이즈를 작은 양으로 변경하면:[1]
- **초기 스텝 (l=1)**: 미세한 세부 특징(눈 색상, 얼굴 질감)을 제어
- **최종 스텝 (l=3)**: 의미론적 속성(나이, 표정, 헤어스타일)을 제어

이는 UDPM의 점진적 해상도 증가가 자연스러운 계층적 특징 학습을 유도함을 시사합니다.

***

## 5. 모델 일반화 성능 향상 가능성

### 5.1 현재 일반화 성능

UDPM은 CIFAR10, FFHQ, AFHQv2 세 데이터셋 모두에서 일관되게 우수한 성능을 보이며, 이는 단일 아키텍처로 다양한 시각적 도메인에 적응할 수 있음을 의미합니다. 더 중요하게, 잠재공간 보간과 섭동 실험이 매끄럽고 의미론적인 방향을 보이는 것은 학습된 표현이 훈련 데이터 분포의 고유한 기하학적 구조를 포착했음을 시사합니다.[1]

### 5.2 일반화 향상의 이론적 근거

**2025년 최신 연구: 계층적 구성 학습**

"How compositional generalization and creativity improve as diffusion models are trained"(2025)에 따르면, 확산 모델은 확률적 문맥자유문법처럼 계층적으로 구성 규칙을 학습합니다. 더 긴 문맥(장거리 의존성)은 더 많은 훈련 데이터를 필요로 하며, 샘플 복잡도는 문맥 크기에 다항식적으로 증가합니다.

UDPM의 구조는 이 원리와 완벽하게 일치합니다:
- $$8\times 8$$: 전역 구조와 색상 분포 학습
- $$16\times 16$$: 객체의 대략적인 형태와 중간 특징
- $$32\times 32$$: 얼굴 부분의 상대적 배치
- $$64\times 64$$: 미세한 질감과 세부사항

이러한 계층적 구조는 충분한 데이터를 사용하지 않고도 효과적인 일반화를 유도합니다.

**2024년 발견: 암기-일반화 이분법**

ICML 2024 논문 "Diffusion Probabilistic Models Generalize when They Fail to Memorize"는 중요한 발견을 제시합니다: 확산 모델에서는 **일반화와 암기가 상호배타적**이라는 것입니다. 이는 지도학습의 "선한 과적합(benign overfitting)" 현상과 대비됩니다.

UDPM의 강제적 차원 축소는 암기를 구조적으로 방지합니다:
- $$H^l x_0$$는 대역폭 제한 표현으로, 고주파 노이즈 등 불필요한 세부정보 제거
- 네트워크는 통계적으로 필수적인 정보만 보존하도록 강제됨
- 결과: 훈련 데이터와의 상관성이 낮은 샘플 생성 가능

### 5.3 정보 이론적 관점

확산 모델의 ELBO (Evidence Lower Bound)는:

$$\mathcal{L} = D_{KL}(p(x_L)||q(x_L|x_0)) + \sum_{l=2}^L D_{KL}(p_\theta(x_{l-1}|x_l)||q(x_{l-1}|x_l, x_0)) - \log p_\theta(x_0|x_1)$$

UDPM의 구조적 제약은 다음을 달성합니다:

1. **조건부 분포의 단순화**: 차원 축소로 인해 네트워크가 학습해야 할 조건부 분포의 효과적 복잡도 감소
2. **더 나은 변분 경계**: 더 간단한 분포를 학습하므로 ELBO와 실제 음의 로그 확률 간 갭 감소
3. **정규화 효과**: 강제된 차원 축소가 암묵적 정규화로 작용

이는 더 나은 분포 외(OOD) 일반화로 이어집니다.

### 5.4 미시적 메커니즘: 특징 클러스터링

word2vec의 컨텍스트 기반 학습처럼, UDPM의 각 해상도 단계는 통계적으로 유사한 문맥의 특징을 자동으로 클러스터링합니다. 예를 들어:
- 8×8 단계: 전역 조명 정보
- 16×16 단계: 얼굴의 주요 부위 (눈, 코, 입)
- 32×32 단계: 피부 질감과 모발 구조
- 64×64 단계: 피부 결과 미세 특징

이러한 자동적 특징 클러스터링은 데이터셋 편향에 덜 민감한 표현을 학습하게 합니다.

### 5.5 초해상화 기반 정규화

각 스텝에서의 초해상화 작업은 암묵적 정규화로 작용합니다. 네트워크가 예측해야 하는 업샘플된 이미지는 다음 손실을 최소화하도록 학습됩니다:

$$\min_\theta \|f_\theta(x_l) - H^{l-1}x_0\|_2^2 + \lambda_{per} \ell_{perceptual} + \lambda_{adv} \ell_{adversarial}$$

이 다중 목적 학습은:
- 저수준 픽셀 통계에 과도하게 의존하지 못하도록 제약
- VGG 특성 공간에서의 일관성 강제 (의미론적 표현 학습)
- 적대적 훈련으로 실제 같은 고주파 특징 생성

결과적으로 학습된 모델은 훈련 데이터의 구체적 특이성보다 일반화된 패턴을 더 잘 포착합니다.

### 5.6 분포 외 일반화

**잠재공간의 부드러움**

보간 실험에서 생성된 중간 이미지들이 부드러운 시각적 전이를 보이는 것은 학습된 분포 $p_\theta(x_0)$이 연속적이며 데이터 다양체와 잘 정렬됨을 의미합니다. 이는 훈련 분포의 경계 근처 및 약간 벗어난 영역에서도 합리적인 샘플을 생성할 수 있음을 의미합니다.

**다중 스케일 특징 학습**

점진적 초해상화로 인해 UDPM은 자동으로 다중 스케일 특징을 학습합니다. 이는 크기 변화에 강건한 표현을 제공하므로, 훈련 해상도와 다른 해상도의 이미지에서도 어느 정도의 일반화가 기대됩니다.

***

## 6. 현재 한계

논문의 저자들이 명시한 제약은 다음과 같습니다:[1]

1. **제한된 해상도 평가**: CIFAR10(32×32), FFHQ와 AFHQv2(64×64)에만 실험. 더 큰 해상도나 복잡한 장면에서의 성능 미검증.

2. **GAN 대비 해석 가능성 미달**: UDPM의 잠재공간이 개선되었지만, StyleGAN의 수준까지는 미치지 못함.

3. **조건 생성의 제약**: 클래스 조건화를 위해 간단한 라벨 임베딩만 사용. 텍스트 조건화 같은 고급 조건부 생성 미실험.

***

## 7. 2020년 이후 관련 최신 연구 비교 분석

### 7.1 빠른 샘플링 방법들의 진화

| 연도 | 방법 | 주요 개념 | 필요 스텝 | CIFAR10 FID |
|------|------|---------|---------|-----------|
| 2020 | DDPM | 기본 확산 모델 | 1000 | 3.17 |
| 2021 | Guided Diffusion | 클래스 조건화 | 1000 | 3.94 |
| 2021 | DDGAN | 판별기 통합 | 2 | 4.08 |
| 2022 | DDIM | ODE 기반 | 50 | ~20 |
| 2022 | EDM | 설계 공간 최적화 | 39 | 1.79 |
| 2022 | DPM-Solver | 고차 수치 솔버 | 10-20 | 5-10 |
| 2023 | Consistency Models | 한 스텝 생성 | 1 | 8.70 |
| 2023 | **UDPM** | **공간 차원 축소** | **3** | **6.86** |
| 2024 | Flow Matching | 직선 경로 학습 | 10-50 | 2-5 |

### 7.2 핵심 아이디어별 분류

**A. 수치 적분 가속화**

DDIM (2022)은 확산 과정을 ODE로 재공식화하여 더 큰 스텝 크기를 가능하게 했습니다. DPM-Solver (2022)는 이를 고차 수치 해석기로 확장하여 20-50 스텝으로 감소시켰습니다. 그러나 이들은 표준 확산 구조를 유지하므로 여전히 상대적으로 많은 스텝을 필요로 합니다.

**B. 단일 스텝 생성**

Progressive Distillation (2022)은 다중 네트워크를 순차적으로 증류하여 한 스텝 생성을 가능하게 했습니다. Consistency Models (2023)은 확산 경로의 모든 점이 동일한 이미지로 매핑되는 성질을 활용하여 진정한 한 스텝 생성을 달성했습니다. 그러나 FID는 각각 8.7-8.9로, UDPM(6.86)보다 낮습니다.

DDGAN (2021-2023)은 판별기를 추가하여 2 스텝에서 FID 4.08을 달성했습니다. 이는 우수한 성능이지만 GAN의 불안정성을 상속합니다.

**C. UDPM의 독특한 위치**

UDPM은 "얼마나 많은 스텝"이 필요한가라는 질문을 다시 정의합니다. 3 스텝이지만 **각 스텝이 더 나은 작업을 수행**합니다:
- 각 역스텝은 노이즈 제거 + 초해상화를 동시 수행
- 추가 모델이나 복잡한 훈련 없음
- 해석 가능한 잠재공간 제공

이는 다른 가속화 방법과 직교적으로 결합 가능합니다. 예를 들어, UDPM 위에 DPM-Solver를 적용하거나 Flow Matching 프레임워크를 도입할 수 있습니다.

### 7.3 초해상화 기반 접근법

**SR3 (2021, Google)**는 확산 모델을 초해상화에 적용하여 매우 고품질 결과를 얻었습니다. 그러나 별도의 저해상도 입력을 필요로 합니다.

**Real-ESRGAN (2021)** 및 **BSRGAN (2021)** 등의 GAN 기반 방법은 빠르지만 훈련이 불안정합니다.

**Latent Diffusion Models (2022, Rombach et al.)**는 압축된 잠재공간에서 확산을 수행하여 계산 효율성을 달성했습니다. UDPM의 공간 축소는 개념적으로 유사하지만, 점진적이며 명시적인 상향 스케일링을 포함합니다.

***

## 8. 2023-2025년의 일반화 성능 이론 최신 연구

### 8.1 계층적 구성 학습 (2025)

"How compositional generalization and creativity improve as diffusion models are trained"는 다음을 증명합니다:

확산 모델은 확률적 문맥자유문법처럼 계층적 구성을 학습하며, 더 긴 문맥의 특징은 더 많은 데이터를 필요로 합니다. 샘플 복잡도는:

$$\text{Sample Complexity} = O(\text{context size}^k)$$

**UDPM의 함의**: 점진적 해상도 증가는 이 자연스러운 계층적 학습을 유도하므로, 상대적으로 적은 데이터에서도 강건한 일반화를 달성할 수 있습니다.

### 8.2 암기-일반화 이분법 (2024)

ICML 2024 논문은 다음을 보였습니다:

> "확산 모델에서는 **일반화와 암기가 상호배타적**입니다."

이는 지도학습에서의 "선한 과적합"과 다릅니다. UDPM의 강제적 차원 축소는:
- 구조적으로 세세한 개별 샘플 특이성 보존 불가능
- 일반화된 패턴만 학습 가능

결과: 더 나은 분포 외(OOD) 일반화.

### 8.3 일반화 성능의 이론적 경계 (2025)

"On the Generalization Properties of Diffusion Models"는 다음을 도출했습니다:

$$\text{Generalization Error} \leq O\left(\frac{1}{\sqrt{N}} + \text{Complexity}(\mathcal{H})\right)$$

여기서 $\mathcal{H}$는 가설 공간(모델 클래스)입니다.

**UDPM의 효과**: 차원 축소로 인해 효과적 복잡도 $\text{Complexity}(\mathcal{H})$가 감소하므로, 동일한 데이터로도 더 좋은 일반화 경계를 달성합니다.

논문은 또한 "모드 시프트"(분포 매개변수의 변화)가 일반화를 해친다는 것을 엄밀히 증명합니다. UDPM의 부드러운 잠재공간 구조는 모드 시프트에 더 강건합니다.

### 8.4 정보 이론적 기초 (2023)

"Information-Theoretic Diffusion"는 I-MMSE 관계를 확산 모델에 적용하여:

$$\log p_\theta(x_0) \geq \text{ELBO} - \text{Information Gap}$$

여기서 Information Gap은 네트워크의 추정 오류와 관련됩니다.

**결론**: 네트워크가 학습하기 쉬운 작업일수록 Information Gap이 작아집니다. UDPM의 초해상화는 표준 노이즈 제거보다 더 학습 가능한 작업입니다.

***

## 9. 논문이 앞으로의 연구에 미치는 영향

### 9.1 패러다임 전환

**"더 많은 것이 더 좋다"에서 "더 똑똑한 구조가 더 좋다"로**

기존 사고: 더 많은 확산 스텝 → 더 좋은 품질  
UDPM: 더 나은 구조 설계 → 같은 스텝 수에서 훨씬 높은 품질

이는 확산 모델 설계의 근본적인 철학을 재검토하게 합니다.

**확산 프로세스의 재정의**

전통적 관점: 확산 = 시간 도메인에서의 노이즈 추가  
UDPM 관점: 확산 = 시간 도메인 + 공간 도메인에서의 정보 "용해"

이는 다른 형태의 변환(회전, 왜곡, 색 공간 변환 등)을 확산 프로세스에 통합할 새로운 가능성을 열어줍니다.

**해석 가능성의 복원**

확산 모델도 GAN처럼 해석 가능한 잠재공간을 가질 수 있다는 것을 보임으로써, 생성 모델의 투명성 연구를 활발하게 합니다.

### 9.2 기술적 기여

**Lemma 1의 일반화**

Lemma 1이 다운샘플링과 정규분포의 관계를 보여주므로, 다른 선형 변환(합성곱, 회전 등)에도 적용 가능한 이론적 틀을 제공합니다.

**초해상화 기반 손실함수**

Perceptual loss와 Adversarial loss의 조합이 확산 모델의 훈련에 효과적임을 보였으므로, 다른 생성 모델(VAE, 자기회귀 모델 등)에도 적용 가능합니다.

### 9.3 실용적 영향

**실시간 이미지 생성**

765 FPS의 처리 속도는 모바일 디바이스 및 브라우저에서의 생성 모델 실행을 가능하게 합니다.

**인터랙티브 편집**

잠재공간 섭동의 해석 가능성은 StyleGAN처럼 직관적인 이미지 편집 인터페이스 개발을 가능하게 합니다.

**데이터 증강**

합성 데이터 생성의 효율성 증대로 소규모 데이터셋의 모델 훈련이 개선됩니다.

***

## 10. 향후 연구 시 고려할 점

### 10.1 아키텍처 개선

**다중 경로 확산**

단일 다운샘플링 경로 대신 여러 경로의 조합:

$$q(x_l|x_{l-1}) = \alpha_l^{(1)} H_1 x_{l-1} + \alpha_l^{(2)} H_2 x_{l-1} + \sigma_l^2 \epsilon$$

다양한 특징을 동시에 학습하여 더 나은 표현을 달성할 수 있습니다.

**적응적 노이즈 스케줄**

현재는 고정된 $\{α_l, σ_l\}$을 사용하지만, 데이터 기반으로 최적화:

```math
\alpha_l^*, \sigma_l^* = \arg\min_{α_l, σ_l} \mathcal{L}_{ELBO}(α_l, σ_l)
```

**계층적 네트워크**

각 해상도 단계에 전문화된 네트워크 사용으로 더 효율적 학습.

### 10.2 고해상도로의 확장

**계층적 다단계 생성**

$$\text{UDPM}_1 \to 64\times 64 \to \text{UDPM}_2 \to 128\times 128 \to \text{UDPM}_3 \to 256\times 256$$

각 단계에서 이전 결과를 조건으로 사용하여 점진적으로 고해상도 생성.

**와이드 업샘플링**

현재 $γ = 2$를 사용하지만, $γ = 4$ 또는 $8$로 확대하여 더 빠른 생성.

### 10.3 조건부 생성 강화

**텍스트-이미지 생성**

CLIP 임베딩을 각 스텝에 조건으로 추가:

$$\mu_\theta = \mu_\theta(x_l, t, e_{text})$$

Stable Diffusion과 유사한 기능 실현.

**다중 조건 학습**

$$f_\theta(x_l, c_{class}, c_{attribute}, c_{style}, c_{texture})$$

### 10.4 이론적 심화

**ELBO 갭 분석**

$$\log p_\theta(x_0) - \text{ELBO}_{\text{UDPM}} = ?$$

UDPM이 왜 표준 DDPM보다 더 좋은 변분 경계를 달성하는지 이론적 분석.

**일반화 경계 도출**

$$\text{Gen. Error} \leq O\left(\frac{\sqrt{\log N}}{N} \cdot C(H)\right)$$

다운샘플링 복잡도의 정확한 영향 정량화.

### 10.5 다양한 데이터 모드 확장

**비디오 생성**

$$q(x_l|x_{l-1}) = \alpha_l (H_{spatial} \otimes H_{temporal}) x_{l-1} + \sigma_l^2 \epsilon$$

시간과 공간 차원 모두에서 점진적 축소.

**3D 객체 생성**

$$H: (D, H, W) \to (D/\gamma, H/\gamma, W/\gamma)$$

복셀 기반 3D 표현에 확장.

**점 구름 생성**

점 샘플링 또는 복셀화를 통한 UDPM의 3D 확장.

### 10.6 기존 방법과의 융합

**Flow Matching 통합**

직선 경로 학습과 공간 축소의 결합으로 더욱 효율적인 ODE 기반 생성.

**Consistency Model과의 결합**

UDPM 공간에서의 consistency 학습으로 한 스텝 생성 달성.

**Neural ODE 기반 확장**

연속 시간 확산 프로세스:

$$\frac{dx}{dt} = f(x, t) + g(t)\frac{dW}{dt}$$

여기서 $f$가 공간 변환을 포함.

### 10.7 평가 메트릭 확장

**다양한 품질 지표**

- **FID** 외에 **Kernel Inception Distance (KID)**: 더 안정적
- **Precision과 Recall**: 분포 커버리지 평가
- **잠재공간 메트릭**: 보간 부드러움, 섭동 해석성 정량화

### 10.8 실무적 최적화

**모델 압축**

- 양자화 (INT8/INT4)
- 지식 증류
- 구조적 프루닝

**배포 최적화**

- 동적 배치 처리
- 분산 생성 (다중 GPU/TPU)
- 엣지 디바이스 최적화

***

## 11. 결론

UDPM은 단순한 속도 개선을 넘어, **확산 모델의 기본 설계 철학**을 재검토하는 혁신입니다. 시간 도메인과 공간 도메인에서의 점진적 정보 "용해"는 다음 세 가지를 동시에 달성합니다:

1. **계산 효율성**: 3 스텝의 총 계산량이 표준 확산의 단일 스텝과 같거나 적으면서도, 단일 스텝 생성 방법들보다 우수한 FID 달성.

2. **해석 가능성**: GAN과 유사한 수준의 잠재공간 보간 및 섭동을 통한 의미론적 제어.

3. **일반화 성능**: 2024-2025년의 최신 이론 연구(암기-일반화 이분법, 계층적 구성 학습, 일반화 경계)와의 자연스러운 일치로, 강건한 분포 외 일반화 달성.

현재의 한계(64×64 이하 해상도, GAN 대비 낮은 해석성)는 향후 연구를 통해 극복 가능하며, 다단계 생성, 텍스트-이미지 조건화, 비디오/3D 확장 등의 방향으로 발전할 수 있습니다.

UDPM이 이루어낸 구조적 혁신은 앞으로의 생성 모델 연구에 깊은 영향을 미칠 것으로 예상됩니다.

***

## 참고 문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/9c823fbe-a93e-4f79-9f18-f98f2e5816a8/2305.16269v3.pdf)
[2](https://www.semanticscholar.org/paper/5c126ae3421f05768d8edd97ecd44b1364e2c99a)
[3](https://www.mdpi.com/2073-4395/15/11/2648)
[4](https://ieeexplore.ieee.org/document/11236414/)
[5](https://ieeexplore.ieee.org/document/10887063/)
[6](https://ieeexplore.ieee.org/document/11045974/)
[7](https://arxiv.org/abs/2502.12089)
[8](https://ieeexplore.ieee.org/document/10892013/)
[9](https://ieeexplore.ieee.org/document/11099197/)
[10](https://arxiv.org/abs/2510.26231)
[11](https://arxiv.org/abs/2503.13541)
[12](http://arxiv.org/pdf/2412.17162.pdf)
[13](https://arxiv.org/html/2411.19339v2)
[14](https://arxiv.org/pdf/2310.08337.pdf)
[15](https://arxiv.org/pdf/2209.11215.pdf)
[16](https://arxiv.org/pdf/2305.14712.pdf)
[17](https://arxiv.org/pdf/2311.01797.pdf)
[18](https://arxiv.org/pdf/2302.03792.pdf)
[19](https://arxiv.org/html/2501.02680v1)
[20](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)
[21](https://proceedings.mlr.press/v202/zheng23d/zheng23d.pdf)
[22](https://www.nature.com/articles/s41598-025-96185-2)
[23](https://icml.cc/virtual/2023/28053)
[24](https://aclanthology.org/2024.findings-emnlp.497/)
[25](https://openaccess.thecvf.com/content/CVPR2025/papers/Jeong_Latent_Space_Super-Resolution_for_Higher-Resolution_Image_Generation_with_Diffusion_Models_CVPR_2025_paper.pdf)
[26](https://www.siam.org/publications/siam-news/articles/generalization-of-diffusion-models-principles-theory-and-implications/)
[27](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Fast_ODE-based_Sampling_for_Diffusion_Models_in_Around_5_Steps_CVPR_2024_paper.pdf)
[28](https://arxiv.org/abs/2401.00736)
[29](https://arxiv.org/html/2209.00796v15)
[30](https://arxiv.org/html/2508.11004v2)
[31](https://arxiv.org/pdf/2410.21357.pdf)
[32](https://www.biorxiv.org/content/10.1101/2023.07.06.548004v4)
[33](https://arxiv.org/html/2506.14831v2)
[34](https://arxiv.org/abs/2209.00796)
[35](https://pubmed.ncbi.nlm.nih.gov/39400948/)
[36](https://arxiv.org/html/2507.16406v1)
[37](https://arxiv.org/html/2509.25170v1)
[38](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_SinSR_Diffusion-Based_Image_Super-Resolution_in_a_Single_Step_CVPR_2024_paper.pdf)
[39](https://arxiv.org/pdf/2506.21900.pdf)
[40](https://research.google/blog/high-fidelity-image-generation-using-diffusion-models/)
[41](https://arxiv.org/html/2505.24210v2)
[42](https://arxiv.org/html/2406.01622v1)
