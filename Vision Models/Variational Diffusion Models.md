# Variational Diffusion Models

### 1. 논문의 핵심 주장 및 주요 기여도

**Variational Diffusion Models (VDM)** 논문은 확산 기반의 생성 모델이 비자기회귀(autoregressive) 모델들이 오랫동안 지배해온 이미지 밀도 추정(density estimation) 벤치마크에서 최고 성능을 달성할 수 있음을 입증하는 획기적인 연구입니다.[1]

이 논문의 가장 근본적인 주장은 다음과 같습니다:

1. **확산 모델의 우수성**: CIFAR-10과 ImageNet 64×64 데이터셋에서 상태 최고 성능(SOTA)의 로그 우도값(log-likelihood)을 달성하여, 확산 모델이 생성의 질(perceptual quality)뿐만 아니라 우도 기반 모델링에서도 탁월함을 증명했습니다.[1]

2. **이론적 단순화 및 통찰**: 변분 하한(Variational Lower Bound, VLB)을 신호-대-잡음 비(Signal-to-Noise Ratio, SNR)로 표현하는 간단한 수식으로 변환함으로써, 확산 모델 클래스에 대한 이해를 획기적으로 개선했습니다.[1]

3. **학습 가능한 노이즈 스케줄**: 고정된 노이즈 스케줄을 사용하던 기존 방식과 달리, 이 논문은 노이즈 스케줄을 신경망을 통해 학습 가능하도록 만들어 최적화 효율을 극대화할 수 있음을 보였습니다.[1]

### 2. 해결하고자 하는 문제 및 기존의 한계

**기존 문제점:**

확산 모델은 이미지 생성의 지각적 품질(FID, Inception Score)에서는 뛰어났지만, 객관적인 우도 기반 밀도 추정 벤치마크에서는 PixelCNN++, Image Transformer, Sparse Transformer 등의 자기회귀 모델에 뒤처지고 있었습니다. 이는 확산 모델이 우도 최적화에 특화되지 않았음을 시사했습니다.[1]

### 3. 제안하는 방법론 및 수식

#### 3.1 확산 과정(Forward Diffusion Process)

모델은 데이터 $$x$$에서 시작하여 시간 $$t \in $$에 따라 점진적으로 노이즈가 추가되는 가우시안 확산 과정을 정의합니다:[1]

$$q(z_t|x) = \mathcal{N}(\alpha_t x, \sigma_t^2 I)$$

여기서 $$\alpha_t$$와 $$\sigma_t^2$$는 시간의 함수이며, 신호-대-잡음 비는 다음과 같이 정의됩니다:[1]

$$\text{SNR}(t) = \alpha_t^2 / \sigma_t^2$$

#### 3.2 학습 가능한 노이즈 스케줄

고정 스케줄 대신, 다음과 같이 신경망 매개변수 $$\eta$$로 노이즈 스케줄을 학습합니다:[1]

$$\sigma_t^2 = \text{sigmoid}(\gamma_\eta(t))$$

$$\alpha_t^2 = \text{sigmoid}(-\gamma_\eta(t))$$

$$\text{SNR}(t) = \exp(-\gamma_\eta(t))$$

#### 3.3 변분 하한(Variational Lower Bound)

이산 시간의 경우, 확산 손실은:[1]

$$L_T(x) = \frac{T}{2} \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I), i \sim U\{1,T\}} \left[ (\text{SNR}(s) - \text{SNR}(t)) \|\mathbf{x} - \hat{\mathbf{x}}_\theta(z_t; t)\|^2_2 \right]$$

여기서 $$z_t = \alpha_t \mathbf{x} + \sigma_t \epsilon$$이며, $$\hat{\mathbf{x}}_\theta$$는 노이즈를 제거한 데이터를 예측하는 신경망입니다.[1]

매개변수 대입 후 단순화하면:[1]

$$L_T(x) = \frac{T}{2} \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I), i \sim U\{1,T\}} \left[ (\exp(\gamma_\eta(t) - \gamma_\eta(s)) - 1) \|\epsilon - \hat{\epsilon}_\theta(z_t; t)\|^2_2 \right]$$

#### 3.4 연속 시간 모델($$T \to \infty$$)

$$T$$를 무한대로 취할 때, 확산 손실은:[1]

$$L_\infty(x) = -\frac{1}{2}\mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)} \int_0^1 \text{SNR}'(t) \|\mathbf{x} - \hat{\mathbf{x}}_\theta(z_t; t)\|^2_2 dt$$

또는 매개변수화된 형태로:[1]

$$L_\infty(x) = \frac{1}{2}\mathbb{E}_{\epsilon \sim \mathcal{N}(0,I), t \sim U(0,1)} \left[ \gamma'_\eta(t) \|\epsilon - \hat{\epsilon}_\theta(z_t; t)\|^2_2 \right]$$

#### 3.5 푸리에 특성(Fourier Features)

정밀한 픽셀 수준의 세부사항을 포착하기 위해 다음과 같은 푸리에 특성을 입력에 추가합니다:[1]

$$f_n = \sin(2^n \pi z), \quad g_n = \cos(2^n \pi z), \quad n \in \{n_{\min}, ..., n_{\max}\}$$

이는 고주파 정보를 증폭하여 우도 최적화에 도움이 됩니다.[1]

### 4. 모델 구조

VDM의 구조는 다음 요소들로 구성됩니다:[1]

1. **U-Net 기반 노이즈 예측 모델**: 입력 $$z_t$$에서 노이즈를 예측하는 신경망으로, 원래 해상도에서만 처리됩니다.

2. **시간 임베딩**: 시간 정보는 $$\gamma_t$$ 형태로 인코딩되어 U-Net에 통합됩니다.

3. **학습 가능한 노이즈 스케줄 네트워크**: 3개의 선형층으로 구성된 단조 신경망이 $$\gamma_\eta(t)$$를 학습합니다.

4. **분산 최소화**: 연속 시간 모델에서 VLB 추정치의 분산을 최소화하기 위해 노이즈 스케줄을 최적화합니다.

### 5. 성능 향상 및 실증 결과

#### 5.1 우도 추정 성능

VDM은 주요 벤치마크에서 이전의 모든 모델을 능가했습니다:[1]

- **CIFAR-10 (데이터 증강 없음)**: 2.65 bits/dim (이전 최고: 2.80)
- **CIFAR-10 (데이터 증강 포함)**: 2.49 bits/dim (이전 최고: 2.53)
- **ImageNet 32×32**: 3.72 bits/dim
- **ImageNet 64×64**: 3.40 bits/dim (이전 최고: 3.43)

특히, Sparse Transformer의 이전 최고 성능 2.80 bits/dim을 10배 이상 빠른 훈련 속도로 달성했습니다.[1]

#### 5.2 수렴 향상

학습 가능한 노이즈 스케줄과 푸리에 특성의 조합이 훈련 수렴을 크게 가속화했습니다. 분산 최소화를 통해 VLB 추정 분산이 다음과 같이 감소했습니다:[1]

- 학습된 스케줄: 0.53
- log SNR-linear: 6.35
- β-Linear (Ho et al.): 31.6
- α-Cosine (Nichol & Dhariwal): 31.1

#### 5.3 무손실 압축

비트-백 코딩을 사용한 무손실 압축에서 이론적 최적값에 가까운 성능을 달성했습니다. CIFAR-10 테스트 세트에서 $$T_{\text{eval}} = 1000$$일 때 2.67 bits/dim을 기록했습니다.[1]

### 6. 연속 시간에서의 확산 모델 동치성

VDM의 핵심 이론적 기여 중 하나는 다음의 놀라운 불변성을 증명한 것입니다:[1]

**연속 시간 VLB의 불변성**: 연속 시간 설정에서, VLB는 두 끝점의 SNR 값 $$\text{SNR}(0)$$과 $$\text{SNR}(1)$$을 제외하고 노이즈 스케줄의 구체적인 형태에 대해 불변입니다.

이는 다음을 의미합니다:[1]

$$L_\infty(x) = \frac{1}{2}\mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)} \int_{\text{SNR}_{\min}}^{\text{SNR}_{\max}} \|\mathbf{x} - \tilde{\mathbf{x}}_\theta(z_v, v)\|^2_2 dv$$

따라서 분산 보존(variance-preserving)과 분산 폭발(variance-exploding) 확산 과정 등 다양한 스케줄은 모두 동치적이며, 단순히 시간 의존적 스케일링으로만 다릅니다.

### 7. 모델의 일반화 성능 향상 가능성

#### 7.1 이론적 근거

VDM의 설계는 여러 방식으로 일반화 성능을 개선합니다:

**정규화 메커니즘**: 학습 가능한 노이즈 스케줄을 통해 모델이 효율적으로 훈련 데이터의 분포를 학습할 수 있게 하며, 과적합을 방지합니다. 분산 최소화는 추정 분산을 감소시켜 보다 안정적인 학습을 가능하게 합니다.[1]

**푸리에 특성**: 고주파 정보를 명시적으로 모델링함으로써, 네트워크가 세밀한 픽셀 수준의 패턴을 더 잘 학습하고 일반화할 수 있습니다.[1]

#### 7.2 실증적 증거

실험 결과는 다음을 보여줍니다:[1]

- 더 많은 이산 시간 단계($$T$$)를 사용할수록 VLB가 향상되며, 이는 연속 시간 모델이 이론적으로 최상의 성능을 제공함을 의미합니다.
- 아블레이션 연구에서 학습된 노이즈 스케줄이 없으면 최대 로그-SNR이 약 8에 머물러 우도가 4 bits/dim을 초과했지만, 스케줄 학습으로 로그-SNR이 13.3으로 증가하여 성능이 크게 향상되었습니다.[1]

#### 7.3 아키텍처 개선

VDM은 주의 메커니즘을 대부분 제거하여 과적합 위험을 줄였습니다. 대신 중간의 단일 주의 블록을 유지하여 장거리 의존성을 포착합니다.[1]

### 8. 모델의 한계

#### 8.1 계산 효율성

비트-백 코딩을 통한 무손실 압축에서, 큰 $$T_{\text{eval}}$$에 대해 이론적 우도와 실제 코드길이 사이에 약 0.05 bits/dim의 갭이 발생합니다. 이는 매우 깊은 모델을 위한 비트-백 코딩의 구현 부정확성 때문입니다.[1]

#### 8.2 지각적 품질 vs. 우도

모델이 우도 최적화에 중점을 두기 때문에, 지각적 품질(FID)은 상대적으로 낮습니다. 우도 중심 하이퍼파라미터로 훈련된 CIFAR-10 모델의 FID는 7.41이지만, 이는 최근의 우도 무관 확산 모델들의 FID(약 3-4)보다 높습니다.[1]

#### 8.3 높은 훈련 계산 비용

CIFAR-10에서 최고 성능을 달성하기 위해 8개의 TPUv3 칩에서 9일이 소요되었습니다. ImageNet의 경우는 128개의 칩과 1주 이상이 필요합니다.[1]

#### 8.4 데이터 효율성

모델이 충분히 큰 데이터셋에 의존하므로, 소규모 데이터셋에서의 성능은 명시적으로 평가되지 않았습니다.

### 9. 2020년 이후 관련 최신 연구

#### 9.1 확산 모델의 일반화 성능 이론

**"Generalization in VAE and Diffusion Models: A Unified Information-Theoretic Analysis" (2025)**: Chen et al.은 정보 이론적 도구를 활용하여 확산 모델과 VAE의 일반화 성능에 대한 통일된 이론적 프레임워크를 제시했습니다. 특히 확산 시간 $$T$$에 대한 명시적 거래 관계(trade-off)를 도출했으며, 훈련 데이터만을 사용한 계산 가능한 한계를 제공하여 최적 $$T$$를 선택할 수 있게 합니다.[2]

#### 9.2 노이즈 스케줄 최적화 심화

**"Improved Noise Schedule for Diffusion Training" (2025)**: Hang et al.은 로그 SNR 값이 0 근처에 확률 밀도를 집중시키는 노이즈 스케줄이 훈련 효율을 크게 향상시킨다는 것을 발견했습니다. 라플라스 스케줄이 특히 유효함을 보여주었습니다.[3]

#### 9.3 확산 모델의 생성화 메커니즘

**"Towards a Mechanistic Explanation of Diffusion Model Generalization" (2025)**: 연구자들은 훈련 전 메커니즘으로 확산 모델의 일반화 행동을 설명하고, 다양한 신경망 아키텍처에서 공유된 국소적 귀납 편향(local inductive bias)을 식별했습니다.[4]

**"Generalization of Diffusion Models: Principles, Theory, and Implications" (2024)**: SIAM 논문은 모델 재현성(model reproducibility) 현상을 발견했으며, 이는 확산 모델이 메모리화 체제와 일반화 체제 사이를 전환함을 보여줍니다. 이 현상이 일반화 성능의 핵심 열쇠입니다.[5]

#### 9.4 도메인 일반화 응용

**"Boosting Domain Generalized and Adaptive Detection with Diffusion Models" (2025)**: 확산 모델을 도메인 일반화 및 적응 작업에 적용하여, 추론 시간을 75% 감소시키면서 성능을 향상시켰습니다.[6]

**"What's in a Latent? Leveraging Diffusion Latent Space for Domain Generalization" (2025)**: 확산 모델의 잠재 공간이 명시적 도메인 레이블 없이도 도메인을 분리하는 데 탁월함을 보였으며, 표준 ERM 대비 최대 4% 이상의 성능 향상을 기록했습니다.[7]

#### 9.5 효율성 향상

**"Efficient Diffusion Models: A Comprehensive Survey from Principles to Practices" (2024)**: 확산 모델의 계산 효율성을 개선하기 위한 다양한 기법들을 종합적으로 조사했습니다.[8]

**"Accelerating Convergence of Score-Based Diffusion Models, Provably" (2024)**: DDIM과 DDPM 샘플러를 가속화하기 위한 훈련 전 알고리즘을 설계하여 수렴 속도를 $$O(1/T^2)$$로 개선했습니다.[9]

#### 9.6 저수준 구조 활용

**"Low-dimensional adaptation of diffusion models: Convergence in total variation" (2025)**: 확산 모델이 저차원 구조를 활용하여 샘플링을 가속화하는 방식을 분석했으며, DDIM과 DDPM의 반복 복잡도가 $$O(k/\varepsilon)$$임을 증명했습니다.[10]

#### 9.7 최대 우도 추정 기법

**"Improved Techniques for Maximum Likelihood Estimation for Diffusion Models" (2023)**: Zheng et al.은 확산 확산 과정에 특화된 절삭 정규분포 이산화와 중요도 가중 우도 추정기를 도입했습니다.[11]

#### 9.8 데이터 증강 및 생성

**"Boosting GAN Performance Through Dataset Augmentation with Denoising Diffusion Models" (2025)**: DDPM과 GAN을 결합하여 제한된 훈련 데이터로부터 다양한 이미지를 생성함으로써 CIFAR-10에서 82.96% 성능 향상을 기록했습니다.[12]

#### 9.9 신경망 아키텍처

**"Diffusion Models: A Comprehensive Survey of Methods and Applications" (2022)**: DDPM, DDIM, Score SDE 등을 포함한 확산 모델의 종합적인 개요를 제공했습니다.[13]

### 10. 논문의 영향과 앞으로의 연구 방향

#### 10.1 이론적 영향

1. **확산 과정의 동치성**: VDM의 주요 기여인 연속 시간 동치성 증명은 이후 확산 모델 이론을 단순화하고, 다양한 스케줄 간의 관계를 명확히 했습니다.[1]

2. **SNR 중심의 관점**: 시간 매개변수 대신 SNR을 중심으로 확산 과정을 이해하는 패러다임 전환은 이후 많은 연구에 영향을 미쳤습니다. 최근 연구들이 노이즈 스케줄 최적화에 집중하는 것도 이러한 기초 위에서 비롯됩니다.[3]

#### 10.2 실무적 영향

1. **우도 기반 모델링의 재평가**: VDM은 확산 모델이 단순히 생성 품질뿐만 아니라 우도 기반 평가에서도 경쟁력 있음을 입증하여, 확산 모델의 적용 범위를 확대했습니다.[1]

2. **푸리에 특성의 활용**: 이 논문에서 제안한 푸리에 특성은 이후 다양한 고주파 모델링 작업에 영감을 주었습니다.[1]

#### 10.3 앞으로의 연구 고려사항

#### 10.3.1 노이즈 스케줄 학습의 개선

- **적응적 스케줄**: 데이터셋이나 작업에 따라 동적으로 조정되는 노이즈 스케줄 개발이 필요합니다. 현재의 단조 신경망 기반 방식을 넘어, 더 복잡한 적응 메커니즘을 탐구할 가치가 있습니다.

- **다중 스케일 스케줄**: 이미지의 해상도가 다양한 경우, 각 해상도에 최적화된 노이즈 스케줄을 학습하는 연구가 필요합니다.

#### 10.3.2 일반화 성능의 이론적 심화

- **정규화 메커니즘의 명시화**: 학습 가능한 노이즈 스케줄과 푸리에 특성이 구체적으로 어떻게 정규화 효과를 제공하는지에 대한 더 깊은 이론적 분석이 필요합니다.

- **도메인 외(Out-of-Distribution) 성능**: VDM이 훈련 분포와 크게 다른 테스트 데이터에서 어떻게 일반화하는지에 대한 연구가 필요합니다. 최근 도메인 일반화 연구들이 이 방향의 시작이지만, 더 체계적인 이론적 기초가 필요합니다.[6][7]

#### 10.3.3 계산 효율성과 확장성

- **비트-백 코딩의 개선**: 깊은 모델을 위한 효율적인 비트-백 코딩 구현은 여전히 열려있는 문제입니다. 수치적 정확성을 높이면서도 계산 효율을 유지하는 방법의 개발이 필수적입니다.[1]

- **저차원 구조의 활용**: 최근 연구가 보여준 바와 같이, 확산 모델이 데이터의 저차원 구조를 활용할 수 있다면, 이를 명시적으로 활용하는 아키텍처 설계가 샘플링 속도를 크게 향상시킬 수 있습니다.[10]

#### 10.3.4 다양한 도메인으로의 확장

- **소규모 데이터셋**: VDM은 대규모 데이터셋에서 검증되었지만, 의료 이미징, 위성 이미지 등 데이터가 제한적인 도메인에서의 성능 개선이 필요합니다.

- **비정형 데이터**: 현재 VDM은 주로 이미지에 적용되었지만, 비정형 데이터(시간 시계열, 텍스트, 음성 등)로의 확장 시 노이즈 스케줄과 아키텍처를 어떻게 적응시킬 것인가가 중요한 질문입니다.

#### 10.3.5 조건부 모델링 강화

- **다중 조건**: 텍스트, 이미지, 카테고리 등 다양한 조건을 동시에 처리하는 VDM의 개발이 필요합니다.

- **경쟁하는 조건들 사이의 균형**: 여러 조건이 상충할 때 모델이 어떻게 균형을 맞추는지에 대한 이론적 분석이 필요합니다.

#### 10.3.6 정규화와 일반화의 심화

- **명시적 정규화**: 학습 가능한 노이즈 스케줄이 암묵적으로 정규화 효과를 제공한다면, 이를 명시적으로 제어할 수 있는 메커니즘 개발이 가능합니다.

- **구조적 정규화**: 신경망 아키텍처 자체에서 오는 정규화 효과(주의 메커니즘 제거 등)와 노이즈 스케줄 학습의 상호작용을 더 체계적으로 분석하는 것이 필요합니다.

#### 10.3.7 이론과 실무의 간극 좁히기

- **연속-이산 간극**: 이론적으로는 $$T \to \infty$$일 때 최적이지만, 실제 구현에서는 유한한 $$T$$를 사용합니다. 이 간극을 좁히기 위한 방법론 개발이 필요합니다.

- **분산 추정의 개선**: 현재의 저불일치 샘플러(low-discrepancy sampler)와 분산 최소화 기법을 넘어, 더 효과적인 분산 감소 방법의 개발이 필요합니다.

#### 10.3.8 최신 연구 동향 통합

최근 2024-2025년의 연구들은 다음의 방향들을 강조합니다:

- **정보 이론적 관점**: Chen et al.의 작업은 일반화 성능을 정보 이론으로 이해하는 새로운 틀을 제시했습니다. 이를 VDM의 학습 가능한 노이즈 스케줄과 결합하면, 더 강력한 이론적 기초를 구축할 수 있습니다.[2]

- **메커니즘의 명시화**: Yang et al.의 "inductive bias" 개념은 VDM의 아키텍처 선택(주의 메커니즘 제거 등)이 실제로 일반화를 어떻게 개선하는지 이해하는 데 도움이 됩니다.[4]

- **도메인 적응**: Liu et al.이 보여준 바와 같이, 확산 모델의 잠재 공간은 도메인 정보를 캡슐화하는 능력이 있습니다. VDM의 우도 기반 학습이 이러한 도메인 구조를 어떻게 활용할 수 있는지 탐구하는 것이 흥미로운 방향입니다.[7]

### 결론

**Variational Diffusion Models**은 확산 모델 이론과 실무 모두에 혁신을 가져온 획기적인 논문입니다. SNR 중심의 관점, 학습 가능한 노이즈 스케줄, 푸리에 특성의 조합은 이후 확산 모델 연구의 기초를 형성했습니다. 특히 연속 시간 동치성의 증명은 다양한 확산 과정들이 본질적으로 같음을 보임으로써, 확산 모델을 이해하는 방식을 근본적으로 변화시켰습니다.[1]

2020년 이후의 최신 연구들은 VDM의 기초 위에서 일반화 성능, 계산 효율성, 다양한 도메인 적용 등의 방향으로 진화하고 있습니다. 앞으로의 연구는 이론적 깊이(정보 이론, 메커니즘적 이해)와 실무적 확장성(저차원 구조 활용, 도메인 특화 최적화) 사이의 균형을 맞추면서, 확산 모델을 더욱 강력하고 효율적인 생성 모델로 발전시켜 나갈 것으로 예상됩니다.[5][2][6][4]

***

### 참고 문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c868f21c-abbf-4fe0-99cb-ff54759ef0fb/2107.00630v6.pdf)
[2](https://www.mdpi.com/2076-3417/15/20/11150)
[3](https://openaccess.thecvf.com/content/ICCV2025/papers/Hang_Improved_Noise_Schedule_for_Diffusion_Training_ICCV_2025_paper.pdf)
[4](https://arxiv.org/html/2411.19339v2)
[5](https://www.siam.org/publications/siam-news/articles/generalization-of-diffusion-models-principles-theory-and-implications/)
[6](https://arxiv.org/abs/2506.00849)
[7](http://arxiv.org/pdf/2503.06698.pdf)
[8](http://arxiv.org/pdf/2410.11795.pdf)
[9](https://arxiv.org/pdf/2403.03852.pdf)
[10](https://arxiv.org/pdf/2501.12982.pdf)
[11](https://proceedings.mlr.press/v202/zheng23c/zheng23c.pdf)
[12](https://ieeexplore.ieee.org/document/10986500/)
[13](https://arxiv.org/pdf/2209.00796v8.pdf)
[14](https://arxiv.org/abs/2506.21042)
[15](https://www.ijcai.org/proceedings/2025/50)
[16](https://jurnal.pascabangkinang.ac.id/index.php/jrmi/article/view/201)
[17](https://link.springer.com/10.3758/s13428-025-02819-8)
[18](https://arxiv.org/abs/2510.05976)
[19](https://ieeexplore.ieee.org/document/10972526/)
[20](https://ieeexplore.ieee.org/document/11064424/)
[21](http://arxiv.org/pdf/2412.17162.pdf)
[22](https://arxiv.org/pdf/2311.01797.pdf)
[23](http://arxiv.org/pdf/2405.15020.pdf)
[24](https://arxiv.org/html/2412.00665v1)
[25](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_Versatile_Diffusion_Text_Images_and_Variations_All_in_One_Diffusion_ICCV_2023_paper.pdf)
[26](https://www.themoonlight.io/ko/review/improved-probabilistic-regression-using-diffusion-models)
[27](https://arxiv.org/pdf/2506.00849.pdf)
[28](https://arxiv.org/abs/2510.23606)
[29](https://arxiv.org/abs/2102.09672)
[30](https://academic.oup.com/nsr/article/11/12/nwae348/7810289)
[31](https://letter-night.tistory.com/598)
[32](https://www.nature.com/articles/s41598-024-51400-4)
[33](https://arxiv.org/html/2209.00796v15)
[34](https://www.semanticscholar.org/paper/945a899a93c03eb63be5e3197e318c077473cef9)
[35](https://iopscience.iop.org/article/10.1149/MA2025-031244mtgabs)
[36](https://arxiv.org/html/2306.04848v4)
[37](https://arxiv.org/pdf/2401.02414.pdf)
[38](https://arxiv.org/html/2412.14422)
[39](https://arxiv.org/pdf/2302.04638.pdf)
[40](http://arxiv.org/pdf/2102.09672.pdf)
[41](https://arxiv.org/html/2402.13369)
[42](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
[43](https://openaccess.thecvf.com/content/CVPR2022/papers/Sehwag_Generating_High_Fidelity_Data_From_Low-Density_Regions_Using_Diffusion_Models_CVPR_2022_paper.pdf)
[44](https://arxiv.org/html/2502.04669v1)
[45](https://kookie12.tistory.com/13)
[46](https://arxiv.org/html/2305.03935v4)
[47](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/noise-scheduling/)
[48](https://ostin.tistory.com/110)
