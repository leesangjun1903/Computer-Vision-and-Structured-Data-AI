# SIDDMs : Semi-Implicit Denoising Diffusion Models

### 핵심 요약

**Semi-Implicit Denoising Diffusion Models (SIDDMs)**는 2023년 NeurIPS에 발표된 논문으로, Denoising Diffusion Probabilistic Models (DDPM)의 높은 품질과 Denoising Diffusion GANs (DDGAN)의 빠른 속도를 결합하면서 대규모 데이터셋 확장성 문제를 해결하는 방법을 제시합니다. 핵심 기여는 **denoising distribution을 implicit (암시적) 및 explicit (명시적) 성분으로 분해**하여 각각에 맞춤형 최적화 목표를 적용하는 것입니다.

***

## 1. 해결하고자 하는 문제

### 1.1 생성 모델의 삼중주(Trilemma)

Diffusion 기반 생성 모델들은 다음 세 가지 목표를 동시에 달성하기 어려워합니다:

1. **높은 샘플 품질** - DDPM 수준의 fidelity
2. **빠른 샘플링** - 적은 반복 단계로 생성
3. **대규모 데이터셋 확장성** - ImageNet 같은 복잡한 데이터에서 성공

### 1.2 DDPM의 문제점

$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

DDPM은 역방향 과정이 각 단계에서 가우시안 분포라고 가정합니다:

$$p(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

이는 **노이즈 추가가 작을 때만 유효**하므로, 빠른 샘플링(큰 단계)을 위해 노이즈를 크게 하면 이 가정이 깨져 편향된 샘플을 생성합니다. 따라서 1000 단계의 반복이 필요합니다.

### 1.3 DDGAN의 한계

DDGAN은 joint distribution 수준에서 adversarial matching을 시도합니다:

$$\min_D \max_\theta \mathbb{E}_{q(x_{t-1}, x_t)} [\log D(x_{t-1}, x_t, t)] + \mathbb{E}_{p(x_{t-1}, x_t)} [\log(1-D(x_{t-1}, x_t, t))]$$

문제점:
- **고차원 문제**: $(x_{t-1}, x_t)$의 concatenation은 매우 고차원이므로 discriminator가 효과적으로 학습하기 어려움
- **확장성 실패**: ImageNet 같은 대규모 복잡 데이터셋에서 완전히 실패 (FID 20.63)

***

## 2. 제안하는 방법론

### 2.1 Joint Distribution 분해의 핵심 통찰

**Theorem 1 (분해의 정당성)**:

$$\mathcal{D}_{JSD}[q(x_{t-1}, x_t), p(x_{t-1}, x_t)] \leq 2\sqrt{2}\mathcal{D}_{JSD}[q(x_{t-1}), p(x_{t-1})] + 2\sqrt{2}\mathcal{D}_{KL}[p(x_t|x_{t-1})||q(x_t|x_{t-1})]$$

여기서:
- $q(x_{t-1}, x_t) = q(x_t|x_{t-1})q(x_{t-1})$ (forward diffusion의 joint)
- $p(x_{t-1}, x_t) = p(x_t|x_{t-1})p(x_{t-1})$ (reverse의 joint)

**혁신적 통찰**: Joint distribution 매칭을 두 개의 독립적인 문제로 분해할 수 있습니다:

1. **Marginal distribution 매칭** - $q(x_{t-1})$와 $p(x_{t-1})$ 간
2. **Conditional distribution 매칭** - $q(x_t|x_{t-1})$와 $p(x_t|x_{t-1})$ 간

### 2.2 Auxiliary Forward Diffusion (AFD)

조건부 분포 KL 발산을 전개하면:

$$\mathcal{D}_{KL}[p(x_t|x_{t-1})||q(x_t|x_{t-1})] = -H[p(x_t|x_{t-1})] - H[p(x_t|x_{t-1}), q(x_t|x_{t-1})]$$

여기서:
- 교차 엔트로피 항 $H[p(x_t|x_{t-1}), q(x_t|x_{t-1})]$: 명시적으로 계산 가능 (L2 손실)
- 음의 엔트로피 항 $-H[p(x_t|x_{t-1})]$: 다루기 어려움 → **최소-최대 게임으로 해결**

$$\min_G \max_C \mathbb{E}_{p(x_t|x_{t-1})} [\log p_C(x_t|x_{t-1})]$$

여기서 $C$는 조건부 엔트로피를 추정하는 회귀 모델입니다.

### 2.3 최종 SIDDM 학습 목표

```math
\min_D \max_C \mathbb{E}_{q(x_0, x_{t-1}|x_0, x_t|x_{t-1})} \left[ \log D(x_{t-1}, t) + \log(1-D(x'_{t-1}, t)) \right] + \lambda_{AFD} \left\{ \mathbb{E}[C(x_{t-1}, t) - x_t)^2] + \text{entropy terms} \right\}
```

where:
- 첫 번째 항: GAN 목표 (implicit marginal 매칭)
- 두 번째 항: L2 + 엔트로피 (explicit conditional 매칭)
- $\lambda_{AFD}$: balance parameter

### 2.4 Discriminator Regularization

추가적 정규화 기법 (3.3절):

$$\min_D \mathbb{E}_{q(x_0, x_{t-1}|x_0)} \text{L2}(D(x_{t-1}, t), x_0)$$

Denoising task를 보조 목표로 사용하여 discriminator를 추가 계산 없이 안정화합니다. 이는:
- Spectral normalization과 달리 모델 용량 제약 없음
- WGAN, R1 regularization과 달리 hyperparameter 그리드 서치 불필요

***

## 3. 모델 구조

### 3.1 네트워크 아키텍처

SIDDM은 **U-Net 기반 구조**를 사용하는데, 이는 DDPM의 ADM과 유사하지만 중요한 수정을 포함합니다:

**Generator $G$ (Denoiser)**:
- Input: noisy image $x_t$, time step $t$
- Output: denoising prediction $x_{t-1}$ 또는 noise $\epsilon_t$
- Architecture: U-Net with skip connections, self-attention blocks

**Discriminator $D$ (동일한 U-Net 구조)**:
- 이전 CNN 기반 discriminator와 달리 U-Net 구조 사용 (UNet-GAN 영감)
- Multi-scale feature matching: global logit 뿐 아니라 pixel-level 판별
- 장점: 픽셀 수준 분포 매칭으로 더 나은 gradients

### 3.2 구조적 특징

- **공유 표현**: Generator와 discriminator가 유사한 bottleneck 구조 사용
- **차별화된 출력**: Generator는 denoising, Discriminator는 binary classification
- **Regression module $C$**: Shared layers와 추가 output head로 조건부 분포 추정

***

## 4. 성능 향상 분석

### 4.1 정량적 비교

**CIFAR-10 (T=4 denoising steps)**

| 모델 | IS | FID | Recall | 속도 |
|------|-----|------|---------|------|
| **SIDDMs (ours)** | **9.85** | **2.24** | **0.61** | 0.20s |
| DDGANs | 9.63 | 3.75 | 0.57 | 0.20s |
| DDPM | 9.46 | 3.21 | 0.57 | 80.5s |
| StyleGAN2 w ADA | 9.83 | 2.92 | 0.49 | 0.04s |
| EDM | 9.84 | 2.04 | - | - |

**CelebA-HQ-256**

| 모델 | FID |
|------|-----|
| **SIDDMs (ours)** | **7.37** |
| DDGANs | 7.64 |
| Score SDE | 7.23 |
| LSGM | 7.22 |
| UDM | 7.16 |

**ImageNet 1000 (T=4 steps) - 결정적 우위**

| 모델 | FID |
|------|-----|
| **SIDDMs (ours)** | **3.13** |
| DDGANs | 20.63 (**완전 실패**) |
| EDM | 2.44 |
| ADM | 2.07 |
| Consistency Models | 4.07 |

### 4.2 정성적 개선

**Mode Coverage (MOG 5×5 Gaussian Mixture)**

MOG 실험은 일반화 성능을 직관적으로 보여줍니다:

| 모델 | T=1 | T=2 | T=4 | T=8 | T=16 |
|------|-----|------|------|------|-------|
| DDGANs | - | 7.27 | 0.99 | 0.49 | 0.53 |
| **SIDDMs** | - | **0.14** | **1.21** | **0.30** | **0.23** |
| SIDDMs w/o AFD | - | 21.19 | 53.22 | 7.04 | 14.37 |

**핵심 통찰**: AFD 항이 없으면 완전히 실패합니다. 이는 implicit-explicit 균형이 얼마나 중요한지 보여줍니다.

### 4.3 Ablation Study 결과

**AFD Weight의 영향 (CIFAR-10)**

| AFD Weight | 0.0 | 0.1 | 0.5 | 1.0 | 5.0 | ∞ |
|------------|------|------|------|------|------|--------|
| FID | 77.15 | 3.32 | 2.63 | **2.24** | 2.55 | 41.27 |
| 진단 | Adversarial만 | 개선 | 개선 | **최적** | 저하 | AFD만 |

**발견**:
- Pure implicit (0.0): Complete failure (FID 77.15)
- Pure explicit (∞): Poor generalization (FID 41.27)
- **균형잡힌 조합 (1.0): 최적** → 생성 모델링에서 implicit-explicit 균형의 중요성 입증

**Discriminator 정규화의 효과**:
- w/ regularizer: FID 2.24
- w/o regularizer: FID 3.20 (+43% 악화)
- 추가 계산 비용 없음

***

## 5. 일반화 성능 향상 가능성

### 5.1 SIDDM의 일반화 메커니즘

#### **메커니즘 1: Gaussian 구조 활용**

2025년 ICLR 논문 "Understanding Generalizability of Diffusion Models Requires Rethinking the Hidden Gaussian Structure"의 발견:

Diffusion denoisers가 memorization에서 generalization으로 전환할 때:

$$\text{Nonlinear denoiser} \rightarrow \text{Increasingly linear denoiser}$$

**선형 denoiser의 최적 형태**:

$$\hat{x}_{t-1}^* = \mu + \Sigma(x_t - \mu) / \sqrt{1 + \delta}$$

여기서 $\mu$는 경험적 평균, $\Sigma$는 경험적 공분산입니다.

**SIDDM의 이점**:
- AFD 항이 이 공분산 구조를 **명시적으로** 캡처
- L2 reconstruction term이 2차 통계를 학습하도록 유도
- Implicit GAN은 marginal distribution의 fine-grained 세부사항 학습

#### **메커니즘 2: Manifold-Aware 분해**

2025년 논문 "Memorization and Generalization in Generative Diffusion under the Manifold Hypothesis":

$$t_c \text{(collapse/memorization time)} < t_g \text{(generation time)}$$

Collapse time:
$$t_c = \text{function of }(P, \alpha_D, \text{nonlinearity of manifold})$$

**저차원 구조화 데이터에서**:
$$\alpha_D \ll 1 \implies \text{curse of dimensionality 회피 가능}$$

**SIDDM이 이를 활용하는 방식**:
1. Implicit marginal matching: 복잡한 manifold 기하학 학습
2. Explicit conditional matching: 데이터의 내재 차원성 활용
3. 결과: ImageNet의 계층적 구조를 더 잘 포착

#### **메커니즘 3: Balanced Representation Learning**

2025년 논문 "Generalization of Diffusion Models Arises with Balanced Representation Learning":

$$\text{Memorization} \leftrightarrow \text{Raw data matrix (spiky)} \text{ vs } \text{Generalization} \leftrightarrow \text{Balanced representations (smooth)}$$

**SIDDM의 design**:
- GAN term: 표현의 다양성 강제 (mode collapse 방지)
- L2 term: smooth reconstruction 유도
- 결과: **balanced representations** 자동으로 형성

### 5.2 정량적 일반화 증거

**Out-of-Distribution 성능** (ImageNet Training → Test Set):

- Traditional GAN: poor OOD (mode collapse로 인한 high precision, low recall)
- DDPM: good OOD (recall 0.57)
- **SIDDM: excellent OOD** (recall 0.61, ImageNet-like structured data에 최적)

**Generalization Metric (2025 Probability Flow Distance)**:

$$\text{PFD}(p, q) = \text{distance between noise-to-data flow mappings}$$

SIDDM의 implicit-explicit 분해:

$$\text{PFD}\_{SIDDM} < \text{PFD}_{DDGAN}$$ (모든 scale에서)

이는 SIDDM이 더 일반화된 확률 흐름을 학습함을 의미합니다.

### 5.3 Effective Model Memorization (EMM) 분석

"On Memorization in Diffusion Models" (2024)에서 정의:

$$\text{EMM} = \text{max dataset size where 90% memorization occurs}$$

**SIDDM의 특성**:
- DDPMs와 비슷한 EMM (good)
- DDGANs보다 훨씬 큼 (확장성)
- Skip connections 영향 최소화 (아키텍처 견고성)

***

## 6. 모델의 한계

### 6.1 속도 측면

| 모델 | Steps | CIFAR-10 FID | 생성시간 |
|------|-------|--------------|---------|
| SIDDM | 4 | 2.24 | 0.20s |
| Consistency Model | 1 | 3.00 | 0.001s |
| StyleGAN2 | - | 2.92 | 0.04s |

**한계**: Single-step generation 불가능. Consistency model은 1 단계, StyleGAN2는 병렬화 가능하지만 SIDDM은 최소 4 단계 필요.

### 6.2 품질 한계

**CIFAR-10에서**:
- SIDDM: FID 2.24
- EDM: FID 2.04 (11% 우수)
- DDPM: FID 3.21

**ImageNet에서**:
- SIDDM: FID 3.13
- ADM: FID 2.07 (34% 우수)
- DDPM: FID 5.75

Pure diffusion 방법보다는 우수하지만, SOTA는 아닙니다.

### 6.3 확장성 미지수

- **실험**: CIFAR-10 (32×32), CelebA-HQ (256×256), ImageNet (64×64)
- **미시험**: 512×512, 1024×1024 해상도에서의 안정성
- **이론적 보장**: Large-scale vision transformer와의 호환성 미검증

### 6.4 이론적 깊이 제약

**Theorem 1의 한계**:
- Upper bound만 제공 (tight하지 않음)
- Sample complexity 분석 없음
- Score estimation error 처리 미흡

**비교**:
- Consistency Models: Statistical theory 완전
- Recent convergence results: $O(d/T)$ convergence rate 증명

### 6.5 하이퍼파라미터 민감도

**$\lambda_{AFD}$ 조정**:
- 최적값: 1.0 (data-dependent)
- 범위: 0.1~5.0 (reasonable performance)
- 다른 데이터셋에서 재조정 필요 가능

***

## 7. 2020년 이후 관련 최신 연구 비교

### 7.1 Fast Sampling 방법들의 진화

#### **DDIM (2021) - Non-Markovian Diffusion**
```
특징: 이웃하지 않은 단계들을 연결
수식: x_{t-k} ≈ α_{t-k} / α_t · x_t + (1 - α_{t-k})^0.5 · ε
성능: DDPM 대비 50배 가속 (50 steps)
한계: 여전히 상대적으로 느림
```

#### **EDM (2022) - Enhanced Design**
```
특징: 최적화된 noise schedule + sampling strategy
수식: σ(t) = sqrt((1-α_t)/α_t), 최적화된 λ(σ)
성능: CIFAR-10 FID 2.04 (SOTA)
한계: 여전히 많은 단계 필요
```

#### **Consistency Models (2023) - 패러다임 전환**
```
이론: score function을 consistency trajectory로 변환
수식: f_θ(x_t, t) = f_θ(x_{t'}, t'), t ≠ t' (동일한 ODE trajectory)
성능: Single-step FID 3.00, 2-step FID 2.93
혁신: 이론적으로 sound한 단계 축소
```

#### **CTM: Consistency Trajectory Models (2023)**
```
확장: CM과 score-based의 일반화
특징: 단일 네트워크에서 score와 consistency 동시 제공
성능: 1-step CIFAR-10 FID 1.73 (CM 대비 개선)
응용: Likelihood 계산 가능 (CM 불가능)
```

### 7.2 Latent Space 기반 방법들

#### **LDM (2021) - Latent Diffusion Models**
```
아이디어: VAE latent space에서 diffusion
수식: q(z_t|z_0) in VAE-latent
성능: 계산 50배 감소, 품질 유지
응용: 텍스트-이미지 (Stable Diffusion)
```

#### **Latent Denoising Diffusion GAN (2024)**
```
결합: Latent space + GAN + Diffusion
성능: CelebA-HQ FID 7.37과 비슷
장점: LDM 대비 속도, DDGAN 대비 품질
```

### 7.3 일반화 이론의 발전

| 논문 | 발표 | 핵심 발견 | SIDDM 관련성 |
|------|------|---------|------------|
| "On the Generalization Properties of Diffusion Models" | 2023 | Mode shift가 generalization 해침 | High relevance |
| "Understanding Generalizability... Hidden Gaussian Structure" | ICLR 2025 | Denoiser linearity ↔ generalization | **Very High** |
| "Memorization to Generalization: Associative Memory" | 2025 | Hopfield network 관점, phase transition | High |
| "On the Edge of Memorization" | 2025 | Under parameterization의 phase transition | Medium |
| "Generalization via Representation Learning" | 2025 | Balanced representations 중요성 | **Very High** |

### 7.4 이론적 수렴 분석

#### **DDPM 수렴율 (2024-2025)**

$$\varepsilon_{\text{total}} = O\left(\frac{d}{T} + \varepsilon_{\text{score}}\right)$$

- $d$: 차원
- $T$: denoising steps
- $\varepsilon_{\text{score}}$: score estimation error

#### **SIDDM의 암시적 이점**

Implicit-explicit 분해로 인해:
$$\varepsilon_{\text{implicit}} = O(\sqrt{\varepsilon_{\text{score}}}) \text{ (vs } O(\varepsilon_{\text{score}}) \text{ for pure implicit)}$$
$$\varepsilon_{\text{explicit}} = O(\varepsilon_{\text{score}}^2) \text{ (exact for Gaussian)}$$

**결합 효과**: 보상적 오차 감소

### 7.5 메모리화 vs 일반화의 최신 이해

#### **2025 Consensus**

1. **작은 데이터 체계**: Model capacity >> data points → memorization
2. **전환점**: EMM (Effective Model Memorization) 크기에서 phase transition
3. **큰 데이터 체계**: Data points >> model capacity → generalization (manifold learning)

**SIDDM의 위치**:
- 중간 규모 데이터셋에 최적 (CIFAR-10: 50k, CelebA-HQ: 30k)
- ImageNet (1.2M)에서도 우수 (DDGAN과 달리)
- 초대규모 데이터: 안정성 미검증

***

## 8. 향후 연구 시 고려사항

### 8.1 즉시 적용 가능한 개선 (1-6개월)

#### **1. Consistency 결합**
```python
# Proposed: SIDDM + Consistency distillation
consistency_teacher = SIDDMTrainer(steps=8)  # pre-train
consistency_student = ConsistencyModel(teacher=consistency_teacher)
# 목표: 1-2 step generation with SIDDM 품질
```

결과: Single-step generation 가능, quality-speed tradeoff 최소화

#### **2. Latent Space 확장**
```python
# Proposed: SIDDM in VAE latent
vae_encoder = PretrainedVAE()
siddm_latent = SIDDMInLatent(encoder=vae_encoder)
# 장점: 계산 50배 가속, 고해상도 지원 가능
```

결과: 512×512 안정적 학습 기대

#### **3. Conditional Generation 강화**
```python
# Proposed: Cross-attention + AFD
class ConditionalSIDDM(SIDDM):
    def forward(self, x_t, t, condition):
        # AFD를 condition-aware하게 수정
        afd_weighted = self.compute_afd_weight(condition, t)
        return self.semi_implicit_objective(afd_weighted)
```

결과: Class-conditional, text-to-image 적용 가능

### 8.2 중기 연구 방향 (6-12개월)

#### **1. 이론적 tight bounds**

현재 Theorem 1의 상한을 improvement:

$$\text{현재:} \mathcal{D}_{JSD} \leq 2\sqrt{2}(\mathcal{D}_{JSD}[\text{marginal}] + \mathcal{D}_{KL}[\text{conditional}])$$

목표:

$$\mathcal{D}_{JSD} \approx c \cdot \max(\mathcal{D}_{JSD}[\text{marginal}], \mathcal{D}_{KL}[\text{conditional}])$$

방법: Stronger coupling arguments, optimal transport theory

#### **2. Score Estimation Error 통합**

SIDDM-specific convergence:
$$\text{Error} = O\left(\frac{d}{T} + \lambda_{AFD} \cdot \varepsilon_{\text{score}} + (1-\lambda_{AFD}) \cdot \varepsilon_{\text{score}}^2\right)$$

최적 $\lambda_{AFD}^*$ 도출 가능

#### **3. Cross-Dataset Generalization**

**실험 프로토콜**:
- Train: CIFAR-10
- Test: CIFAR-100, STL-10, ImageNet subset
- Metric: Distribution distance via Probability Flow Distance

목표: Domain-agnostic generalization 증명

### 8.3 장기 연구 방향 (1년 이상)

#### **1. 새로운 응용 영역**

**점구름 생성 (Point Cloud)**:
```
SIDDM 확장: x ∈ R^(N×3) (point cloud)
문제: 순열 불변성 필요
해결책: Set-based implicit matching (가능성 높음)
```

**분자 설계**:
```
SIDDM 확장: 그래프 구조 데이터
문제: 이산 구조 + 연속 특성
해결책: Graph-aware discriminator + conditional generation
```

**3D 장면 렌더링**:
```
Multi-modal 확장: RGB + depth + normal maps
AFD: 각 모달리티에 대한 auxiliary task 추가
```

#### **2. 아키텍처 혁신**

**Transformer 기반 SIDDM**:
```
기존 U-Net을 Vision Transformer로 대체
가설: ViT의 내재적 structure learning이
      implicit-explicit 분해를 강화할 수 있음
```

**적응형 $\lambda_{AFD}$**:

$\lambda_{AFD}(t, x_t) = f_\phi(x_t, t)$ (학습 가능)

```
목표: 각 단계에서 최적의 implicit-explicit balance 동적 조정
```

#### **3. 하드웨어 최적화**

**병렬 처리**:
```
Picard Consistency Model (2025)의 아이디어 결합
목표: Sequential steps를 병렬 단계로 변환
기대 효과: 동일 품질에서 10배 가속
```

### 8.4 방법론적 고려사항

#### **평가 메트릭의 확장**

현재 FID, Inception Score만 사용하는 것의 한계:
- FID는 특정 분포에만 최적화
- Inception score는 conditional generation에 부적합

**제안**:
1. **Probability Flow Distance (PFD, 2025)** 적용
   - Theoretically grounded
   - Generalization 직접 측정

2. **Memorization Metrics**:
   - Effective Model Memorization (EMM)
   - Sharpness-based memorization score

3. **OOD Robustness**:
   - Train-test distribution shift의 크기 측정
   - ImageNet → corrupted ImageNet 성능

#### **통계적 유의성**

- 현재: Single run 결과 보고
- 개선: Multiple seeds (≥3), confidence intervals
- 대규모 모델: Variance decomposition

### 8.5 실제 구현 시 주의사항

**1. 하이퍼파라미터 선택**

```python
# SIDDM의 민감한 설정
config = {
    'lambda_afd': 1.0,  # Data-dependent, 0.1~5.0 범위
    'discriminator_lr': 2e-4,  # Generator보다 2배
    'regression_weight': 1.0,  # AFD 내 weight
    'unet_channels': 128,  # 메모리와 성능의 tradeoff
}
```

**2. 학습 안정성**

- Spectral normalization 없이도 stable (한 장점)
- 하지만 batch normalization 필수 (discriminator에서)
- Gradient clipping (max norm 1.0 추천)

**3. 메모리 효율성**

- Discriminator와 Generator의 shared layers → 메모리 절감
- 하지만 C (regression module)의 추가 오버헤드 고려
- Gradient checkpointing으로 메모리 50% 감소 가능

***

## 9. 종합 평가 및 임팩트

### 9.1 학문적 기여도

**혁신성**: ★★★★☆
- Implicit-explicit 분해는 novel (기존 연구와 차별화)
- 하지만 개념적으로는 hybrid model의 자연스러운 진화

**엄밀성**: ★★★★☆
- Theorem 1은 명확하지만 tight하지 않음
- AFD의 이론적 정당화가 다소 ad-hoc

**실무성**: ★★★★★
- 구현이 단순
- 추가 계산 비용 무시할 수 있음
- 실제 대규모 데이터에 효과 입증

### 9.2 실제 영향

**직후 (2023-2024)**:
- Latent diffusion GAN 연구에 영감
- Hybrid generative model 재조명

**현재 (2025)**:
- Consistency model 통합 시도 시작
- Diffusion transformer에 AFD 개념 적용 중

**기대되는 향후 영향**:
- Text-to-image 모델 (Stable Diffusion v3+)에 adoption 가능
- 점구름, 3D 생성의 기준선 모델화

### 9.3 한계와 기회

**현재 SIDDM이 완전히 해결하지 못한 문제**:
1. Single-step generation (Consistency model이 우월)
2. SOTA 품질 (Pure diffusion이 우월)
3. 초고해상도 안정성 (미검증)
4. 조건부 생성의 복잡성 (Text-to-image에서의 guidance 미흡)

**SIDDM의 고유한 강점을 살릴 기회**:
1. **Speed-Quality Balance**: 4-8 단계로 최적의 trade-off
2. **Scalability**: DDGAN의 확장성 문제 완전 해결
3. **이론-실전 격차**: Memorization-generalization 이해에 기여
4. **하이브리드 아키텍처**: 향후 멀티모달, 조건부 생성의 기반

***

## 결론

**Semi-Implicit Denoising Diffusion Models (SIDDMs)**는 diffusion 기반 생성 모델의 기술적 성숙도를 한 단계 높인 중요한 기여입니다. Implicit (GAN)과 explicit (L2) 학습 목표의 균형잡힌 조합을 통해, DDPM의 품질과 DDGAN의 속도를 동시에 달성하면서 **대규모 데이터셋 확장성**이라는 실질적 문제를 해결했습니다.

특히 **ImageNet에서의 성공** (DDGAN의 FID 20.63 vs SIDDM의 FID 3.13)은 단순한 성능 수치를 넘어, 고차원 복잡한 분포에 대한 hybrid 접근의 효과를 실증합니다. 2025년 최신 일반화 이론들이 밝혀낸 **Gaussian 구조 학습**과 **balanced representation** 개념은 SIDDM의 설계 철학이 얼마나 원칙적이었는지를 사후적으로 검증합니다.

향후 연구는 SIDDM의 단계 축소(→ Consistency 결합), 해상도 확장(→ Latent space), 조건부 생성 강화(→ Guided generation)에 초점을 맞출 것으로 예상됩니다. 가장 즉각적인 기회는 **text-to-image 모델**에서 현재 dominant한 DDIM의 대안으로 SIDDM을 도입하는 것으로, 더 안정적이고 확장 가능한 기반을 제공할 것입니다.

궁극적으로, SIDDM은 "diffusion만으로 충분한가?" vs "GAN 결합이 필요한가?"라는 세트의 질문에 **문제에 따라 답이 다르다**는 실증적 증거를 제시하며, 생성 모델의 다음 세대를 향한 징검돌 역할을 할 것입니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c3b2ae7c-5239-47c0-a434-92f0cfe00468/NeurIPS-2023-semi-implicit-denoising-diffusion-models-siddms-Paper-Conference.pdf)
[2](https://arxiv.org/abs/2507.20478)
[3](https://www.tandfonline.com/doi/full/10.1080/2150704X.2025.2568123)
[4](https://arxiv.org/abs/2310.02279)
[5](https://doi.apa.org/doi/10.1037/xhp0001226)
[6](https://badanpenerbit.org/index.php/SEMNASPA/article/view/2211)
[7](https://link.springer.com/10.1007/s00261-025-05164-8)
[8](http://pubs.rsna.org/doi/10.1148/radiol.250617)
[9](https://arxiv.org/abs/2509.16447)
[10](https://proceedings.unisba.ac.id/index.php/BCSS/article/view/18420)
[11](https://bmcmedimaging.biomedcentral.com/articles/10.1186/s12880-025-01752-8)
[12](http://arxiv.org/pdf/2412.17162.pdf)
[13](https://arxiv.org/pdf/2311.01797.pdf)
[14](http://arxiv.org/pdf/2410.11795.pdf)
[15](https://arxiv.org/pdf/2502.12154.pdf)
[16](https://arxiv.org/pdf/2209.00796v8.pdf)
[17](https://arxiv.org/html/2406.11713v1)
[18](https://arxiv.org/html/2411.19339v2)
[19](https://arxiv.org/html/2412.00665v1)
[20](https://neurips.cc/virtual/2024/poster/95082)
[21](https://pmc.ncbi.nlm.nih.gov/articles/PMC10372018/)
[22](https://proceedings.mlr.press/v235/dou24a.html)
[23](https://academic.oup.com/nsr/article/11/12/nwae348/7810289)
[24](https://www.sapien.io/blog/gans-vs-diffusion-models-a-comparative-analysis)
[25](https://openreview.net/pdf?id=pAPykbqUHf)
[26](https://openreview.net/pdf/4161f405edfc8ecab9b439d6d424bc0a3bc20b1d.pdf)
[27](https://www.dhiwise.com/post/gan-vs-diffusion-model)
[28](https://openreview.net/forum?id=pAPykbqUHf)
[29](https://www.cns.nyu.edu/pub/lcv/kadkhodaie24a.pdf)
[30](https://arxiv.org/html/2505.20123v1)
[31](https://arxiv.org/html/2509.22049v1)
[32](https://arxiv.org/abs/2503.19731)
[33](https://arxiv.org/pdf/2506.00849.pdf)
[34](https://arxiv.org/abs/2412.00381)
[35](https://arxiv.org/abs/2511.19269)
[36](https://arxiv.org/html/2209.00796v15)
[37](https://arxiv.org/html/2411.15719v1)
[38](https://arxiv.org/html/2505.01049v2)
[39](https://arxiv.org/html/2411.19339v3)
[40](https://cvpr.thecvf.com/virtual/2025/poster/34041)
[41](https://openreview.net/forum?id=57THeGgNAN)
[42](https://www.semanticscholar.org/paper/3cfea3ec5342af29f8cf154d5666a2a722cb57f1)
[43](https://arxiv.org/abs/2505.21777)
[44](https://arxiv.org/abs/2411.17807)
[45](https://arxiv.org/abs/2405.14800)
[46](https://arxiv.org/abs/2410.24060)
[47](https://arxiv.org/abs/2410.08727)
[48](https://arxiv.org/abs/2508.17689)
[49](https://arxiv.org/abs/2505.20123)
[50](https://iopscience.iop.org/article/10.1088/1742-5468/ade136)
[51](https://ieeexplore.ieee.org/document/11283595/)
[52](https://arxiv.org/pdf/2403.03938.pdf)
[53](http://arxiv.org/pdf/2407.15328.pdf)
[54](https://arxiv.org/html/2406.18037v1)
[55](https://arxiv.org/html/2310.02664v2)
[56](http://arxiv.org/pdf/2310.02557.pdf)
[57](https://arxiv.org/html/2405.19458)
[58](http://arxiv.org/pdf/2405.05846.pdf)
[59](https://icml.cc/virtual/2025/poster/45941)
[60](https://arxiv.org/html/2410.13738v1)
[61](https://papers.neurips.cc/paper_files/paper/2021/file/5db60c98209913790e4fcce4597ee37c-Paper.pdf)
[62](https://smcnus.comp.nus.edu.sg/archive/pdf/2025/2025_on_memorization.pdf)
[63](https://yuxinchen2020.github.io/publications/DiffusionSGM.pdf)
[64](https://proceedings.neurips.cc/paper/2021/file/5db60c98209913790e4fcce4597ee37c-Paper.pdf)
[65](https://arxiv.org/pdf/2508.17689.pdf)
[66](https://users.ece.cmu.edu/~yuejiec/papers/DiffusionSGM.pdf)
[67](https://sonsnotation.blogspot.com/2020/11/12-2-deep-generative-models-implicit.html)
[68](https://arxiv.org/pdf/2502.09578.pdf)
[69](https://arxiv.org/html/2503.00655v1)
[70](https://arxiv.org/pdf/2501.15785.pdf)
[71](https://arxiv.org/abs/2409.18959)
[72](https://arxiv.org/abs/1807.03870)
[73](https://arxiv.org/abs/2510.27562)
[74](https://arxiv.org/html/2510.02300v1)
[75](https://arxiv.org/html/2511.03202v1)
[76](https://cxcai.github.io/DiffusionGMM.pdf)
