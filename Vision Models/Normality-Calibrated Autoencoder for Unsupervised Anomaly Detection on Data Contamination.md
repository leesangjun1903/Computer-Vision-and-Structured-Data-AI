# Normality-Calibrated Autoencoder for Unsupervised Anomaly Detection on Data Contamination (NCAE)

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

NCAE(Normality-Calibrated Autoencoder)는 **데이터 오염(Data Contamination) 환경**에서, 즉 학습 데이터셋에 이상 샘플이 섞여 있는 현실적인 상황에서, **사전 정보나 명시적인 이상 샘플 없이** 비지도 방식으로 이상 탐지(Anomaly Detection) 성능을 향상시킬 수 있다는 것을 주장합니다.

### 주요 기여

| 기여 항목 | 설명 |
|---|---|
| **오염 견고성** | 훈련 데이터 오염에 강건한 완전 비지도 이상 탐지 방법론 제안 |
| **GAN 기반 정상 샘플 생성** | 저엔트로피 잠재 공간에서 고신뢰도 정상 샘플을 적대적으로 생성 |
| **NCR Loss** | 정상 샘플의 재구성 오차를 최소화하고 오염 샘플의 재구성 오차를 최대화하는 손실 함수 설계 |
| **오염 샘플 마이닝** | 레이블 없이 오염 샘플을 동적으로 식별하는 메커니즘 제안 |
| **경쟁력 있는 성능** | 비지도 방법 대비 우수하며, 레이블을 활용하는 반지도 방법과도 비교 가능한 성능 달성 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

기존 이상 탐지 방법의 대부분은 **학습 데이터가 오직 정상 샘플로만 구성**되어 있다고 가정합니다. 그러나 현실의 데이터셋은 쉽게 오염되며, 이러한 오염된 샘플은 모델의 이상 탐지 성능을 심각하게 저하시킵니다.

기존 접근법의 한계:
- **오염 비율 기반 필터링**: 오염 비율을 사전에 알아야 함
- **반지도 학습**: 명시적인 이상 샘플 레이블 필요
- **기하학적 거리 기반 방법**: 이상 샘플이 정상 분포에서 멀리 위치한다고 가정

하지만 Figure 1이 보여주듯, **오염 비율이 10% 이상인 경우 오염 샘플도 저엔트로피 공간에 위치**할 수 있어 기존 방법이 실패합니다.

---

### 2.2 제안 방법 및 수식

#### Step 1: 기본 오토인코더 목적 함수

기본 오토인코더 $f$(인코더), $g$(디코더)의 목적 함수:

$$\min_{f,g} \mathbb{E}_{x \sim p_\mathcal{X}} \|x - \bar{x}\|^2, \quad \bar{x} = g \cdot f(x) \tag{1}$$

그러나 오토인코더는 **과신뢰(over-confidence)** 문제, 즉 이상 샘플도 낮은 재구성 오차를 보이는 문제를 가집니다. 이는 데이터가 오염되면 더욱 심해집니다.

---

#### Step 2: Normality-Calibrated Reconstruction (NCR) Loss

정상 샘플의 재구성 오차를 최소화하고, 오염 샘플의 재구성 오차를 최대화하는 NCR 손실:

$$\min_{f,g} \mathbb{E}_{x \sim p_{\mathcal{X}^N}} \|x - \bar{x}\|^2 - \mathbb{E}_{x^c \sim p_{\mathcal{X}^C}} \|x^c - \bar{x}^c\|^2 \tag{2}$$

여기서 $p_{\mathcal{X}^N}$은 정상 샘플 분포, $p_{\mathcal{X}^C}$는 오염 샘플 분포를 나타냅니다. 이 손실 함수를 최적화하려면 **어떤 샘플이 오염되었는지 식별**해야 합니다.

---

#### Step 3: GAN 기반 잠재 공간 분포 변환 (Adversarial Loss for Latent Space)

인코더 $f$의 잠재 특징 분포를 가우시안 분포로 변환하기 위한 적대적 손실:

$$\min_{f} \max_{D_l} \mathbb{E}_{\omega \sim \mathcal{N}(\mu_\mathcal{Z}, I_d)} [\log D_l(\omega)] + \mathbb{E}_{x \sim P_\mathcal{X}} [\log (1 - D_l(f(x)))] \tag{3}$$

여기서 $D_l$은 잠재 특징에 대한 판별자(discriminator), $\mathcal{N}(\mu_\mathcal{Z}, I_d)$는 잠재 특징의 평균 $\mu_\mathcal{Z} \in \mathbb{R}^d$와 단위 행렬 $I_d \in \mathbb{R}^{d \times d}$를 공분산으로 하는 가우시안 분포입니다.

학습 중 $\mu_\mathcal{Z}$는 매 스텝 업데이트됩니다:

$$\mu_\mathcal{Z}^{t+1} = \mu_\mathcal{Z}^t - \gamma \frac{1}{m} \sum_{i=1}^m (\mu_\mathcal{Z}^t - z_i), \quad \mu_\mathcal{Z}^0 = \frac{1}{n} \sum_{i=1}^n z_i^0 \tag{4}$$

---

#### Step 4: 고신뢰도 정상 샘플 생성 (Adversarial Loss for Samples)

디코더 $g$를 이용해 저엔트로피 공간의 노이즈로부터 고신뢰도 정상 샘플을 생성하는 적대적 손실:

$$\min_{g} \max_{D_s} \mathbb{E}_{x \sim P_\mathcal{X}} [\log D_s(x)] + \mathbb{E}_{\hat{\omega} \sim \mathcal{N}(\mu_\mathcal{Z}, \sigma I_d)} [\log (1 - D_s(g(\hat{\omega})))] \tag{5}$$

여기서 $D_s$는 샘플에 대한 판별자, $\sigma \in [0, 1]$는 노이즈 분포의 컴팩트함을 조절하는 하이퍼파라미터입니다. **$\sigma$가 작을수록 분포의 중심에 가까운 샘플을 생성**하여 더 높은 신뢰도의 정상 샘플을 생성합니다.

---

#### Step 5: 오염 샘플 마이닝 (Pseudo Contamination Score)

생성된 정상 샘플의 잠재 특징 딕셔너리 $\mathcal{M} = [\hat{z}_i]\_{i=1:m}$를 이용해, 각 입력 샘플 $x_i$의 **유사 오염 점수(pseudo contamination score)**를 계산:

$$c_i = \frac{1}{m} \sum_{j=1}^m f(x_i) \cdot \hat{z}_j^\top, \quad \hat{z}_j \in \mathcal{M} \tag{6}$$

$l_2$-정규화를 적용하여 벡터 스케일 변화에 대한 견고성을 높입니다. 점수 내림차순으로 정렬 후 상위 $\tau\%$ 샘플을 오염 샘플로 예측합니다:

$$\mathcal{X}^C = \{x_t\}_{t \in C[1:\lceil \tau m \rceil]}, \quad C = \arg\text{sort}_i \, c_i, \quad \text{w.r.t.} \; 1 \leq i \leq m$$

---

#### Step 6: 전체 목적 함수 (Joint Learning Objective)

$$\min_{f,g} \max_{D_l, D_s} \underbrace{\mathbb{E}_{x \sim p_{\mathcal{X}^N}} \|x - f \cdot g(x)\|^2 - \mathbb{E}_{x \sim p_{\mathcal{X}^C}} \|x - \bar{x}'\|^2}_{(a)} $$
$$+ \underbrace{\mathbb{E}_{\omega \sim \mathcal{N}(\mu_\mathcal{Z}, I_d)} [\log D_l(\omega)] + \mathbb{E}_{x \sim P_\mathcal{X}} [\log (1 - D_l(f(x)))]}_{(b)} $$
$$+ \underbrace{\mathbb{E}_{x \sim P_\mathcal{X}} [\log D_x(\omega)] + \mathbb{E}_{\omega' \sim \mathcal{N}(\mu_\mathcal{Z}, \sigma I_d)} [\log (1 - D_s(g(\omega')))]}_{(c)} \tag{7}$$

- **(a)**: NCR Loss (정상 재구성 최소화 + 오염 재구성 최대화)
- **(b)**: 잠재 공간 분포 변환을 위한 적대적 손실
- **(c)**: 고신뢰도 정상 샘플 생성을 위한 적대적 손실

여기서 $\bar{x}'$는 오염 샘플에 대해 생성된 고신뢰도 정상 샘플 중 잠재 특징 공간에서 가장 가까운 샘플입니다.

---

### 2.3 모델 구조

```
입력 샘플 x
    │
    ▼
[Encoder f] ──────────────────────────────────────────────────────┐
    │                                                              │
    ▼                                                              ▼
잠재 특징 z                                               [Discriminator D_l]
    │                    ▲                                         ▲
    │                    │                                         │
    │         N(μ_Z, I_d) 가우시안 샘플링 (ω)                     │
    │                                                              │
    ├──────────────────────────────────────────────────────────────┘
    │
    ▼
μ_Z 업데이트 (식 4)
    │
    ▼
N(μ_Z, σI_d) 에서 노이즈 샘플링 (ω')
    │
    ▼
[Decoder g] → 생성된 정상 샘플 x̂
    │
    ├──→ [Discriminator D_s] ← 실제 입력 샘플 x
    │
    ▼
잠재 특징 딕셔너리 M = [ẑ_i]
    │
    ▼
Contamination Score 계산 (식 6)
    │
    ▼
오염 샘플 X^C 예측 (top-τ%)
    │
    ▼
NCR Loss 최적화 (식 2/7)
```

- **백본**: LeNet 기반 CNN (MNIST, Fashion-MNIST)
- **최적화**: Adam optimizer (lr=0.01, decay×0.1 per 10 epochs)
- **배치 크기**: 128
- **하이퍼파라미터**: $\sigma=0.1$, $\tau=0.1$

---

### 2.4 성능 향상

**Table 1 결과** (AUC 기준):

| 오염 비율 ($\rho$) | Deep SVDD | SSAD (반지도) | Deep SAD (반지도) | **NCAE (비지도)** |
|:---:|:---:|:---:|:---:|:---:|
| 0.01 (MNIST) | 92.1 | 96.6 | 95.5 | **97.2** |
| 0.05 (MNIST) | 89.4 | 93.4 | 93.5 | **97.0** |
| 0.10 (MNIST) | 86.5 | 90.7 | 91.2 | **92.6** |
| 0.20 (MNIST) | 81.5 | 87.4 | 86.6 | **89.8** |
| 0.10 (F-MNIST) | 76.2 | 85.6 | 78.2 | **91.5** |
| 0.20 (F-MNIST) | 69.3 | 81.9 | 74.8 | **88.9** |

- **레이블을 사용하는 반지도 방법(SSAD, Deep SAD)보다도 높은 성능** 달성
- 오염 비율이 높을수록 기존 방법 대비 성능 격차가 더욱 두드러짐

---

### 2.5 한계

1. **비오염 데이터에서의 성능 저하**: $\rho = 0.0$일 때 NCAE는 기존 방법 대비 낮은 성능을 보임. 데이터가 오염되지 않아도 오염 샘플을 찾으려 해 재구성 오차를 높이는 부작용 발생. 논문에서도 "critical defect"로 인정.

2. **하이퍼파라미터 민감성**: $\sigma$(노이즈 컴팩트함)와 $\tau$(오염 비율 예측)에 성능이 민감하게 반응하며, 최적값을 결정하기 위해 ablation study가 필요함.

3. **제한적인 벤치마크**: MNIST와 Fashion-MNIST만 실험. 고해상도 이미지, 시계열, 표형 데이터, 의료 영상 등 다양한 도메인에 대한 검증 부재.

4. **GAN 학습 불안정성**: 두 개의 판별자($D_l$, $D_s$)를 동시에 학습하므로 GAN 고유의 학습 불안정성 문제가 존재.

5. **오염 비율 $\tau$ 사전 결정 필요**: $\tau$는 실제 오염 비율과 독립적으로 설정되지만, 적절한 값을 선택하는 명확한 기준이 없음.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능을 높이는 설계 요소

**(1) 저엔트로피 잠재 공간 기반 정상 샘플 생성**

가우시안 분포 $\mathcal{N}(\mu_\mathcal{Z}, \sigma I_d)$의 **중심 근방**에서 샘플링함으로써 정상 클래스의 핵심적이고 전형적인(prototypical) 패턴을 포착합니다. $\mu_\mathcal{Z}$가 학습 중 동적으로 업데이트되므로(식 4), 다양한 데이터 분포에 적응 가능합니다.

**(2) 도메인/데이터 독립적 설계**

기존 기하학적 거리 기반 방법이나 반지도 학습과 달리, NCAE는 특정 데이터 구조에 의존하지 않습니다. 인코더-디코더 구조와 GAN을 결합한 방식은 원칙적으로 이미지, 시계열, 텍스트 등 다양한 모달리티에 적용 가능합니다.

**(3) 적대적 정상화(Adversarial Normalization)**

잠재 공간을 가우시안 분포로 정규화함으로써, 특정 데이터셋에 과적합(overfitting)되는 것을 방지하고 정상 분포를 보다 구조화된 형태로 학습합니다. 이는 VAE의 KL divergence 정규화와 유사한 역할을 하며 일반화에 기여합니다.

**(4) 동적 오염 샘플 마이닝**

$\tau$ 기반의 동적 마이닝 메커니즘은 배치 단위로 오염 샘플을 갱신하므로, 학습 초기의 부정확한 예측이 점진적으로 개선될 수 있습니다. 이는 커리큘럼 학습(curriculum learning)과 유사한 자기 교정(self-correcting) 특성을 부여합니다.

### 3.2 일반화 성능의 한계와 개선 방향

| 한계 | 개선 가능 방향 |
|---|---|
| MNIST/Fashion-MNIST에만 검증 | ImageNet, MVTec AD, 의료 이미지 등 확장 실험 |
| $\sigma$, $\tau$ 수동 설정 | 메타러닝 또는 베이지안 최적화로 자동화 |
| $\tau$와 실제 오염 비율 불일치 가능성 | 적응형 임계값(adaptive threshold) 메커니즘 도입 |
| GAN 학습 불안정성 | Diffusion model 기반 생성 모델로 대체 고려 |
| 고오염 환경에서 $\mu_\mathcal{Z}$ 편향 가능성 | Robust mean estimation 기법 적용 |

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 비교 분석 테이블

| 논문 | 방법 | 오염 견고성 | 레이블 필요 | 주요 특징 | 비교 관점 |
|---|---|---|---|---|---|
| **NCAE** (Yu et al., 2021) | AE + GAN | ✅ 강함 | ❌ 불필요 | 동적 오염 마이닝, NCR Loss | 본 논문 |
| **Deep SAD** (Ruff et al., 2020) | SVDD 기반 | △ 중간 | ✅ 필요(일부) | 반지도, 하이퍼스피어 학습 | 레이블 없이도 유사 성능 달성 |
| **DROCC** (Goyal et al., 2020) | AE + 적대적 샘플 | ❌ 약함 | ❌ 불필요 | 일반화 곡면 학습 | 오염 환경 미고려 |
| **Patch SVDD** (Yi & Yoon, 2020) | SVDD + 패치 | ❌ 약함 | ❌ 불필요 | 이미지 패치 기반 | 산업 이상 탐지 특화 |
| **PANDA** (Reiss et al., 2021) | 사전학습 + SVDD | ❌ 약함 | ❌ 불필요 | ImageNet 사전학습 피처 활용 | 오염 없는 환경 가정 |
| **SimpleNet** (Liu et al., 2023) | Teacher-Student | ❌ 약함 | ❌ 불필요 | 피처 디스크리미네이터 | 산업 이상 탐지 SOTA |
| **CutPaste** (Li et al., 2021) | 자기지도학습 | ❌ 약함 | ❌ 불필요 | 데이터 증강 기반 대조학습 | 오염 없는 환경 가정 |
| **Diffusion-AD** (He et al., 2023) | Diffusion model | △ 중간 | ❌ 불필요 | 생성 모델 기반 재구성 | 고품질 복원 |

### 4.2 핵심 비교 분석

**NCAE의 차별점:**

1. **완전 비지도(Fully Unsupervised) + 오염 견고성**: 2020년 이후 DROCC, PANDA, SimpleNet 등 대부분의 SOTA 비지도 방법들은 여전히 **깨끗한 학습 데이터**를 가정합니다. NCAE는 이 현실적 가정을 명시적으로 완화한 드문 연구 중 하나입니다.

2. **Diffusion-AD와의 비교**: 최근 확산 모델 기반 이상 탐지는 높은 재구성 품질을 보이지만, 오염된 학습 데이터에서의 성능은 검증되지 않았습니다. NCAE의 NCR Loss 개념은 확산 모델 기반 방법에도 적용 가능합니다.

3. **사전학습 모델 미활용**: PANDA 등 최신 방법들은 ImageNet 사전학습 피처를 활용해 높은 성능을 내지만, NCAE는 처음부터 학습합니다. 사전학습 피처를 결합하면 추가적인 성능 향상이 기대됩니다.

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5.1 앞으로의 연구에 미치는 영향

**1. 현실적 벤치마크 설정의 필요성 제기**

NCAE는 기존 연구의 "깨끗한 학습 데이터" 가정이 비현실적임을 명확히 보여줍니다. 앞으로의 이상 탐지 연구는 다양한 오염 비율 $\rho$에서의 성능 평가를 표준 벤치마크로 포함해야 한다는 방향성을 제시합니다.

**2. 생성 모델과 이상 탐지의 결합**

GAN을 활용해 고신뢰도 정상 샘플을 생성하고 이를 참조점(reference)으로 활용하는 아이디어는, 최근 활발한 **Diffusion Model 기반 이상 탐지** 연구에 자연스럽게 확장될 수 있습니다. 특히 확산 모델의 더 안정적인 학습과 고품질 생성 능력은 NCAE의 GAN 불안정성 문제를 해결할 수 있습니다.

**3. 자기 교정(Self-Correcting) 학습 패러다임**

레이블 없이 오염 샘플을 동적으로 식별하고 이를 학습에 활용하는 아이디어는 **노이즈 레이블 학습(Learning with Noisy Labels)** 및 **커리큘럼 학습(Curriculum Learning)** 분야와 깊이 연결됩니다.

**4. 산업 이상 탐지로의 확장**

MVTec AD 등 산업 데이터셋에서는 제조 과정의 특성상 소량의 결함 샘플이 학습 데이터에 포함될 수 있습니다. NCAE의 오염 견고성은 이 분야에서 직접적인 응용 가능성을 가집니다.

---

### 5.2 앞으로의 연구에서 고려할 점

**[방법론적 개선]**

1. **적응형 $\tau$ 설정**: 실제 오염 비율을 알 수 없는 상황에서 $\tau$를 자동으로 추정하는 메커니즘 (예: 가우시안 혼합 모델로 오염 비율 추정) 연구 필요

2. **GAN 대체**: GAN의 학습 불안정성을 해소하기 위해 **VAE**, **Score-based 생성 모델**, **Normalizing Flow** 등으로 대체하는 연구

3. **사전학습 피처 통합**: PANDA, WinCLIP 등과 같이 대규모 사전학습 모델의 피처를 NCAE에 결합하면 일반화 성능이 크게 향상될 가능성

4. **$\rho = 0.0$ 성능 저하 해결**: 오염 여부를 판단하는 별도의 메타-러닝 모듈을 추가하거나, NCR Loss에 적응적 가중치를 부여하는 방식으로 비오염 환경에서도 안정적인 성능 확보

**[벤치마크 및 평가]**

5. **다양한 도메인 검증**: 의료 영상(CheXpert, BraTS), 시계열(KPI, SMD), 표형 데이터(KDD Cup 등), 산업 이상 탐지(MVTec AD, VisA)로 확장 검증

6. **다양한 오염 유형 실험**: 단순 랜덤 오염 외에도 의미론적으로 유사한 오염 샘플(near-distribution outlier), 적대적 오염 샘플 등 다양한 오염 시나리오 고려

7. **계산 효율성 평가**: 두 개의 GAN 판별자를 동시에 학습하는 계산 비용이 실제 산업 환경에서 수용 가능한지에 대한 분석

**[이론적 기반 강화]**

8. **이론적 수렴 분석**: 동적 $\mu_\mathcal{Z}$ 업데이트의 수렴 조건 및 오염 샘플 마이닝의 오류율(false positive rate) 이론적 분석

9. **오염 비율과 성능 간의 이론적 관계**: 어떤 오염 비율까지 NCAE가 안정적으로 동작하는지에 대한 이론적 경계(theoretical bound) 연구

---

## 참고 자료

1. **본 논문**: Yu, J., Oh, H., Kim, M., & Kim, J. (2021). *Normality-Calibrated Autoencoder for Unsupervised Anomaly Detection on Data Contamination*. NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications. arXiv:2110.14825v1.

2. Ruff, L., et al. (2020). *Deep Semi-Supervised Anomaly Detection*. ICLR 2020. arXiv:1906.02694

3. Goodfellow, I., et al. (2014). *Generative Adversarial Nets*. NIPS 2014.

4. Zong, B., et al. (2018). *Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection*. ICLR 2018.

5. Li, T., et al. (2021). *Deep Unsupervised Anomaly Detection*. WACV 2021.

6. Lai, C.-H., Zou, D., & Lerman, G. (2020). *Robust Subspace Recovery Layer for Unsupervised Anomaly Detection*. ICLR 2020.

7. Pidhorskyi, S., Almohsen, R., & Doretto, G. (2018). *Generative Probabilistic Novelty Detection with Adversarial Autoencoders*. NeurIPS 2018.

8. Berg, A., Ahlberg, J., & Felsberg, M. (2019). *Unsupervised Learning of Anomaly Detection from Contaminated Image Data using Simultaneous Encoder Training*. arXiv:1905.11034.

> **⚠️ 주의**: 2020년 이후 최신 연구(DROCC, PANDA, SimpleNet, CutPaste, Diffusion-AD 등)와의 비교 분석 부분은 해당 논문의 출판 시점(2021년 10월) 이후의 연구들을 포함하므로, NCAE 논문 자체에서 직접 비교한 내용이 아닌 reviewer로서의 외부 분석임을 명시합니다. 해당 최신 논문들의 구체적 수치는 원 논문을 직접 확인하시기 바랍니다.
