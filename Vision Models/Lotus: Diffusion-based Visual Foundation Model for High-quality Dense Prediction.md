
# Lotus: Diffusion-based Visual Foundation Model for High-quality Dense Prediction

> **논문 정보**: He, J., Li, H., Yin, W., Liang, Y., Li, L., Zhou, K., Zhang, H., Liu, B., & Chen, Y.-C. (2024). *Lotus: Diffusion-based Visual Foundation Model for High-quality Dense Prediction*. arXiv:2409.18124.

---

## 1. 📌 핵심 주장 및 주요 기여 요약

사전 학습된 텍스트-투-이미지(text-to-image) 확산 모델(diffusion model)의 시각적 사전 지식(visual priors)을 활용하는 것은 밀집 예측(dense prediction) 태스크에서 제로샷 일반화 성능을 높이는 유망한 방법이지만, 기존 방법들은 밀집 예측과 이미지 생성 사이의 근본적인 차이를 간과한 채 원래의 diffusion 공식을 그대로 사용해왔다.

이 논문은 이러한 문제 인식에서 출발하여 세 가지 핵심 기여를 제안합니다:

| 기여 항목 | 내용 |
|---|---|
| ① **파라미터화 전환** | 노이즈 예측 → 직접 어노테이션 예측 ($x_0$-prediction) |
| ② **단일 스텝 확산 프로세스** | 다중 스텝 → 단일 스텝으로 단순화 |
| ③ **Detail Preserver** | 파인튜닝 시 세밀한 디테일 보존 전략 도입 |

훈련 데이터나 모델 용량을 늘리지 않고도 Lotus는 다양한 데이터셋에서 제로샷 깊이(depth) 및 법선(normal) 추정에서 최첨단(SoTA) 성능을 달성했으며, 기존 대부분의 diffusion 기반 방법들보다 훨씬 빠른 추론 속도를 보여주었다. 또한 Lotus의 우수한 품질과 효율성은 공동 추정(joint estimation), 단일/다중 시점 3D 복원 등 다양한 실용적인 응용을 가능하게 한다.

---

## 2. 🔬 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능

### 2.1 해결하고자 하는 문제

이 논문은 품질과 효율성 측면에서 밀집 예측을 위한 diffusion 공식화에 대한 체계적인 분석을 제공한다. 이미지 생성을 위한 원래의 파라미터화 방식—노이즈를 예측하도록 학습하는 방식—은 밀집 예측에 해롭고, 다중 스텝 노이즈 추가/제거(noising/denoising) 확산 프로세스 또한 불필요하며 최적화하기 어렵다는 것을 발견했다.

구체적으로 세 가지 핵심 문제를 규명합니다:

**문제 ①: 노이즈 예측 파라미터화의 유해성**

표준 diffusion 모델에서 사용되는 노이즈 예측(noise prediction) 파라미터화는 상당한 분산(variance)을 도입하여 오류를 전파시킨다. 이를 대신해, Lotus는 직접 어노테이션을 예측하도록 훈련하는 방식을 제안하여 분산을 완화하고 더 안정적이고 정확한 출력을 달성한다.

**문제 ②: 다중 스텝 프로세스의 비효율성**

전통적인 diffusion 모델은 계산 집약적일 뿐만 아니라 오류 누적에 취약한 다중 스텝 프로세스에 의존한다. 실험 결과, 단일 스텝 diffusion 공식화는 더 단순할 뿐만 아니라 특히 훈련 데이터가 제한된 경우 더 나은 성능으로 이어지는 것으로 나타났다.

**문제 ③: Catastrophic Forgetting (파국적 망각)**

원래의 diffusion 모델은 디테일한 이미지 생성에 탁월하다. 그러나 밀집 어노테이션을 예측하도록 적응시킬 때, 예기치 않은 파국적 망각(catastrophic forgetting)으로 인해 세밀한 생성 능력을 잃을 수 있다. 이는 복잡한 영역에서의 밀집 어노테이션 예측에 어려움을 초래한다.

---

### 2.2 제안하는 방법 (수식 포함)

#### 🔷 핵심 적응 프로토콜 (Adaptation Protocol)

사전 학습된 VAE 인코더 $\mathcal{E}$가 이미지 $\mathbf{x}$와 어노테이션 $\mathbf{y}$를 잠재 공간(latent space)에 인코딩한 후, ① denoiser U-Net 모델 $f_\theta$는 $x_0$-prediction을 사용하여 파인튜닝되고, ② 더 나은 커버리지를 위해 타임스텝 $t=T$에서 단일 스텝 확산 공식화를 사용하며, ③ switcher $s$를 통해 모델이 이미지를 재구성하거나 밀집 예측을 생성할 수 있도록 하는 novel detail preserver를 제안한다.

#### 🔷 파라미터화 전환: $\epsilon$-prediction → $x_0$-prediction

표준 DDPM에서의 노이즈 예측 파라미터화는 다음과 같이 정의됩니다:

$$
\mathbf{z}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{z}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

기존 $\epsilon$-prediction 방식의 학습 목표:

$$
\mathcal{L}_{\epsilon} = \mathbb{E}_{t, \mathbf{z}_0, \boldsymbol{\epsilon}}\left[\left\|\boldsymbol{\epsilon} - f_\theta(\mathbf{z}_t, t)\right\|^2\right]
$$

Lotus에서는 직접 클린 데이터($x_0$)를 예측하는 $x_0$-prediction으로 전환합니다:

$$
\mathcal{L}_{x_0} = \mathbb{E}_{t, \mathbf{z}_0^y, \boldsymbol{\epsilon}}\left[\left\|\mathbf{z}_0^y - f_\theta(\mathbf{z}_t^y, \mathbf{z}^x, t)\right\|^2\right]
$$

여기서:
- $\mathbf{z}^x$ = 인코딩된 RGB 이미지 (컨디셔닝 입력)
- $\mathbf{z}_0^y$ = 인코딩된 어노테이션(클린 타겟)
- $\mathbf{z}_t^y$ = 타임스텝 $t$에서 노이즈가 추가된 어노테이션 잠재 변수

#### 🔷 단일 스텝 확산 공식화

모델 수렴을 돕고 제한된 고품질 데이터로도 더 나은 최적화 성능을 달성하기 위해, 순수 노이즈에서 클린 출력까지 단 하나의 스텝만 거치는 단일 스텝 공식화를 도입한다. 이는 훈련 및 추론 효율성 모두를 상당히 향상시킨다.

타임스텝 $t=T$ (최대 노이즈 스텝)에서 단일 스텝으로 직접 예측:

$$
\mathbf{z}_T^y = \sqrt{\bar{\alpha}_T}\,\mathbf{z}_0^y + \sqrt{1-\bar{\alpha}_T}\,\boldsymbol{\epsilon} \approx \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \quad (\bar{\alpha}_T \approx 0 \text{ 일 때})
$$

최종 추론 (단일 스텝):

$$
\hat{\mathbf{z}}_0^y = f_\theta\!\left(\mathbf{z}_T^y,\, \mathbf{z}^x,\, T\right)
$$

최종 어노테이션은 VAE 디코더 $\mathcal{D}$를 통해 복원됩니다:

$$
\hat{\mathbf{y}} = \mathcal{D}(\hat{\mathbf{z}}_0^y)
$$

#### 🔷 Detail Preserver

입력 이미지의 풍부한 디테일을 보존하기 위해, "Detail Preserver"라고 불리는 새로운 정규화 전략을 도입한다. 이전 연구들에서 영감을 받아, 태스크 switcher $s \in \{s_x, s_y\}$를 활용하여 denoiser 모델 $f_\theta$가 어노테이션을 생성하거나 입력 이미지를 재구성할 수 있도록 한다.

novel detail preserver는 태스크 switcher를 통해 구현되며, 모델이 어노테이션 생성과 입력 이미지 재구성 사이를 전환할 수 있도록 한다. 이를 통해 밀집 어노테이션 생성 과정에서 입력 이미지의 세밀한 디테일을 더 잘 보존하여 효율성 저하, 추가 파라미터 도입, 또는 표면 텍스처의 영향 없이 더 높은 성능을 달성한다.

결합된 훈련 손실:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{x_0}^{(y)} + \lambda \cdot \mathcal{L}_{x_0}^{(x)}
$$

- $\mathcal{L}_{x_0}^{(y)}$: 어노테이션 예측 손실 (switcher $s_y$ 활성화)
- $\mathcal{L}_{x_0}^{(x)}$: 이미지 재구성 손실 (switcher $s_x$ 활성화, 망각 방지)
- $\lambda$: 균형 하이퍼파라미터

---

### 2.3 모델 구조

Lotus는 **두 가지 변형**을 제공합니다:

**Lotus-G (Generative)**: 불확실성 정량화가 필요한 태스크를 위해 가우시안 노이즈 입력을 통합하며, **Lotus-D (Discriminative)**: 분산 없이 안정적인 출력을 선호하는 사용자에 더 적합한 결정론적 예측 모델로 노이즈 요소를 제거한다.

표준 가우시안 노이즈 $\mathbf{z}_T^y$와 인코딩된 RGB 이미지 $\mathbf{z}^x$를 연결하여 입력을 형성하며, $t=T$와 switcher를 $s_y$로 설정하여 denoiser U-Net 모델이 잠재 밀집 예측을 출력하고, 이를 디코딩하여 최종 출력을 얻는다.

모델의 전체 구조 흐름:

```
[RGB 이미지 x]  ──→  VAE 인코더 ε  ──→  z^x (조건)
                                              ↓
[어노테이션 y]  ──→  VAE 인코더 ε  ──→  z^y_0
                                              ↓ 노이즈 추가 (t=T)
                                         z^y_T (순수 노이즈)
                                              ↓
                            [z^x + z^y_T, timestep T, switcher s]
                                              ↓
                                denoiser U-Net f_θ (파인튜닝)
                                              ↓
                                    z^y_0 (예측 latent)
                                              ↓
                              VAE 디코더 D  ──→  ŷ (최종 예측)
```

---

### 2.4 성능 향상

Lotus는 최첨단 결과를 단 0.059백만 이미지라는 매우 적은 훈련 데이터로 달성하는데, 이는 수천만 이미지를 활용하는 일부 심도 추정 베이스라인과 대조적이다. 이는 강력한 diffusion prior를 효과적으로 활용하는 Lotus의 적응 프로토콜의 효율성을 잘 보여준다.

깊이 추정에서 Lotus-G는 다른 모든 방법들을 능가하며, Lotus-D는 DepthAnything에 약간 뒤처진다. 주목할 점은 DepthAnything은 6,260만 이미지로 학습된 반면, Lotus는 단 5만 9천 이미지만으로 학습된다는 것이다.

제로샷 법선 추정 정량적 비교에서는 Lotus-G와 Lotus-D 모두 다른 모든 방법들을 유의미한 차이로 능가한다.

Lotus는 Marigold보다 수백 배 빠르며, 고해상도에서 DepthAnything V2보다 약간 더 빠르다.

---

## 3. 🌐 일반화 성능 향상 가능성

일반화 성능 향상은 Lotus의 가장 중요한 측면 중 하나입니다.

### 3.1 사전 학습 확산 모델의 Visual Prior 활용

사전 학습된 텍스트-투-이미지 diffusion 모델의 시각적 사전 지식(visual priors)을 활용하는 것은 밀집 예측 태스크에서 제로샷(zero-shot) 일반화 성능을 향상시키는 유망한 방법이다.

이미지 생성을 위해 수십억 개의 이미지-텍스트 쌍으로 학습된 Stable Diffusion은 광범위한 시각적 세계 모델(world model)을 내재하고 있습니다. Lotus는 이 풍부한 사전 지식을 활용하여 적은 데이터로도 높은 일반화 성능을 달성합니다.

### 3.2 최소 훈련 데이터로 달성한 제로샷 성능

전통적인 discriminative 방법들과 비교했을 때, Lotus는 강력한 diffusion prior를 효과적으로 활용하여 단 59K의 훈련 샘플만으로 탁월한 결과를 달성한다. 생성적 접근 방법들 중에서도 Lotus는 이전 방법들을 정확도와 효율성 모두에서 능가하며, Marigold와 같은 방법들보다 상당히 빠르다.

### 3.3 혼합 데이터셋 전략을 통한 도메인 일반화

처음에는 Hypersim 데이터셋만을 사용하여 기준선(baseline)을 설정하고, 이후 Virtual KITTI를 포함한 혼합 데이터셋 전략으로 훈련 데이터를 확장하여 서로 다른 도메인 간 모델의 일반화 능력을 향상시키는 것을 목표로 한다.

### 3.4 일반화 성능 한계 요인

DepthAnything은 6,350만 이미지로 학습되었지만, discriminative한 특성으로 인해 훈련 이미지와 크게 다른 이미지에 대한 일반화 능력이 제한될 수 있다. 또한 그 결과는 풍부한 기하학적 세부 사항을 포착하는 데 실패한다.

Lotus는 이와 달리 generative prior의 힘으로 "보지 못한(unseen)" 도메인에서도 강건한 성능을 유지합니다.

---

## 4. 🔄 2020년 이후 주요 관련 연구 비교 분석

| 모델 | 연도 | 방법론 | 주요 특징 | 한계 |
|---|---|---|---|---|
| **DPT** | 2021 | Discriminative (ViT 기반) | 대규모 데이터 필요 | 도메인 외 일반화 약함 |
| **MiDaS v3** | 2022 | Discriminative | 다중 데이터셋 혼합 학습 | 세밀한 디테일 부족 |
| **Marigold** | 2024 (CVPR) | Diffusion (Stable Diffusion 파인튜닝) | 합성 데이터만으로 zero-shot 일반화 | 느린 추론(다중 스텝 필요) |
| **GeoWizard** | 2024 (ECCV) | Diffusion (joint depth+normal) | 장면 분포 디커플링 | 실내/실외 선택 필요 |
| **DepthAnything V2** | 2024 | Discriminative (ViT-Large) | 초대규모 데이터 학습 | 데이터 의존적, 세밀한 디테일 부족 |
| **GenPercept** | 2024 | Single-step Diffusion | 단일 스텝 도입 | 체계적 분석 부재 |
| **Lotus** | 2024 | Diffusion ($x_0$-pred + 단일 스텝) | 최소 데이터로 SoTA, 빠른 추론 | 학습 도메인 제한 (합성 데이터) |
| **Lotus-2** | 2024 | Two-stage Deterministic | LCM + 정류 흐름(rectified flow) 정제 | 추가 단계로 인한 약간의 복잡성 증가 |

Marigold는 단안 깊이 추정을 위한 diffusion 모델 및 파인튜닝 프로토콜로, 현대 생성 이미지 모델에 저장된 풍부한 시각적 지식을 활용한다. Stable Diffusion에서 파생되어 합성 데이터로 파인튜닝된 모델로, 보지 못한 데이터에 제로샷 전이가 가능하다.

GeoWizard는 훈련 중 서로 다른 장면 분포를 분리하는 디커플러 모듈을 도입하여 혼합 데이터로 인한 흐림 현상과 모호성을 줄임으로써 Marigold를 개선했다.

최근 GenPercept와 StableNormal도 단일 스텝 diffusion을 채택했다. 그러나 GenPercept는 결정론적 특성을 위해 먼저 노이즈 입력을 제거하고, 표면 텍스처 간섭을 피하기 위한 단일 스텝 전략을 채택한다. 하지만 diffusion 공식에 대한 체계적인 분석이 부족하며 U-Net을 단순히 결정론적 백본으로 취급하여 여전히 성능이 부족하다.

후속 연구인 Lotus-2에 대해서도 주목할 필요가 있습니다: Lotus-2는 안정적이고 정확하며 세밀한 기하학적 밀집 예측을 위한 2단계 결정론적 프레임워크로, 사전 학습된 생성 prior를 최대한 활용하기 위한 최적의 적응 프로토콜을 제공하는 것을 목표로 한다. 첫 번째 단계에서는 핵심 예측기가 단일 스텝 결정론적 공식화와 경량화된 로컬 연속성 모듈(LCM)을 사용하고, 두 번째 단계에서는 디테일 샤프너가 핵심 예측기가 정의한 다양체 내에서 제한된 다중 스텝 정류 흐름 정제를 수행하여 노이즈 없는 결정론적 흐름 매칭을 통해 세밀한 기하학적 구조를 향상시킨다.

---

## 5. ⚠️ 한계점

1. **합성 데이터 의존성**: Lotus는 단 0.059백만 이미지라는 최소한의 훈련 데이터로 최첨단 결과를 달성하지만, 훈련 데이터가 주로 Hypersim과 Virtual KITTI 같은 합성 데이터셋으로 구성되어 실세계 분포와의 간극이 존재할 수 있습니다.

2. **세밀한 영역에서의 예측 불명확성**: 모델이 매우 세밀한 영역에서 종종 모호한 예측을 출력하는 것이 관찰되었다. 이 모호성은 파국적 망각(catastrophic forgetting)에 기인하는데, 사전 학습된 diffusion 모델이 파인튜닝 과정에서 점진적으로 세밀한 영역을 생성하는 능력을 잃어버리기 때문이다.

3. **적용 태스크 범위**: Lotus가 더 크고 복잡한 시각적 입력에 어떻게 확장될 것인지, 또는 다중 객체 상호작용이나 장면 이해와 같이 더 고차원적인 추론을 요구하는 태스크를 어떻게 처리할 것인지가 불명확하다.

---

## 6. 🔮 미래 연구에 미치는 영향 및 고려할 점

### 6.1 연구에 미치는 영향

**① Diffusion 모델의 밀집 예측 활용 패러다임 전환**

diffusion 모델의 역할을 밀집 예측을 위해 재정의하는 것이 핵심이다. 확산 기반 생성 모델을 확률적 이미지 생성기에서 구조화된 세계 prior로 재정립하여, 그 강점이 샘플링 궤적 자체보다는 사전 학습된 가중치에 내재된 세계 모델링 능력에 있음을 강조한다.

**② 데이터 효율적 학습 방향 제시**

Lotus는 매우 적은 데이터(59K)로도 수천만 장의 데이터를 활용한 모델과 경쟁할 수 있음을 보여줌으로써, 데이터가 부족한 전문 도메인(의료 영상, 위성 이미지 등)에서의 밀집 예측에 새로운 가능성을 열었습니다.

**③ 후속 연구 Lotus-2로의 직접적 연결**

이 전례 없는 데이터 효율성은 추론 안정성 및 세밀한 충실도와 결합되어 결정론적 적응 프로토콜의 효능을 검증한다. 궁극적으로 이 연구는 생성적 diffusion 모델에 축적된 방대한 지식이 효율적이고 정확하며 물리적으로 일관된 기하학적 추론을 가능하게 하는 방향으로 재활용될 수 있음을 보여주며, 전통적인 discriminative 및 생성적 방법을 넘어서는 새로운 패러다임을 제시한다. 이 발견은 기반 생성 모델에서 구조화된 지식을 추출하고 활용하는 향후 연구에 유망한 방향을 열어준다.

---

### 6.2 향후 연구 시 고려할 점

**① Diffusion Formulation 재설계 측면**
- $x_0$-prediction과 flow matching의 결합 가능성 탐구
- 노이즈 스케줄(noise schedule)을 태스크 특성에 맞게 최적화
- 단일 스텝 vs. 소수 스텝(few-step) 사이의 품질-속도 트레이드오프 정교화

**② 일반화 성능 강화 측면**
- 도메인 어댑테이션(domain adaptation) 기법과의 결합
- 실세계 데이터와 합성 데이터 혼합 비율 최적화
- Virtual KITTI 포함 같은 혼합 데이터셋 전략을 더 다양한 도메인으로 확장하여 도메인 간 모델의 일반화 능력을 향상시키는 방향 모색

**③ Catastrophic Forgetting 완화 측면**
- 모델이 복잡한 구조의 세밀한 영역에서 세부 사항을 생성하는 능력을 잃는 경향이 있다는 것이 관찰되었으므로, EWC(Elastic Weight Consolidation) 등 지속 학습(continual learning) 기법의 통합이 유용할 수 있습니다.

**④ 응용 확장 측면**
- 동영상 깊이 추정, 의미론적 분할(semantic segmentation), 광학 흐름(optical flow) 등 다른 밀집 예측 태스크로의 확장
- 광학 흐름 추정, 오픈 어휘 의미 분할, 단안 깊이 추정, 표면 법선 예측 등 다양한 dense predictive 비전 태스크에 diffusion이 강력한 백본으로 빠르게 부상하고 있다.

**⑤ 모델 스케일링 및 Foundation Model 측면**
- 단일 모델로 다중 태스크를 동시에 수행하는 범용 밀집 예측 foundation model로의 발전
- 태스크별 아키텍처나 파인튜닝 없이 광범위한 밀집 예측 문제를 처리할 수 있는 generalist 모델 구축을 향한 연구

---

## 📚 참고 자료 및 출처

| 번호 | 출처 |
|---|---|
| 1 | **Lotus 논문 (arXiv)**: He, J. et al. (2024). *Lotus: Diffusion-based Visual Foundation Model for High-quality Dense Prediction*. arXiv:2409.18124. https://arxiv.org/abs/2409.18124 |
| 2 | **Lotus 프로젝트 페이지**: https://lotus3d.github.io/ |
| 3 | **Lotus GitHub**: https://github.com/EnVision-Research/Lotus |
| 4 | **Lotus HuggingFace 논문 페이지**: https://huggingface.co/papers/2409.18124 |
| 5 | **OpenReview**: https://openreview.net/forum?id=stK7iOPH9Q |
| 6 | **Lotus-2 (arXiv)**: He, J. et al. (2024). *Lotus-2: Advancing Geometric Dense Prediction with Powerful Image Generative Model*. arXiv:2512.01030. https://arxiv.org/abs/2512.01030 |
| 7 | **Marigold (CVPR 2024)**: Ke, B. et al. (2024). *Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation*. CVPR 2024. https://marigoldmonodepth.github.io/ |
| 8 | **GeoWizard (ECCV 2024)**: Fu, X. et al. (2024). *GeoWizard: Unleashing the Diffusion Priors for 3D Geometry Estimation from a Single Image*. ECCV 2024. |
| 9 | **Survey on Monocular Metric Depth Estimation**: arXiv:2501.11841. https://arxiv.org/abs/2501.11841 |
| 10 | **저자 블로그**: Ying-Cong Chen (2024). https://www.yingcong.me/post/2024-lotus/ |
| 11 | **Moonlight Literature Review**: https://www.themoonlight.io/en/review/lotus-diffusion-based-visual-foundation-model-for-high-quality-dense-prediction |
| 12 | **Liner Quick Review**: https://liner.com/review/lotus-diffusionbased-visual-foundation-model-for-highquality-dense-prediction |

> ⚠️ **주의**: 본 답변의 수식 일부(특히 손실 함수 세부 계수 등)는 논문의 공개된 내용을 기반으로 재구성되었으며, 논문 원문에서 모든 표기가 명시적으로 공개된 것은 아닙니다. 정확한 수식의 완전한 형태 및 하이퍼파라미터 설정은 반드시 원문(arXiv:2409.18124)을 직접 확인하시기 바랍니다.
