
# Guiding a Diffusion Model with a Bad Version of Itself

> **논문 정보**
> - **저자**: Tero Karras, Miika Aittala, Tuomas Kynkäänniemi, Jaakko Lehtinen, Timo Aila, Samuli Laine (NVIDIA)
> - **발표**: NeurIPS 2024 (Oral)
> - **arXiv**: [2406.02507](https://arxiv.org/abs/2406.02507)
> - **공식 코드**: [GitHub NVlabs/edm2](https://github.com/NVlabs/edm2)

---

## 1. 핵심 주장 및 주요 기여 요약

### 🔑 핵심 주장 (Core Claim)

더 작고 덜 훈련된 모델 자체의 열등한 버전을 가이딩 모델로 사용함으로써, 이미지 다양성(variation)을 희생하지 않고도 이미지 품질에 대한 **분리된(disentangled) 제어**를 획득할 수 있다는 놀라운 관찰을 제시합니다.

이 방법론을 **Autoguidance**라고 부릅니다.

### 🏆 주요 기여 (Key Contributions)

| 기여 항목 | 내용 |
|---|---|
| 새로운 Guidance 방식 | CFG의 unconditional 모델 대신, 동일 모델의 열등한 버전 사용 |
| 품질-다양성 분리 | 품질 향상과 프롬프트 정렬 효과를 독립적으로 제어 |
| State-of-the-Art FID | ImageNet 64×64: FID 1.01, 512×512: FID 1.25 |
| 비조건부 모델 적용 | CFG가 적용 불가능한 unconditional 모델에도 적용 가능 |

이 방법은 ImageNet 생성에서 공개적으로 사용 가능한 네트워크를 활용해 64×64에서 FID 1.01, 512×512에서 FID 1.25라는 기록적인 성과를 달성합니다.

---

## 2. 해결하고자 하는 문제, 제안하는 방법, 모델 구조, 성능, 한계

### 🔴 2.1 해결하고자 하는 문제

#### CFG(Classifier-Free Guidance)의 한계

첫째, CFG는 가이던스 신호가 조건부·비조건부 결과의 차이에 기반하기 때문에 **조건부 생성에만 적용 가능**합니다. 둘째, 비조건부 및 조건부 denoiser가 다른 작업을 수행하도록 훈련되기 때문에 **샘플링 궤적이 원하는 조건부 분포를 초과(overshoot)** 하여 왜곡되거나 지나치게 단순화된 이미지 구성을 만들어낼 수 있습니다. 셋째, **프롬프트 정렬 개선과 품질 향상 효과를 독립적으로 제어할 수 없습니다**.

#### 훈련 목표의 근본적 문제

Diffusion 모델의 훈련 목표는 전체 (조건부) 데이터 분포를 커버하는 것을 목표로 합니다. 이는 저확률 영역에서 문제를 일으키는데, 모델은 그것을 표현하지 못한 것에 대해 크게 페널티를 받지만, 그에 해당하는 좋은 이미지를 생성하는 방법을 학습하기에 충분한 데이터가 없습니다.

---

### 🟢 2.2 제안하는 방법: Autoguidance

#### 기본 아이디어

이 논문은 CFG가 이미지 품질을 개선하는 이유에 대한 새로운 통찰을 제공하고, 이 효과를 **autoguidance**라는 새로운 방법으로 분리하는 방법을 보여줍니다. 이 방법은 가이딩 모델로 **메인 모델의 열등한 버전**을 사용하기 때문에 task discrepancy 문제가 없습니다.

가이딩 모델은 단순히 모델 용량 및/또는 훈련 시간을 제한함으로써 얻을 수 있습니다.

#### 수식 표현

**CFG (Classifier-Free Guidance) 기본 수식:**

$$\tilde{D}_\theta(\mathbf{x}_\sigma; \sigma, c) = D_\theta(\mathbf{x}_\sigma; \sigma, \emptyset) + w \cdot \left[D_\theta(\mathbf{x}_\sigma; \sigma, c) - D_\theta(\mathbf{x}_\sigma; \sigma, \emptyset)\right]$$

여기서:
- $D_\theta$: denoiser network
- $\mathbf{x}_\sigma$: 노이즈가 추가된 입력
- $\sigma$: 노이즈 레벨
- $c$: 조건 (클래스 레이블, 텍스트 등)
- $\emptyset$: 비조건부
- $w$: guidance 강도

**Autoguidance 수식:**

Autoguidance는 추론 시 모델의 열등화된(degraded) 버전을 사용하여 diffusion 모델을 가이딩합니다. 메인 모델과 약한 모델 간의 불일치(discrepancy)가 교정 신호로 작용하여, 메인 모델이 더 약한 대응 모델로부터 얼마나 벗어나는지를 나타냅니다.

$$\tilde{D}(\mathbf{x}_\sigma; \sigma, c) = D_{\theta_{\text{main}}}(\mathbf{x}_\sigma; \sigma, c) + w \cdot \left[D_{\theta_{\text{main}}}(\mathbf{x}_\sigma; \sigma, c) - D_{\theta_{\text{bad}}}(\mathbf{x}_\sigma; \sigma, c)\right]$$

여기서:
- $D_{\theta_{\text{main}}}$: 잘 훈련된 메인 모델
- $D_{\theta_{\text{bad}}}$: 열등한 (작거나 덜 훈련된) 가이딩 모델
- $w$: guidance weight (일반적으로 $w > 0$)

**Score function 관점:**

$$\nabla_{\mathbf{x}} \log p(\mathbf{x}_\sigma; \sigma) \approx \frac{D_\theta(\mathbf{x}_\sigma; \sigma) - \mathbf{x}_\sigma}{\sigma^2}$$

Diffusion 모델은 노이즈가 추가된 샘플 $\mathbf{x}\_\sigma = \mathbf{x}\_0 + \sigma \mathbf{n}$을 denoise하도록 훈련됩니다. denoiser network $D_\theta(\mathbf{x}\_\sigma; \sigma, c)$는 주어진 노이즈 입력, 노이즈 레벨, 조건 $c$로부터 $\mathbf{x}\_0$을 예측하는 것을 학습하며, 이 훈련 목표는 score matching과 밀접하게 관련되어 있습니다. Score function 근사식은 다음과 같습니다: $\nabla_x \log p(x_\sigma; \sigma) \approx \frac{D_\theta(x_\sigma; \sigma) - x_\sigma}{\sigma^2}$.

**Autoguidance의 확장 수식 (다중 가이딩 모델):**

$$\tilde{D} = D_{\text{main}} + \sum_i w_i \cdot (D_{\text{main}} - D_{\text{bad},i})$$

autoguidance를 CFG와 결합하기 위해 여러 가이딩 모델을 포함하도록 수식을 확장하고, 선형 보간(linear interpolation)을 통해 전체 가이던스 가중치를 분배합니다.

---

### 🏗️ 2.3 모델 구조

#### 기반 모델: EDM2 (Elucidated Diffusion Model v2)

이 논문은 "Analyzing and Improving the Training Dynamics of Diffusion Models"(CVPR 2024 oral)과 함께 EDM2 아키텍처를 기반으로 합니다.

#### 가이딩 모델 생성 방식

두 가지 열화(degradation) 전략을 결합:

1. **용량 축소 (Reduced Capacity)**: 더 작은 크기의 모델 사용 (예: EDM2-XXL → EDM2-S)
2. **훈련 시간 축소 (Reduced Training)**: 더 이른 체크포인트(early snapshot) 사용

각 열화 효과를 개별적으로 측정한 결과, 가이딩 모델을 메인 모델과 동일한 용량으로 설정하고 더 짧은 시간만 훈련하면 FID가 1.51로 악화되고, 줄어든 용량의 가이딩 모델을 메인 모델만큼 오래 훈련하면 FID가 2.13으로 훨씬 더 악화됩니다. 따라서 **두 열화 전략 모두 이점이 있으며 상호 보완적**이지만, 개선의 대부분은 가이딩 모델의 **훈련 시간 단축에서 비롯**됩니다.

#### DeepFloyd IF 적용 방식

대규모 이미지 생성기의 맥락에서 본 방법을 연구하기 위해, DeepFloyd IF에 적용했습니다. DeepFloyd IF는 기본 모델과 두 개의 초해상도 단계로 이루어진 3개의 diffusion 모델 캐스케이드로 이미지를 생성합니다. Autoguidance는 기본 모델에만 적용되며, 후속 단계는 항상 CFG를 사용합니다.

---

### 📊 2.4 성능 향상

#### 조건부 ImageNet 생성 결과

결과를 보면, ImageNet-512에서 소형 모델(EDM2-S)을 사용한 autoguidance는 FID를 2.56에서 1.34로 개선합니다. 이는 동시에 제안된 CFG + Guidance Interval이 달성한 1.68보다 우수하며, 모델 크기에 관계없이 해당 데이터셋에서 보고된 최고의 결과입니다. 가장 큰 모델(EDM2-XXL)을 사용하면 기록이 1.25로 더욱 개선됩니다.

#### 비조건부 모델 성능 개선

Autoguidance는 비조건부 모델 성능을 FID 14.79에서 8.42로 대폭 감소시키면서 다양한 출력을 유지합니다.

EDM2-S는 비조건부 설정에서 FID 11.67을 달성하는데, 이는 사실상 생성된 이미지 중 발표할 수 있는 품질의 것이 없음을 나타냅니다. Autoguidance를 활성화하면 FID가 상당히 3.86으로 낮아지며, FDDINOv2의 개선도 마찬가지로 유의미합니다.

**성능 요약 표:**

| 데이터셋 | 기준 FID | Autoguidance FID | 개선율 |
|---|---|---|---|
| ImageNet-512 (EDM2-S) | 2.56 | 1.34 | ~48% ↓ |
| ImageNet-512 (EDM2-XXL) | - | **1.25** | SOTA |
| ImageNet-64 | - | **1.01** | SOTA |
| 비조건부 ImageNet-512 | 14.79 | 8.42 | ~43% ↓ |

---

### ⚠️ 2.5 한계 (Limitations)

미래 연구 방향으로는 autoguidance가 유익한 조건을 공식적으로 증명하고, 최적의 가이딩 모델 선택을 위한 실용적인 규칙을 도출하는 것이 있습니다. 초기 스냅샷 + 소형 모델이라는 제안은 원칙적으로는 충족하기 쉽지만, **현재 대규모 이미지 생성기의 경우 실제로는 이용 가능하지 않은 경우가 많습니다**. 또한 그런 생성기들은 종종 훈련 데이터가 중간에 변하는 연속 단계들로 훈련되어, 스냅샷들 사이에 분포 이동(distribution shift)이 발생할 수 있으며, 이는 본 방법의 가정을 위반합니다.

추가적인 한계:

합성 열화로 가이딩 모델을 메인 모델로부터 유도하는 방법은 전혀 효과가 없었으며, 이는 가이딩 모델이 **메인 모델과 동일한 종류의 열화를 겪어야 한다**는 추가 증거를 제공합니다. 또한 메인 모델을 양자화한 경우, 이를 더 낮은 정밀도로 양자화하는 것은 유용한 가이딩 모델을 만들지 못했습니다.

Autoguidance의 실용적인 한계는 두 번째 보조 모델을 훈련, 저장, 로딩해야 한다는 점입니다. 이는 계산 및 저장 비용을 증가시키고, 훈련 및 배포 파이프라인을 복잡하게 만듭니다.

---

## 3. 모델의 일반화 성능 향상 가능성 🔬

### 3.1 핵심 메커니즘: 왜 일반화가 향상되는가?

CFG와 달리, AutoGuidance는 단일 모델 내의 서로 다른 훈련 단계를 활용합니다. 훈련 데이터에 과적합(overfit)하는 경향이 있는 완전히 훈련된 모델이, **더 큰 다양성을 보유한 부분적으로 훈련된 모델에 의해 균형을 잡히게 됩니다**. 이 접근법은 생성된 이미지의 변동성을 향상시키는 데 중요하며, 이는 생성 모델 성능 향상을 위한 근본적인 요건입니다.

이를 수식으로 표현하면, 메인 모델이 고확률 영역의 매니폴드에 과적합된 분포 $p_\text{main}(\mathbf{x})$를 학습할 때:

$$p_\text{guided}(\mathbf{x}) \propto \frac{p_\text{main}(\mathbf{x})^\alpha}{p_\text{bad}(\mathbf{x})^{\alpha-1}}, \quad \alpha > 1$$

이는 실제 데이터 분포 $p_\text{data}$에 더 가까운 분포를 산출하는 방향으로 작동합니다.

### 3.2 저확률 영역 처리 개선

Diffusion 모델은 저확률 영역을 커버하기 위한 충분한 훈련과 데이터가 부족하기 때문에, 이 영역에 해당하는 고품질 이미지를 생성하지 못한 것에 대해 페널티를 받습니다. 더 나은 생성 품질을 달성하기 위해 CFG와 같은 가이던스 전략이 샘플링 단계에서 고확률 영역으로 샘플을 안내할 수 있습니다.

Autoguidance는 메인 모델과 동일한 조건(conditioning) 하에 작동하므로, **task discrepancy 없이** 고확률 영역으로 샘플링을 집중시킵니다.

### 3.3 조건부·비조건부 양쪽 일반화

이 방법은 **조건부 및 비조건부 확산 모델 모두에 적용 가능**하며, 다양한 합성 및 실용적 테스트를 통해 검증되었습니다.

### 3.4 과적합 방지 관점

Autoguidance는 **다양성을 유지하면서 품질을 향상시키는 강력하고 신뢰할 수 있는 레버**입니다. 데이터 선택만으로는 넘어서기 어려운 견고한 베이스라인을 실험에서 설정합니다.

메인 모델의 '과적합된 분포'를 열등한 모델의 '더 분산된 분포'로 교정하는 방식은 일반화에 유리하게 작동합니다. 이는 앙상블(ensemble)의 다양성 효과와도 유사한 원리입니다.

### 3.5 Autoguidance 확장: AutoLoRA

AutoLoRA는 LoRA 모델에서 생성 이미지의 다양성을 높이고 데이터 편향을 줄일 수 있게 합니다. 오버핏된 모델이 낮은 품질이지만 더 큰 다양성을 가진 모델에 의해 컨디셔닝됨으로써 개선될 수 있다는 AutoGuidance의 일반적인 아이디어를 활용합니다. AutoLoRA는 LoRA로 조정하기 이전의 모델을 사용해 최종 LoRA 미세 조정 모델을 컨디셔닝합니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점 🔭

### 4.1 연구에 미치는 영향

#### (1) Guidance 패러다임의 전환

이 논문은 CFG가 이미지 품질을 개선하는 이유에 대한 새로운 통찰을 제공하고, 이 효과를 분리하는 새로운 방법(autoguidance)을 제시합니다. 이 방법은 가이딩 모델로 **변경되지 않은 조건부(conditioning)를 유지한 채 메인 모델의 열등한 버전**을 사용하기 때문에 task discrepancy 문제가 없습니다.

#### (2) 비조건부 생성 분야 새 가능성

Autoguidance의 특별한 강점은 비조건부 모델에도 적용할 수 있다는 점입니다. 조건부 ImageNet 생성은 포화 상태에 가까워지고 있을 수 있지만, **비조건부 결과는 놀라울 정도로 열악한 상태**로 남아 있습니다.

#### (3) 파생 연구: In-situ Autoguidance

Autoguidance의 실용적인 비용 없이 개념적 이점을 달성하기 위해 **In-situ Autoguidance**가 제안되었습니다. 이 방법은 어떠한 보조 모델도 필요하지 않으며, 대신 추론 과정의 각 단계에서 동적으로 메인 모델의 일시적인 "나쁜" 버전을 생성합니다.

#### (4) 다양한 도메인으로의 확장 가능성

Autoguidance 접근법은 더 정교한 열화 기법이나 합성 및 실제 데이터 열화를 결합한 하이브리드 모델을 포함하는 새로운 방법에 영감을 줄 수 있습니다. 또한 텍스트 또는 오디오 생성과 같은 다른 유형의 생성 작업에 대한 autoguidance의 함의는 연구를 위한 유망한 방향을 제시합니다.

추천 시스템(RS) 분야에서도 Diffusion 모델이 강력한 성능을 보이고 있으며, 반복적인 denoising 과정이 사용자-아이템 상호작용의 불균형을 증폭시킬 수 있습니다. 이를 해결하기 위한 Adaptive Autoguidance 기반의 공정성(fairness) 향상 방법이 탐색되고 있습니다.

---

### 4.2 앞으로 연구 시 고려할 점

#### ① 가이딩 모델 선택 이론화

실제로 별도로 훈련되거나 다른 반복 횟수로 훈련된 모델들은 정확도 차이뿐만 아니라 무작위 초기화, 훈련 데이터 셔플링 등에서도 차이가 납니다. 가이던스가 성공적이려면, **품질 격차가 이러한 무작위 효과보다 체계적인 밀도 확산을 압도할 만큼 충분히 커야 합니다**.

→ **열화 전략의 체계적 이론화**가 필요합니다.

#### ② 대규모 모델에서의 스냅샷 관리

초기 스냅샷 + 소형 모델 제안은 원칙적으로 충족하기 쉽지만, **현재 대규모 이미지 생성기에서는 실제로 이용 가능하지 않은 경우가 많습니다**. 이러한 생성기들은 종종 훈련 데이터가 중간에 변할 수 있는 연속 단계들로 훈련되어, 스냅샷들 간의 잠재적인 분포 이동이 본 방법의 가정을 위반할 수 있습니다.

→ **훈련 데이터 분포 이동에 강건한 가이딩 모델 설계** 연구가 필요합니다.

#### ③ 계산 비용 최적화

In-situ Autoguidance는 보조 구성 요소 없이 모델 자체에서 가이던스를 이끌어냅니다. 이 접근법은 추론 시간에 확률론적 순전파(stochastic forward pass)를 사용하여 동적으로 열등한 예측을 생성하고, 가이던스를 **추론 시간 자기 교정(inference-time self-correction)**의 한 형태로 재구성합니다.

→ 단일 모델 내에서 보조 모델 없이 Autoguidance를 모사하는 연구가 진행 중입니다.

#### ④ 이론적 형식화

이 논문은 autoguidance가 최적의 결과를 제공하는 특정 조건에 대한 이해를 위한 추가 탐구를 촉진합니다. 모델 용량, 훈련 기간, 가이딩 모델에 사용되는 열화 유형 간의 관계에 관한 **새로운 연구 방향**을 제시합니다.

#### ⑤ 모드 붕괴(Mode Collapse) 및 편향 위험 관리

저하된 모델을 가이던스에 사용할 때 발생하는 잠재적 위험이나 한계, 특히 **모드 붕괴나 의도치 않은 편향**에 관해 연구할 필요가 있습니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 📊 Guidance 방법 비교 표

| 방법 | 연도 | 핵심 아이디어 | 조건부 전용? | 추가 훈련 필요? | 다양성 유지 |
|---|---|---|---|---|---|
| **DDPM** (Ho et al.) | 2020 | 기본 확산 모델 | ❌ | - | ✅ |
| **Classifier Guidance** (Dhariwal & Nichol) | 2021 | 외부 분류기 그래디언트 활용 | ✅ | ✅ (분류기) | ⚠️ |
| **CFG** (Ho & Salimans) | 2021/22 | 조건부/비조건부 모델 결합 | ✅ | ✅ (비조건부) | ❌ |
| **SAG** (Hong et al.) | 2023 | Self-attention 맵 블러링 | ❌ | ❌ | ⚠️ |
| **PAG** (Ahn et al.) | 2024 | Self-attention 행렬 → Identity 치환 | ❌ | ❌ | ⚠️ |
| **Guidance Interval** (Kynkäänniemi et al.) | 2024 | 노이즈 레벨 제한 구간에만 적용 | ✅ | ❌ | ⚠️ |
| **Autoguidance** (Karras et al.) | 2024 | 열등한 자신으로 가이딩 | **❌** | ⚠️ (소형 모델) | ✅ |
| **In-situ Autoguidance** | 2025 | 내부 stochastic forward pass | **❌** | **❌** | ✅ |

### 5.1 CFG (Classifier-Free Guidance) – 2021/22

CFG에서는 조건부 및 비조건부 diffusion 모델을 공동으로 훈련하고, 결과적인 조건부 및 비조건부 점수 추정치를 결합하여 **classifier guidance를 사용하여 얻은 것과 유사한 샘플 품질과 다양성 간의 트레이드오프**를 달성합니다.

→ Autoguidance는 CFG의 "unconditional 모델" 역할을 "열등한 자신"으로 대체하여 task discrepancy 문제를 해결합니다.

### 5.2 SAG (Self-Attention Guidance) – 2023

SAG의 주요 아이디어는 unconditional 모델을 **패치가 블러 처리된 조건부 모델**로 대체하는 것입니다. Self-attention 맵에 기반하여 활성화도가 높은 패치들이 블러 처리(adversarial blurring)됩니다.

→ SAG는 단일 모델로 가이던스를 적용하지만, 특정 attention 레이어 선택이 필요한 한계가 있습니다.

### 5.3 PAG (Perturbed-Attention Guidance) – 2024

PAG는 부정(negative) 모델에 대해 self-attention 맵을 identity 행렬로 교체합니다.

→ 추가 훈련 없이 조건부·비조건부 모델 모두 적용 가능하나, Autoguidance 대비 FID 성능은 낮습니다.

### 5.4 Guidance Interval – 2024 (동시 제안)

Autoguidance(EDM2-S, ImageNet-512)가 달성한 FID 1.34는 동시에 제안된 CFG + Guidance Interval이 달성한 1.68보다 **우수한 성과**입니다.

### 5.5 In-situ Autoguidance – 2025

Autoguidance는 잘 훈련된 모델을 별도로 훈련된 열등한 자신의 버전(더 작거나 더 적은 반복으로 훈련된)으로 안내합니다. 이는 다양성을 희생하지 않고 생성 충실도에서 새로운 기록을 세우면서 품질 개선과 프롬프트 정렬을 우아하게 분리합니다.

하지만 이 해결책은 보조 모델이 필요하다는 상당한 오버헤드를 도입합니다. In-situ Autoguidance는 어떠한 보조 구성 요소 없이 모델 자체로부터 가이던스를 이끌어냄으로써 이 전제조건에 도전합니다. 이 접근법은 확률론적 순전파(stochastic forward pass)를 사용하여 동적으로 열등한 예측을 생성하고, 가이던스를 추론 시간 자기 교정의 한 형태로 재구성합니다.

---

## 📚 참고 자료 (References)

1. **Karras, T., Aittala, M., Kynkäänniemi, T., Lehtinen, J., Aila, T., & Laine, S. (2024).** "Guiding a Diffusion Model with a Bad Version of Itself." *NeurIPS 2024 Oral.* https://arxiv.org/abs/2406.02507

2. **NVlabs/edm2 – Official PyTorch Implementation.** https://github.com/NVlabs/edm2

3. **OpenReview – Guiding a Diffusion Model with a Bad Version of Itself.** https://openreview.net/forum?id=bg6fVPVs3s

4. **NeurIPS 2024 Poster.** https://neurips.cc/virtual/2024/poster/94471

5. **Emergent Mind – Autoguidance Diffusion Models.** https://www.emergentmind.com/papers/2406.02507

6. **Ho, J. & Salimans, T. (2022).** "Classifier-Free Diffusion Guidance." *NeurIPS Workshop 2021 / arXiv:2207.12598.* https://arxiv.org/abs/2207.12598

7. **Gu, E. et al. (2025).** "In-situ Autoguidance: Eliciting Self-Correction in Diffusion Models." *ICML 2025.* https://arxiv.org/abs/2510.17136

8. **Kasymov, A. et al. (2024).** "AutoLoRA: AutoGuidance Meets Low-Rank Adaptation for Diffusion Models." https://arxiv.org/abs/2410.03941

9. **AI Summer – Overview of classifier-free diffusion guidance (Part 2).** https://theaisummer.com/classifier-free-guidance-part-2/

10. **Adaptive Autoguidance for Item-Side Fairness in Diffusion Recommender Systems.** https://arxiv.org/html/2602.14706

11. **Autoguided Online Data Curation for Diffusion Model Training.** https://arxiv.org/abs/2509.15267

12. **Ho, J. et al. (2020).** "Denoising Diffusion Probabilistic Models." https://arxiv.org/abs/2006.11239
