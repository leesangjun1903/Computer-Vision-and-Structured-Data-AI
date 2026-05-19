
# Revelio: Interpreting and Leveraging Semantic Information in Diffusion Models

> **논문 정보:**
> - **저자:** Dahye Kim\*, Xavier Thomas\*, Deepti Ghadiyaram (Boston University, Runway)
> - **발표 학회:** ICCV, 2025
> - **arXiv ID:** [2411.16725](https://arxiv.org/abs/2411.16725) (2024년 11월 제출)
> - **코드:** https://github.com/revelio-diffusion/revelio

---

## 1. 핵심 주장 및 주요 기여 요약

### 🔑 핵심 주장

이 논문은 다양한 diffusion 아키텍처의 여러 레이어와 denoising 타임스텝에서 풍부한 시각적 의미 정보가 어떻게 표현되는지를 연구한다. 저자들은 k-sparse autoencoders (k-SAE)를 활용해 단일 의미(monosemantic)의 해석 가능한 특징(feature)을 발견하고, 경량 분류기를 통한 전이 학습으로 이 해석을 검증한다.

생성 모델이 시각 세계를 정확히 시뮬레이션하려면 잠재 공간(latent space)이 풍부한 시각적 의미(semantics)를 포착해야 하며, 실제로 탐지(detection), 분할(segmentation), 분류(classification), 의미 대응(semantic correspondence), 깊이 추정(depth estimation) 등 식별 태스크에서 diffusion 특징을 활용하는 시도가 증가하고 있지만, 이러한 풍부한 의미 정보가 모델 내에서 어떻게 표현되는지에 대한 명확한 통찰은 부족했다.

### 🏆 주요 기여

1. 저자들은 기존의 추가 loss, 학생 모델 증류(distillation), 또는 피처 맵 융합 방법 없이 off-the-shelf diffusion 모델의 특징 위에 매우 경량의 분류기(Diff-C)를 학습하여, 기존 방법 대비 **4 orders of magnitude(4자릿수)의 추론 속도 향상**을 달성한다.

2. Diff-C는 diffusion 특징을 표현 학습에 활용한 모든 이전 연구 대비 최고 성능을 달성하면서, 강력한 자기 지도 시각 모델(DINO) 및 멀티모달 모델(CLIP)과도 경쟁력 있는 성능을 보인다.

3. 다양한 diffusion 아키텍처, 사전학습 데이터셋, 언어 모델 컨디셔닝이 시각 표현 세분성(granularity), 귀납적 편향(inductive biases), 전이 학습 능력에 미치는 영향을 심층 분석함으로써, **블랙박스 diffusion 모델의 해석 가능성을 심화하는 중요한 첫 단계**를 제시한다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### ❓ 해결하고자 하는 문제

모델이 시각 정보를 어떻게 학습하는지 이해하는 것은 여러 중요한 이점을 제공한다. 현재 시각 생성 모델은 본질적으로 블랙박스이며, 무해한 프롬프트가 때로 안전하지 않은 출력을 생성하거나 동일한 프롬프트의 미세한 변형이 매우 다른 출력을 내는 이유가 불분명하다.

둘째, 다양한 레이어, 타임스텝, 모델 아키텍처에 걸쳐 표현된 의미 정보의 세분성을 파악하는 것은 의미 및 스타일 제어를 가능하게 하는 보다 효율적인 알고리즘 설계에 도움이 된다. 이를 위해 저자들은 "기계론적 해석(mechanistic interpretation)" 기법을 채택하여 단일 의미의 시각 개념으로 이루어진 희소 사전(sparse dictionary)을 학습한다.

---

### 🔬 제안하는 방법 및 수식

#### (1) Diffusion 모델 기본 수식

저자들은 diffusion 모델을 $T$개의 타임스텝에 걸쳐 노이즈 교란 과정을 통해 데이터 분포 $p(x)$를 학습하는 확률론적 생성 모델로 정의한다. 순방향 과정(forward process)은 입력 이미지에 점진적으로 노이즈를 추가하고, 역방향 과정(reverse process)은 이를 반복적으로 제거한다. 목적 함수는 각 타임스텝에서 실제 입력과 예측된 노이즈 간의 재구성 오차를 최소화하는 것이다.

**순방향 과정 (Forward Process):**

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t \mathbf{I})$$

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I})$$

여기서 $\bar{\alpha}\_t = \prod_{s=1}^{t}(1 - \beta_s)$이고, $\beta_t$는 노이즈 스케줄이다.

**학습 목적 함수 (Training Objective):**

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \left\| \epsilon - \epsilon_\theta(x_t, t) \right\|^2 \right]$$

---

#### (2) k-Sparse Autoencoder (k-SAE)

인간의 시각 시스템이 소수의 기저 함수를 이용해 가장 반복적인 시각 패턴을 희소하게 인코딩한다는 생리학적 증거에 착안하여, 저자들은 언어 모델 해석에 활용된 k-sparse autoencoders (k-SAE)를 통해 해석 가능한 특징을 발굴한다.

k-SAE는 diffusion 모델의 중간 레이어에서 추출된 특징 벡터 $\mathbf{h} \in \mathbb{R}^d$를 입력으로 받아 다음과 같이 동작한다:

**인코더 (Top-k 활성화):**

$$\mathbf{z} = \text{TopK}(W_e \mathbf{h} + \mathbf{b}_e)$$

- $W_e \in \mathbb{R}^{n \times d}$: 인코더 가중치 행렬
- $\text{TopK}(\cdot)$: 상위 $k$개의 활성값만 유지하고 나머지는 0으로 설정하는 연산 (희소성 부여)

**디코더 (재구성):**

$$\hat{\mathbf{h}} = W_d \mathbf{z} + \mathbf{b}_d$$

**손실 함수:**

$$\mathcal{L}_{\text{k-SAE}} = \|\mathbf{h} - \hat{\mathbf{h}}\|_2^2 + \lambda \|\mathbf{z}\|_1$$

- 재구성 오차(Reconstruction error) + $\ell_1$ 정규화 항으로 희소 표현 강제

---

#### (3) Diff-C: 경량 분류기

저자들은 off-the-shelf diffusion 모델의 특징 위에 매우 경량의 분류기 **Diff-C**를 학습함으로써, 추가적인 loss, 학생 모델 학습, 피처 맵 융합 방법 없이 다양한 태스크에서 diffusion 특징의 놀라운 효과를 보인다. 기존 방법 대비 **추론 속도를 4 orders of magnitude** 향상시킨다.

Diff-C의 분류 목표 함수는 다음과 같이 정의할 수 있다:

```math
\hat{y} = f_\phi \left( \phi_{\text{diff}}(x, t^*, l^*) \right)
```

여기서:
- $\phi\_{\text{diff}}(x, t^\*, l^\*)$: 타임스텝 $t^\*$, 레이어 $l^*$에서 추출된 diffusion 특징
- $f_\phi$: 경량 분류기 (linear probe 또는 소규모 MLP)
- $t^\*, l^\*$: 각 데이터셋에 대해 최적의 타임스텝과 레이어를 선택

학습에는 AdamW 옵티마이저, 학습률 $1 \times 10^{-4}$를 사용하며, 코사인 어닐링 학습률 스케줄로 30 에포크 학습한다. 입력 이미지는 $512 \times 512$로 랜덤 크롭 및 리사이즈하고 랜덤 수평 플립으로 증강한다.

---

### 🏗️ 모델 구조 분석

표현 세분성(representation granularity)은 모델 깊이에 따라 비선형적으로 변하며, 서로 다른 diffusion 레이어가 거친(coarse) 형태, 질감, 색상 패턴부터 세밀한 동물 품종 세부 사항, 나아가 카메라 각도나 객체 자세 같은 전역적 시각 개념까지 다양한 수준의 시각 의미 정보를 포착한다.

표현 세분성과 일반화 능력은 diffusion 아키텍처, 사전학습 데이터, 잠재(latent) 또는 픽셀 공간, cross/self-attention 메커니즘에 따라 달라지며, 이는 전반적인 픽셀 생성 품질과 학습 효율성 향상을 위해 이루어진 설계 선택들의 결과이다.

레이어별 역할:
- **Bottleneck 레이어:** 배경 대비 객체 위치 등의 매우 거친(coarse) 패턴을 분리한다.
- **up_ft1 레이어:** 명확한 클래스 특이적 특징이 관찰되어 세밀한 품종 등을 구별하는 데 도움을 준다.
- **up_ft2 레이어:** up_ft2 이후 레이어에서는 성능이 급격히 감소하며, 이는 해당 레이어의 특징이 이미지 생성을 위한 픽셀 재구성이라는 사전학습 목적에 더 정렬되어 전이 학습에 덜 일반화됨을 시사한다.

---

### 📊 성능

저자들은 경량 분류기를 통한 전이 학습으로 기계론적 해석을 검증하며, 4개의 데이터셋에서 표현 학습에 대한 diffusion 특징의 효과성을 입증한다.

diffusion 특징을 사용하는 모델 중 **Diff-C가 최고 성능**을 보이며, 텍스트 컨디셔닝을 사용하는 CLIP의 zero-shot 성능과도 경쟁력 있는 결과를 보인다.

---

### ⚠️ 한계

up_ft2 레이어에서 성능이 급격히 감소하는데, 이는 해당 레이어 특징이 이미지 생성을 위한 픽셀 재구성 목적에 더 정렬되어 전이 학습에 덜 일반화되기 때문이다. 이는 언어 모델을 기계론적으로 해석할 때 후반 레이어에 대해 유사한 관찰이 이루어진 것과 일치한다.

또한, 논문이 주로 **SD 1.5**, **SD 2.1**, **DiT** 등 특정 아키텍처에 집중되어 있어, 최신 대형 Diffusion Transformer 모델(SD3, FLUX 등)에 대한 분석은 추가 연구가 필요하다.

---

## 3. 일반화 성능 향상 가능성

저자들은 diffusion 특징의 일반화 가능성을 명시적으로 연구한다.

표현 세분성과 일반화 가능성은 diffusion 아키텍처, 사전학습 데이터, 잠재(latent) 또는 픽셀 공간, cross/self-attention 메커니즘 등 전반적인 픽셀 생성 품질과 학습 효율성을 개선하기 위해 이루어진 설계 선택들에 따라 달라진다.

핵심적인 일반화 관련 발견:

1. **타임스텝과 일반화의 관계:**
Caltech-101 데이터셋에서는 $t=200$에서 추출된 특징이 가장 낮은 $\sigma_{\text{label}}$을 산출하며, 이 결과는 Diff-C 결과와도 일치한다. 저자들은 $t=200$에서 추가된 노이즈가 특징을 더 일반화 가능하게 만드는 데 도움이 될 수 있다고 가설을 세운다.

2. **레이어 선택과 일반화:**
up_ft2 레이어 이후에는 성능이 급격히 감소하며, 이는 해당 레이어 특징이 픽셀 재구성 목적에 더 정렬되어 전이 학습에 덜 일반화됨을 시사한다.

3. **사전학습 데이터의 영향:**
다양한 diffusion 아키텍처, 사전학습 데이터셋, 언어 모델 컨디셔닝이 시각 표현 세분성, 귀납적 편향, 전이 학습 능력에 어떤 영향을 미치는지에 대한 심층 분석을 제공한다.

4. **k-SAE의 역할:**
k-sparse autoencoder는 모델 상태 전반에 걸쳐 단일 의미(monosemantic)의 시각적 속성을 체계적으로 분리하여 블랙박스 diffusion 모델을 해석하는 데 도움을 준다.

이러한 인사이트는 최적의 레이어와 타임스텝을 선택하는 가이드라인을 제공함으로써, 다운스트림 태스크에서의 일반화 성능 향상에 직접적으로 기여한다.

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 🔭 미치는 영향

| 영역 | 영향 |
|------|------|
| **모델 해석 가능성 (XAI)** | Diffusion 모델의 블랙박스 문제를 기계론적 접근으로 해소하는 새로운 패러다임 제시 |
| **표현 학습** | Generative 모델의 특징을 Discriminative 태스크에 효율적으로 활용하는 Diff-C 방법론 정립 |
| **안전성 및 제어** | 특정 레이어/타임스텝의 의미 정보 파악을 통해 편향(bias), 유해 콘텐츠 제어에 활용 가능 |
| **아키텍처 설계** | 어떤 설계 선택이 전이 학습 성능을 향상시키는지에 대한 empirical 근거 제공 |

고품질의 사실적이고 창의적인 시각 콘텐츠를 생성하는 diffusion 모델 연구는 급성장하는 분야이며, 생성 모델이 시각 세계를 정확히 시뮬레이션하려면 그 잠재 공간이 풍부한 시각적 의미와 실제 세계의 물리적 동역학을 포착해야 한다.

### 📌 향후 연구 시 고려할 점

1. **더 다양한 아키텍처 커버리지:**
   향후 연구는 다양한 diffusion 아키텍처, 사전학습 데이터셋, 언어 모델 컨디셔닝이 시각 표현 세분성, 귀납적 편향, 전이 학습 능력에 미치는 영향을 더욱 심층적으로 분석할 필요가 있다. 특히 최신 아키텍처(FLUX, SD3 등)로의 확장이 필요하다.

2. **타임스텝 선택의 이론적 근거:**
   특정 타임스텝에서 추가된 노이즈가 특징을 더 일반화 가능하게 만드는 데 도움이 될 수 있다는 가설에 대한 이론적 근거 확립이 필요하다.

3. **도메인 특수 적용:**
   Diffusion 모델은 이미지 및 비디오 합성을 넘어 제로샷 분류, 탐지, 분할, 의미 대응, 깊이 추정 등 다양한 분야에서 활용되고 있으며, 의료 영상, 위성 이미지 등 특수 도메인에서의 일반화 검증이 요구된다.

4. **언어-비전 상호작용 분석:**
   다양한 언어 모델 컨디셔닝이 시각 표현에 미치는 영향에 대한 분석을 바탕으로, 텍스트-이미지 정렬 메커니즘을 더 깊이 이해하는 연구가 필요하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 방법 | Revelio와의 차이 |
|------|------|------|-----------------|
| **DDPM** (Ho et al.) | 2020 | Denoising Diffusion Probabilistic Model 기반 생성 | Revelio는 이를 특징 추출기로 재활용 |
| **Label-efficient Semantic Segmentation** (Baranchuk et al.) | 2021 | Diffusion 특징으로 세분화 | 특정 태스크에 한정, 해석 불포함 |
| **DINOv2** (Oquab et al.) | 2023 | 자기 지도 학습 기반 표현 학습 | Revelio의 Diff-C와 경쟁적 성능 비교 대상 |
| **CLIP** (Radford et al.) | 2021 | 멀티모달(text-image) 표현 학습 | Diff-C와 zero-shot 성능 비교 대상 |

기존 여러 연구들은 zero-shot 분류, 탐지, 분할, 의미 대응, 뷰 합성, 이미지 편집 등에 diffusion 특징을 활용해왔으나, Revelio는 이들과 두 가지 중요한 차별점이 있다. 첫째, distillation이나 고비용 hypernetwork 없이 diffusion 특징을 식별 태스크에 적용하는 간단한 방법을 제안하고, 둘째, 단순한 특징 활용을 넘어 시각 의미 정보가 어떻게 표현되는지를 해석한다.

---

## 📚 참고 자료 (출처)

1. **논문 원문 (arXiv):** Dahye Kim, Xavier Thomas, Deepti Ghadiyaram. *Revelio: Interpreting and leveraging semantic information in diffusion models.* arXiv:2411.16725, 2024. https://arxiv.org/abs/2411.16725

2. **논문 PDF (arXiv):** https://arxiv.org/pdf/2411.16725

3. **HTML 버전 (arXiv):** https://arxiv.org/html/2411.16725v1

4. **ICCV 2025 공식 논문 (CVF):** https://openaccess.thecvf.com/content/ICCV2025/papers/Kim_Revelio_Interpreting_and_leveraging_semantic_information_in_diffusion_models_ICCV_2025_paper.pdf

5. **저자 발표 페이지:** https://kim-dahye.github.io/publications/

6. **BibBase 논문 메타데이터:** https://bibbase.org/network/publication/kim-thomas-ghadiyaram-reveliointerpretingandleveragingsemanticinformationindiffusionmodels-2024

7. **Bytez 논문 요약:** https://bytez.com/docs/arxiv/2411.16725/paper

8. **PaperReading Club:** https://paperreading.club/page?id=268567

9. **Moonlight Literature Review:** https://www.themoonlight.io/en/review/textitrevelio-interpreting-and-leveraging-semantic-information-in-diffusion-models

> ⚠️ **정확도 주의사항:** k-SAE의 구체적 손실 함수 수식 및 Diff-C의 세부 하이퍼파라미터 일부는 논문 PDF 원문의 직접 접근에서 확인된 부분과 표준적인 k-SAE 문헌을 결합하여 구성하였습니다. 완전한 수식은 arXiv 원문(2411.16725)을 직접 확인하시길 권장합니다.
