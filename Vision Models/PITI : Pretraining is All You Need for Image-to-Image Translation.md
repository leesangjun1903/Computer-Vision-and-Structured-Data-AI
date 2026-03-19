# Pretraining is All You Need for Image-to-Image Translation

---

# 1. 핵심 주장 및 주요 기여 (요약)

PITI는 사전학습(pretraining)을 활용하여 범용적인 Image-to-Image(I2I) 변환을 강화하자는 제안으로, 각 I2I 변환 문제를 하나의 다운스트림 태스크로 간주하고, 사전학습된 확산 모델(diffusion model)을 다양한 I2I 변환에 적용하는 간결하고 범용적인 프레임워크를 소개합니다.

**주요 기여:**
1. 사전학습된 생성적 사전지식(generative prior)을 활용하여 모든 특정 I2I 문제를 다운스트림 태스크로 처리하며, 자연 이미지에 대한 사전학습된 지식을 통해 전례 없는 품질을 달성할 수 있음을 보여줍니다.
2. 확산 모델 학습에서 텍스처 합성을 강화하기 위한 적대적 학습(adversarial training)과 생성 품질을 향상시키기 위한 정규화된 가이던스 샘플링(normalized guidance sampling)을 제안합니다.
3. ADE20K, COCO-Stuff, DIODE 등의 도전적인 벤치마크에서 광범위한 실험을 통해 PITI가 전례 없는 사실감과 충실도를 가진 이미지를 합성할 수 있음을 보여줍니다.

---

# 2. 상세 분석

## 2.1 해결하고자 하는 문제

기존의 I2I 변환 방법들은 전용 아키텍처 설계가 필요하고, 개별 변환 모델을 처음부터(from scratch) 학습해야 하며, 특히 쌍(pair)으로 된 학습 데이터가 충분하지 않을 때 복잡한 장면의 고품질 생성에 어려움을 겪습니다.

기존 방법들은 서로 다른 태스크를 개별적으로 처리하며, 제한된 태스크별 학습 데이터만으로 처음부터 학습해야 합니다.

핵심 문제는 다음과 같습니다:
- **데이터 부족**: 쌍으로 된(paired) 학습 데이터의 부족
- **태스크 특화 설계**: 각 태스크마다 개별적으로 모델 아키텍처를 설계해야 하는 비효율성
- **일반화 부족**: 개별 학습으로 인해 자연 이미지의 복잡한 의미구조를 충분히 학습하지 못함

## 2.2 제안하는 방법 (수식 포함)

### (A) 사전학습된 확산 모델의 활용

PITI는 GLIDE와 같은 대규모 텍스트-이미지 확산 모델을 사전학습 백본으로 사용합니다. 확산 모델의 기본 프레임워크는 다음과 같습니다:

**Forward Process (전방 확산):**

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\, x_{t-1}, \beta_t \mathbf{I})$$

여기서 $\beta_t$는 시간 단계 $t$에서의 노이즈 스케줄이며, 이를 누적하면:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}\, x_0, (1-\bar{\alpha}_t)\mathbf{I})$$

여기서 $\bar{\alpha}\_t = \prod_{s=1}^{t}(1-\beta_s)$ 입니다.

**Reverse Process (역방향 생성):**

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 \mathbf{I})$$

학습 목표는 다음의 단순화된 디노이징 목표 함수입니다:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0, \epsilon, t}\left[\| \epsilon - \epsilon_\theta(x_t, t) \|^2\right]$$

### (B) 태스크 적응 (Task Adaptation)

PITI의 프레임워크는 두 단계로 구성됩니다. 첫 번째 단계는 확산 모델을 이용한 다양한 I2I 변환 태스크에 대한 사전학습이며, 두 번째 단계는 다운스트림 태스크에 대한 미세조정(fine-tuning)입니다.

조건부 입력 $c$(예: 시맨틱 마스크, 스케치)를 받아 타겟 이미지 $x_0$를 생성하는 조건부 모델로 변환합니다:

$$\mathcal{L}_{\text{task}} = \mathbb{E}_{x_0, \epsilon, t, c}\left[\| \epsilon - \epsilon_\theta(x_t, t, f(c)) \|^2\right]$$

여기서 $f(\cdot)$는 조건부 입력 $c$를 사전학습된 모델의 중간 표현 공간으로 매핑하는 인코더 함수입니다.

### (C) 적대적 확산 업샘플러 (Adversarial Diffusion Upsampler)

생성 품질을 개선하기 위해 PITI는 확산 모델 학습에서 텍스처 합성을 강화하는 적대적 학습과 정규화된 가이던스 샘플링을 사용합니다.

업샘플 단계에서 적대적 손실을 추가합니다:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{simple}} + \lambda_{\text{adv}} \mathcal{L}_{\text{adv}}$$

여기서 $\mathcal{L}_{\text{adv}}$는 판별기(discriminator) $D$를 활용한 적대적 손실입니다:

$$\mathcal{L}_{\text{adv}} = -\mathbb{E}[\log D(\hat{x}_0)]$$

이 적대적 학습은 특히 업샘플링 네트워크의 텍스처 세부사항을 날카롭게 만들어 확산 모델이 본래 가지는 블러링(blurring) 경향을 보완합니다.

### (D) 정규화된 Classifier-Free Guidance

기존의 classifier-free guidance는:

$$\hat{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \varnothing) + s \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing))$$

여기서 $s$는 가이던스 스케일입니다. PITI는 이를 **정규화(normalized)**하여 사용합니다:

$$\hat{\epsilon}_\theta^{\text{norm}} = \epsilon_\theta(x_t, \varnothing) + s \cdot \frac{\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing)}{\|\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing)\|}  \cdot \|\epsilon_\theta(x_t, \varnothing)\|$$

이 정규화를 통해 가이던스의 크기(magnitude)를 안정화하여 과도한 채도나 아티팩트 없이 더 나은 생성 품질을 달성합니다.

## 2.3 모델 구조

PITI는 **2단계 캐스케이드 구조**를 사용합니다:

| 단계 | 해상도 | 역할 |
|------|--------|------|
| **Base Model** | $64 \times 64$ | 조건부 입력으로부터 저해상도 이미지 생성 |
| **Upsampler** | $64 \rightarrow 256$ | 적대적 학습이 결합된 확산 업샘플링 |

모델은 COCO 데이터셋에서 학습되었으며, Base-64×64 모델과 Upsample-64-256 모델이 Mask-to-Image 및 Sketch-to-Image 태스크에 대해 각각 제공됩니다.

- **백본**: GLIDE 기반의 U-Net 아키텍처 (사전학습된 텍스트-이미지 확산 모델)
- **조건부 인코더**: 시맨틱 맵/스케치 등의 입력을 사전학습된 모델의 중간 표현 공간으로 변환
- **판별기**: 업샘플링 단계에서 텍스처 품질 향상을 위한 보조 네트워크

## 2.4 성능 향상

PITI는 간단하고 범용적인 프레임워크로 사전학습의 힘을 다양한 I2I 변환 태스크에 가져오며, 적대적 확산 업샘플러와 정규화된 classifier-free guidance 같은 기법으로 강화되어 특히 도전적인 시나리오에서 최첨단(SOTA) 합성 품질을 크게 향상시켰습니다.

주요 성능 결과:
- **ADE20K**: 기존 SOTA (OASIS, SPADE 등) 대비 FID 대폭 개선
- **COCO-Stuff**: Mask-to-Image, Sketch-to-Image 태스크에서 우수한 사실감
- **DIODE**: Depth-to-Image 변환에서의 높은 충실도

## 2.5 한계점

본 방법의 한계는 샘플링된 이미지가 주어진 입력과 충실하게 정렬되기 어려우며 작은 객체를 놓칠 수 있다는 점입니다. 한 가지 가능한 이유는 사전학습된 모델의 중간 표현 공간이 정확한 공간적(spatial) 정보를 부족하게 가지고 있기 때문입니다.

추가 한계점:
- 확산 모델의 본질적인 **추론 속도 저하** (반복적 디노이징 과정)
- 대규모 사전학습 모델에 의존하므로 **컴퓨팅 비용**이 높음
- 입력 조건과의 세밀한 공간적 일치(pixel-level alignment)가 부족할 수 있음

---

# 3. 일반화 성능 향상 가능성

PITI의 핵심 철학은 바로 **일반화(generalization)**에 있습니다:

### 3.1 사전학습을 통한 일반화

핵심 아이디어는 사전학습된 신경망을 사용하여 자연 이미지 매니폴드를 캡처하고, 이미지 변환을 이 매니폴드를 탐색하여 입력 시맨틱과 관련된 실현 가능한 포인트를 찾는 것과 동등하게 보는 것입니다.

이를 수식으로 표현하면:

$$x^* = \arg\min_{x \in \mathcal{M}} \mathcal{D}(x, G(c))$$

여기서 $\mathcal{M}$은 사전학습을 통해 학습된 자연 이미지 매니폴드, $G(c)$는 조건부 입력 $c$로부터의 생성, $\mathcal{D}$는 거리 함수입니다.

### 3.2 일반화를 가능하게 하는 메커니즘

| 메커니즘 | 일반화 기여 |
|----------|-----------|
| **대규모 사전학습** | 수억 장의 이미지-텍스트 쌍으로 학습된 풍부한 시각적 사전지식 |
| **태스크 어댑테이션** | 소량의 태스크 특화 데이터로도 새로운 태스크에 빠르게 적응 |
| **정규화된 가이던스** | 과적합 방지 및 다양한 입력에 대한 안정적 생성 |
| **적대적 학습** | 텍스처 세부사항의 사실감 향상으로 도메인 간 전이 강화 |

### 3.3 few-shot/zero-shot 시나리오에서의 일반화

사전학습된 확산 모델이 이미 자연 이미지의 광범위한 분포를 학습했기 때문에, 소량의 쌍 데이터만으로도 다양한 도메인에서 높은 품질의 I2I 변환이 가능합니다. 이는 기존 GAN 기반 방법(pix2pix, SPADE 등)이 대량의 쌍 데이터에 의존하는 것과 대조됩니다.

---

# 4. 향후 연구에 미치는 영향 및 고려사항

## 4.1 연구 영향

저자들은 이 연구가 이 경로를 따라 더 많은 작업을 영감을 주고, 사실적인 합성(realistic synthesis) 분야를 발전시키기를 희망합니다.

PITI가 열어놓은 연구 방향:

1. **"사전학습 → 미세조정" 패러다임의 확산**: NLP에서 성공한 이 패러다임이 생성 모델에도 본격적으로 적용될 수 있음을 입증
2. **조건부 제어 메커니즘의 발전**: ControlNet, T2I-Adapter 등 후속 연구에 직접적 영감을 제공
3. **통합 프레임워크에 대한 탐색**: 다양한 I2I 태스크를 하나의 프레임워크로 통합하는 연구 촉진

## 4.2 향후 연구 시 고려할 점

저자들은 향후 사전학습을 위한 다른 방법들을 탐구할 계획입니다.

| 고려 사항 | 설명 |
|-----------|------|
| **공간적 정밀도** | 사전학습 모델의 중간 표현에서 공간 정보 보존 방법 연구 |
| **추론 효율성** | Consistency Model, DDIM 등 빠른 샘플링 기법과의 결합 |
| **다중 모달 사전학습** | 텍스트-이미지 뿐만 아니라 더 다양한 모달리티의 사전학습 활용 |
| **소형 모델에의 적용** | 지식 증류(knowledge distillation)를 통한 경량화 연구 |
| **윤리적 측면** | 딥페이크 등 오용 가능성에 대한 안전장치 마련 |

---

# 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 방법론 | PITI와의 관계/차이점 |
|------|------|--------|---------------------|
| **PITI** (Wang et al.) | 2022 | 사전학습된 확산모델 + 적대적 업샘플러 | 기준 논문 |
| **ControlNet** (Zhang & Agrawala) | 2023 | 확산 모델에 조건을 추가하는 신경망 구조로, 네트워크 블록의 가중치를 "잠금(locked)" 사본과 "학습 가능(trainable)" 사본으로 복사하여, 학습 가능 사본이 조건을 학습하고 잠금 사본이 원래 모델을 보존합니다. PITI보다 더 세밀한 공간적 제어가 가능하며, zero-convolution으로 원본 모델을 안전하게 보존합니다. |
| **InstructPix2Pix** (Brooks et al.) | 2023 | 텍스트 기반 이미지 편집 지시를 따르는 생성 모델 학습 방법으로, 입력 캡션/편집 지시/편집된 캡션 생성 후 Prompt-to-Prompt으로 이미지 쌍 데이터셋을 만들어 모델을 훈련합니다. PITI가 시맨틱 맵/스케치 조건에 집중한 반면, 텍스트 지시에 초점합니다. |
| **Pix2Pix-Zero** (Parmar et al.) | 2023 | 입력 이미지의 콘텐츠를 수동 프롬프트 없이 보존하는 I2I 변환 방법으로, 텍스트 임베딩 공간에서 원하는 편집 방향을 자동 발견하고, cross-attention 가이던스를 통해 콘텐츠 구조를 보존합니다. Zero-shot 방식으로 PITI의 미세조정 단계를 제거합니다. |
| **Palette** (Saharia et al.) | 2022 | 조건부 확산 모델로 다양한 I2I 태스크 통합 | PITI와 유사하게 확산 모델 기반이나, 대규모 사전학습 활용 없이 직접 학습합니다. |
| **SDEdit** (Meng et al.) | 2022 | 노이징과 디노이징 절차를 수행하여 구조적 정보를 유지하면서 세부 사항을 변경합니다. 학습 없이(training-free) 작동하나 세밀한 제어가 어렵습니다. |
| **BBDM** (Li et al.) | 2023 | I2I 변환을 확률적 브라운 브릿지(Brownian Bridge) 과정으로 모델링하며, 조건부 생성 과정이 아닌 양방향 확산 과정으로 두 도메인 간 변환을 직접 학습합니다. 이론적으로 더 우아하지만 사전학습 활용은 부족합니다. |
| **One-Step I2I** (Parmar et al.) | 2024 | 단일 스텝 확산 모델을 적대적 학습 목표를 통해 새로운 태스크와 도메인에 적응시키는 일반적 방법으로, 바닐라 잠재 확산 모델의 다양한 모듈을 단일 end-to-end 생성기 네트워크로 통합하여 입력 이미지 구조를 보존하면서 과적합을 줄입니다. PITI의 느린 추론 문제를 해결합니다. |
| **SADM** (Yang et al.) | 2024 | 구조 판별기에 대해 확산 생성기를 미니맥스 게임으로 적대적 학습하여, 각 학습 배치 내 샘플 간 매니폴드 구조를 학습하도록 합니다. ImageNet에서 256×256 해상도의 클래스 조건부 이미지 생성에서 FID 1.58의 새로운 SOTA를 달성합니다. PITI의 적대적 학습 아이디어를 구조적 수준으로 확장합니다. |

### 패러다임 변화 흐름도

```
PITI (2022)                    ControlNet (2023)              One-Step I2I (2024)
[사전학습 + 미세조정]    →    [어댑터 기반 제어]        →    [단일 스텝 추론]
    ↓                              ↓                              ↓
 적대적 업샘플링              zero-convolution              적대적 후처리 학습
 정규화 가이던스              공간적 조건 보존              속도 + 품질 동시 달성
```

---

# 참고자료

1. **Wang, T., Zhang, T., Zhang, B., Ouyang, H., Chen, D., Chen, Q., & Wen, F.** (2022). "Pretraining is All You Need for Image-to-Image Translation." *arXiv:2205.12952*. [https://arxiv.org/abs/2205.12952](https://arxiv.org/abs/2205.12952)
2. **PITI Project Page**: [https://tengfei-wang.github.io/PITI/index.html](https://tengfei-wang.github.io/PITI/index.html)
3. **PITI GitHub Repository**: [https://github.com/PITI-Synthesis/PITI](https://github.com/PITI-Synthesis/PITI)
4. **ResearchGate – PITI 논문 전문**: [https://www.researchgate.net/publication/360859547](https://www.researchgate.net/publication/360859547)
5. **Zhang, L. & Agrawala, M.** (2023). "Adding Conditional Control to Text-to-Image Diffusion Models" (ControlNet). *ICCV 2023*. [https://github.com/lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet)
6. **Brooks, T. et al.** (2023). "InstructPix2Pix: Learning to Follow Image Editing Instructions." [https://www.researchgate.net/publication/373317131](https://www.researchgate.net/publication/373317131)
7. **Michael X.** (2023). "Diffusion Models for Image-to-Image and Segmentation." *Medium*. [https://medium.com/@myschang/diffusion-models-for-image-to-image-and-segmentation-d30468114b27](https://medium.com/@myschang/diffusion-models-for-image-to-image-and-segmentation-d30468114b27)
8. **Yang, L. et al.** (2024). "Structure-Guided Adversarial Training of Diffusion Models" (SADM). *CVPR 2024*. [https://arxiv.org/html/2402.17563v2](https://arxiv.org/html/2402.17563v2)
9. **Parmar et al.** "One-Step Image Translation with Text-to-Image Models." *Semantic Scholar*. [https://www.semanticscholar.org/paper/b12bfd15bb9ef0fe2f7067f5b3033bc0f24468b9](https://www.semanticscholar.org/paper/b12bfd15bb9ef0fe2f7067f5b3033bc0f24468b9)
10. **Lilian Weng.** "What are Diffusion Models?" *Lil'Log*. [https://lilianweng.github.io/posts/2021-07-11-diffusion-models/](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

> **참고**: 위 수식들은 논문의 핵심 아이디어를 기반으로 한 표준적 확산 모델 수식의 정리입니다. 정규화된 classifier-free guidance 수식은 논문에서 제안한 방법을 수학적으로 표현한 것이며, 세부 구현은 원 논문 및 코드를 직접 참조하시기를 권장합니다.
