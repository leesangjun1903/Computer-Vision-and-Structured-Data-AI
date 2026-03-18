# Patched Denoising Diffusion Models For High-Resolution Image Synthesis 

"Patched Denoising Diffusion Models For High-Resolution Image Synthesis" (Patch-DM)** 논문을 종합적으로 분석합니다. 이 논문은 Zheng Ding, Mengqi Zhang, Jiajun Wu, Zhuowen Tu 저자의 ICLR 2024 발표 논문입니다.

---

# 1. 핵심 주장 및 주요 기여 요약

Patch-DM은 작은 크기의 이미지 패치(예: $64 \times 64$)로 학습하여 고해상도 이미지(예: $1024 \times 512$)를 생성하는 효과적인 denoising diffusion model이다.

**핵심 기여:**

1. **Feature Collage 전략**: Feature collage는 인접 패치의 부분 특징(feature)을 체계적으로 잘라내고 결합하여 이동된(shifted) 이미지 패치의 특징을 예측함으로써, 패치 특징 공간에서의 중첩(overlap)을 통해 전체 이미지를 이음새 없이(seamless) 생성할 수 있도록 한다.

2. **State-of-the-Art 성능**: Patch-DM은 자체 수집한 자연 이미지 데이터셋($1024 \times 512$), LHQ($1024 \times 1024$), FFHQ($1024 \times 1024$), 그리고 LSUN-Bedroom, LSUN-Church, FFHQ($256 \times 256$) 등 총 6개 데이터셋에서 이전 패치 기반 생성 방법들을 능가하는 최고 수준의 FID 점수를 달성했다.

3. **메모리 효율성**: Patch-DM은 기존 전체 이미지 기반 확산 모델 대비 메모리 복잡도를 줄인다.

---

# 2. 상세 분석

## 2.1 해결하고자 하는 문제

생성적 확산 모델은 뛰어난 이미지 품질을 생성하지만, 직접적인 픽셀 공간 최적화와 다중 타임스텝 학습으로 인해 고해상도 이미지 생성으로의 확장에 어려움을 겪는다. 기존의 최첨단 접근법들은 super-resolution에 의존하거나 잠재 공간(latent space)에서 최적화하며, 고해상도 이미지 생성 시 대용량 메모리와 대규모 모델이 필요하다.

핵심 문제점:
- 전체 고해상도 이미지를 직접 모델링하면 GPU 메모리 소비가 급증
- 기존 패치 기반 접근법은 **경계 아티팩트(boundary artifact)** 문제 발생
- Super-resolution 기반 방법은 저해상도 → 고해상도의 2단계 파이프라인이 필요

## 2.2 제안 방법 (수식 포함)

### (a) DDPM 기본 프레임워크

Denoising Diffusion Probabilistic Model(DDPM)의 forward process는 다음과 같다:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\, x_{t-1},\, \beta_t \mathbf{I})$$

여기서 $\beta_t$는 노이즈 스케줄이며, 누적된 형태로:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}\, x_0,\, (1-\bar{\alpha}_t)\mathbf{I})$$

$$\bar{\alpha}_t = \prod_{s=1}^{t}(1-\beta_s)$$

학습 목표(loss)는:

$$L_{\text{simple}} = \mathbb{E}_{x_0, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

### (b) 패치 기반 학습

전체 이미지를 학습에 사용하는 대신, 패치만을 학습 및 추론에 사용하고, 제안한 feature collage 메커니즘으로 인접 패치의 부분 특징을 체계적으로 결합한다.

훈련 이미지 $x_0 \in \mathbb{R}^{C \times H \times W}$를 패치 그리드로 분할:

$$x_0^{(i,j)} \in \mathbb{R}^{C \times p \times p}, \quad i \in \{1,\dots, H/p\},\ j \in \{1,\dots, W/p\}$$

여기서 $p$는 패치 크기(논문에서는 $p = 64$).

### (c) Feature Collage 메커니즘

기존 픽셀 공간에서의 패치 콜라주 대신, 특징(feature) 공간에서 패치 콜라주를 수행한다. 특징 공간에서의 패치 콜라주는 더 깊은(in-depth) 수준이며 다중 레벨(multi-level) 상호작용을 지원한다.

구체적으로, U-Net 내부에서 인접 패치 특징의 일부를 잘라내어 결합:

```math
F_{\text{collage}}^{(i,j)} = \text{Crop\&Combine}\left(F^{(i-1,j)},\, F^{(i+1,j)},\, F^{(i,j-1)},\, F^{(i,j+1)},\, F^{(i,j)}\right)
```

이 결합된 특징은 shifted 패치 위치에 대한 예측을 수행하며, 슬라이딩 윈도우 기반의 shifted 이미지 패치 생성 프로세스를 구현하여 인접 이미지 패치 간의 일관성을 보장한다.

### (d) Global Condition

전역적 스타일 일관성을 위해 CLIP 사전 학습 모델에서 이미지 임베딩을 추출하고 학습 중 최적화하여 시맨틱 코드(semantic code)를 초기화한다. 비조건부 이미지 생성에서는 최적화된 시맨틱 코드 임베딩 공간에 대해 잠재 확산 모델(latent diffusion model)을 학습하여 무한한 새로운 전역 시맨틱 조건을 제공한다.

Global condition을 패치 수준 diffusion에 주입:

$$\epsilon_\theta(x_t^{(i,j)}, t, c_{\text{global}}, \text{pos}(i,j))$$

여기서 $c_{\text{global}}$은 CLIP 기반 전역 시맨틱 코드, $\text{pos}(i,j)$는 위치 임베딩이다.

## 2.3 모델 구조

Patch-DM의 전체 파이프라인은 다음 구성 요소로 이루어진다:

| 구성 요소 | 설명 |
|---|---|
| **Patch-level U-Net** | 패치 크기($64 \times 64$)에서 동작하는 denoising 네트워크 |
| **Feature Collage Module** | U-Net 내부에서 인접 패치 특징을 슬라이딩 윈도우로 결합 |
| **Semantic Encoder** | CLIP 기반 전역 조건 추출기 |
| **Latent DPM** | 시맨틱 코드 생성용 잠재 확산 모델 |
| **Position Embedding** | 패치의 공간적 위치를 인코딩 |

전체 Patch-DM 모델은 154M 파라미터로, ADM(552M)이나 SR3(625M)보다 상당히 작으며, 시맨틱 인코더나 잠재 DPM 없는 버전은 70M 파라미터까지 줄어든다.

## 2.4 성능 분석

### 정량적 성능

Patch-DM은 FFHQ, LSUN-Bedroom, LSUN-Church 등 $256 \times 256$ 이미지 합성 벤치마크에서 기존 생성 모델과 경쟁적인 성능을 보이며, FFHQ에서 FID 10.02, sFID 10.58을 달성하여 COCO-GAN(FID 34.02)이나 InfinityGAN(FID 28.87) 등 이전 패치 기반 방법을 크게 능가한다.

Patch-DM은 평가된 모든 $256 \times 256$ 데이터셋에서 FID, sFID, Precision, Recall 기준으로 이전 패치 기반 생성 방법을 일관되게 능가한다. 예컨대 LSUN-Bedroom에서 Patch-DM의 FID 6.04는 InfinityGAN의 10.71, Anyres-GAN의 15.65에 비해 크게 개선되었다.

### 한계점

Patch-DM이 전반적으로 우수한 성능을 보이지만, 모든 $256 \times 256$ 데이터셋에서 절대적인 최고 FID 점수를 일관되게 달성하지는 못한다. 예를 들어 LSUN-Bedroom에서 ADM은 FID 1.90을 달성하는 반면 Patch-DM은 6.04이며, LSUN-Church에서 LDM-8은 FID 4.23 대비 Patch-DM은 5.49이다. 이는 Patch-DM이 패치 기반 접근으로서는 매우 효과적이지만, 일부 전체 이미지 기반 확산 모델이 이들 벤치마크에서 여전히 약간의 우위를 점하고 있음을 시사한다.

---

# 3. 모델의 일반화 성능 향상 가능성

Patch-DM의 일반화 성능과 관련된 핵심 설계 요소는 다음과 같다:

### (1) 해상도 비의존적(Resolution-Agnostic) 설계

패치만을 학습 및 추론에 사용하므로, Patch-DM은 고해상도 이미지 생성에 수반되는 높은 계산 비용 문제를 해결할 수 있으며, 해상도에 비의존적(resolution-agnostic)이다.

이는 학습 시 본 적 없는 해상도로의 일반화를 가능하게 한다:

$$\text{학습: } p \times p \quad \Rightarrow \quad \text{추론: } (N_h \cdot p) \times (N_w \cdot p) \quad \text{(임의의 } N_h, N_w\text{)}$$

### (2) 학습 해상도 이상의 이미지 생성

Patch-DM은 추가 학습 없이 학습된 해상도보다 높은 해상도에서 효과적으로 이미지를 생성할 수 있다. 이는 패치 기반 접근법의 본질적인 장점으로, 패치 레벨에서의 분포를 학습하면 이를 타일링하여 임의 크기의 이미지를 생성할 수 있기 때문이다.

### (3) Image Outpainting & Inpainting

Patch-DM은 시각적 품질과 일관성을 유지하면서 이미지 아웃페인팅을 수행할 수 있으며, 별도의 학습 없이도 손상된 이미지에 대해 컨텍스트 일관성을 보장하는 이미지 인페인팅이 가능하다.

### (4) Feature Collage를 통한 일관성 보장

Feature collage 메커니즘은 인접 이미지 패치 간의 특징 공유와 일관성을 촉진하여 추가 파라미터 없이 경계 아티팩트를 완화하고 컴팩트한 패치 기반 모델을 가능하게 한다.

### (5) 일반화 향상을 위한 향후 방향

- **다양한 패치 크기 혼합 학습**: 다양한 스케일의 패치를 학습에 활용하면 멀티스케일 일반화 가능
- **텍스트 조건부 확장**: 현재 비조건부(unconditional) 생성 중심이므로, 텍스트 프롬프트 기반 조건부 생성으로 확장 시 범용성 대폭 향상
- **다른 도메인 적용**: 의료 영상, 위성 영상 등 고해상도가 필수인 도메인으로의 전이

---

# 4. 향후 연구에 미치는 영향 및 고려할 점

## 4.1 연구에 미치는 영향

1. **패치 기반 확산 패러다임 확립**: Patch-DM은 확산 모델에서 "전체 이미지가 아닌 패치 단위 학습"이라는 새로운 패러다임을 제시하였으며, 이후 후속 연구에 직접적으로 영향을 미쳤다. 이후 NeurIPS 2023에 발표된 Patch Diffusion 논문에서도 이를 패치 기반 denoising diffusion의 선도적 연구로 인용하며, "메모리 효율적인 고해상도 이미지 합성에서 뛰어나고 경계 아티팩트 도입을 회피한다"고 평가했다.

2. **비디오 생성으로의 확장**: CVPR 2024에 발표된 Hierarchical Patch Diffusion Models(HPDM)은 PDM을 두 가지 원칙적 방식으로 개선하였다. 첫째, 패치 간 일관성을 위해 deep context fusion을 개발하고, 둘째 학습 및 추론 가속을 위한 adaptive computation을 제안하여 UCF-101에서 FVD 66.32의 최고 성능을 달성했다.

3. **효율적 학습의 가능성 제시**: 패치 단위 학습은 학습 데이터 효율성과 메모리 효율성을 동시에 개선할 수 있음을 보여주어, 자원이 제한된 환경에서의 확산 모델 활용 가능성을 열었다.

## 4.2 향후 연구 시 고려할 점

| 고려 사항 | 설명 |
|---|---|
| **전역 일관성** | 패치 단위 생성 시 이미지 전체의 의미적 일관성(semantic coherence) 유지가 핵심 과제 |
| **조건부 생성 확장** | 텍스트-이미지, 레이아웃-이미지 등 조건부 생성 태스크로의 확장 필요 |
| **추론 속도** | 패치별 순차 생성으로 인한 추론 시간 증가 문제 해결 필요 |
| **3D/비디오 확장** | 패치 기반 접근을 3D 생성 및 비디오 합성으로 확장 |
| **Latent Diffusion과의 결합** | LDM의 잠재 공간에서 패치 기반 전략을 적용하면 추가적인 효율성 확보 가능 |

---

# 5. 2020년 이후 관련 최신 연구 비교 분석

| 모델 | 연도 | 학회 | 접근 방식 | 핵심 특징 | Patch-DM과의 비교 |
|---|---|---|---|---|---|
| **DDPM** (Ho et al.) | 2020 | NeurIPS | 전체 이미지 확산 | 확산 모델 기초 | Patch-DM의 기반 프레임워크 |
| **LDM** (Rombach et al.) | 2022 | CVPR | 잠재 공간 확산 | VAE 인코딩 후 확산 수행 | Patch-DM은 픽셀 공간에서 직접 동작하여 별도 인코더 불필요 |
| **ADM** (Dhariwal & Nichol) | 2021 | NeurIPS | 전체 이미지 확산 + classifier guidance | FID 성능 극대화 (552M params) | Patch-DM은 154M로 경량이나 일부 벤치마크에서 ADM에 뒤처짐 |
| **SDXL** (Podell et al.) | 2023 | ICLR 2024 | 잠재 확산 (대규모) | 이전 Stable Diffusion 대비 3배 큰 UNet 백본, 더 많은 attention 블록과 두 번째 텍스트 인코더 사용 | Patch-DM과 철학이 반대 — SDXL은 스케일업, Patch-DM은 효율화 |
| **Patch Diffusion** (NeurIPS 2023) | 2023 | NeurIPS | 다중 패치 크기 학습 | 패치 옵션을 사용하며, 학습 중 무작위로 패치 크기를 샘플링하여 작은 패치부터 큰 패치까지 조건부 score function을 학습 | Patch-DM과 보완적 — Patch Diffusion은 학습 효율성에 초점, Patch-DM은 고해상도 생성에 초점 |
| **HPDM** (Skorokhodov et al.) | 2024 | CVPR | 계층적 패치 확산 | 패치 간 일관성을 위한 deep context fusion과 학습/추론 가속을 위한 adaptive computation | Patch-DM의 직접적 후속 연구로, 비디오 생성으로 확장 |
| **Simple Diffusion** (Hoogeboom et al.) | 2023 | arXiv | 종단간 고해상도 확산 | 노이즈 스케줄 조정으로 고해상도 직접 생성 | 전체 이미지 기반으로 메모리 요구량 큼 |
| **Latent Patch Diffusion** | 2024 | IEEE | 잠재 공간 패치 확산 | 학습·추론 시간 증가 없이 인접 오버래핑 패치를 활용하여 그리드 아티팩트를 회피하는 잠재 패치 확산 모델 | Patch-DM의 아이디어를 잠재 공간으로 확장 |

### 패러다임 비교 요약

$$\text{LDM: } x \xrightarrow{\text{Encode}} z \xrightarrow{\text{Diffusion}} \hat{z} \xrightarrow{\text{Decode}} \hat{x}$$

$$\text{CDM: } x_{\text{low}} \xrightarrow{\text{Diffusion}_1} \hat{x}_{\text{low}} \xrightarrow{\text{SR-Diffusion}_2} \hat{x}_{\text{high}}$$

$$\text{Patch-DM: } \{x^{(i,j)}\}_{i,j} \xrightarrow{\text{Feature Collage + Diffusion}} \{\hat{x}^{(i,j)}\}_{i,j} \xrightarrow{\text{Seamless Merge}} \hat{x}_{\text{full}}$$

---

# 참고자료

1. **Ding, Z., Zhang, M., Wu, J., & Tu, Z.** (2024). "Patched Denoising Diffusion Models For High-Resolution Image Synthesis." *ICLR 2024*. [arXiv:2308.01316](https://arxiv.org/abs/2308.01316)
2. **Patch-DM 프로젝트 페이지**: [https://patchdm.github.io/](https://patchdm.github.io/)
3. **Patch-DM GitHub**: [https://github.com/mlpc-ucsd/Patch-DM](https://github.com/mlpc-ucsd/Patch-DM)
4. **ICLR 2024 OpenReview**: [https://openreview.net/forum?id=TgSRPRz8cI](https://openreview.net/forum?id=TgSRPRz8cI)
5. **Podell, D., et al.** (2023). "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis." *ICLR 2024*. [arXiv:2307.01952](https://arxiv.org/abs/2307.01952)
6. **Zheng, H., et al.** (2023). "Patch Diffusion: Faster and More Data-Efficient Training of Diffusion Models." *NeurIPS 2023*.
7. **Skorokhodov, I., et al.** (2024). "Hierarchical Patch Diffusion Models for High-Resolution Video Generation." *CVPR 2024*.
8. **Liner.com Quick Review** — Patch-DM 리뷰: [https://liner.com/review/patched-denoising-diffusion-models-for-highresolution-image-synthesis](https://liner.com/review/patched-denoising-diffusion-models-for-highresolution-image-synthesis)
9. **Ho, J., Jain, A., & Abbeel, P.** (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS 2020*.
10. **Rombach, R., et al.** (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." *CVPR 2022*.

---

> **참고**: Feature Collage의 정확한 내부 수식(예: 특징 크롭·결합의 구체적 연산)은 논문 원문(ICLR 2024 camera-ready 버전)에 상세히 기술되어 있으며, 위에서 제시한 수식은 논문의 핵심 아이디어를 수학적으로 표현한 것입니다. 특정 수식의 정확한 표현이 필요하시면 원문 PDF를 참조하시기 바랍니다.
