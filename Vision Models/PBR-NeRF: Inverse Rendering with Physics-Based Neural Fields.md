
# PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields

> **논문 정보**
> - **제목**: PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields
> - **저자**: Sean Wu, Shamik Basu, Tim Broedermann, Luc Van Gool, Christos Sakaridis (ETH Zürich, Computer Vision Lab)
> - **발표**: CVPR 2025
> - **arXiv**: [2412.09680](https://arxiv.org/abs/2412.09680) (v1: 2024.12.12, v2: 2025.04.07)
> - **코드**: [https://github.com/s3anwu/pbrnerf](https://github.com/s3anwu/pbrnerf)

---

## 1. 핵심 주장 및 주요 기여 요약

이 논문은 물리 기반 렌더링(PBR) 이론으로부터 영감을 얻은 Neural Radiance Field(NeRF) 접근법으로 3D 재구성의 **역렌더링(Inverse Rendering) 문제**를 해결하는 **PBR-NeRF**를 제안합니다.

기존 NeRF 및 3D Gaussian Splatting 접근법의 핵심 한계인 "장면 재료(material) 및 조명(illumination)을 모델링하지 않고 시점 의존적 외관만 추정하는 문제"를 해결하기 위해, 장면의 기하학(geometry), 재료(materials), 조명(illumination)을 **동시에 추정**하는 역렌더링 모델을 제시합니다.

### 🔑 핵심 기여 (Key Contributions)

모델은 기존 NeRF 기반 IR 접근법 위에 구축되지만, IR 추정을 더 잘 제약(constrain)하는 **두 가지 새로운 물리 기반 Prior**를 도입합니다. 이 Prior들은 직관적인 손실 함수(loss terms)로 정형화되어 novel view synthesis 품질을 저하시키지 않으면서 **최고 수준(SOTA)의 재료 추정**을 달성합니다.

| 기여 항목 | 내용 |
|---|---|
| **두 가지 물리 기반 Loss** | Conservation of Energy Loss ( $\mathcal{L}\_{\text{cons}}$ ) + NDF-weighted Specular Loss ($\mathcal{L}_{\text{spec}}$) |
| **통합 역렌더링 프레임워크** | 기하학, 재료, 조명의 동시 추정 |
| **SOTA 달성** | 재료 추정 및 novel view synthesis에서 최고 성능 |
| **범용 적용 가능성** | 다른 역렌더링 및 3D 재구성 프레임워크에 쉽게 적용 가능 |

---

## 2. 해결하고자 하는 문제, 제안 방법(수식), 모델 구조, 성능

### 2.1 해결하고자 하는 문제

**① 역렌더링의 Ill-Posed 특성**

역렌더링 문제는 근본적으로 **Material-Lighting Ambiguity** 때문에 ill-posed합니다. 즉, 동일한 이미지가 장면 속성의 무한한 조합으로 설명될 수 있습니다.

**② 기존 NeRF/3DGS의 Black-Box 한계**

NeRF, 3DGS 및 그 파생 방법들은 장면을 빛 운반(light transport)의 물리를 무시하는 "블랙박스"로 취급합니다. 이 방법들은 수백만 개의 반투명 파티클을 볼륨 렌더링하여 빛을 표현하고, 각 파티클의 방사를 위치와 시점 방향에 의존하도록 합니다. 이 방식은 물리적으로 정확한 결과를 보장하지 않으며, 예를 들어 반사 표면이 불적절하게 모델링되어 **"baked-in" specular highlights** 같은 아티팩트를 유발합니다.

**③ 기존 역렌더링 방법의 재료 Prior 부재**

NeILF 및 NeILF++는 Disney BRDF의 metallicness와 roughness 파라미터 정규화에 **Lambertian Prior**를 사용합니다. 그러나 완벽히 거칠고 비금속적인 재료라는 강한 가정은 BRDF 표현력을 제한하며, Specular 효과가 종종 Lambertian Prior가 Specular Lobe를 억제함으로 인해 Diffuse Lobe에 나타납니다.

---

### 2.2 제안 방법 및 핵심 수식

#### 🔹 렌더링 방정식 (Rendering Equation)

역렌더링의 이론적 기반이 되는 반사 방정식은 다음과 같습니다:

$$L_o(\mathbf{p}, \boldsymbol{\omega}_o) = \int_\Omega f_r(\mathbf{p}, \boldsymbol{\omega}_i, \boldsymbol{\omega}_o) \cdot L_i(\mathbf{p}, \boldsymbol{\omega}_i) \cdot (\boldsymbol{\omega}_i \cdot \mathbf{n}) \, d\boldsymbol{\omega}_i$$

여기서:
- $L_o$: 나가는 방향 $\boldsymbol{\omega}_o$의 출사 광휘도(Outgoing radiance)
- $f_r$: BRDF (Bidirectional Reflectance Distribution Function)
- $L_i$: 입사 광휘도(Incident radiance)
- $\mathbf{n}$: 표면 법선(Surface normal)
- $\Omega$: 반구 적분 영역

#### 🔹 Disney BRDF 모델

PBR-NeRF는 표면 재료를 표현하기 위해 **Disney BRDF**를 사용합니다. PBR-NeRF의 물리 기반 손실 함수는 이 BRDF 모델을 직접 제약하도록 설계되어 논문의 핵심 기술적 기여에서 근본적입니다.

Disney BRDF는 다음으로 구성됩니다:

$$f_r = f_d + f_s$$

- **Diffuse Lobe** $f_d$: Lambertian 반사 성분

$$f_d = \frac{\text{albedo}}{\pi}$$

- **Specular Lobe** $f_s$ (Cook-Torrance Microfacet Model):

$$f_s = \frac{D(\mathbf{h}) \cdot F(\boldsymbol{\omega}_i, \mathbf{h}) \cdot G(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o, \mathbf{n})}{4(\boldsymbol{\omega}_i \cdot \mathbf{n})(\boldsymbol{\omega}_o \cdot \mathbf{n})}$$

여기서 $D$: Normal Distribution Function (NDF), $F$: Fresnel, $G$: Geometry Term

---

#### 🔹 핵심 기여 1: Conservation of Energy Loss ($\mathcal{L}_{\text{cons}}$)

Disney BRDF와 다른 마이크로패싯 모델들은 에너지를 보존하지 못합니다. 이는 역렌더링에서 재료가 에너지를 생성(받은 것보다 더 많이 반사)하거나 파괴(너무 적게 반사)하는 것을 허용하기 때문에 중대한 도전입니다. 이러한 부정확성은 재료 추정을 왜곡하고 조명 추정을 방해하여, 추정된 조명이 과보상하여 너무 밝거나 어둡게 나타납니다.

에너지 보존 조건은 다음과 같습니다:

$$\int_\Omega f_r(\boldsymbol{\omega}_i, \mathbf{n}) \cdot (\boldsymbol{\omega}_i \cdot \mathbf{n}) \, d\boldsymbol{\omega}_i \leq 1$$

이를 위반하는 경우를 페널티로 주는 Loss:

$$\mathcal{L}_{\text{cons}} = \mathbb{E}\left[\max\left(0, \int_\Omega f_r(\boldsymbol{\omega}_i \cdot \mathbf{n}) \, d\boldsymbol{\omega}_i - 1\right)\right]$$

PBR-NeRF의 Conservation of Energy Loss는 **받은 에너지보다 더 많이 반사하는 재료에 페널티를 부여**하여 물리적으로 유효한 BRDF를 강제합니다.

---

#### 🔹 핵심 기여 2: NDF-weighted Specular Loss ($\mathcal{L}_{\text{spec}}$)

두 번째 물리 기반 손실 함수는 NeILF++와 같이 Lambertian 반사를 가정하는 역렌더링 방법에서 자주 관찰되는 **Diffuse Lobe와 Specular Lobe 사이의 불균형**을 목표로 합니다. 실제 재료들은 종종 이상적인 Lambertian 동작을 위반하여, Diffuse Lobe가 불충분한 Specular 반사를 보상하는 "baked-in" specular highlights를 초래합니다. 이 불균형은 비경면 각도에서 잘못된 Diffuse 동작을 유발하고 재료 추정 품질을 저하시킵니다.

NDF-weighted Specular Loss는 Specular 영역에서의 과도한 Diffuse 반사에 페널티를 부여함으로써, **Diffuse와 Specular BRDF Lobe의 분리를 촉진**하고 "baked-in" 하이라이트를 수정합니다.

NDF 가중치를 이용한 Specular Loss 수식:

$$\mathcal{L}_{\text{spec}} = \mathbb{E}\left[D(\mathbf{h}) \cdot f_d(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o)\right]$$

- NDF 값 $D(\mathbf{h})$가 높은 곳(즉, Specular 반사가 강한 방향)에서 Diffuse 성분 $f_d$가 크면 페널티를 부과

---

#### 🔹 전체 재료 손실 함수

전체 재료 손실 함수는 다음과 같이 구성됩니다:

$$\mathcal{L}_{\text{mat}} = \lambda_{\text{pbr}}\mathcal{L}_{\text{pbr}} + \lambda_{\text{ref}}\mathcal{L}_{\text{ref}} + \lambda_{\text{smth}}\mathcal{L}_{\text{smth}} + \mathcal{L}_{\text{mat,physics}}$$

여기서:

$$\mathcal{L}_{\text{mat,physics}} = \lambda_{\text{cons}}\mathcal{L}_{\text{cons}} + \lambda_{\text{spec}}\mathcal{L}_{\text{spec}}$$

---

### 2.3 모델 구조

완전한 PBR-NeRF 모델은 **단계적(stage-wise) 방식으로 최적화되는 다중 신경장(Multiple Neural Fields)**으로 구성됩니다: (1) 방사와 기하학을 모델링하는 표준 NeRF+SDF, (2) 공간적으로 변화하는 조명을 모델링하는 Neural Incident Light Field(NeILF), (3) Disney BRDF를 통해 재료를 모델링하는 BRDF Field.

```
┌──────────────────────────────────────────────────────────┐
│               PBR-NeRF Architecture                       │
│                                                            │
│  ┌─────────────┐  ┌──────────────────┐  ┌─────────────┐  │
│  │  NeRF+SDF   │  │  NeILF           │  │  BRDF Field │  │
│  │  Network    │  │  (Neural         │  │  (Disney    │  │
│  │             │  │  Incident Light  │  │   BRDF)     │  │
│  │  Geometry   │  │  Field)          │  │             │  │
│  │  + Radiance │  │  Spatially-      │  │  Albedo     │  │
│  │             │  │  Varying         │  │  Roughness  │  │
│  │             │  │  Illumination    │  │  Metallic   │  │
│  └─────────────┘  └──────────────────┘  └──────┬──────┘  │
│                                                 │          │
│              Physics-Based Losses               │          │
│       ┌─────────────────────────────────────┐  │          │
│       │  L_cons (Conservation of Energy)    │◄─┘          │
│       │  L_spec (NDF-weighted Specular)     │             │
│       └─────────────────────────────────────┘             │
└──────────────────────────────────────────────────────────┘
```

**최적화 단계 (Stage-wise Optimization)**:

Joint Optimization 단계에서, NeRF SDF, BRDF, NeILF MLP들은 이전 기하학 및 재료 단계에 의해 사전 훈련되어, 모든 네트워크를 동시에 공동 최적화할 수 있습니다.

1. **Stage 1 (Geometry Phase)**: NeRF+SDF로 기하학 및 방사 학습
2. **Stage 2 (Material Phase)**: BRDF Field로 재료 추정 (Physics-based Losses 적용)
3. **Stage 3 (Joint Phase)**: 모든 네트워크 동시 최적화

물리 기반 손실 함수는 BRDF Field에 귀중한 귀납적 편향(inductive biases)을 제공하여, 역렌더링에서 고유한 재료-조명 모호성을 상당 부분 해결하고 결과적으로 **최고 수준의 품질을 가진 novel scene views**를 합성합니다.

---

### 2.4 성능 향상

#### 정량적 결과

두 손실 함수( $\mathcal{L}\_{\text{cons}}$ 및 $\mathcal{L}_{\text{spec}}$ )의 결합 적용은 재료 추정에서 가장 유의미한 개선을 달성합니다. 구체적으로, 전체 방법(ID 4)은 Baseline 대비 albedo PSNR을 **3.28 향상**, roughness PSNR을 **0.35 향상**, metallicness PSNR을 **0.27 향상**합니다.

DTU 데이터셋에서, 두 손실 함수를 결합한 PBR-NeRF는 Baseline 대비 **mean RGB PSNR 0.78 향상**과 **Chamfer Distance 0.339mm 개선**을 달성합니다.

| 메트릭 | Baseline (NeILF++) | PBR-NeRF | 향상 |
|---|---|---|---|
| Albedo PSNR | 기준 | +3.28 | ↑ |
| Roughness PSNR | 기준 | +0.35 | ↑ |
| Metallic PSNR | 기준 | +0.27 | ↑ |
| RGB PSNR (DTU) | 기준 | +0.78 | ↑ |
| Chamfer Distance | 기준 | -0.339mm | ↓(개선) |

PBR-NeRF는 더 정확한 재료 추정과 최고 수준의 novel view synthesis를 달성하며, 물리적 원칙을 강제함으로써 DTU 데이터셋에서 NeILF++ 대비 **0.37 PSNR 개선**을 달성합니다.

#### 정성적 결과

세 가지 까다로운 Mix 조명 조건에서, 물리 기반 손실 함수는 조명 및 재료 추정을 모두 개선합니다. 추정된 조명은 더 낮은 엔트로피와 더 집중된 광원, 특히 환경 맵의 상단 절반에서 더 적은 아티팩트를 보입니다.

실험 결과는 PBR-NeRF가 역렌더링에서 **재료 추정의 새로운 SOTA**를 달성함을 보여주며, 특히 다양한 조명 하에서 일관성 있는 신뢰할 수 있는 Albedo 추정을 제공하고, Metallicness 및 Roughness 추정도 개선합니다.

---

### 2.5 한계점 (Limitations)

논문 및 관련 분석을 바탕으로 한 주요 한계:

1. **간접 조명(Inter-reflection) 미고려**: NeRF 기반 IR 프레임워크는 그림자와 재료를 분리하는 데 어려움이 있다는 점이 지적됩니다.

2. **계산 비용**: Stage-wise 최적화로 인한 긴 학습 시간

3. **NeILF++ 프레임워크에 대한 의존성**: NeILF++가 PBR-NeRF가 직접 구축되는 프레임워크로, 저자들이 이를 위에 구현하고 주요 실험 Baseline으로 사용하며 그 한계를 해결하고자 합니다.

4. **실내/대규모 장면 제약**: 평가가 주로 제한된 객체 중심 장면(DTU, NeILF++ 데이터셋)에서 이루어짐

5. **Disney BRDF 모델 한계**: 극도로 복잡한 재료(서브서피스 스캐터링 등)는 처리하기 어려움

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 왜 물리 기반 Prior가 일반화를 향상시키는가?

PBR-NeRF의 핵심 인사이트는 물리 기반 렌더링 이론이 역문제를 제약하는 강력한 Prior를 제공한다는 것입니다. 데이터 기반 학습이나 약한 정규화 항에만 의존하는 대신, 에너지 보존과 올바른 Diffuse-Specular 분리 같은 **근본적인 물리 법칙을 명시적으로 강제**합니다.

이는 일반화 성능에 다음과 같이 기여합니다:

| 일반화 인자 | 설명 |
|---|---|
| **도메인-불변 Prior** | 에너지 보존 법칙은 모든 물리적 장면에 적용 |
| **데이터 독립적 제약** | 특정 훈련 분포에 과적합되지 않음 |
| **재료-조명 분리** | 다양한 조명 환경에서도 올바른 재료 추정 |

특히, 이 방법은 재료 추정이 필요한 다른 역렌더링 및 3D 재구성 프레임워크에 쉽게 적용될 수 있습니다.

### 3.2 다양한 조명 조건에서의 일반화

모델은 **다양한 조명 환경에서도 일관성 있는 Albedo 추정**을 제공하는데, 이는 효과적인 Diffuse-Specular 분리 덕분입니다.

평가에서도 City, Studio, Castel이라는 세 가지 장면과 Env(환경 맵), Mix(환경 맵 + 포인트 + 영역 광원)라는 다양한 조명 조건에서 일관된 성능 향상을 보입니다.

### 3.3 실제 데이터셋(DTU)에서의 일반화

역렌더링 방법들 중에서 PBR-NeRF는 DTU에서 novel view synthesis와 기하학 추정에서 **최고 수준의 결과**를 달성합니다.

DTU는 실제 캡처 이미지로 구성되어 있어, 합성 데이터로 훈련된 방법들과 달리 실제 환경에서의 일반화 능력을 보여줍니다.

### 3.4 프레임워크 독립적 적용 가능성

PBR-NeRF는 PBR에서 영감을 받은 Loss로 최적화된 신경장을 활용하는 역렌더링 방법입니다. 이 Loss들은 물리적으로 유효한 귀납적 편향을 제공하여, 신경장이 조명으로부터 재료를 더 잘 분리할 수 있도록 합니다.

이러한 Loss 함수들은 3D Gaussian Splatting 기반 방법에도 원칙적으로 적용 가능하여, 광범위한 일반화 가능성을 시사합니다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 NeRF 기반 역렌더링의 발전

| 연구 | 연도 | 특징 | PBR-NeRF와 비교 |
|---|---|---|---|
| **NeRF** (Mildenhall et al.) | ECCV 2020 | View synthesis의 시초, 물리 모델 없음 | PBR-NeRF의 기반이 되는 방법 |
| **NeRFactor** | SIGGRAPH Asia 2021 | Shape + BRDF + 조명 분리 | 재료 추정 정확도 낮음 |
| **PhySG** | CVPR 2021 | Spherical Gaussians 기반 역렌더링 | 조명 모델 표현력 제한 |
| **NeILF** | ECCV 2022 | Neural Incident Light Field 도입 | PBR-NeRF의 직접 Baseline |
| **TensoIR** | CVPR 2023 | 텐서 분해 기반 역렌더링 | 순수 MLP 방식의 낮은 용량과 높은 계산 비용을 극복하여 다중 뷰 이미지에서 장면 기하학, 표면 반사율, 환경 조명을 추정 |
| **NeILF++** | ICCV 2023 | Inter-reflection 고려 | PBR-NeRF가 직접 개선하는 방법 |
| **Relightable 3D Gaussian** | ECCV 2024 | 3DGS 기반 BRDF 분해 | 실시간 렌더링 가능하지만 물리 Loss 부재 |
| **PBR-NeRF** | CVPR 2025 | 물리 기반 Loss로 재료-조명 분리 | 현 분석 대상 |

### 4.2 TensoIR vs. PBR-NeRF

기존 NeRF 기반 및 SDF 기반 방법들은 고비용 MLP 평가로 인해 이차 효과(그림자, 간접 조명)를 단순히 무시하거나, 추가 MLP에서 사전 계산하여 근사합니다. TensoIR은 효율적인 텐서 분해 표현으로 낮은 비용의 second-bounce 레이 마칭으로 가시성과 간접 조명을 명시적으로 계산합니다.

TensoIR은 **간접 조명 모델링**에서 강점이 있지만, PBR-NeRF는 **재료-조명 분리의 물리적 정확성**에서 우위를 가집니다.

### 4.3 3DGS 기반 방법과의 비교

표준 NeRF와 3DGS는 반사 같은 시점 의존적 효과를 기억하는 블랙박스처럼 장면을 처리하며, 빛 운반 물리를 무시합니다.

Relightable 3DGS 및 GaussianShader 같은 방법들은 실시간 렌더링이 가능하지만, 에너지 보존 같은 물리 제약을 명시적으로 강제하지 않아 재료 정확도에서 열위입니다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 앞으로의 연구에 미치는 영향

방법의 합성 및 실제 데이터 모두에서의 향상, 모듈식 설계 및 공개 코드 가용성은 **물리적으로 일관된 신경 렌더링을 향한 중요한 단계**로 자리매김합니다. 분야가 더 정교한 장면 이해 및 조작 역량으로 나아감에 따라, PBR-NeRF의 물리 기반 제약 통합은 신경 역렌더링의 미래 발전에 견고한 기반을 제공합니다.

**구체적 파급 효과:**

1. **3D Gaussian Splatting에의 통합**: 물리 기반 Loss를 3DGS 프레임워크에 적용하는 연구 촉진
2. **Diffusion Model 기반 역렌더링**: 물리 Prior + 생성 모델의 결합
3. **실시간 물리 기반 렌더링**: 물리 제약을 유지하면서 실시간성을 달성하는 방법 탐색
4. **다운스트림 응용 개선**: VR, AR, 상세한 3D 모델링과 장면 상호작용이 필요한 모든 도메인에서 상당한 향상

저자들은 물리 기반 접근법이 **PBR-구동의 분해된 신경장에 대한 추가 연구에 영감을 줄 것**이라 믿습니다.

### 5.2 앞으로 연구 시 고려할 점

**① 물리 모델 확장**

향후 연구에서는 신경장 프레임워크 내에서 더 정교한 PBR 모델의 탐색이 추가 발전을 가져올 수 있으며, 다른 형태의 Prior 지식 통합이나 제한된 데이터 가용성을 가진 렌더링 응용에서 추가 이익을 가져올 수 있는 **반지도 학습(semi-supervised learning)** 접근법 탐색도 고려해야 합니다.

**② 확장성 고려**

현재 PBR-NeRF는 소규모 객체 중심 장면에서 검증되었습니다. 대규모 야외 장면이나 동적 장면으로의 확장을 위해서는:

- **계산 효율화**: 해시 인코딩(Instant-NGP 방식) 등 효율적 표현 활용
- **동적 장면 처리**: 시간에 따른 재료 변화 모델링

**③ 간접 조명 처리**

TensoIR처럼 그림자 및 간접 조명 같은 이차 효과를 정확하게 모델링하는 것은 높은 품질의 장면 재구성에 중요합니다. PBR-NeRF는 현재 직접 조명 중심으로 설계되어 있으므로, 전역 조명(Global Illumination)과의 통합이 중요한 과제입니다.

**④ 벤치마크 다양화**

현재 NeILF++ 및 DTU 데이터셋 중심 평가를 넘어, Stanford-ORB 등 더 다양한 실제 환경 벤치마크에서의 검증이 필요합니다.

**⑤ 3DGS와의 융합**

PBR-NeRF는 물리적 통찰을 신경 렌더링 과정에 통합하여, **물리 기반 신경 렌더링 방법론의 미래 탐색을 위한 기반**을 마련합니다. 따라서 3DGS의 속도 이점과 PBR-NeRF의 물리 정확성을 결합하는 것이 유망한 연구 방향입니다.

---

## 📚 참고 자료

| # | 제목 및 출처 |
|---|---|
| 1 | **PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields** — arXiv:2412.09680, CVPR 2025. Sean Wu, Shamik Basu, Tim Broedermann, Luc Van Gool, Christos Sakaridis (ETH Zürich). https://arxiv.org/abs/2412.09680 |
| 2 | **PBR-NeRF Project Page** — https://s3anwu.github.io/pbrnerf/ |
| 3 | **PBR-NeRF GitHub Repository** — https://github.com/s3anwu/pbrnerf |
| 4 | **PBR-NeRF CVPR 2025 Paper (CVF)** — https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_PBR-NeRF_Inverse_Rendering_with_Physics-Based_Neural_Fields_CVPR_2025_paper.pdf |
| 5 | **PBR-NeRF arXiv HTML** — https://arxiv.org/html/2412.09680v1 |
| 6 | **PBR-NeRF ETH Zürich Preprint** — https://people.ee.ethz.ch/~csakarid/PBR-NeRF/ |
| 7 | **[Quick Review] PBR-NeRF** — Liner.com, https://liner.com/review/pbrnerf-inverse-rendering-with-physicsbased-neural-fields |
| 8 | **[Literature Review] PBR-NeRF** — Moonlight, https://www.themoonlight.io/en/review/pbr-nerf-inverse-rendering-with-physics-based-neural-fields |
| 9 | **PBR-NeRF on alphaXiv** — https://www.alphaxiv.org/overview/2412.09680v1 |
| 10 | **PBR-NeRF on EmergentMind** — https://www.emergentmind.com/papers/2412.09680 |
| 11 | **TensoIR: Tensorial Inverse Rendering** — arXiv:2304.12461, CVPR 2023. https://arxiv.org/abs/2304.12461 |
| 12 | **NeILF: Neural Incident Light Field** — ECCV 2022, Yao Yao et al. |
| 13 | **NeILF++: Inter-reflectable Light Fields** — ICCV 2023, Jingyang Zhang et al. |
| 14 | **NeRF: Representing Scenes as Neural Radiance Fields** — ECCV 2020, Mildenhall et al. |
| 15 | **Awesome-Inverse-Rendering** — GitHub 컬렉션, https://github.com/ingra14m/Awesome-Inverse-Rendering |
| 16 | **IEEE Xplore — PBR-NeRF** — https://ieeexplore.ieee.org/document/11094442/ |
