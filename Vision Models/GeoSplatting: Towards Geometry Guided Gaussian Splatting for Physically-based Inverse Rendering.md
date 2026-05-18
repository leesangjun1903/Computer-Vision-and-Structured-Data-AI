
# GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering

> **논문 정보:**
> - **저자:** Kai Ye, Chong Gao, Guanbin Li, Wenzheng Chen, Baoquan Chen
> - **발표:** ICCV 2025 (arXiv: 2410.24204, 2024년 10월 31일 제출)
> - **공식 코드:** [GitHub - PKU-VCL-Geometry/GeoSplatting](https://github.com/PKU-VCL-Geometry/GeoSplatting)
> - **OpenReview:** [ICLR 2025 submission](https://openreview.net/forum?id=l5VA9wHJ8u)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

GeoSplatting은 3D Gaussian Splatting(3DGS)을 활용한 물리 기반 역렌더링(Physically-based Inverse Rendering) 문제를 다룬다. 최근 3DGS 방법들이 신규 시점 합성(NVS)에서 탁월한 성과를 보였지만, 고정밀 형상 모델링과 물리적으로 해석 가능한 재질·조명 분해는 여전히 어려운 과제로 남아 있다.

기존 3DGS 방법들은 표면 법선을 근사적으로 추정하는 방식에 의존하지만, 노이즈가 많은 지역 형상으로 인해 부정확한 법선 추정과 비최적(suboptimal) 재질-조명 분해라는 문제를 겪는다. 이 논문은 3DGS에 명시적 형상 가이던스(explicit geometric guidance)와 미분 가능한 PBR 방정식을 결합한 새로운 하이브리드 표현 방식인 GeoSplatting을 제안한다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| 명시적 형상 가이던스 | FlexiCubes 기반 최적화 가능한 삼각 메시 도입 |
| MGadapter | 메시 위에 Gaussian 포인트를 미분 가능하게 생성하는 모듈 |
| BVH 가속 레이 트레이싱 | 조명 전달(light transport) 정밀 계산 |
| End-to-end 최적화 | 형상·재질·조명의 통합 최적화 |

---

## 2. 상세 설명

### 2-1. 해결하고자 하는 문제

기존 접근 방식은 Gaussian 포인트를 surfel 형태로 납작하게 만든 후, depth-normal 정규화 같은 암묵적 기하학적 제약을 통해 표면 법선을 근사한다. 하지만 정확한 광 전달 모델링은 정밀한 법선 방향과 불투명 표면이 모두 필요하다. 그 결과, 근사된 법선과 Gaussian 포인트의 반투명 특성에 의존하는 기존 3DGS 기반 역렌더링 방법들은 광 전달 모델링에 어려움을 겪으며, 노이즈가 많은 재질 분해와 오류가 있는 리라이팅 결과를 초래한다.

구체적으로 기존 방식의 두 가지 핵심 문제는 다음과 같다:

1. **부정확한 법선 추정:** 기존 접근 방식들은 Gaussian 포인트의 법선을 근사하는 방식에 의존하며, 이는 암묵적 기하학적 제약에 해당한다.
2. **불투명 표면 부재:** Gaussian 포인트는 본질적으로 반투명하여 정확한 광선-표면 교차점 계산이 불가능하다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### 전체 파이프라인

GeoSplatting은 메시 표면에 구조화된 Gaussian 포인트를 미분 가능하게 구성함으로써 3DGS와 메시 표현을 효과적으로 연결한다. 이를 위해 FlexiCubes를 사용하여 학습 가능한 아이소값 그리드(isovalue grid)로부터 삼각 메시를 추출하고, 이어서 MGadapter가 메시 표면 위에 미분 가능하게 3DGS 포인트를 생성한다.

#### Step 1: FlexiCubes 기반 메시 추출

스칼라 필드 $\boldsymbol{\zeta}: \mathbb{R}^3 \rightarrow \mathbb{R}$가 공간 위치를 스칼라 값에 매핑하며, 아이소서페이스는 그리드 꼭짓점에 저장된 학습 가능한 값으로 $\boldsymbol{\zeta}$를 이산화하여 표현된다.

즉, 학습 가능한 스칼라 필드로부터 아이소서페이스를 추출:

$$\mathcal{M} = \text{FlexiCubes}(\boldsymbol{\zeta})$$

여기서 $\mathcal{M}$은 삼각 메시, $\boldsymbol{\zeta}$는 학습 가능한 그리드 정점 값이다.

#### Step 2: MGadapter — Mesh-to-Gaussian 변환

MGadapter는 각 삼각형 면(facet)에 적용되며, 사전 정의된 패턴에 따라 6개의 Gaussian 포인트를 생성한다. 이 패턴은 중심좌표(barycentric coordinate) 공간에서 사전 정의되어, 메시 법선과 Gaussian 법선 사이의 일관성을 추가 최적화 없이 유지하게 한다.

각 Gaussian 포인트 $i$의 속성 생성:

$$\{\boldsymbol{\mu}_i, \mathbf{s}_i, \mathbf{R}_i, \mathbf{n}_i\}_{i=1}^K = \text{MGadapter}(\mathcal{M})$$

- $\boldsymbol{\mu}_i \in \mathbb{R}^3$: 포인트 위치
- $\mathbf{s}_i$: 스케일
- $\mathbf{R}_i$: 회전 행렬
- $\mathbf{n}_i$: 법선 벡터 (메시 법선으로부터 직접 획득)

MGadapter는 Gaussian 포인트와 메시 가이던스 사이의 형상 일관성을 보장하도록 설계되었으며, 이 고유한 형상 일관성으로 인해 훈련 중 형상 가이던스의 end-to-end 최적화가 가능하고, 정밀한 법선 추정 및 정확한 광 전달 모델링이 가능해진다.

#### Step 3: PBR 렌더링 방정식

출사 방사휘도(outgoing radiance) $L_o(\mathbf{x}, \boldsymbol{\omega}_o)$는 BRDF 함수 $f_r(\mathbf{x}, \boldsymbol{\omega}_i, \boldsymbol{\omega}_o)$, 입사광 $L_i(\mathbf{x}, \boldsymbol{\omega}_i)$, 그리고 표면 법선 $\mathbf{n}$과 입사광 방향 $\boldsymbol{\omega}_i$ 사이의 각도를 고려하는 코사인 항 $|\mathbf{n} \cdot \boldsymbol{\omega}_i|$의 반구 $\mathcal{H}^2$ 상에서의 적분으로 계산된다.

$$L_o(\mathbf{x}, \boldsymbol{\omega}_o) = \int_{\mathcal{H}^2} f_r(\mathbf{x}, \boldsymbol{\omega}_i, \boldsymbol{\omega}_o) \, L_i(\mathbf{x}, \boldsymbol{\omega}_i) \, |\mathbf{n} \cdot \boldsymbol{\omega}_i| \, d\boldsymbol{\omega}_i \tag{Eq. 2}$$

#### Step 4: GGX 마이크로패싯 BRDF 모델

GeoSplatting은 물리 기반 렌더링 방정식을 통해 고차 조명 효과를 표현하며, BRDF 재질은 GGX 마이크로패싯 모델로 정식화된다.

GGX 모델은 BRDF 함수 $f_r(\mathbf{x}, \boldsymbol{\omega}_i, \boldsymbol{\omega}_o)$를 두 성분으로 정의한다: 확산(diffuse) 항과 정반사(specular) 항.

$$f_r(\mathbf{x}, \boldsymbol{\omega}_i, \boldsymbol{\omega}_o) = (1 - m)\frac{\mathbf{c}_d}{\pi} + \frac{DFG}{4|\mathbf{n} \cdot \boldsymbol{\omega}_i||\mathbf{n} \cdot \boldsymbol{\omega}_o|} \tag{Eq. 3}$$

- $m$: metalness (금속성)
- $\mathbf{c}_d$: diffuse albedo
- $D$: Normal Distribution Function (NDF, GGX)
- $F$: Fresnel 항
- $G$: Geometry/Shadowing 항

#### Step 5: 조명 전달 모델링 (Light Transport)

입사광 $L_i(\mathbf{x}, \boldsymbol{\omega}\_i)$의 정확한 모델링은 환경 조명뿐 아니라 다중 반사에 의한 간접 조명도 포함하기 때문에 어렵다. 기존 방식들은 계산 비용이 높은 경로 추적 기법 대신 단일 반사 모델로 조명을 근사하여, $L_i(\mathbf{x}, \boldsymbol{\omega}\_i)$를 직접 조명 항 $L_{\text{dir}}(\boldsymbol{\omega}\_i)$와 간접 조명 항 $L_{\text{ind}}(\mathbf{x}, \boldsymbol{\omega}_i)$로 분리한다.

$$L_i(\mathbf{x}, \boldsymbol{\omega}_i) = V(\mathbf{x}, \boldsymbol{\omega}_i) \cdot L_{\text{dir}}(\boldsymbol{\omega}_i) + L_{\text{ind}}(\mathbf{x}, \boldsymbol{\omega}_i)$$

이러한 일관성은 효율적인 광 전달 계산을 위한 메시 기반 레이 트레이싱 기법의 사용을 촉진하며, 그림자 효과와 상호 반사를 효과적으로 처리하면서 우수한 최적화 효율성을 제공한다.

특히 BVH(Bounding Volume Hierarchy) 가속 레이 트레이싱을 통해 가시성(visibility) $V(\mathbf{x}, \boldsymbol{\omega}_i)$를 효율적으로 계산한다:

$$V(\mathbf{x}, \boldsymbol{\omega}_i) = \text{BVH-RayTrace}(\mathbf{x}, \boldsymbol{\omega}_i, \mathcal{M})$$

#### Step 6: PBR 속성을 위한 해시 그리드

기존 연구들을 따라, PBR 속성을 위한 다중 해상도 해시 그리드 $\mathbf{E}_d$와 $\mathbf{E}_s$를 도입한다.

- $\mathbf{E}_d$: Diffuse (albedo) 속성 저장
- $\mathbf{E}_s$: Specular (roughness, metalness) 속성 저장

#### 전체 파이프라인 요약

$$\underbrace{\boldsymbol{\zeta}}_{\text{Scalar Field}} \xrightarrow{\text{FlexiCubes}} \underbrace{\mathcal{M}}_{\text{Mesh}} \xrightarrow{\text{MGadapter}} \underbrace{\mathcal{G}}_{\text{3DGS Points}} \xrightarrow{\text{PBR + BVH}} \underbrace{\hat{I}}_{\text{Rendered Image}}$$

GeoSplatting은 먼저 스칼라 필드에서 중간 메시를 추출하고, 그 위에 Gaussian 포인트를 샘플링하여 PBR 방정식으로 렌더링한다. 최종적으로 Gaussian 래스터라이제이션 파이프라인을 통해 이미지로 합성되며, 전체 과정이 완전히 미분 가능하여 end-to-end 학습이 가능하다.

---

### 2-3. 모델 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                        GeoSplatting Pipeline                     │
│                                                                   │
│  ┌──────────────┐   FlexiCubes   ┌──────────────┐               │
│  │ Learnable    │ ─────────────► │ Triangular   │               │
│  │ Isovalue     │                │ Mesh (M)     │               │
│  │ Grids (ζ)    │                └──────┬───────┘               │
│  └──────────────┘                       │ MGadapter             │
│                                          ▼                       │
│                                ┌──────────────────┐             │
│                                │ Structured 3DGS  │             │
│                                │ Points (G)       │             │
│                                │ - Position μ     │             │
│                                │ - Scale s        │             │
│                                │ - Rotation R     │             │
│                                │ - Normal n (from M)│           │
│                                └────────┬─────────┘             │
│                                         │                        │
│           ┌─────────────┐               │ PBR Framework         │
│           │ Hash Grids  │ ─────────────►│                        │
│           │ Ed (diffuse)│               │ Rendering Eq.         │
│           │ Es (specular│               │ + GGX BRDF            │
│           └─────────────┘               │                        │
│                                         │ BVH Ray Tracing       │
│           ┌─────────────┐ ◄────────────┘ (Shadow/Inter-refl.)  │
│           │  Env. Light │                                        │
│           └─────────────┘                                        │
│                    │                                              │
│                    ▼                                              │
│           ┌──────────────────────────────┐                       │
│           │  3DGS Rasterization          │                       │
│           │  + Alpha Compositing         │                       │
│           └──────────────────────────────┘                       │
│                    │                                              │
│                    ▼                                              │
│           ┌──────────────────────────────┐                       │
│           │  Final Rendered Image (Î)    │                       │
│           └──────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

---

### 2-4. 성능 향상

광범위한 실험을 통해 GeoSplatting의 효과를 검증하였으며, GeoSplatting이 기하학적 정확도와 재질-조명 분해 측면에서 기존 3DGS 기반 역렌더링 베이스라인을 크게 능가하는 동시에, 3DGS 기반 및 암묵적 필드 기반 역렌더링 방법들과 비교하여 탁월한 효율성을 제공함을 보여주었다.

GeoSplatting은 NeRF 데이터셋과 DTU 실제 세계 데이터셋 모두에서 새로운 최고 수준의 훈련 효율성과 역렌더링 성능을 달성하였으며, 향상된 형상, 보다 정밀한 재질-조명 분해, 그리고 기존 Gaussian Splatting 베이스라인 대비 우수한 신규 시점 합성을 보여준다.

특히 GeoSplatting에서 3DGS가 메시 형상 위에 기반하기 때문에, 정밀한 표면 법선 모델링이 가능해지고, 이는 PBR 프레임워크를 통한 재질 분해에 활용되어 반사성이 높은 경우에도 우수한 분해 성능을 달성한다.

#### 주요 성능 비교 (정성적)

| 방법 | 법선 정확도 | 재질 분해 | 리라이팅 품질 | 훈련 효율 |
|---|---|---|---|---|
| GS-IR | 근사 (낮음) | 보통 | 제한적 | 빠름 |
| R3DG | 근사 (낮음) | 보통 | 보통 | 빠름 |
| TensoIR (NeRF 계열) | 보통 | 좋음 | 좋음 | 느림 |
| **GeoSplatting** | **높음 (메시 기반)** | **우수** | **우수** | **빠름** |

---

### 2-5. 한계점

GeoSplatting은 정확한 알베도(albedo)와 조명 복원에 성공하지만, 러프니스(roughness)가 간접 상호 반사의 영향을 약간 받는다. 그럼에도 불구하고 여전히 최고 수준의 리라이팅 효과를 달성한다.

추가적으로 파악된 한계점:

- **위상 변환 제약:** FlexiCubes 기반 메시는 토폴로지 변화가 어려워, 복잡한 위상을 가진 장면(예: 손가락 분리 등)에 취약할 수 있다.
- **초기화 민감성:** 훈련 초기 단계에서 FlexiCubes의 아이소값이 무작위로 초기화되어 과도한 수의 삼각형 슬라이스가 생성되며, 각 face에 6개의 Gaussian 포인트를 직접 샘플링하는 것은 상당한 메모리 비용을 초래하고 훈련 효율을 저하시킬 수 있다.
- **단일 반사 모델:** 다중 반사(multi-bounce) 광 전달을 완전히 처리하지 않고 단일 반사 모델을 근사적으로 사용한다.
- **비강체/동적 장면 미지원:** 현재 구조는 정적 장면에 초점을 맞추고 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 명시적 형상 가이던스를 통한 일반화

GeoSplatting은 암묵적 기하학적 제약에 의존하여 Gaussian 포인트 법선을 반복적으로 근사하는 기존 3DGS 기반 역렌더링 방법들과 달리, 최적화 가능한 명시적 메시로부터 표면 정렬 Gaussian 포인트를 미분 가능하게 구성한다. 그 결과, 정밀한 광 전달 모델링을 위해 메시의 잘 정의된 법선과 불투명 메시 표면을 활용하여 우수한 재질-조명 분리 및 향상된 리라이팅 품질을 달성한다.

이러한 명시적 형상 표현은 다음과 같은 일반화 이점을 제공한다:

1. **반사성 물체에 대한 일반화:** GeoSplatting에서 3DGS가 메시 형상 위에 기반하여 정밀한 표면 법선 모델링이 가능해지고, 반사성이 높은 경우에서도 우수한 분해 성능을 달성한다.

2. **다양한 데이터셋에서의 일반화:** 다양한 데이터셋에 걸친 포괄적인 평가를 통해 GeoSplatting의 효과성이 입증되었으며, 우수한 효율성과 최고 수준의 역렌더링 성능을 강조한다.

3. **End-to-end 학습을 통한 범용성:** MGadapter의 완전한 미분 가능성으로 인해 훈련 중 형상 가이던스의 end-to-end 최적화가 가능하며, 메시와 3DGS 간의 일관성이 보장된다.

### 3-2. 일반화 한계 및 개선 방향

- **학습 데이터 의존성:** 현재 단일 장면 최적화(per-scene optimization) 방식이므로, 새로운 장면에 대한 제로샷(zero-shot) 일반화는 지원되지 않는다.
- **복잡한 재질 처리:** 투명, 반투명 재질(subsurface scattering 등)에 대한 일반화는 현재 GGX 모델의 가정으로 인해 제한적이다.
- **향후 개선 방향:** 대규모 사전 학습과 결합(예: generative prior 활용)하거나, 장면별 빠른 적응(few-shot adaptation)을 지원하는 방향으로 일반화 성능을 향상시킬 수 있다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4-1. 계보 정리

```
NeRF (Mildenhall et al., 2020)
    │
    ├── NeRFactor (Zhang et al., 2021) — NeRF 기반 역렌더링
    ├── InvRender (Zhang et al., 2022) — 암묵적 역렌더링
    ├── NVDiffrec (Munkberg et al., 2022) — 메시+미분가능렌더링
    ├── TensoIR (Jin et al., 2023) — NeRF 기반 고품질 역렌더링
    │
3DGS (Kerbl et al., 2023)
    │
    ├── GS-IR (Liang et al., 2023) — 3DGS 기반 첫 역렌더링 시도
    ├── R3DG / Relightable 3DGS (2023) — BVH + BRDF
    ├── GaussianShader (Jiang et al., 2024) — 간소화 셰이딩
    ├── GI-GS (2024) — 글로벌 일루미네이션 분해
    └── GeoSplatting (Ye et al., 2024/ICCV2025) — 메시 가이던스 + MGadapter
```

### 4-2. 주요 방법론 비교

| 방법 | 표현 | 법선 추정 | 광 전달 | 속도 | 재질 품질 |
|---|---|---|---|---|---|
| **NeRFactor (2021)** | NeRF (암묵적) | 볼륨 렌더링 | 사전 계산 | 느림 | 보통 |
| **TensoIR (2023)** | Tensor NeRF | 볼륨 기반 | 경로 추적 | 보통 | 좋음 |
| **GS-IR (2023)** | 3DGS | 깊이 미분 근사 | SH 기반 베이킹 | 빠름 | 보통 |
| **R3DG (2023)** | 3DGS | 근사 | BVH 가시성 | 빠름 | 보통 |
| **GI-GS (2024)** | 3DGS | 근사 | Deferred+경로추적 | 빠름 | 좋음 |
| **GeoSplatting (2024)** | 3DGS+메시 | **메시 법선 (정밀)** | BVH+메시 레이트레이싱 | **빠름** | **최고** |

### 4-3. GS-IR과의 비교

3DGS를 역렌더링에 도입할 때 두 가지 주요 문제가 있다: 첫째, 3DGS는 자체적으로 그럴듯한 법선을 생성하지 않으며, 둘째, 래스터라이제이션/스플래팅 같은 순방향 매핑은 레이 트레이싱 같은 역방향 매핑처럼 폐색(occlusion)을 추적할 수 없다.

R3DG와 GS-IR 같은 기존 3DGS 기반 역렌더링 방법들은 근사된 법선 방향에 의존하여 광 전달 모델링의 부정확성과 재질 분해 노이즈 및 리라이팅 오류를 야기한다. 이에 반해 GeoSplatting은 명시적 형상 가이던스로 3DGS를 증강하여 법선 추정과 광 전달 모델링을 개선하며, 우수한 분해와 리라이팅 품질을 달성한다.

### 4-4. GI-GS와의 비교

GI-GS는 3D Gaussian Splatting 기반 역렌더링 프레임워크로, 각 Gaussian의 원래 속성에 표면 법선과 BRDF를 추가하여 미분 가능한 PBR 파이프라인을 통해 최적화한다.

GI-GS의 핵심은 deferred shading과 경로 추적을 결합하여 정확한 글로벌 일루미네이션 분해를 가능하게 하면서 실시간 렌더링을 보장하는 것이다.

GeoSplatting과의 차이: GI-GS는 여전히 법선을 근사하는 방식이나, GeoSplatting은 메시 법선을 직접 활용하여 근본적으로 더 정밀한 형상 표현을 제공한다.

### 4-5. RTR-GS (2025)와의 비교

RTR-GS는 임의의 반사율 특성을 가진 물체를 강인하게 렌더링하고, BRDF와 조명을 분해하며, 신뢰할 수 있는 리라이팅 결과를 제공하는 역렌더링 프레임워크로, 방사 전달을 위한 순방향 렌더링과 반사를 위한 deferred 렌더링을 결합한 하이브리드 렌더링 모델을 통해 기하학적 구조를 복원한다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5-1. 연구에 미치는 영향

#### (1) 메시-가우시안 하이브리드 표현의 패러다임 확립
이 연구는 3DGS와 명시적 형상 표현(메시)을 통합하는 학습 가능한 장면 모델을 제안하며, end-to-end 학습을 통해 메시와 외관을 동시에 학습하는 장면 업데이트 적응의 고유한 장점을 보여준다.

#### (2) 역렌더링 정밀도의 새로운 기준점 설정
GeoSplatting은 기하학적 정확도와 재질-조명 분해 측면에서 기존 3DGS 기반 역렌더링 베이스라인을 크게 능가하는 동시에, 3DGS 기반 및 암묵적 필드 기반 방법들과 비교하여 탁월한 효율성을 제공한다는 것을 광범위한 실험으로 검증하였다.

#### (3) 산업적 응용 확장
재구성된 객체는 상세한 형상과 사실적인 외관을 갖추어 새로운 시점에서의 렌더링을 가능하게 하지만, 재질과 조명의 분리 부재로 인해 다양한 조명 조건에 적응이 제한된다. 이는 기존 3D 재구성 방법이 게임, 영화 제작, AR/VR 같은 다운스트림 응용에 적합한 다용도 3D 자산 생성에 한계가 있음을 의미한다. GeoSplatting은 이 간극을 메우는 핵심 역할을 한다.

### 5-2. 향후 연구 시 고려할 점

#### 🔬 기술적 개선 과제

| 과제 | 현재 한계 | 개선 방향 |
|---|---|---|
| **위상 처리** | FlexiCubes는 단순 위상에 최적화 | Dynamic Topology, neural mesh 활용 |
| **다중 반사** | 단일 반사 근사 | 경로 추적 기반 다중 반사 통합 |
| **투명/반투명** | 불투명 표면 가정 | BSDF 확장, 체적 렌더링 통합 |
| **동적 장면** | 정적 장면 한정 | 4D GeoSplatting, 시간축 메시 변형 |
| **제로샷 일반화** | 장면별 최적화 | 대규모 사전 학습 결합 |

#### 🏗️ 연구 방향성

1. **생성 모델과의 결합:** GeoSplatting의 명시적 형상 표현을 3D 생성 모델(예: 3D-GAN, Diffusion 기반 3D 생성)과 결합하면 제로샷 역렌더링이 가능할 수 있다.

2. **실시간 역렌더링:** 3DGS는 타일 기반 래스터라이제이션 파이프라인을 사용하여 효율적인 렌더링을 실현한다. 이를 기반으로 실시간 역렌더링 달성을 위한 추가 최적화 연구가 필요하다.

3. **개방형 세계 일반화:** 기존 연구들은 제한된 객체 범주에서 평가되었으므로, 복잡한 실외 장면이나 다중 객체 상호작용 장면으로의 확장성 검증이 필요하다.

4. **신경망 BRDF 통합:** GGX 모델 대신 신경망 기반 BRDF를 적용하면 복잡한 재질(예: 직물, 피부, 머리카락)에 대한 처리가 가능해질 것이다.

5. **불확실성 정량화:** 재질 분해의 고유한 모호성(material-illumination ambiguity)을 해결하기 위한 베이지안 접근법 또는 앙상블 방법의 통합.

---

## 📚 참고 자료 (출처 목록)

| # | 제목 | 링크/출처 |
|---|---|---|
| 1 | **GeoSplatting (arXiv 원문)** | arxiv.org/abs/2410.24204 |
| 2 | **GeoSplatting (ICCV 2025 Open Access)** | openaccess.thecvf.com/content/ICCV2025/... |
| 3 | **GeoSplatting (HTML 전문)** | arxiv.org/html/2410.24204v3 |
| 4 | **GeoSplatting (OpenReview)** | openreview.net/forum?id=l5VA9wHJ8u |
| 5 | **GeoSplatting (공식 GitHub)** | github.com/PKU-VCL-Geometry/GeoSplatting |
| 6 | **GeoSplatting (Semantic Scholar)** | semanticscholar.org/paper/... |
| 7 | **GeoSplatting (ResearchGate PDF)** | researchgate.net/publication/385443643 |
| 8 | **GS-IR: 3D Gaussian Splatting for Inverse Rendering** | arxiv.org/html/2311.16473 |
| 9 | **GI-GS: Global Illumination decomposition on Gaussian Splatting** | arxiv.org/html/2410.02619v2 |
| 10 | **Relightable 3D Gaussian (R3DG)** | nju-3dv.github.io/projects/Relightable3DGaussian |
| 11 | **RTR-GS: 3DGS for Inverse Rendering with Radiance Transfer** | arxiv.org/abs/2507.07733 |
| 12 | **GeoSplatting Project Page (SYSU)** | sysu-hcp.net/projects/cv/175.html |

> ⚠️ **정확도 주의사항:** 본 답변은 arXiv, ICCV Open Access, OpenReview, GitHub 공식 소스에 기반하여 작성되었습니다. 수식의 일부 세부 표기(특히 MGadapter 내부 파라미터)는 논문 원문 HTML에서 LaTeX 파싱 오류로 일부 기호가 불완전하게 추출되었으므로, 정확한 수식은 [원문 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Ye_GeoSplatting_Towards_Geometry_Guided_Gaussian_Splatting_for_Physically-based_Inverse_Rendering_ICCV_2025_paper.pdf)를 직접 확인하시기를 권장합니다.
