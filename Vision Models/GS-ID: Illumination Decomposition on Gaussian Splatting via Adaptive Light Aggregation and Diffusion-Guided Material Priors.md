
# GS-ID: Illumination Decomposition on Gaussian Splatting via Adaptive Light Aggregation and Diffusion-Guided Material Priors

> **논문 정보**
> - 제목: *GS-ID: Illumination Decomposition on Gaussian Splatting via Adaptive Light Aggregation and Diffusion-Guided Material Priors*
> - 저자: Kang Du, Zhihao Liang, Yulin Shen, Zeyu Wang
> - 소속: The Hong Kong University of Science and Technology (Guangzhou), South China University of Technology
> - 발표: **ICCV 2025**
> - arXiv: [2408.08524](https://arxiv.org/abs/2408.08524) (v1: 2024.08.16 / v2: 2025.08.04)
> - 공식 프로젝트 페이지: https://kangdu.top/gsid/
> - GitHub: https://github.com/dukang/GS-ID

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 배경 및 핵심 문제 인식

Gaussian Splatting(GS)은 사진처럼 사실적인 렌더링을 위한 효과적인 표현 방식으로 부상했지만, 기하학(geometry), 재질(material), 조명(lighting)이 서로 얽혀 있어 장면 편집을 방해한다. 특히 기존 GS 기반 방법들은 반사성(specularities)과 그림자(shadows)가 존재하는 비-Lambertian 조건에서 이러한 구성 요소들을 분리(disentangle)하는 데 어려움을 겪는다.

3DGS의 실용적인 채택을 가로막는 핵심 한계는 다중 뷰 재구성 과정에서 기하학, 재질, 조명 성분이 본질적으로 얽혀 있어 개별 성분의 편집을 불가능하게 만드는 것이다. 3DGS에서의 조명 분해는 분리된 구성 요소를 활용하여 조명 및 재질을 변경하는 장면 합성 등 다목적 GS 편집을 가능하게 하므로 매우 중요한 가치를 지닌다.

### 1.2 핵심 기여 (Contributions)

GS-ID는 **적응적 광원 집합(adaptive light aggregation)**과 **확산 모델 기반 재질 프라이어(diffusion-based material priors)**를 통합한 조명 분해를 위한 엔드-투-엔드 프레임워크를 제안한다. 주변 조명(ambient illumination)을 위한 학습 가능한 환경 맵(learnable environment map)에 더해, 장면 콘텐츠와 공동으로 최적화되는 **비등방성 구형 가우시안 혼합(anisotropic Spherical Gaussian Mixtures, SGMs)**을 사용하여 공간적으로 변화하는 국소 조명(spatially-varying local lighting)을 모델링한다. 그림자를 더 잘 포착하기 위해 각 스플랫(splat)에 여러 광원의 그림자 방향을 인코딩하는 학습 가능한 단위 벡터(learnable unit vector)를 연결하여 재질 및 조명 추정을 더욱 향상시킨다.

GS-ID는 3D Gaussian Splatting에서의 조명 분해를 위한 엔드-투-엔드 프레임워크로서, 적응형 조명 모델, 그림자 제거 모듈(deshadowing module), 사전 훈련된 확산 모델에서 추출한 기하학/재질 프라이어를 통합한다. 또한 수렴성과 효율성을 높이기 위해 CUDA 기반 최적화 파이프라인과 지연 쉐이딩(deferred shading)을 사용한다. 실험 결과 GS-ID는 기존 방법을 능가하는 조명 및 본질적 분해(intrinsic decomposition) 성능을 달성한다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

조명 분해는 세 가지 주요 도전에 직면한 불량 정치 문제(ill-posed problem)이다: (1) 기하학과 재질에 대한 프라이어가 종종 부족하고, (2) 복잡한 조명 조건은 여러 개의 알 수 없는 광원을 포함하며, (3) 수많은 광원으로 표면 쉐이딩을 계산하는 것은 계산 비용이 매우 크다.

현재의 방법론들은 주로 두 가지 패러다임을 따른다: (1) 알 수 없는 조명 조건에서는 암시적 신경 표현(예: NeRF 변형의 MLP 기반 좌표 네트워크) 또는 명시적 파라메트릭 모델을 활용한다. 신경 광장(neural light field)은 MLP를 통한 미분 가능한 표면 재구성을 달성하지만 공간 인식 편집 제어가 부족하다. 명시적 방향성 조명 모델은 직관적인 환경 맵 교체를 가능하게 하지만 국소적 광원 상호작용을 해결하는 데 실패한다.

### 2.2 제안 방법 및 수식

#### 2.2.1 렌더링 방정식 (Rendering Equation)

GS-ID는 Kajiya(1986)의 렌더링 방정식을 기반으로 물리 기반 렌더링(PBR)을 수행한다. 물리 기반 렌더링(PBR)을 사용하여 뷰 종속적 외형을 모델링하고 효과적인 조명 분해를 가능하게 한다. 렌더링 방정식을 따라 표면 점의 나가는 복사(outgoing radiance)를 모델링하며, 여기서 입사 방향과 뷰 방향이 각각 사용된다.

출사 복사는 다음과 같이 표현된다:

$$L_o(\mathbf{x}, \boldsymbol{\omega}_o) = \int_{\Omega} f_r(\mathbf{x}, \boldsymbol{\omega}_i, \boldsymbol{\omega}_o) \, L_i(\mathbf{x}, \boldsymbol{\omega}_i) \, (\boldsymbol{\omega}_i \cdot \mathbf{n}) \, d\boldsymbol{\omega}_i$$

여기서 $\mathbf{x}$는 표면 점, $\boldsymbol{\omega}_i$, $\boldsymbol{\omega}_o$는 각각 입사 및 출사 방향, $f_r$은 BRDF, $L_i$는 입사 복사, $\mathbf{n}$은 표면 법선이다.

#### 2.2.2 BRDF 모델링 (Cook-Torrance)

Cook-Torrance 모델을 사용하여 BRDF(양방향 반사 분포 함수)를 공식화하며, albedo, roughness, metallicity가 각각 사용된다. Normal Distribution Function(NDF)과 기하학 항 $G$는 물리적 재질에서 도출된다.

Cook-Torrance BRDF는 다음과 같이 표현된다:

$$f_r(\mathbf{x}, \boldsymbol{\omega}_i, \boldsymbol{\omega}_o) = \frac{\mathbf{A}}{\pi}(1 - M) + \frac{D(\mathbf{h}) F(\boldsymbol{\omega}_o, \mathbf{h}) G(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o, \mathbf{h})}{4(\boldsymbol{\omega}_i \cdot \mathbf{n})(\boldsymbol{\omega}_o \cdot \mathbf{n})}$$

여기서:
- $\mathbf{A}$: albedo
- $M$: metallicity (금속성)
- $D(\mathbf{h})$: 법선 분포 함수 (Normal Distribution Function, NDF)
- $F$: 프레넬 항 (Fresnel term)
- $G$: 기하학 항 (Geometry term)
- $\mathbf{h} = (\boldsymbol{\omega}_i + \boldsymbol{\omega}_o)/\|\boldsymbol{\omega}_i + \boldsymbol{\omega}_o\|$: 반벡터 (half vector)

#### 2.2.3 적응형 조명 모델 (Adaptive Lighting Model) — SGM

적응형 조명 모델을 사용하여 입사 복사를 다음과 같이 공식화한다: $L_i = L_i^{SGM} + L_i^{env}$, 여기서 $L_i^{SGM}$은 개별 발광체로부터의 고주파 효과를 담당하고, $L_i^{env}$는 먼 광원으로부터의 주변 조명을 담당한다.

구형 가우시안(SG) 함수는 다음과 같이 정의된다:

$$SG(\boldsymbol{\omega}; \boldsymbol{\mu}, \lambda, a) = a \cdot \exp\left(\lambda (\boldsymbol{\mu} \cdot \boldsymbol{\omega} - 1)\right)$$

여기서 $\boldsymbol{\mu}$는 주축(lobe axis), $\lambda$는 sharpness, $a$는 진폭(amplitude)이다.

GS-ID는 구형 가우시안(SGM) 혼합 모델을 사용하여 국소 하이라이트 조명을 표현하며, 여기서 BRDF 함수 $f_r^{SGM}$과 구형 가우시안 함수 $SG$가 사용된다.

SGM 기반 직접 조명 출사 복사는 다음과 같이 표현된다:

$$L_o^{SGMs}(\mathbf{x}, \boldsymbol{\omega}_o) = \sum_{k=1}^{N_{light}} \int_{\Omega} f_{r,k}^{SGM}(\mathbf{x}, \boldsymbol{\omega}_i, \boldsymbol{\omega}_o) \cdot SG_k(\boldsymbol{\omega}_i) \cdot (\boldsymbol{\omega}_i \cdot \mathbf{n}) \, d\boldsymbol{\omega}_i$$

#### 2.2.4 그림자 인식 벡터 (Shadow-Aware Vector)

그림자를 더 잘 포착하기 위해 각 스플랫에 여러 광원이 유발하는 그림자 방향을 나타내는 학습 가능한 단위 벡터를 연결하여 조명 및 재질 추정을 향상시킨다.

각 스플랫 $i$에 대한 학습 가능한 그림자 방향 벡터 $\mathbf{s}_i \in \mathbb{R}^3$, $\|\mathbf{s}_i\| = 1$ 을 도입하며, 이를 통해 가시도(visibility)를 근사한다:

$$V(\mathbf{x}_i, \mathbf{s}_i) = \sigma\left(\text{MLP}(\mathbf{x}_i, \mathbf{s}_i)\right)$$

#### 2.2.5 지연 쉐이딩 (Deferred Shading)

먼저 확산 모델의 법선 프라이어를 활용하여 거친 3DGS 장면을 재구성한 후, 재질 프라이어를 활용하여 공동 최적화를 통해 조명과 본질적 속성을 추정한다. 본질적 속성은 G-Buffer 맵으로 저장되어 지연 쉐이딩에 사용되어 훈련을 가속화한다. SGM 세트와 학습 가능한 환경 맵을 포함하는 적응형 조명 모델을 사용하여 조명을 표현한다.

지연 쉐이딩은 훈련을 가속화하며 포인트 수에 따라 더 빠르게 확장되어 최대 4배의 속도 향상을 달성한다. 또한 미분 가능한 렌더링의 CUDA 구현을 통해 40% 이상의 저장 공간을 절약할 수 있다.

### 2.3 모델 구조 (파이프라인)

GS-ID는 **3단계(three-stage) 프레임워크**로 구성된다:

GS-ID는 3단계로 구성된다. Stage 1에서는 수정된 2DGS와 Omnidata 모델을 사용하여 정확한 기하학을 재구성한다. Stage 2와 Stage 3에서는 2DGS로 렌더링된 G-buffer 패키지를 입력으로 받아 정의된 광원 필드(light field)를 사용한 물리 기반 렌더링을 수행한다. 이 접근법은 장면의 광원 필드에 대한 견고한 이해를 제공하여 사용자 친화적인 편집을 가능하게 한다.

| 단계 | 내용 |
|------|------|
| **Stage 1** | 2DGS + 확산 모델 법선 프라이어로 정밀 기하학 재구성 |
| **Stage 2** | G-Buffer 생성 + Diffusion-guided 재질 프라이어로 albedo, roughness, metallicity 추정 |
| **Stage 3** | SGM 기반 적응적 조명 모델 + 환경 맵 + 그림자 벡터를 통한 공동 최적화 |

```
[멀티뷰 입력 이미지]
       ↓
[Stage 1] 2DGS + Omnidata 법선 프라이어 → 정밀 기하학 (Normal Map)
       ↓
[Stage 2] G-Buffer 생성 + Diffusion Prior → PBR 재질 (Albedo, Roughness, Metallic)
       ↓
[Stage 3] 적응형 조명 모델(SGMs + Env.Map) + Shadow Vector 최적화
       ↓ (Deferred Shading + CUDA 가속)
[출력] Decomposed: Geometry / Material / Illumination
       → Relighting / Scene Composition / Light Editing
```

### 2.4 성능 향상

SGM과 확산 모델의 본질적 프라이어를 결합함으로써 GS-ID는 광원-재질-기하학 상호작용의 모호성을 크게 줄이고 역 렌더링(inverse rendering) 및 재조명(relighting) 벤치마크에서 최첨단(state-of-the-art) 성능을 달성한다. 실험은 또한 재조명과 장면 합성 등 다운스트림 응용에서 GS-ID의 효과를 입증한다.

또한 파라메트릭 광원을 사용하면 추가 훈련 없이 분해된 장면에서 조명을 추출하여 다른 장면에 통합할 수 있다.

**속도 측면:**
지연 쉐이딩은 훈련을 가속화하며 포인트 수에 따라 더 빠르게 확장되어 최대 4배의 속도 향상을 달성한다. 또한 CUDA 구현은 40% 이상의 GPU 메모리를 절약한다.

**조명 편집 측면:**
조명을 위한 구형 가우시안(SG) 혼합의 사용은 장면 조명의 유연하고 직관적인 표현을 제공한다. 훈련 후 광원의 방출 가중치(emission weights), 위치, SG 파라미터를 독립적으로 수정할 수 있다. 이 명시적 발광 공식(emissive formulation)은 물리적으로 그럴듯하고 인터랙티브한 조명 편집을 가능하게 한다.

**장면 합성 측면:**
이 방법은 장면의 조명을 국소 조명과 환경 조명의 조합으로 모델링하여 완전한 조명 분해를 가능하게 한다. 이 분해는 서로 다른 장면 간의 재조명 가능한 콘텐츠의 원활한 통합을 가능하게 한다. 분해된 TensoIR 합성 객체를 실제 Mip-NeRF 장면에 통합하여 현실적인 조명 일관성을 달성함을 시연한다.

### 2.5 한계

이 연구는 조명 분해에서 좋은 성능을 달성했지만, 몇 가지 방향은 추가 탐구가 필요하다. 자연 재질과 기하학은 종종 등방성(isotropic)인 반면, 현재의 3DGS 표현은 비등방성(anisotropic)이어서 여분의 가우시안 형태를 생성한다는 문제가 있다.

확산 프라이어는 빛이 물리적 환경에서 실제로 어떻게 동작하는지의 단순화된 모델이다. 더 발전된 광원 전달 시뮬레이션 기법이 조명 분해의 정확도를 향상시킬 수 있다. 현재의 GS-ID 방법은 현실적인 장면 조명의 중요한 구성 요소인 매우 반사적인 하이라이트 효과를 정확하게 분해하는 데 어려움을 겪을 수 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 확산 모델 프라이어가 일반화에 미치는 기여

조명 분해를 위해서는 법선과 재질의 정확한 추정이 중요하다. 이를 위해 본질적 확산 프라이어(intrinsic diffusion priors)를 도입하여 2DGS의 법선 및 재질 추정을 안내한다.

도입된 프라이어가 법선 추정과 ID 결과를 향상시킨다. 재질 프라이어를 사용할 때와 비교하여 TensoIR 데이터셋에서 albedo 결과가 크게 낮아진다. 특히 재질 프라이어가 없는 경우 새로운 뷰 합성의 품질이 저하된다.

대규모 데이터로 사전 훈련된 확산 모델의 프라이어를 활용함으로써, GS-ID는 특정 장면에만 과적합(overfit)되는 경향을 줄이고, 새로운 장면에서도 합리적인 재질 및 법선 추정을 수행할 수 있는 **일반화된 능력**을 획득한다.

### 3.2 적응형 조명 모델(SGM)의 일반화

학습 가능한 환경 맵이 주변 조명을 포착하는 것 외에도, 최적화 과정에서 비등방성 및 공간적으로 변화하는 구형 가우시안 혼합 세트를 적응적으로 집합하여 복잡한 국소 조명 조건을 모델링한다. 여러 광원이 그림자를 유발하는 방식을 나타내기 위해 각 스플랫과 학습 가능한 단위 벡터를 연결한다. SGM과 확산 모델의 본질적 프라이어를 함께 사용함으로써 GS-ID는 광원-기하학-재질 모호성을 크게 줄이고 최첨단 조명 분해 성능을 달성한다.

SGM이 장면에 따라 **적응적으로** 광원 위치, 개수, 방향을 학습하므로, 단순한 단일 광원 가정에서 벗어나 복잡한 다중 조명 환경에서도 일반화 능력이 높아진다.

### 3.3 그림자 인식 벡터의 일반화 기여

현재의 GS 기반 접근법들은 비-Lambertian 조건 하에서 복잡한 광원-기하학-재질 상호작용을 분리하는 데 있어 특히 정반사와 그림자를 처리할 때 중요한 도전에 직면한다.

각 스플랫에 할당된 그림자 방향 벡터는 다양한 광원 배치 시나리오(실내, 실외, 복수 광원 등)에서도 그림자 방향을 독립적으로 학습할 수 있어 **새로운 조명 환경에 대한 일반화**를 지원한다.

### 3.4 장면 합성에서의 일반화 증거

GS-ID가 정확한 조명 편집을 효과적으로 처리하여 사실적인 렌더링을 생성함을 보여준다. 또한 파라메트릭 광원을 사용하면 추가 훈련 없이 분해된 장면에서 조명을 추출하여 다른 장면에 통합할 수 있다.

이는 **추가 훈련 없이도 새로운 장면에 조명 정보를 전이(transfer)**할 수 있음을 의미하며, 이는 일반화 성능의 중요한 증거이다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려 사항

### 4.1 앞으로의 연구에 미치는 영향

#### (1) 3DGS 기반 역 렌더링(Inverse Rendering) 연구 촉진
GS-ID는 SGM과 확산 모델의 본질적 프라이어를 결합하여 광원-재질-기하학 상호작용의 모호성을 크게 줄이고 역 렌더링 및 재조명 벤치마크에서 최첨단 성능을 달성한다. 이러한 접근 방식은 후속 연구들이 대규모 사전 훈련 모델(LDM, Stable Diffusion 등)의 지식을 3DGS 기반 역 렌더링에 결합하는 방향성을 제시한다.

#### (2) 편집 가능한 3D 표현 연구의 새로운 패러다임
조명을 위한 구형 가우시안(SG) 혼합은 장면 조명의 유연하고 직관적인 표현을 제공하며, 훈련 후 광원의 방출 가중치, 위치, SG 파라미터를 독립적으로 수정할 수 있다. 이러한 **파라메트릭 조명 표현** 패러다임은 3D 콘텐츠 생성, VFX, AR/VR 응용 분야의 편집 워크플로우에 큰 영향을 줄 수 있다.

#### (3) G-Buffer 기반 지연 쉐이딩의 표준화 가능성
본질적 속성은 G-Buffer 맵으로 저장되어 지연 쉐이딩에 사용되며 훈련을 가속화한다. 이 접근 방식은 대규모 실외/실내 장면에 대한 확장성 있는 역 렌더링 파이프라인 설계에 있어 방향성을 제공한다.

#### (4) 다운스트림 응용 확장
이것은 조명에 대한 정밀한 제어가 현실적이고 미적으로 뛰어난 결과를 달성하는 데 중요한 시각 효과(VFX), 제품 시각화, 건축 설계 등 다양한 응용에 중요한 함의를 가질 수 있다.

### 4.2 앞으로 연구 시 고려할 점

#### (1) 간접 조명 및 전역 조명(Global Illumination) 처리
GI-GS의 핵심 통찰은 지연 쉐이딩과 경로 추적(path tracing)을 결합하여 실시간 렌더링을 보장하면서 정확한 전역 조명 분해를 가능하게 하는 것이다. GS-ID는 환경 맵과 SGM을 통해 직접 조명 모델링에 초점을 맞추고 있으므로, 향후 연구는 경로 추적이나 Monte Carlo 샘플링을 통한 **간접 조명(inter-reflection)의 정확한 모델링**을 추가적으로 통합하는 방향을 고려해야 한다.

#### (2) 등방성 가정 문제 해결
자연 재질과 기하학은 종종 등방성(isotropic)인 반면, 현재의 3DGS 표현은 비등방성(anisotropic)이어서 여분의 가우시안 형태를 생성한다. 이를 해결하기 위한 **등방성 제약 정규화** 또는 Surfel 기반 표현의 도입이 연구 과제로 남는다.

#### (3) 동적 장면(Dynamic Scene)으로의 확장
현재 GS-ID는 정적 장면을 가정하고 있다. 동적 조명이나 움직이는 객체가 있는 실세계 장면에 적용하기 위해서는 **시간적 일관성(temporal consistency)**을 유지하는 동적 3DGS 프레임워크와의 통합이 필요하다.

#### (4) 확산 모델 프라이어의 도메인 편향 문제
확산 모델(Omnidata 등)로부터 추출된 프라이어는 학습 데이터의 분포에 의존하므로, 매우 비정형적인 재질(예: 반투명체, 이방성 재질)이나 극단적인 조명 조건에서는 프라이어의 부정확성이 역으로 성능 저하를 유발할 수 있다. 복잡한 기하학에 대해 가우시안 기저 함수를 3D 장면 데이터에 피팅하는 과정은 계산 집약적일 수 있다.

#### (5) 물리적 정확도 향상
확산 프라이어는 빛이 물리적 환경에서 실제로 어떻게 동작하는지의 단순화된 모델이다. 더 발전된 광원 전달 시뮬레이션 기법이 조명 분해의 정확도를 향상시킬 수 있다. Monte Carlo 경로 추적 기반의 SGM 표현, 또는 신경 복사 캐시(Neural Radiance Cache)와의 결합이 고려될 수 있다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 베이스 표현 | 조명 표현 | 그림자/간접광 | 확산 모델 활용 | 발표 |
|------|------------|----------|------------|-------------|------|
| **NeRFactor** (Zhang et al.) | NeRF (MLP) | 환경 맵 | 가시도 필드 | ✗ | SIGGRAPH Asia 2021 |
| **TensoIR** (Jin et al.) | TensoRF | HDR 환경 맵 | Ray Tracing | ✗ | CVPR 2023 |
| **GS-IR** (Liang et al.) | 3DGS | IBL (환경 맵) | Baking 기반 | ✗ | CVPR 2024 |
| **GaussianShader** (Jiang et al.) | 3DGS | 환경 맵 + 잔류 항 | 부분적 | ✗ | CVPR 2024 |
| **Relightable 3D Gaussians** (Gao et al.) | 3DGS | 환경 맵 + 간접 SH | BVH Ray Tracing | ✗ | ECCV 2024 |
| **GI-GS** (Chen et al.) | 3DGS | 환경 맵 | 경로 추적(Path Tracing) | ✗ | arXiv 2024 |
| **GS-ID (Ours)** | 2DGS → 3DGS | **SGM + 환경 맵** | **Shadow-aware 벡터** | **✓** | ICCV 2025 |

기존의 암시적 신경 표현(NeRF 등)을 사용하는 역 렌더링 방법들은 낮은 표현력과 높은 계산 복잡도를 가진다. 3DGS를 역 렌더링에 도입하는 것은 법선 생성의 어려움과 전방 매핑(forward mapping)으로 가려짐(occlusion)을 추적하는 어려움 때문에 도전적이다.

GI-GS는 새로운 뷰 합성에서 이전의 NeRF 기반 및 3DGS 기반 역 렌더링 방법들을 능가하며, 우수한 계산 효율성으로 비교 가능한 재조명을 달성한다.

GS-ID의 차별점은 다음과 같다:
1. **SGM(비등방성 구형 가우시안 혼합)** 기반 적응형 국소 광원 모델링으로 명시적·직관적 조명 편집 가능
2. **확산 모델 프라이어** 활용으로 재질/법선 추정의 모호성을 외부 지식으로 해소
3. **Per-splat 그림자 방향 벡터** 도입으로 다중 광원 그림자 효과를 명시적으로 모델링
4. **G-Buffer + 지연 쉐이딩 + CUDA 최적화**로 훈련 속도 4배 향상, 메모리 40% 절감

---

## 📚 참고 문헌 및 출처

1. **GS-ID 논문 (v2, ICCV 2025)**: Kang Du, Zhihao Liang, Yulin Shen, Zeyu Wang. "GS-ID: Illumination Decomposition on Gaussian Splatting via Adaptive Light Aggregation and Diffusion-Guided Material Priors." ICCV 2025. arXiv:2408.08524 — https://arxiv.org/abs/2408.08524
2. **GS-ID 프로젝트 페이지**: https://kangdu.top/gsid/
3. **GS-ID GitHub**: https://github.com/dukang/GS-ID
4. **GS-ID ICCV 2025 포스터**: https://iccv.thecvf.com/virtual/2025/poster/1934
5. **GS-ID 논문 HTML (v1)**: https://arxiv.org/html/2408.08524v1
6. **GS-ID 논문 HTML (v2)**: https://arxiv.org/html/2408.08524v2
7. **CIS Lab HKUST(GZ) 공식 페이지**: https://cislab.hkust-gz.edu.cn/publications/gs-id/
8. **GS-IR (CVPR 2024)**: Zhihao Liang et al. "GS-IR: 3D Gaussian Splatting for Inverse Rendering." CVPR 2024. arXiv:2311.16473 — https://arxiv.org/abs/2311.16473
9. **GI-GS**: Hongze Chen et al. "GI-GS: Global Illumination Decomposition on Gaussian Splatting for Inverse Rendering." arXiv:2410.02619 — https://arxiv.org/abs/2410.02619
10. **Relightable 3D Gaussians (ECCV 2024)**: Jian Gao et al. "Relightable 3D Gaussians: Realistic Point Cloud Relighting with BRDF Decomposition and Ray Tracing." ECCV 2024. arXiv:2311.16043
11. **GlossGau**: "GlossGau: Efficient Inverse Rendering for Glossy Surface with Anisotropic Spherical Gaussian." arXiv:2502.14129 — https://arxiv.org/abs/2502.14129
12. **RTR-GS**: "RTR-GS: 3D Gaussian Splatting for Inverse Rendering with Radiance Transfer and Reflection." ACM MM 2025. arXiv:2507.07733
13. **aimodels.fyi 논문 요약**: https://www.aimodels.fyi/papers/arxiv/gs-id-illumination-decomposition-gaussian-splatting-via
14. **ResearchGate**: https://www.researchgate.net/publication/383216652
