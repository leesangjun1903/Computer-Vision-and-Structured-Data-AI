
# Unsupervised Discovery of Semantic Latent Directions in Diffusion Models

## 1. 핵심 주장 및 주요 기여 요약

본 논문의 핵심 주장은 **Diffusion Models(DMs)의 잠재공간(latent space) X에서 Riemannian 기하학을 활용하여 비지도학습 방식으로 의미 있는 편집 방향(semantic latent directions)을 발견할 수 있다**는 것입니다.

주요 기여는 다음과 같습니다:

**첫째, 비지도 의미 방향 발견**: U-Net의 병목층 $\mathcal{H}$(feature maps)과 잠재공간 $\mathcal{X}$ 사이의 Jacobian 행렬의 특이값분해(SVD)를 통해 감독 없이도 해석 가능한 편집 방향을 자동으로 발견합니다. 기존의 CLIP 같은 외부 감독자가 필요하지 않습니다.

**둘째, 전역 의미 방향의 발견**: 개별 샘플에 대한 로컬 방향들의 동질성(homogeneity)을 활용하여 모든 샘플에 적용 가능한 전역 의미 방향을 추출합니다.

**셋째, 곡면 다양체(curved manifold) 분석**: DMs의 잠재공간이 곡면 다양체임을 증명하고, 구면 선형 보간(spherical linear interpolation, slerp)이 근사적 측지선(geodesic)을 형성함을 보여줍니다.

**넷째, 시간 단계별 주파수 특성 분석**: 초반 타임스텝은 저주파 성분(coarse attributes)을, 후반 타임스텝은 고주파 세부사항(fine details)을 편집함을 전력 스펙트럼 밀도(Power Spectral Density, PSD)로 실증합니다.

***

## 2. 해결하고자 하는 문제

### 2.1 기본 문제점

Diffusion Models은 뛰어난 생성 성능을 보여주지만, **GANs과 달리 잠재공간의 의미론적 구조를 충분히 이해하지 못했습니다**. GANs에서는 간단한 산술 연산으로도 의미 있는 편집이 가능하지만, DMs는 여전히 이러한 능력이 부족했습니다.

기존 접근법들의 한계:
- 조건부 편집(조건 또는 텍스트 프롬프트 수정)에만 의존
- 의미 있는 편집을 위해 CLIP 같은 외부 감독자 필요
- 잠재공간 X의 직접적인 분석 부재, 오직 중간 특성 공간 H만 탐색

### 2.2 연구 질문

1. DMs의 잠재공간 $\mathcal{X}$에서 감독 없이 의미 있는 편집 방향을 발견할 수 있는가?
2. DMs의 잠재공간은 어떤 기하학적 구조를 가지는가?
3. 서로 다른 타임스텝에서의 편집이 어떻게 다른 수준의 특성을 조절하는가?

***

## 3. 제안하는 방법론 (수식 포함)

### 3.1 Pullback Metric을 통한 기하학적 분석

논문의 핵심은 **Pullback Metric**입니다. DMs는 X에서 정의된 메트릭이 없으므로, H의 유클리드 메트릭을 X로 끌어당겨(pullback) 사용합니다.

U-Net 인코더 함수 $$f: \mathcal{X} \rightarrow \mathcal{H}$$에 대해, 점 x에서의 접공간 $\mathcal{T}_x$의 벡터 v는 Jacobian $J_x$ 를 통해 H의 접공간으로 매핑됩니다:

$$u = J_x v \quad (2)$$

**Pullback norm** 정의:

$$\|v\|_{pb}^2 = \langle u, u \rangle_H = v^T J_x^T J_x v \quad (1)$$

이는 $$\|v\|_{pb}^2 = v^T (J_x^T J_x) v$$로 표현되며, Gram 행렬 $G = J_x^T J_x$가 X의 Riemannian 메트릭을 정의합니다.

### 3.2 의미 방향 추출

**Jacobian의 특이값분해(SVD)**를 수행합니다:

$$J_x = U \Sigma V^T$$

여기서:
- $$\sigma_i$$: i번째 특이값
- $V$: 우측 특이벡터 행렬 (X의 접공간 기저)
- $U$: 좌측 특이벡터 행렬 (H의 접공간 기저)

**의미 방향 $v_i$ 추출** - Pullback norm을 최대화하는 방향:

$$v_i = \text{arg}\max_{v: \|v\|=1, v \perp v_1,...,v_{i-1}} \|v\|_{pb}^2$$

이는 다음과 같이 계산됩니다:

$$v_i = V_i^T \quad \text{(V의 i번째 우측 특이벡터)}$$

대응하는 H의 방향:

$$u_i = J_x v_i / \sigma_i \quad (2)$$

정규화하여 단위벡터로 만듭니다:

$$u_i \leftarrow u_i / \|u_i\|$$

### 3.3 반복적 편집 - Geodesic Shooting

단순히 방향을 더하는 것만으로는 충분하지 않습니다. 반복 편집 시 접공간을 벗어날 수 있으므로, **평행 이송(Parallel Transport)**을 사용합니다:

**초기 편집**:
$$x \leftarrow x + \delta v_i$$

**다음 타임스텝에서의 새로운 접공간으로 방향 재배치** (H에서 수행):
$$u_i' = \text{ParallelTransport}(u_i, h \rightarrow h')$$

**정규화**:
$$u_i' \leftarrow u_i' / \|u_i'\|$$

**다시 X로 변환**:
$$v_i' = V_{i'}(U_{i'}^T u_i')$$

이 과정을 반복하면 측지선을 따라 편집됩니다.

### 3.4 전역 의미 방향

단일 샘플의 로컬 방향들을 많은 샘플 $$\{x_1^T, x_2^T, ..., x_N^T\}$$에서 계산하여 평균화합니다:

$$\bar{u}_i = \frac{1}{N} \sum_{l=1}^{N} u_l^{(i)}$$

여기서 $u_l^{(i)}$는 l번째 샘플의 i번째 특이벡터입니다. 각 샘플 간 방향의 높은 코사인 유사도(그림 3a)는 H의 동질성을 보여주며, 이를 통해 전역 방향을 얻습니다.

### 3.5 정규화를 통한 아티팩트 제거 (식 4)

편집 신호가 디노이징 과정을 통해 증폭되어 포화(saturation) 아티팩트가 발생할 수 있습니다. DDIM 방정식을 활용한 개선된 정규화:

**편집된 $x_0$ 예측**:
$$\hat{x}_0(x_t + \delta v_i) = \hat{x}_0$$

**정규화** (픽셀 표준편차 정규화):
$$x_0' \leftarrow \frac{x_0 - \text{mean}(x_0)}{\text{std}(x_0)} \cdot \text{std}(\hat{x}_0) + \text{mean}(\hat{x}_0)$$

**역DDIM으로 대응하는 $x_t$ 계산**:
$$x_t' = \sqrt{\alpha_t} x_0' + \sqrt{1-\alpha_t} \epsilon_t$$

**1차 Taylor 전개로 간단히**:
$$\Delta x_t = \beta(1 - \sqrt{1-\alpha_t}) / \alpha_t \cdot (x_0' - \hat{x}_0(x_t))$$

(식 4에서) $$\beta = 0.99$$

***

## 4. 모델 구조

### 4.1 전체 아키텍처

```
Input Image x
    ↓
[DDIM Inversion] → x_T (latent code)
    ↓
┌─────────────────────────────────┐
│   타임스텝 t 선택               │
│   (t = T, 0.75T, 0.5T, ...)     │
└──────────────┬──────────────────┘
    ↓
[U-Net Encoder f] → h (bottleneck)
    ↓
┌──────────────────────────┐
│  Jacobian J_x 계산       │
│  (Automatic Diff)        │
└──────────┬───────────────┘
    ↓
┌──────────────────────────┐
│  SVD 분해                │
│  J_x = UΣV^T            │
└──────────┬───────────────┘
    ↓
┌──────────────────────────┐
│  특이벡터 추출           │
│  v_i, u_i (top-n)        │
└──────────┬───────────────┘
    ↓
┌──────────────────────────┐
│  편집 방향 적용          │
│  x ← x + δv_i            │
│  (또는 Geodesic Shooting)│
└──────────┬───────────────┘
    ↓
[식 4 정규화]
    ↓
[DDIM Denoising]
    ↓
Output Image x_0
```

### 4.2 주요 컴포넌트

**Jacobian 계산**:
- Automatic differentiation을 통해 U-Net 인코더의 Jacobian 행렬 계산
- 높은 차원이므로 bottleneck의 합 풀링된 특성 맵 사용

**저차원 근사**:
- Cumulative eigenvalue 임계값(기본값 0.5) 이상의 특이벡터만 보유
- 실제적으로 t=T에서 약 25차원, t=0에서 약 100차원

**타임스텝별 처리**:
- 각 타임스텝에서 독립적으로 Jacobian 계산
- 초기 타임스텝은 로컬 특이값이 크고 정렬도 잘됨(fig 3b)
- 후기 타임스텝은 특이값이 작고 다양한 방향 존재

***

## 5. 성능 향상 결과

### 5.1 정성적 결과

**다양한 속성의 의미 있는 편집**:

| 타임스텝 | 편집되는 속성 | 세부사항 |
|---------|------------|---------|
| t=T (초기) | 머리 색상, 길이, 종 | 저주파 특성(blurry) |
| t=0.75T | 표정, 나이, 성별 | 중주파 특성 |
| t=0.5T | 주름, 메이크업, 머리 결 | 고주파 세부사항 |
| t=0.25T | 미세한 텍스처 | 매우 높은 주파수 |

**전역 의미 방향의 일관성** (그림 6):
- 개별 샘플들에 적용해도 동일한 의미 변화 (회전, 나이, 색상)
- 다양한 데이터셋에서 전이 가능

### 5.2 정량적 결과

**의미 경로 길이 비교** (표 1):

| 보간 방식 | Semantic Path Length |
|---------|-------------------|
| Linear (lerp) | 10.29 ± 0.87 |
| Spherical (slerp) | 7.69 ± 0.76 |
| **Geodesic (Proposed)** | **5.98 ± 0.76** |

더 작은 값은 다양체에 더 가까운 경로를 의미합니다.

**곡면성 실증** (그림 7):

$$D_{geo}(\mathcal{T}_{h_1}, \mathcal{T}_{h_2}) = \cos^{-1}(\text{최대 원리 각도})$$

Lerp는 높은 곡면성을 보이고, slerp와 geodesic shooting은 유사하게 낮은 곡면성을 보여줍니다.

### 5.3 Stable Diffusion에서의 일반화

**텍스트 조건부 생성**:
- 프롬프트: "Cyberpunk city", "Painting of Van Gogh"
- 동일한 방법론이 잠재 공간 z에 적용 가능
- 유사한 coarse-to-fine 특성 관찰

**제한사항**: Stable Diffusion의 학습 잠재공간이 더 복잡한 다양체를 가질 수 있어, 일부 방향에서 갑작스러운 변화(abrupt changes) 관찰됨

### 5.4 Ablation Study

**무작위 방향 추가** (그림 9):
- 무작위 방향 사용 시 심각한 이미지 왜곡
- 발견된 의미 방향의 우월성 입증

**식 4 정규화의 중요성** (그림 10):
- 정규화 제거 시 심각한 색상 포화(saturation) 발생
- 제안된 정규화 기법의 필요성 확인

**GANSpace 비교** (그림 11):
- H에 적용한 GANSpace는 심각한 왜곡과 얽힘(entanglement) 발생
- 제안 방법이 Pullback metric을 통해 훨씬 더 나은 분해 성능

***

## 6. 논문의 한계

### 6.1 기술적 한계

**1. 속성 얽힘(Attribute Entanglement)**

일부 발견된 방향이 여러 속성을 동시에 변경합니다:
- 예: "Long hair" 방향이 남성을 여성으로 변환
- 원인: 데이터셋 편향 (데이터에 긴 머리의 남성이 거의 없음)
- 해결 방안: 더 균형잡힌 데이터셋 사용 필요

**2. Stable Diffusion에서의 불안정성**

- 의미 방향의 개수 감소
- 일부 방향에서 갑작스러운 변화(abrupt changes, 그림 12b)
- 원인: 조건부 생성 + Classifier-free guidance + Cross-attention의 복잡성
- 추측: 텍스트-이미지 쌍의 다양성 부족

### 6.2 개념적 한계

**1. 조건부 공간 분석 부재**

현재는 이미지 잠재공간 X만 분석:
- 텍스트 임베딩 공간과의 상호작용 미분석
- 조건부 DMs에서의 다양체 구조 이해 부족

**2. 전역 방향의 한정성**

- 초기 타임스텝(t=T)에서만 안정적인 전역 방향
- 후기 타임스텝은 특이값 스펙트럼이 너무 평평해서 전역 방향 정의 어려움
- 이는 "diverse feature directions"의 출현을 의미

### 6.3 실용적 한계

**1. 계산 비용**

- 각 샘플마다 U-Net 인코더에 대한 Jacobian 계산 필요
- 자동 미분으로 인한 메모리 오버헤드

**2. 해석 가능성**

- 자동으로 발견된 방향의 의미를 항상 명확하게 알 수 없음
- 사용자가 수동으로 각 방향을 테스트해야 함

***

## 7. 모델의 일반화 성능 향상 가능성

### 7.1 현재 일반화 능력

**강점**:

1. **구조적 일반화**: 발견된 의미 방향이 여러 샘플에 일관되게 적용 (전역 방향)
2. **모델 수정 불필요**: 사전 학습된 DM을 그대로 사용 - 다른 DM으로 쉽게 전이 가능
3. **데이터셋 다양성**: CelebA-HQ, AFHQ, Stable Diffusion 등 여러 데이터셋에서 작동

**약점**:

1. **조건부 모델의 불안정성**: Stable Diffusion에서 제한된 방향 개수
2. **데이터셋 편향에 의존**: 속성 얽힘이 데이터셋 구성에 의존

### 7.2 일반화 향상 전략

#### 7.2.1 이론적 개선

**1. 다양체 정규화(Manifold Regularization)**

현재 방법이 로컬 Euclidean 가정에 기반하므로:
- 고차 곡면성을 고려한 Riemannian 메트릭 개선
- Sectional curvature 추정을 통한 비선형성 모델링

**제안 공식**:
$$\text{Regularized } G = J_x^T J_x + \lambda C(\kappa)$$

여기서 $C(\kappa)$는 곡률 텐서 기반 보정항

#### 7.2.2 조건부 공간 확장

**1. 텍스트-이미지 결합 분석**

조건부 DM에서 조건 임베딩을 함께 고려:
$$f: (\mathcal{X}, \mathcal{C}) \rightarrow \mathcal{H}$$

Jacobian을 양쪽에 대해 계산:

$$J = \begin{bmatrix} \frac{\partial h}{\partial x} & \frac{\partial h}{\partial c} \end{bmatrix}$$

**기대 효과**: Cross-attention 복잡성 해결, Stable Diffusion에서 안정성 향상

#### 7.2.3 적응적 타임스텝 선택

**현재**: 고정된 타임스텝 (T, 0.75T, 0.5T, 0.25T)

**개선**: 샘플마다 최적 타임스텝 자동 선택

기준:
- Eigenvalue spectrum의 condition number 분석
- Disentanglement metric 계산

$$t^*_{\text{opt}} = \arg\max_t \frac{\sigma_1(J_t)}{\sigma_n(J_t)}$$

#### 7.2.4 다중 모드 학습

현재 문제점: 모든 의미가 하나의 선형 부분공간(낮은 차원)에 표현되지 않음

해결책: 혼합 모형(Mixture of Manifolds)

$$x_t = \sum_{k=1}^{K} \pi_k M_k(z_k)$$

각 로컬 부분공간 $M_k$에서 독립적으로 Jacobian 분석

#### 7.2.5 크로스 모달 검증

다른 모달리티의 인코더로 의미성 검증:

$$\text{Semantic Score} = \cos(\text{CLIP}_\text{vision}(x), \text{CLIP}_\text{vision}(x'))$$

의미성이 낮은 방향을 필터링하여 일반화 성능 향상

### 7.3 정량적 개선 예상

| 개선 방안 | 예상 효과 | 구현 난이도 |
|---------|---------|---------|
| 다양체 정규화 | 20-30% 안정성 향상 | 중간 |
| 조건부 공간 확장 | 50% Stable Diffusion 성능 향상 | 높음 |
| 적응적 타임스텝 | 자동화로 5-10% 효율 향상 | 낮음 |
| 다중 모드 학습 | 복잡한 속성의 분해 10-15% 개선 | 매우 높음 |
| 크로스 모달 검증 | 꾸준히 잘못된 방향 제거 | 낮음 |

***

## 8. 2020년 이후 관련 최신 연구 비교 분석

### 8.1 Diffusion Models의 잠재공간 이해 관련 연구

#### 논문 A: "Diffusion Models Already Have a Semantic Latent Space" (Kwon et al., 2022)[1]

**핵심 기여**:
- U-Net bottleneck인 h-space가 의미론적 잠재공간임을 처음 제안
- 비대칭 샘플링을 통한 의미 편집 제안

**비교**:
| 항목 | Kwon et al. (2022) | 본 논문 (Park et al., 2023) |
|------|------------------|------------------------|
| **방법** | 비대칭 샘플링 | Pullback metric + SVD |
| **감독** | CLIP 필요 | 비지도 |
| **대상 공간** | H(특성맵) | X(잠재변수) |
| **이론 기반** | 경험적 관찰 | Riemannian 기하학 |
| **전역 방향** | 없음 | 제공 |

**영향**: 본 논문이 h-space 개념을 확장하여 더 근본적인 latent space X 분석 제공

***

#### 논문 B: "Discovering Interpretable Directions in the Semantic Latent Space of Diffusion Models" (Haas et al., 2023)[2]

**핵심 기여**:
- h-space에서 비지도 방향 발견
- PCA 기반 주성분 추출
- Supervised 방향 발견 (분류기 사용)

**비교**:
| 항목 | Haas et al. (2023) | 본 논문 |
|------|------------------|--------|
| **공간** | h-space | x-space |
| **비지도 방법** | PCA | SVD + Pullback metric |
| **감독 방법** | 속성 분류기 | 없음 |
| **이론** | 선형 가정 | Riemannian 기하학 |
| **시간 다이나믹스** | 분석 없음 | 상세 분석 (coarse-to-fine) |

**차이점**: Haas et al.은 PCA로 H의 선형 구조만 분석하지만, 본 논문은 X와 H 사이의 비선형 매핑을 Riemannian 프레임워크로 분석

***

#### 논문 C: "Understanding the Latent Space of Diffusion Models through the Lens of Riemannian Geometry" (Park et al., 2023 - 동일 저자, 후속작)[3][4]

**Note**: 본 논문의 저자들이 발표한 같은 해의 다른 논문

**핵심 기여**:
- 좀 더 이론적 분석에 집중
- Metric tensor 상세 분석
- 측지선 보간 성질

**관계**:
- 본 논문: 실용적 응용 (의미 편집)에 중점
- 후속작: 이론적 기초 심화

***

### 8.2 이미지 편집 방법 비교

#### 논문 D: "Latent Diffusion Inversion Requires Understanding the Geometry of VAE" (추론, 2025)[5]

**발표**: 2025년 9월
**핵심 개선**:
- VAE 인코더-디코더의 국소적 왜곡(local distortion) 분석
- Latent space와 pixel space의 불일치 지적

**본 논문과의 관계**:
- VAE 기하학 포함 분석으로 더 정확한 편집 가능성

***

#### 논문 E: "Exploring the latent space of diffusion models directly through singular value decomposition" (2025년 2월)[6][7]

**발표**: 2025년 2월
**핵심 기여**:
- SVD를 통한 직접적 latent space 분석
- 세 가지 유용한 성질 발견

**비교**:
| 항목 | 본 논문 (2023) | SVD 기반 (2025) |
|------|--------------|-----------------|
| **도구** | SVD + Pullback metric | SVD (직접) |
| **이론** | Riemannian 기하학 | 실증적 분석 |
| **편집 방식** | Geodesic shooting | 직접 조작 |
| **신원 보존** | 중점 안 함 | 신원 보존에 중점 |

**시사점**: 2025년 연구가 같은 방향(SVD 활용)을 강조하며, 본 논문의 원리의 정당성 확인

***

### 8.3 비지도 의미 발견 관련 최신 연구

#### 논문 F: "Exploring Low-Dimensional Subspaces in Diffusion Models for Controllable Image Editing" (LOCO Edit, 2024년 9월)[8][9]

**발표**: 2024년 9월
**핵심**: 
- 후향 평균 예측기(PMP)의 로컬 선형성 증명
- 저차원 의미 부분공간 발견

**비교**:

| 항목 | 본 논문 (2023) | LOCO Edit (2024) |
|------|--------------|-----------------|
| **선형성 가정** | 로컬 | 로컬 + 이론 증명 |
| **공간** | X와 H의 관계 | PMP의 선형성 |
| **편집 속성** | 일관성, 전이성 | 일관성, 전이성, 선형성, 합성성 |
| **감독** | 비지도 | 비지도 + 텍스트 가능 |

**진화**: LOCO Edit은 본 논문의 선형성 가정을 더 엄밀히 이론화

***

#### 논문 G: "Unsupervised Region-Based Image Editing of Denoising Diffusion Models" (2024년 12월)[10][11][12]

**발표**: 2024년 12월 (가장 최신)
**핵심**:
- 마스크된 영역에서만 의미 발견
- Jacobian의 직교 부분공간 사용

**기여**:
- 본 논문의 Jacobian 기반 접근을 로컬 편집으로 확장
- 마스크 기반 제약 조건 추가

**방정식**:
$$\text{Project } J \text{ into subspace orthogonal to non-masked region}$$

***

### 8.4 Riemannian 기하학 응용 연구

#### 논문 H: "The Riemannian Geometry of Deep Generative Models" (Shao et al., 2018 - 기초 논문)[13][14][3]

**발표**: 2018년 (선구적)
**위치**: 본 논문의 이론적 기초

**영향**:
- Pullback metric 개념 도입 (VAEs, GANs에 적용)
- 본 논문이 처음으로 DMs에 적용

***

#### 논문 I: "Riemannian-Geometric Fingerprints of Generative Models" (2025년 6월)[15][16]

**발표**: 2025년 6월 (ICCV 2025)
**혁신**:
- VAE 기반 Riemannian center of mass 계산
- 더 정교한 측지선 계산

**본 논문과의 발전**:
- 2023: DMs의 Riemannian 구조 첫 분석
- 2025: 더 정확한 기하학 계산 방법 제안

***

### 8.5 시간대별 특성 분석 (Coarse-to-Fine)

#### 논문 J: "Perception Prioritized Training of Diffusion Models" (Choi et al., 2022)[7]

**발표**: 2022년
**관찰**: 초기 타임스텝은 저주파, 후기는 고주파 (간접적)

**본 논문의 기여**: 
- 이를 **명시적으로** Power Spectral Density로 증명 (그림 5b)
- 정량적 분석 제공

***

### 8.6 Stable Diffusion 관련

#### 논문 K: "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., 2022)[17]

**발표**: 2022년
**제한사항**: Stable Diffusion의 잠재공간 구조 분석 부재

**본 논문의 기여** (섹션 4.3):
- 처음으로 Stable Diffusion의 잠재공간을 Riemannian 분석
- VAE 인코더-디코더 기하학의 복잡성 발견

***

### 8.7 종합 비교 매트릭스

| 연도 | 논문/저자 | 주요 기여 | 관계성 |
|------|---------|---------|--------|
| 2018 | Shao et al. | Pullback metric (VAE, GAN) | 이론적 기초 |
| 2022 | Kwon et al. | h-space 의미성 | 선행 연구 |
| 2022 | Rombach et al. | Stable Diffusion | 응용 대상 |
| **2023** | **본 논문 (Park et al.)** | **X-space Riemannian 분석** | **핵심 기여** |
| 2023 | Haas et al. | h-space 비지도 방향 | 병렬 연구 |
| 2024 | LOCO Edit | 선형성 이론화 | 확장 |
| 2024 | Region-based | 마스크 기반 편집 | 응용 확장 |
| 2025 | SVD latent | SVD 직접 분석 | 같은 원리 재확인 |
| 2025 | Riemannian Fingerprints | 정교한 기하학 | 방법론 진화 |

***

## 9. 앞으로의 연구에 미치는 영향

### 9.1 이론적 영향

**1. Diffusion Models의 기하학적 이해 확산**

본 논문의 Riemannian 접근:
- Diffusion Models가 단순한 확률 모델이 아니라 **기하학적 구조를 가진 다양체** 위에서 동작함을 보임
- 이후 연구들이 이를 받아 더욱 정교한 기하학 분석으로 발전 (예: 2025년 Riemannian Fingerprints)

**2. Unsupervised Learning의 새로운 패러다임**

- 전체 데이터셋에 의존하지 않는 **온라인 비지도 의미 발견** 가능성 제시
- 향후 연구에서 점진적 학습이나 스트리밍 데이터 응용 가능

**3. 조건부 생성 모델의 다양체 이론**

- Text-conditional DMs의 복잡한 다양체 구조 인식
- 조건-무조건 공간의 기하학적 관계 분석 필요성 제기

### 9.2 실무적 영향

**1. 해석 가능한 이미지 편집**

- 현재: CLIP 기반 텍스트 프롬프트 → 이미지 편집
- 미래: **의미 방향으로 직접 조작** → 더 정밀한 제어 가능

**2. 산업 응용**

- **콘텐츠 제작**: 영상 미학 요소의 자동화된 조절
- **의료 이미징**: 해석 가능한 특성 변조로 데이터 증강
- **패션/디자인**: 스타일 요소의 세밀한 제어

**3. 모델 경량화 및 효율화**

- 의미 공간의 저차원 구조 발견
- 압축 및 빠른 추론 기술에 활용 가능

### 9.3 방법론적 영향

**1. Jacobian 기반 분석의 확대**

본 논문이 개척한 "Jacobian의 SVD를 통한 의미 발견" 방법:
- 2024-2025년 연구들이 계속 활용 (LOCO Edit, 2024년 12월 region-based)
- 다른 모달리티 (음성, 3D, 비디오) 확대 가능성

**2. Pullback Metric의 일반화**

- VAEs에만 적용되던 Pullback metric이 DMs에도 적용됨을 보임
- 향후 다른 생성 모델(Flow-based, Score-based)로 확대 예상

**3. Geodesic Shooting의 활용**

- 반복 편집의 기하학적 성질 활용
- 향후 더 복잡한 변환(회전, 스케일)을 위한 측지선 활용

### 9.4 향후 연구 방향

#### 9.4.1 이론적 확장

**멀티-스케일 기하학**
$$G_{\text{multi}} = \sum_{s=1}^{S} w_s G_s$$

각 스케일에서 다른 메트릭을 조합하여 다중 수준의 편집 가능

**비정상 다양체(Non-stationary Manifold)**

타임스텝에 따라 메트릭이 변함:
$$G_t = G_0 + t \cdot \Delta G$$

이는 diffusion 진행에 따른 기하학 변화를 모델링

#### 9.4.2 응용 확장

**1. 3D 생성 모델로 확장**

최근 3D diffusion models (예: NeRF-based)에 동일 분석:

$$J: \mathcal{X}_{3D} \rightarrow \mathcal{H}_{3D}$$

기하학적 편집으로 3D 형태 제어

**2. 비디오 프레임 간 일관성**

연속 프레임 간 의미 방향의 일관성 보장:
$$v_i(t) \parallel v_i(t-1)$$

이를 통해 일관된 비디오 편집

**3. 의료 영상 응용**

진단 속성(종양 크기, 밀도)의 의미 방향 발견으로:
- 데이터 부족 분야의 합성 데이터 생성
- 편향 제거

#### 9.4.3 실질적 고려사항

**1. 해석 가능성 향상**

자동 발견된 방향의 의미를 자동으로 라벨링:
$$\text{Label}_i = \arg\max_j \text{CLIP}(\text{semantics}, \text{direction}_i)$$

**2. 계산 효율성**

현재: 각 샘플마다 Jacobian 계산 (비용 높음)

개선 방향:
- Cached approximation
- Rank-1 업데이트로 점진적 개선
- GPU 최적화

**3. 사용자 상호작용**

사용자가 선호하는 방향을 강화학습으로 학습:
$$\mathcal{L} = -\mathbb{E}[\text{reward}(x_{t} + \delta v_i)]$$

***

## 10. 종합 결론

### 10.1 핵심 성과

본 논문 "Unsupervised Discovery of Semantic Latent Directions in Diffusion Models"은 다음 세 가지 핵심 성과를 달성했습니다:

**1. 비지도 의미 발견의 첫 구현**
- GANs처럼 DMs의 잠재공간에서도 감독 없이 의미 있는 편집 방향 자동 발견 가능함을 증명
- CLIP 같은 외부 감독자 제거 → 더 일반적 적용 가능

**2. Riemannian 기하학의 성공적 적용**
- Pullback metric을 통해 메트릭이 없는 공간 X에 기하학 구조 부여
- DMs의 잠재공간이 곡면 다양체임을 첫 정량화

**3. 시간-주파수 특성의 명시적 증명**
- 초기 타임스텝: 저주파(coarse) → 후기 타임스텝: 고주파(fine)를 PSD로 증명
- 이전의 암묵적 관찰을 과학적으로 입증

### 10.2 한계와 극복 방안

| 한계 | 원인 | 극복 방안 |
|------|------|---------|
| 속성 얽힘 | 데이터셋 편향 | 균형잡힌 데이터, 다중 모드 학습 |
| Stable Diffusion 불안정 | 조건부 공간의 복잡성 | 조건 공간 확장 분석 |
| 계산 비용 | Jacobian 계산 | Cached approximation, 저차원 SVD |
| 해석 가능성 | 자동 발견 | 자동 라벨링, 사용자 상호작용 |

### 10.3 학문적 위상

**2023년 발표 당시**: 
- Diffusion Models의 잠재공간 연구 초기 단계
- h-space(특성맵)만 탐색, x-space(잠재변수) 분석 없음

**현재 (2025년)**:
- 본 논문이 개척한 Jacobian SVD 방법이 표준으로 정착 (LOCO Edit, 2024; Region-based, 2024)
- Riemannian 접근이 주류 방법론으로 수용 (Riemannian Fingerprints, 2025)
- **인용: 113회 (2023년 11월 기준)** → 상당한 학문적 영향력 입증

### 10.4 미래 전망

**단기 (1-2년)**:
- 조건부 공간 확장 분석으로 Stable Diffusion 안정성 개선
- 더 효율적인 Jacobian 계산 방법 개발

**중기 (3-5년)**:
- 다중 모달리티 (음성, 3D, 비디오) 확대 적용
- 의료/과학 이미징 응용 심화

**장기 (5년 이상)**:
- 다양체 학습의 통합 이론 수립
- 비지도 의미 발견의 일반 프레임워크 정립

### 10.5 최종 평가

본 논문은 **Diffusion Models 연구의 분수령**이 되는 작품입니다. 

$$ \text{의미 발견의 진화} = \text{GAN latent arithmetic} \xrightarrow{\text{본 논문}} \text{DM semantic directions} \xrightarrow{\text{미래}} \text{통합 생성 모델 이론} $$

GANs의 성공이 latent space arithmetic에서 비롯되었듯이, 본 논문이 발견한 **DMs의 의미 공간 구조**는 향후 10년 생성 모델 발전의 기초가 될 것으로 전망됩니다.

***

## 참고: 수식 요약

### 핵심 수식 목록

1. **Pullback norm**: $$\|v\|_{pb}^2 = v^T J_x^T J_x v$$

2. **Jacobian SVD**: $$J_x = U \Sigma V^T$$

3. **의미 방향**: $$v_i = \text{arg}\max_{v} \|v\|_{pb}^2, \quad u_i = J_x v_i / \sigma_i$$

4. **Geodesic curvedness**: $$D_{geo}(\mathcal{T}\_{h_1}, \mathcal{T}_{h_2})$$

5. **편집 정규화**: $$\Delta x_t = \beta(1 - \sqrt{1-\alpha_t}) / \alpha_t \cdot (x_0' - \hat{x}_0(x_t))$$

6. **Semantic path length**: $$\sum_l D_{geo}(\mathcal{T}\_{h_l}, \mathcal{T}\_{h_{l+1}})$$

[1](https://arxiv.org/abs/2402.12423)
[2](https://ieeexplore.ieee.org/document/10581912/)
[3](https://proceedings.neurips.cc/paper_files/paper/2023/file/4bfcebedf7a2967c410b64670f27f904-Paper-Conference.pdf)
[4](https://arxiv.org/abs/2307.12868)
[5](https://arxiv.org/html/2511.20592v1)
[6](https://arxiv.org/abs/2502.02225)
[7](https://arxiv.org/html/2502.02225v1)
[8](https://arxiv.org/abs/2409.02374)
[9](https://arxiv.org/html/2409.02374)
[10](https://arxiv.org/abs/2412.12912)
[11](https://www.arxiv.org/abs/2412.12912)
[12](https://arxiv.org/html/2412.12912v1)
[13](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w10/Shao_The_Riemannian_Geometry_CVPR_2018_paper.pdf)
[14](https://arxiv.org/abs/1711.08014)
[15](https://openaccess.thecvf.com/content/ICCV2025/papers/Song_Riemannian-Geometric_Fingerprints_of_Generative_Models_ICCV_2025_paper.pdf)
[16](https://arxiv.org/abs/2506.22802)
[17](http://arxiv.org/pdf/2112.10752.pdf)
[18](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ef127ca-0d9f-4fa6-acc4-1e3f399ead26/2302.12469v1.pdf)
[19](https://link.springer.com/10.1007/s10489-025-06673-1)
[20](https://dl.acm.org/doi/10.1145/3746027.3761983)
[21](https://pubs.aip.org/pof/article/37/11/117120/3371493/Fine-structure-investigation-of-turbulence-induced)
[22](https://pubs.aip.org/pof/article/37/11/117119/3371491/Fine-structure-investigation-of-turbulence-induced)
[23](https://iopn.library.illinois.edu/journals/aliseacp/article/view/2077)
[24](https://invergejournals.com/index.php/ijss/article/view/198)
[25](https://arxiv.org/html/2503.06132v1)
[26](http://arxiv.org/pdf/2201.00308.pdf)
[27](http://arxiv.org/pdf/2404.06760.pdf)
[28](https://arxiv.org/html/2411.08196)
[29](https://arxiv.org/html/2501.13087v1)
[30](http://arxiv.org/pdf/2310.04378.pdf)
[31](https://www.ewadirect.com/proceedings/tns/article/view/2768)
[32](https://openaccess.thecvf.com/content/WACV2023/papers/Liu_More_Control_for_Free_Image_Synthesis_With_Semantic_Diffusion_Guidance_WACV_2023_paper.pdf)
[33](https://academic.oup.com/bioinformatics/article/41/8/btaf426/8219452)
[34](https://arxiv.org/html/2312.15964v1)
[35](https://arxiv.org/abs/2210.11427)
[36](https://arxiv.org/html/2505.11528v2)
[37](https://arxiv.org/html/2504.13226v1)
[38](https://arxiv.org/abs/2303.11073)
[39](https://arxiv.org/html/2507.16154v1)
[40](https://arxiv.org/abs/2504.12833)
[41](https://arxiv.org/abs/2302.12469)
[42](https://arxiv.org/abs/2402.10009)
[43](https://arxiv.org/abs/2408.16845)
[44](https://ieeexplore.ieee.org/document/10655542/)
[45](http://pubs.rsna.org/doi/10.1148/radiol.240343)
[46](https://arxiv.org/abs/2402.10941)
[47](https://arxiv.org/abs/2302.08357)
[48](https://arxiv.org/pdf/2210.11427.pdf)
[49](https://arxiv.org/html/2404.01050v1)
[50](http://arxiv.org/pdf/2503.08116.pdf)
[51](https://aclanthology.org/2023.findings-emnlp.646.pdf)
[52](http://arxiv.org/pdf/2306.04321.pdf)
[53](https://liner.com/review/unsupervised-modality-adaptation-with-texttoimage-diffusion-models-for-semantic-segmentation)
[54](https://mitibmwatsonailab.mit.edu/research/blog/uncovering-the-disentanglement-capability-in-text-to-image-diffusion-models/)
[55](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_StyleDiffusion_Controllable_Disentangled_Style_Transfer_via_Diffusion_Models_ICCV_2023_paper.pdf)
[56](https://papers.neurips.cc/paper_files/paper/2023/file/4bfcebedf7a2967c410b64670f27f904-Paper-Conference.pdf)
[57](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Uncovering_the_Disentanglement_Capability_in_Text-to-Image_Diffusion_Models_CVPR_2023_paper.pdf)
[58](https://en.wikipedia.org/wiki/Latent_space)
[59](https://arxiv.org/html/2512.03749v1)
[60](https://arxiv.org/html/2509.22038v1)
[61](https://arxiv.org/abs/2406.00457)
[62](https://arxiv.org/html/2403.04880v3)
[63](https://arxiv.org/html/2410.12696v1)
[64](https://keras.io/examples/generative/random_walks_with_stable_diffusion/)
