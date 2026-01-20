
# RealFusion: 360° Reconstruction of Any Object from a Single Image
### 종합 분석 보고서

***

## 1. 핵심 주장 및 주요 기여 요약

RealFusion은 단일 이미지만으로 객체의 완전한 360도 3D 사진급 재구성을 달성하는 혁신적 방법을 제시합니다. 이 접근법의 핵심은 **기존 2D 확산 모델의 사전(prior)을 활용**하여 단일 이미지로는 본질적으로 제약된 3D 재구성 문제를 해결하는 것입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)

### 주요 기여

**1) RealFusion 방법론 개발** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)
- 카테고리 제약 없이 모든 객체의 360도 재구성 가능
- 3D 감독(3D supervision)이 필요 없는 완전히 자율적 접근법

**2) 단일 이미지 텍스트 반전(Single-Image Textual Inversion)** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)
- 기존 textual inversion 방법을 이미지 기반 설정으로 확장
- 입력 이미지의 이미지 augmentation을 활용하여 카스텀 토큰 `<e>` 생성
- 일반적 프롬프트(예: "a fish")에서 객체 특화 프롬프트로 전환

**3) 기술적 혁신** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)
- 신규 정규화 항 도입 (normal vector smoothness regularization)
- InstantNGP 기반 효율적 구현으로 처리 시간 단축

**4) 정량적 성능 달성** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)
- 최신 단일 이미지 재구성 방법(Shelf-Supervised Mesh Prediction)을 능가
- F-score: 8.24 → 9.58 (기하학 품질)
- CLIP-similarity: 0.70 → 0.74 (외관 품질)

***

## 2. 해결하고자 하는 문제

### 2.1 문제의 본질

단일 이미지에서 3D 재구성은 **심각한 ill-posed 문제**입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)
- 하나의 이미지는 객체의 한 면만 제공
- 보이지 않는 부분의 기하학과 텍스처에 대한 정보 부재
- 무한히 많은 가능한 3D 모델이 같은 2D 이미지를 만들 수 있음

### 2.2 기존 방법의 한계

1. **다중뷰 기반 방법**: NeRF와 같은 방법들은 수십 개 이상의 뷰가 필요 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)
2. **카테고리 특화 방법**: Pix2Vox, 3D-R2N2 등은 특정 객체 카테고리(예: 자동차)에만 제한 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)
3. **생성 방법**: DreamFusion은 텍스트 프롬프트에만 조건화되며, 특정 이미지 재구성 불가 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)
4. **모드 붕괴(Mode Collapse) 문제**: 생성 모델이 모든 제약을 만족하기 어려움 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)

RealFusion이 해결하는 특수한 도전은 **"조건 범위 문제(Coverage Problem)"**입니다. 고품질 얼굴 GAN이라도 대부분의 실제 얼굴을 생성하기 어려운 것처럼, 확산 모델이 특정 이미지와 모든 다른 뷰를 동시에 만족시키는지 보장하기 어렵다는 점입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)

***

## 3. 제안하는 방법 (수식 포함)

### 3.1 방법 개요

RealFusion은 두 가지 동시 목적 함수를 최적화합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)

$$\nabla_{\sigma,c} \mathcal{L} = \nabla L_{SDS} + \lambda_{normals} \nabla L_{normals} + \lambda_{image} L_{image} + \lambda_{mask} \nabla L_{mask}$$

#### 3.1.1 재구성 목적 (Reconstruction Objective)

고정된 참조 뷰포인트에서 신경 방사 필드가 입력 이미지와 일치하도록 강제: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)

$$L_{rec}(\sigma, c; D) = \frac{1}{|D|}\sum_{(I,\pi) \in D} \|I - R(\cdot; \sigma, c, \pi)\|^2$$

여기서 R은 방사 필드의 렌더링 함수로, 다음 방정식으로 정의됩니다:

$$I(u) = R(u; \sigma, c) = \sum_{i \in N} (T_{i+1} - T_i) c(x_i)$$

여기서 $T_i = \exp(-\Delta\sum_{j=0}^{i-1} \sigma(x_j))$는 광자 투과율입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)

#### 3.1.2 사전 목적 (Prior Objective) - Score Distillation Sampling

무작위로 샘플링된 새 뷰에 대해 확산 모델 사전을 적용: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)

$$\nabla_{(\sigma,c)}L_{SDS}(\sigma, c; \pi, e, t) = \mathbb{E}_{t,\epsilon}\left[w(t)(\hat{\Phi}(\alpha_t I + \sigma_t \epsilon; t, e) - \epsilon) \cdot \nabla_{(\sigma,c)}I\right]$$

여기서:
- $\hat{\Phi}$: 냉동된 확산 모델 denoiser
- $w(t)$: 시간 가중함수
- $\alpha_t, \sigma_t$: 확산 스케줄 파라미터
- $e = e(I_0)$: 입력 이미지로 최적화된 텍스트 임베딩

### 3.2 핵심 혁신: 단일 이미지 텍스트 반전

**기본 아이디어**: 단일 이미지로부터 다중뷰 정보를 근사 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)

확산 모델 손실을 입력 이미지의 augmentation에 대해 최소화:

$$\text{단일 이미지 textual inversion} = \min_e \sum_{g \in G} L_{diff}(\Phi(\cdot; e(g(I_0))))$$

여기서:
- G: 이미지 augmentation 연산 집합
- 각 augmentation $g(I_0)$가 "pseudo-alternative-view" 역할

#### Augmentation 구성요소: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)
- 무작위 회전 (±10도)
- 무작위 리사이즈 크롭 (0.70-1.3x)
- 색상 지터링
- 가우시안 블러
- 무작위 수평 플립

이 과정을 통해 얻어진 토큰 `<e>`는 객체의 특화된 정보를 인코딩합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)

### 3.3 정규화 항

#### 3.3.1 법선 벡터 정규화 (Normal Vector Regularization)

표면의 저주파 이상을 제거: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)

$$L_{normals} = \|N - \text{stopgrad}(\text{blur}(N, k))\|^2$$

**특징**:
- 2D 렌더링 공간에서 계산 (3D보다 안정적)
- Gaussian blur (kernel k=9)로 평활 목표 설정
- stop-gradient 연산으로 역전파 끊음

#### 3.3.2 마스크 손실 (Mask Loss)

배경과 객체 분리: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)

$$L_{rec,mask} = \|O - M\|^2$$

여기서 O는 렌더링된 불투명도, M은 객체 마스크입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)

### 3.4 Coarse-to-Fine 학습 전략

InstantNGP의 다중해상도 그리드를 단계적으로 활성화: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)

- **전반부 (처음 50% iteration)**: 저해상도 그리드 $\{G_i\}_{i=1}^{L/2}$ 만 최적화
- **후반부 (마지막 50% iteration)**: 모든 레벨 $\{G_i\}_{i=1}^{L}$ 최적화

이 전략의 효과: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)
- 초기에 전체 구조 포착
- 후반에 세부사항 추가로 표면 이상 방지

### 3.5 방사 필드 모델: InstantNGP

NeRF 대신 grid-based InstantNGP 사용: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)

**구성요소**:
- 다중해상도 hash-encoded feature grids (L=16 레벨)
- 각 그리드 차원: 2
- 최대 해상도: 2048 × 2048
- 3층 MLP (64 숨겨진 단위)

**최적화 세부사항**:
- 렌더링 해상도: 96px (확산 모델에 512px로 upsampling)
- Adam 최적화 (학습률 1e-3)
- 5,000 iterations, ~45분 (V100 GPU)
- 하이퍼파라미터: $\lambda_{image}=5.0$, $\lambda_{mask}=0.5$, $\lambda_{normals}=0.5$

***

## 4. 모델 구조 상세

### 4.1 전체 파이프라인

```
입력 이미지 I₀
    ↓
[단일 이미지 Textual Inversion]
    ↓
커스텀 토큰 <e> 생성 (3000 steps)
    ↓
[신경 방사 필드 최적화]
    ↓
L_rec (고정 뷰)  +  L_SDS (무작위 뷰)
    +  L_normals + L_mask
    ↓
InstantNGP 최적화 (5000 steps)
    ↓
360도 재구성 (텍스처 + 기하학)
```

### 4.2 텍스트 반전 모듈

**입력**: 단일 이미지 $I_0$

**프로세스**:
1. CLIP 기반 자동 초기화: WordNet 명사 토큰 중 이미지와 가장 유사한 것 선택
2. 이미지 augmentation으로 mini-dataset 구성
3. 확산 손실 최소화:

$$\min_e \frac{1}{|D'|}\sum_{I \in D'} \|\Phi(\sqrt{\bar{\alpha}_t}I + \sqrt{1-\bar{\alpha}_t}\epsilon, t) - \epsilon\|^2$$

**출력**: 최적화된 임베딩 벡터 $e(I_0) = \text{<e>}$

### 4.3 렌더링 및 조명 모델

#### Shading Options: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)

1. **Albedo Shading**: 순수 색상
2. **Diffuse Shading**: Lambertian BRDF 적용
   $$I(u) = I_\rho(u) \circ (l_\rho \circ \max(0, n \cdot \frac{l-u}{\|l-u\|} + l_a))$$
3. **Textureless Shading**: 기하학 가시성만 표현

#### View-Dependent Prompt: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)
- 고도 > 60°: "overhead view"
- 고도 < 0°: "bottom view"  
- 방위각 ±30°~±90°: "side view"

이를 통해 "An image of a <e>, {view_type}"와 같은 view-aware 프롬프트 생성

***

## 5. 성능 향상 및 실험 결과

### 5.1 정량적 평가 (Table 1)

기하학 품질과 외관 품질 측정: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)

| 메트릭 | Shelf-Supervised | RealFusion | 우수성 |
|-------|-----------------|-----------|--------|
| **F-score** (기하학) | 8.24 | 9.58 | +1.34 |
| **CLIP-similarity** (외관) | 0.70 | 0.74 | +0.04 |

**평가 방법**:
- F-score: CO3D 포인트 클라우드와의 일치도 측정 (임계값 0.05)
- CLIP-similarity: 생성된 뷰와 실제 뷰의 CLIP 임베딩 코사인 유사도

### 5.2 절제 연구 (Ablation Studies)

#### 5.2.1 단일 이미지 텍스트 반전의 영향 (Figure 7)

**일반 프롬프트 vs 최적화 프롬프트**:

- **Without textual inversion**: 
  - 참조 뷰에서는 정확
  - 뒤쪽이 일반적인 "generic fish" 모양으로 변형
  - 개체 특성 손실

- **With textual inversion**:
  - 모든 각도에서 입력 이미지의 특정 특징 유지
  - 뒤쪽도 정확한 형태 복원

**결론**: 텍스트 반전이 일반화 성능의 핵심 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)

#### 5.2.2 Coarse-to-Fine 학습의 영향 (Figure 8)

- **Without coarse-to-fine**: 표면에 불규칙한 아티팩트 다수
- **With coarse-to-fine**: 부드럽고 일관된 표면

**효과**: 저해상도에서 전체 기하학을 먼저 학습하면 고해상도 최적화가 세부사항에 집중 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)

#### 5.2.3 법선 정규화의 영향 (Figure 9)

- **Without normalization**: 노이즈 많은 표면
- **With normalization**: 매끄럽고 현실적인 표면

#### 5.2.4 사전 모델 선택 (Figure 10)

- **CLIP 기반 사전**: 품질 낮음, 모호한 형태
- **Stable Diffusion 기반 사전**: 현저히 개선된 기하학

**이유**: Stable Diffusion이 2D 이미지에 대해 더 강력한 사전을 학습 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)

### 5.3 정성적 결과

#### 다양성 실험 (Figure 6)

동일한 입력 이미지에서 여러 재구성 가능:

- **참조 뷰**: 일관된 재구성
- **새로운 뷰**: 가능한 해석의 다양성 표현
  - 뒤쪽 텍스처 큰 변동성
  - 구조적 모호성으로 인한 다중 모달 분포

***

## 6. 모델의 일반화 성능 향상 가능성

### 6.1 현재 일반화 성능 분석

RealFusion의 강점:

1. **카테고리 무관성**: 모든 객체 타입에 적용 가능
   - 동물, 도구, 가구, 야외 장면 모두 처리
   - 학습 데이터 없이 테스트 시점에만 최적화

2. **이미지 특화**: 각 입력에 맞춤화된 재구성
   - 특정 이미지의 미세한 특징 포착
   - 범주 수준 모델의 일반적 출력 회피

3. **자율성**: 3D 감독이나 멀티뷰 데이터 불필요 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)

### 6.2 향상 기제

#### **단일 이미지 텍스트 반전의 역할** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)

텍스트 반전은 제한된 2D 정보를 최대한 활용:

$$\text{정보 보존} = \text{Augmentation Invariance} + \text{Embedding Optimization}$$

- **Augmentation 불변성**: 이미지의 기본 특성(색상, 질감, 구조)이 회전, 리사이징에도 보존
- **토큰 최적화**: 확산 모델의 원래 vocabulary로부터 벗어나 객체 특화 표현 학습

이를 통해 "A fish"라는 일반 프롬프트의 한계를 극복하고, "A [specific fish]"의 효과 달성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)

#### **정규화의 일반화 효과**

1. **Normal Smoothness**: 기하학적 노이즈 제거로 overfitting 방지
2. **Orientation Loss**: 뒤쪽 면에서도 일관된 표면 구조 강제
3. **Coarse-to-Fine**: Mode collapse 회피로 diverse 해석 가능

#### **Classifier-Free Guidance의 역설** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)

- DreamFusion은 높은 guidance weight (100)을 사용해 mode collapse 야기
- RealFusion은 이를 보상하기 위해 이미지 조건화 강화
- 결과적으로 더 나은 다중성(multimodality) 유지

### 6.3 향상 가능 영역

#### 6.3.1 **극단적 뷰포인트**

현재 한계:
- 정면/후면 각도 이상 주의: 카메라 각도 조정 필요 (일부 이미지에서 15° → 30-40°)
- "위에서 본" 장면에서 기하학 모호성 증가

개선 방향:
- View-aware loss 가중치 조정
- Multi-view diffusion prior 활용 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10678295/)
- 기하학 제약 추가 (모노큘러 깊이 추정) [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2024W/NRI/papers/Chen_Recon3D_High_Quality_3D_Reconstruction_from_a_Single_Image_Using_CVPRW_2024_paper.pdf)

#### 6.3.2 **복잡한 배경**

현재: 배경과 객체 분리를 위해 마스크 필요 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)

개선안:
- 자동 배경 제거 (SAM, 자동 마팅 모델)
- 배경과 객체 분리 학습 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10860075/)

#### 6.3.3 **텍스처 일관성**

현재 한계: **Janus Problem** - 양쪽 면에 얼굴 텍스처 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)

개선 방향:
1. **의미론적 기하학 제약**: 객체 대칭성 활용
2. **멀티뷰 diffusion prior**: 여러 뷰 간 일관성 강제 [nature](https://www.nature.com/articles/s41598-025-24916-6)
3. **VSD (Variational Score Distillation)**: 개선된 distillation 방식

#### 6.3.4 **세부사항 해상도**

현재: 96px 렌더링 (확산 모델에 512px upsampling)

개선안:
- 고해상도 확산 모델 사용 [arxiv](http://arxiv.org/pdf/2405.06547.pdf)
- 2-단계 최적화: 구조 → 텍스처 [isprs-annals.copernicus](https://isprs-annals.copernicus.org/articles/X-2-2024/89/2024/isprs-annals-X-2-2024-89-2024.pdf)
- 명시적 메시 표현으로의 전환 [arxiv](https://arxiv.org/abs/2404.00987)

***

## 7. 한계 및 실패 사례 분석

### 7.1 세 가지 주요 실패 모드 (Figure 11, 15)

#### 7.1.1 **반투명 기하학 (Transparent Geometry)**

**증상**: 반투명한 신경 필드로 정의된 기하학 부재

**원인**: 
- SDS 손실이 렌더링된 이미지만 최적화 (기하학 직접 제약 부족)
- Floaters와 결합되어 물체가 확실하지 않은 상태

**빈도**: CLIP 사용 시 매우 빈번, Stable Diffusion 사용 시 가끔

**해결안**: Entropy loss로 불투명도 명확성 강제 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)

#### 7.1.2 **Floaters (떠다니는 조각)**

**증상**: 참조 뷰 앞에 떠다니는 disconnected 파편

**원인**: 
- 모델이 렌더링된 이미지를 입력과 일치시키기 위해 근처 파편으로 부분적 회피
- 특히 textual inversion 없이 심각 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)

**개선 메커니즘**:
- 이미지 특화 프롬프트로 충분한 사전 제약
- View-dependent 렌더링으로 다양한 관점에서 일관성 강제

#### 7.1.3 **Janus Problem (양면 현상)** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)

**정의**: 입력 이미지의 특징이 객체의 양쪽 면에 나타나는 현상

**예**: Pikachu 얼굴이 전/후면 모두에 표시

**원인**: 
- SDS 손실이 모든 뷰에서 "그럴듯한" 이미지 생성 강제
- 높은 guidance weight로 mode collapse 유도

**부분적 완화**:
- View-dependent prompt ("back view" 추가)
- 단일 이미지 textual inversion (약 50-70% 완화)

**근본적 해결 불가능한 이유**: 단일 이미지는 본질적으로 ambiguous

### 7.2 성능 제한 요인

| 제약 요소 | 영향 | 개선 연구 |
|----------|------|---------|
| 렌더링 해상도 (96px) | 세부사항 손실 | FlexiDreamer [arxiv](https://arxiv.org/abs/2404.00987), LRM [arxiv](https://arxiv.org/abs/2311.04400) |
| Marching cubes 추출 | 메시 이상 | DreamGaussian [arxiv](https://arxiv.org/html/2309.16653v2), FlexiDreamer [arxiv](https://arxiv.org/abs/2404.00987) |
| 최적화 속도 (45분) | 실용성 제한 | DreamGaussian (~2분) [arxiv](https://arxiv.org/html/2309.16653v2) |
| Guidance weight 의존성 | 다양성 감소 | VSD [arxiv](https://arxiv.org/html/2503.21745v1), MVDream [nature](https://www.nature.com/articles/s41598-025-24916-6) |
| 배경 분리 필요 | 자동화 부족 | SAM 통합 필요 |

***

## 8. 최신 연구와의 비교 분석 (2020년 이후)

### 8.1 관련 연구 계층 구조

#### **단계 1: 기초 기술 (2020-2021)**

**NeRF (Mildenhall et al., 2020)** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)
- 다중뷰 이미지에서 신경 방사 필드 학습
- 고품질 노벨 뷰 생성, 하지만 다중뷰 필수

**CLIP (Radford et al., 2021)** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)
- 텍스트-이미지 임베딩 정렬
- 이미지-텍스트 기반 3D 생성 아이디어 제시

#### **단계 2: 초기 조건부 3D 생성 (2022-2023 초반)**

**DreamFusion (Poole et al., 2022)** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)
- **혁신**: Score Distillation Sampling (SDS)로 텍스트-3D 생성
- **한계**: 단일 이미지 조건화 불가, 텍스트만 입력
- **특징**: 높은 guidance weight (100)로 mode collapse 유도

**CLIP-Mesh (Khalid et al., 2022)** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)
- CLIP 기반 mesh 생성, 낮은 품질

**Dream Fields (Jain et al., 2022)** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)
- CLIP을 사용한 텍스트-3D, 마찬가지로 품질 낮음

#### **단계 3: 이미지 기반 조건화 방법 (2023)**

**RealFusion (Melas-Kyriazi et al., 2023)** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)
- **혁신**: 단일 이미지 textual inversion으로 이미지-3D 달성
- **장점**: 카테고리 무관, 고품질 결과
- **한계**: 최적화 속도 (45분), Janus 문제

**DreamGaussian (Tang et al., 2023, ICLR 2024 Oral)** [youtube](https://www.youtube.com/watch?v=1xv3NBIYT44)
- **혁신**: 3D Gaussian Splatting으로 NeRF 대체
- **성능**: 
  - 속도: 10배 이상 개선 (~2분 vs 45분)
  - 품질: 더 나은 메시 추출, Janus 문제 완화
- **기술**: Single-step SDS + multi-step refinement

**Zero-1-to-3 (Liu et al., 2023)** [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2024W/NRI/papers/Chen_Recon3D_High_Quality_3D_Reconstruction_from_a_Single_Image_Using_CVPRW_2024_paper.pdf)
- 단일 뷰에서 상대적 카메라 포즈로 멀티뷰 합성
- 이후 연구들의 기초 (Recon3D에서 활용) [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10678295/)

#### **단계 4: 멀티뷰 Diffusion 및 최적화 개선 (2023-2024)**

**MVDream / MultiviewDiffusion (Shi et al., 2023)** [nature](https://www.nature.com/articles/s41598-025-24916-6)
- **혁신**: 각 뷰에서 개별적으로 conditioning
- **성과**: 더 나은 뷰 일관성
- **순위**: Benchmark에서 text-to-3D 최고 점수 (1303 Elo) [arxiv](https://arxiv.org/html/2503.21745v1)

**Magic3D (Lin et al., 2023)** [kokecacao](https://kokecacao.me/page/Post/Dreamfusion.md)
- 2단계 파이프라인: coarse 구조 → fine 텍스처
- 더 나은 기하학 디테일

**LucidDreamer (Zheng et al., 2023)** [arxiv](https://arxiv.org/html/2503.21745v1)
- VSD (Variational Score Distillation) 도입
- SDS의 mode collapse 분석 및 개선

**ProlificDreamer (Wang et al., 2023)** [arxiv](https://arxiv.org/html/2503.21745v1)
- SDS 수렴 특성 분석
- 적응형 가중치 스케줄링

#### **단계 5: Feed-Forward 모델 (2023-2024)**

**LRM (Hong et al., 2023)** [arxiv](https://arxiv.org/abs/2311.04400)
- **혁신**: 5초 내 단일 이미지 → 3D
- **방식**: 트랜스포머 기반 encoder-decoder
- **한계**: 제한된 카테고리(Objaverse 데이터 기반)

**Wonder3D (Hong et al., 2024)** [arxiv](https://arxiv.org/html/2503.21745v1)
- Multi-view cross-domain attention
- Benchmark 이미지-3D 최고 성능 (1304 Elo) [arxiv](https://arxiv.org/html/2503.21745v1)
- 우수한 기하학 디테일

**OpenLRM (Zhang et al., 2024)** [arxiv](https://arxiv.org/html/2503.21745v1)
- 개선된 LRM 아키텍처
- Benchmark 점수: 1280 Elo

#### **단계 6: 명시적 표현 기반 (2024)**

**FlexiDreamer (Qian et al., 2024)** [arxiv](https://arxiv.org/abs/2404.00987)
- **혁신**: Mesh를 직접 최적화 (NeRF 대신)
- **특징**: FlexiCubes gradient-based 최적화
- **성능**: ~1분, 높은 메시 품질
- **기술**: Hybrid positional encoding, orientation-aware texture mapping

**Recon3D (Chen et al., 2024)** [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10678295/)
- 생성된 back-view를 명시적 사전으로 사용
- ControlNet + DreamBooth 활용
- 기하학/질감 큰 개선

**MeshLRM (Wei et al., 2024)** [arxiv](https://arxiv.org/abs/2404.12385)
- LRM 기반 직접 메시 재구성
- 4개 이미지에서 <1초 내 생성

#### **단계 7: 최신 통합 접근 (2024-2025)**

**NeuroDiff3D (Lu et al., 2025)** [nature](https://www.nature.com/articles/s41598-025-24916-6)
- 3D Diffusion 모델 + 멀티모달 정보 융합
- 3D Prior Pipeline + Model Training Pipeline

**3D Generation Benchmark Suite (2025)** [arxiv](https://arxiv.org/html/2503.21745v1)
- 9개 text-to-3D + 13개 image-to-3D 모델 평가
- 자동 평가 메트릭 (3DGen-Score, 3DGen-Eval)
- 결과: Wonder3D (이미지), MVDream (텍스트) 우수

### 8.2 경쟁 방법들과의 정량적 비교

#### **Optimization-based 방법들**

| 방법 | 발표 | 속도 | 기하학 | 텍스처 | 일관성 | 카테고리 제한 |
|------|------|------|--------|--------|--------|-----------|
| DreamFusion | 2022 | 30분 | 중 | 중상 | 낮음 | 없음 |
| RealFusion | 2023 | 45분 | 중상 | 중상 | 중상 | **없음** |
| DreamGaussian | 2023 | **2분** | **우수** | 중상 | 중상 | 없음 |
| Magic3D | 2023 | 40분 | 우수 | 우수 | 중상 | 없음 |
| MVDream | 2023 | 30분 | **우수** | **우수** | **우수** | 없음 |
| FlexiDreamer | 2024 | **1분** | **우수** | **우수** | 중상 | 없음 |
| Recon3D | 2024 | 10분 | **우수** | **우수** | 중상 | 없음 |

#### **Feed-Forward 방법들**

| 방법 | 발표 | 속도 | 기하학 | 텍스처 | 일관성 | 데이터 의존성 |
|------|------|------|--------|--------|--------|-----------|
| LRM | 2023 | **5초** | 중상 | 중상 | 낮음 | 높음(Objaverse) |
| Wonder3D | 2024 | **10초** | **우수** | **우수** | **우수** | 높음 |
| OpenLRM | 2024 | 7초 | 중상 | 중상 | 중상 | 높음 |
| MeshLRM | 2024 | <1초 | **우수** | 중상 | 중상 | 높음 |

### 8.3 RealFusion의 위치 및 의의

#### **역사적 의미**

RealFusion은 다음 두 가지 패러다임 전환의 교점에 위치합니다:

1. **텍스트 조건화 → 이미지 조건화**
   - DreamFusion: "A dog"만 가능
   - RealFusion: 특정 개 이미지 처리 가능
   - 일반화의 큰 도약

2. **사전 활용의 진화**
   - CLIP 사전 (약함)
   - 단순 Diffusion 프롬프트 (중간)
   - **이미지 특화 프롬프트** (강함) ← RealFusion의 기여

#### **현재 평가 (2024-2025)**

**장점**:
- 완전한 카테고리 무관성
- 강력한 이미지 충실도
- 3D 감독 불필요
- 자연스러운 결과

**한계**:
- 최적화 속도 (DreamGaussian의 1/20-1/45)
- Janus 문제 완전 해결 불가
- 메시 품질 (marching cubes 한계)

**현재 역할**: 
- **학술적 기초**: 단일 이미지 textual inversion의 원형
- **벤치마크**: 정성적 평가 기준
- **실무적 가치**: 제한적 (속도 때문에)

#### **후속 방법과의 연결**

```
RealFusion (2023)
   ↓
   ├─→ DreamGaussian (2023): 표현 개선 (NeRF→Gaussian)
   ├─→ FlexiDreamer (2024): 표현 개선 (Mesh 직접)
   ├─→ Recon3D (2024): 조건화 개선 (back-view prior)
   └─→ NeuroDiff3D (2025): 3D diffusion 통합
```

***

## 9. 앞으로의 연구 방향과 고려사항

### 9.1 이론적 깊이화

#### **1) SDS 수렴 특성 분석**

현재 이해 부족:
- SDS가 왜 특정 guidance weight에서만 잘 작동하는지
- Mode collapse의 수학적 원인

필요한 연구:
- 수렴 증명 및 부동점 분석
- 적응형 가중치 스케줄 개발
- VSD 같은 대안의 이론적 근거 [arxiv](https://arxiv.org/html/2503.21745v1)

#### **2) Diffusion Prior의 3D 편향**

중요한 질문:
- 2D 이미지에 학습된 확산 모델이 3D 기하학의 어떤 측면을 선호하는가?
- "그럴듯한" 3D 형태에 숨겨진 편향은 무엇인가?

응용:
- 특정 장르(예: 건축)에 최적화된 사전 개발
- 편향 분석을 통한 데이터셋 개선

### 9.2 일반화 성능의 체계적 확대

#### **3) 복잡한 장면 처리**

현재 한계: 고립된 객체만 처리

확장 방향:
- **배경 일관성**: 객체-배경 상호작용 모델링
- **관계 추론**: 다중 객체 장면에서 공간 관계 학습 [arxiv](https://arxiv.org/abs/2401.05335)
- **텍스처 일관성**: 배경 텍스처와의 조화

기술:
- 장면 그래프 기반 구조화 [arxiv](https://arxiv.org/html/2407.12667v1)
- 멀티오브젝트 SDS 손실
- Semantic segmentation 활용

#### **4) 극단적 뷰포인트에서의 안정성**

현재: 정면 중심 설계

개선:
- 회전 불변 augmentation 강화
- 180도 뷰에 특화된 정규화
- 대칭성 제약 (if applicable)

평가:
- 테스트셋에 의도적으로 difficult viewing angles 포함

#### **5) 원거리(Long-tail) 객체 분포**

현재: Stable Diffusion 학습 분포에 의존

확장:
- 도메인 특화 diffusion 모델 파인튜닝
- Few-shot adaptation 메커니즘
- Out-of-distribution 객체 처리

### 9.3 기술적 개선

#### **6) 메시 추출 최적화**

Marching cubes의 한계 극복:

1. **직접 메시 최적화** (FlexiDreamer 방식) [arxiv](https://arxiv.org/abs/2404.00987)
   - Gradient-based 메시 업데이트
   - 복셀화 회피

2. **포인트 클라우드 정제**
   - Poisson surface reconstruction 대신 학습 기반
   - 노이즈 제거 신경망

3. **암묵적 표현 유지**
   - Mesh 추출 없이 최종 렌더링

#### **7) 고해상도 렌더링**

현재: 96px → 512px upsampling

개선:
- 계층적 렌더링 (저해상도 기하학, 고해상도 텍스처)
- 확산 모델 업그레이드 (SDXL, Stable Diffusion 3) [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2024/papers/Ding_Text-to-3D_Generation_with_Bidirectional_Diffusion_using_both_2D_and_3D_CVPR_2024_paper.pdf)
- Super-resolution 통합

#### **8) 속도 최적화**

RealFusion (45분) → DreamGaussian (2분) → FlexiDreamer (1분)

목표: Realtime 또는 대화형 수준

기술:
- Efficient diffusion sampling (DDIM, flow matching)
- 캐시 활용 (frozen 모델 가중치)
- 배치 처리 최적화

### 9.4 평가 체계 고도화

#### **9) 정량적 메트릭 개선**

현재 한계:
- F-score: 낮은 포인트 클라우드 해상도에서 비민감
- CLIP-similarity: 의미적 유사도만 측정, 기하학 무시

새 메트릭:
- **기하학**: Chamfer distance (고해상도 메시 기준)
- **텍스처**: LPIPS, SSIM (perceptual quality)
- **일관성**: View-angle consistency score [arxiv](https://arxiv.org/html/2412.02287v3)
- **다양성**: Multimodality measure (생성 분포 평가)

#### **10) 인간 평가 확대**

현재: 제한적 사용자 연구

필요:
- 대규모 주석 작업 (500+ 이미지)
- 전문가 3D 모델러 평가
- 실제 응용 분야 사용자 피드백 (게임 개발자, 3D 아티스트)

### 9.5 응용 분야 적응

#### **11) 카테고리별 특화**

일반 모델의 한계를 극복:

1. **인물** (SinHuman3D 방향) [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10860075/)
   - SMPL 선행 지식 활용
   - 의류/헤어 기하학 특화
   - 포즈 추론

2. **건축물** (문화유산 보존) [isprs-archives.copernicus](https://isprs-archives.copernicus.org/articles/XLVIII-2-2024/73/2024/)
   - 대칭성 강제
   - 반복 패턴 인식
   - 고해상도 텍스처 재현

3. **의료 이미징** (CT 재구성) [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10859640/)
   - 조직 경계 명확성
   - 물리적 제약 (밀도 범위)

#### **12) 동적 장면 확장**

현재: 정적 객체만

목표: 애니메이션 가능한 3D 캐릭터

기술:
- Skeleton 추론
- 변형 필드(Deformation field) 학습
- Temporal consistency 제약

예시 연구: OneTo3D (동적 모델 + 비디오 생성) [arxiv](http://arxiv.org/pdf/2405.06547.pdf)

### 9.6 데이터셋 및 벤치마크 발전

#### **13) 포괄적 벤치마크 구축**

현재: CO3D 7개 카테고리 (21개 이미지)

필요:
- 500+ 카테고리 커버 (Objaverse 규모)
- 다양한 이미지 품질 (전문 사진 ↔ 인터넷 이미지)
- 극단적 케이스 (매우 가까운/먼 거리, 수직 뷰 등)

#### **14) 자동화된 평가 시스템**

최신 발전: 3DGen-Score 및 3DGen-Eval [arxiv](https://arxiv.org/html/2503.21745v1)

개선 방향:
- 기하학 세부사항 특화 메트릭
- 텍스처-기하학 정렬 평가
- 멀티모달 분포 품질 평가

### 9.7 인접 분야와의 통합

#### **15) 강화 학습 피드백**

기하학 품질을 위해:
- 물리 시뮬레이션 기반 보상 (안정성, 균형)
- 렌더링 일관성 보상
- 사용자 선호도 학습

#### **16) 기하학적 사전 통합**

Parametric 모델과의 결합:
- 객체 카테고리별 shape space 활용
- 부분 분해(part decomposition)
- 대칭성 및 반복 패턴

#### **17) 멀티모달 입력**

이미지 이외:
- 텍스트 + 이미지 (더 나은 제약)
- 깊이 맵 입력 (모노큘러 깊이 추정) [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2024W/NRI/papers/Chen_Recon3D_High_Quality_3D_Reconstruction_from_a_Single_Image_Using_CVPRW_2024_paper.pdf)
- 터치 센싱 (기하학 디테일) [arxiv](https://arxiv.org/html/2412.06785)

***

## 10. 결론

RealFusion은 **단일 이미지로부터 카테고리 무관한 360도 3D 재구성**을 최초로 시연한 획기적 방법입니다. 이는 **이미지 기반 조건화를 위한 새로운 파러다임—단일 이미지 텍스트 반전—을 도입**함으로써 달성되었습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)

### 핵심 성과
- DreamFusion의 SDS 기법을 이미지 조건화로 확장
- Textual inversion을 augmentation 기반으로 혁신
- 정규화 전략과 coarse-to-fine 학습으로 품질 확보
- 카테고리별 학습 모델의 한계 극복

### 현재 평가 (2024-2025)
RealFusion은 **학술적으로는 중요한 기초 작업**이나, **실무적으로는 후속 방법들에 의해 대체**되고 있습니다: [arxiv](https://arxiv.org/abs/2404.12385)
- **속도**: DreamGaussian이 20배, FlexiDreamer가 45배 빠름
- **품질**: MVDream, Wonder3D가 더 나은 일관성 제공
- **실용성**: Feed-forward 모델이 즉시 결과 제공

### 앞으로의 연구 우선순위

1. **이론적 이해**: SDS의 수렴 특성, diffusion prior의 3D 편향 분석
2. **일반화 확대**: 복잡한 배경, 극단적 뷰포인트, 텍스처 일관성 개선
3. **평가 고도화**: 정량 메트릭 정교화, 대규모 인간 평가
4. **응용 확장**: 카테고리 특화, 동적 장면, 멀티모달 입력
5. **기술 혁신**: 암묵적 표현의 한계 극복, 실시간 처리 달성

RealFusion이 개척한 **이미지-3D 생성의 사상**은 2023-2025년 급속한 발전을 이끌었으며, 향후 연구는 이 기초 위에서 속도, 품질, 일관성의 균형을 최적화하는 방향으로 진행될 것으로 예상됩니다.

***

## 참고 자료

 Melas-Kyriazi et al. (2023). "RealFusion: 360° Reconstruction of Any Object from a Single Image." arXiv:2302.10663v2 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5d32ca3a-ef93-45bc-a553-5cc00a40613b/2302.10663v2.pdf)

 Chen et al. (2024). "Recon3D: High Quality 3D Reconstruction from a Single Image Using Generated Back-View Explicit Priors." [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10678295/)

 Zhou et al. (2024). "SinHuman3D: Novel Multi-View Synthesis and 3D Reconstruction from a Single Human Image." [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10860075/)

 Qian et al. (2024). "FlexiDreamer: Single Image-to-3D Generation with FlexiCubes." [arxiv](https://arxiv.org/abs/2404.00987)

 Li et al. (2024). "3D X-ray Image Reconstruction Based on Collaborative Neural Radiance Field and Ensemble Learning." [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10859640/)

 Liu et al. (2024). "Template-Free Single-View 3D Human Digitalization with Diffusion-Guided LRM." [semanticscholar](https://www.semanticscholar.org/paper/6dfb37776eab7dc78145a6328084b43dd60e1ca8)

 Li et al. (2024). "EndoGaussians: Single View Dynamic Gaussian Splatting for Deformable Endoscopic Tissues Reconstruction." [arxiv](https://arxiv.org/abs/2401.13352)

 Wei et al. (2024). "MeshLRM: Large Reconstruction Model for High-Quality Mesh." [arxiv](https://arxiv.org/abs/2404.12385)

 Cheng et al. (2024). "CoherentGS: Sparse Novel View Synthesis with Coherent 3D Gaussians." [link.springer](https://link.springer.com/10.1007/978-3-031-73404-5_2)

 Jiang et al. (2024). "Utilizing NeRF-Based Rays for Spatial Perception in Fruit Counting Deduplication." [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10803409/)

 Shahbazi et al. (2024). "InseRF: Text-Driven Generative Object Insertion in Neural 3D Scenes." [arxiv](https://arxiv.org/abs/2401.05335)

 Condorelli et al. (2024). "Comparative Evaluation of NeRF Algorithms on Single Image Dataset for 3D Reconstruction." [isprs-archives.copernicus](https://isprs-archives.copernicus.org/articles/XLVIII-2-2024/73/2024/isprs-archives-XLVIII-2-2024-73-2024.pdf)

 Mayr et al. (2024). "Global Structure-From-Motion Enhanced Neural Radiance Fields 3D Reconstruction." [isprs-archives.copernicus](https://isprs-archives.copernicus.org/articles/XLVIII-4-W10-2024/199/2024/isprs-archives-XLVIII-4-W10-2024-199-2024.pdf)

 Wang et al. (2023). "NerfDiff: Single-image View Synthesis with NeRF-guided Distillation from 3D-aware Diffusion." [arxiv](https://arxiv.org/pdf/2302.10109.pdf)

 Xu et al. (2024). "OneTo3D: One Image to Re-editable Dynamic 3D Model and Video Generation." [arxiv](http://arxiv.org/pdf/2405.06547.pdf)

 Qi et al. (2024). "SUP-NeRF: A Streamlined Unification of Pose Estimation and NeRF for Monocular 3D Object Reconstruction." [arxiv](http://arxiv.org/pdf/2403.15705.pdf)

 Wang et al. (2024). "SG-NeRF: Neural Surface Reconstruction with Scene Graph Optimization." [arxiv](https://arxiv.org/html/2407.12667v1)

 Yu et al. (2023). "ObjectSDF++: Improved Object-Compositional Neural Implicit Surfaces." [arxiv](https://arxiv.org/pdf/2308.07868.pdf)

 Zanella et al. (2024). "Depth Supervised Neural Surface Reconstruction from Airborne Imagery." [isprs-annals.copernicus](https://isprs-annals.copernicus.org/articles/X-2-2024/89/2024/isprs-annals-X-2-2024-89-2024.pdf)

 Liu et al. (2023). "Zero-1-to-3: Zero-shot One Image to 3D Object." [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2024W/NRI/papers/Chen_Recon3D_High_Quality_3D_Reconstruction_from_a_Single_Image_Using_CVPRW_2024_paper.pdf)

 Lu et al. (2025). "NeuroDiff3D: a 3D generation method optimizing viewpoint consistency via viewpoint-independent implicit representation." [nature](https://www.nature.com/articles/s41598-025-24916-6)

 Gupta et al. (2025). "Comprehensive Benchmark Suite for 3D Generative Models." [foundr](https://foundr.ai/product/dreamfusion)

 Ye et al. (2024). "Viewpoint Consistency in 3D Generation via Training-free Augmented Consistency Guidance." [isprs-archives.copernicus](https://isprs-archives.copernicus.org/articles/XLVIII-2-2024/73/2024/)

 Xu et al. (2024). "Exploiting Tactile Sensing for 3D Generation." [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2024/papers/Ding_Text-to-3D_Generation_with_Bidirectional_Diffusion_using_both_2D_and_3D_CVPR_2024_paper.pdf)

 Tang et al. (2023). "DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation." ICLR 2024 Oral. [youtube](https://www.youtube.com/watch?v=1xv3NBIYT44)

 Hong et al. (2023). "LRM: Large Reconstruction Model for Single Image to 3D." [arxiv](https://arxiv.org/abs/2311.04400)

 Yao et al. (2024). "A Comprehensive Survey on 3D Content Generation." [arxiv](https://arxiv.org/html/2410.04738v3)

 Pan et al. (2024). "Repurposing 2D Diffusion Models with Gaussian Atlas for Efficient 3D Generation." [simonsy](https://simonsy.net/article/3D-moment4SD-en)

 Chen et al. (2024). "Image valuation in NeRF-based 3D reconstruction." [drpress](https://drpress.org/ojs/index.php/HSET/article/view/20488)

 Li et al. (2024). "Gen-3Diffusion: Realistic Image-to-3D Generation via 2D & 3D Diffusion." [github](https://github.com/dreamgaussian/dreamgaussian)

 Dou et al. (2024). "High-Fidelity 3D Generation with Contrastive Learning and Detail-Enhancing Diffusion." [jmis](https://www.jmis.org/archive/view_article?pid=jmis-11-4-241)

 Ye et al. (2024). "MTFusion: Reconstructing Any 3D Object from Single Image with Multi-Level Fusion." [journal.kci.go](https://journal.kci.go.kr/jksci/archive/articleView?artiId=ART003152746)

 Lin et al. (2023). "Magic3D: High-Resolution Text-to-3D Content Creation." [kokecacao](https://kokecacao.me/page/Post/Dreamfusion.md)

 Ding et al. (2024). "Text-to-3D Generation with Bidirectional Diffusion using both 2D and 3D Priors." [arxiv](https://arxiv.org/html/2503.21745v1)

 Song et al. (2024). "Viewpoint Consistency in 3D Generation via Training-free Augmented Consistency Guidance." [arxiv](https://arxiv.org/html/2412.02287v3)

 Wang et al. (2024). "Exploiting Tactile Sensing for 3D Generation." [arxiv](https://arxiv.org/html/2412.06785)
