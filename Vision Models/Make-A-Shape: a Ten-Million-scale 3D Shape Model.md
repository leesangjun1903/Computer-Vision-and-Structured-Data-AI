
# Make-A-Shape: a Ten-Million-scale 3D Shape Model

> **논문 정보**
> - **제목:** Make-A-Shape: a Ten-Million-scale 3D Shape Model
> - **저자:** Ka-Hei Hui, Aditya Sanghi, Arianna Rampini, Kamal Rahimi Malekshan, Zhengzhe Liu, Hooman Shayani, Chi-Wing Fu
> - **학회:** ICML 2024 (Proceedings of the 41st International Conference on Machine Learning), pp. 20660–20681
> - **arXiv:** [2401.11067](https://arxiv.org/abs/2401.11067)
> - **GitHub:** [AutodeskAILab/Make-a-Shape](https://github.com/AutodeskAILab/Make-a-Shape)
> - **기관:** Autodesk AI Lab, CUHK

---

## 1. 핵심 주장과 주요 기여 요약

### 1.1 핵심 주장

자연어 및 이미지 분야의 대규모 생성 모델 훈련은 상당한 진전을 이뤘으나, 3D 생성 모델의 발전은 막대한 학습 자원 요구, 비효율적이고 비압축적이며 표현력이 낮은 표현 방식으로 인해 저해되어 왔다.

이 논문은 1,000만 개의 공개 3D 형상을 활용할 수 있는 대규모 효율적 학습을 위한 새로운 3D 생성 모델 **Make-A-Shape**를 소개한다.

### 1.2 주요 기여 (4가지 핵심 요소)

| 기여 | 설명 |
|------|------|
| ① Wavelet-Tree Representation | 3D 형상의 컴팩트·고품질 인코딩 |
| ② Subband Coefficient Filtering | 정보가 풍부한 계수만 선택적으로 보존 |
| ③ Subband Coefficients Packing | Diffusion 모델로 생성 가능한 구조로 변환 |
| ④ Subband Adaptive Training Strategy | 거친 형상~세부 구조까지 균형 있게 학습 |

또한 단일/다중뷰 이미지, 포인트 클라우드, 저해상도 복셀 등 다양한 입력 모달리티로 조건부 생성을 지원하며, 무조건 생성, 형상 완성, 조건부 생성 등 다양한 애플리케이션을 시연한다.

이 접근법은 최신 기법들을 능가하는 고품질 결과를 제공할 뿐 아니라, 대부분의 조건에서 **단 2초** 내에 형상을 효율적으로 생성한다.

---

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

3D 모델링의 경우 추가적인 공간 차원이 신경망이 모델링해야 하는 입력 변수의 수를 크게 증가시켜 훨씬 더 많은 네트워크 파라미터를 필요로 한다. 이는 U-Net 기반 Diffusion 모델에서 특히 두드러지는데, GPU가 처리하기엔 너무 큰 메모리 집약적인 feature map을 생성해 학습 시간을 연장시킨다.

3D 형상을 표현하는 방법은 여러 가지가 있으며, 높은 표현 품질과 효율적인 학습을 위한 좋은 압축성을 동시에 달성하는 최적의 방법이 무엇인지는 여전히 불분명하다.

세 가지 핵심 문제:
1. **메모리 비효율성:** 3D U-Net의 feature map이 GPU 처리 한계를 초과
2. **데이터 처리 복잡성:** 10M 규모의 3D 데이터 스트리밍/저장 비용
3. **표현 품질-압축률 트레이드오프:** 기존 표현법(voxel, point cloud, mesh, NeRF, SDF)의 한계

---

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: TSDF 인코딩 및 웨이블릿 분해

형상은 먼저 **Truncated Signed Distance Field (TSDF)**로 인코딩된 후, 웨이블릿 트리 구조의 다중 스케일 웨이블릿 계수로 분해된다.

3D TSDF 그리드 $\mathbf{F} \in \mathbb{R}^{N \times N \times N}$에 대해 3D 웨이블릿 분해를 수행하면:

$$\mathbf{F} = \mathcal{W}^{-1}\left(\{C_{\text{LL}}\} \cup \{D^{(l)}_{k}\}_{l,k}\right)$$

- $C_{\text{LL}}$: 거친(Coarse) 저주파 서브밴드 계수
- $D^{(l)}_{k}$: 레벨 $l$, 방향 $k$의 세부(Detail) 고주파 서브밴드 계수
- $\mathcal{W}^{-1}$: 역 웨이블릿 변환

#### Step 2: Subband Coefficient Filtering

계수들 간의 관계를 활용하는 서브밴드 계수 필터링 절차를 설계하여 웨이블릿 트리 표현에서 정보가 풍부한 웨이블릿 성분(거친 계수와 세부 계수 모두)을 보존함으로써, 효율적인 저장 및 스트리밍을 위한 충실하고 컴팩트한 3D 형상 표현을 가능하게 한다.

부모-자식 계수 관계를 이용한 필터링:

$$\text{Mask}^{(l)}_{k}(i,j,m) = \mathbf{1}\left[|D^{(l-1)}_{\text{parent}(i,j,m)}| > \tau\right]$$

즉, 부모 계수의 크기가 임계값 $\tau$를 초과하는 위치의 자식 계수만 보존한다.

#### Step 3: Subband Coefficients Packing (Diffusible 표현)

서브밴드 계수 패킹 방식을 제안하여 웨이블릿 트리 표현을 관리 가능한 공간 해상도의 규칙적인 그리드 구조로 재배열함으로써, Denoising Diffusion 모델이 표현을 효과적으로 생성할 수 있게 한다.

다중 스케일 계수를 단일 저해상도 그리드 $\mathbf{G} \in \mathbb{R}^{H \times W \times D \times C}$로 패킹:

$$\mathbf{G} = \text{Pack}\left(C_{\text{LL}},\; D^{(1)}_{1:7},\; D^{(2)}_{1:7},\; \ldots\right)$$

이를 통해 일반 3D U-Net 기반 Diffusion 모델로 생성 가능한 구조가 형성된다.

#### Step 4: Subband Adaptive Training Strategy

형상 정보는 서브밴드와 스케일에 따라 다양하게 분포하며, 세부 계수는 매우 희소하지만 형상 세부 정보를 풍부하게 포함한다. 표준 균일 MSE 손실로 학습하면 모델 붕괴나 세부 정보의 비효율적 학습으로 이어질 수 있다. 이를 해결하기 위해 서브밴드 적응형 학습 전략을 도입하여 다양한 서브밴드의 계수에 선택적으로 집중하며, 이 접근법은 학습 과정에서 거친 서브밴드부터 미세한 서브밴드까지 형상 정보의 효과적인 균형을 허용하고 모델이 형상의 구조적·세부적 측면을 모두 학습하도록 장려한다.

학습 손실:

$$\mathcal{L} = \sum_{l=0}^{L} \lambda_l \cdot \mathbb{E}_t\left[\left\|\epsilon^{(l)} - \hat{\epsilon}^{(l)}_\theta(\mathbf{G}_t, t)\right\|^2\right]$$

- $\lambda_l$: 서브밴드 레벨 $l$에 대한 적응형 가중치 (희소한 세부 계수에 더 큰 가중치)
- $\mathbf{G}_t$: 시간 $t$에서 노이즈가 추가된 패킹 그리드
- $\hat{\epsilon}^{(l)}_\theta$: 레벨 $l$의 노이즈 예측값

#### Step 5: DDPM 기반 생성 과정

표준 DDPM(Denoising Diffusion Probabilistic Model) 역방향 과정:

$$p_\theta(\mathbf{G}_{t-1}|\mathbf{G}_t) = \mathcal{N}\left(\mathbf{G}_{t-1};\; \mu_\theta(\mathbf{G}_t, t),\; \Sigma_\theta(\mathbf{G}_t, t)\right)$$

생성 후 역 웨이블릿 변환과 Marching Cubes로 최종 3D 메시를 추출한다.

---

### 2.3 모델 구조

Make-A-Shape는 웨이블릿 트리 표현과 서브밴드 적응형 학습 전략을 사용하여 3D 형상의 거친 세부 정보와 미세한 세부 정보를 효과적으로 캡처하며, 48개의 A10G GPU에서 학습되었다.

```
입력 3D 형상
    │
    ▼
[TSDF 변환] → 고해상도 SDF 그리드
    │
    ▼
[3D Wavelet Decomposition]
    │
    ├─ Coarse Subband (C_LL)
    └─ Detail Subbands (D^(l)_k, l=1..L, k=1..7)
    │
    ▼
[Subband Coefficient Filtering]
    │ (정보-풍부 계수만 선택)
    ▼
[Subband Coefficients Packing]
    │ (저해상도 3D 그리드로 재배열)
    ▼
[3D U-Net 기반 Diffusion Model]
    │ (서브밴드 적응형 손실로 학습)
    ▼
[역 웨이블릿 변환 + Marching Cubes]
    │
    ▼
출력: 3D Mesh
```

**조건부 생성 확장:**
단순 무조건 생성을 넘어서, 단일/다중뷰 이미지, 복셀, 포인트 클라우드 등의 조건을 지원하는 조건부 생성으로 방법을 확장한다.

- **단일뷰 이미지 → 3D:** CNN/ViT 인코더로 이미지 특징 추출 후 cross-attention
- **다중뷰 이미지 → 3D:** 다중 이미지 특징 집계 후 조건부 생성
- **포인트 클라우드 → 3D:** PointNet 계열 인코더 활용
- **저해상도 복셀 → 3D:** 3D CNN 인코더 활용 (형상 완성)

---

### 2.4 성능 향상

이미지-to-3D 생성 작업에 대한 시각적 비교에서 Make-A-Shape는 Point-E, Shap-E, One-2-3-45 등 세 가지 주요 생성 모델을 능가한다. 단일뷰 모델은 이러한 기준 모델들에 비해 더 정확한 형상을 생성하고, 다중뷰 모델은 추가적인 뷰 정보로 형상 충실도를 더욱 향상시킨다.

이 프레임워크는 덜 강력한 GPU(A10G vs. A100)를 사용함에도 불구하고 이러한 방법들보다 하루에 평균 **2배~6배** 더 많은 학습 형상을 처리한다.

논문에서 사용한 검증 데이터셋은 **19개의 서로 다른 오픈소스 3D 형상 데이터셋**에서 선택된 객체들로 구성된다.

주요 정량적 지표 비교 (논문 Table 기준):

| 모델 | 데이터 규모 | 생성 시간 | 품질 |
|------|------------|----------|------|
| Point-E | ~수백만(비공개) | ~수 분 | 낮음 |
| Shap-E | ~수백만(비공개) | ~수 분 | 보통 |
| One-2-3-45 | ShapeNet | ~45초 | 보통 |
| **Make-A-Shape** | **10M (공개)** | **~2초** | **높음** |

---

### 2.5 한계

Google Scanned Objects(GSO) 데이터셋을 추가 평가 세트로 활용하여 방법의 **크로스 도메인 일반화 능력**을 평가한다(학습 데이터에 포함되지 않은 도메인).

논문에서 인정하거나 추론 가능한 한계:

1. **텍스처 미지원:** SDF 기반 표현으로 인해 **기하학(geometry)만 생성**하며, 색상/텍스처는 별도 처리 필요
2. **고해상도 한계:** Diffusion 모델의 저해상도 그리드 패킹 단계에서 초고해상도 세부 정보 일부 손실 가능
3. **폐쇄적 표면 전제:** TSDF 표현은 닫힌(watertight) 메시 가정에 의존하여 얇은 구조나 열린 표면 처리가 제한적
4. **텍스트 조건 미지원:** 이미지·포인트 클라우드·복셀만 조건으로 지원하며, 텍스트 기반 생성은 미지원
5. **도메인 갭:** 학습 데이터의 다양성에도 불구하고 완전히 새로운 도메인(예: 실제 스캔 객체)에 대한 일반화 성능은 추가 검증 필요

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 위한 핵심 설계 요소

#### (1) 대규모 다양한 데이터 학습

Make-A-Shape는 1,000만 개 이상의 다양한 3D 형상으로 학습된 대규모 3D 생성 모델이다. 이를 통해 광범위한 객체 카테고리에 걸쳐 복잡한 기하학적 세부 정보, 그럴듯한 구조, 비자명한 위상(topology), 깨끗한 표면을 특징으로 하는 다양한 3D 형상을 무조건 생성하는 능력을 보인다.

검증 데이터셋은 **19개의 서로 다른 오픈소스 3D 형상 데이터셋**으로부터 선택된 객체들로 구성되어 있어, 다양한 도메인에서의 일반화를 평가한다.

#### (2) 크로스 도메인 일반화 평가 (GSO 데이터셋)

저자들은 Google Scanned Objects(GSO) 데이터셋을 추가 평가 세트로 활용하여 **학습에 포함되지 않은** 도메인에 대한 크로스 도메인 일반화 능력을 평가한다.

이는 학습 데이터와 다른 실제 스캔 객체들에 대해 모델이 얼마나 잘 일반화되는지를 측정하는 중요한 지표다.

#### (3) 다중 모달리티 지원

이 생성 프레임워크는 이미지, 포인트 클라우드, 복셀 등 다양한 입력 모달리티를 조건으로 처리할 수 있어, 무조건 생성, 완성, 조건부 생성 등 다양한 다운스트림 애플리케이션을 가능하게 한다.

#### (4) 웨이블릿 트리 표현의 다중 스케일 특성

웨이블릿 트리 표현은 고해상도 SDF 그리드에 웨이블릿 분해를 적용하여 거친 계수 서브밴드와 다중 스케일 세부 계수 서브밴드를 생성한다.

다중 스케일 표현은 형상의 글로벌 구조(저주파)와 지역 세부 정보(고주파)를 분리하여 학습하게 하므로, 새로운 형상에 대한 일반화에 유리하다.

### 3.2 일반화 성능 향상을 위한 수식적 고찰

서브밴드 적응형 손실 함수의 관점에서:

$$\mathcal{L}_{\text{adaptive}} = \underbrace{\lambda_0 \cdot \mathcal{L}_{\text{coarse}}}_{\text{전역 구조 학습}} + \underbrace{\sum_{l=1}^{L} \lambda_l \cdot \mathcal{L}^{(l)}_{\text{detail}}}_{\text{세부 정보 학습}}$$

여기서:
- 거친(Coarse) 손실: 새로운 카테고리에도 전이 가능한 글로벌 형상 prior 학습
- 세부(Detail) 손실: 희소하지만 고유한 형상 특성 학습 → 세부 일반화

이 분리 학습 구조는 새로운 도메인에서 **coarse 구조를 빠르게 적응**시키고 **detail을 정교화**하는 두 단계 fine-tuning 전략으로 확장 가능하다.

### 3.3 일반화 성능의 한계 및 향후 개선 방향

| 현재 한계 | 향후 개선 방향 |
|---------|--------------|
| 텍스처 미지원 | Color/texture subband 추가 |
| 텍스트 조건 미지원 | CLIP/LLM과의 cross-modal alignment |
| 실제 스캔 객체 도메인 갭 | Domain adaptation / fine-tuning |
| 닫힌 메시 전제 | Open surface SDF 확장 |

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 주요 관련 연구 비교표

| 논문 | 연도 | 표현 방식 | 생성 방법 | 규모 | 특징 |
|------|------|----------|----------|------|------|
| **DDPM** (Ho et al.) | 2020 | 2D 이미지 | Diffusion | - | Diffusion 기반 생성의 기반 |
| **Point-E** (Nichol et al.) | 2022 | Point Cloud | Diffusion | 수백만(비공개) | 텍스트→3D, 빠른 생성 |
| **Shap-E** (Jun & Nichol) | 2022 | Implicit (SDF+NeRF) | Diffusion | 수백만(비공개) | 텍스트/이미지→3D |
| **LION** (Zeng et al.) | 2022 | Point Cloud (계층적) | Hierarchical DDM | ShapeNet | 다중 모달 생성 |
| **LION**은 글로벌 형상 잠재 표현과 포인트 구조 잠재 공간을 결합한 계층적 잠재 공간을 가진 VAE로 설정되며, 생성을 위해 이 잠재 공간에서 두 개의 계층적 DDM을 학습한다. | | | | | |
| **3DShape2VecSet** (Zhang et al.) | 2023 | Neural Field (벡터 집합) | Diffusion | ShapeNet | Transformer 친화적 표현 |
| **3DShape2VecSet**은 생성적 Diffusion 모델을 위해 설계된 신경 필드를 위한 새로운 형상 표현을 도입하며, 표면 모델 또는 포인트 클라우드로 주어진 3D 형상을 인코딩하고 신경 필드로 표현한다. | | | | | |
| **OctFusion** (Xiong et al.) | 2024 | Octree 기반 | Diffusion (다중 스케일 U-Net) | ShapeNet, Objaverse | 임의 해상도 생성 |
| **OctFusion**은 단일 Nvidia 4090 GPU에서 2.5초 만에 임의 해상도의 3D 형상을 생성할 수 있으며, 핵심 구성 요소는 octree 기반 잠재 표현과 그에 맞는 diffusion 모델이다. 이 표현은 암묵적 신경 표현과 명시적 공간 octree의 장점을 결합하며 octree 기반 변분 오토인코더로 학습된다. |
| **Make-A-Shape (본 논문)** | 2024 | Wavelet-Tree SDF | Diffusion (3D U-Net) | **10M (공개)** | 다중 모달, 2초 생성 |
| **3D-WAG** (Bourached et al.) | 2024 | Wavelet + VQ-VAE | Autoregressive | - | Make-A-Shape 표현 계승, AR 생성 |
| 최근 방법들은 웨이블릿 압축 표현을 적용하여 Diffusion 모델로 고해상도 3D 형상을 생성한다. 기존 방법에서 발전하여, 이 접근법은 자동회귀 프레임워크에서 고해상도 SDF 생성을 위한 컴팩트 웨이블릿 기반 공간 주파수 표현을 채택한다. |

### 4.2 표현 방식별 특성 비교

```
           표현력 (High)
               │
  NeRF/Neural  │  ●Shap-E      ●Make-A-Shape
  Fields       │              (Wavelet-Tree SDF)
               │
  Point Cloud  │  ●LION
               │  ●Point-E
  Voxel        │
               │
               └─────────────────────────────
            압축성 (High) →      (Low)
```

---

## 5. 향후 연구에 미치는 영향 및 고려할 점

### 5.1 향후 연구에 미치는 영향

#### (1) 대규모 3D Foundation Model의 가능성 제시

이 프레임워크는 광범위한 객체 카테고리에 걸쳐 **1,000만 개의 3D 형상**으로 구성된 방대한 데이터셋에서 무조건 생성 모델과 다양한 입력 조건 하의 확장 모델 학습을 용이하게 한다.

이는 2D 이미지의 ImageNet 규모 학습이 Vision Foundation Model로 이어진 것처럼, 3D 도메인에서 Foundation Model 패러다임을 여는 중요한 이정표다.

#### (2) 웨이블릿 기반 3D 표현의 확산

3D-WAG와 같은 후속 연구들은 다중 스케일 웨이블릿 변환으로 형상 데이터를 인코딩하여 저주파와 고주파 정보를 모두 캡처하고, 가장 정보가 풍부한 웨이블릿 세부 계수를 선택적으로 식별 및 보존하며, 이 표현이 더 많은 형상 세부 정보를 컴팩트하게 포함할 수 있도록 한다.

Make-A-Shape의 웨이블릿 트리 표현은 이미 후속 연구들에 채택되어 표준 3D 표현으로 자리잡고 있다.

#### (3) 다중 모달리티 3D 생성의 표준화

이 생성 프레임워크는 이미지, 포인트 클라우드, 복셀 등 다양한 입력 모달리티에 조건을 걸 수 있어, 무조건 생성, 완성, 조건부 생성 등 다양한 다운스트림 애플리케이션을 지원하는 다재다능한 특성을 보인다.

#### (4) 생성 속도 기준 향상

전반적으로 이 생성 모델은 효과적으로 학습될 수 있으며, 빠른 추론도 가능하고 기존 방법들에 비해 고품질 형상을 단 몇 초 만에 생성할 수 있다.

이전 방법들이 수 분~수십 분 걸리던 것과 비교하여 **2초** 생성이라는 새로운 실용적 기준을 제시한다.

---

### 5.2 향후 연구 시 고려할 점

#### ① 텍스처 및 외관 통합
현재 Make-A-Shape는 기하학(geometry)만 생성하므로, 텍스처·색상·재질(PBR)을 동시에 생성하는 **완전한 3D 에셋 생성**으로의 확장이 필요하다.

$$\mathbf{F}_{\text{complete}} = [\mathbf{F}_{\text{geometry}}, \mathbf{F}_{\text{albedo}}, \mathbf{F}_{\text{roughness}}, \mathbf{F}_{\text{normal}}]$$

#### ② 텍스트-3D 연결 (Text-to-3D)
CLIP, LLM과의 cross-modal alignment를 통해 텍스트 조건부 3D 생성으로 확장해야 한다:

$$p(\mathbf{G}|\text{text}) = p(\mathbf{G}|\mathbf{e}_{\text{text}}), \quad \mathbf{e}_{\text{text}} = \text{CLIP}_{\text{text}}(\text{text})$$

#### ③ 스케일링 법칙 (Scaling Laws) 연구
AR 기반 대규모 언어 모델의 확장성 및 일반화 능력은 스케일링 법칙으로 더 작은 모델에서 더 큰 모델의 성능을 예측할 수 있음이 연구된 바 있다.

3D 생성 모델에서도 데이터 규모, 모델 파라미터, 계산량의 관계를 정량화하는 스케일링 법칙 연구가 필요하다.

#### ④ 도메인 적응 (Domain Adaptation) 전략
실제 스캔 데이터(GSO 등)와 합성 데이터(Objaverse 등) 간의 도메인 갭을 줄이는 연구가 필요하다:
- **Source:** 합성 3D 데이터셋 (10M 규모)
- **Target:** 실제 스캔 객체, 산업용 CAD, 의료 형상 등
- **방법:** Domain-adaptive fine-tuning, Few-shot adaptation

#### ⑤ 동적 형상 및 시퀀스 생성
현재는 **정적 형상**만 생성하므로, 시간 차원 $t$를 추가한 4D 생성:

$$p(\mathbf{G}_{1:T}) = \prod_{t=1}^{T} p(\mathbf{G}_t | \mathbf{G}_{1:t-1})$$

#### ⑥ 조건부 생성의 불확실성 정량화
단일 뷰 이미지 등 모호한 입력 조건에서 **다양한 형상 샘플링** 및 불확실성 추정:

$$\hat{\mathbf{G}}^{(1)}, \hat{\mathbf{G}}^{(2)}, \ldots, \hat{\mathbf{G}}^{(K)} \sim p(\mathbf{G} | \mathbf{I}_{\text{input}})$$

#### ⑦ 평가 지표 표준화
현재 3D 생성 모델의 평가 지표(FID, CD, IoU 등)가 표준화되지 않아 논문 간 공정한 비교가 어렵다. 2D FID에 상응하는 **3D-FID 또는 CLIP-3D Score** 등의 표준 지표 개발이 필요하다.

---

## 참고 자료 (출처 목록)

1. **[주 논문]** Hui, K.-H. et al., "Make-A-Shape: a Ten-Million-scale 3D Shape Model," *ICML 2024*, pp. 20660–20681. https://proceedings.mlr.press/v235/hui24a.html
2. **[arXiv]** arXiv:2401.11067. https://arxiv.org/abs/2401.11067
3. **[HTML 전문]** arXiv HTML (v1, v2). https://arxiv.org/html/2401.11067v1, v2
4. **[공식 GitHub]** AutodeskAILab/Make-a-Shape. https://github.com/AutodeskAILab/Make-a-Shape
5. **[Autodesk Research]** https://www.research.autodesk.com/publications/generative-ai-make-a-shape/
6. **[ACM DL]** https://dl.acm.org/doi/10.5555/3692070.3692899
7. **[OpenReview]** https://openreview.net/forum?id=8l1KYguM4w
8. **[Semantic Scholar]** https://www.semanticscholar.org/paper/022af7d9ba658ef305772386863701e1ac9d3822
9. **[Hugging Face]** https://huggingface.co/papers/2401.11067
10. **[비교 논문] LION:** Zeng et al., "LION: Latent Point Diffusion Models for 3D Shape Generation," arXiv:2210.06978
11. **[비교 논문] 3DShape2VecSet:** Zhang et al., "3DShape2VecSet: A 3D Shape Representation for Neural Fields and Generative Diffusion Models," arXiv:2301.11445
12. **[비교 논문] OctFusion:** Xiong et al., "OctFusion: Octree-based Diffusion Models for 3D Shape Generation," arXiv:2408.14732
13. **[후속 논문] 3D-WAG:** "3D-WAG: Hierarchical Wavelet-Guided Autoregressive Generation for High-Fidelity 3D Shapes," arXiv:2411.19037
14. **[비교] Apple ML Research, Shape Tokens:** https://machinelearning.apple.com/research/3d-shape-tokenization
15. **[DDPM 기반 이론]** Ho, J. et al., "Denoising Diffusion Probabilistic Models," NeurIPS 2020.
