# SV3D: Novel Multi-view Synthesis and 3D Generation from a Single Image using Latent Video Diffusion

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

SV3D(Stable Video 3D)는 **단일 이미지로부터 고해상도의 다중 시점 이미지를 생성하고, 이를 통해 고품질 3D 메시를 생성**하는 latent video diffusion 기반 프레임워크입니다. 기존 이미지 기반 2D 생성 모델들이 다중 시점 일관성(multi-view consistency)과 일반화(generalization) 측면에서 한계를 가졌던 반면, SV3D는 **비디오 확산 모델(SVD: Stable Video Diffusion)을 NVS(Novel View Synthesis)에 적용**함으로써 이 두 가지 문제를 동시에 해결하고자 합니다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **Video Diffusion 기반 NVS** | SVD를 NVS에 최초로 명시적 카메라 포즈 제어 방식으로 적용 |
| **세 가지 핵심 속성 달성** | Pose-controllable, Multi-view consistent, Generalizable |
| **삼각형 CFG 스케일링** | 추론 시 오버샤프닝 방지를 위한 Triangle CFG scaling 제안 |
| **Masked SDS Loss** | 미관측 영역에만 SDS loss를 적용하는 소프트 마스킹 메커니즘 |
| **조명 분리 모델** | Spherical Gaussians(SGs) 기반 조명 모델로 baked-in lighting 문제 완화 |
| **점진적 파인튜닝 전략** | Static orbit → Dynamic orbit 순서의 점진적 난이도 학습 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 NVS 및 3D 생성 방법들은 다음 세 가지 측면에서 한계가 있었습니다:

1. **일반화 부족**: 이미지 기반 2D 생성 모델을 파인튜닝한 방법들(Zero123, MVDream 등)은 대규모 3D 데이터 부족으로 인해 실세계 이미지에 대한 일반화가 제한됨
2. **다중 시점 불일관성**: 단일 뷰를 하나씩 생성하는 방법들(Zero123 계열)은 뷰 간 기하학적·텍스처 불일관성 발생
3. **카메라 제어 불가**: 동시 다중 뷰 생성 방법들(MVDream, SyncDreamer 등)은 임의의 시점 제어 불가

### 2.2 제안하는 방법

#### 2.2.1 문제 설정 (Problem Setting)

입력 이미지 $\mathbf{I} \in \mathbb{R}^{3 \times H \times W}$가 주어졌을 때, 카메라 궤적 $\boldsymbol{\pi} \in \mathbb{R}^{K \times 2} = \{(e_i, a_i)\}_{i=1}^{K}$를 따라 $K = 21$개의 다중 시점 이미지로 구성된 orbital video $\mathbf{J} \in \mathbb{R}^{K \times 3 \times H \times W}$를 생성합니다.

이를 학습된 조건부 분포로부터의 반복적 노이즈 제거(denoising)를 통해 생성합니다:

$$p(\mathbf{J} \mid \mathbf{I}, \boldsymbol{\pi})$$

#### 2.2.2 SV3D 아키텍처

SVD의 UNet 구조를 기반으로 다음 네 가지 수정을 가합니다:

**(i) 불필요한 컨디셔닝 제거:**
- `fps_id`, `motion_bucket_id` 등 비디오 전용 컨디셔닝 벡터 제거

**(ii) 이미지 컨디셔닝 (Latent Concatenation):**
- 컨디셔닝 이미지를 VAE 인코더로 잠재 공간에 임베딩한 후 노이즈 상태 $\mathbf{z}_t$에 채널 방향으로 연결(concatenate)

**(iii) CLIP 임베딩:**
- 컨디셔닝 이미지의 CLIP 임베딩 행렬을 각 transformer block의 cross-attention 레이어에 key와 value로 제공

**(iv) 카메라 포즈 컨디셔닝:**
- 카메라 포즈 $(e_i, a_i)$와 노이즈 타임스텝 $t$를 sinusoidal 위치 임베딩으로 변환
- 이를 선형 변환 후 합산하여 매 residual block에 주입

$$\mathbf{h}_{\text{block}} \leftarrow \mathbf{h}_{\text{block}} + \text{Linear}(\text{Embed}(e_i, a_i) + \text{Embed}(t))$$

#### 2.2.3 Static vs. Dynamic Orbits

- **Static Orbit**: 동일한 앙각(elevation)에서 등간격 방위각(azimuth)으로 순환. 상단/하단 정보 부재 위험
- **Dynamic Orbit**: 불규칙 방위각 + 변동하는 앙각. 정현파의 랜덤 가중 조합으로 구성하여 루프형 궤적 보장

#### 2.2.4 삼각형 CFG 스케일링 (Triangular CFG Scaling)

SVD의 선형 증가 CFG (1→4)는 마지막 프레임을 과선명화(over-sharpening)합니다. 이를 해결하기 위해:

$$\text{CFG}(i) = \begin{cases} 1 + \frac{3}{2} \cdot \frac{i}{K/2} & \text{if } i \leq K/2 \\ 2.5 - \frac{3}{2} \cdot \frac{i - K/2}{K/2} & \text{if } i > K/2 \end{cases}$$

프론트뷰 CFG = 1, 백뷰 CFG = 2.5로 삼각파형(triangle wave) 적용

#### 2.2.5 세 가지 SV3D 모델 변형

| 모델 | 설명 |
|---|---|
| $\text{SV3D}^u$ | 포즈 비컨디셔닝 모델. 정적 오비트만 생성 |
| $\text{SV3D}^c$ | 포즈 컨디셔닝 모델. 동적 오비트로 직접 학습 |
| $\text{SV3D}^p$ | 점진적 파인튜닝: 정적 오비트(55k iter) → 동적 오비트(50k iter) |

### 2.3 3D 생성 파이프라인

#### 2.3.1 Coarse-to-Fine 학습

**Coarse 단계**: Instant-NGP NeRF로 낮은 해상도에서 전체적인 형상 학습 (SDS loss 없이 광도계적 손실만 사용)

**Fine 단계**: NeRF에서 Marching Cubes로 메시 추출 후 DMTet으로 풀 해상도에서 정밀화

#### 2.3.2 SDS Loss

$$\mathcal{L}_{\text{sds}} = w(t)\left(\epsilon_\phi(\mathbf{z}_t; \mathbf{I}, \boldsymbol{\pi}_{\text{rand}}, t) - \epsilon\right)\frac{\partial \hat{\mathbf{J}}}{\partial \theta}$$

여기서 $w(t)$는 타임스텝 의존 가중치, $\epsilon$은 추가된 노이즈, $\epsilon_\phi$는 SV3D가 예측한 노이즈, $\phi$와 $\theta$는 각각 SV3D와 NeRF/DMTet의 파라미터입니다.

#### 2.3.3 Masked SDS Loss

랜덤 카메라 뷰에서 가시성 마스크 $M$을 계산하여 미관측 영역에만 SDS loss 적용:

표면 점 $\mathbf{p}$에서 참조 카메라 $i$를 향한 방향 벡터:

$$\mathbf{v}_i = \frac{\bar{\boldsymbol{\pi}}^i_{\text{ref}} - \mathbf{p}}{||\bar{\boldsymbol{\pi}}^i_{\text{ref}} - \mathbf{p}||}$$

최대 가시성 참조 카메라 선택:

$$c = \max_i (\mathbf{v}_i \cdot \mathbf{n})$$

소프트 마스크 (smoothstep 함수 사용):

$$M = 1 - f_s(\mathbf{v}_c \cdot \mathbf{n},\ 0,\ 0.5)$$

$$f_s(x; f_0, f_1) = \hat{x}^2(3 - 2\hat{x}), \quad \hat{x} = \frac{x - f_0}{f_1 - f_0}$$

최종 Masked SDS Loss:

$$\mathcal{L}_{\text{mask-sds}} = M \mathcal{L}_{\text{sds}}$$

#### 2.3.4 조명 분리 모델 (Disentangled Illumination)

24개의 Spherical Gaussians(SG) 기반 조명 모델 사용. 입력 이미지 $\mathbf{I}$의 HSV-value와 렌더링된 조명 $L$ 사이의 손실:

$$\mathcal{L}_{\text{illum}} = |V(\mathbf{I}) - L|^2, \quad V(\mathbf{c}) = \max(c_r, c_g, c_b)$$

렌더링된 이미지: $\hat{\mathbf{I}} = \mathbf{c}_d L$ (여기서 $\mathbf{c}_d$는 diffuse albedo)

SG 내적 공식:

$$G_1(\mathbf{x}) \cdot G_2(\mathbf{x}) = \int_\Omega G_1(\mathbf{x}) G_2(\mathbf{x}) d\mathbf{x} = \frac{1}{d_m}\left(2\pi a_1 a_2 e^{d_m - \lambda_m}(1.0 - e^{-2d_m})\right)$$

$$\lambda_m = \lambda_1 - \lambda_2, \quad d_m = ||\lambda_1 \boldsymbol{\mu}_1 + \lambda_2 \boldsymbol{\mu}_2||$$

### 2.4 모델 구조 요약

```
입력 이미지 I
    │
    ├─► VAE Encoder → 잠재 벡터 → z_t에 concatenate
    └─► CLIP Encoder → 임베딩 → Cross-attention (Key, Value)

카메라 포즈 (e_i, a_i) → Sinusoidal Embedding
노이즈 타임스텝 t    → Sinusoidal Embedding
                          └─► 합산 후 Linear 변환 → Residual Block에 주입

UNet (SVD-xt 기반):
  각 레이어: Conv3D Block + Spatial Attention + Temporal Attention
        ↓
출력: z_{t-1} → VAE Decoder → 21개 Novel View 이미지

↓
3D Optimization:
  Coarse: Instant-NGP NeRF (600 steps)
  Fine: Marching Cubes → DMTet (1000 steps)
  → UV Unwrapping (xatlas) → 최종 3D Mesh
```

### 2.5 성능 향상

**NVS 성능 (GSO Static Orbit, Table 1):**

| 모델 | LPIPS↓ | PSNR↑ | SSIM↑ |
|---|---|---|---|
| Zero123 | 0.13 | 17.29 | 0.79 |
| Stable Zero123 | 0.13 | 18.34 | 0.78 |
| EscherNet | 0.13 | 16.73 | 0.79 |
| **SV3D $^p$ ** | **0.08** | **21.26** | **0.88** |

**3D 생성 성능 (GSO, Table 6):**

| 모델 | CD↓ | 3D IoU↑ |
|---|---|---|
| Stable Zero123 | 0.039 | 0.550 |
| EscherNet | 0.042 | 0.466 |
| **SV3D $^p$ ** | **0.024** | **0.614** |

**사용자 연구 (User Study):**
SV3D 생성 비디오가 Zero123XL 대비 96%, Stable Zero123 대비 99%, EscherNet 대비 96%, Free3D 대비 98%의 선호도를 기록

### 2.6 한계점

1. **자유도 제한**: 현재 모델은 앙각(elevation)과 방위각(azimuth) 2개의 자유도만 처리. 전체 카메라 행렬 컨디셔닝 불가
2. **반사 표면 처리 한계**: 거울 같은 반사면에서 시점 불일관성 발생
3. **램버시안 반사 가정**: 현재 조명 모델이 Lambertian shading만 지원하여 정반사(specular reflection) 표현 불가
4. **Synthetic-to-Real 갭**: Objaverse 합성 데이터로만 학습하여 실세계 복잡한 장면에 대한 일반화 한계
5. **처리 시간**: SDS 포함 시 전체 메시 생성에 약 20분 소요

---

## 3. 모델의 일반화 성능 향상 가능성

SV3D의 일반화 성능은 다음 세 가지 핵심 요소로부터 비롯됩니다.

### 3.1 Video Diffusion Prior의 활용

SVD는 LAION, LVD 등 **대규모 이미지·비디오 데이터**로 사전 학습된 모델입니다. 3D 데이터는 상대적으로 희소하지만, 비디오 데이터는 풍부하게 존재합니다. SV3D는 이 비디오 사전 지식(temporal consistency)을 **공간적 3D 일관성으로 재해석(repurpose)**하여 활용합니다:

> *"Our main idea is to repurpose temporal consistency in a video diffusion model for spatial 3D consistency of an object."*

이는 기존 이미지 기반 모델 대비 월등한 일반화를 가능하게 합니다.

### 3.2 점진적 파인튜닝 (Progressive Finetuning)

$\text{SV3D}^p$는 **쉬운 태스크(정적 오비트)에서 어려운 태스크(동적 오비트)**로 점진적으로 파인튜닝됩니다. 이 전략은:
- 모델이 기본적인 3D 구조를 먼저 학습한 후 복잡한 시점 변화에 적응하도록 유도
- Static orbit에서도 $\text{SV3D}^u$(정적만 학습)보다 $\text{SV3D}^p$가 더 높은 성능을 보임으로써 전이 학습 효과 입증

### 3.3 Dynamic Orbit 학습

동적 오비트는 ±30도의 앙각 변화를 포함하여 **객체의 상단·하단 정보를 다양하게 학습**할 수 있습니다. 이를 통해:
- 실세계에서 다양한 촬영 각도로 찍힌 이미지에 대한 적응력 향상
- Sine-30 동적 오비트가 정적 오비트 대비 Chamfer Distance에서 0.028 → 0.024로 향상

### 3.4 해상도와 배경 다양성

- **576×576 고해상도** 학습으로 세밀한 질감과 기하학 학습
- 랜덤 RGB 배경 및 백색 배경을 모두 사용하여 배경 변화에 강건한 모델 학습

### 3.5 향후 일반화 향상 가능성

| 방향 | 설명 |
|---|---|
| **전체 카메라 행렬 컨디셔닝** | 현재 2DOF(elevation, azimuth)에서 6DOF로 확장 시 임의 장면 처리 가능 |
| **대규모 실세계 3D 데이터 학습** | CO3D, MVImgNet 등 실세계 멀티뷰 데이터셋 추가 활용 |
| **도메인 적응 기법 적용** | Synthetic-to-Real 갭을 줄이기 위한 도메인 적응 파인튜닝 |
| **더 많은 프레임 생성** | 21프레임에서 더 많은 뷰 생성으로 3D 재구성 품질 향상 |
| **PBR 재질 모델 확장** | Lambertian에서 완전한 PBR(Physically Based Rendering)로 확장 |

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 NVS 방법론 계열 비교

| 연구 | 연도 | 기반 모델 | 해상도 | 다중 시점 일관성 | 임의 시점 제어 | 특징 |
|---|---|---|---|---|---|---|
| **NeRF** (Mildenhall et al.) | 2020 | - | 임의 | ✅ (최적화 기반) | ✅ | 단일 장면 최적화, 일반화 불가 |
| **Zero123** (Liu et al.) | 2023 | SD | 256 | ❌ (1뷰씩 생성) | ✅ | 최초 확산 기반 NVS |
| **Zero123XL** (Deitke et al.) | 2023 | SD | 256 | ❌ | ✅ | Objaverse-XL로 품질 향상 |
| **MVDream** (Shi et al.) | 2023 | SD | 256 | ✅ (4뷰 동시) | ❌ (고정 뷰) | 크로스 어텐션으로 일관성 강화 |
| **SyncDreamer** (Liu et al.) | 2023 | SD | 256 | ✅ (16뷰) | ❌ (고정 뷰) | 3D feature volume 공유 |
| **EscherNet** (Kong et al.) | 2024 | SD | 256 | 부분 ✅ | ✅ | 카메라 포즈 임베딩 설계 |
| **Free3D** (Zheng et al.) | 2023 | SD | 256 | 부분 ✅ | ✅ | 3D 표현 없이 일관성 달성 |
| **IM-3D** (Melas-Kyriazi et al.) | 2024 | 비디오 | - | ✅ | ❌ (고정 고도) | 반복적 멀티뷰 확산 |
| **Vivid-1-to-3** (Kwak et al.) | 2024 | 비디오+이미지 | - | ✅ | 부분 | 비디오+이미지 결합 |
| **SV3D** (본 논문) | 2024 | SVD (비디오) | **576** | ✅ | ✅ | 임의 포즈 제어 + 비디오 prior |

### 4.2 3D 생성 방법론 계열 비교

| 연구 | 연도 | 접근법 | 3D 표현 | 질감 품질 | 속도 |
|---|---|---|---|---|---|
| **DreamFusion** (Poole et al.) | 2022 | Text + SDS | NeRF | 낮음 (Janus 문제) | 느림 |
| **Magic3D** (Lin et al.) | 2023 | Text + SDS | NeRF → DMTet | 중간 | 느림 |
| **Point-E** (Nichol et al.) | 2022 | 확산 3D 직접 | Point Cloud | 낮음 | 빠름 |
| **Shap-E** (Jun et al.) | 2023 | 확산 3D 직접 | Implicit | 낮음 | 빠름 |
| **One-2-3-45** (Liu et al.) | 2023 | NVS → 3D | NeRF | 중간 | 중간 |
| **One-2-3-45++** (Liu et al.) | 2023 | 멀티뷰 + 3D diffusion | - | 중간 | 중간 |
| **DreamGaussian** (Tang et al.) | 2023 | 3D Gaussian Splatting | Gaussian | 중간 | 빠름 |
| **Magic123** (Qian et al.) | 2023 | 2D+3D diffusion | NeRF → DMTet | 높음 | 느림 |
| **Stable Zero123** (StabilityAI) | 2023 | NVS → 3D | DMTet | 중간 | 중간 |
| **SV3D** (본 논문) | 2024 | 비디오 NVS → 3D | NeRF → DMTet | **매우 높음** | 중간 (~20분) |

### 4.3 핵심 차별화 포인트

**기존 비디오 확산 모델 활용 연구와의 차이:**

- SVD-MV와 IM-3D는 비디오 모델을 NVS에 사용하지만 **동일한 앙각의 정적 오비트**만 지원
- SV3D는 최초로 **임의의 앙각·방위각 조합**에 대한 명시적 포즈 제어를 비디오 확산 모델에서 달성

---

## 5. 앞으로의 연구에 미치는 영향 및 고려점

### 5.1 앞으로의 연구에 미치는 영향

#### 5.1.1 패러다임 전환: 비디오 Prior의 3D 활용

SV3D는 **비디오 모델의 시간적 일관성을 공간적 3D 일관성으로 재활용**하는 패러다임을 확립했습니다. 이는 향후 다음과 같은 연구로 확장될 수 있습니다:
- 동적 객체(deformable objects)의 4D 생성 (시공간 확장)
- 장면(scene) 단위의 대규모 3D 생성
- 비디오 생성 모델과 3D 표현의 통합적 학습

#### 5.1.2 단일 이미지 3D 재구성의 실용화

SV3D는 약 20분이라는 합리적인 시간 내에 고품질 3D 메시를 생성함으로써 **실용적 3D 콘텐츠 생성 파이프라인의 기준점**을 제시했습니다. 이는 게임, AR/VR, e-commerce 분야에서 즉각적인 응용 가능성을 보여줍니다.

#### 5.1.3 Foundation Model로서의 잠재력

논문은 SV3D를 "**3D 객체 생성을 위한 견고한 foundation model**"로 위치시킵니다. 이는 이후 연구들이 SV3D를 기반 모델로 삼아 다양한 다운스트림 태스크(재질 편집, 애니메이션 등)에 적용하는 방향으로 발전할 수 있음을 시사합니다.

#### 5.1.4 Masked SDS의 영향

소프트 마스킹 메커니즘을 통해 SDS loss를 미관측 영역에만 선택적으로 적용하는 아이디어는, **재구성 손실과 생성 손실의 충돌 문제**를 해결하는 범용적 기법으로 타 연구에도 적용될 수 있습니다.

### 5.2 향후 연구 시 고려할 점

#### 5.2.1 카메라 파라미터 확장
현재 2DOF(elevation, azimuth)를 완전한 **6DOF 카메라 행렬**로 확장하여 비정형 촬영 조건에도 대응해야 합니다. 이를 위해서는 더 다양한 카메라 파라미터를 포함한 대규모 데이터셋 구축이 필요합니다.

#### 5.2.2 실세계 데이터 학습
현재 Objaverse(합성 데이터) 기반 학습의 한계를 극복하기 위해 **CO3D, MVImgNet, RealEstate10K** 등 실세계 멀티뷰 데이터를 통합하는 연구가 필요합니다. Synthetic-to-Real 도메인 갭을 줄이는 데이터 증강 전략도 중요한 연구 방향입니다.

#### 5.2.3 재질 및 조명 모델의 정교화
현재의 Lambertian + SG 조명 모델을 **완전한 PBR(Physically Based Rendering)** 프레임워크(GGX BRDF 등)로 확장하면 반사면 처리 한계를 극복할 수 있습니다. 이는 metallic, glossy 재질의 객체에 대한 일반화 향상으로 이어집니다.

#### 5.2.4 효율성 개선
비디오 확산 모델의 샘플링에 약 40초, 3D 최적화에 약 20분이 소요되는 현재 파이프라인을 단축하기 위해:
- **Consistency Models** 또는 **Flow Matching** 기반 빠른 샘플링
- **3D Gaussian Splatting** 기반 빠른 재구성과의 결합
- **Feed-forward 3D 예측 네트워크** (LRM 계열)와의 통합

#### 5.2.5 스케일 확장성
현재 단일 객체(object-centric) 생성에 국한된 SV3D를 **복잡한 장면(scene)**으로 확장하려면 배경 처리, 객체 간 상호작용, 대규모 공간 추론 능력을 추가로 갖추어야 합니다.

#### 5.2.6 평가 기준의 다양화
현재 GSO, OmniObject3D 등의 벤치마크는 주로 가정용 물건 중심입니다. **유기적 형상, 인체, 자연 환경** 등 더 다양한 도메인에 대한 평가 기준을 마련하는 것이 중요합니다.

#### 5.2.7 윤리적 고려
- 생성된 3D 콘텐츠의 **저작권 및 귀속 문제** 명확화
- **딥페이크 3D** 생성 방지를 위한 워터마킹 기술 연구
- 훈련 데이터의 **편향성(bias)** 분석 및 완화

---

## 참고 자료

**주요 참고 논문 (논문 내 인용 기준):**

1. **본 논문**: Voleti, V., Yao, C.-H., Boss, M., et al. "SV3D: Novel Multi-view Synthesis and 3D Generation from a Single Image using Latent Video Diffusion." *arXiv:2403.12008v1*, 2024.

2. **SVD (기반 모델)**: Blattmann, A., Dockhorn, T., Kulal, S., et al. "Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets." *arXiv:2311.15127*, 2023.

3. **Zero123**: Liu, R., Wu, R., Van Hoorick, B., et al. "Zero-1-to-3: Zero-shot One Image to 3D Object." *ICCV*, 2023.

4. **DreamFusion**: Poole, B., Jain, A., Barron, J. T., Mildenhall, B. "DreamFusion: Text-to-3D using 2D Diffusion." *arXiv*, 2022.

5. **NeRF**: Mildenhall, B., Srinivasan, P. P., Tancik, M., et al. "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis." *ECCV*, 2020.

6. **Instant-NGP**: Müller, T., Evans, A., Schied, C., Keller, A. "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding." *ACM Trans. Graph.*, 2022.

7. **DMTet**: Shen, T., Gao, J., Yin, K., et al. "Deep Marching Tetrahedra: A Hybrid Representation for High-Resolution 3D Shape Synthesis." *NeurIPS*, 2021.

8. **SyncDreamer**: Liu, Y., Lin, C., Zeng, Z., et al. "SyncDreamer: Generating Multiview-consistent Images from a Single-view Image." *arXiv:2309.03453*, 2023.

9. **MVDream**: Shi, Y., Wang, P., Ye, J., et al. "MVDream: Multi-view Diffusion for 3D Generation." *arXiv:2308.16512*, 2023.

10. **EscherNet**: Kong, X., Liu, S., Lyu, X., et al. "EscherNet: A Generative Model for Scalable View Synthesis." *arXiv:2402.03908*, 2024.

11. **Stable Diffusion**: Rombach, R., Blattmann, A., Lorenz, D., et al. "High-Resolution Image Synthesis with Latent Diffusion Models." *CVPR*, 2022.

12. **Objaverse**: Deitke, M., Schwenk, D., Salvador, J., et al. "Objaverse: A Universe of Annotated 3D Objects." *CVPR*, 2023.

13. **Magic3D**: Lin, C.-H., Gao, J., Tang, L., et al. "Magic3D: High-Resolution Text-to-3D Content Creation." *CVPR*, 2023.

14. **Free3D**: Zheng, C., Vedaldi, A. "Free3D: Consistent Novel View Synthesis without 3D Representation." *arXiv:2312.04551*, 2023.

15. **프로젝트 페이지**: https://sv3d.github.io/
