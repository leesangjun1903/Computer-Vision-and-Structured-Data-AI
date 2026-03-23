# 360-GS: Layout-guided Panoramic Gaussian Splatting For Indoor Roaming

---

## 1. 핵심 주장 및 주요 기여 요약

**360-GS**는 소수의 실내 파노라마(equirectangular) 이미지로부터 3D Gaussian Splatting(3D-GS)을 최적화하여 **실시간 파노라마 렌더링과 고품질 Novel View Synthesis(NVS)**를 달성하는 파이프라인이다. 핵심 주장은 다음 세 가지로 압축된다:

1. **360° Gaussian Splatting 알고리즘**: 3D Gaussian을 구면(spherical surface)에 직접 스플래팅하는 대신, 단위 구의 **접선 평면(tangent plane)** 에 투영한 뒤 구면 좌표로 매핑하여, 2D Gaussian으로 표현 불가능한 왜곡 문제를 해결하고 직접적인 파노라마 렌더링을 가능하게 함.

2. **Layout-guided Initialization**: 실내 장면의 방 레이아웃(벽·바닥·천장 경계)과 깊이 맵으로부터 고품질 3D 포인트 클라우드를 생성하여, SfM이 실패하는 sparse-view 환경에서도 신뢰성 있는 3D Gaussian 초기화를 제공함.

3. **Layout-guided Regularization**: 레이아웃 평면 위의 3D Gaussian 위치 이동을 법선(normal) 방향으로 제약하여, texture-less 평면 영역에서 발생하는 **floater 아티팩트**를 효과적으로 억제함.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

| 문제 | 설명 |
|------|------|
| **구면 투영 모델링 불가** | 3D-GS의 로컬 아핀 근사(local affine approximation)는 perspective projection에 특화되어 있어, equirectangular 이미지의 구면 왜곡을 2D Gaussian으로 표현할 수 없음 (Figure 3 참조) |
| **Sparse 입력** | 실내 파노라마는 보통 4장 이하의 희소 뷰만 확보 가능하며, SfM 기반 초기화가 실패함 |
| **Texture-less 평면** | 벽·바닥·천장 등 텍스처 없는 영역은 cross-view correspondence가 부족하여 기하 추정이 부정확하고 floater 발생 |

### 2.2 제안 방법 (수식 포함)

#### (A) 3D Gaussian 기본 표현

각 3D Gaussian은 위치 $\boldsymbol{\mu} \in \mathbb{R}^3$와 공분산 행렬 $\Sigma \in \mathbb{R}^{3 \times 3}$로 정의된다:

$$G(\mathbf{x}) = e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})}$$

$$\Sigma = \mathbf{R}\mathbf{S}\mathbf{S}^T\mathbf{R}^T$$

기존 3D-GS에서는 Jacobian $\mathbf{J}$를 이용한 로컬 아핀 근사로 2D 공분산을 구한다:

```math
\Sigma' = \mathbf{J}\mathbf{W}\Sigma\mathbf{W}^T\mathbf{J}^T
```

#### (B) 360° Gaussian Splatting — 접선 평면 투영 + 구면 매핑

**Step 1: 접선 평면 투영.** 3D Gaussian의 중심 $\boldsymbol{\mu}$를 카메라 좌표계로 변환한 후, 단위 구 위의 투영점 $\boldsymbol{\mu}'$를 통과하는 접선 평면으로 사영한다:

$$\boldsymbol{\mu}'(\mathbf{x} - \boldsymbol{\mu}') = 0$$

투영 변환 $\varphi$는:

$$(t'_0, t'_1, t'_2)^T = \varphi(\mathbf{t}, \boldsymbol{\mu}') = \mathbf{t} \frac{(\boldsymbol{\mu}')^T \boldsymbol{\mu}'}{(\boldsymbol{\mu}')^T \mathbf{t}}$$

Taylor 전개에 의한 로컬 아핀 근사:

$$\varphi_k(\mathbf{t}, \boldsymbol{\mu}') = \varphi_k(\mathbf{t}_k, \boldsymbol{\mu}') + \mathbf{J}_k \cdot (\mathbf{t} - \mathbf{t}_k)$$

여기서 Jacobian $\mathbf{J}_k$는:

```math
\mathbf{J}_k = \frac{-1}{\left(\mu'_0 t_0 + \mu'_1 t_1 + \mu'_2 t_2\right)^2} \begin{bmatrix} \mu'_1 t_1 + \mu'_2 t_2 & \mu'_1 t_0 & \mu'_2 t_0 \\ \mu'_0 t_1 & \mu'_0 t_0 + \mu'_2 t_2 & \mu'_2 t_1 \\ \mu'_0 t_2 & \mu'_1 t_2 & \mu'_0 t_0 + \mu'_1 t_1 \end{bmatrix}
```

접선 평면 위의 2D Gaussian 분포:

$$G'(\mathbf{t}') = e^{-\frac{1}{2}(\mathbf{t}'-\boldsymbol{\mu}')^T (\mathbf{J}_k \mathbf{W}\Sigma\mathbf{W}^T \mathbf{J}_k^T)^{-1}(\mathbf{t}'-\boldsymbol{\mu}')}$$

**Step 2: 구면 매핑.** 접선 평면의 좌표를 구면 극좌표 $(\theta, \phi)$로 변환:

```math
\begin{pmatrix} \theta \\ \phi \end{pmatrix} = P_{360}(\mathbf{t}') = \begin{pmatrix} \text{atan2}\left(-t'_1, \sqrt{t'^2_0 + t'^2_2}\right) \\ \text{atan2}(t'_0, t'_2) \end{pmatrix}
```

파노라마 픽셀 좌표와의 관계:

```math
\begin{cases} r = -\theta \cdot H/\pi + H/2 \\ c = \phi \cdot W/(2\pi) + W/2 \end{cases}
```

#### (C) Layout-guided Initialization

1. 각 파노라마에서 HorizonNet으로 바닥-벽 경계 $\mathbf{B}_f \in \mathbb{R}^{1 \times W}$, 천장-벽 경계 $\mathbf{B}_c \in \mathbb{R}^{1 \times W}$를 추정
2. 모든 파노라마의 경계를 2D union 연산으로 병합하여 글로벌 레이아웃 생성
3. 3D bounding box를 구성하고 균일 샘플링으로 **레이아웃 포인트 클라우드** 생성
4. SliceNet으로 깊이 맵 추정 → 깊이 포인트 클라우드 생성 → 스케일 정렬 후 합산
5. 결합된 포인트 클라우드로 3D Gaussian 초기화

#### (D) Layout-guided Regularization

레이아웃 포인트 클라우드에서 초기화된 Gaussian의 이동 방향을 법선 방향으로 제약:

$$\mathcal{L}_{\text{layout}} = \sum \frac{\mathbf{n} \cdot (\boldsymbol{\mu} - \mathbf{u}_0)}{\|\mathbf{n}\| \times \|\boldsymbol{\mu} - \mathbf{u}_0\|}$$

여기서 $\mathbf{u}_0$은 초기 위치, $\mathbf{n}$은 레이아웃 표면의 법선 벡터이다.

#### (E) 최종 손실 함수

$$\mathcal{L} = \lambda_1 \|\mathbf{C} - \hat{\mathbf{C}}\|_1 + \lambda_2 \mathcal{L}_{\text{D-SSIM}} + \lambda_3 \mathcal{L}_{\text{layout}}$$

- 4-view: $\lambda_1=0.8,\ \lambda_2=0.2,\ \lambda_3=0.1$
- 32-view: $\lambda_3=0.01$

### 2.3 모델 구조 (Figure 2 기반)

```
입력 파노라마 → [HorizonNet] → 룸 레이아웃 → 3D 포인트 클라우드 ─┐
              → [SliceNet]   → 깊이 맵    → 3D 포인트 클라우드 ─┤
                                                              ├→ 정렬·병합 → 3D Gaussian 초기화
                                                              │
3D Gaussians → [360° Gaussian Splatting] → 파노라마 공간 → [Tile Rasterizer] → 렌더링 결과
     ↑                                                                            │
     └──── Layout-guided Regularization ←── Gradient Flow ────────────────────────┘
```

### 2.4 성능 향상

| 설정 | 메트릭 | MipNeRF-360 | INGP | 3D-GS* | 3D-GS(SfM) | **360-GS(Ours)** |
|------|--------|-------------|------|--------|-------------|-----------------|
| 4-view | PSNR↑ | **19.15** | 15.49 | 13.92 | - | 18.96 |
| 4-view | LPIPS↓ | 0.374 | 0.586 | 0.547 | - | **0.344** |
| 32-view | PSNR↑ | 26.72 | **28.23** | 21.65 | 26.74 | 28.22 |
| 32-view | SSIM↑ | 0.835 | 0.860 | 0.704 | 0.837 | **0.871** |
| **FPS** | - | 0.07 | 3.08 | 60 | - | **60** |

**Ablation 결과** (4-view):
- Baseline(random init): PSNR 13.64
- +Init: 15.98 (+2.34)
- +Init+360GS: 16.66 (+0.68)
- +Init+360GS+LR: **17.72** (+1.06)

핵심: **MipNeRF-360 대비 ~857배 빠른 렌더링** (60 vs 0.07 FPS)으로 비교 가능한 품질 달성

### 2.5 한계

1. **Off-the-shelf 네트워크 의존성**: HorizonNet, SliceNet 등의 사전 학습 모델이 복잡한 장면에서 부정확한 prior를 제공할 수 있음
2. **저장 공간**: 레이아웃 포인트 클라우드가 dense plane 샘플링으로 인해 디스크 공간을 더 많이 차지
3. **Atlanta World 가정**: 방 레이아웃이 수직 벽·수평 바닥/천장으로 구성된다는 가정이 비정형 실내 구조(예: 다락방, 계단)에 적용 어려움

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화 측면에서의 강점

- **Scene prior 활용**: 룸 레이아웃이라는 **장면 수준(scene-level)** 구조 정보를 활용하여, 뷰 수에 둔감한(robust) 성능을 보임. Figure 8에서 4-view부터 32-view까지 일관되게 높은 성능을 유지함.
- **360° splatting의 범용성**: 접선 평면 투영 + 구면 매핑 방식은 어떠한 equirectangular 이미지에도 적용 가능한 수학적 프레임워크를 제공.

### 3.2 일반화 성능 향상을 위한 방향

| 방향 | 설명 |
|------|------|
| **다양한 레이아웃 모델** | Atlanta World 가정을 완화하여 비맨해튼/비직교 구조(곡면 벽, 경사 천장)를 지원하는 레이아웃 추정 모델 도입 |
| **Foundation Model 기반 depth** | Depth Anything V2, Metric3D 등 최신 foundation 깊이 추정 모델을 활용하면 다양한 장면에 대해 더 정확한 depth prior 확보 가능 |
| **자기지도 정규화** | 레이아웃 외에 의미론적 일관성(semantic consistency), 광학 흐름(optical flow) 등을 추가 정규화로 도입하면 레이아웃이 없는 옥외 장면으로도 확장 가능 |
| **Cross-scene 학습** | 다수의 장면에서 공유되는 구조적 패턴을 학습하는 generalizable Gaussian splatting 모델로 발전 가능 |
| **동적 장면 확장** | 시간 축을 추가하여 4D Gaussian으로 확장하면 동적 실내 장면(사람 이동 등)에도 적용 가능 |

### 3.3 주요 일반화 제약

- 레이아웃 추정 네트워크 자체의 도메인 의존성 (학습 데이터셋의 편향)
- 초기화 포인트 클라우드 품질이 최종 성능을 크게 좌우하는 구조적 의존성
- 비정형 가구·투명체·반사체 등 레이아웃으로 기술 불가능한 객체 영역에 대한 약점

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구 영향

1. **Panoramic Gaussian Splatting의 기초 확립**: 3D-GS를 panoramic 도메인으로 확장하는 최초의 체계적 프레임워크로, 이후 PanoGS, OmniGS 등의 후속 연구에 직접적 기반이 됨.

2. **Scene-level Prior + 3D-GS 결합 패러다임**: 기존의 pixel-wise depth supervision과 달리 scene-level structural prior(레이아웃)를 3D Gaussian 최적화에 명시적으로 결합하는 새로운 접근법을 제시. 이는 semantic prior, floor plan prior 등 다양한 고수준 정보 활용의 물꼬를 터줌.

3. **Real-time Indoor Roaming**: VR/AR 실내 투어에서 실시간(60 FPS) 파노라마 렌더링의 실용성을 입증하여, 산업 응용(부동산 가상 투어, 인테리어 시뮬레이션)에 직접 기여.

### 4.2 향후 연구 시 고려사항

| 항목 | 고려사항 |
|------|----------|
| **레이아웃 추정 정확도** | 복잡한 실내 환경에서 레이아웃 추정 오류가 Gaussian 초기화 및 정규화에 연쇄적으로 영향. 레이아웃 불확실성을 모델에 반영하는 방법론 필요 |
| **확장성** | 다중 방(multi-room) 환경, 대규모 건물로의 확장 시 레이아웃 병합 전략과 메모리 효율화 필요 |
| **Hybrid Representation** | 2D Gaussian Splatting, 메쉬 기반 표현 등과의 하이브리드 접근으로 평면 영역 표현 효율 향상 가능 |
| **Anti-aliasing** | 구면 매핑 시 극지방(pole)에서의 aliasing 문제에 대한 추가 처리 필요 |
| **외부 장면 적용** | 실외 환경에서는 레이아웃 가정이 성립하지 않으므로, 대체 prior(예: ground plane, sky model) 설계가 요구됨 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 표현 방식 | 입력 | 핵심 특징 | 360-GS와의 차이 |
|------|------|----------|------|----------|----------------|
| **NeRF** (Mildenhall et al.) | 2020 | Implicit (MLP) | Perspective | 볼륨 렌더링으로 NVS | 실시간 불가, 파노라마 미지원 |
| **Mip-NeRF 360** (Barron et al.) | 2021 | Implicit | Perspective | Anti-aliasing, unbounded scene | 12시간 학습, 0.07 FPS — 실시간 불가 |
| **Instant NGP** (Müller et al.) | 2022 | Hash grid + MLP | Perspective/Panorama | 빠른 학습·렌더링 | 3.08 FPS(파노라마), sparse view에서 성능 저하 심각 |
| **3D Gaussian Splatting** (Kerbl et al.) | 2023 | Explicit 3D Gaussians | Perspective | 실시간 렌더링, SfM 초기화 | 파노라마 직접 지원 불가, stitching 아티팩트 |
| **360Roam** (Huang et al.) | 2022 | NeRF (omnidirectional) | Panorama | 파노라마 NeRF의 선구 연구 | 실시간 불가, layout prior 미활용 |
| **360FusionNeRF** (Kulkarni et al.) | 2022 | NeRF + semantic loss | Panorama | Semantic consistency loss | 실시간 불가, 구조 정보 간접 활용 |
| **SparseNeRF** (Guangcong et al.) | 2023 | NeRF + depth ranking | Perspective | Few-shot을 위한 depth ranking distillation | Pixel-wise prior만 활용, scene-level prior 미사용 |
| **FSGS** (Zhu et al.) | 2023 | 3D-GS | Perspective | Few-shot Gaussian Splatting | 파노라마 미지원, perspective 기반 |
| **SparseGS** (Xiong et al.) | 2023 | 3D-GS | Perspective | 360° sparse view에서 3D-GS | 파노라마 직접 렌더링 미지원 |
| **StructNeRF** (Chen et al.) | 2022 | NeRF + structural hints | Perspective | 구조적 정규화로 실내 NeRF 개선 | NeRF 기반, 실시간 불가 |
| **Layout-Guided NVS** (Xu et al.) | 2021 | Neural features | Single panorama | 레이아웃 가이드 뷰 합성 | 단일 뷰, 3D Gaussian 미사용, 신경망 기반 |
| **360-GS (본 논문)** | **2024** | **3D Gaussians** | **Panorama (sparse)** | **360° splatting + layout prior** | **실시간 파노라마 렌더링, scene-level prior 명시적 활용** |

### 핵심 비교 인사이트

1. **NeRF 계열 vs. 360-GS**: NeRF 방법들(Mip-NeRF 360, INGP)은 높은 렌더링 품질을 달성하지만 실시간성이 부족. 360-GS는 60 FPS로 비교 가능한 품질을 달성.

2. **기존 3D-GS vs. 360-GS**: 기존 3D-GS는 perspective 투영에 한정되어 파노라마에 직접 적용 시 stitching 아티팩트가 불가피. 360-GS의 접선 평면 투영 방식은 이를 근본적으로 해결.

3. **Few-shot 방법들과의 비교**: SparseNeRF, FSGS 등은 pixel-wise depth를 prior로 활용하지만, 360-GS는 **scene-level layout prior**를 결합하여 sparse view에서 더 robust한 기하 재구성을 달성.

---

## 참고 자료

- Bai, J. et al. (2024). "360-GS: Layout-guided Panoramic Gaussian Splatting For Indoor Roaming." *arXiv:2402.00763v1* — 본 논문
- Kerbl, B. et al. (2023). "3D Gaussian Splatting for Real-Time Radiance Field Rendering." *ACM Transactions on Graphics (TOG)*, 42.
- Mildenhall, B. et al. (2020). "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis." *ECCV 2020*.
- Barron, J. T. et al. (2021). "Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields." *CVPR 2022*.
- Müller, T. et al. (2022). "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding." *ACM TOG*, 41.
- Huang, H. et al. (2022). "360Roam: Real-Time Indoor Roaming Using Geometry-Aware 360° Radiance Fields." *arXiv:2208.02705*.
- Sun, C. et al. (2019). "HorizonNet: Learning Room Layout with 1D Representation and Pano Stretch Data Augmentation." *CVPR 2019*.
- Pintore, G. et al. (2021). "SliceNet: deep dense depth estimation from a single indoor panorama using a slice-based representation." *CVPR 2021*.
- Zwicker, M. et al. (2002). "EWA Splatting." *IEEE TVCG*, 8(3), 223–238.
- Zhu, Z. et al. (2023). "FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting." *arXiv:2312.00451*.
- Xiong, H. et al. (2023). "SparseGS: Real-Time 360° Sparse View Synthesis using Gaussian Splatting." *arXiv:2312.00206*.
- Guangcong et al. (2023). "SparseNeRF: Distilling Depth Ranking for Few-shot Novel View Synthesis." *ICCV 2023*.
- Chen, Z. et al. (2022). "StructNeRF: Neural Radiance Fields for Indoor Scenes with Structural Hints." *arXiv:2209.05277*.
- Xu, J. et al. (2021). "Layout-Guided Novel View Synthesis from a Single Indoor Panorama." *CVPR 2021*.
- Kulkarni, S. et al. (2022). "360FusionNeRF: Panoramic Neural Radiance Fields with Joint Guidance." *arXiv:2209.14265*.
- Pintore, G. et al. (2020). "AtlantaNet: Inferring the 3D Indoor Layout from a Single 360° Image Beyond the Manhattan World Assumption." *ECCV 2020*.
