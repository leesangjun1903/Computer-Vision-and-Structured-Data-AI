# Street Gaussians: Modeling Dynamic Urban Scenes with Gaussian Splatting

---

# 1. 핵심 주장 및 주요 기여 요약

이 논문은 자율주행 장면의 동적 도시 거리(dynamic urban streets)를 모델링하는 문제를 해결하고자 합니다. 기존 NeRF 기반 방법은 추적된 차량 포즈를 활용하여 동적 장면의 photo-realistic 뷰 합성을 수행하지만, 학습 및 렌더링 속도가 느리다는 심각한 한계를 갖고 있습니다.

이를 해결하기 위해 **Street Gaussians**라는 새로운 명시적(explicit) 장면 표현을 도입하며, 동적 도시 장면을 시맨틱 로짓(semantic logits)과 3D Gaussian을 가진 포인트 클라우드 집합으로 표현합니다. 각 포인트 클라우드는 전경 차량(foreground vehicle) 또는 배경(background)에 연결됩니다.

**주요 기여 (Key Contributions):**

1. 포인트 클라우드를 사용하여 동적 장면을 구축하는 새로운 명시적 장면 표현으로, 학습 및 렌더링 효율성을 크게 높이면서 추적된 차량 포즈의 정확도에 대한 의존성을 줄입니다.
2. 시변(time-varying) 외관을 시계열 함수로 구형 조화(spherical harmonics) 계수를 예측하는 **4D Spherical Harmonics** 모델로 표현합니다.
3. 약 30분 이내의 학습으로 실시간 렌더링(135 FPS@1066×1600)이 가능하며, 추적 포즈 최적화(tracked pose optimization) 전략을 개발했습니다.
4. off-the-shelf tracker의 포즈만을 사용하면서도 ground-truth 포즈와 비슷한 결과를 달성합니다.

---

# 2. 상세 분석

## 2.1 해결하고자 하는 문제

자율주행을 위한 동적 도시 거리 모델링에서 기존 NeRF 기반 방법들은 차량 포즈를 통해 동적 객체를 애니메이션화하지만, 학습 및 렌더링 속도가 느린 치명적 한계가 있습니다. 또한 기존 방법들은 관측 공간(observation space)에서 표준 공간(canonical space)으로의 매핑에 NeRF 네트워크를 사용하므로 추적 바운딩 박스의 정확도에 민감하며, 높은 학습 비용과 낮은 렌더링 속도라는 한계를 가집니다.

## 2.2 제안하는 방법 (수식 포함)

### (A) 장면 분해 (Scene Decomposition)

핵심 아이디어는 포인트 클라우드를 사용하여 동적 장면을 구축하는 것이며, 도시 거리 장면을 정적 배경과 이동 차량으로 분해하여 각각 3D Gaussian으로 구축합니다.

전체 장면은 배경 모델과 $N$개의 전경 객체 모델의 합성으로 표현됩니다:

$$\mathcal{S} = \{\mathcal{B}, \mathcal{O}_1, \mathcal{O}_2, \ldots, \mathcal{O}_N\}$$

여기서 $\mathcal{B}$는 배경 모델, $\mathcal{O}_i$는 $i$번째 전경 객체 모델입니다.

### (B) 3D Gaussian 표현

각 포인트에는 위치(position), 불투명도(opacity), 회전과 스케일로 구성된 공분산(covariance)을 포함하는 3D Gaussian이 할당되어 기하학적 구조를 표현합니다.

각 3D Gaussian $G$는 다음과 같이 정의됩니다:

$$G(\mathbf{x}) = e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})}$$

여기서 $\boldsymbol{\mu} \in \mathbb{R}^3$는 중심 위치, $\boldsymbol{\Sigma} \in \mathbb{R}^{3\times3}$는 공분산 행렬입니다. 공분산 행렬은 스케일 행렬 $\mathbf{S}$와 회전 행렬 $\mathbf{R}$로 분해됩니다:

$$\boldsymbol{\Sigma} = \mathbf{R}\mathbf{S}\mathbf{S}^T\mathbf{R}^T$$

### (C) 동적 전경 모델링 — Optimizable Tracked Poses

전경 차량의 동적 특성을 처리하기 위해 기하학을 최적화 가능한 추적 차량 포즈를 가진 포인트 집합으로 모델링하며, 각 포인트는 학습 가능한 3D Gaussian 파라미터를 저장합니다.

각 전경 객체 $\mathcal{O}_i$의 포인트 $\mathbf{p}^{\text{obj}}$는 객체 좌표계에서 정의되며, 시간 $t$에서의 월드 좌표는 다음과 같이 변환됩니다:

$$\mathbf{p}^{\text{world}}_t = \mathbf{T}_t^{(i)} \cdot \mathbf{p}^{\text{obj}}$$

여기서 $\mathbf{T}_t^{(i)} \in SE(3)$는 시간 $t$에서 $i$번째 객체의 최적화 가능한 강체 변환(rigid transformation)입니다. 이 포즈는 초기값이 off-the-shelf tracker로부터 주어지되, 학습 과정에서 공동으로 최적화됩니다.

### (D) 4D Spherical Harmonics (동적 외관 모델)

배경 포인트에는 일반 구형 조화 모델을, 전경 포인트에는 **동적 구형 조화(dynamic spherical harmonics)** 모델을 할당합니다. 시변 외관은 시계열 함수를 사용하여 임의의 시간 단계에서 구형 조화 계수를 예측하는 4D 구형 조화 모델로 표현됩니다.

기존 3D Gaussian Splatting에서 색상은 SH 계수 $\mathbf{c}_l^m$으로 표현됩니다:

$$c(\mathbf{d}) = \sum_{l=0}^{L} \sum_{m=-l}^{l} \mathbf{c}_l^m Y_l^m(\mathbf{d})$$

Street Gaussians에서는 전경 객체의 시변 외관을 위해 SH 계수를 시간의 함수로 확장합니다:

$$c(\mathbf{d}, t) = \sum_{l=0}^{L} \sum_{m=-l}^{l} \mathbf{c}_l^m(t) \cdot Y_l^m(\mathbf{d})$$

여기서 시간 종속 계수 $\mathbf{c}_l^m(t)$는 다음과 같은 다항식 시계열 함수로 모델링됩니다:

$$\mathbf{c}_l^m(t) = \sum_{k=0}^{K} \mathbf{a}_{l,k}^m \cdot t^k$$

$\mathbf{a}_{l,k}^m$는 학습 가능한 파라미터이며, $K$는 다항식의 차수입니다. 논문에서는 $K=5$로 설정하여 성능과 저장 비용 간의 균형을 유지합니다.

### (E) 렌더링 및 합성 (Composition & Rendering)

명시적 포인트 기반 표현은 별도 모델의 쉬운 합성을 가능하게 하여 고품질 이미지와 시맨틱 맵의 실시간 렌더링, 전경 객체 분해를 통한 편집 애플리케이션을 지원합니다.

렌더링은 타일 기반 래스터라이저를 사용하여 알파 블렌딩으로 수행됩니다:

$$\mathbf{C}(\mathbf{u}) = \sum_{i \in \mathcal{N}} \mathbf{c}_i \cdot \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)$$

여기서 $\mathbf{u}$는 픽셀 좌표, $\mathbf{c}_i$는 $i$번째 Gaussian의 색상, $\alpha_i$는 불투명도입니다.

### (F) 손실 함수 (Loss Function)

학습을 위한 총 손실 함수는 다음과 같습니다:

$$\mathcal{L} = \mathcal{L}_{\text{rgb}} + \lambda_{\text{ssim}} \mathcal{L}_{\text{ssim}} + \lambda_{\text{reg}} \mathcal{L}_{\text{reg}}$$

여기서:
- $\mathcal{L}_{\text{rgb}}$: L1 photometric loss
- $\mathcal{L}_{\text{ssim}}$: D-SSIM loss (구조적 유사도)
- $\mathcal{L}_{\text{reg}}$: 분해된 전경 객체의 누적 알파 값에 대한 엔트로피 손실(entropy loss)로, 적응적 제어 과정 후에 추가되어 전경과 배경을 더 잘 구분하게 합니다.

### (G) 포인트 클라우드 초기화

SfM은 under-observed 영역이나 텍스처가 없는 영역에서 좋은 초기화를 제공하지 못하므로, 대신 에고 차량이 캡처한 집계된 LiDAR 포인트 클라우드를 초기화로 사용합니다.

## 2.3 모델 구조

전체 파이프라인은 다음과 같은 구조입니다:

| 구성 요소 | 설명 |
|-----------|------|
| **배경 모델 $\mathcal{B}$** | 월드 좌표계의 정적 포인트 클라우드 + 정적 SH |
| **전경 모델 $\mathcal{O}_i$** | 객체 좌표계의 포인트 클라우드 + 최적화 가능 포즈 $\mathbf{T}_t^{(i)}$ + 4D SH |
| **시맨틱 헤드** | 각 포인트에 시맨틱 로짓 할당 (선택적) |
| **포즈 최적화** | tracker 초기 포즈를 gradient descent로 공동 최적화 |
| **하늘 모델** | 하늘 영역 별도 모델링 |

## 2.4 성능 향상

Waymo Open과 KITTI 데이터셋에서 평가하여 렌더링 품질 측면에서 최첨단 성능을 달성하며, 이전 방법들보다 100배 이상 빠른 렌더링 속도를 기록합니다.

| 성능 지표 | Street Gaussians |
|-----------|-----------------|
| **렌더링 속도** | 135 FPS @ 1066×1600 해상도 |
| **학습 시간** | 약 30분 이내 |
| **렌더링 속도 비교** | 이전 방법 대비 100배 이상 빠름 |

Street Gaussians는 객체 분해(object decomposition), 시맨틱 분할(semantic segmentation), 장면 편집(scene editing) 등 다양한 컴퓨터 비전 태스크에 적용 가능합니다.

## 2.5 한계점

이 연구에는 알려진 한계가 있습니다. 첫째, 이 방법은 정적 거리에 이동 차량만 있는 강체(rigid) 동적 장면 재구성에 제한되며, 보행자와 같은 비강체(non-rigid) 동적 객체를 처리할 수 없습니다.

추가적인 한계:
- **3D 바운딩 박스 의존성**: Street Gaussians는 명시적 3D 바운딩 박스에 의존하여 분해 문제를 단순화합니다.
- **추적 오류에 대한 민감성**: Street Gaussians는 3D tracker의 추적 오류로 인해 잘못된 재구성이나 객체 누락이 발생할 수 있습니다.
- **센서 데이터 의존성**: LiDAR 초기화에 의존하므로 LiDAR가 없는 환경에서는 적용이 제한적입니다.

---

# 3. 모델의 일반화 성능 향상 가능성

## 3.1 현재의 일반화 능력

논문은 제안된 장면 표현이 off-the-shelf tracker의 포즈만으로 정밀한 ground-truth 포즈와 비슷한 성능을 달성할 수 있음을 보여줍니다. 이는 명시적 표현의 더 나은 gradient 전파 덕분에 가능하며, 이를 통해 다양한 트래커와의 호환성이라는 일반화 이점을 제공합니다.

Waymo Open과 KITTI 데이터셋이라는 서로 다른 환경 조건에서 모두 평가하여 방법의 범용성을 입증합니다.

## 3.2 일반화 성능 향상을 위한 가능한 방향

| 방향 | 설명 |
|------|------|
| **비강체 동적 객체 확장** | 보행자, 자전거 등 비강체 객체로의 확장 (SMPL 기반 모델 통합 등) |
| **3D 바운딩 박스 의존성 제거** | DeSiRe-GS처럼 자기지도 분해(self-supervised decomposition)를 사용하면 3D 어노테이션 없이도 효과적인 분해가 가능합니다. |
| **2D 트래커 기반 확장** | 후속 연구인 "Street Gaussians without 3D Object Tracker"는 ground-truth 차량 포즈나 3D tracker 대신 2D deep tracker를 활용하여 robustness를 향상시킵니다. |
| **다양한 환경 조건** | 날씨, 조명 변화, 야간 환경 등에서의 일반화 성능 개선 |
| **멀티카메라 일관성** | 다시점 카메라 시스템에서의 일관성 확보 |

## 3.3 Bézier Curve 기반 포즈 일반화

최근 연구에서는 학습 가능한 Bézier 곡선을 사용하여 동적 객체의 운동 궤적을 표현하고, 포즈 오류를 자동으로 수정하는 방법이 제안되었습니다. 이는 Street Gaussians의 이산적(discrete) 포즈 최적화를 연속적(continuous) 궤적 모델링으로 발전시킨 것입니다.

---

# 4. 향후 연구에 미치는 영향 및 고려할 점

## 4.1 연구에 미치는 영향

1. **실시간 자율주행 시뮬레이션의 새 패러다임 제시**: NeRF의 implicit 표현에서 3D Gaussian Splatting의 explicit 표현으로의 전환을 동적 도시 장면에 성공적으로 적용하여, 후속 연구의 기반이 되었습니다.

2. **장면 편집 가능성**: 인스턴스 인식(instance-aware) 장면 표현은 차량 회전, 이동, 교체 등 다양한 장면 편집 작업을 가능하게 합니다. 이는 자율주행 시뮬레이션의 핵심 요소입니다.

3. **후속 연구의 폭발적 증가**: Street Gaussians 이후 PVG, DrivingGaussian, S³Gaussian, DeSiRe-GS, OmniRe 등 다수의 후속 연구가 3DGS 기반 동적 도시 장면 모델링으로 전환되었습니다.

## 4.2 향후 연구 시 고려할 점

| 고려 사항 | 세부 내용 |
|-----------|----------|
| **비강체 객체 지원** | 보행자, 동물 등의 비강체 동적 객체 모델링 |
| **어노테이션 비용 절감** | 3D 바운딩 박스 없이도 동적/정적 분해가 가능한 자기지도 방법 |
| **시간 도메인 언더샘플링** | 자기지도 방법들은 관측된 타임스탬프에 과적합(overfit)되어 중간 프레임에서 성능이 저하되는 문제가 있으며, 특히 빠르게 움직이는 객체에서 동적 학습의 신뢰성이 떨어집니다. |
| **대규모 장면 확장** | 도시 전체 규모의 장면으로 확장 시의 메모리 효율성 |
| **표면 재구성 품질** | 단순한 뷰 합성을 넘어 물리적으로 합리적인 기하학적 재구성 |
| **센서 다양성** | LiDAR 없는 카메라 전용 시스템에서의 적용 가능성 |

---

# 5. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 연도/학회 | 표현 방식 | 동적 모델링 | 3D 어노테이션 필요 | 실시간 렌더링 | 핵심 특징 |
|------|----------|----------|------------|-------------------|-------------|----------|
| **NSG** (Ost et al.) | 2021/CVPR | NeRF (implicit) | Scene graph + NeRF | ✅ | ❌ | 그래프 기반 장면 분해 |
| **SUDS** (Turki et al.) | 2023/CVPR | NeRF (implicit) | 시간 인코딩 | ❌ (자기지도) | ❌ | 확장성 있는 도시 동적 장면 |
| **MARS** (Wu et al.) | 2023/CICAI | NeRF (implicit) | 인스턴스 인식 모듈형 시뮬레이터 | ✅ | ❌ | 모듈형 자율주행 시뮬레이터 |
| **EmerNeRF** (Yang et al.) | 2024/ICLR | NeRF (implicit) | 자기지도 기반 시공간 장면 분해 | ❌ | ❌ | Hash grid, self-supervised |
| **DrivingGaussian** (Zhou et al.) | 2024/CVPR | 3DGS (explicit) | Incremental Static 3D Gaussians + Composite Dynamic Gaussian Graphs | ✅ | △ | 서라운드뷰, LiDAR prior |
| **PVG** (Chen et al.) | 2024/ICLR | 3DGS (explicit) | 주기적 진동 기반 시간적 역학을 도입한 3DGS 확장으로 다양한 객체와 요소를 통합적으로 표현 | ❌ | ✅ | 통합 표현, 수명 메커니즘 |
| **Street Gaussians** (Yan et al.) | 2024/ECCV | 3DGS (explicit) | 포즈 최적화 + 4D SH | ✅ (tracker) | ✅ (135 FPS) | 고속 렌더링, 장면 편집 |
| **S³Gaussian** (Huang et al.) | 2024 | 3DGS (explicit) | 자기지도 | ❌ | ✅ | 자기지도 동적 분리 |
| **DeSiRe-GS** (Peng et al.) | 2025/CVPR | 3DGS (explicit) | 자기지도 분해 — 동적 영역의 3DGS 재구성이 흐릿하다는 관찰에 기반 | ❌ | ✅ | 표면 재구성 + 분해 |
| **OmniRe** (Chen et al.) | 2025/ICLR | 3DGS (explicit) | NSG에 증분적 재구성을 결합하고 보행자와 자전거 이용자를 위한 변형 및 SMPL 노드를 사용하여 모델링을 강화 | ✅ | ✅ | 비강체 객체 포함 |
| **Bézier GS** (2025) | 2025 | 3DGS (explicit) | 학습 가능한 Bézier 곡선 궤적 모델링으로 객체 어노테이션 정밀도 의존성을 제거 | ❌ | ✅ | 연속 궤적, 자동 포즈 보정 |

### 주요 비교 분석 요약

**NeRF → 3DGS 패러다임 전환**: Street Gaussians는 동적 도시 장면 재구성에서 NeRF 기반 implicit 표현의 한계(느린 학습·렌더링)를 3DGS 기반 explicit 표현으로 극복한 선구적 연구입니다.

**어노테이션 의존성 추세**: 초기 방법(NSG, MARS, Street Gaussians)은 3D 바운딩 박스에 의존하였으나, NSG 기반 방법들의 보정된 3D 바운딩 박스에 대한 과도한 의존성이 강건성을 제한한다는 인식이 확산되면서, 최근 연구(PVG, DeSiRe-GS, Bézier GS)는 이를 제거하는 방향으로 발전하고 있습니다.

**비강체 객체로의 확장**: Street Gaussians의 강체 객체 한계를 넘어 OmniRe 등은 보행자와 자전거 이용자까지 모델링 범위를 확장했습니다.

---

## 참고자료 (References)

1. Yan, Y., Lin, H., Zhou, C., Wang, W., Sun, H., Zhan, K., Lang, X., Zhou, X., Peng, S. "Street Gaussians: Modeling Dynamic Urban Scenes with Gaussian Splatting." **ECCV 2024**. [arXiv:2401.01339](https://arxiv.org/abs/2401.01339)
2. GitHub Repository: [zju3dv/street_gaussians](https://github.com/zju3dv/street_gaussians)
3. ECCV 2024 Proceedings (Springer): [Lecture Notes in Computer Science, vol 15131](https://link.springer.com/chapter/10.1007/978-3-031-73464-9_10)
4. ECVA Open Access PDF: [ecva.net/papers/eccv_2024/papers_ECCV/papers/09243.pdf](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09243.pdf)
5. Project Page: [zju3dv.github.io/street_gaussians](https://zju3dv.github.io/street_gaussians/)
6. Chen, Y. et al. "Periodic Vibration Gaussian: Dynamic Urban Scene Reconstruction and Real-time Rendering." **ICLR 2024**. [arXiv:2311.18561](https://arxiv.org/html/2311.18561)
7. Zhou, X. et al. "DrivingGaussian: Composite Gaussian Splatting for Surrounding Dynamic Autonomous Driving Scenes." **CVPR 2024**. [arXiv:2312.07920](https://www.emergentmind.com/papers/2312.07920)
8. Peng et al. "DeSiRe-GS: 4D Street Gaussians for Static-Dynamic Decomposition and Surface Reconstruction for Urban Driving Scenes." **CVPR 2025**. [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2025/papers/Peng_DeSiRe-GS_4D_Street_Gaussians_for_Static-Dynamic_Decomposition_and_Surface_Reconstruction_CVPR_2025_paper.pdf)
9. "Street Gaussians without 3D Object Tracker." [arXiv:2412.05548](https://arxiv.org/html/2412.05548)
10. "Bézier Curve Gaussian Splatting for Dynamic Urban Scene Reconstruction." [arXiv:2506.22099](https://arxiv.org/html/2506.22099v2)
11. AI Research Paper Details: [aimodels.fyi](https://www.aimodels.fyi/papers/arxiv/street-gaussians-modeling-dynamic-urban-scenes-gaussian)
12. "VDEGaussian: Video Diffusion Enhanced 4D Gaussian Splatting for Dynamic Urban Scenes Modeling." [arXiv:2508.02129](https://arxiv.org/html/2508.02129v1)

> **주의**: 수식의 구체적 변수 기호와 세부 형태는 논문 원문과 관련 공개 자료를 기반으로 작성하였으며, 논문 PDF에서 직접 확인할 수 없는 일부 세부 수식 표기는 3D Gaussian Splatting의 표준 공식과 논문의 텍스트 기술을 바탕으로 재구성한 것입니다. 정확한 수식은 원문을 직접 확인하시기를 권장합니다.
