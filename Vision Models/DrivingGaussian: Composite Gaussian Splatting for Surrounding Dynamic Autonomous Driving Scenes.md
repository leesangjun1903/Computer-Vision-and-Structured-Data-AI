# DrivingGaussian: Composite Gaussian Splatting for Surrounding Dynamic Autonomous Driving Scenes

---

## 1. 핵심 주장 및 주요 기여 요약

DrivingGaussian은 주변 동적 자율주행 장면을 위한 효율적이고 효과적인 프레임워크입니다. 이 논문은 복잡한 주행 장면을 다중 센서의 순차적 데이터를 사용하여 계층적으로(hierarchically) 모델링하는 것을 핵심 아이디어로 제시합니다.

**주요 기여(Contributions):**

1. DrivingGaussian은 Composite Gaussian Splatting에 기반한 대규모 동적 주행 장면의 최초 표현 및 모델링 프레임워크입니다.
2. Incremental Static 3D Gaussians와 Composite Dynamic Gaussian Graphs라는 두 가지 새로운 모듈이 도입되었으며, 전자는 정적 배경을 점진적으로 재구성하고 후자는 다수의 동적 객체를 Gaussian 그래프로 모델링합니다.
3. LiDAR prior를 Gaussian Splatting에 활용하여 더 높은 디테일로 장면을 재구성하고 파노라마 일관성을 유지합니다.
4. 동적 장면 구성 및 코너 케이스 시뮬레이션이 가능하여, 자율주행 시스템의 안전성과 견고성 검증을 촉진합니다.

> **발표:** CVPR 2024, 저자: Xiaoyu Zhou, Zhiwei Lin, Xiaojun Shan, Yongtao Wang, Deqing Sun, Ming-Hsuan Yang (Peking University, Google Research, UC Merced)

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 3D Gaussian Splatting(3D-GS)은 고정된 Gaussian과 제한된 표현 용량으로 인해 대규모 동적 주행 장면 모델링에 상당한 어려움을 겪습니다. 일부 연구들이 3D-GS를 동적 장면으로 확장했으나, 이들은 개별 동적 객체에만 초점을 맞추며 정적-동적 혼합 영역과 고속 이동하는 다수의 객체를 포함하는 복잡한 주행 장면을 처리하지 못합니다.

차량 장착 센서의 희소한 데이터로부터 복잡한 3D 장면을 재구성하는 것은 특히 에고 차량이 고속으로 움직일 때 매우 어렵습니다. 기존 NeRF 및 3D Gaussian Splatting 방법들은 계산 집약성, 밀집 뷰에 대한 의존성, 다수의 고속 이동 객체에 대한 제한된 처리 능력 때문에 대규모 동적 주행 장면에서 어려움을 겪습니다.

### 2.2 제안하는 방법 (Method)

#### A. Composite Gaussian Splatting 개요

DrivingGaussian은 Composite Gaussian Splatting을 채택하여 전체 장면을 정적 배경과 동적 객체로 분해하고, 각 부분을 개별적으로 재구성합니다.

#### B. 3D Gaussian Splatting 기본 수식

각 3D Gaussian은 위치 $\mu$, 공분산 행렬 $\Sigma$, 불투명도 $\alpha$, 색상 정보(Spherical Harmonics) $c$로 정의됩니다:

$$G(x) = e^{-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu)}$$

공분산 행렬은 스케일링 행렬 $S$와 회전 행렬 $R$로 분해됩니다:

$$\Sigma = R S S^T R^T$$

렌더링 시 각 픽셀의 색상은 $\alpha$-blending으로 계산됩니다:

$$C = \sum_{i \in N} c_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)$$

여기서 $c_i$는 $i$번째 Gaussian의 색상, $\alpha_i$는 해당 Gaussian의 불투명도입니다.

#### C. Incremental Static 3D Gaussians (IS3G)

복잡한 장면에서 움직이는 객체들에 대해, 먼저 전체 장면의 정적 배경을 Incremental Static 3D Gaussian으로 순차적이고 점진적으로 모델링합니다.

다중 카메라 뷰에서 순차적으로 정적 배경을 구성하는 과정은 다음과 같이 형식화됩니다. 시간 $t$에서의 정적 Gaussian 집합은:

$$\mathcal{G}_{static}^{t} = \mathcal{G}_{static}^{t-1} \cup \Delta \mathcal{G}_{static}^{t}$$

여기서 $\Delta \mathcal{G}_{static}^{t}$는 시간 $t$에서 새로 추가되는 Gaussian들이며, 이전 프레임에서 보이지 않았던 영역을 커버합니다. 이 과정에서 동적 객체가 차지하는 영역은 마스킹 처리되어 제외됩니다.

#### D. Composite Dynamic Gaussian Graph (CDGG)

Composite Dynamic Gaussian Graph를 활용하여 다수의 이동 객체를 처리하며, 각 객체를 개별적으로 재구성하고 장면 내에서 정확한 위치와 가려짐(occlusion) 관계를 복원합니다.

동적 객체 $k$에 대해 Gaussian 그래프 노드는 다음과 같이 정의됩니다:

$$\mathcal{G}_{dynamic}^{k} = \{(\mu_i^k, \Sigma_i^k, \alpha_i^k, c_i^k)\}_{i=1}^{N_k}$$

시간 $t$에서 객체 $k$의 변환은 rigid body transformation으로 표현됩니다:

$$\mu_i^{k,t} = R_t^k \mu_i^{k,0} + T_t^k$$

여기서 $R_t^k$와 $T_t^k$는 각각 시간 $t$에서 객체 $k$의 회전 행렬과 이동 벡터입니다.

Composite Dynamic Gaussian Graph는 각 이동 객체를 개별적으로 재구성하고 Gaussian 그래프를 기반으로 정적 배경에 동적으로 통합합니다. 이를 통한 전역 렌더링은 정적 배경과 동적 객체를 포함한 실제 세계의 가려짐 관계를 포착합니다.

최종 합성 렌더링은:

$$C_{final} = \text{Render}(\mathcal{G}_{static} \cup \bigcup_{k=1}^{K} \mathcal{T}_t^k(\mathcal{G}_{dynamic}^{k}))$$

여기서 $\mathcal{T}_t^k$는 시간 $t$에서 객체 $k$에 적용되는 변환 함수, $K$는 동적 객체의 수입니다.

#### E. LiDAR Prior를 활용한 초기화

LiDAR prior를 이용한 초기화가 왜곡이나 흐림 없이 더 높은 품질의 기하학적 구조를 생성합니다.

LiDAR 깊이 손실 함수는 다음과 같이 정의됩니다:

$$\mathcal{L}_{LiDAR} = \frac{1}{|\mathcal{P}|} \sum_{p \in \mathcal{P}} \| D_{render}(p) - D_{LiDAR}(p) \|_1$$

여기서 $D_{render}(p)$는 렌더링된 깊이, $D_{LiDAR}(p)$는 LiDAR로부터 얻은 깊이 값, $\mathcal{P}$는 유효한 LiDAR 포인트 집합입니다.

#### F. 손실 함수 (Loss Functions)

전체 학습 손실은 여러 구성 요소의 가중 합으로 구성됩니다:

$$\mathcal{L}_{total} = \mathcal{L}_1 + \lambda_1 \mathcal{L}_{TSSIM} + \lambda_2 \mathcal{L}_{Robust} + \lambda_3 \mathcal{L}_{LiDAR}$$

- $\mathcal{L}_1$: L1 photometric loss
- $\mathcal{L}_{TSSIM}$: Temporal SSIM loss — 시간적 일관성을 보장하기 위한 구조적 유사성 손실
- $\mathcal{L}_{Robust}$: Robust loss — 아티팩트 제거 및 텍스처 디테일 향상

$\mathcal{L}\_{Robust}$는 렌더링 품질을 현저히 향상시키고 텍스처 디테일을 강화하며 아티팩트를 제거합니다. $\mathcal{L}_{LiDAR}$는 LiDAR prior의 도움을 받아 Gaussian이 더 나은 기하학적 사전 정보를 획득하도록 합니다.

### 2.3 모델 구조

DrivingGaussian은 멀티 카메라 이미지와 LiDAR를 포함하는 다중 센서로부터의 순차적 데이터를 입력으로 받습니다. 대규모 동적 주행 장면을 표현하기 위해 두 가지 구성 요소로 이루어진 Composite Gaussian Splatting을 제안합니다.

```
┌──────────────────────────────────────────────────────┐
│                DrivingGaussian Pipeline               │
├──────────────────────────────────────────────────────┤
│                                                       │
│  Multi-Camera Images + LiDAR Point Clouds             │
│          │                                            │
│          ▼                                            │
│  ┌─────────────────────────────────────────┐          │
│  │  LiDAR Prior Initialization             │          │
│  │  (3D Gaussian 위치 초기화)                │          │
│  └─────────────┬───────────────────────────┘          │
│                │                                      │
│       ┌────────┴──────────┐                           │
│       ▼                   ▼                           │
│  ┌──────────┐     ┌───────────────┐                   │
│  │  IS3G    │     │   CDGG        │                   │
│  │ (정적배경) │     │ (동적 객체)    │                   │
│  └────┬─────┘     └─────┬─────────┘                   │
│       │                 │                             │
│       └────────┬────────┘                             │
│                ▼                                      │
│  ┌─────────────────────────────────┐                  │
│  │  Composite Rendering            │                  │
│  │  (Global Gaussian Splatting)    │                  │
│  └─────────────────────────────────┘                  │
│                │                                      │
│                ▼                                      │
│       Photorealistic Output                           │
│  (Novel View Synthesis / Simulation)                  │
└──────────────────────────────────────────────────────┘
```

Composite Gaussian Splatting의 첫 번째 부분은 광범위한 정적 배경을 점진적으로 재구성하며, 두 번째 부분은 Gaussian 그래프로 다수의 동적 객체를 구성하여 장면에 동적으로 통합합니다.

### 2.4 성능 향상

#### nuScenes 데이터셋 결과

LiDAR 초기화 DrivingGaussian(Ours-L)은 PSNR 28.74, SSIM 0.865, LPIPS 0.237을 달성하여, 차상위 경쟁 모델 S-NeRF(PSNR 25.43, SSIM 0.730, LPIPS 0.302)보다 현저히 우수합니다.

SfM 초기화 버전(Ours-S)도 PSNR 28.36, SSIM 0.851, LPIPS 0.256으로 우수한 성능을 보이며, LiDAR prior는 기하학적 정확도를 더욱 향상시킵니다.

| 방법 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| Mip-NeRF360 | 24.41 | 0.691 | 0.390 |
| S-NeRF | 25.43 | 0.730 | 0.302 |
| EmerNeRF | 25.76 | 0.741 | 0.296 |
| **Ours-S (SfM)** | **28.36** | **0.851** | **0.256** |
| **Ours-L (LiDAR)** | **28.74** | **0.865** | **0.237** |

#### 렌더링 속도

DrivingGaussian은 렌더링 품질과 속도 간의 탁월한 균형을 달성합니다. LiDAR 초기화 버전의 프레임당 렌더링 시간은 약 0.963초로, 대부분의 NeRF 기반 방법(예: Mip-NeRF360 11.86초, EmerNeRF 21.91초)보다 현저히 빠릅니다.

#### KITTI-360 데이터셋 결과

DrivingGaussian은 단안(monocular) 시나리오에 특화 설계되지 않았음에도 불구하고, 단안 주행 장면 표현에서 우수한 적응성과 견고성을 보이며 기존 SOTA 접근법을 능가합니다.

### 2.5 한계점

논문과 관련 문헌에서 식별되는 주요 한계는 다음과 같습니다:

1. **3D Bounding Box 의존성:** DrivingGaussian과 Street Gaussians는 정적-동적 분해를 위해 명시적 3D Bounding Box에 의존하며, 이는 분해 문제를 상당히 단순화하지만 annotation 비용이 발생합니다.

2. **LiDAR 커버리지 한계:** Gaussian 중심을 LiDAR 포인트 클라우드에 정렬하도록 하는 추가 손실을 도입하는 접근법은 LiDAR 포인트 클라우드가 카메라 가시 영역을 완전히 커버한다고 가정하며, LiDAR 측정이 없는 영역에서 이미지 품질이 저하됩니다.

3. **장면별 최적화(Per-scene Optimization):** Street Gaussian과 같은 전통적 4DGS 방법들은 장면별 최적화에 의존합니다. 이러한 비데이터 주도 패러다임은 장면 간 공유된 구조적 지식을 활용하지 못하며 모든 새 환경에서 상당한 계산 오버헤드를 발생시켜 확장성이 제한됩니다.

4. **보행자/비차량 객체 처리 미흡:** NeRF 기반 자율주행 접근법과 유사하게, 자율주행 장면에서 보행자와 기타 비차량 동적 행위자(actors)를 간과하는 경우가 많습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재의 일반화 능력

광범위한 실험을 통해 공공 자율주행 데이터셋에서 SOTA 성능이 입증되었으며, LiDAR prior 없이도 유망한 성능을 보여 대규모 동적 장면 재구성에서의 다재다능함(versatility)을 입증합니다.

실험 결과는 DrivingGaussian이 LiDAR prior 없이도 잘 수행되어 다양한 초기화 방법에 대한 강건한 견고성을 보여줍니다.

### 3.2 일반화 성능 향상을 위한 핵심 방향

#### (1) Feed-forward 모델로의 확장

현재의 재구성 알고리즘은 새 장면을 구축하기 위해 시간이 많이 소요되는 반복 최적화가 필요합니다. 일부 연구에서는 새로운 장면에 일반화되는 방식으로 전체 재구성 작업을 직접 학습하는 신경망을 제안하여 이를 크게 가속화하고 있습니다.

$$\hat{\mathcal{G}} = f_\theta(I_1, I_2, ..., I_N; P_1, P_2, ..., P_N)$$

여기서 $f_\theta$는 학습된 feed-forward 네트워크, $I_i$는 입력 이미지, $P_i$는 카메라 포즈입니다. 이 방식은 장면별 최적화 없이 단일 forward pass로 Gaussian 파라미터를 예측합니다.

#### (2) Self-supervised 분해 기법

3D annotation 없이도, PVG와 S3Gaussian과 같은 최근 self-supervised 방법들이 분해를 시도하고 있습니다. DeSiRe-GS는 3DGS에서 재구성된 동적 영역이 흐릿하다는 관찰에 기반한 효과적인 self-supervised 분해를 달성합니다.

#### (3) 다중 센서 융합 강화

LIV-GaussMap과 같은 다중 모달 센서 융합 시스템은 LiDAR, IMU, 비전 데이터를 통합하여 장면 기하학 정보를 캡처 및 복원하고, 정확한 맵 구조를 구축합니다.

#### (4) Pose-free 접근법

기존 방법들이 사전 계산된 포즈와 SfM 알고리즘 또는 고가의 센서에 크게 의존하는 한계를 극복하기 위해, VDG와 같은 연구는 self-supervised VO를 pose-free 동적 Gaussian 방법에 통합합니다.

---

## 4. 연구 영향 및 향후 연구 시 고려할 점

### 4.1 연구 영향

대규모 동적 장면의 표현과 모델링은 BEV 인식, 3D 탐지, 모션 플래닝 등 일련의 자율주행 태스크에 기여하는 3D 장면 이해의 기초입니다. 주행 장면에 대한 뷰 합성 및 제어 가능한 시뮬레이션은 코너 케이스와 안전-중요 상황의 생성을 가능하게 하여, 자율주행 시스템의 안전성 검증과 강화를 더 낮은 비용으로 수행할 수 있게 합니다.

DrivingGaussian은 다음과 같은 방향으로 후속 연구에 직접적 영향을 미치고 있습니다:

- **DrivingGaussian++:** DrivingGaussian++는 현실적인 재구성 및 제어 가능한 편집을 위한 프레임워크로, Incremental 3D Gaussians로 정적 배경을 모델링하고, Composite Dynamic Gaussian Graph로 이동 객체를 재구성합니다. LiDAR prior를 통합하여 상세하고 일관된 장면 재구성을 달성하며, 학습이 필요 없는(training-free) 제어 가능 편집(텍스처 수정, 날씨 시뮬레이션, 객체 조작)을 지원합니다.

- **코너 케이스 시뮬레이션:** 실제 주행 장면에서 코너 케이스를 시뮬레이션하는 효과가 입증되었으며, 재구성된 Gaussian 필드에 임의의 동적 객체를 삽입할 수 있습니다.

### 4.2 향후 연구 시 고려할 점

| 고려 사항 | 설명 |
|-----------|------|
| **확장성(Scalability)** | 도시 규모의 광범위한 장면 처리를 위한 효율적인 메모리 관리 및 분산 학습 전략 |
| **실시간 렌더링** | 폐루프(closed-loop) 시뮬레이션을 위한 실시간 렌더링 달성 |
| **Annotation 의존도 감소** | 3D Bounding Box 없는 self-supervised 정적/동적 분해 |
| **일반화(Generalization)** | 다양한 도메인(날씨, 조명, 도로 유형)에 걸친 강건한 일반화 |
| **비강체(Non-rigid) 모델링** | 보행자, 자전거 등 비강체 동적 객체의 정밀한 모델링 |
| **시간적 일관성** | 장시간 시퀀스에서의 시간적 일관성 유지 |
| **편집 가능성** | 장면 수준의 의미적 편집 및 제어 가능한 합성 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 방법론 | 주요 특징 | DrivingGaussian 대비 |
|------|------|--------|-----------|---------------------|
| **NeRF (Mildenhall et al.)** | 2020 | Implicit Neural Radiance Field | 암묵적 볼륨 렌더링 기반 NVS | 렌더링 속도 10~20배 느림 |
| **3D-GS (Kerbl et al.)** | 2023 | Explicit 3D Gaussians | 실시간 렌더링, 정적 장면 | 동적 장면 미지원 |
| **EmerNeRF (Yang et al.)** | 2024 | Self-supervised NeRF | 시공간 장면 분해 | PSNR 25.76 vs. 28.74 |
| **Street Gaussians (Yan et al.)** | 2024 | 4D Spherical Harmonics | 동적 전경에 4D 구면 조화 모델 사용, 133 FPS에서 고품질 렌더링 생성 | 유사한 계층적 접근, 더 빠른 렌더링 |
| **PVG (Chen et al.)** | 2023 | Periodic Vibration Gaussian | 주기적 진동 기반 시간 역학을 효율적 정적 3DGS에 도입하여 동적/정적 요소를 통합 | Bounding Box 불필요, 통합 표현 |
| **4D-GS (Wu et al.)** | 2024 | 4D Gaussian Splatting | 정적 3D-GS 대신 동적 장면의 총체적 표현으로 4D Gaussian Splatting을 제안, 800×800 해상도에서 82 FPS 실시간 렌더링 | 시간 차원 명시적 모델링 |
| **OmniRe (Chen et al.)** | 2024 | Gaussian Scene Graphs | Gaussian Scene Graphs로 도시 동적 장면을 모델링하며 다양한 유형의 노드(하늘, 배경, 강체/비강체 이동 객체)를 처리하고, SMPL 모델로 인체를 파라미터화 | 비강체 객체 모델링 우위 |
| **DeSiRe-GS (Peng et al.)** | 2025 | 4D Street Gaussians | Bounding Box 같은 추가 3D annotation 없이 self-supervised 정적-동적 분해 및 고충실도 표면 재구성 | Self-supervised, annotation 불필요 |
| **AD-GS (Xu et al.)** | 2025 | B-Spline Gaussian | 객체 Gaussian의 파라미터를 학습 가능한 B-spline 곡선과 삼각함수로 변형하여 지역 및 전역 피팅 달성, 양방향 시간 가시성 마스크로 객체의 갑작스러운 출현/소실 처리 | Self-supervised, 정밀한 모션 모델링 |
| **VDG (Li et al.)** | 2024 | Pose-free Dynamic GS | self-supervised VO를 pose-free 동적 Gaussian 방법에 통합하여 포즈 및 깊이 초기화와 정적-동적 분해를 향상 | 사전 포즈 불필요 |
| **ReconDrive** | 2025 | Feed-forward 4DGS | 3D 기초 모델 VGGT를 확장한 feed-forward 프레임워크로 빠르고 고충실도 4DGS 생성 | 장면별 최적화 불필요, 일반화 가능 |
| **LiHi-GS** | 2025 | LiDAR-supervised GS | 명시적 LiDAR 센서 모델링을 포함한 GS 방법을 제안하여 고속도로 시나리오의 한계를 극복 | 보다 정밀한 LiDAR 통합 |

### 기술 발전 추이 요약

$$\text{NeRF (2020)} \rightarrow \text{3D-GS (2023)} \rightarrow \text{DrivingGaussian (2024)} \rightarrow \text{Feed-forward 4DGS (2025)}$$

3D Gaussian Splatting은 NeRF로 대표되는 암묵적 표현을 대체하는 최신 명시적 복사 필드 기술로, 3D 장면 재구성에서 가장 활발한 연구 방향이 되었습니다.

---

## 참고 자료 및 출처

1. **Zhou, X., Lin, Z., Shan, X., Wang, Y., Sun, D., & Yang, M.-H.** (2024). "DrivingGaussian: Composite Gaussian Splatting for Surrounding Dynamic Autonomous Driving Scenes." *CVPR 2024*, pp. 21634–21643. [arXiv:2312.07920](https://arxiv.org/abs/2312.07920)
2. **CVPR 2024 Open Access Repository** — [thecvf.com](https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_DrivingGaussian_Composite_Gaussian_Splatting_for_Surrounding_Dynamic_Autonomous_Driving_Scenes_CVPR_2024_paper.html)
3. **DrivingGaussian GitHub Repository** — [github.com/VDIGPKU/DrivingGaussian](https://github.com/VDIGPKU/DrivingGaussian)
4. **DrivingGaussian Project Page (PKU)** — [pkuvdig.github.io/DrivingGaussian](https://pkuvdig.github.io/DrivingGaussian/)
5. **Springer AI Review** — "Scene reconstruction techniques for autonomous driving: a review of 3D Gaussian splatting" (2024). [Springer Nature Link](https://link.springer.com/article/10.1007/s10462-024-10955-4)
6. **Liner Quick Review** — DrivingGaussian 분석. [liner.com](https://liner.com/review/drivinggaussian-composite-gaussian-splatting-for-surrounding-dynamic-autonomous-driving-scenes)
7. **DrivingGaussian++** — Xiong et al. (2025). [arXiv:2508.20965](https://arxiv.org/abs/2508.20965)
8. **DeSiRe-GS** — Peng et al. (2025). [arXiv:2411.11921](https://arxiv.org/html/2411.11921)
9. **AD-GS** — Xu et al. (2025). *ICCV 2025*. [thecvf.com](https://openaccess.thecvf.com/content/ICCV2025/papers/Xu_AD-GS_Object-Aware_B-Spline_Gaussian_Splatting_for_Self-Supervised_Autonomous_Driving_ICCV_2025_paper.pdf)
10. **VDG: Vision-Only Dynamic Gaussian** — Li et al. (2024). [arXiv:2406.18198](https://arxiv.org/html/2406.18198v1)
11. **ReconDrive** — (2025). [arXiv:2603.07552](https://arxiv.org/html/2603.07552)
12. **LiHi-GS** — (2025). [arXiv:2412.15447](https://arxiv.org/html/2412.15447v3)

---

> **정확도 관련 참고사항:** 본 분석의 수식 중 일부(특히 IS3G의 점진적 축적 수식, 동적 그래프 합성 렌더링 수식)는 논문의 핵심 원리를 기반으로 구성한 일반적 형식입니다. 원본 논문의 정확한 수식 표기를 확인하시려면 [CVPR 2024 PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_DrivingGaussian_Composite_Gaussian_Splatting_for_Surrounding_Dynamic_Autonomous_Driving_Scenes_CVPR_2024_paper.pdf)를 직접 참조하시기 바랍니다.
