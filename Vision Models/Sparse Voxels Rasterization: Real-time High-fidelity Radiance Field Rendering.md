
# Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering

> **논문 정보**: Cheng Sun, Jaesung Choe, Charles Loop, Wei-Chiu Ma, Yu-Chiang Frank Wang
> **발표**: CVPR 2025 (arXiv: 2412.04459, 2024년 12월 5일 제출)
> **소속**: NVIDIA, Cornell University, National Taiwan University
> **공식 코드**: [github.com/NVlabs/svraster](https://github.com/NVlabs/svraster)
> **프로젝트 페이지**: [svraster.github.io](https://svraster.github.io)
> **CVPR Open Access**: [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_Sparse_Voxels_Rasterization_Real-time_High-fidelity_Radiance_Field_Rendering_CVPR_2025_paper.html)

---

## 1. 📌 핵심 주장 및 주요 기여 요약

이 논문은 신경망(Neural Network)이나 3D Gaussian 없이, 적응형 희소 복셀(Adaptive Sparse Voxels)에 대한 래스터화(Rasterization) 프로세스를 통합한 효율적인 Radiance Field 렌더링 알고리즘을 제안합니다.

### 🔑 두 가지 핵심 기여

| 기여 | 내용 |
|------|------|
| **① 적응형 희소 복셀 할당** | 장면의 다양한 Level of Detail에 복셀을 적응적으로 할당 |
| **② 커스텀 래스터라이저** | 올바른 깊이 순서 렌더링을 보장하는 Morton Order 기반 래스터라이저 |

첫 번째 기여는 장면 내 서로 다른 레벨 오브 디테일(LoD)에 희소 복셀을 적응적이고 명시적으로 할당하여, $65536^3$ 그리드 해상도로 장면 세부 사항을 충실하게 재현하면서도 높은 렌더링 프레임 속도를 달성하는 것입니다.

두 번째 기여는 효율적인 적응형 희소 복셀 렌더링을 위한 커스텀 래스터라이저로, 광선 방향 의존 Morton Order를 사용하여 올바른 깊이 순서로 복셀을 렌더링함으로써 Gaussian Splatting에서 흔히 발생하는 popping artifact를 방지합니다.

이 방법은 이전의 neural-free 복셀 모델 대비 **4dB 이상의 PSNR 향상**과 **10배 이상의 FPS 속도 향상**을 달성하여, state-of-the-art에 비견되는 novel-view synthesis 결과를 보입니다.

---

## 2. 🔍 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

Gaussian Splatting은 novel view synthesis의 가장 유망한 솔루션 중 하나로 부상했으며, 뛰어난 렌더링 속도와 장면의 세밀한 디테일을 캡처하는 능력 덕분에 여러 커뮤니티에서 폭넓은 주목을 받고 있습니다. 그러나 기존 방식들은 다음과 같은 문제를 안고 있었습니다:

- **Gaussian Splatting의 Popping Artifact**: 3DGS는 Primitive 중심 기반 정렬이 부정확하여, 시점 변화 시 갑작스러운 화면 깜빡임(popping)이 발생
- **신경망 의존성**: 기존 NeRF 계열은 MLP 등 신경망이 필수적이어서 실시간 렌더링이 어려움
- **밀도 그리드의 확장성 한계**: 이전 그리드 기반 방법들은 여전히 어떤 형태로든 dense 3D 그리드를 사용하는데, 예를 들어 free-space skipping을 위한 dense occupancy grid나 희소 복셀 조회를 지원하기 위한 dense pointer grid를 사용합니다.

복셀 표현은 현대 그래픽 엔진과 본질적으로 호환 가능하며 효율적으로 래스터화할 수 있습니다. 또한 이전 연구에서 볼류메트릭 광선 캐스팅을 통해 장면 볼륨 밀도를 모델링하는 능력이 입증되었습니다. 이는 복셀이 래스터화와 볼류메트릭 표현 사이의 완벽한 가교가 될 수 있음을 시사합니다.

---

### 2.2 제안하는 방법 및 수식

#### 📐 볼류메트릭 렌더링 기본 수식

SVRaster는 고전적인 볼류메트릭 렌더링 방정식을 따릅니다. 픽셀 색상 $\hat{C}$는 다음과 같이 계산됩니다:

$$
\hat{C} = \sum_{i=1}^{N} T_i \cdot \alpha_i \cdot c_i
$$

여기서:
- $T_i = \prod_{j=1}^{i-1}(1 - \alpha_j)$: 누적 투과율(transmittance)
- $\alpha_i$: $i$번째 복셀의 불투명도(opacity)
- $c_i$: 복셀의 색상(view-dependent)

#### 📐 복셀 내부 Alpha 계산

밀도 필드는 각 복셀의 코너 그리드 포인트 값으로 모델링되는 삼선형(trilinear) 변화를 가지며, 그리드 포인트 밀도는 인접 복셀 간에 공유됩니다. 렌더링 시 픽셀 광선에 대한 복셀의 알파 기여 값을 계산하기 위해 광선-복셀 교차 세그먼트 내부에서 $K$개의 점을 균등하게 샘플링하여 볼류메트릭 통합을 수행합니다.

복셀 $v$에 대한 alpha 값은:

$$
\alpha_v = 1 - \exp\left(-\sum_{k=1}^{K} \sigma(\mathbf{x}_k) \cdot \Delta t\right)
$$

여기서 $\sigma(\mathbf{x}_k)$는 삼선형 보간된 밀도, $\Delta t$는 샘플 간격입니다.

#### 📐 구면 조화 함수(SH) 기반 색상 표현

각 복셀은 뷰 의존적 외관을 위한 고유한 구면 조화 계수(Spherical Harmonic coefficient)를 가집니다. 색상 필드는 렌더링 효율성을 위해 복셀 내부에서 상수로 근사됩니다.

뷰 의존적 색상:

$$
c_i(\mathbf{d}) = \sum_{l=0}^{L} \sum_{m=-l}^{l} k_{l,m}^{(i)} \cdot Y_l^m(\mathbf{d})
$$

여기서 $Y_l^m$은 구면 조화 기저 함수, $k_{l,m}^{(i)}$는 $i$번째 복셀의 SH 계수, $\mathbf{d}$는 시점 방향입니다.

#### 📐 Morton Order 기반 올바른 정렬

SVRaster는 광선 방향 의존 Morton Order로 복셀을 정렬하여 올바른 프리미티브 블렌딩 순서를 수학적으로 보장합니다. 3DGS처럼 프리미티브 중심으로 정렬하면 부정확한 렌더링이 발생할 수 있습니다.

Morton Code 계산의 핵심은 복셀의 Octree 좌표 $(i, j, k)$ 와 뷰 방향 $\mathbf{d} = (d_x, d_y, d_z)$에 따라 좌표 비트를 반전(flip)하는 방식입니다:

```math
\tilde{i} = \begin{cases} 2^L - 1 - i & \text{if } d_x < 0 \\ i & \text{if } d_x \geq 0 \end{cases}, \quad \tilde{j}, \tilde{k} \text{ similarly}
```

```math
\text{MortonCode}(\tilde{i}, \tilde{j}, \tilde{k}) = \text{interleave\_bits}(\tilde{i}, \tilde{j}, \tilde{k})
```

이를 통해 복셀 간의 올바른 front-to-back 정렬이 수학적으로 보장됩니다.

---

### 2.3 모델 구조

SVRaster의 장면 표현은 프리미티브(primitive) 컴포넌트와 볼류메트릭(volumetric) 컴포넌트의 하이브리드 모델입니다.

```
┌──────────────────────────────────────────────────────────┐
│                      SVRaster 구조                        │
│                                                          │
│   입력: 멀티뷰 이미지 + 카메라 파라미터 (COLMAP 등)         │
│                       │                                  │
│          ┌────────────▼────────────┐                     │
│          │  Primitive Component    │                     │
│          │  - Octree Layout        │                     │
│          │  - 다중 LoD 복셀 할당    │                     │
│          │  - 1D Array 저장        │                     │
│          └────────────┬────────────┘                     │
│                       │                                  │
│          ┌────────────▼────────────┐                     │
│          │  Volumetric Component   │                     │
│          │  - Trilinear Density    │                     │
│          │  - SH Color Field       │                     │
│          │  - K-point Sampling     │                     │
│          └────────────┬────────────┘                     │
│                       │                                  │
│          ┌────────────▼────────────┐                     │
│          │  Direction-Dependent    │                     │
│          │  Morton Order Sorter    │                     │
│          └────────────┬────────────┘                     │
│                       │                                  │
│          ┌────────────▼────────────┐                     │
│          │  CUDA 래스터라이저       │                     │
│          │  (front-to-back blend)  │                     │
│          └────────────┬────────────┘                     │
│                       │                                  │
│               렌더링된 픽셀 색상                           │
└──────────────────────────────────────────────────────────┘
```

#### 핵심 구성 요소 상세

**① Octree 기반 복셀 할당:**
SVRaster는 희소 복셀 표현으로 3D 장면을 구성하며, Octree 공간 분할 규칙에 따라 복셀을 할당합니다. 다만 부모-자식 포인터가 있는 전통적인 Octree 구조를 복제하지는 않으며, Octree 리프 노드의 복셀만 유지하고 조상 노드는 없습니다.

**② 1D 배열 저장:**
서로 다른 크기의 모든 비어있지 않은 복셀을 단순히 1D 배열에 저장합니다. 래스터라이저가 렌더링 시 모든 것이 올바르게 정렬되도록 보장합니다.

**③ 적응형 복셀 세분화 (Adaptive Subdivision):**
복셀의 샘플링 비율(sampling rate) $\mathbf{v}_{\text{rate}}$를 기반으로 세분화를 결정합니다:

$$
\mathbf{v}_{\text{rate}} = \max_{c \in \text{cameras}} \frac{W}{d_c \cdot s_v}
$$

여기서 $W$는 이미지 너비, $d_c$는 카메라까지의 거리, $s_v$는 복셀 크기입니다.

**④ 정규화 손실:**
더 나은 기하학적 구조를 위해 normal regularization 손실(`--lambda_normal_dmean`, `--lambda_normal_dmed`)을 함께 학습하며, COLMAP의 sparse depth 도 보조 지도 신호로 활용할 수 있습니다.

---

### 2.4 성능 향상

SVRaster는 Mip-NeRF360 데이터셋에서 3D Gaussian Splatting(3DGS)에 필적하는 평균 FPS를 달성하며, 빠른 렌더링 변형의 경우 **258 FPS**에 도달합니다.

NeRF 및 그 변형들과 비교하여 학습 시간이 크게 단축되며, SVRaster의 빠른 렌더링 변형은 **9분** 만에 학습이 완료되어 3DGS(24분)보다 실질적으로 빠릅니다.

시각적 품질 측면에서 SVRaster는 LPIPS 기준(0.210 vs. 0.216)으로 3DGS를 능가하고 Zip-NeRF도 뛰어넘는 결과를 보이며, 시지각적으로 설득력 있고 세밀한 novel view를 생성합니다.

| 지표 | SVRaster | 3DGS | NeRF 계열 |
|------|----------|------|-----------|
| 학습 시간 | **9분 (fast)** | 24분 | 수시간~수일 |
| 렌더링 FPS | **~258 FPS** | 유사 | 느림 |
| LPIPS | **0.210** | 0.216 | - |
| PSNR | 4dB+ ↑ (vs 이전 voxel) | - | - |

또한 TSDF-Fusion 및 Marching Cubes를 통합하여 메쉬 복원에서도 우수한 정확도를 달성합니다.

---

### 2.5 한계점

한 가지 제한 사항은 효율적인 검색을 지원하지 않는다는 점입니다(즉, 특정 포인트가 주어졌을 때 해당 복셀을 찾는 것). 향후 연구에서는 복셀을 정렬하고 이를 위한 이진 탐색(binary search)을 구현해야 합니다.

SVRaster는 3DGS(1.8GB, 0.7GB)와 비교하여 더 높은 메모리 사용량과 모델 크기(GPU 메모리 3.9GB, 모델 크기 1.8GB)를 보입니다. 또한 학습 뷰의 노출 변화와 같은 장면 특성에 성능이 민감하여, Tanks&Temples 같은 어려운 데이터셋에서 밝기 경계 및 floater와 같은 아티팩트가 발생할 수 있습니다.

- **카메라 모델 제한**: 현재 핀홀(pinhole) 카메라 모드만 지원합니다.
- **동적 장면**: 정적 장면에 최적화되어 있어 동적 장면 확장이 필요함
- **대용량 장면 확장**: 균일한 복셀 크기의 희소 복셀만으로는 스케일업이 불가능하여 LoD가 필수적

---

## 3. 🌐 일반화 성능 향상 가능성

SVRaster는 여러 측면에서 일반화 가능성을 열어두고 있습니다.

### 3.1 2D 모달리티 Fusion을 통한 일반화

장면 최적화 완료 후, 2D 비전 파운데이션 피처 또는 의미론적 분할 결과를 복셀에 쉽고 빠르게 퓨전할 수 있습니다. 이 퓨전은 자연스럽게 멀티뷰 간 불일치한 예측을 평활화합니다.

학습된 희소 복셀로 2D 모달리티를 리프팅(lifting)하는 것이 고전적인 Volume Fusion을 통합함으로써 단순하고 효율적입니다. RADIO에서의 비전 파운데이션 모델 피처 필드, Segformer에서의 의미론적 필드, 렌더링된 깊이로부터의 부호화 거리 필드(SDF) 등의 예시가 가능하여 광범위한 응용에 유연하고 적합합니다.

### 3.2 고전적 3D 알고리즘과의 통합

특히 SVRaster의 희소 복셀 표현은 TSDF-Fusion 및 Marching Cubes와 같은 고전적인 3D 알고리즘과 자연스럽게 통합되어 다양한 응용에 유연하게 활용될 수 있습니다.

### 3.3 Depth Supervision을 통한 일반화

더 나은 기하학적 구조를 위해 DepthAnythingV2 상대 깊이 손실 및 MASt3R 메트릭 깊이 손실을 지원합니다. 이러한 단안 깊이 추정 모델과의 결합은 적은 입력 뷰에서도 견고한 기하학 구조 학습을 가능하게 합니다.

### 3.4 ScanNet++ 등 실내 장면 일반화

ScanNet++ 데이터셋을 지원하며, 50개의 실내 장면에 대한 3rd-party 숨겨진 세트 평가에서 공식 벤치마크 결과를 제공합니다.

### 3.5 SVRecon으로의 확장

SVRaster는 희소 복셀을 기본 3D 프리미티브로 사용하는 명시적 신경 렌더링 프레임워크로 최근 소개되었으며, 계층 구조를 통한 효율적인 미분 가능 래스터화를 가능하게 합니다. 그러나 SVRaster는 원래 밀도를 측정값으로 사용하여 기하학적 인코딩이 모호해질 수 있다는 한계가 있습니다.

이를 보완하여 희소 복셀 래스터화 패러다임을 SDF(Signed Distance Function)와 통합한 SVRecon이 제안되어, 고품질 표면 복원으로 확장하고 있습니다.

---

## 4. 🔮 앞으로의 연구 영향 및 고려할 점

### 4.1 연구에 미치는 영향

**① 뉴럴 네트워크 없는 3D 표현의 부활**

이 연구는 효율적인 래스터라이저와 다중 레벨 희소 복셀 장면 표현을 통합한 새로운 미분 가능 Radiance Field 렌더링 시스템을 제시합니다. 이는 NeRF 이후 지배적이던 신경망 의존 방식에서 벗어나, 순수 명시적(explicit) 표현의 경쟁력을 재입증했다는 점에서 큰 의미가 있습니다.

**② Gaussian Splatting의 대안 패러다임 제시**

렌더링이 popping artifact로부터 자유로운 이유는 3D 공간이 서로 분리된(disjoint) 복셀로 분할되어 있고, 정렬이 올바른 렌더링 순서를 보장하기 때문입니다. 이는 3DGS의 고질적인 popping 문제를 근본적으로 해결하는 방향성을 제시합니다.

**③ 다운스트림 태스크 확장성**

복셀 표현은 Volume Fusion, Voxel Pooling, Marching Cubes와 같은 그리드 기반 3D 처리 기술과 원활하게 호환되어 광범위한 미래 확장 및 응용을 가능하게 합니다.

**④ 경량 라이트필드 렌더링으로의 확장**

최근 연구들은 3DGS와 SVRaster와 같은 희소 복셀 등 명시적/하이브리드 데이터 구조 및 래스터화 기반 파이프라인을 활용하여 렌더링을 가속화하고 있습니다. 라이트필드 디스플레이 등 신규 응용 분야로의 확장이 기대됩니다.

---

### 4.2 향후 연구 시 고려할 점

| 연구 방향 | 구체적 내용 |
|-----------|------------|
| **효율적인 검색 구조** | 현재 효율적인 검색(주어진 포인트에서 해당 복셀 탐색)을 지원하지 않으므로, 향후 복셀 정렬 및 이진 탐색 구현이 필요합니다. |
| **동적 장면 확장** | 정적 장면 위주 평가를 넘어, 시간 변화 복셀 표현 연구 필요 |
| **노출 불일치 처리** | SVRaster의 성능은 학습 뷰의 노출 변화에 민감하여, 다양한 입력 이미지에 걸친 광도 일관성(photometric consistency) 향상이 필요합니다. |
| **메모리 효율화** | 3DGS 대비 높은 GPU 메모리(3.9GB)와 모델 크기(1.8GB)를 가지므로, 복셀 압축 및 양자화 연구가 필요합니다. |
| **카메라 모델 다양화** | 현재 핀홀 카메라 모드만 지원하므로, fisheye·panoramic 카메라 지원 확장이 필요합니다. |
| **Generalizable 모델** | 장면별 최적화 없이 새로운 장면을 바로 렌더링할 수 있는 feed-forward 일반화 방향 |
| **SDF 기반 표면 복원** | SVRecon처럼 SDF와 통합하여 기하학 모호성 문제를 해결하는 방향의 연구가 중요합니다. |
| **DepthAnyhing 등과 결합** | DepthAnythingV2 상대 깊이 손실 및 MASt3R 메트릭 깊이 손실 등과의 결합으로 더 나은 기하학적 구조를 얻을 수 있습니다. |

---

## 5. 📊 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 연도 | 표현 방식 | 렌더링 속도 | 품질 | 신경망 | 주요 특징 |
|------|------|-----------|------------|------|--------|-----------|
| **NeRF** | 2020 | Implicit MLP | 매우 느림 | 고품질 | ✅ | 최초 neural radiance field |
| **DVGO** | 2022 | Dense Voxel Grid | 빠름 | 중간 | ❌ | 직접 복셀 최적화 |
| **Instant-NGP** | 2022 | Hash Grid | 매우 빠름 | 고품질 | ✅ | 해시 기반 인코딩 |
| **Mip-NeRF 360** | 2022 | Implicit MLP | 느림 | 최고품질 | ✅ | 무제한 장면, 안티에일리어싱 |
| **3DGS** | 2023 | 3D Gaussian | 실시간 | 고품질 | ❌ | 명시적 프리미티브 |
| **Zip-NeRF** | 2023 | Hash+MLP | 느림 | 최고품질 | ✅ | 3DGS 대비 PSNR 우수 |
| **SVRaster** | 2024 | Sparse Voxel | **실시간 (258 FPS)** | **고품질** | **❌** | **Popping 없음, 고전 3D 통합** |
| **SVRecon** | 2024 | Sparse Voxel + SDF | 실시간 | 고품질 | ❌ | 표면 복원 확장 |

SVRaster는 3DGS의 래스터화 효율성과 그리드 기반 표현의 구조적 볼류메트릭 접근법을 결합한 새로운 프레임워크로, (1) 다중 레벨 희소 복셀로 3D 장면을 모델링하고, (2) 적응형 크기 희소 복셀 표현에서의 래스터화 렌더링을 가능하게 하는 direction-dependent Morton Order 인코딩을 구현합니다.

---

## 📚 참고 자료 (출처 목록)

| # | 제목 | 출처 |
|---|------|------|
| 1 | **[논문 원문]** Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering | arXiv:2412.04459, https://arxiv.org/abs/2412.04459 |
| 2 | **[CVPR 2025 Open Access]** Sun et al., CVPR 2025, pp.16187-16196 | https://openaccess.thecvf.com/content/CVPR2025/html/Sun_Sparse_Voxels_Rasterization_Real-time_High-fidelity_Radiance_Field_Rendering_CVPR_2025_paper.html |
| 3 | **[공식 프로젝트 페이지]** SVRaster Project Page | https://svraster.github.io/ |
| 4 | **[공식 코드]** NVlabs/svraster (GitHub, NVIDIA) | https://github.com/NVlabs/svraster |
| 5 | **[NVIDIA Research]** Sparse Voxels Rasterization – NVIDIA Research Taiwan | https://research.nvidia.com/labs/twn/publication/cvpr_2025_svraster/ |
| 6 | **[논문 리뷰]** Liner.com Quick Review – SVRaster | https://liner.com/review/sparse-voxels-rasterization-realtime-highfidelity-radiance-field-rendering |
| 7 | **[관련 후속 연구]** SVRecon: Sparse Voxel Rasterization for Surface Reconstruction (arXiv:2511.17364) | https://arxiv.org/html/2511.17364 |
| 8 | **[관련 응용 연구]** Real-time 3D Visualization of Radiance Fields on Light Field Displays (arXiv:2508.18540) | https://arxiv.org/html/2508.18540 |
| 9 | **[HuggingFace Paper Page]** arXiv:2412.04459 | https://huggingface.co/papers/2412.04459 |
| 10 | **[ChatPaper 요약]** Sparse Voxels Rasterization 논문 요약 | https://chatpaper.com/chatpaper/paper/87945 |
| 11 | **[CVPR 2025 Poster]** SVRaster CVPR Poster | https://cvpr.thecvf.com/virtual/2025/poster/34749 |
