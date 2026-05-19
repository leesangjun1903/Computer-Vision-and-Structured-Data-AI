# Z-Order Transformer for Feed-Forward Gaussian Splatting

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문은 **비정형(unstructured) Gaussian 집합을 Z-order 곡선(Morton curve)으로 공간적으로 일관된 1D 시퀀스로 재구성**함으로써, Transformer 기반의 sparse attention이 Gaussian 간의 공간적·의미적 관계를 효율적으로 포착할 수 있다고 주장합니다. 이를 통해 적은 수의 Gaussian primitive만으로 고품질 Novel View Synthesis(NVS)를 단일 forward pass에서 수행할 수 있습니다.

### 주요 기여 (4가지)

| 기여 | 설명 |
|------|------|
| **Z-Order Splat Transformer (ZFormer)** | Z-order 기반 직렬화로 Gaussian을 공간적 시퀀스로 변환 후 sparse attention 적용 |
| **Sparse Attention 메커니즘** | Group Attention + Top-K Attention의 결합으로 계산 복잡도 절감 |
| **Z-Order Pooling** | 비트 이동 연산 기반의 계층적 Gaussian 압축으로 primitive 수 감소 |
| **Z-Order 기반 최대 커버리지 뷰포인트 선택** | Dense view 입력 시 중복 제거 알고리즘으로 추론 효율화 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

#### 기존 방법의 한계

**① Per-scene 최적화 3DGS의 문제**
- 장면마다 gradient descent로 수백~수천 회 반복 최적화 필요 (수 분 소요)
- 일반화 불가: 새 장면마다 처음부터 재훈련

**② Pixel-level Feed-forward GS의 문제**
- $512 \times 512$ 이미지 2장 → $5 \times 10^5$개 이상의 Gaussian primitive 생성
- 메모리·렌더링 비용 폭증

**③ Voxel-based GS의 문제**
- 3D 공간 이산화 → 양자화 오류(fine detail 손실)
- 해상도 증가 시 메모리가 $O(r^3)$으로 증가
- 희소/불규칙 영역에서 빈 셀(empty cell)로 인한 낭비

### 2.2 제안 방법 및 수식

#### Step 1: Feed-forward 3DGS 공식화

$N$개의 입력 이미지 $\mathbf{I} = \{I\}_{i=1}^{N} \in \mathbb{R}^{N \times 3 \times H \times W}$로부터 신경망 $\mathcal{F}$가 Gaussian primitive를 직접 예측:

$$\{G_k : (\mu_k, \sigma_k, r_k, s_k, c_k)\}_{k=1}^{K} = \mathcal{F}(\{I\}_{i=1}^{N}) $$

각 Gaussian $G_k$의 파라미터:
- $\mu_k \in \mathbb{R}^3$: 위치(평균)
- $\sigma_k \in \mathbb{R}^+$: 불투명도(opacity)
- $r_k \in \mathbb{R}^4$: 회전 쿼터니언
- $s_k \in \mathbb{R}^3$: 비등방성 스케일
- $c_k \in \mathbb{R}^{27}$: 2차 구면 조화 함수(SH) 색상 계수

#### Step 2: Z-Order 곡선 인코딩

3D 점 $P = (x, y, z)$의 각 좌표를 $d$비트 이진수로 표현:

$$x = \sum_{i=0}^{d-1} x_i 2^i, \quad y = \sum_{i=0}^{d-1} y_i 2^i, \quad z = \sum_{i=0}^{d-1} z_i 2^i $$

비트 인터리빙을 통한 Z-order 코드 생성:

$$\mathbf{Z}(x, y, z) = \sum_{i=0}^{d-1} \left( x_i \cdot 2^{3i} + y_i \cdot 2^{3i+1} + z_i \cdot 2^{3i+2} \right) $$

이를 통해 **3D 공간에서 인접한 점들이 1D 시퀀스에서도 인접**하게 배치됩니다.

#### Step 3: ZFormer Block - Sparse Attention

**Query, Key, Value 생성:**

$$\mathbf{Q} = \mathbf{F}_{\text{sorted}} W_Q, \quad \mathbf{K} = \mathbf{F}_{\text{sorted}} W_K, \quad \mathbf{V} = \mathbf{F}_{\text{sorted}} W_V $$

여기서 $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{(NHW) \times (96/3)}$

**블록 단위 평균 풀링 연산자 $\mathcal{C}$:**

$$\mathcal{C}(\mathbf{X})_i = \frac{1}{L} \sum_{j=1}^{L} \mathbf{X}_{(i-1)L+j} \quad \text{for } i = 1, 2, \ldots, B $$

$$\hat{\mathbf{Q}} = \mathcal{C}(\mathbf{Q}), \quad \hat{\mathbf{K}} = \mathcal{C}(\mathbf{K}), \quad \hat{\mathbf{V}} = \mathcal{C}(\mathbf{V}) $$

**Group Attention (지역 정보 포착):**

$$\mathbf{Attn}_{\text{grp}}(\hat{\mathbf{Q}}, \hat{\mathbf{K}}, \hat{\mathbf{V}}) = \text{softmax}\left(\frac{\hat{\mathbf{Q}}\hat{\mathbf{K}}^\top}{\sqrt{d}}\right)\hat{\mathbf{V}} $$

**Top-K Attention (전역 세밀 정보 포착):**

Group attention의 가중치 $w = \text{softmax}\left(\frac{\hat{\mathbf{Q}}\hat{\mathbf{K}}^\top}{\sqrt{d}}\right)$를 사용하여 중요 블록 $k$개 선택 후:

$$\mathbf{Attn}_{\text{sel}} = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}_{\text{sel}}^\top}{\sqrt{d}}\right)\mathbf{V}_{\text{sel}} $$

**게이트 네트워크를 통한 적응적 융합:**

$$\mathbf{F}_{\text{gate}} = g_1(\mathbf{F}_{\text{sorted}}) \odot \mathbf{Attn}_{\text{grp}} + g_2(\mathbf{F}_{\text{sorted}}) \odot \mathbf{Attn}_{\text{sel}} $$

#### Step 4: Z-Order Pooling (Gaussian 압축)

pooling depth $h$를 정의하고 비트 우측 이동 연산:

$$\mathbf{Z} = \mathbf{Z} >> h$$

동일한 Z-order 코드를 가진 점들을 클러스터링 후 평균 풀링 → 압축된 표현 $\mathbf{R} = \{\mathbf{P}\_{\text{pool}}, \mathbf{F}\_{\text{pool}}, \mathbf{I}\_{\text{pool}}\}$, $\mathbf{P}_{\text{pool}} \in \mathbb{R}^{M \times 3}$ (M $\ll$ NHW)

#### Step 5: Gaussian Head

두 레이어 MLP로 multi-level Gaussian 파라미터 예측:

$$G_{L1}, G_{L2} = \mathcal{F}_{\text{head}}(\mathbf{R}_{L1}), \mathcal{F}_{\text{head}}(\mathbf{R}_{L2}) $$

최종 Gaussian 중심: $\mu = \mathbf{P}_{\text{pool}} + \Delta\mu$

#### Step 6: 학습 손실 함수

**Depth distillation loss (Depth Anything V2로부터):**

$$\mathcal{L}_{\text{depth}} = \left| \mathcal{F}_{\text{depth}}(\mathbf{I}) - \hat{\mathcal{F}}_{\text{depth}}(\mathbf{I}) \right| $$

**렌더링 품질 손실 (MSE + LPIPS):**

$$\mathcal{L}_{\text{color}} = \sum_{i=1}^{M} \left[ \text{MSE}(\mathcal{R}(G_{Li}, \mathbf{c}), \mathbf{I}_{\text{gt}}) + \text{LPIPS}(\mathcal{R}(G_{Li}, \mathbf{c}), \mathbf{I}_{\text{gt}}) \right] $$

### 2.3 모델 구조

```
Multi-View Images (N×3×H×W)
        ↓
[DINOv2-Small Transformer Encoder]
  ├── Global Feature Map: F_global ∈ R^{N×32×H×W}
  └── [DPT Depth Head]
        └── Geometry Feature: F_geom ∈ R^{N×64×H×W}
              └── Depth Maps: D ∈ R^{N×H×W}
                    └── Point Map: P ∈ R^{N×(HW)×3}

Feature Fusion: F = Concat(F_global, F_geom) → R^{N×96×H×W}

[ZFormer Block #L1]
  1. Z-order Serialization & Sorting (3D→1D)
  2. Group Attention (지역적 블록 집계)
  3. Top-K Attention (중요 블록 선택)
  4. Gated Fusion
  5. Z-order Pooling (차원 압축)
  → R_L1 = {P_pool^L1, F_pool^L1, I_pool^L1}

[ZFormer Block #L2]
  (동일 과정, 더 압축)
  → R_L2 = {P_pool^L2, F_pool^L2, I_pool^L2}

[Gaussian Head (2-layer MLP)]
  → G_L1, G_L2 (Multi-Level Gaussian Primitives)

[3DGS Renderer] → Novel View Images
```

### 2.4 성능 향상

#### RealEstate10K (360×640) 결과

| Method | 2 Views PSNR↑ | 12 Views PSNR↑ | #GS (2V) |
|--------|--------------|----------------|----------|
| 3DGS | 16.80 | 26.73 | 6.27×10⁵ |
| DepthSplat | 26.03 | 26.33 | 4.61×10⁵ |
| AnySplat | 22.55 | 26.94 | 3.53×10⁵ |
| **Ours#L1** | **26.43** | **28.56** | **2.85×10⁵** |
| **Ours#L2** | 26.42 | 28.12 | **1.42×10⁵** |

#### 추론 속도 비교 (2 Views 기준)

| Method | 추론 시간 |
|--------|---------|
| 3DGS | 2분 15초 |
| MipSplatting | 1분 18초 |
| DepthSplat | 0.142초 |
| **Ours#L1** | **0.123초** |

→ **3DGS 대비 약 1,000배 빠름**, Gaussian primitive 수 **2~3배 감소**

### 2.5 한계점

1. **고해상도 한계**: 1K 이상 해상도에서 세부 디테일 포착 어려움 (메모리-복잡도 트레이드오프)
2. **Z-order 블록 수의 제한**: 3개 이상 블록 사용 시 성능 급격히 저하 (Fig. 7 참조, #L3에서 PSNR 25.71로 하락)
3. **더 높은 압축의 어려움**: primitive 수를 더 줄이면서 품질을 유지하는 것이 여전히 도전적

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 Cross-Dataset 일반화 실험 결과

논문은 ACID 데이터셋을 **훈련에 전혀 사용하지 않고** 테스트에만 활용하여 일반화 성능을 측정했습니다:

| 평가 설정 | Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|---------|--------|-------|-------|--------|
| RealEstate10K → ACID | DepthSplat | 26.05 | 0.810 | 0.181 |
| | AnySplat | 22.71 | 0.685 | 0.298 |
| | **Ours#L1** | **27.56** | **0.853** | **0.172** |
| DL3DV → ACID | DepthSplat | 25.58 | 0.796 | 0.203 |
| | AnySplat | 23.64 | 0.737 | 0.242 |
| | **Ours#L1** | **27.34** | **0.845** | **0.169** |

### 3.2 일반화 향상의 구조적 원인

#### ① Z-order의 뷰 수 불변성 (View-count Invariance)

Z-order serialization은 입력 뷰 수와 무관하게 동일한 공간 분할 원리로 작동합니다. 2~64개의 다양한 뷰 수에 걸쳐 일관된 성능을 보여줍니다:

$$\mathbf{Z}(x, y, z) = \sum_{i=0}^{d-1} \left( x_i \cdot 2^{3i} + y_i \cdot 2^{3i+1} + z_i \cdot 2^{3i+2} \right)$$

이 공간 인덱싱은 장면의 종류나 뷰 수에 관계없이 동일하게 적용 가능하므로, **도메인 불변적 공간 표현**을 제공합니다.

#### ② 멀티태스크 학습의 일반화 기여

깊이 추정( $\mathcal{L}\_{\text{depth}}$ )과 렌더링 품질($\mathcal{L}_{\text{color}}$)을 동시에 훈련함으로써:
- 2D 이미지 기반 깊이 추정의 부정확성을 보정
- 기하학적 이해도(geometry understanding)가 향상되어 미보학 장면에서도 robust한 동작

#### ③ Sparse View 강건성

2-view 입력에서 **Ours#L1 PSNR 26.43 vs DepthSplat 26.03** (RealEstate10K)으로 특히 희소 뷰에서 우수한 성능을 보입니다. 이는 Z-order 기반 공간 집계가 관측되지 않은 영역을 더 잘 추론함을 의미합니다.

#### ④ SH 파라미터 초기화를 통한 색상 일반화

픽셀 색상 $\mathbf{I}_{\text{pool}}$을 SH 계수 초기값으로 사용:

$$c_{\text{init}} = \text{RGB2SH}(\mathbf{I}_{\text{pool}})$$

이는 다양한 조명 환경을 가진 새로운 데이터셋에서도 안정적인 수렴을 제공합니다. (Ours w/o SH: PSNR 27.81 → Ours: 28.56, Tab. 5)

#### ⑤ 백본 독립성 (Backbone Agnosticism)

보충 실험에서 VGGT 백본으로 교체해도 프레임워크가 작동함을 확인 (Tab. S10):

| 백본 | PSNR | 추론 시간 |
|-----|------|---------|
| Depth-Anything-V2-Small (24.8M) | 28.56 | 0.337s |
| VGGT (1B) | 28.81 | 0.815s |

이는 **ZFormer 구조 자체가 특정 feature extractor에 종속되지 않음**을 보여주며, 더 강력한 백본 적용 시 성능 추가 향상 가능성을 시사합니다.

#### ⑥ Z-order 기반 최대 커버리지 뷰포인트 선택의 일반화

Dense view 상황(64 views → 16 views 선택)에서:

| 설정 | PSNR | 시간 |
|-----|------|-----|
| 전체 64뷰 사용 | 29.44 | 1.891s |
| 랜덤 16뷰 | 28.50 | 0.421s |
| **Z-order 16뷰** | **29.13** | 0.498s |

Z-order 선택이 랜덤 대비 PSNR +0.63 향상, 전체 대비 -0.31만 손실. 이 알고리즘은 **장면 구조에 적응적으로 뷰를 선택**하므로 새로운 환경에서도 효과적입니다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 계보 분류

```
NeRF (2020, Mildenhall et al.)
    ↓ 속도 문제
3DGS (2023, Kerbl et al.) → per-scene, slow
    ↓ 일반화 문제
Feed-Forward GS 계열:
  ├── PixelSplat (CVPR 2024): pixel-level, 많은 primitive
  ├── MVSplat (ECCV 2024): 희소 멀티뷰
  ├── DepthSplat (CVPR 2025): 깊이+pixel GS
  ├── AnySplat (arXiv 2025): voxel 기반
  ├── NoPoSplat (ICLR 2025): 포즈 없이
  ├── MonoSplat (CVPR 2025): 단안 깊이
  └── Z-Order Transformer (본 논문, 2025): Z-order+sparse attention
```

### 4.2 주요 방법 비교 (RealEstate10K 256×256, 2 Views)

| 방법 | 연도/학회 | 핵심 기술 | PSNR | SSIM | LPIPS | 일반화 |
|-----|---------|---------|------|------|-------|--------|
| PixelSplat | CVPR 2024 | Pixel-level epipolar features | 25.89 | 0.858 | 0.142 | 제한적 |
| MVSplat | ECCV 2024 | Cost volume + pixel GS | 26.39 | 0.869 | 0.128 | 제한적 |
| NoPoSplat | ICLR 2025 | 포즈 없는 희소 GS | 27.41 | 0.884 | 0.116 | 우수 |
| MonoSplat | CVPR 2025 | 단안 깊이 기반 GS | 26.68 | 0.875 | 0.123 | 보통 |
| FLARE | CVPR 2025 | 비보정 희소 뷰 | 23.78 | 0.801 | 0.191 | 보통 |
| DepthSplat | CVPR 2025 | 깊이 연계 GS | ~26.0 | - | - | 우수 |
| AnySplat | arXiv 2025 | Voxel 기반 집계 | ~22.5 | - | - | 우수 |
| **Z-Order Transformer** | arXiv 2025 | Z-order+Sparse Attn | **27.89** | **0.892** | **0.110** | **최우수** |

### 4.3 기술적 혁신 비교

| 측면 | PixelSplat/MVSplat | DepthSplat | AnySplat | **Z-Order Transformer** |
|-----|-------------------|-----------|---------|----------------------|
| Gaussian 표현 | Pixel-level | Pixel+Depth | Voxel | **Z-order (적응적)** |
| 공간 관계 모델링 | 암묵적 | 깊이 기반 | 복셀 집계 | **Z-order sparse attention** |
| Primitive 수 | 매우 많음 | 많음 | 중간 | **적음** |
| 추론 속도 | 빠름 | 빠름 | 보통 | **매우 빠름** |
| 고해상도 확장성 | 메모리 문제 | 메모리 문제 | 큐빅 증가 | **효율적** |
| 가변 뷰 대응 | 제한적 | 가능 | 가능 | **우수** |

### 4.4 Point Transformer V3와의 비교 (영감의 원천)

Point Transformer V3 (CVPR 2024, Wu et al.)은 포인트 클라우드에서 Z-order 직렬화를 도입했습니다. 본 논문은 이를 **3DGS의 feed-forward 추론**에 창의적으로 적용하여:

- 포인트 클라우드 분석 → **Gaussian primitive 예측**으로 확장
- 정적 3D 장면 처리 → **멀티뷰 이미지로부터의 동적 재구성**으로 적용 확대
- 단일 스케일 → **Multi-level 계층적 표현** 도입

---

## 5. 향후 연구에 미치는 영향 및 고려 사항

### 5.1 향후 연구에 미치는 영향

#### ① 공간 인덱싱과 Neural Rendering의 통합 패러다임 확립

Z-order curve를 Gaussian 처리에 적용한 최초의 체계적 연구로서, **공간 데이터베이스 기법(spatial indexing)을 신경 렌더링에 적용하는 방법론적 교량** 역할을 합니다. 이는 향후 다음과 같은 연구 방향을 열어줍니다:

- Hilbert curve, Peano curve 등 다른 space-filling curve의 적용
- 적응적(learnable) 공간 분할 전략 연구

#### ② Feed-forward 3DGS의 효율성 기준 재설정

Gaussian primitive 수를 기존 대비 2~3배 줄이면서 품질을 향상시켜, **primitive 효율성**이 feed-forward GS의 핵심 평가 지표로 자리잡도록 유도합니다.

#### ③ Sparse Attention의 3D 비전 적용 확산

Group attention + Top-K attention의 결합이 3D 비전 태스크에서 효과적임을 실증하여, **3D point cloud, scene flow, dynamic 3DGS 등에서 유사한 접근법 확산** 예상됩니다.

#### ④ Multi-task 학습의 중요성 재확인

깊이 추정과 렌더링을 동시에 학습하는 멀티태스크 전략이 일반화에 핵심적임을 보여줌으로써, 향후 **semantic segmentation, surface normal 등 추가 auxiliary task와의 결합** 연구를 촉진할 것입니다.

### 5.2 향후 연구 시 고려해야 할 점

#### ① Z-order 압축의 이론적 한계 극복

현재 3개 이상의 Z-order 블록 사용 시 성능 저하가 발생합니다. 이를 극복하기 위해:

- **학습 가능한 집계 깊이(learnable aggregation depth)**: 장면 복잡도에 따라 압축 깊이를 동적으로 결정
- **계층적 다중 스케일 표현**: 세부 디테일을 보존하는 skip connection 도입

$$h^* = \arg\min_{h} \mathcal{L}_{\text{color}} + \lambda \cdot M(h)$$

여기서 $M(h)$는 pooling depth $h$에서의 primitive 수.

#### ② 고해상도(>1K) 확장 전략

현재 1K 이상 해상도에서 어려움 존재. 해결 방향:

- **계층적 Z-order 처리**: 이미지를 타일로 분할 후 개별 처리 + 경계 통합
- **Progressive refinement**: 저해상도에서 시작하여 점진적 해상도 증가
- **메모리 효율적 FlashAttention 확장**: 더 큰 시퀀스 처리를 위한 I/O 최적화

#### ③ 동적 장면(Dynamic Scene) 대응

현재 정적 장면에만 적용. 동적 장면으로의 확장:

- 시간 차원을 Z-order 코드에 통합: $\mathbf{Z}(x, y, z, t)$
- 4DGS와의 결합 (Liu et al., 2026 참고)

#### ④ 더 강력한 백본과의 통합

논문의 보충 실험(Tab. S10)에서 VGGT 백본이 더 좋은 결과를 보임:

$$\Delta\text{PSNR} = 28.81 - 28.56 = +0.25 \text{ (VGGT vs Depth-Anything-V2)}$$

효율성을 희생하지 않는 범위에서 **더 강력한 vision foundation model과의 통합** 연구가 필요합니다.

#### ⑤ 비보정(Uncalibrated) 환경에서의 적용

현재 알려진 카메라 파라미터를 가정합니다. 카메라 포즈 추정과의 end-to-end 통합:

- NoPoSplat (ICLR 2025)처럼 포즈 없는 환경에서도 작동하도록 확장
- VGGT 기반 카메라 추정 모듈과의 통합

#### ⑥ 다양한 입력 모달리티 확장

- **실내·실외 혼합 데이터셋**: 현재 주로 실내(RealEstate10K) 환경에서 평가
- **360도 파노라마**: Z-order를 구형(spherical) 좌표계로 확장
- **LiDAR 포인트 클라우드와의 융합**: 더 정확한 기하학적 초기화

#### ⑦ 객체 수준(Object-level) 3DGS로의 적용

현재는 장면 수준(scene-level). 개별 객체의 세밀한 표현을 위한 **hierarchical Z-order 표현** 연구가 필요합니다.

---

## 참고 자료

**주요 논문 (본 논문 내 인용 포함):**

1. **Wang et al. (2025)** - "Z-Order Transformer for Feed-Forward Gaussian Splatting" (arXiv:2605.13465v1, 본 논문)
2. **Kerbl et al. (2023)** - "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (ACM Trans. Graph.)
3. **Charatan et al. (2024)** - "PixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction" (CVPR 2024)
4. **Chen et al. (2024)** - "MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images" (ECCV 2024)
5. **Xu et al. (2025)** - "DepthSplat: Connecting Gaussian Splatting and Depth" (CVPR 2025)
6. **Jiang et al. (2025)** - "AnySplat: Feed-Forward 3D Gaussian Splatting from Unconstrained Views" (arXiv:2505.23716)
7. **Wu et al. (2024)** - "Point Transformer V3: Simpler Faster Stronger" (CVPR 2024)
8. **Yu et al. (2024)** - "Mip-Splatting: Alias-Free 3D Gaussian Splatting" (CVPR 2024)
9. **Yang et al. (2024)** - "Depth Anything V2" (NeurIPS 2024)
10. **Oquab et al. (2025)** - "DINOv2: Learning Robust Visual Features without Supervision" (TMLR 2025)
11. **Morton (1966)** - "A Computer Oriented Geodetic Data Base and a New Technique in File Sequencing" (IBM)
12. **Ye et al. (2025)** - "NoPoSplat: Surprisingly Simple 3D Gaussian Splats from Sparse Unposed Images" (ICLR 2025)
13. **Liu et al. (2025)** - "MonoSplat: Generalizable 3D Gaussian Splatting from Monocular Depth Foundation Models" (CVPR 2025)
14. **Yuan et al. (2025)** - "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention" (arXiv:2502.11089)
15. **Zeng et al. (2025)** - "ZETA: Leveraging Z-Order Curves for Efficient Top-K Attention" (ICLR 2025)
16. **Wang et al. (2025)** - "VGGT: Visual Geometry Grounded Transformer" (CVPR 2025)
17. **Ranftl et al. (2021)** - "Vision Transformers for Dense Prediction (DPT)" (ICCV 2021)
18. **Zhang et al. (2018)** - "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS)" (CVPR 2018)
19. **Gupta et al. (2021)** - "Memory-Efficient Transformers via Top-K Attention" (SimpleNLP Workshop 2021)
20. **Shah et al. (2024)** - "FlashAttention-3" (NeurIPS 2024)
