
# ZipMap: Linear-Time Stateful 3D Reconstruction via Test-Time Training 

> **논문 정보**
> - **제목:** ZipMap: Linear-Time Stateful 3D Reconstruction via Test-Time Training
> - **저자:** Haian Jin, Rundi Wu, Tianyuan Zhang, Ruiqi Gao, Jonathan T. Barron, Noah Snavely, Aleksander Hołyński
> - **소속:** Google DeepMind, Cornell University, MIT
> - **발표:** CVPR 2026
> - **arXiv:** [arXiv:2603.04385](https://arxiv.org/abs/2603.04385)
> - **프로젝트 페이지:** https://haian-jin.github.io/ZipMap/
> - **GitHub:** https://github.com/Haian-Jin/ZipMap

---

## 1. 핵심 주장 및 주요 기여 요약

Feed-forward transformer 기반의 최신 3D 재구성 모델들(VGGT, $\pi^3$)은 입력 이미지 수에 대해 **이차적(quadratic)** 계산 비용을 가지며, 대규모 이미지 컬렉션에 비효율적입니다.

순차적 재구성 방법은 이 비용을 줄이지만 재구성 품질을 희생합니다. ZipMap은 **선형 시간(linear-time)** 으로 양방향(bidirectional) 3D 재구성을 달성하면서 이차 시간 방법의 정확도를 따라가거나 능가하는 **stateful feed-forward 모델**입니다.

### 주요 기여 요약

| 기여 | 설명 |
|------|------|
| ① 선형 시간 복잡도 | $O(N)$ 시간으로 대규모 이미지 집합 처리 |
| ② TTT 기반 Scene State | 전체 이미지를 압축한 암묵적 장면 표현 |
| ③ 양방향 처리 | 순차 방법과 달리 bidirectional 처리 가능 |
| ④ 실시간 쿼리 및 스트리밍 | 새로운 시점 쿼리 및 스트리밍 확장 지원 |
| ⑤ 속도 | 단일 H100 GPU에서 700프레임을 10초 이내 처리 (VGGT 대비 $20\times$ 이상 빠름) |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

기존 SOTA 시스템들은 기하학적 일관성을 확보하기 위해 비싼 **global attention** 메커니즘에 의존하며, 입력 이미지 수가 증가할수록 재구성 시간이 **이차적으로(quadratically)** 증가하여 대규모 처리에 비현실적입니다.

CUT3R, Point3R, TTT3R 같은 방법들은 순차적 모델링이나 로컬 분할을 통해 이 문제를 해결하려 하지만, 이러한 전략은 재구성 품질을 저하시키는 경우가 많습니다.

핵심 문제를 수식으로 정리하면:

$$\text{기존 방법: } T(N) = O(N^2) \quad \text{(quadratic 비용)}$$

$$\text{ZipMap 목표: } T(N) = O(N) \quad \text{(linear 비용, 품질 유지)}$$

---

### 2.2 제안 방법 (수식 포함)

#### ① Test-Time Training (TTT) 레이어 — 핵심 메커니즘

ZipMap의 핵심은 **Test-Time Training (TTT) 레이어** 사용입니다. 전체 이미지 컬렉션을 MLP의 "fast-weights"로 구성된 compact hidden state로 압축하여 단일 forward pass에서 처리합니다. 이 상태 집계(state aggregation)는 고효율적이고 전역적으로 일관되어, 대규모 이미지 컬렉션으로의 확장성을 가능하게 합니다.

TTT는 모델의 파라미터 일부를 "fast-weight" 메모리로 취급하여, 온라인으로 경사 하강법을 통해 업데이트함으로써 문맥 정보를 캡처하며, 이를 통해 선형 및 비선형 순환 아키텍처의 설계 공간을 확장합니다.

TTT 레이어의 fast-weight 업데이트 원리를 수식으로 나타내면:

$$\mathbf{W}^* = \arg\min_{\mathbf{W}} \sum_{i} \ell\left(f_{\mathbf{W}}(\mathbf{x}_i),\ \mathbf{x}_i\right)$$

여기서 $f_{\mathbf{W}}$는 fast-weight MLP (SwiGLU-MLP), $\ell$은 self-supervised 손실 함수.

단일 forward pass에서 gradient step은:

$$\mathbf{W}_{t+1} = \mathbf{W}_t - \eta_t \cdot \nabla_{\mathbf{W}} \ell\left(f_{\mathbf{W}_t}(\mathbf{x}_t),\ \mathbf{x}_t\right)$$

Newton-Schulz 정규화(Eq. 4)와 gated unit(Eq. 7)은 핵심 구성요소로, 이 중 하나를 제거해도 성능이 저하됩니다.

Newton-Schulz 정규화를 수식으로 표현하면:

$$\mathbf{G}_{\text{NS}} = \text{NewtonSchulz}(\nabla_{\mathbf{W}} \ell) \approx \mathbf{U}\mathbf{V}^\top, \quad \text{where } \mathbf{G} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$$

이를 통해 그래디언트를 직교화(orthogonalize)하여 fast-weight 업데이트의 안정성을 확보합니다.

Gated unit (게이트 메커니즘):

$$\mathbf{h}_t = \mathbf{g}_t \odot f_{\mathbf{W}_t}(\mathbf{x}_t) + (1 - \mathbf{g}_t) \odot \mathbf{x}_t, \quad \mathbf{g}_t = \sigma(\mathbf{W}_g \mathbf{x}_t + \mathbf{b}_g)$$

#### ② 전체 복잡도

| 방법 | 복잡도 |
|------|--------|
| VGGT, $\pi^3$ (global attention) | $O(N^2)$ |
| CUT3R, TTT3R (sequential) | $O(N)$ but 낮은 GPU utilization |
| ZipMap (TTT fast-weight + local window) | $O(N)$, 고 GPU utilization |

---

### 2.3 모델 구조

ZipMap은 대형 feed-forward transformer와 Test-Time Training의 아키텍처 원리를 결합하여, 입력 수에 대해 선형으로 복잡도가 증가하는 양방향 모델을 만들어 대규모 이미지 컬렉션을 수초 내에 처리합니다.

주요 구조 요소는 다음과 같습니다:

```
입력 이미지 시퀀스 {I_1, ..., I_N}
        ↓
[Image Feature Extraction (Vision Backbone)]
        ↓
┌──────────────────────────────────────────────┐
│  ZipMap Bidirectional Encoder                │
│  ┌─────────────────────────────────────────┐ │
│  │ Global TTT Layer                        │ │
│  │ - SwiGLU-MLP fast-weights               │ │
│  │ - Newton-Schulz 정규화                  │ │
│  │ - Per-token 동적 학습률 (η_i)           │ │
│  │ - Gated unit                            │ │
│  ├─────────────────────────────────────────┤ │
│  │ Local Window Attention                  │ │
│  │ - 각 프레임 내 국소 공간 관계 처리      │ │
│  └─────────────────────────────────────────┘ │
│  (교차 반복: TTT → LocalAttn → TTT → ...)    │
└──────────────────────────────────────────────┘
        ↓
  Compact Hidden Scene State (fast-weights)
        ↓
┌─────────────────────────────────────┐
│ Decoder / Prediction Head           │
│ - Depth Maps (픽셀별 깊이)          │
│ - Point Maps (3D 좌표)              │
│ - Camera Pose / Trajectory          │
└─────────────────────────────────────┘
        ↓
  (Option) Real-time Novel-View Querying
  (Option) Streaming Reconstruction
```

이 stateful 표현은 추가적인 이점을 제공합니다: 실시간으로 새로운 시점(novel viewpoint)에서 픽셀 정렬 기하학 및 외관 정보를 쿼리할 수 있는 암묵적 장면 표현(implicit scene representation)으로 기능하며, 순차적 스트리밍 방식의 재구성으로 쉽게 확장될 수 있습니다.

간단한 fine-tuning 절차를 통해 TTT 기반 scene state를 한 번에 한 뷰씩 업데이트하는 스트리밍 방식으로 모델을 배포할 수 있습니다.

---

### 2.4 성능 향상 (Quantitative)

ZipMap은 단일 H100 GPU에서 700프레임 이상을 10초 내에 재구성하는 반면, VGGT 같은 이차적 방법은 200초 이상 소요됩니다. 또한 더 작은 모델임에도 불구하고 CUT3R, TTT3R 같은 이전 선형 시간 방법보다 약 $3\times$ 빠릅니다.

중요하게도, 이 속도는 품질의 희생 없이 달성됩니다. ZipMap은 ScanNetV2에서 $\pi^3$에 필적하고 VGGT보다 우수한 **ATE(Absolute Trajectory Error)** 를 달성하며, CUT3R와 TTT3R를 크게 능가합니다.

ZipMap은 카메라 포즈, 포인트 맵, 비디오/단안 깊이 추정 등 포괄적인 3D 태스크 집합에서 평가됩니다.

성능 비교 요약표:

| 모델 | 복잡도 | 700프레임 처리 시간 | ATE (ScanNetV2) |
|------|--------|---------------------|-----------------|
| VGGT | $O(N^2)$ | ~200초 | 기준 |
| $\pi^3$ | $O(N^2)$ | OOM/느림 | 최고 수준 |
| CUT3R | $O(N)$ | 빠름 | ZipMap보다 낮음 |
| TTT3R | $O(N)$ | 빠름 | ZipMap보다 낮음 |
| **ZipMap** | $O(N)$ | **<10초** | $\pi^3$와 동등 or 우수 |

모든 결과는 추가적인 최적화 없이 단일 H100 GPU에서 **75 FPS** 로 동작하는 ZipMap의 순수 feed-forward pass에 의해 생성됩니다.

---

### 2.5 한계 (Limitations)

논문 내 Appendix E에서 한계를 논의하고 있으며, 검색 결과 및 관련 연구들로부터 다음과 같은 한계를 파악할 수 있습니다:

1. **스트리밍 fine-tuning 컨텍스트 제한:** 시간 제약으로 인해 24-view 컨텍스트 길이로만 fine-tuning했으며, 스트리밍 베이스라인은 최대 64 views로 훈련됩니다. 따라서 fine-tuning 컨텍스트 길이를 더 늘리면 더 큰 성능 향상이 기대됩니다.

2. **Lossy Compression 가능성:** bidirectional 전체 모델(VGGT, $\pi^3$)들은 로컬 추론에 뛰어나지만 이차 비용이 장기 스케일링을 방해하며, 선형 메모리 대안들(CUT3R, TTT3R)은 연산 병목을 해결하지만 미세한 기하학적 정렬을 저하시키는 손실 압축(lossy compression)을 도입합니다.

3. **TTT 구성요소 민감도:** Newton-Schulz 정규화와 gated unit은 필수적이므로, 이를 제거하면 성능이 저하됩니다. 즉 하이퍼파라미터 설계가 중요합니다.

---

## 3. 일반화 성능 향상 가능성

### 3.1 Reference View 제거를 통한 일반화

최종 훈련 단계에서 명시적 reference-view 선택을 제거하고 $\pi^3$에서 제안된 **affine-invariant loss**로 훈련합니다. ZipMap 설정에서 reference view 제거는 표준 벤치마크에서 명확하거나 일관된 이점을 보이지 않지만, **긴 입력 시퀀스에서의 정확도와 일반화를 향상**시킵니다.

이 affine-invariant loss는 다음 형태를 가집니다:

```math
\mathcal{L}_{\text{aff-inv}} = \sum_{i} \left\| \frac{\hat{D}_i - \text{med}(\hat{D}_i)}{\text{MAD}(\hat{D}_i)} - \frac{D_i^* - \text{med}(D_i^*)}{\text{MAD}(D_i^*)} \right\|_1
```

여기서 $\hat{D}_i$는 예측 깊이, $D_i^*$는 GT 깊이, med와 MAD는 중앙값과 중앙절대편차.

이 방식은 특정 reference 좌표계에 의존하지 않으므로, **임의의 장면 규모 및 카메라 배치에 대한 일반화** 능력을 향상시킵니다.

### 3.2 Implicit Scene State 쿼리 능력

모델이 보이지 않는 영역의 공통 3D 구조(예: 벽, 바닥, 지면)를 추론할 수 있어, 기본적인 3D 장면 사전(scene prior)에 대한 이해를 나타냅니다.

이는 ZipMap의 fast-weight scene state가 단순 메모리를 넘어 **장면의 일반화된 3D 구조를 학습**했음을 시사합니다.

### 3.3 스트리밍 확장을 통한 다양한 입력 도메인 일반화

간단한 fine-tuning 절차로 ZipMap을 스트리밍 설정에서 TTT 기반 scene state를 한 번에 한 뷰씩 업데이트하도록 배포할 수 있으며, 스트리밍 변형은 포인트맵 재구성, 비디오 깊이, 카메라 포즈 추정에서 일반적으로 CUT3R와 TTT3R를 능가합니다.

### 3.4 일반화 향상 전략 (아키텍처 수준)

| 전략 | 일반화 기여 |
|------|-------------|
| Affine-invariant loss | 장면 규모 불변성 |
| Reference-view 제거 | 긴 시퀀스 일반화 |
| TTT fast-weight | 각 테스트 시 장면에 적응 |
| 양방향 처리 | 순방향/역방향 컨텍스트 통합 |
| Implicit scene state | 미관측 영역까지 일반화 |

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 Feed-forward 3D 재구성의 계보

```
DUSt3R (2023)
  └→ MASt3R (2024)
      └→ VGGT (2025, O(N²))          ← 고품질, 비효율
      └→ π³ (2025, O(N²))            ← 고품질, 비효율
CUT3R (2025, O(N), sequential)       ← 효율적, 품질 저하
TTT3R (2025, O(N), TTT 기반)         ← 효율 개선, 순차 처리
ZipMap (2026, O(N), bidirectional TTT) ← 효율 + 품질 동시 달성
```

### 4.2 비교표

| 모델 | 연도 | 복잡도 | 방향성 | 품질 | Scene State |
|------|------|--------|--------|------|-------------|
| DUSt3R | 2023 | $O(N^2)$ | bidirectional | 중 | ✗ |
| VGGT | 2025 | $O(N^2)$ | bidirectional | 최고 | ✗ |
| $\pi^3$ | 2025 | $O(N^2)$ | bidirectional | 최고 | ✗ |
| CUT3R | 2025 | $O(N)$ | sequential | 낮음 | ✓(외재적) |
| TTT3R | 2025 | $O(N)$ | sequential | 중간 | ✓ |
| **ZipMap** | **2026** | $O(N)$ | **bidirectional** | **최고급** | **✓(암묵적)** |

ZipMap은 자기 주의(self-attention) 기반 설계를 선형 스케일링 stateful 모델로 대체하여, 다른 순차적 해결책과 달리 **순환적 처리(recurrent processing)가 필요 없어** 오류 누적에 덜 취약합니다.

### 4.3 동시대 경쟁 연구

- **VGG-T³** (NVIDIA): TTT3R이 장면을 완전히 재구성하지 못하는 반면 VGG-T³는 완전한 재구성을 제공하며, VGGT의 품질은 약간 더 높지만 11배 이상 오래 걸립니다. 이는 TTT 기반 선형 시간 global attention 대체의 효과를 강조합니다.

- **LoGeR**: 완전한 bidirectional 모델(VGGT, $\pi^3$)은 로컬 추론이 뛰어나지만 이차 비용이 장기 스케일링을 방해하고, 선형 메모리 대안들(CUT3R, TTT3R)은 연산 병목을 해결하지만 미세한 기하 정렬을 저하시키는 손실 압축을 도입한다고 지적합니다.

---

## 5. 향후 연구에 미치는 영향 및 고려할 점

### 5.1 향후 연구에 미치는 영향

ZipMap은 선형 시간으로 확장되는 stateful bidirectional 아키텍처로 SOTA 이차 시간 모델을 따라가거나 능가하며, 학습된 scene state는 실시간 novel-view 포인트맵 예측을 위해 쿼리 가능하고 스트리밍 재구성으로 쉽게 확장됩니다. 이러한 결과들은 대규모 이미지 컬렉션에서의 **확장 가능한 고품질 3D 지각을 향한 새로운 경로**를 제시합니다.

주요 영향 영역:

1. **실시간 AR/VR 시스템:** 대규모 장면의 실시간 3D 재구성 가능성 확대
2. **로보틱스 내비게이션:** 장시간 스트리밍 입력에서의 실시간 지도 작성(mapping)
3. **자율주행:** 연속적인 카메라 스트림의 효율적 3D 이해
4. **암묵적 신경 표현(Neural Implicit Representations):** TTT fast-weight를 NeRF/3DGS 등과 결합하는 연구
5. **TTT 패러다임의 확산:** 다른 3D 비전 태스크(semantic segmentation, object detection 등)로 TTT 기반 효율화 확장 자극

### 5.2 향후 연구 시 고려할 점

**① 스트리밍 fine-tuning 컨텍스트 확장**
현재 24-view 컨텍스트로만 fine-tuning했으며, 스트리밍 베이스라인은 최대 64 views로 훈련됩니다. 따라서 fine-tuning 컨텍스트 길이를 추가로 확장하면 더 큰 성능 이점이 기대됩니다.

**② TTT 하이퍼파라미터 안정성**

Test-Time Training 안정성 측면에서 내부 루프 하이퍼파라미터(학습률 $\eta_i$, 업데이트 빈도), 수렴, 노이즈가 많거나 모호한 구간에서의 과적합/드리프트(drift)에 대한 분석이 아직 충분하지 않습니다.

수식적으로, 다음 조건에서 안정적 업데이트가 필요합니다:

$$\|\mathbf{W}_{t+1} - \mathbf{W}_t\|_F \leq \epsilon \quad \forall t, \quad \text{where } \epsilon \text{는 허용 드리프트 임계값}$$

**③ 동적 장면(Dynamic Scene) 처리**

현재 ZipMap의 구조는 정적 장면에 최적화되어 있으며, 움직이는 객체가 포함된 **동적 장면**에서의 fast-weight 업데이트 전략은 추가 연구가 필요합니다.

**④ Fast-Weight 용량 확장성**

현재는 1M–4M의 "state size"를 가진 compact MLP만 탐색되었으며, 대안적 아키텍처(저순위 어댑터, 잔차 메모리, key-value 캐시, 선형 state-space 모듈)나 더 크거나 작은 용량과의 비교가 이루어지지 않았습니다.

**⑤ 다중 모달 확장**

현재는 RGB 이미지 기반이나, **RGB-D, LiDAR, 이벤트 카메라** 등 다중 센서 입력을 TTT fast-weight 내에 통합하는 연구가 가능합니다.

**⑥ 3DGS/NeRF와의 통합**

ZipMap이 생성하는 implicit scene state(fast-weights)를 3D Gaussian Splatting이나 NeRF 초기화로 활용하는 파이프라인 구성이 유망한 연구 방향입니다.

---

## 참고 자료 (출처)

1. **논문 원문 (arXiv):** Haian Jin et al., "ZipMap: Linear-Time Stateful 3D Reconstruction via Test-Time Training," *arXiv:2603.04385*, CVPR 2026. https://arxiv.org/abs/2603.04385
2. **논문 HTML 전문 (최신 v3):** https://arxiv.org/html/2603.04385
3. **논문 HTML 전문 (v1):** https://arxiv.org/html/2603.04385v1
4. **프로젝트 공식 페이지:** https://haian-jin.github.io/ZipMap/
5. **GitHub 저장소:** https://github.com/Haian-Jin/ZipMap
6. **TTT3R 논문 (비교 대상):** X. Chen et al., "TTT3R: 3D Reconstruction as Test-Time Training," *arXiv:2509.26645* — https://arxiv.org/html/2509.26645
7. **VGG-T³ 프로젝트 (NVIDIA, 비교 대상):** https://research.nvidia.com/labs/dvl/projects/vgg-ttt/
8. **LoGeR 프로젝트 (비교 대상):** https://loger-project.github.io/
9. **Scal3R (관련 연구):** https://www.emergentmind.com/papers/2604.08542
10. **InfiniteVGGT (관련 연구):** *arXiv:2601.02281*
11. **ZipMap 분석 (spatial_agi GitHub):** https://github.com/ahangchen/spatial_agi/blob/main/papers/2026-03-06_ZipMap_Linear_Time_3D.md

> ⚠️ **정확도 관련 주의:** 본 답변에서 제시된 수식 중 일부(affine-invariant loss, gated unit 등)는 논문 본문에서 직접 확인된 구조를 바탕으로 합리적으로 표현한 것이며, 논문의 정확한 수식 번호(예: Eq. 4, Eq. 7)에 해당하는 세부 계수는 PDF 전문에서 확인을 권장합니다.
