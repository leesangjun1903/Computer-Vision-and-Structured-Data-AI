
# LODGE: Level-of-Detail Large-Scale Gaussian Splatting with Efficient Rendering

> **논문 정보**
> - **제목:** LODGE: Level-of-Detail Large-Scale Gaussian Splatting with Efficient Rendering
> - **저자:** Jonas Kulhanek, Marie-Julie Rakotosaona, Fabian Manhardt, Christina Tsalicoglou, Michael Niemeyer, Torsten Sattler, Songyou Peng, Federico Tombari
> - **arXiv:** [2505.23158](https://arxiv.org/abs/2505.23158) (2025년 5월 29일 제출, v2: 2025년 10월 29일)
> - **학회:** NeurIPS 2025 (Spotlight)
> - **프로젝트 페이지:** [lodge-gs.github.io](https://lodge-gs.github.io/)
> - **OpenReview:** [openreview.net/forum?id=Iqu63cYI3z](https://openreview.net/forum?id=Iqu63cYI3z)

---

## 1. 핵심 주장 및 주요 기여 요약

본 논문은 메모리 제약이 있는 디바이스에서 대규모 장면의 실시간 렌더링을 가능하게 하는 3D Gaussian Splatting을 위한 새로운 Level-of-Detail (LOD) 방법을 제시합니다.

한 줄 요약(TL;DR): LODGE는 대규모 3D 장면에서 뛰어난 품질과 우수한 렌더링 속도를 제공하며, 모바일 디바이스에서도 실시간 렌더링을 가능하게 합니다.

### 주요 기여 4가지

| 기여 항목 | 설명 |
|---|---|
| ① 계층적 LOD 표현 | 카메라 거리 기반 Gaussian 부분집합 선택 |
| ② Depth-aware 3D Smoothing Filter | LOD 레벨 간 시각적 품질 유지 |
| ③ Chunk 기반 렌더링 + Opacity Blending | GPU 메모리 절감 및 경계 아티팩트 제거 |
| ④ 자동 임계값 선택 전략 | 씬별 수동 하이퍼파라미터 튜닝 불필요 |

본 방법은 공간 영역에 대해 활성 Gaussian 집합을 미리 계산하는 방식으로 프레임별 오버헤드를 회피하는 청크 기반 렌더링과 다중 레벨 LOD 표현을 결합하며, 청크 간 전환 시 자동 임계값 선택 전략과 2-클러스터 Opacity Blending 방식을 추가로 제안합니다.

---

## 2. 해결하고자 하는 문제, 제안 방법(수식 포함), 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

Novel View Synthesis는 AR/VR, 게임, 인터랙티브 지도 등 다양한 응용에서 핵심 연구 분야입니다. NeRF와 3D Gaussian Splatting(3DGS)의 등장으로 실시간 렌더링이 가능해졌으며, 이러한 방법들을 더욱 크고 복잡한 장면에 적용하려는 관심이 높아지고 있습니다. 그러나 표준 방법들은 대규모 환경으로의 확장성이 부족하며, 세밀한 디테일을 표현하기 위해서는 매우 많은 수의 Gaussian이 필요합니다.

기존의 대규모 장면을 위한 LOD 방법들은 렌더링 속도 향상에 주로 집중하면서도 GPU 메모리에 로드되는 Gaussian 수를 제한하지 않아 소형 디바이스에서의 렌더링이 어렵습니다. 또한 이러한 방법들은 새로운 프레임마다 렌더링에 사용할 Gaussian 부분집합을 재계산해야 하는 오버헤드가 발생하며, 더 나아가 모든 LOD에서의 Gaussian(심지어 3DGS보다 더 많은 수)을 항상 GPU 메모리에 올려두어야 합니다.

### 2-2. 제안하는 방법 (수식 포함)

#### (A) 계층적 LOD 표현

방법은 기본 3DGS 모델 $G^{(0)}$을 훈련하는 것으로 시작하며, 2D 안티에일리어싱 필터, 수정된 densification 전략, 중요도 기반 pruning 등의 개선 사항을 선택적으로 통합합니다. 이 기본 모델은 가장 세밀한 디테일로 장면을 표현하며, $L > l \geq 0$인 다수의 Gaussian 집합 $G^{(l)}$로 구성된 계층적 LOD 표현이 구축됩니다.

LOD 계층 구조의 핵심 원리는 다음과 같이 표현됩니다:

$$G^{(0)} \supseteq G^{(1)} \supseteq \cdots \supseteq G^{(L-1)}$$

여기서 $G^{(0)}$은 가장 상세한 기본 모델이며, $G^{(l)}$은 카메라로부터 최소 $d_l$ 거리에서 충분한 품질을 제공하도록 설계됩니다:

$$0 = d_0 < d_1 < d_2 < \cdots < d_{L-1}$$

카메라 위치 $\mathbf{c}$에서 렌더링 시 활성 LOD 레벨 $l^*$는 다음과 같이 결정됩니다:

$$l^* = \max\{l \mid d_l \leq \|\mathbf{c} - \mathbf{p}_{\text{scene}}\|\}$$

#### (B) Depth-Aware 3D Smoothing Filter

각 LOD 레벨은 Depth-aware 3D Smoothing Filter를 적용한 후, 시각적 충실도를 유지하기 위한 중요도 기반 Pruning과 Fine-tuning을 통해 구축됩니다.

깊이 인식 스무딩 필터는 카메라로부터의 거리 $d$에 비례하여 Gaussian의 공분산 행렬 $\mathbf{\Sigma}$를 조정합니다. 거리 $d$에서의 Gaussian $i$에 대한 스무딩은 다음과 같이 표현됩니다:

$$\mathbf{\Sigma}^{(l)}_i = \mathbf{\Sigma}^{(0)}_i + \sigma^2(d_l) \cdot \mathbf{I}$$

여기서 $\sigma^2(d_l)$는 LOD 레벨 $l$에 해당하는 거리 $d_l$에서의 허용 블러 수준을 나타내는 스케일 파라미터입니다.

#### (C) 중요도 기반 Pruning

LOD 표현의 일부로 RadSplat에서 도입된 중요도 Pruning을 적용합니다. 3DGS 훈련 중 많은 Gaussian이 투명도가 낮아지거나 앞의 Gaussian이 불투명해지면서 가시성이 감소합니다. 3DGS는 주기적으로 낮은 불투명도의 Gaussian을 제거하지만 다른 씬 지오메트리 뒤에 가려진 Gaussian은 제거하지 않습니다. RadSplat에서는 각 Gaussian의 중요도(중요도 점수 $\tau_i$)를 모든 훈련 카메라의 모든 픽셀에 대한 Gaussian의 기여도(알파 블렌딩에서의 렌더링 가중치)의 최댓값으로 측정합니다. 특정 임계값보다 낮은 중요도 점수를 가진 모든 Gaussian을 제거함으로써 렌더링에 거의 영향을 미치지 않는 Gaussian을 효과적으로 제거하여 렌더링 속도를 높이고 메모리를 줄일 수 있습니다.

이를 수식으로 표현하면:

$$\tau_i = \max_{\mathbf{r} \in \mathcal{R}} w_i(\mathbf{r})$$

$$G^{(l)} = \{g_i \in G^{(0)} \mid \tau_i \geq \tau_{\text{thresh}}^{(l)}\}$$

여기서 $w_i(\mathbf{r})$는 광선 $\mathbf{r}$에 대한 Gaussian $i$의 알파 블렌딩 기여도이며, $\mathcal{R}$는 모든 훈련 카메라의 광선 집합입니다.

#### (D) Densification 전략

Densification을 위해 H3DGS의 수정된 전략을 채택합니다. 구체적으로 클로닝/스플리팅에 사용되는 2D 위치 그래디언트 노름에 대한 원래의 하드 임계값을 다음 조건으로 대체합니다. `max_radii_2D`는 Gaussian이 투영되는 가장 큰 반지름(마지막 densification 이후)이며, 그래디언트 통계를 평균화하는 대신 최댓값을 취합니다.

$$\text{densify}_i = \left[\frac{\|\nabla_{\mathbf{x}_i^{2D}}\|}{r_{\max,i}^{2D}} > \tau_{\text{densify}}\right]$$

#### (E) 청크 기반 렌더링 + Opacity Blending

메모리 오버헤드를 더욱 줄이기 위해 씬을 공간 청크로 분할하고 렌더링 중에 관련 Gaussian만 동적으로 로드하며, 청크 경계에서의 시각적 아티팩트를 방지하기 위해 Opacity Blending 메커니즘을 사용합니다.

씬은 다중 LOD로 표현되며, 카메라 거리에 따라 훈련 중에 '활성 Gaussian'이 선택됩니다. 청크 기반 렌더링에서는 카메라를 청크로 군집화하고 청크별로 '활성 Gaussian'을 미리 계산한 후, 가장 가까운 두 청크를 'Opacity Blending'으로 렌더링합니다.

청크 간 전환 시 시간적 일관성을 보장하기 위해 Opacity Blending을 제안합니다.

두 인접 청크 $C_k$와 $C_{k+1}$ 사이의 전환 시 블렌딩은 다음과 같이 정의됩니다:

$$\alpha_k(t) = 1 - \phi\left(\frac{d(t) - d_{k+1}}{d_{k+1} - d_k}\right), \quad \alpha_{k+1}(t) = 1 - \alpha_k(t)$$

$$C_{\text{rendered}} = \alpha_k(t) \cdot C_k + \alpha_{k+1}(t) \cdot C_{k+1}$$

여기서 $d(t)$는 카메라의 현재 위치에서 각 청크 중심까지의 거리이며, $\phi(\cdot)$는 부드러운 전환을 위한 sigmoid 계열 함수입니다.

#### (F) 자동 임계값 선택

본 방법은 기존 방법들과 달리 각 프레임마다 사용할 Gaussian 목록을 재계산하지 않아 모바일 디바이스에서도 대규모 장면의 렌더링이 가능하며, 대부분의 다른 방법들이 각 3D 씬에 대해 수동으로 하이퍼파라미터를 조정해야 하는 것과 달리 최적의 LOD 분할 하이퍼파라미터를 자동으로 선택하는 전략도 설계합니다.

### 2-3. 모델 구조

LODGE의 전체 파이프라인을 도식화하면 다음과 같습니다:

```
[입력: 대규모 장면 이미지 + 카메라 포즈]
          │
          ▼
[기본 3DGS 학습: G^(0)]
  - Mip-Splatting 2D 필터
  - H3DGS 수정 Densification
  - RadSplat 중요도 Pruning
          │
          ▼
[계층적 LOD 구축: G^(1), ..., G^(L-1)]
  - Depth-aware 3D Smoothing Filter 적용
  - 중요도 기반 Pruning (τ_thresh^(l))
  - Fine-tuning
          │
          ▼
[공간 청크 분할]
  - 카메라 클러스터링으로 청크 경계 정의
  - 각 청크별 활성 Gaussian 사전 계산
          │
          ▼
[렌더링]
  - 카메라 거리 기반 LOD 레벨 선택
  - 가장 가까운 2개 청크 동적 로딩
  - Opacity Blending으로 부드러운 전환
          │
          ▼
[출력: 실시간 고품질 Novel View Synthesis]
```

기본 3DGS 표현 품질을 높이기 위해 최근 개선 사항들을 도입했으며, Mip-Splatting에서 제안된 2D 필터로 보강된 원래의 3DGS 렌더러를 사용합니다.

### 2-4. 성능 향상

H3DGS, OctreeGS, FLOD와 같은 최신 방법들과 비교할 때, LODGE는 렌더링 속도(FPS)를 크게 향상시키고 GPU 메모리에 로드되는 Gaussian 수를 줄이면서도 경쟁력 있거나 우수한 렌더링 품질(PSNR, SSIM, LPIPS)을 달성합니다.

Ablation 연구는 개별 구성 요소의 기여를 검증하며, 다중 LOD 레벨, 자동 임계값 선택, 가시성 필터링을 통한 청크 기반 렌더링, Opacity Blending 각각이 성능 및/또는 품질을 향상시킴을 보여줍니다.

특히, LODGE는 다른 대규모 3DGS 방법들이 로드하거나 실시간으로 렌더링하지 못하는 메모리 제약이 있는 모바일 디바이스(iPhone, MacBook Air, Chromebook)에서 실시간 렌더링 성능을 달성합니다.

LODGE는 대규모 3D 장면의 효율적인 렌더링을 위한 새로운 방법으로, 씬에 대해 여러 디테일 레벨을 생성하여 렌더링 성능을 최적화하며, 시각적 품질을 유지하면서 최대 4배 빠른 렌더링을 달성합니다.

### 2-5. 한계

본 방법은 모바일 디바이스에서 실시간 렌더링을 가능하게 하지만, Gaussian의 로딩—그리고 청크 경계를 넘을 때의 재로딩—이 효율적으로 수행될 수 있다고 가정합니다.

모바일 디바이스에의 배포를 위해서는 청크 전환 시 끊김 없이 처리하기 위해 서버/저장소에서 디바이스의 GPU 메모리로 Gaussian을 효율적으로 비동기 로딩/스트리밍하고 압축하는 것이 필요합니다.

매우 대규모 장면의 경우 디테일 레벨을 생성하기 위한 전처리 시간이 상당할 수 있으며, 빠른 카메라 움직임 시 디테일 레벨 전환 중 아티팩트가 나타날 수 있습니다. 또한 연구는 다양한 씬 유형에 대한 더 많은 테스트와 다른 최적화 기법과의 비교로부터 혜택을 받을 수 있습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

LODGE의 일반화 가능성은 다음 측면에서 평가할 수 있습니다.

### 3-1. 설계상의 일반화 요인

기존 방법들과 달리 LODGE의 방법은 표준 재구성 위에 LOD 구조를 구축하며, 다양한 기존 방법에 적용할 수 있습니다. 또한 coarse-to-fine 전략은 초기 집합이 너무 sparse할 경우 densification이 실패하기 때문에 대규모 장면에서는 실패하는 경향이 있습니다.

실외(Hierarchical 3DGS 데이터셋)와 실내(Zip-NeRF 데이터셋)의 두 개 대규모 데이터셋을 사용하여 방법을 검증합니다.

**다양한 씬 타입 지원:**
대규모 3D 장면에 적용된 LODGE는 우수한 렌더링 속도를 유지하면서 뛰어난 품질을 달성합니다. 또한 모바일 디바이스에서의 실시간 렌더링도 가능합니다.

### 3-2. 일반화를 방해하는 요인

1. **씬 종속적 청크 분할**: 공간 청크의 경계와 크기는 씬마다 달리 설정될 수 있으며, 완전히 새로운 씬 타입(예: 동적 장면)에서는 사전 계산된 청크가 유효하지 않을 수 있습니다.

2. **훈련 뷰 의존성**: 중요도 점수 $\tau_i$는 훈련 카메라의 관측에 기반하기 때문에 훈련 분포 밖의 뷰포인트에서는 중요한 Gaussian이 pruning될 위험이 있습니다.

3. **디바이스 의존성**: 자동 임계값 선택 전략은 씬에 대해 자동화되었지만, 타겟 디바이스의 메모리/연산 제약에 따라 최적 설정이 달라집니다.

### 3-3. 일반화 향상을 위한 제안 방향

| 방향 | 설명 |
|---|---|
| 동적 장면 확장 | 시간적 Gaussian 변화를 수용하는 4D LOD 표현 |
| 학습 기반 임계값 | 씬의 복잡도를 학습하여 자동으로 LOD 수 결정 |
| 압축 통합 | Gaussian 압축(양자화)과 LOD를 결합한 더 낮은 비트레이트 지원 |
| 도메인 적응 | 항공/위성/도시 등 다양한 도메인의 씬에 대한 자동 파라미터 적응 |

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 방법 | 연도 | 주요 특징 | LODGE와의 차이 |
|---|---|---|---|
| **NeRF** (Mildenhall et al.) | 2020 | 암묵적 neural radiance field, 고품질 | 실시간 렌더링 불가 |
| **3DGS** (Kerbl et al.) | 2023 | 명시적 Gaussian 표현, 실시간 | 대규모 장면 확장성 없음 |
| **Hierarchical 3DGS (H3DGS)** | 2024 | 청크 독립 훈련 후 계층 병합 | 매 프레임 활성 Gaussian 재계산 필요 |
| **Octree-GS** | 2024 | 옥트리 구조 LOD, 공간 분할 계층 | GPU 메모리에 전체 LOD 유지 필요 |
| **FLoD** | 2024 | 유연한 LOD, 다양한 GPU 설정 지원 | 프레임별 재계산 오버헤드 존재 |
| **LODGE (본 논문)** | 2025 | 청크 기반 사전 계산, Opacity Blending | 프레임별 재계산 없음, 모바일 지원 |
| **CLoD-GS** | 2025 | 연속 LOD (Continuous LoD) | 이산이 아닌 연속 전환 제공 |
| **A LoD of Gaussians** | 2025 | Sequential Point Tree, External Memory | 청크 분할 없이 초대규모 스트리밍 |

3DGS 맥락에서 LOD 접근법은 메모리 제약이 있거나 모바일 디바이스에서의 효율적인 렌더링을 가능하게 하기 위해 탐구되었으며, 압축 기반 전략으로는 코드북을 통한 속성 양자화, 낮은 영향력의 Gaussian Pruning, 프리미티브당 구면 조화함수 차수 조정 등이 있습니다.

이 개념은 Octree-GS에서 계층적 LOD 렌더링으로 확장되어 공간 세분화를 통한 디테일 레벨의 실시간 제어를 가능하게 합니다.

H3DGS와 CLOG는 연속적인 다중 레벨 표현을 구축하며, FLoD와 OctreeGS는 이산적인 레벨을 사용하는데(본 논문의 접근법과 유사), 각 레벨은 Gaussian 집합입니다. 이러한 방법들은 coarse한 Gaussian 집합에서 시작하여 더 세밀한 레벨을 얻기 위해 점진적으로 densification합니다.

기존의 모든 LOD 방법들은 렌더링된 각 프레임에 대해 동적으로 '활성 Gaussian' 집합을 선택하는데, 이는 렌더링을 느리게 하고 모든 Gaussian이 GPU 메모리에 로드되어 있어야 합니다.

여러 연구들이 DLoD 철학을 3DGS에 적용하여 계층적 표현을 생성했으며, 대표적인 예로 LODGE(Kulhanek et al., 2025)는 스무딩 필터와 중요도 기반 pruning을 반복적으로 적용하여 여러 개의 이산적인 Gaussian 집합을 생성합니다.

---

## 5. 향후 연구에 미치는 영향 및 고려사항

### 5-1. 향후 연구에 미치는 영향

**① 모바일/엣지 3D 렌더링 패러다임 전환**

실내 및 실외 데이터셋 모두에 대한 광범위한 실험은 본 방법이 렌더링 품질과 속도 면에서 최신 기준선을 능가함을 보여줍니다. 특히 중요한 것은 본 방법이 모바일 디바이스에 배포 가능하여 다른 방법들이 실패하는 곳에서 실시간 성능을 달성한다는 점입니다.

이는 AR/VR 헤드셋, 스마트폰 내비게이션, 디지털 트윈 등 엣지 디바이스 기반 3D 응용 연구를 직접적으로 촉진합니다.

**② 스트리밍 및 압축 연구 촉진**

모바일 디바이스 배포를 위해서는 서버/저장소에서 디바이스 GPU 메모리로의 효율적인 비동기 로딩/스트리밍 및 Gaussian 압축이 필요합니다.

이는 Gaussian 압축, 비동기 스트리밍, 네트워크 대역폭 적응형 품질 조절 등의 연구 방향을 열어줍니다.

**③ 관련 후속 연구의 직접적 영향**

LODGE와 관련하여 이미 "FLoD: Integrating Flexible Level of Detail into 3D Gaussian Splatting for Customizable Rendering (2024)", "Virtualized 3D Gaussians: Flexible Cluster-based Level-of-Detail System for Real-Time Rendering of Composed Scenes (2025)", "A LoD of Gaussians: Unified Training and Rendering for Ultra-Large Scale Reconstruction with External Memory (2025)" 등의 후속 연구들이 이어지고 있습니다.

### 5-2. 향후 연구 시 고려할 점

**① 동적 씬 대응**

미래 연구는 자동 파라미터 튜닝과 동적 씬의 더 나은 처리를 탐구할 수 있습니다.

현재 LODGE는 정적 씬을 전제로 청크를 사전 계산하므로, 움직이는 물체나 시간에 따라 변화하는 장면에서는 사전 계산 구조가 무효화될 수 있습니다. 4D Gaussian Splatting과의 결합이 필요합니다.

**② 압축과 LOD의 통합**

압축 기법은 일반적으로 세 가지 중복되는 카테고리로 분류됩니다: Gaussian 수를 줄이는 컴팩션, 프리미티브당 저장 공간을 줄이는 속성 압축, 압축률을 높이기 위해 데이터를 정리하는 구조적 표현입니다.

LOD와 압축을 통합하면 모바일 환경에서의 스트리밍 효율성을 더욱 극대화할 수 있습니다.

**③ 자동 씬 분석 기반 적응형 청킹**

청크 분할 전략이 씬의 기하학적 특성(예: 공간 밀도, 중요 랜드마크 분포)을 자동으로 분석하여 최적 경계를 설정하는 학습 기반 접근이 필요합니다.

**④ 훈련 뷰 외 일반화**

중요도 점수가 훈련 카메라에만 의존하는 문제를 극복하기 위해, 훈련 시 보지 못한 뷰포인트에서도 견고한 Gaussian 중요도를 평가하는 방법(예: 3D 공간 커버리지 기반 중요도 계산)이 필요합니다.

**⑤ 연속 LOD로의 발전**

Fast 3D Gaussian Splatting Rendering using Continuous Level of Detail 연구는 3DGS가 실시간 photo-realistic 3D 장면 렌더링의 가능성을 보여주지만 덜 강력한 하드웨어에서 렌더링이 여전히 과제임을 지적하며, Continuous LOD(CLOD) 알고리즘을 도입하여 품질을 최대한 보존하면서 성능을 향상시킵니다. 이 접근법은 중요도 기반으로 splat을 순서화하여 임의의 splat 수에 대해 대표적이고 사실적인 씬을 렌더링할 수 있도록 최적화합니다.

이처럼 이산 LOD(LODGE)와 연속 LOD(CLoD-GS, CLOD)를 결합하는 하이브리드 접근법이 향후 유망한 연구 방향입니다.

---

## 📚 참고 자료 (출처 목록)

| # | 제목 / 링크 | 비고 |
|---|---|---|
| 1 | [arxiv.org/abs/2505.23158](https://arxiv.org/abs/2505.23158) — LODGE 원문 (arXiv) | 논문 원문 |
| 2 | [arxiv.org/html/2505.23158v2](https://arxiv.org/html/2505.23158v2) — LODGE HTML 버전 (v2) | 상세 방법론 |
| 3 | [lodge-gs.github.io](https://lodge-gs.github.io/) — LODGE 공식 프로젝트 페이지 | 시각 자료 |
| 4 | [openreview.net/forum?id=Iqu63cYI3z](https://openreview.net/forum?id=Iqu63cYI3z) — OpenReview (NeurIPS 2025 Spotlight) | 리뷰 및 메타 정보 |
| 5 | [neurips.cc/virtual/2025/poster/118763](https://neurips.cc/virtual/2025/poster/118763) — NeurIPS 2025 포스터 | 발표 정보 |
| 6 | [semanticscholar.org — LODGE](https://www.semanticscholar.org/paper/LODGE:-Level-of-Detail-Large-Scale-Gaussian-with-Kulh%C3%A1nek-Rakotosaona/6d117b0156a3f43b3cf9fd896bb92edea9715820) | 인용 관계 |
| 7 | [themoonlight.io — LODGE 리뷰](https://www.themoonlight.io/en/review/lodge-level-of-detail-large-scale-gaussian-splatting-with-efficient-rendering) | 문헌 리뷰 요약 |
| 8 | [aimodels.fyi — LODGE](https://www.aimodels.fyi/papers/arxiv/lodge-level-detail-large-scale-gaussian-splatting) | 성능 요약 |
| 9 | [arxiv.org/html/2507.01110v1](https://arxiv.org/html/2507.01110v1) — "A LoD of Gaussians" (후속 연구) | 관련 연구 비교 |
| 10 | [arxiv.org/html/2510.09997v1](https://arxiv.org/html/2510.09997v1) — "CLoD-GS: Continuous Level-of-Detail via 3DGS" | 관련 연구 비교 |
| 11 | emergentmind.com — [LODGE 관련 LOD 연구 맥락](https://www.emergentmind.com/topics/lod-aware-rendering-strategy) | 연구 맥락 |

> ⚠️ **정확도 주의사항**: 본 답변에서 제시된 수식 중 일부(LOD 레벨 선택 기준, Opacity Blending 수식, Densification 조건식 등)는 논문의 서술적 설명을 바탕으로 일반화된 형태로 표현한 것입니다. 정확한 수식의 세부 파라미터 정의는 [arXiv 원문 전문](https://arxiv.org/abs/2505.23158)을 직접 확인하시기를 강력히 권장합니다. 공개된 검색 결과로는 모든 수식의 정확한 표기를 완전히 재현하기 어렵습니다.
