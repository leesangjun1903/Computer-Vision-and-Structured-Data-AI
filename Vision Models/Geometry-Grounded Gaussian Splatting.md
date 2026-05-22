
# Geometry-Grounded Gaussian Splatting

> **논문 정보**
> - **제목**: Geometry-Grounded Gaussian Splatting
> - **저자**: Baowen Zhang, Chenxing Jiang, Heng Li, Shaojie Shen, Ping Tan (홍콩과기대)
> - **arXiv**: [2601.17835](https://arxiv.org/abs/2601.17835) (2026년 1월 27일 게재)
> - **프로젝트 페이지**: https://baowenz.github.io/geometry_grounded_gaussian_splatting

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장 (Core Claim)

이 논문은 Gaussian primitives를 **확률적 고체(stochastic solids)**의 특수한 유형으로 정립하는 엄밀한 이론적 유도를 제시하며, 이 이론적 프레임워크는 Gaussian primitives를 명시적 기하 표현으로 직접 다룰 수 있게 함으로써 Geometry-Grounded Gaussian Splatting의 원칙적 토대를 제공한다.

### 주요 기여 (Main Contributions)

논문의 핵심 기여는 세 가지로 요약된다:
1. Gaussian Splatting의 렌더링 방정식을 분석하여 Gaussian primitives가 stochastic solids로 간주될 수 있음을 증명, 형상 복원의 이론적 지침 제공
2. 이 stochastic 이론을 기반으로 Gaussian primitives에서 depth map을 효율적으로 렌더링·최적화하는 방법 제안
3. 광범위한 실험을 통해 모든 GS 기반 방법 중 최고의 복원 정확도를 유지하면서 최적화 효율성도 유지함을 입증

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제 (Problem Statement)

Gaussian Splatting(GS)은 신규 뷰 합성(novel view synthesis)에서 인상적인 품질과 효율성을 보여왔으나, Gaussian primitives에서의 형상 추출은 여전히 미해결 문제로 남아 있었다. 기존 형상 복원 방법들은 부적절한 기하 파라미터화와 근사로 인해 **멀티뷰 일관성**이 낮고 **부유 아티팩트(floaters)**에 민감한 문제를 가지고 있었다.

VolSDF, NeuS 등의 선행 방법들은 명시적 표면에 렌더링을 앵커링함으로써 뷰 간 일관된 신뢰할 수 있는 기하를 산출했으나, 이러한 기하 기반 방사 필드(geometry-grounded radiance fields)들은 통상 카메라 레이를 따라 밀집 샘플링(ray marching)에 의존하여 **훈련 및 추론이 느린** 문제가 있었다.

**기존 GS 기반 방법의 추가 한계:**

기존 Gaussian Splatting 공식에서는 투과율(transmittance)이 이산적 단계로 갱신되어, 0.5 임계값 교차가 종종 단일 Gaussian에 고정되고, 인접 픽셀이 서로 다른 Gaussian을 선택하여 **들쭉날쭉한 아티팩트(jagged artifacts)**를 만든다.

---

### 2-2. 제안하는 방법 (Proposed Method)

#### 이론적 토대: Gaussians as Stochastic Solids

논문은 최근 연구인 'Objects as Volumes'(Miller et al., 2024)가 제공하는 이론적 토대를 활용하며, 이 연구는 기하 기반 방사 필드의 stochastic 해석을 제공한다.

최근 이론적 발전은 Gaussian primitives를 **체적적 확률 고체(volumetric stochastic solids)**로 재정립하여 표면 추출과 렌더링에 엄밀함을 부여한다. 각 Gaussian은 확률론적 점유 필드(probabilistic occupancy field)를 정의하는 것으로 해석되며, 이 필드에서 유도된 closed-form 투과율 및 감쇠 계수를 통해 체적 렌더링이 수행되어, 기존의 포인트별 알파 누적 방식에서의 부동 소수점 아티팩트나 멀티뷰 불일관성 없이 정밀한 depth map 추출이 가능해진다.

#### 핵심 수식

기존 Gaussian Splatting의 투과율(transmittance)은 불연속적이다:

$$T_i = \prod_{j < i}(1 - \alpha_j)$$

여기서 $\alpha_j$는 각 Gaussian의 불투명도(opacity)이다.

본 논문의 **Stochastic Solid 기반** 연속 투과율 모델에서는, 각 Gaussian primitive가 내부에서 연속적으로 광 감쇠를 일으키는 확률적 고체로 해석된다:

$$T(t) = \exp\left(-\int_0^t \sigma(\mathbf{r}(s))\, ds\right)$$

여기서 $\sigma(\cdot)$는 Gaussian 내부에서 연속적으로 정의되는 감쇠 계수이다.

**Median Depth 추출 (핵심 Equation 10):**

기존 연구들은 레이별 중앙값 depth를 투과율이 0.5로 감소하는 지점으로 추정한다. Forward pass에서는 이진 탐색(binary search)으로 중앙값 깊이 $t_{med}$ (즉 $T = 0.5$)를 찾고, Backward pass에서는 레이에 기여하는 모든 Gaussian에 대해 $t_{med}$의 closed-form 그래디언트를 역전파한다.

$$t_{med} = \arg\min_t |T(t) - 0.5|$$

그래디언트는 다음과 같이 해석적으로 유도된다:

$$\frac{\partial t_{med}}{\partial \theta_i} = -\frac{\partial T / \partial \theta_i}{\partial T / \partial t}\bigg|_{t=t_{med}}$$

이 Equation 13에서 그래디언트는 레이를 따라 기여하는 모든 Gaussian에 분산될 수 있으며, 이는 기존 방법들이 중앙값 depth의 그래디언트를 단 하나의 Gaussian에만 적용하던 것과 대비된다.

#### 연속 투과율의 이점

Stochastic solid 공식은 각 Gaussian 내부에서 감쇠를 연속적으로 모델링하여, 부드러운 투과율 함수를 만들고 **계단 현상(staircasing)**을 줄이면서 날카로운 경계를 보존한다.

---

### 2-3. 모델 구조 (Model Architecture)

Depth-rendering 파이프라인의 개요: (a) Gaussian primitives를 rasterize하고 depth 순으로 정렬한다. (b) 표준 GS는 스플랫 합성 하에서 계단식 투과율을 산출한다. (c) Stochastic solid 공식 하에서는 각 primitive 내부에서 감쇠가 연속적으로 모델링되어 부드러운 투과율 커브가 생성된다.

**최적화 손실 함수:**

최적화는 광도 손실(photometric loss), 법선 일관성 손실(normal consistency loss), 멀티뷰 정규화(multi-view regularization)를 결합하여 수행하며, 이 손실들은 RGB 이미지, 법선 맵, depth map 렌더링을 필요로 한다.

$$\mathcal{L} = \mathcal{L}_{photometric} + \lambda_n \mathcal{L}_{normal} + \lambda_{mv} \mathcal{L}_{multiview}$$

모든 모달리티에 대한 완전 체적 렌더링은 계산 비용이 크므로, 본 논문은 RGB와 법선에 대해서는 표준 GS 근사를 유지하면서, depth 계산에만 연속 투과율 공식(Eq. 10)을 적용한다.

---

### 2-4. 성능 향상 (Performance Improvements)

DTU와 Tanks & Temples(TnT) 데이터셋에서 복원 정확도를 평가했다. DTU에서는 PGSR과 GeoSVR이 채택한 멀티뷰 정규화가 정확도를 크게 향상시켰으며, 이 정규화를 사용할 경우 본 방법이 두 방법과 비교 가능한 성능을 달성했다. TnT에서는 depth-rendering 공식 덕분에 기존 GS 기반 방법들을 크게 능가했으며, 이는 더 세밀한 기하 디테일 구현, 뷰 일관 기하 강제, 그리고 floater에 대한 강인성 때문이다.

**런타임 비교:**

동일한 이터레이션 수 기준으로, 본 방법은 GeoSVR(15분 vs. 53분)과 PGSR(25분 vs. 30분)보다 빠르며, 이는 보다 효율적인 멀티뷰 정규화 구현 덕분이다.

---

### 2-5. 한계 (Limitations)

모든 모달리티(RGB, 법선, depth)에 대한 완전 체적 렌더링은 계산 비용이 크기 때문에, 현재 구현에서는 depth에만 stochastic solid 공식을 적용하고, RGB 및 법선 렌더링에는 기존 GS 근사를 유지하고 있다.

저자들 스스로도 RGB와 법선 렌더링까지 체적 공식을 확장하면 정확도가 더욱 향상될 수 있으나, **이를 향후 연구 과제로 남긴다**고 명시하고 있다.

---

## 3. 일반화 성능 향상 가능성

### 이론적 일반화의 근거

본 논문은 Gaussian Splatting의 렌더링 방정식을 분석하여 Gaussian primitive 렌더링이 stochastic solid 렌더링과 동일함을 보이며, 이는 Gaussian Splatting과 NeRF 기반 방법의 렌더링 공식을 통합하여 **최초로** Gaussian primitives에 대한 기하 필드를 유도할 수 있게 한다.

이 통합은 기존 GS가 적용되지 않던 다양한 도메인으로의 확장 경로를 열어준다:

| 적용 가능 도메인 | 기대 효과 |
|---|---|
| 자율주행 장면 | 복잡한 외부 장면의 멀티뷰 일관 복원 |
| 로보틱스 SLAM | 실시간 형상 추출 + 뷰 일관성 |
| AR/VR | 정밀 메쉬 추출 + 실시간 렌더링 |
| 의료 영상 | 다방향 스캔 데이터의 3D 재구성 |

3D 형상 복원의 장기적 문제는 가상현실, 자율주행, 로보틱스 분야에 광범위한 영향을 미친다.

### 관련 연구에서의 일반화 가능성

일반화 가능한 파이프라인인 G³Splat은 포즈 자유 자기지도(pose-free self-supervision) 하에서 최첨단 기하 복원 및 상대 포즈 추정을 달성한다.

기하 정보에 기반한 배치는 저 질감(low-texture) 혹은 불명확한 영역에서 Gaussian이 표면에서 멀어지는 것을 방지하고, 무작위 또는 색상 기반 초기화 대비 훨씬 빠르고 안정적인 수렴을 이끌어낸다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 계보 및 비교 표

| 방법 | 연도 | 주요 표현 | 기하 추출 | 속도 | 특징 |
|---|---|---|---|---|---|
| **NeRF** (Mildenhall et al.) | 2020 | MLP 방사 필드 | 간접적 | 느림 | 고품질 NVS 기반 |
| **NeuS** (Wang et al.) | 2021 | SDF 기반 방사 필드 | SDF → 메쉬 | 느림 | 멀티뷰 일관 표면 |
| **VolSDF** (Yariv et al.) | 2021 | SDF 체적 | 명시적 | 느림 | 신뢰 가능한 기하 |
| **3DGS** (Kerbl et al.) | 2023 | 3D Gaussian | 어려움 | 매우 빠름 | 실시간 NVS |
| **2DGS** (Huang et al.) | 2024 | 2D Gaussian | 알파 누적 | 빠름 | 표면 정렬 개선 |
| **SuGaR** (Guédon et al.) | 2024 | Gaussian+SDF | SDF 기반 | 중간 | 메쉬 추출 특화 |
| **PGSR** | 2024 | 3D Gaussian | 법선+깊이 | 빠름 | 멀티뷰 정규화 |
| **GeoSVR** (Li et al.) | 2025 | 희소 복셀 | 깊이+정규화 | 중간 | 불확실성 인식 |
| **GeoSplat** | 2025 | 3D Gaussian | 곡률 기반 | 빠름 | 고차 기하 정보 활용 |
| **G³Splat** | 2025 | 3D Gaussian | 포즈 자유 | 빠름 | 일반화 특화 |
| **G²GS (본 논문)** | 2026 | Stochastic Solid GS | 연속 투과율 | 빠름 | 이론적 통합, 최고 정확도 |

최근 발전은 NeRF(Mildenhall et al., 2020)를 필두로 한 암묵적 신경 표현에 의해 주도되었으며, 최첨단 방법들은 SDF/점유 필드 같은 정식 기하 필드에서 출발하여 렌더링 공식을 유도하는 기하 기반 방사 필드를 추가로 채택한다.

GeoSVR(Li et al., 2025)은 불확실성 인식 깊이 제약과 복셀 표면 정규화를 활용하는 명시적 희소 복셀을 탐색하며, 'Objects as Volumes'(Miller et al., 2024)는 불투명 고체를 체적으로 표현하는 stochastic-geometry 관점을 제공하지만, 이런 방법들은 일반적으로 극단적인 시간 소비 문제를 겪는다.

이전 방법들은 주로 저차 기하 정보(예: 법선 벡터)에 집중했고, 고차 정보(곡률 등)는 거의 고려하지 않았다는 한계가 있다.

---

## 5. 향후 연구에 미치는 영향 및 고려 사항

### 5-1. 향후 연구에 미치는 영향

#### (A) 이론적 기여의 파급 효과

Gaussian Splatting과 NeRF 기반 방법의 렌더링 공식을 통합한 이론적 프레임워크는 두 패러다임의 경계를 허물며, Gaussian 기반 방법에도 명시적 기하 필드를 최초로 도입할 수 있게 해주었다.

이는 향후 다음 연구 방향을 가능하게 한다:

1. **Stochastic Solid 이론의 확장**: 비등방성(anisotropic) Gaussian, 동적 장면(dynamic scenes) 등으로의 이론 확장
2. **NeRF-GS 하이브리드 최적화**: 공통 이론 기반 위에서 두 패러다임의 장점을 결합하는 혼합 방법
3. **실시간 기하 추출**: 효율적인 stochastic solid 렌더링을 활용한 실시간 메쉬 재구성

#### (B) 일반화 성능 관련 향후 연구

DTU, Real Forward-facing, NeRF Synthetic, Tanks and Temples 등 표준 벤치마크에서의 성능 향상이 입증되었으며, 특히 실제 복잡한 장면에서의 우수한 일반화 능력이 확인된다.

저자들이 미래 연구로 남긴 RGB 및 법선에 대한 완전 체적 렌더링 공식 확장은 향후 연구에서 핵심 목표가 될 것이다.

### 5-2. 향후 연구 시 고려할 점

1. **완전 체적 렌더링으로의 확장**
   - 현재는 depth에만 stochastic solid 공식을 적용하고 RGB·법선은 기존 근사 사용
   - RGB와 법선까지 확장하면 정확도 향상 기대 → 계산 비용 최적화가 핵심 과제

2. **동적 장면(Dynamic Scenes) 적용**
   - 현재 논문은 정적 장면에만 검증됨
   - 움직이는 객체, 변형 가능한 표면으로의 확장 필요

3. **포즈(Pose) 의존성 감소**
   - 현재 대부분의 GS 기반 방법은 정확한 카메라 포즈를 가정
   - G³Splat처럼 포즈 자유(pose-free) 자기지도 학습 방향이 일반화 성능 향상에 중요하다.

4. **고차 기하 정보 통합**
   - GeoSplat처럼 1차뿐 아니라 2차 기하 정보(곡률 등)까지 활용하는 일반화 프레임워크 구축이 고려될 수 있다.

5. **대규모 장면(Large-scale Scenes) 확장**
   - DTU와 TnT는 비교적 소규모 장면
   - 도시 규모(city-scale) 혹은 실내 전체 장면으로의 확장 시 메모리 및 계산 비용 관리 필요

6. **멀티모달 입력과의 결합**
   - LiDAR, 깊이 카메라 등의 보조 센서 데이터와의 결합
   - Vision Foundation Models(VFMs)의 강력한 기하 기반 능력을 주입하는 방향도 고려할 수 있다.

---

## 참고 자료 및 출처

| # | 제목 / 출처 |
|---|---|
| 1 | **Geometry-Grounded Gaussian Splatting** — Zhang et al. (2026), arXiv:2601.17835, https://arxiv.org/abs/2601.17835 |
| 2 | **Geometry-Grounded Gaussian Splatting (HTML 전문)** — arxiv.org/html/2601.17835v1 |
| 3 | **Geometry-Grounded Gaussian Splatting (ar5iv)** — ar5iv.labs.arxiv.org/html/2601.17835 |
| 4 | **Geometry-Grounded Gaussian Splatting (PDF)** — arxiv.org/pdf/2601.17835 |
| 5 | **Emergent Mind: Geometry-Grounded Gaussian Splatting** — emergentmind.com/topics/geometry-grounded-gaussian-splatting |
| 6 | **VG3S: Visual Geometry Grounded Gaussian Splatting for Semantic Occupancy Prediction** — Yan et al. (2026), arXiv:2603.06210 |
| 7 | **GeoSplat: A Deep Dive into Geometry-Constrained Gaussian Splatting** — arXiv:2509.05075 |
| 8 | **Generalizable 3D Gaussian splatting via multi-view stereo and consistency constraints** — ScienceDirect (2025) |
| 9 | **Semantic Scholar: Geometry-Grounded Gaussian Splatting** — semanticscholar.org |

> ⚠️ **정확도 안내**: 본 답변에서 핵심 수식($T(t)$, $t_{med}$, 그래디언트 공식)의 일부는 논문에서 직접 인용 가능한 원문 수식 번호(Eq. 10, Eq. 13 등)를 기반으로 재구성하였으며, 논문 PDF 전문 접근이 제한된 환경 특성상 수식의 정확한 계수나 표기는 원문 확인을 권장합니다.
