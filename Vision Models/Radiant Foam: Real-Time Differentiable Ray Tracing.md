
# Radiant Foam: Real-Time Differentiable Ray Tracing

> **논문 정보**
> - **저자**: Shrisudhan Govindarajan\*, Daniel Rebain\*, Kwang Moo Yi, Andrea Tagliasacchi (\*equal contribution)
> - **소속**: Simon Fraser University, University of British Columbia, University of Toronto, Google DeepMind
> - **게재**: ICCV 2025 (pp. 4135–4145)
> - **arXiv**: [arXiv:2502.01157](https://arxiv.org/abs/2502.01157) (2025년 2월 3일)
> - **프로젝트 페이지**: https://radfoam.github.io/
> - **코드**: https://github.com/theialab/radfoam

---

## 1. 핵심 주장 및 주요 기여 (요약)

### 🎯 핵심 주장

최근 미분 가능 장면 표현 연구는 Splatting 방법의 보급으로 이어졌는데, 이는 레이 기반 렌더링 대신 래스터화(rasterization)를 채택하여 렌더링 속도를 크게 향상시켰다. 그러나 래스터화를 효율적으로 만드는 근사(approximation)는 반사(reflection)나 굴절(refraction) 같은 광 전달 현상(light transport phenomena)의 구현을 매우 어렵게 만드는 대가를 치렀다.

이에 Radiant Foam은 다음을 핵심 주장으로 내세웁니다:

> 래스터화의 근사를 피하면서도, 수십 년 된 효율적인 체적 메시(volumetric mesh) 레이 트레이싱 알고리즘을 활용하여 Splatting의 효율성과 재구성 품질을 유지하는 새로운 장면 표현을 제안한다. 결과 모델인 Radiant Foam은 래스터화의 제약 없이 Gaussian Splatting과 유사한 렌더링 속도와 품질을 달성하며, 하드웨어 레이 트레이싱 가속을 사용하는 레이 트레이스 Gaussian 모델과 달리 표준 프로그래머블 GPU 이상의 특수 하드웨어나 API가 필요 없다.

### 🏆 주요 기여 요약

| 기여 | 설명 |
|---|---|
| **새로운 장면 표현** | 3D Voronoi 다이어그램 기반의 폼(foam) 구조 |
| **실시간 차분 레이 트레이싱** | 특수 하드웨어 없이 GPU에서 동작 |
| **미분 가능성** | Voronoi 사이트 위치의 미분 가능 최적화 |
| **Coarse-to-Fine 학습** | 적응형 해상도 메시 구성 전략 |
| **광 전달 효과** | 반사, 굴절, 비선형 카메라 모델 지원 |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

**① Splatting의 구조적 한계**

래스터화 알고리즘과 하드웨어의 효율성 덕분에 렌더링 속도가 크게 향상되었지만, 래스터화를 효율적으로 만드는 근사가 반사, 굴절 같은 광 전달 현상의 구현을 어렵게 만든다. 이에 이러한 근사를 피하면서도 컴퓨터 비전 연구에서 최근 간과되어 온 효율적인 체적 메시 레이 트레이싱 알고리즘을 활용한 새로운 장면 표현을 제안한다.

**② BVH 기반 레이 트레이싱의 한계**

Gaussian 프리미티브는 구조화되지 않고 심하게 겹치기 때문에 공간의 진정한 체적 분할을 정의하지 못한다. 따라서 이 장면들의 레이 트레이싱에는 BVH(Bounding Volume Hierarchy)의 구성과 탐색이 필요하다. 하드웨어 가속이 이 기법을 사용한 실시간 레이 트레이싱을 가능하게 했지만, 레이 탐색의 복잡도를 장면의 프리미티브 수에 근본적으로 결합시킨다. 반면 Radiant Foam은 체적 메시를 통해 공간의 명시적 분할을 모델링하여 중복을 제거하고, BVH의 로그 복잡도에서 **상수 시간(constant time)** 복잡도로 레이 탐색을 줄인다.

---

### 2-2. 제안 방법 및 수식

#### 🔷 핵심 표현: Voronoi 다이어그램

이 표현은 3D 공간의 밀도 있는 Voronoi 분할(Voronoi tessellation)에 불과하며, 각 점은 정확히 하나의 Voronoi 셀에 속한다. Voronoi 사이트의 위치는 미분 가능하여 경사 기반 최적화에 적합하다.

**Voronoi 셀 정의**: 점 집합 $\mathbf{P} = \{p_1, p_2, \ldots, p_N\} \subset \mathbb{R}^3$이 주어졌을 때, 각 사이트 $p_i$에 대한 Voronoi 셀은:

$$V_i = \{ x \in \mathbb{R}^3 \mid \|x - p_i\| \leq \|x - p_j\|, \; \forall j \neq i \}$$

이는 Delaunay 삼각분할의 쌍대(dual)로 구성된다.

#### 🔷 레이 트레이싱 알고리즘

레이(빨간색)가 셀에 진입하면, 레이가 빠져나가는 면을 식별하기 위해 모든 평면 셀 면을 반복한다. 이 출구 교차점은 레이 방향과 법선 벡터가 90도 미만인 최초 교차점(앞면)이며, 다른 교차점(뒷면)은 무시된다. 각 면이 이웃 셀에 대응하므로, 출구 교차점과 연관된 셀로 진입하여 과정을 반복함으로써 추적이 진행된다.

이 알고리즘은 로그 쿼리 복잡도를 가진 계층적 가속 구조에 의존하는 일반적인 레이 트레이싱 가속 방법보다 훨씬 더 효율적이다.

#### 🔷 체적 렌더링 수식

각 레이는 순서대로 $N$개의 Voronoi 셀을 통과하며, 표준 체적 렌더링 적분에 따라 색상을 누적한다:

$$C(\mathbf{r}) = \sum_{n=1}^{N} T_n \cdot \alpha_n \cdot \mathbf{c}_n$$

여기서:
- $T_n = \prod_{k=1}^{n-1}(1 - \alpha_k)$ : 누적 투과율(transmittance)
- $\alpha_n = 1 - \exp(-\sigma_n \cdot \delta_n)$ : 불투명도(opacity), $\sigma_n$은 셀의 밀도, $\delta_n$은 셀 내 레이 구간 길이
- $\mathbf{c}_n$ : 셀 $n$의 뷰 의존적 색상 (구면 조화 함수 SH로 표현)

밀도 필드는 차폐(occlusion)를 생성하고, 복사 필드는 관측된 빛의 밝기를 결정한다. 체적 렌더링은 시점과 밀도 및 복사 필드값을 포함한 모든 자유도에 대해 완전히 연속적이며, 이 특성이 경사 기반 최적화에 매우 적합하게 만든다.

#### 🔷 뷰 의존적 색상: 구면 조화 함수(SH)

$$\mathbf{c}(\mathbf{d}) = \sum_{l=0}^{L} \sum_{m=-l}^{l} k_{lm} \cdot Y_l^m(\mathbf{d})$$

학습 파이프라인은 Adam 옵티마이저를 사용하며, 3DGS와 유사하게 포인트별 위치, 밀도, 그리고 3차 구면 조화 함수(degree-3 SH)를 통한 뷰 의존 색상을 직접 최적화한다.

#### 🔷 정규화 손실(Regularization Loss)

Mip-NeRF 360의 distortion loss와 유사하게, 레이를 따른 체적 렌더링 적분의 기여 분포에 대한 정규화를 적용한다. 이 추가 손실 함수는 밀도가 표면에 집중되도록 유도하고 "플로터(floater)" 아티팩트를 줄인다.

총 학습 손실:

$$\mathcal{L} = \mathcal{L}_{\text{photo}} + \lambda \cdot \mathcal{L}_{\text{dist}}$$

- $\mathcal{L}_{\text{photo}}$: 픽셀 단위 L2 포토메트릭 손실
- $\mathcal{L}_{\text{dist}}$: Distortion 정규화 손실

#### 🔷 Delaunay 삼각분할과 Edge Flip 문제

Delaunay 그래프의 연결성은 작은 위치 변동에 민감하여 삼각분할에서 "edge flip"을 유발한다. 이러한 이산적 변화는 두 인접 심플렉스의 외접구(circumsphere)가 동일해지는 구성에서 발생한다.

Voronoi 다이어그램은 연결성의 이산적 변화가 면적이 0인 면(zero-surface area faces)에 효과적으로 숨겨지는 특성이 있어, Delaunay 메시를 직접 사용할 경우 연결성이 변할 때마다 큰 불연속성이 발생하는 것을 피할 수 있다.

---

### 2-3. 모델 구조

Radiant Foam은 실시간 차분 레이 트레이싱을 가능하게 하는 새로운 표현이다. 방법의 핵심은 NVIDIA RT 코어와 같은 전용 하드웨어에 의존하지 않고 효율적인 체적 메시 레이 트레이싱 알고리즘을 적용할 수 있게 해주는 다면체 셀(polyhedral cells)의 폼 구조이다. 이 셀들을 Voronoi 다이어그램으로 매개변수화하여 체적 렌더링 하에서 미분 가능함을 보이며 연속적으로 최적화한다.

**학습 과정 구조:**

Coarse-to-Fine 학습 접근법을 제안하며, 이는 적응형 해상도를 가진 메시 모델의 빠른 구성을 가능하게 한다.

초기화 및 워밍업 학습 후, 점들이 유용한 위치에 배치되도록 Voronoi 사이트의 수를 점진적으로 증가시킨다. 총 학습 반복의 절반까지 최대 원하는 포인트 수에 도달할 때까지 선형적으로 포인트 수를 증가시킨다.

가지치기(pruning) 전략은 밀도가 매우 낮고 밀도가 낮은 이웃으로 둘러싸인 Voronoi 사이트를 제거한다. 이 가지치기는 표면에 기여하지도 않고 정의하지도 않는 사이트들을 제거하여 객체 경계의 정확도를 유지한다.

---

### 2-4. 성능 향상

Mip-NeRF 360 및 Deep Blending 데이터셋에서 품질 측면으로 3DGS 및 3DGRT(최첨단 차분 레이 트레이싱 방법)와 유사하거나 약간 낮은 결과를 달성한다. 그러나 렌더링 속도에서 탁월하며, 효율적인 레이 트레이싱 구현은 경우에 따라 300 FPS 이상을 달성하여 3DGRT(119 FPS)보다 두 배 이상 빠르다.

| 방법 | 렌더링 패러다임 | 특수 HW 필요 | 속도 (FPS) | 반사/굴절 지원 |
|---|---|---|---|---|
| **NeRF** | Ray marching | ❌ | ~0.1 FPS | ✅ |
| **3DGS** | Rasterization | ❌ | ~100 FPS | ❌ |
| **3DGRT** | Ray Tracing (BVH) | ✅ (RT Core) | ~119 FPS | ✅ |
| **Radiant Foam** | Ray Tracing (Voronoi) | ❌ | **>300 FPS** | ✅ |

레이 트레이싱은 래스터화로 근사하기 어려운 많은 효과의 구현을 단순화한다. 반사, 굴절, 비선형 카메라 모델을 렌더링 파이프라인에 통합하는 예를 제시하며, 각각은 래스터화로 달성하기 복잡하지만 렌더링 코드에 경미한 수정만으로 구현 가능하다.

---

### 2-5. 한계점

Voronoi 기반 표현은 연속적 최적화를 통해 폼 모델 구성에 매우 효과적이지만, 렌더링 파이프라인에서 사용할 수 있는 가능한 폼 모델의 공간은 Voronoi로 매개변수화되는 것보다 훨씬 크다. 특히 현재 모델은 항상 셀 경계가 이웃 포인트들 사이의 등거리(equidistant)여야 하므로, 표면을 정의하기 위해 많은 수의 작은 빈 셀이 필요하다. 향후 연구에서 Voronoi를 넘어서 일반화함으로써 이 요구사항을 완화할 수 있다.

추가적인 한계:

포인트가 많은 장면의 경우 GPU 메모리 사용량이 높을 수 있으며, 24GB GPU에서 실외 장면을 학습하려면 `final_points` 설정을 줄여야 할 수 있다.

Radiant Foam에서 고도로 텍스처가 있는 영역을 표현하려면 체적 셀 수를 늘려야 하는 경우가 많다. 이를 해결하기 위해서는 기하학과 외관을 분리하여 셀 예산을 크게 줄여야 한다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. NeRF 기법과의 호환성

Radiant Foam은 3DGS와 유사한 렌더링 속도와 품질을 제공하면서도, NeRF와 유사한 레이 기반 학습 모달리티를 가진다. 이는 많은 NeRF 기법들이 이 방법에 seamlessly 적용될 수 있으며, 기반 기하학이 체적 메시로 명시적으로 표현된다는 상당한 이점이 있음을 의미한다.

### 3-2. Voronoi 표현의 일반화 잠재력

Voronoi 기반 표현은 연속적 최적화를 통해 폼 모델을 구성하는 데 매우 효과적이지만, 렌더링 파이프라인에 사용될 수 있는 가능한 폼 모델의 공간은 Voronoi로 매개변수화되는 것보다 훨씬 크다.

### 3-3. SDF와의 결합을 통한 일반화

Radiant Foam은 명시적 Voronoi 다이어그램으로 레이 트레이싱을 Gaussian Splatting에 필적하는 처리량으로 되살렸다. 그러나 언급된 모든 방법들은 정밀한 메시 재구성에 어려움을 겪는다. SDFoam은 명시적 Voronoi 다이어그램과 암시적 SDF(Signed Distance Field)를 결합하여 이 격차를 해소한다.

SDFoam에서는 Radiant Foam과 같이 장면을 3D Voronoi 다이어그램으로 표현하지만, 각 Voronoi 셀은 중심과 색상뿐만 아니라 국소적으로 정의된 부호 거리값(signed distance value)으로 매개변수화되어, Voronoi-Delaunay 구조를 미분 가능하고 공간적으로 일관된 결합 암시적-명시적 표현으로 변환한다. 부호 거리 필드는 차분 렌더링을 통해 Voronoi 구조와 함께 학습된다.

### 3-4. 비선형 카메라 모델 지원

레이 트레이싱은 래스터화로 근사하기 어려운 많은 효과의 구현을 단순화한다. 반사, 굴절, 비선형 카메라 모델을 렌더링 파이프라인에 통합하는 예시를 제시한다.

이는 어안 렌즈, 롤링 셔터 카메라, 수중 환경 등 다양한 실제 캡처 환경으로의 일반화 가능성을 크게 높입니다.

### 3-5. Power Foam으로의 확장

기존 폼 표현은 공간의 명시적 체적 분할을 통해 상수 시간 레이 탐색을 가능하게 하지만, 잠재적으로 무한한 셀들이 효율적인 타일 기반 래스터화를 방해한다. Power Foam은 제어 가능한 셀 범위를 가진 경계 있는 Power Diagram으로 Voronoi 폼을 일반화하여 이 한계를 해결한다. 기하학과 외관을 분리하는 미분 가능 텍스처도 도입한다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 렌더링 방식 | 핵심 특징 | Radiant Foam 대비 |
|---|---|---|---|---|
| **NeRF** (Mildenhall et al.) | 2020 | Ray marching + MLP | 암시적 장면 표현 | 속도 크게 열세 |
| **Instant-NGP** (Müller et al.) | 2022 | Ray marching + HashGrid | 빠른 NeRF 학습 | 여전히 래스터화 불가 |
| **3DGS** (Kerbl et al.) | 2023 | Rasterization | 실시간 렌더링 | 광 전달 표현 어려움 |
| **3DGRT** (Moenne-Loccoz et al.) | 2024 | Ray Tracing (BVH + RT Core) | HW 가속 레이 트레이싱 | 특수 HW 필요, 속도 열세 |
| **Radiant Foam** | 2025 | Ray Tracing (Voronoi) | 상수 시간 셀 탐색 | **본 논문** |
| **SDFoam** | 2025 | Ray Tracing (Voronoi + SDF) | 기하+외관 통합 | 메시 재구성 강화 |
| **Power Foam** | 2026 | Ray Tracing + Rasterization | Voronoi → Power Diagram | 래스터화도 지원 |

Power Foam은 실시간 레이 트레이싱과 래스터화 모두를 위한 통합 렌더링 패러다임을 가능하게 하는 새로운 3D 표현이다. 핵심은 효율적인 래스터화를 용이하게 하면서 명시적 체적 메시의 고유한 레이 트레이싱 효율성을 유지하는 경계 있는 다면체 셀로 구성된 폼 기반 구조이다. 두 렌더링 패러다임 모두에서 수학적으로 동일한 결과를 생성하여 Splatting 방법의 팝핑 아티팩트와 뷰 불일치를 피한다. 특히 레이 트레이싱 부문에서는 Radiant Foam, 래스터화 부문에서는 3DGS에 필적하는 성능을 보인다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려 사항

### 5-1. 연구적 영향

**① 하드웨어 독립적 레이 트레이싱의 재발견**

체적 메시로 표현된 필드가 특수 하드웨어가 필요 없는 매우 효율적인 레이 트레이싱 알고리즘을 허용한다는 것을 보였으며, 이 접근법이 체적 렌더링 방법의 부활에서 간과되었음을 지적하고 이를 차분 렌더링 커뮤니티에 재소개하고자 한다.

**② IPU 등 새로운 하드웨어로의 확장**

Graphcore Mk2 IPU와 같은 다양한 하드웨어에서 Radiant Foam의 Voronoi 셀 체적 표현을 위한 완전한 인-SRAM 분산 렌더러가 개발되었다. 시스템은 씬을 타일에 분산하고 계층적 라우팅 오버레이를 통해 레이를 전달하여 예측 가능한 통신으로 온칩 SRAM에서 완전히 레이 마칭을 가능하게 한다.

### 5-2. 향후 연구 시 고려할 점

**① Voronoi를 넘어선 일반화**

Voronoi 기반 표현은 연속적 최적화를 통해 폼 모델을 구성하는 데 매우 효과적이지만, 현재 모델은 셀 경계가 항상 이웃 포인트들 사이의 등거리여야 하므로 표면을 정의하기 위해 많은 소형 빈 셀이 필요하다. 향후 연구에서 Voronoi를 넘어선 일반화로 이 요구사항을 완화할 수 있다.

**② GPU 메모리-FPS 트레이드오프 최적화**

포인트가 많은 장면에서 GPU 메모리 사용량이 높을 수 있으며, 24GB GPU에서 실외 장면 학습 시 포인트 수 감소가 필요할 수 있다. 따라서 대규모 장면 표현 시 메모리 효율화 전략이 중요한 연구 방향입니다.

**③ 래스터화와의 통합**

기존 폼 표현은 공간의 명시적 체적 분할을 통해 상수 시간 레이 탐색을 가능하게 하지만, 잠재적으로 무한한 셀들이 효율적인 타일 기반 래스터화를 방해한다. Power Diagram으로 Voronoi 폼을 일반화하면 제어 가능한 셀 범위로 공간 경계 프리미티브를 구현하면서 비용이 많이 드는 Delaunay 삼각분할 없이 학습이 가능하다.

**④ 동적 장면(Dynamic Scene)으로의 확장**

현재 Radiant Foam은 정적 장면을 대상으로 설계되어 있으며, 동적 객체나 시간적 변화를 포함하는 장면으로의 확장은 아직 미개척 분야입니다. Voronoi 사이트의 시간 의존적 최적화 전략 설계가 필요합니다.

**⑤ 글로벌 일루미네이션 모델링**

Radiant Foam의 레이 트레이싱 특성은 2차 반사, 그림자, 간접 조명 등 글로벌 일루미네이션 효과를 포함한 물리 기반 렌더링(PBR)으로의 확장에 유리하며, 이는 향후 핵심 연구 방향이 될 것입니다.

---

## 📚 참고 자료 및 출처

1. **Radiant Foam 논문 (arXiv)**: Govindarajan et al., "Radiant Foam: Real-Time Differentiable Ray Tracing," arXiv:2502.01157, 2025. https://arxiv.org/abs/2502.01157
2. **Radiant Foam 프로젝트 페이지**: https://radfoam.github.io/
3. **Radiant Foam 공식 구현 (GitHub)**: https://github.com/theialab/radfoam
4. **ICCV 2025 Open Access**: Govindarajan et al., ICCV 2025, pp. 4135–4145. https://openaccess.thecvf.com/content/ICCV2025/html/Govindarajan_Radiant_Foam_Real-Time_Differentiable_Ray_Tracing_ICCV_2025_paper.html
5. **ar5iv (논문 HTML 렌더링)**: https://ar5iv.labs.arxiv.org/html/2502.01157
6. **3dvar.com PDF**: https://www.3dvar.com/Govindarajan2025Radiant.pdf
7. **Power Foam (후속 연구)**: Govindarajan et al., "Power Foam: Unifying Real-Time Differentiable Ray Tracing and Rasterization," arXiv:2604.24994, 2026. https://arxiv.org/html/2604.24994 / https://powerfoam.github.io/
8. **SDFoam (관련 연구)**: "SDFoam: Signed-Distance Foam for explicit surface reconstruction," arXiv:2512.16706. https://arxiv.org/html/2512.16706
9. **In-SRAM Radiant Foam Rendering (관련 연구)**: "In-SRAM Radiant Foam Rendering on a Graph Processor," arXiv:2601.04382. https://arxiv.org/html/2601.04382
10. **Semyeong Yu 블로그 (논문 리뷰)**: https://semyeong-yu.github.io/blog/2025/radfoam/

> ⚠️ **정확도 주의**: 본 분석은 공개된 arXiv 논문, 프로젝트 페이지, 공식 GitHub 및 ICCV 2025 오픈 액세스를 기반으로 작성되었습니다. 특히 세부 수식(체적 렌더링 적분, SH 표현 등)은 논문에서 확인된 공식 구조를 따르나, 논문 내 일부 구체적 수치 및 부록 내용은 전체 PDF 직접 접근 없이는 완전히 검증이 어려울 수 있음을 밝힙니다.
