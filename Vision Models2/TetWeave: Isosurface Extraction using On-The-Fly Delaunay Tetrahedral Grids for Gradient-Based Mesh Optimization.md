# TetWeave: Isosurface Extraction using On-The-Fly Delaunay Tetrahedral Grids for Gradient-Based Mesh Optimization

> **참고 문헌 목록 (전체 출처)**
> - Binninger, A., Wiersma, R., Herholz, P., Sorkine-Hornung, O. (2025). *TetWeave: Isosurface Extraction using On-The-Fly Delaunay Tetrahedral Grids for Gradient-Based Mesh Optimization*. ACM Trans. Graph. 44(4), SIGGRAPH 2025. arXiv:2505.04590 (v2), DOI: 10.1145/3730851
> - ETH Zurich IGL 프로젝트 페이지 (igl.ethz.ch/projects/tetweave/)
> - 저자 개인 프로젝트 페이지 (alexandrebinninger.com/TetWeave/)
> - GitHub 저장소 (github.com/AlexandreBinninger/TetWeave)
> - Ruben Wiersma 개인 출판 페이지
> - Moonlight AI Literature Review (themoonlight.io)
> - NovaReviewHub 논문 리뷰
> - 비교 참고: Shen et al. 2021 *Deep Marching Tetrahedra* (DMTet, arXiv:2111.04276); Shen et al. 2023 *FlexiCubes* (github.com/nv-tlabs/FlexiCubes); *Tetrahedron Splatting for 3D Generation* (arXiv:2406.01579); *Subgrid Marching Tetrahedra* (arXiv:2606.00454); *Dual Contouring of Signed Distance Data* (arXiv:2604.00157, 동일 저자 후속 연구로 추정); *MeshCone* (arXiv:2412.08484, TetWeave 출력물을 후처리하는 응용 연구)

---

## 1. Executive Summary (10문장 이내)

TetWeave는 그래디언트 기반 메시 최적화(gradient-based mesh optimization)를 위한 새로운 등위면(isosurface) 표현 방법으로, Marching Tetrahedra에 사용되는 사면체 격자의 배치와 각 점에서의 방향성 부호 거리(directional signed distance)를 동시에 최적화한다. TetWeave는 Delaunay 삼각분할을 통해 사면체 격자를 즉석(on-the-fly)에서 생성함으로써 사전에 정의된 격자(predefined grid) 대비 유연성을 높인다. 추출된 메시는 워터타이트(watertight), 이중다양체(two-manifold), 교차 없음(intersection-free)이 보장된다. 이러한 유연성은 재구성 오차가 큰 영역에 새로운 점을 배치하는 리샘플링 전략을 가능하게 하며, 재구성 오차를 희생하지 않고도 메시의 페어니스(fairness)를 유도할 수 있어 최소한의 메모리 사용량과 적은 파라미터로 고품질의 적응형 메시를 만들어낸다. 결과적으로 TetWeave는 출력 메시의 정점 수에 대해 거의 선형적인(near-linear) 메모리 스케일링을 보이는데, 이는 사전 정의된 격자 방식 대비 상당한 개선이다. 저자들은 멀티뷰 3D 재구성, 메시 압축, 기하학적 텍스처 생성 등 다양한 도전적 과제에 TetWeave의 적용 가능성을 시연한다. 다만 저자들 스스로 TetWeave가 범용 적응형 메싱 기법이나 고정 스칼라 필드로부터의 등위면 추출용으로 설계된 것이 아니라, 멀티뷰 3D 재구성과 같은 그래디언트 기반 메시 처리에 특화된 표현임을 명시하고 있다.

### 1-1. 연구 목적과 필요성

기존 DMTet, FlexiCubes와 같은 방법들은 미리 정의된 격자(정육면체 또는 사면체 격자, 예: BCC 격자)를 사용하기 때문에, 사전 정의된 격자에서 시작하는 준적응형(semi-adaptive) 격자 방법들은 유연성이 제한되고 메싱이 사전 정의된 격자의 구성에 영향을 받으며, 출력 메시에서 유사한 수준의 디테일을 얻기 위해 사용되지 않는 빈 공간 때문에 배경 격자에 훨씬 더 많은 노드가 필요하다. 이는 메모리 낭비와 세부 표현력의 한계로 이어진다. TetWeave는 이 문제를 해결하기 위해 비정형(unstructured) 사면체 격자의 배치와 각 격자 노드의 부호 거리를 함께 최적화하는 새로운 미분 가능 등위면 표현을 제안하며, 이는 고정 그리드에 의존하는 DMTet나 FlexiCubes와 대조된다.

---

## 2. 핵심 주장과 근거 (표)

| # | 핵심 주장 | 근거(논문 인용) | 위치 |
|---|---|---|---|
| 1 | Delaunay 삼각분할 기반 즉석 격자 생성으로 유연성 확보 | "TetWeave constructs tetrahedral grids on-the-fly via Delaunay triangulation, enabling increased flexibility compared to predefined grids." | Abstract, Fig. 1, Fig. 2 |
| 2 | 위상적 보장 (watertight/manifold/intersection-free) | "The extracted meshes are guaranteed to be watertight, two-manifold and intersection-free." | Abstract, §3 (방법론) |
| 3 | 오차 기반 리샘플링 + 페어니스 정규화 | "The flexibility of TetWeave enables a resampling strategy that places new points where reconstruction error is high and allows to encourage mesh fairness without compromising on reconstruction error." | §4.3 (리샘플링), Fig. 19 |
| 4 | 메모리 효율성 (준선형 스케일링) | "TetWeave exhibits near-linear memory scaling relative to the vertex count of the output mesh - a substantial improvement over predefined grids." | §7 Discussion, Fig. 19, Table 3 |
| 5 | FlexiCubes 대비 저장 항목 수 감소 | "Our shape representation is particularly memory-efficient... we store only the points, their SDF values, and, optionally, spherical harmonics coefficients. In contrast, FlexiCubes... requires storing SDF values, deformation parameters per point, and 21 coefficients per voxel." | §7 Discussion |
| 6 | 방향성 SDF (구면 조화함수 활용)가 노멀 정확도 향상 | "the use of spherical harmonics does not significantly affect the chamfer distance or F1-score, it consistently enhances the EF1-score and reduces the percentage of inaccurate normals... spherical harmonics enable directional adjustments of the signed distance, effectively aligning normals at a highly localized level." | Table 5 (ablation) |
| 7 | 페어니스 항이 불필요한 미세 삼각형 제거 | "36% reduction in vertex count, as the fairness term eliminates triangles with extremely small areas while preserving reconstruction quality. However, this comes at the cost of a slight increase in the percentage of inaccurate normals." | Ablation section |
| 8 | ODT 에너지가 엣지 챔퍼 거리 개선 | "incorporating an ODT energy improves the edge chamfer distance by over..." (수치 일부 소스에서 확인 불가) | Ablation, Table 5 |
| 9 | 런타임은 FlexiCubes 대비 느림 (메모리-속도 트레이드오프) | "Our memory efficiency comes with the trade-off of increased runtime... For our method, most of the time in the forward pass is spent on Delaunay triangulation. Although the theoretical time complexity of Delaunay triangulation is $O(n \ln(n))$..." | Table 3, §7 |
| 10 | 다양한 응용 가능성 (멀티뷰 재구성, 메시 압축, 텍스처 생성) | "We demonstrate the applicability of TetWeave to a broad range of challenging tasks in computer graphics and vision, such as multi-view 3D reconstruction, mesh compression and geometric texture generation." | §5, §6 (응용 섹션) |

---

## 2-1. 상세 설명

### (1) 해결하고자 하는 문제
기존의 그래디언트 기반 등위면 표현(DMTet, FlexiCubes)은 **고정/준고정 격자**(정육면체 그리드 또는 BCC 사면체 그리드)에 의존한다. 이로 인해:
- 표면 근처가 아닌 격자 노드도 메모리를 차지하는 "빈 공간(empty space)" 낭비가 발생 (그림 1 오른쪽 상단 그래프에서 확인 가능한, "유사한 수준의 디테일을 얻기 위해 배경 격자에 훨씬 더 많은 노드가 필요"한 현상).
- 격자 해상도가 메시 디테일의 상한을 결정하여 국소적으로 세밀한 표현이 어렵다.

### (2) 제안 방법 (수식 및 개념)
TetWeave의 파이프라인은 다음 단계로 구성된다 (본문에서 명시적으로 확인된 서술 기반):

1. **점군(point cloud) 표현**: 각 점 $p_i$가 부호 거리 값 $s_i$를 가짐 ("a point cloud where each point is associated with a signed distance value").
2. **Delaunay 삼각분할**: "The forward pass of TetWeave is divided into two components: the Delaunay triangulation step for generating the tetrahedral grid (Si, 2015) and our implementation of the Marching Tetrahedra algorithm (Doi and Koide, 1991), adapted to incorporate spherical harmonics coefficients."
3. **활성 엣지(active edges) 탐색 및 방향성 SDF 계산**: "The process starts by generating a tetrahedral grid through Delaunay triangulation. Next, active edges are identified, and a directional signed distance is computed for each active point using spherical harmonics."

Marching Tetrahedra의 표준 원리(사면체 정점 값의 부호가 반대인 엣지에서 선형 보간으로 교차점 산출)는 다음과 같이 표현할 수 있다 (교차점 파라미터 $t$):

$$t_{ij} = \frac{s_i}{s_i - s_j}, \quad \text{when } s_i \cdot s_j < 0$$

TetWeave의 핵심 차별점은 이 $s_i$를 **방향(direction) $\omega$에 의존하는 함수**로 확장한 것으로 추정되며, 구면 조화 함수(spherical harmonics) 계수 $c_i^{lm}$를 이용해 각 엣지 방향에 따라 거리 값을 보정하는 방식이다:

$$d(p_i, \omega) \approx s_i + \sum_{l,m} c_i^{lm}\, Y_l^{m}(\omega)$$

> ⚠️ **주의(정확도 표시)**: 위 수식은 논문에서 서술된 개념("A directional signed distance function to more accurately capture the distance to the surface along tetrahedral edges", "adapted to incorporate spherical harmonics coefficients (of degree 1 in this experiment)")을 바탕으로 한 **개념적 재구성**이며, 논문 원문의 정확한 수식(계수 정의, 정규화 방식)은 검색 결과에서 전문(verbatim)으로 확인하지 못했다. 따라서 이 수식은 저자의 정확한 표기와 다를 수 있음을 명시한다.

4. **정규화 항 (2종)**: "Two regularization terms to improve the quality of our meshes" — 하나는 격자 품질을 위한 **ODT(Optimal Delaunay Triangulation) 에너지**, 다른 하나는 출력 메시의 **페어니스(fairness) 손실**로 추정된다 ("Two regularization terms to improve mesh quality (ODT energy for grid, fairness loss for mesh).").
5. **적응형 리샘플링**: "A method to adapt the tetrahedral grid to an unknown surface"이 언급되며, 재구성 오차가 높은 영역에 새 점을 추가하는 중요도 함수 $h$ 기반 방식을 사용한다 ("our resampling relies on an importance value function... which operates over a voxel grid decomposition of the current reconstructed shape. In the adaptive setting, ... h is computed based on rendering errors of the normal map.").

### (3) 모델 구조 (요약)
- **입력**: 임의의 점 집합 $P=\{p_1,...,p_n\}$ (점 위치는 학습 가능한 파라미터), 각 점의 부호 거리 $s_i$, (옵션) 구면 조화 계수.
- **전방 연산**: TetGen을 이용한 Delaunay 삼각분할 → "It then builds a Delaunay tetrahedral grid via Tetgen, and uses Marching Tetrahedra to reconstruct a mesh. If spherical harmonics are used, our implementation of Marching Tets incorporates the computation of the directional signed distance."
- **손실 함수**: 렌더링/챔퍼 거리 기반 재구성 손실 + ODT 정규화 + 페어니스 정규화 (다단계 최적화, "multi-stage optimization pipeline for stable training" as noted in review sources).

### (4) 성능 향상
- "This allows each point to contribute to a higher number of output triangles, significantly reducing the number of points needed to achieve a desired level of detail and Chamfer Distance. As illustrated in Fig. 19, our approach achieves better Chamfer Distance results with far fewer points compared to FlexiCubes, translating to significantly lower memory requirements. The graph demonstrates that TetWeave scales more efficiently than FlexiCubes, allowing us to reconstruct shapes at a higher resolution."
- 정성적으로도 "Comprehensive quantitative results comparing TetWeave against strong baselines across numerous metrics (geometry, rendering, performance) provide strong evidence for the claims regarding quality, efficiency, and scaling. Visual comparisons effectively illustrate the advantages in detail capture and mesh fairness. Ablation studies support the importance of novel method components." (제3자 리뷰 요약)

### (5) 한계 (저자 명시)
- "TetWeave is not designed as a general-purpose adaptive meshing technique or for isosurface extraction from fixed scalar fields. Rather, we propose a specialized representation optimized for gradient-based mesh processing, particularly suited for applications like multi-view 3D reconstruction."
- "Runtime. Our memory efficiency comes with the trade-off of increased runtime." — Delaunay 삼각분할이 주요 병목.
- 페어니스 정규화는 "a slight increase in the percentage of inaccurate normals"라는 트레이드오프를 유발.

---

## 3. 페이지/그림/표 표시 요약

| 주장 | 위치 |
|---|---|
| 방법 개요 및 비교(반정형 격자 vs. TetWeave) | Fig. 1 |
| 파이프라인 전체 흐름 (점군 → Delaunay → MT) | Fig. 2 |
| 성능(챔퍼거리 vs. 포인트 수, 메모리 스케일링) | Fig. 19 |
| 런타임 비교 | Table 3 |
| 구면 조화함수 및 정규화 항 ablation | Table 5 |
| 논문 전체 분량 | 19 pages, 21 figures (Comments: ACM Trans. Graph. 44, 4. SIGGRAPH 2025. 19 pages, 21 figures) |

---

## 4. 저자 보고 결과 vs. 필자 해석 분리

**[저자 보고 — 원문 인용 기반]**
- 근접 선형 메모리 스케일링, 위상 보장(watertight/manifold/intersection-free), 구면조화함수의 노멀 정확도 개선 효과는 모두 저자가 직접 실험/증명한 것으로 명시되어 있다.

**[필자 해석 — 추론/종합]**
- Delaunay 기반 격자가 "국소적으로 점을 재배치"할 수 있다는 점에서, 본질적으로 이는 **적응형 메시 심플리피케이션(adaptive mesh simplification)**과 유사한 원리를 그래디언트 최적화 맥락에 적용한 것으로 해석할 수 있다. 이는 저자가 명시적으로 이렇게 프레이밍하지는 않았으나, 필자의 종합적 판단이다.
- 방향성 SDF(구면조화 활용)는 개념적으로 뉴럴 임플리시트 표현에서 흔히 쓰이는 "방향 의존적 특징(view/direction-dependent feature)" 아이디어(예: NeRF의 view-dependent color)를 부호 거리 함수에 적용한 것으로 볼 수 있다 — 이는 필자의 유추이며 논문이 명시적으로 이런 비교를 하는지는 확인되지 않았다.
- 런타임 트레이드오프($O(n\log n)$ Delaunay 삼각분할)는 실시간 응용(예: 게임 엔진, 실시간 스캐닝)에는 부적합할 수 있다는 점은 필자의 추론이며, 논문이 이를 직접적으로 "실시간 부적합"이라 언급했는지는 확인되지 않았다.

---

## 5. 통계적으로 취약하거나 비교 불가능한 수치

- "36% reduction in vertex count" — 이 수치가 어떤 특정 형상(shape)/설정에서 도출된 것인지, 여러 형상에 대한 평균인지 검색 결과에서 명확히 확인되지 않아 **일반화 가능성이 불명확**하다.
- ODT 에너지의 "엣지 챔퍼 거리 개선(over ...%)" 수치는 "Additionally, incorporating an ODT energy improves the edge chamfer distance by over..."에서 문장이 잘려 있어 **정확한 수치를 확인할 수 없다** — 이 부분은 원문 PDF 확인이 필요하다.
- FlexiCubes와의 메모리/런타임 비교(Table 3, Fig. 19)는 "Experiments were conducted on an NVIDIA RTX 3090 GPU" 단일 하드웨어 설정 기준으로, 다른 GPU/배치 크기에서 일반화되는지는 검증되지 않았다.
- 3rd party 리뷰(NovaReviewHub, Moonlight)는 "강력한 정량적 증거"라고 평가하지만, 이는 **AI 기반 자동 리뷰 요약**으로 원 논문 저자의 검증된 주장과 동일한 신뢰도로 취급해서는 안 된다.

---

## 6. 문서가 답하지 않는 질문

1. TetWeave의 학습/최적화에 소요되는 **총 GPU 메모리 및 훈련 시간**이 절대값으로 어느 정도인지 (상대적 비교만 제시됨).
2. Delaunay 삼각분할의 $O(n\log n)$ 비용이 **매우 큰 점 개수(수백만 이상)**에서도 실용적인지에 대한 구체적 스케일링 실험 데이터.
3. 방향성 SDF에서 구면조화 함수의 **차수(degree)를 1보다 높일 경우**의 트레이드오프(품질 vs 연산 비용)가 충분히 탐구되었는지.
4. **동적(dynamic) 형상**(애니메이션, 시간에 따라 변하는 메시)에 대한 적용 가능성.
5. 임의의 초기 점 분포(초기화 전략)가 최종 수렴 품질에 미치는 영향에 대한 체계적 분석.

---

## 7. 가장 중요한 그림 5개 해석

| 그림 | 해석 |
|---|---|
| **Fig. 1** | "TetWeave jointly optimizes a tetrahedral grid and a directional signed distance function used for Marching Tetrahedra. Our method weaves a background grid around the surface, which is regularized to give fair output meshes. The results are compared with semi-adaptive grid methods (top row), which start from a predefined grid. These methods have limited flexibility... They require many more nodes in the background grid because of unused empty space to get a similar level of detail in the output mesh." — 논문의 핵심 주장을 시각적으로 압축한 대표 그림. |
| **Fig. 2** | 파이프라인 개요: 점군 → Delaunay 삼각분할 → 활성 엣지 식별 → 구면조화 기반 방향성 SDF 계산이 단계별로 도식화되어 있어, 방법론의 전체 흐름을 이해하는 데 핵심적이다. |
| **Fig. 19** | "As illustrated in Fig. 19, our approach achieves better Chamfer Distance results with far fewer points compared to FlexiCubes, translating to significantly lower memory requirements. The graph demonstrates that TetWeave scales more efficiently than FlexiCubes." — 메모리 효율성이라는 핵심 기여를 정량적으로 증명하는 그래프. |
| **Table 3 (런타임 비교)** | "we present the average runtime for both FlexiCubes and TetWeave, measured on the shape shown in Fig. 18. For our method, most of the time in the forward pass is spent on Delaunay triangulation." — 메모리-속도 트레이드오프를 명확히 보여주는 정량 지표. |
| **Table 5 (Ablation)** | "We examine the impact of incorporating spherical harmonics for computing directional signed distances in Table 5. While the use of spherical harmonics does not significantly affect the chamfer distance or F1-score, it consistently enhances the EF1-score and reduces the percentage of inaccurate normals." — 개별 구성 요소의 기여도를 검증하는 핵심 근거표. |

---

## 8. 결론: 시사점 및 후속 연구

저자들은 "We conclude by exploring future research directions to further advance unstructured mesh representations."라고 명시하며, 비정형(unstructured) 메시 표현의 발전 방향을 논의로 남기고 있다. 핵심 시사점은 (1) 고정 격자를 벗어난 **비정형 적응형 표현**이 메모리 효율성과 세부 표현력을 동시에 개선할 수 있다는 것, (2) **방향 의존적 거리 함수**가 저해상도 격자에서도 표면 디테일(특히 노멀 정확도)을 향상시킬 수 있다는 것이다.

### 8-1. 모델의 일반화 성능 향상 가능성

TetWeave의 일반화 가능성에 대해 다음을 논할 수 있다:
- **긍정적 요인**: Delaunay 삼각분할은 임의의 점 분포에 대해 잘 정의된 수학적 성질(공차원 최소화, 국소 최적성)을 가지므로, 특정 형상 카테고리에 국한되지 않고 "dynamically builds adaptive grids via Delaunay triangulation and uses Marching Tetrahedra to extract intersection-free, watertight, manifold meshes"라는 점에서 형상 무관하게 위상적 보장이 유지된다. 이는 다양한 형상 카테고리(유기적 형태, 인공물, 텍스처가 있는 표면)에 대한 일반화에 유리한 구조적 특성이다.
- **한계 요인**: 저자 스스로 "TetWeave is not designed as a general-purpose adaptive meshing technique or for isosurface extraction from fixed scalar fields"라고 명시함으로써, 이 방법이 **그래디언트 기반 최적화 시나리오에 특화**되어 있으며, 정적 스칼라 필드(예: CT/MRI 볼륨 데이터)로부터의 직접적 등위면 추출과 같은 다른 도메인으로의 일반화는 검증되지 않았음을 인정하고 있다.
- **필자 해석**: 일반화 성능을 더욱 높이려면 (a) 다양한 위상학적 복잡도(genus가 높은 형상, 얇은 구조물)에 대한 강건성 검증, (b) 서로 다른 렌더링/센서 노이즈 조건에서의 강건성 평가, (c) 텍스트/이미지 조건부 생성 모델과의 통합 시 형상 다양성 대응력에 대한 추가 실험이 필요할 것으로 판단된다. 이는 논문에서 직접 다루지 않은 부분으로, 향후 연구 과제로 제안한다.

### 8-2. 2020년 이후 관련 최신 연구 비교 분석 및 향후 연구 고려사항

| 연구 (연도) | 핵심 아이디어 | TetWeave와의 관계 |
|---|---|---|
| **DMTet** (2021, arXiv:2111.04276) | 고정 사면체 그리드 위에서 SDF와 정점 변위를 예측하고, Chamfer Distance 및 normal consistency 손실로 표면 정렬을 학습 | TetWeave가 극복하고자 한 "고정 격자" 패러다임의 대표 선행 연구 |
| **FlexiCubes** (2023) | DMTet 대비 더 균일한 테셀레이션과 미세한 기하학적 디테일을 포착하며, 정점당 21개의 복셀 계수를 저장 | TetWeave의 주요 정량적 비교 대상 (메모리, 런타임) |
| **Tetrahedron Splatting** (2024, arXiv:2406.01579) | Eikonal 손실과 정점 기반 노멀 일관성 손실을 사면체에 적용 | 사면체 기반 정규화 항 설계 측면에서 유사한 문제의식 공유 |
| **Subgrid Marching Tetrahedra** (2026, arXiv:2606.00454) | 고전적 marching 방식이 어떤 노드에도 샘플링되지 않은 얇은 곡선/튜브형 특징을 놓칠 수 있음을 지적 | Marching Tetrahedra 계열 알고리즘의 세부 특징 포착 한계를 후속적으로 다룸 |
| **Dual Contouring of SDF Data** (2026, 동일 저자 포함 추정) | "the edges of the regular tetrahedralization are not guaranteed to align with a shape's sharp features, leading to smoothed and chamfered corners" | TetWeave를 포함한 사면체 기반 방법들의 **날카로운 특징(sharp feature) 표현 한계**를 명시적으로 지적하며 개선을 시도하는 저자들의 직접적 후속 연구로 추정됨 |
| **MeshCone** (2025, arXiv:2412.08484) | TetWeave로 생성된 메시를 볼록 최적화(convex optimization)로 후처리하여 평균 재구성 오차를 최대 31.5%까지 개선 | TetWeave 출력물의 **응용/후처리 파이프라인**으로서의 실용적 활용 사례 — TetWeave의 실제 영향력을 보여주는 사례 |

**향후 연구 시 고려할 점 (필자 종합)**:
1. **날카로운 특징 보존**: 후속 연구(Dual Contouring of SDF Data)에서 이미 지적되었듯, Delaunay 사면체 격자는 본질적으로 매끄러운 보간에 유리하지만 날카로운 모서리 표현에는 한계가 있을 수 있다 — 이는 TetWeave 계열 방법의 공통적 개선 과제다.
2. **연산 효율성**: Delaunay 삼각분할의 반복적 재계산 비용을 줄이기 위한 점진적(incremental) 갱신 알고리즘 연구가 필요하다.
3. **생성 모델과의 결합**: FlexiCubes가 "plug-and-play differentiable mesh extraction module"로 "producing significantly improved mesh quality"라는 위치를 점했듯, TetWeave 역시 대규모 3D 생성 모델(디퓨전 기반 등)의 메시 추출 모듈로 통합될 가능성이 있으며, 이 경우 배치(batch) 처리 효율성 개선이 중요한 후속 과제가 될 것이다.
4. **후처리 파이프라인과의 결합**: MeshCone 사례처럼 TetWeave 출력을 추가로 정제하는 후처리 기법과의 결합이 실용적 활용도를 높일 수 있다.

---

**※ 참고 자료 전체 출처 목록 (재정리)**
1. arXiv:2505.04590 (Abstract, HTML v2, PDF) — TetWeave 원 논문
2. ACM Digital Library, DOI: 10.1145/3730851
3. ETH Zurich IGL 프로젝트 페이지 (igl.ethz.ch/projects/tetweave/)
4. 저자 개인 페이지 (alexandrebinninger.com/TetWeave/)
5. GitHub: AlexandreBinninger/TetWeave
6. Ruben Wiersma 개인 출판 페이지
7. Moonlight AI 문헌 리뷰 (themoonlight.io)
8. NovaReviewHub 논문 리뷰
9. arXiv:2111.04276 — Deep Marching Tetrahedra (DMTet)
10. GitHub: nv-tlabs/FlexiCubes
11. arXiv:2406.01579 — Tetrahedron Splatting for 3D Generation
12. arXiv:2606.00454 — Subgrid Marching Tetrahedra
13. arXiv:2604.00157 — Dual Contouring of Signed Distance Data
14. arXiv:2412.08484 — MeshCone
