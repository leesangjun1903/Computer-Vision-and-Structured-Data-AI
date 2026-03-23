# SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering

> **논문 정보**: Antoine Guédon & Vincent Lepetit (LIGM, Ecole des Ponts, Univ Gustave Eiffel, CNRS, France)
> **발표**: CVPR 2024 | arXiv: 2311.12775

---

## 1. 핵심 주장 및 주요 기여 요약

SuGaR는 3D Gaussian Splatting으로부터 정밀하고 극도로 빠른 메시 추출을 가능하게 하는 방법을 제안합니다. Gaussian Splatting은 NeRF보다 훨씬 빠른 학습 속도로 사실적인 렌더링을 제공하여 최근 큰 인기를 얻고 있습니다. 그러나 수백만 개의 미세한 3D Gaussian들은 최적화 후 비정렬 상태가 되어 메시 추출이 어려웠으며, 이를 위한 방법이 제안된 바 없었습니다.

### 3대 핵심 기여 (Three Main Contributions)

1. **정규화 항(Regularization Term)**: 3D Gaussian이 장면 표면과 정렬되도록 유도하는 정규화 항 도입
2. **확장 가능한 메시 추출 방법**: 3D Gaussian에 특화된 확장 가능한(scalable) 메시 추출 기법
3. **리파인먼트 방법**: 새로운 3D Gaussian을 메시의 삼각형에 바인딩하는 정제(refinement) 방법으로, 하이브리드 표현(hybrid representation) 생성

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

실제 장면의 Gaussian Splatting 표현은 일반적으로 서로 다른 스케일과 회전을 갖는 수백만 개의 3D Gaussian으로 구성되며, 대부분은 텍스처와 세부사항을 재현하기 위해 극도로 작습니다. 이로 인해 밀도 함수(density function)가 거의 모든 곳에서 0에 가까워지며, Marching Cubes 알고리즘은 정밀한 복셀 그리드를 사용하더라도 이러한 희소 밀도 함수의 적절한 레벨 셋을 추출하지 못합니다.

즉, 핵심 문제는:
- 3DGS의 Gaussian들이 **비구조적(unorganized)** 으로 분포하여 직접적인 메시 추출이 불가능
- Neural SDF 기반 방법(BakedSDF 등)은 메시 추출에 **수 시간** 소요
- Marching Cubes는 Gaussian Splatting의 희소 밀도 함수에서 실패

### 2.2 제안하는 방법 (수식 포함)

#### (A) 3D Gaussian Splatting 기초

3DGS에서 각 Gaussian $g$는 평균(mean) $\boldsymbol{\mu}$, 공분산 행렬 $\boldsymbol{\Sigma}$, 불투명도(opacity) $\alpha$, 구면 조화 계수(spherical harmonics)로 매개변수화됩니다. 이미지 집합이 주어지면 Gaussian 집합은 SfM으로 생성된 포인트 클라우드로부터 초기화되며, Gaussian의 파라미터(평균, 쿼터니언, 스케일링 벡터, 불투명도, 구면 조화 파라미터)는 렌더링이 입력 이미지와 일치하도록 최적화됩니다.

각 Gaussian의 밀도 함수는 다음과 같이 정의됩니다:

$$d_g(\mathbf{x}) = \alpha_g \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_g)^T \boldsymbol{\Sigma}_g^{-1} (\mathbf{x}-\boldsymbol{\mu}_g)\right)$$

전체 장면의 밀도 함수는 개별 Gaussian들의 합으로 표현됩니다:

$$d(\mathbf{x}) = \sum_{g} d_g(\mathbf{x})$$

#### (B) Surface Alignment Regularization (표면 정렬 정규화)

SuGaR의 핵심 혁신은 Gaussian들이 장면 표면과 정렬되도록 유도하는 정규화 항입니다. 이 정규화 항은 Gaussian Splatting 최적화 과정에 도입되어 Gaussian들이 장면 표면에 정렬되고 표면 위에 잘 분포되도록 장려합니다.

**이상적 조건**: 만약 Gaussian이 표면에 완벽히 정렬된다면, 각 Gaussian은 하나의 스케일링 팩터가 다른 두 개보다 훨씬 작은 "편평한(flat)" 형태가 됩니다. 이상적인 경우 단일 가우시안의 밀도 함수는:

$$d_g(\mathbf{x}) \approx \alpha_g \exp\left(-\frac{\text{dist}(\mathbf{x}, \mathcal{P}_g)^2}{2s_g^2}\right)$$

여기서 $\mathcal{P}_g$는 Gaussian $g$의 가장 큰 두 스케일에 의해 정의되는 평면(tangent plane), $s_g$는 가장 작은 스케일링 값, $\text{dist}(\mathbf{x}, \mathcal{P}_g)$는 점 $\mathbf{x}$에서 평면까지의 거리입니다.

**SDF 기반 정규화**: 임의의 밀도 함수 $d$에 대해 이상적인 거리 함수를 다음과 같이 정의합니다:

$$f(\mathbf{x}) = s_g^* \cdot \sqrt{-2\log(d(\mathbf{x}))}$$

여기서 $s_g^*$는 점 $\mathbf{x}$에 가장 가까운 Gaussian의 최소 스케일링 값입니다.

이 함수가 이상적인 SDF(Signed Distance Function)에 가까워지도록 하는 **정규화 손실(regularization loss)** 은 다음과 같습니다:

$$\mathcal{L}_{\text{reg}} = \lambda_{\text{reg}} \sum_{\mathbf{p} \in \mathcal{S}} \left( \left\| \nabla f(\mathbf{p}) \right\| - 1 \right)^2$$

여기서 $\mathcal{S}$는 장면에서 샘플링된 점들의 집합이고, $\lambda_{\text{reg}}$는 정규화 가중치입니다. 이 **Eikonal loss** 조건은 $f$의 그래디언트 노름이 1이 되도록 강제하여 SDF의 성질을 만족시킵니다.

**밀도 기반 정규화**: 논문은 두 가지 정규화 방법을 제공합니다 — 밀도 정규화(density regularization)와 SDF 정규화입니다. 밀도 정규화는 더 단순하며 장면 중심에 위치한 객체에 잘 작동합니다. SDF는 더 강한 정규화를 제공하며, 특히 배경 영역에서 우수합니다. 따라서 SDF 정규화가 표준 데이터셋에서 더 높은 메트릭을 달성합니다. 그러나 360° 커버리지로 장면 중심 객체를 재구성할 때는 단순한 밀도 정규화가 일반적으로 더 나은 메시를 생성합니다.

전체 최적화 손실함수는:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{render}} + \mathcal{L}_{\text{reg}}$$

여기서 $\mathcal{L}_{\text{render}}$는 원래 3DGS의 photometric loss ($\mathcal{L}_1$ + D-SSIM)입니다.

#### (C) 메시 추출 (Mesh Extraction via Poisson Reconstruction)

밀도 함수의 레벨 셋의 가시적 부분에서 포인트를 매우 효율적으로 샘플링하는 방법을 도입하여, 이 포인트들에 Poisson 재구성 알고리즘을 실행하여 삼각형 메시를 얻습니다. 이 접근법은 Marching Cubes 알고리즘과 달리 확장 가능(scalable)하며, Neural SDF에 의존하는 다른 최신 방법들에 비해 단일 GPU에서 수분 내에 매우 상세한 표면 메시를 재구성합니다.

실제로 메시 추출을 위해 두 개의 Poisson 재구성을 적용합니다: 하나는 전경 포인트용, 하나는 배경 포인트용입니다. 전경 포인트는 모든 학습 카메라 포즈의 바운딩 박스 내부에 위치한 포인트로, 배경 포인트는 외부에 위치한 포인트로 정의됩니다. 이 단순한 전경/배경 구분은 가능한 한 일반적인 접근법을 설계하기 위해 선택되었습니다.

#### (D) 하이브리드 리파인먼트 (Joint Refinement)

선택적 리파인먼트 전략으로, Gaussian을 메시 표면에 바인딩하고 Gaussian splatting 렌더링을 통해 Gaussian과 메시를 공동으로 최적화합니다. 이를 통해 Gaussian 자체가 아닌 메시를 조작하여 Gaussian의 편집, 조각, 애니메이션 및 릴라이팅이 용이해집니다.

리파인먼트에서는 메시의 각 삼각형 면에 고정된 수의 Gaussian을 바인딩합니다. 각 Gaussian의 위치는 삼각형의 바리센트릭 좌표(barycentric coordinates)로 표현됩니다:

$$\boldsymbol{\mu}_g = \sum_{i=1}^{3} w_i \mathbf{v}_i, \quad \text{where} \quad \sum_{i=1}^{3} w_i = 1, \; w_i \geq 0$$

여기서 $\mathbf{v}_i$는 삼각형의 꼭짓점 좌표이고, $w_i$는 학습 가능한 바리센트릭 가중치입니다.

Low-poly 메시는 200,000개 꼭짓점과 삼각형당 6개 Gaussian, High-poly 메시는 1,000,000개 꼭짓점과 삼각형당 1개 Gaussian을 사용합니다.

### 2.3 모델 구조 (전체 파이프라인)

전체 SuGaR 파이프라인은 4개의 주요 단계와 1개의 선택적 단계로 구성됩니다:
1. **Short vanilla 3DGS 최적화**: 추가 정규화 없이 7k 반복으로 vanilla 3DGS 모델 최적화 (Gaussian이 장면에 위치하도록 함)
2. **SuGaR 최적화**: 장면 표면과 Gaussian 정렬 최적화
3. **메시 추출**: 최적화된 Gaussian으로부터 메시 추출
4. **Gaussian 바인딩**: 추출된 메시에 새로운 Gaussian을 바인딩
5. **(선택) Joint Refinement**: Gaussian과 메시의 공동 최적화

```
[Input Images] → [SfM Point Cloud] → [Vanilla 3DGS (7k iter)]
    → [SuGaR Regularized Optimization] → [Poisson Mesh Extraction]
    → [Gaussian-Mesh Binding] → [Joint Refinement] → [Hybrid Output]
```

### 2.4 성능 평가

SuGaR는 Mip-NeRF360, Tanks&Temples, DeepBlending을 포함한 다양한 실세계 데이터셋에서 Mobile-NeRF, NeRFMeshing 등 메시 기반 최신 novel view synthesis 방법보다 우수한 렌더링 품질을 보입니다. 이는 높은 PSNR, SSIM 및 낮은 LPIPS 점수로 입증되며, 메시에 Gaussian을 바인딩하는 것이 시각적 충실도와 디테일 보존을 크게 향상시킵니다.

주목할 점은, SuGaR가 메시 추출에 초점을 맞추고 있음에도 불구하고 Instant-NGP, Plenoxels 등 비메시 기반 novel view synthesis 모델과 동등하거나 이를 능가하는 렌더링 성능을 달성한다는 것입니다.

| 방법 | 메시 추출 시간 | PSNR | SSIM | 메시 품질 |
|------|------------|------|------|----------|
| BakedSDF | 수 시간 | - | - | 높음 |
| **SuGaR** | **수 분** | **높음** | **높음** | **높음** |
| Mobile-NeRF | 수 시간 | 낮음 | 낮음 | 중간 |
| Vanilla 3DGS | N/A (메시 없음) | 최고 | 최고 | N/A |

SuGaR는 메시에 의존하기 때문에 렌더링 품질 면에서 vanilla 3D Gaussian Splatting보다는 뒤처지지만, 메시를 복원하지 못하는 다른 방법들보다는 높은 성능을 보입니다.

### 2.5 한계점

1. **Vanilla 3DGS 대비 렌더링 품질 감소**: SuGaR는 메시에 의존하므로 vanilla 3D Gaussian Splatting에 비해 렌더링 품질이 다소 낮습니다.

2. **대규모 장면에서의 제약**: 도시 구역과 같은 훨씬 더 큰 데이터셋을 재구성하려면 Gaussian의 위치 및 스케일링 팩터의 학습률을 낮추어야 합니다. 장면이 광범위할수록 이 값이 더 낮아야 합니다. SuGaR에서도 매우 큰 장면을 재구성할 때 이러한 학습률을 낮춰야 합니다.

3. **바운딩 박스 민감성**: 사용자가 장면의 특정 객체를 높은 디테일로 재구성하거나 장면이 매우 크거나 카메라 중심이 장면에서 매우 멀 경우, 기본 바운딩 박스가 최적이 아닐 수 있습니다.

4. **다양한 장면 유형에 대한 불확실성**: 더 다양한 장면 유형에 대한 처리에 대해서는 아직 의문이 남아 있습니다.

5. **Poisson 재구성 파라미터 민감성**: Poisson 재구성의 기본 하이퍼파라미터가 Gaussian 크기에 비해 너무 세밀할 경우 메시 표면에 타원체 범프가 나타날 수 있습니다. 이는 카메라 궤적이 단순한 전경 객체에 매우 가까운 경우 발생할 수 있습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화 관련 특성

SuGaR는 per-scene optimization 방식으로, 각 장면마다 개별적으로 최적화가 필요합니다. 이는 3DGS 기반 방법의 본질적 특성입니다.

**일반화를 지원하는 설계 요소:**
- 전경/배경 구분을 가능한 한 일반적인 접근법으로 설계하였습니다.
- 두 가지 정규화 옵션(density / SDF)을 제공하여 장면 유형에 따른 적응성 확보
- SuGaR는 장면 표면과 입자를 정렬하면서도 디테일을 유지하는 새로운 초기화 및 학습 접근법을 기여합니다.

### 3.2 일반화 성능 향상을 위한 방향

1. **Cross-Scene 학습**: 현재의 per-scene 최적화를 넘어 feed-forward 방식의 일반화 모델과 결합 가능성
   - 사전 학습된 depth prior (Depth-Anything 등)와의 통합
   - 저텍스처 및 덜 관측되는 영역(실내 장면 등)에서 3DGS는 제한된 입력 뷰에 과적합하는 경향이 있어 추가적인 사전 단서(prior cues)를 통한 규제가 필요합니다.

2. **Sparse View 재구성으로의 확장**: Gaussian Splatting의 장면 기하학 최적화에 대한 최근 발전은 이미지로부터 상세한 표면의 효율적인 재구성을 가능하게 했습니다. 그러나 입력 뷰가 희소한 경우 이러한 최적화는 과적합에 취약하여 재구성 품질이 저하됩니다.

3. **Geometric Prior 통합**: FDS(Flow-Depth Supervision) 등의 매칭 사전 정보 통합으로 cross-view consistency 강화 가능

4. **적응적 정규화 강도**: 장면 복잡도에 따라 $\lambda_{\text{reg}}$를 자동 조정하는 메커니즘 개발

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 연구 영향

1. **Gaussian Splatting ↔ Mesh 브릿지 확립**: SuGaR는 3DGS의 명시적 표현과 전통적 메시 표현을 연결하는 선구적 연구로, 이후 수많은 후속 연구를 촉발했습니다.

2. **하이브리드 표현의 실용화**: Blender, Unity, Unreal Engine 등 전통적 소프트웨어를 사용하여 Gaussian 자체가 아닌 메시를 조작함으로써 편집, 조각, 리깅, 애니메이션, 릴라이팅이 가능합니다.

3. **효율성 패러다임**: SuGaR가 수분 내에 높은 렌더링 품질과 정밀하고 편집 가능한 메시를 달성하는 능력은 BakedSDF처럼 수 시간이 걸리는 방법 대비 중대한 진보입니다. 이러한 효율성은 고품질 렌더링과 편집 가능한 3D 에셋이 모두 필요한 응용에 매우 실용적인 솔루션으로 자리매김합니다.

### 4.2 향후 연구 시 고려할 점

| 고려 사항 | 세부 내용 |
|-----------|----------|
| **표면 정합 품질** | Eikonal loss 기반 정규화 외에 normal consistency, depth consistency 등 다중 기하학적 제약 조건 통합 |
| **확장성** | 대규모/복잡 장면에서의 메모리 효율성 및 학습률 스케줄링 전략 |
| **Sparse View** | 소수 뷰에서의 robustness 향상을 위한 generalization prior 필요 |
| **동적 장면** | 현재 정적 장면에 특화 → 4D/동적 장면으로의 확장 |
| **품질-속도 트레이드오프** | 메시 해상도(vertex 수), Gaussian/삼각형 비율 최적화 |
| **반사/투명 표면** | Specular, glossy 표면에서의 기하학 재구성 한계 극복 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 주요 비교 대상 연구

| 연구 | 연도/학회 | 핵심 접근법 | SuGaR 대비 특징 |
|------|----------|-----------|----------------|
| **NeRF** (Mildenhall et al.) | 2020, ECCV | Neural Radiance Fields + Volume Rendering | 렌더링 우수하나 메시 추출 불가, 학습 매우 느림 |
| **3D Gaussian Splatting** (Kerbl et al.) | 2023, SIGGRAPH | Explicit Gaussian primitives + Rasterization | 빠른 학습/렌더링이나 직접 메시 추출 불가 |
| **NeuS** (Wang et al.) | 2021, NeurIPS | SDF 기반 Neural Implicit Surface | 높은 기하학 품질이나 느린 학습 |
| **Neuralangelo** (Li et al.) | 2023, CVPR | Hash-grid + Multi-res SDF | GS2Mesh 등의 방법이 Splatting 기반 방법 중 최고 성능을 달성하며, Neuralangelo과 비슷한 수준이면서 실행 시간은 훨씬 짧습니다. |
| **2D Gaussian Splatting (2DGS)** (Huang et al.) | 2024, SIGGRAPH | 2D 평면 Gaussian 디스크로 장면을 표현하여 객체 표면과 밀접하게 정렬하며, depth-normal 정규화로 기하학을 향상시킵니다. | 3D→2D Gaussian으로 본질적 표면 정합 |
| **GOF** (Yu et al.) | 2024, ACM TOG | GOF는 기존 3DGS 기반 방법 중 표면 재구성과 novel view synthesis에서 최고 성능을 달성합니다. SuGaR에 비해 전경 객체와 배경 모두에서 더 상세하고 매끄러운 기하학을 재구성할 수 있습니다. |
| **NeuSG** (Chen et al.) | 2023, arXiv | SuGaR, NeuSG, GeoGaussian은 가장 작은 스케일을 최소화하여 Gaussian을 편평하게 만드는 정규화 항을 제안하여 얇은 Gaussian이 복잡한 표면에 정렬되도록 합니다. |
| **GS2Mesh** (Wolf et al.) | 2024, ECCV | 노이지한 3DGS 표현과 매끄러운 3D 메시 표현 간의 격차를 해소하기 위한 새로운 접근법을 제안하며, Gaussian 속성에서 직접 기하학을 추출하는 대신 사전 학습된 stereo-matching 모델을 통해 기하학을 추출합니다. |
| **PGSR** (Chen et al.) | 2024, IEEE TVCG | PGSR은 편향 없는 깊이 렌더링과 단일 및 다중 뷰 정규화 손실을 제안하여 기하학적 일관성을 보존합니다. |
| **FDS** (ICLR 2025) | 2025, ICLR | FDS는 매칭 사전 정보를 활용하여 절대 스케일을 복원하고 Gaussian radiance field의 기하학적 품질을 크게 향상시킵니다. |

### 5.2 기술 발전 흐름 분석

```
NeRF (2020) → Instant-NGP (2022) → 3DGS (2023)
                                        ↓
                              SuGaR (2023/CVPR2024) ← 정규화 + Poisson
                                        ↓
                    ┌──────────────┼──────────────┐
                  2DGS           GOF            GS2Mesh
               (SIGGRAPH'24)  (ACM TOG'24)    (ECCV'24)
                    ↓              ↓               ↓
                 PGSR          3DGSR          FDS (ICLR'25)
              (IEEE TVCG)   (arXiv'24)     CityGaussianV2
```

### 5.3 핵심 비교 분석

| 측면 | SuGaR | 2DGS | GOF | GS2Mesh |
|------|-------|------|-----|---------|
| **표현** | 3D Gaussian + Mesh (하이브리드) | 2D Gaussian Surfels | 3D Gaussian + Opacity Field | 3DGS + Stereo Prior |
| **메시 추출** | Poisson Reconstruction | TSDF Fusion | Marching Cubes | Stereo Depth → TSDF |
| **정규화** | Density/SDF Eikonal | Depth distortion + Normal consistency | Opacity Field | External stereo model |
| **학습 시간** | ~15-25분 (총 파이프라인) | ~30분 | ~30-60분 | 3DGS + 소량 오버헤드 |
| **편집 가능성** | ✅ (메시 바인딩) | ❌ | ❌ | ❌ |
| **배경 재구성** | 중간 | 2DGS는 배경 기하학 재구성에 실패합니다. | 우수 | 우수 |

---

## 참고 자료 및 출처

1. **Guédon, A. & Lepetit, V.** "SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering." *CVPR 2024*. arXiv:2311.12775
2. **SuGaR 공식 프로젝트 페이지**: https://imagine.enpc.fr/~guedona/sugar/
3. **SuGaR GitHub Repository**: https://github.com/Anttwo/SuGaR
4. **arXiv HTML 전체 논문**: https://arxiv.org/html/2311.12775v3
5. **CVPR 2024 공식 PDF**: https://openaccess.thecvf.com/content/CVPR2024/papers/Guedon_SuGaR_Surface-Aligned_Gaussian_Splatting_for_Efficient_3D_Mesh_Reconstruction_and_CVPR_2024_paper.pdf
6. **IEEE Xplore**: https://ieeexplore.ieee.org/iel8/10654794/10654797/10655755.pdf
7. **Hugging Face Papers 리뷰**: https://huggingface.co/papers/2311.12775
8. **Liner Quick Review**: https://liner.com/review/sugar-surfacealigned-gaussian-splatting-for-efficient-3d-mesh-reconstruction-and
9. **Kerbl, B. et al.** "3D Gaussian Splatting for Real-Time Radiance Field Rendering." *SIGGRAPH 2023*
10. **Huang, B. et al.** "2D Gaussian Splatting for Geometrically Accurate Radiance Fields." *SIGGRAPH 2024*
11. **Yu, Z. et al.** "Gaussian Opacity Fields: Efficient Adaptive Surface Reconstruction in Unbounded Scenes." *ACM TOG 2024*. https://niujinshuchong.github.io/gaussian-opacity-fields/
12. **Wolf, Y. et al.** "GS2Mesh: Surface Reconstruction from Gaussian Splatting via Novel Stereo Views." *ECCV 2024*. https://gs2mesh.github.io/
13. **FDS (Flow-Depth Supervision)**, *ICLR 2025*. https://nju-3dv.github.io/projects/fds/
14. **Chen, H. et al.** "NeuSG: Neural Implicit Surface Reconstruction with 3D Gaussian Splatting Guidance." arXiv:2312.00846, 2023
15. **Petrovska, O. & Jutzi, B.** "Seeing beyond vegetation: A comparative occlusion analysis..." *ScienceDirect*, 2025
16. **3DGS and Beyond Docs**: https://github.com/yangjiheng/3DGS_and_Beyond_Docs
17. **3DV Tutorial — Surface Reconstruction and 3DGS**: https://3dgstutorial.github.io/3dv_part4.pdf
18. **Semantic Scholar — SuGaR Figure Analysis**: https://www.semanticscholar.org/paper/SuGaR-Guédon-Lepetit/e3f80d950e6f841bd7eea4c24d4e1e5aa2bd85c7

> **주의사항**: 본 분석에서 제시한 수식은 논문의 공개된 내용과 공식 프로젝트 페이지를 기반으로 재구성한 것이며, 일부 수식의 세부 표기는 논문 원문을 직접 확인하시기를 권장합니다. 특히 Equation (6)에 대한 수정이 있었음이 공식 프로젝트 페이지에서 언급되고 있습니다.
