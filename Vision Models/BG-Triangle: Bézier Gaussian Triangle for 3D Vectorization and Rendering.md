# BG-Triangle: Bézier Gaussian Triangle for 3D Vectorization and Rendering

## 1. 핵심 주장 및 주요 기여 요약

**BG-Triangle (Bézier Gaussian Triangle)** 은 3D 장면 표현에서 **결정론적(mesh)** 과 **확률론적(NeRF/3DGS)** 양극단 사이에 위치하는 새로운 **하이브리드 벡터화 표현**을 제안합니다. 이 논문의 핵심 기여는 다음과 같습니다.

**핵심 기여 1**: 형상 수준에서는 **Bézier 삼각 패치(parametric surface)** 를 사용하여 명시적 경계를 정의하고, 픽셀 수준에서는 **Gaussian 기반 sub-primitive**를 동적으로 생성하여 미분가능 렌더링을 보장하는 이중 표현 구조를 도입했습니다.

**핵심 기여 2**: **Discontinuity-Aware Alpha Blending** 기법을 통해 경계 영역에서 Gaussian의 확률적 불확실성(블러)을 억제하면서도 미분가능성을 유지합니다.

**핵심 기여 3**: 적응적 분할(splitting)과 가지치기(pruning) 전략을 통해 거친 점군(coarse point cloud) 초기화만으로도 효율적 학습이 가능하며, **3DGS 대비 50배 이상 적은 파라미터**(예: 343.5K vs 17.07M)로 비슷한 렌더링 품질을 달성했다고 보고합니다.

---

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

기존 미분가능 렌더링(differentiable rendering) 방식들은 양극단의 한계를 가집니다.

- **3D Mesh 기반**: 명시적 연결성으로 형상은 잘 보존하지만 미분가능성이 제한적이고 초기화에 민감함
- **NeRF / 3DGS 기반**: 미분가능성과 최적화는 우수하지만 Gaussian 커널의 본질적 저역통과(low-pass) 특성 때문에 **객체 경계에서 블러**가 발생하며, 학습 시점에서 멀리 떨어진 close-up 뷰에서 품질 저하가 두드러짐

저자들은 "표현 스펙트럼(representation spectrum)" 관점에서 **결정론적 경계 정의력 + 확률론적 미분가능성**을 동시에 갖는 새로운 위치를 찾고자 했습니다.

### 2.2 제안 방법: 수식 정리

#### (a) Bézier 삼각 패치 정의

차수 $n$ 의 Bézier 삼각형은 제어점 $\mathbf{p}_{i,j,k} \in \mathbb{R}^3$ ($i+j+k=n$, $i,j,k \geq 0$) 과 무게중심좌표 $(u,v,w)$ ($u+v+w=1$) 로 정의됩니다.

$$\mathbf{S}(u,v,w) = \sum_{i=0}^{n}\sum_{j=0}^{n-i} B_{i,j,k}^{n}(u,v,w)\,\mathbf{p}_{i,j,k}$$

여기서 Bernstein 다항식은:

$$B_{i,j,k}^{n}(u,v,w) = \frac{n!}{i!\,j!\,k!}\, u^{i} v^{j} w^{k}$$

#### (b) 픽셀 정렬 Sub-Primitive 생성

테셀레이션 후 각 전경 픽셀 $\mathbf{q}$ 에 대해 좌표맵 $\mathbf{I}\_{uv}$, 인덱스맵 $\mathbf{I}_{id}$ 을 사용하여 3D 좌표를 보간합니다.

$$\mathbf{S}\_{\mathbf{q}}(\mathbf{I}_{uv}(\mathbf{q}), \mathbf{I}_{id}(\mathbf{q})) = \sum_{i=0}^{n}\sum_{j=0}^{n-i} B_{i,j,k}^{n}(\mathbf{I}_{uv}(\mathbf{q}))\,\mathbf{p}_{i,j,k}(\mathbf{I}_{id}(\mathbf{q}))$$

확산색(diffuse color) $\mathbf{c}\_{\mathbf{q}}$ 도 색 제어점 $\mathbf{p}_{i,j,k}^{c}$ 으로 동일하게 보간합니다.

$$\mathbf{c}_{\mathbf{q}} = \sum_{i=0}^{n}\sum_{j=0}^{n-i} B_{i,j,k}^{n}(\mathbf{I}_{uv}(\mathbf{q}))\,\mathbf{p}_{i,j,k}^{c}(\mathbf{I}_{id}(\mathbf{q}))$$

회전, 스케일, SH 계수 같은 다른 속성은 다중 해상도 2D 속성맵 $\mathbf{M}_h$ 으로부터 보간합니다.

$$\mathbf{a}_h(\mathbf{q}) = \Theta(\mathbf{M}_h(\mathbf{I}_{id}(\mathbf{q})), \mathbf{I}_{uv}(\mathbf{q}))$$

#### (c) Discontinuity-Aware Alpha Blending

경계 픽셀 $\mathbf{b}_i$ 의 영향 반경 $\sigma_i$ 내에서 블렌딩 계수 $w(\mathbf{q})$ 를 정의합니다.

$$w(\mathbf{q}) = \begin{cases} 0, & \text{if } \mathbf{I}_{id}(\mathbf{b}_i) \neq g \\ \gamma(\|\mathbf{q}-\mathbf{b}_i\|_2; \sigma_i), & \text{if } \mathbf{I}_{id}(\mathbf{b}_i) = \mathbf{I}_{id}(\mathbf{q}) \\ 1 - \gamma(\|\mathbf{q}-\mathbf{b}_i\|_2; \sigma_i), & \text{otherwise} \end{cases}$$

블러링 함수 $\gamma$ 는:

$$\gamma(d;\sigma) = \min\!\left(2^{\frac{d}{\sigma}-1}, 1\right)$$

이는 거리 $d$ 를 경계 반경 $\sigma$ 안에서 $[0.5, 1.0]$ 구간으로 매핑하여 부드럽게 전이시킵니다.

최종 알파 값은:

$$\alpha(\mathbf{q}) = o \cdot w(\mathbf{q}) \cdot \exp\!\left(-\tfrac{1}{2}(\mathbf{q}-\mu)^{\top}\Sigma^{-1}(\mathbf{q}-\mu)\right)$$

#### (d) 손실 함수

3DGS와 마찬가지로 광도 손실(photometric loss)을 사용합니다.

$$\mathcal{L} = (1-\lambda)\mathcal{L}_2 + \lambda\, \mathcal{L}_{\text{D-SSIM}}, \quad \lambda = 0.2$$

#### (e) 역전파 미분

제어점에 대한 손실 함수의 기울기는 연쇄 법칙으로 다음과 같이 계산됩니다.

$$\frac{\partial \ell}{\partial \mathbf{p}_{i,j,k}(\mathbf{I}_{id}(\mathbf{q}))} = \frac{\partial \ell}{\partial \mathbf{S}_{\mathbf{q}}}\, B_{i,j,k}^{n}(\mathbf{I}_{uv}(\mathbf{q}))$$

### 2.3 모델 구조 (3-stage pipeline)

(1) **Primitive Rasterization**: Bézier 삼각형을 작은 평면 삼각형으로 테셀레이션한 후 래스터화하여 좌표맵 $\mathbf{I}\_{uv}$, 인덱스맵 $\mathbf{I}_{id}$, 경계점 집합 $\mathcal{B}$ 를 생성합니다.

(2) **Sub-Primitive Generation**: 각 픽셀에서 동적으로 Gaussian sub-primitive를 생성합니다. 3DGS와 달리 **시점마다 다른 Gaussian 집합**이 동일한 BG-Triangle로부터 샘플링됩니다.

(3) **Discontinuity-Aware Alpha Blending**: 경계 인식 블렌딩 계수를 splatted Gaussian에 적용하여 경계 영역에서 선명한 에지를 보존합니다. 타일 기반 렌더링과 ID 기반 정렬·이진 검색으로 가속화합니다.

### 2.4 성능 향상

- **NeRF Synthetic** (343.5K params): SSIM 0.937 / PSNR 29.16 / LPIPS 0.050 → 동일 파라미터 수의 3DGS(0.922 / 27.18 / 0.103) 대비 모든 지표에서 우수, 특히 **LPIPS에서 약 2배 개선**
- **Close-up view** (40% 거리): SSIM 0.736 / PSNR 21.19 / LPIPS 0.306으로 모든 비교군 대비 최고 성능, 특히 경계 보존에서 우수
- **압축률**: 17.07M → 343.5K 로 ~50배 파라미터 감소

### 2.5 한계

저자들이 명시한 한계는 다음과 같습니다.

- 2D 이미지 벡터화처럼 약간의 품질 손실이 발생하며, **PSNR/SSIM 절대값에서는 full 3DGS보다 다소 낮음**
- **단일 레이어 sub-primitive**만 생성하므로 반투명·옥(translucency) 재질 처리 불가
- Cube 초기화 시 더 많은 반복과 하이퍼파라미터 민감도가 증가
- CUDA rasterization 구현이 비효율적이라 backward 시간이 큼 (Lego scene 기준 35분 학습)

---

## 3. 모델의 일반화 성능 향상 가능성

본 논문은 **명시적으로 "일반화(generalization)" 성능을 정량 평가하지는 않았지만**, 표현 구조 자체에 일반화 잠재력을 시사하는 여러 요소가 내재되어 있습니다. 다만 이는 **논문이 직접 입증한 결과가 아닌 구조적 추론**임을 명확히 합니다.

**(1) 학습 시점에서 벗어난 뷰포인트에 대한 강건성**: 3DGS는 학습 뷰의 픽셀 샘플링 비율에 맞춰 Gaussian이 over-fitting 되어 close-up에서 블러나 spike artifact가 발생합니다. 반면 BG-Triangle은 **벡터 표현 + 시점별 동적 Gaussian 샘플링** 구조로 인해 해상도 독립적(resolution-independent) 렌더링이 가능합니다. 실험적으로 학습 거리의 40% 거리(close-up)에서 다른 모든 baseline을 뛰어넘는 성능을 보였으며 (Tab. 2), 이는 **분포 외(out-of-distribution) 뷰포인트로의 일반화** 잠재력의 간접적 증거로 볼 수 있습니다.

**(2) Hard constraint를 통한 정규화 효과**: Scaffold-GS가 anchor-Gaussian 간 soft constraint만 부과하는 데 비해, BG-Triangle은 **하나의 Bézier 패치 내 Gaussian들이 동일한 SH 계수와 부드럽게 전이되는 색을 공유**하도록 강제합니다. 이는 모델 자유도를 줄여 **암묵적 정규화(implicit regularization)** 역할을 하므로, 적은 학습 뷰만으로도 안정적 수렴이 가능할 수 있습니다 (sparse-view 일반화 잠재력).

**(3) Level-of-Detail(LoD) 적응성**: 거친 단계에서는 하나의 primitive가 큰 영역을 표현하고, 세밀한 단계에서는 분할되어 더 많은 디테일을 담을 수 있습니다. 이는 **다중 해상도/다중 거리 렌더링**에서의 일반화에 유리합니다.

**(4) 한계점 — 일반화 검증의 부족**: 그러나 논문은 NeRF Synthetic과 마스킹된 Tanks & Temples만 평가했으며, 다음에 대한 검증이 부족합니다.

- Sparse view (예: 3-view, 10-view) 환경에서의 일반화
- Cross-scene generalization (학습된 표현의 다른 장면 전이)
- 동적 장면, 비강체 객체로의 확장
- 반투명/반사 재질 (저자도 한계로 인정)

따라서 **일반화 성능 향상은 표현의 구조적 잠재력 수준**이며, 본격적인 일반화 검증은 향후 연구에서 수행되어야 합니다.

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 향후 연구에 미치는 영향

**(1) 표현 스펙트럼의 재정의**: BG-Triangle은 NeRF·3DGS·Mesh 사이에 위치하는 새로운 좌표를 제시합니다. 향후 연구는 "어느 정도의 결정론적 구조 + 어느 정도의 확률적 유연성"이 특정 응용에 최적인지를 탐색하는 방향으로 확장될 가능성이 높습니다.

**(2) 벡터화 3D의 부활**: DiffVG, Im2Vec 등 2D 벡터 그래픽스 연구를 3D로 확장하는 데 BG-Triangle이 강력한 baseline을 제공합니다. 특히 3D 라인 스트로크 추출(Fig. 10) 등 **시맨틱 인식이나 스타일라이제이션**에 활용 가능성이 큽니다.

**(3) 평가 메트릭 재고**: 저자들이 강조하듯 PSNR/SSIM이 **경계 선명도를 포착하지 못하는 문제**를 부각시켰으며, 이는 향후 NVS(novel view synthesis) 평가 프로토콜 개선 논의의 촉매가 될 수 있습니다.

**(4) 경량화 모델 트렌드**: 343.5K 파라미터로 17M 파라미터 3DGS와 경쟁한 점은 **모바일·VR/AR·임베디드 환경**에서의 3D 렌더링 응용을 활성화할 수 있습니다.

### 4.2 향후 연구 시 고려할 점

- **반투명·복잡 재질 확장**: 다층(multi-layer) sub-primitive 또는 BSDF 모델링과의 결합 필요
- **동적 장면 확장**: 4D Bézier 패치 또는 시간 의존적 제어점 변형(deformation)
- **초기화 강건성**: cube 초기화의 어려움이 시사하듯, 초기 점군 품질에 대한 의존성을 줄이는 메타학습/사전학습 전략
- **렌더링 가속**: 현재 CUDA 구현이 비효율적이라 실시간성 개선 필요 (예: 하드웨어 테셀레이터 활용)
- **자동 평가 지표**: 경계 보존을 정량화하는 **edge-aware metric** 개발

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 핵심 아이디어 | BG-Triangle과의 차이 |
|------|--------------|---------------------|
| **3DGS** (Kerbl et al., 2023) | 비등방성 3D Gaussian의 고정 집합으로 장면 표현 | BG-Triangle은 동적 Gaussian 샘플링 + 명시적 경계 |
| **Mip-Splatting** (Yu et al., 2024) | 2D Mip 필터 + 3D smoothing 필터로 안티앨리어싱 | 주파수 제한으로 sharp edge 모델링이 어려움 |
| **2DGS** (Huang et al., 2024) | 2D 평면 disk로 표면 정렬 | 표면 재구성에 강하나 vector 경계 표현은 없음 |
| **Scaffold-GS** (Lu et al., 2024) | Anchor 기반 + MLP로 Gaussian 속성 예측 | Soft constraint만 사용, BG-Triangle은 hard constraint |
| **GES** (Hamdi et al., 2024) | Generalized Exponential 커널로 sharp signal 표현 | 커널 형태만 변경, 여전히 명시적 경계 없음 |
| **2DGH** (Yu et al., 2024) | Hermite 급수로 변조한 2D Gaussian으로 sharp edge 표현 | BG-Triangle과 가장 유사한 동기, 그러나 패치 단위 vector 표현은 아님 |
| **DRK** (Deformable Radial Kernel, 2024) | 학습 가능한 radial basis로 edge sharpness 직접 제어 | BG-Triangle은 패치-수준 구조까지 부과 |
| **EdgeGaussians** (2024) | 3D edge mapping에 특화된 GS | BG-Triangle은 벡터 패치 자체가 표현 단위 |
| **COB-GS** (2025) | 3DGS 분할에 boundary-adaptive splitting | 본 논문과 유사한 동기지만 segmentation 응용 |

수식적으로 비교하면, **3DGS의 픽셀 색상**은 다음과 같이 정해진 Gaussian 집합 $\{\mathcal{G}_i\}$ 의 alpha-blending으로 표현됩니다.

$$C = \sum_{i=0}^{n-1} T_i \alpha_i c_i + T_n c_{\text{bg}}, \quad T_i = \prod_{j=0}^{i-1}(1-\alpha_j)$$

반면 **BG-Triangle**은 위 식을 그대로 사용하되, $\alpha_i$ 가 추가로 블렌딩 계수 $w(\mathbf{q})$ 로 변조됩니다.

$$\alpha(\mathbf{q}) = o \cdot w(\mathbf{q}) \cdot \exp\!\left(-\tfrac{1}{2}(\mathbf{q}-\mu)^{\top}\Sigma^{-1}(\mathbf{q}-\mu)\right)$$

이 단순한 변형이 **명시적 경계 정보를 alpha-blending에 주입**하는 핵심 메커니즘입니다.

---

## 참고자료 출처

**원 논문 (분석 대상)**
- Wu, M., Dai, H., Yao, K., Tuytelaars, T., Yu, J. (2025). *BG-Triangle: Bézier Gaussian Triangle for 3D Vectorization and Rendering*. arXiv:2503.13961v1. (업로드된 PDF 문서)

**논문이 직접 인용한 주요 참고문헌**
- Kerbl, B. et al. (2023). *3D Gaussian Splatting for Real-Time Radiance Field Rendering*. ACM TOG 42(4). [3DGS]
- Mildenhall, B. et al. (2020). *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*. ECCV.
- Yu, Z. et al. (2024). *Mip-Splatting: Alias-Free 3D Gaussian Splatting*. CVPR.
- Lu, T. et al. (2024). *Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering*. CVPR.
- Fang, G., Wang, B. (2024). *Mini-Splatting: Representing Scenes with a Constrained Number of Gaussians*. arXiv:2403.14166.
- Huang, B. et al. (2024). *2D Gaussian Splatting for Geometrically Accurate Radiance Fields*. SIGGRAPH 2024.
- Farin, G. (1986). *Triangular Bernstein–Bézier Patches*. CAGD 3(2).
- Li, T-M. et al. (2020). *DiffVG: Differentiable Vector Graphics Rasterization*. ACM TOG 39(6).

**최신 연구 비교를 위해 검색·참조한 자료**
- Hamdi, A. et al. (2024). *GES: Generalized Exponential Splatting for Efficient Radiance Field Rendering*. arXiv:2402.10128. (https://arxiv.org/pdf/2402.10128)
- Yu, R. et al. (2024). *2DGH: 2D Gaussian-Hermite Splatting for High-quality Rendering and Better Geometry Reconstruction*. arXiv:2408.16982. (https://arxiv.org/pdf/2408.16982)
- *Deformable Radial Kernel Splatting* (2024). arXiv:2412.11752. (https://arxiv.org/pdf/2412.11752)
- *EdgeGaussians: 3D Edge Mapping via Gaussian Splatting* (2024). arXiv:2409.12886. (https://arxiv.org/html/2409.12886v2)
- Zhang, J. et al. (2025). *COB-GS: Clear Object Boundaries in 3DGS Segmentation*. arXiv:2503.19443. (https://arxiv.org/pdf/2503.19443)
- *Recent advances in 3D Gaussian splatting* (Springer Nature, 2024). Computational Visual Media. (https://link.springer.com/article/10.1007/s41095-024-0436-y)
- Liu, R. et al. (2024). *AtomGS: Atomizing Gaussian Splatting for High-Fidelity Radiance Field*. arXiv:2405.12369. (https://rongliu-leo.github.io/AtomGS/)

---

**주의사항**: 본 분석에서 "일반화 성능 향상 가능성"에 대한 논의는 논문이 명시적으로 일반화 평가를 수행한 것이 아니라 **구조적 특성으로부터 추론한 잠재력**입니다. 실제 sparse-view, cross-scene, OOD 일반화 성능은 논문에서 직접 검증되지 않았으므로 이를 단정할 수 없으며, 향후 연구에서 별도 검증이 필요합니다.
