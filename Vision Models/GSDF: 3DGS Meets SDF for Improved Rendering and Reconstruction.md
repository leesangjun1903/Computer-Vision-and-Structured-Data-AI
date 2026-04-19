
# GSDF: 3DGS Meets SDF for Improved Rendering and Reconstruction 

> **논문 정보**
> - **제목**: GSDF: 3DGS Meets SDF for Improved Neural Rendering and Reconstruction
> - **저자**: Mulin Yu, Tao Lu, Linning Xu, Lihan Jiang, Yuanbo Xiangli, Bo Dai
> - **학회**: NeurIPS 2024
> - **arXiv**: [2403.16964](https://arxiv.org/abs/2403.16964) (v1: 2024.03.25, v2: 2024.10.13)
> - **GitHub**: [city-super/GSDF](https://github.com/city-super/GSDF)
> - **프로젝트 페이지**: [city-super.github.io/GSDF](https://city-super.github.io/GSDF/)

---

## 1. 핵심 주장과 주요 기여 요약

GSDF는 3D Gaussian Splatting(3DGS)의 장점과 neural Signed Distance Field(SDF)를 결합하는 **새로운 이중 분기(dual-branch) 아키텍처**를 제안하며, 핵심 아이디어는 두 분기 각각의 강점을 활용하고 **상호 가이던스(mutual guidance)와 결합 감독(joint supervision)**을 통해 한계를 극복하는 것입니다.

요약하면, GSDF는 3DGS와 SDF의 이중 분기 아키텍처를 결합하여 **렌더링 품질과 재건 품질을 모두 향상**시킵니다.

### 주요 기여 3가지


1. **GS→SDF 가이던스**: GS 분기에서 래스터화된 깊이(depth)를 SDF 분기의 레이 샘플링에 활용하여 볼륨 렌더링 효율을 높이고 지역 극솟값(local minima)을 회피합니다.
2. **SDF→GS 가이던스**: SDF 가이던스를 3DGS의 밀도 제어(density control)에 적용하여, 표면 근방에서는 Gaussian을 성장시키고 그 외 영역에서는 제거(pruning)합니다.
3. **기하학적 정렬(Geometry Alignment)**: 두 분기에서 각각 추정된 깊이와 법선(normal)을 정렬합니다.

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

멀티뷰 이미지로부터 3D 장면을 표현하는 것은 컴퓨터 비전과 그래픽스의 핵심 과제입니다. 이 문제는 크게 **렌더링(rendering)**과 **재건(reconstruction)**이라는 두 요구사항으로 나뉘며, 최신 렌더링 품질은 대개 신경 볼륨 렌더링 기법으로 달성되지만, 이는 색상 집계에 의존하고 장면의 기저 기하 정보를 무시합니다.

현재 연구들은 밀도 필드(density field)의 분포 혹은 기본 요소(primitive)의 형태를 제한하는 방식을 취하는데, 이로 인해 렌더링 품질이 저하되고 학습된 표면에 결함이 생깁니다. 이러한 방법들의 효과는 선택된 신경 표현 방식의 내재적 한계에 의해 제한되며, 특히 복잡하고 대형 장면에서 미세한 표면 세부 사항을 포착하기 어렵습니다.

3DGS 기반 방법들은 뷰 합성(view synthesis) 과제에 특화되어 있어 정확한 장면 기하에 대한 엄격한 제약을 부과하지 않습니다. 이로 인해 일반적으로 흐릿한(fuzzy) 부드러운 형태의 체적 밀도 필드를 학습하며, 이로부터 고품질 표면을 추출하기 어렵습니다.

### 2-2. 제안 방법 (수식 포함)

#### ① 전체 프레임워크

GSDF는 동기적으로 최적화되는 이중 분기 시스템을 제안하며, **렌더링과 재건을 하이브리드 표현으로 처리**합니다. 이 방법은 상호 가이던스와 결합 감독으로 두 태스크 간 균형을 맞추며, GS 분기는 렌더링, SDF 분기는 표면 재건을 각각 담당합니다.

#### ② GS 분기: 3D Gaussian Splatting

각 Gaussian은 아래와 같이 정의됩니다.

$$G(\mathbf{x}) = e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})}$$

여기서 $\boldsymbol{\mu} \in \mathbb{R}^3$은 평균(위치), $\boldsymbol{\Sigma} \in \mathbb{R}^{3\times3}$는 공분산 행렬입니다.

각 Gaussian은 평균 $\mu \in \mathbb{R}^3$과 공분산 $\Sigma \in \mathbb{R}^{3\times3}$으로 정의됩니다.

래스터화 기반 깊이 맵 $D_{GS}$는 다음과 같이 렌더링됩니다:

$$D_{GS}(\mathbf{r}) = \sum_{i} d_i \cdot \alpha_i \prod_{j < i}(1-\alpha_j)$$

여기서 $d_i$는 Gaussian $i$의 깊이, $\alpha_i$는 불투명도(opacity)입니다.

#### ③ SDF 분기 (NeuS 기반)

NeuS의 SDF-to-density 변환을 사용합니다:

$$\rho(t) = \max\!\left(\frac{-\frac{d\Phi_s}{dt}(f(\mathbf{r}(t)))}{\Phi_s(f(\mathbf{r}(t)))},\ 0\right)$$

여기서 $\Phi_s(x) = (1+e^{-sx})^{-1}$은 시그모이드 함수, $f$는 SDF, $s$는 학습 가능한 스케일 파라미터입니다.

SDF 볼륨 렌더링 색상:

$$\hat{C}_{SDF}(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\, \rho(t)\, c(\mathbf{r}(t), \mathbf{d})\, dt$$

#### ④ Mutual Guidance 메커니즘 (핵심)

**[Guidance 1] GS → SDF: 깊이 기반 레이 샘플링 가속화**

GS 분기가 렌더링한 깊이 맵을 이용해 SDF 분기의 레이 샘플링을 안내합니다. 절대 SDF 값 $|s|$를 조회하여 $2k|s|$ 범위 내에서 점을 샘플링합니다(예: $k=4$).

샘플링 구간:

$$t_i \sim \mathcal{U}\!\left[D_{GS}(\mathbf{r}) - 2k|s|,\ D_{GS}(\mathbf{r}) + 2k|s|\right]$$

**[Guidance 2] SDF → GS: SDF 기반 밀도 제어**

예측된 SDF 값이 GS 분기의 밀도 제어를 안내하며, 표면 근처에서 Gaussian을 성장시키고 표면에서 벗어난 것들을 제거합니다.

표면 근접 여부 판별:

$$\text{near-surface} \iff |f(\boldsymbol{\mu}_i)| < \epsilon$$

**[Guidance 3] 기하 정렬 손실**

두 분기의 깊이 및 법선을 정렬합니다:

$$\mathcal{L}_{depth} = \left\| D_{GS} - D_{SDF} \right\|_1$$

$$\mathcal{L}_{normal} = 1 - \langle \mathbf{n}_{GS},\ \mathbf{n}_{SDF} \rangle$$

곡률 손실(curvature loss)은 PermutoSDF 방식을 따르며, 임의의 점을 법선에 수직인 접선 평면(tangent plane) 내에서 무작위로 교란한 후, 원래 점과 교란된 점 사이의 법선 코사인 유사도로 측정합니다.

#### ⑤ 최종 손실 함수

$$\mathcal{L}_{total} = \mathcal{L}_{rgb}^{GS} + \mathcal{L}_{rgb}^{SDF} + \lambda_1 \mathcal{L}_{depth} + \lambda_2 \mathcal{L}_{normal} + \lambda_3 \mathcal{L}_{eikonal} + \lambda_4 \mathcal{L}_{curv}$$

Eikonal 정규화:

$$\mathcal{L}_{eikonal} = \mathbb{E}_{\mathbf{x}}\!\left[\left(\|\nabla f(\mathbf{x})\|_2 - 1\right)^2\right]$$

### 2-3. 모델 구조

시스템은 렌더링을 위한 3D Gaussian Splatting 분기와 표면 재건을 위한 Signed Distance Field 분기로 구성됩니다.

| 구성 요소 | 내용 |
|---|---|
| **GS 분기** | Scaffold-GS 기반, 앵커(anchor) 계층 구조 활용, 타일 기반 래스터라이저 |
| **SDF 분기** | Instant-NSR 기반(NeuS의 해시 인코딩 버전), 연속 암시적 표면 표현 |
| **상호작용** | 깊이 가이던스, 밀도 제어, 기하 정렬의 3가지 상호 가이던스 |
| **표면 추출** | Marching Cubes로 SDF 제로-레벨셋 추출 |

각 앵커는 특징 벡터와 함께 최적화되며, neural Gaussian의 색상, 중심, 분산, 불투명도를 예측합니다.

### 2-4. 성능 향상

상호 가이던스가 합성 장면과 실세계 장면 모두에서 견고성과 정확성을 보장하며, 실험 결과 SDF 최적화 과정이 더 세밀한 기하를 재건하는 방향으로 향상되고, Gaussian 기본 요소가 기저 기하와 정렬되어 렌더링에서 floater와 흐릿한 경계 아티팩트가 감소합니다.

이중 분기 설계를 통해 GS 분기는 표면에 밀착된 구조화된 기본 요소를 생성하여 floater를 줄이고 뷰 합성에서 디테일과 경계 품질을 향상시킵니다. 또한 SDF 분기의 수렴이 가속화됩니다.

**평가 데이터셋**: DTU, Tanks & Temples, MipNeRF360 (bicycle, bonsai, counter, garden, kitchen, room, stump)

특히 texture-less 영역에서 vanilla 3D Gaussian이 작은 누적 그래디언트로 인해 어려움을 겪는 반면, GSDF는 표면 영역에서 앵커를 성장시켜 이러한 한계를 극복하고 향상된 정확도와 장면 세부 사항을 제공합니다.

### 2-5. 한계

GSDF는 렌더링과 재건을 분리하는 이중 분기 최적화를 채택하여 **모델 복잡도가 증가하고 외관(appearance)과 기하(geometry) 사이의 일관성이 약화**될 수 있다는 한계가 있습니다.

SDF 최적화는 시간이 많이 소요되는 과정입니다. GS 분기가 SDF 분기의 수렴을 도와주지만, 추가적인 반복을 통해 더 많은 개선이 가능할 것으로 예상됩니다.

GSDF는 Gaussian의 깊이를 이용하여 SDF 샘플링을 안내하지만, SDF 분기에서도 여전히 색상을 렌더링해야 하므로 효율성에 한계가 있습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 일반화 강점의 근거

상호 가이던스는 합성 장면과 실세계 장면 **모두에서** 견고성과 정확성을 보장합니다.

견고성 테스트를 위해 2D-GS, Scaffold-GS, GSDF를 랜덤 초기화 Gaussian 기본 요소로 실험하였고, 정량적 결과에서 GSDF의 기하학적 가이던스의 이점이 두드러졌으며, 랜덤 입력에서도 우수한 성능과 안정성을 보였습니다.

GSDF는 **랜덤 초기화 포인트**로부터 시작해도 더 나은 렌더링 품질을 달성할 수 있습니다.

### 3-2. 일반화 가능성의 구조적 기반

GSDF 프레임워크는 **각 분기의 미래 발전을 수용할 수 있도록 설계**되어 있어, 더 나은 GS 표현이나 SDF 네트워크가 등장하면 손쉽게 통합 가능합니다.

이 설계는 Gaussian 기본 요소의 렌더링 효율성과 충실도를 유지하면서, NeuS에서 적용된 SDF 필드로부터 정확하게 장면 표면을 근사합니다.

### 3-3. 일반화의 한계 요인

이러한 방법들의 효과는 선택된 신경 표현 방식의 내재적 한계에 의해 여전히 제한되며, 특히 **더 크고 복잡한 장면**에서 미세한 표면 세부 사항을 포착하기 어렵습니다.

대규모 장면 표면 재건 문제를 다루기 위한 후속 연구(GigaGS)가 이미 등장하고 있어, GSDF의 대형 장면 일반화에 대한 한계가 인식되고 있음을 알 수 있습니다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 방법 | 연도 | 렌더링 | 재건 | 주요 특징 |
|---|---|---|---|---|
| **NeRF** | 2020 | ★★★★ | ★★ | 볼륨 렌더링, 느린 속도 |
| **NeuS** | 2021 | ★★★ | ★★★★ | SDF + 볼륨 렌더링 |
| **3DGS** | 2023 | ★★★★★ | ★★ | 실시간 래스터화, 기하 부정확 |
| **SuGaR** | 2024 | ★★★★ | ★★★ | Gaussian → Mesh 바인딩 |
| **2DGS** | 2024 | ★★★★ | ★★★★ | 2D Gaussian 서페이스 |
| **NeuSG** | 2024 | ★★★ | ★★★★ | NeuS + 3DGS 결합, MVS 필요 |
| **3DGSR** | 2024 | ★★★★ | ★★★★ | SDF 값으로 GS 감독 |
| **GSDF (ours)** | 2024 | ★★★★★ | ★★★★★ | 듀얼 분기, 상호 가이던스 |

동시 연구인 NeuSG는 NeuS와 3DGS를 결합하여, 3DGS의 법선과 NeuS가 예측한 법선을 정렬하도록 납작한 Gaussian을 장려했습니다. NeuS는 3DGS에서 파생된 포인트 클라우드로 정규화되었으며, 예측된 SDF 값이 0에 가깝도록 강제되었습니다.

그러나 NeuSG는 Vis-MVSNet을 사용하여 고밀도의 구조화된 포인트 클라우드를 얻어야 했으며, MVS 실행은 시간 집약적이고 배경 영역 처리에 어려움이 있었습니다.

3DGSR도 3DGS의 기하를 SDF 필드와 정렬시켰습니다. 신경 표면 재건의 발전에도 불구하고, Gaussian 기본 요소의 명시적 정규화로 인한 충실도 격차가 지속됩니다. GSDF는 형태를 제한하는 대신 기하학적 단서로 기본 요소를 최적화하여 구조적 완전성을 높이면서 표현력을 유지합니다.

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5-1. 향후 연구에 미치는 영향

**① 하이브리드 표현 패러다임의 선도**

GSDF는 렌더링에 전용 GS 분기, 신경 표면 학습에 전용 SDF 분기로 구성된 견고한 프레임워크로, 두 분기 간 상호 가이던스를 활용하여 렌더링과 재건 모두에서 최신 성능을 달성합니다.

**② 다양한 응용으로의 확장**

이미 반사 물체 재조명(relighting)을 위한 GS-ROR 등 GSDF의 SDF 사전 정보를 활용한 후속 연구들이 등장하고 있습니다.

**③ 물리 시뮬레이션과의 연계**

PhysGaussian을 이용한 GSDF 체크포인트 시뮬레이션이 가능하며, GSDF는 특히 섬세한 기하, texture-less 및 관측이 적은 영역에서 더 넓은 장면에서도 기준선을 능가합니다.

### 5-2. 향후 연구 시 고려할 점

**① 계산 효율성 개선**

SDF 최적화는 여전히 시간 집약적입니다. GS 분기가 SDF 수렴을 돕지만, 더 많은 반복으로 추가 개선이 가능하며, 동일한 반복 횟수에서 Instant-NSR보다 우수한 재건 품질을 보입니다.

**② 분기 간 일관성 강화**

GSDF의 이중 분기 최적화는 렌더링과 재건을 분리하여 **모델 복잡도를 높이고 외관과 기하 사이의 일관성을 약화**시킬 수 있습니다. 향후 연구는 두 표현 간의 더 강한 결합 메커니즘을 모색해야 합니다.

**③ 대형·동적 장면으로의 확장**

현재 방법의 효과는 선택된 신경 표현의 내재적 한계에 의해 제한되며, 특히 더 크고 복잡한 장면에서 미세한 표면 세부 사항을 포착하기 어렵습니다. 동적 장면이나 대규모 야외 환경으로의 확장이 중요한 연구 과제입니다.

**④ 일반화를 위한 사전 정보(Prior) 활용**

현재 GSDF는 Gaussian 기본 요소를 잠재적 표면 근처에 위치시키도록 안내하고 SDF 수렴을 가속화합니다. 향후에는 대규모 사전 학습 모델이나 단안 깊이 추정(monocular depth)을 결합하여 일반화 성능을 더욱 강화할 수 있습니다.

**⑤ 다운스트림 태스크와의 통합**

통합 표현은 고충실도 렌더링과 정확한 기하 재건을 동시에 지원하여 다운스트림 애플리케이션을 더 잘 지원하는 방향으로 발전해야 합니다.

---

## 📚 참고 문헌 및 출처

| # | 제목 / 출처 |
|---|---|
| 1 | **GSDF: 3DGS Meets SDF for Improved Rendering and Reconstruction** — arXiv:2403.16964, NeurIPS 2024. https://arxiv.org/abs/2403.16964 |
| 2 | **GSDF 프로젝트 페이지** — https://city-super.github.io/GSDF/ |
| 3 | **GSDF GitHub 코드** — https://github.com/city-super/GSDF |
| 4 | **NeurIPS 2024 공식 논문** — https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea13534ee239bb3977795b8cc855bacc-Abstract-Conference.html |
| 5 | **NeurIPS 2024 포스터** — https://neurips.cc/virtual/2024/poster/93457 |
| 6 | **OpenReview** — https://openreview.net/forum?id=r6V7EjANUK |
| 7 | **Semantic Scholar** — https://www.semanticscholar.org/paper/GSDF:-3DGS-Meets-SDF-for-Improved-Rendering-and-Yu-Lu/45550a04a8e29ef949d2eba88052f85b5f7d896e |
| 8 | **ACM DL** — https://dl.acm.org/doi/10.5555/3737916.3742031 |
| 9 | **DiGS (후속 비교 논문)** — arXiv:2509.07493. https://arxiv.org/html/2509.07493v1 |
| 10 | **GS-ROR2 (응용 논문)** — https://arxiv.org/html/2406.18544 |
| 11 | **NeuSG** — arXiv:2312.00846 |
| 12 | **NeuS** — arXiv:2106.10689 (Wang et al., 2021) |
| 13 | **3DGS** — Kerbl et al., SIGGRAPH 2023 |
| 14 | **Scaffold-GS** — Lu et al., arXiv 2023 |
| 15 | **3DGSR** — arXiv:2404.00409 (Lyu et al., 2024) |
