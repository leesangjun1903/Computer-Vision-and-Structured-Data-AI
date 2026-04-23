
# Shape of Motion: 4D Reconstruction from a Single Video

> **논문 정보**
> - **저자:** Qianqian Wang, Vickie Ye, Hang Gao, Weijia Zeng, Jake Austin, Zhengqi Li, Angjoo Kanazawa
> - **학회:** ICCV 2025 (arXiv 최초 공개: 2024년 7월 18일)
> - **arXiv ID:** 2407.13764
> - **프로젝트 페이지:** https://shape-of-motion.github.io/

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

Shape of Motion은 단일 단안(monocular) 비디오로부터 4D 장면을 복원한다. 단안 동적 복원(Monocular Dynamic Reconstruction)은 문제 자체가 고도로 불량 조건(ill-posed)이라는 특성으로 인해 오랫동안 해결하기 어려운 비전 문제였으며, 기존 접근 방식들은 템플릿에 의존하거나, 거의 정적인 장면에서만 효과적이거나, 3D 모션을 명시적으로 모델링하지 못하는 한계가 있었다.

이 논문은 무심코 촬영된(casually captured) 단안 비디오로부터 세계 좌표계에서의 명시적이고 지속적인(persistent) 3D 모션 궤적을 갖는 일반적인 동적 장면을 복원하는 방법을 제안한다.

### 주요 기여 (Two Key Insights)

**첫 번째:** $\mathrm{SE}(3)$ 모션 베이시스(motion bases)의 컴팩트한 집합으로 장면 모션을 표현함으로써 3D 모션의 저차원 구조를 활용한다. 각 점의 모션은 이러한 베이시스들의 선형 결합으로 표현되며, 이를 통해 장면을 여러 강체(rigidly-moving) 그룹으로 소프트(soft) 분해하는 것을 가능하게 한다.

**두 번째:** 단안 깊이 맵(monocular depth maps)과 장거리(long-range) 2D 트랙을 포함하는 포괄적인 데이터 기반 사전(prior)을 활용하고, 이러한 잡음 있는 감독 신호들을 효과적으로 통합하여 동적 장면의 전역적으로 일관된(globally consistent) 표현을 도출하는 방법을 고안한다.

결과적으로, 이 방법은 단안 비디오에서의 **장거리 3D 추적(long-range 3D tracking)**과 **새로운 시점 합성(novel view synthesis)**을 결합한 새로운 방법을 제시하며, 글로벌 3D 가우시안(3D Gaussians) 집합으로 시간에 따라 이동하고 회전하는 밀집 동적 장면 요소를 모델링한다.

---

## 2. 해결 문제 · 제안 방법 · 모델 구조 · 성능 · 한계

### 2-1. 해결하고자 하는 문제

동영상 전반에 걸쳐 지속적인 형상(geometry)과 그것의 3D 모션을 복원하는 것은 물리적 세계를 이해하고 상호작용하는 데 매우 중요하다. 최근 정적(static) 3D 장면 모델링에서 인상적인 진보가 있었음에도 불구하고, 특히 단일 비디오에서 복잡한 동적 3D 장면의 형상과 모션을 복원하는 것은 여전히 열린 과제로 남아 있다.

대부분의 방법은 동기화된 다시점(multi-view) 비디오나 추가적인 LiDAR/깊이 센서에 의존하며, 최근의 단안 방법들은 일반적으로 3D 장면 모션을 연속 시간 간의 단기 장면 흐름(short-range scene flow)이나 변형 필드(deformation field)로 모델링한다.

---

### 2-2. 제안 방법 (수식 포함)

#### (A) 장면 표현: 3D Gaussian Splatting 기반

각 3D 가우시안은 위치 $\boldsymbol{\mu} \in \mathbb{R}^3$, 스케일 $\mathbf{s} \in \mathbb{R}^3$, 회전 쿼터니언 $\mathbf{q} \in \mathbb{R}^4$, 불투명도(opacity) $\alpha$, 구면 조화(spherical harmonics) 색상 $\mathbf{c}$로 정의된다. 공분산 행렬은 다음과 같이 구성된다:

$$\Sigma = R S S^T R^T$$

여기서 $S$는 스케일 행렬, $R$은 쿼터니언으로부터 유도된 회전 행렬이다.

#### (B) $\mathrm{SE}(3)$ 모션 베이시스 표현 (핵심 기여)

$\mathrm{SE}(3)$ 모션 베이시스의 컴팩트한 집합으로 장면 모션의 저차원 구조를 활용한다. 각 점의 모션은 이러한 베이시스들의 선형 결합으로 표현되며, 이를 통해 장면을 여러 강체 이동 그룹으로 소프트 분해한다.

구체적으로, $K$개의 $\mathrm{SE}(3)$ 베이시스 $\{\mathbf{B}\_k(t)\}_{k=1}^{K}$를 정의하면, $i$번째 3D 가우시안의 시각 $t$에서의 변환(transformation)은 다음과 같이 표현된다:

$$\mathbf{T}_i(t) = \sum_{k=1}^{K} w_{ik} \cdot \mathbf{B}_k(t), \quad \mathbf{B}_k(t) \in \mathrm{SE}(3)$$

여기서 $w_{ik}$는 $i$번째 점이 $k$번째 베이시스에 기여하는 가중치(coefficient)이며, 각 점에 대해 학습된다. 이 공식화는 장면의 **소프트 분해(soft decomposition)**를 가능하게 한다.

시각 $t$에서 $i$번째 가우시안의 세계 좌표계 위치는:

$$\boldsymbol{\mu}_i(t) = \mathbf{R}_i(t)\boldsymbol{\mu}_i^{(0)} + \mathbf{t}_i(t)$$

여기서 $\mathbf{R}_i(t)$와 $\mathbf{t}_i(t)$는 $\mathbf{T}_i(t)$에서 분해된 회전 및 이동이다.

#### (C) 데이터 기반 Prior 통합

제안된 방법은 TAPIR(Tracking Any Point with per-frame Initialization and temporal Refinement)로부터 얻은 장거리 2D 트랙을 감독 신호로 명시적으로 활용하고, 이를 전역적으로 일관된 4D 표현으로 통합한다.

3D 트랙 손실( $\mathcal{L}\_{\text{track}}$ ), 깊이 정렬 손실( $\mathcal{L}\_{\text{depth}}$ ), 광도 손실( $\mathcal{L}_{\text{photo}}$ )의 조합으로 최적화가 이루어진다:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{photo}} \mathcal{L}_{\text{photo}} + \lambda_{\text{track}} \mathcal{L}_{\text{track}} + \lambda_{\text{depth}} \mathcal{L}_{\text{depth}} + \lambda_{\text{reg}} \mathcal{L}_{\text{reg}}$$

여기서 $\mathcal{L}_{\text{reg}}$는 모션의 부드러움(smoothness) 및 강체(rigidity)를 장려하는 정규화 항이다.

광도 재건 손실은 렌더링된 이미지와 실제 이미지 사이의 일관성을 강제하며, 모션 지역성 손실은 가우시안의 시간적 진화를 정규화한다. 이 지역성 항은 등방성(isometry), 강체성(rigidity), 상대 회전(relative rotation), 속도(velocity), 가속도(acceleration) 제약을 포함한다.

---

### 2-3. 모델 구조

이 방법은 단안 비디오에서 캡처된 동적 장면으로부터 장거리 3D 추적과 새로운 시점 합성을 결합한 새로운 방법을 제시하며, 시간에 따라 이동하고 회전하는 글로벌 3D 가우시안 집합으로 밀집 동적 장면 요소를 모델링한다.

주요 파이프라인은 다음과 같다:

1. **초기화 단계:** 단안 깊이 추정 모델(Depth Anything 등)로부터 깊이 맵 추출, TAPIR/CoTracker 등으로부터 2D 장거리 트랙 추출
2. **3D 가우시안 초기화:** 깊이 맵을 활용해 각 프레임에서 3D 가우시안 초기화
3. **$\mathrm{SE}(3)$ 모션 베이시스 학습:** 모든 프레임에 걸쳐 공유되는 $K$개의 $\mathrm{SE}(3)$ 베이시스 및 각 가우시안의 가중치 계수 공동 최적화
4. **전역 일관성 최적화:** 2D 트랙, 깊이, 렌더링 손실을 결합하여 전역적으로 일관된 4D 장면 표현 생성

이 연구는 단일 비디오로부터의 고품질 새로운 시점 합성과 전역적으로 일관된 장거리 3D 추적을 동시에 달성하는 첫 번째 방법으로서 중요한 기여를 한다. 명시적인 $\mathrm{SE}(3)$ 모션 파라미터화와 노이즈가 있는 사전 정보들의 효과적인 융합은 단안 4D 복원에서 최신 기술을 진보시키는 핵심 기술 혁신을 나타낸다.

---

### 2-4. 성능 향상

실험 결과, 이 방법은 동적 장면에서의 장거리 3D/2D 모션 추정과 새로운 시점 합성 모두에서 최신 기술 수준의 성능을 달성한다.

단안 캡처 환경에서 문제의 불량 조건적(ill-posed) 특성을 극복하기 위해, 잡음 있는 데이터 기반 관측들을 장면의 외관, 형상, 모션에 대한 전역적으로 일관된 추정으로 통합하도록 모델을 설계하였다. 합성 및 실제 벤치마크 모두에 대한 광범위한 평가를 통해 2D/3D 장거리 추적 및 새로운 시점 합성 작업 모두에서 이전 최신 기술 방법들을 크게 능가하는 것을 입증하였다.

---

### 2-5. 한계

이 방법은 이동하는 객체의 마스크 생성을 위해 사용자 입력에 의존한다. 미래의 유망한 연구 방향으로는 제약 없는 단안 비디오로부터 카메라 포즈, 장면 형상, 모션 궤적을 공동으로 추정하는 피드포워드 네트워크 접근 방식을 설계하는 것을 제시한다.

이 방법은 단안 깊이 추정과 같은 기성(off-the-shelf) 방법에 의존하며, 이는 오류가 있을 수 있다.

빠른 모션과 폐색(occlusion)은 이 방법에 도전적인 상황이다.

이 방법은 정확한 3D 형상 및 모션 추정에 의존하며, 이는 매우 복잡하거나 폐색된 장면에서는 어려울 수 있다. 또한, 평가가 비교적 통제된 데이터셋에 한정되어 있으며, 더 현실적인 야생(in-the-wild) 비디오에서의 성능은 불명확하다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 현재 일반화를 위한 설계적 강점

이 접근 방식은 변형 가능한 객체(deformable objects), 관절체(articulated bodies), 객체 간의 복잡한 상호작용을 포함한 광범위한 동적 장면에 적용 가능하다.

포괄적인 데이터 기반 사전(data-driven priors)을 활용하고, 잡음 있는 감독 신호들을 효과적으로 통합하여 동적 장면의 전역적으로 일관된 표현을 얻는다.

**일반화를 위한 핵심 설계 요소:**

- **$\mathrm{SE}(3)$ 저차원 구조 활용:** 물리적으로 의미 있는 강체 변환 그룹 $\mathrm{SE}(3)$을 모션 베이시스로 사용함으로써, 특정 도메인에 국한되지 않고 일반적인 강체·비강체 복합 모션을 모두 표현할 수 있다. 이는 모델이 다양한 장면 유형에 걸쳐 일반화될 수 있는 구조적 귀납 편향(inductive bias)을 제공한다.

- **기성 사전 정보 활용:** 단안 깊이 맵, 장거리 2D 트랙 등의 기성 데이터 기반 사전 정보를 적극 활용하고, 이러한 잡음 있는 감독 신호들을 효과적으로 통합한다. 이는 특정 데이터셋에 학습된 표현에 의존하는 것이 아니라, 범용적인 비전 모델로부터 사전 정보를 추출하므로 여러 도메인에서의 일반화에 유리하다.

### 3-2. 일반화 향상을 위한 잠재적 방향

유망한 연구 방향으로는 제약 없는 단안 비디오로부터 카메라 포즈, 장면 형상, 모션 궤적을 공동으로 추정하는 **피드포워드 네트워크 방식**을 설계하는 것이 있다. 이는 현재의 최적화 기반(optimization-based) 접근법이 갖는 장면별(per-scene) 최적화의 한계를 극복할 수 있다.

VAE 기반 모델링은 일반적으로 잘 구조화된 잠재 분포를 형성하기 위해 대규모 다양한 훈련 데이터가 필요하며, 제한되고 협소한 4D 데이터셋으로만 훈련되면 복잡한 모션 패턴을 포착하지 못하고 일반화가 저하된다. 고품질 4D 훈련 데이터의 희소성을 고려할 때, 4D 생성을 3D 형상 생성과 모션 복원의 조합으로 재정식화하는 것이 핵심 아이디어가 된다.

SoM, MoSca, Marbles 등의 방법들은 저차원 모션 베이시스를 사용하여 변형을 정규화하지만, 이들은 입력 시점에 가까운 검증 시점에서는 고품질 렌더링을 보여주지만 폐색 뒤의 모션이나 신뢰할 수 있는 가우시안을 명시적으로 모델링하지 않는다.

**일반화 향상을 위한 구체적 제안:**

| 방향 | 내용 |
|---|---|
| **Feed-forward 네트워크화** | 비디오-to-4D를 위한 feed-forward 모델 학습, 추론 속도와 일반화 모두 향상 |
| **불확실성 모델링** | 깊이/트랙의 신뢰도를 가우시안별로 모델링하여 노이즈 robustness 향상 |
| **대규모 동적 비디오 사전학습** | 더 강력한 비디오 기반 사전(video foundation model)을 활용 |
| **자동화된 동적/정적 분리** | 사용자 마스크 의존성 제거를 위한 자동 분리 모듈 |

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4-1. 연구에 미치는 영향

통합 표현은 시각적 합성(visual synthesis)과 명시적 모션 이해를 모두 필요로 하는 증강현실(AR), 로보틱스, 콘텐츠 제작 등의 새로운 응용 분야를 가능하게 한다.

**주요 영향:**

1. **동적 3D 표현의 패러다임 전환:** 기존의 변형 필드(deformation field) 중심 접근법에서 **명시적 $\mathrm{SE}(3)$ 모션 베이시스** 중심으로의 패러다임 전환을 이끈다.

2. **추적 + 새로운 시점 합성의 통합:** 이전까지 별개로 다루어졌던 두 과제를 하나의 프레임워크로 통합하였으며, 이는 후속 연구들이 두 과제를 동시에 다루도록 유도하는 방향성을 제시한다.

3. **기성 비전 모델 파이프라인의 표준화:** 이 논문 이후 여러 동시 연구들도 이 단안 4D 복원 설정을 다루고 있으며, 이들 모두 강력한 기성 데이터 기반 사전 정보를 활용하는 최적화 기반 접근 방식이다. 이는 해당 연구가 후속 연구의 방법론적 기준선을 설정하였음을 의미한다.

4. **후속 연구의 토대:** 실제로 SoM(Shape of Motion), MoSca, Marbles, 4D-Rotor 등 여러 후속 연구들이 저차원 모션 베이시스를 사용하여 변형을 정규화하는 방식을 채택하고 있다.

### 4-2. 앞으로 연구 시 고려할 점

1. **확장성 및 실시간성 문제:**
   현재 방법은 최적화 기반이므로 각 장면에 대해 별도의 최적화가 필요하다. 실시간 응용(AR/로봇 등)을 위해서는 피드포워드 네트워크화가 필수적이다.

2. **사용자 입력 의존성 제거:**
   현재 방법은 이동하는 객체의 마스크 생성을 위해 사용자 입력에 의존한다. 자동화된 동적/정적 분리 모듈의 통합이 실용적 배포를 위해 필요하다.

3. **불확실성 정량화:**
   이러한 목적 함수들은 입력 시점에서는 효과적이지만, 깊이, 광학 흐름 등과 같은 불안정한 2D 사전 정보에 크게 의존하기 때문에 폐색이나 극단적인 새로운 시점에서 취약하다. 따라서 불확실성 모델링의 통합이 중요한 연구 방향이다.

4. **카메라 포즈 추정 통합:**
   MoSca와 같은 후속 연구에서는 다른 포즈 추정 도구 없이 번들 조정(bundle adjustment)을 사용하여 카메라 초점 거리와 포즈를 추정할 수 있다. Shape of Motion은 카메라 포즈를 사전 알고 있다고 가정하므로, 포즈 불명 환경에서의 확장이 중요하다.

5. **비-강체(non-rigid) 모션의 정밀 처리:**
   $\mathrm{SE}(3)$ 베이시스는 강체 모션을 잘 표현하나, 복잡한 비강체 변형(예: 천, 유체)에서의 정밀도는 추가 연구가 필요하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 연도/학회 | 핵심 접근법 | 입력 | 비교 특징 |
|---|---|---|---|---|
| **NeRF** (Mildenhall et al.) | 2020, ECCV | Neural Radiance Field (정적) | 다시점 이미지 | 동적 장면 미지원, 기초 방법 |
| **Nerfies** (Park et al.) | 2021, ICCV | Deformable NeRF | 단안 비디오 | 소규모/준정적 장면, 단기 변형 |
| **HyperNeRF** (Park et al.) | 2021, SIGGRAPH | Hyperspace NeRF | 단안 비디오 | 토폴로지 변화 처리, 단기적 |
| **4D Gaussians (Wu et al.)** | 2024, CVPR | HexPlane+Deform. Field | 다시점 | 실시간 렌더링, 다시점 필요 |
| **Deformable 3DGS (Yang et al.)** | 2024, CVPR | MLP 변형 필드 | 단안 | 암묵적 변형, 과평활화(over-smoothing) 경향 |
| **Shape of Motion (본 논문)** | 2024→ICCV 2025 | $\mathrm{SE}(3)$ 모션 베이시스+3DGS | **단안** | 장거리 3D 추적+NVS 동시 달성 |
| **Dynamic Gaussian Marbles** | 2024, SIGGRAPH Asia | 등방성(isotropic) 가우시안, bottom-up 병합 | 단안 | 추적 성능 우수, NVS 품질 저하 |
| **MoSca** (Lei et al.) | 2024, arXiv | 4D Motion Scaffold+번들 조정 | 단안(포즈 불명) | 카메라 포즈 자동 추정, 더 자동화 |
| **MotionGS** | 2024, NeurIPS | 광학흐름 분해(Camera/Motion flow) | 단안 | 명시적 모션 제약, 인접 프레임 의존 |
| **Motion4D** | 2025 | 3DGS+세만틱/모션 결합 최적화 | 단안 | 세만틱 분할+추적+NVS 통합 |

**주요 비교 분석:**

Shape of Motion의 주요 비교 대상인 Deformable 3DGS는 변형 필드(deformation field) 접근법으로 동적 장면을 모델링하는데, 이는 Shape of Motion의 $\mathrm{SE}(3)$ 모션 베이시스 표현 대비 신규성을 보여주는 직접적인 비교 포인트가 된다.

Dynamic Gaussian Marbles의 경우, 기존의 4D Gaussian 방법들은 동기화된 다시점 비디오를 감독으로 가정하여 제어된 캡처 환경으로 사용이 제한되었으며, 단안 설정에서 기존 4D Gaussian 방법들이 극적으로 실패함을 보인다.

MoSca는 야생에서 무심코 캡처된 단안 비디오로부터 동적 장면을 복원하고 새로운 시점을 합성하도록 설계된 현대적인 4D 복원 시스템으로, 기초 비전 모델로부터의 사전 지식을 활용하여 컴팩트하고 부드럽게 모션/변형을 인코딩하는 새로운 Motion Scaffold 표현으로 비디오 데이터를 변환한다.

---

## 참고 자료 및 출처

1. **Shape of Motion 공식 프로젝트 페이지:** https://shape-of-motion.github.io/
2. **arXiv 논문 (v1):** Wang et al., "Shape of Motion: 4D Reconstruction from a Single Video," arXiv:2407.13764, 2024. https://arxiv.org/abs/2407.13764
3. **ICCV 2025 CVF 공개 논문:** https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_Shape_of_Motion_4D_Reconstruction_from_a_Single_Video_ICCV_2025_paper.pdf
4. **alphaXiv 분석 페이지:** https://www.alphaxiv.org/overview/2407.13764v2
5. **Hugging Face 논문 페이지:** https://huggingface.co/papers/2407.13764
6. **MoSca 논문:** Lei et al., "MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds," arXiv:2405.17421, 2024. https://arxiv.org/abs/2405.17421
7. **Dynamic Gaussian Marbles:** Stearns et al., SIGGRAPH Asia 2024. https://dl.acm.org/doi/full/10.1145/3680528.3687681
8. **4D Gaussian Splatting (Wu et al.):** CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_4D_Gaussian_Splatting_for_Real-Time_Dynamic_Scene_Rendering_CVPR_2024_paper.pdf
9. **MotionGS:** NeurIPS 2024 논문. https://arxiv.org/html/2410.07707v1
10. **Motion4D:** arXiv:2512.03601, 2025. https://arxiv.org/html/2512.03601
11. **Uncertainty in 4D Gaussian Splatting (USplat4D):** arXiv:2510.12768. https://arxiv.org/html/2510.12768
12. **Gaussian Sequences with Multi-Scale Dynamics:** arXiv:2602.13806. https://arxiv.org/html/2602.13806
13. **Gaussian Splatting Wikipedia:** https://en.wikipedia.org/wiki/Gaussian_splatting
14. **ADS Abstract:** https://ui.adsabs.harvard.edu/abs/2024arXiv240713764W/abstract

> ⚠️ **주의:** 본 논문의 상세 수식(특히 손실 함수의 정확한 구성 요소 및 계수)은 공개된 HTML 버전(https://arxiv.org/html/2407.13764v1)에서 일부 수식이 렌더링되지 않아, 논문의 방법론적 설명과 관련 후속 연구의 분석을 참조하여 재구성하였습니다. 세부 수식의 정확한 형태를 확인하시려면 PDF 원문을 직접 참조하시기를 권장합니다.
