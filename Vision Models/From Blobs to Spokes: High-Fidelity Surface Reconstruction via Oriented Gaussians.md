
# From Blobs to Spokes: High-Fidelity Surface Reconstruction via Oriented Gaussians

> **저자:** Diego Gomez, Antoine Guédon, Nissim Maruani, Bingchen Gong, Maks Ovsjanikov
> **소속:** LIX, École Polytechnique / Inria, Côte d'Azur, France
> **arXiv:** [2604.07337](https://arxiv.org/abs/2604.07337) (April 2026)
> **코드:** [github.com/diego1401/GaussianWrapping](https://github.com/diego1401/GaussianWrapping)

---

## 1. 핵심 주장 및 주요 기여 요약

### 🔑 핵심 주장

3D Gaussian Splatting(3DGS)은 빠른 Novel View Synthesis에서 혁신을 이뤘지만, opacity 기반의 formulation은 표면 추출을 근본적으로 어렵게 만든다. Signed Distance Field나 Occupancy 기반의 implicit 방법들과 달리, 3DGS는 전역적인 기하학적 필드(global geometric field)가 없어, 기존 접근법들은 블렌딩된 depth map의 TSDF 융합 같은 휴리스틱에 의존할 수밖에 없다.

이 논문의 핵심 주장은, 표준 3DGS는 Gaussian을 대칭적인 "blob" 밀도로 취급하는데, 이는 비대칭적인 경계인 표면의 본질과 충돌한다. 이 대칭 편향(symmetry bias)은 재건된 표면을 안쪽 또는 바깥쪽으로 끌어당기고, 특히 얇은 구조물에서 반대편 Gaussian tail이 서로 간섭하여 큰 문제를 야기한다는 것이다.

이에 대한 해결책으로, **Gaussian Wrapping** 프레임워크를 제안한다: 다중 시점 RGB 이미지로부터 3D Gaussian을 확률론적 방향성 표면 요소(stochastic oriented surface elements)로 해석하여, watertight하고 텍스처가 있는 전체 3D 장면의 표면 메시를 재건한다.

### 📌 주요 기여 (Contributions)

① **Oriented Gaussians 도입**: 멀티뷰 설정에서 Oriented Gaussian과 그 학습 전략을 소개하고, 3DGS와 implicit 표면 재건 사이의 이론적 연결을 Objects as Volumes 프레임워크에 영감을 받아 유도하며, 이는 추가적인 학습 파라미터 없이 임의 위치에서 법선 및 점유 필드의 폐쇄형 표현식(closed-form expressions)을 이끌어 낸다.

② **Consistency Loss + Densification**: 기하학적 구멍을 닫아 전체 표면을 감싸도록 Gaussian을 강제하는 새로운 consistency loss와 전용 densification 전략을 도입하여 방향성 primitive의 완전한 shell을 보장한다.

③ **Primal Adaptive Meshing(PAM)**: 도출된 Gaussian 필드를 활용해 제어 가능한 해상도의 고품질 watertight 메시를 생성하는 메시 추출 절차로, 자전거 바퀴살(bicycle spokes)같은 극도로 얇은 구조물 복원을 가능하게 한다.

④ **평가 프로토콜 개선**: 표준 표면 평가 프로토콜의 근본적인 편향을 드러내고 두 가지 더 엄격한 대안을 제안한다.

---

## 2. 해결하고자 하는 문제, 제안하는 방법(수식 포함), 모델 구조, 성능 향상 및 한계

### 🎯 2.1 해결하고자 하는 문제

기존 방법들의 한계는 3DGS가 Gaussian을 대칭적인 구름(symmetric clouds)으로 취급한다는 데 있다. 이 논문은 각 Gaussian을 방향성 반공간(oriented half-space) 내에서만 기하학을 나타내는 방향성 확률론적 표면 요소(oriented probabilistic surface elements)로 재해석함으로써 이를 해결한다.

구체적으로 해결하려는 문제는 세 가지이다:

1. 3DGS의 대칭 Gaussian "blob" 표현이 비대칭적인 표면 경계 표현에 부적합하며, 얇은 구조물에서 반대쪽 Gaussian의 간섭이 발생하는 문제.
2. 전역적인 기하학 필드가 없어 TSDF 융합 같은 휴리스틱에 의존해야 하는 문제.
3. Gaussian이 밀봉된 경계(sealed boundary)를 형성하지 못해 표면이 불완전해지는 문제. 이를 각 Gaussian에 학습 가능한 방향 법선(oriented normal)을 부여하고 가시적인 Gaussian이 점유된 영역에서 멀리, 카메라 방향으로 향하도록 강제하는 wrapping 방식으로 해결한다.

---

### 🔬 2.2 제안 방법 (수식 포함)

#### ① Oriented Attenuation 정의

각 Gaussian은 이제 바깥을 향하는 면(outward-facing side)에서만 밀도 감쇠(density decay)를 모델링하고, 안쪽을 향하는 면(inward-facing side)은 완전히 점유된 것으로 취급한다. 이는 모든 Gaussian을 명확한 내부와 외부가 있는 확률론적 방향성 표면 요소로 재해석하는 것이다.

각 Gaussian $i$에 대해, 중심 $\mu_i$, 법선 $n_i \in \mathcal{S}^2$, 표준 Gaussian 함수 $G_i(x)$를 이용한 **Oriented Attenuation**은 다음과 같이 정의된다:

$$a_i(x) = \begin{cases} 1 - G_i(x) & \text{if } n_i^\top (x - \mu_i) \geq 0 \quad \text{(outward)} \\ 1 & \text{if } n_i^\top (x - \mu_i) < 0 \quad \text{(inward)} \end{cases}$$

#### ② Vacancy Field

Objects as Volumes(OaV) 프레임워크를 적용하여, 전체 장면의 **Vacancy Field** $v(x)$ (비점유 확률)는 다음과 같이 정의된다:

$$v(x) = \prod_{i=1}^{N} a_i(x) = \prod_{\{i \,:\, n_i^\top(x-\mu_i) \geq 0\}} \left(1 - G_i(x)\right)$$

#### ③ Gaussian Vector & Normal Field (핵심 이론적 기여)

Objects as Volumes 결과를 reciprocal attenuation에 적용하면 주요 이론적 기여인 closed-form 벡터 필드 $V$와 그 정규화인 Gaussian Normal Field $\mathcal{N}$을 얻는다:

$$V(x) := \nabla \log v(x) = \sum_{i=1}^{N} \mathbf{1}_{n_i^\top(x - \mu_i)\,\geq\, 0}\; \nabla \log\!\left(1 - G_i(x)\right)$$

$$\mathcal{N}(x) := \frac{V(x)}{\|V(x)\|}$$

이 때 $\mathcal{N}(x)$는 표면의 근방에서 잘 정의되며 기대 확률론적 표면의 실제 법선 필드와 일치한다.

#### ④ Normal Alignment Loss (Consistency Loss)

재건이 올바르게 작동하려면 Gaussian이 물체를 감싸는 밀봉된 연속 shell을 형성해야 한다. 이를 두 가지 상호 보완적인 메커니즘으로 촉진한다: **Normal alignment loss**. Splatting rasterizer로 깊이와 법선을 렌더링하면 표면 깊이와 표면 근처 Gaussian의 예상 방향을 추정하고, 정렬을 강제한다.

렌더링된 법선 $\hat{n}$과 학습된 Gaussian 법선 $n_i$의 정렬을 강제하는 loss:

$$\mathcal{L}_{align} = \sum_{p} \left(1 - \langle \hat{n}(p),\, \mathcal{N}(p) \rangle \right)$$

전체 학습 손실은 다음과 같이 구성된다:

$$\mathcal{L}_{total} = \mathcal{L}_{render} + \lambda_1 \mathcal{L}_{align} + \lambda_2 \mathcal{L}_{dist}$$

- $\mathcal{L}_{render}$: 기존 3DGS 렌더링 손실 (photometric)
- $\mathcal{L}_{align}$: 법선 정렬 일관성 손실
- $\mathcal{L}_{dist}$: Gaussian 분포의 depth distortion 손실

---

### 🏗️ 2.3 모델 구조 (Pipeline)

포즈가 주어진 이미지들로부터 시작하여, Gaussian당 학습 가능한 방향 법선이 추가된 Gaussian Splatting 모델을 학습한다. 법선 정렬 손실과 법선 인식 densification이 wrapping shell을 강제한다.

그 후 학습된 Gaussian으로부터 폐쇄형 vacancy $v$와 normal field $\mathcal{N}$을 도출하고, Pivot-Based Marching Tetrahedra 또는 새로운 Primal Adaptive Meshing을 통해 watertight 메시를 추출한다.

전체 파이프라인을 요약하면:

```
멀티뷰 RGB 이미지 (COLMAP 포즈)
        ↓
[1단계] Gaussian Splatting 학습
  - 각 Gaussian에 법선 n_i ∈ S² 추가
  - Normal alignment loss + Depth distortion loss
  - Normal-aware densification (법선 인식 densification)
        ↓
[2단계] 기하학 필드 도출
  - Oriented Attenuation 계산
  - Vacancy Field v(x) 도출 (closed-form)
  - Gaussian Normal Field N(x) 도출 (closed-form)
        ↓
[3단계] 메시 추출
  - Pivot-Based Marching Tetrahedra (PbMT)
  - Primal Adaptive Meshing (PAM) [신규 제안]
        ↓
고품질 Watertight + Textured Mesh
```

PAM은 staircase artifacts 없이 기반 Gaussian isosurface를 충실히 반영하는 위상적으로 깔끔한 표면을 생성한다.

PAM은 Gaussian당 2개의 pivot만 사용하여(9개 대비) 품질을 유지하면서 더 적은 정점을 생성한다.

---

### 📊 2.4 성능 향상

전체적으로 Gaussian Wrapping 방법은 DTU와 Tanks and Temples에서 새로운 최고 성능(state-of-the-art)을 달성하며, 동시대 연구 대비 훨씬 작은 메시 크기로 완전하고 watertight한 메시를 생성한다.

메시는 배경 기하학과 자전거 바퀴살 같은 극도로 얇은 구조물을 포함한 전체 장면을 동시대 연구들보다 **현저히 더 compact한 표현**으로 나타낸다.

watertight하고 최신 방법들이 재건한 메시보다 가벼우면서도, stump 장면의 몇 개 풀잎 같은 **극도로 미세한 디테일**을 재건할 수 있다.

**벤치마크 비교 요약:**

| 방법 | DTU F-Score | T&T F-Score | 메시 품질 | 얇은 구조물 |
|---|---|---|---|---|
| SuGaR (2024) | 중간 | 중간 | Poisson 기반 노이즈 | ❌ |
| 2DGS (2024) | 중간~높음 | 중간 | TSDF 기반 | △ |
| GOF (2024) | 높음 | 높음 | levelset 기반 | △ |
| **Gaussian Wrapping (2026)** | **최고** | **최고** | **Watertight + Compact** | ✅ |

---

### ⚠️ 2.5 한계

논문에서 직접 기술된 한계 및 구조적으로 도출되는 한계를 정리하면:

1. **정적 장면(Static Scene) 가정**: 현재 파이프라인은 COLMAP 포즈가 주어진 정적 장면을 가정하며, 동적 장면으로의 확장은 미비하다.
2. **wrapping 가정의 의존성**: Gaussian이 장면을 적절히 감쌀 때 oriented attenuation이 Gaussian Splatting 이미지 형성 모델의 유효한 근사가 된다. 즉, wrapping이 불완전하면 이론적 보장이 약화될 수 있다.
3. **멀티뷰 일관성 요구**: depth의 멀티뷰 일관성이 잘 정의된 occupancy field를 위한 필수 조건이다. 관측이 부족한 영역에서는 성능이 저하될 수 있다.
4. **학습 데이터 요구**: 기본적으로 멀티뷰 RGB 이미지와 COLMAP 포즈가 필요하여, 단일 이미지나 스파스 뷰 시나리오에는 적합하지 않다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 🌐 3.1 일반화를 가능하게 하는 구조적 특성

#### (a) 추가 학습 파라미터 최소화로 과적합 방지
Gaussian이 장면을 제대로 감쌀 때, oriented attenuation에 적용된 Objects as Volumes 프레임워크는 Gaussian 파라미터로부터 직접 closed-form 기하학적 양을 도출한다—**추가적인 학습 가능한 파라미터 없이**.

이는 네트워크 파라미터 과잉 없이도 풍부한 기하학 표현이 가능함을 의미하며, 새로운 장면에서도 과적합 없이 일반화되기에 유리하다.

#### (b) Closed-form 표현의 원칙적(principled) 수학적 기반
3DGS와 implicit 표면 재건 사이의 이론적 연결을 Gaussian을 방향성 표면 요소로 formulate함으로써 유도하며, 이는 closed-form normal 및 occupancy 필드 표현식으로 이어진다.

이 수학적 엄밀성은 특정 장면 유형에 맞춘 경험적 휴리스틱과 달리, 다양한 장면 범주에 걸쳐 일반적으로 적용 가능한 성질을 제공한다.

#### (c) 임의 해상도 메시 추출
Primal Adaptive Meshing은 도출된 Gaussian 필드를 활용해 **제어 가능한 해상도**에서 고품질의 watertight 메시를 생성하여, 자전거 바퀴살 같은 극도로 얇은 구조물의 복원을 가능하게 한다.

해상도 조절 가능성은 다양한 스케일의 장면과 하드웨어 조건에서도 유연하게 적용할 수 있음을 시사한다.

#### (d) Consistency Loss에 의한 구조적 규제
각 Gaussian에 학습 가능한 방향 법선을 부여하고 표면 근처의 가시 Gaussian이 점유 영역에서 멀어지며 카메라를 향해 바깥쪽을 가리키도록 wrapping 동작을 강제한다. 이 wrapping은 자전거 바퀴살처럼 극도로 얇은 구조물 주위에서도 방향성 primitive의 연속적인 shell을 생성한다.

이 구조적 규제는 특정 학습 데이터의 편향에 덜 의존하게 만들어 일반화에 기여한다.

### 🚀 3.2 일반화 성능 향상을 위한 잠재적 방향

1. **대규모 사전 학습(Pre-training)으로의 확장**: Oriented Gaussian 표현은 대규모 3D 데이터셋으로 사전 학습된 foundation model과 결합할 경우, few-shot 또는 zero-shot 표면 재건으로 일반화 가능성이 있다.

2. **동적 장면으로의 확장**: 포즈가 주어진 이미지들로부터 학습하는 구조를 시간축으로 확장하면, 동적 장면에서도 일반화된 표면 재건이 가능할 수 있다.

3. **스파스 뷰 환경 적용**: Normal-aware densification 전략은 본질적으로 표면 커버리지를 유도하는 방향이므로, 뷰 수가 적은 환경에서도 일관성 있는 결과를 낼 수 있는 잠재력이 있다.

4. **평가 프로토콜의 일반화**: 표준 표면 평가 프로토콜의 근본적 편향을 드러내고 더 엄격한 대안을 제안하며, DTU와 Tanks and Temples에서 새로운 state-of-the-art를 달성한다. 이처럼 평가 방법론 자체를 개선함으로써, 다양한 벤치마크에서 일반화 성능을 더 공정하게 측정할 수 있는 기틀이 마련된다.

---

## 4. 최신 관련 연구 비교 분석 (2020년 이후)

### 📚 4.1 관련 연구 계보

| 논문 | 연도 | 주요 방법 | 한계 |
|---|---|---|---|
| **NeRF** (Mildenhall et al.) | 2020 | MLP 기반 volume rendering | 느린 학습/렌더링, 표면 추출 어려움 |
| **NeuS** (Wang et al.) | 2021 | SDF + volume rendering | 느린 최적화, real-time 불가 |
| **3DGS** (Kerbl et al.) | 2023 | 3D Gaussian 명시적 표현 | 표면 추출 heuristic 의존 |
| **SuGaR** (Guédon & Lepetit) | 2024 | 3DGS + Poisson 재건 | 노이즈, 얇은 구조물 실패 |
| **2DGS** (Huang et al.) | 2024 | 2D Gaussian surfel | TSDF 의존, 얇은 구조물 부분적 복원 |
| **GOF** (Yu et al.) | 2024 | Ray-tracing 기반 opacity field | 배경 기하학 노이즈 |
| **Gaussian Wrapping** (이 논문) | 2026 | Oriented Gaussian + closed-form field | 정적 장면 한정 |

SuGaR는 Poisson 재건을 사용하고, 2DGS는 3D Gaussian을 평면 surfel로 붕괴시킨다.

GOF는 3D Gaussian의 ray-tracing 기반 volume rendering에서 유도되어 Poisson 재건이나 TSDF 융합 없이 level set을 식별함으로써 직접 기하학을 추출하고, Gaussian의 표면 법선을 ray-Gaussian 교차 평면의 법선으로 근사한다.

2DGS는 표면에 정렬된 타원형 Gaussian을 도입하여 ray-ellipse 교차 formulation을 통해 메시 재건 정확도를 향상시킨다.

Geometry Field Splatting은 OaV를 활용하여 파티클로부터 기하학 필드를 정의하지만, Gaussian surfel에만 분석을 적용하고 메시 추출에 TSDF를 사용하여 세부 복원 및 배경 기하학으로의 확장성에 한계가 있다.

동시대 연구인 Geometry-Grounded Gaussian Splatting은 이 논문과 유사한 설정을 채택하여 연속적인 transmittance를 도출하지만, depth의 멀티뷰 일관성을 보장하지 않아 잘 정의된 occupancy field를 위한 필수 조건을 만족하지 못한다.

2DGS-R 연구에 따르면 법선 일관성을 적용하면 DTU에서 적용하지 않을 때보다 재건 정확도가 46% 향상된다.

---

## 5. 향후 연구에 미치는 영향 및 고려할 점

### 🔭 5.1 앞으로의 연구에 미치는 영향

#### (a) 이론적 기여: 3DGS와 Implicit 표현의 가교
이 연구는 Gaussian을 방향성 표면 요소로 formulate하여 3DGS와 implicit 표면 재건 사이의 이론적 연결을 유도한다. 이는 명시적(explicit) 표현과 암시적(implicit) 표현의 이분법을 넘어서는 새로운 통합 방향을 제시한다.

#### (b) Gaussian 기반 방법론의 패러다임 전환
Gaussian을 대칭적 구름이 아닌 oriented probabilistic surface elements로 보는 시각은 앞으로의 Gaussian 기반 연구가 표면 재건, 물리 시뮬레이션, SLAM 등 다양한 하위 과제에서 primitive의 기하학적 의미를 더욱 엄밀하게 설계하도록 촉발할 것이다.

#### (c) 평가 방법론 개선
표준 표면 평가 프로토콜의 근본적인 편향 노출과 더 엄격한 대안 제시는 이 분야의 후속 연구가 보다 공정하고 의미 있는 비교를 수행하도록 유도할 것이다.

#### (d) 얇은 구조물 처리의 새로운 기준
표준 방법들이 외형만 몇 개의 비방향 primitive로 복원하고 기하학은 복원하지 못하는 자전거 바퀴살 같은 극도로 얇은 구조물에서도 방향성 primitive의 연속적 shell을 생성한다. 이는 자율주행, 의료 영상, 문화재 복원 등 얇고 복잡한 구조물이 중요한 응용 분야에서 새로운 표준이 될 잠재력이 있다.

---

### 🧭 5.2 앞으로 연구 시 고려할 점

#### ① 동적 장면 및 비정적 환경으로의 확장
현재 방법은 정적 장면과 COLMAP 포즈 입력을 가정한다. 동적 장면에서의 Gaussian wrapping 전략이나 포즈 추정과의 joint optimization이 중요한 연구 방향이 될 것이다.

#### ② 스파스 뷰 / 단안(Monocular) 설정에서의 일반화
재건이 작동하려면 Gaussian이 물체 주위에 밀봉된 연속적 shell을 형성해야 한다. 뷰 수가 적을 때 이 조건이 충족될 수 있도록, depth prior나 normal prior 등의 외부 신호를 활용하는 연구가 필요하다.

#### ③ wrapping 가정의 이론적 완화
Oriented attenuation이 OaV 적용을 위해 필요한 기하학적 성질을 만족하며 특정 가정 하에서 Gaussian Splatting 이미지 형성 모델의 유효한 근사임을 이론적으로 정당화하고 있다. 이 가정이 완화될 수 있는 더 일반적인 이론 프레임워크 개발이 중요한 연구 과제이다.

#### ④ Foundation Model 및 대규모 데이터와의 결합
Oriented Gaussian의 closed-form 필드 특성은 대규모 사전 학습된 모델과 결합하여 zero-shot 또는 generalizable surface reconstruction으로 나아가는 데 활용될 수 있다.

#### ⑤ 반사·투명 재질 및 특수 환경 처리
2DGS가 3DGS보다 더 나은 기하학적 일관성으로 미세한 표면을 포착하는 반면, 3DGS는 높은 시각적 품질을 제공하지만 surface artifacts를 유발한다. 반사나 투명 물체처럼 표면 법선을 신뢰성 있게 추정하기 어려운 재질에 대한 robustness 연구가 필요하다.

#### ⑥ 평가 프로토콜 표준화
표준 평가 프로토콜의 근본적 편향을 노출한 이 논문의 통찰을 바탕으로, 커뮤니티 차원의 공정하고 일관된 새로운 평가 기준 수립이 시급한 연구 과제이다.

---

## 📚 참고 자료 및 출처

| # | 제목 | 출처 |
|---|---|---|
| 1 | **From Blobs to Spokes: High-Fidelity Surface Reconstruction via Oriented Gaussians** | [arXiv:2604.07337](https://arxiv.org/abs/2604.07337) |
| 2 | 논문 공식 웹페이지 | [diego1401.github.io/BlobsToSpokesWebsite](https://diego1401.github.io/BlobsToSpokesWebsite/) |
| 3 | 공식 구현 코드 | [github.com/diego1401/GaussianWrapping](https://github.com/diego1401/GaussianWrapping) |
| 4 | **Gaussian Opacity Fields (GOF)** — Yu et al., 2024 | [arXiv:2404.10772](https://arxiv.org/abs/2404.10772) |
| 5 | **2DGS-R: Revisiting the Normal Consistency Regularization in 2D Gaussian Splatting** | [arXiv:2510.16837](https://arxiv.org/html/2510.16837) |
| 6 | **DiGS: Accurate and Complete Surface Reconstruction from 3D Gaussians via Direct SDF Learning** | [arXiv:2509.07493](https://arxiv.org/abs/2509.07493) |
| 7 | **3DGSR: Implicit Surface Reconstruction with 3D Gaussian Splatting** | ResearchGate / arXiv |
| 8 | **A survey on surface reconstruction based on 3D Gaussian splatting** | [PeerJ Computer Science](https://peerj.com/articles/cs-3034/) |
| 9 | MrNeRF X (Twitter) 요약 | [x.com/janusch_patas](https://x.com/janusch_patas/status/2042092432968372578) |
| 10 | 공식 발표 영상 | [YouTube](https://www.youtube.com/watch?v=zOwNjSboTE8) |

> ⚠️ **정확도 주의사항**: 본 답변은 공개된 arXiv 논문(2604.07337), 공식 웹페이지, GitHub 저장소 및 관련 문헌을 기반으로 작성되었습니다. 수식 중 일부(특히 학습 손실 $\mathcal{L}_{total}$의 구체적인 계수 $\lambda$)는 논문 원문의 appendix에 상세히 기술되어 있으나 검색 결과에서 직접 확인되지 않아, 구조적으로 명확한 부분만 제시했습니다. 논문 전문 PDF에서 최종 확인을 권장합니다.
