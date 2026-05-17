
# Radiance Surfaces: Optimizing Surface Representations with a 5D Radiance Field Loss

> **논문 정보:**
> - **저자:** Ziyi Zhang, Nicolas Roussel, Thomas Müller, Tizian Zeltner, Merlin Nimier-David, Fabrice Rousselle, Wenzel Jakob
> - **소속:** EPFL (École Polytechnique Fédérale de Lausanne) & NVIDIA
> - **게재:** SIGGRAPH 2025 (Conference Papers, Vancouver, BC, Canada)
> - **arXiv:** [2501.18627](https://arxiv.org/abs/2501.18627)
> - **DOI:** [10.1145/3721238.3730713](https://doi.org/10.1145/3721238.3730713)

---

## 1. 핵심 주장과 주요 기여 요약

### 🔑 핵심 주장

이 논문은 이미지를 radiance surface 기반 씬 표현으로 변환하는 빠르고 단순한 기법을 제안한다. 기존 방사 체적(radiance volume) 재구성 알고리즘을 기반으로, 손실 함수를 미묘하지만 효과적으로 수정하여—**레이를 따라 방사장을 적분하고 결과 이미지를 감독하는 대신**—훈련 이미지를 씬에 직접 투영하여 시공간 방향(spatio-directional) 방사장을 감독하는 방식을 도입한다.

이 논문은 미분 가능한 렌더링 분야에서 비교적 새로운 패러다임인 **"many worlds" 패러다임**—즉 비상호작용(non-interacting) 프리미티브의 분포를 최적화하는 방법—을 radiance surface 재구성에 적용한다. 특히 주목할 점은 유도(derivation)의 출발점이 진화하는 표면이었음에도 불구하고, 결과적으로 체적 씬 재구성과 놀랍도록 유사한 방정식—레이를 따라 색상이 아닌 손실(loss)이 적분되는 형태—에 도달했다는 것이다.

### 🏆 주요 기여 요약

| 기여 항목 | 설명 |
|---|---|
| **5D Radiance Field Loss** | 이미지 공간 손실 대신 spatio-directional 방사장 직접 감독 |
| **Alpha Blending/Ray Marching 제거** | 이미지 형성 모델에서 제거 → 손실 연산으로 이동 |
| **Radiance Surface 의미론적 정의** | 방사장의 2D 부분집합에 명시적 의미 부여 |
| **레벨셋 추출** | 고품질 표면 모델 직접 추출 |
| **단순성** | 기존 코드 몇 줄 수정만으로 구현 가능 |

---

## 2. 해결하고자 하는 문제, 제안 방법(수식), 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

기존 체적(volumetric) 재구성은 블렌딩된 색상의 이미지-공간 손실을 최소화하는 방식을 사용한다. 이와 달리, 제안하는 방법은 블렌딩된 방사장 손실(blended radiance field loss)을 최소화함으로써 표면 분포를 도출하며, 이로부터 marching cubes 등을 통해 표면 표현을 자연스럽게 추출할 수 있다.

기존 체적 접근법은 손실 계산 전에 레이를 따라 색상을 적분하며, 이 때문에 레이를 따라 모든 점이 통합 색상이 너무 어둡거나 밝을 경우 동일한 부호의 기울기(gradient)를 받아 **상관된 조정(correlated adjustments)**이 발생하는 문제가 있다.

기하학에 대한 렌더링의 미분화는 작은 기하학적 섭동이 결과 이미지에 미치는 영향을 드러낸다. 그러나 이러한 미분값은 표면 자체에서만 0이 아니기 때문에, 최적화에 사용할 때 수렴 문제를 일으킨다. 이를 극복하기 위해 기존에 보이는 표면 위 일정 거리에 작은 표면 패치를 도입하는 방법을 고려하며, 이 수정은 렌더링된 이미지에도 영향을 미치며 더 일반적인 비국소적(non-local) 미분의 섭동으로 해석할 수 있다.

---

### 2.2 제안하는 방법과 수식

#### 🔷 기존 NeRF의 체적 렌더링 (Volume Rendering)

기존 NeRF는 5D 입력 $(x, y, z, \theta, \phi)$를 받아 색상 $\mathbf{c}$와 밀도 $\sigma$를 출력하고, 레이를 따라 색상을 적분하여 최종 픽셀 색상을 계산한다:

$$
\hat{C}(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\,\sigma(\mathbf{r}(t))\,\mathbf{c}(\mathbf{r}(t), \mathbf{d})\,dt
$$

$$
T(t) = \exp\!\left(-\int_{t_n}^{t}\sigma(\mathbf{r}(s))\,ds\right)
$$

기존 방법의 이미지 공간 손실 (Image-space Loss):

$$
\mathcal{L}_{\text{NeRF}} = \sum_{\mathbf{r} \in \mathcal{R}} \left\| \hat{C}(\mathbf{r}) - C_{\text{gt}}(\mathbf{r}) \right\|^2
$$

여기서 $C_{\text{gt}}(\mathbf{r})$는 ground-truth 픽셀 색상이다. **이 방식의 문제점은 모든 포인트가 동일 방향의 gradient를 받아 표면으로의 수렴이 어렵다는 것이다.**

---

#### 🔷 제안하는 5D Radiance Field Loss (RFL)

제안하는 방법은 블렌딩된 방사장 손실(blended radiance field loss)을 최소화하며, 결과적으로 표면 분포를 도출한다. 이 방사장 손실은 레이를 따라 각 점을 표면 후보로 간주하고, 해당 레이의 픽셀 색상에 매칭되도록 개별적으로 최적화하여 원하는 표면 분포를 유도한다.

핵심 아이디어는 **훈련 이미지를 씬에 투영**하여 5D 방사장을 직접 감독하는 것이다. 각 점 $t_i$에서 ground-truth 색상 $C_{\text{gt}}$를 직접 할당하고, 개별 손실을 계산한다:

$$
\ell(\mathbf{r}, t) = \left\| \mathbf{c}(\mathbf{r}(t), \mathbf{d}) - C_{\text{gt}}(\mathbf{r}) \right\|^2
$$

최종 Radiance Field Loss는 이 개별 손실을 체적 가중치로 블렌딩하여 계산한다:

$$
\mathcal{L}_{\text{RFL}} = \sum_{\mathbf{r} \in \mathcal{R}} \int_{t_n}^{t_f} T(t)\,\sigma(\mathbf{r}(t))\,\ell(\mathbf{r}, t)\,dt
$$

이산화(discretization)하면:

$$
\mathcal{L}_{\text{RFL}} \approx \sum_{\mathbf{r}} \sum_{i} w_i \left\| \mathbf{c}(\mathbf{r}(t_i), \mathbf{d}) - C_{\text{gt}}(\mathbf{r}) \right\|^2
$$

여기서 가중치 $w_i$는 기존 NeRF 체적 렌더링의 가중치와 동일하다:

$$
w_i = T_i\,\alpha_i, \quad T_i = \prod_{j < i}(1-\alpha_j), \quad \alpha_i = 1 - \exp(-\sigma_i \delta_i)
$$

> ⚠️ **핵심 차이:** NeRF는 **색상을 먼저 적분한 후 손실 계산**, RFL은 **손실을 먼저 각 점에서 계산한 후 적분**한다.

이 방사장 손실은 레이를 따라 각 점을 표면 후보로 간주하고, 각각 독립적으로 최적화한다. 이 방식의 장점은 레이를 따라 각 점이 독립적인 기울기(gradient)를 받으므로, 한 점에서는 색상이나 밀도가 증가하고 다른 점에서는 동시에 감소할 수 있다는 것이다.

---

### 2.3 모델 구조

이 구현은 Instant-NGP를 기반으로 하며, 새로운 학습 모드와 설정 옵션을 도입한다. 표준 NeRF로 씬 학습을 시작한 후 RFL 모드로 실시간 전환하여 즉각적인 효과—예: samples/ray의 급격한 감소—를 관찰할 수 있다.

구체적인 모델 구조 구성요소는 다음과 같다:

| 구성 요소 | 설명 |
|---|---|
| **Backbone** | Instant NGP (Multiresolution Hash Encoding + tiny-cuda-nn MLP) |
| **입력** | 5D 좌표: 위치 $(x,y,z)$ + 방향 $(\theta, \phi)$ |
| **출력** | 색상 $\mathbf{c}$ + 밀도 $\sigma$ |
| **손실** | 5D Radiance Field Loss (기존 이미지-공간 손실 대체) |
| **표면 추출** | Level Set 추출 (Marching Cubes) |
| **추가 기여** | Stochastic Background Distribution |

또한, **확률적 배경 분포(stochastic background distribution)**를 도입한다. 이는 위상적 변화(topological changes)를 가능하게 하고 재구성 품질을 실질적으로 향상시킨다. 이 전략을 기댓값(expectation) 형태로 저렴하게 평가하는 방법을 제시하여 NeRF와 알고리즘적 동등성을 유지한다.

제안 방법에서 표면 렌더링은 서로 다른 레벨셋 임계값에 걸쳐 최소한의 변화를 보이며, 이는 **점유 필드(occupancy field)가 표면에서 near-Heaviside 계단 함수로 수렴**하여 표면 기반 표현의 추출이 가능함을 나타낸다.

---

### 2.4 성능 향상

이 방법은 기준 알고리즘의 속도와 품질 대부분을 유지한다. 예를 들어, Instant NGP를 적절히 수정한 변형은 비교 가능한 계산 효율성을 유지하면서 평균 PSNR이 단 0.1 dB 낮은 성능을 달성한다. 가장 중요하게는, 이 방법이 지수 체적(exponential volume) 대신 명시적 표면을 생성하며, 이는 기존 연구에서는 볼 수 없던 수준의 단순성을 갖춘다.

이 방법은 NeRF 스타일 방법의 속도와 견고성으로 표면을 재구성한다. 체적 기반 방법이 2D 이미지 손실을 최소화하는 것과 달리, 시공간 방향 방사장 손실 공식을 채택한다. 각 단계에서 이 방법은 광학적으로 독립적인 표면의 분포를 고려하여, 참조 이미지와 일치하는 후보들의 신뢰도를 높인다. 최적화 중 어느 반복(iteration)에서든 의미 있는 표면을 추출할 수 있다.

---

### 2.5 한계

Laplacian smoothing 전략은 뷰 의존적 외관(view-dependent appearance) 때문에 평평한 캔 표면 재구성에 실패한다. Laplacian 가중치를 높이면 도움이 되지만, 기하학적 디테일도 함께 억제된다. 또한, 고주파 색상 변화는 체적 표현에 비해 표면에서 정확하게 표현하기 더 어렵다.

재구성 작업이 어려워질수록, 공간의 어떤 영역이 표면으로 표현될지 아니면 체적으로 표현될지 결정하는 것이 핵심 과제가 된다. 완화(relaxed) 변형 방법이 효과적인 휴리스틱을 제공하지만, 이는 이 중요한 문제에 대한 원칙적인 답의 필요성을 강조한다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 향상을 가능하게 하는 핵심 특성

제안하는 방사장 손실은 체적 재구성 방법과 놀랍도록 유사한 방정식을 도출한다. 실용적으로 이는 기존 체적 프레임워크에 방법을 통합하기 쉽다는 것을 의미한다. 또한, 표면 추출을 위한 추가적인 휴리스틱에 의존하지 않고도 기존 연구의 많은 장점을 계승한다는 것을 의미한다.

즉, 기존 NeRF 계열 방법의 다양한 일반화 기법(정규화, 하이퍼파라미터 튜닝 등)이 RFL에도 직접 적용될 수 있는 **플러그인 호환성**을 가진다.

### 3.2 독립적 기울기(Independent Gradient)의 역할

레이를 따라 각 점이 독립적인 기울기(gradient)를 받으므로, 한 점에서 색상 또는 밀도가 증가할 때 다른 점에서 동시에 감소할 수 있다.

이는 **국소적 정보를 더욱 세밀하게 학습**할 수 있어, 특정 장면에 과적합되지 않고 다양한 씬에서 더 안정적인 수렴이 가능하다는 일반화 측면의 이점을 제공한다.

### 3.3 위상 변화와 Stochastic Background Distribution

확률적 배경 분포(stochastic background distribution)의 도입은 위상적 변화(topological changes)를 가능하게 하고 재구성 품질을 실질적으로 향상시킨다. 이를 기댓값 형태로 저렴하게 평가하는 방법을 제시하여 NeRF와 알고리즘적 동등성을 유지한다.

위상적 변화가 가능하다는 것은 **다양한 형태의 씬(열린 표면, 복잡한 위상구조 등)에서도 유연하게 일반화**될 수 있음을 시사한다.

### 3.4 최적화 중 어느 시점에서도 표면 추출 가능

각 단계에서 광학적으로 독립적인 표면의 분포를 고려하여 참조 이미지와 일치하는 후보들의 신뢰도를 높이며, 최적화 중 어느 반복(iteration)에서든 의미 있는 표면을 추출할 수 있다.

이는 빠른 수렴 특성과 결합되어, 데이터가 제한적인 상황(few-shot 씬 재구성 등)에서도 유용한 결과를 제공할 수 있다.

### 3.5 Future Engineering Effort에 대한 낙관적 전망

저자들은 NeRF 기반 재구성을 위한 최적화된 알고리즘, 정규화기(regularizers), 휴리스틱 설계에 투입된 많은 엔지니어링 노력의 상당 부분이 radiance field loss에도 그대로 전달되어 미래에 최신(state-of-the-art) 결과를 낼 수 있을 것이라는 기대를 표명하고 있다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 연구 | 방법 | 표면 추출 | 속도 | 특징 |
|---|---|---|---|---|
| **NeRF** (Mildenhall et al., 2020) | MLP + Volume Rendering | ❌ 직접 불가 | 느림 | 최초 NeRF, 기준 방법 |
| **Instant NGP** (Müller et al., 2022) | Hash Encoding + Volume | ❌ 직접 불가 | 매우 빠름 | 속도 혁신 |
| **NeuS** (Wang et al., 2021) | SDF + Volume Rendering | ✅ SDF 기반 | 중간 | 표면 재구성 특화 |
| **SuGaR** (Guédon & Lepetit, 2024) | 3D Gaussian Splatting | ✅ 메시 추출 | 빠름 | GS 기반 메시 |
| **2DGS** (Huang et al., 2024) | 2D Gaussian Splatting | ✅ 기하 정확도 향상 | 빠름 | 기하학적 정확도 우선 |
| **Radiance Surfaces (본 논문, 2025)** | RFL + Instant NGP | ✅ 레벨셋 | 매우 빠름 | 단순성 + 품질 균형 |

"many worlds" 패러다임—비상호작용 프리미티브의 분포를 최적화하는 것—은 미분 가능한 렌더링 분야에서 비교적 새로운 접근법이다. 이 논문은 이를 radiance surface 재구성에 적용하여 기존 연구들에 비해 빠르고 단순한 대안을 제시한다. 특히, 유도의 출발점이 진화하는 표면임에도 불구하고, 결과적으로 체적 씬 재구성과 놀랍도록 유사한 방정식—레이를 따라 색상이 아닌 손실이 적분되는 형태—에 도달했다는 점이 주목할 만하다.

표면 표현의 매력은 물리적 현실과의 자연스러운 정렬 외에도, 편집, 애니메이션, 효율적인 렌더링에 대한 적합성에 있으며, 이것이 3D 그래픽스 응용 프로그램에서 거의 보편적으로 사용되는 이유를 설명한다.

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5.1 향후 연구에 미치는 영향

#### ✅ 기존 NeRF 생태계와의 통합 용이성
제안하는 방사장 손실은 체적 재구성 방법과 놀랍도록 유사한 방정식을 도출하며, 실용적으로 이는 기존 체적 프레임워크에 간단하게 통합될 수 있다는 것을 의미한다.

→ 이는 **기존 NeRF 기반 연구(Instant NGP, 3DGS, NeuS 등)에 손쉽게 접목**되어 표면 재구성 성능을 향상시킬 수 있음을 시사한다.

#### ✅ "Many Worlds" 패러다임의 확산
"many worlds" 패러다임—비상호작용 프리미티브의 분포를 최적화하는 것—은 미분 가능한 렌더링 분야에서 비교적 새로운 접근이다. 이 논문은 이를 radiance surface 재구성에 적용하여 기존 연구의 빠르고 단순한 대안을 제시한다. 유도의 출발점이 진화하는 표면임에도 불구하고, 체적 씬 재구성과 놀랍도록 유사한 방정식에 도달했다는 점이 특히 주목할 만하다.

→ 이 개념은 **물리 기반 렌더링(PBR), 동적 씬 재구성, 의료 영상 등 다양한 도메인으로 확장** 가능성이 높다.

#### ✅ 손실 함수 설계 패러다임 전환
이 논문의 기여는 단순성을 활용하여 특화된 방법을 개발하는 데 있다. radiance surface에 고유한 최적화를 식별하고 구현하여 many-worlds 아이디어의 잠재력을 완전히 실현한다.

→ **이미지 공간이 아닌 5D 방사장 공간에서 직접 감독**하는 아이디어는 다양한 3D 재구성 방법에서의 손실 함수 설계에 새로운 방향을 제시할 것이다.

---

### 5.2 향후 연구 시 고려할 점

#### ⚠️ 표면 vs. 체적 결정 문제
재구성 작업이 어려워질수록, 공간의 어떤 영역이 표면으로 표현될지 아니면 체적으로 표현될지 결정하는 것이 핵심 과제가 된다. 완화(relaxed) 변형 방법이 효과적인 휴리스틱을 제공하지만, 이는 이 중요한 문제에 대한 원칙적인 답의 필요성을 강조한다.

→ **반투명 물체, 연기, 불꽃** 등 체적 특성이 강한 씬에서는 어떤 방식을 사용할지에 대한 이론적 근거가 필요하다.

#### ⚠️ 뷰 의존적 외관 처리
Laplacian smoothing 전략은 뷰 의존적 외관 때문에 평평한 캔 표면 재구성에 실패할 수 있다. 또한, 고주파 색상 변화는 체적 표현에 비해 표면에서 정확하게 표현하기 더 어렵다.

→ **반사, 굴절, 거울 표면** 등 강한 뷰 의존성을 가진 물체에서의 성능 향상이 필요하다.

#### ⚠️ PBR과의 통합 한계
radiance surface 렌더링에서 이미지 형성은 레이와 가장 가까운 교차 표면 사이의 직접적인 1:1 매핑인 반면, PBR-MW(물리 기반 렌더링 - many worlds)는 재료, 조명, 기하학에 대한 복잡한 중첩 적분이 필요하다. 훈련 이미지를 씬에 투영하여 radiance field loss를 설정하는 접근 방식은 이 1:1 매핑에 의존하므로, 전역 조명(global illumination) 렌더러의 중첩 적분 구조에는 효율적으로 적용되기 어렵다.

→ **전역 조명, PBR 재료 분해(material decomposition)** 연구로의 확장 시 추가적인 방법론 개발이 필요하다.

#### ⚠️ 정규화(Regularization) 연구
NeRF 기반 3D 재구성을 위한 최적화된 알고리즘, 정규화기(regularizers), 휴리스틱 설계에 많은 엔지니어링 노력이 투입되어 왔다.

→ RFL에 맞는 **특화된 정규화 기법(Laplacian Smoothing 개선, 기하학적 정규화 등)**의 연구가 필요하다.

---

## 📚 참고 자료 및 출처

| 번호 | 제목 / 출처 | 링크 |
|---|---|---|
| 1 | **논문 원문 (arXiv)**: Radiance Surfaces: Optimizing Surface Representations with a 5D Radiance Field Loss (arXiv:2501.18627) | https://arxiv.org/abs/2501.18627 |
| 2 | **논문 HTML 전문 (arXiv v2)** | https://arxiv.org/html/2501.18627v2 |
| 3 | **ACM Digital Library (SIGGRAPH 2025)**: DOI 10.1145/3721238.3730713 | https://dl.acm.org/doi/10.1145/3721238.3730713 |
| 4 | **EPFL RGL 연구 그룹 공식 페이지** | https://rgl.epfl.ch/publications/Zhang2025Radiance |
| 5 | **NVIDIA Research 공식 페이지** | https://research.nvidia.com/publication/2025-07_radiance-surfaces-optimizing-surface-representations-5d-radiance-field-loss |
| 6 | **GitHub 공식 코드 (INGP-RFL)**: ziyi-zhang/INGP-RFL | https://github.com/ziyi-zhang/INGP-RFL |
| 7 | **관련 기반 논문**: Instant Neural Graphics Primitives (Müller et al., SIGGRAPH 2022) | https://nvlabs.github.io/instant-ngp/ |
| 8 | **관련 논문**: Many-Worlds Inverse Rendering (Zhang et al., 2024, arXiv:2408.16005) | https://arxiv.org/abs/2408.16005 |
| 9 | **관련 논문**: SuGaR: Surface-aligned Gaussian Splatting (Guédon & Lepetit, CVPR 2024) | CVPR 2024 |
| 10 | **관련 논문**: 2D Gaussian Splatting for Geometrically Accurate Radiance Fields (Huang et al., SIGGRAPH 2024) | https://doi.org/10.1145/3641519.3657428 |

> ⚠️ **정확도 주의사항**: 본 답변의 수식 중 일부(특히 $\mathcal{L}_{\text{RFL}}$의 구체적 형태)는 논문의 공개된 HTML 전문 및 abstract를 기반으로 재구성한 것으로, 논문 본문의 최종 표기와 세부적으로 다를 수 있습니다. 정확한 수식 확인을 위해서는 [arXiv 원문](https://arxiv.org/abs/2501.18627) 또는 [ACM DL](https://dl.acm.org/doi/10.1145/3721238.3730713)을 직접 참조하시기 바랍니다.
