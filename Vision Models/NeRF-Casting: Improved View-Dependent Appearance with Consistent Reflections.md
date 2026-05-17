
# NeRF-Casting: Improved View-Dependent Appearance with Consistent Reflections 

> **논문 정보**
> - **저자:** Dor Verbin, Pratul P. Srinivasan, Peter Hedman, Ben Mildenhall, Benjamin Attal, Richard Szeliski, Jonathan T. Barron
> - **발표:** SIGGRAPH Asia 2024 (Tokyo, Japan, December 2024)
> - **arXiv:** [2405.14871](https://arxiv.org/abs/2405.14871) (2024년 5월 23일)
> - **DOI:** https://doi.org/10.1145/3680528.3687585

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

NeRF는 시점(viewpoint) 변화에 따라 외관이 빠르게 달라지는 **고반사(highly specular) 객체**의 재구성 및 렌더링에 근본적인 어려움을 겪는다.

기존 연구들은 멀리 있는 환경 조명의 반사(distant environment illumination)를 렌더링하는 데 개선을 이뤘지만, **가까운 콘텐츠(nearby content)의 일관된 반사(consistent reflections)**를 합성하지 못한다.

이러한 기법들은 나가는 방사휘도(outgoing radiance)를 모델링하기 위해 **대형의 계산 비용이 높은 신경망**에 의존하므로, 최적화 및 렌더링 속도를 심각하게 제한한다.

**NeRF-Casting의 핵심 주장:** 레이 트레이싱(ray tracing)을 NeRF의 렌더링 모델에 통합함으로써 위의 두 가지 문제를 동시에 해결할 수 있다.

### 1.2 주요 기여

| 기여 | 설명 |
|------|------|
| **Ray Casting 기반 반사 렌더링** | 값비싼 MLP 대신 반사 광선(reflection rays)을 추적하여 feature vector를 렌더링 |
| **반사 안티앨리어싱(Anti-aliasing)** | 다수의 반사 광선(5개)과 feature downweighting 기법으로 앨리어싱 방지 |
| **Near-field 반사 합성** | 가까운 물체의 반사를 정확하게 합성하는 최초의 NeRF 방법 |
| **효율성** | 소규모 MLP를 사용하여 SOTA 대비 유사한 최적화 시간 유지 |

이 모델은 반짝이는 물체가 포함된 장면의 뷰 합성에서 기존 방법들을 능가하며, **실제 세계 장면에서 사실적인 반사를 합성할 수 있는 유일한 NeRF 방법**임을 시연하였다.

---

## 2. 해결 문제 · 제안 방법(수식 포함) · 모델 구조 · 성능 및 한계

### 2.1 해결하고자 하는 문제

전통적인 NeRF는 3D 좌표와 시점 방향을 체적 밀도(density)와 색상(color)으로 매핑하는 MLP를 사용하는데, 이는 고해상도 장면과 반사와 같은 복잡한 시점 의존적 효과를 모델링할 때 **계산적으로 매우 비싸다**. 반사를 개선하려는 기존 방법들은 원거리 반사에 한정되거나 대형 MLP를 요구하여 **느리고 확장하기 어렵다**.

구체적으로 두 가지 핵심 문제:

1. **근거리 반사(Near-field reflection)** 불가 — 가까운 물체의 반사를 일관성 있게 합성 불가
2. **계산 비용** — 대형 MLP에 의한 최적화·렌더링 속도 저하

### 2.2 제안 방법

#### (a) 전체 파이프라인

본 방법은 **레이 트레이싱을 NeRF의 렌더링 모델에 도입**함으로써 문제를 해결한다. 카메라 광선을 따라 각 점에서 시점 의존적 외관을 위해 비싼 MLP를 쿼리하는 대신, 해당 점들로부터 **반사 광선(reflection rays)을 NeRF 기하 구조 내로 투사하고**, 적절히 안티앨리어싱된 특징(features)을 반사된 장면 콘텐츠에서 샘플링한 뒤, **소규모 MLP**를 사용하여 이 특징들을 반사 색상으로 디코딩한다.

**단계별 과정:**

**① 카메라 광선 샘플링**

카메라 광선을 장면에 투사하고, 광선을 따라 점들을 샘플링한다. Zip-NeRF의 해시 그리드(hash grid)에 기반한 **공간 인코더**를 사용하여 각 점을 체적 밀도($\tau$), 거칠기($\rho$), 표면 법선($\mathbf{n}$)으로 인코딩한다. 샘플링된 밀도들을 기반으로 **알파 합성(alpha compositing)**을 통해 카메라 광선의 기대 종료점(expected termination point) $\bar{\mathbf{x}}$와 표면 법선 $\bar{\mathbf{n}}$을 계산한다.

**② 반사 광선 방향 계산**

원래 광선 방향 $\mathbf{d}$를 종료점 $\bar{\mathbf{x}}$에서의 표면 법선 $\bar{\mathbf{n}}$에 대해 반사하여 반사 광선 방향 $\mathbf{d}'$를 계산한다. 불완전한 반사와 Zip-NeRF가 광선 대신 콘(cone)을 추적한다는 사실을 반영하기 위해, **반사 광선들의 콘 형태 분포(cone-like distribution)**를 모델링한다.

수식으로 표현하면, 반사 방향:

$$\mathbf{d}' = \mathbf{d} - 2(\mathbf{d} \cdot \bar{\mathbf{n}})\bar{\mathbf{n}}$$

거칠기(roughness) $\rho$를 고려한 **von Mises-Fisher (vMF) 분포**에서 $K$개의 반사 광선 방향 $\{\mathbf{d}'\_j\}_{j=1}^K$를 샘플링:

$$p(\mathbf{d}'_j) \propto \exp\left(\kappa(\rho) \cdot \mathbf{d}' \cdot \mathbf{d}'_j\right)$$

**③ 특징 벡터 평균화**

K개의 반사 광선으로부터 얻은 특징 벡터들을 평균하여 반사 콘을 나타내는 단일 특징 벡터 $\bar{f}$를 얻는다:
$$\bar{f} = \frac{1}{K} \sum_{j=1}^{K} \bar{f}(\mathbf{d}'_j)$$

**④ 반사 특징 다운웨이팅 (Reflection Feature Downweighting)**

반사된 표면의 거칠기가 높을 경우 대표 샘플 광선들이 멀리 떨어져 있어 특징 $\bar{f}$가 앨리어싱될 수 있다. 이를 위해 **Zip-NeRF에서 영감을 받은 feature downweighting 기법**을 도입하여, vMF 콘에 비해 작은 복셀(voxel)의 특징에 작은 승수(multiplier)를 곱한다. 이는 앨리어싱을 줄이는 데 도움이 된다.

다운웨이팅 수식:

$$f_{aa}(\mathbf{x}) = \text{erf}\!\left(\frac{1}{\sqrt{8}\,\nu\,\sigma(\mathbf{x})}\right) \odot f(\mathbf{x})$$

여기서:

$$\sigma(\mathbf{x}) = \gamma \cdot (r + \bar{\rho}) \cdot \|\mathbf{x} - \mathbf{o}'\| \cdot \det\!\left(J_{\text{dir}}^C(\mathbf{x})\right)$$

$$\det\!\left(J_{\text{dir}}^C(\mathbf{x})\right) = \frac{2\max(1,\|\mathbf{x}\|) - 1}{\max(1,\|\mathbf{x}\|)^2}$$

$\gamma$는 상수, $r$은 광선 반경(ray radius), $\bar{\rho}$는 평균 거칠기, $\nu$는 해시 그리드의 복셀 크기(voxel size).

이 기법은 Zip-NeRF에서 영감을 받은 것으로, vMF 콘에 비해 상대적으로 작은 복셀의 특징을 작은 승수로 곱하여 **앨리어싱을 감소**시킨다.

**⑤ 최종 색상 디코딩**

소규모 MLP $g_\theta$로 최종 색상 출력:

$$c = g_\theta\left(\bar{f},\, \mathbf{d}',\, \bar{\rho}\right)$$

### 2.3 모델 구조

공간 인코더는 **Zip-NeRF의 해시 그리드(hash grid)**에 기반하여 각 점을 체적 밀도($\tau$), 거칠기($\rho$), 표면 법선($\mathbf{n}$)으로 인코딩한다.

| 구성 요소 | 설명 |
|-----------|------|
| **Spatial Encoder** | Zip-NeRF 기반 해시 그리드 → $\tau, \rho, \mathbf{n}$ 출력 |
| **Reflection Ray Sampler** | vMF 분포에서 $K=5$개 반사 광선 샘플링 |
| **Feature Renderer** | 반사 광선 따라 NeRF 특징 볼륨 렌더링 |
| **Anti-aliasing Module** | 2D directional Jacobian 기반 downweighting |
| **Color Decoder (Small MLP)** | 평균 feature vector → 최종 색상 디코딩 |

전체적으로 본 방법은 각 카메라 광선을 따른 점에서 시점 의존적 외관을 위해 **값비싼 MLP를 쿼리하는 대신**, 해당 점들로부터 반사 광선을 NeRF 기하 구조 내로 투사하고, 반사된 장면 콘텐츠에서 **적절히 안티앨리어싱된 특징을 샘플링**하며, **소규모 MLP**를 사용하여 특징을 반사 색상으로 디코딩한다.

### 2.4 어블레이션 연구 결과 (Ablation Study)

단일 반사 광선만 사용하거나, 특징 다운웨이팅을 적용하지 않거나, 본 방법의 2D directional Jacobian 대신 Zip-NeRF의 3D Jacobian을 사용하는 경우 모두 **부정확한 반사**를 초래한다. 광택도가 낮은 물체라도 최적화 중 반사의 앨리어싱은 모델이 반사 표면 기하 구조와 반사된 콘텐츠를 정확하게 재구성하지 못하게 한다.

근거리 장면 콘텐츠의 추적(tracing)을 생략하면 구형, 조각상 등 근거리 반사의 렌더링이 **현저히 저하**된다.

### 2.5 성능 향상

본 방법은 세부적인 경면(specular) 반사를 보여주는 반짝이는 영역에서 **정량적으로** 기존 뷰 합성 기법들을 능가한다. 특히 본 방법으로 합성된 **반사의 부드럽고 일관된 움직임**은 베이스라인 방법들이 렌더링한 시점 의존적 외관보다 현저히 더 사실적이다.

본 방법은 관찰된 이미지에서 **직접 관찰되지 않은 나무나 가로등**의 반사까지도 렌더링할 수 있다.

### 2.6 한계점

본 방법은 특정 효과를 완전히 모델링하지 못한다. 각 카메라 광선의 **기대 종료점(expected termination point)**에서만 반사하기 때문에 **반투명 표면(semi-transparent surfaces)** 렌더링에 어려움을 겪는다. 또한, 카메라 조작자(operator)가 반사 표면에 자주 보이지만, 모델은 이 잠재적 오류 원인을 고려하지 않는다.

반투명이지만 반사성을 갖는 표면(예: 창문)은 본 방법에 어려운 도전 과제가 된다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 Near-field 반사로의 일반화

반사 광선을 NeRF 내로 투사하면 **가까운 콘텐츠와 먼 콘텐츠 모두의 일관된 반사**를 자연스럽게 합성한다. 또한, 레이 트레이싱으로 외관을 계산함으로써 **장면의 각 점에서 대형 MLP로 시점 의존적 함수를 표현해야 하는 부담**을 줄인다.

### 3.2 장면 유형 일반화

확산(diffuse) 위주의 장면에서는 본 방법의 렌더링이 Zip-NeRF와 동등하며, 다른 베이스라인보다 눈에 띄게 우수하다.

이는 고반사 장면에만 특화되지 않고, **다양한 장면 유형**에서도 안정적인 성능을 유지함을 의미한다.

### 3.3 Zip-NeRF 프레임워크와의 호환성

본 방법은 Zip-NeRF 렌더링 모델에 레이 트레이싱을 도입하는 방식으로 **기존 NeRF 파이프라인과 모듈식으로 통합**할 수 있어 일반화 가능성이 높다.

### 3.4 직접 관측 불가 영역으로의 일반화

본 방법은 심지어 **직접 촬영된 이미지에서 한 번도 정면으로 관측되지 않은** 나무나 가로등의 반사까지도 일관성 있게 렌더링할 수 있다.

이는 NeRF가 학습한 암묵적 표현(implicit representation)을 반사 광선이 효과적으로 탐색하기 때문으로, **관측되지 않은 뷰(unseen view)로의 일반화** 능력을 시사한다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 논문 | 연도 | 핵심 방법 | 장점 | 한계 |
|------|------|-----------|------|------|
| **NeRF** (Mildenhall et al.) | ECCV 2020 | MLP + 볼륨 렌더링 | 범용적, 간단 | 고반사 불가, 느림 |
| **Ref-NeRF** (Verbin et al.) | CVPR 2022 | 반사 방향으로 외관 재파라미터화 | 경면 반사 개선 | 근거리 반사 불가 |
| **Mip-NeRF 360** (Barron et al.) | CVPR 2022 | 무한 장면 안티앨리어싱 | 언바운디드 장면 | 반사 미지원 |
| **Zip-NeRF** (Barron et al.) | ICCV 2023 | 해시 그리드 + 콘 트레이싱 | 빠르고 고품질 | 반사 미지원 |
| **SpecNeRF** (Ma et al.) | 2023 | 가우시안 방향 인코딩 | 경면 반사 개선 | 근거리 반사 제한 |
| **UniSDF** (Wang et al.) | 2023 | SDF + 신경 반사 | 기하 재구성 우수 | 느림 |
| **NeRF-Casting** (Verbin et al.) | **SIGGRAPH Asia 2024** | **NeRF 내 반사 광선 추적** | 근거리·원거리 반사 모두 합성, 효율적 | 반투명 표면 한계 |
| **Planar Reflection-Aware NeRF** | 2024 | 평면 반사체 명시 모델링 | 창문 등 평면 특화 | 일반 곡면 반사 제한 |

Ref-NeRF는 시점 방향을 지역 법선에 대해 반사한 함수로 나가는 방사휘도를 명시적으로 파라미터화하여 경면 반사 표현에 유망한 결과를 보이지만, **반사를 분리하는 능력이 부족하다**.

MS-NeRF는 다수의 특징 필드를 소규모 MLP로 렌더링 및 블렌딩하여 장면을 표현하지만, 이 방법들은 자기지도 방식으로 장면을 독립적인 구성요소로 분해하여 때때로 **반사로 인한 잘못된 기하 구조(false geometry)가 지속되는 최적화 미달(suboptimal decomposition)**이 발생한다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 연구에 미치는 영향

1. **NeRF + 물리 기반 렌더링(PBR)의 통합 패러다임 제시**
   본 방법은 장면 표면에서 콘을 반사하고 NeRF를 통해 추적하며, 이 반사를 안티앨리어싱하는 새로운 기법들을 통해 **원거리 및 근거리 콘텐츠 모두의 정확하고 세부적인 반사를 일관되고 부드럽게 합성**한다는 새로운 방향을 열었다.

2. **3D Gaussian Splatting(3DGS)으로의 확장 가능성**
   이 연구에 영향을 받아 SpecTRe-GS (2025 CVPR)와 같이 **3D Gaussian Splatting에서도 반사 광선을 추적**하여 고반사 표면을 모델링하는 연구가 이미 등장하고 있다.

3. **역렌더링(Inverse Rendering) 연구 촉진**
   반사 광선을 NeRF 내부에서 추적하는 아이디어는 재질·조명 분리, 재조명(relighting) 연구에도 강력한 기반을 제공한다.

### 5.2 앞으로 연구 시 고려할 점

#### ① 반투명 표면 처리
본 방법은 각 카메라 광선의 **기대 종료점에서만 반사**하기 때문에, 반투명 표면(창문 등)을 정확히 모델링하지 못한다는 한계가 남아 있다. 향후 연구에서는 투과(transmission)와 반사를 동시에 처리하는 방식이 필요하다.

#### ② 복수 반사(Multiple Bounces) 처리
현재 방법은 1차 반사(single bounce)만 처리한다. 상호반사(interreflection)를 완전히 모델링하기 위해서는 다중 바운스(multiple bounces)를 효율적으로 처리하는 방안이 필요하다.

#### ③ 카메라 자신의 반사(Camera Operator Occlusion)
카메라 조작자(operator)가 반사 표면에 자주 보이지만, 현재 모델은 이 **잠재적 오류 원인**을 고려하지 않는다. 이를 명시적으로 모델링하는 연구가 필요하다.

#### ④ 일반화 성능과 장면 독립성
현재 NeRF-Casting은 여전히 **장면별(per-scene) 최적화** 방식이다. 여러 장면에 걸쳐 일반화되는 feed-forward 방식의 반사 렌더링 모델로의 확장이 중요한 연구 방향이다.

#### ⑤ 동적 장면(Dynamic Scene)으로의 확장
반사 광선 추적은 정적 장면(static scene)을 가정하므로, 움직이는 물체가 있는 동적 장면에서의 반사 합성은 별도의 연구가 필요하다.

#### ⑥ 실시간 렌더링 가능성
본 방법은 뷰 의존적 외관을 개선하기 위한 기존 방법들보다 **효율적이며 특히 근거리 콘텐츠에서 더 높은 품질의 반사를 렌더링**한다. 그러나 $K=5$개의 반사 광선 샘플링은 여전히 실시간 응용에 비용이 크므로, 가속 자료구조(예: 해시 그리드, 옥트리)와의 결합을 통한 실시간화가 중요하다.

---

## 📚 참고 문헌 및 출처

| # | 자료 | 링크 |
|---|------|------|
| 1 | **NeRF-Casting (arXiv 원문)** | https://arxiv.org/abs/2405.14871 |
| 2 | **NeRF-Casting (SIGGRAPH Asia 2024, ACM DL)** | https://dl.acm.org/doi/10.1145/3680528.3687585 |
| 3 | **NeRF-Casting (ACM Full HTML)** | https://dl.acm.org/doi/fullHtml/10.1145/3680528.3687585 |
| 4 | **NeRF-Casting (공식 프로젝트 페이지)** | https://dorverbin.github.io/nerf-casting/ |
| 5 | **NeRF-Casting (Moonlight Literature Review)** | https://www.themoonlight.io/en/review/nerf-casting-improved-view-dependent-appearance-with-consistent-reflections |
| 6 | **Planar Reflection-Aware NeRF (arXiv 2411.04984)** | https://arxiv.org/abs/2411.04984 |
| 7 | **SpecTRe-GS: 3DGS 반사 추적 (CVPR 2025)** | ACM DL (doi:10.1109/CVPR52734.2025.01504) |
| 8 | **Ref-NeRF (Verbin et al., CVPR 2022)** | CVPR 2022 |
| 9 | **Zip-NeRF (Barron et al., ICCV 2023)** | ICCV 2023 |
| 10 | **SpecNeRF (Ma et al., arXiv 2312.13102)** | https://arxiv.org/abs/2312.13102 |
| 11 | **UniSDF (Wang et al., arXiv 2312.13285)** | https://arxiv.org/abs/2312.13285 |
| 12 | **Mip-NeRF 360 (Barron et al., CVPR 2022)** | CVPR 2022 |

> ⚠️ **정확도 안내:** 수식 중 일부 세부 내용(특히 downweighting 수식의 파라미터)은 공개된 리뷰 사이트 및 공식 프로젝트 페이지를 기반으로 정리되었으며, 전체 수식의 완전한 검증을 위해서는 [공식 논문 PDF](https://nerf-casting.github.io/static/paper_anon.pdf)를 직접 참조하시길 권장합니다.
