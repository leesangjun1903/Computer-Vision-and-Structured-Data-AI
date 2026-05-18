
# IRIS: Inverse Rendering of Indoor Scenes from Low Dynamic Range Images

> **논문 정보**
> - **제목:** IRIS: Inverse Rendering of Indoor Scenes from Low Dynamic Range Images
> - **저자:** Chih-Hao Lin, Jia-Bin Huang, Zhengqin Li, Zhao Dong, Christian Richardt, Tuotuo Li, Michael Zollhöfer, Johannes Kopf, Shenlong Wang, Changil Kim
> - **소속:** Meta, University of Illinois Urbana-Champaign, University of Maryland College Park
> - **발표:** CVPR 2025 (pp. 465–474)
> - **arXiv:** [2401.12977](https://arxiv.org/abs/2401.12977)
> - **프로젝트 페이지:** https://irisldr.github.io/
> - **GitHub (Meta Research):** https://github.com/facebookresearch/iris

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

Inverse rendering은 촬영된 이미지로부터 3D 형상, 표면 재질, 조명을 복원하는 기술로 novel-view synthesis, relighting, 가상 객체 삽입 등의 응용을 가능하게 하지만, 기존 대부분의 기법들은 **HDR(High Dynamic Range) 이미지를 입력으로 요구**하여 일반 사용자의 접근성을 제한한다.

이에 IRIS는 **다시점 LDR(Low Dynamic Range) 이미지로부터 물리 기반 재질, 공간적으로 변화하는 HDR 조명, 그리고 카메라 응답 함수(CRF)를 복원하는 inverse rendering 프레임워크**로, HDR 입력 의존성을 제거함으로써 접근성을 크게 향상시킨다.

### 주요 기여

IRIS는 LDR 이미지로부터 공간적으로 변화하는 HDR 조명, 물리 기반 재질, CRF를 충실히 추정하며 여러 최신 방법들을 능가한다. 특히 파이프라인에서 **LDR 이미지 형성 과정을 명시적으로 모델링**하여 LDR 이미지를 직접 사용할 수 있도록 접근성을 넓혔다.

또한 IRIS는 합성 및 실세계 장면에서 포괄적으로 평가되어, 다양하고 현실적인 view synthesis, relighting, 객체 삽입을 시연한다.

---

## 2. 해결하고자 하는 문제 / 제안 방법 / 모델 구조 / 성능 및 한계

### 2.1 해결하고자 하는 문제

**문제 1: LDR 이미지에서의 조명 정보 손실**

일반적인 이미징 센서는 장면의 발광 영역과 어두운 영역을 충분한 다이내믹 레인지로 포착하지 못한다. 더욱이 카메라는 저장·전송을 위해 원시 센서 데이터를 8비트 LDR 이미지로 변환하는데, 이 비선형 매핑과 양자화로 인해 추가적인 조명 정보 손실이 발생한다.

실세계 장면은 복사 휘도(radiance)에서 극심한 분산을 보여, LDR 이미지만으로는 노출 부족이나 포화(saturation) 없이 포착하기 어렵다. 클리핑과 비선형 CRF 매핑으로 중요한 조명 정보가 소실되어 기존 역 렌더링 방법들에게 심각한 과제를 제기한다.

**문제 2: 기존 방법들의 한계**

FIPT는 HDR 입력을 전제하고, NeILF는 다중 반사(multi-bounce) 광 전송을 무시한다. 두 방법 모두 정확한 재질 및 HDR 조명 추정에 실패한다.

여러 최신 NeRF 기반 접근법들은 재질을 신경망 필드로 파라미터화하고 volume rendering으로 최적화하지만, 원거리 조명을 가정하여 실내 환경의 공간적으로 변화하는 조명을 잘 처리하지 못한다.

NeILF와 NeILF++는 공간적으로 변화하는 조명을 신경 필드로 표현하고, I²-SDF는 emitter 마스크로부터 조명을 구성하지만, 이들은 일반적으로 **단일 반사(single-bounce) 광 전송**만 고려하여 재질 및 조명 복원 품질이 저하된다.

---

### 2.2 제안 방법 및 수식

#### (A) LDR 이미지 형성 모델

장면 조사량(irradiance) $E$와 노출 시간 $\Delta t$가 주어지면, 센서의 한계로 인해 관측 복사 휘도는 다음과 같이 특정 최댓값에서 클리핑된다:

$$Z_c = \min(E \Delta t, 1)$$

클리핑된 강도는 LDR 픽셀 값을 생성하기 위해 비선형 CRF로 추가 변환된다:

$$Z = \text{CRF}(Z_c) = \text{CRF}(\min(E \Delta t, 1))$$

#### (B) 렌더링 방정식

IRIS는 물리 기반 광 전송 모델링을 위해 렌더링 방정식을 사용한다:

$$L_o(\mathbf{x}, \boldsymbol{\omega}_o) = L_e(\mathbf{x}, \boldsymbol{\omega}_o) + \int_{\Omega^+} L_i(\mathbf{x}, \boldsymbol{\omega}_i) f(\mathbf{x}, \boldsymbol{\omega}_i, \boldsymbol{\omega}_o) \, d\boldsymbol{\omega}_i$$

여기서 $L_o$는 3D 위치 $\mathbf{x}$에서 방향 $\boldsymbol{\omega}_o$로 관측되는 복사 휘도이며, $L_e$는 방출 항(emission term), $f$는 BRDF이다.

#### (C) CRF 파라미터화 (EMoR)

IRIS는 CRF 모델링을 위해 Grossberg et al.의 **EMoR(Empirical Model of Response)**을 사용한다. 이 모델은 201개 실세계 CRF 데이터베이스로부터 평균 곡선 $\bar{\mathbf{g}}$와 PCA 기저 $\mathbf{g}_b$를 계산하여 CRF를 저차원으로 파라미터화한다:

$$\mathbf{g} = \bar{\mathbf{g}} + \sum_b w_b \mathbf{g}_b, \quad \mathbf{g} \in \mathbb{R}^{1024}$$

이를 통해 **CRF를 소수의 가중치 벡터 $\{w_b\}$로 추정**하여 최적화의 자유도를 크게 줄인다.

#### (D) 공간적으로 변화하는 조명 표현

환경 맵은 객체 중심 및 실외 장면을 잘 처리하지만, 실내의 복잡한 공간적으로 변화하는 조명을 포착할 수 없다. 이를 해결하기 위해 IRIS는 장면 메시(mesh) 위에서 직접 조명을 정의하고, 발광을 시점-독립 복사 휘도(view-independent radiance)로 측정한다:

$$L_e(\mathbf{x}) \in \mathbb{R}^3$$

#### (E) Cook–Torrance BRDF

IRIS는 공간적으로 변화하는 재질을 **Cook–Torrance BRDF**로 표현한다. 표면 알베도 $a(\mathbf{x}) \in \mathbb{R}^3$, 거칠기 $\sigma(\mathbf{x}) \in \mathbb{R}$, 금속성 $m(\mathbf{x}) \in \mathbb{R}$이 확산 반사율 $k_d = a \cdot (1 - m)$과 정반사율 $k_s = 0.04 \cdot (1 - m)$을 모델링하며, 이들은 신경 필드 $f : \mathbf{x} \mapsto (a, \sigma, m)$으로 파라미터화된다.

---

### 2.3 모델 구조 (2단계 파이프라인)

다시점 포즈 LDR 이미지와 표면 메시가 주어지면, IRIS의 inverse rendering 파이프라인은 두 주요 단계로 나뉜다. **초기화 단계**에서는 BRDF를 초기화하고, 표면 광 필드(Surface Light Field)를 추출하며, emitter 형상을 추정한다. **최적화 단계**에서는 먼저 LDR 입력으로부터 HDR 복사 휘도를 복원하고, 셰이딩 맵(shading maps)을 베이킹한 뒤, BRDF와 CRF 파라미터를 공동 최적화한다.

이 세 단계(HDR 복원 → 셰이딩 맵 베이킹 → BRDF/CRF 공동 최적화)는 수렴할 때까지 반복된다.

주요 구성 요소:
- 표면 알베도는 **IRISFormer**를 사용하여 추정하며, 더 나은 성능을 위해 RGB-X로 대체할 수 있다.
- 표면 법선은 기하 복원을 위해 **OmniData**로 추정되며, 기하는 SDFStudio의 **BakedSDF**로 재구성된다.
- 인수분해된(factorized) 광 전송 공식은 BRDF 파라미터와 셰이딩으로 분리하여 이들을 교대로 업데이트한다.

**최적화 핵심 문제 해결:**

조명, 알베도, CRF를 동시에 추정하면 모호성(ambiguities) 때문에 불안정한 최적화가 발생한다. IRIS는 이 모호성을 극복하여 세 가지 모두를 고품질로 추정하는 최적화 전략을 설계한다.

---

### 2.4 성능 향상

IRIS는 LDR 또는 HDR 입력을 받는 최신 inverse rendering 방법들과 비교 평가되었으며, LDR 이미지를 입력으로 사용하는 기존 방법들을 능가하고 매우 현실적인 relighting과 객체 삽입을 지원한다.

비교 베이스라인:
- 공정한 비교를 위해 일정 노출(constant exposure) 설정을 채택하였으며, IRIS는 **다양한 노출 수준의 LDR 이미지를 처리하고 CRF를 공동 추정**하는 능력도 추가로 시연한다.

실험 장면:
- IRIS는 FIPT의 합성 장면(Kitchen, Living Room, Bedroom, Bathroom)에서 다수 베이스라인과 비교 평가되며, 이 장면들에는 재질, 기하, 조명의 Ground Truth가 제공된다.

---

### 2.5 한계

Inverse rendering은 본질적인 모호성(inherent ambiguity)으로 인해 극도로 어렵고 비정칙(ill-posed) 문제이다.

데이터 기반(data-driven) 방법들의 경우, 높은 일반화 능력 달성을 위해서는 **고품질 데이터셋이 필수적**이다.

추가로 확인된 한계:
- FIPT 등 관련 방법들이 현실 장면에서 견고성이 제한된 것처럼, **실세계(real-world) 장면에서의 강건성**은 여전히 개선 여지가 있다.
- 평가에 사용된 LDR 이미지는 과노출 영역에서 클리핑되고, 비선형 CRF로 압축되며, 256단계로 양자화되어 8비트 PNG 형식으로 저장된 것으로, **심각한 정보 손실**을 내포한다.
- 기하 재구성(BakedSDF), 알베도 초기화(IRISFormer) 등 여러 외부 모듈에 의존하므로, 각 모듈의 오차가 최종 결과에 **누적 전파**될 수 있다.

---

## 3. 일반화 성능 향상 가능성

### 3.1 현재 일반화 관련 설계

현재 수많은 방법들은 intrinsic image decomposition, SVBRDF 추정, 조명 추정, relighting 등 다양한 태스크를 위해 **대규모 데이터셋에서 학습된 딥 프라이어**를 활용한다.

IRIS는 데이터 기반 IRISFormer 추정을 활용하여 좋은 알베도 초기화를 제공하고, 최종 결과는 물리 기반 렌더링 모델로 정제(refine)하는 방식을 취한다.

표면 알베도 추정에 IRISFormer를 사용하며, 이를 RGB-X로 교체하면 더 나은 성능을 얻을 수 있다. 표면 법선 추정에는 OmniData가 사용되며, 더 최신 연구로 교체할 시 성능이 향상될 수 있다.

이는 **모듈식 아키텍처**가 최신 구성 요소로의 업그레이드를 통해 일반화 성능을 점진적으로 향상시킬 수 있음을 의미한다.

### 3.2 일반화 성능 향상을 위한 핵심 고려 사항

| 요소 | 현재 방식 | 향상 가능성 |
|------|-----------|------------|
| 기하 추정 | BakedSDF (SDFStudio) | 최신 Gaussian Splatting 기반 기법으로 대체 |
| 알베도 초기화 | IRISFormer | RGB-X 또는 더 강력한 단일 이미지 추정기 |
| CRF 모델 | EMoR (PCA 기반, 저차원) | 더 많은 실세계 카메라 CRF 포함 데이터베이스 확장 |
| 다중 반사 모델 | 팩토라이즈드 경로 추적 | 완전한 미분 가능 경로 추적으로 확장 |
| 데이터 다양성 | FIPT/ScanNet++ 합성·실세계 장면 | 더 다양한 실내 유형 및 조명 조건 포함 |

IRIS는 일정 노출과 다양한 노출 설정 모두에서 LDR 이미지를 직접 소비할 수 있도록 파이프라인에서 LDR 이미지 형성을 명시적으로 모델링하여, 다양한 카메라와 촬영 조건으로의 일반화를 지원한다.

그러나 데이터 기반 방법들이 설득력 있는 일반화 능력을 달성하기 위해서는 **고품질 데이터셋이 필수적**이며, 이는 일반화 성능의 근본적인 병목으로 작용한다.

### 3.3 최신 연구와의 비교 (2020년 이후)

| 논문 | 발표 | 입력 | 주요 특징 | IRIS 대비 |
|------|------|------|-----------|-----------|
| **NeILF** (Yao et al.) | ECCV 2022 | HDR, 다시점 | 신경 incident light field | Single-bounce만 고려, LDR 미지원 |
| **I²-SDF** (Zhu et al.) | CVPR 2023 | HDR, emitter 마스크 | Neural SDF + emitter 마스크 | 추가 마스크 입력 필요 |
| **FIPT** (Wu et al.) | ICCV 2023 | HDR | 팩토라이즈드 inverse path tracing | HDR만 지원, LDR 미지원 |
| **SIR** | 2024 | HDR | 분해 가능한 그림자 역 렌더링 | HDR 전제, 그림자 처리 특화 |
| **PBR-NeRF** | arXiv 2024 | HDR | 에너지 보존 물리 손실 | HDR 전제, 물리 프라이어 강화 |
| **IRIS (Ours)** | CVPR 2025 | **LDR** | CRF 공동 추정, spatially-varying HDR 복원 | **LDR 입력 유일 고품질 방법** |

기존의 FIPT와 RawNeRF는 알려진 감마 보정 함수를 가정하고, NeILF++는 단일 감마 보정 파라미터를 학습하며, HDR-NeRF는 MLP로 톤 매핑 함수를 파라미터화한다. 그러나 이 방법들은 HDR 이미지 또는 다중 노출 LDR 이미지를 입력으로 사용하도록 설계되어 있다.

---

## 4. 미래 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

#### (1) 접근성 혁신
IRIS는 HDR 입력 의존성을 제거함으로써 **일반 소비자용 카메라로 촬영한 LDR 이미지만으로도 고품질 inverse rendering이 가능**하게 하여, 기술의 대중화와 실용화에 큰 발판을 마련했다.

#### (2) LDR 역 렌더링 패러다임 정립
LDR 이미지로부터 HDR 조명과 CRF를 공동 복원하는 새로운 inverse rendering 프레임워크를 제안함으로써, 이 문제를 정식화(formulation)하고 향후 연구의 기준점(benchmark)을 제시한다.

#### (3) 모듈형 파이프라인의 확장 가능성
알베도 추정(IRISFormer → RGB-X), 표면 법선 추정(OmniData → 최신 연구)과 같이 각 모듈이 독립적으로 교체 가능하여, 개별 구성 요소의 발전이 전체 시스템 성능 향상으로 자연스럽게 이어지는 연구 방향을 제시한다.

#### (4) AR/VR 및 실용 응용 가속화
LDR 이미지로부터 공간적으로 변화하는 HDR 조명, 물리 기반 재질, CRF를 추정함으로써 AR/VR에서의 객체 삽입, 조명 편집, relighting 등 고급 응용의 실용화를 가속화한다.

---

### 4.2 앞으로 연구 시 고려할 점

#### ① 데이터셋 확장 및 다양성
데이터 기반 방법들이 설득력 있는 일반화 능력을 달성하려면 **고품질 데이터셋이 필수적**이다. 향후 연구에서는 더 다양한 실내 유형(사무실, 산업 공간, 야외-실내 혼합 환경)과 조명 조건을 포괄하는 대규모 벤치마크 데이터셋 구축이 필요하다.

#### ② 재질-조명-CRF 삼중 모호성 해소
재질, 조명, CRF를 동시에 추정하면 모호성으로 인한 불안정한 최적화가 발생하며, IRIS는 이를 해결하는 새로운 최적화 전략을 설계했다. 그러나 극단적인 조명 조건(야간, 창문을 통한 강한 직사광 등)에서의 삼중 모호성 문제는 여전히 추가 연구가 필요하다.

#### ③ 동적 장면 및 비정적 조명 처리
현재 IRIS는 정적 조명 조건을 가정한다. 조명이 시간적으로 변화하는 동적 실내 장면(TV 화면, 자연광 변화 등)으로의 확장이 필요하다.

#### ④ 3D Gaussian Splatting 기반 기하 표현 통합
현재 기하 복원에 BakedSDF(SDFStudio)를 사용하지만, 최근 급속히 발전하는 3D Gaussian Splatting 기반 표현과의 통합을 통해 렌더링 속도와 재질 분해 품질을 동시에 향상시킬 수 있다.

#### ⑤ 단일 이미지 입력으로의 확장
현재 IRIS는 다시점(multi-view) 이미지를 필요로 한다. 학습 기반 방법들은 단일 또는 소수의 이미지를 입력으로 받아 기존 측정 기반 방법의 촬영 요구사항을 줄이는 방향으로 발전하고 있다. 이러한 흐름에 맞춰 단일 LDR 이미지로부터의 inverse rendering으로 확장하는 연구가 유망하다.

#### ⑥ 실시간 처리 및 경량화
현재 파이프라인은 SLF 베이킹, 에미터 추출, BRDF/CRF 최적화 등 다중 단계를 포함하여 계산 비용이 높다. AR/VR 실시간 응용을 위해 경량화·가속화 연구가 중요하다.

#### ⑦ 물리 기반 손실 강화
현재 inverse rendering 방법들은 표면의 정반사(specular) 특성과 확산(diffuse) 특성을 구별하는 재질 프라이어가 부족하다. 이로 인해 정반사 하이라이트가 확산 알베도의 변화로 잘못 귀인되는 경우가 있으며, 재질-조명 모호성 해소를 위한 재질 프라이어가 중요하다.

---

## 📚 참고 자료 및 출처

| # | 제목 | 출처 |
|---|------|------|
| 1 | **IRIS: Inverse Rendering of Indoor Scenes from Low Dynamic Range Images** (주 논문) | arXiv:2401.12977 / CVPR 2025, pp.465-474 |
| 2 | IRIS 프로젝트 페이지 | https://irisldr.github.io/ |
| 3 | IRIS GitHub (facebookresearch) | https://github.com/facebookresearch/iris |
| 4 | IRIS CVPR 2025 Open Access PDF | https://openaccess.thecvf.com/content/CVPR2025/papers/Lin_IRIS_... |
| 5 | IRIS IEEE Xplore | https://ieeexplore.ieee.org/document/11093722/ |
| 6 | IRIS OpenReview | https://openreview.net/forum?id=RVMgexbcrh |
| 7 | CVPR 2025 Poster Page | https://cvpr.thecvf.com/virtual/2025/poster/35190 |
| 8 | **FIPT**: "Factorized Inverse Path Tracing for Efficient and Accurate Material-Lighting Estimation" (Wu et al., ICCV 2023) | https://jerrypiglet.github.io/fipt-ucsd/ |
| 9 | **NeILF**: "Neural Incident Light Field for Physically-Based Material Estimation" (Yao et al., ECCV 2022) | arXiv 비교 인용 |
| 10 | **I²-SDF**: "Intrinsic Indoor Scene Reconstruction and Editing via Raytracing in Neural SDFs" (Zhu et al., CVPR 2023) | CVPR 2023 |
| 11 | **SIR**: "Multi-view Inverse Rendering with Decomposable Shadow for Indoor Scenes" (2024) | arXiv:2402.06136 |
| 12 | **PBR-NeRF**: "Inverse Rendering with Physics-Based Neural Fields" (2024) | arXiv:2412.09680 |
