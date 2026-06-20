
# Residual Primitive Fitting of 3D Shapes with SuperFrusta

> **논문 정보:**
> - **저자:** Aditya Ganeshan\*, Matheus Gadelha†, Thibault Groueix†, Zhiqin Chen†, Siddhartha Chaudhuri†, Vladimir Kim†, Wang Yifan†, Daniel Ritchie\* (\*Brown University, †Adobe Research)
> - **발표:** CVPR 2026 (Oral & Award Candidate Paper, Top 1.8%)
> - **arXiv:** [arXiv:2512.09201](https://arxiv.org/abs/2512.09201)
> - **프로젝트 페이지:** [bardofcodes.github.io/superfit](https://bardofcodes.github.io/superfit/)
> - **코드:** [github.com/BardOfCodes/superfit](https://github.com/BardOfCodes/superfit)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문은 3D 형상을 간결하고 편집 가능한 해석적 기본체(analytic primitives) 어셈블리로 변환하는 프레임워크를 소개하며, **재구성 충실도(reconstruction fidelity)와 간결성(parsimony) 사이의 고질적인 상충 관계**를 직접적으로 해결한다. 핵심 기여는 **SuperFrustum**이라는 새로운 기본체와 **Residual Primitive Fitting(ResFit)** 알고리즘의 결합으로 이루어진다.

### 2대 핵심 기여

| 기여 | 설명 |
|---|---|
| **SuperFrustum** | 8개 파라미터로 다양한 형상을 표현하는 해석적 SDF 기본체 |
| **ResFit** | 전역 형상 분석과 지역 최적화를 교대로 수행하는 비지도 반복 알고리즘 |

### 성능 수치 요약

다양한 3D 벤치마크에서 본 방법은 **IoU를 9점 이상 개선**하면서 기존 연구 대비 **절반 수준의 기본체 수**만을 사용하여 최고 성능을 달성한다.

ResFit은 표준 3D 벤치마크에서 기존 최고 수준 방법들 대비 재구성 정확도(IoU)를 **6~9점 향상**시키며, 이전 방법보다 약 절반의 기본체만을 사용하면서 부피적 중첩(volumetric overlap)도 약 3배 감소시킨다.

---

## 2. 논문 상세 분석

### 2-1. 해결하고자 하는 문제

기존에 자주 사용되는 기본체 계열인 **큐보이드(cuboids), 슈퍼쿼드릭(superquadrics), 타원체(ellipsoids)**는 3D 자산의 풍부한 형상 변화를 표현하기 위해 많은 수의 인스턴스가 필요하다. 또한 추론 절차 자체에도 고유한 한계가 있는데, 입력의 완전한 분할(segmentation)에 먼저 의존하는 방법들은 기본체가 효율적으로 표현할 수 없는 고정된 파티션에 의존하게 된다. 이는 초기 분할 오류가 최종 어셈블리까지 전파되기 때문에 취약한 결과를 낳는다.

반면, 처음부터 많은 기본체들의 "수프(soup)"를 맞추는 최적화 주도 방법들은 **매우 비볼록한(non-convex) 손실 지형**을 탐색해야 한다.

쉽게 정리하면 다음과 같은 두 가지 딜레마가 존재한다:

- **표현력 부족:** 큐보이드·슈퍼쿼드릭은 원통, 원환(torus), 중공(hollow) 구조 등을 표현하기 어렵다
- **알고리즘적 취약성:** 한 번에 전체를 분할하는 방식 또는 대규모 동시 최적화 방식 모두 수렴 실패·파티션 불일치 문제를 야기한다

---

### 2-2. 제안하는 방법 (수식 포함)

#### ① SuperFrustum — 새로운 해석적 기본체

SuperFrustum은 동시에 세 가지 특성을 갖는 해석적 기본체이다: **(1) 표현력** — 원통, 구, 원뿔, 그리고 테이퍼(taper)·굴곡(bent) 형태를 포함한 다양한 일반 솔리드를 모델링 가능, **(2) 편집 가능성** — 단 **8개 파라미터**로 간결하게 파라미터화됨, **(3) 최적화 가능성** — 부호 거리 함수(signed distance field)가 파라미터에 대해 거의 모든 곳에서 미분 가능함.

SuperFrustum은 **팽창(dilation), 테이퍼(taper), 불지(bulge), 양파 형태 중공화(onion-like hollowing), 프로파일 원형도(profile roundness), 축방향 스케일링(axial scaling)**을 제어하는 오직 8개의 파라미터를 가진 통합 해석적 SDF 기본체이다. 그 SDF는 $C^0$-연속이고 거의 모든 곳에서 완전히 미분 가능하다.

SuperFrustum의 설계는 Shadertoy와 Demoscene 커뮤니티에서 발견된 해석적 함수들을 기반으로 하며, 이 정식들이 최소한의 기술 길이(description length)로 높은 표현력을 가진다는 점에서 역모델링(inverse modeling)에 매우 적합하다는 것을 확인하였다.

8개 파라미터를 $\boldsymbol{\theta} = (r_1, r_2, h, k, s, e, t, o)$ 로 표현하면:

| 파라미터 | 의미 |
|---|---|
| $r_1, r_2$ | 상하단 반지름 (taper) |
| $h$ | 높이 (axial scaling) |
| $k$ | 굴곡도 (bulge/bend) |
| $s$ | 단면 라운드니스 (profile roundness) |
| $e$ | 팽창/수축 (dilation) |
| $t$ | 테이퍼 비율 |
| $o$ | 중공화 두께 (onion hollowing) |

SuperFrustum의 SDF는 로컬 좌표계 $\mathbf{p}$에서 다음과 같이 쓸 수 있다:

$$
f_{\boldsymbol{\theta}}(\mathbf{p}) = \text{SDF}_{\text{SuperFrustum}}(\mathbf{p};\, r_1, r_2, h, k, s, e, t, o)
$$

그 연속적이며 조각별 $C^1$ 정식은 큐보이드, 원통, 원뿔, 원환 등 다양한 형상을 아우르며, 거의 모든 곳에서 미분 가능하여 안정적인 경사 기반 피팅(gradient-based fitting)을 가능하게 한다.

전체 어셈블리의 SDF는 **Soft Union** 연산으로 구성된다:

$$
F(\mathbf{p}) = \text{SoftUnion}\left(f_{\boldsymbol{\theta}_1}(\mathbf{p}),\, f_{\boldsymbol{\theta}_2}(\mathbf{p}),\, \ldots,\, f_{\boldsymbol{\theta}_N}(\mathbf{p})\right)
$$

---

#### ② ResFit — 잔차 기반 반복 피팅 알고리즘

ResFit은 전역 형상 분석과 지역 최적화를 교대로 수행하는 **비지도(unsupervised) 절차**로, 형상의 미설명 잔차(unexplained residual)에 반복적으로 기본체를 맞춰 나가며 각 입력 형상에 대한 간결하면서도 정확한 분해를 발견한다.

**전체 알고리즘 흐름 (ResFit):**

```
입력: 3D 형상 S (SDF 그리드 또는 메쉬)
출력: 기본체 어셈블리 {θ_1, ..., θ_N}

for round r = 1, ..., R_max:
  1. [분석] MSD로 현재 잔차 형상에서 가장 두꺼운 연결 영역 추출
  2. [초기화] 추출된 영역으로 새 SuperFrustum 초기화
  3. [최적화] 새 기본체를 로컬 지지체(local support) 기반으로 최적화
  4. [전역 재최적화] 전체 어셈블리를 재최적화
  5. [가지치기] 목적함수를 저하시키는 기본체 제거
  6. 잔차 업데이트 → 수렴 또는 R_max 도달 시 종료
```

각 단계에서 ResFit은 기존 기본체들로 아직 설명되지 않은 형상의 영역을 식별하고, 그 영역을 커버할 새 기본체를 제안하며, 전체 어셈블리를 정제한다.

**최적화 목적 함수:**

고수준 목적함수 $\mathcal{O}$는 **재구성 충실도 + 간결성 정규화**의 조합으로 구성된다:

$$
\mathcal{O}(\{\boldsymbol{\theta}_i\}) = \mathcal{L}_{\text{recon}}(\{\boldsymbol{\theta}_i\}, S) + \lambda \cdot \mathcal{R}_{\text{parsimony}}(\{\boldsymbol{\theta}_i\})
$$

- $\mathcal{L}_{\text{recon}}$: IoU 기반 또는 SDF 기반 재구성 손실
- $\mathcal{R}_{\text{parsimony}}$: 중복성을 페널티화하는 소프트 정규화 항 (간결성 인식 최적화)

과도한 파라미터화(over-parameterization)를 방지하기 위해 매 라운드에 소수의 기본체만을 시드(seed)하고 **간결성 인식 최적화(parsimony-aware optimization)**를 적용한다: 소프트 정규화기가 피팅 중 중복성을 페널티화하고, 하드 가지치기(hard pruning)가 목적함수를 저하시키는 파트를 제거한다. 과소 파라미터화(under-parameterization) 해결을 위해 기본체는 로컬 지지체 기반으로 최적화되고, 전체 어셈블리는 매 라운드 재최적화된다.

---

#### ③ MSD — Morphological Shape Decomposition

MSD는 **"가장 두꺼운 부분을 먼저 벗겨내는(peel the thickest part first)"** 반복적 기법으로, 각 단계에서 대략 균일한 두께의 가장 큰 연결 영역을 찾아 추출하고 해당 부분을 형상에서 제거한 뒤 잔차에서 이를 반복한다.

MSD는 자전거 타이어, 고양이의 구부러진 꼬리, 그릇 테두리와 같은 비볼록 구조를 포착하는 적절한 초기화 시드(seed)를 형성하는 반면, CoACD(근사 볼록 분해)는 이러한 영역을 의미론적으로 정렬되지 않은 많은 볼록 조각으로 과분할(over-partition)한다.

---

### 2-3. 모델 구조

전체 파이프라인의 구조를 도식화하면:

```
입력 3D 형상 (메쉬 또는 SDF 그리드, 128³ voxel)
          │
          ▼
┌─────────────────────────────────────┐
│   MSD (Morphological Shape          │
│   Decomposition)                    │
│   → 두께 기반 부위 분할 + 시드 생성  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   SuperFrustum 초기화               │
│   (8-파라미터 SDF 기본체)           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   ResFit 반복 루프 (최대 10 라운드) │
│   ┌─────────────────────────────┐   │
│   │ ① 잔차 분석 (MSD 7회 반복) │   │
│   │ ② 새 기본체 제안            │   │
│   │ ③ 로컬 최적화               │   │
│   │ ④ 전역 재최적화             │   │
│   │ ⑤ 간결성 가지치기           │   │
│   └─────────────────────────────┘   │
└──────────────┬──────────────────────┘
               │
               ▼
출력: 간결한 SuperFrustum 어셈블리
(편집 가능, WebGL 실시간 렌더링 지원)
```

ResFit은 수렴하거나 최대 10회의 피팅 라운드 동안 실행되며, 각 라운드에서 7회의 MSD 반복을 적용한다.

---

### 2-4. 성능 향상

제안 방법은 **재구성-간결성 파레토 경계(Pareto frontier)를 이동**시켜, 최고 수준 방법인 Marching Primitives(MPS)와 Primitive Anything(PA)에 비해 훨씬 적은 수의 기본체를 사용하면서 현저히 낮은 재구성 오류를 달성한다.

어블레이션 연구에서 SuperFrustum은 큐보이드(Cuboids), 슈퍼쿼드릭(Superquadrics, SQ), 슈퍼프리미티브(SuperPrimitive)보다 우수한 성능을 보인다.

ResFit 절차와 단일 샷 피팅 베이스라인을 비교한 결과, 단일 샷 방법은 모든 기본체를 초기 분해 이후 동시에 최적화하기 때문에 초기 파티션에 민감하고, 미설명 영역에 용량을 재배분하는 메커니즘이 없어 덜 정확한 결과를 낳는다. ResFit은 분석과 최적화를 교대로 수행함으로써 단일 샷 피팅 방법보다 일관되게 우월한 성능을 보인다.

**추가 응용 사례:**

- 프레임워크는 소프트 유니온 어셈블리를 위해 설계되었지만, 정준 솔리드(큐보이드, 원통, 원뿔, 구)로 구성된 이산적 **CSG(Constructive Solid Geometry) 프로그램**도 추론 가능하며, 전용 방법들과 거의 동등한 CSG 재구성 정확도를 더 적은 기본체로 달성한다.

- 본 방법은 3D 생성 모델과 매끄럽게 결합되어 이미지-투-기본체 어셈블리를 달성할 수 있으며, 최신 텍스트/이미지-투-3D 모델 이후에 ResFit을 적용하면 어떤 이미지든 편집 가능한 기본체 어셈블리로 변환할 수 있다.

- 텍스처 메쉬를 텍스처 기본체 어셈블리로 변환할 수 있고, 이 어셈블리를 단순한 WebGL 셰이더에서 직접 렌더링·상호작용할 수 있다. SuperFrusta의 간결한 해석적 형태는 브라우저에서 효율적으로 실행되는 실시간 구 추적(sphere tracing)과 대화형 편집 기능을 가능하게 한다.

---

### 2-5. 한계점

ResFit은 **순수 가산적 구성(purely additive composition)**에만 제한되어 있어, **빼기(subtractive) 연산이 필요한 형상은 여전히 어렵다**. 향후 연구로는 분해 전략 확장(예: 형상의 트리 구조), CSG 모델링, 대화형 편집, 구조적 장면 이해 등의 응용 개발이 포함된다.

또한 Toys4K 테스트 세트에서 10라운드 전체 버전의 ResFit은 형상당 평균 **652.6초**가 소요되어 실시간 처리가 어렵다는 실용적 한계도 존재한다. 다만 커스텀 CUDA 커널(VarAxisSF)을 적용하면 B200 벤치마크 기준 end-to-end 속도가 최대 8.5배 향상되고, forward-only 평가에서는 최대 38.8배까지 빠르다.

---

## 3. 일반화 성능 향상 가능성

### 3-1. 비지도 학습 기반의 본질적 일반화

ResFit은 전역 형상 분석과 지역 최적화를 교대로 수행하는 **비지도(unsupervised) 절차**로, 형상의 미설명 잔차에 반복적으로 기본체를 맞춰 나가며 각 입력 형상에 대한 간결하면서도 정확한 분해를 발견한다.

이는 지도 학습 기반 방법들이 겪는 **카테고리 특화 한계**를 본질적으로 극복한다. 학습 기반 방법들은 카테고리별 학습에 의존하는 제약이 있는데, 이는 전역 형상 특징만을 인코딩하는 모델 설계에 기인하며 카테고리 내 일반화에는 충분하지만 카테고리 외 객체 분해에는 효과적이지 않다.

### 3-2. SuperFrustum의 표현력을 통한 일반화

본 방법의 기본체 어셈블리는 중공 형태(vase), 곡선·원환형 부품(bike), 복잡한 기하(ladder, robot), 그리고 부드러운 유기적 형상(crab) 등 **광범위한 형상을 포착**한다. 이는 단일 기본체 계열이 다양한 카테고리의 형상에 일반화될 수 있음을 입증한다.

### 3-3. MSD의 일반화 기여

ResFit은 형상 분해 방법이 생성하는 체적 영역으로부터 기본체를 초기화하며, 선택된 분해 전략이 기본체 계열의 표현력과 일치할 때 성능이 향상된다. 최근 연구들이 기본체 초기화를 위해 근사 볼록 분해(ACD)를 적용하는 것과 달리, 본 연구는 MSD의 적응 변형이 SuperFrusta 초기화에 더 적합함을 발견한다.

### 3-4. 생성 모델과의 결합을 통한 일반화 확장

본 방법은 3D 생성 모델과 매끄럽게 결합될 수 있어, 텍스트/이미지-투-3D 모델과 결합함으로써 임의의 이미지를 편집 가능한 기본체 어셈블리로 변환하는 이미지-투-기본체 어셈블리가 가능해진다. 이는 훈련 데이터에 없던 새로운 객체나 장면에도 적용 가능한 **오픈-월드 일반화 능력**을 제공한다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 기본체 타입 | 방법 | 주요 특징 | 한계 |
|---|---|---|---|---|---|
| **Superquadrics Revisited** (Paschalidou et al.) | 2019 | 슈퍼쿼드릭 | 학습 기반 | 최초 슈퍼쿼드릭 신경망 | 카테고리 의존 |
| **D²CSG** | 2023 | CSG 트리 | 비지도 학습 | Dual Complement + Dropout | 감산 연산 전용 |
| **Marching Primitives (MPS)** | 2023 | 슈퍼쿼드릭 | 최적화 기반 | SDF 그리드 기반 직접 최적화 | 기본체 과다 사용 |
| **Primitive Anything (PA)** | 2024 | 큐보이드+원통+타원체 | 학습 기반 | 대규모 데이터 학습, TTO 변형 | 카테고리 의존, 표현 제한 |
| **SuperDec** | 2025 | 슈퍼쿼드릭 | 최적화 기반 | 3D 장면 분해 | 단일 기본체 계열 |
| **Light-SQ** | 2025 | 슈퍼쿼드릭 | 구조-인식 학습 | 생성 메쉬 대상 | 생성 데이터 전용 |
| **SuperFit (본 논문)** | 2025/2026 | **SuperFrustum** | 비지도 최적화 | 8-파라미터 SDF, 잔차 반복 피팅, MSD 초기화 | 가산 연산만 가능, 속도 |

기존 연구들에서 사용된 기본체 타입은 표현력이 제한적이어서, 이들 연구는 주로 입력 형상을 **대략적인 피팅으로 추상화**하는 데 초점을 맞추었다.

최근 수년간 슈퍼쿼드릭은 낮은 피팅 오류 달성 능력 덕분에 다시 주목받고 있다. 그러나 본 논문의 SuperFrustum은 슈퍼쿼드릭을 뛰어넘어 원환(torus), 테이퍼, 중공 형태까지 단일 기본체로 통합한다.

AI 생성 3D 형상들은 인상적이지만 의미 있는 구조가 없어 편집이 어렵다는 문제가 있는데, SuperFit은 복잡한 3D 형상을 단순하고 해석 가능한 빌딩 블록인 SuperFrusta의 간결한 어셈블리로 변환하는 프레임워크를 제시함으로써 이 문제를 정면 돌파한다.

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5-1. 향후 연구에 미치는 영향

**① 3D 형상 표현의 새로운 패러다임**

SuperFrustum(표현력 있고 간결하며 최적화 가능한 해석적 기본체)과 ResFit(형상 분석과 기본체 최적화를 결합한 반복 추론 알고리즘)의 결합은 재구성-간결성 파레토 경계를 이동시켜, 벤치마크 전반에서 최고 성능을 달성하면서 고충실도의 편집 가능한 형상 프로그램을 생성한다.

**② 3D 생성 AI와의 연결**

SuperFit은 복잡한 3D 형상을 단순하고 해석 가능한 빌딩 블록인 SuperFrusta의 간결한 어셈블리로 변환하는 프레임워크를 소개하여, **3D 데이터와 인간이 제어 가능한 디자인 사이의 간격을 좁힌다**.

**③ 실시간 렌더링 및 편집 파이프라인**

이 어셈블리들은 단순한 WebGL 셰이더에서 직접 렌더링·상호작용될 수 있으며, SuperFrusta의 간결한 해석적 형태는 브라우저에서 효율적으로 실행되는 **실시간 구 추적(sphere tracing)과 대화형 편집 기능**을 가능하게 한다.

**④ CSG 프로그램 합성 분야 기여**

향후 연구 방향으로는 분해 전략의 확장(예: tree-of-shapes), **CSG 모델링, 대화형 편집, 구조적 장면 이해** 등의 풍부한 응용 개발이 포함된다.

---

### 5-2. 향후 연구 시 고려할 점

#### ① 감산(Subtractive) 연산 지원
ResFit은 여전히 순수 가산적 구성에만 제한되어 있어 **감산 연산이 필요한 형상(예: 구멍, 오목 부위)은 여전히 어렵다**. 완전한 CSG 트리를 통합하거나, D²CSG 등 감산 전용 방법과의 하이브리드 접근이 유망하다.

#### ② 계산 효율성 개선
Toys4K 테스트 세트에서 형상당 평균 652.6초가 소요되는 속도 문제는 실시간 또는 대규모 처리를 위해 반드시 해결해야 한다. 커스텀 CUDA 커널을 통해 최대 8.5배 속도 향상이 달성되었으나, 추가적인 병렬화·근사 알고리즘 연구가 필요하다.

#### ③ 파트 의미론(Part Semantics) 통합
현재 방법은 기하학적 피팅에 집중하며 의미론적 파트 레이블은 부가적이다. ResFit은 거친 의미론적 레이블 향상에도 활용될 수 있다. 향후에는 의미론적 일관성을 목적함수에 직접 통합하는 연구가 필요하다.

#### ④ 장면 수준(Scene-level) 확장
현재는 주로 객체 수준에 적용되며, 대규모 실내·실외 장면으로의 확장은 MSD 계산 복잡도와 기본체 수 관리 문제를 수반한다. SuperDec 등 장면 분해 관련 연구와의 결합이 고려되어야 한다.

#### ⑤ 변형 가능한(Deformable) 기본체와의 결합
정적 파라미터 최적화 외에, 시간에 따라 변형되는 형상이나 점진적으로 학습 가능한 기본체 계열로 확장하면 애니메이션·동적 장면에서의 활용도가 높아질 것이다.

#### ⑥ 신경망 기반 초기화 가속
MSD 기반 초기화는 직관적이지만 계산 비용이 크다. 경량 신경망으로 초기화 시드를 예측하고 ResFit으로 정제하는 **하이브리드 학습-최적화 파이프라인**이 속도와 품질의 균형점을 제공할 수 있다.

---

## 📚 참고 자료 및 출처

| 번호 | 출처 |
|---|---|
| 1 | **arXiv 논문 원문:** Ganeshan et al., "Residual Primitive Fitting of 3D Shapes with SuperFrusta," arXiv:2512.09201, Dec. 2025. https://arxiv.org/abs/2512.09201 |
| 2 | **CVPR 2026 공식 논문:** https://openaccess.thecvf.com/content/CVPR2026/papers/Ganeshan_Residual_Primitive_Fitting_of_3D_Shapes_with_SuperFrusta_CVPR_2026_paper.pdf |
| 3 | **공식 프로젝트 페이지:** https://bardofcodes.github.io/superfit/ |
| 4 | **GitHub 공식 코드:** https://github.com/BardOfCodes/superfit |
| 5 | **Adobe Research 블로그:** "A Single Primitive to Rule Them All: SuperFit at CVPR 2026," https://research.adobe.com/news/superfit-at-cvpr-2026-compresses-3d-shapes-into-editable-building-blocks/ |
| 6 | **Wang Yifan 연구자 페이지:** https://yifita.netlify.app/publication/superfit/ |
| 7 | **Thibault Groueix 연구자 페이지:** http://imagine.enpc.fr/~groueixt/ |
| 8 | **ResearchGate 논문 페이지:** https://www.researchgate.net/publication/398559903 |
| 9 | **Semantic Scholar:** https://www.semanticscholar.org/paper/Residual-Primitive-Fitting-of-3D-Shapes-with-Ganeshan-Gadelha/915398d40e7d5e9875d2d41d6697e6d2790e898d |
| 10 | **arXiv HTML 버전 (상세 내용 포함):** https://arxiv.org/html/2512.09201v1 |
| 11 | **관련 연구 — SuperDec:** Fedele et al., "SuperDec: 3D Scene Decomposition with Superquadric Primitives," ICCV 2025. arXiv:2504.00992 |
| 12 | **관련 연구 — D²CSG:** "D²CSG: Unsupervised Learning of Compact CSG Trees," arXiv:2301.11497 |
| 13 | **관련 연구 — CPFN:** "CPFN: Cascaded Primitive Fitting Networks," arXiv:2109.00113 |
| 14 | **관련 연구 — Light-SQ:** "Light-SQ: Structure-aware Shape Abstraction with Superquadrics," arXiv:2509.24986 |
| 15 | **GameDev News 분석:** https://gamedev.net/news/624/ |

> ⚠️ **정확도 주의:** 본 답변에서 SuperFrustum의 SDF 완전 수식(내부 구현 세부 공식)은 논문 보충자료(supplementary)에 있으며 공개 출처에서 완전히 확인되지 않아 파라미터 의미와 정의 수준으로만 기술하였습니다. 완전한 수식은 arXiv 원문 또는 GitHub 코드를 직접 참조하시기 바랍니다.
