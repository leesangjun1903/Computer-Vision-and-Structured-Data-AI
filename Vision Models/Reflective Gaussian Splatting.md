
# Reflective Gaussian Splatting

> **논문 정보**
> - **제목**: Reflective Gaussian Splatting
> - **저자**: Yuxuan Yao, Zixuan Zeng, Chun Gu, Xiatian Zhu, Li Zhang
> - **arXiv**: [2412.19282](https://arxiv.org/abs/2412.19282) (2024년 12월)
> - **학회**: ICLR 2025 (Conference Paper)
> - **OpenReview**: [xPxHQHDH2u](https://openreview.net/forum?id=xPxHQHDH2u)

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

Novel view synthesis는 NeRF 및 3DGS 기반 방법들의 발전으로 큰 진보를 이루었으나, **반사 객체 재구성**은 실시간 고품질 렌더링과 상호 반사(inter-reflection)를 동시에 달성하는 솔루션이 부재한 과제로 남아 있었다. 이를 해결하기 위해 **Ref-Gaussian** 프레임워크를 제안하며, (I) split-sum 근사를 통한 픽셀 수준 물리 기반 디퍼드 렌더링, (II) Gaussian Splatting 패러다임 내에서 최초로 상호 반사(inter-reflection)를 실현하는 Gaussian 기반 상호 반사 모듈이라는 두 가지 핵심 구성 요소를 포함한다.

### 1.2 주요 기여 (3가지)

주요 기여는 다음과 같다:
- **(I)** 3D 공간에서 상호 반사를 포함한 반사 물체의 실시간 고품질 렌더링 실현
- **(II)** 물리 기반 디퍼드 렌더링과 Gaussian 기반 상호 반사를 특징으로 하는 **Ref-Gaussian** 프레임워크 제안 — 2D Gaussian primitives 채택, material-aware 법선 전파, per-Gaussian shading 초기화를 포함한 기하학 중심 최적화로 더욱 강화
- **(III)** 반사 및 비반사 장면 모두에서 정량적 지표, 시각적 품질, 연산 효율 면에서 기존 방법 대비 우월한 성능, 리라이팅 및 편집 등 다운스트림 애플리케이션 지원

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

NeRF는 3D 장면을 신경 암묵 필드로 표현하며 고품질 렌더링을 달성하지만 연산 비용이 크고 실시간 응용에 한계가 있다. 이를 보완하기 위해 3DGS(3D Gaussian Splatting, Kerbl et al. 2023)가 3D Gaussian을 사용하여 래스터화와 알파 블렌딩을 결합, 실시간 고품질 렌더링을 달성했다. 그러나 **대부분의 NeRF 및 3DGS는 고주파 정반사(specular) 성분 포착 능력의 근본적 한계로 인해 반사 표면 모델링에 어려움**을 겪는다.

기존 Gaussian 기반 방법들의 한계를 정리하면:

| 방법 | 한계 |
|------|------|
| **GShader** (Jiang et al., 2024) | 각 Gaussian에 단순화된 음영 함수를 적용하여 빠른 수렴을 이루지만, 기하학·재질·조명에 상당한 노이즈 발생 |
| **3DGS-DR** (Ye et al., 2024) | 픽셀 수준에서 단순화된 음영 함수를 적용하는 디퍼드 셰이딩으로 최적화를 안정화하지만, 상호 반사 효과를 모델링할 수 없음 |
| **RelightableGaussian** (Gao et al., 2023) | Gaussian 기반 레이 트레이싱으로 가시성을 추론하지만, Monte Carlo 샘플링으로 인해 과도한 노이즈와 계산 오버헤드가 발생 |

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 기반 표현: 2D Gaussian Primitives

3D Gaussian은 고유한 다시점(multi-view) 불일치로 인해 세부 기하학 포착에 어려움이 있다. 따라서 Ref-Gaussian은 시점 일관성 있는 2D 지향 평면 Gaussian 디스크를 렌더링 프리미티브로 사용하는 **2D Gaussian Splatting(Huang et al., 2024)**을 채용한다.

각 2D Gaussian은 공간 위치 $\boldsymbol{\mu} \in \mathbb{R}^3$, 두 주축 벡터 $(\mathbf{t}_u, \mathbf{t}_v)$, 스케일 파라미터 $(s_u, s_v)$로 정의되며, 영향도(influence)는 다음과 같이 표현됩니다:

$$\mathcal{G}(\mathbf{u}) = \exp\!\left(-\frac{u^2}{2s_u^2} - \frac{v^2}{2s_v^2}\right)$$

여기서 $(u, v)$는 2D Gaussian의 로컬 좌표계 내 좌표입니다.

#### 2.2.2 물리 기반 디퍼드 렌더링 (Physically-based Deferred Rendering)

Ref-Gaussian 프레임워크의 개요: 먼저 splatting 프로세스로 피처 맵을 생성하고, 추출된 메시에 레이 트레이싱을 수행하여 렌더링 방정식의 정반사 항에 대한 가시성을 계산한다. 그런 다음 픽셀 수준 피처 맵으로 split-sum 근사를 적용한 렌더링 방정식을 사용하여 최종 물리 기반 렌더링 결과를 산출한다. Ref-Gaussian의 물리 기반 디퍼드 렌더링은 **Disney BRDF 모델의 단순화 버전**을 채용한다.

**렌더링 방정식 (Rendering Equation)**:

$$L_o(\mathbf{x}, \boldsymbol{\omega}_o) = \int_{\Omega} f_r(\mathbf{x}, \boldsymbol{\omega}_i, \boldsymbol{\omega}_o) \cdot L_i(\mathbf{x}, \boldsymbol{\omega}_i) \cdot (\boldsymbol{\omega}_i \cdot \mathbf{n}) \, d\boldsymbol{\omega}_i$$

- $L_o$: 출사 복사 휘도(outgoing radiance)
- $f_r$: BRDF (Bidirectional Reflectance Distribution Function)
- $L_i$: 입사 복사 휘도(incident radiance)
- $\mathbf{n}$: 표면 법선

**BRDF 분해**: Disney BRDF 모델 기반으로 diffuse 항 $f_d$와 specular 항 $f_s$로 분리:

$$f_r(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) = f_d + f_s(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o)$$

$$f_d = \frac{\mathbf{a}}{\pi}$$

$$f_s(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o, \mathbf{x}) = \frac{D \cdot F \cdot G}{4(\boldsymbol{\omega}_i \cdot \mathbf{n})(\boldsymbol{\omega}_o \cdot \mathbf{n})}$$

- $\mathbf{a}$: albedo (확산 색상)
- $D$: 법선 분포 함수 (Normal Distribution Function, NDF)
- $F$: Fresnel 항
- $G$: 기하학적 감쇠(Shadowing-Masking) 함수

**Split-Sum 근사 (Split-Sum Approximation)**:

직접 Monte Carlo 샘플링의 계산 부담을 피하기 위해 specular 적분을 두 항으로 분리:

$$\int_{\Omega} f_s \cdot L_i(\boldsymbol{\omega}_i) \cdot (\boldsymbol{\omega}_i \cdot \mathbf{n}) \, d\boldsymbol{\omega}_i \approx \underbrace{\int_{\Omega} f_s \cdot (\boldsymbol{\omega}_i \cdot \mathbf{n}) \, d\boldsymbol{\omega}_i}_{\text{BRDF LUT}} \cdot \underbrace{\int_{\Omega} L_i(\boldsymbol{\omega}_i) \cdot (\boldsymbol{\omega}_i \cdot \mathbf{n}) \, d\boldsymbol{\omega}_i}_{\text{Pre-filtered Env. Map}}$$

이 방식을 통해 **픽셀 수준 재질 속성(BRDF 포함)으로 렌더링 방정식을 강화**하고, split-sum 근사로 Monte Carlo 샘플링의 과도한 연산을 회피한다.

#### 2.2.3 Gaussian 기반 상호 반사 (Gaussian-Grounded Inter-Reflection)

Ref-Gaussian은 픽셀 수준 디퍼드 셰이딩을 채용하고, **추출된 메시에 레이 트레이싱을 수행하여 렌더링 방정식의 정반사 항에서 가시성을 계산**한다. Split-sum 근사와 함께 렌더링 속도를 유지하면서 상호 반사 효과를 모델링한다.

상호 반사를 포함한 입사광 분해:

$$L_i(\boldsymbol{\omega}_i, \mathbf{x}) = V(\boldsymbol{\omega}_i, \mathbf{x}) \cdot L_{\text{dir}}(\boldsymbol{\omega}_i) + L_{\text{ind}}(\boldsymbol{\omega}_i, \mathbf{x})$$

- $V(\boldsymbol{\omega}_i, \mathbf{x})$: 가시성 함수 (메시 레이 트레이싱으로 계산)
- $L_{\text{dir}}$: 직접 조명 (환경 맵 쿼리)
- $L_{\text{ind}}$: 간접 조명 (상호 반사, inter-reflection)

#### 2.2.4 기하학 향상 기법

**(a) Material-Aware Normal Propagation (재질 인식 법선 전파)**

반사 강도(reflective strength)가 상대적으로 큰 Gaussian은 거의 정확한 법선 벡터를 가진다는 관찰에 기반하여, 이러한 반사 Gaussian을 확장하여 인근 Gaussian에 법선 벡터를 전파한다.

**(b) Per-Gaussian Shading 초기화**

기하학 중심 최적화는 2D Gaussian primitives 채택, material-aware 법선 전파, **per-Gaussian shading 초기화**를 포함한다.

---

### 2.3 모델 구조 (파이프라인)

Ref-Gaussian의 전체 파이프라인은 다음 단계로 구성됩니다:

```
[입력: 멀티뷰 이미지]
        ↓
[1단계: Per-Gaussian Shading 초기화]
  - 각 2D Gaussian에 재질 속성 (albedo, roughness, metallic) 할당
        ↓
[2단계: 2D Gaussian Splatting (Rasterization Pass)]
  - 픽셀 수준 피처 맵 생성:
    normal map, albedo map, roughness map, metallic map
        ↓
[3단계: 메시 추출 + 레이 트레이싱]
  - 2DGS로부터 메시 추출
  - 메시에 레이 트레이싱 → 가시성 V(ω_i, x) 계산
        ↓
[4단계: 물리 기반 디퍼드 렌더링 (Shading Pass)]
  - Split-Sum 근사 적용
  - diffuse + specular(direct + indirect) 합산
  - 최종 픽셀 컬러 생성
        ↓
[출력: 고품질 반사 렌더링]
```

2D Gaussian primitives를 씬 표현에 채용하여 더 정확한 기하학 재구성을 달성한다.

---

### 2.4 성능 향상 및 한계

#### 성능 향상

표준 데이터셋에서의 광범위한 실험을 통해 Ref-Gaussian은 정량적 지표, 시각적 품질, 연산 효율 모든 면에서 기존 방법들을 능가한다는 것이 증명되었다. 또한 **반사 및 비반사 장면 모두를 위한 통합 솔루션**으로서 기능하며, 기존 대안들이 반사 장면에만 집중했던 것을 넘어선다.

Ref-Gaussian은 실시간·고품질 반사 객체/장면 재구성을 위한 새로운 Gaussian splatting 프레임워크이다. 이는 물리 기반 디퍼드 렌더링과 Gaussian 기반 상호 반사의 두 통합 구성 요소로 달성된다. **3D 대비 2D Gaussian primitives가 더 우수함**도 증명되었다. 또한 Ref-Gaussian은 리라이팅 및 편집 등 다른 응용 프로그램에도 활용 가능함을 보인다.

#### 한계

상호 반사를 위해 레이 트레이싱을 결합하는 이 계열의 방법들은 **상당한 계산 오버헤드를 발생시키고 훈련 속도를 현저히 저하시켜**, Gaussian splatting의 효율성 이점을 잠식한다. 근본적으로 이러한 한계는 단일 표현 내에서 시점 불변 기하학과 복잡한 시점 의존 정반사 반사를 모두 인코딩하려는 시도에서 비롯된다.

또한 메시 추출 정확도에 의존하는 가시성 계산의 특성상, 복잡하고 얇은 구조나 투명 객체에서 메시 품질 저하가 렌더링 품질에 영향을 줄 수 있습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 반사·비반사 장면 통합 처리

Ref-Gaussian은 반사 및 비반사 장면 **모두를 위한 통합 솔루션**으로 기능하며, 기존에 반사 장면에만 초점을 맞추던 대안들을 넘어선다.

이는 일반화 성능의 핵심 지표로, 특정 도메인에 과적합되지 않고 다양한 장면 유형에 적용 가능한 범용성을 의미합니다.

### 3.2 2DGS의 기하학적 정확성 기여

3D Gaussian은 고유한 다시점 불일치 문제로 세부 기하학 포착에 어려움이 있다. Ref-Gaussian이 채용하는 **뷰 일관성 있는 2D 지향 평면 Gaussian 디스크** 방식은 이를 해결하여 다양한 뷰포인트와 씬 구조에 대한 일반화 기반을 강화한다.

### 3.3 물리 기반 분해(Disentanglement)의 일반화 효과

반사 효과를 처리하는 최근 방법들은 렌더링 색상을 확산(diffuse)과 정반사(specular) 성분으로 분리하고, PBR 프레임워크를 통해 조명 분해를 학습한다. 각 Gaussian primitive에 metallic, roughness 같은 학습 가능한 반사 관련 속성을 부여하고, splatting pass에서 이 속성들을 스크린-공간 재질 맵으로 래스터화한다.

이처럼 기하학(geometry), 재질(material), 조명(lighting)을 물리적으로 분해함으로써:
- **새로운 조명 조건** 하의 장면에도 일반화 가능 (relighting)
- **미학습 재질 유형**에 대해서도 BRDF의 물리적 근거로 추론 가능
- **도메인 갭 감소**: 합성 데이터 → 실세계 적용 가능성 향상

### 3.4 다운스트림 애플리케이션 지원

Ref-Gaussian은 리라이팅(relighting) 및 편집(editing) 등 다른 응용 프로그램에서도 효과를 보인다.

이는 단순한 novel view synthesis를 넘어 편집, AR/VR, 디지털 트윈 등 다양한 하위 태스크로의 확장 가능성을 의미합니다.

---

## 4. 최신 관련 연구 비교 분석 (2020년 이후)

### 4.1 NeRF 계열 주요 선행 연구

| 연구 | 년도 | 핵심 기여 | Ref-Gaussian 대비 한계 |
|------|------|----------|----------------------|
| **NeRF** (Mildenhall et al.) | 2020 | 신경 복사 필드로 뷰 합성 | 반사 처리 없음, 실시간 불가 |
| **Ref-NeRF** (Verbin et al.) | 2022 | 반사 방향 인코딩으로 반사 표현 향상 | 재질과 환경 조명을 분해하지 않고 단순화된 함수로 셰이딩하여 리라이팅·편집 등 다운스트림 태스크 적용에 한계 |
| **NeRO** (Liu et al.) | 2023 | 반사 객체의 신경 기하학 및 BRDF 재구성 | 훈련을 두 단계로 분할(split-sum으로 기하학 획득 → 정확한 샘플링으로 조명/재질 복원), 실시간 불가 |

### 4.2 3DGS 계열 주요 관련 연구

3D Gaussian splatting (3DGS)(Kerbl et al., 2023)은 3D Gaussian primitives와 타일 기반 미분 가능 래스터라이저를 사용하여, NeRF 기반 방법보다 뛰어난 렌더링 품질과 효율성을 달성했다.

| 연구 | 년도/학회 | 핵심 아이디어 | Ref-Gaussian 대비 위치 |
|------|----------|------------|----------------------|
| **GaussianShader** (Jiang et al.) | CVPR 2024 | 각 3D Gaussian에 셰이딩 함수 적용 | 단순화된 음영 함수로 빠른 수렴을 보이지만 기하학·재질·조명에 상당한 노이즈 발생, 상호 반사 불가 |
| **3DGS-DR** (Ye et al.) | SIGGRAPH 2024 | 디퍼드 셰이딩으로 반사 렌더링 | 픽셀 수준 디퍼드 셰이딩으로 그래디언트 스무딩에 유효하지만, **단순화된 음영 함수 한계로 복잡한 반사 구조 및 상호 반사 모델링 불가** |
| **2DGS** (Huang et al.) | SIGGRAPH 2024 | 기하학적으로 정확한 래디언스 필드를 위한 2D Gaussian | Ref-Gaussian이 2DGS를 기하학 기반으로 채용 |
| **IRGS** (Gu et al.) | CVPR 2025 | 2D Gaussian 레이 트레이싱으로 완전한 렌더링 방정식 구현 | 전체 렌더링 방정식을 단순화 없이 적용하고 미분 가능 2D Gaussian 레이 트레이싱으로 on-the-fly 입사 복사 계산; **효율적인 최적화 스킴**으로 Monte Carlo 샘플링의 계산 부담 처리 |
| **MaterialRefGS** (Zhang et al.) | 2025 | 다시점 일관 재질 추론 | 다시점 관점에서 문제를 재검토하여 다시점 일관 재질 추론과 더 물리 기반의 환경 모델링이 정확한 반사 학습의 핵심임을 증명; 2D Gaussian이 디퍼드 셰이딩 중 다시점 일관 재질 맵을 생성하도록 강제 |
| **Ref-DGS** (Fan et al.) | 2026 | 듀얼 Gaussian으로 기하학과 반사 분리 | 효율적인 래스터화 기반 파이프라인 내에서 표면 재구성과 정반사 반사를 분리; 기하학 Gaussian과 근거리 정반사 상호작용을 포착하는 로컬 반사 Gaussian의 듀얼 표현 도입 — 명시적 레이 트레이싱 없이 |

### 4.3 기술적 비교 요약표

| 방법 | 실시간 | 상호 반사 | 리라이팅 | 기하학 품질 | PBR 분해 |
|------|--------|----------|---------|-----------|---------|
| NeRF | ❌ | ❌ | ❌ | 중간 | ❌ |
| Ref-NeRF | ❌ | ❌ | ❌ | 중간 | ❌ |
| 3DGS | ✅ | ❌ | ❌ | 낮음 | ❌ |
| GaussianShader | ✅ | ❌ | ✅ | 낮음 | ✅ |
| 3DGS-DR | ✅ | ❌ | ✅ | 중간 | 부분 |
| **Ref-Gaussian** | **✅** | **✅** | **✅** | **높음(2DGS)** | **✅** |
| IRGS | ✅ | ✅ | ✅ | 높음 | ✅ |
| Ref-DGS | ✅ | ✅(근사) | ✅ | 높음 | ✅ |

---

## 5. 향후 연구에 미치는 영향 및 고려 사항

### 5.1 향후 연구에 미치는 영향

**① 3DGS 기반 역 렌더링(Inverse Rendering) 연구의 가속화**

Ref-Gaussian이 포함된 deferred shading 기반 PBR 파이프라인은 기본 표준 접근법으로 자리매김하고 있으며, 이는 뷰 의존 반사 효과를 뷰 독립 재질 속성으로 분해하여 빛이 물체와 상호작용하는 방식을 물리적으로 고려하는 방향으로 연구를 유도한다.

**② 상호 반사 모델링의 새로운 기준 제시**

GShader나 3DGS-DR처럼 상호 반사를 모델링하지 못했던 기존 방법들의 한계를 극복하며, Gaussian 기반 레이 트레이싱의 가시성 선계산 방식은 Monte Carlo 샘플링으로 인해 속도가 느리고 부정확한 가시성으로 재구성 품질이 저하되는 문제가 있었다. Ref-Gaussian의 메시 기반 레이 트레이싱 접근법은 이에 대한 실용적 해법을 제시합니다.

**③ Dual 표현 패러다임의 촉진**

단일 표현 내에서 시점 불변 기하학과 복잡한 시점 의존 정반사 반사를 함께 인코딩하려는 시도의 근본적 한계가 인식됨에 따라, 이후 연구에서 효율적인 래스터화 기반 파이프라인 내에서 표면 재구성과 정반사 반사를 분리하거나, 기하학 Gaussian과 로컬 반사 Gaussian으로 구성된 듀얼 표현 방식이 활발히 탐구되고 있습니다.

**④ 물리 기반 다운스트림 응용의 표준화**

Ref-Gaussian은 반사 및 비반사 장면 모두에서 정량적 지표, 시각적 품질, 연산 효율 면에서 우수하며, 리라이팅 및 편집 같은 다운스트림 응용을 지원한다. 이는 AR/VR, 게임, 영화 VFX 등 산업 응용을 위한 물리 기반 3D 자산 생성 파이프라인의 표준화를 촉진합니다.

### 5.2 향후 연구 시 고려할 점

**① 계산 효율성 vs. 물리적 정확도의 트레이드오프**

상호 반사를 위해 레이 트레이싱을 결합하는 최근 방법들은 상당한 계산 오버헤드를 발생시키고 훈련 속도를 현저히 저하시켜 Gaussian splatting의 효율성 이점을 잠식한다. 이 문제를 해결하는 경량화 근사 방법의 개발이 핵심 과제입니다.

**② 메시 추출 품질 의존성 극복**

레이 트레이싱을 위해 메시 추출에 의존하는 구조는 메시 품질에 따라 성능이 달라지는 병목이 됩니다. 명시적 메시 없이 레이 트레이싱을 수행하거나, 메시-Gaussian 하이브리드 표현의 품질을 개선하는 연구가 필요합니다.

**③ 근거리(near-field) vs. 원거리(far-field) 반사 통합**

기존 Gaussian splatting 방법들은 근거리 정반사 반사를 모델링하지 못하거나, 명시적 레이 트레이싱에 의존하여 상당한 계산 비용을 발생시키는 문제 간의 트레이드오프를 안고 있다. 이 두 가지를 효율적으로 통합하는 접근이 중요합니다.

**④ 다시점 재질 일관성(Multi-view Material Consistency)**

다시점 일관 재질 추론과 더 물리 기반의 환경 모델링이 정확한 반사 학습의 핵심으로 밝혀졌습니다. 훈련 데이터가 부족하거나 뷰 간격이 넓을 때의 재질 일관성 확보가 일반화 성능의 주요 연구 과제입니다.

**⑤ 동적 씬(Dynamic Scene)으로의 확장**

현재 Ref-Gaussian은 정적 장면에 특화되어 있습니다. 시간에 따라 변화하는 반사 재질을 포함한 동적 씬 처리는 중요한 미래 연구 방향입니다.

**⑥ 데이터셋 다양성 및 실세계 일반화**

3D Gaussian Splatting(3DGS)은 명시적 표현으로 효율적인 뷰 렌더링을 가능하게 하지만, 반사 표면에서의 성능은 암묵적 신경 방법보다 뒤처지며, 특히 세밀한 기하학과 표면 법선 복원에서 그러하다. 이 격차를 좁히기 위한 새로운 학습 전략과 다양한 실세계 반사 씬 데이터셋 구축이 필요합니다.

---

## 📚 참고 자료

1. **Reflective Gaussian Splatting (Ref-Gaussian)** - Yuxuan Yao, Zixuan Zeng, Chun Gu, Xiatian Zhu, Li Zhang
   - arXiv: https://arxiv.org/abs/2412.19282
   - ICLR 2025: https://proceedings.iclr.cc/paper_files/paper/2025/file/abf3682c9cf9245a0294a4bebe4544ff-Paper-Conference.pdf
   - OpenReview: https://openreview.net/forum?id=xPxHQHDH2u

2. **3D Gaussian Splatting with Deferred Reflection (3DGS-DR)** - Keyang Ye, Qiming Hou, Kun Zhou
   - ACM SIGGRAPH 2024: https://dl.acm.org/doi/10.1145/3641519.3657456
   - Project Page: https://gapszju.github.io/3DGS-DR/

3. **GaussianShader: 3D Gaussian Splatting with Shading Functions for Reflective Surfaces** - Yingwenqi Jiang et al.
   - CVPR 2024: https://ar5iv.labs.arxiv.org/html/2311.17977

4. **IRGS: Inter-Reflective Gaussian Splatting with 2D Gaussian Ray Tracing** - Chun Gu et al.
   - CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/papers/...
   - arXiv: https://arxiv.org/html/2412.15867v1

5. **MaterialRefGS: Reflective Gaussian Splatting with Multi-view Consistent Material Inference** - Wenhang Zhang et al.
   - arXiv 2025: https://arxiv.org/html/2510.11387v1

6. **Ref-DGS: Reflective Dual Gaussian Splatting** - Ningjing Fan et al.
   - arXiv 2026: https://arxiv.org/abs/2603.07664

7. **GS-ROR: 3D Gaussian Splatting for Reflective Object Relighting via SDF Priors**
   - arXiv 2024: https://arxiv.org/html/2406.18544

8. **PolGS++: Physically-Guided Polarimetric Gaussian Splatting for Fast Reflective Surface Reconstruction**
   - arXiv 2026: https://arxiv.org/html/2603.10801

> ⚠️ **정확도 참고**: 본 답변의 수식은 논문 본문의 기술적 설명 및 관련 PBR/BRDF 수식의 표준 형태를 기반으로 재구성되었습니다. 논문 내 정확한 수식 기호 표기는 원문 PDF를 직접 확인하시기 바랍니다.
