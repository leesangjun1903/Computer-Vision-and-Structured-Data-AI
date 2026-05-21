
# SuperDec: 3D Scene Decomposition with Superquadric Primitives

> **논문 정보**
> - **제목**: SuperDec: 3D Scene Decomposition with Superquadric Primitives
> - **저자**: Elisabetta Fedele, Boyang Sun, Leonidas Guibas, Marc Pollefeys, Francis Engelmann
> - **소속**: ETH Zurich, Stanford University, Microsoft
> - **arXiv**: [2504.00992](https://arxiv.org/abs/2504.00992) (2025년 4월 1일 제출, v2: 2026년 3월 19일)
> - **학회**: ICCV 2025 (Accepted)
> - **프로젝트 페이지**: https://super-dec.github.io
> - **코드**: https://github.com/elisabettafedele/superdec

---

## 1. 핵심 주장 및 주요 기여 요약

### 🔑 핵심 주장

SuperDec는 슈퍼쿼드릭(superquadric) 프리미티브로의 분해를 통해 간결한 3D 씬 표현을 생성하는 접근법입니다. 최근 대부분의 방법들이 사실적인 3D 재구성을 위해 기하학적 프리미티브를 활용하는 것과 달리, SuperDec는 이를 활용하여 **간결하면서도 표현력 높은(compact yet expressive)** 표현을 얻고자 합니다. 이를 위해, 임의의 객체의 포인트 클라우드를 소수의 슈퍼쿼드릭 집합으로 효율적으로 분해하는 새로운 아키텍처를 설계하였습니다.

기존 표현들(NeRF, 3DGS 등)은 사실성(photorealism)에 뛰어나지만, 컴팩트성에 대한 명시적인 제어를 제공하지 못하며, 크고 비모듈형의 씬 인코딩을 초래하여 명시적 공간 추론이 필요한 작업에는 적합하지 않습니다.

### 📌 주요 기여 3가지


1. **SuperDec 소개**: 슈퍼쿼드릭 프리미티브를 활용한 3D 씬 분해를 위한 새로운 방법 제안
2. **최고 성능 달성**: 다중 클래스를 동시 학습(jointly trained)한 ShapeNet에서 최첨단 객체 분해 점수 달성
3. **다양한 다운스트림 응용 검증**: 로봇 작업 및 제어 가능한 생성 콘텐츠 창작에서의 3D 슈퍼쿼드릭 씬 표현의 유효성 입증


---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

3D 씬 표현에는 포인트 클라우드, 메쉬, SDF, 복셀 그리드 등 다양한 형식이 있으며, 최근에는 NeRF, Gaussian Splatting 같은 멀티뷰 접근법이 인기를 얻고 있습니다. 이러한 방법들은 광도(photometric) 손실을 최적화하여 기반 표현이 관측된 이미지와 일치하도록 합니다.

그러나 이러한 접근법들은 사실성(photorealism)은 뛰어나지만, 컴팩트성에 대한 명시적 제어가 부족하여 대규모의 비모듈형 씬 인코딩을 초래하는 문제가 있습니다.

또한 기존 학습 기반 방법들은 카테고리 특화 학습(category-specific training)에 의존하는 한계가 있었습니다. 이는 전역 형태 특징만을 인코딩하여 카테고리 내부 일반화에는 충분하지만, **카테고리 외부 객체(out-of-category objects)의 분해에는 효과적이지 않다**는 문제가 있습니다.

---

### 2-2. 제안 방법 및 수식

#### 슈퍼쿼드릭(Superquadric) 기본 개념

슈퍼쿼드릭의 표면은 다음 음함수(implicit function)로 정의됩니다:

$$F(\mathbf{x}; \boldsymbol{\lambda}) = \left[\left(\frac{x_1}{a_1}\right)^{\frac{2}{\varepsilon_2}} + \left(\frac{x_2}{a_2}\right)^{\frac{2}{\varepsilon_2}}\right]^{\frac{\varepsilon_2}{\varepsilon_1}} + \left(\frac{x_3}{a_3}\right)^{\frac{2}{\varepsilon_1}} = 1$$

여기서:
- $a_1, a_2, a_3$: 각 축 방향의 크기(scale) 파라미터
- $\varepsilon_1, \varepsilon_2$: 형태를 결정하는 지수(shape exponents), $\varepsilon \in (0, 2]$

전역 좌표계로 확장하기 위해서는 6개의 추가 파라미터(평행이동 3개 + 회전 3개)가 필요하며, 슈퍼쿼드릭 1개당 총 **11개의 파라미터**로 표현됩니다.

전체 파라미터 벡터는 다음과 같이 정의됩니다:

$$\boldsymbol{\lambda} = (a_1, a_2, a_3, \varepsilon_1, \varepsilon_2, \mathbf{t}, \mathbf{R})$$

$$\boldsymbol{\lambda} \in \mathbb{R}^{11}$$

#### 손실 함수(Loss Function)

재구성 손실은 다음과 같이 구성됩니다:

$$\mathcal{L}_{rec} = \mathcal{L}_{P \to SQ} + \mathcal{L}_{SQ \to P} + \mathcal{L}_{N}$$

각 항의 의미:

- $\mathcal{L}_{P \to SQ}$: 포인트 클라우드에서 슈퍼쿼드릭 표면까지의 거리 손실 (Chamfer Distance 방향 1)
- $\mathcal{L}_{SQ \to P}$: 슈퍼쿼드릭 표면에서 포인트 클라우드까지의 거리 손실 (Chamfer Distance 방향 2)

$\mathcal{L}_{N}$ 항은 Yang et al.의 재구성 손실로 정의되며, 학습 중 **법선(normal) 정보를 통합**하여 수렴 속도를 가속화하는 데 사용됩니다.

또한 정확도뿐만 아니라 컴팩트성도 추구하기 위해, **더 적은 프리미티브의 사용을 장려하는 간결성 손실(parsimony loss)** 을 도입합니다.

전체 최종 손실:

$$\mathcal{L}_{total} = \mathcal{L}_{rec} + \lambda_{pars} \mathcal{L}_{pars}$$

여기서 $\lambda_{pars}$는 간결성 손실의 가중치입니다.

**소프트 할당 행렬(Soft Assignment Matrix)**:

정제된 슈퍼쿼드릭 특징 $F_{SQ}$와 포인트 특징 $F_{PC}$는 두 개의 예측 헤드로 전달되며, 세분화 헤드(segmentation head)는 포인트를 슈퍼쿼드릭에 할당하는 소프트 할당 행렬 $M \in \mathbb{R}^{N \times P}$를 예측합니다.

$$M \in \mathbb{R}^{N \times P}, \quad M_{ij} = P(\text{point } i \text{ belongs to superquadric } j)$$

---

### 2-3. 모델 구조

SuperDec는 포인트 클라우드 데이터에 슈퍼쿼드릭을 피팅하기 위해 **Point-Voxel CNN과 트랜스포머 디코더(Transformer Decoder)** 를 사용하여 슈퍼쿼드릭 파라미터를 예측합니다.

$N$개의 포인트로 구성된 객체의 포인트 클라우드가 주어지면, Transformer 기반 신경망이 $P$개의 슈퍼쿼드릭에 대한 파라미터를 예측하고, 포인트를 슈퍼쿼드릭에 할당하는 소프트 세분화 행렬(soft segmentation matrix)도 예측합니다. 예측되는 파라미터에는 **11개의 슈퍼쿼드릭 파라미터**와 **객체성 점수(objectness score)** 가 포함됩니다.

이러한 예측은 이후 **Levenberg–Marquardt (LM) 최적화**를 위한 효과적인 초기화를 제공하며, LM 최적화 단계가 슈퍼쿼드릭을 세밀하게 조정합니다.

전체 파이프라인 구조를 정리하면:

```
입력 포인트 클라우드 (N points)
        ↓
Point-Voxel CNN (포인트 특징 추출, FPC)
        ↓
Transformer Decoder (슈퍼쿼드릭 쿼리 → FSQ)
        ↓
예측 헤드 1: 슈퍼쿼드릭 파라미터 (11개 × P개)
예측 헤드 2: 소프트 할당 행렬 M ∈ ℝ^{N×P}
        ↓
Levenberg–Marquardt 최적화 (LM 정제)
        ↓
최종 슈퍼쿼드릭 표현 (P개, compact)
```

**씬(scene) 전체로의 확장 구조:**

객체별로 문제를 지역적으로 풀고, 인스턴스 세분화(instance segmentation) 방법의 능력을 활용하여 솔루션을 전체 3D 씬으로 확장합니다. 이를 통해 임의의 객체 포인트 클라우드를 소수의 슈퍼쿼드릭 집합으로 효율적으로 분해하는 새로운 아키텍처를 설계합니다.

---

### 2-4. 성능 향상

SuperDec는 다중 클래스를 동시에 학습한 ShapeNet에서 **최첨단 객체 분해 점수를 달성**하였으며, 로봇 작업 및 제어 가능한 생성 콘텐츠 창작에서 3D 슈퍼쿼드릭 씬 표현의 유효성을 입증하였습니다.

ShapeNet에서 학습하고 ScanNet++에서의 객체 인스턴스와 Replica 씬에서의 일반화 능력을 입증하였으며, 로봇 조작 및 제어 가능한 시각 콘텐츠 생성을 포함한 다양한 다운스트림 응용을 지원합니다.

특히 로봇 작업에서는 **경로 계획(path planning)과 객체 파지(object grasping)**, 편집 가능한 3D 씬 표현을 위한 **제어 가능한 이미지 생성**에서도 실용적 유용성을 보였습니다.

---

### 2-5. 한계점

기존 방법들의 한계는 카테고리 특화 학습에 있었고, SuperDec는 이를 극복하려 했지만, 여전히 **전역 형태 특징만을 인코딩하는 모델 설계에서는 카테고리 외부 객체 분해에 한계**가 있을 수 있습니다.

LM 최적화를 후처리 단계로 활용하는데, 서로 다른 LM 최적화 라운드 수가 최종 예측에 미치는 영향을 실험한 결과, **카테고리 외부(out-of-category)에서 카테고리 내부(in-category)보다 더 큰 개선 효과**가 나타나는 것을 확인할 수 있으며, 이는 동시에 카테고리 내부 성능은 상대적으로 포화 상태에 이를 수 있음을 시사합니다.

또한 슈퍼쿼드릭은 매끄러운 기하 형태(ellipsoid, cylinder 등)를 표현하는 데 강점이 있지만, **날카로운 모서리나 복잡한 오목(concave) 형상**에 대한 표현력의 물리적 한계도 존재합니다.

---

## 3. 일반화 성능 향상 가능성

### 3-1. 크로스 카테고리 일반화 (Cross-Category Generalization)

SuperDec의 가장 중요한 일반화 관련 설계 결정은 **다중 클래스 동시 학습(jointly multi-class training)** 입니다:

ShapeNet에서 모델을 학습시키고, ScanNet++에서 추출한 객체 인스턴스와 전체 Replica 씬에서의 **일반화 능력을 입증**하였습니다.

기존 방법들이 카테고리 특화 학습에 의존하는 것과 달리, SuperDec는 이 한계를 극복하기 위해 전역이 아닌 **로컬 형태 특징(local shape features)** 을 활용하는 아키텍처를 설계하였으며, 이는 카테고리 내부 일반화를 넘어 **카테고리 외부 객체에도 효과적인 분해**가 가능하게 합니다.

### 3-2. 합성-실제 도메인 일반화 (Synthetic-to-Real Generalization)

ShapeNet(합성 데이터)에서 학습한 아키텍처가 ScanNet++(실제 환경에서 캡처된 객체 인스턴스)와 Replica(실내 씬) 데이터셋에서 일반화 능력을 입증한 것은, **합성-실제(sim-to-real) 도메인 갭** 극복 측면에서도 중요한 성과입니다.

### 3-3. LM 최적화를 통한 일반화 강화

네트워크의 예측은 이후 Levenberg–Marquardt (LM) 최적화를 위한 효과적인 초기화를 제공하며, 이 과정이 슈퍼쿼드릭을 정제합니다.

이 설계는 학습 중 보지 못한 새로운 도메인의 객체에도 적용될 때, 신경망 예측 + LM 수치 최적화의 **2단계 파이프라인**이 일반화 성능을 더욱 견고하게 만들어 줍니다. LM 최적화는 카테고리 외부 객체에서 더 큰 개선 폭을 보이므로, 미지의 도메인에서도 활용 가능성이 높습니다.

### 3-4. 일반화 성능 향상을 위한 향후 방향

- **더 다양한 실세계 데이터셋 포함**: 옥외 씬, 의료 데이터, 산업 환경 등으로의 훈련 데이터 확장
- **더 풍부한 인스턴스 세분화 모델 활용**: Open-vocabulary 3D segmentation (OpenMask3D 등)과의 결합
- **도메인 적응(domain adaptation) 기법 통합**: 실세계 노이즈·결측·밀도 불균형을 다루기 위한 증강 전략

---

## 4. 연구에 미치는 영향 및 향후 고려 사항

### 4-1. 앞으로의 연구에 미치는 영향

#### (A) 로봇 공학과의 접점
로봇 작업에서 경로 계획 및 객체 파지를 위한 씬 표현으로서의 유효성이 입증되어, **로봇 공학과 3D 씬 이해 연구의 교량 역할**을 할 것으로 기대됩니다.

#### (B) 생성형 AI와의 결합
슈퍼쿼드릭 기반 컴팩트 표현이 로봇 조작과 **제어 가능한 시각 콘텐츠 생성 및 편집** 등 다양한 다운스트림 응용에 유용하게 활용될 수 있음을 보였습니다. 이는 ControlNet, 3D 생성 모델 등과의 결합 연구를 자극할 것입니다.

#### (C) 3D Gaussian Splatting과의 비교/결합 연구 자극
PartGS, GaussianBlock과 같은 후속 연구들이 슈퍼쿼드릭 표면에 Gaussian을 배치하여 슈퍼쿼드릭의 기하학적 간결성과 Gaussian Splatting의 고품질 렌더링을 결합하는 방향으로 발전하고 있으며, SuperDec는 이러한 연구 방향에 중요한 기반이 됩니다.

#### (D) 파트 기반 씬 이해 연구
SuperDec가 임의의 3D 씬을 컴팩트하고 모듈형의 슈퍼쿼드릭 프리미티브 집합으로 표현할 수 있음을 보인 것은, **파트 기반 객체 이해**, **씬 그래프 생성**, **기능적 추론** 등의 연구에 새로운 방향성을 제시합니다.

---

### 4-2. 향후 연구 시 고려할 점

| 항목 | 내용 |
|---|---|
| **표현력 vs. 간결성의 균형** | 간결성 손실($\mathcal{L}_{pars}$)의 가중치 조절은 하이퍼파라미터에 민감하며, 복잡한 씬에서의 최적 균형 연구 필요 |
| **오목 형상 처리** | 슈퍼쿼드릭은 볼록(convex) 형태에 강점이 있어, 오목 형상은 다수의 프리미티브 조합으로만 처리 가능 — 비볼록 프리미티브 확장 연구 고려 |
| **인스턴스 세분화 오류의 영향** | 인스턴스 세분화 방법에 의존하여 전체 씬으로 확장하므로, 세분화 오류가 SuperDec의 품질에 직접 영향을 미침 — 세분화 오류에 강건한 파이프라인 설계 필요 |
| **실시간성** | LM 최적화 단계가 추가 계산 비용을 발생시키므로, 실시간 로봇 응용을 위한 경량화 연구 필요 |
| **텍스처·외관 정보 통합** | 현재는 기하학적 형태만을 다루므로, 색상·재질 정보를 함께 표현하는 다중 모달 프리미티브 확장 가능성 |
| **동적 씬(dynamic scene) 처리** | 현재 정적 씬에 초점 — 시간적 변화를 다루는 동적 씬 표현으로의 확장 연구 여지 |
| **평가 지표 다양화** | Chamfer Distance 외에도 IoU, Hausdorff Distance, 의미적 일관성(semantic consistency) 등 다양한 지표 도입 필요 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 발표 | 방식 | 프리미티브 | 장점 | 단점 |
|---|---|---|---|---|---|
| **SQ-Parsing** (Paschalidou et al.) | CVPR 2019 | 학습 기반 | Superquadric | 표현력 높은 프리미티브 | 카테고리 특화, 일반화 부족 |
| **EMS** (Liu et al.) | CVPR 2022 | 최적화 기반 (MLE/GUM) | Superquadric | 노이즈·이상치 강건 | 느린 속도, 분해 한계 |
| **MonteBoxFinder** (Ramamonjisoa et al.) | ECCV 2022 | 최적화 기반 | Box/Cuboid | 노이즈 포인트 클라우드 처리 | 표현력 제한 |
| **3D Gaussian Splatting** (Kerbl et al.) | SIGGRAPH 2023 | 멀티뷰 최적화 | 3D Gaussian | 고품질 렌더링 | 비모듈형, 대용량 |
| **PartGS / GaussianBlock** | 2024 | 혼합(학습+렌더링) | SQ + Gaussian | 사실적 렌더링 + 기하 분해 | 복잡한 파이프라인 |
| **SuperQ-Grasp** (Tu & Desingh) | arXiv 2024 | 로봇 응용 | Superquadric | 로봇 파지에 특화 | 범용 씬 처리 어려움 |
| **SuperQuadricOcc** | arXiv 2025 | 자기지도 학습 | SQ + Gaussian 근사 | 실시간 점유 예측 | 렌더링 품질 보통 |
| **⭐ SuperDec (본 논문)** | ICCV 2025 | 학습+최적화 (Transformer+LM) | Superquadric | 다중 클래스 일반화, 씬 전체 처리, 다운스트림 응용 | 인스턴스 세분화 의존, 오목 형상 한계 |

학습 기반 방법들은 신경망이 적절한 재구성 손실을 갖출 때 기하학적 프리미티브 파라미터를 직접 예측하여 포인트 클라우드를 최소한의 프리미티브 집합으로 분해할 수 있음을 보여왔습니다. SuperDec는 이 흐름 위에서 **다중 카테고리 일반화와 전체 씬 처리**라는 두 가지 핵심 과제를 동시에 해결한 것이 차별점입니다.

---

## 📚 참고 자료 (출처)

1. **[주 논문]** Fedele, E., Sun, B., Guibas, L., Pollefeys, M., Engelmann, F. (2025). *SuperDec: 3D Scene Decomposition with Superquadric Primitives*. arXiv:2504.00992. https://arxiv.org/abs/2504.00992
2. **[ICCV 2025 공식 게재]** IEEE Xplore: https://ieeexplore.ieee.org/document/11445486/
3. **[ICCV 2025 CVF Open Access]** https://openaccess.thecvf.com/content/ICCV2025/papers/Fedele_SuperDec_3D_Scene_Decomposition_with_Superquadrics_Primitives_ICCV_2025_paper.pdf
4. **[프로젝트 페이지]** https://super-dec.github.io
5. **[공식 코드]** GitHub: https://github.com/elisabettafedele/superdec
6. **[ICCV 2025 포스터]** https://iccv.thecvf.com/virtual/2025/poster/867
7. **[ResearchGate]** https://www.researchgate.net/publication/390405362_SuperDec_3D_Scene_Decomposition_with_Superquadric_Primitives
8. **[NASA ADS Abstract]** https://ui.adsabs.harvard.edu/abs/2025arXiv250400992F/abstract
9. **[관련 연구 - SuperQuadricOcc]** arXiv:2511.17361. https://arxiv.org/html/2511.17361
10. **[관련 연구 - EMS/GUM]** Liu et al. (2022). *Robust and Accurate Superquadric Recovery: A Probabilistic Approach*. CVPR 2022.
11. **[관련 연구 - MonteBoxFinder]** Ramamonjisoa et al. (2022). *MonteBoxFinder*. ECCV 2022.
12. **[관련 연구 - OpenMask3D]** Takmaz et al. (2023). *OpenMask3D*. NeurIPS 2023.
13. **[관련 연구 - SuperQ-Grasp]** Tu & Desingh (2024). *SuperQ-Grasp*. arXiv 2024.

> ⚠️ **정확도 관련 고지**: 본 논문의 세부 수식 및 아키텍처 내용 일부(예: 파서모니 손실의 정확한 수식, 네트워크의 정확한 레이어 구성)는 논문 PDF의 접근 가능한 HTML 버전에서 일부 수식 렌더링이 불완전하게 표시되어 있어, 원 논문 PDF를 직접 확인하여 검증하실 것을 권장합니다.
