
# CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization

> **논문 정보**
> - **제목**: CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization
> - **저자**: Jiawei Zhang, Jiahe Li, Xiaohan Yu, Lei Huang, Lin Gu, Jin Zheng, Xiao Bai
> - **게재**: ECCV 2024 (arXiv:2405.12110)
> - **출판**: Lecture Notes in Computer Science, vol. 15059, Springer, pp. 335–352

---

## 1. 핵심 주장 및 주요 기여 요약

3D Gaussian Splatting(3DGS)은 3D 가우시안으로 구성된 래디언스 필드를 생성하여 장면을 표현하는 방법이지만, 희소한(sparse) 학습 뷰에서는 과적합이 쉽게 발생하여 렌더링 품질에 악영향을 미친다. 이 논문은 Sparse-View 3DGS를 개선하기 위한 새로운 **Co-Regularization(공동 정규화) 관점**을 제안한다.

### 핵심 주장 (Key Claims)

| 항목 | 내용 |
|------|------|
| 관찰(Observation) | 동일한 희소 뷰로 두 개의 3DGS를 독립 학습 시 **Point Disagreement**와 **Rendering Disagreement**가 발생함 |
| 핵심 발견 | 두 불일치(disagreement)와 정확한 재구성 사이에 **부(−)의 상관관계**가 존재함 |
| 제안 방법 | Co-Pruning + Pseudo-view Co-Regularization으로 inaccurate Gaussian 억제 |
| 의의 | GT(ground-truth) 정보 없이 비지도 방식으로 재구성 품질을 예측·개선 가능 |

### 주요 기여 (Contributions)

① **Point Disagreement와 Rendering Disagreement를 정의·측정**하여 두 불일치가 정확한 재구성과 부적 상관관계에 있음을 최초로 입증하고, ground-truth 없이도 재구성 품질을 평가하는 척도로 활용 가능함을 보였다. ② **Co-Pruning**과 **Pseudo-view Co-Regularization**을 제안하여 각각 Point/Rendering Disagreement를 억제함으로써 더 정확한 3D 가우시안 래디언스 필드를 달성했다.

Co-Pruning과 Pseudo-view Co-Regularization을 갖춘 CoR-GS는 일관성 있고 컴팩트한 기하 구조를 재구성하며, 여러 벤치마크에서 SOTA 대비 경쟁력 있는 품질을 달성한다.

---

## 2. 해결하고자 하는 문제, 제안 방법(수식 포함), 모델 구조, 성능 향상 및 한계

---

### 2-1. 해결하고자 하는 문제 (Problem Statement)

희소한 학습 뷰 하에서 3D-2D 투영의 모호성으로 인해 최적화가 가우시안을 정확히 보정하지 못하고, 결과적으로 차이(difference)가 누적된다.

기존 방법들은 사전 학습된 깊이 추정기(depth estimator)의 예측을 정규화로 활용하여 재구성 기하학을 보정하는 방식을 취해왔다. 그러나 외부 감독 신호는 추가적인 노이즈를 도입하여 재구성에 악영향을 줄 수 있다.

이 논문은 두 3D 가우시안 래디언스 필드 사이의 불일치(disagreement)를 억제하는 새로운 Co-Regularization 관점을 도입한다.

---

### 2-2. 배경: 3D Gaussian Splatting 기초

각 3D 가우시안은 중심 위치 $\boldsymbol{\mu} \in \mathbb{R}^3$, 공분산 행렬 $\boldsymbol{\Sigma} \in \mathbb{R}^{3 \times 3}$, 불투명도 $\alpha$, 그리고 색상(SH 계수)으로 정의된다.

$$
G(\mathbf{x}) = e^{-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}
$$

기본 3DGS의 학습 손실은 렌더링 이미지 $\hat{I}$와 GT $I$ 사이의 포토메트릭 손실이다:

$$
\mathcal{L}_{photo} = (1 - \lambda) \mathcal{L}_{1}(\hat{I}, I) + \lambda \mathcal{L}_{D\text{-}SSIM}(\hat{I}, I)
$$

3DGS는 L1과 SSIM을 결합한 포토메트릭 손실을 사용하여 렌더링 이미지와 GT 이미지 사이를 최적화한다.

---

### 2-3. 핵심 관찰: 두 Disagreement

동일한 희소 뷰로 두 3D 가우시안 래디언스 필드를 학습할 때, 두 필드는 **Point Disagreement**와 **Rendering Disagreement**를 나타내며 이는 Densification의 샘플링 구현에서 발생한다. 두 불일치의 부적 상관관계는 ground-truth 없이도 inaccurate 재구성을 식별할 수 있게 한다.

**① Point Disagreement (점 불일치)**

Point Disagreement는 가우시안 위치의 차이를 나타내며, 두 래디언스 필드의 Gaussian 포인트 클라우드 표현 간의 레지스트레이션(registration)으로 평가된다.

두 래디언스 필드 $\mathcal{G}^A, \mathcal{G}^B$에서 가우시안 $g_i^A \in \mathcal{G}^A$에 대한 Point Disagreement는 최근접 이웃 거리로 측정된다:

$$
d_{\text{point}}(g_i^A) = \min_{g_j^B \in \mathcal{G}^B} \| \boldsymbol{\mu}_i^A - \boldsymbol{\mu}_j^B \|_2
$$

**② Rendering Disagreement (렌더링 불일치)**

Rendering Disagreement는 렌더링된 픽셀의 차이를 나타낸다.

두 필드의 렌더링 이미지 $\hat{I}^A, \hat{I}^B$ 간 pixel-wise 차이:

$$
d_{\text{render}}(\mathbf{p}) = \| \hat{I}^A(\mathbf{p}) - \hat{I}^B(\mathbf{p}) \|_1
$$

---

### 2-4. 제안 방법: CoR-GS 구조

CoR-GS는 두 3D 가우시안 래디언스 필드를 동일한 뷰로 학습시키면서 Co-Regularization을 수행한다. Point Disagreement와 Rendering Disagreement를 기반으로 부정확한 재구성을 식별하고 억제하여 Sparse-View 3DGS를 개선한다.

#### 구조 다이어그램 (개념)

```
[Sparse Input Views]
        │
   ┌────┴────┐
   ▼         ▼
 3DGS-A    3DGS-B   ← 동일 뷰, 독립적 랜덤 초기화
   │         │
   └────┬────┘
        │
  ┌─────┴──────┐
  ▼             ▼
Co-Pruning   Pseudo-view
(Point       Co-Regularization
Disagreement)(Rendering
              Disagreement)
  │             │
  └──────┬──────┘
         ▼
   Regularized 3DGS
```

---

#### (1) Co-Pruning

CoR-GS는 Co-Pruning으로 Point Disagreement를 억제한다. Co-Pruning은 두 3D 가우시안 래디언스 필드를 두 개의 포인트 클라우드로 취급하여 포인트별 매칭을 수행한다. 상대 포인트 클라우드에서 가까운 매칭 포인트가 없는 가우시안을 **이상값(outlier)**으로 간주하여 제거한다.

이상값 판단 기준 (임계값 $\tau$):

$$
g_i^A \text{ is pruned} \quad \text{if} \quad d_{\text{point}}(g_i^A) > \tau
$$

이를 통해 두 필드에서 공통적으로 지지되지 않는 "부유 가우시안(floating Gaussians)"을 제거한다.

#### (2) Pseudo-view Co-Regularization

Rendering Disagreement를 억제하기 위해 CoR-GS는 Pseudo-view Co-Regularization을 사용한다. 훈련 뷰를 보간하여 온라인 Pseudo-view를 샘플링하고, 높은 Rendering Disagreement를 보이는 픽셀을 부정확하게 렌더링된 것으로 간주한다. 부정확한 렌더링을 억제하기 위해 렌더링된 픽셀의 차이를 정규화 항으로 계산하여 훈련 뷰 손실과 결합한다.

Pseudo-view $v_{pseudo}$에서의 Co-Regularization 손실:

$$
\mathcal{L}_{co\text{-}reg}(v_{pseudo}) = \sum_{\mathbf{p}} w(\mathbf{p}) \cdot \| \hat{I}^A(\mathbf{p}) - \hat{I}^B(\mathbf{p}) \|_1
$$

여기서 $w(\mathbf{p})$는 Rendering Disagreement에 비례하는 픽셀 가중치이다.

#### 최종 훈련 손실

$$
\mathcal{L}_{total} = \mathcal{L}_{photo}(\hat{I}, I) + \lambda_{co} \cdot \mathcal{L}_{co\text{-}reg}
$$

Co-Pruning은 컴팩트한 표현을 위해 재구성된 장면에서 먼 가우시안을 제거하고, Pseudo-view Co-Regularization은 주변 가우시안을 교정하는 역할을 하여 두 방법이 서로 보완 관계를 이룬다.

---

### 2-5. 성능 향상

LLFF, Mip-NeRF360, DTU, Blender 데이터셋에서의 결과는 CoR-GS가 장면 기하학을 효과적으로 정규화하고, 컴팩트한 표현을 재구성하며, 희소 학습 뷰 하에서 **SOTA(State-of-the-Art) 수준의 새로운 뷰 합성 품질**을 달성함을 보여준다.

Co-Pruning과 Pseudo-view Co-Regularization을 통합한 CoR-GS는 일관성 있고 컴팩트한 기하학을 재구성하며, LLFF, Mip-NeRF360, DTU, Blender 데이터셋에서 SOTA 수준의 Sparse-View 렌더링 성능을 달성한다. 실험 결과는 다양한 장면 상황에서 Sparse-View 3DGS를 정규화하는 방법의 **범용적(universal)** 능력을 보여준다.

평가 지표: **PSNR↑, SSIM↑, LPIPS↓**

---

### 2-6. 한계 (Limitations)

Co-Regularization 기법은 품질 향상에 도움이 되지만, 더 단순한 3D 재구성 방법에 비해 **계산 복잡성이 증가**할 수 있다. 저자들은 상세한 런타임 분석이나 비교를 제공하지 않는다.

이 Co-Regularization은 명시적인 GT 기하학에 의존하지 않고 재구성 아티팩트를 효과적으로 줄이지만, 두 모델을 학습시켜야 하므로 **훈련 복잡성과 계산 오버헤드가 증가**하며, 고도로 대칭적이거나 극단적으로 희소한 장면에서는 효과가 감소한다.

입력으로 희소 뷰를 사용하는 경우, 입력 이미지 간 낮은 공동 가시성(co-visibility)으로 인해 **COLMAP이 실패**할 수 있다는 점도 한계로 지적된다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 외부 사전 지식 불필요: 범용성의 핵심

두 불일치와 정확한 재구성 사이의 부적 상관관계를 경험적 연구로 입증하여, **ground-truth 정보 없이 비지도 방식으로 inaccurate 재구성을 식별**할 수 있다.

기존 방법들은 사전 학습된 깊이 추정기에 의존하는데, 이는 외부 노이즈를 도입할 수 있다. CoR-GS는 이러한 외부 감독 없이도 정규화를 수행함으로써 다양한 장면 유형에 대한 적응력을 높인다.

### 3-2. 다양한 데이터셋 검증

LLFF(전방 향 실제 장면), Mip-NeRF360(무경계 360° 장면), DTU(객체 중심 합성), Blender(합성 데이터셋)에 걸쳐 SOTA 성능을 달성하며, 다양한 장면 상황에서 Sparse-View 3DGS를 정규화하는 **범용적 능력**을 입증하였다.

### 3-3. Co-Regularization의 도메인 독립성

두 신경망 예측의 일치(agreement)는 준지도 학습(semi-supervised learning) 및 노이즈 레이블 학습 등 다양한 task에서 활용되어 왔다. 이 전형적인 파이프라인은 두 신경망을 동시에 훈련시키며, 서로 다른 네트워크 예측의 일치를 통해 레이블되지 않은 데이터에 의사 레이블(pseudo-label)을 부여하거나 노이즈를 제거한다.

이러한 Co-Regularization 아이디어는 **도메인 독립적**이며, 3DGS 이외의 NeRF 계열 방법론이나 다른 3D 표현 방식에도 확장 적용 가능성이 높다.

### 3-4. 일반화 한계

실험이 제한된 합성 및 실제 데이터셋에 집중되어 있어, **복잡한 기하학이나 재질을 가진 더 다양하고 어려운 실제 장면에 대한 일반화** 성능은 불분명하다.

Sparse-View 재구성의 근본적인 한계는 극단적인 데이터 희소성으로 인한 감독 신호의 부족이며, 이는 **교차 뷰 일반화를 제한하고 기하학적 일관성을 저해**한다. 이로 인해 이러한 방법들은 시각적 품질의 부분적 개선만 제공하며 핵심 문제를 근본적으로 해결하지 못한다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4-1. Sparse-View Novel View Synthesis 연구 계보

| 방법 | 연도 | 표현 | 핵심 아이디어 | 주요 특징 |
|------|------|------|--------------|----------|
| **NeRF** (Mildenhall et al.) | 2020 | Implicit Neural | MLP + 볼류메트릭 렌더링 | 기초 모델 |
| **RegNeRF** | 2022 | Implicit Neural | 패치 기반 정규화 | 희소 뷰 NeRF 정규화 |
| **FreeNeRF** | 2023 | Implicit Neural | 주파수·오클루전 정규화 | 외부 감독 없이 안정화 |
| **SparseNeRF** | 2023 | Implicit Neural | 깊이 순위 사전(depth ranking prior) | 약한 감독 신호 |
| **3DGS** (Kerbl et al.) | 2023 | Explicit (Gaussian) | 실시간 Differentiable Rasterization | 기초 모델 |
| **FSGS** | 2023/24 | Gaussian | Gaussian Unpooling + 모노큘러 깊이 | 실시간 Few-shot |
| **SparseGS** | 2023/24 | Gaussian | Floater 제거 + 깊이 렌더링 기법 | 360° 장면 |
| **DNGaussian** | 2024 | Gaussian | Global-Local 깊이 정규화 | CVPR 2024 |
| **CoherentGS** | 2024 | Gaussian | 2D 공간 제어 구조화된 Gaussian | 극히 희소한 입력 |
| **CoR-GS (본 논문)** | 2024 | Gaussian | Co-Regularization (외부 감독 없음) | ECCV 2024 |
| **SCGaussian** | 2024 | Gaussian | 매칭 사전 기반 구조 일관성 | NeurIPS 2024 |
| **CuriGS** | 2025 | Gaussian | 커리큘럼 기반 학생 뷰 정규화 | 과적합 근본 해결 시도 |

### 4-2. 심층 비교: CoR-GS vs. 주요 경쟁 방법

DNGaussian, CoherentGS, DRGS는 깊이 맵 정규화에 집중하며, Few-shot NVS 기반 3DGS 방법들은 대부분 사전 학습된 깊이 추정 네트워크를 파이프라인에 통합하려 한다.

3DGS는 희소 뷰 입력에서 제한된 제약을 가진 비구조적 3D 가우시안이 주어진 소수의 뷰에 과적합되는 경향이 있다. 일부 연구들은 깊이 사전 정보를 추가 제약으로 사용하지만, 신경망 사전 정보는 종종 노이즈가 많고 불분명하여 래디언스 필드 학습을 정확히 안내하지 못한다.

**CoR-GS의 차별점**:
- ✅ **외부 깊이 추정기 불필요** → 노이즈 사전 정보로 인한 오류 없음
- ✅ **비지도 품질 예측** → GT 없이 재구성 품질 평가 가능
- ✅ **기존 3DGS 파이프라인에 플러그인** 형태로 통합 가능
- ⚠️ 두 모델 동시 학습 → 메모리/연산 비용 증가

CuriGS(2025)와 같은 후속 연구는 커리큘럼 기반 전략을 통해 과적합을 실질적으로 완화하고, 보이지 않는 시점에 대한 일반화를 개선하며 극단적인 희소성 조건에서도 미세한 기하 세부사항을 보존하는 방향으로 발전하고 있다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려 사항

### 5-1. 연구에 미치는 영향

**① Co-Regularization 패러다임의 확산**

이 논문의 관찰과 논의가 3D 가우시안 래디언스 필드의 랜덤성(randomness)에 대한 더 깊은 사고를 자극하기를 기대한다.

이 패러다임은 두 모델의 앙상블 불일치를 자기 감독 신호로 활용하는 아이디어를 3D 표현 학습 분야에 최초로 도입했다는 점에서 의미가 크다.

**② 외부 사전 지식 의존 탈피 경향 강화**

외부 사전 정보 없이 스스로 감독 신호를 활용하는 방향의 연구가 확산되고 있으며, 이는 특히 새로운 환경에서 더 robust한 일반화를 가능케 한다.

**③ 3DGS 이외 분야로의 확장 가능성**

Co-Regularization 아이디어는 다음 분야에도 응용 가능성이 있다:
- Dynamic NeRF / 4D Gaussian Splatting
- 의료 영상 재구성 (희소 CT 등)
- 로봇 내비게이션을 위한 실시간 3D 재구성

### 5-2. 앞으로 연구 시 고려할 점

**① 계산 효율성 문제**

Co-Regularization 기법은 품질 향상에 기여하지만 계산 복잡성이 증가할 수 있으며, 저자들은 상세한 런타임 분석을 제공하지 않았다. 단일 모델로 유사한 정규화 효과를 달성하는 경량화 방향이 필요하다.

**② 극단적 희소 뷰 시나리오**

희소 뷰 재구성의 근본적 한계는 극단적인 데이터 희소성으로 인한 감독 신호 부족에 있으며, 이는 교차 뷰 일반화를 제한하고 기하학적 일관성을 저해한다. 따라서 이러한 방법들은 부분적 개선만 제공하며 핵심 문제를 근본적으로 해결하지 못한다. 1~2장의 극단 희소 조건에서의 성능 한계를 극복하는 연구가 필요하다.

**③ 일반화 범위 확장 연구 필요**

현재 실험은 제한된 합성 및 실제 데이터셋에 집중되어 있어, 복잡한 기하학이나 재질을 가진 더 다양하고 어려운 실제 장면에서의 일반화 성능은 불분명하다. 특히 야외 대규모 장면, 반사/투명 재질, 동적 장면 등에 대한 검증이 요구된다.

**④ Disagreement 측정 방식의 고도화**

현재의 Point Disagreement는 단순 최근접 거리 기반이므로, Gaussian의 방향(orientation), 크기(scale), 색상 속성까지 포함한 더 풍부한 불일치 측정 방식 개발이 고려될 수 있다.

**⑤ Generative Prior와의 결합**

Diffusion Model 기반 생성 사전(DiffusioNeRF 등)과 Co-Regularization을 결합하여 보이지 않는 영역의 환각적 채움(hallucination)을 방지하면서도 높은 시각적 품질을 달성하는 방향이 유망하다.

---

## 📚 참고 자료 및 출처

| 번호 | 자료명 | 출처/링크 |
|------|--------|----------|
| 1 | **CoR-GS 논문 (arXiv)** | https://arxiv.org/abs/2405.12110 |
| 2 | **CoR-GS 프로젝트 페이지** | https://jiaw-z.github.io/CoR-GS/ |
| 3 | **CoR-GS GitHub 공식 코드** | https://github.com/jiaw-z/CoR-GS |
| 4 | **CoR-GS ECCV 2024 공식 논문 PDF** | https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00139.pdf |
| 5 | **CoR-GS Springer 출판본** | https://link.springer.com/chapter/10.1007/978-3-031-73232-4_19 |
| 6 | **CoR-GS Macquarie University** | https://researchers.mq.edu.au/en/publications/cor-gs-sparse-view-3d-gaussian-splatting-viaco-regularization/ |
| 7 | **CoR-GS HTML 전문 (arXiv v1)** | https://arxiv.org/html/2405.12110v1 |
| 8 | **DNGaussian (CVPR 2024)** | https://fictionarry.github.io/DNGaussian/ |
| 9 | **SparseGS (arXiv 2312.00206)** | https://arxiv.org/html/2312.00206v3 |
| 10 | **CuriGS (arXiv 2511.16030)** | https://arxiv.org/html/2511.16030v2 |
| 11 | **Sparse-View 3D Reconstruction Survey** | https://arxiv.org/html/2507.16406 |
| 12 | **FewViewGS (NeurIPS 2024)** | https://proceedings.neurips.cc/paper/2024/.../FewViewGS |
| 13 | **SCGaussian (NeurIPS 2024)** | https://proceedings.neurips.cc/paper/2024/.../SCGaussian |
| 14 | **CoherentGS (ECCV 2024)** | https://dl.acm.org/doi/10.1007/978-3-031-73404-5_2 |
| 15 | **Recent Advances in 3DGS (Springer CVM)** | https://link.springer.com/article/10.1007/s41095-024-0436-y |
| 16 | **ADS Abstract (CoR-GS)** | https://ui.adsabs.harvard.edu/abs/2024arXiv240512110Z/abstract |

> ⚠️ **정확도 관련 주의사항**: 본 답변의 수식(Point/Rendering Disagreement 정량화 공식 등)은 논문의 HTML 전문 및 공식 PDF에서 확인 가능한 범위 내에서 재구성하였습니다. 정확한 구현 세부사항(임계값 $\tau$, 가중치 $w(\mathbf{p})$의 정확한 정의 등)은 공식 논문 PDF(ecva.net) 및 GitHub 코드를 직접 확인하시기를 권장합니다.
