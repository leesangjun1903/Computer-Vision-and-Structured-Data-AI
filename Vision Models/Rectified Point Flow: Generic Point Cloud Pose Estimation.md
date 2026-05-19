
# Rectified Point Flow: Generic Point Cloud Pose Estimation

> **📌 논문 정보**
> - **저자:** Tao Sun\*, Liyuan Zhu\* (Stanford), Shengyu Huang (NVIDIA Research), Shuran Song, Iro Armeni (Stanford)
> - **학회:** NeurIPS 2025 (**Spotlight**)
> - **arXiv:** [2506.05282](https://arxiv.org/abs/2506.05282)
> - **프로젝트 페이지:** https://rectified-pointflow.github.io/
> - **코드:** https://github.com/GradientSpaces/Rectified-Point-Flow

---

## 1. 핵심 주장 및 주요 기여 요약

이 논문은 **Rectified Point Flow (RPF)**를 제안하며, 이는 쌍별 포인트 클라우드 등록(pairwise registration)과 다중 파트 형상 조립(multi-part shape assembly)을 단일 조건부 생성 문제로 통합하는 파라미터화 방식이다.

### 주요 기여 (Contributions)

| # | 기여 |
|---|------|
| ① | **통합 생성적 파라미터화**: 두 가지 이질적 태스크를 하나의 프레임워크로 통합 |
| ② | **대칭성의 암묵적 학습**: 레이블 없이 조립 대칭성 내재화 |
| ③ | **Overlap-aware 자기지도 인코더**: 파트 간 기하학적 관계 사전학습 |
| ④ | **SOTA 달성**: 6개 벤치마크에서 최고 성능 |

특히, 통합된 형식화가 다양한 데이터셋에 대한 효과적인 공동 학습(joint training)을 가능하게 하여 공유 기하학적 사전(shared geometric priors) 학습을 촉진하고, 결과적으로 정확도를 향상시킨다.

---

## 2. 상세 설명

### 2-1. 해결하고자 하는 문제

기존 접근법들의 과도한 세분화(fragmentation)는 좁은 도메인에서는 잘 동작하지만, 태스크·객체 카테고리·실세계 모호성 전반에서 일반화에 실패한다. 특히 다중 파트 형상 조립은 고유한 도전을 제시한다. 파트들이 종종 대칭적이거나, 교환 가능하거나, 기하학적으로 모호하여 여러 그럴듯한 구성이 존재할 수 있다. 그 결과 기존의 파트별 등록은 국소적으로는 유효하지만 전체적으로는 모순된 구성을 생성할 수 있다. 이러한 모호성을 극복하려면 강한 감독이나 수작업 휴리스틱 없이 파트 정체성, 상대적 배치, 전체 형상 일관성을 공동으로 추론할 수 있는 모델이 필요하다.

이를 해결하기 위해 논문은 포인트 클라우드 포즈 추정을 **조건부 생성 태스크**로 재정식화한다. 이 접근법으로 Rectified Point Flow는 추출된 포인트 특징을 활용하여 다중 파트 포인트 클라우드에 걸쳐 모든 실현 가능한 조립 상태의 조건부 분포에서 샘플링하며, 조건부 입력 포인트 클라우드의 가능도를 최대화하는 추정치를 생성한다. 포즈 추정을 생성 문제로 재정의함으로써, 데이터의 대칭성과 파트 교환 가능성에서 발생하는 내재적 모호성을 자연스럽게 수용한다.

---

### 2-2. 제안 방법 (수식 포함)

#### (A) 문제 정의

$\Omega$를 파트 인덱스 집합이라 하자. 주어진 미정렬(unposed) 파트 포인트 클라우드:

$$\{\bar{X}_i \in \mathbb{R}^{3 \times N_i}\}_{i \in \Omega}$$

RPF는 각 파트의 포인트 클라우드를 목표 조립 상태 $\{\hat{X}\_i(0)\}_{i \in \Omega}$에서 예측한다. 이후 포즈를 조립된 형상으로부터 복원한다.

#### (B) Rectified Flow 기반 속도장 학습

핵심 아이디어는 3D 포즈 회귀를 재방문하여, 입력 기하구조 위에서 연속적인 포인트별 흐름 필드(flow field)를 학습하는 것으로 문제를 캐스팅하는 조건부 생성 모델을 제안하는 것이다. 구체적으로 Rectified Point Flow는 유클리드 공간의 랜덤 가우시안 노이즈로부터 조립된 객체의 포인트 클라우드를 향해 포인트들의 움직임을 모델링하며, 미정렬 파트 포인트 클라우드를 조건으로 한다.

시간 $t \in [0, 1]$에서의 선형 보간(linear interpolation):

$$X_i(t) = (1 - t)\, \epsilon + t\, X_i^*$$

여기서 $\epsilon \sim \mathcal{N}(0, I)$는 가우시안 노이즈, $X_i^*$는 목표 조립 상태의 포인트 클라우드.

네트워크 $v_\theta$가 학습하는 **속도장(velocity field)**:

$$\frac{dX(t)}{dt} = v_\theta(X(t),\, t,\, \{\bar{X}_i\})$$

Rectified Flow는 노이즈 분포 $\pi_0$와 데이터 분포 $\pi_1$ 사이의 최단 직선 경로를 설정하여 샘플링 효율을 크게 향상시키며, 상수 속도 $x_1 - x_0$로 이 직선 궤적을 보정함으로써 쉽게 최적화할 수 있다.

**학습 목적함수 (Flow Matching Loss):**

```math
\mathcal{L}_{FM} = \mathbb{E}_{t,\, \epsilon,\, X^*} \left\| v_\theta\!\left(X(t),\, t,\, \{\bar{X}_i\}\right) - \left(X^* - \epsilon\right) \right\|_2^2
```

#### (C) 포즈 복원 (Procrustes via SVD)

조건부 포인트 클라우드 $\bar{X}_i$와 추정된 포인트 클라우드 $\hat{X}_i(0)$ 사이에서 SVD를 통해 Procrustes 문제를 풀어 각 비앵커 파트에 대한 강체 변환 $\hat{T}_i$를 복원한다.

$$\hat{T}_i = \arg\min_{T \in SE(3)} \left\| T \cdot \bar{X}_i - \hat{X}_i(0) \right\|_F^2$$

SVD 분해: $\bar{X}_i^\top \hat{X}_i(0) = U \Sigma V^\top$ $\Rightarrow$ $R = V U^\top$, $\mathbf{t} = \bar{\mu}^* - R\bar{\mu}$

#### (D) Overlap-Aware 사전학습

파트 간 관계에 대한 기하학적 인식을 더욱 강화하기 위해, 조건부 포인트 클라우드의 인코더를 대규모 3D 형상 데이터셋에서 자기지도 태스크로 사전학습한다: 파트 전반에 걸친 포인트별 겹침(overlap)을 이진 분류 태스크로 예측한다. 비교 기법인 GARF는 메시 기반 물리 시뮬레이션에 의존하는 반면, 이 논문은 파트 간 기하학적 겹침을 계산하여 경량하고 확장 가능한 대안을 제안한다.

**Overlap 예측 손실:**

$$\mathcal{L}_{overlap} = -\sum_j \left[ y_j \log \hat{p}_j + (1 - y_j) \log(1 - \hat{p}_j) \right]$$

여기서 $y_j \in \{0, 1\}$은 포인트 $j$의 overlap 여부 이진 레이블.

---

### 2-3. 모델 구조

흐름 모델은 인코더와 위치 임베딩(Position Embedding), 그리고 순차적인 DiT 블록(N=6)으로 구성된다. 각 블록은 Part-wise Attention, Global Attention, MLP, AdaLayerNorm 레이어로 이루어진다. 흐름 모델은 6개의 순차적 DiT 블록으로 구성되며, 각 블록의 hidden dimension은 512이다. Multi-head self-attention에서는 attention head 수를 8로 설정하여 head dimension이 64가 된다.

Part-wise Attention과 Global Attention 연산을 분리 적용하여 파트 내부(intra-part) 및 파트 간(inter-part) 기하학적 관계를 모두 포착한다.

```
┌─────────────────────────────────────────────────┐
│         Rectified Point Flow 구조                │
│                                                 │
│  {X̄_i} (Unposed Parts)                         │
│      │                                          │
│      ▼                                          │
│  [Overlap-Aware Encoder (Point Transformer)]    │
│      │  Self-supervised pretrained              │
│      ▼                                          │
│  Conditioning Features                          │
│      │                                          │
│  Noisy Points X(t) + Time Emb.                  │
│      │                                          │
│      ▼                                          │
│  ┌──────────────────────────┐                   │
│  │  DiT Block × 6           │                   │
│  │  - Part-wise Attention   │                   │
│  │  - Global Attention      │                   │
│  │  - MLP + AdaLayerNorm    │                   │
│  └──────────────────────────┘                   │
│      │                                          │
│      ▼                                          │
│  Velocity Field v_θ                             │
│      │  ODE Integration                         │
│      ▼                                          │
│  {X̂_i(0)} → SVD (Procrustes) → {T̂_i}          │
└─────────────────────────────────────────────────┘
```

흐름 모델은 여섯 개의 데이터셋으로 학습되며, 인코더는 이 여섯 개 데이터셋과 추가적인 PartField로 분할된 전처리된 Objaverse v1 데이터셋(약 38k 객체)에서 사전학습된다.

---

### 2-4. 성능 향상

Overlap-aware 인코더와 결합하여 Rectified Point Flow는 쌍별 등록과 형상 조립에 걸친 **6개 벤치마크에서 새로운 SOTA 성능**을 달성한다. 특히 통합된 형식화가 다양한 데이터셋에 대한 효과적인 공동 학습을 가능하게 하여, 공유 기하학적 사전의 학습을 촉진하고 결과적으로 정확도를 향상시킨다.

파라미터화가 서로 다른 등록 태스크에 걸쳐 공동 학습을 지원하며, 각 개별 태스크에서의 성능을 향상시킨다는 것을 보인다.

### 2-5. 한계

파트들이 종종 대칭적이거나 교환 가능하거나 기하학적으로 모호하여 다중의 그럴듯한 국소 구성이 존재할 수 있고, 기존의 파트별 등록은 국소적으로는 유효하지만 전체적으로는 일관성이 없는 구성을 생성할 수 있다는 문제가 내재되어 있다. 논문 자체에서 공개적으로 명시한 한계는 검색된 자료에서 확인하기 어려우나, 다음과 같은 구조적 한계를 고려할 수 있습니다:

- **추론 비용**: ODE 수치 적분이 필요하여 직접 회귀 방법 대비 추론 시간이 길 수 있음
- **포인트 수 의존성**: 포인트 클라우드의 밀도 및 노이즈에 민감할 수 있음
- **대규모 파트 처리**: 매우 많은 파트 수의 경우 Global Attention의 계산 복잡도가 증가

---

## 3. 일반화 성능 향상 가능성 (중점)

이 논문은 3D 포즈 회귀를 재방문하여 입력 기하구조 위에서 연속적인 포인트별 플로우 필드를 학습하는 것으로 문제를 캐스팅하는 **생성적 접근법**을 제안하며, 이는 조립된 형상에 대한 사전(prior)을 효과적으로 포착한다.

### 3-1. 통합 파라미터화에 의한 일반화

Overlap-aware 인코더와 함께 Rectified Point Flow는 쌍별 등록과 형상 조립에 걸친 6개 벤치마크에서 새로운 SOTA를 달성한다. 특히 **통합된 형식화가 다양한 데이터셋에서의 효과적인 공동 학습을 가능하게 하여, 공유 기하학적 사전 학습을 촉진하고 결과적으로 정확도를 높인다.**

### 3-2. 사전학습 전략에 의한 일반화

제안 방법은 여러 3D 형상 데이터셋 전반에 걸쳐 파트 간 관계의 기하학적 인식을 갖춘 **일반화 가능한 사전학습 전략**을 제안하며, 이를 포인트별 overlap 예측으로 형식화한다.

파트 간 기하학적 겹침을 계산하여 사전학습 데이터를 구성하는 경량하고 확장 가능한 대안을 도입함으로써, 다양한 데이터셋에 적용 가능한 일반화 전략을 실현한다.

### 3-3. 대칭성에 대한 암묵적 일반화

기존 연구가 임시방편적 대칭성 처리(ad-hoc symmetry handling)로 파트별 포즈를 회귀하는 것과 달리, 이 방법은 대칭성 레이블 없이 조립 대칭성을 내재적으로 학습한다.

논문은 **일반화 능력(Generalization Ability)**에 관한 별도 섹션을 두어, 동일 카테고리 및 교차 카테고리 파트를 포함한 미지의(unseen) 조립물에 대한 정성적 결과를 제시한다.

또한 이론적 근거를 제공하는 **일반화 경계(Generalization Bounds)** 섹션을 통해 일반화 위험 보증(generalization risk guarantees)을 도출하고 기존 6-DoF 방법들과 비교한다.

### 3-4. 확장성 (대규모 데이터 활용)

흐름 모델은 여섯 개의 데이터셋으로 학습되며, 인코더는 이 여섯 개 데이터셋 외에도 PartField로 분할된 전처리 Objaverse v1 데이터셋(약 38k 객체)을 추가로 활용하여 사전학습된다. 이는 대규모 비지도 3D 데이터를 활용한 일반화 가능성을 보여준다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 방법 | 연도 | 태스크 | 핵심 접근 | RPF와의 차이 |
|------|------|--------|-----------|-------------|
| **DCP** | 2019 | Registration | Attention + SVD | 단일 태스크, 생성 모델 아님 |
| **GARF** | 2024 | Assembly | Flow Matching | 물리 시뮬레이션 기반 사전학습 의존 |
| **RayDiffusion** | 2024 | Camera Pose | Diffusion (Ray 표현) | 2D 이미지 기반, 3D 포인트 클라우드 비대상 |
| **RPF (본 논문)** | 2025 | **Registration + Assembly** | **Rectified Flow** | **두 태스크 통합, 경량 사전학습** |

파트 간 관계에 대한 기하학적 인식을 강화하기 위해 대규모 3D 형상 데이터셋에서 자기지도 태스크(이진 분류로 형식화된 포인트별 overlap 예측)로 인코더를 사전학습한다. GARF도 흐름 모델에 대한 인코더 사전학습의 가치를 강조하지만, 메시 기반 물리 시뮬레이션에 의존한다. 이와 대조적으로 RPF는 파트 간 기하학적 겹침을 계산하여 사전학습 데이터를 구성하는 경량하고 확장 가능한 대안을 도입한다.

또한 논문은 포즈 추정을 위한 Flow Matching을 탐구하는 여러 동시대 연구들을 인식하고 있다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려 사항

### 5-1. 연구에 미치는 영향

#### (1) 통합 프레임워크 패러다임의 확산
3D 포즈 추정을 연속적인 포인트별 플로우 필드 학습으로 재정식화하는 **생성적 접근법**이 향후 3D 비전의 다양한 태스크 통합에 새로운 방향을 제시한다. Rectified Point Flow는 유클리드 공간의 랜덤 가우시안 노이즈로부터 조립된 객체의 포인트 클라우드를 향한 포인트들의 움직임을 모델링하며, 학습된 플로우는 파트 레벨 변환을 암묵적으로 인코딩하여 단일 프레임워크 내에서 판별적 포즈 추정과 생성적 형상 조립을 모두 가능하게 한다.

#### (2) 생성 모델의 3D 기하학 적용 촉진
Rectified Flow Matching은 거의 직선 경로를 따라 확률 질량을 수송하는 결정론적 신경 ODE 프레임워크로, 학습된 속도장을 쌍 표본 간의 상수 벡터와 정렬하는 손실을 사용한다. 이 패러다임이 포인트 클라우드에 효과적으로 적용됨을 입증함으로써, 3D 장면 이해, 로보틱스, 의료 영상 등에서의 후속 연구를 촉진할 것으로 기대된다.

#### (3) 자기지도 기하학적 표현 학습의 방향 제시
여러 3D 형상 데이터셋 전반에 걸쳐 파트 간 관계의 기하학적 인식을 갖춘 일반화 가능한 사전학습 전략과 포인트별 overlap 예측 형식화는 레이블이 부족한 3D 도메인에서 자기지도 학습 전략의 새로운 가능성을 보여준다.

### 5-2. 앞으로 연구 시 고려할 점

#### ① **추론 효율화**
ODE 수치 적분 기반 추론은 실시간 응용에서 병목이 될 수 있다. Rectified Flow는 직선 궤적을 장려하는 강한 기하학적 편향을 가지며, 이 구조적 제약은 단일 Euler 스텝만으로 고품질 생성을 가능하게 하는 등 계산 효율성 측면에서 뚜렷한 이점을 제공한다. 따라서 Consistency Model이나 Flow Distillation 기법을 접목하여 추론 단계를 최소화하는 연구가 중요하다.

#### ② **멀티모달 속도장 처리**
표준 MSE 손실을 사용하기 때문에 학습된 속도장은 실제로는 서로 다른 방향을 가리키는 "ground-truth" 속도장들의 평균이 되어 멀티모달하지 않게 된다. 대칭적 파트에 대한 조립 다양성을 더욱 잘 포착하기 위해 Variational Rectified Flow 등의 방향을 통합하는 것이 유망하다.

#### ③ **일반화 경계의 이론적 강화**
일반화 경계 섹션에서 일반화 위험 보증을 도출하고 기존 6-DoF 방법들과 비교하나, 더욱 엄밀한 PAC-Bayes 또는 Rademacher 복잡도 분석이 향후 이론적 기여로 필요하다.

#### ④ **실세계 포인트 클라우드로의 확장**
현재 방법은 주로 합성 데이터셋에서 평가되었을 가능성이 높다. LiDAR나 RGB-D 센서로 취득한 노이즈가 많은 실세계 포인트 클라우드에서의 강건성 연구가 필수적이다.

#### ⑤ **동적 및 비강체 변형 처리**
현재 프레임워크는 강체 변환( $SE(3)$ )을 가정한다. 관절형 객체(articulated objects)나 비강체 변형(non-rigid deformation)으로의 확장이 로보틱스 및 인체 포즈 추정 분야에서 중요한 연구 방향이다.

---

## 📚 참고 자료 및 출처

| # | 자료 | 링크 |
|---|------|------|
| 1 | **arXiv 논문 원문** | https://arxiv.org/abs/2506.05282 |
| 2 | **arXiv PDF** | https://arxiv.org/pdf/2506.05282 |
| 3 | **프로젝트 공식 페이지** | https://rectified-pointflow.github.io/ |
| 4 | **GitHub 코드 저장소** (NeurIPS 2025 Spotlight) | https://github.com/GradientSpaces/Rectified-Point-Flow |
| 5 | **Hugging Face Papers** | https://huggingface.co/papers/2506.05282 |
| 6 | **OpenReview (NeurIPS 2025)** | https://openreview.net/forum?id=bNTezDPlFH |
| 7 | **NeurIPS 2025 Poster** | https://neurips.cc/virtual/2025/poster/117185 |
| 8 | **arXiv HTML (v1)** | https://arxiv.org/html/2506.05282v1 |
| 9 | **arXiv HTML (v2)** | https://arxiv.org/html/2506.05282 |
| 10 | Variational Rectified Flow Matching (arXiv:2502.09616) | https://arxiv.org/pdf/2502.09616 |
| 11 | Order-Optimal Sample Complexity of Rectified Flows (arXiv:2601.20250) | https://arxiv.org/pdf/2601.20250 |

> ⚠️ **정확도 주의사항**: 본 분석은 공개된 초록, HTML 논문 본문, 프로젝트 페이지, GitHub README, OpenReview, NeurIPS 포스터 정보에 기반합니다. 수식의 세부 계수 및 정확한 벤치마크 수치(표 형태)는 논문 PDF 전문에서 직접 확인하시기를 권장합니다.
