
# Mean Flows for One-step Generative Modeling

> **논문 정보**
> - **제목:** Mean Flows for One-step Generative Modeling
> - **저자:** Zhengyang Geng (CMU), Mingyang Deng, Xingjian Bai (MIT), J. Zico Kolter (CMU), Kaiming He (MIT)
> - **arXiv ID:** 2505.13447 (2025년 5월 19일)
> - **발표:** NeurIPS 2025

---

## 1. 🔑 핵심 주장 및 주요 기여 요약

이 논문은 **one-step 생성 모델링을 위한 원칙적이고 효과적인 프레임워크**를 제안합니다. Flow Matching이 순간 속도(instantaneous velocity)를 모델링하는 것과 대조적으로, **평균 속도(average velocity)**라는 새로운 개념을 도입하여 흐름 필드(flow field)를 특성화합니다. 평균 속도와 순간 속도 사이의 수학적 항등식(identity)을 유도하여 신경망 훈련을 지도하는 데 사용하며, 사전학습(pre-training), 지식 증류(distillation), 커리큘럼 학습(curriculum learning) 없이 자기완결적(self-contained)으로 작동합니다.

이 방법(MeanFlow)은 ImageNet 256×256에서 단 1회의 함수 평가(1-NFE)로 **FID 3.43**을 달성하며, 이전 최첨단 one-step 확산/흐름 모델들을 크게 능가합니다.

### 주요 기여 정리

| 기여 항목 | 내용 |
|---|---|
| 이론적 기여 | 평균 속도 개념 및 MeanFlow Identity 유도 |
| 방법론적 기여 | 사전학습·증류 없이 scratch에서 학습 가능 |
| 실증적 기여 | ImageNet 256×256 1-NFE FID 3.43 달성 |
| 관계 통합 | Flow Matching, Consistency Model의 통합 관점 제공 |

---

## 2. 🔬 상세 분석

### 2-1. 해결하고자 하는 문제

Flow Matching과 확산 모델 모두 생성 과정에서 반복적인 샘플링을 수행하며, 최근 연구는 소수 스텝, 특히 단일 스텝(feedforward) 생성 모델에 집중하고 있습니다.

Consistency Model이 이 방향의 선구자 역할을 했으나, 일관성 제약이 네트워크 행동의 속성으로만 부과될 뿐, **학습을 지도해야 할 실제 ground-truth 필드의 속성은 알려지지 않은 문제**가 있었습니다.

Consistency Model들이 scratch에서의 few-step 생성을 가능하게 했지만, **기존 few-step 모델과 multi-step 확산 모델 사이의 성능 격차는 여전히 크게 남아 있었습니다.**

---

### 2-2. 제안하는 방법 (수식 포함)

#### 📐 핵심 개념: 평균 속도 (Average Velocity)

평균 속도는 시간 간격에 대한 변위의 비율로 정의되며, 변위는 순간 속도의 시간 적분으로 주어집니다. 오직 이 정의로부터, 평균 속도와 순간 속도 사이의 명확한 내재적 관계가 유도됩니다.

MeanFlow 프레임워크는 평균 속도를 다음과 같이 정의합니다:

$$u(\mathbf{z}_t, r, t) = \frac{1}{t - r} \int_r^t v(\mathbf{z}_\tau, \tau)\, d\tau$$

여기서:
- $\mathbf{z}_t$: 시각 $t$에서의 상태 변수
- $v(\mathbf{z}_\tau, \tau)$: 시각 $\tau$에서의 **순간 속도**
- $u(\mathbf{z}_t, r, t)$: 구간 $[r, t]$ 에 걸친 **평균 속도**
- $r$: 시작 시각, $t$: 종료 시각

#### 📐 MeanFlow Identity (핵심 항등식)

훈련에 적합한 형식을 갖추기 위해 정의식을 재작성하고, $t$에 대해 양변을 미분합니다 (곱 규칙과 미적분학의 기본 정리 사용). 항을 재배열하여 **MeanFlow Identity**를 얻습니다:

$$\frac{\partial}{\partial t}\left[(t - r)\, u(\mathbf{z}_t, r, t)\right] = v(\mathbf{z}_t, t)$$

이를 전개하면:

$$(t - r)\frac{\partial u}{\partial t} + u + (t-r)\left(\frac{\partial u}{\partial \mathbf{z}_t}\right)^\top v = v$$

즉, 정리하면:

$$u(\mathbf{z}_t, r, t) = v(\mathbf{z}_t, t) - (t - r)\left[\frac{\partial u}{\partial t} + \left(\frac{\partial u}{\partial \mathbf{z}_t}\right)^\top v\right]$$

이 MeanFlow Differential Identity는 효율적인 생성 모델링의 핵심 수학적 관계로, **구간 전체의 평균 변위를 나타내는 평균 속도 필드와 기저 흐름 방정식에 의해 정의되는 순간 속도 필드 사이의 정밀한 연결**을 제공합니다. 이 항등식의 엄밀한 정식화는 원칙적인 손실 함수, 안정적인 훈련 역학, 그리고 확장 가능한 구현으로 이어집니다.

#### 📐 손실 함수 (Training Loss)

핵심 통찰은 평균 속도의 정의 방정식을 조작하여 **순간 속도만 접근 가능한 상황에서도 훈련에 적합한 최적화 목표**를 구성할 수 있다는 것입니다.

신경망 $u_\theta(\mathbf{z}_t, r, t)$가 MeanFlow Identity의 우변(RHS)을 target으로 학습하도록 구성됩니다:

$$\mathcal{L}(\theta) = \mathbb{E}\left[\left\| u_\theta(\mathbf{z}_t, r, t) - \text{sg}\left(v_\theta(\mathbf{z}_t, t) - (t - r)\left[\frac{\partial u_\theta}{\partial t} + \text{JVP}(u_\theta, v_\theta)\right]\right) \right\|^2\right]$$

여기서:
- $\text{sg}(\cdot)$: stop-gradient 연산자
- $\text{JVP}$: Jacobian-Vector Product (연쇄 미분 항 계산에 사용)

훈련 시 JVP를 사용해 MeanFlow Identity의 연쇄 미분 보정 항을 계산하며, 이는 훈련 오버헤드를 약 20% 증가시키는 데 그칩니다.

#### 📐 One-step 생성 (Inference)

MeanFlow 모델의 궁극적인 목표는 신경망 $u_\theta(\mathbf{z}\_t, r, t)$로 평균 속도를 근사하는 것입니다. 이를 정확하게 근사할 경우, 단 한 번의 $u_\theta(\epsilon, 0, 1)$ 평가로 전체 흐름 경로를 근사할 수 있습니다. 이 접근법은 추론 시 시간 적분을 명시적으로 근사할 필요가 없어 단일 또는 소수 스텝 생성에 훨씬 더 적합합니다.

---

### 2-3. 모델 구조

ImageNet 256×256 실험에서 표준 VAE 토크나이저를 사용하여 잠재 표현을 추출하며, 잠재 크기는 $32 \times 32 \times 4$입니다.

MeanFlow는 다양한 모델 크기와 훈련 시간에 걸쳐 1-NFE FID 결과를 평가하며, Transformer 기반 확산/흐름 모델(DiT, SiT)과 일관된 행동으로 **유망한 확장성(scalability)**을 보여줍니다.

신경망 $u_\theta(\mathbf{z}_t, r, t)$는 기존 DiT(Diffusion Transformer) 구조를 기반으로 하되, 입력에 시작 시각 $r$과 종료 시각 $t$ 두 가지 시각 조건을 함께 사용합니다.

---

### 2-4. 성능 향상

MeanFlow 모델은 1-NFE 생성에서 강력한 성능을 보여주며, ImageNet 256×256에서 1-NFE(함수 평가 횟수 1회)로 **FID 3.43**을 달성합니다. 이는 이전 SOTA 방법들을 크게 능가합니다.

MeanFlow는 구간 평균 속도를 회귀하고 Euler나 고차 솔버로 인한 적분 오차 없이 직접 $O(1)$ 스텝 추론을 달성합니다. 경험적으로 MeanFlow는 one-step과 multi-step 접근 방식 간의 품질 격차를 크게 줄입니다 (ImageNet 256에서 FID 3.43).

MeanFlow 프레임워크는 더 안정적인 훈련과 더 나은 CFG(Classifier-Free Guidance) 통합을 가능하게 하여, few-step과 multi-step 방식으로 scratch 학습된 확산 모델 사이의 격차를 크게 좁힙니다.

MeanFlow는 확산 및 multi-step flow matching 대비 **10~100배의 속도 향상**을 제공합니다.

---

### 2-5. 한계점

MeanFlow의 주요 트레이드오프는 반복적 접근 방식 대비 one-step 모델 편향(bias)이 약간 증가하는 것이지만, 강력한 백본과 적절한 훈련에서 실질적으로 이 격차는 최소화됩니다.

주요 미해결 문제 및 한계로는: **(1)** Jacobian 계산 의존도 축소(또는 더 저렴한 대안으로 대체), **(2)** 훈련 중 구간 선택 및 가중치를 위한 적응형 스케줄 개발, **(3)** 이산(discrete) 또는 하이브리드 데이터 공간, 고차 흐름, 매우 높은 기하학적·의미론적 보존이 요구되는 도메인으로의 일반화, **(4)** 유한 데이터 하에서 평균 속도 필드 학습으로 인한 근사 오차 및 백본 아키텍처와 연계된 표현력(expressivity) 한계의 이론적 분석이 있습니다.

---

## 3. 🌐 일반화 성능 향상 가능성

### 3-1. 이론적 기반

표준 Flow Matching 방법이 알려진 궤적에서 도출된 지도 목표(supervised target)에 순간 속도 $v$를 맞추는 것과 달리, MeanFlow는 **시간 평균된 양(time-averaged quantities)**에 맞춥니다. 이는 2차 구조를 도입하고, 국소 노이즈에 대한 민감도를 줄이며 안정적인 훈련을 가능하게 하는 **거친(coarse), 전역적 정보에 기반한 훈련**을 허용합니다.

MeanFlow 미분 항등식은 순간적인 시스템 거동 모델링에서 구간 평균 역학 포착으로의 전환을 나타냅니다. 그 일반화(대수적 구간 분할 및 고차 미분 항등식)는 점점 더 표현력 있고 효율적인 생성 모델을 위한 이론적 기반을 제공합니다.

### 3-2. 확장성 (Scalability)

MeanFlow 모델은 더 큰 모델 크기와 다양한 훈련 기간에 걸쳐 1-NFE FID를 평가하며, Transformer 기반 확산/흐름 모델(DiT, SiT)과 일관된 행동으로 **1-NFE 생성에서의 유망한 확장성**을 보여줍니다.

2차 MeanFlow 샘플링의 회로 복잡도(Transformer 네트워크 기반)는 상수 깊이, 다항식 크기의 임계 회로 내에 유지되어 **더 풍부한 역학 표현에서도 실용적인 확장성을 보장합니다**.

### 3-3. 다양한 도메인 적용

강화학습(MuJoCo 벤치마크에서 확산 정책과 비교할 만한 성능), 추천 시스템, 음성 합성 등 다양한 도메인에 MeanFlow 기반 모델이 배포되었으며, 반복적 모델과 비교해 거의 품질 저하 없이 one-step 또는 few-step 샘플링이 가능합니다.

One-step 및 two-step SplitMeanFlow 모델은 Doubao 등 대규모 음성 합성 제품에 성공적으로 배포되어 **20배의 속도 향상**을 달성했습니다.

### 3-4. 자기 일관성(Self-Consistency)

MeanFlow Identity(또는 그 대수적 유사체)의 충족은 생성된 궤적이 미분(MeanFlow) 또는 대수적(SplitMeanFlow) 의미에서 **자기 일관성**을 가지도록 보장하며, 전체 흐름을 평균으로 요약함으로써 큰 이산화 오차 없이 one-step 또는 few-step 샘플링이 가능합니다.

---

## 4. 🔭 후속 연구에 미치는 영향 및 고려 사항

### 4-1. 후속 연구에 미치는 영향

이 연구는 one-step 확산/흐름 모델과 multi-step 선행 모델 사이의 격차를 크게 줄였으며, 향후 연구들이 이 강력한 모델들의 기초를 재검토하도록 동기를 부여합니다.

실제로 MeanFlow 등장 이후 다양한 후속 연구들이 파생되었습니다:

- **Modular MeanFlow (MMF, 2025):** 시간 평균 속도 필드 학습을 위한 유연하고 이론적으로 탄탄한 접근으로, 미분 항등식 기반 손실 함수 패밀리와 그래디언트 변조 메커니즘을 도입하며, 커리큘럼 방식 워밍업 스케줄을 제안합니다. 기존 일관성 기반 및 flow matching 방법을 통합·일반화하면서 고차 미분을 회피합니다.

- **SplitMeanFlow (2025):** MeanFlow의 핵심인 미분 항등식이 대수적 일관성의 극한으로 복원됨을 증명하여, SplitMeanFlow가 평균 속도 필드 학습을 위한 직접적이고 더 일반적인 기반임을 확립했습니다.

- **AlphaFlow / 관련 연구들:** MeanFlow가 시간 평균 속도를 연속 일관성으로 예측하는 반면, 이산 일관성 탐색, Consistency-FM, FACM, IMM 등 다양한 하이브리드 접근법들이 등장했습니다.

### 4-2. 2020년 이후 주요 관련 연구 비교 분석

| 연구 | 방법 | NFE | 특징 | 한계 |
|---|---|---|---|---|
| **DDPM** (2020) | Score Matching | 수백 | 고품질 생성 | 매우 느린 샘플링 |
| **Flow Matching** (2022~2023) | 순간 속도 회귀 | 50~100 | 개념적으로 단순 | 여전히 다수 스텝 필요 |
| **Rectified Flow** (ICLR 2023) | 직선화 흐름 학습 | 수십 | 빠른 샘플링 | 증류 필요 |
| **Consistency Models** (2023) | 일관성 제약 | 1~2 | 최초 scratch 1-step | ground-truth field 불명확 |
| **DMD** (CVPR 2024) | 분포 매칭 증류 | 1 | 강력한 1-step | 사전학습 teacher 필요 |
| **Shortcut Models** (ICLR 2025) | 직접 매핑 + shortcut | 1 | 단순 | 이론적 기반 약함 |
| **MeanFlow** (NeurIPS 2025) | 평균 속도 회귀 | **1** | scratch 학습, FID 3.43 | JVP 계산 오버헤드 |

MeanFlow는 DDPM($\|\epsilon - \epsilon_\theta\|^2$, 수백 스텝), Conditional Flow Matching($\|v_\theta(x_t,t)-(x_1-x_0)\|^2$, 50~100 ODE 스텝)을 일반화하며, 구간 평균 속도를 회귀하여 Euler 또는 고차 솔버의 적분 오차 없이 직접 $O(1)$ 스텝 추론을 달성합니다.

### 4-3. 향후 연구 시 고려할 점

1. **JVP 계산 효율화**
   비싼 미분 계산의 제거, 확장 가능한 계산 복잡도, 경험적으로 검증된 샘플 충실도는 이 프레임워크를 차세대 생성 모델 및 시뮬레이션-프리 계산 패러다임의 유망한 후보로 자리매김합니다. 그러나 현재 JVP 계산은 여전히 필수이므로 이를 더 저렴한 대안으로 대체하는 연구가 필요합니다.

2. **훈련 안정성 개선**
   Modular MeanFlow(2025)가 λ 기반 그래디언트 차단 연산자와 커리큘럼을 도입하여 훈련을 개선하고 있으며, α-Flow(2025)는 MeanFlow 손실을 궤적 flow matching 항과 궤적 일관성 항으로 분해하여 음의 그래디언트 상관관계를 발견하고 α-어닐링 훈련 스케줄을 제안합니다.

3. **이산 및 하이브리드 데이터로의 확장**
   향후 연구는 평균 기반 일관성 원리의 추가 통합, 고급 가이던스 메커니즘, 그리고 거친 역학이 바람직하거나 확장성이나 일반화를 희생하지 않으면서 고차 정밀도가 필요한 더 넓은 도메인으로의 응용을 탐구할 것입니다.

4. **이론적 한계 분석**
   유한 데이터 하에서 평균 속도 필드 학습으로 인한 근사 오차 및 백본 아키텍처와 연계된 표현력 한계에 대한 이론적 분석이 필요합니다.

5. **다운스트림 태스크 일반화**
   고급 가이던스, 혼합 기반 흐름(mixture-based flows), 또는 정책 학습 및 제어를 위한 컨트롤러 정규화와의 통합이 미래 연구의 중요한 방향입니다.

---

## 📚 참고 자료

1. **Geng et al. (2025)** — "Mean Flows for One-step Generative Modeling," arXiv:2505.13447, NeurIPS 2025. https://arxiv.org/abs/2505.13447
2. **HTML Full Paper** — https://arxiv.org/html/2505.13447v1
3. **OpenReview** — https://openreview.net/forum?id=uWj4s7rMnR
4. **NeurIPS 2025 Poster** — https://neurips.cc/virtual/2025/poster/115487
5. **Emergent Mind (MeanFlow)** — https://www.emergentmind.com/topics/meanflow
6. **Emergent Mind (MeanFlow Differential Identity)** — https://www.emergentmind.com/topics/meanflow-differential-identity
7. **Emergent Mind (MeanFlow-based Models)** — https://www.emergentmind.com/topics/meanflow-based-model
8. **You et al. (2025)** — "Modular MeanFlow," arXiv:2508.17426. https://arxiv.org/abs/2508.17426
9. **ResearchGate (Modular MeanFlow)** — https://www.researchgate.net/publication/394940463
10. **SplitMeanFlow (2025)** — "Interval Splitting Consistency in Few-Step Generative Modeling," arXiv:2507.16884. https://arxiv.org/abs/2507.16884
11. **AlphaFlow (2025)** — "Understanding and Improving MeanFlow Models," arXiv:2510.20771. https://arxiv.org/abs/2510.20771
12. **Unofficial PyTorch Implementation** — https://github.com/haidog-yaqub/MeanFlow
13. **NVIDIA AYF (Align Your Flow)** — https://research.nvidia.com/labs/toronto-ai/AlignYourFlow/
