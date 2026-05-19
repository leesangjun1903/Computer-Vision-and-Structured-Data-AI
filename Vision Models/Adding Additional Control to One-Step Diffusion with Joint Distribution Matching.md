
# Adding Additional Control to One-Step Diffusion with Joint Distribution Matching

> **출처 및 참고자료**
> - arXiv:2503.06652v2 (https://arxiv.org/abs/2503.06652)
> - ICCV 2025 Proceedings (https://openaccess.thecvf.com/content/ICCV2025/papers/Luo_Adding_Additional_Control_to_One-Step_Diffusion_with_Joint_Distribution_Matching_ICCV_2025_paper.pdf)
> - Moonlight Literature Review (https://www.themoonlight.io/en/review/adding-additional-control-to-one-step-diffusion-with-joint-distribution-matching)
> - 관련 논문: Diff-Instruct\* (arXiv:2410.20898), Trajectory Distribution Matching (arXiv:2503.06674)

---

## 1. 핵심 주장과 주요 기여 요약

### 🔑 핵심 주장

Variational Score Distillation과 같은 diffusion distillation 방법으로 one-step 생성이 가능해졌지만, 새로운 구조적 제약이나 사용자 선호 등 새로운 제어 조건을 추가할 때는 기반 diffusion 모델을 수정하고 재증류(redistillation)해야 하며, 이는 계산 비용이 크고 시간이 많이 소요된다.

이를 해결하기 위해 Yihong Luo, Tianyang Hu, Yifan Song, Jiacheng Sun, Zhenguo Li, Jing Tang이 제안한 **Joint Distribution Matching (JDM)**은 one-step 생성 모델에서 새로운 제어 조건을 효율적으로 추가하는 방법론이다.

### ✅ 주요 기여 (3가지)

| 기여 | 내용 |
|---|---|
| ① 새로운 프레임워크 | One-step student가 teacher 모델이 모르는 제어 조건을 처리 |
| ② CFG 개선 | Classifier-Free Guidance의 활용도 향상 |
| ③ HFL 통합 | Human Feedback Learning의 seamless 통합 |

JDM은 이미지-조건 결합 분포(image-condition joint distributions) 간의 역 KL 발산을 최소화하며, tractable upper bound를 유도하여 fidelity learning과 condition learning을 분리한다. 이 비대칭 증류 방식은 teacher 모델이 알지 못하는 제어 조건을 one-step student가 처리할 수 있게 하며, CFG 활용 개선 및 인간 피드백 학습(HFL)의 원활한 통합을 지원한다.

---

## 2. 해결하고자 하는 문제 / 제안하는 방법 / 모델 구조 / 성능 및 한계

### 🚨 해결하고자 하는 문제

#### 문제 1: One-step 생성 모델에서의 제어 추가 어려움

기존 base diffusion 모델에 대해서는 denoising score matching(DSM)으로 학습된 ControlNet 모델로 새로운 제어 가능성을 얻는 방식을 사용했지만, one-step 생성에 ControlNet을 확장하면 세밀한 제어 성능 저하와 낮은 샘플 품질 등 심각한 한계가 발생한다.

#### 문제 2: 기존 증류 방법의 구조적 한계

현재 one-step 생성을 위한 diffusion distillation 방법론은 teacher diffusion 모델의 능력을 복제하는 student 모델 증류에만 집중하며, teacher의 능력을 넘어서는 방법은 연구되지 않았다.

이러한 한계는 원래 diffusion 모델이 처리하도록 설계되지 않은 새로운 제어 조건을 추가할 때 특히 심각하게 드러난다.

---

### 📐 제안하는 방법 (수식 포함)

#### 핵심 목적함수: Reverse KL Divergence on Joint Distribution

JDM의 핵심 목표는 생성된 이미지와 조건 입력, 즉 두 결합 분포(joint distributions) 간의 역 Kullback-Leibler(KL) 발산을 최소화하는 것이다.

JDM의 최적화 목표를 수식으로 표현하면:

$$\mathcal{L}_{\text{JDM}} = D_{\text{KL}}\left( p_\theta(x, c) \,\|\, q(x, c) \right)$$

여기서:
- $p_\theta(x, c)$: student 모델이 생성하는 이미지 $x$와 조건 $c$의 결합 분포
- $q(x, c)$: 목표 결합 분포 (teacher 기반)
- $D_{\text{KL}}(\cdot \| \cdot)$: 역 KL 발산 (reverse KL divergence)

역 KL 발산을 결합 분포로 전개하면:

$$D_{\text{KL}}(p_\theta(x, c) \| q(x, c)) = \mathbb{E}_{p_\theta(x,c)}\left[\log \frac{p_\theta(x,c)}{q(x,c)}\right]$$

이를 조건부 분포로 분해하면 (chain rule 적용):

$$= D_{\text{KL}}(p_\theta(x) \| q(x)) + \mathbb{E}_{p_\theta(x)}\left[D_{\text{KL}}(p_\theta(c|x) \| q(c|x))\right]$$

JDM은 이 발산에 대한 tractable upper bound를 유도하여 fidelity learning과 condition learning을 효과적으로 분리하고, 이 비대칭적 목적함수를 통해 teacher diffusion 모델이 알지 못하는 제어 조건까지 처리하는 one-step student를 얻는다.

이를 통해 upper bound는 두 개의 독립적인 항으로 나뉜다:

$$\mathcal{L}_{\text{JDM}} \leq \underbrace{\mathcal{L}_{\text{fidelity}}}_{\text{Teacher로부터 이미지 품질 학습}} + \underbrace{\mathcal{L}_{\text{condition}}}_{\text{새로운 제어 조건 학습}}$$

> ⚠️ **주의**: 위 upper bound의 구체적인 전개 수식은 검색된 결과에서 완전히 공개되지 않았습니다. 논문 원문(PDF) 확인을 권장합니다.

이 분리 메커니즘은 CFG(Classifier-Free Guidance)의 개선된 활용을 가능하게 하며, 학습 과정에서 Human Feedback Learning(HFL)의 원활한 통합도 지원한다. 결과적으로 JDM은 대부분의 경우 단 one-step만으로 multi-step ControlNet 기반 방법을 능가한다.

---

### 🏗️ 모델 구조

JDM의 학습 구조는 **비대칭 증류(Asymmetric Distillation)** 패러다임에 기반한다:

```
┌─────────────────────────────────────────────────────┐
│           JDM 학습 파이프라인                         │
│                                                     │
│  Teacher DM (Multi-step, frozen)                    │
│       ↓  fidelity signal                            │
│  One-Step Student (학습 대상)                         │
│       ↑  condition signal                           │
│  New Control Module (e.g., ControlNet 형태)          │
│       ↑                                             │
│  New Control Input (depth, edge, HFL reward...)     │
└─────────────────────────────────────────────────────┘
```

- **Teacher 모델**: 기존 multi-step diffusion 모델 (frozen, fidelity 학습에 활용)
- **Student 모델**: One-step 생성기 (JDM 목적함수로 학습)
- **Control Module**: Teacher가 모르는 새로운 조건 신호를 처리하는 별도 모듈

JDM의 핵심 구조적 특징은 fidelity learning과 condition learning이 분리(decoupled)된다는 점이며, 이를 통해 비대칭 증류 방식이 one-step student로 하여금 teacher 모델이 알지 못하는 제어 조건을 처리하고 CFG 및 HFL을 원활하게 통합할 수 있도록 한다.

---

### 📊 성능 향상

실험 결과, JDM은 대부분의 경우 단 one-step으로 multi-step ControlNet과 같은 기준 방법들을 능가하며, CFG 또는 HFL 통합을 통해 one-step text-to-image 합성에서 최첨단(state-of-the-art) 성능을 달성했다.

| 평가 기준 | JDM의 성과 |
|---|---|
| 제어 조건 대응 | Teacher 미지(unknown) 조건도 처리 |
| 속도 | 단 1 NFE (one-step) |
| 비교 대상 | multi-step ControlNet 초과 |
| 추가 기능 | CFG 개선 + HFL 통합 |

---

### ⚠️ 한계

ControlNet을 one-step 생성에 직접 확장할 경우 발생하는 세밀한 제어 성능 저하 문제가 JDM의 동기가 되었으며, 논문 내 명시적 한계로는 다음을 추론할 수 있습니다:

1. **Teacher 모델 의존성**: Fidelity 학습은 여전히 teacher DM에 의존하므로, teacher의 품질 상한선에 제약받을 가능성 존재
2. **학습 데이터 요구**: 새로운 조건에 대한 paired 데이터(이미지-조건 쌍)가 필요
3. **복잡한 조건 조합**: 다수의 새로운 제어를 동시에 추가할 경우 성능 보장 불명확

> ⚠️ 이 한계 분석은 검색 결과에서 명시적으로 확인된 내용과 논문의 구조적 특성에서 합리적으로 도출한 것으로, 논문 원문의 Limitation 섹션 직접 확인을 권장합니다.

---

## 3. 모델의 일반화 성능 향상 가능성

JDM이 일반화 성능에 기여하는 핵심 메커니즘은 세 가지입니다:

### 🌐 (1) Teacher-Unknown Control 처리 능력

비대칭 목적함수의 특성으로 인해 teacher diffusion 모델이 알지 못하는 새로운 제어를 처리할 수 있으며, 이 분리 메커니즘은 CFG의 개선된 사용을 촉진한다.

이는 곧 **학습 시 보지 못한 제어 조건에 대한 일반화**를 의미하며, 새로운 사용자 요구사항이나 도메인에 유연하게 적응할 수 있는 가능성을 시사합니다.

### 🧩 (2) Fidelity-Condition 분리에 의한 모듈형 일반화

$$\mathcal{L}_{\text{JDM}} \leq \mathcal{L}_{\text{fidelity}}(\theta) + \mathcal{L}_{\text{condition}}(\theta, \phi)$$

- $\mathcal{L}_{\text{fidelity}}$는 teacher로부터 이미지 품질을 학습 (도메인 불변적 특성)
- $\mathcal{L}_{\text{condition}}$은 새로운 조건에 특화된 학습 (플러그인 방식)

이처럼 학습 목표가 분리되면 각 모듈이 독립적으로 일반화될 수 있어, 새로운 제어 조건이 추가되더라도 fidelity 성능의 catastrophic forgetting을 방지할 가능성이 높습니다.

### 🤝 (3) Human Feedback Learning (HFL) 통합을 통한 분포 외(OOD) 일반화

JDM의 비대칭 증류 방식은 teacher 모델이 알지 못하는 제어를 처리할 수 있게 하며, CFG 사용 개선 및 Human Feedback Learning(HFL)의 원활한 통합을 지원한다.

HFL 통합은 단순히 학습 데이터 분포에 맞추는 것을 넘어, 인간 선호도라는 *분포 외 일반화 신호*를 학습에 반영할 수 있어 실제 사용 환경(real-world deployment)에서의 일반화 성능을 높입니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 🔮 연구에 미치는 영향

#### ① One-step 생성 모델의 제어 확장성 패러다임 제시

지금까지 diffusion distillation로 one-step 생성이 가능해졌지만, 새로운 제어 조건 추가는 base 모델 수정과 재증류가 필요했다. JDM은 이 패러다임을 깨고, **재증류 없이 새로운 제어를 추가하는 방향**을 제시합니다. 이는 continual learning, modular AI 연구에 직접적 영향을 미칩니다.

#### ② 결합 분포 관점의 새로운 증류 이론적 기반

KL 발산을 결합 분포 $p(x, c)$ 수준에서 정의하고 이를 두 항으로 분해하는 접근법은, 향후 다른 생성 모델(예: flow matching, consistency model)에도 적용 가능한 이론적 프레임워크를 제공합니다.

#### ③ 인간 피드백 + 빠른 생성의 융합 연구 촉진

이 분리 메커니즘은 CFG의 개선된 활용을 촉진하고, 학습 과정에서 Human Feedback Learning(HFL)의 원활한 통합을 가능하게 한다. 이는 RLHF와 생성 속도 최적화를 동시에 추구하는 연구의 기반이 됩니다.

---

### 🔬 앞으로 연구 시 고려할 점

| 고려 사항 | 상세 내용 |
|---|---|
| **다중 제어 조건의 동시 추가** | 단일 새 조건이 아닌, 다수의 이질적 조건을 동시에 추가할 때의 목적함수 충돌(conflict) 방지 방법 연구 필요 |
| **Upper Bound의 tightness** | Tractable upper bound가 실제 KL 발산과 얼마나 근접한지에 따라 학습 효율이 달라지므로, tighter bound 탐색이 중요 |
| **Teacher 모델 품질 의존성** | Teacher의 생성 품질이 낮으면 fidelity 항의 한계가 student에 전파되므로, teacher-free 또는 multi-teacher 확장 가능성 검토 필요 |
| **비디오/3D 생성으로의 확장** | 현재 이미지 합성에 집중되어 있으나, 시공간적 제어가 필요한 비디오/3D 도메인 확장 가능성 탐구 |
| **CFG Scale 최적화** | CFG 개선 효과가 구체적으로 어떤 scale/schedule에서 최대화되는지에 대한 이론적 분석 부족 |
| **paired data 의존성 완화** | 조건-이미지 쌍 데이터가 항상 필요하다면 적용 범위가 제한됨. 약지도(weakly-supervised) 또는 비지도 학습 방향 탐색 필요 |

---

## 📚 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 방법 | 주요 특징 | JDM과의 비교 |
|---|---|---|---|
| **Diff-Instruct** (Luo et al., 2023) | VSD 기반 one-step 증류 | Teacher-student KL 최소화 | JDM의 직접적 전신; teacher 범위 내 제어만 가능 |
| **Diff-Instruct\*** (arXiv:2410.20898) | Score 기반 인간 선호 정렬 | RLHF + one-step 생성기 | HFL 통합 아이디어 공유, but 새로운 조건 추가 불가 |
| **ControlNet** (Zhang et al., 2023) | DSM 기반 조건 추가 | 구조 제어 (edge, depth 등) | Multi-step 기반; one-step 적용 시 성능 저하 |
| **Variational Score Distillation (VSD)** | KL 최소화 | 단일 분포 매칭 | 결합 분포 매칭으로 발전시킨 것이 JDM |
| **Trajectory Distribution Matching** (arXiv:2503.06674) | 궤적 분포 매칭 | Few-step 학습 | Multi-step 지원; JDM은 one-step 특화 |
| **SANA-Sprint** (2025) | Continuous-time CD | 고속 one-step 생성 | 속도 중심; 제어 추가 메커니즘 부재 |
| **$f$-Divergence Distribution Matching** (2025) | $f$-발산 일반화 | 다양한 발산 함수 지원 | KL 발산의 일반화; JDM과 상보적 |

관련 연구로는 "One-step Diffusion Models with $f$-Divergence Distribution Matching", "SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation", "Learning Few-Step Diffusion Models by Trajectory Distribution Matching" 등이 2025년 동시기에 발표되며 one-step/few-step 생성 방법론이 활발히 연구되고 있음을 보여준다.

---

## 📌 종합 결론

JDM은 **one-step diffusion 모델의 제어 가능성 확장**이라는 실용적이고 이론적으로도 탄탄한 문제를 해결합니다. 특히:

1. **이론적 기여**: 결합 분포 KL 발산의 tractable upper bound 유도 및 fidelity-condition 분리
2. **실용적 기여**: 재증류 없이 새로운 조건 추가 가능, CFG/HFL 통합
3. **일반화 잠재력**: Teacher-unknown 조건 처리 능력과 HFL 통합으로 실제 환경 일반화 강점

향후 연구는 **다중 조건 동시 처리**, **비디오/3D 확장**, **tighter upper bound 설계**, **teacher 의존성 완화** 방향을 중심으로 발전할 것으로 전망됩니다.
