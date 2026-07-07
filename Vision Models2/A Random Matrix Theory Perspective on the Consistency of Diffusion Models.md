
# A Random Matrix Theory Perspective on the Consistency of Diffusion Models

> **저자:** Binxu Wang, Jacob Zavatone-Veth, Cengiz Pehlevan
> **발표:** ICML 2026 Oral
> **arXiv:** [2602.02908](https://arxiv.org/abs/2602.02908) (2026년 2월)
> **참고 출처:** arXiv:2602.02908, OpenReview (K8TEFs6aFn), co-r-e.com 분석 글, analytic-diffusion.github.io

---

## 1. 🔑 핵심 주장 및 주요 기여 요약

### 핵심 현상 (Observation)

서로 다른, 겹치지 않는 데이터셋 부분집합(split)으로 훈련된 Diffusion 모델들은 동일한 노이즈 시드(noise seed)가 주어졌을 때 매우 유사한 출력을 생성한다.

### 핵심 주장

이 논문은 그 일관성(consistency)을 단순한 선형 효과로 추적한다: 데이터 분할 전반에 걸친 공유된 가우시안 통계(Gaussian statistics)가 생성된 이미지의 상당 부분을 이미 예측한다. 이를 형식화하기 위해, 저자들은 유한 데이터셋이 선형 설정에서 학습된 디노이저(denoiser)와 샘플링 맵(sampling map)의 기댓값 및 분산을 어떻게 형성하는지 정량화하는 랜덤 행렬 이론(RMT) 프레임워크를 개발한다.

### 주요 기여 3가지

| # | 기여 내용 |
|---|---|
| ① | 유한 데이터 효과를 노이즈 레벨의 renormalization으로 설명하는 RMT 프레임워크 |
| ② | 분산(fluctuation)의 세 가지 핵심 요소 규명 |
| ③ | 샘플링 궤적 전체(entire trajectory) 분석을 위한 결정론적 등가 도구(deterministic equivalence) 확장 |

이 이론은 선형 Diffusion 모델의 거동을 정밀하게 예측하며, 비기억화(non-memorization) 구간에서 UNet과 DiT 아키텍처에 대한 검증을 수행하고, 훈련 데이터 분할에 따라 샘플이 어디서 어떻게 달라지는지를 규명한다. 이는 Diffusion 훈련의 재현성(reproducibility)에 대한 원칙적인 기준선을 제공하며, 데이터의 스펙트럼 속성을 생성 출력의 안정성과 연결한다.

---

## 2. 🔬 상세 분석

### 2-1. 해결하고자 하는 문제

데이터 분할 간의 일관성은 Diffusion 모델이 특정 훈련 세트에 민감하지 않은 데이터 매니폴드의 측면을 복원한다는 것을 시사한다. 이는 모델이 훈련 샘플을 어떻게 일반화하는지, 얼마나 특이한 데이터를 암기하는지, 그리고 출력이 분포의 보편적인 통계적 규칙성을 반영하는지에 대한 근본적인 질문을 제기한다.

**구체적인 문제:**
1. 왜 독립적인 데이터 분할로 훈련된 모델들이 동일한 이미지를 생성하는가?
2. 유한 데이터(finite sample)가 학습된 디노이저에 어떤 편향(bias)을 유발하는가?
3. 데이터 분할 간 불일치(disagreement)는 어디서 발생하는가?

---

### 2-2. 제안 방법 및 핵심 수식

#### (A) 가우시안 선형 Diffusion 설정

데이터 $\mathbf{x} \in \mathbb{R}^d$가 가우시안 분포 $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$를 따른다고 가정할 때, 노이즈 레벨 $\sigma$에서의 최적 선형 디노이저(Bayes-optimal denoiser)는 다음과 같다:

$$
D^*(\mathbf{y}; \sigma) = \boldsymbol{\mu} + \boldsymbol{\Sigma}(\boldsymbol{\Sigma} + \sigma^2 \mathbf{I})^{-1}(\mathbf{y} - \boldsymbol{\mu})
$$

유한 샘플 $n$개의 경험적 공분산 $\hat{\boldsymbol{\Sigma}} = \frac{1}{n}\sum_{i=1}^n (\mathbf{x}_i - \hat{\boldsymbol{\mu}})(\mathbf{x}_i - \hat{\boldsymbol{\mu}})^\top$ 으로 추정할 경우:

$$
\hat{D}(\mathbf{y}; \sigma) = \hat{\boldsymbol{\mu}} + \hat{\boldsymbol{\Sigma}}(\hat{\boldsymbol{\Sigma}} + \sigma^2 \mathbf{I})^{-1}(\mathbf{y} - \hat{\boldsymbol{\mu}})
$$

#### (B) 핵심 수식: 노이즈 레벨의 Self-consistent Renormalization

기댓값 관점에서, 샘플링 변동성(sampling variability)은 자기 일관적 관계(self-consistent relation)를 통해 노이즈 레벨의 재규격화(renormalization)로 작용한다:

$$
\sigma^2 \mapsto \kappa(\sigma^2)
$$

핵심 아이디어는 복잡한 랜덤 행렬(경험적 공분산)을 동일한 기댓값을 갖는 더 단순한 결정론적 대리물(deterministic surrogate)로 대체하는 것이다. 핵심 결과는 경험적 공분산 행렬을 디노이저 공식에 넣으면, 재규격화된 노이즈 레벨을 갖는 실제 모집단 공분산처럼 거동한다는 것이다. 즉, 유한 데이터로 훈련하는 것은 무한 데이터로 훈련하되 더 많은 노이즈가 있는 것과 동등하다.

$\kappa(\sigma^2)$는 다음의 self-consistent 방정식을 만족한다:

$$
\kappa(\sigma^2) = \sigma^2 + \frac{1}{n} \sum_{k=1}^{d} \frac{\lambda_k}{\lambda_k + \kappa(\sigma^2)}
$$

여기서 $\lambda_k$는 모집단 공분산 $\boldsymbol{\Sigma}$의 $k$번째 고유값, $n$은 데이터 샘플 수, $d$는 데이터 차원이다.

이로 인해 제한된 데이터는 저분산 방향(low-variance directions)을 과도하게 축소(overshrink)하고 샘플을 데이터셋 평균 쪽으로 끌어당긴다.

#### (C) 분산 공식: Cross-split Disagreement

분산(fluctuation)에 대해, 분산 공식은 분할 간 불일치 뒤의 세 가지 핵심 요인을 드러낸다: 고유 모드(eigenmodes) 간의 비등방성(anisotropy), 입력 간의 불균질성(inhomogeneity), 그리고 데이터셋 크기에 따른 전체 스케일링.

구체적으로, 디노이저 출력의 분산은 다음 형태로 분해된다:

$$
\text{Var}[\hat{D}(\mathbf{y}; \sigma)] \propto \frac{1}{n} \sum_{k} \left(\frac{\lambda_k}{\lambda_k + \kappa}\right)^2 \cdot v_k(\mathbf{y})
$$

여기서 $v_k(\mathbf{y})$는 입력 $\mathbf{y}$의 $k$번째 고유 방향 성분에 의존하는 불균질성 항이다.

모델 출력의 분산은 데이터 공분산의 고유값 중 재규격화된 노이즈 레벨 $\kappa$에 근접한 값을 가진 모드에서 최대화된다. 매우 큰 고유값을 갖는 모드(주요 구조적 특징)는 일관되게 추정된다.

#### (D) 샘플링 궤적 분석 (Probability Flow ODE)

결정론적 등가 도구(deterministic-equivalence tools)를 분수 행렬 거듭제곱(fractional matrix powers)으로 확장하여 전체 샘플링 궤적을 분석한다.

확률 흐름 ODE(Probability Flow ODE)에 대한 선형 해는 행렬의 분수 거듭제곱을 포함하며:

$$
\mathbf{x}(t) = \left(\boldsymbol{\Sigma} + \sigma(t)^2 \mathbf{I}\right)^{\alpha(t)} \mathbf{y}_0
$$

이를 RMT 프레임워크에서 Balakrishnan identity를 활용하여 분석한다.

---

### 2-3. 모델 구조 및 검증 아키텍처

이 이론은 선형 Diffusion 모델의 거동을 정밀하게 예측하며, **비기억화(non-memorization) 구간**에서 **UNet**과 **DiT(Diffusion Transformer)** 아키텍처에 대해 예측을 검증한다.

- **선형 모델 (이론):** Gaussian denoiser, closed-form analytic solution
- **비선형 모델 (실험):** UNet (CNN 기반), DiT (Transformer 기반)
- 요약하면, (i) 독립적인 분할로 훈련된 Diffusion 모델들은 거의 동일한 샘플링 맵으로 수렴하고, (ii) 이 속성은 아키텍처 전반에서 유지되며, (iii) 단순한 가우시안 예측기가 이미 효과의 상당 부분을 포착한다.

---

### 2-4. 성능 향상 및 한계

#### ✅ 성능 및 기여
- RMT 프레임워크를 통한 **정량적·해석적 예측** 가능
- 최근 연구는 많은 Diffusion 시간(즉, 신호 대 잡음비)에서 학습된 신경 스코어가 데이터에 맞는 가우시안의 선형 스코어로 잘 근사됨을 보였다. 이 가우시안 선형 스코어는 확률 흐름 ODE에 대한 폐쇄형 해를 허용하며, 이는 샘플링 가속화 및 품질 향상에 활용 가능하다.

#### ⚠️ 한계
1. **선형성 가정:** 이론은 가우시안 선형 설정에서 엄밀히 성립하며, 비선형 심층 신경망에 대한 직접 적용에는 간극이 있음
2. 선형 Diffusion은 심층 네트워크보다 더 일관적이며, 심층 네트워크는 고차 통계를 활용할 수 있다
3. **조건부 생성 미포함:** 대부분의 실제 Diffusion 시스템은 조건부(conditioning)를 사용하는데, 이 일관성 프레임워크가 그 설정으로 어떻게 확장되는지는 즉각적인 실용적 관련성이 있는 미해결 문제이다.
4. 논문은 65페이지, 53개의 그림으로 구성된 방대한 작업이다.

---

## 3. 🌱 모델의 일반화 성능 향상 가능성

### 3-1. 유한 데이터 편향의 교정

$\sigma^2 \mapsto \kappa(\sigma^2)$ 관계를 통해, 유한 데이터로 인한 편향(bias)을 수식적으로 정량화할 수 있다.

노이즈 재규격화 관점은 훈련 중 명시적 정규화가 단순히 더 많은 데이터를 수집하는 것과 유사한 효과를 달성할 수 있다는 것을 시사한다. $\kappa$가 데이터 제한에 따라 증가하는 유효 노이즈 레벨을 나타낸다면, 훈련 중 의도적으로 노이즈를 추가하는 것이 특정 실패 모드를 안정화하는 데 도움이 될 수 있다.

### 3-2. 일반화-기억화 전이(Transition)와의 연결

이 선형 구조는 Diffusion 모델에서의 일반화-기억화 전이(generalization–memorization transition)와도 연결되어 있다.

### 3-3. 데이터 분할 전략의 설계

논문의 반사실적(counterfactual) 실험은 일관성이 데이터의 통계적 동질성에 결정적으로 의존한다는 것을 보여준다. 데이터셋에 뚜렷한 하위 집단이 포함되어 있으면, 무작위 분할이 다른 통계를 가진 분할을 생성하여 일관성 가정을 깨뜨릴 수 있다.

### 3-4. 일반화 스펙트럼 분해

RMT 분석은 어떤 고유 방향(eigenmode)이 일반화되고 어떤 방향이 기억화되는지를 스펙트럼 기준으로 예측한다:

- $\lambda_k \gg \kappa$: **고분산 방향** → 일관되게 추정, 일반화 우세
- $\lambda_k \approx \kappa$: **중간 분산 방향** → 분할 간 불일치 최대, 불확실성 높음
- $\lambda_k \ll \kappa$: **저분산 방향** → 과도한 축소(overshrinking), 평균 쪽으로 편향

---

## 4. 🔭 앞으로의 연구에 미치는 영향 및 고려 사항

### 4-1. 연구에 미치는 영향

| 분야 | 영향 |
|------|------|
| **이론적 기반** | 데이터의 스펙트럼 속성과 생성 출력의 안정성을 연결하는 Diffusion 훈련 재현성의 원칙적 기준선 제공 |
| **샘플링 가속** | RMT 기반 closed-form 해를 통한 ODE solver 개선 가능 |
| **데이터 효율성** | 유한 샘플 편향 이해를 통한 소규모 데이터 훈련 전략 설계 |
| **아키텍처 설계** | 스펙트럼 편향을 고려한 모델 설계 지침 제공 |

### 4-2. 향후 연구 시 고려 사항

1. **비선형 확장:** 현재는 가우시안 선형 모델에 국한 → 비선형 신경망에서의 고차 통계 효과 분석 필요
2. **조건부 생성:** 텍스트-이미지, 클래스 조건부 생성에서의 일관성 분석 확장
3. **정규화 설계:** $\kappa$ 기반의 적응적 노이즈 스케줄 및 정규화 기법 개발
4. **데이터 이질성:** 일관성은 데이터 분할이 진정으로 다른 통계적 속성을 가질 때(예: 한 분할은 정면 얼굴, 다른 분할은 측면 얼굴) 붕괴된다는 점을 고려한 학습 데이터 구성 전략
5. **Flow Matching과의 통합:** Flow matching, Probability Flow ODE, 결정론적 등가 등과의 통합 이론 구축

---

## 5. 📊 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 접근법 | 주요 기여 | 본 논문과의 관계 |
|------|--------|-----------|----------------|
| **Biroli et al. (2024)** *Dynamical regimes of diffusion models* (Nature Comm.) | 통계물리학 | 생성 궤적의 임계 전이(critical transition) 식별 | RMT 분석의 기반이 되는 메모라이제이션-일반화 상전이 |
| **Li et al. (2024c)** | 이론 분석 | 선형 구조와 일반화-기억화 전이의 연결 | 가우시안 선형 근사의 이론적 근거 제공 |
| **Kadkhodaie et al. (2024)** | 기하학적 분석 | 기하학-적응적 고조파 표현으로 일반화 설명 | 비선형 일반화 메커니즘의 보완적 관점 |
| **Kamb & Ganguli (2024)** | 해석적 이론 | 합성곱 Diffusion 모델에서의 창의성 이론 | 학습된 스코어가 경험적 데이터 분포의 스코어와 정확히 일치하면 새로운 샘플을 생성하지 못함을 설명 |
| **Wang & Vastola (2024)** *Unreasonable Effectiveness of Gaussian* | 선형 근사 | 가우시안 스코어 근사의 광범위한 유효성 | 본 논문의 핵심 전제 실험적 검증 |
| **Provable Separations: Memorization vs. Generalization (2025)** | 비점근적 분석 | 메모라이제이션을 통계적 추정과 신경 함수 근사의 이중 렌즈로 이론적 설명; 메모라이제이션은 Denoising Score Matching 손실의 통계적 속성과 근본적으로 연결됨 | 유한 샘플 효과 분석의 상호 보완적 접근 |
| **Towards a Mathematical Theory for Consistency Training (2024)** arXiv:2402.07802 | 수렴 이론 | 계산 오버헤드 완화를 위한 단일 단계 샘플링을 가능하게 하는 Consistency 모델의 수학적 이론 | 본 논문의 "consistency"는 다른 의미이나 이론적 체계 참고 가능 |

---

## 📚 참고 자료 (출처)

1. **arXiv:2602.02908** - Wang, B., Zavatone-Veth, J., Pehlevan, C. "A Random Matrix Theory Perspective on the Consistency of Diffusion Models" (2026). https://arxiv.org/abs/2602.02908
2. **OpenReview (ICML 2026)** - https://openreview.net/forum?id=K8TEFs6aFn
3. **HTML 전문** - https://arxiv.org/html/2602.02908v1
4. **co-r-e.com 분석글** - "Why Do Diffusion Models Agree? A Random Matrix Theory Explanation" https://co-r-e.com/method/diffusion-model-consistency-rmt
5. **CVPR 2026 Tutorial 페이지** - Analytic Understanding of Diffusion Models https://analytic-diffusion.github.io/
6. **arXiv:2511.03202** - "Provable Separations between Memorization and Generalization in Diffusion Models" (2025)
7. **arXiv:2402.07802** - Gen Li et al., "Towards a mathematical theory for consistency training in diffusion models" (2024)
8. **arXiv:2502.00336** - "Denoising Score Matching with Random Features: Insights on Diffusion Models from Precise Learning Curves" (2025)

> ⚠️ **정확도 주의:** 본 논문의 수식 일부(특히 $\kappa$ self-consistent equation의 정확한 형태)는 논문 원문의 세부 표기를 직접 확인하지 못한 부분이 있으므로, 정확한 수식은 arXiv 원문(2602.02908)을 직접 참조하시기 바랍니다.
