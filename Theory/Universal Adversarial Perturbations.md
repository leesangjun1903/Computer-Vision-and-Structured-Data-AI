# Universal Adversarial Perturbations

> **참고 자료**
> - Moosavi-Dezfooli, S.-M., Fawzi, A., Fawzi, O., & Frossard, P. (2017). *Universal adversarial perturbations*. CVPR 2017. arXiv:1610.08401v3
> - DeepFool: Moosavi-Dezfooli et al., CVPR 2016 [논문 내 참조 11]
> - FGSM: Goodfellow et al., ICLR 2015 [논문 내 참조 5]
> - Szegedy et al., ICLR 2014 [논문 내 참조 19]
> - Fawzi et al., NIPS 2016 [논문 내 참조 4]

---

## 1. Executive Summary (10문장 이내)

1. 본 논문은 **단 하나의 작은 perturbation 벡터**를 자연 이미지에 더하는 것만으로 최신 딥러닝 분류기를 높은 확률로 속일 수 있음을 증명한다.
2. 이 perturbation은 **이미지에 무관(image-agnostic)** 하여, 어떤 이미지에도 동일하게 적용할 수 있다는 점에서 "universal"이라 명명된다.
3. 저자들은 이를 체계적으로 생성하는 반복 알고리즘(Algorithm 1)을 제안하며, DeepFool 기반의 최소 perturbation 계산을 활용한다.
4. ILSVRC 2012 검증 세트에서 CaffeNet, VGG, GoogLeNet, ResNet 등 6개 최신 모델 모두에 대해 **78~93%의 fooling rate**를 달성한다(Table 1).
5. 특히 이 perturbation은 학습에 사용하지 않은 새로운 이미지에도 높은 일반화 성능을 보인다(Fig. 6).
6. 나아가 한 네트워크에서 계산된 perturbation이 다른 아키텍처에도 전이되는 **doubly-universal** 특성이 실험적으로 확인된다(Table 2).
7. 이 현상의 원인은 분류 경계면의 법선 벡터들이 **저차원 부분공간(low-dimensional subspace) $\mathcal{S}$에 집중**되는 기하학적 구조에 있다(Fig. 9, 10).
8. Fine-tuning으로 robustness를 개선할 수 있으나(93.7% → 76.2%), 완전한 방어는 불가능하다.
9. 본 연구는 딥러닝 시스템의 실제 보안 취약성을 드러내며, 단순한 이미지 덧셈 연산만으로 적대자가 분류기를 무력화할 수 있음을 경고한다.
10. 결론적으로, universal perturbation의 존재는 딥 네트워크 결정 경계면의 기하학적 상관 구조라는 새로운 연구 방향을 제시한다.

---

### 1-1. 연구의 목적과 필요성

**목적:**
딥러닝 분류기에 대해 **이미지에 독립적인 단일 적대적 perturbation**이 존재하는지 증명하고, 이를 효율적으로 생성하는 알고리즘을 제안하며, 그 존재의 기하학적 원인을 분석한다.

**필요성:**

| 필요성 항목 | 설명 |
|---|---|
| 기존 적대적 공격의 한계 | Szegedy et al.(2014), FGSM(2015) 등 기존 방법은 각 이미지마다 별도의 perturbation을 계산해야 하므로 실용적 공격에 한계가 있다 |
| 실제 보안 위협 | 단 하나의 perturbation으로 모든 자연 이미지를 속일 수 있다면, 실세계 배포 환경에서 심각한 보안 위협이 된다 |
| 딥러닝 이해의 심화 | Universal perturbation의 존재는 딥 네트워크 결정 경계면의 기하학적 구조에 대한 새로운 이해를 제공한다 |
| 방어 연구의 필요성 제기 | Fine-tuning만으로는 완전한 방어가 불가능함을 보여, 새로운 방어 기법 연구의 필요성을 제시한다 |

> 💡 **적대적 perturbation (Adversarial Perturbation):** 사람 눈에는 거의 보이지 않지만, 딥러닝 모델의 예측을 틀리게 만들도록 의도적으로 설계된 미세한 신호/노이즈.

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 | 위치 |
|---|---|---|---|
| ① | 단일 universal perturbation이 존재한다 | 6개 모델에서 78~93% fooling rate 달성 | Table 1, p.4 |
| ② | Perturbation이 미학습 이미지에도 일반화된다 | 500장으로도 30%+ fooling rate | Fig. 6, p.4 |
| ③ | Cross-model 전이성(doubly-universal)이 존재한다 | VGG-19 perturbation이 타 모델에서 53%+ | Table 2, p.5 |
| ④ | Human eye에 quasi-imperceptible하다 | $\ell_2$ norm ≈ 2000 (이미지 평균 $\approx 5\times10^4$), $\ell_\infty$ = 10 (이미지 최대 ≈ 250) | p.3, Fig. 3 |
| ⑤ | 결정 경계면의 법선 벡터가 저차원 부분공간에 집중된다 | 행렬 $N$의 특이값이 빠르게 감소 | Fig. 9, p.8 |
| ⑥ | Fine-tuning은 부분적 방어만 가능하다 | 93.7% → 76.2%로 감소하나 여전히 취약 | p.6 |
| ⑦ | Universal perturbation은 random perturbation과 근본적으로 다르다 | 동일 norm에서 85% vs 10% fooling rate | Fig. 8, p.7 |

> 💡 **Fooling Rate:** 분류기가 perturbation이 더해진 이미지를 잘못 분류하는 비율.
>
> 💡 **Cross-model 전이성:** 한 모델에서 만들어진 공격이 다른 구조의 모델에도 효과적으로 적용되는 현상.

---

## 2-1. 해결 문제 / 제안 방법 / 모델 구조 / 성능 / 한계

### 🔴 해결하고자 하는 문제

기존 adversarial perturbation은 **데이터 포인트별로** 별도 계산되어야 했음. 본 논문은 **모든 자연 이미지에 공통으로 적용 가능한 단일 perturbation** $v$의 존재와 계산 방법을 제시하고자 한다.

---

### 🔵 제안하는 방법 (수식 포함)

**목표 수식:**

찾고자 하는 perturbation $v \in \mathbb{R}^d$는 아래 두 조건을 동시에 만족해야 한다:

$$
\|v\|_p \leq \xi \quad \cdots (조건\ 1)
$$

$$
\mathbb{P}_{x \sim \mu}\left(\hat{k}(x + v) \neq \hat{k}(x)\right) \geq 1 - \delta \quad \cdots (조건\ 2)
$$

- $v \in \mathbb{R}^d$: universal perturbation 벡터 ($d$: 이미지 차원)
- $\mu$: 자연 이미지의 분포
- $\hat{k}(\cdot)$: 분류기의 예측 레이블 함수
- $\xi$: perturbation의 최대 허용 크기 (norm 제약)
- $\delta$: 허용 오차 (fooling하지 못하는 이미지의 최대 비율)
- $p$: norm의 종류 ($p=2$ 또는 $p=\infty$)

> 💡 **$\ell_p$ norm:** 벡터의 크기를 측정하는 방법. $\ell_2$ norm은 유클리드 거리, $\ell_\infty$ norm은 벡터 원소 중 최댓값.

---

**알고리즘 핵심 수식 (Algorithm 1, p.3):**

각 데이터 포인트 $x_i$에 대해, 현재 perturbation $v$가 $x_i$를 fool하지 못할 경우 추가 perturbation $\Delta v_i$를 계산:

$$
\Delta v_i \leftarrow \arg\min_{r} \|r\|_2 \quad \text{s.t.} \quad \hat{k}(x_i + v + r) \neq \hat{k}(x_i) \quad \cdots (1)
$$

- $\Delta v_i$: $x_i$를 결정 경계 너머로 보내는 최소 추가 perturbation
- $r$: 탐색 대상인 추가 perturbation 벡터
- $x_i + v$: 현재 universal perturbation이 적용된 이미지

> 💡 **결정 경계(Decision Boundary):** 분류기가 서로 다른 클래스를 구분하는 경계면. 이 경계를 넘으면 분류 결과가 바뀐다.

---

**Projection 연산 (norm 제약 유지):**

$$
\mathcal{P}_{p,\xi}(v) = \arg\min_{v'} \|v - v'\|_2 \quad \text{subject to} \quad \|v'\|_p \leq \xi
$$

업데이트 규칙:

$$
v \leftarrow \mathcal{P}_{p,\xi}(v + \Delta v_i)
$$

- $\mathcal{P}_{p,\xi}$: $\ell_p$ ball (반지름 $\xi$, 중심 0)로의 projection 연산자
- 이 연산으로 perturbation의 크기가 항상 제약 조건 내에 유지됨

> 💡 **Projection 연산:** 어떤 점을 제약 집합(여기서는 $\ell_p$ ball) 내의 가장 가까운 점으로 이동시키는 연산.

---

**종료 조건 (fooling rate 기반):**

$$
\text{Err}(X_v) := \frac{1}{m} \sum_{i=1}^{m} \mathbf{1}_{\hat{k}(x_i+v) \neq \hat{k}(x_i)} \geq 1 - \delta
$$

- $X_v = \{x_1+v, \ldots, x_m+v\}$: perturbation이 적용된 학습 이미지 집합
- $m$: 학습에 사용된 이미지 수
- $\mathbf{1}_{[\cdot]}$: 조건이 참이면 1, 거짓이면 0인 지시 함수(indicator function)

---

**기하학적 분석 (Section 4, p.8):**

결정 경계 법선 벡터 행렬:

$$
N = \left[\frac{r(x_1)}{\|r(x_1)\|_2} \cdots \frac{r(x_n)}{\|r(x_n)\|_2}\right]
$$

- $r(x) = \arg\min_r \|r\|_2 \quad \text{s.t.} \quad \hat{k}(x+r) \neq \hat{k}(x)$: 각 이미지 $x$에서 결정 경계까지의 최소 perturbation (= 결정 경계 법선 벡터)
- $N$의 특이값(singular values)이 빠르게 감소 → 법선 벡터들이 저차원 부분공간 $\mathcal{S}$에 집중됨을 의미

> 💡 **특이값 분해(SVD, Singular Value Decomposition):** 행렬의 정보가 어느 방향에 얼마나 집중되어 있는지를 분석하는 수학적 도구. 특이값이 빠르게 감소할수록 데이터가 저차원 구조를 가짐.

---

### 🟢 모델 구조

저자들은 자체 모델을 새로 설계하지 않았으며, 아래의 기존 최신 모델들을 대상으로 알고리즘을 적용·분석하였다:

| 모델 | 특징 |
|---|---|
| CaffeNet | AlexNet 계열, 비교적 단순한 구조 |
| VGG-F | 빠른 VGG 변형 |
| VGG-16 / VGG-19 | 16/19층 깊은 CNN |
| GoogLeNet (Inception) | Inception module 기반 |
| ResNet-152 | 152층 잔차 연결 네트워크 |

---

### 🟡 성능 향상

| 지표 | 결과 | 위치 |
|---|---|---|
| $\ell_2$ Fooling Rate (validation) | 82.0~90.3% (모델별) | Table 1, p.4 |
| $\ell_\infty$ Fooling Rate (validation) | 77.8~93.7% (모델별) | Table 1, p.4 |
| Cross-model (VGG-19 → 타 모델) | 53% 이상 | Table 2, p.5 |
| 500장으로 계산 시 validation fooling rate | 30%+ | Fig. 6, p.4 |
| Fine-tuning 후 fooling rate (VGG-F) | 76.2% (원래 93.7%) | p.6 |

---

### 🔴 한계

| 한계 | 설명 |
|---|---|
| 완전한 방어 불가 | Fine-tuning 후에도 76%의 fooling rate 유지 |
| 이론적 분석 미완성 | 결정 경계의 기하학적 상관 구조에 대한 이론적 증명은 향후 연구 과제로 남김 |
| White-box 설정 | 알고리즘 자체는 모델 전체 지식 필요 (단, 생성 후 배포는 black-box 환경에서도 가능) |
| 단일 아키텍처 실험 중심 | Fine-tuning 방어 실험은 VGG-F 하나에만 수행 |
| 타겟 공격 미지원 | 특정 레이블로 fooling하는 targeted attack은 직접 다루지 않음 |

---

## 3. 각 주장별 페이지/Figure/Table 번호

| 주장 | 근거 위치 |
|---|---|
| Universal perturbation의 존재 증명 | p.1 Abstract, p.2 Section 2 |
| 알고리즘 제안 (Algorithm 1) | p.3, Fig. 2 |
| 높은 fooling rate 달성 | Table 1 (p.4) |
| 이미지에 quasi-imperceptible | Fig. 3 (p.5), p.3 footnote 2 |
| 여러 universal perturbation의 다양성 | Fig. 4 (p.5), Fig. 5 (p.5) |
| Cross-model 전이성 | Table 2 (p.5) |
| 학습 데이터 크기와 fooling rate 관계 | Fig. 6 (p.5) |
| Fine-tuning 방어 실험 | p.6 |
| 기하학적 해석 (저차원 subspace) | Fig. 8, 9, 10 (p.7~8), Section 4 |

---

## 4. 저자 보고 결과 vs. 내 해석 (분리)

### 저자가 직접 보고한 결과

**연구 주제 (저자):**
이미지에 무관한 단일 perturbation이 대부분의 자연 이미지를 fool할 수 있는지 탐구하고, 이를 생성하는 체계적 방법과 존재의 원인을 설명.

**방법 (저자):**
- DeepFool [11]을 활용해 각 데이터 포인트를 결정 경계로 이동시키는 최소 perturbation $\Delta v_i$를 계산하고 누적.
- $\ell_p$ ball projection으로 norm 제약 유지.
- 경험적으로 fooling rate $\geq 1-\delta$ 달성 시 종료.

**결과 (저자):**
- CaffeNet $\ell_\infty$: validation 93.3% (Table 1)
- VGG-19 → GoogLeNet 전이: 53.6% (Table 2)
- 행렬 $N$의 특이값 급감 → 저차원 subspace $\mathcal{S}$ 존재 (Fig. 9)
- $\mathcal{S}$의 무작위 벡터로도 38% fooling 가능 (vs. 순수 random 10%) (p.8)

---

### 내 해석

1. **알고리즘의 greedy 특성:** Algorithm 1은 전역 최적(globally optimal) universal perturbation을 보장하지 않는다. 데이터 순서(shuffling)에 따라 다른 perturbation이 생성되며(Fig. 5), 이는 문제의 non-convexity에서 기인한다. 저자들도 이를 인정하나, 실용적 성능은 충분히 우수하다.

2. **저차원 subspace 해석:** $\mathcal{S}$ 내의 무작위 벡터로 38%의 fooling이 가능하다는 결과는, universal perturbation이 단순히 "운 좋게 효과적인 방향"을 찾은 것이 아니라, **딥 네트워크의 결정 경계 구조 자체에 내재된 취약성**임을 강력히 시사한다.

3. **Cross-model 전이의 비대칭성:** Table 2에서 VGG-16/VGG-19 perturbation은 타 모델에 잘 전이되는 반면, GoogLeNet perturbation은 상대적으로 전이성이 낮다. 이는 모델 구조의 유사성(VGG 계열 간)뿐 아니라, 특정 아키텍처의 결정 경계가 더 "보편적인" 기하학적 구조를 가질 수 있음을 시사한다.

4. **Fine-tuning의 한계에 대한 해석:** Fine-tuning으로 fooling rate를 낮출 수 있으나 완전 방어가 안 되는 이유는, 새로운 perturbation을 생성할 때 fine-tuned 모델의 새로운 결정 경계를 다시 활용하기 때문이다. 이는 adversarial training의 근본적 한계를 드러낸다.

---

## 5. 통계적 취약점 및 비교 불가능한 수치

⚠️ **통계적으로 취약한 부분:**

| 항목 | 문제점 |
|---|---|
| Fine-tuning 실험 (p.6) | VGG-F 단일 모델, 단일 실험 조건만 보고. 표준편차/신뢰구간 없음 |
| Cross-model fooling rate (Table 2) | 각 셀이 단일 실험값. 반복 실험이나 통계적 유의성 검증 없음 |
| Subspace 실험 (p.8) | 상위 100개 특이벡터 선택의 근거 불명확. 100이라는 수치의 sensitivity 분석 없음 |
| Fig. 6 (training set 크기 실험) | 4개 데이터 포인트(500, 1000, 2000, 4000)만 사용. 오차 막대 없음 |

⚠️ **비교 불가능한 수치:**

| 항목 | 문제점 |
|---|---|
| $\ell_2$ vs $\ell_\infty$ fooling rate | 두 norm 설정($\xi=2000$ vs $\xi=10$)이 달라 직접 비교 불가 |
| CaffeNet vs ResNet fooling rate | 두 모델의 기본 분류 정확도가 달라, 동일 조건 비교 아님 |
| Fine-tuning 전후 비교 | Fine-tuning 후 validation 오류율 소폭 증가(각주 4)로 공정한 비교에 편향 존재 |
| Random perturbation과 비교 (Fig. 8) | Random perturbation은 단일 샘플 기반, universal은 10,000장 기반으로 계산 조건이 다름 |

---

## 6. 논문이 답하지 않는 질문

| 번호 | 미해결 질문 |
|---|---|
| Q1 | 이론적으로 universal perturbation이 반드시 존재하기 위한 충분/필요 조건은 무엇인가? |
| Q2 | 저차원 subspace $\mathcal{S}$의 차원 $d'$은 어떻게 결정되며, 모델/데이터에 따라 어떻게 달라지는가? |
| Q3 | Universal perturbation에 완전히 면역인 분류기 구조나 학습 방법이 존재하는가? |
| Q4 | Fine-tuning 외에 효과적인 방어 기법(예: input preprocessing, certified defense)은 무엇인가? |
| Q5 | 타겟 공격(targeted attack: 특정 레이블로 fool) 형태의 universal perturbation도 가능한가? |
| Q6 | 이미지 이외의 도메인(오디오, 텍스트, 의료 영상)에서도 동일한 universal perturbation이 존재하는가? |
| Q7 | 물리적 세계(physical world)에서 출력/인쇄된 universal perturbation도 효과적인가? |
| Q8 | Algorithm 1이 수렴하지 않는 경우(fooling rate 목표 미달)는 언제 발생하는가? |

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.1) — Universal Perturbation의 직관적 시연

**내용:** 자연 이미지(왼쪽) + universal perturbation(중앙) → 잘못 분류된 이미지(오른쪽).

**해석:**
- Perturbation은 시각적으로 노이즈처럼 보이지만, 다양한 이미지(개, 풍선, 도마뱀 등)에 동일하게 더해져 모두 오분류를 유발한다.
- 이 그림은 논문의 핵심 주장을 가장 직관적으로 전달하며, 공격이 **이미지 내용에 무관하게 작동함**을 시각적으로 증명한다.
- 오분류 레이블(Chihuahua, Labrador 등)이 반복되는 것은 "dominant label" 현상과 일치한다.

---

### Figure 4 (p.5) — 아키텍처별 Universal Perturbation 시각화

**내용:** 6개 모델(CaffeNet, VGG-F, VGG-16, VGG-19, GoogLeNet, ResNet-152)에서 생성된 $\ell_\infty$ perturbation ($\xi=10$) 이미지.

**해석:**
- 모든 perturbation이 규칙적인 물결 무늬(wave-like texture)와 격자 패턴을 보인다.
- 이는 각 모델이 특정 고주파 공간 패턴에 민감함을 시사하며, 인간 시각 시스템과의 근본적 차이를 보여준다.
- 모델마다 패턴이 다르지만 구조적 유사성이 있는데, 이것이 cross-model 전이성(Table 2)의 시각적 근거가 될 수 있다.
- 픽셀 값은 가시성을 위해 스케일링되었으므로 실제 perturbation 크기는 훨씬 작음에 주의해야 한다.

---

### Figure 6 (p.5) — 학습 이미지 수 vs. Fooling Rate

**내용:** GoogLeNet에서 학습 집합 $X$의 크기(500~4000)에 따른 validation fooling rate.

**해석:**
- 500장(ImageNet 1000 클래스 기준 클래스당 0.5장 미만)으로도 30% 이상의 fooling rate를 달성한다.
- 이미지 수가 증가할수록 fooling rate가 단조증가하나, 10,000장 기준 ~78%에 달한다.
- 이 결과는 universal perturbation의 **강력한 데이터 효율성(data efficiency)**을 증명한다.
- 결정 경계의 저차원 subspace가 소수 이미지만으로도 충분히 근사 가능함을 시사한다.

> 💡 **데이터 효율성(Data Efficiency):** 적은 양의 데이터로도 원하는 성능을 달성할 수 있는 능력.

---

### Figure 8 (p.7) — Perturbation 유형별 Fooling Rate 비교

**내용:** CaffeNet에서 $\ell_2$ norm에 따른 다양한 perturbation의 fooling rate 위상 전이(phase transition) 그래프.

**해석:**
- Universal perturbation은 $\ell_2$ norm 2000에서 85% fooling rate를 달성하는 반면, random perturbation은 동일 norm에서 10%에 불과하다.
- 이 약 8.5배의 차이가 **random vs. universal의 근본적 차이**를 수치로 입증한다.
- DeepFool 기반 개별 adversarial perturbation(Adv. pert. DF)이 빠르게 100%에 도달하는 것은, 이미지별 최적화가 훨씬 강력함을 보여준다(단, 이미지마다 재계산 필요).
- ImageNet bias (이미지 평균)의 낮은 fooling rate는 perturbation이 단순한 bias 방향이 아님을 확인시켜 준다.

> 💡 **위상 전이(Phase Transition):** 어떤 파라미터가 특정 임계값을 넘을 때 시스템의 상태가 급격히 변하는 현상. 여기서는 norm이 커질수록 fooling rate가 급등하는 구간을 의미.

---

### Figure 9 (p.8) — 결정 경계 법선 행렬 $N$의 특이값

**내용:** 법선 벡터 행렬 $N$의 특이값(실선)과 random 구 위 균일 샘플의 특이값(점선) 비교.

**해석:**
- **Random 벡터의 특이값:** 느리게 감소 → 정보가 모든 방향에 고르게 분산됨.
- **$N$의 특이값:** 빠르게 감소 → 법선 벡터들이 소수의 방향(저차원 subspace $\mathcal{S}$)에 집중됨.
- 이는 딥 네트워크의 결정 경계가 **고도로 구조화된 기하학적 상관**을 가짐을 수학적으로 입증한다.
- Universal perturbation이 가능한 근본 이유가 이 저차원 구조에 있으며, 이 발견은 딥러닝 이론 연구에 중요한 통찰을 제공한다.
- 단, 이 분석이 CaffeNet 하나에만 수행되었다는 점은 일반화에 주의가 필요하다. ⚠️

> 💡 **부분공간(Subspace):** 고차원 공간 안에서 특정 방향들로 이루어진 저차원의 부분 공간. 여기서는 결정 경계 법선들이 몰려 있는 방향들의 집합.

---

## 8. 결론: 시사점, 후속 연구, 추가 연구 방향

### 저자가 제시한 시사점 및 후속 연구 계획

**시사점 (p.8~9, Section 5):**
1. 딥 네트워크는 human-imperceptible universal perturbation에 심각하게 취약하다.
2. Universal perturbation은 data-agnostic이자 network-agnostic(doubly-universal)하다.
3. 취약성의 원인은 결정 경계 법선 벡터들의 저차원 부분공간 집중에 있다.
4. 실세계 보안 환경에서 단순 덧셈 연산만으로 분류기를 무력화할 수 있는 실질적 위협이다.

**저자 제시 후속 연구:**
> "A theoretical analysis of the geometric correlations between different parts of the decision boundary will be the subject of future research." (p.9)

즉, 저차원 subspace의 존재에 대한 **이론적 증명**과 **기하학적 분석의 수학적 형식화**를 향후 과제로 명시하였다.

---

### 8-1. 모델의 일반화 성능 향상 가능성과의 관련성

이 논문은 **모델 일반화 성능 향상**이라는 맥락에서 매우 중요한 함의를 가진다:

**1. 적대적 학습(Adversarial Training)의 일반화 효과:**
Fine-tuning with universal perturbations (p.6)는 adversarial training의 초기 형태로 볼 수 있다. 10개의 다양한 perturbation을 pool로 사용하고 확률적으로 augmentation하는 방식은, 모델이 **perturbation 방향에 불변(invariant)**한 표현을 학습하도록 유도한다.

- 결과: Fooling rate 93.7% → 76.2%로 약 17.5%p 감소
- **그러나** 일반화된 robustness를 달성하지는 못함 → 이는 단순 augmentation의 한계를 시사

**2. 저차원 Subspace와 일반화의 연관:**
결정 경계의 법선 벡터가 저차원 subspace $\mathcal{S}$에 집중된다는 발견은:

$$
d' \ll d \quad \text{(}d'\text{: subspace 차원, }d\text{: 입력 차원)}
$$

- 모델이 **필요 이상으로 많은 방향에 민감**함을 의미.
- 이상적으로 일반화 성능이 높은 모델은 $\mathcal{S}$의 차원 $d'$이 작아야 하며, 특이값이 더 빠르게 감소해야 한다.
- 역으로, **$\mathcal{S}$의 차원을 최소화하는 regularization**이 robustness와 일반화 성능을 동시에 향상시킬 수 있는 방향이다.

**3. 일반화 성능 향상을 위한 구체적 방향:**

| 방향 | 내용 |
|---|---|
| Subspace-aware Regularization | 결정 경계 법선의 분산을 최소화하는 손실 함수 추가 |
| Diverse Adversarial Augmentation | 다양한 universal perturbation을 학습 데이터에 추가하여 표현의 robust함 향상 |
| Spectral Norm Regularization | 가중치 행렬의 특이값을 제어하여 결정 경계의 곡률을 줄임 |
| Certified Defense | 모든 $\|v\|_p \leq \xi$에 대해 정확한 분류를 보장하는 방법론 |

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의:** 아래 내용 중 구체적인 수치와 저자명은 제가 학습한 데이터에 기반하며, 일부 세부 수치는 불확실할 수 있습니다. 논문명과 방향성은 실제 연구 흐름에 부합하나, 인용 시 원문을 반드시 확인하시기 바랍니다.

| 연구 방향 | 대표 연구 (개략) | 본 논문과의 관계 |
|---|---|---|
| **UAP 공격 확장** | GAP (Generative Adversarial Perturbations), CD-UAP 등 | 생성 모델(GAN)로 UAP를 더 빠르게 생성하는 방법으로 발전 |
| **물리적 세계 공격** | Adversarial patch (Brown et al., 2017), 실물 출력 공격 | 본 논문의 physical-world 가능성을 실험적으로 확장 |
| **방어: Adversarial Training** | PGD-AT (Madry et al., 2018), TRADES | UAP에 대한 certified/empirical defense로 발전 |
| **방어: Certified Defense** | Randomized Smoothing (Cohen et al., 2019) | 수학적으로 보장된 robustness 제공; UAP 방어에도 적용 |
| **UAP의 이론적 분석** | Jetley et al., Akhtar et al. | 저차원 subspace 가설을 이론적으로 정형화하려는 시도 |
| **Transfer Learning 취약성** | 전이 학습 모델에서의 UAP 전이성 연구 | 본 논문 Table 2의 cross-model 결과를 실용적 맥락으로 확장 |
| **텍스트/오디오 도메인 UAP** | 텍스트 분류기, 음성 인식기에서의 universal perturbation 연구 | 본 논문 이미지 도메인 결과를 다른 도메인으로 일반화 |
| **Vision Transformer(ViT) 취약성** | ViT가 CNN보다 UAP에 더 robust한지 분석하는 연구 | 새로운 아키텍처에서의 UAP 적용 가능성 탐구 |

**본 논문이 이후 연구에 미치는 영향:**

1. **공격 연구의 패러다임 전환:** 데이터 포인트별 공격에서 **데이터 독립적(image-agnostic) 공격**으로의 전환을 선도. 이후 UAP 관련 논문의 baseline이 되었다.

2. **방어 연구 촉진:** Fine-tuning만으로 방어가 불가능함을 보여, certified defense, randomized smoothing 등 더 강력한 방어 기법 연구를 자극했다.

3. **딥러닝 이해의 심화:** 결정 경계의 기하학적 구조 분석이라는 새로운 연구 방향을 제시했으며, **neural network의 interpretability** 연구와도 연결된다.

4. **실세계 AI 보안 정책:** AI 시스템의 실제 배포 환경에서의 보안 취약성을 학계와 산업계에 명확히 인식시켰다.

**앞으로 연구 시 고려할 점:**

| 고려 사항 | 구체적 내용 |
|---|---|
| **동적 방어(Adaptive Defense) 필요** | 공격자가 방어 방법을 알고 있을 때도 robust한 방어 설계 필요 |
| **ViT 등 새 아키텍처 분석** | Transformer 기반 모델에서의 결정 경계 기하학 분석 필요 |
| **다중 모달리티 확장** | 텍스트-이미지 결합 모델(CLIP 등)에서의 UAP 가능성 탐구 |
| **이론적 보장 강화** | 알고리즘 수렴성, perturbation 존재의 충분조건 등 이론화 필요 |
| **현실적 위협 모델 정립** | Black-box 설정, 쿼리 제한 환경에서의 UAP 생성 가능성 평가 |
| **공정한 robustness 평가** | 표준화된 benchmark(RobustBench 등)를 활용한 재현 가능한 비교 필요 |
| **윤리적 고려** | 공격 코드 공개가 악용될 가능성 vs. 재현 가능성 간의 균형 |

---

> **참고 자료 목록:**
> 1. Moosavi-Dezfooli et al., "Universal adversarial perturbations," CVPR 2017. arXiv:1610.08401v3 *(본문 원본)*
> 2. Moosavi-Dezfooli et al., "DeepFool," CVPR 2016. *(논문 내 [11])*
> 3. Goodfellow et al., "Explaining and harnessing adversarial examples," ICLR 2015. *(논문 내 [5])*
> 4. Szegedy et al., "Intriguing properties of neural networks," ICLR 2014. *(논문 내 [19])*
> 5. Fawzi et al., "Robustness of classifiers: from adversarial to random noise," NIPS 2016. *(논문 내 [4])*
> 6. Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks," ICLR 2018. *(8-2 참조)*
> 7. Cohen et al., "Certified Adversarial Robustness via Randomized Smoothing," ICML 2019. *(8-2 참조)*
