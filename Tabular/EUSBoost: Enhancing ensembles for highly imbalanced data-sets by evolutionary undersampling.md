# EUSBoost: Enhancing ensembles for highly imbalanced data-sets
by evolutionary undersampling

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

EUSBoost는 **고도로 불균형한 이진 분류 문제**에서 랜덤 언더샘플링 대신 **진화적 언더샘플링(Evolutionary Undersampling, EUS)**을 Boosting 프레임워크에 통합함으로써, 기존 앙상블 기법들보다 우수한 분류 성능을 달성할 수 있다는 것을 실증적으로 증명합니다.

### 1.2 주요 기여

| 기여 항목 | 내용 |
|----------|------|
| **새로운 앙상블 알고리즘** | EUS를 AdaBoost.M2에 내장한 EUSBoost 제안 |
| **다양성 촉진 메커니즘** | Q-통계량 기반 피트니스 함수 수정으로 기저 분류기 다양성 향상 |
| **Kappa-AUC 오류 다이어그램** | 불균형 데이터셋 시나리오에 맞게 kappa-error diagram을 AUC 기반으로 재설계 |
| **통계적 검증** | 33개 실제 데이터셋 기반 비모수 통계 검정으로 우수성 입증 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

**클래스 불균형(Class Imbalance)** 문제는 두 클래스 간 인스턴스 수의 차이가 극단적으로 큰 경우 발생합니다.

- **불균형 비율(Imbalance Ratio, IR)** 정의:

$$IR = \frac{n^-}{n^+}$$

여기서 $n^-$는 다수 클래스 인스턴스 수, $n^+$는 소수 클래스 인스턴스 수입니다. 본 논문은 $IR > 9$인 **고도 불균형** 데이터셋에 집중합니다.

**주요 문제점:**
- 표준 분류기는 정확도(accuracy)를 최대화하도록 설계되어 다수 클래스에 편향
- 소수 클래스가 노이즈로 취급되어 무시됨
- 소표본 크기(small sample size), 클래스 중첩(overlapping), 소규모 분리(small disjuncts) 발생
- RUSBoost와 같은 랜덤 기반 방법은 유용한 다수 클래스 인스턴스를 무작위로 제거할 가능성 존재

### 2.2 성능 평가 지표

ROC 곡선 기반 AUC를 주요 평가 지표로 사용합니다:

$$AUC = \frac{1 + TPrate - FPrate}{2}$$

여기서:
- $TPrate = \frac{TP}{TP + FN}$ (민감도, Sensitivity)
- $FPrate = \frac{FP}{FP + TN}$ (1 - 특이도)

### 2.3 제안하는 방법 (수식 포함)

#### 2.3.1 진화적 언더샘플링(EUS)

**염색체 표현:** EUS에서 각 염색체는 다수 클래스 인스턴스의 포함 여부를 나타내는 이진 벡터:

$$V = (v_{x_1}, v_{x_2}, v_{x_3}, v_{x_4}, \ldots, v_{x_{n^-}})$$

여기서 $v_{x_i} \in \{0, 1\}$이며, $n^-$는 다수 클래스 인스턴스 수입니다. **소수 클래스 인스턴스는 항상 포함됩니다.**

**기하평균(Geometric Mean):**

$$GM = \sqrt{TPrate \cdot TNrate}$$

**EUS 피트니스 함수 (원본):**

$$\text{fitness}_{EUS} = \begin{cases} GM - \left|1 - \frac{n^+}{N^-} \cdot P\right| & \text{if } N^- > 0 \\ GM - P & \text{if } N^- = 0 \end{cases}$$

여기서 $n^+$는 소수 클래스 수, $N^-$는 선택된 다수 클래스 수, $P = 0.2$는 클래스 균형 페널티 인수입니다.

**CHC 알고리즘** 기반 진화:
- 이질적 균일 교차(Heterogeneous Uniform Crossover, HUX) 사용
- 두 염색체의 해밍 거리가 임계값($L/4$) 초과 시에만 재조합
- 돌연변이 없이 최우수 염색체를 템플릿으로 재초기화(최우수 유전자의 35% 무작위 변경)

#### 2.3.2 다양성 촉진 메커니즘

**Q-통계량 계산:**

$$Q_{i,j} = \frac{N^{11}N^{00} - N^{01}N^{10}}{N^{11}N^{00} + N^{01}N^{10}}$$

여기서 $N^{ab}$는 첫 번째 벡터에서 값이 $a$이고 두 번째 벡터에서 값이 $b$인 인스턴스 수입니다. Q값은 $[-1, 1]$ 범위이며, **낮을수록 다양성이 높음**을 의미합니다.

**전역 다양성 Q 계산 (최대값 사용):**

$$Q = \max_{i=1,\ldots,t} Q_{i,j}$$

여기서 $V_j$는 후보 해, $V_i (i=1,\ldots,t)$는 이전 반복에서 사용된 모든 해입니다.

**수정된 피트니스 함수 (Q-통계량 기반):**

$$\text{fitness}_{EUS_Q} = \text{fitness}_{EUS} \cdot \frac{1.0}{\beta} - \frac{10.0}{IR} - Q \cdot \beta$$

**반복에 따른 가중치 인수 $\beta$:**

$$\beta = \frac{N - t - 1}{N}$$

여기서 $N$은 앙상블 반복 횟수(실험에서 10), $t$는 현재 반복입니다.

**해밍 거리 기반 피트니스 함수 (대안):**

$$\text{fitness}_{EUS_H} = \text{fitness}_{EUS} \cdot \frac{1.0}{\beta} - \frac{10.0}{IR} + H \cdot \beta$$

여기서 $H$는 후보 염색체와 기존 모든 염색체 간의 최소 해밍 거리(코드워드 길이로 정규화)입니다.

> **Remark:** $\beta$ 인수의 역할
> - 초기 반복: 다양성($Q$)에 높은 가중치 → 다양한 언더샘플된 부분집합 선호
> - 후기 반복: 정확도($\text{fitness}_{EUS}$)에 높은 가중치 → Boosting 철학과 일치 (어려운 인스턴스 집중)
> - $IR$이 클수록: 다양성 영향력 증가 (더 많은 다수 클래스 인스턴스 제거 가능)

#### 2.3.3 EUSBoost 알고리즘 (Algorithm 1)

**입력:** 훈련 세트 $S = \{x_i, y_i\}_{i=1}^{N}$, $y_i \in \{c_1, c_2\}$, 반복 횟수 $T$, 약한 학습기 $I$

**출력:** Boosted 분류기:
$$H(x) = \arg\max_{y \in C} \sum_{t=1}^{T} \ln\left(\frac{1}{\beta_t}\right) h_t(x, y)$$

**핵심 알고리즘 단계:**
1. $D_1(i) \leftarrow 1/N$ for $i = 1, \ldots, N$
2. $w_{i,y}^1 \leftarrow D_1(i)$ for $i = 1, \ldots, N$, $y \neq y_i$
3. **for** $t = 1$ to $T$ **do**
4. $W_i^t \leftarrow \sum_{y \neq y_i} w_{i,y}^t$
5. $q_t(i, y) \leftarrow \frac{w_{i,y}^t}{W_i^t}$ for $y \neq y_i$
6. $D_t(i) \leftarrow \frac{W_i^t}{\sum_{i=1}^{N} W_i^t}$
7. $S' = \text{EvolutionaryUndersampling}(S)$ ← **[EUSBoost 핵심 단계]**

8. 
```math
D'_t(k) \leftarrow \begin{cases} \frac{W_i^t}{\sum_{x_i \in S'} W_i^t} & \text{if } x_i \in S' \\ 0 & \text{otherwise} \end{cases}
```

9. $h_t \leftarrow I(S, D'_t)$
10. $\epsilon_t \leftarrow \frac{1}{2} \sum_{i=1}^{N} D_t(i) \left(1 - h_t(x_i, y_i) + \sum_{i, y \neq y_i} q_t(i, y) h_t(x_i, y)\right)$
11. $\beta_t = \frac{\epsilon_t}{1 - \epsilon_t}$
12. $w_{i,y}^{t+1} = w_{i,y}^t \cdot \beta_t^{(1/2)(1 + h_t(x_i, y_i) - h_t(x_i, y))}$
13. **end for**

### 2.4 모델 구조

```
[EUSBoost 구조]
┌─────────────────────────────────────────────────────────────┐
│                    EUSBoost 프레임워크                         │
│                                                               │
│  훈련 데이터 S ──→ [AdaBoost.M2 루프 (T회 반복)]              │
│                         │                                     │
│                    ┌────▼────────────────────┐                │
│                    │  EUS (CHC 기반 진화)     │                │
│                    │  ┌─────────────────────┐ │               │
│                    │  │ 피트니스 함수:        │ │               │
│                    │  │ fitness_EUS_Q =      │ │               │
│                    │  │ accuracy + diversity │ │               │
│                    │  └─────────────────────┘ │               │
│                    └────────────┬────────────┘               │
│                                 │                             │
│              언더샘플된 S' ─────▼                             │
│              가중치 재정규화 → 기저 분류기 훈련(C4.5)          │
│                                 │                             │
│              ┌──────────────────▼──────────────────┐          │
│              │    앙상블 결합 (가중치 다수결 투표)    │          │
│              └─────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### 2.5 실험 설정

| 항목 | 설정 |
|------|------|
| 기저 분류기 | C4.5 결정 트리 |
| 데이터셋 | KEEL 저장소의 33개 고도 불균형 이진 데이터셋 (IR: 9.22 ~ 128.87) |
| 검증 방법 | 5-fold 층화 교차검증 × 3회 반복 |
| 평가 지표 | AUC |
| 비교 방법 | RUSBoost, SMOTEBoost, UnderBagging, SMOTEBagging, EasyEnsemble |
| 통계 검정 | Friedman aligned-ranks test + Holm post-hoc test + Wilcoxon test |
| Boosting 반복 | 10회 (Boosting 기반), 40회 (Bagging 기반) |
| EUS 파라미터 | 집단 크기=50, 평가 횟수=10,000, P=0.2 |

### 2.6 성능 향상

| 비교 대상 | 통계 검정 결과 |
|-----------|--------------|
| EasyEnsemble (EASY) | $p = 0.00000$ → 유의하게 EUSBoost 우수 |
| SMOTEBoost (SBO) | $p = 0.00015$ → 유의하게 EUSBoost 우수 |
| SMOTEBagging (SBAG) | $p = 0.00678$ → 유의하게 EUSBoost 우수 |
| UnderBagging (UB) | $p = 0.01468$ → 유의하게 EUSBoost 우수 |
| **RUSBoost (RUS)** | Wilcoxon: $R^+ = 399.0$, $R^- = 162.0$, $p = 0.03327$ → 유의하게 EUSBoost 우수 |

**평균 AUC 결과:**

| 방법 | 평균 AUC |
|------|---------|
| $\text{EUSB}^1_Q$ (제안) | **0.8626** |
| EUSB $^1$ | 0.8544 |
| RUSBoost | 0.8530 |
| UnderBagging | 0.8499 |
| SMOTEBagging | 0.8434 |
| EasyEnsemble | 0.8334 |
| SMOTEBoost | 0.8295 |

### 2.7 Kappa-AUC 오류 다이어그램

**카파($\kappa$) 계산:**

$$\kappa_{i,j} = \frac{2(N^{00}N^{11} - N^{01}N^{10})}{(N^{00}+N^{01})(N^{00}+N^{10}) + (N^{01}+N^{11})(N^{10}+N^{11})}$$

**쌍별 AUC 오류:**

$$\text{AUC error}_{ij} = 1 - \frac{AUC_i + AUC_j}{2}$$

Kappa-AUC 움직임 다이어그램 분석 결과: EUSBoost는 대부분의 데이터셋에서 **약간의 다양성 감소를 대가로 AUC 오류를 크게 감소**시켰으며(우하향 화살표), 일부 케이스에서는 다양성도 함께 증가했습니다(좌하향 화살표).

### 2.8 계산 복잡도 (한계)

EUSBoost의 추가 계산 복잡도:

$$O(n \cdot n_{ref} \cdot m \cdot maxeval \cdot (T-1))$$

여기서:
- $n$: 데이터셋 총 인스턴스 수
- $n_{ref} \approx 2n^+$: 언더샘플된 참조 집합 크기 (소수 클래스 수의 약 2배)
- $m$: 속성 수
- $maxeval = 10,000$: EUS 허용 평가 횟수
- $T = 10$: Boosting 반복 횟수

**핵심 한계점:**
1. **높은 훈련 시간**: RUSBoost 대비 EUS 실행으로 인한 상당한 계산 비용 증가
2. **이진 분류 제한**: 두 클래스 문제에만 적용 (다중 클래스 확장 미검증)
3. **오프라인 학습 전제**: 온라인/스트리밍 환경 부적합
4. **기저 분류기 제한**: C4.5만 사용 (다른 약한 학습기 미검증)
5. **다양성 측도 의존성**: Hamming 거리 기반 방법(EUSBH)은 효과 미흡

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능 향상의 핵심 메커니즘

#### 3.1.1 감독된 언더샘플링을 통한 일반화 개선

랜덤 언더샘플링은 $IR$이 클수록 유용한 다수 클래스 인스턴스를 제거할 확률이 높아집니다. EUS는 **GM을 피트니스 함수로 사용한 진화적 최적화**를 통해 다음을 달성합니다:

$$GM = \sqrt{TPrate \cdot TNrate}$$

GM을 최대화함으로써 두 클래스 모두에서 균형 잡힌 정확도를 달성하고, 소수 클래스에 대한 과소학습(underfitting)과 다수 클래스에 대한 과적합(overfitting)을 동시에 방지합니다.

#### 3.1.2 다양성-정확도 균형을 통한 일반화

$\beta = \frac{N-t-1}{N}$ 인수는 반복이 진행됨에 따라 **정확도와 다양성의 가중치를 동적으로 조절**합니다:

$$\text{fitness}_{EUS_Q} = \underbrace{\text{fitness}_{EUS} \cdot \frac{1.0}{\beta}}_{\text{정확도 항}} - \underbrace{\frac{10.0}{IR} - Q \cdot \beta}_{\text{다양성 항}}$$

- **초기 반복** ($t \approx 0$, $\beta \approx 1$): 다양성 강조 → 다양한 결정 경계 탐색
- **후기 반복** ($t \approx T-1$, $\beta \approx 0$): 정확도 강조 → 어려운 인스턴스에 집중 (Boosting 철학)

이러한 균형은 **편향-분산 트레이드오프(Bias-Variance Tradeoff)** 관점에서:
- 다양한 기저 분류기 → 분산(Variance) 감소
- 정확한 기저 분류기 → 편향(Bias) 감소

#### 3.1.3 IR에 따른 적응적 다양성 제어

$$\frac{10.0}{IR}$$

$IR$이 클수록 이 항의 절대값이 작아져 다양성 촉진의 영향력이 상대적으로 증가합니다. 이는 **극단적 불균형 상황에서 더욱 적극적으로 다양한 다수 클래스 부분집합을 탐색**하게 하여, 희소한 소수 클래스의 결정 경계를 더 잘 학습할 수 있게 합니다.

#### 3.1.4 통계적 독립성을 통한 일반화

Q-통계량을 최대값으로 집계:

$$Q = \max_{i=1,\ldots,t} Q_{i,j}$$

평균값이 아닌 최대값을 사용함으로써, **어떤 이전 부분집합과도 가장 다른 새 부분집합**을 선호합니다. 이는 앙상블 내에서 상호보완적인 오류 패턴을 생성하여 일반화 성능을 향상시킵니다.

### 3.2 일반화 성능의 한계 및 고려 사항

- **데이터셋 의존성**: IR이 중간 수준(~10)인 경우 정확도 기반 학습이 더 중요하지만, 극단적 불균형(IR > 30)에서는 다양성 효과가 더 두드러짐
- **과적합 가능성**: AUC 대신 GM을 EUS 내부 평가 지표로 사용하는 것은 논문 자체에서 "GM이 AUC보다 과적합 문제가 적다"고 언급
- **결정 경계의 불안정성**: CHC 알고리즘의 확률론적 특성으로 인해 실행마다 결과가 달라질 수 있어, 소규모 데이터셋에서 일반화 불안정성 발생 가능

---

## 4. 앞으로의 연구에 미치는 영향 및 고려 사항

### 4.1 연구에 미치는 영향

#### 4.1.1 방법론적 기여

1. **EUS 기반 앙상블 설계 패러다임 제시**: 랜덤 샘플링을 진화적 최적화로 대체하는 접근법은 이후 연구에서 다양한 메타휴리스틱(PSO, ABC 등)을 Boosting에 통합하는 연구로 확장 가능
2. **다양성-정확도 동적 균형 개념**: $\beta$ 인수를 통한 동적 가중치 조절은 적응형 앙상블 학습의 새로운 방향 제시
3. **Kappa-AUC 다이어그램**: 불균형 도메인에 특화된 시각화 도구로, 이후 앙상블 분석 연구의 표준 도구로 활용 가능

#### 4.1.2 실용적 기여

- **KEEL 플랫폼 기반 재현 가능한 벤치마크**: 33개 표준 데이터셋과 실험 설정 공개로 후속 연구의 비교 기준 마련
- **불균형 의료/이상 탐지 분야**: IR이 매우 높은 실제 문제(사기 탐지, 희귀 질병 예측)에 직접 적용 가능

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> **⚠️ 중요 고지**: 아래 연구들은 제공된 PDF 원문에는 포함되지 않으며, 본 분석가의 학습 데이터 기반 지식을 활용한 것입니다. 2021년 이후 데이터에 대한 완전한 접근은 불가하므로, **각 논문의 세부 수치와 저자 정보는 반드시 원본 논문에서 직접 확인하시기 바랍니다.**

### 5.1 딥러닝 기반 불균형 학습 연구

#### Self-Paced Ensemble (SPE)
- **Liu et al., "Self-paced Ensemble for Highly Imbalanced Massive Data Classification," ICDE 2020**
- **핵심 차별점**: EUSBoost의 진화적 최적화 대신 난이도 기반 샘플링(hardness-aware sampling)을 사용하여 계산 효율성 대폭 향상
- AUC 기반 "hardness"를 추정하여 언더샘플링 분포를 결정, EUSBoost의 GM 기반 EUS와 개념적으로 유사하나 진화 비용 없음
- **EUSBoost 대비 장점**: 대규모 데이터(빅데이터)에도 적용 가능한 확장성
- **EUSBoost 대비 단점**: 감독된 인스턴스 선택의 최적성 보장 미흡

#### MESA (Meta-Sampler for Imbalanced Learning)
- **Liu et al., NeurIPS 2020**
- 메타학습 기반으로 샘플링 전략 자동 학습
- EUSBoost의 수동 설계된 다양성 메커니즘 대신 데이터 적응형 샘플링 전략 학습

### 5.2 GAN 기반 오버샘플링과의 결합

#### CTGAN 기반 불균형 학습
- **조건부 GAN(CTGAN)을 활용한 소수 클래스 생성** 연구들이 2020년 이후 활발히 진행
- EUSBoost의 EUS(언더샘플링)와 GAN 오버샘플링의 하이브리드 접근법 가능성 제시
- **한계**: GAN 훈련 불안정성 + 소규모 소수 클래스에서의 모드 붕괴

### 5.3 Transformer 기반 불균형 분류

- BERT, TabTransformer 등을 기반으로 테이블 데이터의 불균형 분류에 적용하는 연구들이 등장
- EUSBoost의 C4.5 기반 접근법과 달리 특성 표현 학습 능력 보유
- **EUSBoost 대비 단점**: 소규모 불균형 데이터에서의 과적합 위험

### 5.4 비교 요약표

| 비교 항목 | EUSBoost (2013) | SPE (2020) | GAN 기반 방법 | Transformer 기반 |
|-----------|----------------|-----------|--------------|-----------------|
| **불균형 처리 방식** | 진화적 언더샘플링 | 난이도 기반 샘플링 | 생성적 오버샘플링 | 표현 학습 |
| **계산 비용** | 높음 ( $O(n \cdot n_{ref} \cdot m \cdot maxeval \cdot T)$ ) | 낮음 | 매우 높음 | 높음 |
| **대규모 데이터 적용성** | 제한적 | 우수 | 제한적 | 우수 |
| **해석 가능성** | 높음 (C4.5) | 중간 | 낮음 | 낮음 |
| **다양성 메커니즘** | 명시적 (Q-stat 기반) | 암묵적 | 없음 | 없음 |
| **소규모 데이터 성능** | 우수 | 중간 | 미흡 | 미흡 |
| **다중 클래스 지원** | ✗ | ✓ | ✓ | ✓ |

---

## 6. 앞으로 연구 시 고려할 점

### 6.1 방법론적 확장

1. **다중 클래스 불균형으로 확장**: 현재 이진 분류에 국한된 EUSBoost를 One-vs-Rest 또는 One-vs-One 전략과 결합하여 다중 클래스 문제에 적용

2. **비쌍별(non-pairwise) 다양성 측도 탐색**: 논문 자체에서 제안한 미래 방향으로, 앙상블 전체를 동시에 평가하는 다양성 측도 개발 필요

```math
\text{diversity}_{global} = f(V_1, V_2, \ldots, V_T) \neq \max_{i,j} Q_{i,j}
```

3. **메타학습 기반 자동 파라미터 조정**: $P$, $\beta$, EUS 집단 크기 등의 하이퍼파라미터를 데이터셋 메타 특성에 기반하여 자동 선택

4. **온라인/증분 학습 적응**: 스트리밍 데이터에서의 동적 불균형 처리를 위한 EUSBoost의 온라인 버전 개발

5. **다양한 기저 분류기 탐색**: C4.5 외 SVM, 신경망 등 다양한 약한/강한 학습기와의 호환성 검증

### 6.2 평가 체계 강화

6. **추가 성능 지표 고려**: AUC 외 G-Mean, F1-score, Matthews Correlation Coefficient (MCC) 등 다양한 지표로 평가

   $$MCC = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

7. **노이즈 강건성 평가**: 실제 데이터에서 빈번히 발생하는 레이블 노이즈, 특성 노이즈에 대한 EUSBoost의 강건성 분석

8. **도메인 이동(Domain Shift) 평가**: 훈련과 테스트 데이터의 분포 차이 시나리오에서의 일반화 성능 평가

### 6.3 확장성 및 효율성

9. **병렬화**: EUS의 각 반복(Boosting 루프 내)을 병렬 처리하여 계산 비용 절감

10. **대용량 데이터 적응**: 극단적으로 큰 데이터셋($n > 10^6$)에서의 EUS 적용 가능성 탐색, 예를 들어 근사 1NN 알고리즘(ANN) 활용

11. **딥러닝과의 통합**: CNN, LSTM 등 딥러닝 모델을 기저 분류기로 활용하는 Deep EUSBoost 탐색 (단, 계산 비용 및 약한 학습기 가정 위반 문제 검토 필요)

---

## 참고 자료

### 논문 원문 (제공된 PDF)
- **Galar, M., Fernández, A., Barrenechea, E., & Herrera, F. (2013). EUSBoost: Enhancing ensembles for highly imbalanced data-sets by evolutionary undersampling. *Pattern Recognition*, 46(12), 3460–3471. https://doi.org/10.1016/j.patcog.2013.05.006**

### 논문 내 인용 핵심 참고문헌 (PDF 원문에서 확인됨)
- Seiffert, C., Khoshgoftaar, T., Van Hulse, J., & Napolitano, A. (2010). RUSBoost. *IEEE Transactions on Systems, Man, and Cybernetics, Part A*, 40(1), 185–197.
- Chawla, N.V., Bowyer, K.W., Hall, L.O., & Kegelmeyer, W.P. (2002). SMOTE. *Journal of Artificial Intelligence Research*, 16, 321–357.
- Chawla, N.V., Lazarevic, A., Hall, L.O., & Bowyer, K.W. (2003). SMOTEBoost. *PKDD'03*, 107–119.
- García, S., & Herrera, F. (2009). Evolutionary undersampling for classification with imbalanced datasets. *Evolutionary Computation*, 17, 275–306.
- Galar, M., Fernández, A., Barrenechea, E., Bustince, H., & Herrera, F. (2012). A review on ensembles for the class imbalance problem. *IEEE Transactions on Systems, Man, and Cybernetics, Part C*, 42(4), 463–484.
- Liu, X.-Y., Wu, J., & Zhou, Z.-H. (2009). EasyEnsemble. *IEEE Transactions on Systems, Man, and Cybernetics, Part B*, 39(2), 539–550.
- Wang, S., & Yao, X. (2009). SMOTEBagging. *IEEE CIDM 2009*.
- Kuncheva, L.I. (2004). *Combining Pattern Classifiers: Methods and Algorithms*. Wiley-Interscience.
- Freund, Y., & Schapire, R.E. (1997). AdaBoost. *Journal of Computer and System Sciences*, 55(1), 119–139.

### 2020년 이후 비교 분석 참고 (학습 데이터 기반, 원문 확인 필요)
- Liu, Z., Cao, W., Gao, Z., et al. (2020). Self-paced Ensemble for Highly Imbalanced Massive Data Classification. *ICDE 2020*.
- Liu, Z., et al. (2020). MESA: Boost Ensemble Imbalanced Learning with MEta-SAmpler. *NeurIPS 2020*.
