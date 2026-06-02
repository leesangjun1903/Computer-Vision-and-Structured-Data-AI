# Diversity Analysis on Imbalanced Data Sets by Using Ensemble Models

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문(Wang & Yao, 2009 IEEE CIDM)의 핵심 주장은 다음과 같습니다:

> **앙상블 모델에서 다양성(Diversity)과 정확도(Accuracy)는 트레이드오프 관계에 있으며, 불균형 데이터셋에서 최적 성능은 "높은 정확도 + 낮은 다양성"이 아닌 "중간 정확도 + 중간 다양성" 상태에서 달성된다.**

### 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| 다양성 분석 프레임워크 | 재샘플링 비율 조절을 통한 다양성-성능 관계 체계적 분석 |
| 세 가지 앙상블 모델 제안 | UnderBagging, OverBagging, SMOTEBagging |
| 다중 클래스 확장 | SMOTE를 다중 소수 클래스 문제에 적용 가능하도록 확장 |
| 클래스별 다양성 측정 | Q-통계량을 전체 데이터 및 개별 클래스별로 각각 계산 |
| 실증적 분석 | 8개 UCI 데이터셋(6개 이진 + 2개 다중 클래스)에서 검증 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**클래스 불균형 문제(Class Imbalance Problem)**:
- 현실 세계 데이터에서 다수 클래스(majority class)에 비해 소수 클래스(minority class)의 샘플이 극히 적음
- 표준 머신러닝 알고리즘은 정확도 최대화를 목표로 하여 다수 클래스에 편향됨
- 의료 진단, 사기 탐지, 텍스트 분류 등 실용 분야에서 치명적

**기존 앙상블 기법의 한계**:
- 앙상블 성능에 영향을 미치는 두 요인(정확도, 다양성) 중 **다양성이 소수 클래스에 미치는 영향이 불분명**함
- 기존 연구(SMOTEBoost, DataBoost, BEV 등)가 이진 분류 문제에만 집중

---

### 2.2 주요 수식

#### (1) F-measure

$$F\text{-}value = \frac{(1 + \beta^2) \cdot recall \cdot precision}{\beta^2 \cdot recall + precision} \tag{1}$$

- $\beta$: precision과 recall의 상대적 중요도 (일반적으로 $\beta = 1$)
- $TP$: True Positive, $FP$: False Positive, $FN$: False Negative
- $recall = \frac{TP}{TP + FN}$, $precision = \frac{TP}{TP + FP}$

#### (2) Q-통계량 (두 분류기 $L_i$, $L_k$ 간)

$$Q_{i,k} = \frac{N^{11}N^{00} - N^{01}N^{10}}{N^{11}N^{00} + N^{01}N^{10}} \tag{2}$$

- $N^{ab}$: $L_i$가 결과 $a$, $L_k$가 결과 $b$를 출력한 인스턴스 수
  - $a, b = 1$: 정분류, $a, b = 0$: 오분류
- Q값 범위: $[-1, 1]$
  - $Q > 0$: 같은 인스턴스를 정분류하는 경향 (낮은 다양성)
  - $Q < 0$: 서로 다른 인스턴스에서 오류 발생 (높은 다양성)
  - $Q = 0$: 통계적 독립

#### (3) 앙상블 전체 평균 Q-통계량

$$Q_{av} = \frac{2}{M(M-1)} \sum_{i=1}^{M-1} \sum_{k=i+1}^{M} Q_{i,k} \tag{3}$$

- $M$: 앙상블 내 분류기 수
- Q값이 클수록 다양성이 낮음을 의미

#### (4) G-mean (전체 성능 평가)

$$G\text{-}mean = \sqrt[C]{\prod_{i=1}^{C} recall_i} \tag{4}$$

- $C$: 클래스 수
- 각 클래스의 recall의 기하 평균

---

### 2.3 모델 구조

#### (A) UnderBagging / OverBagging (통합 알고리즘)

재샘플링 비율 $a\%$를 조절하여 두 모델을 연속적으로 전환:

$$\text{재샘플링 비율 (클래스 } i\text{)} = \frac{N_C}{N_i} \cdot a\%$$

- $N_C$: 가장 많은 인스턴스를 가진 클래스의 수
- $N_i$: $i$번째 클래스의 인스턴스 수
- $a = 10$: UnderBagging (낮은 샘플링 → 높은 다양성)
- $a = 100$: OverBagging (높은 샘플링 → 낮은 다양성)

```
[학습 과정]
원본 훈련 데이터 S
    ↓ 재샘플링 비율 a% 적용
서브셋 Sk 생성 (각 클래스 동일 인스턴스 수)
    ↓
C4.5 결정 트리 학습 (base learner)
    ↓
M=20개 분류기 생성
    ↓
[테스트 과정]
다수결 투표(majority vote) → 최종 클래스 결정
(동점 시 소수 클래스 반환)
```

#### (B) SMOTEBagging (핵심 신규 제안)

다중 클래스를 위한 SMOTE 확장 모델. 비율 $b\%$로 원본/합성 인스턴스 비율 조절:

$$N_{SMOTE} = \frac{N_C}{N_i} \cdot (1 - b\%) \cdot 100$$

```
[SMOTEBagging 학습 과정]
Step 1: 클래스 C를 100% 복원 샘플링
Step 2: 각 클래스 i에 대해:
    - 원본에서 (N_C/N_i) × b% 복원 샘플링
    - SMOTE(k, N)으로 (N_C/N_i) × (1-b%) × 100개 합성
Step 3: C4.5로 분류기 학습
Step 4: b% 변경 (10% → 100%, 총 M=20개)
Step 5: 다수결 투표
```

**핵심 차이점**: 각 분류기마다 다른 $b\%$ 값을 사용하여 앙상블 다양성을 인위적으로 증가시킴

---

### 2.4 실험 설정

| 항목 | 내용 |
|------|------|
| 데이터셋 | 8개 UCI 데이터셋 (Hepatitis, Heart, Liver, Pima, Ionosphere, Breast-w, Glass, Yeast) |
| Base Learner | C4.5 결정 트리 |
| 교차검증 | 10-fold × 30회 반복, 평균값 사용 |
| 앙상블 크기 | M = 20개 분류기 |
| 평가 지표 | Recall, F-measure, G-mean, Q-통계량 |

---

### 2.5 성능 향상 결과

#### 이진 분류 (Pima 데이터셋 Q-통계량 예시)

| 재샘플링 비율 | 소수 클래스 Q-통계량 | 전체 Q-통계량 |
|--------------|---------------------|--------------|
| 10% | 0.449 | 0.496 |
| 50% | 0.547 | 0.625 |
| 100% | 0.552 | 0.638 |

→ 재샘플링 비율이 높아질수록 Q값 증가 = **다양성 감소**

#### 다중 클래스 OverBagging vs SMOTEBagging

| 데이터셋 | 모델 | G-mean | Q-통계량 |
|----------|------|--------|----------|
| Glass | OverBagging | 0.927 | 0.664 |
| Glass | **SMOTEBagging** | **0.960** | **0.621** |
| Yeast | OverBagging | 0.941 | 0.675 |
| Yeast | **SMOTEBagging** | **0.969** | **0.615** |

→ SMOTEBagging이 Q값 감소(다양성 증가) + G-mean 향상

#### 핵심 발견: 다양성-성능 관계

```
다양성(높음) ──────────────► 다양성(낮음)
재샘플링 비율(낮음)          재샘플링 비율(높음)

소수 클래스 Recall:  [높음] ──────────────► [낮음]
소수 클래스 F-value: [낮음] ──► [최고점] ──► [감소/유지]
G-mean:              [낮음] ──► [최고점] ──► [감소/유지]
```

**"최적점은 중간 재샘플링 비율(약 40~60%)에서 나타남"**

---

### 2.6 한계점

1. **다중 클래스 데이터셋 부족**: 오직 2개(Glass, Yeast)의 다중 클래스 데이터셋만 사용
2. **Base Learner 단일화**: C4.5 결정 트리만 사용 → SVM, 신경망 등 다른 학습기에 대한 일반화 미검증
3. **최적 재샘플링 비율 자동화 부재**: 최적 $a\%$ 또는 $b\%$ 값을 자동으로 결정하는 방법 미제시
4. **SMOTE의 고유 한계**: 노이즈/경계 지역 샘플에 취약한 SMOTE의 문제가 SMOTEBagging에도 내재
5. **다중 클래스 평가 지표**: 다중 클래스에 적합한 평가 기준 추가 연구 필요 (논문에서도 인정)
6. **클래스 간 상호작용**: 다중 소수 클래스 간 내부 불균형(inner-imbalance) 문제의 완전한 해결 미달

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 다양성이 일반화에 기여하는 메커니즘

앙상블 이론에 따르면 일반화 오류는 편향(Bias)과 분산(Variance)으로 분해됩니다:

$$E_{generalization} = Bias^2 + Variance + Noise$$

다양성이 높은 앙상블에서:

$$Variance_{ensemble} = \frac{1}{M} Variance_{single} \cdot (1 + (M-1)\bar{\rho})$$

- $\bar{\rho}$: 분류기 간 평균 상관계수
- 다양성이 높을수록 $\bar{\rho}$ 감소 → 분산 감소 → **일반화 성능 향상**

### 3.2 논문에서 도출된 일반화 관련 핵심 통찰

#### (A) 과적합 방지 메커니즘

OverBagging에서 재샘플링 비율 100% 사용 시:
- 소수 클래스 인스턴스가 과도하게 복제됨
- 분류 경계가 과도하게 특수화(over-specific)
- $N^{01} \cdot N^{10}$이 감소 → Q값 증가 → **다양성 감소 = 과적합**

반면 중간 재샘플링 비율(약 40%)에서:
- 비슷한 F-value를 유지하면서 더 높은 recall 확보
- 과적합 없이 소수 클래스 탐지 능력 유지

#### (B) SMOTEBagging의 일반화 우월성

SMOTE가 생성하는 합성 샘플은:
$$x_{new} = x_i + \lambda \cdot (x_{nn} - x_i), \quad \lambda \in [0, 1]$$
- $x_i$: 소수 클래스 샘플
- $x_{nn}$: $k$-최근접 이웃 중 하나
- $\lambda$: 균일 분포 난수

이를 통해:
- **결정 경계를 다수 클래스 방향으로 확장** (단순 복제보다 일반화에 유리)
- 각 분류기마다 다른 $b\%$ → 서로 다른 훈련 분포 → **높은 다양성 확보**

#### (C) "중간 정확도 + 중간 다양성" 상태의 일반화 우위

논문에서 정의한 4가지 상태 중 **Status 4 (Medium Accuracy + Medium Diversity)**가 최적:

| 상태 | 정확도 | 다양성 | 일반화 성능 |
|------|--------|--------|-------------|
| Status 1 | 낮음 | 낮음 | 매우 불량 |
| Status 2 | 낮음 | 높음 | 소수 클래스 recall ↑, 전체 불량 |
| Status 3 | 높음 | 낮음 | 과적합, 소수 클래스 불량 |
| **Status 4** | **중간** | **중간** | **최적 일반화** |

#### (D) 통계적 유의성 검증 (T-test 결과)

Glass, Yeast 데이터셋에서 **12개 소수 클래스 중 9개**가 최적 F-value와 100% 재샘플링(최저 다양성) 간 90% 신뢰구간에서 유의미한 차이 확인:

```math
\text{예: Yeast Class 1} \quad F_{best} = 0.925 \pm 0.026 \quad \text{vs} \quad F_{100\%} = 0.868 \pm 0.064 \quad (p < 0.1)
```

→ 높은 다양성 유지가 통계적으로 유의미한 일반화 성능 향상을 가져옴

### 3.3 일반화 성능 향상의 실용적 시사점

1. **재샘플링 비율의 최적화가 핵심**: 맹목적인 오버샘플링보다 적절한 다양성 유지가 중요
2. **합성 데이터 생성의 이점**: 단순 복제보다 SMOTE 계열이 일반화에 유리
3. **도메인별 최적점 탐색 필요**: 최적 재샘플링 비율은 데이터 특성에 따라 다름

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 향후 연구에 미치는 영향

#### (A) 이론적 기여

- **다양성-성능 관계의 체계화**: 불균형 학습에서 다양성 분석의 표준 프레임워크 제시
- **클래스별 다양성 측정**: Q-통계량을 클래스 단위로 적용하는 새로운 분석 방법론 제공
- **다중 클래스 불균형 연구의 촉진**: 이진 분류 중심 연구에서 다중 클래스로의 확장 방향 제시

#### (B) 방법론적 기여

- SMOTEBagging은 이후 ADASYN+Bagging, BorderlineSMOTE+Ensemble 등 다양한 합성 샘플링+앙상블 연구의 기반
- "재샘플링 비율을 통한 다양성 제어" 아이디어는 이후 적응형 앙상블 연구로 발전

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 비교는 제 학습 데이터(2021년 초까지)에 기반하며, 구체적 수치는 해당 논문 직접 확인이 필요합니다.

### 5.1 주요 발전 방향 비교

| 구분 | Wang & Yao (2009) | 2020년 이후 연구 동향 |
|------|-------------------|----------------------|
| **다양성 제어** | 재샘플링 비율로 수동 조절 | 메타러닝/AutoML로 자동 최적화 |
| **샘플링 방법** | SMOTE (선형 보간) | CTGAN, CVAE 등 딥러닝 기반 생성 |
| **Base Learner** | C4.5 단일 | XGBoost, LightGBM, 딥러닝 혼합 |
| **다양성 측정** | Q-통계량 | 상호정보량, 신경 탄젠트 커널 기반 |
| **다중 클래스** | 2개 데이터셋 | 수십~수백 클래스 대규모 적용 |

### 5.2 주목할 만한 2020년 이후 연구 방향

#### (A) 딥러닝 기반 불균형 학습

**관련 연구 방향**: Transformer 기반 앙상블, Self-supervised Learning을 활용한 소수 클래스 표현 학습

기존 논문의 SMOTE가 갖는 선형 보간 한계를 극복:
$$x_{new} = G_\theta(z), \quad z \sim \mathcal{N}(0, I)$$
- GAN/VAE 기반 생성 모델로 더 현실적인 소수 클래스 샘플 생성
- 단순 선형 보간 대비 결정 경계 근방의 복잡한 분포 학습 가능

#### (B) 적응형 앙상블 (Adaptive Ensemble)

Wang & Yao의 고정된 재샘플링 비율 한계를 극복하는 방향:
- **Dynamic Ensemble Selection (DES)**: 테스트 인스턴스에 따라 동적으로 최적 분류기 선택
- **Self-paced Ensemble** (Liu et al., 2020): 난이도 기반 동적 샘플링

#### (C) 페더레이티드 러닝 + 불균형 데이터

분산 환경에서의 클래스 불균형 문제: 각 클라이언트의 로컬 불균형이 글로벌 앙상블에 미치는 다양성 영향 분석

#### (D) 그래프 기반 소수 클래스 증강

**GraphSMOTE** (Zhao et al., 2021) 계열:
$$x_{new} = x_i + \lambda \cdot (x_j - x_i), \quad \text{단 } (x_i, x_j) \in E_{graph}$$
- 그래프 구조를 활용한 이웃 정의로 더 의미 있는 합성 샘플 생성

### 5.3 Wang & Yao (2009)와의 핵심 차별점

| 비교 항목 | Wang & Yao (2009) | 최신 연구 |
|-----------|-------------------|-----------|
| **다양성 최적화** | 수동, 경험적 | 자동화, 이론적 근거 강화 |
| **데이터 생성** | 선형 SMOTE | 비선형 딥러닝 생성 |
| **평가 지표** | F-value, G-mean, Q-통계량 | AUROC, AUPRC, MCC, 도메인 특화 지표 |
| **확장성** | 소규모 UCI | 대규모 실세계 데이터 |
| **해석가능성** | 제한적 | SHAP, LIME 통합 |

---

## 참고 자료

**주 논문**:
- Wang, S., & Yao, X. (2009). "Diversity Analysis on Imbalanced Data Sets by Using Ensemble Models." *2009 IEEE Symposium on Computational Intelligence and Data Mining (CIDM)*, pp. 1-8. IEEE.

**논문 내 인용 참고문헌** (논문 원문 기준):
- [1] Chawla, N. V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." *Journal of Artificial Intelligence Research*, pp. 341–378.
- [9] Breiman, L. (1996). "Bagging Predictors." *Machine Learning*, 24(2), pp. 123–140.
- [16] Yule, G. U. (1900). "On the association of attributes in statistics." *Philosophical Transactions of the Royal Society of London*, A194, pp. 257–319.
- [17] Kuncheva, L. I., & Whitaker, C. J. (2003). "Measures of diversity in classifier ensembles and their relationship with the ensemble accuracy." *Machine Learning*, 51, pp. 181–207.
- [18] Brown, G., Wyatt, J. L., & Tino, P. (2005). "Managing diversity in regression ensembles." *The Journal of Machine Learning Research*, 6, pp. 1621–1650.

**2020년 이후 관련 연구 방향** (일반적 지식 기반, 개별 논문 직접 확인 권장):
- Liu, Z., et al. (2020). "Self-paced Ensemble for Highly Imbalanced Massive Data Classification." *ICDE 2020*.
- Zhao, T., et al. (2021). "GraphSMOTE: Imbalanced Node Classification on Graphs with Graph Neural Networks." *WSDM 2021*.
