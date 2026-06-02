
# A Review on Ensembles for the Class Imbalance Problem: Bagging-, Boosting-, and Hybrid-Based Approaches

> **서지 정보:** Galar, M., Fernández, A., Barrenechea, E., Bustince, H., & Herrera, F. (2012). *IEEE Transactions on Systems, Man, and Cybernetics, Part C: Applications and Reviews*, 42(4), 463–484.
> DOI: 10.1109/TSMCC.2011.2161285

---

## 1. 🔍 핵심 주장 및 주요 기여 요약

클래스 불균형(Class Imbalance) 문제는 한 클래스의 샘플 수가 다른 클래스보다 현저히 적을 때 발생하는, 데이터 마이닝 분야의 핵심 난제입니다.

머신러닝에서 앙상블 분류기는 여러 분류기를 결합하여 단일 분류기의 정확도를 높이는 것으로 알려져 있지만, 이러한 학습 기법 단독으로는 클래스 불균형 문제를 해결하지 못하며, 이를 다루기 위해서는 앙상블 학습 알고리즘이 특별히 설계되어야 합니다.

**주요 기여:**

1. 불균형 데이터에서 클래스 불균형을 해결하는 앙상블 기반 방법들에 대한 **분류 체계(Taxonomy)**를 제안하며, 각 제안 방법은 그것이 기반하는 내부 앙상블 방법론에 따라 분류됩니다.

2. 또한, 제안된 분류 체계 내의 가장 중요하게 출판된 방법들을 고려하여 **철저한 실증 비교**를 수행함으로써, 어떤 방법이 차이를 만드는지를 보여줍니다.

3. 이 비교를 통해 랜덤 언더샘플링 기법과 배깅(Bagging) 또는 부스팅(Boosting) 앙상블을 결합하는 **가장 단순한 접근 방식이 좋은 성능**을 보임을 확인하였습니다.

4. 앙상블 제안들 중, Bagging과 Boosting을 전처리 기법과 결합하는 방법이 소수 클래스의 분류 성능을 향상시키는 능력을 입증하였습니다.

---

## 2. 🧩 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

클래스 불균형 문제는 하나의 클래스 샘플 수가 다른 클래스보다 훨씬 적을 때 발생하며, 의료 진단, 사기 탐지, 네트워크 침입 탐지 등 많은 실제 응용 분야에서 연구자들의 관심이 증가하고 있습니다.

클래스 불균형의 핵심 문제:
- 분류기가 **다수 클래스(Majority Class)** 예측에 편향됨
- **소수 클래스(Minority Class)** 재현율(Recall) 급감
- 전체 정확도(Accuracy)는 높아 보여도 실질적인 분류 성능 저하

---

### 2.2 제안하는 방법 및 수식

논문은 세 가지 카테고리로 앙상블 방법을 분류합니다.

앙상블 방법은 크게 **배깅 기반(Bagging-style)** 방법(UnderBagging, OverBagging, SMOTEBagging 등), **부스팅 기반(Boosting-based)** 방법(SMOTEBoost, RUSBoost, DataBoost-IM 등), **하이브리드(Hybrid)** 앙상블 방법(EasyEnsemble, BalanceCascade 등)으로 분류됩니다.

---

#### 🔷 (A) 배깅 기반 (Bagging-Based)

**표준 배깅의 앙상블 예측:**

$$H(\mathbf{x}) = \arg\max_{c \in \mathcal{Y}} \sum_{t=1}^{T} \mathbb{1}[h_t(\mathbf{x}) = c]$$

**UnderBagging (랜덤 언더샘플링 + 배깅):**  
각 배깅 라운드 $t$에서 다수 클래스 샘플을 소수 클래스 수에 맞게 랜덤 언더샘플링:

$$D_t^{-} = \text{RandomUnderSample}(D^{-}, |D^{+}|), \quad \tilde{D}_t = D_t^{-} \cup D^{+}$$

**SMOTEBagging:**  
각 반복마다 SMOTE 오버샘플링 비율 $r_t$를 달리하여 배깅:

```math
D_t^{syn} = \text{SMOTE}(D^{+},\ r_t), \quad r_t \in \{0\%, 100\%, \ldots\}
```

---

#### 🔷 (B) 부스팅 기반 (Boosting-Based)

**AdaBoost 기반 가중치 업데이트:**

$$w_t(i) \leftarrow w_t(i) \cdot \exp\left(\alpha_t \cdot \mathbb{1}[h_t(\mathbf{x}_i) \neq y_i]\right)$$

$$\alpha_t = \frac{1}{2} \ln \left( \frac{1 - \epsilon_t}{\epsilon_t} \right), \quad \epsilon_t = \sum_{i: h_t(\mathbf{x}_i) \neq y_i} w_t(i)$$

**SMOTEBoost:**  
각 부스팅 라운드 $t$에서 SMOTE를 적용하여 소수 클래스 합성 샘플 생성:

$$D_t' = D_t \cup \text{SMOTE}(D^{+}, N)$$

이후 $D_t'$ 위에서 약한 학습기 $h_t$ 훈련, 가중치 갱신.

**RUSBoost:**  
RUSBoost는 데이터 리샘플링을 위해 랜덤 언더샘플링(RUS)을 사용하고, 부스팅을 위해 AdaBoost 기법을 결합한 앙상블 학습 접근 방식입니다.

$$D_t' = \text{RUS}(D_t),\quad \epsilon_t = \sum_{i \in D_t'} \frac{w_t(i)}{\sum_j w_t(j)} \cdot \mathbb{1}[h_t(\mathbf{x}_i) \neq y_i]$$

RUSBoost는 RUS가 계산 비용이 크지 않아 SMOTEBoost보다 단순한 방법입니다.

---

#### 🔷 (C) 하이브리드 기반 (Hybrid-Based)

**EasyEnsemble:**

EasyEnsemble의 경우, 다수 클래스 데이터셋을 여러 서브셋으로 나누고, 각 서브셋을 소수 클래스와 합쳐 AdaBoost로 직렬로 앙상블을 학습합니다.

수식으로 표현하면:

$$H(\mathbf{x}) = \text{sign}\left(\sum_{s=1}^{S} \sum_{t=1}^{T} \alpha_{s,t} h_{s,t}(\mathbf{x})\right)$$

여기서 $S$는 서브셋의 수, $T$는 각 AdaBoost 라운드 수.

**BalanceCascade:**

BalanceCascade는 학습 알고리즘에 의해 언더샘플링 과정이 안내되는 지도 전략으로, 현재 반복에서 학습 알고리즘이 올바르게 분류한 다수 클래스의 패턴을 폐기하는 방식입니다.

$$D^{-}_{t+1} = D^{-}_t \setminus \{\mathbf{x}_i \in D^{-}_t \mid h_t(\mathbf{x}_i) = \text{correct}\}$$

---

### 2.3 모델 구조 요약

```
[전체 불균형 데이터셋]
        │
        ├─ [배깅 계열] ─ 샘플링(RUS/SMOTE) → 부트스트랩 반복 → 다수결 투표 → H(x)
        │
        ├─ [부스팅 계열] ─ 샘플링(RUS/SMOTE) → AdaBoost 가중치 갱신 → 가중합 → H(x)
        │
        └─ [하이브리드] ─ EasyEnsemble: AdaBoost 앙상블의 앙상블
                        └─ BalanceCascade: 반복 제거 + 순차 학습
```

---

### 2.4 성능 평가 및 주요 결과

논문에서는 AUC(Area Under ROC Curve), G-mean 등 불균형 데이터에 적합한 지표를 사용합니다.

$$\text{AUC} = \int_0^1 \text{TPR}(t)\, d\text{FPR}(t)$$

$$\text{G-mean} = \sqrt{\text{Sensitivity} \times \text{Specificity}} = \sqrt{\frac{TP}{TP+FN} \times \frac{TN}{TN+FP}}$$

실증 결과는 앙상블 기반 알고리즘이 전처리 기법만을 단독 사용하는 경우보다 성능이 뛰어남을 보여줌으로써, 복잡도 증가가 통계적 유의성에 의해 정당화됨을 보여줍니다.

---

### 2.5 한계점

| 한계 | 설명 |
|------|------|
| **이진 분류 한정** | 논문은 이진 클래스 문제에 집중하며, 다중 클래스 불균형에 대한 분석은 제한적 |
| **고차원 데이터** | 특성 수가 매우 많은 경우의 성능 분석 부족 |
| **RUS의 정보 손실** | RUS는 데이터셋의 중요한 정보를 손실할 수 있음 |
| **딥러닝 미포함** | 2012년 논문으로, 딥러닝 기반 방법론 비교 없음 |
| **데이터 내재적 복잡성** | 클래스 오버랩, 소규모 개념, 노이즈 등 구체적인 데이터 복잡성 요인 분석 부족 |

---

## 3. 🎯 모델의 일반화 성능 향상 가능성

일반화 성능(Generalization)은 불균형 학습에서 특히 중요한 주제입니다.

### 3.1 앙상블이 일반화에 기여하는 이유

**분산-편향 트레이드오프(Bias-Variance Tradeoff):**

$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

- **배깅**은 주로 **분산(Variance)** 을 줄여 일반화 성능 향상
- **부스팅**은 주로 **편향(Bias)** 을 줄임 (불균형 상황에서는 소수 클래스에 더 집중)
- **하이브리드**는 두 효과를 동시에 활용

### 3.2 일반화를 위한 샘플링 전략

비교 결과 랜덤 언더샘플링 기법과 배깅을 결합하는 가장 단순한 접근 방식이 좋은 성능을 보였으며, 샘플링 기법과 배깅 간의 긍정적인 시너지가 두드러졌습니다.

$$\text{일반화 성능} \approx \frac{1}{T}\sum_{t=1}^{T} \text{Error}(h_t) + \text{Diversity}$$

앙상블의 다양성(Diversity) 향상이 일반화에 핵심적 역할:

$$\text{Diversity} = \frac{1}{T(T-1)}\sum_{t \neq t'} \text{Disagreement}(h_t, h_{t'})$$

### 3.3 소수 클래스 일반화의 특수성

훈련 데이터에서 소수 클래스 샘플의 수가 적으면 최적 분류 학습이 어렵고, 다수 클래스의 빈번한 샘플이 소수 클래스와의 분류 경계의 일반화를 방해합니다.

이를 극복하기 위해 논문은 **다양한 균형 서브셋**으로 반복 학습하는 방식이 모델이 소수 클래스의 **결정 경계**를 더 잘 학습하게 한다는 점을 실증합니다.

---

## 4. 🚀 향후 연구에 미치는 영향 및 고려사항

### 4.1 이 논문이 미친 영향

이 논문(Galar et al., 2012)은 1,500회 이상 인용되며, 불균형 학습 앙상블 연구의 **핵심 참조 문헌**이 되었습니다. 주요 영향은 다음과 같습니다:

- **체계적 분류 체계 제공:** 이후 모든 불균형 앙상블 연구가 이 Taxonomy를 기반으로 자신의 위치를 정의
- **RUSBoost·SMOTEBoost의 표준화:** 이후 연구들의 베이스라인으로 광범위하게 활용
- 이 논문에서 제안한 앙상블 기반 기법의 분류 체계는 각 방법을 특정 앙상블 방법론에 따라 분류하고, 앙상블 분야의 주요 발표 기법들에 대한 완전한 실험적 비교를 발전시킴으로써 이후 연구들에 큰 영향을 미쳤습니다.

---

### 4.2 2020년 이후 최신 연구 비교 분석

#### 📌 ① GAN 기반 소수 클래스 합성 (2020~)

GAN으로 생성된 합성 샘플은 더 포괄적인 데이터셋을 구성할 뿐만 아니라, 머신러닝 모델이 지배적인 클래스에 편향되지 않고 모든 클래스에 걸쳐 더 잘 학습하고 일반화할 수 있도록 합니다.

GAN 기반의 심층 생성 모델은 더 풍부한 분포 모델링을 제공하지만, 심각한 불균형 상황에서 훈련 불안정성과 모드 붕괴(mode collapse)로 어려움을 겪습니다.

| 비교 항목 | Galar et al. (2012) | GAN 기반 연구 (2020~) |
|-----------|--------------------|-----------------------|
| 샘플 생성 | SMOTE(선형 보간) | GAN(비선형 분포 학습) |
| 일반화 | 배깅/부스팅의 다양성 | 생성 분포의 다양성 |
| 계산 비용 | 낮음 | 높음 |
| 안정성 | 높음 | 모드 붕괴 위험 |

#### 📌 ② 딥러닝 + 앙상블 통합 (2021~)

SleepEGAN은 수면 단계의 불균형 분류를 위해 GAN 기반 앙상블 딥러닝 모델을 개발하며, 클래스 불균형 완화를 위해 EEG 신호 특성에 맞는 새로운 GAN 아키텍처를 제안합니다.

비용 없는 앙상블 학습 전략을 설계하여 검증 셋과 테스트 셋 간의 이질성으로 인한 모델 추정 분산을 줄여 예측 성능의 정확도와 견고성을 향상시킵니다.

#### 📌 ③ 앙상블 + 데이터 증강 통합 리뷰 (2023)

전통적인 데이터 증강 방법인 SMOTE와 랜덤 오버샘플링(ROS)이 선택된 클래스 불균형 문제에서 GAN보다 성능이 우수할 뿐만 아니라 계산 비용도 더 저렴합니다.

#### 📌 ④ 클래스 불균형 비율(CIR)의 영향 분석 (2023)

특별히 설계된 앙상블 기반 방법은 전통적인 분류기의 어려움을 극복하고 클래스 불균형 문제를 처리할 수 있으며, 44개의 불균형 데이터셋에 대해 19개의 앙상블 방법의 성능을 평가하고, 데이터셋을 약간 불균형(SI), 중간 불균형(MI), 심각한 불균형(HI)으로 나눠 클래스 불균형 비율의 효과를 관찰합니다.

---

### 4.3 앞으로 연구 시 반드시 고려할 점

| 고려사항 | 세부 내용 |
|----------|-----------|
| **다중 클래스 불균형** | 실세계 문제 대부분은 다중 클래스이므로, 이진 분류 중심 방법론의 확장 연구 필요 |
| **데이터 내재적 복잡성** | 클래스 오버랩(overlap), 소규모 개념(small disjuncts), 클래스 내 불균형(within-class imbalance) 등 세분화된 분석 필요 |
| **딥러닝 통합** | 딥러닝은 강력한 표현 능력을 지니나, 클래스 불균형과 개인 이질성이 머신러닝 알고리즘의 분류 성능에 크게 영향을 미칩니다. |
| **고차원/희소 데이터** | 텍스트, 유전체 데이터 등 고차원 공간에서의 샘플링 전략 재검토 |
| **평가 지표 선택** | Accuracy 대신 AUC, F1-score, G-mean, MCC 등 불균형에 강건한 지표 사용 |
| **GAN 불안정성 해소** | 훈련 불안정성, 모드 붕괴, 하이퍼파라미터 민감도 문제가 여전히 극한 불균형 상황에서 도전적인 과제로 남아 있어 개선 연구 필요 |
| **설명 가능성(XAI)** | 앙상블 모델의 복잡도 증가에 따른 해석 가능성 확보 연구 병행 필요 |
| **스트리밍/실시간 불균형** | 동적으로 변화하는 불균형 비율에 적응하는 온라인 앙상블 학습 연구 필요 |

---

## 📚 참고 자료 및 출처

| # | 자료 | 출처 |
|---|------|------|
| 1 | **[핵심 논문]** Galar et al. (2012), "A Review on Ensembles for the Class Imbalance Problem" | [IEEE Xplore](https://ieeexplore.ieee.org/document/5978225/) / [ACM DL](https://dl.acm.org/doi/abs/10.1109/TSMCC.2011.2161285) |
| 2 | **[논문 정보]** SciSpace (2012, 1519 Citations) | [scispace.com](https://scispace.com/papers/a-review-on-ensembles-for-the-class-imbalance-problem-7a1d2cor6s) |
| 3 | **[2023 최신 리뷰]** "A broad review on class imbalance learning techniques" | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1568494623004337) |
| 4 | **[2023 최신 리뷰]** "A review of ensemble learning and data augmentation models for class imbalanced problems" (arXiv:2304.02858) | [arxiv.org](https://arxiv.org/abs/2304.02858) |
| 5 | **[2023 연구]** "Impact of class imbalance ratio on ensemble methods" | [SAGE Journals](https://journals.sagepub.com/doi/10.3233/JIFS-223333) |
| 6 | **[GAN 기반]** "Addressing the class imbalance in tabular datasets from a GAN approach" (2023) | [SAGE Journals](https://journals.sagepub.com/doi/10.1177/17483026231215186) |
| 7 | **[딥러닝+앙상블]** "SleepEGAN: A GAN-enhanced Ensemble Deep Learning Model" (arXiv:2307.05362) | [arxiv.org](https://arxiv.org/pdf/2307.05362) |
| 8 | **[Ensemble-GAN]** "Generative multi-adversarial network for abdominal image segmentation" | [PubMed/PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7603459/) |
| 9 | **[최신 연구]** "Learning Majority-to-Minority Transformations with MMD and Triplet Loss" (arXiv:2509.11511) | [arxiv.org](https://arxiv.org/pdf/2509.11511) |
| 10 | **[TLUSBoost]** "TLUSBoost algorithm: a boosting solution for class imbalance problem" | [Springer](https://link.springer.com/article/10.1007/s00500-018-3629-4) |
| 11 | **[Semantic Scholar]** Paper citation & summary | [semanticscholar.org](https://www.semanticscholar.org/paper/A-Review-on-Ensembles-for-the-Class-Imbalance-and-Galar-Fern%C3%A1ndez/afcc28d71be4ea6a48a339f9e4e5557d1b2b25be) |

> ⚠️ **정확도 주의:** 논문 원문의 일부 세부 수식(예: 정확한 가중치 업데이트 파라미터)은 접근 제한으로 인해 AdaBoost/SMOTEBoost/RUSBoost의 표준 공식을 기반으로 재구성하였습니다. 완전한 수식 확인을 위해서는 IEEE Xplore 원문을 직접 참조하시길 권장합니다.
