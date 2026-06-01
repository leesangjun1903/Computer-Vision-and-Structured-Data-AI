# The Effects of Ensembling on Long-Tailed Data

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문(Buchanan et al., NeurIPS 2023)의 핵심 주장은 다음과 같습니다:

> **균형 데이터셋에서는 로짓 앙상블과 확률 앙상블 간 유의미한 차이가 없으나, 롱테일(불균형) 데이터셋에서는 imbalance bias 감소 손실 함수(특히 Balanced Softmax)와 결합된 로짓 앙상블이 일관된 성능 향상을 제공한다.**

### 주요 기여

1. **체계적 비교 실험**: 균형 및 불균형 데이터셋에서 로짓/확률 앙상블을 다양한 모델과 손실 함수로 비교
2. **균형 데이터셋 검증**: CIFAR10, ImageNet 등에서 두 앙상블 방식 간 정확도 및 캘리브레이션 차이가 미미함을 확인
3. **롱테일 데이터셋 핵심 발견**: Balanced Softmax Loss + 로짓 앙상블 조합이 최고 성능 달성
4. **다양성(Diversity) 분석**: 데이터 불균형이 앙상블 멤버 간 다양성을 심화시키며, 특히 테일 클래스에서 이 효과가 극대화됨을 규명
5. **이론적 뒷받침**: Lemma E.1과 Proposition E.2를 통해 실증 결과에 대한 이론적 근거 제시

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

실제 세계 데이터는 클래스별 샘플 수가 매우 불균형한 **롱테일 분포(Long-Tailed Distribution)**를 따릅니다. 이런 환경에서:

- 기존 연구들은 로짓 앙상블 또는 확률 앙상블 중 하나를 임의로 선택하여 사용
- 두 앙상블 방식이 편향(bias)-분산(variance)에 미치는 영향이 다름에도 불구하고, 불균형 데이터에서의 체계적인 비교 연구가 부재
- 어떤 앙상블 방식이 불균형 데이터에 더 적합한지 이론적·실증적 근거가 없었음

**연구 질문**: *"불균형 데이터에서 확률 앙상블과 로짓 앙상블은 실질적으로 얼마나 차이가 있는가?"*

---

### 2.2 제안하는 방법 및 수식

#### 기본 앙상블 정의

$M$개의 독립 모델 $f_1(\cdot), \ldots, f_M(\cdot)$에 대해 두 가지 앙상블 방식을 정의합니다:

$$\bar{f}_{\text{logit}}(\boldsymbol{x}) \triangleq \text{softmax}\!\left(\frac{1}{M}\sum_{i=1}^{M} \boldsymbol{z}_i(\boldsymbol{x})\right)$$

$$\bar{f}_{\text{prob}}(\boldsymbol{x}) \triangleq \frac{1}{M}\sum_{i=1}^{M} \text{softmax}\!\left(\boldsymbol{z}_i(\boldsymbol{x})\right)$$

여기서 $\boldsymbol{z}_i(\boldsymbol{x})$는 모델 $i$의 로짓 출력입니다.

---

#### 손실 함수 정의 (Table 1)

| 손실 함수 | 수식 |
|-----------|------|
| Softmax CE (ERM) | $\mathcal{L}_{\text{ce}} = -\log(p_y)$ |
| Weighted Softmax CE | $\mathcal{L}_{\text{wce}} = -\frac{1}{\pi_y}\log(p_y)$ |
| d-Weighted Softmax CE | $\mathcal{L}_{\text{dwce}} = -\frac{1}{C\pi_y}\log(p_y)$ |
| Balanced Softmax CE | $\mathcal{L}_{\text{bs}} = -\log\!\left(\frac{\pi_y \exp(z_y)}{\sum_j \pi_j \exp(z_j)}\right)$ |
| Temperature Scaling | $\mathcal{L}_{\text{ts}} = -\log\!\left(\frac{\exp(z_y/T)}{\sum_j \exp(z_j/T)}\right)$ |

$\pi_k = n_k/n$: 클래스 $k$의 레이블 빈도, $T$: 온도 스케일링 파라미터

---

#### 앙상블 NLL의 편향-분산 분해 (Bias-Variance Decomposition)

Gupta et al. [7]과 Wood et al. [8]의 이론을 기반으로 두 앙상블의 NLL을 분해합니다:

**로짓 앙상블 NLL:**

$$\underbrace{-\mathbb{E}_{\mathcal{D}}\!\left[\boldsymbol{y}^T \cdot \ln \bar{\boldsymbol{q}}\right]}_{\text{logit ensemble NLL}} = \underbrace{-\frac{1}{M}\sum_{i=1}^{M} \boldsymbol{y}^T \cdot \ln \boldsymbol{q}_i^*}_{\text{average bias}} + \underbrace{\frac{1}{M}\sum_{i=1}^{M}\mathbb{E}_{\mathcal{D}}\!\left[D_{\text{KL}}(\boldsymbol{q}_i^* \| \boldsymbol{q}_i)\right]}_{\text{average variance}} - \underbrace{\mathbb{E}_{\mathcal{D}}\!\left[\frac{1}{M}\sum_{i=1}^{M} D_{\text{KL}}(\bar{\boldsymbol{q}} \| \boldsymbol{q}_i)\right]}_{\text{diversity}} $$

**확률 앙상블 NLL:**

$$\underbrace{-\mathbb{E}_{\mathcal{D}}\!\left[\boldsymbol{y}^T \cdot \ln \boldsymbol{q}^\dagger\right]}_{\text{prob. ensemble NLL}} = -\frac{1}{M}\sum_{i=1}^{M} \boldsymbol{y}^T \cdot \ln \boldsymbol{q}_i^* + \frac{1}{M}\sum_{i=1}^{M}\mathbb{E}_{\mathcal{D}}\!\left[D_{\text{KL}}(\boldsymbol{q}_i^* \| \boldsymbol{q}_i)\right] - \underbrace{\mathbb{E}_{\mathcal{D}}\!\left[\sum_{i=1}^{M}\frac{1}{M}\left[\log\frac{1}{M} - \log\frac{q_i^{(y)}}{\sum_{j=1}^{M}q_j^{(y)}}\right]\right]}_{\text{dependency}} $$

**두 NLL의 차이:**

$$\mathbb{E}_{\mathcal{D}}\!\left[\boldsymbol{y} \cdot \ln \bar{\boldsymbol{q}} - \boldsymbol{y} \cdot \ln \boldsymbol{q}^\dagger\right] = -\mathbb{E}_{\mathcal{D}}\!\left[\frac{1}{M}\sum_{i=1}^{M} D_{\text{KL}}(\bar{\boldsymbol{q}}\|\boldsymbol{q}_i) + \sum_{i=1}^{M}\frac{1}{M}\left[\log\frac{1}{M} - \log\frac{q_i^{(y)}}{\sum_{j=1}^{M}q_j^{(y)}}\right]\right] $$

> **핵심 해석**: 식 (4)에 따르면, **diversity 항 > dependency 항**일 때, 즉 앙상블 멤버들이 모든 클래스에 대해 다양한 예측을 할 때 로짓 앙상블이 확률 앙상블보다 우수합니다.

---

#### 이론적 뒷받침 (Appendix E)

**Lemma E.1**: Balanced Softmax Loss의 $\epsilon$-근사 최적해 $z(x)$는 다음을 만족해야 합니다:

```math
\sum_{j=1}^{K} \pi_j \exp(z_j(x))\!\left(\frac{\exp(z_y(x) - z_y^*(x))}{\exp(z_j(x) - z_j^*(x))} - \exp(\epsilon)\right) < 0, \quad \forall(x, y)
```

**Proposition E.2**: 균형 소프트맥스 손실의 앙상블 NLL 이점:

$$\text{Diff}_{\text{balanced}} - \text{Diff}_{\text{classical}} \leq \beta\sum_{j=1}^{K} \bar{q}^{(j)} \log \pi_j - \log \delta$$

이는 **$\pi_j$가 작을수록(테일 클래스일수록) 로짓 앙상블의 이점이 더 커짐**을 이론적으로 증명합니다.

---

### 2.3 모델 구조

#### 실험 설정

| 구분 | 내용 |
|------|------|
| 앙상블 크기 | $M = 4$ (동일 손실, 동일 데이터로 다른 랜덤 시드로 독립 학습) |
| 균형 데이터셋 모델 | CIFAR10: 137개 모델(32 아키텍처), ImageNet: 78개 모델 |
| 불균형 데이터셋 모델 | ResNet-32, ResNet-110 (5 seeds) |
| 앙상블 방식 | 로짓 평균, 확률 평균 |
| 캘리브레이션 | Pool-then-calibrate (온도 스케일링 후 앙상블) |

#### 데이터셋

- **균형**: CIFAR10, ImageNet
- **불균형**: CIFAR10-LT, CIFAR100-LT (지수 함수적 샘플 수 감소: $n_i = n\mu^i$, $\mu = 0.5$)
- **OOD 평가**: CINIC10, CIFAR10.1, ImageNetV2MF, ImageNet-C

#### 평가 지표

$$\text{NLL}(f(\boldsymbol{x}), y) \triangleq -\log f^{(y)}(\boldsymbol{x}), \quad \text{B}(f(\boldsymbol{x}), y) \triangleq \|f(\boldsymbol{x}) - \mathbf{1}_y\|_2^2 $$

추가적으로 Calibration AUC (ROC curve, PR curve)를 사용하여 NLL/Brier Score의 순위 불안정성 문제를 보완합니다.

---

### 2.4 성능 향상 결과

#### 균형 데이터셋 (CIFAR10, ImageNet)

- 로짓 앙상블과 확률 앙상블 간 **유의미한 차이 없음** (MSE < $10^{-5}$ 수준)
- 0-1 Error, F1 Score, Calibration PR AUC 모두에서 동등한 성능

#### 불균형 데이터셋 (CIFAR10-LT, ResNet-32)

| 학습 손실 | 앙상블 유형 | Acc. | F1 | Brier | NLL | Cal-PR AUC |
|-----------|------------|------|----|-------|-----|------------|
| Softmax CE | 단일 모델 | 67.68 | 66.17 | 0.528 | 1.33 | 0.839 |
| Softmax CE | avg. logits | 69.93 | 68.33 | 0.471 | 1.119 | 0.874 |
| Softmax CE | avg. probs | 69.92 | 68.32 | 0.446 | 1.037 | 0.878 |
| **Balanced Softmax** | **단일 모델** | **74.41** | **74.18** | **0.387** | **0.842** | **0.913** |
| **Balanced Softmax** | **avg. logits** | **79.01** | **78.93** | **0.308** | **0.654** | **0.945** |
| Balanced Softmax | avg. probs | 78.05 | 77.91 | 0.315 | 0.661 | 0.941 |

> **결론**: Balanced Softmax + 로짓 앙상블 조합이 모든 지표에서 최고 성능을 달성

#### CIFAR100-LT에서 기존 SOTA 비교 (Table 3)

| 방법 | 정확도 |
|------|--------|
| Softmax (ERM) | 41.4 |
| BBN [20] | 44.7 |
| Balanced Softmax [25] | 46.1 |
| RIDE [37] | 48.0 |
| SADE [19] | 49.8 |
| **Logit Ensemble + Balanced Softmax** | **52.0** |

---

### 2.5 한계점

1. **앙상블 크기 고정**: 실험에서 $M=4$로 고정되어 더 큰 앙상블에서의 일반화 가능성 미검증
2. **계산 비용**: Deep Ensemble은 $M$배의 추론 비용 발생; 실제 배포 환경에서의 효율성 문제
3. **손실 함수 범위**: Balanced Softmax에 집중되었으며, Logit Adjustment Loss 등 다른 디바이어싱 손실과의 결합 효과는 실험적으로 충분히 검증되지 않음
4. **아키텍처 의존성**: ResNet 계열에 집중; Transformer 기반 모델(ViT 등)에서의 효과 미검증
5. **대규모 불균형 데이터셋**: ImageNet-LT와 같은 대규모 데이터셋에서의 직접 검증 부재

---

## 3. 일반화 성능 향상 가능성 (핵심 분석)

### 3.1 OOD(Out-of-Distribution) 일반화

논문은 불균형 학습 데이터로 훈련된 모델을 OOD 데이터셋(CINIC-10)으로 평가합니다:

**CINIC-10 OOD 평가 (ResNet-32, CIFAR10-LT 학습)**:
- ERM + avg. logits: 59.87% → ERM + avg. probs: 60.01%
- **Balanced Softmax + avg. logits: 68.45%** (최고)
- Balanced Softmax + avg. probs: 67.32%

이는 로짓 앙상블 + Balanced Softmax 조합이 **분포 변화(distribution shift)**에도 강건한 일반화 성능을 제공함을 시사합니다.

### 3.2 다양성(Diversity)과 일반화의 관계

$$\text{diversity} > \text{dependency} \implies \text{로짓 앙상블이 확률 앙상블보다 NLL 기준 우월}$$

**Fig. 2 분석**: 테일 클래스(클래스 5-9)에서 앙상블 멤버 간 불일치도(disagreement)가 헤드 클래스(0-4)보다 현저히 높습니다. 이는:

- **불균형이 자연스러운 앙상블 다양성을 증가시킴**
- Balanced Softmax를 사용했을 때만 diversity > dependency 조건이 성립

**Fig. 3 분석**: Balanced Softmax Loss로 훈련된 모델에서만 diversity 항이 dependency 항을 초과하며, 이것이 로짓 앙상블의 우월성을 설명합니다.

### 3.3 편향 수정과 일반화

Balanced Softmax Loss의 핵심은 학습-테스트 분포 불일치를 수정하는 것입니다:

$$\mathcal{L}_{\text{bs}} = -\log\!\left(\frac{\pi_y \exp(z_y)}{\sum_j \pi_j \exp(z_j)}\right)$$

이 손실은 학습 시 불균형 분포($\pi$)를 명시적으로 반영하여, 테스트 시 균형 분포에 대한 일반화를 개선합니다. 로짓 앙상블은 이 디바이어싱 효과를 **보존**하는 반면, 확률 앙상블은 이를 **임의적으로 변형**시킬 수 있습니다.

### 3.4 온도 스케일링 후 일반화 (Table 4)

Pool-then-calibrate 적용 후에도 Balanced Softmax + 로짓 앙상블의 우월성이 유지됩니다:

- Balanced Softmax + avg. logits (T=1.105): **Acc 82.26%, Cal-PR AUC 0.962**
- Balanced Softmax + avg. probs (T=1.221): Acc 81.55%, Cal-PR AUC 0.960

이는 캘리브레이션 이후에도 일반화 성능 향상이 지속됨을 의미합니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

#### (1) 앙상블 설계 원칙의 재정립
기존 연구들이 로짓/확률 앙상블을 임의로 선택하던 관행에서 벗어나, **데이터 특성(균형/불균형)과 손실 함수를 고려한 Loss-Aware Ensemble 설계 원칙**을 제시합니다. 이는 불균형 학습 문헌에서 앙상블 전략 선택의 이론적 기반이 됩니다.

#### (2) 롱테일 학습 연구의 새로운 방향
롱테일 인식(recognition) 연구에서 암묵적(implicit) 앙상블 방법들(RIDE, SADE, BBN)이 명시적 deep ensemble보다 성능이 낮음을 보여줌으로써, **계산 효율성과 성능의 트레이드오프** 문제를 새롭게 제기합니다.

#### (3) 편향-분산 분해 프레임워크의 실용화
Gupta et al. [7]과 Wood et al. [8]의 이론적 프레임워크를 실제 불균형 데이터 시나리오에 적용하여, 이론과 실증 사이의 간극을 좁혔습니다.

#### (4) 캘리브레이션 지표의 중요성 부각
NLL, Brier Score 대신 Calibration PR AUC와 같은 순위 기반 지표의 중요성을 강조함으로써, 불균형 데이터에서의 올바른 평가 방법론 정립에 기여합니다.

---

### 4.2 앞으로 연구 시 고려할 점

#### (1) 대규모 불균형 데이터셋으로의 확장
현재 연구는 CIFAR 규모에 집중되어 있으므로, **iNaturalist, Places-LT, ImageNet-LT** 등 대규모 실세계 롱테일 데이터셋에서의 검증이 필요합니다.

#### (2) Transformer 기반 모델로의 확장
ResNet 계열 모델에서의 결과가 **Vision Transformer(ViT), Swin Transformer** 등에서도 동일하게 적용되는지 검증해야 합니다. Self-attention 메커니즘의 특성상 로짓 분포가 CNN과 다를 수 있습니다.

#### (3) 계산 효율적인 대안 탐색
Deep Ensemble의 $M$배 계산 비용 문제를 해결하기 위해:
- **BatchEnsemble** [42], **Monte Carlo Dropout** [40] 등 implicit ensemble 방법과의 결합
- **Knowledge Distillation**을 통한 앙상블 압축
- **Logit Adjustment Loss** [39] 등 다른 디바이어싱 손실과의 결합 효과 탐구

#### (4) 동적 롱테일 분포 (Dynamic Imbalance)
실제 환경에서는 클래스 분포가 시간에 따라 변화하는 **Online Long-Tail Learning** 또는 **Continual Learning** 시나리오에서의 앙상블 전략 연구가 필요합니다.

#### (5) 멀티모달 및 대규모 언어 모델 적용
**CLIP, GPT** 등 대규모 사전학습 모델을 롱테일 학습에 파인튜닝할 때의 앙상블 전략 연구가 필요합니다. 사전학습 모델의 다양성이 기존 ResNet 앙상블과 다를 수 있습니다.

#### (6) 공정성(Fairness)과의 연결
데이터 불균형은 종종 소수 집단(minority group)에 대한 모델 편향과 연결됩니다. 로짓 앙상블의 테일 클래스 성능 향상이 **알고리즘적 공정성** 개선에 기여할 수 있는지 탐구할 필요가 있습니다.

#### (7) 불확실성 정량화와의 통합
불균형 데이터에서의 **인식론적 불확실성(epistemic uncertainty)** 정량화와 앙상블 다양성의 관계를 더 깊이 탐구하여, 신뢰할 수 있는 AI 시스템 구축에 활용할 수 있습니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 방법 | CIFAR100-LT 정확도 | 특징 | 본 논문과의 관계 |
|------|------|-------------------|------|----------------|
| BBN (CVPR 2020) [20] | Bilateral-Branch Network (확률 앙상블) | 44.7% | 암묵적 앙상블, 확률 평균 | 명시적 로짓 앙상블에 비해 열등 |
| Balanced Softmax (NeurIPS 2020) [25] | 단일 모델, BS Loss | 46.1% | 레이블 분포 이동 수정 | 본 논문의 핵심 손실 함수 |
| RIDE (ICLR 2021) [37] | 다중 분류 헤드, 암묵적 앙상블 | 48.0% | 분포 인식 전문가 라우팅 | 명시적 앙상블보다 4% 낮음 |
| SADE (NeurIPS 2022) [19] | 자기 지도 다중 전문가, 로짓 앙상블 | 49.8% | 테스트 무관 학습 | 명시적 앙상블보다 2.2% 낮음 |
| BSCE (CVPR 2023) [18] | 균형 캘리브레이션 전문가 곱 | - | 캘리브레이션 중심 | 로짓 앙상블과의 결합 가능성 |
| **본 논문 (NeurIPS 2023)** | **Logit Ensemble + Balanced Softmax** | **52.0%** | **명시적 앙상블, Loss-Aware** | **SOTA** |

### 비교 분석 시사점

1. **암묵적 앙상블의 한계**: RIDE, SADE, BBN 등 암묵적 앙상블 방법들이 계산 효율성을 추구하지만, 명시적 deep ensemble의 성능을 따라잡지 못함
2. **손실 함수의 중요성**: 단순히 앙상블 크기를 늘리는 것보다, **적절한 손실 함수 선택이 더 중요**
3. **향후 방향**: 대규모 사전학습 모델(Foundation Model)을 활용한 롱테일 학습에서도 유사한 원칙이 적용될 가능성이 높음

---

## 참고 자료

**논문 원문 (주 참고 자료)**:
- **Buchanan, E.K., Pleiss, G., Wang, Y., Cunningham, J.P.** (2023). *The Effects of Ensembling on Long-Tailed Data*. NeurIPS 2023. [제공된 PDF]

**논문 내 인용 문헌 (직접 참조)**:
- [1] Lakshminarayanan et al. (2017). *Simple and scalable predictive uncertainty estimation using deep ensembles*. NeurIPS.
- [7] Gupta et al. (2022). *Ensembles of classifiers: a bias-variance perspective*. TMLR.
- [8] Wood et al. (2023). *A unified theory of diversity in ensemble learning*. arXiv:2301.03962.
- [14] Ashukha et al. (2020). *Pitfalls of in-domain uncertainty estimation and ensembling in deep learning*. arXiv:2002.06470.
- [19] Zhang et al. (2022). *Self-supervised aggregation of diverse experts for test-agnostic long-tailed recognition*. NeurIPS.
- [20] Zhou et al. (2020). *BBN: Bilateral-branch network with cumulative learning for long-tailed visual recognition*. CVPR.
- [24] Zhang et al. (2023). *Deep long-tailed learning: A survey*. IEEE TPAMI.
- [25] Ren et al. (2020). *Balanced meta-softmax for long-tailed visual recognition*. NeurIPS.
- [34] Rahaman et al. (2021). *Uncertainty quantification and deep ensembles*. NeurIPS.
- [37] Wang et al. (2021). *Long-tailed recognition by routing diverse distribution-aware experts*. ICLR.
- [39] Menon et al. (2020). *Long-tail learning via logit adjustment*. arXiv:2007.07314.
