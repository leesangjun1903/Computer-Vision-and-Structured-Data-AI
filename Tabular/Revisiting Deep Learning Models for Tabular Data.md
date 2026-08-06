# Revisiting Deep Learning Models for Tabular Data

> **참고 자료:**
> - Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). *Revisiting Deep Learning Models for Tabular Data*. NeurIPS 2021. arXiv:2106.11959v5
> - 논문 내 인용 문헌 전체 (본문 References 섹션 참조)
> - GitHub: https://github.com/yandex-research/tabular-dl-revisiting-models

---

## 1. Executive Summary (10문장 이내)

1. 본 논문은 테이블형 데이터(tabular data)를 위한 딥러닝 모델들이 서로 다른 벤치마크와 프로토콜로 평가되어 왔다는 문제의식에서 출발한다.
2. 저자들은 기존 DL 아키텍처 패밀리를 체계적으로 검토하고, 두 가지 강력한 기본 모델을 제안한다.
3. 첫 번째는 **ResNet 기반 아키텍처**로, 기존 문헌에서 간과되어 온 강력한 베이스라인임을 실증한다.
4. 두 번째는 **FT-Transformer(Feature Tokenizer + Transformer)**로, 모든 피처를 임베딩으로 변환한 뒤 Transformer 레이어를 적용하는 구조이다.
5. 동일한 훈련·튜닝 프로토콜 하에 11개 공개 데이터셋에서 9개 모델을 비교 평가하였다.
6. FT-Transformer는 대부분의 태스크에서 최고 성능을 기록하며 가장 보편적인 DL 모델임을 보인다.
7. ResNet은 단순성에도 불구하고 어떤 경쟁 모델도 일관되게 능가하지 못하는 강력한 베이스라인으로 확인된다.
8. GBDT(XGBoost, CatBoost)와의 비교에서, DL 모델이 일부 태스크에서 우위를 보이지만 보편적 우월성은 입증되지 않는다.
9. FT-Transformer는 GBDT가 ResNet을 능가하는 태스크에서 특히 강점을 발휘하는 "보편적 모델"의 특성을 보인다.
10. 저자들은 향후 테이블형 DL 연구에서 본 논문의 두 모델을 베이스라인으로 활용할 것을 권고한다.

### 1-1. 연구의 목적과 필요성

| 문제 | 설명 |
|------|------|
| **벤치마크 부재** | ImageNet(CV), GLUE(NLP)와 달리 테이블형 DL에는 표준 벤치마크가 없음 (p.1) |
| **비교 불가 상태** | 기존 논문들이 서로 다른 데이터셋과 프로토콜을 사용하여 공정 비교가 불가 (p.1) |
| **약한 베이스라인** | MLP가 주요 베이스라인이나 강력한 경쟁자가 되지 못함 (p.2) |
| **GBDT와의 관계 불명확** | DL이 GBDT를 실질적으로 능가하는지 불분명 (p.1) |

**필요성:** 연구 커뮤니티가 신뢰할 수 있는 공정한 비교 환경과, 실용적으로 활용 가능한 강력한 베이스라인 모델이 필요하다.

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 근거 | 위치 |
|-----------|------|------|
| ResNet은 강력한 베이스라인이다 | 11개 데이터셋 중 어떤 DL 모델도 ResNet을 일관되게 능가하지 못함; 평균 랭크 3.3 | Table 2, p.7 |
| FT-Transformer가 대부분 태스크에서 최고 성능 | 평균 랭크 1.8; 11개 중 다수 데이터셋에서 top 결과 | Table 2, p.7 |
| FT-Transformer는 더 보편적인 모델이다 | GBDT가 ResNet을 이기는 태스크에서 FT-T가 격차를 좁힘 | Section 4.6, p.8 |
| DL과 GBDT 사이에 보편적 우월자 없음 | 튜닝 후 GBDT가 CA, AD, YA에서 DL보다 우수 | Table 4, p.8 |
| Feature bias가 FT-Transformer 성능에 필수적 | bias 제거 시 성능 저하 확인 | Table 5, p.9 |
| Attention map이 효율적인 feature importance 추정 가능 | Permutation Test와 rank correlation 비교에서 유사한 성능 | Table 6, p.10 |
| 하이퍼파라미터 튜닝이 MLP/ResNet을 경쟁력 있게 만든다 | Optuna 기반 튜닝 후 단순 모델 성능 향상 | Section 4.4, p.7 |

### 2-1. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

#### 해결하고자 하는 문제
- 테이블형 DL 모델들 간의 **공정하고 체계적인 비교 부재**
- **신뢰할 수 있는 강력한 베이스라인 모델의 부재**
- DL과 GBDT 중 어느 것이 더 나은지에 대한 **불명확한 결론**

#### 제안하는 방법 (수식 포함)

**① MLP (기존 베이스라인, Eq. 1, p.3):**

$$\text{MLP}(x) = \text{Linear}(\text{MLPBlock}(\ldots(\text{MLPBlock}(x))))$$

$$\text{MLPBlock}(x) = \text{Dropout}(\text{ReLU}(\text{Linear}(x)))$$

**② ResNet (제안 강화 베이스라인, Eq. 2, p.3):**

$$\text{ResNet}(x) = \text{Prediction}(\text{ResNetBlock}(\ldots(\text{ResNetBlock}(\text{Linear}(x)))))$$

$$\text{ResNetBlock}(x) = x + \text{Dropout}(\text{Linear}(\text{Dropout}(\text{ReLU}(\text{Linear}(\text{BatchNorm}(x))))))$$

$$\text{Prediction}(x) = \text{Linear}(\text{ReLU}(\text{BatchNorm}(x)))$$

**③ FT-Transformer Feature Tokenizer (p.4):**

수치형 피처 $j$의 임베딩:
$$T_j^{(\text{num})} = b_j^{(\text{num})} + x_j^{(\text{num})} \cdot W_j^{(\text{num})} \in \mathbb{R}^d$$

범주형 피처 $j$의 임베딩:
$$T_j^{(\text{cat})} = b_j^{(\text{cat})} + e_j^T W_j^{(\text{cat})} \in \mathbb{R}^d$$

전체 토큰 행렬:
$$T = \text{stack}\left[T_1^{(\text{num})}, \ldots, T_{k^{(\text{num})}}^{(\text{num})}, T_1^{(\text{cat})}, \ldots, T_{k^{(\text{cat})}}^{(\text{cat})}\right] \in \mathbb{R}^{k \times d}$$

여기서 $e_j^T$는 범주형 피처에 대한 one-hot 벡터이고, $b_j$는 피처별 bias이다.

**④ FT-Transformer Transformer 레이어 적용 (p.4):**

$$T_0 = \text{stack}([\text{CLS}],\ T), \quad T_i = F_i(T_{i-1})$$

$$\hat{y} = \text{Linear}(\text{ReLU}(\text{LayerNorm}(T_L^{[\text{CLS}]})))$$

**⑤ FT-Transformer 전체 구조 (Supplementary E.1, p.17):**

$$\text{FT-Transformer}(x) = \text{Prediction}(\text{Block}(\ldots(\text{Block}(\text{AppendCLS}(\text{FeatureTokenizer}(x))))))$$

$$\text{Block}(x) = \text{ResidualPreNorm}(\text{FFN},\ \text{ResidualPreNorm}(\text{MHSA},\ x))$$

$$\text{ResidualPreNorm}(\text{Module},\ x) = x + \text{Dropout}(\text{Module}(\text{Norm}(x)))$$

$$\text{FFN}(x) = \text{Linear}(\text{Dropout}(\text{Activation}(\text{Linear}(x))))$$

**⑥ Attention Map 기반 Feature Importance (Section 5.3, p.10):**

$$p = \frac{1}{n_{\text{samples}}} \sum_i p_i, \quad p_i = \frac{1}{n_{\text{heads}} \times L} \sum_{h,l} p_{ihl}$$

여기서 $p_{ihl}$은 $l$번째 레이어, $h$번째 헤드의 [CLS] 토큰 attention map이다.

**⑦ 회귀 타깃 표준화 (Supplementary B.2, p.14):**

$$y_{\text{new}} = \frac{y_{\text{old}} - \text{mean}(y_{\text{train}})}{\text{std}(y_{\text{train}})}$$

**⑧ 합성 실험 (Section 5.1, p.9):**

$$x \sim \mathcal{N}(0, I_k), \quad y = \alpha \cdot f_{\text{GBDT}}(x) + (1 - \alpha) \cdot f_{\text{DL}}(x)$$

여기서 $f_{\text{GBDT}}$는 30개 랜덤 결정 트리의 평균 예측, $f_{\text{DL}}$은 랜덤 초기화된 3-layer MLP이다.

#### 모델 구조 비교

| 모델 | 핵심 구조 | 특징 |
|------|-----------|------|
| **ResNet** | BatchNorm → Linear → ReLU → Dropout → Linear → Dropout + 잔차 연결 | 단순하고 빠름, 강력한 베이스라인 |
| **FT-Transformer** | Feature Tokenizer → [CLS] 추가 → L개 PreNorm Transformer 블록 → [CLS] 예측 | 범주형·수치형 피처 통합 임베딩, MHSA 적용 |
| **MLP** | Linear → ReLU → Dropout 반복 | 가장 단순한 베이스라인 |

#### 성능 향상

- **FT-Transformer** 평균 랭크: **1.8** (전체 DL 모델 중 최고) — Table 2, p.7
- **ResNet** 평균 랭크: **3.3** (단순 모델 중 최고) — Table 2, p.7
- 앙상블 시 FT-Transformer가 NODE를 능가 (Table 3, p.7)
- 기본 하이퍼파라미터(default) FT-Transformer가 대부분 튜닝된 GBDT를 능가 (Table 4, p.8)

#### 한계

| 한계 | 설명 | 위치 |
|------|------|------|
| 높은 계산 비용 | MHSA의 피처 수에 대한 $O(k^2)$ 복잡도; Yahoo 데이터셋에서 ResNet 대비 13.8배 오버헤드 | Table 10, p.16 |
| 피처 수 제약 | 피처 수가 매우 많은 경우 적용 곤란 (하드웨어·시간 예산에 의존) | Section 3.3, p.5 |
| CO₂ 배출 우려 | 테이블형 문제가 보편적이므로 FT-Transformer의 광범위한 사용은 탄소 배출 증가 초래 가능 | Section 3.3, p.5 |
| 벤치마크 편향 | 사용된 벤치마크가 다소 "DL-friendly" 문제로 편향되어 있음을 저자들 스스로 인정 | Section 4.5, p.8 |

---

## 3. 각 주장에 페이지/Figure/Table 번호 표시

| 주장 | 근거 위치 |
|------|-----------|
| ResNet이 강력한 베이스라인 | Table 2 (p.7), Section 4.4 (p.7) |
| FT-Transformer가 최고 DL 모델 | Table 2 (p.7), Table 3 (p.7) |
| FT-Transformer의 보편성 | Section 4.6 (p.8), Figure 3 (p.9), Table 4 (p.8) |
| DL vs GBDT: 보편적 우월자 없음 | Table 4 (p.8), Section 4.5 (p.7-8) |
| Feature bias의 필요성 | Table 5 (p.9), Table 23 (p.23) |
| Feature Tokenizer 구조 | Figure 2(a) (p.4), Eq. Feature Tokenizer (p.4) |
| FT-Transformer 전체 구조 | Figure 1 (p.4), Figure 2(b) (p.4) |
| Attention map의 feature importance | Table 6 (p.10), Section 5.3 (p.10) |
| 튜닝 시간 예산 분석 | Table 10 (p.16), Table 11 (p.17) |
| 합성 실험 | Figure 3 (p.9), Section 5.1 (p.8-9) |

---

## 4. 저자 보고 결과 vs. 내 해석 분리

### 저자가 직접 보고한 결과

| 항목 | 저자 보고 내용 |
|------|----------------|
| FT-Transformer 평균 랭크 | **1.8 (std: 1.2)** — 11개 데이터셋 기준 (Table 2) |
| ResNet 평균 랭크 | **3.3 (std: 1.8)** — 어떤 DL 모델도 일관되게 능가 못함 (Table 2) |
| GBDT 우위 데이터셋 | California Housing (CA), Adult (AD), Yahoo (YA) — 튜닝 후 (Table 4) |
| 기본 FT-T vs 기본 GBDT | CA, AD를 제외한 대부분에서 FT-T 앙상블이 기본 GBDT 앙상블 능가 (Table 4) |
| 학습 시간 오버헤드 | FT-Transformer가 ResNet 대비 평균 ~2-3배, Yahoo에서 13.8배 (Table 10) |
| Attention map rank correlation | CA: 0.81, HI: 0.91, YE: 0.92 등 (Table 6) |

### 내 해석 (논문이 직접 명시하지 않은 함의)

| 항목 | 내 해석 |
|------|---------|
| FT-Transformer의 보편성 원인 | Self-attention이 피처 간 상호작용을 명시적으로 모델링하여 GBDT-friendly 함수(비선형 분기 구조)에도 적응 가능한 것으로 보임 |
| 벤치마크 편향 문제 | 저자가 인정하듯 11개 데이터셋 중 금융·물리·검색 데이터가 많아 산업 현장의 고차원 범주형 피처 중심 데이터는 충분히 커버되지 않음 |
| ResNet의 실용적 가치 | 단순성과 안정성 측면에서 ResNet은 실무에서 첫 번째 시도 모델로 적합하며, 튜닝 비용 대비 성능이 우수함 |
| Feature bias의 역할 | 각 피처에 고유한 bias를 부여함으로써 피처 간 스케일 차이를 임베딩 공간에서 흡수하는 효과로 해석 가능 |

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

| 구분 | 내용 | 위치 |
|------|------|------|
| ⚠️ **단일 분할 사용** | 각 데이터셋마다 단 하나의 train/val/test 분할만 사용; 분할 방식에 따른 성능 변동성 미검증 | Section 4.2, p.5 |
| ⚠️ **GrowNet 비교 불완전** | GrowNet은 다중 클래스 지원 안 함 → HE, JA, AL, CO 결과 없음 (Table 2에 "–") | Table 2, p.7 |
| ⚠️ **NODE의 Helena/ALOI** | NODE가 Helena, ALOI에서 스케일링 불가로 기본 설정만 사용 → 공정 비교 불가 | Supplementary F.6, p.21 |
| ⚠️ **FT-Transformer Yahoo** | Yahoo 데이터셋에서 FT-T는 기본 설정 결과를 보고 (튜닝 미수행) → 다른 모델과 비교 조건 상이 | Supplementary E.2, p.18 |
| ⚠️ **ALOI에서 GBDT 결과 없음** | XGBoost, CatBoost가 ALOI(1000 클래스)에서 극도로 느린 학습으로 튜닝 불가, "–" 처리 | Table 4, p.8 |
| ⚠️ **합성 실험의 한계** | 합성 실험(Section 5.1)은 특정 방식으로 생성된 $f_{\text{GBDT}}, f_{\text{DL}}$에 의존; 실제 데이터의 다양성 미반영 | Section 5.1, p.8-9 |
| ⚠️ **통계 검정 기준** | Wilcoxon one-sided test ($p = 0.01$) 사용; 다중 비교 문제(multiple comparison problem) 미보정 | Supplementary C, p.14 |
| ⚠️ **벤치마크 편향 자인** | 저자 스스로 "benchmark is slightly biased towards DL-friendly problems" 인정 | Section 4.5, p.8 |
| ⚠️ **비교 불가 수치** | Epsilon 데이터셋에서 전처리를 미적용(raw features)하여 다른 데이터셋과 전처리 조건 상이 | Section 4.3, p.6 |

---

## 6. 문서가 답하지 않는 질문

| 미답 질문 | 중요도 |
|-----------|--------|
| FT-Transformer가 왜 GBDT-friendly 태스크에서 강한지에 대한 이론적 설명이 부재 | 높음 |
| 피처 수가 수천 이상인 고차원 데이터(예: 유전체, 텍스트 피처화)에서의 성능은? | 높음 |
| 준지도 학습(semi-supervised) 또는 사전 학습(pretraining) 적용 시 성능 변화는? | 높음 |
| 데이터 수가 매우 적은 소규모 데이터셋(수백~수천 샘플)에서의 동작은? | 높음 |
| 범주형 피처 수가 많거나 카디널리티가 매우 높은 경우(예: 수백만 카테고리)의 처리는? | 중간 |
| FT-Transformer의 최적 스케일링 법칙(scaling law)은 무엇인가? | 중간 |
| PostNorm Transformer가 PreNorm보다 실제로 더 좋은지 테이블 DL에서 체계적 비교는? | 중간 |
| 모델 불확실성(uncertainty quantification) 및 캘리브레이션(calibration) 성능은? | 중간 |
| 데이터 증강(data augmentation) 또는 정규화 기법 추가 시 성능 향상 폭은? | 낮음 |
| FT-Transformer를 더 가벼운 모델로 지식 증류(knowledge distillation) 시의 효과는? | 낮음 |

---

## 7. 가장 중요한 그림/표 5개 해석

### ① Figure 1 — FT-Transformer 전체 아키텍처 (p.4)

```
입력 x → Feature Tokenizer → T (k×d 임베딩) → [CLS] 토큰 추가 → T₀
→ Transformer 블록 L개 적용 → T_L → [CLS] 표현 → Predict → ŷ
```

**해석:** Feature Tokenizer는 모든 피처(수치형·범주형)를 동일한 $d$차원 임베딩으로 변환한다. 이 구조는 NLP의 BERT와 유사하게 [CLS] 토큰을 도입하여 전체 입력의 표현을 집약한다. 핵심 혁신은 피처 자체가 "토큰"이 되어 Self-Attention이 **피처 간 상호작용**을 자동으로 학습한다는 점이다. 이는 기존 MLP/ResNet이 피처 상호작용을 암묵적으로만 학습하는 것과 대비된다.

---

### ② Figure 2 — Feature Tokenizer와 Transformer 블록 상세 (p.4)

**(a) Feature Tokenizer:**
- 수치형: $T_j^{(\text{num})} = b_j^{(\text{num})} + x_j^{(\text{num})} \cdot W_j^{(\text{num})}$ (스칼라 값을 벡터로 확장)
- 범주형: $T_j^{(\text{cat})} = b_j^{(\text{cat})} + e_j^T W_j^{(\text{cat})}$ (룩업 테이블 방식)

**(b) Transformer 블록:**
- PreNorm 방식: Norm → MHSA → Add, Norm → FFN → Add

**해석:** 수치형 피처를 임베딩하는 방식(element-wise 곱)이 핵심이다. 기존 AutoInt가 이 bias 항($b_j$)을 포함하지 않은 반면, FT-Transformer는 이를 포함하여 성능 향상을 달성한다(Table 5). PreNorm 사용은 추가적인 learning rate warmup 없이도 안정적 학습을 가능하게 한다.

---

### ③ Table 2 — DL 모델 비교 결과 (p.7)

| 모델 | 평균 랭크 |
|------|-----------|
| FT-Transformer | **1.8** |
| ResNet | 3.3 |
| NODE | 3.9 |
| DCN2 | 4.7 |
| MLP | 4.8 |
| ... | ... |
| TabNet | 7.5 |

**해석:** FT-Transformer의 압도적 1위와 ResNet의 안정적 2위가 핵심이다. 특히 주목할 점은 TabNet, AutoInt, NODE 등 테이블형 DL을 위해 특별히 설계된 모델들이 단순한 ResNet보다 평균적으로 낮은 랭크를 기록한다는 것이다. 이는 "특수 목적 아키텍처 > 범용 아키텍처"라는 직관이 꼭 맞지 않음을 보여준다. 또한 표준편차를 함께 보면 FT-Transformer(std: 1.2)가 가장 안정적이다.

---

### ④ Figure 3 — 합성 실험: FT-Transformer의 보편성 (p.9)

$$y = \alpha \cdot f_{\text{GBDT}}(x) + (1-\alpha) \cdot f_{\text{DL}}(x)$$

- $\alpha = 0$: 순수 DL-friendly 태스크
- $\alpha = 1$: 순수 GBDT-friendly 태스크

**해석:**
- ResNet은 $\alpha$가 증가할수록 RMSE가 급격히 상승 (DL-friendly 태스크 특화)
- CatBoost는 $\alpha$가 감소할수록 RMSE 상승 (GBDT-friendly 태스크 특화)
- **FT-Transformer는 전 범위에서 낮고 안정적인 RMSE 유지** (보편성 입증)

이 실험은 FT-Transformer가 GBDT-friendly 함수(트리 구조 기반 분기 패턴)를 근사하는 능력이 ResNet보다 뛰어남을 합성 데이터로 보여준다. Section 4.6의 실제 데이터 관찰(FT-T가 GBDT가 ResNet을 이기는 데이터셋에서 특히 강함)과 일관된 결과이다.

---

### ⑤ Table 4 — DL 앙상블 vs GBDT 앙상블 비교 (p.8)

| 조건 | 핵심 관찰 |
|------|-----------|
| 기본 설정 FT-T | CA, AD 제외 대부분에서 기본 GBDT 능가 |
| 튜닝 후 GBDT | CA( $\downarrow$ 0.423), AD( $\uparrow$ 0.874), YA( $\downarrow$ 0.732+)에서 DL 능가 |
| 튜닝 후 FT-T | ALOI( $\uparrow$ 0.967), CO( $\uparrow$ 0.973)에서 GBDT 능가 |

**해석:** 이 표는 두 가지 중요한 메시지를 전달한다. (1) **"out-of-the-box" 실용성**: 기본 설정 FT-T가 기본 GBDT를 대부분에서 능가하므로, 실무에서 빠른 배포 시 FT-T가 유리할 수 있다. (2) **"no free lunch"**: 충분히 튜닝된 환경에서는 특정 데이터셋에 GBDT가 여전히 강점을 보인다. ALOI(1000 클래스)처럼 GBDT가 아예 튜닝 불가능한 경우는 비교 자체가 불공정하다는 점도 주의해야 한다(⚠️ 비교 불가 수치).

---

## 8. 결론 및 후속 연구

### 저자들이 제시한 시사점 및 후속 연구 계획

**저자 제시 시사점 (Section 6, p.10):**
1. ResNet-like 아키텍처가 간과된 강력한 베이스라인임을 확인 → 미래 연구의 비교 기준으로 활용 권고
2. FT-Transformer가 가장 보편적인 DL 솔루션이며, 대부분의 태스크에서 최고 성능 달성
3. GBDT와 DL 사이에 보편적 우월자가 없으며, 두 접근법 모두 의미 있음
4. 두 모델의 코드 오픈소스화로 재현 가능성 및 후속 연구 기반 제공

**저자 언급 향후 연구 방향:**
- 효율적 MHSA 근사(Tay et al., 2020)를 통한 피처 수 확장성 개선
- FT-Transformer에서 더 단순한 아키텍처로의 지식 증류(knowledge distillation)
- DL이 GBDT를 능가하지 못하는 데이터셋에 집중한 연구

---

### 8-1. 모델의 일반화 성능 향상 가능성

본 논문이 제시하는 두 모델의 일반화 성능 향상과 관련하여 다음과 같은 측면을 분석할 수 있다.

#### (1) FT-Transformer의 일반화 강점 원인 분석

$$\text{Self-Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Transformer의 Self-Attention 메커니즘은 **피처 간 상호작용을 동적으로 가중치화**하여 학습한다. 이는 데이터셋마다 중요한 피처 조합이 다를 때 자동으로 적응하는 능력으로 이어진다. 이것이 FT-Transformer의 보편적 일반화 성능의 구조적 원인으로 추정된다.

#### (2) 일반화 성능 향상을 위한 논문 내 관찰

| 기법 | 일반화 효과 | 위치 |
|------|-------------|------|
| **Feature bias ($b_j$)** | 피처별 bias로 각 피처의 고유 특성 반영 → 다양한 데이터셋에 적응 | Table 5, p.9 |
| **PreNorm** | 추가 초기화 없이도 안정적 학습, 범용 적용 가능 | Supp. E.1, p.17 |
| **Attention dropout** | 항상 유익하다고 관찰됨 → 과적합 방지에 기여 | Supp. E.1, p.17 |
| **AdamW optimizer** | Weight decay 분리(Loshchilov & Hutter, 2019)로 정규화 효과 | Section 4.3, p.6 |
| **앙상블** | FT-Transformer가 ResNet보다 앙상블 효과가 더 큼 (Fort et al., 2020 인용) | Section 4.5, p.7 |

#### (3) 일반화 성능 향상의 미탐구 영역 (내 해석)

- **사전 학습(Pre-training):** 저자들은 의도적으로 pretraining을 제외했으나(Section 4.1), 레이블 없는 테이블 데이터에 대한 자기지도 사전 학습은 일반화 성능을 크게 향상시킬 가능성이 있다
- **데이터 증강:** 테이블형 데이터에 특화된 augmentation(피처 마스킹, 노이즈 추가 등) 미탐구
- **소규모 데이터 일반화:** 본 논문의 데이터셋은 최소 20K 샘플 이상; 수백 샘플 규모에서의 일반화 성능은 미검증
- **분포 외 일반화(OOD Generalization):** 학습-테스트 분포가 다를 때의 견고성 미검증

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> **주의:** 아래 분석은 제공된 논문(arXiv:2106.11959v5, 최종 업데이트 2023년 10월)의 내용과, 해당 논문 이후 발표된 주요 연구들에 대한 일반적 지식을 바탕으로 합니다. 개별 후속 논문의 구체적 수치는 원본 논문을 직접 확인하시기 바랍니다.

#### 본 논문 이후 주요 연구 흐름

| 연구 방향 | 대표 연구 | 본 논문과의 관계 |
|-----------|-----------|-----------------|
| **수치형 피처 인코딩 개선** | "On Embeddings for Numerical Features in Tabular Deep Learning" (Gorishniy et al., 2022, 동일 저자) | FT-Transformer의 Feature Tokenizer를 확장; piecewise-linear, periodic 인코딩 제안 |
| **자기지도 사전 학습** | SCARF (Bahri et al., 2022), TabNet + self-supervised | 본 논문이 제외한 pretraining의 효과 탐구 |
| **GBDT+DL 하이브리드** | NODE+, TabPFN | DL과 GBDT의 상호보완적 결합 |
| **대규모 테이블 FM** | TabPFN (Hollmann et al., 2022) | In-context learning으로 소규모 데이터 일반화 |
| **피처 중요도 학습** | 본 논문 Section 5.3의 연장선 | Attention map 활용 해석 가능성 |

#### 본 논문이 이후 연구에 미친 영향

1. **표준 베이스라인 확립:** ResNet과 FT-Transformer는 이후 테이블형 DL 논문에서 필수 비교 대상이 되었다
2. **공정 비교 문화 정착:** 동일 프로토콜(Optuna 튜닝, 15 시드, 단일 분할) 활용이 후속 연구의 기준이 됨
3. **Feature Tokenizer 패러다임:** 수치형 피처를 임베딩으로 변환하는 접근법이 후속 연구(예: 수치 임베딩 방법론)의 출발점이 됨
4. **"No free lunch" 재확인:** DL이 GBDT를 무조건 능가하지 않는다는 결론이 이후 연구들의 현실적 출발점 제공

#### 앞으로 연구 시 고려할 점

| 고려 사항 | 구체적 권고 |
|-----------|-------------|
| **벤치마크 다양성** | 본 논문의 11개 데이터셋 외에도 고차원 희소 피처, 불균형 레이블, 시계열 테이블 등 포함 필요 |
| **소규모 데이터 성능** | FT-Transformer는 대규모 데이터에 강하지만, 의료·금융 도메인의 소규모 데이터에서 별도 검증 필요 |
| **사전 학습 통합** | 본 논문이 의도적으로 제외한 pretraining + FT-Transformer 조합 탐구가 유망 |
| **효율적 Attention** | MHSA의 $O(k^2)$ 복잡도 문제 해결을 위한 Linear Attention 등 적용 (Tay et al., 2020 참조) |
| **범주형 피처 처리** | 고카디널리티 범주형 피처에 대한 더 정교한 임베딩 전략 필요 |
| **OOD 일반화** | 훈련-배포 분포 차이가 있는 실세계 시나리오에서의 견고성 평가 |
| **재현성 기준 강화** | 단일 데이터 분할의 한계를 극복하기 위한 k-fold 교차 검증 도입 권고 |
| **탄소 효율성** | FT-Transformer의 높은 계산 비용을 고려한 효율적 변형 모델 연구 |

---

**[참고 자료 목록]**
1. Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). *Revisiting Deep Learning Models for Tabular Data*. NeurIPS 2021. arXiv:2106.11959v5
2. Vaswani, A. et al. (2017). *Attention is All You Need*. NIPS 2017
3. He, K. et al. (2015). *Deep Residual Learning for Image Recognition*. arXiv:1512.03385
4. Popov, S., Morozov, S., & Babenko, A. (2020). *Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data*. ICLR 2020
5. Arik, S. O., & Pfister, T. (2020). *TabNet: Attentive Interpretable Tabular Learning*. arXiv:1908.07442v5
6. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. SIGKDD 2016
7. Prokhorenkova, L. et al. (2018). *CatBoost: Unbiased Boosting with Categorical Features*. NeurIPS 2018
8. Akiba, T. et al. (2019). *Optuna: A Next-Generation Hyperparameter Optimization Framework*. KDD 2019
9. Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR 2019
10. Tay, Y. et al. (2020). *Efficient Transformers: A Survey*. arXiv:2009.06732
11. Song, W. et al. (2019). *AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks*. CIKM 2019
12. Wang, R. et al. (2020). *DCN V2: Improved Deep & Cross Network*. arXiv:2008.13535
13. Sundararajan, M. et al. (2017). *Axiomatic Attribution for Deep Networks*. ICML 2017
14. Fort, S. et al. (2020). *Deep Ensembles: A Loss Landscape Perspective*. arXiv:1912.02757
15. GitHub 소스코드: https://github.com/yandex-research/tabular-dl-revisiting-models
