# Network On Network for Tabular Data Classification in Real-world Applications

> **참고 자료**
> - Luo, Y., Zhou, H., Tu, W.-W., Chen, Y., Dai, W., & Yang, Q. (2020). *Network On Network for Tabular Data Classification in Real-world Applications*. SIGIR '20. arXiv:2005.10114v2
> - 본 답변은 제공된 PDF 원문에 근거하며, 원문에서 확인되지 않는 내용은 명시적으로 표기합니다.

---

## 1. Executive Summary (10문장 이내)

NON(Network On Network)은 실세계 태뷸러(tabular) 데이터 분류의 정확도를 높이기 위해 4Paradigm에서 제안한 딥러닝 모델이다.  
기존 Wide & Deep, DeepFM, xDeepFM, AutoInt 등의 모델은 서로 다른 필드(field)의 임베딩을 직접 합산·결합하여 **intra-field 정보**(같은 필드 내 특징들이 동일 필드에 속한다는 정보)를 무시한다.  
또한 여러 연산(operation)의 출력을 단순 가중합(weighted sum)으로 결합함으로써 **비선형 상호작용**을 포착하지 못하는 한계가 있다.  
NON은 이를 해결하기 위해 세 계층으로 구성된다:  

(1) **Field-wise Network** – 각 필드별 독립 DNN으로 intra-field 정보 포착,  
(2) **Across Field Network** – 데이터 기반으로 최적 연산 선택,  
(3) **Operation Fusion Network** – 선택된 연산 출력을 DNN으로 비선형 융합.  

깊은 구조로 인한 훈련 어려움을 해결하기 위해 GoogLeNet에서 영감을 받은 **보조 손실(auxiliary loss)** 기법을 모든 DNN 레이어에 적용한다.  
Criteo, Avazu, Movielens 등 6개 실세계 데이터셋 실험에서 NON은 모든 데이터셋에서 기존 SOTA 모델 대비 AUC 0.64%~0.99% 향상을 달성한다.  
정성적(t-SNE 시각화) 및 정량적(코사인 거리) 분석으로 intra-field 정보 포착 효과를 검증한다. DNN with auxiliary losses는 동일 AUC 달성 시 훈련 속도를 약 1.67배 향상시킨다.  
본 논문은 태뷸러 데이터의 고차원 범주형 필드 처리에 특화된 실용적 아키텍처를 제시한다.

### 1-1. 연구의 목적과 필요성

| 측면 | 내용 | 근거 |
|------|------|------|
| **응용 분야** | 온라인 광고, 추천 시스템, 사기 탐지, 의료 진단 등 태뷸러 데이터가 핵심 | p.1, Introduction |
| **산업적 중요성** | 오프라인 AUC 0.275% 향상 → 온라인 CTR 3.9% 향상 → 수백만 달러 수익 증가 | p.6, §4.1.3 |
| **기존 모델의 한계 ①** | intra-field 정보 미활용: 서로 다른 필드 임베딩을 구분 없이 직접 융합 | p.2, §1 |
| **기존 모델의 한계 ②** | 고정된 연산 조합(predefined operations): 데이터에 무관하게 동일 연산 사용 | p.2, Table 1 |
| **기존 모델의 한계 ③** | 비선형 상호작용 무시: 가중합(weighted sum)의 선형성으로 인해 연산 간 상호작용 소실 | p.2, §1 |

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 (실험/이론) | 위치 |
|---|-----------|-----------------|------|
| 1 | Field-wise Network이 intra-field 정보를 효과적으로 포착 | t-SNE 시각화: 처리 후 동일 필드 내 특징이 군집화됨 / 코사인 거리 최대 2 오더 증가 | Fig. 7, Table 6 |
| 2 | NON이 모든 데이터셋에서 SOTA 모델 대비 최고 AUC 달성 | 6개 데이터셋에서 DNN 대비 0.64%~0.99% AUC 향상 | Table 4 |
| 3 | Auxiliary loss가 훈련 효율성과 성능 모두 향상 | Criteo 서브셋에서 동일 AUC 도달 시 1.67× 속도 향상 | Fig. 5, Table 3 |
| 4 | 최적 연산 조합은 데이터셋마다 다름 | 7가지 조합 실험에서 단일 최적 조합 없음, AUC 격차 0.1%~0.9% | Fig. 6, Table 5 |
| 5 | 소규모 데이터는 단순 연산, 대규모 데이터는 복잡 연산 선호 | Talkshow/Social/Sports: DNN+LR만 최적; Criteo/Avazu: Attention 등 추가 | Table 5 |
| 6 | NON의 각 컴포넌트가 성능에 기여 | Ablation study: 컴포넌트 추가 시 단계적 AUC 향상 | Table 3 |

---

## 2-1. 해결 문제, 제안 방법, 모델 구조, 성능 및 한계

### 해결하고자 하는 문제

기존 태뷸러 분류 모델은 세 가지 근본적 문제를 가진다:
1. **Intra-field 정보 무시**: 'advertiser_id', 'user_id'처럼 필드가 의미론적으로 구분됨에도 불구하고, 모든 필드의 임베딩을 동등하게 직접 연결(concatenate)하여 처리
2. **고정된 연산 조합**: Table 1처럼 Wide & Deep은 항상 Linear+DNN, DeepFM은 항상 FM+DNN을 사용 → 데이터 특성 무시
3. **비선형 상호작용 소실**: 최종 예측 시 $\sigma(\mathbf{w}^T[\mathbf{h}_1, \mathbf{h}_2, \ldots])$ 형태의 가중합 사용 → 연산 간 비선형 관계 포착 불가

### 제안하는 방법 (수식 포함)

#### (1) Field-wise Network

각 필드 $i$에 대해 독립적인 DNN을 적용하여 intra-field 정보를 학습:

$$\mathbf{e}'_i = \text{DNN}_i(\mathbf{e}_i) $$

병렬 처리를 위한 배치 행렬 곱:

$$\mathbf{X} = \text{stack}([\mathbf{X}_1, \mathbf{X}_2, \ldots, \mathbf{X}_c]) \in \mathbb{R}^{c \times b \times d_1}$$

$$\mathbf{W} = \text{stack}([\mathbf{W}_1, \mathbf{W}_2, \ldots, \mathbf{W}_c]) \in \mathbb{R}^{c \times d_1 \times d_2}$$

$$\mathbf{X}' = \text{ReLU}(\text{matmul}(\mathbf{X}, \mathbf{W}) + \mathbf{b})$$

원본 임베딩과의 정제(refinement):

$$\hat{\mathbf{e}}_i = F(\mathbf{e}'_i, \mathbf{e}_i)$$

여기서 $F$는 concatenation, element-wise product, 또는 gating mechanism

#### (2) Across Field Network - Bi-Interaction

$$\mathbf{v} = \sum_{i}^{m}\sum_{j}^{m} x_i \mathbf{e}_i \odot x_j \mathbf{e}_j$$

여기서 $\odot$는 element-wise product, $x_i$는 특징값, $\mathbf{e}_i$는 $i$번째 필드 임베딩

#### (3) Operation Fusion Network

$$\mathbf{x}_{\text{ofn}} = \text{concat}([\mathbf{o}_1, \mathbf{o}_2, \ldots, \mathbf{o}_k]) \in \mathbb{R}^{\sum_i d_i}$$

$$y' = \text{DNN}_{\text{ofn}}(\mathbf{x}_{\text{ofn}})$$

#### (4) DNN with Auxiliary Losses

$i$번째 레이어의 보조 손실:

$$\ell^i_{\text{aux}} = \ell\left(\text{sigmoid}\left(\mathbf{W}^T_{\text{aux}_i} \mathbf{h}_i\right), y\right) $$

전체 손실 함수:

$$\ell = \ell(y', y) + \alpha \sum_i \ell^i_{\text{aux}} + \gamma \|\mathbf{W}\|$$

여기서 $\alpha$, $\gamma$는 하이퍼파라미터, $\|\mathbf{W}\|$는 L2 정규화 항

### 모델 구조 (Figure 3 기반)

```
[Input: Categorical + Numerical Fields]
         ↓
┌─────────────────────────────────────┐
│      Field-wise Network (Bottom)    │  ← 각 필드별 독립 DNN
│  DNN₁  DNN₂  ...  DNNₘ            │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│     Across Field Network (Middle)   │  ← 데이터 기반 연산 선택
│  LR | DNN | Self-Attention | Bi-Int │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│  Operation Fusion Network (Top)     │  ← DNN으로 비선형 융합
│  DNN_ofn                           │
└─────────────────────────────────────┘
         ↓
    [Final Prediction y']
```

### 성능 향상

| 데이터셋 | DNN (baseline) | NON | AUC 향상 |
|----------|---------------|-----|----------|
| Criteo | 0.8063 | 0.8115 | +0.64% |
| Avazu | 0.7763 | 0.7838 | +0.97% |
| Movielens | 0.6988 | 0.7057 | +0.99% |
| Talkshow | 0.8451 | 0.8533 | +0.97% |
| Social | 0.6969 | 0.7032 | +0.90% |
| Sports | 0.8506 | 0.8561 | +0.65% |

*(Table 4, p.7)*

### 한계점

- **파라미터 증가**: 필드별 독립 DNN으로 인해 파라미터 수가 $m$배 증가 가능 (논문에서 정량적 비교 미제시) ⚠️
- **하이퍼파라미터 민감성**: 연산 조합이 하이퍼파라미터로 처리되어 탐색 비용 발생
- **행동 시퀀스 데이터 미적용**: 사용자 행동 시퀀스 정보가 없는 시나리오에 한정 (p.3, §2.3)
- **트리 기반 방법 비교 없음**: 수치형 필드 중심 데이터에서의 XGBoost 등과의 공정 비교 부재
- **운영(production) 환경 지연 시간(latency) 미보고** ⚠️

---

## 3. 주장별 페이지/Figure/Table 번호

| 주장 | 위치 |
|------|------|
| 세 가지 기존 모델의 문제점 정의 | p.2, §1, Table 1 |
| Field-wise Network 수식 | p.3, Eq.(1), §3.1.1 |
| Across Field Network - Bi-Interaction 수식 | p.4, §3.1.2 |
| Operation Fusion Network 수식 | p.4-5, §3.1.3 |
| Auxiliary Loss 수식 | p.5, Eq.(2), §3.2, Fig. 4 |
| 시간 복잡도 분석 | p.5, §3.3 |
| 훈련 속도 1.67× 향상 | p.6, Fig. 5 |
| Ablation study 결과 | p.7, Table 3 |
| SOTA 비교 결과 | p.7, Table 4 |
| 데이터셋별 최적 연산 | p.8, Table 5, Fig. 6 |
| t-SNE 시각화 결과 | p.9, Fig. 7 |
| 코사인 거리 정량 분석 | p.9, Table 6 |

---

## 4. 저자 보고 결과 vs. 내 해석 분리

### 저자가 직접 보고한 결과

| 항목 | 저자 보고 내용 | 위치 |
|------|--------------|------|
| 연구 주제 | 태뷸러 데이터 분류를 위한 NON 제안 | Abstract |
| 핵심 방법 | Field-wise/Across-field/Operation fusion 3단계 구조 + auxiliary loss | §3 |
| 성능 | NON이 6개 데이터셋 모두에서 SOTA 대비 최고 AUC 달성 (0.64%~0.99% 향상) | Table 4 |
| 훈련 효율 | Auxiliary loss로 1.67× 훈련 속도 향상 | Fig. 5 |
| Intra-field 포착 | 코사인 거리 최대 2 오더(order) 증가 | Table 6 |
| 연산 선택 | LR은 모든 데이터셋에서 선택됨; 대규모 데이터는 복잡 연산 선호 | Table 5 |

### 내 해석 (원문에 없는 분석)

| 항목 | 내 해석 |
|------|---------|
| **일반화 타당성** | 3개 데이터셋이 자사 고객 데이터(Talkshow, Social, Sports)로, 공개 재현이 불가능하여 독립적 검증에 제한이 있음 ⚠️ |
| **성능 향상의 원인** | 단순히 모델 용량(capacity) 증가가 아닌 intra-field 정보와 비선형 융합의 복합 효과로 해석됨 (Ablation Table 3이 이를 지지) |
| **Auxiliary loss의 역할** | GoogLeNet의 중간 감독(intermediate supervision)과 유사하나, 모든 레이어에 적용하는 점은 더 적극적 정규화로 볼 수 있음 |
| **데이터 규모-연산 복잡도 관계** | 대규모 데이터에서 복잡 연산이 과적합 없이 성능 향상을 가져오는 것은 데이터양이 정규화 효과를 대신하는 것으로 해석 가능 |
| **공정성 우려** | Wide & Deep의 원본은 전문가 특징(handcrafted features)을 사용하는데, 이 논문은 원본 특징으로 대체하여 불리한 조건에서 비교 → Wide & Deep이 실제보다 낮게 평가될 수 있음 |

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치 ⚠️

| 문제 유형 | 구체적 내용 |
|-----------|-------------|
| **재현 불가 데이터** ⚠️ | Talkshow, Social, Sports는 자사 고객 비공개 데이터 → 독립 재현 실험 불가 |
| **신뢰구간 미제시** ⚠️ | 모든 AUC 수치가 단일 값으로만 보고됨; 표준편차, 신뢰구간, 통계적 유의성 검정 없음 |
| **Wide & Deep 불공정 비교** ⚠️ | 원본 Wide & Deep은 전문가 설계 특징 사용; 이 논문은 원본 특징으로 대체 → 직접 비교 불가 |
| **FFM 하이퍼파라미터 분리** ⚠️ | FFM만 탐색 공간이 다름(임베딩 차원=4 고정) → 다른 모델과 동등 조건 비교 아님 |
| **1.67× 속도 향상의 범위** ⚠️ | Criteo 서브셋에서만 측정; 다른 데이터셋 및 전체 훈련 조건에서의 결과 미보고 |
| **랜덤 서치 60회의 통계적 충분성** ⚠️ | 60회 랜덤 서치의 결과가 최적에 수렴했는지 검증 없음 |
| **파라미터 수 비교 미제시** ⚠️ | 모델별 파라미터 수, 추론 시간, 메모리 사용량 비교 없음 |

---

## 6. 문서가 답하지 않는 질문

| 질문 | 설명 |
|------|------|
| **추론 지연 시간(latency)은?** | 실제 배포 환경에서의 추론 속도 미보고 |
| **파라미터 수 대비 성능은?** | 공정한 모델 용량 비교가 없어 성능 향상이 순수 설계 개선인지 용량 증가 때문인지 불분명 |
| **다중 클래스 분류 적용 가능성은?** | 이진 분류(CTR, 클릭 예측)에 집중; 다중 클래스나 회귀 문제로의 확장 미논의 |
| **희소 필드가 매우 많을 때의 거동은?** | 필드 수가 수백~수천 개인 극단적 경우 field-wise network의 확장성 미검토 |
| **연산 선택의 자동화 가능성은?** | 현재 연산 조합을 하이퍼파라미터로 탐색; NAS(Neural Architecture Search) 등 완전 자동화 방안 미논의 |
| **사전 훈련(pre-training) 효과는?** | NFM에서 pre-training 없이 비교하는 등 사전 훈련 전략 비교 부재 |
| **cold-start 문제는?** | 새로운 사용자/광고주(ID) 등장 시 임베딩 처리 방법 미언급 |
| **비교뷸러 데이터(이미지, 텍스트 포함 혼합)에 적용 가능한가?** | 순수 태뷸러에 국한 |

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.2) - 태뷸러 데이터 예시

온라인 광고 데이터에서 `click`, `user_id(c)`, `advertiser_id(c)`, `age(n)`, `salary(n)`, `occupation(c)` 필드를 보여준다. 이 그림은 논문의 핵심 동기를 시각화한다: `user_id`와 `advertiser_id`는 의미론적으로 완전히 다른 개체(사용자 vs. 광고주)이므로, 이들의 임베딩을 구분 없이 합치는 것은 정보 손실을 야기한다. 'n'과 'c'로 수치형/범주형을 구분하여 혼합 데이터의 특성을 명확히 보여준다.

**핵심 시사점**: 같은 ID 공간에 있어도 필드의 의미가 다르므로 필드별 독립 처리가 논리적으로 타당함을 직관적으로 이해시킨다.

---

### Figure 3 (p.4) - NON 전체 구조

NON의 3계층 아키텍처를 도식화한다. 하단의 Field-wise Network에서 범주형($i$번째)과 수치형($j$번째) 필드가 각각 독립 DNN(F 블록)을 통과한다. 중간 Across Field Network에서 DNN, Self-Attention 등 다양한 연산이 병렬로 수행된다. 상단 Operation Fusion Network에서 이 출력들이 다시 DNN으로 통합된다.

**핵심 시사점**: "Network On Network"라는 명칭의 의미를 구조적으로 보여준다 - 네트워크(field-wise) 위에 네트워크(across field)를, 그 위에 또 다른 네트워크(fusion)를 쌓는 계층적 설계.

---

### Figure 5 (p.6) - Auxiliary Loss 훈련 효과

Criteo 서브셋에서 일반 DNN(점선)과 Auxiliary Loss 적용 DNN(실선)의 훈련 과정 AUC를 비교한다. 동일한 AUC 달성 시점에서 약 1.67배 빠른 수렴을 보인다. DNN with Auxiliary Losses는 최종 AUC도 더 높게 수렴한다.

**핵심 시사점**: 단순 훈련 가속화를 넘어 최종 성능도 향상시킨다는 점에서 auxiliary loss가 정규화 효과도 가짐을 시사한다. 그러나 이 결과가 Criteo 서브셋에서만 측정된 점은 일반화 한계로 볼 수 있다. ⚠️

---

### Figure 6 (p.8) - 연산 조합별 AUC 비교

6개 데이터셋에서 7가지 연산 조합(a~g)의 AUC를 막대그래프로 보여준다. 가장 중요한 관찰은:
- **어떤 단일 조합도 모든 데이터셋에서 최고가 아님** → 데이터 기반 연산 선택의 필요성 입증
- Criteo/Avazu(대규모): 복잡 연산(attention 포함) 조합이 높은 성능
- Talkshow/Social/Sports(소규모): 단순 조합(DNN+LR)이 오히려 우수

**핵심 시사점**: "one-size-fits-all" 접근법의 한계를 경험적으로 증명하며, NON의 data-driven 연산 선택 설계를 정당화한다.

---

### Figure 7 (p.9) - Field-wise Network 전후 임베딩 시각화

6개 데이터셋에서 각 2개 필드의 t-SNE 2D 시각화를 Field-wise Network 처리 전(상단 행)과 후(하단 행)로 비교한다.
- **처리 전**: 두 필드(빨간색, 녹색)의 임베딩이 혼재되어 구분 불명확
- **처리 후**: 동일 필드 내 특징들이 명확히 군집화되고, 서로 다른 필드 간 분리가 뚜렷해짐

**핵심 시사점**: 정성적 증거로서 Field-wise Network가 실제로 intra-field 정보를 임베딩 공간에 반영함을 직관적으로 보여준다. Table 6의 코사인 거리 정량적 증거(최대 2 오더 향상)와 상호 보완적이다.

---

## 8. 결론 및 후속 연구

### 저자가 제시한 시사점 및 결론 (p.8-9, §5)

1. 태뷸러 데이터 분류에서 intra-field 정보는 성능에 유의미한 영향을 미치며, 기존 방법들이 이를 간과해왔음
2. NON의 3단계 설계(Field-wise + Across-field + Operation Fusion)가 상호 보완적으로 작동함
3. 데이터 기반 연산 선택(data-driven operation selection)이 단일 고정 연산 조합보다 우수함
4. Auxiliary loss가 깊은 구조의 훈련 어려움을 효과적으로 완화함

**저자가 제시한 후속 연구 방향**: 논문 내 명시적 future work 섹션 없음 ⚠️

---

### 8-1. 모델의 일반화 성능 향상 가능성

#### 현재 일반화 관련 강점

| 요소 | 설명 |
|------|------|
| **다양한 도메인 검증** | 광고(Criteo), 모바일(Avazu), 추천(Movielens), 토크쇼, 소셜, 피트니스 6개 도메인에서 일관된 성능 |
| **Auxiliary Loss 정규화** | 중간 레이어 감독이 과적합 방지 효과 제공 |
| **Data-driven 연산 선택** | 도메인 특성에 맞는 연산을 자동 선택하여 이식성 향상 |

#### 일반화 개선 가능성 (내 분석)

1. **도메인 적응(Domain Adaptation)**: 현재 NON은 단일 도메인 학습에 특화되어 있으나, 필드 의미가 다른 도메인 간 전이 학습(transfer learning)에 Field-wise Network가 도메인별 특화 표현을 학습하는 유리한 귀납적 편향(inductive bias)을 제공할 수 있음

2. **Meta-learning 통합**: 새로운 도메인에 소수의 샘플만으로 적응하는 Few-shot 시나리오에서, 각 필드별 DNN이 필드 의미론적 특성을 학습하므로 MAML 등 메타러닝 프레임워크와 결합 가능성이 있음

3. **정규화 강화**: 비공개 데이터셋(Talkshow 등)이 소규모(~2M)인 점을 고려하면, Dropout, BatchNorm 등 추가 정규화 기법과의 결합이 소규모 데이터에서의 일반화를 더욱 향상시킬 수 있음

4. **임베딩 공유 전략**: 현재 각 필드별 독립 DNN이 파라미터를 증가시키는 문제가 있으므로, 필드 유사성 기반 파라미터 공유(parameter sharing)가 일반화와 효율성을 동시에 향상시킬 수 있음

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> **⚠️ 중요 고지**: 아래 분석은 제공된 PDF 원문 범위를 벗어나는 내용입니다. 2020년 이후 연구에 대한 제 지식(학습 데이터 기준)을 바탕으로 제시하며, 일부 세부 수치는 확인이 필요합니다. 정확도가 확실하지 않은 부분은 ⚠️로 표시합니다.

| 연구 | 발표 | 핵심 기여 | NON과의 관계 |
|------|------|-----------|-------------|
| **TabNet** (Arik & Pfister, 2021, AAAI) | 2021 | Sequential attention으로 특징 선택; 해석 가능성 강조 | NON은 필드 단위 처리 우수; TabNet은 특징 중요도 명시화에 강점 |
| **FT-Transformer** (Gorishniy et al., 2021, NeurIPS) ⚠️ | 2021 | 태뷸러 데이터에 Transformer 직접 적용 | NON의 Across-field self-attention과 유사한 동기; FT-Transformer는 더 범용적 아키텍처 |
| **SAINT** (Somepalli et al., 2021) ⚠️ | 2021 | Row-wise 및 Column-wise attention 결합 | NON의 intra-field vs. inter-field 구분과 개념적으로 유사 |
| **AutoML/NAS for Tabular** (다수) | 2021-2023 | 아키텍처 자동 탐색 | NON의 data-driven 연산 선택을 더 정교하게 확장 |
| **XTab** (Zhu et al., 2023) ⚠️ | 2023 | 크로스 도메인 태뷸러 사전 훈련 | NON의 도메인 특화 접근과 대조적인 범용 사전 훈련 패러다임 |

#### NON이 후속 연구에 미치는 영향

1. **Field-level 처리의 선구적 제안**: NON이 제안한 필드별 독립 DNN 처리는 이후 태뷸러 Transformer 연구(FT-Transformer, SAINT 등)에서 column-wise attention 개념으로 발전됨 ⚠️

2. **Data-driven 연산 선택**: 이후 AutoML 연구들이 더 정교한 탐색 알고리즘(예: Bayesian optimization, NAS)을 적용하는 방향으로 발전

3. **산업적 실용성 강조**: NON이 실제 고객 데이터(Talkshow, Social, Sports)로 검증한 점은 이후 산업 응용 논문들에 유사한 검증 방식을 장려함

#### 앞으로 연구 시 고려할 점

| 고려사항 | 설명 |
|----------|------|
| **Transformer 아키텍처와의 비교** | FT-Transformer 등 최신 태뷸러 Transformer와의 공정한 비교가 필요 |
| **사전 훈련 패러다임** | 도메인별 사전 훈련된 태뷸러 모델과의 결합 가능성 탐색 |
| **해석 가능성(Explainability)** | Field-wise Network의 가중치를 해석하여 특징 중요도 제공 방법 개발 필요 |
| **연속 학습(Continual Learning)** | 실제 산업 환경에서 데이터 분포가 변화할 때의 적응 메커니즘 필요 |
| **공정한 벤치마크** | 최근 TabPFN, GBDT+DL 하이브리드 등과의 포괄적 비교 필요 |
| **재현성(Reproducibility)** | 비공개 데이터셋 의존도를 줄이고 완전 공개 벤치마크(예: OpenML-CC18) 사용 권장 |
| **Privacy-Preserving 학습** | 고객 데이터의 민감성을 고려한 연합학습(Federated Learning)과의 결합 |

---

**참고 자료 목록**:
1. Luo, Y. et al. (2020). *Network On Network for Tabular Data Classification in Real-world Applications*. SIGIR '20. arXiv:2005.10114v2 (**주요 분석 대상**)
2. Cheng, H.-T. et al. (2016). *Wide & Deep Learning for Recommender Systems*. [ref 9 in paper]
3. Guo, H. et al. (2017). *DeepFM: A Factorization-Machine Based Neural Network for CTR Prediction*. IJCAI '17. [ref 15 in paper]
4. Lian, J. et al. (2018). *XDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems*. KDD '18. [ref 20 in paper]
5. Song, W. et al. (2019). *AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks*. CIKM '19. [ref 29 in paper]
6. He, X. & Chua, T.-S. (2017). *Neural Factorization Machines for Sparse Predictive Analytics*. SIGIR '17. [ref 16 in paper]
7. Szegedy, C. et al. (2015). *Going Deeper with Convolutions*. CVPR. [ref 30 in paper]
8. Rendle, S. (2010). *Factorization Machines*. ICDM. [ref 28 in paper]
9. Arik, S.Ö. & Pfister, T. (2021). *TabNet: Attentive Interpretable Tabular Learning*. AAAI 2021. (**2020년 이후 비교 분석 - 원문 외 참조**)
