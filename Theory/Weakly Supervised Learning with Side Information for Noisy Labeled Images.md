# Weakly Supervised Learning with Side Information for Noisy Labeled Images

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장
이 논문은 **노이즈가 포함된 웹 이미지 학습**에서 이미지 관련 **사이드 정보(Side Information)**(캡션, 태그, 텍스트 설명, WordNet 계층 구조 등)를 활용하면 레이블 노이즈의 부정적 영향을 효과적으로 줄일 수 있다고 주장합니다.

### 주요 기여

| 기여 | 설명 |
|------|------|
| **SINet 제안** | 사이드 정보 네트워크(Side Information Network)로 인간 어노테이션 없이 노이즈 레이블 처리 |
| **AliProducts 데이터셋 공개** | 50,000 클래스, 250만 이미지의 대규모 세밀 분류 데이터셋 |
| **SOTA 달성** | WebVision, ImageNet, Clothing-1M, AliProducts에서 최고 성능 달성 |
| **WebVision Challenge 2019 1위** | 5,000 카테고리 분류 태스크에서 Top-5 82.54% 달성 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

대규모 웹 이미지 데이터셋(예: WebVision)은 다음과 같은 문제를 내포합니다:
- 인터넷에서 자동 수집된 이미지는 레이블 노이즈가 심각 (일부 데이터셋에서 40% 이상)
- DNN은 노이즈 레이블에 쉽게 과적합(overfit)되어 성능 저하
- 기존 방법들은 **깨끗한 레이블 데이터(clean supervision)** 를 별도로 요구하거나, 단일 전환 확률(transition probability) 가정에 의존하여 실제 노이즈 시나리오에서 비효율적

> 예시: "apple" 레이블로 수집하면 과일 사과와 Apple 스마트폰 이미지가 혼재 → 텍스트 설명(사이드 정보)으로 이를 구분 가능

---

### 2.2 제안하는 방법 (SINet)

SINet은 **3개의 핵심 모듈**로 구성됩니다.

#### 모듈 1: Class Relation Graph (클래스 관계 그래프 생성)

두 가지 방법으로 클래스 간 의미 유사도 그래프를 구성합니다.

**① WordNet 기반 그래프 $\mathbb{G}_w$:**

WordNet 트리에서 두 클래스 간 최단 경로 거리를 기반으로 유사도 행렬 $S_w \in \mathbb{R}^{C \times C}$ 를 구성합니다.

**② 레이블 임베딩 기반 그래프 $\mathbb{G}_l$:**

BERT + Bidirectional LSTM을 사용해 각 클래스의 텍스트 설명으로부터 레이블 임베딩 $\mathbf{v}_i \in \mathbb{R}^d$ 를 추출하고, 코사인 유사도로 ICS(Inter-Class Similarity) 행렬을 구성합니다:

$$S_l^{ij} = \frac{\mathbf{v}_i^T \mathbf{v}_j}{||\mathbf{v}_i||_2 ||\mathbf{v}_j||_2} $$

**③ 하이브리드 그래프 $\mathbb{G}_t$:**

두 그래프를 결합하여 최종 관계 행렬을 생성합니다:

$$S_t = S_l + S_w $$

---

#### 모듈 2: Visual Prototype Generation (시각적 프로토타입 생성)

각 클래스 $c$에 대해 신뢰할 수 있는 시각적 프로토타입 $v_c$를 생성합니다.

**핵심 아이디어:** 신뢰할 수 있는 이미지는 시각적 공간에서의 클래스 간 관계가 사이드 정보로 구성된 클래스 관계 그래프와 유사해야 합니다.

**일관성 점수(Consistency Score) 계산:**

이미지 $x_i$의 시각적 유사도 벡터 $s_v^i$와 의미 유사도 벡터 $s_t^i$ 사이의 KL Divergence로 일관성 점수를 계산합니다:

$$p_i = \frac{1}{\left( KL\left( \psi(s_t^i), \psi(s_v^i) \right) + \epsilon \right)^\gamma} $$

- $\psi$: 정규화 함수 (L2 norm 또는 softmax)
- $\gamma$: 대비(contrast) 조절 파라미터
- $\epsilon$: 분모가 0이 되는 것을 방지하는 소수

**시각적 프로토타입 생성 (가중 평균):**

$$v_c = \frac{\sum_{i=1}^{N} g_i p_i}{\sum_{i=1}^{N} p_i} $$

- $g_i$: 이미지 $x_i$의 CNN 시각적 특징 벡터

---

#### 모듈 3: Noise Weighting (노이즈 가중치 부여)

이미지의 시각적 특징 $g_i$와 클래스 프로토타입 $v_c$ 사이의 유클리드 거리로 중요도 가중치를 계산합니다:

```math
w_{i,c} = \max\left\{0, \left[\alpha - ||v_c - g_i||_2\right]^\beta\right\}
```

- $\alpha$: 훈련에 참여하는 노이즈 데이터 양을 제어하는 shift factor
- $\beta$: 점수 차이를 날카롭게 하는 contrast factor

**최종 학습 목표 (가중 크로스 엔트로피 손실):**

$$\theta^* = \arg\min \sum_{i=1}^{N} w_i \cdot L(y_i, \mathcal{G}(\theta, x_i)) $$

$$\text{Loss}_{ce} = \sum_{i=1}^{N} \sum_{c=1}^{C} w_{i,c} \cdot \log(p_{i,c}) $$

---

### 2.3 모델 구조

```
[입력 이미지]
     ↓
[Visual Encoder (ResNeXt-101 등)]
     ↓ g_i
[Query Embed] ──── [L2 Distance] ──── [Noise Weighting: w = φ(d)]
                        ↑                         ↓
               [Visual Prototype v_k]    [Cross Entropy Loss × w]
                        ↑
         ┌──────────────────────────────┐
         │  Prototype Generation Phase  │
         │  ┌──────────────────────┐   │
         │  │ Visual Encoder → ICS │   │
         │  │ (Visual CRG)        │   │
         │  └──────────────────────┘   │
         │           ↓ Graph Matching  │
         │  ┌──────────────────────┐   │
         │  │Textual Encoder→ ICS  │   │
         │  │(BERT+LSTM)(Text CRG) │   │
         │  └──────────────────────┘   │
         └──────────────────────────────┘
```

**주요 구현 세부사항:**
- 백본: ResNeXt-101, SE-ResNeXt-101, SE-Net154
- 옵티마이저: mini-batch SGD, 배치 크기 2,500
- 학습률: 초기 0.1, epoch 30/60/80/90에서 10배씩 감소, 총 100 epochs
- 정규화: random cropping, mirror flip, autoaugment, dropout(0.25)
- **Topk Label Smoothing**: 고신뢰도 이미지로 초기 모델 훈련 후 나머지 이미지에 대해 Top-k 예측과 GT를 혼합한 스무딩 레이블 적용 → Top-5 accuracy +0.2%
- **Adaptive Spatial Resolution**: 224×224 초기 훈련 후 256×256, 312×312로 파인튜닝 → Top-5 accuracy +0.5%

---

### 2.4 성능 향상

#### WebVision 2.0 (ResNeXt-101)

| 방법 | Top-1 | Top-5 |
|------|-------|-------|
| Model-A (전체 데이터, 노이즈 포함) | 51.05% | 74.94% |
| Model-B (고신뢰도 이미지만, 가중치 없음) | 47.81% | 72.08% |
| **Model-C (SINet 적용)** | **55.57%** | **78.34%** |

Model-C는 Model-A 대비 Top-1 **+4.52%**, Top-5 **+3.40%** 향상

#### WebVision 1.0 vs SOTA (ResNet-50)

| 방법 | WebVision1.0 Top1/Top5 | ImageNet Top1/Top5 |
|------|------------------------|---------------------|
| Baseline | 67.8 / 85.8 | 58.9 / 79.8 |
| CleanNet | 70.3 / 87.8 | 63.4 / 84.6 |
| MentorNet | 70.8 / 88.0 | 62.5 / 83.0 |
| CurriculumNet | 72.1 / 89.2 | 64.8 / 84.9 |
| **SINet** | **73.8 / 90.6** | **66.8 / 85.9** |

#### Clothing-1M

| 설정 | CleanNet | MetaCleaner | DeepSelf | **SINet** |
|------|----------|-------------|----------|-----------|
| Noise1M + Clean(25k) | 74.69 | 76.00 | 76.44 | **77.26** |
| Noise1M + Clean(50k) | 79.9 | 80.78 | 81.16 | **81.32** |

#### WebVision Challenge 2019 최종 성능 (앙상블 5개 모델)

| 아키텍처 | Top-1 | Top-5 |
|----------|-------|-------|
| ResNeXt-101 | 55.56% | 78.15% |
| SE-ResNeXt-101 | 55.57% | 78.34% |
| SE-Net154 | 56.87% | 79.61% |
| **앙상블 최종** | - | **82.54%** |

---

### 2.5 한계점

1. **사이드 정보 의존성**: WordNet 계층 구조나 텍스트 설명이 없는 도메인에서는 적용이 제한적
2. **계산 비용**: 프로토타입 생성을 위해 전체 데이터셋에 대한 특징 추출 필요, 반복 업데이트로 연산 부담 증가
3. **End-to-End 학습 미지원**: 프로토타입 생성 단계와 훈련 단계가 분리되어 있어 완전한 end-to-end 최적화가 이루어지지 않음 (논문 결론에서 future work로 명시)
4. **하이퍼파라미터 민감성**: $\alpha$, $\beta$, $\gamma$ 등 여러 하이퍼파라미터의 수동 조정 필요
5. **초기 프로토타입 품질 한계**: 초기 모델의 분류 신뢰도 기반으로 top-k 이미지를 선택하므로, 초기 모델이 이미 잘 학습되어야 한다는 전제가 있음

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능 향상의 핵심 메커니즘

#### ① 다중 모달 정보 융합을 통한 편향 감소
시각적 특징과 텍스트 기반 의미 정보를 결합함으로써 단일 모달에서 발생하는 편향(bias)을 상호 보완합니다. 이는 특정 시각적 패턴에만 과적합되는 현상을 방지합니다.

$$S_t = S_l + S_w$$

위와 같이 언어 임베딩 기반 유사도($S_l$)와 WordNet 계층 구조 기반 유사도($S_w$)를 결합하여 더 견고한 클래스 표현을 생성합니다.

#### ② 노이즈 샘플에 대한 점진적 가중치 조정

```math
w_{i,c} = \max\left\{0, \left[\alpha - ||v_c - g_i||_2\right]^\beta\right\}
```

노이즈 샘플을 완전히 제거하지 않고 연속적인 가중치를 부여함으로써:
- **정보 손실 방지**: 경계 샘플(hard sample)이 버려지지 않음
- **커리큘럼 학습 효과**: 신뢰도 높은 샘플부터 점진적 학습

#### ③ Adaptive Label Smoothing의 일반화 기여

고신뢰도 이미지로 학습된 초기 모델의 확률 분포를 활용하여 레이블을 스무딩함으로써, 원-핫 레이블에 과적합되는 현상을 방지합니다. 이는 Szegedy et al.의 label smoothing과 유사한 효과로, 모델이 과도하게 확신(over-confident)하는 것을 억제합니다.

#### ④ 반복적 프로토타입 업데이트

$$v_c = \frac{\sum_{i=1}^{N} g_i p_i}{\sum_{i=1}^{N} p_i}$$

훈련 과정에서 프로토타입이 반복적으로 갱신되어, 모델이 점점 더 신뢰할 수 있는 표현으로 수렴합니다. 이는 EM(Expectation-Maximization) 알고리즘과 유사한 자기 정제(self-refinement) 메커니즘으로 볼 수 있습니다.

#### ⑤ Cross-domain 일반화 증거

WebVision 훈련 데이터로 학습된 모델을 ImageNet 검증 세트에서 평가할 때 SINet(66.8%)이 CurriculumNet(64.8%)을 상회하는 것은, 사이드 정보 기반 학습이 **도메인 간 일반화**에도 효과적임을 보여줍니다.

#### ⑥ 세밀 분류(Fine-grained)에서의 일반화

AliProducts(50,000 클래스, 세밀 분류)에서 86.29%로 SOTA를 달성한 것은, 시각적으로 구별하기 어려운 세밀 카테고리에서도 계층적 사이드 정보가 일반화에 기여함을 보여줍니다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려사항

### 4.1 연구에 미치는 영향

#### ① 멀티모달 노이즈 학습 패러다임 확립
SINet은 텍스트 사이드 정보를 노이즈 레이블 학습에 체계적으로 통합한 초기 연구 중 하나로, 이후 **멀티모달 노이즈 레이블 학습** 연구의 방향을 제시했습니다.

#### ② 클래스 관계 그래프 활용의 선례
클래스 간 의미 관계를 그래프로 모델링하여 노이즈 감지에 활용하는 아이디어는, 이후 GNN(Graph Neural Network) 기반 노이즈 레이블 연구의 선구적 역할을 했습니다.

#### ③ 대규모 세밀 분류 데이터셋 제공
AliProducts 데이터셋은 50,000 클래스의 실세계 노이즈 데이터셋으로, 이후 연구의 벤치마크로 활용될 수 있는 자원을 제공합니다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래 연구들은 제가 학습된 지식에 기반한 것으로, 논문 원문에서 직접 인용된 것이 아닙니다. 일부 세부 수치는 부정확할 수 있으므로 원문 확인을 권장합니다.

#### 주요 후속 연구 동향

| 연구 | 핵심 아이디어 | SINet과의 차이점 |
|------|--------------|----------------|
| **DivideMix** (Li et al., ICLR 2020) | GMM으로 clean/noisy 분리 후 MixMatch 적용 | 사이드 정보 없이 순수 시각적 특징만 활용 |
| **SELF** (Nguyen et al., CVPR 2021) | 자기 앙상블을 통한 소프트 레이블 추정 | End-to-end 학습 가능, 프로토타입 불필요 |
| **ELR** (Liu et al., NeurIPS 2020) | 초기 예측을 정규화 항으로 사용 | 경량, 추가 모듈 불필요 |
| **UNICON** (Karim et al., CVPR 2022) | 대조 학습 + GMM 기반 분리 | 자기지도학습(SSL) 접목 |
| **NCE+RCE** (Ma et al., ICCV 2021) | 대칭적 손실함수 설계 | 아키텍처 변경 없이 손실함수만 변경 |

#### SINet의 차별점 및 한계

**장점:**
- 사이드 정보 활용으로 클래스 의미 관계를 명시적으로 모델링
- 인간 어노테이션 불필요
- 대규모(5,000+ 클래스) 시나리오에서 검증

**후속 연구 대비 한계:**
- DivideMix, UNICON 등 대조 학습 기반 방법들은 별도의 사이드 정보 없이도 SINet에 필적하거나 능가하는 성능을 일부 벤치마크에서 달성
- End-to-end 학습 미지원으로 인한 최적화 한계

---

### 4.3 앞으로 연구 시 고려할 점

#### ① End-to-End 학습 가능한 통합 프레임워크 개발
논문 결론에서 직접 언급한 future work로, 프로토타입 생성과 분류 모델 훈련을 통합된 목표함수로 최적화하는 방향이 필요합니다.

$$\mathcal{L}_{total} = \mathcal{L}_{cls} + \lambda_1 \mathcal{L}_{proto} + \lambda_2 \mathcal{L}_{consistency}$$

#### ② 대조 학습(Contrastive Learning)과의 결합
최근 MoCo, SimCLR 등 자기지도학습 방법과 노이즈 레이블 학습을 결합하면 사이드 정보 없이도 더 강건한 표현을 학습할 수 있습니다. SINet의 사이드 정보를 대조 학습의 positive pair 구성에 활용하는 방향이 유망합니다.

#### ③ LLM 기반 사이드 정보 활용 강화
최근 GPT-4, LLaMA 등 대형 언어 모델의 발전으로, BERT+LSTM 조합 대신 더 풍부한 의미 표현 추출이 가능합니다. 특히 CLIP과 같은 비전-언어 사전학습 모델을 활용하면 시각-텍스트 정렬이 개선될 수 있습니다.

#### ④ 동적 그래프 업데이트 메커니즘
현재 클래스 관계 그래프는 정적으로 구성되지만, 훈련 과정에서 모델이 학습한 시각적 유사도를 피드백으로 그래프를 동적으로 갱신하는 방법을 고려할 수 있습니다.

#### ⑤ 장기 꼬리 분포(Long-tail Distribution) 문제와의 결합
실세계 웹 데이터는 노이즈 레이블과 클래스 불균형이 동시에 발생합니다. SINet의 프로토타입 기반 접근법을 장기 꼬리 학습과 결합하는 연구가 필요합니다.

#### ⑥ 사이드 정보가 없는 도메인으로의 전이
의료 영상, 위성 영상 등 WordNet이나 텍스트 설명이 충분하지 않은 도메인에서도 작동하는 범용적 프레임워크 개발이 중요합니다.

---

## 참고 자료 (출처)

**논문 원문:**
- Lele Cheng et al., "Weakly Supervised Learning with Side Information for Noisy Labeled Images," arXiv:2008.11586v2 [cs.CV], 4 Sep 2020.

**논문 내 인용 참고문헌 (원문에서 직접 확인):**
- Lee et al., "CleanNet: Transfer learning for scalable image classifier training with label noise," arXiv:1711.07131, 2017
- Jiang et al., "MentorNet: Regularizing very deep neural networks on corrupted labels," arXiv:1712.05055, 2017
- Guo et al., "CurriculumNet: Weakly supervised learning from large scale web images," ECCV 2018
- Zhang & Sabuncu, "Generalized cross entropy loss for training deep neural networks with noisy labels," NeurIPS 2018
- Han et al., "Deep self-learning from noisy labels," arXiv:1908.02160, 2019
- Zhang et al., "MetaCleaner: Learning to Hallucinate Clean Representations for Noisy-Labeled Visual Recognition," CVPR 2019
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," ACL 2019
- Li et al., "Webvision database: Visual learning and understanding from web data," CoRR abs/1708.02862, 2017
- Xiao et al., "Learning from massive noisy labeled data for image classification," CVPR 2015 (Clothing-1M)
- Miller, "WordNet: A Lexical Database for English," ACM 1995

**비교 분석에 활용된 후속 연구 (학습 지식 기반, 원문 미포함):**
- Li et al., "DivideMix: Learning with Noisy Labels as Semi-supervised Learning," ICLR 2020
- Liu et al., "Early-Learning Regularization Prevents Memorization of Noisy Labels," NeurIPS 2020
- Ma et al., "Normalized Loss Functions for Deep Learning with Noisy Labels," ICML 2020
