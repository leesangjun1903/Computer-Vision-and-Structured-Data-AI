# Noise-Aware Fully Webly Supervised Object Detection 

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문은 **완전 웹 지도 객체 탐지(fWebSOD: fully Webly Supervised Object Detection)** 라는 새로운 태스크를 정의하고, 웹에서 수집한 이미지-레벨 레이블만을 사용하여 객체 탐지기를 학습하는 end-to-end 프레임워크를 제안합니다. 웹 데이터의 이미지 레벨 레이블에 존재하는 **이질적(heterogeneous) 노이즈**를 두 가지 유형으로 분류하고, 각각에 특화된 방법으로 노이즈의 부정적 영향을 줄이는 것이 핵심 주장입니다.

### 주요 기여

| 기여 항목 | 설명 |
|---|---|
| **노이즈 유형 분류** | 배경 노이즈(Background Noise)와 전경 노이즈(Foreground Noise)로 이질적 노이즈를 명시적으로 구분 |
| **잔차 학습 구조(RD Head)** | 배경 노이즈를 분해하고 클린 데이터를 모델링하는 잔차 탐지 헤드 설계 |
| **공간 민감 엔트로피 기준(SSE)** | 배경 레이블이 노이즈일 확률을 추정하는 새로운 기준 제안 |
| **Bagging-Mixup 학습** | 전경 노이즈의 영향을 억제하고 훈련 데이터 다양성을 유지하는 데이터 증강 전략 |
| **데이터셋 구축** | Flickr-VOC, Flickr-COCO 데이터셋 구축 및 공개 |
| **End-to-End 파이프라인** | 기존 다단계 방식과 달리 단일 end-to-end 학습 프레임워크 제시 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**fWebSOD의 도전 과제:**
- 기존 완전 지도(Fully Supervised) 탐지기: 바운딩 박스 어노테이션 필요 → 비용 과다
- 약지도(WSOD): 수동 이미지-레벨 어노테이션 필요
- WebSOD 기존 연구([11],[7],[53]): 노이즈를 명시적으로 처리하지 않거나, 추가 클린 데이터셋(PASCAL VOC) 사용

논문에서 정의하는 두 가지 노이즈 유형:

- **배경 노이즈(Background Noise, BN)**: Missing label 문제. 이미지에 존재하는 객체 카테고리가 레이블에 누락된 경우.
  - 예: 사람과 비행기가 함께 있는 이미지인데, 레이블에는 `aeroplane`만 있고 `person`이 누락
- **전경 노이즈(Foreground Noise, FN)**: 레이블에는 해당 카테고리가 있지만 실제 이미지에는 해당 객체가 없는 경우.
  - 예: 레이블이 `aeroplane`인데 이미지에 비행기가 없음

---

### 2.2 제안하는 방법 및 수식

#### (A) 기본 구조: WSDDN 기반

$$N_c$$개 카테고리에 대해, 웹 이미지로부터 훈련 데이터 $D = \{I_i, \mathbf{t}\_i\}_{i=1}^{N_D}$를 구성합니다. $N_b$개 객체 제안(proposal)의 특징 $\phi$를 추출하고, 두 스트림(분류 스트림, 탐지 스트림)으로 분기합니다.

**소프트맥스 정규화:**

$$\sigma(X^c)_{ij} = \frac{e^{X^c_{ij}}}{\sum_{k=1}^{N_c} e^{X^c_{ik}}}, \quad \sigma(X^d)_{ij} = \frac{e^{X^d_{ij}}}{\sum_{r=1}^{N_b} e^{X^d_{rj}}} $$

- $X^c$: 카테고리 방향 소프트맥스 (분류 스트림)
- $X^d$: 제안 방향 소프트맥스 (탐지 스트림)

**탐지 점수 행렬:**

$$X^s = \sigma(X^c) \odot \sigma(X^d)$$

**이미지 레벨 분류 점수:**

$$y_k = \sum_{r=1}^{N_b} X^s_{rk}$$

**기본 크로스 엔트로피 손실:**

```math
\mathcal{L}^{\text{baseline}} = \sum_{k=1}^{N_c} \left\{ t_k \log y_k + (1 - t_k) \log(1 - y_k) \right\}
```

---

#### (B) 노이즈 분해: 약 탐지 헤드(WD) + 잔차 탐지 헤드(RD)

**WD 헤드 손실 (카테고리 $k$):**

$$\mathcal{L}^{\text{WD}}_k = t_k \log y_k + (1 - t_k) \log(1 - y_k) $$

**RD 헤드:** WD 헤드의 제안 특징 $\phi^{\text{fc}}$에 잔차 특징 $\bar{\phi}^{\text{fc}}$를 더하여 노이즈 특징 생성:

$$\hat{\phi}^{\text{fc}}_i = \bar{\phi}^{\text{fc}}_i + \phi^{\text{fc}}_i$$

**RD 헤드 손실 (카테고리 $k$):**

$$\mathcal{L}^{\text{RD}}_k = t_k \log \hat{y}_k + (1 - t_k) \log(1 - \hat{y}_k) $$

**전체 손실 함수 (노이즈 신뢰도 $p_k$로 가중합):**

```math
\mathcal{L} = \sum_{k=1}^{N_c} \left\{ (1 - p_k) \mathcal{L}^{\text{WD}}_k + p_k \mathcal{L}^{\text{RD}}_k \right\}
```

여기서 $p_k \in [0, 1]$은 $k$번째 배경 레이블이 노이즈일 확률입니다.

> **핵심 메커니즘:** $p_k$가 낮으면 WD 헤드가 주도하여 신뢰할 수 있는 정보를 학습하고, $p_k$가 높으면 RD 헤드가 노이즈를 분해하는 역할을 합니다. $p_k$는 **정보 게이트(Information Gate)** 역할을 수행합니다.

---

#### (C) 공간 민감 엔트로피 기준 (SSE)

배경 카테고리의 정확한 탐지 결과는 공간적으로 분산되고 점수가 균일하다는 관찰에서 출발합니다.

**탐지 점수의 섀넌 엔트로피:**

$$E_{rk} = -X^s_{rk} \ln X^s_{rk} $$

**Jaccard 인덱스 행렬** $J \in \mathbb{R}^{N_b \times N_b}$, $J_{ij} = \text{IoU}(b_i, b_j)$

**공간 정보를 활용한 엔트로피 정규화:**

$$G = E \oslash (JE) $$

여기서 $\oslash$는 하다마드 나눗셈입니다. 분모 $JE$는 제안 간 IoU를 가중치로 하여 엔트로피를 합산합니다.

**정제된 엔트로피:**

$$\hat{E} = G \odot E $$

**배경 레이블 $k$가 노이즈일 신뢰도:**

$$p_k = \begin{cases} 1 - \dfrac{\sum_r^{N_b} \hat{E}_{rk}}{z_k} & \text{if } t_k = 0 \\ 0 & \text{if } t_k = 1 \end{cases} $$

여기서 $z_k = -y_k \ln \frac{y_k}{N_b}$는 최대 엔트로피입니다.

> **검증:** SSE 기준값의 전경(foreground)과 배경(background) 평균 $\hat{E}$는 각각 0.07, 0.78이며, 배경 노이즈(BN)의 평균 $p$는 0.93, SSE와 BN의 피어슨 상관계수는 **0.91**로 높은 상관성을 보입니다.

---

#### (D) Bagging-Mixup 학습

전경 노이즈 억제를 위한 데이터 증강 전략입니다.

**학습 과정 (3단계):**
1. 같은 레이블 $\mathbf{t}$를 가진 $N_a$개의 웹 이미지 $\{I_i\}_{i=1}^{N_a}$ 랜덤 샘플링
2. 디리클레 분포에서 혼합 비율 샘플링: $\{\lambda_i\}\_{i=1}^{N_a} \sim \text{Dir}(\alpha_1, \ldots, \alpha_{N_a})$, 여기서 $\alpha_1 = N_a \alpha_2 = \cdots = N_a \alpha_{N_a}$
3. 합성 훈련 이미지 생성:

$$\hat{I}_i = \lambda_1 I_i + \sum_{\substack{m,n \\ m \in \{2,\ldots,N_a\} \\ n \in \{1,\ldots,N_a\} \setminus i}} \lambda_m I_n $$

**기존 Mixup [57]과의 차이점:**

| 구분 | Mixup [57] | Bagging-Mixup (본 논문) |
|---|---|---|
| 카테고리 특이성 | 카테고리 무관 | 동일 레이블 이미지만 혼합 (카테고리 특화) |
| 생성 이미지 수 | 하나의 이미지 쌍 → 단일 이미지 | 백(bag) 내 모든 이미지의 볼록 조합 → 다수 이미지 생성 |
| 다양성 유지 | 제한적 | Dirichlet 분포로 다양성 유지 |

---

### 2.3 모델 구조 (전체 파이프라인)

```
웹 이미지 입력
      ↓
[Bagging-Mixup] ─── 전경 노이즈 억제된 합성 이미지 생성
      ↓
[Backbone CNN (VGG-F/M/16)] + RoI Pooling
      ↓
   φ (pooled features)
   ┌──────────────────────────────────────┐
   │         Weak Detection (WD) Head     │
   │  φ → FC → φ^fc → σ(X^c) ⊙ σ(X^d) → X^s → y   │
   │  Loss: L^WD_k, Weight: (1-p_k)      │
   └──────────────┬───────────────────────┘
                  │ φ^fc
   ┌──────────────▼───────────────────────┐
   │      Residual Detection (RD) Head    │
   │  φ → FC → φ̄^fc                      │
   │  φ̂^fc = φ̄^fc + φ^fc                 │
   │  → σ(X̂^c) ⊙ σ(X̂^d) → X̂^s → ŷ    │
   │  Loss: L^RD_k, Weight: p_k           │
   └──────────────────────────────────────┘
            ↑
    [SSE Criterion] → p_k 계산
    (Shannon Entropy + IoU 공간 정보)
```

- **추론 시**: WD 헤드의 $X^s$만을 최종 탐지 점수로 사용
- **NMS**: IoU 임계값 0.5 적용
- **제안 생성**: MCG 알고리즘, 최대 2,048개 제안

---

### 2.4 성능 향상

#### PASCAL VOC 2007 (mAP, IoU@0.5)

| 방법 | 훈련 데이터 | mAP (VGG16) |
|---|---|---|
| Divvala et al. [11] | Google+웹 | 17.1% |
| Chen et al. [7] | Flickr | 24.4% |
| Tao et al. [53] | Flickr-Clean + VOC | 25.4% |
| **본 논문 (Ours)** | **Flickr-VOC만** | **35.1%** |
| WSDDN (수동 어노테이션) | PASCAL VOC | 34.8% |

#### PASCAL VOC 2012

| 방법 | mAP (VGG16) |
|---|---|
| Tao et al. [53] (VOC 포함) | 21.7% |
| **Ours (Flickr-VOC만)** | **32.7%** |

#### MS COCO (AP@0.5)

| 방법 | 훈련 데이터 | AP@0.5 |
|---|---|---|
| WSDDN VGG16 | Flickr-COCO | 7.0% |
| **Ours VGG16** | **Flickr-COCO** | **10.6%** |

#### Ablation Study (VOC 2007, VGG16)

| 구성 | mAP |
|---|---|
| Baseline (WSDDN on Flickr-VOC) | 27.6% |
| + RD | 29.8% |
| + RD + EW (일반 엔트로피) | 30.7% |
| + RD + SSE | 33.4% |
| + RD + SSE + BM2 | **35.1%** |
| + RD + SSE + BM3 | 35.2% |

---

### 2.5 한계점

논문에서 명시적으로 인정하거나 분석을 통해 도출되는 한계점은 다음과 같습니다:

1. **백본 네트워크의 한계**: VGG 계열 백본만 사용. 현대의 ResNet, DETR 등 강력한 백본과의 통합 미검증
2. **제안 기반 방식(Proposal-based)의 의존성**: MCG 등 외부 제안 알고리즘에 의존하여 제안 품질이 성능에 영향
3. **MS COCO에서의 성능 격차**: fWebSOD(10.6%)와 완전 지도(41.5%)의 큰 격차 → 복잡한 데이터셋에서 여전히 한계
4. **대규모 카테고리 확장성**: Flickr API에 의존하는 데이터 수집의 제약
5. **BM 하이퍼파라미터 민감도**: $N_a > 2$에서의 성능 향상이 미미하여 최적 $N_a$ 탐색 필요
6. **합성 이미지의 의미론적 일관성**: Bagging-Mixup으로 생성된 이미지가 시각적으로 자연스럽지 않을 수 있음

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문의 핵심 가치 중 하나는 **다양한 환경에서의 일반화 성능** 향상에 있습니다. 관련 내용을 중점적으로 분석합니다.

### 3.1 일반화 성능 향상에 기여하는 요소

#### (A) 노이즈 견고성을 통한 일반화

노이즈 데이터에서 학습 시 모델이 노이즈 패턴을 암기(memorization)하는 현상을 방지합니다. SSE 기준을 통해 모델이 **노이즈에 덜 민감한 표현(representation)** 을 학습하게 됩니다.

$$p_k = 1 - \frac{\sum_r^{N_b} \hat{E}_{rk}}{z_k} \quad (t_k = 0)$$

이 수식에서 탐지 결과가 공간적으로 분산될수록($\hat{E}$가 높을수록) $p_k$가 낮아져, 해당 레이블을 더 신뢰합니다. 이는 모델이 단순히 레이블 통계를 암기하지 않고, **탐지 결과의 공간 분포를 활용한 의미론적 이해** 를 강화합니다.

#### (B) Bagging-Mixup을 통한 데이터 다양성 유지

디리클레 분포를 활용한 혼합 비율 샘플링:

$$\{\lambda_i\}_{i=1}^{N_a} \sim \text{Dir}(\alpha_1, \ldots, \alpha_{N_a})$$

이는 단순 Mixup보다 다양한 조합의 합성 이미지를 생성하여, 훈련 데이터의 분포 다양성을 높입니다. 이는 **훈련-테스트 분포 갭(train-test distribution gap)** 을 줄이는 데 기여합니다.

#### (C) 도메인 불변 특징 학습

RD 헤드의 잔차 학습 구조는 노이즈와 클린 신호를 분리하여, WD 헤드가 **도메인 불변적인(domain-invariant)** 특징을 학습하도록 유도합니다. 웹 도메인(Flickr)에서 학습하고 벤치마크 도메인(VOC, COCO)에서 테스트하는 **크로스 도메인 일반화** 관점에서 효과적입니다.

#### (D) 다중 백본에서의 일관된 성능 향상

| 백본 | Baseline → Ours | 향상 |
|---|---|---|
| VGG-F | 25.9% → 32.9% | +7.0% |
| VGG-M | 25.5% → 33.3% | +7.8% |
| VGG16 | 27.6% → 35.1% | +7.5% |

세 가지 백본에서 일관된 성능 향상은 제안 방법이 특정 아키텍처에 과적합되지 않음을 시사합니다.

#### (E) 노이즈가 더 많은 데이터에서의 일반화

Flickr-Clean(전처리된 41K 이미지)보다 더 많은 노이즈를 포함하는 Flickr-VOC(88K 이미지)에서 훈련했음에도 더 높은 성능을 달성합니다:

| 훈련 데이터 | mAP (VGG16) |
|---|---|
| Flickr-Clean (전처리) | 33.7% |
| Flickr-VOC (노이즈 포함) | **35.1%** |

이는 제안 프레임워크가 더 많고 더 노이즈한 데이터에서도 효과적으로 일반화함을 보여줍니다.

### 3.2 일반화 성능의 한계와 개선 가능성

- **현재 한계**: 80개 카테고리(MS COCO)에서는 20개 카테고리(VOC) 대비 일반화 성능이 상대적으로 낮음
- **개선 방향**: 더 강력한 백본, 자기 지도 사전 훈련(self-supervised pretraining), 메타 학습(meta-learning) 결합

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

#### (A) fWebSOD 분야의 기초 확립
이 논문은 **완전 웹 지도 객체 탐지(fWebSOD)** 라는 태스크를 명확히 정의하고 최초의 체계적 해법을 제시하여, 이후 연구의 기준점(baseline)으로 기능합니다.

#### (B) 노이즈 유형 분류 체계의 영향
배경 노이즈/전경 노이즈의 이분법적 분류는 이후 노이즈 레이블 학습(Noisy Label Learning) 연구에서 참조되는 개념적 틀을 제공합니다.

#### (C) 자동화된 대규모 탐지기 학습 방향 제시
수동 어노테이션 없이 웹 데이터만으로 경쟁력 있는 탐지기를 학습할 수 있음을 실증적으로 보여, **오픈 월드 객체 탐지(Open World Object Detection)** 와 **대규모 어휘 탐지(Large Vocabulary Object Detection)** 연구의 방향을 제시합니다.

#### (D) 약지도 학습과 웹 데이터 학습의 가교
WSOD와 WebSOD의 방법론을 통합하여, 두 분야의 연구자들이 서로의 성과를 활용할 수 있는 가교 역할을 합니다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

아래는 논문에서 직접 언급된 관련 연구와 본 논문 발표 이후 동향을 기반으로 분석한 내용입니다. **2020년 이후 최신 논문의 구체적 수치는 직접 논문을 확인하시기를 권장합니다.**

#### (A) 트랜스포머 기반 탐지와의 연계

DETR(Carion et al., ECCV 2020) 및 이후 Deformable DETR 등 트랜스포머 기반 탐지기의 등장은 fWebSOD 프레임워크에 새로운 백본 및 탐지 헤드 설계 가능성을 열었습니다. 본 논문의 RD/WD 헤드 구조는 트랜스포머 어텐션 메커니즘과 결합하면 더 풍부한 문맥 정보를 활용할 수 있습니다.

#### (B) 자기 지도 학습(Self-Supervised Learning)과의 연계

MoCo(He et al., CVPR 2020), SimCLR(Chen et al., ICML 2020), DINO(Caron et al., ICCV 2021) 등 자기 지도 사전 훈련 방법은 노이즈 레이블 환경에서도 품질 높은 초기 표현을 제공할 수 있어, fWebSOD의 성능 향상에 기여할 수 있습니다.

#### (C) 오픈 어휘/대규모 어휘 탐지 연구

ViLD(Gu et al., ICLR 2022), Detic(Zhou et al., ECCV 2022) 등의 연구는 대규모 어휘 탐지를 위해 웹 이미지와 언어-비전 정렬을 활용합니다. 본 논문의 fWebSOD 프레임워크는 이러한 연구의 선구적 역할을 합니다.

#### (D) 노이즈 레이블 학습의 발전

DivideMix(Li et al., ICLR 2020), ELR(Liu et al., NeurIPS 2020) 등 노이즈 레이블 학습 방법의 발전은 SSE 기준을 더 정교한 노이즈 추정 방법으로 대체할 가능성을 시사합니다.

| 연구 방향 | 본 논문과의 관계 | 발전 가능성 |
|---|---|---|
| CLIP/ALIGN 기반 웹 학습 | 언어-비전 정렬로 노이즈 필터링 강화 | 높음 |
| Foundation Model 활용 | SAM, DINO 등으로 더 강한 사전 지식 활용 | 매우 높음 |
| 준지도 학습 결합 | 소량의 클린 데이터 활용 | 중간 |
| 연속 학습(Continual Learning) | 새 카테고리 확장성 향상 | 중간 |

---

### 4.3 앞으로 연구 시 고려할 점

#### (A) 더 강력한 백본 및 탐지 아키텍처 통합
본 논문은 VGG 계열 백본에 한정됩니다. ResNet, Swin Transformer, ViT 등 최신 아키텍처와의 통합, 그리고 anchor-free 탐지기(FCOS, CenterNet)와의 결합 가능성을 탐구해야 합니다.

#### (B) 노이즈 추정의 정교화
SSE 기준의 수식:

$$p_k = 1 - \frac{\sum_r^{N_b} \hat{E}_{rk}}{z_k}$$

이는 탐지 결과의 공간 분포만을 활용합니다. 향후에는 **언어-비전 정렬 모델(CLIP 등)** 을 활용하여 이미지와 레이블 간의 의미론적 일관성을 추가적인 노이즈 추정 신호로 활용할 수 있습니다.

#### (C) 대규모 카테고리 확장성
MS COCO(80개)에서 성능 격차가 여전히 큽니다. 수천 개 카테고리를 다루는 Objects365, OpenImages 수준으로 확장하기 위한 계층적 카테고리 구조 활용, 언어 임베딩 기반 지식 전이 등을 고려해야 합니다.

#### (D) 데이터 수집 전략 개선
현재 Flickr API 기반 수집은 카테고리별 약 4,000장으로 제한됩니다. 더 효율적인 데이터 수집 및 능동 학습(Active Learning) 기반의 선별적 데이터 수집 전략이 필요합니다.

#### (E) 다중 레이블 및 관계 노이즈 처리
현재 프레임워크는 단일 이미지의 레이블 노이즈를 처리하지만, 이미지 간 관계(예: 유사 이미지 클러스터)를 활용한 노이즈 처리 전략이 추가적인 성능 향상을 가져올 수 있습니다.

#### (F) 평가 지표의 다양화
현재 mAP@IoU0.5 중심의 평가에서, 더 엄격한 COCO 스타일 mAP(0.5:0.95)와 소형 객체 탐지 성능에 대한 별도 평가가 필요합니다.

#### (G) 설명 가능성(Explainability) 연구
SSE 기준이 왜 특정 레이블을 노이즈로 판단하는지에 대한 시각화 및 해석 가능성 연구가 실용적 활용을 위해 중요합니다.

---

## 참고자료

**주요 참고 논문 (논문 내 인용 기반):**

- **본 논문**: Shen, Y., Ji, R., Chen, Z., et al. "Noise-Aware Fully Webly Supervised Object Detection." *CVPR 2020*. (pp. 11326–11335)
- [6] Bilen, H. and Vedaldi, A. "Weakly Supervised Deep Detection Networks." *CVPR 2016*.
- [7] Chen, X. "Webly Supervised Learning of Convolutional Networks." *ICCV 2015*.
- [11] Divvala, S.K., Farhadi, A., and Guestrin, C. "Learning everything about anything: Webly-supervised visual concept learning." *CVPR 2014*.
- [53] Tao, Q., Yang, H., and Cai, J. "Zero-Annotation Object Detection with Web Knowledge Transfer." *ECCV 2018*.
- [57] Zhang, H., Cisse, M., Dauphin, Y.N., and Lopez-Paz, D. "mixup: Beyond Empirical Risk Minimization." *ICLR 2018*.
- [40] Ren, S., He, K., Girshick, R., and Sun, J. "Faster R-CNN." *NeurIPS 2015*.
- [43] Shen, Y., Ji, R., et al. "Cyclic Guidance for Weakly Supervised Joint Detection and Segmentation." *CVPR 2019*.
- [55] Wei, Y., et al. "STC: A Simple to Complex Framework for Weakly-supervised Semantic Segmentation." *TPAMI 2017*.

**코드 및 데이터셋**: https://github.com/shenyunhang/NA-fWebSOD

> **주의사항**: 2020년 이후 최신 연구(DETR, CLIP, DINO, Detic 등)와의 비교 분석 부분은 해당 논문들의 개요 및 일반적 동향을 기반으로 작성하였으며, 구체적 수치 비교는 각 논문을 직접 확인하시기를 강력히 권장합니다.
