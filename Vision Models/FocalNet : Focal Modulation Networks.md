# Focal Modulation Networks

---

## 1. 핵심 주장 및 주요 기여 요약

**Focal Modulation Networks (FocalNets)**는 Vision Transformer의 핵심인 Self-Attention(SA)을 **완전히 대체**하는 새로운 토큰 상호작용 메커니즘인 **Focal Modulation**을 제안한다. 핵심 주장은 다음과 같다:

- **SA의 "상호작용 후 집계(late aggregation)" 패러다임을 뒤집어**, "집계 후 상호작용(early aggregation)" 방식으로 전환하면 더 효율적이면서도 우수한 성능을 달성할 수 있다.
- Focal Modulation은 세 가지 구성요소로 이루어진다: **(i) 계층적 문맥화(Hierarchical Contextualization)**, **(ii) 게이트 집계(Gated Aggregation)**, **(iii) 요소별 어파인 변환(Element-wise Affine Transformation, 즉 Modulation)**.
- FocalNet은 Swin Transformer, Focal Transformer 등 SoTA SA 기반 모델을 **유사한 계산 비용**에서 이미지 분류, 객체 검출, 세그멘테이션 전 태스크에서 **일관되게 능가**한다.
- **뛰어난 해석 가능성(interpretability)**을 보유: CAM/Grad-CAM 없이도 modulator가 자동으로 인식 대상 영역에 수렴한다.

**주요 기여:**
1. Self-Attention 없는(attention-free) 새로운 비전 아키텍처 제안
2. 다중 스케일 계층적 문맥 집계와 입력 의존적(input-dependent) modulation 메커니즘 설계
3. ImageNet 분류, COCO 객체 검출(64.4 mAP test-dev로 새 SoTA), ADE20K 세그멘테이션에서 기존 SA 기반 모델 대비 우수한 성능 달성
4. Monolithic(ViT-like) 및 Multi-scale(Swin-like) 아키텍처 모두에서 효과 입증

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

Vision Transformer의 **Self-Attention(SA)**은 입력 의존적 글로벌 상호작용을 가능하게 하지만 두 가지 근본적 문제가 존재한다:

1. **이차 복잡도(Quadratic Complexity):** 비주얼 토큰 수에 대해 $O((HW)^2 C)$의 계산 복잡도를 가져 고해상도 입력에서 비효율적이다.
2. **무거운 상호작용-집계 구조:** SA는 모든 query-key 쌍에 대해 무거운 상호작용(attention score 계산) 후, 동등하게 무거운 query-value 집계를 수행한다. 이 과정이 **각 query에 대해 독립적으로** 이루어져 계산이 공유되지 않는다.

논문은 근본적 질문을 던진다: *"SA보다 입력 의존적 장거리 상호작용을 모델링하는 더 나은 방법이 있는가?"*

### 2.2 제안하는 방법 (수식 포함)

#### Self-Attention vs. Focal Modulation의 형식적 비교

**Self-Attention (Late Aggregation):**

$$\boldsymbol{y}_i = \mathcal{M}_1(\mathcal{T}_1(\boldsymbol{x}_i, \mathbf{X}), \mathbf{X}) $$

여기서 $\mathcal{T}_1$은 query-key 상호작용(attention score 계산), $\mathcal{M}_1$은 attention score에 기반한 value 집계이다. 집계 $\mathcal{M}_1$이 상호작용 $\mathcal{T}_1$ **이후에** 수행된다.

**Focal Modulation (Early Aggregation):**

$$\boldsymbol{y}_i = \mathcal{T}_2(\mathcal{M}_2(i, \mathbf{X}), \boldsymbol{x}_i) $$

여기서 문맥 특징이 먼저 공유 연산자(depth-wise convolution)를 통해 **위치 $i$에서 집계**($\mathcal{M}_2$)된 후, query와 집계된 특징 간의 **경량 상호작용**($\mathcal{T}_2$)이 수행된다.

#### Focal Modulation의 구체적 인스턴스화

$$\boldsymbol{y}_i = q(\boldsymbol{x}_i) \odot m(i, \mathbf{X}) $$

여기서:
- $q(\cdot)$: query 프로젝션 함수 (선형 변환)
- $\odot$: 요소별 곱셈 (element-wise multiplication)
- $m(\cdot)$: 문맥 집계 함수, 출력을 **modulator**라 부름

#### Step 1: 계층적 문맥화 (Hierarchical Contextualization)

입력 특징 맵 $\mathbf{X} \in \mathbb{R}^{H \times W \times C}$를 선형 레이어로 프로젝션한 후:

$$\mathbf{Z}^0 = f_z(\mathbf{X}) \in \mathbb{R}^{H \times W \times C}$$

$L$개의 depth-wise convolution을 순차적으로 적용하여 계층적 문맥을 추출한다. Focal level $\ell \in \{1, ..., L\}$에서:

$$\mathbf{Z}^{\ell} = f_a^{\ell}(\mathbf{Z}^{\ell-1}) \triangleq \text{GeLU}(\text{DWConv}(\mathbf{Z}^{\ell-1})) \in \mathbb{R}^{H \times W \times C} $$

여기서 $f_a^{\ell}$은 커널 크기 $k^{\ell}$의 depth-wise convolution 후 GeLU 활성화 함수를 적용한다.

- Level $\ell$에서의 유효 수용 영역(effective receptive field): $r^{\ell} = 1 + \sum_{i=1}^{\ell}(k^i - 1)$
- 전역 문맥 캡처를 위해 global average pooling 적용: $\mathbf{Z}^{L+1} = \text{Avg-Pool}(\mathbf{Z}^L)$
- 총 $(L+1)$개의 특징 맵 $\{\mathbf{Z}^{\ell}\}_{\ell=1}^{L+1}$ 획득

#### Step 2: 게이트 집계 (Gated Aggregation)

선형 레이어를 통해 공간- 및 레벨-인식 게이팅 가중치를 생성한다:

$$\mathbf{G} = f_g(\mathbf{X}) \in \mathbb{R}^{H \times W \times (L+1)}$$

가중합을 통해 단일 특징 맵으로 응축:

$$\mathbf{Z}^{out} = \sum_{\ell=1}^{L+1} \mathbf{G}^{\ell} \odot \mathbf{Z}^{\ell} \in \mathbb{R}^{H \times W \times C} $$

여기서 $\mathbf{G}^{\ell} \in \mathbb{R}^{H \times W \times 1}$은 레벨 $\ell$에 대한 게이팅 슬라이스이다.

채널 간 통신을 위해 선형 레이어 $h(\cdot)$를 적용하여 최종 modulator 맵을 생성한다: $\mathbf{M} = h(\mathbf{Z}^{out}) \in \mathbb{R}^{H \times W \times C}$

#### 최종 Focal Modulation (토큰 레벨)

$$\boldsymbol{y}_i = q(\boldsymbol{x}_i) \odot h\left(\sum_{\ell=1}^{L+1} g_i^{\ell} \cdot \boldsymbol{z}_i^{\ell}\right) $$

여기서 $g_i^{\ell}$과 $\boldsymbol{z}_i^{\ell}$는 각각 위치 $i$에서의 게이팅 값과 시각 특징이다.

#### Focal Modulation의 핵심 속성:
- **Translation Invariance**: 위치 임베딩 불사용, query 중심 연산
- **Explicit Input-Dependency**: modulator가 입력에 의존적으로 계산
- **Spatial- and Channel-Specific**: 위치별·채널별 modulation 가능
- **Decoupled Feature Granularity**: $q(\cdot)$는 세밀한 정보 보존, $m(\cdot)$은 거친 문맥 추출

### 2.3 복잡도 분석

**학습 가능 파라미터 수:**

$$3C^2 + C(L+1) + C\sum_{\ell}(k^{\ell})^2$$

**시간 복잡도 (전체 특징 맵):**

$$O\left(HW \times \left(3C^2 + C(2L+3) + C\sum_{\ell}(k^{\ell})^2\right)\right)$$

**비교:**
- Swin Transformer (윈도우 크기 $w$): $O(HW \times (3C^2 + 2Cw^2))$
- Vanilla ViT: $O((HW)^2 C + HW \times 3C^2)$

FocalNet은 $L$과 $(k^{\ell})^2$이 $C$보다 훨씬 작으므로, SA 대비 효율적이며 Swin과 유사한 복잡도를 가진다.

### 2.4 모델 구조

FocalNet은 Swin Transformer와 **동일한 stage 레이아웃 및 hidden dimension**을 사용하되, SA 모듈을 Focal Modulation 모듈로 대체한다.

| 변형 | Depth | Dimension | Focal Levels ($L$) | 시작 커널 크기 ($k^1$) | 유효 수용 영역 ($r^L$) |
|------|-------|-----------|--------------------|---------------------|---------------------|
| FocalNet-T (SRF/LRF) | [2,2,6,2] | [96,192,384,768] | [2,2,2,2] / [3,3,3,3] | [3,3,3,3] | [7,7,7,7] / [13,13,13,13] |
| FocalNet-B (SRF/LRF) | [2,2,18,2] | [128,256,512,1024] | [2,2,2,2] / [3,3,3,3] | [3,3,3,3] | [7,7,7,7] / [13,13,13,13] |
| FocalNet-H | [2,2,18,2] | [352,704,1408,2816] | [4,4,4,4] | [3,3,3,3] | [21,21,21,21] |

- 커널 크기는 하위 레벨에서 상위 레벨로 2씩 증가: $k^{\ell} = k^{\ell-1} + 2$
- Patch embedding: 비중첩 convolution (시작: 4×4, stride 4; 스테이지 간: 2×2, stride 2)

### 2.5 성능 향상

#### 이미지 분류 (ImageNet-1K)

| 모델 | #Params (M) | FLOPs (G) | Top-1 (%) |
|------|-------------|-----------|-----------|
| Swin-Tiny | 28.3 | 4.5 | 81.2 |
| **FocalNet-T (SRF)** | 28.4 | 4.4 | **82.1 (+0.9)** |
| Swin-Base | 87.8 | 15.4 | 83.5 |
| **FocalNet-B (LRF)** | 88.7 | 15.4 | **83.9 (+0.4)** |

ImageNet-22K 사전학습 후 384² finetuning 시: **87.3%** top-1 accuracy 달성.

#### 객체 검출 (COCO, Mask R-CNN 1×)

| Backbone | $AP^b$ | 개선 |
|----------|--------|------|
| Swin-Tiny | 43.7 | - |
| **FocalNet-T (SRF)** | **45.9** | **+2.2** |
| Swin-Base | 46.9 | - |
| **FocalNet-B (LRF)** | **49.0** | **+2.1** |

특히 **FocalNet-T/B의 1× 성능이 Swin-T/B의 3× 스케줄 성능**(46.0/48.5)을 능가하거나 대등하다.

#### 대규모 검출 (COCO test-dev, DINO 사용)

FocalNet-H + DINO로 **64.4 mAP**를 달성하여, Swinv2-G (3.0B 파라미터), BEIT-3 (1.9B 파라미터) 등 훨씬 큰 모델을 **더 적은 파라미터와 사전학습 데이터**로 능가하며 새로운 SoTA를 수립하였다.

#### 시맨틱 세그멘테이션 (ADE20K, UPerNet)

| Backbone | mIoU | +MS |
|----------|------|-----|
| Swin-Base | 48.1 | 49.7 |
| **FocalNet-B (LRF)** | **50.5** | **51.4** |

FocalNet-B는 single-scale에서 Swin-B의 multi-scale 성능(49.7)을 능가한다.

### 2.6 한계

논문에서 명시적으로 언급한 한계와 분석을 통해 파악할 수 있는 한계는 다음과 같다:

1. **NLP 등 다른 도메인으로의 적용 미검증:** 비전 태스크에 한정된 실험이며, NLP 등 다른 도메인에서의 효과는 추가 연구가 필요하다.
2. **Cross-Attention 확장의 어려움:** SA는 query와 key를 교환하여 cross-attention으로 쉽게 변환 가능하나, Focal Modulation은 개별 query에 대한 문맥 수집이 필요하므로 멀티모달 학습에서의 "cross-modulation" 구현이 추가 탐구를 요한다.
3. **커널 크기에 대한 민감도:** 객체 검출에서 커널 크기가 너무 작거나 크면 성능이 하락하며, 최적 커널 크기는 태스크/해상도에 따라 달라진다 (Figure 8 참조).
4. **대규모 학습 시 안정성:** ImageNet-22K 사전학습에서 LayerScale을 사용해야 학습 안정성이 확보되었다.
5. **해석 가능성과 예측 정확도의 상관관계:** modulator의 localization 능력과 최종 예측의 정확도 간 상관관계에 대한 체계적 분석이 부족하다.

---

## 3. 모델의 일반화 성능 향상 가능성

FocalNet의 일반화 성능과 관련하여 논문은 여러 측면에서 강력한 증거를 제시한다.

### 3.1 다양한 태스크로의 전이(Transfer) 성능

FocalNet은 **분류 → 검출 → 세그멘테이션**으로의 전이에서 특히 강점을 보인다:

- **Dense prediction 태스크에서의 우위가 더 크다:** 분류에서의 개선폭(+0.4~0.9%)보다 검출(+2.1~2.2 box mAP)과 세그멘테이션(+2.1~2.4 mIoU)에서의 개선이 훨씬 크다. 이는 다중 스케일 계층적 문맥 집계가 고해상도의 밀집 예측 태스크에서 특히 효과적임을 시사한다.
- **다양한 검출 프레임워크에서의 일관된 성능 향상:** Mask R-CNN, Cascade Mask R-CNN, Sparse R-CNN, ATSS 등 다양한 검출 방법에서 모두 성능 향상을 보여 backbone으로서의 일반성을 입증하였다 (Table 7).

### 3.2 해상도 일반화 (Resolution Generalization)

- FocalNet은 depth-wise convolution 기반이므로 **위치 임베딩(positional embedding)을 사용하지 않아** 해상도 변경에 대한 유연성이 높다.
- 224² 사전학습 후 384² finetuning으로의 전이에서 Swin 대비 일관된 우수성을 보인다 (Table 4).
- 커널 크기 조정만으로 고해상도 입력에 적응 가능: 사전학습 시 $k^{\ell=1}=3$으로 학습한 모델이 검출에서 다른 커널 크기로 finetuning해도 성능이 유지됨 (Figure 8).

### 3.3 Zero-shot 일반화

Language-Image Contrastive Learning (ELEVATER 벤치마크)에서 FocalNet-B는 Swin-B 대비 **20개 데이터셋 평균 +0.8, ImageNet-1K에서 +2.0**의 zero-shot 성능 향상을 달성하였다 (Table 5). 이는 FocalNet이 학습하는 표현이 더 전이 가능하고 일반적임을 나타낸다.

### 3.4 스케일링 일반화

- Tiny → Small → Base → Large → Huge까지 **일관된 성능 향상**을 보이며, 스케일링에 강건하다.
- 특히 Huge 모델(746M 파라미터)에서 COCO test-dev 64.4 mAP로 3.0B 파라미터의 Swinv2-G를 능가하여 **파라미터 효율적 스케일링**을 입증하였다.

### 3.5 모델 증강 기법의 범용적 적용 가능성

Vision Transformer를 위해 개발된 기법들이 FocalNet에도 효과적으로 적용된다:
- **Overlapped patch embedding:** +0.3% (Table 2)
- **Deeper and thinner 구성:** +0.4~0.6% (Table 3)
- 이러한 결과는 FocalNet이 기존 비전 모델 생태계와의 호환성이 높음을 의미한다.

### 3.6 일반화 성능 향상의 구조적 원인 분석

Focal Modulation의 일반화 성능 향상에 기여하는 구조적 요인:

1. **Translation Invariance:** 위치 임베딩 불사용으로 공간적 일반화가 자연스럽다.
2. **다중 스케일 문맥의 계층적 결합:** 단일 스케일만 사용할 때보다 -0.4% 하락 (Table 13, Top-only), 계층적 집계가 세밀-거친 문맥을 모두 포착하여 일반화에 기여한다.
3. **입력 의존적 게이팅:** 게이팅 제거 시 -0.4% 하락 (Table 13), 적응적 문맥 수집이 다양한 입력 분포에 대한 강건성을 제공한다.
4. **특징 세분화의 분리(Decoupled Feature Granularity):** query의 세밀한 정보와 modulator의 거친 문맥이 분리되어 있어, 과적합 없이 충분한 표현력을 확보한다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

1. **"Attention이 전부가 아니다"의 실증적 증거:** FocalNet은 Self-Attention 없이도 비전 태스크에서 SA 기반 모델을 능가할 수 있음을 보여, **attention-free 아키텍처** 연구의 새로운 방향을 제시한다.

2. **Modulation 패러다임의 부상:** "상호작용 후 집계"에서 "집계 후 변조"로의 패러다임 전환은 효율성과 성능을 동시에 달성하는 새로운 설계 원리로, 향후 다양한 아키텍처 설계에 영감을 줄 수 있다.

3. **해석 가능성의 새로운 차원:** Modulator가 CAM/Grad-CAM 없이 자동으로 객체 영역에 수렴하는 특성은, gradient 기반이 아닌 **새로운 모델 해석 방법론**의 가능성을 열었다.

4. **Dense prediction 태스크에서의 backbone 설계 방향:** 분류보다 검출/세그멘테이션에서 더 큰 성능 향상을 보인 점은, 다중 스케일 문맥 집계가 밀집 예측 백본 설계에서 핵심적임을 재확인시킨다.

5. **효율적 스케일링의 가능성:** 더 적은 파라미터로 더 큰 모델을 능가한 결과는, 계산 효율적인 대규모 비전 모델 개발에 대한 새로운 경로를 제시한다.

### 4.2 앞으로 연구 시 고려할 점

1. **멀티모달 확장 (Cross-Modulation):** SA는 cross-attention으로 자연스럽게 확장되지만, Focal Modulation의 cross-modulation 구현은 추가 설계가 필요하다. 이는 VLM(Vision-Language Model) 시대에 중요한 연구 과제이다.

2. **NLP 및 다른 모달리티 적용:** 시퀀스 데이터에서 depth-wise convolution 기반 계층적 문맥화가 효과적인지 검증이 필요하다.

3. **강건성(Robustness) 분석:** Adversarial 공격, 분포 이동(distribution shift), 손상된 입력 등에 대한 강건성은 체계적으로 분석되지 않았다.

4. **자기지도 학습(Self-supervised Learning)과의 결합:** MAE, DINO 등 자기지도 학습 프레임워크에서 Focal Modulation의 효과를 검증할 필요가 있다.

5. **동적 커널 크기 및 레벨 수:** 현재 커널 크기와 레벨 수는 고정되어 있으나, 입력에 따라 동적으로 조절하는 방법이 추가 성능 향상을 가져올 수 있다.

6. **하드웨어 최적화:** Depth-wise convolution과 element-wise multiplication은 GPU에서 SA의 행렬 곱셈만큼 최적화되지 않을 수 있으므로, 커스텀 커널 구현이 실제 배포에 중요하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 메커니즘 | FocalNet과의 비교 |
|------|------|-------------|----------------|
| **ViT** [Dosovitskiy et al.] | 2020 | 글로벌 Self-Attention | FocalNet-B/16이 ViT-B/16 대비 +0.6% (82.4 vs 81.8), 더 강한 해석가능성 |
| **Swin Transformer** [Liu et al.] | 2021 | 윈도우 기반 SA + shifted windows | FocalNet이 전 크기에서 일관되게 능가 (분류 +0.4~0.9%, 검출 +2.1, 세그멘테이션 +2.4) |
| **Focal Transformer** [Yang et al.] | 2021 | 다중 레벨 focal attention | FocalNet은 동일 영감을 공유하나, query 위치에서 modulator를 추출하여 효율성·성능 모두 개선. 처리 속도 약 2배 빠름 |
| **ConvNeXt** [Liu et al.] | 2022 | 현대화된 순수 ConvNet | FocalNet이 대부분 태스크에서 우세 (Table 15). ConvNeXt도 DWConv 사용하나 modulation 없음 |
| **PoolFormer / MetaFormer** [Yu et al.] | 2021 | 풀링 기반 토큰 믹싱 | FocalNet의 Pooling Aggregator 변형 대비 -1.8% 하락. 풀링은 순열 불변(permutation-invariant)으로 시각 구조 포착 불가 |
| **Swin Transformer V2** [Liu et al.] | 2022 | 스케일업된 SA + log-space attention | FocalNet-H가 3.0B의 SwinV2-G를 746M으로 능가 (64.4 vs 63.1 COCO mAP) |
| **BEIT-3** [Wang et al.] | 2022 | 멀티모달 사전학습 + SA | 1.9B 파라미터 + 대규모 멀티모달 데이터에도 FocalNet-H에 뒤짐 (63.7 vs 64.4 COCO test-dev) |
| **DINO (Detection)** [Zhang et al.] | 2022 | Denoising anchor boxes + DETR 개선 | FocalNet은 DINO 프레임워크의 backbone으로 사용되어 상호보완적 관계 |
| **MLP-Mixer** [Tolstikhin et al.] | 2021 | 순수 MLP 기반 토큰 믹싱 | FocalNet이 훨씬 우수 (82.3 vs 76.4%, Tiny급 비교) |
| **DW-Net** [Han et al.] | 2021 | Depth-wise conv 기반 로컬 비전 트랜스포머 | FocalNet의 DW ConvNet 변형이 DW-Net과 유사하나, modulation 추가 시 +0.7% 향상 |
| **SE-Net** [Hu et al.] | 2018 | Global squeeze-and-excitation | FocalNet에서 $L=0$ 설정 시 SE와 유사해지나, 글로벌 문맥만으로는 -6.7% 하락 |

### 핵심 비교 인사이트

**SA 기반 모델 대비:**
- FocalNet은 "query-key 상호작용 → value 집계"를 "문맥 집계 → query 변조"로 **순서를 뒤집음**으로써, 집계를 query 간에 공유(amortize)하여 효율성을 확보하면서 입력 의존성은 유지한다.

**ConvNet 기반 모델 대비:**
- 단순 depth-wise convolution(DW-Net, ConvNeXt)과 달리 **다중 레벨 계층적 집계 + 게이팅 + modulation**의 세 가지 구성요소가 결합되어 유의미한 성능 차이를 만든다.

**MLP 기반 모델 대비:**
- MLP 모델은 글로벌 또는 로컬 토큰 믹싱을 수행하지만, FocalNet의 다중 스케일 문맥 집계는 단기-장기 범위를 자연스럽게 포착하여 더 나은 정확도-효율성 트레이드오프를 달성한다.

---

## 참고 자료

1. Yang, J., Li, C., Dai, X., Yuan, L., & Gao, J. (2022). **"Focal Modulation Networks."** arXiv:2203.11926v3 [cs.CV]. (본 논문 원문)
2. Liu, Z., et al. (2021). **"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows."** arXiv:2103.14030.
3. Dosovitskiy, A., et al. (2020). **"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale."** arXiv:2010.11929.
4. Liu, Z., et al. (2022). **"A ConvNet for the 2020s."** (ConvNeXt) arXiv:2201.03545.
5. Yang, J., et al. (2021). **"Focal Self-Attention for Local-Global Interactions in Vision Transformers."** arXiv:2107.00641.
6. Yu, W., et al. (2021). **"MetaFormer is Actually What You Need for Vision."** (PoolFormer) arXiv:2111.11418.
7. Liu, Z., et al. (2022). **"Swin Transformer V2: Scaling Up Capacity and Resolution."** arXiv:2111.09883.
8. Wang, W., et al. (2022). **"Image as a Foreign Language: BEIT Pretraining for All Vision and Vision-Language Tasks."** (BEIT-3) arXiv:2208.10442.
9. Zhang, H., et al. (2022). **"DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection."** arXiv:2203.03605.
10. Hu, J., Shen, L., & Sun, G. (2018). **"Squeeze-and-Excitation Networks."** CVPR 2018.
11. Tolstikhin, I., et al. (2021). **"MLP-Mixer: An All-MLP Architecture for Vision."** arXiv:2105.01601.
12. Han, Q., et al. (2021). **"Demystifying Local Vision Transformer: Sparse Connectivity, Weight Sharing, and Dynamic Weight."** (DW-Net) arXiv:2106.04263.
13. GitHub 저장소: https://github.com/microsoft/FocalNet
