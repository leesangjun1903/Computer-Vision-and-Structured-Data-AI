# I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture

## 종합 분석 보고서

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장
I-JEPA(Image-based Joint-Embedding Predictive Architecture)는 **hand-crafted 데이터 증강(data augmentation) 없이** 추상적인 표현 공간(representation space)에서 예측을 수행함으로써, 높은 의미론적(semantic) 수준의 이미지 표현을 학습할 수 있음을 주장한다. 핵심 아이디어는 단일 context 블록으로부터 동일 이미지 내 다양한 target 블록의 **표현(representation)**을 예측하는 것이다.

### 주요 기여
1. **비생성적(non-generative) 자기지도학습 아키텍처 제안**: 픽셀 공간이 아닌 표현 공간에서 예측함으로써, 불필요한 픽셀 수준의 세부 정보를 제거하고 의미론적 특징을 학습
2. **Multi-block 마스킹 전략**: 충분히 큰 스케일의 target 블록과 공간적으로 분산된 context 블록을 사용하는 마스킹 전략이 의미론적 표현 학습에 핵심적임을 실증
3. **높은 확장성과 효율성**: ViT-Huge/14를 ImageNet에서 16개 A100 GPU로 72시간 이내 훈련하여, MAE 대비 10배 이상, iBOT 대비 2.5배 이상의 연산 효율성 달성
4. **다양한 다운스트림 태스크에서의 범용 성능**: 선형 분류, 객체 카운팅, 깊이 예측 등 다양한 추상 수준의 태스크에서 우수한 성능을 보여 일반화 능력 입증

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

자기지도학습(self-supervised learning)의 기존 두 가지 접근법에는 각각 근본적 한계가 존재한다:

**불변성 기반(Invariance-based) 방법의 한계:**
- Random cropping, color jittering 등 hand-crafted 데이터 증강에 강하게 의존
- 특정 다운스트림 태스크나 다른 데이터 분포에 해로울 수 있는 강한 귀납적 편향(inductive bias) 도입
- 이미지 분류와 인스턴스 분할은 동일한 불변성을 요구하지 않으므로, 다양한 추상 수준의 태스크에 일반화하기 어려움
- 오디오 등 다른 모달리티로 확장이 어려움

**생성적(Generative) 방법의 한계:**
- 픽셀/토큰 수준에서 재구성하므로, 학습된 표현이 낮은 의미론적 수준에 머무르는 경향
- 선형 프로빙(linear probing) 등 off-the-shelf 평가에서 불변성 기반 방법에 비해 성능이 낮음
- 최대한의 성능을 얻으려면 end-to-end 미세조정이 필요

### 2.2 제안하는 방법

#### 전체 프레임워크

I-JEPA는 **Joint-Embedding Predictive Architecture(JEPA)**의 이미지 도메인 구현체로, Energy-Based Model(EBM) 프레임워크 내에서 호환 입력에 낮은 에너지를, 비호환 입력에 높은 에너지를 할당하는 것을 목표로 한다.

세 가지 자기지도학습 아키텍처의 비교:

| 아키텍처 | 손실 함수 적용 공간 | 주요 특징 |
|---------|-----------------|---------|
| Joint-Embedding (JEA) | 임베딩 공간 | 뷰 불변성, 데이터 증강 필요 |
| Generative | 입력(픽셀) 공간 | 재구성 기반, 낮은 의미론적 수준 |
| **JEPA (제안)** | **임베딩 공간** | **예측 기반, 증강 불필요** |

#### Targets 생성

입력 이미지 $y$를 $N$개의 비중첩 패치로 변환하고, target-encoder $f_{\bar{\theta}}$를 통해 패치 수준 표현을 얻는다:

$$s_y = \{s_{y_1}, \ldots, s_{y_N}\}$$

여기서 $s_{y_k}$는 $k$번째 패치의 표현이다. 이로부터 $M$개의 (중첩 가능한) 블록을 랜덤 샘플링한다. $i$번째 블록의 마스크를 $B_i$, 해당 패치 수준 표현을 $s_y(i) = \{s_{y_j}\}_{j \in B_i}$로 표기한다.

**핵심 설계 선택**: Target 블록은 target-encoder의 **출력(output)**에서 마스킹하며, 입력에서 마스킹하지 않는다. 이는 높은 의미론적 수준의 target 표현을 보장하는 데 매우 중요하다 (Table 11에서 출력 마스킹 시 67.3% vs. 입력 마스킹 시 56.1%).

기본 설정으로 $M = 4$, aspect ratio 범위 $(0.75, 1.5)$, scale 범위 $(0.15, 0.2)$를 사용한다.

#### Context 생성

단일 context 블록 $x$를 이미지에서 샘플링한다:
- Scale 범위: $(0.85, 1.0)$
- Aspect ratio: 1 (unit)
- Target 블록과의 중첩 영역을 context에서 제거하여 비자명한(non-trivial) 예측 과제 보장

마스크된 context 블록 $x$는 context encoder $f_\theta$를 통해 패치 수준 표현을 얻는다:

$$s_x = \{s_{x_j}\}_{j \in B_x}$$

#### Prediction

Predictor $g_\phi(\cdot, \cdot)$는 context encoder 출력 $s_x$와 예측 대상 위치의 마스크 토큰 $\{m_j\}_{j \in B_i}$를 입력받아 target 블록 표현을 예측한다:

$$\hat{s}_y(i) = \{\hat{s}_{y_j}\}_{j \in B_i} = g_\phi(s_x, \{m_j\}_{j \in B_i})$$

마스크 토큰은 공유 학습 가능 벡터에 위치 임베딩(positional embedding)을 더하여 구성된다. $M$개의 target 블록에 대해 predictor를 $M$번 적용한다.

#### 손실 함수 (Loss)

예측된 패치 수준 표현 $\hat{s}_y(i)$와 target 패치 수준 표현 $s_y(i)$ 간의 평균 $L_2$ 거리:

$$\mathcal{L} = \frac{1}{M} \sum_{i=1}^{M} D\left(\hat{s}_y(i), s_y(i)\right) = \frac{1}{M} \sum_{i=1}^{M} \sum_{j \in B_i} \left\| \hat{s}_{y_j} - s_{y_j} \right\|_2^2$$

#### 파라미터 업데이트
- **Predictor ($\phi$)** 및 **context encoder ($\theta$)**: 그래디언트 기반 최적화 (AdamW)
- **Target encoder ($\bar{\theta}$)**: context encoder 파라미터의 **지수 이동 평균(EMA)**으로 업데이트:

$$\bar{\theta} \leftarrow \alpha \bar{\theta} + (1 - \alpha) \theta$$

여기서 모멘텀 값 $\alpha$는 0.996에서 1.0으로 선형 증가시킨다. 이 비대칭 구조가 표현 붕괴(representation collapse) 방지에 핵심적이다.

### 2.3 모델 구조

I-JEPA는 세 가지 Vision Transformer (ViT) 구성 요소로 이루어진다:

| 구성 요소 | 역할 | 구조적 특징 |
|---------|------|----------|
| **Context Encoder** $f_\theta$ | 가시적(visible) context 패치만 처리 | 표준 ViT (B/16, L/16, H/14, G/16) |
| **Target Encoder** $f_{\bar{\theta}}$ | 전체 이미지의 패치 표현 생성 | Context encoder와 동일 구조, EMA 업데이트 |
| **Predictor** $g_\phi$ | Context 표현 + 위치 토큰 → target 표현 예측 | **경량(narrow) ViT**: 임베딩 차원 384 고정 |

Predictor의 구체적 설정:
- ViT-B/16 encoder → 깊이 6
- ViT-L/16, ViT-H/16, ViT-H/14 encoder → 깊이 12
- ViT-G/16 encoder → 깊이 16
- Self-attention 헤드 수: backbone encoder와 동일

**[cls] 토큰 미사용**: I-JEPA는 [cls] 토큰 없이 사전훈련하며, 평가 시 target encoder 출력의 average pooling으로 전역 이미지 표현을 생성한다.

### 2.4 성능 향상

#### ImageNet-1K 선형 평가 (Table 1)

| 방법 | 아키텍처 | Epochs | Top-1 (%) |
|------|---------|--------|-----------|
| MAE | ViT-H/14 | 1600 | 77.2 |
| data2vec | ViT-L/16 | 1600 | 77.3 |
| CAE | ViT-L/16 | 1600 | 78.1 |
| **I-JEPA** | **ViT-H/14** | **300** | **79.3** |
| **I-JEPA** | **ViT-H/16₄₄₈** | **300** | **81.1** |
| iBOT (증강 사용) | ViT-L/16 | 250 | 81.0 |

- I-JEPA는 데이터 증강 없이도 증강 기반 방법(iBOT)에 필적하는 성능 달성
- MAE 대비 동일 아키텍처에서 2.1%p 향상 (77.2 → 79.3), 훈련 epochs는 5.3배 적음

#### 전이 학습 (Table 3)

| 방법 | CIFAR100 | Places205 | iNat18 |
|------|----------|-----------|--------|
| MAE (ViT-H/14) | 77.3 | 55.0 | 32.9 |
| data2vec (ViT-L/16) | 81.6 | 54.6 | 28.1 |
| **I-JEPA (ViT-H/14)** | **87.5** | **58.4** | **47.6** |
| DINO (ViT-B/8, 증강) | 84.9 | 57.9 | 55.9 |

#### 저수준 태스크 (Table 4)

| 방법 | Clevr/Count | Clevr/Dist |
|------|------------|------------|
| DINO (ViT-B/8) | 86.6 | 53.4 |
| iBOT (ViT-L/16) | 85.7 | 62.8 |
| **I-JEPA (ViT-H/14)** | **86.7** | **72.4** |

I-JEPA는 뷰 불변성 방법(DINO, iBOT)을 특히 깊이 예측(Clevr/Dist)에서 큰 차이로 능가한다.

#### 확장성 (Section 7)
- ViT-H/14 사전훈련: **1,200 GPU 시간 미만** (iBOT ViT-S/16 대비 2.5배 빠름, MAE ViT-H/14 대비 10배 효율적)
- MAE 대비 반복당 약 7% 느리지만, 약 5배 적은 반복으로 수렴

### 2.5 한계

1. **뷰 불변성 방법과의 완전한 성능 격차 해소 미달**: iNat18 등 일부 전이 태스크에서 DINO/iBOT와 여전히 성능 차이 존재 (I-JEPA 47.6% vs. iBOT 57.3%)
2. **표현 붕괴 방지의 이론적 근거 부족**: EMA 기반 비대칭 아키텍처가 경험적으로 효과적이나, 왜 붕괴를 방지하는지에 대한 이론적 분석 부재
3. **이미지 도메인 한정 실험**: JEPA 프레임워크의 멀티모달 확장 가능성을 언급하지만, 실제로는 이미지에서만 실험
4. **End-to-end 미세조정 성능**: 전체 ImageNet 미세조정 시 MAE(87.8%)에 비해 I-JEPA(87.1%)가 약간 낮음 (Table 15)
5. **대규모 패치 사용 시 저수준 태스크 성능 저하**: ViT-G/16은 더 큰 패치를 사용하여 저수준 태스크에서 성능 저하 발생

---

## 3. 모델의 일반화 성능 향상 가능성

I-JEPA의 일반화 성능 향상에 기여하는 핵심 요소들을 분석한다.

### 3.1 Hand-crafted 증강 제거를 통한 편향 감소

기존 불변성 기반 방법들은 특정 데이터 증강(cropping, color jittering 등)에 의존하여 표현에 강한 편향을 주입한다. 이러한 편향은:
- 분류에는 유리하지만 객체 카운팅, 깊이 예측 등 다른 태스크에는 해로울 수 있음
- 다른 데이터 분포를 가진 사전훈련 태스크에도 해로울 수 있음 [2]

I-JEPA는 이러한 hand-crafted 증강을 **완전히 제거**함으로써, 다양한 추상 수준의 태스크에 적용 가능한 더 일반적인 표현을 학습한다. 실증적으로 Table 4에서 I-JEPA는 저수준 태스크(Clevr/Dist: 72.4%)에서 DINO(53.4%)와 iBOT(62.8%)를 대폭 능가하며, 동시에 고수준 분류 태스크에서도 경쟁력 있는 성능을 유지한다.

### 3.2 추상적 표현 공간에서의 예측

I-JEPA가 표현 공간에서 예측하는 것의 핵심적 이점은 target encoder가 **불필요한 픽셀 수준 디테일을 자동으로 제거**하는 추상적 예측 타겟을 생성할 수 있다는 것이다. Table 7의 ablation 결과가 이를 명확히 보여준다:

| 예측 타겟 | Top-1 (1% ImageNet) |
|---------|---------------------|
| Target-Encoder 출력 (표현 공간) | 66.9% |
| 픽셀 | 40.7% |

표현 공간에서의 예측이 26.2%p 높은 성능을 보이며, 이는 의미론적 일반화에 표현 공간 예측이 필수적임을 시사한다.

### 3.3 Multi-block 마스킹 전략의 역할

I-JEPA의 multi-block 마스킹 전략은 일반화에 핵심적이며, 두 가지 원칙을 따른다:

**(a) 충분히 큰 스케일의 target 블록 (semantic)**

Table 8의 ablation에서 target 블록 스케일의 영향:
- Scale $(0.075, 0.2)$: 19.2%
- Scale $(0.15, 0.2)$: **54.2%** (최적)
- Scale $(0.2, 0.3)$: 33.6%

너무 작은 target은 저수준 텍스처 예측에 그치고, 너무 큰 target은 context의 정보가 불충분해져 성능이 저하된다.

**(b) 충분히 정보가 풍부한 (공간적으로 분산된) context 블록**

Table 9에서 context 스케일의 영향:
- Scale $(0.40, 1.0)$: 31.2%
- Scale $(0.85, 1.0)$: **54.2%** (최적)

### 3.4 데이터 크기 및 모델 크기 확장

Table 5에서 I-JEPA는 더 크고 다양한 데이터셋에서 이점을 얻는다:

| 사전훈련 데이터 | 아키텍처 | CIFAR100 | iNat18 |
|-------------|---------|----------|--------|
| IN1K | ViT-H/14 | 87.5 | 47.6 |
| IN22K | ViT-H/14 | **89.5** | **50.5** |
| IN22K | ViT-G/16 | 89.5 | **55.3** |

데이터셋 확장(IN1K → IN22K)과 모델 확장(ViT-H → ViT-G) 모두에서 의미론적 태스크 성능이 향상되며, 이는 I-JEPA의 **확장 가능한 일반화 능력**을 보여준다.

### 3.5 Predictor의 위치적 불확실성 포착

Section 8의 시각화 분석에서 I-JEPA predictor는:
- 위치적 불확실성(positional uncertainty)을 정확히 포착
- 올바른 포즈의 고수준 객체 부분을 생성 (예: 새의 등, 자동차의 윗부분)
- 정밀한 저수준 디테일과 배경 정보는 적절히 버림

이는 I-JEPA가 MSN 등과 달리 **전역 의미 정보와 지역 구조 정보를 동시에 보존**하는 표현을 학습함을 보여준다 (Figure 7 vs. Figure 8).

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

#### (1) 자기지도학습 패러다임의 전환
I-JEPA는 hand-crafted 데이터 증강에 의존하지 않는 제3의 자기지도학습 경로를 제시한다. 이는 기존의 "불변성 기반 vs. 생성적" 이분법을 넘어, **표현 공간에서의 예측적 학습**이라는 새로운 패러다임을 확립한다. Yann LeCun의 "A Path Towards Autonomous Machine Intelligence" [48]에서 제안한 JEPA 프레임워크의 첫 번째 성공적 이미지 도메인 구현체로서, 후속 연구(V-JEPA 등)의 토대를 마련하였다.

#### (2) 멀티모달 확장의 청사진
Hand-crafted 이미지 증강이 불필요하므로, 오디오, 비디오, 텍스트 등 다른 모달리티로의 확장이 용이하다. 이는 data2vec이 시도한 범용 자기지도학습 프레임워크와 유사한 방향이나, 더 높은 의미론적 수준의 표현을 학습할 수 있다는 점에서 차별화된다.

#### (3) 연산 효율성의 새로운 기준
표현 공간에서의 예측이 픽셀 공간 예측 대비 5배 빠른 수렴을 보여, 대규모 자기지도학습의 실용적 접근성을 높였다. 이는 학계에서 제한된 연산 자원으로도 대규모 모델 사전훈련을 가능하게 한다.

#### (4) 마스킹 전략의 중요성 재인식
Multi-block 마스킹 전략에 대한 체계적 ablation은 마스킹 기반 자기지도학습에서 "무엇을 마스킹하는가"가 학습되는 표현의 의미론적 수준을 결정짓는 핵심 설계 선택임을 보여준다.

### 4.2 향후 연구 시 고려할 점

1. **표현 붕괴 방지 메커니즘의 이론적 이해**: EMA 기반 비대칭 구조가 왜 효과적인지, 다른 방법(VICReg, Barlow Twins 등)의 정규화와 결합 시 어떤 시너지가 있는지 연구 필요

2. **멀티모달 JEPA로의 확장**: 비디오(V-JEPA), 오디오, 텍스트 등 다양한 모달리티에서의 JEPA 프레임워크 검증 및 크로스모달 예측 학습 탐구

3. **밀집 예측(Dense Prediction) 태스크 강화**: 객체 검출, 세분화 분할(semantic segmentation) 등 밀집 예측 태스크에서의 성능 개선. I-JEPA의 패치 수준 표현이 이러한 태스크에 자연스럽게 적합할 수 있음

4. **마스킹 전략의 적응적 학습**: 고정된 마스킹 파라미터 대신, 학습 과정에서 자동으로 최적 마스킹 전략을 탐색하는 방법 연구

5. **Predictor 구조의 최적화**: Predictor의 깊이, 너비가 성능에 미치는 영향(Table 12, 14)을 고려하여, 더 효율적이고 표현력 있는 predictor 구조 탐구

6. **스케일링 법칙(Scaling Laws) 연구**: 모델 크기, 데이터 크기, 연산량의 상호작용에 대한 체계적 스케일링 법칙 규명

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 비교 연구 목록

| 연구 | 연도 | 카테고리 | 핵심 접근법 |
|------|------|---------|----------|
| **MAE** (He et al.) | 2022 | 생성적 | 픽셀 공간 마스크 재구성 |
| **data2vec** (Baevski et al.) | 2022 | JEPA 유사 | 표현 공간 예측, 멀티모달 |
| **BEiT** (Bao et al.) | 2021 | 생성적 | 토큰화된 공간 예측 |
| **DINO** (Caron et al.) | 2021 | 불변성 기반 | 자기증류(self-distillation) |
| **iBOT** (Zhou et al.) | 2022 | 하이브리드 | DINO + 패치 수준 재구성 |
| **MSN** (Assran et al.) | 2022 | 불변성 기반 | 마스킹 + 뷰 불변성 |
| **CAE** (Chen et al.) | 2022 | 하이브리드 | 재구성 + 정렬 제약 |
| **VICReg** (Bardes et al.) | 2021 | 불변성 기반 | 분산-불변-공분산 정규화 |
| **SimMIM** (Xie et al.) | 2021 | 생성적 | 단순 마스크 이미지 모델링 |

### 5.2 상세 비교

#### MAE vs. I-JEPA

| 측면 | MAE | I-JEPA |
|------|-----|--------|
| 예측 공간 | 픽셀 | 표현 |
| 데이터 증강 | 불필요 | 불필요 |
| 선형 프로빙 (ViT-H/14) | 77.2% | **79.3%** |
| 미세조정 (ViT-H) | **87.8%** | 87.1% |
| 수렴 속도 | 1600 epochs | **300 epochs** |
| 의미론적 수준 | 낮음 | **높음** |

MAE는 미세조정 시 강력하지만, off-the-shelf 표현의 의미론적 수준이 낮다. I-JEPA는 표현 공간 예측을 통해 이 문제를 해결하며, 선형 프로빙에서 2.1%p 향상을 달성한다.

#### data2vec vs. I-JEPA

| 측면 | data2vec | I-JEPA |
|------|----------|--------|
| 멀티모달 지원 | ✓ (비전, 텍스트, 음성) | 이미지 전용 |
| 선형 프로빙 (ImageNet) | 77.3% (ViT-L) | **79.3%** (ViT-H) |
| 1% ImageNet | 73.3% (ViT-L) | **73.3%** (ViT-H) |
| 마스킹 전략 | 랜덤 마스킹 | **Multi-block 마스킹** |
| 연산 효율성 | 1600 epochs | **300 epochs** |

data2vec은 멀티모달 범용성이 강점이나, I-JEPA는 더 적은 연산으로 동등 이상의 의미론적 표현을 학습한다. I-JEPA의 multi-block 마스킹이 data2vec의 랜덤 마스킹보다 우수함이 ablation에서 확인된다 (Table 6: multi-block 54.2% vs. random 17.6%).

#### DINO/iBOT vs. I-JEPA

| 측면 | DINO | iBOT | I-JEPA |
|------|------|------|--------|
| 데이터 증강 필요 | ✓ | ✓ | ✗ |
| 다중 뷰 처리 | ✓ | ✓ | ✗ (단일 뷰) |
| ImageNet 선형 프로빙 | 80.1% (ViT-B/8) | **81.0%** (ViT-L) | **81.1%** (ViT-H/16₄₄₈) |
| Clevr/Dist (깊이 예측) | 53.4% | 62.8% | **72.4%** |
| GPU 효율성 | 중간 | 낮음 | **높음** |

I-JEPA는 증강 없이도 분류에서 iBOT에 필적하며, 저수준 태스크에서는 크게 능가한다. 특히 iBOT의 ViT-S/16보다 I-JEPA의 ViT-H/14가 더 적은 연산을 사용한다.

### 5.3 후속 연구 동향

I-JEPA 이후 JEPA 프레임워크는 다음과 같은 방향으로 확장되고 있다:

- **V-JEPA (2024, Meta AI)**: 비디오 도메인으로의 JEPA 확장. 시간적 마스킹을 활용하여 비디오의 시공간 표현 학습. I-JEPA의 핵심 원칙(표현 공간 예측, hand-crafted 증강 회피)을 비디오에 적용.
- **data2vec 2.0 (2022, Baevski et al.)**: 효율적인 아키텍처로 멀티모달 학습 가속화, I-JEPA와 병행 연구.
- **DINOv2 (2023, Meta AI)**: 대규모 데이터 큐레이션과 불변성 기반 학습을 결합하여 강력한 범용 시각 표현 학습. I-JEPA와는 다른 접근이나 유사한 목표(범용 표현)를 추구.

---

## 참고자료

1. **주 논문**: Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., & Ballas, N. (2023). "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture." *arXiv preprint arXiv:2301.08243v3*.
2. LeCun, Y. (2022). "A Path Towards Autonomous Machine Intelligence Version 0.9. 2." (논문 내 참고문헌 [48])
3. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). "Masked Autoencoders Are Scalable Vision Learners." *IEEE/CVF CVPR*. (논문 내 참고문헌 [36])
4. Baevski, A., Hsu, W.-N., Xu, Q., Babu, A., Gu, J., & Auli, M. (2022). "data2vec: A General Framework for Self-Supervised Learning in Speech, Vision and Language." *arXiv preprint arXiv:2202.03555*. (논문 내 참고문헌 [8])
5. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., & Joulin, A. (2021). "Emerging Properties in Self-Supervised Vision Transformers." *arXiv preprint arXiv:2104.14294*. (논문 내 참고문헌 [18])
6. Zhou, J., Wei, C., Wang, H., Shen, W., Xie, C., Yuille, A., & Kong, T. (2022). "iBOT: Image BERT Pre-training with Online Tokenizer." *ICLR*. (논문 내 참고문헌 [79])
7. Assran, M., Caron, M., Misra, I., Bojanowski, P., Bordes, F., Vincent, P., Joulin, A., Rabbat, M., & Ballas, N. (2022). "Masked Siamese Networks for Label-Efficient Learning." *ECCV*. (논문 내 참고문헌 [4])
8. Chen, X., Ding, M., Wang, X., Xin, Y., Mo, S., Wang, Y., Han, S., Luo, P., Zeng, G., & Wang, J. (2022). "Context Autoencoder for Self-Supervised Representation Learning." *arXiv preprint arXiv:2202.03026*. (논문 내 참고문헌 [22])
9. Bardes, A., Ponce, J., & LeCun, Y. (2021). "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning." *arXiv preprint arXiv:2105.04906*. (논문 내 참고문헌 [10])
10. Bordes, F., Balestriero, R., & Vincent, P. (2022). "High Fidelity Visualization of What Your Self-Supervised Representation Knows About." *Transactions on Machine Learning Research*. (논문 내 참고문헌 [13])
