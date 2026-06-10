# Learning with Noisy Class Labels for Instance Segmentation

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문의 핵심 주장은 **인스턴스 분할(Instance Segmentation)에서 노이즈가 있는 클래스 레이블은 두 가지 서로 다른 하위 태스크에서 다른 역할을 수행한다**는 것입니다.

- **전경-배경 하위 태스크(Foreground-Background Sub-task)**: 객체가 어디 있는지 탐지
- **전경-인스턴스 하위 태스크(Foreground-Instance Sub-task)**: 탐지된 객체의 클래스를 분류

기존 분류(Classification) 태스크에서 사용되던 노이즈-강건 손실(noise-robust loss)을 인스턴스 분할에 그대로 적용하면, 전경-배경 구분에 필요한 올바른 그래디언트 정보까지 억제되어 성능이 저하된다는 문제를 최초로 체계적으로 분석하고 해결책을 제시합니다.

### 주요 기여 (Contributions)

| 기여 항목 | 설명 |
|-----------|------|
| 문제 정의 | 인스턴스 분할에서 노이즈 레이블의 이중적 역할 최초 분석 |
| 방법론 제안 | 샘플 유형별 차별화된 손실 함수 적용 전략 |
| 이론적 분석 | Reverse Cross Entropy의 대칭 손실 특성 및 그래디언트 분석 |
| 실험 검증 | Pascal VOC, Cityscapes, COCO 3개 데이터셋에서 검증 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**문제 배경:**
대규모 데이터셋에서 어노테이터의 경험 부족, 클래스 간 외형 유사성으로 인해 클래스 레이블에 노이즈가 발생합니다. 예를 들어 Cityscapes에서 *motorcycle*이 *bicycle*로 잘못 표기되는 경우입니다.

**핵심 문제:**
기존 분류 태스크의 노이즈-강건 손실(예: Symmetric Loss)을 인스턴스 분할에 직접 적용할 경우:

1. **전경-배경 하위 태스크 관점**: 노이즈 레이블도 "이 샘플이 전경임"을 정확히 가리키므로 올바른 그래디언트를 제공
2. **전경-인스턴스 하위 태스크 관점**: 노이즈 레이블은 잘못된 클래스를 가리키므로 잘못된 그래디언트를 제공

따라서 두 태스크를 동일한 손실로 처리하는 것은 부적절합니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### (A) 기본 멀티 클래스 분류 손실 (Cross Entropy)

$$l_{ce} = \frac{1}{N}\sum_{i=1}^{N} l_{ce,i} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=0}^{K} q(k|x_i)\log p(k|x_i) \tag{1}$$

- $p(k|x_i)$: 샘플 $x_i$에 대한 클래스 $k$의 분류 신뢰도
- $q(k|x_i)$: one-hot 인코딩된 레이블 ($q(y_i|x_i)=1$, 나머지 $=0$)

#### (B) 1단계 학습 손실 (Early Stage: 1~6 epoch)

$$Loss_1 = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=0}^{K} q(k|x_i)\log p(k|x_i) \tag{2}$$

초기 단계에서는 모델이 클린 샘플을 먼저 학습하는 경향이 있으므로, 모든 샘플에 표준 CE 손실을 적용합니다.

#### (C) 단계 분할 기준

$$E_1 = \frac{1}{2}E, \quad s.t. \; \forall\eta \tag{3}$$

- $E$: 전체 에폭 수 (일반적으로 12)
- $E_1$: 1단계 에폭 수 (일반적으로 6)
- $\eta$: 노이즈 비율

#### (D) 2단계 학습 손실 (Mature Stage: 7~12 epoch)

$$Loss_2 = -\frac{1}{N}\left[\sum_{i=1}^{N_1+N_2}\sum_{k=0}^{K} q(k|x_i)\log p(k|x_i) + \sum_{m=1}^{N_3} 0 + \sum_{j=1}^{N_4}(-l_{sl,j})\right] \tag{4}$$

- $N_1$: 네거티브 샘플 수 (배경)
- $N_2$: 유사 네거티브 샘플 수 (Pseudo Negative Samples, PSN)
- $N_3$: 잠재 노이즈 샘플 수 (Potential Noisy Samples, PON) → 손실 = 0으로 격리
- $N_4$: 기타 샘플 수
- $N = N_1 + N_2 + N_3 + N_4$

#### (E) Reverse Cross Entropy Loss (대칭 손실)

$$l_{rce,i} = -\sum_{k=0}^{K} p(k|x_i)\log q(k|x_i) \tag{5}$$

이 손실의 대칭성 조건:

$$\sum_{y=0}^{K} l(f(x), y) = C, \quad \forall x \in \mathcal{X}, \; \forall f \tag{6}$$

$C$는 상수이며, 노이즈 비율 $\eta < \frac{K}{K+1}$일 때 노이즈 강건성이 보장됩니다:

```math
R^\eta(f^*) - R^\eta(f) = \left(1 - \frac{\eta(K+1)}{K}\right)(R(f^*) - R(f)) \leq 0
```

따라서 $f_\eta^* = f^*$이 성립합니다.

#### (F) 그래디언트 분석

**Reverse Cross Entropy의 그래디언트:**

$$\frac{\partial l_{rce}}{\partial z_j} = \begin{cases} Ap_j - Ap_j^2, & q_j = q_y = 1 \\ -Ap_j p_y, & q_j = 0 \end{cases} \tag{7}$$

**Cross Entropy의 그래디언트:**

$$\frac{\partial l_{ce}}{\partial z_j} = \begin{cases} p_j - 1, & q_j = q_y = 1 \\ p_j, & q_j = 0 \end{cases} \tag{8}$$

식 (8)에서 $q_j = 1$일 때, $p_j$가 작을수록 그래디언트가 커져 학습 후반부에 노이즈 샘플이 지배적 역할을 하게 됩니다. 반면 식 (7)에서 RCE는 $p_j = 0.5$ 대칭으로 $p_j \approx 0$이면 그래디언트도 $\approx 0$으로 노이즈에 강건합니다.

---

### 2.3 모델 구조

```
입력 이미지
    ↓
[ResNet-50-FPN 백본] ← 특징 추출
    ↓
[RPN (Region Proposal Network)] ← 이진 분류 손실 (변경 없음)
    ↓
[Box Head] ← ★ 본 논문의 핵심 수정 부분 (멀티클래스 분류 손실)
    ↓
[Segmentation Branch] ← 이진 분류 손실 (변경 없음)
    ↓
출력: 클래스 + 바운딩박스 + 마스크
```

**샘플 분류 체계 (2단계):**

```
전체 샘플 공간 (Ω)
├── 네거티브 샘플 (y=0) → CE 손실 적용
└── 포지티브 샘플 (y≠0)
    ├── 유사 네거티브 샘플 (PSN): argmax_k p(k|x)=0 → CE 손실 적용
    ├── 잠재 노이즈 샘플 (PON): l_ce > γ (γ=6.0) → 손실=0 (격리)
    └── 기타 샘플 (OS) → Reverse CE 손실 적용
```

---

### 2.4 성능 향상

#### Pascal VOC 결과 (mAP 기준)

| 방법 | 0% | 20% | 40% | 60% | 80% |
|------|----|-----|-----|-----|-----|
| CE (Mask R-CNN) | 39.3 | 34.2 | 31.5 | 27.1 | 20.7 |
| SCE | 39.4 | 34.6 | 32.1 | 27.9 | 21.2 |
| **Our Method** | **39.7** | **38.5** | **38.1** | **33.8** | **25.5** |
| **향상폭** | +0.4 | **+4.3** | **+6.6** | **+6.7** | **+4.8** |

#### COCO test-dev 결과 (mAP 기준)

| 방법 | 0% | 20% | 40% | 60% | 80% |
|------|----|-----|-----|-----|-----|
| CE (Mask R-CNN) | **34.2** | 31.3 | 29.3 | 27.1 | 21.7 |
| **Our Method** | 33.7 | **33.1** | **31.3** | **30.8** | **26.6** |

> ⚠️ **주목할 점**: COCO에서 노이즈 없는 경우(0%) 성능이 CE(34.2) 대비 소폭 하락(33.7)합니다. 이는 RCE 손실 적용이 클린 데이터에서는 약간의 정보 손실을 일으킬 수 있음을 시사합니다.

#### Ablation Study 결과 (Pascal VOC, η=40%)

| 구성 | AP | AP50 | AP75 |
|------|----|------|------|
| CE (기준선) | 31.5 | 57.4 | 31.0 |
| ST | 34.3 | 59.9 | 35.1 |
| N & PSN | 37.3 | 63.5 | 39.0 |
| ST & N & PSN & PON (전체) | **38.1** | 64.5 | **40.0** |

---

### 2.5 한계점

1. **클린 데이터 성능 저하**: COCO 0% 노이즈에서 CE 대비 약 0.5% mAP 하락
2. **하이퍼파라미터 민감성**: $\gamma$ (노이즈 임계값) 설정이 성능에 영향을 미침
3. **인공적 노이즈 가정**: 실험에서 노이즈를 인위적으로 생성(label flipping)하여 실제 자연 발생 노이즈와 차이 존재
4. **Mask R-CNN에 특화**: 제안 방법이 Mask R-CNN 구조를 기반으로 설계되어 다른 인스턴스 분할 아키텍처로의 즉각적 확장성 검증 부족
5. **비대칭 노이즈 효과 제한**: 비대칭 노이즈(asymmetric noise)에서의 성능 향상폭(0.6~2.4%)이 대칭 노이즈 대비 작음
6. **에폭 비율 경험적 결정**: $E_1 = \frac{1}{2}E$ 설정이 이론적 근거 없이 경험적으로 결정됨

---

## 3. 모델 일반화 성능 향상 가능성

### 3.1 일반화 성능 향상의 메커니즘

본 논문의 방법이 일반화 성능을 향상시키는 핵심 메커니즘은 다음과 같습니다.

**① 그래디언트 오염 방지 (Gradient Contamination Prevention)**

노이즈 샘플로부터의 잘못된 그래디언트가 모델 파라미터를 오염시키는 것을 방지합니다. RCE 손실에서 $p_j \approx 0$인 샘플의 그래디언트가 0에 수렴하는 특성이 이를 가능하게 합니다:

$$\frac{\partial l_{rce}}{\partial z_j}\bigg|_{p_j \to 0} = Ap_j - Ap_j^2 \approx 0$$

**② 전경-배경 정보 보존**

노이즈 레이블이 존재하더라도 "해당 샘플이 전경임"이라는 정보는 유효하므로, PSN과 NEG 샘플에 CE 손실을 유지하여 전경-배경 구분 능력을 보존합니다.

**③ 단계적 학습 전략 (Curriculum Learning 관점)**

$$\text{1단계(Epoch } 1 \sim E_1): \text{CE 손실로 기본 특징 학습}$$
$$\text{2단계(Epoch } E_1 \sim E): \text{샘플별 차별화 손실로 노이즈 강건성 확보}$$

이는 커리큘럼 학습(Curriculum Learning) 관점에서 쉬운 샘플(클린)을 먼저 학습하고, 어려운 샘플(노이즈)의 영향을 후반부에 제어하는 방식입니다.

**④ 실제 노이즈 환경에서의 일반화 근거**

대칭 손실의 이론적 보장:

$$\text{노이즈 비율 } \eta < \frac{K}{K+1} \text{이면 } f_\eta^* = f^* \text{성립}$$

이는 노이즈 환경에서 학습된 최적해가 클린 환경의 최적해와 동일함을 의미합니다.

### 3.2 다양한 노이즈 환경에 대한 강건성

```
노이즈 비율(η):  20%    40%    60%    80%
                ──────────────────────────
성능 향상폭:   +4.3%  +6.6%  +6.7%  +4.8%  (Pascal VOC 대비 CE)
```

노이즈 비율이 증가할수록 성능 향상폭이 증가하다가 80%에서 감소하는 패턴을 보이는데, 이는 극단적 노이즈(80%)에서도 방법의 유효성이 유지됨을 보여줍니다.

### 3.3 일반화 성능의 한계 요인

- **클린 데이터 환경**: RCE 손실이 적용되는 OS 샘플에서 학습 신호가 약화될 수 있어, 클린 데이터에서는 오히려 소폭 성능 저하 발생
- **극도로 높은 노이즈(>80%)**: 이론적 보장 조건 $\eta < \frac{K}{K+1}$을 벗어날 가능성

---

## 4. 연구 영향 및 앞으로의 고려사항

### 4.1 앞으로의 연구에 미치는 영향

**① 인스턴스 분할 + 노이즈 레이블 연구의 개척**

본 논문 이전에는 인스턴스 분할에서 노이즈 레이블 문제를 체계적으로 다룬 연구가 없었습니다. 이 논문은 해당 분야의 **선구적 연구(pioneering work)**로서, 이후 연구의 벤치마크 및 방법론적 기준점 역할을 합니다.

**② 멀티 태스크 학습에서 노이즈 처리 패러다임 전환**

단일 손실 함수로 모든 태스크를 처리하는 것이 아니라, **태스크별 노이즈 역할 분석 후 차별화된 손실 설계**라는 패러다임을 제시합니다. 이는 다음 분야로 확장 가능합니다:
- 파노프틱 분할(Panoptic Segmentation)
- 멀티태스크 물체 탐지
- 3D 인스턴스 분할

**③ 실용적 구현 가능성**

Mask R-CNN의 Box Head의 분류 손실만 수정하므로, **최소한의 코드 변경으로 기존 파이프라인에 통합 가능**합니다.

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ **주의**: 아래는 본 논문과 관련된 분야의 일반적인 연구 동향을 서술한 것으로, 특정 논문의 세부 수치가 부정확할 수 있습니다. 정확한 비교는 원문 논문을 직접 확인하시기 바랍니다.

| 연구 방향 | 본 논문 접근 | 2020년 이후 동향 |
|-----------|-------------|-----------------|
| 노이즈 탐지 | 손실값 임계치($\gamma$) 기반 | GMM 기반 확률적 탐지 (DivideMix, NeurIPS 2020) |
| 손실 함수 | RCE + CE 조합 | 적응적 가중치 손실 학습 |
| 레이블 정정 | 미적용 | 반지도학습 기반 레이블 재추정 |
| 아키텍처 | Mask R-CNN 특화 | Transformer 기반 (예: QueryInst, Mask2Former) |

**DivideMix (Li et al., NeurIPS 2020)** 계열 연구는 가우시안 혼합 모델(GMM)을 이용해 클린/노이즈 샘플을 확률적으로 구분하는 방식을 제안하는데, 본 논문의 고정 임계치 $\gamma$ 기반 노이즈 탐지보다 더 유연한 접근을 취합니다.

**UNICON (Karim et al., CVPR 2022)** 등의 연구는 대조 학습(Contrastive Learning)과 노이즈 레이블 학습을 결합하여 더 강건한 특징 표현을 학습하는 방향으로 발전하였습니다.

### 4.3 향후 연구 시 고려해야 할 사항

**① 동적 임계치 설정**

현재 $\gamma = 6.0$으로 고정된 하이퍼파라미터를 학습 진행에 따라 동적으로 조절하는 방법을 고려해야 합니다:

$$\gamma(t) = f(\text{epoch}, \eta, K)$$

**② 트랜스포머 기반 아키텍처 적용**

Mask2Former, QueryInst 등 최신 트랜스포머 기반 인스턴스 분할 모델에서의 노이즈 레이블 처리 방법 연구가 필요합니다. 트랜스포머의 어텐션 메커니즘이 노이즈 레이블에 어떻게 반응하는지 분석이 요구됩니다.

**③ 실제 노이즈(Real-world Noise) 검증**

현재 논문은 인위적으로 생성한 노이즈를 사용합니다. 실제 어노테이션 과정에서 발생하는 자연 노이즈 환경(예: 크라우드소싱 데이터)에서의 검증이 필요합니다.

**④ 마스크 품질 노이즈와의 결합**

본 논문은 클래스 레이블 노이즈만 다루지만, 실제 환경에서는 마스크 품질(위치, 경계) 노이즈도 함께 발생합니다. 이를 통합적으로 처리하는 연구가 필요합니다.

**⑤ 준지도학습 및 능동학습과의 결합**

노이즈 레이블 처리와 반지도학습을 결합하여 소량의 클린 레이블과 다량의 노이즈 레이블을 효과적으로 활용하는 방향을 탐색할 수 있습니다.

**⑥ 클래스 불균형 처리**

노이즈 레이블과 클래스 불균형이 동시에 존재하는 현실적 시나리오에서의 성능 분석이 필요합니다.

---

## 참고 자료

**주요 참고 논문 (논문 내 인용 기준):**

1. **Yang et al.** - "Learning with Noisy Class Labels for Instance Segmentation" (본 논문, 제공된 PDF)
2. **He et al.** - "Mask R-CNN", ICCV 2017
3. **Wang et al.** - "Symmetric Cross Entropy for Robust Learning with Noisy Labels", ICCV 2019
4. **Zhang & Sabuncu** - "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels", NeurIPS 2018
5. **Ghosh et al.** - "Making Risk Minimization Tolerant to Label Noise", Neurocomputing 2015
6. **Arpit et al.** - "A Closer Look at Memorization in Deep Networks", ICML 2017
7. **Han et al.** - "Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels", NeurIPS 2018
8. **Patrini et al.** - "Making Deep Neural Networks Robust to Label Noise: A Loss Correction Approach", CVPR 2017
9. **Tanaka et al.** - "Joint Optimization Framework for Learning with Noisy Labels", CVPR 2018
10. **Zhang et al.** - "Understanding Deep Learning Requires Rethinking Generalization", ICLR 2017

**2020년 이후 관련 연구 (일반적 동향 참고):**
- Li et al., "DivideMix: Learning with Noisy Labels as Semi-supervised Learning", ICLR 2020
- Karim et al., "UniCon: Combating Label Noise Through Uniform Selection and Contrastive Learning", CVPR 2022
