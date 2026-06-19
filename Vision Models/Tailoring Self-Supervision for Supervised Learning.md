# Tailoring Self-Supervision for Supervised Learning

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

본 논문의 핵심 주장은 **기존의 자기지도학습(Self-Supervised Learning, SSL) pretext task들이 비지도 표현 학습(unsupervised representation learning)을 위해 설계되었기 때문에, 지도학습(Supervised Learning)에 보조 태스크로 활용될 때 그 이점이 제한적**이라는 것입니다. 특히 가장 널리 사용되던 전역 회전(global rotation) 예측 태스크는 지도학습에 적용 시 성능을 오히려 저하시킬 수 있음을 실험적으로 입증합니다.

이를 해결하기 위해 저자들은 지도학습에 맞게 tailoring된 새로운 pretext task인 **LoRot(Localizable Rotation)**을 제안합니다.

### 주요 기여

1. **세 가지 바람직한 속성 정의**: 지도학습을 위한 보조 자기지도 태스크가 갖춰야 할 세 가지 속성 제시
   - 풍부한 표현 학습 (Rich Representation Learning)
   - 데이터 분포 유지 (Maintaining Data Distribution)
   - 높은 적용 가능성 (High Applicability)

2. **LoRot 제안**: 이미지의 일부 영역만 회전시키는 localizable rotation pretext task 도입 (두 가지 변형: LoRot-I, LoRot-E)

3. **다양한 태스크에서의 SOTA 달성**: OOD 검출, 불균형 분류, 적대적 공격 방어, 이미지 분류, 위치 추정, 전이학습 등에서 성능 향상 입증

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

지도학습에 기존 pretext task를 그대로 적용할 때 발생하는 세 가지 문제를 지적합니다:

**문제 1: 표현의 비보완성(Non-complementary Representations)**

CNN 기반 지도학습 모델은 학습 과정에서 **shortcut learning** 현상이 발생합니다. 즉, 가장 판별력이 높은 특징에만 집중하여 객체의 세부적인 부분을 무시합니다. 기존 전역 회전 예측도 비슷한 판별 영역에 집중하므로 보완적이지 않습니다.

**문제 2: 데이터 분포 이동(Data Distribution Shift)**

전역 회전이나 직소 퍼즐 같은 변환은 훈련 데이터의 분포를 크게 변화시켜, 멀티태스크 학습 시 주 태스크의 성능을 저하시킵니다.

아래 Table 1에서 확인:

| Method | Accuracy |
|--------|----------|
| Baseline | 95.01 |
| +Rot (DA) | 92.76 |
| +Rot (MT) | 93.38 |

**문제 3: 낮은 적용 가능성(Low Applicability)**

기존 방법들은 Parallel-Task Learning이나 Label Augmentation 방식을 사용해 계산 비용이 매우 높습니다. CSI는 LoRot 대비 약 56배 더 많은 훈련 샘플(280M vs 5M)을 사용합니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### 변환 함수 정의

입력 이미지 $X \in \mathbb{R}^{H \times W \times C}$에 대해 변환된 샘플을 다음과 같이 정의합니다:

$$X^{\hat{y}} = T(X|\hat{y}, S) \tag{2}$$

여기서 $S$는 패치 선택 전략(patch selection strategy)이고, $\hat{y}$는 pretext 레이블입니다.

#### 전체 최적화 목적함수 (Multi-task Learning)

$$\min_{\theta} -\frac{1}{N}\sum_{i=1}^{N}\left(\log(P_{u}^{y}(X_{i}^{\hat{y}})) + \lambda \log(P_{v}^{\hat{y}}(X_{i}^{\hat{y}}))\right) \tag{3}$$

- $P_u(X^{\hat{y}}) = \sigma_u(F_\theta(X^{\hat{y}}))$: 주 태스크(분류)의 예측 확률
- $P_v(X^{\hat{y}}) = \sigma_v(F_\theta(X^{\hat{y}}))$: pretext 태스크(LoRot)의 예측 확률
- $F_\theta$: 공유 특징 추출기
- $\lambda$: LoRot 손실의 가중치 하이퍼파라미터 (실험에서 $\lambda = 0.1$로 설정)
- $N$: 배치 크기

#### LoRot-I의 패치 샘플링 전략

LoRot-I는 랜덤 위치/크기의 패치를 샘플링합니다:

$$S(l, p_x, p_y) \begin{cases} l \sim \mathrm{U}(2, \min(\lfloor W/2 \rfloor, \lfloor H/2 \rfloor)), \\ p_x \sim \mathrm{U}(0, W-l), \\ p_y \sim \mathrm{U}(0, H-l) \end{cases} \tag{4}$$

- $l$: 정사각형 패치의 한 변의 길이
- $(p_x, p_y)$: 패치 좌상단 좌표
- 패치 크기를 $\min(H, W)/2$ 이하로 제한하여 과도한 변환 방지

#### 분포 이동 측정 지표 (Affinity Score)

$$\text{Affinity} = \frac{A(m, D'_{val})}{A(m, D_{val})} \tag{1}$$

- $m$: 원본 훈련 데이터로 훈련된 모델
- $A(m, D)$: 모델 $m$의 데이터셋 $D$에서의 정확도
- $D_{val}$: 원본 검증 세트
- $D'_{val}$: 변환된 검증 세트

점수가 낮을수록 분포 이동이 크다는 의미:

| 변환 방법 | Affinity Score |
|---------|---------------|
| Rotation | 58.06 |
| LoRot-I | 93.78 |
| LoRot-E | 90.15 |

---

### 2.3 모델 구조

#### 두 가지 LoRot 변형

**LoRot-I (Implicit Localization)**
- 이미지에서 랜덤 위치, 랜덤 크기의 정사각형 패치를 선택
- 해당 패치를 $\{0°, 90°, 180°, 270°\}$ 중 하나로 회전
- pretext 레이블: 4개 클래스 (회전 각도만)
- 위치 추정이 암묵적으로 학습됨

**LoRot-E (Explicit Localization)**
- 이미지를 $K \times K$ (기본 $2 \times 2$) 그리드로 분할
- 한 셀을 선택하여 $\{0°, 90°, 180°, 270°\}$ 중 하나로 회전
- pretext 레이블: 16개 클래스 (4개 위치 × 4개 각도)
- 위치 추정이 명시적으로 학습됨
- $0°$ 케이스를 모든 셀에 중복 포함(원본 이미지에 더 많은 가중치 부여 목적)

#### 전체 아키텍처 (Multi-task Learning)

```
Input X → Transform T → X^ŷ
                         ↓
                    F_θ (공유 특징 추출기)
                    ↙          ↘
              σ_u               σ_v
         (주 분류기)        (LoRot 분류기)
              ↓                  ↓
         CE Loss           LoRot Loss
              ↘              ↙
            λ-weighted 통합 손실
```

단 **하나의 추가 분류기**만 필요하며, 단일 변환된 입력 배치를 두 태스크가 공유합니다.

---

### 2.4 성능 향상

#### OOD 검출 (AUROC, CIFAR-10 → 각종 OOD 데이터셋)

| Method | SVHN | LSUN | IN | LSUN(FIX) | IN(FIX) | CIFAR-100 | 훈련샘플수 |
|--------|------|------|----|-----------|---------|-----------|--------|
| Cross Entropy | 84.6 | 90.9 | 87.8 | 84.3 | 85.3 | 83.5 | 5M |
| Rotations | 96.1 | 97.3 | 96.9 | 91.0 | 91.8 | 89.1 | **25M** |
| CSI | 96.5 | 96.3 | 96.2 | 92.1 | 92.4 | 90.5 | **280M** |
| **LoRot-I** | 92.6 | **98.6** | **98.0** | **94.4** | **93.6** | 90.1 | **5M** |
| **LoRot-E** | 94.4 | **98.7** | **98.1** | 94.1 | 93.1 | **90.6** | **5M** |

LoRot은 CSI 대비 **단 3.6%의 훈련 시간**으로 대부분의 벤치마크에서 동등하거나 우수한 성능을 달성합니다.

#### 불균형 분류 (CIFAR-10/100, LDAM-DRW 기반)

| Method | CIFAR-10 (0.01) | CIFAR-100 (0.01) |
|--------|----------------|-----------------|
| LDAM-DRW | 77.03 | 42.04 |
| +LoRot-I | 81.13 (+4.10) | 45.82 (+3.78) |
| **+LoRot-E** | **81.82 (+4.79)** | **46.48 (+4.44)** |

#### ImageNet 분류 (ResNet50)

| Method | Top-1 | Top-5 |
|--------|-------|-------|
| Baseline | 76.32 | 92.95 |
| +Rot(MT) | 76.68 | 93.10 |
| **LoRot-I** | **77.71** | **93.60** |
| **LoRot-E** | **77.72** | **93.65** |

---

### 2.5 한계점

논문에서 명시적으로 언급되거나 실험을 통해 관찰된 한계점은 다음과 같습니다:

1. **CutMix와의 비보완성**: LoRot과 CutMix를 함께 사용할 경우 성능 향상이 없음($\pm 0\%$). 두 방법이 유사한 이미지 영역을 수정하여 서로 간섭합니다.

2. **단순 배경 이미지에서의 한계**: SVHN 데이터셋처럼 단색 배경을 가진 이미지에서는 작은 패치 회전이 판별 정보를 제공하지 못해 상대적으로 낮은 OOD 성능을 보입니다.

3. **회전 의미론적 적합성**: 의미론적으로 회전 방향이 중요하지 않은 객체(예: 원형 물체)에 대해서는 효과가 제한될 수 있습니다.

4. **LoRot-I의 패치 크기 트레이드오프**: 패치가 크면 강건성이 높지만 정확도가 낮고, 작으면 그 반대. 최적 크기를 태스크에 따라 튜닝해야 합니다.

5. **ViT 등 최신 아키텍처에서의 검증 부재**: 실험이 주로 ResNet 계열에 국한되어 있어, Vision Transformer 등에서의 효과는 별도 검증이 필요합니다.

---

## 3. 모델의 일반화 성능 향상 가능성 (심층 분석)

### 3.1 일반화 향상의 메커니즘

LoRot이 일반화 성능을 향상시키는 근본적인 메커니즘은 다음 세 가지 경로를 통해 작동합니다:

#### (1) 풍부한 특징 학습을 통한 Shortcut Learning 완화

일반화 성능 저하의 주요 원인 중 하나는 모델이 훈련 데이터의 편향된 특징(shortcut)에 과도하게 의존하는 것입니다. LoRot은 이미지의 다양한 위치에서 회전 예측 퀴즈를 생성함으로써, 모델이 전체 객체의 다양한 부분에서 특징을 추출하도록 강제합니다.

CAM(Class Activation Map) 분석에서, LoRot으로 훈련된 모델은 객체의 더 넓은 영역에 걸쳐 활성화를 보여주어 **풍부하고 다양한 특징 학습**이 이루어졌음을 확인합니다.

#### (2) 분포 유지를 통한 안정적인 학습

$$\text{Affinity(LoRot-I)} = 93.78\% \quad \text{vs} \quad \text{Affinity(Rotation)} = 58.06\%$$

LoRot은 이미지의 대부분을 원본 상태로 유지하면서 작은 패치만 변환하므로, 데이터 분포를 거의 그대로 유지합니다. 이는 멀티태스크 학습에서 주 태스크(지도학습)와 보조 태스크(LoRot)가 충돌 없이 협력하게 합니다.

#### (3) 위치 추정 능력 향상

LoRot으로 훈련된 모델은 자연스럽게 **약지도 객체 위치 추정(Weakly Supervised Object Localization)** 능력을 획득합니다:

| Threshold | Baseline | CutMix | LoRot-I | LoRot-E |
|-----------|---------|--------|---------|---------|
| 0.5 | 46.72 | 47.39 | 49.73 | **50.24** |
| 0.6 | 31.55 | 30.24 | 35.49 | **36.07** |
| 0.7 | 14.49 | 13.86 | 17.21 | **17.81** |

이 위치 추정 능력은 일반화 성능의 핵심 요소로, 모델이 다양한 하위 태스크(object detection, segmentation 등)에 더 잘 적응하게 합니다.

### 3.2 전이학습에서의 일반화

ImageNet pretrained 가중치를 사용한 다운스트림 태스크에서:

| Pretrained | RetinaNet (AP) | SOLOv2 (AP) |
|-----------|---------------|------------|
| Baseline | 33.8 | 33.7 |
| LoRot-I | **35.3 (+1.5)** | **34.5 (+0.8)** |
| LoRot-E | 35.2 | 34.4 |

이는 LoRot이 **도메인 전반에 걸쳐 일반화 가능한 특징**을 학습한다는 강력한 증거입니다.

### 3.3 데이터 불균형 상황에서의 일반화

불균형 분류는 일반화의 극단적인 테스트 케이스입니다. LoRot은 소수 클래스에 대한 특징 다양성을 증가시켜:

- CIFAR-100 (Imbalance Ratio 0.01): **+4.44%p 향상** (42.04 → 46.48)
- 심한 불균형 상황에서 더 큰 효과를 발휘 (적은 샘플에서의 일반화 능력 입증)

### 3.4 OOD 일반화 메커니즘

t-SNE 시각화에서 확인된 바와 같이, LoRot으로 훈련된 모델의 특징 공간은:
- 인분포(in-distribution) 클래스의 클러스터가 더 컴팩트함
- 아웃오브분포(OOD) 샘플이 인분포 클러스터에서 더 명확하게 분리됨

이는 LoRot이 단순히 알려진 클래스를 더 잘 분류하는 것을 넘어, **"알 수 없는 것을 알아보는 능력"**을 향상시킨다는 것을 의미합니다.

### 3.5 다양한 학습 전략과의 호환성

| 기반 방법 | CIFAR-10 | CIFAR-100 |
|---------|---------|---------|
| SupCLR | 95.75 | 76.52 |
| SupCLR + LoRot-I | **96.79 (+1.04)** | **78.78 (+2.26)** |

LoRot은 데이터 증강(Mixup, AutoAug, RandAug)이나 대조 학습(SupCLR)과 결합할 때도 일관적인 성능 향상을 보여주어, **범용적인 일반화 부스터**로서의 역할을 합니다.

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 향후 연구에 미치는 영향

#### (1) Pretext Task 설계 패러다임의 전환

이 논문은 SSL pretext task를 설계할 때 "단순히 표현 학습 효과가 좋은가"가 아니라 **"지도학습과 얼마나 보완적인가"**라는 새로운 질문을 던집니다. 세 가지 속성(풍부한 표현, 분포 유지, 높은 적용성)은 앞으로의 pretext task 설계에 있어 평가 기준으로 활용될 수 있습니다.

#### (2) 효율적인 자기지도 보조 학습 연구 활성화

LoRot은 단 5M개의 훈련 샘플(CSI의 1/56 수준)로 SOTA에 준하는 성능을 보여줌으로써, **경량 보조 자기지도 태스크** 연구의 가능성을 보여줍니다. 이는 Edge AI, 모바일 환경 등 자원 제한 환경에서의 연구에 큰 영향을 미칠 것입니다.

#### (3) 다양한 응용 분야로의 확장

- **의료 영상**: 데이터 불균형이 심각한 의료 분야에서 LoRot의 불균형 분류 성능 향상 효과는 직접적으로 적용 가능합니다.
- **자율주행**: OOD 검출 능력 향상은 미지의 도로 상황에 대한 강건성을 높입니다.
- **보안 시스템**: 적대적 공격에 대한 내성 향상은 보안이 중요한 응용 분야에 직접적으로 기여합니다.

#### (4) 멀티태스크 학습 관점의 재정립

LoRot은 보조 태스크의 **입력 분포 호환성**이 멀티태스크 학습의 성공에 얼마나 중요한지를 보여줍니다. 이는 향후 멀티태스크 학습 연구에서 태스크 간 분포 정렬을 명시적으로 고려하는 방향으로 영향을 미칠 것입니다.

---

### 4.2 앞으로 연구 시 고려할 점

#### (1) Vision Transformer(ViT) 아키텍처 적용

LoRot은 주로 CNN 기반 모델(ResNet)에서 검증되었습니다. ViT는 패치 기반으로 이미지를 처리하므로, LoRot의 패치 회전이 ViT의 내부 어텐션 메커니즘과 어떻게 상호작용하는지 연구가 필요합니다. 특히 LoRot-E의 그리드 기반 접근이 ViT의 패치 분할과 자연스럽게 연계될 수 있습니다.

#### (2) 자동화된 Pretext Task 탐색

LoRot의 설계 원칙(세 가지 속성)을 기반으로 **Neural Architecture Search(NAS)**나 **AutoML** 방식으로 최적의 pretext task를 자동으로 탐색하는 연구가 가능합니다. 주어진 데이터셋과 주 태스크에 따라 최적의 보조 태스크를 자동 선택하는 것이 목표가 될 수 있습니다.

#### (3) Few-shot Learning과의 결합

LoRot이 불균형 분류에서 보인 강력한 성능은 Few-shot Learning과의 시너지 가능성을 암시합니다. 매우 적은 수의 샘플(1-shot, 5-shot)에서 LoRot의 풍부한 특징 학습 효과가 더욱 두드러질 것으로 예상됩니다.

#### (4) 연속 학습(Continual Learning)과의 통합

LoRot이 다양한 객체 부분의 특징을 학습한다는 점에서, 새로운 태스크 학습 시 이전 태스크의 특징을 보존하는 **Catastrophic Forgetting 완화** 가능성이 있습니다. 연속 학습 환경에서의 LoRot 효과 연구가 필요합니다.

#### (5) 패치 선택의 지능화

현재 LoRot-I는 랜덤 패치 선택을 사용합니다. **Saliency map이나 Attention map을 활용한 적응적 패치 선택**을 통해 더 유의미한 영역에 회전을 적용함으로써 성능을 더욱 향상시킬 수 있습니다. 단, 이 경우 계산 비용과 데이터 분포 변화에 대한 추가 분석이 필요합니다.

#### (6) 다중 모달(Multi-modal) 학습으로의 확장

LoRot의 원칙은 텍스트-이미지 멀티모달 학습으로 확장 가능합니다. 예를 들어, 이미지의 특정 영역 회전과 이에 대응하는 텍스트 기술 간의 관계를 활용한 멀티모달 pretext task 설계가 가능합니다.

#### (7) 이론적 분석 심화

LoRot의 효과에 대한 경험적 증거는 충분하지만, **왜** 특정 크기의 패치가 강건성과 정확도 사이의 최적 트레이드오프를 만들어내는지에 대한 이론적 분석이 부족합니다. 정보이론적 관점(예: Mutual Information 최대화)이나 PAC-Bayes 이론을 통한 일반화 bound 분석이 향후 연구 주제가 될 수 있습니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

논문에서 직접 비교하거나 참조한 2020년 이후 연구들을 중심으로 분석합니다.

### 5.1 직접 비교된 주요 방법들

| 연구 | 연도 | 접근법 | 핵심 아이디어 | LoRot 대비 강/약점 |
|-----|------|--------|-------------|-----------------|
| **CSI** (Tack et al.) | 2020 | 대조 학습 기반 OOD | 분포 이동된 인스턴스 대조 학습 | 높은 OOD 성능, **280M 샘플** 필요 |
| **SupCLR** (Khosla et al.) | 2020 | 지도 대조 학습 | 레이블 정보를 대조 학습에 활용 | 배치 크기에 민감, 높은 메모리 사용 |
| **SLA+SD** (Lee et al.) | 2020 | 레이블 증강 | 클래스×변환 Cartesian 레이블 공간 | 추론 시 모든 변환 필요, **20M 샘플** |
| **SSP** (Yang & Xu) | 2020 | 자기지도 사전학습 | 불균형 학습을 위한 SSL 사전훈련 | 사전훈련 단계 추가 필요 |

### 5.2 LoRot의 포지셔닝

```
계산 효율성
    높음 ↑
         |
  LoRot  ●──────── (5M 샘플, 낮은 비용)
         |
SLA+SD  ●──────── (20M 샘플, 중간 비용)
         |
Rotations●──────── (25M 샘플, 중간 비용)
         |
SupCLR  ●──────── (70M 샘플, 높은 비용)
         |
   CSI  ●──────── (280M 샘플, 매우 높은 비용)
         |
    낮음 ↓
         ←────────────────────────────→
      낮음           성능             높음
```

### 5.3 LoRot 이후의 관련 연구 동향 (2022 이후)

아래 내용은 논문 제출 이후의 연구 동향을 **일반적인 AI 연구 트렌드**를 바탕으로 기술하는 것입니다. 구체적인 논문 인용에 대해서는 실제 논문을 직접 확인하시기 바랍니다.

**주요 연구 방향:**

1. **Foundation Model에서의 SSL**: CLIP, DINO, DINOv2 등 대규모 사전학습 모델에서 적절한 pretext task 설계가 더욱 중요해지고 있으며, LoRot의 "분포 유지" 원칙이 이 맥락에서도 관련성을 가집니다.

2. **Masked Image Modeling(MIM)의 부상**: MAE(Masked Autoencoders, He et al., 2022)는 이미지의 일부를 마스킹하여 복원하는 방식으로, LoRot과 유사하게 **지역적 변환**을 활용합니다. 그러나 MAE는 비지도 사전학습에, LoRot은 지도학습 보조에 초점을 맞춥니다.

3. **데이터 증강과 SSL의 통합**: LoRot과 같이 변환 기반 SSL을 데이터 증강과 결합하는 연구가 지속되고 있으며, 두 방법의 최적 결합 방식을 찾는 연구가 진행 중입니다.

---

## 참고 자료

**주 논문:**
- Moon, W., Kim, J.-H., & Heo, J.-P. (2022). **Tailoring Self-Supervision for Supervised Learning**. arXiv:2207.10023v1. [https://arxiv.org/abs/2207.10023](https://arxiv.org/abs/2207.10023)

**논문 내 주요 참고문헌:**
- Gidaris, S., Singh, P., & Komodakis, N. (2018). **Unsupervised representation learning by predicting image rotations**. ICLR 2018.
- Tack, J., Mo, S., Jeong, J., & Shin, J. (2020). **CSI: Novelty detection via contrastive learning on distributionally shifted instances**. NeurIPS 2020.
- Khosla, P., et al. (2020). **Supervised contrastive learning**. NeurIPS 2020.
- Lee, H., Hwang, S.J., & Shin, J. (2020). **Self-supervised label augmentation via input transformations**. ICML 2020.
- Hendrycks, D., Mazeika, M., Kadavath, S., & Song, D. (2019). **Using self-supervised learning can improve model robustness and uncertainty**. NeurIPS 2019.
- Yun, S., et al. (2019). **CutMix: Regularization strategy to train strong classifiers with localizable features**. ICCV 2019.
- Cao, K., et al. (2019). **Learning imbalanced datasets with label-distribution-aware margin loss**. NeurIPS 2019.
- He, K., et al. (2016). **Deep residual learning for image recognition**. CVPR 2016.
- Geirhos, R., et al. (2020). **Shortcut learning in deep neural networks**. Nature Machine Intelligence.
- Gontijo-Lopes, R., et al. (2020). **Tradeoffs in data augmentation: An empirical study**. ICLR 2020.

**GitHub 코드:**
- [https://github.com/wjun0830/Localizable-Rotation](https://github.com/wjun0830/Localizable-Rotation)

> **정확도 주의사항**: 본 답변은 제공된 논문 PDF(arXiv:2207.10023v1)를 직접 분석한 내용을 기반으로 합니다. "2020년 이후 관련 최신 연구 비교" 섹션 중 LoRot 발표 이후(2022년 이후)의 연구 동향은 일반적인 AI 연구 트렌드를 기반으로 기술하였으며, 구체적인 후속 연구 인용은 직접 논문 검색을 통해 확인하시기 바랍니다.
