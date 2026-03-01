# Patch SVDD: Patch-level SVDD for Anomaly Detection and Segmentation

Patch SVDD는 Deep SVDD를 패치 단위로 확장하고 self‑supervised 학습과 다단계(hierarchical) 인코더를 결합해, 산업 이미지에서 이상 탐지와 픽셀 단위 분할을 동시에 수행하면서 MVTec AD에서 당시 SOTA 수준의 AUROC(검출 0.921, 분할 0.957)를 달성한 방법입니다. 특히 “정상 패치 메모리 + 최근접 이웃 거리”라는 단순한 스코어링 구조와 self‑supervised로 형성된 다중 모드 특성 공간을 통해 일반화 가능성을 높인 것이 핵심입니다.[^1_1][^1_2][^1_3]

***

## 1. 핵심 주장과 주요 기여

- **Deep SVDD의 패치 기반 확장(Patch SVDD)**: 이미지 전체를 하나의 벡터로 매핑하던 Deep SVDD를, 패치 단위 인코딩과 최근접 이웃 기반 스코어링으로 확장하여 패치 수준 이상 탐지 및 픽셀 수준 분할이 가능하도록 함.[^1_3][^1_1]
- **Self‑supervised + 변형된 SVDD 손실 결합**: 공간적으로 인접한 패치들을 서로 가깝게, 동시에 상대 위치 예측 pretext task를 통해 의미(semantic)를 반영한 다중 모드 클러스터를 형성하도록 학습하는 신규 목적함수 $L_{\text{PatchSVDD}}$ 제안.[^1_1]
- **다단계(hierarchical) 인코딩과 멀티스케일 융합**: 서로 다른 receptive field(예: 32, 64 픽셀)를 갖는 인코더를 계층적으로 구성해 크기가 다른 결함을 잡아내고, 각 스케일의 anomaly map을 곱셈으로 융합해 정밀한 위치 추정 달성.[^1_1]
- **성능 향상**: MVTec AD에서 이미지 수준 AUROC 0.921, 픽셀 수준 AUROC 0.957로, 이전 SOTA 대비 이상 탐지 +9.8pt, 분할 +7.0pt 향상 보고.[^1_2][^1_1]
- **일반화와 표현 분석**: t‑SNE, intrinsic dimension, 손실 구성(ablation), 랜덤 인코더 및 raw patch 최근접 이웃 분석을 통해, self‑supervised 패치 표현이 더 낮은 내재 차원과 의미 기반 클러스터를 형성하며, 단순한 패치 거리 기반 방법도 상당한 분리력을 갖는다는 점을 보임.[^1_4][^1_1]

***

## 2. 논문이 다루는 문제

### 2.1 문제 정의

- **목표**:
    - *이상 탐지*: 입력 이미지가 이상(결함)을 포함하는지 binary 결정.
    - *이상 분할*: 이미지 내 어떤 픽셀/영역이 결함인지 localization (anomaly map).[^1_1]
- **학습 설정**:
    - 제조 라인 등에서 이상 샘플은 희소하거나 수집이 어려우므로, *one‑class* 설정에서 정상 이미지만으로 모델을 학습.[^1_1]
- **데이터셋**:
    - MVTec AD (15 클래스, object 10, texture 5)에서 클래스별 정상(train) + 정상/이상(test) 이미지와 픽셀 수준 마스크 제공.[^1_5][^1_1]
- **기존 한계**:
    - Autoencoder/재구성 기반 방법은 복잡한 배경·고주파 텍스처에서 재구성이 과도하게 잘 되어 이상이 “씻겨 나가거나”, 후처리 없이 정확한 localization이 어려움.[^1_1]
    - Deep SVDD와 같은 전역(one‑vector) 표현 기반 one‑class 방법은 “어디가 이상인지”는 말해주지 못하고, 이미지 전체의 정상/이상만 판별.[^1_1]
    - 패치 수준으로 확장하려 해도, 객체/배경/텍스처가 섞인 패치들의 **intra‑class variation**이 커서, 단일 중심 구로 묶는 전통적 SVDD는 표현-내용 간 연결이 약해지고 검출 성능이 떨어짐.[^1_1]

Patch SVDD는 **(1) 패치 단위 표현 학습**, **(2) self‑supervised 기반 의미 클러스터 형성**, **(3) 멀티스케일 패치 탐색 + 최근접 이웃 스코어링**으로 이 문제를 해결하려고 합니다.[^1_3][^1_1]

***

## 3. 제안 방법 (수식 포함)

### 3.1 Deep SVDD 복습

Deep SVDD는 encoder $f_\theta$가 정상 데이터 $\{x_i\}$를 feature 공간의 작은 구(hypersphere) 안으로 모으도록 학습합니다.[^1_1]

손실은

$$
L_{\text{SVDD}} = \sum_i \| f_\theta(x_i) - c \|_2^2
$$

이며,[^1_1]

구의 중심은 사전 계산된 평균

$$
c = \frac{1}{N}\sum_{i=1}^{N} f_\theta(x_i)
$$

로 정의됩니다.[^1_1]

테스트 시 이미지 $x$의 이상 점수는 중심과의 거리 $\|f_\theta(x)-c\|_2^2$입니다.[^1_1]

### 3.2 패치 기반 확장과 수정된 SVDD 손실

이미지 전체 대신 패치 $p$를 입력으로 하는 패치 인코더 $f_\theta$를 학습하고, 각 패치의 특징을 이용해 이상을 판별하도록 Deep SVDD를 확장합니다.[^1_1]

하지만 서로 다른 위치의 패치는 배경·객체·가장자리 등 내용이 크게 달라, 모두를 단일 중심 $c$로 모으는 것은 부적절합니다. 이를 피하기 위해 중심을 명시적으로 두지 않고, *공간적으로 인접한 패치 쌍*이 feature 공간에서 서로 가깝도록 만드는 수정된 손실을 사용합니다.[^1_1]

$$
L_{\text{SVDD}'} = \sum_{i,i'} \| f_\theta(p_i) - f_\theta(p_{i'}) \|_2^2,
$$

여기서 $p_{i'}$는 $p_i$ 주변(3×3 그리드 이웃)에서 샘플된 패치입니다.[^1_1]

이 손실은 “**인접 패치는 비슷한 의미를 가지므로, feature도 가깝게**”라는 가정 하에, 의미적으로 유사한 패치들을 자연스럽게 모으도록 유도합니다.[^1_1]

### 3.3 Self‑supervised 상대 위치 예측 손실

Doersch et al.(2015)의 *context prediction* pretext task를 도입해, 인코더가 단순히 “가까운 것끼리 뭉치게”만이 아니라 **위치 정보를 반영한 의미 표현**을 학습하도록 합니다.[^1_3][^1_1]

- 패치 $p_1$을 임의로 선택하고, 3×3 이웃 중 하나에서 $p_2$를 샘플링.[^1_1]
- 상대 위치 레이블 $y\in\{0,\dots,7\}$ (8방향)을 정의.
- classifier $C_\phi$가 $(f_\theta(p_1), f_\theta(p_2))$로부터 $y$를 예측.

self‑supervised 손실은

$$
L_{\text{SSL}} = \text{Cross-entropy}\bigl(y,\; C_\phi(f_\theta(p_1), f_\theta(p_2))\bigr).
$$

[^1_1]

### 3.4 최종 Patch SVDD 목적함수

최종 학습 목표는 다음과 같이 두 손실을 가중 합한 형태입니다.[^1_1]

$$
L_{\text{PatchSVDD}} = \lambda L_{\text{SVDD}'} + L_{\text{SSL}},
$$

여기서 $\lambda$는 “패치 간 집합성(모으기)”과 “self‑supervised 의미 정보” 사이의 트레이드오프를 조절하는 하이퍼파라미터입니다.[^1_1]

- $\lambda$가 크면 패치들을 더 강하게 모으고(저차원, 콤팩트),
- $\lambda$가 작으면 self‑supervised signal 비중이 커져 더 풍부하지만 분산된 표현을 만듭니다.[^1_1]

실험적으로, **object 클래스는 작은 $\lambda$**, **texture 클래스는 큰 $\lambda$**에서 성능이 더 좋다는 것을 보여주며, 이는 객체 이미지의 패치들이 위치·형태에 따라 의미가 크게 달라 self‑supervision의 이득이 크기 때문이라고 해석합니다.[^1_1]

### 3.5 계층적(hierarchical) 인코딩과 멀티스케일

이상은 다양한 크기로 나타나므로, 단일 receptive field로는 작은 스크래치와 큰 구조 결함을 동시에 잘 잡기 어렵습니다. 이를 위해 계층적 인코더를 정의합니다.[^1_1]

- 작은 인코더 $f_{\text{small}}$: receptive field $K=32$.[^1_1]
- 큰 인코더 $f_{\text{big}}$: 입력 패치를 2×2 서브패치로 나눈 뒤, 각각을 $f_{\text{small}}$로 인코딩하고 aggregate 함수 $g_{\text{big}}$로 합친 형태.

$$
f_{\text{big}}(p) = g_{\text{big}}\bigl(f_{\text{small}}(p_{1}), f_{\text{small}}(p_{2}), f_{\text{small}}(p_{3}), f_{\text{small}}(p_{4})\bigr).
$$

[^1_1]

각 인코더는 자신에 맞는 패치 크기($K=32,64$)와 stride($S=4,16$)로 패치를 슬라이딩하며, 동일한 $L_{\text{PatchSVDD}}$ 목표로 학습됩니다.[^1_1]

### 3.6 이상 점수와 anomaly map 생성

1. **정상 패치 feature 메모리 구축(훈련 단계)**
    - 모든 정상 이미지에서 모든 패치 $p_{\text{normal}}$을 추출하고, 인코더를 통해
$\{f_\theta(p_{\text{normal}})\}$를 계산해 메모리에 저장합니다.[^1_1]
2. **패치 수준 이상 점수(테스트 단계)**
    - 테스트 이미지의 각 패치 $p$에 대해, feature 공간에서 가장 가까운 정상 패치까지의 L2 거리를 이상 점수로 정의:

$$
A_\theta^{\text{patch}}(p) = \min_{p_{\text{normal}}} \| f_\theta(p) - f_\theta(p_{\text{normal}}) \|_2^2.
$$

[^1_1]

- 계산량을 줄이기 위해 approximate k‑NN 라이브러리(NGT)를 사용합니다.[^1_1]

3. **픽셀 수준 anomaly map**
    - 패치 단위 점수를 해당 패치가 덮는 모든 픽셀에 분배하고, 중복되는 패치들에서 오는 점수의 평균을 취해 anomaly map $M$을 얻습니다.[^1_1]
    - 두 스케일의 인코더로부터 얻은 맵 $M_{\text{small}}, M_{\text{big}}$을 요소별 곱으로 통합:

$$
M_{\text{multi}} = M_{\text{small}} \odot M_{\text{big}}.
$$

[^1_1]

4. **이미지 수준 이상 점수**
    - 최종 이미지 이상 점수는 anomaly map의 최대값으로 정의:

$$
A_\theta^{\text{image}}(x) = \max_{i,j} M_{\text{multi}}(x)_{ij}.
$$

[^1_1]

이 구조 덕분에 **패치 수준 최근접 이웃 기반 one‑class 분류**로 이미지 전체의 이상 여부와 픽셀 수준 위치를 동시에 얻을 수 있습니다.[^1_1]

***

## 4. 모델 구조

- **인코더**:
    - 오직 convolution layer로 구성된 CNN, bias 제거, LeakyReLU($\alpha=0.1$) 활성화 사용.[^1_1]
    - feature 차원 $D$는 실험적으로 64일 때 내재 차원이 포화되면서 성능이 안정화됨.[^1_1]
- **self‑supervised classifier $C_\phi$**:
    - 두 패치 feature 차이 $f_\theta(p_1)-f_\theta(p_2)$를 입력으로 하는 2‑layer MLP(각 128 hidden units) 구조.[^1_1]
- **계층적 구조**:
    - 큰 인코더 내부에 작은 인코더를 중첩시킨 hierarchical 설계로, 파라미터 수를 크게 늘리지 않으면서 멀티스케일 정보를 포착.[^1_1]
    - 동일 파라미터 수를 갖는 non‑hierarchical 대안과 비교 시 계층 구조가 AUROC를 추가로 개선함.[^1_1]
- **추론 속도 및 구현**:
    - PyTorch 기반 구현, RTX 2080 Ti 기준 MVTec 이미지 1장 당 약 0.48초 검사(approx. NN 포함).[^1_1]

***

## 5. 성능 향상과 한계

### 5.1 정량적 성능

MVTec AD 전체 15 클래스 평균 결과는 다음과 같습니다.[^1_3][^1_1]

- **이상 탐지 (이미지 수준 AUROC)**
    - Deep SVDD: 0.592
    - GEOM: 0.672
    - GANomaly: 0.762
    - ITAE: 0.839
    - **Patch SVDD: 0.921** (이전 SOTA 대비 +9.8pt)[^1_2][^1_1]
- **이상 분할 (픽셀 수준 AUROC)**
    - 간단 L2‑AE: 0.804
    - SSIM‑AE: 0.818
    - VE‑VAE: 0.861
    - VAE‑Proj: 0.893
    - **Patch SVDD: 0.957** (이전 SOTA 대비 +7.0pt)[^1_1]

클래스별로도 대부분 0.9 이상 AUROC를 달성하며, 특히 toothbrush, bottle, tile 등 다양한 object/texture 클래스에서 강한 성능을 보입니다.[^1_1]

### 5.2 분석적 결과

- **손실 구성 효과**:
    - 단순 $L_{\text{SVDD}}$에서 $L_{\text{SVDD}'}$로 바꿀 때도 성능이 상승하며, $L_{\text{SSL}}$을 추가하면 탐지/분할 AUROC가 각각 0.921/0.957로 크게 향상됩니다.[^1_1]
- **t‑SNE 시각화**:
    - self‑supervised 없이 학습한 feature는 하나의 덩어리(uni‑modal cluster)에 모이는 반면, $L_{\text{SSL}}$을 포함하면 위치·색 등 의미에 따라 잘 분리된 다중 모드 클러스터가 형성됩니다.[^1_1]
- **Intrinsic dimension(ID)**:
    - $L_{\text{PatchSVDD}}$로 학습한 feature가 가장 낮은 내재 차원을 가져, “낮은 차원에 잘 정렬된, 분리 가능한 표현”임을 보여줍니다.[^1_1]
- **랜덤 인코더 및 raw patch 최근접 이웃**:
    - 무작위 초기화 CNN feature나 raw 패치 자체에 대한 최근접 이웃만으로도 몇몇 클래스(wood, bottle 등)에서 꽤 높은 AUROC를 얻을 수 있음을 보여주며, 이는 정상 패치들 간 유사성이 매우 높으면 단순 거리 기반 방법도 강력할 수 있음을 시사합니다.[^1_1]


### 5.3 한계

- **메모리와 계산량**:
    - 모든 정상 패치 feature를 메모리에 저장하고 각 테스트 패치마다 k‑NN 거리를 계산해야 하므로, 클래스 수·해상도가 커질수록 메모리와 추론 시간이 선형 이상으로 증가.[^1_1]
- **클래스/도메인별 모델 학습**:
    - 논문에서는 클래스별로 별도 학습을 수행하며, 서로 다른 도메인(예: 전혀 다른 산업 품목)으로의 zero‑shot 일반화나 multi‑class 공용 모델에 대한 논의는 제한적입니다.[^1_1]
- **고수준 의미 인식의 한계**:
    - self‑supervision과 위치 정보로 의미를 어느 정도 반영하지만, 전체 object 레벨의 semantics(예: 조립 구조, 기능적 이상)까지 포착하는 데에는 한계가 있어, 구조적/기하학적 이상에는 강하지만 의미적 이상(예: 잘못된 숫자 인쇄)에는 취약할 수 있습니다.[^1_1]

***

## 6. 일반화 성능 관점에서의 고찰

Patch SVDD의 일반화 성능과 그 향상 가능성은 **표현 학습 방식, 스코어링 구조, 분석적 관찰**에서 읽을 수 있습니다.[^1_1]

### 6.1 표현 학습: self‑supervised + 패치 중심

- self‑supervised 상대 위치 예측은 패치를 단순한 텍스처 벡터가 아닌, **위치·형태·구조 정보를 반영한 의미 기반 embedding**으로 만들어 줍니다.[^1_1]
- t‑SNE 결과에서 보이듯, object 클래스에서 서로 다른 위치의 패치들이 self‑supervised를 통해 잘 분리된 클러스터를 형성하며, 이는 새로운 결함 패치가 기존 클러스터와 멀리 떨어지기 쉽도록 만듭니다.[^1_1]
- 낮은 intrinsic dimension은 feature가 저차원 다양체 위에 잘 정렬되어 있음을 시사하며, 이는 일반적으로 **샘플 효율성과 일반화**에 유리한 구조입니다.[^1_1]


### 6.2 스코어링 구조: 최근접 이웃 기반 one‑class

- 이상 점수는 항상 “가장 가까운 정상 패치까지의 거리”로 정의되므로, 훈련 분포에서 크게 벗어나는 어떤 패턴이든 이상으로 간주됩니다.[^1_1]
- 이 구조는 decision boundary를 parametric하게 근사하지 않고, **정상 데이터 manifold 주변에 비선형적으로 감긴 non‑parametric 경계**를 암묵적으로 형성한다는 장점이 있습니다.[^1_1]
- raw patch 및 랜덤 인코더 분석은, 표현이 비교적 단순해도 정상 패치 manifold가 충분히 조밀하면 좋은 일반화 성능을 낼 수 있음을 보여줍니다.[^1_1]


### 6.3 일반화의 잠재적 약점

- **도메인 이동(domain shift)**:
    - 학습 시 사용한 MVTec 내 한 클래스의 정상 데이터 분포에 강하게 특화되어 있으므로, 조명·재질·카메라 변화 등 큰 distribution shift가 발생하면, self‑supervised로 얻은 표현이 충분히 견고한지 검증이 필요합니다.[^1_1]
- **하이퍼파라미터 민감도**:
    - $\lambda$와 feature 차원 $D$에 따라 클래스별 성능 편차가 크며, 특정 클래스에 튜닝된 설정이 다른 클래스에선 최선이 아닐 수 있습니다.[^1_1]


### 6.4 일반화 성능 향상을 위한 방향

- **사전학습(backbone)과 결합**:
    - PaDiM, PatchCore 등은 ImageNet 사전학습 CNN을 backbone으로 사용해 다양한 도메인에 대한 강한 generalization을 확보합니다.[^1_6][^1_7][^1_8][^1_9]
    - Patch SVDD 구조를 ResNet·ViT 등 대형 사전학습 모델 위에 얹으면, self‑supervised와 one‑class 특성을 유지하면서도 더 넓은 도메인으로 일반화를 확대할 수 있습니다(후술).
- **더 강한 self‑supervision**:
    - 상대 위치 예측 대신/추가로 contrastive, masked image modeling, patch inpainting 등 최근 self‑supervised 기법을 결합하면, noise나 domain shift에 더 강인한 표현을 얻을 수 있다는 가능성이 있습니다.[^1_10][^1_11]
- **메모리·계산 구조 개선**:
    - PatchCore처럼 coreset selection을 통한 **대표 패치 서브샘플링**이나, Gaussian 모델링(예: PaDiM)을 도입하면, 메모리를 줄이면서도 manifold 구조를 더 매끈하게 근사해 일반화 경계를 안정화할 수 있습니다.[^1_7][^1_8][^1_9][^1_6]

***

## 7. 2020년 이후 관련 최신 연구 비교

Patch SVDD 이후, 산업 이미지 이상 탐지 분야에서는 “패치 기반 + 사전학습 표현 + 단순 one‑class 스코어링”이라는 패러다임이 확산되었습니다. 대표적인 방법들을 Patch SVDD와 비교하면 아래와 같습니다.[^1_8][^1_9][^1_6][^1_7]

### 7.1 주요 방법 개요

| 방법 | 연도/발표 | 핵심 아이디어 | 표현 소스 | 스코어링 방식 | 비고 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| **Patch SVDD** | ACCV 2020(arXiv 2006.16067) | 패치 수준 Deep SVDD + self‑supervised 상대 위치 예측 + 멀티스케일 | scratch CNN | 패치 k‑NN 거리 + 멀티스케일 맵 곱 | MVTec에서 당시 SOTA 검출/분할 AUROC.[^1_1][^1_2] |
| **PaDiM** | ICPR 2020 | 프리트레인 CNN 다중 레이어 패치 feature를 위치별 다변량 Gaussian으로 모델링, Mahalanobis distance 기반 스코어링.[^1_6][^1_7] | ImageNet CNN | 위치별 Gaussian, Mahalanobis | 높은 성능 + 메모리 효율(평균·공분산만 저장). |
| **CutPaste** | CVPR 2021 | 패치를 잘라 다른 위치에 붙이는 self‑supervised 증강을 분류하는 pretext task + one‑class generative classifier.[^1_12][^1_13] | scratch 또는 ImageNet CNN | 학습된 one‑class classifier 출력 | 간단한 증강 기반 self‑supervision으로 높은 AUROC 달성. |
| **PatchCore** | CVPR 2022 | 사전학습 CNN patch feature 메모리에서 coreset selection으로 대표 패치 집합 구성 후 k‑NN anomaly score 계산.[^1_8][^1_9] | ImageNet CNN | 코어셋 기반 패치 k‑NN + 이미지 최대값 | MVTec 이미지 AUROC 99.6%까지 달성, 빠른 추론. |
| **DRAEM** | ICCV 2021 | 정상 이미지에 합성 anomalous 패턴을 삽입해 “재구성+세그멘테이션” 네트워크를 discriminative하게 학습.[^1_14][^1_15] | U‑Net 타입 CNN | 재구성/마스크 간 차이 | unsupervised 기준 SOTA에 근접/초과하는 성능. |
| **Inpainting Transformer (InTra)** | 2021 | 전적으로 self‑attention 기반 Transformer로 패치 inpainting 수행, reconstruction error로 이상 탐지/분할.[^1_10] | Transformer | 재구성 오차 | MVTec에서 기존 AE류보다 우수한 분할 성능 보고. |
| **VT‑ADL (ViT 기반 AD)** | 2021 | Vision Transformer로 패치 embedding 후 Gaussian mixture로 이상 스코어링.[^1_11] | Vision Transformer | Gaussian density | Transformer 기반 AD의 초기 연구. |

### 7.2 Patch SVDD와 비교한 특징

1. **프리트레인 vs scratch**
    - Patch SVDD는 비교적 얕은 CNN을 scratch로 학습하는 반면, PaDiM·PatchCore·CutPaste(Transfer)·VT‑ADL 등은 ImageNet으로 학습된 강력한 backbone을 사용해 **도메인 일반화**에서 강점을 가집니다.[^1_9][^1_12][^1_11][^1_6][^1_7][^1_8][^1_1]
    - 반대로, Patch SVDD는 특정 산업 도메인에 맞춘 상대 위치 self‑supervision으로 **도메인 특화 표현**을 학습해, 프리트레인 없이도 높은 성능을 달성한 점이 장점입니다.[^1_1]
2. **스코어링 구조**
    - Patch SVDD와 PatchCore는 모두 “정상 패치 메모리 + k‑NN 거리”라는 구조를 공유하지만, PatchCore는 coreset selection으로 메모리와 속도를 크게 개선하고, 이미지-level AUROC 99.6%로 Patch SVDD를 상당히 상회합니다.[^1_8][^1_9]
    - PaDiM은 위치별 Gaussian 파라미터만 저장해 메모리 footprint를 줄이는 대신, 분포를 Gaussian으로 가정하는 parametric 접근을 취합니다.[^1_6][^1_7]
3. **self‑supervision의 역할**
    - Patch SVDD와 CutPaste는 모두 self‑supervised pretext task(상대 위치 vs Cut‑Paste)를 통해 의미 있는 패치 표현을 학습한다는 공통점이 있습니다.[^1_12][^1_13][^1_1]
    - CutPaste는 pretext 이후 별도의 one‑class classifier를 학습하는 2‑stage 구조인 반면, Patch SVDD는 self‑supervision과 one‑class 목적을 하나의 encoder에 통합해 jointly 학습합니다.[^1_12][^1_1]
4. **재구성 vs 비‑재구성**
    - DRAEM, InTra, VT‑ADL 등은 reconstruction 기반으로 이상을 탐지하지만, Patch SVDD/PaDiM/PatchCore/CutPaste는 “정상 feature 분포 근처/원거리”라는 **비‑재구성 방식**으로 접근합니다.[^1_14][^1_11][^1_7][^1_9][^1_10][^1_6][^1_8][^1_1]
    - 재구성 기반은 직관적인 시각화가 가능한 반면, 복잡한 텍스처에서 normal 재구성이 너무 잘 되면 이상이 희석되는 문제가 있어, Patch SVDD류 패치‑분포 기반 방법이 MVTec에서는 전반적으로 더 좋은 성능을 보이는 경향이 있습니다.[^1_7][^1_9][^1_14][^1_8][^1_1]
5. **고해상도·도메인 일반화·continual learning 방향**
    - 최근 HiAD 같은 고해상도 이상 탐지 프레임워크는 멀티해상도 feature fusion과 detector pool 개념을 도입해, 고해상도에서도 효율적으로 anomaly를 찾는 방향으로 확장되고 있습니다.[^1_16]
    - Domain‑independent PatchCore 변형, continual anomaly detection 벤치마크 등은 Patch SVDD/PaDiM/PatchCore류 방법을 바탕으로 도메인 일반화·continual learning을 탐구합니다.[^1_17][^1_18]

Patch SVDD는 이러한 후속 연구들이 **“패치 기반 representation + 간단한 이상 스코어링”**을 표준 패턴으로 삼는 데 기여한 초기 작업 중 하나로 볼 수 있습니다.[^1_9][^1_6][^1_7]

***

## 8. 앞으로의 연구 영향과 고려할 점

### 8.1 영향

- **패치 단위 one‑class 학습의 정당화**:
Patch SVDD는 복잡한 산업 이미지에서도 패치 수준 feature + k‑NN만으로도 강력한 이상 탐지/분할이 가능하다는 것을 실증하며, 이후 PaDiM·PatchCore 등 패치 기반 AD 방법들의 설계에 이론적·실험적 근거를 제공했습니다.[^1_6][^1_7][^1_8][^1_9][^1_1]
- **self‑supervised + one‑class 통합 프레임워크**:
self‑supervised pretext task를 one‑class 목적 함수와 함께 encoder 학습에 직접 통합한 구조는 CutPaste, DRAEM, InTra 등 다양한 self‑supervised AD 방법의 설계 철학과도 맞닿아 있습니다.[^1_14][^1_10][^1_12][^1_1]
- **표현 분석 관점 도입**:
t‑SNE, intrinsic dimension, 랜덤 인코더/ raw patch 분석을 통해 “왜” 잘 되는지를 계량적으로 보여준 점은, AD 모델을 단순 벤치마크 점수 이상으로 해석하려는 후속 연구들에 영감을 주었습니다.[^1_18][^1_1]


### 8.2 향후 연구 시 고려할 점

연구자로서 Patch SVDD를 확장/활용할 때 고려할 수 있는 방향은 다음과 같습니다.

1. **사전학습 모델과의 결합**
    - ResNet, ViT 등 대형 backbone에서 추출한 patch feature에 Patch SVDD의 $L_{\text{PatchSVDD}}$와 k‑NN 스코어링을 적용하면, PatchCore·PaDiM의 일반화 장점과 Patch SVDD의 self‑supervised 특성을 결합한 형태를 탐구할 수 있습니다.[^1_11][^1_7][^1_8][^1_9][^1_6]
2. **메모리 효율 개선**
    - PatchCore의 coreset selection, PaDiM의 Gaussian 파라미터 저장 방식을 Patch SVDD에 접목해, 대규모·고해상도 데이터에서도 실용적인 메모리 사용량과 추론 속도를 달성할 수 있는 구조를 설계할 수 있습니다.[^1_7][^1_8][^1_9][^1_6]
3. **더 강력한 self‑supervised signal**
    - 상대 위치 예측 대신,
        - contrastive learning (SimCLR류)
        - masked image modeling (MAE류)
        - transformer‑based inpainting (InTra)
등을 패치 단위로 결합해, domain shift와 noise에 더 robust한 표현을 학습하는 방향이 유망합니다.[^1_10][^1_11]
4. **도메인 일반화·continual learning**
    - 서로 다른 공장/라인/제품으로의 도메인 변화를 고려한 domain‑independent AD, continual anomaly detection 벤치마크에 Patch SVDD 구조를 적용·비교해, 실제 산업 환경에서의 지속 학습 성능을 검증할 필요가 있습니다.[^1_17][^1_16][^1_18]
5. **고해상도 및 3D/멀티모달 확장**
    - 고해상도 이미지(HDR, 대형 PCB, 웨이퍼)나 3D 포인트 클라우드, 멀티모달(영상+센서) 환경으로 Patch SVDD의 패치 기반 아이디어를 확장하는 것도 중요한 연구 방향입니다.[^1_19][^1_16]

***

## 9. 참고한 주요 오픈 액세스 자료 목록 (제목/링크)

- Jihun Yi, Sungroh Yoon, “Patch SVDD: Patch-level SVDD for Anomaly Detection and Segmentation,” ACCV 2020 / arXiv:2006.16067.[^1_2][^1_3][^1_1]
- Thomas Defard et al., “PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization,” ICPR 2020 / arXiv:2011.08785.[^1_20][^1_6][^1_7]
- Chun-Liang Li et al., “CutPaste: Self-Supervised Learning for Anomaly Detection and Localization,” CVPR 2021 / arXiv:2104.04015.[^1_21][^1_13][^1_12]
- Karsten Roth et al., “Towards Total Recall in Industrial Anomaly Detection (PatchCore),” CVPR 2022 / arXiv:2106.08265.[^1_8][^1_9]
- Vitjan Zavrtanik et al., “DRÆM – A discriminatively trained reconstruction embedding for surface anomaly detection,” ICCV 2021 / arXiv:2108.07610.[^1_15][^1_14]
- Jonathan Pirnay, Keng Chai, “Inpainting Transformer for Anomaly Detection (InTra),” arXiv:2104.13897.[^1_10]
- Pankaj Mishra et al., “VT-ADL: A Vision Transformer Network for Image Anomaly Detection and Localization,” arXiv:2104.10036.[^1_11]
- Ximiao Zhang et al., “Towards High-Resolution Industrial Image Anomaly Detection (HiAD),” arXiv:2508.12931.[^1_16]
- Jonas Bühler et al., “Domain-independent detection of known anomalies,” arXiv:2407.02910.[^1_17]
- Nikola Bugarin et al., “A Benchmark for Pixel-Level Anomaly Detection in Continual Learning,” arXiv:2403.15463.[^1_18]
<span style="display:none">[^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51]</span>

<div align="center">⁂</div>

[^1_1]: 2006.16067v2.pdf

[^1_2]: https://arxiv.org/abs/2006.16067

[^1_3]: https://ar5iv.labs.arxiv.org/html/2006.16067

[^1_4]: https://www.emergentmind.com/papers/2006.16067

[^1_5]: https://openaccess.thecvf.com/content/ACCV2020/papers/Yi_Patch_SVDD_Patch-level_SVDD_for_Anomaly_Detection_and_Segmentation_ACCV_2020_paper.pdf

[^1_6]: https://ar5iv.labs.arxiv.org/html/2011.08785

[^1_7]: https://arxiv.org/abs/2011.08785

[^1_8]: https://arxiv.org/abs/2106.08265

[^1_9]: https://ar5iv.labs.arxiv.org/html/2106.08265

[^1_10]: https://ar5iv.labs.arxiv.org/html/2104.13897

[^1_11]: https://arxiv.org/pdf/2104.10036.pdf

[^1_12]: https://arxiv.org/abs/2104.04015

[^1_13]: https://openaccess.thecvf.com/content/CVPR2021/papers/Li_CutPaste_Self-Supervised_Learning_for_Anomaly_Detection_and_Localization_CVPR_2021_paper.pdf

[^1_14]: https://arxiv.org/abs/2108.07610v2

[^1_15]: https://arxiv.org/abs/2108.07610

[^1_16]: https://arxiv.org/html/2508.12931v1

[^1_17]: https://arxiv.org/html/2407.02910v1

[^1_18]: https://arxiv.org/html/2403.15463v1

[^1_19]: https://arxiv.org/html/2508.01311v1

[^1_20]: https://cumulu-s.tistory.com/44

[^1_21]: https://ar5iv.labs.arxiv.org/html/2104.04015

[^1_22]: https://arxiv.org/html/2509.25856v1

[^1_23]: https://arxiv.org/pdf/2508.16034.pdf

[^1_24]: https://arxiv.org/html/2408.15113v2

[^1_25]: https://arxiv.org/html/2508.16034v1

[^1_26]: https://www.semanticscholar.org/paper/Patch-SVDD:-Patch-level-SVDD-for-Anomaly-Detection-Yi-Yoon/62b77e5cb85fc61b84edd532f6d65714be152596

[^1_27]: https://arxiv.org/html/2503.02691

[^1_28]: https://arxiv.org/html/2506.09368v1

[^1_29]: https://arxiv.org/html/2503.23451v1

[^1_30]: https://arxiv.org/html/2509.17670v1

[^1_31]: https://arxiv.org/pdf/2204.11014.pdf

[^1_32]: https://dev.to/dataroots/anomaly-detection-in-images-using-patchcore-2l0j

[^1_33]: https://dataroots.io/blog/anomaly-detection-in-images-using-patchcore

[^1_34]: https://velog.io/@maseully_hoit/PaDiM-a-Patch-Distribution-Modeling-Framework-for-Anomaly-Detection-and-Localization

[^1_35]: https://ojs.bonviewpress.com/index.php/AIA/article/download/6321/1811

[^1_36]: https://github.com/tiskw/patchcore-ad

[^1_37]: https://github.com/amazon-science/patchcore-inspection

[^1_38]: https://ffighting.net/deep-learning-paper-review/anomaly-detection/padim/

[^1_39]: https://arxiv.org/abs/2108.07610v1

[^1_40]: https://ar5iv.labs.arxiv.org/html/2108.07610

[^1_41]: http://arxiv.org/pdf/2108.07610.pdf

[^1_42]: http://arxiv.org/list/cs/2021-04?skip=550\&show=2000

[^1_43]: https://openaccess.thecvf.com/content/ICCV2021/papers/Zavrtanik_DRAEM_-_A_Discriminatively_Trained_Reconstruction_Embedding_for_Surface_Anomaly_ICCV_2021_paper.pdf

[^1_44]: https://www.emergentmind.com/papers/2108.07610

[^1_45]: https://github.com/Abde951/anomaly-detection-patchcore

[^1_46]: https://www.emergentmind.com/papers/2104.04015

[^1_47]: https://huggingface.co/papers/2108.07610

[^1_48]: https://github.com/ComputermindCorp/patchcore

[^1_49]: https://blog.outta.ai/297

[^1_50]: https://shacoding.com/2024/05/18/논문-리뷰-towards-total-recall-in-industrial-anomaly-detection/

[^1_51]: https://www.youtube.com/watch?v=mD2HL_1MiO4

