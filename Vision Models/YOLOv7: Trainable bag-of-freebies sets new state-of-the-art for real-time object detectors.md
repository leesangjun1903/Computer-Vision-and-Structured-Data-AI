# YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors

YOLOv7 논문은 **추가 추론 비용 없이(= bag-of-freebies)** 학습 과정만 바꾸는 기법들을 체계적으로 설계해, COCO 기준 실시간 객체 검출에서 당시 속도–정확도 SOTA를 달성하는 것이 핵심 주장입니다. 동시에, **재파라미터화·모델 스케일링·동적 레이블 할당 문제를 분석하고 해결 전략을 제안**했다는 점이 주요 기여입니다.[^1_1][^1_2]

***

## 1. 핵심 주장과 주요 기여

- **Trainable bag-of-freebies 프레임워크 제안**
추론 시 연산량과 파라미터 수를 늘리지 않으면서, 학습 시에만 구조를 확장·보조 헤드를 추가·EMA 등 기법을 사용하는 “trainable bag-of-freebies”를 체계적으로 정리해 적용합니다.[^1_2][^1_1]
- **E-ELAN 아키텍처와 concatenation 기반 compound scaling**
기존 ELAN/VoVNet류 concatenation 기반 백본을 확장한 E-ELAN과, 이 구조에 특화된 depth–width 결합 스케일링 방법을 제안해, 파라미터·FLOPs 대비 성능을 극대화합니다.[^1_1]
- **Planned re-parameterized convolution**
RepVGG류 re-parameterization을 잔차/concat 구조에 그대로 넣으면 성능이 떨어지는 문제를 분석하고, 어디에 어떤 형태의 RepConv를 배치해야 하는지 “planned” 전략을 제안합니다.[^1_1]
- **Coarse-to-fine lead guided label assignment**
deep supervision에서 lead head와 auxiliary head에 soft label을 어떻게 나눠 줄지 새 문제를 정의하고, lead head의 예측을 이용한 coarse/fine 이중 레이블 할당 전략으로 mAP를 소폭 개선합니다.[^1_1]
- **실험적 성과**
COCO를 “from scratch”로만 학습하면서도, V100 기준 30 FPS 이상 영역에서 56.8% AP를 달성하고, YOLOv4/YOLOv5/YOLOX/YOLOR/PP-YOLOE 및 여러 DETR/ConvNeXt/Swin 기반 검출기보다 더 나은 속도–정확도 균형을 보입니다.[^1_3][^1_4][^1_2][^1_1]

***

## 2. 해결하고자 하는 문제

논문이 명시적으로 다루는 핵심 문제는 다음 네 가지로 정리할 수 있습니다.[^1_2][^1_1]

1. **“추론 비용을 늘리지 않고” 정확도를 더 올릴 수 있는가?**
기존 실시간 검출기들은 구조를 키우거나 anchor-free·새 백본을 도입해 정확도를 올렸지만, 추론 비용이 늘어나는 경우가 많았습니다. YOLOv7은 학습 시에만 비용이 드는 bag-of-freebies 설계를 통해 이 부분을 개선하려 합니다.[^1_3][^1_1]
2. **재파라미터화 모듈을 다양한 아키텍처에 어떻게 안전하게 쓸 것인가?**
RepVGG(RepConv)를 ResNet, DenseNet, CSPDarknet 등에 그대로 적용하면, identity branch가 기존 residual/concat 경로와 충돌해 성능이 떨어질 수 있음을 실험으로 확인합니다. 이 문제를 “planned re-parameterized model”로 해결합니다.[^1_1]
3. **동적 레이블 할당 + deep supervision에서 레이블을 어떻게 나눌 것인가?**
최근 One-stage detector는 예측 품질을 이용한 동적 label assignment(OTA, PAA, TAL 등)를 쓰는데, 여러 출력 헤드(aux/lead)가 있을 때 “각 헤드에 soft label을 어떻게 할당할지”가 새로운 문제로 등장합니다. YOLOv7은 이를 coarse/fine 레이블과 lead-guided 전략으로 정의합니다.[^1_3][^1_1]
4. **concatenation 기반 네트워크의 모델 스케일링**
DenseNet/VoVNet/ELAN처럼 채널을 concat하는 구조에서 depth만 늘리면, 그 다음 transition layer 입력 채널 수까지 같이 늘어나는 문제가 있어, 기존 EfficientNet식 독립적 scaling 규칙을 그대로 쓰기 어렵습니다. YOLOv7은 이를 고려한 compound scaling을 제안합니다.[^1_1]

***

## 3. 제안 방법 (개념 및 수식)

### 3.1 E-ELAN 아키텍처

ELAN은 “최단·최장 gradient 경로 길이를 제어하면 깊은 네트워크도 안정적으로 학습된다”는 관점을 가지는 concatenation 기반 블록입니다. YOLOv7의 **Extended-ELAN(E-ELAN)**은 다음 아이디어를 추가합니다.[^1_1]

- 각 computational block 내부에서 **group convolution로 채널 수와 cardinality를 확장**
- 블록별 feature map을 $g$개의 그룹으로 나눈 뒤 shuffle \& concat, 최종적으로 group-wise sum(merge cardinality) 수행
- transition layer(블록 사이 연결부)는 그대로 유지해 원래 gradient path 구조를 보존

간단히 표현하면, 입력 $x$에 대해 $g$개의 group conv 경로 $f_k$를 두고,

$$
z_k = f_k(x), \quad k = 1,\dots,g
$$

$$
z_{\text{shuffle}} = \text{Shuffle}([z_1,\dots,z_g]), \quad y = \sum_{k=1}^g z_{\text{shuffle},k}
$$

처럼 **여러 group의 표현을 섞고 더하는 방식으로 표현력을 늘리되, shortcut/concat 경로의 길이는 유지**합니다. 이는 파라미터 증가 대비 mAP 향상에 크게 기여하며, 이후 YOLOv7-E6E 등 대형 모델에 사용됩니다.[^1_1]

### 3.2 Concatenation 기반 compound scaling

concatenation 기반 구조에서 depth를 곱하기 $\alpha$배로 키우면, 그 블록의 출력 채널 수도 함께 늘어나, 다음 transition layer의 입력 폭이 증가합니다. 이 때문에 width와 depth를 독립적으로 scaling하기 어렵고, YOLOv7은 아래와 같은 “연동 스케일링”을 사용합니다.[^1_1]

- 어떤 computational block의 깊이가 $d$, 출력 채널이 $c$일 때,

$$
d' = \alpha d, \quad c' = c + \Delta c
$$
- 이어지는 transition layer의 convolution은 입력 채널을 $c + \Delta c$로 보고, 너비 스케일링 factor $\beta$를 적용해

$$
c'' = \beta (c + \Delta c)
$$

로 맞추는 식입니다.[^1_1]

실험에서는 대략 $\alpha \approx 1.5$, $\beta \approx 1.25$를 사용하는 **compound scaling**이, depth only / width only보다 동일 FLOPs에서 더 높은 AP를 보였습니다.[file:1, Table 3]

### 3.3 Planned re-parameterized convolution

기존 RepConv는 학습 시에는 $3\times3$, $1\times1$, identity 세 branch를 두고, 추론 시 이들을 하나의 $3\times3$ conv로 합치는 방식입니다.[^1_1]
일반적으로 학습 시 출력은

$$
y = (W_{3\times3} * x) + (W_{1\times1} * x) + x + b
$$

처럼 세 경로를 더하지만, 추론 시에는 이들을 합성해

$$
y = W_{\text{rep}} * x + b_{\text{rep}}
$$

형태의 단일 convolution으로 바꿉니다.[^1_1]

YOLOv7은 **residual이나 concat이 이미 있는 레이어에서 identity를 또 추가하면 gradient 다양성이 줄어들어 오히려 성능이 떨어질 수 있음**을 지적합니다. 그래서,[^1_1]

- **잔차/concat 경로가 있는 레이어에는 identity 없는 RepConvN만 사용**
- identity 포함 RepConv는 Plain conv층 등 제한된 위치에만 배치

하는 “planned re-parameterized model”을 제안합니다. ablation에서 concatenation/residual 기반 모델 모두에서 이 전략이 성능을 향상시킵니다.[^1_1]

### 3.4 Lead-guided \& coarse-to-fine label assignment

YOLOv7은 **deep supervision**을 위해 중간 피라미드에 auxiliary head를 두고, 최종 출력을 담당하는 head를 lead head라 부릅니다. 문제는, 동적 label assignment 하에서[^1_1]

- 기존 방식: lead/aux 각각 자기 prediction으로 soft label을 계산(독립 label assigner)
- 제안 방식: lead head 예측을 기준으로 두 head의 label을 **동시에** 정함

입니다.[^1_1]

개념적으로, grid $i$에 대해

- lead head 예측과 GT를 이용해 soft label $y^{\text{fine}}_i$ (정밀) 생성
- positive 조건을 완화해 더 많은 grid를 양성으로 보는 $y^{\text{coarse}}_i$ (조 coarse) 생성
- lead head는 $y^{\text{fine}}_i$, auxiliary head는 $y^{\text{coarse}}_i$를 학습

하게 됩니다. 각 head의 objectness 손실은 예를 들어[^1_1]

$$
L_{\text{obj}} = - \sum_i \left[ y_i \log p_i + (1-y_i)\log(1-p_i) \right]
$$

꼴의 binary cross-entropy로 볼 수 있고, 여기서 $y_i$가 soft label(coarse 또는 fine)입니다. YOLOv7은 coarse label의 영향이 과도해지지 않도록, 객체 중심에서 멀수록 objectness upper bound를 낮추는 제약을 decoder에 두어 fine label의 최적 상한이 항상 더 크도록 설계합니다.[^1_1]

실험적으로

- auxiliary head를 도입하면 AP가 약 $+0.2\sim0.3$ p 향상
- lead-guided, coarse-to-fine 전략이 독립 label assignment보다 AP/ $AP_{50}$ / $AP_{75}$ 모두에서 더 좋음

을 보입니다.[file:1, Table 6–8]

### 3.5 기타 trainable bag-of-freebies

논문이 “우리 아이디어는 아니지만 YOLOv7에 통합한” trainable bag-of-freebies로 명시하는 것들은 다음과 같습니다.[^1_1]

- **Conv–BN–Activation 구조에서 BN-folding**
학습 시에는 batch normalization을 사용하지만, 추론 시 BN의 평균/분산과 scale/shift를 conv weight와 bias에 흡수해 연산 수를 줄입니다.[^1_1]
- **YOLOR의 implicit knowledge 재활용**
YOLOR에서 사용한 implicit vector를 미리 연산해 conv의 bias/weight에 흡수하는 방식으로, 추론 시 추가 연산 없이 정보량을 늘립니다.[^1_1]
- **EMA(Exponential Moving Average) 모델 사용**
mean teacher 방식처럼, 학습 중 파라미터 $\theta_t$의 EMA $\theta^{\text{EMA}}\_t = \lambda \theta^{\text{EMA}}_{t-1} + (1-\lambda)\theta_t$를 유지하고, 최종 추론에는 EMA 모델만 사용해 일반화를 개선합니다.[^1_1]

이들 모두 **추론 그래프는 동일하게 유지하면서 학습 시에만 추가 비용을 쓰는 bag-of-freebies**에 해당합니다.[^1_1]

***

## 4. 모델 구조와 변형

YOLOv7 계열의 큰 틀은 “CSPNet/ELAN 계열 백본 + FPN/PAN류 neck + YOLO head”라는 YOLOv4/Scaled-YOLOv4/YOLOX 계열의 한 축을 계승합니다.[^1_3][^1_1]

- **YOLOv7-tiny**: edge GPU용 소형 모델, leaky ReLU 사용, 약 6.2M 파라미터, COCO val 기준 AP 35.2% (416 해상도).[^1_1]
- **YOLOv7 (base)**: 일반 GPU용, 36.9M 파라미터, 104.7G FLOPs, 640 입력에서 val AP 51.2%.[^1_1]
- **YOLOv7-X**: base의 stack scaling 및 compound scaling 버전으로 71.3M 파라미터, AP 52.9%.[^1_1]
- **YOLOv7-W6/E6/D6/E6E**: 1280 해상도용 대형 모델 계열로, E-ELAN과 auxiliary head를 적극 활용해 최대 56.8% AP까지 도달합니다.[^1_1]

모든 모델은 COCO train2017만으로 “from scratch” 학습되며, 다른 데이터나 외부 pretrain은 사용하지 않는다고 명시합니다.[^1_2][^1_1]

***

## 5. 성능 향상과 한계

### 5.1 정량적 성능 향상

논문이 보고하는 주요 비교를 요약하면 다음과 같습니다.[^1_2][^1_1]

- **Baseline(YOLOv4, YOLOR) 대비**
    - YOLOv7 vs YOLOv4: 파라미터 75%↓, FLOPs 36%↓, AP +1.5p (49.7→51.2).[^1_1]
    - YOLOv7 vs YOLOR-CSP: 파라미터 43%↓, FLOPs 15%↓, AP +0.4p (50.8→51.2).[^1_1]
- **Tiny 계열**
    - YOLOv7-tiny vs YOLOv4-tiny-3l (320): 파라미터 39%↓, FLOPs 49%↓, AP 동일(30.8), $AP_L$은 +0.7p 개선.[^1_1]
- **대형 모델**
    - YOLOv7-E6 vs YOLOR-E6: 파라미터 19%↓, FLOPs 33%↓, AP +0.2p (55.7→55.9).[^1_1]
    - YOLOv7-E6E는 real-time(≥30 FPS) 조건에서 56.8% AP로 당시 최고 정확도 실시간 검출기입니다.[^1_2][^1_1]
- **다른 실시간 검출기와의 비교 (YOLOv5, YOLOX, PP-YOLOE 등)**
    - YOLOv7(51.4% AP, 161 FPS) vs PP-YOLOE-L(51.4% AP, 78 FPS): 같은 AP에서 약 2배 이상 빠르고, 파라미터는 41% 적음.[^1_4][^1_3][^1_1]
    - YOLOv7-X(52.9% AP, 114 FPS) vs YOLOv5-L(49.0% AP, 99 FPS): 비슷한 FLOPs에서 AP +3.9p, FPS +15.[^1_1]

또한 V100/A100 환경에서 SWIN-L Cascade Mask R-CNN, ConvNeXt-XL Cascade Mask R-CNN, Deformable DETR, DINO 등과 비교해, **동일 AP대비 훨씬 높은 FPS**를 보여 “real-time” 영역에서 우위를 주장합니다.[^1_2][^1_1]

### 5.2 한계 및 비평

- **데이터셋 다양성 부족**
실험은 거의 전적으로 MS COCO만 사용하며, 타 데이터셋(VOC, Cityscapes 등)이나 도메인 전이 실험(날씨·센서 변화 등)은 제시하지 않습니다. 따라서 “도메인 일반화” 수준을 정량적으로 판단하기 어렵습니다.[^1_1]
- **복잡한 학습 스케줄과 구현 난이도**
planned re-parameterization, coarse-to-fine label assignment, EMA, implicit knowledge 등 여러 기법이 얽혀 있어, 구현·재현 난이도가 높습니다. 논문은 상위 개념과 ablation 결과는 제공하지만, 일부 세부 수식/하이퍼파라미터는 appendix나 코드에 의존합니다.[^1_1]
- **Transformer 기반 최신 검출기와의 장기적 비교**
논문 시점에서는 Swin/ConvNeXt 계열 2-stage 검출기와 비교해 real-time 영역에서 우수함을 보였으나, 이후 DETRs Beat YOLOs on Real-time Object Detection 같은 연구는 최적화된 DETR 변형이 real-time 설정에서 YOLO 계열을 능가할 수 있음을 보입니다.[^1_5][^1_1]
- **작은 객체, 극단적 조건에 대한 세부 분석 부족**
$AP_S$ / $AP_M$/ $AP_L$는 보고하지만, 작은 객체·occlusion·rare category 등에 대한 세부 error analysis는 한정적입니다.[^1_1]

***

## 6. 일반화 성능 향상 가능성

논문이 직접 “일반화”를 주제로 한 이론 분석이나 cross-dataset 실험을 제공하지는 않지만, 설계와 후속 연구를 바탕으로 다음과 같은 **일반화 잠재력**을 논의할 수 있습니다.[^1_1]

1. **EMA와 weight averaging의 효과**
EMA 모델을 최종 inference 모델로 사용하는 것은 넓은 basin(평탄한 최소점)을 선택해 test 성능을 개선한다는 weight averaging 계열 연구들과 맥을 같이 합니다. YOLOv7이 EMA를 기본 설정으로 사용해, 복잡한 훈련 스케줄에도 overfitting을 완화하고 안정적인 성능을 달성한 것은 일반화에 긍정적 신호입니다.[^1_6][^1_1]
2. **Bag-of-freebies = 학습 시 ensemble, 추론 시 단일 모델**
    - planned re-parameterization은 학습 시 여러 branch를 사용하는 ensemble-like 구조를 허용하면서, 추론 시 단일 branch로 합칩니다.[^1_1]
    - coarse-to-fine auxiliary head는 다양한 공간적 후보를 학습 시 포괄적으로 탐색(coarse), 최종 head는 정밀 후보(fine)에 집중하게 해 overfitting을 줄이는 역할을 할 수 있습니다.[^1_1]
이런 구조적 ensemble 효과는, 파라미터 수 대비 일반화 성능을 끌어올리는 데 기여합니다.
3. **계통적인 모델 스케일링과 size 간 일관성**
YOLOv7-tiny부터 E6E까지, 같은 설계 철학(E-ELAN, compound scaling, bag-of-freebies)을 공유하면서도 다양한 스케일에서 안정적인 성능 향상 곡선을 보입니다. 이는 **모델 크기를 바꿔도 최적화와 일반화 특성이 크게 망가지지 않도록 구조를 설계했다**는 의미로 볼 수 있습니다.[^1_1]
4. **도메인 특화 개선 연구에서의 활용성**
    - 수중 목표 검출을 위한 improved YOLOv7에서는, backbone/neck/head를 약간 수정하고 데이터 증강을 조정해 원본 YOLOv7보다 mAP와 recall을 개선하면서도 파라미터와 FLOPs를 줄이는 결과를 보고합니다.[^1_7]
    - 송전선 절연자 결함 검출을 위한 improved YOLOv7에서도, 경량화 모듈과 디코더 개선으로 $mAP_{0.5:0.95}$와 속도를 모두 개선합니다.[^1_8]
이런 후속 작업들은, YOLOv7 구조가 **새 도메인·잡음 환경에 맞게 변형하기 좋은 기반**임을 시사하며, 실험적으로도 꽤 좋은 generalization capacity를 보여줍니다.

다만, **정식 도메인 일반화 세팅(훈련 도메인 A, 테스트 도메인 B)**에서의 체계적 비교는 아직 부족하며, 향후 연구 과제로 남습니다.

***

## 7. 2020년 이후 관련 최신 연구 비교

아래는 YOLOv7과 밀접한 최근 실시간 객체 검출 연구들을, 개념과 관계 중심으로 정리한 표입니다.


| 모델 | 연도 | 핵심 아이디어 | YOLOv7과의 관계 |
| :-- | :-- | :-- | :-- |
| YOLOv4 [Bochkovskiy et al.] | 2020 | CSPDarknet 기반 백본, 다양한 bag-of-freebies/ specials를 조합해 단일 GPU에서 고성능 달성.[^1_1][^1_6] | YOLOv7은 YOLOv4/Scaled-YOLOv4의 계보 위에서 ELAN, CSPNet, bag-of-freebies 개념을 더 구조화해 성능과 효율을 동시에 개선.[^1_1] |
| PP-YOLOE [Xu et al.] | 2022 | PP-YOLOv2 개선: anchor-free paradigm, CSPRepResStage backbone, ET-head, TAL(dynamic label assignment) 등으로 COCO test-dev 51.4 mAP@78 FPS(V100) 달성.[^1_3][^1_9][^1_4] | 동시대 industrial SOTA 중 하나로, YOLOv7과 유사하게 label assignment·head 설계를 중시하지만, YOLOv7은 concat 기반 backbone과 trainable bag-of-freebies에 더 초점을 둡니다.[^1_1][^1_3] |
| YOLOX [Ge et al.] | 2021 | Anchor-free, decoupled head, SimOTA label assignment를 도입해 YOLOv3/5 대비 큰 향상.[^1_3] | YOLOv7이 비교 대상으로 삼는 anchor-free 실시간 검출기. label assignment 문제를 심화시키는 출발점 중 하나로, YOLOv7의 coarse-to-fine 전략과 개념적으로 연결됩니다.[^1_1] |
| YOLOv6 [Meituan] | 2022 | EfficientRep backbone, Rep-PAN neck, hardware-aware 설계와 학습 스킴으로 COCO에서 YOLOv5/YOLOX/PP-YOLOE를 상회하는 speed–accuracy.[^1_10][^1_11][^1_12] | YOLOv7과 마찬가지로 re-parameterization과 deployment-friendly 설계를 강조. YOLOv7이 학술 측면에서 re-param 위치와 label assignment 이슈를 분석했다면, YOLOv6는 산업 배포 지향 최적화를 더 강조.[^1_1][^1_12] |
| DETRs Beat YOLOs on Real-time Object Detection | 2023 | 최적화된 DETR 변형과 구현으로, 동일 하드웨어/지연 제한에서 최적 튜닝된 YOLO계열보다 더 나은 성능을 달성할 수 있음을 보임.[^1_5] | YOLOv7이 제시한 real-time SOTA 지위를 transformer 기반 DETR 계열이 따라잡고 일부 설정에서는 능가함을 보여, 향후 연구 방향(Transformer+YOLO 하이브리드 등)에 시사점을 줍니다. |
| Improved YOLOv7 for underwater detection | 2023–2024 | YOLOv7 backbone과 head를 개선해 수중 이미지의 낮은 대비·노이즈 환경에서 mAP와 속도 모두 향상.[^1_7] | YOLOv7의 구조가 도메인 특화 개선의 기반이 되며, 다양한 환경에서의 일반화 잠재력을 보여줌. |
| Improved YOLOv7 for insulator defect detection | 2025 | YOLOv7 기반으로 attention 및 경량화 모듈을 추가해 전력 설비 결함 검출에서 정확도 및 속도를 모두 개선.[^1_8] | 실시간 산업 응용에서 YOLOv7의 확장성을 보여주며, bag-of-freebies + 경량화 설계의 조합 가능성을 확인. |


***

## 8. 앞으로의 연구에 미치는 영향과 연구 시 고려점

### 8.1 영향

1. **“학습 전용” 기법의 체계화**
YOLOv7은 성능 향상 기법을 “추론 비용 증가 없음”이라는 제약 하에 설계·분류해, 이후 연구들이 **training-time-only tricks(EMA, auxiliary heads, dynamic label assignment 등)**을 조합하는 방향에 중요한 레퍼런스를 제공합니다.[^1_12][^1_3][^1_1]
2. **re-parameterization 설계 지침 제공**
단순히 RepConv를 넣는 것이 아니라, residual/concat 경로와의 상호작용까지 고려해야 한다는 분석은, 이후 RepConv/RepVGG 계열 백본 설계와 YOLOv6·PP-YOLOE의 CSPRepResStage 등에도 개념적으로 영향을 줍니다.[^1_4][^1_12][^1_1]
3. **label assignment 연구의 심화**
lead-guided, coarse-to-fine label assignment는 TAL, SimOTA, ATSS 등 동적 label assignment에 “multi-head/auxiliary” 차원을 추가하는 시발점 역할을 하며, 향후 rotated detection(PP-YOLOE-R) 등에서도 다양한 label assignment 변형이 등장하는 계기가 됩니다.[^1_13][^1_3][^1_1]
4. **concatenation 기반 모델 scaling의 참고 사례**
DenseNet/VoVNet류 아키텍처에서 depth–width 결합 scaling을 어떻게 안전하게 할지에 대한 실제 설계·실험 결과를 제공해, 후속 경량 네트워크 및 NAS 연구에서 설계 제약 조건으로 활용될 수 있습니다.[^1_1]

### 8.2 앞으로 연구 시 고려할 점

1. **명시적 도메인 일반화·강건성 평가 필요**
    - COCO-only from-scratch 실험만으로는 도메인 이동(날씨, 센서, 스타일 변화)에 대한 일반화 능력을 평가하기 어렵습니다.[^1_1]
    - 향후 YOLOv7/후속 모델 연구에서는 **cross-dataset 벤치마크**(예: COCO→Cityscapes, BDD100K, OpenImages 등)와 **corruption/robustness 벤치마크**를 포함하는 것이 필요합니다.
2. **Label assignment와 loss 설계의 공통 프레임워크화**
    - TAL, SimOTA, YOLOv7의 coarse-to-fine, Varifocal Loss, GFL 등 label·loss 설계가 점점 복잡해지고 있습니다.[^1_3][^1_1]
    - 향후에는 이들을 하나의 수식적 프레임워크로 정리해, 학습 안정성·일반화와의 관계를 이론적으로 분석하는 연구가 요구됩니다.
3. **Transformer·CNN 하이브리드의 real-time 최적점 탐색**
    - DETR 변형들이 real-time 영역에서 YOLO를 넘어서고 있다는 결과가 보고되면서, CNN 기반 YOLOv7과 transformer 기반 DETR 사이의 **혼합 구조**(예: CNN backbone + lightweight decoder, 또는 토큰 기반 head)가 중요한 연구축이 될 것입니다.[^1_5]
    - 이때도 YOLOv7이 제안한 bag-of-freebies 개념(훈련 시만 복잡, 추론 시 단순)을 적용하면, 실시간 제약을 만족하면서 일반화를 개선할 수 있습니다.
4. **경량화·배포 관점에서의 재설계**
    - YOLOv7 큰 모델(E6E 등)은 여전히 edge CPU/모바일 NPU에는 무거우므로, pruning, quantization-aware training, low-rank factorization 등과 bag-of-freebies를 결합하는 연구가 필요합니다.[^1_1]
    - RepConv, BN-folding, implicit knowledge 흡수 등은 이미 deployment-friendly한 방향이므로, 이를 극단적으로 밀어붙인 ultra-light 버전 설계도 유망합니다.[^1_12][^1_1]
5. **작은 객체·희귀 클래스에 특화된 bag-of-freebies 탐색**
    - 현재 bag-of-freebies는 주로 전체 mAP 개선을 겨냥하고 있으며, 작은 객체( $AP_S$ ), rare class에 대한 특화 기법은 상대적으로 덜 다루어졌습니다.[^1_1]
    - curriculum-style label assignment, scale-aware auxiliary head, rare class re-weighting 등과 YOLOv7 구조를 결합한 연구가 향후 가치 있을 것입니다.

***

## 9. 관련 공개 접근 논문 및 자료 (제목·링크)

아래 자료들은 모두 오픈 액세스이며, 위 설명에서 인용한 주요 출처입니다.

- **YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors**
Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao, arXiv:2207.02696.[^1_14][^1_2][^1_1]
https://arxiv.org/abs/2207.02696
- **PP-YOLOE: An evolved version of YOLO**
Shangliang Xu et al., arXiv:2203.16250.[^1_9][^1_15][^1_4][^1_3]
https://arxiv.org/abs/2203.16250
- **YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications**
Meituan Vision, arXiv:2209.02976 \& OpenReview preprint.[^1_10][^1_11][^1_12]
https://arxiv.org/abs/2209.02976
- **DETRs Beat YOLOs on Real-time Object Detection**
arXiv:2304.08069.[^1_5]
https://arxiv.org/abs/2304.08069
- **Improved YOLOv7 model for insulator defect detection**
arXiv:2502.07179.[^1_8]
- **Underwater target detection based on improved YOLOv7**
arXiv:2302.06939.[^1_7]
- **A Decade of You Only Look Once (YOLO) for Object Detection**
Survey on YOLO 계열 발전사, arXiv:2504.18586.[^1_6]
- **Ultralytics YOLOv7 문서 (요약·벤치마크 표)**
https://docs.ultralytics.com/ko/models/yolov7/[^1_16]

이 자료들을 함께 읽으면, YOLOv7의 위치와 이후 YOLO 계열·DETR 계열의 진전을 더 입체적으로 파악할 수 있습니다.
<span style="display:none">[^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24]</span>

<div align="center">⁂</div>

[^1_1]: 2207.02696v1.pdf

[^1_2]: https://ar5iv.labs.arxiv.org/html/2207.02696

[^1_3]: https://ar5iv.labs.arxiv.org/html/2203.16250

[^1_4]: https://arxiv.org/pdf/2203.16250.pdf

[^1_5]: https://arxiv.org/html/2304.08069v3

[^1_6]: https://arxiv.org/html/2504.18586v1

[^1_7]: https://arxiv.org/pdf/2302.06939.pdf

[^1_8]: https://arxiv.org/pdf/2502.07179.pdf

[^1_9]: https://arxiv.org/abs/2203.16250

[^1_10]: https://arxiv.org/abs/2209.02976

[^1_11]: https://arxiv.org/pdf/2209.02976.pdf

[^1_12]: https://openreview.net/pdf?id=7c3ZOKGQ6s

[^1_13]: https://arxiv.org/abs/2211.02386

[^1_14]: https://arxiv.org/abs/2207.02696

[^1_15]: https://www.semanticscholar.org/paper/PP-YOLOE:-An-evolved-version-of-YOLO-Xu-Wang/5f85aec081f4ee224441899b6c1ceec1535bd58f

[^1_16]: https://docs.ultralytics.com/ko/models/yolov7/

[^1_17]: https://arxiv.org/pdf/2207.02696.pdf

[^1_18]: https://arxiv.org/html/2602.15926v1

[^1_19]: https://arxiv.org/html/2412.13006v1

[^1_20]: https://arxiv.org/html/2502.04161v1

[^1_21]: https://blog.csdn.net/djfjkj52/article/details/142337003

[^1_22]: https://eehoeskrap.tistory.com/651

[^1_23]: https://deeplearning-research.tistory.com/23

[^1_24]: https://www.sciencedirect.com/science/article/pii/S1110016825003850

