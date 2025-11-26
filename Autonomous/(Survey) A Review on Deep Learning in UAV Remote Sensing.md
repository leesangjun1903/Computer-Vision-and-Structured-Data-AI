# A Review on Deep Learning in UAV Remote Sensing

### 1. 핵심 주장과 주요 기여

**핵심 주장**[1]

이 논문의 핵심 주장은 **딥러닝(Deep Learning, DL)이 무인항공기(UAV) 기반 원격감시 영상 처리에서 획기적인 방법**이라는 것입니다. 저자들은 232개의 국제 학술논문을 분석하여, UAV 고해상도 영상이 제공하는 풍부한 공간 정보를 깊은 신경망이 효과적으로 활용할 수 있음을 입증합니다. 특히 **환경 모니터링(46.6%), 농업 정밀성(26.4%), 도시 계획(27.2%) 분야에서 CNN, RNN, GAN 등의 딥러닝 아키텍처가 자동화되고 정확한 영상 분석을 제공**한다고 주장합니다.[1]

**주요 기여**[1]

1. **딥러닝 기초 개념과 UAV 적용**: 분류, 객체 검출, 의미론적 분할(semantic segmentation) 접근법의 기초적 이론과 UAV 영상 기반 매핑 작업에 대한 포괄적 설명
2. **응용 분야 분류**: 환경, 도시, 농업 분야로 체계적으로 분류한 232개 논문의 센서 유형 및 네트워크 구조 검토
3. **공개 데이터셋 정리**: UAV로 수집한 객체 검출 및 분할 작업용 라벨링된 공개 데이터셋의 체계적 정리
4. **미래 연구 방향**: 실시간 처리, 도메인 적응, 소수 학습(few-shot learning), 주의 메커니즘 등의 도전과제 및 전망 제시

***

### 2. 해결하고자 하는 문제와 제안하는 방법

**문제 인식**[1]

논문이 해결하고자 하는 핵심 문제는 **"고해상도 UAV 영상 분석에서 수동 검사의 비효율성과 부정확성"**입니다. 저자들은 다음을 지적합니다:

- 대량의 UAV 영상 데이터에 대한 자동화된 처리 방법의 부재
- 기존 기계학습 방법의 특성 추출 한계
- 다양한 기하학적 관점과 조명 조건에서의 객체 검출 어려움
- 도메인 간 일반화(generalization) 능력 부족

**제안하는 방법**[1]

#### 2.1 기본 신경망 구조

**DNN(Deep Neural Network)의 기본 원리**[1]

$$X_{input} \xrightarrow{hidden\ layers} Y_{output}$$

신경망은 입력층(input layer)에서 수집된 데이터를 받아 은닉층(hidden layers)을 거쳐 점진적으로 고수준의 특성을 학습합니다. 각 층에서:

$$z = \sum_{i=1}^{n} w_i x_i + b$$

$$a = f(z)$$

여기서 $$w_i$$는 가중치, $$b$$는 편향, $$f$$는 활성화 함수입니다.[1]

**활성화 함수**[1]

가장 널리 사용되는 ReLU(Rectified Linear Unit) 함수:

$$ReLU(x) = \max(0, x)$$

최근 탐색되는 Mish 함수:

$$Mish(x) = x \tanh(\text{softplus}(x)) = x \tanh(\ln(1 + e^x))$$

#### 2.2 주요 아키텍처

**합성곱 신경망(CNN)**[1]

CNN은 합성곱층, 풀링층, 완전 연결층으로 구성되며, 특히 UAV 영상 처리에서 가장 널리 사용됩니다(91.2% of surveyed papers).

주요 백본(backbone) 네트워크:
- **VGG-16**: 3×3 합성곱 필터 사용
- **ResNet-50**: 깊이 기울기 소실 문제 해결을 위한 잔차 연결(residual connections)
- **HRNet**: 고해상도 특성 맵 유지
- **RegNet, Res2Net, ResNesT**: 최신 아키텍처

**RNN과 LSTM**[1]

시계열 데이터 처리를 위한 구조:

$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$

LSTM은 메모리 셀을 통해 장기 의존성을 학습합니다:

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$
$$C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$$
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t * \tanh(C_t)$$

여기서 $$f_t, i_t, o_t$$는 각각 망각, 입력, 출력 게이트입니다.[1]

**CNN-LSTM 통합**[1]

시공간적 특성을 활용한 구조로, 다시점 시계열 데이터 분석에 유용합니다.

**GAN(Generative Adversarial Network)**[1]

생성기(generator)와 판별기(discriminator) 간의 대박관계:

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

최근 원격감시에서 이미지 변환 및 데이터 증강에 활용됩니다.

#### 2.3 객체 검출 방법

**2단계 검출기 (Two-Stage Detectors)**[1]

Faster R-CNN의 Region Proposal Network(RPN):

$$P(object) = sigmoid(p_1) = \frac{1}{1 + e^{-p_1}}$$

주요 발전: Cascade R-CNN, Trident-Net, Dynamic R-CNN, DetectoRS

**1단계 검출기 (One-Stage Detectors)**[1]

바운딩 박스 인코딩 전략:

$$\text{bbox loss} = \sum_{i = 1}^{N} L_{box}(\hat{y}_i, y_i)$$

여러 샘플링 전략(Libra R-CNN, ATSS, GFL, VFNet) 활용으로 클래스 불균형 문제 해결

**목적 함수 (Loss Function)**[1]

분류 손실:
$$L_{cls} = -\sum_{i} [p_i \log(\hat{p}_i) + (1-p_i) \log(1-\hat{p}_i)]$$

회귀 손실:
$$L_{reg} = \sum_{i} \text{smooth}_{L1}(t_i - \hat{t}_i)$$

전체 손실:
$$L = L_{cls} + \lambda L_{reg}$$

#### 2.4 의미론적 분할 (Semantic Segmentation)

**U-Net 아키텍처**[1]

인코더-디코더 구조로 픽셀 수준 분류 수행:

$$output = \text{softmax}(\text{decoder}(\text{encoder}(image)))$$

각 픽셀에 대한 확률 벡터 생성.

**Focal Loss** (클래스 불균형 해결)[1]

$$FL(p_t) = -\alpha_t (1-p_t)^{\gamma} \log(p_t)$$

여기서 $$\gamma$$는 focusing parameter, $$\alpha_t$$는 클래스 가중치

#### 2.5 최적화 알고리즘

일반적으로 사용되는 적응학습률 알고리즘:[1]

**Adam 옵티마이저**:
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\theta_t = \theta_{t-1} - \frac{\alpha \sqrt{1-\beta_2^t}}{1-\beta_1^t} \frac{m_t}{\sqrt{v_t} + \epsilon}$$

**AdaGrad, RMSProp, AdaDelta** 등 다양한 변형 사용

***

### 3. 모델 구조와 성능 향상

**네트워크 아키텍처 진화**[1]

논문에서 조사한 주요 발전:

| 아키텍처 | 주요 특징 | 응용 분야 |
|---------|---------|---------|
| AlexNet | 깊은 CNN의 시초 | 초기 UAV 영상 분류 |
| VGG | 소형 합성곱 필터 사용 | 회귀 작업 |
| ResNet | 잔차 학습 | 광범위한 기본 네트워크 |
| HRNet | 고해상도 특성 유지 | UAV 고해상도 영상 처리 |
| DeepLabV3+ | 주의 깊은 공간 피라미드 풀링 | 의미론적 분할 |
| Mask R-CNN | 인스턴스 분할 | 개별 객체 검출 |
| Vision Transformer (ViT) | 패치 기반 주의 | 다중 스케일 특성 추출 |
| DETR | 변환기 기반 검출 | 실시간 객체 검출 |

**특성 추출 메커니즘**[1]

**Feature Pyramid Network (FPN)**:

낮은 해상도의 의미론적으로 강한 특성과 높은 해상도의 의미론적으로 약한 특성을 결합:

$$P_l = W_l \cdot \text{upsample}(P_{l+1}) + C_l$$

여기서 $$C_l$$은 백본의 특성 맵, $$P_l$$은 피라미드 수준

**성능 평가 지표**[1]

- **정확도 (Accuracy)**: $$A = \frac{TP + TN}{TP + TN + FP + FN}$$

- **Precision**: $$P = \frac{TP}{TP + FP}$$

- **Recall**: $$R = \frac{TP}{TP + FN}$$

- **F1 Score**: $$F1 = 2 \cdot \frac{P \cdot R}{P + R}$$

- **IoU (Intersection over Union)**: $$IoU = \frac{\text{Area of Overlap}}{\text{Area of Union}}$$

- **mIoU (Mean IoU)**: 모든 클래스의 평균 IoU

**실제 응용 성과**[1]

- **객체 검출 (53.9%)**: 일반적인 적용 비율
- **이미지 분할 (40.7%)**: 점증하는 추세
- **장면 분류 (5.4%)**: 제한적 사용
- **RGB 센서 활용 (52.4%)**: 저비용 및 가용성 때문
- **환경 모니터링 (46.6%)**: 삼림, 야생동물, 홍수 감지 등

***

### 4. 모델 일반화 성능 향상

**일반화 문제와 한계**[1]

논문에서 강조하는 주요 한계:

1. **도메인 시프트 (Domain Shift)**: 다양한 지리적 영역, 센서, 시점에서의 성능 저하
2. **소수 표본 (Few-shot) 학습**: 라벨링 데이터 부족 문제
3. **클래스 불균형**: 배경과 전경 간의 샘플 불균형
4. **작은 객체 검출**: 다중 스케일 특성 추출의 어려움

**일반화 성능 향상 방법**[1]

#### 4.1 전이 학습 (Transfer Learning)

사전 훈련된 모델의 가중치를 활용하여 초기 특성 추출 능력 향상:

$$L_{target} = L_{task}(\theta_{pre-trained}, D_{target})$$

주요 사례: ImageNet 사전 훈련 가중치 사용으로 수렴 속도 및 정확도 개선

#### 4.2 도메인 적응 (Domain Adaptation)[2][3]

**비지도 도메인 적응 (Unsupervised Domain Adaptation)**:

$$\min_\theta \mathbb{E}_{x_s \sim S} [\ell(f_\theta(x_s), y_s)] + \lambda \cdot d(S, T)$$

여기서 $$d(S, T)$$는 원본 도메인 S와 목표 도메인 T 간의 거리

**GAN 기반 스타일 변환**: 원본 도메인 이미지를 목표 도메인 특성으로 변환[2]

**자기 돌연변이 네트워크 (Self-Mutating Network)**: 매개변수 동적 조정을 통한 도메인 적응[3]

#### 4.3 데이터 증강 (Data Augmentation)

여러 증강 기법의 조합:

- **회전 및 뒤집기**: $$I_{augmented} = \text{rotate}(I, \theta)$$
- **색상 변환**: $$I' = \alpha I + \beta$$
- **혼합 (Mixup)**: $$\tilde{x} = \lambda x_i + (1-\lambda) x_j$$, $$\tilde{y} = \lambda y_i + (1-\lambda) y_j$$
- **자르기 및 확대**: 국소 특성 강조

**최신 연구 결과 (2024-2025)**: Transfer Learning + Data Augmentation 결합 시 IoU 0.814~0.858 달성[4]

#### 4.4 주의 메커니즘 (Attention Mechanisms)[1]

**채널 주의 (Channel Attention)**:

$$M_c = \sigma(MLP(\text{AvgPool}(F)) + MLP(\text{MaxPool}(F)))$$

**공간 주의 (Spatial Attention)**:

$$M_s = \sigma(Conv([\text{AvgPool}(F); \text{MaxPool}(F)]))$$

**전역 주의 업샘플 모듈**: 저수준과 고수준 특성의 결합

**Vision Transformer (ViT)와 DETR**: 패치 기반 주의 메커니즘으로 다중 스케일 특성 추출 개선[5][6]

#### 4.5 소수 학습 (Few-Shot Learning)[1]

메타 학습 기반 접근:

$$\min_\theta \mathbb{E}_{T \sim p(T)} [\mathcal{L}(f_\theta, S, Q)]$$

여기서 S는 지원 세트(support set), Q는 쿼리 세트(query set)

- **전이 학습과의 차이**: 사전 훈련된 모델의 미세 조정보다 메타 학습이 더 유연
- **UAV 맥락에서의 적용**: 새로운 지리적 영역이나 객체 클래스에 대한 빠른 적응

#### 4.6 반감독 및 비감독 학습[1]

**대조 손실 (Contrastive Loss)**:

$$L_{con} = -\log \frac{\exp(\text{sim}(z_i, z_{pos}) / \tau)}{\sum_j \exp(\text{sim}(z_i, z_j) / \tau)}$$

**클러스터링 기반 방법**: 유사한 특성을 가진 이미지 그룹화

라벨링되지 않은 대량의 UAV 영상 활용으로 일반화 능력 향상

***

### 5. 한계 및 과제

**논문에서 확인된 주요 한계**[1]

1. **라벨 데이터 부족**: 특히 다중 분광 및 초분광 데이터의 공개 데이터셋 부재
2. **실시간 처리 능력**: GPU 의존으로 임베디드 시스템에서의 실시간 처리 곤란
3. **모델 해석성**: 블랙박스 특성으로 인한 의사결정 과정의 불명확성
4. **높은 계산량**: 깊은 신경망의 매개변수 개수로 인한 연산 부담
5. **도메인 적응의 한계**: 완벽한 도메인 일치 달성의 어려움

**해결 방안**[1]

- **경량 모델 (Lightweight Models)**: MobileNets, SqueezeNet 등의 압축 아키텍처
- **양자화 (Quantization)**: 32비트를 8비트 또는 이진 표현으로 축소
  
  $$w_{quantized} = \text{round}\left(\frac{w \cdot (2^{bits} - 1)}{\max(|w|)}\right)$$

- **지식 증류 (Knowledge Distillation)**: 대규모 교사 네트워크의 지식을 소규모 학생 네트워크로 이전
- **신경망 아키텍처 검색 (NAS)**: 자동화된 모델 설계

***

### 6. 최신 연구 기반 향후 연구 방향

**최신 동향 (2024-2025)**[7][8][9][10][11][12][6][13][2]

#### 6.1 비전 트랜스포머와 Segment Anything Model (SAM)

**Vision Transformer의 발전**:
- ViT는 이미지 패치 기반의 자기 주의(self-attention) 메커니즘 활용
- 다중 스케일 특성 추출과 부분 폐색 환경에서의 우수한 성능
- DETR(Detection Transformer)의 간소화된 검출 파이프라인[6]

**Segment Anything Model (SAM)**:
- 미세 조정된 SAM이 홍수 감지에서 IoU 0.85, 정확도 0.90 달성[8]
- U-Net 및 Mask R-CNN 대비 우수한 성능 시연

#### 6.2 멀티모달 접근법

**다중 센서 융합**:[9][5]
- RGB, 열적외선, RF 신호 결합
- 멀티모달 Transformer로 정보 융합 효율성 향상
- 비전-언어 모델 활용하여 작물 분할 작업을 텍스트 기반 좌표 예측으로 변환[9]

#### 6.3 실시간 처리 최적화

**경량 아키텍처**:[10][13]
- YOLOv5/v8 계열: 84.8% AP 정확도 달성 with 빠른 추론 속도
- MLD-DETR: VisDrone 데이터셋에서 36.7% AP50 달성, 20% 매개변수 축소[13]
- 모바일 및 엣지 디바이스(NVIDIA Jetson) 최적화

#### 6.4 일반화 능력 개선[4]

**실제 데이터의 도전**:
- 실제 UAV 데이터셋이 벤치마크 데이터셋보다 훨씬 어려움 입증
- Transfer Learning + Data Augmentation 결합으로 IoU 0.814-0.858 달성[4]
- 도메인 적응 및 오픈셋 인식 필요성 강조

#### 6.5 자감독 및 대조 학습[2]

**동적 손실 함수**:
- 교차 센서, 계절별 이미지 불균형 문제 해결[7]
- 자기 감독 대비 학습으로 라벨 데이터 필요성 감소
- 대규모 라벨 미지정 UAV 영상 활용[1]

#### 6.6 다중 작업 학습 (Multitask Learning)[1]

**통합 작업 처리**:
- 의미론적 분할 + 높이 추정 + 경계 검출 동시 수행
- 식물 검출 + 식재 행 감지 동시 처리로 성능 향상
- 관련 작업들 간의 지식 공유로 일반화 개선

#### 6.7 고해상도 특성 유지

**HRNet 및 후속 아키텍처**:
- UAV 영상의 고해상도(0.5~5cm GSD) 장점 활용
- 더 깊은 층에서도 고해상도 유지로 작은 객체 검출 개선
- 실시간 처리와의 트레이드오프 조정 필요

#### 6.8 포토그래메트리 처리 통합

**구조-동작(Structure-from-Motion) 기반 개선**:
- DL을 활용한 특성 매칭 개선
- DSM(Digital Surface Model) 필터링에 DL 적용
- 고품질 정사영상(orthomosaic) 생성 최적화

***

### 7. 결론

**종합 평가**

이 논문은 **UAV 원격감시와 딥러닝의 결합이 제공하는 광범위한 가능성**을 체계적으로 제시합니다. 232개 논문의 체계적 분석을 통해:

1. **CNN 중심의 현황**: 91.2%가 CNN 기반이며, CNN-LSTM과 GAN 도입 증가
2. **RGB 센서 지배**: 저비용과 가용성으로 52.4% 이상의 논문에서 활용
3. **객체 검출 우위**: 53.9% 논문이 객체 검출에 집중, 분할 기법도 40.7%로 증가
4. **응용 영역 다양성**: 환경(46.6%), 도시(27.2%), 농업(26.4%) 광범위한 분포

**향후 중요 과제**[1]

- 공개 라벨 데이터셋 확충 (특히 다중분광, 초분광 데이터)
- 실시간 처리를 위한 경량 모델 개발
- 도메인 적응 및 일반화 능력 강화
- 오픈셋 인식, 소수 학습, 다중 작업 학습의 통합
- 고해상도 특성 유지와 계산 효율성의 균형

**최신 기술 적용 방향**

2024-2025년 최신 연구는 **Vision Transformer, Segment Anything Model, 멀티모달 융합, 자감독 학습**으로의 전환을 보여줍니다. 특히 SAM 미세 조정으로 홍수 감지에서 0.90 정확도 달성, YOLOv8의 84.8% AP 성과 등 실질적 개선을 입증합니다. **일반화 성능 향상을 위해서는 Transfer Learning과 Data Augmentation의 결합, 도메인 적응 기법의 정교화, 대규모 비라벨 데이터 활용**이 필수적입니다.

이 논문은 UAV-DL 교집합에 처음으로 포괄적 리뷰를 제공함으로써, 기술의 현재 수준을 명확히 하고 미래 연구의 청사진을 제시하는 중요한 기여를 합니다.

***

#### 참고 문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4bda9d5f-b671-4898-969e-d3fcc46074c0/2101.10861v4.pdf)
[2](https://arxiv.org/abs/2405.07520)
[3](https://pmc.ncbi.nlm.nih.gov/articles/PMC8662429/)
[4](https://inz-min.online/index.php/im/en/article/view/1360/2577)
[5](https://arxiv.org/html/2511.15312v1)
[6](https://www.scitepress.org/Papers/2025/134679/134679.pdf)
[7](https://ieeexplore.ieee.org/document/11108583/)
[8](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13818/3082298/Flood-detection-for-satellite-and-UAV-remote-sensing-based-on/10.1117/12.3082298.full)
[9](https://ieeexplore.ieee.org/document/11136213/)
[10](https://www.agroengineering.org/jae/article/view/1641)
[11](https://www.mdpi.com/2077-0472/15/12/1309)
[12](https://ieeexplore.ieee.org/document/11053843/)
[13](https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2025.1599099/full)
[14](https://link.springer.com/10.1007/s40808-024-02222-w)
[15](https://www.mdpi.com/2075-5309/14/8/2344)
[16](https://www.mdpi.com/2073-4441/17/21/3160)
[17](https://www.mdpi.com/2504-446X/7/4/236/pdf?version=1680070990)
[18](https://www.mdpi.com/2072-4292/15/7/1873/pdf?version=1680363388)
[19](https://www.mdpi.com/2072-4292/13/7/1358/pdf)
[20](https://www.mdpi.com/2673-3951/5/4/92)
[21](https://www.mdpi.com/2072-4292/16/5/925/pdf?version=1709718717)
[22](https://arxiv.org/pdf/2211.04324.pdf)
[23](http://arxiv.org/pdf/2101.10861.pdf)
[24](https://www.mdpi.com/1424-8220/19/22/4837/pdf)
[25](https://openaccess.thecvf.com/content/ICCV2021/papers/Lee_Self-Mutating_Network_for_Domain_Adaptive_Segmentation_in_Aerial_Images_ICCV_2021_paper.pdf)
[26](https://www.sciencedirect.com/science/article/abs/pii/S0043135425006694)
[27](https://ieeexplore.ieee.org/iel8/4609443/10766875/10815625.pdf)
[28](https://www.scitepress.org/Papers/2025/131450/131450.pdf)
[29](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2024.1435016/full)
