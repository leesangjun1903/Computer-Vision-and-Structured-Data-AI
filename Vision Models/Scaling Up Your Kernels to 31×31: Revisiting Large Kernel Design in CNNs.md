# Scaling Up Your Kernels to 31×31: Revisiting Large Kernel Design in CNNs

### 1. 핵심 주장과 주요 기여

이 논문은 Vision Transformers(ViTs)의 등장으로 인한 CNN의 성능 격차를 극복하기 위해 **대규모 합성곱 커널(31×31)의 재도입**을 제안합니다. 핵심 주장은 많은 작은 커널(3×3)을 쌓는 대신 **소수의 큰 커널을 사용하는 것이 더 강력한 패러다임**이라는 것입니다. 이를 통해 RepLKNet 아키텍처를 제안하며, 이는 Swin Transformer에 필적하거나 우수한 성능을 달성하면서도 더 낮은 레이턴시를 보여줍니다.[1]

### 2. 해결 문제와 제안 방법

#### 2.1 핵심 문제

전통적인 CNN 설계에서는 커널 크기가 3×3에 국한되어 있으며, Receptive Field(RF)를 확장하기 위해 많은 계층을 깊이 있게 쌓습니다. 그러나 깊은 신경망은 최적화 문제를 야기하고, 실제 유효 수용장(Effective Receptive Field, ERF)은 이론적 수용장보다 훨씬 작습니다.[1]

#### 2.2 제안된 해결책: 5가지 설계 가이드라인

논문은 효율적인 대규모 커널 CNN 설계를 위해 다음 5가지 가이드라인을 제시합니다:[1]

**가이드라인 1: 깊이별 대규모 합성곱의 효율성**
깊이별 합성곱(Depthwise Convolution)을 적용하면 계산 복잡도를 크게 줄일 수 있습니다. 예를 들어, RepLKNet에서 커널 크기를 (3,3,3,3)에서 (31,29,27,13)으로 변경할 때 FLOPs는 18.6%, 매개변수는 10.4%만 증가합니다.[1]

**가이드라인 2: 항등원 단축(Identity Shortcut)의 필수성**
아이덴티티 단축이 없는 MobileNet V2에서 13×13 커널은 정확도를 53.98%로 급락시키지만, 단축이 있으면 72.53%를 달성합니다. 이는 대규모 커널 네트워크의 최적화에 매우 중요합니다.[1]

**가이드라인 3: 구조적 재매개변수화(Structural Re-parameterization)**
수식으로 표현하면, 학습 중에는 큰 커널과 작은 커널을 병렬로 구성하고:[1]

$$\text{Output} = \text{LargeConv}(x) + \text{SmallConv}(x)$$

훈련 후 두 커널을 배치 정규화와 함께 합쳐서 단일 커널로 만듭니다. 이 기법으로 9×9에서 13×13로 커널 크기를 증가시킬 때 정확도 감소를 방지합니다.[1]

**가이드라인 4: 다운스트림 작업에서의 더 강력한 부스팅**
대규모 커널은 ImageNet에서는 소폭의 개선을 제공하지만 다운스트림 작업에서 훨씬 큰 이득을 제공합니다. MobileNet V2에서 커널 크기를 3×3에서 9×9로 증가시킬 때 ImageNet 정확도는 1.33% 증가하지만, Cityscapes mIoU는 3.99% 증가합니다.[1]

**가이드라인 5: 작은 특성 맵에서도 유용한 대규모 커널**
커널 크기가 특성 맵 크기와 유사하거나 더 클 때도 성능 향상이 관찰됩니다. MobileNet V2의 마지막 단계(7×7 특성 맵)에서 13×13 커널을 사용하면 Cityscapes에서 2.31 mIoU 개선을 달성합니다.[1]

### 3. RepLKNet 모델 구조

#### 3.1 전체 아키텍처

RepLKNet은 다음과 같은 구성요소로 이루어집니다:[1]

**Stem 모듈**: 초기 특성 추출 층
- 3×3 합성곱 (stride 2)
- 3×3 깊이별 합성곱
- 1×1 합성곱
- 3×3 깊이별 합성곱 (stride 2)

**RepLK Block**: 재매개변수화된 대규모 커널을 활용
- 1×1 합성곱
- K×K 깊이별 합성곱 (K ∈ {13, 27, 29, 31})
- 1×1 합성곱
- 각 레이어는 배치 정규화 포함

**ConvFFN Block**: Transformer의 Feed-Forward Network에서 영감

$$\text{ConvFFN}(x) = \text{Conv}_{1×1}(\text{GELU}(\text{Conv}_{1×1}(x))) + x$$

여기서 내부 채널은 입력 채널의 4배입니다.[1]

**Transition Block**: 단계 간 해상도 조정
- 1×1 합성곱으로 채널 증가
- 3×3 깊이별 합성곱으로 2배 다운샘플링

#### 3.2 아키텍처 하이퍼파라미터

RepLKNet은 다음과 같이 정의됩니다:[1]

$$(B_1, B_2, B_3, B_4, C_1, C_2, C_3, C_4, K_1, K_2, K_3, K_4)$$

여기서:
- B: 각 단계의 RepLK Block 개수
- C: 채널 차원
- K: 커널 크기

RepLKNet-31 기본 모델은:
- $$(2, 2, 18, 2, 128, 256, 512, 1024, 31, 29, 27, 13)$$

### 4. 성능 향상 분석

#### 4.1 ImageNet 분류 성능[1]

| 모델 | 입력해상도 | 정확도(%) | 매개변수 | FLOPs |
|------|----------|---------|--------|-------|
| RepLKNet-31B | 224×224 | 83.5 | 79M | 15.3G |
| Swin-B | 224×224 | 83.5 | 88M | 15.4G |
| RepLKNet-31B | 384×384 | 84.8 | 79M | 45.1G |
| Swin-B | 384×384 | 84.5 | 88M | 47.0G |
| RepLKNet-31L | 384×384 | 86.6 | 172M | 96.0G |
| Swin-L | 384×384 | 87.3 | 197M | 103.9G |

#### 4.2 다운스트림 작업 성능

**의미론적 분할 (ADE20K)**:[1]
- RepLKNet-31B: 51.5% mIoU (ImageNet-1K 사전학습)
- Swin-B: 50.0% mIoU
- RepLKNet-XL: 56.0% mIoU (MegData73M 사전학습)
- Swin-L: 53.5% mIoU

**물체 감지 (COCO)**:[1]
- RepLKNet-31B (Cascade Mask R-CNN): 52.2 AP
- Swin-B: 51.9 AP
- RepLKNet-XL: 55.5 AP
- Swin-L: 53.9 AP

### 5. 일반화 성능 향상 가능성

#### 5.1 유효 수용장(ERF) 확대[1]

유효 수용장은 다음과 같이 정의됩니다:

$$\text{ERF} \propto \sqrt{K \cdot L}$$

여기서 K는 커널 크기, L은 깊이입니다. 커널 크기가 깊이보다 더 효율적으로 ERF를 증가시킵니다.[1]

**정량적 분석**: ResNet-101과 RepLKNet-31을 비교하면:
- ResNet-101에서 99% 기여도의 픽셀이 이미지의 23.4%만 차지
- RepLKNet-31에서는 98.6% 기여도의 픽셀이 98.6% 영역에 분포

이는 RepLKNet이 훨씬 더 광범위한 공간 정보를 활용함을 의미합니다.[1]

#### 5.2 형태 편향(Shape Bias) 향상[1]

소규모 커널 CNN은 텍스처 편향을 보이는 반면, 대규모 커널 CNN은 형태 편향을 보입니다. Shape-texture 실험 결과:

- RepLKNet-31 (ImageNet-1K): ~65% 형태 편향
- Swin-B (ImageNet-1K): ~55% 형태 편향
- ResNet-152: ~25% 형태 편향

형태 편향은 다운스트림 작업에서의 전이 성능을 향상시킵니다.

#### 5.3 확장성(Scalability)[1]

RepLKNet은 대규모 데이터와 모델에 대해 우수한 확장성을 보입니다:
- MegData73M(7,300만 이미지)으로 사전학습한 RepLKNet-XL: 87.8% ImageNet top-1 정확도
- ADE20K에서 56.0% mIoU 달성

### 6. 한계점[1]

논문이 인정하는 주요 한계:

1. **대규모 데이터/모델에서의 격차**: 데이터와 모델 규모가 증가함에 따라 RepLKNet이 Swin Transformer보다 뒤떨어지는 경향
   - ImageNet-22K 사전학습 시: RepLKNet-31L (86.6%)이 Swin-L (87.3%)보다 0.7% 낮음

2. **하이퍼파라미터 튜닝**: 최적의 성능을 위한 세밀한 튜닝 필요

3. **구현 복잡성**: CUDA 커널 최적화 필요

### 7. 최신 연구 기반 영향과 향후 고려사항

#### 7.1 후속 연구에 미친 영향[2][3][4]

**1) 대규모 커널의 확장**: SLaK(Sparse Large Kernel Network, 2022)는 RepLKNet의 개념을 확장하여 51×51에서 61×61까지 커널을 확대하면서 희소성(sparsity)을 활용.[5]

**2) 범용 모델로의 확장**: UniRepLKNet(2024)은 RepLKNet을 음성, 비디오, 포인트 클라우드, 시계열 예측 등 여러 모달리티로 확장하여 대규모 커널 설계의 보편성을 입증.[3][2]

**3) 다양한 응용 분야**:
- 의료 영상 분할: LK-UNet(2024)[6]
- 원격 감지 영상: ULKNet(2023)[7]
- 초해상도: PLKSR(2024)[8]
- 약한 감독 객체 로컬리제이션: CAM Back Again(2024)[9]

**4) 지식 증류에서의 우수성**: 2023년 ICML 논문에서 대규모 커널 CNN(RepLKNet)이 Vision Transformer보다 작은 커널 CNN에 대한 더 효과적인 교사(teacher) 역할을 함을 증명.[10]

#### 7.2 주요 아키텍처 트렌드[11][12][13]

**ConvNeXt와의 수렴**: 2022년 Meta의 ConvNeXt는 7×7 깊이별 합성곱을 중심으로 설계하여 RepLKNet의 아이디어와 유사한 방향으로 발전. 이후 ConvNeXt V2(2023)는 마스크된 자동인코더와 결합하여 성능을 한 단계 더 높였습니다.[12][13]

**하이브리드 아키텍처의 부상**: Large Kernel Attention(LKA)을 활용한 Visual Attention Networks(VAN)은 CNN의 국소성과 Transformer의 전역성을 결합하여 선형 계산 복잡도를 유지하면서 성능을 달성.[14]

**웨이블릿 기반 접근**: 최근(2024) WTConv는 웨이블릿 변환을 활용하여 매개변수를 로그적으로만 증가시키면서 지수적으로 수용장을 확대.[15]

#### 7.3 향후 연구 시 고려할 점[3][14][1]

**1) 효율성-성능 트레이드오프**
- 모바일/엣지 디바이스를 위한 경량 대규모 커널 설계
- 구조적 재매개변수화 기법의 추가 발전
- 부분 대규모 커널 설계(선택적으로 특정 단계에만 적용)

**2) 데이터 규모에 따른 적응**
- 소규모 데이터: 구조적 재매개변수화 및 정규화 기법 강화 필요
- 대규모 데이터: 재매개변수화 제거 가능(ViT와 유사)

**3) 다양한 작업에 최적화된 커널 크기**
- 분류: 중간 크기 커널(13×13~25×25) 충분
- 의미론적 분할: 더 큰 커널(27×27~31×31) 유리
- 객체 감지: 작업 특성에 따른 동적 커널 크기 조정

**4) 희소 연산과의 통합**
- 계산 효율성 유지 하에서 초대형 커널(61×61+) 구현
- 동적 희소성 메커니즘 도입

**5) 이해 가능성(Interpretability)**
- 대규모 커널이 어떤 특징을 학습하는지의 심층 분석
- 형태 편향 증가의 원인에 대한 이론적 분석

**6) 크로스모달 적용**
- 3D 의료 영상, 포인트 클라우드, 시계열 데이터 등에서의 대규모 커널 설계 최적화
- 각 모달리티에 맞는 커널 크기 결정 프레임워크 개발

### 결론

"Scaling Up Your Kernels to 31×31" 논문은 CNN의 설계 철학에 근본적인 질문을 제기하고, 대규모 커널의 효율적 활용을 통해 Vision Transformer와의 성능 격차를 크게 좁혔습니다. 5가지 설계 가이드라인과 구조적 재매개변수화 기법은 이제 CNN 커뮤니티의 표준 관행이 되었으며, 특히 다운스트림 작업에서의 우수한 성능과 확장성은 향후 연구의 중요한 토대가 되었습니다. 형태 편향 향상과 유효 수용장 확대는 생물학적 시각과의 더 나은 일치를 시사하며, 이는 인간과 유사한 인공 시각 시스템 개발의 길을 열었습니다.[1]

***

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f6c52f3e-8c06-4f36-b05d-df0d0ef381df/2203.06717v4.pdf)
[2](https://arxiv.org/pdf/2410.08049.pdf)
[3](http://arxiv.org/pdf/2311.15599.pdf)
[4](https://openreview.net/pdf?id=QJLGj57MfZ)
[5](https://arxiv.org/abs/2207.03620)
[6](https://ieeexplore.ieee.org/document/10446818/)
[7](https://ieeexplore.ieee.org/document/10312040/)
[8](http://arxiv.org/pdf/2404.11848.pdf)
[9](https://arxiv.org/abs/2403.06676)
[10](https://arxiv.org/abs/2305.19412)
[11](https://arxiv.org/pdf/2309.01439.pdf)
[12](http://arxiv.org/pdf/2301.00808.pdf)
[13](https://towardsdatascience.com/the-cnn-that-challenges-vit/)
[14](https://www.scirp.org/journal/paperinformation?paperid=147138)
[15](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07137.pdf)
[16](https://ieeexplore.ieee.org/document/9880273/)
[17](https://www.semanticscholar.org/paper/63f1f2dad0a2e84d37a97258008c5609195487f0)
[18](https://www.mdpi.com/2079-9292/11/20/3351)
[19](https://aclanthology.org/2022.semeval-1.55)
[20](https://ieeexplore.ieee.org/document/10177422/)
[21](https://arxiv.org/abs/2207.08810)
[22](https://link.springer.com/10.1007/s12652-022-04025-2)
[23](https://ieeexplore.ieee.org/document/10136788/)
[24](https://www.hindawi.com/journals/cin/2022/4316812/)
[25](https://arxiv.org/pdf/2203.06717.pdf)
[26](http://arxiv.org/pdf/2402.14307.pdf)
[27](https://pmc.ncbi.nlm.nih.gov/articles/PMC9775903/)
[28](https://www.mdpi.com/2076-3425/12/12/1633/pdf?version=1669706096)
[29](https://openaccess.thecvf.com/content/CVPR2022/papers/Ding_Scaling_Up_Your_Kernels_to_31x31_Revisiting_Large_Kernel_Design_CVPR_2022_paper.pdf)
[30](https://pmc.ncbi.nlm.nih.gov/articles/PMC9871543/)
[31](https://www.nature.com/articles/s41598-023-36724-x)
[32](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Chen_Large_Kernel_Frequency-enhanced_Network_for_Efficient_Single_Image_Super-Resolution_CVPRW_2024_paper.pdf)
[33](https://www.sciencedirect.com/science/article/abs/pii/S0957417423018547)
[34](https://ieeexplore.ieee.org/document/10461992/)
[35](https://linkinghub.elsevier.com/retrieve/pii/S0957417423018547)
[36](https://ieeexplore.ieee.org/document/10982291/)
[37](https://www.mdpi.com/2072-4292/17/8/1461)
[38](https://ieeexplore.ieee.org/document/9863852/)
[39](https://ieeexplore.ieee.org/document/10635609/)
[40](https://ieeexplore.ieee.org/document/11094588/)
[41](http://arxiv.org/pdf/2303.09975.pdf)
[42](https://arxiv.org/pdf/2303.16900.pdf)
[43](https://arxiv.org/pdf/2207.03620.pdf)
[44](https://arxiv.org/pdf/2206.10555.pdf)
[45](https://d-nb.info/1275843778/34)
[46](https://arxiv.org/html/2208.07463v4)
[47](https://ise.thss.tsinghua.edu.cn/mig/2022-10.pdf)
[48](https://past.date-conference.com/proceedings-archive/2019/pdf/0715.pdf)
[49](https://proceedings.neurips.cc/paper/2021/file/c404a5adbf90e09631678b13b05d9d7a-Paper.pdf)
[50](https://openaccess.thecvf.com/content/CVPR2024/papers/Ding_UniRepLKNet_A_Universal_Perception_Large-Kernel_ConvNet_for_Audio_Video_Point_CVPR_2024_paper.pdf)
[51](https://arxiv.org/html/2406.16004v2)
[52](https://pmc.ncbi.nlm.nih.gov/articles/PMC3840297/)
[53](https://wuyaho.tistory.com/30)
