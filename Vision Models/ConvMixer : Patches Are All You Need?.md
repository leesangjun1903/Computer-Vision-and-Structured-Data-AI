# Patches Are All You Need?

### 1. 핵심 주장과 주요 기여

**"Patches Are All You Need?"** 논문의 핵심 주장은 **Vision Transformer(ViT)의 우수한 성능이 Transformer 아키텍처 자체보다는 Patch 기반의 입력 표현에서 비롯되었을 가능성**이 높다는 것입니다. 저자들은 이를 증명하기 위해 **ConvMixer**라는 극히 단순한 CNN 기반 모델을 제안했습니다.[1]

**주요 기여:**

1. **Patch 표현의 중요성 규명**: 기존 연구들이 자동어텐션(Self-Attention)이나 MLP 같은 새로운 연산에만 초점을 맞췄다면, 이 논문은 patch-based representation 자체가 성능 향상의 핵심 요인임을 보여주었습니다.[2][1]

2. **초단순 아키텍처의 강력한 성능**: ConvMixer는 약 6줄의 Python 코드로 구현 가능하면서도 ViT, MLP-Mixer, ResNet 대비 동등하거나 우수한 성능을 달성했습니다.[1]

3. **Isotropic 아키텍처의 가치 입증**: 모든 레이어에서 동일한 크기와 해상도를 유지하는 구조가 충분히 강력함을 증명했습니다.[1]

***

### 2. 해결하고자 하는 문제와 제안하는 방법

#### 2.1 핵심 문제

ViT와 같은 Transformer 기반 모델이 이미지 분류에서 CNN을 능가하기 시작했지만, 다음과 같은 의문이 제기됩니다:

- ViT의 성능이 **Self-Attention 메커니즘** 때문인가?
- 아니면 **Patch 기반의 입력 표현** 때문인가?

Self-Attention의 계산복잡도는 입력 길이의 제곱에 비례하므로, ViT가 이미지에 적용되려면 먼저 이미지를 패치로 분할한 후 각 패치를 토큰으로 처리해야 합니다. 이는 근본적인 한계입니다.[1]

#### 2.2 ConvMixer 아키텍처

ConvMixer는 다음과 같이 구성됩니다:

**1) Patch Embedding Layer**

이미지를 패치로 분할하고 임베딩하는 과정:

```math
z_0 = \text{BN}(\sigma(\text{Conv}_{c_{in} \to h}(X, \text{stride}=p, \text{kernel\_size}=p)))
```

여기서:
- $$z_0$$: 패치 임베딩 출력
- $$p$$: 패치 크기
- $$h$$: 임베딩 차원(히든 차원)
- $$\sigma$$: 활성화 함수(GELU)
- $$\text{BN}$$: Batch Normalization

**2) ConvMixer Block (반복 적용)**

각 블록은 **Spatial Mixing**과 **Channel Mixing**의 두 단계로 구성:

**Spatial Mixing (Depthwise Convolution):**

$$z'_l = \text{BN}(\sigma(\text{ConvDepthwise}(z_{l-1}))) + z_{l-1}$$

- **Depthwise Convolution**: 각 채널에 독립적으로 작동하는 그룹 컨볼루션($$\text{groups}=h$$)
- **목적**: 공간적 위치 간 정보 혼합(Spatial mixing)
- **핵심 특징**: **매우 큰 커널 크기**(예: $$k=9$$)를 사용하여 먼 거리의 픽셀 간 상관관계 학습
  - Self-Attention의 전역 수용장(Global receptive field)을 모방
  - 작은 커널 여러 개 대신 큰 커널 하나 사용이 효과적

**Channel Mixing (Pointwise Convolution):**

$$z_{l+1} = \text{BN}(\sigma(\text{ConvPointwise}(z'_l)))$$

- **Pointwise Convolution**: $$1 \times 1$$ 커널을 사용한 컨볼루션
- **목적**: 채널 간 정보 혼합(Channel mixing)

**PyTorch 구현 (약 6줄의 코드):**

```python
def ConvMixer(h, depth, kernel_size=9, patch_size=7, n_classes=1000):
    Seq, ActBn = nn.Sequential, lambda x: Seq(x, nn.GELU(), nn.BatchNorm2d(h))
    Residual = type('Residual', (Seq,), {'forward': lambda self, x: self[0](x) + x})
    return Seq(ActBn(nn.Conv2d(3, h, patch_size, stride=patch_size)),
        *[Seq(Residual(ActBn(nn.Conv2d(h, h, kernel_size, groups=h, padding="same"))),
        ActBn(nn.Conv2d(h, h, 1))) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(h, n_classes))
```

**3) 최종 분류 계층**

Global Average Pooling → Fully Connected Layer로 분류

**설계 하이퍼파라미터:**

- $$h$$: 히든 차원(임베딩 차원)
- $$d$$: 깊이(ConvMixer 블록 반복 횟수)
- $$p$$: 패치 크기
- $$k$$: Depthwise Convolution의 커널 크기

***

### 3. 모델 구조의 특징

#### 3.1 주요 설계 원칙

1. **Isotropic Design (균등한 해상도 유지)**
   - 모든 레이어에서 동일한 공간 해상도 유지
   - 기존 CNN의 Pyramid 구조(점진적 다운샘플링)와 대조적

2. **Spatial-Channel 혼합 분리**
   - Depthwise Conv: 공간 정보만 혼합
   - Pointwise Conv: 채널 정보만 혼합
   - 각 차원의 혼합을 명확히 분리

3. **대규모 커널 사용**
   - 일반적인 $$3 \times 3$$ 커널 대신 $$7 \times 9 \times 9$$ 등 사용
   - ViT와 MLP-Mixer처럼 원거리 의존성 학습 가능

4. **Residual Connection**
   - Depthwise Convolution 이후 Residual 연결
   - 깊은 네트워크 학습 안정화

***

### 4. 성능 향상 및 실험 결과

#### 4.1 ImageNet-1k 성능 (224×224 입력)

| 모델 | 파라미터(×10⁶) | 정확도(%) | 특징 |
|------|----------------|---------|------|
| ConvMixer-1536/20 | 51.6 | 81.37 | 큰 히든 차원, 얕은 깊이 |
| ConvMixer-768/32 | 21.1 | 80.16 | 작은 히든 차원, 깊은 구조 |
| ResNet-152 | 60.2 | 79.64 | 기존 CNN 베이스라인 |
| DeiT-B | 86.0 | 81.8 | Vision Transformer |
| ResMLP-B24/8 | 129 | 81.0 | MLP-Mixer 변형 |

**주요 발견:**

1. **동일한 파라미터 수에서 우수한 성능**
   - ConvMixer-1536/20: DeiT-B보다 35M 적은 파라미터로 유사한 정확도 달성[1]
   - ConvMixer-768/32: DeiT-S보다 0.9M 적은 파라미터로 더 높은 정확도(80.16% vs 79.8%)[1]

2. **커널 크기의 중요성**
   - $$k=9$$일 때: 81.37% 정확도
   - $$k=3$$일 때: 80.43% 정확도 (약 0.94% 감소)[1]
   - 깊이를 늘려도 큰 커널의 효과를 완전히 대체할 수 없음

3. **패치 크기의 영향**
   - 작은 패치 크기가 우수한 성능 제공
   - 패치 크기를 7에서 14로 증가 시 정확도 78.92%로 감소(~2.45% 감소)[1]
   - 더 큰 패치는 더 깊은 네트워크 필요

#### 4.2 CIFAR-10 성능 (저용량 데이터셋)

- ConvMixer-256/8: 96.04% 정확도 (약 0.9M 파라미터)
- ConvMixer로 100K 파라미터 미만에서도 91% 이상 정확도 달성
- **데이터 효율성** 입증: 작은 데이터셋에서도 우수한 성능 유지[1]

#### 4.3 파라미터 공식

ConvMixer의 파라미터 수는 다음 공식으로 정확히 계산됨:

```math
\#\text{params} = h[d(k^2 + h + 6) + c_{in}p^2 + n_{classes} + 3] + n_{classes}
```

여기서:
- $$h$$: 히든 차원
- $$d$$: 깊이
- $$k$$: 커널 크기
- $$p$$: 패치 크기
- $$c_{in}$$: 입력 채널 수 (3)
- $$n_{classes}$$: 클래스 수

***

### 5. 일반화 성능 향상 가능성 (심화 분석)

#### 5.1 일반화 성능 향상의 메커니즘

**1) Patch Embedding의 역할**

Patch Embedding은 한 번에 이미지를 다운샘플링하여:
- 효과적인 수용장(Effective Receptive Field) 증가
- 전역 정보 혼합 용이
- 초기 계층부터 "높은" 수준의 추상화 가능[3][4][1]

**2) 큰 커널의 효과**

Depthwise Convolution에서 큰 커널 사용:
- **장점**: 먼 거리 의존성 학습, Self-Attention 모방
- **한계**: 계산 비용 증가, 초기 가중치 분포 문제

최근 연구(2024)에서 **UniRepLKNet**은 이를 개선:[5]
- 대규모 데이터셋에서 ImageNet 정확도 **88.0%** 달성
- ADE20K 분할(Segmentation)에서 **55.6% mIoU**
- COCO 객체 탐지에서 **56.4% box AP**
- 멀티모달(audio, video, time-series) 확장으로 범용성 입증[5]

**3) Isotropic 아키텍처의 장점**

동일한 해상도 유지:
- 각 레이어에서 공간적 세부사항 보존
- 의미론적 정보(Semantic information) 손실 최소화
- 객체 탐지, 의미론적 분할 같은 Dense Prediction 태스크에 유리[1]

#### 5.2 최근 연구의 일반화 성능 개선

**1) 데이터 효율성**

에서 제안한 **Large Kernel Convolutional Attention(LKCA)**:[6]
- 제한된 데이터 상황에서 ViT 능가
- CIFAR-10: 98% 이상 정확도
- CIFAR-100: 약 80% 정확도 달성
- **핵심 개선**: 대규모 데이터셋 의존성 감소[6]

**2) 멀티모달 확장 가능성**

의 UniRepLKNet:[5]
- Vision 뿐만 아니라 **Audio, Video, Time-Series** 처리 가능
- 동일한 아키텍처 원칙으로 다양한 도메인에 적용
- **일반화 성능**: 모든 도메인에서 경쟁력 있는 성능 달성[5]

**3) 도메인 적응(Domain Adaptation)**

의 OAMixer:[2]
- Object-aware Mixing Layer로 ConvMixer 개선
- 배경 강건성(Background Robustness) 향상
- 자기지도학습(Self-Supervised Learning) 성능 개선[2]

#### 5.3 일반화 성능의 한계 및 고려사항

1. **계산 효율성**
   - 큰 커널의 낮은 처리량(Throughput)
   - ConvMixer-1536/20: 134 img/sec (DeiT-B: 792 img/sec)[1]
   - 최적화 필요: 저수준 구현, 희소성(Sparsity) 활용

2. **충분한 정규화 필요**
   - 원논문은 제한된 하이퍼파라미터 튜닝 수행
   - 더 나은 정규화 설정으로 추가 개선 가능
   - Stochastic Depth, DropPath 같은 고급 정규화 미적용[1]

3. **아키텍처 일반화**
   - 대규모 사전학습(Large-scale Pre-training)에서의 성능 미검증
   - ImageNet-1k 정도의 중규모 데이터셋에서 검증만 수행

***

### 6. 모델의 한계

#### 6.1 구조적 한계

1. **처리량 문제**
   - Depthwise Convolution이 GPU 최적화되어 있지 않음
   - 같은 파라미터의 ResNet 대비 4-6배 느림
   - 실제 배포에서의 제약

2. **패치 크기의 트레이드오프**
   - 작은 패치: 정확도 우수하지만 처리량 저하
   - 큰 패치: 빠르지만 정확도 감소
   - 최적의 균형점 찾기 어려움

3. **깊이 vs 너비의 설계**
   - 깊은 네트워크 학습이 어려움
   - 같은 파라미터 수에서 깊이 증가는 정확도 감소
   - 최적 설계점 탐색 필요

#### 6.2 실험적 한계

1. **제한된 하이퍼파라미터 튜닝**
   - 원논문에서 일반적인 설정만 사용
   - "Common sense" 파라미터로 일관성 유지
   - 추가 튜닝 시 성능 향상 여지[1]

2. **대규모 사전학습 미검증**
   - ImageNet-1k에서만 평가
   - ViT의 강점인 대규모 데이터셋(예: ImageNet-21k) 성능 미평가
   - 사전학습 + 미세조정 시나리오 미검증

3. **Dense Prediction 태스크 미검증**
   - 객체 탐지, 의미론적 분할 성능 미평가
   - Isotropic 구조의 장점 활용 미검증

***

### 7. 향후 연구에 미치는 영향 및 최신 동향 (2023-2025)

#### 7.1 핵심 영향

**1) 큰 커널 컨볼루션의 재조명**

ConvMixer 이후 대규모 커널 연구가 활성화:

- **Scaling Up Your Kernels (CVPR 2022, 2024)**:[7][8][5]
  - UniRepLKNet으로 ImageNet 88.0% 달성
  - 최대 31×31 커널 사용 가능성 입증
  - 멀티모달 확장으로 범용성 확보[5]

- **LKCA(2024)**:[6]
  - Large Kernel 다시 정의
  - Attention으로 해석 가능
  - 데이터 제한 상황에서 우수성 입증[6]

**2) Patch 기반 아키텍처의 표준화**

- Patch Embedding이 단순한 "필요악"이 아닌 **핵심 설계 원칙**으로 인정
- Vision Transformer, MLP-Mixer, ConvMixer 등 patch-based 아키텍처 표준화[9][2][1]

**3) CNN 르네상스**

CNN이 ViT의 대안이 아닌 **동등한 경쟁자** 위치 확보:
- 적절한 설계(대규모 커널, Patch 기반)로 ViT 능가 가능
- 계산 효율성, 데이터 효율성 측면에서 장점[10][6][5]

#### 7.2 주요 파생 연구 (2023-2025)

| 연구 | 주요 기여 | 일반화 성능 개선 |
|------|---------|----------------|
| **UniRepLKNet**[5] | 멀티모달 대규모 커널 아키텍처 | ImageNet 88.0%, 여러 도메인 SOTA |
| **LKCA**[6] | 어텐션으로 대규모 커널 재해석 | 제한된 데이터에서 ViT 능가 |
| **ShiftwiseConv**[11] | 다중 경로 희소 의존성 | 3×3으로도 9×9 효과 달성 |
| **OAMixer**[2] | Object-aware 혼합 | 배경 강건성, SSL 성능 향상 |
| **ConvMixer-ECA**[12] | ECA 어텐션 모듈 추가 | 채널 의존성 명시적 학습 |
| **RepNeXt**[13] | 구조적 재파라미터화 | 모바일 환경에서 우수성 |

#### 7.3 최신 동향 (2024-2025)

**1) 효율성 개선**

- **RecConv(2024)**: 재귀적 컨볼루션으로 파라미터/FLOP 감소[14]
- **FPGA 가속**: 임의의 커널 크기 지원 하드웨어 구현[15]

**2) 멀티모달 확장**

- **UniRepLKNet** + 음성, 비디오, 시계열: 범용 모델 실현[5]
- **시계열 예측**: PatchMixer 등으로 Transformer 대체 시도[16]

**3) 도메인별 적용**

- **의료 이미징**: EEG 기반 ADHD 진단(ConvMixer-ECA)[12]
- **지진 예측**: 1D ConvMixer로 이온층 신호 분석[17]
- **초음파 이미지**: Underwater 이미지 복원[18]

#### 7.4 향후 연구 시 고려할 점

**1) 이론적 이해 심화**

- Patch 크기, 커널 크기, 깊이의 최적 조합에 대한 이론 부족
- 왜 큰 커널이 효과적인지에 대한 근본적 이해 필요
- 수용장과 성능의 정량적 관계 분석

**2) 효율성-성능 균형**

```
- 처리량 개선 (현재 병목):
  - 저수준 구현 최적화
  - 희소 커널 설계
  - 모바일/엣지 환경 최적화
```

**3) 대규모 사전학습 검증**

- ImageNet-21k 같은 대규모 데이터셋 평가 필요
- 사전학습 + 미세조정 시나리오 체계적 분석
- 전이학습 성능 비교

**4) Dense Prediction 확장**

- 객체 탐지, 인스턴스 분할, 의미론적 분할에서의 성능 평가
- Isotropic 구조의 잠재력 활용
- 고해상도 출력 요구 태스크에서의 우수성 검증

**5) 도메인 적응 및 강건성**

- Cross-domain 성능 평가
- 적대적 공격(Adversarial Attack)에 대한 강건성 분석
- Distribution shift 상황에서의 성능 평가

**6) 하이브리드 아키텍처**

- CNN + Attention의 최적 조합
- Local(CNN) + Global(Attention) 정보의 효율적 통합
- 컨퓨테이션 비용 대비 최고 성능

***

### 8. 결론

**"Patches Are All You Need?"** 논문은 간단하지만 강력한 ConvMixer 모델을 통해 **Patch 기반 표현이 Vision Transformer의 성능 향상의 핵심 요인**임을 입증했습니다.[1]

이 발견은:

1. **CNN의 부활**: 적절한 설계(대규모 커널, Patch 기반)로 CNN이 Transformer와 경쟁 가능함 증명
2. **아키텍처 재평가**: 새로운 연산(Self-Attention, MLP)이 아닌 **기본 원칙(Patch representation)**의 중요성 강조
3. **다방향 연구 촉발**: UniRepLKNet, LKCA, ShiftwiseConv 등 대규모 커널 연구 활성화

**일반화 성능 측면에서** ConvMixer는:
- 제한된 데이터(CIFAR-10)에서 뛰어난 데이터 효율성
- 동일 파라미터로 ViT/ResNet 능가
- 최신 개선(UniRepLKNet)으로 ImageNet 88% 달성

그러나 **실제 배포** 측면에서는:
- 처리량 개선 필요 (현재 GPU 최적화 미흡)
- 대규모 사전학습 검증 부족
- Dense Prediction 태스크 성능 미평가

향후 연구는 **효율성-성능 균형**, **멀티모달 확장**, **이론적 이해 심화**에 중점을 두어야 하며, 이는 AI 비전 분야의 **기초 아키텍처 설계 원칙**을 재정립하는 중요한 기여가 될 것입니다.[7][5][1]

***

## 참고문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b8cc960d-8abf-4ad0-a8b5-28fc64ce52e8/2201.09792v1.pdf)
[2](https://arxiv.org/pdf/2212.06595.pdf)
[3](https://jjhdata.tistory.com/41)
[4](https://jhtobigs.oopy.io/conv_mixer)
[5](https://openaccess.thecvf.com/content/CVPR2024/papers/Ding_UniRepLKNet_A_Universal_Perception_Large-Kernel_ConvNet_for_Audio_Video_Point_CVPR_2024_paper.pdf)
[6](https://www.semanticscholar.org/paper/ac901c1946fb2a9726c54869e82feee592f7742c)
[7](https://openaccess.thecvf.com/content/CVPR2022/papers/Ding_Scaling_Up_Your_Kernels_to_31x31_Revisiting_Large_Kernel_Design_CVPR_2022_paper.pdf)
[8](https://arxiv.org/html/2410.08049v1)
[9](http://arxiv.org/pdf/2409.06963.pdf)
[10](https://ieeexplore.ieee.org/document/10653307/)
[11](https://ieeexplore.ieee.org/document/11094588/)
[12](https://pmc.ncbi.nlm.nih.gov/articles/PMC11118831/)
[13](https://arxiv.org/html/2406.16004v1)
[14](https://arxiv.org/pdf/2412.19628.pdf)
[15](http://arxiv.org/pdf/2402.14307.pdf)
[16](http://arxiv.org/pdf/2310.00655.pdf)
[17](https://ieeexplore.ieee.org/document/11053753/)
[18](https://ieeexplore.ieee.org/document/10890803/)
[19](https://ieeexplore.ieee.org/document/10414094/)
[20](https://ieeexplore.ieee.org/document/10378631/)
[21](https://www.semanticscholar.org/paper/3425495ee3b6ead009f35aeb70edeac4e6eb2d10)
[22](https://mji.ui.ac.id/journal/index.php/mji/article/view/8312)
[23](https://aacrjournals.org/clincancerres/article/31/12_Supplement/P2-06-28/752256/Abstract-P2-06-28-AI-driven-feature-discovery)
[24](https://invergejournals.com/index.php/ijss/article/view/189)
[25](https://arxiv.org/pdf/2112.13692.pdf)
[26](https://arxiv.org/pdf/2201.09792.pdf)
[27](https://arxiv.org/abs/2106.09011)
[28](https://arxiv.org/abs/2104.12753)
[29](https://arxiv.org/pdf/2405.18240.pdf)
[30](https://blog.outta.ai/291)
[31](https://www.nature.com/articles/s41598-023-36724-x)
[32](https://arxiv.org/html/2404.00357v1)
[33](https://arxiv.org/abs/2201.09792)
[34](https://openreview.net/pdf?id=_Qaz9ZZSIHc)
[35](https://arxiv.org/html/2401.05738v1)
[36](https://research.google.com/pubs/archive/46649.pdf)
[37](https://link.springer.com/10.1007/s10845-024-02458-4)
[38](https://ieeexplore.ieee.org/document/10475506/)
[39](https://ieeexplore.ieee.org/document/10708984/)
[40](https://link.springer.com/10.1007/978-3-031-53302-0_6)
[41](https://ieeexplore.ieee.org/document/11022615/)
[42](https://ieeexplore.ieee.org/document/10650251/)
[43](https://arxiv.org/pdf/2410.08049.pdf)
[44](https://arxiv.org/pdf/2310.10563.pdf)
[45](https://arxiv.org/pdf/2211.05778.pdf)
[46](http://arxiv.org/pdf/2311.15599.pdf)
[47](http://arxiv.org/pdf/2408.09453.pdf)
[48](https://openaccess.thecvf.com/content_ICCV_2019/papers/Tsai_Domain_Adaptation_for_Structured_Output_via_Discriminative_Patch_Representations_ICCV_2019_paper.pdf)
[49](https://alinlab.kaist.ac.kr/resource/AI602_Lec11_Domain_transfer_and_adaptation.pdf)
[50](https://arxiv.org/html/2411.07118v1)
[51](https://arxiv.org/html/2404.11269v2)
[52](https://scoste.fr/posts/convmixer/)
[53](https://www.sciencedirect.com/science/article/pii/S1053811925002836)
