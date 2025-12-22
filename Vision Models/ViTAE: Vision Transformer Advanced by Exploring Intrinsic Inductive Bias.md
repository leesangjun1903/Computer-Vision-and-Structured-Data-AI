# ViTAE: Vision Transformer Advanced by Exploring Intrinsic Inductive Bias

### 1. 논문의 핵심 주장과 기여도

ViTAE는 Vision Transformer(ViT)의 근본적 문제를 정확히 지적한다. ViT는 이미지를 1D 토큰 시퀀스로 평탄화함으로써 CNN이 내재적으로 보유한 두 가지 귀납 편향(inductive bias)을 상실한다: **locality(국소성)**과 **scale-invariance(스케일 불변성)**. 결과적으로 ViT는 이를 대규모 데이터와 연장된 학습 일정을 통해 암묵적으로 학습해야 하며, 이는 계산 비용 증가와 데이터 의존성 심화를 초래한다.

본 논문의 세 가지 핵심 기여는 다음과 같다:

첫째, **이론적 통찰**: CNN의 두 가지 귀납 편향을 Vision Transformer에 명시적으로 도입할 수 있음을 증명. 이는 단순한 하이브리드 아키텍처 설계를 넘어 구조적 개선의 근본 원리를 제시한다.

둘째, **아키텍처 혁신**: Reduction Cell(RC)과 Normal Cell(NC)이라는 두 가지 기본 셀 설계. 이들은 다중 스케일 문맥과 국소-전역 특성을 모두 포괄하는 통합적 접근을 제시한다.

셋째, **성능의 실증적 우수성**: ImageNet 기준 4.8M 파라미터로 75.3% 정확도 달성(T2T-ViT-7 71.7% 대비 3.6% 향상), 20% 데이터로 학습 시 전체 데이터 학습 모델과 동등 성능, 100 에포크로 300 에포크 모델 능가.

***

### 2. 해결하고자 하는 문제와 제안 방법

#### 2.1 문제의 정식화

Vision Transformer의 한계:

$$\text{ViT}: x \in \mathbb{R}^{H \times W \times C} \rightarrow \text{flat patches} \rightarrow 1D \text{ token sequence} \rightarrow \text{loss of spatial structure}$$

이 과정에서:
- **국소 구조 손실**: 인접 픽셀 간의 상관성 무시
- **스케일 불변성 부재**: 다양한 크기의 객체를 구별하지 못함
- **데이터 의존성**: 대규모 사전학습 필수

#### 2.2 제안 방법의 수학적 표현

**Pyramid Reduction Module (PRM)**로 다중 스케일 문맥 추출:

$$f^{ms}_i = \text{Cat}([\text{Conv}_{ij}(f_i; s_{ij}, r_i) | s_{ij} \in S_i, r_i \in R])$$

여기서 $S_i = \{1,2,3,4\}$는 다양한 확장 비율(dilation rate), $R$은 공간 감소 비율을 나타낸다.

**병렬 구조로 국소성과 전역 의존성 동시 모델링**:

$$f^{lg}_i = f^g_i + \text{PCM}_i(f_i)$$

$$= \text{MHSA}_i(\text{Img2Seq}(f^{ms}_i)) + \text{PCM}_i(f_i)$$

이 설계는 단순히 convolution과 attention을 더하는 것이 아니라, 각 모듈이 서로 다른 수용장(receptive field)에서 보완적 역할을 수행하도록 한다:

- $f^g_i$: 다중 스케일 문맥 내 전역 의존성 포착
- $\text{PCM}_i(f_i)$: 3×3 convolution으로 국소 특성 추출

**Normal Cell의 토큰 처리**:

$$t^{nc} = \text{FFN}(t^{lg}) + t^{lg}, \quad t^{lg} = t^g + t^l$$

이는 skip connection과 결합되어 gradient flow를 원활히 하며, 국소(PCM)와 전역(MHSA) 특성이 독립적으로 학습되면서도 상호작용하는 구조를 만든다.

#### 2.3 모델 구조

ViTAE의 전체 파이프라인:

```
Input Image (H × W × C)
    ↓
[RC1: 7×7 conv, stride 4, dilation [1,2,3,4]] → H/4 × W/4
    ↓
[RC2: 3×3 conv, stride 2, dilation [1,2,3]] → H/8 × W/8
    ↓
[RC3: 3×3 conv, stride 2, dilation [1,2]] → H/16 × W/16
    ↓
Flatten & Positional Encoding
    ↓
[7× NC: parallel MHSA + PCM + FFN]
    ↓
Class Token → Linear Head → Output
```

RC와 NC의 차이:
- **RC**: PRM으로 토큰 생성, 공간 해상도 급격히 감소
- **NC**: PRM 제거(이미 multi-scale 정보 임베딩됨), 토큰 길이 유지, Group convolution으로 경량화

***

### 3. 성능 향상 분석

#### 3.1 ImageNet 성능 비교

| 모델 | 파라미터 | MACs | 정확도 | 대비 우수성 |
|------|---------|------|------|-----------|
| T2T-ViT-7 | 4.3M | 1.2G | 71.7% | baseline |
| **ViTAE-T** | **4.8M** | **1.5G** | **75.3%** | **+3.6% (T2T 대비)** |
| DeiT-T (distilled) | 5.7M | 2.6G | 74.5% | -0.8% (ViTAE 대비) |
| ResNet-18 | 11.7M | 3.6G | 70.3% | -5.0% |

ViTAE-T는 T2T-ViT-7 대비 5% 파라미터 증가로 **3.6% 정확도 향상**을 달성했으며, DeiT의 지식 증류 없이도 DeiT-T⚗를 능가한다.

| 모델 | 파라미터 | 정확도 | 특징 |
|------|---------|------|------|
| **ViTAE-S** | **23.6M** | **82.0%** | 파라미터 효율성 우수 |
| T2T-ViT-14 | 21.5M | 81.5% | -0.5% (ViTAE 대비) |
| ResNet-50 | 25.6M | 76.7% | -5.3% (ViTAE 대비) |
| Swin-T | 29.0M | 81.3% | -0.7% (ViTAE 대비, 더 많은 파라미터) |

#### 3.2 데이터 효율성 분석

ViTAE의 가장 놀라운 특성은 **데이터 효율성**이다:

**제한된 데이터셋에서의 성능**:

| 데이터 비율 | ViTAE-T | T2T-ViT-7 | 성능 격차 |
|----------|---------|---------|---------|
| 20% | 64.1% | 59.7% | +4.4% |
| 60% | 71.6% | 68.1% | +3.5% |
| 100% | 75.3% | 71.7% | +3.6% |

**핵심 통찰**: ViTAE는 제한된 데이터에서 더 강력한 성능을 보인다. 20% 데이터로 학습한 ViTAE(64.1%)가 100% 데이터로 학습한 T2T-ViT(68.7%)와 거의 동등하다.

#### 3.3 훈련 효율성 분석

| 에포크 | ViTAE-T | T2T-ViT-7 | 성능 격차 |
|------|---------|---------|---------|
| 100 | 74.2% | 68.7% | +5.5% |
| 200 | 75.0% (추정) | 70.8% | +4.2% |
| 300 | 75.3% | 71.7% | +3.6% |

**분석**: ViTAE는 초기 수렴 속도가 월등히 빠르다. 이는 귀납 편향이 early training phase에서 가장 효과적임을 의미한다.

***

### 4. 모델의 일반화 성능 향상 가능성

#### 4.1 다운스트림 태스크 성능

ViTAE의 일반화 능력을 검증하기 위해 여러 다운스트림 태스크를 평가했다:

**세분화 이미지 분류 (Fine-grained Classification)**:

| 데이터셋 | ViTAE-T | ViTAE-S | 최고 성능 모델 |
|---------|---------|---------|-------------|
| CIFAR-10 | 97.3% | 98.8% | DeiT-B: 99.1% |
| CIFAR-100 | 86.0% | 90.8% | DeiT-B: 90.8% |
| iNaturalist-19 | 92.6% | 94.2% | - |
| Cars | 73.3% | 76.0% | - |
| Flowers | 89.5% | 91.4% | - |
| Pets | 97.5% | 97.8% | - |

ViTAE-S는 CIFAR-100과 iNaturalist-19에서 SOTA 성능을 달성했으며, 특히 세분화된 객체 분류에서 우수한 성능을 보인다.

**객체 탐지 (Object Detection)**:

| 백본 | 프레임워크 | Box mAP | Mask mAP | 파라미터 |
|------|---------|---------|---------|--------|
| ResNet-50 | Mask RCNN | 38.2 | 34.7 | 44M |
| Swin-T | Mask RCNN | 43.7 | 39.8 | 48M |
| **ViTAE-S-Stage** | **Mask RCNN** | **44.6** | **40.2** | **37M** |

ViTAE는 11M 파라미터 감소로 0.9 mAP 향상을 달성했다.

**의미론적 분할 (Semantic Segmentation - ADE20K)**:

| 백본 | mIoU | mIoU (multi-scale+flip) | 파라미터 | 계산 효율성 |
|------|------|----------------------|--------|-----------|
| Swin-T | 44.5% | 45.8% | 60M | 기준 |
| **ViTAE-S-Stage** | **45.4%** | **47.8%** | **49M** | **+18% 효율** |

ViTAE는 11M 파라미터 감소(18% 경량화)로 1.9% mIoU 향상을 달성했다.

**자세 추정 (Human Pose Estimation - COCO)**:

| 백본 | mAP | mAR | 파라미터 |
|------|-----|-----|--------|
| ResNet-50 | 71.8% | 77.3% | 34M |
| **ViTAE-S-Stage** | **73.7%** | **79.0%** | **27M** |

**비디오 객체 분할 (Video Object Segmentation)**:

| 데이터셋 | ViTAE-T-Stage | ResNet-50 | 파라미터 효율 |
|---------|--------------|----------|------------|
| DAVIS-2016 | 89.8% | 89.3% | -20M (51% 경량화) |
| DAVIS-2017 | 82.5% | 81.8% | -20M (51% 경량화) |

#### 4.2 일반화 성능의 근본 원인 분석

**주의 거리(Attention Distance) 분석**:

Grad-CAM을 통한 시각화 결과, ViTAE는 다음과 같은 특성을 보인다:

1. **얕은 층**: PCM(병렬 convolution)이 국소성을 담당하므로, attention은 더 먼 거리에 초점 가능 (평균 주의 거리 증가)
2. **깊은 층**: 전역 의존성이 중요하므로 두 모델이 유사한 주의 거리 유지
3. **배경 노이즈 감소**: ViTAE는 배경을 덜 주의하고 객체에 집중
4. **다중 스케일 안정성**: 작은, 중간, 큰 객체에 모두 일관되게 초점

이는 **"divide-and-conquer"** 설계 철학이 성공적으로 구현되었음을 증명한다:
- Convolution: 국소성 담당 → 빠른 처리, 안정적 학습
- Attention: 전역 의존성 담당 → 유연한 모델링, 강력한 표현력

**특성 맵 분석(Attention Distance 그래프)**:

RC와 NC의 병렬 구조로 인해:
- **정보 흐름 이중화**: 국소 경로(PCM) + 전역 경로(MHSA)가 독립적으로 학습
- **상호 보완**: 각 경로의 부족을 다른 경로가 보충
- **효율적 수렴**: 두 가지 목표(국소성, 전역성)를 동시 최적화

***

### 5. 한계 및 미해결 문제

#### 5.1 구조적 한계

1. **대규모 사전학습 미실시**: 논문 저자들도 인정하듯, ImageNet-21K 또는 JFT-300M으로의 확장 실험이 없다. 이는 다음의 의문을 남긴다:
   - 대규모 데이터에서는 귀납 편향의 이점이 감소할 수 있는가?
   - ViT의 "극단적 스케일" 학습 성능과 비교했을 때 ViTAE의 상대적 위치는?

2. **제한된 귀납 편향 탐색**: 오직 scale-invariance와 locality만 도입. Viewpoint invariance, temporal consistency 등의 추가 편향 탐색 부재.

3. **위치 인코딩의 역할 모호**: 위치 인코딩 제거 후에도 성능이 거의 변하지 않는다 (75.3% → 75.3%). 이는:
   - RC와 NC의 convolution이 위치 정보를 자동 인코딩하는가?
   - Positional encoding과 convolution의 상호작용 메커니즘 규명 필요

#### 5.2 실험적 한계

1. **Ablation Study의 제한성**: Table 3에서 보이듯이:
   - RC와 NC를 개별적으로 평가하지만, 상호작용 분석 미흡
   - Batch Normalization의 역할이 과도함 (69.6% → 72.6%, 3% 향상)
   - 이는 normalization 선택의 중요성을 시사하나 심층 분석 부재

2. **비교 기준의 편차**: 다양한 모델이 다양한 augmentation strategy와 학습 schedule을 사용하여 직접 비교의 공정성 문제

#### 5.3 이론적 한계

1. **수렴 보장 없음**: 병렬 구조의 최적화 이론 분석 부재. 국소(PCM)와 전역(MHSA) 목적이 상충할 가능성?

2. **일반화 경계(Generalization Bound)**: PAC-Bayes 또는 Rademacher complexity를 통한 형식적 일반화 분석 미흡

3. **귀납 편향의 효과 정량화**: 데이터 복잡도 관점에서 필요한 샘플 수의 이론적 하한 도출 없음

***

### 6. 2020년 이후 관련 최신 연구 비교 분석

#### 6.1 주요 모델들의 귀납 편향 도입 방식 비교

| 모델 | 발표 | 주요 기여 | 귀납 편향 | 한계 |
|------|------|---------|---------|------|
| **ViT** | 2020 | 순수 attention 기반 아키텍처 | 없음 | 대규모 데이터 의존 |
| **DeiT** | 2020 | 지식 증류로 데이터 효율성 개선 | 간접적(teacher) | Teacher 모델 의존 |
| **T2T-ViT** | 2021 | Progressive tokenization | 국소성 | Multi-scale 미흡 |
| **Swin** | 2021 | 윈도우 기반 hierarchical attention | 국소성 + 계층성 | 초기 설계 제약 |
| **ViTAE** | 2021 | 병렬 multi-scale + 국소 모델링 | 국소성 + Scale불변성 | 대규모 데이터 미검증 |
| **ConViT** | 2021 | Soft positional bias | Learnable locality | 수렴 속도 |
| **LocalViT** | 2021 | 순차적 conv + attention | 강한 국소성 | Global context 감소 |

#### 6.2 차별성 심화 분석

**T2T-ViT와의 비교**:
- T2T-ViT: Overlapping patches로 국소 구조 포착, 하지만 **단일 스케일** 컨텍스트
- ViTAE: Dilation rates 로 **다중 스케일** 컨텍스트 동시 처리[1][2][3][4]
- **우수성**: 객체 크기 변화에 더 견고

**Swin Transformer와의 비교**:
- Swin: Hierarchical architecture, 윈도우 기반 attention 제약으로 계산 효율성
- ViTAE: Non-hierarchical, 병렬 구조로 전역 의존성 유지, 데이터 효율성 우수
- **트레이드오프**: Swin은 dense prediction task에 최적화, ViTAE는 classification에 최적화

**ConViT와의 비교**:
- ConViT: Soft inductive bias (학습 가능한 positional gating)
- ViTAE: Hard architecture change (convolution 모듈 직접 추가)
- **설계 철학**: ConViT는 점진적 이완, ViTAE는 구조적 상보성

#### 6.3 2020년 이후의 발전 궤적

**Phase 1 (2020)**: 기초 확립
- ViT: Transformer의 가능성 증명
- DeiT: 훈련 효율성 개선의 필요성 인식

**Phase 2 (2021)**: 귀납 편향 도입 경쟁
- T2T-ViT, Swin, ConViT, LocalViT, **ViTAE** 등의 변형 모델 등장
- **핵심 인식**: 순수 attention만으로는 부족, CNN의 귀납 편향 재평가

**Phase 3 (2022-현재)**: 이론화 및 최적화
- 동적 토큰 정규화(Dynamic Token Normalization)
- 공간 엔트로피 정규화(Spatial Entropy Regularization)
- 적응적 국소 편향 통합(Adaptive Local Bias)

#### 6.4 ViTAE의 위치

ViTAE는 Phase 2의 절정으로, 다음과 같이 평가된다:

**강점**:
1. **병렬 처리 원칙**: 국소성과 전역성을 완전히 분리, 각자 최적화 가능
2. **데이터 효율성**: 제한 데이터셋에서의 성능이 모든 concurrent work보다 우수
3. **파라미터 효율성**: 동급 성능에서 가장 적은 파라미터 사용
4. **확장성**: RC와 NC 구조의 일관성으로 다양한 크기 모델 생성 용이

**약점**:
1. **대규모 데이터 미검증**: ImageNet-21K, JFT-300M에서의 성능 미확인
2. **이론적 근거 부족**: 왜 병렬 구조가 순차적보다 우수한가에 대한 형식적 분석 없음
3. **Downstream task 특화 부족**: Swin처럼 dense prediction task에 최적화되지 않음

***

### 7. 향후 연구에 미치는 영향과 고려 사항

#### 7.1 직접적 영향

**아키텍처 설계 패러다임 변화**:
- CNN의 귀납 편향을 "수용"하되 "완전히 종속"되지 않는 설계의 가능성 제시
- 순차적 stacking 대신 병렬 처리의 장점을 실증적으로 증명

**데이터 효율성 개선의 새로운 방향**:
- Knowledge distillation이 아닌 구조적 개선을 통한 데이터 효율성 달성
- 이는 리소스 제약적 환경(edge computing, 소규모 기업 연구)에서의 적용성 증대

#### 7.2 후속 연구 방향

**1. 이론적 심화**:
```
병렬 CNN-Attention 구조의 최적화 이론
├─ Convergence Analysis: 국소+전역 최적화의 수렴 속도
├─ Generalization Bound: VC dimension, Rademacher complexity 분석
└─ Feature Learning Theory: 각 branch의 특성 공간 분석
```

**2. 아키텍처 확장**:
```
ViTAE의 다양한 변형
├─ Hierarchical ViTAE-Stage (이미 수행, Table 6-8)
├─ 추가 귀납 편향 통합
│  ├─ Temporal consistency (비디오 처리)
│  ├─ Viewpoint invariance (3D 인식)
│  └─ Frequency domain priors (robust learning)
├─ Multi-modal ViTAE (vision + language)
└─ Efficient ViTAE (pruning, quantization)
```

**3. 대규모 데이터 검증**:
```
ViTAE의 확장성 검증
├─ ImageNet-21K 사전학습
├─ JFT-300M (또는 동등 규모) 사전학습
├─ 각 단계에서의 downstream task 전이학습 분석
└─ 대규모 데이터에서의 귀납 편향의 영향 감소 검증
```

**4. 응용 분야 확대**:
```
ViTAE 기반 특수 도메인 모델
├─ 의료 영상 (CT, MRI, 병리학)
├─ 원격 탐사 (멀티스펙트럼 위성 영상)
├─ 자동 운전 (다시점, 다중 센서)
├─ 고에너지 물리 (입자 검출기)
└─ 생명과학 (미생물 이미지, 세포 분석)
```

#### 7.3 설계 시 고려 사항

**1. 귀납 편향의 선택**:
- ViTAE는 locality + scale-invariance에 특화
- 다른 도메인에서는 다른 편향이 필요할 수 있음
- 예: 의료 영상에서는 intensity invariance, 자동 운전에서는 viewpoint consistency가 중요

**2. 계산 비용 vs. 성능 트레이드오프**:
- PRM의 다양한 dilation rates가 계산 비용 증가
- Group convolution으로 부분 해결하나, dense prediction task에서의 성능-효율성 곡선 재분석 필요

**3. 정규화 기법의 상호작용**:
- Batch Normalization vs. Layer Normalization의 상호작용 (Table 10 분석 필요)
- Convolution + Attention의 특성 스케일 차이로 인한 정규화 문제

**4. 위치 인코딩의 역할 재평가**:
- ViTAE에서 positional encoding이 무시되는 현상 (Table 9)
- Convolution 자체가 위치 정보 인코딩?
- 이는 다른 아키텍처에서 위치 인코딩 설계에 영향

***

### 8. 결론

ViTAE는 Vision Transformer 발전의 중요한 이정표이다. 단순한 성능 향상을 넘어 **귀납 편향의 명시적 도입**이라는 원칙적 접근을 제시함으로써, 후속 연구들이 따를 설계 철학을 제공했다.

특히 **병렬 처리 원칙**—국소성과 전역 의존성을 구조적으로 분리하면서도 상호작용하도록 설계—은 단순한 하이브리드 아키텍처를 넘어 새로운 패러다임을 제시한다.

그러나 대규모 데이터셋에서의 성능, 이론적 근거, 다양한 도메인에서의 일반화 가능성은 여전히 미해결 문제로 남아있다. 이는 ViTAE를 기반으로 한 향후 연구의 풍부한 기회를 제공한다.

무엇보다 ViTAE의 **데이터 효율성**은 실무적 가치가 높다. 대규모 사전학습이 불가능한 소규모 조직이나 리소스 제약적 환경에서 Vision Transformer의 적용을 현실화했다는 점에서, 이 논문의 기여는 학술적 가치를 넘어 산업적 임팩트를 갖는다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/48995c2a-8cf6-4177-b28b-4d4ebc2fce6e/2106.03348v4.pdf)
[2](https://www.semanticscholar.org/paper/164e41a60120917d13fb69e183ee3c996b6c9414)
[3](https://www.semanticscholar.org/paper/da74a10824193be9d3889ce0d6ed4c6f8ee48b9e)
[4](https://www.semanticscholar.org/paper/cf5e6e3c50a798d87033e0e108e88b3647738bbe)
[5](https://www.semanticscholar.org/paper/7c8c6286a62a023f5d0d71fb315f9a0d4b9a2058)
[6](https://link.springer.com/10.1007/978-3-031-19803-8_9)
[7](https://iopscience.iop.org/article/10.1088/1742-5468/ac9830)
[8](https://ieeexplore.ieee.org/document/9879858/)
[9](https://ojs.aaai.org/index.php/AAAI/article/view/20103)
[10](https://ieeexplore.ieee.org/document/9742030/)
[11](https://www.semanticscholar.org/paper/c7650fe09c2b34e43646e785e09aefe290247e52)
[12](https://arxiv.org/pdf/2306.06635.pdf)
[13](https://arxiv.org/pdf/2211.13852.pdf)
[14](https://arxiv.org/ftp/arxiv/papers/2310/2310.00369.pdf)
[15](https://arxiv.org/pdf/2206.07662.pdf)
[16](http://arxiv.org/pdf/2210.01370.pdf)
[17](http://arxiv.org/pdf/2206.04636.pdf)
[18](https://arxiv.org/pdf/2305.08551.pdf)
[19](http://arxiv.org/pdf/2406.06072.pdf)
[20](https://www.sciencedirect.com/science/article/abs/pii/S0031320324002619)
[21](https://naokishibuya.github.io/blog/2022-11-04-swin-transformer-2021/)
[22](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Yuan_Tokens-to-Token_ViT_Training_ICCV_2021_supplemental.pdf)
[23](http://proceedings.mlr.press/v139/d-ascoli21a/d-ascoli21a.pdf)
[24](https://kikaben.com/swin-transformer-2021/)
[25](https://junha1125.github.io/blog/artificial-intelligence/2021-03-25-T2T/)
[26](https://dl.acm.org/doi/10.1007/s10015-022-00845-9)
[27](https://arxiv.org/pdf/2103.14030.pdf)
[28](https://arxiv.org/abs/2101.11986)
[29](https://en.wikipedia.org/wiki/Vision_transformer)
[30](https://arxiv.org/pdf/2112.03552.pdf)
[31](https://arxiv.org/pdf/2107.02174.pdf)
[32](https://openaccess.thecvf.com/content/ICCV2021/papers/Yuan_Tokens-to-Token_ViT_Training_Vision_Transformers_From_Scratch_on_ImageNet_ICCV_2021_paper.pdf)
[33](https://arxiv.org/pdf/2010.11929.pdf)
[34](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf)
[35](https://openaccess.thecvf.com/content/ICCV2021/html/Yuan_Tokens-to-Token_ViT_Training_Vision_Transformers_From_Scratch_on_ImageNet_ICCV_2021_paper.html)
[36](https://arxiv.org/pdf/2202.10108.pdf)
[37](https://arxiv.org/html/2507.18405v2)
[38](https://arxiv.org/pdf/2101.11986v1.pdf)
[39](https://arxiv.org/html/2310.00369v4)
