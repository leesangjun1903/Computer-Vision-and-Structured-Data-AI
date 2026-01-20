
# InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions
## 요약

InternImage는 **Deformable Convolution v3 (DCNv3)**을 핵심 연산자로 활용하여 Vision Transformer(ViT)의 성능 범위에 도달하는 첫 번째 대규모 CNN 기반 파운데이션 모델입니다. 1.08억 개의 파라미터와 4억 27백만 개의 학습 이미지를 활용한 InternImage-H는 COCO에서 65.4 mAP, ADE20K에서 62.9 mIoU를 달성하여 최고의 ViT를 능가합니다. 본 논문은 CNN 기반 모델도 적절한 설계를 통해 대규모 확장 가능함을 입증합니다.

***

## 1. 핵심 주장과 주요 기여

### 1.1 핵심 주장

인턴이미지는 다음 세 가지 핵심 가설에 기반합니다:

**첫째**, CNN의 성능 부족은 아키텍처의 근본적 한계가 아니라 **설계 선택의 차이**에서 비롯됩니다. 전통 CNN은 강한 귀납적 편향(inductive bias)으로 인해 장거리 의존성과 적응형 공간 집계 능력이 부족합니다.

**둘째**, 이러한 한계는 **Deformable Convolution**을 개선하면 극복할 수 있습니다. Deformable Convolution은 고정된 샘플링 그리드를 입력 데이터에 따라 동적으로 조정함으로써 ViT의 multi-head self-attention(MHSA)과 유사한 특성을 가집니다.

**셋째**, CNN은 **transformer 스타일의 블록 설계, 스택 규칙, 스케일링 전략**을 적용할 때 대규모 매개변수와 데이터에서 이득을 얻을 수 있습니다.

### 1.2 주요 기여

**기여 1: 첫 번째 대규모 CNN 파운데이션 모델**
- 10억 개 이상의 파라미터로 확장 가능한 첫 CNN 제시
- ImageNet-1K에서 89.6% top-1 정확도(대규모 사전학습 후)
- COCO 65.4 mAP(SwinV2-G 63.1보다 2.3포인트 향상)
- ADE20K 62.9 mIoU(이전 최고치 61.4 초과)

**기여 2: DCNv3를 중심으로 한 효율적 확장**
DCNv3는 세 가지 핵심 개선으로 비전 파운데이션 모델 요구사항을 충족합니다:
- 샘플링 오프셋(offsets)으로 동적 장거리 의존성 학습
- 변조 스칼라(modulation scalars)로 입력 조건부 적응형 집계
- 기존 large kernel CNN과 달리 3×3 커널로 최적화 용이

**기여 3: 포괄적 벤치마크 검증**
ImageNet, COCO, ADE20K, LVIS, Pascal VOC, Places, iNaturalist 등 다양한 벤치마크에서 일관된 우수 성능 입증.

***

## 2. 해결하는 문제

### 2.1 전통 CNN의 제한

**제한 1: 고정된 receptive field**
- 3×3 convolution을 깊게 쌓아도 effective receptive field(ERF)가 제한적
- Vision Transformer는 각 층에서 전역 receptive field를 가짐
- 다운스트림 작업(detection, segmentation)에서 성능 격차 발생

**제한 2: 정적 가중치와 강한 귀납적 편향**
- 정규 convolution은 고정된 가중치, 2D locality 등 강한 선입견 포함
- 대규모 데이터에서 다양한 패턴 학습 제약
- 입력 데이터에 따른 동적 적응 불가능

**제한 3: Large kernel CNN의 비효율**
- RepLKNet (31×31): 최적화 어려움, 보조 기법 필요
- SLaK (51×51): 계산 오버헤드, 성능 포화

### 2.2 제안된 해결책

InternImage는 세 가지 차원에서 이러한 문제를 해결합니다:

1. **연산자 수준**: DCNv3로 ViT 같은 장거리 의존성 + 적응형 집계 실현
2. **블록 수준**: Layer Normalization, FFN, GELU 등 transformer 구성요소 도입
3. **아키텍처 수준**: 계층형 구조, 체계적 스케일링 규칙 적용

***

## 3. 제안된 방법: DCNv3와 InternImage

### 3.1 DCNv3 수식 및 개선사항

**기본 DCNv2 공식:**
$$y_{p_0} = \sum_{k=1}^{K} w_k m_k \cdot x(p_0 + p_k + \Delta p_k) \quad \text{(1)}$$

여기서:
- $p_0$: 현재 픽셀 위치
- $K$: 샘플링 포인트 수
- $w_k$: $k$번째 샘플링 포인트의 선형 투영 가중치
- $m_k$: 변조 스칼라(sigmoid 정규화)
- $p_k$: 사전정의된 그리드 위치
- $\Delta p_k$: 학습 가능한 오프셋

**개선된 DCNv3 공식:**
$$y_{p_0} = \sum_{g=1}^{G} \sum_{k=1}^{K} w^g m_k^g \cdot x^g(p_0 + p_k + \Delta p_k^g) \quad \text{(2)}$$

#### 원리
##### 1단계: 채널 그룹화 $(\(x_{g}\))$ 
입력 픽셀 데이터 $\(x\)$ 는 여러 개의 **그룹 $(\(G\))$ **으로 나뉩니다.  
만약 입력이 128채널이고 $\(G=4\)$ 라면, 각 그룹 $\(x_{g}\)$ 는 32채널씩 담당합니다.  
이는 각 그룹이 이미지의 서로 다른 특징(예: 1그룹은 모양, 2그룹은 질감)을 독립적으로 추출하게 하여 효율성을 높입니다.

##### 2단계: 샘플링 위치 결정 $(\(p_{0}+p_{k}+\Delta p_{gk}\))$ 
전통적인 CNN은 정해진 격자에서만 값을 가져오지만, DCNv3는 위치를 이동시킵니다.  
- $\(p_{0}\)$ : 현재 계산 중인 기준 픽셀의 중심 좌표입니다.
- $\(p_{k}\)$ : 원래의 격자 구조에 따른 상대 위치입니다 (예: $\(3\times 3\)$ 커널의 경우 9개의 지점).
- $\(\Delta p_{gk}\)$ : 앞서 설명한 오프셋 생성기가 계산한 이동량입니다.
- 결과: 이 세 값을 더한 최종 좌표는 소수점 단위일 수 있으며, 해당 위치의 값을 **쌍선형 보간(Bilinear Interpolation)**을 통해 소스 픽셀에서 추출합니다.

##### 3단계: 중요도와 특징값의 결합 $(\(m_{gk}\cdot x_{g}\))$ 
샘플링된 위치의 값 $(\(x_{g}\))$ 에 해당 지점의 **중요도 $(\(m_{gk}\))$ **를 곱합니다. 
- $\(m_{gk}\)$ : 0~1 사이의 값으로, "이 지점의 정보가 얼마나 중요한가?"를 나타냅니다.
- 효과: 이를 통해 모델은 배경처럼 불필요한 정보는 억제하고, 객체의 형태를 나타내는 핵심 픽셀 정보만 강하게 반영합니다.

##### 4단계: 가중치 투영 및 최종 합산 $(\(w_{g}\cdot \dots \))$ 
마지막으로 추출된 정보에 학습된 가중치 $\(w_{g}\)$ 를 적용합니다. 
- $\(w_{g}\)$ : 그룹별 투영 가중치입니다. 샘플링된 값들을 선형 변환하여 의미 있는 특징으로 정제합니다. 각 그룹 내의 채널들은 동일한 투영 가중치 $\(w_{g}\)$ 를 공유합니다. 이는 Depthwise Separable Convolution과 유사한 방식으로, 파라미터 수를 획기적으로 줄이면서도 각 그룹이 서로 다른 특징(질감, 윤곽선 등)을 학습하게 합니다. 최종적으로 샘플링된 값들에 이 가중치를 곱하여 선형 변환(Linear Projection)을 수행합니다.
- $\(\sum \)$ : 모든 샘플링 지점 $(\(K\))$ 과 모든 채널 그룹 $(\(G\))$ 의 결과를 다 더하여 최종적인 출력 픽셀 $\(y(p_{0})\)$를 만듭니다. 

이 수식은 **"현재 픽셀 $(\(p_{0}\))$ 의 값을 결정하기 위해, 이미지 곳곳을 유연하게 돌아다니며 $(\(\Delta p\))$ 중요한 정보를 선별하고 $(\(m\))$ , 이를 그룹별로 정교하게 가공 $(\(w\))$ 하여 합친다"**는 과정을 수학적으로 정의한 것입니다.

#### 어떻게 오프셋과 중요도를 뽑아낼 수 있는 것인가? : 오차 역전파의 힘 (Gradient Descent)
이것이 핵심입니다. 모델은 처음에는 (예시 : 27개) 채널에 아무 의미 없는 랜덤한 숫자를 채웁니다.
- 초기 단계: 모델이 랜덤하게 좌표를 찍고(오프셋), 랜덤하게 중요도를 곱합니다. 결과는 엉망이고 Loss는 매우 높습니다.
- 미분(Gradient): Loss를 줄이기 위해 역전파가 발생합니다. 이때 "좌표를 담당하는 부분(앞 18개)"의 가중치와 "중요도를 담당하는 부분(뒤 9개)"의 가중치가 각각 자신이 맡은 역할에서 어떻게 변해야 Loss가 줄어들지 계산됩니다.
- 최적화: 수만 번의 학습을 거치면, 앞쪽 18개 채널을 생성하는 뉴런들은 "정확도를 높이기 위해 최적의 샘플링 위치를 찾아내는 전문가"로 진화하게 됩니다.

**세 가지 핵심 개선:**

**개선 1: 컨볼루션 뉴런 간 가중치 공유**

문제: 원본 DCNv2는 각 샘플링 포인트마다 독립적인 투영 가중치 유지 → 파라미터 선형 증가

해결: Separable convolution 아이디어 차용
$$w^g \in \mathbb{R}^{C/G \times C}$$ (group 차원)
$$m_k^g \in \mathbb{R}$$ (location-aware modulation)

효과: InternImage-H에서 unshared weights 대비 **파라미터 42배, GPU 메모리 84.2배 감소** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e5c22a26-c4c8-442d-ad7c-c65b49ce8ce9/2211.05778v4.pdf)

**개선 2: 다중 그룹 메커니즘 도입**

구조:
$$\text{그룹별 샘플링 오프셋}: p_k^g, \quad \text{그룹별 변조}: m_k^g$$

장점:
- 각 그룹이 서로 다른 표현 부분공간에서 독립적 학습
- Multi-head self-attention(MHSA)의 다중 헤드처럼 다양한 공간 집계 패턴 학습
- ImageNet 1.2%p 향상, COCO 3.4%p 향상 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e5c22a26-c4c8-442d-ad7c-c65b49ce8ce9/2211.05778v4.pdf)

**개선 3: 변조 스칼라의 Softmax 정규화**

원본 방식의 문제:
$$m_k^{\text{sigmoid}} \in, \quad \sum_k m_k \in [0, K]$$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e5c22a26-c4c8-442d-ad7c-c65b49ce8ce9/2211.05778v4.pdf)
→ 불안정한 기울기, 대규모 학습에서 수렴 어려움

개선된 방식:
$$m_k^g = \text{softmax}(m_k^g) \quad \text{along dimension } K$$
$$\Rightarrow \sum_{k=1}^{K} m_k^g = 1 \text{ (안정적)}$$

효과: 다양한 스케일의 모델 학습 안정성 보장

### 3.2 InternImage 모델 구조

**계층형 구조:**
```
입력 → Stem (4배 다운샘플링) 
       → Stage 1 (4개 기본 블록)
       → Downsampling
       → Stage 2 (4개 기본 블록)
       → Downsampling
       → Stage 3 (18-32개 기본 블록)
       → Downsampling
       → Stage 4 (4-6개 기본 블록)
       → Classification/Detection/Segmentation Head
```

**기본 블록 설계:**
$$\text{Input} \xrightarrow{LN} \text{DCNv3} \xrightarrow{LN} \text{FFN(4D)} \xrightarrow{\text{skip}} \text{Output}$$

특징: Transformer 스타일의 post-normalization, GELU 활성화

### 3.3 스택 규칙 및 스케일링 전략

**스택 규칙 (검색공간 축소: 12 → 4 하이퍼파라미터)**

원래 12개 하이퍼파라미터: $\{C_1, C_2, C_3, C_4, G_1, G_2, G_3, G_4, L_1, L_2, L_3, L_4\}$

**4가지 제약 규칙:**
1. **채널 규칙**: $C_i = C_1 \times i$ (i = 2,3,4)
2. **그룹 규칙**: $G_i = C_i / 16$
3. **블록 규칙**: 패턴 AABA (A,A,B,A 패턴으로 블록 수 결정)
4. **깊이 제약**: 검색 공간을 약 30개 모델로 축소

결과: 원본 30M 모델(InternImage-T) 기반으로 스케일링

**스케일링 규칙** (깊이 D, 너비 C 두 차원):
$$D' = \alpha D, \quad C'_1 = \beta C_1$$

여기서 $\alpha = 1.09, \beta = 1.36, \alpha \beta^2 \approx 2.0$

이 설정으로 InternImage-T → InternImage-H까지 일관된 성능 향상 달성

***

## 4. 성능 향상 분석

### 4.1 이미지 분류 (ImageNet-1K)

| 모델 | 파라미터 | 사전학습 | 정확도 | 대비 개선 |
|------|--------|--------|-------|---------|
| ConvNeXt-T | 29M | IN-1K | 82.1% | -1.4%p |
| **InternImage-T** | **30M** | **IN-1K** | **83.5%** | **기준** |
| ConvNeXt-B | 88M | IN-1K | 83.8% | -1.1%p |
| **InternImage-B** | **97M** | **IN-1K** | **84.9%** | **+1.1%p** |
| Swin-L | 197M | IN-22K→1K | 87.3% | -0.4%p |
| **InternImage-L** | **223M** | **IN-22K→1K** | **87.7%** | **+0.4%p** |
| SwinV2-G | 3.00B | JFT-300M | 90.2% | -0.6%p |
| **InternImage-H** | **1.08B** | **427M 데이터** | **89.6%** | **-0.6%p** |

**해석:**
- 소규모 모델: ConvNeXt 대비 일관되게 1-2% 향상
- 대규모 모델: ViT와 경쟁 가능 (1B 파라미터로 90% 근처)
- 파라미터 효율: SwinV2-G(3B) 대비 3배 적으면서 비슷한 성능

### 4.2 객체 감지 (COCO)

**Mask R-CNN with 1× schedule:**

| 모델 | 파라미터 | 박스 AP | 향상도 | 마스크 AP | 향상도 |
|------|--------|--------|-------|---------|-------|
| Swin-T | 48M | 42.7% | - | 39.3% | - |
| **InternImage-T** | **49M** | **47.2%** | **+4.5%p** | **42.5%** | **+3.2%p** |
| ConvNeXt-B | 108M | 47.0% | - | 42.7% | - |
| **InternImage-B** | **115M** | **48.8%** | **+1.8%p** | **44.0%** | **+1.3%p** |

**DINO 검출기 (고급 설정):**
- **InternImage-H (2.18B)**: **65.4 mAP** (test-dev)
- FD-SwinV2-G (3.0B): 64.2 mAP
- **향상도: +1.2 mAP (27배 적은 파라미터)**

### 4.3 의미론적 분할 (ADE20K)

| 모델 | 파라미터 | 입력 크기 | mIoU (MS) |
|------|--------|---------|----------|
| ConvNeXt-XL | 391M | 640² | 54.0% |
| **InternImage-XL** | **368M** | **640²** | **55.3%** |
| SwinV2-G | 3.00B | 896² | 59.9% |
| **InternImage-H** | **1.12B** | **896²** | **60.3%** |
| **InternImage-H+Mask2Former** | **1.31B** | **896²** | **62.9%** |

**성과:** ImageNet 사전학습만으로 경쟁력 있는 성능. Mask2Former와 결합시 62.9% (BEiT-3 62.8% 능가)

***

## 5. 모델 일반화 성능 향상 가능성

### 5.1 Effective Receptive Field (ERF) 분석

**ERF 특성:** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e5c22a26-c4c8-442d-ad7c-c65b49ce8ce9/2211.05778v4.pdf)
- **초기 단계 (Stage 1-2)**: 상대적으로 작은 ERF (locality 강조)
- **후기 단계 (Stage 3-4)**: 전역 ERF로 확대

```
[학습 전]  InternImage는 오프셋 미학습 → 작은 3×3 영역만 반응
[학습 후]  동적 오프셋으로 각 단계마다 의도적 ERF 형성
           - 초기: 텍스처·경계 감지 (작은 ERF)
           - 후기: 객체 형태·구조 감지 (큰 ERF)
```

**ViT와의 차이:**
- ViT: 모든 층에서 전역 ERF (처음부터)
- InternImage: 계층적 ERF 진화 → 생물학적 시각 체계와 유사 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e5c22a26-c4c8-442d-ad7c-c65b49ce8ce9/2211.05778v4.pdf)

### 5.2 도메인 외 강건성

**ImageNet-R, ImageNet-Sketch, ImageNet-C 평가:**

InternImage는 ConvNeXt, Swin 대비 다음 변환에 우수합니다:

**Translation Invariance (이동 불변성):**
- 64픽셀 변환까지 일관성 유지
- 다른 모델: 약 16픽셀부터 급격한 하락
- 원인: 적응형 샘플링으로 이동에 강건

**Rotation Invariance (회전 불변성):**
- 10도 이상 회전에서 최고 성능
- 일관성 점수: 45도 회전시 90% (ConvNeXt 80%)
- 원인: 동적 오프셋으로 회전 기하학 적응

**Scaling Invariance (스케일 불변성):**
- 업스케일링 (1.0x → 3.0x): 최고 성능
- 다운스케일링: 모든 모델 취약 (예상된 결과)
- 성능 격차: 스케일 1.75x에서 mAP +15% 우위

### 5.3 데이터 규모에 따른 강건성

**다양한 데이터 양에서의 성능:** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e5c22a26-c4c8-442d-ad7c-c65b49ce8ce9/2211.05778v4.pdf)

| 데이터 비율 | ResNet-50 | ConvNeXt-T | Swin-T | **InternImage-T** |
|-----------|----------|-----------|--------|---------------|
| 1% | 12.2% | 8.4% | 실패 | **5.9%** |
| 10% | 57.5% | 52.6% | 12.1% | **56.0%** |
| 100% | 80.4% | 82.1% | 81.3% | **83.5%** |

**해석:**
- **소규모 데이터 (1%, 10%)**: CNN의 강한 귀납적 편향 유리
  - InternImage는 이 시점에서 낮은 성능 (ViT보다 weak)
  - 하지만 CNN 중에서 가장 우수 (ConvNeXt 능가)
  
- **대규모 데이터 (100%)**: 동적 적응이 우위
  - InternImage: 83.5% (ConvNeXt 82.1% 능가)
  - ViT와 경쟁 가능

**결론:** InternImage는 **큰 데이터셋과 함께 스케일할 때** 일반화 성능이 급격히 향상됩니다.

### 5.4 다운스트림 작업에서의 강건성

**아블레이션 연구:** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e5c22a26-c4c8-442d-ad7c-c65b49ce8ce9/2211.05778v4.pdf)

**Weight Sharing의 영향:**
- Unshared weights 모델: ImageNet 83.6%, COCO mAP 47.4%
- Shared weights (DCNv3): ImageNet 83.5%, COCO mAP 47.2%
- **효율성 이득**: 파라미터 42배 감소, GPU 메모리 84.2배 감소, 성능 유사

**Multi-group Spatial Aggregation의 영향:**
- Without groups: ImageNet 82.3%, COCO mAP 43.8%
- With groups (G=4-32): ImageNet 83.5%, COCO mAP 47.2%
- **향상도**: +1.2% ImageNet, +3.4% COCO
- **해석**: 다양한 표현 부분공간에서 서로 다른 공간 패턴 학습의 중요성

**Softmax Normalization의 영향:**
- Sigmoid (원본): ImageNet 65.7%, COCO mAP 38.7%
- Softmax (개선): ImageNet 83.5%, COCO mAP 47.2%
- **향상도**: +17.8% ImageNet, +8.5% COCO
- **해석**: 큰 규모 학습에서 안정성이 절대적 중요

***

## 6. 한계 및 비판적 분석

### 6.1 명시적 한계 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e5c22a26-c4c8-442d-ad7c-c65b49ce8ce9/2211.05778v4.pdf)

**지연시간 (Latency) 문제:**
- InternImage-B (224²): 775 imgs/s (ConvNeXt-B 881 imgs/s, 12% 느림)
- 고해상도 (800²): 54 imgs/s (ConvNeXt-B 58 imgs/s, 7% 느림)
- 원인: DCN의 동적 오프셋 계산 오버헤드

**해결 방안 제시:** "효율적 DCN 개발이 필요"하지만 구체 방법 미제시

### 6.2 암묵적 한계

**소규모 데이터에서의 약점:**
- ImageNet-1K만으로: ConvNeXt, Swin과 경쟁 불가
- 필요 조건: 대규모 사전학습 (ImageNet-22K 또는 427M 데이터)
- 제한점: 학술 환경이나 자원 제약 상황에서 활용성 감소

**구조적 복잡성:**
- 스택 규칙 4가지, 스케일링 규칙 2개 등 설계 복잡
- 재현성: 코드 공개되었으나, 하이퍼파라미터 튜닝 어려움

**메모리 효율:**
- 대규모 모델학습 시 GPU 메모리 요구량 높음
- Inference: 일괄 처리(batch)에서만 효율적, 단일 이미지 처리는 상대적으로 비효율

***

## 7. 2020년 이후 관련 최신 연구 비교 분석

### 7.1 대규모 Vision Transformer 연구 경향

| 시간 | 모델 | 주요 특징 | 규모 | 성능 |
|------|------|--------|------|------|
| **2020** | ViT | 순수 Transformer, 패치 기반 | 86M-303M | 79.9-88.6% |
| **2021** | Swin Transformer | 계층형, 이동 윈도우 | 29M-197M | 81.3-87.3% |
| **2021** | DeiT | 지식 증류 기반 | 5M-305M | 72.2-85.2% |
| **2022** | **ConvNeXt** | **Transformer 영감 CNN** | **29M-350M** | **82.1-87.8%** |
| **2022** | **RepLKNet** | **31×31 큰 커널** | **30M-335M** | **82.5-87.8%** |
| **2022** | Swin-V2 | Swin 개선, 3B 모델 | 27M-3.0B | 81.8-90.2% |
| **2022** | BEiT-3 | 멀티모달 사전학습 | 1.9B | 88.7% |
| **2022** | **InternImage** | **DCNv3 기반 CNN** | **30M-1.08B** | **83.5-89.6%** |
| **2023** | ViT-22B | 22B 대규모 모델 | 22.0B | 90.45% |
| **2023** | ResFormer | 다해상도 학습 | 86M-335M | 83.2-88.4% |
| **2024** | UniRepLKNet | 멀티모달 큰 커널 | - | - |

### 7.2 주요 기술 혁신 비교

**Large Kernel 패러다임 (2022년 시작):**

| 접근 | 모델 | 커널크기 | 최적화 방식 | 장점 | 단점 |
|------|------|--------|---------|------|------|
| Dense Kernel | RepLKNet | 31×31 | Re-parameterization | 명확한 ERF 확대 | 최적화 어려움 |
| Sparse Kernel | SLaK | 51×51 | Factorized sparse | 더 큰 receptive field | 복잡한 구현 |
| **Deformable Kernel** | **InternImage** | **3×3 dynamic** | **학습 가능 오프셋** | **효율성 + 적응성** | **지연시간** |
| Peripheral | PeLK | 101×101 | 주변부만 샘플 | 극도로 큰 커널 | 매우 새로운 방식 |

**평가:**
- Large kernel CNN은 병렬 발전 경로 (ViT와 다른 방향)
- InternImage의 deformable 접근: ViT 장점 유지하면서 CNN 효율성 확보
- 2024-2025: 더 효율적 커널 설계로 시프트 (PeLK, InceptionNeXt 등)

### 7.3 파운데이션 모델 확장 추세

**ViT 확장 경향:**
- 2021: ViT-G (2B)
- 2022: Swin-V2-G (3B)  
- 2022: BEiT-3 (1.9B, 멀티모달)
- 2023: ViT-22B (22B) → **포화 시작**

**CNN 확장 경향:**
- 2022: RepLKNet (335M)
- 2022: SLaK (sparse, 51×51)
- **2022: InternImage (1.08B) ← 첫 1B급 CNN**
- 2024: UniRepLKNet (멀티모달)

**관찰:** InternImage는 CNN 확장에서 milestone 역할. 이후 대부분 ViT 또는 하이브리드 모델로 진행.

### 7.4 일반화와 강건성 연구

**도메인 외(Out-of-Distribution) 강건성:**
- ViT-22B 연구: shape bias 87% (texture bias에서 전환)
- InternImage: 회전 불변성, 스케일 불변성 우수성 입증
- 최신 방향: 자동 증강, 적대적 학습과 결합

**적응형 아키텍처:**
- Deformable Attention (2022) vs InternImage DCNv3 비교
- InternImage가 더 효율적 (sparse sampling)
- 후속: Adaptive Patch (2024) - 패치 크기 동적 조정

***

## 8. 향후 연구에 미치는 영향 및 고려 사항

### 8.1 이론적 영향

**1. CNN vs ViT 패러다임 재평가**
- 기존 통념: 대규모에서 ViT가 절대 우위
- InternImage 입증: 설계 차이일 뿐, CNN도 가능
- 영향: 아키텍처 독립적인 **스케일 법칙(scaling law) 연구** 촉발

**2. 귀납적 편향 재해석**
- CNN의 "약점"으로 여겨진 국소성(locality)이 실제로는:
  - 초기 단계 특징 학습의 장점 (계층적 표현)
  - 인간 시각계와 생물학적 일치
- 결과: **계층형 구조의 중요성** 재조명

**3. 효율성과 성능의 트레이드오프**
- 기존: 큰 receptive field = 더 큰 모델
- InternImage: 3×3 동적 샘플링 = 실질적 큰 receptive field + 작은 모델
- 영향: **효율적 표현 능력(expressive efficiency)** 개념 부상

### 8.2 실무적 영향

**1. 프로덕션 배포 전략**
- InternImage-B (97M, 84.9%): ResNet-50 수준 규모, ImageNet-1K만으로 ConvNeXt 능가
- 기업 관점: ResNet 대체 가능성 높음
- 지연시간: 여전히 개선 필요하지만, 배치 처리 상황에서 타당

**2. 전이 학습 기반 설계**
- ImageNet-22K 또는 대규모 데이터 필요 (구현 난점)
- 결과: 기성 사전학습 모델 공급 중요성 증대
  - HuggingFace, TIMM 등 모델 hub에 InternImage 포함 (확대 추세)

**3. 멀티모달 확장 기초**
- DCNv3의 적응형 샘플링: 텍스트-이미지 정렬에도 활용 가능
- 후속 연구: CLIP, DALL-E 스타일 멀티모달 모델에 DCNv3 적용 시작

### 8.3 미해결 문제 및 향후 연구 방향

**문제 1: 효율성 병목**
- **현상**: Latency 12% 손실 (배치 처리 아닐 시 무시할 수 없음)
- **근본 원인**: 각 위치마다 오프셋 계산 필요
- **해결 가능 방향:**
  - Sparse offset computation (일부 위치만 학습)
  - Hardware-aware DCN (CUDA kernel 최적화)
  - Mobile deployment용 경량화

**문제 2: 작은 데이터셋에서의 성능**
- **현상**: 소규모 데이터는 여전히 ViT/ConvNeXt 이하
- **원인**: 적응형 샘플링이 과적합 유발 가능
- **해결 가능 방향:**
  - 정규화 기법 강화
  - 지식 증류 (기존 모델에서)
  - 데이터 증강 (자동 증강) 결합

**문제 3: 이론적 이해 부족**
- **현상**: DCNv3가 왜 효과적인지 수학적 증명 부재
- **영향**: 아키텍처 개선 방향 제한
- **해결 가능 방향:**
  - 스펙트럼 분석 (frequency domain 특성)
  - 신경망 이중 하강 이론과의 연결
  - 동적 커널과 최적화 동역학 분석

### 8.4 추천 연구 과제

**우선순위 1: 효율성 개선**
1. Sparse DCNv3: 중요 위치만 오프셋 학습 (예: 경계, 코너)
2. 계층별 희소화: 초기 층은 dense, 후기 층만 sparse
3. 벤치마크: 지연시간 < 5% 손실 목표

**우선순위 2: 강건성 심화**
1. 적대적 강건성 (adversarial robustness) 평가
2. 분포 외 도메인 (medical imaging, satellite) 테스트
3. 모드 침입 공격(model inversion) 강건성

**우선순위 3: 멀티모달 확장**
1. DCNv3 기반 비전-언어 모델 (CLIP 스타일)
2. 동적 오프셋이 텍스트 정렬에 미치는 영향 분석
3. 크로스-모달 강건성 평가

**우선순위 4: 이론적 기초**
1. Effective Receptive Field와 일반화 오차의 관계
2. 동적 가중치의 표현 용량(representation capacity) 분석
3. CNN vs ViT의 최적 구조 원리 규명

***

## 9. 결론

InternImage는 **CNN 기반 파운데이션 모델이 대규모로 확장 가능함을 입증**한 획기적 연구입니다. Deformable Convolution v3을 중심으로 한 설계는 Vision Transformer의 핵심 장점(장거리 의존성, 적응형 집계)을 3×3 커널로 구현하면서도 **계층형 아키텍처의 귀납적 편향을 보유**합니다.

**주요 성과:**
- **성능**: COCO 65.4 mAP, ADE20K 62.9 mIoU로 ViT 능가
- **효율성**: 1B 파라미터로 3B Swin-V2-G 능가 (27배 적은 파라미터)
- **강건성**: 회전, 이동, 스케일 변환에 ViT/CNN보다 우수
- **확장성**: 30M~1.08B 범위에서 일관된 성능 향상

**제한점:**
- 지연시간 ~12% 증가 (배치 처리에서는 무시할 수 있으나, 실시간 응용에는 부담)
- 소규모 데이터셋에서는 여전히 ViT/ConvNeXt 이하
- 대규모 사전학습 필수 (ImageNet-1K만으로는 불충분)

**향후 영향:**
InternImage는 CNN과 Transformer의 false dichotomy를 제거했습니다. **효율적 설계, 적응형 연산자, 체계적 스케일링 규칙**이 대규모 비전 모델의 보편적 원리임을 보여주었으며, 이는 다음 세대 파운데이션 모델 설계의 핵심 교훈이 될 것입니다.

**2024-2026 시사점:**
- 현재 연구는 InternImage의 효율성 개선 (sparse DCN, hardware optimization)
- 멀티모달 응용 확대 (CLIP-style 모델)
- 이론적 기초 강화 (스케일 법칙, 강건성 원리 규명)
으로 발전하고 있습니다.

***

## 참고문헌

 InternImage 논문 원문 (2211.05778v4, 2023 CVPR Highlight) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e5c22a26-c4c8-442d-ad7c-c65b49ce8ce9/2211.05778v4.pdf)
 Scaling Vision Transformers (Zhai et al., 2022) [semanticscholar](https://www.semanticscholar.org/paper/96da196d6f8c947db03d13759f030642f8234abf)
 A ConvNet for the 2020s: ConvNeXt (Liu et al., 2022) [semanticscholar](https://www.semanticscholar.org/paper/186295f7c79e46c0e4e5f40e094267c09714043d)
 Revisiting Large Kernel Design in CNNs: RepLKNet (Ding et al., 2022) [semanticscholar](https://www.semanticscholar.org/paper/649b706ba282de4eb5a161137f80eb49ed84a0a8)
 Vision Transformer (Dosovitskiy et al., 2020) [semanticscholar](https://www.semanticscholar.org/paper/4d491b6fbe529a3986ef50cc34ede7c9ad88126c)
 Swin Transformer (Liu et al., 2021) [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9878563/)
 Scaling Vision Transformers to 22 Billion Parameters (Padlewski et al., 2023) [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9880094/)
 Deformable Convolutional Networks (Dai et al., 2017) [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9711309/)
