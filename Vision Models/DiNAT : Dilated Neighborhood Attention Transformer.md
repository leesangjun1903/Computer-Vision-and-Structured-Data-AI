
# Dilated Neighborhood Attention Transformer 

## 1. 핵심 주장 및 주요 기여

**Dilated Neighborhood Attention Transformer (DiNAT)**는 Ali Hassani와 Humphrey Shi가 2022년 9월에 arXiv에 제출한 논문으로, 계층적 비전 트랜스포머의 핵심 한계를 해결하기 위해 설계되었습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

### 핵심 문제점 (Core Problem)
기존의 계층적 비전 트랜스포머(예: Swin Transformer, NAT)는 계산 효율성을 위해 국소 주의(local attention)를 사용하지만, 이는 두 가지 중요한 특성을 약화시킵니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

1. **장거리 상호의존성 모델링 능력 약화** - 전역 수용장(global receptive field)을 활용하지 못함
2. **수용장 성장의 선형성** - 네트워크 깊이에 따라 k+1 형태로만 증가 (k는 커널 크기)

### 주요 기여 (Major Contributions)

| 기여 영역 | 설명 |
|---------|------|
| **DiNA 메커니즘** | Neighborhood Attention을 확장하여 희소 전역 주의 구현 - 계산 추가 비용 없음 |
| **지수적 수용장 확장** | 적절한 희석 값(dilation)을 사용하면 k^ℓ 형태로 지수적 성장 |
| **DiNAT 아키텍처** | NA와 DiNA의 조합으로 구성된 새로운 계층적 트랜스포머 |
| **SOTA 성능** | 다중 비전 작업에서 기존 최고 성능 달성: COCO, ADE20K, Cityscapes |

***

## 2. 문제 해결 방법 및 수식

### 2.1 기본 자기주의(Self-Attention) 공식

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

여기서:
- $Q, K, V$ = 쿼리, 키, 값 선형 투영
- $d$ = 임베딩 차원
- 복잡도: $O(n^2d)$ (n = 토큰 수) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

### 2.2 Neighborhood Attention (NA) 공식

$$A^k_i = \left[Q_iK_{⊘_1i}^T + B_{i,⊘_1i}, Q_iK_{⊘_2i}^T + B_{i,⊘_2i}, \ldots, Q_iK_{⊘_ki}^T + B_{i,⊘_ki}\right]$$

$$V^k_i = \left[V_{⊘_1i}^T, V_{⊘_2i}^T, \ldots, V_{⊘_ki}^T\right]^T$$

$$\text{NA}^k_i = \text{softmax}\left(\frac{A^k_i}{\sqrt{d}}\right)V^k_i$$

여기서 $⊘_{ji}$는 i번째 토큰의 j번째 가장 가까운 이웃 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

**특징**: 
- 복잡도: $O(ndk)$ (k = 이웃 수, n >> k)
- 선형 시간 복잡도 달성

### 2.3 Dilated Neighborhood Attention (DiNA) 공식

$$⊘_{ji} = \{⊘_\tau: j \in \mathbb{Z}, j \bmod \tau \equiv i \bmod \tau\}$$

여기서 $\tau$ = 희석 매개변수 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

**희석된 주의 가중치**:

$$A^{k,\tau}_i = \left[Q_iK_{⊘^{(\tau)}_1i}^T + B_{i,⊘^{(\tau)}_1i}, \ldots, Q_iK_{⊘^{(\tau)}_ki}^T + B_{i,⊘^{(\tau)}_ki}\right]$$

$$\text{DiNA}^k_i = \text{softmax}\left(\frac{A^{k,\tau}_i}{\sqrt{d}}\right)V^{k,\tau}_i$$

**핵심 특성**:
- 희석 값 범위: $1 \le \tau \le \lfloor\frac{n}{k}\rfloor$
- 복잡도: $O(ndk)$ (여전히 선형)
- 수용장: $\tau=1$일 때 $k+1$, 최대 희석일 때 $k^\ell$ (지수적) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

### 2.4 수용장(Receptive Field) 비교

| 모델 구조 | 시간 복잡도 | 공간 복잡도 | 수용장 |
|---------|-----------|-----------|-------|
| Self-Attention (SA) | $O(n^2d)$ | $O(n^2)$ | $n$ |
| Convolution | $O(ndk)$ | $O(ndk)$ | $(k-1) \times \ell + 1$ |
| Window SA (Swin) | $O(ndk)$ | $O(ndk)$ | $k$ |
| NA (NAT) | $O(ndk)$ | $O(ndk)$ | $(k-1)\ell + 1$ |
| **NA + DiNA (DiNAT)** | $O(ndk)$ | $O(ndk)$ | **$k^\ell$ (지수)** |

여기서 $\ell$ = 모델 깊이, $k$ = 커널/이웃 크기 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

***

## 3. 모델 구조

### 3.1 전체 아키텍처 개요

DiNAT는 계층적 구조를 가진 4단계 네트워크로 설계되었습니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

```
Input Image (H × W)
    ↓
Initial Downsampler (2×3×3 Conv, stride=2×2) → H/4 × W/4
    ↓
Level 1 (N₁ blocks) + Downsampler (3×3 Conv, stride=2)
    ↓
Level 2 (N₂ blocks) + Downsampler
    ↓
Level 3 (N₃ blocks) + Downsampler
    ↓
Level 4 (N₄ blocks)
    ↓
Classifier/Detection Head
```

### 3.2 DiNAT 블록 구조

각 DiNAT 블록은 다음의 계층으로 구성: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

```
Input Feature Map (B × C × H × W)
    ↓
Layer Norm
    ↓
Neighborhood Attention (홀수 층) 또는 Dilated NA (짝수 층)
    ↓
Skip Connection (+ 입력)
    ↓
Layer Norm
    ↓
MLP (Multi-Layer Perceptron)
    ↓
Skip Connection (+ 주의 출력)
    ↓
Output
```

### 3.3 모델 변형(Variants)

| 변형 | 블록 구조 | 채널 | 깊이 | 파라미터 | FLOP |
|-----|---------|------|------|---------|------|
| DiNAT-Mini | 3,4,6,5 | 32 | 2/3배 | 20M | 2.7G |
| DiNAT-Tiny | 3,4,18,5 | 32 | 2/3배 | 28M | 4.3G |
| DiNAT-Small | 3,4,18,5 | 32 | 3/2배 | 51M | 7.8G |
| DiNAT-Base | 3,4,18,5 | 32 | 4/2배 | 90M | 13.7G |
| DiNAT-Large | 3,4,18,5 | 32 | 6/2배 | 200M | 30.6G |

모든 변형에서 기본 커널 크기는 7×7 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

### 3.4 희석 값 설정 전략

ImageNet-1K (224×224 해상도)에서의 설정: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

$$\text{희석값} =  \text{ (Level 1-4)}$$ [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10205440/)

하위 작업(더 큰 해상도)에서는 확장:
- COCO 객체 탐지 (800×800): [28, 14, 3-7, 3]
- ADE20K 의미 분할 (2048×512): [16, 8, 2-4, 2]

**점진적 희석(Gradual Dilation)**: DiNA 층에서 과 같이 점진적으로 증가하는 패턴 사용 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10641658/)

***

## 4. 성능 향상 및 실험 결과

### 4.1 이미지 분류 성능 (ImageNet-1K)

#### 224×224 해상도에서의 결과

| 모델 | Top-1 (%) | FLOPs (G) | 처리량 (imgs/s) | 메모리 (GB) |
|-----|----------|----------|----------------|----------|
| NAT-Tiny | 83.2 | 4.3 | 1,537 | 2.5 |
| **DiNAT-Tiny** | **82.7** | **4.3** | **1,500** | **2.5** |
| Swin-Small | 83.0 | 8.7 | 1,056 | 5.0 |
| ConvNeXt-Small | 83.1 | 8.7 | 1,549 | 3.5 |
| NAT-Small | 83.7 | 7.8 | 1,049 | 3.7 |
| **DiNAT-Small** | **83.8** | **7.8** | **1,058** | **3.7** |
| NAT-Base | 84.3 | 13.7 | 781 | 5.0 |
| **DiNAT-Base** | **84.4** | **13.7** | **764** | **5.0** |

**관찰**:
- 소형 모델(Tiny)에서는 약간의 성능 저하 (-0.5%)
- Small 이상에서는 일관된 개선 (+0.1% ~ +0.4%)
- ImageNet-22K 사전학습 후 미세조정 시 더 강력한 개선 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

#### 384×384 고해상도 결과

| 모델 | Top-1 (%) | FLOPs (G) | 처리량 (imgs/s) |
|-----|----------|----------|----------------|
| Swin-Large | 87.3 | 104.0 | 169 |
| ConvNeXt-Large | 87.5 | 101.1 | 221 |
| **DiNAT-Large (7×7)** | **87.4** | **89.7** | **161** |
| **DiNAT-Large (11×11)** | **87.5** | **92.4** | **110** |

- DiNAT-Large는 Swin-Large의 처리량 대비 약 2배 빠름 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

### 4.2 객체 탐지 및 인스턴스 분할 (MS-COCO)

#### Mask R-CNN 기준 결과

| 백본 | Box AP↑ | Mask AP↑ | 개선량 (vs NAT) |
|-----|--------|---------|---------------|
| NAT-Tiny | 47.7 | 42.6 | - |
| **DiNAT-Tiny** | **48.3** | **43.4** | **+0.6 AP, +0.8 mask** |
| NAT-Small | 48.4 | 43.2 | - |
| **DiNAT-Small** | **49.3** | **44.0** | **+0.9 AP, +0.8 mask** |
| NAT-Base | 52.3 | 45.1 | - |
| **DiNAT-Base** | **53.4** | **46.2** | **+1.1 AP, +1.1 mask** |
| NAT-Large (ImageNet-22K) | 53.7 | 46.4 | - |
| **DiNAT-Large** | **55.3** | **47.8** | **+1.6 AP, +1.4 mask** |

Swin-Large와 비교하면 각각 +1.6 box AP, +1.4 mask AP 개선 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

#### Cascade Mask R-CNN (더 강력한 기준)

- DiNAT-Large: 53.4 box AP, 58.2 mask AP (Swin-L 대비 +1.1 AP) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

### 4.3 의미 분할 성능 (ADE20K + UPerNet)

| 백본 | mIoU (%) | 처리량 (fps) | 개선량 |
|-----|---------|-----------|-------|
| NAT-Small | 49.5 | 17.9 | - |
| **DiNAT-Small** | **49.9** | **18.1** | **+0.4** |
| NAT-Base | 49.7 | 15.6 | - |
| **DiNAT-Base** | **50.4** | **15.4** | **+0.7** |
| Swin-Large | 53.5 | 8.5 | - |
| **DiNAT-Large** | **54.9** | **9.0** | **+1.4** |

ConvNeXt-Large 대비: +1.2 mIoU 개선 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

### 4.4 Mask2Former을 이용한 분할 작업 (SOTA 결과)

#### 인스턴스 분할 (MS-COCO)

| 작업 | DiNAT-L AP | Swin-L AP | 개선량 |
|-----|----------|----------|--------|
| Instance (MS-COCO) | 50.8 AP | 50.1 AP | +0.7 |
| Instance (ADE20K) | 35.4 AP | 34.9 AP | +0.5 |
| Instance (Cityscapes) | 45.1 AP | 43.7 AP | +1.4 |

#### 의미 분할

| 데이터셋 | DiNAT-L mIoU | Swin-L mIoU | 개선량 |
|---------|-------------|-----------|--------|
| ADE20K | 57.3 - 58.1 | 56.1 - 57.3 | +1.0 - 1.2 |
| Cityscapes | 83.9 - 84.5 | 83.3 - 84.3 | +0.2 - 0.6 |

#### 전경 분할 (Panoptic Segmentation)

| 벤치마크 | DiNAT-L PQ | Swin-L PQ | 개선량 |
|---------|-----------|----------|--------|
| MS-COCO | 58.5 | 57.8 | +0.7 |
| ADE20K | 49.4 | 48.1 | +1.3 |

**현재 SOTA 기록** (추가 데이터 없음):
- MS-COCO 전경 분할: 58.5 PQ
- ADE20K 전경 분할: 49.4 PQ
- Cityscapes 인스턴스 분할: 45.1 AP [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

### 4.5 절제 연구 (Ablation Studies)

#### 희석 값의 영향

ImageNet-Tiny에서의 성능:

| 희석 설정 | Top-1 (%) | COCO AP | COCO Mask | ADE20K mIoU |
|---------|----------|---------|-----------|------------|
|  [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf) (NA만) | 83.2 | 47.7 | 42.6 | 48.4 |
|  [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10873692/) (권장) | 82.7 | 48.0 | 42.9 | 48.5 |
|  [arxiv](https://arxiv.org/pdf/2502.13693.pdf) | - | 48.3 | 43.4 | 48.5 |
| 최대값 (동적) | 82.7 | 48.6 | 43.5 | 48.7 |
| 점진적 | - | 48.6 | 43.5 | 48.8 |

**핵심 발견**: 점진적 희석이 모든 작업에서 최고 성능 제공 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

#### 계층 구조 순서 비교

| 구조 | ImageNet | COCO Box | COCO Mask | ADE20K |
|-----|---------|---------|-----------|---------|
| NA-NA (NAT) | 83.2 | 47.7 | 42.6 | 48.4 |
| **NA-DiNA (DiNAT)** | **82.7** | **48.3** | **43.4** | **48.5** |
| DiNA-NA | 82.6 | 48.5 | 43.5 | 47.9 |
| DiNA-DiNA | 82.2 | 44.9 | 40.5 | 45.8 |

**중요한 발견**: NA-DiNA 순서(국소→전역)가 최적 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

***

## 5. 모델의 일반화 성능 향상

### 5.1 전이 학습(Transfer Learning) 성능

DiNAT의 가장 두드러진 강점은 **하위 작업(downstream tasks)에서의 일관된 개선**입니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

**ImageNet-1K 분류 vs 하위 작업 성능 개선**:

| 모델 | ImageNet 개선 | COCO 개선 | ADE20K 개선 |
|-----|-------------|---------|-----------|
| Tiny | -0.5% | +0.6% | +0.1% |
| Small | +0.1% | +0.9% | +0.4% |
| Base | +0.1% | +1.1% | +0.7% |
| Large | 0.0% | +1.6% | +1.4% |

**패턴**: 모델이 커질수록 하위 작업에서 더 큰 개선 달성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

### 5.2 이등방형 변형(Isotropic Variants) 실험

ViT와 직접 비교를 위해 고정 해상도에서의 성능 검증: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

| 모델 | FLOPs (G) | 처리량 (imgs/s) | Top-1 (%) |
|-----|----------|----------------|----------|
| ViT-Small | 4.6 | 3,086 | 81.2 |
| NAT-S iso. | 4.3 | 3,255 | 80.0 |
| **DiNAT-S iso.** | **4.3** | **3,160** | **80.8** |
| ViT-Base | 17.5 | 1,284 | 82.5 |
| NAT-B iso. | 16.9 | 1,350 | 81.6 |
| **DiNAT-B iso.** | **16.9** | **1,316** | **82.1** |

- DiNAT은 ViT 대비 약 -0.4% mAP 수준 (훨씬 효율적)
- 메모리 사용 ViT 대비 25% 감소 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

### 5.3 하이브리드 주의 메커니즘 분석

다양한 주의 패턴 조합의 성능:

| 계층 구조 | FLOPs (G) | Top-1 (%) | 관찰 |
|---------|----------|----------|-----|
| NA-NA | 4.32 | 80.0 | 순수 국소 주의 |
| DiNA-DiNA | 4.32 | 77.9 | 순수 전역 주의 (효율적이지만 약함) |
| NA-DiNA | 4.32 | 80.8 | **최적: 국소-전역 조합** |
| SA-SA (모든 ViT) | 4.58 | 81.2 | 전체 자기주의 기준 |
| SA-DiNA | 4.45 | 81.1 | 하이브리드 (좋지만 더 많은 FLOP) |

**결론**: NA와 DiNA의 조합이 효율성-정확도 트레이드오프에서 최적 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

### 5.4 다양한 해상도에서의 일반화

테스트 시간 희석 값 변경에 대한 민감도: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

| 학습 희석 | 테스트 희석 | ImageNet | COCO AP | ADE20K mIoU |
|---------|----------|---------|---------|-----------|
|  [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10873692/) |  [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10873692/) | 82.7 | 48.0 | 48.5 |
|  [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10873692/) |  [arxiv](https://arxiv.org/pdf/2502.13693.pdf) | 81.0 | 42.6 | 46.3 |
|  [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10873692/) | 최대값 | 78.2 | 43.0 | 41.5 |
|  [arxiv](https://arxiv.org/pdf/2502.13693.pdf) |  [arxiv](https://arxiv.org/pdf/2502.13693.pdf) | - | 48.3 | 48.5 |
|  [arxiv](https://arxiv.org/pdf/2502.13693.pdf) | 최대값 | - | 47.4 | 48.6 |

**중요한 발견**: 최적 성능을 위해서는 학습과 테스트 희석이 근접해야 함 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

***

## 6. 한계(Limitations) 및 미해결 과제

### 6.1 성능 한계

1. **작은 모델에서의 성능 저하**: 
   - Tiny 변형에서 ImageNet 분류 성능 -0.5% (83.2% → 82.7%)
   - 제한된 학습 신호에서 DiNA의 이점이 충분히 활용되지 못함 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

2. **메모리 최적화 부족**:
   - 현재 NATTEN 구현은 여전히 초기 단계 (CUDA 최적화 제한)
   - Tensor Core 미활용으로 인한 처리량 저하 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

3. **SOTA 달성 한계**:
   - 의미 분할에서 SeMask 등 특화 모델에 뒤짐
   - Cityscapes 의미 분할에서 2위 (84.5% vs 84.5% SOTA) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

### 6.2 설계 및 구현 한계

1. **희석 값의 입력 종속성**:
   - 희석 값 범위: $1 \le \tau \le \lfloor n/k \rfloor$ (입력 해상도에 따라 변함)
   - 서로 다른 해상도에 대해 수동 조정 필요 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

2. **기하학적 제약**:
   - NA와 DiNA는 홀수 크기 커널만 지원 (대칭성 유지)
   - Swin의 짝수 크기 커널(예: 12×12) 미지원 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

3. **메모리 접근 패턴 파괴**:
   - 희석 처리로 인해 캐시 효율성 저하
   - 실제 처리량이 이론적 FLOP보다 낮음 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

### 6.3 학습 관련 한계

1. **작은 데이터셋에서의 불안정성**:
   - 제한된 데이터로 학습 시 NA→DiNA 미세조정이 초기 성능 저하 초래 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

2. **하이퍼파라미터 민감성**:
   - 최적 희석 값이 작업 및 해상도에 따라 다름
   - 일반적 지침 부족 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

3. **추론 시간 다양성**:
   - 희석 값에 따라 처리량 변동 (희석값 크수록 느림)
   - DiNA 계층의 메모리 접근 패턴 불규칙성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

***

## 7. 최신 관련 연구 비교 분석 (2020-2025)

### 7.1 주요 관련 방법들과의 비교

| 방법 | 발표 | 주요 특징 | 복잡도 | 장점 | 단점 |
|-----|------|---------|-------|------|------|
| **Swin Transformer** | 2021 | Shifted window (WSA+SWSA) | $O(ndk)$ | 간단, 빠름 | 대칭성 깨짐, 선형 RF |
| **NAT** | 2022 | Sliding-window neighborhood | $O(ndk)$ | 대칭성 유지, 효율적 | 선형 수용장 |
| **DiNAT** | 2022 | NA + 희석 (이 논문) | $O(ndk)$ | **지수적 RF, 일반화↑** | 구현 복잡, 하이퍼파라미터 |
| **MaxViT** | 2022 | Block-local + dilated global | $O(ndk)$ | 다축 주의 | 복잡도 증가 |
| **BiFormer** | 2023 | Bi-level routing attention | $O(ndk)$ | 동적 희소성, 쿼리 인식 | 더 복잡한 구현 |
| **LongNet** | 2023 | 희석 주의 (수열용) | $O(n\log n)$ | 매우 긴 수열 지원 | 비전 특화 아님 |
| **DeBiFormer** | 2024 | Deformable bi-level routing | $O(ndk)$ | 의미론적 관련성↑ | 변형 점 학습 오버헤드 |

### 7.2 성능 수렴(Performance Convergence)

최신 SOTA 모델들과의 경쟁 상황:

**ImageNet-1K (224×224, ImageNet-22K 사전학습)**:
```
ConvNeXt-Large:    86.6%
DiNAT-Large:       86.6% ← 동등
Swin-Large:        86.3%
```

**COCO 객체 탐지 (Cascade Mask R-CNN)**:
```
ConvNeXt-Large:    54.8 AP
DiNAT-Large:       55.3 AP ← +0.5 개선
Swin-Large:        53.7 AP
```

**ADE20K 의미 분할 (UPerNet)**:
```
ConvNeXt-Large:    53.7 mIoU (53.2 단일 스케일)
DiNAT-Large:       54.9 mIoU (57.3 다중 스케일) ← 최고 성능
Swin-Large:        53.5 mIoU
```

### 7.3 최근 응용 분야 (2023-2025)

1. **의료 영상 분할**:
   - DiNAT-IR (2025): 이미지 복원용 확장 버전 [medcraveonline](http://medcraveonline.com/IRATJ/IRATJ-11-00301.pdf)
   - 뇌종양, 폴립, 망막 혈관 분할에 우수한 성능

2. **원격 감지**:
   - 변화 탐지 (BTNIFormer - DiNA 적용)
   - 작물 분류 (DWViT-ES)

3. **저수준 비전 작업**:
   - 이미지 초해상도 (HiT-SR)
   - 모서리 감지 (EdgeNAT)
   - 이미지 복원 (DiNAT-IR)

4. **음성 처리**:
   - 화자 검증 (PCF-NAT)

***

## 8. 이론적 통찰

### 8.1 수용장 성장 분석

$\ell$ 깊이의 네트워크에서:

$$\text{RF}_{\text{DiNAT}} = k^{\ell}$$

vs 

$$\text{RF}_{\text{Swin/NAT}} = (k-1)\ell + 1$$

**예시** (k=7, ℓ=4층):
- DiNAT: $7^4 = 2401$ (bounded by input n)
- Swin/NAT: $(7-1) \times 4 + 1 = 25$

이는 **지수적 vs 선형** 차이 → 장거리 의존성 학습에 결정적 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

### 8.2 희소성의 정규화 효과

제한된 희석이 자동으로 정규화를 도입:

- 불필요한 토큰과의 상호작용 억제
- 배경이나 반복적 패턴 무시
- 수렴 속도 개선 가능성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/4fdb4a14-84e3-4e43-b8f1-8a2586bf3971/2209.15001v3.pdf)

### 8.3 일반화 경계(Generalization Bound)

주의 메커니즘 미세조정에 대한 최근 이론적 분석: [ijcai](https://www.ijcai.org/proceedings/2025/0760.pdf)

- $W_v$ (값) 행렬이 가장 중요한 학습 성분
- DiNA의 희소성이 $W_v$ 학습을 더 효율적으로 가능 [openreview](https://openreview.net/pdf?id=P98KMCf60l)

***

## 9. 향후 연구 방향 및 고려사항

### 9.1 아키텍처 개선

1. **적응형 희석(Adaptive Dilation)**:
   - 입력 콘텐츠에 따른 동적 희석값 학습
   - 메커니즘: 경량 라우팅 네트워크

2. **멀티헤드 희석 다양화**:
   - 각 주의 헤드가 다른 희석 패턴 학습
   - BiFormer의 다양화 효과 결합 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10203555/)

3. **메모리 계층 최적화**:
   - 더 나은 CUDA 커널 구현 (Tensor Core 활용)
   - Flash Attention 통합으로 처리량 2-3배 향상 기대 [arxiv](http://arxiv.org/pdf/2403.04690.pdf)

### 9.2 데이터 및 학습 전략

1. **작은 모델/데이터셋 적응**:
   - Tiny 모델 성능 개선 (현재 -0.5%)
   - 지식 증류(Knowledge Distillation) 응용

2. **도메인 적응(Domain Adaptation)**:
   - 의료 영상 같은 특수 도메인에 특화된 사전학습
   - 현재 의료 영상에서의 일반화 성능 뛰어남 [arxiv](https://arxiv.org/pdf/2502.13693.pdf)

3. **멀티태스크 학습(Multi-task Learning)**:
   - 통합 백본으로 분류, 탐지, 분할 동시 학습
   - Mask2Former 성과의 확대

### 9.3 이론적 탐구

1. **희소 주의의 표현력 분석**:
   - DiNA가 정말로 전역 주의의 표현력을 유지하는가?
   - 근사 오차 경계 정식화

2. **수렴 특성 연구**:
   - NA→DiNA 교대 구조의 최적성 증명
   - Lottery Ticket Hypothesis 적용

3. **일반화 이론**:
   - 희소성이 과적합(overfitting)을 어떻게 억제하는가?
   - 작은 데이터 체제에서의 표본 복잡도 분석

### 9.4 응용 확대

1. **3D 비전 확장**:
   - 3D 객체 탐지, 인스턴스 분할
   - 비디오 이해 (시공간 DiNA)

2. **강화학습**:
   - 정책 학습 (Policy Gradient)
   - 상태 표현 학습

3. **멀티모달 학습**:
   - CLIP/BLIP와의 결합
   - 텍스트-이미지 정렬 개선

***

## 10. 결론 및 종합 평가

### 핵심 기여의 평가

DiNAT는 세 가지 핵심에서 기여합니다:

| 차원 | 기여 수준 | 근거 |
|-----|---------|------|
| **효율성** | 높음 | $O(ndk)$ 유지하며 SOTA 성능 달성 |
| **일반화** | 매우 높음 | 하위 작업에서 일관되고 큰 개선 (특히 Large 모델) |
| **이론적 근거** | 중간 | 지수적 RF는 명확하나, 일반화의 이론적 설명 부족 |
| **구현 및 재현성** | 우수 | 오픈소스 NATTEN, 체계적인 실험 |

### 주요 발견 요약

1. **지수적 수용장의 실제 가치**: ImageNet 분류에서는 미미하지만, 하위 작업의 밀집 예측에서 큰 이점

2. **하이브리드의 강력함**: NA(국소) + DiNA(전역)의 조합이 순수 희소 주의보다 훨씬 우수

3. **해상도 적응성**: 점진적 희석 전략으로 다양한 해상도에 자동 적응

4. **계산-성능 트레이드오프**: Swin 대비 2배 빠른 처리량에 우월한 성능

### 제한사항 인정

- 작은 모델에서의 성능 저하
- 메모리 최적화 구현의 미숙함
- 희석값 하이퍼파라미터의 입력 종속성

### 최종 평가

**DiNAT는 계층적 비전 트랜스포머 설계의 중요한 진전을 나타냅니다.** 특히 밀집 예측 작업(분할, 탐지)에서 효율성과 성능의 우수한 균형을 달성했습니다. 

**앞으로의 영향**:
- 희소 주의 설계의 새로운 패러다임 제시
- 의료, 원격 감지 등 특수 도메인에서 실질적 도입 시작
- 더 효율적한 Transformer 계열 모델의 기초가 될 가능성 [arxiv](https://arxiv.org/html/2507.17892v1)

***

## 참고문헌
<span style="display:none">[^1_100][^1_101][^1_102][^1_103][^1_104][^1_105][^1_106][^1_107][^1_108][^1_109][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_90][^1_91][^1_92][^1_93][^1_94][^1_95][^1_96][^1_97][^1_98][^1_99]</span>

<div align="center">⁂</div>

[^1_1]: 2209.15001v3.pdf

[^1_2]: https://ieeexplore.ieee.org/document/10205440/

[^1_3]: https://ieeexplore.ieee.org/document/10641658/

[^1_4]: https://ieeexplore.ieee.org/document/10873692/

[^1_5]: https://www.ewadirect.com/proceedings/ace/article/view/28255

[^1_6]: https://arxiv.org/pdf/2502.13693.pdf

[^1_7]: http://medcraveonline.com/IRATJ/IRATJ-11-00301.pdf

[^1_8]: https://www.ijcai.org/proceedings/2025/0760.pdf

[^1_9]: https://openreview.net/pdf?id=P98KMCf60l

[^1_10]: https://ieeexplore.ieee.org/document/10203555/

[^1_11]: http://arxiv.org/pdf/2403.04690.pdf

[^1_12]: https://arxiv.org/html/2507.17892v1

[^1_13]: https://arxiv.org/abs/2209.15001

[^1_14]: https://ieeexplore.ieee.org/document/10887572/

[^1_15]: https://ieeexplore.ieee.org/document/11253071/

[^1_16]: https://www.mdpi.com/2073-8994/17/8/1250

[^1_17]: https://www.mdpi.com/2078-2489/15/7/414

[^1_18]: https://ieeexplore.ieee.org/document/10581692/

[^1_19]: http://arxiv.org/pdf/2204.07143.pdf

[^1_20]: https://www.frontiersin.org/articles/10.3389/fonc.2024.1389396/full

[^1_21]: https://arxiv.org/pdf/2109.06684.pdf

[^1_22]: https://www.mdpi.com/2072-4292/15/23/5459/pdf?version=1700662305

[^1_23]: https://www.aclweb.org/anthology/P19-1288.pdf

[^1_24]: https://blog.csdn.net/qq_41442511/article/details/124783277

[^1_25]: https://www.nature.com/articles/s41598-025-22649-0

[^1_26]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11047.pdf

[^1_27]: https://arxiv.org/abs/2204.07143v1

[^1_28]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10130448/

[^1_29]: https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_EfficientViT_Memory_Efficient_Vision_Transformer_With_Cascaded_Group_Attention_CVPR_2023_paper.pdf

[^1_30]: https://openaccess.thecvf.com/content/CVPR2023/papers/Hassani_Neighborhood_Attention_Transformer_CVPR_2023_paper.pdf

[^1_31]: https://arxiv.org/pdf/2103.14030.pdf

[^1_32]: https://arxiv.org/abs/2305.07027

[^1_33]: https://arxiv.org/abs/2204.07143

[^1_34]: https://kimjy99.github.io/논문리뷰/swin-transformer/

[^1_35]: https://kimjy99.github.io/논문리뷰/efficientvit/

[^1_36]: https://huggingface.co/docs/transformers/main/en/model_doc/nat

[^1_37]: https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper

[^1_38]: https://velog.io/@softwarerbfl/논문-리뷰-EfficientViT-Memory-Efficient-Vision-Transformer-With-Cascaded-Group-Attention

[^1_39]: https://arxiv.org/pdf/2209.15001.pdf

[^1_40]: https://arxiv.org/html/2501.06480v2

[^1_41]: https://openaccess.thecvf.com/content/ICCV2023W/RCV/papers/Zheng_Lightweight_Vision_Transformer_with_Spatial_and_Channel_Enhanced_Self-Attention_ICCVW_2023_paper.pdf

[^1_42]: https://arxiv.org/html/2106.03180v4

[^1_43]: https://openaccess.thecvf.com/content/CVPR2023/html/Liu_EfficientViT_Memory_Efficient_Vision_Transformer_With_Cascaded_Group_Attention_CVPR_2023_paper.html

[^1_44]: https://arxiv.org/abs/2204.05585

[^1_45]: https://arxiv.org/abs/2303.13755

[^1_46]: https://arxiv.org/html/2403.04690v2

[^1_47]: https://arxiv.org/html/2508.04422v1

[^1_48]: https://arxiv.org/abs/2309.02031

[^1_49]: https://link.springer.com/10.1007/s00371-024-03416-0

[^1_50]: https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13164/3017809/Vision-transformer-with-source-target-attention-from-a-dilated-convolutional/10.1117/12.3017809.full

[^1_51]: https://linkinghub.elsevier.com/retrieve/pii/S0952197623014124

[^1_52]: https://linkinghub.elsevier.com/retrieve/pii/S0957417424022863

[^1_53]: https://link.springer.com/10.1007/978-981-96-6594-5_21

[^1_54]: https://ieeexplore.ieee.org/document/10440355/

[^1_55]: https://ieeexplore.ieee.org/document/10867966/

[^1_56]: https://link.springer.com/10.1007/s11063-024-11533-z

[^1_57]: https://www.ijsce.org/portfolio-item/D364414040924/

[^1_58]: http://arxiv.org/pdf/2204.01697.pdf

[^1_59]: http://arxiv.org/pdf/2303.08810.pdf

[^1_60]: https://arxiv.org/pdf/2302.01791.pdf

[^1_61]: https://www.mdpi.com/1424-8220/23/7/3447/pdf?version=1680001445

[^1_62]: https://arxiv.org/pdf/2309.01430.pdf

[^1_63]: https://arxiv.org/pdf/2108.08224.pdf

[^1_64]: https://arxiv.org/pdf/2303.06908.pdf

[^1_65]: https://openaccess.thecvf.com/content/ACCV2024/papers/BaoLong_DeBiFormer_Vision_Transformer_with_Deformable_Agent_Bi-level_Routing_Attention_ACCV_2024_paper.pdf

[^1_66]: https://www.nature.com/articles/s41598-024-57784-7

[^1_67]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05709.pdf

[^1_68]: https://www.sciencedirect.com/science/article/abs/pii/S0952197623014124

[^1_69]: https://cs.brown.edu/media/filer_public/c2/72/c272a1f8-1186-4a85-8f97-cfe8a1a7278a/zhouzhiyuan_honors_thesis.pdf

[^1_70]: https://www.ijcai.org/proceedings/2023/0523.pdf

[^1_71]: https://www.sciencedirect.com/science/article/pii/S0950705126001152

[^1_72]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10607952/

[^1_73]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7615082/

[^1_74]: https://kimjy99.github.io/논문리뷰/dinat/

[^1_75]: https://openaccess.thecvf.com/content/CVPR2023/papers/Pan_Slide-Transformer_Hierarchical_Vision_Transformer_With_Local_Self-Attention_CVPR_2023_paper.pdf

[^1_76]: https://dl.acm.org/doi/abs/10.1007/978-981-96-6594-5_21

[^1_77]: https://arxiv.org/abs/2407.05878

[^1_78]: https://arxiv.org/pdf/2412.06590.pdf

[^1_79]: https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Attention_Bridging_Network_for_Knowledge_Transfer_ICCV_2019_paper.pdf

[^1_80]: https://arxiv.org/pdf/2106.11360.pdf

[^1_81]: https://arxiv.org/pdf/2307.02486.pdf

[^1_82]: https://arxiv.org/html/2512.22252v1

[^1_83]: https://arxiv.org/html/2508.02806v1

[^1_84]: https://www.arxiv.org/pdf/2507.03026.pdf

[^1_85]: https://arxiv.org/abs/2510.18825

[^1_86]: https://arxiv.org/html/2503.12355v1

[^1_87]: https://arxiv.org/html/2511.06161v1

[^1_88]: https://arxiv.org/html/2507.06411v1

[^1_89]: https://pubmed.ncbi.nlm.nih.gov/40648456/

[^1_90]: https://arxiv.org/html/2407.05878v1

[^1_91]: https://link.springer.com/10.1007/978-3-031-76163-8

[^1_92]: https://www.mdpi.com/1424-8220/24/2/586

[^1_93]: https://arxiv.org/abs/2408.04579

[^1_94]: https://www.sciltp.com/journals/ijndi/2024/2/411

[^1_95]: https://www.frontiersin.org/articles/10.3389/fnins.2024.1401329/full

[^1_96]: https://scientiairanica.sharif.edu/article_23650.html

[^1_97]: https://ejournal.ptti.web.id/index.php/jahir/article/view/297

[^1_98]: https://dl.acm.org/doi/10.1145/3706890.3707006

[^1_99]: https://link.springer.com/10.1007/978-3-031-81854-7_4

[^1_100]: https://link.springer.com/10.1007/978-3-031-83274-1_17

[^1_101]: https://arxiv.org/pdf/2310.18656.pdf

[^1_102]: http://arxiv.org/pdf/2311.06031.pdf

[^1_103]: https://arxiv.org/pdf/2310.18642.pdf

[^1_104]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10528428/

[^1_105]: https://arxiv.org/pdf/2107.05274.pdf

[^1_106]: https://arxiv.org/pdf/2409.19483.pdf

[^1_107]: https://arxiv.org/pdf/2312.17183.pdf

[^1_108]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11637142/

[^1_109]: https://openaccess.thecvf.com/content/WACV2024/html/Shen_Med-DANet_V2_A_Flexible_Dynamic_Architecture_for_Efficient_Medical_Volumetric_WACV_2024_paper.html
