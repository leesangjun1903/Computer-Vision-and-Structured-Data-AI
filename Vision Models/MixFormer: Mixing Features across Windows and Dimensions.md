# MixFormer: Mixing Features across Windows and Dimensions

---

## 1. 핵심 주장 및 주요 기여 요약

**MixFormer**는 Local-window self-attention이 가진 두 가지 근본적 한계 — **(1) 제한된 수용 영역(limited receptive field)** 과 **(2) 채널 차원에서의 약한 모델링 능력(weak modeling capability)** — 을 동시에 해결하기 위해 제안된 효율적 범용 비전 트랜스포머(general-purpose vision transformer)이다.

### 주요 기여:
1. **병렬 설계(Parallel Design):** Local-window self-attention과 depth-wise convolution을 순차적(successive)이 아닌 병렬적(parallel)으로 결합하여, 윈도우 내부(intra-window)와 윈도우 간(cross-window) 관계를 동시에 모델링함으로써 수용 영역을 확장.
2. **양방향 상호작용(Bi-directional Interactions):** 채널(channel) 상호작용과 공간(spatial) 상호작용을 통해 두 병렬 브랜치 간에 상보적 단서를 제공, 채널 및 공간 차원 모두에서 모델링 능력을 향상.
3. **다중 태스크에서의 SOTA 성능:** ImageNet-1K 분류에서 EfficientNet과 경쟁적 성능을 달성하고, MS COCO, ADE20K, LVIS 등 5가지 dense prediction 태스크에서 Swin Transformer 대비 적은 계산량으로 유의미한 성능 향상을 입증.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

Local-window self-attention은 비겹침(non-overlapped) 윈도우 내에서만 self-attention을 수행하므로 두 가지 문제가 발생한다:

- **문제 1 — 제한된 수용 영역:** 윈도우 간 연결(cross-window connection)이 없어 receptive field가 윈도우 크기에 제한됨.
- **문제 2 — 약한 모델링 능력:** Local-window self-attention은 채널 차원에서 가중치를 공유(weight sharing on channel dimension)하고, depth-wise convolution은 공간 차원에서 가중치를 공유(weight sharing on spatial dimension)한다. 이러한 가중치 공유는 해당 차원에서의 모델링 능력을 제한한다.

각 연산의 가중치 공유 차원 및 FLOPs는 다음과 같다 (입력: $N \times C \times H \times W$):

| 연산 | 가중치 공유 차원 | FLOPs |
|------|-------------|-------|
| Global Self-Attention | Channel Dim | $2NCH^2W^2$ |
| W-Attention (Local-window) | Channel Dim | $2NCHWK^2$ |
| Convolution | Spatial Dim | $NC^2HWK^2$ |
| DwConv (Depth-wise) | Spatial Dim | $NCHWK^2$ |

### 2.2 제안하는 방법 (수식 포함)

#### (A) Mixing Block의 전체 수식

Mixing Block은 다음과 같이 정의된다:

$$\hat{X}^{l+1} = \text{MIX}(\text{LN}(X^l), \text{W-MSA}, \text{CONV}) + X^l $$

$$X^{l+1} = \text{FFN}(\text{LN}(\hat{X}^{l+1})) + \hat{X}^{l+1} $$

여기서:
- $\text{MIX}$: W-MSA(Window-based Multi-Head Self-Attention) 브랜치와 CONV(Depth-wise Convolution) 브랜치 간의 피처 혼합 함수
- $\text{LN}$: Layer Normalization
- $\text{FFN}$: Feed-Forward Network (두 개의 선형 레이어와 GELU 활성화로 구성된 MLP)

#### (B) 병렬 설계 (Parallel Design)

이전 연구들 (Swin [34], Shuffle Transformer [26] 등)은 local-window self-attention과 depth-wise convolution을 **순차적(successive)**으로 결합했으나, MixFormer는 이를 **병렬(parallel)**로 배치한다:

- **W-MSA 브랜치:** $7 \times 7$ 윈도우 크기의 local-window self-attention → 윈도우 내부 관계(intra-window relations) 모델링
- **DwConv 브랜치:** $3 \times 3$ 커널의 depth-wise convolution → 윈도우 간 관계(cross-window relations) 모델링

두 브랜치의 출력은 각각 다른 정규화 레이어(BN, LN)로 정규화된 후 **concatenation**되어 FFN으로 전달된다. FLOPs 비율에 따라 각 브랜치의 채널 수가 조정된다.

#### (C) 양방향 상호작용 (Bi-directional Interactions)

**채널 상호작용 (Channel Interaction):**

DwConv 브랜치에서 추출된 채널 정보를 W-MSA 브랜치의 Value에 적용한다. SE(Squeeze-and-Excitation) 레이어 [24] 구조를 따르지만 두 가지 차이점이 있다:
1. 입력이 동일 브랜치가 아닌 **다른 병렬 브랜치**에서 옴
2. 모듈 출력이 아닌 **Value에만** 채널 어텐션을 적용

구조: Global Average Pooling → $1 \times 1$ Conv → BN → GELU → $1 \times 1$ Conv → Sigmoid

**공간 상호작용 (Spatial Interaction):**

W-MSA 브랜치에서 추출된 공간 정보를 DwConv 브랜치에 적용한다.

구조: $1 \times 1$ Conv → BN → GELU → $1 \times 1$ Conv → Sigmoid

이를 통해 W-MSA의 $7 \times 7$ 윈도우에서 추출된 강한 공간 단서가 $3 \times 3$ DwConv 브랜치에 전달된다.

### 2.3 모델 구조

MixFormer는 4-stage 피라미드 구조를 채택한다:

| 구성 요소 | 설명 |
|---------|-----|
| **Convolution Stem** | 3개의 연속 컨볼루션으로 채널을 3 → $C$로 증가 |
| **Stage 1~4** | 각 스테이지에 Mixing Block을 $N_i$개 쌓음, downsampling rate: $\{4, 8, 16, 32\}$ |
| **Downsampling** | Stride-2 convolution으로 해상도 축소 |
| **Projection Layer** | 선형 레이어로 채널을 1280으로 증가 (특히 소형 모델에 효과적) |
| **Classification Head** | 분류용 헤드 |

**아키텍처 변형:**

| 모델 | $C$ | #Blocks | #Heads | FLOPs |
|------|-----|---------|--------|-------|
| MixFormer-B0 | 24 | [1,2,6,6] | [3,6,12,24] | 0.4G |
| MixFormer-B1 | 32 | [1,2,6,6] | [2,4,8,16] | 0.7G |
| MixFormer-B2 | 32 | [2,2,8,8] | [2,4,8,16] | 0.9G |
| MixFormer-B3 | 48 | [2,2,8,6] | [3,6,12,24] | 1.9G |
| MixFormer-B4 | 64 | [2,2,8,8] | [4,8,16,32] | 3.6G |
| MixFormer-B5 | 96 | [1,2,8,6] | [6,12,24,48] | 6.8G |
| MixFormer-B6 | 96 | [2,4,16,12] | [6,12,24,48] | 12.7G |

### 2.4 성능 향상

#### ImageNet-1K 분류:
- **MixFormer-B4:** 83.0% Top-1 accuracy (3.6G FLOPs) → Swin-T (81.3%, 4.5G) 대비 **+1.7%**, 20% 적은 FLOPs
- **MixFormer-B1:** 78.9% (0.7G) → EfficientNet-B1 (79.1%, 0.7G)과 경쟁적
- Swin-S (83.0%, 8.7G)와 동등한 성능을 **2.4배** 적은 FLOPs로 달성

#### MS COCO Object Detection (Mask R-CNN 1×):
- **MixFormer-B4:** $AP^b = 45.1$, $AP^m = 41.2$ → Swin-T 대비 **+2.9 box mAP**, **+2.1 mask mAP**
- **MixFormer-B1 (0.7G):** ResNet-50 (4.1G) 대비 **+2.3 box mAP**, **+2.9 mask mAP**

#### ADE20K Semantic Segmentation (UperNet):
- **MixFormer-B4:** 46.8 mIoU (ss), 48.0 mIoU (ms) → Swin-T 대비 **+2.2 mIoU**

#### Ablation 결과 (Table 7, MixFormer-B1 기준):
- 병렬 설계만: ImageNet +0.7%, COCO $AP^b$ +1.2, ADE20K +0.9
- 양방향 상호작용 추가: ImageNet 추가 +0.3%, COCO $AP^b$ 추가 +0.9, ADE20K 추가 +1.1
- **총 개선:** ImageNet +1.0%, COCO $AP^b$ +2.1, $AP^m$ +1.6, ADE20K +2.0

### 2.5 한계

1. **Global self-attention에 대한 적용 한계:** MixFormer는 window-based self-attention의 문제를 해결하도록 설계되었으므로, DeiT-Tiny [44]와 같은 global attention 모델에 Mixing Block을 적용하면 오히려 성능이 하락함 (72.2% → 71.3%)
2. **수동적 아키텍처 설계:** NAS(Network Architecture Search) 없이 수동으로 모델 구성을 설정하여, 최적의 아키텍처 탐색이 제한적

---

## 3. 모델의 일반화 성능 향상 가능성

MixFormer의 일반화 능력은 여러 측면에서 입증되었다:

### 3.1 다중 태스크에서의 일관된 성능 향상
MixFormer는 **5가지 dense prediction 태스크** (object detection, instance segmentation, semantic segmentation, keypoint detection, long-tail instance segmentation)에서 모두 Swin Transformer를 일관되게 초과 달성:

| 태스크 | 벤치마크 | Swin-T 대비 향상 |
|-------|---------|--------------|
| Object Detection | COCO (Mask R-CNN 1×) | +2.9 $AP^b$ |
| Instance Segmentation | COCO (Mask R-CNN 1×) | +2.1 $AP^m$ |
| Semantic Segmentation | ADE20K (UperNet) | +2.2 mIoU |
| Keypoint Detection | COCO | +1.1 $AP^{kp}$ |
| Long-tail Instance Seg. | LVIS 1.0 | +1.0 $AP^{mask}$ |

### 3.2 LVIS에서의 강건성
LVIS 1.0은 약 1000개의 롱테일 분포 카테고리를 가진 데이터셋으로, 백본이 학습한 표현의 **판별 능력(discriminative power)**에 의존한다. MixFormer-B4가 Swin-T 대비 +1.0 $AP^{mask}$를 달성한 것은 학습된 표현의 **강건성(robustness)**을 입증한다.

### 3.3 ConvNet에 대한 범용 적용
Mixing Block을 기존 ConvNet에 적용한 결과 (Table 12):
- **ResNet-50 + Mixing Block:** +1.6% Top-1 (79.0% → 80.6%), FLOPs 감소 (4.1G → 3.9G)
- **MobileNetV2 + Mixing Block:** +1.9% Top-1 (71.7% → 73.6%), FLOPs 동일 (0.3G)

이는 Mixing Block이 트랜스포머에 국한되지 않고 **ConvNet에도 범용적으로 적용 가능**함을 시사한다.

### 3.4 스케일러빌리티
MixFormer는 0.4G (B0, mobile-level)에서 12.7G (B6)까지 확장 가능하며, 각 규모에서 경쟁력 있는 성능을 보인다. 특히 **소형 모델에서의 스케일링**이 우수하다 — DeiT와 PVT가 모델 축소 시 급격한 성능 저하를 보이는 반면 ($-7.7\%$, $-4.7\%$), MixFormer는 안정적으로 스케일링된다.

### 3.5 일반화 성능 향상의 핵심 메커니즘
- **병렬 설계:** intra-window와 cross-window 관계의 동시 모델링으로 피처 표현의 다양성 증가
- **양방향 상호작용:** 채널과 공간 차원의 상보적 정보 교환으로 weight sharing에 의한 정보 손실 보완
- **Cascade Mask R-CNN** 등 더 강력한 검출기에서도 일관된 향상 (+1.1/1.2 box/mask mAP), 검출기에 independent한 향상

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 연구적 영향

1. **병렬 브랜치 설계 패러다임 확립:** 기존의 순차적 결합(self-attention → convolution)에서 벗어나 병렬 결합의 우수성을 입증하여, 이후 ConvNeXt V2, EfficientViT, FastViT 등의 hybrid 아키텍처 설계에 영향을 미침.

2. **Weight Sharing 관점의 분석 프레임워크:** Self-attention(채널 공유)과 DwConv(공간 공유)의 가중치 공유 특성을 상보적으로 활용하는 관점은 네트워크 설계의 이론적 근거를 제공.

3. **소형 모델 설계 가능성:** 0.4G FLOPs에서도 76.5% Top-1 정확도를 달성하여 모바일/엣지 비전 트랜스포머 설계의 가능성을 제시.

4. **범용 백본 아키텍처:** 분류뿐 아니라 5가지 dense prediction 태스크에서의 일관된 성능 향상으로, 효율적 범용 백본으로서의 비전 트랜스포머의 잠재력을 확인.

### 4.2 향후 연구 시 고려할 점

1. **Global Attention으로의 확장:** 현재 Mixing Block은 global self-attention에서 성능 하락을 보이므로, global attention 환경에서의 적응적 설계가 필요.

2. **NAS 기반 아키텍처 탐색:** 수동 설계의 한계를 극복하기 위해 Neural Architecture Search를 통한 최적 블록 배치 및 하이퍼파라미터 탐색.

3. **학습 전략 고도화:** Self-supervised pre-training (MAE, BEiT 등)과의 결합을 통한 추가적 일반화 성능 향상 가능성 탐색.

4. **추론 효율성:** 실제 하드웨어에서의 latency 최적화 — FLOPs는 낮지만 병렬 브랜치 구조가 메모리 접근 패턴에 미치는 영향 분석 필요.

5. **다양한 도메인 확장:** 의료 영상, 위성 영상, 동영상 이해 등 다양한 도메인에서의 일반화 성능 검증.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 모델 | 연도 | 핵심 아이디어 | ImageNet Top-1 | FLOPs | MixFormer와의 비교 |
|------|------|----------|---------------|-------|-----------------|
| **Swin Transformer** [34] | 2021 | Shifted window self-attention | 81.3% (T) | 4.5G | MixFormer-B4가 +1.7% 높고 FLOPs 20% 적음 |
| **Twins** [6] | 2021 | Local + global sub-sampled attention | 81.7% (S) | 2.9G | MixFormer-B3가 동등 (81.7%), Twins-S 대비 COCO/ADE20K에서 우수 |
| **Focal Transformer** [57] | 2021 | Focal self-attention (local-global) | 82.2% (T) | 4.9G | MixFormer-B4가 +0.8% 높고 FLOPs 27% 적음 |
| **Shuffle Transformer** [26] | 2021 | Spatial shuffle + NWC | 82.5% (T) | 4.6G | MixFormer-B4가 +0.5% 높고 FLOPs 22% 적음 |
| **PVTv2** [48] | 2021 | Improved pyramid vision transformer | - | - | MixFormer는 FFN에 DwConv 추가 시 PVTv2 스타일과 유사하나 큰 추가 이득 없음 |
| **ConvNeXt** (Liu et al., CVPR 2022) | 2022 | 순수 ConvNet의 modernization | 82.1% (T) | 4.5G | MixFormer-B4 (83.0%, 3.6G)가 더 높은 효율성 |
| **EfficientNet** [43] | 2019 | Compound scaling + NAS | 82.9% (B4) | 4.2G | MixFormer-B4가 경쟁적 (83.0%, 3.6G) |
| **HRFormer** [61] | 2021 | High-resolution representation for dense prediction | - | - | Keypoint detection에서 MixFormer-B4가 +0.8 $AP^{kp}$ 우수 |
| **CSWin Transformer** [10] | 2021 | Cross-shaped window self-attention | - | - | 윈도우 간 연결의 다른 접근법; MixFormer는 DwConv로 더 효율적으로 달성 |
| **Conformer** [39] | 2021 | Dual branch (CNN + Transformer) | - | - | 유사한 병렬 구조이나 동기 상이: Conformer는 local-global 결합, MixFormer는 weight sharing 보완 |

### 핵심 차별점 요약:

1. **Swin vs. MixFormer:** Swin은 shifted window로 cross-window 연결을 달성하나 순차적 설계. MixFormer는 DwConv 병렬 결합으로 shift 없이도 cross-window 연결을 달성하며, 양방향 상호작용으로 모델링 능력을 추가 강화.

2. **Focal/Shuffle vs. MixFormer:** 이들은 윈도우 확장/셔플링으로 receptive field를 넓히나, MixFormer는 더 단순한 $3 \times 3$ DwConv로 동등 이상의 효과를 달성하며 계산 비용이 낮음.

3. **ConvNeXt vs. MixFormer:** ConvNeXt는 순수 ConvNet modernization, MixFormer는 self-attention과 convolution의 상보적 결합이라는 근본적 설계 철학 차이. MixFormer는 self-attention의 dynamic weight 생성 능력을 보존하면서 convolution의 local 관계 모델링을 활용.

---

## 참고자료

1. **Qiang Chen, Qiman Wu, Jian Wang, et al.** "MixFormer: Mixing Features across Windows and Dimensions." *arXiv preprint arXiv:2204.02557v2*, 2022. (본 논문)
2. **Ze Liu, et al.** "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." *arXiv:2103.14030*, 2021. [34]
3. **Jie Hu, Li Shen, Gang Sun.** "Squeeze-and-Excitation Networks." *CVPR*, 2018. [24]
4. **Qi Han, et al.** "Demystifying Local Vision Transformer: Sparse Connectivity, Weight Sharing, and Dynamic Weight." *arXiv:2106.04263*, 2021. [17]
5. **Zilong Huang, et al.** "Shuffle Transformer: Rethinking Spatial Shuffle for Vision Transformer." *arXiv:2106.03650*, 2021. [26]
6. **Jianwei Yang, et al.** "Focal Self-attention for Local-Global Interactions in Vision Transformers." *arXiv:2107.00641*, 2021. [57]
7. **Xiangxiang Chu, et al.** "Twins: Revisiting the Design of Spatial Attention in Vision Transformers." *arXiv:2104.13840*, 2021. [6]
8. **Mingxing Tan, Quoc Le.** "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." *ICML*, 2019. [43]
9. **Hugo Touvron, et al.** "Training Data-Efficient Image Transformers & Distillation through Attention." *ICML*, 2021. [44]
10. **Wenhai Wang, et al.** "Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions." *arXiv:2102.12122*, 2021. [49]
11. **Yuhui Yuan, et al.** "HRFormer: High-Resolution Transformer for Dense Prediction." *NeurIPS*, 2021. [61]
12. **Zhiliang Peng, et al.** "Conformer: Local Features Coupling Global Representations for Visual Recognition." *ICCV*, 2021. [39]
13. **Zhuang Liu, et al.** "A ConvNet for the 2020s (ConvNeXt)." *CVPR*, 2022.
