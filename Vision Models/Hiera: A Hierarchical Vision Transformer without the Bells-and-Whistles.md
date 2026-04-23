# Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

Hiera 논문의 핵심 주장은 다음과 같습니다:

> **"현대 계층적 Vision Transformer에 추가된 복잡한 Vision-specific 모듈(convolutions, shifted windows, relative position embeddings 등)은 MAE(Masked AutoEncoder)와 같은 강력한 사전학습(pretext task)을 활용하면 불필요하다."**

즉, 모델이 복잡한 귀납적 편향(inductive bias)을 아키텍처에 직접 하드코딩하는 대신, 강력한 사전학습을 통해 이를 **학습**할 수 있다는 것입니다.

### 주요 기여

| 기여 항목 | 설명 |
|-----------|------|
| **Hiera 아키텍처 설계** | 불필요한 모듈을 제거한 순수 계층적 ViT |
| **Mask Unit Attention** | MAE와 호환되는 로컬 어텐션 메커니즘 |
| **Sparse MAE 사전학습 호환** | 계층적 모델에서 희소(sparse) MAE 적용 방법 |
| **속도와 정확도의 동시 향상** | 이미지/비디오 태스크에서 SotA 달성 |
| **다중 태스크 일반화** | 이미지 분류, 비디오 분류, 객체 탐지, 분할 등 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

#### 문제 1: 계층적 ViT의 복잡성과 속도 저하

Swin Transformer, MViTv2 등 최신 계층적 ViT들은 높은 정확도를 달성하기 위해 다양한 Vision-specific 모듈을 추가했습니다:

- **Shifted Windows** (Swin): 윈도우 간 정보 교환을 위한 복잡한 연산
- **Decomposed Relative Position Embeddings** (MViTv2): 위치 정보 인코딩의 복잡화
- **Pooling Attention with Convolutions** (MViTv2): 공간적 편향 추가를 위한 합성곱
- **Cross-shaped Windows** (CSWin): 더 넓은 수용 영역을 위한 특수 윈도우

이러한 모듈들은 FLOP 수는 줄여보이지만, **실제 추론 속도(throughput)는 오히려 느려지는** 역효과를 낳았습니다.

#### 문제 2: 계층적 모델과 MAE의 비호환성

MAE는 masked token을 **삭제**(sparse)하여 학습 효율을 높입니다. 그러나 기존 계층적 모델들은 2D 그리드 구조에 의존하므로:

$$\text{문제: MAE의 sparse 마스킹} \Rightarrow \text{2D 그리드 파괴} \Rightarrow \text{합성곱/윈도우 어텐션 오류}$$

기존 해결책들의 단점:
- **MaskFeat/SimMIM**: `[mask]` 토큰으로 대체 → 비가시 토큰에도 연산 낭비, 매우 느림
- **UM-MAE**: 특수 마스킹 전략 → 정확도 손실
- **MCMAE**: Masked Convolution 사용 → 효율성 저하

---

### 2.2 제안하는 방법 (수식 포함)

#### 핵심 전략: MViTv2에서 Hiera로의 단계적 단순화

**Step 1: Mask Unit 개념 도입**

기존 MAE는 $16 \times 16$ 픽셀 패치를 마스킹하지만, 계층적 모델은 $4 \times 4$ 픽셀 토큰을 사용합니다. 이를 해결하기 위해 **Mask Unit**을 도입합니다:

$$\text{Mask Unit Size} = 32 \times 32 \text{ pixels}$$

$$\text{Stage } s\text{에서 Mask Unit당 토큰 수} = \left(\frac{32}{4 \cdot 2^{s-1}}\right)^2$$

구체적으로:
- Stage 1: $8^2 = 64$ tokens/unit
- Stage 2: $4^2 = 16$ tokens/unit  
- Stage 3: $2^2 = 4$ tokens/unit
- Stage 4: $1^2 = 1$ token/unit

**Step 2: Absolute Position Embedding 사용**

MViTv2의 Decomposed Relative Position Embedding을 단순한 절대 위치 임베딩으로 교체:

$$\mathbf{z}_0 = \mathbf{x}_{\text{patch}} + \mathbf{p}_{\text{abs}}$$

여기서 $\mathbf{p}_{\text{abs}} \in \mathbb{R}^{N \times D}$는 학습 가능한 절대 위치 임베딩입니다. 논문 Table 1(a)에 따르면 이 변경만으로도 정확도 유지(85.6%) 및 속도 향상(219.8 → 253.3 im/s)을 달성합니다.

**Step 3: Convolution 제거 및 MaxPool로 대체**

pooling attention에서 $3 \times 3$ 합성곱을 MaxPool로 교체한 뒤, stride=1인 불필요한 pooling 레이어 삭제:

$$\text{Q Pooling}: \mathbf{Q}' = \text{MaxPool}_{k=s}(\mathbf{Q}), \quad k = s \text{ (kernel size = stride)}$$

커널 크기를 stride와 동일하게 설정하면:
- **패딩이 불필요**해짐
- **Mask Unit 간 정보 누출 방지**
- sparse MAE 사전학습과 완전히 호환

**Step 4: Q-Attention Residual 제거**

MViTv2의 residual pooling connection:

$$\text{MViTv2: } \text{Attention}(\mathbf{Q}', \mathbf{K}, \mathbf{V}) + \mathbf{Q}'$$

Hiera에서는 이를 완전히 제거:

$$\text{Hiera: } \text{Attention}(\mathbf{Q}', \mathbf{K}, \mathbf{V})$$

**Step 5: Mask Unit Attention (핵심 기여)**

KV Pooling Attention을 **Mask Unit Attention**으로 교체:

기존 Pooling Attention (MViTv2):
$$\text{Attn}(\mathbf{Q}, \mathbf{K}', \mathbf{V}'), \quad \mathbf{K}' = \text{Pool}(\mathbf{K}), \mathbf{V}' = \text{Pool}(\mathbf{V})$$

여기서 Pool은 전체 feature map에 대한 전역 풀링.

제안하는 Mask Unit Attention:
$$\text{MU-Attn}(\mathbf{Q}_m, \mathbf{K}_m, \mathbf{V}_m), \quad \forall m \in \{1, \ldots, M\}$$

여기서 $m$은 각 mask unit 인덱스, $M$은 전체 mask unit 수. 즉 **각 mask unit 내부에서만 로컬 어텐션**을 수행합니다.

Window Attention과의 차별점:

$$\text{Window Attn: 고정 크기 } w \times w \text{ 윈도우 (다운샘플 후 mask unit에 누출)}$$
$$\text{MU Attn: 현재 해상도의 mask unit 크기에 동적 적응}$$

**MAE 목적함수**:

$$\mathcal{L}_{\text{MAE}} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \left\| \hat{\mathbf{x}}_i - \mathbf{x}_i \right\|^2$$

여기서 $\mathcal{M}$은 마스킹된 mask unit의 집합, $\hat{\mathbf{x}}_i$는 재구성된 픽셀(또는 HOG 특징).

**Multi-Scale Decoder**:

Hiera의 계층적 구조를 활용하여 모든 stage의 표현을 디코더에 융합:

$$\mathbf{d} = \text{Decoder}\left(\text{Fuse}(\mathbf{f}_1, \mathbf{f}_2, \mathbf{f}_3, \mathbf{f}_4)\right)$$

여기서 $\mathbf{f}_s$는 stage $s$의 특징 맵. 논문 Table 3(a)에 따르면 이 multi-scale decoder는 이미지 +0.6%, 비디오 +1.7%의 성능 향상을 가져옵니다.

---

### 2.3 모델 구조

#### Hiera 아키텍처 개요

```
Input Image (224×224)
        ↓
[Patch Embedding: 4×4 stride, 4×4 pixel tokens]
        ↓
Stage 1: Mask Unit Attention (Local, 8×8 tokens/unit) + Global Attn
        ↓ [Q-Pooling: MaxPool 2×2, channels ×2]
Stage 2: Mask Unit Attention (Local, 4×4 tokens/unit)  
        ↓ [Q-Pooling: MaxPool 2×2, channels ×2]
Stage 3: Global Attention
        ↓ [Q-Pooling: MaxPool 2×2, channels ×2]
Stage 4: Global Attention
        ↓
Classification Head
```

#### Hiera 변형 모델 구성 (Table 2)

| 모델 | 채널 수 | 블록 수 | 헤드 수 | FLOPs | 파라미터 |
|------|---------|---------|---------|-------|----------|
| Hiera-T | [96-192-384-768] | [1-2-7-2] | [1-2-4-8] | 5G | 28M |
| Hiera-S | [96-192-384-768] | [1-2-11-2] | [1-2-4-8] | 6G | 35M |
| Hiera-B | [96-192-384-768] | [2-3-16-3] | [1-2-4-8] | 9G | 52M |
| Hiera-B+ | [112-224-448-896] | [2-3-16-3] | [2-4-8-16] | 13G | 70M |
| Hiera-L | [144-288-576-1152] | [2-6-36-4] | [2-4-8-16] | 40G | 214M |
| Hiera-H | [256-512-1024-2048] | [2-6-36-4] | [4-8-16-32] | 125G | 673M |

Stage 해상도: $[56^2, 28^2, 14^2, 7^2]$

#### Hiera Block 구조

각 Hiera Block은 표준 ViT Block과 동일:

$$\mathbf{z}' = \mathbf{z} + \text{MSA}(\text{LN}(\mathbf{z}))$$
$$\mathbf{z}'' = \mathbf{z}' + \text{FFN}(\text{LN}(\mathbf{z}'))$$

단, Stage 전환 시 Q-Pooling 블록:

$$\mathbf{z}_{\text{out}} = \text{Linear}(\text{MaxPool}_{k=s}(\mathbf{z}_{\text{in}}))$$

채널 수 두 배 증가: $D \rightarrow 2D$

#### 비디오 확장

비디오의 경우 Mask Unit을 시공간으로 확장:

$$\text{Video Mask Unit} = 2 \text{ frames} \times 32 \times 32 \text{ pixels}$$
$$= 1 \times 8 \times 8 \text{ tokens (Stage 1)}$$

이미지와 동일한 구현 공유 (mask unit 크기만 변경).

---

### 2.4 성능 향상

#### 단계별 단순화에 따른 성능 변화 (Hiera-L 기준)

| 단계 | 변경 사항 | 이미지 정확도 | 이미지 속도(im/s) | 비디오 정확도 | 비디오 속도(clip/s) |
|------|-----------|--------------|-----------------|--------------|-------------------|
| 기준 | MViTv2-L Supervised | 85.3 | 219.8 | 80.5 | 20.5 |
| (a) | Relative → Absolute Pos Emb | **85.6** | 253.3 | **85.3** | 20.7 |
| (b) | Conv → MaxPool | 84.4 | 99.9† | 84.1 | 10.4† |
| (c) | stride=1 MaxPool 삭제 | 85.4 | 309.2 | 84.3 | 26.2 |
| (d) | Kernel size = Stride | **85.7** | 369.8 | **85.5** | 29.4 |
| (e) | Q Attention Residual 삭제 | 85.6 | 374.3 | 85.5 | 29.8 |
| **(f)** | **KV Pooling → MU Attention** | **85.6** | **531.4** | **85.5** | **40.8** |

최종 Hiera-L은 MViTv2-L 대비 이미지 **2.4×**, 비디오 **5.1×** 빠르면서 정확도는 오히려 향상됩니다.

#### ImageNet-1K 성능 비교 (Table 8)

| 모델 | 사전학습 | Top-1 Acc | FLOPs |
|------|---------|-----------|-------|
| ViT-B | MAE | 83.6 | 18G |
| MViTv2-B | Supervised | 84.4 | 10G |
| MCMAE-B | MCMAE | 85.0 | 28G |
| **Hiera-B** | **MAE** | **84.5** | **9G** |
| **Hiera-B+** | **MAE** | **85.2** | **13G** |
| ViT-L | MAE | 85.9 | 62G |
| **Hiera-L** | **MAE** | **86.1** | **40G** |
| ViT-H | MAE | 86.9 | 167G |
| **Hiera-H** | **MAE** | **86.9** | **125G** |

#### Kinetics-400 비디오 분류 (Table 4)

| 모델 | 사전학습 | Top-1 Acc | FLOPs |
|------|---------|-----------|-------|
| ViT-B | MAE | 81.5 | $180 \times 3 \times 5$ |
| MViTv2-L | MaskFeat | 84.3 | $377 \times 1 \times 10$ |
| ViT-L | MAE | 85.2 | $597 \times 3 \times 5$ |
| **Hiera-L** | **MAE** | **87.3** | **$413 \times 3 \times 5$** |
| ViT-H | MAE | 86.6 | $1192 \times 3 \times 5$ |
| **Hiera-H** | **MAE** | **87.8** | **$1159 \times 3 \times 5$** |

#### 훈련 속도 (Figure 7)

- Hiera-L: MViTv2-L Supervised 대비 이미지 **3×**, 비디오 **9.5×** 빠른 학습
- Hiera-L 200 epochs MAE: MViTv2-L 200 epochs Supervised보다 높은 비디오 정확도 (81.8 vs 80.5)

---

### 2.5 한계점

논문에서 명시적으로 언급된 한계 및 분석에서 도출되는 한계:

**1. MAE 사전학습 의존성**

논문 Appendix B에서 명확히 밝히듯, MAE 사전학습 없이 **from scratch 지도학습**을 하면 bells-and-whistles 제거 시 오히려 성능이 **단조적으로 하락**합니다 (MViT-B 84.4% → Hiera-B 80.8%). 즉, Hiera는 반드시 강력한 사전학습이 전제되어야 합니다.

**2. 소규모 데이터셋에서의 한계**

MAE 사전학습의 이점은 대규모 데이터셋에서 두드러집니다. 소규모 데이터셋에서 Hiera의 성능 이점이 유지되는지에 대한 실험은 제한적입니다.

**3. 다운스트림 태스크에서의 패러다임 전환 필요**

Hiera는 ViT처럼 동작하므로, MViT/Swin에서 작동하던 기존 Mask R-CNN 헤드를 직접 사용하기 어렵습니다. ViTDet과 같은 Transformer 기반 솔루션이 필요합니다.

**4. COCO 탐지에서 ViTDet 대비 약세**

Table 10에서 Hiera-L은 ViTDet-L 대비 $\text{AP}^\text{box}$ 55.0 vs 55.6으로 약간 낮습니다.

**5. Flash Attention 등 최적화 미적용**

논문의 속도 벤치마크는 Flash Attention 등 추가 최적화 없이 측정되었으며, 실제 구현에서는 더 큰 이점을 볼 수 있을 것으로 언급됩니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 MAE 사전학습과 일반화의 관계

Hiera의 핵심 일반화 메커니즘은 **MAE를 통한 공간적 편향 학습**입니다.

$$\text{공간적 편향 학습} = \underbrace{\text{MAE 재구성 손실}}_{\mathcal{L}_\text{MAE}} + \underbrace{\text{Mask Unit 구조}}_{\text{귀납적 편향}}$$

MAE 사전학습이 일반화에 미치는 효과:

**Drop path rate의 역할 (Table 3d)**

놀랍게도, 기존 MAE 레시피(He et al., 2022)에서는 사전학습 시 drop path를 사용하지 않지만, Hiera는 깊이가 ViT-L(24층)의 두 배인 48층이므로 사전학습 시 drop path가 **필수적**입니다:

$$\text{Drop Path Rate} = 0.2 \text{ (최적)} \Rightarrow \text{이미지 } 85.2\% \rightarrow 85.6\%, \text{ 비디오 } 84.5\% \rightarrow 85.5\%$$

Drop path 없이는 Hiera가 **MAE 태스크에 과적합**되어 일반화 성능이 저하됩니다.

### 3.2 전이 학습(Transfer Learning) 성능

#### iNaturalist & Places 분류 (Table 9)

| 모델 | iNat17 | iNat18 | iNat19 | Places365 |
|------|--------|--------|--------|-----------|
| ViT-B MAE | 70.5 | 75.4 | 80.5 | 57.9 |
| **Hiera-B** | **73.3** | **77.9** | **83.0** | **58.9** |
| ViT-L MAE | 75.7 | 80.1 | 83.4 | 59.4 |
| **Hiera-L** | **76.8** | **80.9** | **84.3** | **59.6** |
| ViT-H MAE | 79.3 | 83.0 | 85.7 | 59.8 |
| **Hiera-H** | **79.6** | **83.5** | **85.7** | **60.0** |

Hiera는 **동일 파라미터 크기의 ViT MAE보다 일관되게 우수**한 전이 학습 성능을 보입니다. 이는 계층적 표현이 다양한 도메인에서 더 풍부한 특징을 학습함을 의미합니다.

#### COCO 객체 탐지 및 분할 (Table 10)

$$\text{Hiera-B vs. ViTDet-B: } \Delta\text{AP}^\text{box} = +0.6, \quad -34\% \text{ params}, \quad -15\% \text{ inference time}$$
$$\text{Hiera-L vs. MViTv2-L: } \Delta\text{AP}^\text{box} = +1.8, \quad -24\% \text{ inference time}$$

#### AVA Action Detection (Table 7)

$$\text{Hiera-L vs. ViT-L MAE (K400 pretrain): } \Delta\text{mAP} = +2.8$$
$$\text{Hiera-H vs. ViT-H MAE: } \Delta\text{mAP} = +3.0$$

### 3.3 일반화 향상의 메커니즘 분석

**1. 계층적 멀티스케일 표현**

Hiera는 4단계에서 다양한 스케일의 특징을 학습합니다:

$$\text{Stage 1: Local features, high resolution } (56^2)$$
$$\text{Stage 2: Mid-level features } (28^2)$$
$$\text{Stage 3: Semantic features } (14^2)$$
$$\text{Stage 4: Global context } (7^2)$$

이 계층적 표현은 FPN(Feature Pyramid Network)과 자연스럽게 결합 가능하여 탐지/분할 태스크에서 유리합니다.

**2. Learned Spatial Bias vs. Hard-coded Spatial Bias**

핵심 일반화 철학:

$$\text{Hard-coded Bias (Swin, MViTv2)} \rightarrow \text{Overfitting to supervised ImageNet distribution}$$
$$\text{Learned Bias (Hiera + MAE)} \rightarrow \text{더 유연한 공간적 이해, 다양한 태스크에 적응 가능}$$

논문 Figure 8에서 확인되듯, 지도학습만으로 훈련 시 bells-and-whistles가 필요하다는 사실은 이들이 **특정 지도학습 목적함수에 최적화된 편향**임을 시사합니다. 반면 MAE로 학습된 편향은 더 범용적입니다.

**3. 긴 사전학습 스케줄의 효과 (Table 3f)**

| 사전학습 에폭 | 이미지 정확도 | 비디오 정확도 |
|-------------|------------|------------|
| 400 | 85.6 | 84.0 |
| 800 | 85.8 | 85.5 |
| 1600 | 86.1 | 86.4 |
| 3200 | 86.1 | **87.3** |

비디오에서는 포화 없이 지속적으로 향상되어, 더 긴 학습이 **복잡한 시공간 이해** 능력을 계속 향상시킴을 보여줍니다.

**4. 효율적 학습자(Efficient Learner)로서의 특성**

400 epoch에서 Hiera-L은 ViT-L MAE보다 +0.7% 높지만, 1600 epoch에서는 격차가 +0.2%로 줄어듭니다. 이는 Hiera가 **빠르게 수렴**하는 더 효율적인 학습자임을 의미합니다.

### 3.4 일반화의 잠재적 한계와 가능성

**잠재적 한계**:
- 소규모/저해상도 데이터에서의 성능 검증 부족
- 의료 이미지, 위성 이미지 등 특수 도메인 검증 필요

**미래 일반화 가능성**:
- EMA Teacher(data2vec 등) 결합으로 추가 성능 향상 가능 (논문에서 언급)
- 3D 포인트 클라우드, 멀티모달 등으로의 확장 가능성

---

## 4. 최신 연구 비교 분석 (2020년 이후)

### 4.1 Vision Transformer 계열 비교

| 모델 | 연도 | 특징 | ImageNet Top-1 | 한계 |
|------|------|------|---------------|------|
| **ViT** (Dosovitskiy et al.) | 2021 | 순수 Transformer, non-hierarchical | 85.9 (L, MAE) | 파라미터 비효율, 느린 추론 |
| **Swin** (Liu et al.) | 2021 | Shifted Window, 계층적 | 85.4 (L, SimMIM) | 복잡한 윈도우 연산, MAE 비호환 |
| **MViT** (Fan et al.) | 2021 | Pooling Attention, 계층적 | - | 합성곱 의존성 |
| **MViTv2** (Li et al.) | 2022 | Decomposed Rel Pos, 계층적 | 85.3 (L, Sup.) | 복잡성, 속도 저하 |
| **MAE** (He et al.) | 2022 | ViT + 마스크 사전학습 | 86.9 (H) | 비계층적 구조 |
| **VideoMAE** (Tong et al.) | 2022 | 비디오 MAE | - | 계층적 구조 아님 |
| **MCMAE** (Gao et al.) | 2022 | 마스크 합성곱 + MAE | 86.2 (L) | 합성곱 오버헤드 |
| **ConvNextV2** (Woo et al.) | 2023 | 합성곱 + MAE | 85.3 (B, FCMAE) | 합성곱 의존 |
| **Hiera** (Ryali et al.) | 2023 | 순수 계층적 ViT + MAE | **86.1 (L)**, **86.9 (H)** | MAE 의존성 |

### 4.2 핵심 방법론 비교

#### 공간적 편향 추가 방식

```
Swin:     Hard-coded Shifted Windows (복잡, 빠른 기울기 소멸)
MViTv2:   Convolution + Relative Position Embedding (느림)
MCMAE:    Masked Convolution (MAE 부분 호환, 무거움)
ConvNextV2: 합성곱 + FCMAE (합성곱 구조 유지)
Hiera:    MAE 사전학습으로 학습 (단순, 빠름)
```

#### MAE 호환성 비교

| 방법 | MAE Sparse 지원 | 계층적 구조 | 복잡도 |
|------|----------------|-----------|--------|
| MaskFeat | ✗ (Dense) | ✓ | 중간 |
| SimMIM | ✗ (Dense) | ✓ | 낮음 |
| UM-MAE | 부분적 | ✓ | 중간 |
| MCMAE | 부분적 | ✓ | 높음 |
| **Hiera** | **✓ (완전)** | **✓** | **낮음** |

### 4.3 비디오 분류 발전 흐름

$$\text{TimeSformer (2021)} \rightarrow \text{ViViT (2021)} \rightarrow \text{VideoMAE (2022)} \rightarrow \text{Hiera (2023)}$$

Hiera의 비디오 성능은 이전 SotA 대비 **+2.1% (K400)**, **+2.8% (K700)**의 획기적 개선을 달성하며 새로운 기준점을 설정합니다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 앞으로의 연구에 미치는 영향

#### 영향 1: "단순성의 미덕" 패러다임 전환

Hiera는 **복잡한 아키텍처 설계보다 강력한 사전학습 패러다임이 더 중요**할 수 있음을 실증합니다. 이는 향후 연구 방향을 다음과 같이 전환시킬 수 있습니다:

$$\text{기존: } \text{더 복잡한 아키텍처} \rightarrow \text{더 높은 성능}$$
$$\text{Hiera 이후: } \text{단순한 아키텍처} + \text{강력한 사전학습} \rightarrow \text{더 높은 성능}$$

#### 영향 2: Masked Pretraining의 범용화

Hiera의 Mask Unit 개념은 계층적 구조와 MAE의 호환성 문제를 우아하게 해결합니다. 이는 향후 **더 다양한 모달리티(오디오, 3D, 멀티모달)**에서의 sparse MAE 적용 연구를 촉진할 것입니다.

#### 영향 3: 효율적인 비디오 이해 모델

기존 비디오 Transformer는 연산 비용이 높아 실용적 적용이 어려웠습니다. Hiera는 비디오에서 **ViT 대비 2.3-2.8× 빠르면서** 더 높은 정확도를 달성하여, 실시간 비디오 처리 연구의 기반이 될 것입니다.

#### 영향 4: SAM(Segment Anything Model)과의 연결

실제로 Meta AI의 SAM2(2024)는 Hiera를 백본으로 사용하여 이미지 및 비디오에서의 객체 분할을 달성했습니다. 이는 Hiera가 **범용 비전 백본**으로서의 잠재력을 가짐을 실증합니다.

### 5.2 앞으로 연구 시 고려할 점

#### 고려점 1: 사전학습 데이터 규모와 품질

Hiera의 성능은 MAE 사전학습의 질과 양에 크게 의존합니다:

$$\text{고려사항: } \text{소규모 데이터셋에서의 Hiera 성능} \approx ?$$

향후 연구에서는 **제한된 데이터 환경**에서의 성능 분석과 함께, 데이터 효율적인 사전학습 방법(예: DINO, data2vec와의 결합)을 탐색해야 합니다.

#### 고려점 2: Mask Unit 설계의 최적화

현재 Mask Unit 크기($32 \times 32$ pixels)는 휴리스틱하게 설계되었습니다. 향후 연구에서는:

$$\text{최적 Mask Unit 크기} = f(\text{입력 해상도}, \text{태스크}, \text{도메인})$$

태스크와 도메인에 따른 적응적 Mask Unit 설계가 필요합니다.

#### 고려점 3: 다운스트림 태스크 어댑터

Hiera는 ViT처럼 동작하므로 ViTDet 같은 어댑터가 필요합니다. 더 효율적인 **Hiera 전용 탐지/분할 헤드** 설계가 중요한 연구 방향입니다.

#### 고려점 4: 멀티모달 확장

비디오와 이미지 이외에도:
- **오디오-비주얼 학습**: MAE-Audio와의 결합
- **포인트 클라우드**: 3D 공간에서의 Mask Unit 정의
- **의료 영상**: 다양한 해상도와 모달리티 지원

#### 고려점 5: MAE 이외 사전학습과의 결합

논문에서 언급하듯, Hiera는 EMA Teacher 기반 방법(data2vec, DINO)과 **직교적(orthogonal)**으로 결합 가능합니다:

$$\text{Hiera} + \text{DINO/data2vec} \rightarrow \text{추가 성능 향상 가능성}$$

이 방향의 탐색이 중요합니다.

#### 고려점 6: 경량화와 엣지 배포

현재 Hiera-T/S는 유망한 성능을 보이지만, **모바일/엣지 디바이스**에서의 실용적 배포를 위한 추가적인 경량화 연구(지식 증류, 양자화)가 필요합니다.

#### 고려점 7: 공정한 비교를 위한 벤치마크 재정의

Hiera는 fp16 A100 기준 속도를 보고하지만, 실제 배포 환경(다양한 하드웨어, 배치 크기)에서의 성능 비교가 필요합니다. **표준화된 벤치마크**의 확립이 중요합니다.

---

## 결론

Hiera는 "단순함이 곧 강력함"이라는 메시지를 실증적으로 증명한 중요한 연구입니다. 복잡한 Vision-specific 모듈 없이도 MAE 사전학습을 통해 공간적 편향을 학습함으로써, 속도와 정확도를 동시에 향상시켰습니다. 특히 계층적 구조와 Sparse MAE의 호환성 문제를 Mask Unit이라는 우아한 개념으로 해결한 것은 향후 다양한 모달리티와 태스크로의 확장 가능성을 열어줍니다.

---

## 참고 자료 (출처)

1. **Ryali, C., Hu, Y.-T., Bolya, D., et al.** "Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles." arXiv:2306.00989v1, 2023. (제공된 논문 PDF)

2. **He, K., Chen, X., Xie, S., Li, Y., Dollár, P., Girshick, R.** "Masked Autoencoders Are Scalable Vision Learners." CVPR, 2022.

3. **Liu, Z., Lin, Y., Cao, Y., et al.** "Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows." ICCV, 2021.

4. **Li, Y., Wu, C.-Y., Fan, H., et al.** "MViTv2: Improved Multiscale Vision Transformers for Classification and Detection." CVPR, 2022.

5. **Fan, H., Xiong, B., Mangalam, K., et al.** "Multiscale Vision Transformers." ICCV, 2021.

6. **Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al.** "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR, 2021.

7. **Gao, P., Ma, T., Li, H., et al.** "MCMAE: Masked Convolution Meets Masked Autoencoders." NeurIPS, 2022.

8. **Feichtenhofer, C., Fan, H., Li, Y., He, K.** "Masked Autoencoders as Spatiotemporal Learners." NeurIPS, 2022.

9. **Woo, S., Debnath, S., Hu, R., et al.** "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders." arXiv:2301.00808, 2023.

10. **Tong, Z., Song, Y., Wang, J., Wang, L.** "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training." NeurIPS, 2022.

11. **Li, Y., Mao, H., Girshick, R., He, K.** "Exploring Plain Vision Transformer Backbones for Object Detection." ECCV, 2022. (ViTDet)

12. **Wei, C., Fan, H., Xie, S., et al.** "Masked Feature Prediction for Self-Supervised Visual Pre-Training." CVPR, 2022. (MaskFeat)
