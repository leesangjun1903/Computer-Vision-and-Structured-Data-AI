# "MetaFormer Is Actually What You Need for Vision" 논문 심층 분석

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문의 핵심 주장은 **Transformer 및 MLP-like 모델의 성공은 특정 토큰 믹서(예: Self-Attention, Spatial MLP)가 아니라, 그 이면에 존재하는 일반 아키텍처(General Architecture)인 "MetaFormer"에 기인한다**는 것이다. 저자들은 토큰 믹서를 극도로 단순한 비학습(non-parametric) 연산인 **평균 풀링(Average Pooling)**으로 대체하더라도 경쟁력 있는 성능을 달성할 수 있음을 보여줌으로써 이 가설을 검증하였다.

### 주요 기여

1. **MetaFormer 개념 정립**: Transformer에서 토큰 믹서를 특정하지 않은 일반 아키텍처를 "MetaFormer"로 추상화하고, 이 아키텍처 자체가 모델 성능의 핵심 요인임을 실증적으로 입증하였다.
2. **PoolFormer 제안**: 비학습 파라미터 풀링 연산을 토큰 믹서로 사용한 PoolFormer를 제안하여, ImageNet-1K에서 82.1% top-1 accuracy(PoolFormer-M36)를 달성하고 DeiT-B, ResMLP-B24를 각각 0.3%/1.1% 능가하면서 35%/52% 적은 파라미터, 50%/62% 적은 MACs를 기록하였다.
3. **다양한 비전 태스크 검증**: 이미지 분류, 객체 검출, 인스턴스 세그멘테이션, 시맨틱 세그멘테이션에서 경쟁력 있는 성능을 확인하였다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

2020년 이후 Vision Transformer(ViT)가 컴퓨터 비전에서 큰 성공을 거두면서, 그 성능의 원천이 **Self-Attention 기반 토큰 믹서**에 있다는 것이 일반적 믿음이었다. 그러나 MLP-Mixer, ResMLP 등의 후속 연구에서 Attention을 Spatial MLP로 대체해도 경쟁력 있는 성능이 유지됨이 관찰되었고, FNet에서는 Fourier Transform으로 대체해도 원래 성능의 약 97%를 달성하였다.

이 논문은 다음의 근본적 질문을 제기한다:

> **"Transformer와 그 변형들의 성공에 진정으로 책임이 있는 것은 무엇인가?"**

저자들의 답은 특정 토큰 믹서가 아니라 **일반 아키텍처(MetaFormer)** 자체라는 것이다.

### 2.2 제안하는 방법 (수식 포함)

#### MetaFormer 구조 정의

**입력 임베딩:**

$$X = \mathrm{InputEmb}(I)$$

여기서 $X \in \mathbb{R}^{N \times C}$이며, $N$은 시퀀스 길이, $C$는 임베딩 차원이다.

**토큰 믹서 서브블록 (첫 번째 잔차 서브블록):**

$$Y = \mathrm{TokenMixer}(\mathrm{Norm}(X)) + X$$

여기서 $\mathrm{Norm}(\cdot)$은 Layer Normalization 또는 Batch Normalization이고, $\mathrm{TokenMixer}(\cdot)$는 토큰 간 정보를 혼합하는 모듈이다. Attention, Spatial MLP, 또는 Pooling 등으로 구체화될 수 있다.

**채널 MLP 서브블록 (두 번째 잔차 서브블록):**

$$Z = \sigma(\mathrm{Norm}(Y) W_1) W_2 + Y$$

여기서 $W_1 \in \mathbb{R}^{C \times rC}$, $W_2 \in \mathbb{R}^{rC \times C}$는 학습 가능한 파라미터이고, $r$은 MLP 확장 비율(논문에서는 4), $\sigma(\cdot)$는 GELU와 같은 비선형 활성화 함수이다.

#### PoolFormer의 풀링 연산

토큰 믹서를 비학습 파라미터 평균 풀링으로 대체한다. 입력이 채널-우선 형식 $T \in \mathbb{R}^{C \times H \times W}$일 때:

$$T'_{:, i, j} = \frac{1}{K \times K} \sum_{p,q=1}^{K} T_{:, \, i+p-\frac{K+1}{2}, \, j+q-\frac{K+1}{2}} - T_{:, i, j}$$

여기서 $K$는 풀링 크기(기본값 3)이다. MetaFormer 블록이 이미 잔차 연결을 포함하므로, 입력 자체($T_{:,i,j}$)를 빼주는 것이 핵심이다. 이 연산은 **학습 가능한 파라미터가 전혀 없으며**, 각 토큰이 인접 토큰의 특징을 균등 평균하여 집계하는 극도로 기본적인 토큰 혼합이다.

### 2.3 모델 구조

PoolFormer는 **4-스테이지 계층적(hierarchical) 구조**를 채택한다 (ResNet, Swin Transformer와 유사):

| 스테이지 | 토큰 해상도 | 소형 임베딩 차원 | 중형 임베딩 차원 |
|---------|-----------|-------------|-------------|
| 1 | $\frac{H}{4} \times \frac{W}{4}$ | 64 | 96 |
| 2 | $\frac{H}{8} \times \frac{W}{8}$ | 128 | 192 |
| 3 | $\frac{H}{16} \times \frac{W}{16}$ | 320 | 384 |
| 4 | $\frac{H}{32} \times \frac{W}{32}$ | 512 | 768 |

총 $L$개의 PoolFormer 블록이 있을 때, 스테이지 [1, 2, 3, 4]에 각각 $[L/6, L/6, L/2, L/6]$ 블록이 배치된다. 각 스테이지 사이에는 $3 \times 3$ 패치 임베딩(stride 2)으로 다운샘플링을 수행한다.

**모델 변형:**

| 모델 | 파라미터 (M) | MACs (G) | Top-1 (%) |
|------|-----------|----------|-----------|
| PoolFormer-S12 | 11.9 | 1.8 | 77.2 |
| PoolFormer-S24 | 21.4 | 3.4 | 80.3 |
| PoolFormer-S36 | 30.8 | 5.0 | 81.4 |
| PoolFormer-M36 | 56.1 | 8.8 | 82.1 |
| PoolFormer-M48 | 73.4 | 11.6 | 82.5 |

주요 학습 기법: Modified Layer Normalization(GroupNorm with 1 group), GELU 활성화, Stochastic Depth, LayerScale을 사용한다.

### 2.4 성능 향상

#### ImageNet-1K 이미지 분류

- **PoolFormer-S24** (21M params, 3.4G MACs): **80.3%** → DeiT-S (22M, 4.6G MACs, 79.8%)보다 0.5% 높고 MACs 26% 절감
- **PoolFormer-M36** (56M params, 8.8G MACs): **82.1%** → DeiT-B (86M, 17.5G MACs, 81.8%)보다 0.3% 높으면서 파라미터 35%, MACs 50% 절감
- ResMLP-B24 (116M, 23.0G MACs, 81.0%) 대비 1.1% 높으면서 파라미터 52%, MACs 62% 절감

#### 객체 검출 및 인스턴스 세그멘테이션 (COCO)

- RetinaNet 기반: PoolFormer-S12 **36.2 AP** vs. ResNet-18 **31.8 AP** (4.4 AP 향상)
- Mask R-CNN 기반: PoolFormer-S12 bbox AP **37.3** / mask AP **34.6** vs. ResNet-18 **34.0** / **31.2**

#### 시맨틱 세그멘테이션 (ADE20K)

- PoolFormer-S12: **37.2 mIoU** vs. ResNet-18 **32.9** (+4.3), PVT-Tiny **35.7** (+1.5)
- PoolFormer-S36: **42.0 mIoU** vs. PVT-Medium **41.6** (+0.4), 파라미터 28% 절감

#### Ablation Study 핵심 결과

| 토큰 믹서 | Top-1 (%) |
|---------|-----------|
| Pooling (기본) | 77.2 |
| Identity Mapping | 74.3 |
| Global Random Matrix | 75.8 |
| Depthwise Convolution | 78.1 |
| Pooling + Attention (하이브리드) | 81.0 (16.5M, 2.5G) |

**Identity Mapping으로도 74.3% 달성**이라는 결과는 MetaFormer 아키텍처 자체의 강력함을 가장 극적으로 보여주는 증거이다.

### 2.5 한계

1. **토큰 믹서의 지역성 한계**: 풀링은 지역적(local) 정보만 집계하므로, 전역적(global) 의존성 모델링이 불가능하다. 이는 대규모 객체 인식이나 장거리 의존성이 중요한 태스크에서 성능 제한을 유발할 수 있다.
2. **비전 태스크에 한정된 검증**: NLP, 멀티모달 등 다른 도메인에서의 MetaFormer 가설은 검증되지 않았다.
3. **자기지도 학습, 전이 학습 미검증**: 사전학습-파인튜닝 패러다임이나 self-supervised learning에서의 효과는 확인되지 않았다.
4. **SOTA 달성이 목적이 아님**: 논문의 목적은 아키텍처 가설 검증이므로, 최신 SOTA와의 직접 비교에서는 경쟁력에 제한이 있을 수 있다.
5. **스케일링 한계**: 매우 대규모 데이터셋(JFT-300M 등)에서의 사전학습 효과는 확인되지 않았다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 MetaFormer의 일반화에 대한 이론적 근거

MetaFormer의 일반화 능력은 다음 구성 요소들의 시너지에 기반한다:

1. **잔차 연결(Residual Connection)**: 잔차 연결이 없으면 정확도가 0.1%로 붕괴한다(Table 5). Raghu et al. (2021)은 잔차 연결이 하위 레이어의 특징을 상위 레이어로 효과적으로 전파하여 gradient flow를 안정화시킴을 보여주었다.

2. **정규화(Normalization)**: 정규화 제거 시 정확도가 46.1%로 급락한다. Modified Layer Normalization은 채널과 공간 차원 모두에 대해 정규화하여 학습 안정성과 일반화를 동시에 향상시킨다.

3. **채널 MLP**: 채널 차원에서의 비선형 변환이 특징 표현력을 결정적으로 향상시킨다(제거 시 5.7%).

4. **토큰 믹서의 교체 가능성**: Identity mapping(74.3%), Random matrix(75.8%), Pooling(77.2%), Depthwise Conv(78.1%)로 토큰 믹서를 바꾸어도 MetaFormer는 합리적인 성능을 유지한다. 이는 MetaFormer 아키텍처가 **특정 토큰 믹서에 의존하지 않는 강건한 일반화 능력**을 내재하고 있음을 의미한다.

### 3.2 일반화 향상 전략

#### 하이브리드 스테이지 전략
하위 스테이지(긴 시퀀스)에서 풀링을, 상위 스테이지(짧은 시퀀스)에서 Attention을 사용하는 하이브리드 모델이 **81.0%** 정확도를 16.5M 파라미터, 2.5G MACs로 달성하였다. 이는 ResMLP-B24 (116M params, 23.0G MACs, 81.0%)과 동일 성능이면서 7.0배 적은 파라미터, 9.2배 적은 연산량이다. 이러한 하이브리드 접근은 **효율성과 일반화의 동시 향상** 가능성을 시사한다.

#### 장기 학습에서의 일반화
PoolFormer-S12는 300 에폭에서 77.2% → 2000 에폭에서 78.8%로 1.6% 향상을 보이며 포화한다. 이는 DeiT(400 에폭 포화), ResMLP(800 에폭 포화)와 비교하여 PoolFormer가 더 오랜 학습으로도 지속적으로 개선될 여지가 있음을 시사한다.

#### 다운스트림 태스크로의 전이
COCO 검출/세그멘테이션, ADE20K 세그멘테이션에서 ResNet, PVT 대비 일관된 우위는 PoolFormer(및 MetaFormer)가 **다양한 비전 태스크에 걸쳐 좋은 일반화**를 보임을 증명한다.

### 3.3 Park et al. (2022)의 관점과의 연결

Park & Kim (2022, "How Do Vision Transformers Work?")은 Multi-Head Self-Attention이 **loss landscape를 평탄화(flattening)**하여 정확도와 일반화를 향상시킨다고 보고하였다. MetaFormer 관점에서 보면, 이러한 일반화 이점은 Attention 자체뿐 아니라 **잔차 연결, 정규화, 채널 MLP의 조합**에서도 상당 부분 기인할 수 있다. PoolFormer가 Attention 없이도 경쟁력 있는 일반화를 달성한다는 사실은 이를 뒷받침한다.

---

## 4. 연구 영향 및 향후 고려 사항

### 4.1 이 논문이 미친 영향

1. **아키텍처 설계 패러다임 전환**: 연구 커뮤니티의 초점을 "어떤 토큰 믹서가 최선인가?"에서 "어떤 전반적 아키텍처 설계가 최선인가?"로 전환시켰다. 이는 후속 연구들에 직접적으로 영향을 미쳤다.

2. **효율적 모델 설계의 기반**: 복잡한 Attention 메커니즘 없이도 경쟁력 있는 성능을 달성할 수 있음을 보여줌으로써, 경량화 및 효율적 모델 설계의 새로운 가능성을 열었다.

3. **후속 MetaFormer 연구 촉발**: 동일 저자들의 후속 논문 "MetaFormer Baselines for Vision" (Yu et al., TPAMI 2024)에서 IdentityFormer, RandFormer, ConvFormer, CAFormer 등으로 가설을 더욱 체계적으로 검증하였다.

### 4.2 향후 연구 시 고려할 점

1. **MetaFormer 구성 요소의 체계적 분석**: 잔차 연결, 정규화, 채널 MLP 각각의 기여를 더 세밀하게 분리 분석할 필요가 있다.

2. **다른 도메인으로의 확장**: NLP, 오디오, 멀티모달, 비디오 등에서 MetaFormer 가설의 일반성을 검증해야 한다.

3. **자기지도 학습과의 결합**: MAE, DINO 등 자기지도 학습 프레임워크에서 MetaFormer 아키텍처의 효과를 검증하는 연구가 필요하다.

4. **토큰 믹서 조합 최적화**: 하이브리드 스테이지 실험이 보여주듯, 다양한 토큰 믹서의 최적 조합을 찾는 NAS(Neural Architecture Search) 기반 연구가 유망하다.

5. **스케일링 법칙**: MetaFormer 아키텍처의 스케일링 특성(모델 크기, 데이터 규모, 학습 연산량에 따른 성능 변화)에 대한 체계적 연구가 필요하다.

6. **이론적 분석**: MetaFormer가 왜 효과적인지에 대한 이론적 이해(예: 표현력, 최적화 경관, 일반화 바운드)를 심화해야 한다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 연구 계보 및 비교표

| 연구 | 연도 | 핵심 토큰 믹서 | ImageNet Top-1 | 핵심 기여 |
|------|------|-------------|----------------|---------|
| **ViT** (Dosovitskiy et al.) | 2020 | Self-Attention | 79.7% (ViT-B, IN-1K only) | 순수 Transformer를 비전에 적용 |
| **DeiT** (Touvron et al.) | 2021 | Self-Attention | 81.8% (DeiT-B) | 데이터 효율적 학습, 증류 |
| **MLP-Mixer** (Tolstikhin et al.) | 2021 | Spatial MLP | 76.4% (Mixer-B) | Attention 없는 순수 MLP 모델 |
| **ResMLP** (Touvron et al.) | 2021 | Spatial MLP | 81.0% (ResMLP-B24) | 잔차 연결 기반 MLP |
| **Swin Transformer** (Liu et al.) | 2021 | Shifted Window Attention | 83.5% (Swin-B) | 계층적 구조, 선형 복잡도 |
| **FNet** (Lee-Thorp et al.) | 2021 | Fourier Transform | ~97% of BERT (NLP) | 비학습 Fourier 토큰 믹싱 |
| **gMLP** (Liu et al.) | 2021 | Spatial Gating MLP | 81.6% (gMLP-B) | 게이팅 기반 MLP |
| **ConvNeXt** (Liu et al.) | 2022 | Depthwise Conv | 83.8% (ConvNeXt-B) | Transformer 설계를 CNN에 적용 |
| **PoolFormer** (Yu et al.) | 2022 | Average Pooling | 82.1% (PoolFormer-M36) | **MetaFormer 개념 정립** |
| **MetaFormer Baselines** (Yu et al.) | 2022/2024 | Identity/Random/Conv/CA | 85.5% (CAFormer-B36) | MetaFormer 가설 확장 검증 |
| **EfficientFormer** (Li et al.) | 2022 | Pooling + Attention | 83.3% (EfficientFormer-L7) | MetaFormer 기반 모바일 최적화 |
| **FastViT** (Vasu et al.) | 2023 | RepMixer (Conv 기반) | 84.9% (FastViT-MA36) | MetaFormer 구조의 효율적 변형 |

### 5.2 핵심 비교 분석

#### PoolFormer vs. ConvNeXt (Liu et al., CVPR 2022)

ConvNeXt는 MetaFormer의 관점과 유사한 방향에서 **순수 CNN에 Transformer의 매크로/마이크로 설계 전략**을 적용하였다. ConvNeXt-T(28.6M, 4.5G MACs, 82.1%)와 PoolFormer-M36(56.1M, 8.8G MACs, 82.1%)이 동일 정확도인 점은 흥미롭다. ConvNeXt는 Depthwise Convolution이라는 더 강력한 토큰 믹서를 사용하여 더 적은 파라미터/연산으로 동일 성능을 달성하며, 이는 **MetaFormer 구조 위에 더 나은 토큰 믹서를 사용하면 효율성이 향상됨**을 시사한다.

#### PoolFormer vs. MetaFormer Baselines (Yu et al., TPAMI 2024)

이 후속 논문은 원저자들이 MetaFormer 가설을 더 체계적으로 검증한 것으로:
- **IdentityFormer**: 토큰 믹서 없이(identity mapping) 하위 스테이지를 구성 → 여전히 합리적 성능
- **RandFormer**: 랜덤 토큰 믹싱 → IdentityFormer보다 향상
- **ConvFormer**: Depthwise Separable Convolution → 84.1% (ConvFormer-B36)
- **CAFormer**: 하위 Conv + 상위 Attention 하이브리드 → **85.5%** (CAFormer-B36, IN-1K만 사용)

CAFormer-B36의 85.5%는 ImageNet-1K만 사용한 모델 중 최고 수준이며, 이는 **MetaFormer + 적절한 토큰 믹서 조합**의 위력을 입증한다.

#### PoolFormer vs. EfficientFormer (Li et al., NeurIPS 2022)

EfficientFormer는 MetaFormer 구조를 모바일 환경에 최적화한 것으로, 하위 스테이지에서 풀링/Conv, 상위 스테이지에서 Attention을 사용하는 하이브리드 전략을 채택하였다. 이는 PoolFormer의 하이브리드 ablation 실험에서 영감을 받은 것으로 볼 수 있다.

#### PoolFormer vs. FastViT (Vasu et al., ICCV 2023)

FastViT는 MetaFormer 구조에 **RepMixer**(구조적 재파라미터화 기반 토큰 믹서)를 적용하여 추론 시 효율성을 극대화하였다. 이는 MetaFormer의 토큰 믹서 교체 가능성을 추론 효율성 관점에서 활용한 사례이다.

### 5.3 종합 인사이트

2020년 이후의 비전 아키텍처 연구를 MetaFormer 관점에서 재해석하면:

1. **ViT, DeiT, Swin Transformer**: MetaFormer + Attention 계열 토큰 믹서
2. **MLP-Mixer, ResMLP, gMLP**: MetaFormer + Spatial MLP 계열 토큰 믹서
3. **ConvNeXt**: MetaFormer + Depthwise Convolution 토큰 믹서
4. **PoolFormer**: MetaFormer + 비학습 풀링 토큰 믹서

모든 성공적 모델이 **MetaFormer 구조(잔차 연결, 정규화, 채널 MLP의 반복 블록)**를 공유한다는 사실은 이 논문의 핵심 주장을 강력하게 뒷받침한다. 토큰 믹서의 선택은 성능의 **상한선(ceiling)**을 결정하지만, MetaFormer 아키텍처가 성능의 **하한선(floor)**을 보장한다.

---

## 참고자료

1. **Yu, W., et al.** "MetaFormer Is Actually What You Need for Vision." *arXiv:2111.11418v3*, 2022. (본 논문)
2. **Yu, W., et al.** "MetaFormer Baselines for Vision." *IEEE TPAMI*, 2024. (arXiv:2210.13452)
3. **Dosovitskiy, A., et al.** "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR*, 2021.
4. **Touvron, H., et al.** "Training Data-Efficient Image Transformers & Distillation through Attention." *ICML*, 2021. (DeiT)
5. **Tolstikhin, I., et al.** "MLP-Mixer: An All-MLP Architecture for Vision." *NeurIPS*, 2021.
6. **Touvron, H., et al.** "ResMLP: Feedforward Networks for Image Classification with Data-Efficient Training." *arXiv:2105.03404*, 2021.
7. **Liu, Z., et al.** "Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows." *ICCV*, 2021.
8. **Liu, Z., et al.** "A ConvNet for the 2020s." *CVPR*, 2022. (ConvNeXt)
9. **Li, Y., et al.** "EfficientFormer: Vision Transformers at MobileNet Speed." *NeurIPS*, 2022.
10. **Vasu, P. K. A., et al.** "FastViT: A Fast Hybrid Vision Transformer Using Structural Reparameterization." *ICCV*, 2023.
11. **Lee-Thorp, J., et al.** "FNet: Mixing Tokens with Fourier Transforms." *arXiv:2105.03824*, 2021.
12. **Liu, H., et al.** "Pay Attention to MLPs." *NeurIPS*, 2021. (gMLP)
13. **Park, N. & Kim, S.** "How Do Vision Transformers Work?" *ICLR*, 2022.
14. **Wightman, R., et al.** "ResNet Strikes Back: An Improved Training Procedure in Timm." *arXiv:2110.00476*, 2021.
15. **Dong, Y., et al.** "Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth." *arXiv:2103.03404*, 2021.
16. 공식 코드: https://github.com/sail-sg/poolformer
