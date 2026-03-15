# Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation

---

## 1. 핵심 주장 및 주요 기여 요약

**Mask2Former**는 panoptic, instance, semantic segmentation 등 모든 이미지 분할 과제를 **단일 아키텍처**로 해결하는 **범용(Universal) 이미지 분할 모델**이다. 핵심 주장과 기여는 다음과 같다:

1. **Masked Attention 제안**: 표준 cross-attention이 전체 이미지에 걸쳐 attention을 수행하는 것과 달리, 예측된 마스크의 foreground 영역 내로 attention을 제한하여 **빠른 수렴과 성능 향상**을 달성한다.
2. **효율적 멀티스케일 고해상도 피처 활용**: Feature pyramid의 다양한 해상도를 round-robin 방식으로 Transformer decoder layer에 공급하여 작은 객체 분할 성능을 향상시킨다.
3. **최적화 개선**: Self/cross-attention 순서 변경, learnable query features, dropout 제거 등 추가 연산 없이 성능을 향상시킨다.
4. **학습 효율성**: 랜덤 포인트 샘플링 기반 마스크 손실 계산으로 학습 메모리를 **3배 절감**한다.
5. **SOTA 달성**: COCO panoptic (57.8 PQ), COCO instance (50.1 AP), ADE20K semantic (57.7 mIoU)에서 **최초로 전문화 모델을 모두 능가**하는 범용 아키텍처를 제시한다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

이미지 분할은 semantics에 따라 panoptic, instance, semantic segmentation으로 나뉘지만, 기존 연구는 **각 과제별 전문화(specialized) 아키텍처**를 개발해왔다:

- **Semantic segmentation**: FCN 기반 per-pixel classification (Long et al., 2015)
- **Instance segmentation**: Mask R-CNN (He et al., 2017) 등 mask classification
- **Panoptic segmentation**: 양자를 결합한 별도 아키텍처

이로 인해 연구 노력, 하드웨어 최적화가 **3배 이상 중복**되며, 한 과제에 특화된 모듈이 다른 과제로 일반화되지 않는 문제가 발생한다. 기존 범용 아키텍처(MaskFormer, K-Net)는 특히 **instance segmentation에서 전문화 모델 대비 9 AP 이상 낮은 성능**을 보이며, 학습에도 더 많은 자원과 epoch가 필요했다.

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 Meta Architecture

Mask2Former는 MaskFormer (Cheng et al., NeurIPS 2021)의 메타 아키텍처를 계승한다:

- **Backbone**: 저해상도 특징 추출 (ResNet, Swin Transformer 등)
- **Pixel Decoder**: 저해상도 → 고해상도 per-pixel embedding 생성
- **Transformer Decoder**: Object query를 이미지 특징과 함께 처리

$N$개의 세그먼트에 대해 $N$개의 바이너리 마스크와 $N$개의 카테고리 레이블을 예측한다.

#### 2.2.2 Masked Attention

**표준 cross-attention** (residual path 포함):

$$\mathbf{X}_l = \text{softmax}(\mathbf{Q}_l \mathbf{K}_l^T) \mathbf{V}_l + \mathbf{X}_{l-1} $$

여기서:
- $l$: layer index
- $\mathbf{X}_l \in \mathbb{R}^{N \times C}$: $l$번째 layer의 $N$개 $C$-차원 query features
- $\mathbf{Q}\_l = f_Q(\mathbf{X}_{l-1}) \in \mathbb{R}^{N \times C}$
- $\mathbf{K}_l, \mathbf{V}_l \in \mathbb{R}^{H_l W_l \times C}$: 이미지 특징의 변환

**Masked attention**은 attention matrix를 이전 layer의 마스크 예측으로 조절한다:

$$\mathbf{X}_l = \text{softmax}(\mathcal{M}_{l-1} + \mathbf{Q}_l \mathbf{K}_l^T) \mathbf{V}_l + \mathbf{X}_{l-1} $$

Attention mask $\mathcal{M}_{l-1}$은 feature location $(x, y)$에서:

```math
\mathcal{M}_{l-1}(x, y) = \begin{cases} 0 & \text{if } \mathbf{M}_{l-1}(x, y) = 1 \\ -\infty & \text{otherwise} \end{cases}
```

여기서 $\mathbf{M}_{l-1} \in \{0, 1\}^{N \times H_l W_l}$은 이전 $(l-1)$번째 Transformer decoder layer의 마스크 예측을 0.5 임계값으로 이진화한 결과이다. $\mathbf{M}_0$는 $\mathbf{X}_0$(Transformer decoder 입력 전 query features)로부터 얻은 바이너리 마스크 예측이다.

**핵심 직관**: 로컬 특징만으로도 query features를 충분히 업데이트할 수 있으며, 컨텍스트 정보는 self-attention을 통해 수집할 수 있다. 이는 softmax의 특성상 cross-attention에서 넓은 배경 영역의 작은 attention weight들이 누적되어 전경에 대한 attention이 약해지는 문제를 해결한다.

#### 2.2.3 효율적 멀티스케일 전략

Pixel decoder의 feature pyramid에서 해상도 $1/32$, $1/16$, $1/8$의 특징맵을 사용한다. 각 해상도에 대해:

- Sinusoidal positional embedding: $e_{\text{pos}} \in \mathbb{R}^{H_l W_l \times C}$
- Learnable scale-level embedding: $e_{\text{lvl}} \in \mathbb{R}^{1 \times C}$

이를 저해상도 → 고해상도 순서로 Transformer decoder layer에 **round-robin** 방식으로 공급한다. 3-layer 단위를 $L$번 반복하여 총 $3L$ layers를 구성한다 (기본 $L=3$, 즉 9 layers).

$$H_1 = H/32, \quad H_2 = H/16, \quad H_3 = H/8$$

$$W_1 = W/32, \quad W_2 = W/16, \quad W_3 = W/8$$

#### 2.2.4 최적화 개선

1. **Self/cross-attention 순서 변경**: Masked attention → self-attention → FFN (기존: self → cross → FFN). 초기 query가 이미지 독립적이므로 self-attention 전에 이미지 정보를 먼저 주입한다.
2. **Learnable query features ($\mathbf{X}_0$)**: 기존 DETR의 zero initialization 대신 학습 가능하게 설정하고, Transformer decoder 입력 전에 직접 supervision을 적용한다. 이는 **region proposal network** 역할을 수행한다.
3. **Dropout 제거**: Residual connection과 attention map에서 dropout을 완전히 제거한다.

#### 2.2.5 학습 효율성 개선

PointRend (Kirillov et al., 2020)에서 영감을 받아, 마스크 전체 대신 $K$개의 랜덤 샘플링 포인트에서 마스크 손실을 계산한다:

- **Matching loss**: 모든 예측/GT 마스크에 대해 동일한 $K$개 포인트를 균일 샘플링
- **Final loss**: 각 예측-GT 쌍마다 다른 $K$개 포인트를 importance sampling으로 샘플링
- $K = 12544$ (즉, $112 \times 112$ 포인트)

**효과**: 학습 메모리 **18GB → 6GB** (3배 절감), 성능 저하 없음.

#### 2.2.6 손실 함수

마스크 손실:

$$\mathcal{L}_{\text{mask}} = \lambda_{\text{ce}} \mathcal{L}_{\text{ce}} + \lambda_{\text{dice}} \mathcal{L}_{\text{dice}}$$

여기서 $\lambda_{\text{ce}} = 5.0$, $\lambda_{\text{dice}} = 5.0$이다. Binary cross-entropy loss와 dice loss를 사용한다.

최종 손실:

$$\mathcal{L} = \mathcal{L}_{\text{mask}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}}$$

여기서 $\lambda_{\text{cls}} = 2.0$ (매칭된 예측), $\lambda_{\text{cls}} = 0.1$ ("no object" 예측).

### 2.3 모델 구조

| 구성 요소 | 상세 |
|---|---|
| **Backbone** | ResNet-50/101, Swin-T/S/B/L |
| **Pixel Decoder** | MSDeformAttn (6 layers, 해상도 1/8, 1/16, 1/32) + 1/4 해상도 upsampling |
| **Transformer Decoder** | 9 layers (3×3), 100/200 queries, masked attention |
| **Layer 구성** | Masked Attention → Self-Attention → FFN (+ Add & Norm) |
| **출력** | Per-pixel embedding과 object query의 dot product → 바이너리 마스크 + 클래스 예측 |

### 2.4 성능 향상

| 벤치마크 | 메트릭 | Mask2Former | 이전 SOTA (전문화 모델) | 향상폭 |
|---|---|---|---|---|
| COCO Panoptic | PQ | **57.8** | 52.7 (MaskFormer) / 54.6 (K-Net) | +5.1 / +3.2 |
| COCO Instance | AP | **50.1** | 49.5 (Swin-HTC++) | +0.6 |
| ADE20K Semantic | mIoU | **57.7** | 57.0 (BEiT) | +0.7 |
| COCO Instance | AP $^{\text{boundary}}$ | **36.2** | 34.1 (Swin-HTC++) | +2.1 |

Ablation 분석 (ResNet-50 기준):
- Masked attention 제거 시: AP -5.9, PQ -4.8, mIoU -1.7
- 고해상도 피처 제거 시: AP -2.2, PQ -1.7, mIoU -1.1
- 수렴 속도: MaskFormer 대비 **6배 빠른 수렴** (300 epochs → 50 epochs)

### 2.5 한계

1. **과제별 개별 학습 필요**: 동일 아키텍처이지만 panoptic, instance, semantic 과제별로 별도 학습이 필요하다. Panoptic annotation으로만 학습한 모델은 전용 instance/semantic 모델보다 약간 낮은 성능을 보인다.
2. **작은 객체 분할의 한계**: $\text{AP}^S$에서 여전히 전문화 모델에 뒤처진다 (Mask2Former R50: 23.4 vs. Mask R-CNN LSJ: 23.8).
3. **멀티스케일 피처의 불완전 활용**: 단순 concatenation은 효과가 없으며, feature pyramid 활용 방법의 개선 여지가 있다.
4. **추론 속도 저하**: Swin-L 기준 4.0 fps로, MaskFormer (5.2 fps)보다 다소 느리다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 과제 간 일반화

Mask2Former의 가장 중요한 기여는 **단일 아키텍처의 과제 간 일반화**이다:

- **Panoptic annotation만으로 학습**한 모델에서 $\text{AP}^{\text{Th}}\_{\text{pan}} = 48.6$ (instance), $\text{mIoU}_{\text{pan}} = 67.4$ (semantic)을 달성한다 (Table 1). 이는 panoptic 학습만으로도 instance와 semantic segmentation에 활용 가능함을 보여준다.
- 그러나 Table 7에서 보듯이, 과제 전용 학습 모델과의 격차가 여전히 존재한다:
  - COCO: panoptic 모델 AP 41.7 vs. instance 전용 AP **43.7** (+2.0)
  - ADE20K: panoptic 모델 mIoU 46.1 vs. semantic 전용 mIoU **47.2** (+1.1)

### 3.2 데이터셋 간 일반화

Mask2Former는 COCO, ADE20K, Cityscapes, Mapillary Vistas 등 **4개 데이터셋 모두에서 경쟁력 있는 성능**을 보여 데이터셋 간 일반화 능력을 입증한다 (Tables 6, VII, VIII, IX).

### 3.3 일반화 성능 향상의 핵심 요소

1. **Masked Attention의 역할**: Cross-attention에서 foreground에 평균 20%만 집중하던 것을 **약 60%로 향상** (Figure I, Appendix). 이는 과제/데이터셋에 무관하게 객체 영역에 대한 집중도를 높인다.
2. **Pixel Decoder의 일반성**: MSDeformAttn이 세 과제 모두에서 일관되게 최고 성능을 보인다 (Table 4e). BiFPN은 instance에, FaPN은 semantic에 특화되어 있어 일반화에 불리하다.
3. **Learnable Query Features**: Region proposal 역할을 수행하며, AR@100이 50.3 (learnable queries) → 57.7 (layer 9)로 점진적으로 향상된다 (Figure 3).

### 3.4 일반화 향상을 위한 잠재적 방향

1. **멀티태스크 통합 학습**: 현재 과제별 개별 학습의 한계를 극복하기 위해, 하나의 모델을 여러 과제와 데이터셋에 동시 학습하는 방법이 필요하다.
2. **작은 객체에 대한 개선**: Dilated backbone 활용, 작은 객체 전용 손실 함수 설계 등이 필요하다.
3. **효율적 멀티스케일 추론**: Instance segmentation에서의 multi-scale inference에 NMS 없이 대응하는 방법이 필요하다.

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 학술적 영향

1. **범용 아키텍처 패러다임의 확립**: Mask2Former는 "전문화 모델이 범용 모델보다 우수하다"는 기존 통념을 깨뜨렸다. 이후 연구에서 범용 아키텍처 설계가 주류가 되었다.
2. **Masked Attention의 원리 확산**: Attention을 예측 결과로 조절하는 개념은 object detection, video segmentation, 3D 인식 등으로 확장되었다.
3. **학습 효율성 기준 제시**: Point-based loss 계산을 통한 메모리 절감은 이후 대규모 모델 학습의 표준적 기법이 되었다.

### 4.2 후속 연구 시 고려할 점

1. **멀티태스크/멀티데이터셋 동시 학습**: 저자들이 직접 한계로 지적한 바와 같이, 과제별 별도 학습을 넘어 **한 번의 학습으로 모든 과제를 수행**하는 모델 개발이 핵심 도전이다.
2. **Open-vocabulary/Zero-shot 일반화**: 고정된 카테고리 집합을 넘어, 텍스트 프롬프트 등을 통한 범용 분할로의 확장이 필요하다.
3. **계산 효율성**: Masked attention이 고해상도 피처와 결합될 때의 연산 비용 관리가 중요하다.
4. **소규모 객체 처리**: $\text{AP}^S$ 개선을 위한 구조적 혁신이 여전히 필요하다.
5. **비디오 및 3D로의 확장**: 이미지 분할을 넘어 시공간적 일관성을 갖춘 분할로의 확장 가능성을 고려해야 한다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 특징 | Mask2Former와의 관계 |
|---|---|---|---|
| **DETR** (Carion et al.) | 2020 | End-to-end set prediction, Transformer decoder | Mask2Former의 기초 아키텍처. 느린 수렴(500+ epochs) 문제 존재 |
| **MaskFormer** (Cheng et al.) | 2021 | Mask classification으로 semantic + panoptic 통합 | Mask2Former의 직접 전신. Instance segmentation에서 약세 (AP 40.1) |
| **K-Net** (Zhang et al.) | 2021 | Set prediction을 instance segmentation에 확장, mask pooling | Mask2Former의 masked attention이 mask pooling보다 우수 (Table 4c) |
| **Max-DeepLab** (Wang et al.) | 2021 | End-to-end panoptic segmentation | Panoptic에 특화, 높은 FLOPs (3692G), PQ 51.1 vs. Mask2Former 57.8 |
| **Deformable DETR** (Zhu et al.) | 2021 | Multi-scale deformable attention | Mask2Former의 pixel decoder에 MSDeformAttn으로 채택됨 |
| **SMCA** (Gao et al.) | 2021 | Spatially modulated co-attention | Mask2Former의 masked attention이 SMCA보다 모든 과제에서 우수 (Table 4c) |
| **BEiT** (Bao et al.) | 2021 | BERT-style image Transformer pre-training | Semantic seg. SOTA (57.0 mIoU), Mask2Former가 57.7 mIoU로 능가 |
| **Swin Transformer** (Liu et al.) | 2021 | Hierarchical vision Transformer | Mask2Former의 주요 backbone으로 활용 |
| **SegFormer** (Xie et al.) | 2021 | Efficient Transformer for semantic segmentation | Cityscapes 84.0 mIoU, Mask2Former가 84.5 mIoU로 능가 |
| **QueryInst** (Fang et al.) | 2021 | Instance queries for instance segmentation | AP 48.9 (Swin-L), Mask2Former 50.1로 능가 |
| **OneFormer** (Jain et al.) | 2023 | Task-conditioned joint training, 텍스트 프롬프트 | Mask2Former의 후속, 단일 모델로 3개 과제 동시 학습 가능 |
| **Segment Anything (SAM)** (Kirillov et al.) | 2023 | Promptable segmentation, foundation model | 범용 분할의 새로운 패러다임, class-agnostic 마스크 생성에 특화 |
| **Mask DINO** (Li et al.) | 2023 | Detection + segmentation 통합, DINO 기반 | Mask2Former에 detection branch 추가로 성능 향상 |
| **FC-CLIP** (Yu et al.) | 2024 | Open-vocabulary panoptic segmentation | Mask2Former 아키텍처에 CLIP 결합, open-vocabulary 확장 |

### 주요 비교 포인트

**Mask2Former vs. MaskFormer**: 동일 meta architecture에서 masked attention과 최적화 개선만으로 전 과제에서 대폭 향상. Instance AP: 34.0 → 43.7 (R50), 수렴 속도 6배 향상.

**Mask2Former vs. OneFormer (2023)**: OneFormer는 Mask2Former의 한계(과제별 별도 학습)를 극복하기 위해 task-conditioned joint training을 도입했다. 텍스트 기반 task token을 사용하여 한 번의 학습으로 3개 과제를 모두 처리한다.

**Mask2Former vs. SAM (2023)**: SAM은 promptable segmentation이라는 새로운 패러다임을 제시하며 class-agnostic 마스크 생성에 특화되어 있다. Mask2Former는 category-aware한 반면, SAM은 category 정보 없이 범용 마스크를 생성한다. 두 접근법은 상호보완적이다.

---

## 참고자료

1. Cheng, B., Misra, I., Schwing, A. G., Kirillov, A., & Girdhar, R. (2022). "Masked-attention Mask Transformer for Universal Image Segmentation." *CVPR 2022*. arXiv:2112.01527.
2. Cheng, B., Schwing, A. G., & Kirillov, A. (2021). "Per-Pixel Classification is Not All You Need for Semantic Segmentation." *NeurIPS 2021*.
3. Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020). "End-to-End Object Detection with Transformers." *ECCV 2020*.
4. Zhang, W., Pang, J., Chen, K., & Loy, C. C. (2021). "K-Net: Towards Unified Image Segmentation." *NeurIPS 2021*.
5. Wang, H., Zhu, Y., Adam, H., Yuille, A., & Chen, L.-C. (2021). "MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers." *CVPR 2021*.
6. Zhu, X., Su, W., Lu, L., Li, B., Wang, X., & Dai, J. (2021). "Deformable DETR: Deformable Transformers for End-to-End Object Detection." *ICLR 2021*.
7. Kirillov, A., Wu, Y., He, K., & Girshick, R. (2020). "PointRend: Image Segmentation as Rendering." *CVPR 2020*.
8. He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). "Mask R-CNN." *ICCV 2017*.
9. Liu, Z., et al. (2021). "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." *ICCV 2021*.
10. Jain, J., Li, J., Chiu, M. T., Hassani, A., Orber, N., & Shi, H. (2023). "OneFormer: One Transformer to Rule Universal Image Segmentation." *CVPR 2023*.
11. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., ... & Girshick, R. (2023). "Segment Anything." *ICCV 2023*.
12. Li, F., Zhang, H., Xu, H., Liu, S., Zhang, L., Ni, L. M., & Shum, H.-Y. (2023). "Mask DINO: Towards a Unified Transformer-based Framework for Object Detection and Segmentation." *CVPR 2023*.
13. Bao, H., Dong, L., & Wei, F. (2021). "BEiT: BERT Pre-Training of Image Transformers." *ICLR 2022*.
14. Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." *NeurIPS 2021*.
15. 논문 공식 페이지: https://bowenc0221.github.io/mask2former
