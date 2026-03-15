# OneFormer: One Transformer to Rule Universal Image Segmentation

---

## 1. 핵심 주장 및 주요 기여 요약

**OneFormer**는 semantic, instance, panoptic segmentation 세 가지 이미지 분할 과제를 **단일 아키텍처, 단일 모델, 단일 데이터셋**으로 통합하는 최초의 multi-task universal image segmentation 프레임워크이다. 핵심 주장은 기존의 panoptic 아키텍처(Mask2Former, MaskFormer 등)가 각 과제별로 개별 학습해야 최고 성능을 달성할 수 있는 "반(半)범용적(semi-universal)" 한계를 가지는 반면, OneFormer는 **한 번의 학습(train-once)**으로 세 과제 모두에서 기존 개별 학습 모델을 능가한다는 것이다.

### 주요 기여:

1. **Task-Conditioned Joint Training Strategy**: panoptic annotation에서 semantic/instance label을 파생시켜 균일하게 샘플링하는 multi-task 공동 학습 전략 제안
2. **Task Token 도입**: "the task is {task}" 형태의 텍스트 입력을 토큰화하여 모델을 task-dynamic하게 조건화
3. **Query-Text Contrastive Loss**: object query와 text query 간 대조 학습 손실을 통해 inter-task 및 inter-class 구별력 강화
4. ADE20K, Cityscapes, COCO에서 개별 학습된 Mask2Former를 **1/3의 리소스**로 능가

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 panoptic 아키텍처(Mask2Former [12], MaskFormer [13], K-Net [61])는 동일한 아키텍처로 세 과제를 수행할 수 있지만, **각 과제별로 개별 학습해야 최적 성능**을 달성한다. 예를 들어, Mask2Former는 ADE20K에서 각 과제에 160K iterations씩 총 480K iterations의 학습이 필요하며, 세 개의 별도 모델을 저장·운용해야 한다.

이 문제의 근본 원인은 기존 아키텍처에 **task guidance가 부재**하여, 공동 학습 시 세 과제 간 도메인 차이(inter-task domain differences)를 효과적으로 학습하지 못하기 때문이다 (Tab. 7에서 Mask2Former-Joint의 성능 저하가 이를 실증).

### 2.2 제안하는 방법

#### (A) Task-Conditioned Joint Training

학습 시 각 이미지에 대해 $\texttt{task} \in \{\texttt{panoptic}, \texttt{instance}, \texttt{semantic}\}$를 **균일 확률** $p = 1/3$로 샘플링하고, panoptic annotation으로부터 해당 task의 GT label을 파생한다.

- **Semantic task**: 이미지 내 각 클래스에 대해 하나의 amorphous binary mask
- **Instance task**: thing 클래스에 대해서만 non-overlapping binary masks (stuff 무시)
- **Panoptic task**: stuff 클래스에 대한 amorphous mask + thing 클래스에 대한 non-overlapping masks

각 binary mask에 대해 템플릿 "a photo with a {CLS}"로 텍스트 리스트 $\mathbf{T}\_{\text{list}}$를 구성하고, "a/an {task} photo" 항목으로 패딩하여 고정 길이 $N_{\text{text}}$ 의 $\mathbf{T}_{\text{pad}}$를 생성한다.

#### (B) Task Token과 Query Representations

**Task Input**: "the task is {task}" 텍스트를 토큰화하고 단일 선형 레이어 + LayerNorm을 통해 **task token** $\mathbf{Q}_{\text{task}}$를 생성한다.

**Object Queries** $\mathbf{Q}$:
- 초기화: $\mathbf{Q}' = \underbrace{\mathbf{Q}\_{\text{task}} \oplus \mathbf{Q}\_{\text{task}} \oplus \cdots}_{N-1 \text{ repetitions}}$
- 2-layer transformer에서 flattened 1/4-scale 이미지 특징과의 상호작용을 통해 $\mathbf{Q}'$를 업데이트
- $\mathbf{Q}_{\text{task}}$를 concatenate하여 최종 $N$개의 task-conditioned object queries $\mathbf{Q}$ 생성

**Text Queries** $\mathbf{Q}_{\text{text}}$ (학습 시에만 사용):
- $\mathbf{T}\_{\text{pad}}$를 6-layer transformer text encoder [57]로 인코딩하여 $N_{\text{text}}$개의 임베딩 생성
- $N_{\text{ctx}}$개의 학습 가능한 텍스트 컨텍스트 임베딩 $\mathbf{Q}\_{\text{ctx}}$를 concatenate하여 최종 $N$개의 text queries $\mathbf{Q}_{\text{text}}$ 생성

#### (C) Query-Text Contrastive Loss (수식)

Batch of $B$ object-text query pairs $\{(q_i^{obj}, q_i^{txt})\}_{i=1}^{B}$에 대해:

$$
\mathcal{L}_{\mathbf{Q} \to \mathbf{Q}_{\text{text}}} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(q_i^{obj} \odot q_i^{txt} / \tau)}{\sum_{j=1}^{B} \exp(q_i^{obj} \odot q_j^{txt} / \tau)}
$$

$$
\mathcal{L}_{\mathbf{Q}_{\text{text}} \to \mathbf{Q}} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(q_i^{txt} \odot q_i^{obj} / \tau)}{\sum_{j=1}^{B} \exp(q_i^{txt} \odot q_j^{obj} / \tau)}
$$

$$
\mathcal{L}_{\mathbf{Q} \leftrightarrow \mathbf{Q}_{\text{text}}} = \mathcal{L}_{\mathbf{Q} \to \mathbf{Q}_{\text{text}}} + \mathcal{L}_{\mathbf{Q}_{\text{text}} \to \mathbf{Q}}
$$

여기서 $\odot$는 내적(dot product), $\tau$는 학습 가능한 temperature parameter이다.

#### (D) 최종 손실 함수

$$
\mathcal{L}_{\text{final}} = \lambda_{\mathbf{Q} \leftrightarrow \mathbf{Q}_{\text{text}}} \mathcal{L}_{\mathbf{Q} \leftrightarrow \mathbf{Q}_{\text{text}}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{bce}} \mathcal{L}_{\text{bce}} + \lambda_{\text{dice}} \mathcal{L}_{\text{dice}}
$$

여기서 $\lambda_{\mathbf{Q} \leftrightarrow \mathbf{Q}\_{\text{text}}} = 0.5$, $\lambda_{\text{cls}} = 2$, $\lambda_{\text{bce}} = 5$, $\lambda_{\text{dice}} = 5$로 설정. No-object 예측에 대해서는 $\lambda_{\text{cls}} = 0.1$. 예측과 GT 간 bipartite matching [3, 13]을 사용.

### 2.3 모델 구조

OneFormer의 전체 구조는 세 부분으로 구성된다 (Fig. 2 참조):

**(a) Multi-Scale Feature Modeling:**
- ImageNet 사전학습 backbone (Swin-L, ConvNeXt-L, DiNAT-L 등)으로 multi-scale 특징 추출
- 6개의 MSDeformAttn [64] 기반 pixel decoder로 1/8, 1/16, 1/32 해상도 특징을 점진적으로 업샘플링하여 최종 1/4 해상도 특징 $F_{1/4}$ 생성
- 모든 특징 차원은 256으로 통일

**(b) Unified Task-Conditioned Query Formulation:**
- Task token $\mathbf{Q}_{\text{task}}$의 $N-1$회 반복으로 object queries 초기화
- 2-layer transformer에서 1/4-scale 특징과 상호작용 후 $\mathbf{Q}_{\text{task}}$와 concatenate
- Text mapper (학습 시만 사용)로 $\mathbf{Q}_{\text{text}}$ 생성, contrastive loss 계산

**(c) Task-Dynamic Mask and Class Prediction:**
- $L = 3$ 반복의 multi-scale transformer decoder (총 $3L = 9$ stages)
- 각 stage: masked cross-attention (CA, 1/8, 1/16, 1/32 해상도 교대 사용) → self-attention (SA) → FFN
- 최종 query 출력을 $K+1$ 차원으로 매핑하여 class prediction 생성
- $\mathbf{Q}$와 $F_{1/4}$ 간 einsum 연산으로 mask prediction 생성

### 2.4 성능 향상

**ADE20K val** (Swin-L backbone, 640×640, 160K iters):

| 모델 | 학습 방식 | PQ | AP | mIoU (s.s.) |
|------|---------|-----|-----|-------------|
| Mask2Former-Panoptic | 개별 | 48.7 | 34.2 | 54.5 |
| Mask2Former-Instance | 개별 | — | 34.9 | — |
| Mask2Former-Semantic | 개별 | — | — | 56.1 |
| **OneFormer** | **공동** | **49.8** | **35.9** | **57.0** |

- 단일 모델로 세 과제 모두에서 개별 학습 Mask2Former를 능가
- DiNAT-L backbone 사용 시 PQ 50.5, mIoU 58.3 달성

**Cityscapes val** (Swin-L): PQ 67.2 (+0.6%), AP 45.6 (+1.9%), mIoU 83.0

**COCO val2017** (Swin-L): PQ 57.9 (+0.1%), AP 49.0, mIoU 67.4

**Ablation 결과 핵심:**
- Contrastive loss 제거 시: PQ -8.4%, AP -3.2% (Tab. 5)
- Task token 제거 시: AP -2.3% (Tab. 4)
- Learnable text context 제거 시: PQ -4.5% (Tab. 4)
- Mask2Former-Joint (동일 공동 학습) 대비: PQ +1.1%, AP +2.2%, mIoU +0.8% (Tab. 7)

### 2.5 한계

1. **COCO 데이터셋의 annotation 불일치**: panoptic과 instance annotation 간 심각한 discrepancy가 존재하여 공정한 평가가 어려움 (Appendix F, Fig. III, IV)
2. **Text template 의존성**: 입력 텍스트 템플릿 선택이 성능에 영향을 미치며, 최적 템플릿 탐색이 충분히 이루어지지 않음 (Tab. 6)
3. **추론 시 task 지정 필요**: 추론 시 사용자가 명시적으로 task를 지정해야 하므로, 완전한 자동 범용 추론과는 거리가 있음
4. **대규모 backbone/extra data 대비 성능 한계**: BEiT-3(1.9B params) + extra data 사용 모델 대비 semantic segmentation에서는 여전히 격차 존재 (62.0 vs 58.3 mIoU)
5. **학습 시 text mapper 오버헤드**: 추론 시에는 제거 가능하나, 학습 과정에서 text encoder 및 contrastive loss 계산으로 인한 추가 연산 비용 발생

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

OneFormer의 일반화 성능은 여러 메커니즘을 통해 달성되고 향상될 수 있다:

### 3.1 Task-Conditioned Architecture의 일반화 효과

Task token을 통한 conditioning은 모델이 **단일 파라미터 세트로 세 가지 서로 다른 출력 분포**를 학습하게 한다. Tab. 8의 결과는 이를 실증한다:
- `task=instance`일 때 $PQ^{St}$가 1.5%로 급감하고 $PQ^{Th}$는 유지 → thing에 집중
- `task=semantic`일 때 $PQ^{Th}$, AP가 급감하고 $PQ^{St}$는 유지 → stuff에 집중

이는 모델이 task 간 차이를 **동적으로 학습**하여 각 과제에 맞는 예측을 생성함을 보여준다.

### 3.2 Contrastive Learning의 일반화 기여

Query-text contrastive loss는 두 가지 측면에서 일반화를 촉진한다:
1. **Inter-task 구별**: 동일 이미지에서 서로 다른 task의 query가 서로 다른 representation space에 매핑
2. **Inter-class 구별**: 유사한 클래스 간 혼동(category misclassification) 감소 (Fig. 5: "wall" vs "fence", "vegetation" vs "terrain" 등)

Tab. 5에서 contrastive loss 없이 PQ가 8.4% 하락하는 것은 이 손실이 multi-task 일반화의 핵심임을 입증한다.

### 3.3 Panoptic Annotation의 통합적 활용

Panoptic annotation이 semantic과 instance 정보를 모두 포함한다는 특성을 활용하여, **단일 annotation 세트만으로** 세 과제의 GT를 파생시킨다. 이는:
- 학습 데이터 효율성을 극대화
- Annotation 간 일관성을 보장 (COCO 제외)
- Panoptic segmentation [29]의 본래 통합 목표를 실현

### 3.4 일반화 향상을 위한 추가 가능성

1. **다중 데이터셋 공동 학습**: LMSeg [1]처럼 여러 데이터셋의 taxonomy를 통합하여 cross-dataset 일반화 향상 가능
2. **더 강력한 vision-language 사전학습 모델 활용**: CLIP, BEiT-3 등의 대규모 사전학습 representation 활용
3. **Text template 최적화**: 프롬프트 엔지니어링 또는 learnable prompt를 통한 텍스트 표현 개선
4. **Open-vocabulary 확장**: 학습 시 보지 못한 클래스에 대한 zero-shot 일반화
5. **더 다양한 task 확장**: depth estimation, edge detection 등으로의 확장을 통한 범용성 강화

### 3.5 Backbone 교체에 따른 일반화

다양한 backbone(Swin-L, ConvNeXt-L/XL, DiNAT-L)에서 일관되게 높은 성능을 보이는 것은 OneFormer의 task-conditioning 메커니즘이 **backbone-agnostic**한 일반화 능력을 가짐을 시사한다.

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 향후 연구에 미치는 영향

1. **"Train-Once" 패러다임의 확립**: 이미지 분할 분야에서 단일 모델의 multi-task 학습이 개별 학습을 능가할 수 있음을 입증, 이후 연구의 방향성 제시
2. **Task Conditioning의 보편화**: 텍스트 기반 task token을 통한 모델 조건화는 다른 multi-task 비전 과제에도 적용 가능한 범용적 설계 원리
3. **리소스 효율성 제고**: 학습 시간, 저장 공간, 추론 호스팅을 1/3로 줄여 실용적 배포 용이성 증대
4. **Vision-Language 융합의 가속**: query-text contrastive learning의 성공은 비전-언어 모델의 segmentation 적용을 촉진
5. **Annotation 통합 연구 촉발**: COCO의 panoptic-instance annotation 불일치 문제를 조명하여 데이터셋 정비 필요성 제기

### 4.2 향후 연구 시 고려할 점

1. **완전 자동 task 추론**: 현재는 추론 시 task를 명시적으로 지정해야 함. 입력 이미지에서 자동으로 적절한 task를 결정하거나, task-agnostic한 통합 출력 생성 연구 필요
2. **Open-vocabulary/Zero-shot 확장**: 학습 시 보지 못한 클래스에 대한 분할 능력 확보
3. **Video segmentation 확장**: 시간 축으로의 확장 (video panoptic segmentation 등)
4. **효율성 개선**: Text mapper는 추론 시 제거 가능하나, 학습 시의 추가 비용 최소화 연구
5. **Cross-dataset 일반화**: 단일 데이터셋 학습의 한계를 넘어 다중 데이터셋 공동 학습 시 성능 향상
6. **대규모 Foundation Model과의 통합**: SAM (Segment Anything Model), SEEM 등과의 연계
7. **3D/Point Cloud Segmentation**으로의 확장 가능성

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 특징 | OneFormer와의 비교 |
|------|------|----------|------------------|
| **MaskFormer** [13] | 2021 (NeurIPS) | Semantic seg.을 mask classification으로 재정의 | OneFormer의 기반 아키텍처. 단일 과제 학습만 지원 |
| **Mask2Former** [12] | 2022 (CVPR) | Masked cross-attention, multi-scale deformable attention, 3개 과제 SOTA | Semi-universal: 각 과제별 개별 학습 필요. OneFormer가 단일 모델로 능가 |
| **K-Net** [61] | 2021 (NeurIPS) | Dynamic learnable kernels, bipartite matching | CNN 기반. 개별 학습 필요. OneFormer보다 낮은 성능 |
| **kMaX-DeepLab** [60] | 2022 (ECCV) | K-means clustering 기반 mask transformer | Panoptic 특화. 공동 학습 미지원. PQ에서는 경쟁적이나 multi-task 통합 불가 |
| **Mask DINO** [32] | 2022 (arXiv) | Detection과 segmentation 통합, DINO 기반 | Extra data 사용 시 높은 성능. Task 통합 관점이 다름 (detection + segmentation) |
| **SAM (Segment Anything)** [Kirillov et al., 2023, ICCV] | 2023 | Promptable segmentation, 11M 이미지 학습, zero-shot 일반화 | Class-agnostic mask 생성에 특화. Semantic label 예측 불가. OneFormer와 상보적 |
| **SEEM** [Zou et al., 2023] | 2023 | Multi-modal prompt (text, click, box, audio) 통합 segmentation | Open-vocabulary + interactive. OneFormer보다 유연한 입력 지원하나, 벤치마크 SOTA 비교 어려움 |
| **X-Decoder** [Zou et al., 2023, CVPR] | 2023 | Generalized decoding for pixel, region, language | Open-vocabulary seg. + captioning 통합. OneFormer의 task conditioning 개념을 확장 |
| **Panoptic SegFormer** [33] | 2022 (CVPR) | Panoptic 특화 transformer | 개별 학습. COCO PQ 55.8 vs OneFormer 57.9 |
| **LMSeg** [1] | 2023 (under review) | Multi-dataset taxonomy 통합을 위한 text 활용 | 다중 데이터셋 학습에 초점. OneFormer는 다중 과제 통합에 초점. 상호 보완적 |

### 주요 트렌드 분석:

OneFormer 이후 segmentation 연구의 주요 방향은 다음과 같이 전개되고 있다:

1. **Foundation Model 기반 접근**: SAM, SEEM 등은 대규모 데이터로 학습한 범용 segmentation 모델을 제안하며, OneFormer의 "통합" 철학을 더 넓은 범위로 확장
2. **Open-Vocabulary Segmentation**: 학습 시 보지 못한 클래스에 대한 분할 능력이 핵심 연구 주제로 부상
3. **Vision-Language 통합 심화**: OneFormer의 query-text contrastive learning이 선구적 역할을 했으며, 이후 연구들이 더 깊은 vision-language 통합을 추구

---

## 참고자료

1. Jain, J., Li, J., Chiu, M., Hassani, A., Orlov, N., & Shi, H. (2022). "OneFormer: One Transformer to Rule Universal Image Segmentation." *arXiv:2211.06220v2*. https://github.com/SHI-Labs/OneFormer
2. Cheng, B., Misra, I., Schwing, A. G., Kirillov, A., & Girdhar, R. (2022). "Masked-attention Mask Transformer for Universal Image Segmentation." *CVPR 2022*.
3. Cheng, B., Schwing, A. G., & Kirillov, A. (2021). "Per-Pixel Classification is Not All You Need for Semantic Segmentation." *NeurIPS 2021*.
4. Carion, N. et al. (2020). "End-to-End Object Detection with Transformers (DETR)." *ECCV 2020*.
5. Kirillov, A. et al. (2019). "Panoptic Segmentation." *CVPR 2019*.
6. Zhang, W. et al. (2021). "K-Net: Towards Unified Image Segmentation." *NeurIPS 2021*.
7. Yu, Q. et al. (2022). "k-means Mask Transformer (kMaX-DeepLab)." *ECCV 2022*.
8. Xu, J. et al. (2022). "GroupViT: Semantic Segmentation Emerges from Text Supervision." *CVPR 2022*.
9. Li, F. et al. (2022). "Mask DINO: Towards a Unified Transformer-based Framework for Object Detection and Segmentation." *arXiv 2022*.
10. Kirillov, A. et al. (2023). "Segment Anything." *ICCV 2023*.
11. Zou, X. et al. (2023). "Segment Everything Everywhere All at Once (SEEM)." *NeurIPS 2023*.
12. Zou, X. et al. (2023). "Generalized Decoding for Pixel, Image, and Language (X-Decoder)." *CVPR 2023*.
13. Radford, A. et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision (CLIP)." *ICML 2021*.
14. Zhou, K. et al. (2022). "Conditional Prompt Learning for Vision-Language Models (CoCoOp)." *CVPR 2022*.
15. Li, Z. et al. (2022). "Panoptic SegFormer." *CVPR 2022*.
