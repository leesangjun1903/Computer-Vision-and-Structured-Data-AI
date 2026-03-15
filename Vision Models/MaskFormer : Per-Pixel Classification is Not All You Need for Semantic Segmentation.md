# Per-Pixel Classification is Not All You Need for Semantic Segmentation

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
기존 시맨틱 세그멘테이션은 **per-pixel classification** (픽셀 단위 분류) 패러다임이 지배적이었고, **mask classification** (마스크 분류)은 인스턴스 세그멘테이션에만 사용되었다. 본 논문은 **mask classification이 시맨틱 세그멘테이션과 인스턴스 세그멘테이션을 동일한 모델·손실함수·학습 절차로 통합적으로 해결할 수 있을 만큼 충분히 일반적(general)인 패러다임**임을 주장한다.

### 주요 기여
1. **패러다임 전환**: 시맨틱 세그멘테이션을 per-pixel classification에서 mask classification으로 재정의하고, 이것이 더 우수한 성능을 낼 수 있음을 실증
2. **MaskFormer 모델 제안**: Transformer decoder 기반의 간결한 mask classification 모델로, 기존 per-pixel classification 모델을 seamless하게 변환 가능
3. **통합 프레임워크**: 시맨틱·panoptic 세그멘테이션을 모델 구조, 손실, 학습 파이프라인의 변경 없이 동일하게 처리
4. **SOTA 달성**: ADE20K 시맨틱 세그멘테이션 55.6 mIoU, COCO panoptic 세그멘테이션 52.7 PQ
5. **클래스 수 증가에 따른 우위**: 클래스 수가 많을수록 per-pixel classification 대비 mask classification의 이점이 커짐을 실험적으로 입증

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존의 세그멘테이션 연구는 두 가지 패러다임으로 분리되어 있었다:

- **시맨틱 세그멘테이션**: FCN 이후 per-pixel classification이 주류. 각 픽셀에 $K$개 클래스 중 하나를 할당
- **인스턴스 세그멘테이션/panoptic 세그멘테이션**: Mask R-CNN, DETR 등 mask classification 방식 사용

이러한 패러다임 불일치로 인해:
- 태스크마다 완전히 다른 모델 아키텍처, 손실 함수, 학습 절차가 필요
- Per-pixel classification은 고정된 수의 출력을 가정하여 가변 수의 인스턴스를 다루지 못함
- 기존 mask classification 모델(DETR, Mask R-CNN)은 바운딩 박스 예측에 의존하여 시맨틱 세그멘테이션에 부적합
- 클래스 수가 많을수록 per-pixel classification의 각 픽셀에서 fine-grained recognition이 어려워짐

### 2.2 제안하는 방법 (수식 포함)

#### Per-pixel classification 정의

Per-pixel classification에서 세그멘테이션 모델은 $H \times W$ 이미지의 모든 픽셀에 대해 $K$개 카테고리에 대한 확률 분포를 예측한다:

$$y = \{p_i \mid p_i \in \Delta^K\}_{i=1}^{H \cdot W}$$

여기서 $\Delta^K$는 $K$차원 확률 심플렉스이다. 학습 시 ground truth 레이블 $y^{\text{gt}} = \{y_i^{\text{gt}} \mid y_i^{\text{gt}} \in \{1, \ldots, K\}\}_{i=1}^{H \cdot W}$에 대해 per-pixel cross-entropy 손실을 적용한다:

$$\mathcal{L}_{\text{pixel-cls}}(y, y^{\text{gt}}) = \sum_{i=1}^{H \cdot W} -\log p_i(y_i^{\text{gt}})$$

#### Mask classification 정의

Mask classification은 세그멘테이션을 두 단계로 분리한다:
1. 이미지를 $N$개 영역으로 분할: $\{m_i \mid m_i \in [0,1]^{H \times W}\}_{i=1}^N$ (이진 마스크)
2. 각 영역에 $K$개 카테고리에 대한 확률 분포를 할당

출력은 $N$개의 확률-마스크 쌍의 집합으로 정의된다:

$$z = \{(p_i, m_i)\}_{i=1}^N$$

여기서 $p_i \in \Delta^{K+1}$은 $K$개 카테고리 + "no object" ($\varnothing$) 레이블을 포함한다. **중요한 점은 $N$이 $K$와 같을 필요가 없으며**, 동일 클래스에 대해 여러 마스크를 예측할 수 있어 인스턴스 세그멘테이션에도 적용 가능하다.

#### 매칭 (Matching)

예측 집합 $z$와 $N^{\text{gt}}$개의 ground truth 세그먼트 $z^{\text{gt}} = \{(c_j^{\text{gt}}, m_j^{\text{gt}}) \mid c_j^{\text{gt}} \in \{1, \ldots, K\}, m_j^{\text{gt}} \in \{0,1\}^{H \times W}\}_{j=1}^{N^{\text{gt}}}$ 간의 매칭 $\sigma$가 필요하다.

매칭 비용은 바운딩 박스가 아닌 **클래스와 마스크 예측**을 직접 사용한다:

$$-p_i(c_j^{\text{gt}}) + \mathcal{L}_{\text{mask}}(m_i, m_j^{\text{gt}})$$

#### 손실 함수

매칭이 주어진 후, 메인 mask classification 손실 $\mathcal{L}_{\text{mask-cls}}$는 cross-entropy 분류 손실과 이진 마스크 손실의 조합이다:

$$\mathcal{L}_{\text{mask-cls}}(z, z^{\text{gt}}) = \sum_{j=1}^{N} \left[ -\log p_{\sigma(j)}(c_j^{\text{gt}}) + \mathbb{1}_{c_j^{\text{gt}} \neq \varnothing} \mathcal{L}_{\text{mask}}(m_{\sigma(j)}, m_j^{\text{gt}}) \right]$$

마스크 손실 $\mathcal{L}_{\text{mask}}$는 focal loss와 dice loss의 선형 결합이다:

$$\mathcal{L}_{\text{mask}}(m, m^{\text{gt}}) = \lambda_{\text{focal}} \mathcal{L}_{\text{focal}}(m, m^{\text{gt}}) + \lambda_{\text{dice}} \mathcal{L}_{\text{dice}}(m, m^{\text{gt}})$$

여기서 $\lambda_{\text{focal}} = 20.0$, $\lambda_{\text{dice}} = 1.0$으로 설정한다.

#### 추론 (Inference)

**General inference** (panoptic 및 semantic 공용):

$$\arg\max_{i: c_i \neq \varnothing} p_i(c_i) \cdot m_i[h, w]$$

여기서 $c_i = \arg\max_{c \in \{1, \ldots, K, \varnothing\}} p_i(c)$

**Semantic inference** (시맨틱 세그멘테이션 전용, 행렬 곱으로 구현):

$$\arg\max_{c \in \{1, \ldots, K\}} \sum_{i=1}^{N} p_i(c) \cdot m_i[h, w]$$

이 방식은 여러 쿼리의 확률-마스크 쌍을 marginalization하여 더 나은 시맨틱 세그멘테이션 결과를 산출한다.

### 2.3 모델 구조

MaskFormer는 세 가지 모듈로 구성된다:

#### (1) Pixel-level module
- **Backbone**: ResNet 또는 Swin-Transformer를 사용하여 저해상도 이미지 특징 맵 $\mathcal{F} \in \mathbb{R}^{C_{\mathcal{F}} \times \frac{H}{S} \times \frac{W}{S}}$ 추출 (stride $S = 32$)
- **Pixel decoder**: FPN 기반의 경량 디코더로 특징 맵을 점진적으로 업샘플링하여 per-pixel 임베딩 $\mathcal{E}\_{\text{pixel}} \in \mathbb{R}^{C_{\mathcal{E}} \times H \times W}$ 생성 ($C_{\mathcal{E}} = 256$)

#### (2) Transformer module
- 표준 Transformer decoder (6개 레이어, 100개 쿼리가 기본값)
- $N$개의 학습 가능한 위치 인코딩(쿼리)과 이미지 특징 $\mathcal{F}$로부터 $N$개의 per-segment 임베딩 $\mathcal{Q} \in \mathbb{R}^{C_{\mathcal{Q}} \times N}$ 계산
- 모든 예측을 병렬로 생성 (DETR와 유사)

#### (3) Segmentation module
- **클래스 예측**: Linear classifier + softmax → $\{p_i \in \Delta^{K+1}\}_{i=1}^N$
- **마스크 임베딩**: 2-hidden-layer MLP → $\mathcal{E}\_{\text{mask}} \in \mathbb{R}^{C_{\mathcal{E}} \times N}$
- **마스크 예측**: per-pixel 임베딩과 마스크 임베딩의 내적 + sigmoid:

$$m_i[h, w] = \text{sigmoid}(\mathcal{E}_{\text{mask}}[:, i]^\top \cdot \mathcal{E}_{\text{pixel}}[:, h, w])$$

핵심적으로, 마스크 예측이 **상호 배타적(mutually exclusive)이 아님** — softmax가 아닌 sigmoid를 사용하여 겹치는 마스크를 허용한다.

### 2.4 성능 향상

#### 시맨틱 세그멘테이션 (ADE20K val, 150 classes)

| 방법 | Backbone | mIoU (m.s.) | #params | FLOPs |
|------|----------|-------------|---------|-------|
| DeepLabV3+ | R101c | 46.4 | 63M | 255G |
| **MaskFormer** | **R101c** | **48.1** | **60M** | **80G** |
| Swin-UperNet | Swin-L† | 53.5 | 234M | 647G |
| **MaskFormer** | **Swin-L†** | **55.6** | **212M** | **375G** |

- Swin-L backbone으로 **55.6 mIoU** 달성 (기존 SOTA 대비 +2.1 mIoU)
- 파라미터 10% 감소, FLOPs 40% 감소

#### 클래스 수에 따른 개선 폭 (PerPixelBaseline+ 대비)

| 데이터셋 | 클래스 수 | mIoU 개선 |
|---------|----------|---------|
| Cityscapes | 19 | +0.0 |
| ADE20K | 150 | +2.6 |
| COCO-Stuff | 171 | +2.9 |
| ADE20K-Full | 847 | +3.5 |

**클래스 수가 증가할수록 mask classification의 이점이 명확하게 증가**한다.

#### Panoptic 세그멘테이션 (COCO panoptic val)

| 방법 | Backbone | PQ | PQ $^{\text{Th}}$ | PQ $^{\text{St}}$ |
|------|----------|-----|------|------|
| DETR | R50 + 6 Enc | 43.4 | 48.2 | 36.3 |
| **MaskFormer** | **R50 + 6 Enc** | **46.5** | **51.0** | **39.8** |
| Max-DeepLab | Max-L | 51.1 | 57.0 | 42.2 |
| **MaskFormer** | **Swin-L†** | **52.7** | **58.5** | **44.0** |

- DETR 대비 +3.1 PQ (같은 backbone)
- Max-DeepLab 대비 +1.6 PQ, 복잡한 auxiliary loss 불필요

### 2.5 한계

1. **소수 클래스 데이터셋에서의 한계**: Cityscapes(19 classes)에서 per-pixel classification과 동등한 성능(mIoU 기준). 클래스 인식이 쉬운 데이터셋에서는 마스크 품질(SQ $^{\text{St}}$ )이 per-pixel 방식에 비해 약간 뒤처짐
2. **마스크 품질 vs 인식 품질 트레이드오프**: MaskFormer는 인식 품질(RQ $^{\text{St}}$ )에서 우수하나 픽셀 수준 세그멘테이션 품질(SQ $^{\text{St}}$ )에서 약간 열세
3. **쿼리 수 민감성**: 쿼리 수가 너무 많으면(1000개) 성능이 크게 하락 (ADE20K에서 35.4 mIoU)
4. **DETR 기반 아키텍처의 한계 계승**: Transformer decoder의 수렴 속도가 느린 문제, 학습에 많은 연산 자원 필요
5. **쿼리의 카테고리 그룹핑에 대한 해석 가능성 부족**: 쿼리가 어떤 기준으로 카테고리를 그룹핑하는지 명확한 패턴을 발견하지 못함

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 다양한 데이터셋에 걸친 일반화

MaskFormer는 **5개의 시맨틱 세그멘테이션 데이터셋과 2개의 panoptic 세그멘테이션 데이터셋**에서 일관된 성능을 보인다:
- 소규모 클래스(Cityscapes, 19 classes)부터 대규모 클래스(ADE20K-Full, 847 classes)까지
- 저해상도부터 고해상도(Mapillary Vistas, 최대 4000×6000)까지

### 3.2 태스크 간 일반화

핵심적인 일반화 성능은 **모델, 손실, 학습 절차의 변경 없이** 시맨틱과 panoptic 세그멘테이션을 동시에 처리할 수 있다는 점이다. ground truth 어노테이션의 유형(카테고리 영역 마스크 vs 인스턴스 마스크)만 다르면 된다.

### 3.3 백본 아키텍처에 대한 일반화

- CNN 백본(ResNet-50/101) 및 Transformer 백본(Swin-T/S/B/L) 모두 호환
- 기존 per-pixel classification 모델을 mask classification으로 **seamless하게 변환** 가능

### 3.4 쿼리 수의 데이터셋 독립성

100개의 쿼리가 다양한 데이터셋(150~847 클래스)에서 일관되게 최적 성능을 보인다. 이는 쿼리 수를 데이터셋이나 클래스 수에 맞춰 조정할 필요가 적음을 의미한다. 이미지당 평균 존재 클래스 수가 데이터셋 간 유사하다는 관찰(ADE20K: 8.2, COCO-Stuff: 6.6, ADE20K-Full: 9.1)이 이를 뒷받침한다.

### 3.5 대규모 어휘(Large Vocabulary)로의 확장성

ADE20K-Full(847 classes)에서 MaskFormer는 per-pixel classification 대비:
- **성능 우위**: +3.5 mIoU (PerPixelBaseline+ 대비)
- **메모리 효율성**: 6,529M vs 26,698M (PerPixelBaseline+)

마스크 수($N$)가 클래스 수($K$)와 분리되어 있어, 클래스 수가 수천 개에 달하는 현실 세계 세그멘테이션 문제에서도 확장 가능하다.

### 3.6 매칭 방식에 의한 유연성

Bipartite matching은 fixed matching보다 우수하며(+0.5 mIoU, +3.1 PQ $^{\text{St}}$ ), 총 클래스 수보다 적은 수의 마스크를 예측할 수 있는 유연성을 제공한다. 이는 실제 이미지에서 모든 클래스가 등장하지 않는 현실적 상황에 더 적합하다.

### 3.7 마스크 기반 매칭의 우월성

바운딩 박스 기반 매칭 대비 마스크 기반 매칭이 우수하다(Table 5: 46.5 vs 43.7 PQ). 특히 "stuff" 카테고리에서 큰 개선이 이루어지며, 이는 바운딩 박스로 표현하기 모호한 영역을 마스크가 더 정확하게 포착할 수 있음을 보여준다.

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

1. **세그멘테이션 패러다임 통합**: MaskFormer는 시맨틱, 인스턴스, panoptic 세그멘테이션을 단일 프레임워크로 통합하는 길을 열었다. 이는 이후 **Mask2Former** (Cheng et al., CVPR 2022), **OneFormer** (Jain et al., CVPR 2023) 등 후속 연구의 직접적 기반이 되었다.

2. **Query-based 세그멘테이션의 표준화**: 학습 가능한 쿼리를 통한 set prediction 방식이 세그멘테이션의 새로운 표준으로 자리잡았다.

3. **바운딩 박스 의존성 제거**: 세그멘테이션에서 박스 예측의 필요성을 제거하여, stuff 클래스와 비정형 영역의 세그멘테이션 성능을 크게 향상시켰다.

4. **대규모 어휘 세그멘테이션 가능성 확장**: 마스크 수와 클래스 수의 분리는 open-vocabulary segmentation 연구의 토대를 마련했다.

### 4.2 향후 연구 시 고려할 점

1. **마스크 품질 개선**: MaskFormer는 인식 품질(RQ)에서 우수하지만 세그멘테이션 품질(SQ)에서 개선 여지가 있다. 고해상도 마스크 예측을 위한 multi-scale deformable attention 등의 도입이 필요하다.

2. **Pixel decoder 강화**: 논문에서는 경량 FPN을 사용했으나, 더 강력한 pixel decoder가 마스크 품질을 향상시킬 수 있다.

3. **학습 효율성**: DETR 기반 아키텍처의 느린 수렴 문제를 해결하기 위한 deformable attention, 효율적인 쿼리 초기화 전략 등이 필요하다.

4. **Open-vocabulary/Zero-shot 확장**: 마스크 분류 패러다임은 CLIP 등 vision-language 모델과의 통합을 통해 open-vocabulary 세그멘테이션으로 자연스럽게 확장 가능하다.

5. **비디오 세그멘테이션으로의 확장**: 쿼리 기반 접근법은 프레임 간 객체 추적과 결합하여 비디오 세그멘테이션에 적용 가능하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 Mask2Former (Cheng et al., CVPR 2022)

MaskFormer의 직접적 후속 연구로, 다음과 같은 개선을 도입:
- **Masked attention**: Transformer decoder의 cross-attention에서 예측 마스크 영역에만 attention을 수행하여 수렴 속도와 성능 모두 개선
- **Multi-scale high-resolution features**: Pixel decoder에 deformable attention을 적용한 multi-scale deformable Transformer encoder 도입
- **Optimization improvements**: Query dropout, 학습 가능한 쿼리 개선

| 모델 | ADE20K mIoU | COCO PQ |
|------|-------------|---------|
| MaskFormer | 55.6 | 52.7 |
| **Mask2Former** | **57.6** | **57.8** |

Mask2Former는 시맨틱, 인스턴스, panoptic 세그멘테이션 모두에서 MaskFormer를 크게 상회하며, "universal image segmentation"이라는 개념을 더욱 강화했다.

### 5.2 OneFormer (Jain et al., CVPR 2023)

- MaskFormer/Mask2Former를 확장하여 **단일 모델이 하나의 학습으로** 시맨틱, 인스턴스, panoptic을 동시에 학습
- **Task-conditioned joint training**: 태스크 토큰을 쿼리에 결합하여 태스크별 전문화
- **Text-based query initialization**: 텍스트 기반 쿼리 초기화로 더 나은 일반화

### 5.3 DETR 계열의 발전

| 모델 | 연도 | 주요 개선점 |
|------|------|-----------|
| DETR (Carion et al.) | 2020 | Set prediction + Transformer decoder 기반 |
| Deformable DETR | 2021 | Deformable attention으로 수렴 속도 개선 |
| MaskFormer | 2021 | Box-free mask classification |
| Mask2Former | 2022 | Masked attention + multi-scale features |
| DINO | 2023 | Denoising training + contrastive query 초기화 |

### 5.4 Open-Vocabulary Segmentation

MaskFormer의 마스크 분류 패러다임은 open-vocabulary segmentation으로의 확장을 촉진했다:

- **OpenSeg** (Ghiasi et al., ECCV 2022): Region-word alignment을 통한 open-vocabulary segmentation
- **X-Decoder** (Zou et al., CVPR 2023): 텍스트 쿼리와 mask classification을 결합한 범용 디코더
- **ODISE** (Xu et al., CVPR 2023): Diffusion model의 특징과 mask classification을 결합
- **FC-CLIP** (Yu et al., NeurIPS 2023): Frozen CLIP backbone + mask classification으로 효율적인 open-vocabulary segmentation

이들 연구는 MaskFormer가 제시한 **클래스 수와 마스크 수의 분리**라는 핵심 아이디어를 활용하여, 학습 시 보지 못한 카테고리에 대해서도 세그멘테이션이 가능하도록 확장했다.

### 5.5 SAM (Segment Anything Model, Kirillov et al., ICCV 2023)

- 대규모 데이터(SA-1B, 11M 이미지)로 학습된 promptable segmentation 모델
- MaskFormer와 유사하게 마스크를 직접 예측하지만, 클래스 예측 없이 **class-agnostic** 마스크 생성에 초점
- MaskFormer의 mask classification과 SAM의 mask generation을 결합하는 연구(예: Semantic-SAM, Grounded-SAM)가 활발히 진행 중

### 5.6 비교 요약

| 연구 | 패러다임 | 시맨틱 | 인스턴스 | Panoptic | Open-vocab |
|------|---------|--------|----------|----------|------------|
| MaskFormer (2021) | Mask classification | ✓ | ✗ (직접 평가 없음) | ✓ | ✗ |
| Mask2Former (2022) | Masked attention + Mask classification | ✓ | ✓ | ✓ | ✗ |
| OneFormer (2023) | Task-conditioned mask classification | ✓ | ✓ | ✓ | ✗ |
| X-Decoder (2023) | Text query + mask classification | ✓ | ✓ | ✓ | ✓ |
| SAM (2023) | Promptable mask generation | ✗ (class-agnostic) | ✗ | ✗ | ✗ |

---

## 참고자료

1. **Cheng, B., Schwing, A.G., & Kirillov, A.** (2021). "Per-Pixel Classification is Not All You Need for Semantic Segmentation." *NeurIPS 2021*. arXiv:2107.06278. (본 논문)
2. **Cheng, B., Misra, I., Schwing, A.G., Kirillov, A., & Girdhar, R.** (2022). "Masked-attention Mask Transformer for Universal Image Segmentation." *CVPR 2022*. arXiv:2112.01527.
3. **Jain, J., Li, J., Chiu, M., Hassani, A., Orber, N., & Shi, H.** (2023). "OneFormer: One Transformer to Rule Universal Image Segmentation." *CVPR 2023*. arXiv:2211.06220.
4. **Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S.** (2020). "End-to-End Object Detection with Transformers." *ECCV 2020*.
5. **Wang, H., Zhu, Y., Adam, H., Yuille, A., & Chen, L.-C.** (2021). "MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers." *CVPR 2021*.
6. **Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W.-Y., Dollár, P., & Girshick, R.** (2023). "Segment Anything." *ICCV 2023*. arXiv:2304.02643.
7. **Zou, X., Dou, Z.-Y., Yang, J., Gan, Z., Li, L., Li, C., Dai, X., Beber, H., Wang, J., Yuan, L., Peng, N., Wang, L., Lee, Y.J., & Gao, J.** (2023). "Generalized Decoding for Pixel, Image, and Language." *CVPR 2023*.
8. **Xu, J., Liu, S., Vahdat, A., Byeon, W., Wang, X., & De Mello, S.** (2023). "Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models." *CVPR 2023*.
9. **Yu, Q., He, J., Deng, X., Shen, X., & Chen, L.-C.** (2023). "Convolutions Die Hard: Open-Vocabulary Segmentation with Single Frozen Convolutional CLIP." *NeurIPS 2023*.
10. **Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B.** (2021). "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." *ICCV 2021*. arXiv:2103.14030.
11. **Ghiasi, G., Gu, X., Cui, Y., & Lin, T.-Y.** (2022). "Scaling Open-Vocabulary Image Segmentation with Image-Level Labels." *ECCV 2022*.
12. MaskFormer 프로젝트 페이지: https://bowenc0221.github.io/maskformer
