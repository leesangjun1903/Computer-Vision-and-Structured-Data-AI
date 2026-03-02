# DPT : Vision Transformers for Dense Prediction

---

## 1. 핵심 주장과 주요 기여 요약

**핵심 주장:** Vision Transformer(ViT)를 dense prediction 태스크의 백본으로 활용하면, 기존 fully-convolutional network(FCN) 대비 (1) 더 세밀한(fine-grained) 예측과 (2) 더 전역적으로 일관된(globally coherent) 예측을 동시에 달성할 수 있다.

**주요 기여:**
1. **DPT(Dense Prediction Transformer) 아키텍처 제안:** ViT를 인코더로, convolutional decoder를 결합한 새로운 인코더-디코더 구조를 제안하여 dense prediction에 transformer를 최초로 효과적으로 적용.
2. **Reassemble 모듈 설계:** ViT의 bag-of-words 표현(토큰)을 다양한 해상도의 이미지-유사 특징 맵으로 재구성하는 3단계 연산(Read → Concatenate → Resample) 제안.
3. **대규모 데이터 활용 시 탁월한 성능:** 단안 깊이 추정에서 최대 28%의 상대적 성능 향상, ADE20K 시맨틱 세그멘테이션에서 49.02% mIoU로 새로운 SOTA 달성.
4. **소규모 데이터셋 파인튜닝 가능성 입증:** NYUv2, KITTI, Pascal Context 등 소규모 데이터셋에서도 SOTA 달성.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 dense prediction 아키텍처는 convolutional backbone에 의존하며, 다음과 같은 근본적 한계를 가진다:

- **점진적 다운샘플링으로 인한 해상도 및 세부 정보 손실:** 깊은 레이어로 갈수록 특징 해상도가 크게 감소하여, 디코더에서 이를 완전히 복원하기 어려움.
- **제한된 수용 영역(receptive field):** 개별 convolution은 국소적(local) 연산이므로, 충분한 전역 맥락을 확보하려면 매우 깊은 네트워크를 쌓아야 하며, 이는 메모리와 계산 비용 증가를 초래.
- **표현력의 제약:** convolution은 정의상 선형 연산자이며, 제한된 수용 영역과 표현력 때문에 전역적 일관성 있는 예측이 어려움.

### 2.2 제안하는 방법

#### 2.2.1 Transformer 인코더

ViT는 입력 이미지를 크기 $p^2$의 비중첩 패치로 분할하고, 각 패치를 선형 프로젝션을 통해 $D$차원 토큰으로 임베딩한다. 이미지 크기가 $H \times W$일 때, 토큰의 수는:

$$N_p = \frac{H \cdot W}{p^2}$$

초기 토큰 집합은 위치 임베딩과 결합되며, readout 토큰 $t_0$이 추가되어:

$$t^0 = \{t^0_0, t^0_1, \ldots, t^0_{N_p}\}, \quad t^0_n \in \mathbb{R}^D$$

이 토큰들은 $L$개의 transformer 레이어를 거쳐 $t^l$ ($l$번째 레이어 출력)로 변환된다.

**핵심 특성:**
- 모든 처리 단계에서 **일정한 공간 해상도** 유지 (다운샘플링 없음)
- 매 단계마다 **전역 수용 영역** (MHSA가 모든 토큰 간 관계를 모델링)

#### 2.2.2 Reassemble 연산

토큰을 이미지-유사 특징 맵으로 변환하는 3단계 연산:

$$\text{Reassemble}^{\hat{D}}_s(t) = (\text{Resample}_s \circ \text{Concatenate} \circ \text{Read})(t)$$

여기서 $s$는 입력 이미지 대비 출력 크기 비율, $\hat{D}$는 출력 특징 차원이다.

**Step 1: Read** — readout 토큰 처리

$$\text{Read}: \mathbb{R}^{(N_p+1) \times D} \rightarrow \mathbb{R}^{N_p \times D} $$

세 가지 변형이 제안된다:

- **Ignore:** readout 토큰을 무시

$$\text{Read}\_{\text{ignore}}(t) = \{t_1, \ldots, t_{N_p}\} $$

- **Add:** readout 토큰을 모든 토큰에 더함

$$\text{Read}\_{\text{add}}(t) = \{t_1 + t_0, \ldots, t_{N_p} + t_0\} $$

- **Project:** readout 토큰을 각 토큰에 연결(concatenate)한 후 MLP로 프로젝션

$$\text{Read}\_{\text{proj}}(t) = \{\text{mlp}(\text{cat}(t_1, t_0)), \ldots, \text{mlp}(\text{cat}(t_{N_p}, t_0))\} $$

**Step 2: Concatenate** — 토큰을 공간적으로 배치

$$\text{Concatenate}: \mathbb{R}^{N_p \times D} \rightarrow \mathbb{R}^{\frac{H}{p} \times \frac{W}{p} \times D} $$

**Step 3: Resample** — 목표 해상도로 리샘플링

$$\text{Resample}_s: \mathbb{R}^{\frac{H}{p} \times \frac{W}{p} \times D} \rightarrow \mathbb{R}^{\frac{H}{s} \times \frac{W}{s} \times \hat{D}} $$

$1 \times 1$ convolution으로 채널을 $\hat{D}$로 프로젝션한 후, $s \geq p$이면 strided $3 \times 3$ convolution, $s < p$이면 strided $3 \times 3$ transpose convolution을 적용.

#### 2.2.3 Fusion 디코더

네 개의 서로 다른 transformer 레이어에서 네 가지 해상도($\frac{1}{4}, \frac{1}{8}, \frac{1}{16}, \frac{1}{32}$)로 특징 맵을 추출하여, RefineNet 기반 fusion 블록으로 점진적으로 융합 및 업샘플링한다. 최종 표현은 입력 해상도의 절반이며, 태스크별 출력 헤드가 최종 예측을 생성한다.

- **DPT-Large:** 레이어 $l = \{5, 12, 18, 24\}$에서 특징 추출
- **DPT-Base:** 레이어 $l = \{3, 6, 9, 12\}$에서 특징 추출
- **DPT-Hybrid:** ResNet50의 1, 2번째 블록 + transformer 레이어 $l = \{9, 12\}$

### 2.3 모델 구조 요약

| 구성 요소 | 설명 |
|---|---|
| **인코더** | ViT-Base ($D=768$, 12 layers), ViT-Large ($D=1024$, 24 layers), 또는 ViT-Hybrid (ResNet50 + 12 transformer layers) |
| **패치 크기** | $p = 16$ (모든 실험) |
| **Reassemble** | Read → Concatenate → Resample, 4개 스테이지, $\hat{D} = 256$ |
| **Fusion** | RefineNet 기반, 잔차 합성곱 유닛, 점진적 2배 업샘플링 |
| **출력 헤드** | 태스크별 (깊이 추정: 3개 conv 레이어, 세그멘테이션: conv + dropout + 1×1 conv) |

### 2.4 성능 향상

#### 단안 깊이 추정 (Zero-shot Cross-dataset Transfer)

| 모델 | 평균 상대 성능 향상 (MiDaS 대비) |
|---|---|
| DPT-Large | **28% 이상** |
| DPT-Hybrid | **23% 이상** |

특히 KITTI에서 $\delta > 1.25$ 기준 **64.6% 상대 개선** (DPT-Large).

#### 시맨틱 세그멘테이션 (ADE20K)

| 모델 | pixAcc | mIoU |
|---|---|---|
| DeeplabV3+ResNeSt-200 | 82.45% | 48.36% |
| **DPT-Hybrid** | **83.11%** | **49.02%** |

#### 파인튜닝 결과

- **NYUv2 깊이:** $\delta > 1.25 = 0.904$ (SOTA)
- **KITTI 깊이:** RMSE = 2.573 (SOTA)
- **Pascal Context:** mIoU = 60.46% (SOTA)

### 2.5 한계

1. **대규모 데이터 의존성:** Transformer 특성상 충분한 학습 데이터가 없으면 성능이 제한됨. ADE20K(비교적 소규모)에서 DPT-Large가 DPT-Hybrid보다 성능이 낮은 현상이 이를 뒷받침.
2. **위치 임베딩의 해상도 의존성:** 학습 시와 다른 이미지 크기를 사용할 경우 위치 임베딩을 보간해야 하며, 이로 인한 성능 저하 가능성이 존재 (다만 FCN 대비 더 완만하게 성능이 저하됨).
3. **고정 패치 크기:** $p = 16$으로 고정되어, 초기 임베딩 단계에서의 공간 해상도가 $\frac{1}{16}$로 제한됨.
4. **계산 비용:** Self-attention의 계산 복잡도가 토큰 수의 제곱( $O(N_p^2)$ )에 비례하여 고해상도 입력에서 비용이 급증.

---

## 3. 모델의 일반화 성능 향상 가능성

DPT의 일반화 성능은 이 논문의 가장 핵심적인 강점 중 하나이다.

### 3.1 Zero-shot Cross-dataset Transfer

DPT의 가장 인상적인 결과는 **학습에 사용되지 않은 6개 데이터셋에 대한 zero-shot 전이 성능**이다. DPT-Large는 MiDaS 대비 평균 28% 이상의 상대적 성능 향상을 보이며, 이는 모델이 학습 데이터에 과적합되지 않고 다양한 도메인에 일반화되는 강력한 표현을 학습했음을 시사한다.

### 3.2 전역 수용 영역의 기여

Transformer의 MHSA는 매 레이어에서 전역 수용 영역을 제공한다. 이는:

- **전역적 맥락 파악:** 하늘, 넓은 균일 영역 등에서 일관된 깊이 예측
- **장거리 의존성 모델링:** 이미지 전체에 걸친 상대적 깊이 관계 파악

이러한 특성은 특히 도메인이 변경되었을 때에도 견고한 예측을 가능하게 한다.

### 3.3 일정한 해상도 유지의 이점

ViT 인코더는 초기 임베딩 이후 다운샘플링 없이 일정한 해상도($\frac{H}{p} \times \frac{W}{p}$)를 유지한다. 이로 인해:

- 깊은 레이어에서도 세밀한 공간 정보가 보존됨
- 디코더에서 정보 복원 부담이 감소함

### 3.4 추론 해상도 변화에 대한 강건성

Figure 4의 실험 결과에서, 학습 해상도($384 \times 384$)보다 높은 해상도에서 추론할 때:

- **FCN(ResNet-50, ResNeXt-101):** 성능 저하가 급격 (최대 25% 감소)
- **DPT(ViT-Hybrid, DeIT-Dist):** 성능 저하가 완만 (최대 ~5% 수준)

이는 transformer의 전역 수용 영역이 해상도 변화에 대한 강건성을 부여함을 보여준다.

### 3.5 프리트레이닝 전략의 영향

DeIT-Base-Dist(데이터 효율적 프리트레이닝 + distillation)가 ViT-Base 대비 성능 향상을 보인 실험 결과는, **더 나은 프리트레이닝 전략이 dense prediction의 일반화 성능을 직접적으로 향상시킬 수 있음**을 시사한다.

### 3.6 대규모 데이터와 일반화의 관계

MIX 5 → MIX 6 (약 1.4M 이미지)으로 학습 데이터를 확장했을 때:
- FCN(MiDaS)도 혜택을 받지만, DPT가 **더 큰 폭으로 개선**됨
- 이는 transformer가 대규모 데이터에서 더 효과적으로 일반화 가능한 표현을 학습함을 의미

---

## 4. 향후 연구에 미치는 영향과 고려 사항

### 4.1 연구에 미치는 영향

1. **Dense prediction 패러다임 전환:** 이 논문은 dense prediction의 백본을 CNN에서 Transformer로 전환하는 중요한 전환점을 제시했으며, 이후 Swin Transformer, SegFormer, BEiT 등 후속 연구에 직접적인 영향을 미침.

2. **인코더-디코더 설계 원칙 재정립:** Transformer 출력을 다해상도 특징 맵으로 재구성하는 Reassemble 패턴은 이후 많은 연구에서 채택됨.

3. **프리트레이닝의 중요성 재확인:** 대규모 프리트레이닝이 downstream dense prediction 성능에 미치는 영향을 실증적으로 보여줌.

### 4.2 앞으로의 연구 시 고려할 점

1. **효율적 Self-Attention:** $O(N_p^2)$ 복잡도 문제를 해결하기 위한 효율적 attention 메커니즘 (linear attention, windowed attention 등) 적용 필요.

2. **다양한 패치 크기 및 다중 스케일 임베딩:** 고정 패치 크기의 한계를 극복하기 위한 계층적 또는 다중 스케일 토큰 임베딩 전략.

3. **소규모 데이터셋에서의 효율적 학습:** Self-supervised pretraining, knowledge distillation 등을 통한 데이터 효율성 개선.

4. **실시간 추론:** 모바일 및 엣지 디바이스에서의 실시간 dense prediction을 위한 경량화 연구.

5. **위치 임베딩 설계:** 다양한 해상도에 더 잘 적응하는 위치 임베딩 전략 (상대적 위치 임베딩, 조건부 위치 인코딩 등).

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 아이디어 | DPT 대비 차별점 |
|---|---|---|---|
| **Swin Transformer** (Liu et al.) | 2021 | Shifted window 기반 계층적 transformer | 계층적 다운샘플링으로 다중 스케일 특징 자연스럽게 생성, $O(N)$ 선형 복잡도 |
| **SegFormer** (Xie et al.) | 2021 | 계층적 transformer + 경량 MLP 디코더 | 위치 임베딩 없이 Mix-FFN 사용, 더 단순하고 효율적인 디코더 |
| **BEiT** (Bao et al.) | 2021 | Self-supervised pre-training (masked image modeling) | 라벨 없는 대규모 데이터로 프리트레이닝, dense prediction 일반화 향상 |
| **Mask2Former** (Cheng et al.) | 2022 | Masked attention 기반 범용 세그멘테이션 | Panoptic, instance, semantic segmentation 통합 프레임워크 |
| **DINOv2** (Oquab et al., Meta) | 2023 | Self-supervised ViT, 대규모 데이터 | 라벨 없이 학습한 범용 시각 표현, dense prediction에서 뛰어난 zero-shot 일반화 |
| **Depth Anything** (Yang et al.) | 2024 | 대규모 unlabeled data 활용 단안 깊이 추정 | DPT 구조 기반, 62M unlabeled 이미지로 학습하여 일반화 극대화 |
| **ViT-Adapter** (Chen et al.) | 2023 | ViT에 spatial prior 주입 | plain ViT의 dense prediction 적용 시 부족한 지역적/다중 스케일 정보를 어댑터로 보완 |

### 주요 비교 분석

**DPT vs. Swin Transformer:**
- DPT는 ViT의 **일정한 해상도 + 전역 attention**을 활용하는 반면, Swin은 **계층적 구조 + 지역적 windowed attention**을 채택.
- Swin은 계산 효율성에서 우위이나, DPT의 전역 수용 영역이 일부 태스크에서 더 나은 전역 일관성을 제공.

**DPT vs. SegFormer:**
- SegFormer는 Mix-FFN ($3 \times 3$ depth-wise conv를 FFN에 포함)을 통해 위치 임베딩 없이 위치 정보를 인코딩하며, MLP-only 디코더로 더 경량화됨.
- DPT의 RefineNet 기반 convolutional decoder 대비 SegFormer의 MLP 디코더가 더 단순하면서도 경쟁적 성능.

**DPT vs. Depth Anything:**
- Depth Anything은 DPT의 아키텍처를 기반으로 하되, 대규모 unlabeled 데이터를 활용한 semi-supervised learning으로 일반화 성능을 한 단계 더 끌어올림.
- 이는 DPT가 제시한 "대규모 데이터 + Transformer = 우수한 일반화"라는 핵심 통찰이 후속 연구에서 더욱 확장되었음을 보여줌.

**DPT vs. DINOv2:**
- DINOv2는 self-supervised pretraining으로 라벨 없이도 강력한 시각 표현을 학습하며, dense prediction에서 뛰어난 zero-shot 전이 성능을 보임.
- DPT가 ImageNet supervised pretraining에 의존하는 것에 비해, self-supervised 접근은 더 풍부하고 일반적인 표현 학습 가능성을 제시.

---

## 참고자료

1. Ranftl, R., Bochkovskiy, A., & Koltun, V. (2021). "Vision Transformers for Dense Prediction." *arXiv preprint arXiv:2103.13413*. (본 논문)
2. Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *arXiv preprint arXiv:2010.11929*.
3. Liu, Z., et al. (2021). "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." *ICCV 2021*.
4. Xie, E., et al. (2021). "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." *NeurIPS 2021*.
5. Bao, H., et al. (2021). "BEiT: BERT Pre-Training of Image Transformers." *ICLR 2022*.
6. Cheng, B., et al. (2022). "Masked-attention Mask Transformer for Universal Image Segmentation." *CVPR 2022*.
7. Oquab, M., et al. (2023). "DINOv2: Learning Robust Visual Features without Supervision." *arXiv preprint arXiv:2304.07193*.
8. Yang, L., et al. (2024). "Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data." *CVPR 2024*.
9. Chen, Z., et al. (2023). "Vision Transformer Adapter for Dense Predictions." *ICLR 2023*.
10. Ranftl, R., et al. (2020). "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer." *TPAMI*.
11. Touvron, H., et al. (2020). "Training Data-Efficient Image Transformers & Distillation through Attention." *arXiv preprint arXiv:2012.12877*.
12. Lin, G., et al. (2017). "RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation." *CVPR 2017*.
