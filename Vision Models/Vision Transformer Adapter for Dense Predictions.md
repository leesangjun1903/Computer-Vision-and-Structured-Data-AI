# Vision Transformer Adapter for Dense Predictions

---

## 1. 핵심 주장과 주요 기여 요약

**핵심 주장:** Plain ViT(Vision Transformer)는 vision-specific inductive bias(이미지 관련 귀납적 편향)가 부족하여 dense prediction 태스크(객체 탐지, 인스턴스 분할, 시맨틱 분할)에서 Swin Transformer 등 계층적 비전 트랜스포머에 비해 성능이 떨어진다. 이를 해결하기 위해, ViT의 원래 구조를 변경하지 않으면서 **사전학습이 필요 없는 어댑터(pre-training-free adapter)**를 부착하여 성능 격차를 해소할 수 있다.

**주요 기여:**
1. Plain ViT에 이미지 관련 inductive bias를 주입하는 **새로운 패러다임** 제안 — ViT 구조를 수정하지 않아 멀티모달 사전학습의 유연성을 보존
2. **Spatial Prior Module(SPM)**, **Spatial Feature Injector**, **Multi-Scale Feature Extractor** 세 가지 모듈 설계
3. COCO(객체 탐지/인스턴스 분할)와 ADE20K(시맨틱 분할)에서 **state-of-the-art 성능** 달성 — COCO test-dev에서 60.9 box AP, 53.0 mask AP(추가 탐지 데이터 미사용)

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

Plain ViT는 입력 데이터에 대한 사전 가정이 없는 범용 아키텍처이므로 멀티모달 사전학습에 유리하지만, dense prediction 태스크에서는 다음과 같은 결정적 단점이 있다:

- **지역적 공간 정보(local spatial context)의 부재**: 16×16 패치 임베딩만 사용하여 세밀한 공간 정보를 포착하지 못함
- **단일 스케일 특징(single-scale feature)**: 해상도 1/16의 단일 스케일 특징맵만 생성하여, FPN 등 다중 스케일 특징이 요구되는 탐지/분할에 부적합
- **느린 수렴 속도와 낮은 성능**: Swin, PVTv2 등 vision-specific 트랜스포머 대비 동일 학습 조건에서 성능 열세

### 2.2 제안하는 방법 (수식 포함)

#### (a) 전체 구조

ViT-Adapter는 plain ViT 옆에 병렬로 부착되는 **추가 네트워크**로, 세 가지 핵심 모듈로 구성된다:

1. **Spatial Prior Module (SPM)**: 입력 이미지에서 지역적 공간 특징을 추출
2. **Spatial Feature Injector**: 공간 특징을 ViT에 주입
3. **Multi-Scale Feature Extractor**: ViT의 단일 스케일 특징에서 다중 스케일 특징을 재구성

ViT의 $L$개 인코더 레이어를 $N=4$개 블록으로 균등 분할하고, 각 블록에서 Injector와 Extractor를 통한 특징 상호작용을 수행한다.

#### (b) Spatial Prior Module (SPM)

ResNet에서 차용한 convolutional stem(3개 합성곱 + max-pooling)과 stride-2 3×3 합성곱을 사용하여, 입력 이미지로부터 $D$-차원 특징 피라미드 $\{\mathcal{F}_1, \mathcal{F}_2, \mathcal{F}_3\}$ (해상도 1/8, 1/16, 1/32)를 생성한다. 이를 flatten하고 concatenate하여:

$$\mathcal{F}^{1}_{\text{sp}} \in \mathbb{R}^{\left(\frac{HW}{8^2} + \frac{HW}{16^2} + \frac{HW}{32^2}\right) \times D}$$

#### (c) Spatial Feature Injector

$i$번째 ViT 블록에 대해, ViT 입력 특징 $\mathcal{F}^{i}\_{\text{vit}} \in \mathbb{R}^{\frac{HW}{16^2} \times D}$를 query, 공간 특징 $\mathcal{F}^{i}_{\text{sp}}$를 key/value로 사용하여 cross-attention을 수행한다:

$$\hat{\mathcal{F}}^{i}_{\text{vit}} = \mathcal{F}^{i}_{\text{vit}} + \gamma^{i} \cdot \text{Attention}\left(\text{norm}(\mathcal{F}^{i}_{\text{vit}}),\; \text{norm}(\mathcal{F}^{i}_{\text{sp}})\right) $$

여기서:
- $\text{norm}(\cdot)$: LayerNorm
- $\text{Attention}(\cdot)$: sparse attention (기본값: deformable attention)
- $\gamma^{i} \in \mathbb{R}^{D}$: **0으로 초기화된 학습 가능 벡터** — 사전학습된 ViT 가중치의 특징 분포가 급격히 변하지 않도록 보장

#### (d) Multi-Scale Feature Extractor

$i$번째 블록의 출력 $\mathcal{F}^{i+1}_{\text{vit}}$로부터 다중 스케일 특징을 추출한다:

$$\hat{\mathcal{F}}^{i}_{\text{sp}} = \mathcal{F}^{i}_{\text{sp}} + \text{Attention}\left(\text{norm}(\mathcal{F}^{i}_{\text{sp}}),\; \text{norm}(\mathcal{F}^{i+1}_{\text{vit}})\right) $$

$$\mathcal{F}^{i+1}_{\text{sp}} = \hat{\mathcal{F}}^{i}_{\text{sp}} + \text{FFN}\left(\text{norm}(\hat{\mathcal{F}}^{i}_{\text{sp}})\right) $$

여기서 $\mathcal{F}^{i}\_{\text{sp}}$가 query, $\mathcal{F}^{i+1}_{\text{vit}}$가 key/value로 사용된다. 최종적으로 1/8, 1/16, 1/32 해상도의 특징맵을 분할하고, 1/8 특징맵을 2×2 transposed convolution으로 업샘플링하여 **1/4 스케일 특징맵**을 생성, ResNet 유사 feature pyramid를 구축한다.

### 2.3 모델 구조 상세

| 변형 | ViT Layers | Width | FFN | Heads | ViT Param | Adapter Param | 총 Param |
|---|---|---|---|---|---|---|---|
| Tiny (T) | 12 | 192 | 768 | 3 | 5.5M | 2.5M | 8.0M |
| Small (S) | 12 | 384 | 1536 | 6 | 21.7M | 5.8M | 27.5M |
| Base (B) | 12 | 768 | 3072 | 12 | 85.8M | 14.0M | 99.8M |
| Large (L) | 24 | 1024 | 4096 | 16 | 303.3M | 23.7M | 327.0M |

- **상호작용 횟수**: $N=4$ (기본값)
- **Sparse Attention**: Deformable Attention (sampling points = 4)
- **FFN ratio**: 0.25 (계산 비용 절감)
- 마지막 상호작용에서 Multi-Scale Feature Extractor를 3개 스택

### 2.4 성능 향상

#### 객체 탐지 (COCO val2017, Mask R-CNN 3×+MS)

| 방법 | #Param | $\text{AP}^{b}$ | $\text{AP}^{m}$ |
|---|---|---|---|
| ViT-S (baseline) | 43.8M | 44.0 | 39.9 |
| ViTDet-S | 45.7M | 44.5 | 40.1 |
| **ViT-Adapter-S** | 47.8M | **48.2** | **42.8** |
| Swin-B | 107.1M | 48.6 | 43.3 |
| **ViT-Adapter-B** | 120.2M | **49.6** | **43.6** |
| ViT-Adapter-L† (IN-22K) | 347.9M | **52.1** | **46.0** |

ViT-Adapter-S는 ViT-S 대비 **+4.2 AP $^{b}$ **, **+2.9 AP $^{m}$ ** 향상.

#### 시맨틱 분할 (ADE20K, UperNet 160k)

| 방법 | Pre-train | mIoU | +MS |
|---|---|---|---|
| Swin-B | IN-1K | 48.1 | 49.7 |
| **ViT-Adapter-B** | IN-1K | **48.8** | **49.7** |
| Swin-L† | IN-22K | 52.1 | 53.5 |
| **ViT-Adapter-L†** | IN-22K | **53.4** | **54.4** |
| **ViT-Adapter-L★** | Multi-Modal | **55.0** | **55.4** |

#### SOTA 결과 (COCO test-dev)

| 방법 | $\text{AP}^{b}$ | $\text{AP}^{m}$ |
|---|---|---|
| CB-Swin-L | 60.1 | 52.3 |
| SwinV2-L | 60.8 | 52.7 |
| **ViT-Adapter-L** | **60.9** | **53.0** |

### 2.5 Ablation Study 주요 결과

**컴포넌트별 기여 (ViT-S, Mask R-CNN 1×):**
- SPM만 추가 (Add): +1.4 AP $^{b}$
- + Spatial Feature Injector (Attention): +1.0 AP $^{b}$ 추가
- + Multi-Scale Feature Extractor: +2.1 AP $^{b}$ 추가
- **총합: +4.5 AP $^{b}$ , +2.8 AP $^{m}$ **

**Attention 유형 비교:**
Deformable Attention이 가장 효율적 — 403G FLOPs, 0.36s/iter, 13.7G 메모리로 최고 성능(44.7 AP $^{b}$ ) 달성.

### 2.6 한계

1. **추가 파라미터 오버헤드**: 어댑터가 ViT 대비 약 6~8%의 추가 파라미터를 요구
2. **Deformable Attention 의존성**: 기본 sparse attention으로 deformable attention을 사용하며, 이는 구현 복잡성을 증가시킴
3. **공정한 비교의 어려움**: ViTDet와의 비교에서 학습 에폭 수, 사전학습 방법, 데이터 증강 전략 등이 상이하여 완전히 통제된 비교가 제한됨
4. **분류 태스크와의 디커플링**: 피라미드 prior는 분류에는 이점이 적어, 분류와 dense prediction 간 모델 설계를 분리해야 함
5. **VPT 등 parameter-efficient 방법과의 통합**: ViT-Adapter는 성능 최적화를 목표로 하므로, parameter-efficient transfer learning과의 결합은 향후 과제로 남아 있음

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 핵심 일반화 메커니즘

ViT-Adapter의 일반화 성능 향상은 세 가지 축에서 이루어진다:

**(1) 사전학습-미세조정 패러다임의 분리(Decoupling)**

ViT의 원래 구조를 수정하지 않으므로, **어떠한 방식으로 사전학습된 ViT 가중치라도** 그대로 활용할 수 있다. 이는 다음을 의미한다:

- ImageNet-1K/22K 지도학습
- MAE/BEiT 등 자기지도학습(MIM)
- Uni-Perceiver, BEiT-3 등 **멀티모달 사전학습**

논문에서 실험적으로 확인된 바(Table 4):
- ImageNet-1K → ImageNet-22K: +0.9 AP $^{b}$
- ImageNet-22K → Multi-Modal: +0.7 AP $^{b}$
- Swin-B는 Multi-Modal 사전학습 적용 불가(N/A)

이는 ViT-Adapter가 **사전학습 방법의 발전에 따라 자동으로 성능이 향상되는 확장 가능한(scalable) 프레임워크**임을 보여준다.

**(2) 입력 prior와 태스크 prior의 동시 활용**

기존 방법(ViTDet, SETR 등)은 태스크 prior(다중 스케일 특징맵 재구성)만 사용하지만, ViT-Adapter는 **입력 이미지 자체의 정보**(SPM을 통한 공간 특징)도 함께 활용한다. 이는 다양한 입력 도메인에 대한 적응력을 높인다.

**(3) 반복적 특징 상호작용(Iterative Feature Interaction)**

Injector와 Extractor가 ViT 인코더의 중간 레이어마다 반복적으로 상호작용하여, spatial prior가 ViT의 전역적 특징과 점진적으로 통합된다. 이는 단순한 후처리 방식보다 더 정교한 특징 표현을 가능하게 한다.

### 3.2 주파수 관점에서의 일반화

Fourier 분석(Figure 5)에서 ViT-Adapter가 plain ViT 대비 **고주파 정보(local edges, textures)**를 더 많이 포착함이 확인되었다. ViT가 저주파 전역 신호에 편향되어 있다면, ViT-Adapter는 CNN의 고주파 포착 능력을 이식(graft)하여 **주파수 스펙트럼의 균형을 맞추는 역할**을 한다.

### 3.3 태스크 일반성

ViT-Adapter는 단일 아키텍처로 **4가지 탐지 프레임워크**(Mask R-CNN, Cascade Mask R-CNN, ATSS, GFL)와 **2가지 분할 프레임워크**(Semantic FPN, UperNet)에서 일관된 성능 향상을 보여주어, 태스크에 무관한 범용 어댑터로서의 잠재력을 입증하였다.

### 3.4 frozen ViT에서의 성능

Table 11에서 ViT를 frozen한 상태로 ViT-Adapter-L만 학습시켜도 ADE20K에서 49.0/50.6 mIoU를 달성하여, full-tuning ViT-L(48.3/50.1)을 상회한다. 이는 어댑터 자체가 강력한 일반화 능력을 가짐을 시사한다.

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

**(1) 사전학습-미세조정 디커플링 패러다임의 확산**

이 논문은 "백본 아키텍처를 태스크에 맞게 설계"하는 기존 접근 대신, "범용 백본 + 태스크별 어댑터"라는 새로운 패러다임을 제시하였다. 이는 이후 BEiT-3 (Wang et al., 2022b)에서 실제로 ViT-Adapter를 채택하여 ADE20K에서 62.8 mIoU를 달성하며 검증되었다.

**(2) Foundation Model 시대의 dense prediction 전략 수립**

GPT-4V, CLIP, Segment Anything Model(SAM) 등 대규모 foundation model이 등장하면서, plain ViT 기반 모델의 dense prediction 적용 수요가 증가하고 있다. ViT-Adapter는 이러한 모델들을 dense prediction에 효율적으로 활용하는 방법론적 토대를 제공한다.

**(3) 어댑터 기반 전이 학습 연구의 활성화**

NLP의 어댑터(Houlsby et al., 2019) 개념을 비전의 dense prediction으로 확장한 대표적 사례로, 이후 parameter-efficient transfer learning과의 결합 연구를 촉진하였다.

### 4.2 향후 연구 시 고려할 점

1. **Parameter-efficient 버전 개발**: ViT-Adapter는 성능 최적화에 초점을 맞추어 full-tuning을 수행하지만, 대규모 모델(ViT-G, ViT-22B 등)에는 parameter-efficient 방식과의 결합이 필수적
2. **더 효율적인 attention 메커니즘 탐색**: 현재 deformable attention을 사용하지만, FlashAttention, linear attention 등 최신 방법론으로 대체 시 추가 성능/효율 향상 가능
3. **3D, 비디오, 포인트 클라우드 등 다른 모달리티로의 확장**: plain ViT의 유연성을 활용하여 3D dense prediction으로 확장 가능
4. **학습 효율 문제**: ViTDet가 100 에폭 학습으로 높은 성능을 보인 반면, ViT-Adapter는 36 에폭에서 비교적 빠르게 수렴하나, 수렴 속도와 학습 비용의 최적 균형점 탐색이 필요
5. **Adapter의 구조적 최적화**: SPM의 convolutional stem, 상호작용 횟수 $N$, FFN ratio 등의 하이퍼파라미터에 대한 NAS(Neural Architecture Search) 적용 가능성

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 접근 | ViT-Adapter와의 차이 |
|---|---|---|---|
| **ViTDet** (Li et al., 2022) | 2022 | Plain ViT + simple feature pyramid (up/downsampling) | 태스크 prior만 사용, 입력 prior 미활용; MAE + 100 에폭 필요 |
| **Swin Transformer** (Liu et al., 2021) | 2021 | 계층적 shifted window attention | Vision-specific 설계; 멀티모달 사전학습 불가 |
| **BEiT/BEiTv2** (Bao et al., 2022; Peng et al., 2022) | 2022 | Masked Image Modeling 자기지도학습 | 사전학습 방법론; ViT-Adapter와 결합 시 시너지(58.0→58.5 mIoU) |
| **BEiT-3** (Wang et al., 2022b) | 2022 | 멀티모달 사전학습 foundation model | **ViT-Adapter를 직접 채택**하여 ADE20K 62.8 mIoU 달성 |
| **MAE** (He et al., 2021) | 2021 | Masked Autoencoder 자기지도학습 | 사전학습 기법; ViTDet의 기본 사전학습으로 사용 |
| **Mask2Former** (Cheng et al., 2021) | 2021 | Universal segmentation decoder | Decoder 설계; ViT-Adapter와 조합 가능(Table 9) |
| **ConvNeXt** (Liu et al., 2022) | 2022 | CNN의 현대화(modernized ResNet) | Vision-specific 순수 CNN; 멀티모달 유연성 부재 |
| **Conformer** (Peng et al., 2021) | 2021 | CNN+Transformer 이중 네트워크 | 사전학습 시부터 dual architecture; ViT-Adapter는 사전학습 불필요 |
| **VPT** (Jia et al., 2022) | 2022 | Visual Prompt Tuning | 분류 중심; dense prediction 시 성능 저하(Table 11: 44.0 vs 48.3 mIoU) |
| **AdaptFormer** (Chen et al., 2022) | 2022 | ViT 내부에 경량 adapter 삽입 | 분류/비디오 인식 중심; dense prediction 미검증 |
| **SAM (Segment Anything)** (Kirillov et al., 2023) | 2023 | Prompt 기반 범용 세그멘테이션 | Foundation model 접근; ViT-Adapter와 유사한 plain ViT 활용 철학 |
| **InternImage** (Wang et al., 2023) | 2023 | Large-scale deformable convolution backbone | Vision-specific 설계; ViT-Adapter의 경쟁 모델 |

### 핵심 비교 인사이트

1. **ViTDet vs ViT-Adapter**: 동일 조건(DeiT 사전학습, 3× schedule)에서 ViT-Adapter-S가 ViTDet-S를 3.7 AP $^{b}$ 상회. ViTDet는 MAE + 100 에폭의 강력한 학습 설정이 필요하여 inductive bias 부재로 인한 **느린 수렴 문제**를 간접적으로 노출

2. **BEiT-3 + ViT-Adapter**: BEiT-3 논문에서 ViT-Adapter를 공식 채택한 것은, 이 어댑터의 **범용성과 효과성**에 대한 독립적 검증

3. **VPT/AdaptFormer 계열과의 관계**: 이들은 parameter-efficient learning이 목적이나, dense prediction에서는 성능이 부족. ViT-Adapter는 성능 극대화가 목적이며, 두 접근은 **직교적(orthogonal)**

---

## 참고자료

1. **Chen, Z., Duan, Y., Wang, W., He, J., Lu, T., Dai, J., & Qiao, Y. (2023).** "Vision Transformer Adapter for Dense Predictions." *ICLR 2023.* arXiv:2205.08534v4.
2. **Li, Y., Xie, S., Chen, X., Dollár, P., He, K., & Girshick, R. (2021).** "Benchmarking Detection Transfer Learning with Vision Transformers." arXiv:2111.11429.
3. **Li, Y., Mao, H., Girshick, R., & He, K. (2022).** "Exploring Plain Vision Transformer Backbones for Object Detection." arXiv:2203.16527. (ViTDet)
4. **Liu, Z. et al. (2021).** "Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows." *ICCV 2021.*
5. **He, K. et al. (2021).** "Masked Autoencoders Are Scalable Vision Learners." arXiv:2111.06377. (MAE)
6. **Bao, H., Dong, L., & Wei, F. (2022).** "BEiT: BERT Pre-Training of Image Transformers." *ICLR 2022.*
7. **Peng, Z. et al. (2022).** "BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers." arXiv:2208.06366.
8. **Wang, W. et al. (2022).** "Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks." arXiv:2208.10442. (BEiT-3)
9. **Cheng, B. et al. (2021).** "Masked-Attention Mask Transformer for Universal Image Segmentation." arXiv:2112.01527. (Mask2Former)
10. **Wang, W. et al. (2022).** "PVTv2: Improved Baselines with Pyramid Vision Transformer." *CVMJ.*
11. **Jia, M. et al. (2022).** "Visual Prompt Tuning." arXiv:2203.12119. (VPT)
12. **Houlsby, N. et al. (2019).** "Parameter-Efficient Transfer Learning for NLP." *ICML 2019.*
13. **Zhu, X. et al. (2021).** "Uni-Perceiver: Pre-training Unified Architecture for Generic Perception for Zero-shot and Few-shot Tasks." arXiv:2112.01522.
14. **Dosovitskiy, A. et al. (2020).** "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR 2020.* (ViT)
15. **Liu, Z. et al. (2022).** "A ConvNet for the 2020s." arXiv:2201.03545. (ConvNeXt)
16. GitHub 리포지토리: https://github.com/czczup/ViT-Adapter
