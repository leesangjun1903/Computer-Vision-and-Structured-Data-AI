
# Your ViT is Secretly an Image Segmentation Model

---

## 📌 참고 자료 및 출처

| # | 출처 |
|---|------|
| 1 | Kerssies et al., *"Your ViT is Secretly an Image Segmentation Model"*, CVPR 2025 Highlight, arXiv:2503.19108 |
| 2 | 공식 프로젝트 페이지: https://www.tue-mps.org/eomt/ |
| 3 | GitHub 공식 코드: https://github.com/tue-mps/eomt |
| 4 | CVPR 2025 공식 페이지: https://cvpr.thecvf.com/virtual/2025/poster/33107 |
| 5 | Hugging Face 모델 문서: https://huggingface.co/docs/transformers/model_doc/eomt |
| 6 | Lightly AI 블로그: https://www.lightly.ai/blog/eomt-image-segmentation |
| 7 | Liner.com Quick Review: https://liner.com/review/your-vit-is-secretly-image-segmentation-model |
| 8 | ResearchGate PDF: https://www.researchgate.net/publication/390176912 |
| 9 | DeepWiki: https://deepwiki.com/tue-mps/eomt |
| 10 | CVPR 2025 Open Access PDF: https://openaccess.thecvf.com/content/CVPR2025/papers/Kerssies_Your_ViT_is_Secretly_an_Image_Segmentation_Model_CVPR_2025_paper.pdf |

---

## 1. 🔑 핵심 주장 및 주요 기여 요약

ViT(Vision Transformer)는 다양한 컴퓨터 비전 태스크에서 뛰어난 성능과 확장성을 보였으나, 단일-스케일 ViT를 이미지 세그멘테이션에 적용하기 위해 기존 방법들은 **다중 스케일 특징을 생성하는 합성곱 어댑터(convolutional adapter)**, **특징 융합을 위한 픽셀 디코더(pixel decoder)**, 그리고 **예측을 수행하는 Transformer 디코더** 등의 복잡한 태스크-특화 컴포넌트를 사용해왔습니다. 이 논문은 이러한 컴포넌트들이 도입하는 귀납 편향(inductive bias)이 **충분히 큰 모델과 광범위한 사전학습이 주어진다면 ViT 자체가 학습할 수 있음**을 최초로 실증합니다.

**주요 기여 3가지:**

1. 이 발견에 기반하여 **Encoder-only Mask Transformer (EoMT)** 를 제안하며, 순수한(plain) ViT 아키텍처를 이미지 세그멘테이션에 활용합니다.
2. 대규모 모델과 사전학습으로 EoMT는 태스크-특화 컴포넌트를 사용하는 최신 모델과 유사한 세그멘테이션 정확도를 달성하며, 동시에 아키텍처 단순성 덕분에 ViT-L 기준 **최대 $4\times$ 더 빠른** 속도를 실현합니다. 이는 컴퓨팅 자원을 아키텍처 복잡도 추가가 아닌 ViT 자체 확장에 투자하는 것이 더 효율적임을 시사합니다.
3. 이는 이미지 세그멘테이션의 패러다임 전환을 의미하며, 이 연구는 CVPR 2025 Highlight 논문으로 채택되어 컴퓨터 비전 커뮤니티에 큰 기여를 인정받았습니다.

---

## 2. 🔬 상세 분석

### 2-1. 해결하고자 하는 문제

최신 ViT 기반 세그멘테이션 성능을 달성하기 위해, ViT는 일반적으로 **ViT-Adapter** 및 **Mask2Former(M2F)** 와 같은 계산 집약적이고 태스크-특화적인 컴포넌트들과 결합됩니다.

기존 파이프라인의 구조는 다음과 같습니다:

```math
\text{Image} \xrightarrow{\text{ViT + Adapter}} \{F_4, F_8, F_{16}, F_{32}\} \xrightarrow{\text{Pixel Decoder}} \{\hat{F}_4, \hat{F}_8, \hat{F}_{16}, \hat{F}_{32}\} \xrightarrow{\text{Transformer Decoder}} \text{Masks + Labels}
```

M2F는 ViT-Adapter로 추출된 특징을 더욱 향상시키기 위해 픽셀 디코더를 적용합니다. 이 픽셀 디코더는 $\{F_4, F_8, F_{16}, F_{32}\}$ 특징들을 받아 다중-스케일 변형 가능 어텐션(multi-scale deformable attention) 레이어들을 적용하여 $\{\hat{F}\_4, \hat{F}\_8, \hat{F}\_{16}, \hat{F}_{32}\}$ 를 출력합니다. 이 과정에서 서로 다른 백본 레이어의 다중-스케일 특징이 일관되면서도 스케일-특화된 표현으로 처리됩니다.

이러한 구조적 복잡성이 **성능과 속도 사이의 불필요한 트레이드오프**를 유발한다는 것이 이 논문이 해결하고자 하는 핵심 문제입니다.

---

### 2-2. 제안하는 방법 (EoMT)

#### ① 핵심 아이디어

EoMT는 **순수한 ViT를 활용하여 이미지 패치와 세그멘테이션 쿼리를 토큰으로 함께 인코딩하는** 최소주의적(minimalist) 이미지 세그멘테이션 모델입니다.

별도의 Transformer 디코더를 사용하는 대신, EoMT는 **쿼리 디코딩을 인코더 내부에 직접 통합**합니다. ViT 인코더의 마지막 레이어들에서 학습 가능한 쿼리 토큰(query tokens)을 이미지 토큰 시퀀스 앞에 추가(prepend)하여, 두 세트의 토큰을 함께 처리합니다.

#### ② 아키텍처 수식

EoMT의 핵심 처리 과정을 수식으로 표현하면:

**입력 토큰 구성 (최종 $L_2$개 블록에서):**
$$\mathbf{T} = [\mathbf{q}_1, \mathbf{q}_2, \ldots, \mathbf{q}_N, \mathbf{p}_1, \mathbf{p}_2, \ldots, \mathbf{p}_M]$$

여기서 $\mathbf{q}_i$는 학습 가능한 쿼리 토큰, $\mathbf{p}_j$는 이미지 패치 토큰, $N$은 쿼리 수, $M$은 패치 수입니다.

**ViT 블록 내 셀프-어텐션 (쿼리와 패치 통합 처리):**
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}_{\text{mask}}\right)\mathbf{V}$$

이 방식은 크로스-어텐션(이미지 토큰과 쿼리 간)과 셀프-어텐션(쿼리 내부)이 교차하는 기존 패턴을 **공유 셀프-어텐션(shared self-attention)으로 대체**하며, 쿼리 토큰이 인코더의 어텐션 레이어를 통해 서로 간, 그리고 이미지 토큰에 동시에 어텐션합니다.

**마스크 예측 (각 쿼리 $i$에 대해):**
$$\hat{m}_i = \sigma\left(\mathbf{q}_i \cdot \mathbf{P}^\top\right), \quad \hat{c}_i = \text{softmax}(\mathbf{W}_c \mathbf{q}_i)$$

여기서 $\mathbf{P}$는 이미지 패치 특징 행렬, $\hat{m}_i$는 예측 마스크, $\hat{c}_i$는 클래스 예측입니다.

#### ③ Mask Annealing 전략

EoMT의 핵심 혁신 중 하나는 **추론 시 masked attention이 불필요**하다는 점입니다. 기존 Mask2Former 유사 아키텍처에서는 각 쿼리가 해당 쿼리에 대해 예측된 중간 세그멘테이션 마스크 내에서만 크로스-어텐션을 수행하도록 제한하는 masked attention을 사용합니다.

이를 해결하기 위해 **mask annealing 전략**을 도입하여 masked attention 없이도 추론이 가능하도록 하면서 세그멘테이션 품질을 유지합니다.

훈련 중 마스크 강도를 점차 감소시키는 annealing 스케줄:

$$\mathbf{M}_{\text{mask}}^{(t)} = \alpha(t) \cdot \mathbf{M}_{\text{hard}} + (1 - \alpha(t)) \cdot \mathbf{0}$$

여기서 $\alpha(t)$는 훈련 스텝 $t$에 따라 $1 \to 0$으로 감소하는 annealing 계수입니다.

---

### 2-3. 모델 구조

EoMT는 순수 ViT를 재활용하여 이미지 패치와 세그멘테이션 쿼리를 토큰으로 함께 인코딩하는 최소주의 이미지 세그멘테이션 모델입니다. 어댑터, 디코더, 태스크-특화 컴포넌트를 요구하는 복잡한 다단계 파이프라인과 달리, EoMT는 **오직 ViT 인코더만으로** 최신 결과를 달성합니다.

```
┌─────────────────────────────────────────────┐
│              EoMT 아키텍처                    │
│                                             │
│  Image → Patch Embedding → Patch Tokens     │
│                                ↓            │
│          ViT Blocks (L - L₂ blocks)         │
│                                ↓            │
│  Query Tokens ──→ [Q₁,...,Qₙ | P₁,...,Pₘ] │
│                  (마지막 L₂ 블록에서 통합)    │
│                        ↓                   │
│          ViT Blocks (L₂ blocks, 공동 처리)  │
│                        ↓                   │
│          Mask Head + Class Head             │
│          (마스크 + 클래스 예측)              │
└─────────────────────────────────────────────┘
```

EoMT는 **학습된 객체 쿼리(learned object queries)와 경량 마스크 예측 헤드(lightweight mask prediction head)를 ViT 인코더 내부에 직접 통합**함으로써 태스크-특화 디코더의 필요성을 제거합니다.

이러한 아키텍처적 단순성은 **DINOv2를 통한 대규모 자기지도 사전학습(self-supervised pretraining)** 에 의해 가능해집니다.

---

### 2-4. 성능 향상

EoMT의 주요 성능 결과는 다음과 같습니다:
- **파노틱 세그멘테이션**: COCO에서 EoMT-L (1280×1280) 기준 최대 **58.9 PQ**
- **인스턴스 세그멘테이션**: COCO에서 EoMT-L (1280×1280) 기준 최대 **49.9 mAP**
- **의미론적 세그멘테이션**: ADE20K에서 EoMT-L (512×512) 기준 최대 **59.5 mIoU**

COCO 파노틱 세그멘테이션에서 EoMT는 기존 최신 방법과 동등한 성능을 달성하면서 **최대 2.1배 더 빠른** 속도를 보입니다. 이는 단순화된 인코더-전용 아키텍처가 복잡한 모델들과 경쟁할 수 있음을 보여주며, 특히 복잡한 태스크-특화 컴포넌트들이 불필요함을 입증합니다.

Cityscapes와 ADE20K의 의미론적 세그멘테이션에서 EoMT는 더 복잡한 ViT-Adapter + Mask2Former 기준선과 mIoU 면에서 비슷한 성능을 내면서 **최대 4.4배의 속도 향상**을 달성합니다. 이러한 일관된 효율성 향상은 EoMT의 아키텍처적 이점을 잘 보여줍니다.

---

### 2-5. 한계

논문에서 명시적으로 언급된 한계 및 검색 결과에서 확인된 사항:

1. **소형 모델에서의 한계**: 컴퓨팅 자원을 아키텍처 복잡도 추가가 아닌 **ViT 확장과 사전학습에 투자해야** 높은 성능이 나타나며, 소형 모델에서는 태스크-특화 컴포넌트를 가진 기존 방법 대비 성능 격차가 존재할 수 있습니다.

2. **Frozen Foundation Model 호환성 문제**: 이를 해결하기 위한 후속 모델 PMT(Plain Mask Transformer)가 제안되었으며, 이는 **Frozen Vision Encoder의 특징을 보존**하면서도 EoMT의 미니멀리즘 철학을 유지합니다.

3. **인스턴스 세그멘테이션 정확도**: 인스턴스 세그멘테이션 태스크에서는 일부 정확도 저하가 있을 수 있습니다 (검색 결과 29-11에서 언급되었으나 수치 미확인).

---

## 3. 🌍 모델의 일반화 성능 향상 가능성

### 3-1. 사전학습이 일반화의 핵심

최근 연구들은 ViT가 대규모 사전학습에 매우 적합함을 보여왔으며, 이를 통해 **많은 다운스트림 태스크에서 높은 성능을 달성하는 일반화 가능한 모델**이 탄생합니다.

저자들은 백본 및 사전학습(예: DINOv2, DINOv3)을 확장하는 것이 아키텍처 복잡도를 추가하는 것보다 **더 큰 성능 향상**을 가져온다고 주장합니다.

### 3-2. 도메인 간 일반화

연구에서 탐구되는 중요한 질문 중 하나는 **"EoMT가 분포 외(out-of-distribution) 데이터에서 다른 모델들에 비해 얼마나 잘 일반화하는가, 그리고 신뢰도 추정은 얼마나 신뢰할 수 있는가"** 입니다.

심지어 제로샷(zero-shot) 설정에서도 EoMT는 의미 있는 마스크를 생성하여 세그멘테이션 데이터로 올바르게 파인튜닝 시 강력한 잠재력을 보입니다.

### 3-3. 미래 기술과의 호환성을 통한 일반화 향상

EoMT는 순수 ViT에만 의존하기 때문에 FlashAttention, 전용 하드웨어, 토큰 병합(token merging) 등 **Transformer 관련 미래 발전을 직접 활용**할 수 있으며, 추가적인 비최적화 모듈에 의한 병목 없이 이를 즉시 적용할 수 있습니다.

### 3-4. 멀티태스크 일반화

EoMT는 **의미론적, 인스턴스, 파노틱 세그멘테이션 세 가지 주요 세그멘테이션 태스크 전반에서** 뛰어난 성능을 달성하며, 단일 모델로 다양한 태스크를 처리할 수 있는 일반화 능력을 보여줍니다.

### 3-5. 비디오 세그멘테이션으로의 확장

EoMT 철학의 강력한 일반화 가능성은 후속 연구인 **VidEoMT**("Your ViT is Secretly Also a Video Segmentation Model", CVPR 2026)로 이어졌습니다. VidEoMT는 온라인 비디오 세그멘테이션을 위한 경량 인코더-전용 모델로, 전용 추적 모듈이나 무거운 태스크-특화 헤드 없이 ViT 인코더 내에서 공간적 및 시간적 추론을 모두 수행합니다.

---

## 4. 🔮 연구에 미치는 영향 및 앞으로 연구 시 고려할 점

### 4-1. 연구에 미치는 영향

#### ① 패러다임 전환
이 논문의 발견은 **컴퓨팅 자원을 아키텍처 복잡도 추가가 아닌 ViT 확장과 사전학습에 투자해야 함**을 시사하며, EoMT는 Transformer 및 자기지도 학습 분야의 급속한 발전에 쉽게 적응하는 **차세대 세그멘테이션 모델의 견고한 기반**을 제공합니다.

#### ② 산업적 영향
디코더나 어댑터 등 추가 컴포넌트 없이 구성된 EoMT의 단순한 설계는 구현, 훈련, 배포를 더 쉽게 만듭니다. 이는 특히 엣지 컴퓨팅 및 실시간 응용 분야에서 중요한 의미를 가집니다.

#### ③ 멀티모달 및 비디오로의 확장 영감
후속 모델 PMT(Plain Mask Transformer)는 이미지와 비디오 세그멘테이션 모두를 커버하며, EoMT의 미니멀리즘 철학과 Frozen Foundation Model의 특징 보존 필요성을 조화시킵니다.

#### ④ 스케일링 법칙 재검토
논문은 세그멘테이션 분야에서도 "모델 스케일링 > 아키텍처 복잡도 추가"라는 스케일링 법칙이 성립함을 보여주며, NLP의 Scaling Laws와 유사한 패턴이 비전 세그멘테이션에서도 나타남을 입증합니다.

---

### 4-2. 2020년 이후 관련 최신 연구 비교 분석

| 모델 | 연도 | 특징 | EoMT와의 관계 |
|------|------|------|--------------|
| **ViT (Dosovitskiy et al.)** | 2020 | 순수 Transformer를 비전에 최초 적용 | EoMT의 기반 아키텍처 |
| **Mask2Former (Cheng et al.)** | 2022 | 마스크 어텐션 기반 통합 세그멘테이션 | EoMT가 대체하는 주요 비교 대상 |
| **ViT-Adapter (Chen et al.)** | 2022 | ViT에 합성곱 어댑터로 다중-스케일 특징 추출 | EoMT가 불필요함을 증명 |
| **DINOv2 (Oquab et al.)** | 2023 | 대규모 자기지도 ViT 사전학습 | EoMT의 핵심 사전학습 기반 |
| **EoMT (Kerssies et al.)** | 2025 | 어댑터·디코더 없는 순수 ViT 세그멘테이션 | 본 논문 |
| **VidEoMT** | 2026 | EoMT의 비디오 세그멘테이션 확장 | EoMT의 직접적 후속 연구 |

---

### 4-3. 앞으로 연구 시 고려할 점

1. **사전학습 다양성 탐구**: DINOv2 외에 CLIP, SAM 등 다른 Foundation Model과의 호환성 및 성능 비교 연구가 필요합니다.

2. **소형 모델에서의 성능 격차 해소**: 현재 EoMT의 강점은 대규모 모델과 사전학습에서 두드러지므로, 소형 모델에서도 경쟁력 있는 방법론 개발이 필요합니다.

3. **Frozen Encoder 호환성**: PMT가 제시하듯, Foundation Model의 Frozen Encoder 특징을 보존하면서 세그멘테이션 성능을 유지하는 방향의 연구가 중요합니다.

4. **효율적 어텐션 메커니즘 통합**: FlashAttention, 토큰 병합 등 Transformer 효율화 기술과의 통합으로 추가적인 속도 및 메모리 효율 향상이 가능합니다.

5. **의료·위성·산업 도메인으로의 적용**: 제로샷 일반화 능력을 활용한 특수 도메인 적용 연구와, 해당 도메인 데이터로 파인튜닝 시의 성능 한계 분석이 필요합니다.

6. **신뢰도 추정 및 불확실성 정량화**: 분포 외 데이터에서의 신뢰도 추정 신뢰성 문제는 안전-크리티컬 응용 분야 적용 시 핵심적으로 고려해야 할 사항입니다.

---

> ⚠️ **정확도 관련 참고**: 본 분석은 공식 arXiv 논문, CVPR 2025 공식 자료, 공식 GitHub 및 프로젝트 페이지에 근거하였습니다. EoMT의 세부 수식(마스크 어텐션 어닐링 스케줄 등)은 논문에서 개념적으로 설명되나 정확한 수식 형태는 검색 결과에서 완전히 확인되지 않아, 논문의 개념을 바탕으로 일반적인 수식으로 표현하였음을 밝힙니다.
