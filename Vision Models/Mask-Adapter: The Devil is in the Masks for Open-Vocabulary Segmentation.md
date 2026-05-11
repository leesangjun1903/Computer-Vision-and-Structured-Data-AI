
# Mask-Adapter: The Devil is in the Masks for Open-Vocabulary Segmentation

> **논문 정보**
> - **저자:** Yongkang Li, Tianheng Cheng, Bin Feng, Wenyu Liu, Xinggang Wang (HUST)
> - **학회:** CVPR 2025 (pp. 14998–15008)
> - **arXiv:** 2412.04533 (v1: 2024.12.05, v2: 2025.03.10)
> - **코드:** https://github.com/hustvl/MaskAdapter

---

## 1. 핵심 주장 및 주요 기여 요약

### 🎯 핵심 주장

최근 open-vocabulary segmentation 연구들은 마스크 생성기(mask generator)가 예측한 세그멘테이션 마스크를 CLIP 같은 사전학습 비전-언어 모델(VLM)로 분류하는 **mask pooling** 방식을 채택하고 있다. 그러나 이 방식에는 반직관적인 문제가 존재하는데, **정확한 마스크를 사용하더라도 CLIP 이미지 임베딩을 해당 마스크 영역 내에서 풀링(pooling)하면 정확한 분류 결과를 얻지 못한다**는 점이다.

이 논문의 핵심 주장은 바로 이 **"mask pooling의 구조적 한계"**를 밝히고, 이를 해결하는 Mask-Adapter를 제안하는 것이다.

### 🏆 주요 기여

| 기여 항목 | 설명 |
|---|---|
| ① Mask Pooling의 한계 규명 | GT 마스크 사용 시에도 upper bound 성능이 낮음을 실험적으로 증명 |
| ② Mask-Adapter 설계 | Semantic Activation Map 기반 plug-and-play 모듈 |
| ③ Mask Consistency Loss 제안 | 마스크 변화에 robust한 임베딩 학습 |
| ④ SAM 확장 | 별도 학습 없이 SAM과 통합 가능 |
| ⑤ 다수 벤치마크 SOTA | 여러 제로샷 벤치마크에서 성능 향상 |

---

## 2. 해결 문제 · 제안 방법(수식 포함) · 모델 구조 · 성능 및 한계

### 2-1. 해결하고자 하는 문제

기존 mask embedding 추출 방법은 크게 두 가지로, (1) **Mask Cropping**: 세그멘테이션된 영역을 이미지에서 잘라내어 CLIP에 입력하는 방식, (2) **Mask Pooling**: 제안 마스크와의 dot-product를 통해 영역 특징을 직접 집계하는 더 효율적인 방식이 있었다. 그러나 **두 방법 모두 open-vocabulary segmentation의 성능 상한(upper bound)이 본질적으로 제한**된다는 문제가 있었다.

ADE20K GT 마스크를 입력으로 사용하여 서로 다른 추출 방법의 upper bound를 평가한 결과, mask cropping과 mask pooling은 GT 마스크에서도 제한된 성능을 보인 반면, **Mask-Adapter는 open-vocabulary segmentation의 upper bound를 크게 향상**시켰다.

### 2-2. 제안 방법 및 핵심 수식

#### (a) 핵심 설계: Semantic Activation Map 추출

직접 제안 마스크를 사용하는 것과 달리, Mask-Adapter는 제안 마스크로부터 **semantic activation map을 추출**하여 더 풍부한 문맥 정보(contextual information)를 제공하고 마스크와 CLIP 간의 정렬(alignment)을 보장한다.

$N$개의 클래스 비의존(class-agnostic) 마스크 $M_p$가 주어지면, 두 개의 stride가 있는 $3 \times 3$ 컨볼루션 레이어로 구성된 간단한 블록을 통해 이진 마스크를 패치로 변환하여 mask feature $F_m$을 생성한다.

구체적으로 마스크 임베딩 생성 과정은 다음과 같이 정리된다:

$$F_m = \text{PatchEmbed}(M_p) \quad \in \mathbb{R}^{N \times C \times H \times W}$$

$$A = \text{ConvNeXt}(F_m + F_{\text{CLIP}}) \quad \leftarrow \text{Semantic Activation Maps}$$

$$e_{\text{mask}} = \text{Pool}(F_{\text{CLIP}} \odot A) \quad \leftarrow \text{Mask Embedding}$$

$$\text{score}(c) = \cos(e_{\text{mask}},\, e_{\text{text}}^{(c)}) \quad \leftarrow \text{분류 결과}$$

마스크와 CLIP 특징을 Mask-Adapter에 입력하면 **semantic activation map이 생성**되고, 이는 각 마스크에 대해 정보가 풍부한 영역을 강조(highlight)한다. 이후 CLIP 특징에서 특별한 정보를 pooling을 통해 수집하여 마스크 임베딩을 얻고, 이를 텍스트 임베딩과 매칭하여 분류 결과를 도출한다.

#### (b) Mask Consistency Loss

Mask Consistency Loss는 **비슷한 IoU를 가진 제안 마스크들이 유사한 CLIP 임베딩을 얻도록** 유도함으로써, 다양한 예측 마스크에 대한 모델의 robust성을 향상시킨다.

수식으로 표현하면:

$$\mathcal{L}_{\text{cos}} = 1 - \frac{e_{\text{mask}}^{(i)} \cdot e_{\text{mask}}^{(j)}}{\|e_{\text{mask}}^{(i)}\| \, \|e_{\text{mask}}^{(j)}\|}, \quad \text{if } \text{IoU}(M_i, M_j) > \tau$$

그리고 전체 학습 손실은 다음과 같이 구성된다:

전체 학습 손실은 다음과 같이 정의된다:

$$\mathcal{L} = \lambda_{ce} \cdot \mathcal{L}_{ce} + \lambda_{cos} \cdot \mathcal{L}_{cos}$$

여기서 $\mathcal{L}\_{ce}$는 교차 엔트로피 분류 손실, $\mathcal{L}_{cos}$는 마스크 일관성(코사인 유사도) 손실이다.

#### (c) Geometric Ensemble (Inference)

추론 시, Mask-Adapter의 출력 $\hat{y}\_{\text{out}}$와 기본 모델 내부 출력 $\hat{y}_{\text{in}}$에 대해 **geometric ensemble** 전략을 적용한다:

$$\hat{y}(c) = \begin{cases} \hat{y}_{\text{in}}(c)^{1-\alpha} \cdot \hat{y}_{\text{out}}(c)^{\alpha}, & \text{if } c \in C_{\text{seen}} \\ \hat{y}_{\text{in}}(c)^{1-\beta} \cdot \hat{y}_{\text{out}}(c)^{\beta}, & \text{if } c \in C_{\text{unseen}} \end{cases}$$

FC-CLIP 기준으로 $\alpha = 0.7$, $\beta = 0.9$를 사용하며, MAFTP-Base는 $\alpha=0.7,\,\beta=1.0$, MAFTP-Large는 $\alpha=0.8,\,\beta=1.0$을 사용한다. 이 geometric ensemble은 seen/unseen 양쪽 예측의 강점을 균형 있게 결합한다.

#### (d) IoU-based Matcher

Mask Consistency Loss로 유사한 마스크에서 유사한 임베딩을 강제하고, 과적합(overfitting)을 더욱 완화하기 위해 **Hungarian matcher를 IoU-based matcher로 대체**하여 더 넓은 범위의 제안 마스크에서 모델을 학습할 수 있게 한다.

### 2-3. 모델 구조

Mask-Adapter는 (a) open-vocabulary segmentation에 seamless하게 통합될 수 있으며, CLIP 특징과 제안 마스크로부터 semantic activation map을 추출한다. 제안 마스크와 CLIP 특징이 Mask-Adapter를 통과하여 semantic activation map을 추출하고, 이 강조된 영역과 문맥 정보를 기반으로 CLIP 특징을 집계하여 마스크 임베딩을 구성한다.

```
[입력 이미지]
    │
    ├─ CLIP Image Encoder ──→ CLIP Features (F_CLIP)
    │
    └─ Mask Generator ──────→ Proposal Masks (M_p)
                                        │
                              ┌─────────▼──────────┐
                              │    Mask-Adapter     │
                              │  ┌───────────────┐  │
                              │  │PatchEmbed(M_p)│  │
                              │  │(2x stride 3x3 │  │
                              │  │    conv)      │  │
                              │  └──────┬────────┘  │
                              │         ↓ +F_CLIP   │
                              │  ┌──────────────┐   │
                              │  │ConvNeXt Block│   │
                              │  └──────┬───────┘   │
                              │  Semantic Act. Maps  │
                              └─────────┬────────────┘
                                        │ Pool(F_CLIP ⊙ A)
                                        ▼
                                  Mask Embeddings
                                        │
                              ┌─────────▼──────────┐
                              │  Text Embeddings   │←── CLIP Text Encoder
                              │  (Cosine Matching) │
                              └────────────────────┘
                                        │
                                  Classification
```

Mask-Adapter에서는 주로 백본과 일관된 **ConvNeXt block**을 채택하였으며, ConvNeXt 구조가 다른 블록 설정에 비해 우월한 성능을 보이고 CNN 기반 블록이 일반적으로 **dense prediction 태스크에 더 적합**하다는 ablation study 결과를 제시한다.

Mask-Adapter에서는 각 마스크에 대해 **16개의 semantic activation map**을 추출하고, 대응하는 CLIP 특징을 별도로 집계한 후 평균을 계산한다. 이 설계는 문맥 노이즈(contextual noise)를 효과적으로 완화한다.

### 2-4. 성능 향상

ablation 실험에서, FC-CLIP의 Mask Pooling을 GT 학습된 Mask-Adapter로 교체하면 **seen 카테고리 mIoU가 10.1, unseen 카테고리 mIoU가 5.3 향상**된다. Ground-Truth Warmup과 Mask Consistency Loss를 도입하면 각각 mIoU가 0.3과 0.8 추가 향상된다.

GT 마스크를 이용한 상한 실험에서 Mask-Adapter는 **74.1%의 정확도**를 달성하여 기존 방법들(mask cropping, mask pooling 등)을 크게 상회하며, CLIP의 open-vocabulary 인식 능력을 마스크 분류로 전이하는 효과가 탁월함을 보인다.

세 개의 기준선(baseline)과 비교 시, mIoU를 각각 1.1, 0.4, 0.9 향상시키며, 대형 VLM에서는 Mask-Adapter+MAFTP 조합이 당시 SOTA 방법을 능가한다.

또한 Mask-Adapter는 **학습 없이 SAM(Segment Anything Model)으로 효과적으로 확장** 가능하며, 여러 open-vocabulary segmentation 벤치마크에서 인상적인 결과를 달성한다.

### 2-5. 한계

SAM으로 확장 시, SAM의 지나치게 **세밀한(fine-grained) 마스크 출력**이 open-vocabulary semantic segmentation 태스크의 성능에 부정적인 영향을 줄 수 있다. 이 한계를 해결하는 것은 미해결 문제(open problem)로 남아 있으며, 이 논문의 범위를 벗어난다고 명시하고 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

기존 접근법과 달리 Mask-Adapter는 여러 핵심 장점을 갖는다: **(1)** 배경을 무시하는 대신 전체 이미지에서 마스크 임베딩을 집계하여 문맥 정보를 풍부하게 한다. **(2)** 단순히 타겟 영역의 위치 정보만 전달하는 mask pooling과 달리, semantic activation map은 인식에 관련된 정보가 풍부한 영역을 선택적으로 강조하고 정보가 적은 영역을 억제하여 특징 변별력을 높인다. **(3)** 학습 중 CLIP의 일반화 능력을 보존하면서 동시에 마스크 인식 성능을 향상시킨다.

보이지 않은 카테고리에 대한 모델의 일반화 성능을 보다 정밀하게 평가하기 위해, ablation 실험에서 **seen 카테고리와 unseen 카테고리 각각의 mIoU($\text{mIoU}_s$, $\text{mIoU}_u$)를 별도로 보고**한다.

Mask Consistency 제약의 적용은 **클래스 간 거리(inter-class distance)를 증가시키고 마스크 임베딩의 변별력을 향상**시켜, 모델의 마스크 인식 능력을 개선한다.

단일 semantic activation map과 다중 semantic activation map을 비교한 결과, **다중 semantic activation map의 사용이 과도한 문맥 노이즈를 효과적으로 줄이고 마스크 인식 능력을 향상**시킨다.

일반화 성능과 관련된 핵심 메커니즘을 정리하면:

| 메커니즘 | 일반화 기여 |
|---|---|
| Semantic Activation Map | CLIP의 zero-shot 특징 공간 최대 활용 |
| Mask Consistency Loss | 마스크 품질 변화에 robust한 임베딩 학습 |
| IoU-based Matcher | 다양한 마스크 품질에서 학습 → 분포 변화(distribution shift)에 강건 |
| GT Warmup Training | 고품질 마스크 정보로 초기 파라미터 최적화 |
| SAM 확장 | 학습 불필요, 범용 마스크 생성기와 결합 |
| Geometric Ensemble | Seen/Unseen 분리 조정으로 미학습 카테고리 성능 강화 |

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4-1. Open-Vocabulary Segmentation 연구 흐름

| 연구 | 연도/학회 | 핵심 접근 | Mask-Adapter와의 관계 |
|---|---|---|---|
| **MaskCLIP** | ECCV 2022 | CLIP을 zero-shot dense prediction에 활용 | 기반 기술 제공 |
| **OVSeg (Mask-adapted CLIP)** | CVPR 2023 | CLIP을 masked image에 파인튜닝 | mask pooling 한계 보완 시도 |
| **FC-CLIP** | NeurIPS 2024 | 단일 frozen ConvNeXt CLIP으로 OVS | Mask-Adapter의 주요 baseline |
| **MAFTP** | - | 대형 VLM 기반 OVS | Mask-Adapter와 결합하여 SOTA |
| **SAM** | ICCV 2023 | 범용 class-agnostic 마스크 생성 | Mask-Adapter가 SAM 분류 능력 부여 |
| **Mask-Adapter** | **CVPR 2025** | Semantic Activation Map 기반 plug-and-play | - |

OVSeg(Mask-adapted CLIP)은 클래스 비의존 마스크 제안을 생성한 후 사전학습 CLIP으로 마스크 영역을 분류하는 2단계 방식을 사용하는데, 이 패러다임의 병목이 **마스크 이미지에서 잘 동작하지 않는 CLIP 모델** 자체에 있음을 파악하고, masked image region과 텍스트 설명 쌍으로 CLIP을 파인튜닝하는 방식을 제안한다.

Mask-Adapter는 이와 달리 CLIP을 직접 수정하지 않고, **CLIP의 특징을 semantic activation map을 통해 재가공**함으로써 CLIP의 일반화 능력을 보존하면서도 마스크 인식 성능을 높인다는 점에서 차별화된다.

Mask-Adapter는 detectron2, Mask2Former, FC-CLIP, MAFTP 등의 프로젝트를 기반으로 구현되어 있으며, 이들 위에서 plug-and-play 방식으로 동작한다.

---

## 5. 향후 연구에 미치는 영향 및 고려 사항

### 5-1. 이 논문이 앞으로의 연구에 미치는 영향

1. **Mask Embedding 패러다임 전환의 촉매**
   이 논문은 mask pooling의 성능 한계를 명확히 밝혀냄으로써, open-vocabulary segmentation에서 mask embedding 추출 방법론의 근본적 재검토를 촉구한다. 이후 연구들(MaskCLIP++, FGA-Seg 등)이 이 방향으로 발전하는 데 직접적 영향을 미칠 것이다.

2. **Plug-and-Play 어댑터 설계 방향 제시**
   Mask-Adapter는 mask pooling 기반 open-vocabulary segmentation 방법에 plug-and-play 방식으로 seamlessly 통합되어 더 정확한 분류 결과를 제공한다. 이는 기존 모델 구조를 최소한으로 변경하면서 성능을 향상시키는 "경량 어댑터" 연구 방향을 강화한다.

3. **SAM + CLIP 융합 연구의 기반**
   Mask-Adapter의 SAM 통합 프레임워크는 SAM이 class-agnostic 마스크를 생성하고 CLIP이 특징을 추출하면, Mask-Adapter가 이를 처리하여 semantic activation map과 마스크 임베딩을 생성하고 텍스트 임베딩과 매칭하여 분류를 수행하는 구조를 제시한다. 이는 foundation model 결합 연구에 중요한 방향을 제시한다.

4. **일반화 평가 기준 구체화**
   Seen/unseen 카테고리를 분리하여 mIoU를 각각 보고하는 평가 방식은 open-vocabulary segmentation 일반화 성능 측정의 표준화에 기여한다.

### 5-2. 앞으로 연구 시 고려할 점

1. **SAM의 과세밀 마스크 문제 해결**
   SAM의 지나치게 세밀한 마스크 출력이 성능에 부정적 영향을 미치는 것은 미해결 과제이므로, 마스크 병합(merging) 또는 계층적 마스크 선택 전략 연구가 필요하다.

2. **$\alpha$, $\beta$ Hyperparameter 자동 조정**
   Geometric ensemble의 $\alpha$, $\beta$ 값이 모델별로 수동 설정되므로, 이를 학습 가능하게 하거나 meta-learning 방식으로 자동 조정하는 연구가 유효하다.

3. **효율성 개선**
   16개의 semantic activation map 생성은 계산 비용을 증가시키므로, 적응적 map 수 조정(adaptive map selection) 연구가 필요하다.

4. **다중 태스크 확장**
   현재 주로 semantic segmentation에 집중되어 있으므로, instance segmentation 및 panoptic segmentation으로의 균형 잡힌 확장 연구가 요구된다.

5. **도메인 특화 일반화 평가**
   PASCAL VOC와 COCO-Stuff 간 높은 카테고리 중복으로 인해 실질적인 zero-shot 평가가 어렵다는 점을 고려하여, PASCAL VOC와 PASCAL-Context(PC-59)는 COCO-Stuff와 약 0.9의 Hausdorff similarity를 가지며 PASCAL VOC의 모든 카테고리가 COCO-Stuff와 중복되어 미학습 카테고리 성능 평가가 제한됨을 인지하고, 더 엄격한 zero-shot 벤치마크 설계가 필요하다.

---

## 📚 참고 자료 및 출처

| 번호 | 제목 | 출처 |
|---|---|---|
| 1 | **Mask-Adapter: The Devil is in the Masks for Open-Vocabulary Segmentation** | arXiv:2412.04533 (CVPR 2025) — https://arxiv.org/abs/2412.04533 |
| 2 | CVPR 2025 Open Access (논문 PDF) | https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Mask-Adapter_... |
| 3 | CVPR 2025 Supplemental PDF | https://openaccess.thecvf.com/content/CVPR2025/supplemental/Li_Mask-Adapter_... |
| 4 | 공식 GitHub Repository (hustvl/MaskAdapter) | https://github.com/hustvl/MaskAdapter |
| 5 | arXiv HTML (v2 full text) | https://arxiv.org/html/2412.04533v2 |
| 6 | CVPR 2025 Poster Page | https://cvpr.thecvf.com/virtual/2025/poster/35217 |
| 7 | Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP (OVSeg) | arXiv:2210.04150 — https://arxiv.org/abs/2210.04150 |
| 8 | Awesome-Open-Vocabulary-Semantic-Segmentation (관련 연구 목록) | https://github.com/Qinying-Liu/Awesome-Open-Vocabulary-Semantic-Segmentation |
| 9 | MaskCLIP++: A Mask-Based CLIP Fine-tuning Framework | arXiv:2412.11464 — https://arxiv.org/html/2412.11464v1 |

> ⚠️ **주의사항**: 일부 세부 수식(예: 마스크 임베딩 집계의 정확한 형식)은 공개 자료에서 완전히 확인되지 않아 논문의 기술 설명을 바탕으로 재구성하였습니다. 정확한 수식 확인은 CVPR 2025 공식 논문 PDF를 직접 참조하시기 바랍니다.
