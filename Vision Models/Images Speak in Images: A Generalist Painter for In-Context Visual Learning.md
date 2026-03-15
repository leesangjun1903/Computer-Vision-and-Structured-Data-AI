# Images Speak in Images: A Generalist Painter for In-Context Visual Learning

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장
Painter는 **"이미지가 이미지로 말한다(Images Speak in Images)"**라는 철학 아래, 컴퓨터 비전 분야에서 NLP의 in-context learning에 상응하는 패러다임을 최초로 제안한 연구이다. 핵심 주장은 다음과 같다:

> **대부분의 dense prediction 비전 태스크의 출력을 3채널 "이미지"로 재정의하고, 태스크 프롬프트 역시 이미지 쌍(입력-출력)으로 정의하면, 단일 제너럴리스트 모델이 다양한 비전 태스크를 in-context 방식으로 수행할 수 있다.**

### 주요 기여

| # | 기여 내용 |
|---|---------|
| 1 | **출력 공간의 이미지 통합**: depth estimation, semantic segmentation, instance segmentation, keypoint detection, image restoration 등 7가지 대표 비전 태스크의 출력을 $H \times W \times 3$ 크기의 "이미지"로 재정의 |
| 2 | **이미지 기반 in-context inference 프레임워크 최초 설계**: 언어 지시 없이 입출력 이미지 쌍을 태스크 프롬프트로 사용하여 수행할 태스크를 지정 |
| 3 | **극도로 단순한 학습 파이프라인**: 표준 Masked Image Modeling(MIM) 파이프라인만으로 학습, 태스크별 loss function이나 architecture 수정 불필요 |
| 4 | **in-domain 및 out-of-domain 일반화**: 훈련 시 보지 못한 카테고리/태스크에 대해서도 프롬프트만으로 빠르게 적응 가능 |
| 5 | **경쟁적 성능**: NYUv2 depth estimation에서 SOTA 달성, 다른 제너럴리스트 모델 대비 여러 태스크에서 유의미한 성능 향상 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

NLP에서 GPT-3 이후 in-context learning이 성공적으로 작동하는 이유는 두 가지이다:
1. **출력 공간의 통일**: 모든 NLP 태스크의 출력이 언어 토큰 시퀀스로 통합됨
2. **입출력 동질성**: 태스크 지시문과 예제가 모두 동일한 언어 토큰 공간에 존재

그러나 컴퓨터 비전에서는:
- 태스크마다 출력 표현이 크게 달라 (depth map, segmentation mask, keypoint 좌표 등) 통합이 어렵다
- **범용 태스크 프롬프트를 어떻게 정의할 것인지**가 불명확하다
- 기존 접근들(Pix2Seq, Unified-IO, OFA 등)은 비전 문제를 NLP로 변환(이산화)하여 해결했으나, 이는 **양자화 오차(quantization error)**를 유발하며 비전 신호의 연속적 특성을 무시한다

### 2.2 제안하는 방법

#### 2.2.1 출력 공간의 이미지 재정의

입력 이미지를 $\mathbf{x} \in \mathbb{R}^{H \times W \times 3}$, 태스크 $t$의 기존 ground truth를 $\mathbf{y}^t$, 재정의된 이미지 형태의 ground truth를 $\hat{\mathbf{y}}^t \in \mathbb{R}^{H \times W \times 3}$으로 표기한다.

**Monocular Depth Estimation (NYUv2)**

Ground truth depth 값 $\mathbf{y}^t_{i,j} \in [0, 10]$을 정수 공간 $[0, 255]$로 매핑:

$$\hat{\mathbf{y}}^t_{i,j,0} = \left\lfloor \mathbf{y}^t_{i,j} \times \frac{255}{10} \right\rfloor$$

세 채널 모두 동일한 값을 사용:

$$\hat{\mathbf{y}}^t_{i,j,0} = \hat{\mathbf{y}}^t_{i,j,1} = \hat{\mathbf{y}}^t_{i,j,2}$$

추론 시에는 세 채널의 출력을 평균하고 역변환을 수행하여 $[0, 10]$ 범위의 depth 값을 복원한다.

**Semantic Segmentation (ADE-20K)**

$L$개 카테고리를 $b$-진법으로 인코딩. $b = \lceil L^{1/3} \rceil$, 마진 $m = \lfloor 256 / b \rfloor$:

$$\hat{\mathbf{y}}^t_{i,j,0} = \left\lfloor \frac{l}{b^2} \right\rfloor \times m, \quad \hat{\mathbf{y}}^t_{i,j,1} = \left(\left\lfloor \frac{l}{b} \right\rfloor \bmod b\right) \times m, \quad \hat{\mathbf{y}}^t_{i,j,2} = (l \bmod b) \times m$$

여기서 $l \in [0, L)$은 해당 픽셀의 카테고리 인덱스이다. 예를 들어 ADE-20K ($L=150+1$)에서는 $b=6$, $m=42$이다.

**Keypoint Detection**

17개 키포인트에 대해 다음 두 가지를 분리하여 3채널에 인코딩:
- $\hat{\mathbf{y}}^t_{i,j,0}$: class-agnostic 히트맵 (가우시안 분포, 중심값 255)
- $\hat{\mathbf{y}}^t_{i,j,1}, \hat{\mathbf{y}}^t_{i,j,2}$: 17-category 키포인트 분류 (semantic segmentation과 유사한 색상 인코딩)

**Instance Segmentation**

SOLO 방식을 따라, 이미지를 $16 \times 20 \times 20$ 블록으로 개념적으로 분할하고, 각 인스턴스 마스크의 중심 위치에 따라 고정된 색상을 할당한다.

**Image Restoration** (denoising, deraining, enhancement)

입출력 모두 본래 RGB 공간이므로 별도 변환 없이 직접 통합된다.

#### 2.2.2 Masked Image Modeling (MIM) 프레임워크

**Input Format**

학습 시, 동일 태스크의 두 이미지 쌍 $\{(\mathbf{x}_1, \hat{\mathbf{y}}^t_1), (\mathbf{x}_2, \hat{\mathbf{y}}^t_2)\}$을 가로로 연결(stitch)하여 하나의 큰 이미지를 구성한다. 출력 이미지의 패치를 **75% 비율로 block-wise masking** 후, 모델이 마스크된 픽셀을 복원하도록 학습한다.

**Architecture**

- 인코더: Vanilla ViT-Large (24 블록)
- 4개 블록에서 균등하게 샘플링한 feature map을 concatenation
- 3-layer 경량 head: Linear(1×1 conv) → 3×3 conv → Linear → $16 \times 16 \times 3$ 복원
- **Patch merging**: 입력 이미지와 출력 이미지를 처음 3블록까지 병렬 처리 후, patch-by-patch feature 합산 → 계산 비용 약 50% 절감, 성능 저하 없음

**Loss Function**

마스크된 픽셀에 대해 smooth- $\ell_1$ 회귀 손실 사용:

$$\mathcal{L}_{\text{reg}} = \text{smooth-}\ell_1(\hat{\mathbf{y}}^t_{\text{pred}}, \hat{\mathbf{y}}^t_{\text{gt}}) = \begin{cases} \frac{1}{2}(\hat{\mathbf{y}}^t_{\text{pred}} - \hat{\mathbf{y}}^t_{\text{gt}})^2 / \beta, & \text{if } |\hat{\mathbf{y}}^t_{\text{pred}} - \hat{\mathbf{y}}^t_{\text{gt}}| < \beta \\ |\hat{\mathbf{y}}^t_{\text{pred}} - \hat{\mathbf{y}}^t_{\text{gt}}| - \frac{\beta}{2}, & \text{otherwise} \end{cases}$$

이 단일 loss function이 **모든 태스크에 동일하게** 적용된다.

#### 2.2.3 In-Context Inference

추론 시, 수행할 태스크의 입출력 이미지 쌍 $(\mathbf{x}\_{\text{prompt}}, \hat{\mathbf{y}}^t_{\text{prompt}})$을 **태스크 프롬프트**로 제공하고, 새로운 입력 이미지 $\mathbf{x}_{\text{query}}$와 마스크된 빈 이미지를 연결하여 모델에 입력한다. 모델은 프롬프트 쌍으로부터 **어떤 태스크를 수행해야 하는지** 파악하여 출력을 생성한다.

프롬프트 최적화를 위해 세 가지 전략 제시:
1. **Random**: 학습 데이터에서 무작위 선택
2. **Searched**: 학습 데이터 전체를 탐색하여 최적 성능 쌍 선택
3. **Learned**: 프롬프트를 학습 가능한 텐서로 정의하고, 모델 파라미터는 동결한 채 학습 loss로 프롬프트 최적화

### 2.3 모델 구조 요약

```
[Input Image + Output Image (prompt pair)] ──┐
                                              ├─ Stitch ─→ ViT-Large Encoder (24 blocks)
[Query Image + Masked Output]              ──┘         ↓
                                              Patch merging (at block 3)
                                                       ↓
                                              4 feature maps concatenation
                                                       ↓
                                              3-layer Light Head
                                                       ↓
                                              Pixel-level prediction (H × W × 3)
```

### 2.4 성능 향상

**Table 1: 주요 벤치마크 결과**

| Task | Dataset | Painter | Best Specialist | Best Generalist |
|------|---------|---------|----------------|-----------------|
| Depth Est. | NYUv2 RMSE↓ | **0.288** | 0.330 (BinsFormer) | 0.385 (Unified-IO) |
| Depth Est. | NYUv2 $\delta_1$↑ | **0.950** | 0.925 (BinsFormer) | - |
| Semantic Seg. | ADE-20K mIoU↑ | 49.9 | 57.7 (Mask2Former) | - |
| Panoptic Seg. | COCO PQ↑ | 43.4 | 57.8 (Mask2Former) | 45.8 (UViM) |
| Keypoint Det. | COCO AP↑ | 72.1 | 77.2 (HRFormer) | 64.8 (Pix2Seq v2) |
| Denoising | SIDD PSNR↑ | 38.88 | 39.89 (Uformer) | - |

**주요 성과:**
- NYUv2 depth estimation에서 **SOTA 달성** (RMSE 0.288, 기존 최고 BinsFormer 0.330 대비 12.7% 향상)
- COCO keypoint detection에서 Pix2Seq v2 대비 **+7.3 AP** 향상
- Joint training이 separate training 대비 대부분의 태스크에서 우수 (멀티태스크 시너지)
- Open-vocabulary FSS-1000에서 MAE-VQGAN(58.3 mIoU) 대비 **62.3 mIoU** 달성

### 2.5 한계

1. **Panoptic segmentation 등 어려운 태스크에서 전문 모델 대비 큰 격차**: PQ 43.4 vs. Mask2Former 57.8 (입력 해상도 차이: 448 vs. 1024)
2. **언어 신호 처리 비적합**: 비전 중심 인터페이스로 설계되어 이산적 언어 신호를 자연스럽게 모델링하기 어려움
3. **태스크 간 충돌**: Joint training에서 keypoint detection이 소폭 저하되는 등 태스크 간 interference 존재
4. **출력 공간 재정의의 정보 손실**: 연속 값을 [0, 255]로 양자화하거나, 카테고리를 색상으로 매핑할 때 표현 한계 존재
5. **프롬프트 선택/설계의 민감성**: 프롬프트에 따라 결과가 달라지며, 최적 프롬프트 탐색 전략이 아직 초기 단계

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 달성된 일반화 능력

Painter는 다음과 같은 **out-of-domain 일반화**를 시연하였다:

- **Open-vocabulary keypoint detection**: 훈련에 없던 말(horse), 원숭이(monkey), 빗자루(broom)의 키포인트 감지
- **Open-category object segmentation**: 코알라(koala), 나비(butterfly) 등 미학습 카테고리 분할
- **Open-category instance segmentation**: 태블릿(tablets) 등 새로운 객체 인스턴스 분할
- **FSS-1000 few-shot segmentation**: 1,000개 새로운 클래스에 대해 62.3 mIoU (MAE-VQGAN 대비 +4.0)

### 3.2 일반화 성능 향상의 핵심 메커니즘

**이미지 중심 통합 인터페이스**: 출력을 이미지로 통일함으로써 태스크 간 표현 공간이 공유되어, 하나의 태스크에서 학습한 시각적 패턴(edge, texture, spatial relationship)이 다른 태스크에 전이된다.

**In-context conditioning**: 프롬프트 이미지 쌍이 태스크 정의를 내포하므로, 모델은 프롬프트의 입출력 관계를 추론하여 새로운 태스크에 적응할 수 있다. 이는 NLP의 in-context learning과 유사한 **meta-learning 효과**를 갖는다.

**Joint training 시너지**: Table 2에서 보듯이, 7개 태스크를 함께 학습하면 대부분의 태스크에서 개별 학습보다 성능이 향상된다. 이는 서로 다른 태스크가 공유하는 시각적 특징(geometric structure, semantic understanding)을 상호 보완적으로 학습함을 시사한다.

### 3.3 일반화 성능 향상을 위한 주요 방향

1. **프롬프트 최적화**: Learned prompt가 random prompt 대비 일관된 향상을 보여줌 (Table 3). 향후 더 정교한 프롬프트 생성/검색 전략이 일반화에 핵심적일 수 있다.
   - 현재: RMSE 0.291 (random) → 0.286 (learned) on NYUv2

2. **스케일링**: ViT-B에서 ViT-L로 확장 시 mIoU 31.4 → 41.2 (Table S2b), 더 큰 모델이 일반화에 유리할 가능성이 높다.

3. **더 많은 태스크/데이터 추가**: 모델 아키텍처나 loss 변경 없이 새로운 데이터 쌍만 추가하면 되므로, 태스크/데이터 확장이 용이하다.

4. **입력 해상도 증가**: 현재 $448 \times 448$로 제한되어 있어, 해상도 증가 시 panoptic segmentation 등에서 큰 성능 향상 예상된다.

5. **다중 프롬프트 활용**: 단일 프롬프트가 아닌 복수의 예시를 활용하는 few-shot in-context 방식의 탐색이 가능하다.

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 연구적 영향

1. **비전 in-context learning의 가능성 입증**: NLP에서만 가능하다고 여겨졌던 in-context learning이 비전에서도 이미지 인터페이스를 통해 실현 가능함을 최초로 체계적으로 보여주었다.

2. **태스크 통합 패러다임의 전환**: 기존의 이산 토큰 기반(Pix2Seq, Unified-IO) 접근에서 **연속 이미지 공간 기반** 접근으로의 대안을 제시하였다.

3. **Generalist Vision Model의 설계 단순화**: 태스크별 head, loss function, architecture 변경 없이 단일 MIM 프레임워크로 통합할 수 있음을 보여주었다.

4. **프롬프트 엔지니어링의 비전 확장**: NLP에서의 prompt tuning 개념이 비전에서도 이미지 프롬프트 형태로 적용 가능함을 시연하였다.

### 4.2 향후 연구 시 고려할 점

| 항목 | 고려 사항 |
|------|----------|
| **멀티모달 통합** | 비전 중심 인터페이스와 언어 인터페이스의 결합 방법 탐색 필요 |
| **태스크 간 간섭** | Joint training에서의 태스크 충돌 완화를 위한 학습 전략 연구 |
| **출력 인코딩 정밀도** | [0, 255] 양자화의 한계를 넘는 고정밀 출력 인코딩 방법 탐구 |
| **효율성** | 프롬프트 쌍 포함 시 입력 크기가 커지는 문제의 해결 |
| **평가 프로토콜** | In-context visual learning의 표준화된 평가 벤치마크 구축 |
| **프롬프트 설계** | 최적 프롬프트 자동 생성 및 선택 알고리즘 개발 |
| **스케일링 법칙** | 모델 크기, 데이터 양, 태스크 수에 따른 성능 스케일링 특성 분석 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 비교 대상 연구 개관

| 연구 | 연도 | 핵심 접근 | 인터페이스 | 태스크 범위 |
|------|------|----------|----------|-----------|
| **Pix2Seq** (Chen et al.) | 2021 | 객체 검출을 언어 모델링으로 | 이산 토큰 시퀀스 | 검출 |
| **Pix2Seq v2** (Chen et al.) | 2022 | 비전 태스크 통합 시퀀스 인터페이스 | 이산 토큰 시퀀스 | 검출, 분할, 키포인트, 캡셔닝 |
| **Unified-IO** (Lu et al.) | 2022 | 비전+언어+멀티모달 통합 | 이산 토큰 (T5 스타일) | 비전, 언어, 멀티모달 |
| **OFA** (Wang et al.) | 2022 | Seq2Seq 프레임워크 | 이산 토큰 | 비전, 언어, 멀티모달 |
| **UViM** (Kolesnikov et al.) | 2022 | 학습된 가이딩 코드 기반 | 이산 코드 | 픽셀 레이블링 (태스크별 별도 모델) |
| **MAE-VQGAN** (Bar et al.) | 2022 | Image inpainting 기반 visual prompting | 이산 공간 | Foreground seg., 검출, colorization |
| **Flamingo** (Alayrac et al.) | 2022 | 비전-언어 few-shot | 언어 출력 | VQA, 캡셔닝 등 |
| **Painter** (본 논문) | 2023 | 출력의 이미지 재정의 + MIM | 연속 이미지 | 7가지 dense prediction |
| **SegGPT** (Wang et al.) | 2023 | Painter 확장, 분할 특화 | 연속 이미지 | 범용 분할 |
| **Segment Anything (SAM)** (Kirillov et al.) | 2023 | Promptable segmentation | 점/박스/텍스트 프롬프트 | 범용 분할 |

### 5.2 핵심 차이점 분석

**이산 vs. 연속 출력 공간**

$$\text{Pix2Seq, Unified-IO, OFA:} \quad \mathbf{y} \in \mathcal{V}^{N} \quad (\text{이산 토큰 시퀀스})$$
$$\text{Painter:} \quad \hat{\mathbf{y}}^t \in [0, 255]^{H \times W \times 3} \quad (\text{연속 이미지 공간})$$

Painter는 비전 신호의 연속적 특성을 보존하여 양자화 오차를 줄이는 장점이 있지만, 언어 태스크와의 통합에는 불리하다. 반면 Unified-IO, OFA는 언어 태스크까지 포함할 수 있으나 비전 성능에서 양자화 손실이 발생한다.

**태스크 프롬프트 방식**

- **Pix2Seq v2, Unified-IO**: 언어 토큰이나 특수 토큰으로 태스크 지정
- **MAE-VQGAN**: 이미지 inpainting 기반이나 이산 공간에서만 작동
- **Painter**: 입출력 이미지 쌍 자체가 프롬프트 → **언어 이해 불필요**, out-of-domain 적응 유연

**일반화 능력 비교**

- **UViM**: 태스크별 별도 모델 학습 → in-context 일반화 불가
- **Pix2Seq v2**: 학습된 태스크에만 적용 가능
- **MAE-VQGAN**: in-context 개념 증명 수준, 표준 벤치마크 결과 없음
- **Painter**: 표준 벤치마크에서 경쟁적 성능 + out-of-domain 일반화 시연

**후속 연구: SegGPT와 SAM**

Painter의 후속 연구인 **SegGPT** (Wang et al., 2023)은 Painter의 프레임워크를 분할 태스크에 특화하여 확장하였으며, **SAM** (Kirillov et al., 2023)은 점/박스/텍스트 프롬프트를 사용한 범용 분할 모델로, Painter와는 다른 방향에서 비전 태스크의 일반화를 추구하였다. Painter가 제시한 "이미지를 이미지로" 패러다임은 SegGPT에 직접 계승되었으며, SAM의 promptable segmentation 개념과도 철학적으로 연결된다.

### 5.3 종합 평가

Painter는 **비전 분야 in-context learning의 선구적 연구**로서, 다음과 같은 점에서 차별화된다:

1. 언어 의존성을 완전히 제거한 **순수 비전 인터페이스** 제안
2. **연속 출력 공간**을 통한 정보 보존
3. 학습 파이프라인의 **극단적 단순성** (MIM + smooth- $\ell_1$ )
4. 실제 표준 벤치마크에서의 **경쟁적 성능 입증**

동시에, 전문 모델 대비 성능 격차(특히 panoptic segmentation), 언어 통합의 어려움, 출력 인코딩의 정밀도 한계 등은 향후 연구에서 해결해야 할 과제로 남아 있다.

---

## 참고자료

1. **Wang, X., Wang, W., Cao, Y., Shen, C., & Huang, T.** (2023). "Images Speak in Images: A Generalist Painter for In-Context Visual Learning." *arXiv:2212.02499v2*. [https://arxiv.org/abs/2212.02499](https://arxiv.org/abs/2212.02499)
2. **Brown, T., et al.** (2020). "Language Models are Few-Shot Learners." *arXiv:2005.14165* (GPT-3)
3. **Chen, T., et al.** (2021). "Pix2Seq: A Language Modeling Framework for Object Detection." *ICLR 2022*
4. **Chen, T., et al.** (2022). "A Unified Sequence Interface for Vision Tasks." *NeurIPS 2022* (Pix2Seq v2)
5. **Lu, J., et al.** (2022). "Unified-IO: A Unified Model for Vision, Language, and Multi-Modal Tasks." *arXiv:2206.08916*
6. **Wang, P., et al.** (2022). "OFA: Unifying Architectures, Tasks, and Modalities through a Simple Sequence-to-Sequence Learning Framework." *ICML 2022*
7. **Kolesnikov, A., et al.** (2022). "UViM: A Unified Modeling Approach for Vision with Learned Guiding Codes." *NeurIPS 2022*
8. **Bar, A., et al.** (2022). "Visual Prompting via Image Inpainting." *NeurIPS 2022* (MAE-VQGAN)
9. **Alayrac, J.-B., et al.** (2022). "Flamingo: A Visual Language Model for Few-Shot Learning." *NeurIPS 2022*
10. **He, K., et al.** (2021). "Masked Autoencoders Are Scalable Vision Learners." *arXiv:2111.06377* (MAE)
11. **Cheng, B., et al.** (2021). "Masked-Attention Mask Transformer for Universal Image Segmentation." *arXiv:2112.01527* (Mask2Former)
12. **Dosovitskiy, A., et al.** (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR 2021* (ViT)
13. **Wang, X., et al.** (2023). "SegGPT: Segmenting Everything In Context." *arXiv:2304.03284*
14. **Kirillov, A., et al.** (2023). "Segment Anything." *arXiv:2304.02643* (SAM)
15. **Girshick, R.** (2015). "Fast R-CNN." *CVPR 2015* (smooth- $\ell_1$ loss 원본)
