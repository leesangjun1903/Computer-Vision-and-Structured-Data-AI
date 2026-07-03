# Self-Supervised CycleGAN for Object-Preserving Image-to-Image Domain Adaptation

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

이 논문은 **CycleGAN 기반의 이미지-투-이미지(I2I) 변환 시 발생하는 객체 왜곡(content distortion) 문제**를 **추가적인 픽셀 단위 어노테이션 없이** 자기지도학습(self-supervised learning)으로 해결하는 새로운 프레임워크 **OP-GAN(Object-Preserving GAN)**을 제안합니다.

### 주요 기여

| 기여 | 내용 |
|------|------|
| **어노테이션-프리 객체 보존** | 픽셀 단위 레이블 없이 이미지 콘텐츠 일관성 유지 |
| **멀티태스크 자기지도 학습** | 콘텐츠 등록(content registration) + 도메인 분류(domain classification) |
| **샴 네트워크 통합** | 소스/변환 이미지 패치 간 특징 분리(disentanglement) |
| **범용성 검증** | 자율주행(CamVid), 합성→실세계(SYNTHIA), 의료영상(Colonoscopy) 3개 데이터셋 검증 |

---

## 2. 상세 설명

### 2-1. 해결하고자 하는 문제

**CycleGAN의 기하학적 변환 모호성(Geometric Transformation Ambiguity)**

CycleGAN은 두 가지 손실함수로 학습됩니다.

**① 적대적 손실 (Adversarial Loss):**

$$\mathcal{L}_{adv}(G_{AB}, D_B) = \mathbb{E}_{x_B \sim p_{x_B}}\left[(D_B(x_B) - 1)^2\right] + \mathbb{E}_{x_A \sim p_{x_A}}\left[(D_B(G_{AB}(x_A)))^2\right] \tag{1}$$

**② 사이클 일관성 손실 (Cycle-Consistency Loss):**

$$\mathcal{L}_{cyc}(G_{AB}, G_{BA}) = \mathbb{E}_{x_A \sim p_{x_A}}\left[\|G_{BA}(G_{AB}(x_A)) - x_A\|_1\right] + \mathbb{E}_{x_B \sim p_{x_B}}\left[\|G_{AB}(G_{BA}(x_B)) - x_B\|_1\right] \tag{2}$$

**근본적 한계:** 임의의 전단사 기하학적 변환 $T$와 역변환 $T^{-1}$에 대해 다음 생성자도 사이클 일관성을 만족합니다:

$$G'_{AB} = G_{AB}T, \quad G'_{BA} = G_{BA}T^{-1} \tag{3}$$

즉, 사이클 일관성만으로는 **객체의 위치/형태 왜곡을 막을 수 없습니다.** 기존 해결책(AugGAN, Zhang et al.)은 픽셀 단위 세그멘테이션 어노테이션을 요구해 실용성이 제한됩니다.

---

### 2-2. 제안하는 방법

#### 전체 목적 함수

$$\mathcal{L}(G_{AB}, G_{BA}, D_A, D_B, S) = \mathcal{L}_{adv}(G_{BA}, D_A) + \mathcal{L}_{adv}(G_{AB}, D_B) + \alpha\mathcal{L}_{cyc}(G_{AB}, G_{BA}) + \beta\mathcal{L}_S(G_{AB}, G_{BA}, S) \tag{7}$$

여기서 $\alpha = 10$, $\beta = 1$ (휴리스틱 설정), $S$는 샴 네트워크입니다.

자기지도 손실은:

$$\mathcal{L}_S = \mathcal{L}_{cc} + \mathcal{L}_{dc} \tag{합산}$$

#### 핵심 메커니즘: 멀티태스크 자기지도 학습

**패치 분할 전략:**
- 소스 이미지 $A$와 변환 이미지 $B$를 각각 $3 \times 3$ 그리드로 분할
- 패치 풀: $P \in \{A_1, \ldots, A_9\} \cup \{B_1, \ldots, B_9\}$

**4가지 패치 선택 시나리오:**

| 시나리오 | 설명 |
|---------|------|
| $D_1$ | 두 패치 모두 소스 이미지에서 추출 |
| $D_2$ | 두 패치 모두 변환 이미지에서 추출 |
| $C_1$ | 소스와 변환 이미지의 **같은 위치** 패치 |
| $C_2$ | 소스와 변환 이미지의 **다른 위치** 패치 |

**특징 임베딩:**

$$E_A : P_A \rightarrow Z(c_A, d_A), \quad E_B : P_B \rightarrow Z(c_B, d_B) \tag{4}$$

여기서 $c_A, c_B$는 콘텐츠 특징, $d_A, d_B$는 도메인 특징 (각각 $11 \times 11 \times 512$ 크기)

#### Task 1: 콘텐츠 등록 손실 (Content Consistency Loss)

시나리오 $C_1$에서만 계산 (같은 위치 패치):

$$\mathcal{L}_{cc} = \frac{1}{M \times N} \sum_{x=1}^{M} \sum_{y=1}^{N} (\tilde{p}^A_{x,y} - \tilde{p}^B_{x,y})^2 \tag{5}$$

여기서 $\tilde{p}$는 콘텐츠 어텐션 맵(content attention map), $(x,y)$는 픽셀 좌표입니다.

#### Task 2: 도메인 분류 손실 (Domain Classification Loss)

3-클래스 분류($D_1, D_2, C = \{C_1, C_2\}$)에 대한 교차 엔트로피:

$$\mathcal{L}_{dc} = -\sum_i \log\left(\frac{e^{p_{g_i}}}{\sum_j e^{p_j}}\right) \tag{6}$$

---

### 2-3. 모델 구조

```
[소스 이미지 A] → G_AB → [변환 이미지 B]
                              ↓
                    3×3 패치 분할 (A, B 각각)
                              ↓
              ┌───────────────────────────────┐
              │       샴 네트워크 (S)          │
              │  [공유 가중치 인코더 × 2]      │
              │       ↙           ↘           │
              │  콘텐츠 특징      도메인 특징   │
              │  (c_A, c_B)      (d_A, d_B)  │
              │       ↓               ↓       │
              │  콘텐츠 등록     도메인 분류   │
              │  브랜치(L_cc)    브랜치(L_dc)  │
              └───────────────────────────────┘
                              ↓
              G_AB, G_BA 업데이트 (객체 보존 강화)
```

- **생성자/판별자**: CycleGAN과 동일한 구조 (Instance Normalization + PatchGAN)
- **샴 네트워크 인코더**: 컨볼루션 레이어로 구성, $11 \times 11 \times 512$ 특징 맵 출력
- **최적화 순서**: $S$와 $D_A/D_B$ 고정 → $G_{BA}/G_{AB}$ 업데이트 → $G_{BA}/G_{AB}$ 고정 → $S$, $D_A/D_B$ 업데이트

---

### 2-4. 성능 향상

**CamVid (Cloudy→Sunny) - PSPNet 기반 mIoU(%):**

| 방법 | mIoU |
|------|------|
| Direct Transfer | 41.86 |
| UNIT | 19.94 |
| DRIT | 18.20 |
| CycleGAN | 26.26 |
| **OP-GAN (제안)** | **51.40** |
| AugGAN (어노테이션 사용) | 55.31 |

**SYNTHIA (Night→Day) - PSPNet 기반 mIoU(%):**

| 방법 | mIoU |
|------|------|
| Direct Transfer | 44.49 |
| UNIT | 34.97 |
| DRIT | 12.66 |
| CycleGAN | 22.88 |
| **OP-GAN (제안)** | **50.86** |

**Ablation Study (CamVid mIoU %):**

| 설정 | mIoU |
|------|------|
| Direct Transfer | 41.86 |
| CycleGAN | 26.26 |
| CycleGAN + $\mathcal{L}_{cc}$ | 45.63 |
| CycleGAN + $\mathcal{L}_{dc}$ | 45.86 |
| **CycleGAN + $\mathcal{L}\_{cc}$ + $\mathcal{L}_{dc}$ (OP-GAN)** | **51.40** |

---

### 2-5. 한계

1. **그리드 크기 고정**: $3\times3$ 패치 분할이 항상 최적이 아닐 수 있음 (논문 Appendix에서 분석)
2. **AugGAN 대비 성능 갭**: 픽셀 어노테이션을 사용하는 AugGAN(55.31%)보다 낮음 → 어노테이션이 가능한 경우 상한선 존재
3. **매우 극단적 도메인 격차**: 야간 이미지처럼 정보 손실이 심한 경우 생성된 콘텐츠의 신뢰성 한계
4. **하이퍼파라미터 민감도**: $\alpha=10, \beta=1$은 휴리스틱 설정으로 도메인별 최적화 필요
5. **계산 비용**: 샴 네트워크 추가로 인한 학습 시간 증가

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 어노테이션 독립성이 가져오는 범용성

OP-GAN의 가장 중요한 일반화 기여는 **픽셀 단위 레이블 없이도 객체 인식 능력을 갖춘다**는 점입니다. 이는 다음 측면에서 일반화 성능을 향상시킵니다:

- **데이터 수집 제약 완화**: 의료영상, 위성영상 등 레이블 획득이 어려운 분야에서 직접 적용 가능
- **다중 도메인 확장성**: 동일한 프레임워크를 CamVid(자율주행), SYNTHIA(합성→실세계), Colonoscopy(의료)에 모두 적용하여 범용성 실증

### 3-2. 콘텐츠-도메인 특징 분리(Disentanglement)

$$E_A : P_A \rightarrow Z(c_A, d_A)$$

이 분리 구조는:
- **도메인 변화에 불변한 콘텐츠 표현** 학습을 강제
- 새로운 도메인 쌍에 대해서도 콘텐츠 특징이 안정적으로 작동할 가능성
- 전이학습(transfer learning) 시나리오에서 더 강건한 표현 제공

### 3-3. 자기지도 프록시 태스크의 일반화 효과

패치 위치 기반 자기지도 신호는 **이미지 데이터 자체에서 자동으로 생성**되므로:

$$\mathcal{L}_{cc} = \frac{1}{M \times N} \sum_{x=1}^{M} \sum_{y=1}^{N} (\tilde{p}^A_{x,y} - \tilde{p}^B_{x,y})^2$$

- 어떤 도메인 쌍에도 적용 가능한 **도메인-불가지론적(domain-agnostic)** 학습 신호
- 콘텐츠 어텐션 맵이 건물, 나무, 폴립 등 다양한 객체 유형에 자동 적응

### 3-4. 세그멘테이션 네트워크 범용성

PSPNet과 U-Net 두 가지 세그멘테이션 백본에서 모두 성능 향상을 보여 **특정 다운스트림 모델에 종속되지 않음**을 시사합니다.

---

## 4. 미래 연구에 대한 영향과 고려 사항

### 4-1. 앞으로의 연구에 미치는 영향

**① 자기지도 학습 + GAN 결합 패러다임 확립**

어노테이션 없이 GAN의 콘텐츠 보존 능력을 강화하는 방법론적 틀을 제시하며, 이후 연구들이 다양한 프록시 태스크(대조학습, 마스크 예측 등)를 GAN에 결합하는 방향을 촉진합니다.

**② 의료 영상 도메인 적응의 실용화**

픽셀 어노테이션이 극도로 제한적인 의료 영상 분야에서 CycleGAN 계열의 실용적 적용 가능성을 열었습니다.

**③ 특징 분리(Disentanglement) 연구 방향 제시**

콘텐츠-도메인 특징을 명시적으로 분리하는 구조는 이후 변분 오토인코더(VAE)나 트랜스포머 기반 I2I 변환 연구에도 영향을 미칩니다.

---

### 4-2. 2020년 이후 관련 최신 연구 비교 분석

> **⚠️ 중요 안내**: 아래 연구들은 제가 학습한 지식 범위 내의 정보이며, 논문에서 직접 인용된 것이 아닙니다. 세부 수치는 해당 논문 원문에서 반드시 확인하시기 바랍니다.

#### (1) CUT (Contrastive Unpaired Translation, Park et al., ECCV 2020)

**논문**: *Contrastive Learning for Unpaired Image-to-Image Translation* (Park, Efros, Zhang, Zhu, ECCV 2020)

| 비교 항목 | OP-GAN | CUT |
|---------|--------|-----|
| 콘텐츠 보존 방식 | 패치 위치 기반 자기지도 | Patch-NCE (대조학습) |
| 어노테이션 필요 | 불필요 | 불필요 |
| 단방향/양방향 | 양방향 | 단방향 가능 |
| 핵심 손실 | $\mathcal{L}\_{cc} + \mathcal{L}_{dc}$ | Patch-NCE Loss |

CUT은 **대조학습(Contrastive Learning)**을 활용하여 동일 위치의 소스-변환 패치를 긍정 쌍으로, 다른 위치를 부정 쌍으로 학습하는 방식으로 OP-GAN의 $\mathcal{L}\_{cc}$와 개념적으로 유사하나, InfoNCE 손실을 통해 더 풍부한 음성 샘플 활용이 가능합니다. OP-GAN은 명시적인 도메인 분류 태스크( $\mathcal{L}_{dc}$ )를 추가로 사용한다는 점에서 차별됩니다.

#### (2) DAFormer (Hoyer et al., CVPR 2022)

**논문**: *DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation* (Hoyer, Dai, Van Gool, CVPR 2022)

트랜스포머 기반 백본을 활용한 도메인 적응 세그멘테이션으로, OP-GAN과 같은 이미지 레벨 변환 접근법과 달리 **특징 레벨(feature-level) 정렬**에 초점을 맞춥니다. 두 접근법은 상호 보완적으로 결합 가능성이 있습니다.

#### (3) EGSDE (Zhao et al., NeurIPS 2022)

**논문**: *EGSDE: Unpaired Image-to-Image Translation via Energy-Guided Stochastic Differential Equations* (Zhao et al., NeurIPS 2022)

확산 모델(Diffusion Model)을 I2I 변환에 적용한 연구로, GAN 기반의 OP-GAN과 달리 **점진적 잡음 제거** 과정을 통해 더 안정적인 변환을 달성합니다. 그러나 추론 속도가 느리다는 한계가 있습니다.

#### (4) StyleGAN 계열과의 비교

StyleGAN2/3 및 이를 활용한 도메인 적응 연구들은 더 높은 이미지 품질을 달성하지만, OP-GAN이 목표로 하는 **객체 위치/형태 보존**에 특화된 명시적 메커니즘은 부재한 경우가 많습니다.

#### 연구 흐름 요약

```
CycleGAN (2017) → OP-GAN (2020): 자기지도 객체 보존
                → CUT (2020): 대조학습 기반 패치 일관성
                → DAFormer (2022): 트랜스포머 특징 정렬
                → Diffusion 기반 I2I (2022~): 고품질 생성
```

---

### 4-3. 앞으로 연구 시 고려할 점

#### ① 더 강력한 자기지도 신호 탐색

단순한 위치 기반 패치 매칭 대신, **마스크 자동 인코더(MAE)**, **DINO 특징**, **대조학습(InfoNCE)** 등 더 풍부한 자기지도 표현 학습 방법론과의 결합:

$$\mathcal{L}_{NCE} = -\sum_{l}\frac{1}{|S_l|}\sum_{s \in S_l}\log\frac{\exp(\hat{z}^l_s \cdot z^l_s / \tau)}{\sum_{n=0}^{N}\exp(\hat{z}^l_s \cdot z^l_n / \tau)}$$

이러한 손실 함수를 $\mathcal{L}_{cc}$ 대신 활용하면 더 세밀한 콘텐츠 보존이 가능할 수 있습니다.

#### ② 확산 모델(Diffusion Model)과의 결합

최근 DDPM, Score-based 모델 등 확산 모델이 GAN을 능가하는 이미지 생성 품질을 보이고 있습니다. OP-GAN의 자기지도 콘텐츠 보존 메커니즘을 확산 모델 기반 I2I 변환에 통합하는 연구가 유망합니다.

#### ③ 트랜스포머 기반 구조로의 확장

OP-GAN의 생성자/샴 네트워크를 **Vision Transformer(ViT)** 또는 **Swin Transformer** 기반으로 교체하면 전역적(global) 콘텐츠 일관성을 더 잘 포착할 수 있습니다.

#### ④ 멀티 도메인 확장

현재 양방향(A↔B) 변환에 국한되어 있으므로, 3개 이상의 도메인에 대한 **멀티 도메인 OP-GAN** 확장이 필요합니다.

#### ⑤ 패치 분할 전략의 적응적 설계

고정된 $3 \times 3$ 그리드 대신, 객체 감지 결과나 SAM(Segment Anything Model) 기반의 **시맨틱 패치 분할**을 적용하면 더 의미 있는 콘텐츠 단위로 일관성을 강제할 수 있습니다.

#### ⑥ 평가 지표의 다양화

현재 mIoU와 F1 Score만 사용하였으나, **FID(Fréchet Inception Distance)**, **LPIPS(Learned Perceptual Image Patch Similarity)** 등 이미지 품질 지표를 병행하여 시각적 사실성과 콘텐츠 보존 간의 트레이드오프를 더 정밀하게 분석해야 합니다.

---

## 참고 자료

**주요 참고 논문 (논문 내 인용)**:
1. Xie, X., Chen, J., Li, Y., Shen, L., Ma, K., Zheng, Y. — *Self-Supervised CycleGAN for Object-Preserving Image-to-Image Domain Adaptation* (본 논문, ECCV 2020)
2. Zhu, J., Park, T., Isola, P., Efros, A.A. — *Unpaired image-to-image translation using cycle-consistent adversarial networks*, ICCV 2017
3. Chen, T., Zhai, X., Ritter, M., Lucic, M., Houlsby, N. — *Self-supervised GANs via auxiliary rotation loss*, CVPR 2019
4. Huang, S., Lin, C., et al. — *AugGAN: Cross domain adaptation with GAN-based data augmentation*, ECCV 2018
5. Zhang, Z., Yang, L., Zheng, Y. — *Translating and segmenting multimodal medical volumes with cycle- and shape-consistency GAN*, CVPR 2018
6. Liu, M.Y., Breuel, T., Kautz, J. — *Unsupervised image-to-image translation networks*, NeurIPS 2017
7. Lee, H.Y., et al. — *Diverse image-to-image translation via disentangled representations (DRIT)*, ECCV 2018
8. Zhao, H., et al. — *Pyramid scene parsing network (PSPNet)*, CVPR 2017

**비교 분석 참고 논문 (2020년 이후)**:
- Park, T., Efros, A.A., Zhang, R., Zhu, J.Y. — *Contrastive Learning for Unpaired Image-to-Image Translation (CUT)*, ECCV 2020
- Hoyer, L., Dai, D., Van Gool, L. — *DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation*, CVPR 2022
