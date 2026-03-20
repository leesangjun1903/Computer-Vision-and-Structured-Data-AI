# DiffiT: Diffusion Vision Transformers for Image Generation

**저자:** Ali Hatamizadeh, Jiaming Song, Guilin Liu, Jan Kautz, Arash Vahdat (NVIDIA)
**발표:** ECCV 2024 (arXiv: 2312.02139, 2023년 12월)

---

## 1. 핵심 주장 및 주요 기여 요약

DiffiT는 ViT(Vision Transformer)를 확산 기반 생성 학습에 적용하여, 디노이징 과정의 세밀한 제어를 가능하게 하는 **Time-dependent Multihead Self Attention (TMSA)** 메커니즘을 도입한 모델이다. DiffiT는 높은 충실도의 이미지를 생성하면서도 현저히 우수한 파라미터 효율성을 보인다.

### 핵심 기여 (Contributions)

1. **TMSA 메커니즘 도입:** 시간 의존적 멀티헤드 셀프 어텐션(TMSA)은 디노이징 과정 중 공간적·시간적 의존성 및 그 상호작용에 대한 세밀한 제어를 가능하게 한다. TMSA는 key, query, value 가중치를 디노이징의 각 시간 단계별로 적응시켜, 네트워크가 서로 다른 단계에서 어텐션 메커니즘을 동적으로 변경할 수 있게 한다.

2. **Image Space 및 Latent Space 모델 제안:** 논문은 두 가지 아키텍처를 제안한다: U자형 인코더-디코더 구조의 Image Space 아키텍처와, VAE를 활용한 Latent Space 아키텍처이다.

3. **SOTA 성능 달성:** Latent DiffiT는 ImageNet-256 데이터셋에서 FID 점수 1.73의 새로운 SOTA를 달성했다. 이는 MDT와 DiT 같은 다른 Transformer 기반 확산 모델보다 각각 19.85%, 16.88% 적은 파라미터로 달성된 결과이다.

---

## 2. 상세 분석: 문제, 방법, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

확산 모델의 이미지 생성은 반복적 디노이징을 수행하는 신경망에 의존하지만, 디노이징 네트워크 아키텍처의 역할은 충분히 연구되지 않았으며, 대부분의 연구가 합성곱 잔차 U-Net에 의존하고 있었다. 특히 셀프 어텐션 모듈에서 시간 의존성(time-dependence)을 포착하는 데 있어 더 나은 제어 방법이 필요했다.

핵심 문제는 다음과 같다:
- 확산 모델의 디노이징 과정은 시간 단계($t$)마다 서로 다른 수준의 노이즈를 처리해야 하지만, 기존 어텐션 메커니즘은 이러한 시간적 변화를 충분히 반영하지 못함
- U-Net 기반 구조의 확장성(scalability) 한계
- Transformer 기반 확산 모델의 파라미터 효율성 문제

### 2.2 제안하는 방법 (수식 포함)

#### (a) 확산 모델의 기본 프레임워크 (DDPM)

확산 모델의 **Forward Process**는 데이터 $\mathbf{x}_0$에 점진적으로 가우시안 노이즈를 추가한다:

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$$

여기서 $\bar{\alpha}\_t = \prod_{s=1}^{t}\alpha_s$, $\alpha_t = 1 - \beta_t$이며, $\beta_t$는 노이즈 스케줄이다.

**Reverse Process**에서는 신경망 $\epsilon_\theta$가 노이즈를 예측한다:

$$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \sigma_t^2\mathbf{I})$$

학습 목적 함수(Training Objective):

$$\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]$$

#### (b) Time-dependent Multihead Self Attention (TMSA)

DiffiT의 핵심 혁신인 TMSA에서, 시간 의존적 queries $\mathbf{q}$, keys $\mathbf{k}$, values $\mathbf{v}$는 공간 토큰 임베딩과 시간 토큰 임베딩을 공유 공간으로 선형 투영하여 계산된다. 따라서 queries, keys, values는 모두 시간과 공간 토큰의 선형 함수가 되어, 서로 다른 시간 단계에서 어텐션 메커니즘의 행동을 적응적으로 수정할 수 있다.

구체적으로, 공간 임베딩 $\mathbf{x}_s$와 시간 임베딩 $\mathbf{x}_t$가 주어졌을 때:

$$\mathbf{q} = W_q^{(s)}\mathbf{x}_s + W_q^{(t)}\mathbf{x}_t$$

$$\mathbf{k} = W_k^{(s)}\mathbf{x}_s + W_k^{(t)}\mathbf{x}_t$$

$$\mathbf{v} = W_v^{(s)}\mathbf{x}_s + W_v^{(t)}\mathbf{x}_t$$

여기서 $W_q^{(s)}, W_q^{(t)}, W_k^{(s)}, W_k^{(t)}, W_v^{(s)}, W_v^{(t)}$는 각각 학습 가능한 투영 행렬이다.

어텐션 출력은 다음과 같이 계산된다:

$$\text{TMSA}(\mathbf{q}, \mathbf{k}, \mathbf{v}) = \text{Softmax}\left(\frac{\mathbf{q}\mathbf{k}^\top}{\sqrt{d}} + \mathbf{B}\right)\mathbf{v}$$

여기서 $d$는 keys $\mathbf{K}$에 대한 스케일링 팩터이고, $\mathbf{B}$는 각 어텐션 헤드에 걸쳐 정보를 인코딩할 수 있는 상대적 위치 바이어스(relative position bias)이다.

이 설계의 핵심 장점: 시간 임베딩이 Q, K, V 모두에 직접 통합되므로, 각 디노이징 단계에서 어텐션 패턴이 자연스럽게 적응된다.

#### (c) Window-based TMSA

TMSA를 window 기반 방식으로 확장하여 지역 간 교차 통신 없이도 동작하게 했다. 이 설계는 토큰 시퀀스 길이를 줄여 셀프 어텐션의 계산 비용을 감소시키면서도 놀랍도록 효과적이다.

Window-based attention의 계산 복잡도:

$$\mathcal{O}(W^2 \cdot N) \quad \text{vs. Global:} \quad \mathcal{O}(N^2)$$

여기서 $W$는 윈도우 크기, $N$은 전체 토큰 수이다.

### 2.3 모델 구조

DiffiT는 두 가지 아키텍처 변형을 제안한다:

#### (a) Image Space DiffiT

Image Space DiffiT 아키텍처는 대칭적 U자형 인코더-디코더 구조로, 수축 경로와 확장 경로가 skip connection으로 연결된다. 각 해상도는 L개의 연속된 DiffiT 블록으로 구성되며, 각 경로의 시작에 합성곱 레이어를 사용하여 특징 맵 수를 맞추고, 해상도 간 전환을 위해 합성곱 업샘플링/다운샘플링 레이어를 사용한다.

Image Space DiffiT에서는 DiffiT Transformer 블록과 합성곱 레이어를 잔차 연결(residual connection)로 결합한 DiffiT ResBlock이 정의된다.

구조 다이어그램:

```
Input → Conv↓ → [DiffiT Block × L] → Conv↓ → [DiffiT Block × L] → ...
                     ↓ skip                        ↓ skip
... → [DiffiT Block × L] → Conv↑ → [DiffiT Block × L] → Conv↑ → Output
```

#### (b) Latent Space DiffiT

Latent Space DiffiT에서는 이미지가 먼저 VAE(Variational Autoencoder)로 인코딩되고, 생성된 특징 맵이 비중첩 패치로 변환되어 새로운 임베딩 공간으로 투영된다. 그 후 Vision Transformer가 latent space에서 디노이징 네트워크로 사용되며, 업샘플링/다운샘플링 레이어 없이 동작한다. 최종 레이어는 출력 디코딩을 위한 단순 선형 레이어이다.

### 2.4 성능 향상

| 모델 | 데이터셋 | 해상도 | FID-50K↓ | 파라미터 수 |
|------|---------|--------|----------|------------|
| **DiffiT (Latent)** | ImageNet | 256×256 | **1.73** | ~542M |
| DiT-XL/2 | ImageNet | 256×256 | 2.27 | ~675M |
| MDT-XL/2 | ImageNet | 256×256 | 1.79 | ~676M |
| U-ViT-H/2 | ImageNet | 256×256 | 2.29 | ~501M |
| ADM-U | ImageNet | 256×256 | 3.94 | ~608M |

기존 DDPM++ 모델에 TMSA를 적용했을 때, VE와 VP 설정 각각에서 FID 점수가 0.28, 0.25만큼 감소하여, TMSA가 다양한 샘플링 단계에 동적으로 적응하고 시간 정보를 포착하는 효과를 입증했다.

### 2.5 한계점

TMSA 메커니즘의 계산 복잡도가 DiffiT의 초고해상도 또는 실시간 응용에 대한 확장성을 제한할 수 있다.

또한, 논문은 분포 이동(distribution shift)에 대한 모델의 강건성이나 테스트된 데이터셋을 넘어서 다양한 실세계 이미지 도메인으로의 일반화 능력을 다루지 않았다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 TMSA의 일반화 관점

TMSA는 시간 의존적 어텐션 메커니즘으로, 다음과 같은 일반화 성능 향상 가능성을 내포한다:

**(a) 다른 디노이징 네트워크로의 이식성 (Plug-in 특성)**

TMSA 레이어의 범용 효과는 기존 DDPM++ 모델의 원래 셀프 어텐션을 TMSA로 교체하는 실험으로 검증되었다. 원래 하이퍼파라미터를 변경하지 않고도 VE와 VP 설정 모두에서 FID가 개선되었다. 이 결과는 TMSA가 다양한 샘플링 단계에 동적으로 적응하여 시간 정보를 포착하는 효과를 입증한다.

이는 TMSA가 DiffiT 고유의 구조에 국한되지 않고, **범용적 모듈(plug-and-play)**로 활용될 수 있음을 시사한다.

$$\text{FID}_{\text{TMSA}} < \text{FID}_{\text{vanilla SA}} \quad \forall \text{ tested settings}$$

**(b) 다양한 해상도 및 태스크로의 확장**

DiffiT는 다양한 클래스 조건부(class-conditional) 및 비조건부(unconditional) 합성 작업에서 서로 다른 해상도에 걸쳐 SOTA 성능을 보였다. DiffiT는 FFHQ-64와 CIFAR10 데이터셋에서도 image space 생성 작업에서 SOTA 성능을 달성했다.

**(c) 어텐션 맵의 점진적 특화**

DiffiT 모델의 어텐션 맵은 샘플링 궤적 동안 세부적인 주요 특징들을 향한 점진적 지역화(progressive localization)를 보여주어, TMSA를 사용한 모델이 더 나은 이미지 생성 품질을 갖는다.

이는 모델이 디노이징 초기 단계에서는 전역적(global) 구조에, 후기 단계에서는 지역적(local) 디테일에 자연스럽게 집중함을 의미하며, 일반화의 근거가 된다.

### 3.2 일반화의 잠재적 방향

향후 연구에서는 DiffiT 내부 표현의 해석 가능성, 대안적 생성 모델 대비 샘플 효율성 및 학습 안정성을 탐구할 수 있다. 또한 이러한 강력한 이미지 합성 능력의 사회적 영향을 조사하는 것도 가치 있을 것이다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구적 영향

DiffiT는 확산 모델과 Vision Transformer의 강점을 결합하여 개선된 파라미터 효율성으로 SOTA 이미지 생성 성능을 달성하는 새로운 모델이다. 세밀한 디노이징 제어 및 TMSA 메커니즘은 서로 다른 머신러닝 접근법의 통합이 생성 모델링의 한계를 넓힐 잠재력을 보여준다.

구체적 영향:

1. **아키텍처 패러다임 전환**: U-Net → Transformer 기반 확산 모델로의 전환을 가속화
2. **시간 조건화 방법론의 발전**: 단순 임베딩 덧셈 → 어텐션 레벨의 시간 통합
3. **효율성 중시 설계 철학**: 파라미터 수 감소와 성능 향상의 동시 달성 가능성 제시

### 4.2 향후 연구 시 고려사항

| 고려사항 | 상세 내용 |
|---------|----------|
| **계산 효율성** | TMSA의 추가 투영 행렬($W^{(s)}, W^{(t)}$)로 인한 오버헤드 최소화 |
| **초고해상도 확장** | 1024×1024 이상에서의 Window TMSA 최적화 |
| **분포 이동 강건성** | 훈련 데이터 외 도메인에서의 성능 검증 필요 |
| **다중 모달 확장** | Text-to-Image, Video 생성으로의 TMSA 적용 |
| **해석 가능성** | 시간 단계별 어텐션 패턴의 의미론적 분석 |
| **학습 안정성** | 대규모 학습 시 수렴 특성 비교 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 주요 Transformer 기반 확산 모델 비교

| 모델 | 연도 | 아키텍처 | 시간 조건화 방식 | ImageNet-256 FID↓ | 핵심 특징 |
|------|------|---------|----------------|-------------------|----------|
| **DDPM** (Ho et al.) | 2020 | U-Net | Sinusoidal embedding + Residual | - | 확산 모델 기초 확립 |
| **ADM** (Dhariwal & Nichol) | 2021 | U-Net + Attention | Class. Guidance | 3.94 | Diffusion > GANs 입증 |
| **U-ViT** (Bao et al.) | 2022 | ViT + U-Net skip | Token concatenation | 2.29 | ViT backbone 첫 적용 |
| **DiT** (Peebles & Xie) | 2022 | Pure ViT | adaLN-Zero | 2.27 | Transformer 확장성 입증 |
| **MDT** (Gao et al.) | 2023 | ViT + Masking | adaLN | 1.79 (v1) / 1.58 (v2) | Masked latent modeling |
| **DiffiT** (Hatamizadeh et al.) | 2023 | ViT (Hybrid U-shape) | **TMSA** | **1.73** | 시간 의존적 어텐션 |
| **SiT** | 2024 | ViT | Interpolant transport | ~2.06 | Flow-based 대안 |
| **DyDiT** | 2025 (ICLR) | Dynamic ViT | Timestep-wise dynamic width | 효율성 중심 | 동적 아키텍처 |

### 5.2 주요 차이점 분석

#### DiT vs DiffiT

DiT-XL/2는 ImageNet 256×256에서 FID 2.27을 달성했으며, U-Net의 inductive bias(지역 합성곱 구조)가 고품질 이미지 생성에 필수적이지 않음을 보여주었다.

그러나 DiT는 adaLN-Zero를 통해 시간/클래스 정보를 주입하므로, 어텐션 자체는 시간에 독립적이다:

$$\text{DiT: } \mathbf{q} = W_q \cdot \text{adaLN}(\mathbf{x}_s, t), \quad \text{DiffiT: } \mathbf{q} = W_q^{(s)}\mathbf{x}_s + W_q^{(t)}\mathbf{x}_t$$

DiffiT의 TMSA는 어텐션 메커니즘 자체에 시간 정보를 통합하여 더 세밀한 제어를 가능하게 한다.

#### MDT vs DiffiT

MDT는 확산 확률 모델이 이미지 내 객체 부분 간의 관계를 학습하는 문맥적 추론 능력이 부족하여 학습 속도가 느리다는 문제를 해결하기 위해 마스크 잠재 모델링 방식을 도입했다. MDTv2는 ImageNet에서 FID 1.58의 SOTA를 달성하며, 이전 SOTA인 DiT보다 10배 이상 빠른 학습 속도를 보였다.

DiffiT(FID 1.73)은 MDTv2(FID 1.58)보다 FID에서 약간 뒤지지만, 파라미터 효율성에서 우위를 보인다.

#### U-ViT vs DiffiT

U-ViT는 CNN 잔차 블록을 Vision Transformer 블록으로 대체하여 U-Net과 Transformer 아키텍처를 연결한다. DiffiT는 이를 넘어서 TMSA를 통한 시간 의존적 어텐션을 도입하여 더 높은 성능을 달성했다.

#### 최근 동향 — Dynamic DiT (2025)

DyDiT(Dynamic Diffusion Transformer)는 timestep-wise 동적 폭(width)과 spatial-wise 동적 토큰을 통해 모델 아키텍처 및 토큰 중복성 관점에서 효율성을 개선한다. 이는 DiffiT의 "시간 단계별 적응" 철학을 아키텍처 수준으로 확장한 것으로 볼 수 있다.

### 5.3 연구 흐름 요약

```
DDPM (2020) → ADM (2021) → [U-ViT / DiT] (2022-2023) → [MDT / DiffiT] (2023) → [SiT / DyDiT / FiT] (2024-2025)
     ↓              ↓                  ↓                        ↓                         ↓
  U-Net 기반    Guidance 도입    Transformer 전환         시간 적응 + 효율성        동적/유연 아키텍처
```

---

## 6. 종합 결론

DiffiT는 확산 모델에서 **시간 의존적 어텐션**이라는 명확한 기여를 통해, Transformer 기반 확산 모델의 성능과 효율성을 동시에 끌어올렸다. 특히 TMSA의 plug-and-play 특성은 기존 다양한 확산 모델에 바로 적용 가능하여 넓은 일반화 가능성을 시사한다. 다만, 분포 이동에 대한 강건성, 초고해상도 확장성, 그리고 텍스트-이미지 등 다중 모달 태스크로의 확장에 관한 추가 연구가 필요하다.

---

## 참고 자료 및 출처

1. **Hatamizadeh et al. (2023).** "DiffiT: Diffusion Vision Transformers for Image Generation." arXiv:2312.02139, ECCV 2024. — [arxiv.org/abs/2312.02139](https://arxiv.org/abs/2312.02139)
2. **NVIDIA Research Publication Page** — [research.nvidia.com/publication/2024-09_diffit](https://research.nvidia.com/publication/2024-09_diffit-diffusion-vision-transformers-image-generation)
3. **NVlabs/DiffiT GitHub Repository** — [github.com/NVlabs/DiffiT](https://github.com/NVlabs/DiffiT)
4. **DiffiT arXiv HTML 전문** — [arxiv.org/html/2312.02139](https://arxiv.org/html/2312.02139)
5. **Springer ECCV 2024 Proceedings** — [link.springer.com/chapter/10.1007/978-3-031-73242-3_3](https://link.springer.com/chapter/10.1007/978-3-031-73242-3_3)
6. **Peebles & Xie (2023).** "Scalable Diffusion Models with Transformers (DiT)." ICCV 2023. — [arxiv.org/abs/2212.09748](https://arxiv.org/abs/2212.09748)
7. **Gao et al. (2023).** "Masked Diffusion Transformer is a Strong Image Synthesizer (MDT)." ICCV 2023. — [github.com/sail-sg/mdt](https://github.com/sail-sg/mdt)
8. **Bao et al. (2023).** "All are Worth Words: A ViT Backbone for Diffusion Models (U-ViT)." CVPR 2023.
9. **Dynamic Diffusion Transformer (DyDiT).** ICLR 2025. — [proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2025/file/a44a70acd5d0abc1a252ada9719dd06d-Paper-Conference.pdf)
10. **ICLR 2026 Blog: From U-Nets to DiTs** — [iclr-blogposts.github.io/2026/blog/2026/diffusion-architecture-evolution](https://iclr-blogposts.github.io/2026/blog/2026/diffusion-architecture-evolution/)
11. **AIModels.fyi 분석** — [aimodels.fyi/papers/arxiv/diffit-diffusion-vision-transformers-image-generation](https://www.aimodels.fyi/papers/arxiv/diffit-diffusion-vision-transformers-image-generation)
12. **DiffiT 비공식 구현** — [github.com/luca-zanchetta/DiffiT-Implementation](https://github.com/luca-zanchetta/DiffiT-Implementation)
13. **OpenReview (DiffiT)** — [openreview.net/forum?id=uAKk0I3xxm](https://openreview.net/forum?id=uAKk0I3xxm)
14. **Hugging Face Papers** — [huggingface.co/papers/2312.02139](https://huggingface.co/papers/2312.02139)

> **참고:** TMSA의 구체적 수식(특히 시간 임베딩의 Q/K/V 투영 방식)은 논문 원문과 공개 구현체를 기반으로 재구성한 것이며, 일부 세부 표기는 원문 표기와 미세하게 다를 수 있습니다. 정확한 수식은 arXiv 원문(2312.02139)의 Section 3을 참조해 주시기 바랍니다.
