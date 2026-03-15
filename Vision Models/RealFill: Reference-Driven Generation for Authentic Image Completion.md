# RealFill: Reference-Driven Generation for Authentic Image Completion

**논문 정보:** Tang, L., Ruiz, N., Chu, Q., Li, Y., Holynski, A., Jacobs, D.E., Hariharan, B., Pritch, Y., Wadhwa, N., Aberman, K., & Rubinstein, M. (2023). *ACM Transactions on Graphics (TOG)*, Vol. 43, SIGGRAPH 2024.

---

## 1. 핵심 주장 및 주요 기여 요약

기존의 outpainting/inpainting 모델들은 고품질의 그럴듯한(plausible) 이미지 콘텐츠를 생성할 수 있지만, 이렇게 생성된 콘텐츠는 실제 장면을 모르기 때문에 본질적으로 **비진정(inauthentic)**하다. RealFill은 이미지의 누락된 영역을 "있었어야 할 콘텐츠(what should have been there)"로 채우는 새로운 생성적 접근법이다.

### 주요 기여:

1. **"Authentic Image Completion" 문제 정의:** "Authentic Image Completion" — 즉, 누락 영역을 "있을 수 있었던 것(what could have been there)"이 아닌 "있었어야 할 것(what should have been there)"으로 채우는 과제를 공식적으로 정의하였다.

2. **개인화된 생성 모델:** 소수의 참조 이미지(1~5장)만으로 개인화되는 생성적 인페인팅 모델을 제안하였으며, 참조 이미지는 타겟 이미지와 정렬될 필요가 없고, 시점·조명·카메라 조리개·이미지 스타일이 크게 다를 수 있다.

3. **Correspondence-Based Seed Selection:** 생성적 추론의 확률적 특성을 고려하여, 생성된 콘텐츠와 참조 이미지 간의 실제 대응관계(true correspondences)를 활용하여 고품질 생성 결과를 자동으로 선택하는 기법을 제안하였다.

4. **새로운 벤치마크:** 10개의 인페인팅 및 23개의 아웃페인팅 예제와 해당 ground-truth를 포함하는 데이터셋을 수집하였으며, 다양한 이미지 유사도 메트릭에서 기존 방법을 큰 차이로 능가함을 보였다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 기하학 기반 파이프라인(correspondence matching, depth estimation, 3D transformation)은 장면의 구조를 정확히 추정할 수 없는 경우, 특히 복잡한 기하학 구조나 동적 객체가 포함된 경우 치명적 실패를 겪는다.

한편, 확산 모델(diffusion model) 기반 생성 모델은 인페인팅/아웃페인팅에서 강력한 성능을 보이지만, 텍스트 프롬프트에만 의존하여 실제 장면 구조와 세밀한 디테일을 복원하는 데 어려움을 겪는다.

**공식적 문제 정의:** 참조 이미지 집합 $\{I_{\text{ref}}^{(k)}\}\_{k=1}^{K}$ 와 누락 영역이 있는 타겟 이미지 $I_{\text{tgt}}$가 주어졌을 때, 출력 이미지는 그럴듯하고 사실적일 뿐만 아니라 참조 이미지에 충실하여 실제 장면에 존재했던 콘텐츠와 세부 사항을 복원해야 한다. 본질적으로, "있을 수 있었던 것" 대신 "있었어야 할 것"을 생성하는 **authentic image completion**을 달성하고자 한다.

### 2.2 제안하는 방법

#### (A) 배경: 확산 모델(Diffusion Model)

확산 모델의 학습은 다음 **denoising loss**를 최소화하는 방식으로 이루어진다:

$$\mathcal{L}_{\text{DM}} = \mathbb{E}_{z, \epsilon \sim \mathcal{N}(0,1), t}\left[\|\epsilon - \epsilon_\theta(z_t, t, c)\|_2^2\right]$$

여기서:
- $z_t$: 시간 단계 $t$에서의 노이즈가 추가된 잠재 변수(noisy latent)
- $\epsilon$: 추가된 가우시안 노이즈
- $\epsilon_\theta$: 노이즈 예측 네트워크 (UNet 기반)
- $c$: 조건(conditioning) 정보 (텍스트 프롬프트 등)

#### (B) RealFill의 인페인팅 학습 목적함수

RealFill의 핵심 방법론은 Latent Diffusion Models (Stable Diffusion)의 아키텍처와 개념 위에 직접 구축된다. 인페인팅 모델의 경우, 조건은 마스크 $m$과 마스킹된 이미지 $I \odot (1-m)$을 포함하므로, 학습 목적함수는 다음과 같다:

$$\mathcal{L}_{\text{inpaint}} = \mathbb{E}_{z, \epsilon \sim \mathcal{N}(0,1), t}\left[\|\epsilon - \epsilon_\theta(z_t, t, c_{\text{text}}, z_{m}, m)\|_2^2\right]$$

여기서:
- $z_m$: 마스킹된 이미지의 잠재 인코딩
- $m$: 바이너리 마스크 (누락 영역 표시)

#### (C) 개인화(Personalization) — DreamBooth 방식 확장

RealFill의 핵심 기법인 소수의 이미지 세트에 대해 확산 모델을 미세 조정하여 특정 장면을 학습하는 방식은 DreamBooth의 subject-driven generation 방법에서 직접 영감을 받았다.

RealFill은 참조 이미지 $\{I_{\text{ref}}^{(k)}\}$와 타겟 이미지 $I_{\text{tgt}}$를 **모두** 사용하여 사전 학습된 인페인팅 확산 모델을 미세 조정한다:

$$\mathcal{L}_{\text{RealFill}} = \mathbb{E}_{I \in \{I_{\text{ref}}, I_{\text{tgt}}\}}\left[\mathbb{E}_{z, \epsilon, t}\left[\|\epsilon - \epsilon_{\theta'}(z_t, t, c, z_{m}, m)\|_2^2\right]\right]$$

- 학습 시 참조 이미지와 타겟 이미지 모두에 **무작위 마스킹(random masking)**을 적용
- 이 미세 조정 과정은 적응된 모델이 좋은 이미지 prior를 유지할 뿐만 아니라 입력 이미지의 콘텐츠, 조명, 스타일을 학습하도록 설계되었다.

#### (D) LoRA를 활용한 효율적 미세 조정

이 접근법은 Stable Diffusion v2 인페인팅 모델 위에 구축되며, 메모리 효율적 개인화를 위해 Low-Rank Adaptation (LoRA)을 통합한다.

LoRA의 핵심 수식은 다음과 같다:

$$W' = W + \Delta W = W + BA$$

여기서:
- $W \in \mathbb{R}^{d \times d}$: 원래의 사전 학습된 가중치 행렬 (동결)
- $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$: 학습 가능한 저차원 행렬
- $r \ll d$: LoRA 랭크 (표현력과 효율성의 균형)

LoRA는 사전 학습된 모델 가중치를 동결하고 각 Transformer 블록에 학습 가능한 랭크-분해 행렬(rank-decomposition matrices)을 삽입하여, 학습 가능 파라미터 수와 GPU 메모리 요구량을 크게 줄인다.

### 2.3 모델 구조

RealFill은 주어진 장면에 대해, 먼저 사전 학습된 인페인팅 확산 모델을 참조 이미지와 타겟 이미지에 대해 미세 조정하여 개인화된 생성 모델을 생성한다.

**전체 파이프라인 (2단계):**

| 단계 | 설명 |
|------|------|
| **Training Phase** | 참조+타겟 이미지에 random masking → 사전학습 Stable Diffusion v2 Inpainting 모델을 LoRA로 fine-tune |
| **Inference Phase** | 미세 조정된 모델로 타겟 이미지의 누락 영역 완성 → Correspondence-Based Seed Selection으로 최종 결과 선택 |

추론 시 DDPM 샘플러를 200 스텝, 가이던스 가중치 1.0(즉, classifier-free guidance 없이)으로 사용한다.

**Correspondence-Based Seed Selection:**

여러 시드로 다수의 후보 $\{\hat{I}^{(s)}\}_{s=1}^{S}$를 생성한 뒤, 참조 이미지와의 특징점 대응(feature correspondence) 수를 기준으로 최적의 결과를 선택한다:

$$s^* = \arg\max_{s} \sum_{k=1}^{K} \text{NumCorr}(\hat{I}^{(s)}, I_{\text{ref}}^{(k)})$$

### 2.4 성능 향상

시스템은 정량적·정성적으로 기존 방법을 크게 능가하는 사실적인 완성 결과를 생성하며, 사용자 연구에서 87.2%의 충실도 선호도를 달성하였다.

**비교 대상 및 결과:**

| 방법 | 문제점 |
|------|--------|
| **Paint-by-Example** | CLIP 임베딩에 의존하여 고수준 의미 정보만 캡처하므로 복잡한 장면이나 객체 디테일 복원에 한계가 있다. |
| **TransFill** | 좋은 이미지 prior의 부재와 기하학 기반 파이프라인의 한계로 인해 출력 품질이 낮으며, 특히 평면을 넘어서는 복잡한 깊이 변화를 가진 장면 구조에서 호모그래피 변환이 근사하기 어렵다. |
| **Photoshop Generative Fill** | 그럴듯한 결과를 생성하지만, 프롬프트의 표현력 한계로 인해 참조 이미지와 일치하지 않는다. |

### 2.5 한계점

이 방법은 계산 비용이 높고(장면당 약 1시간의 미세 조정 필요), 참조 이미지가 부족할 때 극단적인 3D 시점 변화에서 어려움을 겪으며, 기본 확산 모델의 약점인 텍스트나 사람 얼굴 같은 세밀한 디테일 생성에서의 한계를 물려받는다.

또한, 기존 기준선보다 우수하지만 ground truth와 직접 비교하면 여전히 아티팩트가 나타날 수 있다.

추가적으로 RealFill은 같은 장면에 대해 3~5장의 이미지가 필요하며, 미세 조정된 모델은 학습된 장면의 로컬 영역만 완성할 수 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화의 제약

RealFill의 가장 큰 일반화 제약은 **장면별 미세 조정(per-scene fine-tuning)** 패러다임에 있다. 모델은 각 새로운 장면에 대해 재학습이 필요하므로:

- **시간적 비용:** 장면당 약 1시간의 미세 조정이 필요하다.
- **장면 특정적:** 미세 조정된 모델은 학습된 장면의 로컬 영역만 완성할 수 있다.
- **참조 이미지 의존성:** 참조 이미지의 가용성과 관련성에 성능이 크게 좌우된다.

### 3.2 일반화 향상 방향

1. **제로샷(Zero-shot) 접근법:** MimicBrush와 같은 방법은 비디오 클립에서 두 프레임을 무작위로 샘플링하여 하나의 프레임 영역을 마스킹하고, 다른 프레임의 정보를 이용해 마스킹 영역을 복원하도록 학습하는 자기 지도 방식으로, 장면별 미세 조정 없이 이미지 간 의미론적 대응을 포착한다.

2. **기하학 정보 통합:** GeoComplete는 명시적 3D 구조적 가이던스를 통합하여 완성 영역의 기하학적 일관성을 강화하며, 투영된 포인트 클라우드에 대한 확산 과정 조건화와 참조 단서를 안내하는 target-aware masking을 핵심 아이디어로 제시한다.

3. **LoRA 최적화 전략:**
   - 랭크 $r$의 적응적 조정으로 표현력과 과적합 방지 균형
   - Prior-preservation loss를 통한 일반화 능력 유지:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{RealFill}} + \lambda \cdot \mathcal{L}_{\text{prior}}$$

4. **멀티뷰 합성 활용:** FaithFill처럼 단일 참조 이미지에서 다중 뷰를 생성하여 학습 데이터를 보강하는 방식은 참조 이미지가 적을 때의 일반화 성능을 향상시킬 수 있다.

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 연구 영향

RealFill은 제어 가능한 이미지 생성에서 상당한 진전을 이루며, authentic image completion 문제를 공식적으로 정의하고 해결하였다. 이 연구는 비제약 실세계 변동을 처리하면서 다중 참조 이미지에 대해 확산 모델을 효과적으로 조건화하는 방법을 보여준다.

**파급 효과:**
- **3D 인페인팅으로의 확장:** NeRFiller 등 3D 캡처의 누락 부분을 생성적으로 완성하는 연구로 확장
- **영상(Video) 완성:** 시간적 일관성을 유지하면서 비디오 프레임의 누락 영역 완성
- **산업 응용:** 사진 편집, 건축 시각화, 의류 가상 피팅 등

### 4.2 향후 연구 시 고려 사항

| 고려 사항 | 세부 내용 |
|-----------|-----------|
| **계산 효율성** | 장면별 미세 조정의 계산 비용을 줄이기 위한 경량화 기법 (예: 더 작은 LoRA 랭크, 더 적은 학습 스텝) |
| **기하학적 일관성** | 명시적 3D 정보(깊이, 카메라 포즈)를 활용하여 대규모 시점 변화에서의 일관성 확보 |
| **제로샷 일반화** | 장면별 미세 조정 없이 범용적으로 작동하는 모델 개발 |
| **세밀한 디테일** | 텍스트, 얼굴 등 세밀한 디테일 복원 능력 향상 |
| **평가 메트릭** | "Authenticity"를 보다 정확하게 측정할 수 있는 새로운 평가 지표 개발 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 년도 | 핵심 접근법 | RealFill과의 차이점 |
|------|------|------------|-------------------|
| **LaMa** (Suvorov et al.) | 2021 | Large mask inpainting with Fourier convolutions | 참조 이미지 미사용, 텍스처 기반 단일 이미지 인페인팅 |
| **Stable Diffusion Inpainting** | 2022 | Text-conditioned latent diffusion inpainting | 텍스트 프롬프트에만 의존, 장면 특정 정보 없음 |
| **Paint-by-Example** (Yang et al.) | 2023 | CLIP 임베딩 기반 exemplar-guided inpainting | CLIP 임베딩은 단일 참조 이미지의 고수준 의미 정보만 캡처 가능 |
| **TransFill** (Zhou et al.) | 2021 | Multi-homography transformed fusion | 좋은 이미지 prior의 부재와 기하학 기반 파이프라인의 한계로 품질이 낮음 |
| **DreamBooth** (Ruiz et al.) | 2023 | Subject-driven text-to-image generation | RealFill의 개인화 기법의 개념적 기반을 제공하였으나, 인페인팅이 아닌 subject 생성에 초점 |
| **MimicBrush** (Chen et al.) | 2024 | Self-supervised dual U-Net for reference-based editing | 장면별 미세 조정 없이 제로샷으로 작동하나, RealFill과 달리 같은 장면 3-5장 불필요 |
| **FaithFill** (Abdalla et al.) | 2024 | 단일 참조 이미지에서 다중 뷰를 생성하여 활용하는 파이프라인 | 단일 참조로도 작동, 객체 수준 완성에 집중 |
| **GeoComplete** (Lin et al.) | 2025 | 명시적 3D 구조적 가이던스를 통합하여 기하학적 일관성을 강화하는 프레임워크 | RealFill의 무작위 마스킹 대신, 정보성 있는 영역을 선택적으로 마스킹하여 확산 모델을 의미 있는 단서로 안내하며, PSNR에서 SOTA 대비 17.1% 향상 |

### 진화 흐름 요약

```
단일 이미지 인페인팅 (LaMa, 2021)
    ↓
텍스트 조건 확산 인페인팅 (Stable Diffusion Inpainting, 2022)
    ↓
참조 기반 인페인팅 (Paint-by-Example, 2023)
    ↓
★ 장면 개인화 기반 Authentic 완성 (RealFill, 2023)
    ↓
단일 참조 + 다중뷰 생성 (FaithFill, 2024)
    ↓
제로샷 참조 기반 편집 (MimicBrush, 2024)
    ↓
기하학 인식 참조 기반 완성 (GeoComplete, 2025)
```

---

## 참고자료 출처

1. **[arXiv] Tang et al.**, "RealFill: Reference-Driven Generation for Authentic Image Completion," arXiv:2309.16668 — https://arxiv.org/abs/2309.16668
2. **[Project Page]** RealFill Official Project Page — https://realfill.github.io/
3. **[ACM TOG]** RealFill, ACM Transactions on Graphics — https://dl.acm.org/doi/10.1145/3658237
4. **[alphaXiv]** RealFill Overview & Analysis — https://www.alphaxiv.org/overview/2309.16668
5. **[MarkTechPost]** "Researchers from Google and Cornell Propose RealFill" — https://www.marktechpost.com/2023/10/03/
6. **[arXiv] Lin et al.**, "GeoComplete: Geometry-Aware Diffusion for Reference-Driven Image Completion," arXiv:2510.03110 — https://arxiv.org/abs/2510.03110
7. **[arXiv] Abdalla et al.**, "FaithFill: Faithful Inpainting for Object Completion Using a Single Reference Image," arXiv:2406.07865 — https://arxiv.org/abs/2406.07865
8. **[NeurIPS 2024] Chen et al.**, "MimicBrush: Zero-shot Image Editing with Reference Imitation" — https://proceedings.neurips.cc/paper_files/paper/2024/file/98b2b307aa4aa323df2ba3a83460f25e-Paper-Conference.pdf
9. **[arXiv] Hu et al.**, "LoRA: Low-Rank Adaptation of Large Language Models," arXiv:2106.09685 — https://arxiv.org/abs/2106.09685
10. **[GitHub]** Unofficial RealFill Implementation — https://github.com/thuanz123/realfill
11. **[HuggingFace Blog]** "Using LoRA for Efficient Stable Diffusion Fine-Tuning" — https://huggingface.co/blog/lora
12. **[Semantic Scholar]** RealFill citation graph — https://www.semanticscholar.org/paper/4584dee8505ce8cdaa09d7c3f4b4ab6568b3e766
13. **[OpenReview / NeurIPS 2025]** GeoComplete paper — https://openreview.net/pdf?id=1EnpXg8s4v

> **참고:** 본 분석에서 제시된 수식 중, 논문에 직접 명시되지 않은 일부 수식(예: Correspondence-Based Seed Selection의 공식화, total loss 등)은 논문의 서술 내용을 기반으로 공식적으로 표현한 것입니다. 정확한 수식은 원논문 PDF를 직접 참조하시기 바랍니다.
