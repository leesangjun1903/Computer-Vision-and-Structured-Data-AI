
# REPA : Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think
**저자:** Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, Saining Xie
**학회:** ICLR 2025 (Oral) | arXiv: 2410.06940

---

## 1. 핵심 주장 및 주요 기여 요약

최근 연구들은 Diffusion 모델의 Denoising 과정이 내부적으로 의미 있는 (discriminative) 표현을 유도할 수 있음을 보였지만, 그 품질은 최신 자기지도학습(SSL) 방법론에 비해 여전히 뒤처지는 수준이었다. 저자들은 대규모 Diffusion 모델 학습의 주요 병목이 바로 이 **표현(representation)을 효과적으로 학습하는 것**에 있다고 주장한다.

이를 해결하기 위해 저자들은 **REPresentation Alignment (REPA)** 라는 간단한 정규화 기법을 제안한다. 이는 Denoising 네트워크 내 노이즈가 포함된 입력 Hidden State의 투영(projection)을 외부의 사전학습된 비전 인코더에서 얻은 깨끗한 이미지 표현과 정렬(align)하는 방식이다. 결과는 놀랍게도, 이 단순한 전략이 DiT, SiT 같은 인기 있는 Diffusion/Flow 기반 Transformer에서 **학습 효율과 생성 품질 모두를 크게 향상**시킨다.

### 🔑 핵심 기여 요약

| 항목 | 내용 |
|------|------|
| **문제 제기** | Diffusion 모델 내부 표현의 질적 낙후 → 학습 병목 |
| **제안 방법** | REPA 정규화 (외부 SSL 표현과의 정렬) |
| **적용 대상** | DiT, SiT 등 Diffusion/Flow Transformer |
| **성과** | 17.5× 이상의 학습 가속 + FID 1.42 달성 (SOTA) |
| **학회** | ICLR 2025 Oral |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

이 논문은 Diffusion 모델 학습의 핵심 난제가 고품질의 내부 표현 $\mathbf{h}$를 학습해야 한다는 점에서 비롯된다는 사실을 규명한다. 외부 표현 $\mathbf{y}_*$의 지원이 있을 때 생성적 Diffusion 모델의 학습 과정이 훨씬 쉽고 효과적으로 이루어짐을 실험적으로 보여준다.

저자들은 노이즈가 포함된 잠재 표현(latent representation)이 이미지의 의미론적(semantic) 측면을 충분히 반영하지 못한다는 점에 주목하고, DINO v2와 같은 강력한 인코더가 생성하는 표현에 내부 표현을 가깝게 만드는 정규화 항을 추가하여 이를 풍부하게 만들 것을 제안한다.

---

### 2-2. 제안하는 방법 (수식 포함)

REPA는 Denoising 네트워크 내 노이즈가 포함된 입력 Hidden State의 투영(projection)을, 외부의 사전학습된 시각 인코더로부터 얻은 깨끗한 이미지 표현과 정렬하는 방식이다.

**REPA 손실 함수 (Alignment Loss):**

Denoising 네트워크의 $n$-번째 레이어 Hidden State를 $H_t^{[n]}$, 학습 가능한 Projection Head를 $h_\phi$, 외부 사전학습 인코더의 $i$번째 패치 표현을 $z_i$라 하면:

$$\mathcal{L}_{\text{REPA}} = -\frac{1}{N}\sum_{i=1}^{N} \text{sim}\left(h_\phi(H_t^{[n],i}),\, z_i\right)$$

여기서 $\text{sim}(\cdot,\cdot)$은 코사인 유사도(cosine similarity)와 같은 유사도 함수이며, $N$은 이미지 내 패치의 수이다. 이 손실은 Diffusion 모델의 학습 목표 함수에 통합되며, 전체 손실 함수는 다음과 같이 표현된다:

$$\mathcal{L} = \mathcal{L}_{\text{velocity}} + \lambda \mathcal{L}_{\text{REPA}}$$

여기서 $\lambda$는 주요 Denoising 목표에 대한 표현 정렬의 기여도를 조절하는 하이퍼파라미터이다.

$h_\phi$는 학습 가능한 Projection Head이며, 유사도 함수로는 코사인 유사도(cosine similarity) 또는 NT-Xent 손실이 사용될 수 있다.

**Flow/SiT 모델의 경우 velocity prediction 손실:**

$$\mathcal{L}_{\text{velocity}} = \mathbb{E}_{t, x_0, \epsilon}\left[\|v_\theta(x_t, t) - (x_0 - \epsilon)\|^2\right]$$

이를 합산한 최종 목표:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{velocity}} + \lambda \mathcal{L}_{\text{REPA}}$$

실험적으로, NT-Xent(Temperature-scaled Cross Entropy)는 초기 학습 단계(50–100K iterations)에서 이점을 보이지만, 시간이 지남에 따라 차이가 줄어든다. 이 때문에 저자들은 최종적으로 코사인 유사도(cos. sim.)를 채택하였다.

---

### 2-3. 모델 구조

REPA는 Diffusion 및 Flow 기반 Transformer 모델의 학습을 정규화하는 기법으로, Denoising 네트워크의 내부 Hidden State를 사전학습된 시각 인코더의 표현과 정렬한다. 연구팀은 Diffusion 모델이 표현을 완전히 처음부터 학습하는 것에만 의존하는 대신, 대규모 이미지 데이터셋으로 학습된 다른 컴퓨터 비전 모델의 사전학습 표현을 도입하는 방식을 제안한다.

**핵심 구조 요소:**

| 구성 요소 | 설명 |
|-----------|------|
| **Backbone** | DiT-XL/2, SiT-XL/2 (Diffusion/Flow Transformer) |
| **외부 인코더** | DINOv2 (ViT-L/14), CLIP, I-JEPA 등 사전학습 SSL 모델 |
| **Projection Head** | 학습 가능한 MLP ($h_\phi$), noisy hidden state → 인코더 feature space로 투영 |
| **정렬 위치** | Transformer의 특정 레이어(early~middle layer, 기본 n=8번째 레이어) |
| **VAE** | SD-VAE 등 잠재 공간 인코더/디코더 (REPA-E에서는 End-to-End 튜닝 가능) |

요약하면, 이 정규화는 사전학습된 자기지도 시각 표현을 Diffusion Transformer에 간단하고 효과적으로 증류(distillation)하는 방식으로, 의미론적으로 풍부한 외부 표현을 Diffusion 모델이 활용할 수 있게 한다.

---

### 2-4. 성능 향상

SiT-XL/2 모델 기준 ImageNet 256×256 벤치마크에서, REPA는 400K iterations 미만에서 FID 7.9를 달성하는 반면, Vanilla 모델은 동등한 수준(FID 8.3)에 도달하기 위해 7M iterations를 필요로 한다 — 이는 **17.5배의 학습 가속**을 의미한다. 또한 최종 생성 품질 면에서도, SiT-XL/2 + REPA는 classifier-free guidance를 사용하여 ImageNet 256×256에서 **FID 1.42**라는 SOTA를 달성하며, Vanilla 모델(FID 2.06)을 크게 능가한다.

저자들은 DiT, SiT 등 여러 Diffusion 모델 아키텍처에 REPA를 적용하여 학습 과정을 최대 17.5배 가속하고 최종 생성 품질을 개선함을 보인다. 특히, **더 큰 모델일수록 정렬 과정으로부터 더 많은 이점을 얻어** 고품질 결과로 더 빠르게 수렴한다는 확장성(scalability)도 확인되었다.

---

### 2-5. 한계점

REPA는 잠재 공간(latent space)에서의 학습 가속에는 효과적이지만, **픽셀 공간 Diffusion Transformer(JiT: Just image Transformers)에는 실패할 수 있다**. JiT에 REPA를 적용하면 FID가 오히려 악화되고, 사전학습 의미론적 인코더의 표현 공간에서 밀집된 이미지 서브셋에 대한 다양성(diversity)이 붕괴된다. 이는 Denoising이 고차원 이미지 공간에서 이루어지는 반면, 의미론적 타겟은 강하게 압축되어 있어 직접적인 회귀가 단축 학습(shortcut objective)이 되기 때문이다.

또한, 표준 REPA에서는 VAE가 고정되어 있기 때문에, REPA 정렬의 최대치가 VAE의 피처에 의해 병목(bottleneck)된다는 한계도 존재한다. REPA 정렬 점수(CKNNA 기준)가 높을수록 생성 성능(낮은 FID)과 상관관계가 있다는 것도 확인된다.

학습 후반부로 갈수록 REPA 손실과 Denoising 손실 간의 그래디언트 방향이 점차 반대가 되어, 학생 모델이 학습하려는 세부 디테일을 오히려 지우는 현상이 나타날 수 있다. 즉, REPA의 정렬 목표와 Denoising 목표 사이의 목표 불일치(objective conflict) 문제가 학습 후반부에 발생한다.

---

## 3. 모델의 일반화 성능 향상 가능성

REPA는 다양한 데이터셋과 구성 설정에서 성능 향상을 효과적으로 유지함으로써, **여러 태스크에 걸친 일반화 능력**을 보인다.

일반적으로, 더 강력한 표현 인코더와 정렬할수록 생성 결과와 선형 탐사(linear probing) 성능이 모두 향상된다. 또한, **Diffusion Transformer 모델의 크기가 커질수록 REPA로 인한 수렴 가속 효과도 더 크게 나타난다.**

REPA의 확장 버전인 REPA-E에서 확인된 바에 따르면, REPA의 이점은 다양한 Diffusion 모델 크기(SiT-B/L/XL), VAE 아키텍처(SD-VAE, IN-VAE, VA-VAE), REPA 지각 인코더(DINOv2, CLIP, I-JEPA), 그리고 REPA 정렬 깊이에 걸쳐 일관되게 유지된다.

공식 GitHub 저장소에 따르면, REPA는 512×512 해상도의 ImageNet 학습과 MS-COCO 기반의 텍스트-이미지 생성에도 확장 적용이 지원되며, 이는 도메인 일반화 가능성을 보여준다.

**일반화 성능 향상 관련 핵심 요인 정리:**

| 요인 | 효과 |
|------|------|
| 더 강한 외부 인코더 사용 | 생성 품질 + 선형 분류 성능 동반 향상 |
| 더 큰 모델 크기 | REPA 효과 상대적으로 더 크게 나타남 |
| 다양한 VAE/인코더 조합 | 아키텍처 독립적 이점 유지 |
| 해상도 확장 (256→512) | 고해상도에서도 일반화 효과 유지 |
| 텍스트-이미지 생성 확장 | 멀티모달 태스크로 일반화 가능성 제시 |

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4-1. 연구에 미치는 영향

이 연구는 대규모 생성적 Diffusion 모델의 학습과 성능 향상에 있어 중요한 진전을 의미하며, REPA는 Diffusion/Flow 기반 Transformer 모델의 학습 효율과 생성 품질을 동시에 크게 향상시키는 단순하지만 효과적인 기법이다. 이들 모델의 내부 표현을 고품질 외부 시각 인코더와 정렬함으로써 대규모 Diffusion 모델 학습의 핵심 병목을 극복한다.

**파생 연구의 흐름:**

1. **REPA-E (2025):** REPA 손실을 VAE를 통해 역전파함으로써 잠재 공간도 적응적으로 개선하며, VAE와 Diffusion 모델을 처음부터 단일 단계로 공동 학습하는 것이 가능해졌다.

2. **PixelREPA (2026):** 픽셀 공간 Diffusion에서의 정보 비대칭 문제(고차원 이미지 공간 Denoising vs. 압축된 의미론적 타겟)를 해결하기 위해 PixelREPA를 제안, Masked Transformer Adapter로 정렬을 제약한다.

3. **REG (Representation Entanglement, 2025):** SiT-XL/2 + REG는 SiT-XL/2 대비 63배, SiT-XL/2 + REPA 대비 23배 빠른 수렴 가속을 달성하며 REPA의 한계를 추가로 극복한다.

4. **U-REPA (2025):** U-Net Hidden State를 ViT 인코더 피처와 정렬하는 U-REPA 프레임워크를 제안하여 다양한 아키텍처로의 일반화를 추구한다.

---

### 4-2. 향후 연구 시 고려할 점

**① 픽셀 공간 확장의 한계 극복**
REPA는 JiT와 같은 픽셀 공간 Diffusion Transformer에서 실패할 수 있으며, 학습이 진행될수록 FID가 악화되고 이미지 다양성이 붕괴되는 현상이 발생한다. 이 문제를 해결하기 위한 픽셀 공간 특화 정렬 기법 연구가 필요하다.

**② 학습 후반부의 목표 충돌(Gradient Conflict)**
학습 후반부로 갈수록 REPA 그래디언트가 Denoising 학습이 학습하려는 세부 디테일을 지우는 방향으로 작용할 수 있다. 따라서 **정렬 손실의 점진적 감소 전략(Early Stopping, 스케줄링)**을 연구할 필요가 있다.

**③ 외부 인코더 의존성 및 편향 문제**
REPA는 DINOv2와 같은 최신 자기지도 학습 비전 모델에 의존하는데, 이러한 인코더가 특정 데이터셋(예: ImageNet)에 편향될 경우 생성 모델의 다양성에 제약이 생길 수 있다. 따라서 더 다양하고 범용적인 인코더 또는 다중 인코더 앙상블 전략의 연구가 필요하다.

**④ 멀티모달 및 타 도메인 확장**
REPA의 핵심 아이디어인 "외부 표현 정렬"을 텍스트, 오디오, 비디오 생성 등 다양한 멀티모달 도메인으로 확장하는 연구가 유망하다. 이미 MS-COCO 기반 텍스트-이미지 생성에 적용되었음이 확인된 만큼, 확장 가능성이 높다.

**⑤ VAE 병목 문제 해결**
표준 REPA에서 VAE가 고정될 경우 달성 가능한 최대 정렬 수준이 VAE의 피처에 의해 병목되므로, REPA 손실을 VAE를 통해 역전파하여 잠재 공간을 적응시키는 연구 방향이 중요하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 발표 | 핵심 방법 | 주요 성과 | REPA와의 관계 |
|------|------|-----------|-----------|--------------|
| **DDPMs** (Ho et al., 2020) | NeurIPS 2020 | DDPM denoising | 이미지 생성 기반 확립 | REPA의 기반 |
| **DiT** (Peebles & Xie, 2023) | ICCV 2023 | Transformer 기반 Diffusion | Scalable 생성 | REPA 주요 적용 대상 |
| **SiT** (Ma et al., 2024) | 2024 | Flow Matching + ViT | Diffusion Transformer 개선 | REPA 주요 적용 대상 |
| **DINOv2** (Oquab et al., 2024) | TMLR 2024 | Self-supervised ViT | 범용 시각 표현 | REPA의 주요 외부 인코더 |
| **REPA** (Yu et al., 2025) | ICLR 2025 Oral | 표현 정렬 정규화 | 17.5× 가속, FID 1.42 | **본 논문** |
| **REPA-E** (2025) | 2025 | End-to-End VAE 튜닝 | VAE 병목 해결, FID 개선 | REPA 확장 |
| **REG** (2025) | 2025 | Representation Entanglement | 63× 가속 | REPA 개선 후속 연구 |
| **PixelREPA** (Shin et al., 2026) | 2026 | 픽셀 공간 정렬 | JiT에서의 REPA 실패 해결 | REPA 한계 극복 |

---

## 📚 참고 자료 및 출처

1. **arXiv 원문:** [arXiv:2410.06940](https://arxiv.org/abs/2410.06940) — "Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think", Sihyun Yu et al.
2. **ICLR 2025 공식 페이지:** [OpenReview](https://openreview.net/forum?id=DJSZGGZYVi) — ICLR 2025 Oral 발표 논문
3. **ICLR 2025 proceedings:** [proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d9e42b4d7163931f3689d6d6fbaa11d0-Abstract-Conference.html)
4. **공식 GitHub 저장소:** [github.com/sihyun-yu/REPA](https://github.com/sihyun-yu/REPA)
5. **논문 PDF:** [arxiv.org/pdf/2410.06940](https://arxiv.org/pdf/2410.06940)
6. **ResearchGate:** [researchgate.net - REPA 논문](https://www.researchgate.net/publication/384770224)
7. **AI Models FYI 분석:** [aimodels.fyi - REPA](https://www.aimodels.fyi/papers/arxiv/representation-alignment-generation-training-diffusion-transformers-is)
8. **The Moonlight Literature Review:** [themoonlight.io - REPA 리뷰](https://www.themoonlight.io/en/review/representation-alignment-for-generation-training-diffusion-transformers-is-easier-than-you-think)
9. **AlphaXiv 개요:** [alphaxiv.org - REPA](https://www.alphaxiv.org/overview/2410.06940)
10. **후속 연구 - PixelREPA:** [arXiv:2603.14366](https://arxiv.org/abs/2603.14366) — "Representation Alignment for Just Image Transformers is not Easier than You Think"
11. **후속 연구 - REG:** [arXiv:2507.01467](https://arxiv.org/abs/2507.01467) — "Representation Entanglement for Generation"
12. **후속 연구 - REPA-E:** [emergentmind.com - REPA-E](https://www.emergentmind.com/papers/2504.10483)
13. **후속 연구 - U-REPA:** [arXiv:2503.18414](https://arxiv.org/pdf/2503.18414) — "U-REPA: Aligning Diffusion U-Nets to ViTs"
14. **후속 연구 - REPA 한계 분석:** [arXiv:2505.16792](https://arxiv.org/pdf/2505.16792) — "REPA Works Until It Doesn't"
