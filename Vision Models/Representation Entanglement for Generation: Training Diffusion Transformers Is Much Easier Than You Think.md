
# REG: Representation Entanglement for Generation
### *Training Diffusion Transformers Is Much Easier Than You Think*
> **저자**: Ge Wu, Shen Zhang, Ruijing Shi, Shanghua Gao, Zhenyuan Chen, Lei Wang, Zhaowei Chen, Hongcheng Gao, Yao Tang, Jian Yang, Ming-Ming Cheng, Xiang Li†
> **기관**: Nankai University 외
> **arXiv**: [2507.01467](https://arxiv.org/abs/2507.01467) (2025년 7월 2일)

---

## 1. 핵심 주장 및 주요 기여 요약

### 🔑 핵심 주장

기존의 REPA 및 변형 방법들은 사전학습된 모델의 외부 시각적 표현을 diffusion 모델 학습에 활용하지만, 이 외부 정렬(external alignment)은 추론(inference) 전 과정에서 부재하기 때문에 판별적 표현(discriminative representation)의 잠재력을 충분히 활용하지 못한다는 문제를 지적합니다.

이에 대한 해결책으로, 저자들은 **Representation Entanglement for Generation (REG)**이라는 간단한 방법을 제안합니다. 이는 사전학습된 파운데이션 모델(foundation model)의 단일 고수준(high-level) 클래스 토큰을 저수준(low-level) 이미지 잠재 벡터(image latents)와 **엉킴(entangle)**시켜 디노이징에 활용합니다.

### 🏆 주요 기여 요약

| 기여 항목 | 내용 |
|---|---|
| **새로운 패러다임** | 순수 이미지 디노이징 → 이미지-클래스 동시 디노이징 패러다임으로 전환 |
| **구조적 통합** | 추론 시에도 클래스 토큰 유지 (REPA의 학습 전용 외부 정렬 한계 극복) |
| **효율성** | 단 1개의 추가 토큰으로 구현, 추가 추론 오버헤드 최소화 |
| **학습 가속** | SiT-XL/2 대비 63×, REPA 대비 23× 수렴 속도 향상 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

REG는 REPA와 같은 선행 방법들의 한계, 즉 학습 중 사전학습된 시각적 표현을 외부 정렬 방식으로 활용하지만 **추론 과정에서는 이 판별적 정보를 완전히 활용하지 못하는 문제**를 해결하고자 합니다.

구체적으로 다음 두 가지 문제를 해결합니다:

1. **Train-Inference Gap**: 학습 시에만 존재하는 외부 정렬 신호가 추론 시에는 완전히 사라짐
2. **판별적 표현의 미활용**: 파운데이션 모델의 고수준 의미 정보가 생성 과정에서 능동적으로 기여하지 못함

---

### 2.2 제안하는 방법 (수식 포함)

#### (1) 기본 프레임워크: Stochastic Interpolant (SiT 기반)

REG는 flow 및 diffusion 모델의 통합 관점인 연속 시간 스토캐스틱 인터폴런트(stochastic interpolant) 기반의 SiT 프레임워크를 활용합니다. 시간 $t \in [0, 1]$에서의 중간 상태 $x_t$는 데이터 $x^*$와 노이즈 $\epsilon \sim \mathcal{N}(0, I)$ 사이의 보간(interpolation)으로 정의됩니다:

$$x_t = \alpha_t x^* + \sigma_t \epsilon$$

여기서 $\alpha_t$와 $\sigma_t$는 스케줄 함수입니다.

#### (2) REG의 핵심 메커니즘: 엔탱글먼트 (Entanglement)

REG는 사전학습된 파운데이션 모델에서 추출한 **단일 고수준 클래스 토큰**을 저수준 이미지 잠재 벡터와 **동기화된 노이즈 주입(synchronized noise injection) 및 공간적 연결(spatial concatenation)**을 통해 엔탱글합니다.

이를 수식으로 표현하면:

$$\tilde{x}_t^{\text{REG}} = \text{Concat}\left[ x_t^{\text{image}},\ x_t^{\text{cls}} \right]$$

클래스 토큰에도 동일한 노이즈 스케줄이 적용됩니다:

$$x_t^{\text{cls}} = \alpha_t \cdot c_{\text{cls}} + \sigma_t \cdot \epsilon_{\text{cls}}$$

여기서 $c_{\text{cls}}$는 파운데이션 모델(예: DINOv2)에서 추출한 클래스 토큰입니다.

#### (3) 학습 목적함수 (Velocity Prediction Loss)

REG의 의미 재구성 능력은 두 가지 핵심 설계 요소에서 비롯됩니다: (1) 학습 중 클래스 토큰과 이미지 잠재 벡터의 **구조적 엔탱글먼트**, (2) SiT의 **속도 예측 손실(velocity prediction loss)을 두 요소 모두에 일관되게 적용**하는 것입니다.

전체 손실 함수는:

$$\mathcal{L}_{\text{REG}} = \mathcal{L}_{\mathbf{v}}^{\text{image}} + \beta \cdot \mathcal{L}_{\mathbf{v}}^{\text{cls}}$$

여기서:
- $\mathcal{L}_{\mathbf{v}}^{\text{image}}$: 이미지 잠재 벡터에 대한 속도 예측 손실
- $\mathcal{L}_{\mathbf{v}}^{\text{cls}}$: 클래스 토큰에 대한 속도 예측 손실
- $\beta$: 클래스 토큰 손실의 가중치 하이퍼파라미터

실험적으로 $\beta = 0.03$이 전반적인 평가 지표에서 최상의 성능을 달성합니다.

---

### 2.3 모델 구조

REG 프레임워크는 파운데이션 모델에서 도출된 의미 클래스 임베딩과 공간 시각 표현을 구조적으로 통합합니다. 이 아키텍처 설계는 디노이징 단계에서 **지역적 패턴 복원과 전체적 개념 표현을 동시에 정제**할 수 있게 하여, 전체 생성 과정에 걸쳐 지속되는 컨텍스트 인식 의미 유도(context-aware semantic steering)를 가능하게 합니다.

```
┌──────────────────────────────────────────────────────────────┐
│                    REG Architecture                          │
│                                                              │
│  Foundation Model (DINOv2) ──→ cls token c_cls              │
│                                      ↓                      │
│  Image x* ──→ VAE Encoder ──→ z (image latents)             │
│                                      ↓                      │
│  Noise Injection (synchronized): x_t^cls + x_t^image        │
│                                      ↓                      │
│  Spatial Concat: [x_t^image | x_t^cls] ──→ SiT/DiT         │
│                                      ↓                      │
│  Velocity Prediction: v̂(x_t, t) for both image & cls       │
│                                      ↓                      │
│  Inference: Concurrent reconstruction of image + semantics  │
└──────────────────────────────────────────────────────────────┘
```

디노이징 과정은 이미지 잠재 벡터와 해당하는 전역 의미(global semantics)를 동시에 재구성하며, 획득된 의미 지식이 이미지 생성 과정을 능동적으로 안내하고 향상시킵니다. 이 모든 과정이 단 하나의 토큰 추가를 통해 최소한의 계산 비용으로 이루어집니다.

**REPA vs REG 비교**:

| 항목 | REPA | REG |
|---|---|---|
| 외부 정렬 | 학습 시에만 존재 | 없음 (내부적 통합) |
| 추론 시 의미 정보 | 부재 | 클래스 토큰이 능동적으로 가이드 |
| 추가 파라미터 | alignment head | 1개의 추가 토큰 |
| 추론 FLOPs 증가 | 0% | <0.5% |

---

### 2.4 성능 향상

ImageNet 256×256에서 SiT-XL/2 + REG는 SiT-XL/2 대비 **63배**, SiT-XL/2 + REPA 대비 **23배** 빠른 수렴 속도를 보입니다. 더 인상적으로, SiT-L/2 + REG를 400K 이터레이션만 학습시킨 결과가 SiT-XL/2 + REPA를 4M 이터레이션(10배 더 오랜 학습) 훈련시킨 결과를 능가합니다.

구체적으로, 400K 이터레이션에서 SiT-XL/2 + REPA가 FID 5.9를 달성하는 반면, SiT-XL/2 + REG는 FID 3.4를 달성합니다. SiT-L/2 + REG (400K iter, FID 4.6)는 SiT-XL/2 + REPA (4M iter, FID 5.9)를 능가합니다.

**성능 비교 요약표** (ImageNet 256×256):

| 모델 | Iterations | FID ↓ |
|---|---|---|
| SiT-XL/2 (baseline) | 4M | ~20+ |
| SiT-XL/2 + REPA | 4M | 5.9 |
| SiT-L/2 + REG | 400K | **4.6** |
| SiT-XL/2 + REG | 400K | **3.4** |

REG는 모든 구성에서 REPA 대비 일관된 성능 향상을 보여주며, FID 감소 폭이 4.19~7.16 포인트에 달합니다. 이러한 성능 향상은 클래스 토큰의 직접 삽입으로 인해 모든 레이어에 글로벌 이산 가이던스(discrete global guidance)가 제공되기 때문입니다.

### 2.5 한계

논문에서 확인된 한계점은 다음과 같습니다:

1. **클래스 조건부 한계**: 현재 프레임워크는 ImageNet의 클래스 레이블 기반 실험 위주이며, 텍스트 조건부 생성(text-to-image)으로의 직접 확장은 추가 연구 필요
2. **파운데이션 모델 의존성**: 클래스 토큰 소스로 DINOv2-B와 같은 자기지도 학습 모델이 특히 효과적이며, 최적의 사전학습 모델 선택이 성능에 영향을 미침
3. **이론적 분석 부족**: 엔탱글먼트 메커니즘이 왜 효과적인지에 대한 심층적인 이론적 분석이 제한적

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

다양한 사전학습된 자기지도 인코더(self-supervised encoder)의 클래스 토큰을 명시적 표현 정렬 없이 SiT-B/2에 통합한 실험에서, **클래스 토큰 엔탱글먼트만으로도 모든 변형에서 일관되게 생성 품질이 향상**됨이 입증되었습니다. FID 개선 폭은 0.95~6.33 포인트에 달하며, 특히 DINOv2-B는 FID 19.18% 감소 및 IS 35.86% 향상을 달성합니다.

이 결과는 **명시적 정렬 없이도 클래스 토큰으로부터 고수준 의미 가이던스를 효과적으로 활용할 수 있음**을 보여주며, 클래스 토큰 기반 엔탱글먼트의 생성 모델링에 대한 범용성과 강건성을 부각시킵니다.

### 일반화 관점에서의 강점

클래스 토큰의 직접 삽입은 모든 레이어에 이산 글로벌 가이던스를 제공하며, 이는 선택된 특징만 목표 표현과 정렬되는 REPA의 간접 감독 메커니즘과 대조됩니다. 결과적으로 REG는 나머지 레이어가 REPA보다 풍부한 고주파 세부 정보를 포착할 수 있게 하여 관찰된 성능 향상에 기여합니다.

다양한 모델 크기(SiT-B/2, SiT-L/2, SiT-XL/2)에서 일관적인 성능 향상이 나타난다는 점은 **스케일 불변 일반화 가능성**을 시사합니다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 논문 | 연도 | 핵심 아이디어 | FID (ImageNet 256) | REG와의 차이 |
|---|---|---|---|---|
| **DiT** (Peebles & Xie) | 2022 | Transformer 기반 Diffusion | 2.27 (완전 학습) | REG의 backbone |
| **SiT** (Ma et al.) | 2024 | Stochastic Interpolant + Transformer | ~20+ (기준) | REG의 기반 프레임워크 |
| **REPA** (Yu et al.) | 2024 | 학습 중 외부 표현 정렬 | 5.9 (4M iter) | 추론 시 정렬 부재 |
| **REG** (Wu et al.) | **2025** | 학습+추론 모두 클래스 토큰 엔탱글 | **3.4 (400K iter)** | - |
| **REGLUE** | 2025 | 전역+지역 의미 통합 엔탱글 | REPA, REG 능가 | 패치 수준 의미까지 통합 |

REPA는 대규모 diffusion 모델 학습의 주요 병목이 표현 학습에 있다고 주장하며, 고품질 외부 시각 표현을 통합함으로써 학습이 용이해질 수 있다고 제안합니다. REG는 이를 한 단계 발전시켜 추론까지 의미 정보를 유지합니다.

REGLUE(REG의 후속 연구)는 VAE 이미지 잠재 벡터, 콤팩트한 지역(패치 수준) VFM 의미 정보, 전역 [CLS] 토큰을 단일 SiT 백본 내에서 공동 모델링하는 통합 잠재 diffusion 프레임워크를 도입합니다.

ImageNet 256×256에서 REGLUE는 SiT-B/2, SiT-XL/2 기준선뿐만 아니라 REPA, ReDi, REG를 능가하여 FID와 수렴 속도 모두를 일관되게 향상시킵니다.

---

## 5. 향후 연구에 미치는 영향 및 고려할 점

### 📌 연구에 미치는 영향

#### 1. 패러다임 전환
REG는 현재의 순수 이미지 디노이징 파이프라인 대신 **이미지-클래스 동시 디노이징 패러다임**을 처음으로 도입한 프레임워크로, 판별적 정보의 생성 활용 잠재력을 완전히 해방시킵니다. 이는 생성 모델 설계 방향에 근본적인 변화를 촉구합니다.

#### 2. 학습 효율성 패러다임 재정의
단 400K 이터레이션으로 4M 이터레이션의 기존 방법을 능가함으로써, **컴퓨팅 자원 제약 환경에서의 고품질 생성 모델 학습** 가능성을 크게 넓혔습니다.

#### 3. 후속 연구 촉발
방대한 실험을 통해 생성 충실도, 학습 수렴 가속화, 판별적 의미 학습에서 REG의 우수한 성능이 입증되어, 그 효과성과 확장성이 검증되었습니다. 이는 REGLUE와 같은 후속 연구를 활발히 촉발시키고 있습니다.

---

### 🔬 앞으로 연구 시 고려할 점

| 고려 항목 | 세부 내용 |
|---|---|
| **텍스트 조건부 확장** | 클래스 토큰 → 텍스트 임베딩(CLIP, T5)으로 확장 시 text-to-image 생성으로 일반화 가능성 탐색 |
| **최적 파운데이션 모델 선택** | DINOv2, CLIP, MAE 등 어떤 사전학습 모델의 클래스 토큰이 특정 도메인에 가장 효과적인지 체계적 분석 필요 |
| **멀티모달 확장** | 이미지-텍스트-오디오 등 멀티모달 토큰 엔탱글먼트로의 확장 가능성 |
| **비디오 생성** | 시간적 일관성을 위한 비디오 생성 모델에서의 의미 토큰 엔탱글먼트 적용 |
| **이론적 근거 강화** | 엔탱글먼트 메커니즘이 학습을 가속하는 이유에 대한 정보 이론적/수학적 분석 필요 |
| **$\beta$ 파라미터 최적화** | 다양한 데이터셋 및 모델 크기에 따른 $\beta$ 값의 적응적 조정 전략 연구 |
| **경량화 파운데이션 모델** | 추론 시 파운데이션 모델 없이 클래스 토큰을 자체 생성하는 방향 탐색 |

---

## 📚 참고 자료 (출처)

1. **[arXiv:2507.01467v1]** Ge Wu et al., "Representation Entanglement for Generation: Training Diffusion Transformers Is Much Easier Than You Think," arXiv, July 2025. https://arxiv.org/abs/2507.01467
2. **[arXiv:2507.01467 HTML]** Full paper HTML version: https://arxiv.org/html/2507.01467v1
3. **[TheMoonlight.io Literature Review]** "Representation Entanglement for Generation" Review: https://www.themoonlight.io/en/review/representation-entanglement-for-generationtraining-diffusion-transformers-is-much-easier-than-you-think
4. **[arXiv:2410.06940]** Sihyun Yu et al., "Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think" (REPA), arXiv, Oct 2024. https://arxiv.org/abs/2410.06940
5. **[arXiv:2212.09748]** W. Peebles & S. Xie, "Scalable Diffusion Models with Transformers" (DiT), arXiv, 2022. https://arxiv.org/abs/2212.09748
6. **[arXiv:2512.16636]** "REGLUE: Representation Entanglement with Global–Local Unified Encoding," arXiv, 2025.
7. **[OpenReview]** REG OpenReview page: https://openreview.net/forum?id=koEALFNBj1
8. **[NASA ADS Abstract]** https://ui.adsabs.harvard.edu/abs/2025arXiv250701467W/abstract
9. **[HuggingFace Papers]** https://huggingface.co/papers/2507.01467
10. **[GitHub]** REG Official Code: https://github.com/Martinser/REG

> ⚠️ **정확도 주의사항**: 본 답변은 arXiv 논문 원문(2507.01467) 및 검색된 리뷰 자료에 기반합니다. 논문이 2025년 7월에 공개된 최신 연구로, 일부 세부 수식(특히 $\mathcal{L}_{REG}$ 구성)은 논문 원문의 표기를 최대한 반영했으나, 완전한 수식 체계 확인을 위해서는 원문 PDF를 직접 참조하시기를 권장합니다.
