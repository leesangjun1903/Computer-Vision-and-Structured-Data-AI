
# Weak-to-Strong Diffusion with Reflection (W2SD)

> **논문 정보**
> - **제목**: Weak-to-Strong Diffusion with Reflection
> - **저자**: Lichen Bai, Masashi Sugiyama, Zeke Xie
> - **arXiv**: [2502.00473](https://arxiv.org/abs/2502.00473) (2025.02)
> - **학회**: ICLR 2026 (accepted)
> - **코드**: [GitHub - xie-lab-ml/Weak-to-Strong-Diffusion-with-Reflection](https://github.com/xie-lab-ml/Weak-to-Strong-Diffusion-with-Reflection)

---

## 1. 핵심 주장 및 주요 기여 (요약)

확산 생성 모델(Diffusion Generative Model)의 목표는 스코어 매칭(score matching)을 통해 학습된 분포를 실제 데이터 분포와 일치시키는 것이지만, 학습 데이터 품질·모델링 전략·아키텍처 설계의 본질적 한계로 인해 생성 결과와 실제 데이터 사이의 간극이 불가피하게 존재한다.

이 간극을 줄이기 위해 논문은 **Weak-to-Strong Diffusion (W2SD)**을 제안하는데, 이는 기존의 약한 모델(weak model)과 강한 모델(strong model) 사이의 추정된 차이(weak-to-strong difference)를 활용하여 이상적인 모델(ideal model)과 강한 모델 사이의 간극을 근사하는 방법이다. Denoising과 Inversion을 교대로 수행하는 반사(reflection) 연산을 통해, W2SD는 샘플링 궤적을 따라 잠재 변수(latent variable)를 실제 데이터 분포 영역으로 유도한다.

### 주요 기여 요약

| 항목 | 내용 |
|---|---|
| **핵심 아이디어** | Weak–Strong 모델 쌍의 차이로 이상적 모델 근사 |
| **방법론** | Denoising ↔ Inversion 반복(Reflective Operation) |
| **적용 범위** | UNet, DiT, MoE 등 다양한 아키텍처; 이미지, 비디오 등 다양한 모달리티 |
| **성능** | HPSv2 기준 최대 90% 승률 향상 |
| **확장성** | 추가 학습 없이 추론(inference) 단계에서만 적용 가능 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

확산 생성 모델의 목표는 스코어 매칭으로 학습된 분포를 실제 데이터 분포에 정렬시키는 것이나, 훈련 데이터 품질·모델링 전략·아키텍처 설계의 내재적 한계로 인해 생성 결과와 실제 데이터 사이의 간극이 필연적으로 존재한다.

이를 수식으로 표현하면, 이상적 모델의 스코어 함수 $\nabla \log p_{\text{ideal}}(\mathbf{x}\_t)$와 실제 학습된 강한 모델의 스코어 $\nabla \log p_{\text{strong}}(\mathbf{x}_t)$ 사이에는 다음과 같은 간극이 존재합니다:

$$\Delta_{\text{ideal-strong}} = \nabla \log p_{\text{ideal}}(\mathbf{x}_t) - \nabla \log p_{\text{strong}}(\mathbf{x}_t) \neq 0$$

---

### 2-2. 제안하는 방법 (핵심 수식 포함)

#### ① Weak-to-Strong Difference 정의

W2SD는 weak-to-strong difference를 활용하여 strong-to-ideal difference를 근사한다.

$$\Delta_{\text{w2s}} = \epsilon_{\theta_{\text{strong}}}(\mathbf{x}_t, c) - \epsilon_{\theta_{\text{weak}}}(\mathbf{x}_t, c)$$

여기서:
- $\epsilon_{\theta_{\text{strong}}}$: 강한 모델의 노이즈 예측
- $\epsilon_{\theta_{\text{weak}}}$: 약한 모델의 노이즈 예측
- $c$: 조건(텍스트 프롬프트 등)

이 차이가 이상적 모델에 도달하기 위한 보정량을 근사한다고 가정합니다:

$$\nabla \log p_{\text{ideal}}(\mathbf{x}_t) \approx \nabla \log p_{\text{strong}}(\mathbf{x}_t) + \lambda \cdot \Delta_{\text{w2s}}$$

#### ② W2SD 수정된 스코어 추정

CFG(Classifier-Free Guidance)와 유사하게, W2SD의 수정된 스코어 추정은 다음과 같이 쓸 수 있습니다:

$$\hat{\epsilon}_{\text{W2SD}}(\mathbf{x}_t, c) = \epsilon_{\theta_{\text{strong}}}(\mathbf{x}_t, c) + \lambda \left[\epsilon_{\theta_{\text{strong}}}(\mathbf{x}_t, c) - \epsilon_{\theta_{\text{weak}}}(\mathbf{x}_t, c)\right]$$

$$= (1+\lambda)\,\epsilon_{\theta_{\text{strong}}}(\mathbf{x}_t, c) - \lambda\,\epsilon_{\theta_{\text{weak}}}(\mathbf{x}_t, c)$$

이는 CFG의 일반화된 형태로, strong/weak 모델 쌍을 조건부/비조건부 쌍 대신 사용하는 구조입니다.

#### ③ Reflective Operation (핵심 메커니즘)

W2SD는 LLM 분야에서 광범위하게 연구된 반사 메커니즘(reflective mechanism)—즉, 이전 상태를 기반으로 생성 결과를 수정하는 과정—을 참조하여, weak-to-strong difference를 경험적으로 추정한다.

Denoising과 Inversion을 weak-to-strong difference와 교대로 수행하는 반사 연산을 통해, W2SD는 샘플링 궤적을 따라 잠재 변수를 실제 데이터 분포 영역으로 유도함을 이론적으로 이해할 수 있다.

Reflective Operation의 반복 단계는 개념적으로 다음과 같습니다:

**Step 1 (Denoising):**

$$\mathbf{x}_{t-1} = \text{Denoise}_{\text{strong}}(\mathbf{x}_t) + \lambda \cdot \Delta_{\text{w2s}}(\mathbf{x}_t)$$

**Step 2 (Inversion):**

$$\mathbf{x}_{t}^{'} = \text{Invert}(\mathbf{x}_{t-1})$$

**Step 3 (Re-Denoising):**

$$\mathbf{x}_{t-1}^{'} = \text{Denoise}_{\text{strong}}(\mathbf{x}_{t}^{'}) + \lambda \cdot \Delta_{\text{w2s}}(\mathbf{x}_{t}^{'})$$

이 지그재그(Zigzag) 방식의 반복이 잠재 변수를 실제 데이터 분포 쪽으로 점진적으로 이동시킵니다.

---

### 2-3. 모델 구조

W2SD는 DreamShaper vs. SD1.5, MoE의 good experts vs. bad experts 등 다양한 weak-to-strong 모델 쌍의 전략적 선택을 통해 다양한 개선을 가능하게 하는 유연하고 광범위하게 적용 가능한 프레임워크이다.

```
┌─────────────────────────────────────────────────────────────┐
│                     W2SD Framework                          │
│                                                             │
│  Input Noise z_T                                            │
│        │                                                    │
│        ▼                                                    │
│  ┌──────────────┐    ┌──────────────┐                      │
│  │  Strong Model│    │  Weak Model  │                      │
│  │  ε_strong    │    │  ε_weak      │                      │
│  └──────┬───────┘    └──────┬───────┘                      │
│         │                   │                              │
│         └────────┬──────────┘                              │
│                  │                                         │
│         Δ_w2s = ε_strong - ε_weak                         │
│                  │                                         │
│         ε_ideal ≈ ε_strong + λ·Δ_w2s                     │
│                  │                                         │
│   ┌──────────────▼─────────────────┐                       │
│   │  Reflective Operation          │                       │
│   │  Denoise ↔ Inversion 교대 반복 │                       │
│   └──────────────┬─────────────────┘                       │
│                  │                                         │
│         Output z_0 (Real Data 분포 근접)                    │
└─────────────────────────────────────────────────────────────┘
```

적용 가능한 weak-strong 쌍의 예:

| Weak Model | Strong Model | 목적 |
|---|---|---|
| SD 1.5 | DreamShaper | 일반 품질 향상 |
| DDIM (no ControlNet) | ControlNet 파이프라인 | 참조 이미지 정렬 |
| Bad Expert (MoE) | Good Expert (MoE) | MoE 내 전문가 선택 최적화 |
| Low CFG Scale | High CFG Scale | 프롬프트 충실도 향상 |
| No LoRA | Strong LoRA | 스타일/개념 강화 |

예를 들어, ControlNet을 strong model pipeline으로, DDIM을 weak model pipeline으로 설정하면 W2SD가 참조 이미지와의 정렬을 개선한다.

---

### 2-4. 성능 향상

광범위한 실험을 통해 W2SD가 인간 선호도, 미적 품질, 프롬프트 충실도를 크게 향상시키며, 이미지·비디오 등 다양한 모달리티, UNet 기반·DiT 기반·MoE 등 다양한 아키텍처, 여러 벤치마크에서 SOTA 성능을 달성함을 입증하였다. 예를 들어, W2SD가 적용된 Juggernaut-XL은 HPSv2 winning rate에서 기존 결과 대비 최대 90%까지 향상되었다.

더욱이, W2SD로 인한 성능 향상은 추가적인 연산 오버헤드를 크게 상회하며, 서로 다른 weak-to-strong difference에서 얻은 누적적 개선이 실용성과 배포 가능성을 더욱 공고히 한다.

---

### 2-5. 한계점

공개된 자료를 기반으로 파악 가능한 한계는 다음과 같습니다:

1. **추론 비용 증가**: Reflective Operation은 denoising + inversion을 반복하므로 기본 샘플링 대비 추론 시간이 증가합니다 (단, 논문은 이 오버헤드가 성능 향상에 비해 작다고 주장).
2. **약한 모델 선택의 의존성**: 적절한 weak-strong 모델 쌍을 선택하는 문제가 남아 있으며, 쌍 선택에 따라 성능이 달라질 수 있습니다.
3. **이론적 근사의 한계**: Weak-to-strong difference로 ideal model을 근사하는 것은 이론적 가정이므로, 모델 쌍이 충분히 다르지 않거나 너무 다를 경우 근사 오류가 발생할 수 있습니다.
4. **새로운 도메인에서의 검증 부족**: 주로 이미지/비디오 생성에서 검증되었으며, 의료 영상, 3D 생성 등 특수 도메인에서의 일반화 성능은 미검증입니다.

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

W2SD는 기존 weak-strong 모델 쌍 사이의 추정된 간극을 활용하여 이상적인 모델과 강한 모델 사이의 간극을 연결하며, 잠재 변수를 실제 데이터 분포 영역으로 유도함으로써 일반화를 향상시킨다.

일반화 성능 향상 가능성을 구체적으로 정리하면 다음과 같습니다:

### ① 아키텍처 무관 일반화 (Architecture-agnostic Generalization)

W2SD는 이미지·비디오 등 다양한 모달리티, UNet 기반·DiT 기반·MoE 등 다양한 아키텍처, 그리고 다양한 벤치마크에서 SOTA 성능을 달성하며 인간 선호도, 미적 품질, 프롬프트 충실도를 크게 개선한다.

이는 W2SD가 특정 아키텍처에 종속되지 않고 **추론 시 플러그인 방식**으로 범용 적용이 가능함을 시사합니다.

### ② 학습 불필요 (Training-free) 특성에 의한 도메인 이전 가능성

W2SD는 추가적인 파인튜닝 없이 추론 단계에서만 작동하므로, 새로운 도메인에서도 기존에 학습된 strong/weak 모델 쌍을 그대로 활용할 수 있어 **제로샷(zero-shot) 일반화** 가능성이 높습니다.

### ③ 누적적 개선을 통한 일반화 강화

W2SD로 인한 성능 향상은 추가 연산 오버헤드를 크게 상회하며, 서로 다른 weak-to-strong difference에서의 누적적 개선이 실용성과 배포 가능성을 공고히 한다.

서로 다른 weak-strong 쌍을 **앙상블(ensemble)** 방식으로 조합하면, 단일 모델보다 더 강인한 일반화 성능을 기대할 수 있습니다.

### ④ OOD(Out-of-Distribution) 견고성

확산 모델의 OOD 과제에 대한 성능은 여전히 탐구가 부족한 과제이며, 훈련 데이터셋에만 존재하는 이미지 특징을 환각(hallucinate)하는 현실적인 재구성 문제가 발생할 수 있다.

W2SD는 실제 데이터 분포에 잠재 변수를 더 가깝게 유도함으로써, OOD 상황에서도 분포 외 아티팩트 생성 가능성을 줄이는 방향으로 일반화 성능에 기여할 수 있습니다.

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4-1. 향후 연구에 미치는 영향

#### ① CFG(Classifier-Free Guidance)의 일반화 프레임워크로서의 역할
W2SD는 기존 CFG의 "조건부/비조건부 모델 차이"를 "strong/weak 모델 차이"로 일반화합니다. 이는 **더 넓은 의미의 "Guidance" 프레임워크**로 확장될 수 있으며, 다양한 모델 쌍의 차이를 활용한 새로운 guidance 기법 연구를 촉진할 것입니다.

#### ② LLM의 Self-Reflection 개념을 확산 모델로 이전
W2SD는 LLM 분야에서 광범위하게 연구된 반사 메커니즘—이전 상태를 기반으로 생성 결과를 수정하는 과정—을 참조한다. 이 개념의 이전(transfer)은 **확산 모델과 LLM 간의 방법론적 융합** 연구를 자극할 것입니다.

#### ③ 선행 연구 ZigZag Diffusion과의 관계
동일 저자의 선행 연구인 "ZigZag Diffusion Sampling: Diffusion models can self-improve via self-reflection"이 ICLR에서 발표되었으며, W2SD는 이 self-reflection 개념을 weak-strong 모델 쌍으로 확장한 것입니다. 이러한 **"자기 반성 기반 샘플링"** 연구 계보가 확산 모델 추론 최적화의 새로운 방향을 제시합니다.

#### ④ MoE(Mixture of Experts) 최적화에의 활용
DreamShaper vs. SD1.5, MoE의 good experts vs. bad experts 쌍 등 다양한 선택이 가능하다는 점에서, W2SD는 대형 MoE 모델의 추론 최적화에 새로운 방향을 제시합니다.

---

### 4-2. 향후 연구 시 고려할 점

| 고려 사항 | 설명 |
|---|---|
| **최적 weak-strong 쌍 선택 기준** | 어떤 쌍이 최적인지에 대한 이론적/경험적 기준이 필요. 자동화된 쌍 탐색(NAS-like 방법) 연구 필요 |
| **$\lambda$ 하이퍼파라미터 스케줄링** | W2SD의 강도 $\lambda$를 타임스텝 $t$에 따라 동적으로 조정하는 스케줄링 전략 연구 필요 |
| **추론 비용 최적화** | Reflection 횟수와 성능 향상 간의 트레이드오프 분석 및 효율적인 반복 전략 설계 필요 |
| **이론적 수렴 보장** | Reflective Operation이 항상 실제 데이터 분포로 수렴하는지에 대한 더 강한 이론적 보장 필요 |
| **3D/의료/과학 도메인 검증** | 이미지·비디오 외 특수 도메인(의료 영상, 분자 생성 등)으로의 확장 가능성 검증 필요 |
| **멀티모달 확장** | 텍스트·오디오 등 다른 모달리티의 확산 모델에 W2SD 적용 가능성 연구 |
| **약한 모델의 최소 조건** | Weak 모델이 얼마나 "약해야" 유효한지에 대한 분석 (너무 약하면 noise가 증폭될 수 있음) |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 핵심 방법 | W2SD와의 관계 |
|---|---|---|
| **DDPM** (Ho et al., NeurIPS 2020) | Score matching 기반 확산 모델 기초 | W2SD가 해결하려는 분포 간극 문제의 기원 |
| **CFG** (Ho & Salimans, 2021) | 조건부-비조건부 스코어 차이로 guidance | W2SD의 직접적 일반화 (약/강 모델 쌍으로 확장) |
| **DDIM** (Song et al., ICLR 2021) | 결정론적 inversion/sampling | W2SD의 reflective inversion에 활용 가능한 기반 기술 |
| **ZigZag Diffusion** (Bai et al., ICLR 2025) | Self-reflection 기반 자기 개선 | W2SD의 직접적 선행 연구 (동일 저자) |
| **Universal Guidance** (Bansal et al., CVPR 2023) | 범용 guidance 프레임워크 | 유사한 목표, 그러나 그래디언트 기반으로 계산 비용 높음 |
| **SDXL** (Podell et al., 2023) | 대형 UNet 기반 고품질 이미지 생성 | W2SD의 strong model로 활용 가능 |
| **Stable Diffusion 3 / FLUX (DiT)** (2024) | Flow Matching + DiT 아키텍처 | W2SD의 DiT-based 실험에서 검증됨 |
| **TeEFusion** (2025) | 텍스트 임베딩 혼합으로 CFG 증류 | W2SD를 strong/weak 모델을 명시적으로 통합한 확장으로 인용 |

---

## 📚 참고 자료 및 출처

1. **arXiv 원문**: Lichen Bai, Masashi Sugiyama, Zeke Xie. "Weak-to-Strong Diffusion with Reflection." arXiv:2502.00473 (2025). https://arxiv.org/abs/2502.00473

2. **arXiv HTML 전문**: https://arxiv.org/html/2502.00473v1

3. **arXiv PDF**: https://arxiv.org/pdf/2502.00473

4. **OpenReview (ICLR 2026)**: https://openreview.net/forum?id=tg19FVh3p1

5. **Hugging Face Papers**: https://huggingface.co/papers/2502.00473

6. **GitHub 공식 코드**: https://github.com/xie-lab-ml/Weak-to-Strong-Diffusion-with-Reflection

7. **선행 연구 (ZigZag Diffusion)**: Lichen Bai et al. "Zigzag diffusion sampling: Diffusion models can self-improve via self-reflection." ICLR 2025. arXiv:2502.00473 관련 참조.

8. **인용 논문 (TeEFusion)**: "TeEFusion: Blending Text Embeddings to Distill Classifier-Free Guidance." arXiv:2507.18192 (2025). https://arxiv.org/html/2507.18192v1

---

> ⚠️ **정확도 관련 안내**: 위 수식 중 W2SD의 핵심 아이디어(Δ_w2s, 수정된 스코어 추정)는 논문의 공개 내용과 CFG 이론을 바탕으로 재구성한 것입니다. 논문 본문의 세부 증명 수식(Theorem 등)은 arXiv PDF 원문(https://arxiv.org/pdf/2502.00473)을 직접 확인하시기를 강력히 권장드립니다.
