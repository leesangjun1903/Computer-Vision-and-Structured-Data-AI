
# DNF-Intrinsic: Deterministic Noise-Free Diffusion for Indoor Inverse Rendering

> **📌 출처 및 참고자료**
> - **[1]** Zheng et al., "DNF-Intrinsic: Deterministic Noise-Free Diffusion for Indoor Inverse Rendering," arXiv:2507.03924 (2025) — https://arxiv.org/abs/2507.03924
> - **[2]** ICCV 2025 논문 전문 — https://openaccess.thecvf.com/content/ICCV2025/papers/Zheng_DNF-Intrinsic_...
> - **[3]** arXiv HTML 전문 — https://arxiv.org/html/2507.03924v2
> - **[4]** GitHub 공식 코드 — https://github.com/OnlyZZZZ/DNF-Intrinsic
> - **[5]** ResearchGate — https://www.researchgate.net/publication/393477048
> - **[6]** EmergentMind — https://www.emergentmind.com/topics/diffusion-based-neural-renderer
> - **[7]** Channel-wise Noise Scheduled Diffusion (arXiv:2503.09993)
> - **[8]** RenderFlow (arXiv:2601.06928)

---

## 1. 핵심 주장 및 주요 기여 요약

기존의 사전학습된 diffusion 모델을 fine-tuning하여 image-conditioned **noise-to-intrinsic mapping**으로 역렌더링을 수행하는 연구들이 주목받아 왔으나, 이 패러다임은 구조(structure)와 외관(appearance) 정보가 손상된 노이즈 이미지를 사용하므로 고품질 결과를 안정적으로 생성하지 못하는 한계가 있었습니다.

이 문제를 해결하기 위해 DNF-Intrinsic은 **Gaussian noise 대신 소스 이미지 자체를 입력으로 사용**하여 flow matching을 통해 결정론적(deterministic) 고유 속성(intrinsic properties)을 직접 예측하는 방식을 제안하며, 물리적으로 신뢰할 수 있는 결과를 보장하는 **Generative Renderer**를 함께 설계하였습니다.

### 핵심 주장 요약

| 항목 | 내용 |
|------|------|
| **발표 학회** | ICCV 2025 |
| **저자** | Rongjia Zheng 외 3인 |
| **핵심 전환점** | Noise-to-Intrinsic → **Image-to-Intrinsic** |
| **핵심 기법** | Flow Matching + LoRA fine-tuning + Generative Renderer |
| **추정 대상** | Albedo, Metallic, Roughness, Normal, Depth |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

기존 diffusion 기반 역렌더링 방식은 세 가지 핵심 한계를 가집니다:
1. **품질 불안정**: Noise-to-Intrinsic 패러다임은 이미지의 완전한 구조·외관 정보를 활용할 수 없어 고품질 예측이 어려움
2. **느린 추론 속도**: 랜덤 노이즈에서 고유 속성으로 매핑하기 위해 수많은 denoising 스텝이 필요
3. **물리적 비일관성**: 예측된 고유 속성으로부터 원본 이미지를 재구성하는 명시적 제약이 없어 물리적으로 설득력 없는 결과가 발생

### 2-2. 제안 방법 (수식 포함)

#### (A) 기존 방식의 학습 목표 (Noise-to-Intrinsic)

기존 방식은 albedo $A \in \mathbb{R}^{W \times H \times 3}$를 조건 이미지 $I \in \mathbb{R}^{W \times H \times 3}$에 대한 조건부 분포 $\mathcal{D}(A \mid I)$로 모델링합니다.

사전학습된 diffusion 모델은 보통 조건부 분포 $\mathcal{D}(A \mid I)$를 학습하도록 훈련됩니다:

$$\mathcal{L}_{\text{noise}} = \mathbb{E}_{t, \epsilon \sim \mathcal{N}(0,I)} \left[ \left\| \epsilon - \epsilon_\theta(Z_t, t, I) \right\|^2 \right]$$

여기서 $Z_t$는 noisy latent, $\epsilon_\theta$는 noise 예측 네트워크입니다.

---

#### (B) DNF-Intrinsic의 핵심: Image-to-Intrinsic Flow Matching

DNF-Intrinsic은 기존의 image-conditioned noise-to-intrinsic mapping 대신, **소스 이미지를 직접 입력으로 취하여 flow matching을 통해 결정론적으로 고유 속성을 예측**하는 image-to-intrinsic mapping을 제안합니다.

**Flow Matching의 궤적 보간:**

사전학습된 VAE 인코더 $E$를 사용하여 입력 이미지의 latent code $Z$를 flow 궤적의 시작점으로 초기화하고, 시간 스텝 $t$에서의 noised latent $Z_t$는 $Z$와 latent intrinsic $Z_i$ 사이의 **보간(interpolation)**을 통해 얻습니다:

$$Z_t = (1 - t) \cdot Z + t \cdot Z_i, \quad t \in [0, 1]$$

**Flow Matching 학습 목표:**

$$\mathcal{L}_{\text{flow}} = \mathbb{E}_{t, Z, Z_i} \left[ \left\| \mu_\theta(Z_t, t, p_i) - (Z_i - Z) \right\|^2 \right]$$

여기서 $\mu_\theta$는 flow velocity 예측 네트워크이며, $p_i$는 각 고유 속성에 대응하는 **텍스트 프롬프트**입니다.

**추론 과정:**

추론 시에는 latent code $Z$에서 출발하여, 대응하는 텍스트 프롬프트에 의해 트리거된 각 intrinsic flow를 순회함으로써 예측된 latent intrinsic을 얻고, VAE 디코더 $D$를 통해 최종 고유 속성을 복원합니다.

구체적으로, 정규화된 타임스텝 $t \in \{0, \ldots, K-1\}/K$에 대해 반복적으로:

$$Z_{t+1/K} = Z_t + \frac{1}{K} \cdot \mu_\theta(Z_t, t, p_i)$$

---

#### (C) Generative Renderer와 재구성 손실

각 고유 속성은 학습 데이터셋의 ground truth로 지도학습되지만, 이들 속성과 입력 이미지 사이의 직접적인 제약이 없어 잠재적 관계가 무시될 수 있습니다. 이를 해결하기 위해 **Generative Renderer**를 개발하고 재구성 손실(reconstruction loss) $\mathcal{L}_{\text{rec}}$을 설계합니다.

물리 기반 렌더링 방정식을 근사하여:

$$\hat{I} = \mathcal{R}(\hat{A}, \hat{M}, \hat{R}, \hat{N}) $$

$$\mathcal{L}_{\text{rec}} = \left\| \hat{I} - I \right\|_1$$

전체 학습 목표:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{flow}} + \lambda \cdot \mathcal{L}_{\text{rec}}$$

조명 정보의 부재를 보완하기 위해, 학습 반복 동안 다양한 조명을 샘플링합니다. 이는 모든 가능한 조명을 적분하는 것과 기능적으로 동일하며, 재구성 손실을 최소화함으로써 최종적으로 **조명 독립적(lighting-independent)인 고유 속성**으로 수렴하도록 모델이 학습됩니다.

---

### 2-3. 모델 구조

UNet 기반 diffusion denoising을 주로 사용하는 기존 방법들과 달리, DNF-Intrinsic은 **Diffusion Transformer (DiT)**가 더 효과적인 flow estimator임을 실험적으로 확인하였으며, 전역 정보(global information) 활용이 역렌더링 성공에 핵심적입니다. 이에 따라 **Stable Diffusion V3의 사전학습된 DiT**를 flow estimator $\mu_\theta$로 사용하여 fine-tuning합니다.

전체 사전학습 모델을 fine-tuning하거나 추가적인 ControlNet 브랜치를 추가하는 대신, 원래의 DiT 아키텍처를 유지하고 **LoRA(Low-Rank Adaptation)**를 적용하여 매우 적은 수의 파라미터만 fine-tuning합니다.

공식 코드 기준 추론 명령은 Stable Diffusion 3 medium 사전학습 모델과 LoRA checkpoint를 기반으로 하며, `--num_inference_steps` 파라미터로 스텝 수를 조정합니다.

**모델 구조 도식:**

```
Input Image I
      │
   VAE Encoder E
      │
  Latent Z (flow 시작점)
      │
  Text Prompt p_i → [LoRA fine-tuned DiT]
      │
  Flow Matching (Z → Z_i)
      │
  VAE Decoder D
      │
  Intrinsic Map (Albedo / Metallic / Roughness / Normal / Depth)
      │
  Generative Renderer R → Î
      │
  ℒ_rec = ||Î - I||
```

---

### 2-4. 성능 향상

InteriorVerse 합성 실내 장면 데이터셋에서 fine-tuning하여, 단일 RGB 실내 이미지로부터 albedo, metallic, roughness, normal, depth를 복원하는 SOTA 성능을 달성합니다. Albedo 추정 기준 **단일 스텝에서 PSNR 21.05**를 기록하며, InverseIndoor, IndoorIR, IntrinsicAnything, IntrinsicDiff, RGBX 등 기존 방법들을 크게 능가합니다.

또한 **추론 속도 0.1초(fastest)**와 **학습 가능 파라미터 18.87M(fewest)**라는 이중 우위를 달성하여 실용적 효율성을 증명하였습니다.

| 방법 | PSNR (Albedo) | 추론 시간 | 학습 파라미터 |
|------|---------------|-----------|---------------|
| IntrinsicDiff | < 21.05 | 느림 (다수 스텝) | 더 많음 |
| IntrinsicAnything | < 21.05 | 느림 | 더 많음 |
| RGBX | < 21.05 | 느림 | 더 많음 |
| **DNF-Intrinsic** | **21.05** | **0.1초** | **18.87M** |

---

### 2-5. 한계

해당 연구의 잠재적 한계로는 ① 계산 복잡도로 인한 실시간 응용 제한, ② 다양한 실내 환경 유형에 따른 성능 편차, ③ 최적 결과를 위한 상당량의 학습 데이터 필요성 등이 지적됩니다.

추가적으로, 논문의 학습 데이터인 **InteriorVerse**는 합성(synthetic) 데이터셋에 한정되어, 실세계 도메인 갭(domain gap)이 존재할 수 있습니다. 또한 실외(outdoor) 환경에 대한 일반화 가능성은 별도로 검증이 필요합니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 현재 일반화 전략

기존 방법들이 image-conditioned noise-to-intrinsic mapping을 학습하는 것과 달리, DNF-Intrinsic은 소스 이미지를 Gaussian noise 대신 입력으로 사용하여 flow matching을 통해 결정론적 image-to-intrinsic mapping을 학습합니다. 이를 통해 소스 이미지의 시각 정보를 최대한 활용함으로써 더 강인한 intrinsic 예측 성능을 달성합니다.

합성 및 실제 데이터셋(synthetic and real-world) 모두에서의 실험 결과, 본 방법이 기존 SOTA 방법들을 명확히 능가함을 보여줍니다.

### 3-2. 일반화 가능성의 핵심 요인

#### ① 사전학습 모델의 강력한 Prior 활용
사전학습된 diffusion 모델의 prior가 noise-free image-to-target mapping으로 fine-tuning될 때도 여전히 유효하다는 것이 실험적으로 검증되어 있습니다. 이는 Stable Diffusion V3가 광범위한 실세계 이미지로 사전학습되어 강력한 **범용 시각 표현(visual representation)**을 갖추고 있기 때문입니다.

#### ② LoRA의 파라미터 효율성
원본 DiT 아키텍처를 보존하면서 LoRA만을 적용하는 전략은 사전학습 지식(prior knowledge)을 최대한 보존하면서 task-specific 지식만을 추가하므로, **domain shift에 대한 강인성**을 높입니다.

#### ③ 텍스트 프롬프트 기반 조건화
각 고유 속성(albedo, metallic 등)을 텍스트 프롬프트로 트리거함으로써, **새로운 속성이나 조건으로의 확장**이 용이합니다.

#### ④ Pseudo GT를 통한 In-the-wild 일반화
모델의 예측과 다중 조명(multi-illumination) 데이터를 활용하여 **밀집된 pseudo ground truth를 생성하는 방법**을 개발함으로써, in-the-wild 이미지로의 일반화를 가능케 합니다.

### 3-3. 향후 일반화 향상을 위한 방향

| 전략 | 설명 |
|------|------|
| **도메인 적응** | 실외/혼합 환경 데이터셋 포함 학습 |
| **자기지도 학습** | 레이블 없는 실세계 이미지 활용 |
| **Test-Time Adaptation** | 추론 시 입력 분포 적응 |
| **Multi-view 확장** | 다시점 기하학적 일관성 제약 추가 |

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4-1. 향후 연구에 미치는 영향

#### ① Deterministic Diffusion 패러다임의 확산
DNF-Intrinsic은 확률적 noise-to-intrinsic mapping을 넘어 flow matching을 통한 결정론적 image-to-intrinsic mapping을 직접 학습함으로써, **고품질의 물리적으로 일관된 장면 속성 복원을 고속으로 달성**하는 새로운 패러다임을 제시합니다.

이는 확률론적 추론이 Image-to-Normal과 같은 결정론적 특성의 작업과 상충된다는 기존의 문제의식과 맞닿아 있으며, 역렌더링, 법선 추정, 깊이 추정 등 **다양한 intrinsic vision 작업에 적용 가능한 통합 프레임워크**로 발전할 수 있습니다.

#### ② AR/VR 및 씬 편집으로의 직접 응용
역렌더링을 통해 획득한 고유 속성들은 AR/VR, 세그멘테이션·추적, 재조명(relighting), 재질 편집(material editing) 등 폭넓은 응용으로 이어집니다.

#### ③ 단일 스텝 추론의 실용화
0.1초 단일 스텝 추론이라는 성능은 **실시간 혹은 온디맨드(on-demand) 역렌더링 파이프라인**의 현실적 구현 가능성을 열어줍니다.

---

### 4-2. 관련 최신 연구 비교 분석 (2020년 이후)

최근 연구 트렌드는 사전학습된 diffusion 모델의 강력한 이미지 prior를 기하 예측, 재질 추정, 역렌더링에 활용하는 방향으로 발전해 왔으나, 이들은 대부분 **image-conditioned noise-to-target 패러다임에 의존하여 랜덤 노이즈 초기화에 민감하고 불필요한 디테일 왜곡을 유발**한다는 공통 한계를 가집니다.

| 연구 | 방법론 | 특징 | 한계 |
|------|--------|------|------|
| **IntrinsicDiff** (2023) | Diffusion (noise-to-intrinsic) | 생성적 역렌더링 | 느린 추론, 불안정 |
| **IntrinsicAnything** (2023) | Diffusion fine-tuning | 범용 intrinsic 분해 | 노이즈 민감성 |
| **RGBX** (2024) | Diffusion multi-output | 다중 속성 동시 예측 | 느린 추론 |
| **Uni-Renderer** (2024) | 렌더링·역렌더링 조건부 diffusion | 이중 스트림, cycle-consistency | 복잡한 아키텍처 |
| **Channel-wise Noise Diffusion** (2025) | 채널별 노이즈 스케줄 | 단일/다중 솔루션 가능 | 모달 간 독립 추론 문제 |
| **DiffusionRenderer** (2025) | 비디오 diffusion G-buffer | 비디오 기반 편집 가능 | 3D 재구성 불필요하나 비디오 필요 |
| **DNF-Intrinsic** (2025) | Flow Matching + LoRA + DiT | 결정론적, 단일 스텝, 18.87M 파라미터 | 합성 데이터 의존, 실외 미검증 |

---

### 4-3. 앞으로 연구 시 고려할 점

1. **도메인 갭 해소**: InteriorVerse 합성 데이터 의존에서 벗어나, 실세계 데이터를 포함한 **혼합 학습 전략(mixed training)**이 필요합니다.

2. **전역-지역 정보 균형**: DiT의 전역 정보 활용이 역렌더링 성공에 핵심적이나, 세밀한 로컬 텍스처 복원을 위한 **다중 스케일 어텐션 메커니즘** 연구가 필요합니다.

3. **조명 모델링의 명시적 확장**: 현재 조명을 재구성 손실에서 샘플링으로 처리하는 방식을 넘어, **공간적으로 변하는 조명(spatially-varying illumination)의 명시적 모델링**이 향후 과제입니다.

4. **멀티모달 일관성**: 역렌더링에서는 재질이 어두울수록 조명이 밝아야 하는 등 모달 간 의존성이 중요한데, 기존 diffusion 기반 방법들은 각 모달리티를 독립적으로 추론하여 이러한 의존성을 포착하기 어렵습니다. 따라서 **모달리티 간 상호작용 메커니즘** 설계가 필수적입니다.

5. **평가 지표의 다양화**: PSNR 외에도 지각적 품질(LPIPS), 물리적 정확도(물질 보존 법칙 준수 여부), 다운스트림 응용 품질(relighting 성능) 등을 포함한 **다차원 평가 체계** 구축이 권장됩니다.

6. **확장성(Scalability) 검토**: LoRA 18.87M 파라미터의 경량 설계가 더 복잡한 실내 장면이나 고해상도 입력에서도 유효한지 **스케일링 법칙(scaling law)** 분석이 필요합니다.

---

> ⚠️ **정확도 관련 고지**: 본 논문은 2025년 7월 arXiv에 게재되어 ICCV 2025에 채택된 최신 논문입니다. 수식의 일부(특히 flow velocity 예측의 세부 formulation)는 논문 전문 PDF의 접근 범위에서 검색된 정보를 기반으로 재구성하였으므로, 정확한 표기를 위해서는 arXiv 원문([arXiv:2507.03924](https://arxiv.org/abs/2507.03924)) 및 [ICCV 2025 논문 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Zheng_DNF-Intrinsic_Deterministic_Noise-Free_Diffusion_for_Indoor_Inverse_Rendering_ICCV_2025_paper.pdf) 직접 확인을 권장합니다.
