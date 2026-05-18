
# Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think

**논문 정보**
- **저자**: Gonzalo Martin Garcia, Karim Abou Zeid (Knaebel), Christian Schmidt, Daan de Geus, Alexander Hermans, Bastian Leibe
- **소속**: RWTH Aachen University 및 Eindhoven University of Technology
- **게재**: IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2025
- **arXiv**: [2409.11355](https://arxiv.org/abs/2409.11355)

---

## 1. 핵심 주장 및 주요 기여 요약

### 🎯 핵심 주장

대형 Diffusion 모델은 depth estimation을 이미지 조건부 이미지 생성 태스크로 재구성함으로써 고정밀 monocular depth estimator로 재활용될 수 있음이 알려졌다. 그러나 multi-step inference로 인한 높은 연산 비용이 많은 시나리오에서 활용을 제한하였으며, 본 논문은 이 비효율성의 원인이 지금까지 발견되지 않은 inference pipeline의 결함에 있음을 밝힌다.

### 🏆 주요 기여 3가지

| # | 기여 | 설명 |
|---|------|------|
| ① | **Inference Pipeline 결함 발견 및 수정** | DDIM 스케줄러의 timestep spacing 설정 오류 발견 |
| ② | **단일 단계(Single-Step) 결정론적 모델** | 200배 이상의 속도 향상 |
| ③ | **E2E Fine-Tuning 프로토콜** | Task-specific loss로 SOTA 성능 달성 |

수정된 모델은 기존 최고 설정과 비슷한 성능을 내면서 200배 이상 빠르며, 단일 단계 모델 위에 태스크 특화 손실로 end-to-end fine-tuning을 수행하여 모든 diffusion 기반 depth 및 normal estimation 모델을 능가하는 결정론적 모델을 얻는다. 놀랍게도 이 fine-tuning 프로토콜이 Stable Diffusion에 직접 적용해도 현재 SOTA 수준의 성능에 도달함을 발견했으며, 이는 이전 연구들의 일부 결론에 의문을 제기한다.

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

#### 배경: Marigold (CVPR 2024)
Marigold는 monocular depth estimation을 위한 diffusion 모델로, 현대 생성 이미지 모델에 저장된 풍부한 시각적 지식을 활용하는 것이 핵심 원리이다. Stable Diffusion으로부터 파생되어 합성 데이터로 fine-tuning된 이 모델은 미학습 데이터셋에 zero-shot 전이가 가능하여 SOTA 결과를 제공한다.

#### 문제점
기존 모델들의 주요 문제는 multi-step inference로 인한 높은 연산 비용이었다. 저자들은 inference pipeline의 치명적인 결함을 발견했는데, 이전 연구들이 inference 중 noise level 샘플링에 suboptimal한 설정을 사용하고 있었음을 밝혔다.

구체적으로, 이 모델은 DDIM 스케줄러와 10~50 denoising step을 사용하도록 설계되었으나, `timestep_spacing: "trailing"` 설정을 재정의하면 단 1 step만으로도 좋은 예측을 얻을 수 있다.

Marigold v1.0은 inference 시 leading timestep을 사용하는 DDIM을 사용하여 최고 성능을 위해 4~10번의 함수 평가(NFE)가 필요했으나, Garcia et al.이 DDIM에서 trailing timestep으로 전환할 것을 제안하였고, 이 변경은 Marigold의 효율성을 크게 향상시켜 단 1 DDIM step(NFE=1)에서 성능이 포화되었다.

---

### 2.2 제안하는 방법 (수식 포함)

#### ① DDIM Timestep Spacing 수정

기존 Marigold는 **leading timestep spacing**을 사용:

```math
t_{\text{leading}} = \left\{T, T - \frac{T}{S}, T - \frac{2T}{S}, \ldots, \frac{T}{S}\right\}
```

여기서 $T=1000$, $S$는 총 denoising step 수. Leading spacing은 첫 step이 $T$부터 시작하지만, 마지막 step이 $0$에 도달하지 **못한다**.

수정된 **trailing timestep spacing**:

```math
t_{\text{trailing}} = \left\{T, T - \frac{T}{S}, \ldots, \frac{T}{S}, 0\right\}
```

trailing spacing은 마지막 step을 **$t=0$에 정확히 도달**하게 하여, 단 1 step($S=1$)만으로 $t=T \to t=0$ 전이가 완성된다.

#### ② 단일 스텝 결정론적 추론 설정

E2E FT 모델은 단일 단계 결정론적 depth/normal estimator로, 모델에 latent으로 평균 노이즈(즉, zeros)가 주어지고 timestep은 $t=1000$으로 고정된다.

수식적으로, 단일 step 추론 시:

$$\hat{x}_0 = \frac{x_T - \sqrt{1 - \bar{\alpha}_T}\, \hat{\epsilon}_\theta(x_T, t=T, c)}{\sqrt{\bar{\alpha}_T}}$$

- $x_T \sim \mathcal{N}(0, I)$ → fine-tuning 이후 $x_T = \mathbf{0}$ (zeros)으로 고정
- $\bar{\alpha}_T$: noise schedule에서 $t=T$일 때의 누적 product
- $\hat{\epsilon}_\theta$: U-Net의 노이즈 예측
- $c$: 조건 이미지(RGB input)의 인코딩

#### ③ End-to-End Fine-Tuning (E2E FT)

단일 단계 모델 위에 task-specific loss로 end-to-end fine-tuning을 수행하여, 일반적인 zero-shot 벤치마크에서 모든 diffusion 기반 depth/normal estimation 모델을 능가하는 결정론적 모델을 얻는다.

**Depth Estimation Loss** (affine-invariant):

$$\mathcal{L}_{\text{depth}} = \frac{1}{N}\sum_{i=1}^{N} \left( \log \hat{d}_i - \log d_i^* \right)^2 - \frac{\lambda}{N^2}\left(\sum_{i=1}^{N} \log \hat{d}_i - \log d_i^*\right)^2$$

**Surface Normal Estimation Loss**:

$$\mathcal{L}_{\text{normal}} = 1 - \frac{1}{N}\sum_{i=1}^{N} \hat{n}_i \cdot n_i^*$$

여기서:
- $\hat{d}_i$: 예측 depth, $d_i^*$: GT depth
- $\hat{n}_i$: 예측 법선 벡터, $n_i^*$: GT 법선 벡터
- $\lambda$: scale-invariance를 위한 가중치 항

#### ④ 학습 데이터셋

제안된 방법은 두 개의 합성 데이터셋을 훈련에 활용한다: photorealistic indoor scene을 위한 Hypersim과 driving 시나리오를 위한 Virtual KITTI 2.

팀은 Marigold의 하이퍼파라미터를 따르며, AdamW optimizer를 사용하여 모든 모델을 20,000 iterations 동안 훈련한다.

---

### 2.3 모델 구조

```
[입력 RGB 이미지]
       │
       ▼
┌─────────────────────────────────────────┐
│        VAE Encoder (Stable Diffusion)   │
│   이미지 → Latent Space (z_img)         │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│   Noise Latent: z_T = 0 (zeros)        │
│   Timestep: t = T = 1000 (고정)        │
└─────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│     Conditional U-Net (Denoiser)        │
│  [z_T; z_img] concatenation → 8채널    │
│  Text/Image Encoder 조건 적용           │
└─────────────────────────────────────────┘
       │ Single-step denoising (trailing)
       ▼
┌─────────────────────────────────────────┐
│        VAE Decoder                      │
│   Latent → Depth/Normal Map             │
└─────────────────────────────────────────┘
       │
       ▼
[예측 Depth Map / Surface Normal Map]
```

"E2E FT" 표기의 모델은 사전 훈련된 diffusion estimator 또는 Stable Diffusion으로부터 직접 task-specific loss로 end-to-end fine-tuning된 모델이다. fine-tuned 모델은 단일 단계 결정론적 모델이므로, 노이즈는 항상 zeros이고 ensemble 크기와 inference step 수는 항상 1이어야 한다.

---

### 2.4 성능 향상

#### 속도 향상
수정된 모델은 기존 최고 보고 설정과 비교 가능한 성능을 보이면서 200배 이상 빠르다.

#### 벤치마크 평가
평가를 위해 indoor 환경의 NYUv2와 ScanNet, mixed indoor-outdoor의 ETH3D와 DIODE, outdoor driving의 KITTI를 포함한 다양한 벤치마크가 사용되었다.

실험 결과 Marigold의 multi-step denoising process는 denoising step이 증가함에 따라 성능이 하락하는 문제가 있었으나, 수정된 DDIM 스케줄러는 모든 step 수에서 우월한 성능을 보였다. vanilla Marigold, LCM 변형과 연구팀의 single-step 모델을 비교한 결과, 수정된 DDIM 스케줄러는 단일 step에서 ensemble 없이도 비슷하거나 더 나은 결과를 달성했다.

---

### 2.5 한계점

논문은 제안된 방법의 잠재적 한계나 주의사항을 충분히 논의하지 않는다. 예를 들어 fine-tuned 모델이 더 다양하거나 도전적인 데이터셋에서 어떻게 수행되는지, 또는 신경망 기반이나 전통적인 컴퓨터 비전 알고리즘 기반의 다른 depth estimation 기법과 어떻게 비교되는지가 불분명하다.

Stable Diffusion에 fine-tuning 방법이 직접 적용된다는 발견은 흥미롭지만, 논문은 이 결과에 대한 충분한 분석이나 맥락을 제공하지 않아 독자가 더 광범위한 함의를 추론해야 한다.

추가적으로 확인 가능한 한계:
- **해상도 제약**: 모델은 기본 diffusion 모델의 유효 해상도인 약 768픽셀을 상속한다.
- **절대적(metric) 깊이 미제공**: affine-invariant depth만 예측하므로 실제 scale(미터 단위)을 직접 출력하지 않는다.
- **합성 데이터 학습**: Hypersim, Virtual KITTI 2라는 소규모 합성 데이터만 사용하여 특수한 실제 환경(의료 영상, 수중 등)에서 일반화가 제한될 수 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 Zero-Shot 일반화의 핵심 원천

Marigold는 monocular depth estimation을 위한 diffusion 모델로, 핵심 원리는 현대 생성 이미지 모델에 저장된 풍부한 시각적 지식을 활용하는 것이다. Stable Diffusion으로부터 파생되어 합성 데이터로 fine-tuning된 이 모델은 미학습 데이터셋에 zero-shot 전이가 가능하다.

본 논문의 E2E FT 접근은 이 zero-shot 능력을 **더욱 강화**하는 방향으로 작동한다:

$$\underbrace{\mathcal{L}_{\text{E2E FT}}}_{\text{태스크 특화}} = \mathcal{L}_{\text{task}} + \lambda_{\text{reg}}\mathcal{L}_{\text{reg}}$$

- $\mathcal{L}_{\text{task}}$: depth/normal에 대한 metric-specific 손실
- $\mathcal{L}_{\text{reg}}$: 사전 학습된 diffusion 모델의 일반화 능력 보존을 위한 정규화

### 3.2 Stable Diffusion 직접 적용 가능성

task-specific loss로 단일 단계 모델 위에 end-to-end fine-tuning을 수행하면 모든 diffusion 기반 depth/normal estimation 모델을 능가하는 결정론적 모델을 얻는다. 놀랍게도 이 fine-tuning 프로토콜이 Stable Diffusion에 직접 적용해도 현재 SOTA diffusion 기반 depth/normal estimation 모델과 비교 가능한 성능을 달성하며, 이는 이전 연구들의 일부 결론에 의문을 제기한다.

이는 곧 **복잡한 중간 pre-training 없이도** 일반 목적 diffusion 모델을 직접 dense prediction task에 적용할 수 있음을 의미하며, 다음과 같은 일반화 시나리오를 가능하게 한다:

| 일반화 방향 | 설명 |
|------------|------|
| **도메인 일반화** | indoor → outdoor, 실내 → 야외 등 도메인 간 전이 |
| **태스크 일반화** | depth → normal → optical flow 등 다른 dense prediction 태스크 |
| **모달리티 일반화** | RGB → RGB-D, thermal 등 다양한 입력 모달리티 |
| **베이스 모델 일반화** | SD → SDXL, SD3 등 더 큰 diffusion 모델 기반 적용 |

### 3.3 Fine-Tuning 전략과 일반화

inference pipeline의 결함을 수정하고 효과적인 fine-tuning 프로토콜을 구현함으로써 널리 사용되는 벤치마크에서 SOTA 결과를 달성했다. 이 연구는 monocular depth estimation 분야를 발전시킬 뿐 아니라, 주의 깊은 최적화와 개선 과정으로 이미지 관련 태스크에 복잡한 diffusion 모델의 잠재력을 밝힌다.

결론적으로, diffusion 모델을 기하학적 추정(geometry estimation)에 재활용하는 것은 end-to-end fine-tuning만큼 간단하다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 관련 연구 계보

```
[생성 모델 기반 Dense Prediction의 발전]

DDPM (Ho et al., 2020)
       │
Stable Diffusion / LDM (2022)
       │
       ├─── Marigold (CVPR 2024) ← 본 논문이 개선한 대상
       │    [Ke et al., SD 기반 depth estimation, 50 DDIM steps]
       │
       ├─── GeoWizard (2024)
       │    [Fu et al., depth + normal 동시 추정, decoupler module]
       │
       ├─── ★ 본 논문: E2E FT (WACV 2025)
       │    [trailing timestep fix + E2E fine-tuning]
       │
       ├─── BetterDepth (NeurIPS 2024)
       │    [plug-and-play diffusion refiner]
       │
       ├─── Marigold-DC (ICCV 2025)
       │    [zero-shot depth completion]
       │
       └─── Depth Pro (Apple, 2024)
            [sharp metric depth, < 1 second]
```

### 4.2 주요 논문 비교표

| 논문 | 년도 | 방법 | 속도 | 특징 |
|------|------|------|------|------|
| **Marigold** (Ke et al.) | CVPR 2024 | 50-step DDIM (leading) | 느림 | Zero-shot, SD 기반 |
| **GeoWizard** (Fu et al.) | 2024 | Multi-step diffusion | 느림 | Depth + Normal 동시 추정 |
| **★ 본 논문 (E2E FT)** | WACV 2025 | 1-step trailing + E2E FT | **200×+ 빠름** | 결정론적, SOTA |
| **BetterDepth** | NeurIPS 2024 | Plug-and-play refiner | 중간 | 기존 모델 보강 |
| **Marigold-DC** | ICCV 2025 | Test-time guidance | 중간 | Zero-shot depth completion |
| **Depth Any Camera** | 2024 | Non-standard imaging | 빠름 | Fisheye, 360° 카메라 지원 |

Marigold는 diffusion 기법을 depth estimation에 적용한 최초 모델 중 하나로, 전통적 discriminative 접근법보다 우월한 구조적 일관성과 엣지 충실도를 가진 출력을 생성한다. 반사성 또는 투명 객체가 포함된 도전적인 시나리오에서도 우수한 성능을 보이나, 복잡한 공간 배치의 다중 객체/다중 장면 구성에서는 어려움이 있다. GeoWizard는 훈련 중 scene distribution을 분리하는 decoupler 모듈을 도입하여 혼합 데이터로 인한 blurring과 ambiguity를 줄임으로써 Marigold를 개선했다.

대규모 text-to-image(T2I) pre-training 덕분에 최근 연구들은 T2I diffusion 모델을 dense perception 태스크에 단순 fine-tuning함으로써 유망한 결과를 보이고 있다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려 사항

### 5.1 연구에 미치는 영향

#### 📌 영향 1: DDIM Timestep Spacing의 재고찰
본 논문이 발견한 **trailing timestep 수정**은 이미 Marigold 공식 팀에 채택되었다. Garcia et al.이 DDIM에서 trailing timestep으로의 전환을 제안했고, 이 변경은 Marigold의 효율성을 크게 향상시켜 단 1 DDIM step에서 성능이 포화되었으며, 이는 모든 Marigold v1.1 모델의 기본 설정이 되었다.

이는 앞으로 diffusion 기반 dense prediction 연구에서 scheduler 설정의 세밀한 검토가 필수적임을 시사한다.

#### 📌 영향 2: 복잡한 Pre-training의 필요성 재검토

이 fine-tuning 프로토콜이 Stable Diffusion에 직접 적용해도 SOTA 성능에 도달함을 발견하여 이전 연구들의 일부 결론에 의문을 제기하며, 결과적으로 diffusion 모델을 geometry estimation에 활용하는 것은 end-to-end fine-tuning만큼 간단하다.

#### 📌 영향 3: 실시간 응용으로의 확장
Monocular depth estimation(MDE)은 이미지/비디오 편집, 장면 재구성, novel view synthesis, 로봇 내비게이션 등 다양한 응용에서 중요한 역할을 한다. 이 태스크는 내재적인 scale-distance ambiguity로 인해 ill-posed 문제라는 도전이 있다.

200배 이상의 속도 향상으로 실시간 로봇, 자율주행, AR/VR 등 실용적 응용이 가능해졌다.

---

### 5.2 앞으로의 연구 시 고려할 점

#### 🔬 고려 사항 1: Scheduler 세부 설정의 중요성
- Diffusion 기반 모델 사용 시 `timestep_spacing`(leading vs. trailing), `SNR` 설정 등을 세밀하게 검토해야 함
- Marigold v1.1은 zero-SNR 및 trailing timestamp을 포함한 업데이트된 noise scheduler 설정과 augmentation으로 훈련된 업데이트 체크포인트를 출시했다.

#### 🔬 고려 사항 2: E2E FT와 Metric Depth의 결합
- 본 논문의 모델은 **affine-invariant depth** 예측에 집중하나, 실제 응용에서는 절대적 거리(미터 단위)가 필요하다. Zero-shot metric depth로의 확장이 중요한 연구 방향이다.
- 예: defocus blur cue를 inference time에 Marigold에 주입하여 훈련 없이 Marigold를 metric depth predictor로 전환하는 연구가 이미 제안되었다.

#### 🔬 고려 사항 3: 더 강력한 베이스 모델의 활용
- SD 1.5 기반에서 **SDXL**, **SD3**, **Flux** 등 최신 대형 diffusion 모델로 E2E FT를 확장하면 일반화 성능이 더욱 향상될 가능성이 있다.

#### 🔬 고려 사항 4: 동영상/시계열 확장
- Rolling Depth (CVPR 2025)는 video depth estimation에서 우월한 시간적 일관성을 달성했다. 단일 이미지 모델을 시계열로 확장하는 것은 중요한 오픈 문제이다.

#### 🔬 고려 사항 5: 다양한 Dense Prediction Task로의 범용화
- 본 논문의 프로토콜이 depth와 normal에서 작동함을 보였으나, optical flow, semantic segmentation, intrinsic image decomposition 등으로의 확장 가능성을 체계적으로 검증할 필요가 있다.
- Marigold 팀은 이미 depth estimation, surface normal 예측, intrinsic decomposition을 포함한 dense image analysis 태스크에 적용되는 조건부 생성 모델 패밀리를 제시했다.

#### 🔬 고려 사항 6: 소규모 합성 데이터 학습의 한계 극복
- fine-tuned 모델은 Hypersim과 Virtual KITTI 2 데이터셋으로만 훈련된다. 더 다양한 실제 환경 데이터나 대규모 데이터로의 확장을 통해 일반화 성능을 더욱 향상시킬 수 있다.

---

## 📚 참고 자료 및 출처

| # | 제목 / 설명 | 출처 |
|---|------------|------|
| 1 | **주 논문** - "Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think" | [arXiv:2409.11355](https://arxiv.org/abs/2409.11355) |
| 2 | 논문 공식 Project Page | [gonzalomartingarcia.github.io/diffusion-e2e-ft](https://gonzalomartingarcia.github.io/diffusion-e2e-ft/) |
| 3 | 논문 GitHub 코드 저장소 | [github.com/VisualComputingInstitute/diffusion-e2e-ft](https://github.com/VisualComputingInstitute/diffusion-e2e-ft) |
| 4 | IEEE/CVF WACV 2025 논문 PDF | [openaccess.thecvf.com](https://openaccess.thecvf.com/content/WACV2025/papers/Garcia_Fine-Tuning_Image-Conditional_Diffusion_Models_is_Easier_than_You_Think_WACV_2025_paper.pdf) |
| 5 | HuggingFace Paper Page | [huggingface.co/papers/2409.11355](https://huggingface.co/papers/2409.11355) |
| 6 | **Marigold (기반 모델)** - "Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation" (CVPR 2024) | [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2024/papers/Ke_Repurposing_Diffusion-Based_Image_Generators_for_Monocular_Depth_Estimation_CVPR_2024_paper.pdf) |
| 7 | Marigold v1.1 HuggingFace 모델 및 Timestep 설정 | [huggingface.co/prs-eth/marigold-depth-v1-0](https://huggingface.co/prs-eth/marigold-depth-v1-0) |
| 8 | Marigold GitHub 공식 저장소 | [github.com/prs-eth/Marigold](https://github.com/prs-eth/Marigold) |
| 9 | **Marigold-DC** - "Zero-Shot Monocular Depth Completion with Guided Diffusion" (ICCV 2025) | [arXiv:2412.13389](https://arxiv.org/abs/2412.13389) |
| 10 | **Marigold 저널 확장판** - "Marigold: Affordable Adaptation of Diffusion-Based Image Generators for Image Analysis" (2025) | [arXiv:2505.09358](https://arxiv.org/html/2505.09358v1) |
| 11 | MarkTechPost 기사 - 논문 요약 및 해설 | [marktechpost.com](https://www.marktechpost.com/2024/09/25/simplifying-diffusion-models-fine-tuning-for-faster-and-more-accurate-depth-estimation/) |
| 12 | SummarizePaper - 논문 AI 요약 (참고용) | [summarizepaper.com](https://www.summarizepaper.com/en/arxiv-id/2409.11355v1/) |
| 13 | **Survey on Monocular Metric Depth Estimation** (2025) | [arXiv:2501.11841](https://arxiv.org/html/2501.11841v3) |
| 14 | **Repurposing Marigold for Zero-Shot Metric Depth via Defocus Blur** (NeurIPS 2025) | [arXiv:2505.17358](https://arxiv.org/abs/2505.17358) |
| 15 | 논문 Eindhoven University 연구 포털 | [research.tue.nl](https://research.tue.nl/en/publications/fine-tuning-image-conditional-diffusion-models-is-easier-than-you) |

---

> ⚠️ **정확도 주석**: 수식 중 E2E FT의 정규화 항($\mathcal{L}_\text{reg}$)과 일부 loss function의 정확한 형태는 논문 원문 PDF의 수식을 직접 확인하기를 권장합니다. 위에 제시된 loss 수식은 Marigold 계열 연구에서 널리 사용되는 affine-invariant depth loss 및 cosine normal loss의 일반적 표현이며, 본 논문에서도 이와 유사한 형태를 사용한다고 알려져 있으나, 세부 계수나 추가 항은 원문에서 확인이 필요합니다.
