
# Improved Distribution Matching Distillation for Fast Image Synthesis

> **저자:** Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Frédo Durand, William T. Freeman
> **소속:** MIT · Adobe Research
> **발표:** NeurIPS 2024 Oral 🔥
> **arXiv:** [2405.14867](https://arxiv.org/abs/2405.14867)
> **프로젝트 페이지:** [https://tianweiy.github.io/dmd2/](https://tianweiy.github.io/dmd2/)
> **GitHub:** [https://github.com/tianweiy/DMD2](https://github.com/tianweiy/DMD2)

---

## 1. 핵심 주장 및 주요 기여 요약

### 🎯 핵심 주장

Distribution Matching Distillation (DMD)은 teacher 모델의 샘플링 궤적과 1:1 대응을 강제하지 않고, 분포 수준에서 teacher와 일치하는 one-step generator를 생성하는 방법론입니다. 그러나 안정적인 훈련을 위해 DMD는 결정론적 샘플러의 다단계로 teacher가 생성한 대규모 noise-image pair 데이터셋을 필요로 하는 추가 회귀 손실(regression loss)이 필요하며, 이는 대규모 text-to-image 합성에서 비용이 매우 크고 student 품질을 teacher의 원래 샘플링 경로에 지나치게 종속시킨다는 문제가 있습니다.

DMD2는 이 한계를 극복하기 위한 일련의 기법들을 제안합니다.

### 🏆 주요 기여 4가지

DMD2는 이 한계를 극복하기 위한 기법들의 집합으로, 첫째로 회귀 손실과 고비용 데이터셋 구축의 필요성을 제거합니다.

훈련 불안정성이 "가짜(fake)" critic이 생성 샘플의 분포를 정확히 추정하지 못하기 때문임을 밝히고, 이를 해결하기 위해 **두 시간 척도 업데이트 규칙(two time-scale update rule)**을 제안합니다.

둘째, **GAN 손실(GAN loss)**을 증류 과정에 통합하여 생성 샘플과 실제 이미지를 구분합니다. 이를 통해 student 모델을 실제 데이터로 훈련하여 teacher 모델의 불완전한 실제 점수 추정을 완화하고 품질을 향상시킵니다.

셋째, multi-step 샘플링을 가능하게 하는 새로운 훈련 절차를 도입하며, 훈련 시 추론 시간의 generator 샘플을 시뮬레이션함으로써 이전 연구의 훈련-추론 입력 불일치 문제를 해결합니다.

---

## 2. 상세 분석

### 🔴 2-1. 해결하고자 하는 문제

**① 고비용 데이터 구축 문제**

DMD는 안정적인 훈련을 위해 추가적인 회귀 손실이 필요하며, 이를 위해 teacher 모델의 전체 샘플링 단계를 실행하여 수백만 개의 noise-image pair를 생성해야 합니다. 이는 특히 text-to-image 합성에서 매우 비용이 큽니다.

**② Student 품질의 상한선 문제**

회귀 손실은 student 품질이 teacher에 의해 상한이 결정되도록 만들어 DMD의 unpaired 분포 매칭 목적함수의 핵심적 이점을 무효화합니다.

**③ 기존 방법들의 품질 저하 문제**

수많은 증류 방법이 teacher 확산 모델을 효율적인 few-step student generator로 변환하기 위해 개발되었지만, student 모델이 teacher의 pairwise noise-to-image 매핑을 학습하는 손실로 훈련되어 그 행동을 완벽하게 모방하는 데 어려움을 겪고 품질이 저하되는 문제가 있습니다.

---

### 🔵 2-2. 제안하는 방법 (수식 포함)

#### ① 분포 매칭 목적함수 (Distribution Matching Objective)

DMD의 핵심 목적함수는 **역방향 KL 발산(Reverse KL Divergence)** 을 최소화하는 것입니다.

DMD는 근사 KL 발산(KL divergence)을 최소화함으로써 multi-step diffusion teacher를 few-step generator $G$로 증류합니다.

$$\mathcal{L}_{\text{DM}} = \mathbb{E}_{z \sim p_{\text{fake}}} \left[ \log p_{\text{fake}}(z) - \log p_{\text{real}}(z) \right] = D_{\text{KL}}(p_{\text{fake}} \| p_{\text{real}})$$

이를 score function을 이용해 미분하면, generator의 gradient는 다음과 같이 표현됩니다:

$$\nabla_\theta \mathcal{L}_{\text{DM}} = \mathbb{E}_{t, \epsilon} \left[ \omega(t) \left( \hat{\epsilon}_{\text{fake}}(x_t, t) - \hat{\epsilon}_{\text{real}}(x_t, t) \right) \frac{\partial G_\theta(z)}{\partial \theta} \right]$$

여기서:
- $\hat{\epsilon}_{\text{real}}$: teacher의 **실제** 점수 함수 (real score function)
- $\hat{\epsilon}_{\text{fake}}$: student의 생성 분포에 대한 **가짜** 점수 함수 (fake score function)
- $\omega(t)$: 타임스텝 가중치
- $x_t = \alpha_t G_\theta(z) + \sigma_t \epsilon$: noised generator output

#### ② 두 시간 척도 업데이트 규칙 (Two Time-Scale Update Rule, TTUR)

회귀 손실을 단순히 제거하면 불안정성이 발생합니다. 이를 해결하기 위해 DMD2는 **두 시간 척도 업데이트 규칙(TTUR)**을 사용하여, generator보다 fake score estimator를 더 자주 업데이트합니다.

구체적으로, fake critic을 $n_c$번 업데이트할 때마다 generator를 1번 업데이트:

$$\theta_{\text{fake}} \leftarrow \theta_{\text{fake}} - \eta_c \nabla_{\theta_{\text{fake}}} \mathcal{L}_{\text{fake}}$$

$$\theta_G \leftarrow \theta_G - \eta_G \nabla_{\theta_G} \left( \mathcal{L}_{\text{DM}} + \lambda \mathcal{L}_{\text{GAN}} \right)$$

#### ③ GAN 손실 통합

GAN 손실을 증류 절차에 통합하여 생성 샘플과 실제 이미지를 구분합니다. 이를 통해 student 모델을 실제 데이터로 훈련하여 teacher 모델의 불완전한 "실제" 점수 추정을 완화하고 품질을 향상시킵니다.

$$\mathcal{L}_{\text{GAN}} = \mathbb{E}_{x \sim p_{\text{real}}} \left[ \log D(x) \right] + \mathbb{E}_{z \sim p_z} \left[ \log (1 - D(G_\theta(z))) \right]$$

전체 목적함수:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{DM}} + \lambda_{\text{GAN}} \mathcal{L}_{\text{GAN}}$$

#### ④ Backward Simulation (역방향 시뮬레이션)

세 번째로, multi-step 샘플링을 가능하게 하는 새로운 훈련 절차를 도입하여, 훈련 시 추론 시간의 generator 샘플을 시뮬레이션함으로써 **훈련-추론 입력 불일치(training-inference input mismatch)** 문제를 해결합니다.

Multi-step 설정에서 $t$번째 스텝의 입력은:

$$x_t^{(k)} = G_\theta^{(k-1)}(z, x_{t+1}^{(k-1)})$$

훈련 시 실제 추론 과정의 중간 출력을 시뮬레이션함으로써 분포 불일치를 방지합니다.

---

### 🟢 2-3. 모델 구조

DMD2의 훈련은 두 단계를 반복합니다. (1) 분포 매칭 목적함수의 gradient와 GAN 손실을 이용하여 generator를 최적화하는 단계, (2) "가짜" 샘플의 분포를 모델링하는 점수 함수와 가짜 샘플과 실제 이미지를 구분하는 GAN 판별자(discriminator)를 훈련하는 단계. Student generator는 one-step 또는 multi-step 모델이 될 수 있습니다.

구조도 요약:

```
[Noise z] → [Student Generator G_θ] → [Fake Image x̂]
                        ↑                         ↓
              Gradient (DM + GAN)         [Fake Score Estimator]
                                                   ↓
[Real Images x] ─────────────────── [GAN Discriminator D]
```

이 통합된 접근 방식은 GAN만 사용(분포 매칭 목적함수 없이)하는 것보다 성능이 우수하며, GAN 단독에 TTUR을 추가해도 성능이 향상되지 않아 분포 매칭과 GAN을 통합 프레임워크로 결합하는 것의 효과를 강조합니다.

---

### 🔴 2-4. 성능 향상 및 한계

#### ✅ 성능 향상

이러한 개선들이 합쳐져 one-step 이미지 생성에서 새로운 기준을 세웠습니다. ImageNet-64×64에서 FID 1.28, zero-shot COCO 2014에서 8.35를 달성하여 추론 비용을 500배 줄이면서도 원래 teacher를 능가합니다.

또한 SDXL을 증류하여 megapixel 이미지 생성이 가능하며, few-step 방법들 중 탁월한 시각적 품질을 보여주고 teacher를 능가합니다.

4-step DMD2 generator는 COCO 2014에서 FID 19.32, Patch FID 20.86을 달성하며, LCM-SDXL, SDXL-Turbo, SDXL-Lightning 등의 최첨단 방법들을 능가하고, 훨씬 적은 추론 단계로 100-step SDXL teacher 모델의 FID 19.36에 필적합니다.

특히 teacher 대비 이미지 품질에서 24%의 샘플에서 teacher를 능가하고, 전진 패스를 25배 더 적게 사용(4 vs 100)하면서도 비슷한 프롬프트 정렬을 달성합니다.

텍스트-이미지 정렬 측면에서도 DMD2의 4-step generator는 CLIP 점수 0.332를 달성하여 SDXL teacher 모델과 비슷하며, teacher의 의미론적 이해를 효과적으로 보존함을 시사합니다.

#### ⚠️ 한계점

대부분의 이전 증류 방법처럼 훈련 중 고정된 guidance scale을 사용하여 사용자 유연성이 제한됩니다. 가변 guidance scale 도입이 향후 연구의 유망한 방향일 수 있습니다. 또한 분포 매칭에 최적화되어 있어, human feedback이나 다른 reward function을 통합하면 성능을 더 향상시킬 수 있습니다.

마지막으로, 대규모 생성 모델 훈련은 계산 집약적이어서 대부분의 연구자들이 접근하기 어렵습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 🌐 3-1. 실제 데이터 학습을 통한 일반화

GAN 손실을 통합함으로써 student 모델이 실제 데이터로 훈련될 수 있어, teacher 모델로부터의 불완전한 실제 점수 추정을 완화하고 품질을 향상시킵니다. 이는 teacher의 편향으로부터 자유로운 **더 넓은 데이터 분포 학습**을 가능하게 하여 일반화 성능을 높입니다.

### 🌐 3-2. 회귀 손실 제거를 통한 일반화

새로운 분포 매칭 증류 기법은 안정적인 훈련을 위한 회귀 손실을 필요로 하지 않으며, 이를 통해 비용이 많이 드는 데이터 수집의 필요성을 제거하고 **더 유연하고 확장 가능한 훈련**이 가능합니다.

회귀 손실 제거는 student 모델이 teacher의 특정 샘플링 경로에 종속되지 않도록 하므로, student는 teacher가 커버하지 못하는 **분포 영역도 탐색** 가능합니다.

### 🌐 3-3. 훈련-추론 불일치 해소를 통한 일반화

DMD2는 훈련 중 추론 시간의 generator 샘플을 시뮬레이션하여 이전 연구에서 만연했던 **훈련-추론 입력 불일치** 문제를 해결하는 새로운 multi-step 샘플링 훈련 절차를 도입합니다. 이는 실제 추론 환경과 훈련 환경의 gap을 줄여 **실전 일반화 성능**을 크게 향상시킵니다.

### 🌐 3-4. 다양한 아키텍처로의 확장성

SDXL로부터 증류하여 고품질 megapixel 이미지를 생성하는 접근법의 확장성을 입증하며, few-step 방법 중 새로운 기준을 확립합니다.

DMD 기반 알고리즘으로 증류된 합성 세트는 다양한 아키텍처와 태스크 도메인(신호/데이터/시간/주파수 도메인 등) 전반에 걸쳐 효과적으로 일반화됩니다.

### 🌐 3-5. 한계 - 일반화 관련 후속 연구에서 제기된 문제

후속 연구에서는 DMD2에서 귀인한 fake score estimator의 근사 오차만이 아니라, student와 teacher 분포 간 지원 집합(support sets)의 낮은 겹침이 더 근본적인 문제임을 지적합니다. 즉, score distillation은 특히 매우 적은 단계로 증류할 때 초기화에 대한 더 높은 요구 사항을 부과합니다.

---

## 4. 최신 관련 연구 비교 분석 (2020년 이후)

| 방법 | 발표 | 핵심 특징 | One-step FID (ImageNet-64) |
|------|------|-----------|--------------------------|
| DDPM | NeurIPS 2020 | 기초 확산 모델, 수백~수천 스텝 | - |
| Progressive Distillation | ICLR 2022 | 단계적으로 스텝 수 절반 감소 | ~5.0 |
| Consistency Models | ICML 2023 | Self-consistency 조건 부과 | ~3.55 |
| DMD (원본) | CVPR 2024 | 분포 매칭, 회귀손실 필요 | ~2.62 |
| **DMD2** | **NeurIPS 2024** | **회귀손실 제거 + GAN + Backward Sim** | **1.28** |
| ADM (DMDX) | 2025 | 적대적 분포 매칭, 더 적은 GPU | <1.28 (SDXL) |

### 주요 경쟁 방법과의 비교

**① Adversarial Diffusion Distillation (ADD / LADD)**

Latent Adversarial Diffusion Distillation (LADD)은 ADD의 한계를 극복하는 새로운 증류 접근 방식으로, 사전 훈련된 잠재 확산 모델의 생성적 특징을 활용하여 고해상도 다중 비율 이미지 합성을 가능하게 합니다.

**② ADM (Adversarial Distribution Matching)**

ADM은 적대적 방식으로 score distillation을 위해 real 및 fake score estimator 간 잠재 예측을 정렬하기 위해 확산 기반 판별자를 활용하는 새로운 프레임워크로, DMD2보다 더 적은 GPU 시간을 소비하면서 SDXL에서 우수한 one-step 성능을 달성합니다.

**③ CFG Augmentation 관점의 비교**

복잡한 text-to-image 생성과 같은 태스크에서 few-step distillation의 주요 동인은 분포 매칭이 아니라 CFG Augmentation(CA)이며, 분포 매칭(DM) 항은 훈련 안정성을 보장하고 완화하는 정규화 역할을 한다는 점이 밝혀졌습니다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려사항

### 📌 5-1. 미치는 영향

**① 실용적 확산 모델 배포의 새 기준 제시**

분포 매칭의 한계를 GAN 프레임워크 통합으로 극복하고, 'backward simulation'이라는 새로운 훈련 절차로 few-step 샘플링을 가능하게 했습니다. 이는 산업 현장에서의 실시간 이미지 생성 응용을 현실화합니다.

**② 증류 패러다임의 전환**

안정적인 훈련을 위한 회귀 손실이 불필요한 새로운 분포 매칭 증류 기법을 제안함으로써, 비용이 많이 드는 데이터 수집의 필요성을 제거하고 더 유연하고 확장 가능한 훈련을 가능하게 합니다.

**③ 비디오 및 3D 생성으로의 확장 가능성**

DMD 타입 파이프라인을 사용한 megapixel 이미지 및 비디오 합성이 실시간 프레임 속도로 시연되었으며, MagicDistillation의 4-step 비디오 합성은 28-step teacher를 능가합니다.

**④ 민주화된 연구 생태계 기여**

효율적인 접근 방식과 최적화된 사용자 친화적 코드베이스가 이 분야의 향후 연구를 민주화하는 데 도움이 되기를 기대합니다.

---

### 📌 5-2. 앞으로 연구 시 고려할 점

#### 🔬 기술적 고려사항

**1. 가변 Guidance Scale 지원**

대부분의 이전 증류 방법과 마찬가지로 훈련 중 고정 guidance scale을 사용하여 사용자 유연성이 제한됩니다. 가변 guidance scale 도입이 향후 연구의 유망한 방향이 될 수 있습니다.

**2. Human Feedback 및 Reward 통합**

분포 매칭에 최적화된 현재 방법에 human feedback이나 다른 reward function을 통합하면 성능을 더 향상시킬 수 있습니다.

**3. Support Set 겹침 문제 해결**

student와 teacher 분포 간 지원 집합의 낮은 겹침이 핵심 문제이며, score distillation은 특히 매우 적은 단계로 증류할 때 초기화에 대한 더 높은 요구 사항을 부과합니다. 이를 해결하는 초기화 전략 연구가 필요합니다.

**4. 계산 효율성**

대규모 생성 모델 훈련은 계산 집약적이어서 대부분의 연구자들이 접근하기 어렵습니다. 소규모 GPU 환경에서도 작동하는 경량화 버전 개발이 필요합니다.

#### 🔬 일반화 성능 관련 고려사항

- **도메인 전이(Domain Transfer):** 특정 도메인 데이터로 훈련된 student가 타 도메인에서의 성능 저하를 최소화하는 방안 연구 필요
- **Few-shot 환경:** 극도로 적은 실제 데이터만 가용한 상황에서의 GAN 판별자 안정적 훈련 방법 탐색
- **Reward 다양화:** 단순 FID 지표를 넘어 인간 선호도, 텍스트 정렬, 다양성을 균형 있게 최적화하는 복합 reward 설계

---

## 📚 참고 자료 및 출처

| # | 출처 | URL |
|---|------|-----|
| 1 | **arXiv 원문 (2405.14867)** | https://arxiv.org/abs/2405.14867 |
| 2 | **공식 프로젝트 페이지** | https://tianweiy.github.io/dmd2/ |
| 3 | **GitHub 공식 저장소** | https://github.com/tianweiy/DMD2 |
| 4 | **NeurIPS 2024 공식 논문 PDF** | https://proceedings.neurips.cc/paper_files/paper/2024/file/54dcf25318f9de5a7a01f0a4125c541e-Paper-Conference.pdf |
| 5 | **OpenReview (NeurIPS 2024 심사 내용)** | https://openreview.net/forum?id=tQukGCDaNT |
| 6 | **NeurIPS 2024 포스터 페이지** | https://neurips.cc/virtual/2024/poster/93335 |
| 7 | **ACM DL (NeurIPS 2024 논문집)** | https://dl.acm.org/doi/10.5555/3737916.3739421 |
| 8 | **Semantic Scholar** | https://www.semanticscholar.org/paper/...cc6fc3c546b354abf6a0aa3b553f28a6b812489f |
| 9 | **Hugging Face Papers 페이지** | https://huggingface.co/papers/2405.14867 |
| 10 | **arXiv HTML 전문** | https://arxiv.org/html/2405.14867v1 |
| 11 | **Liner.com 리뷰** | https://liner.com/review/improved-distribution-matching-distillation-for-fast-image-synthesis |
| 12 | **EmergentMind DMD2 분석** | https://www.emergentmind.com/papers/2405.14867 |
| 13 | **후속 연구: ADM/DMDX (arXiv:2507.18569)** | https://arxiv.org/html/2507.18569v1 |
| 14 | **관련 선행 연구: One-step DMD (CVPR 2024)** | Yin et al., "One-step Diffusion with Distribution Matching Distillation," CVPR 2024 |
| 15 | **관련 연구: ADD** | Sauer et al., "Adversarial Diffusion Distillation," arXiv:2311.17042 |
