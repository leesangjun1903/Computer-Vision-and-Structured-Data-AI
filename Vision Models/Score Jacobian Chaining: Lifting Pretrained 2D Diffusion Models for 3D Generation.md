
# Score Jacobian Chaining: Lifting Pretrained 2D Diffusion Models for 3D Generation

> **논문 정보**
> - **제목**: Score Jacobian Chaining: Lifting Pretrained 2D Diffusion Models for 3D Generation
> - **저자**: Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A. Yeh, Greg Shakhnarovich
> - **학회**: CVPR 2023 (arXiv: 2212.00774)
> - **GitHub**: https://github.com/pals-ttic/sjc
> - **프로젝트 페이지**: https://pals.ttic.edu/p/score-jacobian-chaining

---

## 1. 핵심 주장 및 주요 기여 (요약)

### 1.1 핵심 주장

Diffusion 모델은 그래디언트의 벡터 필드(vector field of gradients)를 예측하도록 학습된다. SJC는 학습된 그래디언트에 연쇄 법칙(chain rule)을 적용하여, Diffusion 모델의 score를 미분 가능한 렌더러(differentiable renderer)의 야코비안(Jacobian)을 통해 역전파(back-propagate)한다. 이 설정은 여러 카메라 시점에서의 2D score를 하나의 3D score로 집계(aggregate)하여, 사전 학습된 2D 모델을 3D 데이터 생성에 재활용(repurpose)한다.

### 1.2 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| ① Score Jacobian Chaining | 2D score를 3D로 끌어올리는 수학적 프레임워크 제안 |
| ② PAAS (Perturb-and-Average Scoring) | 분포 불일치(distribution mismatch) 문제 해결을 위한 새로운 추정 메커니즘 |
| ③ Off-the-shelf 모델 활용 | Stable Diffusion 등 공개된 2D 모델 그대로 활용 가능 |

이 논문은 분포 불일치(distribution mismatch)라는 기술적 도전 과제를 식별하고 이를 해결하기 위한 새로운 추정 메커니즘을 제안한다. 알고리즘을 Stable Diffusion 등 여러 off-the-shelf diffusion 이미지 생성 모델에서 실행하였다.

---

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

**핵심 문제**: 3D 학습 데이터의 부재 및 비용 문제

기존 3D 생성 모델들은 대규모 3D 데이터셋이 필요하거나, CLIP 기반 안내(guidance)를 사용하였다. CLIP 기반 최적화 3D 생성 모델들은 2D 렌더링을 기반으로 3D 자산을 최적화하는 유사한 철학을 공유한다. DreamFields와 PureClipNeRF는 NeRF를 미분 가능한 렌더러로 사용한다. 그러나 CLIP은 진정한 의미의 2D 생성 모델이 아니기 때문에, 실제 이미지와는 매우 다른 추상적 콘텐츠를 생성하는 경우가 많다. 이에 반해 SJC는 diffusion 모델을 활용하여 보다 현실적인 3D 결과물을 생성한다.

**기술적 도전**: 분포 불일치(Distribution Mismatch)

Voxel grid는 체적 렌더링(volumetric rendering)을 위한 매우 강력한 3D 표현이다. 그러나 노이즈가 있는 2D 가이던스 하에서 모델이 전체 그리드에 작은 밀도(small densities)를 채워 특정 시점에서 그럴듯한 이미지를 환각(hallucinate)하는 방식으로 속임수를 쓸 수 있다.

또한, FFHQ에서 사전 학습된 denoiser에 볼록(blob) 형태의 입력을 직접 적용하면, 모델이 얼굴 이미지로 올바르게 보정하지 못하는 OOD(Out-of-Distribution) 문제가 발생한다. 반면 노이즈가 추가된 입력($x_{blob} + \sigma n$)을 평가하면 blob과 얼굴 매니폴드가 성공적으로 합쳐진 결과를 얻는다.

---

### 2.2 제안 방법 (수식 포함)

#### ① 핵심 공식: Score Jacobian Chaining

SJC의 핵심은 2D score를 렌더러의 야코비안을 통해 역전파하여 3D score를 도출하는 것이다.

$$
\underbrace{\nabla_{\boldsymbol{\theta}}\log q_\sigma(\boldsymbol{\theta})}_{\text{3D score}} = \mathbb{E}_{\pi} \left[ \underbrace{\nabla_{\boldsymbol{x}_\pi}\log p_\sigma(\boldsymbol{x}_\pi)}_{\text{2D score (pretrained)}} \cdot \underbrace{J_\pi}_{\text{renderer Jacobian}} \right]
$$

위 수식에서 3D score $\nabla_{\boldsymbol{\theta}}\log q_\sigma(\boldsymbol{\theta})$는 카메라 시점 $\pi$에 대한 기댓값으로, 사전 학습된 2D score $\nabla_{\boldsymbol{x}\_\pi}\log p_\sigma(\boldsymbol{x}\_\pi)$와 렌더러의 야코비안 $J_\pi$의 곱으로 표현된다.

- $\boldsymbol{\theta}$: 3D 자산 파라미터 (voxel radiance field)
- $\boldsymbol{x}_\pi$: 시점 $\pi$에서 렌더링된 2D 이미지
- $p_\sigma(\boldsymbol{x}_\pi)$: 사전 학습된 2D diffusion 모델의 분포
- $q_\sigma(\boldsymbol{\theta})$: 3D 자산 공간의 분포
- $J_\pi = \frac{\partial \boldsymbol{x}_\pi}{\partial \boldsymbol{\theta}}$: 미분 가능한 렌더러의 야코비안

#### ② Score 정의

Hyvärinen의 정의에 따라 score는 데이터에 대한 로그 밀도 함수의 그래디언트로 정의된다. 다양한 패밀리의 diffusion 모델은 노이즈 레벨 $\sigma$에서의 denoising score $\nabla_x \log p_\sigma(x)$를 모델링하는 것으로 해석할 수 있다.

Denoising score의 실용적인 추정 공식:

$$
\text{score}(\boldsymbol{x}_\pi, \sigma) \triangleq \frac{D(\boldsymbol{x}_\pi, \sigma) - \boldsymbol{x}_\pi}{\sigma^2}
$$

여기서 $D(\boldsymbol{x}_\pi, \sigma)$는 denoiser 신경망의 출력이다.

Diffusion 모델에서 샘플 생성은 $\sigma$가 큰 값에서 작은 값으로 반복적으로 score 함수를 평가하여, 샘플 $x$가 점차 데이터 매니폴드에 가까워지게 하는 과정을 포함한다.

#### ③ PAAS (Perturb-and-Average Scoring)

분포 불일치 문제를 해결하기 위해 제안된 핵심 메커니즘이다.

SJC는 Perturb-and-Average Scoring (PAAS)이라 불리는 score 추정 방법을 직접 사용하는 새로운 접근법을 제시한다. 이 작업은 DreamFusion에서 등장하는 U-Net 야코비안이 불필요하다는 점을 보여주며, 공개된 Stable Diffusion을 사용하여 강력한 기준선을 구성한다. PAAS의 perturb-and-average score는 팽창된 노이즈 레벨을 가진 score에 근사하며, 기댓값은 실제로 몬테 카를로 샘플링으로 추정된다.

PAAS 추정:

$$
\nabla_{\boldsymbol{x}_\pi} \log p_\sigma(\boldsymbol{x}_\pi) \approx \mathbb{E}_{\boldsymbol{n}} \left[ \frac{D(\boldsymbol{x}_\pi + \sigma\boldsymbol{n},\ \sigma) - (\boldsymbol{x}_\pi + \sigma\boldsymbol{n})}{\sigma^2} \right]
$$

여기서 $\boldsymbol{n} \sim \mathcal{N}(0, I)$는 추가 가우시안 노이즈이다.

---

### 2.3 모델 구조

SJC는 학습된 그래디언트에 연쇄 법칙을 적용하여, Diffusion 모델의 score를 미분 가능한 렌더러의 야코비안을 통해 역전파하며, 이를 voxel radiance field로 구체화한다. 이 설정은 여러 카메라 시점에서의 2D score를 3D score로 집계한다.

전체 파이프라인은 다음과 같다:

```
[텍스트 프롬프트]
     ↓
[Stable Diffusion (고정, 사전학습된 2D 모델)]
     ↓  2D Score 추정 (PAAS)
[∇_{x_π} log p_σ(x_π)]
     ↓  × Jacobian of Differentiable Renderer (J_π)
[∇_θ log q_σ(θ)] ← 3D Score
     ↓  최적화
[Voxel Radiance Field (3D 표현)]
     ↓
[다중 시점 렌더링 결과]
```

**주요 컴포넌트:**

| 컴포넌트 | 역할 |
|--------|------|
| 사전학습된 2D Diffusion 모델 (Stable Diffusion v1.5) | 2D score 제공 |
| 미분 가능한 렌더러 (Differentiable Renderer) | 3D → 2D 변환 및 야코비안 계산 |
| Voxel Radiance Field | 3D 장면 표현 |
| PAAS | OOD 문제 해결을 위한 score 추정 |
| 정규화 전략 | 빈 공간(emptiness), 깊이(depth) 등 |

SJC repo에는 SJC 외에도 Karras sampler 구현체와 커스터마이즈된 간단한 voxel NeRF도 포함되어 있다.

---

### 2.4 성능

SJC 알고리즘을 Stable Diffusion(LAION 5B 데이터셋으로 학습된)을 포함한 여러 off-the-shelf 이미지 생성 diffusion 모델에서 실행하였다.

DreamFusion(Poole et al.)은 SJC와 독립적으로 동시에 진행된 연구로, 의사코드(pseudo-code) 수준에서 유사한 알고리즘을 제안하였다. 그러나 DreamFusion은 diffusion 모델의 학습 손실을 최소화하는 이미지 파라미터화를 탐색하는 수학적 설정을 사용하는 반면, SJC는 2D score에 연쇄 법칙을 적용하는 데 초점을 맞춘다.

SJC는 diffusion 모델을 denoiser로 해석하는 관점에서 PAAS라는 score 추정 방법을 직접 사용하며, DreamFusion에서 등장하는 U-Net 야코비안이 불필요하다는 점을 보여주고, 공개된 Stable Diffusion을 활용하여 강력한 기준 모델(baseline)을 형성한다.

---

### 2.5 한계점

#### ① 야누스 문제 (Janus Problem)

기존 score distillation 기반 텍스트-3D 생성 기법들은 상당한 가능성에도 불구하고 시점 불일치(view inconsistency) 문제에 자주 직면한다. 가장 두드러진 문제는 야누스 문제(Janus problem)로, 물체의 정면 시점(예: 얼굴이나 머리)이 다른 시점에서도 나타나는 현상이다.

실험에서 기준선(SJC)은 야누스 문제를 그대로 보여주며, 360° 시점 전반에서 얼굴이 모든 방향에서 나타난다.

#### ② 과채화(Over-saturation) 및 다양성 부족

DreamFusion은 언어 조건화된 diffusion 모델이 비정상적으로 높은 guidance scale을 활용하여 이미지 분포를 좁힘으로써 최적화를 더 쉽게 만든다는 통찰을 보여준다. SJC는 이 통찰에 영향을 받았으나, 이 방식은 과채화된 색상(over-saturated colors)과 언어 프롬프트당 제한된 콘텐츠 다양성이라는 단점이 있으며, 조건 없는 diffusion 모델(unconditioned diffusion model)에 어떻게 적용할지는 현재 불분명하다.

#### ③ 비조건적(Unconditioned) 모델 적용의 어려움

초기에는 Dhariwal과 Nicol의 비조건적 diffusion 모델인 LSUN Bedroom 모델에서 작업하였으나, PAAS를 비조건적 diffusion 모델에서 작동시키는 것은 매우 어려운 것으로 밝혀졌다.

#### ④ Voxel Grid의 아티팩트

Voxel grid는 체적 렌더링을 위한 강력한 3D 표현이지만, 노이즈가 있는 2D 가이던스 하에서 모델이 전체 그리드에 작은 밀도(small densities)를 채워 특정 시점에서 그럴듯한 이미지를 환각(hallucinate)하는 방식으로 속임수를 쓸 수 있다.

---

## 3. 일반화 성능 향상 가능성

### 3.1 Off-the-shelf 2D 모델의 재사용성

SJC는 여러 off-the-shelf diffusion 이미지 생성 모델에서 실행 가능하며, 특히 대규모 LAION 데이터셋으로 학습된 Stable Diffusion에서도 작동한다. 이는 SJC가 특정 모델 구조에 종속되지 않고, 어떤 사전학습된 2D diffusion 모델도 3D 생성에 활용 가능하다는 것을 의미한다.

### 3.2 DreamBooth 등 파인튜닝된 모델과의 통합

DreamBooth로 파인튜닝된 모델과 SJC를 결합하면 모델 출력 분포가 이미 좁혀져 있으므로 낮은 guidance scale을 사용하는 것이 도움이 될 수 있다. 그러나 지나친 mode-seeking은 멀티-페이스 문제(multi-face problem)의 원인 중 하나이며, view-dependent prompt 파인튜닝을 통한 DreamBooth 통합을 시도하였으나 아직 완전히 준비되지 않은 상태이다.

### 3.3 다양한 도메인 적용 가능성

얼굴처럼 정렬된 단순한 도메인에서는 단일 $\sigma=1.5$를 사용한 간단한 스케줄링으로도 좋은 이미지를 생성할 수 있다. 그러나 침실(bedroom)과 같이 다양성이 높은 도메인에서는 어닐링(annealing)이 여전히 필요하다.

즉, **단순하고 정렬된 도메인**에서는 높은 일반화 성능을 보이지만, **복잡하고 다양한 도메인**에서는 추가 기법이 필요하다.

### 3.4 Score Distillation 프레임워크로의 일반화

SJC 및 DreamFusion 등 이전 연구의 3D 장면 파라미터에 관한 그래디언트 가정을 일반화하고 확장하면, 추정된 score 내의 문제 원인을 보다 잘 파악할 수 있다. score는 비조건적 score와 포즈-프롬프트 그래디언트로 더 분리될 수 있으며, 이 둘 모두 3D 장면에 대한 편향 없는 그래디언트 추정을 방해한다.

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 연구에 미치는 영향

#### ① Score Distillation 패러다임의 확립

SJC와 DreamFusion은 "2D diffusion model → 3D 생성" 패러다임을 확립한 선구적 연구이다.

최근 zero-shot 텍스트-3D 생성 분야에서는 score distillation 기법과 diffusion 모델을 neural radiance field 최적화에 통합하는 방향으로 상당한 발전이 이루어졌다. 이러한 방법들은 3D 감독(supervision) 없이도 텍스트 입력으로부터 다양한 3D 객체를 생성할 수 있는 솔루션을 제공한다.

#### ② 후속 연구 촉발

SJC는 이후 다수의 후속 연구에 영향을 주었다:

- **Magic3D** (Lin et al., 2022): 고해상도 텍스트-3D 콘텐츠 생성으로 확장
- **ProlificDreamer** (Wang et al., 2023): Variational Score Distillation(VSD)을 통해 다양성과 품질 개선
- **Debiasing Scores and Prompts** (NeurIPS 2023): SJC의 야누스 문제 해결을 목표로 score/prompt debiasing 제안
- **MVDream**: 다중 뷰 일관성을 위한 확장

최근 텍스트-3D 생성의 발전은 주로 고품질 3D 데이터셋의 희소성을 감안하여 사전 학습된 2D diffusion 모델을 활용하는 방향으로 이루어졌다. DreamFusion과 SJC와 같은 연구에서 일반적인 전략은 SDS(Score Distillation Sampling)를 사용하여 NeRF와 같은 3D 표현을 최적화하는 것이다.

### 4.2 향후 연구 시 고려할 점

#### ① 야누스 문제 해결

야누스 문제는 Magic3D, SJC, DreamFusion, Latent-NeRF 등 방법들의 적용 가능성을 제한하지만, 이전 문헌에서는 거의 공식화되거나 신중하게 분석되지 않았다. 향후 연구는 3D 시점 인식(view-aware) 학습이나 멀티뷰 일관성 강화를 통해 이를 극복해야 한다.

debiasing 방법들이 야누스 문제를 효과적으로 해결하더라도, 일부 프롬프트에 대한 결과는 여전히 완벽하지 않다. 이는 주로 Stable Diffusion의 view-conditioned 프롬프트에 대한 제한적인 이해 때문이다.

#### ② 비조건적 모델 및 복잡한 도메인 적용

언어 조건화 모델에서의 강점과 달리, 비조건적 diffusion 모델에서의 적용은 여전히 미해결 과제이다. 향후 연구는 다양하고 복잡한 도메인(침실, 야외 장면 등)에서의 일반화 성능을 향상시켜야 한다.

#### ③ 3D 표현 개선

Voxel grid 대신 **3D Gaussian Splatting**, **Instant-NGP**, **DMTet** 등의 표현을 활용하면 계산 효율성과 품질을 동시에 개선할 수 있다. Magic3D 등이 이미 coarse-to-fine 전략으로 이를 시도하고 있다.

#### ④ 계산 비용 절감

SJC 기반 실험은 단일 NVIDIA 3090 RTX GPU로 10,000 스텝 최적화에 약 20분이 소요된다. 대규모 활용을 위해 최적화 속도 개선이 필요하다.

#### ⑤ 멀티모달 조건 확장

텍스트뿐 아니라 이미지, 스케치, 포인트 클라우드 등 다양한 조건 입력으로 확장하여 활용 범위를 넓힐 수 있다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 특징 | SJC 대비 차이점 |
|------|------|------|----------------|
| **NeRF** (Mildenhall et al.) | 2020 | 암시적 3D 표현 | 3D GT 데이터 필요 |
| **CLIP-NeRF / DreamFields** | 2022 | CLIP 기반 2D 가이던스 | 생성 품질이 추상적 |
| **DreamFusion** (Poole et al.) | 2022 | SDS 기반 2D→3D | Imagen(비공개) 사용, U-Net Jacobian 활용 |
| **SJC (본 논문)** | 2022/2023 | Chain rule 기반 2D→3D | Stable Diffusion(공개) 사용, PAAS 제안 |
| **Magic3D** (Lin et al.) | 2022 | Coarse-to-fine 2단계 | 고해상도, DMTet 활용 |
| **Latent-NeRF** (Metzer et al.) | 2022 | Latent space에서 최적화 | 메모리 효율성 향상 |
| **ProlificDreamer** | 2023 | Variational Score Distillation | 다양성 및 고품질 동시 달성 |
| **MVDream** | 2023 | 멀티뷰 diffusion | 3D 일관성 강화 |
| **Debiasing SJC** | 2023 | Score/Prompt Debiasing | SJC의 야누스 문제 해결 |

DreamFusion의 SDS와 SJC의 PAAS는 수학적 유도 방식은 다르지만, 서로 다른 가중치 규칙과 샘플러를 사용한다는 차이를 제외하면 실질적으로 동일한 추정 결과를 도출한다.

---

## 참고 자료 및 출처

| 번호 | 자료 |
|------|------|
| 1 | **Wang, H. et al.** (2022/2023). *Score Jacobian Chaining: Lifting Pretrained 2D Diffusion Models for 3D Generation*. CVPR 2023. arXiv:2212.00774. https://arxiv.org/abs/2212.00774 |
| 2 | **GitHub 공식 저장소**: https://github.com/pals-ttic/sjc |
| 3 | **프로젝트 페이지**: https://pals.ttic.edu/p/score-jacobian-chaining |
| 4 | **CVPR 2023 Open Access**: https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Score_Jacobian_Chaining_Lifting_Pretrained_2D_Diffusion_Models_for_3D_CVPR_2023_paper.pdf |
| 5 | **IEEE Xplore**: https://ieeexplore.ieee.org/document/10203874 |
| 6 | **CVPR 2023 Poster 페이지**: https://cvpr.thecvf.com/virtual/2023/poster/22504 |
| 7 | **Debiasing Scores and Prompts** (NeurIPS 2023). arXiv:2303.15413. https://arxiv.org/abs/2303.15413 |
| 8 | **ar5iv (2303.15413)**: https://ar5iv.labs.arxiv.org/html/2303.15413 |
| 9 | **OpenReview (Debiasing)**: https://openreview.net/forum?id=jgIrJeHHlz |
| 10 | **DreamCS 비교 연구**: https://arxiv.org/html/2506.09814 |
