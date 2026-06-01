
# Generative Physical AI in Vision: A Survey

> **⚠️ 중요 전제**: 이 논문(arXiv: 2501.10928)은 **서베이(Survey/리뷰) 논문**입니다. 따라서 단일 수식·단일 모델 구조·특정 벤치마크 수치를 제안하는 논문이 아니며, 기존 연구를 **체계적으로 분류·분석·종합**하는 것이 핵심 기여입니다. 수식은 서베이가 다루는 **대표 패러다임의 개념적 수식**으로 표현합니다.

---

## 1. 📌 핵심 주장 및 주요 기여 (간결 요약)

이 논문은 Daochang Liu 외 7인이 저술한 서베이로, Generative AI가 컴퓨터 비전 분야를 급격히 발전시켜 왔음을 전제로 한다.

전통적인 생성 모델들은 주로 **시각적 충실도(visual fidelity)**에 초점을 맞추면서, 생성된 콘텐츠의 **물리적 타당성(physical plausibility)**을 종종 무시해 왔으며, 이는 로보틱스·자율 시스템·과학 시뮬레이션 등 실세계 물리 법칙 준수가 요구되는 응용에서 효과를 제한해 왔다.

### 핵심 주장 요약

| 구분 | 내용 |
|------|------|
| **핵심 문제 의식** | 생성 AI ≠ 물리적으로 타당한 AI |
| **핵심 목표** | 물리 인식 생성(Physics-Aware Generation, PAG)의 체계적 분류 |
| **분류 축** | PAG-E (명시적 시뮬레이션) vs PAG-I (암묵적 학습) |
| **응용 범위** | 이미지·비디오·3D·4D 생성, 로보틱스, 자율주행, 과학 시뮬레이션 |

### 주요 기여

이 서베이는 물리 지식의 통합 방식(명시적 시뮬레이션 또는 암묵적 학습)에 따라 방법론을 분류하는 체계적 리뷰를 제공하며, 핵심 패러다임 분석·평가 프로토콜 논의·미래 연구 방향 제시를 통해 물리적으로 근거 있는 비전 생성의 발전에 기여하고자 한다.

리뷰된 논문 목록은 [https://github.com/BestJunYu/Awesome-Physics-aware-Generation](https://github.com/BestJunYu/Awesome-Physics-aware-Generation)에 정리되어 있다.

---

## 2. 🔬 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2-1. 해결하고자 하는 문제

물리 법칙 준수가 요구되는 로보틱스·자율 시스템·과학 시뮬레이션에서의 생성 모델 한계를 극복하고, 생성 모델이 물리적 사실성과 동적 시뮬레이션을 통합하여 "세계 시뮬레이터(World Simulator)"로 기능하도록 한다.

기존 모델은 시각적 품질만을 최적화하여 부자연스러운 변형, 불안정한 동작, 비일관적인 객체 상호작용 같은 아티팩트가 발생하며, 물리 제약 없이 훈련된 대부분의 생성 모델은 재료 특성·객체 역학·힘 상호작용을 제대로 포착하지 못한다.

---

### 2-2. 제안하는 방법 및 분류 체계

서베이는 **물리 인식 생성(PAG)**을 실세계 물리에 대한 강력한 이해를 바탕으로 한 생성 과정으로 정의하고, 이를 **명시적 물리 시뮬레이션을 사용하는 PAG-E**와 **암묵적 학습을 사용하는 PAG-I**로 나누며, 이 구분은 생성 모델이 물리 시뮬레이션 모델을 명시적으로 활용하는지 여부에 따라 이루어진다.

#### 🔷 PAG-E: 명시적 물리 시뮬레이션 기반 생성

물리 법칙(뉴턴 역학, 유체 역학 등)을 시뮬레이터로 명시적으로 모델링하여 생성 과정에 결합하는 방식이다.

대표적인 개념적 목적함수:

$$\mathcal{L}_{\text{PAG-E}} = \mathcal{L}_{\text{gen}}(\hat{x}, x) + \lambda \cdot \mathcal{L}_{\text{phys}}(\hat{x}, \phi)$$

- $\mathcal{L}_{\text{gen}}$: 시각적 재구성 손실 (예: Diffusion denoising loss)
- $\mathcal{L}_{\text{phys}}$: 물리 시뮬레이터 $\phi$ 에 의해 정의된 물리 일관성 손실
- $\lambda$: 물리 제약 가중치

명시적 방법의 예로, PIN-WM과 같은 모델은 미분 가능한 강체(rigid-body) 동역학을 계산 그래프에 직접 내장하여, 최적화 탐색을 물리적으로 해석 가능한 수량 공간으로 제한하며, 이는 순수 생성 모델에서 전형적으로 나타나는 "단축 학습(shortcut learning)"을 방지하고 네트워크가 텍스처 통계가 아닌 역학을 해결하도록 강제한다.

#### 🔷 PAG-I: 암묵적 물리 학습 기반 생성

대규모 비디오/이미지 데이터로부터 물리 법칙을 암묵적으로 학습하는 방식이다.

$$p_\theta(\mathbf{x}_{0:T}) = \int p(\mathbf{x}_T) \prod_{t=1}^{T} p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) \, d\mathbf{x}_{1:T}$$

여기서 Diffusion Model은 다음의 denoising 과정을 학습한다:

$$\mathcal{L}_{\text{DM}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right\|^2 \right]$$

- $\boldsymbol{\epsilon}$: 가우시안 노이즈
- $\boldsymbol{\epsilon}_\theta$: denoising 신경망
- $t$: 타임스텝

이러한 모델들은 대규모 인터넷 비디오 데이터로 스케일업되어 훈련되면서, 데이터에 내재된 특정 물리 역학과 인과 관계를 암묵적으로 포착·재현하는 능력을 보여주었다.

비디오는 생성 AI에서 중추적인 역할을 담당하는데, 온라인에 방대하게 존재하는 비디오 데이터가 실세계 정보의 풍부한 저장소를 포함하기 때문이다. 이러한 맥락에서 비디오는 세계의 암묵적 물리 모델로 간주될 수 있으며, 자율주행·과학 시뮬레이션·로보틱스·구현 지능 등 다양한 다운스트림 태스크를 가능하게 한다.

---

### 2-3. 모델 구조 (다루는 주요 생성 모델)

이 서베이가 다루는 생성 모델로는 Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), Diffusion Models (DMs), Neural Radiance Fields (NeRFs), Gaussian Splatting (GS), Visual Autoregressive Models (VARs) 등이 포함되며, 이들은 시각 데이터의 기저 분포를 포착하기 위해 점점 강력한 아키텍처를 활용해 생성 학습의 경계를 계속 확장해 왔다.

서베이는 GAN, Diffusion Model, NeRF, Gaussian Splatting을 포함하여 광범위하게 다루며, 이 모델들은 이미지·비디오·3D·4D·인터랙티브 환경 등 다양한 데이터 모달리티와 일반 데이터·인간 중심 데이터·실내 장면 등 다양한 도메인을 처리한다.

이 중 Diffusion Model이 특히 주목할 만한데, 학습된 노이즈 제거 과정을 통해 랜덤 노이즈를 반복적으로 정제함으로써 예외적인 강건성과 다목적성을 보이며 최근 생성 방법론의 핵심 축이 되었다.

#### GAN의 목적함수:

$$\min_G \max_D \; \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} [\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z} [\log(1 - D(G(\mathbf{z})))]$$

GAN은 생성기(Generator)와 판별기(Discriminator)라는 두 신경망이 경쟁적 과정에 참여하는 구조로, 생성기는 실세계 데이터와 유사한 합성 데이터를 만들고 판별기는 실제와 생성 데이터를 구별하려 한다.

---

### 2-4. 성능 향상 및 한계

#### 성능 향상

이 서베이가 보여주는 진보는 기존 생성 태스크의 의미론적·시간적·공간적 이해에서 상호작용성과 물리적 사실성을 포함하는 방향으로 발전하는 것이며, 물리 인식 생성은 모델이 실세계 역학을 시뮬레이션하게 하여 다양한 응용을 위한 범용 세계 모델을 향해 나아간다.

물리적으로 현실적인 합성 데이터를 생성함으로써 로봇은 시뮬레이션에서 더 효과적으로 훈련되고 실세계 운용으로 원활하게 전환될 수 있다.

#### 한계

현재 모델들은 특히 대규모 또는 복잡한 역학을 다룰 때 물리적으로 일관된 결과를 생성하는 데 여전히 어려움을 겪고 있으며, 이는 주로 기존 접근법이 물리적 프롬프트에 등방적(isotropically)으로 반응하고 생성 콘텐츠와 국소화된 물리 단서 간의 세밀한 정렬을 무시하기 때문이다.

대부분의 생성 모델은 명시적 물리 제약이 없는 데이터셋으로 훈련되어 재료 특성, 객체 역학, 힘 상호작용을 포착하는 데 실패한다.

4D 동적 생성의 경우 기존 모델링 기법이 여전히 미성숙하여 복잡한 객체 변형과 시간적 일관성을 포착하는 능력을 개선하기 위한 추가 탐구가 필요하다.

---

## 3. 🚀 모델의 일반화 성능 향상 가능성

물리적으로 현실적인 합성 데이터를 통해 로봇을 시뮬레이션에서 더 효과적으로 훈련하고 실세계로 원활히 전환할 수 있으며, 특히 Vision-Language-Action 모델에 물리적 추론 능력을 명시적으로 주입하면 물리 지식을 활용해 복잡하고 다양한 환경에서 행동과 결과를 더 잘 예측하고 더 잘 일반화할 수 있다.

### 일반화 성능 향상을 위한 핵심 전략

#### ① Sim-to-Real Transfer (시뮬레이션→실세계 전이)

미래 연구는 생성 모델에서 물리 인식 사전(physics-aware priors)의 통합을 강화하고 다양한 재료와 동적 환경에 대한 적응성을 향상시키는 데 초점을 맞춰야 하며, Sim2Real 전이와 구현된 AI(Embodied AI)를 발전시켜 시뮬레이션과 실세계 상호작용의 격차를 더욱 좁혀야 한다.

수식으로 표현하면, Sim-to-Real 목적함수는 다음과 같이 표현할 수 있다:

$$\mathcal{L}_{\text{S2R}} = \mathcal{L}_{\text{task}}(\pi_\theta, \mathcal{E}_{\text{sim}}) + \alpha \cdot d(\mathcal{E}_{\text{sim}}, \mathcal{E}_{\text{real}})$$

- $\pi_\theta$: 정책 (policy)
- $\mathcal{E}\_{\text{sim}}, \mathcal{E}_{\text{real}}$: 시뮬레이션 및 실세계 환경 분포
- $d(\cdot)$: 도메인 격차 측정치 (domain gap measure)

#### ② Physics-Informed Regularization (물리 정보 기반 정규화)

물리적 기반은 감각 모달리티의 교차 일관성을 통해 암묵적으로도 강제될 수 있는데, 예컨대 비디오·깊이·키포인트 역학을 공동 최적화하면 정규화기(regularizer)로 작용한다.

물리 정규화 목적함수:

$$\mathcal{L}_{\text{phys-reg}} = \underbrace{\mathcal{L}_{\text{data}}}_{\text{데이터 적합}} + \lambda_1 \underbrace{\left\| \frac{\partial^2 \mathbf{u}}{\partial t^2} - \mathbf{f}(\mathbf{u}, \nabla \mathbf{u}) \right\|^2}_{\text{PDE 잔차 (물리 제약)}}$$

- $\mathbf{u}$: 생성된 물리량 (속도, 변형 등)
- $\mathbf{f}$: 물리 법칙 함수 (Navier-Stokes 등)

#### ③ Zero-Shot / Cross-Domain 일반화

PhysWorld와 같은 접근법은 단일 이미지와 태스크 명령으로 태스크 조건부 비디오를 생성하고 물리 세계를 재구성하여, 실제 로봇 데이터 수집 없이도 제로샷(zero-shot) 일반화 로봇 조작을 가능하게 한다.

---

## 4. 🔮 앞으로의 연구에 미치는 영향 및 고려 사항

### 4-1. 향후 연구에 미치는 영향

이 서베이는 물리 인식 생성 AI 비전 분야를 포괄적으로 검토하여 생성된 시각 콘텐츠의 물리적 사실성과 기능을 향상시키기 위한 노력을 부각시키며, 생성 모델에 물리 시뮬레이션을 통합하는 것을 핵심으로 다루고, 물리 인식 생성이 가상과 물리 현실 사이의 간극을 연결하는 변혁적 분기점에 있음을 강조한다.

생성 AI가 물리적 사실성과 동적 시뮬레이션을 점점 통합하면서 "세계 시뮬레이터"로 기능할 잠재력이 확장되어, 물리에 의해 지배되는 상호작용의 모델링과 가상 및 물리 현실 사이의 간극 연결을 가능하게 한다.

### 4-2. 향후 연구 시 고려할 점

#### ① 평가 지표(Evaluation Protocol) 표준화
단순한 FID, PSNR 등 시각적 지표를 넘어, 물리 법칙 준수도를 정량적으로 측정하는 새로운 벤치마크 설계가 필요하다.

$$\text{PhysScore} = \frac{1}{N}\sum_{i=1}^N \mathbf{1}\left[\left\|\mathbf{r}_i^{\text{gen}} - \mathbf{r}_i^{\text{sim}}\right\|_2 < \epsilon\right]$$

- $\mathbf{r}_i^{\text{gen}}$: 생성된 물리 궤적
- $\mathbf{r}_i^{\text{sim}}$: 물리 시뮬레이터 기준 궤적

#### ② 4D 생성의 성숙화
물리 기반 생성 모델을 정적 3D 생성, 동적 3D 생성, 4D 생성으로 분류하여 발전시켜야 하며, 시각 기반·NeRF 기반·GS 기반 동적 3D 생성 접근법을 포괄하는 체계적 관점을 확립해야 한다.

#### ③ World Model과의 통합
불확실성 인식 상상(uncertainty-aware imagination)을 통해 학습함으로써 에이전트는 실세계 시행착오의 비용이나 위험 없이 전이 가능한 기술을 습득할 수 있다.

#### ④ 고위험 도메인 적용 시 신뢰성
의료 분야와 같이 반사실적 추론이 필수적이고 "시각적 환각(visual hallucinations)"이 용납되지 않는 환경에서는 세계 모델링이 단순한 향상이 아니라 자율 의사결정의 전제 조건임을 보여준다.

---

## 5. 📊 2020년 이후 관련 최신 연구 비교 분석

| 연구 / 모델 | 연도 | 핵심 기여 | 물리 통합 방식 |
|---|---|---|---|
| **NeRF** (Mildenhall et al.) | 2020 | 볼륨 렌더링 기반 뷰 합성 | 암묵적 (PAG-I) |
| **DDPM / DDIM** (Ho et al.) | 2020~21 | Diffusion 생성 패러다임 확립 | 암묵적 |
| **3DGS** (Kerbl et al.) | 2023 | 가우시안 기반 실시간 렌더링 | 명시적/암묵적 혼합 |
| **DreamFusion** (Poole et al.) | 2023 | SDS 기반 텍스트→3D | 암묵적 |
| **ProPhy** | 2024 | Mixture-of-Physics-Experts | 명시적 (PAG-E) |
| **PhysWorld** | 2024 | 비디오+물리재구성→로봇 | 명시적 (PAG-E) |
| **Liu et al. Survey** | 2025 | PAG-E/PAG-I 분류 체계 | 서베이 (분류) |

DreamFusion, Magic3D 같은 모델은 Diffusion Model을 통합하여 합성 사실성을 향상시켰으나, 주로 시각적 품질만을 최적화하여 부자연스러운 변형·불안정한 동작·비일관적 객체 상호작용 같은 물리적 타당성 문제를 발생시킨다.

ProPhy는 명시적 물리 인식 조건화와 이방성(anisotropic) 생성을 가능하게 하는 Progressive Physical Alignment Framework를 제안하며, 의미론 수준의 물리 원칙과 토큰 수준의 물리 역학을 각각 추론하는 Mixture-of-Physics-Experts(MoPE) 메커니즘을 채택하여 물리 법칙을 더 잘 반영하는 세밀한 물리 인식 비디오 표현을 학습한다.

---

## 📚 참고 자료 (출처 목록)

1. **논문 원문 (arXiv)**: Daochang Liu et al., *"Generative Physical AI in Vision: A Survey"*, arXiv:2501.10928, 2025. [https://arxiv.org/abs/2501.10928](https://arxiv.org/abs/2501.10928)
2. **논문 PDF**: [https://arxiv.org/pdf/2501.10928](https://arxiv.org/pdf/2501.10928)
3. **논문 HTML (v1)**: [https://arxiv.org/html/2501.10928v1](https://arxiv.org/html/2501.10928v1)
4. **논문 HTML (v2)**: [https://arxiv.org/html/2501.10928](https://arxiv.org/html/2501.10928)
5. **ResearchGate**: [https://www.researchgate.net/publication/388231760](https://www.researchgate.net/publication/388231760)
6. **aimodels.fyi 요약**: [https://www.aimodels.fyi/papers/arxiv/generative-physical-ai-vision-survey](https://www.aimodels.fyi/papers/arxiv/generative-physical-ai-vision-survey)
7. **Moonlight 리뷰**: [https://www.themoonlight.io/en/review/generative-physical-ai-in-vision-a-survey](https://www.themoonlight.io/en/review/generative-physical-ai-in-vision-a-survey)
8. **관련 서베이**: *"Grounding Creativity in Physics: A Brief Survey of Physical Priors in AIGC"*, arXiv:2502.07007, IJCAI-25. [https://arxiv.org/pdf/2502.07007](https://arxiv.org/pdf/2502.07007)
9. **관련 논문**: *"From Generative Engines to Actionable Simulators"*, arXiv:2601.15533. [https://arxiv.org/html/2601.15533v1](https://arxiv.org/html/2601.15533v1)
10. **관련 논문**: *"ProPhy: Progressive Physical Alignment"*, arXiv:2512.05564. [https://arxiv.org/pdf/2512.05564](https://arxiv.org/pdf/2512.05564)
11. **관련 논문**: *"Robot Learning from a Physical World Model (PhysWorld)"*, arXiv:2511.07416. [https://arxiv.org/pdf/2511.07416](https://arxiv.org/pdf/2511.07416)
12. **GitHub 논문 목록**: [https://github.com/BestJunYu/Awesome-Physics-aware-Generation](https://github.com/BestJunYu/Awesome-Physics-aware-Generation)

> ⚠️ **정확도 주의**: 이 서베이 논문의 내부 세부 수식 및 전체 표 내용은 PDF 전문 접근의 제한으로 인해, 개념적 수식 일부는 서베이가 다루는 대표 패러다임에 기반하여 설명적으로 표현하였습니다. 정확한 세부 수식 및 모든 분류 체계는 원문 PDF([arXiv:2501.10928](https://arxiv.org/pdf/2501.10928))를 직접 참조하시기를 강력히 권장합니다.
