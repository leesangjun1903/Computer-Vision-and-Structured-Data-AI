
# Diffusion Models Are Real-Time Game Engines

> **논문 정보**
> - **제목**: Diffusion Models Are Real-Time Game Engines
> - **저자**: Dani Valevski, Yaniv Leviathan, Moab Arar, Shlomi Fruchter
> - **소속**: Google Research, Tel Aviv University, Google DeepMind
> - **arXiv**: [2408.14837](https://arxiv.org/abs/2408.14837) (2024.08.27)
> - **학회**: ICLR 2025 (Poster)
> - **프로젝트 페이지**: [gamengen.github.io](https://gamengen.github.io)

---

## 1. 🔑 핵심 주장 및 주요 기여 요약

**GameNGen**은 뉴럴 모델만으로 완전히 구동되는 최초의 게임 엔진으로, DOOM이라는 고전 게임 위에서 훈련되어 장시간 고품질 인터랙티브 시뮬레이션을 가능하게 한다.

### ✅ 주요 기여 (4가지)

| 기여 항목 | 내용 |
|---|---|
| **최초 신경망 게임 엔진** | 전통적 게임 엔진 없이 신경망만으로 복잡한 3D FPS 게임 실시간 구동 |
| **2단계 학습 파이프라인** | RL 에이전트 + Diffusion 모델 결합 |
| **Noise Augmentation** | 장기 자기회귀 생성 안정화 기법 제안 |
| **인간 인식 수준의 품질** | 실제 게임과 시뮬레이션을 구분하기 어려운 수준 달성 |

GameNGen은 단일 TPU에서 초당 20프레임으로 실행되며, 다음 프레임 예측의 PSNR은 29.4로 손실 압축 JPEG와 비교 가능한 수준이다. 사람 평가자들은 게임 실제 클립과 시뮬레이션 클립을 구분하는 데 무작위 확률보다 약간 우수한 정도에 그쳤다.

---

## 2. 🔬 상세 분석

### 2-1. 해결하고자 하는 문제

기존 생성 모델, 특히 확산 모델은 이미지·비디오 생성에서 상당한 발전을 이루었지만, 인터랙티브 월드 시뮬레이션은 **지속적인 행동 조건화**라는 고유한 도전 과제를 부과한다. 기존의 신경망 기반 인터랙티브 게임 시뮬레이션 접근법은 **게임 복잡성, 시뮬레이션 속도, 장기 안정성, 시각적 품질** 중 하나 이상이 부족하여 실시간·고품질 인터랙티브 경험이 어렵다.

특히 영상 생성 분야에서 인상적인 발전에도 불구하고, **비디오 확산 모델은 실시간 응용에 사용하기에는 여전히 너무 느리다**는 문제가 있었다.

---

### 2-2. 제안하는 방법

#### 📌 Phase 1: RL 에이전트를 통한 데이터 수집

인간 게임플레이 데이터를 대규모로 수집할 수 없기 때문에, 첫 번째 단계로 게임과 상호작용하는 RL 에이전트를 자동으로 훈련시켜 그 에피소드를 수집한다.

#### 📌 Phase 2: Diffusion 모델 기반 Next-Frame Prediction

훈련은 두 단계로 이루어진다: (1) RL 에이전트가 게임을 플레이하는 방법을 배우고 훈련 세션이 녹화되며, (2) 과거 프레임과 행동 시퀀스를 조건으로 다음 프레임을 생성하도록 확산 모델이 훈련된다.

#### 📌 기본 수식 체계

**잠재 확산 모델(LDM)의 노이즈 추가 과정:**

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t ; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t) \mathbf{I})$$

여기서 $\bar{\alpha}\_t = \prod_{s=1}^{t}(1 - \beta_s)$이며, $\beta_s$는 노이즈 스케줄이다.

**역방향 디노이징 (V-prediction):**

GameNGen은 확산 손실 파라미터화를 **v-prediction** 방식으로 변경한다.

$$\mathbf{v}_t = \sqrt{\bar{\alpha}_t} \boldsymbol{\epsilon} - \sqrt{1 - \bar{\alpha}_t} \mathbf{x}_0$$

모델은 $\mathbf{v}_t$를 예측하도록 학습하며, 이는 다음 훈련 목표로 표현된다:

$$\mathcal{L} = \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}, t} \left[ \left\| \mathbf{v}_t - \hat{\mathbf{v}}_\theta(\mathbf{x}_t, t, \mathbf{c}) \right\|^2 \right]$$

여기서 $\mathbf{c}$는 과거 프레임 및 행동으로 구성된 조건 벡터이다.

**조건부 생성 (Classifier-Free Guidance):**

컨텍스트 프레임 조건은 확률 0.1로 드롭아웃되어 추론 시 CFG(Classifier-Free Guidance)를 사용할 수 있도록 한다.

$$\hat{\mathbf{v}}_\theta(\mathbf{x}_t, t, \mathbf{c}) = (1 + w)\hat{\mathbf{v}}_\theta(\mathbf{x}_t, t, \mathbf{c}) - w \cdot \hat{\mathbf{v}}_\theta(\mathbf{x}_t, t, \emptyset)$$

#### 📌 Noise Augmentation (핵심 기법)

노이즈 보강을 위해 v-prediction을 사용하도록 확산 손실 함수를 수정하고, **최대 노이즈 레벨 0.7**, 10개의 임베딩 버킷으로 노이즈 보강을 적용한다.

과거 컨텍스트 프레임에 노이즈를 추가하는 방식:

$$\tilde{\mathbf{f}}_{t-k} = \sqrt{\bar{\alpha}_{n}} \mathbf{f}_{t-k} + \sqrt{1 - \bar{\alpha}_{n}} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

여기서 $n$은 최대 노이즈 레벨 0.7 이하에서 무작위로 샘플링된다.

> **핵심 직관**: 이 기법은 에이전트가 순차적인 프레임을 '교정'하도록 보상받으며, 이는 모델이 **에러 수정/안정적 유지**를 학습하게 하는 데 핵심적이었다.

---

### 2-3. 모델 구조

#### 📐 전체 아키텍처 개요

DOOM 게임을 **Stable Diffusion v1.4의 확장 버전**인 신경망 위에서 실시간으로 구동하며, 모든 시뮬레이션 모델은 Stable Diffusion 1.4의 사전 훈련 체크포인트에서 시작하여 U-Net 파라미터 전체를 언프리징하여 파인튜닝한다.

```
┌────────────────────────────────────────┐
│         GameNGen 아키텍처              │
├────────────────────────────────────────┤
│ 입력: 과거 프레임 (3.2초 히스토리)    │
│       + 행동 시퀀스 (a_t)             │
│                ↓                       │
│ 잠재 인코더 (VAE Encoder)              │
│   8x8 픽셀 패치 → 4 latent 채널       │
│                ↓                       │
│ 조건부 U-Net (Stable Diffusion 1.4)    │
│   - Cross-Attention (행동 임베딩)      │
│   - Noise Augmented Context Frames     │
│   - v-prediction 손실                  │
│                ↓                       │
│ 파인튜닝된 잠재 디코더                 │
│   (MSE Loss로 HUD 텍스트 개선)         │
│                ↓                       │
│ 출력: 다음 프레임 (Next Frame)         │
└────────────────────────────────────────┘
```

#### 📐 잠재 디코더 파인튜닝

Stable Diffusion v1.4의 사전 훈련된 오토인코더는 8x8 픽셀 패치를 4개의 잠재 채널로 압축하는데, 게임 프레임을 예측할 때 의미 있는 아티팩트가 발생하여 작은 세부사항과 특히 하단 HUD 바에 영향을 미친다. 이를 개선하기 위해 타깃 프레임 픽셀에 대해 **MSE 손실로 잠재 오토인코더의 디코더만 파인튜닝**한다.

#### 📐 훈련 하이퍼파라미터

배치 사이즈 128, 학습률 2e-5(고정), Adafactor 옵티마이저(Weight Decay 없음), 그래디언트 클리핑 1.0을 사용하며, 128개의 TPU-v5e 디바이스로 데이터 병렬화 훈련을 수행한다. 논문의 모든 결과는 기본적으로 700,000 훈련 스텝 이후의 결과이다.

#### 📐 추론 속도 최적화

추론 시 각 프레임을 빠르게 생성하기 위해 일반적인 수십 스텝 대신 **단 4회의 디노이징 스텝**만 사용한다.

4회 디노이징 스텝을 사용하면 총 U-Net 비용이 40ms(오토인코더 포함 총 추론 비용 50ms)로, 초당 20프레임의 속도를 달성한다. 품질 저하가 미미한 이유는 (1) 제한된 이미지 공간, (2) 이전 프레임에 의한 강력한 조건화의 조합 때문이라고 가설을 세운다.

---

### 2-4. 성능 향상

| 지표 | 값 | 비고 |
|---|---|---|
| **PSNR** | 29.4 | JPEG 손실 압축(품질 20-30) 수준 |
| **LPIPS** | 0.249 | 지각 유사도 |
| **FPS** | 20 (4-step) / 50 (distilled) | 단일 TPU-v5 |
| **인간 구분율** | ~58~60% | 무작위(50%)와 거의 동등 |

GameNGen은 단기 궤적에서 PSNR 29.43, LPIPS 0.249를 달성하여 품질 설정 20-30의 손실 JPEG 압축과 비교할 만한 수준을 보인다. 인간 평가에서도 평가자들이 짧은 시간(1.6초~3.2초) 동안 GameNGen 클립과 실제 게임을 구분하는 비율이 58%~60%에 불과했다.

---

### 2-5. 한계

GameNGen은 **제한된 메모리**라는 한계를 지닌다. 모델은 약 3초 조금 넘는 히스토리에만 접근할 수 있으며, 그럼에도 많은 게임 로직이 훨씬 더 긴 시간 지평선 동안 유지된다는 것이 놀랍다.

훈련에 상당한 에너지와 컴퓨팅이 소모되어 독립 복제가 어렵고, 확산 모델이 여전히 객체를 환각하여 장시간 세션 후 게임플레이 결함 위험이 있다. 자기회귀 드리프트 문제는 노이즈 보강에도 불구하고 완전히 해결되지 않았다.

또한 모델 가중치에 DOOM 텍스처와 레벨 레이아웃이 내재되어 있어 자산 IP에 관한 법적 문제가 제기된다. 기존 엔진이 코드와 저작권 예술을 분리하는 것과 대조적으로, 이 방식은 그 경계를 흐린다.

---

## 3. 🧬 모델의 일반화 성능 향상 가능성

### 3-1. 현재 일반화 능력

일부 게임 상태는 화면 픽셀을 통해 유지되지만(예: 탄약, 체력, 사용 가능한 무기 등), **모델은 의미 있는 일반화를 가능하게 하는 강력한 휴리스틱을 학습한 것으로 보인다**. 예를 들어, 렌더링된 뷰에서 플레이어 위치를 추론하고, 탄약 및 체력 수치에서 플레이어가 이미 해당 지역을 지나쳐 적을 물리쳤는지 추론할 수 있다.

3초 남짓한 제한된 메모리에도 불구하고 체력, 탄약, 오브젝트 상호작용을 포함한 장기 게임 상태 유지는 중요한 성과이며, 이는 모델이 **명시적인 컨텍스트 창을 넘어서는 게임 로직을 일반화하는 강력한 휴리스틱을 학습**했음을 시사한다.

### 3-2. 다른 게임/도메인으로의 이전 가능성

논문은 DOOM에서 GameNGen을 시연하며, 다른 게임이나 더 일반적으로 다른 인터랙티브 소프트웨어 시스템으로 테스트하는 것이 흥미로울 것이라고 밝혔다. 특히 **기술에서 DOOM에 특화된 부분은 RL 에이전트의 보상 함수뿐**임을 강조한다.

실제로 연구팀은 간단한 플랫폼 게임인 "Chrome Dino"에서도 GameNGen의 **다른 게임 유형 시뮬레이션 능력**을 시연하였다.

### 3-3. 노이즈 보강이 일반화에 미치는 영향

노이즈 보강은 단순히 안정화 기법을 넘어 **모델의 강건성(robustness)과 일반화**에 직접 기여한다:

$$\tilde{\mathbf{f}}_{t-k} = \sqrt{\bar{\alpha}_n} \mathbf{f}_{t-k} + \sqrt{1 - \bar{\alpha}_n} \boldsymbol{\epsilon}$$

이 기법은 다음 이유로 일반화에 기여한다:
- 테스트 시 발생하는 **누적 예측 오류에 대한 내성** 학습
- 모델이 완벽한 프레임이 아닌 **잡음 있는 관측에서도 추론**할 수 있게 됨
- 훈련-테스트 분포 차이(distribution shift) 완화

노이즈 보강 조건화는 **장기 궤적에 걸친 안정적인 자기회귀 생성**을 보장하는 데 도움이 되며, 디코더 파인튜닝은 시각적 세부사항과 텍스트의 충실도를 향상시킨다.

### 3-4. Stable Diffusion 사전훈련의 역할

훈련 과정은 Stable Diffusion 1.4의 사전 훈련 체크포인트에서 시작한다. 이는 일반화 측면에서 중요한 의미를 가진다:
- 대규모 이미지 데이터로 학습된 **시각적 표현 능력의 전이(transfer)**
- 도메인 특화 학습이 적어도 되므로 **적은 게임 데이터로도 효과적인 시뮬레이션** 가능
- 다른 게임 도메인으로의 빠른 파인튜닝(fine-tuning) 가능성

---

## 4. 📊 2020년 이후 관련 최신 연구 비교 분석

관련 연구들의 핵심 비교는 다음과 같다 (NeurIPS 2024, ICLR 2025 기준):

| 모델 | 발표 | 게임 | 해상도 | 주요 특징 |
|---|---|---|---|---|
| **GameGAN** (NVIDIA) | 2020 | Pac-Man | 낮음 | GAN + 메모리 모듈 |
| **Genie 1** (DeepMind) | 2024.02 | 2D 플랫포머 | - | 레이블 없는 행동 학습 |
| **GameNGen** (Google) | 2024.08 | DOOM (3D FPS) | 240p | RL + Diffusion, 실시간 |
| **DIAMOND** (Alonso et al.) | 2024 (NeurIPS Spotlight) | Atari, CS:GO | 280×150 | RL 통합 확산 월드 모델 |
| **Oasis** (Decart) | 2024.10 | Minecraft | 640×360 | Transformer + Diffusion |
| **GameGen-X** | 2025 (ICLR) | AAA 게임 | 720p | Diffusion Transformer |
| **Genie 2** (DeepMind) | 2024.12 | 다양한 3D | 360p | 장기 메모리, OOD 일반화 |
| **Genie 3** (DeepMind) | 2025.08 | 범용 3D 세계 | 고해상도 | 실시간 + 일관성 대폭 향상 |

NVIDIA의 2020년 GameGAN은 메모리 모듈과 GAN을 결합하여 Pac-Man을 재현했지만, 1인칭 시점과 사실적인 그래픽이 부족했다. 아타리 프레임을 잠재 코드로 압축하는 World Models도 20 FPS에서의 연속적인 제어를 시도하지 않았다. 반면 GameNGen은 마우스 조준이 있는 3차원 슈터에서 실시간 생성형 확산을 구현했다.

Alonso et al.(2024, DIAMOND)은 GameNGen과 동시에 확산 월드 모델을 학습하여 관측 히스토리를 기반으로 다음 관측을 예측하고, 아타리 게임에서 월드 모델과 RL 모델을 반복적으로 훈련한다.

Decart와 Etched가 만든 Oasis는 Minecraft 유사 생성 인터랙티브 월드 모델의 기술 시연으로, 키보드 입력을 받아 실시간 물리 기반 게임플레이를 생성하며 이동, 점프, 아이템 획득, 블록 파괴 등이 가능하다.

GameNGen은 실시간으로 렌더링되는 인터랙티브 환경을 제공했지만, 해상도가 낮고 DOOM 게임 환경에만 한정되었다는 한계가 있었다.

이후 Genie 3는 Genie 2와 비교하여 일관성과 사실성을 개선하면서 **실시간 상호작용을 허용하는 최초의 월드 모델**이 되었다.

---

## 5. 🔮 앞으로의 연구에 미치는 영향 및 고려 사항

### 5-1. 앞으로의 연구에 미치는 영향

**① 신경망 게임 엔진 패러다임의 개막**

GameNGen은 게임 엔진의 새로운 패러다임으로 가는 길에서 중요한 질문 중 하나에 답한다. 이미지와 비디오가 최근 뉴럴 모델에 의해 생성되는 것처럼, **게임이 자동으로 생성**되는 방향이다. 다만 이러한 신경망 게임 엔진의 훈련 방법과 게임이 처음부터 효과적으로 만들어지는 방법, 그리고 인간 입력을 최적으로 활용하는 방법 등 핵심 질문들이 남아 있다.

**② AI 에이전트 훈련 환경으로의 활용**

세계 모델은 환경이 어떻게 진화하고 행동이 환경에 어떻게 영향을 미치는지를 예측할 수 있어 에이전트 훈련에 활용 가능하다. 또한 AGI로 가는 경로에서 핵심 디딤돌로, 에이전트를 풍부한 시뮬레이션 환경의 무한한 커리큘럼에서 훈련할 수 있게 해준다.

**③ 로보틱스 및 자율주행으로의 확장**

세계 모델의 즉각적인 사용 사례는 전통적인 게임 환경을 훨씬 넘어선다. 로보틱스 분야에서는 복잡한 환경을 동적으로 이해하고 반응하는 실시간 인터랙티브 비디오 모델을 구동하여 더 직관적이고 적응적인 로봇 시스템의 길을 열 수 있다.

---

### 5-2. 앞으로 연구 시 고려할 점

#### 🔵 기술적 도전 과제

**① 메모리 및 장기 컨텍스트**

더 복잡한 게임/소프트웨어를 위해서는 **더 정교한 아키텍처**가 필요하며, 메모리를 효과적으로 확장하는 실험이 필수적이다.

$$\text{Context Length} \uparrow \Rightarrow \text{Game State Consistency} \uparrow$$

**② 일반화 가능한 학습 파이프라인 구축**

기술에서 DOOM에 특화된 부분은 RL 에이전트의 보상 함수뿐이므로, **보상 함수 자동화 또는 인간 피드백 기반 일반화 파이프라인** 개발이 중요하다.

**③ 소비자 하드웨어 최적화**

TPU에서 20 또는 50 FPS로 실행되는 것을 넘어서, **더 높은 프레임율과 소비자용 하드웨어에서의 구동**을 위한 최적화 기술 실험이 필요하다.

**④ 양자화 및 모델 증류**

Decart Oasis가 개척한 새로운 GPU 효율적 기법과 양자화 및 모델 증류의 발전으로 소비자 하드웨어에서 복잡한 시뮬레이션 실행이 가능해지고 있다.

#### 🔵 윤리 및 법적 고려 사항

모델 가중치에 DOOM 텍스처와 레벨 레이아웃이 내재되어 자산 IP에 관한 법적 문제가 제기된다. 기존 엔진이 코드와 저작권 예술을 분리하는 것과 대조적으로, 이 Real-Time Generative 파이프라인은 그 경계를 흐린다.

#### 🔵 연구 로드맵 제안

```
단기 (1-2년)
├── 다양한 게임 장르로의 확장 실험
├── 보상 함수 자동화 (LLM 활용 가능)
└── 소비자 GPU에서 실시간 구동 달성

중기 (2-4년)
├── 다중 게임/도메인 통합 월드 모델
├── 인간 게임플레이 데이터 통합
└── 장기 메모리 아키텍처 (Mamba, RWKV 등)

장기 (5년+)
├── AGI 훈련용 범용 시뮬레이션 환경
├── 자율주행·로보틱스 도메인 전이
└── 완전 AI 생성 게임 제작 파이프라인
```

GameNGen은 신경망 모델에 의해 인터랙티브한 세계가 자율적으로 생성되는 새로운 게임 엔진 시대로의 중요한 도약을 나타내며, **완전 AI 기반 게임 개발의 문을 열어 가상 환경이 만들어지고 경험되는 방식을 근본적으로 변화**시킬 가능성을 제시한다.

---

## 📚 참고 자료 목록

| # | 제목 / 출처 | URL |
|---|---|---|
| 1 | **Diffusion Models Are Real-Time Game Engines** (arXiv 원문) | https://arxiv.org/abs/2408.14837 |
| 2 | **GameNGen 공식 프로젝트 페이지** | https://gamengen.github.io |
| 3 | **OpenReview (ICLR 2025)** | https://openreview.net/forum?id=P8pqeEkn1H |
| 4 | **Hugging Face Paper Page** | https://huggingface.co/papers/2408.14837 |
| 5 | **Semantic Scholar** | https://www.semanticscholar.org/paper/...75d7591078a74aa6bfcdaf5bde7dbe5146a45ecd |
| 6 | **Synced Review 분석** | https://syncedreview.com/2024/09/06/googles-gamengen... |
| 7 | **AI CERTs News (한계·비교 분석)** | https://www.aicerts.ai/news/googles-real-time-generative-gamengen-breakthrough |
| 8 | **Deepgram - DIAMOND 분석** | https://deepgram.com/learn/diffusion-models-reimagining-game-environments-diamond |
| 9 | **Lightspeed VP - Hello, World Models!** | https://lsvp.com/stories/hello-world-models/ |
| 10 | **TechCrunch - Genie 2** | https://techcrunch.com/2024/12/04/deepminds-genie-2... |
| 11 | **Google DeepMind - Genie 3** | https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/ |
| 12 | **GameFactory (ICCV 2025, 비교 테이블)** | https://openaccess.thecvf.com/content/ICCV2025/... |
| 13 | **Genie World Model (Wikipedia)** | https://en.wikipedia.org/wiki/Genie_(world_model) |
| 14 | **Liner.com Quick Review** | https://liner.com/review/diffusion-models-are-realtime-game-engines |
| 15 | **Genie: Generative Interactive Environments** (arXiv 2402.15391) | https://arxiv.org/html/2402.15391v1 |
