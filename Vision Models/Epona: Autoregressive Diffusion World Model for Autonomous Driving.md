
# Epona: Autoregressive Diffusion World Model for Autonomous Driving

> **논문 정보**
> - **저자:** Kaiwen Zhang, Zhenyu Tang, Xiaotao Hu, Xingang Pan, Xiaoyang Guo, Yuan Liu, Jingwei Huang, Li Yuan, Qian Zhang, Xiao-Xiao Long, Xun Cao, Wei Yin
> - **게재:** ICCV 2025 (Proceedings of the IEEE/CVF International Conference on Computer Vision)
> - **arXiv:** [arXiv:2506.24113](https://arxiv.org/abs/2506.24113) (2025. 6. 30.)
> - **공식 프로젝트 페이지:** [https://kevin-thu.github.io/Epona/](https://kevin-thu.github.io/Epona/)
> - **공식 코드:** [https://github.com/Kevin-thu/Epona](https://github.com/Kevin-thu/Epona)

---

## 1. 핵심 주장 및 주요 기여 (요약)

Diffusion 모델은 비디오 생성에서 뛰어난 시각적 품질을 보여주어 자율주행 세계 모델(world model)로서 주목받고 있다. 그러나 기존의 비디오 diffusion 기반 세계 모델들은 유연한 길이의 장기 예측(long-horizon prediction)과 궤적 계획(trajectory planning) 통합에 어려움을 겪고 있다. 이는 기존 비디오 diffusion 모델들이 고정 길이 프레임 시퀀스의 전역 결합 분포(global joint distribution)를 모델링하는 방식에 의존하기 때문이다.

이를 해결하기 위해, 본 논문은 **Epona**를 제안한다 — 두 가지 핵심 혁신을 통해 지역화된 시공간 분포(localized spatiotemporal distribution) 모델링을 가능하게 하는 자기회귀 diffusion 세계 모델이다: 1) 시간적 동역학 모델링과 세밀한 미래 세계 생성을 분리하는 **분리형 시공간 인수분해(Decoupled Spatiotemporal Factorization)**, 2) 모션 플래닝과 시각적 모델링을 엔드-투-엔드 프레임워크로 통합하는 **모듈형 궤적 및 비디오 예측(Modular Trajectory and Video Prediction)**.

**주요 기여 요약:**

| 기여 항목 | 내용 |
|---|---|
| 🔑 모델 구조 | 자기회귀 Diffusion 세계 모델 (Epona) |
| 🔑 핵심 기법 1 | 분리형 시공간 인수분해 |
| 🔑 핵심 기법 2 | 비동기 멀티모달 생성 |
| 🔑 학습 전략 | Chain-of-Forward Training |
| 📊 성능 | FVD 7.4% 향상, 수분 단위 장기 예측 |
| 🚗 적용 | End-to-end 실시간 모션 플래너 |

---

## 2. 해결하고자 하는 문제, 제안 방법(수식), 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

기존 비디오 diffusion 방법들은 과거 및 미래 프레임의 결합 시공간 분포(joint spatial-temporal distribution)를 모델링하는데, 이는 시간적 잠재 모델링에서 명시적 인과 제약(causality constraints)이 부족하여 긴 시퀀스에서 오류가 누적되는 문제가 있다.

구체적으로 세 가지 핵심 문제가 있다:

1. **고정 길이 제약:** 기존 비디오 diffusion 기반 세계 모델은 유연한 길이의 장기 예측을 수행하기 어렵고, 고정 길이 프레임 시퀀스의 전역 결합 분포에 의존한다.

2. **자기회귀 드리프트(Autoregressive Drift):** 장기 생성 과정에서 자기회귀 드리프트 문제가 발생한다. 학습 시에는 모델이 실제 과거 프레임(ground-truth historical context)을 사용하지만, 추론 시에는 자신이 생성한 예측값을 사용하기 때문에 도메인 갭이 발생하고 오류가 누적되어 품질이 급격히 저하된다.

3. **궤적-비디오 결합 부재:** 기존 연구들은 고화질 미래 예측과 궤적 모델링을 분리하여 연구하는 경향이 있으며, HERMES와 Epona처럼 생성적 상상력(generative imagination)과 다운스트림 플래닝을 명시적으로 연결하는 연구는 극히 소수에 불과하다.

---

### 2-2. 제안 방법 및 수식

#### 🔧 핵심 혁신 1: 분리형 시공간 인수분해 (Decoupled Spatiotemporal Factorization)

Epona는 시공간-분리(spacetime-disentangled) 처리를 통해 이를 해결한다: **GPT 스타일 Transformer**는 인과 어텐션(causal attention)을 사용하여 압축된 잠재 공간에서 시간적 동역학을 처리하고, 두 개의 **쌍둥이 diffusion transformer**가 공간적 렌더링과 궤적 생성을 각각 처리한다.

이 분리형 시공간 인수분해에 따라, 다음 프레임 분포는 다음과 같이 분해된다:

$$p(x_{T+1} \mid x_{0:T}) = p_{\text{time}}(\tau_{T+1} \mid \tau_{T}) \cdot p_{\text{space}}(v_{T+1} \mid \tau_{T+1}, v_{0:T})$$

여기서:
- $x_{T+1}$: 다음 프레임 전체
- $\tau_{T+1}$: 시간적 잠재 변수 (temporal latent)
- $v_{T+1}$: 공간적(시각적) 잠재 변수 (spatial/visual latent)
- $p_{\text{time}}$: GPT-style Transformer가 담당하는 시간적 동역학
- $p_{\text{space}}$: VisDiT (시각 Diffusion Transformer)가 담당하는 공간적 생성

이를 통해 전체 모델의 생성 과정은 자기회귀적으로 다음과 같이 표현된다:

$$p(x_{1:T}) = \prod_{t=1}^{T} p(x_t \mid x_{0:t-1})$$

---

#### 🔧 핵심 혁신 2: 비동기 멀티모달 생성 (Asynchronous Multi-modal Generation)

궤적 플래닝과 시각적 생성은 별도의 병렬 디노이징 프로세스를 통해 분리되어 실행된다.

세계 모델은 멀티모달 시공간 트랜스포머(multimodal spatiotemporal transformer)를 사용하여 처음 $T$ 프레임의 과거 맥락을 처리하고, **Next-Frame Prediction DiT**(VisDiT)로 $T+1$ 프레임을 생성하며, **Trajectory Planning DiT**(TrajDiT)로 미래 $N$개 프레임의 포즈 궤적을 예측한다.

궤적 예측을 위한 Diffusion 역방향 프로세스:

$$p_\theta(\tau_{0:N} \mid x_{0:T}) = \int p(\tau_K) \prod_{k=1}^{K} p_\theta(\tau_{k-1} \mid \tau_k, x_{0:T}) \, d\tau_{1:K}$$

여기서:
- $\tau_{0:N}$: 미래 $N$개 시점의 포즈 궤적
- $k$: Diffusion 타임스텝 인덱스 (총 $K$개)
- $p_\theta(\tau_{k-1} \mid \tau_k, x_{0:T})$: 과거 프레임 조건부 궤적 디노이징

---

#### 🔧 핵심 혁신 3: Chain-of-Forward Training 전략

Epona는 Chain-of-Forward 전략을 사용하여 학습을 수행하는데, 합성된 예측값(synthetic predictions)을 학습 중 다시 맥락으로 피드백하여 모델이 장기 자기회귀 롤아웃에서 마주칠 오류에 노출되도록 함으로써 수분 단위 예측을 안정화한다.

Chain-of-Forward 훈련 없이는 시각 품질이 10~20초 후에 급격히 저하되지만, Chain-of-Forward 훈련을 적용하면 동일한 주행 장면에서 고시각적 품질이 유지되어 눈에 띄는 성능 저하 없이 수분 단위 영상을 생성할 수 있다.

학습 객관식으로는 Diffusion 모델의 표준 denoising 손실을 사용하며, 시각 생성(VisDiT)과 궤적 생성(TrajDiT) 각각에 대해:

$$\mathcal{L}_{\text{vis}} = \mathbb{E}_{t, \epsilon}\left[\left\| \epsilon - \epsilon_\theta\left(v_t^{(k)}, k, c_{\text{vis}}\right) \right\|^2\right]$$

$$\mathcal{L}_{\text{traj}} = \mathbb{E}_{t, \epsilon}\left[\left\| \epsilon - \epsilon_\phi\left(\tau_t^{(k)}, k, c_{\text{traj}}\right) \right\|^2\right]$$

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{vis}} + \lambda \cdot \mathcal{L}_{\text{traj}}$$

여기서 $c_{\text{vis}}, c_{\text{traj}}$는 각각 시각 및 궤적 DiT에 입력되는 조건 벡터이다.

> ⚠️ **주의:** 위 손실 함수 표현은 논문의 일반적인 Diffusion 학습 방식을 참조하여 재구성한 것으로, 논문 원문의 정확한 수식 표기와 다를 수 있습니다. 정확한 수식은 [원문 PDF](https://arxiv.org/pdf/2506.24113)를 직접 참조하시기 바랍니다.

---

### 2-3. 모델 구조

```
 ┌─────────────────────────────────────────────────────────────┐
 │               Epona Architecture Overview                   │
 │                                                             │
 │  Historical Frames x_{0:T}                                  │
 │         │                                                   │
 │         ▼                                                   │
 │  ┌─────────────────────────────┐                            │
 │  │  Multimodal Spatiotemporal  │  ← GPT-style Transformer   │
 │  │      Transformer (MST)      │    (Causal Attention)      │
 │  └────────────┬────────────────┘                            │
 │               │  Temporal Latent τ_{T+1}                    │
 │      ┌────────┴────────────┐                                │
 │      ▼                     ▼                                │
 │  ┌─────────┐          ┌──────────┐                          │
 │  │ VisDiT  │          │ TrajDiT  │                          │
 │  │(시각 DiT)│          │(궤적 DiT)│  ← Parallel Denoising   │
 │  └────┬────┘          └────┬─────┘                          │
 │       │                    │                                │
 │  x_{T+1} (next frame)  τ_{1:N} (future trajectory)         │
 │                                                             │
 │  ← Autoregressive Loop (x_{T+1} fed back as x_{T+1}) →    │
 └─────────────────────────────────────────────────────────────┘
```

GPT 스타일 트랜스포머는 인과 어텐션으로 압축된 잠재 공간에서 시간적 동역학을 처리하고, 두 개의 쌍둥이 diffusion transformer (VisDiT, TrajDiT)가 각각 공간적 렌더링과 궤적 생성을 담당한다.

학습에는 nuPlan과 nuScenes (700 scenes)를 사용하였고, 이미지 해상도는 512×1024이다. NVIDIA A100 GPU 48개를 사용하여 약 2주간 600K iterations, 배치 크기 96으로 학습하였다. Chain-of-Forward Training은 매 10 스텝마다 3회의 순전파(forward pass)를 수행한다.

---

### 2-4. 성능 향상

NuScenes 검증 세트에서 수행한 생성 비디오 비교에서, Epona는 기존 주행 세계 모델 대비 최고 수준의 FVD 점수를 달성하였으며, 영상 길이를 2분 이상으로 확장하였다.

Epona의 아키텍처는 고해상도 장기 생성을 가능하게 하며, Chain-of-Forward 학습 전략으로 자기회귀 루프의 오류 누적을 해소하여 **7.4% FVD 향상**과 수분 단위 예측 지속시간을 기존 연구 대비 달성하였다.

Epona 세계 모델은 NAVSIM 벤치마크의 전체 PDMS(예측 드라이버 모델 점수)에서 강력한 엔드-투-엔드 플래너들을 능가하는 성능을 보였다.

| 지표 | 성능 |
|---|---|
| FVD (NuScenes val) | SOTA (기존 대비 7.4% ↓) |
| 최대 생성 길이 | 수분 이상 (2분+) |
| 벤치마크 | NuScenes, NuPlan, NAVSIM |
| PDMS (NAVSIM) | 강력한 end-to-end planner 초과 |

---

### 2-5. 한계점

논문 및 관련 자료에서 확인된 한계는 다음과 같다:

1. **단일 카메라/도메인 의존성:** 프로젝트 페이지에 따르면 중국 주행 장면의 대규모 인하우스(in-house) 데이터셋으로 파인튜닝된 결과물을 보여주는 예시가 있어, 특정 데이터 도메인에 대한 파인튜닝이 필요함을 시사한다.

2. **대규모 학습 자원 요구:** A100 GPU 48개로 2주 동안 학습해야 하므로, 재현 및 확장에 상당한 컴퓨팅 자원이 필요하다.

3. **오류 누적의 근본적 한계:** 장기 생성에서 자기회귀 드리프트 문제는 학습과 추론 간의 도메인 갭에서 비롯되는 근본적인 문제이며, Chain-of-Forward가 이를 완화하나 완전히 제거하지는 못한다.

4. **멀티뷰 미지원(암시적):** 현재 단일 전방 카메라 기준 시나리오에 집중되어 있어, 멀티카메라 또는 LiDAR 통합에 대한 확장성은 향후 과제로 남아 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 자기회귀 구조가 가져오는 일반화 이점

Epona는 연속 공간(continuous space)에서 지역화된 시공간 분포를 순차적으로 모델링함으로써 유연한 길이의 장기 예측과 통합된 궤적 플래닝을 가능하게 한다. 이는 기존 고정 길이 공동 모델링 방식에 비해 다양한 도로 상황 및 시나리오로의 일반화에 근본적으로 유리하다.

### 3-2. Chain-of-Forward를 통한 일반화 강화

시공간 인수분해와 메모리 강화 모듈은 오류 누적을 완화하고 효율적인 장기 스트리밍을 가능하게 한다. Chain-of-Forward 훈련과 같은 고급 학습 전략은 노출 편향(exposure bias)을 줄여 실시간, 유연하고 제어 가능한 비디오 합성을 지원한다.

### 3-3. 도메인 일반화 가능성

사전 정의된 포즈 궤적이 주어지면 다양한 조건 프레임으로부터 해당 모션 경로에 맞는 미래 프레임을 생성할 수 있어, 극단적 시나리오에서의 자율주행 비디오 획득에 중요하다. 이는 학습 데이터에 포함되지 않은 희귀 시나리오에 대한 일반화 능력을 간접적으로 뒷받침한다.

최근 비디오 생성 모델의 급속한 발전과 함께 세계 모델은 물리적 세계 시뮬레이션 및 자율 의사결정을 위한 강력한 패러다임으로 부상하고 있다. 이러한 파운데이션 모델들은 에이전트가 세계 지식을 이해하고 미래 동역학을 예측할 수 있게 하여, 자율주행에 특히 유망하다.

### 3-4. 일반화 성능 향상을 위한 잠재적 방향

| 방향 | 설명 |
|---|---|
| **멀티도메인 학습** | nuScenes, nuPlan 외 다양한 지역 데이터(중국, 유럽 등) 통합 |
| **파인튜닝 용이성** | DeepSpeed를 활용한 학습 및 파인튜닝 스크립트 제공으로 새로운 도메인 적응이 용이하다. |
| **다중 센서 확장** | LiDAR, 멀티카메라 통합으로 공간 이해 강화 |
| **시나리오 다양성** | 악천후, 야간, 극단 시나리오 데이터 증강 |

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 모델 | 연도 | 방법론 | 장기 예측 | 궤적 통합 | 주요 특징 |
|---|---|---|---|---|---|
| **GAIA-1** (Wayve) | 2023 | Autoregressive (Token) | 제한적 | ❌ | 대규모 언어-비전 결합 |
| **DriveDreamer** | 2023 | Video Diffusion | 고정 길이 | 부분적 | 조건부 비디오 생성 |
| **Vista** | 2024 | Video Diffusion (25-frame) | 롤아웃 | ❌ | 25프레임 고정 길이 비디오 diffusion 모델로, 더 긴 비디오 생성을 위해 롤아웃을 수행해야 한다. |
| **DrivingWorld** | 2024 | GPT (Discrete Token) | 가능 | 부분적 | 공간-시간 융합 메커니즘을 통해 고충실도 장기 비디오 생성을 가능하게 하는 자율주행용 GPT 스타일 세계 모델이다. |
| **GenAD** | 2024 | End-to-End Generative | 제한적 | ✅ | 생성-플래닝 통합 |
| **DrivingGPT** | 2025 | Multimodal Autoregressive | 가능 | ✅ | 멀티모달 자기회귀 통합 |
| **HERMES** | 2025 | Generative Imagination | 가능 | ✅ | 생성적 상상력과 다운스트림 플래닝을 명시적으로 연결하는 선구적 연구 중 하나이다. |
| **Epona (본 논문)** | 2025 | AR + Diffusion (Hybrid) | ✅ 수분+ | ✅ End-to-end | 분리형 시공간 + CoF Training |

Epona는 Vista 등 기존 최첨단 오픈소스 주행 세계 모델과 비교하여 일관된 장기 주행 장면을 고충실도 시각과 세밀한 구조 및 차량 묘사로 생성한다.

---

## 5. 향후 연구에 미치는 영향 및 고려 사항

### 5-1. 향후 연구에 미치는 영향

**① 자율주행 시뮬레이션 패러다임 전환**

Epona는 과거 주행 맥락으로부터 고해상도로 수분 단위의 일관된 미래 주행 장면을 생성할 수 있고, 다양한 궤적으로 제어가 가능하며, 실제 교통 지식을 이해하고, 미래 궤적을 예측하여 엔드-투-엔드 실시간 모션 플래너로 기능할 수 있다. 이는 향후 자율주행 시뮬레이터의 데이터 증강 및 테스트 환경 구축에 직접적인 영향을 미칠 것이다.

**② 세계 모델의 통합적 접근법 확산**

이 아키텍처는 고해상도 장기 생성을 가능하게 하며, 7.4% FVD 향상과 수분 단위 예측 지속시간을 달성하였다. 또한 학습된 세계 모델은 NAVSIM 벤치마크에서 강력한 엔드-투-엔드 플래너를 능가하는 실시간 모션 플래너로 기능한다. 이를 통해 세계 모델이 단순 시각 생성을 넘어 플래닝 모듈로서의 역할을 겸하는 통합적 연구 방향이 촉진될 것이다.

**③ Chain-of-Forward의 범용 적용 가능성**

Chain-of-Forward 전략은 합성 예측을 학습 맥락으로 피드백하여 장기 자기회귀 롤아웃의 오류에 모델을 노출시키는 방식으로, 다른 자기회귀 생성 모델에도 범용적으로 적용될 수 있는 학습 전략이다.

### 5-2. 향후 연구 시 고려할 점

**① 멀티센서 통합**
현재 단일 전방 카메라 기반 시나리오에 집중되어 있으므로, LiDAR, 레이더, 멀티카메라(360°)를 통합한 3D-consistent 세계 모델로의 확장이 필요하다.

**② 확장성(Scalability) vs. 효율성 트레이드오프**
A100 GPU 48개에서 2주간 학습이 요구되는 현실에서, 더 효율적인 경량화 아키텍처나 지식 증류(knowledge distillation) 기법 연구가 병행되어야 한다.

**③ 도메인 일반화(Out-of-Distribution) 평가 체계 구축**
nuScenes/nuPlan 외의 다양한 지역 및 기상 조건에서의 일반화 성능을 체계적으로 평가하는 벤치마크 구축이 필요하다.

**④ 안전-비판적(Safety-Critical) 시나리오 생성**
자율주행 시스템의 신뢰성 확보를 위해 드문 코너 케이스(edge cases), 악천후, 위험 상황을 의도적으로 생성할 수 있는 제어 가능성 강화 연구가 중요하다.

**⑤ 멀티에이전트(Multi-Agent) 상호작용 모델링**
현재 에고 차량 중심의 예측에서 벗어나, 주변 차량·보행자·오토바이 등 다수 에이전트의 상호작용 동역학을 포함하는 방향으로 확장이 필요하다.

**⑥ 실시간 추론 최적화**
현재 추론 스크립트는 단일 NVIDIA 4090 GPU에서 실행 가능하지만, 임베디드 차량 컴퓨터(예: NVIDIA Drive 플랫폼)에서의 실시간 배포를 위한 최적화 연구가 요구된다.

---

## 📚 참고자료 및 출처

| # | 제목 / 출처 |
|---|---|
| 1 | **Epona: Autoregressive Diffusion World Model for Autonomous Driving** — arXiv:2506.24113, ICCV 2025. https://arxiv.org/abs/2506.24113 |
| 2 | **Epona 공식 HTML 논문 페이지** — https://arxiv.org/html/2506.24113v1 |
| 3 | **Epona 공식 프로젝트 페이지** — https://kevin-thu.github.io/Epona/ |
| 4 | **Epona 공식 GitHub 코드** — https://github.com/Kevin-thu/Epona |
| 5 | **ICCV 2025 공식 게재 페이지** — https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_Epona_Autoregressive_Diffusion_World_Model_for_Autonomous_Driving_ICCV_2025_paper.pdf |
| 6 | **ICCV 2025 Virtual Poster** — https://iccv.thecvf.com/virtual/2025/poster/2672 |
| 7 | **Moonlight Literature Review: Epona** — https://www.themoonlight.io/en/review/epona-autoregressive-diffusion-world-model-for-autonomous-driving |
| 8 | **Next Diffusion Blog: Epona AI Model** — https://www.nextdiffusion.ai/blogs/epona-ai-model-driving-simulations |
| 9 | **EmergentMind: Diffusion World Models Topics** — https://www.emergentmind.com/topics/diffusion-world-models |
| 10 | **EmergentMind: Autoregressive Video Diffusion** — https://www.emergentmind.com/topics/autoregressive-video-diffusion |
| 11 | **Speaker Deck: CV Study Group 66 – Epona 발표자료** — https://speakerdeck.com/kentosasaki/... |
| 12 | **DriveWorld-VLA (참조 논문)** — arXiv:2602.06521, https://arxiv.org/html/2602.06521 |
| 13 | **DrivingWorld: Constructing World Model via Video GPT** — arXiv:2412.19505, Semantic Scholar |
| 14 | **Awesome VLA for AD (GitHub)** — https://github.com/worldbench/awesome-vla-for-ad |

> ⚠️ **정확도 고지:** 본 답변은 arXiv 공개 논문 및 ICCV 2025 공식 자료에 기반하였습니다. 일부 수식(특히 손실 함수)은 논문의 일반적 Diffusion 학습 방식을 토대로 재구성하였으며, 정확한 수식 및 세부 구현은 [원문 PDF](https://arxiv.org/pdf/2506.24113)를 직접 확인하시기를 강력히 권장합니다.
