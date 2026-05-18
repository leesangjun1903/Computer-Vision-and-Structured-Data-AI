
# Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models (sCM)

> **논문 정보**
> - **제목:** Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models
> - **저자:** Cheng Lu, Yang Song (OpenAI)
> - **arXiv:** [2410.11081](https://arxiv.org/abs/2410.11081) (v1: 2024.10.14, v2: 2025.03.01)
> - **학회:** ICLR 2025 게재

---

## 1. 핵심 주장 및 주요 기여 요약

Consistency Models(CMs)은 빠른 샘플링에 최적화된 강력한 Diffusion 기반 생성 모델이다. 그러나 기존 대부분의 CMs는 이산화된(discretized) 타임스텝으로 훈련되어 추가 하이퍼파라미터를 필요로 하고 이산화 오류에 취약하며, 연속-시간(continuous-time) 공식화 방법은 이 문제를 완화할 수 있으나 훈련 불안정성으로 성공이 제한되어 왔다.

이 논문의 핵심 주장은 **"sCM(simple, stable, scalable Consistency Model)"** 이라 불리는 새 프레임워크를 통해:

연속-시간 Consistency Models의 이론적 공식화를 단순화하여 대규모 데이터셋에 대한 훈련을 안정화하고 확장할 수 있다는 것이다.

### 주요 기여 요약 (3가지)

| 기여 | 내용 |
|---|---|
| **① 단순화 (Simplify)** | TrigFlow 프레임워크 — EDM과 Flow Matching 통합 |
| **② 안정화 (Stabilize)** | 불안정성 근본 원인 규명 및 해결책 제시 |
| **③ 확장 (Scale)** | 1.5B 파라미터, ImageNet 512×512까지 확장 |

이를 통해 전례 없는 규모인 1.5B 파라미터의 ImageNet 512×512 연속-시간 CM 훈련이 가능해졌으며, 단 2번의 샘플링 스텝만으로 CIFAR-10 FID 2.06, ImageNet 64×64 FID 1.48, ImageNet 512×512 FID 1.88을 달성하였다.

---

## 2. 해결 문제 · 제안 방법 · 모델 구조 · 성능 향상 · 한계

### 2-1. 해결하고자 하는 문제

기존 결과들은 모두 이산-시간 CMs 기반으로, 이는 이산화 오류를 유발하고 타임스텝 그리드의 세심한 스케줄링을 요구하여 최적이 아닌 샘플 품질로 이어질 수 있다. 반면 연속-시간 CMs는 이러한 문제를 회피할 수 있지만, 훈련 불안정성이라는 과제에 직면해 있었다.

연속-시간 CMs는 훈련 목적 함수를 CM의 탄젠트 공간(tangent space)에서 점수 매칭(score matching)으로 재구성하여 이산화 오류와 사전 훈련된 모델에서의 점수 명시적 평가를 피하지만, 이는 수치적으로나 훈련 다이나믹스에서 다양한 불안정성을 초래한다.

### 2-2. 제안하는 방법

#### ① **TrigFlow 프레임워크 (핵심 기여)**

TrigFlow는 EDM(Karras et al., 2022; 2024)과 Flow Matching을 통합하는 새로운 공식화로, 확산 모델, 관련 확률 흐름 ODE, CM의 공식화를 크게 단순화한다.

TrigFlow에서 확산 과정은 삼각함수 기반 보간(interpolant)으로 표현된다:

$$\mathbf{x}_t = \cos(t)\, \mathbf{x}_0 + \sin(t)\, \boldsymbol{\epsilon}, \quad t \in \left[0,\ \frac{\pi}{2}\right]$$

여기서 $\mathbf{x}_0$은 데이터, $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma_d^2 \mathbf{I})$는 노이즈이다.

TrigFlow 공식화는 $\sin(t)$와 $\cos(t)$를 보간자(interpolant)로 사용하여 경계 조건을 적용하며, 이 공식화는 이전에 제안된 확산 형식들을 통합하는 동시에 안정화가 더 쉽다.

TrigFlow는 Flow Matching(혹은 Stochastic Interpolants, Rectified Flows라고도 알려진)의 특수한 경우이며 $v$-prediction 파라미터화이다.

이에 대응하는 **확률 흐름 ODE(PF-ODE)**는 다음과 같다:

$$\frac{d\mathbf{x}_t}{dt} = \frac{\mathbf{x}_t - \cos(t)\, \mathbf{f}_\theta(\mathbf{x}_t, t)}{\sin(t)}$$

#### ② **Consistency Model의 TrigFlow 파라미터화**

경계 조건을 만족시키기 위해, CM을 PF-ODE의 1차 ODE 솔버를 사용한 단일 스텝 해로 파라미터화한다.

TrigFlow 하에서 CM은 다음과 같이 표현된다:

$$\mathbf{f}_\theta(\mathbf{x}_t, t) = \cos(t)\, \mathbf{x}_t - \sin(t)\, F_\theta(\mathbf{x}_t,\, c_{\text{noise}}(t))$$

여기서 $F_\theta$는 신경망, $c_{\text{noise}}(t)$는 시간 변환 함수이다.

#### ③ **연속-시간 CM 훈련 목적 함수 (sCD / sCT)**

연속-시간 Consistency Distillation(sCD)의 손실 함수는 다음과 같은 형태이다:

$$\mathcal{L}_{\text{sCD}}(\theta) = \mathbb{E}_{t, \mathbf{x}_t}\left[\lambda(t) \cdot d\!\left(\mathbf{f}_\theta(\mathbf{x}_t, t),\ \mathbf{f}_{\theta^-}(\hat{\mathbf{x}}_{t'}, t')\right)\right]$$

여기서 핵심은 **탄젠트 함수(tangent)** $\frac{d\mathbf{f}_{\theta^-}}{dt}$의 안정적인 계산이며, 이를 위해 **Jacobian-Vector Product (JVP)** 를 활용한다:

$$\frac{d\mathbf{f}_{\theta^-}(\mathbf{x}_t, t)}{dt} = \underbrace{\nabla_{\mathbf{x}_t}\mathbf{f}_{\theta^-} \cdot \frac{d\mathbf{x}_t}{dt}}_{\text{JVP}} + \frac{\partial \mathbf{f}_{\theta^-}}{\partial t}$$

이 탄젠트 공간으로의 점수 투영은 고차원 이미지의 스칼라(시간)에 대한 도함수인 Jacobian-Vector Products(JVPs)를 효율적으로 계산하기 위해 순방향 자동 미분(forward mode auto-differentiation)을 필요로 한다.

저자들은 그래디언트의 분산을 완화하기 위해 **적응적 가중치(adaptive weighting)** 전략과 **탄젠트 정규화(tangent normalization)** 를 제안한다.

또한 연속-시간 CMs를 위한 훈련 목적 함수를 재공식화하여 핵심 항들의 적응적 가중치(adaptive weighting)와 정규화, 그리고 안정적이고 확장 가능한 훈련을 위한 **점진적 어닐링(progressive annealing)** 을 통합한다.

### 2-3. 모델 구조 (네트워크 아키텍처 개선)

연구자들은 다음과 같은 아키텍처 개선을 제안하였다: **향상된 시간 컨디셔닝(Enhanced Time-Conditioning)** — 모델 내 시간 표현 개선; **적응적 그룹 정규화(Adaptive Group Normalization, AdaGN)** — 데이터 변화 처리를 위한 네트워크 아키텍처 개선; **개정된 훈련 목적 함수(Revised Training Objectives)** — 훈련 다이나믹스 안정화를 위한 손실 함수 재공식화.

구체적으로, 훈련 안정성을 위해 학습된 또는 Fourier 임베딩 대신 **위치 임베딩(Positional Embeddings)** 을 사용하고, 시간 임베딩 주입을 위해 **PixelNorm을 포함한 AdaGN** 을 사용하며, 적응적 손실 가중치를 위한 **logvar 출력**을 추가한다.

Fourier 임베딩에서의 큰 Fourier 스케일이 불안정성을 유발함을 발견하였으며, EDM 공식화는 $t \to \frac{\pi}{2}$일 때 수치 문제가 발생하는 반면, TrigFlow(위치 임베딩 사용)는 두 경우 모두 안정적인 편미분을 갖는다.

### 2-4. 성능 향상

연구자들은 CIFAR-10, ImageNet 64×64, ImageNet 512×512 데이터셋에서 sCMs를 훈련시켜 전례 없는 15억 파라미터까지 확장하였다. sCMs는 계산 자원이 증가함에 따라 샘플 품질이 향상되는 **예측 가능한 확장성**을 보이며, 단 2번의 샘플링 스텝으로 SOTA 확산 모델과 경쟁하는 결과를 달성하였다. FID 스코어: CIFAR-10 **2.06**, ImageNet 64×64 **1.48**, ImageNet 512×512 **1.88**로 선도 확산 모델과의 성능 격차를 10% 이내로 좁혔다.

sCMs는 단 2번의 샘플링 스텝으로 확산 모델과 비교 가능한 품질의 샘플을 생성하며, **약 50배의 wall-clock 속도 향상**을 달성하였다. 예를 들어, 15억 파라미터의 최대 모델은 단일 A100 GPU에서 추론 최적화 없이 단 0.11초 만에 샘플 하나를 생성한다.

핵심 발견으로, sCMs는 교사 확산 모델이 확장될 때 비례하여 개선된다. 구체적으로, FID 점수 비율로 측정된 샘플 품질의 상대적 차이가 여러 자릿수 크기의 모델 크기에 걸쳐 일관되게 유지되어, 규모가 커질수록 절대적인 샘플 품질 차이가 줄어드는 것을 보인다.

**VSD 대비 비교:**

VSD는 확산 모델에서 큰 가이던스 스케일 적용과 유사한 아티팩트를 나타내어 정밀도(precision)를 높이는 반면 다양성(recall)을 감소시키며, 이 효과는 가이던스 스케일이 증가할수록 더욱 두드러져 결국 심각한 모드 붕괴(mode collapse)를 유발한다. 반면 2-step sCD의 정밀도 및 재현율 점수는 교사 확산 모델과 비교 가능하여 VSD보다 나은 FID 점수를 달성한다.

### 2-5. 한계

최고의 sCMs도 초기화와 증류를 위해 사전 훈련된 확산 모델에 의존하여 교사 확산 모델에 비해 작지만 일관된 샘플 품질 격차가 존재한다. 또한, 샘플 품질 지표로서 FID 자체의 한계가 있어 FID 점수가 가깝다고 해서 항상 실제 샘플 품질을 반영하지는 않는다. 따라서 sCMs의 품질은 특정 응용 프로그램의 요구 사항에 따라 다르게 평가될 필요가 있다.

또한 논문은 노이즈가 많거나 불완전한 데이터에 직면했을 때 이 모델들이 어떻게 동작할지, 또는 빠르게 변화하거나 예측 불가능한 입력을 어떻게 처리할지에 대해 다루지 않는다. 또한 제안된 확장성 기법들은 유망하지만, 일부 시나리오에서 적용을 방해할 수 있는 추가적인 복잡성을 도입할 수 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 확장 법칙(Scaling Law)과 일반화

sCMs는 교사 확산 모델이 확장됨에 따라 비례적으로 개선된다. FID 점수 비율로 측정된 샘플 품질의 상대적 차이가 여러 자릿수 모델 크기에 걸쳐 일관되게 유지되어 절대적 품질 차이가 규모에서 줄어든다. 또한, sCMs의 샘플링 스텝 수를 늘리면 품질 격차가 더욱 줄어들며, 2-step sCM의 샘플은 이미 수백 스텝이 필요한 교사 확산 모델의 샘플과 비교 가능하다(FID 상대 차이 10% 이내).

이는 **일반화 측면에서 중요한 시사점**을 제공한다. 모델 크기 증가에 따른 예측 가능한 품질 향상 패턴이 확인되었으므로, 더 많은 데이터와 계산 자원이 주어졌을 때 **체계적으로 일반화 성능이 향상될 것**으로 기대된다.

### 3-2. 훈련 안정화가 일반화에 미치는 영향

정체성 시간 변환(identity time transformation), 위치 임베딩, 적응적 정규화 등 안정화 기법이 15억 파라미터로의 모델 확장을 가능하게 한다. 적응적 가중치와 점진적 어닐링은 성능을 향상시켜 더 적은 계산 자원으로 경쟁력 있는 FID 점수를 달성한다.

이러한 안정화 기법은 단순히 훈련 속도를 향상시키는 것을 넘어, 그래디언트 분산을 줄이고 수치적으로 안정적인 최적화를 달성함으로써 다양한 도메인에서의 **일반화 가능성**을 높인다.

### 3-3. 다양한 도메인으로의 일반화 가능성

커스텀화된 시스템 최적화를 통한 추가 가속이 쉽게 달성 가능하여, **이미지, 오디오, 비디오** 등 다양한 도메인에서 실시간 생성 가능성을 열어준다.

이러한 단순화는 성능을 희생하지 않고 더욱 확장 가능한 모델 아키텍처를 가능하게 하여, 생성적 AI에서 더 크고 복잡한 모델을 향한 경로를 제시한다.

### 3-4. 다운스트림 모델 활용 — SANA-Sprint 사례

SANA-Sprint는 사전 훈련된 이미지 생성 모델 SANA와 연속-시간 Consistency Models(sCMs)의 최근 발전을 기반으로 하며, SANA(Flow Matching 모델)를 sCM 증류에 필요한 TrigFlow 모델로 변환하였다. 이는 sCM 프레임워크의 일반화 가능성을 실증한다.

---

## 4. 연구에 미치는 영향 및 향후 고려 사항

### 4-1. 앞으로의 연구에 미치는 영향

#### (1) 확산 모델 증류(Distillation) 패러다임 전환
TrigFlow 프레임워크와 그 결과인 sCMs는 생성 모델 발전에 있어 중요한 이정표를 나타낸다. 훈련 불안정성과 확장 과제를 해결함으로써, 연속-시간 Consistency Models의 잠재력을 실현하여 샘플 품질, 확장성, 효율성에서 새로운 벤치마크를 설정하였다. 2-step 생성 프로세스와 획기적인 성능 지표를 갖춘 sCMs는 Diffusion 기반 생성 모델링의 미래 혁신을 위한 길을 열어준다.

#### (2) 새로운 후속 연구들 촉진
후속 연구로 **Inductive Moment Matching(IMM)** (arXiv:2503.07565)은 단일 스테이지 훈련 절차로 1~수 스텝 샘플링을 위한 새로운 생성 모델 클래스를 제안하며, 확산 및 Flow Matching 모델의 트레이드오프를 해결한다.

**Mean Flows** (arXiv:2505.13447)는 단일 스텝 생성을 위한 평균 속도(average velocity) 개념을 도입하여 처음부터 학습 시 ImageNet 256×256에서 FID 3.43을 달성하였다.

#### (3) 실시간 생성 AI의 실현 가능성 확보
sCD(사전 훈련 네트워크로부터의 증류)는 sCT보다 더 나은 작업 성능을 제공하고, Classifier-Free Guidance와 호환되며 더 계산 효율적이다. 또한, sCD가 교사 모델과 동일한 비율로 확장되는 바람직한 특성이 있다.

#### (4) 도메인 특화 적용 연구 활성화
**MotionPCM** (arXiv:2501.19083)은 실시간 텍스트 조건 인간 모션 합성을 위한 Phased Consistency Model 기반 접근법으로, 단일 스텝 추론에서 30 FPS 이상을 달성하여 sCM 접근법이 모션 생성에도 확장됨을 보여준다.

### 4-2. 향후 연구 시 고려할 점

#### ① 교사 모델 의존성 탈피
최고의 sCMs도 초기화와 증류를 위해 사전 훈련된 확산 모델에 의존하므로, 교사 확산 모델 대비 작지만 일관된 샘플 품질 격차가 남아있다. 이를 극복하는 **독립적인 자기-일관성 훈련(self-consistency training)** 방법론 개발이 필요하다.

#### ② 다양한 도메인 적용 연구
미래 연구는 계산 효율성을 추가로 최적화하거나 새로운 하드웨어 가속기와 통합하는 방향에 집중할 수 있다. 또한, 비디오나 3D 생성과 같은 다른 도메인으로의 탐색이 더 넓은 적용 가능성을 보여줄 수 있다.

#### ③ 평가 지표 다양화
FID는 샘플 품질 지표로서 자체적인 한계가 있어, FID 점수가 가깝다고 해서 항상 실제 샘플 품질을 반영하지는 않으며 그 반대도 마찬가지다. 따라서 sCMs의 품질은 특정 응용 프로그램의 요구 사항에 따라 다르게 평가될 필요가 있다. → CLIP 점수, 인간 평가, 다운스트림 태스크 성능 등 다각적 평가 필요.

#### ④ 메모리 및 연산 효율화
대규모 확산 모델 훈련은 FP16 및 Flash Attention과 같은 최적화가 필요하며, 시간 변수가 $0$ 또는 $\frac{\pi}{2}$와 같은 임계점에 접근할 때 정확한 탄젠트 계산을 달성하는 것이 훈련 안정성 보장에 핵심적이다. 이 논문은 JVP 재배열을 통한 탄젠트 계산의 새로운 접근법을 소개한다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 연도 | 방법 | FID (ImageNet 64×64) | 샘플링 스텝 | 특징 |
|---|---|---|---|---|---|
| **DDPM** (Ho et al.) | 2020 | Diffusion | ~3.17 | ~1000 | 기초 확산 모델 |
| **EDM** (Karras et al.) | 2022 | 개선된 Diffusion | 1.97 | ~35 | 노이즈 스케줄 최적화 |
| **Consistency Models** (Song et al.) | 2023 | CM (이산-시간) | 6.20 | 1 | CM 최초 제안 |
| **Improved CM** (Song & Dhariwal) | 2023 | iCM (이산-시간) | 4.02 | 2 | 이산 CM 개선 |
| **Latent Consistency Model** (Luo et al.) | 2023 | LCM (잠재 공간) | - | 2-4 | 잠재 공간 CM |
| **Phased CM** | 2024 | PCM | - | 1-16 | LCM 핵심 결함 개선 |
| **Multistep CM** (Heek et al.) | 2024 | MCM | - | 다중 | 다중 스텝 개선 |
| **sCM (본 논문)** | 2024 | 연속-시간 CM | **1.48** | **2** | TrigFlow, 1.5B |
| **SANA-Sprint** (후속 연구) | 2025 | sCM + LADD | - | 1 | T2I, 0.10s/image |

일관성 모델(Consistency Models)은 ICML 2023의 획기적인 연구에서 처음 도입되었으며, 노이즈에서 데이터로의 자기-일관성(self-consistency) 특성을 만족하는 매핑을 학습하여 사전 훈련된 확산 모델로부터 직접 증류하거나 처음부터 훈련하는 것이 가능하다. 이 접근법은 이미지 합성, 오디오, 비디오 등 다양한 응용 분야에서 소수 스텝 생성의 SOTA 결과를 이끌었으며, 이후 안정성, 확장성, 로봇공학, 3D 생성 등 새로운 도메인으로의 확장에 집중하는 연구들이 이어졌다.

---

## 📚 참고자료 및 출처

1. **[arXiv:2410.11081]** Cheng Lu, Yang Song. *"Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models"* (v2: 2025.03.01) — https://arxiv.org/abs/2410.11081
2. **[OpenAI 공식 블로그]** *"Simplifying, stabilizing, and scaling continuous-time consistency models"* — https://openai.com/index/simplifying-stabilizing-and-scaling-continuous-time-consistency-models/
3. **[OpenReview (ICLR 2025)]** *"Simplifying, Stabilizing and Scaling Continuous-time Consistency Models"* — https://openreview.net/forum?id=LyJi5ugyJx
4. **[ICLR 2025 Proceedings PDF]** — https://proceedings.iclr.cc/paper_files/paper/2025/file/7e9c2053258b1bdd32ff2654802cd594-Paper-Conference.pdf
5. **[Graphcore Research Blog]** *"Simplifying, Stabilizing & Scaling Continuous-Time Consistency Models"* — https://graphcore-research.github.io/ssscm/
6. **[Hugging Face Papers]** — https://huggingface.co/papers/2410.11081
7. **[Synced Review]** *"Redefines Consistency Models: OpenAI's TrigFlow Narrows FID Gap to 10%"* (2024.11.26) — https://syncedreview.com
8. **[MarkTechPost]** *"OpenAI Stabilizing Continuous-Time Generative Models"* (2024.10.27) — https://www.marktechpost.com
9. **[AZoAI]** *"OpenAI Simplifies and Scales Continuous-Time Consistency Models"* (2024.10.21) — https://www.azoai.com
10. **[Emergent Mind]** Paper summary — https://www.emergentmind.com/papers/2410.11081
11. **[GitHub: Awesome-Consistency-Models]** — https://github.com/G-U-N/Awesome-Consistency-Models
12. **[arXiv:2503.09641]** *"SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation"* (후속 연구)
13. **[ResearchGate]** — https://www.researchgate.net/publication/384938499
