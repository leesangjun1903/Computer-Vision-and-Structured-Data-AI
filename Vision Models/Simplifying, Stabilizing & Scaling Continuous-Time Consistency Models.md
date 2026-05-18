
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

# Simplifying, Stabilizing & Scaling Continuous-Time Consistency Models

## 1. 핵심 주장과 주요 기여

### 1.1 문제 정의 및 제안된 해결책

논문의 핵심 주장은 **연속시간 일관성 모델(Continuous-Time Consistency Models, CMs)은 이론적으로 우월하지만 훈련 불안정성으로 인해 실용화되지 못했다**는 것이다. 저자들은 이 불안정성의 근본 원인을 분석하고, 세 가지 차원에서의 개선을 제시한다:

1. **TrigFlow**: 편미분 방정식 공식화의 단순화
2. **아키텍처 개선**: 시간 조건화 및 정규화 최적화
3. **훈련 목표 재구성**: 적응형 가중치 및 정규화 추가

### 1.2 주요 기여도 (Contributions)

| 기여 항목 | 설명 | 임팩트 |
|---------|------|--------|
| **TrigFlow 프레임워크** | EDM과 Flow Matching을 통합하는 단순화된 공식 | 이론적 명확성 증대, 분석 용이성 |
| **연속시간 안정화** | 불안정성의 근본 원인 파악 및 5가지 기술적 해결 | 최초의 안정적 연속시간 CM 훈련 |
| **대규모 확장** | 1.5B 파라미터 모델까지 훈련 가능 증명 | CM의 스케일 한계 돌파 |
| **성능 달성** | 2-step으로 state-of-the-art FID 달성 | 실용적 생성 모델로서의 가치 입증 |

***

## 2. 문제 정의, 제안 방법, 모델 구조

### 2.1 해결하고자 하는 문제

#### A. 이산시간 CM의 근본적 한계
기존 연속시간 이전의 모든 CM은 **이산화 오류(discretization error)**를 포함한다:

$$\text{이산시간 CM 목표: } \mathbb{E}_{x_t,t}[w(t)d(f_\theta(x_t, t), f_{\theta^-}(x_{t-\Delta t}, t-\Delta t))]$$

이 공식은:
- 수치 ODE 솔버(numerical ODE solver)를 사용하여 $x_{t-\Delta t}$ 추정 필요
- $\Delta t$ 크기에 민감한 하이퍼파라미터 튜닝 필요
- $\Delta t \to 0$일 때 수렴하지만, 실제로는 유한한 $\Delta t$만 사용 가능

#### B. 연속시간 CM의 훈련 불안정성
연속시간 CM의 공식(Song et al., 2023):

$$\nabla_\theta \mathbb{E}_{x_t,t}\left[w(t)f_\theta^T(x_t, t) \frac{df_{\theta^-}(x_t, t)}{dt}\right]$$

**불안정성의 원인**:

$$\frac{df_{\theta^-}(x_t, t)}{dt} = \underbrace{-\cos(t)(\sigma_d F_{\theta^-} - \frac{dx_t}{dt})}_{\text{상대적으로 안정}} - \underbrace{\sin(t)(x_t + \sigma_d\frac{dF_{\theta^-}}{dt})}_{\text{극히 불안정}}$$

특히 $\sin(t)\frac{\partial F_{\theta^-}}{\partial t}$ 항이 시간 단계에서 극심한 진동을 유발

#### C. 대규모 모델 훈련 실패
- Song et al. (2023): 최대 500M 파라미터 정도에서만 시도
- Song & Dhariwal (2023): ImageNet 512×512에서 iCT-deep으로 ~4.5 FID에 멈춤
- 1B 이상의 모델: 훈련 불가능 상태

### 2.2 제안된 방법의 완전한 수식화

#### A. TrigFlow 프레임워크

**Diffusion Process**:
$$x_t = \cos(t)x_0 + \sin(t)z, \quad t \in [0, \pi/2], \quad z \sim \mathcal{N}(0, \sigma_d^2 I)$$

**특성**: 
- $x_0$에서 $x_{\pi/2} \sim \mathcal{N}(0, \sigma_d^2 I)$로의 선형 보간
- 삼각함수의 간결한 성질로 미분 계산 용이

**Probability Flow ODE**:
$$\frac{dx_t}{dt} = \sigma_d F_\theta\left(\frac{x_t}{\sigma_d}, c_{\text{noise}}(t)\right)$$

**Diffusion 모델 목표**:

$$L_{\text{Diff}}(\theta) = \mathbb{E}_{x_0,z,t}\left[\left\|\sigma_d F_\theta\left(\frac{x_t}{\sigma_d}, c_{\text{noise}}(t)\right) - v_t\right\|_2^2\right]$$

여기서 $v_t = \cos(t)z - \sin(t)x_0$는 속도(velocity) 벡터

**Consistency 모델 파라미터화**:

$$f_\theta(x_t, t) = \cos(t)x_t - \sin(t)\sigma_d F_\theta\left(\frac{x_t}{\sigma_d}, c_{\text{noise}}(t)\right)$$

경계 조건: $f_\theta(x, 0) = x$ 자동 만족

#### B. 연속시간 CM 개선된 훈련 목표

Tangent 함수의 분해:
$$\frac{df_{\theta^-}(x_t,t)}{dt} = -\cos(t)\left(\sigma_d F_{\theta^-} - \frac{dx_t}{dt}\right) - \sin(t)\left(x_t + \sigma_d\frac{dF_{\theta^-}}{dt}\right)$$

**1단계: Tangent Normalization** - 극심한 기울기 분산 억제

$$\frac{df_{\theta^-}}{dt} \rightarrow \frac{df_{\theta^-}/dt}{||df_{\theta^-}/dt|| + 0.1}$$

또는 clip version: $\text{clip}(df_{\theta^-}/dt, -1, 1)$

**2단계: Adaptive Weighting** - 시간 단계별 손실 분산 균형

$$L_{\text{sCM}}(\theta, \phi) = \mathbb{E}_{x_t,t}\left[e^{w_\phi(t)}\frac{1}{D}\left\|F_\theta\left(\frac{x_t}{\sigma_d}, t\right) - F_{\theta^-}\left(\frac{x_t}{\sigma_d}, t\right) - \cos(t)\frac{df_{\theta^-}(x_t, t)}{dt}\right\|_2^2 - w_\phi(t)\right] \quad (8)$$

여기서:
- $e^{w_\phi(t)}$는 시간 단계별 손실 크기 적응
- 사전 가중치: $w(t) = \frac{1}{\sigma_d \tan(t)}$ 통합

**3단계: JVP Rearrangement** - 수치 오버플로우 방지

$$\cos(t)\sin(t)\frac{dF_{\theta^-}}{dt} = (\nabla_{x_t/\sigma_d}F_{\theta^-}) \cdot (\cos(t)\sin(t)\frac{dx_t}{dt}) + \partial_t F_{\theta^-} \cdot (\cos(t)\sin(t)\sigma_d)$$

FP16 훈련에서 중간 레이어 오버플로우 해결

**4단계: Tangent Warmup** - 초기 훈련 안정화

$$\sin(t) \rightarrow r \cdot \sin(t), \quad r = \min(1, \text{iterations}/10000)$$

### 2.3 모델 구조 및 아키텍처 개선

#### A. Time Conditioning 개선

**기존 EDM 공식의 문제**:
$$c_{\text{noise}}(t) = \log(\sigma_d \tan(t)) \Rightarrow \sin(t) \cdot \partial_t c_{\text{noise}} = \frac{1}{\cos(t)} \rightarrow \infty \text{ as } t \to \pi/2$$

**sCM의 해결책**:
$$c_{\text{noise}}(t) = t \quad \text{(Identity transformation)}$$

결과: 시간 도함수의 안정성 Figure 4에서 명확히 증명
- EDM (Fourier scale 16.0): 불안정
- EDM (positional embedding): 개선되지만 여전히 문제
- TrigFlow (positional embedding): 안정적

#### B. Adaptive Double Normalization

**표준 AdaGN** (Dhariwal & Nichol, 2021):
$$y = \text{norm}(x) \odot s(t) + b(t)$$

**문제**: CM 훈련에서 발산 유발 (Song & Dhariwal, 2023)

**sCM의 개선**:
$$y = \text{norm}(x) \odot \text{pnorm}(s(t)) + \text{pnorm}(b(t))$$

Pixel normalization $\text{pnorm}(a) = a / \sqrt{\text{mean}(a^2) + \epsilon}$를 두 번 적용하여:
- AdaGN의 표현력 유지
- CM 훈련의 안정성 확보

#### C. Network Architecture 선택

- **주요 선택**: EDM2 기반 (Karras et al., 2024)
- **이유**: 
  - U-Net 구조의 검증된 효율성
  - Efficient backbone (CNN 기반)
  - Transformer 대비 ImageNet 우월성 입증
- **크기**: S, M, L, XL, XXL (280M ~ 1.5B 파라미터)

***

## 3. 성능 향상 분석

### 3.1 벤치마크 성능

#### CIFAR-10

| 방법 | 1-step FID | 2-step FID | 참고 |
|------|-----------|-----------|------|
| Song et al. (2023) CD | 3.55 | 2.93 | 기존 최선 |
| Song & Dhariwal (2023) iCT-deep | 2.51 | 2.24 | 개선된 이산시간 |
| Geng et al. (2024) ECT | 3.60 | 2.11 | 더 나은 distillation |
| **sCM (ours) sCD** | **3.66** | **2.52** | 연속시간, 우수 |
| **sCM (ours) sCT** | **2.85** | **2.06** | 최고 성능 ⭐ |

**분석**:
- sCT 2-step FID 2.06은 기존 대비 2.1% 개선
- 이산시간 이론의 한계를 연속시간으로 극복

#### ImageNet 64×64

| 방법 | 1-step FID | 2-step FID | 파라미터 |
|------|-----------|-----------|---------|
| Song et al. (2023) CD | 6.20 | 4.70 | 작음 |
| Geng et al. (2024) ECT | 2.49 | 1.67 | ~500M |
| **sCM sCD (S size)** | 2.44 | 1.66 | 280M |
| **sCM sCD (XL size)** | **2.40** | **1.93** | 1.1B |
| **sCM sCT (XL size)** | **2.04** | **1.48** | 1.1B ⭐ |
| EDM2-XXL (teacher) | - | 1.33 | 1.5B |

**주요 통찰**:
- sCD의 교사 모델 대비 FID 비율이 모든 모델 크기에서 일정 (Figure 6b)
- sCT는 소규모에서 효율적, 대규모에서 분산 증가
- 2-step sCT는 교사 모델 FID의 111% 수준 (비교: 다른 방법 120~150%)

#### ImageNet 512×512 (최대 규모 실험)

| 방법 | 1-step FID | 2-step FID | 파라미터 |
|------|-----------|-----------|---------|
| EDM2-XXL (teacher) | - | 1.73 | 1.5B |
| **sCM sCD-XXL** | 2.28 | **1.88** | 1.5B ⭐ |
| **sCM sCT-XXL** | 4.29 | 3.76 | 1.5B |

**critical insight**:
- 2-step sCD: 교사 모델 대비 1.88/1.73 = 1.087 (10.9% 격차)
- 논문의 claim: "10% 이내 격차 달성" 달성
- sCT의 latent space 한계 (높은 분산) 명확

### 3.2 스케일링 동역학 (Figure 6)

연속시간 CM의 **스케일 일관성** 증명:

$$\text{FID Ratio} = \frac{\text{FID}_{sCD}}{\text{FID}_{\text{teacher}}}$$

결과:
- 모든 모델 크기 (S, M, L, XL, XXL)에서 비율 약 1.10~1.15
- Step 수 증가 시 비율 감소 (수렴)
- **의의**: sCD가 교사 모델과 동일한 스케일링 법칙 따름

### 3.3 VSD와의 비교 (Figure 7)

**정밀도(Precision) vs 재현율(Recall) 분석**:

| 가이던스 수준 | Precision | Recall | FID |
|-------------|-----------|--------|-----|
| 1.0 (기준) | 0.87 | 0.60 | 5.2 |
| **VSD 1-step** | 0.89 ↑ | 0.54 ↓ | 6.1 |
| **sCD 2-step** | 0.87 | 0.60 | 4.2 |
| **Diffusion 기준** | 0.85 | 0.62 | 5.0 |

**결론**:
- VSD: 높은 가이던스에서 모드 붕괴 (recall ↓↓)
- sCM: 다양성-품질 균형 유지

***

## 4. 모델의 일반화 성능 향상 가능성

### 4.1 이론적 근거

**정리 (논문 Figure 5c 실증)**:
$$\lim_{\Delta t \to 0} \text{이산시간 CM 성능} = \text{연속시간 CM 성능}$$

실험 결과:
- N (이산화 스텝)가 증가할 때:
  - N ≤ 1024: 성능 개선
  - N > 1024: 수치 정밀도 문제로 악화
  - 연속시간: 항상 최고 성능

### 4.2 확장 가능성 차원별 분석

#### A. 모델 크기 확장
✓ **검증됨**: S(280M) → XXL(1.5B) 안정적 성능
- sCD: 모든 크기에서 일정한 FID 비율
- sCT: 분산 증가 (latent space 인코더/디코더 최적화 필요)

#### B. 해상도 확장  
✓ **검증됨**: 
- CIFAR-10 (32×32)
- ImageNet 64×64
- ImageNet 512×512 (최대 실험)

✓ **예상 확장**:
- 1024×1024: 아키텍처 수정으로 가능성 높음
- 비디오 생성: 시간축 확장 가능성 높음

#### C. 아키텍처 다양화
△ **제약 있음**:
- Adaptive group norm 수정 필요
- Positional embedding 요구
- Transformer 아키텍처 호환성 미검증

#### D. 데이터 도메인
△ **부분 검증**:
- 이미지(픽셀/latent): 검증됨
- 3D: 미탐색
- 오디오: 미탐색
- 비디오: 미탐색 (기술적으로 가능성 높음)

### 4.3 실제 일반화 한계

#### 명시된 한계 (논문 Limitations)

1. **sCT의 Latent Space 비효율성**
   ```
   원인: 인코더/디코더의 ill-conditioned 매핑
   해결책: 더 나은 VAE/VQGAN 개발 필요
   ```

2. **아티팩트 존재**
   ```
   ImageNet의 클래스 레이블 조건화 한계
   → Caption 기반 데이터에서 개선 예상
   ```

3. **CFG(Classifier-Free Guidance) 불호환**
   ```
   sCT는 CFG 미지원
   sCD는 지원하나 안정성 확인 필요
   ```

4. **아키텍처 의존성**
   ```
   네트워크 특정 수정(adaptive norm 등) 필수
   일반적 적용성 제한
   ```

***

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 연구 진화 시간선

```
2020-2022: 기초 - Diffusion 기반 설립 (DDPM, EDM)
    ↓
2023: 변곡점 - Consistency Models 도입
    ├─ Song et al.: 첫 CM (이산시간)
    ├─ Song & Dhariwal: 개선 기법 (이차 미분 고려)
    └─ Lipman et al.: Flow Matching (대안 패러다임)
    ↓
2024: 고도화 - 안정화 및 스케일링
    ├─ Karras et al.: EDM2 (적응형 가중치, 아키텍처)
    ├─ Geng et al.: ECT (더 나은 distillation)
    ├─ This paper: sCM (연속시간 안정화 ⭐)
    ├─ Yang et al.: Consistency-FM (velocity consistency)
    └─ Wang et al.: VSD (또 다른 distillation)
    ↓
2025: 통합 - Hybrid 접근
    ├─ LCFM: Latent + Flow Matching + Consistency
    ├─ SCFM: Flow Matching distillation 고도화
    └─ sLCT: Latent CM 훈련 안정화
```

### 5.2 주요 연구별 비교표

| 논문 | 시기 | 핵심 기여 | 최대 규모 | 주요 성과 | sCM과의 관계 |
|------|------|---------|---------|---------|-----------|
| **Song et al.** | 2023.3 | 첫 번째 CM | ~400M | FID 3.55 (CIFAR) | 기초 개념 제공 |
| **Song & Dhariwal** | 2023.10 | iCT, 이차 고려 | ~800M | FID 2.24 (CIFAR) | 불안정 연속시간 모델 |
| **Lipman et al.** | 2023.1 | Flow Matching | ~600M | 우수한 샘플 품질 | sCM의 TrigFlow와 유사 |
| **Karras et al. EDM2** | 2024.3 | 적응형 가중치 | 1.5B | FID 1.81 (IN512) | 적응형 가중치 상용 |
| **Geng et al. ECT** | 2024.6 | 더 나은 타겟 | ~1B | FID 2.11 (CIFAR) | 여전히 이산시간 |
| **sCM (This)** | 2024.10 | 연속시간 안정화 | **1.5B** | **FID 1.88 (IN512)** | **최고 수준** ⭐ |
| **Yang et al. CFM** | 2024.7 | Velocity consistency | ~700M | 빠른 수렴 | 유사한 안정화 목표 |
| **Wang et al. VSD** | 2024.5/6 | 직접 최적화 | ~600M | 높은 FID | 모드 붕괴 문제 |
| **LCFM** | 2025.1 | Hybrid 접근 | ~1B | 이론적 보장 | 상호 보완 가능 |
| **SCFM** | 2025.2 | FM distillation | ~5B | 3-step 생성 | 최신 FM 확장 |

### 5.3 sCM이 해결한 미해결 문제

#### 문제 1: 연속시간 CM의 훈련 불안정성
| 논문 | 해결? | 방법 |
|------|------|------|
| Song et al. (2023) | ✗ | "불안정성 발견, 미해결" |
| Song & Dhariwal (2023) | △ | AdaGN 제거 (임시방편) |
| Geng et al. (2024) | △ | ECT로 개선 (이산시간만) |
| **sCM** | **✓** | TrigFlow + 5가지 기술 |

#### 문제 2: 이산화 오류의 원칙적 해결
| 접근 | 이산화 오류 | 제한사항 |
|------|-----------|--------|
| 이산시간 CM | 있음 | Δt > 0 필수 |
| 고차 solver | 약간 감소 | 계산 비용 증가 |
| **연속시간 CM (sCM)** | **없음** | 훈련 불안정 → 이제 해결! |

#### 문제 3: 대규모 모델 확장 불가능
| 모델 | 최대 규모 | 상태 |
|------|---------|------|
| Song et al. | ~500M | 제한됨 |
| Song & Dhariwal | ~800M | 제한됨 |
| Geng et al. ECT | ~1B | 근처 |
| **sCM** | **1.5B** | ✓ 첫 번째 |

### 5.4 패러다임 비교: 3가지 고속 샘플링 방식

#### A. 이산시간 CM (Song et al., 2023 ~ Geng et al., 2024)
```
장점: 훈련 상대적 안정
단점: 이산화 오류, 하이퍼파라미터 민감
예시: iCT-deep (FID 2.24), ECT (FID 2.11)
```

#### B. 연속시간 CM (sCM - 이 논문)
```
장점: 원칙적 우월성, 큰 규모 지원, 이산화 오류 없음
단점: 훈련 복잡성 높음
성과: FID 2.06 (sCT), 1.88 (sCD)
```

#### C. Flow Matching + Consistency (LCFM, CFM, SCFM - 2024~2025)
```
장점: 더 직선적 궤적, 이론적 수렴 보장
단점: sCM보다 최근 (비교 부족)
성과: 높은 효율성, 멀티모달 응용
```

***

## 6. 앞으로의 연구에 미치는 영향과 고려사항

### 6.1 즉각적 영향 (Short-term: 2025)

#### A. 확산 모델 가족의 확대
- **Impact**: 연속시간 CM이 이제 실용적 → 새로운 연구 방향 열림
- **응용**: 
  - 멀티모달 생성 (3D, 비디오, 음성)
  - Domain adaptation
  - 세밀한 제어 (guided generation)

#### B. 산업 적용 가능성 증가
- **2-step 생성으로 배포 가능**
  - 모바일 디바이스 (정제된 버전)
  - 실시간 생성 (비디오 처리)
  - 저지연 시스템

#### C. 이론적 기여의 파급
- **TrigFlow의 통일 프레임워크**
  - EDM, Flow Matching, Velocity Prediction을 하나로
  - 향후 변형 모델의 설계 기초

### 6.2 중기 영향 (Medium-term: 2025-2026)

#### A. Latent Space 최적화 연구
sCM의 한계: **sCT의 Latent Space 비효율성**

```
현재: Encoder/Decoder → Ill-conditioned mapping
향후 연구:
1. Better VAE/VQGAN 설계
2. sCT 특화 인코더 학습
3. Hybrid: sCD for latent, sCT for pixel
```

#### B. 아키텍처 확장성 연구
```
sCM 현상태: Adaptive norm + Positional embedding 필수
향후:
- Transformer 완전 호환성
- Vision Transformer (ViT) 기반 모델
- 하이브리드 아키텍처 (CNN-ViT)
```

#### C. 도메인 확장
| 도메인 | 가능성 | 난제 |
|--------|------|------|
| **3D 생성** | 높음 | 기하학 표현 정의 |
| **비디오** | 높음 | 시간축 일관성 |
| **음성** | 중간 | Spectrogram vs raw audio |
| **분자** | 중간 | Graph 구조 처리 |
| **로봇** | 중간 | Action space 설계 |

### 6.3 장기 영향 (Long-term: 2026+)

#### A. 생성 모델 패러다임 통합
```
현재 상황:
├─ Diffusion (느리지만 안정적)
├─ Flow Matching (빠르지만 새로움)
├─ GAN (빠르지만 불안정)
└─ Autoregressive (느리지만 정확)

2026 이후 전망:
→ 하이브리드 프레임워크 (sCM + FM + Consistency)
→ Task별 최적화 모델 생태계
```

#### B. Theoretical Understanding 심화
```
sCM이 열어주는 이론 문제:
1. 연속시간 PF-ODE의 명시적 해석
2. Consistency 학습의 최적성 조건
3. Generalization bounds
```

#### C. 에너지 효율성
```
2-step 생성 = 기존 대비 10-100배 빠름
→ 탄소 발자국 극적 감소
→ 엣지 디바이스 배포 현실화
```

### 6.4 향후 연구 시 고려할 점 (Critical Considerations)

#### 1. **Latent Space 인코더/디코더 최적화**

**문제**: sCT의 높은 분산은 VAE/VQGAN의 ill-conditioned 특성에서 비롯

```python
# 연구 방향:
1. Variational bottleneck 약화 (β-VAE 접근)
2. 연속시간 특화 인코더 설계
3. Differentiable quantization (VQ-GAN-2)
```

**Expected Outcome**: sCT를 sCD 수준으로 끌어올리면 
- 사전훈련된 모델 불필요 (완전 독립 훈련 가능)
- 도메인 특화 모델 빠른 개발

#### 2. **Architecture-specific modifications 통일**

**현재 문제**: Adaptive norm, positional embedding 등 임시방편이 많음

```
개선 방향:
1. Normalization 기법의 일반화 이론
2. Time-conditioning 최적 설계
3. Guidance-compatible architecture
```

#### 3. **다중 스텝 생성 최적화**

**현재 상황**: 2-step이 목표, 1-step은 FID 격차 큼

```
향후 연구:
- 1-step과 2-step의 trade-off 분석
- 적응형 step 선택 (동적 계산)
- 계층적 생성 (coarse-to-fine)
```

#### 4. **Guidance Mechanism의 재설계**

**VSD 문제**: High guidance에서 모드 붕괴

```
sCM의 잠재력:
- Guidance 호환 연속시간 CM
- Semantic control의 정밀성
- Instruction-following generation
```

#### 5. **Theoretical Convergence Analysis**

**현재 부족**: 수치적 증거는 있지만 이론적 수렴 보장 없음

```
해결 필요:
1. Lipschitz continuity 증명
2. Convergence rate 분석 (O(1/T^α) 형태)
3. Generalization error bounds
```

***

## 7. 결론 및 전략적 의의

### 7.1 논문의 전략적 위치

sCM은 **일관성 모델 연구의 임계점**을 나타낸다:

```
2023 (발견): CM의 개념적 우월성 입증
  ↓
2023-2024 (고민): 연속시간 안정성 미해결
  ↓
2024.10 (돌파): sCM으로 연속시간 안정화 ⭐
  ↓
2025+ (확산): 다양한 도메인으로 응용 전개
```

### 7.2 성능의 의미

| 메트릭 | 달성 | 의미 |
|--------|------|------|
| FID 2.06 (CIFAR, 2-step) | 기존 대비 2% 개선 | 이산시간 한계 극복 |
| FID 1.88 (IN512, 2-step) | 교사 모델 대비 10% 격차 | 실용적 대체 가능 |
| 1.5B 파라미터 | 역대 최대 CM | 스케일 한계 돌파 |
| 안정적 훈련 | 5개의 기술 통합 | 공학적 성숙도 증명 |

### 7.3 미래 전망

**긍정적 시나리오 (2025-2026)**:
- Latent space 최적화로 sCT 개선 → 사전훈련 불필요
- 다양한 도메인 적용 (비디오, 3D, 음성)
- 멀티모달 일관성 모델 출현
- 1-step 생성의 실현 (현재는 2-step 최적)

**도전과제**:
- 아키텍처 복잡성으로 인한 채택 저해
- Latent space 한계의 기본적 해결 필요
- 이론적 이해 부족

**결론**: sCM은 **기술적으로는 해결책을 제시했지만, 실무 적용을 위해서는 추가 최적화가 필수**이다. 다만 연속시간 CM의 가능성을 증명함으로써 향후 10년 생성 모델 연구에 새로운 방향을 제시했다.

***

<span style="display:none">[^1_1][^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_2][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_3][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_4][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_5][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_6][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_7][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_8][^1_80][^1_81][^1_82][^1_83][^1_9]</span>

<div align="center">⁂</div>

[^1_1]: 2410.11081v2.pdf

[^1_2]: http://arxiv.org/pdf/2406.04485.pdf

[^1_3]: http://arxiv.org/pdf/2310.14189v1.pdf

[^1_4]: https://arxiv.org/abs/2303.01469

[^1_5]: https://arxiv.org/pdf/2502.17440.pdf

[^1_6]: https://arxiv.org/pdf/2301.04655.pdf

[^1_7]: https://arxiv.org/html/2503.08117v1

[^1_8]: https://arxiv.org/pdf/2307.01898.pdf

[^1_9]: http://arxiv.org/pdf/2407.13072.pdf

[^1_10]: https://syncedreview.com/2023/03/08/openais-consistency-models-support-fast-one-step-generation-for-diffusion-models/

[^1_11]: https://proceedings.mlr.press/v202/zheng23d/zheng23d.pdf

[^1_12]: https://www.openaccess.thecvf.com/content/CVPR2025/papers/Schusterbauer_Diff2Flow_Training_Flow_Matching_Models_via_Diffusion_Model_Alignment_CVPR_2025_paper.pdf

[^1_13]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5348747

[^1_14]: https://qsh-zh.github.io/deis/

[^1_15]: https://www.youtube.com/watch?v=7NNxK3CqaDk

[^1_16]: https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Fast_ODE-based_Sampling_for_Diffusion_Models_in_Around_5_Steps_CVPR_2024_paper.pdf

[^1_17]: https://openreview.net/forum?id=PqvMRDCJT9t

[^1_18]: https://cacm.acm.org/blogcacm/the-challenge-of-consistency-in-generative-ai-will-we-adapt-or-fix-the-system/

[^1_19]: https://arxiv.org/abs/2211.13449

[^1_20]: https://arxiv.org/abs/2506.02221

[^1_21]: https://papers.cumincad.org/data/works/att/caadria2025_567.pdf

[^1_22]: https://arxiv.org/abs/2204.13902

[^1_23]: https://arxiv.org/abs/2506.02070

[^1_24]: https://arxiv.org/pdf/2510.11677.pdf

[^1_25]: https://arxiv.org/abs/2106.00132

[^1_26]: https://arxiv.org/html/2510.17858v1

[^1_27]: https://pubmed.ncbi.nlm.nih.gov/40966479/

[^1_28]: https://arxiv.org/abs/2402.17376?utm

[^1_29]: https://arxiv.org/html/2510.20771v1

[^1_30]: https://pdfs.semanticscholar.org/4fa5/eccda27b3ff4932ec7bc46d60829484dc4f9.pdf

[^1_31]: https://arxiv.org/abs/2401.01008

[^1_32]: https://arxiv.org/html/2512.15657v1

[^1_33]: https://arxiv.org/html/2510.13852v1

[^1_34]: https://arxiv.org/abs/2410.18804

[^1_35]: https://arxiv.org/html/2512.02826v1

[^1_36]: https://arxiv.org/abs/2505.18825

[^1_37]: https://arxiv.org/abs/2402.09970

[^1_38]: https://arxiv.org/html/2506.08604

[^1_39]: https://www.semanticscholar.org/paper/8b7cce220c3b19f9b2d4a6c531907ed3b592b55e

[^1_40]: https://arxiv.org/abs/2311.05556

[^1_41]: https://doi.apa.org/doi/10.1037/xge0001344

[^1_42]: https://ieeexplore.ieee.org/document/10331300/

[^1_43]: https://doi.apa.org/doi/10.1037/tra0001499

[^1_44]: https://arxiv.org/abs/2310.20003

[^1_45]: https://archive.johs.org.uk/article/doi/10.54531/tzfd6375

[^1_46]: https://arxiv.org/abs/2306.05004

[^1_47]: https://doi.apa.org/doi/10.1037/pspp0000487

[^1_48]: https://doi.apa.org/doi/10.1037/tra0001465

[^1_49]: http://arxiv.org/pdf/2406.00356.pdf

[^1_50]: http://arxiv.org/pdf/2310.04378.pdf

[^1_51]: https://arxiv.org/abs/2312.09109

[^1_52]: https://arxiv.org/html/2408.02993

[^1_53]: http://arxiv.org/abs/2503.12615

[^1_54]: https://arxiv.org/html/2503.08377v1

[^1_55]: https://arxiv.org/html/2502.01441v2

[^1_56]: http://arxiv.org/pdf/2405.02791.pdf

[^1_57]: https://kimjy99.github.io/논문리뷰/latent-consistency-model/

[^1_58]: https://proceedings.neurips.cc/paper_files/paper/2024/file/dd540e1c8d26687d56d296e64d35949f-Paper-Conference.pdf

[^1_59]: https://www.emergentmind.com/topics/latent-consistency-flow-matching-lcfm

[^1_60]: https://www.youtube.com/watch?v=y0Tw9Zb4Sy4

[^1_61]: https://arxiv.org/html/2506.13763v1

[^1_62]: https://neurips.cc/virtual/2025/poster/116548

[^1_63]: https://arxiv.org/abs/2310.04378

[^1_64]: https://openreview.net/pdf?id=sn1kl4Dbm7

[^1_65]: https://liner.com/review/consistency-flow-matching-defining-straight-flows-with-velocity-consistency

[^1_66]: https://blog.outta.ai/17

[^1_67]: https://openreview.net/forum?id=OHZRUCa1HW

[^1_68]: https://openaccess.thecvf.com/content/ICCV2025/papers/Xu_Fast_Image_Super-Resolution_via_Consistency_Rectified_Flow_ICCV_2025_paper.pdf

[^1_69]: https://github.com/luosiallen/latent-consistency-model

[^1_70]: https://sander.ai/2024/06/14/noise-schedules.html

[^1_71]: https://openreview.net/forum?id=bS76qaGbel

[^1_72]: https://arxiv.org/pdf/2311.05556.pdf

[^1_73]: https://arxiv.org/pdf/2510.12537.pdf

[^1_74]: https://arxiv.org/html/2508.14807v1

[^1_75]: https://arxiv.org/html/2509.01819v1

[^1_76]: https://arxiv.org/pdf/2310.04378.pdf

[^1_77]: https://arxiv.org/html/2410.11081v1

[^1_78]: https://openaccess.thecvf.com/content/ICCV2025/papers/You_Consistency_Trajectory_Matching_for_One-Step_Generative_Super-Resolution_ICCV_2025_paper.pdf

[^1_79]: https://arxiv.org/html/2310.04378

[^1_80]: https://arxiv.org/html/2508.07926v1

[^1_81]: https://arxiv.org/html/2510.12537v1

[^1_82]: https://arxiv.org/html/2502.03500v2

[^1_83]: https://github.com/NVlabs/edm2/blob/main/README.md

