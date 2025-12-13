
# TRACT: Denoising Diffusion Models with Transitive Closure Time-Distillation

## 1. 논문의 핵심 주장 및 주요 기여

**TRACT(Transitive Closure Time-Distillation)**는 확산 모델의 추론 속도를 극적으로 향상시키기 위한 새로운 시간 증류 방법으로, 단일 단계 생성에서 최첨단 성능을 달성합니다. 이 논문의 핵심 주장은 다음과 같습니다:[1]

- **BTD의 문제점 극복**: 기존 이진 시간 증류(Binary Time-Distillation, BTD)의 두 가지 치명적 문제를 식별합니다. 첫째, 목적 함수의 퇴화(objective degeneracy)로 인해 누적된 근사 오류가 증류 단계를 거치며 악화됩니다. 둘째, BTD의 다중 위상 구조로 인해 공격적인 확률적 가중치 평균(Stochastic Weight Averaging, SWA)을 활용할 수 없습니다.[1]

- **TRACT의 혁신**: 학생 모델이 자기-교사(self-teacher)의 지수이동평균(EMA)을 활용하여 시간 t에서 t'으로의 증류를 부트스트랩 방식으로 수행합니다. 이를 통해 증류 단계를 log₂(T)에서 1-2 단계로 감소시킵니다.[1]

- **성능 향상**: CIFAR-10에서 단일 단계 FID를 9.12에서 5.02로 개선하고(BTD 대비 4.5배 향상), 64×64 ImageNet에서 17.5에서 7.43으로 개선(2.4배 향상)하며, EDM 교사 모델 사용 시 CIFAR-10에서 3.8, ImageNet에서 7.4의 최첨단 FID를 달성합니다.[1]

## 2. 해결하고자 하는 문제 및 제안 방법

### 2.1 문제 정의

확산 모델은 생성 품질에서 우수하지만 수천 개의 반복 단계가 필요하여 추론 시간이 매우 깁니다. Song et al.의 연구에 따르면 확산 모델의 추론은 신경 상미분방정식(Neural ODE)으로 표현될 수 있으며, 이산화 오류가 감소할수록 샘플 품질이 향상됩니다. 이러한 구조적 특성은 자원 제약 환경에서 배포를 어렵게 합니다.[1]

BTD는 이 문제를 해결하려 시도했지만 두 가지 근본적인 한계를 가집니다:[1]

1. **목적 함수 퇴화**: 이전 증류 단계의 학생이 불완전한 교사가 되어 다음 단계로 오류가 누적됩니다.
2. **일반화 저해**: log₂(T)개의 분리된 단계로 인해 고모멘트 EMA(예: 0.99 이상)를 활용하기 어려워져 일반화 성능이 저하됩니다.[1]

### 2.2 제안 방법 및 수식

#### DDIM 배경 설정

분산 보존(Variance Preserving, VP) 설정에서 노이즈 샘플은 다음과 같이 생성됩니다:[1]

$$x_t = x_0\sqrt{\gamma_t} + \epsilon\sqrt{1-\gamma_t}$$

여기서 γₜ ∈ [0,1)은 시간 스케줄이고, t=0일 때 γ₀ = 1입니다. DDIM의 결정론적 추론은 다음 단계 함수로 표현됩니다:[1]

$$x_{t'} = \delta(f_\theta, x_t, t, t') := x_t\frac{\sqrt{1-\gamma_{t'}}}{\sqrt{1-\gamma_t}} + f_\theta(x_t, t)\left(\sqrt{\gamma_{t'}(1-\gamma_t)} - \sqrt{\gamma_t(1-\gamma_{t'})}\right)\frac{1}{\sqrt{1-\gamma_t}}$$

여기서 $f_\theta(x_t, t)$는 $x_0$에 대한 신호 예측을 수행합니다.[1]

#### BTD 공식화

BTD는 학생 $g_φ$가 연속된 두 교사 단계를 근사하도록 학습합니다:[1]

$$\delta(g_φ, x_t, t, t-2) \approx x_{t-2} := \delta(f_\theta, \delta(f_\theta, x_t, t, t-1), t-1, t-2)$$

증류 대상은 다음과 같이 계산됩니다:[1]

$$\hat{x} = \frac{x_{t-2}\sqrt{1-\gamma_t} - x_t\sqrt{1-\gamma_{t-2}}}{\sqrt{\gamma_{t-2}}\sqrt{1-\gamma_t} - \sqrt{\gamma_t}\sqrt{1-\gamma_{t-2}}}$$

손실 함수는:[1]

$$L(\phi) = \frac{\gamma_t}{1-\gamma_t}\|g_φ(x_t, t) - \hat{x}\|_2^2$$

#### TRACT의 핵심 혁신: 자기-교사를 이용한 이행적 폐포

TRACT는 학생 모델을 위한 EMA 버전 $\tilde{\phi} = \text{EMA}(\phi, \mu_S)$를 도입합니다. 모멘트 파라미터 $\mu_S \in $은 하이퍼파라미터입니다. 이를 통해 이행적 폐포를 재귀적으로 표현할 수 있습니다:[1]

$$\delta(g_φ, x_t, t, t_i) \approx x_{t_i} := \delta(g_{\tilde{\phi}}, \delta(f_\theta, x_t, t, t-1), t-1, t_i)$$

이 재귀 관계로부터 증류 대상을 유도합니다:[1]

$$\hat{x} = \frac{x_{t_i}\sqrt{1-\gamma_t} - x_t\sqrt{1-\gamma_{t_i}}}{\sqrt{\gamma_{t_i}}\sqrt{1-\gamma_t} - \sqrt{\gamma_t}\sqrt{1-\gamma_{t_i}}}$$

특수한 경우 $t_i = t-1$일 때는 $\hat{x} = f_\theta(x_t, t)$입니다. 손실 함수는 표준 DDIM 증류 손실입니다:[1]

$$L(\phi) = \frac{\gamma_t}{1-\gamma_t}\|g_φ(x_t, t) - \hat{x}\|_2^2$$

#### EMA의 편향 수정 구현

학습 중 자기-교사 가중치는 다음과 같이 업데이트됩니다:[1]

$$\tilde{\phi}_i = \left(1 - \frac{1-\mu_S}{1-\mu_S^i}\right)\tilde{\phi}_{i-1} + \frac{1-\mu_S}{1-\mu_S^i}\phi_i$$

여기서 각 학습 단계 i > 0에서 편향 수정을 적용합니다.[1]

#### Variance Exploding (VE) 설정으로의 확장

EDM(Elucidating the Design space of Diffusion Models) 프레임워크에 적용하기 위해, TRACT는 VE 노이즈 스케줄을 지원합니다. VE 설정에서 샘플은:[1]

$$x_t = x_0 + \sigma_t\epsilon$$

Runge-Kutta (RK) 교사와 DDIM 학생 간의 증류에서, 증류 대상은:[1]

$$\hat{x} = \frac{\sigma_t x_s - \sigma_s x_t}{\sigma_t - \sigma_s}$$

손실 함수는 EDM의 가중화 방식을 따릅니다:[1]

$$L(\phi) = \lambda(\sigma_t)\|g_φ(x_t, t) - \hat{x}\|_2^2$$

## 3. 모델 구조

TRACT의 아키텍처는 다중 위상 구조를 가집니다:[1]

### 3.1 전체 증류 파이프라인

1. **초기화**: 학생 가중치 $\phi_0$을 교사 가중치 $\theta$로 초기화하고, 자기-교사 $\tilde{\phi}_0$도 동일하게 초기화합니다.[1]

2. **단계적 감소**: T 단계의 스케줄을 T' < T 단계로 감소시키기 위해, T 단계를 T' 개의 연속된 그룹으로 분할합니다.[1]

3. **트레이닝 루프**: 각 배치에서 무작위로 그룹 시작 위치 s와 그룹 내 인덱스 p를 샘플링하여 시간 단계 t = s + p를 결정합니다.[1]

### 3.2 알고리즘 1: 단일 위상 TRACT 학습

Algorithm 1에 명시된 트레이닝 절차는:[1]

1. 노이즈 샘플 $\epsilon$을 생성하고 시간 단계를 샘플링합니다.
2. 교사 모델로 한 단계 역감소 처리를 수행합니다: $x_{t-1} = \delta(f_\theta, x_t, t, t-1)$.
3. s = t-1인 경우 $x_s = x_{t-1}$을 설정하고, 그 외의 경우 자기-교사를 사용합니다: $x_s = \delta(g_{\tilde{\phi}}, x_{t-1}, t-1, s)$.
4. 증류 손실을 계산하고 학생을 업데이트합니다.
5. 자기-교사 가중치를 편향-수정 EMA로 업데이트합니다.[1]

### 3.3 실험 설정

CIFAR-10의 경우:[1]
- 글로벌 배치 크기: 256 (8개 GPU에 분산)
- 최적화 도구: Adam (학습률 $2 \times 10^{-4}$, 가중치 감쇠 없음, 드롭아웃 없음)
- 자기-교사 모멘트: $\mu_S = 0.5$
- 추론 시간 EMA 모멘트: $\mu_I = 0.99997$ (96M) 또는 $0.99999$ (256M)
- 증류 스케줄: 1024 → 32 → 1 (2단계)

64×64 ImageNet의 경우:[1]
- 글로벌 배치 크기: 256 (8개 GPU)
- 동일한 최적화 설정, $\mu_I = 0.99995$
- 신호와 노이즈 동시 예측 (BTD 설정 따름)

## 4. 성능 향상 및 실험 결과

### 4.1 주요 정량적 성과

CIFAR-10 단일 단계 결과:[1]
- **TRACT-96M**: FID 5.02 (BTD 9.12 대비 4.5배 개선)
- **TRACT-256M**: FID 4.45
- **TRACT-EDM-256M**: FID 3.78 ±0.01

64×64 ImageNet 단일 단계 결과:[1]
- **TRACT-96M**: FID 7.43 (BTD 17.5 대비 2.4배 개선)
- **TRACT-EDM-96M**: FID 7.52 ±0.05

2단계 결과:[1]
- CIFAR-10: FID 3.32 ±0.02 (TRACT-256M)
- ImageNet: FID 4.97 ±0.03 (TRACT-EDM-256M)

### 4.2 절제 연구(Ablation Studies)

#### 자기-교사 EMA 모멘트($\mu_S$) 영향

표 3에 따르면 $\mu_S$ 값이 성능에 크게 영향을 미칩니다:[1]
- $\mu_S = 0.0$: FID 6.32
- $\mu_S = 0.5$: FID 5.24 (최적)
- $\mu_S = 0.9$: FID 6.04
- $\mu_S = 0.99$: FID 7.61

낮은 $\mu_S$ 값은 불안정한 자기-교사 신호를 야기하고, 높은 값은 느린 수렴을 초래합니다.[1]

#### 추론 시간 EMA 모멘트($\mu_I$) 영향

표 4의 결과:[1]
- $\mu_I = 0.999$: FID 6.91
- $\mu_I = 0.9999$: FID 5.5
- $\mu_I = 0.99995$: FID 5.24 (최적)
- $\mu_I = 0.99999$: FID 8.73

$\mu_I$는 신중하게 조정해야 하며, 논문에서는 $\epsilon = 10^{-4}$ 휴리스틱을 제안합니다: $\mu_I = 1 - (1/N)^{\epsilon}$, 여기서 N은 학습 단계 수입니다.[1]

#### 증류 단계 수의 영향

증류 단계가 많을수록 성능이 저하됩니다:[1]

| 증류 스케줄 | 단계 수 | 전체 학습 길이 | 1단계 FID |
|-----------|--------|---------------|----------|
| 1024 → 1 | 1 | 96M | 14.40 |
| 1024 → 32 → 1 | 2 | 96M | **5.24** |
| 4096 → 256 → 16 → 1 | 3 | 96M | 6.06 |
| 4096 → 512 → 64 → 8 → 1 | 4 | 96M | 7.27 |

이는 목적 함수 퇴화 가설을 강력히 지지합니다. 2단계가 최적임을 보여줍니다.[1]

### 4.3 아키텍처 일반화

더 작은 아키텍처로의 증류도 시도되었습니다:[1]
- 60.0M 파라미터 → 19.4M 파라미터: FID 5.02 → 6.47 (1단계)

### 4.4 생성 샘플의 질적 분석

Algorithm 1의 증류 과정을 통해 초기 노이즈와 최종 생성 결과 간의 결정론적 매핑이 대부분 보존됩니다. 단일 단계 샘플은 다단계 샘플 대비 약간의 이미지 품질 저하를 보이지만, 극적인 속도 향상으로 인한 실용적 이득이 더 큽니다.[1]

## 5. 모델의 일반화 성능 향상 가능성

### 5.1 일반화를 제한하는 요소

**메모리화 대 일반화 전이**: 최근 연구(Kadkhodaie et al., 2024)에 따르면 확산 모델은 학습 데이터 크기가 모델 용량보다 작을 때 메모리화 체제에 진입합니다. 따라서:[2]

- **메모리화 체제**: 모델 용량 >> 학습 데이터 크기일 때, 확산 모델은 경험적 분포를 메모리화합니다.
- **일반화 체제**: 모델 용량이 학습 데이터를 메모리화할 수 없을 때, 새로운 샘플을 생성합니다.

이 전이는 유한 샘플 크기(예: CIFAR-10에서 ~10,000개)에서 발생합니다.[2]

### 5.2 TRACT의 일반화 개선 메커니즘

#### 1. SWA 활용 극대화

TRACT의 핵심 일반화 개선은 공격적인 SWA를 가능하게 한다는 점입니다. 기존 BTD의 log₂(T) 단계 구조는 전체 학습 길이를 분산시켜 고모멘트 EMA를 활용하기 어렵게 합니다. 반면 TRACT는:[1]

- **1-2단계 구조**: 전체 학습 길이를 단축하지 않으면서도 적은 증류 단계를 유지합니다.
- **높은 모멘트 EMA 가능**: $\mu_I = 0.99995$와 같은 높은 값을 사용하여 더 넓은 최솟값을 탐색합니다.[1]

#### 2. 목적 함수 퇴화 해결

목적 함수 퇴화는 이전 단계의 불완전한 학생이 다음 단계의 교사가 되면서 누적되는 오류입니다. TRACT의 자기-교사 메커니즘은:[1]

- 현재 학생의 가중치 EMA를 사용하여 부트스트랩 방식의 증류를 수행합니다.
- 이를 통해 각 증류 위상에서 더 일관성 있는 목표를 제공합니다.

표 5에서 단계 수를 증가시킬 때 성능 저하가 명확하게 관찰되는데, TRACT의 2단계 설정이 이를 최소화합니다.[1]

#### 3. 저차원 유효 파라미터 활용

최근 연구(Karras et al., 2024)에 따르면 일반화 가능한 확산 모델은 저차원 의미론적 부분공간을 학습합니다. TRACT의 구조는:[2][1]

- 시간 단계별로 서로 다른 유효 파라미터를 활용하여 파라미터 중복성을 활용합니다.
- 특정 시간 스텝에서 필요한 파라미터만 선택적으로 학습합니다.

### 5.3 일반화 성능 측정 및 결과

#### 다양한 학습 길이에서의 성능

CIFAR-10에서 학습 길이 변화에 따른 성능:[1]
- 96M 샘플 (TRACT-96M): FID 5.02
- 256M 샘플 (TRACT-256M): FID 4.45

더 긴 학습으로 개선됨을 보여줍니다. 이는 SWA를 통한 더 나은 일반화 달성을 시사합니다.[1]

#### 아키텍처 이동성(Architecture Transfer)

표 9의 지식 증류 결과:[1]
- TRACT 1024 → 32 → 1 (60M): FID 5.02
- TRACT 1024 → 32 → 32 → 1 (19.4M, 336M 샘플): FID 6.47

더 작은 아키텍처로도 합리적인 성능을 유지하여 도메인 간 일반화 가능성을 시사합니다.[1]

#### 분포 외 일반화 가능성

TRACT의 구조적 특성이 분포 외 일반화를 향상시킬 가능성:[2]

- **저차원 매니폴드 학습**: 실제 이미지 데이터셋은 고차원이지만 저차원 매니폴드에 위치합니다. TRACT의 효율적인 시간 증류는 이러한 저차원 구조를 더 잘 포착할 수 있습니다.
- **모드 커버리지 개선**: 자기-교사 메커니즘이 더 다양한 생성 경로를 탐색하도록 장려합니다.

## 6. 한계 및 개선 가능 영역

### 6.1 확인된 한계

1. **단일 단계에서의 모드 손실**: 1단계 생성에서 완전한 분포를 포착하기 어려워 일부 모드가 손실될 수 있습니다.[1]

2. **자기-교사의 감소된 효율성**: 표 4.4의 결과에 따르면, 동일 스케줄에서 BTD(감독 학습)이 TRACT(자기-교사)보다 더 나은 성능을 보입니다:[1]
   - BTD (1024 → 512 → ... → 1, 10 위상): FID 5.95
   - TRACT (동일 스케줄): FID 6.8

이는 자기-교사 목표가 감독 학습보다 덜 효율적일 수 있음을 시사합니다.[1]

3. **초기 단계에서의 기울기 소실**: 단일 위상 증류(T: 1024 → 1)에서 FID 14.40의 매우 나쁜 결과는 긴 시간 단계 체인에서 기울기 소실과 유사한 현상이 발생함을 시사합니다.[1]

4. **하이퍼파라미터 민감도**: EMA 모멘트 $\mu_I$에 매우 민감하며, 신중한 튜닝이 필요합니다.[1]

### 6.2 향후 연구 방향

논문에서 제시된 개선 가능 영역:[1]

1. **더 높은 단계 수 교사 활용**: TRACT의 임의적 단계 감소로 인해 기존 방법이 활용하지 못한 8192 단계 이상의 교사 모델을 사용할 수 있는 가능성.[1]

2. **다른 도메인으로의 확장**: 현재까지 이미지 데이터셋에만 한정되어 있으며, 비디오, 오디오, 텍스트 등 다른 도메인으로의 적용은 향후 과제입니다.[1]

3. **이종 아키텍처 사이의 증류**: Transformer와 CNN 간의 크로스 아키텍처 증류 가능성 탐색이 필요합니다.[1]

## 7. 관련 최신 연구 동향 (2020년 이후)

### 7.1 핵심 관련 연구들

#### Progressive Distillation (Salimans & Ho, 2022)[4]
- TRACT의 직접적인 선행 작업
- 단계별로 2배씩 샘플링 단계 감소
- CIFAR-10에서 4단계로 FID 3.0 달성
- log₂(T) 단계 필요 (TRACT는 1-2 단계로 개선)

#### DDIM (Song et al., 2021)[6]
- TRACT의 기초가 되는 결정론적 샘플링 프레임워크
- DDPM의 변형으로 샘플링 단계 대폭 감소
- 확산 모델의 ODE 해석 기여

#### EDM: Elucidating the Design Space of Diffusion Models (Karras et al., 2022)[8]
- TRACT-EDM 변형의 기반
- Variance Exploding (VE) 노이즈 스케줄 및 Runge-Kutta 샘플러 도입
- 교사 모델로 사용되어 더 높은 FID 달성 가능

#### EM Distillation (2024)[10]
- 최대 우도 기반 일단계 증류
- EM 프레임워크를 통한 분포 매칭
- TRACT보다 이론적 근거 제공

#### SCott: Stochastic Consistency Distillation (2025)[11]
- 확률적 일관성 증류로 2-4 단계에서 고품질 생성
- TRACT과 유사한 목표이지만 다른 방법론

#### Multistep Distillation via Moment Matching (Salimans et al., 2024)[12]
- 모멘트 매칭을 통한 다단계 증류
- 분포 기반 증류 방식으로 성능 우수
- 8단계 이상에서 SOTA 달성

#### SDXL-Lightning (2024)[13]
- 고해상도(1024px) 이미지 생성을 위한 진행 증류
- 대규모 모델에 TRACT 원리 적용

### 7.2 일반화 관련 최신 연구

#### On the Generalization Properties of Diffusion Models (2024)[15]
- 확산 모델의 메모리화 대 일반화 전이 분석
- 모드 시프트와 일반화 성능 관계 규명
- TRACT의 일반화 개선 이론적 기반 제공

#### Generalization in Graph Neural Networks with Diffusion (2023)[16]
- 그래프 신경망에서 확산의 안정성 측정
- Hessian 기반 일반화 경계 제공

#### What's in a Latent? Leveraging Diffusion Latent Space for Domain Generalization (2025)[17]
- 확산 모델의 잠재 공간이 도메인 불변 특징 포착
- 도메인 외 일반화 4% 이상 개선

#### Efficient Diffusion Models: A Comprehensive Survey (2024)[18]
- 효율적 확산 모델의 종합 분석
- 증류, ODE 솔버, 아키텍처 최적화 포함

### 7.3 일반적 증류 및 가중치 평균화 기술

#### Stochastic Weight Averaging (Izmailov et al., 2018)[20]
- TRACT의 SWA 기초
- 더 넓은 최솟값 탐색 원리

#### Analyzing and Improving Training Dynamics of Diffusion Models (Karras et al., 2024)[21]
- EMA 파라미터의 사후(post-hoc) 설정 방법
- 전력 함수 기반 EMA 휴리스틱 제시

#### BELAY: Damped Harmonic Averaging (Patsenker et al., 2023)[22]
- EMA의 물리학 기반 해석
- 스프링-질량 시스템 유추로 안정성 분석

## 8. 앞으로의 연구에 미치는 영향 및 고려사항

### 8.1 학술적 영향

#### 1. 효율적 생성 모델의 패러다임 전환
TRACT는 확산 모델이 실시간 응용에 진입할 수 있는 길을 열었습니다. 단일 단계에서 합리적인 FID를 달성하는 것은:[1]
- 엣지 디바이스 배포 가능성
- 인터랙티브 생성 애플리케이션 가능
- 비용 제약이 있는 환경에서의 활용 증대

#### 2. 증류 방법론의 혁신
자기-교사 메커니즘은:[1]
- 단순한 부트스트랩 방식으로 복잡한 목표를 달성
- 다른 생성 모델(VAE, GAN)에도 적용 가능한 원리 제시
- 나중의 연구(SCott, EMD 등)의 영감 제공

#### 3. 일반화 이론 발전
TRACT의 경험적 결과가:[2]
- 확산 모델의 메모리화-일반화 전이 이해 촉진
- 저차원 매니폴드 학습의 중요성 강조
- EMA와 일반화 성능 관계 규명

### 8.2 응용적 영향

#### 1. 실시간 이미지 생성
- 모바일 장치에서의 고품질 생성 가능
- 대화형 AI 애플리케이션(채팅 기반 이미지 생성)
- 비디오 편집 도구에서의 즉시 피드백

#### 2. 의료 이미징 응용[24]
- TRACT 원리를 의료 이미지 합성에 적용
- 저선량 CT 이미지 개선 및 빠른 추론

#### 3. 조건부 생성 모델
- 텍스트-이미지 생성에서의 속도 향상[13]
- 클래스 조건부 생성의 효율화

### 8.3 향후 연구 시 고려할 점

#### 1. 이론적 분석의 필요성
현재 TRACT는 주로 경험적 결과에 기반합니다. 향후 연구에서는:[1]
- 자기-교사 메커니즘의 수렴성 보장 분석
- 증류 오류의 상한 유도
- EMA 모멘트 선택의 이론적 근거 제시

#### 2. 하이퍼파라미터 자동 선택
$\mu_I = 1-(1/N)^{\epsilon}$ 휴리스틱이 제시되었지만:[1]
- 다양한 아키텍처 및 데이터셋에서의 보편성 검증 필요
- 적응적 $\mu_I$ 선택 메커니즘 개발
- 학습 길이에 따른 최적값 자동 계산

#### 3. 멀티모달 및 조건부 생성으로의 확장
현재 결과는 주로 무조건부 이미지 생성입니다. 향후에는:[1]
- 클래스 조건 TRACT 최적화
- 텍스트 조건 생성의 효율화
- 기하학적 제약(예: 특정 객체 위치)이 있는 조건부 생성

#### 4. 어댑티브 아키텍처
TRACT의 1-2단계 구조는 새로운 아키텍처 설계를 제시합니다:[1]
- 시간 단계별로 다른 깊이/폭의 네트워크
- 부분적 정밀도(mixed precision) 기반 설계
- 동적 계산 그래프

#### 5. 안정성 및 견고성
고성능에도 불구하고 단일 단계 생성에서:[1]
- 특정 프롬프트나 객체에 대한 성능 저하 분석
- 적대적 예제에 대한 견고성
- 도메인 이동(domain shift) 시 성능 변화

#### 6. 비증류 기반 고속 샘플링과의 통합
TRACT 외에도 여러 고속 샘플링 방법이 개발 중입니다:[18]
- 고차 ODE 솔러(예: Runge-Kutta, DPM-Solver)와의 결합
- 일관성 모델(Consistency Model)과의 앙상블
- 확률적 다중-단계와의 하이브리드 접근

### 8.4 거시적 연구 방향

#### 1. 통합 프레임워크 개발
현재 여러 가속화 방법들이:[18]
- 증류 기반 (TRACT, EDM, EMD)
- ODE 솔버 기반 (DPM-Solver, Heun's method)
- 일관성 기반 (Consistency Model)

이들을 통합하는 프레임워크 개발이 필요합니다.

#### 2. 규모별 성능 특성화
TRACT의 성능이:[1]
- 모델 크기(60M vs 296M)
- 데이터 크기(96M vs 256M vs 1.2B)
- 해상도(CIFAR-10 vs 64×64 ImageNet)

에 따라 어떻게 변하는지 체계적 분석.

#### 3. 비전 너머로의 확장
현재 이미지 중심이지만:[1]
- 3D 생성 모델 증류
- 비디오 생성에서의 TRACT 적용 (시간차원 추가)
- 음성/음악 합성에서의 효율화

## 결론

TRACT는 시간 증류의 패러다임을 근본적으로 변경하여 **단일 단계 확산 기반 생성을 현실화**했습니다. BTD의 목적 함수 퇴화와 일반화 문제를 자기-교사 메커니즘과 공격적인 SWA를 통해 우아하게 해결하며, 동시에 이론적으로도 의미 있는 개선을 제시합니다.

특히 **일반화 성능 측면에서**:[2][1]
- 감소된 증류 단계로 더 긴 학습 시간 확보
- 고모멘트 EMA로 더 넓은 최솟값 탐색
- 저차원 유효 파라미터 활용

등을 통해 단순히 속도만 아니라 품질도 향상됩니다.

향후 연구는:[2][1]
1. 이론적 기초 강화 (수렴성, 오류 경계)
2. 다중 모달 및 고해상도 생성 확장
3. 다른 생성 모델로의 일반화
4. 하이퍼파라미터 자동화

에 초점을 맞춰야 하며, TRACT가 제시한 자기-교사 및 이행적 폐포 개념은 향후 효율적 생성 모델 설계의 핵심 원리로 작용할 것으로 예상됩니다.

***

## 참고 자료

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/32dfb82d-567e-4005-9b16-817a55e64aed/2303.04248v1.pdf)
[2](https://www.siam.org/publications/siam-news/articles/generalization-of-diffusion-models-principles-theory-and-implications/)
[3](https://arxiv.org/abs/2202.00512)
[4](https://www.frontiersin.org/articles/10.3389/fdgth.2025.1653369/full)
[5](https://arxiv.org/html/2412.00665v1)
[6](https://iopscience.iop.org/article/10.1149/MA2025-031244mtgabs)
[7](https://huggingface.co/learn/diffusion-course/en/unit2/2)
[8](https://arxiv.org/abs/2510.10807)
[9](https://arxiv.org/html/2405.16852v2)
[10](https://ojs.acad-pub.com/index.php/ADECP/article/view/3728)
[11](https://arxiv.org/html/2403.01505)
[12](https://proceedings.neurips.cc/paper_files/paper/2024/file/3f66d5cdbe032bb750f2dc523357b7a5-Paper-Conference.pdf)
[13](https://arxiv.org/pdf/2402.13929.pdf)
[14](https://arxiv.org/pdf/2311.01797.pdf)
[15](https://arxiv.org/pdf/2312.06899.pdf)
[16](https://arxiv.org/abs/2302.04451)
[17](http://arxiv.org/pdf/2503.06698.pdf)
[18](http://arxiv.org/pdf/2410.11795.pdf)
[19](https://apxml.com/courses/advanced-diffusion-architectures/chapter-1-diffusion-foundations-advanced-noise/ddim-recap)
[20](https://arxiv.org/pdf/2409.03550.pdf)
[21](https://openaccess.thecvf.com/content/CVPR2024/papers/Karras_Analyzing_and_Improving_the_Training_Dynamics_of_Diffusion_Models_CVPR_2024_paper.pdf)
[22](https://www.emergentmind.com/topics/learnable-weight-averaging-mechanism)
[23](https://ieeexplore.ieee.org/document/10268250/)
[24](https://icml.cc/virtual/2025/oral/47231)
[25](http://pubs.rsna.org/doi/10.1148/radiol.240238)
[26](https://arxiv.org/html/2311.14028v2)
[27](https://arxiv.org/abs/2507.02686)
[28](https://pubs.aip.org/pof/article/37/11/117119/3371491/Fine-structure-investigation-of-turbulence-induced)
[29](https://pubs.aip.org/pof/article/37/11/117120/3371493/Fine-structure-investigation-of-turbulence-induced)
[30](https://iopscience.iop.org/article/10.1149/MA2025-031223mtgabs)
[31](https://drpress.org/ojs/index.php/ajst/article/view/31992)
[32](https://arxiv.org/pdf/2304.04262.pdf)
[33](https://arxiv.org/pdf/2202.00512.pdf)
[34](https://openaccess.thecvf.com/content/CVPR2023/papers/Meng_On_Distillation_of_Guided_Diffusion_Models_CVPR_2023_paper.pdf)
[35](https://arxiv.org/pdf/2202.00512v2.pdf)
[36](https://huggingface.co/docs/diffusers/v0.17.1/en/api/schedulers/ddim)
[37](https://openreview.net/pdf?id=TIdIXIpzhoI)
[38](https://arxiv.org/html/2502.08364v1)
[39](https://www.reddit.com/r/StableDiffusion/comments/10ya8b9/so_ddim_is_the_best_sampler/)
[40](https://arxiv.org/abs/2302.13335)
[41](https://arxiv.org/abs/2308.10510)
[42](https://ieeexplore.ieee.org/document/10804854/)
[43](https://dl.acm.org/doi/10.1145/3625687.3625798)
[44](https://ieeexplore.ieee.org/document/10376944/)
[45](https://arxiv.org/abs/2310.04414)
[46](https://ieeexplore.ieee.org/document/10484417/)
[47](https://arxiv.org/abs/2304.04774)
[48](http://arxiv.org/pdf/2412.17162.pdf)
[49](https://arxiv.org/pdf/2502.12154.pdf)
[50](https://aclanthology.org/2023.acl-long.248.pdf)
[51](http://arxiv.org/pdf/2411.15199.pdf)
[52](https://openaccess.thecvf.com/content/CVPR2024/papers/Yin_One-step_Diffusion_with_Distribution_Matching_Distillation_CVPR_2024_paper.pdf)
[53](https://www.cns.nyu.edu/pub/lcv/kadkhodaie24a.pdf)
[54](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhu_DiMO_Distilling_Masked_Diffusion_Models_into_One-step_Generator_ICCV_2025_paper.pdf)
[55](https://arxiv.org/html/2209.00796v15)
[56](https://openreview.net/pdf/e62641364c927f7a8d5ccab9c3ada448f18e525a.pdf)
[57](https://arxiv.org/html/2502.14123v1)
[58](https://academic.oup.com/nsr/article/11/12/nwae348/7810289)
[59](https://hnry.li/assets/pdf/ema.pdf)
