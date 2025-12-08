# Fast Sampling of Diffusion Models via Operator Learning

### 1. 핵심 주장과 주요 기여

본 논문(ICML 2023)의 **핵심 주장**은 신경 연산자(Neural Operator)를 활용하여 확산 모델의 샘플링을 획기적으로 가속화할 수 있다는 것입니다. 기존의 확산 모델은 고품질의 샘플을 생성하기 위해 수백에서 수천 번의 네트워크 평가가 필요하여 실시간 응용에 부적합했지만, 이 논문은 **단 하나의 모델 평가**로 고품질의 이미지를 생성할 수 있는 방법을 제시합니다.[1]

**주요 기여**는 다음과 같습니다:[1]

- **신경 연산자 기반의 확산 모델 샘플링(DSNO)** 제안: 가우시안 분포로부터 연속시간 확산 궤적으로의 매핑을 학습하는 신경 연산자 개발

- **푸리에 공간에서 매개변수화된 시간 컨볼루션 블록** 도입: 기존 확산 모델 아키텍처에 쉽게 통합 가능하며 모델 크기 증가는 10% 수준으로 제한

- **첫 번째 병렬 디코딩 방법** 제안: 비순차적으로 궤적의 서로 다른 시간 위치에서 이미지를 생성할 수 있는 혁신적인 방식

- **최첨단 성능**: CIFAR-10에서 FID 3.78, ImageNet-64에서 FID 7.83 달성 (단일 모델 평가 설정)

***

### 2. 해결하고자 하는 문제

**근본적인 문제**: 확산 모델은 높은 품질의 생성 결과를 얻지만, 역확산 과정을 수치적으로 해결하기 위해 과도한 계산량이 필요합니다. 확률 흐름 ODE(Probability Flow ODE)의 특성상 정확한 근사를 위해 많은 이산화 단계가 필요하며, 이는 실시간 응용(AI 예술, 설계, 의사결정 생성 모델 등)에 장애물입니다.[1]

기존의 해결 시도들:
- **학습 없는 샘플링**: DDIM, DPM-solver 등은 이산화 단계 수를 줄이려 하지만 여전히 10~30회의 모델 평가 필요
- **학습 기반 방법**: Progressive Distillation은 4단계로 축소했으나 여전히 순차적 특성을 벗어나지 못함

***

### 3. 제안하는 방법 및 수식

#### 3.1 확률 흐름 ODE의 수학적 배경

확산 모델은 다음의 확률 흐름 ODE로 표현됩니다:[1]

$$\frac{dx}{dt} = f(x, t)dt - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x) dt$$

여기서 점수 함수 $\nabla_x \log p_t(x)$는 신경망으로 근사되며, 반선형 형태의 ODE를 얻기 위해 $f(x,t) = h(t)x$로 표현하면:

$$x(t) = \phi(t,s)x(s) - \int_t^s \phi(t,\tau)g(\tau)^2/2 \nabla_x \log p_\tau(x) d\tau$$

여기서 $\phi(t,s) = \exp(\int_t^s h(\tau)d\tau)$입니다.[1]

#### 3.2 신경 연산자의 기본 구조

푸리에 신경 연산자(FNO)는 다음과 같이 정의됩니다:[1]

$$G_\theta := Q \circ \sigma(W_L + K_L) \circ \cdots \circ \sigma(W_1 + K_1) \circ P$$

적분 커널 연산자 $K$는 푸리에 공간에서:

$$K_i v_i(t) = \mathcal{F}^{-1}(R_i \cdot (\mathcal{F}v_i))(t)$$

여기서 $R_i$는 푸리에 공간의 학습 가능한 매개변수이고, $\mathcal{F}$와 $\mathcal{F}^{-1}$은 푸리에 변환 및 역변환입니다.[1]

#### 3.3 시간 컨볼루션 블록

제안된 시간 컨볼루션 레이어는:

$$(Tu)(t) = u(t) + \sigma((Ku)(t))$$

여기서 $\sigma$는 점 방향 비선형 함수이고, $K$는 푸리에 컨볼루션 연산자입니다. 컨볼루션 정리에 의해:[1]

$$(Ku)(t) = \int_D (\mathcal{F}^{-1}R)(\tau)u(t-\tau)d\tau$$

이는 방정식 (3)의 가중 적분 형태와 구조적 유사성을 갖습니다.

#### 3.4 이산화 및 구현

시간 영역 $D$가 $M$개 점 $\{t_1, ..., t_M\}$으로 이산화될 때, 입력함수 $u(t)$는 $\mathbb{R}^{M \times d}$ 텐서로 표현되고, $R \in \mathbb{C}^{J \times d \times d}$입니다 ($J$는 최대 모드 수). 점곱 연산은:[1]

$$R \cdot (\mathcal{F}u)_{j,k} = \sum_{l=1}^d R_{j,k,l}(\mathcal{F}u)_{j,l}$$

#### 3.5 학습 목적함수

DSNO의 학습은 가중 적분 형태의 목적함수를 최소화합니다:[1]

$$\min_\theta \mathbb{E}_{x_T \sim \mathcal{N}(0,I)} \int_D \lambda(t) \|G_\theta(x_T)(t) - G^*(x_T)(t)\| dt$$

실제 구현에서는 경험적 위험을 최소화합니다:

$$\min_\theta \frac{1}{N}\sum_{j=1}^N \frac{1}{M}\sum_{i=1}^M \lambda(t_i) \|G_\theta(x_T^{(j)})(t_i) - G^*(x_T^{(j)})(t_i)\|$$

여기서 $\lambda(t) = \sqrt{\alpha_t/\sigma_t}$는 SNR 가중 함수의 제곱근이며, 작은 시간에 더 많은 가중치를 할당합니다.[1]

***

### 4. 모델 구조

#### 4.1 DSNO 아키텍처

DSNO는 기존 확산 모델(U-Net 백본)에 시간 컨볼루션 블록을 추가하는 설계입니다:[1]

- **파란색 블록** (기존 구조): 픽셀-채널 차원 $(C \times H \times W)$에서 작동
- **노란색 블록** (시간 컨볼루션): 시간-채널 차원 $(M \times C)$에서 작동
- 시간 영역과 공간 영역이 분리되어 매우 병렬화 가능

#### 4.2 병렬 디코딩의 원리

핵심은 **조건부 독립성**입니다: 초기 조건 $x(T)$가 주어질 때, ODE 궤적의 서로 다른 시간에서의 해는 조건부 독립입니다. 따라서:[1]

- 푸리에 계수 $R \cdot \mathcal{F}u$는 모든 $t_i$에서 동일
- 역푸리에 변환을 모든 시간점에서 병렬로 계산 가능
- 다른 모듈들은 시간 차원을 배치 차원처럼 처리하여 병렬 처리 가능

#### 4.3 컴팩트 파워 스펙트럼

중요한 발견: 확산 ODE 궤적은 시간 차원에서 컴팩트한 에너지 스펙트럼을 가집니다. 이는:[1]
- 고주파 모드가 학습 목표에 큰 기여하지 않음을 의미
- 푸리에 신경 연산자가 적은 수의 이산화 단계로 궤적을 효율적으로 모델링 가능하게 함

***

### 5. 성능 향상 및 한계

#### 5.1 성능 향상

**정량적 성과**:[1]

| 데이터셋 | 방법 | 평가 횟수 | FID |
|---------|------|---------|-----|
| CIFAR-10 | DSNO | 1 | **3.78** |
| CIFAR-10 | Progressive Distillation | 4 | 3.00 |
| CIFAR-10 | Progressive Distillation | 2 | 4.51 |
| ImageNet-64 | DSNO | 1 | **7.83** |
| ImageNet-64 | Progressive Distillation (1단계) | 1 | 15.99 |

**속도 개선**:[1]

- CIFAR-10에서 4단계 Progressive Distillation 대비 **2.6배 빠름**
- ImageNet-64에서 Progressive Distillation 대비 **1.7배 빠름**
- 모델 크기 증가: 기존 대비 10% 수준 (예: CIFAR-10에서 60M → 65.8M)

#### 5.2 절제 연구 결과

**시간 컨볼루션 블록의 중요성**:[1]

| 훈련 단계 | U-Net 만 | U-Net + 시간 컨볼루션 |
|----------|---------|----------------------|
| 300k | 8.09 | **4.23** |
| 400k | 7.85 | **4.12** |

**시간 해상도의 영향**:[1]

| 시간 해상도 | 2 | 4 | 8 |
|-----------|---|---|---|
| FID | 5.01 | **4.21** | 3.98 |

- 해상도 4에서 8로 증가시 이득은 미미하므로, 계산 효율성을 고려하면 4가 최적

**손실 함수 선택**:[1]

| 손실 함수 | ℓ1 | LPIPS |
|---------|-----|-------|
| FID | 4.12 | **3.78** |

#### 5.3 한계

**직접적인 한계**:

1. **유도된 샘플링의 한계**: 제안 모델이 사전 학습된 확산 모델의 궤적으로부터 생성한 데이터로 학습되므로, 교사 모델의 품질에 의존[1]

2. **조건부 생성의 확장성**: 논문은 주로 무조건부 및 클래스-조건부 생성만 다루며, 텍스트-이미지 같은 복잡한 조건부 생성은 더 큰 데이터셋과 모델 크기 필요[1]

3. **고해상도 생성의 확장성**: ImageNet-64 수준에서만 검증되었으며, 1024x1024 이상의 고해상도 확장 가능성 미검증

4. **일반화 성능의 의문**: Temporal resolution 4로 학습한 모델이 resolution 8에서 완벽하지 않은 성능을 보임 (부록 A.4)[1]

***

### 6. 모델의 일반화 성능 향상 가능성

#### 6.1 현재의 일반화 능력

**적응적 해상도 능력**: 논문의 A.4 섹션에서 언급하듯, 시간 해상도 4로 학습한 DSNO가 해상도 8에서 예측할 때 만족스럽지 못한 궤적을 생성합니다. 이는 신경 연산자의 **이산화 불변성(Discretization Invariance)**이 완전히 달성되지 않음을 시사합니다.[1]

#### 6.2 일반화 성능 향상 방안

**이론적 근거**:

신경 연산자의 보편적 근사 정리(Proposition 3.1)에 따르면, 제안된 아키텍처는 확산 ODE의 해 연산자를 임의로 잘 근사할 수 있습니다. 따라서 다음과 같은 개선이 가능합니다:[1]

1. **다양한 시간 해상도에서의 학습**: 단일 해상도가 아닌 여러 해상도의 궤적으로 학습하면 해상도-불변 표현 학습 가능

2. **푸리에 모드 수의 동적 조정**: 컴팩트 파워 스펙트럼 성질을 활용하여, 필요한 푸리에 모드 수를 적응적으로 조정하는 메커니즘 구현

3. **메타 학습(Meta-Learning) 접근**: 다양한 확산 모델과 데이터셋에서의 학습으로 보편적인 시간 상관관계 패턴을 학습하는 것으로 전이 학습 성능 향상 가능

4. **잠재 공간에서의 확장**: 전체 픽셀 공간이 아닌 잠재 확산 모델에 적용하면 계산 효율성 증대로 더 큰 모델 학습 가능

#### 6.3 구조적 개선 가능성

**수학적 최적화**:

$$\min_\theta \mathbb{E}_{x_T} \int_D \lambda(t) \|G_\theta(x_T)(t) - G^*(x_T)(t)\|_{\text{Sobolev}} dt$$

Sobolev 노름 도입으로 함수 공간에서의 수렴 속도 개선이 가능합니다.[1]

**아키텍처 개선**:
- 적응적 푸리에 커널 크기
- 다중 스케일 시간 컨볼루션 블록
- 잔여 연결의 강화를 통한 심층 신경 연산자 설계

***

### 7. 앞으로의 연구에 미치는 영향

#### 7.1 이론적 영향

1. **연산자 학습 패러다임의 확대**: 신경 연산자가 PDE 솔버뿐만 아니라 생성 모델 가속화에도 강력함을 보였으며, 이는 연산자 학습의 응용 범위를 크게 확장

2. **확산 ODE의 구조적 이해 심화**: 시간 차원의 컴팩트 스펙트럼 발견은 확산 과정의 본질적 특성을 이해하는 데 기여

3. **병렬 디코딩의 일반화**: 시간 차원에서의 병렬 디코딩이 최초 시도로서, 이후 다른 생성 모델(플로우 매칭, 정규화 흐름 등)으로의 확장 가능성 제시

#### 7.2 실용적 영향

1. **실시간 생성 모델의 가능성**: 단일 평가로 고품질 샘플 생성으로 인터랙티브 애플리케이션(AI 아트, 디자인 도구) 활성화

2. **에지 디바이스 배포**: 10% 수준의 모델 크기 증가와 단일 평가로 모바일 등 저사양 환경에서의 실용 가능성 증대

3. **다중 모드 생성 모델의 확장**: 제안 방법이 모델 아키텍처에 무관하게 적용 가능하므로, 음성, 비디오, 3D 등 다양한 영역으로 확장 가능

***

### 8. 2020년 이후 관련 최신 연구

#### 8.1 일관성 모델(Consistency Models)

**핵심 기여**: Song et al. (2023)이 제시한 일관성 모델은 Score-based 생성 모델의 확률 흐름 ODE의 **일관성 함수**를 학습하여 단일 단계로 고품질 샘플 생성을 실현합니다.[2][3][4]

- **성과**: CIFAR-10에서 FID 3.55, ImageNet 64×64에서 FID 6.20 달성 (단일 단계)
- **DSNO와의 차이**: DSNO는 전체 궤적을 학습하는 반면, 일관성 모델은 임의 시간점 간의 점프를 학습
- **개선 방향**: Multistep Consistency Models (2024)은 속도와 품질의 트레이드오프 가능하게 개선[3][5]

#### 8.2 플로우 매칭(Flow Matching)

**개념**: Lipman et al. (2022)의 플로우 매칭은 확산 과정을 일반화하여 최적 수송(Optimal Transport) 경로를 사용하는 생성 모델입니다.[6][7]

- **장점**: 확산보다 효율적인 경로, 더 빠른 훈련과 샘플링
- **최신 발전**: NeurIPS 2024 튜토리얼에서 다양한 도메인(비유클리드 기하학, 이산 영역) 확장 제시[8]
- **DSNO와의 시너지**: 신경 연산자 원리를 플로우 매칭에 적용하면 추가 가속 가능

#### 8.3 고급 수치 해석 방법

**DPM-Solver (2022)** 및 **DEIS (2022)**[9][10]

- 확산 ODE의 반선형 특성을 활용한 고차 수치 적분 기법
- 10-20 단계로 고품질 샘플 생성 가능
- DSNO의 데이터 생성에 사용되는 기초 방법

**Exponential Integrator (2024)**[10]

- 지수적 적분 개념 활용
- 100 NFE에서 FID 2.36 달성 (CIFAR-10)

#### 8.4 증류 기반 방법의 진화

**Progressive Distillation의 한계 극복**:

1. **Distribution Matching Distillation (DMD, 2024)**: 다단계 확산 출력의 분포 매칭으로 단일 단계 생성 최적화[11]
   - ImageNet 64×64에서 FID 2.62 달성

2. **Latent Consistency Models (LCM, 2024)**: 잠재 공간에서의 일관성 모델로 텍스트-이미지 생성 가속[2]
   - PIXART-δ: 1024×1024 이미지를 0.5초에 생성 (7배 개선)

3. **Easy Consistency Tuning (ECT, 2024)**: 사전 학습된 확산 모델에서 효율적으로 일관성 모델 미세조정[12]
   - CIFAR-10에서 2단계 FID 2.73 달성 (1시간, A100 1개 GPU)

#### 8.5 특화된 도메인별 가속

**의료 영상** (2023-2024):
- DiffCMR: 조건부 디노이징 확산 모델로 심장 MRI 고속 재구성[13]

**분자 생성** (2024):
- Equivariant Latent Progressive Distillation: 분자 구조 생성에 7.5배 속도 개선[14][15]

**영상-3D 변환** (2025):
- GECO: 1초 이내 고품질 3D 생성으로 실시간 처리 가능[16]

#### 8.6 이론적 기초 연구

**일관성 모델의 수렴 보장 (2023)**:[4]

Consistency Models의 수렴 속도 분석으로 이론적 근거 제공. Multistep sampling으로 오류 감소 가능함을 증명.

**적응형 타임스텝 샘플링 (2025)**:[17]

확산 모델의 학습 중에도 타임스텝별 그래디언트 분산에 기반한 비균일 샘플링으로 수렴 가속화.

**신경 연산자의 PDE 응용 (2025)**:[18]

위상장 모델링에서 신경 연산자(U-AFNO)로 50,000 타임스텝을 한 번에 점프 가능하게 하는 성과.

***

### 9. 앞으로 연구 시 고려할 점

#### 9.1 이론적 고려사항

1. **일반화 이론의 강화**
   - 현재 신경 연산자의 일반화 한계(해상도 외삽)를 수학적으로 분석
   - Rademacher 복잡도나 VC 차원 관점에서 표본 복잡도 경계 유도

2. **최적 네트워크 설계**
   - 푸리에 모드 수 $J$와 시간 해상도 $M$의 최적 관계식 도출
   - 컴팩트 스펙트럼 성질의 정량적 특성화

#### 9.2 방법론적 개선

1. **멀티스케일 시간 구조**
   - 다양한 시간 스케일의 특징을 동시에 학습하는 계층적 신경 연산자
   - 조조(Coarse)에서 세밀(Fine) 시간 해상도로의 점진적 개선

2. **적응적 푸리에 표현**
   - 데이터 기반으로 필요한 푸리에 모드를 동적으로 선택
   - 스파스 푸리에 신경 연산자로 계산 효율성 증대

3. **전이 학습 체계**
   - 다양한 데이터셋에서 학습된 시간 상관관계 패턴의 재사용
   - 메타-신경 연산자로 새로운 확산 모델에 빠른 적응

#### 9.3 실제 응용 확장

1. **조건부 생성의 확대**
   - 텍스트-이미지, 이미지 수정, 음성 합성 등 다양한 조건 처리
   - 클래스-조건부에서 벗어나 복합 조건 처리 가능성 연구

2. **고해상도 생성의 실현**
   - 잠재 확산 모델(Stable Diffusion 등)과의 결합
   - 계층적 생성(coarse-to-fine) 방식의 신경 연산자 설계

3. **실시간 시스템 구현**
   - 엣지 디바이스 최적화 (양자화, 가지치기)
   - 배치 처리와 스트리밍 처리의 병합

#### 9.4 신흥 방향

1. **다중 생성 패러다임 통합**
   - 플로우 매칭, 일관성 모델, 정규화 흐름 등 다양한 ODE 기반 생성 모델에 신경 연산자 적용
   - 확산 외 다른 역학 시스템의 고속화

2. **인과관계 구조 활용**
   - 시간 차원에서의 인과관계를 명시적으로 모델링
   - 특정 시간 구간의 결정적 중요도 학습 (Critical Window 개념 활용)[19]

3. **확률론적 확장**
   - 행렬식 점 과정(Determinantal Point Processes)을 이용한 다양성 보장
   - 베이지안 신경 연산자로 불확실성 정량화

#### 9.5 벤치마킹 및 평가

1. **표준화된 평가 체계**
   - FID, Inception Score뿐 아니라 계산 시간, 메모리, 에너지 소비 포함
   - 정성적 평가(사용자 선호도)와 정량적 평가의 균형

2. **일반화 성능의 정밀 분석**
   - 도메인 외 적응(Out-of-Distribution) 평가
   - 작은 데이터셋에서의 성능 평가

***

### 결론

"Fast Sampling of Diffusion Models via Operator Learning"은 신경 연산자의 이론적 강점을 확산 모델의 실제 문제에 적용하여, 단 하나의 모델 평가로 고품질 샘플을 생성하는 획기적 방법을 제시합니다.[1]

**향후 연구의 핵심 방향**은 다음과 같습니다:

1. **이론과 실제의 격차 해소**: 일반화 한계를 극복하고 더 큰 모델과 고해상도로 확장

2. **통합 생성 프레임워크**: 플로우 매칭, 일관성 모델 등 신흥 방법과의 시너지

3. **실용적 배포**: 엣지 장치와 클라우드에서의 효율적 구현으로 실시간 인터랙티브 응용 실현

이 논문은 생성 모델 분야에서 **계산 효율성**이라는 중대한 병목을 해결하는 새로운 관점을 제공하였으며, 2023년 이후 다양한 일관성 모델, 플로우 매칭 기반 방법들과 함께 확산 모델의 실용화를 가속화하는 연쇄 반응을 초래했습니다.[7][20][3][6][2]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a07d5153-4b5b-49b8-8a6b-b5ac12e73807/2211.13449v3.pdf)
[2](https://arxiv.org/abs/2401.05252)
[3](https://arxiv.org/abs/2403.06807)
[4](https://arxiv.org/abs/2308.11449)
[5](https://arxiv.org/html/2403.06807)
[6](https://arxiv.org/abs/2210.02747)
[7](https://openreview.net/forum?id=PqvMRDCJT9t)
[8](https://neurips.cc/virtual/2024/tutorial/99531)
[9](https://arxiv.org/pdf/2302.04867v2.pdf)
[10](https://arxiv.org/pdf/2311.00157.pdf)
[11](https://openaccess.thecvf.com/content/CVPR2024/papers/Yin_One-step_Diffusion_with_Distribution_Matching_Distillation_CVPR_2024_paper.pdf)
[12](https://arxiv.org/abs/2406.14548)
[13](https://arxiv.org/abs/2312.04853)
[14](https://arxiv.org/abs/2404.13491)
[15](http://arxiv.org/pdf/2404.13491.pdf)
[16](https://ieeexplore.ieee.org/document/11141031/)
[17](https://cvpr.thecvf.com/virtual/2025/poster/34841)
[18](https://www.nature.com/articles/s41524-024-01488-z)
[19](https://arxiv.org/abs/2403.01633)
[20](https://openai.com/index/simplifying-stabilizing-and-scaling-continuous-time-consistency-models/)
[21](https://edu.pubmedia.id/index.php/ptk/article/view/1603)
[22](https://ejurnal.stpkat.ac.id/index.php/jutipa/article/view/369)
[23](https://arxiv.org/abs/2402.07211)
[24](https://www.ahajournals.org/doi/10.1161/cir.151.suppl_1.P3006)
[25](https://link.aps.org/doi/10.1103/PhysRevD.110.016030)
[26](https://ieeexplore.ieee.org/document/11198028/)
[27](https://theaspd.com/index.php/ijes/article/view/9120)
[28](http://arxiv.org/pdf/2503.07699.pdf)
[29](https://arxiv.org/html/2410.07761)
[30](https://arxiv.org/pdf/2202.00512.pdf)
[31](https://arxiv.org/html/2402.09970)
[32](https://arxiv.org/abs/2410.12557v1)
[33](https://liner.com/ko/review/fast-sampling-diffusion-models-via-operator-learning)
[34](https://proceedings.mlr.press/v202/zheng23d/zheng23d.pdf)
[35](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/shortcut-model/)
[36](https://proceedings.iclr.cc/paper_files/paper/2025/file/be5ab4915580f581564d326e975235ff-Paper-Conference.pdf)
[37](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_DiffFNO_Diffusion_Fourier_Neural_Operator_CVPR_2025_paper.pdf)
[38](https://arxiv.org/abs/2410.12557)
[39](https://arxiv.org/abs/2401.01008)
[40](https://arxiv.org/abs/2401.02620)
[41](https://arxiv.org/abs/2310.02279)
[42](https://www.semanticscholar.org/paper/9e73a3beffc299ccabedc98512b3dc234d2b0350)
[43](https://www.semanticscholar.org/paper/945a899a93c03eb63be5e3197e318c077473cef9)
[44](https://arxiv.org/abs/2411.01212)
[45](http://arxiv.org/pdf/2310.02279.pdf)
[46](https://arxiv.org/html/2403.01505)
[47](https://arxiv.org/pdf/2402.07802.pdf)
[48](http://arxiv.org/pdf/2410.11081.pdf)
[49](http://arxiv.org/pdf/2311.15736.pdf)
[50](http://arxiv.org/pdf/2310.14189v1.pdf)
[51](http://arxiv.org/pdf/2406.00356.pdf)
[52](https://huggingface.co/papers/2303.01469)
[53](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830001.pdf)
[54](https://dl.acm.org/doi/10.5555/3618408.3619743)
[55](https://proceedings.mlr.press/v235/li24ad.html)
[56](https://arxiv.org/abs/2206.04029)
[57](https://proceedings.neurips.cc/paper_files/paper/2024/file/29d4e09f060a95118762296d240b5e63-Paper-Conference.pdf)
