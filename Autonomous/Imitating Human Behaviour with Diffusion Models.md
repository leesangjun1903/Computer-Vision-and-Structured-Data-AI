
# Imitating Human Behaviour with Diffusion Models

## 1. 논문의 핵심 주장 및 주요 기여

본 논문(Pearce et al., 2023, ICLR 2023)은 **확산 모델(Diffusion Models)을 행동 모방 학습(Behavior Cloning)에 적용**하는 혁신적인 접근법을 제시한다. 핵심 주장은 다음과 같다:

기존의 행동 모방 학습 방법들(MSE, 이산화, K-Means 등)은 간단한 학습을 위해 행동 분포에 대한 근본적인 근사를 도입하는데, 이는 **다중모달성(multimodality), 작용 차원 간의 상관관계, 확률적 행동 분포 표현** 같은 인간 행동의 본질적인 특성을 충분히 포착하지 못한다는 문제가 있다. 

본 논문은 최근 이미지 생성 분야에서 성공한 확산 모델이 이러한 제한을 극복하는 **자연스러운 해결책**임을 주장한다. 주요 기여는:

1. **비근사적 분포 모델링**: 확산 모델은 결합 행동 공간에서 표현적인 조건부 분포를 학습하며 어떤 정규화 가능한 분포도 표현 가능
2. **순차적 환경 적응**: 관찰-행동 확산 모델을 위한 적절한 아키텍처 설계, 신뢰할 수 있는 샘플링 전략, 분류기 없는 안내(Classifier-Free Guidance) 실패 분석
3. **실무적 성능 향상**: 시뮬레이션 로봇 제어와 3D 게임 환경에서 최신 기법 대비 현저한 성능 향상

***

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

**행동 모방의 본질적 문제**: 

기존 행동 모방 기법들의 분포 모델링 한계:

| 방법 | 개념 | 주요 문제 |
|------|------|----------|
| **MSE** | 점 추정(point estimate)으로 평균값만 학습 | 분산·다중모달성 포착 불가, "평균화된" 행동 학습 |
| **이산화** | 각 행동 차원을 B개 빈으로 나누어 분류 | 양자화 오류, 차원 독립 학습으로 "비조직화된" 행동 |
| **K-Means** | 전체 데이터에서 K개 클러스터 생성 후 분류 | K 선택 민감도, 관측치별 분포 무시 |
| **K-Means+Residual** | K-Means에 MSE 잔차 추가 | 여전히 K개 점 추정 제한, 평균화 문제 지속 |

**핵심 통찰**: 인간 행동은 본질적으로 **다중모달성(한 관측에서 여러 최적 행동 존재)**, **구조화된 상관관계(행동 차원 간 의존성)**, **확률적 분포**를 가지는데, 기존 방법들은 이를 적절히 모델링할 수 없다.

### 2.2 제안하는 방법: 확산 모델 기반 행동 모방

#### 2.2.1 확산 모델의 기본 원리

**Denoising Diffusion Probabilistic Models (DDPM)**를 기반으로:

**훈련 프로세스** - 점진적 노이즈 추가:

$$a_\tau = \sqrt{\bar{\alpha}_\tau}a + \sqrt{1-\bar{\alpha}_\tau}z$$

여기서 $\bar{\alpha}_\tau$는 분산 스케줄, $z \sim \mathcal{N}(0, I)$는 무작위 노이즈

**손실 함수**:

$$\mathcal{L}_{DDPM} := \mathbb{E}_{o,a,\tau,z} \left[ \|\epsilon(o, a_\tau, \tau) - z\|_2^2 \right]$$

네트워크는 추가된 노이즈를 예측하도록 학습된다.

**샘플링 프로세스** - 반복적 노이즈 제거:

$$a_{\tau-1} = \frac{1}{\sqrt{\alpha_\tau}}\left(a_\tau - \frac{1-\alpha_\tau}{\sqrt{1-\bar{\alpha}_\tau}}\epsilon(o, a_\tau, \tau)\right) + \sigma_\tau z$$

$T$번의 반복 단계를 통해 $a_T \sim \mathcal{N}(0,I)$에서 시작하여 $a_0$라는 깨끗한 샘플을 생성한다.

#### 2.2.2 분류기 없는 안내(Classifier-Free Guidance) 분석

**CFG 공식**:

$$\hat{z}_\tau = (1+w)\epsilon_{cond.}(a_\tau-1, o, \tau) - w\epsilon_{uncond.}(a_\tau-1, \tau)$$

여기서 $w$는 안내 가중치.

**중요한 발견**: 텍스트-이미지 생성에서 유용한 CFG는 **순차적 행동 모방에서 오히려 해롭다**. CFG는 $p(o|a)$를 최대화하도록 작용하여, **관측에 고유한 드문 행동**을 더 자주 샘플링하도록 유도한다.

**수학적 직관**:
$$p(o|a) = \frac{p(a|o)p(o)}{p(a)}$$

CFG는 이 확률을 증가시키는 행동을 선호하므로, 인간 시연에서 드물지만 특정 관측과만 쌍을 이루는 행동이 과도하게 선택된다.

### 2.3 모델 구조

#### 2.3.1 세 가지 아키텍처 설계

논문은 이미지 생성용 U-Net이 적절하지 않음을 인식하고, **낮은 차원 행동 벡터**에 최적화된 세 가지 아키텍처를 제시:

**1. Basic MLP**
- 입력: $[a_{\tau-1}, o, \tau]$ 단순 연결
- 구조: 3개 은닉층(512유닛), GELU 활성화
- 장점: 간단하고 빠름 (24Hz 샘플링 속도)
- 단점: 관찰 히스토리 처리 제한

**2. MLP Sieve** 
- 관찰 인코더: $o \to o_e$ (임베딩 차원 128)
- 타임스텝 인코더: $\tau \to t_e$ (사인파 위치 인코딩)
- 행동 인코더: $a_{\tau-1} \to a_e$ (임베딩)
- 디노이징 네트워크: $[o_e, t_e, a_e] \to$ 예측된 노이즈
- 특징: 관찰 인코더는 테스트 시간에 한 번만 실행
- 성능: Basic MLP보다 우수 (16Hz)

**3. Transformer**
- 아키텍처: 4개 인코더 블록, 16개 어텐션 헤드
- 입력 토큰: $[o_e, t_e, a_e]$ (히스토리 추가 가능)
- 장점: 최고 성능, 긴 히스토리 처리 가능
- 단점: 샘플링 느림 (4Hz)

### 2.4 신뢰할 수 있는 샘플링 전략

기본 Diffusion BC는 드물게 분포 밖의 행동을 생성할 수 있는 문제가 있어, 두 가지 개선 방법 제시:

**Diffusion-X**:
- 표준 $T$단계 반복 후, 타임스텝을 $\tau=1$로 고정하고 추가로 $M$단계 진행
- 직관: 샘플들이 높은 확률 영역으로 계속 이동
- 효과: 저확률 행동 제거, 다중모달성 유지

**Diffusion-KDE**:
- 여러 행동 샘플($K$개) 생성
- 커널 밀도 추정기(KDE)를 적용하여 각 샘플의 우도 계산
- 최고 우도 행동 선택
- 효과: 더 안정적이지만 계산 비용 증가

***

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 실험 결과: 주요 성능 지표

#### 로봇 제어 환경(Kitchen Task)

| 방법 | 평가 지표 |  |  |  |  |  |
|------|----------|---|---|---|---|---|
| | **4개 작업 완료율** ↑ | **작업 Wasserstein** ↓ | **시간 Wasserstein** ↓ | **상태 Wasserstein** ↓ | **Density** ↑ | **Coverage** ↑ |
| MSE (Transformer) | 0.69±0.02 | 1.47±0.13 | 5.85±0.27 | 0.397±0.034 | 0.81±0.01 | 0.42±0.01 |
| K-Means+Residual (Transformer) | 0.34±0.02 | 2.25±0.16 | 7.80±0.87 | 0.426±0.018 | 0.66±0.02 | 0.38±0.01 |
| **Diffusion BC (Transformer)** | **0.77±0.01** | **1.35±0.11** | **4.11±0.05** | **0.340±0.003** | **0.74±0.01** | **0.44±0.00** |
| **Diffusion-X (Transformer)** | **0.88±0.01** | **1.17±0.13** | **4.65±0.47** | **0.365±0.013** | **0.94±0.02** | **0.45±0.01** |
| **Diffusion-KDE (Transformer)** | **0.89±0.01** | **1.31±0.03** | **5.28±0.41** | **0.418±0.012** | **0.97±0.02** | **0.43±0.01** |

**주요 발견**:
- Diffusion 기반 방법들이 모든 아키텍처에서 우수
- Diffusion-X/KDE는 0.88-0.89의 작업 완료율로 기존 SOTA(0.44)를 **2배 초과** 달성
- Wasserstein 거리 지표들도 인간 시연 분포에 훨씬 가까움

#### 비디오 게임 환경(Counter-Strike: Global Offensive)

| 방법 | Game Score ↑ | 1×timesteps ↓ | 16×timesteps ↓ | 32×timesteps ↓ |
|------|--------------|---------------|---|---|
| MSE | 17.8 | 5.5 | 28.1 | 48.9 |
| K-Means+Residual | 16.8 | 3.8 | 29.2 | 51.8 |
| **Diffusion BC** | **19.0** | **6.3** | **29.5** | **50.4** |
| **Diffusion-X** | **24.0** | **4.5** | **24.5** | **44.4** |
| Human | 36.5 | 0.73 | 0.57 | 0.38 |

### 3.2 일반화 성능 향상 메커니즘

**1. 분포 표현의 완전성**

확산 모델은 임의의 정규화 가능한 분포를 표현할 수 있으므로:
- 다중모달 행동 분포의 모든 모드를 정확히 캡처
- 행동 차원 간 복잡한 비선형 상관관계 학습
- 실제 인간 행동의 "평균화" 또는 "비조직화" 문제 제거

**2. 아키텍처 유연성**

- MLP Sieve와 Transformer는 관찰 히스토리를 효과적으로 처리
- 토큰 기반 처리로 시간적 의존성 모델링 개선
- 어텐션 메커니즘이 행동 차원 간 상관관계 명시적으로 학습

**3. 신뢰할 수 있는 샘플링의 영향**

Diffusion-X와 Diffusion-KDE의 추가 메커니즘:
- 저확률 행동 자동 필터링
- 테스트 시간 동적 샘플 검증
- 분포 밖 상태 진입 위험 감소

### 3.3 일반화 한계 및 도메인 적응성

**인식된 한계**:

1. **단일 타임스텝 제약**: 확산 모델은 각 타임스텝의 행동만 모델링
   - 시간 상의 행동 상관관계는 미처리
   - 순차적 의존성은 인코더의 히스토리를 통해서만 처리

2. **계산 효율성**:
   - MSE: 666 Hz (Kitchen), 200 Hz (CSGO)
   - Diffusion BC: 16 Hz (Kitchen), 32 Hz (CSGO)  
   - Diffusion-KDE: 8 Hz (Kitchen)
   - **약 20-80배 느린 샘플링**

3. **하이퍼파라미터 복잡성**:
   - 분산 스케줄 선택 ($\beta$ 스케줄)
   - 디노이징 스텝 수 $T$ 설정
   - Diffusion-X의 추가 스텝 $M$
   - Diffusion-KDE의 샘플 개수 및 KDE 대역폭

***

## 4. 모델의 분석: 자세한 설명

### 4.1 CFG 실패 분석 - 수치 실험

**Arcade Claw 게임 실험**:

CFG 가중치 변화에 따른 영향:

| Guidance 가중치 $w$ | 4개 작업 완료율 | 첫 작업이 Bottom Burner인 비율 |
|-----------------|------------|----------------------|
| 0.0 (CFG 없음) | 0.63 | 7.3% |
| 1.0 | 0.61 | 12.7% |
| 4.0 | 0.45 | 17.0% |
| 8.0 | 0.08 | 24.7% |
| 인간 시연 | 0.63 | 10.1% |

**해석**: 
- CFG 없을 때: 인간과 유사하게 7.3%에서 Bottom Burner 선택
- CFG $w=8$: 과도하게 24.7%로 증가
- **CFG가 비전형적인 행동을 부당하게 강화**

**수학적 설명** (Appendix E의 격자 세계 예제):

4개 상태와 3개 이동 옵션(좌회전, 우회전, 직진)이 있는 환경:
- 상태 0→1→{2,3}: 경로 선택 지점
- 상태 1에서: 우회전 확률 $p(a=\text{Right}|o_1) = 0.1$ (드물음)

베이즈 규칙 적용:

$$p(o_1|a=\text{Right}) = \frac{p(a=\text{Right}|o_1)p(o_1)}{p(a=\text{Right})} = \frac{0.1 \times 1/3}{0.1 \times 1/3} = 1$$

$$p(o_1|a=\text{Straight}) = \frac{0.9 \times 1/3}{1/3 + 1/3 \times 0.9 + 1/3} = \frac{0.3}{2.9} \approx 0.31$$

CFG는 $p(o|a)$를 최대화하므로, 드물지만 고유한 우회전을 선호하게 된다.

### 4.2 아키텍처 성능 비교

**Kitchen 환경에서의 4개 작업 완료율**:

| 아키텍처 | MSE | 이산화 | K-Means+Residual | **Diffusion BC** | **Diffusion-X** | **Diffusion-KDE** |
|---------|-----|--------|------------|---|---|---|
| Basic MLP | 0.50 | 0.18 | 0.23 | 0.45±0.03 | **0.58±0.02** | **0.59±0.01** |
| MLP Sieve | 0.50 | 0.18 | 0.23 | 0.68±0.02 | **0.77±0.02** | **0.79±0.04** |
| Transformer | 0.69 | 0.34 | 0.34 | 0.77±0.01 | **0.88±0.01** | **0.89±0.01** |

**아키텍처 진화의 이점**:
1. **Basic MLP**: 과적합 위험, 히스토리 활용 미흡
2. **MLP Sieve**: 잔차 연결과 분리된 인코더로 안정성 향상
3. **Transformer**: 명시적 어텐션으로 차원 간 상관관계 모델링

***

## 5. 논문이 미치는 영향과 앞으로의 연구 방향

### 5.1 학술적 영향

**1. 패러다임 전환**: 생성 모델링 vs. 판별 모델링
- 기존: BC는 판별 분류/회귀 문제로 접근
- 새로운: 생성 모델로서 완전한 분포 학습
- 의의: 이미지 생성의 성공이 행동 모델링에 직접 적용 가능함을 증명

**2. 다중모달성의 해결**
- 기존의 VAE, 정규화 흐름(Normalizing Flow) 등과 달리 **명시적 근사 없이** 다중모달 분포 표현
- 로봇 조작(pick vs. place), 게임 플레이(여러 전략) 등 다중 최적점이 있는 작업에 자연스럽게 적용

### 5.2 2024-2025년의 관련 연구 동향

최근 연구는 다음과 같은 방향으로 발전:

#### A. 온라인 RL과의 결합

**QVPO (Q-weighted Variational Policy Optimization, 2024)**:
- 온라인 RL에서 확산 정책의 정책 개선 문제 해결
- Q-함수 가중치를 이용한 변분 손실 하한 도입
- 기존 Diffusion BC의 오프라인 학습 한계를 온라인 환경으로 확장

**MaxEnt RL with Diffusion Policy (2025)**:
- Soft Actor-Critic(SAC)와 확산 정책 결합
- 최대 엔트로피 RL 목표 달성으로 탐색 능력 향상

#### B. 효율성 개선

**Streaming Diffusion Policy (2024)**:
- 변수 노이즈 확산으로 디노이징 스텝 감소
- 실시간 로봇 제어를 위한 속도 최적화
- Diffusion BC의 20-80배 느린 문제 부분 해결

**D3P (Dynamic Denoising Diffusion Policy, 2025)**:
- 행동별 동적 디노이징 스텝 할당
- 중요한 행동(예: 미세 조작)과 루틴 행동(이동) 구분
- 2.2배 추론 속도 향상 달성

#### C. 데이터 효율성 향상

**Latent Diffusion Planning (ICML 2025)**:
- 변분 오토인코더(VAE)를 통해 이미지 임베딩 공간에서 계획
- 행동 없는 시연(action-free demonstrations) 활용 가능
- 불완전한 데이터(부분 행동 정보)로도 학습 가능

**Diffusion Imitation from Observation (DIFO, 2024)**:
- 상태만으로 학습 (행동 라벨 불필요)
- 적대적 모방 학습(AIL) 프레임워크와 결합
- 상태 전이를 이용한 전문가 행동 판별

#### D. 강건성 및 일반화

**C3DM (Constrained-Context Conditional Diffusion Models, 2024)**:
- 시각적 주의 산만(distraction)에 강건한 행동 모델
- 비전 언어 모델의 사전 지식 활용
- 실세계 전개 시 성능 저하 감소

**How Generalizable Is My Behavior Cloning Policy? (2024)**:
- 확산 정책의 분포 이동 강건성 정량화
- 신뢰도 구간을 통한 일반화 성능 예측 방법론
- 테스트 환경에서의 성능 불확실성 평가

**Score-Based Diffusion Policy Compatible with RL (2025, OTPR)**:
- 최적 운송(Optimal Transport) 이론으로 IL과 RL 결합
- 분포 시프트 견디기 향상
- 의미있는 재샘플링으로 훈련 안정성 개선

### 5.3 앞으로 연구 시 고려할 점

#### 1. **계산 효율성**
- **문제**: 디노이징 단계가 실시간 제어에 걸림돌
- **개선 방향**: 
  - 증류(distillation)를 통한 소형 모델 생성 (DDIL, 2025)
  - 조건부 계산: 상황에 따라 스텝 수 동적 조정
  - 병렬 디노이징 또는 계층적 생성 구조

#### 2. **시간적 의존성 모델링**
- **현재 한계**: 단일 타임스텝 행동만 생성
- **개선 방향**:
  - 확산 기반 궤적 계획(Diffuser, 2022)
  - 연쇄적 조건화(sequential conditioning)로 미래 행동 순서 정보 활용
  - 상태-행동 궤적에 대한 결합 확산 모델

#### 3. **온라인 정책 개선**
- **현재 한계**: BC는 시연 분포로 제한되어 분포 밖 상태 대응 어려움
- **개선 방향**:
  - Q-함수 안내 샘플링을 통한 정책 최적화
  - 보상 함수와 확산 모델의 결합
  - 제약 조건을 고려한 정책 개선

#### 4. **멀티태스크 학습**
- **현재 한계**: 단일 작업에 대한 모방
- **개선 방향**:
  - 과제 조건화 확산 정책
  - 계층적 행동 분해 (기술 학습)
  - 메타 학습으로 새로운 과제에의 빠른 적응

#### 5. **강건성 및 도메인 일반화**
- **문제**: 훈련-테스트 간 시각적 분포 변화에 취약
- **개선 방향**:
  - 비전 언어 모델의 시맨틱 정보 활용 (Imit Diff, 2025)
  - 적대적 강화로 도메인 변화 대응
  - 불확실성 정량화와 검증 전략

#### 6. **이론적 기초 강화**
- **필요성**: 확산 기반 모방학습의 수렴 보장 및 오류 한계 분석 부재
- **개선 방향**:
  - 확산 모델의 정책 학습 능력에 대한 근사 오류 분석
  - KL 발산 또는 Wasserstein 거리의 하한 유도
  - 샘플 복잡도(sample complexity) 특성화

#### 7. **실세계 로봇 배포**
- **현재 진전**: CSGO 게임에서 시연 (이미지 입력, 혼합 행동 공간)
- **개선 방향**:
  - 물리 로봇 실험 대규모 확대
  - 실시간 제어와 안전성 보장
  - 교정 학습(learning from mistakes)으로 온라인 개선

***

## 6. 논문의 기술적 혁신점 정리

| 측면 | 기여 |
|------|-----|
| **모델 클래스** | 행동 모방의 첫 체계적 확산 모델 적용 |
| **아키텍처 설계** | 이미지가 아닌 벡터 행동 생성을 위한 경량 구조 제시 |
| **CFG 분석** | 순차 환경에서 CFG의 반직관적 실패 원인을 수학적으로 증명 |
| **샘플링 안정화** | Diffusion-X/KDE로 신뢰할 수 있는 샘플링 메커니즘 개발 |
| **실증 평가** | 로봇 제어와 게임의 두 도메인에서 SOTA 달성 |
| **다중모달성** | 비근사적 다중모달 분포 표현의 실용적 효과 검증 |

***

## 결론

"Imitating Human Behaviour with Diffusion Models"은 생성 모델의 표현력을 행동 모방 학습에 결합하는 **패러다임 전환**을 제시한다. 기존 방법의 근본적 한계(다중모달성, 상관관계, 편향)를 체계적으로 분석하고, 확산 모델이 이를 자연스럽게 해결함을 이론과 실험으로 증명한다.

이후 2024-2025년 연구는 **효율성, 온라인 정책 개선, 데이터 효율성, 강건성** 등 다각적 측면에서 확산 기반 정책의 실무적 가치를 증대시키고 있다. 특히 **Latent Diffusion Planning, D3P, OTPR** 등의 최근 논문들은 확산 모델이 다양한 학습 시나리오(불완전 데이터, 온라인 학습, 실시간 제어)에 적응할 수 있음을 보여준다.

향후 연구는 **계산 효율성, 시간적 의존성, 온라인 적응, 강건성** 등에서의 기술적 돌파구를 필요로 하며, 궁극적으로 실세계 로봇 제어와 자율 에이전트 개발의 핵심 기술로 자리 잡을 것으로 예상된다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1030efc5-cf8d-4048-af3b-034e8118c4a3/2301.10677v2.pdf)
[2](https://edu.pubmedia.id/index.php/ptk/article/view/1603)
[3](https://www.semanticscholar.org/paper/945a899a93c03eb63be5e3197e318c077473cef9)
[4](https://arxiv.org/abs/2504.16081)
[5](https://arxiv.org/abs/2507.20478)
[6](https://badanpenerbit.org/index.php/SEMNASPA/article/view/2211)
[7](https://ejurnal.stpkat.ac.id/index.php/jutipa/article/view/369)
[8](https://journal.stitmadani.ac.id/index.php/JPI/article/view/487)
[9](https://jurnalp4i.com/index.php/academia/article/view/4981)
[10](https://jurnal.uns.ac.id/SHES/article/view/97214)
[11](https://journal-laaroiba.com/ojs/index.php/mk/article/view/3398)
[12](http://arxiv.org/pdf/2410.13855.pdf)
[13](https://arxiv.org/html/2502.09649v1)
[14](https://arxiv.org/pdf/2311.01419.pdf)
[15](https://arxiv.org/abs/2412.12953)
[16](http://arxiv.org/pdf/2406.04806.pdf)
[17](https://arxiv.org/html/2410.11971v2)
[18](https://arxiv.org/pdf/2312.06348.pdf)
[19](https://arxiv.org/html/2402.16075)
[20](https://icml.cc/virtual/2025/poster/43658)
[21](https://www.ri.cmu.edu/publications/toward-fast-and-generalizable-decision-making-with-diffusion-models/)
[22](https://pmc.ncbi.nlm.nih.gov/articles/PMC2474742/)
[23](https://openreview.net/pdf?id=k1qVBh5fnb)
[24](https://tri-ml.github.io/stochastic_verification/)
[25](https://pmc.ncbi.nlm.nih.gov/articles/PMC6764782/)
[26](https://www.youtube.com/watch?v=tJrA-BP3hHY)
[27](https://msl.stanford.edu/papers/vincent_how_2024.pdf)
[28](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2014.01364/full)
[29](https://www.trossenrobotics.com/post/the-rise-of-diffusion-models-in-imitation-learning)
[30](https://pmc.ncbi.nlm.nih.gov/articles/PMC5112760/)
[31](https://pmc.ncbi.nlm.nih.gov/articles/PMC4928591/)
[32](https://arxiv.org/abs/2402.04080)
[33](https://ieeexplore.ieee.org/document/10529228/)
[34](https://arxiv.org/abs/2405.16173)
[35](https://arxiv.org/abs/2405.20555)
[36](https://ieeexplore.ieee.org/document/11106367/)
[37](https://arxiv.org/abs/2502.11612)
[38](https://arxiv.org/abs/2502.12631)
[39](https://ieeexplore.ieee.org/document/11127451/)
[40](https://ieeexplore.ieee.org/document/11073292/)
[41](https://arxiv.org/abs/2508.06804)
[42](https://arxiv.org/pdf/2404.06356.pdf)
[43](https://arxiv.org/pdf/2502.02316.pdf)
[44](https://arxiv.org/abs/2305.13122)
[45](https://arxiv.org/html/2411.10809v1)
[46](http://arxiv.org/pdf/2405.16173.pdf)
[47](https://arxiv.org/html/2410.11338v1)
[48](https://arxiv.org/pdf/2409.01427v3.pdf)
[49](https://arxiv.org/html/2409.00588)
[50](https://www.sciencedirect.com/science/article/abs/pii/S089360802500574X)
[51](https://diffusion-policy.cs.columbia.edu/diffusion_policy_ijrr.pdf)
[52](https://proceedings.neurips.cc/paper_files/paper/2024/file/f7faa46b563c2e5343a728c85bace833-Paper-Conference.pdf)
[53](https://wnzhang.net/teaching/sjtu-rl-2024/slides/15-diffusion-rl.pdf)
[54](https://supersglzc.github.io/projects/ddiffpg/)
[55](https://cse.buffalo.edu/~kaiyiji/cse705/robot.pdf)
[56](https://proceedings.neurips.cc/paper_files/paper/2024/file/6111371a868af8dcfba0f96ad9e25ae3-Paper-Conference.pdf)
[57](https://arxiv.org/abs/2508.13922)
[58](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/diffusion-bc/)
[59](https://diffusion-steering.github.io)
[60](https://openaccess.thecvf.com/content/CVPR2024/papers/Foo_Action_Detection_via_an_Image_Diffusion_Process_CVPR_2024_paper.pdf)
