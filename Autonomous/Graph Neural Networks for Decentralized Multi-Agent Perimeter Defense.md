# Graph Neural Networks for Decentralized Multi-Agent Perimeter Defense

### 1. 핵심 주장 및 주요 기여

**"Graph Neural Networks for Decentralized Multi-Agent Perimeter Defense"**(Lee et al., 2023)는 대규모 다중로봇 경계 방어 문제에서 분산형 의사결정을 실현하기 위해 그래프 신경망(GNN)과 모방 학습을 결합한 첫번째 프레임워크를 제시합니다.[1]

논문의 핵심 주장은 다음과 같습니다:

**첫째, 확장성 문제의 해결**: 중앙집중식 최대 매칭 알고리즘은 조합론적 복잡도(NP-hard)로 인해 대규모 문제에서 계산이 불가능합니다. 저자들은 GNN의 분산 아키텍처가 이웃 에이전트와의 상호작용만으로 의도(intention)를 암묵적으로 전달하여, 중앙 전문가의 성능을 모방하면서도 확장성을 달성할 수 있음을 입증했습니다.[1]

**둘째, 일반화 능력**: 소규모(N=10)에서 훈련된 모델이 대규모(N=100)로 일반화되는 특성을 활용하여, 매개변수 개수가 팀 규모에 독립적임을 보였습니다. 이는 순열 동등성(permutation equivalence)이라는 GNN의 내재적 특성 덕분입니다.[1]

**셋째, 반구형 환경에서의 첫 성공**: 기존 연구는 2D 평면 또는 완전히 3D 공간의 경계 방어만 다루었으나, 본 논문은 반구형 표면의 물리적 제약이 있는 환경에서 처음으로 분산형 전략을 학습했습니다.

***

### 2. 문제 정의, 제안 방법론 및 수식

#### 2.1 반구형 경계 방어 게임 공식화

문제는 다음과 같이 정의됩니다. N명의 방어자 $D = \{D_i\}\_{i=1}^{N}$가 반구형 표면 위에 제약되어 움직이고, N명의 침입자 $A = \{A_j\}_{j=1}^{N}$가 지면 평면에서 반구의 반지름 $R$인 경계에 도달하려 합니다. 방어자는 침입자를 거리 $\epsilon$ 이내로 포획할 수 있습니다.[1]

#### 2.2 최적 위협 지점 (Optimal Breaching Point)

1 vs 1 게임에서, 최적 위협 지점은 defender와 attacker 모두가 전략적으로 이동해야 할 지점입니다. 이를 결정하는 핵심 방정식들은 다음과 같습니다:[1]

```math
\beta^* = \cos^{-1}\left(\frac{\nu}{\cos \phi_D \sin \theta^*}\sqrt{1 - \cos^2 \phi_D \cos^2 \theta^*}\right) 
```

$$\theta^* = \psi - \beta^* + \cos^{-1}\left(\frac{\cos \beta^*}{r}\right) $$

여기서 $\beta^\*$는 최적 접근 각도, $\theta^*$는 최적 위협 각도입니다.

#### 2.3 페이오프 함수 (Payoff Function)

각 defender-intruder 쌍에 대해 게임의 가치를 정의합니다:[1]

$$p(z_D, z_A, z_B) = \tau_D(z_D, z_B) - \tau_A(z_A, z_B) $$

여기서 $\tau_D$는 defender의 위협 지점 도달 시간, $\tau_A$는 intruder의 도달 시간입니다. $p < 0$이면 defender가 더 빨리 도달합니다.

#### 2.4 최대 매칭을 통한 최적 할당

N vs N 게임에서 포획 최대화는 이분 그래프 매칭 문제로 변환됩니다. 최적 매칭은 음의 페이오프를 가진 쌍의 개수를 최대화하면서, 게임 값을 최소화합니다:[1]

$$V = \sum_{(D_i, A_j) \in E^*} p_{ij} $$

여기서 $E^* = \{(D_i, A_j) \in E | p_{ij} < 0\}$입니다.

#### 2.5 그래프 신경망 아키텍처

**그래프 이동 연산 (Graph Shift Operation)**:

각 defender의 특성 벡터 $x_i \in \mathbb{R}^F$를 수집하여 특성 행렬을 구성합니다:[1]

```math
X = \begin{bmatrix} x_1^T \\ \vdots \\ x_N^T \end{bmatrix} = [x_1, \cdots, x_F] \in \mathbb{R}^{N \times F}
```

이동 연산은 이웃 특성의 선형결합입니다:[1]

$$[SX]_{if} = \sum_{j=1}^{N} [S]_{ij}[X]_j^f = \sum_{j \in N_i} s_{ij}x_j^f $$

**그래프 컨볼루션 (Graph Convolution)**:

K-hop 통신을 통한 다층 특성 조합:[1]

$$H(X; S) = \sum_{k=0}^{K} S^k X H_k $$

여기서 $H_k \in \mathbb{R}^{F \times G}$는 계수 행렬이고, $S^k X$는 k번의 통신 교환을 나타냅니다.

**그래프 신경망 모듈**:

비선형 활성화 함수를 적용한 전체 GNN:[1]

$$X^\ell = \sigma\left(H^\ell(X^{\ell-1}; S)\right) \quad \text{for } \ell = 1, \cdots, L $$

입력 $X^0 = X$에서 시작하여, 최종 출력 $X^L \in \mathbb{R}^{N \times G}$는 K-hop 통신을 통해 교환되고 융합된 defender 팀의 정보를 표현합니다.

#### 2.6 모방 학습 목표

네트워크는 전문가 정책의 할당 가능성(assignment likelihood) $L_g$를 모방하도록 교차 엔트로피 손실로 훈련됩니다:[1]

$$\mathcal{L} = -\sum_{j=1}^{N_A^f} L_g[j] \log L[j]$$

***

### 3. 모델 구조

#### 3.1 전체 프레임워크

논문의 시스템은 세 가지 주요 모듈로 구성됩니다:[1]

**인식 모듈 (Perception Module)**:
- 상대 침입자 위치 $Z_{A_i}$: 가장 가까운 $N_A^f = 10$개 (구면 좌표)
- 상대 defender 위치 $Z_{D_i}$: 가장 가까운 $N_D^f = 3$개
- 특성 추출: 2층 MLP (16, 8 은닉층) + ReLU

**학습 및 계획 모듈 (Learning & Planning Module)**:
- 그래프 신경망: 2층 (32, 128 은닉층)
- K-hop 통신 (기본값 K=1, $r_c = 1$)
- 후보 매칭: 단층 MLP로 할당 가능성 생성

**제어 모듈 (Control Module)**:
- 매칭된 defender-intruder 쌍에 대해 최적 위협 지점 계산
- SO(3) 명령(추력 및 모멘트) 생성으로 제어 루프 폐쇄

#### 3.2 아키텍처 설계 선택

논문에서 특이한 선택은 행동을 직접 학습하지 않고, 할당만 학습한다는 점입니다. 일단 defender-intruder 쌍이 결정되면, 최적 전략(섹션 3.5의 Nash 균형)은 결정론적으로 계산될 수 있기 때문입니다. 이는 신경망의 역할을 단순화하고 일반화 가능성을 향상시킵니다.[1]

#### 3.3 순열 동등성 (Permutation Equivalence)

GNN 기반 접근의 핵심 특성은 침입자 ID의 순서에 무관하게 작동한다는 점입니다. 시간 $t=1$에서 ID 인 침입자가 $t=2$에서 로 변경되어도, 네트워크는 일관되게 로컬 기하학만으로 의사결정합니다. 이를 통해 가변 수의 에이전트를 처리할 수 있습니다.[2][3][4][5][1]

***

### 4. 성능 향상 및 일반화 분석

#### 4.1 실험 설정

훈련 데이터: 반구형 게임의 10백만 샘플 (전문가 최대 매칭으로 생성)
- 훈련: 60%, 검증: 20%, 테스트: 20%
- 기본 팀 크기: $N_{def} = 10$
- 반구 반지름: $R = \sqrt{N/N_{def}}$ (크기 조정)

비교 알고리즘:
- **Expert**: 최대 매칭 (N≤10에서만 가능)
- **GNN**: 제안 방법
- **Greedy**: 로컬 페이오프 최소화
- **Random**: 무작위 선택
- **MLP**: GNN 없이 MLP만 사용

#### 4.2 성능 메트릭 및 결과

**절대 정확도** (포획된 침입자 수 / 총 침입자 수):

| Team Size | N=2 | N=4 | N=6 | N=8 | N=10 |
|-----------|-----|-----|-----|-----|------|
| GNN       | 0.40| 0.50| 0.53| 0.63| 0.63 |
| Expert    | N/A | N/A | N/A | N/A | 0.66 |
| Greedy    | 0.35| 0.48| 0.47| 0.50| 0.52 |

| Team Size | N=20 | N=40 | N=60 | N=80 | N=100 |
|-----------|------|------|------|------|-------|
| GNN       | 0.53 | 0.59 | 0.53 | 0.55 | 0.54  |
| Greedy    | 0.47 | 0.37 | 0.37 | 0.36 | 0.36  |
| MLP       | 0.44 | 0.30 | 0.21 | 0.17 | 0.16  |

**상대 정확도** (GNN vs 다른 알고리즘의 포획 수 비율):[1]

소규모(N≤10)에서:
- GNN vs Expert: 0.80-0.95 (95% 정도 근접)
- GNN vs Greedy: 1.05-1.25배 (5-25% 우수)
- GNN vs Random: 1.33-2.38배 (33-138% 우수)
- GNN vs MLP: 1.14-1.72배 (14-72% 우수)

대규모(N>10)에서:
- GNN vs Greedy: 1.13-1.59배 (13-59% 우수)
- GNN vs Random: 1.63-1.93배 (63-93% 우수)
- GNN vs MLP: 1.20-3.37배 (20-237% 우수)

#### 4.3 일반화 성능 분석

**크기 일반화**:
GNN 방법은 N=10에서 훈련되었음에도 N=100까지 안정적인 성능을 유지합니다. 이는 세 가지 메커니즘 덕분입니다:[1]

1. **고정 입력 크기**: 항상 $N_A^f=10$ 침입자와 $N_D^f=3$ defender만 고려
2. **매개변수 독립성**: 신경망 파라미터 $\theta$가 N에 무관
3. **분산 정보 처리**: K-hop 통신으로 로컬 정보에서 글로벌 의도 추출

이를 입증하기 위해 저자들은 여러 분석을 수행했습니다:

**훈련 데이터 효율성**:[1]
- 10백만 샘플: 최적 성능
- 1백만 샘플: 약간 저하 (~5%)
- 100k 샘플: 현저한 저하 (~15%)
- 10k 샘플: 심각한 저하 (~25%)

데이터 부족 시 모방 학습의 한계가 드러납니다.

**반지름 변화**:[1]
기본 팀 크기 40으로 고정하고 반지름을 100m에서 800m로 증가시키면, 포획률이 0에 수렴합니다. 이는 팀이 더 넓은 영역을 방어해야 하므로 자연스러운 결과입니다.

**감지 특성 수**:[1]
$N_A^f$ (감지된 침입자 수)를 1에서 10으로 증가시키면 성능이 향상됩니다. 이는 더 많은 정보가 더 나은 할당을 가능하게 함을 보여줍니다.

#### 4.4 GNN vs MLP 비교의 의미

MLP 단독은 대규모에서 완전히 실패합니다 (N=100일 때 16%, GNN은 54%). 왜일까요?[1]

- **MLP의 한계**: 모든 defender의 상태를 입력으로 받기 때문에, 입력 차원이 N에 선형으로 증가
- **N=10에서 훈련된 MLP**: 입력 차원 = 30 (3 defender × 10)
- **N=100에서 실행 시**: 입력 차원 = 300 (3 defender × 100)
- **분포 이동 문제**: 완전히 다른 입력 차원의 작업을 수행하게 됨

GNN의 K-hop 통신은 이 문제를 해결합니다. 각 defender는 항상 정확히 3개의 이웃과만 통신하므로, 입력 차원이 N에 무관합니다.

***

### 5. 모델의 일반화 성능 향상 가능성

#### 5.1 현재 일반화 메커니즘

논문의 GNN 접근이 일반화되는 이유는 다음과 같습니다:[1]

**1) 로컬 정보 구조 보존**
- 각 defender는 항상 $N_D^f=3$개의 이웃과만 통신
- 침입자 감지는 항상 상위 10개로 제한
- 이 구조는 팀 크기와 무관하게 동일

**2) 암묵적 글로벌 정보 전파**
K-hop 통신을 통해 로컬 정보가 확산됩니다:
- K=1: 직접 이웃 정보만
- K=2: 2단계 이웃 정보
- K=3 이상: 점차적으로 더 멀리 떨어진 defender의 의도 캡처

**3) 그래프 구조의 불변성**
반구형 topology에서 defender의 배치는 스케일링에 불변입니다. 10명이든 100명이든, 각 defender의 상대적 통신 구조는 동일합니다.

#### 5.2 일반화 성능 향상의 한계와 개선 방안

**현재 한계**:

1. **분포 이동 (Distribution Shift)**: 훈련 분포(N=10)와 테스트 분포(N=100)의 초기 조건이 다를 수 있음
2. **경계 효과**: 반구의 가장자리에서의 동작이 중앙과 다를 수 있음
3. **통신 제약 깊이**: K=1로 제한되면 먼 defender의 의도를 캡처 못함

**개선 방안**:

1. **더 깊은 K-hop 통신**: K=2 또는 K=3으로 증가시켜 글로벌 정보 전파 개선
   - 계산 비용 증가, but 복잡한 팀 상호작용 캡처 가능

2. **전이 학습 (Transfer Learning)**: 다양한 팀 크기(5, 10, 20, 40)에서 사전 훈련 후 미세 조정
   - 예: 10→100 전이 시 적응 층 추가

3. **메타 학습**: 신속한 적응을 위해 MAML 등의 메타러닝 기법 적용

4. **적응형 입력 처리**: 팀 크기에 따라 $N_D^f$를 동적으로 조정
   - 현재: 고정 3개
   - 제안: N에 비례하는 로그 함수 (예: $\lceil \log N \rceil$)

#### 5.3 이론적 분석

논문은 이론적 일반화 보장을 제공하지는 않지만, 다음과 같은 직관을 제시합니다:[1]

GNN 기반 정책이 충분히 깊으면 (K가 크면), 전문가 정책의 동작을 근사할 수 있습니다. 이는 GNN이 graph shift operator의 다항식(polynomial)이기 때문입니다.

$$X^{(K)} = \text{poly}(S) \approx \text{Expert Policy}$$

그러나 이를 엄밀하게 증명하려면 추가 분석이 필요합니다.

***

### 6. 한계 (Limitations)

#### 6.1 문제 공식화의 한계

**점 입자 가정**:[1]
실제 로봇은 유한 크기를 가지지만, 모델은 이를 무시합니다. 저자들은 Lee et al. (2021)의 예비 작업이 1 vs 1 경우에 대해 3D 로봇 크기를 다루지만, 다중 에이전트로 확장하려면 추가 연구가 필요함을 인정합니다.

**1차 역학**:
defender와 intruder는 최대 속도로 움직일 수 있지만, 가속도 제약이 없습니다. 실제 항공기는 선회 반경 제약이 있으므로 이 모델은 낙관적입니다.

**완벽한 상태 추정**:
모든 에이전트의 정확한 위치가 알려져 있다고 가정합니다. 실제 센서는 노이즈가 있으므로, 불완전 정보 게임으로 문제를 재공식화해야 합니다.

#### 6.2 방법론적 한계

**대규모 전문가 정책 부재**:[1]
N > 10일 때 최대 매칭 알고리즘은 계산 불가능하므로, 저자들은 greedy, random, MLP와만 비교합니다. 따라서 대규모에서 GNN이 진정으로 얼마나 최적에 가까운지 알 수 없습니다. 저자들은 "강화학습으로 대규모 환경에서 최적 정책을 찾는 것"을 미래 방향으로 제시합니다.

**고정 이웃 크기**:
$N_D^f=3$과 $N_A^f=10$은 대칭적이지 않으며, 경험적으로 선택되었습니다. 동적 환경에서는 감지 범위가 시간에 따라 변할 수 있는데, 이를 처리하려면 적응형 메커니즘이 필요합니다.[1]

**모방 학습의 근본적 한계**:
모방 학습은 "상태 분포 불일치 (state distribution mismatch)" 문제로 고생합니다. 훈련 중 전문가가 선택한 상태만 본다면, 테스트 중 GNN의 행동이 전문가 분포에서 벗어나면 오류가 쌓입니다.

#### 6.3 실험적 한계

**제한된 규모**:
최대 100명의 에이전트로만 테스트했습니다. 1000명 규모는 어떨까요?

**단일 환경 유형**:
반구형만 테스트했습니다. 원통형, 불규칙한 모양은 어떨까요?

**침략자 행동 단순화**:
침략자는 항상 최적 경로(위협 지점)로 이동한다고 가정합니다. 실제 적은 방어자를 피해서 움직일 수 있습니다.

***

### 7. 2020년 이후 관련 최신 연구 비교

#### 7.1 GNN 기반 분산 제어 연구

| 논문 | 저자 | 연도 | 주요 기여 | 차이점 |
|------|------|------|---------|--------|
| Learning Decentralized Controllers for Robot Swarms with GNNs | Tolstaya et al. | 2020 | 비행 제어, K-hop 통신 | 경계 방어 문제 아님, MPC 기반 전문가 |
| Message-aware Graph Attention Networks | Li et al. | 2021 | 다중로봇 경로 계획 | 연속 공간이 아닌 그리드 기반 |
| Learning Scheduling Policies with Graph Attention Networks | Wang & Gombolay | 2020 | 작업 할당 문제 | 최대 매칭 대신 조합 최적화 |
| Graph Neural Networks for Multi-robot Submodular Action | Zhou et al. | 2021 | 목표 추적 | 개별 목표 할당 |
| Our Paper | Lee et al. | 2023 | 반구형 경계 방어, 3D 공간 | **첫번째 경계 방어 GNN 솔루션** |

#### 7.2 경계 방어 게임 연구

| 논문 | 저자 | 연도 | 환경 | 방식 | 에이전트 수 |
|------|------|------|------|------|-----------|
| Perimeter-defense game (2D) | Shishika & Kumar | 2018 | 2D 평면 | 해석적 | 이론적 |
| Perimeter Defense via Flow Networks | Chen et al. | 2021 | 2D 평면 | 선형 계획법 | 제약 없음 |
| Conical Environment Perimeter Defense | Bajaj et al. | 2021 | 원뿔 | 기하학적 | 제약 없음 |
| Decentralized Perimeter Defense (Multitask) | Velhal et al. | 2022 | 볼록 모양 경계 | DMRST-MTA | 시뮬레이션 (~20) |
| **Our Paper** | **Lee et al.** | **2023** | **반구형** | **GNN 모방** | **100** |

**새로운 점**:
- 첫번째 높은 차원 반구형 환경에서의 분산 학습
- GNN 기반 확장 가능한 솔루션
- 모방 학습으로 100명까지 스케일

#### 7.3 최신 트렌드 (2023-2025)

**제어 장벽 함수 + GNN**:
- GCBF+ (2024): 안전성 보장하면서 GNN으로 학습
- 차이: 본 논문은 순수 성능 최적화, 이들은 안전성-성능 트레이드오프

**더 깊은 협력 학습**:
- HIPPO-MAT (2025): GraphSAGE + IPPO로 작업 할당
- MAGNNET (2025): GNN + CTDE + PPO
- 차이: 강화학습 기반, 본 논문은 모방 학습

**동적 통신 그래프**:
- Dynamic Graph Communication (2024)
- 차이: 통신 토폴로지가 시간에 따라 변함

**Lyapunov 안정성**:
- Lyapunov-Based GNNs (2025)
- 차이: 이론적 안정성 보장 추가

**비교 결론**:
본 논문의 접근은:
- ✅ 장점: 모방 학습으로 빠른 훈련, 결정론적 최적 행동 계산
- ❌ 단점: 안정성 보장 없음, 강화학습의 유연성 부족

***

### 8. 앞으로의 연구 방향 및 고려사항

#### 8.1 논문이 제시한 직접적 향후 연구

**비전 기반 지각**:[1]
현재는 완벽한 상태 추정을 가정하지만, 실제 배포를 위해 카메라 영상 기반 감지 필요. 이를 위해:
- 자세 추정 신경망 통합
- 부분적 가시성 처리
- 시각적 노이즈 견고성

**대규모 전문가 정책**:[1]
강화학습으로 N > 10에서 최적 정책을 학습하여, GNN의 실제 성능 평가 가능. 예:
- DDPG/PPO로 중앙 집중식 정책 학습
- 이를 GNN으로 모방

**실제 로봇 검증**:[1]
수치 시뮬레이션에서 실제 쿼드로터 또는 UAV로 검증

#### 8.2 광범위한 미래 연구 기회

**1) 문제 확장**

- **이기종 팀**: Hsu et al. (2022)처럼 다양한 속도의 defender
- **다중 경계**: 반구 여러 개 또는 복잡한 3D 모양
- **동적 환경**: 장애물, 이동 경계, 다중 침입자 팀
- **불완전 정보**: 통신 제약, 센서 노이즈, 적의 움직임 불확실성

**2) 방법론 개선**

| 개선 방향 | 기법 | 기대 효과 |
|---------|------|---------|
| 더 깊은 K-hop | K=2,3,4 | 글로벌 조정 향상 |
| 주의 메커니즘 | Graph Attention Networks | 중요 defender 자동 선택 |
| 메타 학습 | MAML, Prototypical Networks | 새 환경에 빠른 적응 |
| 강화학습 혼합 | RL + IL | 훈련 분포 밖에서 개선 |
| 적응형 입력 | 동적 이웃 수 | 다양한 팀 크기에 자동 조정 |

**3) 이론적 분석**

- GNN이 근사할 수 있는 함수 클래스의 특성화
- 일반화 오차의 상한 도출
- K-hop 깊이와 성능의 관계식

**4) 응용 확장**

- **감시 (Surveillance)**: 움직이는 목표 추적
- **협력 게임**: 여러 defender 팀의 경쟁
- **주어진 예산 제약**: 제한된 defender로 최대 구간 방어
- **통신 대역폭 제약**: 메시지 크기 제한

#### 8.3 연구 시 고려할 점

**1) 확장성 검증**
- 1000+ 에이전트 규모에서의 계산 시간 측정
- 메모리 효율성 분석
- 실시간 성능 (100Hz 이상) 가능 여부

**2) 견고성 (Robustness)**
- 에이전트 실패: 일부 defender가 고장나면?
- 통신 장애: 일부 메시지 손실?
- 센서 노이즈: 상태 추정 오류?

**3) 공정성 (Fairness)**
- 모든 defender가 동등하게 기여하는가?
- 일부 defender가 과부하되지 않는가?

**4) 환경 적응**
- 훈련과 다른 반지름 반구에서 성능?
- 반구가 아닌 다른 모양(원통, 구)에서?

**5) 계산 효율성**
- 훈련 시간: 몇 시간/일?
- 추론 시간: 각 스텝 몇 ms?
- GPU/CPU 요구사항?

***

### 결론

"Graph Neural Networks for Decentralized Multi-Agent Perimeter Defense"는 대규모 다중로봇 방어 문제에서 분산형 의사결정의 실현 가능성을 처음으로 입증한 중요한 논문입니다. GNN의 순열 동등성과 로컬 통신 구조를 활용하여, 10배 규모의 문제로 일반화되는 신경 정책을 달성했습니다.

**핵심 강점**:
1. 반구형 환경에서의 첫 분산 GNN 솔루션
2. 모방 학습으로 중앙 전문가와 유사한 성능 달성
3. 명확한 확장성 메커니즘 (고정 입력 + K-hop 통신)

**주요 한계**:
1. 대규모 최적 정책 부재로 실제 성능 한계 불명확
2. 점 입자, 1차 역학, 완벽한 상태 추정 등 단순화된 가정
3. 모방 학습의 분포 이동 문제

**향후 영향**:
이 연구는 GNN을 통한 분산 제어의 표준 패러다임을 제시합니다. 경계 방어, 경로 계획, 작업 할당 등 다양한 다중로봇 문제에 적용 가능하며, 안정성, 견고성, 적응성을 추가하는 확장이 활발하게 진행 중입니다.

***

### 참고문헌

 Lee, E. S., Zhou, L., Ribeiro, A., & Kumar, V. (2023). Graph Neural Networks for Decentralized Multi-Agent Perimeter Defense. *Frontiers in Control Engineering*, 4, 1104745. https://doi.org/10.3389/fcteg.2023.1104745[1]

 Tolstaya, E., Gama, F., Paulos, J., Pappas, G., Kumar, V., & Ribeiro, A. (2019). Learning Decentralized Controllers for Robot Swarms with Graph Neural Networks. In *Conference on Robot Learning* (pp. 411-426).[2]

 Shishika, D., & Kumar, V. (2020). A review of multi agent perimeter defense games. *arXiv preprint arXiv:2008.01657*.[3]

 Velhal, S., Sundaram, S., & Sundararajan, N. (2022). A Decentralized Multirobot Spatiotemporal Multitask Assignment Approach for Perimeter Defense. *IEEE Transactions on Robotics*, 38(5), 3085-3096.[4]

출처
[1] 2301.09689v1.pdf https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3e5935ba-03bf-4813-8784-12851f58eaaa/2301.09689v1.pdf
[2] Decentralized Control with Graph Neural Networks https://www.semanticscholar.org/paper/7505a336e7ca5a9837e2adbb335d489d0202876b
[3] Graph Neural Networks for Decentralized Controllers https://ieeexplore.ieee.org/document/9414563/
[4] AoI-Driven Queue Management and Power Control in V2V Networks: A GNN-Enhanced MARL Approach https://www.semanticscholar.org/paper/754ca046a3e33755f50a23c900a0d5102f205584
[5] Deep Implicit Coordination Graphs for Multi-agent Reinforcement Learning https://www.semanticscholar.org/paper/9cc24deaca99010e7661cab7e32e3d30247a2a80
[6] IG-RL: Inductive Graph Reinforcement Learning for Massive-Scale Traffic Signal Control https://ieeexplore.ieee.org/document/9405489/
[7] Leveraging Graph Neural Networks and Multi-Agent Reinforcement Learning for Inventory Control in Supply Chains https://arxiv.org/abs/2410.18631
[8] Learning Scalable Decentralized Controllers for Heterogeneous Robot Swarms with Graph Neural Networks https://asmedigitalcollection.asme.org/dynamicsystems/article/146/6/061107/1201143/Learning-Scalable-Decentralized-Controllers-for
[9] Neural Graph Control Barrier Functions Guided Distributed Collision-avoidance Multi-agent Control https://arxiv.org/abs/2311.13014
[10] Graph Attention Network-Based QMIX for Coordinated Multi-Agent Traffic Signal Control https://ieeexplore.ieee.org/document/11302199/
[11] Multi-Agent Path Finding Based on Graph Neural Network https://ieeexplore.ieee.org/document/10240345/
[12] GCBF+: A Neural Graph Control Barrier Function Framework for Distributed
  Safe Multi-Agent Control http://arxiv.org/pdf/2401.14554.pdf
[13] HIPPO-MAT: Decentralized Task Allocation Using GraphSAGE and Multi-Agent
  Deep Reinforcement Learning https://arxiv.org/html/2503.07662v1
[14] MAGNNET: Multi-Agent Graph Neural Network-based Efficient Task
  Allocation for Autonomous Vehicles with Deep Reinforcement Learning https://arxiv.org/pdf/2502.02311.pdf
[15] Dynamic Graph Communication for Decentralised Multi-Agent Reinforcement
  Learning https://arxiv.org/html/2501.00165v1
[16] Lyapunov-Based Graph Neural Networks for Adaptive Control of Multi-Agent
  Systems https://arxiv.org/pdf/2503.15360.pdf
[17] LLM-Powered Decentralized Generative Agents with Adaptive Hierarchical
  Knowledge Graph for Cooperative Planning https://arxiv.org/html/2502.05453
[18] Learning Decentralized Strategies for a Perimeter Defense Game with
  Graph Neural Networks https://arxiv.org/pdf/2211.01757.pdf
[19] A Graph Neural Network Based Decentralized Learning Scheme https://www.mdpi.com/1424-8220/22/3/1030/pdf
[20] Mathematics https://arxiv.org/list/math/new
[21] Learning Decentralized Strategies for a Perimeter Defense Game with https://arxiv.org/pdf/2211.01757v1.pdf
[22] [PDF] Imitation Learning with Graph Neural Networks for Improving Swarm ... https://pdfs.semanticscholar.org/69b8/c81359dab6f1f3a05dec22817c8a2004aa12.pdf
[23] Physics https://arxiv.org/list/physics/new
[24] Graph Neural Networks for Decentralized https://arxiv.org/pdf/2301.09689.pdf
[25] Learning to Imitate Spatial Organization in Multi-robot Systems http://arxiv.org/pdf/2407.11592.pdf
[26] 1 http://arxiv.org/pdf/2307.09954.pdf
[27] Learning Decentralized Controllers for Robot Swarms with ... https://arxiv.org/pdf/1903.10527.pdf
[28] Graph Neural Networks for Decentralized Multi-Agent Perimeter ... https://arxiv.org/abs/2301.09689
[29] Learning Decentralized Controllers for http://arxiv.org/pdf/1903.10527.pdf
[30] Vision-based Perimeter Defense via Multiview Pose Estimation https://arxiv.org/pdf/2209.12136.pdf
[31] Robotic Manipulation via Imitation Learning: Taxonomy ... https://arxiv.org/html/2508.17449v1
[32] [PDF] arXiv:2109.02852v1 [cs.RO] 7 Sep 2021 https://arxiv.org/pdf/2109.02852.pdf
[33] Offline Imitation Learning Through Graph Search and Retrieval https://arxiv.org/html/2407.15403v1
[34] Perimeter-defense Game on Arbitrary Convex Shapes https://arxiv.org/pdf/1909.03989.pdf
[35] Integration of Decentralized Graph-Based Multi-Agent ... https://impact.ornl.gov/en/publications/integration-of-decentralized-graph-based-multi-agent-reinforcemen/
[36] Towards Scalable Imitation Learning for Multi-Agent ... https://openreview.net/forum?id=HJeANgBYwr
[37] Graph neural networks for decentralized multi-agent ... https://www.frontiersin.org/journals/control-engineering/articles/10.3389/fcteg.2023.1104745/full
[38] SwarmNet: Towards Imitation Learning of https://simplecore.intel.com/ai/wp-content/uploads/sites/69/SwarmNet_-Towards-Imitation-Learning-of-Multi-Robot-Behavior-with-Graph-Neural-Networks.pdf
[39] Leveraging graph neural networks and multi-agent ... https://www.sciencedirect.com/science/article/pii/S0098135425001152
[40] [T-Ro] Decentralized Approach for Perimeter Defense Problem(PDP). https://www.youtube.com/watch?v=O_yS4szWPSE
[41] SwarmNet: Towards Imitation Learning of Multi- ... http://www.robot-learning.ml/2019/files/papers/SwarmNet:%20Towards%20Imitation%20Learning%20of%20Multi-Robot%20Behavior%20with%20Graph%20Neural%20Networks.pdf
[42] Learning Decentralized Strategies for a Perimeter Defense ... https://arxiv.org/abs/2211.01757
[43] Learning Decentralized Controllers for Robot Swarms with ... http://proceedings.mlr.press/v100/tolstaya20a/tolstaya20a.pdf
[44] A graph attention network-based multi-agent reinforcement ... https://www.nature.com/articles/s41598-025-14032-w
[45] A Review of Multi Agent Perimeter Defense Games https://dl.acm.org/doi/10.1007/978-3-030-64793-3_26
[46] Graph Neural Networks for Multi-Robot Active Information ... https://arxiv.org/pdf/2209.12091.pdf
[47] ModGNN: Expert Policy Approximation in Multi-Agent Systems with a
  Modular Graph Neural Network Architecture http://arxiv.org/pdf/2103.13446.pdf
[48] Graph Neural Network Meets Multi-Agent Reinforcement Learning:
  Fundamentals, Applications, and Future Directions https://arxiv.org/pdf/2404.04898.pdf
[49] Less is More: Hop-Wise Graph Attention for Scalable and Generalizable
  Learning on Circuits http://arxiv.org/pdf/2403.01317.pdf
[50] GraphBridge: Towards Arbitrary Transfer Learning in GNNs http://arxiv.org/pdf/2502.19252.pdf
[51] CATGNN: Cost-Efficient and Scalable Distributed Training for Graph
  Neural Networks http://arxiv.org/pdf/2404.02300.pdf
[52] Reliable and Efficient Multi-Agent Coordination via Graph Neural Network
  Variational Autoencoders http://arxiv.org/pdf/2503.02954.pdf
[53] Graph Neural Networks with Model-based Reinforcement Learning for
  Multi-agent Systems http://arxiv.org/pdf/2407.09249.pdf
[54] Scalable and Transferable Reinforcement Learning for Multi-Agent Mixed Cooperative–Competitive Environments Based on Hierarchical Graph Attention https://www.mdpi.com/1099-4300/24/4/563/pdf?version=1650264922
[55] arXiv:2012.07421v3 [cs.LG] 16 Jul 2021 https://arxiv.org/pdf/2012.07421.pdf
[56] (PDF) The Social Robot in Rehabilitation and Assistance https://pdfs.semanticscholar.org/5170/7b82c5d84e26e64905d7ca7c6b7d457c28f7.pdf
[57] Pursuit-Evasion for Car-like Robots with Sensor Constraints https://arxiv.org/html/2405.05372v2
[58] Health workers' social networks and their influence in the ... https://journals.plos.org/globalpublichealth/article/file?id=10.1371%2Fjournal.pgph.0000798&type=printable
[59] A Three-Dimensional Pursuit-Evasion Game Based on http://arxiv.org/pdf/2503.08013.pdf
[60] Health workers' social networks and their influence in the ... https://journals.plos.org/globalpublichealth/article?id=10.1371%2Fjournal.pgph.0000798
[61] An Efficient Algorithm for Multiple-Pursuer-Multiple-Evader https://arxiv.org/pdf/1909.04171.pdf
[62] Value of Multiple-pursuer Single-evader Pursuit-evasion ... https://www.arxiv.org/pdf/2510.27271.pdf
[63] Abstract—Pursuit-evasion games are ubiquitous in nature https://arxiv.org/ftp/arxiv/papers/2104/2104.01445.pdf
[64] Learning Information Trade-offs in Pursuit-Evasion Games https://arxiv.org/pdf/2510.07813.pdf
[65] Learning to Play Pursuit-Evasion with Dynamic and Sensor ... https://arxiv.org/html/2405.05372v1
[66] Pursuit-Evasion on a Sphere and When It Can Be ... https://arxiv.org/html/2403.15188v1
[67] Reinforcement learning in pursuit-evasion differential game: safety, stability and robustness http://www.arxiv.org/abs/2507.19516
[68] Pursuit-Evasion for Car-like Robots with Sensor Constraints - arXiv https://arxiv.org/abs/2405.05372
[69] Decentralized Multi-Agents by Imitation of a Centralized ... https://proceedings.mlr.press/v145/lin22a/lin22a.pdf
[70] A Cooperative Pursuit-Evasion Game for Non-Holonomic ... https://skoge.folk.ntnu.no/prost/proceedings/ifac2014/media/files/1992.pdf
[71] Scalable Full-Graph GNN Training on Multiple GPUs https://dl.acm.org/doi/10.1145/3626733
[72] Imitation Learning via Expert Policy Support Estimation https://proceedings.mlr.press/v97/wang19d.html
[73] Pursuit-evasion game switching strategies for spacecraft ... https://www.sciencedirect.com/science/article/abs/pii/S1270963821006222
[74] [2403.13093] Graph Neural Network-based Multi-agent ... https://arxiv.org/abs/2403.13093
[75] Imitation Learning for Multi-turn LM Agents via On-policy ... https://arxiv.org/abs/2512.14895
[76] Orbital Impulsive Pursuit–Evasion Game Formulation and ... https://arc.aiaa.org/doi/abs/10.2514/1.A35956
[77] Graph Neural Network Meets Multi-Agent Reinforcement Learning https://arxiv.org/html/2404.04898v1
[78] Deep imitation reinforcement learning with expert ... https://digital-library.theiet.org/doi/full/10.1049/joe.2018.8314
[79] Fast and the Furious: Hot Starts in Pursuit-Evasion Games https://arxiv.org/abs/2510.10830
[80] Research on GNNs with stable learning | Scientific Reports https://www.nature.com/articles/s41598-025-12840-8
[81] apexrl/Imitation-Learning-Paper-Lists https://github.com/apexrl/Imitation-Learning-Paper-Lists
[82] Performance Analysis of Pursuit-Evasion Game-Based ... https://koreascience.kr/article/JAKO201021751975410.page
