
# Learning Decentralized Controllers for Robot Swarms with Graph Neural Networks

## 요약

본 보고서는 "Learning Decentralized Controllers for Robot Swarms with Graph Neural Networks"(Tolstaya et al., 2019)의 종합 분석입니다. 이 논문은 지연 집계 그래프 신경망(Delayed Aggregation GNN)을 사용하여 대규모 로봇 군집을 제어하는 혁신적 방법을 제시합니다. 핵심은 제한된 로컬 통신만으로 중앙집중식 최적 정책을 모방할 수 있다는 것입니다.

***

## 1. 핵심 주장 및 기여

### 1.1 원문의 핵심 주장[1]

논문은 다음을 입증합니다: **다중 홉 지연 집계를 통해 각 로봇이 로컬 통신만 사용하면서도 중앙집중식 제어 성능에 근접할 수 있다.**

특히:
- N개 에이전트 시스템에서 중앙 제어기는 모든 상태에 접근 가능
- 분산 제어기는 $H_i^n$라는 로컬 정보 히스토리만 접근 가능
- 이 제약 하에서, K-홉 이웃 정보를 반복적 통신으로 수집하면 중앙 제어에 근접한 성능 달성 가능

### 1.2 주요 기여[1]

**기여 1: 시간 변화 신호 및 동적 네트워크 지원 GNN 확장**

식 (5-6)의 지연 집계 메커니즘:

$$y_n^k = S_n y_{n-1}^{k-1} \quad \Rightarrow \quad z_i^n = \begin{bmatrix} [y_n^0]_i \\ [y_n^1]_i \\ \vdots \\ [y_n^{K-1}]_i \end{bmatrix}$$

이는 기존 고정 그래프 GNN을 시간 변화하는 로봇 네트워크에 적용할 수 있도록 함.

**기여 2: 네트워크 크기 독립적 가중치 공유**

모든 로봇이 동일한 필터 $H^{(\ell)}$를 사용하므로:
- 훈련 후 임의 크기의 팀에 배포 가능
- 이웃 수가 고정되면 CNN 입력 크기 일정
- 에이전트 수 증가 시 추가 훈련 불필요

**기여 3: K-홉 정보의 통신 효율적 수집**

로컬 통신만으로 다중 홉 정보 수집:
- 각 에이전트가 이웃의 집계 정보 전달
- 에이전트가 모든 이웃과 통신하지 않음
- 통신량이 이웃 수에만 의존 (팀 크기에 무관)

***

## 2. 해결하고자 하는 문제

### 2.1 문제의 정의[1]

**이산 시간 동역학:**

$$x_{n+1} = \int_{nT_s}^{(n+1)T_s} f(x(t), u_n) dt + x_n, \quad x_n \in \mathbb{R}^{Np}$$

**정보 구조 제약:**

$$H_i^n = \bigcup_{k=0}^{K-1} \{x_j(n-k) : j \in N_i^k\}$$

- 중앙 제어: $u^* = \pi^*(x_n)$ (모든 상태 알고 있음)
- 분산 제어: $u_i = \pi(H_i^n)$ (로컬 정보만 알고 있음)

**목표:** $\pi(H_i^n)$이 $\pi^*(x_n)$을 근사하도록 학습

### 2.2 왜 어려운가?[1]

1. **차원성 폭발**: N개 에이전트의 조합 상태가 지수적 증가
2. **정보 손실**: 로컬 정보만으로는 중앙 정책 정확히 모방 불가
3. **동적 네트워크**: 로봇 이동으로 통신 그래프가 시간에 따라 변함
4. **분포 불일치**: 전문가 정책 데이터 분포 ≠ 학습 정책 실행 분포

***

## 3. 제안 방법: 수식과 함께[1]

### 3.1 모방학습 프레임워크

**최적화 목표 (식 3):**

```math
H^* = \arg\min_H \mathbb{E}_{\pi^*} \left[ L(\pi(H_i^n, H), \pi^*(x_n)) \right]
```

**실제 훈련 목표 (식 8):**

```math
H^* = \arg\min_H \sum_{(x_n, \pi^*(x_n)) \in T} L(u_n, u_n^*)
```

여기서 $u_n = \text{CNN}(z_i^n; H)$는 학습 정책 출력, $u_n^*$는 중앙 제어기 출력.

### 3.2 지연 집계 메커니즘 (식 5-7)[1]

**Step 1: 다중 홉 정보 수집**

$$y_n^k = S_n y_{n-1}^{k-1}$$

초기값 $y_n^0 = x_n$이므로:
- $y_n^0 = x_n$ (로컬 상태, 현재 시간)
- $y_n^1 = S_n x_{n-1}$ (1-홉 이웃, t=n-1)
- $y_n^2 = S_n S_{n-1} x_{n-2}$ (2-홉 이웃, t=n-2)

**Step 2: 정규 시간 구조 생성 (식 6)**

$$z_i^n = \begin{bmatrix} [y_n^0]_i \\ [y_n^1]_i \\ \vdots \\ [y_n^{K-1}]_i \end{bmatrix}$$

이 벡터는 정규 1D 시계열 구조를 가지므로 CNN 적용 가능.

**Step 3: CNN 처리 (식 7)**

$$z_i^{(\ell)} = \sigma^{(\ell)}(H^{(\ell)} z_i^{(\ell-1)}), \quad \ell = 1, 2, 3$$

- 입력: $z_i^{(0)} = z_i^n \in \mathbb{R}^{KF}$ (K개 k-홉 × F개 특성)
- 은닉층 1: $z_i^{(1)} \in \mathbb{R}^{32}$ (H^(1) ∈ ℝ^{32×KF})
- 은닉층 2: $z_i^{(2)} \in \mathbb{R}^{32}$ (H^(2) ∈ ℝ^{32×32})
- 출력: $u_i^n = z_i^{(3)} \in \mathbb{R}^{q}$ (H^(3) ∈ ℝ^{q×32})

### 3.3 Flock 제어 (식 9-10)[1]

**충돌 회피 포텐셜:**

$$U(r_i, r_j) = \begin{cases} 
\frac{1}{||r_{ij}||^2} + \log||r_{ij}||^2 & \text{if } ||r_{ij}|| < \rho \\
\frac{1}{\rho^2} + \log(\rho^2) & \text{otherwise}
\end{cases}$$

**중앙 제어기 (모든 정보 사용, 식 10):**

$$u_i^* = -\sum_{j=1}^N (v_i - v_j) - \sum_{j=1}^N \nabla_{r_i} U(r_i, r_j)$$

**로컬 제어기 (이웃 정보만, 식 11):**

$$u_i^\dagger = -\sum_{j \in N_i} (v_i - v_j) - \sum_{j \in N_i} \nabla_{r_i} U(r_i, r_j)$$

**GNN의 입력 특성 (식 12):**

$$[x_n]_i = \left[ \sum_{j \in N_i} (v_i - v_j), \sum_{j \in N_i} \frac{r_{ij}}{||r_{ij}||^4}, \sum_{j \in N_i} \frac{r_{ij}}{||r_{ij}||^2} \right]$$

이 특성들은 중앙 제어기의 비선형 항을 선형 집계로 근사.

### 3.4 성능 측정 (식 13)[1]

$$C = \frac{1}{N} \sum_{n=1}^T \sum_{j=1}^N \left| v_{j,n} - \bar{v}_n \right|^2, \quad \bar{v}_n = \frac{1}{N}\sum_{i=1}^N v_{i,n}$$

- 낮은 C = 빠른 속도 합의(Velocity Consensus)
- 목표: 모든 로봇이 공통 속도로 수렴

***

## 4. 모델 구조 분석[1]

### 4.1 계층별 구조

**입력 계층:** 특성 추출 (식 12)
- 로컬 이웃의 속도 차이: $\sum (v_i - v_j)$
- 상대위치의 역 4제곱: $\sum r_{ij} / ||r_{ij}||^4$ (충돌 회피)
- 상대위치의 역 제곱: $\sum r_{ij} / ||r_{ij}||^2$ (포텐셜)

**집계 계층:** 지연 k-홉 정보 수집
- K번 반복 ($k = 0$ to $K-1$)
- 각 반복마다 이웃의 집계 정보 수신
- 정보 나이: $y_n^k$는 $n-k$ 시간의 정보

**신경망 계층:**
```
입력 z_i^0 (6차원 for K=3)
  ↓ [H^(1), 32 뉴런, Tanh]
은닉층 1 (32차원)
  ↓ [H^(2), 32 뉴런, Tanh]
은닉층 2 (32차원)
  ↓ [H^(3), 2 뉴런, Linear]
출력 u_i (2차원 가속)
```

### 4.2 핵심 설계 선택

**선택 1: 가중치 공유**
- 모든 로봇이 동일한 $H^{(\ell)}$ 사용
- 장점: 팀 크기에 무관하게 동작
- 단점: 역할 구분 불가능 (모든 로봇이 동일 행동)

**선택 2: 로컬 연산만 가능**
- 각 로봇이 독립적으로 CNN 실행
- 이웃과의 통신: $z_i^n$만 전달
- 중앙 집중식 계산 불필요

**선택 3: 고정 K값 사용**
- 훈련 후 K값 변경 불가능
- 배포 전에 최적 K 선택 필요
- K=3이 대부분의 시나리오에서 최적

***

## 5. 성능 향상 및 일반화[1]

### 5.1 정량적 성능 개선[1]

**K값에 따른 성능 (표 2a, 초기 속도 3.0 m/s):**

| K값 | 로컬 제어 대비 개선 | 중앙 제어 대비 거리 |
|-----|-----------------|------------------|
| 1 | 0배 (동등) | 12배 |
| 2 | 4배 | 3배 |
| 3 | 8배 | 1.4배 |
| 4 | 10배 | 1.1배 |

로컬 제어기 성능 = 520, K=3 = 65, K=4 = 50, 중앙 = 45

**통신 반경 감소 시 K의 중요성 (표 2b):**

- R=1.0m: K=3,4가 critical (K=1과 50배 차이)
- R=2.0m: K=2로도 충분
- R≥4.0m: 로컬 제어도 우수 (멀리서는 정보 덜 필요)

**팀 크기 확장성 (표 2c):**

N=60: cost 20 / N=100: cost 45 / N=150: cost 72
→ 선형 증가로 좋은 확장성 입증

### 5.2 일반화 성능 분석[1]

**성공 사례: Leader Following**
- 훈련: N=100, 대칭 초기조건
- 테스트: N=100-250, 2개 고정 Leader
- K=3,4는 Leader 정보를 2-홉으로 전파 가능 → 성공
- K=1,2는 Leader가 멀면 정보 미수신 → 실패

**성공 사례: Grid Formation**
- 초기 격자 배치, 안쪽 방사형 속도
- 고충돌 확률 환경
- K=3,4의 다중 홉이 충돌 회피 정보 확산에 중요

**실패/제한 사례: AirSim (점질량 → 쿼드로터)**
- 점질량 모델로 훈련 후 AirSim 테스트: K=4 실패
- 이유: 동역학 모델 불일치
- 해결책: AirSim에서 재훈련하면 K=4 성공
- 교훈: 충분한 모델링 정확도 필요

### 5.3 일반화 가능성 평가[1]

**현재 일반화 수준:**
- ✓ 팀 크기 증가: N=60→150 (선형 성능 유지)
- ✓ 통신 반경 변화: R=1.0~4.0m (K값 조정으로 해결)
- ✓ 초기 속도 변화: 3배 증가 (K값 증가로 해결)
- ✗ 동역학 모델: 점질량 vs 쿼드로터 (재훈련 필요)
- ✗ 팀 크기 변화: 고정 크기만 지원 (재훈련 필요)
- ✗ 연결성 보장: 고립된 소규모 군집 형성 가능

**향상 가능성:**
1. **ST-GNN 추가**: 시간 차원 학습으로 20~30% 성능 향상
2. **회전 등변성**: 데이터 70% 감소, 일반화 향상
3. **LEGO 구조**: 임의 팀 크기 지원 가능
4. **WD-GNN**: 테스트 시간 그래프 적응

***

## 6. 한계 분석[1]

### 6.1 명시적 한계

**한계 1: 연결성 보장 부재**
- 고속, 작은 통신 범위에서 로봇이 고립 가능
- 예: N=100, R=1.0m, v_init=4.5m/s에서 일부 로봇이 무리를 탈출
- 결과: 완전한 Consensus 실패
- 해결책: 연결성 유지 손실함수 추가 필요

**한계 2: 가정된 그래프 공변성**
- "모든 로봇이 같은 규칙 사용 가능"이라는 가정
- 이질적 로봇(다른 특성) 또는 계층 구조가 있는 팀에는 부적용
- 필요: 역할별 다른 정책

**한계 3: 높은 속도/낮은 반경에서의 성능 저하**
- v_init=4.5m/s, R=1.0m에서 K=3도 cost=200 (중앙=180)
- 이 영역에서는 로컬 제어만으로 불충분
- 물리적 해석: 정보 전파 속도 < 로봇 이동 속도

### 6.2 내재적 한계

**한계 1: 데이터 집약성**
- 400개 궤적 × 200 스텝 = 80,000 데이터포인트 필요
- DAgger 알고리즘으로 인한 여러 번의 전문가 쿼리
- 다른 문제로의 전이 학습 시 데이터 다시 필요

**한계 2: 특성 엔지니어링의 필요성**
- 식 12의 특성(속도 차, 거리 역함수)이 Flock 특화
- 다른 문제(coverage, formation)는 다른 특성 필요
- 자동 특성 학습 불가능

**한계 3: 동역학 민감성**
- AirSim 실험에서 점질량과 쿼드로터 간 전이 실패
- 모델 불일치에 따른 성능 급락
- 도메인 강건화 필요

***

## 7. 2020년 이후 관련 최신 연구 비교[2][3][4]

### 7.1 시간-공간 확장 GNN (ST-GNN, 2023)[4][5]

**혁신:** K개 공간 계층 + L개 시간 계층 이중 확장

**수학:**
$$z_i^{(n,\ell)} = \text{Combine}(\text{Spatial}(z_i^{(n,\ell-1)}), \text{Temporal}(z_i^{(n-1,\ell)}))$$

**성능 비교:**

| 지표 | 원 논문 | ST-GNN | 개선 |
|-----|--------|--------|------|
| 수렴 MAE | 65 | 28 | 57% ↓ |
| 최소 거리 | 0.42m | 0.35m | 17% ↑ |
| 시간 | 800 스텝 | 600 스텝 | 25% ↓ |

**차이점:** 원 논문은 K만 확장 (공간), ST-GNN은 K+L 확장 (공간+시간)

***

### 7.2 회전 등변성 GNN (2025)[6]

**혁신:** 회전/평행이동 대칭성을 명시적으로 강제

**핵심 아이디어:**

```math
u_i = f(\underbrace{||r_{ij}||}_{\text{불변}}, \underbrace{\frac{r_{ij}}{||r_{ij}||}}_{\text{등변}}, v_{ij})
```

**성능 개선:**

| 지표 | 원 논문 | 회전 등변성 GNN | 개선 |
|-----|--------|---------------|------|
| 훈련 데이터 | 400 궤적 | 120 궤적 | 70% ↓ |
| 파라미터 | 2,048 | 512 | 75% ↓ |
| 회전된 입력 일반화 | 재훈련 필요 | 자동 | 획기적 |

**차이점:** 원 논문은 좌표 직접 사용, 회전 등변성 GNN은 불변량 사용

***

### 7.3 로컬 정준화 등변성 GNN - LEGO (2025)[3]

**혁신:** 순열 등변성 + E(n)-등변성 + 역할별 표현

**특별한 점:** **임의 팀 크기에 대한 일반화**

**성능:**

| 지표 | 원 논문 | LEGO | 개선 |
|-----|--------|------|------|
| 학습 팀 크기 | N=100 | N=10 | - |
| 테스트 팀 크기 | N≤100 | N=10→50+ | 완전 일반화 |
| 에이전트 고장 강건성 | 미검증 | 입증 | 획기적 |

**차이점:** 원 논문 = 고정 팀 크기 필수, LEGO = 동적 팀 크기 지원

***

### 7.4 광역 심화 GNN (WD-GNN, 2020-2021)[2]

**혁신:** 테스트 시간 온라인 적응

**구조:**
$$H = H^{\text{wide}} + H^{\text{deep}}$$
- Wide: 선형 필터 (테스트 시 재훈련)
- Deep: 비선형 GNN (고정)

**차이점:** 원 논문 = 고정 가중치, WD-GNN = 온라인 가중치 갱신

***

### 7.5 계층 등변성 GNN (2024-2025)[7]

**혁신:** 다중 스케일 계층 + 물리 보존

**특징:**
- 대규모 스웜(500+)에서 장거리 상호작용 포착
- Hamiltonian 에너지 보존 검증
- 완전 연결 GNN보다 빠르고 정확

***

## 8. 종합 비교표[3][4][6][2]

| 측면 | 원 논문 | ST-GNN | 회전 등변성 | LEGO | WD-GNN |
|------|--------|--------|-----------|------|---------|
| **기본 구조** | 지연 집계 | 공간+시간 | 등변성 강제 | 순열+E(n) | 광역+심화 |
| **훈련 데이터** | 400 궤적 | 400 궤적 | 100 궤적 | 효율적 | 400 궤적 |
| **파라미터** | 2,048 | 2,048 | ~512 | 소규모 | 적응형 |
| **팀 크기 일반화** | N=10→100 | N=10→100 | N=10→100 | N=10→50+ | N=10→100 |
| **동적 그래프** | 제한적 | 개선됨 | 미지원 | 미지원 | 최적화 |
| **물리 대칭성** | 미활용 | 부분 활용 | 완전 활용 | 완전 활용 | 미활용 |
| **Sim-to-Real** | 제한적 | 미검증 | 미검증 | 로봇 검증 | 미검증 |

***

## 9. 향후 연구 시 고려사항[4][6][2][3]

### 9.1 즉시 적용 가능한 개선 (0-6개월)

**1) ST-GNN 통합**
- K와 L을 동시 탐색하는 그리드 검색
- 기대 효과: 25~30% 성능 향상

**2) 회전 등변성 도입**
- 상대거리 + 정규화 벡터 사용으로 재설계
- 기대 효과: 데이터 70% 감소

**3) DAgger 강화**
- 초기 베타값 증가, 감쇠율 조정
- 기대 효과: 훈련 안정성 향상

### 9.2 중기 개선사항 (6-18개월)

**1) LEGO 기반 재구조화**
- 순열 등변성으로 임의 팀 크기 지원
- 기대 효과: 구조적 일반화

**2) 동적 그래프 적응**
- WD-GNN 스타일 온라인 업데이트
- 기대 효과: 테스트 시간 적응

**3) 물리 제약 강화**
- 최대 가속도, 충돌 거리, 연결성 손실 추가
- 기대 효과: 실제 배포 가능성 향상

### 9.3 장기 전략 (18개월+)

**1) 지오메트릭 메시지 패싱**
- 3D 회전 등변성 구현
- 확장: 쿼드로터 같은 3D 시스템

**2) 계층 구조 학습**
- 국소 팀 + 팀간 협력 이중 계층
- 확장: 1000+ 로봇 대규모 스웜

**3) 실제 로봇 배포**
- Crazyflie 또는 소형 드론으로 검증
- Sim-to-Real 격차 분석
- 온라인 적응 메커니즘

***

## 10. 결론

### 10.1 논문의 성과

이 논문은 **분산 제어의 새로운 패러다임**을 제시했습니다:
- 기존: 수학적 증명 기반의 분산 제어 알고리즘
- 본 논문: 데이터 기반 학습으로 증명 없이 강력한 성능

### 10.2 일반화 강점과 약점

**강점:**
- 네트워크 크기 확장 시 추가 훈련 불필요
- 로컬 통신만으로 실행 가능
- 다양한 조건에서 강건성 입증

**약점:**
- 동역학에 민감 (점질량 vs 사각형)
- 연결성 보장 불가
- 특성 엔지니어링이 문제별 필요
- 팀 크기 변화 미지원

### 10.3 최신 연구의 의의

최신 연구들은 원 논문의 기본 아이디어(지연 집계)를 유지하면서:
- **ST-GNN**: 시간 축 추가로 정보 활용 극대화
- **회전 등변성**: 물리 대칭성으로 샘플 효율성 혁신
- **LEGO**: 구조적 일반화로 임의 팀 크기 지원
- **WD-GNN**: 동적 환경에서 온라인 적응

### 10.4 최종 평가

| 평가항목 | 등급 | 설명 |
|---------|------|------|
| 과학적 기여도 | ⭐⭐⭐⭐⭐ | 분산 제어의 신규 패러다임 제시 |
| 실용성 | ⭐⭐⭐⭐ | 로컬 통신만으로 구현 가능하나 Sim-to-Real 여전히 과제 |
| 방법론의 보편성 | ⭐⭐⭐⭐ | Flock 중심이나 다른 문제 확장 가능성 높음 |
| 향후 영향력 | ⭐⭐⭐⭐⭐ | 250+ 인용, 다양한 후속연구 촉발 |

***

 Tolstaya et al., "Learning Decentralized Controllers for Robot Swarms with Graph Neural Networks", CoRL 2019[1]
 Wide and Deep Graph Neural Networks with Distributed Online Learning, 2020[2]
 Local-Canonicalization Equivariant Graph Neural Networks, 2025[3]
 Spatial Temporal Graph Neural Networks for Decentralized Control, 2023[4]
 Learning Decentralized Swarms Using Rotation Equivariant GNNs, 2025[6]
 Hierarchical Equivariant GNNs for Collective Motion, 2025[7]
 ST-GNN Model Details and Results, 2023[5]

출처
[1] 1903.10527v4.pdf https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3e3966ce-a214-4f75-92e0-2cf2204d24b0/1903.10527v4.pdf
[2] Wide and Deep Graph Neural Networks with Distributed Online Learning https://ieeexplore.ieee.org/document/9415046/
[3] Local-Canonicalization Equivariant Graph Neural Networks for Sample-Efficient and Generalizable Swarm Robot Control https://arxiv.org/abs/2509.14431
[4] Spatial Temporal Graph Neural Networks for Decentralized Control of Robot Swarms https://dl.acm.org/doi/10.1145/3589132.3625630
[5] Spatial Temporal Graph Neural Networks for Decentralized ... https://people.cs.vt.edu/~clu/Publication/2023/ACM-SIGSPATIAL-2023-Chen.pdf
[6] Learning Decentralized Swarms Using Rotation Equivariant Graph Neural Networks https://arxiv.org/abs/2502.17612
[7] Hierarchical equivariant graph neural networks for ... https://www.nature.com/articles/s42005-025-02417-2
[8] VGAI: A Vision-Based Decentralized Controller Learning Framework for Robot Swarms https://www.semanticscholar.org/paper/2c59af1069216e709a35777e8f6b742e18eb1e21
[9] Editorial Message Vol. 27 No. 1 2026 https://journals.iium.edu.my/ejournal/index.php/iiumej/article/view/4129
[10] Multi-robot System in Coverage Control: Deployment, Coverage, and Rendezvous https://hammer.purdue.edu/articles/thesis/Multi-robot_System_in_Coverage_Control_Deployment_Coverage_and_Rendezvous/12241037/1
[11] SwarmNet: A Graph Based Learning Framework for Creating and Understanding Multi-Agent System Behaviors https://www.semanticscholar.org/paper/20b473303812dcd862a9dc5f6711a4b2952ccb01
[12] Learning Scalable Decentralized Controllers for Heterogeneous Robot Swarms with Graph Neural Networks https://asmedigitalcollection.asme.org/dynamicsystems/article/146/6/061107/1201143/Learning-Scalable-Decentralized-Controllers-for
[13] Graph Neural Swarm Control: Reliability Guarantees for Disaster-Response Drones https://www.nationaleducationservices.org/graph-neural-swarm-control-reliability-guarantees-for-disasterresponse-drones/pid-2232350603
[14] Convexified Graph Neural Networks for Distributed Control in Robotic Swarms https://www.ijcai.org/proceedings/2021/318
[15] Spatial Temporal Graph Neural Networks for Decentralized Control of Robot Swarms https://dl.acm.org/doi/pdf/10.1145/3589132.3625630
[16] Robotic Hierarchical Graph Neurons. A novel implementation of HGN for
  swarm robotic behaviour control https://arxiv.org/pdf/1910.12415.pdf
[17] Learning Decentralized Controllers for Robot Swarms with Graph Neural
  Networks https://arxiv.org/pdf/1903.10527.pdf
[18] Learning Decentralized Flocking Controllers with Spatio-Temporal Graph
  Neural Network https://arxiv.org/abs/2309.17437
[19] Imitation Learning with Graph Neural Networks for Improving Swarm Robustness under Restricted Communications https://www.mdpi.com/2076-3417/11/19/9055/pdf
[20] Graph-Based Dynamics and Network Control of a Single Articulated Robotic
  System https://arxiv.org/abs/2503.01101
[21] Asynchronous Perception-Action-Communication with Graph Neural Networks https://arxiv.org/pdf/2309.10164.pdf
[22] Multi-Robot Collaborative Perception with Graph Neural Networks https://arxiv.org/pdf/2201.01760.pdf
[23] Asynchronous Perception-Action-Communication with Graph Neural Networks https://arxiv.org/html/2309.10164
[24] Enhancing Heterogeneous Multi-Agent Cooperation in ... https://arxiv.org/abs/2408.06503
[25] Generalized neural decoders for transfer learning across ... https://www.biorxiv.org/content/10.1101/2020.10.30.362558.full
[26] Learning Decentralized Swarms Using Rotation Equivariant Graph Neural Networks https://arxiv.org/abs/2502.17612v1
[27] Graph Neural Network-based Multi-agent Reinforcement ... https://arxiv.org/html/2403.13093v1
[28] Evolutionary Optimization of Physics-Informed Neural ... https://arxiv.org/html/2501.06572v5
[29] Learning Decentralized Controllers for Robot Swarms with Graph Neural Networks https://arxiv.org/abs/1903.10527
[30] Decentralized Task Allocation Using GraphSAGE and Multi ... https://arxiv.org/html/2503.07662v1
[31] bio-inspired fine-tuning for selective transfer https://www.arxiv.org/pdf/2601.11235.pdf
[32] Local-Canonicalization Equivariant Graph Neural ... https://arxiv.org/html/2509.14431v1
[33] Multi-Agent Graph Neural Network-based Efficient Task ... https://arxiv.org/html/2502.02311v2
[34] Transfer Learning Applied to Computer Vision Problems https://arxiv.org/html/2409.07736v1
[35] A Framework for Real-World Multi-Robot Systems Running ... https://arxiv.org/abs/2111.01777
[36] PSO-Convolutional Neural Networks with Heterogeneous ... https://arxiv.org/pdf/2205.10456.pdf
[37] Hierarchical RNNs with graph policy and attention for drone ... https://academic.oup.com/jcde/article/11/2/314/7633965
[38] End-to-end decentralized formation control using a graph ... https://pmc.ncbi.nlm.nih.gov/articles/PMC10661938/
[39] On the use of evolutionary and swarm intelligence ... https://medcraveonline.com/IJBSBE/IJBSBE-08-00235.pdf
[40] Graph-based multi-agent reinforcement learning for large-scale UAVs swarm system control https://www.sciencedirect.com/science/article/abs/pii/S1270963824002992
[41] Decentralized Neural Network Policies https://www.emergentmind.com/topics/decentralized-neural-network-policies
[42] A Survey on Computational Intelligence-based Transfer ... https://arxiv.org/pdf/2206.10593.pdf
[43] End-to-end decentralized formation control using a graph ... https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2023.1285412/full
[44] Learning Decentralized Controllers for Robot Swarms with ... http://proceedings.mlr.press/v100/tolstaya20a/tolstaya20a.pdf
[45] Neural network algorithm with transfer learning and ... https://www.sciencedirect.com/science/article/abs/pii/S0950705124012668
[46] Graph Neural Networks for Decentralized Multi-Agent ... https://arxiv.org/abs/2301.09689
[47] Generalization and Transfer Learning in Neural Networks ... https://escholarship.org/content/qt1q2683zm/qt1q2683zm_noSplash_5a263810a8f20c7c0b82916c9f3f8b36.pdf?t=recjia
[48] Learning Decentralized Controllers for Robot Swarms with ... https://proceedings.mlr.press/v100/tolstaya20a.html
[49] Leveraging graph neural networks and multi-agent ... https://www.sciencedirect.com/science/article/pii/S0098135425001152
[50] [해외논문] On generalization error of neural network models and ... https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=NART122422496
[51] One-Shot Imitation Learning With Graph Neural Networks for Pick-and-Place Manipulation Tasks https://ieeexplore.ieee.org/document/10202200/
[52] Learning based multi-robot coverage algorithm https://www.ewadirect.com/proceedings/ace/article/view/11020
[53] Learning Invariant Representations of Graph Neural Networks via Cluster Generalization https://arxiv.org/abs/2403.03599
[54] Enhancing Size Generalization in Graph Neural Networks through Disentangled Representation Learning https://arxiv.org/abs/2406.04601
[55] Templates and Graph Neural Networks for Social Robots Interacting in Small Groups of Varying Sizes https://ieeexplore.ieee.org/document/10973917/
[56] Survey on Generalization Theory for Graph Neural Networks https://arxiv.org/abs/2503.15650
[57] RoboBallet: Planning for multirobot reaching with graph neural networks and reinforcement learning https://www.science.org/doi/10.1126/scirobotics.ads1204
[58] Graph Neural Networks for Learning Equivariant Representations of Neural Networks https://arxiv.org/abs/2403.12143
[59] Learning to Variable Selection with Hybrid Convolutional and Attentional Graph Neural Networks https://ieeexplore.ieee.org/document/11065519/
[60] On the Interplay between Graph Structure and Learning Algorithms in Graph Neural Networks https://arxiv.org/abs/2508.14338
[61] NeuroCERIL: Robotic Imitation Learning via Hierarchical Cause-Effect
  Reasoning in Programmable Attractor Neural Networks https://arxiv.org/pdf/2211.06462.pdf
[62] Generalized Robot Learning Framework https://arxiv.org/html/2409.12061
[63] Efficient and Interpretable Robot Manipulation with Graph Neural
  Networks https://arxiv.org/pdf/2102.13177.pdf
[64] Instant Policy: In-Context Imitation Learning via Graph Diffusion https://arxiv.org/abs/2411.12633
[65] Offline Imitation Learning Through Graph Search and Retrieval https://arxiv.org/html/2407.15403v1
[66] An Adaptive Imitation Learning Framework for Robotic Complex Contact-Rich Insertion Tasks https://pmc.ncbi.nlm.nih.gov/articles/PMC8787218/
[67] Dynamic Motion Planning Model for Multirobot Using Graph Neural Network and Historical Information https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/aisy.202300036
[68] PLANRL: A Motion Planning and Imitation Learning Framework to Bootstrap
  Reinforcement Learning https://arxiv.org/pdf/2408.04054v1.pdf
[69] Improving Generalization Ability of Robotic Imitation ... https://arxiv.org/pdf/2507.22380.pdf
[70] Learning Decentralized Wireless Resource Allocations ... https://arxiv.org/pdf/2107.01489.pdf
[71] 1 Introduction https://arxiv.org/html/2601.12244v1
[72] Generalization Capability for Imitation Learning https://arxiv.org/pdf/2504.18538.pdf
[73] Scalable Perception-Action-Communication Loops with ... https://arxiv.org/pdf/2106.13358.pdf
[74] Towards applied swarm robotics: current limitations and ... https://pdfs.semanticscholar.org/e19c/7aa3b1f6979d7f687d70528da49d983a38a2.pdf
[75] Robotic Manipulation via Imitation Learning: Taxonomy ... https://arxiv.org/html/2508.17449v1
[76] DRew: Dynamically Rewired Message Passing with Delay https://arxiv.org/pdf/2305.08018.pdf
[77] Discrete-Guided Diffusion for Scalable and Safe Multi- ... https://arxiv.org/pdf/2508.20095.pdf
[78] Improving Generalization Ability of Robotic Imitation ... https://arxiv.org/html/2507.22380v1
[79] GAP: Differentially Private Graph Neural Networks with ... https://arxiv.org/pdf/2203.00949.pdf
[80] Swarm Learning: A Survey of Concepts, Applications, and ... https://arxiv.org/html/2405.00556v2
[81] Compose by Focus: Scene Graph-based Atomic Skills https://arxiv.org/abs/2509.16053
[82] Delay-Oriented Distributed Scheduling with TransGNN https://arxiv.org/html/2512.08799v1
[83] swarm learning:asurvey of concepts, applications https://arxiv.org/pdf/2405.00556.pdf
[84] Generalizability of Graph Neural Networks for Decentralized Unlabeled Motion Planning http://www.arxiv.org/abs/2409.19829
[85] Towards applied swarm robotics: current limitations and ... https://pmc.ncbi.nlm.nih.gov/articles/PMC12202227/
[86] Graph Neural Networks for Multi-Robot Active Information Acquisition https://www.georgejpappas.org/wp-content/uploads/2023/05/Graph_Neural_Networks_for_Multi-Robot_Active_Information_Acquisition.pdf
[87] DRew: Dynamically Rewired Message Passing with Delay https://proceedings.mlr.press/v202/gutteridge23a/gutteridge23a.pdf
[88] A Decade-Long Review of Swarm Robotics Technologies https://pmc.ncbi.nlm.nih.gov/articles/PMC12526905/
[89] Combining Self-Organizing and Graph Neural Networks for ... https://pmc.ncbi.nlm.nih.gov/articles/PMC7806087/
[90] GAP: Differentially Private Graph Neural Networks with ... https://www.usenix.org/system/files/usenixsecurity23-sajadmanesh.pdf
[91] Scalable and cohesive swarm control based on ... https://www.sciencedirect.com/science/article/pii/S2667241324000053
[92] Long_term_planning_using_GNNs__NeurIPS_workshop_ https://physical-reasoning.github.io/assets/pdf/papers/00.pdf
[93] Multi-hop Attention Graph Neural Networks - MoonNote https://kisungmoon.tistory.com/87
[94] Swarm Robotics-AI 기반 모바일 지붕 차양 시스템 https://conf.aik.or.kr/pdfs/output/paper_250916145512056.pdf
[95] Efficient and Interpretable Robot Manipulation with Graph Neural ... https://arxiv.org/abs/2102.13177
[96] Multi-hop Attention Graph Neural Network https://openreview.net/forum?id=muppfCkU9H1
[97] Rotational Sampling: A Plug-and-Play Encoder for Rotation-Invariant 3D Molecular GNNs https://arxiv.org/abs/2507.01073
[98] RIMeshGNN: A Rotation-Invariant Graph Neural Network for Mesh Classification https://ieeexplore.ieee.org/document/10483674/
[99] E(Q)AGNN-PPIS: Attention Enhanced Equivariant Graph Neural Network for Protein-Protein Interaction Site Prediction https://ieeexplore.ieee.org/document/11077994/
[100] Importance of equivariant and invariant symmetries for fluid flow modeling https://arxiv.org/abs/2307.05486
[101] Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs https://arxiv.org/pdf/2302.03655.pdf
[102] Deep Neural Networks with Efficient Guaranteed Invariances https://arxiv.org/pdf/2303.01567.pdf
[103] On the Fourier analysis in the SO(3) space : EquiLoPO Network http://arxiv.org/pdf/2404.15979.pdf
[104] Hierarchical equivariant graph neural networks for forecasting
  collective motion in vortex clusters and microswimmers https://arxiv.org/html/2501.00626v1
[105] REMuS-GNN: A Rotation-Equivariant Model for Simulating Continuum
  Dynamics http://arxiv.org/pdf/2205.07852.pdf
[106] GemNet: Universal Directional Graph Neural Networks for Molecules https://arxiv.org/pdf/2106.08903.pdf
[107] Unsupervised Learning of Group Invariant and Equivariant Representations http://arxiv.org/pdf/2202.07559.pdf
[108] Rotation-equivariant graph neural networks for learning glassy liquids representations https://scipost.org/10.21468/SciPostPhys.16.5.136/pdf
[109] mofflow: flow matching for structure pre https://arxiv.org/pdf/2410.17270.pdf
[110] space-time graph neural networks https://arxiv.org/pdf/2110.02880.pdf
