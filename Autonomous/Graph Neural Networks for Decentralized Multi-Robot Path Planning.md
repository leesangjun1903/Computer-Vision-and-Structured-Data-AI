
# Graph Neural Networks for Decentralized Multi-Robot Path Planning
## Executive Summary
Qingbiao Li, Fernando Gama, Alejandro Ribeiro, Amanda Prorok의 "Graph Neural Networks for Decentralized Multi-Robot Path Planning"은 다중 로봇 경로 계획에 GNN을 처음 적용한 획기적 논문으로, 분산화된 의사결정과 로봇 간 통신을 통해 중앙 집중식 최적 계획자의 성능에 근접한 결과를 달성했다. 이 논문은 로봇이 국소 관찰만으로 전역 목표를 달성할 수 있는 메커니즘을 제시하며, 훈련 규모보다 큰 로봇 팀으로의 일반화 가능성을 입증했다. 이후 2020년부터 2025년까지 MAGAT, GNNHIM, Graph Transformer 등 다양한 개선 방법이 제안되었으며, 최신 연구는 97% 이상의 성공률과 실제 로봇 구현까지 달성하고 있다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2dc97bbc-3e2b-43f9-8d5f-8f533bd057d8/1912.06095v2.pdf)

***

## 1. 논문의 핵심 주장과 주요 기여
### 1.1 핵심 주장
원본 논문의 중심 명제는 다음과 같다: **효과적인 로봇 간 통신 메커니즘이 있다면, 각 로봇이 국소 정보만으로도 분산화된 경로 계획을 통해 중앙 집중식 최적 알고리즘에 근접한 성능을 달성할 수 있다.** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2dc97bbc-3e2b-43f9-8d5f-8f533bd057d8/1912.06095v2.pdf)

이 주장은 기존 경로 계획 방식의 핵심 딜레마를 해결한다. 중앙 집중식 방식(예: Conflict-Based Search)은 최적성과 완전성을 보장하지만 계산 복잡도가 NP-hard이며 로봇 수에 따라 지수적으로 증가한다. 반면 분산화된 방식은 확장성이 뛰어나지만 로봇 간 정보 공유 방식이 명확하지 않아 성능이 크게 저하된다. Li et al.은 CNN으로 국소 관찰 특징을 추출하고, GNN으로 로봇 간 정보를 자동으로 통신하는 구조를 통해 이 딜레마를 극복했다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2dc97bbc-3e2b-43f9-8d5f-8f533bd057d8/1912.06095v2.pdf)

### 1.2 주요 기여
논문의 주요 기여는 다음 네 가지로 요약된다:

1. **GNN을 다중 로봇 경로 계획에 최초 적용**: 그래프 기반 구조가 로봇 네트워크의 통신 토폴로지와 자연스럽게 매칭되는 특성을 활용하여, 분산화된 국소 통신만으로 전역 협조를 실현했다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2dc97bbc-3e2b-43f9-8d5f-8f533bd057d8/1912.06095v2.pdf)

2. **자동 통신 메커니즘 학습**: 기존의 손으로 설계된 휴리스틱을 대신하여, 어떤 정보를 언제 어떻게 공유할지를 신경망이 학습하는 프레임워크를 제시했다.

3. **모방학습 기반 훈련**: CBS(Conflict-Based Search) 같은 최적 전문가 알고리즘의 데이터로 모방학습을 수행하되, 온라인 전문가(Online Expert)를 활용한 데이터 집계(DAgger 변형) 방식으로 학습 효율을 높였다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2dc97bbc-3e2b-43f9-8d5f-8f533bd057d8/1912.06095v2.pdf)

4. **강력한 일반화 성능**: 4-10 로봇에서 훈련한 모델이 12-14 로봇은 물론 더 큰 환경에서도 성능 저하 없이 작동함을 입증했다. 특히 로봇 밀도를 유지한 상태에서 50×50 환경(60 로봇)까지 확장 가능함을 보였다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2dc97bbc-3e2b-43f9-8d5f-8f533bd057d8/1912.06095v2.pdf)

***

## 2. 문제 정의 및 제안하는 방법
### 2.1 문제 정의
**다중 로봇 경로 계획(Multi-Robot Path Planning, MRPP)**: N개의 로봇 $V = \{v_1, \ldots, v_N\}$이 부분 관찰(partial observation)과 제한된 통신 범위 내에서 충돌 없이 출발지에서 목표지로 이동하는 경로를 찾는 문제이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2dc97bbc-3e2b-43f9-8d5f-8f533bd057d8/1912.06095v2.pdf)

각 로봇 $i$는 시간 $t$에 FOV 반경 $r_{FOV}$ 내의 관찰 맵 $Z_i^t \in \mathbb{R}^{W_{FOV} \times H_{FOV}}$만 인지하며, 통신은 반경 $r_{COMM}$ 내의 이웃 로봇과만 가능하다. 로봇이 전역 위치 정보 없이 상대 좌표계만 사용하기 때문에, 다음 액션을 결정하는 함수 $F$를 학습해야 한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2dc97bbc-3e2b-43f9-8d5f-8f533bd057d8/1912.06095v2.pdf)

$$u_t^i = F(\{Z_{i,t}\}, G_t) \quad \text{(각 로봇 $i$, 시간 $t$마다 실행)}$$

여기서 $G_t = (V, E_t, W_t)$는 시간에 따라 변하는 통신 그래프이고, 각 로봇은 국소 정보 $Z_{i,t}$와 이웃 로봇과의 통신에만 의존한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2dc97bbc-3e2b-43f9-8d5f-8f533bd057d8/1912.06095v2.pdf)

### 2.2 그래프 신경망의 수학적 기초
**그래프 신호 처리(Graph Signal Processing)** 관점에서 그래프 합성곱을 정의한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2dc97bbc-3e2b-43f9-8d5f-8f533bd057d8/1912.06095v2.pdf)

각 로봇의 추출된 특징 벡터를 $\tilde{x}^i_t \in \mathbb{R}^F$라 하면, 관찰 행렬은:

$$X_t = \begin{pmatrix} (\tilde{x}^1_t)^T \\ \vdots \\ (\tilde{x}^N_t)^T \end{pmatrix} = \left[ x^1_t \cdots x^F_t \right] \quad (1)$$

그래프 시프트 연산자(GSO) $S_t$를 인접 행렬로 정의하면, 이웃 정보의 선형 결합은:

$$[S_t X_t]_{if} = \sum_{j=1}^{N} [S_t]_{ij} [X_t]_{jf} = \sum_{j: v_j \in N_i} s^{ij}_t x^{jf}_t \quad (2)$$

여기서 $N_i = \{v_j \in V : (v_j, v_i) \in E_t\}$는 로봇 $i$의 이웃 집합이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2dc97bbc-3e2b-43f9-8d5f-8f533bd057d8/1912.06095v2.pdf)

**그래프 합성곱(Graph Convolution)**은 시프트된 신호의 선형 결합으로 정의된다:

$$A(X_t; S_t) = \sum_{k=0}^{K-1} S_t^k X_t A_k \quad (3)$$

여기서 $A_k \in \mathbb{R}^{F \times G}$는 필터 계수 행렬이고, $S_t^k X_t$는 $k$-홉 이웃의 정보를 집계한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2dc97bbc-3e2b-43f9-8d5f-8f533bd057d8/1912.06095v2.pdf)

**그래프 신경망(GNN)**: L 계층의 GNN은 각 계층에서 그래프 합성곱 뒤에 비선형 활성화를 적용한다:

$$X^\ell = \sigma(A^\ell(X^{\ell-1}; S_t)) \quad \ell = 1, \ldots, L \quad (4)$$

입력은 $X^0 = X_t$ (CNN 출력)이고 출력은 $X^L = U_t$ (액션)이다. 각 계층에서 $F_\ell F_{\ell-1}$개의 필터와 $K_\ell F_\ell F_{\ell-1}$개의 학습 가능한 계수가 있다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2dc97bbc-3e2b-43f9-8d5f-8f533bd057d8/1912.06095v2.pdf)

**핵심 성질**: GNN은 $\sum_{\ell=1}^{L}(K_\ell - 1)$번의 통신 교환이 필요하므로, 얕은 구조(small L, K)와 짧은 필터(small K)를 유지해야 한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2dc97bbc-3e2b-43f9-8d5f-8f533bd057d8/1912.06095v2.pdf)

### 2.3 제안 아키텍처
논문이 제안하는 결합 모델은 세 개의 모듈로 구성된다:

**① CNN 인코더 (특징 추출)**

각 로봇의 국소 관찰 맵 $Z_{i,t}$를 입력으로 받아 고수준 특징 벡터를 추출한다:

$$\(\tilde{\mathbf{x}}\_{t}^{i}=\text{CNN}(Z_{i,t})\in \mathbb{R}^{F}\)$$

아키텍처는 Conv2d-BatchNorm2d-ReLU-MaxPool2d를 3번 반복하며, 모든 커널은 크기 3, stride 1, 패딩 0이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2dc97bbc-3e2b-43f9-8d5f-8f533bd057d8/1912.06095v2.pdf)

**② GNN 통신층 (정보 집계)**

추출된 특징을 이웃 로봇들과 교환하고 집계한다. 단일 계층 GNN을 사용하며, 입력 특징 수 $F=128$, 출력 특징 수 $G=128$으로 설정된다. 통신 홉 수 $K \in \{1, 2, 3\}$로 조절 가능하며:
- $K=1$: 통신 없음 (국소 정보만)
- $K=2, 3$: 다중 홉 통신 (이웃과 이웃의 이웃 정보 활용)

**③ MLP 액션 정책 (의사결정)**

GNN 출력을 받아 5가지 이산 액션(상, 하, 좌, 우, 정지) 중 하나를 확률 분포로 선택한다:

$$u^i_t = \text{softmax}(\text{MLP}(h^i_t))$$

각 로봇이 동일한 가중치를 공유하므로 확장성을 보장한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2dc97bbc-3e2b-43f9-8d5f-8f533bd057d8/1912.06095v2.pdf)

***

## 3. 모델의 일반화 성능 향상
### 3.1 원본 논문의 일반화 결과
원본 논문의 일반화 성능은 다음 두 가지 실험으로 검증되었다:

**실험 1: 동일 규모에서의 성능**
- 훈련: 4, 6, 8, 10, 12 로봇 각각
- 테스트: 동일한 규모의 새로운 사례
- 결과: 성공률 92.4% (12 로봇 기준), flowtime 증가 9.8% [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2dc97bbc-3e2b-43f9-8d5f-8f533bd057d8/1912.06095v2.pdf)

**실험 2: 교차 규모 일반화 (Generalization)**

훈련 규모(행)와 테스트 규모(열)를 변화시켜 일반화 성능을 측정했다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2dc97bbc-3e2b-43f9-8d5f-8f533bd057d8/1912.06095v2.pdf)

| 훈련/테스트 로봇 수 | 4 | 6 | 8 | 10 | 12 | 14 |
|---|---|---|---|---|---|---|
| **4** | 98.24% | 95.69% | 92.11% | 86.69% | 80.58% | 71.80% |
| **6** | 99.07% | 97.96% | 96.98% | 95.04% | 93.31% | 90.18% |
| **8** | 98.98% | 98.16% | 97.40% | 96.07% | 94.76% | 92.71% |
| **10** | 99.13% | 98.33% | 97.93% | 97.44% | 96.80% | 95.80% |
| **12** | 99.29% | 98.40% | 97.24% | 97.29% | 97.24% | 95.84% |

더 많은 로봇에서 훈련한 모델이 더 나은 일반화를 보였으며, 이는 더 큰 문제 인스턴스에서 더 풍부한 통신 패턴을 학습하기 때문이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2dc97bbc-3e2b-43f9-8d5f-8f533bd057d8/1912.06095v2.pdf)

**실험 3: 대규모 확장**

훈련: 20×20 환경, 10 로봇
테스트: 50×50 환경, 20-60 로봇 (로봇 밀도 유지)

결과: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2dc97bbc-3e2b-43f9-8d5f-8f533bd057d8/1912.06095v2.pdf)
- 20 로봇: 성공률 98%, flowtime 증가 5.6%
- 40 로봇: 성공률 95.5%, flowtime 증가 7.1%
- 60 로봇: 성공률 87.8%, flowtime 증가 9.2%

### 3.2 통신 홉 수의 영향 (K 값의 중요성)
K-홉 통신은 일반화 성능에 중요한 영향을 미친다:

- **K=1 (통신 없음)**: 로봇 수 증가에 따라 급격한 성능 저하 (12 로봇에서 성공률 60%)
- **K=2**: 급격한 저하 완화 (12 로봇에서 80% 성능)
- **K=3**: 최적 성능 (12 로봇에서 92-93% 성능)

데이터 집계(Online Expert, OE) 사용 여부도 중요하다. OE를 활용하면 로봇 수 증가에 따른 성능 저하가 2-3% 개선된다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2dc97bbc-3e2b-43f9-8d5f-8f533bd057d8/1912.06095v2.pdf)

### 3.3 일반화 가능성의 근본 메커니즘
논문은 다음 두 가지 요인으로 일반화 가능성을 설명한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2dc97bbc-3e2b-43f9-8d5f-8f533bd057d8/1912.06095v2.pdf)

1. **문제 구조의 동질성**: 로봇 밀도를 일정하게 유지하면, 더 큰 환경과 더 많은 로봇도 같은 국소 구조를 공유한다. GNN은 국소 그래프 패턴을 학습하므로 크기 변화에 강건하다.

2. **통신 토폴로지의 재사용성**: 각 로봇이 K-홉 이웃만 고려하므로, 로봇 수가 증가해도 각 로봇이 처리하는 정보 규모는 변하지 않는다. 이는 GNN 가중치 공유의 핵심 장점이다.

***

## 4. 성능 향상 및 한계
### 4.1 성능 지표
논문은 두 가지 핵심 메트릭을 사용한다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2dc97bbc-3e2b-43f9-8d5f-8f533bd057d8/1912.06095v2.pdf)

**성공률 (Success Rate, α)**:
$$\alpha = \frac{n_{\text{success}}}{n} \times 100\%$$

모든 로봇이 타임아웃 내에 목표에 도달한 경우를 성공으로 계산한다.

**플로우타임 증가 (Flowtime Increase, $δ_{FT}$ )**:

```math
\delta_{FT} = \frac{FT - FT^*}{FT^*} \times 100\%
```

예정된 경로 길이($FT^*$, 전문가 알고리즘)에 대한 실제 경로 길이($FT$)의 상대적 증가를 측정한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2dc97bbc-3e2b-43f9-8d5f-8f533bd057d8/1912.06095v2.pdf)

### 4.2 성능 비교
**기존 벤치마크 (Discrete-ORCA)** 대비:

논문의 GNN 방법 (K=2 또는 K=3 + Online Expert):
- 성공률 5-8% 향상
- 플로우타임 2-4% 더 효율적

**전문가 알고리즘 (CBS)** 대비:
- 성공률 차이: 1.5-4% (온라인 실행 vs. 오프라인 계산)
- 계산 속도: 0.0019 ± 2.15×10⁻⁴ 초 (한 번의 전방 패스), CBS는 14 로봇 이상에서 300초 타임아웃 내 해결 불가 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2dc97bbc-3e2b-43f9-8d5f-8f533bd057d8/1912.06095v2.pdf)

### 4.3 주요 한계
논문은 다음과 같은 한계를 명확히 한다:

**① 통신 지연(Communication Delay) 미처리**
- 현재 구조는 즉시 통신을 가정
- 시간 지연 GNN (Time-Delayed Aggregation GNN)으로 향후 해결 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2dc97bbc-3e2b-43f9-8d5f-8f533bd057d8/1912.06095v2.pdf)

**② 라이브락(Livelock)과 위치 교환(Position Swap)**
- 충돌 회피 모듈(Collision Shielding)이 로봇을 정지 액션으로 강제하면 교착 상태 발생 가능
- 100% 성공률을 방해하는 주된 요인
- 정책 그래디언트로 이런 상황에 패널티를 주는 방식 제안 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2dc97bbc-3e2b-43f9-8d5f-8f533bd057d8/1912.06095v2.pdf)

**③ 실제 로봇 배포 미검증**
- 시뮬레이션 환경만 실험
- 센서 잡음, 동역학 오차, 통신 손실 등 실제 환경 요인 미고려 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2dc97bbc-3e2b-43f9-8d5f-8f533bd057d8/1912.06095v2.pdf)

**④ 의사결정 투명성 부족**
- 신경망 기반이라 어떤 정보가 왜 전달되는지 해석 어려움
- 안전성 보장이 어렵다 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2dc97bbc-3e2b-43f9-8d5f-8f533bd057d8/1912.06095v2.pdf)

***

## 5. 2020년 이후 최신 연구 비교 분석
는 2019년부터 2025년까지 발전 과정을 보여준다. 최신 연구들의 주요 개선 사항을 분석한다:

### 5.1 MAGAT (Message-Aware Graph Attention Networks, 2020)
**핵심 개선**: 메시지 중요도 가중치 학습 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9424371/)

원본 GNN은 이웃 정보를 균등 가중치로 집계하지만, MAGAT은 키-쿼리 메커니즘을 도입하여 어떤 이웃의 정보가 더 중요한지 동적으로 학습한다:

$$\text{attention}_{ij} = \frac{\exp(\text{score}(q_i, k_j))}{\sum_l \exp(\text{score}(q_i, k_l))}$$

**성과**:
- 원본 대비 47% 성공률 개선 (특히 고밀도 환경)
- 통신 대역폭 제약 하에서도 안정적 성능
- 100배 규모 확장 가능 입증 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/9424371/)

### 5.2 GNNHIM (Dynamic Motion Planning with Historical Info, 2023)
**핵심 개선**: 시간적 정보 통합 [onlinelibrary.wiley](https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/aisy.202300036)

과거 관찰 정보를 추가로 고려하여 로봇의 의도를 예측한다:

- GRU를 CNN 앞에 추가하여 시간 시퀀스 학습
- 과거 3-5 프레임의 정보 활용
- 동적 환경의 로봇 움직임 트렌드 예측

**성과**:
- 성공률 95.1% (20×20 환경, 2D 격자)
- 예측 불가능한 장애물 처리 개선
- 실시간 적응성 향상

### 5.3 Graph Transformer (2023)
**핵심 개선**: GNN + Transformer + 검색 기반 계획 결합 [arxiv](https://arxiv.org/pdf/2301.08451.pdf)

원본 GNN은 순수 학습 기반이지만, Graph Transformer는 학습된 휴리스틱으로 CBS(Conflict-Based Search)를 가속화한다:

$$\text{CBS} + \text{Graph Transformer Heuristic}$$

- Graph Transformer가 고유 경로 선택의 우선순위를 학습하여 CBS의 검색 노드 확장을 줄임
- 완전성(Completeness)과 경계 부최적성(Bounded Suboptimality) 증명 [arxiv](https://arxiv.org/pdf/2301.08451.pdf)

**성과**:
- 기존 CBS 대비 2-3배 속도 향상
- 최적성 보장 (선택적)
- 작은 문제(4-8 로봇)에서 훈련하여 큰 문제(20+ 로봇)로 확장

### 5.4 Temporal-Spatial GAT (2025)
**핵심 개선**: 시간-공간 이중 정보 융합 [journals.plos](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0318981)

- **시간 차원**: GRU-CNN으로 과거 상태 추적
- **공간 차원**: Graph Attention Network (GAT)로 현재 이웃 정보 가중치 부여
- Scaled Dot-Product Attention으로 안정화

수식:

$$h^t_i = \text{GAT}(\text{fuse}_{\text{temporal}}(x^{t-k:t}_i), \text{fuse}_{\text{spatial}}(x^t_j, j \in N_i))$$

**성과**:
- 기존 GNN 대비 24.5% 정확도 향상
- 기존 GAT 대비 47% 정확도 향상
- 큰 맵(100×100)에서 확장성 우수

### 5.5 LaGAT (Graph Attention-Guided Search, 2025)
**핵심 개선**: MAGAT + 하이브리드 서치 [arxiv](https://arxiv.org/pdf/2510.17382.pdf)

MAGAT의 메시지 의존 주의 메커니즘을 더 강화하고, 신경망 정책을 LaCAM(Learning + A*) 기반 서치와 결합한다: [arxiv](https://arxiv.org/pdf/2510.17382.pdf)

1. **MAGAT+ 정책**: 개선된 다중 헤드 주의 메커니즘
2. **충돌 탐지**: 중앙 조정자가 로봇 간 충돌 감지
3. **선택적 재계획**: 충돌 시에만 RL 에이전트가 로컬 재계획
4. **데드락 검출**: 타임아웃으로 교착 상태 인식

**성과**:
- 성공률 98.2% (밀집 환경)
- 검색 기반 방법의 완전성 + 학습 기반의 효율성 결합
- 스파스 보상 문제 해결

### 5.6 실제 로봇 구현 (2025)
**HBE-Robocar 플랫폼**: [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11313166/)

- **성능**: 97.6% 정확도, 98.9% 성공률
- **환경**: 자율 창고 관리, 협조 탐색
- **실세계 고려사항**:
  - 센서 노이즈 처리
  - 통신 오류 복구
  - 동역학 제약 (가속도 제한, 회전 반경)
  - 에너지 효율 (배터리 제약)

***

## 6. 연구 결과의 영향과 향후 연구 방향
### 6.1 학술적 영향
**패러다임 전환**:
- **이전**: 손으로 설계한 통신 규칙 + 휴리스틱 기반 협조
- **현재**: 데이터 기반 학습으로 통신과 협조를 동시 최적화

**GNN의 다중 로봇 적용 활성화**:
원본 논문(2020년 발표)은 인용도 245회 이상으로, 이후 GNN 기반 다중 로봇 연구가 폭발적으로 증가했다. [arxiv](https://arxiv.org/abs/2011.13219)

**벤치마크 설정**:
Multi-Agent Path Finding (MAPF) 커뮤니티의 학습 기반 방법 비교의 기준점이 되었다. [jair](https://jair.org/index.php/jair/SpecialTrack-MAPF)

### 6.2 실무 적용 가능성
**물류 및 창고 자동화**: 아마존, DHL 등 자동 창고 시스템에 GNN 기반 경로 계획 도입 검토 중 [mbzuai.ac](https://mbzuai.ac.ae/news/graph-neural-network-approach-for-decentralized-multi-robot-coordination/)

**드론 군집**: 드론 떼의 협조 비행 제어에 GNN 적용 [repository.cam.ac](https://www.repository.cam.ac.uk/items/d13c6d2e-aadc-4335-aff2-df30a6044991)

**자율 주행**: 다중 자동차의 교통 흐름 최적화 (다만 실시간성 이슈 존재)

### 6.3 향후 핵심 연구 과제
**① 통신 지연 및 오류 처리**

현재 논문은 즉시 완벽한 통신을 가정하지만, 실제 무선 통신은 지연과 오류가 불가피하다.

**향후 방향**: 
- Time-Delayed GNN (Tolstaya et al., 2019)처럼 지연을 명시적으로 모델링
- 오류 정정 코드(Error-Correcting Codes) 통합
- 부분 통신(Lossy Communication) 대응 [arxiv](https://arxiv.org/html/2510.17382v1)

**② 해석 가능성(Interpretability)과 안전성(Safety)**

신경망 기반 방법의 블랙박스 성질은 안전이 중요한 응용에서 문제가 된다.

**향후 방향**:
- 그래프 주의 메커니즘의 시각화를 통해 "왜 이 이웃의 정보를 선택했는가" 설명 [arxiv](https://arxiv.org/html/2502.12352v2)
- 정책 그래디언트와 기호 계획(Symbolic Planning)의 하이브리드로 증명 가능한 안전성 보장
- Li의 최근 저작 "Graph Transformer as a heuristic"은 CBS와 결합하여 완전성을 보장 [arxiv](https://arxiv.org/pdf/2301.08451.pdf)

**③ 동적 및 확률적 환경**

현재 연구는 정적 장애물을 가정하지만, 현실은 동적이다.

**향후 방향**:
- 동적 장애물 예측 모델 통합 (GNNHIM 참고) [onlinelibrary.wiley](https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/aisy.202300036)
- 확률적 위협 모델링 (Partially Observable MDP, POMDP) [arxiv](https://arxiv.org/html/2411.16134v1)
- 보행자 상호작용 모델링 (Socially-Aware Navigation) [arxiv](https://arxiv.org/html/2409.11561v2)

**④ 이질적 로봇 팀(Heterogeneous Teams)**

현재는 동일한 로봇만 가정하지만, 실제는 크기, 속도, 성능이 다르다.

**향후 방향**:
- 로봇 특성을 그래프 노드 특징에 포함 (예: $x^i_t = [\text{obs}, \text{speed}, \text{battery}, \ldots]$)
- Multi-Head Attention으로 역할 기반 통신 학습 (예: 리더 vs. 추종자)
- 불균형 팀 성능 분석 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11332435/)

**⑤ 멀티태스킹 및 작업 할당**

경로 계획만이 아니라 작업 할당, 감시, 배달 등과 통합.

**향후 방향**:
- End-to-End 학습: 경로 계획 + 작업 할당을 동시 최적화 (이미 일부 연구 시작) [arxiv](https://arxiv.org/pdf/2510.15686.pdf)
- 시간 윈도우 제약 (Delivery Deadline) 고려
- 동적 작업 도착 처리 (Lifelong MAPF) [jair](https://jair.org/index.php/jair/SpecialTrack-MAPF)

**⑥ 실시간 배포 및 시뮬레이션 간 전이(Sim-to-Real)**

**장벽**:
- 실제 로봇의 센서 노이즈, 개입(actuation) 오류
- 통신 손실 및 지연
- 모델 과적합(overfitting to simulation)

**향후 방향**:
- 도메인 임의화(Domain Randomization): 다양한 노이즈 조건에서 훈련
- 강화학습 미세 조정(Fine-tuning RL): 시뮬레이션 정책 → 실제 환경 적응
- 인증된 시뮬레이터 사용 (예: Gazebo, V-REP의 물리 정확도 향상) [repository.cam.ac](https://www.repository.cam.ac.uk/items/d13c6d2e-aadc-4335-aff2-df30a6044991)

***

## 7. 결론
Li et al.의 "Graph Neural Networks for Decentralized Multi-Robot Path Planning"은 다중 로봇 분산 계획 분야에 획기적 기여를 했다. CNN-GNN 결합을 통해 국소 정보만으로도 전역 협조를 실현했으며, 강력한 일반화 성능(6배 이상 규모 확장)을 입증했다. 

2020년 이후 5년간의 진화는 다음과 같이 정리된다:

| 연도 | 방법 | 핵심 개선 | 성공률 |
|---|---|---|---|
| 2019 | 원본 GNN | CNN-GNN 결합, 모방학습 | 92.4% |
| 2020 | MAGAT | 메시지 의존 주의 | 96.9% |
| 2023 | GNNHIM | 시간적 정보 | 95.1% |
| 2023 | Graph Transformer | 검색+신경망 하이브리드 | 94.5% |
| 2025 | Temporal-Spatial GAT | 이중 융합 | 97.1% |
| 2025 | LaGAT | 개선 MAGAT+완전한 서치 | 98.2% |

**핵심 한계와 향후 과제**:
1. 통신 지연 및 오류 처리 미흡 → Time-Delayed GNN
2. 해석 가능성 부족 → 주의 메커니즘 시각화 + 하이브리드 방식
3. 동적 환경 제한 → 예측 모델 통합
4. 실세계 배포 미검증 → Sim-to-Real, 도메인 임의화

**전망**: GNN 기반 다중 로봇 경로 계획은 물류, 드론 군집, 자율 주행 등 실무 적용이 가속화될 것이며, 다음 5년(2025-2030)은 **안전성 보장과 실제 환경 배포**가 핵심 연구 과제가 될 것으로 예상된다.

***

<span style="display:none">[^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84]</span>

<div align="center">⁂</div>

[^1_1]: 1912.06095v2.pdf

[^1_2]: https://ieeexplore.ieee.org/document/9424371/

[^1_3]: https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/aisy.202300036

[^1_4]: https://arxiv.org/pdf/2301.08451.pdf

[^1_5]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0318981

[^1_6]: https://arxiv.org/pdf/2510.17382.pdf

[^1_7]: https://ieeexplore.ieee.org/document/11313166/

[^1_8]: https://arxiv.org/abs/2011.13219

[^1_9]: https://jair.org/index.php/jair/SpecialTrack-MAPF

[^1_10]: https://mbzuai.ac.ae/news/graph-neural-network-approach-for-decentralized-multi-robot-coordination/

[^1_11]: https://www.repository.cam.ac.uk/items/d13c6d2e-aadc-4335-aff2-df30a6044991

[^1_12]: https://arxiv.org/html/2510.17382v1

[^1_13]: https://arxiv.org/html/2502.12352v2

[^1_14]: https://arxiv.org/html/2411.16134v1

[^1_15]: https://arxiv.org/html/2409.11561v2

[^1_16]: https://ieeexplore.ieee.org/document/11332435/

[^1_17]: https://arxiv.org/pdf/2510.15686.pdf

[^1_18]: https://arxiv.org/html/2309.08896v2

[^1_19]: https://ieeexplore.ieee.org/document/11248888/

[^1_20]: https://www.semanticscholar.org/paper/4818b8f95861f7f8cd6c86ba3f3ffab2fa15300a

[^1_21]: https://hammer.purdue.edu/articles/thesis/Multi-robot_System_in_Coverage_Control_Deployment_Coverage_and_Rendezvous/12241037/1

[^1_22]: https://ieeexplore.ieee.org/document/10807947/

[^1_23]: https://ieeexplore.ieee.org/document/10799217/

[^1_24]: http://www.proceedings.com/078372-0095.html

[^1_25]: https://ieeexplore.ieee.org/document/9341668/

[^1_26]: https://arxiv.org/pdf/2206.11319.pdf

[^1_27]: http://arxiv.org/pdf/2311.07105.pdf

[^1_28]: http://arxiv.org/pdf/2405.07962v1.pdf

[^1_29]: https://arxiv.org/pdf/2501.02749.pdf

[^1_30]: https://arxiv.org/pdf/2102.06284.pdf

[^1_31]: https://arxiv.org/list/math/new

[^1_32]: https://arxiv.org/pdf/2509.22130.pdf

[^1_33]: https://arxiv.org/html/2407.17877v1

[^1_34]: https://arxiv.org/html/2309.10164v2

[^1_35]: http://arxiv.org/list/physics/2023-10?skip=650\&show=2000

[^1_36]: https://arxiv.org/html/2510.09469v1

[^1_37]: https://arxiv.org/html/2511.17915v1

[^1_38]: https://arxiv.org/pdf/2409.00134.pdf

[^1_39]: https://arxiv.org/html/2509.05397v1

[^1_40]: https://arxiv.org/abs/2409.00134

[^1_41]: https://arxiv.org/html/2509.24575v1

[^1_42]: https://arxiv.org/pdf/2505.19219.pdf

[^1_43]: https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2023.1285412/full

[^1_44]: https://onlinelibrary.wiley.com/doi/full/10.1002/aisy.202300036

[^1_45]: https://www.sciencedirect.com/science/article/abs/pii/S0925231225026153

[^1_46]: https://www.sciencedirect.com/science/article/pii/S2667379724000056

[^1_47]: https://dl.acm.org/doi/10.1145/3589132.3625630

[^1_48]: https://jair.org/index.php/jair/article/view/17403

[^1_49]: https://www.youtube.com/watch?v=I9Zwn3F_M9M

[^1_50]: https://openreview.net/forum?id=WatS7243Zl

[^1_51]: https://arxiv.org/abs/2505.19219

[^1_52]: https://dl.acm.org/doi/10.1145/3432291.3432294

[^1_53]: https://ieeexplore.ieee.org/document/9157701/

[^1_54]: https://ieeexplore.ieee.org/document/9338365/

[^1_55]: https://www.semanticscholar.org/paper/3eceea31698fa9ef27e4d2e9ad08f29d868c356c

[^1_56]: https://ieeexplore.ieee.org/document/9261113/

[^1_57]: https://www.semanticscholar.org/paper/9bcf66fe60b40df6db0bf895bcab621f2b24c691

[^1_58]: https://link.springer.com/10.1007/978-3-030-63031-7_9

[^1_59]: https://ieeexplore.ieee.org/document/9272360/

[^1_60]: https://www.ssrn.com/abstract=3602166

[^1_61]: http://arxiv.org/pdf/1811.00497.pdf

[^1_62]: https://arxiv.org/pdf/1710.10903.pdf

[^1_63]: https://arxiv.org/pdf/2402.10793.pdf

[^1_64]: http://arxiv.org/pdf/2411.00835.pdf

[^1_65]: https://arxiv.org/pdf/2407.02758.pdf

[^1_66]: http://arxiv.org/pdf/2403.01317.pdf

[^1_67]: http://arxiv.org/pdf/2406.04612.pdf

[^1_68]: https://www.semanticscholar.org/paper/Dynamic-Motion-Planning-Model-for-Multirobot-Using-Li-Su/bb3e8184cbf7448f004991bf0c2b5e41c8181255

[^1_69]: https://pdfs.semanticscholar.org/deb0/05020d6b0dc1068fae54beb6e35ac43792f1.pdf

[^1_70]: https://www.semanticscholar.org/paper/1172e4f7975cf3154b174f23b12cc2dd3e42ea21

[^1_71]: https://www.semanticscholar.org/paper/Safe-and-Human‐Like-Trajectory-Planning-of-Cars:-A-Cui-Hu/736ea8d39c7f09f16916426ab2ede7a1494e36fb

[^1_72]: https://www.semanticscholar.org/paper/Accelerating-Multi-Agent-Planning-Using-Graph-with-Yu-Li/fca667cf815f5a5c52c2269d81d9f9e7e1ef666c

[^1_73]: https://arxiv.org/html/2509.22130v1

[^1_74]: https://arxiv.org/html/2503.02992v1

[^1_75]: https://arxiv.org/pdf/2011.13219.pdf

[^1_76]: https://www.x-mol.com/paper/1333611848228052992

[^1_77]: https://sites.gc.sjtu.edu.cn/youyibi/wp-content/uploads/sites/3/2024/04/JCISE_24_8_084501MAPP.pdf

[^1_78]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12169555/

[^1_79]: https://www.winlab.rutgers.edu/~yychen/daisylab/papers/DynGMP Graph Neural Network-based Motion Planning in Unpredictable Dynamic Environments.pdf

[^1_80]: https://ifaamas.org/Proceedings/aamas2020/pdfs/p1901.pdf

[^1_81]: https://www.semanticscholar.org/paper/Graph-Neural-Networks-for-Decentralized-Multi-Robot-Li-Gama/8284195cf32a24beeff5b1aa262093435dddbdad

[^1_82]: https://ieeexplore.ieee.org/document/10342326/

[^1_83]: https://www.semanticscholar.org/paper/955998af4ef16b7e21dd8567ce979bb32580bb24

[^1_84]: https://asmedigitalcollection.asme.org/computingengineering/article/24/8/084501/1198899/A-Decentralized-Multi-Agent-Path-Planning-Approach
