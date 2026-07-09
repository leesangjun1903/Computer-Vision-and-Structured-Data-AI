# Video-Based Optimal Transport for Feedback-Efficient Offline Preference-Based Reinforcement Learning

## 참고 자료

**주 논문:**
- Luu, T. M., Kim, H., Lee, Y., & Yoo, C. D. (2026). *Video-Based Optimal Transport for Feedback-Efficient Offline Preference-Based Reinforcement Learning*. arXiv:2606.16856v1. ICML 2026.

**주요 인용 논문 (논문 내 참고문헌 기반):**
- Christiano et al. (2017). *Deep Reinforcement Learning from Human Preferences*. NeurIPS.
- Park et al. (2022). *SURF: Semi-supervised Reward Learning with Data Augmentation*. ICLR.
- Hejna & Sadigh (2023). *Inverse Preference Learning*. NeurIPS.
- Hejna et al. (2024). *Contrastive Preference Learning*. ICLR.
- Choi et al. (2024). *Listwise Reward Estimation (LiRE)*. ICML.
- Zhang et al. (2024). *Flow-to-Better (FTB)*. ICLR.
- Fu et al. (2020). *D4RL*. arXiv.
- Yu et al. (2020). *Meta-World*. CoRL.
- Cuturi (2013). *Lightspeed Computation of Optimal Transportation Distances*. NeurIPS.
- Kostrikov et al. (2022). *Offline RL with Implicit Q-Learning (IQL)*. ICLR.
- Liu et al. (2024). *PEARL*. ICML.
- Xie et al. (2018) & Miech et al. (2020). *S3D / HowTo100M*.
- Kim et al. (2023). *Preference Transformer*. ICLR.

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

VOTP(Video-based Optimal Transport Preference)는 **극소수의 인간 피드백(예: 10개)만으로도 고품질 보상 함수를 학습**할 수 있는 반지도 학습(semi-supervised) 프레임워크이다. 핵심 통찰은 다음과 같다:

> *인간의 선호는 시각적 행동 지각에 의해 형성되며, Video Foundation Models(ViFMs)의 풍부한 표현 공간을 활용하면 새로운 행동에 대한 선호를 기존 선호 예시와의 비교를 통해 추론할 수 있다.*

### 주요 기여

| 기여 | 설명 |
|------|------|
| **ViFM 기반 궤적 표현** | 오프더쉘프 Video Foundation Model(S3D)을 사용하여 시공간 정보를 모두 담은 궤적 임베딩 생성 |
| **OT 기반 의사 레이블 생성** | Optimal Transport를 활용한 레이블-무레이블 세그먼트 간 정렬을 통한 고품질 의사 선호 레이블 자동 생성 |
| **피드백 효율성 대폭 향상** | 기존 최신 오프라인 PbRL 대비 월등한 성능을 단 10개의 레이블로 달성 |
| **시각적 견고성** | ViFM의 일반화 능력으로 조명 변화, 텍스처 변화, 동적 배경 등 시각적 방해 요소에 강인 |
| **실제 로봇 검증** | 실제 7-DoF Sawyer 로봇 팔에서 효과 입증 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**기존 PbRL의 한계:**

- **높은 레이블링 비용:** 효과적인 보상 함수 학습을 위해 수백~수천 개의 인간 피드백 필요 (Christiano et al., 2017)
- **확장성 문제:** 레이블 수가 늘수록 인간 작업 부담 급증
- **반지도 학습의 불안정성:** 기존 SURF(Park et al., 2022)는 불완전한 보상 모델로 의사 레이블을 생성하여 **확증 편향(confirmation bias)** 발생
- **시각적 지각의 미활용:** 인간 선호가 시각적 인식에 기반함에도 이를 활용한 연구 부재

### 2.2 제안 방법

#### 2.2.1 기본 설정

MDP: $\langle \mathcal{S}, \mathcal{A}, \mathcal{T}, r, \gamma \rangle$

반환(Return):

$$G_t = \sum_{k=0}^{\infty} \gamma^k r(\mathbf{s}_{t+k}, \mathbf{a}_{t+k})$$

#### 2.2.2 선호 기반 보상 학습 (Bradley-Terry 모델)

주어진 세그먼트 쌍 $(\sigma^0, \sigma^1)$에 대한 선호 확률:

$$P(\sigma^0 \succ \sigma^1; \psi) = \frac{\exp\sum_t \hat{r}_\psi(\mathbf{s}_t^0, \mathbf{a}_t^0)}{\exp\sum_t \hat{r}_\psi(\mathbf{s}_t^0, \mathbf{a}_t^0) + \exp\sum_t \hat{r}_\psi(\mathbf{s}_t^1, \mathbf{a}_t^1)}$$

크로스 엔트로피 손실:

$$\mathcal{L}(\psi) = \mathbb{E}_{\mathcal{D}}\left[(1-\tilde{y})\log P(\sigma^0 \succ \sigma^1; \psi) + \tilde{y}\log P(\sigma^1 \succ \sigma^0; \psi)\right] \tag{1}$$

여기서 $\tilde{y} \in \{0, 1, 0.5\}$는 선호 레이블.

#### 2.2.3 궤적 표현 (ViFM 인코더)

각 세그먼트를 짧은 비디오 클립으로 모델링:

$$\mathbf{z} = f_\phi(\mathbf{o}_{1:H}) \tag{3}$$

- $f_\phi$: Video Foundation Model 인코더 (S3D, HowTo100M 사전학습)
- 공간적 세부사항 + 시간적 동역학을 모두 포착
- **Actor-agnostic** 표현으로 새로운 환경에 일반화

#### 2.2.4 이산 최적 운송 (Discrete Optimal Transport)

두 확률 측도 $\mu_x = \sum_{i=1}^n p_i \delta_{x_i}$, $\mu_y = \sum_{j=1}^m q_j \delta_{y_j}$ 간의 OT 문제:

$$\mathcal{W}_2^2(\mu_x, \mu_y) = \min_{\mu \in \mathcal{M}} \sum_{i=1}^n \sum_{j=1}^m c(x_i, y_j)\mu_{ij} \tag{2}$$

여기서 $\mathcal{M} = \{\mu \in \mathbb{R}_+^{n \times m} : \mu \mathbf{1}_m = \mu_x, \mu^\top \mathbf{1}_n = \mu_y\}$.

#### 2.2.5 의사 선호 레이블 생성 (VOTP 핵심)

**선호 행렬 정의:**

$$R_{ij} = \begin{cases} -1 & \text{if } \sigma_i \succ \sigma_j, \\ 1 & \text{if } \sigma_j \succ \sigma_i, \\ 0 & \text{for } i=j, \text{ ties, or no preference} \end{cases}$$

$R$은 반대칭 행렬: $R^\top = -R$

**OT 계획 최적화:**

$$\mu^* = \arg\min_{\mu \in \mathcal{M}} \sum_{i=1}^N \sum_{i'=1}^M c(\sigma_i, \bar{\sigma}_{i'})\mu_{ii'} \tag{4}$$

비용 함수: $c(\sigma_i, \bar{\sigma}\_{i'}) = d(f_\phi(\sigma_i), f_\phi(\bar{\sigma}_{i'}))$

**선호 점수 계산:**

$$S(\bar{\sigma}_{i'}, \bar{\sigma}_{j'}) = \sum_{i=1}^N \sum_{j=1}^N R_{ij}\left(\mu_{ii'}\mu_{jj'} - \mu_{ij'}\mu_{ji'}\right) \tag{5}$$

**해석:** $\mu_{ii'}\mu_{jj'}$는 레이블 쌍 $(i,j)$와 무레이블 쌍 $(i', j')$의 정렬 강도를 측정하고, $\mu_{ij'}\mu_{ji'}$는 역방향 정렬을 측정한다. 그 차이가 양수이면 선호가 전파된다.

**정규화:**

$$S_{\max} = \sum_{i=1}^N \sum_{j=1}^N \frac{1}{N^2} \mathbb{1}(R_{ij} \neq 0) \tag{6}$$

이를 통해 $S_{\text{norm}} \in [-1, 1]$로 정규화.

**최종 의사 레이블 결정:**

$$\tilde{y} = \begin{cases} \frac{1}{2}(1 + \text{sign}(S_{\text{norm}}(\bar{\sigma}_{i'}, \bar{\sigma}_{j'}))) & \text{if } |S_{\text{norm}}| \geq \tau_P, \\ 0.5 & \text{otherwise} \end{cases} \tag{7}$$

여기서 $\tau_P$는 선호 임계값(preference threshold).

#### 2.2.6 구현 세부사항

- **OT 솔버:** Sinkhorn 알고리즘 (POT 라이브러리, Cuturi 2013)
- **궤적 인코더:** S3D (31M 파라미터, HowTo100M 사전학습)
- **오프라인 RL:** IQL (Kostrikov et al., 2022)
- **비용 함수:** 유클리드 거리

### 2.3 모델 구조

```
[오프라인 데이터셋 B]
    ↓
[레이블된 세그먼트 쌍 Dl (N_l=10)]   [무레이블 세그먼트 쌍 Du (10k~50k)]
    ↓                                       ↓
[ViFM 인코더 f_φ (S3D)]←―――――――――――――――――――[ViFM 인코더 f_φ (S3D)]
    ↓                                       ↓
[레이블 잠재 표현 L]              [무레이블 잠재 표현 U]
    ↓                   ↓
[선호 행렬 R]    [OT 계획 μ* (Sinkhorn)]
         ↘         ↙
      [선호 점수 S (식 5)]
              ↓
      [의사 레이블 ỹ (식 7)]
              ↓
    [보상 모델 r̂_ψ 학습 (식 1)]
              ↓
    [오프라인 데이터셋 보상 재레이블링]
              ↓
    [정책 πθ 학습 (IQL)]
```

### 2.4 성능 향상

#### 정량적 결과 (Table 1)

| 방법 | D4RL 평균 | MetaWorld 평균 |
|------|-----------|----------------|
| P-IQL (기준) | 65.3 | 31.0 |
| SURF | 59.5 | 51.0 |
| LiRE | 83.2 | 64.0 |
| FTB | 85.4 | 48.6 |
| **VOTP** | **92.8** | **67.6** |
| Oracle | 92.4 | 80.1 |
| IQL+GT | 93.6 | 71.0 |

**주요 성과:**
- D4RL에서 Oracle 수준 성능을 단 10개의 레이블로 달성
- FTB 대비 훨씬 빠른 속도 (2시간 vs. 2일)
- `door-open`에서 단 10개 레이블로 GT 보상 훈련 정책을 능가

#### 보상 모델 품질 (Pearson 상관계수)

| 데이터셋 | P-IQL | VOTP |
|----------|-------|------|
| hopper-medium-replay | 0.04 | 0.59 |
| walker2d-medium-replay | 0.84 | 0.94 |
| door-open | 0.57 | 0.93 |
| drawer-open | 0.59 | 0.91 |

#### 의사 레이블 정확도 (Table 13)

| 태스크 | 정확도 |
|--------|--------|
| hopper-medium-expert | 90.3% |
| walker2d-medium-replay | 98.8% |
| door-open | 93.1% |
| drawer-open | 97.4% |
| sweep-into | 67.0% |

### 2.5 한계점

1. **ViFM 편향 전파:** ViFMs에 내재된 편향이 보상 함수 및 정책에 반영될 수 있음
2. **OT 계산 비용 스케일링:** 레이블 수 증가에 따라 OT 계획 계산 비용 증가 (N=500일 때 순차 처리로 60분)
3. **시각 렌더링 비용:** 무레이블 데이터의 시각 세그먼트 렌더링으로 인해 무레이블 데이터셋 크기가 고정됨
4. **sweep-into 의사 레이블 정확도 저하:** 67.0%로 다른 태스크 대비 낮음 (미세한 행동 차이 구별의 어려움)
5. **온라인 RL 미검증:** 현재 오프라인 설정에만 실험 수행

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 시각적 방해 요소에 대한 강인성

논문은 MetaWorld 환경에서 다양한 시각적 방해 요소를 체계적으로 실험하였다 (Table 2):

| 방해 유형 | door-open | drawer-open | 평균 |
|-----------|-----------|-------------|------|
| 기본 (동일 도메인) | 84.0±8.4 | 71.2±11.7 | 77.6 |
| 조명 변화 (위치+방향) | 88.8±3.0 | 74.4±9.2 | **81.6** |
| 조명 변화 (주변+확산) | 79.2±3.0 | 77.6±7.4 | 78.4 |
| 텍스처 변화 | 76.8±12.7 | 72.0±4.4 | 74.4 |
| 비디오 배경 (쉬움) | 79.2±6.9 | 68.0±5.7 | 73.6 |
| 비디오 배경 (어려움) | 80.4±4.1 | 68.8±8.2 | 74.6 |

**핵심 발견:** 조명 방향/위치 변화 조건에서 오히려 성능이 향상됨 (77.6 → 81.6). 이는 ViFM이 포착하는 의미론적 표현이 특정 시각적 노이즈에 대해 **정규화 효과**를 가질 수 있음을 시사한다.

### 3.2 ViFM의 일반화 능력: 핵심 메커니즘

ViFMs(특히 S3D)는 **HowTo100M** 데이터셋으로 사전학습되어 있으며:

- **다양한 배우(Actors):** 특정 로봇/에이전트에 과적합되지 않은 actor-agnostic 표현
- **다양한 시점(Viewpoints):** 카메라 각도 변화에 강인
- **다양한 조명 조건:** 실세계 비디오의 다양한 광원 환경 학습
- **다양한 배경:** 배경 변화에 독립적인 행동 표현

이러한 특성이 **새로운 시각적 도메인으로의 전이(Transfer)** 능력을 부여한다.

### 3.3 ViFM 선택에 따른 일반화 성능 비교 (Figure 3)

```
성능 우위 순서 (대체로):
ViFMs (S3D, VideoCLIP, InternVideo) > IFMs (R3M, CLIP)
```

| 인코더 | 파라미터 수 | 특성 |
|--------|-------------|------|
| R3M | - | 이미지 기반, 로봇 조작 특화 |
| CLIP | - | 이미지-텍스트 정렬 |
| **S3D** | **31M** | **시공간 특징, 경량** |
| VideoCLIP | 208M | 비디오-텍스트 정렬 |
| InternVideo | 478M | 대규모 비디오 이해 |

S3D가 파라미터 대비 가장 우수한 성능을 보이며, 더 고급 ViFM 사용 시 추가 성능 향상 가능성이 열려있다.

### 3.4 인간 교사 피드백에 대한 일반화

스크립트 교사 vs. 실제 인간 교사 비교 (Table 12):

| 데이터셋 | 스크립트 | 인간 |
|----------|----------|------|
| hopper-medium-expert | 105.7 | 109.3 |
| walker2d-medium-expert | 108.1 | 90.8 |
| door-open | 84.0 | 85.6 |
| drawer-open | 71.2 | 70.4 |
| **평균** | **78.6** | **75.5** |

일부 성능 하락이 있으나, 대부분의 태스크에서 안정적인 성능 유지. 이는 VOTP가 **노이즈가 있는 실제 인간 레이블에도 어느 정도 강인**함을 보여준다.

### 3.5 실제 로봇 환경으로의 일반화

시뮬레이션 → 실제 로봇 (7-DoF Rethink Sawyer) 전이:

| 방법 | Lift Banana | Drawer Open |
|------|-------------|-------------|
| BC | 20.0% | 40.0% |
| P-IQL | 50.0% | 50.0% |
| **VOTP** | **80.0%** | **70.0%** |

VOTP는 시뮬레이션에서 학습된 원리를 실세계 로봇 조작으로 효과적으로 전이한다.

### 3.6 일반화 성능 향상을 위한 추가 가능성

논문이 제시하는 미래 방향:

1. **더 강력한 ViFM 활용:** InternVideo2, Video-LLaMA 등 최신 대형 비디오 모델
2. **자기지도 사전학습 통합:** DINOv2(Oquab et al., 2023), IJEPA(Assran et al., 2023) 등
3. **3D 장면 이해 통합:** 깊이 정보를 활용한 더 강인한 표현
4. **비용 함수 다양화:** 코사인 거리 외 더 정교한 거리 함수 탐색

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 반지도 PbRL 연구 흐름

| 논문 | 연도 | 방법 | 한계 | VOTP 대비 |
|------|------|------|------|-----------|
| **SURF** (Park et al.) | 2022 | 학습된 보상 모델로 의사 레이블 생성 | 확증 편향, 불안정 | VOTP가 D4RL에서 우위 |
| **IPL** (Hejna & Sadigh) | 2023 | 보상 함수 없이 역선호 학습 | 저데이터 성능 취약 | VOTP가 전 도메인 우위 |
| **CPL** (Hejna et al.) | 2024 | 대조적 선호 학습 (RL 없이) | 저데이터에서 성능 낮음 | VOTP가 전 도메인 우위 |
| **LiRE** (Choi et al.) | 2024 | 리스트형 피드백으로 2차 정보 활용 | 레이블 효율 한계 | VOTP가 hop-m-r 등에서 우위 |
| **FTB** (Zhang et al.) | 2024 | 확산 모델로 더 선호된 궤적 생성 | 훈련 시간 2일 | VOTP가 빠르고 동등 이상 |
| **APPO** (Kang & Oh) | 2025 | 정책-동역학 2인 게임 프레임 | D4RL에서 성능 저하 | VOTP가 전 도메인 우위 |
| **SEQUEL** (Marta et al.) | 2024 | 잠재 보간으로 쿼리 합성 | 합성 데이터 의존 | 직접 비교 없음 |

### 4.2 Foundation Model 기반 보상 학습

| 논문 | 연도 | 방법 | 한계 | VOTP와의 관계 |
|------|------|------|------|--------------|
| **VLM-RL** (Rocamonde et al.) | 2024 | CLIP으로 제로샷 보상 | 노이즈, 불일관성 | VOTP는 비디오 모달리티 활용 |
| **RL-VLM-F** (Wang et al.) | 2024 | VLM 피드백으로 보상 | 프롬프트 엔지니어링 의존 | VOTP는 프롬프트 불필요 |
| **PEARL** (Liu et al.) | 2024 | OT로 도메인 간 선호 전이 | 동일 상태/행동 공간 필요 | VOTP는 동일 도메인 내 작동, 시각 입력 지원 |
| **RoboClip** (Sontakke et al.) | 2024 | 단일 시연으로 정책 학습 | 단일 예시 의존 | VOTP는 다수 레이블 활용 |

### 4.3 최적 운송 × 강화학습

| 논문 | 연도 | 방법 | 적용 영역 |
|------|------|------|----------|
| **OTR** (Fickinger et al.) | 2022 | OT로 교차 도메인 모방 학습 | 모방 학습 |
| **OTIL** (Luo et al.) | 2023 | 오프라인 모방 학습에 OT 적용 | 오프라인 IL |
| **OTMatch** (Tan et al.) | 2024 | 반지도 분류에 OT 적용 | 분류 |
| **PEARL** (Liu et al.) | 2024 | OT로 선호 전이 | PbRL (교차 도메인) |
| **VOTP** (Luu et al.) | 2026 | OT + ViFM으로 의사 선호 레이블 생성 | 오프라인 PbRL (동일 도메인) |

### 4.4 연구 트렌드 시사점

```
2020-2021: 온라인 PbRL 기초 (PEBBLE, B-Pref)
2022-2023: 피드백 효율화 시도 (SURF, IPL, LiRE)
2024:      Foundation Model 통합 시작 (VLM 보상, CPL, FTB)
2025-2026: ViFM + OT 결합 고도화 (VOTP)
```

---

## 5. 향후 연구에 미치는 영향과 고려사항

### 5.1 향후 연구에 미치는 영향

#### 5.1.1 PbRL 패러다임 전환

VOTP는 **"레이블이 곧 선호"** 라는 기존 패러다임을 **"소수 레이블 + 다량 무레이블 + 풍부한 시각 표현"** 패러다임으로 전환시킨다. 이는 실세계 PbRL 적용의 문턱을 획기적으로 낮출 수 있다.

#### 5.1.2 ViFM의 RL 활용 확산

Video Foundation Model이 단순한 특징 추출기를 넘어 **의사 레이블러(pseudo-labeler)** 로 기능할 수 있음을 실증함으로써, 다양한 RL 분야에서 ViFM 활용 연구를 촉진할 것으로 예상된다.

#### 5.1.3 OT의 RL 응용 다양화

OT가 모방 학습 외에 선호 기반 보상 학습에도 효과적임을 보임으로써, OT의 RL 응용 범위가 확장될 것이다.

#### 5.1.4 실제 로봇 적용 가능성 향상

소수 레이블(5-10개)만으로 실제 로봇 태스크에서 높은 성공률을 달성함으로써, **실용적 인간-로봇 협업 시스템** 구현에 중요한 기반을 제공한다.

### 5.2 향후 연구 시 고려사항

#### 5.2.1 방법론적 고려사항

**① 더 강력한 ViFM 통합**

```python
# 현재: S3D (31M 파라미터)
# 향후 탐색: InternVideo2, Video-LLaMA, Gemini 비디오 인코더
# 고려사항: 파라미터 효율 vs. 표현 품질 트레이드오프
```

**② OT 계산 효율화**

논문이 언급한 계층적/근사 OT(Halmos et al., 2025) 도입 시 대규모 데이터셋에서의 확장성 확보 가능:

$$\text{복잡도: } O(N^2 M) \rightarrow O(N \log N \cdot M) \text{ (근사 OT)}$$

**③ 비균일 사전 분포 탐색**

현재의 균일 사전 분포 가정을 넘어 데이터 품질이나 태스크 관련성에 따른 비균일 가중치 적용:

$$p_i \propto \text{quality}(\sigma_i), \quad q_{i'} \propto \text{relevance}(\bar{\sigma}_{i'})$$

**④ 생성 정책과의 결합**

논문 Appendix D에서 언급된 대로, 확산 정책(Diffusion Policy)이나 Flow Matching 기반 정책과의 결합:

$$\pi_\theta \sim \text{FlowMatching}(\hat{r}_\psi^{\text{VOTP}})$$

이는 복잡한 다중 모달 행동 분포 학습에 유리하다 (Xia et al., 2025; Miao et al., 2025).

#### 5.2.2 일반화 관련 고려사항

**① 교차 도메인 전이**

현재 VOTP는 동일 도메인 내 pseudo-labeling에 집중하나, PEARL처럼 교차 도메인 선호 전이를 ViFM과 결합하면 더 강력한 일반화 가능:

$$\mu^* = \arg\min_{\mu \in \mathcal{M}} \sum_{i,i'} c_{\text{cross}}(\sigma_i^{\mathcal{S}}, \bar{\sigma}_{i'}^{\mathcal{T}})\mu_{ii'}$$

**② 3D 표현 통합**

깊이 카메라나 포인트 클라우드 정보를 ViFM 표현에 통합하여 기하학적 이해 강화:

$$\mathbf{z} = f_\phi(\mathbf{o}_{1:H}^{\text{RGB}}) \oplus g_\phi(\mathbf{p}_{1:H}^{\text{3D}})$$

**③ 자기지도 표현 학습과의 결합**

VOTP 훈련 중 ViFM 인코더를 미세조정(fine-tuning)하거나, DINOv2/IJEPA와 같은 자기지도 학습으로 도메인 특화 표현 개선.

#### 5.2.3 안전성 및 신뢰성 고려사항

**① ViFM 편향 감사(Auditing)**

ViFMs이 특정 인구통계적 그룹이나 물체에 편향된 특징을 학습할 경우, 로봇 정책에 부적절한 행동 유발 가능:

```
권고사항: 배포 전 다양한 인구/환경에서 편향 테스트 필수
```

**② 의사 레이블 신뢰도 보정**

의사 레이블의 불확실성을 명시적으로 모델링하여 보상 학습에 반영:

$$\hat{r}_\psi \leftarrow \text{학습 시 의사 레이블 불확실성 } u_{i'j'} \text{ 가중치 적용}$$

**③ 안전 임계 태스크에서의 검증**

VOTP를 의료 로봇, 자율주행 등 고위험 분야에 적용하기 전 체계적인 안전성 평가 프레임워크 필요.

#### 5.2.4 연구 방향 로드맵

```
단기 (1-2년):
├── 더 강력한 ViFM 통합 (InternVideo2, Video-LLaMA)
├── 온라인 PbRL으로 확장
├── 활성 쿼리 선택과 결합
└── 계층적 OT로 계산 효율화

중기 (2-4년):
├── 생성 정책(Diffusion, Flow)과 결합
├── 교차 도메인 선호 전이
├── VLA(Vision-Language-Action) 모델 미세조정에 적용
└── 3D 장면 이해 통합

장기 (4년 이상):
├── 헬스케어, 교육 등 비로봇 도메인 확장
├── 연속 학습(Continual Learning)과 결합
└── 범용 RLHF 플랫폼으로 발전
```

---

## 결론

VOTP는 **극소수의 인간 피드백(10개)**으로 강력한 보상 함수를 학습하는 혁신적인 프레임워크로, Video Foundation Model의 풍부한 시각 표현과 Optimal Transport의 수학적 엄밀성을 결합하여 기존 방법론의 한계를 극복하였다. 특히 ViFM의 actor-agnostic 특성이 시각적 방해 요소와 실제 로봇 환경으로의 일반화를 가능하게 하는 핵심 메커니즘임이 실험적으로 검증되었다. 향후 더 강력한 ViFM, 생성 기반 정책, 그리고 3D 표현과의 결합을 통해 실용적 인간-로봇 협업 시스템 구현에 크게 기여할 것으로 기대된다.
