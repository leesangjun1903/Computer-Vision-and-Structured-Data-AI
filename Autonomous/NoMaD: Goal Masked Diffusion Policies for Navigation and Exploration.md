# NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

NoMaD는 **단일 통합 확산 정책(unified diffusion policy)**을 통해 목표 지향 내비게이션(goal-conditioned navigation)과 비지향적 탐색(undirected exploration)을 동시에 수행할 수 있는 최초의 로봇 항법 모델입니다. 기존 연구들이 두 행동을 별도 모델로 처리하던 한계를 극복하고, 하나의 모델로 두 가지 행동을 모두 학습함으로써 더 효율적이고 일반화된 성능을 달성할 수 있다고 주장합니다.

### 주요 기여

| 기여 항목 | 설명 |
|-----------|------|
| **Goal Masking** | 이진 마스크 $m$을 이용해 목표 이미지 조건부/비조건부 추론을 단일 모델에서 유연하게 수행 |
| **Action Diffusion Policy** | 복잡한 다봉(multimodal) 행동 분포를 직접 모델링하는 확산 기반 정책 |
| **통합 아키텍처** | Transformer 인코더 + 확산 모델 디코더의 결합으로 표현력과 효율성 동시 달성 |
| **실세계 배포** | 목표 조건부 확산 정책을 실제 로봇에 성공적으로 배포한 최초 사례 |
| **효율성** | 최신 기법(Subgoal Diffusion, 335M 파라미터) 대비 **15배 적은 파라미터(19M)**로 25% 이상의 성능 향상 |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

로봇 항법에는 두 가지 핵심 능력이 필요합니다:

1. **목표 지향 내비게이션**: 사용자가 지정한 목표(이미지)로 이동
2. **탐색(Exploration)**: 목표를 알 수 없는 미지 환경에서 자율적으로 탐색

기존 접근법(ViNT 등)은 이 두 역할을 **별도 모델**로 처리했으며, 특히 탐색을 위해 대형 이미지 생성 모델(300M 파라미터의 서브골 제안 모델)을 추가로 활용했습니다. 이는 다음 문제를 야기했습니다:

- 시스템 복잡도 증가
- 엣지 컴퓨팅 환경에서의 실시간 실행 불가
- 다봉 행동 분포의 표현 한계 (단순 회귀 기반 정책의 경우)

---

### 2-2. 제안하는 방법 (수식 포함)

#### (A) Goal Masking

이진 마스크 $m \in \{0, 1\}$을 도입하여 목표 토큰 $\phi(\mathbf{o}_t, o_g)$의 어텐션 참여 여부를 제어합니다.

$$c_t = f\bigl(\psi(o_i),\; \phi(\mathbf{o}_t, o_g),\; m\bigr)$$

- $m = 0$: 목표 토큰이 어텐션에 참여 → **목표 지향 내비게이션**
- $m = 1$: 목표 토큰을 마스킹 → **비지향적 탐색**

훈련 시 $m$은 베르누이 분포에서 샘플링됩니다:

$$m \sim \text{Bernoulli}(p_m), \quad p_m = 0.5$$

#### (B) Diffusion Policy (확산 정책)

관측 컨텍스트 $c_t$에 조건부인 행동 분포 $p(\mathbf{a}_t | c_t)$를 확산 모델로 모델링합니다.

**반복적 역확산(Iterative Denoising) 과정:**

$$\mathbf{a}_t^{k-1} = \alpha \cdot \left(\mathbf{a}_t^k - \gamma \epsilon_\theta(c_t, \mathbf{a}_t^k, k) + \mathcal{N}(0, \sigma^2 I)\right) \tag{1}$$

여기서:
- $k$: 현재 디노이징 스텝
- $\epsilon_\theta$: 파라미터 $\theta$로 매개변수화된 노이즈 예측 네트워크
- $\alpha, \gamma, \sigma$: 노이즈 스케줄의 함수

가우시안 노이즈 $\mathbf{a}_t^K$에서 시작하여 $K=10$ 스텝의 디노이징을 통해 최종 행동 시퀀스 $\mathbf{a}_t^0$를 생성합니다.

#### (C) 학습 손실 함수

$$\mathcal{L}_{\text{NoMaD}}(\phi, \psi, f, \theta, f_d) = \underbrace{\text{MSE}(\epsilon^k,\; \epsilon_\theta(c_t, \mathbf{a}_t^0 + \epsilon^k, k))}_{\text{Diffusion Loss}} + \lambda \cdot \underbrace{\text{MSE}(d(\mathbf{o}_t, o_g),\; f_d(c_t))}_{\text{Temporal Distance Loss}} \tag{2}$$

- $\psi, \phi$: 관측 및 목표 이미지의 시각 인코더
- $f$: Transformer 레이어
- $\theta$: 확산 과정의 파라미터
- $f_d$: 시간적 거리 예측기
- $\lambda = 10^{-4}$: 시간 거리 손실의 가중치 하이퍼파라미터

---

### 2-3. 모델 구조

```
RGB 관측 (과거 5 timestep, 96×96×3)
        ↓
  EfficientNet-B0 인코더 ψ (관측 토큰)
        +
  EfficientNet-B0 인코더 ϕ (목표 융합 토큰)
        ↓
  Goal Masking (m=0 or 1)
        ↓
  Transformer Decoder (4 layers, 4 heads, 5M params)
        ↓
  컨텍스트 벡터 c_t (Average Pooling → 256-D)
        ↓
  ┌─────────────────────────┐
  │  1D Conditional U-Net   │  ← 노이즈 예측 네트워크 ε_θ
  │  (15 conv layers)       │     K=10 denoising steps
  └─────────────────────────┘
        ↓
  8개 미래 행동 시퀀스 a_t
```

**주요 하이퍼파라미터:**

| 항목 | 값 |
|------|----|
| 총 파라미터 수 | 19M |
| 시각 인코더 | EfficientNet-B0 |
| 임베딩 차원 | 256-D |
| Transformer | 4 layers, 4 heads |
| 노이즈 스케줄러 | Square Cosine (Nichol & Dhariwal, 2021) |
| 디노이징 스텝 | K = 10 |
| 옵티마이저 | AdamW, lr= $10^{-4}$ |
| 배치 크기 | 256 |
| 훈련 에폭 | 30 |
| 목표 마스킹 확률 | $p_m = 0.5$ |

---

### 2-4. 성능 향상

**Table I: 탐색 및 내비게이션 성능 비교**

| 방법 | 파라미터 | 탐색 성공률 | 충돌 횟수 | 내비게이션 성공률 |
|------|---------|-----------|---------|----------------|
| Masked ViNT | 15M | 50% | 1.0 | 30% |
| VIB | 6M | 30% | 4.0 | 15% |
| Autoregressive | 19M | 90% | 2.0 | 60% |
| Random Subgoals | 30M | 70% | 2.7 | 90% |
| Subgoal Diffusion | **335M** | 77% | 1.7 | 90% |
| **NoMaD (제안)** | **19M** | **98%** | **0.2** | **90%** |

**Table II: 통합 vs. 전용 정책 비교**

| 방법 | 파라미터 | 비지향 탐색 | 목표 조건부 |
|------|---------|-----------|-----------|
| Diffusion Policy | 15M | 98% | ✗ |
| ViNT Policy | 16M | ✗ | 92% |
| **NoMaD** | **19M** | **98%** | **92%** |

**핵심 성능 향상 포인트:**
- 최신 기법(Subgoal Diffusion) 대비 **탐색 성공률 25% 이상 향상**
- **충돌 횟수 대폭 감소**: 1.7 → 0.2
- 모델 크기 **15배 감소** (335M → 19M)
- 전용 정책 모델(Diffusion Policy, ViNT) 수준의 성능을 **단일 모델**로 달성

---

### 2-5. 한계점

논문이 명시적으로 인정한 한계:

1. **목표 지정 방식의 제한**: 목표를 이미지로만 지정 가능 → 언어 명령, GPS 좌표 등의 다양한 모달리티 미지원
2. **탐색 전략의 단순성**: 프론티어 기반 탐색(frontier-based exploration)만 사용 → 의미론적(semantic) 정보나 사전 지식을 활용한 지능적 탐색 미구현
3. **ViT 인코더 호환 문제**: Vision Transformer 기반 인코더는 확산 모델과의 엔드-투-엔드 학습 시 최적화 어려움 발생 (성공률 32%)
4. **실내외 복잡 환경의 극단적 케이스**: 가장 어려운 환경(6번째 환경)에서는 실패 사례 존재

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

### 3-1. 일반화를 가능하게 하는 핵심 요소

#### (A) 대규모 이종 데이터셋 학습

NoMaD는 **GNM(General Navigation Model)** 데이터셋과 **SACSoN** 데이터셋의 조합으로 훈련됩니다:
- 100시간 이상의 실세계 궤적 데이터
- 보행자 밀집 환경 포함
- 다양한 로봇 플랫폼에서 수집된 이종(heterogeneous) 데이터

이러한 다양한 데이터 기반 학습은 특정 환경이나 플랫폼에 과적합되지 않도록 하여 미지 환경(unseen environments)에서의 일반화를 촉진합니다.

#### (B) 공유 표현 학습의 시너지 효과

Table II에서 확인할 수 있듯이, 목표 지향과 비지향 탐색을 **동시에** 학습한 NoMaD는 두 전용 모델의 성능을 각각 복제합니다. 이는 두 행동이 **공유된 시각적 어포던스(visual affordances)**를 학습하기 때문으로, 저자들은 다음과 같이 해석합니다:

> *"Training for these two behaviors involves learning shared representations and affordances, and a single policy can indeed excel at both task-agnostic and task-oriented behaviors simultaneously."*

목표가 없는 탐색 데이터는 환경의 일반적인 내비게이션 패턴(장애물 회피, 복도 따라가기 등)을 학습하게 하고, 목표 조건부 데이터는 특정 목적지로의 방향성을 학습하게 합니다. 두 학습 신호가 서로 **정규화(regularization)** 역할을 하여 과적합을 방지합니다.

#### (C) 다봉 행동 분포 모델링

일반화의 핵심 병목 중 하나는 교차로, 분기점 등 **모호한 상황에서의 의사결정**입니다. 단순 회귀 기반 정책(Masked ViNT)은 평균적 행동만 예측하는 반면, 확산 정책은 좌회전/우회전 등의 **다봉 분포**를 명시적으로 모델링합니다:

$$p(\mathbf{a}_t | c_t) \approx \text{Multimodal distribution via denoising}$$

Figure 3과 5에서 시각화된 바와 같이, NoMaD는 목표 없이는 양방향의 가능한 행동을 동시에 표현하고, 목표가 주어지면 해당 방향으로 분포가 집중됩니다. 이는 특히 미지 환경에서의 일반화에 결정적입니다.

#### (D) 토폴로지 맵과의 결합

단기 정책의 일반화를 장기 목표 달성으로 확장하기 위해 **ViKiNG** 방식의 토폴로지 그래프를 활용합니다:
- 노드: 로봇의 시각적 관측
- 엣지: 정책의 목표 조건부 거리 예측으로 결정된 이동 가능 경로

이 구조는 미지 환경에서 온라인으로 구축되며, 단기 정책이 강건할수록 장기 탐색의 일반화 성능도 향상됩니다.

### 3-2. 일반화 성능 향상의 잠재적 방향

논문에서 제시된 한계와 실험 결과를 바탕으로 다음의 일반화 향상 가능성을 도출할 수 있습니다:

**① 멀티모달 목표 조건화**
현재 이미지 기반 목표만 지원하지만, **언어 임베딩(CLIP 등)**이나 **GPS 좌표**를 목표 토큰으로 통합하면 더 광범위한 환경에서의 일반화가 가능합니다. 목표 마스킹 메커니즘은 이러한 확장에 유연하게 대응할 수 있는 구조를 이미 갖추고 있습니다.

**② 의미론적 탐색 전략**
현재의 프론티어 기반 탐색을 의미론적 단서(semantic cues)로 보강하면 미지 환경에서의 목표 발견 효율을 높일 수 있습니다.

**③ 온라인 적응(Online Adaptation)**
사전 학습된 NoMaD를 새로운 환경에서 소수의 샘플로 미세 조정하는 메타 학습 접근법을 결합하면 도메인 이동(domain shift)에 강건한 일반화가 가능합니다.

**④ 더 큰 이종 데이터셋 활용**
GNM/SACSoN 외에 항공 드론, 자율주행 차량 데이터 등 더 다양한 로봇 플랫폼 데이터를 포함하면 플랫폼 간 일반화(cross-platform generalization)가 향상될 것입니다.

---

## 4. 최신 연구 비교 분석 (2020년 이후)

| 연구 | 연도 | 핵심 방법 | 일반화 전략 | NoMaD와의 관계 |
|------|------|-----------|-----------|--------------|
| **DDPM** (Ho et al.) | 2020 | 기본 확산 모델 | - | NoMaD의 확산 디노이징 기반 |
| **VIB** (Shah et al.) | 2021 | 잠재 목표 모델, VIB | 실세계 경험 직접 학습 | NoMaD가 탐색에서 25%+ 능가 |
| **ViNG** (Shah et al.) | 2021 | 시각 목표 기반 오픈월드 | 토폴로지 메모리 | NoMaD의 고수준 계획 기반 |
| **Diffusion Policy** (Chi et al.) | 2023 | 행동 확산 정책 | 시연 데이터 기반 | NoMaD의 확산 디코더 기반 |
| **ViNT** (Shah et al.) | 2023 | 파운데이션 내비게이션 모델 | 다중 로봇 대규모 데이터 | NoMaD의 직접 베이스라인 |
| **ViKiNG** (Shah & Levine) | 2022 | 킬로미터급 비전 내비게이션 | GPS 힌트 결합 | NoMaD의 고수준 계획 구조 기반 |
| **RT-1** (Brohan et al.) | 2022 | 로봇 트랜스포머 | 대규모 실세계 데이터 | 유사한 스케일 학습 방향 |
| **Diffuser** (Janner et al.) | 2022 | 계획을 위한 확산 | 오프라인 RL 데이터 | NoMaD는 행동 공간만 확산 모델링 |

**NoMaD의 차별점:**
- ViNT: 서브골 이미지 생성(300M 이미지 확산) → NoMaD: 행동 직접 확산(19M)으로 **15배 효율적**
- Diffusion Policy: 비지향 탐색만 가능 → NoMaD: **통합 목표 마스킹**으로 두 모드 동시 지원
- VIB: 단봉 분포 가정 → NoMaD: **다봉 분포 명시적 모델링**

---

## 5. 앞으로의 연구에 미치는 영향 및 고려사항

### 5-1. 연구에 미치는 영향

**① 통합 정책 패러다임의 확립**
NoMaD는 탐색과 내비게이션이라는 서로 다른 행동을 단일 확산 모델로 통합할 수 있음을 증명했습니다. 이는 향후 **멀티태스크 로봇 정책 연구**에서 별도 모듈 대신 통합 모델 설계를 지향하는 흐름을 강화할 것입니다.

**② 확산 모델의 로봇공학 적용 확산**
NoMaD는 확산 정책을 실제 실외 로봇에 실시간으로 배포한 첫 성공 사례로, 이후 **드론 항법, 조작(manipulation), 사회적 내비게이션** 등 다양한 로봇 태스크에 확산 정책을 적용하는 연구를 촉진할 것입니다.

**③ 효율적 멀티모달 행동 모델링**
이미지 생성 없이 행동 공간에서만 확산을 수행하는 방식은 계산 비용을 크게 절감하면서도 표현력을 유지합니다. 이는 **엣지 컴퓨팅** 환경의 로봇 학습 연구에 중요한 방향을 제시합니다.

**④ Goal Masking의 범용성**
베르누이 마스킹이라는 단순한 기법이 놀라운 효과를 보인 것은, 다양한 조건부 생성 모델에서 **유연한 컨디셔닝 메커니즘**으로 활용될 수 있음을 시사합니다.

### 5-2. 앞으로 연구 시 고려할 점

**① 목표 표현의 다양화**
이미지 외에도 언어(예: CLIP, LLM 기반), GPS 좌표, 포인트 클라우드 등의 목표 표현을 통합하는 연구가 필요합니다. 이를 위해 **크로스 모달 어텐션** 메커니즘과 Goal Masking의 결합을 탐색할 수 있습니다.

**② 의미론적 고수준 계획자**
현재의 프론티어 기반 탐색을 LLM이나 VLM(Vision-Language Model)과 결합하여 의미 있는 탐색 전략을 수립하는 방향이 유망합니다 (예: "주방을 먼저 탐색하라").

**③ 안전성(Safety) 보장**
확산 정책이 다봉 분포를 표현할 수 있지만, 특정 행동의 안전성을 보장하기 위한 **제약 기반 확산(constrained diffusion)** 또는 **안전 필터링** 메커니즘의 통합이 실용화에 중요합니다.

**④ 연속적 학습(Continual Learning)**
새로운 환경에 노출될수록 토폴로지 맵이 업데이트되지만, 정책 자체가 온라인으로 적응하지 못합니다. **메타 학습이나 점진적 미세 조정** 방법론과의 결합이 장기 배포 시 일반화 성능 유지에 중요합니다.

**⑤ 불확실성 정량화**
확산 모델은 다봉 분포를 표현하지만 예측의 신뢰도를 명시적으로 제공하지 않습니다. **베이지안 확산 모델** 또는 **앙상블 기반 불확실성 추정**을 결합하면 더 안전한 탐색이 가능합니다.

**⑥ 시뮬레이션-실세계 전이(Sim2Real)**
논문은 시뮬레이션 학습 정책의 실세계 전이 한계를 언급합니다. NoMaD의 구조에 **도메인 무작위화(domain randomization)**나 **적응적 정규화** 기법을 결합하면 sim2real 갭을 줄일 수 있습니다.

---

## 참고자료

1. **Sridhar, A., Shah, D., Glossop, C., & Levine, S.** (2023). *NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration.* arXiv:2310.07896v1 [cs.RO]. — **본 논문 (제공된 PDF)**

2. **Shah, D., Sridhar, A., Dashora, N., Stachowicz, K., Black, K., Hirose, N., & Levine, S.** (2023). *ViNT: A Foundation Model for Visual Navigation.* 7th Annual Conference on Robot Learning (CoRL).

3. **Chi, C., Feng, S., Du, Y., Xu, Z., Cousineau, E., Burchfiel, B., & Song, S.** (2023). *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* Robotics: Science and Systems (RSS).

4. **Ho, J., Jain, A., & Abbeel, P.** (2020). *Denoising Diffusion Probabilistic Models.* Neural Information Processing Systems (NeurIPS).

5. **Shah, D., Eysenbach, B., Rhinehart, N., & Levine, S.** (2021). *Rapid Exploration for Open-World Navigation with Latent Goal Models.* Conference on Robot Learning (CoRL).

6. **Shah, D., & Levine, S.** (2022). *ViKiNG: Vision-Based Kilometer-Scale Navigation with Geographic Hints.* Robotics: Science and Systems (RSS).

7. **Nichol, A. Q., & Dhariwal, P.** (2021). *Improved Denoising Diffusion Probabilistic Models.* International Conference on Machine Learning (ICML).

8. **Janner, M., Du, Y., Tenenbaum, J., & Levine, S.** (2022). *Planning with Diffusion for Flexible Behavior Synthesis.* International Conference on Machine Learning (ICML).

9. **Shah, D., Sridhar, A., Bhorkar, A., Hirose, N., & Levine, S.** (2023). *GNM: A General Navigation Model to Drive Any Robot.* International Conference on Robotics and Automation (ICRA).

10. **Tan, M., & Le, Q. V.** (2020). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* (Referenced as EfficientNet-B0 encoder in NoMaD)
