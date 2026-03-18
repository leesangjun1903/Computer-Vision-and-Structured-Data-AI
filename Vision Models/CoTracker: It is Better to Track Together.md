# CoTracker: It is Better to Track Together

---

## 1. 핵심 주장 및 주요 기여 요약

**핵심 주장:** 비디오에서 2D 포인트를 추적할 때, 각 포인트를 **독립적으로** 추적하는 기존 방식과 달리 **여러 포인트를 동시에 공동(jointly) 추적**하면 포인트 간의 통계적 의존성(예: 같은 객체 위의 점들)을 활용하여 추적 정확도와 강건성을 크게 향상시킬 수 있다.

**주요 기여 (4가지):**

1. **Joint point tracking:** 트랜스포머의 어텐션 메커니즘을 통해 추적 포인트 간 정보를 공유하는 공동 추적 개념 도입
2. **Support points:** 사용자가 요청하지 않은 추가적인 보조 포인트를 함께 추적하여 문맥(context) 정보를 확장
3. **Proxy tokens:** 메모리 복잡도를 $O(N^2)$에서 $O(NK + K^2)$로 감소시켜 단일 GPU에서 최대 70k 포인트를 동시에 추적 가능
4. **Unrolled training:** 슬라이딩 윈도우를 순환 신경망처럼 펼쳐(unroll) 학습하여 장기간 추적 및 장기 가림(occlusion) 처리 성능 향상

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

비디오에서 임의의 2D 포인트들을 장기간에 걸쳐 추적하는 **Tracking Any Point (TAP)** 문제를 다룬다. 기존 방법의 두 가지 핵심 한계:

- **Optical flow:** 모든 점의 움직임을 동시에 추정하지만 순간적(infinitesimal) 이동만 예측하며, 시간 누적 시 드리프트(drift) 발생
- **독립적 포인트 트래킹 (PIPs, TAPIR, PIPs++):** 각 포인트를 통계적으로 독립적인 것으로 취급하여, 같은 객체에 속한 점들 간의 상관관계를 활용하지 못함. 특히 가림(occlusion) 상황에서 추적 실패가 빈번

### 2.2 제안하는 방법

#### (a) 트랜스포머 기반 공동 추적 구조

비디오 $V = (I_t)_{t=1}^{T}$에서 $N$개의 2D 포인트 트랙 $P^i_t = (x^i_t, y^i_t) \in \mathbb{R}^2$와 가시성 플래그 $v^i_t \in \{0, 1\}$를 예측한다.

**이미지 특징 추출:** CNN을 사용하여 각 프레임에서 $d$차원 밀집 특징을 추출한다:

$$\phi(I_t) \in \mathbb{R}^{d \times \frac{H}{k} \times \frac{W}{k}}, \quad k=4$$

다중 스케일 특징 $\phi_s(I_t)$를 $S=4$ 스케일로 average pooling을 통해 생성한다.

**공간 상관 특징 (Spatial Correlation Features):** RAFT에서 영감을 받아, 트랙 특징 $Q^i_t$와 이미지 특징 간의 내적으로 상관 특징을 계산한다:

$$[C^i_t]_{s\delta} = \langle Q^i_t,\; \phi_s(I_t)[\hat{P}^i_t / ks + \delta] \rangle$$

여기서 $s = 1, \ldots, S$는 특징 스케일, $\delta \in \mathbb{Z}^2, \|\delta\|_\infty \leq \Delta$는 오프셋이다. $C^i_t$의 차원은 $(2\Delta+1)^2 S = 196$ ($S=4, \Delta=3$).

**입력 토큰 구성:** 위치, 가시성, 외형, 상관관계를 결합한 토큰을 구성한다:

$$G^i_t = \left(\hat{P}^i_t - \hat{P}^i_1,\; \hat{v}^i_t,\; Q^i_t,\; C^i_t,\; \eta(\hat{P}^i_t - \hat{P}^i_1)\right) + \eta'(\hat{P}^i_1) + \eta'(t)$$

여기서 $\eta$는 사인 위치 인코딩, $\eta'$은 시작 위치와 시간에 대한 인코딩이다.

**출력 토큰:** $O^i_t = (\Delta\hat{P}^i_t, \Delta Q^i_t)$로, 위치와 외형의 갱신량을 포함한다.

#### (b) 반복적 트랜스포머 적용 (Iterated Transformer)

트랜스포머 $\Psi$를 $M$번 반복 적용하여 추적 추정값을 점진적으로 개선한다:

$$O(\Delta\hat{P}, \Delta Q) = \Psi\left(G\left(\hat{P}^{(m)}, \hat{v}^{(0)}, Q^{(m)}\right)\right)$$

$$\hat{P}^{(m+1)} = \hat{P}^{(m)} + \Delta\hat{P}, \quad Q^{(m+1)} = Q^{(m)} + \Delta Q$$

가시성은 마지막 반복 후 한 번만 예측한다:

$$\hat{v}^{(M)} = \sigma(W Q^{(M)})$$

여기서 $\sigma$는 시그모이드 활성화 함수이다.

#### (c) 프록시 토큰을 통한 확장성

시간과 트랙 차원으로 어텐션을 분리(factorize)하여 복잡도를 $O(N^2T^2)$에서 $O(N^2 + T^2)$로 줄인다. 여기에 $K$개의 **프록시 트랙**(학습 가능한 고정 토큰, $K \ll N$)을 도입하여:

- 시간 어텐션: 정규 트랙과 프록시 트랙을 동일하게 처리
- 트랙 어텐션: 정규 트랙은 프록시에만 크로스-어텐션, 서로 간에는 어텐션하지 않음

$$\text{복잡도: } O(NK + K^2 + T^2)$$

이를 통해 단일 80GB GPU에서 약 70k 포인트 동시 추적이 가능해진다 (프록시 미사용 시 ~9.4k).

#### (d) 윈도우 추론 및 언롤드 학습

비디오를 길이 $T$의 슬라이딩 윈도우 $J = \lceil 2T'/T - 1 \rceil$개로 분할하며, $T/2$ 프레임씩 중첩한다. 순환 신경망처럼 동작하며, 언롤드 학습을 통해 최적화한다.

**손실 함수:**

$$\mathcal{L}_1(\hat{P}, P) = \sum_{j=1}^{J} \sum_{m=1}^{M} \gamma^{M-m} \|\hat{P}^{(m,j)} - P^{(j)}\|$$

여기서 $\gamma = 0.8$은 초기 반복의 기여를 감쇠시키는 할인 계수이다.

**가시성 손실:**

$$\mathcal{L}_2(\hat{v}, v) = \sum_{j=1}^{J} \text{CE}\left(\hat{v}^{(M,j)}, v^{(j)}\right)$$

### 2.3 모델 구조

| 구성 요소 | 세부 사항 |
|---|---|
| **특징 추출 CNN** | PIPs와 동일; 7×7 conv (stride 2) + 8 residual blocks + 1×1 conv, 128채널, stride 4 |
| **트랜스포머** | 시간 어텐션과 트랙(크로스) 어텐션을 교차 배치. 6개 시간 어텐션 + 6개 크로스-트랙 어텐션 레이어 |
| **프록시 토큰** | 64개 (최적), 학습 가능한 고정 토큰 |
| **슬라이딩 윈도우** | 학습 및 추론 시 $T=8$, $T/2=4$ 프레임 중첩 |
| **반복 업데이트** | 학습 $M=4$, 추론 $M=6$ |
| **학습 데이터** | TAP-Vid-Kubric (24프레임 합성 시퀀스 6,000개) |
| **학습 설정** | 50,000 iterations, 32 A100 GPUs, batch size 32, AdamW, lr $5 \times 10^{-4}$ |

### 2.4 성능 향상

**TAP-Vid-DAVIS (First 프로토콜):**

| 메트릭 | TAPIR | CoTracker | 향상 |
|---|---|---|---|
| AJ | 56.2 | **62.2** | +6.0 |
| $\delta^{\text{vis}}_{\text{avg}}$ | 70.0 | **75.7** | +5.7 |
| OA | 86.5 | **89.3** | +2.8 |

**Dynamic Replica (가림 추적 정확도):**

| 메트릭 | TAPIR | CoTracker |
|---|---|---|
| $\delta^{\text{occ}}_{\text{avg}}$ | 27.2 | **37.6** (+38.2%) |

**PointOdyssey:** 윈도우 크기 8프레임만으로 128프레임 윈도우의 PIPs++보다 높은 Survival rate (55.2 vs 47.0)

**Joint tracking의 효과 (Ablation):**

| 모드 | AJ (DAVIS) | $\delta^{\text{occ}}_{\text{avg}}$ (DR) |
|---|---|---|
| No joint | 55.6 | 28.8 |
| Joint | **62.2** | **37.6** (+30.6%) |

가림 포인트 추적에서의 향상이 가시 포인트보다 훨씬 크다는 것은 공동 추적이 장면의 전체적 움직임을 이해하는 데 효과적임을 보여준다.

### 2.5 한계

- **합성 데이터만으로 학습:** 반사(reflection), 그림자(shadow) 등 복잡한 시각적 장면에서 일반화가 어려운 경우 존재
- **그림자 추적 문제:** 그림자를 객체와 함께 추적하는 경향 (동영상 편집에는 유용할 수 있으나 모션 분석에는 부적절)
- **불연속 비디오:** 다중 샷이 포함된 비디오(예: Kinetics)에서 성능 저하 — 연속 비디오 가정에 의존
- **온라인 인과적 추론의 제약:** TAPIR, OmniMotion 같은 오프라인 방법들은 전체 비디오 접근이 가능하나, CoTracker는 인과적 슬라이딩 윈도우에 의존

---

## 3. 모델의 일반화 성능 향상 가능성

CoTracker의 일반화 성능과 관련된 핵심 내용을 정리하면:

### 3.1 합성→실제 일반화

CoTracker는 **TAP-Vid-Kubric(합성 데이터)에서만 학습**했음에도 불구하고 **실제 비디오 벤치마크인 TAP-Vid-DAVIS에서 SOTA** 성능을 달성했다. 이는 다음 요인에 기인한다:

- **공동 추적 메커니즘:** 포인트 간 의존성 학습이 도메인에 비교적 독립적인 구조적 관계(예: 강체 운동, 공통 움직임 패턴)를 포착
- **데이터 증강:** Color Jitter, Gaussian Blur, occlusion augmentation, random scaling 적용
- **다중 스케일 상관 특징:** $S=4$ 스케일의 특징 매칭이 다양한 해상도와 장면에 대한 강건성 제공

### 3.2 일반화 향상을 위한 방향

1. **실제 데이터 혼합 학습:** 현재 합성 데이터만 사용하므로, 실제 비디오 데이터와의 혼합 학습(mixed training)이나 자기 지도 학습(self-supervised learning)을 통해 반사, 그림자, 투명 물체 등에 대한 일반화 향상 가능
2. **더 다양한 합성 데이터:** 논문에서도 언급하듯이 TAP-Vid-Kubric는 24프레임의 단순한 강체 충돌 시나리오로 제한적. PointOdyssey와 같이 더 현실적이고 장기적인 합성 데이터셋으로 학습 시 성능 향상이 관찰됨
3. **기반 모델(foundation model) 특징 활용:** DINOv2, SAM 등의 사전학습된 시각 특징을 CNN 백본 대신 활용하면 도메인 일반화 성능이 크게 향상될 수 있음 (CoTracker v2에서 실제로 탐구됨)
4. **Support points의 적응적 선택:** 현재 고정된 격자(grid) 패턴을 사용하지만, 장면 내용에 따라 적응적으로 보조 포인트를 선택하면 일반화 성능이 향상될 수 있음
5. **Unrolled training의 확장:** 더 긴 학습 시퀀스와 더 많은 윈도우 언롤링이 장기 추적의 일반화에 기여 — 현재 24프레임 제한이 병목

### 3.3 Cross-domain 일반화 실험 결과

| 학습 데이터 | 평가 데이터 | 성격 | 결과 |
|---|---|---|---|
| Kubric (합성) | DAVIS (실제) | 합성→실제 | AJ 62.2 (SOTA) |
| Kubric (합성) | RGB-Stacking (합성) | 합성→합성 | AJ 71.6 (SOTA) |
| Kubric (합성) | Dynamic Replica (합성) | 합성→합성 | $\delta_{\text{avg}}$ 61.6 (SOTA) |
| PointOdyssey | PointOdyssey | 동일 도메인 | $\delta_{\text{avg}}$ 30.2 (SOTA) |

특히 Kubric에서 학습하고 DAVIS에서 평가한 결과가 매우 우수하여, 공동 추적 메커니즘 자체가 도메인 간 전이 가능한 구조적 사전(prior)을 학습한다는 것을 시사한다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구 영향

1. **포인트 추적 패러다임의 전환:** "독립 추적"에서 "공동 추적"으로의 패러다임 전환을 촉진. 이후 연구들(CoTracker v2, TAPIR v2, BootsTAP 등)이 이 아이디어를 채택
2. **메모리 효율적 설계의 표준화:** 프록시 토큰을 통한 확장 가능한 설계가 대규모 밀집 추적 연구의 기반이 됨
3. **3D 재구성과의 통합:** 저자들이 결론에서 언급하듯, 밀집 장기 추적은 3D 재구성, 동적 장면 이해, 비디오 편집 등 하위 작업의 핵심 입력으로 활용 가능
4. **벤치마크 발전 촉진:** 가림 포인트 추적의 중요성을 부각시켜 PointOdyssey, Dynamic Replica 등에서의 $\delta^{\text{occ}}_{\text{avg}}$ 평가가 표준화

### 4.2 향후 연구 시 고려할 점

- **실시간 추론:** 현재 CoTracker는 10k 포인트 기준 ~27초 소요. 실시간 응용을 위한 경량화 필요
- **양방향 추적:** CoTracker는 인과적(causal) 트래커로, 미래 정보를 활용하지 못함. 양방향 추적 통합 시 성능 향상 가능
- **3D 정보 통합:** 2D 포인트 추적에 깊이 정보나 3D 구조 사전을 결합하여 물리적으로 일관된 추적 가능
- **자기 지도 학습:** 대규모 비라벨 비디오 데이터를 활용한 사전 학습으로 도메인 일반화 향상
- **가림 예측의 개선:** 현재 가시성은 마지막 반복에서만 한 번 예측. 반복적 가시성 추정이나 가림 지속 시간 모델링이 필요

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 특징 | 공동 추적 | 장기 추적 | 가림 처리 | 학습 데이터 |
|---|---|---|---|---|---|---|
| **RAFT** [Teed & Deng, 2020] | 2020 | 반복적 optical flow 갱신, 4D cost volume | ✓ (dense) | ✗ (2-프레임) | ✗ | FlyingChairs/Things |
| **PIPs** [Harley et al., 2022] | 2022 | 슬라이딩 윈도우, 가림 추적 가능 | ✗ | 제한적 | ✓ | FlyingThings++ |
| **TAP-Vid** [Doersch et al., 2022] | 2022 | 벤치마크 + 기본 베이스라인 | ✗ | ✓ | 제한적 | Kubric |
| **TAPIR** [Doersch et al., 2023] | 2023 | TAP-Vid 매칭 + PIPs 정제의 2단계 구조 | ✗ | ✓ | ✓ | Kubric |
| **PIPs++** [Zheng et al., 2023] | 2023 | PIPs 단순화, 128프레임 윈도우 | ✗ | ✓ | ✓ | PointOdyssey |
| **OmniMotion** [Wang et al., 2023] | 2023 | 테스트 시 볼류메트릭 최적화 | ✓ (implicit) | ✓ | ✓ | 테스트 시 최적화 |
| **MFT** [Neoral et al., 2023] | 2023 | 다중 프레임 optical flow 체인 | ✗ | ✓ | 제한적 | Kubric+ |
| **VideoFlow** [Shi et al., 2023] | 2023 | 3-5프레임 양방향 flow | ✓ (dense) | 제한적 | 제한적 | — |
| **CoTracker** [Karaev et al., 2023] | 2023 | **공동 추적, 프록시 토큰, 언롤드 학습** | **✓** | **✓** | **✓** | Kubric |

### 주요 비교 포인트:

| 비교 항목 | CoTracker vs TAPIR | CoTracker vs PIPs++ | CoTracker vs OmniMotion |
|---|---|---|---|
| **추론 방식** | 온라인 (인과적) vs 오프라인 | 온라인 vs 온라인 | 온라인 vs 테스트 시 최적화 |
| **공동 추적** | ✓ vs ✗ | ✓ vs ✗ | ✓ vs ✓ (implicit) |
| **확장성** | 70k 포인트 vs 독립 처리 | 70k vs 독립 처리 | 70k vs 비실용적 (최적화 비용) |
| **AJ (DAVIS First)** | **62.2** vs 56.2 | **62.2** vs — | **62.2** vs 52.8 |
| **$\delta^{\text{occ}}_{\text{avg}}$ (DR)** | **37.6** vs 27.2 | **37.6** vs 28.5 | — |
| **실용성** | 높음 | 높음 | 낮음 (테스트 시 최적화) |

CoTracker의 핵심 차별점은 **공동 추적을 통한 가림 상황에서의 대폭적 성능 향상**과 **프록시 토큰을 통한 확장 가능한 밀집 추적**의 결합에 있다. 특히 $\delta^{\text{occ}}_{\text{avg}}$에서의 향상(TAPIR 대비 +38.2%, PIPs++ 대비 +31.9%)은 공동 추적 패러다임의 가치를 명확히 입증한다.

---

## 참고자료

1. Karaev, N., Rocco, I., Graham, B., Neverova, N., Vedaldi, A., & Rupprecht, C. (2024). "CoTracker: It is Better to Track Together." *arXiv:2307.07635v3* [cs.CV]. https://co-tracker.github.io/
2. Doersch, C., et al. (2022). "TAP-Vid: A Benchmark for Tracking Any Point in a Video." *arXiv*.
3. Doersch, C., et al. (2023). "TAPIR: Tracking Any Point with Per-frame Initialization and Temporal Refinement." *arXiv*.
4. Harley, A.W., Fang, Z., & Fragkiadaki, K. (2022). "Particle Video Revisited: Tracking Through Occlusions Using Point Trajectories." *Proc. ECCV*.
5. Zheng, Y., et al. (2023). "PointOdyssey: A Large-Scale Synthetic Dataset for Long-Term Point Tracking." *Proc. ICCV*.
6. Teed, Z. & Deng, J. (2020). "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow." *Proc. ECCV*.
7. Wang, Q., et al. (2023). "Tracking Everything Everywhere All at Once." *arXiv*.
8. Neoral, M., Šerých, J., & Matas, J. (2023). "MFT: Long-Term Tracking of Every Pixel." *arXiv*.
9. Shi, X., et al. (2023). "VideoFlow: Exploiting Temporal Cues for Multi-frame Optical Flow Estimation." *arXiv*.
