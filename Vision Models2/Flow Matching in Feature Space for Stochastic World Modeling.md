# Flow Matching in Feature Space for Stochastic World Modeling

---

## 📌 참고 자료

- **주 논문**: Porcher, F., Carion, N., Alahari, K., & Chen, S. (2026). *Flow Matching in Feature Space for Stochastic World Modeling*. arXiv:2606.29059v1.
- **코드**: https://github.com/facebookresearch/flowwm
- **관련 논문들** (논문 내 인용 기반):
  - Lipman et al. (2023). *Flow Matching for Generative Modeling*. ICLR.
  - Liu et al. (2022). *Flow Straight and Fast: Rectified Flow*. arXiv:2209.03003.
  - Zhou et al. (2024). *DINO-WM: World Models on Pre-trained Visual Features Enable Zero-Shot Planning*. arXiv:2411.04983.
  - Walker et al. (2025). *Generalist Forecasting with Frozen Video Models via Latent Diffusion*. arXiv:2507.13942.
  - Zheng et al. (2025). *Diffusion Transformers with Representation Autoencoders (RAE)*. arXiv:2510.11690.
  - Peebles & Xie (2023). *Scalable Diffusion Models with Transformers (DiT)*. ICCV.
  - Esser et al. (2024). *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis*. arXiv:2403.03206.
  - Siméoni et al. (2025). *DINOv3*. arXiv:2508.10104.
  - Ho et al. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS.
  - Wan et al. (2025). *WAN: Open and Advanced Large-Scale Video Generative Models*. arXiv:2503.20314.
  - Mousakhan et al. (2025). *Orbis: Overcoming Challenges of Long-Horizon Prediction in Driving World Models*. arXiv:2507.13162.
  - Assran et al. (2025). *V-JEPA 2*. arXiv:2506.09985.
  - Karypidis et al. (2024). *DINO-Foresight*. arXiv:2412.11673.
  - Sun et al. (2020). *Waymo Open Dataset*. CVPR.
  - Raissi et al. (2019). *Physics-Informed Neural Networks*. Journal of Computational Physics.
  - Su et al. (2024). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. Neurocomputing.

---

## 1. 핵심 주장과 주요 기여 요약

### 🎯 핵심 주장

FlowWM은 다음 두 가지 기존 접근법의 근본적 한계를 동시에 극복하고자 한다:

| 기존 방법 | 한계 |
|-----------|------|
| VAE 기반 Stochastic World Model | 저차원 재구성 잠재 공간 → 지각(perception) 성능 제한 |
| 사전학습 피처 기반 Deterministic World Model | 다중 미래를 단일 평균값으로 붕괴 (mode collapse) |

**핵심 주장**: 고차원 사전학습 피처 공간(DINOv3)에서 직접 Flow Matching을 수행하는 확률론적 세계 모델이 두 목표(다양한 미래 생성 + 지각 친화적 표현 유지)를 모두 달성할 수 있다.

### 🏆 주요 기여

1. **FlowWM 제안**: 고차원 사전학습 피처 공간에서 직접 동작하는 최초의 체계적 확률론적 세계 모델
2. **고차원 피처 공간 Flow Matching을 위한 설계 원칙 규명**: 아키텍처(Wide Projection Head), 타임스텝 스케줄링(shifted schedule), 훈련 목표(시간 일관성 + 태스크 기반 손실)
3. **미분 가능 One-Step Projection 메커니즘**: 전체 ODE 궤적 역전파 없이 효율적으로 다운스트림 손실 적용 가능
4. **두 가지 새로운 벤치마크 도입**: 합성(Bouncing Shapes) + 실세계(FuturePerception on Waymo)

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

#### 문제 1: 결정론적 예측기의 Mode Collapse

결정론적 예측기를 $\ell_2$ 손실로 훈련하면, 이분 미래 분포에서:

$$Y = \begin{cases} y_L, & \text{with probability } \frac{1}{2} \\ y_R, & \text{with probability } \frac{1}{2} \end{cases}$$

$\ell_2$ 위험 함수를 최소화하면:

$$R_2(\hat{y}) = \frac{1}{2}(y_L - \hat{y})^2 + \frac{1}{2}(y_R - \hat{y})^2$$

최적 해가 $\hat{y}^\star = \frac{1}{2}(y_L + y_R)$로, **어떤 유효한 미래에도 대응하지 않는 점**으로 붕괴된다.

더 일반적으로, 결정론적 예측기의 최적해는 조건부 기댓값:

$$\hat{y}^\star(x_t) = \mathbb{E}[Y \mid X_t = x_t]$$

이며, 이는 모든 가능한 미래의 평균으로서 어떤 유효한 미래 모드도 아니다.

#### 문제 2: VAE 잠재 공간의 의미론적 빈곤

VAE 잠재 공간은 픽셀 수준 재구성에 최적화되어 있어 객체 검출, 깊이 추정 등 고수준 지각 태스크에 부적합하다.

#### 문제 3: 고차원 피처 공간에서의 확산 모델 불안정성

표준 DiT 레시피는 저차원 VAE 잠재 공간을 위해 설계되어, 고차원 DINOv3 피처 공간($D=384$, 패치 토큰 단위)에서는 불안정하거나 준최적이다.

---

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 문제 정의: 잠재 공간 세계 모델링

비디오 시퀀스 $I_{1:T}$에서 첫 $T_{\text{context}}$ 프레임을 동결된 인코더 $E$ (DINOv3)로 인코딩:

$$x_{\text{ctx}} = x_{1:T_{\text{context}}} = E(I_{1:T_{\text{context}}}), \quad x_{\text{ctx}} \in \mathbb{R}^{T_{\text{context}} \times H \times W \times D}$$

목표: 미래 잠재 $\hat{x}\_{\text{future}} = \hat{x}\_{T_{\text{context}}+1:T} \in \mathbb{R}^{T_{\text{target}} \times H \times W \times D}$ 생성

#### 2.2.2 Flow Matching

표준 선형 확률 경로를 사용:

$$\boxed{x_\tau = (1 - \tau) x_0 + \tau x_1, \quad \tau \in [0, 1]}$$

- $x_0 \sim p_{\text{noise}}$: 노이즈 잠재
- $x_1$: 실제 미래 프레임의 데이터 잠재 (컨텍스트 $x_{\text{ctx}}$에 조건부)
- 선형 경로에서 ground-truth 속도장: $u^\star(x_\tau, \tau) = x_1 - x_0$

신경망 $u_\theta(x_\tau, x_{\text{ctx}}, \tau)$가 이 속도를 근사, 조건부 분포 $p(x_{\text{future}} | x_{\text{ctx}})$ 학습.

샘플링: ODE $\frac{dx_\tau}{d\tau} = u_\theta(x_\tau, x_{\text{ctx}}, \tau)$를 $\tau=0$에서 $\tau=1$로 적분.

#### 2.2.3 표준 Flow Matching 손실

$$\boxed{\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{\substack{x_1 \sim p_{\text{data}},\, x_0 \sim p_{\text{noise}},\\ \tau \sim \mathcal{U}(0,1)}} \left[ \| u_\theta(x_\tau, \tau) - (x_1 - x_0) \|_2^2 \right]}$$

#### 2.2.4 One-Step Projection (핵심 기법)

주어진 $x_\tau$에서, 전체 ODE 적분 없이 최종 엔드포인트를 미분 가능하게 추정:

$$\boxed{\tilde{x}_1(\tau) = x_\tau + (1 - \tau)\, u_\theta(x_\tau, \tau)}$$

**수학적 근거**: $x_\tau = (1-\tau)x_0 + \tau x_1$을 이용하면:

$$x_{1,\text{pred}} - x_1 = (1 - \tau)\left(u_\theta(x_\tau, \tau) - (x_1 - x_0)\right)$$

엔드포인트 손실이 Flow Matching 손실에 시간 의존적 가중치를 암묵적으로 유도함:

$$\|x_{1,\text{pred}} - x_1\|^2 = (1 - \tau)^2 \|u_\theta(x_\tau, \tau) - (x_1 - x_0)\|^2$$

#### 2.2.5 시간 일관성 손실 (Temporal Consistency Loss)

Ground-truth 시간 미분 정의:

$$\Delta x_t^{\text{GT}} = x_{t+1}^{\text{GT}} - x_t^{\text{GT}}$$

One-step projection 기반 예측 잠재의 시간 차이가 실제 잠재의 시간 차이와 정렬되도록 강제:

$$\boxed{\mathcal{L}_{\text{temporal}} = \sum_{t=1}^{T-1} \left\| \Delta\tilde{x}_t - \Delta x_t^{\text{GT}} \right\|^2}$$

여기서 $\tilde{x}_{1:T}$는 비디오 프레임의 projected endpoints.

#### 2.2.6 태스크 기반 손실 (Task-Driven Objective)

시간 의존적 가중치 스케줄 $\lambda(\tau) = \tau^\gamma$를 사용한 전체 훈련 목표:

$$\mathcal{L} = \mathcal{L}_{\text{flow}} + \lambda_\tau \mathcal{L}_{\text{det}}$$

One-step projection으로 얻은 $\tilde{x}_1$에 동결된 검출기를 적용, 검출 손실을 역전파. 파라미터 기울기:

$$\nabla_\theta \mathcal{L}_{\text{aux}} = (1 - \tau) \left(\frac{\partial u_\theta(x_\tau, \tau)}{\partial \theta}\right)^\top \nabla_{x_{1,\text{pred}}} R$$

전체 샘플링 궤적 역전파 불필요.

#### 2.2.7 타임스텝 스케줄 시프팅

고차원 잠재 공간에서의 신호 대 잡음비(SNR) 유지를 위해 Esser et al. (2024)의 해상도 의존적 타임스텝 시프트 적용:

$$\boxed{\tau' = \frac{\alpha \tau}{1 + (\alpha - 1) \tau}}$$

- $\alpha$: 노이즈 방향으로 확률 질량을 재분배하는 스케일 인수
- 엔드포인트 보존 ($\tau=0, 1$)하면서 중간 분포를 더 노이즈한 타임스텝으로 편향

---

### 2.3 모델 구조

```
입력: 노이즈 타겟 잠재 x_τ ∈ ℝ^{D×T_target×H×W}
조건: 컨텍스트 패치 토큰 x_ctx ∈ ℝ^{D×T_context×H×W}

┌─────────────────────────────────────────┐
│     DiT Backbone (2 blocks, dim=256)    │
│  ┌──────────────────────────────────┐   │
│  │ AdaLN → Self-Attention (RoPE)   │   │
│  │ AdaLN → Cross-Attention (RoPE)  │   │← x_ctx
│  │ AdaLN → FFN                      │   │
│  └──────────────────────────────────┘   │
│         × 2 blocks                      │
├─────────────────────────────────────────┤
│   Wide Projection Head (2 layers)       │
│   (projection dim=1024 > D=384)         │
│   with AdaLN conditioning (τ)           │
└─────────────────────────────────────────┘
출력: 속도장 u_θ(x_τ, τ) ∈ ℝ^{D×T_target×H×W}
```

**주요 설계 선택들**:
- **RoPE(Rotary Position Embedding)**: 시간적(temporal) + 공간적(spatial) 위치 임베딩 분리
- **Query-Key Normalization**: 훈련 안정성 향상
- **Wide Head**: 투영 차원(1024)이 패치당 잠재 차원(384)보다 크게 설정 → 고차원 속도장 예측의 핵심
- **Multi-layer Feature Pooling**: DINOv3 ViT-S의 3, 6, 9, 12번째 레이어에서 피처 평균 풀링

---

### 2.4 성능 향상

#### 합성 벤치마크 (Bouncing Shapes)

| 방법 | Prec. ↓ | Rec. ↓ | F1 ↓ |
|------|---------|--------|------|
| DINO-WM | 16.1 | 19.9 | 17.8 |
| Deterministic Predictor | 15.7 | 18.8 | 17.1 |
| Walker et al. (2025) | 14.5 | 14.3 | 14.4 |
| FlowWM w/o TC | 4.57 | 4.49 | 4.53 |
| **FlowWM (Ours)** | **4.35** | **4.28** | **4.31** |
| Oracle | 1.01 | 0.98 | 1.00 |

- Walker et al. 대비 F1 오류 **70% 감소** (14.4 → 4.31)

#### 실세계 벤치마크 (FuturePerception on Waymo)

| 방법 | APL(3) ↑ | APL(6) ↑ | RMSE ↓ | $\delta_1$ ↑ |
|------|---------|---------|--------|------------|
| DINO-WM | 14.5 | 14.5 | 0.12 | 0.51 |
| Deterministic Predictor | 15.2 | 15.2 | 0.11 | 0.54 |
| WAN 2.2 VAE WM | 17.5 | 18.2 | 0.10 | 0.57 |
| Walker et al. (2025) | 16.5 | 17.1 | 0.102 | 0.56 |
| **FlowWM (Ours)** | **20.9** | **21.7** | **0.078** | **0.723** |
| Oracle | 65.1 | 65.1 | 0.043 | 0.854 |

- 객체 검출 APL(3): Deterministic 대비 **+37.5%** (15.2 → 20.9)
- 깊이 예측 RMSE: **35%** 개선 (0.12 → 0.078)
- FVD: Deterministic 대비 **43%** 개선 (152.4 → 87.3)

#### Ablation 주요 결과

| 구성 요소 | APL(3) ↑ |
|----------|---------|
| Deterministic Predictor | 15.2 |
| + Flow Matching (Walker et al.) | 16.5 |
| + Wide Head + Shifted Schedule ($\alpha=13$) | 19.1 |
| + Temporal Consistency | 20.1 |
| + Task-driven Objective | 20.9 |
| + More Samples (3→6) | 21.7 |

---

### 2.5 한계

논문이 명시한 한계 (Appendix J):

1. **잠재 공간의 "확산 가능성(Diffusability)"**: 어떤 표현 공간이 Flow Matching에 더 적합한지, 그리고 이를 최적화하는 방법(end-to-end 훈련)이 미해결 문제
2. **동결된 인코더 의존성**: DINOv3 인코더가 동결되어 있어 예측 미래의 표현 품질이 인코더 품질에 제한됨
3. **Action-conditioned 미지원**: 현재 컨텍스트 비디오에만 조건부. 행동 조건부 확장이 필요
4. **Oracle과의 큰 격차**: APL(3) 20.9 vs. Oracle 65.1 — 실세계 미래 예측의 본질적 어려움
5. **태스크 기반 손실의 제한적 효과**: 동결된 검출기가 노이즈가 많은 예측 잠재에 덜 견고해 약한 기울기 신호 제공

---

## 3. 일반화 성능 향상 가능성

### 3.1 의미론적 피처 공간 활용의 일반화

FlowWM이 일반화 성능을 향상시킬 수 있는 핵심 메커니즘은 **DINOv3 피처 공간의 의미론적 풍부함**에 있다.

DINOv3는 다양한 시각적 태스크(분할, 검출, 깊이 추정)에서 강력한 표현력을 보이며, 이 공간에서 미래를 예측함으로써:

$$x_{\text{ctx}} \in \mathbb{R}^{T_{\text{context}} \times H \times W \times D} \xrightarrow{u_\theta} \hat{x}_{\text{future}} \in \mathbb{R}^{T_{\text{target}} \times H \times W \times D}$$

예측된 미래 피처는 별도 학습 없이 **여러 하위 태스크에 범용적으로 적용** 가능하다.

### 3.2 다중 미래 샘플링의 일반화 기여

확률론적 모델의 핵심 이점은 동일한 컨텍스트에서 다양한 미래를 샘플링할 수 있다는 점이다. Best-of- $N$ 메트릭 $\text{AP}_L(N)$에서:

- $N=1$: 단일 샘플 검출 성능
- $N$이 증가할수록 성능이 단조 증가

반면 결정론적 예측기는 $N$이 늘어도 성능 향상 없음. 이는 **실세계의 분포 이동(distribution shift)**에 대한 강건성을 의미: 모델이 특정 단일 미래에 과적합하지 않고 가능한 미래 공간을 커버.

### 3.3 태스크 기반 손실의 도메인 일반화

One-step projection을 통한 태스크 기반 손실은 특정 다운스트림 태스크의 inductive bias를 세계 모델에 주입:

$$\mathcal{L} = \mathcal{L}_{\text{FM}} + \lambda_\tau \mathcal{L}_{\text{det}}$$

단, 현재 구현의 한계: 동결된 검출기가 노이즈 잠재에 덜 견고 → 태스크별 기울기 신호 약화.

**개선 가능성**: 
- 태스크별 전문 손실 설계
- 다수 태스크 동시 훈련 (멀티태스크 학습)
- 더 견고한 태스크 헤드(노이즈에 강한 아키텍처) 사용

### 3.4 시간 일관성 손실의 도메인 일반화

$$\mathcal{L}_{\text{temporal}} = \sum_{t=1}^{T-1} \|\Delta\tilde{x}_t - \Delta x_t^{\text{GT}}\|^2$$

이 손실은 태스크 불가지론적(task-agnostic)이며, 물리적으로 타당한 운동 패턴을 강제. 자율주행 외 로보틱스, 스포츠 분석 등 도메인에서도 물리적 일관성이 중요하므로 **도메인 간 전이에 유리**.

### 3.5 타임스텝 시프팅의 차원 일반화

$$\tau' = \frac{\alpha \tau}{1 + (\alpha - 1) \tau}$$

이 전략은 이미지 해상도 증가에서 발견된 원칙을 잠재 차원 증가에도 적용 가능함을 실험적으로 검증($\alpha=13$이 최적). 이는 다른 고차원 피처 공간(예: CLIP, VJEPA-2)으로의 전이에도 적용 가능한 일반적 원칙을 시사.

### 3.6 한계 지점: 일반화 한계

- **인코더 특정성**: DINOv3에 의존. 다른 인코더로 전환 시 $\alpha$ 등 하이퍼파라미터 재조정 필요
- **도메인 특화 평가**: Waymo 자율주행 데이터셋에만 실세계 검증 → 로보틱스, 의료 영상 등 타 도메인 일반화 미검증
- **동결 인코더**: 피처 품질이 인코더 품질에 제한되어, 인코더가 학습하지 못한 시각적 패턴 예측 불가

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 VAE 기반 확률론적 세계 모델

| 논문 | 방법 | 특징 | FlowWM 대비 |
|------|------|------|------------|
| Ho et al., NeurIPS 2020 (DDPM) | 픽셀 공간 확산 | 강력한 생성 품질 | 낮은 의미론적 충실도 |
| Alonso et al., NeurIPS 2024 | Atari용 VAE 잠재 확산 | 게임 환경 특화 | 실세계 지각 태스크 부적합 |
| Wan et al., 2025 (WAN 2.2) | 비디오 VAE ($D=16$) + FM | 고품질 픽셀 생성 | APL(3): 17.5 vs. FlowWM 20.9 |

핵심 차이: VAE 잠재 공간의 의미론적 빈곤 → 재인코딩 실험(17.5 → 16.5 APL)으로 입증.

### 4.2 결정론적 피처 공간 세계 모델

| 논문 | 방법 | 특징 | 한계 |
|------|------|------|------|
| Zhou et al., 2024 (DINO-WM) | DINOv2 피처 자기회귀 예측 | 제로샷 계획 가능 | 단일 미래 예측, APL(3): 14.5 |
| Karypidis et al., 2024 (DINO-Foresight) | DINO 피처 예측 | 지각 지향 | 결정론적 한계 |
| Hansen et al., 2023 (TD-MPC2) | 잠재 공간 MPC | 연속 제어에 강 | 확률론적 환경에 취약 |
| Assran et al., 2025 (V-JEPA 2) | 자기지도 비디오 JEPA | 강력한 인코더 | 예측기는 결정론적 |

### 4.3 고차원 잠재 공간 생성 모델

| 논문 | 방법 | 특징 | 관련성 |
|------|------|------|--------|
| Zheng et al., 2025 (RAE) | DINO 잠재 공간 확산 + Wide Head | 이미지 생성 SOTA | FlowWM Wide Head 설계에 영향 |
| Li & He, 2025 (JIT) | 데이터 매니폴드 직접 투영 | 단순성 강조 | 비디오 시간 일관성 미고려 |
| Esser et al., 2024 (Flux/SD3) | 고해상도 이미지용 타임스텝 시프트 | 해상도 적응 | 시프팅 전략 차용 |

### 4.4 확률론적 고차원 피처 세계 모델

| 논문 | 방법 | 특징 | FlowWM 대비 |
|------|------|------|------------|
| Walker et al., 2025 | DINOv3 피처 잠재 확산 (표준 DiT) | 최초 유사 시도 | 설계 미최적화, APL(3): 16.5 |
| Mousakhan et al., 2025 (Orbis) | DINO → 저차원 VAE 증류 후 확산 | 고차원 회피 | 의미론적 정보 손실 가능 |

### 4.5 태스크 기반 확산 모델 훈련

| 논문 | 방법 | 방식 | FlowWM 대비 |
|------|------|------|------------|
| Clark et al., 2024 | 미분 가능 보상 역전파 (전체 궤적) | BPTT 사용 | 계산 비용 高, 기울기 불안정 |
| Prabhudesai et al., 2023 | 랜덤 $K$-step 역전파 + LoRA | BPTT 부분적 | 여전히 비용 부담 |
| He et al., 2023 (ReFL) | 단일 추가 역전파 스텝 | BPTT 최소화 | 여전히 순방향 패스 필요 |
| **FlowWM** | One-step Projection | BPTT 없음 | 계산 오버헤드 無, 안정적 |

### 4.6 비교 요약표

| 특성 | DINO-WM | Walker et al. | Orbis | WAN 2.2 | **FlowWM** |
|------|---------|--------------|-------|---------|-----------|
| 피처 공간 | 고차원(DINO) | 고차원(DINO) | 저차원(VAE) | 저차원(VAE) | **고차원(DINO)** |
| 확률론적 | ❌ | ✅ | ✅ | ✅ | **✅** |
| 시간 일관성 | 부분적 | ❌ | 부분적 | ❌ | **✅** |
| 태스크 기반 훈련 | ❌ | ❌ | ❌ | ❌ | **✅** |
| 고차원 최적화 설계 | N/A | ❌ | N/A | N/A | **✅** |
| APL(3) | 14.5 | 16.5 | N/A | 17.5 | **20.9** |

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5.1 연구에 미치는 영향

#### 5.1.1 세계 모델 패러다임의 전환

FlowWM은 세계 모델의 핵심 설계 공간을 다음과 같이 재정의한다:

$$\text{(VAE 재구성 공간)} \rightarrow \text{(사전학습 의미론적 공간)} \\ \text{(결정론적)} \rightarrow \text{(확률론적 Flow Matching)}$$

이는 향후 세계 모델 연구가 **"어떤 표현 공간이 세계 모델링에 최적인가"**를 핵심 질문으로 다루게 만들 것으로 예상된다.

#### 5.1.2 One-Step Projection의 광범위한 적용

$$\tilde{x}_1(\tau) = x_\tau + (1 - \tau)\, u_\theta(x_\tau, \tau)$$

이 기법은 Flow Matching/확산 모델에 **임의의 미분 가능 손실을 효율적으로 통합**하는 범용 메커니즘. 다음 분야에서의 적용 연구를 촉진할 것으로 예상:
- 강화학습 보상 신호를 생성 모델에 통합
- 물리 시뮬레이션 제약을 비디오 생성에 적용
- 의료 영상의 임상 지표 기반 생성 모델 훈련

#### 5.1.3 고차원 피처 공간 생성 모델링 연구 촉진

RAE, JIT 등 이미지 생성 분야에서의 고차원 피처 공간 생성을 비디오/세계 모델링으로 확장하는 연구 방향을 공식화.

#### 5.1.4 인식-생성 통합 평가 프레임워크

FuturePerception 벤치마크는 생성 품질을 픽셀 수준(FID, FVD)이 아닌 **다운스트림 인식 태스크(APL, RMSE)**로 평가하는 새로운 패러다임을 제안. 이는 자율주행, 로보틱스 분야의 세계 모델 평가 표준에 영향을 미칠 것.

### 5.2 앞으로 연구 시 고려할 점

#### 5.2.1 인코더 선택과 "확산 가능성(Diffusability)"

**연구 질문**: 어떤 표현 공간이 Flow Matching에 더 적합한가?

고려 요소:
- **등방성(Isotropy)**: 피처 공간이 등방적일수록 가우시안 노이즈 기반 Flow Matching에 적합
- **선형성**: 선형 보간 $x_\tau = (1-\tau)x_0 + \tau x_1$이 의미론적으로 타당한 중간 상태를 생성해야 함
- **차원 수**: 너무 높은 차원은 타임스텝 시프트 $\alpha$ 재조정 필요

```
향후 연구 방향:
DINOv3 vs. CLIP vs. VJEPA-2 vs. SAM2 인코더 공간의 
"확산 가능성" 체계적 분석
```

#### 5.2.2 End-to-End 세계 모델 훈련

현재 FlowWM은 동결된 인코더에 의존. End-to-End 훈련 시 해결해야 할 문제:

- **표현 붕괴(Representation Collapse)**: 인코더가 예측 모델에 과적합
- **JEPA 스타일 훈련**: V-JEPA 2처럼 예측 목표를 통한 인코더 학습
- **안정화 기법**: EMA(Exponential Moving Average), Stop-Gradient 등

#### 5.2.3 Action-Conditioned 확장

자율주행·로보틱스에서의 실용적 배포를 위해:

$$p(x_{\text{future}} | x_{\text{ctx}}, a_{1:T_{\text{target}}})$$

행동 조건부 확률론적 세계 모델에서의 고려사항:
- **행동 인코딩 방식**: Cross-attention vs. FiLM conditioning
- **계획(Planning) 통합**: Cross-Entropy Method, MPPI 등과의 결합
- **불확실성 인식 계획**: 확률론적 미래를 활용한 리스크 인식 경로 계획

#### 5.2.4 타임스텝 가중치 최적화

태스크 기반 손실에서 $\lambda(\tau) = \tau^\gamma$의 최적 $\gamma$ 선택:

- $\gamma$ 너무 작음 → Flow Matching 목표 과제약
- $\gamma$ 너무 큼 → $\tau \to 0$ 영역 무시, 노이즈 제거 능력 저하
- **적응적 $\gamma$ 스케줄링** 또는 **메타러닝 기반 최적화** 연구 필요

#### 5.2.5 추론 효율성 개선

현재 FlowWM은 50-step Euler 적분으로 샘플링:

- **일관성 모델(Consistency Models)**: 단일 스텝 샘플링
- **플로우 蒸留(Flow Distillation)**: 고품질 소수 스텝 샘플링
- **조건부 일관성 증류**: 세계 모델 특화 빠른 샘플링

실시간 자율주행 적용을 위해 추론 지연(latency) 감소는 필수적.

#### 5.2.6 멀티모달 입력 확장

현재는 단일 카메라 비디오에서 컨텍스트 추출. 실세계 확장 고려사항:
- **다중 카메라**: 자율주행의 서라운드뷰 통합
- **LiDAR/Radar 융합**: 3D 공간 인식 강화
- **언어 조건부**: 자연어로 미래 시나리오 지정

#### 5.2.7 평가 메트릭 표준화

FuturePerception 벤치마크가 새로운 평가 패러다임을 제안하지만:
- **Waymo 한정**: 다른 데이터셋(nuScenes, KITTI 등)으로의 확장 필요
- **Oracle 격차 해석**: APL(3)=20.9 vs. Oracle=65.1의 격차가 예측 모델 한계인지 평가 방법론 한계인지 불명확
- **공정한 비교 기준**: 모델 크기, 학습 데이터 규모를 통제한 비교

---

## 🔚 종합 결론

FlowWM은 "**의미론적으로 풍부한 표현 공간에서의 확률론적 세계 모델링**"이라는 새로운 설계 축을 확립한 연구다. One-step projection, 시간 일관성 손실, 타임스텝 시프팅의 세 가지 기여가 시너지를 이루며 기존 방법 대비 압도적인 성능을 달성했다. 향후 연구는 end-to-end 훈련, 행동 조건부 확장, 추론 효율화, 다양한 도메인으로의 일반화를 중심으로 진행될 것으로 전망된다.
