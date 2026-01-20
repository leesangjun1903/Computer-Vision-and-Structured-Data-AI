# Consistency Models

### **1. 핵심 주장 및 주요 기여**

"Consistency Models" (Song et al., ICML 2023)는 확산 모델의 근본적인 한계를 해결하는 새로운 생성 모델 패밀리를 제시합니다. 기존 확산 모델은 고품질 샘플 생성을 위해 10~2000배의 반복 계산이 필요한 반면, 일관성 모델은 **단일 네트워크 평가(one-step generation)로 고품질 샘플을 직접 생성**할 수 있습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)

본 논문의 핵심 기여는 다음과 같습니다:

**첫째, 확률 흐름 ODE(Probability Flow ODE, PF-ODE) 궤적에서의 자가 일관성(self-consistency) 개념**. 임의의 두 시점에서의 데이터-잡음 쌍이 동일한 원래 상태로 매핑되어야 한다는 수학적 원리를 활용합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)

**둘째, 두 가지 실용적 훈련 방식**:
- **일관성 증류(Consistency Distillation)**: 사전 훈련된 확산 모델로부터 지식을 추출 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)
- **일관성 훈련(Consistency Training)**: 사전 훈련된 모델 없이 독립적 학습 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)

**셋째, 영인 학습(zero-shot) 이미지 편집** 기능으로 이미지 복원, 색상화, 초해상도, 인페인팅 등 추가 학습 없이 수행 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)

### **2. 문제 정의 및 제안 방법**

#### **2.1 해결하고자 하는 문제**

확산 모델의 병목:
- CIFAR-10에서 단일 이미지 생성에 수백 번의 반복 필요
- GAN/VAE 대비 10-2000배 높은 계산 비용
- 실시간 응용 불가능
- 기존 증류 방법(Progressive Distillation)은 여러 단계 필요 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)

#### **2.2 핵심 수학적 개념**

**확률 흐름 ODE 정의**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)

$$\frac{dx_t}{dt} = \left[\mu(x_t, t) - \frac{1}{2}\sigma(t)^2 \nabla \log p_t(x_t)\right]dt$$

여기서:
- $p_t(x)$: 시점 $t$에서의 데이터 분포
- $\mu(\cdot,\cdot)$: 드리프트 항
- $\sigma(\cdot)$: 확산 계수
- $\nabla \log p_t(x_t)$: 스코어 함수

**일관성 함수(Consistency Function)**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)

$$f: (x_t, t) \mapsto x_\epsilon$$

자가 일관성 성질: $f(x_t, t) = f(x_{t'}, t')$ for all $t, t' \in [\epsilon, T]$ on same trajectory

매핑 함수 \(f_{\theta }\)는 자기 일관성(self-consistency) 속성을 가지도록 훈련됩니다.  
즉, 동일한 확률 흐름 상미분 방정식(Probability Flow ODE) 궤적에 속하는 임의의 두 시점 $\((x_{t},t)\)와 \((x_{t^{\prime }},t^{\prime })\)$ 에 대해 출력값이 같아야 합니다 

### 어떻게 이게 가능하지?

이러한 자기 일관성(self-consistency) 속성이 가능한 핵심 원리는 확률 흐름 상미분 방정식(PF ODE)의 궤적 자체가 결정론적이기 때문입니다.  
각 궤적은 하나의 고유한 깨끗한 이미지 $(\(x_{0}\))$ 에서 시작하여 하나의 고유한 최종 노이즈 분포 $(\(x_{T}\))$ 로 이어집니다.

- 결정론적 궤적: 확산 모델의 PF ODE는 데이터에서 노이즈로 가는 부드럽고 결정론적인 경로를 정의합니다. 즉, 한 번 경로가 결정되면 그 경로 상의 모든 시점 $\(t\)$ 에서의 점들 $\(x_{t}\)$ 은 서로 수학적으로 고정된 관계를 갖습니다.
- 단일 목적지: 동일한 궤적에 있는 모든 $\(x_{t}\)$ 와 $\(x_{t^{\prime }}\)$ 는 결국 동일한 최종 깨끗한 이미지 $\(x_{\epsilon }\)$ (또는 $\(x_{0}\)$ )에 도달하게 되어 있습니다.
- 매핑 학습: 일관성 모델 $\(f_{\theta }\)$ 는 이 결정론적 관계를 학습하여 궤적 상의 어떤 지점을 입력받더라도 동일한 시작점 $(\(x_{\epsilon }\))$ 으로 매핑하도록 훈련됩니다. 

모델은 다음 두 가지 방식으로 이 속성을 학습합니다. 
- 인접 쌍 비교 (Consistency Distillation): 사전 학습된 확산 모델이나 ODE 솔버를 사용하여 동일한 궤적 상의 인접한 두 시점 $\((x_{t+1},t+1)\)$ 과 $\((x_{t},t)\)$ 의 쌍을 생성합니다. 그 다음, 모델 $\(f_{\theta }\)$ 가 이 두 입력에 대해 동일한 출력(깨끗한 이미지 예측값)을 내놓도록 차이를 최소화하는 방식으로 학습됩니다 (p. 4).
- 경계 조건 (Boundary Condition): 모델은 시작 시점 $(\(t=\epsilon \approx 0\))$ 에서는 입력 $\(x_{\epsilon }\)$ 이 그대로 출력되도록 $(\(f_{\theta }(x_{\epsilon },\epsilon )=x_{\epsilon }\))$ 설계되어, 모델이 자명한 해(trivial solution)인 상수 함수로 수렴하는 것을 방지합니다 (p. 3). 

이러한 훈련 과정을 통해, 모델은 궤적의 모든 점이 하나의 원본 데이터에 해당한다는 암묵적인 지식을 학습하게 됩니다.  
따라서 추론 시에는 가장 노이즈가 많은 초기 상태 $\(x_{T}\)$ 를 모델에 한 번만 입력해도, 학습된 매핑 함수가 즉시 깨끗한 이미지 $\(x_{\epsilon }\)$ 를 출력할 수 있습니다.

#### **2.3 모델 구조**

**1) 경계 조건을 만족하는 파라미터화**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)

$$f_\theta(x,t) = \begin{cases} x & \text{if } t = \epsilon \\ F_\theta(x,t) & \text{if } t \in (\epsilon, T] \end{cases}$$

또는 **스킵 연결 파라미터화**(권장): [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)

$$f_\theta(x,t) = c_{\text{skip}}(t)x + c_{\text{out}}(t)F_\theta(x,t)$$

조건: $c_{\text{skip}}(\epsilon) = 1$, $c_{\text{out}}(\epsilon) = 0$

**2) 일관성 증류 손실함수(Consistency Distillation Loss)**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)

$$\mathcal{L}_{CD}^N(\theta, \theta^-; \phi) := \mathbb{E}[\lambda(t_n)d(f_\theta(x_{t_{n+1}}, t_{n+1}), f_{\theta^-}(\hat{x}_n^\phi, t_n))]$$

여기서:
- $\hat{x}\_n^\phi = x_{t_{n+1}} + (t_n - t_{n+1})\Phi(x_{t_{n+1}}, t_{n+1}; \phi)$ (ODE 솔버 한 단계)
- $\theta^-$: 지수이동평균(EMA)으로 갱신되는 타겟 네트워크 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)
- $d(\cdot,\cdot)$: 메트릭 함수 (L2, L1, 또는 LPIPS)
- $\lambda(t_n)$: 가중치 함수

**3) 일관성 훈련(Consistency Training) 손실함수**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)

$$\mathcal{L}_{CT}^N(\theta, \theta^-) := \mathbb{E}[\lambda(t_n)d(f_\theta(x + t_{n+1}z, t_{n+1}), f_{\theta^-}(x + t_nz, t_n))]$$

여기서 $z \sim \mathcal{N}(0, I)$, $x \sim p_{\text{data}}$

**핵심 통찰**: 스코어 함수의 비편향 추정자: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)

$$\nabla \log p_t(x_t) = -\mathbb{E}\left[\frac{x_t - x}{t^2} \Big| x_t\right]$$

#### **2.4 다단계 샘플링 알고리즘**

**알고리즘 1: 다단계 일관성 샘플링**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)

```
입력: 일관성 모델 f_θ(·,·), 시간 수열 τ₁ > τ₂ > ... > τₙ₋₁
x ← f_θ(x̂_T, T)  # x̂_T ~ N(0, T²I)
for n = 1 to N-1 do
    z ~ N(0, I)
    x̂_{τₙ} ← x + √(τₙ² - ε²)z
    x ← f_θ(x̂_{τₙ}, τₙ)
end for
출력: x
```

이는 **계산량 조절이 가능한 생성**을 실현합니다.

### **3. 성능 향상 및 벤치마크 결과**

#### **3.1 일관성 증류 성능** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)

| 데이터셋 | 방법 | NFE(함수평가) | FID | IS |
|---------|------|-------------|-----|-----|
| CIFAR-10 | 기존 PD (ℓ₂) | 1 | 8.34 | 8.69 |
| CIFAR-10 | **CD (LPIPS)** | **1** | **3.55** | **9.48** |
| CIFAR-10 | **CD (LPIPS)** | **2** | **2.93** | **9.75** |
| ImageNet 64×64 | PD | 1 | 15.39 | - |
| ImageNet 64×64 | **CD** | **1** | **6.20** | - |
| ImageNet 64×64 | **CD** | **2** | **4.70** | - |

**Progressive Distillation 대비 성능**:
- 1단계: CIFAR-10에서 **57.4% FID 개선** (8.34→3.55)
- 2단계: CIFAR-10에서 **47.5% FID 개선** (5.58→2.93)
- 모든 데이터셋에서 일관된 우월성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)

#### **3.2 독립 훈련(Consistency Training) 성능** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)

| 모델 | CIFAR-10 | ImageNet 64×64 |
|-----|---------|---------------|
| StyleGAN2-ADA | 2.92 | - |
| BigGAN | 14.7 | 4.06 |
| VAE 기반 모델들 | 17.9-48.9 | - |
| **CT (1단계)** | **8.70** | **13.0** |
| **CT (2단계)** | **5.83** | **11.1** |

**의미**: GAN 없이 **단일-비적대 생성 모델 중 최강** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)

#### **3.3 메트릭 분석**

**LPIPS 메트릭의 중요성**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)
- Consistency Distillation 성능 분석에서 LPIPS ≫ L2 ≫ L1
- 인지적 유사성을 더 잘 포착하여 일관성 학습에 최적 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)

**ODE 솔버 선택**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)
- Heun의 2차 방법 > Euler의 1차 방법
- N=18 (시간 단계 18개)가 CIFAR-10에서 최적 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)

### **4. 일반화 성능 분석**

#### **4.1 일반화 향상 메커니즘**

**1) 경계 조건의 역할**
- $f_\theta(x, \epsilon) = x$ 강제: 자명한 해(trivial solution) 방지 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)
- 스킵 연결을 통해 미분가능성 보장하여 연속시간 훈련 가능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)

**2) 타겟 네트워크 EMA 갱신**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)

$$\theta^- \leftarrow \text{stopgrad}(\mu \theta^- + (1-\mu)\theta)$$

효과:
- 훈련 안정화
- 과도한 진동 방지
- μ=0.9999로 설정하여 느린 갱신 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)

**3) 적응적 N 스케줄**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)

$$N(k) = \left\lfloor \sqrt{\frac{k}{K}(s_1 + 1)^2 - s_0^2} + s_0^2 - 1 \right\rfloor + 1$$

- 초기: 작은 N (빠른 수렴, 높은 편향)
- 종료: 큰 N (느린 수렴, 낮은 편향, 높은 정확도)

**4) LPIPS 손실의 지각적 일관성**
- 픽셀 공간 L2 손실 vs LPIPS 손실의 성능 차이 > 2배
- 인간 지각과의 정렬 향상 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)

#### **4.2 모드 붕괴 저항성**

**실증 증거**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)
- 동일한 초기 잡음으로부터:
  - CT 생성 이미지: 구조적 다양성 ↑
  - GAN 생성 이미지: 반복된 패턴 ↓
- 설명: CT는 일관성 함수만 학습하므로 모드 붕괴 경향 낮음 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)

**정밀도(Precision) vs 재현(Recall) 분석**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)

| 모델 | Precision | Recall |
|-----|-----------|--------|
| StyleGAN2 | 0.59 | 0.48 |
| CT (1단계) | 0.71 | **0.47** |

- 정밀도는 GAN 수준
- Recall 차이: 다양성 vs 품질 트레이드오프 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)

#### **4.3 한계 및 제약사항**

**1) 다단계 vs 단일단계 품질 격차**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)
- EDM (35 NFE): FID 2.04
- CT (1단계): FID 8.70
- **격차**: ~4.3배 (CIFAR-10)

**2) 이산화 오차 누적**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)
- 정리 1: $\sup_{n,x} \|f_\theta(x,t_n) - f(x,t_n;\phi)\|_2 = O((\Delta t)^p)$
- 더 많은 단계 → 더 낮은 오류이지만 계산 증가 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)

**3) 연속시간 훈련의 불안정성**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)
- 무한 차원 손실 함수의 높은 분산
- 자동 미분 오버헤드
- 실제로는 이산화 훈련이 더 효과적 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)

**4) 초매개변수 민감도**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)
- 시간 단계 수열 $\{t_1, ..., t_N\}$: 탐욕 알고리즘으로 3분 검색
- 삼진 검색의 단봉 가정: 경험적 검증 필요 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)

### **5. 영향 및 미래 연구 방향**

#### **5.1 이론적 기여**

**정리 1 (일관성 증류의 수렴성)**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)
일관성 증류 손실이 0에 수렴하면, ODE 솔버의 국소 오차가 $O((\Delta t)^{p+1})$일 때:

$$\sup_{n,x} \|f_\theta(x,t_n) - f(x,t_n;\phi)\|_2 = O((\Delta t)^p)$$

**정리 2 (일관성 훈련의 동등성)**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/209cffb4-7889-4eb6-ad9f-367e8458aeed/2303.01469v2.pdf)
리프숑 조건과 이중 미분가능성 가정 하에서:

$$\mathcal{L}_{CD}^N(\theta, \theta^-; \phi) = \mathcal{L}_{CT}^N(\theta, \theta^-) + o(\Delta t)$$

#### **5.2 2023-2025년 후속 연구**

**1) 잠재 일관성 모델(LCM, Oct 2023)** [arxiv](https://arxiv.org/abs/2401.05252)
- Stable Diffusion의 잠재 공간에서 훈련
- 768×768 고해상도 2-4단계 생성
- 단일 단계 가이드 증류로 CFG 통합 [arxiv](http://arxiv.org/pdf/2310.04378.pdf)

**2) 절단 일관성 모델(TCM, 2024)** [arxiv](https://arxiv.org/html/2410.14895v2)
- 생성 vs 복원 작업 간 용량 충돌 해결
- 2단계 훈련: 사전훈련 + 절단시간 훈련
- ImageNet FID: 3.0 (일관성 모델 SOTA) [arxiv](https://arxiv.org/html/2410.14895v2)

**3) 흐름 일치(Flow Matching) vs 일관성 모델** [emergentmind](https://www.emergentmind.com/topics/rectified-flow-matching)
- 직선 궤적 학습 vs 일관성 조건
- **Rectified Flow**: 더 빠른 수렴, 때로 낮은 초기 품질
- **통합 접근법** (SCoT, Consistency-FM): 두 가지 장점 결합 [arxiv](https://arxiv.org/html/2502.16972v4)

**4) 분포 일치 증류(DMD, Nov 2023)** [arxiv](https://arxiv.org/abs/2311.18828)
- 다른 증류 목표: FID 2.62 on ImageNet 64×64
- 일관성 모델과 경쟁적 성능 [arxiv](https://arxiv.org/abs/2311.18828)

**5) 위상별 일관성 모델(PCM, 2024)** [semanticscholar](https://www.semanticscholar.org/paper/784d1036151566394ad7611edc9c0dff02f629af)
- LCM 설계의 3가지 결함 규명
- 16단계까지 우월성 유지
- 비디오 생성 확장 [semanticscholar](https://www.semanticscholar.org/paper/784d1036151566394ad7611edc9c0dff02f629af)

**6) 적응 이산화(ADCM, NeurIPS 2025)** [neurips](https://neurips.cc/virtual/2025/poster/119050)
- 가우스-뉴턴 방법으로 자동 단계 최적화
- 국소 일관성 vs 전역 일관성 트레이드오프 [neurips](https://neurips.cc/virtual/2025/poster/119050)

#### **5.3 연구 고려사항**

**1) 이론적 개선 필요**
- 고차원 비가우스 분포의 수렴 보장 부족
- 조건부 생성에 대한 이론적 분석 미흡
- 일반화 오류 경계의 데이터 의존성

**2) 실무적 과제**
- 연속시간 훈련의 분산 감소 전략
- 더 나은 시간 스케줄 자동화
- 매우 큰 모델(>10B 파라미터)에서의 안정성

**3) 응용 확대**
- 의료 영상 생성 (이미 진행 중)
- 3D 생성, 동영상 생성
- 음성 합성, 음악 생성
- 과학적 시뮬레이션

**4) 하이브리드 접근법**
- 일관성 + 흐름 일치의 장점 통합
- 다중 모드 생성 모델링
- 명시적 다양성 제약

### **6. 최신 비교 분석 (2020-2025)**

| 방법 | 연도 | 핵심 아이디어 | 장점 | 단점 | 적용 범위 |
|-----|------|-------------|------|------|---------|
| **DDIM** | 2020 | 빠른 ODE 솔버 | 단순함 | 여전히 50단계 필요 | 기초 |
| **Progressive Distillation** | 2022 | 다단계 증류 | 체계적 | 여러 훈련 단계 | 중간 |
| **Rectified Flow** | 2022 | 직선 궤적 | 빠른 수렴 | 초기 품질 낮음 | 중간 |
| **Consistency Models** | 2023 | 궤적 일관성 | **단일 단계** | 품질 vs 속도 트레이드오프 | **광범위** |
| **Latent CM** | 2023 | 잠재 공간 일관성 | 고해상도 | 사전훈련 필수 | 실용 |
| **Flow Matching** | 2023 | 속도 필드 | 간단한 ODE | 일관성 모델과 차이점 불명확 | 중간 |
| **DMD** | 2023 | 분포 일치 | 경쟁적 FID | 유연성 부족 | 중간 |
| **TCM** | 2024 | 시간 범위 절단 | 더 나은 1단계 | 복잡한 훈련 | 고급 |
| **Consistency-FM** | 2024 | 속도 일관성 | 빠른 훈련 | 새로운 방법 | 신흥 |
| **SCoT** | 2025 | 통합 프레임워크 | **CM + RFM** | 아직 초기 | **미래** |

### **결론**

Consistency Models는 확산 모델의 **속도 병목을 근본적으로 해결**하는 획기적 접근입니다. 단일 네트워크 평가로 고품질 샘플을 직접 생성하면서도, 여전히 계산량과 품질 간의 유연한 트레이드오프를 제공합니다.

**주요 성과**:
- 일관성 함수라는 새로운 생성 모델링 패러다임 도입
- Progressive Distillation 대비 **50% 이상의 FID 개선**
- 사전훈련 없이도 독립적 학습 가능

**미래 전망**:
- 잠재 공간 및 고차원 데이터로 확장 가능
- 흐름 일치와의 통합으로 더욱 강력한 방법론 창출
- 의료, 3D, 비디오 등 다양한 도메인의 실시간 응용 실현

더 이상 일관성 모델은 이론적 호기심을 넘어 **실제 프로덕션 환경의 표준 기술**로 자리잡고 있습니다.

***

### **참고**

<span style="display:none">[^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_90][^1_91]</span>

<div align="center">⁂</div>

[^1_1]: 2303.01469v2.pdf

[^1_2]: https://arxiv.org/abs/2401.05252

[^1_3]: http://arxiv.org/pdf/2310.04378.pdf

[^1_4]: https://arxiv.org/html/2410.14895v2

[^1_5]: https://www.emergentmind.com/topics/rectified-flow-matching

[^1_6]: https://arxiv.org/html/2502.16972v4

[^1_7]: https://arxiv.org/abs/2311.18828

[^1_8]: https://www.semanticscholar.org/paper/784d1036151566394ad7611edc9c0dff02f629af

[^1_9]: https://neurips.cc/virtual/2025/poster/119050

[^1_10]: https://www.emergentmind.com/topics/distillation-of-flow-matching-models

[^1_11]: https://jurnal.uns.ac.id/jkc/article/view/88610

[^1_12]: https://edu.pubmedia.id/index.php/ptk/article/view/1603

[^1_13]: https://journals.sagepub.com/doi/10.1177/17479541251333942

[^1_14]: https://www.annalsofgeophysics.eu/index.php/annals/article/view/9187

[^1_15]: https://jpfis.unram.ac.id/index.php/GeoScienceEdu/article/view/588

[^1_16]: https://academic.oup.com/clinchem/article/doi/10.1093/clinchem/hvaf086.266/8270054

[^1_17]: https://www.esri.ie/publications/projections-of-regional-demand-and-workforce-requirements-for-general-practice-in

[^1_18]: https://ieeexplore.ieee.org/document/11145817/

[^1_19]: https://rjor.ro/considerations-on-the-efficiency-of-time-series-analysis-in-forecasting-new-influenza-cases-in-the-2024-2025-season/

[^1_20]: https://www.ijfmr.com/research-paper.php?id=62200

[^1_21]: http://arxiv.org/pdf/2310.02279.pdf

[^1_22]: https://arxiv.org/html/2410.14895

[^1_23]: http://arxiv.org/pdf/2212.09068.pdf

[^1_24]: https://arxiv.org/html/2312.05440v1

[^1_25]: http://arxiv.org/pdf/2410.11081.pdf

[^1_26]: https://www.aclweb.org/anthology/D19-1405.pdf

[^1_27]: http://arxiv.org/pdf/2503.05239.pdf

[^1_28]: https://aclanthology.org/2023.blackboxnlp-1.19.pdf

[^1_29]: https://icml.cc/virtual/2023/28091

[^1_30]: https://proceedings.neurips.cc/paper_files/paper/2024/file/47ee3941a6f1d23c39b788e0f450e2a7-Paper-Conference.pdf

[^1_31]: https://arxiv.org/abs/2408.08610

[^1_32]: https://proceedings.neurips.cc/paper_files/paper/2024/file/7343a5c976f8399880b695267f1f9e9f-Paper-Conference.pdf

[^1_33]: https://dl.acm.org/doi/10.5555/3618408.3619743

[^1_34]: https://liner.com/review/simple-and-fast-distillation-of-diffusion-models

[^1_35]: https://kimjy99.github.io/논문리뷰/stable-diffusion-3/

[^1_36]: https://arxiv.org/abs/2303.01469

[^1_37]: https://openaccess.thecvf.com/content/CVPR2024/html/Su_D4_Dataset_Distillation_via_Disentangled_Diffusion_Model_CVPR_2024_paper.html

[^1_38]: https://arxiv.org/html/2510.07631v1

[^1_39]: https://liner.com/ko/review/consistency-models-made-easy

[^1_40]: https://www.sciencedirect.com/science/article/pii/S2666827024000811

[^1_41]: https://openreview.net/forum?id=kqHzgTV9AU

[^1_42]: https://arxiv.org/pdf/2511.19269.pdf

[^1_43]: https://arxiv.org/html/2502.08364v2

[^1_44]: https://arxiv.org/html/2510.17858v1

[^1_45]: https://arxiv.org/html/2508.13831v2

[^1_46]: https://arxiv.org/html/2506.07822v2

[^1_47]: https://arxiv.org/abs/2509.10384

[^1_48]: https://arxiv.org/html/2510.17266v1

[^1_49]: https://arxiv.org/abs/2410.07679

[^1_50]: https://arxiv.org/html/2502.09616v1

[^1_51]: https://arxiv.org/html/2508.12222v1

[^1_52]: https://arxiv.org/html/2502.08364v1

[^1_53]: https://arxiv.org/html/2410.07303v2

[^1_54]: https://www.semanticscholar.org/paper/0958594a66f4f6752275cff4e14d463d9de53560

[^1_55]: https://arxiv.org/abs/2408.02993

[^1_56]: https://dl.acm.org/doi/10.1145/3721201.3725428

[^1_57]: https://arxiv.org/abs/2403.12008

[^1_58]: https://arxiv.org/abs/2404.13903

[^1_59]: https://arxiv.org/abs/2404.11925

[^1_60]: https://arxiv.org/abs/2405.01434

[^1_61]: https://arxiv.org/abs/2403.05438

[^1_62]: https://arxiv.org/html/2502.09509v1

[^1_63]: https://arxiv.org/html/2503.08377v1

[^1_64]: https://arxiv.org/html/2407.15171v1

[^1_65]: http://arxiv.org/pdf/2112.10752.pdf

[^1_66]: https://arxiv.org/pdf/2412.19413.pdf

[^1_67]: https://arxiv.org/pdf/2401.16830.pdf

[^1_68]: https://arxiv.org/abs/2405.06535

[^1_69]: https://kimjy99.github.io/논문리뷰/latent-consistency-model/

[^1_70]: https://openaccess.thecvf.com/content/WACV2025W/ImageQuality/papers/Duym_Quantifying_Generative_Stability_Mode_Collapse_Entropy_Score_for_Mode_Diversity_WACVW_2025_paper.pdf

[^1_71]: https://liner.com/ko/review/pixartδ-fast-and-controllable-image-generation-with-latent-consistency-models

[^1_72]: http://mlg.postech.ac.kr/~jtkim/papers/iclr_2022.pdf

[^1_73]: https://liner.com/review/consistency-flow-matching-defining-straight-flows-with-velocity-consistency

[^1_74]: https://www.ieee-hpec.org/2018/2018program/index_htm_files/124.pdf

[^1_75]: https://www.isca-archive.org/interspeech_2025/park25b_interspeech.pdf

[^1_76]: https://docs.openvino.ai/2024/notebooks/latent-consistency-models-image-generation-with-output.html

[^1_77]: https://arxiv.org/pdf/2303.01469.pdf

[^1_78]: https://openaccess.thecvf.com/content/ICCV2025/papers/Xu_Fast_Image_Super-Resolution_via_Consistency_Rectified_Flow_ICCV_2025_paper.pdf

[^1_79]: https://openreview.net/forum?id=duBCwjb68o

[^1_80]: https://thecho7.tistory.com/entry/논문-리뷰-Consistency-Models-설명

[^1_81]: https://arxiv.org/html/2507.03738v1

[^1_82]: https://arxiv.org/abs/2406.05768

[^1_83]: https://arxiv.org/html/2406.05768v5

[^1_84]: https://arxiv.org/html/2504.20900v1

[^1_85]: https://arxiv.org/html/2509.25127v1

[^1_86]: https://arxiv.org/abs/2410.14758

[^1_87]: https://arxiv.org/html/2412.11292v1

[^1_88]: https://arxiv.org/abs/2411.15084

[^1_89]: https://arxiv.org/pdf/2507.03738.pdf

[^1_90]: https://arxiv.org/html/2509.16499v2

[^1_91]: https://arxiv.org/html/2511.17583v1
