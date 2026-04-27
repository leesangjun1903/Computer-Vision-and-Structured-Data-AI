# Autoregressive Image Generation without Vector Quantization

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

본 논문의 핵심 주장은 다음과 같습니다:

> **"자기회귀(Autoregressive) 모델은 벡터 양자화(Vector Quantization)된 토큰과 반드시 결합될 필요가 없다."**

자기회귀 모델링의 본질은 **"이전 토큰을 기반으로 다음 토큰 예측"** 이며, 이는 값이 이산(discrete)이냐 연속(continuous)이냐와 독립적입니다. 필요한 것은 **퍼-토큰 확률 분포(per-token probability distribution)** 를 모델링하는 것이며, 이를 위해 반드시 범주형 분포(categorical distribution)를 사용할 필요가 없습니다.

### 주요 기여

| 기여 항목 | 설명 |
|-----------|------|
| **Diffusion Loss 제안** | 연속값 공간에서 per-token 확률 분포를 모델링하는 새로운 손실 함수 |
| **VQ 토크나이저 제거** | 벡터 양자화 없이도 고품질 이미지 생성 가능 |
| **MAR 프레임워크 통합** | 표준 AR 모델과 마스크 생성 모델을 일반화된 자기회귀 프레임워크로 통합 |
| **SOTA 성능 달성** | ImageNet 256×256에서 FID 1.55 달성 (with CFG) |
| **빠른 생성 속도** | 0.3초 미만/이미지로 FID < 2.0 달성 |

---

## 2. 논문의 상세 설명

### 2.1 해결하고자 하는 문제

기존 이미지 생성을 위한 자기회귀 모델들은 다음과 같은 **구조적 병목**을 가졌습니다:

1. **VQ 토크나이저 의존성**: VQ-VAE, VQ-GAN 등 이산 토크나이저가 필수적으로 요구됨
2. **양자화 손실**: 이산화 과정에서 정보 손실 발생 (rFID: VQ-16의 경우 5.87 vs KL-16의 1.43)
3. **훈련 불안정성**: 그래디언트 근사(Straight-Through Estimator 등)로 인한 훈련 어려움
4. **범주형 분포의 한계**: Softmax 기반 범주형 분포는 연속 공간의 복잡한 분포 표현에 부적합

### 2.2 제안하는 방법: Diffusion Loss

#### 2.2.1 이산 토큰 재고찰

기존 자기회귀 모델에서, 어휘 크기 $K$인 이산 토크나이저를 사용할 때:

$$p(x|z) = \text{softmax}(Wz)$$

여기서 $W \in \mathbb{R}^{K \times D}$는 $K$-way 분류기 행렬, $z \in \mathbb{R}^D$는 자기회귀 모델이 생성한 조건 벡터입니다.

확률 분포 모델링에 필요한 두 가지 필수 요소:
- **(i) 손실 함수(Loss Function)**: 추정 분포와 실제 분포 간의 차이 측정
- **(ii) 샘플러(Sampler)**: 분포 $x \sim p(x|z)$에서 샘플 추출

→ **이 두 가지는 반드시 이산 표현을 필요로 하지 않습니다.**

#### 2.2.2 Diffusion Loss 공식

연속값 벡터 $x \in \mathbb{R}^d$ (예측할 토큰)와 자기회귀 모델이 생성한 조건 벡터 $z \in \mathbb{R}^D$에 대해, 노이즈 제거 기준으로 정의된 손실 함수:

$$\mathcal{L}(z, x) = \mathbb{E}_{\varepsilon, t} \left[ \|\varepsilon - \varepsilon_\theta(x_t | t, z)\|^2 \right] \tag{1}$$

각 구성 요소:
- $\varepsilon \in \mathbb{R}^d$: 가우시안 노이즈 $\varepsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
- 노이즈가 추가된 토큰: $x_t = \sqrt{\bar{\alpha}_t} x + \sqrt{1 - \bar{\alpha}_t} \varepsilon$
- $\bar{\alpha}_t$: 코사인 형태의 노이즈 스케줄 (훈련 시 1000 스텝)
- $\varepsilon_\theta(x_t | t, z)$: 작은 MLP 기반 노이즈 추정 네트워크 (파라미터 $\theta$)
- $t$: 노이즈 스케줄 타임스텝

이 손실 함수는 **스코어 매칭(score matching)**의 형태로 해석 가능:

$$\mathcal{L} \text{ 는 } \nabla \log_x p(x|z) \text{ 와 관련된 손실 함수}$$

#### 2.2.3 샘플러 (역방향 확산)

추론 시 $p(x|z)$로부터 샘플링은 역방향 확산 절차로 수행:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \varepsilon_\theta(x_t | t, z) \right) + \sigma_t \delta$$

여기서 $\delta \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, $\sigma_t$는 타임스텝 $t$에서의 노이즈 수준. $x_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$에서 시작하여 $x_0 \sim p(x|z)$ 샘플 획득.

#### 2.2.4 온도(Temperature) 샘플링

이산 자기회귀 모델의 온도 $\tau$에 대응하는 연속 공간에서의 제어:

$$p(x|z)^{1/\tau} \text{ 에서 샘플링} \Leftrightarrow \text{score function: } \frac{1}{\tau} \nabla \log_x p(x|z)$$

실제 구현에서는 $\sigma_t \delta$를 $\tau$로 스케일링하여 구현.

#### 2.2.5 자기회귀 모델에 Diffusion Loss 적용

토큰 시퀀스 $\{x^1, x^2, \ldots, x^n\}$에 대한 자기회귀 생성:

$$p(x^1, \ldots, x^n) = \prod_{i=1}^{n} p(x^i | x^1, \ldots, x^{i-1}) \tag{2}$$

각 위치에서:
1. 조건 벡터 생성: $z^i = f(x^1, \ldots, x^{i-1})$ (Transformer 사용)
2. Diffusion Loss 적용: $\mathcal{L}(z^i, x^i)$
3. 그래디언트 역전파: $z^i \rightarrow f(\cdot)$ 업데이트

#### 2.2.6 마스크 자기회귀 모델 (MAR)

$K$개 단계로의 일반화된 자기회귀 공식:

$$p(x^1, \ldots, x^n) = p(X^1, \ldots, X^K) = \prod_{k}^{K} p(X^k | X^1, \ldots, X^{k-1}) \tag{3}$$

여기서 $X^k = \{x^i, x^{i+1}, \ldots, x^j\}$는 $k$번째 단계에서 예측할 토큰 집합이며 $\bigcup_k X^k = \{x^1, \ldots, x^n\}$.

### 2.3 모델 구조

```
┌─────────────────────────────────────────────────────────┐
│                    MAR 전체 구조                         │
│                                                         │
│  이미지 → KL-16 토크나이저 → 연속값 토큰 시퀀스         │
│              (16×16 = 256 tokens, d=16)                 │
│                      ↓                                  │
│  ┌──────────────────────────────────┐                   │
│  │         MAE 스타일 인코더         │                   │
│  │   (알려진 토큰 → 위치 임베딩)      │                   │
│  │   Transformer (16 blocks, L기준)  │                   │
│  └──────────────────────────────────┘                   │
│                      ↓                                  │
│  ┌──────────────────────────────────┐                   │
│  │         MAE 스타일 디코더         │                   │
│  │  (마스크 토큰 [m] + 위치 임베딩)   │                   │
│  │   Transformer (16 blocks, L기준)  │                   │
│  └──────────────────────────────────┘                   │
│                      ↓                                  │
│              조건 벡터 z 출력                            │
│                      ↓                                  │
│  ┌──────────────────────────────────┐                   │
│  │       Diffusion Loss (MLP)        │                   │
│  │   - 3 Residual Blocks (기본)      │                   │
│  │   - Width: 1024                   │                   │
│  │   - AdaLN으로 z 조건화            │                   │
│  │   - 파라미터: ~21M                │                   │
│  └──────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

**모델 크기 비교:**

| 모델 | Transformer 블록 수 | Width | 전체 파라미터 |
|------|---------------------|-------|---------------|
| MAR-B | 24 | 768 | 208M |
| MAR-L | 32 | 1024 | 479M |
| MAR-H | 40 | 1280 | 943M |

**주요 구현 세부 사항:**
- **토크나이저**: LDM의 KL-16 (연속값, stride 16)
- **노이즈 스케줄**: 코사인 형태, 훈련 1000스텝 / 추론 100스텝
- **마스킹 비율**: 훈련 시 $[0.7, 1.0]$ 균일 샘플링
- **추론 스텝**: 기본 64 자기회귀 스텝 (최고 성능 시 256 스텝)
- **CFG**: 훈련 시 10% 확률로 클래스 조건 제거
- **최적화**: AdamW, lr=8e-4, batch=2048, 800 epochs (최종)

### 2.4 성능 향상

#### ImageNet 256×256 주요 결과:

| 모델 | 파라미터 | FID↓ (w/o CFG) | FID↓ (w/ CFG) | 속도 |
|------|----------|-----------------|----------------|------|
| AR + CrossEnt (VQ) | ~400M | 19.58 | 4.92 | - |
| MAR + CrossEnt | ~400M | 8.79 | 3.69 | - |
| **MAR + Diff Loss (MAR-L)** | 479M | **2.60** | **1.78** | **<0.3s** |
| **MAR + Diff Loss (MAR-H)** | 943M | **2.35** | **1.55** | - |
| DiT-XL/2 | 675M | 9.62 | 2.27 | >1.0s |
| MAGVIT-v2 | 307M | 3.65 | 1.78 | - |
| MDTv2-XL/2 | 676M | 5.06 | 1.58 | - |

#### Diffusion Loss vs. Cross-entropy Loss 비교 (MAR 기본 설정):

| 손실 함수 | FID (w/o CFG) | IS (w/o CFG) | FID (w/ CFG) |
|-----------|---------------|---------------|---------------|
| Cross-Entropy | 8.79 | 146.1 | 3.69 |
| **Diffusion Loss** | **3.50** | **201.4** | **1.98** |

→ FID 약 **50~60% 상대적 개선**

### 2.5 한계점

논문에서 명시적으로 인정한 한계:

1. **아티팩트 발생**: 학습 데이터 제한(ImageNet)으로 인한 시각적 아티팩트 존재 (Figure 8 참조)
2. **토크나이저 의존성**: 기존 사전 학습된 토크나이저 품질에 성능이 제한됨 (더 나은 연속값 토크나이저 개발은 본 논문 범위 밖)
3. **평가 범위 제한**: 주로 ImageNet 벤치마크에서만 검증 (다양한 실제 시나리오 검증 필요)
4. **추론 복잡성**: 확산 샘플링으로 인해 단순 Softmax 샘플링보다 추론 비용이 높음
5. **고해상도 픽셀 공간 적용 한계**: 토크나이저 없이 픽셀 공간에 직접 적용 시 계산 비용 급증

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 토크나이저 유연성 (Flexibility)

**Diffusion Loss의 가장 중요한 일반화 특성**은 다양한 토크나이저와의 호환성입니다:

| 토크나이저 | 아키텍처 | rFID↓ | FID (w/o CFG)↓ | FID (w/ CFG)↓ |
|------------|----------|--------|-----------------|----------------|
| VQ-16 (이산, VQ 전 latent) | VQ-GAN | 5.87 | 7.82 | 3.64 |
| KL-16 (연속) | VAE-KL | 1.43 | 3.50 | 1.98 |
| KL-8 (stride 불일치) | VAE-KL | 1.20 | 4.33 | 2.05 |
| Consistency Decoder | - | 1.30 | 5.76 | 3.23 |
| KL-16† (ImageNet 학습) | VAE-KL | 1.22 | 2.85 | 1.97 |

**핵심 관찰**: Diffusion Loss는 stride가 불일치하는 토크나이저(예: KL-8을 2×2 그룹화)에도 적용 가능하여 **토크나이저 선택의 자유도**가 크게 높아집니다.

### 3.2 스케일링 법칙 (Scaling Law)

논문은 모델 크기에 따른 일관된 성능 향상을 보고합니다:

$$\text{MAR-B (208M): FID 2.31} \rightarrow \text{MAR-L (479M): FID 1.78} \rightarrow \text{MAR-H (943M): FID 1.55}$$

이는 GPT 등 언어 모델과 유사한 **스케일링 거동(scaling behavior)**을 보여주며, 더 큰 모델로의 확장이 유망함을 시사합니다.

### 3.3 도메인 일반화 가능성

#### 3.3.1 픽셀 공간 직접 적용 가능성

논문의 부록 D.1에서 보고된 실험:
- ImageNet 64×64에서 4×4 픽셀을 하나의 토큰으로 그룹화
- 토크나이저 없이 MAR-L + DiffLoss로 **FID 2.93** 달성
- **토크나이저 없는 이미지 생성의 가능성** 입증 (고해상도는 향후 연구 과제)

#### 3.3.2 타 도메인으로의 확장 가능성

논문이 언급하는 잠재적 적용 도메인:

```
1. 텍스트-이미지 생성 (Text-to-Image)
2. 텍스트-비디오 생성 (Text-to-Video)
3. 로봇 공학 (Action Policy Learning) - Diffusion Policy와 개념적 연관
4. 의료 영상 (Medical Imaging)
5. 오디오 생성 (Audio Generation)
6. 3D 포인트 클라우드 생성 (3D Generation)
7. 분자 구조 생성 (Molecular Generation)
```

이들 모두 **연속값 공간**에서의 생성 문제이며, Diffusion Loss의 구조가 직접 적용 가능합니다.

#### 3.3.3 일반화 성능을 뒷받침하는 수학적 근거

Diffusion Loss가 임의의 확률 분포를 표현할 수 있는 이론적 근거:

$$\mathcal{L}(z, x) = \mathbb{E}_{\varepsilon, t} \left[ \|\varepsilon - \varepsilon_\theta(x_t | t, z)\|^2 \right]$$

이는 스코어 매칭(score matching)과 동치:

$$\varepsilon_\theta \approx -\sqrt{1 - \bar{\alpha}_t} \nabla_{x_t} \log p(x_t | z)$$

즉, 이 손실 함수는 **가우시안 혼합(GIVT)과 달리 분포의 종류에 제약이 없으며**, 임의의 복잡한 다중 모드 분포도 근사 가능합니다.

GIVT의 가우시안 혼합 모델과의 비교:

$$p_{\text{GIVT}}(x|z) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k) \quad \text{(혼합 수 K 사전 결정)}$$

$$p_{\text{DiffLoss}}(x|z) = \text{임의의 분포 (혼합 수 제한 없음)}$$

#### 3.3.4 이산·연속 혼합 시퀀스 처리 가능성

Diffusion Loss는 **이산 토큰과 연속 토큰이 혼재하는 멀티모달 시퀀스**에도 원칙적으로 적용 가능합니다. 예를 들어:
- 텍스트 토큰 (이산): 기존 Cross-Entropy Loss
- 이미지/오디오 토큰 (연속): Diffusion Loss

이러한 하이브리드 접근은 **멀티모달 언어 모델**의 통합 프레임워크로 발전할 수 있습니다.

### 3.4 일반화 성능 관련 정성적 분석

**MAR의 학습-추론 일관성**: MAR은 훈련과 추론 모두에서 **완전 무작위 순서(fully randomized order)**를 사용하여 학습-추론 간 분포 불일치를 최소화합니다. 이는 MAGE/MaskGIT의 on-the-fly 신뢰도 기반 순서 결정과 달리 일반화에 유리합니다.

**양방향 어텐션의 일반화 이점**: 완전 어텐션(full attention)은 모든 토큰 간 통신을 허용하여 인과 어텐션 대비 더 풍부한 표현을 학습합니다:

$$\text{FID 개선: 인과(causal) AR 13.07} \rightarrow \text{양방향(bidirectional) MAR 3.43 (w/o CFG)}$$

---

## 4. 앞으로의 연구에 미치는 영향과 연구 시 고려할 점

### 4.1 연구에 미치는 영향

#### 4.1.1 패러다임 전환

이 논문은 다음과 같은 **근본적 패러다임 전환**을 제시합니다:

```
기존 패러다임:
이미지 → VQ 이산화 → 범주형 AR 생성

새로운 패러다임:
이미지 → KL 연속 인코딩 → 연속값 AR (Diffusion Loss) 생성
```

이는 NLP와 컴퓨터 비전의 경계를 허물고, **통합 멀티모달 생성 모델** 연구를 촉진합니다.

#### 4.1.2 구체적 연구 파급 효과

1. **토크나이저 연구 방향 전환**: VQ 기반 이산 토크나이저 개발 → 연속값 VAE/연속 인코더 품질 향상 연구로 초점 이동

2. **멀티모달 AR 모델**: 텍스트-이미지, 텍스트-오디오-비디오 등 연속값 멀티모달 시퀀스 생성 연구 활성화

3. **손실 함수 혁신**: 전통적인 MSE, Cross-Entropy 대신 **생성적 확산 기반 손실 함수**를 다양한 학습 문제에 적용하는 연구

4. **자기회귀 모델의 확장**: 언어 모델(LLM)과 이미지 생성 모델의 통합 → 향후 GPT-4V, Gemini 계열의 연속값 이미지 생성 통합에 기여

5. **경량화 연구**: 작은 MLP(~21M)로 복잡한 분포를 모델링하는 Diffusion Loss의 효율성 → 경량 생성 모델 연구에 영감 제공

### 4.2 관련 최신 연구 비교 분석 (2020년 이후)

| 연구 | 연도 | 방법론 | ImageNet 256 FID (w/ CFG) | 주요 특징 |
|------|------|--------|---------------------------|-----------|
| DDPM (Ho et al.) | 2020 | 픽셀 공간 확산 | - | 확산 모델 기초 |
| VQ-GAN (Esser et al.) | 2021 | VQ + Transformer AR | - | 이산 토큰 AR 기반 |
| ADM (Dhariwal et al.) | 2021 | 픽셀 확산 + CFG | 4.59 | 확산 모델 SOTA 기점 |
| MaskGIT (Chang et al.) | 2022 | 마스크 생성 (이산) | - | 병렬 이산 토큰 생성 |
| LDM (Rombach et al.) | 2022 | 잠재 공간 확산 | 3.60 | 연속값 잠재 확산 |
| MAE (He et al.) | 2022 | 마스크 자동인코더 | - | 표현 학습 혁신 |
| MAGE (Li et al.) | 2023 | 마스크 생성 + 표현 | - | 통합 프레임워크 |
| DiT (Peebles et al.) | 2023 | Transformer 기반 확산 | 2.27 | 확산+Transformer |
| GIVT (Tschannen et al.) | 2023 | 가우시안 혼합 AR | 3.35 | 연속값 AR (제한적) |
| MAGVIT-v2 (Yu et al.) | 2024 | 개선된 VQ + 마스크 | 1.78 | VQ 품질 개선 |
| MDTv2 (Gao et al.) | 2023 | 마스크 확산 Transformer | 1.58 | 확산+마스크 |
| **MAR (본 논문)** | **2024** | **연속 AR + Diffusion Loss** | **1.55** | **VQ 제거, 빠른 속도** |

#### GIVT와의 핵심 비교:

| 비교 항목 | GIVT | MAR (본 논문) |
|-----------|------|---------------|
| 분포 모델링 | 가우시안 혼합 (K 사전 결정) | 임의 분포 (확산 기반) |
| 표현력 | 제한적 (K 고정) | 이론상 무제한 |
| FID (w/ CFG) | 3.35 | 1.55~1.98 |
| 생성 속도 | 빠름 | 빠름 (<0.3s) |

### 4.3 향후 연구 시 고려할 점

#### 4.3.1 기술적 고려사항

**① 토크나이저 품질의 중요성**

$$\text{생성 FID} \leq \text{재구성 FID(rFID)} + \text{생성 모델 오차}$$

연속값 토크나이저의 재구성 품질이 상한선을 결정합니다. 따라서:
- **더 나은 연속값 인코더** 연구 필요 (예: 더 낮은 rFID를 가진 KL-VAE)
- **도메인 특화 토크나이저** 개발 (의료, 위성 이미지 등)

**② 확산 스텝과 속도 트레이드오프**

추론 시 확산 스텝 수에 따른 품질-속도 트레이드오프:

$$\text{스텝 수} \uparrow \Rightarrow \text{품질} \uparrow, \text{속도} \downarrow$$

향후 연구에서 **일관성 모델(Consistency Models)** 또는 **플로우 매칭(Flow Matching)**을 Diffusion Loss 내부에 적용하면 1~2 스텝으로 가능해질 수 있습니다.

**③ 자기회귀 스텝과의 이중 계층 최적화**

MAR은 두 가지 스텝 수를 동시에 최적화해야 합니다:
- 자기회귀 스텝 수 (토큰 생성 순서)
- 확산 스텝 수 (각 토큰 내부 생성)

이 두 수준의 최적화 상호작용에 대한 심층 연구가 필요합니다.

**④ 스케일링 법칙 추가 분석**

논문이 언급하는 스케일링 거동을 더 체계적으로 분석:

$$\text{FID} \propto N^{-\alpha}$$

여기서 $N$은 모델 파라미터 수, $\alpha$는 스케일링 지수. 언어 모델의 Chinchilla 법칙에 대응하는 **이미지 생성 스케일링 법칙** 도출이 중요합니다.

**⑤ 고해상도 이미지 생성으로의 확장**

현재 주로 256×256, 512×512에서 검증됨. 1024×1024 이상 고해상도에서:
- 시퀀스 길이 증가로 인한 계산 복잡도 ( $O(n^2)$ ) 문제
- **계층적 MAR** 또는 **패치 계층 Diffusion Loss** 도입 검토 필요

#### 4.3.2 일반화 관련 고려사항

**⑥ 텍스트 조건 생성으로의 확장**

현재 클래스 조건(class-conditional) 생성만 검증됨. 텍스트 조건 생성을 위해:
- CLIP, T5 등 텍스트 인코더와의 통합 방법 연구
- 텍스트-이미지 정렬을 위한 Diffusion Loss 확장 ( $p(x|z, \text{text})$ )

**⑦ 비디오 생성으로의 확장**

시간 축을 따른 토큰 시퀀스로의 자연스러운 확장 가능:

$$p(v^1, v^2, \ldots, v^T) = \prod_{t=1}^{T} p(v^t | v^1, \ldots, v^{t-1})$$

각 프레임 $v^t$를 공간적 토큰 집합으로 처리하는 계층적 접근이 필요합니다.

**⑧ 3D 및 기타 비유클리드 공간**

3D 포인트 클라우드, 분자 구조 등 비유클리드 공간에서의 Diffusion Loss 적용을 위해:
- 리만 기하학 기반 확산 과정 (Riemannian Diffusion) 적용 고려
- SO(3), SE(3) 군 위에서의 확산 과정 연구

**⑨ 데이터 편향 및 윤리적 고려**

- 훈련 데이터 편향이 생성 결과에 반영될 위험
- 딥페이크/허위정보 생성에 악용 가능성
- **생성 이미지 워터마킹** 및 감지 기술 병행 연구 필요

#### 4.3.3 아키텍처 혁신 고려사항

**⑩ 더 강력한 디노이징 네트워크**

현재 소형 MLP(~21M)를 사용하지만, 더 복잡한 분포를 위해:
- Transformer 기반 디노이징 네트워크 탐색
- 확산 스텝에 따른 적응형 네트워크 크기 조절

**⑪ 플로우 매칭(Flow Matching)과의 통합**

최근 각광받는 Flow Matching 기법을 Diffusion Loss에 통합하면 더 빠른 수렴과 샘플링이 가능:

$$\mathcal{L}_{\text{FM}}(z, x) = \mathbb{E}_{t, x_0, x_1} \left[ \|v_\theta(x_t | t, z) - (x_1 - x_0)\|^2 \right]$$

이는 추론 시 단 수십 스텝만으로도 고품질 생성이 가능하여 현재의 100스텝 제약을 완화할 수 있습니다.

---

## 참고 자료 및 출처

**주요 논문 (본 논문 및 참조 논문):**

1. **Li, T., Tian, Y., Li, H., Deng, M., & He, K. (2024).** "Autoregressive Image Generation without Vector Quantization." *NeurIPS 2024.* arXiv:2406.11838v3. *(본 분석의 주요 대상 논문)*

2. **Ho, J., Jain, A., & Abbeel, P. (2020).** "Denoising Diffusion Probabilistic Models." *NeurIPS 2020.*

3. **Nichol, A. Q., & Dhariwal, P. (2021).** "Improved Denoising Diffusion Probabilistic Models." *ICML 2021.*

4. **Dhariwal, P., & Nichol, A. (2021).** "Diffusion Models Beat GANs on Image Synthesis." *NeurIPS 2021.*

5. **Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022).** "High-Resolution Image Synthesis with Latent Diffusion Models." *CVPR 2022.*

6. **Peebles, W., & Xie, S. (2023).** "Scalable Diffusion Models with Transformers." *ICCV 2023.*

7. **Chang, H., Zhang, H., Jiang, L., Liu, C., & Freeman, W. T. (2022).** "MaskGIT: Masked Generative Image Transformer." *CVPR 2022.*

8. **He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022).** "Masked Autoencoders Are Scalable Vision Learners." *CVPR 2022.*

9. **Li, T., Chang, H., Mishra, S., Zhang, H., Katabi, D., & Krishnan, D. (2023).** "MAGE: Masked Generative Encoder to Unify Representation Learning and Image Synthesis." *CVPR 2023.*

10. **Tschannen, M., Eastwood, C., & Mentzer, F. (2023).** "GIVT: Generative Infinite-Vocabulary Transformers." arXiv:2312.02116.

11. **Esser, P., Rombach, R., & Ommer, B. (2021).** "Taming Transformers for High-Resolution Image Synthesis." *CVPR 2021.*

12. **Yu, L., Lezama, J., et al. (2024).** "Language Model Beats Diffusion–Tokenizer is Key to Visual Generation (MAGVIT-v2)." *ICLR 2024.*

13. **Gao, S., Zhou, P., Cheng, M.-M., & Yan, S. (2023).** "Masked Diffusion Transformer is a Strong Image Synthesizer (MDTv2)." *ICCV 2023.*

14. **Song, Y., & Ermon, S. (2019).** "Generative Modeling by Estimating Gradients of the Data Distribution." *NeurIPS 2019.*

15. **Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021).** "Score-Based Generative Modeling through Stochastic Differential Equations." *ICLR 2021.*

16. **van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017).** "Neural Discrete Representation Learning (VQ-VAE)." *NeurIPS 2017.*

17. **Vaswani, A., et al. (2017).** "Attention Is All You Need." *NeurIPS 2017.*

18. **Brown, T., et al. (2020).** "Language Models are Few-Shot Learners (GPT-3)." *NeurIPS 2020.*

**공개 코드:**
- MAR 공식 구현: https://github.com/LTH14/mar
