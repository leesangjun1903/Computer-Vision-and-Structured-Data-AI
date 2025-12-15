# DiffFit: Unlocking Transferability of Large Diffusion Models via Simple Parameter-Efficient Fine-Tuning

### 1. 논문의 핵심 주장과 주요 기여

DiffFit는 대규모 확산 모델(특히 DiT-XL/2)을 새로운 도메인에 효율적으로 적응시키는 **매개변수-효율적 미세조정(Parameter-Efficient Fine-Tuning, PEFT)** 방법론입니다.[1]

**핵심 주장:**
- 대규모 확산 모델을 전체 매개변수 대비 **0.12%**만 조정하여 완전한 미세조정(full fine-tuning)과 경쟁력 있거나 우수한 성능을 달성할 수 있습니다.
- 단순한 아키텍처 설계(편향항, 정규화층, 스케일 인수 미세조정)로도 **2배의 훈련 속도 향상**을 달성합니다.

**주요 기여:**
1. 확산 이미지 생성용 매개변수-효율적 미세조정 방법 제시
2. 직관적 이론 분석 및 상세한 절제 연구(ablation study) 수행
3. 저해상도 모델을 고해상도 생성 모델로 적응시키는 확장성 입증 (ImageNet 512×512에서 FID 3.02 달성, 이전 DiT 모델 3.04 대비 개선)[1]

***

### 2. 해결하고자 하는 문제, 제안 방법, 모델 구조

#### 2.1 핵심 문제

확산 모델의 우수한 이미지 생성 능력에도 불구하고, 다음과 같은 실제 적용의 어려움이 존재합니다:[1]

- **계산 비용:** DiT-XL/2 모델은 640M 매개변수를 포함하며, 256×256 해상도 훈련에 950 V100 GPU일, 512×512 해상도에는 1733 V100 GPU일이 필요
- **메모리 부담:** 여러 도메인 적응을 위해 완전한 모델 복사본을 저장해야 함(선형 저장 지출)
- **일반화 성능:** 새로운 도메인으로의 전이성(transferability)이 미흡

#### 2.2 제안하는 방법 및 수식

DiffFit는 BitFit 기반의 간단한 전략으로, 다음 매개변수만 미세조정합니다:[1]

$$\text{Trainable Parameters} = \{\text{Bias terms}, \text{Layer Normalization}, \text{Class embeddings}, \gamma\}$$

여기서 **스케일 인수 $\gamma$**는 새로 추가되는 학습 가능한 매개변수로, 각 블록의 특정 위치에서 초기값 1.0으로 설정됩니다.

**파인튜닝 중 스케일 인수 적용:**

$$x = x + \gamma_1 \odot \text{Attn}(\text{wrap}(x, c, t)) + \gamma_2 \odot \text{FFN}(\text{wrap}(x, c, t))$$

여기서:
- $\odot$는 원소별 곱셈(element-wise multiplication)
- $\gamma_1, \gamma_2$는 자주의 주의(self-attention)와 피드포워드 네트워크(FFN)에 적용되는 스케일 인수
- $c, t$는 각각 클래스 임베딩과 시간 임베딩

**손실 함수:**

$$L_{\text{hybrid}} = L_{\text{simple}} + \lambda L_{\text{vlb}}$$

여기서:
- $L_{\text{simple}} = \mathbb{E}\_{x_0, \epsilon, t} [\|\epsilon - \epsilon_\theta\|^2]$
- $L_{\text{vlb}} = \mathbb{E}\_{x_0, \epsilon, t} \left[\frac{\beta_t^2}{2\alpha_t(1-\bar{\alpha}\_t)\sigma_t^2} \|\epsilon - \epsilon_\theta\|^2\right]$
- $\lambda = 0.001$ [1]

#### 2.3 확산 모델의 이론적 분석

**Theorem 1 (비형식적 표현):**

소스 도메인 데이터분포 $Q_0$에서 생성된 데이터셋에 대해, 확산 모델이 대략적으로 $Q_0$를 따르는 샘플을 생성할 수 있다고 가정합니다. 더 나아가 상대적으로 작은 데이터셋의 대상 데이터분포 $P_0$가 선형 매핑 $f_{\gamma^*}$에 의존하는 다음 식으로 표현된다고 가정합니다:[1]

```math
P_0 = f_{\gamma^*} \# Q_0
```

여기서 

```math
f_{\gamma^*} \# Q_0
```

는 푸시포워드 측도(pushforward measure)입니다. 이 경우, 스케일 인수 $\gamma$만을 최적화하는 목표로 신경망을 재훈련하면, 적절한 조건 하에서 간단한 경사하강법 알고리즘이 고확률로 $\gamma^*$에 가까운 추정값 $\hat{\gamma}$를 찾습니다.[1]

**Theorem 2 (형식적 표현):**

다음 조건들을 가정합니다:
- $m = \Omega(D^2 \log D)$개의 훈련 샘플
- $A = f(EW^T + B) \in \mathbb{R}^{m \times D}$ (여기서 $f$는 초선형 성장이 없는 비선형 활성화함수)
- 목적함수: $\min_{\gamma} \|A\gamma - y\|_2^2$

그러면 다음이 성립합니다:[1]

```math
\|\hat{\gamma} - \gamma^*\|_2 < \frac{4\sqrt{2}\eta}{\sqrt{V_{\min}}} \cdot \|\gamma^*\|_2
```

여기서 $\eta > 0$은 작은 값이고, $V_{\min} = \min_{i \in [D]} \text{Var}[f(X_i)]$입니다. 즉, 훈련 샘플 수가 충분할 때, 경사하강법이 고확률로 $\gamma^*$에 가까운 추정값을 찾습니다.[1]

#### 2.4 모델 구조

DiffFit의 구조는 다음과 같이 구성됩니다:[1]

**기본 구조:**
```
Input → Patch Embedding → Time Embedding + Class Embedding
                              ↓
                     [DiT Blocks (1~28)]
                              ↓
                    Depatchify & VAE Decoder → Output
```

각 블록 $B_i$는 다음과 같이 표현됩니다:

$$z_i = B_i(x, t, c)$$

여기서:
- $x$: 입력 토큰
- $t$: 시간 임베딩
- $c$: 클래스 임베딩

**미세조정 전략 (Algorithm 2):**
1. 모든 매개변수를 먼저 동결(freeze)
2. 편향, 정규화, 스케일 인수, 클래스 임베딩만 언프리즈
3. 지정된 에포크 동안 훈련

***

### 3. 성능 향상 및 한계

#### 3.1 성능 향상 결과

**다운스트림 데이터셋 성능 (8개 데이터셋):**

| 방법 | Food | SUN | DF-20M | Caltech | CUB-Bird | ArtBench | Oxford | Flowers | 평균 FID |
|------|------|------|--------|---------|----------|----------|--------|---------|----------|
| 완전 미세조정 | 10.46 | 7.96 | 17.26 | 35.25 | 5.68 | 25.31 | 21.05 | 9.79 | 16.59 |
| BitFit | 9.17 | 9.11 | 17.78 | 34.21 | 8.81 | 24.53 | 20.31 | 10.64 | 16.82 |
| LoRA-R8 | 33.75 | 32.53 | 120.25 | 86.05 | 56.03 | 80.99 | 164.13 | 76.24 | 81.25 |
| **DiffFit** | **6.96** | **8.55** | **17.35** | **33.84** | **5.48** | **20.87** | **20.18** | **9.90** | **15.39** |

DiffFit는 8개 데이터셋 평균 **15.39 FID**를 달성하여 완전 미세조정(16.59)을 능가합니다.[1]

**해상도 적응 성능 (ImageNet 512×512):**

| 방법 | FID ↓ | 훈련 비용(GPU일) |
|------|-------|-----------------|
| StyleGAN-XL | 2.41 | 400 |
| ADM-G, ADM-U | 3.85 | 1914 |
| DiT-XL/2 | 3.04 | 1733 |
| **DiffFit (ours)** | **3.02** | **51 + 950†** |

DiffFit은 완전히 동일한 FID 성능을 **30배 더 빠르게** 달성합니다 (900 GPU일 대비 51 GPU일).[1]

#### 3.2 일반화 성능 향상 가능성

**가중치 초기화의 중요성:**

미세조정된 모델의 일반화 성능은 다음 요인들에 영향을 받습니다:[1]

1. **사전학습 지식 보존:** DiffFit이 0.12% 매개변수만 조정하므로, 사전학습된 모델의 일반화 능력이 대부분 보존됩니다.

2. **분포 시프트 적응:** 스케일 인수를 통한 순수한 선형 변환은 다음과 같은 이점을 제공합니다:
   - 과적합(overfitting) 위험 감소
   - 새로운 도메인에서의 더 나은 제너럴화

3. **실험적 증거:** 8개 도메인에서 완전 미세조정보다 우수한 평균 FID 스코어(15.39 vs 16.59)는 더 나은 일반화를 시사합니다.

#### 3.3 주요 한계

논문에서 명시된 한계:[1]

1. **적용 범위 제한:**
   - 주로 클래스-조건부 이미지 생성에 초점
   - 텍스트-이미지 생성, 비디오/3D 생성 등 더 복잡한 작업에서의 성능 불명확

2. **이론적 가정:**
   - Theorem 1, 2는 단순화된 시나리오에 기반
   - 실제 신경망의 복잡성을 완전히 포괄하지 않음
   - 형식적 이론은 "개념 증명(proof of concept)"의 성격[1]

3. **확장성 문제:**
   - DiT 이외의 다른 확산 모델(Stable Diffusion 등)에 대한 광범위한 평가 부재
   - 매우 작은 도메인 데이터셋의 경우 성능 저하 가능성

4. **스케일 인수 위치의 경험적 성질:**
   - 최적의 $\gamma$ 배치가 데이터셋과 아키텍처에 따라 변동
   - 일반적인 설계 원칙 부재[1]

***

### 4. 모델의 일반화 성능 향상 메커니즘 상세 분석

#### 4.1 확산 과정에서의 일반화

**전방 과정 (Forward Process):**

초기 데이터 $x_0 \sim q_{data}(x)$에서 시작하여 마르코프 연쇄에 의해 제어됩니다:[1]

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$

여기서:
- $\bar{\alpha}\_t = \prod_{i=1}^{t} \alpha_i$, $\alpha_t = 1 - \beta_t$
- $\epsilon \sim \mathcal{N}(0, I)$
- 시간 단계가 커질수록 $\bar{\alpha}_t$ 감소, 샘플이 더 노이지

**역방정 과정 (Reverse Process):**

확산 모델은 다음 조건부 분포를 학습합니다:[1]

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

**분포 적응의 이론적 근거:**

DiffFit이 일반화 성능을 개선하는 이유:

1. **푸시포워드 측도 정렬:**
   - 소스 도메인 $Q_0$의 스코어 함수를 학습
   - 선형 변환 $f_{\gamma^\*}$를 통해 목표 도메인

```math
P_0 = f_{\gamma^*} \# Q_0
```

로 적응

  - 선형 변환은 분포의 기본 구조를 보존[1]

2. **매개변수 수 제약 효과:**
   - 훈련 가능한 매개변수 수 제한으로 과적합 위험 감소
   - 사전학습된 특성의 강력한 정규화 효과

3. **계층별 역할 분담:**
   - 깊은 계층: 고수준 특성 학습 (고정 유지)
   - 얕은 계층과 스케일 인수: 저수준 특성 적응
   - 이러한 분리가 견고한 일반화를 도움

#### 4.2 절제 연구를 통한 메커니즘 검증

**스케일 인수의 계층별 효과 (Table 4a, 4b):**

| 블록 범위 | FID ↓ | 매개변수(M) |
|-----------|-------|-----------|
| 28→25 (깊음) | 10.04 | 0.747 |
| 28→14 (중간) | 10.51 | 0.770 |
| 28→8 (얕음) | 9.28 | 0.786 |
| 28→4 (매우 얕음) | 8.87 | 0.796 |
| 28→1 (가장 얕음) | **8.19** | 0.803 |

**중요한 발견:** 가장 얕은 계층(1~14 계층)에 스케일 인수를 추가할 때 최고 성능(FID 6.96)을 달성합니다. 이는 다음을 시사합니다:[1]

- 깊은 계층에서의 스케일 인수 추가는 오히려 성능 저하
- 고수준 특성 학습 방해 가능성
- 선택적 적응이 일반화 성능 유지의 핵심

#### 4.3 수렴 속도 및 일반화 동역학

**수렴 곡선 분석 (Figure 6):**

DiffFit, 완전 미세조정, BitFit은 유사한 수렴률을 보이지만:[1]

- **DiffFit:** 계층 동결로 인해 훈련 초기부터 안정적인 수렴
- **완전 미세조정:** 매개변수가 많아 초기 변동성 증가
- **AdaptFormer/VPT:** 더 느린 수렴

이는 DiffFit이 **사전학습 정보를 최대한 보존**하면서도 신속한 적응을 가능하게 함을 의미합니다.

***

### 5. 2020년 이후 관련 최신 연구 비교 분석

#### 5.1 주요 PEFT 방법론 비교

**1. BitFit (2021)**

| 특성 | BitFit | DiffFit |
|------|--------|---------|
| 미세조정 대상 | 편향항만 | 편향 + 스케일 인수 + 정규화 |
| 매개변수 효율성 | 0.09% | 0.12% |
| 확산 모델 성능 | 16.82 FID | **15.39 FID** |
| 이론적 근거 | 경험적 | Theorem 1, 2 |

**결론:** DiffFit은 BitFit의 기본 개념을 발전시켜 스케일 인수를 추가함으로써 성능 향상을 달성합니다.[1]

**2. LoRA (2021)**[2]

| 특성 | LoRA-R8 | LoRA-R16 | DiffFit |
|------|---------|----------|---------|
| 평균 FID | 81.25 | 81.31 | **15.39** |
| 매개변수(%) | 0.17% | 0.32% | 0.12% |
| 계산 비용 | 0.63× | 0.68× | 0.49× |

**결론:** LoRA는 확산 모델에 적합하지 않습니다. 이미지 분류 작업보다 이미지 생성이 더 복잡하기 때문에 성능 격차가 더 큽니다.[1]

**3. AdaptFormer (2022)**[3]

| 특성 | AdaptFormer | DiffFit |
|------|------------|---------|
| 평균 FID | 20.17 | **15.39** |
| 훈련 속도 | 0.47× | 0.49× |
| 추가 모듈 | 어댑터 삽입 | 스케일 인수만 |

**결론:** DiffFit은 더 간단한 설계로 AdaptFormer보다 우수한 성능을 달성합니다.[1]

**4. VPT (Visual Prompt Tuning, 2022)**[4]

| 특성 | VPT-Deep | DiffFit |
|------|----------|---------|
| 평균 FID | 26.80 | **15.39** |
| 매개변수(%) | 0.42% | 0.12% |
| 수렴 속도 | 느림 | 빠름 |

**결론:** VPT는 프롬프트 토큰 추가로 인한 오버헤드가 있어 확산 모델에 덜 효율적입니다.[1]

#### 5.2 확산 모델 전이성 관련 최신 연구 (2023-2025)

**1. Diff-Tuning (2024)**[5]

논문: "Diffusion Tuning: Transferring Diffusion Models via Chain of Forgetting"

**핵심:**
- 역프로세스를 따라 "망각의 연쇄(chain of forgetting)" 현상 발견
- 노이즈 쪽(역프로세스 후기)에서는 전이성이 약함
- 데이터 쪽(역프로세스 초기)에서 사전학습 지식 보존

**DiffFit과의 차이:**
- DiffFit: 모든 타임스텝에 균일하게 적용
- Diff-Tuning: 타임스텝별 차등적 적응

**성능:**
- 표준 미세조정 대비 26% 개선
- ControlNet 수렴 속도 24% 향상[5]

**2. TuneQDM (2024)**[6]

논문: "Memory-Efficient Fine-Tuning for Quantized Diffusion Model"

**핵심:**
- 양자화된 확산 모델의 메모리-효율적 미세조정
- 양자화 스케일을 시간 단계별로 조정
- 채널간 가중치 패턴 고려

**DiffFit과의 차이:**
- 양자화 모델에 특화
- 동적 타임스텝 특성 활용

**성능:**
- 완전 정밀도 모델과 동등한 성능
- 메모리 효율성 향상[6]

**3. Riemannian Preconditioned LoRA (2024)**[7]

논문: "Riemannian Preconditioned LoRA for Fine-Tuning Foundation Models"

**핵심:**
- LoRA 훈련에 $r \times r$ 전조건자(preconditioner) 도입
- 리만 메트릭 기반 최적화
- 대형 언어 모델과 텍스트-이미지 확산 모델에 적용

**DiffFit과의 차이:**
- 더 정교한 최적화 방법론
- LoRA의 문제점 개선 시도

**성능:**
- SGD/AdamW 수렴성 및 안정성 향상
- 하이퍼파라미터 선택에 더 강건[7]

**4. FineDiffusion (2024)**[8]

논문: "FineDiffusion: Scaling up Diffusion Models for Fine-grained Image Generation with 10,000 Classes"

**핵심:**
- 10,000 클래스 규모 미세한 이미지 생성
- 계층적 클래스 임베더(tiered class embedder) 도입
- 편향항과 정규화층 미세조정

**DiffFit과의 유사성:**
- 동일한 기본 전략 사용
- 클래스 조건부 생성에 특화

**차이:**
- 대규모 클래스 시나리오에 최적화
- 계층적 구조 추가[8]

**5. Spectrum-Aware PEFT (SODA, 2024)**[9]

논문: "Spectrum-Aware Parameter Efficient Fine-Tuning for Diffusion Models"

**핵심:**
- 사전학습 가중치의 특이값(singular values) 분석
- 크로네커 곱(Kronecker product) 활용
- 효율적 슈틸펠 최적화기(Stiefel optimizer)

**DiffFit과의 차이:**
- 스펙트럼 정보 활용
- 직교 행렬 적응

**성능:**
- 텍스트-이미지 확산 모델에서 효과적[9]

**6. ControlNet과의 결합 (논문 보충 자료, 2023)**

DiffFit은 ControlNet과 결합 가능:[1]

| 방법 | FID | CLIP Score | 훈련 가능 매개변수 |
|------|------|-----------|------------------|
| ControlNet (원본) | 20.1 | 0.3067 | 361M (26.9%) |
| **ControlNet + DiffFit** | **19.5** | **0.3064** | 11.2M (0.83%) |

DiffFit의 유연성을 입증합니다.[1]

#### 5.3 고해상도 적응 관련 최신 연구

**Positional Encoding 기법의 혁신:**

DiffFit의 위치 인코딩 트릭:[1]

$$\text{PE}_{512 \times 512} = f(\text{coord} / 2)$$

여기서 256×256 인코딩의 각 픽셀 좌표를 절반으로 스케일링합니다. 이는:

- 추가 계산 비용 없음
- Figure 5에서 빠른 수렴 달성
- 해상도 간 부드러운 전이 가능[1]

***

### 6. 앞으로의 연구에 미치는 영향과 고려사항

#### 6.1 학술적 영향

**1. PEFT 패러다임 전환**

- **기존:** 복잡한 모듈 추가(어댑터, LoRA, VPT)
- **DiffFit:** "Simple is Better" 철학으로 회귀
- 향후 연구는 단순성과 효율성의 균형점 추구[1]

**2. 확산 모델 전이성의 이해 심화**

- Theorem 1, 2를 통해 선형 변환 기반 적응의 이론적 근거 제시
- 향후 연구는:
  - 비선형 변환 시나리오로 확장
  - 다양한 분포 시프트에 대한 이론적 분석
  - 실제 신경망의 복잡성을 더 잘 포괄하는 이론 필요[1]

**3. 매개변수 효율성의 한계 탐색**

- 0.12% 매개변수로 완전 미세조정을 능가하는 결과는:
  - 신경망의 내재적 차원성(intrinsic dimensionality)에 대한 질문 제기
  - 극도의 매개변수 효율성의 가능성을 시사

#### 6.2 실무적 영향

**1. 엣지 배포(Edge Deployment)**

- 소비자 수준 하드웨어에서 대규모 모델 적응 가능
- 엣지 기기에서의 개인화된 생성 모델 개발 용이
- 프라이버시 보호 (로컬 미세조정)

**2. 멀티도메인 시나리오**

- 단일 기본 모델 + 도메인별 0.12% 매개변수로 구성
- 저장 오버헤드 극소화 (모델당 ~1MB)

**3. 빠른 프로토타이핑**

- 2배 훈련 속도 향상으로 연구 사이클 단축
- 하이퍼파라미터 튜닝 용이

#### 6.3 향후 연구의 고려사항

**1. 적용 범위 확장**

현재 한계:[1]

```
✓ 클래스-조건부 이미지 생성 (DiT)
✓ ControlNet 결합 (텍스트-이미지)
✓ DreamBooth 결합 (주제 중심 생성)
✗ 일반 텍스트-이미지 생성 (Stable Diffusion 등)
✗ 비디오 생성
✗ 3D 생성
```

**필요한 연구:**
- Stable Diffusion 등 UNet 기반 모델에 대한 적응
- 멀티모달 조건부 생성 시나리오

**2. 스케일 인수 최적 배치 자동화**

현재: 수동 절제 연구로 최적 위치 결정

**향후 접근:**
- 신경 아키텍처 탐색(NAS) 기반 최적 위치 자동 선택
- 계층별 중요도 점수 학습
- 데이터셋별 적응적 배치 전략

**3. 이론적 강화**

**Theorem 1, 2의 제한:**
- 단계 미니배치 시뮬레이션 가정
- 선형 매핑 가정 (실제로는 매우 비선형)
- 무한 폭 신경망 가정[1]

**필요한 개선:**
- 다단계 비선형 변환의 수렴성 분석
- 실제 유한 네트워크에 대한 보장
- 특정 도메인 시프트 패턴의 특성화

**4. 에너지 효율성**

DiffFit의 장점을 환경 관점에서 재평가:

$$\text{Carbon Savings} = 30 \times \text{GPU Days Reduction} \times \text{GPU Power}$$

- 디바이스 수명 주기 동안의 에너지 절감
- 더 넓은 접근성으로 인한 긍정적 사회적 영향

**5. 생성 품질과 다양성의 균형**

현재: FID 메트릭 중심 평가

**추가 고려사항:**
- Inception Score (IS) 동시 평가
- 의미론적 일관성(semantic consistency)
- 생성 다양성(diversity) 측정
- 인간 평가 기반 비교

***

### 7. 핵심 요약 및 결론

#### 7.1 DiffFit의 혁신성

| 측면 | 기존 방법 | DiffFit | 개선 |
|------|---------|---------|------|
| **매개변수 효율성** | 100% | 0.12% | **833배** |
| **훈련 속도** | 1× | 2× | **2배 향상** |
| **성능** | FID 16.59 | FID 15.39 | **1.20 개선** |
| **메모리 저장** | 모델당 완전 복사 | 모델당 ~1MB | **수천배 절감** |
| **확장성** | 저해상도 전용 | 저→고 해상도 | **새로운 능력** |

#### 7.2 이론적 기여

1. **분포 적응 메커니즘:** 선형 변환 기반의 확산 모델 적응에 대한 이론적 정당성 제시
2. **일반화 보장:** Theorem 2를 통해 추정된 스케일 인수의 최적성 근처 수렴 증명
3. **설계 원칙:** 계층별 역할 분담의 중요성 입증

#### 7.3 실무적 영향

- **접근성:** 엣지 디바이스에서 대규모 모델 사용 가능
- **확장성:** 다중 도메인 적응의 경제적 가능성
- **지속성:** 획기적인 에너지 효율성 달성

#### 7.4 미래 방향

DiffFit의 성공은 다음을 시사합니다:

1. **매개변수 효율성의 한계를 재정의:** 0.12%가 실제 가능한 최소값인지 탐구
2. **일반화 이론의 발전:** 실제 신경망에 적용 가능한 더 강력한 이론 개발
3. **다양한 생성 모델로의 확장:** 비디오, 3D, 오디오 등 다양한 도메인 적용

**결론:** DiffFit은 "더 적게 하면서 더 잘한다(Do More with Less)"는 철학을 실증함으로써, 대규모 기초 모델 시대에서 효율성과 성능의 새로운 기준을 제시합니다. 향후 연구는 DiffFit이 제시한 단순성의 우월성을 바탕으로, 이론적 이해를 심화하고 응용 범위를 확대하는 방향으로 진행될 것으로 예상됩니다.

***

### **참고 문헌 (선택된 주요 논문)**

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/45b40f7c-a5d3-44da-91db-7e414779da04/2304.06648v6.pdf)
[2](https://arxiv.org/abs/2401.13942)
[3](https://arxiv.org/abs/2403.14608)
[4](https://arxiv.org/html/2512.03499v1)
[5](https://arxiv.org/abs/2406.00773)
[6](https://link.springer.com/10.1007/978-3-031-72640-8_20)
[7](https://arxiv.org/abs/2402.02347)
[8](http://arxiv.org/pdf/2402.18331.pdf)
[9](https://arxiv.org/html/2405.21050)
[10](https://arxiv.org/abs/2405.19458)
[11](https://dl.acm.org/doi/10.1145/3708036.3708080)
[12](https://arxiv.org/abs/2403.07500)
[13](https://arxiv.org/abs/2407.06135)
[14](https://arxiv.org/abs/2402.02242)
[15](https://arxiv.org/abs/2402.04401)
[16](https://arxiv.org/html/2502.12146v1)
[17](https://arxiv.org/pdf/2304.06648.pdf)
[18](https://arxiv.org/abs/2402.17412)
[19](https://arxiv.org/pdf/2305.10924.pdf)
[20](https://arxiv.org/html/2312.03517v2)
[21](https://arxiv.org/html/2401.13942v1)
[22](https://proceedings.neurips.cc/paper_files/paper/2024/file/d00904cebc0d5b69fada8ad33d0f1422-Paper-Conference.pdf)
[23](https://www.reddit.com/r/computervision/comments/1lusb75/finetuning_a_vision_transformer_with_adaptive/)
[24](https://openreview.net/forum?id=nlWKpfIyMj)
[25](https://openaccess.thecvf.com/content/ICCV2023/papers/Xie_DiffFit_Unlocking_Transferability_of_Large_Diffusion_Models_via_Simple_Parameter-efficient_ICCV_2023_paper.pdf)
[26](https://developer0hye.tistory.com/855)
[27](https://proceedings.neurips.cc/paper_files/paper/2024/file/f782860c2a5d8f675b0066522b8c2cf2-Paper-Conference.pdf)
[28](https://www.merl.com/publications/docs/TR2024-104.pdf)
[29](https://labs.sciety.org/articles/by?article_doi=10.20944%2Fpreprints202510.2514.v1)
[30](https://arxiv.org/abs/2405.16876)
[31](https://arxiv.org/abs/2409.06633)
[32](https://arxiv.org/html/2512.03056v1)
[33](https://arxiv.org/pdf/2410.19878.pdf)
[34](https://arxiv.org/html/2411.19297v1)
[35](https://arxiv.org/pdf/2512.10877.pdf)
[36](https://arxiv.org/abs/2405.21050)
[37](https://arxiv.org/pdf/2412.03587.pdf)
[38](https://arxiv.org/html/2410.17891v2)
[39](https://openaccess.thecvf.com/content/CVPR2024W/PV/papers/Zanella_Low-Rank_Few-Shot_Adaptation_of_Vision-Language_Models_CVPRW_2024_paper.pdf)
[40](https://aclanthology.org/2024.naacl-long.174.pdf)
[41](https://huggingface.co/blog/samuellimabraz/peft-methods)
[42](https://arxiv.org/abs/2409.19589)
[43](https://ieeexplore.ieee.org/document/11094217/)
[44](https://www.semanticscholar.org/paper/75c8598be9acdcb3b66de826d07931a6a7555c0c)
[45](https://arxiv.org/abs/2505.22705)
[46](https://arxiv.org/abs/2410.03456)
[47](https://ieeexplore.ieee.org/document/10964526/)
[48](https://arxiv.org/abs/2503.23580)
[49](https://ieeexplore.ieee.org/document/11214030/)
[50](https://ieeexplore.ieee.org/document/11103412/)
[51](https://www.semanticscholar.org/paper/3c3245547a4f24eabb3aae6d90c2744a7a0cde41)
[52](https://arxiv.org/html/2503.10618)
[53](https://arxiv.org/html/2503.16726v1)
[54](https://arxiv.org/html/2410.03456)
[55](https://arxiv.org/html/2412.06028v1)
[56](https://arxiv.org/abs/2405.04312)
[57](https://arxiv.org/html/2502.20126v1)
[58](http://arxiv.org/pdf/2405.14854.pdf)
[59](https://arxiv.org/html/2405.14430)
[60](https://www.emergentmind.com/topics/diffusion-transformer-dit-architecture)
[61](https://pmc.ncbi.nlm.nih.gov/articles/PMC11562846/)
[62](https://openaccess.thecvf.com/content/WACV2025/papers/Imam_Test-Time_Low_Rank_Adaptation_via_Confidence_Maximization_for_Zero-Shot_Generalization_WACV_2025_paper.pdf)
[63](https://www.scitepress.org/Papers/2024/129378/129378.pdf)
[64](https://arxiv.org/html/2501.09732v1)
[65](https://www.sciencedirect.com/science/article/abs/pii/S1077314225002413)
[66](https://www.emergentmind.com/topics/diffusion-transformer-dit)
[67](https://openreview.net/forum?id=0BJTRUVDf4)
[68](https://apxml.com/courses/advanced-diffusion-architectures/chapter-3-transformer-diffusion-models/diffusion-transformers-dit)
[69](https://arxiv.org/html/2501.00365v2)
[70](https://arxiv.org/abs/2212.09748)
[71](https://arxiv.org/html/2504.06566v1)
[72](https://www.arxiv.org/abs/2511.22699)
[73](https://arxiv.org/abs/2407.12074)
[74](https://arxiv.org/abs/2504.09454)
[75](https://encord.com/blog/diffusion-models-with-transformers/)
[76](https://www.ibm.com/think/topics/diffusion-models)
