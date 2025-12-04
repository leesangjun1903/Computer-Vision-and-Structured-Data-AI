
# Autoregressive Image Generation using Residual Quantization

## 1. 핵심 주장과 주요 기여

**"Autoregressive Image Generation using Residual Quantization"** 논문의 가장 중요한 주장은 **기존의 Vector Quantization (VQ) 기반 자동회귀(AR) 모델이 해결하지 못한 근본적인 문제를 제시하고, 이를 효과적으로 극복할 수 있다**는 것입니다.[1]

### 핵심 주장의 요약

기존 VQ-VAE 접근법의 한계는 **속도-품질 균형(rate-distortion trade-off) 문제**에 있습니다. 이미지의 공간 해상도를 낮출수록 코드북 크기가 지수적으로 증가해야 하는데, 이는 다음과 같은 문제를 초래합니다:

- 매우 큰 코드북으로 인한 모델 파라미터 증가
- 코드북 붕괴(codebook collapse) 현상
- 학습 불안정성

이 논문의 핵심 기여는 **Residual Quantization (RQ)을 도입하여** 이 문제를 해결한 것입니다.[1]

### 주요 기여 3가지

1. **RQ-VAE 제안**: 고정된 크기의 단일 코드북을 사용하면서도 정확한 특징맵 근사가 가능한 구조
2. **RQ-Transformer 제안**: RQ-VAE에서 추출한 코드를 효과적으로 예측하는 아키텍처
3. **성능 개선**: 계산 비용 감소, 생성 속도 향상, 이미지 품질 개선을 동시에 달성

***

## 2. 해결하고자 하는 문제와 제안 방법

### 2.1 문제 정의

**이미지 생성을 위한 AR 모델의 근본적 문제:**

고해상도 이미지를 효율적으로 생성하려면 코드 시퀀스의 길이를 단축해야 하지만, 이는 다음과 같은 딜레마를 야기합니다:[1]

$$\text{HW} \times \log_2 K = \text{일정한 bit수}$$

여기서:
- $$H, W$$: 특징맵의 공간 해상도
- $$K$$: 코드북 크기

**VQ-VAE의 한계 예시:**
- 256×256 이미지에 대해 VQ-VAE는 보통 16×16 해상도에서 16,384개 크기의 코드북 필요
- 8×8 해상도로 줄이려면 $$16,384^4$$ 개의 코드북 필요
- 코드북 붕괴 문제로 실질적으로 불가능[1]

### 2.2 제안하는 방법: Residual Quantization (RQ)

#### 기본 개념

벡터 $$z \in \mathbb{R}^{n_z}$$에 대해 깊이(depth) $$D$$의 RQ는 다음과 같이 정의됩니다:[1]

$$\text{RQ}(z) = (k_1, k_2, \ldots, k_D) \in \mathcal{C}^D$$

여기서 각 깊이에서의 코드 $$k_d$$는 반복적으로 계산됩니다:

$$k_d = Q(r_{d-1}) \in \mathcal{C}, \quad r_d = r_{d-1} - e_{k_d}$$

여기서:
- $$r_0 = z$$ (초기 잔차)
- $$r_d$$: 깊이 $$d$$에서의 잔차
- $$e_{k_d}$$: 코드 $$k_d$$의 임베딩
- $$Q$$: 최근접 이웃 벡터 양자화

**부분 합(Partial Sum) 정의:**

$$z^d = \sum_{i=1}^{d} e_{k_i}$$

이는 깊이 $$d$$까지의 코드 임베딩의 합으로, 점진적으로 더 정교한 근사를 제공합니다.

#### 수학적 장점

RQ는 깊이 $$D$$에서 **최대 $$K^D$$개의 클러스터를 생성**하므로, 기존 VQ-VAE에 비해 훨씬 효율적입니다:[1]

| 비교 항목 | VQ-VAE | RQ-VAE (D=4) |
|--------|---------|-------------|
| 코드북 크기 | K | K |
| 가능한 클러스터 수 | K | K⁴ |
| 공간 해상도 | 16×16 (K=16,384) | 8×8 (K=16,384) |
| 학습 안정성 | 낮음 (큰 코드북) | 높음 (고정 크기) |

### 2.3 RQ-VAE의 상세 구조

**손실 함수:**

RQ-VAE의 학습은 다음 손실 함수를 최소화합니다:[1]

$$\mathcal{L} = \lambda \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{commit}}$$

**재구성 손실(Reconstruction Loss):**

$$\mathcal{L}_{\text{recon}} = \|X - G(Z^D)\|_2^2$$

여기서:
- $$X$$: 원본 이미지
- $$G$$: 디코더
- $$Z^D = \sum_{d=1}^{D} e_{k_d}$$: 최종 양자화된 특징맵

**커밋먼트 손실(Commitment Loss):**

$$\mathcal{L}_{\text{commit}} = \sum_{d=1}^{D} \|Z^{\text{sg}} - Z^d\|_2^2$$

여기서 $$\text{sg}$$는 그래디언트 정지 연산자입니다. 이 손실은 각 깊이에서의 양자화 오차를 순차적으로 감소시킵니다.[1]

### 2.4 RQ-Transformer 아키텍처

RQ-Transformer는 **공간 트랜스포머**와 **깊이 트랜스포머** 두 가지 구성요소로 이루어집니다.[1]

#### 확률론적 모델링

코드맵 $$M \in \mathcal{K}^{H \times W \times D}$$에 대해 래스터 스캔 순서로 재정렬하면:

```math
S = \text{raster scan}(M) \in \mathcal{K}^{T \times D}, \quad T = HW
```

각 위치 $$t$$에서 $$D$$개의 코드 $$S_{t,d}$$가 존재하고, 자동회귀 인수분해는:[1]

$$p(S) = \prod_{t=1}^{T} \prod_{d=1}^{D} p(S_{t,d}|S_{t, < d}, S_{ < t})$$

#### 공간 트랜스포머(Spatial Transformer)

위치 $$t$$의 입력:

$$u_t = \text{PE}_t^T + \sum_{d=1}^{D} e_{S_{t-1,d}}$$

여기서 $$\text{PE}_t^T$$는 공간적 위치 임베딩입니다. 이는 컨텍스트 벡터 $$h_t$$를 생성합니다.[1]

#### 깊이 트랜스포머(Depth Transformer)

깊이 $$d$$의 입력:

$$v_t^d = \text{PE}_d^D + \sum_{d'=1}^{d-1} e_{S_{t,d'}}$$

이는 깊이 $$d-1$$까지의 임베딩 합입니다. 깊이 트랜스포머는 조건부 분포를 예측합니다:[1]

$$p_t^d = p(S_{t,d}|S_{t, < d}, S_{ < t})$$

**계산 복잡도 분석:**

순진한 1D 시퀀스 접근 방식은 복잡도 $$O(N(TD)^2D^2)$$를 가지지만, RQ-Transformer는:[1]

$$\text{Complexity} = O(N_{\text{spatial}}T^2 + N_{\text{depth}}TD^2)$$

이는 훨씬 효율적입니다.

### 2.5 Exposure Bias 해결: 소프트 레이블링과 확률적 샘플링

#### 문제: Exposure Bias

학습 중에는 정답 코드를 입력하지만, 추론 중에는 예측된 코드를 사용하는 불일치가 발생합니다.[1]

#### 해결책 1: 소프트 레이블링(Soft Labeling)

코드 임베딩 사이의 기하학적 관계를 기반으로 한 범주형 분포 정의:[1]

$$Q_\sigma(k|z) = \frac{\exp(-\|z - e_k\|_2^2 / \sigma)}{Z(\sigma)} \text{ for } k \in \mathcal{C}$$

$$\sigma \to 0$$일 때 원-핫 분포로 수렴합니다.

원래의 원-핫 레이블 $$Q_0(r_{t,d-1})$$ 대신 소프트 분포 $$Q_\sigma(r_{t,d-1})$$를 사용하여 손실을 계산합니다.[1]

#### 해결책 2: 확률적 샘플링(Stochastic Sampling)

학습 중 결정론적 코드 선택 대신 분포에서 샘플링:

$$S_t^d \sim Q_\sigma(r_{t,d-1})$$

이는 학습-추론 불일치를 감소시킵니다.[1]

***

## 3. 모델 구조

### 3.1 2단계 파이프라인

RQ-Transformer 프레임워크는 두 단계로 구성됩니다:[1]

```
Stage 1: Image → RQ-VAE → Stacked Code Map (8×8×4)
             ↓
Stage 2: Code Map → RQ-Transformer → Next Codes Prediction
```

### 3.2 RQ-VAE 상세 구조

**입력:** 256×256×3 RGB 이미지
**출력:** 8×8×4 스택 코드맵 (K=16,384)

**구성:**
- 인코더: 256×256 → 8×8 (downsampling factor 32)
- 잔차 양자화 모듈: 깊이 D=4
- 디코더: 8×8 → 256×256

**핵심 특징:** 코드북 붕괴 방지를 위해 지수이동평균(EMA)으로 코드북 업데이트[1]

### 3.3 RQ-Transformer 상세 구조

**아키텍처:**
- 공간 트랜스포머: 24개 레이어 (기본), 최대 42개 (1.4B+ 파라미터)
- 깊이 트랜스포머: 4개 레이어 (기본), 최대 6개 (대규모 모델)

**입력 처리:**
- 공간 위치: T=64 (8×8 그리드)
- 깊이: D=4
- 각 위치에서 4개의 코드를 순차적으로 예측

**조건화 방식:**
- 클래스 조건: 임베딩을 공간 트랜스포머 입력 시작에 추가
- 텍스트 조건: BPE 인코딩된 토큰 시퀀스 (최대 32개) + 이미지 토큰[1]

***

## 4. 성능 향상 및 실험 결과

### 4.1 무조건 이미지 생성(Unconditional Generation)

**평가 데이터셋:** LSUN-Cat, Bedroom, Church, FFHQ

| 모델 | Cat | Bedroom | Church | FFHQ |
|------|-----|---------|--------|------|
| VQ-GAN | 17.31 | 6.35 | 7.81 | 11.4 |
| **RQ-Transformer** | **8.64** | **3.04** | **7.45** | **10.38** |
| StyleGAN2 | 7.25 | 2.35 | 3.86 | 3.8 |

**주요 성과:**
- LSUN-Cat에서 VQ-GAN 대비 **50% FID 개선**
- LSUN-Bedroom에서 **52% FID 개선**
- 작은 데이터셋(FFHQ)에서는 StyleGAN2에 비해 성능 차이 있음 (오버피팅 이슈)[1]

### 4.2 조건부 이미지 생성

#### ImageNet 클래스 조건 생성

**주요 결과:**

| 파라미터 | 거절 샘플링 없음 | FID | IS | 거절 샘플링 포함 | FID | IS |
|---------|---|---|---|---|---|---|
| 480M | - | 15.72 | 86.8 | - | - | - |
| 821M | ✓ | 13.11 | 104.3 | - | - | - |
| 1.4B | ✓ | 8.71* | 119.0 | ✓ | 4.45 | 326.0 |
| 3.8B | ✓ | 7.55 | 134.0 | ✓ | 3.80 | 323.7 |

\* 50 에포크 학습 RQ-VAE 사용

**성과:**
- BigGAN-deep과 경쟁력 있는 성능 달성 (7.55 FID)
- 거절 샘플링 적용 시 **ADM보다 우수** (3.80 vs 4.59 FID)[1]

#### CC-3M 텍스트 조건 생성

| 모델 | 파라미터 | FID | CLIP-s |
|------|---------|-----|---------|
| VQ-GAN | 600M | 28.86 | 0.20 |
| ImageBART | 2.8B | 22.61 | 0.23 |
| **RQ-Transformer** | **654M** | **12.33** | **0.26** |

**성과:**
- ImageBART보다 **2.3배 작은 파라미터로 46% FID 개선**
- CLIP 스코어 13% 향상[1]

### 4.3 계산 효율성

**샘플링 속도 비교 (1.4B 파라미터, NVIDIA A100):**

| 배치 크기 | VQ-GAN 시간/이미지 | RQ-Transformer 시간/이미지 | 속도 향상 |
|---------|---|---|---|
| 100 | - | - | **4.1배** |
| 200 | ~0.0687초 | ~0.0123초 | **5.6배** |
| 500 | 불가능 | ~0.02초 | **7.3배** |

**주요 이점:**
- 짧은 시퀀스 길이(64 vs 256)로 메모리 절감
- 배치 크기 증가 가능 → 병렬화 향상[1]

### 4.4 RQ-VAE 절제 연구(Ablation Study)

**깊이(D)의 영향 (ImageNet 검증 데이터, K=16,384):**

| H×W | D=1 | D=2 | D=3 | D=4 | D=8 |
|-----|-----|-----|-----|-----|-----|
| **8×8** | 17.95 | 10.77 | 7.66 | **4.73** | 2.69 |
| **16×16** | 4.32 | - | - | - | - |

**결론:**
- **깊이 증가가 코드북 크기 증가보다 훨씬 효과적**
- D=4는 계산 비용과 성능의 최적 균형[1]

**조악-정교 근사(Coarse-to-Fine Approximation) 검증:**

재구성 손실, 커밋먼트 손실, 지각 손실 모두 깊이가 증가할수록 단조 감소함을 확인했습니다.[1]

***

## 5. 모델의 일반화 성능 향상 가능성

### 5.1 일반화 성능의 핵심 요소

#### 1. 재귀적 양자화를 통한 정보 보존

RQ-VAE의 재귀적 양자화 메커니즘은 다음과 같은 이유로 일반화를 향상시킵니다:[1]

**다양한 스케일에서의 특징 추출:**

$$z^1, z^2, \ldots, z^D$$

각 깊이에서 다양한 추상화 수준의 특징을 포착합니다. 예를 들어:
- $$z^1$$: 거시적 구조 (색상, 큰 객체)
- $$z^2$$: 중간 수준 특징 (경계, 질감)
- $$z^3, z^4$$: 미세한 세부사항 (텍스처, 잡음)

#### 2. 확률적 샘플링의 정규화 효과

학습 중 소프트 레이블링과 확률적 샘플링은 모델의 견고성을 강화합니다:[1]

$$\mathcal{L} = -\sum_{k} Q_\sigma(k|r_{t,d-1}) \log p(k)$$

이는 임베딩 공간의 기하학적 구조를 활용하여 과적합을 방지합니다.

#### 3. 짧은 시퀀스의 장점

공간 해상도 감소 (256×256 → 8×8)로 인한 일반화 향상:[1]

- **수용장(Receptive Field) 확대**: 각 코드가 더 넓은 공간 범위를 대표
- **장거리 의존성 학습 용이**: 64개 토큰은 256개 토큰보다 장거리 관계를 효과적으로 학습
- **계산 효율화**: 더 큰 배치 크기 → 더 안정적인 그래디언트

### 5.2 교차 데이터셋 일반화 성능

**ImageNet 사전학습 모델을 LSUN 데이터셋에 적용:**

논문에서 ImageNet 사전학습된 RQ-VAE를 단 1 에포크 미세조정으로 LSUN 데이터셋에 적용했을 때:[1]

- LSUN-Cat: FID 8.64 (경쟁력 있음)
- LSUN-Bedroom: FID 3.04 (최고 수준)

이는 **RQ-VAE의 일반화 성능이 우수함**을 시사합니다.

### 5.3 구성적 일반화(Compositional Generalization)

**텍스트 조건 생성에서의 구성적 일반화:**

논문에서 "A cheeseburger in front of a mountain range covered with snow"와 같은 **학습 중에 본 적 없는 시각적 개념의 조합**을 생성할 수 있음을 보였습니다.[1]

이는 다음을 시사합니다:
- RQ-Transformer가 개별 개념의 임베딩을 학습
- 이들을 새로운 조합으로 구성하는 능력
- **좋은 부분 공간 분해(latent space factorization)**

### 5.4 데이터 효율성

**작은 데이터셋(FFHQ, 70K 이미지)에서의 성능:**

- 조기 중단(Early Stopping) 필요
- StyleGAN2에 비해 약간 낮은 성능 (10.38 vs 3.8 FID)
- **AR 모델의 고질적 문제**: 작은 데이터셋에서 암기 경향

향상 가능성:
- 정규화 기법 추가
- 데이터 증강 전략
- 건축적 개선[1]

***

## 6. 모델의 한계

논문에서 명확히 제시한 3가지 주요 한계:[1]

### 한계 1: 작은 데이터셋에서의 성능

**문제:** StyleGAN2(3.8)에 비해 FFHQ에서 낮은 성능(10.38)
- 원인: AR 모델의 과적합 경향성
- 제안: AR 모델 정규화 연구 필요

### 한계 2: 텍스트-이미지 생성의 스케일 제한

**문제:** 대규모 텍스트-이미지 데이터와 모델 크기 미탐색
- 선행 연구(DALL-E 등): 매우 큰 모델과 데이터셋의 효과 입증
- 개선: 모델 스케일 증가 필요

### 한계 3: 양방향 컨텍스트 미지원

**문제:** AR 모델의 근본적 한계 - 이전 토큰만 사용
- 불가능한 작업: 이미지 인페인팅(inpainting), 아웃페인팅(outpainting)
- 양방향 컨텍스트 활용 시 성능 향상 가능[1]

### 한계 4: 환경 영향

**환경 고려사항:**
- 대규모 AR 모델 학습의 높은 에너지 소비
- 탄소 발자국 증가
- 효율적 학습 연구 필요[1]

***

## 7. 앞으로의 연구에 미치는 영향 및 고려사항

### 7.1 이론적 영향

#### 1. 양자화 이론의 새로운 활용

RQ-VAE의 성공은 **정보이론과 신경망의 결합**의 가능성을 제시합니다:[1]

**레이트-왜곡 이론의 실증:**
- 이론적 최적성: $$\log_2(K^D)$$ bits로 $$K^D$$ 클러스터 가능
- 실제 구현: 이 이론적 한계에 가까운 성능 달성

이는 향후 연구자들에게:
- 다른 도메인(오디오, 비디오)에서 RQ 적용 가능성 시사
- 최적 깊이-코드북 크기 선택 이론 개발 필요[1]

#### 2. 자동회귀 모델의 재평가

2020년 이후 **diffusion 모델의 우위**가 당연시되었지만, 이 논문과 후속 연구들(VAR, HART 등)이 보여준 바:[2][3][4]

- AR 모델도 적절한 설계로 diffusion과 경쟁 가능
- 더 빠른 샘플링 속도 가능
- 더 나은 확장성(scalability) 가능

### 7.2 실용적 영향

#### 1. 이미지 생성 파이프라인 표준화

RQ-Transformer의 구조가 영향을 미친 이후 연구들:[3][5][2]

**Visual AutoRegressive (VAR) modeling:** 
- Coarse-to-fine "next-scale prediction" 도입
- RQ의 깊이 개념을 해상도 스케일로 확장[2]

**Hybrid Autoregressive Transformer (HART):**
- 1024×1024 직접 생성 가능
- diffusion 모델과 경쟁 수준 품질[3]

**Compositional Auto-Regressive Transformer (CART):**
- Next-detail 전략으로 향상된 충실도[5]

#### 2. 효율성 개선 연구 활성화

RQ-Transformer의 계산 복잡도 분석과 개선이 이후 연구를 촉발:[6]

**Grouped Speculative Decoding (GSD):**
- AR 이미지 생성 **3.7배 가속화**
- 추가 학습 없이 품질 유지[6]

#### 3. 멀티모달 생성 모델로의 확장

**Ming-Lite-Uni (2025):**
- RQ 기반 접근을 텍스트-이미지 생성으로 확장
- 텍스트 기반 이미지 편집 가능[7]

**UGen (2025):**
- 통합 멀티모달 모델에서 RQ 패러다임 적용
- 텍스트, 이미지 이해 및 생성 동시 처리[8]

### 7.3 앞으로 연구 시 고려할 점

#### 1. 적응적 양자화(Adaptive Quantization) 탐구

**현재 방식의 한계:**
- 모든 데이터에 고정 깊이 D 사용
- 단순한 패턴은 적은 깊이로도 충분

**개선 방향:**
- 콘텐츠별 동적 깊이 조정
- 레이트-왜곡 최적화[9]

#### 2. 장기 시퀀스 모델링

**현재 성과:**
- 8×8×4 = 64 토큰으로 효율적 모델링
- 더 고해상도(16×16 이상)는 아직 미탐색

**연구 방향:**
- 더 긴 시퀀스 처리 능력 개선
- 하이브리드 아키텍처 개발
- 계층적 생성 구조[2]

#### 3. 작은 데이터셋에 대한 정규화 기법

**문제:** AR 모델의 과적합
**해결책:**
- 드롭아웃 강화
- 데이터 증강 전략 개발
- 메타-러닝 접근
- 사전학습 모델 활용 극대화[1]

#### 4. 양방향 컨텍스트 통합

**가능한 접근:**
- 인페인팅을 위한 마스크 기반 생성
- 양방향 트랜스포머와의 하이브리드
- 반복적 정제(iterative refinement)[1]

#### 5. 크로스-모달 일반화

**새로운 방향:**
- 텍스트 설명만으로 새로운 이미지 생성
- 의미론적 개념의 조합 능력 향상
- 제로샷(zero-shot) 일반화 성능 개선[1]

#### 6. 에너지 효율 중심 설계

**고려사항:**
- 학습 시간 단축 기술
- 모델 압축 기법
- 에너지-성능 트레이드오프 분석[1]

***

## 8. 2020년 이후 관련 최신 연구 탐색

### 8.1 VQ-기반 생성 모델의 진화

**핵심 발전 계보:**

| 연도 | 모델 | 주요 기여 | 논문 |
|------|------|---------|------|
| 2020 | VQ-VAE-2 | 계층적 VQ |  |
| 2021 | VQ-GAN | 적대적 학습 추가 | [5] |
| 2022 | **RQ-VAE** | 잔차 양자화 도입 | [1] |
| 2024 | VAR | Next-scale 예측 | [2] |
| 2024 | HART | 하이브리드 아키텍처 | [3] |
| 2024 | LlamaGen | LLM 패러다임 적용 | [10] |
| 2025 | CART | 구성적 생성 | [5] |

### 8.2 속도 최적화 관련 최신 연구

**2025년 최신 진전:**

**Grouped Speculative Decoding (GSD):**[6]
- RQ-Transformer 이후 AR 모델의 주요 병목(느린 샘플링) 해결
- 3.7배 평균 가속화, 최대 4.3배 가속화
- 추가 학습 없이 적용 가능

**ARGenSeg (2025):**[11]
- AR 생성 패러다임을 이미지 분할로 확장
- 픽셀 수준 지각(pixel-level perception) 달성

### 8.3 멀티모달 생성 모델로의 확장

**최신 통합 모델들:**

**Ming-Lite-Uni (2025):**[7]
- 통합 비전-언어 생성 모델
- 텍스트-이미지 생성 + 이미지 편집
- RQ 패러다임의 멀티모달 적용

**UGen (2025):**[8]
- Progressive vocabulary learning
- 텍스트 처리, 이미지 이해, 이미지 생성 동시 수행
- 단일 트랜스포머로 모든 작업 처리

### 8.4 이론적 기초 연구

**레이트 적응형 양자화(RAQ, 2024):**[9]
- 사후 학습 미분 가능한 클러스터링
- RQ-VAE의 이론적 개선

**견고한 RQ-VAE (RVQ-VAE, 2024):**[12]
- 복잡한 데이터셋에 대한 견고성 향상
- 허버 발산(Huber divergence) 활용

### 8.5 다른 도메인으로의 확장

**오디오 생성:**
- Foley 음향 생성에 RVQ 적용[13]
- AR 패러다임 기반 음악 생성

**그래프 생성:**
- 순열 불변성 유지하면서 다음 스케일 예측[14]
- Diffusion 없는 효율적 생성

***

## 9. 결론 및 종합 평가

### 9.1 논문의 역사적 의미

**"Autoregressive Image Generation using Residual Quantization"**은 2022년 발표 이후:

1. **AR 모델의 재평가**: Diffusion 모델 주류 속에서 AR의 경쟁력 입증
2. **기술 표준화**: RQ 개념이 후속 SOTA 모델들의 기초
3. **효율성 혁신**: 계산 비용 대폭 감소 + 성능 향상의 동시 달성

### 9.2 핵심 성과 재정리

| 측면 | 성과 | 영향 |
|------|------|------|
| **구조적 창의성** | RQ 개념의 도입 | 이후 VAR, HART 등에 영향 |
| **성능** | SOTA 달성 (여러 벤치마크) | 경쟁력 있는 대안 제시 |
| **효율성** | 7.3배 샘플링 가속화 | 실무 적용 가능성 증대 |
| **일반화** | 우수한 교차-데이터셋 성능 | 견고한 표현 학습 |

### 9.3 미래 연구의 방향

이 논문은 다음과 같은 향후 연구 방향을 제시합니다:

1. **적응적 양자화**: 콘텐츠별 동적 깊이 조정
2. **고해상도 생성**: 1024×1024 이상 직접 생성
3. **작은 데이터셋 정규화**: 과적합 문제 해결
4. **양방향 모델링**: 이미지 편집 작업 지원
5. **멀티모달 확장**: 통합 생성 모델 개발
6. **환경 효율**: 탄소 중립 학습 기법

***

## 참고: 수식 요약

**핵심 수식 모음:**

$$\text{RQ}(z) = (k_1, \ldots, k_D), \quad k_d = Q(r_{d-1}), \quad r_d = r_{d-1} - e_{k_d}$$

$$z^d = \sum_{i=1}^{d} e_{k_i}$$

$$\mathcal{L} = \lambda \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{commit}} = \lambda \|X - G(Z^D)\|_2^2 + \sum_{d=1}^{D} \|Z^{\text{sg}} - Z^d\|_2^2$$

$$p(S) = \prod_{t=1}^{T} \prod_{d=1}^{D} p(S_{t,d}|S_{t, < d}, S_{ < t})$$

$$u_t = \text{PE}_t^T + \sum_{d=1}^{D} e_{S_{t-1,d}}$$

$$Q_\sigma(k|z) = \frac{\exp(-\|z - e_k\|_2^2 / \sigma)}{Z(\sigma)}$$

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8a4f7528-064d-40bb-b3c4-016277da794a/2203.01941v2-abcugdoem.pdf)
[2](http://arxiv.org/pdf/2404.02905.pdf)
[3](https://arxiv.org/html/2410.10812)
[4](https://juniboy97.tistory.com/56)
[5](https://arxiv.org/html/2411.10180v1)
[6](https://openaccess.thecvf.com/content/ICCV2025/papers/So_Grouped_Speculative_Decoding_for_Autoregressive_Image_Generation_ICCV_2025_paper.pdf)
[7](https://arxiv.org/abs/2505.02471)
[8](http://arxiv.org/pdf/2503.21193.pdf)
[9](https://www.emergentmind.com/topics/residual-quantized-variational-autoencoders-rq-vae)
[10](http://arxiv.org/pdf/2406.06525v1.pdf)
[11](https://neurips.cc/virtual/2025/poster/115738)
[12](https://openreview.net/pdf?id=GkGVNmjAwh)
[13](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003067590)
[14](https://arxiv.org/html/2503.23612v1)
[15](https://www.frontiersin.org/articles/10.3389/fdgth.2025.1653369/full)
[16](https://oarjst.com/node/710)
[17](https://bmcinfectdis.biomedcentral.com/articles/10.1186/s12879-025-11574-2)
[18](http://pubs.rsna.org/doi/10.1148/ryai.240625)
[19](https://www.mdpi.com/1999-4923/17/9/1169)
[20](https://www.richtmann.org/journal/index.php/jesr/article/view/14361)
[21](https://journal.unnes.ac.id/journals/jf/article/view/27967)
[22](http://naukaru.ru/en/nauka/article/103930/view)
[23](https://fg.bmj.com/lookup/doi/10.1136/flgastro-2025-103282)
[24](https://arxiv.org/html/2410.04671)
[25](https://arxiv.org/abs/2503.11073)
[26](https://openaccess.thecvf.com/content/CVPR2021/papers/Esser_Taming_Transformers_for_High-Resolution_Image_Synthesis_CVPR_2021_paper.pdf)
[27](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760106.pdf)
[28](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/vqd/)
[29](https://arxiv.org/html/2509.15185v1)
