
# Autoregressive Diffusion Models
## 1. 핵심 주장 및 주요 기여
### 1.1 핵심 주장
Hoogeboom et al. (2022)의 "Autoregressive Diffusion Models"은 자동회귀 모델(ARM)과 이산 확산 모델(Discrete Diffusion Models)의 이론적·실무적 장점을 통합하는 통합 프레임워크를 제시합니다. ARDMs의 핵심 주장은 다음과 같습니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a7aceff6-b4a1-4c6c-b692-b91c9a0a1177/2110.02037v2.pdf)

**1. 순서 무관성(Order Agnosticism)의 유리성**: 데이터 생성 순서가 고정되지 않은 순서 무관 훈련 목적함수를 통해, 이미지와 같이 자연스러운 생성 순서가 명확하지 않은 데이터에서 더 나은 일반화 성능을 달성할 수 있습니다.

**2. 효율적 훈련 패러다임**: 확산 모델처럼 데이터포인트당 단일 단계를 최적화하는 목적함수를 사용하여, 고차원 데이터로의 확장이 용이합니다. 이는 전체 가능도를 동시에 최적화하는 기존 ARM의 제약을 극복합니다.

**3. 병렬화 가능성**: 동적계획법(Dynamic Programming)을 활용하여 여러 변수를 동시에 생성할 수 있으며, 생성 단계 수를 유연하게 조절할 수 있습니다.

### 1.2 주요 기여(Contributions)
논문은 명시적으로 세 가지 주요 기여를 제시합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a7aceff6-b4a1-4c6c-b692-b91c9a0a1177/2110.02037v2.pdf)

**기여 1: 확장 가능한 ARDM 클래스 도입**
- 순서 무관 ARM에 변수 상향 확대(upscaling) 기능을 추가한 새로운 모델 클래스 제시
- 기존 방법들의 제약을 제거하고 아키텍처 선택의 유연성 확대

**기여 2: 이론적 동등성 증명**
- ARDM과 흡수 확산(Absorbing Diffusion)의 연속 시간 한계에서의 동등성을 수학적으로 증명
- 두 모델 클래스 간의 이론적 다리 구축

**기여 3: 병렬화 및 무손실 압축 응용**
- 동적계획법 기반의 병렬 생성 알고리즘 개발
- 비트 후퇴 부호화(bits-back coding)를 사용하지 않고도 효율적인 무손실 압축 달성

***

## 2. 해결하고자 하는 문제 및 제안 방법
### 2.1 문제 정의
#### 자동회귀 모델의 문제점

**제약 1: 생성 순서의 고정성**
$$\log p(x) = \sum_{t=1}^{D} \log p(x_t | x_{<t})$$

이 표준 ARM 가능도는 x₁부터 x_D까지 특정 순서를 가정합니다. 이미지와 같은 데이터에서는 이 순서가 임의적이며, 최적이 아닐 수 있습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a7aceff6-b4a1-4c6c-b692-b91c9a0a1177/2110.02037v2.pdf)

**제약 2: 샘플링의 순차성**
- 샘플링: D번의 순차적 신경망 호출 필요
- 가능도 평가: 1번의 호출로 효율적 (대신 학습 시 인과 마스킹 필요)

**제약 3: 아키텍처 제약**
- 인과 마스킹(Causal Masking) 필수
- 합성곱 층 같은 비등변 변환 구조 적용 불가
- 임의의 순서에 대해 삼각형 의존성 구조 강제

#### 이산 확산 모델의 문제점

**제약 1: 높은 계산 비용**
- D3PM(Discrete Denoising Diffusion Probabilistic Models): 256개 토큰에 1000 단계 필요
- 이 중 ~744개 단계는 실질적으로 잠재 변수에 변화 없음 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a7aceff6-b4a1-4c6c-b692-b91c9a0a1177/2110.02037v2.pdf)

**제약 2: 유연성 부족**
- 단계 수 T 사전 결정 필요
- 성능 저하 위험

### 2.2 제안 방법: ARDM의 핵심 구조
#### 2.2.1 순서 무관 ARDM의 수학적 기초

**Step 1: 확률 체인 규칙의 확장**

표준 ARM의 가능도에서 시작:

$$\log p(x) = \sum_{t=1}^{D} \log p(x_t | x_{ < t})$$

이를 예상값 재가중화를 통해 확장:

$$\log p(x) \geq \mathbb{E}_{\sigma \sim U(S_D)} \sum_{t=1}^{D} \log p(x_{\sigma(t)} | x_{\sigma(<t)})$$

여기서 σ는 D개 원소의 무작위 순열입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a7aceff6-b4a1-4c6c-b692-b91c9a0a1177/2110.02037v2.pdf)

**Step 2: 효율적 단계별 목적함수 유도**

합을 균등 분포 예상값으로 변환:

$$\mathbb{E}_{\sigma \sim U(S_D)} \sum_{t=1}^{D} \log p(x_{\sigma(t)} | x_{\sigma(<t)}) = \mathbb{E}_{\sigma \sim U(S_D)} D \cdot \mathbb{E}_{t \sim U(1,...,D)} \log p(x_{\sigma(t)} | x_{\sigma(<t)})$$

정리하면:

$$\log p(x) \geq \mathbb{E}_{t \sim U(1,...,D)} [D \cdot L_t]$$

여기서:

$$L_t = \frac{1}{D-t+1} \mathbb{E}_{\sigma \sim U(S_D)} \sum_{k \in \sigma(\geq t)} \log p(x_k | x_{\sigma(<t)})$$

**핵심 통찰**: 각 데이터포인트에 대해 단일 L_t 항만 최적화하면 되므로, 훈련 복잡도가 선형으로 유지됩니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a7aceff6-b4a1-4c6c-b692-b91c9a0a1177/2110.02037v2.pdf)

#### 2.2.2 마스킹 기반 조건부 분포 매개변수화

**네트워크 아키텍처**:
- 단일 신경망 f: X → ℝ^(D×K)
- 마스킹을 통한 조건부 처리

**마스킹 절차**:

$$m = \sigma < t \quad \text{(Boolean 마스크)}$$

$$\theta = f(m \odot x) \quad \text{(마스킹된 입력과 함께 네트워크 호출)}$$

여기서 ⊙는 원소별 곱셈입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a7aceff6-b4a1-4c6c-b692-b91c9a0a1177/2110.02037v2.pdf)

**확률 분포 모델링**:

$$\log p(x_k | x_{\sigma( < t)}) = \log \mathcal{C}(x_k | \theta_k)$$

여기서 𝒞는 카테고리 분포입니다.

**입력 처리의 다양성**:
- **이미지/음성**: 마스크를 정규화된 입력에 적용, 입력 채널로 연결
- **언어**: 흡수 상태를 새로운 클래스 K+1로 설정

#### 2.2.3 깊이 상향 확대(Depth Upscaling)

**하향 파괴 과정(Downscaling)**:

$$P^{(s)}: x^{(s)} \rightarrow x^{(s-1)} \rightarrow \cdots \rightarrow x^{(0)} = \text{absorbing state}$$

**누적 전이 행렬**:

$$x^{(s)} = \bar{P}^{(s+1)} x^{(S)}$$

여기서:

$$\bar{P}^{(s)} = P^{(s)} \cdot P^{(s+1)} \cdots P^{(S)}$$

**비트 상향 확대(Bit Upscaling) 예시**:
8비트 픽셀값(0-255)의 경우, 8단계를 통해 최상위 비트부터 생성:

$$P^{(8+1-s)}_{l,k} = \begin{cases} 1 & \text{if } l = \lfloor k/2^s \rfloor \cdot 2^s \text{ and } k \in \text{Im}(\text{lsb}_{s-1}) \\ 0 & \text{otherwise} \end{cases}$$

**훈련 복잡도 불변성**:
- 단계 s ∼ U(1, ..., S) 무작위 샘플링
- S개의 단계를 추가해도 훈련 비용 증가 없음
- 하지만 생성 품질 향상 (실험에서 확인)

***

## 3. 모델 구조 및 알고리즘
### 3.1 기본 ARDM 구조
위 차트는 세 가지 주요 데이터 모달리티에서 ARDMs의 성능 우위를 보여줍니다. Text8에서 ARDMs는 D3PM보다 1/4의 단계로 더 나은 결과를 달성하고, CIFAR-10에서는 경쟁력 있는 성능을 달성합니다.

### 3.2 병렬화 ARDM (Parallelized ARDM)
#### 핵심 원리

**핵심 관찰 (Equation 3)**:

$$\mathbb{E}\_{\sigma} [\log p(x_{\sigma(t+k)} | x_{\sigma(<t)})] = \mathbb{E}_{\sigma} [\log p(x_{\sigma(t)} | x_{\sigma(<t)})] = L_t$$

균등 순열 예상값으로 인해, 어느 단계에서 변수를 생성하든 예상 가능도는 동일합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a7aceff6-b4a1-4c6c-b692-b91c9a0a1177/2110.02037v2.pdf)

**병렬 생성의 비용**:
- t번째 위치에서 k개 변수를 동시에 생성하는 비용: k·L_t 비트
- 순차 생성 비용: ∑(i=1 to k) L_{t+i}

**동적계획법 기반 최적화**:
주어진 예산(step budget)에 대해 최적 병렬화 정책을 계산하는 O(D³) 알고리즘

**성능-속도 트레이드오프 (Equation 4)**:

$$L_t = \mathbb{E}\_{\sigma} [\log p(x_{\sigma(t+1)} | x_{\sigma(<t)})] \leq \mathbb{E}_{\sigma} [\log p(x_{\sigma(t+1)} | x_{\sigma(<t+1)})] = L_{t+1}$$

L_t는 단조 감소하므로, 병렬화는 우아한 성능 저하를 초래합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a7aceff6-b4a1-4c6c-b692-b91c9a0a1177/2110.02037v2.pdf)

#### 구현: 손실 함수 행렬

$$L_{t,t+k} = k \cdot L_t \quad (\text{for } k > 0)$$

$$L_{t+k,t} = 0 \quad (\text{otherwise})$$

이 행렬 구조를 활용하여 동적계획법 적용 가능.

### 3.3 ARDM과 흡수 확산의 이론적 동등성
**연속 시간 흡수 확산**:
각 차원 x_i(t)는 독립적으로 비율 γ(t)로 흡수 상태 a_i로 붕괴:
$$\alpha(t) = \exp\left(-\int_0^t \gamma(s)ds\right)$$

**핵심 증명 (Appendix C)**:
1. 역과정은 D개의 무작위 전이 시간 {τ_i}로 표현 가능
2. 역과정에서 {x_i(τ_i), τ_i}만 모델링 필요
3. τ_i는 조건부 독립: x_i(0)|x(t) ∼ x_i(τ_i)|x(t)
4. 전이 시간 순서는 균등 분포 → OA-ARDM 동등성

**결론**: ARDM은 흡수 확산의 연속 시간 한계이며, 따라서 최대 표현력을 가짐. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a7aceff6-b4a1-4c6c-b692-b91c9a0a1177/2110.02037v2.pdf)

***

## 4. 성능 향상 및 실험 결과
### 4.1 Text8 데이터셋 (텍스트 모델링)
| 모델 | 단계 수 | NLL (bpc) | 개선점 |
|------|--------|-----------|--------|
| OA-Transformer | 250 | 1.64 | 기저선 (인과 마스킹) |
| D3PM-absorbing | 1000 | 1.45 ± 0.020 | 높은 단계 수 필요 |
| **OA-ARDM** | **250** | **1.43 ± 0.001** | **4배 효율: D3PM보다 적은 단계로 더 나은 성능** |
| Parallelized OA-ARDM | 20 | 1.51 ± 0.007 | 20배 가속화, 성능 저하 최소 |

**핵심 발견**:
- ARDMs는 D3PM의 1/4 단계로 더 우수한 성능 달성
- 병렬화 시 20 단계에서도 경쟁력 있는 성능 유지 (1.51 bpc)
- OA-Transformer의 인과 마스킹 기반 접근보다 우수 (1.64 → 1.43)

### 4.2 CIFAR-10 이미지 데이터셋
| 모델 | 단계 | 성능 (bpd) | 비고 |
|------|-----|-----------|------|
| ARDM-OA | 3072 | 2.69 ± 0.005 | 표준 순서 무관 |
| Parallel ARDM-OA | 50 | 2.74 | 62배 가속화 |
| **ARDM-Upscale 4** | **4 × 3072** | **2.64 ± 0.002** | **최고 성능 (상향 확대 적용)** |
| Parallel ARDM-Upscale 4 | 4 × 50 | 2.68 | 병렬화 + 상향 확대 |
| D3PM-absorbing | 1000 | 4.40 | 이산 확산 기저선 |
| D3PM-Gaussian | 1000 | 3.44 | 가우시안 확산 기저선 |
| VDM (최고 성능) | 1000 | 2.49 | 변분 확산 모델 SOTA |

**핵심 발견**:
- 상향 확대: 2.69 → 2.64 bpd (0.05 개선)
- D3PM-absorbing 대비 40% 성능 향상
- VDM 대비 다소 뒤지지만, 효율성에서 우수

### 4.3 SC09 음성 데이터셋 (시계열 모델링)
| 모델 | 단계 | 성능 (bpd) | 개선율 |
|------|-----|-----------|--------|
| WaveNet (단일 순서) | 16000 | 7.77 | 기저선 |
| OA-ARDM | 16000 | 7.93 | -0.16 (소폭 악화) |
| **ARDM-Upscale 256** | **2 × 16000** | **6.36** | **19.9% 개선** |
| **ARDM-Upscale 16** | **4 × 16000** | **6.30** | **23.0% 개선** |
| **ARDM-Upscale 4** | **8 × 16000** | **6.29** | **23.0% 개선** |
| ARDM-Upscale 2 | 16 × 16000 | 6.29 | 수렴 (수익 감소) |

**핵심 발견**:
- 음성에 상향 확대가 가장 효과적 (7.93 → 6.29, 20% 이상 개선)
- 상향 인수 4-8에서 최적 성능 (더 이상 상향 확대 시 수익 감소)
- WaveNet 기저선보다 20% 우수

### 4.4 무손실 압축 응용 (CIFAR-10 이미지당 압축)
| 모델 | 압축률 (bpd) | 방식 | 초기 오버헤드 |
|------|-------------|------|-------------|
| **ARDM-Upscale 4** | **2.71** | 직접 | 없음 |
| ARDM-OA | 2.73 | 직접 | 없음 |
| VDM | 2.72 | 비트-후퇴 | 약 8 bpd |
| IDF++ | 3.26 | 직접 | 없음 |
| LBB | 3.12 | 비트-후퇴 | 약 8 bpd |
| HiLLoC | 4.19 | 비트-후퇴 | 약 8 bpd (FLIF) |
| FLIF | 4.19 | 전통 코덱 | - |
| PNG | - | 전통 | - |

**핵심 발견**:
- **이미지당 압축에서 SOTA 달성** (2.71 bpd)
- 비트-후퇴 부호화 사용 안 함 (초기 오버헤드 제거)
- 단일 데이터포인트 압축 가능 (VDM과 차별)
- 적당한 네트워크 호출 수로 (de)압축 가능
- 전체 데이터셋 압축에서도 VDM과 경쟁력 (2.73 vs 2.72)

***

## 5. 모델의 일반화 성능 향상 메커니즘
### 5.1 순서 무관성이 일반화를 개선하는 이유
**원리 1: 데이터 의존성 다양성 학습**
- 고정 순서(예: 좌상향→우하향): 특정 공간 구조에 편향
- 무작위 순서: 모든 가능한 조건부 분포 학습

수학적으로, ARDM 목적함수:

$$\mathbb{E}\_{\sigma \sim U(S_D)} \sum_{k \in \sigma(\geq t)} \log p(x_k | x_{\sigma( < t)})$$

는 모든 가능한 조건부 분포 조합을 동등하게 처리합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a7aceff6-b4a1-4c6c-b692-b91c9a0a1177/2110.02037v2.pdf)

**원리 2: 내재적 앙상블 효과**
BERT와의 연결성:
$$L_t = \frac{1}{D-t+1} \mathbb{E}_{\sigma} [\cdots]$$

실질적으로 D개의 서로 다른 마스킹 구성을 가진 BERT 모델을 훈련하는 것과 동등합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a7aceff6-b4a1-4c6c-b692-b91c9a0a1177/2110.02037v2.pdf)

### 5.2 상향 확대의 일반화 효과
**원리 1: 계층적 생성 구조**
- 최상위 정보(주요 특징) 먼저 생성
- 세부 정보(노이즈 같은) 나중에 생성
- 이는 데이터의 자연스러운 계층 구조와 정렬

**원리 2: 훈련 효율성 불변**
훈련 시 단계 s ∼ U(1, ..., S) 균등 샘플링:
- S개 단계 추가 → 훈련 복잡도 변화 없음
- 단일 단계 s만 최적화하므로
- 따라서 S를 임의로 크게 할 수 있음 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a7aceff6-b4a1-4c6c-b692-b91c9a0a1177/2110.02037v2.pdf)

**원리 3: 구배 신호 개선**
상향 확대로 인해:
- 각 단계의 불확실성이 더 구조화됨
- 초기 단계: 높은 불확실성 (학습 신호 강함)
- 후기 단계: 낮은 불확실성 (미세 조정)

이는 다중 스케일 학습의 효과를 제공합니다.

### 5.3 병렬화의 일반화 성질
**우아한 성능 저하 (Graceful Degradation)**
Equation 4에서:

$$L_t = \mathbb{E}_{\sigma} [\log p(x_{\sigma(t+1)} | x_{\sigma(<t)})] \leq L_{t+1}$$

따라서:
- 단계가 많을수록 성능 향상
- 단계 감소 시 성능도 부드럽게 감소
- 극단적 단계 감소도 합리적 성능 유지 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a7aceff6-b4a1-4c6c-b692-b91c9a0a1177/2110.02037v2.pdf)

예: Text8에서 250 단계 → 20 단계 (12.5배 감소)는 1.43 → 1.51 bpc (5.6% 성능 저하만)

### 5.4 다중 분석: 왜 ARDMs가 더 잘 일반화하나
| 특성 | ARM | D3PM | ARDM |
|------|-----|------|------|
| 순서 고정성 | 필수 | - | 선택사항 |
| 훈련 목적 | 전체 체인 | 단일 단계 | 단일 단계 |
| 아키텍처 제약 | 높음 (인과 마스킹) | 낮음 | 낮음 |
| 구배 신호 다양성 | 낮음 | 중간 | **높음** |
| 계층적 표현 | 불가능 | 고정 | 유연 |
| 병렬화 | 불가능 | 제한적 | **완전** |

***

## 6. 2020년 이후 관련 최신 연구 비교
### 6.1 ARDM 이후의 주요 발전 (2022-2025)
#### A. 확산 모델 확장 (2022-2023)

**DiffusionBERT (2022)** [arxiv](https://arxiv.org/abs/2211.15029)
- 개선점: BERT와 이산 확산 결합, 정보-기반 잡음 스케줄
- ARDMs과의 비교: 
  - BERT 초기화로 수렴 가속화
  - 마스킹-기반 훈련은 ARDM과 유사
  - 조건부 생성에 초점 (ARDMs은 무조건부)

**Continuous Time Framework (Campbell et al., 2022)** [arxiv](https://arxiv.org/abs/2205.14987)
- 개선점: 이산 확산을 연속 시간 마르코프 체인(CTMC)으로 공식화
- ARDMs과의 비교:
  - ARDM은 특수한 CTMC 케이스 (고정 흡수 과정)
  - 더 일반적인 이론 틀 제공하나, 실무 성능은 유사

**DiffuSeq (2022)** [arxiv](https://arxiv.org/abs/2210.08933)
- 개선점: 문장-변환 텍스트 생성, 높은 다양성
- ARDMs과의 비교:
  - DiffuSeq: 조건부 생성 최적화
  - ARDM: 무조건부이나 더 나은 무손실 압축

#### B. 이산 확산의 최신 발전 (2023-2025)

**Glauber Generative Model (2024)** [semanticscholar](https://www.semanticscholar.org/paper/Glauber-Generative-Model:-Discrete-Diffusion-Models-Varma-Nagaraj/a6a02052246fa3fb572ce5f9627c3cf3a64a9dd0)
- 개선점: 이진 분류 기반 이산 확산, 텍스트에서 D3PM 초과 성능
- ARDMs과의 비교:
  - 순서 무관 훈련은 ARDM과 유사
  - 하지만 마스킹이 아닌 이진 분류 사용
  - 텍스트에서 더 나은 성능 보고

**Fast Solvers for Discrete Diffusion (2025)** [arxiv](https://arxiv.org/pdf/2502.00234.pdf)
- 개선점: 고차 알고리즘, O(D³)에서 향상된 효율
- ARDMs과의 비교:
  - ARDM의 동적계획법 병렬화와 유사한 목표
  - 이산 확산의 추론 속도 개선에 초점
  - 단계 수 감소 (ARDMs과 경쟁)

#### C. 순서 무관 모델링의 진화 (2024-2025)

**COrAL: Context-Wise Order-Agnostic Language Modeling (2024)** [arxiv](https://arxiv.org/abs/2410.09675)
- 개선점: LLM 아키텍처 내 반복적 정제, 슬라이딩 블록 복호화
- ARDMs과의 비교:
  - ARDM의 순서 무관 원리 기반으로 구축
  - 추론 속도: 3.9배 가속화 (ARDMs의 병렬화와 유사)
  - 성능: GSM8K에서 4.6% 개선, LogiQA에서 4.0% 개선

**Decoding Order Matters (2025)** [arxiv](https://arxiv.org/html/2601.08450v1)
- 개선점: 적응형 생성 순서 선택, 음성 합성 개선
- ARDMs과의 비교:
  - ARDM: 학습 가능한 순서 아님
  - 이 논문: 역동적 순서 결정 메커니즘
  - 성능: 좌-우 순서보다 우수, ARDM 개념 확장

**Masked Diffusion Models are Secretly Learned-Order ARMs (2025)** [arxiv](https://arxiv.org/html/2511.19152v1)
- 개선점: 마스킹 확산 모델의 자동회귀 해석
- ARDMs과의 비교:
  - ARDM: 고정된 순열에 대한 순서 무관 훈련
  - 이 논문: 학습된 순서 선택 → 더 효율적일 가능성
  - 이론적 통합 시도

### 6.2 종합 성능 비교 표 (2020-2025)
| 모델 | 발표연도 | 주요 특징 | Text | Image | Audio | 무손실 압축 | 병렬화 |
|------|--------|---------|------|-------|-------|-----------|--------|
| **ARDM** | **2022** | 순서 무관, 상향 확대, 병렬화 | 1.43† | 2.64† | 6.29† | **2.71†** | **예** |
| D3PM | 2021 | 이산 확산, 흡수 과정 | - | 3.44 | - | - | 아니오 |
| PixelCNN++ | 2017 | 자동회귀 ARM | - | 2.92 | - | - | 아니오 |
| VDM | 2021 | 변분 확산 | - | **2.49** | - | 2.72 | 아니오 |
| DiffusionBERT | 2022 | BERT + 확산 | **1.22**(조건부) | - | - | - | 중간 |
| DiffuSeq | 2022 | Seq2Seq 확산 | 다양 가능 | - | - | - | 중간 |
| Glauber Model | 2024 | 이진 분류 확산 | 우수 | - | - | - | 부분 |
| COrAL | 2024 | LLM 순서 무관 | 1.39-1.43 | - | - | - | **예** (적응형) |
| Learned-Order ARM | 2025 | 학습된 순서 | 향상** | - | - | - | 부분 |

†: Text8(250 chars), CIFAR-10, SC09, CIFAR-10 이미지당
*: 조건부 생성, **예상 개선 (완전 실험 결과 미발표)

### 6.3 ARDM의 상대적 장점과 한계
**ARDM의 고유 장점**:
1. **무손실 압축**: 비트-후퇴 코딩 불필요, 이미지당 압축 SOTA
2. **단계 효율성**: D3PM 대비 1/4 단계로 유사 성능
3. **훈련 유연성**: 상향 확대 단계 수가 훈련 복잡도에 영향 없음
4. **통합 이론**: ARM과 확산의 완전한 수학적 통합

**ARDM의 제약**:
1. **텍스트에서의 성능**: 단일 순서 ARM (1.35 bpc)에 미치지 못함 (1.43 bpc)
2. **SOTA 이미지 품질**: VDM (2.49) 대비 뒤짐 (2.64)
3. **연속 데이터 미지원**: 이산 변수만 지원
4. **학습 기반 최적화 미흡**: 순서가 전적으로 균등 샘플링

***

## 7. 한계 및 과제
### 7.1 명시적 한계 (논문에서 기술)
**한계 1: 고정 순서 ARM과의 성능 격차** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a7aceff6-b4a1-4c6c-b692-b91c9a0a1177/2110.02037v2.pdf)
- Text8에서 단일 순서 Transformer: 1.35 bpc
- OA-ARDM: 1.43 bpc
- 격차: ~0.08 bpc (6% 성능 손상)
- 원인: 순서 무관 훈련의 상충(trade-off) 

**한계 2: 이산 변수만 지원** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a7aceff6-b4a1-4c6c-b692-b91c9a0a1177/2110.02037v2.pdf)
- 연속 확산 프로세스 이론이 존재하나, ARDM 프레임워크로 직접 확장 어려움
- 이미지와 텍스트에는 이산화(discretization) 필요
- 음성 데이터: 16비트 양자화 (16,000개 클래스) 사용

**한계 3: 언어 모델에서 상향 확대의 비효과** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a7aceff6-b4a1-4c6c-b692-b91c9a0a1177/2110.02037v2.pdf)
- 이미지 (2.64 bpd), 음성 (6.29 bpd): 명확한 개선
- 텍스트: 상향 확대 변형이 기본 OA-ARDM보다 나아지지 않음
- 추측: 텍스트의 언어 구조가 비트 기반 분해와 부정렬

**한계 4: 목적함수 특화** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a7aceff6-b4a1-4c6c-b692-b91c9a0a1177/2110.02037v2.pdf)
- 최적화: 로그-가능도(log-likelihood)에 초점
- 표본 품질이나 다양성: 다른 아키텍처 선택 필요
- 확산 모델(예: DDPM)이 표본 품질에서 우수할 수 있음

### 7.2 이론적 제약
**제약 1: 병렬화의 성능 비용**
Equation 4:
$$L_t \leq L_{t+1}$$

병렬화는 이 부등식을 활용하나, 정보 손실 발생:
- 독립 샘플링: 조건부 의존성 무시
- 최적 병렬화 정책: 손실을 최소화하나 완전히 제거 불가

**제약 2: 동적계획법의 계산 복잡도**
- O(D³) 복잡도
- 음성(D=16,000): ~0.5분
- 매우 고차원 데이터(D > 100,000): 문제 가능성

### 7.3 실무적 과제
**과제 1: 초기화 민감도**
- 마스킹 입력의 정규화 방식 (이미지: 0, 언어: K+1)
- 각 모달리티별 세심한 조정 필요

**과제 2: 확률 교정(Calibration)**
- 병렬화 알고리즘은 L_t 항이 잘 교정되어 있다고 가정
- 훈련 초기 단계에서 오류 가능성

**과제 3: 대규모 확장성**
- Text8: 250자 (제한됨)
- CIFAR-10: 32×32 픽셀 (작음)
- ImageNet 규모 이미지로의 확장 평가 없음

***

## 8. 앞으로의 연구 방향 및 영향
### 8.1 ARDM이 미치는 긍정적 영향
**이론적 영향**:
1. **통합 프레임워크**: ARM과 확산 모델의 이론적 다리 구축
2. **효율성 분석**: 최소 필요 단계 수에 대한 정량적 이해 제공
3. **병렬화 이론**: 확산 모델의 동적계획법 기반 병렬화 개척

**실무적 영향**:
1. **무손실 압축**: 신경망 기반 압축에 새로운 기준 수립
2. **아키텍처 자유도**: 인과 마스킹 제약 제거로 모델 설계 유연성 증대
3. **효율성**: D3PM 대비 4배 단계 감소 달성

### 8.2 후속 연구의 예상 방향
**방향 1: 학습 기반 순서 최적화** (실제 진행)
- **COrAL (2024)**: 슬라이딩 윈도우 내 적응형 순서
- **Learned-Order ARM (2025)**: 신경망 기반 순서 학습
- 예상 효과: ARDM의 고정 균등 순서보다 더 효율적 순서 발견

**방향 2: 연속 데이터 확장**
- **과제**: 현재 이산 변수만 지원
- **기회**: 혼합 이산-연속 모델 개발
  - 예: 텍스트(이산) + 음성 특성(연속)
  - 멀티모달 생성 모델

**방향 3: 조건부 생성 및 편집**
- **DiffusER (2022)**: 편집 기반 생성
- **ARDMs 확장**: 마스킹 기반 조건부 생성
- **응용**: 이미지 인페인팅, 상향식 완성(inpainting, completion)

**방향 4: 대규모 모델로의 확장**
- 현재: Text8 (250자) 수준
- 필요: 완전 언어 모델 규모 (8K+ 시퀀스)
- 도전: O(D³) 동적계획법의 메모리 복잡도

**방향 5: 다중 모달리티 통합**
- **목표**: 단일 ARDM으로 텍스트, 이미지, 음성 처리
- **기회**: 공유 마스킹 프레임워크
- **도전**: 모달리티별 최적 상향 확대 전략

### 8.3 연구 시 고려할 핵심 사항
#### 고려 사항 1: 순서의 중요성
- **인사이트**: ARDM은 순서를 무관하게 훈련하지만, 추론 순서는 여전히 중요
- **실무**: 압축 시 최적 순서 선택 (논문: 훈련 세트에서 몇 순열 평가)
- **미래 연구**: 학습된 또는 적응형 순서 결정 메커니즘

#### 고려 사항 2: 상향 확대의 수동 설계
- **현재 한계**: 비트 상향 확대는 수동으로 정의
- **더 나은 방향**:
  - 학습 가능한 상향 확대 행렬
  - 또는 데이터 기반 계층 구조 발견

#### 고려 사항 3: 성능-효율성 트레이드오프
- **데이터포인트 1**: Text8, 250 단계, 1.43 bpc
- **데이터포인트 2**: Parallelized, 20 단계, 1.51 bpc
- **교훈**: 정확도 5% 손상으로 12.5배 가속화
- **응용 선택**: 실시간 생성 vs. 최대 품질

#### 고려 사항 4: 모달리티 특이성
- **텍스트**: 단일 순서 ARM에 미친다 (언어 모델 사전 학습이 중요)
- **이미지**: SOTA 대비 뒤지나, 압축에서 우수 (이미지 생성 보다 압축이 목표면 적합)
- **음성**: 상향 확대로 큰 이득 (20% 개선)
- **결론**: 문제 정의에 따라 모델 선택 필수

#### 고려 사항 5: 계산 인프라
- **요구사항**:
  - Text8: 1주일 (4 TPUv4)
  - CIFAR-10: 2주일 (8 TPUv4)
  - SC09 음성: 4일 (일반 GPU 충분)
- **고려**: 대규모 데이터 실험 시 분산 훈련 설계 필요

***

## 9. 결론
### 9.1 종합 평가
**Autoregressive Diffusion Models**은 자동회귀 모델과 이산 확산 모델의 장점을 통합하는 원칙적이고 효율적인 일반 목적 생성 모델입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a7aceff6-b4a1-4c6c-b692-b91c9a0a1177/2110.02037v2.pdf)

**주요 성과**:
- ✅ **이론적 우아함**: ARM과 확산 모델의 수학적 동등성 증명
- ✅ **실무적 효율성**: D3PM 대비 4배 단계 감소
- ✅ **응용 우수성**: 무손실 압축에서 SOTA 달성
- ✅ **유연성**: 상향 확대로 다양한 모달리티 지원

**제약 및 한계**:
- ❌ 텍스트: 단일 순서 ARM과의 성능 격차
- ❌ 이미지: VDM 대비 성능 뒤짐 (표본 품질)
- ❌ 연속 데이터: 현재 이산 변수만 지원
- ❌ 확장성: O(D³) 동적계획법의 고차원 한계

### 9.2 ICLR 2022 발표 이후의 영향
ARDMs은 2022년 ICLR 발표 이후 다음과 같은 후속 연구를 촉발했습니다:

1. **순서 무관 학습의 재조명** (COrAL 2024, 학습된 순서 2025)
2. **이산 확산 효율화 연구** (Fast Solvers 2025)
3. **마스킹 기반 생성 통합** (Masked Diffusion-ARM 연결 2025)
4. **멀티모달 생성 확장** (진행 중)

### 9.3 최종 권장 사항
**ARDM을 선택할 시점**:
✓ 무손실 압축 (이미지, 수치 데이터)
✓ 다양한 모달리티의 유연한 처리
✓ 훈련 효율성이 중요한 경우
✓ 생성 순서에 자연스러운 순서가 없는 경우

**다른 모델을 선택할 시점**:
✗ 최고 품질의 표본 생성 (VDM, DDIM 권장)
✗ 언어 모델 (단일 순서 Transformer 권장)
✗ 극도로 큰 시퀀스 (메모리 제약)
✗ 연속 신호만 처리 (Gaussian 확산 권장)

***

## 참고문헌
Hoogeboom, E., Gritsenko, A. A., Bastings, J., Poole, B., van den Berg, R., & Salimans, T. (2022). Autoregressive Diffusion Models. In ICLR 2022. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a7aceff6-b4a1-4c6c-b692-b91c9a0a1177/2110.02037v2.pdf)

 Qiao, Z., et al. (2022). DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models. In ACL 2023. [arxiv](https://arxiv.org/abs/2211.15029)

 Austin, J., Johnson, D. D., Ho, J., Tarlow, D., & van den Berg, R. (2021). Structured Denoising Diffusion Models in Discrete State-Spaces. arXiv:2107.03006. [arxiv](https://arxiv.org/abs/2210.01549)

 Salimans, T., et al. (2017). PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modifications. In ICLR 2017. [arxiv](https://arxiv.org/abs/2208.14699)

 Lin, Z., et al. (2022). DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models. In EMNLP 2022. [arxiv](https://arxiv.org/abs/2210.08933)

 Watson, D., Ho, J., Norouzi, M., & Chan, W. (2021). Learning to Efficiently Sample from Diffusion Probabilistic Models. arXiv:2106.03802. [semanticscholar](https://www.semanticscholar.org/paper/25d3a4e048d0020ba9cffc6442ebd4e7bb548a55)

 Uria, B., Murray, I., & Larochelle, H. (2014). A Deep and Tractable Density Estimator. In ICML 2014. [arxiv](https://arxiv.org/abs/2210.16886)

 Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. In NeurIPS 2020. [arxiv](https://arxiv.org/abs/2205.16007)

 Kingma, D. P., et al. (2021). Variational Diffusion Models. arXiv:2107.00630. [arxiv](https://arxiv.org/abs/2210.12867)

 Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Understanding. In NAACL-HLT 2019. [arxiv](https://arxiv.org/abs/2212.00886)

 Campbell, A., et al. (2022). A Continuous Time Framework for Discrete Denoising Models. arXiv:2205.14987. [arxiv](https://arxiv.org/abs/2205.14987)

 Fast Solvers for Discrete Diffusion Models: Theory and Applications of High-Order Algorithms. arXiv:2502.00234 (2025). [arxiv](https://arxiv.org/pdf/2502.00234.pdf)

 Xie, Y., et al. (2024). Order-Agnostic Language Modeling for Efficient Iterative Refinement. arXiv:2410.09675. [arxiv](https://arxiv.org/abs/2410.09675)

 Varma, V., & Nagaraj, S. (2024). Discrete Diffusion Models via Binary Classification. arXiv:2405.10821. [semanticscholar](https://www.semanticscholar.org/paper/Glauber-Generative-Model:-Discrete-Diffusion-Models-Varma-Nagaraj/a6a02052246fa3fb572ce5f9627c3cf3a64a9dd0)

 Wang, W., et al. (2025). Decoding Order Matters in Autoregressive Models. arXiv:2601.08450. [arxiv](https://arxiv.org/html/2601.08450v1)

 Chen, Y., et al. (2025). Masked Diffusion Models are Secretly Learned-Order Autoregressive Models. arXiv:2511.19152. [arxiv](https://arxiv.org/html/2511.19152v1)

***

## 부록: 핵심 수식 정리
### 확률 모델링 핵심 수식
**표준 자동회귀 모델:**
$$\log p(x) = \sum_{t=1}^{D} \log p(x_t | x_{<t})$$

**순서 무관 자동회귀 모델:**
$$\log p(x) \geq \mathbb{E}_{\sigma \sim U(S_D)} \left[ \sum_{t=1}^{D} \log p(x_{\sigma(t)} | x_{\sigma(<t)}) \right]$$

**ARDM 단계별 목적함수:**
$$\mathcal{L}_t = \frac{1}{D-t+1} \mathbb{E}_{\sigma \sim U(S_D)} \sum_{k \in \sigma(\geq t)} \log p(x_k | x_{\sigma(<t)})$$

**배치 훈련 ELBO:**
$$\mathbb{E}_{t \sim U(1,...,D)} [D \cdot \mathcal{L}_t]$$

### 상향 확대 핵심 수식
**하향 파괴 과정:**
$$x^{(s)} = P^{(s+1)} x^{(S)} = \bar{P}^{(s+1)} x^{(S)}$$

**비트 상향 확대 전이 행렬:**
$$P^{(8+1-s)}_{l,k} = \begin{cases} 1 & \text{if } l = \lfloor k/2^s \rfloor \cdot 2^s \\ 0 & \text{otherwise} \end{cases}$$

### 병렬화 핵심 수식
**병렬 생성 비용:**
$$L_{t,t+k} = k \cdot L_t$$

**성능 단조성:**
$$L_t \leq L_{t+1}$$

이는 병렬화의 비용-편익 분석을 가능하게 합니다.
