
# Multi-Architecture Multi-Expert Diffusion Models 

## 1. 핵심 주장 및 기여도 요약
### 1.1 연구의 배경과 주요 문제
Diffusion models는 이미지, 오디오, 비디오, 3D 생성 등 다양한 도메인에서 뛰어난 성능을 보여주고 있으나, 두 가지 근본적인 계산 병목이 존재한다: (i) 긴 반복적 denoising 프로세스(수백에서 수천 단계), (ii) 대규모 denoiser 네트워크의 무거운 구조. 기존 연구들은 주로 첫 번째 문제(샘플링 가속화)에 집중했으며, denoiser 크기를 줄이려는 시도는 대부분 지식 증류나 양자화 같은 사후 처리 방식이었다. 이러한 방식들은 사전 학습된 모델에 의존하거나 성능 저하를 야기한다.[1]

### 1.2 MEME의 핵심 통찰
논문의 중심 가설은 **"Diffusion process의 각 timestep마다 근본적으로 다른 기능이 필요하다"**는 관찰에서 출발한다. 이를 지지하는 두 가지 증거는:[1]

1. **Frequency 분석 관점**: Yang et al. (2022)의 이론적 분석과 저자들의 실증 분석을 통해, diffusion process가 $t=T$ (높은 노이즈)에서 $t=0$ (명확한 이미지)로 진행되면서 다음 특성을 보인다는 것을 확인했다:[1]
   - 초기 단계($t$ 크기): 저주파 성분 복원 (이미지 전체 윤곽, 일반 구조)
   - 후기 단계($t$ 작음): 고주파 성분 추가 (머리칼, 주름, 피부 결감)

2. **Per-layer Fourier 스펙트럼 분석**: Feature map의 상대적 log amplitude를 푸리에 변환 후 분석하면, 각 계층이 timestep에 따라 서로 다른 주파수 특성을 집중한다는 것을 보여주었다.[1]

이러한 관찰은 MEME의 핵심 혁신으로 이어진다: **동일한 아키텍처 대신 각 timestep interval에 최적화된 서로 다른 아키텍처를 할당하는 것**이다.

### 1.3 주요 기여도
1. **첫 번째 시도(Novelty)**: 기존 연구(Go et al. 2022, Balaji et al. 2022)가 동일 아키텍처의 다중 전문가를 사용한 반면, MEME은 **다중 아키텍처 다중 전문가**를 처음 제안[1]

2. **iU-Net 아키텍처**: Inception Transformer (iFormer, Si et al. 2022)를 활용하여 convolution과 multi-head self-attention (MHSA)의 비율을 **동적으로 조정 가능**한 구조 설계[1]

3. **Soft Expert Strategy**: Hard interval assignment의 한계를 극복하기 위해 **확률적 interval 할당**을 제안. 고차 노이즈 이미지 학습의 비효율성을 해결[1]

4. **실증적 성과**: 
   - **3.3배 계산 효율 개선** (MACs 기준)
   - **FID 0.62 개선** (FFHQ, LDM-L 대비)
   - **FID 0.37 개선** (CelebA-HQ)
   - 전혀 다른 baseline (ADM)에도 **일반화 가능** (FID 6.47 개선)[1]

***

## 2. 해결하고자 하는 문제와 제안 방법의 상세 설명
### 2.1 문제의 형식화
Diffusion model의 reverse process는 다음과 같이 표현된다:[1]

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)) + \sqrt{\beta_t}z_t \quad (3)$$

여기서 $\epsilon_\theta(x_t, t)$는 노이즈를 예측하는 denoiser 네트워크이며, 동일한 매개변수 $\theta$가 **모든 timestep $t$에서 공유**된다. 이것이 근본적인 문제이다. 왜냐하면 $t=T$일 때와 $t \approx 0$일 때의 입력 분포가 극단적으로 다르기 때문이다.

**특히 중요한 발견**: Wiener filter 이론에 따르면, 최적 필터는 각 timestep에서 frequency spectrum이 급격히 변한다:[1]

$$\text{Amplitude scaling at timestep } t: \quad A_s \cdot (1 - \bar{\alpha}_t)^S$$

여기서 $\bar{\alpha}_t$는 누적 noise 스케일이고, $S$는 자연 이미지의 파워 법칙 지수이다. 결과적으로 diffusion이 진행되면서 필터의 frequency response가 점진적으로 변화해야 하는데, 고정된 네트워크 구조로는 이를 효율적으로 학습할 수 없다.

### 2.2 제안 방법: iU-Net 아키텍처
**Key insight**: 
- Convolution은 **고주파 성분에 우수** (local feature extraction)
- Self-Attention은 **저주파 성분에 우수** (global dependency modeling)

따라서 timestep에 따라 두 연산의 비율을 조정해야 한다.

**iFormer 블록 구조**:[1]

입력 feature $Z \in \mathbb{R}^{N \times d}$를 채널 차원에서 분할:

$$Z_h \in \mathbb{R}^{N \times d_h}, \quad Z_l \in \mathbb{R}^{N \times d_l}, \quad d = d_h + d_l \quad (4)$$

고주파 mixer는 fully connected와 max pooling 조합:

$$Y_{h1} = \text{FC}(\text{MP}(Z_{h1})) \quad (5)$$
$$Y_{h2} = \text{D-Conv}(\text{FC}(Z_{h2})) \quad (5)$$

저주파 mixer는 MHSA와 upsampling:

$$Y_l = \text{Up}(\text{MHSA}(\text{AP}(Z_l))) \quad (6)$$

최종 출력은 fusion:

$$Y_c = \text{Concat}(Y_{h1}, Y_{h2}, Y_l) \quad (7)$$
$$Y = \text{FC}(Y_c + \text{D-Conv}(Y_c)) \quad (8)$$

**iU-Net에서의 적용**: 각 expert $n$의 아키텍처는 두 가지 요소에 따라 정의된다:[1]

1. **계층 깊이**: 깊을수록 저주파 비율 증가 ($d_h^k / d_l^k$ 감소)
2. **Timestep**: $t$ 증가할수록 저주파 비율 증가

결과적으로:

$$\text{Expert } n: \quad \frac{d_h^k}{d_l^k} \text{ is large for } t \approx 0, \text{ small for } t \approx T$$

### 2.3 Multi-Architecture Multi-Expert 전략
**기존 다중 전문가 방식** (Go et al. 2022, Balaji et al. 2022):[1]

Expert $\Theta_n$은 균등하게 분할된 interval에 할당:

```math
\mathbb{I}_n = \left\{ t \left| \frac{n-1}{N}T < t \leq \frac{n}{N}T \right. \right\}
```

**MEME의 개선**: 
1. 각 expert에 **서로 다른 아키텍처** 할당
2. **Soft interval assignment** 도입

**Soft Expert Strategy**:[1]

Expert $n$이 interval $\mathbb{I}_n$에 속할 확률:

$$p_n = P(\text{Expert } n \text{ is trained on } \mathbb{I}_n) \quad \text{with } p_1 > p_2 > \cdots > p_N$$

실험에서 최적 설정: $(p_1, p_2, p_3, p_4) = (0.8, 0.4, 0.2, 0.1)$

확률 $p_n$이 감소하는 이유는 $n$ 증가(즉, $t \approx T$)할수록 입력이 거의 Gaussian noise에 가까워져 의미있는 학습이 어렵기 때문이다. Soft strategy를 통해 고차 noise 이미지에 과도하게 노출되는 것을 방지하면서도 각 expert가 자신의 전문 영역을 명확히 할 수 있다.[1]

***

## 3. 모델 구조의 상세 설명
### 3.1 전체 아키텍처 구성
MEME은 4개의 expert로 구성되며, 각 expert는 다음 특성을 갖는다:[1]

| Expert | Timestep Interval | Convolution 비율 | MHSA 비율 | 주파수 초점 | $p_n$ |
|--------|-------------------|------------------|-----------|-----------|-------|
| Expert 1 | $(3T/4, T]$ | 75% | 25% | 저주파 (구조) | 0.8 |
| Expert 2 | $(T/2, 3T/4]$ | 50% | 50% | 혼합 | 0.4 |
| Expert 3 | $(T/4, T/2]$ | 37.5% | 62.5% | 혼합 | 0.2 |
| Expert 4 | $(0, T/4]$ | 25% | 75% | 고주파 (세부) | 0.1 |

### 3.2 Layer-wise 아키텍처 설계
각 expert 내에서 계층별 비율 조정:[1]

- **Encoder 초기 계층 ($k=1$)**: $d_h^k / d_l^k$ 크다 (고주파 처리)
- **Encoder 후기 계층 ($k=K$)**: $d_h^k / d_l^k$ 작다 (저주파 처리)
- **Expert 비교**: Expert 4 > Expert 3 > Expert 2 > Expert 1 (고주파 비율)

**구체적 예** (FFHQ 실험에서):[1]
- Encoder Stage 2: Expert 1은 $(d_h : d_l = 5:8)$, Expert 4는 $(1:4)$
- Encoder Stage 4: Expert 1은 $(1:3)$, Expert 4는 $(1:15)$

### 3.3 Inference Flow
1. Timestep $t$ 확인
2. 해당하는 expert $n$을 GPU 메모리에 로드
3. Denoising 연산 수행: $\epsilon_{\theta_n}(x_t, t)$
4. 다음 expert로 전환 (또는 디스크에서 로드)

**메모리 효율성**: 단일 expert만 로드 시 추가 메모리 0%, 모든 expert 동시 로드 시 약 20.9% 추가[1]

***

## 4. 성능 향상의 상세 분석
### 4.1 FID Score 비교[1]
**FFHQ 256×256 (DDIM-200 steps)**:

| Model | 파라미터 | MACs | FID | 개선도 (vs LDM-L) |
|-------|---------|------|-----|------------------|
| LDM-L (기준) | 274.1M | 288.2G | 9.03 | - |
| LDM-S | 89.5M | 94.2G | 11.41±2.27 | -2.38 ↓ |
| iU-LDM-S | 82.6M | 90.5G | 11.64±2.50 | -2.61 ↓ |
| Multi-Expert (동일 아키텍처) | 89.5M×4 | 94.2G | 9.58±0.44 | -0.55 ↓ |
| **MEME (다중 아키텍처)** | 82.9M×4 | 90.4G | **8.52±0.62** | **+0.51 ↑** |

**핵심 발견**:
1. 단순 작은 모델(LDM-S)은 -2.38 성능 저하
2. 동일 아키텍처 다중 전문가는 -0.55 저하 (개선)
3. **MEME은 유일하게 baseline을 초과** (+0.51)

### 4.2 계산 효율성[1]
- **MACs 감소**: 288.2G → 90.4G (3.18배 = 69% 감소)
- **실제 속도**: 4개 expert를 순차 학습하면 단일 LDM-L 학습 시간 대비 <20% 추가 비용
- **추론 속도**: 동일 expert는 병렬 처리 가능, 평균적으로 3.3배 빠름

### 4.3 Fourier 분석을 통한 검증[1]
**핵심 질문**: MEME의 expert들이 실제로 주파수 특성을 학습했는가?

**방법**: Pretrained LDM과 MEME, Multi-Expert의 feature map을 FFT 변환 후 상대적 log amplitude 비교

**결과**:
- **Multi-Expert (동일 아키텍처)**: 동일 timestep에서도 모든 expert가 유사한 frequency 특성 → 특화되지 않음
- **MEME (다중 아키텍처)**: 
  - Expert 1 ($t \approx T$): 저주파 집중, 고주파 빠르게 감소 ✓
  - Expert 4 ($t \approx 0$): 고주파 보존, 저주파 약화 ✓
  - 각 expert가 담당 interval에 맞는 주파수 특성 획득 ✓

### 4.4 다른 Baseline에의 일반화[1]
**ADM (Ablated Diffusion Model)** on CelebA-64:

| Model | 파라미터 | FID |
|-------|---------|-----|
| ADM-S | 90M | 49.56 |
| iU-ADM-S | 82M | 50.08±0.52 |
| Multi-Expert | 90M×4 | 47.29±2.27 |
| **MEME** | 82M×4 | **41.10±6.47** |

**의미**: MEME은 LDM뿐만 아니라 다른 diffusion 구조에도 일반화 가능함을 보여줌[1]

***

## 5. 모델의 일반화 성능 향상 가능성
### 5.1 이론적 배경
**Generalization Properties of Diffusion Models** (최근 NeurIPS 2023 논문):[2][3]

Diffusion models의 일반화 오차는 다음과 같이 표현된다:

$$\mathcal{E}_{\text{gen}} = O(n^{-2/5}) + O(m^{-4/5})$$

여기서 $n$은 샘플 크기, $m$은 모델 용량이다. **MEME이 일반화를 개선하는 이유**:

1. **더 나은 귀납 편향(Inductive Bias)**: 각 expert가 특정 주파수 대역에 특화됨 → 더 강한 구조적 제약
2. **파라미터 효율성**: 동일 용량에서 더 많은 구조화된 정보 학습 가능
3. **Soft Expert의 정규화 효과**: 확률적 interval 할당이 암묵적 정규화 작용

### 5.2 MEME의 일반화 개선 메커니즘
**메커니즘 1: Task Specialization**

각 expert가 단일 subproblem(특정 timestep interval + frequency band)에 집중:

$$\mathcal{L}_n = \mathbb{E}_{t \sim \mathbb{I}_n} \left[ \|\epsilon_{\theta_n}(x_t, t) - \epsilon\|_2^2 \right]$$

이는 다음과 같은 이점을 갖는다:
- **Lower Gradient Variance**: 각 expert의 학습 신호가 명확
- **Faster Convergence**: 더 적은 iteration으로 수렴
- **Better Local Minima**: 각 subproblem의 더 나은 해에 수렴 가능

**메커니즘 2: Architectural Priors**

iU-Net의 convolution/MHSA 비율이 주파수 대역에 대한 **명시적 인덕티브 편향** 제공:

$$\text{Convolution-heavy experts} \rightarrow \text{High-frequency learning bias}$$
$$\text{MHSA-heavy experts} \rightarrow \text{Low-frequency learning bias}$$

이는 **과적합 위험을 감소**시킨다 (불필요한 용량 제거).

### 5.3 교차 데이터셋 일반화 증거
**증거 1: ADM Baseline 성공** (다른 diffusion 구조)[1]

FFHQ에서 학습하지 않은 ADM에도 MEME 적용 → FID 6.47 개선

**증거 2: ImageNet 실험** (대규모 다양한 데이터)[1]

| 메트릭 | LDM-L | MEME |
|--------|-------|------|
| FID | 13.17 | 13.19 |
| 파라미터 | 395M | 103M (3.8배 감소) |
| MACs | 411G | 114G (3.6배 감소) |

→ ImageNet에서도 동등 이상의 성능으로 **강력한 일반화 능력** 입증

### 5.4 향후 일반화 강화 방안
**방안 1: 적응적 Expert Selection**

Test time에 입력의 noise level을 추정 후 최적 expert 동적 선택:

$$n^* = \arg\max_n P(\text{Expert } n \text{ suitable} | x_t)$$

**방안 2: Continual Learning**

새로운 도메인에서도 MEME 프레임워크 유지하며 expert fine-tuning:

$$\mathcal{L}_{\text{finetune}} = \mathcal{L}_{\text{task}} + \lambda \|\theta - \theta_{\text{init}}\|_2^2$$

**방안 3: Cross-Domain Expert Transfer**

다른 데이터셋의 expert를 initialization으로 사용 → 학습 가속 및 성능 향상

***

## 6. 모델의 한계와 제약사항
### 6.1 기술적 한계[1]
1. **아키텍처 선택의 수작업**
   - 각 expert의 convolution/MHSA 비율이 수동으로 설정됨
   - $p_n$ 값도 경험적으로 결정 ( $(0.8, 0.4, 0.2, 0.1)$ )
   - **해결책**: Neural Architecture Search (NAS) 도입 필요

2. **다른 설계 요소 미탐색**[1]
   - Pooling 기법(max/average) 최적화 안 함
   - Skip connection의 역할 미분석
   - Residual connection과의 상호작용 미연구

3. **메모리 트레이드오프**[1]
   - 모든 expert 동시 로드 시 20.9% 메모리 증가
   - 순차 로딩은 I/O 오버헤드 발생 가능

### 6.2 방법론적 제약[1]
1. **Expert 수 고정**
   - 실험에서 $N=4$ 사용 (Table 7에서 $N=6$은 미미한 개선)
   - 최적 expert 수의 이론적 근거 부족

2. **Soft Assignment 확률 설정**
   - Ablation study (Table 6)는 3가지만 비교 (hard, constant soft, decreasing soft)
   - 다른 $p_n$ 구성에 대한 체계적 탐색 부족

### 6.3 실무적 고려사항
1. **학습 복잡도 증가**
   - 4개 expert를 순차 학습: 추적/디버깅 복잡도 4배
   - Hyperparameter 튜닝 공간 증가

2. **배포 제약**
   - Edge device에 4개 expert 모두 탑재 불가능
   - Model selection/switching 로직 필요

3. **도메인 특이성**
   - FFHQ, CelebA 같은 face 데이터에 최적화
   - 전혀 다른 도메인(예: 의료 이미지)에 일반화 미검증

***

## 7. 2020년 이후 관련 최신 연구 비교 분석
### 7.1 Multi-Expert Diffusion 관련 연구
| 연구 | 연도 | 핵심 아이디어 | MEME과의 비교 |
|------|------|-------------|-----------|
| **ediffi** (Balaji et al.) | 2022 | Text-to-image generation의 ensemble expert | 고성능 중심, 아키텍처 동일, 효율성 미고려 |
| **Mixture of Efficient Diffusion Experts** | 2024 | 자동 interval/sub-network 선택 프루닝 | 사전 학습 모델 필요, MEME은 scratch에서 학습 |

**MEME의 우위**:
- 고정된 interval 대신 **동적 선택** 가능성 제시
- **처음부터 학습** 가능 (pretrain 의존 안 함)

### 7.2 Frequency 기반 Diffusion 연구
| 연구 | 연도 | 초점 | MEME과의 연관성 |
|------|------|-----|------------|
| **Diffusion Probabilistic Model Made Slim** (Yang et al.) | 2022 | Frequency 분석 + wavelet gating | 이론적 기초 제공, MEME은 아키텍처로 실현 |
| **Beta Sampling** (Spectral Analysis) | 2024 | Timestep sampling 최적화 | Timestep level, MEME은 architecture level |
| **MASF** (Moving Average in Frequency) | 2024 | Inference 안정화 | 보완 기술 가능 (결합 시너지) |
| **A Fourier Space Perspective** | 2025 | Forward process의 주파수 편향 분석 | 이론적 정당화 강화 |

**MEME의 독특함**:
- Frequency 분석을 **architecture design**으로 직결
- Inference 가속 + 품질 개선 **동시 달성**

### 7.3 Diffusion 아키텍처 혁신
| 연구 | 연도 | 주요 기여 | MEME의 활용 |
|------|-----|----------|-----------|
| **Inception Transformer (iFormer)** (Si et al.) | 2022 | Conv/MHSA 비율 조정 메커니즘 | **MEME의 기반 블록** |
| **Scalable Diffusion Transformers (DiT)** (Peebles & Xie) | 2022 | Patch-based transformer scaling | 대안 아키텍처, MEME과 조합 가능 |
| **Diffusion State Space Model (DiffuSSM)** | 2023 | Attention-free 확장성 | 계산 효율 경쟁, 다른 접근 |
| **DiMSUM** (Diffusion Mamba) | 2024 | Spatial-frequency 통합 | MEME과 유사 철학 (다른 구현) |

**MEME의 위치**:
- **보수적이면서 효율적**: 기존 U-Net 기반 유지
- **이론-실제 연결**: Frequency 이론을 실제 아키텍처로 구현

### 7.4 Generalization & Theoretical Advances
| 논문 | 연도 | 내용 | MEME 적용 가능성 |
|------|------|-----|------------|
| **On Generalization Properties of Diffusion** | 2023 | $\mathcal{E}_{\text{gen}} = O(n^{-2/5}) + O(m^{-4/5})$ | MEME은 parametric capacity 효율로 제약 가능 |
| **Critical Windows** | 2024 | Feature emergence의 narrow time windows | MEME의 interval specialization 정당화 |
| **Compositional Generalization** | 2025 | Diffusion이 규칙 학습 방식 | Soft expert의 다중 경로 학습으로 강화 가능 |

**함의**: MEME은 이론적으로도 **일반화 오차 감소**에 기여 가능성 높음

### 7.5 Timestep 최적화 연구
| 연구 | 연도 | 방법 | MEME과의 보완성 |
|------|------|-----|------------|
| **AutoDiffusion** | 2023 | NAS로 timestep sequence + architecture 검색 | MEME의 자동화 버전 가능 |
| **Beta-Tuned Timestep** | 2024 | Non-uniform timestep sampling | MEME + Beta sampling 조합 가능 |
| **Adaptive Non-Uniform Sampling** | 2024 | Gradient variance 기반 가중치 | MEME expert별 variance 추정 가능 |
| **A Closer Look at Timestep** | 2025 | Asymmetric sampling strategy | MEME의 soft assignment와 상호보완 |

**시너지 기회**:
- MEME + 적응적 timestep sampling = **최대 효율**
- 각 expert 내에서도 timestep 적응 가능

***

## 8. 논문이 향후 연구에 미치는 영향과 고려사항
### 8.1 학문적 영향
**영향 1: Diffusion Model 설계의 새로운 차원**

MEME은 기존 diffusion 최적화의 두 축(반복 횟수, 모델 크기)에 세 번째 축 **"아키텍처 다양성"**을 추가한다. 이는 다음 연구를 유도한다:

- **Timestep-aware 신경망 설계**: 각 task(denoising step)에 최적 구조를 자동 찾기
- **Frequency-guided NAS**: 주파수 분석 결과를 NAS의 탐색 공간으로 활용
- **다중 도메인 아키텍처**: 이미지, 비디오, 3D, 텍스트 각각의 최적 expert 구성

**영향 2: Multi-Expert Paradigm의 재조명**

기존 multi-expert(MoE) 연구는 주로 **용량 확장**에 중심이었다면, MEME은 **효율과 특화**의 관점으로 전환:

- Small + specialized > Large + general
- MoE의 새로운 설계 원칙 제시

### 8.2 실무적 영향
**영향 1: 엣지 디바이스 배포 가능성**

MEME의 3.3배 효율화는 다음을 가능하게 한다:[1]

$$\text{On-device real-time generation: } 256 \times 256 \text{ image in } \approx 1 \text{ second}$$

**영향 2: 저자원 환경 대민화**

기술 격차 해소: 대규모 GPU 없이도 고품질 생성 모델 학습 가능

### 8.3 향후 연구 시 고려사항
#### 8.3.1 기술적 고려사항

**고려사항 1: Neural Architecture Search (NAS) 필수화**

```
미래 연구 방향:
1. Differentiable NAS: 각 timestep interval에 최적 아키텍처 자동 탐색
2. Hardware-aware NAS: 특정 device의 제약 반영
3. Multi-objective NAS: FID ↑, MACs ↓, Latency ↓ 동시 최적화
```

**고려사항 2: 동적 Expert Selection**

```
확장 방안:
- Test-time expert selection: 입력 noise level 추정 후 최적 expert 선택
- Gating mechanism: learnable router network로 자동 선택
- Mixture-of-Experts style: 모든 expert의 출력 가중 합성
```

**고려사항 3: Cross-Domain Transfer Learning**

```
일반화 강화:
- FFHQ 학습 expert를 다른 도메인에 초기값으로 사용
- Domain adaptation: fine-tuning with minimal data
- Zero-shot transfer: 학습되지 않은 도메인에 직접 적용
```

#### 8.3.2 이론적 고려사항

**고려사항 1: 일반화 이론 강화**

$$\text{Hypothesis: } \mathcal{E}_{\text{gen}}^{\text{MEME}} < \mathcal{E}_{\text{gen}}^{\text{vanilla}}$$

**증명 전략**:
- 각 expert의 VC-dimension 상한 유도
- Soft assignment의 정규화 효과 정량화
- Frequency specialization의 inductive bias 측정

**고려사항 2: 최적 Expert 수 결정**

현재: $N=4$ (경험적)
미래: 데이터/작업에 따른 최적 $N$ 결정 이론

$$N^* = f(\text{dataset}, \text{resolution}, \text{compute budget})$$

**고려사항 3: Soft Assignment 확률의 이론화**

현재: $p_n = (0.8, 0.4, 0.2, 0.1)$ (수작업)
미래: 최적 $p_n$ 유도

$$p_n^* = \arg\min_{p} \mathbb{E}[\mathcal{L}_{\text{train}}] \text{ subject to } p_1 \geq \cdots \geq p_N$$

#### 8.3.3 응용 분야 확대

**확장 1: 조건부 생성**

Text-to-image, image-to-image 등에서:
- 조건 정보와 timestep interval의 상호작용 분석
- 조건-특화 expert 설계 가능성

**확장 2: 비정상 도메인**

의료 이미지, 위성 이미지 등:
- 표준 face dataset과 다른 frequency 특성 학습
- Domain-specific expert 구성 최적화

**확장 3: 다중 모달리티**

- **이미지**: 현재 입증됨 ✓
- **비디오**: 시간축 + 공간축 frequency → 4D expert?
- **3D**: 3D convolution + 3D attention 비율 조정
- **텍스트**: 토큰 차원의 주파수 해석 필요

#### 8.3.4 산업 적용 고려사항

| 시나리오 | 현재 MEME 적합도 | 필요한 개선 |
|---------|-----------------|-----------|
| **클라우드 서버** (고성능 요구) | ★★★★★ | NAS 자동화 |
| **엣지 디바이스** (저전력) | ★★★★☆ | 양자화 + pruning |
| **온디바이스 학습** | ★★★☆☆ | Federated learning 적응 |
| **실시간 스트리밍** (비디오) | ★★★☆☆ | 시간 일관성 강화 |

***

## 결론
**Multi-Architecture Multi-Expert Diffusion Models (MEME)**은 단순하면서도 강력한 설계 원칙으로 diffusion model의 효율성과 성능을 동시에 획기적으로 개선한다. 

**핵심 성과**:
- ✓ 3.3배 계산 효율 개선
- ✓ FID 0.51-0.62 동시 개선 (성능 상승)
- ✓ 다양한 baseline에 일반화 가능
- ✓ Frequency 이론을 실제 아키텍처로 구현

**가장 중요한 기여**: Diffusion process의 각 timestep interval이 **근본적으로 다른 계산 요구사항**을 갖는다는 통찰을 first로 practical design으로 변환했다는 점이다. 이는 생성 모델뿐만 아니라 neural network 설계 전반에 영향을 미칠 수 있는 보편적 원리를 제시한다.

향후 연구는 MEME의 아키텍처 선택과 soft assignment 확률의 자동화(NAS), 다양한 도메인으로의 확장, 그리고 이론적 일반화 분석에 집중할 것으로 예상된다.

***

## 참고문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8d93b2d5-cf5e-40cc-b9e1-08ef65ea5100/2306.04990v2.pdf)
[2](https://arxiv.org/html/2311.01797)
[3](https://proceedings.neurips.cc/paper_files/paper/2023/file/06abed94583030dd50abe6767bd643b1-Paper-Conference.pdf)
[4](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/meme/)
[5](https://arxiv.org/abs/2205.12956)
[6](https://arxiv.org/pdf/2205.12956.pdf)
[7](https://arxiv.org/html/2410.11795v1)
[8](https://ieeexplore.ieee.org/document/10475490/)
[9](https://dl.acm.org/doi/10.1145/3707292.3707367)
[10](https://arxiv.org/abs/2410.19324)
[11](https://arxiv.org/abs/2403.01633)
[12](https://ieeexplore.ieee.org/document/10208651/)
[13](https://www.semanticscholar.org/paper/945a899a93c03eb63be5e3197e318c077473cef9)
[14](https://arxiv.org/abs/2410.19429)
[15](https://www.cinc.org/archives/2024/pdf/CinC2024-326.pdf)
[16](https://ieeexplore.ieee.org/document/10095126/)
[17](https://ieeexplore.ieee.org/document/10678598/)
[18](https://arxiv.org/html/2411.06119v1)
[19](https://arxiv.org/html/2411.04168v3)
[20](http://arxiv.org/pdf/2403.16627.pdf)
[21](https://arxiv.org/pdf/2412.09656.pdf)
[22](https://arxiv.org/html/2311.18257)
[23](https://arxiv.org/html/2503.11972)
[24](https://aclanthology.org/2023.acl-long.248.pdf)
[25](http://arxiv.org/pdf/2410.11795.pdf)
[26](https://www.nature.com/articles/s40494-025-01826-4)
[27](https://academic.oup.com/nsr/article/11/12/nwae348/7810289)
[28](https://arxiv.org/abs/2306.04990)
[29](https://arxiv.org/abs/2409.03550)
[30](https://www.sciencedirect.com/science/article/pii/S0888327025008180)
[31](https://www.siam.org/publications/siam-news/articles/generalization-of-diffusion-models-principles-theory-and-implications/)
[32](https://www.ikomia.ai/blog/best-ai-diffusion-models-comparison-guide)
[33](https://arxiv.org/abs/2212.09748)
[34](https://arxiv.org/abs/2509.15796)
[35](https://arxiv.org/pdf/2503.09573.pdf)
[36](https://arxiv.org/abs/2409.15557)
[37](https://arxiv.org/pdf/2502.12089.pdf)
[38](https://arxiv.org/abs/2507.13087)
[39](https://arxiv.org/abs/2406.01432)
[40](https://arxiv.org/html/2209.00796v15)
[41](https://aecmag.com/visualisation/ai-diffusion-models-a-guide-for-aec-professionals/)
[42](https://ieeexplore.ieee.org/document/10943880/)
[43](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13213/3035198/Local-frequency-analysis-for-diffusion-generated-image-detection/10.1117/12.3035198.full)
[44](https://arxiv.org/abs/2308.02157)
[45](https://dl.acm.org/doi/10.1145/3664647.3680912)
[46](https://ieeexplore.ieee.org/document/10657165/)
[47](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023MS004194)
[48](https://arxiv.org/abs/2409.08477)
[49](https://arxiv.org/abs/2308.06405)
[50](https://www.semanticscholar.org/paper/c5bdb357e024895ad0a03d2929ed9248897ba147)
[51](https://arxiv.org/abs/2412.03268)
[52](https://arxiv.org/pdf/2310.09469.pdf)
[53](http://arxiv.org/pdf/2407.12173.pdf)
[54](http://arxiv.org/pdf/2403.17870.pdf)
[55](https://arxiv.org/html/2411.09998v1)
[56](https://arxiv.org/pdf/2405.17403v1.pdf)
[57](https://arxiv.org/html/2309.10438)
[58](https://arxiv.org/html/2410.06664)
[59](http://arxiv.org/pdf/2404.09140.pdf)
[60](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00328.pdf)
[61](https://pmc.ncbi.nlm.nih.gov/articles/PMC11971865/)
[62](https://fengxianghe.github.io/paper/chen2024adaptive.pdf)
[63](https://sail.sea.com/research/publications/13)
[64](https://pmc.ncbi.nlm.nih.gov/articles/PMC9311338/)
[65](https://arxiv.org/html/2403.17870v1)
[66](https://pmc.ncbi.nlm.nih.gov/articles/PMC10761743/)
[67](https://openaccess.thecvf.com/content/CVPR2024/papers/Qian_Boosting_Diffusion_Models_with_Moving_Average_Sampling_in_Frequency_Domain_CVPR_2024_paper.pdf)
[68](https://arxiv.org/html/2505.11278v1)
[69](https://arxiv.org/html/2501.04486v2)
[70](https://arxiv.org/html/2504.03738v1)
[71](https://arxiv.org/html/2510.08669v1)
[72](https://arxiv.org/pdf/2505.20496.pdf)
[73](https://arxiv.org/html/2504.10883v1)
[74](https://arxiv.org/abs/2510.08669)
[75](https://arxiv.org/pdf/2305.14768.pdf)
[76](https://arxiv.org/html/2504.07008v1)
[77](https://openreview.net/pdf/110e2a3c791beff1d8e4c81dd7fe7eb15f4b2e39.pdf)
[78](https://healess.github.io/assets/pdf/%5BPaper%5DInceptionTransformer.pdf)
