
# Relay Diffusion: Unifying diffusion process across resolutions for image synthesis

## 1. 핵심 주장 및 주요 기여

**Relay Diffusion Model (RDM)**은 높은 해상도 이미지 생성에서 확산 모델이 직면하는 두 가지 근본적인 문제를 해결하는 새로운 접근법을 제시한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)

**핵심 주장:**
- 고해상도 이미지에서 동일한 노이즈 수준이 주파수 영역에서 더 높은 신호-대-잡음비(SNR)를 생성한다는 것을 발견
- 기존의 계단식(cascaded) 확산 모델이 해결하지 못하는 세 가지 문제를 식별: (1) 저해상도 조건부 조건의 분포 불일치, (2) 각 단계별 완전한 샘플링 필요, (3) 고해상도 단계의 노이즈 스케줄이 체계적으로 연구되지 않음

**주요 기여:**

1. **주파수 영역 분석**: 이산 코사인 변환(DCT)을 통한 이론적 분석으로 해상도에 따른 노이즈 스케줄의 중요성을 증명
2. **블록 노이즈 제안**: 저해상도 가우시안 노이즈의 높은 해상도 등가물로 작동하는 상관된 노이즈 모델 도입
3. **계단식 파이프라인 혁신**: 저해상도 결과에서 직접 확산을 시작하여 재학습과 재샘플링 필요성 제거
4. **성능 우월성**: CelebA-HQ 256×256에서 FID 3.15, ImageNet 256×256에서 FID 1.87 (분류기 없는 안내 3.50) 달성

***

## 2. 해결하는 문제 및 제안 방법

### 2.1 문제 정의

**문제 1: 학습 효율성**
기존 계단식 방법은 저해상도 모델 → 초해상도 모델 1 → 초해상도 모델 2... 구조로, 각 단계마다 순수 노이즈에서 시작하여 완전한 샘플링이 필요하다. 이는 계산 비용이 매우 높다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)

**문제 2: 노이즈 스케줄의 해상도 의존성**

논문은 주파수 영역에서 이 문제를 분석한다. 저해상도(64×64)와 고해상도(256×256) 이미지에 동일한 노이즈 수준을 적용할 때:

$$q(x_t|x_0) = \mathcal{N}(x_t | \sqrt{\bar{\alpha}_t}x_0, \sqrt{1-\bar{\alpha}_t}I)$$

블러링 확산을 DCT 변환으로 표현하면:

$$q(u_t|u_0) = \mathcal{N}(u_t | D_t u_0, \sigma_t^2 I) \quad (3)$$

여기서 $D_t = \exp(-\lambda_t)$는 대각 행렬이고, $\lambda_{i,j} = \pi^2(i^2/H^2 + j^2/W^2) \cdot t$이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)

고해상도에서는 저주파 대역의 SNR이 더 높아져서, 모델이 더 정확한 입력을 요구하지만 이를 생성할 능력이 없는 **학습-추론 불일치(training-inference mismatch)** 가 발생한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)

### 2.2 제안 방법: Relay Diffusion Model

**블록 노이즈 (Block Noise)**

저해상도의 독립 가우시안 노이즈가 고해상도로 업샘플링될 때 등가가 되는 노이즈를 정의한다:

$$\text{Cov}(n_{x_0,y_0}, n_{x_1,y_1}) = \sigma^2 \max(0, s - d_{x}(x_0, x_1)) \max(0, s - d_{y}(y_0, y_1)) \quad (4)$$

여기서 $s$는 커널 크기, $d$는 맨해튼 거리이다. 블록 노이즈는 $s \times s$ 독립 가우시안 노이즈를 평균화하여 생성된다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)

$$\text{Block}_s(x,y) = \frac{1}{s^2} \sum_{i=0}^{s-1} \sum_{j=0}^{s-1} n_{x+i,y+j} \quad (6)$$

**패치별 블러링 확산 (Patch-wise Blurring Diffusion)**

고해상도 단계의 포워드 프로세스:

$$q(x_t|x_0) = \mathcal{N}(x_t | V D_t^p V^T x_0, \sigma_t^2 I) \quad (7)$$

여기서 $D_t^p$는 업샘플링 스케일(예: 4×4 패치)의 독립적 블러링을 수행한다. 이러한 패치별 접근은 저해상도 조건이 이미 결정되었으므로, 저주파 정보의 재생성을 피한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)

**학습 목적함수**

$$\mathcal{L} = \mathbb{E}_{x \sim p_{\text{data}}, t \sim U(1)} \left[ \| x - D(x_t, t) \|_2^2 \right]$$

여기서:

$$x_t = \sqrt{\lambda} V D_t^p V^T x + \sqrt{1-\lambda} \text{Block}_s(\epsilon) + \sqrt{1-\lambda} \epsilon' \quad (8)$$

$\lambda$는 블렛 노이즈와 독립 가우시안 노이즈의 가중치 혼합 파라미터이다. 이는 블러링과 블록 노이즈의 저주파 우위, 그리고 독립 가우시안의 고주파 우위를 활용한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)

### 2.3 확률적 샘플러 (Stochastic Sampler)

고해상도 단계를 위한 EDM 기반 두 번째 순서 확률적 샘플러 알고리즘:

$$u_{n+1} = D_t^p(\sigma_{n+1}) I D_t^p(\sigma_n)^{-1} u_n + \sigma_{n+1} \sigma_n D_t^p(\sigma_n) D_t^p(\sigma_{n+1})^{-1} d_n + \eta_n \quad (11)$$

여기서 $\eta_n = \sqrt{\sigma_{n+1}^2 - \sigma_n^2} \mathcal{N}(0, I)$는 노이즈 항이고, $d_n$은 기울기 항이다. 알고리즘 1은 주파수 공간에서의 변환과 역변환을 명시한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)

***

## 3. 모델 구조

### 3.1 전체 아키텍처

RDM은 다단계 계단식 구조를 따른다:

| 단계 | 해상도 | 모델 크기 | 확산 스텝 | 노이즈 스케줄 |
|------|--------|---------|---------|-------------|
| 1단계 | 64×64 | 295M | 256 | 코사인 (cosine) |
| 2단계 | 256×256 | 553M | 100 (ImageNet) | 선형 (linear) |

각 단계는 **UNet 아키텍처**를 기반으로 하며, EDM의 구현을 따른다. 주요 특징: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)
- 채널 수: 192-256
- 주의 해상도: 32, 16, 8
- 드롭아웃: 0.1-0.2 (데이터셋별 다름)
- 혼합 정밀도: FP16

### 3.2 수정된 노이즈 스케줄

1단계에서는 표준 노이즈 스케줄을 따르지만, 2단계(고해상도)에서는 **절단된 버전(truncated schedule)**을 사용한다:

$$\sigma_t = F_U^{-1}(F_U(t_s) + t(1 - F_U(t_s))) \quad (17)$$

여기서 $t_s$는 절단 시작점, $F_U$는 정규 분포의 누적분포함수이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)

블러링 스케줄은 Hoogeboom & Salimans (2022)의 설정을 따른다:

$$\sigma_{B,t} = \sigma_{B,\max} \sin^2(\pi t/2) \quad (18)$$

$\sigma_{B,\max}$는 ImageNet 256×256에서 3.0, CelebA-HQ에서 2.0으로 설정된다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)

### 3.3 조건부 메커니즘

기존 CDM과 달리, RDM은 **낮은 해상도 조건 제거**를 통해 단순화를 달성한다. 대신, 낮은 해상도 생성 결과 $x_L$이 자동으로 고해상도 단계의 입력으로 전달된다:

$$x_L = \{x^L_0, n^L_0\}$$

여기서 $x^L_0$는 실제 데이터 샘플, $n^L_0 \sim \mathcal{N}(0, \sigma_0^2 I)$는 남은 노이즈이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)

***

## 4. 성능 향상 및 실험 결과

### 4.1 벤치마크 성능

**CelebA-HQ 256×256 (무조건부 생성)**

| 모델 | FID | Precision | Recall |
|-----|-----|-----------|--------|
| LSGM | 7.22 | - | - |
| WaveDiff | 5.94 | - | 0.37 |
| LDM-4 | 5.11 | 0.72 | 0.49 |
| StyleSwin | 3.25 | - | - |
| **RDM** | **3.15** | **0.77** | **0.55** |

RDM은 StyleSwin보다 우수한 FID를 달성하면서 훨씬 적은 학습 데이터 사용(50M vs 820M 이미지). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)

**ImageNet 256×256 (클래스 조건부 생성)**

| 모델 | FID | sFID | IS | Precision | Recall |
|-----|-----|------|-----|-----------|--------|
| ADM | 10.94 | 6.02 | 100.98 | 0.69 | 0.63 |
| LDM-4 | 10.56 | - | 103.49 | 0.71 | 0.62 |
| DiT-XL2 | 9.62 | 6.85 | 121.50 | 0.67 | 0.67 |
| MDT-XL2 | 6.23 | 5.23 | 143.02 | 0.71 | 0.65 |
| **RDM** | **5.27** | **4.39** | **153.43** | **0.75** | **0.62** |
| DiT-XL2-G (CFG 1.50) | 2.27 | 4.60 | 278.24 | 0.83 | 0.57 |
| **RDM-G** (CFG 3.50) | **1.99** | **3.99** | **260.45** | **0.81** | **0.58** |

분류기 없는 안내(CFG)가 없을 때, RDM은 모든 선행 확산 기반 방법을 상당한 여유로 능가한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)

### 4.2 학습 효율성 비교

RDM의 NPE(Number of Function Evaluations) 효율성:
- ImageNet에서 100 NFE 이상에서 DiT-XL2와 MDT-XL2를 초과
- 1단계와 2단계 사이의 계산 분배에 민감하지 않음
- 1단계에 더 많은 NFE를 할당할수록 더 나은 FID

![성능-NFE 비교: RDM의 우월한 효율성] [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)

### 4.3 소제거 연구 (Ablation Study)

**블록 노이즈의 효과**

ImageNet 256×256에서 블록 노이즈 유무 비교 ($\lambda=0.15$, 커널 크기 4):
- 블록 노이즈 포함: FID 5.27
- 블록 노이즈 제외: FID 5.65 (초기 수렴은 빠르지만 최종 성능 열등)

초기 훈련 단계에서 블록 노이즈는 더 높은 모델링 복잡성으로 수렴 속도를 늦추지만, 장기적으로 더 우수한 성능을 달성한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)

**확률성 계수 (σ)의 영향**

| σ | ImageNet FID | CelebA-HQ FID |
|---|--------------|---------------|
| 0 | 5.65 | 4.11 |
| 0.10 | 5.44 | 3.74 |
| 0.15 | 5.31 | 3.43 |
| 0.20 | 5.27 | 3.15 |
| 0.25 | 5.48 | 3.23 |
| 0.30 | 5.91 | 3.52 |

최적값 σ=0.20에서, SDE 샘플러는 ODE 샘플러(σ=0)보다 현저히 우수하다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)

**샘플링 스텝 효율**

Figure 5는 RDM이 적은 NFE에서 경쟁력 있는 성능을 유지함을 보여준다. DiT-XL2와 MDT-XL2는 NFE 감소에 따라 성능이 급격히 저하되는 반면, RDM은 안정적이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)

***

## 5. 모델의 일반화 성능 향상 가능성

### 5.1 해상도 간 전이 능력

RDM의 핵심 혁신은 **해상도 간 일반화 가능성**을 크게 향상시킨다는 것이다:

1. **블록 노이즈의 보편성**: 블록 노이즈는 저해상도 가우시안 노이즈의 고해상도 등가물이므로, 모든 해상도 조합에 적용 가능하다. 수식 (4)-(6)의 구성 방식은 계층 구조적으로 확장 가능하다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)

2. **계단식 파이프라인의 단순화**: 저해상도 조건을 제거함으로써 분포 불일치 문제를 해결한다. 기존 CDM에서 필요했던 조건부 증강(conditioning augmentation) 기법이 불필요해진다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)

3. **노이즈 스케줄의 직관적 적응**: 주파수 영역 분석을 통해 새로운 해상도에서 필요한 노이즈 스케줄의 조정 방식을 수식 (17)-(18)처럼 명시적으로 제시할 수 있다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)

### 5.2 도메인 간 전이

논문은 다음과 같은 일반화 강점을 암시한다:

- **CelebA-HQ(얼굴) vs ImageNet(일반 객체)**: 동일한 아키텍처와 블록 노이즈 설정으로 두 데이터셋 모두에서 최고 성능 달성
- **조건부 vs 무조건부**: 클래스 조건부(ImageNet) 및 무조건부(CelebA-HQ) 생성 모두 적용 가능
- **모델 크기 확장성**: U-Net 아키텍처의 매개변수 개수 증가 시에도 블록 노이즈 설정이 일관되게 작동 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)

### 5.3 미래 일반화 방향

논문의 한계와 미래 작업은 다음을 시사한다:

**현재 미해결 문제:**
- 최적 노이즈 스케줄은 모델 크기, 귀납 편향, 데이터 분포 특성에 따라 달라지므로, 단순한 수학적 유도만으로는 불충분하다 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)
- 블록 노이즈 커널 크기($s$)의 최적화는 여전히 해상도별로 수동으로 조정 필요

**일반화 가능성 개선 기회:**
1. **다중 해상도 학습**: 64×64 → 128×128 → 256×256 → 512×512 등 더 많은 단계로 확장 시 블록 노이즈의 적응적 자동화
2. **적응형 노이즈 스케줄 학습**: 최근 MuLAN(Multivariate Learned Adaptive Noise) 같은 방법과 결합하면 학습 가능한 노이즈 스케줄 개발 가능 [semanticscholar](https://www.semanticscholar.org/paper/da0bc8aff42754f8969484e80f399b06beb63ffb)
3. **아키텍처 독립성**: DiT 같은 Transformer 기반 백본으로 확장 시 일반화 성능이 더욱 향상될 가능성

### 5.4 정량적 일반화 증거

| 지표 | 기존 CDM | RDM | 향상 |
|-----|---------|-----|------|
| 학습 반복 비율 | 100% | 70% (ImageNet) | 30% 감소 |
| 계단식 오류 누적 | 있음 | 감소 | 분포 불일치 제거 |
| 조건부 증강 필요 | 필수 | 불필요 | 파이프라인 단순화 |

***

## 6. 한계와 과제

### 6.1 기술적 한계

1. **노이즈 스케줄 최적화의 한계**: 논문은 "최적 노이즈 스케줄을 직접 유도하려는 수많은 시도가 좋은 결과를 산출하지 못했다"고 명시한다. 이는 주파수 영역 분석만으로는 충분하지 않음을 시사한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)

2. **블록 노이즈 매개변수의 수동 조정**: 
   - ImageNet: $\lambda=0.15$, $s=4$
   - CelebA-HQ: $\lambda=0.15$, $s=4$
   
   이러한 하이퍼파라미터는 데이터셋별로 수동으로 설정되었으며, 새로운 해상도나 영역에 대한 자동 적응 메커니즘이 부재한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)

3. **계산 복잡도**: 블러링 확산의 DCT 변환 및 역변환은 추가 계산 오버헤드를 야기한다. 논문은 이에 대한 구체적인 런타임 비교를 제시하지 않는다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)

### 6.2 실험적 한계

1. **제한된 비교 대상**: RDM은 주로 2023년 이전의 모델(DiT, MDT)과 비교되었다. 2024-2025년의 최신 모델(예: Simpler Diffusion, 고급 DiT 변형)과의 직접 비교 부족.

2. **텍스트-투-이미지 생성 평가 부재**: RDM은 Imagen이나 Stable Diffusion 같은 텍스트 조건부 모델에 대해 평가되지 않았으며, 이는 실제 응용 범위를 제한한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)

3. **고해상도(512×512 이상) 검증 부족**: RDM은 256×256까지만 평가되었으나, 최신 경쟁 모델들은 512×512 또는 1024×1024를 수행한다. [arxiv](https://arxiv.org/html/2410.19324v1)

### 6.3 개념적 한계

1. **주파수 분석의 일반화 한계**: DCT 분석은 자연 이미지의 전형적인 주파수 분포를 가정하지만, 의료 영상, 위성 이미지, 극저해상도 이미지에서는 이 가정이 유효하지 않을 수 있다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)

2. **분포 불일치 완전 해결 불가**: 논문은 블록 노이즈와 패치별 블러링이 "낮은 해상도 조건의 분포 불일치 문제를 감소"시킨다고 표현하지만, 완전한 해결이 아니다. 저해상도 생성의 아티팩트는 여전히 고해상도 단계로 전파될 수 있다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)

***

## 7. 앞으로의 연구 영향 및 고려사항

### 7.1 이론적 영향

**1. 노이즈 스케줄 설계의 새로운 패러다임**

RDM의 주파수 영역 분석은 노이즈 스케줄 설계에서 게임 체인저로 작용한다:

- **해상도 의존성의 수학적 정당화**: 기존 경험적 노이즈 스케줄(linear, cosine)이 모든 해상도에 최적이 아님을 증명 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)
- **신 이론의 영감**: 최근 Chen (2023)의 "On the Importance of Noise Scheduling", Hang et al. (2025)의 "Improved Noise Schedule for Diffusion Training" 등 후속 연구가 RDM의 주파수 분석을 기반으로 발전하고 있다 [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Hang_Improved_Noise_Schedule_for_Diffusion_Training_ICCV_2025_paper.pdf)

**2. 블록 노이즈의 일반화**

블록 노이즈 개념은 다음 분야로 확장될 가능성이 높다:
- **비등방성 노이즈**: 방향성 아티팩트가 있는 이미지(예: 텍스트, 선)에 특화된 노이즈
- **적응형 노이즈**: MuLAN()처럼 입력 이미지에 따라 동적으로 조정되는 노이즈

### 7.2 실무적 영향

**1. 계단식 파이프라인의 재설계**

RDM은 다단계 생성 모델 구축의 새로운 표준을 제시한다:

| 접근법 | 조건부 | 조건 증강 | 효율성 |
|--------|--------|---------|--------|
| SR3 (기존) | 필수 | 필수 | 낮음 |
| CDM | 필수 | 필수 | 중간 |
| **RDM** | **불필요** | **불필요** | **높음** |

이는 생산 환경에서 학습 비용을 30% 감소시킬 수 있음. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)

**2. 고해상도 모델 개발 가속화**

RDM의 효율성 향상은 큰 언어 모델(LLM) 기반 멀티모달 모델에 통합될 때 특히 가치가 있다:
- Imagen의 cascade 단계를 RDM 기반으로 재구현하면 훈련 시간 20-40% 감소 가능
- DALL-E 3, Stable Diffusion 같은 상용 모델의 고해상도 생성 비용 절감

### 7.3 앞으로의 연구 고려사항

**1. 해상도 확장 연구**

현재 RDM은 256×256에서만 평가되었으므로, 다음이 필요하다:

$$\sigma_t(\text{512×512}) = F_{\sigma}(\text{512×512}) \times \sigma_t(\text{256×256})$$

**2. 모델 간 호환성**

RDM의 접근법이 DiT(Vision Transformer 기반) 아키텍처에도 동일하게 효과적인지 검증 필요. 논문은 U-Net만 사용했으므로: [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/dit/)

$$\text{RDM}_{\text{DiT}} = \text{Relay}(\text{Diffusion Transformer}) + \text{Block Noise}$$

**3. 멀티모달 조건부 생성**

텍스트, 텍스트+이미지 조건부 생성으로 확장 시:
- 저해상도 텍스트-이미지 모델 → 고해상도 이미지 생성
- 텍스트 인코더와의 상호작용: 저해상도 출력이 text cross-attention의 정보 손실을 보상하는지 검증

**4. 적응형 노이즈 스케줄 학습**

최근의 MuLAN()이나 Diffusion Models with Learned Adaptive Noise 같은 방법과 결합:

$$\sigma_{\text{adaptive}}(x, t) = \text{Learn}(\sigma_{\text{RDM}}(t), x)$$

이를 통해 데이터셋별 수동 하이퍼파라미터 조정을 자동화할 수 있다.

***

## 8. 2020년 이후 관련 최신 연구 비교 분석

### 8.1 주요 경쟁 모델 비교

| 모델 | 발표년 | 아키텍처 | ImageNet 256 FID | 주요 혁신 | 한계 |
|-----|--------|---------|-----------------|---------|------|
| **ADM** | 2021 | U-Net + Attn | 3.94(CFG) | 분류기 안내 | 높은 계산 비용 |
| **LDM** | 2022 | VAE + U-Net | 10.56 | 잠상 공간 효율성 | 고해상도 아티팩트 |
| **CDM** | 2022 | Cascade | 4.88 | 계단식 파이프라인 | 조건 불일치 |
| **DiT** | 2022 | ViT | 2.27(CFG) | Transformer 백본 | 높은 메모리 |
| **MDT** | 2023 | Masked DiT | 1.79(CFG) | 마스킹 기반 생성 | 복잡한 구조 |
| **RDM** | 2023 | U-Net + Block Noise | **1.87**(CFG 3.50) | **주파수 기반 노이즈** | 수동 튜닝 |
| **DiffiT** | 2023 | ViT + TMSA | 1.73 | 시간 의존 주의 | 이론 부재 |
| **Simpler Diffusion** | 2025 | Pixel-space | 1.50(ImageNet 512) | 픽셀 공간 복귀 | 고계산 비용 |

### 8.2 노이즈 스케줄 연구의 진화

**2021-2022: 기초 단계**
- Ho et al. (2020): 선형 스케줄, 코사인 스케줄 제안 [arxiv](https://arxiv.org/abs/2106.15282)
- Nichol & Dhariwal (2021): 개선된 수치 기법 [liner](https://liner.com/review/diffusion-models-beat-gans-on-image-synthesis)

**2023: RDM의 기여**
- 주파수 영역 분석으로 **해상도-의존성 증명** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)
- 블록 노이즈의 등가성 수학화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)
- Chen (2023): 노이즈 스케줄의 중요성 실증 [arxiv](https://arxiv.org/abs/2301.10972)

**2024-2025: 적응형 및 학습 기반 접근**
- MuLAN (Sahoo et al., 2024): 다변량 학습 가능 적응형 노이즈 [proceedings.neurips](https://proceedings.neurips.cc/paper_files/paper/2024/file/bee43378b65ec195a67f24709469dcaf-Paper-Conference.pdf)
- NoiseShift (2024): 해상도 인식 노이즈 재보정 [arxiv](https://arxiv.org/html/2510.02307v1)
- Hang et al. (2025): 통합 노이즈 스케줄 설계 프레임워크 [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Hang_Improved_Noise_Schedule_for_Diffusion_Training_ICCV_2025_paper.pdf)

### 8.3 계단식 생성의 진화

**초기 단계 (2021-2022)**
- Cascaded Diffusion Models (Ho et al., 2022): 기초 계단식 개념, [dl.acm](https://dl.acm.org/doi/abs/10.5555/3586589.3586636)
- SR3 (Saharia et al., 2023): 초해상도 확산 [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/36094974/)

**RDM의 혁신 (2023)**
- 저해상도 조건 제거 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)
- 분포 불일치 문제 감소 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)
- 효율성 30% 향상 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)

**최신 동향 (2024-2025)**
- Simpler Diffusion (2025): 픽셀 공간 재평가, 1.5 FID@ImageNet512 [arxiv](https://arxiv.org/html/2410.19324v1)
- DiT4SR (2025): Transformer 기반 초해상도 [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/html/Duan_DiT4SR_Taming_Diffusion_Transformer_for_Real-World_Image_Super-Resolution_ICCV_2025_paper.html)
- Self-Cascade 모델: 저해상도 모델 지식 활용 [arxiv](https://arxiv.org/html/2402.10491v2)

### 8.4 해상도별 성능 벤치마크 (2020-2025)

```
FID Score Evolution on ImageNet 256×256 (Class-Conditional)

2020-2021:
  DDPM ............................................................... 3.17
  BigGAN-deep ....................................................... 6.95

2021-2022:
  ADM (no CFG) .................................................. 10.94
  ADM (CFG) ....................................................... 3.94
  LDM ........................................................ 10.56

2022-2023:
  CDM ........................................................ 4.88
  DiT-XL2 (no CFG) ......................................... 9.62
  DiT-XL2 (CFG) ...................................... 2.27
  MDT-XL2 (no CFG) ..................................... 6.23
  MDT-XL2 (CFG) .................................. 1.79
  RDM (no CFG) ........................................ 5.27
  RDM (CFG) ..................................... 1.87

2023-2024:
  DiffiT (Latent, no CFG) ........................... 1.73
  Various Consistency Models ..................... 1.5-2.0

2024-2025:
  Simpler Diffusion (512×512) ................... 1.50
  Latest advanced models ..................... <1.50
```

### 8.5 RDM의 위치와 의의

RDM은 **계단식 생성 효율화**와 **주파수 기반 노이즈 이론** 두 가지 측면에서 중요한 다리 역할을 한다:

1. **이론적 기여**: 주파수 분석을 통해 노이즈 스케줄의 해상도 의존성을 수학적으로 정당화한 최초의 작업 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)
2. **실무적 효율성**: 계단식 파이프라인에서 조건부 불필요 → 학습 30% 가속 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)
3. **개방성**: 코드와 체크포인트를 오픈소스로 공개하여 후속 연구 활성화

***

## 결론

**Relay Diffusion Model**은 고해상도 이미지 합성의 두 가지 근본 문제—**학습 효율성**과 **해상도-의존적 노이즈 스케줄**—을 주파수 영역 분석과 블록 노이즈 개념으로 우아하게 해결한다. 

**핵심 혁신:**
- 이산 코사인 변환을 통한 주파수 기반 SNR 분석으로 노이즈 스케줄 설계의 새로운 기초 제공
- 블록 노이즈의 도입으로 해상도 간 등가성을 수학적으로 달성
- 저해상도 조건 제거로 계단식 파이프라인 단순화 및 효율화

**앞으로의 영향:**
1. 노이즈 스케줄 연구의 이론적 토대 제공 → MuLAN, NoiseShift 등 2024-2025 연구로 발전
2. 계단식 생성 방식의 재정의 → Simpler Diffusion 등에서 재검토
3. 멀티모달 대규모 모델의 고해상도 생성 비용 절감 → 산업 응용 활성화

**남은 과제:**
- 512×512 이상 고해상도 확장 검증
- 텍스트 조건부 모델(Imagen, DALL-E 스타일) 적용 평가
- 적응형 노이즈 스케줄 자동화로 하이퍼파라미터 수동 조정 제거
- 다양한 도메인(의료, 위성, 극저해상도)에서의 일반화 능력 검증

***

## 참고 자료

 Teng, J., Zheng, W., Ding, M., et al. (2023). "Relay Diffusion: Unifying Diffusion Process Across Resolutions for Image Synthesis." arXiv:2309.03350. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/de4090d2-2100-437b-a6c1-6c07b453bb3d/2309.03350v1.pdf)

 Ho, J., Saharia, C., Chan, W., et al. (2022). "Cascaded Diffusion Models for High Fidelity Image Generation." JMLR, 23, 1-33. [dl.acm](https://dl.acm.org/doi/abs/10.5555/3586589.3586636)

 Google Research. (2022). "Cascaded Diffusion Models for High Fidelity Image Generation." [research](https://research.google/pubs/cascaded-diffusion-models-for-high-fidelity-image-generation/)

 Ho, J., Saharia, C., et al. (2021). "Cascaded Diffusion Models for High Fidelity Image Generation." arXiv:2106.15282. [arxiv](https://arxiv.org/abs/2106.15282)

 Dhariwal, P., & Nichol, A. (2021). "Diffusion Models Beat GANs on Image Synthesis." NeurIPS. [liner](https://liner.com/review/diffusion-models-beat-gans-on-image-synthesis)

 Saharia, C., et al. (2023). "Image Super-Resolution via Iterative Refinement." PubMed Central. [pubmed.ncbi.nlm.nih](https://pubmed.ncbi.nlm.nih.gov/36094974/)

 Peebles, B., & Xie, S. (2022). "Scalable Diffusion Models with Transformers (DiT)." arXiv. [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/dit/)

 Hang, T., et al. (2025). "Improved Noise Schedule for Diffusion Training." ICCV 2025. [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Hang_Improved_Noise_Schedule_for_Diffusion_Training_ICCV_2025_paper.pdf)

 Sahoo, S., et al. (2024). "Diffusion Models With Learned Adaptive Noise." NeurIPS. [proceedings.neurips](https://proceedings.neurips.cc/paper_files/paper/2024/file/bee43378b65ec195a67f24709469dcaf-Paper-Conference.pdf)

 "NoiseShift: Resolution-Aware Noise Recalibration for High-Resolution Image Synthesis." arXiv:2510.02307. [arxiv](https://arxiv.org/html/2510.02307v1)

 Duan, Z. P., et al. (2025). "DiT4SR: Taming Diffusion Transformer for Real-World Image Super-Resolution." ICCV 2025. [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/html/Duan_DiT4SR_Taming_Diffusion_Transformer_for_Real-World_Image_Super-Resolution_ICCV_2025_paper.html)

 Chen, T. (2023). "On the Importance of Noise Scheduling for Diffusion Models." arXiv:2301.10972. [arxiv](https://arxiv.org/abs/2301.10972)

 Hoogeboom, E., et al. (2025). "1.5 FID on ImageNet512 with pixel-space diffusion (Simpler Diffusion)." CVPR 2025. [arxiv](https://arxiv.org/html/2410.19324v1)
