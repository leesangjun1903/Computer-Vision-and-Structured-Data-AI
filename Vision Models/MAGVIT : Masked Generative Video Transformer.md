
# MAGVIT: Masked Generative Video Transformer

## 1. 핵심 주장 및 주요 기여

MAGVIT는 비디오 생성 분야에 네 가지 획기적인 기여를 제시한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2820b430-a45b-4daf-b2d2-4795997b5783/2212.05199v2.pdf)

**첫째, 다중작업 효율성의 새로운 패러다임**: MAGVIT는 단일 학습된 모델로 10가지 서로 다른 비디오 생성 작업(프레임 예측, 프레임 보간, 인페인팅, 아웃페인팅, 클래스 조건부 생성)을 동시에 수행할 수 있는 최초의 마스크 기반 멀티태스크 트랜스포머이다. 이는 기존 단일작업 모델의 한계를 극복한다.

**둘째, 공간-시간 토큰화의 우수성**: 3D-VQ(Vector-Quantized) 오토인코더를 통해 비디오를 고충실도의 공간-시간 시각 토큰으로 양자화한다. 이는 2D-VQ의 시간적 일관성 부족과 기존 3D-VQ의 세부 손실 문제를 모두 해결한다.

**셋째, COMMIT(Conditional Masked Modeling by Interior Tokens)의 혁신**: 내부 조건을 손상된 시각 토큰에 직접 임베딩하는 멀티변수 마스크 방식을 제안했다. 이는 기존의 직관적이지 않은 마스크 방식을 개선하여 비인과적 마스킹 문제를 해결하고 일반화 성능을 향상시킨다.

**넷째, 극단적 효율성**: 인퍼런스 속도에서 확산 모델보다 100배, 자동회귀 모델보다 60배 빠르면서도 최고 수준의 품질을 유지한다.

***

## 2. 해결하고자 하는 문제

### 2.1 기존 방법의 한계

비디오 생성 분야는 세 가지 근본적인 문제를 안고 있었다:

**효율성 문제**: 확산 모델은 256-1,024 스텝이 필요하고, 자동회귀 모델은 토큰 시퀀스 길이만큼의 스텝이 필요하다. 이는 실제 응용에 부적합한 수준이다.

**일반화 부재**: 프레임 예측, 인페인팅, 아웃페인팅 등 각 작업마다 별도의 모델을 학습해야 했다. 다양한 도메인(로봇, 자율주행, 사람-물체 상호작용)에 대한 통일된 접근이 불가능했다.

**토큰화 품질**: 2D-VQ는 시간적 일관성이 떨어지고(프레임 깜빡임), 기존 3D-VQ는 움직이는 물체의 세부를 손실한다. 토큰화 자체가 생성 품질의 상한선을 결정한다.

### 2.2 MAGVIT의 해결 전략

비마스크 토큰 예측 패러다임 도입: 기존의 생성 중심 접근과 달리, MAGVIT는 모든 토큰 예측을 병렬로 처리한다. 이를 통해 12 스텝의 비자동회귀 디코딩으로 완전한 비디오를 생성할 수 있다.

조건부 마스킹의 수학적 정교화: 마스크 함수를 다음과 같이 정의하여 조건 정보를 손실 없이 임베딩한다:

$$m(z_i | \tilde{z}_i) = \begin{cases} \tilde{z}_i & \text{if } s_i \le s^* \land \text{ispad}(\tilde{z}_i) \\ \text{[MASK]} & \text{if } s_i \le s^* \land \neg\text{ispad}(\tilde{z}_i) \\ z_i & \text{if } s_i > s^* \end{cases}$$

여기서 $\tilde{z}_i$는 조건 토큰, $s_i$는 마스크 스코어, $s^*$는 임계값이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2820b430-a45b-4daf-b2d2-4795997b5783/2212.05199v2.pdf)

***

## 3. 제안 방법과 수식

### 3.1 3D-VQ 토큰화

**인코더-디코더 구조**:
비디오 $V \in \mathbb{R}^{T \times H \times W \times 3}$에 대해 다음과 같이 정의된다:

$$f_T : V \to z \in \mathbb{Z}^N$$

인코더는 다음 구조를 가진다:
- 입력: 3×3×3 컨볼루션, 32 채널
- 잔차 블록으로 인코더 구성
- 공간적으로 2×2×2 평균 풀링, 시간적으로 2×2 풀링 교대로 적용
- 최종 공간-시간 압축: 4×8×8
- VQ 코드북: 1,024 크기, 256차원 임베딩

**3D 인플레이션(Inflation)**:
2D-VQ에서 초기화하여 시간축 학습을 안정화:

$$K_{3D,\text{center}}[t,i,j] = K_{2D}[i,j] \text{ for } t = \frac{T-1}{2}$$
$$K_{3D,\text{center}}[t,i,j] = 0 \text{ otherwise}$$

이 중심 인플레이션은 평균 인플레이션보다 시간 일관성이 우수하다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2820b430-a45b-4daf-b2d2-4795997b5783/2212.05199v2.pdf)

**GAN 손실 정규화**:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda_1 \mathcal{L}_{\text{GAN}} + \lambda_2 \mathcal{L}_{\text{LeCam}} + \lambda_3 \mathcal{L}_{\text{perceptual}}$$

### 3.2 COMMIT 멀티태스크 마스킹

**손실 함수 분해**:
전체 손실은 세 개의 목적으로 분해된다:

$$\mathcal{L}(V; \theta) = \mathbb{E}_{p, \tilde{z}, \bar{z}} \mathbb{E}_{p_m} \left[ \sum_i -\log p_\theta(z_i | [p, c, \bar{z}]) \right]$$

여기서 $p$는 작업 프롬프트, $c$는 클래스 토큰이다.

**손실 분해**:

$$\sum_i -\log p_\theta(z_i | [p, c, \bar{z}]) = \underbrace{\sum_{z_i = \tilde{z}_i} -\log p_\theta(z_i | \bar{c})}_{\mathcal{L}_{\text{refine}}} + \underbrace{\sum_{\bar{z}_i = [\text{MASK}]} -\log p_\theta(z_i | \bar{c})}_{\mathcal{L}_{\text{mask}}} + \underbrace{\sum_{\bar{z}_i = z_i} -\log p_\theta(z_i | \bar{c})}_{\mathcal{L}_{\text{recons}}}$$

$\mathcal{L}_{\text{refine}}$은 COMMIT의 핵심 혁신으로, 조건 토큰을 정제하면서 생성 품질을 향상시킨다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2820b430-a45b-4daf-b2d2-4795997b5783/2212.05199v2.pdf)

### 3.3 비자동회귀 디코딩 알고리즘

**Algorithm 1**: COMMIT 디코딩 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2820b430-a45b-4daf-b2d2-4795997b5783/2212.05199v2.pdf)

```
입력: 프리픽스 p, c, 조건 ẑ, 스텝 K, 온도 T
출력: 예측된 시각 토큰 z

s ← 0, s* ← 1, ẑ ← 0^N
for t = 0, 1, ..., K-1 do
    z̄ ← m(ẑ | ẑ; s, s*)
    ẑ_i ~ p_θ(z_i | [p, c, z̄]), ∀i where s_i ≤ s*
    s_i ← p_θ(ẑ_i | [p, c, z̄]), ∀i where s_i ≤ s*
    s_i ← s_i + T(1 - t/K) Gumbel(0,1), ∀i where s_i < 1
    s* ← [⌈γ(t+1/K)N⌉]-th smallest value of s
    s_i ← 1, ∀i where s_i > s*
end for
return z = [ẑ_1, ẑ_2, ..., ẑ_N]
```

각 스텝에서 코사인 스케줄 $\gamma(t/K)$에 따라 예측 확률이 높은 토큰부터 고정되고, 나머지는 다음 반복에서 재예측된다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2820b430-a45b-4daf-b2d2-4795997b5783/2212.05199v2.pdf)

***

## 4. 모델 구조

### 4.1 전체 아키텍처

MAGVIT는 두 단계 구조를 가진다:

**Stage 1: 3D-VQ 토큰화**
- 인코더: Conv 3×3×3 → 5개 ResBlock 스택 + 다운샘플링
- VQ 병목: 코드북 크기 1,024, 임베딩 차원 256
- 디코더: ResBlock + 업샘플링 → Conv 3×3×3 (RGB 복원)
- 판별자: StyleGAN 기반 3D 판별자

압축 비율: $T \times H \times W \times 3 \to \frac{T}{4} \times \frac{H}{8} \times \frac{W}{8} \times 256$

**Stage 2: 멀티태스크 트랜스포머**
- BERT 아키텍처 기반
- Base 모델: 128M 파라미터, 12 헤더, 12 레이어
- Large 모델: 464M 파라미터, 16 헤더, 24 레이어
- 입력 시퀀스: 1 태스크 프롬프트 + 1 클래스 토큰 + 1,024 시각 토큰 = 1,026 토큰

### 4.2 작업별 마스킹 설정

| 작업 | 내부 조건 | 마스킹 유형 | 패딩 방식 |
|------|---------|-----------|----------|
| Frame Prediction (FP) | 처음 1프레임 | 시간 전반부 | 복제 |
| Frame Interpolation (FI) | 첫+마지막 프레임 | 양쪽 끝 | 선형 보간 |
| Central Outpainting (OPC) | 중앙 0.5H×0.5W | 중앙 외부 | 엣지 |
| Dynamic Outpainting (OPD) | 이동하는 수직 스트립 | 동적 영역 | 영 패딩 |
| Class-conditional (CG) | 클래스 레이블만 | 전체 | N/A |

***

## 5. 성능 향상 분석

### 5.1 생성 품질 비교

| 벤치마크 | 메트릭 | 이전 SOTA | MAGVIT | 개선율 |
|----------|-------|---------|--------|------|
| UCF-101 | FVD | 332 (TATS) | 76 | ↓77% |
| UCF-101 | IS | 79.28 | 89.27 | ↑13% |
| BAIR | FVD | 84 (RaMViD) | 62 | ↓26% |
| Kinetics-600 | FVD | 16.2 (Video Diffusion) | 9.9 | ↓39% |

**특징**: 큰 규모 벤치마크(Kinetics-600)에서 더 큰 성능 개선을 달성하여 확장성을 입증한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2820b430-a45b-4daf-b2d2-4795997b5783/2212.05199v2.pdf)

### 5.2 추론 효율성

| 방법 | 모델/VQ | 시퀀스 길이 | 스텝 수 | FVD |
|------|--------|-----------|--------|-----|
| Video Diffusion | 3D U-Net | - | 256-1,024 | 16.2 |
| TATS (AR) | 트랜스포머 | 1,024 | 1,024 | 84 |
| MaskViT | 2D-VQ | 4,096 | 16-64 | 94 |
| MAGVIT | 3D-VQ | 1,024 | 12 | 62 |

**추론 시간**: V100 GPU에서 128×128 해상도, 16프레임 생성 시:
- MAGVIT: 0.027초 (37 fps)
- Video Diffusion: ~2.7초 (2.7 배 느림, 실제로는 100배 차이 존재)
- TATS: ~2초 (60배 느림)

TPUv4i에서 MAGVIT-B는 190 fps, MAGVIT-L은 65 fps 달성. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2820b430-a45b-4daf-b2d2-4795997b5783/2212.05199v2.pdf)

### 5.3 멀티태스크 성능

BAIR에서 8가지 작업 평가 (debiased FVD):

| 작업 | Single-UNC | Single-FP | Multi-MT |
|------|-----------|----------|---------|
| FP | 150.6 | 201.1 | **31.4** |
| FI | 74.0 | 47.7 | **26.4** |
| OPC | 71.4 | 56.2 | **21.3** |
| OPV | 119.0 | 247.1 | **21.2** |
| OPH | 46.7 | 118.5 | **19.5** |
| OPD | 55.9 | 142.7 | **20.9** |
| IPC | 389.3 | 366.3 | **21.3** |
| IPD | 145.0 | 357.3 | **20.3** |
| **평균** | **121.1** | **172.1** | **22.8** |

멀티태스크 학습이 모든 작업에서 단일 작업 모델을 능가하며, 특히 학습되지 않은 작업(회색)에서 큰 개선을 보인다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2820b430-a45b-4daf-b2d2-4795997b5783/2212.05199v2.pdf)

### 5.4 일반화 성능

**다양한 도메인 테스트**:

| 데이터셋 | 특성 | 작업 | MAGVIT-B FVD | MAGVIT-L FVD |
|----------|------|------|-------------|-------------|
| nuScenes | 자율주행 | FP | 29.3 | 20.6 |
| Objectron | 물체-중심 | FI | - | 26.7 |
| Web Videos | 웹 규모 | MT8 | 33.0 | 21.6 |

단일 모델로 완전히 다른 시각적 도메인에 적응하여 높은 일반화 능력을 보여준다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2820b430-a45b-4daf-b2d2-4795997b5783/2212.05199v2.pdf)

***

## 6. COMMIT의 혁신성

### 6.1 문제: 비인과적 마스킹

VQ-VAE의 비국소 수용장(non-local receptive field)으로 인해, 조건 영역의 토큰을 단순히 언마스크하면 생성 토큰이 조건 정보를 직접 참조할 수 있다. 이는 테스트 시 다른 조건에서 일반화되지 않는 문제를 야기한다.

| 방법 | 시퀀스 길이 | FP FVD | MT8 FVD |
|------|-----------|--------|---------|
| MaskGIT (직접 언마스크) | 1,024 | 74 | 151 |
| Prefix 조건 | 1,024-1,792 | 55 | - |
| COMMIT ( $\mathcal{L}\_{\text{mask}} + \mathcal{L}_{\text{recons}}$ ) | 1,024 | 51 | 53 |
| COMMIT (전체: + $\mathcal{L}_{\text{refine}}$) | 1,024 | **48** | **33** |

$\mathcal{L}_{\text{refine}}$은 조건 토큰의 품질을 명시적으로 개선하여 생성 성능을 크게 향상시킨다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2820b430-a45b-4daf-b2d2-4795997b5783/2212.05199v2.pdf)

### 6.2 고정 길이 시퀀스의 이점

일반적인 프리픽스 조건 방식은 조건의 크기에 따라 시퀀스 길이가 변한다:
- 1프레임 조건: 토큰 변동 가능
- 전체 프레임 조건: 더 긴 시퀀스

COMMIT은 **항상 1,026 토큰**의 고정 길이를 유지하여:
- 배치 처리 효율성 증대
- 메모리 사용량 예측 가능
- 다양한 작업의 공동 학습 가능

***

## 7. 한계 및 개선 방향

### 7.1 현존하는 한계

**텍스트-비디오 생성 미지원**: MAGVIT는 모두 프레임 조건, 클래스 레이블, 또는 공간-시간 마스크에 의존한다. 자연어 프롬프트 조건은 구현되지 않았다. 이는 Make-A-Video 같은 동시대 방법과의 주요 차이점이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2820b430-a45b-4daf-b2d2-4795997b5783/2212.05199v2.pdf)

**해상도 제약**: 실험은 128×128(BAIR는 64×64)에서만 수행되었다. 고해상도(512×512 이상) 생성 능력은 입증되지 않았다.

**긴 비디오 생성**: 16프레임 클립 생성에 최적화되었으며, 더 긴 비디오(예: TATS의 수천 프레임)로 확장 성능은 미검증이다.

**조건 품질 의존성**: 프레임 예측에서 초기 프레임 품질이 생성 결과에 큰 영향을 미친다. 노이지한 초기 조건에 대한 강건성 분석 부재.

### 7.2 추가 분석 필요 영역

- **일반화 한계**: 학습되지 않은 시각적 도메인(의료 영상, 극단적 기후 환경)에 대한 성능
- **기하학적 일관성**: 카메라 움직임이 많은 장면에서의 공간적 안정성
- **시간적 일관성**: 매우 긴 비디오에서 누적 오류

***

## 8. 최신 연구 비교분석 (2020년 이후)

### 8.1 주요 경쟁 방법론

#### **Diffusion 기반 모델**

| 모델 | 출판 | 주요 특성 | FVD (K600) | 장점 | 단점 |
|------|------|---------|-----------|------|------|
| Video Diffusion  | ICLR 2022 Workshop | 3D U-Net + 확산 | 16.2 | 높은 품질 | 256-1024 스텝 |
| RaMViD  | CVPR 2022 | 진폭 변조 확산 | 16.5 | 개선된 샘플링 | 느린 속도 |
| VDT  | CVPR 2023 | Diffusion Transformer | - | 유연한 마스킹 | MAGVIT보다 느림 |

#### **Autoregressive 모델**

| 모델 | 출판 | 주요 특성 | 추론 스텝 | 장점 | 단점 |
|------|------|---------|---------|------|------|
| TATS  | ECCV 2022 | 시간-공간 분리 | 1,024 | 안정적 학습 | 극도로 느린 추론 |
| VideoGPT  | ICML 2021 | 자동회귀 VQ-VAE | 1,024 | 초기 시도 | 낮은 품질 |
| Make-A-Video  | ICLR 2023 | 텍스트-비디오 | - | T2V 지원 | 10M 추가 데이터 |

#### **비자동회귀 Transformer (MAGVIT와 동시대)**

| 모델 | 출판 | 주요 특성 | 시퀀스 길이 | 추론 스텝 |
|------|------|---------|-----------|---------|
| MaskViT  | ECCV 2022 | 2D-VQ + 마스킹 | 4,096 | 16-64 |
| MAGVIT  | CVPR 2023 | 3D-VQ + COMMIT | 1,024 | 12 |

### 8.2 MAGVIT-v2의 의미 (2024)

MAGVIT-v2는 2024년 ICLR에서 발표되었으며, 원래 MAGVIT의 한계를 세 가지 측면에서 극복했다: [semanticscholar](https://www.semanticscholar.org/paper/985f0c89c5a607742ec43c1fdc2cbfe54541cbad)

**1. 공동 이미지-비디오 토큰화**
- MAGVIT는 3D CNN 때문에 이미지 토큰화에 어려움
- MAGVIT-v2는 카우살 ViT와 3D CNN 혼합
- 공유 코드북으로 이미지/비디오 동시 처리 가능

**2. 향상된 양자화 (LFQ)**
- 기존 VQ-VAE 대비 큰 코드북 (2^18)
- 조회 없는 양자화로 더 많은 토큰 활용
- 생성 품질 대폭 향상

**3. 성능 수치**
```
MAGVIT-v2 성과:
- ImageNet FID: 1.2 (SOTA, 확산 모델 능가)
- Kinetics-600: 향상된 FVD
- 비디오 압축: VVC (차세대 표준) 수준
```

### 8.3 2024-2026 최신 추세

#### **Tokenizer 개선**

| 연구 | 특징 | 혁신 |
|------|------|------|
| OmniTokenizer  | 공동 이미지-비디오 | 1.11 FID (ImageNet) |
| ProMAG  | 시간 압축 개선 | 4×→16× 압축 유지 |
| Gaussian Video Transformer | 적응형 토큰화 | 정보 기반 토큰 배분 |

#### **생성 모델 진화**

| 영역 | 발전 |
|------|------|
| 텍스트-비디오 | Sora 2, Veo 3.1 등 proprietary 모델 대두 |
| 오픈소스 | Wan2.2, HunyuanVideo, LTX-Video 등장 |
| 평가 지표 | VBench (16차원), FVMD (모션 중심) 도입 |

### 8.4 MAGVIT의 지속적 영향

**학술적 임팩트**:
- 2023년 발표 이후 400+ 인용 [arxiv](https://arxiv.org/abs/2212.05199)
- 마스크 기반 생성의 비자동회귀 패러다임 확립
- 멀티태스크 학습의 효율성 입증

**산업적 활용**:
- Open-MAGVIT2: 오픈소스 복제본 (2024) [arxiv](https://arxiv.org/html/2409.04410v3)
- 여러 최신 모델의 기반 아키텍처 역할
- 에지 디바이스 배포를 위한 참조 구현

***

## 9. 모델의 일반화 성능 향상 가능성

### 9.1 현재 일반화 능력

**검증된 일반화**:

1. **시각적 도메인 확장** (표 8에서 확인)
   - 스포츠/행동 인식: UCF-101
   - 로봇 조작: BAIR
   - 사람-물체 상호작용: Something-Something-v2
   - 자율주행: nuScenes
   - 물체-중심 비디오: Objectron
   - 웹 규모 데이터: 12M YouTube 비디오

2. **작업 일반화** (표 4에서 검증)
   - 학습 데이터: 8-10개 작업
   - 테스트: 동일 데이터셋의 모든 작업
   - 미학습 작업도 우수한 성능 유지

### 9.2 향상 가능성 분석

#### **A. 데이터 확장의 영향**

현재 MAGVIT는 상대적으로 작은 데이터셋(UCF-101: 9.5K)에서 학습됨. 더 큰 데이터 활용 시:

$$\text{성능} \approx a \log(N_{\text{data}}) + b$$

**예상 개선**:
- 100배 데이터 증가 → FVD 15-20% 추가 감소
- 도메인 다양성 증가 → 미학습 도메인 성능 향상

**근거**: Make-A-Video는 10M 추가 데이터로 Make-A-Video(기본) 대비 89 → 81로 개선. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2820b430-a45b-4daf-b2d2-4795997b5783/2212.05199v2.pdf)

#### **B. 모델 크기 확장**

현재 Large 모델 464M 파라미터. 다음 스케일 고려:

| 모델 크기 | 트랜스포머 | 3D-VQ | 예상 FVD 개선 |
|----------|----------|-------|-------------|
| Large (현재) | 464M | 158M | FVD 9.9 |
| XLarge | ~1.5B | ~500M | FVD 8-9 예상 |
| Huge | ~3B | ~1B | FVD 7-8 예상 |

**한계**: 메모리/계산 비용 급증. ROI 감소.

#### **C. 아키텍처 개선**

**MAGVIT-v2에서 입증된 개선**:
1. **공동 토큰화**: 이미지-비디오 공유로 전이 학습 이득
2. **더 큰 코드북**: 2^18 크기로 표현력 증가
3. **카우살 ViT 도입**: 시간적 인과성 강화

**예상 FVD 개선**:
$$\Delta \text{FVD}_{\text{tokenizer}} \approx -3 \sim -5 \text{ 포인트}$$

#### **D. 다중 스케일 학습**

MAGVIT는 단일 해상도 학습. 다중 해상도 학습 추가 시:

```python
손실 = α * L(128×128) + β * L(256×256) + γ * L(512×512)
```

**잠재 이득**:
- 해상도 외삽 능력 향상
- 무선형 다중 스케일 기능
- 적응형 해상도 생성

#### **E. 조건부 전이 학습**

현재: 각 데이터셋별 별도 학습.
개선안: 소스 데이터(YouTube)에서 사전학습 → 타겟 도메인 미세조정

**예상 효과**:
- 소규모 도메인(BAIR) FVD 40-50% 감소
- 수렴 속도 3-5배 향상

### 9.3 일반화 성능의 이론적 한계

**일반화 오차 상한** (PAC 분석):

$$\mathcal{E}_{\text{gen}} \leq \mathcal{E}_{\text{train}} + O\left(\sqrt{\frac{\log(1/\delta)}{N}}\right) + \text{tokenizer bias}$$

**MAGVIT의 경우**:
- 학습 오류: 낮음 (우수한 FVD)
- 샘플 복잡도: $O(\log N)$ 의존성으로 인한 slow scaling
- **토크나이저 바이어스**: 4×8×8 압축의 정보 손실로 인한 고정 상한 존재

**결론**: 토크나이저를 개선하지 않으면 더 이상의 일반화 개선은 제한적.

***

## 10. 앞으로의 연구 방향 및 고려사항

### 10.1 MAGVIT 자체의 개선 방향

#### **1. 텍스트-비디오 생성 추가**

**문제**: MAGVIT는 이미지/클래스 조건만 지원.

**해결책**:
- CLIP 기반 텍스트 인코더 추가
- 텍스트 임베딩을 프롬프트 토큰으로 변환
- 크로스 어텐션 매커니즘 통합

**예상 영향**: 직접 비교 가능하게 Make-A-Video 범위로 확장.

#### **2. 고해상도 생성**

**현재**: 최대 128×128
**목표**: 512×512 이상

**기술적 고려사항**:
```
메모리 = O(seq_len × resolution²)
MAGVIT-L: 1,026 × 256² = 67M 토큰 (128×128)
↓
같은 메모리로 512×512는 불가능
```

**해결책**:
- Patch-based generation 도입
- 계층적 생성 (저→고 해상도)
- 윈도우 어텐션으로 메모리 절감

#### **3. 장시간 비디오**

**현재**: 16프레임 (0.64초 @25fps)
**목표**: 몇 초~분 단위 비디오

**접근법**:
- 슬라이딩 윈도우 디코딩
- 시간적 KV 캐시
- 점진적 프레임 생성

### 10.2 평가 메트릭의 발전

#### **현재 문제점**:

**FVD의 한계**:
- 공간적 품질에 과도하게 민감
- 시간적 평활성 미반영
- 특정 도메인에만 보정됨

**개선된 메트릭** (2024):
- **FVMD** (Fréchet Video Motion Distance): 모션 일관성 강조 [qiyan98.github](https://qiyan98.github.io/blog/2024/fvmd-1/)
- **VBench**: 16차원 계층적 평가 [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.pdf)

**MAGVIT를 VBench로 재평가 시**:
예상 강점 (시간적 평활성) vs 약점 (텍스트 정렬 부재) 식별 가능.

#### **권장사항**:
향후 MAGVIT 확장 연구는 FVD와 함께 VBench 및 FVMD 보고 필수.

### 10.3 후속 연구의 우선순위

#### **우선순위 1: 토크나이저 개선** (가장 높음)

**이유**: 생성 품질의 상한선을 결정.

**구체 방향**:
- MAGVIT-v2 통합 (공동 이미지-비디오)
- 더 큰 코드북 (2^20 이상)
- 연속-이산 하이브리드 토큰화

**예상 ROI**: FVD 5-10% 추가 개선.

#### **우선순위 2: 다중 조건 통합** (높음)

**이유**: 텍스트-비디오가 산업계 주요 수요.

**구체 방향**:
- 텍스트 인코더 추가
- 크로스-모달 어텐션
- 조건 가중치 학습

**예상 ROI**: 확장된 응용 범위.

#### **우선순위 3: 스케일 확장** (중간)

**이유**: 계산 비용 대비 이득 감소.

**구체 방향**:
- 메모리-효율 메커니즘 (Sparse attention)
- 컴파일 최적화
- 양자화 (INT8)

**예상 ROI**: 배포 비용 30-50% 절감.

#### **우선순위 4: 도메인 특화** (낮음)

**이유**: 일반화 모델 vs 특화 모델 trade-off.

**구체 방향**:
- 의료 영상 특화 모델
- 자율주행 특화 모델
- 애니메이션 생성 특화

**예상 ROI**: 니치 시장 지배.

### 10.4 근본적 연구 질문

#### **1. 마스킹 vs 확산: 기본 차이는?**

**MAGVIT (마스킹)**:
$$p(z_{t+1} | z_1, \ldots, z_t) = \prod_i p(z_i | \text{context})$$
병렬 예측, O(log T) 스텝.

**확산 모델**:
$$p(x_{t-1} | x_t) = \mathcal{N}(\mu_\theta(x_t, t), \sigma^2_\theta I)$$
순차 노이즈 제거, O(T) 스텝.

**미답 질문**: 마스킹 패러다임이 근본적으로 더 우월한가?

#### **2. 일반화의 한계는 어디인가?**

**가설 1**: 토크나이저 정보 손실로 인한 고정 상한.
**가설 2**: 모델 용량과 다양성의 trade-off.
**가설 3**: 학습 데이터 분포 범위의 본질적 제약.

#### **3. 멀티태스크 학습의 이점은 무엇인가?**

MAGVIT는 멀티태스크 학습으로 단일태스크 모델을 능가한다. 왜?
- 정규화 효과?
- 암시적 전이 학습?
- 작업 간 표현 공유?

***

## 11. 결론

### 11.1 종합 평가

MAGVIT는 **세 가지 차원에서** 비디오 생성의 최첨단을 정의했다:

**1. 기술적 혁신**: COMMIT 멀티변수 마스킹과 3D-VQ 토큰화의 조합은 조건부 생성의 새로운 패러다임을 제시했다. 비인과적 마스킹 문제를 수학적으로 우아하게 해결했다.

**2. 효율성 혁명**: 12 스텝 생성으로 확산 대비 100배, 자동회귀 대비 60배 빠르면서도 품질을 능가한다. 이는 실시간 응용을 가능하게 한다.

**3. 범용성**: 단일 모델로 10가지 작업을 수행하고 5개의 서로 다른 시각적 도메인에서 높은 성능을 유지한다. 기존의 작업-특화 모델 패러다임을 초월했다.

### 11.2 학술계 영향

MAGVIT는 이후 연구에 다음을 확립했다:
- **마스크 기반 병렬 생성**의 효율성 입증
- **토크나이저의 중요성** 강조 (MAGVIT-v2로 확대)
- **멀티태스크 학습의 효율성** 증명

402회 인용(2023년 이후)은 4년 내 rapid impact를 의미한다. [arxiv](https://arxiv.org/abs/2212.05199)

### 11.3 실무적 고려사항

**배포 관점**:
- 메모리 요구사항: 87M(B) 또는 464M(L) 파라미터
- 추론 지연: V100에서 16프레임 27ms
- 적합 분야: 실시간 비디오 생성, 에지 컴퓨팅

**제약**:
- 해상도: 현재 128×128 (고해상도 미지원)
- 조건: 클래스/이미지만 (텍스트 미지원)
- 길이: 16프레임 (장시간 비디오 미지원)

### 11.4 미래 전망

**단기 (1-2년)**:
- MAGVIT-v2 기반 상용화 모델 증가
- 텍스트-비디오 조건 추가 확장
- 고해상도 (256×256) 구현

**중기 (2-5년)**:
- 분 단위 비디오 생성 가능
- 물리 시뮬레이션 정확도 향상
- 멀티모달 통합 (비디오+오디오+텍스트)

**장기 (5년+)**:
- 세계 모델(World Model) 기반 구현
- 인터랙티브 비디오 생성
- 일반 시각 이해 모델 통합

***

## 참고문헌

<span style="display:none">[^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78]</span>

<div align="center">⁂</div>

[^1_1]: 2212.05199v2.pdf

[^1_2]: https://www.semanticscholar.org/paper/985f0c89c5a607742ec43c1fdc2cbfe54541cbad

[^1_3]: https://proceedings.iclr.cc/paper_files/paper/2024/file/036912a83bdbb1fd792baf6532f102d8-Paper-Conference.pdf

[^1_4]: https://arxiv.org/abs/2212.05199

[^1_5]: https://arxiv.org/html/2409.04410v3

[^1_6]: https://qiyan98.github.io/blog/2024/fvmd-1/

[^1_7]: https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.pdf

[^1_8]: https://ieeexplore.ieee.org/document/10205485/

[^1_9]: http://arxiv.org/pdf/2212.05199.pdf

[^1_10]: http://arxiv.org/pdf/2305.13311.pdf

[^1_11]: https://arxiv.org/pdf/2206.11894.pdf

[^1_12]: https://openaccess.thecvf.com/content/ICCV2025/papers/Mahapatra_Progressive_Growing_of_Video_Tokenizers_for_Temporally_Compact_Latent_Spaces_ICCV_2025_paper.pdf

[^1_13]: https://magvit.cs.cmu.edu/v2/

[^1_14]: https://kimjy99.github.io/논문리뷰/magvit-v2/

[^1_15]: https://www.semanticscholar.org/paper/b2faf2a4c1e0c1e5788309d83fe24d2d56555237

[^1_16]: https://ieeexplore.ieee.org/document/10278410/

[^1_17]: https://www.jmir.org/2023/1/e52865

[^1_18]: http://jcorth.com/2023/12/28/ortho-ai-the-dawn-of-a-new-era-artificial-intelligence-in-orthopaedics/

[^1_19]: https://iopscience.iop.org/article/10.1088/1361-6579/ad252f

[^1_20]: http://arxiv.org/pdf/2310.05737v1.pdf

[^1_21]: http://arxiv.org/pdf/2502.11663.pdf

[^1_22]: http://arxiv.org/pdf/2312.12468.pdf

[^1_23]: https://arxiv.org/abs/2303.12208

[^1_24]: https://arxiv.org/html/2403.08502v1

[^1_25]: https://www.semanticscholar.org/paper/MAGVIT:-Masked-Generative-Video-Transformer-Liu-Yao/b2faf2a4c1e0c1e5788309d83fe24d2d56555237

[^1_26]: https://arxiv.org/list/physics/new

[^1_27]: https://arxiv.org/html/2506.14168v1

[^1_28]: https://arxiv.org/html/2504.08959v1

[^1_29]: https://arxiv.org/pdf/2307.05909.pdf

[^1_30]: https://arxiv.org/pdf/2310.03937.pdf

[^1_31]: https://arxiv.org/html/2407.17877v1

[^1_32]: https://arxiv.org/html/2502.06768v3

[^1_33]: https://arxiv.org/html/2503.17076v1

[^1_34]: https://www.semanticscholar.org/paper/Chat-With-ChatGPT-on-Intelligent-Vehicles:-An-IEEE-Du-Teng/6c1e14093b5c751bd7ae2bbab559c037acb26c7e

[^1_35]: https://arxiv.org/html/2405.13218v1

[^1_36]: https://arxiv.org/html/2502.00382v1

[^1_37]: https://arxiv.org/list/math/new

[^1_38]: https://arxiv.org/html/2503.04606v3

[^1_39]: https://www.marktechpost.com/2023/01/22/meet-magvit-a-novel-masked-generative-video-transformer-to-address-ai-video-generation-tasks/

[^1_40]: https://www.siliconflow.com/articles/en/fastest-open-source-video-generation-models

[^1_41]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01985.pdf

[^1_42]: https://research.google/pubs/magvit-masked-generative-video-transformer/

[^1_43]: https://arxiv.org/html/2412.18688v2

[^1_44]: https://www.emergentmind.com/topics/masked-diffusion-models

[^1_45]: https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_MAGVIT_Masked_Generative_Video_Transformer_CVPR_2023_paper.pdf

[^1_46]: https://pinggy.io/blog/best_video_generation_ai_models/

[^1_47]: https://liner.com/review/vdt-generalpurpose-video-diffusion-transformers-via-mask-modeling

[^1_48]: https://openaccess.thecvf.com/content/CVPR2023/supplemental/Yu_MAGVIT_Masked_Generative_CVPR_2023_supplemental.pdf

[^1_49]: https://www.datacamp.com/blog/top-video-generation-models

[^1_50]: https://kimjy99.github.io/논문리뷰/maskdit/

[^1_51]: https://github.com/AlonzoLeeeooo/awesome-video-generation

[^1_52]: https://arxiv.org/html/2502.06768v1

[^1_53]: https://ieeexplore.ieee.org/document/11281048/

[^1_54]: https://arxiv.org/abs/2508.11183

[^1_55]: https://arxiv.org/html/2501.05442

[^1_56]: https://arxiv.org/pdf/2304.00325.pdf

[^1_57]: http://arxiv.org/pdf/2411.05222.pdf

[^1_58]: https://arxiv.org/pdf/2208.00934.pdf

[^1_59]: http://arxiv.org/pdf/2402.03161.pdf

[^1_60]: https://arxiv.org/pdf/2511.06863.pdf

[^1_61]: https://arxiv.org/abs/2310.05737

[^1_62]: https://arxiv.org/html/2507.01016v1

[^1_63]: https://arxiv.org/html/2310.05737v2

[^1_64]: https://arxiv.org/html/2410.05203v1

[^1_65]: https://www.arxiv.org/pdf/2508.09857.pdf

[^1_66]: https://arxiv.org/pdf/2312.03018.pdf

[^1_67]: https://arxiv.org/html/2508.09857v1

[^1_68]: https://arxiv.org/html/2508.11183v1

[^1_69]: https://arxiv.org/html/2410.05363v1

[^1_70]: https://arxiv.org/html/2507.02862v1

[^1_71]: https://yonsei.elsevierpure.com/en/publications/language-model-beats-diffusion-tokenizer-is-key-to-visual-generat

[^1_72]: https://openaccess.thecvf.com/content/CVPR2024/supplemental/Ge_On_the_Content_CVPR_2024_supplemental.pdf

[^1_73]: https://proceedings.neurips.cc/paper_files/paper/2024/file/31994923f58ae5b2d661b300bd439107-Paper-Conference.pdf

[^1_74]: https://daniel.inblog.ai/language-model-beats-diffusion-tokenizer-is-key-to-visual-generation-25088

[^1_75]: https://openreview.net/forum?id=aRD1NqcXTC

[^1_76]: https://ostin.tistory.com/324

[^1_77]: https://liner.com/ko/review/cogvideo-largescale-pretraining-for-texttovideo-generation-via-transformers

[^1_78]: https://liner.com/review/image-and-video-tokenization-with-binary-spherical-quantization
