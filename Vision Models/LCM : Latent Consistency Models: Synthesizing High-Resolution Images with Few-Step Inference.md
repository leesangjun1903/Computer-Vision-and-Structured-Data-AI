
# Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference

## 1. 핵심 주장과 주요 기여

"Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference"는 2023년 칭화대학교 연구팀(Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, Hang Zhao)이 발표한 논문으로, 사전 학습된 Latent Diffusion Models(LDMs)에서 소수의 단계(2~4단계, 심지어 1단계)로 고해상도 이미지를 생성할 수 있는 새로운 접근 방식을 제시합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)

논문의 핵심 기여는 다음과 같습니다:

**1) Latent Consistency Models (LCMs)의 도입**: Consistency Models를 픽셀 공간에서 이미지의 잠재 공간으로 확장하여, Stable Diffusion 같은 대규모 사전 학습 모델에 직접 적용 가능하게 함. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)

**2) 단일 단계 Guided Distillation**: Classifier-Free Guidance(CFG)를 포함한 증강 Probability Flow ODE(PF-ODE)를 해결하여, 복잡한 2단계 distillation 절차를 단일 단계로 통합. 이를 통해 학습 비용을 44배 감소(45 A100 GPU days → 32 A100 GPU hours). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)

**3) Skipping-Step 기법**: 시간 스케줄에서 k 단계를 건너뛰어 1,000개 timestep을 ~50개로 축소하면서 빠른 수렴 달성. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)

**4) Latent Consistency Fine-tuning (LCF)**: 사전 학습된 LCM을 커스텀 데이터셋에 효율적으로 적응화하는 방법 제시. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)

## 2. 해결 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

Diffusion models는 이미지 생성 품질에서 우수하지만 근본적인 약점이 있습니다. 역확산 과정의 반복적 샘플링으로 인해 생성 속도가 매우 느려 실시간 응용이 제한됩니다. Latent Diffusion Models은 픽셀 공간 대신 압축된 잠재 공간에서 작업하여 효율성을 개선했지만, 여전히 고해상도 이미지 생성에 많은 단계가 필요합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)

기존 가속화 방법들은 제한사항을 가집니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)
- **ODE 솔버 기반** 방법(DDIM, DPM-Solver): 여전히 10~20단계 필요
- **Consistency Models**: 픽셀 공간에만 제한되어 고해상도 생성 부적합
- **Guided-Distillation**: 2단계 distillation으로 누적 오류 발생, 계산 집약적(최소 45 A100 GPU days)

### 2.2 제안하는 방법 및 수식

#### (1) Latent Space에서의 Consistency Distillation

Stable Diffusion의 사전 학습 오토인코더를 활용하여 이미지 $x$를 잠재 벡터 $z = \mathcal{E}(x)$로 압축합니다. 잠재 공간의 PF-ODE는: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)

$$\frac{dz_t}{dt} = f(t)z_t + \frac{g^2(t)}{2\sigma_t}\epsilon_\theta(z_t, c, t), \quad z_T \sim \mathcal{N}(0, \sigma_T^2 I)$$

여기서 $f(t)$는 드리프트 항, $g(t)$는 확산 항, $\epsilon_\theta$는 노이즈 예측 모델입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)

일관성 함수는 다음과 같이 매개변수화됩니다:

$$f_\theta(z, c, t) = c_{\text{skip}}(t)z + c_{\text{out}}(t)\left(z - \sigma_t\frac{\epsilon_\theta(z, c, t)}{\sigma_t}\right)$$

여기서 $c_{\text{skip}}(0) = 1$, $c_{\text{out}}(0) = 0$ 조건을 만족하여 $f_\theta(z, \cdot, 0) = z$를 보장합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)

Latent Consistency Distillation Loss는:

$$\mathcal{L}_{\text{LCD}}(\theta, \theta^-; \Psi) = \mathbb{E}_{z,c,n}\left[d\left(f_\theta(z_{t_{n+1}}, c, t_{n+1}), f_{\theta^-}(\hat{z}_{t_n}^\Psi, c, t_n)\right)\right]$$

여기서 $d(\cdot, \cdot)$는 거리 메트릭(제곱 $\ell_2$), $\theta^-$는 EMA로 업데이트된 target model 파라미터, $\Psi(\cdot, \cdot, \cdot)$는 ODE 솔버입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)

#### (2) Classifier-Free Guidance를 포함한 단일 단계 Guided Distillation

CFG를 적용한 노이즈 예측:

$$\hat{\epsilon}(z_t, \omega, c, t) = (1 + \omega)\epsilon_\theta(z_t, c, t) - \omega\epsilon_\theta(z_t, \emptyset, t)$$

이를 PF-ODE에 통합한 증강 ODE:

$$\frac{dz_t}{dt} = f(t)z_t + \frac{g^2(t)}{2\sigma_t}\hat{\epsilon}(z_t, \omega, c, t)$$

증강 일관성 함수:

$$f_\theta(z, \omega, c, t) = c_{\text{skip}}(t)z + c_{\text{out}}(t)\left(z - \sigma_t\frac{\hat{\epsilon}(z, \omega, c, t)}{\sigma_t}\right)$$

Augmented Consistency Distillation Loss:

$$\mathcal{L}_{\text{LCD}}(\theta, \theta^-; \Psi) = \mathbb{E}_{z,c,\omega,n}\left[d\left(f_\theta(z_{t_{n+1}}, \omega, c, t_{n+1}), f_{\theta^-}(\hat{z}_{t_n}^{\Psi,\omega}, \omega, c, t_n)\right)\right]$$

여기서:

$$\hat{z}\_{t_n}^{\Psi,\omega} = (1 + \omega)\Psi(z_{t_{n+1}}, t_{n+1}, t_n, c) - \omega\Psi(z_{t_{n+1}}, t_{n+1}, t_n, \emptyset)$$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)

#### (3) Skipping-Step 기법

인접한 timestep 사이의 작은 consistency loss 문제를 해결하기 위해, $k$ 단계 떨어진 timestep 간 일관성을 강제합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)

$$\mathcal{L}_{\text{LCD}}(\theta, \theta^-; \Psi) = \mathbb{E}_{z,c,\omega,n}\left[d\left(f_\theta(z_{t_{n+k}}, \omega, c, t_{n+k}), f_{\theta^-}(\hat{z}_{t_n}^{\Psi,\omega}, \omega, c, t_n)\right)\right]$$

일반적으로 $k = 20$을 사용하여 timestep schedule을 크게 축소합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)

#### (4) Latent Consistency Fine-tuning (LCF)

Consistency Training에서 영감을 받아, 커스텀 데이터셋 $\mathcal{D}_s$에 대해 사전 학습 LCM을 fine-tuning합니다. Teacher diffusion model에 의존하지 않으므로: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)

$$\mathcal{L}_{\text{LCF}} = \mathbb{E}_{z,n}\left[d\left(f_\theta(z_{t_{n+k}}, t_{n+k}, c, \omega), f_{\theta^-}(z_{t_n}, t_n, c, \omega)\right)\right]$$

여기서 노이즈는 직접 주입됩니다: $z_{t_{n+k}} = \sqrt{\alpha_{t_{n+k}}} z + \sqrt{1 - \alpha_{t_{n+k}}}\epsilon$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)

### 2.3 모델 구조

LCM은 기존 Stable Diffusion의 U-Net 아키텍처를 활용합니다. 주요 구조적 특징: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)

**1) Consistency Function 설계**: 
- 입력: 잠재 벡터 $z$, 텍스트 조건 $c$, guidance scale $\omega$, 시간 $t$
- 출력: 원래 데이터 $z_0$의 직접 예측
- EMA로 업데이트되는 target network로 안정성 확보

**2) CFG 인코딩**:
- Guidance scale $\omega$를 Fourier embedding으로 변환
- 원본 LCM backbone에 projected embedding 추가
- Zero parameter initialization으로 학습 안정성 확보 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)

**3) ODE 솔버 통합**:
- DDIM, DPM-Solver, DPM-Solver++ 중 선택 가능
- 학습 중에만 사용 (추론에서는 단일 forward pass)

### 2.4 성능 향상

| 메트릭 | 512×512 | 768×768 |
|--------|---------|---------|
| **1-Step FID** | 35.36 | 34.22 |
| **2-Step FID** | 13.31 | 16.32 |
| **4-Step FID** | 11.84 | 13.53 |
| **4-Step CLIP** | 28.84 | 28.60 |

표 1: LAION-5B-Aesthetics 데이터셋에서 LCM의 정량적 성능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)

**기준 모델과의 비교:**
- DDIM (2-step): FID 81.05 vs LCM 13.31 (6.1배 향상)
- DPM-Solver (2-step): FID 72.81 vs LCM 13.31 (5.5배 향상)  
- Guided-Distill (2-step): FID 33.25 vs LCM 13.31 (2.5배 향상) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)

**학습 효율성:**
- Guided-Distill: 45 A100 GPU days → LCM: 32 A100 GPU hours (42배 감소)
- 4,000 training iterations로 고품질 768×768 이미지 생성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)

**Ablation Study 결과:**
- ODE 솔버 선택: DPM-Solver variants가 더 큰 skipping step (k=50)에서 안정적
- Skipping step k=20: 수렴 속도와 성능의 최적 균형
- Guidance scale: $w > 8$에서 CLIP score 향상, 그러나 $w > 12$에서 FID 악화 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)

### 2.5 한계점

**1) 1-step 성능 격차**: 1-step inference에서 FID 34.22 vs 2-step 16.32로 큰 성능 저하. 이는 복잡한 생성 과정을 단일 forward pass로 표현하기 어려움을 시사합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)

**2) Guidance Scale 편향**: 높은 guidance scale에서 CLIP score는 향상되지만 FID는 악화됨. 다양성과 품질 간 트레이드오프 불완전. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)

**3) ODE 솔버 의존성**: 학습 중 고급 ODE 솔버 필요로 추론 단계는 간단하지만 학습 복잡도 증가. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)

**4) 데이터셋 제한**: LAION-5B-Aesthetics 고품질 부분(6.0 이상의 미적 점수)에서만 평가. 다양한 도메인의 일반화 능력 미검증. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)

**5) 아키텍처 유연성 제약**: Stable Diffusion 아키텍처에 특화되어 있어 다른 diffusion model로의 이전 가능성 불명확. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 구조적 일반화 개선 메커니즘

**잠재 공간의 의미론적 구조화**: LCM이 잠재 공간에서 작동함으로써 얻는 핵심 이점은 높은 의미론적 구조입니다. 잠재 공간의 명확한 의미론적 분리로 인해: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)

- 동일 클래스의 잠재들이 유사한 속도 방향을 가짐
- 클래스 간 속도 방향의 명확한 구분으로 ODE 근사 오류 감소
- 결과적으로 소수 단계 샘플링에서도 높은 품질 유지

이는 최근 연구(2024-2025)에서도 확인되어, 의미론적으로 분리된 잠재 공간이 few-step 샘플링에서 significantly better 성능을 보입니다. [arxiv](https://arxiv.org/html/2510.15301v1)

**One-stage Distillation의 오류 누적 감소**: Guided-Distill의 2단계 접근 방식과 달리 LCM의 단일 단계 방식은: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)
- 첫 번째 distillation에서의 오류가 두 번째 단계로 전파되지 않음
- 더 정확한 consistency enforcement 가능
- 학습 수렴 속도 향상

### 3.2 학습 안정성과 수렴 특성

**Skipping-Step 기법의 일반화 효과**: $k$ 값의 적절한 선택으로: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)
- 인접 timestep 간의 작은 consistency loss 문제 완화
- ODE 근사의 오류-이익 트레이드오프 최적화
- k=20 설정으로 다양한 ODE 솔버에서 안정적 수렴

Ablation study에서 k=5, 10, 20 모두 k=1 대비 8배 이상의 빠른 수렴 보여, 이 기법의 일반적 효과를 입증합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)

**EMA 기반 Target Model**: 지수 이동 평균으로 업데이트되는 target network ($\theta^- = \rho\theta^- + (1-\rho)\theta$, $\rho=0.999943$)는: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)
- 학습 안정성 향상으로 더 큰 배치 크기 허용 가능
- 다양한 데이터 분포에 대한 로버스트성 개선

### 3.3 Fine-tuning을 통한 도메인 적응

**Latent Consistency Fine-tuning (LCF)의 효과**: 사전 학습 LCM을 커스텀 데이터셋에 적응화할 때: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)
- Teacher diffusion model 의존성 제거로 자유로운 데이터셋 선택
- Consistency training framework로 빠른 적응 (30K iterations)
- Pokemon, Simpsons 데이터셋 실험에서 스타일 일관성 유지 확인

이는 새로운 도메인으로의 일반화 능력을 강력히 시사합니다.

### 3.4 가능한 일반화 한계 및 극복 방안

**현재 한계:**
- 1-step inference에서 충분하지 않은 성능 (FID 34.22)
- Guidance scale에 따른 성능 편차 (CLIP vs FID 트레이드오프)
- 미적 점수 필터링된 데이터셋에만 최적화

**개선 가능성:**
- 다단계 fine-tuning 전략으로 1-step 성능 향상
- 동적 guidance scale 적응화 메커니즘 개발
- 다양한 도메인의 대규모 데이터셋에서 사전 학습

## 4. 논문이 앞으로의 연구에 미치는 영향과 고려 사항

### 4.1 학문적 영향

**1) Consistency Models의 실제 응용 확장** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)
- Song et al.(2023)의 원본 consistency models는 픽셀 공간의 ImageNet(64×64), LSUN(256×256)에 제한
- LCM은 처음으로 고해상도(768×768) 텍스트-이미지 생성에 성공
- Latent space distillation의 효율성 입증으로 future work의 새로운 방향 제시

**2) Few-step Inference의 현실성 입증**
- 기존의 ODE solver 기반 방법은 10~20단계 필요
- LCM으로 2~4단계에서 고품질 달성 가능함을 실증
- 실시간 생성 응용의 현실성 강화

**3) Guided Distillation의 패러다임 전환** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)
- Guided-Distill의 계산 집약적 2단계 방식 대비 42배 효율 개선
- 단일 단계 guided distillation의 가능성 입증
- 향후 guided generative models의 가속화 연구 방향 제시

### 4.2 산업적 영향

**실시간 고해상도 생성의 가능성**
- 32 A100 GPU hours 학습으로 768×768 고품질 모델 구축 가능
- 추론에서 단일 forward pass만 필요해 edge device 배포 가능성 향상
- PIXART-δ(2024)에서 1024×1024 이미지를 0.5초 내 생성으로 실증 [arxiv](https://arxiv.org/abs/2401.05252)

**비용 효율성**
- 사전 학습 모델 재활용으로 처음부터의 학습 비용 회피
- Latent Consistency Fine-tuning으로 도메인별 맞춤화 용이
- 산업적 배포 시 GPU 비용 대폭 절감

### 4.3 향후 연구 시 고려할 점

**1) 1-step Inference 성능 개선** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)
현재 1-step에서 34.22의 FID는 2-step 대비 2배 이상 높습니다. 향후 연구에서:
- 다중 가능성(multiple modalities) 학습
- Diffusion forcing 같은 고급 distillation 기법 적용
- 부스팅 기법(ensemble methods) 활용

**2) 다양한 조건부 생성 태스크 확대** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)
논문의 결론에서 명시한 미래 방향:
- 텍스트 기반 이미지 편집(text-guided image editing)
- 인페인팅(inpainting)
- 초해상도 확대(super-resolution)
- 비디오 생성 (최근 DOLLAR(2025) 등에서 실증)

**3) 이론적 분석 강화**
- Consistency distillation의 수렴성(convergence guarantees)
- 최적성(optimality conditions) 분석
- ODE 근사 오류와 distillation loss 간의 관계

**4) 다양한 아키텍처 적응** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3a215461-ffe0-41de-97d5-e25cd6ca8ae2/2310.04378v1.pdf)
- Transformer 기반 diffusion models (DiT 등)로의 확장
- 다른 사전 학습 모델(diffusion autoencoders 등)과의 호환성
- Cross-architecture transfer learning 가능성 탐구

**5) 도메인 일반화 검증**
- 의료 이미징, 과학 시뮬레이션 등 특화 도메인 평가
- Cross-domain 일반화 능력 평가
- 분포 이동(distribution shift)에 대한 로버스트성 분석

**6) 샘플링 전략 최적화**
- Guidance scale의 동적 적응
- 단계별 피드백 기반 적응형 샘플링
- Learnable sampler 개발

### 4.4 2020년 이후 관련 최신 연구와의 비교

#### 4.4.1 Diffusion Model 가속화 연구 진화

| 연구 | 년도 | 방법 | 주요 특징 | 한계 |
|------|------|------|----------|------|
| DDIM | 2020 | ODE 기반 스킵 | 10~20 단계 가능 | 여전히 많은 단계 필요 |
| DPM-Solver | 2022 | 고차 ODE 솔버 | 10~20 단계에서 고품질 | ODE 근사 오류 존재 |
| Progressive Distillation | 2022 | 반복 distillation | 정확하지만 느린 수렴 | 다단계 프로세스 |
| Guided-Distill | 2023 | 2단계 guided distillation | CFG 통합 가능 | 계산 집약적, 오류 누적 |
| **LCM (본 논문)** | **2023** | **1단계 guided distillation** | **few-step + 고효율** | **1-step 성능 부족** |
| PIXART-δ | 2024 | LCM + ControlNet | 1024×1024 0.5초 | 통합 프레임워크 |
| Flash Diffusion | 2024 | 범용 distillation | 다양한 조건부 모델 | 상대적으로 낮은 효율 |
| Discrete Diffusion Forcing | 2025 | AR-diffusion 하이브리드 | LLM에서 2.5x 가속화 | 텍스트 도메인 특화 |

표 2: Diffusion model 가속화 기술의 진화 [ndss-symposium](https://www.ndss-symposium.org/wp-content/uploads/2025-2287-paper.pdf)

#### 4.4.2 Consistency Models 관련 응용

**직접 응용:**
- **Consistency Training**: Song et al.(2023)의 원본 framework로 처음부터 학습
- **Reinforcement Learning for Consistency Model (RLCM, 2024)**: RL로 task-specific reward 최적화 [arxiv](https://arxiv.org/html/2404.03673v2)
  - DDPO fine-tuned diffusion 대비 faster training 및 inference
  - 2-4 step inference에서 높은 품질 달성

**간접 영향:**
- **Latent Diffusion 최적화**: 의미론적으로 구조화된 잠재 공간 중요성 강조 [arxiv](https://arxiv.org/html/2510.15301v1)
  - Semantic dispersion이 높을수록 few-step sampling 성능 향상
  - LCM의 이론적 기초 제공

#### 4.4.3 고해상도 이미지 생성의 진화

| 연구 | 추론 속도 | 해상도 | 특징 |
|------|----------|--------|------|
| Stable Diffusion | ~5초 (50단계) | 512×512 | 기준 모델 |
| PIXART-α | ~3초 | 1024×1024 | 효율적 아키텍처 |
| **LCM-Stable Diffusion** | **0.4-0.8초** | **768×768** | **2-4 단계, 단일 GPU** |
| PIXART-δ | **0.5초** | **1024×1024** | LCM 기반 최신 SOTA |
| GECO (3D 생성) | **1초** | 256×256 | 2단계 distillation |

표 3: 고해상도 생성의 속도 진화 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11141031/)

#### 4.4.4 LCM의 차별적 기여

**1) 첫 번째 달성:**
- 고해상도 텍스트-이미지에 consistency models 적용
- Few-step inference (2~4단계)에서 상용 수준의 품질 달성
- Classifier-free guidance를 일관성 프레임워크에 통합

**2) 효율성 측면:**
- 단일 단계 guided distillation으로 계산 비용 42배 감소
- 4,000 iterations로 full-quality 모델 완성
- 사전 학습 모델 재활용으로 zero-shot distillation 가능

**3) 실용성:**
- 노트북 GPU에서도 추론 가능
- 도메인별 fine-tuning 용이 (LCF)
- 다양한 ODE 솔버와 호환

**한계에도 불구하고 영향력 확대:**
- PIXART-δ(2024)에서 ControlNet과 통합으로 controllable generation 달성 [arxiv](https://arxiv.org/abs/2401.05252)
- RLCM(2024)에서 RL 적용으로 reward 최적화 [arxiv](https://arxiv.org/html/2404.03673v2)
- 비디오 생성으로 확대 (DOLLAR, 2025) [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2025/papers/Ding_DOLLAR_Few-Step_Video_Generation_via_Distillation_and_Latent_Reward_Optimization_ICCV_2025_paper.pdf)

***

## 결론

Latent Consistency Models은 diffusion-based 생성 모델의 효율성과 실용성을 근본적으로 변화시킨 중요한 기여입니다. 핵심 혁신인 **1단계 guided distillation** 방식은 단순하지만 강력하며, **skipping-step 기법**으로 가속화된 수렴은 실무 배포 가능성을 크게 향상시켰습니다.

모델의 일반화 성능은 의미론적으로 구조화된 잠재 공간, 안정적인 EMA 학습, 그리고 유연한 fine-tuning 메커니즘으로 인해 좋은 잠재력을 가지고 있으나, 1-step 성능, guidance scale 편향, 데이터셋 제한 등의 한계도 명확합니다.

향후 연구는 **다양한 조건부 생성 태스크로의 확대**, **이론적 수렴성 분석**, **1-step 성능 개선**에 집중해야 합니다. 특히 2024-2025년의 후속 연구들(PIXART-δ, RLCM, DOLLAR, D2F 등)이 LCM의 기초 위에 구축되고 있다는 점은 이 논문의 학문적, 산업적 영향력을 강력히 입증합니다.

***

## 참고문헌

<span style="display:none">[^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46]</span>

<div align="center">⁂</div>

[^1_1]: 2310.04378v1.pdf

[^1_2]: https://arxiv.org/html/2510.15301v1

[^1_3]: https://arxiv.org/abs/2401.05252

[^1_4]: https://www.ndss-symposium.org/wp-content/uploads/2025-2287-paper.pdf

[^1_5]: https://arxiv.org/abs/2402.07211

[^1_6]: http://arxiv.org/pdf/2410.11795.pdf

[^1_7]: http://arxiv.org/pdf/2309.17074.pdf

[^1_8]: https://arxiv.org/html/2406.02347v2

[^1_9]: https://arxiv.org/pdf/2508.09192.pdf

[^1_10]: https://arxiv.org/html/2510.02212v2

[^1_11]: https://arxiv.org/html/2404.03673v2

[^1_12]: https://ieeexplore.ieee.org/document/11141031/

[^1_13]: https://openaccess.thecvf.com/content/ICCV2025/papers/Ding_DOLLAR_Few-Step_Video_Generation_via_Distillation_and_Latent_Reward_Optimization_ICCV_2025_paper.pdf

[^1_14]: https://ieeexplore.ieee.org/document/11198028/

[^1_15]: https://journal-center.litpam.com/index.php/jolls/article/view/3220

[^1_16]: https://link.aps.org/doi/10.1103/PhysRevD.110.016030

[^1_17]: https://www.banglajol.info/index.php/ijss/article/view/85772

[^1_18]: https://arxiv.org/abs/2312.04853

[^1_19]: https://dl.acm.org/doi/10.1145/3711896.3737858

[^1_20]: http://arxiv.org/pdf/2407.01425.pdf

[^1_21]: https://arxiv.org/pdf/2304.11267.pdf

[^1_22]: https://arxiv.org/html/2501.00124v1

[^1_23]: https://arxiv.org/html/2503.01323

[^1_24]: http://arxiv.org/pdf/2408.05636.pdf

[^1_25]: https://arxiv.org/html/2508.09192v1

[^1_26]: https://research.nvidia.com/labs/par/consistory/

[^1_27]: https://kimjy99.github.io/논문리뷰/latent-consistency-model/

[^1_28]: https://liner.com/ko/review/faster-diffusion-rethinking-the-role-of-the-encoder-for-diffusion

[^1_29]: https://openaccess.thecvf.com/content/CVPR2024/papers/Hollein_ViewDiff_3D-Consistent_Image_Generation_with_Text-to-Image_Models_CVPR_2024_paper.pdf

[^1_30]: https://blog.outta.ai/177

[^1_31]: https://github.com/AIoT-MLSys-Lab/Efficient-Diffusion-Model-Survey

[^1_32]: https://kimjy99.github.io/논문리뷰/consistory/

[^1_33]: https://arxiv.org/html/2510.02390v1

[^1_34]: https://neurips.cc/virtual/2024/poster/94408

[^1_35]: https://arxiv.org/abs/2311.10093

[^1_36]: https://openreview.net/pdf/11ae9c61ba598d3ef3620c3e75a108bb9ac8186d.pdf

[^1_37]: https://phd-frog.tistory.com/16

[^1_38]: https://openreview.net/forum?id=2sMk2ShRdP

[^1_39]: https://arxiv.org/html/2311.10093v4

[^1_40]: https://arxiv.org/html/2511.20592v1

[^1_41]: https://arxiv.org/html/2509.25180v1

[^1_42]: https://arxiv.org/pdf/2508.02193.pdf

[^1_43]: https://arxiv.org/abs/2510.14553

[^1_44]: https://arxiv.org/pdf/2509.26328.pdf

[^1_45]: https://arxiv.org/html/2508.03735v1

[^1_46]: https://arxiv.org/html/2510.03206v1
