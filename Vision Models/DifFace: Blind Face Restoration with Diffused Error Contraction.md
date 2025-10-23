# DifFace: Blind Face Restoration with Diffused Error Contraction

## 1. 핵심 주장과 주요 기여

**DifFace**는 사전 학습된 확산 모델(diffusion model)의 강력한 생성 능력을 활용하여 복잡하고 알려지지 않은 열화(degradation)에도 강건하게 대응할 수 있는 blind face restoration 방법입니다.[1]

**핵심 주장:**
- 기존 딥러닝 기반 방법들은 학습 데이터 밖의 복잡한 열화에 취약하며, 여러 손실 함수(fidelity, perceptual, adversarial loss)의 복잡한 조합과 하이퍼파라미터 튜닝이 필요합니다[1]
- DifFace는 단순히 L1 손실만으로 학습된 restoration backbone을 통해 LQ 이미지로부터 사전 학습된 확산 모델의 중간 상태(intermediate state)로의 전이 분포(transition distribution)를 설계합니다[1]
- 이 전이 분포는 restoration backbone의 예측 오차를 수축(error contraction)시켜, 알려지지 않은 열화에 더 강건하게 만듭니다[1]

**주요 기여:**
1. **Error Contraction 메커니즘**: 확산 과정의 스케일링 특성($$\sqrt{\alpha_N} < 1$$)을 활용하여 예측 오차를 자동으로 압축[1]
2. **단순화된 학습 파이프라인**: L1 손실만으로 학습 가능하며, 복잡한 adversarial loss나 perceptual loss 불필요[1]
3. **사전 학습된 확산 모델 활용**: 재학습 없이 기존 확산 모델의 이미지 prior를 leverage하여 효율성 및 일반화 성능 향상[1]
4. **명시적 fidelity-realism 제어**: 시작 timestep $$N$$과 DDIM의 $$\eta$$ 파라미터를 통해 복원 결과의 충실도와 사실성 간 균형 조절 가능[1]

***

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

**Blind Face Restoration (BFR)**은 노이즈, 블러, 다운샘플링 등 복잡한 열화를 겪은 저품질(LQ) 얼굴 이미지로부터 고품질(HQ) 이미지를 복원하는 ill-posed 역문제입니다.[1]

**기존 방법의 한계:**
- 합성 데이터로 학습된 모델은 실제 열화 모델과의 불일치(degradation mismatch)로 인해 성능이 크게 저하됩니다[1]
- 여러 제약 조건(L1/L2 loss, adversarial loss, perceptual loss, face-specific priors)을 동시에 사용하여 학습이 불안정하고 하이퍼파라미터 튜닝이 까다롭습니다[1]
- 심각한 열화 상황에서 복원 성능이 급격히 떨어집니다[1]

### 2.2 제안 방법: Diffused Error Contraction

DifFace는 사후 분포(posterior distribution) $$p(x_0|y_0)$$를 **전이 분포 $$p(x_N|y_0)$$**와 **사전 학습된 확산 모델의 역 마르코프 체인**으로 근사합니다[1].

#### 핵심 수식

**1) 사후 분포 분해:**

$$
p(x_0|y_0) = \int p(x_N|y_0) \prod_{t=1}^{N} p_\theta(x_{t-1}|x_t) dx_{1:N}
$$

여기서 $$x_0$$는 HQ 이미지, $$y_0$$는 LQ 이미지, $$x_N$$은 확산된 중간 상태, $$1 < N < T$$는 시작 timestep입니다.[1]

**2) 전이 분포 설계:**

$$
p(x_N|y_0) = \mathcal{N}(x_N; \sqrt{\alpha_N}f(y_0; w), (1-\alpha_N)I)
$$

- $$f(\cdot; w)$$: L1 손실로 학습된 diffused estimator (예: SwinIR, SRCNN)
- $$\alpha_N = \prod_{l=1}^{N}(1-\beta_l)$$: 확산 모델의 누적 노이즈 스케줄링 파라미터[1]

**3) Error Contraction 원리:**

Diffused estimator의 예측 오차를 $$e = x_0 - f(y_0; w)$$라 하면, 전이 분포에서 샘플링된 $$x_N$$은:

$$
x_N = \sqrt{\alpha_N}f(y_0; w) + \sqrt{1-\alpha_N}\zeta = \sqrt{\alpha_N}x_0 - \sqrt{\alpha_N}e + \sqrt{1-\alpha_N}\zeta
$$

여기서 $$\zeta \sim \mathcal{N}(0, I)$$이며, 예측 오차 $$e$$가 $$\sqrt{\alpha_N} < 1$$ 배로 압축됩니다.[1]

**4) KL Divergence 분석:**

설계된 전이 분포 $$p(x_N|y_0)$$와 목표 분포 $$q(x_N|x_0)$$ 간의 KL divergence는:

$$
D_{KL}[p(x_N|y_0) \| q(x_N|x_0)] = \frac{1}{2\kappa_N}\|e\|_2^2, \quad \kappa_N = \frac{\alpha_N}{1-\alpha_N}
$$

$$\kappa_N$$은 $$N$$이 증가할수록 단조 감소하므로, 더 큰 $$N$$을 선택하면 목표 분포에 더 가깝게 근사할 수 있지만, 과도하게 큰 $$N$$은 fidelity를 감소시킵니다.[1]

**5) DDIM 역방향 샘플링:**

$$x_N$$부터 $$x_0$$까지 DDIM sampler로 복원:

$$
\mu_\theta(x_t, t) = \sqrt{\alpha_{t-1}}\hat{x}_0^{(t)} + \sqrt{1-\alpha_{t-1}-\sigma_t^2}\varepsilon_\theta(x_t, t)
$$

$$
\sigma_t^2 = \eta \cdot \frac{1-\alpha_{t-1}}{1-\alpha_t}\beta_t, \quad \eta \in[1]
$$

여기서 $$\hat{x}\_0^{(t)} = \frac{x_t - \sqrt{1-\alpha_t}\varepsilon_\theta(x_t, t)}{\sqrt{\alpha_t}}$$이며, $$\eta=0.5$$로 설정하여 fidelity와 realism의 균형을 유지합니다.[1]

### 2.3 모델 구조

**전체 파이프라인:**
1. **입력**: LQ 이미지 $$y_0$$ (512×512)
2. **Diffused Estimator** $$f(\cdot; w)$$: 
   - 백본 네트워크: SwinIR 또는 SRCNN
   - 입력/출력 크기 조정: PixelUnshuffle/PixelShuffle 레이어 사용
   - 학습: FFHQ 데이터셋에서 L1 손실로 500k iteration 학습
   - 열화 모델: $$y = [(x \ast k_l) \downarrow_s + n_\sigma]_{JPEG_q} \uparrow_s$$ (Eq. 17)[1]
3. **전이 분포 샘플링**: $$x_N \sim p(x_N|y_0)$$
4. **사전 학습된 확산 모델**: 
   - FFHQ에서 학습된 1000-step 확산 모델
   - DDIM 가속화로 250 steps으로 축소 (실제 inference는 100 steps)[1]
5. **역방향 샘플링**: $$t=N, N-1, \ldots, 1$$에 대해 $$x_{t-1} \sim p_\theta(x_{t-1}|x_t)$$ 반복[1]

**SwinIR 백본 세부사항:**
- PixelUnshuffle로 512×512 → 64×64로 다운샘플링 (3회)
- Swin Transformer 기반 복원 네트워크
- PixelShuffle로 64×64 → 512×512로 업샘플링 (3회)[1]

### 2.4 성능 향상

**정량적 결과 (CelebA-Test):**
| Metric | GFPGAN | CodeFormer | DR2 | **DifFace** |
|--------|--------|------------|-----|-------------|
| PSNR↑ | 21.25 | 22.70 | 21.69 | **23.44** |
| SSIM↑ | 0.615 | 0.644 | 0.661 | **0.690** |
| LPIPS↓ | 0.532 | **0.438** | 0.490 | 0.461 |
| IDS↓ | 73.49 | **64.64** | 72.49 | 64.94 |
| LMD↓ | 12.26 | 8.01 | 8.69 | **6.06** |
| FID-G↓ | 68.61 | 25.99 | 43.11 | **20.29** |

DifFace는 대부분의 메트릭에서 최고 또는 2위 성능을 달성했습니다.[1]

**실제 데이터셋 (FID-F):**
- **WIDER** (심각한 열화): DifFace 37.52 vs CodeFormer 47.93
- **LFW** (경미한 열화): DifFace 46.80 vs CodeFormer 48.50
- **WebPhoto**: DifFace 81.60 vs VQFR 83.27[1]

**강건성 분석:**
- 다운샘플링 scale factor가 4→40으로 증가할 때, DifFace의 LPIPS 저하가 CodeFormer/VQFR보다 완만하여 심각한 열화에 더 강건함을 입증했습니다[1]

### 2.5 한계

1. **추론 속도**: 
   - 반복적 샘플링으로 인해 GAN 기반 방법(0.04~0.24s)보다 느림 (4.32s @ 512×512, Tesla V100)
   - DDIM 가속화(20 steps)로 0.92s까지 단축 가능하지만 성능 저하 발생[1]

2. **Non-reference 메트릭 한계**:
   - NIQE, NRQM, PI 등에서 GAN 기반 방법보다 낮은 점수
   - 확산 모델의 L2 손실 학습 특성상 smooth한 결과 생성으로 sharp image를 선호하는 메트릭과 불일치[1]

3. **하이퍼파라미터 민감성**:
   - 시작 timestep $$N$$ 선택이 fidelity-realism trade-off에 영향
   - 최적값(N=400) 탐색 필요[1]

---

## 3. 일반화 성능 향상 가능성

### 3.1 Error Contraction의 일반화 효과

**이론적 근거:**
- 전이 분포의 오차 압축 특성($$\sqrt{\alpha_N} < 1$$)은 diffused estimator $$f(\cdot; w)$$의 예측 오차가 학습 시 가정한 열화 모델과 다르더라도 자동으로 보정됩니다[1]
- KL divergence 분석에서 $$D_{KL} \propto \frac{1}{\kappa_N}\|e\|_2^2$$로, 오차 $$e$$의 영향이 $$\kappa_N$$ 배로 완화됩니다[1]

**실험적 검증:**
- Scale factor 증가 실험(4→40)에서 DifFace는 학습 시 보지 못한 극심한 열화(s=32~40)에서도 성능 저하가 완만했습니다[1]
- 실제 데이터셋(WIDER, LFW, WebPhoto)에서 합성 데이터로만 학습했음에도 SOTA 성능 달성[1]

### 3.2 사전 학습된 확산 모델의 역할

**풍부한 이미지 Prior:**
- FFHQ에서 unsupervised 방식으로 학습된 확산 모델은 수동으로 합성한 열화 데이터의 분포 불일치 영향을 줄입니다[1]
- 역 마르코프 체인($$x_N \rightarrow x_0$$)이 초기 예측 $$f(y_0; w)$$의 부족한 디테일을 보완하여 realistic한 얼굴 구조 생성[1]

**재학습 불필요:**
- SR3와 달리 확산 모델을 재학습하지 않아도 되므로, 다양한 도메인(얼굴 복원, inpainting, super-resolution)에 즉시 적용 가능[1]
- Natural image super-resolution 실험에서 ImageNet으로 학습된 확산 모델을 그대로 사용하여 IIGDM보다 우수한 성능(PSNR 22.05 vs 18.66)을 보였습니다[1]

### 3.3 다중 복원 결과 생성 능력

- 확산 모델의 stochastic sampling 특성으로 동일한 LQ 이미지로부터 다양하고 그럴듯한(plausible) 여러 HQ 복원 결과를 생성할 수 있습니다[1]
- 이는 ill-posed 문제의 본질(하나의 LQ에 여러 HQ 솔루션 존재)에 부합하며, 사용자가 선택할 수 있는 옵션을 제공합니다[1]

### 3.4 확장성: Inpainting 및 Super-Resolution

**Face Inpainting:**
- LaMa 백본을 diffused estimator로 사용
- 명시적 degradation 모델 $$y_0[i,j] = x_0[i,j]$$ (if M[i,j]=0)을 활용한 최적화 refinement (Eq. 15-16):
  $$
  \ddot{x}_0^{(t)} = \hat{x}_0^{(t)} + \frac{\gamma(1-M)^2 \odot y_0}{1 + \gamma(1-M)^2}
  $$
- CelebA-HQ에서 평균 LPIPS 0.212로 SOTA (RePaint 0.260, LaMa 0.223)[1]

**Face/Natural Image Super-Resolution:**
- 8x/16x face SR에서 재학습이 필요한 SR3보다 우수한 PSNR/SSIM
- 4x natural image SR에서 IIGDM 대비 PSNR +3.39, LPIPS -0.236 향상
- Bicubic 열화 외의 다른 degradation으로 generalization 테스트에서도 강건성 확인[1]

***

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 향후 연구에 미치는 영향

**1) 확산 모델 기반 복원의 새로운 패러다임:**
- 기존의 "LQ → HQ" 직접 매핑 대신 "LQ → 중간 확산 상태 → HQ" 경로를 제시하여, 복잡한 adversarial loss 없이도 고품질 복원이 가능함을 입증했습니다[1]
- Error contraction 개념은 CCDF 등 다른 inverse problem 연구에도 영향을 미치고 있습니다[1]

**2) 일반화 성능 개선 방향 제시:**
- Degradation mismatch 문제를 해결하는 실용적 접근법으로, blind restoration의 실제 응용 가능성을 높였습니다[1]
- 사전 학습된 생성 모델의 prior를 재학습 없이 활용하는 전략은 data-efficient AI 연구의 중요한 사례입니다[1]

**3) 다양한 복원 태스크로의 확장 가능성:**
- Inpainting, deblurring, denoising, colorization 등 다양한 image restoration 태스크에 쉽게 적용 가능한 프레임워크를 제공했습니다[1]

### 4.2 향후 연구 시 고려사항

**1) 추론 속도 최적화:**
- **과제**: 현재 inference time이 GAN 기반 방법보다 10~100배 느림
- **접근 방향**:
  - Consistency models, progressive distillation 등 최신 가속화 기법 적용
  - Knowledge distillation으로 경량 diffused estimator 개발
  - Latent diffusion models(LDM) 활용하여 latent space에서 복원 수행[1]

**2) Fidelity-Realism Trade-off 자동 제어:**
- **과제**: 시작 timestep $$N$$과 $$\eta$$ 설정이 수동적이며 이미지마다 최적값이 다를 수 있음
- **접근 방향**:
  - LQ 이미지의 열화 정도를 자동 추정하여 $$N$$을 adaptive하게 설정
  - 사용자 선호도 학습 기반 $$\eta$$ 자동 조정 메커니즘 개발[1]

**3) 더 강력한 Diffused Estimator 설계:**
- **과제**: L1 손실만으로 학습된 $$f(\cdot; w)$$의 초기 예측 품질이 최종 성능의 하한선을 결정
- **접근 방향**:
  - Self-supervised learning, contrastive learning으로 열화에 더 강건한 representation 학습
  - Transformer 기반 아키텍처(예: Restormer) 적용으로 long-range dependency 모델링 강화
  - Multi-scale feature fusion으로 다양한 열화 패턴 대응[1]

**4) Identity Preservation 강화:**
- **과제**: 극심한 열화 상황에서 identity 정보 손실 가능
- **접근 방향**:
  - DiffBFR처럼 identity restoration module 추가
  - ArcFace 등 face recognition loss를 diffused estimator 학습에 보조적으로 활용
  - 3D facial prior 통합으로 기하학적 일관성 유지[1]

**5) Non-reference Metrics 개선:**
- **과제**: 확산 모델의 smooth한 출력 특성이 NIQE, PI 등 메트릭과 불일치
- **접근 방향**:
  - 확산 모델의 출력 특성을 고려한 새로운 평가 메트릭 개발
  - Perceptual quality를 더 정확히 반영하는 learned metrics 연구[1]

**6) Unified Framework로의 확장:**
- **접근 방향**:
  - 단일 모델로 restoration, enhancement, editing을 모두 처리하는 통합 프레임워크 개발
  - Text-guided diffusion models과 결합하여 "restore the old photo with vintage style" 같은 복합 태스크 지원[1]

**7) 실제 응용을 위한 최적화:**
- **접근 방향**:
  - 모바일/엣지 디바이스 배포를 위한 경량화 연구
  - Batch processing 최적화로 대량 이미지 처리 효율성 향상
  - Privacy-preserving 복원 기법 연구 (federated learning, on-device processing)[1]

***

DifFace는 확산 모델의 강력한 생성 능력과 error contraction 메커니즘을 결합하여 blind face restoration의 일반화 성능을 크게 향상시켰으며, 단순한 학습 파이프라인(L1 손실만 사용)으로도 SOTA 성능을 달성했습니다. 향후 추론 속도 최적화와 adaptive parameter control 연구가 진행된다면, 실제 산업 환경에서도 널리 활용될 수 있는 실용적인 솔루션으로 발전할 가능성이 높습니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/73d659d4-9d33-43e7-91c8-fe6876abfe35/2212.06512v4.pdf)
