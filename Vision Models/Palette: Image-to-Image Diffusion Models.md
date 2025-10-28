# Palette: Image-to-Image Diffusion Models

## 1. 핵심 주장과 주요 기여

Palette는 조건부 확산 모델(conditional diffusion models)을 기반으로 한 통합된 image-to-image 변환 프레임워크입니다. 이 논문의 핵심 기여는 다음과 같습니다:[1]

**범용성과 단순성**: 단일 구조로 colorization, inpainting, uncropping, JPEG restoration 등 4가지 어려운 image-to-image 변환 작업에서 강력한 GAN 및 regression 기준 모델을 능가하며, 작업별 하이퍼파라미터 조정, 구조 맞춤화, 보조 손실 함수 없이도 우수한 성능을 달성했습니다.[1]

**다양성과 품질**: GAN 모델의 mode dropping 문제를 해결하고 높은 충실도와 다양성을 동시에 달성했습니다.[1]

**멀티태스크 학습**: 단일 generalist 모델이 여러 작업을 동시에 학습하여 task-specific 모델과 동등하거나 더 나은 성능을 보였습니다.[1]

**표준화된 평가 프로토콜**: ImageNet 기반의 통일된 평가 프로토콜을 제안하여 향후 연구의 기준을 제시했습니다.[1]

## 2. 해결하고자 하는 문제와 제안 방법

### 문제 정의

Image-to-image 변환 작업은 하나의 입력 이미지에 대해 여러 출력 이미지가 가능한 복잡한 역문제(inverse problem)입니다. 기존 GAN 기반 접근법은 다음과 같은 한계가 있었습니다:[1]

- 학습의 불안정성
- Mode dropping (출력 분포에서 모드 누락)
- 작업별 맞춤화 필요
- 출력 다양성 부족

### 제안 방법: 조건부 확산 모델

Palette는 확산 모델의 denoising 과정을 입력 이미지 $$\mathbf{x}$$에 조건화하여 $$p(\mathbf{y}|\mathbf{x})$$ 분포를 학습합니다[1].

**Forward Diffusion Process (순방향 확산 과정)**:

데이터 포인트 $$\mathbf{y}_0$$에 반복적으로 가우시안 노이즈를 추가하는 마르코프 과정입니다:

$$
q(\mathbf{y}_{t+1}|\mathbf{y}_t) = \mathcal{N}(\mathbf{y}_t; \sqrt{\alpha_t}\mathbf{y}_{t-1}, (1-\alpha_t)\mathbf{I})
$$

$$
q(\mathbf{y}_{1:T}|\mathbf{y}_0) = \prod_{t=1}^{T} q(\mathbf{y}_t|\mathbf{y}_{t-1})
$$

여기서 $$\alpha_t$$는 noise schedule의 하이퍼파라미터입니다. 이 과정은 marginalize하여 다음과 같이 표현할 수 있습니다:

$$
q(\mathbf{y}_t|\mathbf{y}_0) = \mathcal{N}(\mathbf{y}_t; \sqrt{\gamma_t}\mathbf{y}_0, (1-\gamma_t)\mathbf{I})
$$

여기서 $$\gamma_t = \prod_{t'=1}^{t} \alpha_{t'}$$입니다.[1]

**Denoising Loss Function (학습 목표)**:

노이즈가 추가된 이미지 $$\tilde{\mathbf{y}}$$가 주어졌을 때:

$$
\tilde{\mathbf{y}} = \sqrt{\gamma}\mathbf{y}_0 + \sqrt{1-\gamma}\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})
$$

신경망 $$f_\theta$$는 다음 손실 함수를 최소화하도록 학습됩니다:

$$
\mathbb{E}_{(\mathbf{x},\mathbf{y})}\mathbb{E}_{\boldsymbol{\epsilon}\sim\mathcal{N}(0,\mathbf{I})}\mathbb{E}_{\gamma}\left[\|f_\theta(\mathbf{x}, \sqrt{\gamma}\mathbf{y} + \sqrt{1-\gamma}\boldsymbol{\epsilon}, \gamma) - \boldsymbol{\epsilon}\|_p^p\right]
$$

여기서 $$p=2$$ (L2 norm)를 사용했습니다. 이 목표는 노이즈 벡터 $$\boldsymbol{\epsilon}$$을 예측하도록 모델을 학습시킵니다.[1]

**Reverse Process (역방향 과정/추론)**:

추론 시에는 표준 가우시안 노이즈 $$\mathbf{y}_T \sim \mathcal{N}(0, \mathbf{I})$$에서 시작하여 $$T$$번의 반복적 refinement를 통해 이미지를 생성합니다. 각 단계에서 $$\mathbf{y}_0$$의 추정값은:

$$
\hat{\mathbf{y}}_0 = \frac{1}{\sqrt{\gamma_t}}\left[\mathbf{y}_t - \sqrt{1-\gamma_t}f_\theta(\mathbf{x}, \mathbf{y}_t, \gamma_t)\right]
$$

이를 통해 posterior 분포 $$q(\mathbf{y}_{t-1}|\mathbf{y}_0, \mathbf{y}_t)$$의 평균을 다음과 같이 파라미터화합니다:

$$
\mu_\theta(\mathbf{x}, \mathbf{y}_t, \gamma_t) = \frac{1}{\sqrt{\alpha_t}}\left[\mathbf{y}_t - \frac{1-\alpha_t}{\sqrt{1-\gamma_t}}f_\theta(\mathbf{x}, \mathbf{y}_t, \gamma_t)\right]
$$

반복적 refinement는 다음과 같이 수행됩니다:

$$
\mathbf{y}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left[\mathbf{y}_t - \frac{1-\alpha_t}{\sqrt{1-\gamma_t}}f_\theta(\mathbf{x}, \mathbf{y}_t, \gamma_t)\right] + \sqrt{1-\alpha_t}\boldsymbol{\epsilon}_t
$$

여기서 $$\boldsymbol{\epsilon}_t \sim \mathcal{N}(0, \mathbf{I})$$이며, 이는 Langevin dynamics의 한 단계와 유사합니다.[1]

### 모델 구조

**U-Net 기반 아키텍처**: Palette는 Dhariwal & Nichol (2021)의 256×256 class-conditional U-Net을 기반으로 합니다. 주요 수정사항은:

1. Class-conditioning 제거
2. 입력 소스 이미지를 concatenation을 통해 추가 조건화
3. 32×32, 16×16, 8×8 해상도에서 global self-attention layers 사용
4. ResNet blocks와 group normalization 활용[1]

**Self-Attention의 중요성**: 논문은 self-attention layers의 중요성을 실증적으로 입증했습니다. Self-attention을 제거하고 fully convolutional 구조로 대체하면 성능이 저하됩니다. 예를 들어 inpainting 작업에서:

- Global Self-Attention: FID 7.4, IS 164.8, PD 67.1
- Dilated Convolutions: FID 8.0, IS 157.5, PD 70.6
- More ResNet Blocks: FID 8.1, IS 157.1, PD 71.9
- Local Self-Attention: FID 9.4, IS 149.8, PD 78.2[1]

Global self-attention이 long-range dependencies를 모델링하는 데 필수적임을 보여줍니다.

## 3. 성능 향상 및 한계

### 성능 향상

**Colorization**: ImageNet validation set에서 FID 15.78, IS 200.8, CA 72.5%를 달성하여 기존 최고 성능(ColTran의 FID 19.37)을 크게 능가했습니다. Human evaluation에서 fool rate 47.80%를 기록하여 ColTran의 36.55%보다 10% 이상 향상되었습니다.[1]

**Inpainting**: 20-30% free-form mask에서 ImageNet FID 5.2, Places2 FID 11.7을 달성하여 DeepFillv2 (ImageNet FID 9.4, Places2 FID 13.5)와 Co-ModGAN (Places2 FID 12.4)을 능가했습니다.[1]

**Uncropping**: ImageNet에서 FID 5.8, Places2에서 FID 3.53을 달성하여 Boundless (각각 18.7, 11.8)를 큰 폭으로 앞섰습니다. Human evaluation에서 fool rate 40%를 기록했습니다.[1]

**JPEG Restoration**: Quality Factor 5에서 FID 8.3으로 regression baseline (FID 29.0)을 크게 능가했습니다. 기존 연구들이 QF ≥ 10에 제한되었던 것과 달리 QF 5까지 효과적으로 처리했습니다.[1]

**Sample Diversity**: L2 loss를 사용한 Palette는 L1 loss보다 더 높은 sample diversity를 보였습니다. Inpainting에서 LPIPS diversity score: L2 0.13, L1 0.11; Colorization에서: L2 0.15, L1 0.09.[1]

**Multi-task Performance**: 단일 multi-task 모델이 JPEG restoration에서 task-specific 모델을 능가했고 (multi-task FID 7.0 vs task-specific FID 8.3), inpainting과 colorization에서도 경쟁력 있는 성능을 보였습니다.[1]

### 한계

**추론 속도**: Palette는 1000번의 refinement steps가 필요하여 GAN 모델보다 훨씬 느립니다. TPUv4에서 이미지당 0.8초가 소요되어 실시간 응용에는 제한적입니다.[1]

**해상도 일반화**: Group normalization과 self-attention layers 사용으로 임의의 입력 해상도에 일반화할 수 없습니다. 256×256 해상도로 학습되어 다른 해상도에는 fine-tuning이나 patch-based inference가 필요합니다.[1]

**암묵적 편향(Implicit Biases)**: 다른 생성 모델과 마찬가지로 학습 데이터의 편향을 학습할 수 있어 실제 배포 전 연구와 완화가 필요합니다.[1]

**Long-range Consistency**: Panorama uncropping 실험에서 8번 반복 적용 후에도 놀라울 정도로 robust했지만, 매우 긴 시퀀스에서는 일관성 문제가 발생할 수 있습니다.[1]

## 4. 일반화 성능 향상 가능성

### 작업 간 일반화

Palette의 가장 중요한 일반화 능력은 **작업 간 전이(cross-task transfer)**입니다:

**통합 프레임워크의 효과**: 동일한 U-Net 구조와 학습 절차로 4가지 서로 다른 작업에서 SOTA 성능을 달성했습니다. 이는 확산 모델이 다양한 image-to-image 변환 작업의 underlying distribution을 효과적으로 학습할 수 있음을 보여줍니다.[1]

**Multi-task Learning**: ImageNet과 Places2를 혼합하여 학습한 모델 (Palette I+P)은 inpainting에서 ImageNet과 Places2 모두에서 강력한 성능을 보였습니다. 흥미롭게도 ImageNet만으로 학습한 모델도 Places2에서 경쟁력 있는 성능을 보여 도메인 간 일반화 능력을 입증했습니다.[1]

**Generalist vs Specialist**: Colorization, inpainting, JPEG restoration을 동시에 학습한 단일 generalist 모델이 JPEG restoration에서 task-specific 모델을 능가하고 다른 작업에서도 competitive한 성능을 달성했습니다. 이는 multi-task learning이 각 작업의 성능을 저해하지 않고 오히려 향상시킬 수 있음을 시사합니다.[1]

### 데이터셋 간 일반화

**ImageNet-Places2 Transfer**: Inpainting 실험에서 ImageNet으로만 학습한 모델 (Palette I)이 Places2에서도 우수한 성능을 보였습니다. 예를 들어 20-30% free-form mask에서 Palette (I)는 Places2 FID 11.8을 기록했습니다.[1]

**Large-scale Training의 효과**: 논문은 현대적 아키텍처와 large-scale training의 중요성을 강조합니다. L2 regression baseline조차도 task-specific 기존 방법들(PixColor, ColTran)을 능가했습니다.[1]

### 일반화 성능 향상 전략

**Self-Attention의 역할**: Global self-attention layers는 long-range dependencies를 모델링하여 복잡한 장면에서의 일관성을 유지합니다. 이는 다양한 이미지 도메인과 작업에서의 일반화에 핵심적입니다.[1]

**L2 Loss의 다양성**: L2 loss가 L1보다 높은 sample diversity를 제공하여 모델이 출력 분포를 더 충실하게 포착합니다. 이는 다양한 입력에 대해 plausible한 여러 출력을 생성할 수 있는 능력을 향상시킵니다.[1]

**조건부 학습의 이점**: Unconditional 모델을 repurpose하는 대신 직접적인 conditional learning을 통해 더 나은 일반화를 달성했습니다. Palette는 모든 refinement step에서 noiseless observation에 조건화되어 더 안정적인 학습을 제공합니다.[1]

### 잠재적 개선 방향

**더 긴 학습**: Multi-task 모델은 동일한 training steps로 학습되었는데, 더 긴 학습으로 성능이 향상될 것으로 예상됩니다.[1]

**더 많은 작업**: 논문은 4가지 작업만 다루었지만, super-resolution, deblurring, segmentation 등 더 많은 작업으로 확장 가능성이 있습니다.

**해상도 일반화**: Self-attention의 해상도 제약을 해결하는 기술 (예: local attention, axial attention)을 통해 임의 해상도로의 일반화가 가능할 것입니다.

## 5. 향후 연구에 미치는 영향과 고려사항

### 연구에 미치는 영향

**패러다임 전환**: Palette는 image-to-image 변환 분야에서 GAN 중심에서 확산 모델로의 패러다임 전환을 가속화했습니다. 확산 모델이 GAN의 학습 불안정성과 mode collapse 문제 없이도 높은 품질과 다양성을 달성할 수 있음을 입증했습니다.[1]

**통합 프레임워크**: 작업별 맞춤화 없이 다양한 image-to-image 작업을 처리할 수 있는 통합 프레임워크의 가능성을 제시했습니다. 이는 foundation model 연구 방향과 일치합니다.[1]

**평가 표준화**: ImageNet ctest10k와 Places2 places10k를 벤치마크로 제안하고 FID, IS, CA, PD를 포함한 통합 평가 프로토콜을 제시하여 향후 연구의 재현성과 비교 가능성을 크게 향상시켰습니다.[1]

**Multi-task Learning 가능성**: Single generalist 모델이 specialist 모델과 경쟁하거나 능가할 수 있음을 보여 multi-task diffusion models의 추가 탐구를 장려합니다.[1]

### 향후 연구 시 고려사항

**추론 속도 개선**: 
- Fast sampling 기술 적용 필요 (DDIM, DPM-Solver 등)
- 논문 발표 후 여러 연구들이 refinement steps를 10-50 steps로 줄이는 방법을 제안했으며, 이를 Palette에 적용하는 연구가 필요합니다[1]
- Knowledge distillation을 통한 student model 학습 고려

**해상도 확장**:
- Self-attention의 해상도 제약 해결 (local attention, linear attention)
- Patch-based inference 또는 progressive training 전략
- Cascaded diffusion models를 통한 고해상도 생성

**더 많은 작업으로의 확장**:
- Video-to-video translation
- 3D-aware image generation
- Multi-modal conditioning (text, sketch, semantic maps)

**효율성 개선**:
- Latent diffusion models: 픽셀 공간 대신 latent space에서 작동하여 계산 효율성 향상
- Progressive distillation
- Continuous-time formulation

**일반화 능력 강화**:
- Zero-shot 또는 few-shot adaptation
- Domain adaptation 기술
- Cross-domain consistency 유지 메커니즘

**품질과 다양성 균형**:
- L1/L2 loss 외의 다른 손실 함수 탐구
- Classifier-free guidance 적용
- Sample quality와 diversity의 trade-off 조절 메커니즘

**실용성 고려**:
- 학습 데이터 편향 분석 및 완화
- Failure case 분석 및 robustness 개선
- User control 및 interactive editing 기능
- 윤리적 사용 가이드라인

**멀티태스크 최적화**:
- Task balancing 전략
- Task-specific adapter 활용
- Meta-learning 접근법

Palette는 확산 모델이 image-to-image 변환의 강력하고 범용적인 프레임워크임을 입증했으며, 향후 연구는 추론 속도, 해상도 확장성, 더 넓은 작업 범위로의 일반화에 초점을 맞춰야 할 것입니다. 특히 의료 영상 처리 분야에서 bone suppression과 같은 특수 작업에 Palette의 아이디어를 적용할 때는 도메인 특화 데이터의 부족, 정확성 요구사항, 해석 가능성 등을 고려해야 합니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/efd80141-2ea6-4053-9280-b08d8d4b4ac4/2111.05826v2.pdf)
