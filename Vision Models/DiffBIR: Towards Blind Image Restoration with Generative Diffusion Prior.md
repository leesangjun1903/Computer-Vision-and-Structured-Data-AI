# DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior

---

## 1. 핵심 주장과 주요 기여 요약

DiffBIR는 사전 학습된 Stable Diffusion의 사전 지식(prior knowledge)을 활용하여 현실적인 복원 결과를 달성하는 통합 Blind Image Restoration(BIR) 프레임워크입니다.

DiffBIR는 blind image restoration 문제를 두 단계로 분리(decouple)합니다: 1) **열화 제거(degradation removal)**: 이미지 독립적 콘텐츠 제거, 2) **정보 재생(information regeneration)**: 손실된 이미지 콘텐츠 생성.

### 주요 기여 (Contributions)

1. BIR 문제를 복원 모듈(restoration module)과 생성 모듈(generation module)로 분리하는 2단계 설계를 통해, BSR(Blind Super-Resolution), BFR(Blind Face Restoration), BID(Blind Image Denoising)를 하나의 통합 프레임워크에서 최초로 최첨단(SOTA) 성능을 달성했습니다.

2. 텍스트-이미지 확산 사전(prior)을 활용한 **IRControlNet**을 제안하여 사실적인 이미지 재구성을 수행하며, 이것이 BIR 작업을 위한 견고한 생성 모듈 백본임을 포괄적 실험으로 입증하였습니다.

3. 학습 없이 샘플링 과정에서 작동하는 제어 가능한 모듈인 **Region-Adaptive Restoration Guidance**를 도입하여, 다양한 사용자 요구에 맞춰 품질(quality)과 충실도(fidelity) 간 유연한 트레이드오프를 달성합니다.

---

## 2. 상세 기술 분석

### 2.1 해결하고자 하는 문제

Blind Image Restoration(BIR)은 열화(degradation) 유형과 정도를 알 수 없는 상태에서 손상된 이미지를 복원하는 문제입니다. 전통적인 이미지 복원 모델은 고정된 열화 가정에 의존하지만, DiffBIR는 생성적 확산 모델의 힘을 활용하여 다양한 실제 세계 이미지 열화를 blind 방식으로 처리합니다.

기존 방법의 한계:
- GAN 기반 방법(Real-ESRGAN, BSRGAN): 과도하게 부드러운(over-smoothed) 결과 생성
- 단일 모델 방식: 열화 제거와 디테일 생성을 동시 학습 시 최적화 어려움
- ControlNet 직접 적용 시: 색상 이동(color shift) 문제 발생

### 2.2 제안 방법 및 수식

#### Stage 1: Restoration Module (RM) — 열화 제거

첫 번째 단계에서는 다양한 열화(diversified degradations)에 걸쳐 복원 모듈을 사전 학습하여 실제 환경에서의 일반화 능력을 향상시킵니다.

수정된 SwinIR을 복원 모듈로 사용하며, pixel unshuffle 연산으로 저화질 입력 $I_{LQ}$를 스케일 팩터 8로 다운샘플링한 후, $3 \times 3$ 합성곱 층으로 얕은 특징을 추출하고, 이후의 모든 transformer 연산은 저해상도 공간에서 수행됩니다. 깊은 특징 추출은 여러 개의 Residual Swin Transformer Block(RSTB)을 사용합니다.

**Stage 1의 손실 함수** — 복원 모듈의 파라미터는 L2 pixel loss를 최소화하여 최적화합니다:

$$\mathcal{L}_{RM} = \| f_{RM}(I_{LQ}) - I_{HQ} \|_2^2$$

여기서 $f_{RM}$은 복원 모듈, $I_{LQ}$는 저화질 입력, $I_{HQ}$는 고화질 원본입니다. 출력 $I_{RM}$은 열화가 제거된 고충실도 결과입니다.

**열화 모델(Degradation Model):**

더 넓은 열화 공간을 커버하기 위해 다양한 열화와 고차 열화를 고려한 포괄적 열화 모델을 사용합니다. 블러(blur), 리사이즈(resize), 노이즈(noise)가 실제 시나리오의 세 가지 핵심 요소이며, 다양한 열화에는 등방성/비등방성 가우시안 커널, 면적/이중선형/이중입방 리사이즈, 가우시안/포아송/JPEG 압축 노이즈가 포함됩니다.

#### Stage 2: Generation Module (IRControlNet) — 정보 재생

두 번째 단계는 잠재 확산 모델(latent diffusion model)의 생성 능력을 활용하여 사실적인 이미지 복원을 달성합니다. 구체적으로, 사전 학습된 Stable Diffusion의 생성 능력을 유지하면서 미세 조정(finetuning)을 위한 주입적 변조 서브네트워크인 LAControlNet을 도입합니다.

**Stable Diffusion 기초:**

Stable Diffusion 기반으로 구현되며, 오토인코더를 사전 학습하여 이미지 $x$를 인코더 $\mathcal{E}$로 잠재 변수 $z$로 변환하고 디코더 $\mathcal{D}$로 재구성합니다.

확산 과정(diffusion process)에서 분산 스케줄 $\beta_t$에 따른 가우시안 노이즈가 추가됩니다:

$$z_t = \sqrt{\bar{\alpha}_t} \, z_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

여기서 $\alpha_t = 1 - \beta_t$, $\bar{\alpha}\_t = \prod_{s=1}^{t} \alpha_s$

**Latent Diffusion 학습 목적 함수:**

$$\mathcal{L}_{LDM} = \mathbb{E}_{z_0, \epsilon \sim \mathcal{N}(0,I), t} \left[ \| \epsilon - \epsilon_\theta(z_t, c, t, c_{RM}) \|_2^2 \right]$$

여기서 $\epsilon_\theta$는 UNet 디노이저, $c$는 텍스트 조건, $c_{RM} = \mathcal{E}(I_{RM})$은 Stage 1 출력의 잠재 인코딩입니다.

**IRControlNet 구조:**

확산과 역확산은 잠재 공간에서 수행되므로, 조건도 동일한 공간에 투영되어야 합니다. IRControlNet은 이를 인식하고 사전 학습된 VAE 인코더 $\mathcal{E}$를 활용하여 효과적인 인코딩을 수행하며, ControlNet 대비 뚜렷한 향상을 달성했습니다.

조건 잠재 $\mathcal{E}(I_{reg})$을 무작위 샘플링된 노이즈 $z_t$와 연결(concatenate)하여 병렬 모듈의 입력으로 사용합니다. 이 연결 연산으로 인해 증가하는 채널 수에 대응하여, 새로 추가된 파라미터는 0으로 초기화하고 나머지 가중치는 사전 학습된 UNet 체크포인트에서 초기화합니다. 병렬 모듈의 출력은 원래 UNet 디코더에 더해집니다.

이 전략은 skip-connected features만 미세 조정함으로써 소규모 학습 데이터에서의 과적합을 완화하면서, Stable Diffusion의 고품질 생성 능력을 계승합니다.

#### Region-Adaptive Restoration Guidance (추론 시 품질-충실도 균형)

시간 $t$에서 UNet 디노이저가 노이즈 잠재 $z_t$의 노이즈 $\epsilon_t$를 예측한 후, 예측 노이즈 $\epsilon_t$를 $z_t$에서 제거하여 깨끗한 잠재(clean latent) $\tilde{z}_0$를 얻습니다:

$$\epsilon_t = \epsilon_\theta(z_t, c, t, c_{RM}), \quad \tilde{z}_0 = \frac{z_t - \sqrt{1 - \bar{\alpha}_t} \, \epsilon_t}{\sqrt{\bar{\alpha}_t}}$$

이후 고충실도 조건 $I_{RM}$과의 region-adaptive MSE 손실 함수를 픽셀 공간에서 적용하고, gradient descent로 $\tilde{z}_0$를 업데이트합니다. 먼저 Sobel 연산자로 gradient magnitude를 계산하고, 강한 gradient 신호를 가진 픽셀이 희소하므로 분위수 기반 분할을 수행합니다.

Region-adaptive guidance loss:

$$\mathcal{L}_{guide} = \| M \odot (\mathcal{D}(\tilde{z}_0) - I_{RM}) \|_2^2$$

여기서 $M$은 Sobel 연산자 기반으로 구한 region-adaptive 마스크이며, 저주파 영역은 고충실도 가이던스 이미지의 영향을 더 받고, 고주파 영역은 더 많은 생성 능력을 유지합니다. 또한 guidance scale을 조절하여 충실도와 품질 간 부드러운 전환을 달성할 수 있습니다.

업데이트 규칙:

$$\tilde{z}_0' = \tilde{z}_0 - s \cdot \nabla_{\tilde{z}_0} \mathcal{L}_{guide}$$

여기서 $s$는 사용자가 조절 가능한 guidance scale입니다.

### 2.3 모델 구조 요약

```
[전체 파이프라인]

I_LQ → [Stage 1: Restoration Module (SwinIR)] → I_RM (열화 제거)
         │
         ▼
I_RM → [VAE Encoder E] → c_RM (조건 잠재)
         │
         ▼
z_T (random noise) + c_RM → [IRControlNet + Frozen SD UNet] → z_0 → [VAE Decoder D] → I_diff (최종 출력)
         │
    [Region-Adaptive Restoration Guidance] (추론 시 선택적 적용)
```

추론 시 Stage 1 모델로는 다른 논문의 기존 모델을 사용합니다: BSR에는 BSRNet, BFR에는 DifFace의 SwinIR-Face, BID에는 SCUNet-PSNR을 사용하며, 학습된 IRControlNet은 모든 태스크에서 변경 없이 유지됩니다.

### 2.4 성능 향상

광범위한 실험을 통해 DiffBIR가 blind image super-resolution, blind face restoration, blind image denoising 작업에서 합성 및 실제 데이터셋 모두에 대해 최첨단 접근법 대비 우수성을 입증했습니다.

구체적 성과:
- RealSRSet과 Real47 실제 데이터셋에서 모든 메트릭에 걸쳐 최고 점수를 획득하며, 기존 방법 대비 어려운 실제 시나리오 처리에서 우수성을 입증했습니다.
- BSR 방법들 대비 (1) 자연스러운 텍스처 생성, (2) 의미론적 영역 재구성, (3) 작은 디테일 보존, (4) 심각한 열화 극복에서 더 효과적입니다.
- BFR 방법들 대비 (1) 가림(occlusion) 케이스 처리, (2) 얼굴 외 영역(모자, 귀걸이 등)의 만족스러운 복원이 가능합니다.

**Ablation Study 결과:**

- 복원 모듈을 제거하면 모델이 열화를 의미론적 콘텐츠로 해석하여 심각한 왜곡이 발생합니다.
- IRControlNet의 사전 학습된 VAE 인코더 사용은 표준 ControlNet 대비 색상 이동을 방지하고 더 나은 충실도를 유지하며 성능을 크게 향상시킵니다.
- IRControlNet은 모든 모델 변형 중 가장 빠른 모델 수렴을 달성하여 아키텍처 설계의 우수성을 보여줍니다.

### 2.5 한계

1. **추론 속도**: 다단계 확산 샘플링(예: 50 steps)이 필요하여 실시간 응용에 제약이 있음. StableSR과 DiffBIR는 텍스트 프롬프트가 부족하여 풍부한 텍스처 생성이 제한될 수 있습니다.
2. **텍스트/작은 패턴 처리**: Stable Diffusion 자체의 한계를 공유하여 텍스트, 매우 작은 패턴, 작은 얼굴 처리에 어려움
3. **GPU 메모리**: 고해상도 이미지 복원 시 높은 VRAM 요구량
4. **확률적 출력**: 확산 모델 특성상 동일 입력에 대해 매번 다른 결과 생성 가능

---

## 3. 모델의 일반화 성능 향상 가능성

DiffBIR의 일반화 성능은 여러 핵심 설계 요소에 의해 달성됩니다:

### 3.1 넓은 열화 범위를 통한 일반화

넓은 열화 범위를 가진 고전적 열화 모델을 사용하여 생성 모듈 학습을 위한 조건을 획득하며, RealESRGAN의 복잡하지만 좁은 열화 범위와 비교하여 넓은 열화 모델이 생성 능력의 더 나은 활용을 이끌어 복원 결과의 품질을 향상시킴을 관찰했습니다.

### 3.2 2단계 분리(Decoupling) 설계의 일반화 효과

여러 복원 작업을 단일 프레임워크로 통합하면서 최첨단 결과를 달성함으로써, DiffBIR는 별도의 전문 모델을 유지하는 것과 비교하여 배포를 단순화하고 자원 요구를 줄입니다.

핵심 메커니즘:
- **Stage 1 (RM)의 교체 가능성**: 제안된 2단계 파이프라인은 매우 유연하여, SwinIR 대신 다른 우수한 모델로 열화를 제거한 후 Stable Diffusion을 활용하여 디테일을 정제할 수 있습니다.
- **Stage 2 (IRControlNet)의 태스크 불변성**: 한 번 학습된 IRControlNet은 BSR, BFR, BID 등 모든 태스크에서 변경 없이 사용

### 3.3 사전 학습된 확산 모델의 생성적 사전 지식

사전 학습된 대규모 확산 모델을 blind restoration에 효과적으로 통합하는 것은 역문제에 생성적 사전을 적용하는 새로운 연구 방향을 열어줍니다.

### 3.4 일반화 향상을 위한 향후 방향

| 전략 | 설명 |
|------|------|
| **더 강력한 기반 모델** | SDXL, Stable Diffusion 3.0 등 차세대 확산 모델로 교체 |
| **텍스트 프롬프트 활용** | 의미론적 정보를 통한 조건부 생성 강화 (SeeSR, SUPIR 방식) |
| **적응적 열화 모델링** | 입력 이미지 품질에 따른 동적 열화 추정 |
| **교차 도메인 학습** | 의료 영상, 위성 영상 등 다양한 도메인 확장 |

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

2단계 분리 전략과 region-adaptive guidance 메커니즘은 계산 사진학, 의료 영상, 디지털 포렌식 분야에서 미래 개발에 영감을 줄 수 있는 견고한 아키텍처 기반을 확립합니다.

training-free guidance 메커니즘은 기존 디테일 보존부터 시각적 사실감 극대화까지 복원 특성을 동적으로 조정할 수 있는 실용적 가치를 제공하며, 다양한 품질-충실도 트레이드오프가 선호되는 다양한 응용 분야에 적합합니다.

### 4.2 앞으로 연구 시 고려할 점

1. **추론 효율화**: 확산 단계 수를 줄이는 distillation(ResShift, SinSR), 1-step 방법(OSEDiff) 등 경량화 연구 필요
2. **의미론적 정보 통합**: 텍스트 프롬프트, 캡셔닝 모델(LLaVA 등)을 통한 의미 인식 복원
3. **안정성 확보**: 확률적 출력의 일관성 문제 해결 (CCSR 등의 접근 참고)
4. **비디오 확장**: 이미지 기반 확산 모델(DiffBIR 등)을 개별 프레임에 적용하면 사실적 디테일 생성이 가능하지만, 프레임 간 일관성이 부족한 문제가 있습니다.
5. **스케일업**: 더 큰 모델과 데이터셋 활용 (SUPIR의 SDXL + 대규모 데이터 접근법)

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래 표는 확산 사전(diffusion prior)을 활용한 주요 Blind Image Restoration 방법들의 비교입니다:

| 방법 | 연도 | 기반 모델 | 조건 주입 방식 | 텍스트 활용 | 추론 스텝 | 주요 특징 |
|------|------|-----------|--------------|-----------|----------|----------|
| **Real-ESRGAN** | 2021 | RRDB+GAN | — | ✗ | 1 | 고차 열화 모델링 |
| **StableSR** | 2023 | SD 2.1 | Time-aware Encoder + SFT | ✗ | 200 | time-aware 인코더로 LQ 이미지 정보를 추출하여 사전 학습된 SD의 특징에 반영 |
| **DiffBIR** | 2023 | SD 1.5/2.1 | IRControlNet (VAE encoder) | ✗/v2.1에서 ✓ | 50 | 2단계 분리, region-adaptive guidance |
| **PASD** | 2023 | SD | ControlNet + 의미 프롬프트 | ✓ (짧은 캡션) | ~20 | 의미론적 프롬프트(짧은 캡션이나 태그)를 도입하여 더 미세한 의미 디테일로 결과를 풍부하게 함 |
| **SeeSR** | 2024 | SD 2.1 | ControlNet + DAPE | ✓ (태그) | ~50 | 태그와 추가 조건을 사용하여 T2I 모델 생성을 개선 |
| **SUPIR** | 2024 | SDXL | ControlNet | ✓ (긴 텍스트) | ~50 | 데이터셋 스케일업 및 긴 텍스트 설명과 함께 SDXL 사전 학습 모델로 지각 품질 향상 |
| **OSEDiff** | 2024 | SD | LoRA | ✓ | 1 | StableSR 대비 약 105배, SeeSR 대비 39배 빠른 추론 속도를 가지며, 1-step 방법 SinSR 대비 더 빠르면서 더 높은 출력 품질 |
| **ResShift/SinSR** | 2023-24 | 자체 DM | — | ✗ | 4-15 | 복원 목적으로 DM을 처음부터 학습하여 PSNR 등 충실도 메트릭에서 우수하나, 사전 학습 T2I 모델을 활용하지 않음 |
| **CCSR** | 2024 | SD | ControlNet | ✗ | 1-2 | 단일/다단계 확산 모두 유연하게 지원하면서 높은 충실도와 시각적 품질의 안정적 결과 생성 |
| **BIR-D** | 2024 | DDPM | 최적화 가능 커널 | ✗ | ~1000 | DDPM을 효과적 사전으로 활용하고, 최적화 가능 합성곱 커널로 각 역확산 단계에서 열화 함수를 동적으로 시뮬레이션 |
| **AdaptBIR** | 2024 | SD | 이중 인코더 | ✓ | ~50 | IQA(Image Quality Assessment) 방법으로 이미지를 정량적으로 분류하고, IQA 점수 가이드 하에 이중 인코더 열화 제거 모듈을 사용하여 더 나은 정보 보존 달성 |

### 핵심 비교 분석

DiffBIR와 PASD는 ControlNet과 유사한 방식으로 zero convolution을 사용하여 UNet 인코더를 학습함으로써 조건을 도입합니다. 반면:

- 확산 모델은 SD와 같은 사전 학습된 T2I 모델의 강력한 사전을 활용하여 SISR에서 더 나은 품질을 보여주며, StableSR은 미세 조정 인코더로, DiffBIR는 사전 복원 모듈로 반복 역확산 과정을 가이드하는 것이 일반적 패러다임입니다.

- DiffBIR와 SUPIR는 실제 세계 이미지 복원에 특화되어, RealSR과 DRealSR 데이터셋에서 더 나은 지각 품질을 달성합니다.

**DiffBIR 대비 후속 연구의 발전 방향:**
1. **텍스트 프롬프트 강화** (SeeSR → SUPIR): 의미론적 가이던스 추가
2. **추론 가속화** (OSEDiff, SinSR, CCSR): 1-step distillation
3. **모델 스케일업** (SUPIR): SDXL 기반 + 대규모 데이터
4. **플러그앤플레이 향상** (RAP-SR): RAP-SR의 플러그앤플레이 설계는 StableSR, DiffBIR, SeeSR 등 기존 확산 기반 SR 방법과 원활하게 통합되어 시각 품질과 객관적 메트릭 모두를 향상시킵니다.

---

## 참고 자료

1. **[논문 원본]** Lin, X., He, J., et al. "DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior," *ECCV 2024*, arXiv:2308.15070
2. **[GitHub]** XPixelGroup/DiffBIR — https://github.com/XPixelGroup/DiffBIR
3. **[프로젝트 페이지]** https://0x3f3f3f3fun.github.io/projects/diffbir/
4. **[ECCV 2024 Proceedings]** Springer LNCS Vol. 15117, https://doi.org/10.1007/978-3-031-73202-7_25
5. **[alphaXiv 분석]** https://www.alphaxiv.org/overview/2308.15070v3
6. **[DigitalOcean Tutorial]** "DiffBIR: High Quality Blind Image Restoration with Generative Diffusion Prior" — https://www.digitalocean.com/community/tutorials/diffbir
7. **[Survey]** "Diffusion Models for Image Restoration and Enhancement: A Comprehensive Survey," arXiv:2308.09388
8. **[OSEDiff]** "One-Step Effective Diffusion Network for Real-World Image Super-Resolution," NeurIPS 2024
9. **[BIR-D]** "Taming Generative Diffusion Prior for Universal Blind Image Restoration," arXiv:2408.11287
10. **[AdaptBIR]** "Adaptive Blind Image Restoration with latent diffusion prior for higher fidelity," *Pattern Recognition*, 2024
11. **[RAP-SR]** "RAP-SR: RestorAtion Prior Enhancement in Diffusion Models for Realistic Image Super-Resolution," arXiv:2412.07149
12. **[S3Diff]** "Degradation-Guided One-Step Image Super-Resolution with Diffusion Priors," arXiv:2409.17058
13. **[CCSR]** "Improving the Stability and Efficiency of Diffusion Models for Content Consistent Super-Resolution," *TIP 2026*
14. **[Hugging Face 페이퍼 페이지]** https://huggingface.co/papers/2308.15070
15. **[Reading Note]** https://zhangtemplar.github.io/diffbir/
