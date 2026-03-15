# Wavelet Diffusion Models are fast and scalable Image Generators

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
확산 모델(Diffusion Model)은 고품질 이미지 생성에서 GAN을 능가하지만, **훈련 및 추론 속도가 극도로 느려** 실시간 응용에 부적합하다. 본 논문은 **웨이블릿 변환(Wavelet Transform)을 확산 모델에 통합**하여, 이미지 품질을 유지하면서도 **훈련·추론 속도를 획기적으로 개선**할 수 있음을 주장한다.

### 주요 기여
1. **웨이블릿 기반 확산 프레임워크(Wavelet Diffusion Framework)**: 웨이블릿 서브밴드의 차원 축소를 활용하여 확산 모델을 가속하면서도 고주파 성분을 통해 시각적 품질을 유지
2. **이미지 및 피처 수준 모두에서의 웨이블릿 분해**: 모델의 강건성과 실행 속도를 동시에 향상
3. **재구성 손실(Reconstruction Loss)** 도입으로 학습 수렴 속도 향상
4. **SOTA 훈련·추론 속도 달성**: DDGAN 대비 **2.5배 이상 빠른** 추론 속도, StyleGAN에 근접하는 실시간 수준의 성능 달성

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

확산 모델의 근본적 한계는 **느린 샘플링 속도**이다:

| 모델 | 샘플링 스텝(NFE) | 32×32 이미지 생성 시간 |
|------|-----------------|---------------------|
| DDPM | 1000 | ~80.5초 |
| Score SDE | 2000 | ~423초 |
| DDIM | 50 | ~4초 |
| DDGAN | 4 | ~0.21초 |
| **StyleGAN2** | **1** | **~0.04초** |

DDGAN이 샘플링 스텝을 4로 줄여 큰 진전을 이루었지만, 여전히 StyleGAN2 대비 **4배 이상 느리며**, 해상도 증가 시 속도 격차가 **더욱 확대**된다. 또한 DDGAN은 훈련 시간이 길고 수렴이 느리다는 문제도 존재한다.

### 2.2 제안하는 방법

#### (A) 배경: 확산 모델의 수학적 정의

전통적 확산 과정에서, 입력 $x_0$에 대한 시간 단계 $t$에서의 사후 확률은:

$$q(x_t \mid x_0) = \mathcal{N}\left(x_t;\, \sqrt{\bar{\alpha}_t}\, x_0,\, (1-\bar{\alpha}_t)\,\mathbf{I}\right) $$

여기서 $\alpha_t = 1-\beta_t$, $\bar{\alpha}\_t = \prod_{s=1}^{t} \alpha_s$이며, $\beta_t \in (0,1)$는 분산 스케줄이다.

역과정의 파라미터화:

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\left(x_{t-1};\, \mu_\theta(x_t,t),\, \sigma_t^2 \mathbf{I}\right) $$

DDGAN의 적대적 목적 함수:

$$\min_\phi \max_\theta \sum_{t \geq 1} \mathbb{E}_{q(\mathbf{x}_t)}\Big[\mathbb{E}_{q(\mathbf{x}_{t-1}\mid\mathbf{x}_t)}\big[-\log D_\phi(\mathbf{x}_{t-1}, \mathbf{x}_t, t)\big] + \mathbb{E}_{p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)}\big[\log D_\phi(\mathbf{x}_{t-1}, \mathbf{x}_t, t)\big]\Big] $$

#### (B) 웨이블릿 기반 확산 스킴 (Wavelet-based Diffusion Scheme)

**핵심 아이디어**: 원본 이미지 공간이 아닌 **웨이블릿 스펙트럼 공간**에서 디노이징을 수행한다.

입력 이미지 $x \in \mathbb{R}^{3 \times H \times W}$를 Haar 웨이블릿 변환(DWT)으로 분해:

$$x \xrightarrow{\text{DWT}} y \in \mathbb{R}^{12 \times \frac{H}{2} \times \frac{W}{2}}$$

여기서 저주파 필터 

```math
L = \frac{1}{\sqrt{2}}\begin{bmatrix}1 & 1\end{bmatrix}
```
 , 고주파 필터 

 ```math
 H = \frac{1}{\sqrt{2}}\begin{bmatrix}-1 & 1\end{bmatrix}
```
 를 사용하여 4개의 커널 $LL^T, LH^T, HL^T, HH^T$로 stride 2 연산을 수행, 4개 서브밴드 $X_{ll}, X_{lh}, X_{hl}, X_{hh}$를 생성한다.

**효과**:
- 공간 해상도가 **4배 축소** → 연산량 대폭 감소
- 고주파 정보(엣지, 디테일) **명시적 보존** → 생성 품질 유지

**적대적 목적 함수** (웨이블릿 공간):

$$\mathcal{L}\_{adv}^D = -\log D(y_{t-1}, y_t, t) + \log D(y'_{t-1}, y_t, t) $$

$$\mathcal{L}\_{adv}^G = -\log D(y'_{t-1}, y_t, t) $$

**재구성 항(Reconstruction Term)**:

$$\mathcal{L}_{rec} = \|y'_0 - y_0\| $$

이 항은 주파수 정보 손실을 방지하고 웨이블릿 서브밴드 간의 일관성을 보존한다.

**전체 생성기 목적 함수**:

$$\mathcal{L}^G = \mathcal{L}_{adv}^G + \lambda \mathcal{L}_{rec} $$

여기서 $\lambda$는 가중 하이퍼파라미터 (기본값 = 1).

최종 이미지 복원: $x'_0 = \text{IWT}(y'_0)$

#### 샘플링 알고리즘

```
y_T ~ N(0, I)
for t = T, ..., 1 do:
    z ~ N(0, I)
    y'_0 = G(y_t, z, t)
    y_{t-1} ~ q(y_{t-1} | y_t, y'_0)
end for
x_0 = IWT(y_0)
return x_0
```

### 2.3 모델 구조: 웨이블릿 임베디드 생성기(Wavelet-Embedded Generator)

UNet 구조를 기반으로 하되, 세 가지 핵심 컴포넌트를 도입:

#### ① 주파수 인식 다운샘플링/업샘플링 블록 (Frequency-Aware Down/Upsampling Blocks)
- 전통적 블러링 커널 대신 **DWT/IWT**를 활용
- 다운샘플링 시 고주파 서브밴드를 별도 추출하여 업샘플링 블록에 추가 입력으로 전달
- 고주파 정보에 대한 인식 강화

#### ② 주파수 병목 블록 (Frequency Bottleneck Block)
- 중간 단계에서 피처맵 $F_i$를 저주파 $F_{i,ll}$과 고주파 $F_{i,H}$로 분리
- **저주파만 ResNet 블록**으로 심층 처리, 고주파는 원본 그대로 보존
- IWT로 다시 합성 → 저수준 표현 학습에 집중하면서 디테일 유지

#### ③ 주파수 잔차 연결 (Frequency Residual Connection)
- 원본 신호 $Y$를 웨이블릿 다운샘플 레이어로 분해 후 인코더의 피처 피라미드에 추가
- 피처 임베딩의 주파수 원천 인식 강화

### 2.4 성능 향상

#### 정량적 결과

| 데이터셋 | 모델 | FID↓ | Recall↑ | 추론 시간↓ |
|---------|------|------|---------|-----------|
| **CIFAR-10 (32²)** | WaveDiff | 4.01 | 0.55 | **0.08s** |
| | DDGAN | 3.75 | 0.57 | 0.21s |
| | StyleGAN2 w/ ADA | 2.92 | 0.49 | 0.04s |
| **CelebA-HQ (256²)** | WaveDiff + W-Gen | **5.94** | **0.37** | **0.79s** |
| | DDGAN | 7.64 | 0.36 | 1.73s |
| **CelebA-HQ (512²)** | WaveDiff + W-Gen | **6.40** | **0.35** | **0.59s** |
| | DDGAN | 8.43 | 0.33 | 1.49s |
| **LSUN-Church (256²)** | WaveDiff + W-Gen | **5.06** | **0.40** | **1.54s** |
| | DDGAN | 5.25 | - | 3.42s |

#### 연산 효율성 (Table 1)

| 데이터셋 | DDGAN FLOPs | WaveDiff FLOPs | DDGAN MEM | WaveDiff MEM |
|---------|-------------|----------------|-----------|--------------|
| CIFAR-10 | 7.05G | **1.67G** | 0.31G | **0.16G** |
| CelebA-HQ (256) | 70.82G | **28.54G** | 3.21G | **1.07G** |
| CelebA-HQ (512) | 282.00G | **74.35G** | 12.30G | **3.22G** |

FLOPs **2.5~4.2배 감소**, 메모리 사용량 **2~3.8배 감소**.

#### 단일 이미지 생성 시간

1024×1024 해상도 이미지를 **약 0.12초**에 생성 — 확산 모델 최초의 근실시간 성능.

#### Ablation Study (CelebA-HQ 256)

| 설정 | FID↓ |
|------|------|
| w/o residual | 6.25 |
| w/o up & down | 6.23 |
| w/o bottleneck | 6.18 |
| **full model** | **5.94** |

재구성 항 추가 시 FID **약 0.6 포인트 감소** (6.55 → 5.94).

### 2.5 한계

1. **FID 점수에서의 약간의 품질 저하**: CIFAR-10에서 DDGAN(3.75) 대비 WaveDiff(4.01)는 약간 낮은 FID를 보임 — 속도-품질 트레이드오프 존재
2. **저해상도에서의 제한**: 32×32 같은 극소 해상도에서는 웨이블릿 임베디드 생성기 사용 불가 (공간 차원이 너무 작음)
3. **배치 크기 민감성**: 배치 크기에 따라 성능 차이 유의미 (배치 64 vs 128에서 FID 최대 0.6 차이)
4. **DDGAN 프레임워크에 대한 의존**: 독립적 프레임워크가 아닌 DDGAN 위에 구축 — 기반 모델의 한계를 상속
5. **다양한 도메인에 대한 검증 부족**: 얼굴, 교회, CIFAR-10 등 제한된 데이터셋에서만 실험 수행

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 주파수 분해를 통한 일반화 메커니즘

논문의 Section 5.5에서 저자들은 WaveDiff가 빠르고 안정적으로 수렴하는 이유를 설명한다:

> "Instead of learning from an entanglement of coarse and detailed information, our method separates them for efficient training and at multi scales in the feature space."

이는 일반화 성능 향상과 직결되는 두 가지 메커니즘을 내포한다:

1. **저주파 서브밴드의 효율적 학습**: 공간 차원이 축소되어 전역적 구조(global structure)를 더 쉽게 학습
2. **고주파 성분의 희소성 활용**: 고주파 성분은 희소(sparse)하고 반복적이므로 빠르게 학습 가능 → 과적합 위험 감소

### 3.2 수렴 안정성과 일반화

STL-10에서의 실험(Figure 8a)은 WaveDiff가 DDGAN보다 **훨씬 빠르고 안정적으로 수렴**함을 보여준다:
- DDGAN은 초기 400 에폭 동안 객체의 전반적 형태조차 복원하지 못함
- WaveDiff는 초기부터 의미 있는 구조를 생성

이러한 빠른 수렴은 **더 적은 학습 데이터로도 안정적 성능**을 달성할 수 있는 가능성을 시사한다.

### 3.3 다중 해상도에서의 스케일러빌리티

| 해상도 | 추론 시간 (단일 이미지) |
|--------|----------------------|
| 32×32 | 0.07s |
| 64×64 | 0.12s |
| 256×256 | 0.08s |
| 512×512 | 0.1s |
| 1024×1024 | 0.12s |

해상도가 **4배 증가해도** 추론 시간이 **비례적으로 증가하지 않음** — 웨이블릿 변환의 차원 축소 효과가 고해상도에서 더 큰 이득을 제공.

### 3.4 일반화 성능 향상을 위한 잠재적 방향

1. **다른 웨이블릿 변환 적용**: Haar 외에 Daubechies, Symlet 등 다양한 웨이블릿 기저 탐색
2. **적응적 웨이블릿 분해 단계**: 단일 레벨이 아닌 다중 레벨 분해로 더 풍부한 스케일 정보 활용
3. **텍스트-이미지 생성 등 다양한 조건부 생성 과제로의 확장**: Latent Diffusion Models(LDM)과의 결합
4. **데이터 증강과의 시너지**: 주파수 도메인에서의 증강 기법 결합

### 3.5 일반화 관련 잠재적 우려

- **주파수 분해의 도메인 특이성**: 자연 이미지(얼굴, 건축물)에서 검증되었으나, 의료 영상, 위성 영상 등 다른 도메인에서의 성능은 미검증
- **Progressive upsampling 실패 사례**: 저주파·고주파 간 조건부 분포 $p(x_{hi}|x_{lo})$의 불일치 문제 — 다양한 데이터 분포에서 일반화 어려울 수 있음

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 연구 영향

1. **확산 모델의 실시간 응용 가능성 개척**: 최초로 확산 모델에서 1024×1024 해상도 이미지를 ~0.1초에 생성할 수 있음을 입증
2. **주파수 도메인 학습의 패러다임 전환**: 픽셀 공간이 아닌 웨이블릿 공간에서의 확산 과정이 효과적임을 증명 — 이후 많은 후속 연구에 영감
3. **속도-품질 트레이드오프의 새로운 최적점**: GAN 수준의 속도와 확산 모델 수준의 품질 사이의 간극을 크게 좁힘
4. **효율적 네트워크 설계 원칙 제시**: 주파수 인식 다운/업샘플링, 주파수 병목 등의 설계 패턴이 다른 생성 모델에도 적용 가능

### 4.2 향후 연구 시 고려 사항

1. **Latent Diffusion과의 통합**: Rombach et al. (2022)의 LDM처럼 잠재 공간에서의 확산 과정과 웨이블릿 변환을 결합할 가능성 탐색
2. **Classifier-Free Guidance와의 호환성**: 조건부 생성 품질 향상을 위한 가이던스 기법과의 결합
3. **더 다양한 데이터셋 및 태스크**: 텍스트-이미지, 이미지 편집, 인페인팅 등 다양한 다운스트림 태스크에서의 검증
4. **웨이블릿 변환의 학습 가능화**: 고정된 Haar 필터 대신 학습 가능한 웨이블릿 필터 도입
5. **메모리 효율성 추가 개선**: 고해상도(2K, 4K)에서의 실용성을 위한 추가 최적화

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 접근 | 속도 | 품질 (FID) | WaveDiff와의 관계 |
|------|------|----------|------|----------|-----------------|
| **DDPM** (Ho et al.) | 2020 | 1000-step 디노이징 | 매우 느림 (~80s) | CIFAR-10: 3.21 | WaveDiff의 기반 확산 프레임워크 |
| **DDIM** (Song et al.) | 2021 | 비마르코프 체인, 50 스텝 | 느림 (~4s) | CIFAR-10: 4.67 | 스텝 축소 접근의 선행 연구 |
| **Score SDE** (Song et al.) | 2021 | SDE 기반 score matching | 매우 느림 (~423s) | CIFAR-10: 2.20 | 최고 품질 달성하나 속도 문제 |
| **Diffusion Models Beat GANs** (Dhariwal & Nichol) | 2021 | Classifier guidance | 느림 | ImageNet: SOTA | 품질 측면의 벤치마크 |
| **DDGAN** (Xiao et al.) | 2022 | GAN+Diffusion 결합, 4 스텝 | 빠름 (~0.21s) | CIFAR-10: 3.75 | **WaveDiff의 직접적 기반 모델** |
| **Latent Diffusion (LDM)** (Rombach et al.) | 2022 | 잠재 공간에서 확산 | 보통 | 다양한 SOTA | 잠재 공간 접근 vs. 주파수 공간 접근 |
| **Stable Diffusion** | 2022 | LDM + CLIP | 보통 | 고품질 | 텍스트-이미지 응용의 대표 |
| **DALL-E 2** (Ramesh et al.) | 2022 | CLIP latent + Diffusion | 보통 | 매우 높은 품질 | 대규모 조건부 생성의 벤치마크 |
| **Imagen** (Saharia et al.) | 2022 | T5 + cascaded diffusion | 보통 | 포토리얼리스틱 | 텍스트 이해 기반 생성 |
| **WaveGAN** (Yang et al.) | 2022 | 주파수 인식 GAN | 빠름 | few-shot에서 우수 | 웨이블릿+GAN 결합의 선행 연구 |
| **FreGAN** (Yang et al.) | 2022 | 주파수 성분 활용 GAN | 빠름 | limited data에서 우수 | 주파수 도메인 GAN의 관련 연구 |
| **Consistency Models** (Song et al.) | 2023 | 일관성 모델, 1-step 가능 | 매우 빠름 | CIFAR-10: ~3.5 | WaveDiff 이후의 속도 향상 접근 |
| **SDXL / SD3** (Stability AI) | 2023-2024 | 확장된 LDM, DiT 결합 | 보통-빠름 | 매우 높은 품질 | 스케일업 방향의 발전 |
| **DiT** (Peebles & Xie) | 2023 | Diffusion Transformer | 보통 | ImageNet SOTA | 아키텍처 혁신 (Transformer 기반) |

### 핵심 비교 분석

**vs. Latent Diffusion Models (LDM)**:
- LDM은 **사전 학습된 오토인코더의 잠재 공간**에서 확산을 수행하여 차원을 축소
- WaveDiff는 **웨이블릿 변환이라는 수학적으로 가역적인 변환**으로 차원을 축소
- LDM은 인코더/디코더 학습이 필요하지만, WaveDiff의 DWT/IWT는 **학습이 필요 없는 고정 연산**
- 두 접근은 상보적: LDM의 잠재 공간에서 웨이블릿 확산을 수행하는 **하이브리드 접근**이 가능

**vs. Consistency Models**:
- Consistency Models은 **1-step 생성**까지 가능하여 속도에서는 WaveDiff보다 우수할 수 있음
- 그러나 WaveDiff의 **주파수 인식 아키텍처**는 Consistency Models에도 적용 가능한 직교적 기법

**vs. DiT (Diffusion Transformer)**:
- DiT는 UNet을 Transformer로 대체하여 스케일링 우수성 입증
- WaveDiff의 웨이블릿 기법은 **아키텍처에 독립적**이므로 Transformer 기반 확산 모델에도 적용 가능

---

## 참고자료

1. Phung, H., Dao, Q., & Tran, A. (2023). "Wavelet Diffusion Models are fast and scalable Image Generators." *arXiv:2211.16152v2*. (본 논문)
2. Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS 2020*.
3. Xiao, Z., Kreis, K., & Vahdat, A. (2022). "Tackling the Generative Learning Trilemma with Denoising Diffusion GANs." *ICLR 2022*.
4. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." *CVPR 2022*.
5. Song, J., Meng, C., & Ermon, S. (2021). "Denoising Diffusion Implicit Models." *ICLR 2021*.
6. Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S., & Poole, B. (2021). "Score-Based Generative Modeling through Stochastic Differential Equations." *ICLR 2021*.
7. Dhariwal, P. & Nichol, A. (2021). "Diffusion Models Beat GANs on Image Synthesis." *NeurIPS 2021*.
8. Saharia, C. et al. (2022). "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding." *arXiv:2205.11487*.
9. Ramesh, A. et al. (2022). "Hierarchical Text-Conditional Image Generation with CLIP Latents." *arXiv:2204.06125*.
10. Song, Y. et al. (2023). "Consistency Models." *ICML 2023*.
11. Peebles, W. & Xie, S. (2023). "Scalable Diffusion Models with Transformers." *ICCV 2023*.
12. Karras, T. et al. (2020). "Training Generative Adversarial Networks with Limited Data." *NeurIPS 2020*.
13. Yang, M. et al. (2022). "WaveGAN: An frequency-aware GAN for high-fidelity few-shot image generation." *ECCV 2022*.
14. Gal, R. et al. (2021). "SWAGAN: A Style-Based Wavelet-Driven Generative Model." *ACM TOG*.
15. GitHub 저장소: https://github.com/VinAIResearch/WaveDiff.git
