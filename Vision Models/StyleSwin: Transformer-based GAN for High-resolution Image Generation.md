# StyleSwin: Transformer-based GAN for High-resolution Image Generation

---

## 1. 핵심 주장 및 주요 기여 요약

**핵심 주장:** 순수(pure) 트랜스포머 아키텍처만으로도 ConvNet 기반 GAN(예: StyleGAN2)에 필적하거나 이를 능가하는 고해상도($1024 \times 1024$) 이미지 생성이 가능하다.

**주요 기여:**

1. **Swin Transformer 기반 Style 생성기:** 윈도우 기반 로컬 어텐션(local attention)을 style-based 아키텍처에 통합하여, 계산 효율성과 모델링 능력 사이의 균형을 달성
2. **Double Attention 메커니즘:** 어텐션 헤드를 분할하여 regular window와 shifted window에 동시에 어텐드함으로써, 수용 영역(receptive field)을 효율적으로 확장
3. **Local-Global Positional Encoding:** 윈도우 기반 트랜스포머에서 누락되는 절대 위치 정보를 사인파 위치 인코딩(SPE)으로 보완
4. **Wavelet Discriminator를 통한 블로킹 아티팩트 억제:** 고해상도 합성 시 로컬 어텐션의 블록 단위 처리로 인한 주기적 아티팩트를 주파수 도메인에서 효과적으로 제거
5. **CelebA-HQ 1024에서 FID 4.43으로 기존 최고 성능(StyleGAN 포함)을 초과**, FFHQ-1024에서 FID 5.07로 StyleGAN2에 근접

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

트랜스포머는 판별적(discriminative) 비전 과제에서 큰 성공을 거두었으나, **고해상도 이미지 생성**(generative modeling)에서는 ConvNet에 비해 뒤처져 있었다. 주요 장벽은 다음과 같다:

- **이차 계산 복잡도:** 전체 self-attention의 $O(n^2)$ 복잡도로 인해 고해상도($1024 \times 1024$) 피처맵에 대한 직접 적용이 불가능
- **제한된 수용 영역:** 로컬 어텐션 사용 시 장거리 의존성(long-range dependency) 모델링이 어려움
- **절대 위치 정보의 부재:** 윈도우 기반 트랜스포머에서는 ConvNet의 zero padding이 제공하는 암묵적 절대 위치 정보가 누락
- **블로킹 아티팩트:** 블록 단위 로컬 어텐션이 공간적 연속성(spatial coherency)을 깨뜨려 고해상도에서 주기적 아티팩트 발생

### 2.2 제안하는 방법 (수식 포함)

#### (1) 기본 아키텍처: Swin Transformer 블록

입력 피처맵 $\boldsymbol{x}^l \in \mathbb{R}^{H \times W \times C}$에 대해, 연속적인 Swin 블록은 다음과 같이 동작한다:

$$
\hat{\boldsymbol{x}}^l = \text{W-MSA}(\text{LN}(\boldsymbol{x}^l)) + \boldsymbol{x}^l, \quad \boldsymbol{x}^{l+1} = \text{MLP}(\text{LN}(\hat{\boldsymbol{x}}^l)) + \hat{\boldsymbol{x}}^l \quad \text{(regular window)}
$$

$$
\hat{\boldsymbol{x}}^{l+1} = \text{SW-MSA}(\text{LN}(\boldsymbol{x}^{l+1})) + \boldsymbol{x}^{l+1}, \quad \boldsymbol{x}^{l+2} = \text{MLP}(\text{LN}(\hat{\boldsymbol{x}}^{l+1})) + \hat{\boldsymbol{x}}^{l+1} \quad \text{(shifted window)}
$$

여기서 W-MSA, SW-MSA는 각각 regular 및 shifted 윈도우 분할 하의 윈도우 기반 다중 헤드 self-attention이고, LN은 Layer Normalization이다.

#### (2) Style Injection (AdaIN)

잠재 코드 $z$를 비선형 매핑 $f: \mathcal{Z} \to \mathcal{W}$를 통해 $\mathcal{W}$ 공간으로 변환한 후, Adaptive Instance Normalization(AdaIN)을 통해 트랜스포머 블록의 피처맵을 변조한다. 다양한 스타일 주입 방법(AdaLN, AdaBN, Modulated MLP, Cross-attention 등)을 비교한 결과, **AdaIN이 최적의 FID(6.34)**를 달성하였다.

#### (3) Double Attention

$h$개의 어텐션 헤드를 두 그룹으로 분할하여, 전반부는 regular window에, 후반부는 shifted window에 동시에 어텐드한다:

$$
\text{Double-Attention} = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\boldsymbol{W}^O
$$

여기서 $\boldsymbol{W}^O \in \mathbb{R}^{C \times C}$는 출력 프로젝션 행렬이고, 각 헤드는 다음과 같이 계산된다:

$$
\text{head}_i = \begin{cases} \text{Attn}(\boldsymbol{x}_w \boldsymbol{W}_i^Q, \boldsymbol{x}_w \boldsymbol{W}_i^K, \boldsymbol{x}_w \boldsymbol{W}_i^V) & i \leq \lfloor \frac{h}{2} \rfloor \\ \text{Attn}(\boldsymbol{x}_{sw} \boldsymbol{W}_i^Q, \boldsymbol{x}_{sw} \boldsymbol{W}_i^K, \boldsymbol{x}_{sw} \boldsymbol{W}_i^V) & i > \lfloor \frac{h}{2} \rfloor \end{cases}
$$

여기서 $\boldsymbol{x}_w, \boldsymbol{x}\_{sw} \in \mathbb{R}^{\frac{HW}{\kappa^2} \times \kappa \times \kappa \times C}$이고, $\boldsymbol{W}_i^Q, \boldsymbol{W}_i^K, \boldsymbol{W}_i^V \in \mathbb{R}^{C \times (C/h)}$이다.

윈도우 크기 $\kappa$에 대해 하나의 double attention 블록은 각 차원에서 수용 영역을 **$2.5\kappa$** 만큼 확장한다(기존 Swin의 $\kappa$ 대비).

#### (4) Local-Global Positional Encoding (SPE)

스케일 업샘플링 후, 피처맵에 다음과 같은 사인파 위치 인코딩을 더한다:

$$
\left[\underbrace{\sin(\omega_0 i), \cos(\omega_0 i), \cdots}_{\text{horizontal dimension}}, \underbrace{\sin(\omega_0 j), \cos(\omega_0 j), \cdots}_{\text{vertical dimension}}\right] \in \mathbb{R}^C
$$

여기서 $\omega_k = 1/10000^{2k}$이고 $(i, j)$는 2D 공간 좌표이다. RPE(상대 위치 인코딩)는 로컬 윈도우 내 상대 위치를, SPE는 전역 절대 위치를 각각 제공하여 상호보완적으로 작동한다.

#### (5) 학습 손실함수

표준 non-saturating logistic GAN 손실에 $R_1$ gradient penalty를 적용한다:

$$
\mathcal{L}_D = -\mathbb{E}_{x \sim P_x}[\log(D(x))] - \mathbb{E}_{z \sim P_z}[\log(1 - D(G(z)))] + \gamma \cdot \mathbb{E}_{x \sim P_x}[\|\nabla_x D(x)\|_2^2]
$$

$$
\mathcal{L}_G = -\mathbb{E}_{z \sim P_z}[\log(D(G(z)))]
$$

#### (6) Wavelet Discriminator

고해상도에서의 블로킹 아티팩트를 억제하기 위해, 입력 이미지를 계층적으로 다운샘플링하면서 각 스케일에서 이산 웨이블릿 변환(DWT) 후 주파수 불일치를 판별하는 wavelet discriminator를 공간 도메인 discriminator에 보완적으로 사용한다.

### 2.3 모델 구조

| 구성 요소 | 설명 |
|---------|------|
| **생성기** | $4 \times 4 \times 512$ 상수 입력에서 시작, 각 해상도 스케일마다 2개의 double attention 블록 + MLP + bilinear upsampling을 계층적으로 적용 |
| **Style 매핑 네트워크** | $f: \mathcal{Z} \to \mathcal{W}$ (8층 FC) |
| **Style 주입** | AdaIN을 attention 블록과 MLP 사이에 적용 |
| **위치 인코딩** | RPE (블록 내부) + SPE (각 스케일 업샘플링 후) |
| **판별기** | Conv 기반 공간 판별기 (StyleGAN에서 차용) + Wavelet 판별기 |
| **업샘플링** | Bilinear upsampling + anti-aliasing filter |

- StyleSwin-256: $4 \times 4$부터 $256 \times 256$까지 7개 스케일
- StyleSwin-1024: $4 \times 4$부터 $1024 \times 1024$까지 9개 스케일

### 2.4 성능 향상

**256 × 256 해상도 (FID ↓):**

| 데이터셋 | StyleGAN2 | HiT-B | StyleSwin |
|--------|-----------|-------|-----------|
| FFHQ | 3.62 | 2.95 | **2.81** |
| CelebA-HQ | - | 3.39 | **3.25** |
| LSUN Church | 3.86 | - | **2.95** |

**1024 × 1024 해상도 (FID ↓):**

| 데이터셋 | StyleGAN/2 | HiT-B | StyleSwin |
|--------|-----------|-------|-----------|
| FFHQ | **4.41** | 6.37 | 5.07 |
| CelebA-HQ | 5.06 | 8.83 | **4.43** |

**Ablation Study (FFHQ-256):**

| 구성 | FID |
|-----|-----|
| Swin baseline | 15.03 |
| + Style injection | 8.40 |
| + Double attention | 7.86 |
| + Wavelet discriminator | 6.34 |
| + SPE | 5.76 |
| + Larger model | 5.50 |
| + bCR | **2.81** |

**계산 효율성 (1024 × 1024):**

| 모델 | 파라미터 | FLOPs |
|------|---------|-------|
| StyleGAN2 | 30.37M | 74.27B |
| StyleSwin | 40.86M | **50.90B** |

### 2.5 한계

1. **실제 처리량(throughput)의 격차:** 이론적 FLOPs는 StyleGAN2보다 낮지만, 실제 추론 속도는 StyleGAN2(40.05 imgs/sec)에 비해 StyleSwin(11.05 imgs/sec)이 약 3.6배 느림 — 트랜스포머가 CuDNN 등으로 충분히 최적화되지 않았기 때문
2. **bCR(balanced consistency regularization)이 1024 해상도에서는 효과가 없음** — 고해상도에 적합한 정규화 전략이 추가 연구 필요
3. **판별기는 여전히 Conv 기반:** 생성기만 트랜스포머 기반이며, 완전한 트랜스포머 GAN 파이프라인은 아님
4. **블로킹 아티팩트 해결에 외부 wavelet discriminator 의존:** 생성기 자체적으로 해결하지 못하고 판별기 측 보완이 필요
5. **데이터 효율성:** 트랜스포머는 data-hungry하여 데이터 증강(bCR, DiffAug 등)이 필수적

---

## 3. 모델의 일반화 성능 향상 가능성

StyleSwin에서 일반화 성능과 관련된 핵심 요소들을 중점적으로 분석하면 다음과 같다:

### 3.1 트랜스포머의 표현력과 일반화

- **로컬 어텐션의 locality inductive bias:** CNN과 유사한 국소성 편향을 도입하여, 트랜스포머가 이미지의 규칙성을 처음부터 학습할 필요 없이 효율적으로 학습 가능 → 학습 데이터가 제한적인 상황에서도 일반화 성능 향상에 기여
- **Double Attention을 통한 수용 영역 확장:** 하나의 블록에서 $2.5\kappa$ 만큼 수용 영역이 증가하므로, 전역 구조(geometry)와 국소 세부(fine structure) 모두를 효율적으로 모델링 → 다양한 도메인(얼굴, 교회, 자동차)에 걸친 일반화 가능성 입증

### 3.2 다중 도메인에서의 일반화 실증

- **FFHQ(얼굴), CelebA-HQ(유명인 얼굴), LSUN Church(건축물), LSUN Car(자동차)** 등 서로 다른 도메인에서 모두 경쟁력 있는 FID를 달성
- LSUN Church에서 FID 2.95, LSUN Car에서 FID 4.35 등 복잡한 장면과 재질(material)에 대해서도 높은 품질의 합성 시연

### 3.3 데이터 증강과 일반화

- bCR이 256 해상도에서 FID를 2.69 개선 (5.50 → 2.81), 이는 **트랜스포머 기반 GAN이 데이터 증강에 크게 의존**함을 시사
- 그러나 **1024 해상도에서는 bCR이 효과가 없음** → 고해상도에서의 일반화를 위한 새로운 정규화/증강 전략이 필요

### 3.4 위치 인코딩과 일반화

- **SPE(사인파 위치 인코딩)은 학습 가능한 절대 위치 인코딩 대비 translation invariance를 보장** → 다양한 위치에서의 구조 합성에 일반화 가능
- RPE와 SPE의 결합은 로컬-글로벌 위치 정보를 모두 활용 → 새로운 해상도나 도메인으로의 전이 가능성 시사

### 3.5 일반화 성능 향상을 위한 미래 방향

1. **고해상도에서의 효과적인 정규화 기법 개발** (bCR의 한계 극복)
2. **더 다양한 데이터셋에서의 검증** (ImageNet 등 대규모 다중 클래스 데이터)
3. **트랜스포머 판별기의 도입**으로 완전한 트랜스포머 GAN 파이프라인 구축
4. **사전학습(pre-training) 전략** 활용으로 데이터 효율적 일반화
5. **스케일 변환 가능한(resolution-agnostic) 아키텍처** 설계

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

1. **트랜스포머의 생성 모델 적용 가능성 실증:** ConvNet의 독점적 영역이던 고해상도 이미지 생성에서 트랜스포머의 경쟁력을 처음으로 1024 해상도에서 입증하여, 후속 연구의 기반을 마련
2. **로컬 어텐션 기반 생성 아키텍처의 설계 원칙 제시:** double attention, local-global positional encoding 등 실용적 설계 원칙을 제공
3. **블로킹 아티팩트 문제의 인식과 해결:** 윈도우 기반 로컬 어텐션의 생성 과제 특유 문제를 최초로 체계적으로 분석하고, wavelet discriminator라는 실효적 해법을 제시
4. **GAN에서의 스타일 주입 방법론 확장:** 트랜스포머 GAN에 적합한 스타일 주입 방식(AdaIN)을 체계적으로 비교·선정

### 4.2 앞으로 연구 시 고려할 점

1. **추론 속도 최적화:** 트랜스포머의 하드웨어 최적화(FlashAttention, 커널 퓨전 등)를 생성 모델에 적용하여 실용적 처리량 확보
2. **Diffusion Model과의 결합:** 2022년 이후 diffusion 모델이 GAN 대비 우수한 생성 품질을 보이고 있으므로, Swin 기반 아키텍처를 diffusion 프레임워크에 적용하는 연구 가능
3. **조건부 생성(Conditional Generation)으로의 확장:** 텍스트-이미지 생성 등으로의 확장 가능성
4. **학습 안정성:** 트랜스포머 GAN의 학습 불안정성에 대한 추가 연구 필요
5. **사회적·윤리적 고려:** 고해상도 얼굴 합성의 악용 가능성에 대한 대책 (워터마크, 탐지 기술 등)

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 특징 | FFHQ-256 FID | FFHQ-1024 FID | 비교 포인트 |
|------|------|----------|-------------|--------------|-----------|
| **StyleGAN2** (Karras et al.) | 2020 | Conv 기반 style-based, weight demodulation | 3.62 | 4.41 | Conv GAN의 기준점; StyleSwin이 256에서 초과, 1024에서 근접 |
| **TransGAN** (Jiang et al.) | 2021 | 최초의 순수 트랜스포머 GAN | - | - | 256 해상도까지만 지원; CelebA-HQ FID 9.60 |
| **ViTGAN** (Lee et al.) | 2021 | ViT 기반 GAN, Lipschitz 정규화 | - | - | 학습 안정성 개선에 집중; 고해상도 미지원 |
| **HiT** (Zhao et al.) | 2021 | 트랜스포머 GAN, 고해상도 단계에서 MLP 사용 | 2.95 | 6.37 | 1024 지원하지만 고해상도에서 self-attention 미사용 → 세부 품질 저하 |
| **StyleGAN3** (Karras et al.) | 2021 | Alias-free GAN, translation/rotation equivariance | - | - | Conv 기반; 등변성(equivariance)에 초점 |
| **StyleSwin** (본 논문) | 2022 | Swin + Style-based + Double Attention + Wavelet D | **2.81** | 5.07 | 트랜스포머로 1024 해상도에서 ConvNet에 근접하는 최초 사례 |
| **StyleGAN-XL** (Sauer et al.) | 2022 | StyleGAN2 + progressive growing on ImageNet | - | - | 클래스 조건부 대규모 생성; 스케일링에 초점 |
| **DiT** (Peebles & Xie) | 2023 | Diffusion Transformer, ViT 기반 | - | - | GAN이 아닌 diffusion 프레임워크; 트랜스포머 생성기의 가능성을 diffusion 맥락에서 확장 |
| **U-ViT** (Bao et al.) | 2023 | U-Net 구조의 ViT를 diffusion에 적용 | - | - | StyleSwin의 트랜스포머 생성기 설계 원칙이 diffusion으로 이전 가능성 시사 |

### 비교 분석 핵심 시사점

1. **StyleSwin vs HiT:** 가장 직접적 비교 대상. HiT는 고해상도 단계에서 MLP로 퇴보하여 세밀한 구조 합성 능력이 제한되는 반면, StyleSwin은 모든 해상도에서 self-attention을 유지하면서 wavelet discriminator로 아티팩트를 억제 → FFHQ-1024에서 FID 5.07 vs 6.37
2. **StyleSwin vs StyleGAN2:** 256 해상도에서 StyleSwin이 FID에서 우세(2.81 vs 3.62), 1024에서는 StyleGAN2가 소폭 우세(4.41 vs 5.07)하지만, StyleSwin은 path length regularization이나 style-mixing 없이 달성
3. **GAN → Diffusion 전환:** 2022-2023년 이후 DiT, U-ViT 등 트랜스포머 기반 diffusion 모델이 이미지 생성의 주류로 부상. StyleSwin의 local attention, double attention, positional encoding 설계 원칙은 이러한 diffusion 모델에도 적용 가능한 범용적 기여

---

## 참고 자료

1. **Zhang, B., Gu, S., Zhang, B., Bao, J., Chen, D., Wen, F., Wang, Y., & Guo, B.** (2022). "StyleSwin: Transformer-based GAN for High-resolution Image Generation." *arXiv:2112.10762v2* [cs.CV]. — 본 논문 원문
2. **Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., & Aila, T.** (2020). "Analyzing and Improving the Image Quality of StyleGAN." *CVPR 2020*. — StyleGAN2
3. **Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B.** (2021). "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." *ICCV 2021*. — Swin Transformer
4. **Zhao, L., Zhang, Z., Chen, T., Metaxas, D., & Zhang, H.** (2021). "Improved Transformer for High-Resolution GANs." *arXiv:2106.07631*. — HiT
5. **Jiang, Y., Chang, S., & Wang, Z.** (2021). "TransGAN: Two Transformers Can Make One Strong GAN." *NeurIPS 2021*. — TransGAN
6. **Lee, K., Chang, H., Jiang, L., Zhang, H., Tu, Z., & Liu, C.** (2021). "ViTGAN: Training GANs with Vision Transformers." *arXiv:2107.04589*. — ViTGAN
7. **Karras, T., Aittala, M., Laine, S., Härkonen, E., Hellsten, J., Lehtinen, J., & Aila, T.** (2021). "Alias-Free Generative Adversarial Networks." *NeurIPS 2021*. — StyleGAN3
8. **Peebles, W. & Xie, S.** (2023). "Scalable Diffusion Models with Transformers." *ICCV 2023*. — DiT (Diffusion Transformer)
9. **Bao, F., Nie, S., Xue, K., Cao, Y., Li, C., Su, H., & Zhu, J.** (2023). "All are Worth Words: A ViT Backbone for Diffusion Models." *CVPR 2023*. — U-ViT
10. **Sauer, A., Schwarz, K., & Geiger, A.** (2022). "StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets." *SIGGRAPH 2022*. — StyleGAN-XL
11. **Gal, R., Cohen, D., Bermano, A., & Cohen-Or, D.** (2021). "SWAGAN: A Style-based Wavelet-driven Generative Model." *arXiv*. — Wavelet Discriminator
12. **Vaswani, A., et al.** (2017). "Attention Is All You Need." *NeurIPS 2017*. — 트랜스포머 원본
