# ∆ConvFusion : Can We Achieve Efficient Diffusion without Self-Attention? Distilling Self-Attention into Convolutions

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
이 논문의 핵심 주장은 **"확산 모델(Diffusion Model)의 Self-Attention은 이론적으로는 전역(Global) 상호작용을 수행하지만, 실제로는 주로 국소적(Local) 패턴에 집중되어 있으며, 이를 합성곱(Convolution) 연산으로 대체해도 생성 품질을 유지할 수 있다"**는 것입니다.

### 주요 기여

| 기여 | 내용 |
|------|------|
| **분석적 발견** | Self-Attention이 두 가지 핵심 주파수 성분으로 분해됨을 증명 |
| **∆ConvBlock 설계** | Pyramid Convolution + Average Pooling의 이중 구조 제안 |
| **∆ConvFusion 프레임워크** | 지식 증류(Knowledge Distillation)를 통한 효율적 학습 |
| **계산 효율성** | 16K 해상도에서 Self-Attention 대비 **6929×** FLOPs 감소 |
| **범용성** | U-Net(SD1.5, SDXL)과 DiT(PixArt, FLUX) 모두에 적용 가능 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

현대 확산 모델은 Self-Attention의 **이차적(Quadratic) 계산 복잡도** 문제를 가집니다.

$$\text{FLOPs}_{\text{attn}} = 4H'W'C'^2 + 4(H'W')^2C' + 4(H'W')^2$$

- $4H'W'C'^2$: 선형 레이어(Query, Key, Value 투영)에서 발생
- $4(H'W')^2C'$: 공간 상호작용 행렬 곱에서 발생 → **해상도에 이차적으로 증가**
- $4(H'W')^2$: 스케일링 및 Softmax에서 발생

16K 해상도 생성 시 Self-Attention만으로 **11,010조 FLOPs**가 필요합니다.

---

### 2.2 핵심 분석: Self-Attention의 실제 동작 방식

#### (1) 시각적 분석
Figure 2에서 FLUX, PixArt, SD1.5 모두 각 픽셀의 Attention이 인근 이웃에 집중됨을 시각적으로 확인합니다.

Timestep $t$에 걸쳐 집계된 Attention Map:

$$\mathbf{A}^l = \frac{1}{T}\sum_{t=0}^{T} \mathbf{A}^l_t, \quad \mathbf{A}^l_t = \text{Softmax}\left(\frac{\psi_q(\mathbf{z}^l_t)^\top \psi_k(\mathbf{z}^l_t)}{\sqrt{C_k}}\right)$$

#### (2) 주파수 영역 분석 (ASM)

Attention Score Mass(ASM)를 커널 크기 $K$에 대해 정의:

$$\text{ASM}^l(x_i, y_j) = \sum_{(x_m, y_n) \in d_\infty < \frac{K}{2}} \mathbf{A}^l(x_i, y_j)(x_m, y_n) \tag{1}$$

전체 ASM:

$$\text{ASM} = \sum_l \sum_{i,j=1\ldots H', 1\ldots W'} \text{ASM}^l(x_i, y_j)$$

**발견:** $K$와 ASM 사이에 일관된 **이차적(Quadratic) 관계**가 존재하며, 이는 Attention이 주로 **저주파 정보**를 포착함을 의미합니다.

고주파 성분 분석을 위해 **이산 푸리에 변환(DFT)**과 **Butterworth 고역통과 필터**를 적용하여 고주파 Attention Map $\Lambda^l$을 추출하고, 이에 대한 ASM을 재계산:

```math
\hat{k}^l = \min\left\{k \in \{0, \ldots, K\} \;\bigg|\; \text{ASM}^l_\Lambda \geq 0.8\right\}
```

**발견:** PixArt에서 고주파 Attention 신호의 80% 이상이 **10×10 영역 내**에 집중됩니다.

#### 분석 결과 종합: Self-Attention의 두 핵심 성분

| 성분 | 특성 | 대응 모듈 |
|------|------|---------|
| **고주파, 거리 의존적 신호** | 거리에 따라 이차적으로 감쇠, 강한 국소성 | Pyramid Convolution |
| **저주파, 공간 불변 성분** | 공간적으로 균일한 편향(bias) | Average Pooling |

---

### 2.3 제안하는 방법: ∆ConvFusion

#### (1) Pyramid Convolution Block (∆ConvBlock)

입력 특징 맵 $\mathbf{z}^l_t \in \mathbb{R}^{H' \times W' \times C'}$에 대해:

**Step 1. Layer Normalization:**
$$\tilde{\mathbf{z}}^l_t = \text{LN}(\mathbf{z}^l_t)$$

**Step 2. 채널 차원 압축 (1×1 Conv):**
$$\mathbf{z}^l_{t,\text{in}} = \psi_{\text{in}}(\tilde{\mathbf{z}}^l_t) \in \mathbb{R}^{H' \times W' \times C'/2n}$$

여기서 $n$은 Pyramid Stage 수입니다.

**Step 3. Pyramid Stage 연산:**

$$\Delta^l_\theta(\mathbf{z}^l_{t,\text{in}}) = \sum_{i=1}^{n} \Delta_i(\mathbf{z}^l_{t,\text{in}}) = \sum_{i=1}^{n} \left(\uparrow 2^i \left(\rho\left(\downarrow 2^i(\mathbf{z}^l_{t,\text{in}})\right)\right)\right) \tag{3}$$

- $\downarrow 2^i$: 스케일 $2^i$의 Average Pooling (다운샘플링)
- $\uparrow 2^i$: 스케일 $2^i$의 Bilinear Interpolation (업샘플링)
- $\rho(\cdot)$: **Scaled Simple Gate** (수치 안정성 강화)

$$\rho(\mathbf{f}) = \frac{\mathbf{f}_{<C'/2} \cdot \mathbf{f}_{\geq C'/2}}{\sqrt{C'}}$$

> **설계 직관:** 중앙에 가까운 픽셀일수록 더 많은 Pyramid Stage를 통과하여 높은 가중치를 획득 → 고주파 신호의 거리 의존적 감쇠 특성을 모방

#### (2) Average Pooling Branch (저주파 성분)

$$\mathbf{f}^{\text{avg}}_{\text{out}} = \psi_p\left(\frac{\sum_{x=0}^{W}\sum_{y=0}^{H} \tilde{\mathbf{z}}^l_t(x,y)}{HW}\right) \tag{4}$$

선형성에 의해 등가 관계 성립:

$$\frac{1}{HW}\psi_p\left(\sum_{x=0}^{W}\sum_{y=0}^{H} \tilde{\mathbf{z}}^l_t(x,y)\right) = \frac{1}{HW}\sum_{x=0}^{W}\sum_{y=0}^{H}\psi_p(\tilde{\mathbf{z}}^l_t(x,y))$$

---

### 2.4 학습 방법: 지식 증류(Knowledge Distillation)

∆ConvBlock만 학습하고 다른 모든 파라미터는 **동결(Frozen)** 합니다.

#### Feature-Level Loss (특징 수준 정렬):

$$\mathcal{L}_f = \sum_{l=1}^{N} \left\|\Delta^l_\theta(\mathbf{z}^l_t) - \mathbf{z}^l_{t,\text{out}}\right\|^2 \tag{5}$$

$\mathbf{z}^l_{t,\text{out}}$: 원본 Self-Attention 블록의 $l$번째 출력

#### Output-Level Loss (Min-SNR 가중치 적용):

$$\mathcal{L}_z = \min\left(\gamma \cdot \left(\sigma^z_t / \sigma^\epsilon_t\right)^2, 1\right) \cdot \left(\|\tilde{\epsilon} - \hat{\epsilon}\|^2 + \|\epsilon - \hat{\epsilon}\|^2\right) \tag{6}$$

- $\tilde{\epsilon}$: Self-Attention 기반 모델의 출력
- $\hat{\epsilon}$: ∆ConvFusion 모델의 출력
- $\gamma$: Min-SNR 가중치 (논문에서 $\gamma=5$로 설정)

#### 전체 손실 함수:

$$\mathcal{L} = \mathcal{L}_z + \beta \mathcal{L}_f$$

($\beta = 0.001$로 손실 스케일 균형 조정)

---

### 2.5 모델 구조 요약

```
[입력 특징] → Layer Norm
                    ↓
        ┌──────────────────────┐
        │  Pyramid Convolution  │  (고주파 성분: 국소 상호작용)
        │  (∆0, ∆1, ..., ∆n)   │
        └──────────────────────┘
                    +
        ┌──────────────────────┐
        │   Average Pooling    │  (저주파 성분: 전역 편향)
        │   + 1×1 Conv         │
        └──────────────────────┘
                    ↓
        [출력 특징] → (Cross-Attn, FFN 등은 동결 유지)
```

---

### 2.6 성능 향상

#### 계산 효율성 (Table 1)

| 해상도 | Self-Attention | ∆ConvFusion (SD1.5) | 감소 비율 |
|--------|---------------|---------------------|---------|
| 512×512 | 49.67G | 2.82G | **~18×** |
| 1024×1024 | 714.08G | 12.56G | **~57×** |
| 4K | 49,149.21G | 105.94G | **~464×** |
| 16K | 11,010,934.11G | 1,589.10G | **~6929×** |

#### 추론 지연 시간 (Table 2, PixArt 기준)

| 해상도 | Self-Attention | LinFusion | ∆ConvFusion (K=13) |
|--------|---------------|-----------|---------------------|
| 1024×1024 | 5.91ms | 5.82ms | **3.53ms (1.67×)** |
| 4K | 1,077.98ms | 88.59ms | **54.84ms (1.62×)** |

#### 생성 품질 (Table 3)

| 모델 | 방법 | DS↑ | FDD↓ | CLIP↑ |
|------|------|-----|------|-------|
| SD1.5 | Self-Attention | 42.74 | 210.86 | 30.44 |
| SD1.5 | LinFusion | 44.23 | 181.78 | 30.47 |
| SD1.5 | **∆ConvFusion** | **44.72** | 200.15 | **30.73** |
| SDXL | Self-Attention | 42.65 | 147.59 | 30.92 |
| SDXL | **∆ConvFusion** | **45.14** | **143.72** | 30.87 |

---

### 2.7 한계

1. **일부 FDD 지표 저하**: PixArt-512에서 FDD가 Self-Attention(173.88) 대비 ∆ConvFusion(181.06)으로 소폭 증가
2. **대형 커널 크기의 역설**: K=25 적용 시 오히려 K=13보다 지연 시간이 증가할 수 있음 (Table 2)
3. **Cross-Attention 미조정**: Self-Attention만 대체하여 텍스트-이미지 상호작용 모듈은 최적화 미적용
4. **학습 데이터 의존성**: Midjourney-v5 합성 데이터 2M장 + LAION 4K 이미지로 학습하여 데이터 분포 편향 가능성
5. **매우 극단적 해상도에서의 검증 부족**: 논문의 실험은 최대 4K 지연 시간까지만 실측값 제공

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 Cross-Resolution Generalization (핵심 결과)

논문에서 가장 주목할만한 일반화 결과는 **학습 해상도를 벗어난 해상도에서의 성능**입니다.

> **∆ConvFusion(SD1.5)는 512×512에서만 학습했음에도, 1024×1024 생성에서 Self-Attention 기반 모델보다 우수한 품질을 달성합니다.**

반면, Self-Attention 기반 모델은 1024×1024에서 분열되고 비일관적인 아티팩트(왜곡된 개, 선박 이미지)를 생성합니다 (Figure 8).

### 3.2 일반화 성능 향상의 메커니즘

#### 이유 1: 합성곱의 위치 불변성(Translation Invariance)
Self-Attention의 절대적 위치 인코딩 의존성과 달리, 합성곱 연산은 **위치 불변(translation-invariant)** 특성을 가집니다. 이를 통해 학습 해상도와 다른 해상도에서도 일관된 특징 추출이 가능합니다.

#### 이유 2: 선형 복잡도와 메모리 효율성
$$\text{FLOPs}_{\Delta\text{Conv}} \propto O(H'W') \quad \text{vs} \quad \text{FLOPs}_{\text{Attn}} \propto O((H'W')^2)$$

선형 복잡도로 인해 고해상도에서도 메모리 오버플로우 없이 안정적 추론이 가능합니다.

#### 이유 3: 귀납적 편향(Inductive Bias)의 적절한 활용
Pyramid Convolution의 계층적 구조는 다양한 스케일에서 특징을 추출하므로, 해상도 변화에 더 강건합니다.

### 3.3 Effective Receptive Field (ERF) 분석을 통한 검증

Figure 9에서 ∆ConvFusion의 ERF 패턴이 Self-Attention 기반 모델과 **밀접하게 일치**함을 보여줍니다. 이는 ∆ConvBlock이 Attention Map의 고주파 및 저주파 특성 모두를 효과적으로 포착함을 시사합니다.

### 3.4 DiT와 U-Net 모두에 적용 가능한 범용성

∆ConvFusion은 SD1.5, SDXL (U-Net 기반)과 PixArt (DiT 기반) 모두에서 경쟁력 있는 성능을 보여, 특정 아키텍처에 종속되지 않는 일반화 능력을 가집니다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 효율적 확산 모델 연구 계보

```
DDPM (2020) → LDM/SD1.5 (2022) → SDXL (2023) → DiT (2023) → PixArt-Σ (2024) → FLUX (2024)
                                                         ↓
                                        효율화 연구: LinFusion, DiTFastAttn, ∆ConvFusion
```

### 4.2 주요 관련 연구 상세 비교

| 논문 | 핵심 아이디어 | 복잡도 | 한계 |
|------|-------------|--------|------|
| **DDPM** (Ho et al., NeurIPS 2020) | 확산 모델의 기반 | $O(n^2)$ | 매우 느린 추론 |
| **LDM/SD1.5** (Rombach et al., CVPR 2022) | 잠재 공간에서 확산 | $O(n^2)$ | 고해상도 한계 |
| **DiT** (Peebles & Xie, ICCV 2023) | Transformer 기반 확산 | $O(n^2)$ | 높은 계산 비용 |
| **PixArt-Σ** (Chen et al., ECCV 2024) | 약-강 학습 전략 DiT | $O(n^2)$ | Self-Attention 의존 |
| **Mamba/SSM 기반** (Gu & Dao, 2023) | 선형 복잡도 SSM | $O(n)$ | 이미지 생성 품질 저하 가능 |
| **LinFusion** (Liu et al., 2024) | Mamba-2 기반 선형 근사 | $O(n)$ | 512~1024 해상도에서 비효율적 |
| **DiTFastAttn** (Yuan et al., NeurIPS 2024) | 윈도우 Attention 압축 | $O(n^{1.5})$ | 글로벌 상호작용 가정 유지 |
| **∆ConvFusion** (Dong et al., 2025) | Self-Attention → Conv 증류 | $O(n)$ | 극단적 해상도 미검증 |

### 4.3 방법론적 차별성 분석

#### vs. LinFusion (Liu et al., 2024)

LinFusion은 Mamba-2 기반 SSM으로 Self-Attention을 근사합니다:
- **∆ConvFusion의 우위**: 512×512에서 LinFusion(17.03G) > ∆ConvFusion(2.82G), 즉 **저해상도에서도 효율적**
- LinFusion은 512×512 해상도에서 오히려 Self-Attention보다 FLOPs가 증가하는 문제가 있음

추론 지연 비교 (1024×1024):
- LinFusion: 5.82ms
- ∆ConvFusion K=13: 3.53ms (**1.65× 빠름**)

#### vs. DiTFastAttn (Yuan et al., NeurIPS 2024)

DiTFastAttn은 선택적 레이어에 윈도우 Attention을 적용합니다:
- 여전히 글로벌 상호작용의 필요성을 전제
- ∆ConvFusion은 Self-Attention 자체를 제거하는 더 급진적 접근

#### vs. Neighborhood Attention (Hassani et al., CVPR 2023)

논문 내 ablation에서 NA(K=13)로도 유사 성능을 보이지만, NA는 여전히 높은 메모리 비효율성 문제를 가집니다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려사항

### 5.1 연구에 미치는 영향

#### (1) 패러다임 전환적 시사점
이 논문은 **"확산 모델에서 Global Attention이 반드시 필요하다"는 가정에 근본적인 의문**을 제기합니다. 이는 다음과 같은 연구 방향을 촉진할 것입니다:

- Self-Attention의 필요성에 대한 층별(layer-wise) 재검토
- 영상 생성 외 다른 생성 태스크(비디오, 3D 등)에서의 Attention 국소성 분석

#### (2) 엣지 디바이스 배포 가능성 확대
6929× FLOPs 감소는 모바일 및 엣지 디바이스에서의 실시간 이미지 생성 연구를 현실적으로 만듭니다.

#### (3) 지식 증류 프레임워크의 확장성
Feature-level + Output-level 이중 증류 전략은 다른 모달리티(비디오 생성, 음성 합성)에도 적용 가능한 범용 프레임워크입니다.

#### (4) 아키텍처 설계 원칙 재정립
Pyramid Convolution의 **"중앙 근접 픽셀 우대"** 설계 원칙은 향후 효율적 신경망 설계의 새로운 기준점이 될 수 있습니다.

### 5.2 앞으로 연구 시 고려할 점

#### (1) 검증 범위 확장 필요
- 현재 논문은 텍스트-이미지 생성에만 집중. **비디오 생성(Sora, CogVideoX 등)** 모델에서의 시간적(temporal) Self-Attention도 국소적인지 검증 필요
- 16K 해상도에서의 실제 추론 지연 실측값 필요 (논문에서는 FLOPs만 제시)

#### (2) 이론적 보장 강화
- ASM 기반 분석이 직관적이지만, **왜 학습된 Self-Attention이 국소적으로 수렴하는지**에 대한 이론적 설명 부족
- 향후 연구에서는 최적 수송(Optimal Transport) 이론이나 정보 이론적 관점에서의 분석이 필요

#### (3) 적응적 커널 크기 연구
현재 고정된 커널 크기($K=9$ for SD1.5, $K=13$ for SDXL/PixArt)를 사용하는데, **레이어별/태스크별 동적 커널 크기 적응** 연구가 필요합니다.

```math
\hat{k}^l = \min\left\{k \in \{0, \ldots, K\} \;\bigg|\; \text{ASM}^l_\Lambda \geq 0.8\right\}
```

이 수식을 학습 과정에서 동적으로 최적화하는 연구가 가능합니다.

#### (4) 하이브리드 아키텍처 탐구
- 전역 의미 이해가 중요한 일부 레이어는 Self-Attention 유지
- 나머지는 ∆ConvBlock으로 대체하는 **레이어별 최적화 전략** 연구

#### (5) 데이터 편향 문제 해결
- 학습 데이터(Midjourney-v5 합성 이미지 2M)의 스타일 편향이 생성 다양성에 미치는 영향 분석 필요
- 더 다양한 도메인(의료 영상, 위성 영상 등)에서의 성능 검증

#### (6) FP16 수치 안정성 추가 검증
Scaled Simple Gate를 도입했지만, 극단적 해상도나 매우 깊은 네트워크에서의 추가 수치 안정성 검증이 필요합니다.

#### (7) 메트릭의 한계 인식
논문이 MS-COCO 대신 LAION 기반 메트릭(DS, FDD)을 사용하는 것은 합리적이지만, **표준 벤치마크와의 비교 결여**가 재현성 검증을 어렵게 합니다. 향후 연구에서는 GenEval, T2I-Bench 등 표준화된 벤치마크 사용을 고려해야 합니다.

---

## 참고 자료

**주 논문:**
- Dong, Z., Zhou, C., Deng, W., Wei, P., Ji, X., & Lin, L. (2025). *Can We Achieve Efficient Diffusion without Self-Attention? Distilling Self-Attention into Convolutions.* arXiv:2504.21292v1 [cs.CV].

**비교 대상 및 관련 논문 (논문 내 참조):**
- Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS 2020.
- Rombach, R., et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR 2022.
- Peebles, W., & Xie, S. (2023). *Scalable Diffusion Models with Transformers.* ICCV 2023.
- Chen, J., et al. (2024). *PixArt-Σ: Weak-to-strong training of diffusion transformer for 4K text-to-image generation.* ECCV 2024.
- Liu, S., et al. (2024). *LinFusion: 1 GPU, 1 Minute, 16K Image.* arXiv:2409.02097.
- Yuan, Z., et al. (2024). *DiTFastAttn: Attention Compression for Diffusion Transformer Models.* NeurIPS 2024.
- Hassani, A., et al. (2023). *Neighborhood Attention Transformer.* CVPR 2023.
- Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752.
- Hang, T., et al. (2023). *Efficient Diffusion Training via Min-SNR Weighting Strategy.* ICCV 2023.
- Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS 2017.
- Luo, W., et al. (2016). *Understanding the Effective Receptive Field in Deep Convolutional Neural Networks.* NeurIPS 2016.
- Podell, D., et al. (2023). *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis.* arXiv:2307.01952.
- Black Forest Labs. (2024). *FLUX.* https://github.com/black-forest-labs/flux
