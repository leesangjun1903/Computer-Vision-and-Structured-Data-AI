
# DiCo: Revitalizing ConvNets for Scalable and Efficient Diffusion Modeling

> **📚 참고 출처**
> - arXiv:2505.11196 — [https://arxiv.org/abs/2505.11196](https://arxiv.org/abs/2505.11196)
> - NeurIPS 2025 Poster — [https://neurips.cc/virtual/2025/poster/117743](https://neurips.cc/virtual/2025/poster/117743)
> - OpenReview — [https://openreview.net/forum?id=UnslcaZSnb](https://openreview.net/forum?id=UnslcaZSnb)
> - GitHub (공식 코드) — [https://github.com/shallowdream204/DiCo](https://github.com/shallowdream204/DiCo)
> - HuggingFace Papers — [https://huggingface.co/papers/2505.11196](https://huggingface.co/papers/2505.11196)
> - OpenReview PDF — [https://openreview.net/pdf/db6a6892d30093af7353707517ffd1b21d8d0015.pdf](https://openreview.net/pdf/db6a6892d30093af7353707517ffd1b21d8d0015.pdf)
> - 관련 후속 연구: "Reviving ConvNeXt for Efficient Convolutional Diffusion Models" — [https://arxiv.org/html/2603.09408v1](https://arxiv.org/html/2603.09408v1)

---

## 1. 핵심 주장 및 주요 기여 요약

Diffusion Transformer(DiT)는 시각적 생성에서 뛰어난 성능을 보여주지만 상당한 연산 오버헤드를 수반합니다.

흥미롭게도, 사전 학습된 DiT 모델을 분석한 결과, global self-attention은 주로 국소 패턴(local pattern)만을 포착하여 종종 중복적이라는 사실이 밝혀졌으며, 이는 더 효율적인 대안의 가능성을 시사합니다.

이 논문의 **3가지 핵심 기여**는 다음과 같습니다:

| 기여 | 내용 |
|------|------|
| ① 분석적 발견 | DiT의 self-attention이 local pattern에 집중됨을 실증적으로 규명 |
| ② 문제 진단 | ConvNet의 채널 중복성(channel redundancy)이 성능 저하의 원인임을 밝힘 |
| ③ DiCo 제안 | CCA 메커니즘을 통한 완전 합성곱 기반 확산 모델 패밀리 설계 |

DiCo는 NeurIPS 2025 Spotlight 논문으로 채택되었습니다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

#### 🔍 문제 1: DiT의 연산 비효율성

사전 학습된 class-conditional(DiT-XL/2) 및 text-to-image(PixArt-α, FLUX) DiT 모델에서, 특정 anchor token으로 쿼리 시 attention이 주변 공간 토큰에 집중되며 먼 토큰은 거의 무시됩니다. 이는 global attention 계산이 생성 과정에서 중복될 수 있음을 시사하며, local spatial modeling의 중요성을 강조합니다.

인식(recognition) 작업과 달리, 생성(generative) 작업은 세밀한 텍스처와 국소 구조적 충실도(local structural fidelity)를 강조하는 것으로 보입니다.

#### 🔍 문제 2: Naive한 Convolution 대체의 한계

단순히 self-attention을 convolution으로 교체하면 일반적으로 성능이 저하됩니다. 저자들의 조사에 따르면 이 성능 격차는 Transformer에 비해 ConvNet에서 채널 중복성(channel redundancy)이 더 높기 때문입니다.

이 단순 교체는 많은 채널이 생성 과정에서 비활성 상태로 남는 심각한 채널 중복성을 유발합니다. 이는 self-attention이 convolution에 비해 본질적으로 더 강한 표현 능력을 가지기 때문으로 가설화됩니다.

---

### 2-2. 제안 방법 (수식 포함)

#### 📌 핵심 구성요소 1: Conv Module (합성곱 모듈)

DiCo는 먼저 $1\times1$ 합성곱을 적용하여 픽셀 단위의 채널 간 정보를 집계하고, 이후 $3\times3$ depthwise 합성곱으로 채널별 공간적 맥락을 포착합니다. GELU 활성화 함수가 비선형 변환에 사용됩니다.

Conv Module의 연산 흐름은 다음과 같이 표현할 수 있습니다:

```math
\mathbf{x}' = \text{DWConv}_{3\times3}\!\left(\text{Conv}_{1\times1}(\mathbf{x})\right)
```

$$\mathbf{x}'' = \text{GELU}(\mathbf{x}')$$

여기서 $\text{DWConv}\_{3\times3}$는 depthwise convolution, $\text{Conv}_{1\times1}$은 pointwise convolution입니다.

현대의 인식(recognition) ConvNet이 크고 비용이 많이 드는 커널에 의존하는 것과 달리, DiCo는 효율적인 $1\times1$ pointwise convolution과 $3\times3$ depthwise convolution만으로 구성된 간소화된 설계를 채택합니다.

---

#### 📌 핵심 구성요소 2: Compact Channel Attention (CCA)

이를 해결하기 위해, 경량 선형 프로젝션으로 정보성 채널을 동적으로 활성화하는 Compact Channel Attention(CCA) 메커니즘을 도입합니다. 채널 방향의 전역 모델링 접근법인 CCA는 낮은 연산 오버헤드를 유지하면서 모델의 표현 용량과 특징 다양성(feature diversity)을 향상시킵니다.

CCA의 연산은 다음과 같이 수식화할 수 있습니다:

$$\mathbf{s} = \text{GlobalAvgPool}(\mathbf{x}'') \in \mathbb{R}^{C}$$

$$\mathbf{a} = \sigma\!\left(W_2 \cdot \delta\!\left(W_1 \cdot \mathbf{s}\right)\right) \in \mathbb{R}^{C}$$

$$\hat{\mathbf{x}} = \mathbf{a} \odot \mathbf{x}''$$

여기서:
- $\mathbf{s}$: global average pooling을 통해 얻은 채널별 통계
- $W_1, W_2$: 경량 선형 프로젝션 행렬
- $\delta$: GELU 활성화, $\sigma$: Sigmoid 활성화
- $\odot$: channel-wise multiplication

이 구조는 SE-Net(Squeeze-and-Excitation)과 개념적으로 유사하지만, 확산 모델의 채널 중복성 문제를 해소하도록 설계된 경량 버전입니다.

---

#### 📌 핵심 구성요소 3: 확산 프로세스 통합

DiCo는 잠재 공간(latent space)에서 DDPM 프레임워크를 따르며, 노이즈 예측은 다음과 같이 정의됩니다:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, c) \right\|^2 \right]$$

여기서:
- $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\,\boldsymbol{\epsilon}$: $t$-step 노이즈 추가 이미지
- $\boldsymbol{\epsilon}_\theta$: DiCo 기반의 노이즈 예측 네트워크
- $c$: 클래스 레이블 또는 텍스트 조건
- $\bar{\alpha}\_t = \prod_{s=1}^{t}(1-\beta_s)$: 노이즈 스케줄

최종 출력 $z_L$은 정규화된 후 $3\times3$ 합성곱 헤드를 통해 노이즈와 공분산을 모두 예측합니다.

---

### 2-3. 모델 구조

DiCo는 Diffusion Transformer(DiT)에 대한 설득력 있는 대안으로 제안된 새로운 백본입니다. DiCo는 self-attention을 $3\times3$ depthwise convolution과의 조합으로 대체하고, 채널 중복성을 줄이고 특징 다양성을 향상시키기 위한 compact channel attention 메커니즘을 통합합니다.

모델 구조 개요:

```
입력 잠재 벡터 (Latent Vector)
        ↓
  VAE 인코더 (사전 학습된 잠재 공간)
        ↓
  [DiCo Block × N]
  ┌─────────────────────────┐
  │  LayerNorm               │
  │  1×1 Pointwise Conv      │
  │  3×3 Depthwise Conv      │
  │  GELU                    │
  │  CCA (Compact Ch. Attn.) │
  │  AdaLN (시간/조건 주입)   │
  └─────────────────────────┘
        ↓
  3×3 Conv Head (노이즈/공분산 예측)
        ↓
  VAE 디코더 → 생성 이미지
```

모델 스케일 변형(variants):

| 모델 | 파라미터 규모 | 특징 |
|------|-------------|------|
| DiCo-S | 소형 | 빠른 추론 |
| DiCo-B | 중형 | 균형 |
| DiCo-XL | 대형 | 고품질 생성 |
| DiCo-H | ~1B | 최대 성능 |

---

### 2-4. 성능 향상

class-conditional ImageNet 생성 벤치마크에서 DiCo-XL은 $256\times256$ 해상도에서 FID 2.05, $512\times512$에서 FID 2.53을 달성하며, 각각 DiT-XL/2 대비 $2.7\times$ 및 $3.1\times$의 속도 향상을 보입니다.

또한, 가장 큰 모델인 DiCo-H는 1B 파라미터로 스케일 업되어, 추가적인 학습 감독 없이 ImageNet $256\times256$에서 FID 1.90에 도달합니다.

DiG-XL/2(CUDA 최적화 Flash Linear Attention 적용)와 비교하여, DiCo-XL은 $2.9\times$ 빠른 속도와 FID에서 $1.6\times$ 향상을 달성합니다.

DiCo 모델은 Transformer 기반 모델 대비 일관되게 더 적은 GFLOPs를 필요로 하면서도 우수한 생성 성능을 달성합니다.

성능 요약표:

| 모델 | 해상도 | FID↓ | 속도(vs DiT-XL/2) |
|------|--------|------|-------------------|
| DiT-XL/2 | 256×256 | ~2.27 | 1× (기준) |
| DiCo-XL | 256×256 | **2.05** | **2.7×** |
| DiCo-XL | 512×512 | **2.53** | **3.1×** |
| DiCo-H (1B) | 256×256 | **1.90** | - |

---

### 2-5. 한계점

논문 자체에서 명시한 한계(Appendix F)를 바탕으로 정리하면:

1. **Global context 모델링의 부재**: 순수 합성곱 구조로 인해 장거리 의존성(long-range dependency) 처리 능력이 Transformer 대비 구조적으로 제한될 수 있습니다.

2. **고해상도 확장의 이론적 한계**: $3\times3$ depthwise convolution의 receptive field가 고정되어 있어, 매우 고해상도 이미지에서 전역 일관성 유지에 잠재적 한계가 있습니다.

3. **텍스트-이미지 생성의 초기 단계**: MS-COCO에서의 실험 결과는 순수 합성곱 기반 DiCo가 텍스트-이미지 생성에 강한 잠재력을 보임을 보여주지만, 대규모 텍스트-이미지 생성 분야에서의 완전한 검증은 아직 초기 단계입니다.

4. **이론적 보장 부재**: 이 연구는 이론적 유도(theoretical derivation)를 포함하지 않습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 텍스트-이미지 생성으로의 일반화

순수 합성곱 기반 DiCo는 텍스트-이미지 생성에서도 강한 잠재력을 보입니다.

순수 합성곱 기반 DiCo는 텍스트-이미지 생성에서 강한 잠재력을 보이며, 저자들은 DiCo를 더욱 스케일 업하고 광범위한 생성 작업으로 확장하기를 기대합니다.

### 3-2. 스케일러빌리티(Scalability)에 따른 일반화

DiCo-H는 1B 파라미터로 스케일 업되어 추가적인 학습 감독 없이 ImageNet $256\times256$에서 FID 1.90에 도달하는데, 이는 파라미터 스케일에 따른 성능 향상이 안정적으로 이루어짐을 보여줍니다.

### 3-3. 로컬 패턴 모델링의 일반화 근거

인식 작업에서 장거리 상호작용이 전역 의미 추론에 중요한 것과 달리, 생성 작업은 세밀한 텍스처와 국소 구조적 충실도를 강조하는 것으로 보입니다.

이는 ConvNet 기반 구조가 다양한 생성 도메인(이미지 복원, 비디오, 3D 등)으로 일반화될 수 있는 이론적 근거가 됩니다.

### 3-4. CCA의 동적 채널 활성화를 통한 일반화 기여

CCA 메커니즘은 데이터에 따라 동적으로 채널을 활성화하므로, 특정 도메인에 편향되지 않고 다양한 시각적 패턴을 포착할 수 있는 적응적 능력을 제공합니다. 이는 out-of-distribution 일반화에도 긍정적으로 기여합니다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4-1. 확산 모델 아키텍처 발전 흐름

| 연구 | 연도 | 아키텍처 | 핵심 기여 |
|------|------|---------|-----------|
| DDPM (Ho et al.) | 2020 | U-Net | 확산 모델의 기초 확립 |
| ADM (Dhariwal et al.) | 2021 | U-Net + Attention | 분류기 가이던스 |
| DiT (Peebles & Xie) | 2023 | Transformer | ViT 기반 확산 백본 |
| U-ViT | 2023 | ViT + Skip | 모든 입력을 토큰으로 처리 |
| PixArt-α | 2024 | DiT 변형 | 텍스트-이미지 효율화 |
| DiG | 2025 | Gated Linear Attn. | 선형 어텐션으로 효율화 |
| **DiCo** | **2025** | **순수 ConvNet** | **ConvNet 기반 경쟁력 확립** |

### 4-2. DiT vs DiCo 심층 비교

| 비교 항목 | DiT-XL/2 | DiCo-XL |
|-----------|---------|---------|
| 핵심 연산 | Global Self-Attention ( $O(N^2)$ ) | Depthwise Conv + CCA ( $O(N)$ ) |
| FID (256×256) | ~2.27 | **2.05** |
| FID (512×512) | ~3.04 | **2.53** |
| 추론 속도 | 1× (기준) | **2.7~3.1×** |
| 하드웨어 친화성 | 낮음 (VRAM 집약) | **높음** |

### 4-3. DiG(Gated Linear Attention 기반)와의 비교

CUDA 최적화 Flash Linear Attention을 사용하는 DiG-XL/2와 비교해도, DiCo-XL은 $2.9\times$ 빠른 속도와 FID에서 $1.6\times$ 향상을 달성합니다.

### 4-4. 후속 연구: FCDM (Reviving ConvNeXt, 2026)

DiCo와 FCDM 사이에 밀접한 연관성이 관찰되며, FCDM은 DiCo의 설계 선택에 대한 더 효율적인 대안을 제공합니다. 유사한 파라미터 규모에서 FCDM은 DiCo 대비 약 75% FLOPs 효율성을 달성합니다.

DiCo가 convolution module 전반에 걸쳐 채널 차원을 유지하는 반면, FCDM은 ConvNeXt의 inverted bottleneck 구조를 채택하여 블록 내에서 더 풍부한 채널 계산을 위한 초기 채널 확장을 도입합니다.

---

## 5. 향후 연구에 미치는 영향과 고려할 점

### 5-1. 연구에 미치는 영향

**① ConvNet의 재부상 (Renaissance of ConvNets)**
이 연구는 순수 ConvNet을 이용한 확산 기반 이미지 생성의 가능성을 탐색하며, 단순하고 효율적인 ConvNet 설계도 우수한 성능을 발휘할 수 있음을 보입니다. 이는 "Transformer 만이 최선"이라는 편향을 깨는 중요한 전환점입니다.

**② 효율적 확산 모델 연구의 새 방향**

self-attention에 비해 합성곱 연산은 하드웨어에 더 친화적이어서, 대규모 및 자원 제약 환경에서의 배포에 상당한 이점을 제공합니다. 이는 엣지(edge) 환경이나 온디바이스(on-device) 생성 AI 연구에 직접적인 영향을 줍니다.

**③ 채널 중복성 이론의 기여**

채널 중복성이 ConvNet 성능 저하의 핵심 원인임을 실증적으로 밝힌 것은, 이후 ConvNet 기반 아키텍처 설계 연구에서 중요한 기준점이 됩니다.

**④ 멀티태스크 생성으로의 확장**
확산 모델의 다양한 실세계 응용 분야—텍스트-이미지 생성, 이미지 편집, 이미지 복원, 비디오 생성, 3D 콘텐츠 생성—를 고려할 때, DiCo의 효율적 ConvNet 설계는 이 모든 도메인에서 경량화된 대안 모델 개발을 자극할 것입니다.

---

### 5-2. 향후 연구 시 고려할 점

**① 장거리 의존성 보완 전략**
순수 $3\times3$ depthwise convolution은 receptive field가 제한적입니다. 향후 연구에서는 dilated convolution, large kernel convolution, 또는 선택적 글로벌 어텐션 레이어와의 하이브리드 설계를 고려해야 합니다:

$$\text{Effective RF} = k + (k-1)(d-1), \quad d: \text{dilation rate}$$

**② 비디오/3D 생성으로의 확장**
시간 축($T$)이 추가된 비디오 생성의 경우, $3\times3$ spatial conv를 $3\times3\times3$ spatiotemporal conv로 확장하는 연구가 필요합니다.

**③ CCA와 다른 어텐션 메커니즘의 결합**
CCA는 채널 방향 전역 모델링만 수행합니다. 공간 방향의 경량 어텐션(예: Deformable Convolution, CBAM의 Spatial Attention)과의 결합 가능성을 탐구해야 합니다.

**④ 이론적 분석의 부재 보완**
연구의 주장이 이론적·실험적 결과와 일치해야 하며, 결과가 다른 환경에서도 얼마나 일반화될 수 있는지 반영해야 합니다. 이론적 수렴 보장이나 generalization bound 연구가 향후 필요합니다.

**⑤ 스케일 법칙(Scaling Law) 분석**
DiCo-H(1B)의 성공에도 불구하고, ConvNet 기반 확산 모델의 scaling law가 Transformer와 동일한 특성을 보이는지에 대한 체계적 분석이 필요합니다.

**⑥ 하이브리드 아키텍처 탐구**
DiCo와 FCDM 등 후속 연구들은 유사한 고수준 구조를 공유하지만, inverted bottleneck, GRN 등 세부 설계의 차이가 효율성에 큰 영향을 준다는 점에서, 최적 ConvNet 블록 설계의 체계적 탐색이 중요한 연구 과제입니다.

---

## 📌 종합 결론

DiCo는 "Transformer가 지배하는 생성 모델 패러다임"에 ConvNet이 효과적으로 도전할 수 있음을 실증적으로 보인 중요한 연구입니다. 특히 **채널 중복성 문제의 진단과 CCA를 통한 해결**이라는 명확한 인과적 스토리라인, 그리고 **2.7~3.1× 추론 속도 향상과 FID 개선의 동시 달성**은 실용적 관점에서도 매우 의미 있는 성과입니다. 향후 비디오, 3D, 멀티모달 생성 분야에서 ConvNet 기반 확산 모델의 새로운 연구 흐름을 이끌 것으로 기대됩니다.
