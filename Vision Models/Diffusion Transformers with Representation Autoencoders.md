
# Diffusion Transformers with Representation Autoencoders

> **논문 정보**
> - **제목**: Diffusion Transformers with Representation Autoencoders
> - **저자**: Boyang Zheng, Nanye Ma, Shengbang Tong, Saining Xie (New York University)
> - **arXiv**: [2510.11690](https://arxiv.org/abs/2510.11690) (2025년 10월 13일)
> - **프로젝트 페이지**: https://rae-dit.github.io/
> - **GitHub**: https://github.com/bytetriper/RAE

---

## 1. 핵심 주장 및 주요 기여 요약

이 논문은 Diffusion Transformers(DiTs)의 잠재 생성 모델링에서 기존 Variational Autoencoders(VAEs)를 대체하는 새로운 방법으로 **Representation Autoencoders(RAEs)** 를 제안합니다.

핵심 동기는 널리 사용되는 SD-VAE의 한계에서 비롯됩니다. SD-VAE는 구식 합성곱 백본을 사용하고, 정보 용량이 제한된 저차원 잠재 공간을 생성하며, 순수한 재구성 기반 학습으로 인해 약한 표현을 만들어냅니다. 이러한 요인들이 DiTs의 생성 품질과 구조적 단순성을 제약합니다.

### 🔑 주요 기여 요약

| 기여 항목 | 내용 |
|-----------|------|
| **RAE 제안** | 동결된 사전학습 표현 인코더 + 경량 학습 디코더 |
| **DiT 설계 원칙** | 토큰 차원에 맞는 DiT 너비 확장 |
| **차원 의존 노이즈 스케줄** | 고차원 잠재 공간에 맞는 새로운 노이즈 스케줄링 |
| **디코더 노이즈 증강** | 훈련-추론 분포 불일치 해결 |
| **DiT $^{DH}$ 아키텍처** | 계산 비용 없이 너비를 늘리는 새로운 DiT 변형 |
| **성능** | ImageNet FID 1.51 (256×256, 무유도), 1.13 (유도) |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

잠재 생성 모델링에서 사전학습된 오토인코더가 픽셀을 잠재 공간으로 매핑하는 방식이 DiT의 표준 전략이 되었지만, 오토인코더 구성 요소는 거의 발전하지 않았습니다. 대부분의 DiT는 여전히 원래의 VAE 인코더에 의존하며, 이는 구조적 단순성을 저해하는 구식 백본, 정보 용량을 제한하는 저차원 잠재 공간, 순수 재구성 기반 학습으로 인한 약한 표현이라는 여러 한계를 초래합니다.

SD-VAE는 여전히 채널 방향의 강한 압축과 재구성 전용 목적함수에 의존하여, 지역적 외관은 포착하지만 확산 모델의 일반화와 생성 성능에 중요한 전역적 의미 구조가 부족한 낮은 용량의 잠재 표현을 생성합니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### 🔷 (1) Representation Autoencoder (RAE)

이 연구에서는 VAE를 사전학습된 표현 인코더(예: DINO, SigLIP, MAE)와 학습된 디코더로 교체하는 방법을 탐구하여, 이를 **Representation Autoencoders(RAEs)** 라고 명명합니다. RAE는 동결된 사전학습 표현을 인코더로 사용하고, 경량 디코더로 입력 이미지를 압축 없이 재구성합니다.

RAE의 구조를 수식으로 나타내면:

$$\mathbf{z} = f_{\text{frozen}}(\mathbf{x}), \quad \hat{\mathbf{x}} = g_{\theta}(\mathbf{z})$$

여기서:
- $f_{\text{frozen}}$: DINOv2, SigLIP, MAE 등의 **동결된** 사전학습 인코더
- $g_{\theta}$: 학습 가능한 경량 ViT 디코더
- $\mathbf{z} \in \mathbb{R}^{N \times d}$: 고차원 시맨틱 잠재 표현 ($N$: 토큰 수, $d$: 토큰 차원)

디코더는 L1, LPIPS, 적대적 손실(adversarial loss)의 조합으로 학습됩니다:

$$\mathcal{L}_{\text{decoder}} = \lambda_1 \mathcal{L}_{\text{L1}} + \lambda_2 \mathcal{L}_{\text{LPIPS}} + \lambda_3 \mathcal{L}_{\text{GAN}}$$

---

#### 🔷 (2) 차원 의존 노이즈 스케줄 (Dimension-Dependent Noise Schedule Shift)

기존 DiT는 콤팩트한 SD-VAE에 맞게 설계되어, 증가된 차원성으로 인해 어려움을 겪습니다. 특히 해상도 기반의 스케줄 시프트는 픽셀 및 VAE 기반 입력에서 도출된 것으로, 토큰 차원성을 무시합니다.

기존 노이즈 스케줄은 고차원 잠재 공간에 적용 시 준최적(suboptimal)이 됩니다. 이 논문은 유효 데이터 차원 $m = N \times d$ (토큰 수 $\times$ 토큰 차원)에 따라 확산 타임스텝을 재조정하는 **차원 의존 노이즈 스케줄 시프트**를 제안합니다.

Flow Matching 기반의 노이즈 스케줄:

$$t' = t + \alpha \cdot \log\left(\frac{m}{m_{\text{base}}}\right)$$

- $m = N \times d$: 유효 잠재 차원
- $m_{\text{base}}$: 기준 차원(예: $m_{\text{base}} = 4096$)
- $\alpha$: 스케일링 인수
- $t'$: 조정된 타임스텝

---

#### 🔷 (3) 디코더 노이즈 증강 (Decoder Noise Augmentation)

VAE 디코더는 노이즈가 있는 잠재 표현으로부터 이미지를 재구성하도록 학습되어 확산 출력의 작은 노이즈에 더 강인하지만, RAE 디코더는 깨끗한 잠재 표현으로 학습되어 분포를 벗어난 샘플에 어려움을 겪을 수 있습니다.

RAE는 훈련 시의 깨끗한 인코더 잠재 표현과 추론 시의 약간 변형된 잠재 표현 사이의 불일치를 해소하기 위해 **노이즈 증강 디코딩(noise-augmented decoding)** 전략을 제안합니다.

$$\tilde{\mathbf{z}} = \mathbf{z} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 \mathbf{I}), \quad \hat{\mathbf{x}} = g_{\theta}(\tilde{\mathbf{z}})$$

---

### 2.3 모델 구조: DiT $^{DH}$

논문은 DDT에서 영감을 받았지만 다른 설계 관점에 의해 동기 부여된 새로운 DiT 변형인 DiT $^{DH}$를 소개합니다. 이것은 표준 DiT 아키텍처에 경량의 *얕지만 넓은(shallow yet wide)* 헤드를 추가하여, 2차 계산 비용 없이 너비를 늘릴 수 있게 합니다. 경험적으로 이 설계는 고차원 RAE 공간에서의 확산 트랜스포머 학습을 더욱 향상시킵니다.

DiT $^{DH}$ 의 속도 예측(velocity prediction) 수식:

DiT $^{DH}$ 는 표준 DiT 백본 $M$에 경량의 얕고 넓은 트랜스포머 헤드 $H$를 추가하며, 결합 모델은 속도를 다음과 같이 예측합니다: $v_t = H(x_t | z_t, t)$, 여기서 $z_t = M(x_t | t, y)$. 이를 통해 전체 DiT 백본을 확장하는 것과 관련된 2차 계산 비용 없이 유효 모델 너비를 확장할 수 있습니다.

$$v_t = H\bigl(x_t \mid z_t, t\bigr), \quad z_t = M(x_t \mid t, y)$$

여기서:
- $M$: 표준 DiT 백본
- $H$: 얕고 넓은 DDT 헤드
- $y$: 클래스 레이블 조건
- $v_t$: Flow Matching에서의 속도(velocity) 예측

전체 파이프라인 개요:

```
이미지 x
   │
   ▼
[동결된 표현 인코더 f_frozen] → 시맨틱 토큰 z (고차원)
   │
   ├─► [학습된 디코더 g_θ] → 재구성 이미지 x̂ (RAE 학습 단계)
   │
   └─► [DiT^DH 확산 모델] → 생성 단계 (노이즈 스케줄 + flow matching)
              │
              ▼
         생성된 이미지
```

---

### 2.4 성능 향상

경험적으로 RAE는 강력한 시각적 생성 성능을 보입니다. ImageNet에서 RAE 기반 DiT $^{DH}$는 어떤 유도 없이도 256×256에서 FID 1.51을 달성하고, AutoGuidance와 함께 256×256 및 512×512 모두에서 FID 1.13을 달성합니다.

DiT $^{DH}$ -XL은 RAE 잠재 표현으로 VAE 기반(SiT-XL) 및 정렬 기반(REPA-XL) 방법을 능가하는 FID를 달성하며, 각각 **47×** 및 **16×** 의 학습 속도 향상을 보입니다.

SD-VAE는 RAE에 비해 인코더와 디코더에서 각각 약 **6×** 및 **3×** 더 많은 GFLOPs를 요구합니다.

| 모델 | FID (256×256, no guidance) | FID (256×256, w/ guidance) | FID (512×512, w/ guidance) |
|------|:---:|:---:|:---:|
| DiT-XL (VAE 기반) | ~2.27 | - | - |
| SiT-XL (VAE 기반) | ~2.06 | - | - |
| **DiT $^{DH}$ -XL (RAE)** | **1.51** | **1.13** | **1.13** |

---

### 2.5 한계점

RAE는 고차원 잠재 표현에 맞게 DiT 너비를 늘려 유망한 생성 성능을 달성하지만, **재구성 성능은 이전 방법들에 비해 뒤떨어집니다.** 이는 편집(editing) 및 개인화 생성(personalized generation) 같은 태스크에 불리합니다. 또한 RAE는 차원성이 재구성, 생성, 시맨틱 표현에 어떤 영향을 미치는지에 대해 체계적으로 탐구하지 않습니다.

세부 조정(finetuning) 단계에서 VAE 기반 모델은 64 에폭 이후 심각하게 과적합되는 반면, RAE 모델은 256 에폭 동안 안정적으로 유지되는 등 상반되는 동작을 보입니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 시맨틱 표현 기반의 일반화 강점

RAE는 고품질 재구성과 시맨틱이 풍부한 잠재 공간을 모두 제공하며, 스케일러블한 트랜스포머 기반 아키텍처를 가능하게 합니다. RAE는 VAE를 사전학습된 표현 인코더(예: DINO)와 학습된 디코더로 교체한 새로운 클래스의 오토인코더입니다. RAE는 시맨틱과 생성 모델링을 공유된 잠재 표현을 통해 연결하며, 의미적으로 풍부하고 구조적으로 일관되며 확산 친화적인 잠재 공간을 생성합니다.

### 3.2 도메인 확장 및 텍스트-이미지 생성으로의 스케일링

RAE는 ImageNet의 확산 모델링에서 이점을 보여줬으며, 이를 대규모 자유형 텍스트-이미지(T2I) 생성으로 확장할 수 있습니다. RAE 디코더를 웹 데이터, 합성 데이터, 텍스트 렌더링 데이터로 학습시키면 일반적인 충실도는 향상되지만, 텍스트와 같은 특정 도메인에는 타겟 데이터 구성이 필수적입니다.

더 많은 데이터(웹, 합성 및 텍스트)로 학습된 RAE 디코더는 도메인 전반에 걸쳐 일반화됩니다. ImageNet만으로 학습된 디코더는 자연 이미지는 잘 재구성하지만 텍스트 렌더링 장면에는 어려움을 겪습니다. 웹 및 텍스트 데이터를 추가하면 텍스트 재구성이 크게 향상되면서 자연 이미지 품질도 유지됩니다.

### 3.3 스케일 전반에서의 일반화 안정성

RAE는 모든 모델 규모에서 사전학습 전반에 걸쳐 지속적으로 VAE를 능가합니다. 고품질 데이터셋에 대한 세부 조정 시 VAE 기반 모델은 64 에폭 이후 치명적 과적합이 발생하는 반면, RAE 모델은 256 에폭 동안 안정적으로 유지되며 일관되게 더 좋은 성능을 달성합니다. 모든 실험에서 RAE 기반 확산 모델은 더 빠른 수렴과 더 나은 생성 품질을 보입니다.

### 3.4 소규모 인코더에서의 확장성

DINOv2-S, B, L에 걸쳐 재구성 품질이 안정적으로 유지되어, 심지어 작은 표현 인코더 모델도 충분함을 시사합니다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 비교 연구 정리

| 연구 | 방법 | 잠재 공간 | 주요 특징 | 한계 |
|------|------|-----------|-----------|------|
| **LDM/SD (Rombach et al., 2022)** | VAE + Diffusion | 저차원 압축 | 효율적, 광범위 사용 | 구식 CNN 백본, 약한 표현 |
| **DiT (Peebles & Xie, 2023)** | SD-VAE + Transformer | 4채널 잠재 | ViT 기반 확산 | VAE에 완전 의존 |
| **REPA (Yu et al., 2024)** | VAE + 표현 정렬 보조손실 | 저차원 | 시맨틱 정렬 추가 | 보조 손실 필요, 복잡 |
| **VA-VAE (Yao et al., 2025)** | VAE 잠재를 표현 인코더와 정렬 | 저차원 | 재구성+정렬 | 압축 한계 존재 |
| **MAETok (Chen et al., 2025)** | MAE 기반 토크나이저 | 고차원 | 생성 가속 | 고차원에서 생성 품질 저하 |
| **RAE (본 논문, 2025)** | 동결 인코더 + 경량 디코더 | 고차원 비압축 | 보조손실 없이 빠른 수렴 | 재구성 성능 상대적 약세 |
| **Scale-RAE (2026)** | RAE → T2I 스케일링 | 고차원 | 0.5B~9.8B DiT 적용 | 텍스트 도메인 특화 데이터 필요 |

### 4.2 REPA vs. RAE 비교

RAE의 접근법은 **보조 표현 정렬 손실 없이** 더 빠른 수렴을 달성합니다. REPA가 학습 중 별도의 표현 정렬 손실을 필요로 하는 것과 달리, RAE는 동결된 인코더 자체가 이미 풍부한 표현을 담보하므로 학습 파이프라인이 단순합니다.

### 4.3 RecTok과의 관계

RAE는 고차원 잠재 표현을 위해 DiT 너비를 늘려 유망한 생성 성능을 달성하지만, 재구성 성능은 이전 방법들에 비해 뒤떨어집니다. 이 한계는 편집 및 개인화 생성 태스크에 불리하게 작용합니다. 이를 보완하는 후속 연구로 RecTok이 등장하였습니다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 연구에 미치는 영향

#### ① 오토인코더 패러다임의 전환

이 결과들은 오토인코딩을 압축 메커니즘에서 **표현 기반(representation foundation)** 으로 재정의하며, 이를 통해 확산 트랜스포머가 더 효율적으로 학습하고 더 효과적으로 생성할 수 있게 합니다.

#### ② 시맨틱-생성 모델의 통합

RAE 잠재 표현은 미래의 생성 모델링 연구에서 확산 트랜스포머를 효율적이고 강건하게 학습시키기 위한 강력한 후보로 자리매김합니다.

#### ③ 텍스트-이미지 생성으로의 스케일링 확인

RAE는 대규모 텍스트-이미지 생성으로 성공적으로 스케일링될 수 있음이 입증되었습니다. 이 연구 결과는 RAE가 규모가 커질수록 오히려 설계를 단순화시켜주며, 넓은 DDT 헤드와 같은 복잡한 변형이 불필요해짐을 보여줍니다.

#### ④ 노이즈 스케줄링의 일반 원칙 확립

차원 의존 노이즈 시프트를 적용하면 GenEval과 DPG-Bench 점수가 극적으로 향상되며, 유효 잠재 차원에 맞게 스케줄을 조정하는 것이 T2I에 있어 결정적임을 보여줍니다.

---

### 5.2 앞으로 연구 시 고려할 점

#### 🔵 고려점 1: 재구성 성능 개선
RAE의 재구성 성능은 이전 방법들에 비해 뒤떨어지며, 이는 편집 및 개인화 생성 태스크에 불리합니다. 또한 RAE는 차원성이 재구성, 생성, 시맨틱 표현에 어떻게 영향을 미치는지 체계적으로 탐구하지 않습니다. 이를 보완할 후속 연구가 필요합니다.

#### 🔵 고려점 2: 비디오 생성으로의 확장
RAE 잠재 표현은 미래의 생성 모델링 연구에서 확산 트랜스포머를 효율적이고 강건하게 학습시키기 위한 강력한 후보입니다. 이미지를 넘어 비디오, 3D 생성 등으로의 확장 가능성 연구가 필요합니다.

#### 🔵 고려점 3: 도메인 특화 데이터 구성
차원 인식 노이즈 스케줄링은 필수적이지만, RAE의 다른 설계 선택들(더 작은 ImageNet 모델을 위해 개발된 것들)은 T2I 규모에서 수확 체감 현상을 보입니다. 따라서 도메인별 맞춤 학습 레시피 개발이 중요합니다.

#### 🔵 고려점 4: 아키텍처 단순화
스케일링은 프레임워크를 단순화시킵니다. 차원 의존 노이즈 스케줄링은 여전히 중요하지만, 넓은 확산 헤드와 노이즈 증강 디코딩 같은 아키텍처적 복잡성은 스케일에서 무시할 수 있는 이점을 제공합니다. 따라서 모델 규모에 따른 설계 간소화 전략을 연구해야 합니다.

#### 🔵 고려점 5: 멀티모달 표현 인코더 탐색
현재 RAE는 DINOv2, SigLIP, MAE 등 비전 특화 인코더를 사용하지만, 텍스트-비전 정렬 인코더(예: SigLIP-2)와의 조합이 더욱 강력한 생성 기반 모델로 이어질 수 있습니다.

---

## 📚 참고 자료

| # | 출처 | 유형 |
|---|------|------|
| 1 | **Zheng et al. (2025), arXiv:2510.11690** — *Diffusion Transformers with Representation Autoencoders* | 본 논문 |
| 2 | [arxiv.org/abs/2510.11690](https://arxiv.org/abs/2510.11690) | arXiv 공식 페이지 |
| 3 | [rae-dit.github.io](https://rae-dit.github.io/) | 공식 프로젝트 페이지 |
| 4 | [github.com/bytetriper/RAE](https://github.com/bytetriper/RAE) | 공식 GitHub (PyTorch 구현) |
| 5 | [huggingface.co/papers/2510.11690](https://huggingface.co/papers/2510.11690) | HuggingFace 논문 페이지 |
| 6 | **Ma et al. (2026), arXiv:2601.16208** — *Scaling Text-to-Image Diffusion Transformers with Representation Autoencoders* | 후속 연구 |
| 7 | [rae-dit.github.io/scale-rae](https://rae-dit.github.io/scale-rae/) | Scale-RAE 프로젝트 페이지 |
| 8 | [themoonlight.io — Literature Review](https://www.themoonlight.io/en/review/diffusion-transformers-with-representation-autoencoders) | 문헌 리뷰 |
| 9 | **Chen et al. (2025), arXiv:2512.13421** — *RecTok: Reconstruction Distillation along Rectified Flow* | 비교 연구 |
| 10 | **Rombach et al. (2022)** — *High-Resolution Image Synthesis with Latent Diffusion Models* (LDM/Stable Diffusion) | 관련 선행 연구 |
| 11 | **Peebles & Xie (2023)** — *Scalable Diffusion Models with Transformers* (DiT) | 관련 선행 연구 |

> ⚠️ **정확도 주의**: 본 답변의 수식 일부(특히 노이즈 스케줄 시프트의 정확한 형태)는 공개된 arXiv HTML 및 프로젝트 페이지에서 확인 가능한 정보를 기반으로 작성되었으며, 논문 PDF 내 정확한 표기와 세부적으로 다를 수 있습니다. 정밀한 수식 확인을 위해서는 [원문 PDF](https://arxiv.org/pdf/2510.11690)를 직접 참조하시기 바랍니다.
