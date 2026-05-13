
# Masked Autoencoders Are Effective Tokenizers for Diffusion Models (MAETok)

> **출처 / 참고자료**
> - **arXiv**: [arXiv:2502.03444](https://arxiv.org/abs/2502.03444) (v1: 2025.02.05, v2: 2025.05.30)
> - **ICML 2025 공식 게재**: [PMLR v267, pp.8145–8171](https://proceedings.mlr.press/v267/chen25v.html)
> - **OpenReview (ICML 2025 Spotlight)**: [openreview.net/forum?id=dzwUOiBlQW](https://openreview.net/forum?id=dzwUOiBlQW)
> - **MarkTechPost 해설**: [marktechpost.com](https://www.marktechpost.com/2025/02/08/this-ai-paper-introduces-maetok-a-masked-autoencoder-based-tokenizer-for-efficient-diffusion-models/)
> - **GitHub 공식 코드**: [github.com/Hhhhhhao/continuous_tokenizer](https://github.com/Hhhhhhao/continuous_tokenizer)
> - **ResearchGate PDF**: [researchgate.net](https://www.researchgate.net/publication/388755221_Masked_Autoencoders_Are_Effective_Tokenizers_for_Diffusion_Models)
> - **themoonlight.io 리뷰**: [themoonlight.io](https://www.themoonlight.io/en/review/masked-autoencoders-are-effective-tokenizers-for-diffusion-models)

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

최근 Latent Diffusion Model(LDM)은 고해상도 이미지 생성에서 탁월한 성과를 보여왔으나, diffusion model의 학습과 생성을 위한 tokenizer의 잠재 공간(latent space) 특성은 충분히 탐구되지 않았다. 이 논문은 이론적·실험적으로 **생성 품질 향상이 더 적은 Gaussian Mixture 모드와 더 판별적(discriminative)인 특성을 가진 잠재 분포와 긴밀하게 연관된다**는 것을 발견했다.

이 논문의 핵심 주장은 **variational 제약이 아닌 잠재 공간의 구조 자체**가 효과적인 diffusion model을 위해 결정적으로 중요하다는 것이다.

### 1.2 주요 기여

| 기여 항목 | 내용 |
|---|---|
| 이론적 분석 | GMM 모드 수와 diffusion loss의 관계를 이론적으로 증명 |
| MAETok 제안 | Masked Autoencoder 기반의 새로운 tokenizer 설계 |
| VAE 필요성 반박 | VAE의 variational form 없이도 SOTA 달성 |
| 효율성 | $76\times$ 빠른 학습, $31\times$ 높은 추론 처리량 |

논문의 핵심 기여는 **MAETok(MAE-Tok)이라는 혁신적인 masked autoencoder 아키텍처**로, diffusion model의 더 나은 추정과 생성을 위한 판별적 표현을 효율적으로 학습한다는 점이다.

또한 **잠재 공간의 조직화가 모델 성능에서 핵심적 역할**을 한다는 것을 확립하여, variational autoencoder 기반 접근에서 단순한 autoencoder 기반 접근으로의 패러다임 전환을 지지한다.

---

## 2. 상세 분석: 문제 → 방법 → 구조 → 성능 → 한계

### 2.1 해결하고자 하는 문제

Diffusion model에서 핵심적인 문제는 잠재 공간의 품질과 구조이다. 전통적으로 VAE(Variational Autoencoder)가 tokenizer로 사용되어 잠재 공간을 정규화했으나, VAE는 정규화 제약으로 인해 높은 픽셀 수준의 충실도(fidelity)를 달성하기 어려운 경우가 많다.

특히 학습 데이터 분포가 지나치게 복잡하거나 multi-modal, 즉 충분히 판별적이지 않을 경우, denoising network가 잠재 공간의 얽힌(entangled) 글로벌 구조를 포착하기 어려워 생성 품질이 저하될 수 있다.

**기존 방법의 한계 요약:**

$$\text{VAE Loss} = \underbrace{\mathcal{L}_{\text{recon}}}_{\text{재구성 손실}} + \underbrace{\beta \cdot D_{KL}(q_\phi(z|x) \| p(z))}_{\text{KL 정규화 항}}$$

- KL 제약은 잠재 공간을 단순화하지만, 동시에 **표현력과 판별 능력을 저하**시킴
- VAE는 강력한 생성 결과를 낼 수 있으나, 부과된 정규화로 인해 고픽셀 충실도의 재구성을 달성하는 데 어려움을 겪는 경우가 많다.

---

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 이론적 분석: GMM 모드와 Diffusion Loss

연구팀은 AE, VAE, 최근 등장한 representation-aligned VAE를 GMM(Gaussian Mixture Model)을 잠재 공간에 피팅하는 방식으로 연구했다. 실험적으로 더 판별적인 특성을 가진 잠재 공간, 즉 GMM 모드가 더 적은 공간이 더 낮은 diffusion loss를 생성하는 경향이 있음을 보였다. 이론적으로는 GMM 모드가 더 적은 잠재 분포가 diffusion model의 손실을 더 낮게, 즉 추론 중 더 나은 샘플링으로 이어진다는 것을 증명했다.

잠재 데이터 분포를 GMM으로 정의하면:

$$p_0(z) = \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(z; \mu_k, \Sigma_k)$$

여기서 $K$는 GMM 모드 수, $\pi_k$는 혼합 가중치, $\mu_k$는 평균, $\Sigma_k$는 공분산 행렬이다.

DDPM의 score matching loss는 다음과 같다:

$$\mathcal{L}_{\text{DDPM}}(t) = \mathbb{E}_{z_0 \sim p_0, \epsilon \sim \mathcal{N}(0,I)} \left\| \epsilon - \epsilon_\theta\bigl(\underbrace{\alpha_t z_0 + \sigma_t \epsilon}_{z_t}, t\bigr) \right\|^2$$

**Theorem 2.1 (비공식 버전):** 이론적 분석에서 잠재 데이터 분포가 GMM이라 가정하고, DDPM의 score matching loss를 고려했을 때, GMM 모드 수 $K$가 클수록 denoising network의 학습이 더 어려워지고 더 많은 학습 샘플이 필요함이 도출된다.

$$\text{필요 샘플 수} \propto K^4 \cdot d^5 \cdot B^6 / \varepsilon^2$$

여기서 $d$는 잠재 공간 차원, $B$는 GMM 최대 평균 노름, $\varepsilon$는 목표 추정 오차이다. 즉, **$K$(GMM 모드 수)가 줄어들수록 학습 효율과 생성 품질이 향상**된다.

---

#### 2.2.2 MAETok의 학습 목표

MAETok의 전체 학습 손실은 다음과 같이 구성된다:

$$\mathcal{L}_{\text{MAETok}} = \mathcal{L}_{\text{pixel}} + \lambda_{\text{aux}} \cdot \mathcal{L}_{\text{aux}}$$

- $\mathcal{L}_{\text{pixel}}$: 픽셀 재구성 손실 (Decoder $\mathcal{D}$를 통한 이미지 복원)
- $\mathcal{L}_{\text{aux}}$: 보조 목표 손실 — 마스킹된 토큰의 특성을 예측

보조 손실 $\mathcal{L}_{\text{aux}}$는 다음과 같이 다양한 타겟을 포함한다:

$$\mathcal{L}_{\text{aux}} = \sum_{i \in \mathcal{M}} \left\| \hat{f}_{\text{aux}}(z_{\text{seen}})_i - f_{\text{target}}(x_i) \right\|^2$$

여기서 $\mathcal{M}$은 마스킹된 패치 인덱스 집합, $f_{\text{target}}$은 HOG, DINO-v2, CLIP 등 사전 학습된 특성이다.

MAETok은 인코더에서 40~60%의 마스크 비율로 마스크 모델링을 통해 학습되며, 마스킹되지 않은 토큰으로부터 마스킹된 토큰의 HOG, DINO-v2, CLIP 특성 등 여러 타겟 특성을 보조 얕은 디코더(auxiliary shallow decoders)를 사용해 예측한다.

---

### 2.3 모델 구조

MAETok의 방법론은 Vision Transformer(ViT) 기반 아키텍처로 인코더와 디코더를 모두 갖춘 autoencoder를 학습하는 것이다. 인코더는 패치로 나뉜 입력 이미지와 학습 가능한 잠재 토큰들을 함께 처리한다. 학습 중 일부 입력 토큰이 무작위로 마스킹되어 모델이 나머지 보이는 영역으로부터 누락된 데이터를 추론하도록 강제한다. 이 메커니즘은 모델이 판별적이고 의미론적으로 풍부한 표현을 학습하는 능력을 향상시킨다. 또한 보조 얕은 디코더가 마스킹된 특성을 예측하여 잠재 공간의 품질을 더욱 정제한다.

```
[입력 이미지 x]
      │
      ▼
[패치 분할 & 토큰화]  ← 40~60% 무작위 마스킹
      │
      ▼
[ViT Encoder (ViT-Base)]
  ┌───┴────────────────────────────┐
  │                                │
  ▼                                ▼
[Latent Tokens h]        [Aux Decoders]
  (128 tokens)           HOG / DINOv2 / CLIP
  │
  ▼
[Pixel Decoder D]
  │
  ▼
[재구성 이미지 x̂]
```

MAETok은 학습 가능한 잠재 토큰을 사용하는 최근의 1D 토크나이저 설계를 기반으로 구축되었다.

**1D 토크나이저 구조의 핵심 특성:**

| 구성 요소 | 세부 사항 |
|---|---|
| 인코더 | ViT-Base (scratch 초기화) |
| 디코더 | ViT-Base (pixel decoder) |
| 잠재 토큰 수 | **128 tokens** (1D) |
| 마스크 비율 | 40~60% |
| 보조 타겟 | HOG, DINO-v2, CLIP |
| VAE 사용 | ❌ 불필요 (plain AE) |

확산 기반 이미지 생성 작업을 위해 SiT(Li et al., 2024a)와 LightningDiT(Yao & Wang, 2025)를 MAETok 학습 후 사용했다. 패치 크기는 1로 설정하고 1D 위치 임베딩을 사용했다. 분석 및 절제 연구에는 458M 파라미터의 SiT-L을 사용했으며, 주요 결과를 위해서는 675M 파라미터의 SiT-XL을 4M 스텝, LightningDiT는 400K 스텝으로 ImageNet 256 및 512 해상도에서 학습했다.

---

### 2.4 성능 향상

MAETok은 $512 \times 512$ 생성에서 **gFID 1.69**, $76\times$ 빠른 학습, $31\times$ 높은 추론 처리량이라는 현저한 실용적 개선을 달성했다.

128 토큰만을 사용한 순수한 AE 아키텍처로 학습된 SiT-XL은 CFG 없이 일관되게 더 나은 gFID와 IS를 달성했으며, 256 해상도에서 REPA를 3.59 gFID 차이로 능가하고 512 해상도에서 2.79의 SOTA 수준 gFID를 기록했다. CFG를 사용할 경우, 256 해상도에서 VAE로 학습된 경쟁 자기회귀 및 diffusion 기반 기준 모델들과 비슷한 성능을 달성했다. 256 토큰을 사용하는 2B 파라미터 USiT를 능가하고 512 해상도에서 1.69 gFID, 304.2 IS의 새로운 SOTA를 달성했다.

| 모델 | 해상도 | 토큰 수 | gFID | IS |
|---|---|---|---|---|
| SiT-XL + MAETok (no CFG) | 256×256 | 128 | REPA +3.59↓ | - |
| SiT-XL + MAETok (CFG) | 512×512 | 128 | **1.69** | 304.2 |
| LightningDiT + MAETok (no CFG) | 256×256 | 128 | **2.56** | 224.5 |
| LightningDiT + MAETok (CFG) | 256×256 | 128 | **1.72** | - |

잠재 공간의 품질은 잠재 표현에 대한 선형 프로빙(LP) 정확도를 통해 평가했으며, 이를 잠재 코드에서 의미 정보가 얼마나 잘 보존되는지의 대리 지표로 사용했다.

---

### 2.5 한계

논문에서 직접 인정하거나 실험 설계에서 유추되는 한계는 다음과 같다:

1. **평가 범위의 제한**: 실험은 주로 256×256 및 512×512 해상도의 ImageNet 데이터셋에서 수행되었으며, FID 등 지표로 검증되었으나, 더 다양한 도메인(의료, 위성 이미지 등)에 대한 일반화는 직접 검증되지 않았다.

2. **토크나이저 설계의 단순성과 표현력 간 트레이드오프**: 128 토큰이라는 극도로 압축된 표현이 매우 세밀한 텍스처 표현에 얼마나 한계를 가지는지 명확히 분석되지 않았다.

3. **보조 타겟 선택 의존성**: HOG, DINO-v2, CLIP 등의 특성을 보조 타겟으로 사용하기 때문에, 이들 사전 학습 모델의 품질에 의존적이다.

4. **텍스트-이미지 생성으로의 확장**: 기존 접근 방식들은 여전히 계산 오버헤드와 확장성 한계를 겪고 있으며, 텍스트 조건부 생성에서의 검증은 제한적이다.

---

## 3. 일반화 성능 향상 가능성

MAETok은 AE를 Masked Autoencoder(MAE)로 학습시키는 방법을 제안하는데, 이는 **프록시 특성을 재구성함으로써 더 일반화되고 판별적인 표현을 발견할 수 있는 자기지도 학습(self-supervised learning) 패러다임**이다.

일반화 성능 향상 가능성은 다음 세 측면에서 분석할 수 있다:

### 3.1 GMM 모드 감소 → 더 단순한 분포 → 일반화

잠재 공간의 GMM 모드 수가 적을수록 일반적으로 더 낮은 diffusion loss와 더 나은 생성 성능에 대응한다. 충분히 작은 이산화 스텝에서 diffusion model의 생성 품질은 denoising network의 학습 손실에 의해 지배된다.

$$\underbrace{K \downarrow}_{\text{GMM 모드 감소}} \Rightarrow \underbrace{\mathcal{L}_{\text{score}} \downarrow}_{\text{score matching 손실 감소}} \Rightarrow \underbrace{\text{일반화 향상}}$$

### 3.2 판별적 특성 학습 → 의미론적 일반화

이 방식으로 학습된 tokenizer는 **더 판별적인 잠재 공간**을 생성한다 — 즉, 이미지 특성 간 의미 있는 차이를 더 잘 포착한다. 이는 diffusion model에서 사용될 때 더 효과적인 학습과 더 나은 결과로 이어진다.

UMAP 시각화를 통해 (a) raw pixel 타겟 MAETok, (b) HOG 타겟 MAETok, (c) DINOv2 타겟 MAETok, (d) CLIP 타겟 MAETok의 잠재 공간이 서로 다른 구조를 보여주며, DINO-v2와 CLIP 타겟을 사용한 경우 가장 판별적인 클러스터링이 형성됨을 확인할 수 있다.

### 3.3 MS-COCO 검증 → 도메인 외 일반화 단서

Tokenizer 평가를 위해 ImageNet뿐만 아니라 **MS-COCO 검증 셋**에서도 rFID, PSNR, SSIM을 보고하며, 잠재 공간 평가를 위해 선형 프로빙(LP) 정확도를 측정했다.

이는 MAETok이 ImageNet 외 데이터셋에도 일정 수준의 재구성 일반화가 가능함을 시사한다.

### 3.4 자기지도 학습(SSL) 패러다임의 일반화 잠재력

이러한 방식으로 학습된 tokenizer는 더 판별적인 잠재 공간을 생성하여 이미지 특성 간 의미 있는 차이를 더 잘 포착한다. 이는 diffusion model에서 더 효과적인 학습과 더 나은 결과로 이어진다. 이론적 분석과 실용적 실험을 통해 잘 구조화된 잠재 공간을 갖는 것이 VAE와 같은 복잡한 설계를 사용하는 것보다 더 중요하다는 것을 발견했다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 주요 관련 연구 타임라인

```
2020 ─ DDPM (Ho et al.) : 확산 모델의 기초 정립
2021 ─ VQGAN (Esser et al.) : 이산 잠재 공간 토크나이저
2022 ─ LDM (Rombach et al.) : VAE + Diffusion 결합, Stable Diffusion 기반
       MAE (He et al.) : 마스크 자기지도 학습
2023 ─ MAGVIT-v2 (Yu et al.) : 비디오/이미지 통합 토크나이저
       DiT (Peebles & Xie) : Transformer 기반 Diffusion
2024 ─ REPA (Yu et al.) : Representation Alignment
       SiT (Li et al.) : Scalable Interpolant Transformer
       VA-VAE (Yao & Wang) : Vision Foundation Model Aligned VAE
2025 ─ MAETok (Chen et al.) ← 본 논문 [ICML 2025 Spotlight]
```

### 4.2 세부 비교 분석

| 방법 | 토크나이저 유형 | 잠재 공간 특성 | 주요 특징 | 한계 |
|---|---|---|---|---|
| **LDM (2022)** | KL-VAE | Smooth (KL 정규화) | Stable Diffusion 기반 | 판별적 특성 부족 |
| **MAGVIT-v2 (2023)** | 이산 VQ | 이산 토큰 | LLM과 결합 가능 | 연속 잠재 공간 미지원 |
| **REPA (2024)** | VAE | Representation Aligned | DINOv2 정렬로 학습 가속 | VAE 복잡성 유지 |
| **VA-VAE (2025)** | Vision-Aligned VAE | Discriminative | Foundation model 정렬 | Variational form 유지 |
| **MAETok (2025)** | **Plain AE + MAE** | **Most Discriminative** | VAE 불필요, 128토큰, SOTA | ImageNet 중심 검증 |

비교 연구로서 MAGVIT-v2는 비디오와 이미지 모두를 위한 간결하고 표현력 있는 토큰을 생성하는 비디오 토크나이저로, 이 새로운 토크나이저를 장착한 LLM이 ImageNet 및 Kinetics를 포함한 표준 이미지·비디오 생성 벤치마크에서 diffusion model을 능가함을 보였다.

**MAETok의 차별성:**

$$\underbrace{\text{LDM/VAE}}_{\text{KL 제약, 복잡한 분포}} \xrightarrow{\text{개선}} \underbrace{\text{REPA/VA-VAE}}_{\text{표현 정렬, VAE 유지}} \xrightarrow{\text{개선}} \underbrace{\text{MAETok}}_{\text{VAE 불필요, MAE 자기지도, 최소 GMM 모드}}$$

---

## 5. 앞으로의 연구에 미치는 영향과 고려점

### 5.1 앞으로의 연구에 미치는 영향

#### 5.1.1 패러다임 전환: VAE → Plain AE

이론적 분석과 실용적 실험을 통해 잘 구조화된 잠재 공간을 갖는 것이 VAE와 같은 복잡한 설계를 사용하는 것보다 더 중요하다는 것을 발견했다. MAETok은 단 128 토큰만으로 표준 벤치마크에서 SOTA 이미지 생성을 달성하면서 훨씬 더 빠르고 효율적이다. 이 연구는 잠재 표현이 생성 성능에 미치는 영향에 대한 새로운 통찰을 제공하고 고품질 이미지 합성을 위한 실용적이고 확장 가능한 해법을 제시한다.

#### 5.1.2 잠재 공간 이론 연구의 촉진

GMM 모드 수와 diffusion loss의 관계를 이론적으로 규명한 것은, 향후 연구자들이 **tokenizer 설계 시 잠재 공간의 기하학적 구조**를 명시적인 최적화 목표로 삼을 수 있는 이론적 기반을 제공한다.

#### 5.1.3 효율성 혁신

MAETok은 단 128 토큰을 사용하면서 표준 벤치마크에서 SOTA 이미지 생성을 달성하는 동시에 훨씬 더 빠르고 효율적이며, 이는 고품질 이미지 합성을 위한 실용적이고 확장 가능한 해법을 제시한다.

#### 5.1.4 자기지도 학습과 생성 모델의 융합

MAE를 자기지도 패러다임으로 사용하여 **프록시 특성을 재구성함으로써 더 일반화되고 판별적인 표현을 발견**하는 접근은, 자기지도 표현 학습과 생성 모델 연구의 융합이라는 새로운 연구 방향을 열어준다.

---

### 5.2 앞으로 연구 시 고려할 점

#### ① 다양한 도메인과 해상도에서의 검증

현재 실험은 ImageNet 중심으로 이루어졌다. 향후 연구에서는:
- 의료 이미지(MRI, CT), 위성 이미지, 멀티모달 데이터 등에서의 일반화 성능 검증이 필요하다.
- $1024 \times 1024$ 이상의 초고해상도에서 128 토큰의 한계 분석이 필요하다.

#### ② 보조 타겟의 최적 조합 탐색

MAETok은 HOG, DINO-v2, CLIP 특성 등 여러 타겟 특성을 예측하는데, 보조 타겟의 조합이 도메인에 따라 어떻게 달라져야 하는지, 그리고 최적 조합을 자동으로 학습하는 방법(예: NAS, 메타러닝)의 탐구가 필요하다.

#### ③ 텍스트-이미지 생성으로의 확장

MAETok은 클래스 조건부 ImageNet 생성에서 검증되었으나, Stable Diffusion 등의 텍스트-이미지 생성 시스템에 적용할 경우 **텍스트-이미지 정렬(alignment) 유지 여부**를 추가로 검증해야 한다.

#### ④ 이론적 확장: 비-GMM 분포에서의 보장

현재 이론은 GMM 가정 하에 성립한다. 실제 고해상도 이미지 분포는 GMM으로만 충분히 설명되지 않을 수 있으므로, 보다 일반적인 비-GMM 분포에서의 이론적 보장 연구가 필요하다.

#### ⑤ 연속 잠재 토큰 수의 최적화

$$\underbrace{128 \text{ tokens}}_{\text{MAETok}} \text{ vs. } \underbrace{256 \sim 1024 \text{ tokens}}_{\text{기존 VAE 기반}}$$

더 적은 토큰으로도 충분한가, 아니면 특정 작업에서는 더 많은 토큰이 필요한가에 대한 체계적 분석이 남아 있다.

#### ⑥ 비디오 및 3D 생성으로의 확장

MAETok은 autoencoder 프레임워크 내에서 마스크 모델링을 활용하여 더 구조화된 잠재 공간을 개발하는데, 이 원리를 비디오(시간 차원)나 3D 포인트 클라우드로 확장할 경우 추가적인 아키텍처 설계 고려가 필요하다.

---

## 📌 종합 결론

| 항목 | 내용 |
|---|---|
| **논문 게재** | ICML 2025 Spotlight (PMLR v267, pp.8145–8171) |
| **저자 소속** | CMU, HKU, Peking Univ., AMD |
| **핵심 발견** | GMM 모드 수 ↓ → Diffusion Loss ↓ → 생성 품질 ↑ |
| **핵심 제안** | MAETok: Masked AE + 보조 타겟(HOG/DINO/CLIP) |
| **최고 성능** | gFID 1.69 @ 512×512, 128 tokens, $76\times$ 학습 속도 |
| **패러다임 전환** | VAE 불필요 → Plain AE + 판별적 잠재 공간으로 충분 |
| **코드 공개** | github.com/Hhhhhhao/continuous_tokenizer |

MAETok은 $512 \times 512$ 생성에서 gFID 1.69, $76\times$ 빠른 학습, $31\times$ 높은 추론 처리량이라는 현저한 실용적 개선을 달성했으며, 이 연구의 발견은 variational 제약이 아닌 **잠재 공간의 구조 자체**가 효과적인 diffusion model을 위해 결정적으로 중요하다는 것을 보여준다.
