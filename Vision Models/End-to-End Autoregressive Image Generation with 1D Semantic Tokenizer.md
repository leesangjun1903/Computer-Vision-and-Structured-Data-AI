# End-to-End Autoregressive Image Generation with 1D Semantic Tokenizer

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

EOSTok(End-to-end One-dimensional Semantic Tokenizer)은 기존 **2단계 분리 학습 패러다임**의 근본적 한계를 지적하고, **1D 시각 토크나이저와 자기회귀(AR) 생성 모델을 단일 단계에서 공동 최적화**하는 end-to-end 프레임워크를 제안한다.

### 주요 기여 (3가지)

| 기여 항목 | 내용 |
|---|---|
| **End-to-End 단일 단계 학습** | 토크나이저와 AR 모델을 동시에 학습하여 생성 결과가 토크나이저에 직접 감독 신호를 제공 |
| **APR(Autoregressive Prediction Reconstruction) 손실** | NTP 손실과 픽셀 공간 생성 품질 간의 간극을 브리징하는 새로운 손실 함수 설계 |
| **Implicit VFM Alignment** | Vision Foundation Model(DINOv2)의 의미론적 표현을 2D 공간 구조를 강제하지 않고 1D 잠재 공간에 주입하는 전략 |

**결과:** ImageNet-1K 256×256에서 **gFID 1.48 (가이던스 없음)** 달성, state-of-the-art

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

#### 문제 1: 2D 토크나이저와 AR 모델의 구조적 불일치

기존 이미지 AR 모델은 2D 그리드 구조의 토크나이저(VQ-VAE 등)를 사용하는데, 이는 토큰 간 **양방향 의존성(bidirectional dependency)**을 유발한다. 반면 AR 모델은 **단방향 인과적 팩토리제이션(unidirectional causal factorization)**:

```math
p(\mathbf{z}) = \prod_{n=1}^{L} p(z_n \mid z_{ < n})
```

을 요구하므로 구조적으로 불일치한다.

#### 문제 2: 2단계 분리 학습의 한계

- **Stage 1:** 토크나이저를 재구성 손실로만 학습 → 생성에 최적화되지 않은 잠재 공간
- **Stage 2:** 토크나이저 동결 후 AR 모델 학습 → 토크나이저가 생성 태스크로부터 피드백을 받지 못함

#### 문제 3: NTP 손실과 생성 품질의 간극

NTP(Next Token Prediction) 손실이 낮아도 픽셀 공간에서의 생성 품질이 보장되지 않음. 논문에서는 이를 실험적으로 확인:

| 설정 | rFID↓ | gFID↓ | AR 정확도 | 코드 사용률 |
|---|---|---|---|---|
| Baseline | 1.09 | 3.82 | 11.8% | 99.8% |
| Vanilla E2E | 4.92 | 8.01 | **30.2%** | 51.8% |
| **+ APR loss** | **1.02** | **3.32** | 11.9% | 99.7% |

Vanilla E2E는 AR 정확도는 높지만 잠재 공간이 붕괴(codebook collapse)되어 실제 생성 품질(gFID)은 오히려 저하됨.

---

### 2.2 제안하는 방법 (수식 포함)

#### (A) 1D ViT 토크나이저 구조

인코더는 2D 이미지 패치 $\mathbf{x}_{\text{patch}} \in \mathbb{R}^{N \times D}$와 학습 가능한 쿼리 토큰 $\mathbf{q} \in \mathbb{R}^{L \times D}$를 연결하여 처리:

$$[\mathbf{h}_{\text{Enc}}, \mathbf{z}] = \mathcal{E}_\phi([\mathbf{x}_{\text{patch}}, \mathbf{q}])$$

여기서 $\mathbf{h}_{\text{Enc}}$는 히든 패치 임베딩(버려짐), $\mathbf{z} \in \mathbb{R}^{L \times d}$만 1D 잠재 표현으로 사용.

디코더는 1D 잠재 코드와 마스크 토큰을 연결하여 이미지 재구성:

$$[\varnothing, \mathbf{x}_{\text{recon}}] = \mathcal{D}_\psi([\mathbf{z}_q, \mathbf{m}_{\text{patch}}])$$

#### (B) IBQ 양자화 (Straight-Through Estimation)

$$\text{Ind} = \text{onehot}(\arg\max \mathbf{p}) + [\mathbf{p} - \text{stopgrad}(\mathbf{p})]$$

where $\mathbf{p} = \text{softmax}(\text{logits})$, $\text{logits} = [\mathbf{z}^T\mathcal{C}_1, \ldots, \mathbf{z}^T\mathcal{C}_K] \in \mathbb{R}^K$

안정적인 학습을 위해 $\ell_2$ 정규화를 적용:

$$\text{logits} = \left[\frac{\mathbf{z}^T\mathcal{C}_1}{\|\mathbf{z}\|_2\|\mathcal{C}_1\|_2}, \ldots, \frac{\mathbf{z}^T\mathcal{C}_K}{\|\mathbf{z}\|_2\|\mathcal{C}_K\|_2}\right], \quad \mathbf{p} = \text{softmax}(\text{logits}/\tau)$$

#### (C) VQ-VAE 재구성 손실

$$\mathcal{L}_{\text{VQVAE}}(\phi, \psi) = \mathcal{L}_{\text{recon}}(\mathbf{x}, \mathcal{D}_\psi(\mathbf{z}_q)) + \lambda_{\text{reg}}\mathcal{L}_{\text{reg}}$$

- $\mathcal{L}_{\text{recon}}$: $L_1/L_2$ 손실 + Perceptual loss(LPIPS) + GAN 손실
- $\mathcal{L}_{\text{reg}}$: Commitment loss + Entropy loss

#### (D) End-to-End 공동 학습 손실

$$\mathcal{L}_{\text{E2E}}(\phi, \psi, \theta) = \mathcal{L}_{\text{VQVAE}}(\phi, \psi) + \lambda_{\text{NTP}}\mathcal{L}_{\text{NTP}}(\phi, \theta)$$

#### (E) APR (Autoregressive Prediction Reconstruction) 손실 (★핵심)

AR 모델의 teacher-forcing 예측 $\hat{\mathbf{z}}\_q = \mathcal{G}_\theta(\mathbf{z}_q)$를 디코더로 픽셀 공간에 디코딩하여 원본 이미지와 비교:

$$\mathcal{L}_{\text{APR}}(\phi, \psi, \theta) = \|\mathbf{x} - \mathcal{D}_\psi(\mathcal{G}_\theta(\mathbf{z}_q))\|_2^2$$

실제로는 LPIPS perceptual loss도 추가하여 사용. 학습 시 AR 예측 $\hat{\mathbf{z}}_q$와 실제 $\mathbf{z}_q$를 배치 차원으로 연결하여 디코더에 동시 전달.

#### (F) Implicit Alignment 손실 (VFM 표현 주입) ★핵심

히든 패치 임베딩 $\mathbf{h}_{\text{Enc}}^{[n]}$을 VFM 표현 $\mathbf{y}^{[n]} = f(\mathbf{x})^{[n]}$에 정렬:

$$\mathcal{L}_{\text{implicit}}(\omega, \phi) = -\frac{1}{N}\sum_{n=1}^{N}\text{sim}(h_\omega(\mathbf{h}_{\text{Enc}}^{[n]}), \mathbf{y}^{[n]})$$

(비교) Direct Alignment 손실 (2D 공간 구조를 강요하여 성능 저하):

$$\mathcal{L}_{\text{direct}}(\omega, \phi) = -\frac{1}{L}\sum_{\ell=1}^{L}\text{sim}(h_\omega(\mathbf{z}^{[\ell]}), \mathcal{I}(\mathbf{y})^{[\ell]})$$

#### (G) 최종 EOSTok 목적 함수

$$\mathcal{L}_{\text{EOSTok}}(\phi, \psi, \theta) = \mathcal{L}_{\text{VQVAE}}(\phi, \psi) + \lambda_{\text{NTP}}\mathcal{L}_{\text{NTP}}(\phi, \theta) + \lambda_{\text{APR}}\mathcal{L}_{\text{APR}}(\phi, \psi, \theta) + \min_{\omega_1, \omega_2}\lambda_{\text{sem}}(\mathcal{L}_{\text{implicit}}(\omega_1, \phi) + \mathcal{L}_{\text{decoder-align}}(\omega_2, \phi, \psi))$$

---

### 2.3 모델 구조

```
입력 이미지 x
    ↓ (패치화: P=16, N=HW/P²개 패치)
[x_patch | q] → 1D Causal ViT Encoder (하이브리드 어텐션)
    ↓ → h_Enc (hidden patch embedding, Implicit Alignment용)
    z (1D 잠재 표현, L=256, d=64)
    ↓
IBQ Quantizer (코드북 K=4096)
    ↓ → z_q (양자화된 토큰)
    ↓
Autoregressive Transformer (LlamaGen 기반, AdaLN 추가)
    ↓ → ẑ_q (AR 예측 토큰)
    ↓
1D Causal ViT Decoder
    ↓
재구성 이미지 x_recon / APR 이미지 x_APR

VFM (DINOv2-ViT-L, 동결)
    ↓ → y (VFM 표현)
    ↓
Implicit Alignment: h_Enc ↔ y
Decoder Alignment: h_Dec ↔ y
```

**하이브리드 어텐션 마스크:**
- 인코더: 2D 패치 토큰 간 양방향 / 1D 쿼리 토큰 간 인과적
- 쿼리 → 패치 어텐션: 허용 / 패치 → 쿼리 어텐션: 차단

**AR 모델 스케일:**

| 모델 | 토크나이저 파라미터 | AR 파라미터 | gFID (w/o guidance) |
|---|---|---|---|
| EOSTok-S | 165M | 93M | 3.50 |
| EOSTok-B | 165M | 164M | 2.38 |
| EOSTok-L | 165M | 312M | **1.74** |
| EOSTok-H | 388M | 644M | **1.48** |

---

### 2.4 성능 향상

#### ImageNet 256×256 비교

| 방법 | 토크나이저 | gFID (w/o guidance) | gFID (w/ guidance) |
|---|---|---|---|
| LlamaGen-XL | 2D VQ | 14.77 | 2.62 |
| VAR-d20 | 2D MSRQ | - | 2.57 |
| TiTok-L-32 | 1D VQ | 3.15 | 2.77 |
| MAR-L | 2D SD-VAE | 2.60 | 1.78 |
| AliTok-L | 2D VQ | 1.98 | 1.38 |
| Lightning-DiT-XL | 2D VA-VAE | 2.17 | 1.35 |
| **EOSTok-L (Ours)** | **1D IBQ** | **1.74** | **1.35** |
| **EOSTok-H (Ours)** | **1D IBQ** | **1.48** | **1.38** |

- EOSTok-H: 가이던스 없이 **gFID 1.48** (SOTA)
- 샘플링 속도: DiT-XL/2 대비 **20~100배 빠름** (KV 캐시 활용)
- ImageNet 512 확장: EOSTok-L이 gFID 1.98 (가이던스 없음)으로 확장성 입증

---

### 2.5 한계

1. **단일 도메인 평가:** ImageNet에서만 검증, 다양한 도메인(의료 영상, 위성 이미지 등) 일반화 미검증
2. **학습 비용 증가:** 2단계 대비 약 15~18.6%의 추가 연산 오버헤드
3. **시퀀스 길이 트레이드오프:** 길이 192에서 gFID 최적(3.04), 256에서 재구성 최적(rFID 1.02) — 재구성-생성 간 최적 균형점 존재
4. **코드북 크기 트레이드오프:** 코드북이 클수록 재구성은 좋지만 AR 분류 태스크가 어려워짐
5. **VFM 의존성:** DINOv2, SigLIP2 등 사전학습된 외부 모델에 의존
6. **텍스트 조건부 생성 미검증:** 현재는 클래스 조건부(ImageNet 1K) 생성만 실험

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

### 3.1 일반화를 위한 핵심 설계 요소들

#### (1) 1D 토크나이저의 공간적 불가지론(Spatial Agnosticism)

2D 공간 구조를 제거함으로써 이미지의 전역적 의미를 순서에 구애받지 않는 잠재 공간으로 인코딩. 이는 다음을 가능하게 한다:

- 다양한 이미지 도메인에서 2D 위치 편향 없이 의미론적 특징 학습
- 해상도 변경 시 패치 수($N$)만 조정하면 동일 아키텍처 재사용 가능

논문에서 검증 (Table 4):
- 원본 순서 → gFID 4.10
- 역순 → gFID 10.27
- 무작위 순서 → gFID 7.81

**End-to-end 학습이 토크나이저에 특정 순서 구조를 자동으로 학습시킨다**는 것을 입증. 이는 단순히 외부에서 순서를 부과하는 것이 아니라 학습 과정에서 자연스럽게 형성된 것.

#### (2) Vision Foundation Model (VFM) Implicit Alignment

DINOv2의 강력한 의미론적 표현을 암묵적으로 주입함으로써:

- 학습 데이터 이외 도메인에서도 의미론적으로 일관된 잠재 공간 구성 가능
- 서로 다른 VFM(DINOv2 → SigLIP2) 교체 시에도 성능 향상 유지(Table 12: gFID 3.32 → 3.02), **VFM 선택에 대한 로버스트성** 입증

$$\text{DINOv2: gFID} = 3.32 \quad \rightarrow \quad \text{SigLIP2: gFID} = 3.02$$

#### (3) APR 손실의 일반화 기여

APR 손실은 생성 결과를 픽셀 공간으로 디코딩하여 직접 감독. 이는:

- 토크나이저가 AR 모델이 예측하기 쉬운 표현을 학습하도록 유도
- 코드북 붕괴(collapse) 방지 → 코드 사용률 99.7% 유지
- 다양한 이미지 패턴에 대한 풍부한 잠재 표현 유지 가능

#### (4) 스케일링 일반화

| 모델 크기 | 총 파라미터 | gFID (w/o guidance) |
|---|---|---|
| EOSTok-S | ~258M | 3.50 |
| EOSTok-L | ~477M | 1.74 |
| EOSTok-H | ~1B | **1.48** |

일관된 스케일링 법칙이 존재함 → **더 큰 모델이 더 많은 도메인을 커버할 가능성**

#### (5) 해상도 일반화

EOSTok-L을 ImageNet 512 생성으로 직접 확장:
- 동일 아키텍처(패치 크기 16, 시퀀스 길이 256) 유지
- gFID **1.98** 달성 (TiTok 최고 gFID 3.99 대비 대폭 개선)

이는 1D 토크나이저가 공간 구조에 독립적이므로 해상도 변화에 유연하게 대응 가능함을 보여줌.

#### (6) Nested Dropout을 통한 압축 유연성

시퀀스 길이를 훈련 후 동적으로 조절 가능:

| 시퀀스 길이 | rFID | gFID |
|---|---|---|
| 32 | 17.50 | 22.37 |
| 64 | 1.94 | 3.18 |
| 128 | 1.32 | 3.09 |
| 192 | 1.08 | **3.04** |
| 256 | **1.02** | 3.32 |

Nested dropout은 중요 정보를 앞쪽 토큰에 압축하여, 추론 시 필요에 따라 시퀀스 길이를 가변적으로 조절 가능 → **다양한 컴퓨팅 예산에 대한 적응성**

### 3.2 일반화 한계 및 미해결 과제

1. **텍스트-이미지 생성으로의 확장:** 클래스 레이블 조건에서 텍스트 조건으로 확장 시 성능 보장 불명확
2. **Out-of-Distribution 이미지:** 의료, 위성, 예술 등 도메인 이동(domain shift) 시 VFM alignment의 효과 불명확
3. **비자연적 이미지(Non-photorealistic):** 스케치, 다이어그램 등에 대한 1D 토크나이저 표현 능력 미검증
4. **언어-이미지 통합 모델:** SEED 토크나이저처럼 LLM과 결합하는 멀티모달 설정으로의 일반화 미검토

---

## 4. 최신 연구 비교 분석 (2020년 이후)

### 4.1 비교 연구 맵

```
자기회귀 이미지 생성 계보 (2020~2026)

VQ-VAE-2 (2019) → VQGAN (2021) → LlamaGen (2024) → EOSTok (2026)
                                                    ↑
                                         [2D AR 최적화 한계 인식]

TiTok (2025) ─────────────────────────────────────────────────────┐
(1D 토크나이저 선구자, 32토큰, 마스크 AR)                           │
                                                                    ↓
SEED (2023) → FlexTok (2025) → Semanticist (2025) → EOSTok (2026)
(LLM 연동 1D AR)   (가변 길이)     (주성분 기반)     (E2E + Implicit VFM)
```

### 4.2 주요 경쟁 방법 비교

#### 그룹 1: 2D 연속 잠재 공간 + 확산 모델

| 논문 | 핵심 기법 | gFID (w/o guidance) | 비고 |
|---|---|---|---|
| **LDM (Rombach et al., 2022)** [28] | SD-VAE + U-Net Diffusion | 10.56 | 확산 모델 기초 |
| **DiT-XL/2 (Peebles & Xie, 2023)** [25] | Diffusion Transformer | 9.62 | 트랜스포머 확산 |
| **REPA-XL/2 (Yu et al., 2025)** [49] | DINOv2 표현 정렬 + DiT | 5.90 | VFM 정렬의 선구자 |
| **Lightning-DiT-XL (Yao & Wang, 2025)** [44] | VA-VAE + DiT | 2.17 | VA-VAE로 재구성-생성 딜레마 해결 |
| **MAR-L (Li et al., 2024)** [22] | 연속 잠재 공간 + 마스크 AR | 2.60 | VQ 없는 AR |

**EOSTok와의 차이:** EOSTok는 이산 1D 토크나이저로 AR 모델과 함께 학습하며, 확산 모델 없이 gFID 1.48 달성.

#### 그룹 2: 2D 이산 토크나이저 + AR/마스크

| 논문 | 핵심 기법 | gFID (w/o guidance) |
|---|---|---|
| **MAGVIT-v2 (Yu et al., 2023)** [46] | LFQ + 마스크 AR | 3.65 |
| **VAR (Tian et al., 2024)** [37] | Next-Scale Prediction | - (w/ guidance: 2.57) |
| **RAR-L (Yu et al., 2025)** [47] | 랜덤 순서 AR | 5.39 |
| **AliTok-L (Wu et al., 2025)** [42] | 2D+하이브리드 토크나이저 | 1.98 |
| **IBQ-L (Shi et al., 2025)** [32] | Index Backprop 양자화 | - (w/ guidance: 2.45) |

#### 그룹 3: 1D 토크나이저 (직접 경쟁)

| 논문 | 핵심 기법 | gFID (w/o guidance) | 1D 토큰 수 |
|---|---|---|---|
| **TiTok-L-32 (Yu et al., 2025)** [48] | 32토큰 1D VQ + 마스크 AR | 3.15 | 32 |
| **FlexTok (Bachmann et al., 2025)** [1] | 가변 길이 1D + 플로우 AR | - (w/ guidance: 2.02) | 1~256 |
| **Semanticist (Wen et al., 2025)** [41] | 주성분 기반 1D | - (w/ guidance: 2.57) | 1~256 |
| **GigaTok (Xiong et al., 2025)** [43] | 3B 파라미터 1D VQ | - (w/ guidance: 3.26) | 256 |
| **VFMTok (Zheng et al., 2025)** [53] | VFM 직접 토크나이저 | 2.11 | 256 |
| **ResTok (Zhang et al., 2026)** [52] | 계층적 잔차 1D | - (w/ guidance: 2.34) | 128 |
| **EOSTok-H (Ours)** | **E2E + Implicit VFM** | **1.48** | **256** |

### 4.3 핵심 차별점 분석

```
방법론 비교:
                 토크나이저 학습   AR 감독 피드백   VFM 통합    1D 시퀀스
REPA [49]        분리(2단계)       ✗               ✓(직접)     ✗
VA-VAE [44]      분리(2단계)       ✗               ✓(직접)     ✗
TiTok [48]       분리(2단계)       ✗               ✗           ✓
FlexTok [1]      분리(2단계)       ✗               ✗           ✓
VFMTok [53]      분리(2단계)       ✗               ✓(직접대체)  ✓
EOSTok (본논문)  통합(1단계)       ✓(APR)          ✓(암묵적)   ✓
```

**EOSTok의 결정적 우위:** 
1. **End-to-end 학습** → 토크나이저가 생성 태스크에 적응
2. **APR 손실** → NTP-픽셀 품질 간격 해소
3. **Implicit VFM Alignment** → 2D 구조 강제 없이 의미론적 표현 주입

---

## 5. 앞으로의 연구에 미치는 영향과 연구 시 고려할 점

### 5.1 연구에 미치는 영향

#### (1) 패러다임 전환: 분리 학습 → 통합 학습

EOSTok은 **토크나이저와 생성 모델의 공동 최적화**가 분리 학습보다 우월함을 실험적으로 증명. 이는 향후:
- 확산 모델에서도 VAE와 DiT의 공동 학습 패러다임으로의 전환 촉진 (REPA-E [21] 같은 방향)
- 언어 모델에서 토크나이저(BPE 등)와 LLM의 공동 최적화 연구에도 영향 가능

#### (2) 1D 이미지 토크나이저의 정당성 확립

1D 토크나이저가 AR 생성에 더 적합하다는 가설을 SOTA 성능으로 검증. **TiTok의 아이디어를 확장**하여:
- 공격적인 압축(32토큰) 없이도 1D 표현이 효과적임을 증명
- 2D 공간 편향 제거 → 더 자유로운 토큰 순서 학습 가능

#### (3) VFM 활용의 새로운 방향성

Direct Alignment vs. Implicit Alignment의 비교 실험을 통해:
- **2D 공간 구조를 강제하는 VFM 정렬은 1D AR 생성에 해롭다**는 중요한 인사이트 제공
- 이는 향후 멀티모달 모델, 비디오 생성 등에서 VFM 통합 방식 설계에 지침이 됨

#### (4) 코드북 붕괴 문제 해결

APR 손실을 통한 코드북 붕괴 방지 메커니즘은:
- VQ 기반 모든 생성 모델에 적용 가능한 범용적 해결책
- 이산 잠재 공간 모델의 학습 안정성 연구에 새로운 방향 제시

#### (5) AR 모델의 확산 모델 대비 경쟁력 강화

- gFID 1.48 (w/o guidance)은 많은 확산 모델을 능가
- **샘플링 속도 20~100배 향상** (KV 캐시 활용) → 실용적 배포 측면에서 AR 모델의 우위

---

### 5.2 앞으로 연구 시 고려할 점

#### (1) 텍스트 조건부 생성으로의 확장

**현재 한계:** ImageNet 클래스 조건부만 검증.

**연구 방향:**
- 텍스트 임베딩(CLIP, T5)을 AR 모델의 조건 신호로 통합
- LVLM(Large Vision-Language Model)과 1D 토크나이저 결합 (SEED 토크나이저 [11] 방향 발전)
- 텍스트-이미지 쌍 데이터에서 APR 손실 적용 시 텍스트 정렬도 고려 필요

#### (2) 비디오 및 고해상도 생성

**현재 한계:** 256×256, 512×512까지만 검증.

**연구 방향:**
- 시간 축을 포함하는 3D 토크나이저 설계 (공간+시간 1D 시퀀스)
- 고해상도(1024×1024 이상)에서 시퀀스 길이 폭발 문제 해결
  - Hierarchical 1D tokenization (VFMTok [53], ResTok [52] 참고)
  - Sliding window 또는 청크(chunk) 기반 AR 생성

#### (3) 재구성-생성 트레이드오프 최적화

**현재 한계:** 시퀀스 길이 192가 생성 최적, 256이 재구성 최적.

**연구 방향:**
- Nested dropout 비율의 더 정교한 스케줄링
- 적응적 시퀀스 길이 (이미지 복잡도에 따라 동적 조절)
- 재구성과 생성을 동시에 만족하는 파레토 최적 학습 목표 설계

#### (4) End-to-End 학습 안정성

**현재 한계:** Vanilla E2E 학습 시 코드북 붕괴 발생, APR 손실로 완화.

**연구 방향:**
- APR 손실 외 추가적인 안정화 기법 탐색 (EMA 코드북, 온도 어닐링 등)
- 학습 초기 불안정성 해소를 위한 warm-up 전략
- $\lambda_{\text{NTP}}, \lambda_{\text{APR}}, \lambda_{\text{sem}}$ 가중치의 자동 조절(Adaptive Loss Weighting)

#### (5) VFM 선택 및 업데이트 전략

**현재 한계:** DINOv2 동결하여 사용.

**연구 방향:**
- VFM을 부분 미세조정(partial fine-tuning)하여 생성 태스크에 맞게 적응
- 더 강력한 VFM(DINOv3 [34], SigLIP2 [38]) 활용 효과 체계적 분석
- 도메인 특화 VFM(의료, 위성 등) 활용 시 도메인 일반화 성능 분석

#### (6) 다중 도메인 일반화

**현재 한계:** ImageNet 단일 도메인.

**연구 방향:**
- LAION, CC3M 등 대규모 다양한 데이터셋에서의 학습 및 평가
- Zero-shot / Few-shot 생성 성능 평가
- 의료 영상(CheXpert 등), 원격 탐사, 예술 이미지 등 전문 도메인 적용

#### (7) 멀티모달 통합

**연구 방향:**
- 1D 이미지 토큰과 언어 토큰의 통합 시퀀스 모델링
- 이해(understanding)와 생성(generation)을 동시에 수행하는 통합 AR 모델
- 기존 SEED [11] 방향에서 EOSTok의 E2E 학습 전략 적용

#### (8) 계산 효율성 개선

**현재 한계:** 학습 시 VFM 순전파 비용 추가 (EOSTok-L: 162 GFLOPs).

**연구 방향:**
- VFM 특징을 사전 캐싱(pre-caching)하여 학습 효율 개선
- 경량 VFM 대체제 탐색
- 양자화(quantization), 지식 증류(knowledge distillation)를 통한 추론 최적화

---

## 참고 문헌

본 분석은 다음 논문 및 자료를 기반으로 작성되었습니다:

**주 논문:**
- Wenda Chu et al., "End-to-End Autoregressive Image Generation with 1D Semantic Tokenizer," arXiv:2605.00503v2, May 2026.

**논문 내 인용 참고문헌 (주요):**
- [1] Bachmann et al., "FlexTok: Resampling Images into 1D Token Sequences of Flexible Length," ICML 2025.
- [4] Chang et al., "MaskGIT: Masked Generative Image Transformer," CVPR 2022.
- [10] Esser et al., "Taming Transformers for High-Resolution Image Synthesis (VQGAN)," CVPR 2021.
- [11] Ge et al., "Making LLaMA See and Draw with SEED Tokenizer," arXiv:2310.01218, 2023.
- [22] Li et al., "Autoregressive Image Generation without Vector Quantization (MAR)," NeurIPS 2024.
- [24] Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision," arXiv:2304.07193, 2023.
- [25] Peebles & Xie, "Scalable Diffusion Models with Transformers (DiT)," ICCV 2023.
- [28] Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models (LDM)," CVPR 2022.
- [32] Shi et al., "Scalable Image Tokenization with Index Backpropagation Quantization (IBQ)," ICCV 2025.
- [36] Sun et al., "Autoregressive Model Beats Diffusion: LlamaGen," arXiv:2406.06525, 2024.
- [37] Tian et al., "Visual Autoregressive Modeling: VAR," NeurIPS 2024.
- [41] Wen et al., "Semanticist," arXiv:2503.08685, 2025.
- [42] Wu et al., "AliTok," arXiv:2506.05289, 2025.
- [43] Xiong et al., "GigaTok," arXiv:2504.08736, 2025.
- [44] Yao & Wang, "Lightning-DiT / VA-VAE," arXiv:2501.01423, 2025.
- [46] Yu et al., "MAGVIT-v2," arXiv:2310.05737, 2023.
- [47] Yu et al., "RAR," ICCV 2025.
- [48] Yu et al., "TiTok: An Image is Worth 32 Tokens," NeurIPS 2025.
- [49] Yu et al., "REPA: Representation Alignment for Generation," ICLR 2025.
- [52] Zhang et al., "ResTok," arXiv:2601.03955, 2026.
- [53] Zheng et al., "VFMTok," arXiv:2507.08441, 2025.
