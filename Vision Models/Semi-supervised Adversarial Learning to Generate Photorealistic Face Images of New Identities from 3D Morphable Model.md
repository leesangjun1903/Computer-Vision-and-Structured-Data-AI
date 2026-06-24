# Semi-supervised Adversarial Learning to Generate Photorealistic Face Images of New Identities from 3D Morphable Model

**Gecer et al., ECCV 2018**

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

이 논문은 **3D Morphable Model(3DMM)**에서 샘플링된 합성 얼굴 이미지를 조건으로, **새로운 신원(new identity)**의 다양한 포즈·표정·조명을 가진 포토리얼리스틱 얼굴 이미지를 생성하는 **준지도(semi-supervised) 적대적 학습 프레임워크**를 제안합니다.

기존 방법들이 다음과 같은 한계를 지님을 지적합니다:
- **pix2pix (Isola et al.)**: 대규모 쌍(paired) 데이터가 필요
- **CycleGAN (Zhu et al.)**: 비지도 방식이나, 사이클 일관성만으로는 올바른 도메인 변환을 보장하지 못함 (under-constrained)

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **Semi-supervised 적대적 스타일 변환** | 소수의 쌍(paired) 데이터와 대규모 비쌍(unpaired) 데이터를 조합하여 두 방향 생성 네트워크를 제약 |
| **역매핑 네트워크를 판별기로 활용** | $G'$ 네트워크가 generator와 pair-matching discriminator 이중 역할 수행 |
| **Set-based Identity Loss** | 사전 학습된 임베딩 네트워크 위에서 새로운 신원의 일관성을 보존하는 손실 함수 제안 |

---

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

**도메인 격차(Domain Gap)**: 3DMM으로 렌더링된 합성 얼굴 이미지는 실제 얼굴인식 시스템에 바로 사용하기에는 현실감이 부족합니다. 이를 **도메인 적응(domain adaptation)** 문제, 즉 합성 도메인 $S$를 실제 도메인 $R$로 변환하는 문제로 정식화합니다.

구체적으로 해결해야 할 세 가지 세부 문제:

1. **사이클 일관성만으로는 부족**: CycleGAN의 사이클 손실은 단순 전이성(transitivity)만을 보장하므로, 의도된 도메인 변환을 보장하지 않음
2. **모드 붕괴(Mode Collapse)**: 노이즈 벡터에서 시작하는 GAN은 다양성 부족 문제 발생
3. **신원 보존 문제**: 새로운 신원(unknown identity)의 경우, 기존 softmax 기반 분류기를 사용한 신원 보존이 불가능 (닭-달걀 문제)

---

### 2.2 제안하는 방법 (수식 포함)

#### ① 비지도 도메인 적응 (Unsupervised Domain Adaptation)

합성 이미지 $x \in S$를 생성기 $G: S \rightarrow \hat{R}$로 포토리얼리스틱 도메인으로 변환하고, 역매핑 네트워크 $G': \hat{R} \rightarrow \hat{S}$로 다시 합성 도메인으로 복원하는 단방향 사이클 일관성 손실:

$$\mathcal{L}_{cyc} = \mathbb{E}_{x \in S} \|G'(G(x)) - x\|_1 \tag{1}$$

판별기 $D_R$, $D_S$는 **BEGAN(Boundary Equilibrium GAN)** 구조(오토인코더 기반)로 구성됩니다:

$$\mathcal{L}_G = \mathbb{E}_{x \in S} \|G(x) - D_R(G(x))\|_1 \tag{2}$$

$$\mathcal{L}_{G'} = \mathbb{E}_{x \in S} \|G'(G(x)) - D_S(G'(G(x)))\|_1 \tag{3}$$

$$\mathcal{L}_{D_R} = \mathbb{E}_{x \in S, y \in \mathcal{R}} \|y - D_R(y)\|_1 - k_t^{D_R} \mathcal{L}_G \tag{4}$$

$$\mathcal{L}_{D_S} = \mathbb{E}_{x \in S} \|x - D_S(x)\|_1 - k_t^{D_S} \mathcal{L}_{G'} \tag{5}$$

여기서 균형 항 $k_t^D$는 매 학습 스텝 $t$마다 다음과 같이 업데이트됩니다:

$$k_t^D = k_{t-1}^D + 0.001(0.5\mathcal{L}_D - \mathcal{L}_G)$$

> 이 균형 항은 생성기와 판별기의 학습 균형을 유지하여 훈련을 안정화시킵니다.

---

#### ② 적대적 쌍 매칭 (Adversarial Pair Matching) — 핵심 Semi-supervised 요소

$G'$ 네트워크를 pair-matching discriminator $D_P$로도 활용합니다. 소량의 쌍 데이터 $(\mathcal{P}_S, \mathcal{P}_R)$를 이용해:

$$\mathcal{L}_{D_P} = \mathbb{E}_{s \in \mathcal{P}_S, r \in \mathcal{P}_R} \|s - G'(r)\|_1 - k_t^{D_P} \mathcal{L}_{cyc} \tag{6}$$

이를 통해 생성된 쌍 $(x \in S, G(x) \in \hat{R})$의 상관 분포가 실제 쌍 데이터 $(s \in \mathcal{P}_S, r \in \mathcal{P}_R)$의 분포와 유사해지도록 유도합니다. 즉 $G'$는 다음 두 역할을 동시에 수행합니다:

- **역방향 생성기**: $G': \hat{R} \rightarrow \hat{S}$ (사이클 일관성 유지)
- **쌍 매칭 판별기**: 실제 합성-실제 쌍 분포와 생성된 합성-생성된 쌍 분포를 정렬

---

#### ③ 신원 보존 (Identity Preservation) — Set-based Loss

새로운 신원의 임베딩 일관성을 보존하기 위해 **Center Loss**와 **Pushing Loss**(Magnet Loss의 단순화 버전)를 결합한 Set-based Loss를 제안합니다. 사전 학습된 얼굴 임베딩 네트워크 $C$ 위에서 미니배치 $M$에 대해:

$$\mathcal{L}_C = \mathbb{E}_{x \in S, i_x \in \mathbb{N}^+} \sum_x^M -\log \frac{\exp\!\left(\frac{1}{2\sigma^2}\|C(G(x)) - c_{i_x}\|_2^2 - \eta\right)}{\sum_{j \neq i_x} \exp\!\left(\frac{1}{2\sigma^2}\|C(G(x)) - c_j\|_2^2\right)} \tag{7}$$

- $i_x$: 3DMM 샘플링으로 제공된 신원 레이블
- $c_j$: 신원 $j$의 평균 임베딩(centroid)
- $\eta = 1$: 마진 항
- $\sigma = \sqrt{\frac{\sum_x^M \|C(G(x)) - c_{i_x}\|_2^2}{M-1}}$: 배치 내 분산

**Centroid 업데이트 (모멘텀 방식):**

$$c_j^{t+1} = c_j^t - \beta \cdot \delta(i_x = j)(c_j^t - C(G(x))), \quad \beta = 0.95$$

이 방식은 학습 중 이미지 품질이 변화하더라도 신원 중심이 적응적으로 갱신됩니다. Softmax 레이어는 학습 초기에 빠르게 수렴하여 이후 생성 이미지를 감독하지 못하는 반면, 이 방식은 학습 전체 과정에 걸쳐 유효합니다.

---

#### ④ 전체 목적 함수 (Full Objective)

$$\theta_G = \arg\min_{\theta_G} \mathcal{L}_G + \lambda_{cyc}\mathcal{L}_{cyc} + \lambda_C \mathcal{L}_C \tag{8}$$

$$\theta_{G'} = \arg\min_{\theta_{G'}} \mathcal{L}_{G'} + \lambda_{cyc}\mathcal{L}_{cyc} + \lambda_{D_P} \mathcal{L}_{D_P} \tag{9}$$

$$\theta_{D_R}, \theta_{D_S} = \arg\min_{\theta_{D_R}, \theta_{D_S}} \mathcal{L}_{D_R} + \mathcal{L}_{D_S} \tag{10}$$

하이퍼파라미터 설정: $\lambda_{cyc} = 0.5$, $\lambda_{D_P} = 0.5$, $\lambda_C = 0.001$, $\lambda_{id} = 0.1$

---

### 2.3 모델 구조

```
[3DMM Renderer]
     ↓ x ∈ S (합성 이미지)
     ↓
[G: Generator] ──── skip connections (ResNet, 3 residual blocks)
     ↓ G(x) ∈ R̂ (생성된 포토리얼리스틱 이미지)
     ├──→ [D_R: BEGAN Autoencoder Discriminator] ← y ∈ R (실제 이미지)
     ├──→ [C: FaceNet(NN4) 임베딩 네트워크] → Set-based Identity Loss
     └──→ [G': Inverse Network] ──────────────────────────┐
               ↓ G'(G(x)) ∈ Ŝ                            │
               ├──→ [D_S: BEGAN Autoencoder Discriminator] ← x ∈ S
               └──→ [D_P: Pair Matching] ← (s ∈ P_S, r ∈ P_R) 소량 쌍 데이터
```

| 구성 요소 | 세부 사항 |
|-----------|-----------|
| $G$, $G'$ | Shallow ResNet (3 residual blocks), Skip connections ($G$만), Dropout (keep rate 0.9) |
| $D_R$, $D_S$ | BEGAN 오토인코더 (Wasserstein 거리 기반) |
| $C$ | FaceNet NN4 아키텍처, 입력 $96 \times 96$, Oxford VGG Face로 사전 학습 |
| 이미지 크기 | $108 \times 108$ (학습), $96 \times 96$ (랜덤 크롭) |
| 학습 데이터 | CASIA-WebFace (~500K 실제), 300W-3D + AFLW2000-3D (5K 쌍 데이터), LSFM + Basel FM (합성) |
| 학습 시간 | GTX 1080TI, 약 70시간, 248K iterations, batch size 16 |

---

### 2.4 성능 향상

#### 정성적 평가

- simGAN 대비: 신원별 얼굴 특징 보존 우수
- CycleGAN 대비: 시각적으로 더 자연스러운 이미지 생성
- 포즈·표정·조명 조건부 생성이 가능하며, 극단적 포즈에서는 품질이 다소 저하됨

#### 정량적 평가 (LFW 벤치마크)

| 방법 | Real 데이터 | Synth 데이터 | Acc. (%) | 100% - EER |
|------|------------|--------------|----------|------------|
| FaceNet | 200M | - | 98.87 | - |
| VGG Face | 2.6M | - | 98.95 | 99.13 |
| VGG (100%) 기준선 | 1.8M | - | 94.8 | 94.6 |
| VGG + simGAN | 1.8M | 500K | 94.7 | 94.8 |
| VGG + CycleGAN | 1.8M | 500K | 94.5 | 94.7 |
| **VGG + GANFaces-500K** | **1.8M** | **500K** | **94.9** | **95.1** |
| **VGG + GANFaces-5M** | **1.8M** | **5M** | **95.2** | **95.1** |

- GANFaces 추가 시 모든 실제 데이터 비율(20%, 50%, 100%)에서 LFW 및 IJB-A 성능이 일관되게 향상
- GANFaces의 기여도는 실제 데이터 비율에 반비례 → 실제 데이터가 적을수록 합성 데이터의 효과 극대화

#### 신원 분리 실험 (임베딩 공간)

- 3DMM 합성 이미지보다 GANFaces의 동일 신원 내/간 거리 분포가 더 잘 분리됨
- GANFaces의 분리도가 VGG Face Dataset보다도 우수 (VGG의 노이즈 레이블 문제 영향)

---

### 2.5 한계

1. **낮은 이미지 해상도**: 출력 이미지가 $96 \times 96$으로 당시 최신 방법(FaceNet: $220 \times 220$, VGG Face: $224 \times 224$) 대비 낮음
2. **극단적 포즈 처리 미흡**: 훈련 실제 데이터(CASIA-WebFace)에 극단적 포즈 이미지가 부족하여 생성 품질 저하
3. **3DMM 한계 상속**: 3DMM 자체의 표현력 한계(피부 세부 질감, 헤어, 안경 등)가 생성 이미지에도 반영됨
4. **학습 비용**: 수렴까지 약 70시간 소요
5. **정체성 다양성 한계**: LSFM+Basel FM의 PCA 공간에서 샘플링되므로, 매우 특수한 얼굴 형태(예: 특정 민족 특성)의 다양성이 제한적

---

## 3. 모델 일반화 성능 향상 가능성

### 3.1 실험에서 확인된 일반화 향상 근거

**데이터 증강을 통한 일반화**: 실제 데이터가 적을수록 GANFaces의 기여가 증가합니다 (논문 Fig. 8). 이는 생성 데이터가 모델의 일반화 능력을 실질적으로 향상시킨다는 것을 의미합니다:

$$\text{일반화 향상 효과} \propto \frac{1}{|\text{실제 데이터}|}$$

| VGG 사용 비율 | VGG only LFW Acc. | VGG + GANFaces LFW Acc. | 향상폭 |
|--------------|-------------------|-------------------------|--------|
| 20% | ~88% | ~91% | +3%p |
| 50% | ~92% | ~93% | +1%p |
| 100% | 94.8% | 94.9% | +0.1%p |

### 3.2 일반화 향상의 구조적 메커니즘

#### (a) 3DMM 기반 조건부 다양성 확보

3DMM 파라미터 공간에서 샘플링함으로써, 훈련 데이터에 없는 다양한 포즈·표정·조명 조합의 이미지를 생성합니다. 이는 실제 데이터에서 희귀한 케이스(예: 극단적 측면 포즈)를 커버하여 모델의 in-the-wild 일반화를 도웁니다.

#### (b) Mode Collapse 방지를 통한 분포 다양성

기존 GAN이 노이즈 벡터를 입력으로 받는 것과 달리, **3DMM 이미지를 강한 조건으로** 입력함으로써 모드 붕괴를 구조적으로 억제합니다. 이로 인해 생성 분포가 실제 분포를 더 넓게 커버합니다.

#### (c) Set-based Identity Loss의 적응성

학습 중 이미지 품질이 변화함에 따라 임베딩 공간도 변화하는데, 모멘텀 기반 centroid 업데이트:

$$c_j^{t+1} = c_j^t - \beta \cdot \delta(i_x = j)(c_j^t - C(G(x)))$$

가 이 변화에 적응적으로 대응합니다. 반면 Softmax는 학습 초기에 수렴하여 이후 감독 능력을 잃습니다. 이 adaptive identity supervision이 생성 다양성을 유지하면서도 신원 일관성을 보존하여, **과적합 없이 다양한 신원에 대한 일반화**를 가능하게 합니다.

#### (d) 소량 쌍 데이터의 레버리지 효과

5K 쌍 데이터(300W-3D + AFLW2000-3D)만으로도 두 도메인 간 올바른 매핑을 학습할 수 있습니다. 이는 본 방법이 **데이터 효율적(data-efficient)**임을 시사하며, 새로운 도메인(예: 다른 인종 그룹, 의료 영상 등)에 적용 시 소량의 쌍 데이터만으로도 적응 가능한 잠재력을 갖습니다.

### 3.3 일반화 성능의 한계와 개선 가능성

| 한계 | 개선 방향 |
|------|-----------|
| 낮은 해상도로 인한 세부 특징 손실 | 고해상도 생성 모델(StyleGAN 계열)로 대체 |
| 3DMM의 표현 범위 제한 | 더 표현력 높은 3D 모델(예: FLAME, DECA) 사용 |
| 훈련 데이터 편향(주로 정면/근정면 포즈) | 균형 잡힌 포즈 분포로 샘플링 전략 개선 |
| 단일 종류의 identity loss | ArcFace, CosFace 등 최신 metric learning 손실 적용 |

---

## 4. 앞으로의 연구에 미치는 영향 및 고려 사항

### 4.1 연구 영향

#### (a) 합성 데이터 기반 얼굴인식 연구 방향 제시

본 논문은 **3DMM과 GAN의 결합**이 실제 얼굴인식 성능을 향상시킬 수 있음을 실증적으로 보였습니다. 이후 다수의 연구가 이 방향을 따라 더 고품질의 합성 데이터를 활용한 얼굴인식 연구를 진행합니다.

#### (b) Semi-supervised 이미지 변환의 새로운 패러다임

역매핑 네트워크를 쌍 매칭 판별기로 활용하는 아이디어는, 완전 지도 학습과 완전 비지도 학습 사이의 실용적 균형점을 제시합니다. 이는 레이블 획득이 어렵지만 소수의 쌍 데이터가 존재하는 다양한 도메인 적응 문제에 적용 가능합니다.

#### (c) Set-based Identity 보존 기법의 확산

알려지지 않은 신원에 대한 Set-based Loss는 이후 **few-shot 얼굴인식**, **개인화 얼굴 생성** 연구에서 참조되는 기법이 됩니다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ 주의: 아래 비교는 논문에서 직접 인용된 내용이 아니며, 2020년 이후 연구 동향에 대한 분석입니다. 개별 논문의 세부 수치는 해당 논문을 직접 확인하시기 바랍니다.

#### (a) DiscoFaceGAN (Deng et al., CVPR 2020)

**"Disentangled and Controllable Face Image Generation via 3D Imitative-Contrastive Learning"**

| 비교 항목 | Gecer et al. (2018) | DiscoFaceGAN (2020) |
|-----------|---------------------|---------------------|
| 3D 모델 활용 | 3DMM 입력 → GAN 변환 | 3DMM을 GAN 내부에 직접 통합 |
| 속성 제어 | 포즈·표정·조명 (간접적) | 명시적 파라미터 분리(disentanglement) |
| 신원 보존 | Set-based Loss | Contrastive learning |
| 출력 해상도 | 96×96 | 256×256 |

DiscoFaceGAN은 본 논문의 아이디어를 발전시켜 3DMM 파라미터를 GAN 잠재 공간에 직접 통합함으로써, 더 정밀한 속성 제어를 달성합니다.

#### (b) SynergyNet (Wu et al., 3DV 2021)

**"SynergyNet: Synergy Between 2D and 3D for Face Alignment"**

3DMM 피팅과 2D 얼굴 정렬을 통합한 접근으로, 본 논문에서 3DMM 파라미터 추출에 활용한 아이디어를 더 정교하게 발전시킵니다.

#### (c) FFHQ 기반 StyleGAN 계열 (Karras et al., 2019~2021)

**"Analyzing and Improving the Image Quality of StyleGAN" (StyleGAN2, CVPR 2020)**

| 비교 항목 | Gecer et al. (2018) | StyleGAN2 (2020) |
|-----------|---------------------|------------------|
| 생성 해상도 | 96×96 | 1024×1024 |
| 신원 제어 | 3DMM 조건부 | 잠재 공간 조작 |
| 포즈 제어 | 3DMM 명시적 파라미터 | 암묵적 (제한적) |
| 훈련 데이터 필요량 | 실제+합성 혼합 | 대규모 실제 데이터 |

StyleGAN은 훨씬 높은 품질의 이미지를 생성하지만, **3DMM과 같은 명시적 구조적 제어가 부재**하다는 한계가 있습니다. 이를 보완하기 위해 StyleGAN + 3DMM을 결합하는 후속 연구들이 등장합니다.

#### (d) GAN-based Synthetic Face Dataset 연구 (2020~)

- **SynFace (Qiu et al., ICCV 2021)**: "SynFace: Face Recognition with Synthetic Data"
  - 합성 데이터만으로 얼굴인식을 학습하는 방향 탐구
  - 도메인 격차 해소를 위한 identity mixup 기법 제안
  - 본 논문의 "합성 데이터로 실제 인식 성능 향상" 아이디어를 더 발전

- **DigiFace-1M (Bae et al., WACV 2023)**: "DigiFace-1M: 1 Million Digital Face Images for Face Recognition"
  - Microsoft Research에서 그래픽 엔진으로 생성한 100만 합성 얼굴 이미지로 얼굴인식 학습
  - 본 논문이 GANFaces 데이터셋으로 제시한 방향의 대규모 확장

#### (e) 전반적 발전 동향 비교

```
Gecer et al. 2018          2020~2021               2022~2023
(Semi-sup, 3DMM+GAN)  →  (StyleGAN+3DMM)      →  (Diffusion+3DMM)
  96×96 해상도             256×256~1024×1024        초고해상도
  쌍 데이터 필요             잠재공간 조작             텍스트 조건부 생성
  Set-based Loss           ArcFace 기반             CLIP 기반 제어
```

---

### 4.3 향후 연구 시 고려 사항

#### ① 고해상도 생성으로의 확장

본 논문의 가장 명확한 한계는 $96 \times 96$의 낮은 해상도입니다. 향후 연구에서는:
- **Progressive Growing GAN** 또는 **StyleGAN 아키텍처** 채택
- **Multi-scale discriminator** 활용
- 고해상도에서의 3DMM 조건부 생성 가능성 탐구

#### ② 더 표현력 있는 3D 모델 활용

- **FLAME (Li et al., 2017)**: 머리카락, 눈 등 보다 정밀한 얼굴 모델
- **DECA (Feng et al., 2021)**: 이미지에서 상세 3DMM 피팅
- **Neural Radiance Fields (NeRF) 기반 3D 표현**: 더 풍부한 기하 및 질감 모델링

#### ③ 최신 Metric Learning 손실 적용

Set-based Loss를 다음으로 대체/보완 가능:
- **ArcFace Loss**: $\mathcal{L} = -\log \frac{e^{s(\cos(\theta_{y_i} + m))}}{e^{s(\cos(\theta_{y_i} + m))} + \sum_{j \neq y_i} e^{s\cos\theta_j}}$
- **CosFace**: 코사인 공간에서의 마진 추가
- **Triplet Loss with Hard Mining**: 어려운 샘플 중점 학습

#### ④ 데이터 프라이버시 및 윤리적 고려

실제 인물의 얼굴 데이터(CASIA-WebFace, VGG Face)를 사용하는 것에 대한 **프라이버시 문제**가 2020년 이후 점점 중요해지고 있습니다:
- 완전 합성 데이터만으로 학습하는 방향 탐구 (SynFace, DigiFace-1M)
- GDPR 등 규제 준수를 위한 익명화된 합성 데이터 생성 파이프라인

#### ⑤ Diffusion Model과의 결합

2022년 이후 Diffusion Model이 GAN을 대체하는 추세에서:
- **3DMM 조건부 Diffusion Model**: 더 안정적이고 다양한 이미지 생성
- **ControlNet (Zhang et al., 2023)**: 3DMM 렌더링을 조건으로 한 Diffusion 제어
- Denoising 과정에서의 identity loss 통합 방법 탐구

#### ⑥ 도메인 외 확장 가능성

본 논문의 semi-supervised 프레임워크는 얼굴 이외 도메인에도 적용 가능:
- 의료 영상 (CT/MRI → 실제 외형)
- 자율주행 (시뮬레이터 → 실제 도로 환경)
- 소량의 쌍 데이터만 필요하다는 장점이 핵심

---

## 참고 자료

**주 논문:**
- Gecer, B., Bhattarai, B., Kittler, J., & Kim, T.-K. (2018). "Semi-supervised Adversarial Learning to Generate Photorealistic Face Images of New Identities from 3D Morphable Model." *ECCV 2018*.

**논문 내 주요 인용 문헌:**
- Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A. (2017). "Unpaired image-to-image translation using cycle-consistent adversarial networks." *ICCV 2017.*
- Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A. (2017). "Image-to-image translation with conditional adversarial networks." *CVPR 2017.*
- Berthelot, D., Schumm, T., & Metz, L. (2017). "BEGAN: Boundary equilibrium generative adversarial networks." *arXiv:1703.10717.*
- Shrivastava, A., et al. (2017). "Learning from simulated and unsupervised images through adversarial training." *CVPR 2017.* (simGAN)
- Wen, Y., Zhang, K., Li, Z., & Qiao, Y. (2016). "A discriminative feature learning approach for deep face recognition." *ECCV 2016.* (Center Loss)
- Blanz, V., & Vetter, T. (1999). "A morphable model for the synthesis of 3D faces." *SIGGRAPH 1999.*
- Schroff, F., Kalenichenko, D., & Philbin, J. (2015). "FaceNet: A unified embedding for face recognition and clustering." *CVPR 2015.*
- Parkhi, O. M., Vedaldi, A., & Zisserman, A. (2015). "Deep face recognition." *BMVC 2015.*

**2020년 이후 비교 연구 (개별 논문 확인 권장):**
- Deng, Y., et al. (2020). "Disentangled and Controllable Face Image Generation via 3D Imitative-Contrastive Learning." *CVPR 2020.*
- Karras, T., et al. (2020). "Analyzing and Improving the Image Quality of StyleGAN." *CVPR 2020.* (StyleGAN2)
- Qiu, H., et al. (2021). "SynFace: Face Recognition with Synthetic Data." *ICCV 2021.*
- Bae, G., et al. (2023). "DigiFace-1M: 1 Million Digital Face Images for Face Recognition." *WACV 2023.*
