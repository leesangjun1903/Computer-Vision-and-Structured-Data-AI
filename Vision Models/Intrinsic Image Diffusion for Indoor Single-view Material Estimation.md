# Intrinsic Image Diffusion for Indoor Single-view Material Estimation

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

본 논문은 단일 실내 이미지에서 재질(albedo, roughness, metallic)을 추정하는 문제를 **결정론적(deterministic) 문제가 아닌 확률론적(probabilistic) 문제**로 재정의해야 한다고 주장합니다. 조명과 재질의 분리는 본질적으로 모호하기 때문에, 단 하나의 해를 예측하는 기존 방식은 해 공간의 평균값으로 수렴하여 흐릿하고 세부 정보가 없는 결과를 낳는다고 비판합니다.

### 주요 기여

1. **확률론적 외관 분해(Probabilistic Appearance Decomposition)**: 확산 모델(diffusion model)을 활용하여 가능한 재질 해들을 샘플링하는 프레임워크 제안
2. **실세계 이미지 사전지식(Real-world Image Prior) 활용**: Stable Diffusion V2를 파인튜닝하여 합성-실세계 도메인 갭(domain gap) 완화 → albedo 예측에서 **77.6% FID 개선, 4.04dB PSNR 향상**
3. **조명 최적화(Lighting Optimization)**: 고품질 재질 예측을 기반으로 Spherical Gaussian 기반 전역 조명과 복수의 포인트 광원을 최적화

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

#### 핵심 문제: 외관 분해의 모호성

단일 이미지에서 관찰되는 시각적 외관(appearance)은 다음의 복잡한 상호작용의 결과입니다:

$$I = f(\text{material}, \text{lighting}, \text{geometry})$$

여기서 핵심적인 모호성은:

- **스케일 모호성**: 재질을 임의의 전역 스칼라로 스케일링하고, 조명을 역으로 스케일링해도 동일한 이미지를 생성 가능
- **베이크인(Baked-in) 조명**: 그림자, 반사광(specular highlight) 등이 재질 맵에 혼합될 수 있음
- **실세계 데이터 부족**: 대규모 실세계 진실값(ground truth) 재질 데이터셋 부재

#### 기존 방법의 한계

기존 결정론적 모델들(Li et al. [27], Zhu et al. [48])은 해 공간의 국소 또는 전역 평균을 예측하는 경향이 있어, 결과물이 흐릿하고(blurry) 조명이 재질에 혼합되는 문제가 발생합니다.

---

### 2.2 제안 방법 (수식 포함)

#### 2.2.1 문제 정식화

입력 이미지 $x \in \mathbb{R}^{H \times W \times 3}$가 주어졌을 때, 재질 속성 $\hat{m} \in \mathbb{R}^{H \times W \times 5}$ (albedo 3채널 + roughness/metallic 2채널)를 조건부 분포에서 샘플링합니다:

$$\hat{m} \sim q(m|x)$$

#### 2.2.2 학습 목적함수

Latent Diffusion Model의 노이즈 예측 손실을 사용합니다:

$$\mathcal{L} = \mathbb{E}_{m, \epsilon \sim \mathcal{N}(0, I), t} \left[ ||\epsilon - \epsilon_\theta(\mathcal{E}(m) + \epsilon, t, x)||_2^2 \right] $$

여기서:
- $\mathcal{E}$: 사전 훈련된 고정 인코더
- $\epsilon_\theta$: 학습 가능한 노이즈 예측 네트워크
- $t \sim [1, 1000]$: 타임스텝
- $x$: 조건부 입력 이미지

#### 2.2.3 재질 표현

GGX 마이크로패싯 BRDF(GGX Microfacet BRDF) [Walter et al., 2007]를 사용하여 재질을 표현합니다:
- **Albedo**: 기반 색상 (3채널)
- **BRDF Map**: R채널 = roughness, G채널 = metallic, B채널 = 0 (2채널로 유효 사용)

알베도와 BRDF 속성을 **별도로 인코딩** 후 잠재 공간에서 연결(concatenate):

$$z_{material} = [z_{albedo}; z_{BRDF}]$$

#### 2.2.4 조명 최적화 손실함수

재질 예측 후 조명 최적화는 다음 손실을 사용합니다:

$$L_{pos} = \sum_{i}^{N_{light}} 1/d_{i,near}$$

$$L_{val} = \sum_{i}^{N_{light}} \sum_{j}^{N_{sg}} w_{i,j}$$

$$L = L_{rec} + \lambda_{pos} L_{pos} + \lambda_{val} L_{val} $$

여기서:
- $L_{rec}$: L2 이미지 재구성 손실
- $d_{i,near}$: $i$번째 광원에서 가장 가까운 표면까지의 거리
- $w_{i,j}$: $i$번째 광원의 $j$번째 Spherical Gaussian 가중치

---

### 2.3 모델 구조

```
입력 이미지 x
    │
    ├──→ 훈련 가능한 인코더 E* (랜덤 초기화, 3채널 출력)
    │         │
    │         ↓
    │    Conditioning Features
    │
    ├──→ CLIP 이미지 임베딩 (Cross-Attention 조건부)
    │
    ↓
[GT Albedo + BRDF] → 고정 인코더 E → Material Features
                                              │
                                      + Gaussian Noise
                                              │
                                              ↓
                                    MaterialDiffusion (UNet)
                                    (SD V2 파인튜닝)
                                              │
                                    L2 Loss (예측 노이즈 vs 원본 노이즈)
                                              │
                                              ↓
                              분리 디코딩: Albedo | BRDF
```

#### 핵심 구조적 선택

| 설계 선택 | 이유 |
|-----------|------|
| 랜덤 초기화 인코더 E* | 재질 추정은 이미지 재구성과 다른 특징 집합이 필요함 |
| CLIP 임베딩 Cross-Attention | 시맨틱/지각 정보 활용 |
| 분리 인코딩/디코딩 | Albedo와 BRDF의 분리 예측 가능 |
| 11채널 입력 | 4채널(잠재) + 3채널(E* 출력) + 4채널(노이즈) |

---

### 2.4 성능 향상 및 한계

#### 성능 향상 (정량적)

**합성 데이터(InteriorVerse) 평가:**

| 방법 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ |
|------|--------|--------|---------|-------|
| IIW [4] | 9.73 | 0.62 | 0.47 | 62.22 |
| Li et al. [27] | 12.31 | 0.68 | 0.52 | 77.79 |
| Zhu et al. [48] | 15.92 | 0.78 | 0.34 | 46.21 |
| **Ours - Mean** | **17.42** | **0.80** | **0.22** | **25.42** |
| Ours - Best | 18.43 | 0.77 | 0.26 | — |

- FID 기준 [48] 대비 **44.99% 향상**
- PSNR 기준 [48] 대비 **+1.5dB 향상**

**실세계 데이터(IIW) 평가:**

| 방법 | WHDR ↓ | Perceptual Quality ↑ |
|------|--------|----------------------|
| IIW [4] | 21.00 | 9.63% |
| Li et al. [27] | 21.99 | 0.46% |
| Zhu et al. [48] | 22.90 | 6.11% |
| **Ours - Mean** | 22.02 | **83.80%** |

**이미지 재구성 평가:**

| 방법 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| Zhu et al. [48] | 13.54 | 0.51 | 0.43 |
| [48] w/ Ours | 14.07 | 0.55 | 0.38 |
| **Ours Full** | **21.96** | **0.70** | **0.22** |

#### 한계

1. **합성 데이터 의존**: 합성 데이터(InteriorVerse)와 실세계 사전지식의 조합에 의존하며, 실세계 감독 신호 없음
2. **느린 추론 속도**: DDIM 50스텝으로 10샘플 생성에 약 17초 소요 (결정론적 방법 대비 느림)
3. **독립적 조명 최적화**: 재질 추정과 조명 추정이 결합되지 않고 순차적으로 처리됨
4. **WHDR 지표의 한계**: 고주파 세부 정보를 포착하지 못하는 지표로 인한 성능 과소평가 가능성 (논문에서도 지적)
5. **Occlusion 미처리**: 조명 최적화 시 폐색(occlusion) 효과를 고려하지 않음
6. **훈련 비용**: 4개 A6000 GPU로 약 6일 소요

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문에서 일반화(generalization)는 **합성→실세계 도메인 갭** 해소의 관점에서 핵심적으로 다뤄집니다.

### 3.1 일반화를 위한 핵심 전략: "역방향 파인튜닝"

기존 방법들은 다음의 순서를 따릅니다:

$$\text{합성 데이터 훈련} \rightarrow \text{실세계 데이터(IIW) 파인튜닝}$$

반면 본 논문은:

$$\text{대규모 실세계 이미지로 사전훈련된 SD V2} \rightarrow \text{합성 데이터(InteriorVerse)로 파인튜닝}$$

이 전략의 핵심은 **실세계 이미지에서 학습한 시맨틱/지각 정보를 재질 추정에 전이**하는 것입니다.

### 3.2 사전훈련 효과의 정량적 검증 (Ablation Study)

| 방법 | PSNR(IV) ↑ | FID(IV) ↓ | WHDR(IIW) ↓ |
|------|-----------|-----------|-------------|
| Pix2Pix [17] | 13.69 | 84.28 | 36.42 |
| Ours w/o PT - Mean | 13.38 | 113.31 | 35.60 |
| **Ours - Mean** | **17.42** | **25.42** | **22.02** |

사전훈련 없는 모델(Ours w/o PT)은 합성 데이터에서도 크게 뒤떨어지는 성능을 보입니다. 이는 **사전훈련된 실세계 이미지 prior가 합성 데이터 시나리오에서도 유익**하다는 것을 보여줍니다.

### 3.3 일반화 향상의 메커니즘

1. **시맨틱 정보 활용**: Stable Diffusion이 LAION-5B로 학습하면서 얻은 물체 카테고리, 재질 유형에 대한 지식이 재질 추정에 자연스럽게 전이됨
2. **지각적 일관성**: 실세계 이미지 prior는 물리적으로 그럴듯한(plausible) 재질 분포를 제공하여 합성 데이터에서 학습한 모델보다 실세계에 더 잘 일반화됨
3. **확률론적 샘플링**: 단일 해 대신 다수의 후보를 제시하므로, 실세계의 다양한 조건에 더 유연하게 대응 가능

### 3.4 심도(Depth) 조건부 모델의 일반화 가능성

추가 실험(Appendix E)에서 기하 정보를 추가 조건으로 활용할 때의 효과를 검증합니다:

| 방법 | PSNR(IV) ↑ | FID(IV) ↓ | WHDR(IIW) ↓ |
|------|-----------|-----------|-------------|
| Image-Only | 17.42 | 25.42 | 22.02 |
| GT Depth | 16.57 | 24.36 | 17.05 |
| Pred Depth | **18.31** | **22.60** | **16.66** |

OmniData [11]로 예측한 깊이 정보를 조건으로 추가하면 실세계 일반화(WHDR)가 크게 향상됩니다. 이는 향후 기하 정보와 재질 추정을 결합하는 방향이 일반화에 유망함을 시사합니다.

---

## 4. 최신 관련 연구 비교 분석 (2020년 이후)

### 4.1 직접 비교 대상 방법

| 방법 | 연도 | 접근법 | 특징 |
|------|------|--------|------|
| Li et al. [27] (CVPR 2020) | 2020 | UNet 결정론적 | SVBRDF + 공간적으로 변화하는 조명 |
| Zhu et al. [48] (SIGGRAPH Asia 2022) | 2022 | UNet 결정론적 | 차분 몬테카를로 레이트레이싱, InteriorVerse |
| Zhu et al. [50] / IRISFormer (CVPR 2022) | 2022 | Vision Transformer | 밀집 예측, Self-Attention |
| **Ours (IID)** | **2024** | **조건부 확산 모델** | **확률론적, SD V2 파인튜닝** |

### 4.2 광범위한 관련 연구 흐름

#### (1) 결정론적 역렌더링 방법

- **Wang et al. [43] (ICCV 2021)**: 3D 공간적으로 변화하는 조명을 포함한 실내 역렌더링
- **Li et al. [28] (ECCV 2022)**: 단일 이미지에서 실내 장면 조명의 물리 기반 편집
- **Zhu et al. [49] (2023)**: I²-SDF, Neural SDF를 이용한 실내 장면 재구성 및 편집

이 방법들의 공통 한계: **결정론적** 예측 → 해 공간의 평균화 → 흐릿한 결과

#### (2) 확산 모델 기반 관련 연구

| 방법 | 연도 | 특징 | IID와의 차이 |
|------|------|------|-------------|
| Zero-1-to-3 [29] | 2023 | SD를 파인튜닝하여 3D 생성 | 이미지 공간 타겟, 재질 추정 아님 |
| ControlNet [47] | 2023 | 조건부 제어 추가 | 이미지 생성, 재질 추정 아님 |
| StylitGAN [5] | 2023 | 재조명(relighting) | 조명만 다루고 재질 추정 아님 |
| Lee et al. [23] | 2023 | 확산 prior의 픽셀 수준 예측 | 결정론적 사용 |
| UMAT [35] (CVPR 2023) | 2023 | 불확실성 인식 단일 물체 재질 캡처 | 단일 물체, IID는 다중 물체 실내 장면 |

#### (3) 핵심 차별점

```
기존 역렌더링: 합성 훈련 → 실세계 파인튜닝 (데이터 의존)
IID:           실세계 사전훈련 → 합성 파인튜닝 (Prior 활용)

기존 방법: 단일 결정론적 출력
IID:      확률론적 다중 샘플 + 불확실성 추정
```

---

## 5. 향후 연구에 미치는 영향과 고려사항

### 5.1 향후 연구에 미치는 영향

#### (1) 확률론적 역렌더링 패러다임의 확립

이 논문은 역렌더링(inverse rendering) 분야에서 **확률론적 접근법의 필요성**을 명확히 제시하고 실증했습니다. 기존의 "정확한 단일 해를 예측하라"는 패러다임에서 "가능한 해의 분포를 학습하라"는 패러다임으로의 전환을 촉진할 것입니다.

#### (2) 생성 모델의 역렌더링 적용 가능성 증명

Stable Diffusion의 강력한 사전지식을 **이미지 공간이 아닌 재질 공간**에 적용할 수 있음을 증명했습니다. 이는 향후 다음과 같은 연구들에 영향을 줄 것입니다:
- 기하(geometry), 법선(normal), 깊이(depth) 예측에 확산 모델 적용
- 다른 도메인(의료 영상, 위성 영상 등)에서 유사한 전이 학습 전략 활용

#### (3) 불확실성 추정의 활용 가능성

예측된 분산(variance) 맵을 **불확실성 맵**으로 해석하여, 후처리나 사용자 상호작용에 활용하는 연구가 활성화될 것입니다.

### 5.2 향후 연구 시 고려해야 할 점

#### (1) 약지도 학습(Weakly Supervised Learning)으로의 확장

논문 저자들도 한계로 지적하였듯, 현재는 **합성 데이터의 진실값에 의존**합니다. 실세계 이미지에 대한 약지도 학습(multi-view consistency, photometric consistency 등)을 통한 확장이 중요한 연구 방향입니다.

#### (2) 엔드투엔드 역렌더링 프레임워크

현재 재질 추정과 조명 최적화가 **순차적으로** 처리됩니다. 두 과정을 통합한 엔드투엔드 역렌더링이 성능을 더욱 향상시킬 수 있습니다:

$$\hat{m}, \hat{l} \sim q(m, l | x)$$

즉, 재질과 조명을 **동시에 확률론적으로** 추정하는 접근법이 필요합니다.

#### (3) 평가 지표 개선

WHDR 지표가 고주파 세부 정보를 제대로 평가하지 못한다는 점이 이 논문에서 명확히 드러났습니다. 향후 연구에서는:
- 지각적 품질(perceptual quality)과 물리적 정확도를 동시에 측정하는 지표 개발
- 사용자 연구(user study)를 표준 평가 프로토콜로 도입
- 하위 태스크(downstream task) 성능(예: 재조명, AR 합성 품질)으로 평가

#### (4) 추론 효율성

DDIM 50스텝 기반 추론은 **실시간 응용에 부적합**합니다. 일관성 모델(Consistency Models), Flow Matching 등 최신 빠른 생성 기법을 활용하여 추론 속도를 개선하는 연구가 필요합니다.

#### (5) 기하 정보 통합

Appendix의 Depth-conditioning 실험에서 기하 정보가 일반화 성능을 유의미하게 향상시킴을 확인했습니다. 특히 실세계에서 WHDR이 22.02 → 16.66으로 크게 향상되므로, **기하 추정과 재질 추정의 공동 학습**이 중요한 연구 방향입니다.

#### (6) 텍스트 가이드 재질 편집

논문에서 언급된 바와 같이, 생성 모델의 텍스트 조건부 기능을 활용한 **텍스트 가이드 재질 편집**은 매우 유망한 응용 방향입니다. "소파를 가죽 재질로", "바닥을 대리석으로" 등의 편집이 가능해질 수 있습니다.

#### (7) 다양한 장면 유형으로의 확장

현재 모델은 **실내 장면**에만 특화되어 있습니다. 실외 장면, 얼굴, 자동차 등 다른 도메인으로 확장하기 위한 범용 확산 기반 역렌더링 프레임워크 연구가 필요합니다.

---

## 참고 자료

**주요 논문 (직접 분석 대상):**
- Kocsis, P., Sitzmann, V., & Nießner, M. (2024). *Intrinsic Image Diffusion for Indoor Single-view Material Estimation*. arXiv:2312.12274v2.

**논문 내 인용 핵심 참고문헌:**
- Rombach, R. et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. (Stable Diffusion V2) arXiv:2112.10752
- Zhu, J. et al. (2022). *Learning-based Inverse Rendering of Complex Indoor Scenes with Differentiable Monte Carlo Raytracing*. SIGGRAPH Asia 2022. (InteriorVerse 데이터셋 및 베이스라인 [48])
- Li, Z. et al. (2020). *Inverse Rendering for Complex Indoor Scenes*. CVPR 2020. (베이스라인 [27])
- Bell, S. et al. (2014). *Intrinsic Images in the Wild*. ACM TOG. (IIW 데이터셋 [4])
- Ho, J. et al. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS 2020. ([15])
- Liu, R. et al. (2023). *Zero-1-to-3*. arXiv:2303.11328. ([29])
- Walter, B. et al. (2007). *Microfacet Models for Refraction through Rough Surfaces*. EGSR 2007. (GGX BRDF [42])
- Yeshwanth, C. et al. (2023). *ScanNet++*. ICCV 2023. ([46])
- Rodríguez-Pardo, C. et al. (2023). *UMAT*. CVPR 2023. ([35])
- Zhu, R. et al. (2022). *IRISFormer*. CVPR 2022. ([50])
- Zhang, L. & Agrawala, M. (2023). *ControlNet*. arXiv:2302.05543. ([47])

**프로젝트 페이지:** [peter-kocsis.github.io/IntrinsicImageDiffusion/](https://peter-kocsis.github.io/IntrinsicImageDiffusion/)
