# Noise Robust Generative Adversarial Networks (NR-GAN)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
기존 GAN은 노이즈가 포함된 학습 이미지를 그대로 충실히 재현하려는 **기억화(memorization)** 문제를 가진다. 본 논문은 **노이즈에 대한 완전한 사전 지식 없이도** 클린 이미지 생성기를 학습할 수 있는 NR-GAN(Noise Robust GAN)을 제안한다.

### 주요 기여

| 기여 | 내용 |
|------|------|
| 새로운 문제 정의 | Noise Robust Image Generation 문제를 공식 정의 |
| 새로운 모델 패밀리 | 5종의 NR-GAN 변형 제안 (SI-NR-GAN × 2, SD-NR-GAN × 3) |
| 광범위한 실험 | CIFAR-10에서 152가지 노이즈 조건 실험 |
| 응용 확장 | GN2GC(GeneratedNoise2GeneratedClean)를 통한 이미지 디노이징 적용 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**문제 설정:**

관측 가능한 노이즈 이미지 $\boldsymbol{y}$, 클린 이미지 $\boldsymbol{x}$, 노이즈 $\boldsymbol{n}$에 대해 다음과 같이 정의한다:

$$\boldsymbol{y} = \boldsymbol{x} + \boldsymbol{n}, \quad \boldsymbol{y}, \boldsymbol{x}, \boldsymbol{n} \in \mathbb{R}^{H \times W \times C}$$

- **표준 GAN의 문제**: $p^g(\boldsymbol{y}) = p^r(\boldsymbol{y})$를 학습 → 노이즈까지 복제
- **목표**: $\boldsymbol{y}^r \sim p^r(\boldsymbol{y})$만으로 $p^g(\boldsymbol{x}) = p^r(\boldsymbol{x})$ 달성

**처리 노이즈 유형 (Figure 2 기준 16종):**

```
신호 독립적 노이즈: (A)~(H)  - Gaussian, Local Gaussian, Uniform, Mixture, Brown Gaussian 등
신호 의존적 노이즈: (I)~(P)  - Multiplicative Gaussian, Poisson, Poisson-Gaussian 등
```

---

### 2.2 제안하는 방법 (수식 포함)

#### 기준선: AmbientGAN

$$\min_{G_x} \max_{D_y} \mathbb{E}_{\boldsymbol{y}^r \sim p^r(\boldsymbol{y})}[\log D_y(\boldsymbol{y}^r)] + \mathbb{E}_{\boldsymbol{z}_x \sim p(\boldsymbol{z}_x), \boldsymbol{\theta} \sim p(\boldsymbol{\theta})}[\log(1 - D_y(F_{\boldsymbol{\theta}}(G_x(\boldsymbol{z}_x))))] \tag{1}$$

- $F_{\boldsymbol{\theta}}$: 사전 정의된 노이즈 시뮬레이션 모델 (노이즈 분포 유형, 양, 신호-노이즈 관계 모두 필요)
- **한계**: 완전한 노이즈 사전 지식 필요

#### 기본 두 생성기 모델

$$\min_{G_x, G_n} \max_{D_y} \mathbb{E}_{\boldsymbol{y}^r \sim p^r(\boldsymbol{y})}[\log D_y(\boldsymbol{y}^r)] + \mathbb{E}_{\boldsymbol{z}_x \sim p(\boldsymbol{z}_x), \boldsymbol{z}_n \sim p(\boldsymbol{z}_n)}[\log(1 - D_y(G_x(\boldsymbol{z}_x) + G_n(\boldsymbol{z}_n)))] \tag{2}$$

- **문제**: 제약 없이는 $G_x$와 $G_n$이 각각 이미지와 노이즈를 따로 학습할 인센티브가 없음

---

### 2.3 모델 구조 상세

#### ① SI-NR-GAN-I (신호 독립적, 분포 타입 알려진 경우)

**가정 (Assumption 1)**:
- 노이즈 $\boldsymbol{n}$은 신호 $\boldsymbol{x}$가 주어졌을 때 픽셀 단위 독립
- 노이즈 분포 유형(예: Gaussian)은 사전에 알려짐
- 노이즈 양(표준편차)은 미지수

**수식:**

$$\boldsymbol{y} = \boldsymbol{x} + \boldsymbol{n}, \quad \boldsymbol{n} \sim \mathcal{N}(\mathbf{0}, \text{diag}(\boldsymbol{\sigma})^2) \tag{3}$$

**구현:**
- $\boldsymbol{\sigma} = G_n(\boldsymbol{z}_n)$ (픽셀 단위 표준편차를 학습)
- 보조 변수 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{I})$ 도입
- 재파라미터화 트릭: $\boldsymbol{n} = \boldsymbol{\sigma} \cdot \boldsymbol{\epsilon}$

**적용 가능**: (A)~(D) (Gaussian 계열 고정/가변 노이즈)

---

#### ② SI-NR-GAN-II (신호 독립적, 분포 타입 미지수)

**가정 (Assumption 2)**:
- 노이즈 $\boldsymbol{n}$은 회전, 채널 셔플, 색상 반전에 **불변(invariant)**
- 신호 $\boldsymbol{x}$는 이러한 변환에 **변하는(variant)** 특성을 가짐

**수식:**

$$\boldsymbol{n} = T(\hat{\boldsymbol{n}}), \quad \hat{\boldsymbol{n}} = G_n(\boldsymbol{z}_n)$$

변환 $T$의 종류:
1. **회전**: $d \in \{0°, 90°, 180°, 270°\}$ 중 무작위 선택
2. **채널 셔플**: RGB 채널을 무작위 순서 변환
3. **색상 반전**: 채널별 색상 무작위 반전

**적용 가능**: (A)~(H) (픽셀 상관 노이즈 포함)

---

#### ③ SD-NR-GAN-I (신호 의존적, 관계 알려진 경우)

**Multiplicative Gaussian Noise:**

$$\boldsymbol{y} = \boldsymbol{x} + \boldsymbol{n}, \quad \boldsymbol{n} \sim \mathcal{N}(\mathbf{0}, \text{diag}(\boldsymbol{\sigma} \cdot \boldsymbol{x})^2) \tag{4}$$

신호-노이즈 관계 함수: $R(\boldsymbol{x}, \boldsymbol{\sigma}) = \boldsymbol{\sigma} \cdot \boldsymbol{x} = \hat{\boldsymbol{\sigma}}$

**Poisson Noise (Gaussian 근사):**

$$\boldsymbol{y} = \boldsymbol{x} + \boldsymbol{n}, \quad \boldsymbol{n} \sim \mathcal{N}\!\left(\mathbf{0}, \text{diag}\!\left(\boldsymbol{\sigma} \cdot \sqrt{\boldsymbol{x}}\right)^2\right), \quad \boldsymbol{\sigma} = \sqrt{1/\lambda} \tag{5}$$

관계 함수: $R(\boldsymbol{x}, \boldsymbol{\sigma}) = \boldsymbol{\sigma} \cdot \sqrt{\boldsymbol{x}} = \hat{\boldsymbol{\sigma}}$

---

#### ④ SD-NR-GAN-II (신호 의존적, 관계 미지수)

$$\boldsymbol{\sigma} = G_n(\boldsymbol{z}_n, \boldsymbol{z}_x), \quad \boldsymbol{n} \sim \mathcal{N}(\mathbf{0}, \text{diag}(\boldsymbol{\sigma})^2)$$

- $\boldsymbol{z}_x$를 $G_n$의 입력으로 포함 → 신호-노이즈 관계를 **암묵적으로** 학습
- 적용 가능: (A)~(D), (I)~(P) 모두 포함

---

#### ⑤ SD-NR-GAN-III (신호 의존적, 가장 약한 가정)

$$\hat{\boldsymbol{n}} = G_n(\boldsymbol{z}_n, \boldsymbol{z}_x), \quad \boldsymbol{n} = T(\hat{\boldsymbol{n}})$$

- 변환 $T$: **색상 반전만** 사용 (회전·채널 셔플은 신호-노이즈 픽셀 단위 의존성 파괴)
- 적용 가능: 모든 (A)~(P)

---

**모델 비교 요약:**

| 모델 | 노이즈 분포 유형 | 신호-노이즈 관계 | 노이즈 양 | 적용 가능 노이즈 |
|------|:---:|:---:|:---:|:---:|
| AmbientGAN | Known | Known | Known | 특정 모델링만 |
| SI-NR-GAN-I | Known | — | Unknown | (A)~(D) |
| SI-NR-GAN-II | Unknown | — | Unknown | (A)~(H) |
| SD-NR-GAN-I | Known | Known | Unknown | (I),(J)/(M),(N) |
| SD-NR-GAN-II | Known | Unknown | Unknown | (A)~(D),(I)~(P) |
| SD-NR-GAN-III | Unknown | Unknown | Unknown | (A)~(P) 전체 |

---

### 2.4 실용적 기법

#### 수렴 속도 차이 완화
$G_n$이 $G_x$보다 빨리 수렴 → 초기 단계 모드 붕괴 발생
→ **다양성 민감 정규화(diversity-sensitive regularization)**를 $G_n$에 적용

#### 근사 성능 저하 완화
Poisson 노이즈의 Gaussian 근사 시 이산화 갭 발생
→ $\boldsymbol{x}$에 **수직·수평 블러 필터(anti-alias filter)** 적용 후 $D_y$에 제공

---

### 2.5 성능 및 한계

#### 성능 (FID, 낮을수록 좋음)

**CIFAR-10 주요 결과 (Table 1 기준):**

| 모델 | (A) AGF | (G) BG | (I) MGF |
|------|:---:|:---:|:---:|
| GAN (baseline) | 145.8 | 165.3 | 82.7 |
| AmbientGAN† | 26.7 | 30.3 | 21.4 |
| SI-NR-GAN-I | **26.7** | 163.4 | — |
| SI-NR-GAN-II | 29.8 | **32.2** | — |
| SD-NR-GAN-II | — | — | **24.4** |

> †: 정답 노이즈 모델 제공 (비교 불공정)

**핵심 발견:**
1. NR-GAN 최고 성능은 AmbientGAN과 경쟁적 (최대 FID 차 3.3)
2. NR-GAN은 denoiser+GAN 방식 대부분을 능가
3. 픽셀 상관 노이즈 (G)(H)에서 SI-NR-GAN-II가 압도적 우위 (FID 차 >100)

**이미지 디노이징 응용 (GN2GC, PSNR 기준, Table 3):**

| 방법 | LSUN-BG (G) | FFHQ-AGF (A) |
|------|:---:|:---:|
| N2C♯ (상한선) | 29.67 | 31.93 |
| N2N‡ | 28.76 | 31.33 |
| N2V | 20.73 | 30.95 |
| **GN2GC** | **26.61** | **31.34** |

→ GN2GC는 N2V를 능가하고 N2N과 동등한 수준 달성

#### 한계

1. **복잡한 데이터셋에서의 어려움**: LSUN BEDROOM, FFHQ에서 약한 제약의 SD-NR-GAN-III는 학습 어려움 (특히 Poisson 노이즈)
2. **가정의 한계**: 신호-노이즈 의존성이 복잡할수록 노이즈 분리 어려움
3. **GN2GC의 전제 조건**: GAN 사전 학습 필요 → 응용 범위 제한
4. **가정 위반 시 성능 저하**: SI-NR-GAN-I는 Gaussian 가정을 벗어난 노이즈(E)(G)(H)에서 성능 급락

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 데이터 다양성 확보를 통한 일반화

**핵심 메커니즘**: NR-GAN은 노이즈가 포함된 학습 데이터에서 **클린 이미지 분포**를 직접 추정한다. 이를 통해 다음과 같은 일반화 이점을 제공한다:

$$p^g(\boldsymbol{x}) \rightarrow p^r(\boldsymbol{x}) \quad \text{(노이즈 이미지만으로 클린 분포 추정)}$$

- 실세계 데이터(전자 노이즈, 렌더링 분산 등)에 직접 적용 가능
- 노이즈 조건이 달라도 **동일 모델 재사용** 가능 (특히 SI-NR-GAN-II, SD-NR-GAN-III)

### 3.2 변환 불변성을 통한 일반화

SI-NR-GAN-II에서 사용하는 변환 제약은 **자기지도 학습(self-supervised learning)**에서 영감을 받았다. 이 변환은 노이즈의 본질적 특성(회전/채널/색상 불변)을 활용하여:

- 특정 노이즈 분포 유형을 몰라도 적용 가능
- 다양한 노이즈가 혼합된 실제 환경에서도 동작 (Figure 2의 (F) MIX 노이즈)
- 동일 모델로 Gaussian 노이즈(A)와 Brown Gaussian 노이즈(G) 모두 처리 (Figure 1 참조)

### 3.3 암묵적 노이즈 모델 학습

SD-NR-GAN-II의 경우:

$$\boldsymbol{\sigma} = G_n(\boldsymbol{z}_n, \boldsymbol{z}_x), \quad R(\boldsymbol{x}, \boldsymbol{\sigma}_d, \boldsymbol{\sigma}_i) = \boldsymbol{\sigma}_d \cdot \boldsymbol{x} + \boldsymbol{\sigma}_i$$

신호-노이즈 관계 함수 $R$을 데이터로부터 학습함으로써:
- 신호 독립적 노이즈와 의존적 노이즈의 **조합**(K)(L)(O)(P)도 처리 가능
- 새로운 노이즈 유형에 대한 **제로샷(zero-shot) 적응** 가능성 제시

### 3.4 GN2GC를 통한 디노이저 일반화

```
NR-GAN 학습 (noisy images) → 합성 (clean, noisy) 쌍 생성 → 디노이저 학습
```

이 파이프라인은:
- 클린 이미지 없이도 강력한 디노이저 학습 가능
- N2V, N2S 같은 자기지도 방법보다 우수한 성능 (Table 3)
- GAN의 발전에 따라 디노이저 성능도 향상되는 **시너지 효과**

---

## 4. 앞으로의 연구 영향 및 고려사항

### 4.1 연구에 미치는 영향

#### (1) 실세계 GAN 학습의 새로운 패러다임
실세계 이미지 수집 시 노이즈 제거를 위한 전처리 없이도 직접 GAN 학습이 가능해져, **데이터 수집 파이프라인의 단순화**가 기대된다.

#### (2) 노이즈 모델링의 암묵적 학습 방향
AmbientGAN처럼 노이즈를 명시적으로 모델링하는 대신, **데이터 기반 노이즈 분리**의 연구 방향을 제시한다. 이는 후속 연구들의 핵심 방법론이 되었다.

#### (3) 생성 모델의 데이터 품질 의존성 해소
BigGAN, StyleGAN 등 고성능 GAN도 노이즈에 취약한 문제를 구체적으로 제시하고 해결책을 제안하여, **노이즈 강건성을 GAN 연구의 핵심 주제**로 부각시켰다.

#### (4) 자기지도 디노이징과의 연결
GN2GC를 통해 생성 모델과 디노이징의 **선순환 관계(chicken-and-egg)**를 실증적으로 탐구하여, 두 분야의 협력 연구 방향을 제시하였다.

---

### 4.2 2020년 이후 관련 최신 연구 비교 분석

> **⚠️ 주의**: 아래 내용 중 제가 제공받은 논문 PDF에 직접 언급되지 않은 2020년 이후 논문들에 대해서는, 제 학습 데이터 내의 지식을 바탕으로 설명하며, 일부 세부 사항의 정확성에 불확실성이 있을 수 있습니다. 확인 가능한 논문명과 저자를 명시합니다.

#### (1) Diffusion Models의 등장과 비교

확산 모델(DDPM, Score-based Models)은 노이즈 추가/제거 과정 자체를 학습 메커니즘으로 활용한다:

$$q(\boldsymbol{x}_t | \boldsymbol{x}_{t-1}) = \mathcal{N}(\boldsymbol{x}_t; \sqrt{1-\beta_t}\boldsymbol{x}_{t-1}, \beta_t \boldsymbol{I})$$

| 관점 | NR-GAN | Diffusion Models |
|------|--------|-----------------|
| 노이즈 처리 | 노이즈를 제거하여 클린 이미지 학습 | 노이즈를 학습 메커니즘으로 활용 |
| 학습 데이터 | 노이즈 이미지만 필요 | 클린 이미지 필요 |
| 노이즈 지식 | 불필요 | 가우시안 노이즈 스케줄 필요 |
| 생성 품질 | FID 기준 경쟁적 | 더 높은 품질 달성 |

→ NR-GAN의 핵심 기여인 "노이즈 없는 데이터 가정 제거"는 확산 모델이 클린 이미지를 필요로 한다는 점에서 여전히 중요한 연구 방향을 제시한다.

#### (2) Blind Spot Networks / Noise2Fast 계열

- **Noise2Fast** (Lequyer et al., 2022): 단일 노이즈 이미지로 디노이징 수행
- **Blind2Unblind** (Wang et al., 2022): 재활성화 맵을 통한 맹점 제거

이들은 NR-GAN의 자기지도 디노이징 철학을 계승하면서, GAN 학습 없이도 적용 가능한 더 가벼운 방법을 제시하였다.

#### (3) StyleGAN-ADA (Karras et al., NeurIPS 2020)

*"Training Generative Adversarial Networks with Limited Data"*

적응형 데이터 증강을 통해 소량 데이터에서의 GAN 학습 안정화를 달성. NR-GAN과의 연결점:
- 데이터 증강이 노이즈 강건성과 관련
- 분포 변화에 대한 적응적 학습 전략의 공통점

#### (4) Denoising Diffusion Probabilistic Models (Ho et al., NeurIPS 2020)

확산 모델의 노이즈 스케줄은 명시적으로 설계된 반면, NR-GAN은 암묵적 노이즈 학습 방향을 제시. 두 접근법의 **융합 연구**가 향후 중요한 방향이 될 수 있다.

---

### 4.3 향후 연구 시 고려할 점

#### (1) 복잡한 데이터셋에서의 학습 안정성
논문 자체에서 LSUN BEDROOM, FFHQ 등 복잡한 데이터셋에서 SD-NR-GAN-III의 학습 어려움을 인정하였다. 다음을 고려해야 한다:
- 더 강력한 정규화 기법 연구
- 학습 스케줄 및 GAN 손실 함수의 개선
- Progressive growing 등 점진적 학습 전략 도입

#### (2) 확산 모델과의 통합
NR-GAN의 노이즈 분리 아이디어를 확산 모델에 통합하면:
$$p^g(\boldsymbol{x}_0 | \boldsymbol{y}) \approx p^r(\boldsymbol{x}_0)$$
노이즈 이미지로부터 직접 클린 분포를 학습하는 **노이즈 강건 확산 모델** 개발 가능

#### (3) 노이즈 유형의 자동 감지
현재 NR-GAN은 신호 독립/의존 노이즈에 대해 별도 모델을 설계한다. 노이즈 유형을 자동으로 추론하는 **메타러닝(meta-learning)** 접근법 도입 가능

#### (4) 비정상(Non-stationary) 노이즈 처리
현실의 노이즈는 시간과 공간에 따라 변화한다. 시계열 데이터나 비디오에서의 NR-GAN 확장이 필요하다:

$$\boldsymbol{y}_t = \boldsymbol{x}_t + \boldsymbol{n}_t, \quad \boldsymbol{n}_t \sim p(\boldsymbol{n} | \boldsymbol{n}_{t-1}, \boldsymbol{x}_t)$$

#### (5) 이론적 보장 강화
현재 NR-GAN의 노이즈 분리 성능은 실험적으로 검증되었으나, **이론적 수렴 보장**이 부족하다. 특히:
- $G_x$와 $G_n$의 Nash 균형 조건 분석
- 노이즈 분리의 identifiability 이론적 증명

#### (6) 의료 영상 등 고위험 도메인 적용
의료 CT, MRI 등에서 노이즈는 진단에 직접적 영향을 미친다. NR-GAN의 적용 시:
- 도메인 특화 노이즈 특성 반영 필요
- 생성된 클린 이미지의 신뢰도 정량화 필요

---

## 📚 참고 자료

### 주요 출처 (논문 PDF에서 직접 인용)

1. **Kaneko, T. & Harada, T. (2020). "Noise Robust Generative Adversarial Networks." *CVPR 2020*, pp. 8404–8414.**
   - GitHub: https://github.com/takuhirok/NR-GAN/
   - Project page: https://takuhirok.github.io/NR-GAN/
   - arXiv: arXiv:1911.11776

2. **Bora, A., Price, E., & Dimakis, A.G. (2018). "AmbientGAN: Generative models from lossy measurements." *ICLR 2018*.** (논문 내 [5])

3. **Kingma, D.P. & Welling, M. (2014). "Auto-encoding variational bayes." *ICLR 2014*.** (논문 내 [39], 재파라미터화 트릭)

4. **Lehtinen, J. et al. (2018). "Noise2Noise: Learning image restoration without clean data." *ICML 2018*.** (논문 내 [45])

5. **Krull, A. et al. (2019). "Noise2Void – Learning denoising from single noisy images." *CVPR 2019*.** (논문 내 [43])

6. **Heusel, M. et al. (2017). "GANs trained by a two time-scale update rule converge to a Nash equilibrium." *NIPS 2017*.** (FID 메트릭, 논문 내 [28])

7. **Yang, D. et al. (2019). "Diversity-sensitive conditional generative adversarial networks." *ICLR 2019*.** (다양성 정규화, 논문 내 [73])

### 2020년 이후 비교 연구 (추가 참조, 확인 권장)

8. **Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS 2020*.**

9. **Karras, T. et al. (2020). "Training Generative Adversarial Networks with Limited Data." *NeurIPS 2020*.** (StyleGAN-ADA)

10. **Batson, J. & Royer, L. (2019). "Noise2Self: Blind denoising by self-supervision." *ICML 2019*.** (논문 내 [3])
