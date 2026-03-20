# Residual Denoising Diffusion Models

---

## 1. 핵심 주장 및 주요 기여 요약

**Residual Denoising Diffusion Models (RDDM)**은 기존의 단일 denoising 확산 과정을 **잔차 확산(residual diffusion)**과 **노이즈 확산(noise diffusion)**이라는 **이중 확산 프로세스(dual diffusion process)**로 분리(decouple)하는 새로운 프레임워크를 제안한다.

### 핵심 주장
- 기존 DDPM/DDIM 기반 확산 모델은 이미지 복원(restoration)에 대해 **비해석적(non-interpretable)**이다. 순방향 과정이 열화 이미지에 대한 정보를 포함하지 않으며, 순수 노이즈로부터 역방향 생성을 시작하는 것이 불필요하다.
- **잔차(residual)**는 **확실성(certainty)**을, **노이즈(noise)**는 **다양성(diversity)**을 우선시하여, 이미지 생성과 복원을 **통합적이고 해석 가능한** 하나의 프레임워크로 묶을 수 있다.

### 주요 기여
1. **이중 확산 프레임워크**: 잔차 확산을 도입하여 타깃 이미지에서 열화 입력 이미지로의 방향성 확산을 모델링
2. **부분적 경로 독립 생성 과정(partially path-independent generation process)**: 잔차와 노이즈를 분리하여 각각의 역할을 규명
3. **자동 목적 함수 선택 알고리즘(AOSA)**: 미지의 새로운 태스크에 대해 잔차 예측(SM-Res)과 노이즈 예측(SM-N) 중 최적 방법을 자동 선택
4. **범용 UNet으로 SOTA 수준 달성**: $\ell_1$ 손실과 배치 크기 1만으로 다양한 이미지 복원 태스크에서 최신 방법과 경쟁

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 DDPM [Ho et al., 2020]과 DDIM [Song et al., 2021]은 **순수 노이즈에서 시작하는 역방향 과정**을 사용한다. 이미지 복원에 이를 적용할 때:

- **순방향 과정의 비해석성**: 확산 과정이 열화 이미지에 대한 정보를 전혀 포함하지 않음 (Figure 1(a))
- **불필요한 역방향 시작점**: 이미 열화 이미지가 알려져 있음에도 순수 노이즈에서 시작
- **비효율성**: 열화 이미지를 단순히 조건 입력(condition)으로 사용하여 역방향 과정을 암묵적으로만 유도

### 2.2 제안하는 방법 (수식 포함)

#### 순방향 과정 재정의

RDDM은 기존 DDPM의 $I_T = \epsilon$을 다음과 같이 수정한다:

$$I_T = I_{in} + \epsilon$$

여기서 $I_{in}$은 열화 이미지(복원 시) 또는 $0$(생성 시)이다.

**단일 스텝 순방향 과정**은 다음과 같이 정의된다:

$$I_t = I_{t-1} + I_{res}^t, \qquad I_{res}^t \sim \mathcal{N}(\alpha_t I_{res}, \beta_t^2 \mathbf{I})$$

여기서 $I_{res} = I_{in} - I_0$는 열화 이미지와 타깃 이미지 간의 잔차이며, $\alpha_t$와 $\beta_t$는 각각 잔차 확산과 노이즈 확산의 **독립적인 계수 스케줄**이다.

**재매개변수화(reparameterization)**를 통해 $I_0$로부터 직접 $I_t$를 샘플링할 수 있다:

$$I_t = I_0 + \bar{\alpha}_t I_{res} + \bar{\beta}_t \epsilon$$

여기서 $\epsilon \sim \mathcal{N}(0, \mathbf{I})$, $\bar{\alpha}\_t = \sum_{i=1}^{t} \alpha_i$, $\bar{\beta}\_t = \sqrt{\sum_{i=1}^{t} \beta_i^2}$이다.

이는 기존 DDPM의 **2항 혼합**(two-term mixture: $I_0$와 $\epsilon$)을 넘어선 **3항 혼합**(three-term mixture: $I_0$, $I_{res}$, $\epsilon$)이다.

#### 결합 확률 분포

$$q(I_{1:T}|I_0, I_{res}) := \prod_{t=1}^{T} q(I_t|I_{t-1}, I_{res})$$

$$q(I_t|I_{t-1}, I_{res}) := \mathcal{N}(I_t; I_{t-1} + \alpha_t I_{res}, \beta_t^2 \mathbf{I})$$

#### 역방향 생성 과정

전이 확률 $q_\sigma(I_{t-1}|I_t, I_0, I_{res})$는 다음과 같다:

$$q_\sigma(I_{t-1}|I_t, I_0, I_{res}) = \mathcal{N}\left(I_{t-1}; I_0 + \bar{\alpha}_{t-1}I_{res} + \sqrt{\bar{\beta}_{t-1}^2 - \sigma_t^2} \frac{I_t - (I_0 + \bar{\alpha}_t I_{res})}{\bar{\beta}_t}, \sigma_t^2 \mathbf{I}\right)$$

여기서 $\sigma_t^2 = \eta \beta_t^2 \bar{\beta}_{t-1}^2 / \bar{\beta}_t^2$이고, $\eta$는 확률적($\eta=1$) 또는 결정론적($\eta=0$) 생성 과정을 제어한다.

$I_{t-1}$의 샘플링 공식:

$$I_{t-1} = I_t - (\bar{\alpha}_t - \bar{\alpha}_{t-1})I_{res}^\theta - \left(\bar{\beta}_t - \sqrt{\bar{\beta}_{t-1}^2 - \sigma_t^2}\right)\epsilon_\theta + \sigma_t \epsilon_t$$

결정론적 샘플링($\eta = 0$):

$$I_{t-1} = I_t - (\bar{\alpha}_t - \bar{\alpha}_{t-1})I_{res}^\theta - (\bar{\beta}_t - \bar{\beta}_{t-1})\epsilon_\theta$$

#### 학습 목적 함수

$$L_{res}(\theta) := \mathbb{E}\left[\lambda_{res} \left\|I_{res} - I_{res}^\theta(I_t, t, I_{in})\right\|^2\right]$$

$$L_\epsilon(\theta) := \mathbb{E}\left[\lambda_\epsilon \left\|\epsilon - \epsilon_\theta(I_t, t, I_{in})\right\|^2\right]$$

여기서 $\lambda_{res}, \lambda_\epsilon \in \{0, 1\}$이다.

#### 세 가지 샘플링 전략

| 전략 | $\lambda_{res}$ | $\lambda_\epsilon$ | 특성 |
|------|:-:|:-:|------|
| **SM-Res** | 1 | 0 | 잔차 예측, 복원에 유리 (확실성 우선) |
| **SM-N** | 0 | 1 | 노이즈 예측, 생성에 유리 (다양성 우선) |
| **SM-Res-N** | 1 | 1 | 잔차+노이즈 동시 예측 |

#### DDPM/DDIM과의 호환성

계수 변환을 통해 RDDM의 샘플링 과정이 DDPM/DDIM과 일치함을 증명한다:

$$\bar{\alpha}_t = 1 - \sqrt{\bar{\alpha}_{DDIM}^t}, \quad \bar{\beta}_t = \sqrt{1 - \bar{\alpha}_{DDIM}^t}, \quad \sigma_t^2 = \sigma_t^2(DDIM)$$

### 2.3 모델 구조

- **UNet 아키텍처**: 채널 크기 64, 채널 배율 (1,2,4,8)의 표준 UNet 사용
- **잔차 예측 네트워크** $I_{res}^\theta(I_t, t, I_{in})$와 **노이즈 예측 네트워크** $\epsilon_\theta(I_t, t, I_{in})$
- SM-Res-N-2Net: 두 개의 독립 네트워크로 잔차와 노이즈를 각각 예측
- SM-Res-N-1Net: 하나의 네트워크로 6채널 출력 (0-3: 잔차, 3-6: 노이즈)
- 모든 복원 태스크에서 **동일한 UNet 구조**를 사용하며, 배치 크기 1, $\ell_1$ 손실만 사용

### 2.4 성능 향상

| 태스크 | 데이터셋 | RDDM 성능 | 비교 |
|------|--------|---------|------|
| 그림자 제거 | ISTD | MAE 4.67, PSNR 30.91 | DMTN(4.72) 대비 우수 |
| 저조도 향상 | LOL | PSNR 25.39, SSIM 0.937 | LLFlow(25.19) 대비 우수 |
| 저조도 향상 | SID-RGB | PSNR 23.97, SSIM 0.839 | SNR-Aware(22.87) 대비 **4.8%** PSNR, **34.2%** SSIM 향상 |
| 제비(deraining) | RainDrop | PSNR 32.51, SSIM 0.9563 | RainDiff128(32.43) 대비 우수, 샘플링 스텝 5 vs 50 |
| 이미지 생성 | CelebA | FID 변환 후 DDIM과 동일 | 계수 변환으로 완전 호환 |

**핵심 효율성**: SR3 대비 10배 적은 파라미터, 10배 적은 학습 반복, 10배 빠른 추론, 10% PSNR/SSIM 향상 (ISTD shadow removal).

### 2.5 한계

1. **통합 프로토타입 모델**로서 태스크별 특화 SOTA와 비교 시 성능 한계 존재
2. 이미지 생성에서 SOTA 달성을 위해서는 더 큰 네트워크, 배치 크기, 고급 학습 전략 필요
3. 과도한 확산 속도 변경 시 부분적 경로 독립성이 깨짐 (예: $\alpha_t, \beta_t^2 \to P(x, 5)$ )
4. 배치 크기 1로 학습 시 SM-N(노이즈만 예측)의 복원 성능 저하
5. 이미지 번역(translation) 등 비짝(unpaired) 데이터 태스크에서 품질 한계

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 통합 프레임워크로서의 일반화

RDDM의 가장 핵심적인 일반화 강점은 **하나의 프레임워크로 이미지 생성과 복원을 동시에 처리**한다는 점이다:

- **생성 시**: $I_{in} = 0$으로 설정하면 $I_T = \epsilon$ (순수 노이즈)이 되어 기존 DDPM과 동일
- **복원 시**: $I_{in}$이 열화 이미지로 설정되어 $I_T = I_{in} + \epsilon$ (노이즈를 포함한 열화 이미지)

이러한 설계로 그림자 제거, 저조도 향상, 제비, 디블러링, 인페인팅, 이미지 번역 등 **다양한 태스크에 동일한 UNet을 적용**할 수 있음을 실험적으로 검증하였다 (Table 4).

### 3.2 독립적 이중 계수 스케줄

기존 DDPM이 하나의 계수 스케줄로 노이즈와 이미지의 혼합 비율을 제어하는 반면, RDDM은 두 개의 독립적 계수 스케줄 $\alpha_t$와 $\beta_t^2$를 사용한다:

$$I_t = I_0 + \bar{\alpha}_t I_{res} + \bar{\beta}_t \epsilon$$

이 독립성은 다음을 가능하게 한다:

1. **분리된 확산 속도 곡선 설계**: $\alpha_t$ (선형 감소), $\beta_t^2$ (선형 증가)가 최적 성능을 보임 (Table 2, FID 23.25)
2. **태스크별 노이즈 강도 조절**: $\bar{\beta}_T^2$를 태스크에 따라 조정 (그림자 제거: 0.01, 생성: 1)
3. **부분적 경로 독립 생성 과정**: 테스트 시 확산 속도 곡선을 재조정해도 생성 결과가 의미론적으로 일관됨 (Figure 5, 6)

### 3.3 부분적 경로 독립성과 일반화

Green 정리에 기반한 분석에서, 잘 훈련된 네트워크가 충분히 강건하다면:

$$\frac{\partial I_{res}^\theta(I(t), \bar{\alpha}(t) \cdot T)}{\partial \bar{\beta}(t)} \approx 0, \quad \frac{\partial \epsilon_\theta(I(t), \bar{\beta}(t) \cdot T)}{\partial \bar{\alpha}(t)} \approx 0$$

이 조건이 성립하면 곡선 적분에서 경로 독립의 필요충분조건이 되며, 이는 확산 속도 곡선의 일정 범위 내 변경에도 생성이 안정적임을 의미한다. 이는 **모델의 일반화 강건성**을 이론적으로 뒷받침한다.

### 3.4 자동 목적 함수 선택 알고리즘 (AOSA)

미지의 새로운 태스크에 대한 일반화를 위해, 학습 가능한 파라미터 $\lambda_{res}^\theta$를 도입하여:

$$L_{auto}(\theta) := \lambda_{res}^\theta \mathbb{E}\left[\left\|I_{res} - I_{res}^\theta(I_t, t, I_{in})\right\|^2\right] + (1 - \lambda_{res}^\theta)\mathbb{E}\left[\left\|\epsilon - \epsilon_\theta(I_t, t, I_{in})\right\|^2\right]$$

- 복원 태스크 → 약 300 반복 후 자동으로 SM-Res로 전환
- 생성 태스크 → 약 1000 반복 후 자동으로 SM-N으로 전환
- 추가 학습 비용 1000 반복 이하, 기존 DDPM 방법과 완전 호환

### 3.5 효율적 학습을 통한 일반화

- **배치 크기 1**과 **$\ell_1$ 손실**만으로 SOTA와 경쟁 → 계산 자원 제약 환경에서 확산 모델 활용 가능
- **5 스텝 이하의 샘플링**으로 충분한 복원 품질 달성
- **잔차 예측이 수렴 가속화**에 기여 (Table 11(b): SR3 대비 10배 적은 학습 반복으로 우수한 성능)

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

1. **확산 모델의 패러다임 확장**: 기존 2항 혼합($I_0, \epsilon$)에서 3항 혼합($I_0, I_{res}, \epsilon$)으로의 확장은 확산 모델의 이론적 토대를 넓힌다. 이는 **다차원 확산 과정(multi-dimensional diffusion process)**으로의 발전 가능성을 열어준다.

2. **이미지 복원을 위한 확산 모델의 해석 가능성 확립**: 잔차 확산이 타깃→입력 방향의 의미론적 전이를, 노이즈 확산이 랜덤 섭동을 담당한다는 명확한 역할 분리는 확산 모델의 **블랙박스 성격을 완화**한다.

3. **통합 이미지-이미지 분포 변환 방법론**: 생성, 복원, 인페인팅, 번역을 하나의 프레임워크로 처리할 수 있는 가능성은 **범용 비전 모델(foundation model)** 연구에 시사점을 준다.

4. **효율적 확산 모델 연구 촉진**: 배치 크기 1, 5 스텝 이하 샘플링, 4.8G GPU 메모리만으로 SOTA 수준을 달성한 것은 계산 자원이 제한된 연구 환경에서의 활용 가능성을 크게 확장한다.

5. **계수 스케줄 설계 공간 확장**: 잔차와 노이즈의 독립적 스케줄링은 기존 단일 스케줄 설계를 넘어선 **새로운 최적화 공간**을 제공한다.

### 4.2 향후 연구 시 고려할 점

1. **다차원 확산 과정 확장**: RDDM의 이중 확산을 넘어 3개 이상의 독립 확산 과정을 결합하는 연구 (예: 텍스트-이미지 멀티모달 조건 생성)

2. **적응적 계수 스케줄 학습**: 현재 수동 설정되는 $\alpha_t$, $\beta_t^2$ 스케줄을 데이터로부터 자동 학습하여 샘플링 스텝을 줄이면서 생성 품질을 향상

3. **잠재 공간(latent space)으로의 확장**: Latent Diffusion Model [Rombach et al., 2022]과의 결합을 통한 고해상도 생성/복원

4. **멀티태스크 통합 학습**: 하나의 사전 훈련 파라미터 세트로 여러 다른 태스크를 처리하는 모델 개발

5. **증류(distillation) 연구**: Consistency Models [Song et al., 2023]에 이중 확산 개념을 도입하여 1-2 스텝 생성 달성

6. **적응적 노이즈 강도 학습**: $\bar{\beta}_T^2$를 미지의 태스크에 대해 자동 학습

7. **대규모 생성 모델에서의 검증**: 현재 64×64, 256×256 해상도에서의 실험을 넘어 고해상도, 대규모 데이터셋에서의 성능 검증 필요

8. **이론적 분석 심화**: RDDM과 곡선/다변량 적분, 확률미분방정식(SDE)과의 관계에 대한 더 깊은 수학적 분석

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 모델/방법 | 연도 | 핵심 접근 | RDDM과의 주요 차이 |
|----------|------|---------|----------------|
| **DDPM** [Ho et al.] | 2020 | 노이즈 예측, 2항 혼합 | 복원에 비해석적; RDDM은 3항 혼합으로 확장 |
| **DDIM** [Song et al.] | 2021 | 결정론적 샘플링 | RDDM과 계수 변환으로 호환; RDDM은 독립 이중 스케줄 |
| **IDDPM** [Nichol & Dhariwal] | 2021 | 평균+분산 동시 학습 | 여전히 2항 혼합; RDDM은 잔차+노이즈 동시 예측 |
| **SDEdit** [Meng et al.] | 2021 | SDE로 열화 이미지에서 시작 | 열화 이미지에서 시작하나 잔차 개념 부재 |
| **SR3** [Saharia et al.] | 2022 | 조건부 DDPM으로 초해상도 | 노이즈만 예측; RDDM 대비 10배 많은 파라미터/스텝 |
| **LDM** [Rombach et al.] | 2022 | 잠재 공간 확산 | 픽셀 공간 vs 잠재 공간; RDDM 프레임워크와 결합 가능 |
| **ColdDiffusion** [Bansal et al.] | 2022 | 노이즈 없이 임의 변환 | 노이즈를 완전 제거; RDDM은 노이즈를 유지(다양성 보존) |
| **Rectified Flow** [Liu et al.] | 2023 | 잔차 예측, 노이즈 없는 보간 | RDDM에서 $\bar{\beta}_T=0$일 때의 특수 경우 |
| **I2SB** [Liu et al.] | 2023 | Schrödinger Bridge, 3항 혼합 | 타깃/선형 변환 추정; RDDM은 잔차+노이즈 독립 예측 |
| **InDI** [Delbracio & Milanfar] | 2023 | 직접 반복, 3항 혼합 | 타깃 이미지 추정; RDDM의 SM-Res의 특수 경우 |
| **IR-SDE** [Luo et al.] | 2023 | 평균 회귀 SDE | 3항 혼합이나 독립 이중 스케줄 없음 |
| **ResShift** [Yue et al.] | 2023 | 잔차 이동으로 효율적 초해상도 | 잔차 사용하나 RDDM의 일반적 이중 확산 프레임워크와 차별 |
| **Consistency Models** [Song et al.] | 2023 | 1-2 스텝 생성을 위한 증류 | RDDM의 이중 확산 개념과 결합 가능성 |

### 핵심 차별점 요약

RDDM의 근본적 차별점은 다음과 같다:

1. **독립적 이중 확산**: InDI, I2SB 등이 동일한 3항 혼합 형태를 사용하지만, RDDM만이 잔차와 노이즈를 **독립적 계수 스케줄**로 제어한다.
2. **잔차와 노이즈의 동등한 지위**: 기존 연구들은 노이즈 *또는* 타깃/잔차 *중 하나만* 예측하는 반면, RDDM은 **양쪽 모두를 동시에** 예측하고 태스크에 따라 선택한다.
3. **부분적 경로 독립성**: 테스트 시 확산 속도 곡선을 재조정해도 생성이 안정적인 특성은 RDDM 고유의 발견이다.

---

## 참고자료

1. **Jiawei Liu, Qiang Wang, Huijie Fan, Yinong Wang, Yandong Tang, Liangqiong Qu.** "Residual Denoising Diffusion Models." arXiv:2308.13712v3 [cs.CV], 22 Mar 2024. (본 논문)
2. **Jonathan Ho, Ajay Jain, Pieter Abbeel.** "Denoising Diffusion Probabilistic Models." NeurIPS, 2020.
3. **Jiaming Song, Chenlin Meng, Stefano Ermon.** "Denoising Diffusion Implicit Models." ICLR, 2021.
4. **Alexander Quinn Nichol, Prafulla Dhariwal.** "Improved Denoising Diffusion Probabilistic Models." ICML, 2021.
5. **Mauricio Delbracio, Peyman Milanfar.** "Inversion by Direct Iteration: An Alternative to Denoising Diffusion for Image Restoration." TMLR, 2023.
6. **Guan-Horng Liu, Arash Vahdat, De-An Huang, et al.** "I2SB: Image-to-Image Schrödinger Bridge." ICML, 2023.
7. **Arpit Bansal et al.** "Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise." arXiv:2208.09392, 2022.
8. **Xingchao Liu, Chengyue Gong, Qiang Liu.** "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." ICLR, 2023.
9. **Zongsheng Yue, Jianyi Wang, Chen Change Loy.** "ResShift: Efficient Diffusion Model for Image Super-Resolution by Residual Shifting." NeurIPS, 2023.
10. **Yang Song, Prafulla Dhariwal, Mark Chen, Ilya Sutskever.** "Consistency Models." ICML, 2023.
11. **Robin Rombach, Andreas Blattmann, Dominik Lorenz, et al.** "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR, 2022.
12. **Chitwan Saharia et al.** "Image Super-Resolution via Iterative Refinement." IEEE TPAMI, 2022.
13. **Ziwei Luo et al.** "Image Restoration with Mean-Reverting Stochastic Differential Equations." ICML, 2023.
14. GitHub 저장소: https://github.com/nachifur/RDDM
