# Learning An Explicit Weighting Scheme for Adapting Complex HSI Noise

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장
기존 HSI(Hyperspectral Image) 노이즈 제거 방법들은 노이즈 분포에 대한 **주관적 사전 가정(subjective prior assumption)**에 의존하여 실제 복잡한 노이즈에 일반화하기 어렵다. 본 논문은 **데이터 기반(data-driven) 방식**으로 가중치 부여 원칙(weighting principle)을 명시적 함수로 학습하는 **HWnet(Hyper-Weight-Net)**을 제안한다.

### 주요 기여
| 기여 | 내용 |
|------|------|
| **① 명시적 가중치 함수 학습** | 노이즈 이미지 $Y \to$ 가중치 행렬 $W$로의 매핑을 데이터 기반으로 학습 |
| **② 우수한 일반화 능력** | 학습 시 사용하지 않은 노이즈 유형에도 적용 가능 |
| **③ 플러그 앤 플레이(Plug & Play) 전이** | 다른 가중치 기반 HSI 복원 모델에도 직접 적용 가능 |
| **④ DHP 개선** | Deep Hyperspectral Prior 학습 손실 함수를 개선하여 과적합 억제 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

실제 HSI에서 노이즈는 다음과 같은 특성을 가진다:
- **공간·스펙트럼 방향으로 non-i.i.d.** 분포
- Gaussian, Stripe, Impulse, Deadline 등 복합 노이즈 혼재
- 기존 방법들은 노이즈 분포 가정(mixture model 등)에 의존 → 실제 노이즈와 편차 발생 시 성능 저하

### 2.2 제안 방법 및 수식

#### 기본 가중치 최적화 모델 (식 1)

$$\min_{X} \|W \odot (Y - X)\|_F + \lambda R(X) $$

- $Y, X \in \mathbb{R}^{hw \times b}$: 노이즈 이미지 및 복원 이미지
- $W$: 픽셀별 가중치 행렬 (노이즈가 많은 픽셀 → 작은 가중치)
- $R(\cdot)$: 정규화 항

#### 풀 베이지안 모델 (Full Bayesian Model)

노이즈 생성 과정을 다음 가우시안으로 모델링:

$$Y_{ij} \sim \mathcal{N}(Y_{ij} | X_{ij}, 1/W^2_{ij}), \quad i \in [hw], j \in [b] $$

$W^2_{ij}$를 정밀도(precision)로 해석, 잠재 변수 $X$에 대한 사전분포:

$$X \sim \mathcal{N}(X | X_{gt}, \varepsilon^2) $$

$W^2_{ij}$에 대한 켤레 사전분포(Gamma):

$$W^2_{ij} \sim \Gamma(\rho + 1, \rho\sigma^2_{ij}) $$

전체 결합 확률:

$$p(X, Y, W^2) = p(Y|X, W^2)p(X)p(W^2) $$

#### 변분 사후분포 (Variational Parametric Posterior, 식 6~8)

$$q(X, W^2|Y) = \prod_{ij} q(W^2_{ij}|Y) \prod_{ij} q(X_{ij}|Y, W^2) $$

$$q(W^2_{ij}|Y) = \Gamma(W^2_{ij} | \alpha(Y;\theta)_{ij}, \beta(Y;\theta)_{ij}) $$

$$q(X_{ij}|Y, W^2) = \mathcal{N}(X_{ij} | (G(Y;W))_{ij}, \eta^2) $$

- $(\alpha, \beta)$: HWnet $C_\theta$가 예측하는 Gamma 분포 파라미터
- 최종 가중치: $W^2_{ij} = (\alpha_{ij} - 1)/\beta_{ij}$ (Gamma 분포의 최빈값)

#### WLRMF 알고리즘을 명시적 함수 G로 표현 (식 9~13)

$$\min_{X,U,V} \|W \odot (Y-X)\|^2_F + \lambda\|X - UV^T\|^2_F $$

반복 갱신:

$$\{U_{(k)}V^T_{(k)}\} = \text{SVD}_r(X_{(k-1)}) $$

$$X_{(k)} = \frac{W^2 \odot Y + \lambda_{(k)} U_{(k)}V^T_{(k)}}{W^2 + \lambda_{(k)}} $$

각 반복을 명시적 함수로 표현:

$$X_{(k)} = g_k(X_{(k-1)}, Y; W^2) = \frac{W^2 \odot Y + \lambda_{(k)}\text{SVD}_r(X_{(k-1)})}{W^2 + \lambda_{(k)}} $$

$N$번 반복 후 전체 알고리즘 함수:

$$X_{(N)} = g_N(\cdots g_1(X_{(0)}, Y; W^2), Y; W^2) = G(Y; W^2) $$

#### 손실 함수 (KL Divergence 최소화, 식 14~18)

$$\min_\theta \text{KL}[q(X, W^2|Y) \| p(X, W^2|Y)] $$

변분 추론에 의해 등가:

$$\min_\theta \underbrace{-\mathbb{E}_{q(X,W^2|Y)}[\ln p(Y|X,W^2)]}_{L_1} + \underbrace{\text{KL}[q(W^2|Y)\|p(W^2)]}_{L_2} + \underbrace{\mathbb{E}_{q(W^2|Y)}\{KL[q(X|Y,W^2)\|p(X)]\}}_{L_3} $$

$L_1$ (재구성 손실):

```math
L_1 \approx \sum_{m=1}^{M}\left\{\frac{1}{S}\sum_{s=1}^{S}\frac{1}{2}\|(W^m)^s \odot (Y^m - (X^m_{(N)})^s)\|^2_F + \sum_{ij}\left\{\frac{\eta^2 \alpha^m_{ij}}{2\beta^m_{ij}} + \frac{1}{2}\ln 2\pi - \frac{1}{2}[\psi(\alpha^m_{ij}) - \ln\beta^m_{ij}]\right\}\right\}
```

$L_2$ (가중치 정규화):

```math
L_2 = \sum_{m=1}^{M}\sum_{ij}\left\{(\alpha^m_{ij}-\rho-1)\psi(\alpha^m_{ij}) - \ln\Gamma(\alpha^m_{ij}) + \ln\Gamma(\rho+1) + (\rho+1)(\ln\beta^m_{ij} - \ln\rho(\sigma^m_{ij})^2) + \alpha^m_{ij}\left(\frac{\rho(\sigma^m_{ij})^2}{\beta^m_{ij}} - 1\right)\right\}
```

$L_3$ (이미지 재구성 정규화):

```math
L_3 \approx \sum_{m=1}^{M}\left\{\frac{1}{S}\sum_{s=1}^{S}\frac{1}{2\varepsilon^2}\|(X^m_{(N)})^s - X^m_{gt}\|^2_F\right\} + \frac{\eta^2}{2\varepsilon^2} - \frac{1}{2}\ln\frac{\eta^2}{\varepsilon^2} - \frac{1}{2}
```

#### DHP 확장 (HW-DHP, 식 20~21)

기존 DHP:

$$\min_\mu \|C_\mu(Z) - Y\|^2_F $$

제안 HW-DHP:

$$\min_\mu \|W \odot (C_\mu(Z) - Y)\|^2_F $$

### 2.3 모델 구조

```
HWnet (C_θ) 구조:
입력: 노이즈 HSI Y (hw × b)
↓
(P3D + ReLU) × 1
↓
(P3D + BN + ReLU) × 3
↓
Full 3D Conv
↓
출력: α, β (Gamma 분포 파라미터)
→ W² = (α-1)/β
```

- **P3D(Pseudo-3D) 블록**: $3\times3\times1$ conv + ReLU + $1\times1\times3$ conv 구조로 파라미터 절감
- **DnCNN 유사 5-layer 구조**, 각 레이어 채널 수 64
- WLRMF의 rank $r=3$, 반복 횟수 $N=20$(학습), $N=150$(테스트)

### 2.4 성능 향상

**ICVL 데이터셋 기준 (Tab. 1 요약)**:

| Case | NMoG (SOTA) | HW-LRMF (제안) | 향상 |
|------|------------|--------------|------|
| Case 1 | 30.99 dB | 34.93 dB | +3.94 dB |
| Case 3 | 30.71 dB | **33.67 dB** | +2.96 dB |
| Case 7 | 25.38 dB | **28.38 dB** | +3.00 dB |

**전이 실험 (Tab. 2 요약)**:
- HW-NAILRMA: 모든 Case에서 원본 NAILRMA 능가
- HW-NGmeet: 복잡 노이즈 Case에서 큰 향상
- HW-LLRT: 다른 이미지 prior에도 효과적

### 2.5 한계점

1. **학습 데이터 의존성**: WLRMF 모델로 학습되어 다른 모델로의 전이 시 여전히 일부 편향 존재 (특히 복잡한 노이즈에서 NGmeet, LLRT의 개선폭 감소)
2. **훈련 복잡도**: SVD의 역전파 계산이 추가 계산 비용 유발
3. **소규모 학습 데이터**: CAVE 데이터셋 20장으로만 학습
4. **파라미터 민감성**: $\rho=25$, $\varepsilon^2=10^{-5}$, $\eta^2=10^{-2}$ 등 하이퍼파라미터를 경험적으로 설정
5. **실제 노이즈 대응**: 실제 HSI(HYDICE Urban) 실험 시 파인튜닝 필요

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화의 핵심 메커니즘

논문의 가장 중요한 기여는 **학습 노이즈 유형(Case 2: non-i.i.d. Gaussian)**과 **테스트 노이즈 유형(Case 1~7)**이 다름에도 일반화가 이루어진다는 점이다. 그 원인은 다음과 같다:

**① 베이지안 프레임워크의 일반화 유도**

$L_2$ 손실인 $\text{KL}[q(W^2|Y)\|p(W^2)]$가 가중치 분포가 노이즈 사전 분포에서 너무 벗어나지 않도록 정규화함으로써 과적합 방지.

**② 파라미터 공유**

$\theta$는 모든 학습 쌍에서 공유 → **공통 통계적 추론 원칙**을 추출.

**③ Non-i.i.d. 노이즈 표현**

$W_{ij}^2$가 픽셀별로 독립적으로 추정되므로, 다양한 공간·스펙트럼 분포의 노이즈를 유연하게 처리.

**④ Plug & Play 전이 가능성**

학습된 $C_\theta$는 어떤 weighted 모델 $(1)$에도 적용 가능 → 일반화 범위가 모델 수준까지 확장.

### 3.2 일반화 향상을 위한 추가 가능성

| 방향 | 내용 |
|------|------|
| 다양한 학습 노이즈 | Case 1~7 모두로 학습 시 더 강력한 일반화 기대 |
| Meta-learning 결합 | 소량 데이터에서의 빠른 적응 가능 |
| 자기지도 학습 | Noise2Noise, Blind-spot 등과 결합하여 GT 불필요한 학습 |
| 도메인 적응 | 원격탐사, 의료 등 도메인 간 전이 연구 |

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구에 미치는 영향

**① 명시적 가중치 함수의 패러다임 전환**
- 기존의 수작업 가중치 설계 → 데이터 기반 학습으로의 전환 방향 제시
- 모델 기반(model-driven)과 데이터 기반(data-driven)의 융합 방법론 제안

**② Plug & Play 가중치 모듈의 표준화 가능성**
- 다양한 HSI 복원 모델(저랭크, 텐서, DL 등)에 범용적으로 적용 가능한 외부 가중치 모듈 개념 제시

**③ 손실 함수 설계의 새로운 관점**
- 가중치 손실(weighted loss)이 과적합에 더 강인하다는 실험적 증거 제공 → DHP 등 unsupervised 방법론에 직접 적용 가능

**④ 베이지안 + 변분 추론 + 딥러닝의 융합**
- 불확실성 추정(uncertainty estimation)이 가능한 가중치 부여 방식 → 신뢰도 기반 HSI 분석 연구 촉진

### 4.2 향후 연구 시 고려할 점

**① 학습 데이터 다양성 확보**
- 현재는 CAVE 20장으로 학습 → 더 다양한 장면, 센서, 스펙트럼 범위의 데이터 필요

**② 실제 노이즈 시나리오 대응**
- 실제 노이즈의 경우 GT 획득이 어려움 → Self-supervised 또는 Noise2Noise 방식과 결합 필요

**③ 계산 효율화**
- SVD의 역전파 계산 비용 → 근사 SVD(Randomized SVD) 등 도입 고려

**④ 불확실성 정량화 활용**
- 예측된 Gamma 분포의 분산 정보를 활용한 신뢰도 맵 생성 및 다운스트림 태스크 활용

**⑤ 멀티모달 확장**
- RGB-HSI 융합, LiDAR 결합 등 다중 모달 데이터에서의 가중치 적응

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 방법 | 노이즈 적응 방식 | 일반화 |
|------|------|--------------|------|
| **HWnet (본 논문, CVPR 2021)** | 명시적 가중치 함수 학습 (Bayesian + VI) | 데이터 기반 | 다노이즈 유형, 다모델 전이 |
| **3D-QRNN (Wei et al., IEEE TNNLS 2020)** | 3D Quasi-Recurrent NN | 단일 DL 모델 | 학습 노이즈 유형 의존 |
| **MHF-Net (Xie et al., IEEE TPAMI 2020)** | 해석 가능한 딥 언폴딩, 다중 스펙트럼 융합 | 알고리즘 언롤링 | 융합 특화 |
| **Enhanced 3DTV (Peng et al., IEEE TIP 2020)** | 향상된 3D Total Variation | 수작업 prior | 제한적 |
| **NGmeet (He et al., CVPR 2019)** | Non-local + Global low-rank | 가우시안 특화 | 단순 노이즈 |
| **SST (He et al., ICCV 2023)** | Spectral-Spatial Transformer | 자기지도 주의 | DL 의존 |

### 비교 분석 요약

- **HWnet의 차별점**: 노이즈 분포 가정 없이 일반화 가능한 최초의 명시적 가중치 함수 학습 접근법
- **한계 대비 DL 방법**: SST 등 최신 Transformer 기반 방법이 순수 PSNR 성능에서 우수할 수 있으나, HWnet은 **해석 가능성(interpretability)**과 **전이 가능성(transferability)**에서 차별화
- **모델 기반 방법 대비**: 노이즈 사전 가정 불필요 → 실제 복잡 노이즈 적응력 우수

---

## 참고 자료

- **주 논문**: Xiangyu Rui et al., "Learning An Explicit Weighting Scheme for Adapting Complex HSI Noise," *CVPR 2021*, pp. 6739–6748. (제공된 PDF 원문)
- 논문 내 인용 참고문헌:
  - Wei et al., "3-D Quasi-Recurrent Neural Network for Hyperspectral Image Denoising," *IEEE TNNLS*, 2020. [37]
  - Xie et al., "MHF-Net: An Interpretable Deep Network for Multispectral and Hyperspectral Image Fusion," *IEEE TPAMI*, 2020. [39]
  - Peng et al., "Enhanced 3DTV Regularization and Its Applications on HSI Denoising and Compressed Sensing," *IEEE TIP*, 2020. [27]
  - He et al., "Non-local Meets Global: An Integrated Paradigm for Hyperspectral Denoising," *CVPR 2019*. [15]
  - Sidorov & Hardeberg, "Deep Hyperspectral Prior," *ICCV Workshops*, 2019. [32]

> **주의**: SST(2023) 등 2021년 이후 논문에 대한 상세 수치 비교는 본 논문 원문에 포함되지 않으므로, 해당 부분은 일반적 연구 동향 기반 기술임을 밝힙니다.
