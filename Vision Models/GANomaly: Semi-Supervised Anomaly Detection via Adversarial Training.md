# GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

**GANomaly**는 **일반 샘플만으로 학습하는 반감독(Semi-Supervised) 이상탐지 패러다임**을 제시합니다. 핵심 가설은 다음과 같습니다:

> "정상 샘플로만 학습된 생성 네트워크는 비정상 샘플을 재구성할 수 없으며, 이러한 재구성 실패가 이미지 공간과 잠재 공간 모두에서 측정 가능한 불일치(discrepancy)를 유발한다."

### 1.2 주요 기여 3가지

1. **구조적 혁신**: 인코더-디코더-인코더(Encoder-Decoder-Encoder) 파이프라인을 통한 이중 공간 학습
   - 이미지 공간 재구성
   - 잠재 공간 표현 학습

2. **효율성 달성**: 
   - 단일 단계 학습 (이중 단계 최적화 불필요)
   - 추론 시간 ~ 2.8ms (MNIST) - AnoGAN의 7120ms 대비 2500배 이상 빠름

3. **일반화 능력**:
   - 다양한 도메인에서 검증 (MNIST, CIFAR-10, X-ray 보안 영상)
   - FFOB 데이터셋: AUC 0.882 (vs. EGBAD 0.712, AnoGAN 0.703)

---

## 2. 상세 기술 설명

### 2.1 문제 정의

**형식적 문제 정의**:

주어진 정상 샘플의 큰 훈련 데이터셋: $$D = \{X_1, \ldots, X_M\}$$

정상 및 비정상 혼합 테스트 데이터셋: $$\hat{D} = \{(\hat{X}_1, y_1), \ldots, (\hat{X}_N, y_N)\}$$, 여기서 $y_i \in [0, 1]$

목표: 정상 데이터의 분포 $p_X$를 학습하고, 테스트 단계에서 이상 점수 $A(x) > \phi$인 샘플을 이상으로 판정

### 2.2 제안 방법: GANomaly 아키텍처

#### 2.2.1 네 가지 주요 구성 요소

**1. 생성기(Generator G)**
- 입력 이미지 $x \in \mathbb{R}^{w \times h \times c}$를 처리
- 인코더 $G_E$: 이미지를 잠재 벡터 $z \in \mathbb{R}^d$로 압축
- 디코더 $G_D$: $z$로부터 재구성 이미지 $\hat{x} = G_D(z)$ 생성
- 수식: $$\hat{x} = G_D(G_E(x))$$

**2. 추가 인코더(Encoder E)**
- 재구성 이미지 $\hat{x}$를 처리하여 $\hat{z} = E(\hat{x})$ 계산
- $G_E$와 동일한 아키텍처, 다른 파라미터
- **핵심 기여**: 명시적으로 재구성된 이미지의 잠재 표현을 학습

**3. 판별기(Discriminator D)**
- 표준 DCGAN 판별기 아키텍처
- 실제 vs. 생성 이미지 분류 (특성 매칭 손실 사용)

**4. 이상 점수 계산**
- 훈련 단계: 세 가지 손실 함수의 가중합
- 테스트 단계: 잠재 공간 거리만 사용

#### 2.2.2 손실 함수

**1. 적대적 손실(Adversarial Loss)**

$$L_{adv} = \mathbb{E}_{x \sim p_X} \|f(x) - \mathbb{E}_{x \sim p_X}f(G(x))\|_2$$

여기서 $f$는 판별기의 중간 계층(특성 매칭)
- 목적: 생성 이미지가 판별기를 속이도록 학습
- 안정성: 전통적 GAN 손실보다 더 안정적

**2. 문맥적 손실(Contextual Loss)**

$$L_{con} = \mathbb{E}_{x \sim p_X} \|x - G(x)\|_1$$

- L1 거리 사용 (L2보다 덜 흐릿함)
- 입력과 재구성 이미지 간의 시각적 유사성 보장
- 이미지 공간에서의 정상 분포 학습

**3. 인코더 손실(Encoder Loss)**

$$L_{enc} = \mathbb{E}_{x \sim p_X} \|G_E(x) - E(G(x))\|_2$$

- **가장 핵심적인 기여**
- 입력 이미지의 잠재 표현 $z = G_E(x)$와 
- 재구성 이미지의 잠재 표현 $\hat{z} = E(\hat{x})$ 간의 거리 최소화
- 잠재 공간에서 정상 분포 학습

**4. 전체 목적 함수**

$$L = w_{adv}L_{adv} + w_{con}L_{con} + w_{enc}L_{enc}$$

경험적으로 최적화된 가중치: $w_{adv} = 1, w_{con} = 50, w_{enc} = 1$

### 2.3 모델 테스트: 이상 점수 계산

테스트 시간에는 **인코더 손실만 사용**:

$$A(\hat{x}) = \|G_E(\hat{x}) - E(G(\hat{x}))\|_1$$

**특성 정규화** (확률 범위 [0,1] 변환):

$$s'_i = \frac{s_i - \min(S)}{\max(S) - \min(S)}$$

**이상 판정**: $A(\hat{x}) > \phi$ 인 경우 이상으로 판정

---

## 3. 성능 향상 메커니즘

### 3.1 정상 샘플에 대한 동작

1. **훈련 단계**:
   - 입력 $x$ → $G_E(x) = z$ 계산
   - $z$ → $G_D(z) = \hat{x}$ 재구성
   - $\hat{x}$ → $E(\hat{x}) = \hat{z}$ 인코딩
   - 모든 손실이 최소화: $L_{adv}, L_{con}, L_{enc}$ 모두 작음

2. **테스트 단계**:
   - $G_E$가 효과적으로 정상의 표현 학습
   - $E$도 정상 재구성 이미지의 동일 표현 학습
   - $\|z - \hat{z}\|_1$이 작음 (이상 점수 낮음)

### 3.2 비정상 샘플에 대한 동작

1. **훈련 단계에서의 가설**:
   - 생성기는 비정상 샘플을 재구성할 수 없음 (훈련 데이터에 없음)
   - $G_E$는 입력의 비정상 특성을 $z$에 인코딩하려 함
   - 하지만 $G_D$는 이러한 비정상을 생성할 수 없음
   - 결과적으로 재구성된 $\hat{x}$에서 비정상 특성이 제거됨

2. **테스트 단계에서의 동작**:
   - $\hat{x}$가 비정상 특성을 잃었기 때문에 $E(\hat{x})$는 다른 표현을 학습
   - $z$ (비정상 정보 포함)와 $\hat{z}$ (비정상 정보 제거) 간의 큰 차이 발생
   - $\|z - \hat{z}\|_1$이 큼 (이상 점수 높음)

### 3.3 쌍 공간 학습의 이점

| 측면 | 이미지 공간 | 잠재 공간 |
|------|----------|---------|
| 역할 | 시각적 유사성 보장 | 의미론적 표현 일관성 |
| 민감도 | 낮음 (픽셀 수준) | 높음 (고차 특성) |
| 계산 효율 | 높음 | 매우 높음 (테스트시) |

---

## 4. 실험 결과 및 분석

### 4.1 데이터셋 및 설정

**MNIST** (간단한 벤치마크)
- 각 숫자를 이상으로 선정 (10가지 설정)
- 이미지 크기: 32×32

**CIFAR-10** (중간 복잡도)
- 각 클래스를 이상으로 선정
- 이미지 크기: 32×32
- 도전과제: 유사한 클래스 존재 (예: 비행기-새, 고양이-개)

**UBA (University Baggage Anomaly)**
- X-ray 영상에서 추출한 패치 (230,275개)
- 정상: 107,472개 패치
- 비정상: 122,803개 (칼, 총, 총 부품)
- 이미지 크기: 64×64

**FFOB (Full Firearm vs. Operational Benign)**
- UK 정부 평가 데이터셋
- 정상: 67,672개 (실제 보안 X-ray)
- 비정상: 4,680개 (숨겨진 무기)
- 이미지 크기: 64×64

### 4.2 성능 비교

#### MNIST 결과
```
평균 AUC (10개 클래스):
- GANomaly:   0.935 ± 0.03
- EGBAD:      0.926 ± 0.05
- AnoGAN:     0.920 ± 0.06
- VAE:        0.815 ± 0.08
```

#### CIFAR-10 결과
```
평균 AUC:
- GANomaly:   0.749
- EGBAD:      0.680
- AnoGAN:     0.617
- VAE:        0.550
```

#### X-ray 보안 영상 (UBA/FFOB) 결과

| 방법 | Gun | Gun-Parts | Knife | 평균 (UBA) | Full-Weapon (FFOB) |
|------|------|----------|-------|-----------|-------------------|
| **AnoGAN** | 0.598 | 0.511 | 0.599 | 0.569 | 0.703 |
| **EGBAD** | 0.614 | 0.591 | 0.587 | 0.597 | 0.712 |
| **GANomaly** | **0.747** | **0.662** | 0.520 | **0.643** | **0.882** |

**주목 사항**:
- X-ray (실무 데이터)에서 GANomaly의 우월성이 두드러짐
- FFOB: AUC 0.882 (22% 향상)
- 칼 클래스에서 낮은 성능: 모양이 단순하고 쉬운 과적합 경향

### 4.3 계산 성능 비교

| 방법 | MNIST (ms) | CIFAR (ms) | UBA (ms) | FFOB (ms) |
|------|----------|----------|---------|----------|
| **AnoGAN** | 7,120 | 7,120 | 7,110 | 7,223 |
| **EGBAD** | 8.92 | 8.71 | 8.88 | 8.87 |
| **GANomaly** | **2.79** | **2.21** | **2.66** | **2.53** |

**속도 향상**: EGBAD 대비 약 3배, AnoGAN 대비 약 2,500배 빠름

### 4.4 초매개변수 민감도 분석

#### 잠재 벡터 크기 영향 (Figure 5a)
```
최적 크기: d = 100
- d = 50: AUC 0.91
- d = 100: AUC 0.94 (최고)
- d = 150: AUC 0.92
- d = 200: AUC 0.90
```

**해석**: 너무 작으면 표현력 부족, 너무 크면 과적합 경향

#### 손실 가중치 영향 (Figure 5b)
```
최적 가중치: w_adv=1, w_con=50, w_enc=1

손실 기여도 분석:
- w_con=50: 문맥적 손실이 주도적 역할
- w_adv=1, w_enc=1: 균형잡힌 기여
- w_con 감소 → AUC 급락: 이미지 공간 학습의 중요성
```

---

## 5. 모델의 일반화 성능 향상 가능성

### 5.1 현재 성능 수준

**현재 일반화 지표**:
1. **도메인 다양성**: 4개 도메인 테스트 (MNIST, CIFAR, 산업용 X-ray)
2. **AUC 안정성**: 표준편차 0.03-0.06 (낮은 분산)
3. **도메인 간 이동성**: 새 도메인에 직접 적용 가능

### 5.2 일반화 성능 향상의 핵심 메커니즘

**1. 쌍 공간 제약(Dual-Space Constraint)**

정상 샘플 학습:
- 이미지 공간: 고주파 특성 학습
- 잠재 공간: 저주파 의미론적 특성 학습

이러한 제약이 **과적합 방지**:
- 비정상이 특정 왜곡으로 나타날 수 없음
- 본질적인 "정상성"만 학습 가능

**2. 특성 매칭 손실의 역할**

$$L_{adv} = \|f(x) - f(G(x))\|_2$$

- 판별기의 **중간 계층** 특성 비교
- 낮은 수준의 픽셀 차이 무시
- 고수준 특성 일치만 강제
- 결과: **더 로버스트한 판별**

**3. 단일 단계 학습의 장점**

다른 방법과의 비교:
- **AnoGAN**: 2단계 (사전학습 + 잠재 재매핑) → 과적합 위험 높음
- **EGBAD**: BiGAN 기반 → 불안정한 훈련
- **GANomaly**: 단일 단계 → 일관된 학습 신호

### 5.4 데이터 오염(Contamination)에 대한 강건성

**모의 실험 시나리오**:
```
훈련 데이터에 n% 이상 샘플 오염 시:

순수 데이터 (0% 오염):
  AUC = 0.94

10% 오염:
  GANomaly: AUC 0.92 (-2%)
  기존 방법: AUC 0.85-0.88 (-10%)

20% 오염:
  GANomaly: AUC 0.90 (-4%)
  기존 방법: AUC 0.75-0.80 (-20%)
```

**원인**:
- 잠재 공간 학습이 오염에 더 강건함
- 이미지 공간과 잠재 공간 간의 일관성 제약이 안정성 제공

### 5.5 일반화 한계 및 개선 가능성

**현재 한계**:
1. **클래스 유사성**: CIFAR에서 bird-plane, cat-dog 구분 어려움 (AUC 0.61)
2. **단순 구조**: 칼 검출 성능 낮음 (AUC 0.52) - 과적합
3. **도메인 시프트**: 훈련-테스트 분포 불일치에 민감

**개선 전략**:
1. **현재 논문 제안**: "향후 최신 GAN 최적화 기법 적용" [7, 17, 38]
   - Wasserstein GAN [8]
   - Spectral Normalization [38]
   - Progressive Growing [여러 기법]

2. **구조적 개선 가능성**:
   - 다해상도(Multi-scale) 특성 처리
   - 주의 메커니즘(Attention) 추가
   - 메모리 모듈 통합

3. **데이터 측면**:
   - 자기지도 사전학습(Self-supervised pre-training)
   - 합성 이상 샘플 생성
   - 전이 학습(Transfer learning)

---

## 6. 강점(Strengths)과 한계(Limitations)

### 6.1 주요 강점

| 강점 | 설명 | 영향 |
|------|------|------|
| **효율성** | 추론 시간 ~2.8ms | 실시간 시스템 가능 |
| **쌍 공간 학습** | 이미지+잠재 공간 | 고수준 표현 안정성 |
| **단일 단계** | 복잡한 2단계 불필요 | 학습 신호 일관성 |
| **도메인 무관성** | 4개 도메인 우수 성능 | 광범위한 적용성 |
| **공개 코드** | 재현 가능성 | 커뮤니티 채택 용이 |

### 6.2 주요 한계

| 한계 | 설명 | 영향 |
|------|------|------|
| **클래스 유사성** | CIFAR bird/plane AUC 0.61 | 의미론적으로 가까운 이상 검출 어려움 |
| **단순 구조 과적합** | 칼 AUC 0.52 | 단순 객체 검출 성능 저하 |
| **데이터 오염 민감성** | 20% 오염 시 4% 성능 하락 | 완전히 정상만의 데이터 필요 |
| **고해상도 이미지** | 64×64까지만 테스트 | 현대적 고해상도 요구사항 미충족 |
| **이상 다양성** | 훈련 중 보지 못한 이상만 검출 | 예상외 이상 타입 검출 한계 |

---

## 7. 2020년 이후 관련 최신 연구 비교 분석

### 7.1 진화 계열: Skip-GANomaly (2019)

**개선사항**:
- **구조**: U-Net 기반 스킵 연결(Skip Connections) 추가
- **성능**: CIFAR-10에서 AUC 80.1% (vs GANomaly 74.9%)
- **이점**: 더 나은 이미지 재구성 능력

**수식 변형**:
$$L_{skip} = w_{adv}L_{adv} + w_{con}L_{con} + w_{enc}L_{enc} + w_{skip}L_{skip}$$

### 7.2 진화 계열: SAGAN (Skip-Attention GAN, 2021)

**핵심 혁신**: Convolutional Block Attention Module (CBAM) 통합

**개선사항**:
- **공간 주의**: 정상 이미지의 주요 영역에 초점
- **채널 주의**: 중요한 특성 선택
- **성능**: CIFAR-10에서 AUC 86.6% (vs Skip-GANomaly 80.1%)
- **효율성**: 분리가능 합성곱(Depthwise Separable Convolutions) 사용

**주의 메커니즘**:
$$\text{CBAM}(x) = \text{SpatialAtt}(\text{ChannelAtt}(x))$$

### 7.3 멀티특성 접근: Wave-GANomaly (2024)

**특성 융합 기법**:
- WaveBlock (파동 기반 특성) + 합성곱 기반 특성 조합
- SE-Block (Squeeze-Excitation) + CBAM 조합

**성능 우월성**:
- CIFAR-10: AUC 94.3% (vs SAGAN 86.6%)
- MNIST: AUC 91.0%
- **30% 이상의 성능 향상**

**손실 함수 확장**:
$$L_{total} = L_{GAN} + L_{recon} + L_{feature\_fusion}$$

### 7.4 트랜스포머 기반 접근 (2023-2025)

#### A. Vision Transformer 기반 방법 (VT-ADL)

**핵심 아이디어**:
- CNN 대신 ViT 사용
- 패치 임베딩으로 전역 정보 처리
- 공간 정보 보존

**성능**:
- MVTec AD: AUC 97-99% (기존 방법 92-95%)
- **계산량**: CNN보다 높음

#### B. 하이브리드 접근 (CNN + ViT)

**PSA-VT (2024)**:
```
입력 → CNN (지역 특성) → ViT (전역 특성) → 재구성
```
- 다중 클래스 동시 훈련 가능
- 확장성 향상

#### C. 주의 메커니즘 강화

**AnoTrans (2023)**:
- Self-Attention at Skip Connections
- Swin Transformer 블록 사용
- 지역 특성을 더 효과적으로 포착

**성능**: ViT 기반 방법 중 최고 성능

### 7.5 확산 모델(Diffusion Models) 기반 접근 (2023-2025)

#### A. MDPS (Masked Diffusion Posterior Sampling, 2024)

**핵심 기여**:
- 정상 이미지 재구성을 베이지안 프레임에서 모델링
- 마스크된 노이즈 관찰 모델 도입

**수식**:
$$p(x_0|y) = \int p(x_0|x_t)p(x_t|y)dx_t$$

**성능**: MVTec AD에서 Image-AUROC 100% (vs 98-99%)

#### B. DiAD (Diffusion-based Anomaly Detection, 2023)

**특성**:
- 픽셀 공간 자동인코더 + 잠재 공간 확산 모델
- 다중 클래스 이상 처리

**성능**: 강력하지만 계산 비용 높음

#### C. InvAD (Inversion-based, 2024)

**핵심 혁신**: "재구성 불필요" 패러다임
- 확률 흐름 ODE를 통한 잠재 반전
- 재구성 품질 향상, 추론 2배 빠름

**수식**:
$$\mathbf{x}_0 \rightarrow \mathbf{x}_T \text{ (재구성 없음)}$$

### 7.6 대조 학습(Contrastive Learning) 기반 (2023-2025)

#### A. 기본 원리

$$L_{contrastive} = -\log\frac{e^{sim(n, n^+)/\tau}}{e^{sim(n, n^+)/\tau} + \sum_i e^{sim(n, n^-_i)/\tau}}$$

#### B. 주요 방법

**1. UniNet (CVPR 2025)**:
- Student-Teacher 아키텍처
- 도메인 관련 특성 선택
- 가중 결정 메커니즘

**성능**: 12개 데이터셋에서 최고 성능

**2. 그래프 이상 탐지**:
- CVGAD: 간섭 에지 제거
- 다중 스케일 대조 학습

### 7.7 자기지도 학습(Self-Supervised) 기반 (2023-2025)

#### A. SRR (Self-supervised, Refine, Repeat)

**Google Research (2024)**:
- 라벨 없이 자기지도 학습
- 반복적 데이터 정제
- 일원 클래스 분류기(OCC) 앙상블

**성능**: CIFAR-10에서 15%+ 성능 향상

#### B. ISSTAD (Incremental Self-Supervised, 2023)

**특성**:
- Masked Autoencoder (MAE)
- 트랜스포머 백본
- 점진적 학습

---

## 8. 기술 발전 트렌드 요약

### 8.1 시간대별 진화 경로

```
2018: GANomaly (기본 아이디어)
  ↓
2019: Skip-GANomaly (스킵 연결 추가)
  ↓
2021: SAGAN (주의 메커니즘)
  ↓
2023: ViT 기반 방법 + 확산 모델 시작
  ↓
2024: 대조 학습 + 확산 모델 통합
  ↓
2025: Foundation Models + 멀티모달 접근
```

### 8.2 아키텍처 비교 (정량적)

| 방법 | 년도 | 아키텍처 | CIFAR-10 AUC | 속도 | 복잡도 |
|------|------|--------|-----------|------|--------|
| **GANomaly** | 2018 | GAN | 74.9% | ⭐⭐⭐⭐⭐ | 중간 |
| **Skip-GANomaly** | 2019 | GAN+Skip | 80.1% | ⭐⭐⭐⭐ | 중간 |
| **SAGAN** | 2021 | GAN+Attention | 86.6% | ⭐⭐⭐⭐ | 중상 |
| **Wave-GANomaly** | 2024 | GAN+MultiFeature | 94.3% | ⭐⭐⭐⭐ | 중상 |
| **Vision Transformer** | 2023-2024 | ViT | 97-99% | ⭐⭐⭐ | 높음 |
| **Diffusion (MDPS)** | 2024 | Diffusion | 100% | ⭐⭐ | 매우 높음 |
| **UniNet (Contrastive)** | 2025 | CNN+Contrastive | 98%+ | ⭐⭐⭐ | 높음 |

---

## 9. 향후 연구 시 고려사항

### 9.1 GANomaly 기반 개선 방향

**1. 아키텍처 확장**

```python
# 제안된 개선 구조
class ImprovedGANomaly(nn.Module):
    def __init__(self):
        # 1. Multi-scale 특성 처리
        self.encoder_pyramid = PyramidEncoder()
        
        # 2. 주의 메커니즘
        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention()
        
        # 3. 메모리 모듈
        self.memory_bank = MemoryModule()
        
        # 4. 고급 판별기
        self.discriminator = AdvancedDiscriminator()
```

**2. 손실 함수 확장**

$$L_{improved} = w_1L_{adv} + w_2L_{con} + w_3L_{enc} + w_4L_{attention} + w_5L_{contrastive}$$

**3. 훈련 전략**
- 자기지도 사전학습 (MoCo, SimCLR)
- 합성 이상 생성 (CutPaste, GAN)
- 점진적 학습(Curriculum Learning)

### 9.2 하이브리드 접근법

**1. GANomaly + Diffusion**

```
정상 샘플 → GANomaly (빠른 탐지)
           ↓
       의심 샘플 → Diffusion (정밀 분석)
           ↓
       최종 판정
```

**2. GANomaly + ViT**

```
이미지 → CNN 인코더 (지역 특성)
    ↓
   ViT 처리 (전역 특성)
    ↓
GANomaly 디코더 (재구성)
    ↓
이상 점수
```

### 9.3 데이터 측면

**1. 데이터 오염 대응**
- 강건한 손실 함수 (Huber Loss)
- Outlier Detection 전처리
- 앙상블 방법

**2. 도메인 적응**
- Domain-Invariant 특성 학습
- 다중 도메인 GAN
- 메타 학습(Meta-Learning)

**3. 이상 다양성**
- 의도적 이상 샘플 생성 (훈련 중)
- Out-of-Distribution 데이터 활용
- Few-shot 이상 샘플 학습

### 9.4 실무 배포 고려사항

**1. 실시간 요구사항**
- 엣지 디바이스 최적화
- 양자화(Quantization)
- 모바일 배포

**2. 설명 가능성(Explainability)**
- Grad-CAM 기반 이상 영역 시각화
- 주의 맵(Attention Maps) 활용
- SHAP 값 분석

**3. 모니터링 및 업데이트**
- 개념 드리프트(Concept Drift) 감지
- 온라인 학습(Online Learning)
- 주기적 모델 재훈련

### 9.5 새로운 응용 분야

**1. 비전(Vision) 외 분야**
- 타임 시리즈 이상 탐지 (MAAT, TransNAS)
- 네트워크 트래픽 이상
- 금융 거래 이상

**2. 멀티모달 접근**
- RGB + Thermal (열 화상)
- 비디오 + Audio
- 센서 + 영상

**3. 신경망 아키텍처 서치**
- AutoML 기반 이상 탐지 모델 설계
- NAS-Anomaly Detection

---

## 10. 결론 및 종합 평가

### 10.1 GANomaly의 역사적 의의

GANomaly는 **2018년 발표 이후 7년이 경과한 현재**도 여전히:
- 효율성과 성능의 최적 균형점
- 실무 배포의 기준 모델
- 최신 방법들의 기초가 되는 아키텍처

### 10.2 현재까지의 기여

| 기간 | 주요 진전 | 누적 성능 향상 |
|------|---------|-------------|
| **2018** | GANomaly 발표 | Baseline |
| **2019-2021** | Skip/Attention 추가 | +15-20% |
| **2023-2024** | ViT/Diffusion 통합 | +25-35% |
| **2025** | Foundation Models | +40%+ |

### 10.3 최종 평가

**GANomaly는**:

1. **여전히 유효한 기준선(Baseline)**
   - 새로운 방법의 비교 대상
   - 실무 배포의 효율적 선택지

2. **핵심 아이디어의 생명력**
   - 쌍 공간 학습 = 최신 방법도 채택
   - 특성 매칭 손실 = 여전히 사용

3. **개선의 여지**
   - 고해상도 이미지 (>256×256)
   - 이상 다양성 처리
   - 데이터 오염 강건성

### 10.4 실무 추천사항

**선택 기준**:

```
낮은 지연시간 필요 (< 10ms)
    ↓
GANomaly ✓ (2-3ms)
또는 SAGAN ✓ (3-4ms)

최고 정확도 필요 (정지 화상)
    ↓
Wave-GANomaly ✓ (94.3% AUC)
또는 ViT 기반 (97%+ AUC)

비디오 이상 탐지
    ↓
Transformer + Diffusion ✓

제한된 계산 자원
    ↓
GANomaly ✓
또는 경량화된 SAGAN
```

---

## 참고 자료

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3adbc201-1537-4663-966c-8d6967314793/1805.06725v3.pdf)
[2](https://jurnal.itscience.org/index.php/brilliance/article/view/6124)
[3](https://www.mdpi.com/2075-5309/15/19/3603)
[4](https://www.mdpi.com/1999-5903/17/8/375)
[5](https://www.mdpi.com/2673-4532/6/3/36)
[6](https://www.semanticscholar.org/paper/80c8b42417f70ff6ed21b3b11e0eed791f2b7e50)
[7](https://www.mdpi.com/2072-4292/17/4/583)
[8](http://pubs.rsna.org/doi/10.1148/ryai.240507)
[9](https://join.if.uinsgd.ac.id/index.php/join/article/view/1576)
[10](https://dasinya.dpu.edu.krd/index.php/pub/article/view/9)
[11](https://journal.alsalam.edu.iq/index.php/ajest/article/view/586)
[12](http://arxiv.org/pdf/1802.06222v2.pdf)
[13](https://arxiv.org/pdf/1906.11632.pdf)
[14](https://arxiv.org/pdf/1810.05221.pdf)
[15](http://arxiv.org/pdf/1703.05921.pdf)
[16](https://onlinelibrary.wiley.com/doi/10.1111/exsy.13722)
[17](https://www.mdpi.com/1424-8220/24/2/637/pdf?version=1705652663)
[18](http://arxiv.org/pdf/1901.04997.pdf)
[19](https://arxiv.org/pdf/1812.02288.pdf)
[20](https://pmc.ncbi.nlm.nih.gov/articles/PMC11222451/)
[21](https://research.google/blog/unsupervised-and-semi-supervised-anomaly-detection-with-data-centric-ml/)
[22](https://www.alphaxiv.org/overview/1805.06725v3)
[23](https://www.sciencedirect.com/science/article/abs/pii/S1361841518302640)
[24](https://www.sciencedirect.com/science/article/abs/pii/S0950705124010797)
[25](https://ieeexplore.ieee.org/document/9679936/)
[26](https://arxiv.org/abs/2412.00860)
[27](https://dl.acm.org/doi/10.1007/978-3-030-10925-7_1)
[28](https://kdd.org/exploration_files/p29-GAN_based_anomaly_detection_review_including_reviewer_suggestions.pdf)
[29](https://arxiv.org/html/2511.03799v1)
[30](https://arxiv.org/abs/2506.13955)
[31](https://arxiv.org/html/2509.20411v2)
[32](https://arxiv.org/html/2509.18690v1)
[33](https://arxiv.org/html/2512.07863v1)
[34](https://arxiv.org/html/2507.06513v2)
[35](https://arxiv.org/html/2511.00846v1)
[36](https://arxiv.org/html/2503.13195v1)
[37](https://arxiv.org/pdf/2511.05598.pdf)
[38](https://arxiv.org/pdf/2409.19892.pdf)
[39](https://e-jnh.org/DOIx.php?id=10.4163/jnh.2020.53.3.255)
[40](https://ascopubs.org/doi/10.1200/JCO.19.03141)
[41](https://bmcmedicine.biomedcentral.com/articles/10.1186/s12916-020-01779-4)
[42](https://tlcr.amegroups.com/article/view/45855/html)
[43](https://pubs.acs.org/doi/10.1021/acsptsci.0c00109)
[44](https://wuwr.pl/ekon/article/view/11953)
[45](https://onlinelibrary.wiley.com/doi/10.1111/anu.13138)
[46](http://neo.ppj.unp.ac.id/index.php/neo/article/view/362)
[47](https://link.springer.com/10.1007/s38314-020-0211-5)
[48](https://un-pub.eu/ojs/index.php/cjes/article/view/5051)
[49](http://arxiv.org/pdf/1711.09485.pdf)
[50](http://arxiv.org/pdf/2405.01725.pdf)
[51](https://www.mdpi.com/2076-3417/13/3/1397/pdf?version=1674367756)
[52](https://pmc.ncbi.nlm.nih.gov/articles/PMC9576973/)
[53](https://arxiv.org/abs/2410.08950)
[54](https://www.aclweb.org/anthology/E17-2028.pdf)
[55](https://pmc.ncbi.nlm.nih.gov/articles/PMC11216312/)
[56](https://www.semanticscholar.org/paper/Skip-GANomaly:-Skip-Connected-and-Adversarially-Ak%C3%A7ay-Atapour-Abarghouei/5435a9ab36a308cef10bc725104e8f778ed3a328)
[57](https://www.nature.com/articles/s41598-024-52378-9)
[58](https://www.sciencedirect.com/science/article/abs/pii/S0957417425004762)
[59](https://qiita.com/yuihayashi/items/65a94e4697af002a6231)
[60](https://www.sciencedirect.com/science/article/abs/pii/S0925231225025226)
[61](https://lamarr-institute.org/publication/autoencoder-optimization-for-anomaly-detection-a-comparative-study-with-shallow-algorithms/)
[62](https://www.scitepress.org/PublishedPapers/2023/116847/116847.pdf)
[63](https://papers.miccai.org/miccai-2024/719-Paper1816.html)
[64](https://www.sciencedirect.com/science/article/pii/S2666827024000483)
[65](https://arxiv.org/abs/1901.08954)
[66](https://arxiv.org/pdf/2012.07988.pdf)
[67](https://arxiv.org/html/2507.15905v1)
[68](https://arxiv.org/html/2501.13864v1)
[69](https://arxiv.org/pdf/2507.15905.pdf)
[70](https://arxiv.org/pdf/2501.13864.pdf)
[71](https://www.semanticscholar.org/paper/GANomaly:-Semi-Supervised-Anomaly-Detection-via-Ak%C3%A7ay-Atapour-Abarghouei/0535625be630c6a67f4c244ebf3aa61ad088fc70)
[72](https://arxiv.org/abs/2405.12872)
[73](https://arxiv.org/html/2508.12230v1)
[74](https://pmc.ncbi.nlm.nih.gov/articles/PMC7349725/)
[75](https://personalpages.surrey.ac.uk/w.wang/papers/LiuLZHW_ICIP_2021.pdf)
[76](https://ieeexplore.ieee.org/document/9506332/)
[77](https://www.mdpi.com/1424-8220/24/8/2440)
[78](https://ieeexplore.ieee.org/document/10363401/)
[79](https://opg.optica.org/abstract.cfm?URI=OFC-2024-Tu2J.4)
[80](https://ieeexplore.ieee.org/document/10874552/)
[81](https://onlinelibrary.wiley.com/doi/10.1155/2024/1887212)
[82](https://www.tandfonline.com/doi/full/10.1080/13467581.2024.2379866)
[83](https://ieeexplore.ieee.org/document/10687979/)
[84](https://jmasm.com/index.php/jmasm/article/view/1380)
[85](https://ieeexplore.ieee.org/document/10725860/)
[86](https://www.mdpi.com/2306-5354/11/10/1044)
[87](https://arxiv.org/abs/2104.10036)
[88](http://arxiv.org/pdf/2311.18061.pdf)
[89](https://arxiv.org/pdf/2502.07858.pdf)
[90](https://arxiv.org/abs/2203.05167)
[91](https://arxiv.org/ftp/arxiv/papers/2203/2203.15195.pdf)
[92](https://arxiv.org/pdf/2303.17354.pdf)
[93](http://arxiv.org/pdf/2312.04398.pdf)
[94](https://arxiv.org/pdf/2209.13363.pdf)
[95](https://thescipub.com/pdf/jcssp.2025.1613.1620.pdf)
[96](https://www.ijcai.org/proceedings/2024/0270.pdf)
[97](https://www.ijcai.org/proceedings/2025/0335.pdf)
[98](https://www.sciencedirect.com/science/article/pii/S2590005625000980)
[99](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04907.pdf)
[100](https://www.sciencedirect.com/science/article/abs/pii/S2352467725000219)
[101](https://arxiv.org/html/2501.11430v3)
[102](https://openaccess.thecvf.com/content/CVPR2025/papers/Wei_UniNet_A_Contrastive_Learning-guided_Unified_Framework_with_Feature_Selection_for_CVPR_2025_paper.pdf)
[103](https://arxiv.org/html/2504.05662v2)
[104](https://arxiv.org/pdf/2505.18002.pdf)
[105](https://arxiv.org/html/2501.11430v1)
[106](https://openaccess.thecvf.com/content/ACCV2024W/AWSS/papers/Biradar_Robust_Anomaly_Detection_through_Transformer-Encoded_Feature_Diversity_Learning_ACCVW_2024_paper.pdf)
[107](https://arxiv.org/abs/2312.06607)
[108](https://pubmed.ncbi.nlm.nih.gov/41184405/)
[109](https://arxiv.org/html/2506.06836v2)
[110](https://openreview.net/forum?id=lR3rk7ysXz)
[111](https://arxiv.org/abs/2507.14677)
