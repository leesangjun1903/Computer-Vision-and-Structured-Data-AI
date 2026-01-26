

# MaskGIT: Masked Generative Image Transformer

## I. 핵심 주장 및 주요 기여

### 1. 기본 주장

MaskGIT는 기존 생성 트랜스포머의 근본적 설계 문제를 지적합니다. 종래의 자동회귀(autoregressive) 방식은 이미지를 1D 순서대로 순차 생성하는데, 이는 **이미지의 2D 본질과 불일치**합니다. 저자들은 그림 제작 과정에 비유하며, 화가가 전체 스케치부터 시작해 점진적으로 세부사항을 정제하는 방식이 더 자연스럽다고 주장합니다.[1]

### 2. 주요 기여

MaskGIT의 세 가지 핵심 기여는 다음과 같습니다:

1. **양방향 어텐션 기반 병렬 디코딩**: 모든 이미지 토큰을 동시에 생성한 후 반복적으로 정제
2. **마스크 스케줄 설계**: 코사인 함수 기반 최적의 마스킹 비율을 제시
3. **효율성-품질 균형**: 자동회귀 모델 대비 **64배 빠른 추론**으로 SOTA 품질 달성[1]

***

## II. 해결하고자 하는 문제

### 1. 기존 방법의 한계

**자동회귀(AR) 모델의 문제점:**[1]
- **시간 복잡도**: 256×256 이미지 = 256개 토큰 시퀀스로, 생성 시간이 O(N²)에 비례
- **속도**: 32×32 토큰 생성에 약 30초 소요 (GPU 기준)
- **컨텍스트 불일치**: 순차적 생성 순서가 이미지의 본질적 구조와 맞지 않음
- **오류 누적**: 순차 생성의 순서 편향(exposure bias) 문제

### 2. 제안된 해결책의 범위

MaskGIT는 다음 세 가지 문제를 동시에 해결합니다:

1. **계산 효율성**: 256 자동회귀 스텝을 8-12 반복 디코딩으로 축소
2. **생성 품질**: 양방향 컨텍스트로 더 충실한 이미지 생성
3. **작업 유연성**: 단일 모델로 생성, 인페인팅, 아웃페인팅, 편집 지원[1]

***

## III. 제안하는 방법 (수식 포함)

### 1. MVTM 훈련 단계

MaskGIT는 Masked Visual Token Modeling (MVTM) 학습 목표를 사용합니다:

$$L_{\text{mask}} = -\mathbb{E}_{Y \sim D} \left[ \sum_{i \in [1,N], m_i=1} \log p(y_i | Y_M) \right]$$

여기서:
- $Y = [y_1, y_2, ..., y_N]$: VQ-인코더로부터의 잠재 토큰
- $M = [m_1, m_2, ..., m_N]$: 이진 마스크 ($m_i = 1$이면 $[MASK]$ 토큰으로 교체)
- $Y_M$: 마스크 적용 후의 결과
- $p(y_i | Y_M)$: 양방향 트랜스포머의 예측 확률

**마스크 샘플링 과정:**[1]
1. 비율 $r \sim \text{Uniform}[0,1)$ 샘플링
2. $\lfloor \gamma(r) \cdot N \rfloor$개의 토큰을 균일하게 선택해 마스크 적용
3. 양방향 트랜스포머으로 마스킹된 토큰들의 분포 학습

### 2. 반복적 디코딩 알고리즘

추론 단계에서 MaskGIT는 T 스텝의 반복적 디코딩을 수행합니다:[1]

**스텝 t (t=0부터 T-1):**

1. **예측(Predict)**: 마스킹된 모든 위치에 대해 예측 확률 계산
$$p^{(t)} = \text{Transformer}(Y_M^{(t)}) \in \mathbb{R}^{N \times K}$$

2. **샘플링(Sample)**: 각 마스킹 위치 i에서 토큰 샘플링
$$y_i^{(t)} \sim p_i^{(t)}, \quad c_i = \max(p_i^{(t)})$$

여기서 $c_i$는 신뢰도 점수로 사용됨 (마스킹되지 않은 위치는 $c_i = 1.0$)

3. **마스크 스케줄**: 생성할 토큰 수 계산
$$n = \lfloor \gamma(t/T) \cdot N \rfloor$$

4. **마스킹 업데이트**: 신뢰도 상위 n개 토큰 유지

$`m_i^{(t+1)} = \begin{cases}
1 & \text{if } c_i < \text{sorted}_{j}[c_j][n] \\
0 & \text{otherwise}
\end{cases}`$

### 3. 마스크 스케줄 함수

마스크 스케줄 함수 $\gamma(r)$은 다음 속성을 만족해야 합니다:[1]
- $\gamma(r) \in (0,1]$ for $r \in $[1]
- 단조 감소: $\gamma(0) \to 1$, $\gamma(1) \to 0$

**최고 성능의 코사인 함수:**
$$\gamma_{\text{cosine}}(r) = \cos\left(\frac{\pi r}{2}\right)$$

**다른 함수 후보들:**
- 제곱: $\gamma_{\text{square}}(r) = (1-r)^2$
- 선형: $\gamma_{\text{linear}}(r) = 1-r$
- 제곱근: $\gamma_{\text{sqrt}}(r) = \sqrt{1-r}$

실험 결과, 코사인 함수가 모든 지표에서 최고 성능을 보임.[1]

***

## IV. 모델 구조

### 1. 2단계 아키텍처

**Stage 1 - 토큰화 (VQGAN 활용):**[1]
- 인코더 E: 이미지 $x \in \mathbb{R}^{H \times W \times 3}$ → 잠재 임베딩
- 양자화: 1024개 코드북 토큰 $e_k \in \mathbb{R}^D$
- 디코더 G: 토큰 시퀀스 → 복원된 이미지

**Stage 2 - 생성 (양방향 트랜스포머):**[1]
- 24개 트랜스포머 블록
- 8개 어텐션 헤드
- 768차원 임베딩, 3072차원 피드포워드
- 학습 가능한 2D 위치 임베딩
- LayerNorm 정규화, 드롭아웃 = 0.1

### 2. 핵심 아키텍처 특징

**양방향 자기-어텐션:**
$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

모든 토큰이 서로 어텐션 가능 (마스킹 없음)

**학습 하이퍼파라미터:**[1]
- Adam 옵티마이저: β₁=0.9, β₂=0.96
- 배치 크기: 256
- 장비: TPU v4 4×4
- 훈련: ImageNet 300 에포크, Places2 200 에포크

***

## V. 성능 향상 분석

### 1. ImageNet 256×256에서의 정량적 성능

| 지표 | MaskGIT | VQGAN | BigGAN | ADM | VQVAE-2 |
|------|---------|-------|---------|------|---------|
| FID ↓ | **6.18** | 15.78 | 6.95 | 10.94 | 31.11 |
| IS ↑ | **182.1** | 78.3 | 198.2 | 101.0 | ~45 |
| CAS Top-1 ↑ | **63.14%** | 53.10% | 43.99% | - | 54.83% |
| 생성 스텝 | **8** | 256 | 1 | 250 | 5120 |
| 매개변수 | 227M | 227M | 160M | 554M | 13.5B |

**주요 개선사항:**[1]
- VQGAN 대비 FID 61% 향상 (15.78 → 6.18)
- BigGAN과 경쟁 수준의 품질, 256배 효율성 향상
- 자동회귀 모델 대비 **32배 속도 향상**

### 2. ImageNet 512×512에서의 성능

| 지표 | MaskGIT | BigGAN | VQGAN* | ADM |
|------|---------|---------|---------|------|
| FID ↓ | **7.32** | 8.43 | 26.52 | 23.24 |
| IS ↑ | 156.0 | 232.5 | 66.8 | 58.06 |
| CAS Top-1 ↑ | **63.43%** | 44.02% | 51.29% | - |
| 생성 스텝 | **12** | 1 | 1024 | 250 |

**특징:**[1]
- BigGAN 대비 13% FID 개선
- 다양성(CAS)에서 44% 향상
- **60배 속도 향상** (1024 → 12 스텝)

### 3. 추론 속도 분석

**런타임 비교 (Tesla V100 GPU):**[1]

```
ImageNet 256×256:
- VQGAN:    ~30초 (256 스텝)
- MaskGIT:  ~1초  (8 스텝)
- 향상:     30배

ImageNet 512×512:
- VQGAN:    ~120초 (1024 스텝)
- MaskGIT:  ~2초  (12 스텝)
- 향상:     60배
```

### 4. 다양성 평가 (Classification Accuracy Score)

CAS는 생성 이미지로 학습한 ResNet-50 분류기의 정확도로 다양성을 측정합니다:[1]

| 모델 | Top-1 CAS (%) | Top-5 CAS (%) |
|------|---------------|---------------|
| 실제 ImageNet | 76.6 | 93.1 |
| **MaskGIT** | **63.14** | **84.45** |
| VQGAN | 53.10 | 76.18 |
| BigGAN | 43.99 | 67.89 |

MaskGIT은 높은 품질과 우수한 다양성을 동시에 달성합니다.

***

## VI. 모델의 일반화 성능 향상

### 1. 양방향 어텐션의 역할

**핵심 원리:**[1]
- 자동회귀 모델: 토큰 i는 토큰 1부터 i-1까지만 참조
- MaskGIT: 토큰 i는 모든 토큰 (1부터 N까지) 참조 가능

이는 다음과 같은 이점을 제공합니다:
1. **장거리 의존성 포착**: 객체 간 관계를 더 잘 학습
2. **컨텍스트 풍부성**: 좌측, 우측, 상하 모든 방향의 정보 활용
3. **오류 전파 감소**: 순서 의존성 제거로 초기 오류의 영향 감소

### 2. 마스킹 스케줄의 학습 효과

**실험 결과:** 코사인 스케줄이 선형 및 다른 함수보다 우수한 이유:[1]

$$\gamma_{\text{cosine}}(r) = \cos\left(\frac{\pi r}{2}\right)$$

- **초기 단계** (r 작음): $\gamma \approx 1$ → 많은 토큰 마스킹 → "어려운 경우" 먼저 학습
- **후기 단계** (r 크음): $\gamma \approx 0$ → 적은 토큰 마스킹 → 세부사항 정제

이는 **"적게에서 많게"(less-to-more)** 정보 흐름을 구현하며, 실험에서:
- FID 6.06 (최고)
- IS 181.5 (최고)
- NLL 4.22 (최고)

### 3. 토큰 중복성 분석

**중요한 발견:** 이미지는 고도로 중복된 정보를 포함합니다.[1]

PSNR 및 LPIPS를 통한 재구성 실험:
- **95% 마스킹** (5% 토큰만 사용): 기본 형태와 주요 의미 유지
- **90% 마스킹** (10% 토큰만 사용): 명확한 개선점, PSNR > 35dB
- **50% 마스킹** (50% 토큰만 사용): 고품질 재구성 (LPIPS < 0.05)

**의미:** 단 10-20%의 토큰만으로도 의미론적으로 충분한 정보를 담을 수 있음.[1]

### 4. 다중 해상도 확장성

MaskGIT는 다양한 해상도에서 작동합니다:[1]
- 256×256: 기본 해상도 (학습)
- 512×512: 16배 토큰 증가에도 우수한 품질 (FID 7.32)
- 512×2560: 매우 높은 종횡비의 초고해상도 이미지도 생성 가능

**메커니즘:** 동일한 토큰화기와 모델을 재사용하므로, 아키텍처 변경 없이 여러 해상도에 적응[1]

### 5. 다중 작업 일반화

**추가 학습 없이 즉시 지원되는 작업들:**[1]

1. **클래스-조건 편집**: 특정 객체를 새로운 클래스로 변환
2. **인페인팅(Inpainting)**: 마스킹된 중앙 영역 채우기
3. **아웃페인팅(Outpainting)**: 모든 방향으로 이미지 확장
4. **임의 방향 외삽**: 상하좌우 임의 방향으로 확장

이는 **마스킹된 영역을 조건으로 해석**하는 MaskGIT의 근본적 설계 덕분입니다.

***

## VII. 한계 및 실패 사례

### 1. 주요 한계

**논문에서 명시된 실패 사례들:**[1]

| 문제 | 현상 | 원인 | 예시 |
|------|------|------|------|
| 의미론적 변이 | 길이가 긴 아웃페인팅에서 한쪽 끝의 의미 "잊음" | 제한된 어텐션 범위 | 한 쪽 방향에서 합성된 색상이 다른 쪽에서 변경 |
| 경계 왜곡 | 인페인팅/아웃페인팅 경계에서 객체 손상 | 마스크 경계와 생성 영역의 불일치 | 울타리, 건축물의 경계 선이 깨짐 |
| 오버스무딩 | 인간 얼굴, 텍스트, 대칭 패턴의 세부사항 손상 | 강한 정규화 효과 | 얼굴의 미세한 특징, 읽을 수 있는 텍스트 생성 실패 |
| 고해상도 손실 | 512×512에서 고주파 세부사항 손실 | 토큰화 단계의 정보 압축 | 피부 텍스처, 복잡한 패턴 손실 |

### 2. 개선을 위한 제안

저자들이 제시한 향후 작업 방향:[1]

1. **더 큰 어텐션 범위**: 전역 컨텍스트 확대
2. **계층적 마스킹**: 해상도별 다른 스케줄 적용
3. **적응형 마스킹**: 이미지 내용에 따른 동적 조정
4. **다중 스케일 아키텍처**: 세부사항 보존

***

## VIII. 마스크 스케줄 설계 분석

### 1. 함수별 성능 비교

**ImageNet 256×256 Ablation Study 결과:**[1]

| 함수 | FID ↓ | IS ↑ | NLL | 최적 스텝 T | 특징 |
|------|-------|------|-----|-----------|------|
| **코사인** | **6.06** | **181.5** | **4.22** | **10** | 최고 성능, 빠른 수렴 |
| 제곱 | 6.35 | 179.9 | 4.38 | 10 | 코사인과 유사 |
| 입방 | 7.26 | 165.2 | 4.63 | 9 | 과도한 초기 마스킹 |
| 지수 | 7.89 | 156.3 | 4.83 | 8 | 지수 함수, 성능 저하 |
| 선형 | 7.51 | 113.2 | 3.75 | 16 | 중간 성능 |
| 제곱근 | 12.33 | 99.0 | 3.34 | 32 | 볼록 함수, 좋지 않음 |
| 로그 | 29.17 | 47.9 | 3.08 | 60 | 가장 나쁜 성능 |

### 2. 함수 족별 특성

**오목(Concave) 함수들:**
$$\text{Concave}: \text{초기에 높은 마스킹 비율} \to \text{후기에 낮은 마스킹 비율}$$

- 초기 단계에서 모델이 적은 수의 높은 신뢰도 예측 필요
- 후기 단계에서 더 많은 토큰 정제 필요
- 효과: "구조부터 세부사항"의 자연스러운 정제 과정

**선형 함수:**
$$\gamma_{\text{linear}}(r) = 1 - r$$
- 모든 스텝에서 균등한 토큰 수 마스킹
- 성능은 평균 수준

**볼록(Convex) 함수들:**
$$\text{Convex}: \text{초기에 낮은 마스킹 비율} \to \text{후기에 높은 마스킹 비율}$$
- 초기에 많은 예측 필요 → 신뢰도 낮음
- 후기로 갈수록 마스킹 증가 → 오버피팅 유발
- 성능: 가장 나쁨

### 3. 스텝 수(T)에 따른 성능

**중요한 발견: "스위트 스팟" 현상**[1]

Figure 8 우측 그래프에서:
- T=4-6: 불충분한 정제, FID 악화
- **T=8-12: 최적 범위** (코사인의 경우 T=10이 최고)
- T=20+: 오버정제, 성능 저하

**가설:** 과도한 반복 → 모델이 불확신한 예측 유지에 인센티브 → 다양성 감소[1]

***

## IX. 2020년 이후 관련 최신 연구 비교

### 1. 선행 연구 (기초)

#### BERT (2019) - Devlin et al.
**개념:** 마스킹된 언어 모델링
- 양방향 트랜스포머 아키텍처
- 마스킹된 토큰 예측 학습
- **MaskGIT의 기초:** 이미지 도메인으로 개념 이전

#### BEiT (2021) - Bao et al.
**개념:** "BERT Pre-training of Image Transformers"
- 이미지 패치의 마스킹된 자기지도 학습
- 시각 토큰화 개념 도입
- **역할:** MaskGIT의 선행 연구 (표현 학습)

#### VQGAN (2021) - Esser et al.
**개념:** 고품질 이미지 토큰화
- VQ-VAE에 적대적 손실 추가
- 지각적 손실(perceptual loss) 도입
- **역할:** MaskGIT의 토큰화 단계에 직접 사용[1]

### 2. 동시대 및 후속 연구

#### A. Muse (2023) - Chang et al.
**혁신:** 텍스트-이미지 생성으로 MaskGIT 확장

| 항목 | MaskGIT | Muse |
|------|---------|------|
| 입력 조건 | 클래스 레이블 | 텍스트 (T5-XXL LLM) |
| 모델 구조 | 단일 모델 | 기본 + 초고해상도 2단계 |
| ImageNet FID | 6.18 | - |
| CC3M FID | - | **6.06** |
| 추론 스텝 | 8-12 | 8-16 |

**개선점:**
- 사전훈련된 대규모 언어 모델의 풍부한 의미론 활용
- 더 세밀한 개념 바인딩 (objects, spatial relationships)
- 텍스트 기반 이미지 편집 지원[2]

#### B. Token-Critic (2022) - Lezama et al.
**혁신:** 보조 모델로 토큰 샘플링 신뢰도 개선

**방식:**[3]
- Token-Critic 모델: 원본 vs 생성 토큰 구별
- 원본 이미지 패치 학습 데이터로 사용
- 추론 시 신뢰도 높은 토큰만 유지

**성과:**
- MaskGIT 대비 FID 개선: 6.18 → 5.50+ 달성
- 품질-다양성 트레이드오프 개선

#### C. AutoNAT (2024) - Ni et al.
**혁신:** 비자동회귀 트랜스포머의 훈련 및 생성 전략 자동 최적화[4]

**핵심 발견:**[4]
1. NAT의 성능 저하는 내재적 한계가 아님
2. 훈련/생성 전략의 휴리스틱 설계가 최적이 아님
3. 자동 최적화 프레임워크로 큰 개선 가능

**최적화 문제:**
$$\arg\min_{\alpha, \beta, \sigma} \text{FID}(\text{NAT}(\alpha, \beta, \sigma))$$

**성과:**
- ImageNet-256 FID: **4.30** (MaskGIT 6.18 → 30% 개선)
- 확산 모델과 비교하여 **5배 빠름**
- 매개변수 효율성 유지[4]

#### D. Visual Autoregressive Modeling (VAR, 2024) - Tian et al.
**패러다임 혁신:** 자동회귀 학습을 "다음-스케일 예측"으로 재정의[5]

**기존 AR과의 차이:**
```
기존 AR:  1×1 → 1×2 → 1×3 → ... (래스터 스캔)
VAR:      1×1 → 2×2 → 4×4 → ... (해상도 기반)
```

**성과:**
- ImageNet-256 FID: **1.73** (MaskGIT 6.18 → 72% 개선)
- 디퓨전 모델 초월 (처음으로 AR가 우수)
- 확장 법칙 입증: 매개변수 대비 성능 선형 관계[5]

#### E. MAGVIT (2023) - Yu et al.
**영역 확장:** 비디오 생성으로 MaskGIT 개념 확대[6]

**혁신:**
- 3D 토큰화기 (공간-시간 토큰)
- 멀티태스크 마스킹 전략
- 8가지 비디오 생성 작업 단일 모델 지원

**성과:**
- 비디오 FVD에서 새로운 SOTA
- 확산 모델 대비 **100배 빠름**
- 자동회귀 모델 대비 **60배 빠름**[6]

#### F. 마스킹 기반 디퓨전 (Masked Diffusion Transformer, 2023)
**개념:** 디퓨전 프로세스 + 마스킹된 학습[7]

| 항목 | MaskGIT | MDT |
|------|---------|------|
| 기본 패러다임 | 병렬 마스킹 디코딩 | 반복 디퓨전 |
| 훈련 비용 | 기준 | 25% 감소 |
| FID | 6.18 | 6.83 |
| 다양성 | 우수 | 매우 우수 |

**주요 개선:** 마스킹된 영역에만 손실 계산 → 50% 패치 마스킹으로 **3배 빠른 훈련**[8]

#### G. MaskBit (2024)
**혁신:** 임베딩 없는 생성 (비트 토큰 사용)[9]

**개념:**
- VQGAN 대신 이진 양자화 (bit tokens)
- 각 차원을 0/1로 표현
- 더 가벼운 토큰화 스킴

**성과:**
- ImageNet-256 FID: **1.52** (MaskGIT 6.18 → 75% 개선)
- 305M 매개변수로 최고 성능
- 메모리 효율성 극대화[9]

### 3. 종합 성능 비교표 (ImageNet 256×256)

| 방법 | 연도 | FID | IS | 특징 | 빠르기 |
|------|------|-----|-----|------|---------|
| BigGAN | 2019 | 6.95 | 198 | GAN, 모드 붕괴 위험 | 빠름 |
| VQGAN | 2021 | 15.78 | 78 | 자동회귀, 느린 생성 | 느림 |
| **MaskGIT** | **2022** | **6.18** | 182 | **NAT 기초 모델** | 매우빠름 |
| Muse | 2023 | 6.06 | 203 | 텍스트-이미지 | 빠름 |
| Token-Critic | 2022 | 5.50 | - | MaskGIT 기반 개선 | 빠름 |
| MDT | 2023 | 6.83 | - | 디퓨전+마스킹 | 빠름 |
| AutoNAT | 2024 | **4.30** | - | **NAT 최적화** | 매우빠름 |
| VAR | 2024 | **1.73** | 350 | **AR 재정의** | 빠름 |
| MaskBit | 2024 | **1.52** | - | **비트 토큰** | 빠름 |

**추이:** 2022-2024 동안 FID가 6.18 → 1.52로 75% 개선됨

***

## X. 향후 연구에 미치는 영향

### 1. 즉각적 영향 (2022-2024)

**1) 비자동회귀 트랜스포머 재평가**[4]
- MaskGIT 이전: NAT는 "빠르지만 낮은 품질"로 평가됨
- MaskGIT 이후: "효율성과 품질의 새로운 균형점" 제시
- AutoNAT, VAR 등 후속 연구 촉발

**2) 토큰 기반 생성의 확산**
- 이미지에서 비디오(MAGVIT), 3D(다양한 연구들)로 확대
- 비용-효율적 접근으로 대규모 모델 훈련 가능[6]

**3) 마스킹 기반 학습의 일반화**
- NLP(BERT) → Vision(BEiT, MaskGIT) → Multimodal(Muse, MAGVLT)
- 자기지도 학습의 표준 기법 정착[10]

**4) 단일 모델의 다중 작업 지원**
- 기존: 각 작업마다 별도 모델 필요
- MaskGIT: 추가 학습 없이 인페인팅, 아웃페인팅, 편집 지원[1]

### 2. 전략적 영향

**산업 적용 가능성:**
- 실시간 이미지 생성 시스템 (8-12 스텝 대 250+ 스텝)
- 모바일/엣지 디바이스 배포 (227M 매개변수)
- 빠른 피드백 루프의 인터랙티브 도구[1]

**새로운 연구 방향:**
1. NAT의 이론적 기초 정립 (현재까지 부족)
2. 마스킹 전략의 일반적 원리 규명
3. 스케일링 법칙의 새로운 이해

**패러다임 시프트 신호:**[5]
- VAR의 등장으로 AR/NAT/확산의 상충관계 재검토
- 해상도 기반 학습의 새로운 가능성 제시

### 3. 미해결 질문 (Open Problems)

1. **극한적 일반화:** 희귀한 개념이나 복잡한 관계에 대한 성능[1]
2. **이론적 근거:** 왜 마스킹-예측이 이미지 생성에 특히 효과적?[11]
3. **다양성-충실도 트레이드오프:** 두 지표를 동시에 최대화할 수 있는가?
4. **극대규모 확장성:** 10B+ 매개변수 모델의 성능 특성

***

## XI. 향후 연구 시 고려할 점

### 1. 기술적 고려사항

**마스크 스케줄 최적화:**
```
현재: 고정된 함수 (코사인)
향후: 
  - 이미지 내용에 따른 적응형 스케줄
  - 다중 해상도 계층화 스케줄
  - 객체 복잡도에 따른 동적 조정
```

**어텐션 메커니즘 개선:**
```
과제: 양방향 어텐션의 O(N²) 복잡도
해결 아이디어:
  - 전역 어텐션의 메모리 효율화 (선형 어텐션)
  - 계층적/윈도우 기반 어텐션
  - 동적 토큰 선택 (중요한 토큰만 주의)
```

**토큰화 기법 발전:**
```
개선 방향:
  - 의미론적 토큰화 (개념 단위)
  - 가변 길이 토큰화
  - 학습 가능한 토큰화 (엔드-투-엔드)
```

### 2. 학습 전략 개선

**어려운 샘플 처리:**
- 어려운 이미지에 더 높은 마스킹 비율
- 과정 곡선(curriculum learning) 적용
- 부정적 마이닝(hard negative mining)

**조건부 생성 강화:**
- 더 강력한 조건 인코더 (CLIP, GPT-style)
- 조건-생성 정렬을 위한 대조 손실
- 조건별 마스킹 전략 차등화[1]

### 3. 평가 방법론 개발

**새로운 평가 메트릭:**
- 구성적 정확성: 엔티티 간 관계 정확도
- 의미론적 일관성: 개념의 안정성
- 공간 레이아웃 정확도: 객체 배치 신뢰도

**벤치마크 확대:**
- 전문 도메인 (의료영상, 산업 검사)
- 극단적 경우 (저해상도, 초고해상도)
- 멀티모달 조건 (텍스트+스케치+레이아웃)

### 4. 실제 응용 최적화

**실시간 시스템:**
- 점진적 렌더링 (사용자 체험 개선)
- 모바일 최적화 (양자화, 증류)
- 메모리-계산 트레이드오프[1]

**안전성 및 윤시:**
- 생성 이미지의 근원 추적성
- 생성 바이어스 분석 및 완화
- 동의 없는 얼굴 생성 방지

***

## 결론

MaskGIT는 이미지 생성 분야에서 **획기적인 전환점**을 제시한 작업입니다. 주요 성과는:

### 핵심 성과
1. **효율성-품질 파레토 경계 확대**: 자동회귀의 한계를 넘어 **64배 속도 향상**과 SOTA 품질 동시 달성[1]
2. **양방향 처리의 우월성 입증**: BERT 스타일 마스킹이 이미지 생성에서도 효과적임을 처음 입증
3. **마스크 스케줄의 중요성**: 코사인 함수가 경험적으로 최적임을 체계적으로 입증[1]
4. **다중 작업 통합**: 단일 모델로 생성, 인페인팅, 편집 지원[1]

### 연구 커뮤니티에의 기여
- **NAT 재평가**: 비자동회귀 모델이 강력한 대안임을 입증 → AutoNAT, VAR 촉발
- **마스킹 패러다임 확산**: 이미지 → 비디오 → 3D → 멀티모달로 개념 확대
- **실용적 방향성**: 저비용 대규모 모델 훈련의 길 제시

### 향후 예상 발전 방향
1. **마스킹 기반 생성의 이론적 기초 정립**
2. **극한 조건 (희귀 개념, 매우 높은 해상도)에서의 성능 개선**
3. **다양한 도메인 (의료, 제조 등)으로의 확장**
4. **대규모 멀티모달 모델의 핵심 컴포넌트화**

MaskGIT의 **"모두를 동시에 생성한 후 정제한다"**는 단순하지만 강력한 아이디어는, 생성 AI의 다음 세대를 정의하고 있습니다.

***

## 참고문헌 (논문에서 인용된 주요 항목)

[1] 2202.04200v1.pdf https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b382780b-3022-45d3-be14-c95bd4648c9f/2202.04200v1.pdf
[2] Muse: Text-To-Image Generation via Masked Generative Transformers https://arxiv.org/abs/2301.00704
[3] Improved Masked Image Generation with Token-Critic https://arxiv.org/abs/2209.04439
[4] Revisiting Non-Autoregressive Transformers for Efficient ... https://openaccess.thecvf.com/content/CVPR2024/papers/Ni_Revisiting_Non-Autoregressive_Transformers_for_Efficient_Image_Synthesis_CVPR_2024_paper.pdf
[5] [2404.02905] Visual Autoregressive Modeling: Scalable Image ... https://ar5iv.labs.arxiv.org/html/2404.02905
[6] MAGVIT: Masked Generative Video Transformer https://research.google/pubs/magvit-masked-generative-video-transformer/
[7] Masked Diffusion Transformer is a Strong Image Synthesizer https://openaccess.thecvf.com/content/ICCV2023/papers/Gao_Masked_Diffusion_Transformer_is_a_Strong_Image_Synthesizer_ICCV_2023_paper.pdf
[8] Fast Training of Diffusion Models with Masked Transformers https://arxiv.org/abs/2306.09305
[9] MaskBit: Embedding-free Image Generation via Bit Tokens https://arxiv.org/html/2409.16211v1
[10] MAGVLT: Masked Generative Vision-and-Language Transformer https://arxiv.org/abs/2303.12208
[11] On the Learning of Non-Autoregressive Transformers https://arxiv.org/abs/2206.05975
[12] Stable-Pose: Leveraging Transformers for Pose-Guided Text-to-Image Generation https://arxiv.org/abs/2406.02485
[13] PEM: Prototype-based Efficient MaskFormer for Image Segmentation https://arxiv.org/pdf/2402.19422.pdf
[14] X-LXMERT: Paint, Caption and Answer Questions with Multi-Modal Transformers https://www.aclweb.org/anthology/2020.emnlp-main.707
[15] From radiomics to transformers in pancreatic cancer detection and prognosis https://www.frontiersin.org/articles/10.3389/fmed.2025.1731922/full
[16] UniViLM: A Unified Video and Language Pre-Training Model for Multimodal Understanding and Generation https://www.semanticscholar.org/paper/4243555758433880a67b15b50f752b1e2a8c4609
[17] Editorial Message Vol. 27 No. 1 2026 https://journals.iium.edu.my/ejournal/index.php/iiumej/article/view/4129
[18] UNCAGE: Contrastive Attention Guidance for Masked Generative Transformers in Text-to-Image Generation https://ieeexplore.ieee.org/document/11339498/
[19] MaskMamba: A Hybrid Mamba-Transformer Model for Masked Image Generation https://arxiv.org/abs/2409.19937
[20] MaskSketch: Unpaired Structure-guided Masked Image Generation https://ieeexplore.ieee.org/document/10203438/
[21] MaskBit: Embedding-free Image Generation via Bit Tokens https://arxiv.org/html/2409.16211
[22] MaskGIT: Masked Generative Image Transformer https://arxiv.org/abs/2202.04200
[23] Muse: Text-To-Image Generation via Masked Generative Transformers https://arxiv.org/pdf/2301.00704.pdf
[24] TabMT: Generating tabular data with masked transformers https://arxiv.org/pdf/2312.06089.pdf
[25] Don't Look into the Dark: Latent Codes for Pluralistic Image Inpainting https://arxiv.org/html/2403.18186v2
[26] VisualBERT https://huggingface.co/docs/transformers/model_doc/visual_bert
[27] Text-To-Image Generation via Masked Generative Transformers https://proceedings.mlr.press/v202/chang23b/chang23b.pdf
[28] Improving Visual Quality of Image Synthesis by A Token- ... https://proceedings.neurips.cc/paper/2021/file/b056eb1587586b71e2da9acfe4fbd19e-Paper.pdf
[29] Diffusion Models for Non-autoregressive Text Generation https://www.ijcai.org/proceedings/2023/0750.pdf
[30] [논문리뷰] MaskGIT: Masked Generative Image Transformer https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/maskgit/
[31] [논문리뷰] BEiT: BERT Pre-Training of Image Transformers https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/beit/
[32] Non-autoregressive diffusion-based temporal point ... https://www.sciencedirect.com/science/article/abs/pii/S095741742403077X
[33] Muse: Text-to-image generation via masked ... https://dl.acm.org/doi/10.5555/3618408.3618570
[34] VISUALBERT: A SIMPLE AND PERFORMANT BASELINE ... https://velog.io/@pabiya/VISUALBERT-A-SIMPLE-AND-PERFORMANTBASELINE-FOR-VISION-AND-LANGUAGE
[35] Emage: Non-Autoregressive Text-to-Image Generation https://arxiv.org/abs/2312.14988
[36] [논문리뷰] Muse: Text-To-Image Generation via Masked ... https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/muse/
[37] [논문 리뷰] BEiT: BERT Pre-Training of Image Transformers https://lunaleee.github.io/posts/beit/
[38] [논문 퀵 리뷰] Revisiting Non-Autoregressive Transformers ... https://liner.com/ko/review/revisiting-nonautoregressive-transformers-for-efficient-image-synthesis
[39] TERA: Self-Supervised Learning of Transformer Encoder ... https://ar5iv.labs.arxiv.org/html/2007.06028
[40] Non-Autoregressive Diffusion-based Temporal Point ... https://arxiv.org/pdf/2311.01033.pdf
[41] Advancing 3D Point Cloud Understanding through Deep ... https://arxiv.org/html/2407.17877v1
[42] Vision Foundation Models as Effective Visual Tokenizers ... https://arxiv.org/html/2507.08441v1
[43] Revisiting Non-Autoregressive Transformers for Efficient ... https://arxiv.org/abs/2406.05478
[44] Expand BERT Representation with Visual Information via ... https://arxiv.org/html/2312.01592v2
[45] Video Editing via Interpolative Non-autoregressive Masked ... https://arxiv.org/html/2312.12468v1
[46] Hierarchical Masked Autoregressive Models with Low- ... https://arxiv.org/html/2505.20288v1
[47] Non-autoregressive Conditional Diffusion Models for Time ... https://arxiv.org/abs/2306.05043
[48] Visual Autoregressive Modeling for Instruction-Guided ... https://arxiv.org/html/2508.15772v1
[49] DART: Denoising Autoregressive Transformer for Scalable ... https://arxiv.org/html/2410.08159v1
[50] BEiT v2: Masked Image Modeling with Vector-Quantized ... https://ar5iv.labs.arxiv.org/html/2208.06366
[51] Revisiting Non-Autoregressive Transformers for Efficient Image Synthesis https://ieeexplore.ieee.org/document/10654945/
[52] ListenFormer: Responsive Listening Head Generation with Non-autoregressive Transformers https://dl.acm.org/doi/10.1145/3664647.3681182
[53] Enhancing Low-Light Image Reconstruction via Non-Autoregressive Transformers: A Mask-Aware Latent Integration Framework https://ieeexplore.ieee.org/document/11124638/
[54] Denoising Autoregressive Transformers for Scalable Text-to-Image Generation https://www.semanticscholar.org/paper/d1a47b22d36c1747ec88686a0301626119ec83a4
[55] MADFormer: Mixed Autoregressive and Diffusion Transformers for Continuous Image Generation https://arxiv.org/abs/2506.07999
[56] FourierNAT: A Fourier-Mixing-Based Non-Autoregressive Transformer for Parallel Sequence Generation https://arxiv.org/abs/2503.07630
[57] Alleviating Directional Bias in Non-Autoregressive Transformers https://ieeexplore.ieee.org/document/11228036/
[58] Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction https://arxiv.org/abs/2404.02905
[59] M6-UFC: Unifying Multi-Modal Controls for Conditional Image Synthesis via Non-Autoregressive Generative Transformers https://www.semanticscholar.org/paper/f131e2f7bcf250a7ee25b79a0b9a442f12bd7df1
[60] Revisiting Non-Autoregressive Transformers for Efficient Image Synthesis http://arxiv.org/pdf/2406.05478.pdf
[61] StraIT: Non-autoregressive Generation with Stratified Image Transformer https://arxiv.org/abs/2303.00750
[62] ENAT: Rethinking Spatial-temporal Interactions in Token-based Image
  Synthesis https://arxiv.org/html/2411.06959
[63] AdaNAT: Exploring Adaptive Policy for Token-Based Image Generation https://arxiv.org/html/2409.00342
[64] Retrieving Sequential Information for Non-Autoregressive Neural Machine Translation https://www.aclweb.org/anthology/P19-1288.pdf
[65] Retrieving Sequential Information for Non-Autoregressive Neural Machine
  Translation https://arxiv.org/abs/1906.09444
[66] Glancing Transformer for Non-Autoregressive Neural Machine Translation https://aclanthology.org/2021.acl-long.155.pdf
[67] Directed Acyclic Transformer for Non-Autoregressive Machine Translation https://arxiv.org/pdf/2205.07459.pdf
[68] Exploring Adaptive Policy for Token-Based Image Generation https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02478.pdf
[69] MAGVIT: Masked Generative Video Transformer - Hugging Face https://huggingface.co/papers/2212.05199
[70] Fast Training of Diffusion Transformer with Extreme ... https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11278.pdf
[71] Meet MAGVIT: A Novel Masked Generative Video Transformer To Address AI Video Generation Tasks https://www.marktechpost.com/2023/01/22/meet-magvit-a-novel-masked-generative-video-transformer-to-address-ai-video-generation-tasks/
[72] EDT: An Efficient Diffusion Transformer Framework Inspired by ... https://proceedings.neurips.cc/paper_files/paper/2024/file/f1f9962f76581ce8bf38d04c6d6c96b1-Paper-Conference.pdf
[73] CVPR 2024 Open Access Repository https://openaccess.thecvf.com/content/CVPR2024/html/Ni_Revisiting_Non-Autoregressive_Transformers_for_Efficient_Image_Synthesis_CVPR_2024_paper.html
[74] Revisiting Non-Autoregressive Transformers for Efficient ... https://liner.com/review/revisiting-nonautoregressive-transformers-for-efficient-image-synthesis
[75] MAGVIT: Masked Generative Video Transformer https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_MAGVIT_Masked_Generative_Video_Transformer_CVPR_2023_paper.pdf
[76] [논문리뷰] Fast Training of Diffusion Models with Masked ... https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/maskdit/
[77] MAGVIT: Masked Generative Video Transformer https://openaccess.thecvf.com/content/CVPR2023/supplemental/Yu_MAGVIT_Masked_Generative_CVPR_2023_supplemental.pdf
[78] MDSGen: Fast and Efficient Masked Diffusion Temporal-Aware ... https://arxiv.org/html/2410.02130v2
[79] Appendix A. Implementation Details https://openaccess.thecvf.com/content/CVPR2024/supplemental/Ni_Revisiting_Non-Autoregressive_Transformers_CVPR_2024_supplemental.pdf
[80] Effective and Efficient Masked Image Generation Models https://arxiv.org/pdf/2503.07197.pdf
[81] SceneNAT: Masked Generative Modeling for Language ... https://arxiv.org/html/2601.07218v1
[82] MAGVIT: Masked Generative Video Transformer https://arxiv.org/abs/2212.05199
[83] Diffusion Beats Autoregressive in Data-Constrained Settings https://arxiv.org/html/2507.15857v1
[84] MAGVIT: Masked Generative Video Transformer https://arxiv.org/pdf/2212.05199.pdf
[85] AdaNAT: Exploring Adaptive Policy for Token-Based Image ... https://arxiv.org/html/2409.00342v3
[86] [PDF] MAGVIT: Masked Generative Video Transformer https://www.semanticscholar.org/paper/fe34137e5cc07235eae65ce53a54cd226b9f8b23
[87] Fast Training of Diffusion Models with Masked Transformers https://arxiv.org/html/2306.09305v1
[88] [Literature Review] Revisiting Non-Autoregressive ... https://www.themoonlight.io/en/review/revisiting-non-autoregressive-transformers-for-efficient-image-synthesis
