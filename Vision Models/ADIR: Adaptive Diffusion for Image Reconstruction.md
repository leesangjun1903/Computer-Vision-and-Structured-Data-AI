# ADIR: Adaptive Diffusion for Image Reconstruction

### 1. 핵심 주장 및 주요 기여

ADIR(Adaptive Diffusion for Image Reconstruction)은 사전 훈련된 확산 모델의 강력한 이미지 생성 능력을 활용하면서 측정 데이터와의 일관성을 유지하여 이미지 복원 작업을 해결하는 혁신적 프레임워크입니다.[1]

**핵심 주장:**
- 확산 모델은 자연 이미지 통계에 대한 풍부한 사전 정보를 학습했으므로, 이를 역 문제(inverse problem) 해결에 효과적으로 활용할 수 있습니다.[1]
- 기존의 데이터 주도 방식 복원 방법들은 훈련되지 않은 열화 유형에 대해 과도하게 적합(overfitting)되는 경향이 있습니다.[1]
- 테스트 시점 적응(test-time adaptation)을 통해 퇴화된 입력과 의미적, 시각적으로 유사한 K-NN 이미지로 모델을 적응시키면, 일반화 성능이 현저히 향상됩니다.[1]

**주요 기여:**
1. **조건부 샘플링 프레임워크**: 측정 연산자와의 일관성을 유지하면서 사전 훈련된 확산 모델의 범위 내에서 복원을 수행하는 가이드된 역확산 프로세스 제안[1]

2. **LoRA 기반 적응 전략**: CLIP 같은 오프더쉘프 비전-언어 모델을 활용하여 외부 데이터셋에서 문맥적으로 유사한 이미지를 효율적으로 검색하고, 이를 바탕으로 LoRA를 통해 모델을 적응시키는 novel 방법론[1]

3. **광범위한 실험 검증**: Stable Diffusion과 Guided Diffusion 두 주요 모델에서 초해상도, 디블러링, 컬러화 등 다양한 복원 작업에서 substantial improvement 달성[1]

***

### 2. 해결하고자 하는 문제

#### 2.1 기본 역 문제 정식화

선형 역 문제는 다음과 같이 정의됩니다:[1]

$$y = Ax + e$$

여기서:
- $y \in \mathbb{R}^m$: 열화된 관측 이미지
- $x \in \mathbb{R}^n$: 복원하고자 하는 원본 클린 이미지
- $A \in \mathbb{R}^{m \times n}$: 측정 연산자 (흐림, 마스킹, 부분 샘플링 등)
- $e \in \mathbb{R}^m \sim \mathcal{N}(0, \sigma^2 I_m)$: 측정 노이즈

#### 2.2 기존 방법의 한계

**1. 일반화 부족**
- 훈련 데이터에 포함되지 않은 퇴화 유형에 대해 성능이 급격히 저하됩니다[1]
- 단일 관측 모델에만 특화되어 다른 유형의 퇴화에 직접 전이되지 않습니다[1]

**2. 테스트 시점 유연성 부족**
- 기존의 end-to-end 심층 신경망 방식은 테스트 시점에 적응이 어렵습니다[1]
- 각 퇴화 유형별로 별도의 모델 훈련이 필요합니다[1]

**3. 선행 정보 활용의 비효율성**
- Deep Image Prior와 같은 테스트 시점만의 방법들은 외부 데이터를 활용하지 못해 성능이 낮습니다[1]
- GAN 기반 방식은 멀티모달 분포를 충분히 포착하지 못합니다[1]

***

### 3. 제안 방법 (수식 포함)

#### 3.1 확산 기반 이미지 복원 (Diffusion-based Image Reconstruction)

**확산 모델의 기본 구조:**

확산 모델은 순방향 과정과 역방향 과정으로 구성됩니다. 주어진 훈련 샘플 $x_0 \sim q_x$에 대해, 마르코프 체인 순방향 과정은:[1]

$$q(x_{1:T}|x_0) := \prod_{t=1}^{T} q(x_t|x_{t-1}), \quad q(x_t|x_{t-1}) := \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I_n)$$

마르코프 특성을 이용하면 $x_t|x_0$을 다음과 같이 매개변수화할 수 있습니다[1]:

$$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I_n)$$

여기서 $\bar{\alpha}\_t := \prod_{s=1}^{t}\alpha_s$, $\alpha_s := 1-\beta_s$입니다.[1]

역방향 과정은 다음과 같이 매개변수화됩니다:[1]

$$p_\theta(x_{0:T}) := p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}|x_t), \quad p_\theta(x_{t-1}|x_t) := \mathcal{N}(\mu_\theta(x_t,t), \Sigma_\theta(x_t,t))$$

단순화된 손실 함수는:[1]

$$\ell_{\text{simple}}(x_0, \varepsilon_\theta, t) = \|\varepsilon - \varepsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\varepsilon, t)\|_2^2$$

**조건부 샘플링 (Conditional Sampling):**

후진 확산 과정에서 측정 데이터 $y$에 조건화된 샘플링을 위해, 후방 분포를 다음과 같이 정의합니다:[1]

$$p_\theta(x_t|x_{t+1}, y) \propto p_\theta(x_t|x_{t+1})p_{y|x_t}(y|x_t)$$

여기서 $p_\theta(x_t|x_{t+1}) = \mathcal{N}(\mu_\theta(x_{t+1}; t+1), \Sigma_\theta(x_{t+1}; t+1))$는 학습된 확산 사전입니다[1].

**핵심 근사:**

중간 시점 $t$에서의 우도 함수는 정확히 알려지지 않으므로, ADIR은 다음과 같이 근사합니다:[1]

$$\log p_{y|x_t}(y|x_t) \approx \log p_{y|x_0}(y|\hat{x}_0(x_t))$$

여기서 추정된 원본 이미지는:[1]

$$\hat{x}_0(x_t) := (x_t - \sqrt{1-\bar{\alpha}_t}\varepsilon_\theta(x_t, t))/\sqrt{\bar{\alpha}_t}$$

이를 통해 매 반복 $t$에서의 그래디언트는:[1]

$$g \approx -2A^T(Ax_t - y_t)|_{x_t=\mu_\theta}$$

여기서 $y_t := \sqrt{\bar{\alpha}\_t}y + \sqrt{1-\bar{\alpha}\_t}A\varepsilon_\theta$입니다.[1]

**최종 샘플링 방정식:**

$$x_{t-1} \sim \mathcal{N}(\mu_\theta + \Sigma_\theta g, \Sigma_\theta)$$

#### 3.2 적응적 확산 (Adaptive Diffusion)

**기본 적응 전략:**

사전 훈련된 확산 모델의 매개변수 $\theta$를 측정 이미지 $y$에 적응시키기 위해 다음 최소화 문제를 풉니다:[1]

$$\hat{\theta} = \arg\min_\theta \sum_{t=1}^{T} \ell_{\text{simple}}(y, \varepsilon_\theta, t)$$

**K-NN 기반 개선 적응:**

보다 강력한 적응을 위해, CLIP 임베딩 공간에서 $y$와 의미적으로 유사한 K개의 이미지를 외부 데이터셋에서 검색합니다. 이러한 K-NN 이미지 집합 $\{z_k\}_{k=1}^K$를 사용하여 다음을 최소화합니다:[1]

$$\hat{\theta} = \arg\min_\theta \sum_{k=1}^{K} \sum_{t=1}^{T} \ell_{\text{simple}}(z_k, \varepsilon_\theta, t)$$

**LoRA를 통한 효율적 매개변수 조정:**

전체 모델 미세 조정 대신, LoRA(Low-Rank Adaptation)를 모든 컨볼루션 계층에 적용합니다:[1]

$$\varepsilon_\theta(x_t, t) = \varepsilon_{\theta_0}(x_t, t) + \Delta W \cdot x_t$$

여기서 LoRA 매개변수는 rank $r=16$, scaling $\alpha=8$로 설정되며, Adam 최적화기로 최적화됩니다.[1]

***

### 4. 모델 구조

#### 4.1 전체 시스템 아키텍처

ADIR의 아키텍처는 다음과 같은 세 가지 핵심 컴포넌트로 구성됩니다:[1]

**컴포넌트 1: K-NN 검색 모듈**
- 입력: 열화된 저해상도 이미지 $y$
- 방법: CLIP 인코더를 사용하여 $y$의 임베딩 벡터 추출
- 검색: K-D Tree 자료구조를 활용한 효율적 최근접 이웃 검색
- 출력: Google Open Images Dataset에서 검색된 K개의 유사 이미지 $\{z_1, z_2, ..., z_K\}$
  - Guided Diffusion: K=20
  - Stable Diffusion: K=50[1]

**컴포넌트 2: LoRA 기반 적응 모듈**
- 입력: K-NN 이미지와 사전 훈련된 확산 모델
- 방법: 
  - 모든 컨볼루션 계층에 LoRA 어댑터 추가
  - 확산 스케줄러에서 제공하는 노이즈 스케줄 사용
  - 배치 크기 6, 임의 시간 단계 샘플링
- 훈련 설정:
  - 반복: 400회
  - 학습률: $10^{-4}$
  - 단계: 1000 (Guided Diffusion), 50 (Stable Diffusion)[1]

**컴포넌트 3: 가이드된 역확산 샘플링 모듈**
- 입력: 적응된 확산 모델, 측정 이미지 $y$, 측정 연산자 $A$
- 프로세스:
  1. 가우시안 노이즈에서 초기화: $x_T \sim \mathcal{N}(0, I_n)$
  2. $t=T$에서 $t=1$까지 역방향 반복
  3. 각 단계에서 측정 일관성 그래디언트 계산
  4. 가이드된 샘플링: $x_{t-1} \sim \mathcal{N}(\mu + \Sigma g, \Sigma)$
- 출력: 복원된 고품질 이미지 $x_0$[1]

#### 4.2 주요 혁신 설계 요소

**1. 수치 안정성을 위한 근사:**

정확한 gradient 계산의 불안정성 문제를 해결하기 위해, ADIR은 다음과 같은 이완된 표현을 사용합니다:[1]

$$\|A\hat{x}_0(x_t) - y\|_2^2 = \|Ax_t - \sqrt{\bar{\alpha}_t}y - \sqrt{1-\bar{\alpha}_t}A\varepsilon_\theta\|_2^2 = \|Ax_t - y_t\|_2^2$$

이는 기존 방법들(DiffPIR, Score-based methods)의 불안정한 구현을 개선합니다.[1]

**2. 문맥 인식 검색:**

CLIP 임베딩 공간은 열화 유형에 덜 민감하므로, 의미적으로 유사한 이미지를 효과적으로 검색할 수 있습니다. 이는 픽셀 공간 MSE 기반 검색보다 훨씬 우수합니다.[1]

**3. 계산 효율성:**

- K-D Tree를 사용한 임베딩 공간에서의 검색: 약 100배 빠름 (Table 5)
- LoRA로 인한 매개변수 감소: 대부분의 레이어는 동결
- 총 실행 시간: Guided Diffusion 903초, Stable Diffusion 1308초[1]

***

### 5. 성능 향상 및 실험 결과

#### 5.1 초해상도 (Super-Resolution)

**Guided Diffusion (×8 SR on 512×512)**

| 방법 | LPIPS↓ | AVA↑ | KonIQ↑ |
|------|--------|------|--------|
| GD (Baseline) | 0.365 | 4.36 | 53.99 |
| ADIR (GD) | 0.347 | 4.41 | 55.89 |
| 개선도 | -4.9% | +1.1% | +3.5% |

**Stable Diffusion (×4 SR)**

| 방법 | LPIPS↓ | AVA↑ | KonIQ↑ |
|------|--------|------|--------|
| Stable Diffusion | 0.331 | 5.07 | 69.18 |
| ADIR (SD) | 0.213 | 5.51 | 72.56 |
| 개선도 | -35.6% | +8.7% | +4.9% |

Stable Diffusion의 경우 AVA-MUSIQ와 KonIQ-MUSIQ에서 각각 8.7%와 4.9% 개선[1]

#### 5.2 디블러링 (Deblurring)

5×5 uniform blur with σ=10 noise에서:

| 방법 | KonIQ↑ (256×256) | KonIQ↑ (512×512) |
|------|-----------------|-----------------|
| GD Baseline | 49.19 | 58.66 |
| ADIR | 55.78 | 60.13 |
| 개선도 | +13.4% | +2.5% |

**특히 해상도 외삽(out-of-distribution)에서의 성능**: 512×512 해상도는 훈련 시 256×256이었음에도, ADIR은 강력한 일반화를 보임[1]

#### 5.3 컬러화 (Colorization)

| 방법 | AVA↑ | KonIQ↑ |
|------|------|--------|
| GD (Baseline) | 4.195 | 56.044 |
| ADIR (GD) | 4.214 | 58.679 |
| 개선도 | +0.5% | +4.7% |

#### 5.4 K-NN 검색의 효과 (Ablation Study)

Table 5에서 검색 전략 비교:

| 방법 | LPIPS↓ | AUC score | 검색 시간(s) |
|------|--------|-----------|------------|
| Random NN | 0.430 | 52.89 | 1300 |
| MSE-based NN | 0.434 | 53.12 | 2700 |
| ADIR (CLIP-based) | 0.347 | 55.89 | 1308 |

**결론**: CLIP 기반 K-NN이 의미적 유사성을 더 잘 포착하여 MSE 기반보다 성능이 우수하면서도 계산 효율성을 유지[1]

#### 5.5 텍스트 기반 이미지 편집

- Stable Diffusion 기반 inpainting에서 ADIR 적용 시 더욱 사실적이고 정확한 생성 결과 달성
- GLIDE와 비교 가능한 수준의 결과 생성 (완전 미세 조정 없이)[1]

***

### 6. 일반화 성능 향상 가능성 (중점 분석)

#### 6.1 현재 성과

**1. 도메인 시프트에 대한 강인성**

ADIR은 다음과 같은 이유로 향상된 일반화를 달성합니다:[1]

- **테스트 시점 적응**: 훈련 데이터에 포함되지 않은 새로운 열화 유형에 대해 실시간으로 모델을 적응
- **문맥 인식 선택**: CLIP 임베딩을 통해 의미적으로 유사한 이미지만 선별하여 적응에 사용
- **제한된 매개변수 수정**: LoRA를 통해 작은 수의 매개변수만 조정하므로, 사전 훈련된 모델의 강력한 기능을 보존

**2. 해상도 외삽**

디블러링 실험에서 주목할 점:[1]
- 모델은 256×256에서 훈련됨
- 512×512에서의 테스트 성능: KonIQ 60.13 (vs. baseline 58.66)
- 이는 ADIR이 해상도 도메인 시프트를 성공적으로 처리함을 의미

**3. 여러 열화 유형에 대한 유니버설 처리**

K-NN 적응 덕분에:[1]
- 초해상도, 디블러링, 컬러화 세 가지 완전히 다른 작업에서 개선
- 단일 Stable Diffusion 모델이 모든 작업에 효과적

#### 6.2 일반화의 메커니즘

**메커니즘 1: 자연 이미지 다양성의 명시적 활용**

외부 데이터셋(Google Open Images)에서 K개의 의미적으로 유사한 이미지를 검색함으로써:[1]

1. **다양한 표현 학습**: 동일 범주의 이미지들의 다양한 특징을 모델이 학습
2. **오버피팅 방지**: 단일 열화된 이미지만이 아닌 K개의 정상 이미지로 적응
3. **의미 보존**: CLIP이 열화에 불변인 임베딩을 제공하므로, 진정한 의미적 유사성 기반 검색

**메커니즘 2: 사전 분포와 측정 일관성의 균형**

확산 가이던스 프레임워크는 다음을 동시에 최적화:[1]

$$\text{maximize } p_\theta(x_t|x_{t+1}) \times p_{y|x_t}(y|x_t)$$

- 좌항: 사전 훈련된 확산 모델의 강력한 이미지 선행 정보
- 우항: 실제 측정 데이터 $y$와의 피델리티

이 균형은 새로운 열화 유형에서도 두 목표를 동시에 달성하게 함[1]

**메커니즘 3: 매개변수 효율적 적응**

LoRA의 저순위 구조는:[1]

- **구조적 정규화**: 저순위 제약이 자연스러운 정규화로 작용
- **과도한 적응 방지**: 전체 미세 조정보다 모델의 기본 특성을 보존
- **안정적 수렴**: 하이퍼파라미터에 대한 감도 감소

#### 6.3 한계 및 개선 방향

**현재 한계:**[1]

1. **데이터셋 의존성**: 다양한 외부 데이터셋이 필요하며, 낮은 다양성의 데이터셋에서는 성능 저하

2. **연산 비용**: 테스트 시점에 K-NN 검색과 LoRA 적응으로 추가 시간 필요 (약 900~1300초)

3. **블라인드 설정 미지원**: 측정 연산자 $A$를 알아야 함 (현실에서 $A$를 모르는 경우가 많음)

4. **랜덤성**: 확산 과정의 내재적 랜덤성으로 인해 결과에 변동성 존재

***

### 7. 최신 관련 연구 분석 (2020년 이후)

#### 7.1 확산 모델 기반 이미지 복원 (2022-2025)

**주요 발전**:

1. **DDNM (Diffusion-based Denoising for Noisy Measurements, 2022)**[2]
   - 일관성 제약을 역확산 프로세스에 통합
   - ADIR의 근사 방식과 유사한 접근

2. **Diffusion Models for Medical Image Reconstruction (2024)**[3]
   - 의료 이미징에서 확산 모델의 상태-최신 성과 입증
   - MRI, CT, PET 모달리티에서 높은 정확도 달성
   - **핵심 통찰**: 비감독 확산 모델이 도메인 시프트에 강건함

3. **DiffIR2VR-Zero (2025)**: 비디오 복원으로 확장
4. **SIR-DIFF (2025)**: 다중 뷰 확산 모델을 통한 이미지 복원

#### 7.2 테스트 타임 적응 및 일반화 (2023-2025)

**핵심 개발:**

1. **GDA: Generalized Diffusion for Robust Test-time Adaptation (2024)**[4]
   - 테스트 타임에 확산 모델을 사용한 적응
   - OOD(out-of-distribution) 샘플에 대해 state-of-the-art 개선
   - ADIR과 유사한 테스트 타임 적응 패러다임

2. **TT-SaD: Test-Time Stain Adaptation (2024)**[5]
   - 의료 이미징의 염색 변화를 역 문제로 정식화
   - 확산 모델로 테스트 타임 적응 해결

3. **Distribution Shift Inversion (2023)**[6]
   - OOD 샘플을 학습 분포로 변환하기 위해 확산 모델 사용
   - 다중 도메인 일반화에서 3-4% 개선

#### 7.3 파라미터 효율적 미세 조정 (2023-2025)

**LoRA 기반 확산 개선:**

1. **DiffuseKronA (2024)**[7]
   - Kronecker 곱 기반 적응으로 LoRA 개선
   - 매개변수 99.947% 감소 (원본 DreamBooth 대비)
   - 하이퍼파라미터 민감도 감소

2. **SuperLoRA (2024)**[8]
   - LoRA 변형의 일반화된 프레임워크
   - 그룹화, 투영, 텐서 분해로 10배 매개변수 효율성 개선

3. **StyleInject (2024)**[9]
   - 텍스트-이미지 모델용 specialized 미세 조정
   - 여러 병렬 저순위 행렬로 스타일 다양성 유지

#### 7.4 CLIP 및 비전-언어 모델의 발전 (2023-2025)

**K-NN 검색 개선:**

1. **CLIP-PING (2025)**[10]
   - 최근접 이웃 감독을 통한 CLIP 개선
   - 경량 모델 성능 향상

2. **kNN-CLIP (2024)**[11]
   - 지속적인 어휘 확장을 위한 훈련 없는 전략
   - 재앙적 망각 방지

3. **CODER: Cross-modal Neighbor Representation (2024)**[12]
   - CLIP의 최근접 이웃 관점 재해석
   - 세밀한 이미지 분류에서 성능 개선

#### 7.5 확산 모델의 신뢰성 및 일반화 (2024-2025)

**중요 발견:**

1. **Principled Out-of-Distribution Generalization (2025)**[13]
   - 확산 모델의 합성적 일반화 능력 조사
   - 구성 요소 간 상호작용 이해

2. **Test-Time Adaptation Improves Inverse Problems (2025)**[14]
   - 패치 기반 확산 모델의 테스트 타임 적응
   - IEEE Computational Imaging에 게재

3. **Differentially Private Fine-Tuning (2025)**[15]
   - LoRA 미세 조정 시 프라이버시 보호
   - 차등 프라이버시와 모델 유틸리티의 트레이드오프

***

### 8. ADIR의 앞으로의 연구 영향 및 고려 사항

#### 8.1 학계적 영향

**1. 확산 모델 기반 역 문제 해결의 새로운 패러다임**

ADIR은 다음과 같은 중요한 통찰을 제시합니다:[1]

- 단순히 사전 훈련된 모델을 사용하는 것보다 **테스트 타임 적응**이 매우 효과적임
- **외부 데이터와 현재 입력의 의미적 유사성**을 활용하면 성능을 크게 향상시킬 수 있음
- LoRA 같은 효율적 적응 방법이 모델의 일반화를 보존하면서 성능을 개선할 수 있음

이는 향후 역 문제 연구의 **표준 패러다임**이 될 가능성이 높습니다.

**2. 의료 영상 재구성으로의 자연스러운 확장**

의료 이미징 분야에서:[3]
- MRI 가속화 재구성
- CT 스파스 뷰 재구성
- 저선량 PET 재구성

이미 확산 모델이 state-of-the-art 결과를 얻고 있으며, ADIR의 테스트 타임 적응은 이러한 영역에서 추가 개선을 가져올 수 있습니다.

**3. 도메인 시프트에 강건한 모델 설계의 기준**

ADIR의 성공은 **도메인 시프트 강인성**이 다음을 통해 달성 가능함을 보여줍니다:
- 강력한 사전 분포 학습
- 테스트 타임 적응
- 의미적 유사성 기반 샘플 선택

#### 8.2 실무 응용 가능성

**1. 소비자 애플리케이션**

- **스마트폰 사진 개선**: 저화질 사진의 초해상도, 디블러링
- **실시간 비디오 처리**: 클라우드 기반 비디오 복원 서비스
- **아카이브 복원**: 오래된 사진/필름 디지털화

**2. 산업 응용**

- **의료 영상**: 방사선량 감소 또는 스캔 시간 단축
- **위성 영상**: SAR 이미지 초해상도, 노이즈 제거
- **보안 카메라**: 저해상도 영상에서 얼굴 인식 개선

**3. 엔터테인먼트 및 미디어**

- **영상 복원**: 이전 영상 자료의 품질 향상
- **영상 편집**: 자연스러운 인페인팅 및 스타일 전이
- **생성 콘텐츠**: 더욱 사실적인 이미지/비디오 생성

#### 8.3 앞으로 연구 시 고려할 점

**1. 알고리즘적 개선**

| 개선 방향 | 구체적 방안 | 예상 영향 |
|---------|---------|---------|
| 계산 속도 | 빠른 K-NN 검색 (예: LSH) 또는 사전 계산 캐시 | 배포 속도 향상 |
| 적응 전략 | 하이브리드: K-NN + 직접 이미지 $y$ 혼합 | 다양성과 특이성 균형 |
| 매개변수 효율성 | 더 낮은 LoRA rank 또는 선택적 계층 적응 | 메모리/계산 감소 |
| 블라인드 설정 | 비지도 또는 약지도 $A$ 추정 | 실제 응용 확대 |

**2. 이론적 분석 필요**

- **왜 K-NN 적응이 효과적인가?** 
  - 정보이론적 분석: K개 이미지의 상호 정보 구조
  - 통계적 분석: 표본 복잡도와 일반화 오차 경계

- **도메인 시프트 강인성의 형식화**
  - 확산 모델의 구조적 성질이 어떻게 일반화를 보장하는가?
  - 테스트 타임 적응의 수렴 속도와 최종 오류 경계

**3. 실제 문제 해결**

| 도전 과제 | 제안 해결책 | 연구 방향 |
|---------|---------|---------|
| 비다양성 데이터셋 | 합성 데이터 생성 또는 데이터 증강 | 확산 모델 기반 데이터 생성 |
| 랜덤성 제어 | 확정적 샘플링 또는 앙상블 방법 | DDIM 등 고정 예측 샘플러 |
| 하이퍼파라미터 선택 | 자동 튜닝 또는 적응적 선택 | 메타 학습 적용 |
| 프라이버시 문제 | 차등 프라이버시 LoRA[15] | 공동 학습 및 연합 학습 |

**4. 기술 스택 개선**

```
현재 (ADIR):
입력 → CLIP 임베딩 → K-D Tree → K-NN 검색 → LoRA 적응 → 확산 가이던스 → 출력

향상 방향:
1. CLIP 대체: BLIP, CyCLIP, 또는 더 강력한 VLM 사용
2. 검색 최적화: 근사 최근접 이웃 (ANN) 사용
3. 적응 개선: 메타 학습 기반 LoRA 초기화
4. 샘플링 가속: 더 빠른 확산 샘플러 (DDIM, ODE 기반)
```

**5. 도메인별 맞춤 설계**

- **의료**: 프라이버시 보호 + 재현성 (확정적 샘플링)
- **위성 영상**: 대규모 외부 데이터 활용 + 스펙트럼 특성 반영
- **모바일**: 경량 모델 + 엣지 컴퓨팅 최적화

#### 8.4 근본적인 오픈 문제

**1. 이상적 K 값의 결정**

현재: GD는 K=20, SD는 K=50으로 휴리스틱하게 설정[1]
- 질문: K와 성능의 정확한 관계는? 최적 K 자동 선택 가능한가?
- 연구: 정보이론적 분석 필요

**2. 비선형 측정 연산자로의 확장**

현재 ADIR: 선형 $y = Ax + e$ 가정[1]
- 비선형 CT, MRI 등의 실제 역 문제 적용
- 비선형 근사 또는 선형화 전략 필요

**3. 불확실성 정량화**

확산 모델의 장점: 다중 모드 분포 포착
- ADIR의 현재 결과는 단일 샘플이므로 불확실성 미추정
- 베이지안 해석 및 신뢰 구간 제공 가능성

***

### 결론

ADIR은 확산 모델을 역 문제 해결에 효과적으로 적용하는 혁신적 프레임워크입니다. 측정 일관성을 강제하면서도 강력한 이미지 사전을 활용하고, 특히 **테스트 타임 K-NN 기반 LoRA 적응**을 통해 일반화 성능을 현저히 향상시킵니다.

**핵심 기여:**
1. 선형 역 문제를 위한 조건부 확산 가이던스 프레임워크
2. 의미적 유사성 기반 K-NN 검색과 LoRA 적응의 결합
3. 다양한 복원 작업에서 4-36% 성능 개선 달성

**향후 방향:**
- 비선형 역 문제로의 확장
- 계산 속도 개선
- 이론적 강건성 분석
- 의료, 위성, 엔터테인먼트 응용 개발

이 연구는 **도메인 시프트 강건성**과 **테스트 타임 적응**이 현대 컴퓨터 비전에서 얼마나 중요한지를 입증하며, 향후 복원 작업의 표준 접근법이 될 것으로 예상됩니다.

***

### 참고 문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0529f26f-7b0c-47f4-978a-d4d6e8b1c24b/2212.03221v2.pdf)
[2](https://arxiv.org/abs/2403.17042)
[3](https://academic.oup.com/bjrai/article/doi/10.1093/bjrai/ubae013/7745314)
[4](https://openaccess.thecvf.com/content/CVPR2024/papers/Tsai_GDA_Generalized_Diffusion_for_Robust_Test-time_Adaptation_CVPR_2024_paper.pdf)
[5](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05175.pdf)
[6](https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_Distribution_Shift_Inversion_for_Out-of-Distribution_Prediction_CVPR_2023_paper.pdf)
[7](https://ieeexplore.ieee.org/document/10943802/)
[8](https://arxiv.org/abs/2403.11887)
[9](https://dl.acm.org/doi/10.1145/3730403)
[10](https://arxiv.org/html/2412.03871v1)
[11](https://openreview.net/pdf/07c979de89022d0454f8a1266c45ba79518a70c2.pdf)
[12](https://openaccess.thecvf.com/content/CVPR2024/papers/Yi_Leveraging_Cross-Modal_Neighbor_Representation_for_Improved_CLIP_Classification_CVPR_2024_paper.pdf)
[13](https://arxiv.org/abs/2505.22622)
[14](https://arxiv.org/pdf/2508.01975.pdf)
[15](https://openaccess.thecvf.com/content/ICCV2025/papers/Tsai_Differentially_Private_Fine-Tuning_of_Diffusion_Models_ICCV_2025_paper.pdf)
[16](https://arxiv.org/abs/2411.03053)
[17](https://ieeexplore.ieee.org/document/10447579/)
[18](https://ieeexplore.ieee.org/document/10656848/)
[19](https://ieeexplore.ieee.org/document/10837160/)
[20](https://arxiv.org/abs/2406.06372)
[21](https://ieeexplore.ieee.org/document/11131971/)
[22](https://arxiv.org/abs/2406.07487)
[23](https://arxiv.org/abs/2404.07191)
[24](http://arxiv.org/pdf/2407.03636.pdf)
[25](https://arxiv.org/html/2503.14463v1)
[26](https://arxiv.org/html/2410.17752)
[27](http://arxiv.org/pdf/2407.01519v3.pdf)
[28](http://arxiv.org/pdf/2409.19589.pdf)
[29](https://arxiv.org/pdf/2308.09388.pdf)
[30](https://arxiv.org/html/2406.19030v1)
[31](http://arxiv.org/pdf/2311.14900v2.pdf)
[32](https://academic.oup.com/bjrai/article/1/1/ubae013/7745314)
[33](https://airlabkhu.github.io/DGA2/)
[34](https://www.sciencedirect.com/science/article/abs/pii/S1051200421003249)
[35](https://arxiv.org/html/2308.09388v2)
[36](https://www.sciencedirect.com/science/article/abs/pii/S1077314225001626)
[37](https://www.nature.com/articles/s41598-024-69415-2)
[38](https://www.computationalimaging.org/publications/diffusion-in-the-dark/)
[39](https://iccv.thecvf.com/virtual/2025/poster/2433)
[40](https://www.pnas.org/doi/10.1073/pnas.1907377117)
[41](https://www.sciencedirect.com/science/article/pii/S0895611125001028)
[42](https://ieeexplore.ieee.org/document/10516655/)
[43](https://arxiv.org/abs/2408.01415)
[44](https://arxiv.org/abs/2402.02347)
[45](https://www.mdpi.com/2673-2688/5/4/88)
[46](https://ieeexplore.ieee.org/document/10678270/)
[47](https://arxiv.org/abs/2409.08482)
[48](https://ojs.acad-pub.com/index.php/CAI/article/view/1498)
[49](https://arxiv.org/pdf/2306.07967.pdf)
[50](https://arxiv.org/pdf/2410.18720.pdf)
[51](https://arxiv.org/abs/2402.17412)
[52](https://arxiv.org/html/2410.03941)
[53](http://arxiv.org/pdf/2410.20777.pdf)
[54](http://arxiv.org/pdf/2404.19245.pdf)
[55](https://arxiv.org/pdf/2503.24354.pdf)
[56](https://arxiv.org/html/2401.13942v1)
[57](https://huggingface.co/blog/lora)
[58](https://ieeexplore.ieee.org/abstract/document/11084593/)
[59](https://papers.nips.cc/paper_files/paper/2024/file/8716aa6a02bcc3c8e69a3a42be192236-Paper-Conference.pdf)
[60](https://ieeexplore.ieee.org/document/10504785/)
[61](https://arxiv.org/abs/2412.04429)
[62](http://link.springer.com/10.1007/978-3-540-73400-0)
[63](http://arxiv.org/pdf/2406.06973.pdf)
[64](https://arxiv.org/html/2410.23370)
[65](http://arxiv.org/pdf/2307.09233.pdf)
[66](https://arxiv.org/pdf/2201.05729.pdf)
[67](http://arxiv.org/pdf/2306.08658.pdf)
[68](http://arxiv.org/pdf/2407.01408.pdf)
[69](https://arxiv.org/pdf/2112.02399.pdf)
[70](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05391.pdf)
[71](https://openai.com/index/simplifying-stabilizing-and-scaling-continuous-time-consistency-models/)
[72](https://openaccess.thecvf.com/content/WACV2025/papers/Colussi_ReC-TTT_Contrastive_Feature_Reconstruction_for_Test-Time_Training_WACV_2025_paper.pdf)
[73](https://www.cs.cornell.edu/gomes/pdf/2025_kong_aistats_diffusion.pdf)
[74](https://proceedings.neurips.cc/paper_files/paper/2024/file/5ab6f836f464d0f4e4f6aaa523249280-Paper-Conference.pdf)
[75](https://eccv.ecva.net/virtual/2024/poster/1800)
[76](https://openaccess.thecvf.com/content/CVPR2024/papers/Lee_DiSR-NeRF_Diffusion-Guided_View-Consistent_Super-Resolution_NeRF_CVPR_2024_paper.pdf)
[77](https://aclanthology.org/2024.emnlp-main.1257.pdf)
