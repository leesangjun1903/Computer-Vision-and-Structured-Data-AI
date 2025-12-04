
# Diffusion models for Handwriting Generation

## 1. 논문의 핵심 주장과 주요 기여

**"Diffusion models for Handwriting Generation"**은 온라인 손글씨 생성을 위해 **확산 확률 모델(Diffusion Probabilistic Model, DPM)**을 처음으로 적용한 획기적인 연구입니다.[1]

### 핵심 주장
- **기존 방법의 한계 극복**: RNN 기반 모델은 온라인 필체 데이터만 지원하고, GAN 기반 모델은 학습 불안정성과 모드 붕괴(mode collapse) 문제를 겪습니다. 확산 모델은 이러한 문제들을 우아하게 해결합니다.
- **보조 네트워크 불필요**: 텍스트 인식 네트워크, 스타일 분류 네트워크 등의 보조 손실 함수가 필요 없습니다.[1]
- **오프라인 스타일 특성 활용**: 온라인 데이터를 기록할 필요 없이 오프라인 이미지에서 직접 필체 스타일을 추출할 수 있습니다.[1]

### 주요 기여
1. **조건부 온라인 손글씨 생성**: 텍스트 콘텐츠와 필자 스타일을 동시에 제어하는 단순한 확산 모델 구축
2. **개선된 샘플링 절차**: 기존 샘플링 방식(Equation 9) 대신 개선된 방식(Equation 12)을 도입하여 더 사실적인 표본 생성
3. **스타일-텍스트 결합**: MobileNetV2를 통한 스타일 특성 추출과 주의 메커니즘(Attention)을 활용한 텍스트-스타일 정렬

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조

### 2.1 해결하고자 하는 문제

손글씨 생성의 근본적인 도전 과제들:

**콘텐츠 정확성**: 생성된 텍스트가 정확하게 철자되어야 합니다.[1]

**스타일 충실성**: 특정 필자의 고유한 서체 특성을 보존해야 합니다.[1]

**데이터 형식의 유연성**: 온라인 형식(펜의 궤적)과 오프라인 형식(정적 이미지) 모두를 처리해야 합니다.[1]

**모델 안정성**: GAN 기반 방법의 학습 불안정성을 해결해야 합니다.[1]

### 2.2 제안 방법: 확산 확률 모델

#### 수학적 기초

**순방향 확산 과정**:

$$q(y_{1:T} | y_0) = \prod_{t=1}^{T} q(y_t | y_{t-1}), \quad q(y_t | y_{t-1}) = \mathcal{N}\left(y_t; \sqrt{1-\beta_t} y_{t-1}, \beta_t I\right)$$[1]

여기서 $\{β_1, ..., β_T\}$는 고정된 노이즈 스케줄입니다.[1]

**역방향 과정** (생성 과정):

$$p_\theta(y_0:T) = p(y_T) \prod_{t=1}^{T} p_\theta(y_{t-1} | y_t)$$[1]

$$p_\theta(y_{t-1} | y_t) = \mathcal{N}(y_{t-1}; \mu_\theta(y_t, t), \sigma_t^2 I)$$[1]

**닫힌 형태의 ELBO** (증거 하한):

```math
\text{ELBO} = \mathbb{E}_q\left[D_{KL}(q(y_T | y_0) \| p(y_T)) + \sum_{t=2}^{T} D_{KL}(q(y_{t-1} | y_t, y_0) \| p_\theta(y_{t-1} | y_t)) - \log p_\theta(y_0 | y_1)\right]
```

[1]

**노이즈 스케일 계산**:

$$\bar{\alpha}_t = \prod_{s=1}^{t} (1-\beta_s), \quad y_t = \sqrt{\bar{\alpha}_t} y_0 + \sqrt{1-\bar{\alpha}_t} \epsilon_0, \quad \epsilon_0 \sim \mathcal{N}(0, I)$$

[1]

**손실 함수**:

$$L_{t-1} = \mathbb{E}_{y_0, \epsilon} \left[\frac{1}{2\sigma_t^2}\left\|\epsilon - \epsilon_\theta\left(y_t, t\right)\right\|^2\right]$$

[1]

또는 점수 기반 재해석:

$$L_{t-1} = \mathbb{E}_{t, \epsilon} \left[C_t \left\|\epsilon - \epsilon_\theta(y_t, t)\right\|^2\right], \quad C_t = \frac{\sigma_t^2}{2\sigma_{t-1}^2}$$

[1]

#### 개선된 샘플링 절차

**원본 방식** (Equation 9):

$$y_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left(y_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(y_t, t)\right) + \sqrt{\beta_t} z$$

[1]

**개선된 방식** (Equation 12):

$$y_0 \approx \frac{y_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(y_t, t)}{\sqrt{\bar{\alpha}_t}}$$

$$y_{t-1} = \sqrt{\bar{\alpha}_{t-1}} y_0 + \sqrt{1-\bar{\alpha}_{t-1}} z, \quad z \sim \mathcal{N}(0, I)$$

[1]

이 개선된 방식은 더 현실적인 표본을 생성하지만, 다양성이 약간 감소하는 트레이드오프가 있습니다.[1]

### 2.3 조건부 손글씨 생성 메커니즘

**이진 변수 처리**: 펜의 상태(뜬 상태/내린 상태)는 베르누이 분포로 모델링합니다:
$$L_{\text{drawn}} = -d_0 \log d_0 + (1-d_0) \log(1-d_0)$$[1]

**노이즈 레벨 조건화**: 연속적인 노이즈 레벨 조건화를 위해:
$$l = \log \frac{\sigma}{\sigma_t}, \quad \text{Sampling: } \sigma \sim \text{Uniform}(l_{t-1}, l_t)$$[1]

**스타일 특성 추출**: MobileNetV2를 통해 오프라인 이미지에서 국소 특성을 추출하고, 텍스트 시퀀스와의 주의(Attention)를 계산합니다.[1]

### 2.4 모델 구조

#### 전체 아키텍처

모델은 **텍스트-스타일 인코더**와 **확산 확률 모델**로 구성됩니다.[1]

**텍스트-스타일 조건화**:
- MobileNetV2: 필자 샘플 이미지에서 특성 추출
- 문자 수준 임베딩: 텍스트 시퀀스 표현
- 주의 메커니즘: 텍스트 시퀀스와 추출된 특성 간의 정렬 계산
- 피드포워드 네트워크: 최종 조건 생성

**노이즈 레벨 조건화**:
- 2개의 완전 연결층으로 이루어진 피드포워드 네트워크
- 모든 합성곱 계층 이후에 조건부 아핀 변환(Affine transformation) 적용

**확산 모델 구조**:
- **합성곱 블록** (Figure 7):
  - 3개의 합성곱 계층 + 합성곱 스킵 연결
  - 각 합성곱 계층 이후 조건부 아핀 변환 적용
  
- **주의 블록** (Figure 8):
  - 2개의 다중 헤드 주의 계층 + 피드포워드 네트워크
  - 첫 번째 주의: 스트로크 시퀀스와 텍스트-스타일 인코더 출력 간
  - 두 번째 주의: 자기 주의(Self-attention)
  - 정현파 위치 인코딩(Sinusoidal positional encoding) 추가
  - 계층 정규화(Layer normalization) 후 조건부 아핀 변환

#### 훈련 알고리즘

**Algorithm 1: Training**
```
while not converged do
  y₀ ~ q(y₀)
  t ~ Uniform(1, ..., T)
  σ ~ Uniform(l_{t-1}, l_t)
  ε ~ N(0, I)
  y_t = √ᾱ_t · y₀ + √(1-ᾱ_t) · ε
  Take gradient descent step on L_stroke + L_drawn
end
```

**Algorithm 2: Sampling**
```
y_T ~ N(0, I)
for t = T, ..., 1 do
  z ~ N(0, I)
  y_{t-1} = √((1-α_t)/(1-ᾱ_t)) y_t + √((1-ᾱ_{t-1})α_t)/(1-ᾱ_t)) ε_θ(y_t, c, s, σ_t)
  y_{t-1} = y_{t-1} + √(β_t) · z if t > 1
  d₀ = d_θ(y_t, c, s, σ_t)
end
return y₀, d₀
```

## 3. 성능 향상 및 한계

### 3.1 성능 향상

**객관적 평가 지표**:
- **FID (Fréchet Inception Distance)**: 모델의 7.10 vs 실제 데이터의 2.91[1]
- **기하 점수 (Geometry Score, GS)**: 모델의 3.3×10⁻³ vs 실제 데이터의 5.4×10⁻⁴[1]

절제 연구에서 개선된 샘플링 절차는 상대적으로 높은 FID 점수(8.05 → 7.10)를 보였지만, 기하 점수에서는 현저히 낮은 값(2.7×10⁻³ → 3.3×10⁻³)으로 더 사실적인 결과를 나타냈습니다.[1]

**주의 가중치 분석**: 텍스트-스트로크 정렬이 명시적으로 학습되며, 역방향 과정의 초반(높은 노이즈 단계)에도 대각선 정렬이 유지됩니다.[1]

**스타일 보간**: 두 필자의 스타일 사이를 부드럽게 보간할 수 있습니다.[1]

### 3.2 한계

**불완전한 표현 학습**: GAN 기반 방법들이 표본 다양성 부족으로 인해 높은 오류율(CER 39.8-39.9%)을 보이는 반면, 이 논문도 정확한 글자 생성에서 완전하지 못합니다.[1]

**계산 비용**: 60개의 확산 단계가 필요하며, 각 단계마다 전체 신경망을 통과해야 합니다.[1]

**장문 시퀀스 처리**: 최대 문장 길이에 제한이 있습니다.[1]

## 4. 모델의 일반화 성능 향상 가능성

### 4.1 현재 일반화 성능 분석

**도메인 내 성능**: IAM 온라인 데이터베이스에서 학습한 모델은 같은 필자와 같은 데이터셋 내에서 양호한 성능을 보입니다.[1]

**도메인 외 시나리오의 도전**: 
- 보이지 않은 필자의 스타일 생성 능력 제한
- 새로운 언어나 문자 체계로의 확장 어려움
- 서로 다른 필기 스타일(필기체, 인쇄체 혼합) 처리 미흡

### 4.2 일반화 성능 향상 전략

**1. 마스크된 자동인코더 (Masked Autoencoder, MAE) 기반 스타일 추출**

최근 연구(Brandenbusch 2024)에서는 필자 ID 임베딩 대신 MAE를 사용하여 보이지 않은 필자 스타일 생성을 달성했습니다. 이는 MobileNetV2만을 사용하는 것보다 더 강력한 특성 추출을 가능하게 합니다.[2]

**2. 반지도 학습 (Semi-Supervised Learning)**

미표기 데이터를 포함한 반지도 훈련 체계는 새 데이터셋으로의 도메인 적응을 향상시킵니다. 텍스트 조건화를 마스킹하면서 스타일 특성은 유지하는 방식으로, 레이블이 없는 데이터도 활용할 수 있습니다.[2]

**3. 하이브리드 스타일 인코더 (Metric Learning + Classification)**

DiffusionPen(Nikolaidou et al. 2024)은 삼중 손실(Triplet Loss)과 분류 손실을 결합하여:
- 필자 간 거리를 최대화
- 동일 필자의 스타일 변동을 최소화
- 연속적이고 의미 있는 스타일 공간 구성[3]

**4. 다중 스케일 스타일 특성 (Multi-scale Style Features)**

Layout Stroke Imitation(Hanif & Latecki 2025)에서는 국소 및 전역 스타일 특성을 다중 스케일로 추출하고, 단어 간격(레이아웃)을 명시적 특성으로 포함하여 일반화를 개선했습니다.[4]

**5. Glyph-Style 분리 (Disentanglement)**

DiffInk(Pan et al. 2025)는 InkVAE를 통해:
- **글리프 정확도 손실**: OCR 기반 손실로 문자 수준 정확성 강제
- **스타일 분류 손실**: 필기 스타일 보존
- **잠재 공간 분해**: 콘텐츠와 스타일을 효과적으로 분리[5]

이러한 분리는 보이지 않은 스타일과 단어에 대한 일반화를 크게 향상시킵니다.

**6. 분류기 자유 가이던스 (Classifier-Free Guidance)**

적절한 가이던스 스케일(예: 2-3)의 선택은 생성 품질과 다양성 간의 균형을 최적화합니다.[2]

### 4.3 일반화 성능의 정량적 개선

| 방법 | CER (%) | WER (%) | 주요 개선 |
|------|---------|---------|----------|
| 원본 논문 (이 논문) | - | - | FID: 7.10 (baseline) |
| WordStylist (2021) | 8.26 | 23.36 | 클래스 기반 스타일 |
| DiffusionPen (2024) | 6.94 | 18.11 | 보이지 않은 스타일 생성 |
| Semi-Supervised 적응 (2024) | 7.2-7.8 | 19.5-21.2 | 새 데이터셋 도메인 적응 |
| DiffInk (2025) | 최고 성능 | 최고 성능 | 전체 라인 생성, 글리프-스타일 분리 |[2][3][4][5]

## 5. 논문이 앞으로의 연구에 미치는 영향

### 5.1 기본 아이디어의 확산

이 논문은 **생성 모델로서의 확산 모델의 유효성**을 손글씨 생성 분야에 증명했습니다. 이후 6년간의 연구에서:

**확산 기반 방법이 표준 접근법으로 확립**: GAN 기반 방법에서 확산 기반 방법으로의 산업 표준 전환[3][2]

**모드 붕괴 해결**: 확산 모델의 내재적 특성이 GAN의 모드 붕괴 문제를 자연스럽게 해결[1]

**보조 네트워크 제거**: 더 간결하고 안정적인 학습 절차 가능[1]

### 5.2 스타일 표현 방법론의 발전

**특성 추출 개선**:
- 단순 분류 (Wordstylist): 클래스 임베딩
- 메트릭 학습 (DiffusionPen): 삼중 손실 + 분류 손실
- 자기지도 학습 (Semi-supervised): 마스크된 자동인코더
- 글리프 인식 분리 (DiffInk): OCR 기반 + 스타일 분류 손실[4][5][3][2]

**조건화 메커니즘 고도화**:
- 텍스트-스타일 교차 주의 (원본 논문)
- 내용 인코더를 통한 다양한 조건화 방식 (Semi-supervised)
- 변압기 기반 잠재 확산 (DiffInk)
- 다중 모드 가이던스 (최신 연구)[5][3][4][2]

### 5.3 도메인 적응 및 일반화 연구로의 확대

**도메인 외 시나리오 처리**: 
- 보이지 않은 필자 스타일 생성
- 새로운 데이터셋으로의 도메인 적응
- 다국어 손글씨 생성

**확산 기반 일반화**: 더 넓은 범위의 컴퓨터 비전 작업에서 확산 모델을 도메인 일반화 도구로 활용[6][7]

### 5.4 실무 응용 분야 확대

**데이터 증강**: 손글씨 인식 시스템의 학습 데이터 생성[3][2]

**필체 개인화**: 특정 사용자의 필체를 학습하여 개인화된 손글씨 생성[1]

**접근성**: 서술 장애가 있는 사람들의 문서 작성 지원[2]

**포렌식 및 보안**: 손글씨 생성 및 검증 기술 발전[3]

### 5.5 아키텍처 발전의 촉발

**변압기 통합**: 초기 합성곱 기반 주의에서 완전 변압기 기반 아키텍처로 진화[5]

**다중 목적 손실**: 복수의 정규화 손실을 통한 더 나은 특성 학습[5]

**효율성 개선**: 온라인 생성에서 전체 라인 생성으로 확대하면서도 효율성 유지[5]

## 6. 앞으로의 연구 시 고려할 점

### 6.1 기술적 고려사항

**1. 확산 단계 최적화**
- 현재 60개 단계는 DDIM(Denoising Diffusion Implicit Models) 등 고속 샘플링 방법으로 감소 가능
- 원샷 확산 모델(One-shot diffusion) 기술 활용으로 추론 속도 향상

**2. 스타일 공간 구조화**
- 더 의미 있는 스타일 공간 구축 (현재: 일부 필자의 스타일만 잘 표현)
- 연속적 스타일 보간의 신뢰성 향상

**3. 콘텐츠-스타일 균형**
- 최근 DiffInk의 글리프-스타일 분리 접근이 유망하나, 더 정교한 분리 메커니즘 개발 필요
- 장문 문장 생성 시 스타일 일관성 유지

**4. 다국어 및 다중 스크립트 지원**
- 현재: 영문 라틴 문자 중심
- 필요: 한글, 중국어, 아랍어 등 다양한 필기 체계 지원
- 문자 특성의 언어별 차이 처리

**5. 계산 효율성**
- 변압기 기반 모델의 복잡성 증가
- 모바일 환경에서의 실시간 생성 가능성 탐색
- 매개변수 효율적 미세 조정 (LoRA, QLoRA 등)

### 6.2 데이터 및 평가 개선

**1. 데이터셋 확장**
- 다양한 필기 특성 포함 (왼손/오른손, 필기체/인쇄체)
- 다양한 연령, 문화적 배경의 필자 포함
- 필기 장애 또는 특이한 필기 패턴

**2. 평가 지표 정교화**
- FID, 기하 점수 이외의 필기 특화 지표 개발
- 글자 인식 성능뿐 아니라 **정확한 글자 생성률 (Character Accuracy)** 평가
- 필기 스타일 유사도를 측정하는 객관적 지표 (Writer Identity Preservation Index)

**3. 다중 작업 평가**
- 생성된 이미지의 인식 성능
- 인간 평가자 판별 능력 (Turing test 유사)
- 필체 개인 식별 불가능성 평가

### 6.3 이론적 진전

**1. 확산 과정의 수렴성 분석**
- 조건부 확산의 이론적 수렴 보장
- 다양한 조건화 전략의 수렴 특성 비교

**2. 일반화 한계 이론**
- 보이지 않은 스타일로의 일반화 한계 분석
- 도메인 시프트에 대한 강건성 이론

**3. 스타일 공간의 기하학**
- 필체 스타일 공간의 내재 차원(Intrinsic Dimension) 분석
- 스타일 관련 특성의 수학적 특성화

### 6.4 응용 및 윤리적 고려사항

**1. 응용 분야 확대**
- **문서 정규화**: 스캔된 손글씨 문서의 자동 정규화
- **폰트 생성**: 사용자 필체 기반 개인 폰트 생성
- **필체 치료**: 필기 장애 환자의 재활 지원
- **교육**: 학생 필기 스타일 분석 및 개선 제안

**2. 신원 보호 및 보안**
- 생성된 손글씨가 추적 불가능하도록 보장
- 서명 위조 방지 기술과의 상호작용
- 법적, 윤리적 프레임워크 개발

**3. 데이터 프라이버시**
- 학습 데이터의 개인 정보 보호
- 생성 모델에서의 멤버십 추론 공격 방어
- 차등 프라이버시(Differential Privacy) 적용

**4. 포렌식 함의**
- 손글씨 위조 탐지 기술 개발
- 인공 생성 손글씨의 법적 증거 가치 평가
- 필기 분석가를 위한 도구 개발

## 7. 2020년 이후 관련 최신 연구 동향 종합

### 7.1 주요 연구 흐름 (2020-2025)

**Phase 1: 확산 모델의 도입 (2020-2021)**
- **Diffusion models for Handwriting Generation (2020)**: 이 논문 - 기초 확립
- **WordStylist (2021)**: 잠재 확산 모델로의 확장, 클래스 기반 스타일

**Phase 2: 스타일 표현 고도화 (2021-2023)**
- **GANwriting 확장 연구들**: 보조 손실 함수 개선
- **Transformer 기반 방법들**: Vision Transformer 통합
- **확산 모델의 안정성 이점** 입증

**Phase 3: 일반화 및 적응 (2023-2024)**
- **DiffusionPen (2024)**: 보이지 않은 필자 스타일 생성 (메트릭 학습)
- **Semi-Supervised Adaptation (2024)**: 도메인 적응을 위한 반지도 학습
- **DiffusionBERT 및 다른 텍스트 확산**: 이산 토큰 처리 개선

**Phase 4: 전체 라인 및 멀티모달 생성 (2024-2025)**
- **DiffInk (2025)**: 글리프-스타일 분리, 전체 라인 생성
- **ScriptViT (2025)**: Vision Transformer 기반 스타일 인코더
- **Layout Stroke Imitation (2025)**: 레이아웃 가이드 생성

### 7.2 핵심 기술 진화

| 기술 요소 | 초기 (2020) | 현재 (2024-2025) | 개선사항 |
|----------|-----------|-----------------|---------|
| 스타일 추출 | MobileNetV2 | MAE, Vision Transformer | 자기지도 학습으로 레이블 불필요 |
| 조건화 | 교차 주의 | 변압기 기반 다중 경로 | 콘텐츠-스타일 명시적 분리 |
| 생성 범위 | 단어 | 전체 라인 및 문단 | 효율성 유지하면서 확대 |
| 도메인 적응 | 단일 도메인 | 반지도 + 무지도 | 새 스타일/언어로 확장 가능 |
| 성능 (CER %) | 미평가 | 6.94-7.8 | FID 기준 75% 개선 |[2][3][4][5]

### 7.3 오픈 문제 및 미해결 과제

**1. 극도로 어려운 필기 체계**
- 복잡한 연결 구조 (예: 아랍어, 콘선 필기)
- 필기 흔적 간 높은 중첩도

**2. 스타일 정교함**
- 필기 압력 변화 (온라인에서 추정 필요)
- 필기 속도에 따른 미묘한 변화
- 시간에 따른 필기 스타일 변화

**3. 효율성과 확장성**
- 실시간 생성 가능성 (현재: 오프라인)
- 수천 명의 필자 스타일 동시 처리

**4. 강건성**
- 노이즈가 많은 입력 스타일 샘플 처리
- 도메인 시프트에 대한 강건성 (필기 기구, 표면, 조명 변화)

## 결론

"Diffusion models for Handwriting Generation" 논문은 **확산 확률 모델**을 손글씨 생성에 처음 적용하여 GAN과 RNN 기반 방법의 한계를 우아하게 극복했습니다. 단순하면서도 효과적인 아키텍처, 개선된 샘플링 절차, 그리고 보조 네트워크 없이 스타일을 직접 처리할 수 있는 능력은 이후 6년간의 연구 방향을 결정했습니다.

2024-2025년의 최신 연구들(DiffusionPen, Semi-supervised Adaptation, DiffInk, ScriptViT, Layout Stroke Imitation)은 이 기초 위에서:

**도메인 일반화**: 보이지 않은 필자, 새로운 언어, 새로운 데이터셋으로의 확장[4][2][3][5]

**표현 정교화**: 메트릭 학습, 마스크된 자동인코더, 글리프-스타일 분리를 통한 더 나은 특성 학습[4][2][3][5]

**생성 확대**: 단어에서 전체 라인, 문단 수준으로의 확장[5]

**효율성 개선**: 변압기 기반 아키텍처와 다양한 최적화 기법으로 추론 속도 향상[5]

을 달성했습니다.

앞으로의 연구는 **다국어 지원, 실시간 생성, 향상된 평가 지표, 윤리적 프레임워크** 등에 초점을 맞춰야 하며, **확산 기반 일반화의 이론적 기초**를 더욱 견고히 할 필요가 있습니다.[7][6][2][3][4][5]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/184da9c6-bbb0-4de5-afb8-318a7afcf58c/2011.06704v1.pdf)
[2](https://www.semanticscholar.org/paper/2aab1a79341e4967e31b8efab4dfaf1f96596b74)
[3](https://ieeexplore.ieee.org/document/10972549/)
[4](https://arxiv.org/abs/2509.23624)
[5](https://www.semanticscholar.org/paper/7ab5295eae0a323edb3133079a93fdc58460e45d)
[6](https://arxiv.org/abs/2509.15678)
[7](https://dl.acm.org/doi/10.1145/3746027.3762245)
[8](https://arxiv.org/abs/2502.00688)
[9](https://www.frontiersin.org/articles/10.3389/fdgth.2025.1653369/full)
[10](https://aacrjournals.org/cancerres/article/85/8_Supplement_1/2439/758009/Abstract-2439-Spatial-transcriptomics-informed)
[11](https://link.springer.com/10.1007/s10032-025-00533-x)
[12](http://arxiv.org/pdf/2409.06065.pdf)
[13](https://arxiv.org/pdf/2412.15853.pdf)
[14](https://arxiv.org/abs/2403.01693)
[15](https://arxiv.org/pdf/2212.05895.pdf)
[16](https://arxiv.org/abs/2409.00786)
[17](http://arxiv.org/pdf/2408.07259.pdf)
[18](https://arxiv.org/html/2503.08133v1)
[19](https://arxiv.org/html/2409.04004)
[20](https://arxiv.org/html/2508.03256v1)
[21](https://arxiv.org/html/2505.13235v1)
[22](https://peerj.com/articles/cs-1905/)
[23](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/dmhg/)
[24](https://www.sciencedirect.com/science/article/pii/S0031320325000172)
[25](https://openreview.net/pdf?id=3s9IrEsjLyk)
[26](https://www.semanticscholar.org/paper/Diffusion-models-for-Handwriting-Generation-Luhman-Luhman/2aab1a79341e4967e31b8efab4dfaf1f96596b74)
[27](https://arxiv.org/html/2508.21040v1)
[28](https://www.ijcai.org/proceedings/2023/0750.pdf)
[29](https://www.manuscriptlink.com/society/kips/conference/ask2025/file/downloadSoConfManuscript/abs/KIPS_C2025A0202F)
[30](https://ieeexplore.ieee.org/document/10484417/)
[31](http://www.proceedings.com/079017-2351.html)
[32](https://ieeexplore.ieee.org/document/10943719/)
[33](https://ieeexplore.ieee.org/document/10678654/)
[34](https://arxiv.org/abs/2406.18516)
[35](https://ieeexplore.ieee.org/document/10657563/)
[36](https://ieeexplore.ieee.org/document/10635862/)
[37](https://arxiv.org/abs/2411.01168)
[38](https://ieeexplore.ieee.org/document/10561561/)
[39](https://dl.acm.org/doi/10.1145/3707292.3707367)
[40](https://arxiv.org/abs/2404.00095)
[41](https://arxiv.org/pdf/2402.04929.pdf)
[42](http://arxiv.org/pdf/2310.09213.pdf)
[43](http://arxiv.org/pdf/2503.06698.pdf)
[44](https://arxiv.org/html/2408.03353v2)
[45](https://arxiv.org/pdf/2410.16020v1.pdf)
[46](https://arxiv.org/pdf/2207.03442.pdf)
[47](https://arxiv.org/pdf/2311.18071.pdf)
[48](https://www.sciencedirect.com/science/article/abs/pii/S0893608024009602)
[49](https://cdn.techscience.cn/files/cmc/2024/online/CMC0520/TSP_CMC_49007/TSP_CMC_49007.pdf)
[50](https://github.com/MingkunLei/Awesome-Style-Transfer-with-Diffusion-Models)
[51](https://openaccess.thecvf.com/content/ICCV2025/papers/He_Boosting_Domain_Generalized_and_Adaptive_Detection_with_Diffusion_Models_Fitness_ICCV_2025_paper.pdf)
[52](https://arxiv.org/html/2505.16360v2)
[53](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05806.pdf)
[54](https://dl.acm.org/doi/10.1016/j.patcog.2025.111357)
[55](https://www.nature.com/articles/s41598-025-17899-x)
[56](https://arxiv.org/abs/2312.01850)
