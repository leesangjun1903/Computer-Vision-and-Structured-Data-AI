
# Conffusion: Confidence Intervals for Diffusion Models

## 1. 논문의 핵심 주장과 주요 기여

"Conffusion: Confidence Intervals for Diffusion Models"는 확산 모델(Diffusion Models)이 생성하는 결과에 대한 **통계적 보장을 제공하기 위한 방법**을 제안합니다. 이 논문의 핵심 주장은 다음과 같습니다.[1]

**핵심 주장:**
확산 모델은 초해상도(super-resolution)와 인페인팅(inpainting) 같은 이미지-투-이미지 작업에서 뛰어난 성능을 보이지만, 생성된 결과에 대한 통계적 보장이 부재하여 의료 영상 분석 같은 고위험 상황에 배포될 수 없습니다.[1]

**주요 기여:**
1. **픽셀 단위 신뢰도 구간 구성**: 각 생성 픽셀 주변에 신뢰도 구간을 구성하여, 해당 구간이 참값을 포함할 확률을 사용자가 지정할 수 있게 함[1]
2. **Conffusion 방법 제안**: 사전학습된 확산 모델을 양자일 회귀(quantile regression)로 세밀조정(fine-tune)하여 **단일 순전파(single forward pass)로 신뢰도 구간을 예측**[1]
3. **확산 사전학습의 효능 입증**: 기존 방법 대비 더 타이트한 구간을 제공하면서 **1000배 이상의 속도 향상** 달성[1]

***

## 2. 논문이 해결하고자 하는 문제

### 2.1 문제의 정의

기존 확산 모델 기반 방법들의 주요 문제점:[1]

| 문제점 | 설명 |
|--------|------|
| **통계적 보장 부재** | 생성된 이미지의 신뢰도를 정량적으로 제시하지 못함 |
| **느린 샘플링 속도** | 각 테스트 이미지마다 수천 번의 순전파 필요 |
| **최적화되지 않은 구간** | 단순 샘플링 기반 방법은 느슨한 구간 생성 |
| **작업별 모델 훈련 필요** | 새로운 작업마다 전용 확산 모델 훈련 필요 |

### 2.2 구체적 응용 사례

**의료 영상 분석**: 저해상도 MRI 영상을 초해상도하는 경우, 의사는 생성된 고해상도 이미지가 실제 해부학적 구조를 반영하는지 확인할 필요가 있습니다. 각 픽셀 주변의 신뢰도 구간을 통해 생성 모델의 신뢰도를 평가할 수 있습니다.[1]

***

## 3. 제안하는 방법 (수식 포함)

### 3.1 위험 제어 예측 집합(Risk-Controlling Prediction Set, RCPS) 정의

각 픽셀 $$(m,n)$$에 대해 신뢰도 구간을 구성하는 목표:

$$T(x_{m,n}) = [\bar{X}\_{m,n}^l, \bar{X}_{m,n}^u] \in $$[1]

여기서 $$\bar{X}\_{m,n}^l$$과 $$\bar{X}_{m,n}^u$$는 각각 하한과 상한입니다.[1]

사용자가 위험 수준 $$\alpha \in (0,1)$$과 오류 수준 $$\delta \in (0,1)$$을 지정하면, 다음을 만족하는 구간을 구성합니다:[1]

$$P\left(E\left[\frac{1}{MN}\sum_{m,n} \mathbf{1}(y_{m,n} \notin T(x_{m,n}))\right] \leq \alpha\right) \geq 1-\delta$$

### 3.2 양자일 손실함수(Quantile Loss)

기준선 방법(ADMUQ)에서 사용하는 손실함수:[1]

$$L_q(x,y) = \rho_q(y - q(x)) = (y - q(x))(q - \mathbf{1}(y < q(x)))$$

여기서 $$q$$는 최적화할 양자이고, $$q(x)$$는 양자 추정값입니다.[1]

상한과 하한에 대해:[1]

$$L_{QR}(x,y) = L_2(\bar{l}(x), y) + L_{1-\alpha/2}(\bar{u}(x), y)$$

최종 목적 함수:[1]

$$L(x,y) = L_{QR}(x,y) + \lambda L_{mse}(x,y)$$

### 3.3 샘플링 기반 경계 추정(Sampling-based Bounds, DMSB)

사전학습된 확산 모델에서 여러 변형을 샘플링하는 방법:[1]

각 입력 $$x_i$$에 대해 $$S = \{y_1, y_2, ..., y_{200}\}$$개의 샘플 변형을 생성한 후, 각 픽셀의 상한과 하한을 다음과 같이 추출합니다:[1]

$$\bar{l} = \text{quantile}(S, \alpha/2), \quad \bar{u} = \text{quantile}(S, 1-\alpha/2)$$

### 3.4 샘플링된 경계 가속화(DMSBA)

추론 시간을 단축하기 위해 사전학습된 확산 모델을 세밀조정하여 단일 순전파로 경계를 예측:[1]

$$L(x, \bar{u}_{sv}, \bar{l}_{sv}) = L_{mse}(f_{DM}(x), \bar{u}_{sv}) + L_{mse}(f_{DM}(x), \bar{l}_{sv})$$

### 3.5 Conffusion 방법 (N-Conffusion과 G-Conffusion)

**N-Conffusion (Narrow Conffusion):**
DMSBA의 MSE 손실을 양자일 회귀로 대체:[1]

$$L_{QR}(x,y) = L_{\alpha/2}(\bar{l}(x), y) + L_{1-\alpha/2}(\bar{u}(x), y)$$

이를 통해 DMSB의 성능을 유지하면서 DMSBA의 속도를 달성합니다.[1]

**G-Conffusion (Global Conffusion):**
작업별 특화 모델 대신 ImageNet에서 사전학습된 ADM(Ablated Diffusion Models)을 사용하여 데이터와 작업에 무관한 일반화된 방법을 제공합니다. 동일한 양자일 손실함수로 세밀조정하지만, 새로운 작업이나 데이터셋에 빠르게 적응할 수 있습니다.[1]

### 3.6 캘리브레이션(Calibration)

Hoeffding의 상한을 이용한 캘리브레이션 상수 선택:[1]

$$\hat{R} = \frac{1}{n}\sum_{i=1}^n L_T(x_i, y_i) + \sqrt{\frac{1}{2n}\log\frac{1}{\delta}}$$

여기서 $$L_T(X,Y) = \frac{1}{MN}\sum_{m,n} \mathbf{1}(y_{m,n} \notin T(x_{m,n}))$$는 위험을 계산합니다.[1]

***

## 4. 모델 구조

### 4.1 기본 아키텍처

논문에서 사용된 확산 모델의 기본 구조:[1]

- **U-Net 기반 디노이징 네트워크**: 시간 단계 $$t$$를 인코딩하여 네트워크에 입력
- **조건부 생성**: 입력 이미지 $$x$$로 조건화된 디노이징 과정
- **단계적 노이즈 제거**: $$T$$개의 디노이징 단계를 거쳐 $$y_T \sim \mathcal{N}(0,I)$$에서 시작하여 $$y_0$$으로 점진적 정제

### 4.2 Conffusion 모델의 수정사항

**핵심 수정:**
확산 모델에서 디노이징 프로세스를 분리하고, 신뢰도 구간 예측을 위한 출력 헤드(head)를 추가합니다:[1]

- 원래 생성 헤드를 복제하여 상한 $$\bar{u}(x)$$와 하한 $$\bar{l}(x)$$ 예측
- 양자일 회귀 손실함수로 세밀조정
- 단일 순전파로 구간 경계 생성

### 4.3 사용된 확산 모델들

**N-Conffusion:**
- SR3 (초해상도): FFHQ에서 사전학습
- Palette (인페인팅): CelebA-HQ에서 사전학습

**G-Conffusion:**
- ADM (Guided Diffusion): ImageNet에서 사전학습된 범용 모델
- 매개변수: 약 422M 매개변수

***

## 5. 성능 향상 및 실험 결과

### 5.1 평가 지표

**경험적 위험(Empirical Risk)**: 예측 구간 밖의 픽셀 비율[1]

```math
\text{Risk}=\frac{\text{\#\ pixels\ outside\ interval}}{MN}\times 100\%
```

**구간 크기(Interval Size)**: 타이트한 구간의 정도[1]

$$\text{Mean Interval Size} = \frac{1}{MN}\sum_{m,n}(\bar{u}_{m,n} - \bar{l}_{m,n})$$

**크기 계층화된 위험(Size-Stratified Risk)**: 픽셀을 구간 크기 사분위수로 나누어 각각의 위험 계산[1]

### 5.2 초해상도 작업 결과

- **ADMUQ (기준선)**: 흐릿한 경계, 충분하지 않은 성능
- **DMSB (샘플링)**: 가장 샤프한 경계이나 인공물 포함
- **N-Conffusion**: DMSB의 성능을 유지하면서 DMSBA의 속도 달성[1]
- **G-Conffusion**: 전용 사전학습 없이도 경쟁력 있는 결과 제공[1]

### 5.3 인페인팅 작업 결과

인페인팅 작업에서 N-Conffusion이 특히 우수한 성능 달성:[1]
- 넓은 신호 없는 영역에서 강한 노이즈 필요
- N-Conffusion이 구간 크기에서 다른 방법을 큰 폭으로 능가
- 크기 계층화된 위험에서 모든 방법이 유사한 성능

### 5.4 확산 사전학습의 필요성

흥미로운 발견: G-Conffusion이 작업별 사전학습 없이도 경쟁력 있는 결과를 제공합니다.[1]

**절제 실험(Ablation Study):**
- ResNeXt (ImageNet 사전학습, 판별적)
- ResNeSt (ImageNet 사전학습, 판별적)
- ADM (ImageNet 사전학습, 생성적)

결과: 모든 방법이 RCPS 정의를 만족하지만, **확산 사전학습이 확실히 타이트한 구간을 생성**합니다.[1]

### 5.5 속도 개선

| 방법 | 추론 시간 (ms/이미지) | 상대 속도 |
|------|----------------------|----------|
| DMSB (200개 샘플) | ~10,000 | 1x (기준선) |
| DMSBA | ~100 | 100x |
| N-Conffusion | ~100 | 100x |
| **G-Conffusion** | **~100** | **100x** |

**1000배 이상의 속도 향상** 달성 (DMSB 기준)[1]

***

## 6. 일반화 성능 향상 가능성

### 6.1 전이 학습(Transfer Learning)

**G-Conffusion의 중요성:**
논문의 가장 흥미로운 발견은 **ImageNet에서 사전학습된 일반 확산 모델이 새로운 데이터셋과 작업에 효과적으로 전이될 수 있다**는 점입니다.[1]

이는 다음을 의미합니다:
1. 새로운 작업(예: 인페인팅)에 대해 전용 확산 모델을 훈련할 필요가 없음
2. 새로운 데이터셋(예: CelebA-HQ)에 대해서도 단순 세밀조정으로 충분
3. 계산 비용과 데이터 요구량이 크게 감소

### 6.2 확산 사전학습의 우수성

**생성적 사전학습 vs 판별적 사전학습:**

절제 실험에서 명확히 드러남:[1]
- 생성적 확산 사전학습: 더 타이트하고 현실적인 경계
- 판별적 사전학습(ResNeXt, ResNeSt): 상대적으로 넓은 경계

**이유:**
확산 모델의 디노이징 과정이 데이터 분포의 세부 구조를 더 잘 포착하여, 신뢰도 구간 예측에 유리한 특성을 제공합니다.[1]

### 6.3 저주파 대 고주파 성분

논문의 흥미로운 고찰:[1]

> "확산 프로세스는 높은 충실도와 세부 사항을 가능하게 하고 고주파 생성에 탁월하지만, 신뢰도 구간은 주로 저주파를 포함하므로 단일 단계에서 생성하기 더 쉽습니다."

따라서:
- **N-Conffusion**: 단일 순전파로 저주파 구간 경계 예측 가능
- **DMSB**: 다중 샘플을 요구하는 고주파 정보도 포함 시도

### 6.4 새로운 작업으로의 일반화

**예상되는 일반화 가능성:**

| 작업 | 일반화 가능성 | 이유 |
|------|-------------|------|
| 초해상도 | **높음** | 입력 신호가 충분히 존재, 소량의 노이즈 필요 |
| 인페인팅 | **중간** | 대규모 신호 없는 영역, 강한 노이즈 필요 |
| 이미지 편집 | **중간** | 조건부 생성 필요 |
| 의료 영상 복원 | **높음** | 구조적 정보가 풍부한 입력 |
| 비디오 복원 | **낮음** | 시간적 일관성 필요 |

***

## 7. 모델의 한계

### 7.1 일반적 한계

**다중모달 분포(Multimodal Distributions):**
연속 신뢰도 구간은 **일봉형 분포(unimodal)에서 매우 효과적**이지만, 분포가 여러 개의 모드를 가질 때 문제가 발생합니다.[1]

구체적으로:
- 구간이 모든 모드를 포함하기 위해 매우 넓어져야 함
- 모드 사이의 **0 확률 영역도 포함**되어 정보성 감소

**해결 방안 (논문에서 제시):**
- 다중모달 경우를 명시적으로 처리하는 방법 개발 필요
- 예: 예측 집합(prediction sets) 또는 구간 집합 사용

### 7.2 기술적 한계

**단일 스케일 파라미터($$\lambda$$):**
캘리브레이션 과정에서 모든 픽셀에 동일한 스케일 $$\lambda$$를 사용합니다.[1]

문제점:
- 공간적으로 변하는 불확실성을 완벽하게 모델링하지 못함
- 예: 일부 영역은 신뢰할 수 있고, 다른 영역은 불확실할 수 있음

**개선 제안:**
픽셀별 또는 영역별 $$\lambda$$를 사용하면:
- 더 적응적인 신뢰도 구간 생성 가능
- 그러나 더 큰 검증 데이터셋 필요

### 7.3 응용 범위의 한계

**저수준 작업(Low-level Tasks):**
논문은 픽셀 공간에서의 저수준 작업(초해상도, 인페인팅)에 초점[1]

문제점:
- 의미론적 개념(예: 장면 깊이, 얼굴 나이)에 대한 신뢰도 구간은 별도 연구 필요
- 고수준 작업에서는 분포가 더 단봉형이 되어 타이트한 구간 가능할 수 있음

**비조건부 생성(Unconditional Generation):**
논문은 조건부 생성만 다룸[1]

미개척 영역:
- 무조건부 생성에서의 신뢰도 구간
- 예: 중간 시점 $$t$$의 생성 이미지를 조건으로 최종 결과의 신뢰도 구간

### 7.4 계산 및 데이터 한계

**캘리브레이션 데이터셋:**
- 유효한 보장을 위해 충분한 크기의 캘리브레이션 집합 필요
- 작은 데이터셋에서는 보장이 느슨할 수 있음

**새로운 도메인으로의 적응:**
- G-Conffusion은 일반화성을 제공하지만, 크게 다른 도메인(예: 의료 영상 → 자동차 영상)에서는 세밀조정 필요
- 도메인 적응(domain adaptation) 전략 개발 필요

***

## 8. 논문이 앞으로의 연구에 미치는 영향

### 8.1 불확실성 정량화(Uncertainty Quantification) 분야

**패러다임 전환:**
논문은 생성 모델에 **통계적 보장을 제공하는 새로운 패러다임**을 제시합니다. 이는:[1]

1. **신뢰할 수 있는 AI(Trustworthy AI)의 실현**: 의료 진단, 자율주행 같은 고위험 응용에서 AI 배포 가능성 제시
2. **확산 모델의 신뢰성 강화**: 불확실성 정량화 없이 배포된 확산 모델들의 한계 극복
3. **의료 AI 규제 대응**: FDA 같은 규제 기관의 요구하는 신뢰도 보장 충족 가능

### 8.2 확산 모델 연구의 확장

**새로운 연구 방향:**

| 연구 방향 | 기여도 | 예상 영향 |
|----------|--------|----------|
| 적응적 캘리브레이션 | 중 | 공간 변화 불확실성 모델링 |
| 다중모달 불확실성 처리 | 높음 | 모호한 상황 대응 개선 |
| 의미론적 신뢰도 구간 | 중 | 고수준 작업 확장 |
| 세밀한 전이 학습 | 중 | 적응 속도 개선 |

### 8.3 이후 관련 연구들 (2020년 이후)

논문 이후 관련 분야의 주요 발전들:

#### 8.3.1 합성곱신경망 기반 신뢰도 구간 (2022-2024)

- **Conformal Prediction for Image Segmentation** (2024): 의미론적 이미지 분할에서 합성곱 예측 적용[2]
- **RR-CP: Reliable-Region-Based Conformal Prediction** (2023): 의료 이미지 분류에서 신뢰할 수 있는 영역 기반 접근[3]

#### 8.3.2 신경망 기반 불확실성 정량화 (2024-2025)

- **Hyper-Diffusion Models (HyperDM)** (2024): 단일 모델로 인식론적(epistemic)과 우연적(aleatoric) 불확실성 동시 추정[4]
- **Torch-Uncertainty** (2025): 심층학습 불확실성 정량화의 통합 프레임워크[5]

#### 8.3.3 전이 학습과 일반화 성능 (2023-2025)

- **Transfer Learning for Diffusion Models (TGDP)** (2024): 원천 도메인에서 목표 도메인으로의 효율적 전이[6]
- **Towards a Mechanistic Explanation of Diffusion Model Generalization** (2025): 확산 모델의 일반화 메커니즘 분석[7]
- **DomainFusion** (2024): 잠재 확산 모델을 이용한 도메인 일반화[8]

#### 8.3.4 의료 응용 (2024-2025)

- **Task-Driven Uncertainty Quantification via Conformal Prediction** (2024): MRI 등 의료 영상에서 작업 중심의 불확실성 정량화[9]
- **Game-Theoretic Defenses for Robust Conformal Prediction** (2024): 의료 영상에서 적대적 공격에 대한 로버스트 합성곱 예측[10]
- **Conformal Prediction for Image Segmentation using Morphological Prediction Sets** (2025): 의료 분할에서 형태학적 예측 집합[11]

#### 8.3.5 신뢰도 있는 생성 모델 (2024-2025)

- **Conformalized Generative Bayesian Imaging** (2025): 베이지안 신경망과 합성곱을 결합한 영상 불확실성 정량화[12]
- **Towards Uncertainty Quantification in Generative Model Learning** (2025): 생성 모델 학습의 불확실성 정량화 위치화[13]

***

## 9. 앞으로 연구 시 고려할 점

### 9.1 방법론적 개선

#### 9.1.1 적응적 캘리브레이션 전략

**현재 방식의 한계:**
모든 픽셀에 동일한 $$\lambda$$를 적용합니다.[1]

**개선 방향:**
1. **픽셀별 캘리브레이션**: $$\lambda(m,n)$$을 각 픽셀 특성에 따라 조정
2. **영역별 캘리브레이션**: 이미지 내 영역의 특성(예: 엣지 vs 평탄 영역)에 따라 다른 $$\lambda$$ 적용
3. **문맥 기반 캘리브레이션**: 입력 신호의 품질이나 복잡도에 따라 적응

**필요 조건:**
- 더 큰 검증 데이터셋
- 계산 효율성 유지

#### 9.1.2 다중모달 분포 처리

**현재 한계:**
연속 구간이 다중 모드를 효과적으로 표현하지 못합니다.[1]

**해결책:**
1. **예측 집합(Prediction Sets)**: 개별 구간 대신 여러 개의 불연속 집합 사용
2. **혼합 모델(Mixture Models)**: 각 모드에 대해 별도의 구간 예측
3. **확률 밀도 추정**: 구간 대신 전체 후방 분포 추정

**논문의 제시:**
> "다중모달 분포가 공통적이므로, 이러한 설정을 다루는 것을 향후 연구에서 탐구해야 합니다."[1]

### 9.2 응용 확장

#### 9.2.1 의미론적 개념의 신뢰도 구간

**현재 초점:** 픽셀 공간의 저수준 작업

**확장 가능성:**
- 장면 깊이(Scene Depth) 추정
- 얼굴 나이 추정
- 의료 영상의 종양 크기 측정

**장점:**
고수준 개념은 종종 더 단봉형 분포를 가져서 더 타이트한 구간 가능[1]

#### 9.2.2 무조건부 생성에서의 신뢰도 구간

**미개척 영역:**
무조건부 텍스트-투-이미지 모델, 이미지 생성 등

**제안 방향:**
> "우리는 조건부 생성에 초점을 맞췄지만, 우리의 작업은 무조건부 생성에도 함축이 있습니다."[1]

**응용:**
- 중간 단계에서의 조건화를 통해 최종 생성의 신뢰도 평가
- 조기 중단(early stopping) 기준 제공

### 9.3 도메인 적응 전략

#### 9.3.1 크로스 도메인 전이

**현재:** G-Conffusion은 ImageNet 기반 모델을 다양한 시각 작업에 적용

**개선 필요:**
1. **의료 영상 도메인**: 전혀 다른 특성의 이미지
2. **자동차 카메라**: 다른 통계적 특성
3. **위성 영상**: 또 다른 도메인 특성

**해결책:**
- 도메인 특정 데이터셋에 대한 효율적 세밀조정 전략
- 도메인 간 특성 전이의 이론적 이해

#### 9.3.2 약한 감독 학습(Weakly Supervised Learning)

제한된 라벨이 있는 상황에서의 신뢰도 구간 구성:
- 노이즈 있는 라벨
- 불완전한 라벨
- 약한 라벨(weak labels)

### 9.4 이론적 기초 강화

#### 9.4.1 확산 모델의 일반화 이론

**최근 발전** (2025):[14][7]
- 확산 모델의 로컬 귀납 편향(local inductive bias) 분석
- 일반화 메커니즘의 기계적 설명 제시
- 사전학습과 미세조정의 상호작용 이론화

**필요 연구:**
이러한 이론적 통찰을 신뢰도 구간 예측에 적용하여:
- 언제 타이트한 구간이 가능한지 예측
- 필요한 데이터량 추정
- 전이 학습 가능성 평가

#### 9.4.2 합성곱 기반 보장의 정교화

합성곱 예측의 분포 자유 보장을 다음으로 확장:
- **조건부 커버리지**: 입력 특성에 따른 조건부 보장
- **로컬 보장**: 공간적으로 변하는 보장
- **시간적 보장**: 비디오나 시계열 데이터

### 9.5 실무 배포 고려사항

#### 9.5.1 계산 효율성

**현재 성능:**
- 단일 순전파: ~100ms
- 실시간 응용: 초당 10fps 이상 필요

**개선 방향:**
- 모델 압축(quantization, pruning)
- 에지 디바이스 배포 최적화
- 배치 처리 최적화

#### 9.5.2 규제 준수

**의료 응용의 경우:**
- FDA 510(k) 규제 요구사항
- 임상 검증 프로토콜
- 안전성 인증

**필요 연구:**
신뢰도 구간이 규제 기관의 요구를 충족하는지 확인:
- 통계적 보장의 실제 적용성
- 임상 의사결정에의 영향 평가
- 거짓 긍정/음성률 관리

#### 9.5.3 사용자 인터페이스 설계

신뢰도 구간을 최종 사용자(의료인, 엔지니어)에게 효과적으로 전달:
- **시각화**: 히트맵, 오버레이 등
- **요약 통계**: 영역별 신뢰도
- **의사결정 지원**: 불확실성 높은 영역 자동 플래그

***

## 10. 결론

**Conffusion** 논문은 확산 모델에 통계적 보장을 제공하는 혁신적 접근법을 제시합니다. 주요 성과는:

1. **이론적 기여**: Risk-Controlling Prediction Set 프레임워크의 확산 모델 적용
2. **실무적 기여**: 1000배 속도 향상으로 실용성 확보
3. **일반화 성과**: 작업/데이터 무관한 G-Conffusion으로 전이 학습의 가능성 제시

논문이 직면한 한계와 미래 연구 방향은 다음으로 요약됩니다:

| 영역 | 현재 상태 | 개선 필요 사항 |
|------|---------|-------------|
| **분포 모델링** | 일봉형 분포 최적화 | 다중모달 분포 처리 |
| **공간 적응성** | 전역 캘리브레이션 | 픽셀/영역별 적응 |
| **응용 범위** | 저수준 작업 | 고수준 의미론적 개념 |
| **도메인 일반화** | ImageNet 기반 | 크로스 도메인 효율성 |
| **이론적 근거** | 경험적 성공 | 일반화 메커니즘 규명 |

이 연구는 생성 모델의 신뢰성을 확보하는 길을 열어, **의료, 자율주행, 과학 연구 등 높은 신뢰도가 요구되는 분야에서 AI 활용을 가능**하게 할 것으로 예상됩니다.

***

## 참고 자료

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/eaef662d-44cd-4780-9542-ce1cac1b2d4c/2211.09795v1.pdf)
[2](https://link.springer.com/10.1007/s10278-024-01286-5)
[3](https://arxiv.org/pdf/2309.04760.pdf)
[4](https://proceedings.neurips.cc/paper_files/paper/2024/file/c693c3ff83259aebcd55a41ab19a5d84-Paper-Conference.pdf)
[5](https://neurips.cc/virtual/2025/poster/121463)
[6](https://papers.nips.cc/paper_files/paper/2024/file/f782860c2a5d8f675b0066522b8c2cf2-Paper-Conference.pdf)
[7](https://arxiv.org/html/2411.19339v2)
[8](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05806.pdf)
[9](https://arxiv.org/abs/2405.18527)
[10](https://arxiv.org/abs/2411.04376)
[11](https://arxiv.org/pdf/2503.05618.pdf)
[12](https://arxiv.org/html/2504.07696v1)
[13](https://arxiv.org/html/2511.10710v1)
[14](https://arxiv.org/pdf/2311.01797.pdf)
[15](https://arxiv.org/abs/2211.09795)
[16](https://link.springer.com/10.1007/978-3-031-13448-7_11)
[17](https://ieeexplore.ieee.org/document/9776189/)
[18](https://www.mdpi.com/2072-6643/14/19/4128)
[19](https://academic.oup.com/jbmr/article/38/1/198-213/7500033)
[20](https://dx.plos.org/10.1371/journal.pone.0279499)
[21](https://ieeexplore.ieee.org/document/9875295/)
[22](http://scik.org/index.php/cmbn/article/view/6900)
[23](https://www.frontiersin.org/articles/10.3389/fphar.2022.937029/full)
[24](https://bmcmedimaging.biomedcentral.com/articles/10.1186/s12880-022-00864-9)
[25](https://arxiv.org/pdf/2410.13738.pdf)
[26](https://arxiv.org/pdf/2305.00624.pdf)
[27](https://arxiv.org/pdf/2209.11215.pdf)
[28](https://arxiv.org/pdf/2209.00796v8.pdf)
[29](https://arxiv.org/pdf/2211.01324.pdf)
[30](http://arxiv.org/pdf/2406.16213.pdf)
[31](http://arxiv.org/pdf/2410.11081.pdf)
[32](http://arxiv.org/pdf/2404.13309.pdf)
[33](https://www.semanticscholar.org/paper/Conffusion:-Confidence-Intervals-for-Diffusion-Horwitz-Hoshen/46f64799316a76bd7023e0c047fdffd96b62117d)
[34](https://arxiv.org/html/2501.18897v2)
[35](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/conffusion/)
[36](https://www.ijcai.org/proceedings/2025/1095.pdf)
[37](https://aclanthology.org/D18-1487/)
[38](https://eusipco2025.org/wp-content/uploads/pdfs/0001877.pdf)
[39](https://arxiv.org/html/2509.09438v1)
[40](https://scholar.google.com/citations?user=NyLx5nIAAAAJ&hl=en)
[41](https://www.mdpi.com/2079-9292/12/24/5027)
[42](https://www.semanticscholar.org/paper/945a899a93c03eb63be5e3197e318c077473cef9)
[43](https://ojs.aaai.org/index.php/AAAI/article/view/28199)
[44](https://www.semanticscholar.org/paper/ddd8734267db682f7925516a59de4785b424f622)
[45](https://arxiv.org/abs/2409.18168)
[46](https://arxiv.org/abs/2304.04774)
[47](https://arxiv.org/abs/2412.08240)
[48](http://www.emerald.com/ilt/article/77/2/211-218/1239749)
[49](https://arxiv.org/abs/2409.04060)
[50](https://ieeexplore.ieee.org/document/10864650/)
[51](https://arxiv.org/pdf/2305.18455.pdf)
[52](https://arxiv.org/abs/2308.11948)
[53](http://arxiv.org/pdf/2503.06698.pdf)
[54](https://arxiv.org/html/2412.00665v1)
[55](https://aclanthology.org/2023.acl-long.248.pdf)
[56](https://arxiv.org/html/2411.16725v1)
[57](https://openaccess.thecvf.com/content/CVPR2024W/SAIAD/papers/Mossina_Conformal_Semantic_Image_Segmentation_Post-hoc_Quantification_of_Predictive_Uncertainty_CVPRW_2024_paper.pdf)
[58](https://www.cs.ox.ac.uk/teaching/courses/2024-2025/UDL/)
[59](https://papers.miccai.org/miccai-2025/paper/3902_paper.pdf)
[60](https://arxiv.org/html/2405.16876v2)
[61](https://www.sciencedirect.com/science/article/abs/pii/S0951832024005854)
[62](https://iclr.cc/virtual/2023/14351)
[63](https://neurips.cc/virtual/2024/poster/96508)
[64](https://arxiv.org/abs/2405.08886)
[65](https://ieeexplore.ieee.org/document/10782378/)
[66](https://arxiv.org/abs/2405.15912)
[67](https://ieeexplore.ieee.org/document/10678202/)
[68](https://arxiv.org/abs/2407.00499)
[69](https://ebooks.iospress.nl/doi/10.3233/SHTI231113)
[70](https://link.springer.com/10.1007/978-3-031-73158-7_19)
[71](https://arxiv.org/pdf/2408.05037.pdf)
[72](http://arxiv.org/pdf/2107.07511.pdf)
[73](https://arxiv.org/html/2411.04376v1)
[74](https://arxiv.org/abs/2207.02238)
[75](https://arxiv.org/html/2503.04191v1)
[76](https://pubmed.ncbi.nlm.nih.gov/39613981/)
[77](https://academic.oup.com/nsr/article/11/12/nwae348/7810289)
[78](https://papers.miccai.org/miccai-2024/529-Paper1623.html)
[79](https://www.archivinci.com/blogs/diffusion-models-guide)
[80](https://www.semanticscholar.org/paper/Estimating-Epistemic-and-Aleatoric-Uncertainty-with-Chan-Molina/721d2080099677b77b2130488a637f44db35025e)
[81](https://unsuremiccai.github.io/prev_years/2024/)
[82](https://trustworthyai.co.kr/publication/)
[83](https://arxiv.org/abs/2402.03478)
[84](https://www.sciencedirect.com/science/article/pii/S0169260724002268)
