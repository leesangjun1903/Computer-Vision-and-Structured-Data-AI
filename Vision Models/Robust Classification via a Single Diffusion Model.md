# Robust Classification via a Single Diffusion Model

***

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 논문의 핵심 주장

본 논문은 **단일 사전학습 확산 모델로부터 생성 분류기(generative classifier)를 구성하여 높은 적대적 견고성을 달성할 수 있다**고 주장합니다. 

기존의 판별 분류기(discriminative classifier) 기반 방어 방법들이 다음과 같은 한계를 가지고 있다고 지적합니다:

- **확산 기반 정제**: DiffPure 같은 방법들은 강한 적응 공격에 취약하며, 매우 높은 무작위성으로 인해 불안정함
- **적대적 훈련**: 훈련된 특정 공격에만 견고하고 미지의 위협에 대해 급격한 성능 저하를 보임
- **판별 학습의 근본적 한계**: 훈련 분포 외의 입력에 대해 정확한 예측이 불가능

따라서 **확산 모델이 전체 자료 공간에서 정확한 점수 함수(score function)를 추정하는 능력**을 직접 분류에 활용하면 이러한 문제들을 근본적으로 해결할 수 있다는 것이 핵심 주장입니다.

### 1.2 주요 기여

**이론적 기여:**
1. **Theorem 3.1**: 확산 모델을 생성 분류기로 변환하는 방법 제시 - Bayes 정리를 통해 조건부 확률을 확산 손실로 표현
2. **Theorem 3.2 & Corollary 3.3**: 최적 확산 모델의 명시적 형태 도출 및 최적 분류기의 성질 분석
3. **Robustness Analysis**: 최적 조건에서 100% 견고성 달성 가능성을 수학적으로 증명

**방법론적 기여:**
1. **RDC (Robust Diffusion Classifier)**: 두 단계 접근법
   - Likelihood Maximization: 입력을 높은 우도 영역으로 이동
   - Diffusion Classification: 이동된 입력 분류

2. **Multi-head Diffusion**: UNet 아키텍처 수정을 통한 계산 복잡도 감소
   - 기존: $$O(K \times T)$$ → 개선: $$O(T)$$

3. **효율적 샘플링 전략**: 몬테카를로 샘플링의 분산 감소 기법

**경험적 기여:**
1. CIFAR-10에서 75.67% 견고성 달성 (기존 최고 기록 AT-EDM 70.90% 대비 +4.77%)
2. 미지의 위협에 대한 우수한 일반화: StAdv에서 89.45% (기존 방법 대비 +53.90%)
3. Gradient obfuscation이 없음을 실증적으로 입증

***

## 2. 해결하고자 하는 문제 및 제안 방법

### 2.1 문제 정의

**문제 1: 확산 기반 정제 방법의 한계**

DiffPure는 다음과 같은 방식으로 작동합니다:
- 입력 $x$ → 정방향 확산 (노이즈 추가) → 역방향 복원 (깨끗한 이미지 시뮬레이션)

그러나 문제점:
- 역방향 과정에서 적대적 섭동이 완전히 제거되지 않음
- 강한 적응 공격으로 견고성이 71.29%에서 44.53%으로 급락
- 높은 무작위성으로 인한 평가의 불안정성

**문제 2: 적대적 훈련의 한계**

최고 성능의 AT-EDM도 다음의 문제:
- ℓ∞에서는 70.90% 견고성
- StAdv 같은 미지의 위협에서는 35.97% (RDC 대비 53.48% 낮음)
- 특정 공격에 과적합되어 다른 공격에 무방비

**근본 원인:**
판별 분류기는 훈련 분포 내에서만 정확한 확률을 추정할 수 있으며, 적대적 섭동으로 인해 분포 외의 영역에 떨어진 입력에 대해서는 신뢰할 수 없는 예측을 함.

### 2.2 제안하는 방법론

#### **2.2.1 기본 원리: 확산 모델을 분류기로 변환**

Bayes 정리에 기반한 생성 분류 접근:

$$p_\theta(y|x) = \frac{p_\theta(x|y)p(y)}{\sum_{\hat{y}} p_\theta(x|\hat{y})p(\hat{y})}$$

핵심: 확산 모델로부터 조건부 우도 $p_\theta(x|y)$ 추정

#### **2.2.2 조건부 우도의 변분 근사**

**Theorem 3.1** (조건부 우도 근사):

조건부 확산 손실의 변분 하한:

$$\log p_\theta(x|y) \geq -E_{\epsilon,t}\left[w_t||\epsilon_\theta(x_t, t, y) - \epsilon||_2^2\right] + C$$

여기서:
- $x_t = \sqrt{\alpha_t}x + \sigma_t\epsilon$ (정방향 과정)
- $\epsilon_\theta(x_t, t, y)$: 노이즈 예측 네트워크 (클래스 조건부)
- $w_t$: 타임스텝별 가중치

따라서 조건부 확률은:

$$p_\theta(y|x) = \text{softmax}\left(\log p(y) - E_{\epsilon,t}[w_t||\epsilon_\theta(x_t, t, y) - \epsilon||_2^2]\right)$$

#### **2.2.3 최적 확산 분류기의 성질**

**Theorem 3.2** (최적 확산 모델):

최적화된 확산 모델의 노이즈 예측은:

$$\epsilon_{\theta^*_D}(x_t, t, y) = \sum_{x^{(i)} \in D_y} \frac{1}{\sigma_t}s(x_t, x^{(i)}) \cdot (x_t - \sqrt{\alpha_t}x^{(i)})$$

여기서 $s(x_t, x^{(i)})$는 $x_t$가 $x^{(i)}$로부터 생성되었을 확률.

**Corollary 3.3** (최적 분류기의 로짓):

$$f_{\theta^*_D}(x)_y = -E_{\epsilon,t}\left[\frac{\alpha_t}{\sigma_t^2}\left|\left|\sum_{x^{(i)} \in D_y} s(x, x^{(i)}, \epsilon, t) \cdot (x - x^{(i)})\right|\right|_2^2\right]$$

**직관적 해석:**
- 입력 $x$와 클래스 $y$의 훈련 샘플들 간의 가중 거리 기반
- 가중치 $\frac{\alpha_t}{\sigma_t^2}$는 깨끗한 데이터에 높은 가중치, 노이즈 많은 데이터에 낮은 가중치
- **이 최적 분류기는 100% 견고성 달성 가능**

#### **2.2.4 우도 최대화 (Likelihood Maximization) 기법**

**문제:** 실제 훈련된 확산 모델은 최적이 아니므로:
- 부정확한 밀도 추정 $p_\theta(x|y)$
- 우도와 확산 손실 간의 간격: $d(x,y,\theta) = \log p_\theta(x|y) + E_{\epsilon,t}[w_t||\epsilon_\theta(x_t, t, y) - \epsilon||_2^2]$

**해결책:** 비조건부 확산 손실 최소화:

$$\min_{\hat{x}} E_{\epsilon,t}[w_t||\epsilon_\theta(\hat{x}_t, t) - \epsilon||_2^2], \quad s.t. \quad ||\hat{x} - x||_\infty \leq \eta$$

**의미:**
1. 입력을 높은 우도 영역으로 이동
2. $\eta$ 제약을 통해 다른 클래스로 넘어가는 것 방지
3. 경사 기반 최적화로 N 스텝 수행

**수식:**

**Algorithm 1: Robust Diffusion Classifier (RDC)**

```
입력: 사전학습 확산 모델 ε_θ, 입력 x, 최적화 예산 η, 
      단계크기 γ, 최적화 단계 N, 모멘텀 감쇠 μ

1. m ← 0, x̂ ← x

2. for n = 0 to N-1:
     g ← ∇_x E_ε,t[w_t ||ε_θ(x̂_t, t) - ε||²₂]  (단일 ε,t 샘플)
     m ← μ·m - g/||g||₁
     x̂ ← clip_{x,η}(x̂ + γ·m)

3. E_ε,t[w_t ||ε_θ(x̂_t, t, y) - ε||²₂] 모든 y에 대해 계산 (multi-head)

4. p_θ(y|x̂) = softmax(-E_ε,t[...])  (Eq. 6)

5. ỹ ← argmax_y p_θ(y|x̂)
```

#### **2.2.5 계산 효율화: Multi-head Diffusion**

**표준 방법의 문제:**
- 각 클래스 y마다 독립적인 계산 필요
- 복잡도: $O(K \times T)$ (K=클래스 수, T=타임스텝)

**해결책:**
UNet의 마지막 합성곱 층을 수정하여 모든 클래스의 노이즈 예측을 동시 출력:

```
표준 UNet:  ... → [Conv] → ε_θ(x_t, t, y=1) 또는 y=2 또는 ... y=K
Modified:   ... → [Multi-head Conv] → [ε_θ(x_t, t, y=1), ..., ε_θ(x_t, t, y=K)]
                                       (모두 한 번에)
```

결과: 복잡도 감소 $O(K \times T) \to O(T)$

***

## 3. 모델 구조 및 이론적 분석

### 3.1 두 단계 구조

**Stage 1: Likelihood Maximization**
- 목표: 입력을 높은 우도 영역으로 이동
- 작동: 무조건 확산 손실 최소화
- 부작용: Randomness 유도 (견고성 향상에 기여)

**Stage 2: Diffusion Classification**
- 목표: 최적화된 입력 분류
- 작동: 조건부 확산 손실 이용 Bayes 정리 적용
- 특징: 결정적(deterministic, Randomness 없음)

### 3.2 견고성의 이론적 근거

**최적 조건에서의 절대 견고성:**

최적 확산 모델 $\epsilon_{\theta^*_D}$를 사용하면, Corollary 3.3의 분류기는:

$$p(Y = y | X = x^* \text{ where } ||x^* - x||_\infty \leq \epsilon)$$

를 정확하게 계산하므로, 자료 분포 내에서는 완벽한 견고성 달성 가능.

**실제 성능과의 갭:**

실제 모델에서:
- DC만으로는 35.94% 견고성 (이론 100% 대비)
- RDC (LM+DC)로 75.67% 견고성

갭의 원인:
1. $d(x,y,\theta) \not\to 0$ (간격 존재)
2. 조건부 우도 추정의 부정확성

### 3.3 일반화의 이론적 기제

**특정 공격에 비특화:**
- DiffPure, AT 등: 특정 공격 유형에 최적화
- RDC: 자료 분포 자체에만 의존

**자료 분포의 견고성:**
- 적대적 섭동은 자료 분포 외부의 영역
- 높은 우도 영역의 입력은 본질적으로 덜 취약
- 따라서 다양한 공격 방향에 견고

**Wasserstein 거리 최소화:**
우도 최대화 = 자료 분포에 더 가까운 점으로 이동
이는 공격 방향과 무관하게 일어남

***

## 4. 성능 향상 및 한계

### 4.1 정량적 성능 분석

#### **주요 성능 지표 (CIFAR-10)**

| 평가 항목 | 값 | 기존 SOTA | 개선 |
|----------|-----|-----------|------|
| **ℓ∞ 견고성 (ε=8/255)** | 75.67% | 70.90% (AT-EDM) | +4.77% |
| **ℓ2 견고성 (ε=0.5)** | 82.03% | 84.77% (AT-EDM) | -2.74% |
| **평균 견고성** | 82.38% | 77.85% | +4.53% |
| **StAdv 견고성 (ε=0.05)** | 89.45% | 35.97% | **+53.48%** |
| **깨끗함 정확도** | 89.85% | 93.36% | -3.51% |

**해석:**
- ℓ∞: 최고 성능 달성 (이전 기록 갱신)
- ℓ2: AT-EDM에 미흡 (최적화 예산 η=8/255의 영향)
- StAdv: 획기적으로 높은 성능 (미지 위협에 강함)

#### **미지 위협에 대한 일반화**

| 훈련 위협 | 테스트 위협 | RDC 성능 | AT-ℓ∞ 성능 | 차이 |
|-----------|-----------|---------|-----------|------|
| ℓ∞ | ℓ2 | 82.03% | 49.41% | **+32.62%** |
| ℓ2 | ℓ∞ | 75.67% | 63.28% | **+12.39%** |
| 둘 다 | StAdv | 89.45% | 35.97% | **+53.48%** |

**의의:**
- RDC는 훈련되지 않은 위협에 대해 급격한 성능 저하 없음
- 판별 훈련의 근본적 한계를 극복

### 4.2 Gradient Obfuscation 분석

**문제:** 높은 무작위성이 있는 방어는 gradient obfuscation으로 인해 겉으로만 견고해 보일 수 있음

**RDC의 검증:**

**그림 2(a): Gradient 무작위성 측정**

Cosine Similarity 측정 (10회 반복):
- DiffPure: ~0.001 (매우 무작위적)
- DC: ~0.98 (거의 결정적)
- RDC: ~0.98 (거의 결정적)

→ RDC의 견고성은 무작위성 때문이 아님

**표 2: 적응 공격 비교**

| 공격 방식 | N=5 | N=1 |
|---------|-----|-----|
| BPDA | 75.67% | 69.92% |
| 정확 경사 | - | 69.53% |
| Lagrange | 77.54% | - |

→ BPDA ≈ 정확 경사 → BPDA로 신뢰성 있는 평가

### 4.3 주요 한계

#### **한계 1: 간격 문제**

Theorem 3.1의 가정:
$$d(x,y,\theta) = \log p_\theta(x|y) + E_{\epsilon,t}[w_t||\epsilon_\theta(x_t, t, y) - \epsilon||_2^2] \to 0$$

실제 상황:
- 이 간격이 존재하고 상당할 수 있음
- DC만으로는 35.94% 견고성에 그침
- 간격의 상한을 구하는 것도 미해결

#### **한계 2: Multi-head Diffusion의 성능 저하**

문제점:
- 표준 UNet 정확도: 95%+ 
- Multi-head 정확도: 60% (초기 훈련 단계)
- 특성 추출의 혼동 (모든 클래스 동시 예측)

원인 분석:
- 표준: 각 클래스별로 특화된 특성 추출
- Multi-head: 모든 클래스에 적합한 특성 추출 필요 (불가능)

**피처 분석:**
- 표준: 다른 클래스 y에 대해 cosine similarity ~0.5 (다름)
- Multi-head: cosine similarity ~0.98 (거의 같음)

**훈련 손실 변경 시 문제:**
- 교차 엔트로피 손실 사용 시 견고성 0% (완전 실패)

#### **한계 3: 계산 복잡도**

문제:
- 모든 T개 타임스텝 계산 필수
- 일반적으로 T=1000 (설정에 따라 500~4000)
- 실시간 시스템 적용 어려움

표 3의 시간 복잡도 비교:
- RDC: ~30초/이미지 (512개 샘플, 공유 GPU)
- AT-EDM: ~5초/이미지
- DiffPure: ~50초/이미지

#### **한계 4: 평가 범위의 제한**

- CIFAR-10 512개 샘플만으로 주 평가 (AutoAttack 비용)
- ImageNet 등 대규모 데이터셋 미평가
- 다른 모달리티(텍스트, 음성) 미검증

#### **한계 5: ℓ2 노름에서의 부족**

- ℓ2 견고성 82.03% < AT-EDM 84.77%
- 원인: η=8/255는 ℓ∞ 정규화로 ℓ2에는 과도할 수 있음

***

## 5. 모델의 일반화 성능 향상 가능성

### 5.1 일반화 메커니즘 분석

#### **1) 자료 분포 기반의 비특화**

**특정 공격에 최적화된 방법:**
- AT: PGD 공격에 최적화 → ℓ∞에는 강하나 ℓ2에는 약함
- DiffPure: 정제 과정에 최적화 → StAdv 같은 의미론적 공격에 약함

**RDC의 특성:**
- 자료 분포 자체에만 의존
- 공격 방향과 무관하게 일관된 응답
- 따라서 미지의 공격에도 일반화

#### **2) Wasserstein 거리 최소화**

우도 최대화 과정:
$$\min_{\hat{x}} ||x̂ - x||_\infty + \text{높은 우도 추구}$$

이는 입력을 자료 분포에 더 가깝게 이동:
- P_data(x)에 더 가까운 점 = 더 견고함
- 공격의 방향(ℓ∞, ℓ2, 의미론적)과 무관

#### **3) 스코어 함수의 안정성**

전체 자료 공간에서 학습한 스코어 함수:
$$\nabla_x \log p(x)$$

특성:
- 국소적 적대적 섭동에 덜 민감
- 전역적 자료 분포 구조를 반영

### 5.2 정량적 일반화 분석

#### **거리 전환 성능 (Cross-norm Transfer)**

| 훈련 설정 | 테스트 위협 | 성능 | 기존 방법 | 차이 |
|----------|-----------|------|---------|------|
| AT-ℓ∞ 훈련 | ℓ2 테스트 | 82.03% | 49.41% | **+32.62%** |
| AT-ℓ2 훈련 | ℓ∞ 테스트 | 75.67% | 63.28% | **+12.39%** |

**해석:**
- 특정 노름에 최적화되지 않음 (or 모든 노름에 균형적)
- 미지 노름 공격에 대해 예측 불가능한 우수 성능

#### **의미론적 공격 (Semantic Attack)**

StAdv (Spatially Transformed Adversarial) - 공간 변환 기반:

| 방법 | StAdv 성능 | 감소율 |
|------|-----------|--------|
| AT-ℓ∞ | 35.97% | -49% (vs clean) |
| AT-ℓ2 | 52.45% | -45% (vs clean) |
| RDC | **89.45%** | -0.4% (vs clean) |

→ 미지의 공격 유형에 획기적인 일반화

### 5.3 일반화 성능 향상의 이론적 근거

#### **Theorem**: 자료 분포의 견고성

입력 $x$가 자료 분포 $P_{\text{data}}$에 가까울수록:
- 다양한 공격에 견고한 경향
- 이는 정보 이론적으로 설명 가능

**직관:**
$$P(Y=y|X=x) \text{는 } D(x \parallel P_{\text{data}}) \text{에 대해 robust}$$

RDC가 $x$를 $P_{\text{data}}$에 가깝게 이동시키므로, 거리 노름이나 공격 방향과 무관하게 일반적 견고성 달성 가능.

### 5.4 향후 일반화 개선 방향

#### **1) 적응적 최적화 예산**

현재: 고정된 η = 8/255

개선:

$$\eta(x) = \text{argmin}_\eta \left( ||x̂ - x||_\infty + E_y[-\log p_\theta(y|x̂)] \right)$$

기대 효과: +1~2% 성능 향상

#### **2) 다중 확산 모델 앙상블**

현재: 단일 사전학습 확산 모델

개선:
$$p_{\text{ensemble}}(y|x) = \frac{1}{M} \sum_{i=1}^{M} p_{\theta_i}(y|x)$$

기대 효과: 일반화 추가 향상, 다양한 공격에 더욱 견고

#### **3) 조건부 확산 모델의 정밀한 훈련**

현재: 표준 확산 모델 사용

개선: 적대적 견고성을 목표로 한 조건부 확산 모델 재훈련

기대 효과: 간격 $d(x,y,\theta)$ 감소, 이론적 성능에 더 가까워짐

***

## 6. 2020년 이후 최신 연구와의 비교 분석

### 6.1 확산 모델 기반 적대적 방어의 발전 계통

```
2020-2021: 기초
    ↓ Diffusion Models 기본 이론
2022: 초기 적용
    ↓ DiffPure 제안 (71.29% 견고성, 강한 공격에 취약)
2023: RDC (본 논문) - 패러다임 전환
    ├─ 생성 분류 접근 (75.67%)
    ├─ 이론적 기초 수립 (최적성 분석)
    └─ 미지 위협 일반화 (+53%)
2023: 병행 발전
    ├─ DensePure (정제 기반 개선, ~72%)
    ├─ SBGC, HybViT (생성 분류 시도, 실패)
    └─ Chen et al. 후속작 (인증 견고성)
2024: 다각화
    ├─ CausalDiff (인과추론, 86.39%)
    ├─ ADBM (정제 재평가, 75.27%)
    ├─ DIFFender (패치 공격, 77%)
    └─ RCDM (제어 이론 적용)
2025: 이론적 심화
    ├─ Chen et al. (인증 견고성 증명)
    └─ "How Do DMs Improve Robustness?" (압축 메커니즘 분석)
```

### 6.2 주요 방법론과의 직접 비교

#### **표: 방어 방법의 성능 및 특징 비교**

| 방법 | 년도 | 기본 원리 | CIFAR-10 (ℓ∞) | 미지위협 | 훈련필요 | 이론적 근거 |
|------|------|---------|--------------|---------|---------|-----------|
| **DiffPure** | 2022 | 확산 기반 정제 | 44.53%* | 낮음 | 없음 | 직관적 |
| **AT-EDM** | 2023 | 생성 데이터 + 적대훈련 | 70.90% | 매우낮음 | 필수 | 경험적 |
| **RDC** | 2023 | 생성 분류기 | **75.67%** | **높음** | 없음 | **정리 증명** |
| **DensePure** | 2023 | DiffPure 개선 | ~72% | 중간 | 없음 | 우도-거리 분석 |
| **CausalDiff** | 2024 | 인과적 요인 분리 | **86.39%** | **매우높음** | 필수 | 인과 추론 |
| **ADBM** | 2024 | 최적 브릿지 구성 | 75.27% | 중간 | 없음 | 이론적 |

*: 적응 공격 적용 시 71.29% → 44.53%으로 급락

#### **DiffPure vs RDC 상세 비교**

| 항목 | DiffPure | RDC |
|------|----------|-----|
| **기본 메커니즘** | 정방향 확산 + 역방향 복원 | 생성 분류 (Bayes) |
| **작동 방식** | 정제만 (별도 분류기 필요) | 분류와 정제 통합 |
| **이론적 기초** | 직관적 | 엄밀한 정리 |
| **적응 공격 안정성** | 취약 (71.29% → 44.53%) | 견고 (75.67% 유지) |
| **gradient 무작위성** | 매우 높음 (~0.001) | 낮음 (~0.98) |
| **미지 위협 일반화** | 낮음 | **높음 (+53%)** |
| **사전학습 필요** | 표준 확산 모델 | 표준 확산 모델 |
| **계산 비용** | 높음 (~50초) | 높음 (~30초) |
| **구현 난이도** | 낮음 | 중간 |

#### **AT (Adversarial Training) vs RDC**

| 항목 | AT-EDM | RDC |
|------|--------|-----|
| **학습 패러다임** | 판별 학습 | 생성 학습 |
| **훈련 데이터** | 적대적 샘플 생성 필수 | 사전학습 모델만 필요 |
| **ℓ∞ 성능** | 70.90% | 75.67% (+4.77%) |
| **미지 위협 성능** | 35.97% (StAdv) | 89.45% (+53.48%) |
| **계산 비용** | 훈련 매우 높음 (100M 이미지) | 훈련 불필요 |
| **깨끗함 정확도** | 93.36% | 89.85% (-3.51%) |
| **특정 공격 특화** | 매우 높음 | 없음 |

### 6.3 최신 연구의 관점에서 RDC의 평가

#### **CausalDiff (2024)와의 관계**

**CausalDiff의 혁신:**
- 입력을 원인적 요인(causal factors)과 비원인적 요인으로 분리
- 적대적 섭동은 비원인적 요인으로 간주 후 제거
- 성능: 86.39% (RDC 대비 +10.72%)

**RDC와의 관계:**
- 보완적 접근: RDC는 분포 기반, CausalDiff는 인과 추론 기반
- CausalDiff는 RDC와 달리 조건부 확산 모델 훈련 필수
- 두 방법의 조합 가능성 존재

#### **ADBM (2024)과의 관계**

**ADBM의 발견:**
- DiffPure의 견고성이 과대평가되었음을 밝힘
- 71.29% → 44.53% 급락 현상 재확인
- 더 강한 적응 공격으로 평가

**RDC와의 비교:**
- RDC는 ADBM과 유사 성능 (75.67% vs 75.27%)
- 하지만 RDC는 이미 강한 적응 공격 포함하여 평가
- 평가의 신뢰성 면에서 RDC 우수

### 6.4 연구 커뮤니티에 미친 영향

#### **1) 이론적 영향**

- **생성 분류 패러다임의 정당화**: Theorem 3.1, 3.2로 엄밀한 기초 제공
- **최적성 분석**: 최적 조건에서 100% 견고성 가능성 증명
- **후속 인증 견고성 연구** (Chen et al. 2024): RDC의 인증 견고성 증명

#### **2) 방법론적 영향**

- **Multi-head 아키텍처**: 다른 연구들에서 효율성 개선 시도
- **Likelihood Maximization**: 다른 정제 방법에도 적용 (DensePure, CausalDiff)
- **두 단계 접근법**: 이후 DIFFender, RCDM 등에서 채용

#### **3) 평가 방법론의 영향**

- **Gradient Obfuscation 엄격한 검증**: 낮은 무작위성 측정 개발
- **BPDA 적응 공격의 신뢰성**: 정확 경사와의 비교로 검증
- **미지 위협 평가 확대**: StAdv 등 다양한 공격으로 일반화 평가

#### **4) 최신 이론 분석에의 기여**

- **2025년 "How Do DMs Improve Robustness?"**: RDC의 낮은 무작위성 발견이 압축 메커니즘 분석에 영감
- **확산 모델의 견고성 근본 원인**: 정제 효과가 아닌 분포 근접성으로 해석

***

## 7. 향후 연구 시 고려할 점

### 7.1 이론적 개선 방향

#### **1) 간격 분석의 정교화**

**현재 미해결 문제:**
$$d(x,y,\theta) = \log p_\theta(x|y) + E_{\epsilon,t}[w_t||\epsilon_\theta(x_t, t, y) - \epsilon||_2^2]$$

이 간격의 상한과 하한을 구하는 것이 중요:
- 상한이 작을수록: 이론적 성능에 더 접근 가능
- 하한이 크면: 실제 성능의 한계 예측 가능

**접근 방법:**
- 조건부 점수 함수의 추정 오류 분석
- PAC 학습 이론 적용
- 정규성(regularity) 조건 도입

#### **2) 조건부 우도 추정의 개선**

**현재:**
변분 하한 (ELBO):
$$\log p_\theta(x|y) \geq -E_{\epsilon,t}[w_t||\epsilon_\theta(x_t, t, y) - \epsilon||_2^2] + C$$

**개선 방향:**
- 더 타이트한 하한 개발
- 식별 함수(tightness function) 도입
- 적응적 가중치 $w_t$ 최적화

#### **3) 다변량 수렴성 분석**

현재 누락된 영역:
- N (최적화 단계 수)에 대한 수렴
- T (타임스텝 수)에 대한 수렴
- 클래스 수 K에 대한 확장성

**필요한 분석:**
- 표본 복잡도 (sample complexity)
- 최적화 복잡도 (optimization complexity)

### 7.2 방법론적 개선

#### **1) Multi-head Diffusion 재구조화**

**현재 문제:**
- 모든 클래스에 적합한 특성 추출 불가능
- 훈련 초기 정확도 60% → 최종 93%+로 회복하지만 견고성은 여전히 문제

**해결 방안:**
- **선택적 Head Architecture**: 클래스별로 다른 특성 경로
- **Attention Mechanism 추가**: 클래스별로 다른 가중치 적용
- **Task-Specific Fine-tuning**: Multi-head 이후 추가 조정

```
개선 아이디어:
[공유 특성 추출] → [클래스별 Attention] → [클래스별 노이즈 예측]
                        ↑
                    가중치 학습
```

#### **2) 적응적 최적화 예산**

**현재:**
고정된 η = 8/255

**개선:**
동적 η(x) 선택:

$$\eta(x) = \arg\min_\eta \left( \lambda \cdot ||x̂(η) - x||_\infty + (1-\lambda) \cdot \left( -\min_y \log p_\theta(y|x̂(η)) \right) \right)$$

**장점:**
- 입력별 최적 이동 거리 자동 결정
- 깨끗함 정확도와 견고성의 균형 자동 조정

#### **3) 다중 확산 모델 앙상블**

**현재:**
단일 사전학습 확산 모델

**개선:**
$$p_{\text{ensemble}}(y|x) = \frac{1}{M} \sum_{i=1}^M \alpha_i p_{\theta_i}(y|x)$$

**방법:**
- 다양한 사전학습 확산 모델 수집
- 또는 다양한 하이퍼파라미터로 재훈련
- 가중치 $\alpha_i$ 최적화

**기대 효과:**
- 일반화 성능 추가 향상
- 단일 모델의 편향 제거

### 7.3 확장 및 적용 연구

#### **1) 대규모 데이터셋으로의 확장**

**현재 한계:**
- CIFAR-10 512개 샘플만 평가
- ImageNet 등 대규모 데이터셋 미평가

**필요한 연구:**
- ImageNet에서의 성능 측정
- 계산 효율화 (샘플링 최소화)
- 메모리 최적화

**예상 도전과제:**
- 더 높은 해상도에서의 확산 모델 정확도
- 1000개 클래스의 multi-head 확장성

#### **2) 다양한 모달리티 적용**

**미지 영역:**
- **텍스트**: 텍스트 확산 모델의 견고한 분류 활용
- **음성**: 음성 확산 모델 기반 적대적 견고성
- **3D**: 포인트 클라우드 확산 모델

**기술적 도전:**
- 모달리티별 적대적 공격의 정의
- 확산 모델의 조건부 구성

#### **3) 실시간 성능 개선**

**현재:**
~30초/이미지 (512개 샘플 기준)

**개선 방법:**
- **DDIM 가속**: 타임스텝 감소 (1000 → 50)
- **분할 점수 행렬**: GPU 메모리 효율화
- **주의 메커니즘 프루닝**: 불필요한 계산 제거

**목표:**
실시간 응용 가능한 <1초/이미지

### 7.4 이론-실제 갭 해결

#### **1) 최적 모델에 더 가까운 확산 모델 훈련**

**아이디어:**
Theorem 3.2의 최적 모델에 가까워지도록 확산 모델을 직접 최적화:

$$\min_\theta \sum_{x^{(i)}} \sum_y \sum_t E_\epsilon \left[ ||\epsilon_\theta(x_t^{(i)}, t, y) - \epsilon_{\theta^*}(x_t^{(i)}, t, y)||_2^2 \right]$$

여기서 $\epsilon_{\theta^*}$는 Theorem 3.2의 최적 노이즈 예측

**기대 효과:**
- 간격 d(x,y,θ) 감소
- DC만으로도 더 높은 성능

#### **2) 문제 특화 조건부 확산 모델**

**현재:**
일반적인 조건부 확산 모델

**개선:**
적대적 견고성을 목표로 한 특화 훈련:
- 손실 함수에 견고성 목표 포함
- 대적대 훈련 데이터 활용

### 7.5 평가 및 벤치마킹

#### **1) 더 포괄적 위협 모델 정의**

**필요성:**
- StAdv, ℓ∞, ℓ2 외의 다양한 공격
- 의미론적 공격 (회전, 스케일링 등)
- 물리적 공격 (조명 변화, 자연적 왜곡)

#### **2) 다양한 평가 기준**

**현재:**
- 견고 정확도 (robust accuracy)
- 깨끗함 정확도 (clean accuracy)

**추가 평가:**
- 인증 견고성 (certified robustness)
- 분포 shift 견고성
- 공정성(fairness) 측면

#### **3) 공개 벤치마크 개발**

**RobustBench의 확장:**
- 확산 기반 방어 전용 벤치마크
- 공정한 계산 환경 제공
- 투명한 평가 프로토콜

***

## 결론

"Robust Classification via a Single Diffusion Model"은 적대적 견고성 분야에서 **패러다임 전환**을 제시한 중요한 논문입니다.

**핵심 공헌:**
1. **이론적 기초**: 확산 모델 기반 생성 분류기의 엄밀한 정식화
2. **최적성 분석**: 최적 조건에서 절대 견고성 달성 가능성 증명
3. **실제 성능**: CIFAR-10에서 75.67% 견고성으로 당시 SOTA 달성
4. **일반화**: 미지 위협에 +53% 우수한 성능

**한계와 도전:**
1. 계산 복잡도: T개 타임스텝 모두 필요
2. Multi-head 구현: 성능 저하 문제
3. 이론-실제 갭: 100%에서 75.67%로의 하락
4. 평가 범위: CIFAR-10 512개 샘플에만 집중

**향후 영향:**
- 2024년 CausalDiff (86.39%), ADBM (75.27%) 등으로 확대
- 2024년 Chen et al.의 인증 견고성 증명으로 이론 강화
- 2025년 기초 분석 논문들로 메커니즘 이해 심화

이 논문은 단순히 성능 개선을 넘어 **생성 모델을 통한 근본적으로 다른 견고성 접근**을 제시함으로써 이 분야의 미래 연구 방향을 크게 변화시켰습니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a752bd35-4919-453b-9bdb-eaa421009d93/2305.15241v2.pdf)
[2](https://arxiv.org/abs/2405.16341)
[3](https://arxiv.org/abs/2410.23091)
[4](https://www.semanticscholar.org/paper/11417522f57c13898e24d87ef22f9e45fa197cf8)
[5](https://ieeexplore.ieee.org/document/10899230/)
[6](https://ieeexplore.ieee.org/document/11114945/)
[7](https://ieeexplore.ieee.org/document/10650024/)
[8](https://ieeexplore.ieee.org/document/10842888/)
[9](https://ieeexplore.ieee.org/document/10565034/)
[10](https://iopscience.iop.org/article/10.1088/1402-4896/ada476)
[11](https://arxiv.org/abs/2408.02710)
[12](http://arxiv.org/pdf/2311.16124.pdf)
[13](https://arxiv.org/pdf/2305.10388.pdf)
[14](https://arxiv.org/html/2409.09406)
[15](http://arxiv.org/pdf/2309.03702.pdf)
[16](http://arxiv.org/pdf/2410.14089.pdf)
[17](https://arxiv.org/html/2404.13320)
[18](http://arxiv.org/pdf/2402.02316.pdf)
[19](https://arxiv.org/html/2404.10335)
[20](https://proceedings.mlr.press/v202/ouyang23a/ouyang23a.pdf)
[21](https://liner.com/ko/review/robust-inference-via-generative-classifiers-for-handling-noisy-labels)
[22](https://www.sciencedirect.com/science/article/abs/pii/S0893608023004689)
[23](https://arxiv.org/abs/2505.22839)
[24](https://www.sciencedirect.com/science/article/abs/pii/S0020025520310252)
[25](https://arxiv.org/abs/2402.17563)
[26](https://openreview.net/forum?id=EVK0sQHVCd)
[27](https://arxiv.org/abs/2212.07283)
[28](https://arxiv.org/abs/2505.21742)
[29](https://arxiv.org/abs/2302.04638)
[30](https://arxiv.org/pdf/2505.22839.pdf)
[31](https://arxiv.org/abs/2201.04733)
[32](https://arxiv.org/html/2505.22839v1)
[33](https://arxiv.org/abs/2305.15241)
[34](https://arxiv.org/abs/1901.11300)
[35](https://arxiv.org/abs/2508.15020)
[36](https://arxiv.org/html/2403.16067v1)
[37](https://www.sciencedirect.com/science/article/abs/pii/S1566253524004792)
[38](https://arxiv.org/abs/2401.16352)
[39](https://www.semanticscholar.org/paper/dbc799173e76fd306b94214763cb34e70f751a83)
[40](https://arxiv.org/abs/2407.04016)
[41](https://arxiv.org/abs/2409.14940)
[42](https://ojs.aaai.org/index.php/AAAI/article/view/25118)
[43](https://ieeexplore.ieee.org/document/10208609/)
[44](https://ieeexplore.ieee.org/document/10222867/)
[45](https://arxiv.org/abs/2305.14700)
[46](https://pubs.aip.org/jasa/article/156/1/299/3302958/Adversarial-multi-task-underwater-acoustic-target)
[47](https://arxiv.org/pdf/2303.09051.pdf)
[48](http://arxiv.org/pdf/2408.00315.pdf)
[49](https://arxiv.org/html/2408.17064v3)
[50](https://arxiv.org/pdf/2205.07460.pdf)
[51](http://arxiv.org/pdf/2501.19040v1.pdf)
[52](https://arxiv.org/pdf/2411.18956.pdf)
[53](https://arxiv.org/abs/2501.13336)
[54](https://icml.cc/media/icml-2022/Slides/16707.pdf)
[55](https://www.cs.toronto.edu/~urtasun/courses/CSC411_Fall16/08_generative.pdf)
[56](https://arxiv.org/pdf/2305.03935.pdf)
[57](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
[58](https://proceedings.mlr.press/v202/zheng23c/zheng23c.pdf)
[59](https://www.themoonlight.io/en/review/how-do-diffusion-models-improve-adversarial-robustness)
[60](https://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf)
[61](https://arxiv.org/abs/2305.03935)
[62](https://github.com/NVlabs/DiffPure)
[63](https://arxiv.org/pdf/2201.00844.pdf)
[64](https://arxiv.org/html/2511.19274v1)
[65](https://arxiv.org/pdf/2408.00315.pdf)
[66](https://arxiv.org/pdf/2012.13572.pdf)
[67](https://arxiv.org/pdf/2504.07793.pdf)
[68](https://arxiv.org/html/2512.01097v1)
[69](https://arxiv.org/html/2512.11912v1)
[70](https://arxiv.org/html/2404.14309v2)
[71](https://www.sciencedirect.com/science/article/abs/pii/S0304405X01000939)
[72](https://diffpure.github.io)
[73](https://openreview.net/references/pdf?id=XmgewT6iT)
[74](https://mdporter.github.io/DS6030/lectures/gen-classifiers.pdf)
[75](https://openreview.net/pdf?id=Lxc4nBkJuq)
