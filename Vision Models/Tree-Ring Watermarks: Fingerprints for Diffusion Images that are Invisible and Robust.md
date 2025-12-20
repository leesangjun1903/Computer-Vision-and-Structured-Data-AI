
# Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust

## 1. 핵심 주장 및 주요 기여

"Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust" (Wen et al., 2023)는 **생성형 확산(Diffusion) 모델의 출력에 대한 워터마킹 문제를 근본적으로 재정의하는 획기적인 논문**이다.[1]

### 주요 기여

**첫째, 진정한 의미의 "보이지 않는" 워터마크 구현**: 기존 방식들(DwtDct, RivaGAN 등)은 생성된 이미지에 사후적 수정(post-hoc modification)을 가하여 눈에 띄는 패턴이나 감지 가능한 신호를 남긴다. Tree-Ring Watermarking은 이미지 수정을 하지 않고 **확산 과정의 초기 입력인 노이즈 벡터의 푸리에 공간에 패턴을 심음으로써, 생성된 이미지 자체에는 어떤 눈에 띄는 흔적도 남기지 않는다**.[1]

**둘째, 추가 학습 불필요한 플러그-앤-플레이 방식**: 기존의 학습 기반 방식들(예: Stable Signature)은 VAE 디코더를 미세조정해야 하고 새로운 모델에 적용하기 어렵다. Tree-Ring은 **모델의 가중치를 수정하지 않으면서도 기존의 모든 확산 모델에 적용 가능**하다.[1]

**셋째, 극도로 강건한 워터마크 검출**: 기존 최고 성능의 RivaGAN이 공격 환경에서 AUC 0.854를 달성했던 것에 비해, Tree-RingRings는 **AUC 0.975를 달성하며 무려 14% 이상 성능을 개선**했다. 특히 회전(0.935), JPEG 압축(0.999), 크롭(0.961), 가우시안 블러(0.999) 등 다양한 공격에서 우수한 강건성을 보인다.[1]

**넷째, 엄밀한 통계적 검정 프레임워크**: 워터마크 검출에 대한 P-value를 계산하여 **거짓 양성 비율을 명확하게 제어**할 수 있으며, 이는 법적, 사법적 증거로 사용 가능한 수준의 신뢰도를 제공한다.[1]

***

## 2. 문제 정의, 제안 방법, 모델 구조 및 성능

### 2.1 해결하려는 문제

AI가 생성한 이미지는 저작권 침해, 가짜 뉴스 유포, 위조 증거 제작 등 다양한 악의적 사용이 가능하다. 따라서 **생성 이미지에 추적 가능한 워터마크를 남기되, 사람의 눈에는 보이지 않으면서도 악의적 수정에 강해야 한다**는 모순적 요구를 충족해야 한다.[1]

기존 방식들의 문제점:
- **DwtDct 방식**: 후처리로 이미지 주파수 변형 → 공격에 취약 (회전 시 AUC 0.596)
- **RivaGAN**: 가시적 아티팩트 위험, 학습 필요
- **Stable Signature**: 미세조정 공격에 취약 (후속 연구에서 입증)

### 2.2 제안 방법

#### 워터마크 생성 및 임베딩

초기 노이즈 벡터 $$x_T \sim \mathcal{N}(0, I)$$의 푸리에 변환에 특정 패턴을 삽입한다:

$$F(x_T)_i \sim \begin{cases} k^*_i & \text{if } i \in M \\ \mathcal{N}(0, 1) & \text{otherwise} \end{cases} \quad (2)$$

여기서:
- $$F(x_T)$$: 초기 노이즈 벡터의 푸리에 변환
- $$M$$: 이진 마스크 (중심의 반경 r인 원형 영역)
- $$k^*$$: 워터마크 키 (선택된 유형에 따라 다름)

이후 이 변환된 노이즈 벡터는 일반적인 DDIM 샘플링 과정을 거쳐 이미지로 변환된다:

$$x_0 = D_\theta(x_T)$$

여기서 $$D_\theta$$는 DDIM 역과정의 역함수다.

#### 워터마크 검출

생성된 이미지 $$x'_0$$에서 초기 노이즈 벡터를 복원한다:

$$x'\_T = D^\dagger_\theta(x'_0)$$

여기서 $$D^\dagger_\theta$$는 DDIM 역함수로, 다음의 반복 과정을 통해 추정한다:

$$x_{t+1} = \sqrt{\bar{\alpha}_{t+1}} \hat{x}^t_0 + \sqrt{1-\bar{\alpha}_{t+1}} \epsilon_\theta(x_t)$$

여기서 $$\hat{x}^t_0 = \frac{x_t - \sqrt{1-\bar{\alpha}\_t}\epsilon_\theta(x_t)}{\sqrt{\bar{\alpha}_t}}$$

복원된 노이즈 벡터의 푸리에 공간에서 마스크 영역의 L₁ 거리를 계산한다:

$$d_{detection} = \frac{1}{|M|} \sum_{i \in M} |k^*_i - F(x'_T)_i| \quad (3)$$

이 거리가 임계값 τ보다 작으면 워터마크 검출로 판단한다.

#### 푸리에 공간 선택의 이점

푸리에 변환의 특성을 활용하면:
- **회전 불변성**: 픽셀 공간의 회전 → 푸리에 공간의 회전
- **이동 불변성**: 픽셀 이동 → 복소수 곱셈 (진폭 불변)
- **스케일 불변성**: 확대/축소 → 역방향 주파수 변환
- **색상 불변성**: 색 지터 → 제로 주파수 모드만 변경

따라서 낮은 주파수의 원형 마스크는 이들 변환에 자연적으로 불변이다.[1]

### 2.3 워터마크 키(Key) 설계

세 가지 주요 전략이 제시된다:[1]

#### Tree-RingZeros
$$k^* = \mathbf{0}$$ (영 벡터)
- **강점**: 모든 변환에 매우 강건 (회전 AUC 0.994, 노이즈 AUC 0.877)
- **약점**: 가우시안 분포에서 크게 벗어남 (이미지 품질 저하, FID 26.56), 다중 키 미지원

#### Tree-RingRand
$$k^* \sim \mathcal{N}(0, I)^{|M|}$$ (가우시안 분포)
- **강점**: 이미지 품질 최소 영향 (FID 25.47), 사용자별 고유 키 가능
- **약점**: 회전에 취약 (AUC 0.486)

#### Tree-RingRings (권장)
$$k^*_r = \text{const}_r \quad \forall r \in [1, R]$$ (각 반경 r에 대해 가우시안에서 샘플된 상수)
- **강점**: 회전 불변성 유지 (AUC 0.935), 다양한 변환에 강건 (평균 AUC 0.975), 가우시안 분포 근사 유지
- **약점**: 없음 (최고 성능)

### 2.4 통계적 워터마크 검정

워터마크 검출의 신뢰도를 수량화하기 위해 **비중심 χ² 검정**을 도입한다:[1]

귀무가설: $$H_0: y \sim \mathcal{N}(0, \sigma^2 I_C)$$

검정 통계량:
$$\eta = \frac{1}{\sigma^2} \sum_{i \in M} |k^*_i - y_i|^2 \quad (5)$$

여기서 $$\sigma^2 = \frac{1}{|M|} \sum_{i \in M} |y_i|^2$$

귀무가설이 참일 때, η는 비중심성 매개변수 $$\lambda = \frac{1}{\sigma^2} \sum_i |k^*_i|^2$$를 가진 비중심 χ² 분포를 따른다:

$$p = \Pr(\chi^2_{|M|, \lambda} \leq \eta | H_0) = \Phi_{\chi^2}(\eta) \quad (6)$$

이 p-값이 작을수록 워터마크가 검출되었을 확률이 높다. 예를 들어, 그림 3에서 보이는 바와 같이 비워터마크 이미지의 p-값은 0.27~0.91이지만, 워터마크 이미지의 p-값은 3.73×10⁻⁶⁰ 수준으로 극도로 작다.[1]

### 2.5 모델 구조

Tree-Ring Watermarking의 처리 파이프라인은 다음과 같다 (그림 1 참조):[1]

```
[초기 노이즈] → [예측/FFT] → [마스크 적용] → [IFFT] → [DDIM] → [워터마크 이미지]
   x_T         ε_θ        k* 임베드         F⁻¹     D_θ      x_0

[탐지] 역방향:
[이미지] → [DDIM 역변환] → [FFT] → [키 비교] → [거리 계산] → [P-value] → [검출/미검출]
   x'_0      D†_θ           F      vs k*        d           p
```

이 구조의 핵심은 **DDIM 역함수의 근사 성능**이다. 논문에서 실증적으로 확인한 바와 같이, 생성 단계와 탐지 단계의 DDIM 스텝 수가 크게 달라도(0~800 사이 모든 조합) AUC 저하가 최소(최소 0.799)임을 보인다.[1]

### 2.6 성능 향상

#### 주요 성능 지표 (Stable Diffusion-v2)

**청정 이미지 (Clean Setting)**:
- Tree-RingRings: AUC 1.000, TPR@1%FPR 1.000
- 기존 최고 (RivaGAN): AUC 0.999, TPR@1%FPR 0.999

**공격 환경 (Adversarial Setting - 6가지 공격 평균)**:
- Tree-RingRings: AUC 0.975, TPR@1%FPR 0.694
- RivaGAN: AUC 0.854, TPR@1%FPR 0.448
- 성능 향상: **14% AUC 개선, 55% TPR 개선**

**이미지 품질**:
- Tree-RingRings FID: 25.93±0.13 (무워터마크 25.29와 거의 동등)
- CLIP Score: 0.364±0.000 (무워터마크 0.363과 동등)

#### 각 공격별 세부 성능 (표 2)

| 공격 유형 | Tree-RingRings AUC | 기타 최고 방법 | 개선율 |
|-----------|------------------|--------------|-------|
| 회전 (75°) | 0.935 | DwtDctSvd 0.431 | 117% |
| JPEG (25%) | 0.999 | RivaGAN 0.981 | 2% |
| 크롭+스케일 | 0.961 | RivaGAN 0.999 | -4%* |
| 블러 (8×8) | 0.999 | RivaGAN 0.974 | 3% |
| 노이즈 (σ=0.1) | 0.944 | RivaGAN 0.888 | 6% |
| 컬러 지터 | 0.983 | RivaGAN 0.963 | 2% |
| **평균** | **0.975** | **0.854** | **14%** |

*크롭은 마스크 영역 손실로 약간의 성능 저하는 예상되는 결과

### 2.7 한계

#### 기술적 한계

**첫째, DDIM 의존성**: 논문에서 명시된 주요 한계로서, **현재 Tree-Ring은 DDIM 샘플러에만 적용 가능**하다. 향후 Euler, DPM++, LMSDiscrete 등 다른 샘플러가 선호되면 워터마킹 기능이 작동하지 않을 수 있다.[1]

**둘째, DDIM 역함수 정확도**: 생성 단계와 탐지 단계의 DDIM 스텝 수 차이가 클 때 역함수 근사 오차가 증가한다. 논문의 그림 5에서 보이듯이, 생성 50 스텝/탐지 2 스텝 조합에서 AUC가 0.808로 저하되는데, 이는 **근사 오차의 누적**을 의미한다.[1]

**셋째, 모델 접근성 요구**: 워터마크 검출 시 **모델 파라미터 θ에 대한 화이트박스 접근이 필수**이다. 이는 API 소유자만 검출 가능하고 제3자가 독립적으로 검증할 수 없다는 의미다.[1]

#### 설계적 한계

**첫째, 다중 키 용량**: 각 사용자에게 고유의 k*를 할당할 수 있는지, 그렇다면 얼마나 많은 사용자를 지원할 수 있는지에 대한 분석이 부재하다.[1]

**둘째, 조건부 모델 불확실성**: CLIP guidance나 Cross-Attention과 같이 추가 조건이 있을 때, 탐지 시 실제 guidance scale이나 prompt embedding을 모르면 DDIM 역함수 정확도가 저하될 수 있다. 실제로 그림 6(b)에서 guidance scale 18에서도 성능이 유지되지만, 이는 경험적 관찰일 뿐 이론적 보장이 없다.[1]

**셋째, 방어 메커니즘 부재**: 악의적 행위자가 충분히 정교한 적대적 공격(adversarial attack)을 설계하면 워터마크를 제거할 가능성이 있다. 예를 들어, diffusion 기반 이미지 재생성이나 선택적 노이즈 주입을 통해 푸리에 공간의 패턴을 손상시킬 수 있다.[1]

***

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화 성능

#### 크로스 모델 성능

논문은 두 개의 서로 다른 모델에서 Tree-Ring의 일관된 성능을 입증한다:[1]

**Stable Diffusion v2 (조건부, 512×512)**:
- 무공격: AUC 1.000
- 공격 평균: AUC 0.975

**ImageNet 무조건부 모델 (256×256)**:
- 무공격: AUC 0.999
- 공격 평균: AUC 0.966

이는 **Tree-Ring이 조건부/무조건부 모델, 다양한 해상도, 서로 다른 아키텍처에서 강건하게 작동함**을 보여준다. 특히 ImageNet 모델에서도 평균 AUC 0.966을 유지하여, 모델 불변성이 매우 높다.

#### 하이퍼파라미터 강건성

**생성 vs 탐지 스텝 수 차이 (그림 5)**:
- 생성 50 스텝, 탐지 800 스텝: AUC 0.967
- 생성 800 스텝, 탐지 50 스텝: AUC 0.968
- 생성 10 스텝, 탐지 10 스텝: AUC 0.808 (가장 낮음)
- **평균 AUC 범위: 0.799~0.972**

이는 **DDIM 역함수가 스텝 수에 큰 영향을 받지 않음**을 시사하며, 사용자가 생성 시 자유롭게 스텝 수를 선택할 수 있음을 의미한다.

**워터마크 반지름 (그림 6a)**:
$$r \in \{1, 2, 4, 8, 16, 32\}$$에 대해:
- r=1: AUC 0.385, FID 25.35
- r=10: AUC 0.929, FID 25.93
- r=16: AUC 0.929, FID 28.96

r=10이 최적점으로, 작은 r에서는 신호가 약하고 큰 r에서는 이미지 품질이 저하된다. 이는 **일반화와 품질의 자명한 트레이드오프**를 보여준다.

**Guidance Scale (그림 6b)**:
- 범위: 2~18 모두에서 강건한 성능 (AUC 0.8 이상)
- 표준 범위 (7.5) 기준으로 편차 최소

### 3.2 일반화 향상을 위한 이론적 근거

#### 푸리에 공간의 불변성

Tree-Ring이 높은 일반화 성능을 보이는 근본 이유는 **푸리에 변환의 수학적 불변성**에 있다:

회전: 픽셀 공간의 각도 θ 회전은 푸리에 공간에서도 각도 θ 회전으로 나타난다. 원형 마스크는 회전에 불변이므로:

$$F(R_\theta(x))\_i = e^{j\phi} \cdot F(x)_{R_{-\theta}(i)}$$

이동: 픽셀 공간의 이동 $$x \to x + \mathbf{c}$$는 푸리에 공간에서 진폭 불변 위상 이동:
$$F(x + \mathbf{c})_i = e^{j2\pi\langle i, \mathbf{c}\rangle/N} \cdot F(x)_i$$

그러나 마스크 영역의 거리 측정은 진폭만 사용하므로, 이동이 거리에 영향을 주지 않는다.

#### DDIM 역함수의 강건성

DDIM 역과정의 근사 오차 $$\|x_T - x'_T\|$$는 diffusion process 동안 점진적으로 축적되지만, **낮은 주파수 성분은 더 강건하게 보존된다**. 이는 다음 이유에서다:

1. **저주파 성분의 초기 설정**: 초기 노이즈 벡터의 저주파 영역에 임베드된 신호는, diffusion 과정 초기에 이미지 구조의 기본 골격을 결정한다.
2. **누적 오차의 완화**: diffusion은 고주파부터 점진적으로 제거하므로, 저주파 오차는 상대적으로 작다.
3. **가우시안 분포 보존**: forward diffusion은 데이터를 가우시안으로 변환하는 과정이므로, 역함수가 이를 반대로 추정할 때 저주파 성분은 더 정확하게 복원된다.

따라서 **저주파 원형 마스크의 선택은 이론적으로도, 경험적으로도 정당**하다.

### 3.3 향후 일반화 성능 향상 방향

#### 1. 샘플링 방식 독립화

현재 DDIM에만 의존하는 한계를 극복하기 위해:

**가역 diffusion 활용** (Wallace et al., 2022):
$$x_t \leftarrow x_{t+1} - \epsilon_t \Rightarrow x_t = g(x_{t+1}, \epsilon_t)^{-1}$$

이를 통해 정확한 역함수를 보장받을 수 있다.

**범용 역함수 학습**:
신경망 기반 인코더 $$E_\phi(x_0) \approx x_T$$를 학습하여:
- DDIM 외 다른 샘플러에도 적용 가능
- 미리 학습된 어댑터로 빠른 적용

#### 2. 다중 마스크 영역 활용

단일 원형 마스크 대신 **여러 개의 마스크 영역을 사용**하면:
$$M = M_1 \cup M_2 \cup \ldots \cup M_k$$

각 영역에 다른 정보를 인코딩:
- 사용자 ID
- 타임스탐프
- 메타데이터

이는 워터마크 용량을 exponential하게 증가시킨다.

#### 3. 분포 보존 강화

Tree-RingRings를 개선하여 **더욱 엄격한 가우시안 분포 준수**:
$$k^*_r \sim \mathcal{N}(\mu_r, \sigma^2_r)$$

각 반경별로 정규화된 값을 사용하면, diffusion 샘플링 과정에서의 분포 시프트를 최소화할 수 있다.

#### 4. 적응형 마스크 반지름

프롬프트나 이미지 특성에 따라 **동적으로 반지름 조정**:
$$r = r_0 + \Delta r \cdot f(\text{prompt embedding})$$

이는 특정 프롬프트에 최적화된 워터마킹을 가능하게 한다.

#### 5. Guidance-Aware 역함수

조건부 모델의 경우, 탐지 시 guidance scale을 추정하고 이를 역함수에 반영:

$$x'_T = D^\dagger_{\theta, c, s}(x'_0)$$

여기서 s는 추정된 guidance scale이다. 이는 recent works (ROBIN, 2024)에서 제시된 방향이다.[2]

### 3.4 실험적 검증이 필요한 영역

#### 일반화 성능 검증 격차

논문이 명시적으로 테스트하지 않은 영역:

1. **크로스 모델 일반화**: 한 모델에서 학습된 워터마크가 다른 모델에서도 검출 가능한가?
2. **미래 모델 호환성**: 더 큰 모델이나 새로운 아키텍처에 적용 가능한가?
3. **도메인 시프트**: 자연 이미지 외 의료 이미지, 위성 이미지 등에서의 성능?
4. **극단적 공격 조합**: 6가지 공격을 동시에 적용했을 때의 성능 (부록 그림 9에서는 1~5개 공격만 테스트)

***

## 4. 향후 연구 영향 및 고려할 점

### 4.1 학문적 영향

#### 패러다임 전환

Tree-Ring은 **워터마킹 분야에 새로운 패러다임을 제시**했다:

**Before (Post-hoc 방식)**:
- 이미지 생성 후 신호 추가
- 가시성 vs 강건성 트레이드오프 불가피
- 최고 성능: AUC ~0.85

**After (Distribution-modifying 방식)**:
- 생성 과정 중 출력 분포 수정
- 가시성 0 유지하면서 강건성 극대화
- 새로운 성능 기준: AUC ~0.97

이는 이후의 ROBIN, Gaussian Shading, SAT-LDM 등 2024-2025년 최신 논문들이 모두 분포 기반 또는 적대적 최적화 방식으로 진화하는 것으로 입증된다.[3][4][2]

#### 이론적 기여

**저주파 기반 워터마킹 이론화**:
- 푸리에 공간의 수학적 불변성이 강건성을 보장함을 보임
- 원형 마스크의 회전 불변성이 일반화를 가능하게 함

**DDIM 역함수의 강건성 분석**:
- 스텝 수 차이에 강건함을 실증적으로 입증
- 근사 오차의 누적이 제한적임을 보임

**통계적 검정 도입**:
- χ² 검정을 통한 p-value 기반 검출로 법적 신뢰도 제공
- 거짓 양성률을 명시적으로 제어 가능

### 4.2 실무적 영향

#### 산업 채택 가능성

**높은 실용성**:
1. 추가 학습 불필요 (모델 수정 불필요)
2. 계산 오버헤드 최소 (FID 영향 ~0.6%)
3. 플러그-앤-플레이 적용 (모든 DDIM 기반 모델)

이는 Stable Diffusion, Midjourney 등 실제 서비스에서 즉시 도입 가능함을 의미한다.

**법적 증거로 활용**:
- P-value 기반 정량화로 법원 제출 가능
- 예: "이 이미지가 우리 모델에서 생성되었을 확률 >99.9999%"

#### 보안 규제와의 연계

- **EU AI Act**: 생성 AI 모델 출력에 워터마크 요구
- **자가 규제 이니셔티브**: Content Authenticity Initiative (CAI)
- **정부 정책**: NIST AI Risk Management Framework

Tree-Ring은 이들 규제 요구를 실제로 충족시킬 수 있는 기술로서 정책 입안자들의 관심을 받고 있다.

### 4.3 향후 연구 시 고려할 점

#### 기술 발전 추적

**새로운 샘플링 알고리즘 대응**:
- 논문 발표 후 2년간 Euler, DPM++, Flow-Matching 등 새로운 샘플러 출현
- Tree-Ring 적응 방법 개발 필수
- 가역 diffusion이 표준화되면 이를 활용한 정확한 역함수 구현

**더 강한 공격 방어**:
- Diffusion-based regeneration attack (Zhao et al., 2024) 등장
- 논문에서는 6가지 표준 공격만 테스트 → 더 정교한 adversarial attack 필요
- 적응형 공격(adaptive attack)에 대한 강건성 분석

**미세조정 공격**:
- Stable Signature도 미세조정 공격에 취약함이 증명됨
- Tree-Ring도 model update 공격 테스트 필요

#### 다양한 적용 시나리오 탐색

**1. 비디오 생성 모델로 확장**:
- Diffusion-based 비디오 생성 모델 (Runway, Pika) 등장
- 시간 축 정보 추가 필요
- VideoShield (2025) 등 후속 연구에서 다룸[5]

**2. 음성/오디오 확산 모델**:
- TTS 기반 음성 생성 감시 필요
- 음성의 시간-주파수 특성 활용

**3. 다양한 해상도 지원**:
- 현재는 256×256, 512×512에서만 테스트
- 매우 고해상도(4K, 8K) 이미지에서의 성능?

**4. 멀티모달 워터마킹**:
- 이미지 + 텍스트 프롬프트를 함께 워터마크
- 더 많은 정보 용량 가능

#### 보안 고려 사항

**1. 키 관리**:
- k*는 비공개로 유지되어야 함
- 키 유출 시 워터마크 위조 가능
- 따라서 키 로테이션, 백업, 접근 제어 메커니즘 필요

**2. 위조 방지**:
- 공격자가 무작위 노이즈에 임의로 패턴을 삽입하면?
- 논문에서는 통계적 검정으로 자연적 이미지와 구분 가능함을 보였으나,
- 더 정교한 위조(예: 기존 이미지에 유사 패턴 삽입)에 대한 대응 필요

**3. 개인정보 보호**:
- 워터마크에 사용자 ID 인코딩 시, 개인 추적 가능성
- GDPR 등 규제 준수 필요

#### 평가 방법론 개선

**1. 더 광범위한 공격 벤치마크**:
- NIPS 2023 TROJAI 대회 수준의 체계적 공격 평가
- 10개 이상의 공격 방식 포함

**2. 크로스 모델 호환성 테스트**:
- Flux, Hunyuan 등 최신 모델과의 호환성
- 다른 기관의 모델과 호환성

**3. 사실적 사용 시나리오**:
- SNS 업로드 후 재압축 (Instagram, Twitter의 압축률)
- 스크린샷 후 재생성
- 실시간 사용 환경에서의 성능

**4. 사용자 연구**:
- 워터마크 이미지의 가시성이 정말 0인가? (심리물리학적 검증)
- 사용자가 지각할 수 있는 미세한 차이?

### 4.4 2020년 이후 관련 최신 연구 비교 분석

#### 시간 축 진화

**초기 단계 (2020-2022): 기존 기법 적용**
- DwtDct, RivaGAN 등 기존 이미지 워터마크 기법 적용
- 성능: AUC ~0.6-0.85, 강건성 낮음

**전환점 (2023년 중반): 패러다임 전환**
- **Tree-Ring Watermarks** (Wen et al., NIPS 2023): 분포 수정 방식, AUC 0.975
- **Stable Signature** (Fernandez et al., ICCV 2023): 모델 미세조정, AUC ~0.90+

**고도화 단계 (2024): 다양한 접근법 병행**

| 논문 | 년도 | 방식 | AUC | 특징 |
|------|------|------|-----|------|
| ZoDiac | 2024 | 잠재 공간 + Stable Diffusion | 0.98+ | latent vector 활용 |
| ROBIN | 2024 | 적대적 최적화 | 0.95+ | 프롬프트 임베딩 최적화 |
| Gaussian Shading | 2024 | 성능 보존 분포 | 0.95+ | 계산 효율성 |
| Shallow Diffuse | 2024 | 저차원 부분공간 | 0.94+ | 얕은 diffusion 활용 |
| SuperMark | 2024 | Super-Resolution 활용 | 0.96+ | 초해상도로 강건성 확보 |

**최신 단계 (2025): 특수 응용 확장**
- VideoShield (2025): 비디오 생성 모델 확장[5]
- SAT-LDM (2025): 자체 증강으로 일반화 개선[4]
- SleeperMark (2025): 미세조정 강건성[6]
- Watermarking Discrete Diffusion (2025): 이산 diffusion 모델[7]

#### 성능 비교

**강건성 측면**:
- Tree-Ring (2023): 0.975 (6가지 표준 공격)
- ZoDiac (2024): 0.98+ (MS-COCO, DiffusionDB, WikiArt 다중 데이터셋)
- ROBIN (2024): 0.95+ (적대적 공격 포함)
- SuperMark (2024): 0.96+ (재생성 공격 특화)

**이미지 품질 영향**:
- Tree-Ring: FID +0.64, 거의 무시할 수준
- Stable Signature: 미세조정으로 인한 약간의 영향
- Gaussian Shading: "성능 손실 없음" 주장
- CLUE-Mark (2024): "증명된 품질 보존" (이론적 보장)[8]

**적용 난이도**:
- Tree-Ring: 극히 간단 (초기 노이즈 수정)
- Stable Signature: 중간 (VAE 디코더 미세조정)
- ROBIN: 중간 (프롬프트 임베딩 최적화)
- CLUE-Mark: 복잡 (CLWE 구현)

#### 주요 차이점 분석

**Tree-Ring vs Stable Signature**:

| 항목 | Tree-Ring | Stable Signature |
|------|-----------|-----------------|
| 학습 필요 | 없음 | 필요 (미세조정) |
| 모델 수정 | 초기 노이즈만 | VAE 디코더 가중치 |
| 적용 대상 | 모든 DDIM 모델 | Latent Diffusion 모델만 |
| 강건성 | 0.975 (표준) | 0.90+ (미세조정 공격에 취약) |
| 계산 비용 | 무시할 수준 | VAE 미세조정 필요 |
| 다중 모델 지원 | 예 | 예 (각각 미세조정 필요) |

**Tree-Ring vs ROBIN (2024)**:

| 항목 | Tree-Ring | ROBIN |
|------|-----------|-------|
| 핵심 아이디어 | 푸리에 마스크 | 프롬프트 최적화 |
| 강건성 | 0.975 | 0.95+ |
| 보이지 않음 성능 | 우수 | 극우수 (적대적 최적화) |
| 복잡도 | 간단 | 복잡 (최적화 필요) |
| 이론적 근거 | 푸리에 불변성 | 적대적 목표함수 |

#### 한계와 개선 방향

**Tree-Ring의 남은 문제**:
1. DDIM 의존성 → 다른 샘플러 지원 필요
2. 화이트박스 탐지 → 블랙박스 API 탐지 필요
3. 정교한 공격 미검증 → adversarial robustness 평가 필요

**산업이 주목하는 방향** (2024-2025):
1. **더 강한 공격 저항**: Diffusion 기반 재생성 공격 등
2. **비디오/오디오 확장**: 멀티모달 워터마킹
3. **미세조정 강건성**: 사용자 맞춤형 모델 업데이트 상황
4. **증명 가능한 특성**: CLUE-Mark처럼 이론적 보장
5. **높은 용량**: 메타데이터 인코딩 가능성

### 4.5 실제 배포 고려사항

#### 규제 준수

- **EU AI Act**: 생성 AI 투명성 요구 → Tree-Ring이 핵심 기술
- **Content Authenticity Initiative**: 출처 추적 → 통합 필요
- **국가별 규제**: 중국의 생성 AI 규제, 미국의 행정 명령

#### 경제성 분석

**초기 투자**:
- 기술 통합: 낮음 (코드 간단)
- 운영 비용: 거의 없음 (탐지는 API 호출 시만)

**비용 편익**:
- 저작권 소송 회피
- 브랜드 신뢰도 향상
- 규제 컴플라이언스 달성

#### 사용자 경험

**사용자 입장**:
- 생성 시간 증가 없음 (0.6% FID 영향만)
- 이미지 품질 저하 없음
- 워터마크 존재 인식 불가

**관리자 입장**:
- 간단한 통합 프로세스
- 검증 API 구현 필요
- 알고리즘 신뢰성 평가

***

## 결론

Tree-Ring Watermarking은 **확산 모델 기반 이미지 생성의 추적 가능성을 획기적으로 개선**하는 논문이다. 초기 노이즈의 푸리에 공간에 원형 패턴을 심고, DDIM 역함수를 통해 검출하는 우아한 기법은 이론적 엄밀성과 실무적 효율성을 동시에 달성한다.

특히 **일반화 성능 측면에서 이 논문의 가장 큰 강점**은 다음과 같다:

1. **수학적 불변성에 기초한 강건성**: 푸리에 변환의 기하학적 불변성이 다양한 이미지 변환에 자연스럽게 저항
2. **샘플링 방식 불변성**: DDIM 스텝 수 차이에도 큰 성능 저하 없음
3. **모델 아키텍처 불변성**: 조건부/무조건부, 다양한 해상도에서 일관된 성능
4. **분포 보존 메커니즘**: Tree-RingRings 설계로 가우시안 분포 유지

그러나 향후 연구는 다음을 중점적으로 다루어야 한다:

1. **샘플러 독립화**: DDIM 외 다양한 샘플러 지원
2. **강력한 공격 저항**: Diffusion 기반 재생성, 적대적 미세조정 등
3. **블랙박스 탐지**: 모델 접근 없이도 검증 가능한 메커니즘
4. **멀티모달 확장**: 비디오, 오디오 등으로 워터마킹 개념 확대

Tree-Ring이 2023년 NeurIPS에 발표된 이후 다양한 후속 연구(ZoDiac, ROBIN, SAT-LDM 등)가 이어진 것은 **이 논문이 확산 모델 워터마킹 분야의 새로운 방향을 제시했음**을 명확히 보여준다. 2025년 현재, 이 방법은 산업 표준으로 자리 잡을 가능성이 매우 높으며, 규제와 기술의 양측면에서 AI 생성 콘텐츠의 투명성과 추적성을 담보하는 핵심 기술로 인정받고 있다.

***

## 참고문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/1b171c65-a325-4767-abe6-12edc59dc47f/2305.20030v3.pdf)
[2](https://arxiv.org/html/2411.03862)
[3](http://arxiv.org/pdf/2404.04956.pdf)
[4](https://arxiv.org/html/2501.00463v2)
[5](https://arxiv.org/html/2501.14195v1)
[6](https://arxiv.org/pdf/2412.04852.pdf)
[7](https://arxiv.org/html/2511.02083v1)
[8](https://arxiv.org/html/2411.11434v3)
[9](https://arxiv.org/abs/2412.10049)
[10](https://arxiv.org/html/2401.04247v1)
[11](https://ieeexplore.ieee.org/document/10377226/)
[12](https://ieeexplore.ieee.org/document/11259417/)
[13](https://arxiv.org/abs/2405.07145)
[14](http://www.proceedings.com/079017-1215.html)
[15](https://arxiv.org/abs/2303.15435)
[16](https://www.nature.com/articles/s41586-024-08025-4)
[17](https://jnls.alnoor.edu.iq/article_189631.html)
[18](https://edu.pubmedia.id/index.php/ptk/article/view/1603)
[19](https://www.esri.ie/publications/projections-of-regional-demand-and-workforce-requirements-for-general-practice-in)
[20](https://mediasphera.ru/issues/meditsinskie-tekhnologii-otsenka-i-vybor/2023/2/1221906782023021023)
[21](https://jpfis.unram.ac.id/index.php/GeoScienceEdu/article/view/588)
[22](https://badanpenerbit.org/index.php/SEMNASPA/article/view/2211)
[23](https://jurnal.uns.ac.id/SHES/article/view/97214)
[24](https://ejurnal.stpkat.ac.id/index.php/jutipa/article/view/369)
[25](https://jurnal.unej.ac.id/index.php/biograph-i/article/view/47383)
[26](https://journal-laaroiba.com/ojs/index.php/mk/article/view/3398)
[27](http://arxiv.org/pdf/2305.20030v2.pdf)
[28](https://arxiv.org/html/2410.21088)
[29](https://www.sciencedirect.com/science/article/abs/pii/S0957417424010674)
[30](https://www.sciencedirect.com/science/article/abs/pii/S0020025525008199)
[31](https://arxiv.org/abs/2410.07369)
[32](https://proceedings.neurips.cc/paper_files/paper/2023/file/b54d1757c190ba20dbc4f9e4a2f54149-Paper-Conference.pdf)
[33](https://openaccess.thecvf.com/content/CVPR2025W/ReGenAI/papers/Chen_Dynamic_watermarks_in_images_generated_by_diffusion_models_CVPRW_2025_paper.pdf)
[34](https://openaccess.thecvf.com/content/WACV2025/papers/Xu_InvisMark_Invisible_and_Robust_Watermarking_for_AI-Generated_Image_Provenance_WACV_2025_paper.pdf)
[35](https://arxiv.org/html/2508.08836v1)
[36](https://arxiv.org/html/2509.22126v1)
[37](https://www.arxiv.org/pdf/2508.08836.pdf)
[38](https://arxiv.org/html/2510.05978v1)
[39](https://arxiv.org/html/2503.19176v1)
[40](https://arxiv.org/html/2502.08927v2)
[41](https://arxiv.org/abs/2410.18775)
[42](https://arxiv.org/html/2411.07795v2)
[43](https://www.arxiv.org/pdf/2511.02083.pdf)
[44](https://proceedings.neurips.cc/paper_files/paper/2024/file/10272bfd0371ef960ec557ed6c866058-Paper-Conference.pdf)
[45](https://proceedings.neurips.cc/paper_files/paper/2024/file/073c8584ef86bee26fe9d639ec648e28-Paper-Conference.pdf)
[46](https://dl.acm.org/doi/10.1145/3689236.3689266)
[47](https://ieeexplore.ieee.org/document/10208651/)
[48](https://aacrjournals.org/bloodcancerdiscov/article/4/3_Supplement/A15/725966/Abstract-A15-Computational-modeling-of-methylation)
[49](https://arxiv.org/abs/2401.04247)
[50](https://arxiv.org/abs/2412.19834)
[51](https://arxiv.org/abs/2506.00652)
[52](https://dl.acm.org/doi/10.1145/3664647.3681418)
[53](http://arxiv.org/pdf/2401.04247.pdf)
[54](https://arxiv.org/html/2407.13188v1)
[55](https://arxiv.org/html/2412.19834v1)
[56](http://arxiv.org/pdf/2404.00230.pdf)
[57](https://pmc.ncbi.nlm.nih.gov/articles/PMC8816581/)
[58](https://openaccess.thecvf.com/content/ICCV2023/papers/Fernandez_The_Stable_Signature_Rooting_Watermarks_in_Latent_Diffusion_Models_ICCV_2023_paper.pdf)
[59](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03220.pdf)
[60](https://hanlin-zhang.com/impossibility-watermarks/)
[61](https://pierrefdz.github.io/publications/stablesignature/)
[62](https://arxiv.org/html/2503.22330v1)
[63](https://arxiv.org/html/2409.02915v1)
[64](https://arxiv.org/html/2503.12172v3)
[65](https://arxiv.org/html/2404.05607v2)
[66](https://arxiv.org/abs/2502.13345)
[67](https://arxiv.org/html/2412.04653v5)
[68](https://arxiv.org/html/2502.07845v1)
[69](https://openaccess.thecvf.com/content/ICCV2023/html/Fernandez_The_Stable_Signature_Rooting_Watermarks_in_Latent_Diffusion_Models_ICCV_2023_paper.html)
