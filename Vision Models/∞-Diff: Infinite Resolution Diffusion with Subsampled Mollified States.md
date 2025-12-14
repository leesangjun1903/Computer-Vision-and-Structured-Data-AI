# ∞-Diff: Infinite Resolution Diffusion with Subsampled Mollified States

### 1. 논문의 핵심 주장 및 기여

**∞-Diff** (∞-Diff: Infinite Resolution Diffusion with Subsampled Mollified States)는 **무한 해상도 데이터 생성을 위한 혁신적인 접근방식**을 제시합니다. 이 논문의 핵심 기여는 다음과 같습니다:[1]

- **무한차원 Hilbert 공간에서 정의된 가우시안 확산 모델** 도입으로 임의의 해상도에서 복잡한 데이터 생성 가능
- **신경 연산자(Neural Operators) 기반의 함수공간 아키텍처** 설계로 압축 없이 직접적인 공간 문맥 집계 달성
- **Mollified 확산 프로세스** 도입으로 불규칙성을 제거하고 모델의 연속성 개선
- **8배 부표본화(8× subsampling)에서도 고품질 확산 유지** 달성으로 메모리 및 실행 시간 대폭 절감
- 기존 무한차원 생성 모델 대비 **FID 점수 우수성**: FFHQ-256에서 3.87 (StyleGAN2: 2.35 대비, 한정된 데이터로도 경쟁력 있음)

***

### 2. 해결하고자 하는 문제

#### 2.1 근본적인 문제점

전통적인 확산 모델의 한계:[1]

1. **해상도에 따른 메모리/계산 복잡성 증가**: 고정된 균일 격자(uniform grid)에서 작동하므로 해상도가 높아질수록 필요한 메모리와 계산량이 급격히 증가
2. **신경 필드 기반 모델의 품질 저하**: 기존 무한차원 신경 필드 모델들(D2F, DPF 등)은 점별 독립적 함수로 작동하므로 공간 문맥 정보 부족으로 생성 품질이 현저히 떨어짐
3. **잠재 벡터 압축의 한계**: 신경 필드 조건부 모델들이 전역 정보를 위해 압축 잠재 벡터에 의존하면서 정보 손실 발생
4. **해상도 간 일반화 부족**: 특정 해상도에서 학습한 모델이 다른 해상도에서의 생성 성능이 급격히 저하

#### 2.2 설계 아이디어

기존 신경 필드 기반 무한차원 모델의 근본적인 문제를 해결하기 위해 논문은 다음을 주장합니다:[1]

> 압축 기반 신경 필드는 비국소(non-local) 적분 연산자를 사용하는 신경 연산자로 대체되어야 함. 이는 표준 확산 아키텍처 설계 원칙과 일치하며, 매 스텝마다 상태를 잠재 벡터로 압축하는 것의 비실용성을 피할 수 있음.

***

### 3. 제안하는 방법 (수식 포함)

#### 3.1 무한차원 Mollified 확산 프로세스

논문은 Hilbert 공간 $H$에서 L2 함수 공간으로 확산 상태공간을 제한합니다. 가우시안 측도는 특성함수로 정의됩니다:[1]

$$\hat{\mu}(x) = \exp\left(i\langle x, m\rangle + \frac{1}{2}\langle Cx, x\rangle\right)$$

여기서 $m \in H$는 평균, $C: H \rightarrow H$는 자기수반(self-adjoint), 비음(non-negative), trace-class 공분산 연산자입니다.

**핵심 문제**: 백색 잡음(white noise) $N(0, C_I)$은 Hilbert 공간 $H$에 속하지 않음 (trace-class 조건 불만족). 이를 해결하기 위해 **Mollification** 도입:

$$h(c) = \int_{\mathbb{R}^n} K(c-y, l) x(y) dy, \quad K(y,l) = \frac{1}{(4\pi l)^{n/2}} e^{-|y|^2/4l}$$

여기서 $K$는 가우시안 핵이고 $l > 0$은 평활화 파라미터입니다.[1]

**전향 과정의 한계 분포**:[1]

$$q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}Tx_0, (1-\bar{\alpha}_t)TT^*)$$

여기서 계수는 표준 확산 모델과 동일합니다 ( $\bar{\alpha}\_t = \prod_{s=1}^{t} \alpha_s$, $\alpha_s = 1 - \beta_s$ ).

**후향 과정의 사후 분포** (폐쇄형 표현):[1]

$$q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t TT^*)$$

여기서:

$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}Tx_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t$$

$$\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$$

**학습 손실함수**:[1]

잡음 예측 파라미터화를 사용:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}f_\theta(x_t, t)\right)$$

단순화된 손실:

$$L^{\text{simple}}_{t-1} = \mathbb{E}_q\left[\|f_\theta(x_t, t) - T\xi\|^2_H\right]$$

#### 3.2 신경 연산자를 통한 매개변수화

신경 연산자는 두 개의 무한차원 함수 공간을 매핑합니다. **비국소 적분 핵 연산자**로 정의:[1]

$$(K(x; \phi)v_l)(c) = \int_D \kappa_\phi(c, b, x(c), x(b))v_l(b) db, \quad \forall c \in D$$

여러 계층을 쌓으면:[1]

$$v_{l+1}(c) = \sigma(Wv_l(c) + (K(x; \phi)v_l)(c)), \quad \forall c \in D$$

여기서 $W: \mathbb{R}^d \rightarrow \mathbb{R}^d$는 점별 선형 변환, $\sigma$는 활성화 함수입니다.

**푸리에 신경 연산자(FNO) 예시**:[1]

$$(K(x; \phi)v_l)(c) = \mathcal{G}^{-1}(R_\phi \cdot (\mathcal{G}v_t))(c)$$

여기서 $\mathcal{G}$는 푸리에 변환, $R_\phi$는 푸리에 공간에서 학습된 변환입니다.

#### 3.3 희소 신경 연산자 (Sparse Neural Operators)

각 좌표의 국소 영역에서 제한된 번역 불변 핵을 사용:[1]

$$x(c) = \int_{N(c)} \kappa(c-y)v(y) dy, \quad \forall c \in D$$

여기서 $N(c)$는 좌표 $c$의 국소 영역, $\kappa$는 깊이 방향 핵(depthwise kernel)입니다.

***

### 4. 모델 구조

#### 4.1 다중 스케일 아키텍처

논문은 **U-Net 영감의 하이브리드 다중 스케일 아키텍처** 설계:[1]

1. **희소 연산자 계층** (상단): 불규칙하게 표본화된 원본 데이터에서 작동
   - 3개의 희소 잔차 합성곱 연산자 블록
   - 각 블록: 깊이 방향 희소 합성곱 (커널 크기 7, 채널 64) + 3계층 MLP
   - 변조된 계층 정규화(Modulated Layer Normalization)로 확산 타임스텝 조건화

2. **정규 격자 기반 계층** (중간/하단): 글로벌 정보 집계
   - 희소 데이터를 정규 격자로 보간(interpolation)
   - 밀집 합성곱 사용 (FNO보다 성능 우수)
   - 해상도 변화 시에만 희소 연산자 사용

3. **효율성 최적화**:
   - TorchSparse 라이브러리 활용으로 메모리/실행 시간 최적화
   - 깊이 방향 합성곱으로 매개변수 효율성 증대

#### 4.2 아키텍처의 필수 특성[1]

설계된 아키텍처는 다음 특성을 만족:

1. **임의 좌표에서 입력 수용**: 정규 격자에 제약받지 않음
2. **학습 표본과 다른 입력점 개수 일반화**: 다양한 표본화 비율에 대응
3. **글로벌 및 로컬 정보 포착**: 다중 스케일 구조로 구현
4. **높은 확장성**: 대규모 입력점에 효율적 (메모리 및 실행 시간)

***

### 5. 성능 향상 및 실험 결과

#### 5.1 정량적 성능 (FID CLIP 점수)[1]

| 데이터셋 | ∞-Diff | 최고 무한차원 모델 | 최고 유한차원 모델 |
|---------|--------|------------------|------------------|
| CelebA-HQ-64 | **4.57** | GASP: 9.29 | StyleGAN2: - |
| CelebA-HQ-128 | **3.02** | GASP: 27.31 | StyleSwin: 3.39 |
| FFHQ-256 | **3.87** | GASP: 24.37 | UT: 3.05 |
| Church-256 | **10.36** | GASP: 37.46 | StyleGAN2: 6.21 |

#### 5.2 주요 성과

1. **부표본화 강건성**: 8배 부표본화에서도 품질 유지 (FFHQ-128: 4.75 → 6.48)[1]
2. **해상도 일반화**: 256×256에서 학습한 모델이 1024×1024에서 일관되고 다양한 샘플 생성[1]
3. **계산 효율성**:
   - 4배 부표본화: 1.3배 속도향상, 메모리 2배 감소[1]
   - 8배 부표본화: 1.6배 속도향상, 메모리 2.46배 감소[1]

#### 5.3 아키텍처 절제 실험[1]

| 구성 요소 | FID CLIP |
|----------|---------|
| 희소 다운샘플 | 85.99 |
| 비선형 핵 | 24.49 |
| Quasi Monte Carlo | 7.63 |
| 정규 합성곱 | 5.63 |
| **∞-Diff (제안)** | **4.75** |

#### 5.4 추가 응용

**초해상도(Super-resolution)**: 저해상도 이미지 인코딩 후 고해상도에서 샘플링으로 세부정보 추가[1]

**인페인팅(Inpainting)**: 재구성 유도(reconstruction guidance) 사용:[1]

$$x_{t-1} \leftarrow x_{t-1} - \lambda \nabla_{x_t}\|m \odot (\tilde{\mu}_0(x_t, t) - T\bar{x})\|^2_2$$

***

### 6. 모델 일반화 성능 향상 (중점 분석)

#### 6.1 이산화 불변성(Discretisation Invariance)

**핵심 성과**: 임의의 해상도에서 샘플링 가능 (64×64에서 1024×1024)

이는 다음 메커니즘으로 달성:[1]

1. **무한차원 함수 표현**: 학습된 모델이 해상도에 독립적인 함수 공간에서 작동
2. **다양한 크기의 초기 잡음**: 각 해상도에서 서로 다른 크기의 잡음으로 초기화
3. **신경 연산자의 해상도 불변성**: 임의 좌표 $c$에서 평가 가능

**일반화 메커니즘**:

- 전통적 모델: 256×256에서 학습 → 512×512에서 성능 악화
- ∞-Diff: 256×256에서 학습 → 1024×1024에서도 **일관된 품질** 유지

#### 6.2 부표본화에서의 견고성

다양한 부표본화 비율에서의 학습:[1]

$$\text{학습 시 무작위 부표본화} \Rightarrow \text{다양한 표본화 패턴 대응}$$

결과:
- 1×: FID 3.15 (기준)
- 2×: FID 4.12 (±30% 성능 저하)
- 4×: FID 4.75 (±50% 성능 저하)
- **8×: FID 6.48 (±105% 성능 저하)** → 여전히 경쟁력 있음

#### 6.3 모델 예측 일반화

**이유**:

1. **공간 문맥 집계**: 신경 연산자가 전역 및 국소 정보를 통합 (신경 필드의 점별 독립성 문제 해결)
2. **Mollification의 정규화 효과**: 불규칙한 표본화 패턴에 의한 분산 감소
3. **다중 스케일 구조**: 다양한 주파수 성분 포착

***

### 7. 모델의 한계

논문이 명시한 한계:[1]

1. **Mollification의 필요성 불명확**: 데이터 유형에 따라 mollification 필요성이 달라질 수 있음
2. **역 mollification의 수치 불안정성**: 푸리에 역변환 기반 역변환은 Wiener 필터로 근사 필요
3. **부표본화 수준의 한계**: 8배 이상의 부표본화에서 품질 저하 가속
4. **매우 희소한 샘플에서의 성능**: 극도로 적은 좌표에서의 정수 연산자 근사 어려움

***

### 8. 2020년 이후 관련 최신 연구 비교 분석

#### 8.1 시간별 발전 추이

**2023년 초 - 이론적 기초 확립**

**Diffusion Generative Models in Infinite Dimensions** (Kerrigan et al., 2023)[2]
- Hilbert 공간의 가우시안 측도에 대한 엄밀한 이론적 기초 제시
- 잘-정의(well-posed) 무한차원 모델의 필요조건 규명
- ∞-Diff보다 이론적으로 더 깊지만 단순 데이터(MNIST, 가우시안 혼합)만 적용

**Score-based Diffusion Models in Function Space** (Lim et al., 2023)[3]
- 함수공간에서의 점수 기반 확산 모델 정의
- 무한차원 해에서 비용 함수의 유한성 조건 증명
- FNO 기반 아키텍처 사용으로 복잡한 데이터 처리 시도

**Infinite-Dimensional Diffusion Models** (Pidstrigach et al., 2023)[4]
- 변분 미적분학에 기반한 이론적 개발
- Bayesian 역문제 응용

#### 8.2 실제 응용 중심 연구 (2023-2024)

**∞-Brush** (2024.07)[5]
- ∞-Diff를 조건부 설정으로 확장
- 크로스-어텐션 신경 연산자로 함수공간 조건화
- 4096×4096 해상도의 제어 가능한 대규모 이미지 합성 달성
- ∞-Diff의 순수 생성에서 조건부 생성으로 확장

**Multilevel Diffusion** (Hagemann et al., 2023)[6]
- 다중 해상도 레벨에서 일관되게 이산화 가능한 확산 프로세스
- FNO 기반 아키텍처로 다중 레벨 학습
- ∞-Diff보다 더 강한 이론적 보증

**Diffusion Probabilistic Fields** (Zhuang et al., 2023)[1]
- ∞-Diff 동시 개발
- 작은 고정 좌표 부분 집합으로 문맥화
- 최대 64×64 해상도 (∞-Diff의 1024×1024과 비교해 제한적)

#### 8.3 고급 응용 및 이론 확장 (2024-2025)

**Infinite-Dimensional Diffusion Bridge Simulation** (2024.05)[7]
- 점수 매칭과 연산자 학습 결합
- 무한차원 조건부 확산 브릿지 학습
- 생물학적 형상 데이터 진화 모델링 (해상도 독립적)

**Conditional Score-Based Diffusion in Infinite Dimensions** (Baldassari et al., 2024)[8]
- 무한차원 Bayesian 선형 역문제 풀이
- 조건부 점수 함수의 특이성(blow-up) 문제 해결
- 이론적 수렴 분석

**Taming Score-Based Diffusion Priors for Infinite-Dimensional Nonlinear Inverse Problems** (2024.05)[9]
- 비선형 역문제에 대한 함수공간 샘플링
- 확산 모델 기반 사전(prior) + Langevin MCMC
- 수렴성 분석 제시

**Probability-Flow ODE in Infinite-Dimensional Function Spaces** (2025.03)[10]
- 무한차원에서의 확률 흐름 ODE 유도 (최초)
- 확산 기반 모델의 빠른 추론 가능

**DiffFNO** (2025 CVPR)[11]
- 가중 푸리에 신경 연산자(WFNO) 기반 초해상도
- 모드 재조정으로 중요 주파수 성분 포착
- 임의 크기 초해상도 달성

#### 8.4 비교 분석 요약

| 측면 | ∞-Diff | Kerrigan et al. | Lim et al. | Hagemann et al. | ∞-Brush |
|------|---------|-----------------|-----------|-----------------|---------|
| 이론적 엄밀성 | 중간 | 높음 | 높음 | 높음 | 중간 |
| 실제 해상도 | 1024×1024 | ~256×256 | ~512×512 | ~512×512 | 4096×4096 |
| 조건화 지원 | 미지원 | 미지원 | 미지원 | 미지원 | **지원** |
| 부표본화 | 8× | 미지원 | 미지원 | 미지원 | 4× |
| FID 성능 | 3.02-3.87 | N/A | ~5-6 | ~4-5 | 3.5-4.0 |

***

### 9. 논문이 앞으로의 연구에 미치는 영향

#### 9.1 이론적 영향

1. **무한차원 확산 모델의 실제 확장성 입증**
   - 이전 연구들은 주로 수학적 엄밀성에 중점
   - ∞-Diff는 **실무적 확장성** 입증으로 무한차원 모델의 실용성 확인

2. **신경 필드에서 신경 연산자로의 패러다임 전환**
   - 점별 함수(point-wise function) 한계 극복
   - 비국소 적분 연산자의 효과성 입증

3. **Mollification의 실용적 중요성 강조**
   - 불규칙 샘플링 안정화의 구체적 메커니즘 제시

#### 9.2 기술적 영향

1. **다중 스케일 하이브리드 아키텍처의 효율성**
   - 희소 신경 연산자 + 정규 격자 기반 처리의 조합
   - 후속 연구의 표준 아키텍처 패턴 확립

2. **희소 합성곱 라이브러리 활용**
   - TorchSparse 활용으로 효율성 향상 가능성 제시
   - 다른 무한차원 모델에도 즉시 응용 가능

3. **부표본화 기반 학습의 실용성**
   - 8배 부표본화에서도 경쟁력 있는 성능
   - GPU 메모리 제약 환경에서의 고해상도 학습 가능

#### 9.3 응용 분야 확장

1. **∞-Brush로의 직접 확장**: 조건부 생성으로 대규모 이미지 합성
2. **역문제 응용**: Bayesian 역문제의 함수공간 풀이 (후속 연구들이 적극 활용)
3. **다중 모달리티**: 음성, 3D 모델, 시계열 데이터로의 확장 가능성

***

### 10. 앞으로의 연구 시 고려할 점

#### 10.1 이론적 개선 방향

1. **Mollification의 적응형 선택**
   - 현재: 가우시안 mollifier 고정
   - 개선: 데이터 유형별 최적 mollification 커널 학습

2. **무한차원 보증(Infinite-dimensional guarantees)**
   - 현재: Monte Carlo 근사로 실무적 회피
   - 개선: Kerrigan et al.의 trace-class 조건 통합

3. **해상도 간 일반화의 이론적 분석**
   - 현재: 경험적 검증만 존재
   - 개선: 해상도 불변성의 충분조건 규명

#### 10.2 아키텍처 개선

1. **더 강력한 신경 연산자 개발**
   - 현재: 희소 합성곱, FNO, Galerkin 어텐션 등
   - 개선 방향:
     - **Transformer 기반 신경 연산자**: 장거리 의존성 포착
     - **적응형 핵**: 데이터 기반 동적 커널 조정
     - **혼합 피델리티(Multi-fidelity) 연산자**: 다양한 해상도 데이터 활용

2. **더 효율적인 희소 연산**
   - 현재: TorchSparse의 균일 격자 제약
   - 개선: 비정규 격자에서의 고효율 희소 연산

3. **다중 해상도 학습 통합**
   - Multilevel Diffusion의 이점 통합
   - 다양한 해상도에서의 동시 학습 최적화

#### 10.3 응용 확장

1. **조건부 생성의 확장**
   - ∞-Brush 이후 텍스트-이미지, 클래스 조건, 객체 조건 등 다양화
   - 크로스-모달(cross-modal) 생성 탐색

2. **과학 응용**
   - PDE 해의 생성 (현재 관심 증대)
   - 물리 제약 조건 통합 (Physics-informed)
   - Bayesian 역문제의 대규모 문제 응용

3. **고차원 데이터 처리**
   - 비디오 생성 (시간축 추가)
   - 3D 생성 (점 구름, 메시, NeRF)
   - 시계열 데이터 (금융, 기후, 센서)

#### 10.4 방법론적 고려사항

1. **계산 복잡도 분석**
   ```
   현재: O(N log N) (FNO 기반)
   개선: O(N) 알고리즘 탐색
   ```

2. **수치 안정성**
   - Mollification 역변환의 ill-conditioning 해결
   - Wiener 필터 대신 변분 접근법 탐색

3. **학습 효율성**
   - 현재: 확산 자동인코더 사용으로 분산 감소
   - 개선: 더 효율적인 분산 감소 기법 (예: 일관성 모델)

#### 10.5 실험적 고려사항

1. **더 큰 데이터셋 및 해상도 평가**
   - 현재: 256×256 주 학습, 1024×1024 추론
   - 목표: 4K/8K 해상도 학습 및 추론

2. **다양한 도메인 평가**
   - 얼굴, 장면 외 자연 이미지, 의료 영상, 과학 시뮬레이션

3. **사용자 연구**
   - FID만으로는 부족한 무한차원 모델의 실제 유용성 평가

***

## 결론

**∞-Diff**는 무한차원 확산 모델이 이론적 관심사에서 **실제 고해상도 생성의 실용적 도구**로 진화했음을 보여주는 중요한 이정표입니다. 신경 연산자 기반 접근, Mollified 확산, 다중 스케일 아키텍처의 조합은 이후 연구들(∞-Brush, DiffFNO 등)의 기초가 되었으며, 특히 **조건부 생성 및 고해상도 확장**을 가능하게 했습니다.

향후 연구는 다음 세 방향에 집중할 것으로 예상됩니다: **(1) 이론적 엄밀성 강화**, **(2) 아키텍처 효율성 개선**, **(3) 다양한 과학 및 산업 응용 확장**. 특히 무한차원 모델의 고유 장점인 **해상도 독립성**과 **원칙적 이산화 불변성**은 기후 모델링, 의료 영상, 극도 고해상도 이미지 생성 등 미래 응용에서 핵심 경쟁력이 될 것입니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2b2ad240-6162-4eec-bead-7618cef051f0/2303.18242v2.pdf)
[2](https://www.semanticscholar.org/paper/702b245734dfeecba1392141c537498c16f1c1fd)
[3](https://arxiv.org/abs/2302.07400)
[4](https://arxiv.org/abs/2302.10130)
[5](https://arxiv.org/abs/2407.14709)
[6](https://arxiv.org/abs/2303.04772)
[7](https://www.semanticscholar.org/paper/839d23e712ecf84894816aa9c921a8ab975aa39f)
[8](https://arxiv.org/abs/2305.19147)
[9](https://arxiv.org/abs/2405.15676)
[10](http://arxiv.org/pdf/2503.10219.pdf)
[11](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_DiffFNO_Diffusion_Fourier_Neural_Operator_CVPR_2025_paper.pdf)
[12](https://arxiv.org/abs/2402.01434)
[13](https://arxiv.org/abs/2411.01212)
[14](https://arxiv.org/html/2303.18242v2)
[15](http://arxiv.org/pdf/2303.04772.pdf)
[16](https://arxiv.org/abs/2212.00886)
[17](https://arxiv.org/pdf/2306.01984.pdf)
[18](http://arxiv.org/pdf/2408.11001.pdf)
[19](https://arxiv.org/html/2503.08643v1)
[20](https://pmc.ncbi.nlm.nih.gov/articles/PMC12201592/)
[21](https://vivekoommen.github.io/NO_DM/)
[22](https://arxiv.org/abs/2308.13295)
[23](https://royalsocietypublishing.org/doi/10.1098/rspa.2024.0819)
[24](https://proceedings.mlr.press/v151/dupont22a/dupont22a.pdf)
[25](http://www.arxiv.org/abs/2302.10130)
[26](https://proceedings.mlr.press/v202/zheng23d/zheng23d.pdf)
[27](https://yang-song.net/blog/2021/score/)
[28](https://www.semanticscholar.org/paper/Diffusion-Generative-Models-in-Infinite-Dimensions-Kerrigan-Ley/ac3fd54af29f4ae663d5a50992682df90ba57554)
[29](https://arxiv.org/html/2405.18353v1)
[30](https://arxiv.org/html/2506.03131v1)
[31](https://arxiv.org/pdf/2503.10219.pdf)
[32](https://arxiv.org/html/2506.16656v1)
[33](https://arxiv.org/html/2409.08477v1)
[34](https://www.semanticscholar.org/paper/aecdbdde7c437fc135dcd8c4ddae5ccb70e0e538)
