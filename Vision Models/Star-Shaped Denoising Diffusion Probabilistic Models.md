# Star-Shaped Denoising Diffusion Probabilistic Models

### 1. 논문의 핵심 주장과 주요 기여

**Star-Shaped Denoising Diffusion Probabilistic Models (SS-DDPM)**는 기존의 Gaussian 기반 DDPM을 지수족(exponential family) 분포로 일반화하는 획기적인 접근법을 제시한다. 이 논문의 핵심 주장은 다음과 같다:[1]

**핵심 기여**:
- **비-마르코프 포워드 프로세스**: 기존 DDPM의 마르코프 구조를 포기하고 각 단계에서 원본 데이터 $$x_0$$에만 직접 조건부화된 star-shaped 구조를 도입[1]
- **후진 포스터 계산 회피**: 포스터 분포 $$q(x_{t-1}|x_t, x_0)$$를 명시적으로 계산할 필요 없음[1]
- **지수족 일반화**: Beta, von Mises-Fisher, Dirichlet, Wishart 등 다양한 분포 지원[1]
- **충분 꼬리 통계**: Pitman-Koopman-Darmois 정리에 영감을 받은 효율적 인코딩 메커니즘[1]

### 2. 해결하고자 하는 문제, 제안 방법, 모델 구조

#### 2.1 문제 정의

기존 DDPM의 근본적 제약:
1. **분포 제한**: Gaussian 노이즈만 사용 가능. 일반화가 어려운 구조
2. **매니폴드 데이터 부자연스러움**: 구(sphere), 심플렉스(simplex), 대칭 양정부호 행렬(positive definite matrix) 등의 제약된 공간에서 Gaussian 노이즈는 구조를 파괴[1]
3. **계산 복잡성**: 다양한 분포에 대해 마르코프 후진 과정의 포스터를 각각 유도해야 함[1]

#### 2.2 제안 방법: Star-Shaped 확산 프로세스

**포워드 프로세스** - 비-마르코프 구조:

$$q^{SS}(x_{0:T}) = q(x_0) \prod_{t=1}^T q^{SS}(x_t|x_0)$$

여기서 모든 $$x_t$$가 $$x_0$$와 조건부 독립이며, 시간 단계 간 직접 종속성이 없다. 이는 전통적 마르코프 구조 $$q(x_(x_0) \prod_{t=1}^T q(x_t|x_{t-1})$$와 근본적으로 다르다.

**충분 꼬리 통계 (Sufficient Tail Statistic)**:

지수족의 선형 매개변수화 조건 하에서:[1]

$$\eta_t(x_0) = A_t f(x_0) + b_t$$

Theorem 1에 의해 다음 꼬리 통계가 충분하다:[1]

$$G_t = \sum_{s=t}^T A_s^T T(x_s)$$

이를 통해 후진 과정은 전체 꼬리 $$x_{t:T}$$ 대신 저차원 통계량 $$G_t$$에만 조건부화되면 된다:[1]

$$q^{SS}(x_{t-1}|x_{t:T}) = q^{SS}(x_{t-1}|G_t)$$

**후진 프로세스** - 일반화된 구조:

$$p^{SS}_\theta(x_{0:T}) = p^{SS}_\theta(x_T) \prod_{t=1}^T p^{SS}_\theta(x_{t-1}|x_{t:T})$$

각 단계에서 전체 꼬리 정보를 활용하며, 구체적으로는:[1]

$$p^{SS}_\theta(x_{t-1}|x_{t:T}) = q^{SS}(x_{t-1}|x_0)|_{x_0=x_\theta(G_t, t)}$$

#### 2.3 변분 하한 목적함수

$$L^{SS}(\theta) = \mathbb{E}_{q^{SS}} \left[ \log p_\theta(x_0|x_{1:T}) - \sum_{t=2}^T D_{KL}(q^{SS}(x_{t-1}|x_0) \| p_\theta^{SS}(x_{t-1}|x_{t:T})) \right]$$

이 목적함수는 **오직 주변 분포** $$q^{SS}(x_{t-1}|x_0)$$만 필요하며, 마르코프 후진 과정과 달리 전체 꼬리 정보를 활용[1].

#### 2.4 모델 구조의 특징

| 특성 | DDPM | SS-DDPM |
|------|------|---------|
| 포워드 구조 | 마르코프 | 비-마르코프 (star-shaped) |
| 조건부 의존성 | $$x_t \leftarrow x_{t-1}$$ | $$x_t \leftarrow x_0$$ |
| 후진 입력 | $$x_t$$ | $$x_{t:T}$$ 또는 $$G_t$$ |
| 포스터 계산 | 필수 | 불필요 |
| 지원 분포 | Gaussian (및 특수 경우) | 지수족 전체 |

#### 2.5 Gaussian 경우의 동등성

**Theorem 2**에 의해 특정 일정 함수 조건 하에서:[1]

$$\alpha^{SS}_t - \frac{1-\alpha^{SS}_t}{} = \frac{\alpha^{DDPM}_t}{1-\alpha^{DDPM}_t} - \frac{\alpha^{DDPM}_{t+1}}{1-\alpha^{DDPM}_{t+1}}$$

Gaussian SS-DDPM은 정확히 Gaussian DDPM과 동등하다. 이는 SS-DDPM이 진정한 일반화임을 증명한다.[1]

### 3. 성능 향상 및 한계

#### 3.1 성능 향상

**합성 데이터**:[1]
- **Dirichlet 분포**: KL 발산이 0.200 (DDPM)에서 0.011 (SS-DDPM)로 98% 감소
- **Wishart 분포**: 0.096에서 0.037로 61% 감소
- 정확한 분포 사용으로 매니폴드 구조 자동 보존

**이산 데이터** (text8):[1]
- Categorical SS-DDPM: NLL ≤ 1.69 bits/char
- Multinomial Text Diffusion: NLL ≤ 1.72 bits/char
- 동등 성능으로 통합 프레임워크 가능성 입증

**이미지 데이터** (CIFAR-10):[1]
- Beta SS-DDPM: FID = 3.17 (Improved DDPM과 동등)
- 낮은 샘플링 스텝(10-50)에서 DDIM보다 우수
- 높은 스텝(1000)에서 DDPM과 동등 성능

#### 3.2 일반화 성능 향상 메커니즘

**저차원 매니폴드 학습**:[2][3]
- 실제 고차원 이미지 데이터도 저차원 매니폴드에 위치
- SS-DDPM은 자동으로 이러한 기하학적 구조를 반영하는 노이징 분포 선택 가능[1]
- Beta 분포로 $$[0,1]$$ 범위 유지, Dirichlet로 심플렉스 유지[1]

**메모리화 vs 일반화 전환**:[3][2]
- 모델 용량과 데이터 크기의 비율에 따라 명확한 전환 발생
- SS-DDPM의 비-마르코프 구조는 저차원 분포 학습에 더 효율적[1]

**상호정보 매칭 기법**:[1]
- 포워드 프로세스에서 $$I(x_0; G_t) \approx I(x_0; x^{DDPM}_t)$$ 만족하도록 일정 함수 설정
- 이를 통해 Gaussian DDPM의 최적 일정을 다른 분포로 전이[1]

#### 3.3 한계점

**1. 샘플링 효율성**:[1]
- SS-DDPM은 DDIM처럼 샘플링 스텝 수를 줄일 수 없음
- 정확한 꼬리 통계 계산을 위해 고정된 $$T$$ 필요
- 근사적 방법 제시되었으나 추가 오류 발생 가능[1]

**2. 계산 오버헤드**:[1]
- 학습 중 $$x_{t:T}$$를 샘플링하고 $$G_t$$를 계산해야 함
- 병렬 처리로 해결되나 여전히 추가 연산 비용[1]

**3. 일정 함수 설정의 어려움**:[1]
- 선형 매개변수화 조건을 만족하는 분포만 사용 가능[1]
- Beta, Dirichlet 등 공통 분포는 지원하지만, 임의의 지수족 분포에는 제약[1]
- 상호정보 매칭은 휴리스틱이며 최적성 보장 없음[1]

**4. 이미지 생성에서 제한적 개선**:[1]
- Beta SS-DDPM이 DDPM과 동등 성능이지만 우월하지 않음
- 이미지가 실제로 Gaussian 노이징에 잘 맞는 데이터일 가능성[1]
- 특수 매니폴드 데이터에 진가 발휘[1]

**5. 일반화 이론의 미완성**:[3][1]
- 왜 비-마르코프 구조가 더 나은 일반화를 제공하는지 이론적 설명 부족
- 메모리화와 일반화의 전환점 메커니즘 불명확[3]

### 4. 모델의 일반화 성능 향상 가능성 심층 분석

#### 4.1 기하학적 귀납 편향 (Geometric Inductive Bias)

SS-DDPM의 일반화 개선은 **기하학적 귀납 편향** 메커니즘에서 비롯된다:[3][1]

**원리**: 데이터가 제약된 공간에 있을 때, 해당 공간에 특화된 분포를 사용하면:

$$D_{KL}(q^{SS}(x_t|x_0) \| p_\theta(x_t|x_0)) < D_{KL}(q^{Gaussian}(x_t|x_0) \| p_\theta(x_t|x_0))$$

**예시**:[1]
- Dirichlet: 심플렉스 데이터의 경계 구조 자동 유지
- Wishart: 양정부호 행렬의 스펙트럼 구조 보존  
- von Mises-Fisher: 구면 측지선 보존

#### 4.2 충분 통계 기반 차원 축소

Theorem 1의 충분 꼬리 통계는 **정보 손실 없이 차원 축소**:[1]

$$G_t \in \mathbb{R}^{d_\eta} \quad \text{vs} \quad x_{t:T} \in \mathbb{R}^{d \cdot (T-t+1)}$$

여기서 $$d_\eta \ll d \cdot (T-t+1)$$이므로:
- 신경망이 학습해야 할 차원 감소
- 데이터 효율성 향상[1]
- 과적합 위험 감소[1]

#### 4.3 마르코프 vs 비-마르코프 구조

**비-마르코프의 장점**:[3][1]

표준 DDPM에서 Appendix B에 보인 것처럼, 마르코프 후진 과정으로 근사하면:

$$L^{SS}_{Markov} = L^{SS}_* - \sum_{t=1}^T D_{KL}(q^{SS}(x_{t-1}|x_{t:T}) \| q^{SS}(x_{t-1}|x_t))$$

여기서 $D_{KL}$ 항은 **회복 불가능한 갭(irreducible gap)**이다.[1]

직관적으로:
- DDPM: $$x_t$$는 $$x_{t-1}$$을 포함하는 정보 흐름 (중첩)
- SS-DDPM: 각 $$x_t$$가 $$x_0$$의 독립적 정보 제공[1]

#### 4.4 상호정보 기반 일정 함수 최적화

배경: 포워드 프로세스의 소음 일정은 학습 곡선을 좌우한다.[1]

SS-DDPM의 혁신적 접근:[1]

$$\text{Find} \, \nu_t \, \text{such that} \, I(x_0; G_t(\nu_t)) = I(x_0; x^{DDPM}_t)$$

이를 통해:
- DDPM의 검증된 일정(cosine schedule) 활용[1]
- 새로운 분포도 최적 수렴 경로 보장[1]
- 데이터 기하학에 맞춘 동적 노이징[1]

#### 4.5 실험적 증거

**합성 데이터의 극적 개선**:[1]
- Dirichlet: KL 발산 98% 감소
- Wishart: 61% 감소
- DDPM과의 공정한 비교: 동일 아키텍처, 단 분포만 다름

**이미지의 조건부 우월성**:[1]
- 낮은 샘플링 스텝: Beta SS-DDPM > DDPM (Figure 6)
- DDIM보다 고스텝에서 일관되게 우수
- 시사점: 적응적 노이징이 샘플링 안정성 향상

### 5. 논문이 앞으로의 연구에 미치는 영향

#### 5.1 이론적 영향

**1. 확산 모델 일반화 이론의 재정의**:[3][1]

SS-DDPM의 성공은 다음 의문을 제기:[1]
- 왜 특정 분포가 특정 데이터에 더 우수한가?
- 기하학적 귀납 편향을 정량화할 수 있는가?[3]
- 메모리화 vs 일반화의 경계는 무엇인가?[3]

이는 최근 연구 에서 저차원 매니폴드 가설로 부분 해결되었다.[3]

**2. 지수족과 생성 모델의 심화된 연결**:

Pitman-Koopman-Darmois 정리의 생성 모델 맥락에서의 재발견:[1]
- 통계적 기초 강화
- 충분 통계의 기계학습 활용 개척[1]

**3. 비-마르코프 확산의 개념화**:[1]

기존 확산 모델은 암묵적으로 마르코프 가정:[1]
- SS-DDPM이 이를 명시적으로 문제화
- 비-마르코프 프로세스의 실제적 유용성 입증[1]

#### 5.2 방법론적 영향

**1. 매니폴드 데이터 생성 모델링**:[3][1]

바뀐 패러다임:
- 기존: Gaussian에서 시작, 사후 제약 적용
- 신규: 데이터 공간에 기반한 분포 직접 선택

영향을 받은 후속 연구:[3]
- Riemannian Score-based Generative Models (De Bortoli et al., 2022)
- Riemannian Flow Matching (Chen & Lipman, 2023)

**2. Flow Matching 프레임워크의 보완**:[4][5]

Flow Matching은 최적 운송 기반 확률 경로 제시:[5]
- SS-DDPM은 추가로 **분포 선택의 자유도** 제공[1]
- 조합 시 더욱 강력한 프레임워크 가능

**3. 이산 데이터 처리의 통합**:[1]

Categorical SS-DDPM이 D3PM과 동등 성능 제시:[1]
- 통합 프레임워크로의 진화 가능성
- 연속/이산 데이터의 일관된 처리[1]

#### 5.3 응용 분야의 확대

**1. 과학 데이터 생성 (Scientific Data Generation)**:[1]

- 분자 구조: Wishart/Riemannian 기반 생성[1]
- 단백질 설계: 매니폴드 제약 자동 만족[3]
- 화학 구조: Graph 매니폴드 위 생성[1]

**2. 데이터 증강 (Data Augmentation)**:[6][7]

SS-DDPM의 정확한 기하학적 보존은:[1]
- 도메인 특화 증강 가능
- 과적합 위험 감소[7]

**3. 역문제 해결 (Inverse Problems)**:[1]

기존: Gaussian 가정 기반 제약[1]
신규: 관찰 분포에 맞춘 확산 모델 설계[8]

### 6. 앞으로의 연구 시 고려할 점

#### 6.1 기술적 개선 방향

**1. 샘플링 효율성 개선**:[1]

현재 한계: DDIM 같은 가속화 불가[1]

개선 방안:
- 근사 꼬리 통계 이론 개발
- 적응적 스텝 선택 메커니즘
- 계층적 샘플링 전략 탐색

**2. 일정 함수의 자동 선택**:[1]

현재: 휴리스틱 기반 상호정보 매칭[1]

개선 방안:
- 메타-러닝으로 데이터별 최적 일정 학습
- 신경망 기반 일정 함수 매개변수화
- 이론적 최적성 조건 도출

**3. 계산 오버헤드 감소**:[1]

제안:
- 꼬리 통계의 확률적 근사
- 특화된 하드웨어 가속
- 병렬 샘플링 전략 개선

#### 6.2 이론적 심화 연구

**1. 일반화 경계 도출**:[3]

필요한 작업:
- SS-DDPM의 sample complexity 분석
- 기하학적 복잡도와 샘플 요구량의 관계[3]
- 메모리화-일반화 전환점의 명시적 특성화[3]

**2. 마르코프성 완화의 영향**:[3][1]

탐구 방향:
- 비-마르코프 구조의 표현력 증대 분석
- 정보-이론적 관점에서의 갭 정량화[1]
- 다른 비-마르코프 프로세스와의 비교[1]

**3. 다중 매니폴드 데이터**:[3][1]

확장 연구:
- 합성된 데이터 분포를 위한 혼합 분포 모델
- 적응적 분포 선택 메커니즘
- 계층적 매니폴드 구조 처리[3]

#### 6.3 관련 분야와의 통합

**1. Flow Matching과의 통합**:[9][4][5]

SS-DDPM + Flow Matching:
- 최적 운송 기반 확률 경로[5]
- 비-Gaussian 분포의 효율적 경로 설계[1]
- 더 빠른 학습과 샘플링 조합

**2. Score-based Generative Models과의 연결**:[10]

SDE 프레임워크 내에서:[10]
- SS-DDPM의 연속시간 확대
- 다양한 분포의 score matching 공식화
- 적응적 sampler 개발[10]

**3. 제약 조건이 있는 생성 모델**:[11][4]

응용 분야:
- 물리 법칙 제약 (PDE, 보존 법칙)[11]
- 화학 제약 (valence, aromaticity)[1]
- 기하학적 제약 (대칭성, 양성)[1]

#### 6.4 실험적 검증 로드맵

**Phase 1: 기초 매니폴드 (1년)**:
- Stiefel manifold (정규직교 행렬)
- Grassmannian (부분공간)
- Product of spheres (복합 각도)[1]

**Phase 2: 실제 응용 (2년)**:
- 단백질 구조 예측
- 그래프 구조 생성[1]
- 시계열 매니폴드 데이터

**Phase 3: 대규모 적용 (3년)**:
- 고차원 비전 문제
- 조건부 생성 확장
- 다중 모달 데이터[1]

#### 6.5 오픈 문제

**1. 분포 선택의 원리**:
- 주어진 데이터 분포에 최적의 노이징 분포는?
- 데이터 통계량으로부터 자동 추론 가능?

**2. 일반화 메커니즘**:
- 저차원 매니폴드 가설이 왜 작동하는가?[3]
- 비-마르코프성과 일반화의 정확한 관계는?

**3. 확장성**:
- 매우 고차원 데이터(예: 100M+ 차원)에서는?
- 매니폴드가 정확히 알려지지 않으면?

**4. 다중 대상(multi-objective)**:
- 여러 매니폴드 제약을 동시에 만족?
- 가우시안 혼합 같은 복잡 구조?

### 7. 결론

**Star-Shaped DDPM**은 단순히 기술적 확장이 아니라 **확산 모델의 철학적 재정의**를 제시한다. Gaussian 중심의 사고에서 벗어나 데이터의 기하학적 구조를 직접 반영하는 분포를 사용함으로써:

1. **이론적으로**: 지수족 통계 이론과 생성 모델의 심화된 연결 구현
2. **실무적으로**: 매니폴드 데이터에 대한 자연스러운 생성 모델링 제공
3. **미래적으로**: Flow Matching, Score-based 모델과의 통합 기반 마련

특히 **일반화 성능 향상**은 기하학적 귀납 편향 → 충분 통계 기반 차원 축소 → 저차원 학습의 연쇄로 설명되며, 이는 향후 확산 모델의 기본 설계 원칙으로 발전할 가능성이 높다.

2020년 이후 3년간(2023-2025)의 관련 연구 동향을 보면:
- **Flow Matching** 우상향 곡선 (최적 운송 통합)[9][5]
- **Riemannian 확산** 전문화 (구체 매니폴드)[3]
- **일반화 이론** 수렴 (저차원 구조 규명)[12][3]
- **제약 기반 생성** 확산 (물리 기반, 에너지 기반)[13][11]

이들과 SS-DDPM의 유기적 통합은 다음 세대 생성 모델의 핵심 과제가 될 것으로 예상된다.

***

### 참고 문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5cf3c266-6e49-4377-92a7-3c40622e4bf9/2302.05259v3.pdf)
[2](https://ieeexplore.ieee.org/document/10887063/)
[3](https://arxiv.org/abs/2503.13541)
[4](https://www.semanticscholar.org/paper/5c126ae3421f05768d8edd97ecd44b1364e2c99a)
[5](https://arxiv.org/abs/2210.02747)
[6](https://ieeexplore.ieee.org/document/11236414/)
[7](https://arxiv.org/html/2302.07944v3)
[8](https://arxiv.org/abs/2502.05994)
[9](https://openreview.net/forum?id=PqvMRDCJT9t)
[10](https://www.emergentmind.com/topics/score-based-generative-modeling)
[11](https://arxiv.org/abs/2506.04171)
[12](https://www.siam.org/publications/siam-news/articles/generalization-of-diffusion-models-principles-theory-and-implications/)
[13](https://arxiv.org/abs/2504.10612)
[14](https://www.mdpi.com/2073-4395/15/11/2648)
[15](https://ieeexplore.ieee.org/document/11045974/)
[16](https://arxiv.org/abs/2502.12089)
[17](https://ieeexplore.ieee.org/document/10892013/)
[18](https://ieeexplore.ieee.org/document/11099197/)
[19](https://arxiv.org/abs/2510.26231)
[20](http://arxiv.org/pdf/2412.17162.pdf)
[21](https://arxiv.org/pdf/2107.03006.pdf)
[22](https://arxiv.org/html/2411.19339v2)
[23](https://arxiv.org/pdf/2310.08337.pdf)
[24](https://arxiv.org/pdf/2305.14712.pdf)
[25](https://arxiv.org/pdf/2311.01797.pdf)
[26](https://arxiv.org/html/2501.02680v1)
[27](https://arxiv.org/html/2405.18782v1)
[28](https://iclr.cc/virtual/2025/session/31972)
[29](https://link.aps.org/doi/10.1103/PhysRevLett.128.168001)
[30](https://arxiv.org/pdf/2506.00849.pdf)
[31](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/ss-ddpm/)
[32](https://www.nature.com/articles/srep38782)
[33](https://oulurepo.oulu.fi/bitstream/handle/10024/56319/nbnfioulu-202505223830.pdf?sequence=1&isAllowed=y)
[34](https://liner.com/review/starshaped-denoising-diffusion-probabilistic-models)
[35](https://www.sciencedirect.com/science/article/abs/pii/S0952197625003124)
[36](https://arxiv.org/abs/2509.22623)
[37](https://arxiv.org/abs/2510.18072)
[38](https://arxiv.org/abs/2510.05930)
[39](https://www.semanticscholar.org/paper/2bfc6f0eaa67f4ded88580c71f940dbcfdc5e724)
[40](https://arxiv.org/abs/2506.10634)
[41](https://www.semanticscholar.org/paper/5396c55bee2a2abf2207e1cc5e5ae72c9edef9fa)
[42](https://arxiv.org/abs/2504.16262)
[43](https://arxiv.org/abs/2510.02300)
[44](https://arxiv.org/pdf/2311.13443.pdf)
[45](http://arxiv.org/pdf/2302.00482.pdf)
[46](http://arxiv.org/pdf/2210.02747.pdf)
[47](https://arxiv.org/html/2502.13406v1)
[48](https://arxiv.org/html/2412.16906v1)
[49](https://arxiv.org/pdf/2405.14664.pdf)
[50](https://arxiv.org/pdf/2402.03232.pdf)
[51](https://arxiv.org/html/2410.05292v1)
[52](https://www.emergentmind.com/topics/flow-matching-models)
[53](https://leeyngdo.github.io/blog/generative-model/2023-11-03-score-based-generative-models-with-sdes/)
[54](https://neurips.cc/virtual/2025/poster/118382)
[55](https://arxiv.org/abs/2311.09952)
[56](https://liner.com/ko/review/effective-data-augmentation-with-diffusion-models)
[57](https://neurips.cc/virtual/2024/tutorial/99531)
