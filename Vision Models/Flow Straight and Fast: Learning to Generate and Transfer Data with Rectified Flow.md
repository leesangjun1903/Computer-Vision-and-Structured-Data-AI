# Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow

### 1. 핵심 주장과 주요 기여

"Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" 논문은 **Rectified Flow (직선화 흐름)**이라는 간단하면서도 효과적인 접근법을 제시합니다. 이 방법의 핵심 주장은 신경 상미분방정식(Neural ODE)을 학습할 때, **두 분포 사이를 연결하는 경로가 가능한 한 직선에 가까워야 한다**는 것입니다.[1]

**주요 기여는 다음과 같습니다:**

1. **직선 경로 학습을 통한 통일된 프레임워크**: 생성 모델링과 도메인 전환이라는 서로 다른 작업을 하나의 프레임워크로 해결[1]

2. **확장 가능한 최소자승 최적화**: 표준 감독 학습을 넘어서는 추가 매개변수 없이 간단한 비선형 최소자승 문제로 해결[1]

3. **수송 비용 감소**: 단계적 직선화(reflow)를 통해 모든 볼록 비용 함수에 대해 수송 비용이 감소함을 이론적으로 증명[1]

4. **빠른 추론**: 직선 경로는 이산화 없이 정확히 시뮬레이션할 수 있어 단일 오일러 스텝으로도 고품질 결과 생성 가능[1]

### 2. 문제 정의 및 제안 방법

#### 핵심 문제: 수송 매핑 문제

논문이 해결하려는 문제는 다음과 같이 정의됩니다:[1]

**두 분포 $$X_0 \sim \pi_0$$, $$X_1 \sim \pi_1$$의 관찰값이 주어질 때, 수송 함수 $$T: \mathbb{R}^d \to \mathbb{R}^d$$를 찾아 $$Z_1 := T(Z_0) \sim \pi_1$$ (단, $$Z_0 \sim \pi_0$$)을 만족하도록 하는 것**

기존 방법들의 문제점:[1]
- **GAN**: 학습 불안정성과 모드 붕괴
- **VAE**: 계산 복잡성으로 인한 제약
- **확산 모델**: 추론 비용이 매우 높음 (수천 단계 필요)

#### 제안된 방법: Rectified Flow

**기본 아이디어**: 선형 보간 $$X_t = tX_1 + (1-t)X_0$$의 직선 경로를 따르도록 ODE 드리프트(drift) 함수를 학습[1]

**최적화 문제** (식 1):
$$\min_v \int_0^1 \mathbb{E}\left[\|(X_1 - X_0) - v(X_t, t)\|^2\right] dt,$$
여기서 $$X_t = tX_1 + (1-t)X_0$$[1]

**ODE 흐름**:
$$dZ_t = v(Z_t, t)dt$$[1]

#### 알고리즘 1: Rectified Flow 주요 단계[1]

1. **훈련**: 다음 목적함수 최소화

$$\hat{\theta} = \arg\min_{\theta} \mathbb{E}\left[\|X_1 - X_0 - v_\theta(tX_1 + (1-t)X_0, t)\|^2\right], \quad t \sim \text{Uniform}([0,1])$$

2. **샘플링**: $$Z_0 \sim \pi_0$$에서 시작하여 ODE 풀기
   - 정방향: $$dZ\_t = v_\theta(Z_t, t)dt$$
   - 역방향: $$d\tilde{X}\_t = -v_\theta(\tilde{X}_t, t)dt$$

3. **선택사항 - Reflow**: $$Z^{k+1} = \text{RectFlow}((Z^k_0, Z^k_1))$$로 재귀적으로 흐름 직선화

4. **선택사항 - Distillation**: 마지막 단계에서 흐름을 신경망 $$\hat{T}(z_0) = z_0 + v(z_0, 0)$$으로 증류[1]

### 3. 모델 구조와 이론적 기초

#### 최적 드리프트 속도 필드

문제 (1)의 정확한 최솟값은 다음과 같습니다:[1]

$$v_X(x, t) = \mathbb{E}[X_1 - X_0 | X_t = x]$$

조건부 밀도가 존재할 때, 이를 다음과 같이 표현할 수 있습니다:[1]

$$v_X(z, t) = \mathbb{E}\left[\frac{X_1 - z}{1-t}\eta_t(X_1, z)\right],$$

여기서 $$\eta_t(X_1, z) = \frac{\rho\left(\frac{z-tX_1}{1-t}\mid X_1\right)}{\mathbb{E}[\rho\left(\frac{z-tX_1}{1-t}\mid X_1\right)]}$$[1]

#### 핵심 이론 정리

**정리 3.3 - 한계 보존 성질**:[1]
직선화된 흐름 $$dZ_t = v_X(Z_t, t)dt$$는 모든 시간 $$t \in [0,1]$$에서 한계 법칙을 보존합니다:

$$\text{Law}(Z_t) = \text{Law}(X_t), \quad \forall t \in [0,1]$$

**증명 개요**: 연속 방정식(continuity equation)을 이용하여 $$Z_t$$와 $$X_t$$가 동일한 드리프트 속도장으로부터 생성되므로 동일한 한계 법칙을 가집니다.[1]

**정리 3.5 - 볼록 수송 비용 감소**:[1]
모든 볼록 함수 $$c: \mathbb{R}^d \to \mathbb{R}$$에 대해:

$$\mathbb{E}[c(Z_1 - Z_0)] \leq \mathbb{E}[c(X_1 - X_0)]$$

**증명**: Jensen의 부등식을 이용하면,

$$\mathbb{E}[c(Z_1 - Z_0)] = \mathbb{E}\left[c\left(\int_0^1 v_X(Z_t, t)dt\right)\right] \leq \mathbb{E}\left[\int_0^1 c(v_X(Z_t, t))dt\right]$$

$$\text{Law}(Z_t) = \text{Law}(X_t)$$이고 $$v_X(Z_t, t) = \mathbb{E}[X_1 - X_0 | X_t]$$이므로, 다시 한번 Jensen 부등식을 적용하면 원하는 결과를 얻습니다[1].

**정리 3.7 - 직선화 수렴**: $$k$$번째 직선화된 흐름 $$Z^k$$에 대해, 직선성(straightness) 척도

$$S(Z) = \int_0^1 \mathbb{E}\left[\|(Z_1 - Z_0) - \dot{Z}_t\|^2\right] dt$$

는 다음을 만족합니다:[1]

$$\min_{k \leq K} S(Z^k) = \mathcal{O}(1/K)$$

이는 반복적 직선화가 $$O(1/K)$$ 속도로 직선화를 진행함을 보여줍니다.[1]

#### 비선형 확장 및 기존 방법과의 연결

더 일반적인 보간 과정 $$X_t = \alpha_t X_1 + \beta_t X_0$$ (단, $$\alpha_0 = \beta_1 = 1$$, $$\alpha_1 = \beta_0 = 0$$)을 사용할 수 있습니다:[1]
$$\min_v \int_0^1 \mathbb{E}\left[w_t \|v(X_t, t) - \dot{X}_t\|^2\right] dt$$

**중요한 발견**: 확률 흐름 ODE(PF-ODE)와 DDIM은 모두 이 프레임워크의 특수한 경우로 볼 수 있습니다:[1]

- **VP ODE**: $$\alpha_t = \exp\left(-\frac{1}{4}a(1-t)^2 - \frac{1}{2}b(1-t)\right)$$, $$\beta_t = \sqrt{1 - \alpha_t^2}$$
- **Rectified Flow**: $$\alpha_t = t$$, $$\beta_t = 1-t$$ (직선 경로)

Rectified Flow가 우수한 이유:[1]
1. **직선 경로**: $$\alpha_t = t, \beta_t = 1-t$$로 인해 진정한 직선 궤적 생성
2. **균등 속도**: VP ODE와 달리 시간에 따라 균일한 속도 진행
3. **초기 분포 자유도**: 초기 분포 $$\pi_0$$을 임의로 선택 가능

### 4. 성능 향상 분석

#### 실험 결과 1: CIFAR-10 비조건부 이미지 생성[1]

| 방법 | NFE | FID ↓ | Recall ↑ | 비고 |
|------|-----|-------|----------|------|
| 1-Rectified Flow | 127 | 2.58 | 0.57 | 최고 성능 |
| 2-Rectified Flow | 110 | 3.36 | 0.54 | 함수 평가 감소 |
| VP ODE | 140 | 3.93 | 0.51 | - |
| sub-VP ODE | 146 | 3.16 | 0.55 | - |
| VP SDE | 2000 | 2.55 | 0.58 | 높은 계산 비용 |

**단계별 성능** (Figure 8a):[1]
- **1-Step (Distilled)**: FID 4.85, Recall 0.51 (기존 ODE/GAN 최고 성능)
- **2-Step**: FID 12.21 (Reflow 후)
- **3-Step**: FID 8.15

직선화 효과 검증 (Figure 9):[1]
- Reflow 후 straightness measure가 현저히 감소
- 학습 단계 증가에 따라 FID와 Recall 지속적 개선

#### 실험 결과 2: 이미지-이미지 변환[1]

논문은 고양이 ↔ 야생동물, MetFace ↔ CelebA 등의 도메인 간 전환을 시연합니다. 주요 특징:[1]
- N=1 (단일 스텝): 양호한 결과 달성
- N=100: 고품질 다양한 이미지 생성
- 이전 diffusion 기반 방법 대비 추론 속도 대폭 향상

#### 실험 결과 3: 도메인 적응[1]

| 데이터셋 | 기존 최고 | Rectified Flow |
|---------|---------|----------------|
| OfficeHome | 68.7% ± 0.3% | **69.2% ± 0.5%** |
| DomainNet | 41.5% ± 0.2% | 41.4% ± 0.1% |

도메인 표현 공간에서 Rectified Flow를 적용하여 도메인 시프트 완화.[1]

### 5. 일반화 성능 향상 가능성

#### 이론적 보장

**Straightness와 샘플링 오류의 관계**:[1]
직선 경로는 이산화 오류를 최소화합니다. 경로가 완벽하게 직선일 때:
$$Z_1 = Z_0 + v(Z_0, 0) \times 1$$
로 정확히 계산 가능합니다.[1]

**Reflow의 수렴성**: 정리 3.7에서 $$k$$ 번 Reflow 수행 시, 직선성이 $$O(1/k)$$ 속도로 개선되므로, 충분한 Reflow 후 단계별 오류가 거의 무시할 수 있는 수준이 됩니다.[1]

#### 일반화 개선 메커니즘

1. **Deterministic Coupling**: Rectified Flow는 확정적 결합(deterministic coupling)을 학습하므로, 노이즈 기반 방법보다 데이터 간 더 명확한 대응을 학습[1]

2. **모든 Convex Cost에 대한 Pareto 개선**: 특정 비용 함수에만 최적화되지 않고, 모든 볼록 비용에 대해 동시에 개선되므로 다양한 작업에 적응 가능[1]

3. **Flow Crossing 제거**: 선형 보간의 교차점을 제거함으로써 더 간결한 데이터 매니폴드 구조를 학습[1]

### 6. 한계 및 제약사항

#### 이론적 한계

1. **고차원에서의 최적성 부재**:[1]
   - 1차원에서는 직선 결합이 모든 볼록 비용에 대해 동시 최적
   - 고차원($$d \geq 2$$)에서는 특정 비용 함수에 대해 최적이 아닐 수 있음
   - 따라서 특정 작업에 맞춘 수정 필요

2. **Reflow의 오류 누적**:[1]
   - 과도한 Reflow(예: k>3)는 속도장 추정 오류가 누적될 수 있음
   - 실제로는 1-2회 Reflow 권장

3. **조건부 밀도 요구**:[1]
   - 속도장이 잘 정의되려면 조건부 밀도 $$\rho(x_0|x_1)$$이 존재해야 함
   - 조건부 밀도가 불연속인 경우, 가우시안 노이즈 추가 필요

#### 실용적 한계

1. **저차원 비모수 추정의 한계**:[1]
   - 커널 기반 추정(식 5): 대역폭 선택이 중요하며, 계산 복잡도 O(n²)
   - 실제로는 신경망 사용하되, 스무싱 정규화(L2 페널티) 필요

2. **특정 작업에서의 최적성 미확보**:[1]
   - 이차 비용 최적 수송을 위해서는 속도장을 gradient field로 제약해야 함
   - 회전 성분 제거로 인한 추가 복잡성

3. **추론 단계 감소의 한계**:[1]
   - 매우 큰 단계 크기(N<5)에서는 여전히 정확도 저하
   - Distillation 시에만 진정한 1-step 모델 달성

### 7. 앞으로의 연구 영향 및 고려사항

#### 최신 연구 기반 영향 분석 (2024-2025)

**1. Rectified Flow의 산업 적용 확대**[2][3]

최신 텍스트-이미지 생성 모델들이 Rectified Flow를 채택하고 있습니다. 예를 들어:[3]
- **FLUX 모델**: Rectified Flow 기반 Diffusion Transformer로 고해상도 이미지 생성에서 우수한 성능
- **OpenSora**: Rectified Flow와 Transformer 결합으로 비디오 생성 혁신
- 확산 모델 대비 **수십 배 빠른 추론 속도** 달성[3]

**2. 계층적 구조 확장 연구**[4]

2025년 3월 발표된 "Towards Hierarchical Rectified Flow" 논문에서:[4]
- 다중 ODE를 계층적으로 결합하는 구조 제시
- 교차하는 경로를 허용하여 고전적 Rectified Flow보다 **더 직선적인 궤적** 달성
- 신경 함수 평가(NFE) 감소로 추론 효율성 극대화

**3. Flow Matching과의 통합**[5]

"Diff2Flow" 프레임워크 (2025년 6월):[5]
- 사전 학습된 Diffusion 모델의 지식을 Flow Matching으로 효율적 전이
- 타임스텝 재조정, 보간 정렬, 속도장 변환을 통한 seamless 변환
- PEFT(Parameter-Efficient Finetuning) 환경에서도 높은 성능 유지
- **결론**: Rectified Flow 기반 메서드가 기존 Diffusion 생태계 통합의 핵심

**4. 시계열 생성 분야 확장**[2]

"FlowTS: Time Series Generation via Rectified Flow" (2025년 2월):[2]
- Rectified Flow의 직선 경로 특성을 시계열에 적용
- 고차원 시계열에서 확산 모델 대비 **효율성 향상**
- 적응형 배치 크기 조정으로 가변 길이 시계열 처리 개선

**5. 역변환 및 편집 기술 개발**[6]

"Taming Rectified Flow for Inversion and Editing" (2024년 11월):[6]
- RF-Solver: 고차 테일러 전개를 이용한 정확한 ODE 역변환
- FLUX, OpenSora 같은 최신 모델의 편집 능력 향상
- ODE 구간 내 정확한 위치 제어 가능

**6. 이론적 수렴성 분석 강화**[7]

"2-Rectifications are Enough for Straight Flows" (2025년 2월):[7]
- Wasserstein 거리 기반 수렴성 증명
- **2번의 reflow만으로도 충분히 직선적 흐름 달성**
- 계산 비용 대비 성능 최적 지점 제시

#### 앞으로 연구 시 고려사항

**1. 고차원 최적 수송과의 연계**

직선 결합이 특정 비용(특히 이차 비용)에 대해 최적이 아니라는 한계를 극복하기 위해:
- 제약된 최적화(gradient field 제약) 연구 진행 중
- Multi-scale optimal transport와의 통합 모색 필요

**2. 조건부 생성(Conditional Generation) 확장**

현재는 비조건부 생성에 중심이지만:
- 클래스/텍스트 조건 통합의 효율적 방법 개발 필요
- Cross-attention 메커니즘과의 최적 결합 전략 연구 필요

**3. Reflow 최적성 판정**

언제 Reflow를 멈춰야 하는지 자동 판정:
- Straightness 메트릭과 실제 성능 간의 정량적 관계 규명 필요
- Adaptive reflow 전략 개발 진행 중[8]

**4. 도메인 특화 응용**

각 도메인에 맞는 보간(interpolation) 전략:
- 의료 영상, 분자 구조 등에서 비유클리드 기하학 적용
- Manifold 기반 Rectified Flow 개발 필요

**5. 신경망 설계 최적화**

보다 효율적인 속도장 매개변수화:
- 구조화된 정규화 기법(현재는 L2 페널티만 사용)
- 계층적 구조와의 결합으로 추가 효율성 달성 가능[4]

**6. 다중 작업 통합 학습**

생성 + 전환 + 적응을 통일된 프레임워크에서:
- 다중 분포 쌍에 대한 공유 속도장 학습
- 메타 러닝 관점의 Reflow 적응 전략

***

Rectified Flow는 생성 모델 분야에서 **단순성, 이론적 우수성, 실무 효율성**의 완벽한 결합을 제시합니다. 직선 경로 학습이라는 명확한 목표와 $$O(1/K)$$ 수렴 보장은 이론 연구자들에게 매력적이며, 단일 스텝에 가까운 추론 속도는 산업 응용에 획기적입니다. 최근 2024-2025년 연구 동향을 보면, Rectified Flow는 FLUX, OpenSora 같은 최첨단 모델의 백본 기술로 자리 잡았으며, Flow Matching 통합, 시계열 확장, 계층적 개선 등 지속적인 혁신이 진행 중입니다. 향후 연구는 고차원 최적성, 조건부 생성 효율화, 도메인 특화 적용에 집중될 것으로으로 예상됩니다.[5][12][3][4][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/12a7958c-e568-4cc6-8533-ed7b97a6aedb/2209.03003v1.pdf)
[2](https://arxiv.org/html/2411.07506)
[3](http://arxiv.org/pdf/2403.03206.pdf)
[4](https://arxiv.org/html/2502.17436v2)
[5](https://www.themoonlight.io/ko/review/diff2flow-training-flow-matching-models-via-diffusion-model-alignment)
[6](https://arxiv.org/html/2411.04746)
[7](http://arxiv.org/pdf/2410.14949.pdf)
[8](http://arxiv.org/pdf/2405.20320v2.pdf)
[9](http://arxiv.org/pdf/2209.03003v1.pdf)
[10](https://arxiv.org/html/2406.03293)
[11](https://mvje.tistory.com/289)
[12](https://modulabs.co.kr/labs/265)
[13](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/rectified-flow/)
[14](https://blog.outta.ai/177)
[15](https://dmqa.korea.ac.kr/activity/seminar/486)
[16](https://ostin.tistory.com/225)
[17](https://stibee.com/api/v1.0/emails/share/nwWQtd_IFF_-LGnpU9_Oi_au5ACPsNM)
[18](https://bayesian-bacteria.tistory.com/4)
[19](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/stable-diffusion-3/)
