
# Improved Masked Image Generation with Token-Critic
## 요약
"Improved Masked Image Generation with Token-Critic"(Lezama et al., 2022)는 Google Research에서 발표한 논문으로, 비자동회귀 생성 트랜스포머(MaskGIT)의 샘플링 과정을 개선하기 위한 보조 모델인 **Token-Critic**을 제안한다. 이 접근법은 반복 샘플링 중에 생성된 시각 토큰의 타당성을 평가하는 이진 분류 트랜스포머를 도입하여, 토큰 간 상관관계를 포착하고 이전 결정의 수정을 가능하게 한다. 결과적으로 MaskGIT 기준 모델에 대해 ImageNet 256×256에서 FID를 6.56에서 4.69로 개선하고, 외부 분류기와 결합했을 때 최첨단 확산 모델과 경쟁할 수 있는 성능을 달성한다.

***

## 1. 핵심 주장과 주요 기여
### 1.1 해결하고자 하는 문제
MaskGIT는 비자동회귀 방식의 빠른 이미지 생성(8-16 단계)을 실현했으나, 토큰 선택 메커니즘에서 세 가지 근본적인 한계를 가진다:

1. **독립적 신뢰도 점수에 의존**: 생성기의 예측 신뢰도를 사용하여 재샘플링 토큰을 선택하지만, 이는 생성 오류에 민감하며 개별 토큰 수준의 결정으로 제한된다.

2. **토큰 상관관계 무시**: 각 토큰의 가수성(accept/reject) 결정이 독립적으로 이루어져, 이미지의 공간적·의미적 상호의존성을 포착하지 못한다.

3. **탐욕적 비가역 샘플링**: 반복 과정에서 이전에 수락된 토큰을 수정할 수 없어, 새로운 문맥에서 덜 타당해진 토큰을 고정시킨다.

수학적으로, MaskGIT는 마진 분포의 합에 최적화된다[식 1]:

$$L = \sum_{j=1}^{N} \sum_{k=1}^{K} q(x_j | o) \log p(x_j | o)$$

여기서 $q$는 실제 마진 분포이고, 이는 완전히 인수분해된 분포로 근사되어 결합 분포의 풍부한 정보를 상실한다.

### 1.2 핵심 기여
**Token-Critic 프레임워크**: 생성기가 출력한 토큰화된 이미지에 대해 이진 마스크를 예측하는 보조 트랜스포머를 훈련한다. 이는 세 가지 주요 이점을 제공한다:

1. **학습된 토큰 선택**: 생성기의 신뢰도가 아닌 Token-Critic의 예측에 기초한 마스킹으로, 실제 분포 하에서 토큰의 타당성을 학습할 수 있다.

2. **결합 분포 근사**: Token-Critic이 전체 토큰 집합을 집합적으로 평가하여 토큰 간 상관관계를 포착한다.

3. **재귀적 정정 메커니즘**: 이전의 높은 신뢰도 토큰도 새로운 문맥에서 재평가되어 수정될 수 있다.

***

## 2. 제안 방법론
### 2.1 Token-Critic 훈련
Token-Critic은 다음과 같이 훈련된다:

**입력**: 실제 토큰화 이미지 $x_0$, 임의 이진 마스크 $m^t$, 마스크된 이미지 $x^t = x_0 \circ m^t$

**프로세스**:
1. 생성기 $G_\phi$를 사용하여 마스크된 토큰 예측: $\hat{x}_0 \sim p(x_0|x^t, c)$
2. 마스크되지 않은 토큰은 원래대로 유지: $\tilde{x}_0 = \hat{x}_0 \circ m^t + x_0 \circ \overline{m^t}$
3. Token-Critic $\psi$가 입력 $\tilde{x}_0$에 대해 이진 마스크 $m^t$를 예측

**훈련 목표** [식 2]:

$$L = \mathbb{E}_{q(x_0,c,t,m^t)} \sum_{j=1}^{N} \text{BCE}(m_j^t, p_\psi(m_j^t | \tilde{x}_0, c))$$

여기서 $\text{BCE}$는 이진 교차 엔트로피 손실이고, 생성기의 매개변수는 고정된다.

**핵심 통찰**: Token-Critic은 생성기의 오류를 식별하도록 훈련되므로, 실제 분포과 일치하지 않는 토큰 구성을 감지할 수 있다.

### 2.2 Token-Critic을 이용한 샘플링
**반복 샘플링 프로세스** [알고리즘 2]:

초기화: $x^T = [\text{MASK}]^N$ (모든 토큰이 마스크됨)

각 단계 $t = T, \ldots, 1$ 에서:
1. 생성기로 완전한 이미지 예측: $\tilde{x}\_0 = G_\phi(x^t, c)$
2. Token-Critic 점수 계산 (선택 노이즈 추가): $p_i = p_\psi(m_{i}^{t-1} | \tilde{x}_0, c) + n^{(t)}$
3. 낮은 점수 토큰 마스킹: $R = \lfloor \gamma(t/T) \cdot N \rfloor$개의 가장 낮은 점수 토큰 선택
4. 다음 반복을 위한 이미지 구성: $x^{t-1} = \tilde{x}_0 \circ m^{t-1} + x^t \circ \overline{m^{t-1}}$

**이산 확산 과정으로의 해석** [식 3-5]:

다음 상태의 분포는 다음과 같이 근사된다:

$$p(x^{t-1} | x^t, c) \approx \mathbb{E}_{p(x_0|x^t,c)} p(x^{t-1} | x_0, c)$$

여기서 기댓값이 생성기의 단일 샘플로 근사되고, $p(x^{t-1}|x_0, c)$는 Token-Critic이 예측하는 마스크를 통해 결정된다.

### 2.3 모델 구조
| 구성 요소 | 사양 |
|-----------|------|
| 생성기 | 24개 레이어, 16개 헤드, 임베딩 차원 768 |
| Token-Critic | 20개 레이어, 12개 헤드, 임베딩 차원 768 |
| VQ 토크나이저 | 1024개 코드북 항목, 16배 압축 |
| 샘플링 단계 | 18 단계 |
| 선택 노이즈 스케줄 | $n^{(t)} = K \cdot u(t/T)$, $K=1.0$ |
| 온도 스케줄 | 선형 감소: $T_t = a - (a-b) \cdot t/T$ |

**트레이닝 설정**:
- 배치 크기: 256
- 옵티마이저: Adam ($\beta_1=0.9$, $\beta_2=0.96$)
- 드롭아웃: 0.1
- 에포크: 600
- 하드웨어: 8×8 TPU 배열

***

## 3. 성능 향상 및 실증 결과
### 3.1 정량적 평가
**ImageNet 256×256 결과**:
- MaskGIT 기준: FID=6.56, IS=203.6
- Token-Critic 개선: FID=4.69 (-28.4%), IS=174.5
- BigGAN-deep 대비: FID 28.5% 개선
- CDM 대비: FID 3.9% 악화 (약간 낮음), 품질-다양성 균형에서 우위 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b2cec53a-0d43-45a7-88fe-099826757db6/2209.04439v1.pdf)

**ImageNet 512×512 결과**:
- MaskGIT 기준: FID=8.48, IS=167.1
- Token-Critic 개선: FID=6.80 (-19.8%), IS=182.1 (+9.0%)
- 정밀도(Precision): 0.79→0.76 (약간 감소)
- 재현율(Recall): 0.48→0.53 (+10.4% 향상)

**외부 분류기와의 결합**:
- 분류기 기반 거부 샘플링(ResNet-50)을 적용했을 때:
  - 256×256: FID=3.75, IS=287.0 (최고의 IS 달성)
  - 512×512: FID=4.03, IS=305.2
  - StyleGAN-XL 대비: FID는 약간 높으나 IS에서 우위 (305.2 vs 225.6)

### 3.2 품질-다양성 트레이드오프
Token-Critic의 샘플링 온도와 선택 노이즈 조정을 통해 FID-IS 곡선을 탐색할 수 있다:

- **높은 온도/노이즈**: IS 향상 (다양성 증가), FID 악화 (품질 감소)
- **낮은 온도/노이즈**: FID 개선 (품질 향상), IS 감소 (다양성 감소)
- Token-Critic은 기준 MaskGIT에 비해 전체 곡선에서 우월한 성능 제공

### 3.3 VQ 이미지 정제 능력
생성된 이미지의 사후 정제 실험:
- 초기 FID: 6.56 (256×256), 8.48 (512×512)
- 60% 토큰 정제 후: 5.73 (-12.7%), 7.64 (-9.9%)
- IS 향상: 203.6→206.6 (256×256)

이는 Token-Critic이 단순히 샘플링 가이드뿐만 아니라 생성된 이미지의 사후 개선 도구로도 활용될 수 있음을 보여준다.

***

## 4. 일반화 성능 향상 가능성
### 4.1 일반화 메커니즘
Token-Critic의 일반화 성능 향상은 세 가지 근본 원리에 기반한다:

**1. 결합 분포 학습의 강화**:

MaskGIT의 마진 분포 최적화 [식 1]는 다음과 같이 근사되는 KL 발산을 최소화한다:

$$KL(q_{data}||p_{factorized}) \approx \sum_j KL(q(x_j|o)||p(x_j|o))$$

반면 Token-Critic은 부분 마스크된 실제 이미지의 분포 $q(x^t)$와 샘플링된 이미지 분포 $p_{\psi,\phi}(x^t)$ 사이의 KL 발산을 최소화한다[부록 식 6-10]:

$$KL(q(x^t) || p_{\psi,\phi}(x^t)) \leq \mathbb{E}_{q(x^t)}\mathbb{E}_{p(x_0|x^t)} \log p_\psi(m^{t-1}|x_0, c)$$

이는 전체 토큰 구성의 타당성을 평가하므로, 토큰 간의 종속성을 자동으로 포착한다.

**2. 공간적·의미적 상관관계 포착**:

Token-Critic의 트랜스포머 아키텍처는 자기 주의(self-attention)를 통해 장거리 의존성을 모델링할 수 있다:

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

이를 통해 생성기가 독립적으로 예측하지 못하는 공간적 일관성(예: 경계 연속성, 텍스처 균일성)을 평가할 수 있다.

**3. 비탐욕적 정정**:

이전 결정을 수정할 수 있는 능력은 다음과 같은 상황에서 중요하다:
- 초기 반복에서 높은 신뢰도로 샘플링된 토큰이 나중의 문맥에서 부조화적일 수 있음
- Token-Critic의 전역 평가가 이러한 불일치를 감지하고 재샘플링을 유도

### 4.2 세분화된 분석
**정밀도 vs 재현율 트레이드오프**:

| 메트릭 | MaskGIT | Token-Critic | 변화 |
|--------|---------|--------------|------|
| 정밀도(256) | 0.79 | 0.76 | -3.8% |
| 재현율(256) | 0.48 | 0.53 | +10.4% |
| 정밀도(512) | 0.78 | 0.73 | -6.4% |
| 재현율(512) | 0.46 | 0.50 | +8.7% |

**해석**: Token-Critic은 생성 다양성을 우선하는 경향을 보인다. 이는 Token-Critic이 보다 보수적인 토큰 수락 전략을 채택하여, 불확실한 영역에서 더 많은 재샘플링을 유도함을 시사한다. 이는 동일한 생성기에 대해 더 나은 다양성을 제공하지만, 단일 모드 정확도에서는 약간의 손실이 있을 수 있다.

### 4.3 문제점과 한계
**1. 계산 비용 증가**:
- Token-Critic을 사용하면 각 반복 단계에서 두 개의 트랜스포머(생성기, Critic) 순전파 필요
- 기준 MaskGIT 대비 약 2배의 계산량 증가
- 실제 벽 시간(wall-time)에서 18 단계 Token-Critic이 36 단계 MaskGIT과 유사

**2. 일반화 경계**:
- 모두 256×256 및 512×512 ImageNet 클래스 조건 생성에서만 평가됨
- 다른 도메인(자연 이미지, 텍스트-이미지, 의료 이미지)에서의 성능 미검증
- 고분해능 생성(1024×1024 이상)에서의 확장성 불명

**3. 토크나이저 의존성**:
- VQ 토크나이저의 오류가 Token-Critic의 평가에 영향
- 토크나이저 편향이 증폭될 수 있음

***

## 5. 2020년 이후 관련 최신 연구 비교 분석
### 5.1 마스크된 이미지 생성 관련 연구
| 연구 | 연도 | 주요 방법 | 핵심 차이점 | 성능 |
|------|------|----------|-----------|------|
| **MaskGIT** | 2022 | 양방향 마스크 예측 | 생성기 신뢰도 기반 토큰 선택 | FID 6.56 (256) |
| **Token-Critic** | 2022 | 보조 분류기 기반 선택 | 결합 분포 학습, 상관관계 포착 | FID 4.69 (256) |
| **AutoNAT** | 2024 | 자동 전략 최적화 | 훈련/생성 전략 자동 설계 | FID 경쟁력 있음 |
| **eMIGM** | 2025 | 통합 마스크/확산 프레임워크 | 마스크 이미지 모델과 확산 통합 | FID 1.57-1.92 (512) |
| **ViewMask-1-to-3** | 2025 | 다중 뷰 일관성 | 이산 확산으로 다중 뷰 생성 | 기하학적 일관성 향상 |

### 5.2 이산 확산 모델 발전
**SEDD (Score Entropy Discrete Diffusion)** :
- 점수 엔트로피 손실을 통한 이산 확산 모델
- 텍스트 생성에서 우수한 성능
- Token-Critic보다 이론적 기초가 더 명확

**CoM-DAD (Coupled Manifold Discrete Absorbing Diffusion)** :
- 멀티모달 생성(텍스트-이미지) 통합
- 의미 평면과 토큰 수준 생성 분리
- Token-Critic의 개념을 확장하여 다중 모달리티에 적용

**Glauber Generative Model (GGM)** :
- 이산 마르코프 체인(열욕 동역학)을 사용한 확산
- 신호/노이즈 이진 분류 프레임워크
- Token-Critic의 이진 분류 아이디어와 유사하나 이론적 근거 다름

### 5.3 보조 모델 기반 샘플링 정제
**Discriminator Guidance** :
- 사전 훈련된 확산 모델의 샘플링 정제
- 판별기를 이용한 점수 보정
- Token-Critic과 유사한 보조 모델 개념이나 연속 공간에서 작동

**RealSRT (Real-World Super-Resolution with Token-Critic 영감)** :
- Token-Critic 개념을 초해상도 작업에 적용
- 생성기와 신뢰도 예측 모델의 조합
- Token-Critic의 직접적인 응용 연구

### 5.4 효율성 개선 관련
**Fast Solvers for Discrete Diffusion** :
- 고차 알고리즘으로 이산 확산 샘플링 가속
- τ-leaping 및 Runge-Kutta 방법 활용
- Token-Critic의 18 단계 비용을 더 줄일 가능성

**Halton Scheduler for MaskGIT** :
- Halton 수열을 사용한 향상된 스케줄링
- MaskGIT의 반복 전략 개선
- Token-Critic과 조합 가능한 직교 기술

### 5.5 멀티모달 확대
**MAETok (Masked Autoencoders as Tokenizers)** :
- 마스크 모델링을 통한 토크나이저 개선
- VQ 토크나이저의 한계 극복
- Token-Critic의 기반이 되는 VQ 공간 개선

**Unified Multimodal Discrete Diffusion** :
- 이산 확산을 이미지와 텍스트에 통합
- Token-Critic 개념의 멀티모달 확장
- 양방향 생성 능력

***

## 6. 모델의 한계와 개선 방향
### 6.1 현재의 한계
**1. 도메인 특이성**:
- ImageNet 클래스 조건 생성에만 검증
- 자유형 텍스트-이미지 생성, 의료 이미지, 비자연적 데이터에서의 성능 미지

**2. 계산 복잡도**:
- 생성기 + Token-Critic의 이중 순전파로 약 2배 계산량
- 모바일/엣지 장치에서의 적용 어려움

**3. 이론적 완성도**:
- 이산 확산 프로세스와의 엄밀한 수학적 연결 부족
- KL 발산 최소화와의 직접적 인과관계 미명확

**4. 토크나이저 의존**:
- VQ-VAE의 양자화 오류가 Token-Critic 성능 제약
- 최신 토크나이저(VQ-GAN, FSQ) 호환성 검증 필요

### 6.2 미래 연구 방향
**1. 적응형 Token-Critic**:
- 데이터 특성에 따라 Token-Critic 복잡도 조정
- 동적 레이어 수, 헤드 수 선택 메커니즘

**2. 멀티모달 확장**:
- 텍스트-이미지, 이미지-3D, 이미지-오디오 생성으로 확대
- 각 모달리티 간 상관관계 모델링

**3. 이론적 분석**:
- Token-Critic과 이산 확산 프로세스의 정형적 연결
- 샘플링 오류 경계의 이론적 도출

**4. 하이브리드 방식**:
- Token-Critic과 온도 스케줄링의 최적 조합
- 생성기와 Critic의 공동 훈련 가능성 탐색

***

## 7. 앞으로의 연구에 미치는 영향과 고려사항
### 7.1 학문적 영향
**1. 비자동회귀 생성 모델의 재평가**:
- Token-Critic은 비자동회귀 모델이 적절한 보조 모델로 최첨단 성능에 도달할 수 있음을 입증
- 이는 생성 모델링의 구조적 선택이 절대적이지 않음을 시사

**2. 보조 모델 개념의 일반화**:
- 생성기의 오류를 식별하는 보조 모델 개념이 다양한 생성 작업에 적용 가능
- 확산 모델, GAN, 자동회귀 모델 등에서 유사 기법 개발의 시발점

**3. 이산 대 연속 생성 모델의 수렴**:
- 이산 마스크 기반 생성과 연속 확산 프로세스의 개념적 연결 강화
- 후속 연구들(SEDD, CoM-DAD 등)에서 명시적으로 이 관점 채택

### 7.2 실제 응용에서의 고려사항
**1. 배포 최적화**:
- 18단계 × 2 순전파의 계산 비용은 실시간 응용에 도전
- 지식 증류(Knowledge Distillation)를 통한 경량 Token-Critic 개발 필요
- 캐싱 전략을 통한 중복 계산 제거

**2. 해상도 확장**:
- 512×512 이상의 고분해능 생성에서의 메모리/계산 요구사항 증가
- 계층적 생성(저해상도→고해상도)과의 결합 탐색

**3. 도메인 적응**:
- 사전 훈련된 Token-Critic을 새로운 도메인에 미세 조정하는 방법론
- 도메인 외 일반화 성능의 실증적 평가

### 7.3 경쟁 기술과의 비교 운영
최신 연구 트렌드(2024-2025)에서:

**vs. 최신 확산 모델** (EDM2, REPA):
- Token-Critic+분류기: FID 3.75-4.03 (256-512)
- 최신 확산: FID 1.57-1.92 (512)
- **개선 필요 영역**: 절대 성능 격차 여전히 존재, 하지만 효율성 우위

**vs. 최신 비자동회귀 모델** (eMIGM, VAR):
- eMIGM: FID 1.57 (512), 180 NFE (함수 평가)
- Token-Critic: FID 6.80 (512), 36 순전파 (2×18)
- **개선 필요**: 더 효율적인 마스킹 스케줄, 다중 스케일 생성

**vs. GAN** (StyleGAN-XL):
- StyleGAN-XL: FID 3.58 (512), 단일 순전파
- Token-Critic+분류기: FID 4.03 (512)
- **강점**: 다양성(IS 305.2 vs 219.8), 조건부 제어 용이성

### 7.4 미래 연구 우선순위
**고우선순위**:
1. **계산 효율 개선**: 18 단계 감소, 더 작은 Critic 모델, 동적 스케줄링
2. **다중 해상도 지원**: 계층적 생성 프레임워크 개발
3. **멀티모달 확장**: 텍스트 조건, 3D 생성 등으로 검증

**중우선순위**:
4. 이론적 기초 강화 (KL 발산, 수렴 보증)
5. 도메인 특화 Token-Critic 개발
6. 지식 증류를 통한 가벼운 모델

**저우선순위**:
7. 극도로 미세한 아키텍처 최적화
8. 특정 하드웨어에 대한 최적화

***

## 결론
**"Improved Masked Image Generation with Token-Critic"**은 비자동회귀 생성 모델의 샘플링 과정을 개선하기 위한 실질적이고 영향력 있는 접근법을 제시했다. Token-Critic이라는 보조 분류기 트랜스포머를 통해 토큰 간 상관관계를 모델링하고 이전 결정의 수정을 가능하게 함으로써, MaskGIT의 FID를 28% 개선했다.

이 연구는 세 가지 측면에서 중요한 기여를 한다:

1. **기술적 기여**: 마진 분포 최적화에서 결합 분포 근사로의 전환, 보조 모델 기반 샘플링 정제 개념 도입

2. **실증적 강화**: ImageNet 클래스 조건 생성에서 경쟁력 있는 성능 달성, 외부 분류기와의 보완 가능성 입증

3. **이론적 관점**: 이산 확산 프로세스와의 연결, 역 샘플링에서의 기하학적 해석 제공

2024-2025년의 최신 연구들(eMIGM, AutoNAT, CoM-DAD 등)은 Token-Critic이 제시한 보조 모델 개념을 확장하여, 더욱 효율적인 마스킹 전략, 멀티모달 생성, 이론적으로 더 견고한 확산 프로세스 개발로 진화하고 있다. 향후 연구는 **계산 효율**, **다중 도메인 일반화**, **멀티모달 확장**에 초점을 맞춰야 하며, Token-Critic의 핵심 아이디어인 "보조 분류기를 통한 적응형 샘플링 정제"는 생성 모델링의 여러 패러다임에서 계속 영향을 미칠 것으로 예상된다.

***

## 참고 문헌
<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_6][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_7][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_8][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_9][^1_90][^1_91][^1_92]</span>

<div align="center">⁂</div>

[^1_1]: 2209.04439v1.pdf

[^1_2]: https://www.mdpi.com/2075-4418/16/2/211

[^1_3]: https://www.semanticscholar.org/paper/696446b22fbceea5b3d6c3c7ef4eeb4ddb6d911f

[^1_4]: https://www.semanticscholar.org/paper/68ccde7f12cdf6e0547dffeadb45068f895e1942

[^1_5]: https://www.semanticscholar.org/paper/bf1da5e10f1c6369dd31950e359637d7ddb2f0d4

[^1_6]: https://www.semanticscholar.org/paper/f56fe8d53f4cb2cda5e698d5fadc339e28fb5d08

[^1_7]: https://www.mrforum.com/product/9781644900574-42

[^1_8]: https://www.semanticscholar.org/paper/020fc1eebe92dfe7893cf85455f7c38363024b7e

[^1_9]: https://arxiv.org/abs/2503.07197

[^1_10]: https://www.semanticscholar.org/paper/e0a0fadd037941b87a6052102eed3b5aa22fb507

[^1_11]: https://arxiv.org/abs/2502.03444

[^1_12]: http://arxiv.org/pdf/2503.06748.pdf

[^1_13]: https://arxiv.org/html/2501.09008v1

[^1_14]: https://arxiv.org/html/2501.00944v1

[^1_15]: https://arxiv.org/pdf/2403.17004.pdf

[^1_16]: https://arxiv.org/pdf/2410.00483.pdf

[^1_17]: https://arxiv.org/pdf/2406.04329.pdf

[^1_18]: https://arxiv.org/html/2401.07709v2

[^1_19]: http://arxiv.org/pdf/2502.11663.pdf

[^1_20]: https://arxiv.org/pdf/2503.07197.pdf

[^1_21]: https://openaccess.thecvf.com/content/CVPR2024/papers/Ni_Revisiting_Non-Autoregressive_Transformers_for_Efficient_Image_Synthesis_CVPR_2024_paper.pdf

[^1_22]: https://arxiv.org/abs/2202.04200

[^1_23]: https://www.emergentmind.com/topics/masked-diffusion-models

[^1_24]: https://www.themoonlight.io/en/review/revisiting-non-autoregressive-transformers-for-efficient-image-synthesis

[^1_25]: https://www.emergentmind.com/topics/maskgit-implementation

[^1_26]: https://openaccess.thecvf.com/content/CVPR2024/html/Ni_Revisiting_Non-Autoregressive_Transformers_for_Efficient_Image_Synthesis_CVPR_2024_paper.html

[^1_27]: https://openaccess.thecvf.com/content/CVPR2022/papers/Chang_MaskGIT_Masked_Generative_Image_Transformer_CVPR_2022_paper.pdf

[^1_28]: https://liner.com/review/revisiting-nonautoregressive-transformers-for-efficient-image-synthesis

[^1_29]: https://www.reddit.com/r/deeplearning/comments/svse4v/improved_vqgan_explained_maskgit_masked/

[^1_30]: https://kimjy99.github.io/논문리뷰/mdt/

[^1_31]: https://arxiv.org/abs/2303.00750

[^1_32]: https://kimjy99.github.io/논문리뷰/maskgit/

[^1_33]: https://pdfs.semanticscholar.org/2595/ab9b8eb3a7d25d55877acb42f58dfd2d190e.pdf

[^1_34]: https://arxiv.org/html/2410.07836v6

[^1_35]: https://pdfs.semanticscholar.org/7585/d5edd49c3b1dd5483201bb51e36e51005ea9.pdf

[^1_36]: https://arxiv.org/pdf/2312.14988.pdf

[^1_37]: https://openaccess.thecvf.com/content/CVPR2022/supplemental/Chang_MaskGIT_Masked_Generative_CVPR_2022_supplemental.pdf

[^1_38]: https://arxiv.org/html/2407.17877v1

[^1_39]: https://www.semanticscholar.org/paper/Revisiting-Non-Autoregressive-Transformers-for-Ni-Wang/bea8541268e34fbd550a390d2bce242f768d96b7

[^1_40]: https://www.arxiv.org/pdf/2510.04525.pdf

[^1_41]: https://pdfs.semanticscholar.org/1380/1890f74f67f53645ab41ad371c27cd99e028.pdf

[^1_42]: https://openaccess.thecvf.com/content/WACV2023/papers/Yim_Style-Guided_Inference_of_Transformer_for_High-Resolution_Image_Synthesis_WACV_2023_paper.pdf

[^1_43]: https://pdfs.semanticscholar.org/df0f/a076b5cedbe21efde544f401d8e6ee4d1662.pdf

[^1_44]: https://arxiv.org/html/2503.17076v1

[^1_45]: https://arxiv.org/abs/2405.17889

[^1_46]: https://arxiv.org/abs/2412.10193

[^1_47]: https://arxiv.org/abs/2410.07761

[^1_48]: https://ieeexplore.ieee.org/document/10734713/

[^1_49]: https://arxiv.org/abs/2404.10763

[^1_50]: https://arxiv.org/abs/2405.17035

[^1_51]: https://arxiv.org/abs/2401.05252

[^1_52]: https://www.semanticscholar.org/paper/9c85e6e0f58b480801fe6f1fa09305e2b9c46331

[^1_53]: https://ieeexplore.ieee.org/document/11093240/

[^1_54]: https://ieeexplore.ieee.org/document/10657216/

[^1_55]: https://arxiv.org/html/2412.15032v1

[^1_56]: https://arxiv.org/pdf/2502.00234.pdf

[^1_57]: https://arxiv.org/html/2503.20853

[^1_58]: https://arxiv.org/pdf/2107.03006.pdf

[^1_59]: http://arxiv.org/pdf/2409.19589.pdf

[^1_60]: https://arxiv.org/html/2410.14710v1

[^1_61]: https://dl.acm.org/doi/pdf/10.1145/3618342

[^1_62]: https://arxiv.org/pdf/2211.01324.pdf

[^1_63]: https://cs231n.stanford.edu/2024/papers/discrete-diffusion-for-image-generation.pdf

[^1_64]: https://dl.acm.org/doi/10.5555/3618408.3619087

[^1_65]: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830070.pdf

[^1_66]: https://proceedings.iclr.cc/paper_files/paper/2025/file/6a9305d8e1dc254308a2c2e918108007-Paper-Conference.pdf

[^1_67]: http://aai.kaist.ac.kr/bbs/board.php?bo_table=sub5_1\&wr_id=42

[^1_68]: https://openaccess.thecvf.com/content/WACV2025/papers/Wu_Patch_Ranking_Token_Pruning_as_Ranking_Prediction_for_Efficient_CLIP_WACV_2025_paper.pdf

[^1_69]: https://zhengkw18.github.io/blog/2024/mdm/

[^1_70]: https://pure.kaist.ac.kr/en/publications/refining-generative-process-with-discriminator-guidance-in-score-/

[^1_71]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08254.pdf

[^1_72]: https://www.emergentmind.com/topics/discrete-diffusion-model

[^1_73]: https://proceedings.neurips.cc/paper_files/paper/2023/file/4c5722bad9759216474df8fc46c97af2-Paper-Conference.pdf

[^1_74]: https://arxiv.org/html/2510.09012v2

[^1_75]: https://www.youtube.com/watch?v=7P_G_DSNCe4

[^1_76]: https://arxiv.org/abs/2211.17091

[^1_77]: https://openreview.net/forum?id=Q3g5JFnyCb

[^1_78]: https://arxiv.org/html/2501.00289v2

[^1_79]: https://arxiv.org/html/2601.15286v1

[^1_80]: https://arxiv.org/pdf/2508.01603.pdf

[^1_81]: https://arxiv.org/html/2510.01047v1

[^1_82]: https://arxiv.org/html/2501.10928v2

[^1_83]: https://arxiv.org/html/2509.12046v1

[^1_84]: https://arxiv.org/html/2503.20853v1

[^1_85]: https://www.arxiv.org/pdf/2106.00792v1.pdf

[^1_86]: https://arxiv.org/html/2510.09012v1

[^1_87]: https://arxiv.org/html/2407.21243v5

[^1_88]: https://arxiv.org/html/2502.16446v1

[^1_89]: https://arxiv.org/html/2511.14751v1

[^1_90]: https://arxiv.org/html/2505.22524v3

[^1_91]: https://arxiv.org/html/2503.09662v1

[^1_92]: https://arxiv.org/html/2508.01603v2
