# DiffCollage: Parallel Generation of Large Content with Diffusion Models

## 1. 핵심 주장 및 주요 기여

### 1.1 핵심 주장
**DiffCollage**는 **합성적 확산 모델(Compositional Diffusion Model)**로서, 작은 조각들의 생성에 학습된 확산 모델들을 활용하여 **대규모 콘텐츠를 병렬로 생성**할 수 있는 혁신적인 방법이다. 

기존의 자동회귀(Autoregressive) 방식의 문제점을 해결한다:
- 순차 생성으로 인한 오류 누적 제거
- 후행 조각이 선행 조각에 영향을 미칠 수 없는 구조적 한계 극복
- 루프/순환 구조 데이터(360도 이미지, 반복 모션)의 생성 가능

### 1.2 주요 기여
1. **DiffCollage 알고리즘**: 인수 그래프(Factor Graph) 기반 확률 모델로 대규모 콘텐츠 병렬 생성
2. **사전 학습된 모델 활용**: 기존 작은 조각 기반 확산 모델을 그대로 활용 가능
3. **다양한 작업 검증**: 무한 이미지 생성, 파노라마 이미지, 장시간 텍스트-동작 생성, 360도 이미지 등에서 뛰어난 성능 입증

---

## 2. 해결 문제, 제안 방법, 성능

### 2.1 문제 정의

#### 배경
대규모 데이터셋(360도 파노라마, 극단적 종횡비 이미지)이 존재하지 않거나 수집이 불가능한 경우, 작은 조각(정상적 크기 이미지)으로는 학습할 수 있지만 대규모 콘텐츠 생성이 어렵다.

#### 기존 방식의 문제점
자동회귀 아웃페인팅(Autoregressive Outpainting):
1. **순차성**: 조각이 순서대로 생성되어 후행 조각이 선행 조각에 영향 불가능
2. **오류 누적**: 훈련 시 실제 데이터 조건화, 테스트 시 자신의 예측 조건화로 인한 불일치
3. **계산 비효율**: 생성 시간이 콘텐츠 크기에 선형 비례

### 2.2 제안 방법

#### 2.2.1 인수 그래프 표현 (Factor Graph Representation)

간단한 예시:
긴 이미지 $u = [x^{(1)}, x^{(2)}, x^{(3)}]$ 생성의 경우, 다음과 같은 조건부 독립성 가정:

$$q(x^{(3)}|x^{(1)}, x^{(2)}) = q(x^{(3)}|x^{(2)})$$

결합 확률분포:
$$q(u) = q(x^{(1)}, x^{(2)}, x^{(3)}) = q(x^{(1)}, x^{(2)}) \cdot \frac{q(x^{(2)}, x^{(3)})}{q(x^{(2)})}$$

점수 함수 분해:
$$\nabla \log q(u) = \nabla \log q(x^{(1)}, x^{(2)}) + \nabla \log q(x^{(2)}, x^{(3)}) - \nabla \log q(x^{(2)})$$

#### 2.2.2 Bethe 근사를 통한 일반화

일반적인 인수 그래프 $G$에서 결합 변수 $u = [x^{(1)}, x^{(2)}, \ldots, x^{(n)}]$와 인수 노드 $\{f^{(j)}\}_{j=1}^{m}$에 대해:

$$p(u) := \frac{\prod_{j=1}^{m} q(f^{(j)})}{\prod_{i=1}^{n} q(x^{(i)})^{d_i-1}}$$

여기서 $d_i$는 변수 노드 $x^{(i)}$의 차수(degree)이다.

점수 함수의 근사:
$$\nabla \log p(u) := \sum_{j=1}^{m} \nabla \log q(f^{(j)}) + \sum_{i=1}^{n} (1-d_i) \nabla \log q(x^{(i)})$$

이는 **Bethe 근사(Bethe Approximation)**로 알려진 변분 추론 방법과 동일하며, 비순환 그래프에서는 정확하고 순환 구조에서는 실제로 잘 작동한다.

#### 2.2.3 훈련 및 샘플링

**훈련:**
시간 의존적 잡음 분포 $q_t(u, t)$에 대해:

$$\nabla \log p_\theta(u,t) = \sum_{j=1}^{m} \nabla \log q_\theta(f^{(j)}, t) + \sum_{i=1}^{n} (1-d_i) \nabla \log q_\theta(x^{(i)}, t)$$

각 인수 및 변수 노드에 대한 확산 모델을 **독립적으로** 훈련한다.

**샘플링:**
역시 동일한 점수 함수 구성으로 표준 확산 모델 샘플러(DDIM, DEIS, DPM-Solver 등) 활용 가능:

$$u_{k-1} = u_k + \dot{\sigma}_{t_k} \sigma_{t_k} s_\theta(u_k, t_k)(t_k - t_{k-1})$$

#### 2.2.4 인수 그래프 구조

다양한 콘텐츠 형태에 적용 가능:

| 콘텐츠 | 그래프 구조 | 특징 |
|--------|-----------|------|
| 길이 제약 없는 시퀀스 | 선형 체인 | 각 인수가 2개 변수 연결 |
| 길이 제약 없는 루프 | 순환 그래프 | 시작과 끝 프레임 연결 |
| 임의 크기 이미지 | 그리드 | 4개 모서리에서 겹침 |
| 360도 이미지 | 큐브맵 그래프 | 6개 면의 3개 순환 구조 |

### 2.3 모델 구조

**핵심 설계:**

1. **점수 함수 결합**: 여러 확산 모델의 점수 출력을 더하고 빼서 결합 분포의 점수 추정
2. **병렬화 가능성**: 모든 인수/변수 노드의 점수를 동시에 계산 가능
3. **샘플러 비종속성**: 기존 ODE/SDE 샘플러 그대로 활용

**수식:**
$$s_\theta(u_k, t_k) = \sum_{j=1}^{m} s_\theta(f^{(j)}, t_k) + \sum_{i=1}^{n} (1-d_i) s_\theta(x^{(i)}, t_k)$$

### 2.4 성능 향상

#### 무한 이미지 생성 (Infinite Image Generation)

| 방법 | FID+ ↓ | 시간(초) ↓ |
|------|--------|----------|
| 기본 (독립 생성) | 24.15 | 5.61 |
| 대체(Replacement) | 10.25 | 14.99 |
| 재구성(Reconstruction) | 8.97 | 26.43 |
| **DiffCollage** | **4.54** | **6.47** |

**개선사항:**
- FID+ 50% 감소 (8.97 → 4.54)
- 시간은 재구성 방식 대비 **75% 단축** (26.43초 → 6.47초)
- 병렬화로 자동회귀 대비 H/(2W)배, H/W배 속도 향상

#### 장시간 모션 생성 (HumanML3D 벤치마크)

24초 모션 생성 (훈련 평균 7.1초):

| 지표 | 실제 데이터 | 기본선 | 대체 | 재구성 | **DiffCollage** |
|------|----------|--------|------|--------|--------|
| R-Precision (Top 3) ↑ | 0.798 | 0.298 | 0.567 | 0.585 | **0.611** |
| FID ↓ | 0.001 | 10.690 | 1.281 | 1.012 | **0.605** |
| Multimodal Dist ↓ | 2.960 | 7.512 | 5.751 | 5.716 | **5.569** |
| Diversity → | 9.471 | 6.764 | 9.184 | 9.175 | **9.372** |

**주요 성과:**
- FID 개선: 1.012 → 0.605 (40% 감소)
- 실제 데이터 FID(0.001)에 매우 근접
- 다양성 지표 유지

#### 임의 크기 이미지 변환 (Arbitrary-sized Image Translation)

비정사각형 이미지 인페인팅:
- **DiffCollage**: 경계 아티팩트 제거, 전역 정보 기반 일관성 있는 복원
- **기본선**: 패치 분할로 인한 명확한 경계선 발생

#### 360도 파노라마 생성

의미론적 분할 맵 조건화:
- 3개 순환 구조 처리: LFRB, ULDR, UFDB
- 전역 일관성 유지하며 개별 면 생성

### 2.5 한계점

**조건부 독립성 가정:**
- 장거리 의존성이 필요한 경우 제약 (예: 뱀의 머리와 꼬리)
- 전역 조건화 정보로 부분 완화 가능 (의미론적 분할 맵)

**메모리 및 계산:**
- 병렬화로 인한 피크 메모리 증가
- 정보 전파를 위한 메시지 패싱: 그래프 직경에 따라 O(L) 반복 필요

**샘플링 단계:**
- 너무 적은 단계(35단계 이하)에서 아티팩트 가능
- 자동회귀 O(L×K) 대비 여전히 개선 (O(L) vs O(L×K))

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 이론적 기반

#### Bethe 근사의 일반화 가능성

DiffCollage의 성공은 다음과 같은 확률 그래프 모델 이론에 기반:

1. **비순환 그래프의 정확성**: 트리 구조(Tree) 또는 선형 체인에서 Bethe 근사는 **정확한** 하한 제공

2. **순환 그래프의 실질적 효율성**: 일반 그래프에서도 신념 전파(Belief Propagation)의 다양한 변형이 실제로 잘 작동함 (신뢰도 높은 추론)

#### 분해 가능성의 이점

전체 분포를 부분 분포(Marginal Distribution)들의 곱으로 분해:
$$p(u) \propto \frac{\prod_{j} q(f^{(j)})}{\prod_{i} q(x^{(i)})^{d_i-1}}$$

**일반화 향상 메커니즘:**
- 각 부분 분포는 상대적으로 단순한 조건부 분포 학습
- 더 작은 데이터 영역에 대한 학습으로 통계적 효율성 증가
- 겹침 영역(변수 노드)이 상호 제약 조건 제공

### 3.2 실증적 일반화 성능

#### 크로스-데이터셋 일반화

**무한 이미지 생성:**
- LHQ 및 LSUN Tower에서 직접 학습하지 않은 작업에도 적용
- 표준 FID와 FID+ 모두에서 경쟁력 있는 성능

| 데이터셋 | 모델 | FID (정사각형) | FID+ (장변) |
|---------|------|--------|----------|
| LHQ | VQGAN | 58.27 | 62.12 |
| LHQ | ALIS | 12.60 | 14.27 |
| LHQ | **DiffCollage** | **6.28** | **16.43** |
| Tower | VQGAN | 45.18 | 47.32 |
| Tower | ALIS | 11.85 | 15.27 |
| Tower | **DiffCollage** | **7.15** | **13.27** |

**의의**: 문제 특화 학습 없이도 우수한 성능 달성

#### 조건부 생성의 일반화

텍스트-동작 생성에서 다양한 길이의 훈련 데이터:
- 사전 학습 모델(MDM)은 1-2개 동작만 생성
- **DiffCollage**: 수동으로 분해된 장편 프롬프트로 복합 동작 생성
- 개별 동작 간 자연스러운 전환

### 3.3 향상 가능성

#### 1. 계층적 인수 그래프 (Hierarchical Factor Graphs)

현재 한계:
- 360도 이미지에서 각 면이 작은 인수 그래프 형성

미래 방향:
- 다단계 계층 구조로 더 복잡한 의존성 모델링
- Junction Tree 같은 더 표현력 있는 그래프 구조

#### 2. 조건화 신호 개선

**현재:**
- 각 노드에 동일 또는 다른 CLIP 임베딩 적용

**개선안:**
- 적응적 조건 선택: 각 노드의 특성에 맞는 조건화
- 다중 조건 결합 학습

#### 3. 메시지 패싱 최적화

**문제:** O(L) 반복이 필요할 수 있음

**해결책:**
- 가속화된 메시지 전파 알고리즘
- 다중 스케일 처리 (계층적 계산)
- 조기 종료 전략

#### 4. 분포 외(OOD) 일반화

**최근 연구 관련:**
- Simplicity 가설: 단순한 모델이 더 잘 일반화됨
- Diffusion Model의 Compositional Generalization 능력

**DiffCollage의 강점:**
- 부분 분포들의 합성이 근본적으로 조성적(Compositional)
- 학습 분포 밖의 크기/형태에 자연스러운 적응

---

## 4. 연구 영향 및 고려사항

### 4.1 논문의 연구 영향

#### 4.1.1 이론적 기여

1. **확률 그래프 모델과 확산 모델의 통합**
   - Bethe 근사를 확산 모델 점수 함수에 적용한 첫 시도
   - 신념 전파(Message Passing) 알고리즘을 생성 모델에 통합

2. **합성 생성(Compositional Generation) 패러다임 제시**
   - 작은 조각 학습 → 큰 콘텐츠 생성의 체계적 방법론
   - 자동회귀 아웃페인팅의 근본적 대안

#### 4.1.2 방법론적 혁신

1. **병렬 샘플링**
   - 기존 확산 모델의 순차적 성질 극복
   - 대규모 콘텐츠 생성에서 실질적 가속화

2. **모듈식 설계**
   - 사전 학습 모델 활용 가능
   - 기존 확산 모델 샘플러 그대로 적용

3. **다양한 작업 포괄**
   - 이미지, 비디오, 동작 등 다중 모달리티
   - 순환 구조, 극단적 종횡비 등 특수 형태

#### 4.1.3 실제 응용의 새로운 가능성

1. **360도 이미지 생성**: 가상 현실/메타버스 콘텐츠
2. **길이 무제한 비디오**: 영화/애니메이션 제작
3. **복잡 동작 합성**: 로봇공학, 게임 애니메이션
4. **스타일 보간**: 두 이미지 간 자연스러운 전환

### 4.2 앞으로의 연구 방향

#### 4.2.1 이론적 심화

**현재 여전히 부족한 부분:**
- Bethe 근사의 오차 한계 분석 (DiffCollage 맥락)
- 서로 다른 그래프 토폴로지의 수렴 보장
- 조건부 독립성 위반 시 영향 분석

**필요한 연구:**
$$\text{KL}(q(u) \| p_\text{Bethe}(u)) \leq \epsilon$$

이러한 상한이 그래프 구조와 노드 차수에 어떻게 의존하는지 분석

#### 4.2.2 기술적 개선

1. **더 강력한 인수 그래프 학습**
   - 현재는 사전 정의된 그래프 사용
   - 데이터로부터 최적 그래프 구조 학습 가능?

2. **적응적 겹침 영역 최적화**
   - 겹침 비율 자동 결정
   - 컨텐츠 특성에 따른 동적 조정

3. **조건부 독립성 가정 완화**
   - 약한 의존성까지 처리하는 인수 도입
   - 고차 상호작용 모델링

#### 4.2.3 확장성 문제

**메모리 병목:**
- 병렬화로 인한 피크 메모리 증가
- 메모리-효율적 메시지 패싱 알고리즘 개발

**계산 효율성:**
- 그래프 직경 감소를 위한 계층적 구조
- 비핵심 영역의 단계 수 적응적 조정

#### 4.2.4 새로운 응용 영역

1. **3D 콘텐츠 생성**
   - 복셀(Voxel) 그리드를 인수 그래프로 표현
   - 멀티뷰 일관성 자동 보장

2. **장형 문서 생성**
   - 텍스트의 장거리 의존성 문제
   - 계층적 인수 그래프로 절 또는 단락 수준 구조화

3. **멀티모달 정렬 생성**
   - 이미지-텍스트-오디오 동시 생성
   - 크로스모달 겹침 영역 정의

---

## 5. 2020년 이후 관련 최신 연구 탐색

### 5.1 확산 모델 기본 발전

#### 5.1.1 코어 아키텍처

**주요 논문:**
- **Denoising Diffusion Probabilistic Models (DDPM, 2020)**
  - 기본 프레임워크 정립
  
- **Score-Based Generative Modeling through SDEs (Song et al., 2021)**
  - 연속시간 프레임워크
  - 확률 흐름 ODE 개발
  
- **Latent Diffusion Models (Rombach et al., 2022)**
  - 잠재공간에서의 확산
  - 효율성 대폭 향상 (16× 계산 감소)

**DiffCollage와의 관계:**
- 확률 흐름 ODE 활용으로 샘플러 비종속성 달성
- 잠재공간 활용 가능성 제시

#### 5.1.2 가속화 기술

**Consistency Models (Song et al., 2023)**
- 한 단계 또는 적은 단계로 샘플링 가능
- 점수 일관성 원칙

**Consistency Trajectory Models (2023)**
- 임의 시간 간 매핑
- 다단계 샘플링 개선

**LCM-LoRA (2023)**
- LoRA를 통한 가속
- 다양한 모델에 범용 적용

**최근 발전 (2024-2025):**
- **Continuous-time Consistency Models (sCM, OpenAI 2024)**
  - 15억 매개변수 규모
  - 2단계 생성으로 50배 가속
  
- **ETC (2025)**
  - 에러 인식 추세 일관성
  - FLUX에서 2.65배 가속
  
- **Multi-Step Consistency Models (2025)**
  - 이론적 수렴 보장
  - $$O(\log(d/\varepsilon))$$ 반복 복잡도

**DiffCollage의 보완 가능성:**
- 개별 노드의 확산 모델을 Consistency Model로 대체
- 더욱 빠른 병렬 생성 가능

### 5.2 대규모 콘텐츠 생성

#### 5.2.1 이미지 생성 확장

**무한/초고해상도 이미지:**
- **InfinityGAN (2021)**
  - 패딩 없는 생성기로 무한 이미지
  
- **NUWA-Infinity (2022)**
  - 자동회귀 기반 무한 이미지
  
**DiffCollage의 장점:**
- 자동회귀 아웃페인팅 불필요
- 병렬 생성으로 훨씬 빠름

#### 5.2.2 비디오 생성

**긴 비디오 생성:**
- **NUWA-XL (2023)**
  - "Diffusion over Diffusion" 아키텍처
  - 전역 확산으로 키프레임, 지역 확산으로 중간 프레임
  - 계층적 구조로 O(L^m) 길이 생성

- **Video Diffusion Models (2022)**
  - 3D 어텐션으로 프레임 간 일관성
  
- **Motion Consistency Model (2024)**
  - 모션과 모양 분리
  - 비디오 확산 증류

**DiffCollage의 연결:**
- 선형 체인 그래프로 장시간 모션 생성 (실증)
- 계층적 인수 그래프로 NUWA-XL 구조 통합 가능

#### 5.2.3 다중 스케일 생성

**최근 접근:**
- **ZoomLDM (2025)**
  - 스케일 조건화 확산
  - 임의 해상도 생성
  
- **LTX-Video (2025)**
  - 32배 공간 다운샘플링
  - 높은 해상도 비디오
  
**DiffCollage의 적용:**
- 각 스케일을 별도 노드로 취급
- 멀티스케일 일관성 자동 보장

### 5.3 조건부 생성 및 제어

#### 5.3.1 가이던스 메커니즘

**Classifier-Free Guidance (Ho & Salimans, 2022)**
- 조건부/무조건 점수 보간
- DiffCollage와 호환

**ControlNet (Zhang et al., 2023)**
- 공간 제어 신호 (선 그리기, 포즈 등)
- 구조 보존 생성

**Regional Attention (2024)**
- 영역별 다중 프롬프트
- SemanticDraw는 10배 속도 향상

**DiffCollage의 통합:**
- 각 노드에 서로 다른 ControlNet 적용 가능
- 일관성 자동 보장

#### 5.3.2 적응적 조건화

**최신 연구:**
- **LLM과 확산 모델 결합 (2023-2024)**
  - LLM으로 프롬프트 정제
  - 약한 감독 활용
  
- **Personalized Generation (2025)**
  - 사용자별 맞춤형 생성
  - 다양한 기초 모델 활용

**DiffCollage에서의 응용:**
- 프롬프트 분해 자동화 가능
- 계층적 조건화로 다단계 제어

### 5.4 이론 및 분석

#### 5.4.1 수렴성 이론

**확산 모델 이론 발전:**
- **Song et al. (2021)**: 확률 흐름 ODE의 수렴 보장
- **Bao et al. (2022)**: KL 발산 경계 분석
- **Consistency Model 수렴 (Song et al., 2023)**
  - 한 단계 생성 $W_2$ 오차 분석

**DiffCollage를 위한 미지의 이론:**
- Bethe 근사 오차와 생성 품질의 관계
- 그래프 구조가 수렴 속도에 미치는 영향

#### 5.4.2 일반화 이론

**Out-of-Distribution Generalization:**
- **Simplicity Hypothesis (2025)**
  - 단순한 모델이 OOD에서 일반화
  - 정규화 최대 우도 추정기
  
  $$E(\hat{\beta}_\lambda) \leq c\left(\text{Tr}(I_T I_S^{-1}) \frac{\log n}{n} + \frac{B_0^2 \|I_T^{1/2} I_S^{-1} \nabla R(\beta^*)\|_2^2}{\Delta^2} \frac{\log n}{n}\right)$$
  
  여기서 $I_S$, $I_T$는 소스/타겟 Fisher Information

**DiffCollage의 이점:**
- 부분 분포 학습이 근본적으로 더 "단순함"
- OOD 크기에 대한 자연스러운 일반화

### 5.5 응용 확대

#### 5.5.1 3D 생성

**3D 확산 모델:**
- **DiffSplat (2025)**
  - 2D 확산을 3D Gaussian Splat로 전환
  - 웹 규모 2D 사전 활용
  
- **Kiss3DGen (2025)**
  - 번들 이미지 표현
  - 2D 확산 재사용

**DiffCollage 적용 가능성:**
- 3D 복셀 또는 포인트 클라우드를 공간 인수 그래프로
- 멀티뷰 일관성 자동

#### 5.5.2 소프트웨어/프레임워크

**생성 가속 프레임워크:**
- **SGLang Diffusion (2025)**
  - 비디오/이미지 생성 통합 시스템
  - USP (Unified Sequence Parallelism) 활용
  
- **FastVideo**
  - 훈련 최적화 집중

**구현 고려사항:**
- DiffCollage를 이러한 통합 프레임워크에 통합
- 다양한 병렬화 전략 실험

### 5.6 그래프 신경망과의 연결

#### 5.6.1 인수 그래프 기반 방법

**관련 연구:**
- **Factor Graph Neural Networks (2023)**
  - 고차 관계 학습
  - Sum-Product 신념 전파
  
- **PGMax (2022)**
  - JAX 기반 이산 PGM
  - 확장 가능 신념 전파
  
- **Coordinated Manipulation with Diffusion-based Factor Graph (2024)**
  - 조작 계획에 확산 + 인수 그래프
  - 공간-시간 분포 합성

**DiffCollage의 위치:**
- 연속 공간 확산 × 이산 그래프 구조
- 기존 PGM 추론 알고리즘 활용 가능

#### 5.6.2 메시지 패싱

**관련 연구:**
- **Message Passing Algorithms and Homology (2020)**
  - 대수/위상적 기초
  - 신념 전파를 확산 방정식으로
  
- **Diffusion Twigs with Loop Guidance (2024)**
  - 다중 공진화 확산 프로세스
  - 루프 가이던스 정보 흐름 제어

**DiffCollage의 발전:**
- 신념 전파와 점수 기반 확산의 명시적 연결
- 메시지 전파 최적화를 통한 가속

---

## 6. 결론 및 미래 전망

### 6.1 DiffCollage의 위치

DiffCollage는 다음과 같은 점에서 획기적이다:

1. **패러다임 변화**: 자동회귀 → 병렬 합성 생성
2. **이론 통합**: 확률 그래프 모델 × 확산 모델
3. **실용성**: 사전 학습 모델 활용으로 추가 학습 불필요
4. **일반성**: 이미지, 비디오, 모션 등 다중 모달리티

### 6.2 앞으로의 연구 우선순위

**단기 (1-2년):**
- 계층적 인수 그래프 구현 및 검증
- Consistency Model 통합
- 3D 콘텐츠 확대

**중기 (2-4년):**
- 자동 그래프 구조 학습
- 조건부 독립성 가정 완화 이론
- Junction Tree 등 고차 표현 통합

**장기 (4년 이상):**
- 일반화 성능의 이론적 보장
- 멀티모달 대규모 콘텐츠 통합 생성
- 실시간 상호작용 콘텐츠 시스템

### 6.3 실제 영향

**학술:**
- 확산 모델의 구성성(Compositionality) 이해 심화
- 확률 그래프 모델과 생성 모델의 융합 연구 활성화

**산업:**
- 영화/게임: 장편 콘텐츠 자동 생성
- VR/메타버스: 360도 환경 신속 생성
- 로봇공학: 복합 동작 시뮬레이션

**사회:**
- 창작 도구 민주화
- 콘텐츠 개인화 확대
- 윤리적 고려: 합성 미디어의 신뢰성

---

## 참고: 핵심 수식 요약

**결합 분포 근사:**
$$p(u) = \frac{\prod_{j=1}^{m} q(f^{(j)})}{\prod_{i=1}^{n} q(x^{(i)})^{d_i-1}}$$

**점수 함수 분해:**
$$\nabla \log p(u) = \sum_{j=1}^{m} \nabla \log q(f^{(j)}) + \sum_{i=1}^{n} (1-d_i) \nabla \log q(x^{(i)})$$

**시간 종속 버전:**
$$\nabla \log p_\theta(u,t) = \sum_{j=1}^{m} \nabla \log q_\theta(f^{(j)}, t) + \sum_{i=1}^{n} (1-d_i) \nabla \log q_\theta(x^{(i)}, t)$$

**역확산 SDE:**
$$du = -(\frac{1+\eta^2}{2})\frac{\dot{\sigma}_t}{\sigma_t}\nabla_u \log q_t(u)dt + \eta\sqrt{2\frac{\dot{\sigma}_t}{\sigma_t}}dw$$

**Euler 샘플링:**
$$u_{k-1} = u_k + \dot{\sigma}\_{t_k}\sigma_{t_k}s_\theta(u_k, t_k)(t_k - t_{k-1})$$

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/222da000-e482-4bad-a59b-0c9e716469e2/2303.17076v1.pdf)
[2](https://dl.acm.org/doi/10.1145/3731715.3734580)
[3](https://ieeexplore.ieee.org/document/11152024/)
[4](https://arxiv.org/abs/2501.16764)
[5](https://ieeexplore.ieee.org/document/11161497/)
[6](https://ieeexplore.ieee.org/document/11242610/)
[7](https://ieeexplore.ieee.org/document/11092730/)
[8](https://ieeexplore.ieee.org/document/10904573/)
[9](https://arxiv.org/abs/2503.15877)
[10](https://arxiv.org/abs/2502.09935)
[11](https://arxiv.org/abs/2501.15571)
[12](https://arxiv.org/html/2502.06805v1)
[13](https://arxiv.org/pdf/2301.04655.pdf)
[14](https://arxiv.org/pdf/2412.09656.pdf)
[15](https://arxiv.org/pdf/2211.01324.pdf)
[16](https://arxiv.org/html/2403.09055)
[17](https://arxiv.org/pdf/2303.13052.pdf)
[18](https://arxiv.org/html/2403.07860v1)
[19](https://arxiv.org/pdf/2402.16369.pdf)
[20](https://aclanthology.org/2025.acl-long.1201.pdf)
[21](https://huggingface.co/docs/diffusers/v0.18.0/api/pipelines/paradigms)
[22](https://openaccess.thecvf.com/content/ICCV2025/papers/Tsai_LightsOut_Diffusion-based_Outpainting_for_Enhanced_Lens_Flare_Removal_ICCV_2025_paper.pdf)
[23](https://arxiv.org/pdf/2502.09992.pdf)
[24](https://lmsys.org/blog/2025-11-07-sglang-diffusion/)
[25](https://blog.segmind.com/stable-diffusion-inpainting-vs-outpainting/)
[26](https://openaccess.thecvf.com/content/CVPR2025/papers/Han_Enhancing_Creative_Generation_on_Stable_Diffusion-based_Models_CVPR_2025_paper.pdf)
[27](https://arxiv.org/abs/2303.07909)
[28](https://arxiv.org/html/2511.20996v1)
[29](https://openreview.net/forum?id=KnqiC0znVF)
[30](https://peerj.com/articles/cs-1905/)
[31](https://academic.oup.com/nsr/article/11/12/nwae348/7810289)
[32](https://arxiv.org/abs/2303.17076)
[33](https://huggingface.co/blog/OzzyGT/outpainting-differential-diffusion)
[34](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/diffcollage/)
[35](https://www.semanticscholar.org/paper/1c1c361a07b344d227af1f3772e857f206bdb347)
[36](https://ieeexplore.ieee.org/document/9511816/)
[37](http://arxiv.org/abs/2406.08286v2)
[38](https://www.semanticscholar.org/paper/3c8a144ef4d7e6910de45eed51641bc9906311d0)
[39](https://www.semanticscholar.org/paper/d42afcd122357e31a58440c8ea58130de7eb5ea5)
[40](https://dl.acm.org/doi/10.1145/3589132.3625614)
[41](https://arxiv.org/abs/2308.00887)
[42](https://arxiv.org/abs/2506.11869)
[43](https://dl.acm.org/doi/10.1145/3678717.3691235)
[44](https://arxiv.org/abs/2302.10506)
[45](https://arxiv.org/pdf/2009.11631.pdf)
[46](https://arxiv.org/pdf/2202.04110.pdf)
[47](https://arxiv.org/pdf/2401.15617.pdf)
[48](https://arxiv.org/pdf/1212.2486.pdf)
[49](http://arxiv.org/pdf/2403.20221.pdf)
[50](https://arxiv.org/pdf/1207.4136.pdf)
[51](https://arxiv.org/html/2410.24012v1)
[52](http://arxiv.org/pdf/2402.03687.pdf)
[53](https://openreview.net/pdf/efcdde954896a06d851aaa59e69d0c8c159cd9a1.pdf)
[54](https://www.nature.com/articles/s41586-025-09529-3)
[55](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1255566/full)
[56](https://www.doc.ic.ac.uk/~dfg/ProbabilisticInference/old_IDAPISlides11.pdf)
[57](https://www.jmlr.org/papers/volume26/24-1335/24-1335.pdf)
[58](https://www.ijcai.org/proceedings/2025/0764.pdf)
[59](https://ocw.mit.edu/courses/6-438-algorithms-for-inference-fall-2014/3e3e9934d12e3537b4e9b46b53cd5bf1_MIT6_438F14_Lec4.pdf)
[60](https://arxiv.org/pdf/2501.07763.pdf)
[61](https://arxiv.org/html/2307.04726v4)
[62](https://www.youtube.com/watch?v=fXD6KJB1U20)
[63](https://transp-or.epfl.ch/heart/2025/abstracts/hEART_2025_shortpaper_95.pdf)
[64](https://www.themoonlight.io/ko/review/principled-out-of-distribution-generalization-via-simplicity)
[65](https://www.sciencedirect.com/science/article/pii/S1361841524000136)
[66](https://arxiv.org/html/2409.16275v1)
[67](https://arxiv.org/abs/2311.05556)
[68](https://arxiv.org/abs/2510.24129)
[69](https://arxiv.org/abs/2310.02279)
[70](https://www.semanticscholar.org/paper/84c6486feb4c071f1db41d3b2fe295a71a2536a5)
[71](https://www.semanticscholar.org/paper/f0795f91419195ff65fb49c50522901de5b62193)
[72](https://arxiv.org/abs/2505.01049)
[73](https://ieeexplore.ieee.org/document/11245674/)
[74](https://arxiv.org/abs/2308.11449)
[75](https://www.ewadirect.com/proceedings/ace/article/view/28950)
[76](https://ieeexplore.ieee.org/document/10704728/)
[77](http://arxiv.org/pdf/2310.02279.pdf)
[78](https://arxiv.org/html/2403.01505)
[79](https://arxiv.org/html/2406.12303v2)
[80](http://arxiv.org/pdf/2410.11081.pdf)
[81](http://arxiv.org/pdf/2412.17162.pdf)
[82](http://arxiv.org/pdf/2309.10740v1.pdf)
[83](http://arxiv.org/pdf/2404.07946.pdf)
[84](https://arxiv.org/pdf/2402.07802.pdf)
[85](https://github.com/G-U-N/Phased-Consistency-Model)
[86](https://arxiv.org/html/2404.01367v2)
[87](https://openreview.net/pdf/62fe9cdb0c4e4a640f28ae22bf0867f590ed2f9c.pdf)
[88](https://proceedings.neurips.cc/paper_files/paper/2024/file/c859b99b5d717c9035e79d43dfd69435-Paper-Conference.pdf)
[89](https://openreview.net/forum?id=0u7pWfjri5)
[90](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/)
[91](https://arxiv.org/abs/2406.06890)
[92](https://openaccess.thecvf.com/content/CVPR2025/papers/Yellapragada_ZoomLDM_Latent_Diffusion_Model_for_Multi-scale_Image_Generation_CVPR_2025_paper.pdf)
[93](https://pmc.ncbi.nlm.nih.gov/articles/PMC10606505/)
[94](https://openreview.net/forum?id=NsqxN9iOJ7)
[95](https://openai.com/index/simplifying-stabilizing-and-scaling-continuous-time-consistency-models/)
[96](https://sander.ai/2025/04/15/latents.html)
[97](https://aclanthology.org/2023.acl-long.73.pdf)
[98](https://proceedings.iclr.cc/paper_files/paper/2025/file/9ead108421b202494d01b5060d12aa34-Paper-Conference.pdf)
[99](https://github.com/showlab/Awesome-Video-Diffusion)
[100](https://arxiv.org/abs/2303.01469)
