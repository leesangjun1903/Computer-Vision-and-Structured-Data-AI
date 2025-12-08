
# Unlimited-Size Diffusion Restoration

## 1. 핵심 주장 및 주요 기여 요약

이 논문은 **제로샷 이미지 복원 방법이 임의의 크기 이미지로 확장될 수 있는 방법**을 제시합니다. 기존의 Denoising Diffusion Null-space Model (DDNM)은 고정된 크기 이미지에만 처리 가능했지만, 본 논문은 두 가지 파라미터 프리(parameter-free) 방법을 제안하여 **무제한 크기의 이미지 복원 및 생성**을 가능하게 합니다.[1]

주요 기여는 다음과 같습니다:

- **Mask-Shift Restoration (MSR)**: 패치 기반 처리 시 경계 아티팩트(boundary artifacts)를 제거하여 **로컬 일관성** 확보[1]
- **Hierarchical Restoration (HiR)**: 멀티스케일 접근으로 글로벌 의미 정보를 유지하여 **아웃-오브-도메인 문제 완화**[1]
- **다양한 선형 역문제 지원**: 슈퍼 해상도, 컬러화, 인페인팅, 디노이징 등 모든 선형 역문제에 적용 가능[1]

***

## 2. 문제 정의, 제안 방법 및 모델 구조

### 2.1 해결하고자 하는 문제

**주요 제약 사항:**[1]
1. 기존 확산 모델은 고정 크기 이미지(예: 256×256)로 학습되어 임의 크기 입력 시 Out-Of-Domain(OOD) 문제 발생
2. 단순히 패치 분할 후 독립적 처리 시 패치 경계에서 블록 아티팩트 발생
3. 글로벌 의미 정보 손실로 인한 낮은 복원 품질

### 2.2 제안하는 방법 및 수식

#### **Range-Null Space Decomposition (RND) 기반 이론**

선형 역문제 $y = Ax$에서:[1]

$$x^* = A^{\dagger}y + (I - A^{\dagger}A)x_r$$

여기서:
- $A^{\dagger}$: $A$의 의사역행렬 (pseudo-inverse)
- $A^{\dagger}y$: Range-space 성분 (복원 가능한 정보)
- $(I - A^{\dagger}A)x_r$: Null-space 성분 (복원 불가능한 정보)

#### **DDNM 기반 적용**

시간 단계 $t$에서 추정된 클린 이미지 $\hat{x}_{0|t}$를 null-space 변수로 사용:[1]

$$\hat{x}_{0|t}^* = A^{\dagger}y + (I - A^{\dagger}A)\hat{x}_{0|t}$$

이 과정을 역확산 샘플링에 통합합니다:[1]

$$x_{t-1} = a_{t-1}\hat{x}_{0|t}^* + \sigma_{t-1}\epsilon_{t-1} + \frac{\sqrt{1-a_{t-1}^2}-\sigma_{t-1}^2}{\sigma_t}\epsilon_t + \sqrt{\sigma^2_{t-1}}n, \quad n \sim \mathcal{N}(0, I)$$

#### **Mask-Shift Restoration (MSR)**

패치 처리 시 겹치는 영역을 인페인팅 제약으로 추가:[1]

$$\hat{x}_{0|t}^* = A_m \circ x_0' + (I - A_m) \circ \hat{x}_{0|t}^*$$

여기서:
- $A_m$: 이미 복원된 영역의 마스크 연산자
- $x_0'$: 이전 패치의 복원 결과

**Algorithm 2 (MSR 기반 DDNM)**[1]
```
입력: 노이즈 xT ~ N(0, I), 이미 복원된 패치 x₀'
t = T, ..., 1에 대해:
  εt = Z(xt, t)  // 디노이저 적용
  x̂₀|t = (1/at)(xt - σt·εt)
  x̂₀|t* = A†y + (I - A†A)x̂₀|t
  x̂₀|t* = Am ∘ x₀' + (I - Am) ∘ x̂₀|t*  // MSR 제약 추가
  xt-1 = at-1·x̂₀|t* + σt-1·εt-1 + √(√(1-at-1²)-σt-1²)/σt·εt + √σt-1²·n
반환: x₀*
```

#### **Hierarchical Restoration (HiR)**

**Phase 1: 의미 정보 복원**
- 입력 이미지를 2배 다운샘플링
- 다운샘플된 이미지에 MSR 적용하여 저주파 참고 $\hat{x}_0$ 획득

**Phase 2: 텍스처 및 세부사항 복원**

저주파 가이드 제약 추가:[1]

$$\hat{x}_{0|t}^* = A_{sr}^{\dagger}\hat{x}_0 + (I - A_{sr}^{\dagger}A_{sr})\hat{x}_{0|t}^*$$

그 후 다시 MSR 적약 적용:

$$\hat{x}_{0|t}^* = A_m \circ x_0' + (I - A_m) \circ \hat{x}_{0|t}^*$$

**Algorithm 3 (HiR + MSR 통합)**[1]
```
입력: xT ~ N(0, I), 저주파 결과 x̂₀, 다운샘플러 Asr
t = T, ..., 1에 대해:
  εt = Z(xt, t)
  x̂₀|t = (1/at)(xt - σt·εt)
  x̂₀|t* = Asr†·x̂₀ + (I - Asr†·Asr)·x̂₀|t  // 저주파 제약
  x̂₀|t* = A†y + (I - A†A)·x̂₀|t*  // 데이터 일관성 제약
  x̂₀|t* = Am ∘ x₀' + (I - Am) ∘ x̂₀|t*  // MSR 제약
  xt-1 = at-1·x̂₀|t* + σt-1·εt-1 + ...
반환: x₀*
```

### 2.3 모델 구조

**구조적 특징:**[1]

1. **기존 DDNM 기반**: 새로운 네트워크 구조 추가 없이 샘플링 과정만 수정
2. **U-Net 디노이저**: ImageNet 256×256에서 사전학습된 guided-diffusion 사용
3. **패치 처리 설정**:
   - 패치 크기: 256×256
   - 패치 간 겹침: 128 픽셀
   - 처리 순서: 좌→우, 상→하

**일반화 가능성:**[1]
- DDNM, ILVR, RePaint, DPS 등 다양한 제로샷 방법에 적용 가능
- 다른 확산 모델과도 호환성 유지

***

## 3. 성능 향상 및 실험 결과

### 3.1 정성적 성능 비교

**4배 슈퍼 해상도 (4× SR):**[1]
- 단순 패치 분할: 명확한 블록 경계 아티팩트 및 불일치한 색상
- MSR 적용: 경계 아티팩트 제거, 로컬 일관성 확보
- MSR + HiR: 글로벌 의미 정보 복원으로 최고 품질 달성

**대규모 인페인팅 (512×768):**[1]
- DDNM 단독: 작은 패치로 인한 의미 손실
- MSR: 로컬 일관성 개선하나 여전히 불합리한 구조
- MSR + HiR: 의미적으로 일관성 있는 결과 생성

**컬러화 (1268×1024):**[1]
- HiR을 통한 멀티스케일 접근으로 색상 정보를 올바르게 유지하면서 고해상도 처리 가능

### 3.2 정량적 평가

**노이즈가 있는 4× SR 비교 (BSRGAN과의 비교):**[1]
- 제안 방법의 장점:
  - **색상 및 구조 보존**: RND 원리에 의해 LR 이미지의 정보를 충실하게 유지
  - **현실성**: 확산 모델의 생성 능력으로 자연스러운 고주파 성분 생성
  - 예: 나비 샘플에서 BSRGAN은 색상 오류 발생, 제안 방법은 정확한 색상 복원

### 3.3 실험 설정

**기본 설정:**[1]
- 사전학습 확산 모델: ImageNet 256×256 (guided-diffusion)
- 역확산 스텝: T = 100 (4× SR), T = 250 (노이즈 4× SR)
- Time-travel 샘플링: 길이 l = 10, 반복 횟수 r = 3
- Classifier guidance 적용

***

## 4. 모델의 일반화 성능 향상 가능성

### 4.1 OOD (Out-Of-Domain) 문제 해결 메커니즘

**OOD 문제의 원인:**[1]
- CelebA 256×256 데이터로 학습한 모델이 512×512 입력에서 큰 얼굴 생성 불가
- 학습 분포 범위를 벗어난 입력에 대한 성능 저하

**HiR을 통한 일반화 개선:**[1]
1. **멀티스케일 계층 구조**: 작은 패치에서 시작하여 점진적으로 확대
   - 저해상도에서 의미 정보 먼저 확보
   - 고해상도에서 세부사항만 추가 (OOD 정도 감소)

2. **저주파 가이드 활용**:
   - 이미 확보된 저주파 성분을 하드 제약으로 고정
   - 모델이 불확실한 고주파 영역에만 집중
   - 확산 모델의 범위 내에서만 생성

### 4.2 임의 크기로의 일반화

**이론적 근거:**[1]
- U-Net은 완전 합성곱(fully convolutional) 구조로 입력 크기에 무관하게 동작 가능
- RND 원리는 임의의 선형 연산자 $A$에 적용 가능
- MSR, HiR은 **파라미터 프리** → 크기에 무관한 일관성 있는 처리

**확인된 크기 범위:**[1]
- 슈퍼 해상도: 64×32 → 1024×512 (16배)
- 인페인팅: 512×768 (큰 패치 범위)
- 컬러화: 1268×1024 (매우 큰 이미지)

### 4.3 다양한 작업으로의 일반화

**선형 역문제의 일반성:**[1]
저속한 선형 연산자 $A$로 정의되는 모든 문제에 적용:
- **슈퍼 해상도**: $A$ = 평균 풀링 다운샘플러
- **인페인팅**: $A$ = 마스크 연산자
- **컬러화**: $A$ = 색상 채널 선택 연산자
- **인페인팅 (compressed sensing)**: $A$ = 푸리에 샘플링
- **디블러링**: $A$ = 블러 커널의 합성곱

### 4.4 확산 모델 성능에 따른 상한**

**중요한 특성:**[1]
"The upper limit of the restoration performance depends on the pre-trained diffusion models, which are in rapid evolution."

- 더 나은 확산 모델 (예: Imagen)이 개발되면 자동으로 성능 향상 가능
- 추가 재학습이나 모델 수정 필요 없음
- DDNM의 zero-shot 특성 유지

***

## 5. 모델의 한계

### 5.1 계산 비용

**주요 제약:**[1]
- 계산 시간 및 메모리 소비가 감독 학습 기반 방법(BSRGAN 등)보다 훨씬 많음
- 확산 모델의 반복적 샘플링 특성으로 인한 느린 추론 속도

### 5.2 확산 모델 기반 한계

1. **비공개 모델**: Imagen 등 최신 모델 미공개로 최고 성능 달성 어려움
2. **잠재 공간 모델**: Stable Diffusion(LAION-5B 학습)은 잠재 공간에서 동작하여 제로샷 방법 적용 어려움[1]
3. **선형성 제약**: 선형 역문제만 처리 가능 (비선형 문제는 DPS 같은 방법 필요)

### 5.3 명시적 연산자 필요

**제약:**[1]
"The degradation operator is explicitly needed, which makes it difficult for tasks like rain and haze removal."

- 빗줄기 제거, 안개 제거 등 복잡한 비선형 강하 모델이 필요한 작업에는 부적합
- 명확한 $A$ 정의 불가능한 블라인드 복원 문제에 직접 적용 불가

### 5.4 패치 크기 제약

**실제 한계:**[1]
- 패치 크기는 사전학습 모델 크기(256×256)에 의존
- 매우 큰 이미지의 경우 작은 패치로 인한 문맥 손실 가능
- HiR로 일부 완화되지만 근본적 해결은 아님

***

## 6. 최신 관련 연구 탐색 (2020년 이후)

### 6.1 확산 모델 기반 이미지 복원의 진화

**확산 모델 이미지 복원 종합 리뷰 (2025):**[2]
확산 모델이 GAN을 능가하는 성능을 보이고 있으며, 슈퍼 해상도, 이미지 복원, 그리고 빈도 선택적 이미지 복원 작업에서 널리 활용 중입니다. 최근 연구들은 6가지 주요 방법론과 통합 패러다임을 제시합니다.

### 6.2 제로샷 방법의 발전

**DPS (Diffusion Posterior Sampling, 2022-2023):**[3][4]
- DDNM보다 진보된 방식으로 **노이즈 있는 비선형 역문제** 처리 가능[3]
- 측정 통계를 직접 샘플링 프로세스에 통합
- 하드 프로젝션 대신 소프트 제약 사용으로 더 자연스러운 생성 경로

**ILVR, RePaint:**[1]
- DDNM의 대안 제로샷 방법
- MSR, HiR과 호환 가능 (선택적 적용)

### 6.3 멀티스케일 및 계층적 접근법

**Hierarchical Patch Diffusion Models (CVPR 2024):**[5][6]
- 고해상도 비디오 생성을 위한 패치 기반 확산 모델
- **Deep Context Fusion**: 저해상도 패치의 맥락을 고해상도로 전파
- Adaptive Computation: 계산 효율성 증대
- MSR/HiR과 유사한 계층적 설계 원리

### 6.4 OOD 일반화 연구

**A Sampling-Based Domain Generalization Study (2025):**[7]
- 사전학습 확산 모델의 도메인 일반화 능력 분석
- OOD 이미지가 역변환 후 잠재 공간에서 **분리 가능한 가우시안** 형성
- 파인튜닝 없이 샘플링 기반 OOD 생성 가능

### 6.5 물리 기반 제약 통합

**Physical-Aware Diffusion (PhyDiff, 2025):**[8]
- 물리 모델(전송 맵)을 확산 과정에 통합
- 비선형 산란 강하 문제(안개 제거, 수중 이미지 복원) 해결
- Transmission-guided Conditional Generation (TCG)로 동적 가이딩

### 6.6 범용 이미지 복원 모델

**All-in-One Diffusion-Based Restoration (2025):**[9]
- **Multi-Degradation 동시 처리**: 다중 열화 유형을 단일 패스에서 복원
- Dual-Domain Architecture: 공간-주파수 영역 특성 동시 처리
- CLIP Attention 기반 고수준 의미 정보 추출

**UniCoRN (2025):**[10]
- Mixture-of-Experts 기반 다중 헤드 확산 모델
- 특정 강하 유형 사전 가정 없이 학습하는 커리큘럼 학습

### 6.7 잠재 공간 활용 진전

**Zero-Shot Inpainting with Latent Diffusion Models (2025):**[11]
- DDNM을 잠재 공간 기반 Stable Diffusion으로 확장
- 선형성 제약 극복을 위한 공간 특성 보존 기법
- 더 다양한 이미지에 적용 가능

### 6.8 학습 기반 하이브리드 접근

**DiffLoss (2024):**[12]
- 확산 모델을 **제약(constraint)** 으로 활용하여 일반 복원 네트워크 학습
- 빠른 추론 속도 유지하면서 확산 모델의 선행 정보 활용

**RAP-SR (2025):**[13]
- Stable Diffusion의 선행(prior) 강화
- 고수준 미학적 이미지 데이터셋(HFAID) 구성
- Restoration-Oriented Prompt Optimization (ROPO)

### 6.9 접근법 비교: RND 기반 vs 최적화 기반

**관련 연구:**[14][15]
| 방법 | 장점 | 단점 |
|------|------|------|
| **RND 기반** (DDNM, 본 논문의 MSR/HiR) | 선형 역문제에 최적화, 빠른 수렴, SVD 기반 명확한 이론 | 선형 문제만 처리 가능 |
| **최적화 기반** (DPS, CCDF) | 비선형 문제 처리, 유연성, 측정 잡음 처리 | 더 많은 메모리/시간 소비 |

### 6.10 기타 확장 응용

**응용 분야 확대:**[1]
1. **고해상도 비디오 생성**: 시간적 일관성 유지 (2025)
2. **의료 이미징**: MRI, CT, 위상 차 현미경 (2023-2025)
3. **원격 감지**: 대규모 위성 이미지 복원 (2024-2025)
4. **수중 이미지 복원**: 물리 기반 강하 모델 통합 (2025)
5. **텍스트 인식 이미지 복원**: 문자 영역의 충실도 향상 (2025)

***

## 7. 앞으로의 연구에 미치는 영향

### 7.1 패러다임 전환

**Zero-Shot 대 Fine-tuning:**[1]
- 과거: 각 작업마다 특화된 네트워크 필요
- 현재 (본 논문): 사전학습 확산 모델만으로 임의 작업 처리
- **영향**: AI 시스템의 재사용성과 유연성 극대화

### 7.2 이론적 기여

**Range-Null Space분해 재조명:**
- 선형대수 개념을 현대 생성 모델에 우아하게 적용
- 확산 모델의 null-space 학습에 대한 명확한 수학적 기초 제공[1]
- 다른 역문제 해결 프레임워크의 기초 마련

### 7.3 실무 응용 기대효과

**산업 적용 가능성:**[1]
1. **무제한 크기 처리**: 대형 이미지/비디오 플랫폼에 직접 적용 가능
2. **실시간 처리 개선**: HiR의 다단계 구조로 점진적 처리 최적화 가능
3. **도메인 적응**: 새로운 강하 유형에 신속하게 대응 가능 (재학습 불필요)

### 7.4 앞으로 고려할 연구 방향

#### **7.4.1 계산 효율성 개선**

**현재 한계:**[1]
- DDNM+ DDIM 기반 샘플링으로도 수백 스텝 필요
- 실시간 응용에는 부적합

**연구 방향:**
1. **빠른 ODE 솔버**: DPM-Solver 같은 가속 기법 통합[10]
2. **적응적 스텝 크기**: 이미지 영역별로 다른 스텝 수 할당
3. **캐싱 메커니즘**: 반복되는 계산 저장

#### **7.4.2 비선형 문제 확장**

**제약:** "The degradation operator is explicitly needed"[1]

**해결 방안:**
1. **블라인드 복원**: 강하 연산자 자체를 학습하는 멀티태스크 프레임워크
2. **DPS 통합**: 최적화 기반 제약과 RND 제약 결합
3. **물리 모델 통합**: PhyDiff 스타일의 도메인 지식 활용

#### **7.4.3 잠재 공간 호환성**

**과제:** "Stable Diffusion... latent space... makes it difficult to apply zero-shot methods"[1]

**연구 방안:**
1. **잠재-이미지 공간 매핑**: 선형성 보존하는 매핑 함수 설계
2. **하이브리드 접근**: 이미지 공간 + 잠재 공간 제약 동시 적용
3. **암묵적 연산자 학습**: 비선형 편코더를 근사하는 암묵적 연산자

#### **7.4.4 다중 열화 처리**

**현재**: MSR/HiR은 단일 종류의 선형 연산자 가정

**확장 방향:**
1. **혼합 연산자**: 여러 강하가 동시에 작용하는 경우
   - 가우시안 블러 + 노이즈 + 다운샘플링
   - 인페인팅 + 컬러화 동시 처리

2. **계층적 강하 분해**: 열화를 계층별로 분리
   - Phase 1: 주요 강하 (예: 블러) 복원
   - Phase 2: 부차 강하 (예: 노이즈) 제거

#### **7.4.5 일반화 성능 더 높이기**

**현재 수준:** ImageNet 256×256 사전학습 → 임의 크기 처리 가능[1]

**향상 방안:**
1. **랜덤 크롭 사전학습**:  "OOD issue can be solved by training with random cropped images"[1]
   - LAION-5B 같은 비정렬 데이터의 이점 활용

2. **도메인 특화 모델**: 의료 이미징, 위성 이미지, 얼굴 등 분야별 사전학습

3. **메타 학습**: 신규 도메인에 극소수의 예제만으로 적응

#### **7.4.6 패치 처리의 정교화**

**현재 방식:** 고정 크기 패치 (256×256), 고정 겹침 (128 픽셀)[1]

**개선 아이디어:**
1. **적응적 패치 크기**: 이미지 복잡도에 따라 동적 조정
   - 배경 영역: 큰 패치 (빠른 처리)
   - 세부 영역: 작은 패치 (높은 품질)

2. **주의 기반 겹침**: 경계 주변에만 큰 겹침 할당

3. **그래디언트 기반 병합**: 단순 연결 대신 부드러운 천이

#### **7.4.7 확산 모델과의 공진 설계**

**미래 확산 모델을 고려한 설계:** "once a more powerful Diffusion Model is available"[1]

**호환성 확보:**
1. **모델 무관 인터페이스**: DDNM, DPS, ILVR, RePaint 모두 지원하는 통합 프레임워크

2. **버전 관리 시스템**: 새로운 확산 모델의 성능 이득을 자동으로 활용

3. **조건부 생성 지원**: Classifier guidance 외 다른 조건화 메커니즘 통합

#### **7.4.8 해석 가능성 및 검증**

**필요한 연구:**
1. **null-space 시각화**: 복원 불가능한 정보가 실제로 무엇인지 분석

2. **성능 상한선 이론**: 주어진 $A$와 확산 모델에 대한 최대 성능 계산

3. **오류 분석**: 어떤 유형의 입력에서 MSR/HiR이 실패하는지 체계적 분석

#### **7.4.9 실제 시스템 구현**

**배포 고려사항:**
1. **메모리 효율적 샘플링**: 모바일 디바이스, 엣지 컴퓨팅 지원

2. **배치 처리 최적화**: 여러 패치를 병렬 처리하되 일관성 유지

3. **동적 하드웨어 할당**: GPU/CPU 혼합 처리로 비용 최소화

#### **7.4.10 사용자 경험 관점**

**실제 응용에서의 과제:**
1. **성능-속도 트레이드오프**: 사용자가 품질과 속도 선택 가능하도록

2. **대화형 복원**: 사용자 피드백을 통한 반복적 개선

3. **결과 다양성 제어**: 확률적 특성을 활용한 복원 옵션 제시

***

## 8. 결론

**"Unlimited-Size Diffusion Restoration"** 논문은 **Range-Null Space Decomposition 원리에 기반한 우아한 솔루션**을 제시합니다. MSR과 HiR의 두 가지 파라미터 프리 기법을 통해 고정 크기 제약으로부터 해방되면서도 **zero-shot 특성을 완벽히 유지**합니다.[1]

**핵심 혁신:**
- 로컬 일관성 문제 (MSR)와 글로벌 의미 문제 (HiR)를 **별도의 메커니즘**으로 우아하게 분리 해결
- **계산 오버헤드 최소**: 단순 추가 선 하나로 기존 DDNM에 통합 가능

**영향과 한계:**
- ✅ 무제한 크기 처리, 다양한 선형 역문제 지원, 재학습 불필요
- ⚠️ 계산 비용, 선형 문제 제한, 명시적 연산자 필요, 비공개 SOTA 모델 미지원

**미래 연구의 핵심 방향:**
1. **계산 효율화**: 빠른 ODE 솔버, 적응적 처리
2. **비선형 확장**: 블라인드 복원, 복합 강하 모델링
3. **잠재 공간 호환**: Stable Diffusion 등과 통합
4. **이론적 심화**: 성능 상한선, null-space 의미 분석
5. **실무 배포**: 모바일, 엣지 컴퓨팅, 대화형 시스템

이 논문은 **zero-shot 이미지 복원의 실용성을 획기적으로 향상**시키며, 향후 5년간 확산 모델 기반 복원 연구의 **주류 기준점**이 될 것으로 예상됩니다.[16][2][1]

***

## 참고문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e5c82ea6-766b-40df-8b33-b60e4ab88c0f/2303.00354v1.pdf)
[2](https://www.semanticscholar.org/paper/756efc5cfb34e7da5c3901332f738e609401d86b)
[3](https://arxiv.org/abs/2209.14687)
[4](https://pure.kaist.ac.kr/en/publications/diffusion-posterior-sampling-for-general-noisy-inverse-problems/)
[5](https://snap-research.github.io/hpdm/)
[6](https://openaccess.thecvf.com/content/CVPR2024/papers/Skorokhodov_Hierarchical_Patch_Diffusion_Models_for_High-Resolution_Video_Generation_CVPR_2024_paper.pdf)
[7](https://arxiv.org/html/2310.09213v3)
[8](https://www.sciencedirect.com/science/article/pii/S0031320325001335)
[9](https://www.sciencedirect.com/science/article/abs/pii/S0925231225021897)
[10](https://arxiv.org/html/2503.15868)
[11](https://ieeexplore.ieee.org/document/10920775/)
[12](https://arxiv.org/html/2406.19030v1)
[13](https://ojs.aaai.org/index.php/AAAI/article/view/32832)
[14](https://arxiv.org/html/2308.09388v2)
[15](http://arxiv.org/pdf/2212.00490.pdf)
[16](https://www.mdpi.com/2227-7390/13/13/2079)
[17](https://arxiv.org/abs/2506.09993)
[18](https://ieeexplore.ieee.org/document/11226569/)
[19](https://arxiv.org/abs/2505.24406)
[20](https://arxiv.org/abs/2503.22563)
[21](https://ieeexplore.ieee.org/document/11112226/)
[22](https://arxiv.org/abs/2510.25420)
[23](https://ieeexplore.ieee.org/document/11063907/)
[24](http://arxiv.org/pdf/2409.10353.pdf)
[25](http://arxiv.org/pdf/2407.03636.pdf)
[26](https://arxiv.org/html/2402.16907v2)
[27](https://arxiv.org/pdf/2308.09388.pdf)
[28](https://arxiv.org/html/2407.10833v1)
[29](https://arxiv.org/pdf/2311.14760.pdf)
[30](https://s-space.snu.ac.kr/handle/10371/210459)
[31](https://arxiv.org/abs/2212.00490)
[32](https://www.emergentmind.com/topics/diffusion-posterior-sampling-dps)
[33](https://openaccess.thecvf.com/content/CVPR2025/papers/Luo_Visual-Instructed_Degradation_Diffusion_for_All-in-One_Image_Restoration_CVPR_2025_paper.pdf)
[34](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0331465)
[35](https://ieeexplore.ieee.org/document/10642341/)
[36](https://ieeexplore.ieee.org/document/10643973/)
[37](https://ieeexplore.ieee.org/document/10657237/)
[38](https://ieeexplore.ieee.org/document/10446692/)
[39](https://ieeexplore.ieee.org/document/10843251/)
[40](https://ieeexplore.ieee.org/document/10365517/)
[41](https://arxiv.org/abs/2303.15770)
[42](https://ieeexplore.ieee.org/document/10445700/)
[43](http://arxiv.org/pdf/2312.16519.pdf)
[44](https://arxiv.org/html/2312.17161v2)
[45](https://arxiv.org/html/2412.12550v1)
[46](https://arxiv.org/pdf/2303.14353.pdf)
[47](https://arxiv.org/abs/2309.10714)
[48](https://www.e3s-conferences.org/10.1051/e3sconf/202561602026)
[49](http://arxiv.org/pdf/2311.14900v2.pdf)
[50](https://wyhuai.github.io/ddnm.io/)
[51](https://papers.neurips.cc/paper_files/paper/2023/file/5cebc89b113920dbff7c79854ba765a3-Paper-Conference.pdf)
[52](https://openaccess.thecvf.com/content/ICCV2025/papers/Thomas_Whats_in_a_Latent_Leveraging_Diffusion_Latent_Space_for_Domain_ICCV_2025_paper.pdf)
[53](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/ddnm/)
[54](https://openreview.net/forum?id=KTrnOhAN4k)
[55](https://www.youtube.com/watch?v=Mq-_PImmuy0)
