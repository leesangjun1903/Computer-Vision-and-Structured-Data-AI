# Pseudo Numerical Methods for Diffusion Models on Manifolds

### 1. 핵심 주장과 주요 기여

이 논문은 **Denoising Diffusion Probabilistic Models (DDPMs)의 추론 과정을 매니폴드 상에서 미분방정식을 풀이하는 문제로 재해석**하고, 이를 바탕으로 **Pseudo Numerical Methods for Diffusion Models (PNDMs)**을 제안합니다.[1]

핵심 주장은 다음과 같습니다:
- 기존의 고전적 수치해석 방법(Runge-Kutta, 선형 다중단계법)을 DDPM의 역확산 과정에 직접 적용하면 샘플이 데이터의 주 분포 영역을 벗어나 새로운 노이즈가 발생한다는 것을 발견
- 이 문제를 해결하기 위해 **비선형 전달부(transfer part)와 그래디언트부(gradient part)로 수치 방법을 분해**하고, DDIMs가 실제로는 특수한 의사 수치 방법임을 보여줌

주요 기여는:[2][1]
- **20배 가속화 달성**: 1000 단계 DDIM과 동등한 품질을 50 단계 PNDM으로 생성
- **품질 향상**: 250 단계에서 FID를 약 0.4 포인트 개선하여 CelebA에서 새로운 SOTA 달성 (FID 2.71)
- **이론적 증명**: PNDMs가 2차 수렴성을 가지며 DDIM은 1차 수렴성을 가짐을 증명

***

### 2. 문제 정의, 방법론 및 모델 구조

#### 문제 정의

논문이 해결하고자 하는 핵심 문제는 **DDPM의 느린 추론 속도**입니다. 기존 가속화 방법들(분산 스케줄 조정, DDIM)은 다음과 같은 한계가 있었습니다:[1]
- 고속화 비율이 증가하면 샘플 품질이 급격히 저하됨
- 고차 수치 방법(Runge-Kutta)을 사용하면 오히려 DDIM보다 성능이 나쁨
- 이론적으로 약한 설명만 가능

#### 핵심 방법론

**1) 미분방정식 유도**

논문은 DDIM의 역확산 과정을 연속 미분방정식으로 변환합니다. 이산 형태에서 시작하여:

$$x_{t-\delta} - x_t = (\bar{\alpha}_{t-\delta} - \bar{\alpha}_t)\left(\frac{x_t}{\sqrt{\bar{\alpha}_t}(\sqrt{\bar{\alpha}_{t-\delta}} + \sqrt{\bar{\alpha}_t})} - \frac{\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}(\sqrt{(1-\bar{\alpha}_{t-\delta})\bar{\alpha}_t} + \sqrt{(1-\bar{\alpha}_t)\bar{\alpha}_{t-\delta}})}\right)$$

이를 연속화하면:[2]

$$\frac{dx}{dt} = -\bar{\alpha}'(t)\left(\frac{x(t)}{2\bar{\alpha}(t)} - \frac{\epsilon_\theta(x(t), t)}{2\bar{\alpha}(t)\sqrt{1-\bar{\alpha}(t)}}\right)$$

**2) 고전 수치 방법의 문제점 분석**

두 가지 주요 문제를 발견:[2]
- **첫 번째**: 신경망 $\epsilon_\theta$가 잘 정의되는 영역(데이터의 고밀도 영역)이 제한적인데, 고전 수치 방법은 직선을 따라 샘플을 생성하여 이 영역을 벗어남
- **두 번째**: 대부분의 선형 분산 스케줄에서 방정식이 $t \to 0$일 때 무한대로 발산하여 수치 방법의 필요조건을 만족하지 않음

**3) 의사 수치 방법의 설계**

수치 방법을 두 부분으로 분해:

- **전달부(Transfer part)**: 

$$\varphi(x_t, \epsilon_t, t, t-\delta) = \sqrt{\frac{\bar{\alpha}_{t-\delta}}{\bar{\alpha}_t}} x_t - \frac{(\bar{\alpha}_{t-\delta} - \bar{\alpha}_t)}{\sqrt{\bar{\alpha}_t}(\sqrt{(1-\bar{\alpha}_{t-\delta})\bar{\alpha}_t} + \sqrt{(1-\bar{\alpha}_t)\bar{\alpha}_{t-\delta}})} \epsilon_t$$

이는 정확한 노이즈 예측이 주어지면 정확한 $x_{t-\delta}$를 생성하는 성질을 가짐 (Property 3.1)[2]

- **그래디언트부**: 선형 다중단계 방법의 그래디언트 부분

$$e'_t = \frac{1}{24}(55e_t - 59e_{t-\delta} + 37e_{t-2\delta} - 9e_{t-3\delta})$$

#### 모델 구조

**F-PNDM (Full PNDM)**:[2]

```
1. 초기 3 단계: Pseudo Runge-Kutta 방법 사용
   - 각 단계에서 신경망 호출 4회
   
2. 나머지 단계: Pseudo 선형 다중단계 방법 사용
   - 신경망 호출 1회, 이전 4단계 정보 재사용
```

**S-PNDM (Second-order PNDM)**:
더 간단한 2차 방법으로, 초기 단계에 Pseudo 개선된 오일러 방법 사용

**중요한 성질** (Property 3.2):[2]
- S/F-PNDM의 국소 오차: $O(\delta^3)$
- S/F-PNDM의 전역 오차 (수렴 차수): $O(\delta^2)$ - 2차 수렴
- DDIM의 수렴 차수: $O(\delta)$ - 1차 수렴

***

### 3. 성능 향상 및 일반화 능력

#### 성능 향상 분석

**정량적 성능 개선**:[2]

| 데이터셋 | 방법 | 50 단계 FID | 250 단계 FID | 시간(초/단계) |
|---------|------|-----------|-------------|------------|
| CIFAR-10 | DDIM | 6.99 | 4.52 | 0.337 |
| CIFAR-10 | F-PNDM | 3.95 | 3.60 | 0.391 |
| CelebA | DDIM | 8.95 | 4.44 | 1.237 |
| CelebA | F-PNDM | 3.34 | 2.71 | 1.433 |

핵심 성과:
- **50 단계 F-PNDM이 1000 단계 DDIM과 동등 품질 달성** (20배 가속)
- **250 단계에서 0.4 FID 개선** (이전 SOTA 대비)
- **계산 비용 증가 미미** (단계당 약 16% 오버헤드, 단계 수 감소로 보상)

#### 일반화 성능 향상

논문은 다양한 분산 스케줄에 대한 강건성을 시연합니다:[2]

**선형 vs. 코사인 분산 스케줄 테스트**:
- 선형 스케줄: F-PNDM이 일관되게 우수한 성능
- 코사인 스케줄: 더 큰 단계에서 코사인이 이득이지만, F-PNDM은 여전히 경쟁력 있음
- **결론**: PNDM은 다양한 분산 스케줄에 대해 일반화 성능이 우수함

**시각화를 통한 매니폴드 검증**:[2]

논문은 생성 과정에서 샘플의 노름(norm) 변화를 추적합니다:
- **FON (고전 4차 수치 방법)**: 샘플이 데이터 고밀도 영역을 벗어남 → 노이즈 증가
- **PNDM**: 샘플이 고밀도 영역 내에 머물러 있음 → 안정적인 생성
- **DDIM**: PNDM과 유사하게 안정적이지만, 2차 수렴성의 이득이 부족

#### 이론적 수렴 보증

**수렴 차수 분석**:[2]

DDIM의 경우:

$$x(t+\delta) - x_{DDIM}(t+\delta) = O(\delta^2)$$

→ 1차 수렴 (글로벌 오차 $O(\delta)$ )

S/F-PNDM의 경우:

$$x(t+\delta) - x_{S/F-PNDM}(t+\delta) = O(\delta^3)$$

→ 2차 수렴 (글로벌 오차 $O(\delta^2)$ )

이는 **동일한 오차를 유지하면서 단계 수를 $\sqrt{2}$배 감소** 가능함을 의미합니다.

***

### 4. 한계 및 문제점

논문에서 명시한 한계:[2]

1. **분산 스케줄 의존성**: 
   - F-PNDM은 4개 단계의 정보를 사용하므로 스케줄의 부드러움이 중요
   - 최적의 분산 스케줄이 아직 발견되지 않음

2. **수렴 차수의 이론적 제약**:
   - 전달부의 변경으로 인한 오차가 이론적으로는 존재하지만 실험적으로는 영향 최소화
   - 더 높은 차수의 수렴성 달성에 어려움

3. **초기 단계의 계산 비효율**:
   - F-PNDM은 초기 3단계에서 Runge-Kutta 사용으로 인해 초기에는 S-PNDM보다 느림
   - 충분한 단계 수가 필요할 때 장점 발휘

***

### 5. 최신 연구 동향 및 영향

PNDM이 발표된 이후(2022년 ICLR), 확산 모델 가속화 분야에서 다음과 같은 진전이 있었습니다:[3][4][5][6][7][8]

**가속화 연구의 발전 방향**:

1. **수치적 방법 개선** (2024-2025):
   - **A-FloPS** (2025): Flow Matching으로 매니폴드 재매개변수화 + 적응형 속도 분해 → PNDM보다 한 단계 진화
   - **DPM-Solver++, DPM-Solver** 개선: 고차 수치 방법의 안정성 증대

2. **적응형 계산 할당**:
   - **AdaDiff** (2024): 각 단계의 중요도에 따라 동적으로 계산 자원 할당 → 에너지 효율성 증대

3. **병렬화 및 분포 매칭**:
   - **병렬 확산 샘플링** (2024): 자동회귀적 특성 극복을 통한 병렬 처리
   - **비디오 확산 모델 가속** (2024): 분포 매칭을 통한 품질 유지

#### PNDM의 현재 영향도

최신 SOTA 모델들이 PNDM의 아이디어를 계승합니다:[3]

- **이론적 토대 제공**: 확산 모델 = ODE 풀이 관점의 정립
- **매니폴드 개념 도입**: 고밀도 영역 내 샘플링의 중요성 강조
- **하이브리드 방법론**: 고전 수치해석 + 신경망 기반 방법 결합의 모범 사례

**인용 현황**: 816회 인용 (Google Scholar, 2024년 기준)[9]

***

### 6. 향후 연구 시 고려사항

논문에서 제시한 미해결 과제 및 권장 사항:[2]

**1. 최적 분산 스케줄 탐색**
- 현재: 선형/코사인 스케줄에서만 테스트
- 향후 방향: PNDM에 특화된 분산 스케줄 개발

**2. 고차 수렴성 달성**
- 이론적 한계: 전달부의 오차로 인한 수렴 차수 제약
- 가능한 접근: 전달부를 더 높은 차수로 근사화하는 방법 탐색

**3. 확장 가능성**
- **신경 ODE**: 다른 신경 미분방정식 기반 모델로의 확장
- **조건부 생성**: 텍스트-이미지, 이미지-이미지 등 조건부 작업 확대
- **이산 확산 모델**: 텍스트, 그래프 등 이산 데이터 도메인 적용

**4. 최신 관련 연구 (2024-2025)**

**일반화 성능 향상 연구**:[10][11][12]

- **정보 이론적 분석** (2025): VAE/DM의 통일된 일반화 성능 분석 프레임워크 개발
  - DM의 일반화 오차: $O(n^{-2/5} + m^{-4/5})$ (차원 저주 회피)
  - 확산 단계 T에 따른 명시적 트레이드오프 규명

- **모델 재현성과 일반화의 관계** (2024): 서로 다른 DM이 같은 생성 결과를 낳는 "모델 재현성"이 일반화의 지표임을 발견
  - 메모리화 영역 vs. 일반화 영역의 전이 현상
  - 저차원 매니폴드 학습이 고차원 공간에서 차원 저주 회피를 가능하게 함

***

## 결론

**PNDMs은 확산 모델의 가속화 문제에 대한 패러다임 전환**을 제시합니다. 고전 수치해석의 프레임워크를 신경 생성 모델의 특성(매니폴드 구조)에 맞게 재설계함으로써 **20배 속도 향상과 동시에 샘플 품질 개선**을 달성했습니다. 

이후의 연구들은 PNDM의 핵심 인사이트(매니폴드 기반 샘플링, 방정식 분해, 수치적 안정성)를 기반으로 더욱 정교한 가속화 기법과 일반화 이론을 개발하고 있습니다. 특히 **적응형 방법**, **흐름 매칭 재매개변수화**, **정보 이론적 일반화 분석**이 차세대 연구의 중심이 되고 있습니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/06be4b60-f9b0-4c0f-857f-f8887b797056/2202.09778v2.pdf)
[2](http://arxiv.org/pdf/2503.07699.pdf)
[3](https://arxiv.org/pdf/2403.03852.pdf)
[4](http://arxiv.org/pdf/2309.17074.pdf)
[5](https://arxiv.org/html/2402.09970)
[6](https://arxiv.org/pdf/2312.09193.pdf)
[7](http://arxiv.org/pdf/2502.10389.pdf)
[8](https://arxiv.org/html/2509.00036v1)
[9](https://arxiv.org/abs/2202.09778)
[10](https://proceedings.neurips.cc/paper_files/paper/2023/file/06abed94583030dd50abe6767bd643b1-Supplemental-Conference.pdf)
[11](https://arxiv.org/abs/2506.00849)
[12](https://www.siam.org/publications/siam-news/articles/generalization-of-diffusion-models-principles-theory-and-implications/)
[13](https://arxiv.org/html/2309.10438)
[14](https://arxiv.org/html/2412.05899)
[15](https://openreview.net/pdf/08c071f50f706076294158af529e4f1a2556df41.pdf)
[16](https://proceedings.mlr.press/v235/tang24f.html)
[17](https://arxiv.org/html/2410.15248v1)
[18](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/t-stitch/)
[19](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_From_Reusing_to_Forecasting_Accelerating_Diffusion_Models_with_TaylorSeers_ICCV_2025_paper.pdf)
