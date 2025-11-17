
# DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps

## 1. 논문의 핵심 주장 및 기여

DPM-Solver는 **확산 확률 모델(Diffusion Probabilistic Models, DPMs)의 샘플링 속도 개선**이라는 중대한 문제를 해결하는 연구입니다. 핵심 주장은 다음과 같습니다:[1]

**주요 기여:**
- DPM의 샘플링 과정을 상미분 방정식(ODE) 문제로 보고 **반선형(semi-linear) ODE 구조**를 명시적으로 활용하는 새로운 관점 제시[1]
- 확산 ODE의 정확한 해(exact solution)를 수학적으로 도출하고, 선형 부분을 해석적으로 계산하는 방법 제안[1]
- **지수 가중 적분(exponentially weighted integral)** 형태로 해를 단순화하여 효율적인 근사가 가능하도록 함[1]
- 수렴 차수 보장이 있는 1차, 2차, 3차 고차 솔버(DPM-Solver-1/2/3) 개발[1]
- **훈련 없이** 기존의 모든 사전 학습된 DPM에 적용 가능한 플러그-앤-플레이 방식 제시[1]

## 2. 해결하는 문제와 제안 방법

### 2.1 문제 정의

DPM의 샘플링 문제는 다음과 같습니다:[1]
- 고품질 샘플 생성을 위해 **100~1000번의 신경망 함수 평가(NFE)**가 필요
- GAN이나 VAE에 비해 월등히 느린 샘플링 속도
- 기존의 일반 목적 ODE 솔버(예: RK45)는 few-step(약 10단계) 영역에서 수렴 실패

### 2.2 핵심 수학적 해결책

**Proposition 3.1: 확산 ODE의 정확한 해**

논문의 핵심 수식은 다음과 같습니다:[1]

초기값 $$x_s$$에서 시간 $$t$$까지의 확산 ODE 해는:

$$x_t = \frac{\alpha_t}{\alpha_s} x_s - \alpha_t \int_{\lambda_t}^{\lambda_s} e^{-\lambda} \hat{\epsilon}_\theta(\hat{x}_\lambda, \lambda) d\lambda$$

여기서 $$\lambda_t = \log(\alpha_t/\sigma_t)$$는 반-로그 신호-대-잡음 비(half-log SNR)입니다.[1]

**이 공식이 중요한 이유:**

1. **선형 부분의 해석적 계산**: 선형항 $$\frac{\alpha_t}{\alpha_s}$$는 정확하게 계산되며 이산화 오차 제거[1]
2. **변수 변환의 이점**: 기존의 시간 $$t$$에서 $$\lambda$$로 변환하면 계수 $$e^{-\lambda}$$가 노이즈 스케줄과 독립적[1]
3. **지수 가중 적분**: $$\int e^{-\lambda} \hat{\epsilon}_\theta d\lambda$$는 **지수 적분기(exponential integrator)** 문헌에서 잘 연구된 형태[1]

### 2.3 고차 솔버의 구성

**DPM-Solver-k (k=1,2,3)** 는 Taylor 전개를 이용하여 지수 가중 적분을 근사합니다:[1]

**DPM-Solver-1 (1차):**

$$\tilde{x}_{t_i} = \frac{\alpha_{t_i}}{\alpha_{t_{i-1}}} \tilde{x}_{t_{i-1}} - \sigma_{t_i}(e^{h_i} - 1) \epsilon_\theta(\tilde{x}_{t_{i-1}}, t_{i-1})$$

여기서 $$h_i = \lambda_{t_i} - \lambda_{t_{i-1}}$$입니다.[1]

**Theorem 3.2: 수렴 차수 보장**

DPM-Solver-k는 $$k$$차 수렴성을 갖습니다:[1]

$$\tilde{x}_{t_M} - x_0 = O(h_{\max}^k), \quad h_{\max} = \max_{1 \leq i \leq M}(\lambda_{t_i} - \lambda_{t_{i-1}})$$

### 2.4 모델 구조

DPM-Solver는 다음과 같은 주요 특성을 갖습니다:[1]

| 특성 | 설명 |
|------|------|
| **구조** | 반선형 ODE를 위해 최적화된 전용 솔버 |
| **적용 범위** | 연속 시간/이산 시간 DPM 모두 지원[1] |
| **추가 훈련** | 필요 없음 (훈련 무료)[1] |
| **단계 선택** | 균일한 $$\lambda$$ 공간 분할 또는 적응형 단계 크기[1] |
| **조건부 샘플링** | 분류기 가이드와 호환[1] |

## 3. 성능 향상 및 실험 결과

### 3.1 주요 성능 지표

**CIFAR-10 데이터셋에서의 FID 점수 (낮을수록 좋음):**[1]

| NFE | DPM-Solver (ODE) | DDIM | Euler (SDE) |
|-----|-----------------|------|------------|
| 10 | 4.70 | ~15-20 | ~30-50 |
| 20 | 2.87 | ~5-8 | ~8-12 |
| 50 | 2.25 | ~3 | ~3-4 |

**성능 개선:**[1]
- 10 NFE에서 **DDIM 대비 약 4배 개선**
- 이전의 최고 성능 훈련 무료 샘플러 대비 **4~16배 속도 향상**
- ImageNet 256×256에서 정성적 우월성 입증

### 3.2 데이터셋별 결과

논문은 광범위한 실험을 수행했습니다:[1]
- **CIFAR-10**: 연속/이산 시간 모두 테스트
- **CelebA 64×64**: 선형 노이즈 스케줄
- **ImageNet 64×64, 128×128**: 코사인 노이즈 스케줄
- **LSUN bedroom 256×256**: 고해상도 이미지 생성

### 3.3 RK 방법과의 비교

**Table 1: RK vs DPM-Solver 비교 (CIFAR-10):**[1]

| 방법 | NFE=12 | NFE=18 | NFE=24 | NFE=30 |
|------|--------|--------|--------|--------|
| RK2(λ) | 107.81 | 42.04 | 17.71 | 7.65 |
| **DPM-Solver-2** | **5.28** | **3.43** | **3.02** | **2.85** |
| RK3(λ) | 34.29 | 4.90 | 3.50 | 3.03 |
| **DPM-Solver-3** | **6.03** | **2.90** | **2.75** | **2.70** |

DPM-Solver가 동일 차수의 RK 방법보다 **일관되게 우수한 성능**을 보입니다.[1]

## 4. 모델 일반화 성능 향상 관련 내용

### 4.1 논문에서 다루는 일반화 측면

**노이즈 스케줄 독립성:**
논문의 중요한 발견은 Proposition 3.1의 공식이 **특정 노이즈 스케줄에 독립적**이라는 점입니다. $$\lambda$$ 공간에서의 적분 형태로 표현되면:[1]

$$\hat{x}_{\lambda_t} = \frac{\hat{\alpha}_{\lambda_t}}{\hat{\alpha}_{\lambda_s}} \hat{x}_{\lambda_s} - \hat{\alpha}_{\lambda_t} \int_{\lambda_t}^{\lambda_s} e^{-\lambda} \hat{\epsilon}_\theta(\hat{x}_\lambda, \lambda) d\lambda$$

이는 다양한 노이즈 스케줄(선형, 코사인 등)에 대해 **동일한 샘플링 알고리즘**을 적용할 수 있음을 의미합니다.[1]

**이산 시간 DPM 처리:**
논문은 이산 시간 DPM을 연속 시간 모델로 변환하여 처리하는 방법을 제시합니다:[1]

$$\epsilon_\theta(x, t) := \tilde{\epsilon}_\theta\left(x, \frac{(N-1)t}{T}\right)$$

이를 통해 **사전 학습된 모든 DPM에 적용 가능한 보편성**을 확보합니다.

### 4.2 최신 연구에서의 일반화 성능 개선 관점

최근 연구들이 보여주는 일반화 성능 향상 기법들:[2][3][4]

**1. 다양한 데이터로 학습:**[5]
- 광범위한 스타일, 객체, 맥락을 포함한 데이터셋에서 학습하면 모델의 표현 다양성 증대
- 텍스트-이미지 모델의 경우 다양한 프롬프트와 이미지 쌍으로 학습

**2. 아키텍처 개선:**[5]
- 주의 메커니즘(Attention) 사용으로 관련 부분에 집중
- 교차-주의(Cross-attention) 레이어로 텍스트-이미지 정렬 개선
- EMA(Exponential Moving Average) 사용으로 가중치 안정화

**3. 정규화 기법:**[5]
- Dropout, 가중치 감소로 과적합 방지
- 노이즈 주입으로 모델 강건성 증대
- 동적 임계처리로 극값 처리

**4. 앙상블 방법:**[5]
- 여러 모델의 예측 결합으로 편향 감소
- 배깅/부스팅으로 다양한 데이터 부분집합 사용

### 4.3 DPM-Solver의 일반화에 대한 기여

**암시적 일반화 개선:**

1. **단계 수 유연성**: DPM-Solver는 **다양한 NFE 예산에서 작동**하며, 모델 재훈련 없이 적용 가능[1]

2. **노이즈 스케줄 무시성**: Proposition 3.1의 특성상 **다양한 노이즈 스케줄을 통일적으로 처리**하므로 일반화 향상[1]

3. **조건부 샘플링**: 분류기 가이드와 호환되어 **텍스트-이미지, 클래스 조건부 생성 모두 지원**[1]

## 5. 모델의 한계

### 5.1 논문에서 명시한 한계

논문의 "한계와 광범위한 영향" 섹션에서 다음을 언급합니다:[1]

**1. 우도 평가 가속화 불가:**
- DPM-Solver는 샘플링 가속화에만 초점
- 우도(likelihood) 계산 가속화는 지원하지 않음

**2. 실시간 응용 부족:**
- GAN과 비교하면 여전히 느림
- 실시간 응용(예: 로봇 제어)에는 부적합

**3. 악의적 사용 우려:**
- 깊은 생성 모델의 잠재적 오용 가능성

### 5.2 기술적 한계

**1. 4차 이상 솔버의 제한:**[1]
- 4차 이상의 높은 차수 솔버는 더 많은 중간 점이 필요
- 계산 복잡도 급증으로 실용성 부족

**2. 신경망 평활성 가정:**[1]
- 정리 3.2의 증명은 $$\epsilon_\theta$$의 연속성과 미분가능성 가정
- 실제 신경망이 이를 완벽히 만족하지 않을 수 있음

**3. 스텝 크기 선택:**[1]
- 균일 $$\lambda$$ 분할은 启발식이며 최적이 아닐 수 있음
- 적응형 알고리즘도 하이퍼파라미터 의존성 존재

## 6. 향후 연구 방향 및 영향

### 6.1 DPM-Solver의 학문적 영향

**NeurIPS 2022 인정:**[6][1]
- 1982회 인용 (2024년 기준) - 최상위 논문
- 구글의 Diffusers 라이브러리에 공식 지원
- Stable Diffusion, DeepFloyd-IF 등 주요 모델에 통합[6]

### 6.2 후속 연구와 개선

**1. DPM-Solver++ (2023):**[7]
- 가이드된 샘플링(Guided sampling) 최적화
- 큰 가이드 스케일에서 성능 개선
- 100~250 단계에서 DDIM 대비 **유사 FID로 더 적은 단계 필요**

**2. DPM-Solver-v3 (2023):**[8][9]
- **최적 파라미터화** 공식화
- 경험적 모델 통계(Empirical Model Statistics, EMS) 도입
- 다중 단계 메서드 및 예측-보정 프레임워크 적용
- 5~10 NFE에서 **일관되게 개선된 성능**: CIFAR-10 12.21 FID (5 NFE), 2.51 FID (10 NFE)[9][8]

**3. UniPC (2023):**[4]
- 통합 예측-보정 프레임워크
- DPM-Solver-v3과 경쟁 관계로 나타남

**4. 개선된 ODE 해 분석 (2023):**[10]
- 정제된 지수 솔버(Refined Exponential Solver, RES) 제안
- 모든 차수 조건 충족으로 **수치 결함 감소**

**5. 최신 고급 방법들 (2024-2025):**

| 방법 | 개선 사항 | 성능 |
|------|---------|------|
| **SA-Solver** (2024)[11] | Adams 방법 기반, 확률론적 샘플링 | 결정론적 DDIM 개선 |
| **TADA** (2025)[12] | 고차원 초기 잡음 사용 | ImageNet512에서 이전 SOTA 대비 186% 빠름 |
| **DC-Solver** (2025)[13] | 동적 보정으로 동적 보정 편향 완화 | CFG 안정성 개선 |

### 6.3 일반화 성능 관련 최신 연구 (2024-2025)

**1. 확산 모델의 일반화 이론:**[14][15]
- 일반화 격차 상한: $$O(n^{-2/5} + m^{-4/5})$$ (표본 크기 $$n$$, 모델 용량 $$m$$)[15][14]
- 차원의 저주 회피 증명[14][15]
- 조기 종료를 통한 일반화 보장[14]

**2. 확산 모델의 강건성:**[16][17]
- 적대적 공격에 대한 인증된 강건성[16]
- 분포 외(OOD) 표본 처리 능력 증명[16]

**3. 메커니즘 이해:**[17]
- **국소화된 제거 작업**이 일반화 메커니즘
- 국소적 경험적 제거기 통합 알고리즘 제안[17]

**4. 분포 외 일반화:**[18][15]
- 분산 로버스트 최적화(DRO) 접근[18]
- 확산 기반 불확실성 집합 설계[18]
- OOD 성능 향상 증명[18]

### 6.4 구조 보존 방법의 중요성

최근 연구는 **반선형 구조 활용의 효과성**을 재확인합니다:[19][20][12]

- **확률론적 지수 적분기**(Probabilistic Exponential Integrators): 반선형 ODE에서 선형 부분을 정확히 해결하는 확률론적 방법 제시[20]
- **고차원 NODE**: 지수 적분기 활용으로 암시적 방법보다 안정적[19]
- **구조 보존 신경 ODE**: 선형-비선형 분해로 학습과 배포 성능 개선[19]

### 6.5 향후 고려사항

**1. 모델 파라미터화:**
- $$\epsilon$$-예측 vs $$x_0$$-예측 vs $$v$$-예측의 최적 선택[21]
- DPM-Solver-v3의 경험적 모델 통계 방식 확산

**2. 적응형 시간 단계:**
- 더 정교한 오차 추정 기반 알고리즘 개발[22]
- 희소 샘플링 기법 활용

**3. 고해상도 생성:**
- 잠재 공간 DPM 통합
- 정확도와 계산 효율성 트레이드오프 분석

**4. 조건부 생성 최적화:**
- 큰 가이드 스케일에서의 비선형 효과 처리[23]
- 특성 가이드 등 새로운 가이드 방법 개발

**5. 이론적 이해 심화:**
- Wasserstein 거리 기반 수렴 분석[24]
- Heun 샘플러 등 새로운 솔버의 수렴 보증[24]
- 비선형 근사의 최적성 조건 규명

## 결론

DPM-Solver는 **반선형 ODE 구조의 명시적 활용**이라는 우아한 수학적 통찰을 통해 확산 모델 샘플링의 효율성을 혁신적으로 개선했습니다. 노이즈 스케줄 무시성과 훈련 무료 성질은 모든 기존 DPM에 즉시 적용 가능한 보편적 방법을 제공합니다.[1]

최근 2년간의 연속적인 개선(DPM-Solver++, v3, UniPC 등)은 이 기초 위에서 **최적 파라미터화 공식화**, **경험적 모델 통계 도입**, **다중 단계-예측-보정 프레임워크 개발**로 발전했습니다. 동시에 이론적으로는 **일반화 한계 증명**, **강건성 인증**, **구조 보존 방법의 효과성 재확인**이 이루어지고 있습니다.[8][9][15][17][14][16]

향후 연구의 중요 방향은 **초소수 단계 영역(5-10 NFE)의 지속적 개선**, **조건부 생성의 고해상도 확장**, **확률론적 샘플링과의 통합**, 그리고 **수렴성과 일반화 성능의 더 깊은 이론적 이해**입니다.

***

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/dffb892a-4fab-4701-90a7-e98a788c8765/2206.00927v3.pdf)
[2](https://arxiv.org/abs/2206.00927)
[3](https://arxiv.org/pdf/2106.00132.pdf)
[4](https://arxiv.org/pdf/2302.04867v2.pdf)
[5](https://zilliz.com/ai-faq/what-techniques-help-improve-the-generalization-of-diffusion-models)
[6](https://github.com/LuChengTHU/dpm-solver)
[7](https://arxiv.org/abs/2211.01095)
[8](https://openreview.net/forum?id=9fWKExmKa0)
[9](https://proceedings.neurips.cc/paper_files/paper/2023/file/ada8de994b46571bdcd7eeff2d3f9cff-Paper-Conference.pdf)
[10](https://arxiv.org/pdf/2308.02157.pdf)
[11](https://arxiv.org/abs/2309.05019)
[12](https://machinelearning.apple.com/research/tada)
[13](https://arxiv.org/html/2409.03755)
[14](https://proceedings.neurips.cc/paper_files/paper/2023/file/06abed94583030dd50abe6767bd643b1-Paper-Conference.pdf)
[15](https://arxiv.org/pdf/2311.01797.pdf)
[16](http://arxiv.org/pdf/2402.02316.pdf)
[17](https://arxiv.org/html/2411.19339v2)
[18](https://arxiv.org/html/2510.22757v1)
[19](https://arxiv.org/html/2503.01775v3)
[20](https://proceedings.neurips.cc/paper_files/paper/2023/file/7f64034009f4a5fa417a57e1a987c5cd-Paper-Conference.pdf)
[21](https://apxml.com/courses/advanced-diffusion-architectures/chapter-4-advanced-diffusion-training/model-parameterization)
[22](https://arxiv.org/html/2504.01855v1)
[23](http://arxiv.org/pdf/2312.07586v5.pdf)
[24](https://arxiv.org/abs/2508.03210)
[25](https://arxiv.org/pdf/2305.14267.pdf)
[26](https://arxiv.org/abs/2312.07243)
[27](https://milvus.io/ai-quick-reference/what-techniques-help-improve-the-generalization-of-diffusion-models)
[28](https://proceedings.neurips.cc/paper_files/paper/2022/file/260a14acce2a89dad36adc8eefe7c59e-Paper-Conference.pdf)
[29](https://pubs.aip.org/aip/cha/article/35/3/033154/3340549/Training-stiff-neural-ordinary-differential)
[30](https://proceedings.neurips.cc/paper_files/paper/2024/hash/47ee3941a6f1d23c39b788e0f450e2a7-Abstract-Conference.html)
[31](https://proceedings.iclr.cc/paper_files/paper/2025/hash/dcd9ed95f4abeaa1c22c4c2fd4231930-Abstract-Conference.html)
[32](https://proceedings.mlr.press/v235/li24ad.html)
[33](https://openaccess.thecvf.com/content/CVPR2024/papers/Xue_Accelerating_Diffusion_Sampling_with_Optimized_Time_Steps_CVPR_2024_paper.pdf)
[34](https://ml.cs.tsinghua.edu.cn/dpmv3/)
[35](http://arxiv.org/pdf/2412.17162.pdf)
[36](https://arxiv.org/html/2310.05264v3)
[37](http://arxiv.org/pdf/2308.14333.pdf)
[38](http://arxiv.org/pdf/2403.06392.pdf)
[39](http://arxiv.org/pdf/2410.11795.pdf)
[40](https://arxiv.org/html/2503.13868v3)
[41](https://proceedings.mlr.press/v162/schotthofer22a/schotthofer22a.pdf)
[42](https://arxiv.org/html/2502.04669v1)
[43](https://openreview.net/pdf?id=1DVgysiIt7)
[44](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/15384A9F2776B2D1C1F1D3CDA390D779/S0956792521000139a.pdf/structurepreserving_deep_learning.pdf)
[45](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)
[46](https://www.ijcai.org/proceedings/2025/0764.pdf)
