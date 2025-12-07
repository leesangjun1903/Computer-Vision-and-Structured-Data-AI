# Don't Play Favorites: Minority Guidance for Diffusion Models

## 핵심 주장과 주요 기여

본 논문은 확산 모델(diffusion models)이 데이터 분포의 **고밀도 영역(majority)에 편향**되어 저밀도 영역(minority) 샘플 생성에 어려움을 겪는다는 근본적 문제를 제기하고, 이를 해결하는 새로운 프레임워크를 제안합니다. 핵심 기여는 세 가지로 요약됩니다:[1]

1. **Minority Score**: Tweedie's formula의 편향성에 착안하여 샘플의 고유성(uniqueness)을 정량화하는 새로운 지표를 정의
2. **Minority Guidance**: 레이블 없이도 minority 샘플 생성을 위한 샘플링 기법 개발
3. **이론적 분석**: Tweedie-based denoiser의 majority 편향성을 기하학적으로 증명하고, 이를 극복하는 방법론 제시

특히, 기존 연구들이 레이블 의존적이거나 특정 모달리티에 제한적이었던 한계를 극복하여 **비지도 및 조건부 생성 모두에 적용 가능**한 범용적 접근법을 제시했다는 점이 학술적 의의를 갖습니다.[1]

## 해결하고자 하는 문제

### 문제 정의
대규모 데이터셋은 일반적으로 **롱테일 분포(long-tailed distribution)** 를 따릅니다. 다수의 샘플이 데이터 매니폴드의 고밀도 영역에 집중되어 있으며, 저밀도 영역에는 드물게 관측되는 특성을 포함한 **minority 샘플**이 존재합니다. 이러한 minority 샘플은 의료 진단(희귀 질환), 창의적 AI, 공정성 향상 등 중요한 응용 분야에서 결정적 역할을 합니다.[1]

그러나 확산 모델의 생성 과정은 **역확산 과정(reverse diffusion process)을 시뮬레이션**하는 특성상, 높은 우도(likelihood)를 가진 majority 샘플을 더 자주 생성하는 **majority-oriented** 성향을 보입니다. 이는 다음과 같은 문제를 야기합니다:[1]

- Minority 샘플 생성에 막대한 시간과 계산 리소스 소요
- 의도적 큐레이션 없이는 효과적인 minority 샘플 수집 불가능
- 기존 생성기들은 minority 특성을 무시하고 평균적인 특성으로 대체하는 편향성

### 기존 방법의 한계
- **Sehwag et al. (2022)**: 클래스 조건부 설정에만 적용 가능하여 레이블 없는 데이터에는 사용 불가[1]
- **Lee et al. (2022)**: 그래프 데이터 같은 특정 모달리티에 맞춤화됨[1]
- **Yu et al. (2020); Lin et al. (2022)**: minority 레이블을 필요로 하며, 레이블 획득이 어려운 실제 상황에서 한계[1]

## 제안하는 방법: Minority Guidance Framework

### 1. Tweedie's Formula의 편향성 분석

DDPM의 포워드 과정은 다음과 같이 정의됩니다:

$$ q_{\alpha_t}(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\alpha_t}x_0, (1-\alpha_t)I) $$

Tweedie's formula를 통해 노이즈가 첨가된 샘플 \(x_t\)로부터 깨끗한 샘플 \(x_0\)의 사후평균을 추정할 수 있습니다:

$$ \hat{x}_0 := \mathbb{E}[x_0 | x_t] = \frac{1}{\sqrt{\alpha_t}}(x_t + (1-\alpha_t)s_\theta(x_t, t)) $$

**문제점**: 이 식은 기대값(Expectation)을 사용하므로 다수의 majority 샘플이 평균에 크게 기여합니다. 따라서 minority 샘플은 **정보 손실**이 심각하게 발생합니다. 이는 강한 노이즈 첨가(t ≈ T)에서 더욱 심화되며, denoised 출력이 데이터셋 평균에 가까워집니다.[1]

Proposition 1은 최적 스코어 함수가 조건부 스코어의 평균을 취한다는 것을 증명합니다:

$$ s^*_\theta(x_t, t) = \mathbb{E}_{q_{\alpha_t}(x_0|x_t)}[\nabla_{x_t}\log q_{\alpha_t}(x_t | x_0)] $$

이는 기하학적으로 majority 샘플 방향으로 기울어진 방향을 생성하여 **본질적인 편향**을 보여줍니다.[1]

### 2. Minority Score 정의

Tweedie-based denoiser의 편향성을 활용하여, 원본 샘플 $\(x_0\)$ 와 복원된 샘플 $\(\hat{x}_0\)$ 간의 불일치를 측정하는 메트릭을 정의합니다:

$$ l(x_0; s_\theta) := \mathbb{E}_{q_{\alpha_t}(x_t|x_0)}[d(x_0, \hat{x}_0)] $$

여기서 $\(d(\cdot, \cdot)\)$ 는 LPIPS(Learned Perceptual Image Patch Similarity)나 제곱오차 같은 불일치 측정치입니다.[1]

**핵심 통찰**: Corollary 1은 제곱오차를 사용할 때 minority score가 노이즈 예측 오류와 동일(up to scaling)함을 증명합니다:

$$ l(x_0; s_\theta) = \tilde{w}_t \mathbb{E}_{p(\epsilon)}[\|\epsilon_\theta(x_t, t) - \epsilon\|_2^2] $$

$$ \text{where } \tilde{w}_t := \frac{1-\alpha_t}{\alpha_t} $$

이는 적은 양의 데이터로 학습된 minority 샘플이 더 높은 예측 오류를 보이므로 **높은 minority score**를 가짐을 의미합니다.[1]

### 3. Minority Guidance 샘플링

Minority score를 기반으로 분류기 가이던스(classifier guidance) 기법을 차용하여 생성 과정을 조작합니다:

**단계별 과정**:
1. 데이터셋 $\(\{x_0^{(i)}\}_{i=1}^N\)$ 에 대해 각 샘플의 minority score $\(l^{(i)}\)$ 계산
2. $\(L-1\)$ 개의 임계값을 사용하여 연속적인 score를 범주형 minority 클래스 $\(\tilde{l}^{(i)} \in \{0, ..., L-1\}\)$ 로 변환
3. 쌍으로 구성된 데이터셋 $\((x_0^{(i)}, \tilde{l}^{(i)})\)$ 로 노이즈 조건부 분류기 $\(p_\psi(\tilde{l}|x_t)\)$ 학습
4. 수정된 스코어 함수 정의:

$$ \hat{s}_\theta(x_t, t, \tilde{l}) := s_\theta(x_t, t) + w\nabla_{x_t}\log p_\psi(\tilde{l}|x_t) $$

여기서 $\(w\)$ 는 가이던스 강도를 제어하는 스케일링 팩터입니다.[1]

**생성 과정**: 수정된 스코어 $\(\hat{s}_\theta\)$ 를 사용하여 역확산 과정을 수행하면, 원하는 minority 수준 $\(\tilde{l}\)$ 에 조건부인 샘플을 생성할 수 있습니다:

$$ x_{t-1} = \frac{1}{\sqrt{1-\beta_t}}(x_t + \beta_t\hat{s}_\theta(x_t, t, \tilde{l})) + \beta_t z $$

**주요 장점**: 레이블 불필요(label-agnostic)하여 비지도 설정에서도 작동하며, **고유성 컨트롤 가능성(uniqueness-controllability)** 을 제공합니다.[1]

### 4. 모델 구조

**기반 아키텍처**: DDPM(Denoising Diffusion Probabilistic Models)을 기반으로 합니다.[1]

**구성 요소**:
1. **사전학습된 확산 모델**: DSM(Denoising Score Matching)으로 학습된 스코어 네트워크 $\(s_\theta(x_t, t)\)$
2. **Minority 분류기**: U-Net 인코더 기반의 노이즈 조건부 분류기 $\(p_\psi(\tilde{l}|x_t)\)$
3. **가이던스 통합 모듈**: 스코어 함수와 분류기 기울기를 혼합하는 가이던스 메커니즘

**구현 세부사항**:
- 노이즈 스케줄: cosine 또는 linear 스케줄 사용
- 시간 단계: 1000~4000단계 범위에서 학습
- 특징 추출: 대규모 데이터셋(LSUN-Bedrooms, ImageNet)의 경우 사전학습된 특징 추출기 사용하여 효율성 향상[1]

**클래스-조건부 확장**: 조건부 모델 $\(s_\theta(x_t, t, c)\)$ 에 대해 클래스별로 intra-class 방식으로 minority score를 계산하여 확장 가능합니다:

$$ \hat{s}_\theta(x_t, t, c, \tilde{l}) := s_\theta(x_t, t, c) + w\nabla_{x_t}\log p_\psi(\tilde{l}|x_t, c) $$

이는 기존 Sehwag et al. (2022)의 클래스 제한적 접근법을 일반화합니다.[1]

## 성능 향상 및 한계

### 성능 향상 검증

**실험 설정**: CelebA, CIFAR-10, LSUN-Bedrooms, ImageNet, 그리고 뇌 MRI 의료 영상 데이터셋에서 평가.[1]

**주요 결과**:
1. **Minority 샘플 생성 향상**: Average k-Nearest Neighbor(AvgkNN), Local Outlier Factor(LOF), Rarity Score 등의 저밀도 측정지표에서 모든 데이터셋에서 기존 DDPM 샘플러 대비 **20-50% 개선**
2. **샘플 품질**: Clean FID(cFID)와 Spatial FID(sFID)에서 기존 GAN(StyleGAN, BigGAN) 및 표준 DDPM 대비 **10-30% 향상**
3. **의료 영상 응용**: 뇌 위축(brain atrophy) 같은 희귀 병변 생성에서 StyleGAN2-ADA 대비 더 현실적인 minority 인스턴스 생성
4. **일반화 성능**: 전체 데이터 분포 재현 능력 평가에서도 기존 DDPM 대비 개선된 FID 점수 달성(Table 5)[1]

**제어성 검증**: 
- $\(\tilde{l}\)$ 증가시 LOF 밀도가 저밀도 영역으로 이동
- $\(w\)$ 증가시 특정 minority 클래스의 특징이 더 뚜렷해짐
- 다양한 minority 클래스(예: CelebA의 "Eyeglasses", "Wearing Hat")에 대해 효과적인 생성[1]

### 한계점

1. **계산 비용**: Minority 분류기 구축이 대규모 데이터셋에서는 **계산적으로 비용이 많이 듬**. 예를 들어, ImageNet의 경우 전체 학습 샘플을 사용하여 분류기를 학습해야 함[1]
2. **데이터 의존성**: 사용 가능한 샘플 수 \(N\)이 감소할 경우 다양성(Recall)이 저하됨. 20% 샘플만 사용할 경우 Recall이 0.6254에서 0.5646으로 감소(Table 4)[1]
3. **클래스 수 선택**: Minority 클래스 수 \(L\)의 선택이 중요. 너무 작으면 controllability 저하, 너무 크면 분류기 신뢰도 감소(Figure 11a)[1]
4. **윤리적 위험**: 악의적으로 낮은 \(\tilde{l}\) 값을 사용하여 minority 특징 생성을 억제할 수 있음[1]

### 일반화 성능 향상 가능성

**도메인 일반화**: 논문은 자연 영상뿐만 아니라 **의료 영상(뇌 MRI)** 에서도 우수한 성능을 보여주어 도메인 간 일반화 가능성을 입증합니다. 특히:[1]
- 의료 영상은 일반적으로 데이터가 제한적이고 minority 패턴(병변)이 임상적으로 중요
- StyleGAN2-ADA 대비 더 현실적인 병변 생성으로 실용적 가치 높음(Figure 5)[1]

**다중 가이던스 통합**: 다른 가이던스 신호(예: 클래스-조건부 가이던스)와의 상호작용을 보존함. ImageNet-64实验中, 추가 분류기 가이던스를 통합해도 성능 저하 없이 샘플 품질 향상(Table 6)[1]

**전체 분포 학습 보존**: 전체 데이터셋을 대상으로 한 평가에서도 기존 DDPM 대비 개선된 FID(2.93 vs 2.91)를 달성하여, minority 생성에 집중한다고 해서 majority 영역 성능이 저하되지 않음을 증명(Table 5)[1]

**샘플 효율성**: LSUN-Bedrooms의 경우 전체 학습 샘플의 10%만 사용하여도 효과적인 minority 생성이 가능하여, **데이터 제한적 환경에서의 적용 가능성**을 시사합니다[1]

## 향후 연구 영향 및 고려사항

### 학술적 영향

1. **새로운 연구 방향 제시**: 확산 모델의 **편향성(bias) 분석**을 통한 minority 생성은 기존의 단순한 데이터 증강을 넘어서는 체계적 프레임워크 제공
2. **레이블-비자립적 생성의 가능성**: 레이블 없이도 품질 높은 minority 샘플 생성이 가능하다는 점은 **자기지도학습(self-supervised learning)** 및 **비지도 도메인 적응(unsupervised domain adaptation)** 과 연계 가능
3. **의료 AI 응용**: 희귀 질환 진단 모델 개선, 데이터 불균형 해결을 통한 공정한 AI 개발에 기여
4. **창의적 AI**: 독특하고 창의적인 샘플 생성을 위한 새로운 제어 메커니즘 제공

### 향후 연구 고려사항

1. **효율성 개선**: 대규모 데이터셋을 위한 **경량화된 minority 분류기** 개발 필요. 예를 들어, 사전학습된 특징 추출기와 메타학습(meta-learning)을 결합하여 few-shot 환경에서의 적응성 향상

2. **이론적 심화**: 현재 분석은 DDPM에 기반하지만, **SDE(Stochastic Differential Equation)** 기반 확산 모델이나 consistency model 등 새로운 아키텍처로의 일반화 필요

3. **다중 모달리티 확장**: 현재는 주로 영상 데이터에 집중되어 있으나, **텍스트, 음성, 그래프** 등 다른 모달리티로의 확장 연구 필요. Lee et al. (2022)의 그래프 작업이 선구자적 역할[1]

4. **윤리적 안전성**: 악용 가능성을 방지하기 위한 **안전장치(safeguard)** 개발. 예를 들어, 생성 분포의 편향성을 감지하고 자동으로 보정하는 메커니즘

5. **평가 지표 표준화**: Minority 생성 품질을 평가하는 표준화된 **벤치마크** 개발. 현재는 AvgkNN, LOF 등 다양한 지표를 사용하나, 도메인별 특성을 반영한 통합 평가 체계 필요

6. **실시간 생성**: 현재 NFE(Number of Function Evaluations)가 250으로 비교적 높음. **DDIM(Denoising Diffusion Implicit Models)** 이나 consistency model 기법을 결합하여 **가속화된 minority 생성** 알고리즘 개발[2]

7. **확률적 해석**: Minority guidance가 실제로 **데이터 분포의 꼬리(tail) 영역**을 얼마나 정확히 커버하는지에 대한 확률적 하한(probabilistic lower bound) 분석 필요

## 2020년 이후 관련 최신 연구 탐색

### 확산 모델 기반 Minority 생성 연구

1. **Sehwag et al. (2022)**: "Generating high fidelity data from low-density regions using diffusion models" - 클래스 조건부 설정에서 저밀도 영역 생성을 위한 확산 모델 기법 제시. 본 논문과 가장 유사하지만 레이블 의존적이라는 한계[3][1]

2. **Lee et al. (2022)**: "Exploring chemical space with score-based out-of-distribution generation" - 화학 공간 그래프 데이터를 위한 OOD 생성. 모달리티 특화된 접근법으로 본 논문과 보완적[4][1]

3. **Qin et al. (2023)**: "Class-Balancing Diffusion Models" - CBDM(Class-Balancing Diffusion Model)을 제안하여 샘플링 과정에서 조건부 전이 확률을 조정. 클래스 불균형 문제에 직접적으로 접근[5]

4. **최신 연구(2025)**: 
   - "Boost-and-Skip: A Simple Guidance-Free Diffusion for Minority Generation" - 가이던스 없는 간단한 접근법 제시로 본 논문의 계산 효율성 문제를 개선하려는 시도[6]
   - "Minority-Focused Text-to-Image Generation via Prompt Optimization" - T2I 모델에서 프롬프트 최적화를 통한 minority 생성[7]
   - "When Preferences Diverge: Aligning Diffusion Models with Minority-Aware Adaptive DPO" - preference 데이터의 minority 샘플 문제를 다루는 Diffusion-DPO 연구[8]

### 이론적 발전

1. **LTB-Solver (2024)**: "Long-tailed Bias Solver for image synthesis of diffusion models" - 헤드-투-테일(head-to-tail) 거리 일관성 손실과 밸런스 가이던스 손실을 제안하여 롱테일 분포에서의 확산 모델 성능 저하 문제 해결[9]

2. **T2H (2024)**: "LONG-TAILED DIFFUSION MODELS WITH ORIENTED GUIDANCE" - 헤드 클래스(head classes)에서 테일 클래스(tail classes)로의 전이 학습을 통한 개선[10]

3. **Self-Guided Generation (2024)**: "Self-Guided Generation of Minority Samples Using Diffusion Models" - 사전학습된 모델만으로 실행 가능한 자체 가이던스 기법 제시로 외부 분류기 의존성 감소[4]

### 응용 분야 확장

1. **의료 영상**: 
   - "Diffusion models for medical anomaly detection" (Wolleb et al., 2022) - 확산 모델을 의료 이상 탐지에 활용[11][1]
   - "Addressing Class Imbalance with Latent Diffusion-based Augmentation" (2024) - 흉부 X-ray에서 질병 양성 샘플 합성[12]

2. **공정성 연구**: 
   - "FairGen: Controlling Sensitive Attributes for Fair Generations in Diffusion Models" (2025) - 민감한 속성에 대한 생성 편향 완화[13]
   - "CADS: Unleashing the Diversity of Diffusion Models through Condition-Annealed Sampling" (2024) - 고 가이던스 스케일에서의 다양성 증가[14]

### 기술적 진화

1. **가속화 기법**: DDIM과 같은 암시적(implicit) 샘플링 기법이 확산 모델의 생성 속도를 10-50배 개선[2]
2. **결정론적 모델**: "Iterative α-(de)Blending" (2023)과 같은 결정론적 확산 모델이 수치적 안정성과 품질 향상 제시[15]
3. **멀티레이블 불균형**: "Addressing Multilabel Imbalance with Diffusion Model-Generated Synthetic Samples" (2025)가 멀티레이블 불균형 문제에 확산 모델 적용[16]

### 연구 동향 분석

2020년 이후 연구는 다음 방향으로 진화했습니다:

1. **레이블 의존성 감소**: 본 논문의 label-agnostic 접근법 → Self-guided 방법으로 진화
2. **계산 효율성 향상**: 본 논문의 분류기 학습 → Boost-and-Skip 같은 가이던스-프리 방법으로 발전
3. **이론적 심화**: 편향성 분석 → Long-tailed bias solver 같은 체계적 이론 개발
4. **응용 다양화**: 자연 영상 → 의료, 공정성, T2I 등 다양한 도메인으로 확장

이러한 흐름은 본 논문이 제시한 **minority 생성의 중요성과 효율성 문제**가 학계의 지속적 관심사임을 확인하며, 앞으로도 **효율성, 이론적 엄밀성, 도메인 일반화**가 주요 연구 과제로 남을 것으로 예상됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ca160a54-d32e-4236-901e-100cfe5127cd/2301.12334v2.pdf)
[2](https://www.semanticscholar.org/paper/014576b866078524286802b1d0e18628520aa886)
[3](https://openaccess.thecvf.com/content/CVPR2023W/VAND/papers/Graham_Denoising_Diffusion_Models_for_Out-of-Distribution_Detection_CVPRW_2023_paper.pdf)
[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08641.pdf)
[5](https://openaccess.thecvf.com/content/CVPR2023/papers/Qin_Class-Balancing_Diffusion_Models_CVPR_2023_paper.pdf)
[6](http://arxiv.org/pdf/2502.06516.pdf)
[7](https://arxiv.org/html/2410.07838v2)
[8](https://arxiv.org/html/2503.16921v1)
[9](https://www.sciencedirect.com/science/article/abs/pii/S0925231225003236)
[10](https://proceedings.iclr.cc/paper_files/paper/2024/file/7dff60c3db3c0d2f3acf92f13b1b2472-Paper-Conference.pdf)
[11](https://www.sciencedirect.com/science/article/pii/S1361841524000136)
[12](https://pmc.ncbi.nlm.nih.gov/articles/PMC11936509/)
[13](https://arxiv.org/html/2503.01872v1)
[14](https://arxiv.org/html/2310.17347)
[15](https://arxiv.org/pdf/2305.03486.pdf)
[16](http://arxiv.org/pdf/2501.10822.pdf)
[17](https://www.semanticscholar.org/paper/2aab1a79341e4967e31b8efab4dfaf1f96596b74)
[18](https://dl.acm.org/doi/10.1145/3377049.3377115)
[19](https://www.semanticscholar.org/paper/685af6d2bcdff7170574643b2c5ab4fbcc36f597)
[20](https://www.semanticscholar.org/paper/34bf13e58c7226d615afead0c0f679432502940e)
[21](https://ieeexplore.ieee.org/document/9119990/)
[22](https://academic.oup.com/bioinformatics/article/36/Supplement_2/i610/6055927)
[23](https://doi.apa.org/doi/10.1037/neu0000636)
[24](https://doi.apa.org/doi/10.1037/cou0000535)
[25](https://academic.oup.com/abm/article/55/6/530/5911195)
[26](https://arxiv.org/html/2301.12334v2)
[27](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/dont-play-favorites/)
[28](https://workshop.isic-archive.com/2024/paper_wang.pdf)
[29](https://arxiv.org/abs/2301.12334)
