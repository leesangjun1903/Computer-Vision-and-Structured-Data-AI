
# T-Stitch: Accelerating Sampling in Pre-Trained Diffusion Models with Trajectory Stitching

> **논문 정보**
> - **제목**: T-Stitch: Accelerating Sampling in Pre-Trained Diffusion Models with Trajectory Stitching
> - **저자**: Zizheng Pan, Bohan Zhuang, De-An Huang, Weili Nie, Zhiding Yu, Chaowei Xiao, Jianfei Cai, Anima Anandkumar
> - **소속**: Monash University, NVIDIA, University of Wisconsin-Madison, Caltech
> - **발표**: ICLR 2025
> - **arXiv**: [2402.14167](https://arxiv.org/abs/2402.14167)
> - **코드**: [NVlabs/T-Stitch (GitHub)](https://github.com/NVlabs/T-Stitch)
> - **프로젝트 페이지**: [t-stitch.github.io](https://t-stitch.github.io)

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

Diffusion Probabilistic Models (DPMs)로부터의 샘플링은 고품질 이미지 생성을 위해 매우 비용이 많이 들며, 일반적으로 대형 모델로 수많은 스텝을 필요로 한다. T-Stitch는 이 문제를 해결하기 위한 **단순하면서도 효율적인 샘플링 기법**으로, 전체 샘플링 궤적에서 대형 DPM만 단독으로 사용하는 대신, **초기 스텝에서는 소형 DPM을 대형 DPM의 저렴한 대체재로 활용**하고, 이후 단계에서 대형 DPM으로 전환한다.

핵심 통찰(Key Insight)은 **동일한 학습 데이터 분포 하에서 서로 다른 Diffusion Model들이 유사한 인코딩을 학습하며**, 소형 모델도 초기 스텝에서 양호한 전역 구조(global structure)를 생성할 수 있다는 것이다.

### 1.2 주요 기여

T-Stitch는 **학습이 필요 없으며(Training-free)**, 다양한 아키텍처에 범용적으로 적용 가능하고, 기존의 대부분 빠른 샘플링 기법들과 상호 보완적이다. 예를 들어 DiT-XL에서 초기 타임스텝의 40%를 10배 빠른 DiT-S로 대체해도 class-conditional ImageNet 생성에서 성능 저하가 없다.

또한 T-Stitch는 인기 있는 Stable Diffusion(SD) 모델을 가속화할 뿐만 아니라, **공개 모델 저장소의 스타일화된 SD 모델의 프롬프트 정렬(prompt alignment)까지 개선**한다. 명시적 모델 할당 전략 덕분에 학습이나 탐색 필요성을 현저히 줄여 높은 배포 효율성을 달성한다.

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

DPM은 텍스트-이미지 생성, 오디오 합성, 3D 생성 등 다양한 분야에서 놀라운 성과를 거두었으나, 고품질 생성을 위해서는 대형 DPM에서 수백 번의 디노이징 스텝이 필요하며, 각 스텝마다 높은 연산 비용이 소요된다. 예를 들어 고성능 RTX 3090에서도 DiT-XL로 8장의 이미지를 100 디노이징 스텝으로 생성하는 데 16.5초가 걸린다.

기존 가속화 방법들은 크게 두 가지 방향으로 분류된다:

**(1) 네트워크 압축 기법**: Pruning, Quantization 등을 통해 스텝당 연산 비용 감소. **(2) 샘플링 스텝 수 감소**: Distillation, Implicit Sampler, 개선된 미분방정식 솔버 등. **(3) 병렬 샘플링**: 여러 타임스텝을 동시에 계산하는 방법.

T-Stitch는 이들과 **직교하는(complementary)** 새로운 접근법을 제안한다.

---

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 이론적 근거

T-Stitch의 작동 원리는 두 가지 핵심 통찰에 기반한다:

**(1) 공통 잠재 공간(Common Latent Space)**: 동일한 데이터 분포로 학습된 서로 다른 DPM들은 유사한 샘플링 궤적을 공유하며, 이로 인해 모델 크기가 다르더라도, 심지어 아키텍처가 달라도 궤적 간의 스티칭이 가능하다. **(2) 주파수 관점(Frequency Perspective)**: 디노이징 과정에서 초기 스텝은 저주파 성분(전역 구조) 생성에 집중하고, 후기 스텝은 고주파 신호(세부 디테일)를 생성하는 데 집중한다. 소형 모델은 고주파 디테일에는 약하지만 초기 스텝에서의 전역 구조 생성은 충분히 가능하다.

#### 2.2.2 T-Stitch 샘플링 절차

전체 $T$ 타임스텝 디노이징 궤적을 정의하자. 타임스텝은 $t = T, T-1, \ldots, 1$ (노이즈 → 이미지 방향) 순서로 진행된다.

**소형 모델 적용 구간**을 비율 $\alpha \in [0, 1]$로 정의하면:

$$
\hat{x}_{t-1} = \begin{cases} D_S(x_t, t, c) & \text{if } t > (1 - \alpha) \cdot T \\ D_L(x_t, t, c) & \text{if } t \leq (1 - \alpha) \cdot T \end{cases}
$$

여기서:
- $D_S$: 소형(Small) DPM의 디노이저
- $D_L$: 대형(Large) DPM의 디노이저
- $x_t$: 타임스텝 $t$에서의 잠재 변수(latent)
- $c$: 조건(class label 또는 text prompt)
- $\alpha$: 소형 모델이 대체하는 초기 타임스텝 비율

즉, 전체 궤적의 앞부분 $\alpha$ 비율 구간은 소형 모델이, 나머지 $(1-\alpha)$ 비율 구간은 대형 모델이 담당한다.

#### 2.2.3 스위칭 타임스텝 결정

스위칭 타임스텝 $t^* = \lfloor (1 - \alpha) \cdot T \rfloor$로 정의되며, $\alpha$를 조절함으로써 속도-품질 트레이드오프를 유연하게 설정한다.

다양한 비율의 소형 및 대형 DPM을 샘플링 궤적에 할당함으로써, **유연한 속도-품질 트레이드오프**를 달성할 수 있다.

#### 2.2.4 분류기 없는 안내(Classifier-Free Guidance, CFG)

분류기 기반 디노이저와 달리, Classifier-Free Guidance는 조건부 모델과 비조건부 모델을 하나의 네트워크 안에서 공동 학습하며, 조건 신호를 null 임베딩으로 대체하는 방식이다.

CFG 하에서의 예측:

$$
\tilde{\epsilon}_\theta(x_t, t, c) = (1 + w) \cdot \epsilon_\theta(x_t, t, c) - w \cdot \epsilon_\theta(x_t, t, \emptyset)
$$

여기서 $w$는 guidance scale, $\emptyset$은 null 조건이다. T-Stitch는 이 CFG 메커니즘과 완전히 호환된다.

#### 2.2.5 세 가지 모델을 사용한 다중 스티칭 (Multi-Model Stitching)

T-Stitch는 중간 크기 모델을 중간 디노이징 구간에 적용하여 더 많은 속도-품질 트레이드오프를 달성할 수 있다. 예를 들어 DiT-S, DiT-B, DiT-XL 세 모델을 조합하면, DiT-S로 시작→DiT-B 중간 구간→DiT-XL 후반부 디테일 생성 순서로 진행하여 FID와 Inception Score 모두에서 부드러운 Pareto Frontier를 형성한다.

$$
\hat{x}_{t-1} = \begin{cases} D_{S}(x_t, t, c) & \text{if } t > t_1 \\ D_{M}(x_t, t, c) & \text{if } t_2 < t \leq t_1 \\ D_{L}(x_t, t, c) & \text{if } t \leq t_2 \end{cases}
$$

여기서 $t_1 > t_2$이고, $D_M$은 중간 크기(Medium) 모델이다.

---

### 2.3 모델 구조

T-Stitch는 Diffusion Model의 **디노이저(Denoiser) $D$** 에 집중하며, 이는 일반적으로 각 타임스텝마다 높은 FLOPs을 소비하는 다양한 아키텍처의 대형 파라미터 신경망이다.

T-Stitch는 먼저 **DiT(Diffusion Transformer)** 모델 패밀리를 기반으로 탐색하고, 이후 **U-Net**(Rombach et al., 2022)과 **U-ViT**(Bao et al., 2023) 등 다른 아키텍처에도 적용 가능한 범용 기술임을 보인다.

지원 아키텍처:
| 아키텍처 | 소형 모델 | 대형 모델 | 비고 |
|---|---|---|---|
| DiT | DiT-S | DiT-XL | class-conditional ImageNet |
| U-Net (LDM) | LDM-S | LDM | class-conditional ImageNet |
| U-ViT | DiT-S | U-ViT-H | 아키텍처 간 스티칭 |
| Stable Diffusion | BK-SDM Tiny | SD v1.4 / SDXL | text-to-image |

소형 모델 선택 원칙은: **(1) 명확하게 더 빠를 것**, **(2) 충분히 최적화되어 있을 것**, **(3) 대형 모델과 동일한 데이터셋으로 학습되었거나, 최소한 유사한 데이터 분포를 학습했을 것**이다.

기본적으로 T-Stitch는 실제 성능이 매우 우수한 **쌍(pairwise) 디노이저** 방식을 채택한다.

---

### 2.4 성능 향상

#### 2.4.1 DiT 기반 실험

DiT-XL에서 초기 타임스텝의 **40%를 10배 빠른 DiT-S로 대체해도** class-conditional ImageNet 생성에서 성능 저하가 없다.

세 가지 모델 조합 실험에서, 비교 가능한 FID(9.21 vs. 9.19)와 Inception Score(243.82 vs. 245.73)를 유지하면서도 의미 있는 속도 향상을 달성한다.

#### 2.4.2 Stable Diffusion 실험

초기 **30%의 스텝을 BK-SDM Tiny로 대체해도** Inception Score와 CLIP Score에서 유의미한 성능 저하 없이 더 나은 FID를 달성한다. 향후 더 빠르고 우수한 소형 모델을 사용하면 더 좋은 품질-효율성 트레이드오프를 기대할 수 있다.

#### 2.4.3 기존 기법과의 보완성

T-Stitch는 샘플링 스텝을 줄이는 기존 기법들(스텝 수 직접 감소, 고급 샘플러, Distillation 등)과 완전히 **상호 보완적(complementary)**이다.

예를 들어, T-Stitch는 DeepCache와 결합하여 소형 및 대형 Diffusion Model을 동시에 가속화할 수 있으며, 스타일화된 SD의 프롬프트 정렬을 추가로 개선할 수 있다.

또한 T-Stitch는 LCM-SDXL과 같은 스텝 증류(step-distilled) 모델과도 호환되어 인상적인 속도-품질 트레이드오프를 달성한다.

---

### 2.5 한계점

논문에서 인정하거나 도출 가능한 한계점은 다음과 같다:

1. **소형 모델 의존성**: 속도와 품질의 최악/최선 경계는 각각 가장 작은 모델과 가장 큰 디노이저에 의해 결정된다. 즉, 효과는 사용 가능한 소형 모델의 품질에 크게 의존한다.

2. **동일 잠재 공간 요구**: 두 모델이 동일한 모양의 잠재 벡터(latent)를 처리해야 한다.

3. **소형 모델 품질 제약**: 초기 30% 스텝에서는 BK-SDM Tiny 사용이 가능하지만, **더 나은 소형 모델이 개발된다면** 더 높은 품질-효율성 트레이드오프를 달성할 수 있다고 인정한다.

4. **스위칭 포인트 최적화 부재**: 현재는 고정된 비율 $\alpha$를 사용하며, 입력이나 타임스텝에 따른 동적 전환 메커니즘이 없다.

5. **U-ViT-H와 DiT-S 간 아키텍처 차이**: U-ViT-H 샘플링을 소형 DiT-S로 가속할 수 있지만, 이종 아키텍처 간 스티칭에서는 성능 저하가 발생할 수 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 아키텍처 간 일반화

T-Stitch는 학습이 필요 없고, 다양한 아키텍처에 범용적으로 적용 가능하며, 유연한 속도-품질 트레이드오프로 대부분의 기존 빠른 샘플링 기법들과 보완적이다.

동일한 데이터 분포로 학습된 서로 다른 DPM들 사이에는 공통 잠재 공간이 존재하며, 서로 다른 모델 크기뿐 아니라 **아키텍처 간(inter-architecture) 스티칭**도 가능하다는 점이 일반화 가능성의 핵심 근거이다.

### 3.2 스타일화 모델에서의 프롬프트 정렬 향상 (일반화의 역방향 활용)

T-Stitch는 속도 개선에 그치지 않고 스타일화 모델의 **프롬프트 정렬도 개선**한다. 이는 스타일화 모델(예: Ghibli, InkPunk 스타일)의 파인튜닝 과정에서 프롬프트 정렬이 저하되기 때문이며, T-Stitch가 소형 일반 SD 모델을 결합함으로써 프롬프트 정렬을 보완하고 스타일화 이미지 품질도 유지한다.

실용적인 활용 패턴으로, **소형 일반 전문가 모델이 초기 빠른 스케칭과 더 나은 프롬프트 정렬을 담당하고, 이후 스타일화된 SD가 디테일을 세밀하게 묘사**하는 방식이 강력하게 지지된다.

### 3.3 공개 모델 저장소 활용 (일반화 확장성)

공개 커뮤니티(HuggingFace, Civitai 등)에는 다양한 규모의 사전 학습 모델이 수만 개 공개되어 있으며, 이는 T-Stitch의 "무료 점심(free lunch)" 기회를 제공한다. 논문에서 Stable Diffusion 모델을 처음부터 학습하지 않고도 SDXL, ControlNet 등 광범위한 응용을 시연했다.

### 3.4 다양한 샘플러와의 일반화

T-Stitch는 대형 DPM의 속도를 품질 손실 없이 크게 향상시키며, 이 효과는 **다양한 아키텍처와 Diffusion Model 샘플러에 걸쳐 일관되게** 나타난다.

---

## 4. 연구에 미치는 영향 및 앞으로의 연구 시 고려할 점

### 4.1 앞으로의 연구에 미치는 영향

#### 4.1.1 "모델 협업(Model Collaboration)" 패러다임 확립

T-Stitch는 단순히 모델을 압축하거나 스텝을 줄이는 것이 아닌, **서로 다른 크기의 기존 사전 학습 모델들이 협력**하는 새로운 패러다임을 열었다. 이는 언어 모델 분야의 Speculative Decoding과 개념적으로 유사하며, 이미 R-Stitch 같은 후속 연구가 소형 언어 모델(SLM)과 대형 언어 모델(LLM) 사이에서 신뢰도 기반으로 동적 전환하는 토큰 수준 하이브리드 디코딩 프레임워크로 발전하고 있다.

#### 4.1.2 주파수 분해 기반 효율화 연구 촉진

디노이징 과정에서 초기에는 저주파(전역 구조), 후기에는 고주파(세부 디테일)를 생성한다는 인사이트는, 각 타임스텝에서 **모델 또는 아키텍처 구성 요소를 다르게 할당**하는 후속 연구를 자극하고 있다.

#### 4.1.3 공개 모델 생태계 활용 연구 방향

T-Stitch가 보인 "공개된 소형 모델 활용" 접근 방식은 모델 저장소를 재활용하는 다양한 연구로 확산될 가능성이 높다.

#### 4.1.4 멀티모달 및 비디오 생성 확장 가능성

T-Stitch의 원리는 이미지에 국한되지 않으며, 오디오나 3D 생성, 더 나아가 비디오 생성 모델에서의 시간 차원 스티칭 연구로도 확장될 수 있다.

---

### 4.2 앞으로 연구 시 고려할 점

#### (1) 동적 스위칭 전략 연구

현재 T-Stitch는 고정된 비율 $\alpha$를 사용한다. 입력 복잡도, 프롬프트 특성, 또는 중간 잠재 변수의 상태에 따라 **적응적·동적으로 스위칭 포인트를 결정**하는 방법론을 탐구할 필요가 있다.

수식으로 표현하면:

$$
t^* = f_\phi(x_T, c, \text{complexity score})
$$

여기서 $f_\phi$는 학습 가능한 또는 휴리스틱 기반의 스위칭 포인트 결정 함수다.

#### (2) 더 나은 소형 모델 설계 및 소형 모델 매칭 알고리즘

더 나은 소형 모델이 개발된다면 더 좋은 품질-효율성 트레이드오프를 달성할 수 있으므로, 대형 모델의 초기 스텝 행동을 효율적으로 모방하도록 **특화 설계된 소형 모델**(예: early-step 특화 경량 모델)의 학습 방법론 연구가 필요하다.

#### (3) 이종 아키텍처 간 호환성 보장

서로 다른 잠재 공간 차원을 가진 모델들 간의 스티칭을 위한 **어댑터(adapter) 또는 정렬 모듈(alignment module)** 연구가 필요하다.

#### (4) 이론적 수렴 보장 분석

T-Stitch 적용 시 샘플링 궤적이 원래 대형 모델의 분포에 수렴하는지에 대한 **엄밀한 이론적 분석**이 부족하다. 소형 모델이 생성한 중간 잠재 변수의 분포 편향이 최종 품질에 미치는 영향을 정량화할 수 있는 이론 체계가 필요하다.

수렴 조건 탐구:

```math
\|p_L(x_{t^*}) - p_S(x_{t^*})\|_{TV} \leq \epsilon
```

여기서 $p_L$, $p_S$는 각각 대형/소형 모델이 타임스텝 $t^*$에서 생성하는 분포, $\|\cdot\|_{TV}$는 Total Variation Distance다.

#### (5) 다양한 도메인과 조건에서의 일반화 실험

현재 실험은 주로 class-conditional ImageNet과 일부 Stable Diffusion 실험에 국한된다. 의료 영상, 위성 영상, 과학적 데이터 생성 등 **특수 도메인**에서의 T-Stitch 효과 검증이 필요하다.

#### (6) 추론 시간 최적 조건부 할당 (Optimal Transport 관점)

어떤 타임스텝 구간을 어떤 모델이 담당해야 최적인지를 **Optimal Transport 또는 강화학습** 관점에서 자동 탐색하는 방법론 연구가 의미 있을 것이다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 연도 | 핵심 아이디어 | 학습 필요 여부 | 아키텍처 의존성 | 기존 기법 보완 가능성 |
|---|---|---|---|---|---|
| **DDIM** (Song et al.) | 2021 | Non-Markovian 샘플러, 스텝 수 감소 | ✗ (추론만) | 낮음 | ✓ |
| **DPM-Solver** (Lu et al.) | 2022 | 고차 ODE 솔버 적용 | ✗ | 낮음 | ✓ |
| **Knowledge Distillation** (Salimans & Ho) | 2022 | 학생 모델에 궤적 지식 증류 | **✓ (재학습 필요)** | 높음 | △ |
| **Pruning/Quantization** (Fang et al., Li et al.) | 2023 | 네트워크 경량화 | **✓ (파인튜닝)** | 높음 | △ |
| **DeepCache** (Ma et al.) | 2023 | 특징 캐싱으로 스텝당 비용 감소 | ✗ | 중간 | ✓ |
| **LCM (Latent Consistency Model)** | 2023 | Consistency 기반 빠른 증류 | **✓ (증류 필요)** | 높음 | △ |
| **BK-SDM** (Kim et al.) | 2023 | SD 경량화 (Pruning + KD) | **✓** | 높음 | ✓ (T-Stitch 소형 모델로 활용) |
| **T-Stitch (본 논문)** | 2024 | 궤적 스티칭, 소형→대형 전환 | **✗ (완전 Training-free)** | **낮음 (범용)** | **✓✓ (완전 보완)** |

T-Stitch는 기존 빠른 샘플링 접근법들과 **상호 보완적**이며, 대형 DPM이 담당하는 궤적 구간은 여전히 스텝 수 감소나 압축 기법을 통해 추가로 가속화할 수 있다.

---

## 📚 참고자료 및 출처

1. **arXiv 원문**: Zizheng Pan et al., "T-Stitch: Accelerating Sampling in Pre-Trained Diffusion Models with Trajectory Stitching," arXiv:2402.14167, Feb. 2024. https://arxiv.org/abs/2402.14167

2. **공식 프로젝트 페이지**: https://t-stitch.github.io/

3. **ICLR 2025 공식 논문 PDF**: https://proceedings.iclr.cc/paper_files/paper/2025/file/120339238f293d4ae53a7167403abc4b-Paper-Conference.pdf

4. **OpenReview (ICLR 2025 심사)**: https://openreview.net/forum?id=2mqb8bPHeb

5. **공식 GitHub 구현 (NVlabs/T-Stitch)**: https://github.com/NVlabs/T-Stitch

6. **NVIDIA Research 페이지**: https://research.nvidia.com/labs/lpr/publication/t-stitch2024/

7. **관련 후속 연구 - R-Stitch**: "R-Stitch: Dynamic Trajectory Stitching for Efficient Reasoning," arXiv:2507.17307. https://arxiv.org/html/2507.17307v3

8. **관련 연구 - Stitchable Neural Networks** (Pan et al., CVPR 2023): 본 논문의 이론적 선행 연구

9. **관련 연구 - BK-SDM** (Kim et al., 2023): T-Stitch 실험에서 소형 SD로 활용된 모델

> ⚠️ **정확도 관련 주의사항**: 수식의 일부(특히 스위칭 수식)는 논문의 알고리즘 기술을 기반으로 재구성한 것으로, 논문에 명시된 정확한 수식 기호와 약간 다를 수 있습니다. 완전한 수식은 반드시 ICLR 2025 공식 논문 PDF를 직접 확인하시기 바랍니다.
