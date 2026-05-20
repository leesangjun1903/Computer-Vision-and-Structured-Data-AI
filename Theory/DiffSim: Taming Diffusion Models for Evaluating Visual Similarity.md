
# DiffSim: Taming Diffusion Models for Evaluating Visual Similarity 

> **논문 기본 정보**
> - **제목:** DiffSim: Taming Diffusion Models for Evaluating Visual Similarity
> - **저자:** Yiren Song\*, Xiaokang Liu\*, Mike Zheng Shou† (Show Lab, National University of Singapore)
> - **arXiv:** [2412.14580](https://arxiv.org/abs/2412.14580) (2024년 12월 19일)
> - **학회:** ICCV 2025 (Accepted)
> - **공식 코드:** [https://github.com/showlab/DiffSim](https://github.com/showlab/DiffSim)

---

## 1. 핵심 주장 및 주요 기여 요약

이 논문은 **사전 학습된 Diffusion 모델을 시각적 유사도 측정에 활용할 수 있음을 최초로 발견**하고 DiffSim 방법론을 도입하여, 커스텀 생성 태스크에서 지각적 일관성을 포착하지 못하는 기존 지표들의 한계를 해결한다.

DiffSim은 사전 학습된 Diffusion 모델을 활용해 시각적 유사도 평가를 위한 이미지 특징을 추출하며, **인간 판단 일관성(Human Judgment Consistency), 스타일 유사도(Style Similarity), 인스턴스 수준 일관성(Instance-level Consistency)** 세 영역 모두에서 선도적인 성능을 보인다.

### 핵심 주요 기여 (Contributions) 요약

| 기여 항목 | 내용 |
|---|---|
| 최초 발견 | Diffusion 모델을 시각 유사도 평가에 활용 |
| AAS 도입 | Aligned Attention Score: 새로운 특징 정렬 메커니즘 |
| 새 벤치마크 | Sref Bench (스타일), IP Bench (인스턴스) |
| 일반화 | AAS를 CLIP·DINO에도 적용 가능 |
| 추가 파인튜닝 불필요 | Zero-shot으로 SOTA 달성 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2-1. 해결하고자 하는 문제

기존의 지각적 유사도 지표(Perceptual Similarity Metrics)는 주로 **픽셀 및 패치 수준에서 작동**하며, 낮은 수준의 색상과 텍스처를 비교하지만 이미지 레이아웃, 객체 자세, 의미적 내용에서의 중간 수준 유사도와 차이를 포착하지 못한다.

대조 학습 기반의 CLIP과 자기지도학습 기반의 DINO는 의미적 유사도 측정에 자주 사용되지만, **이미지 특징을 과도하게 압축**하여 외관 세부 사항을 충분히 평가하지 못한다.

일부 연구들은 인간 정렬 유사도 평가 방법을 제안하여 유사도 트리플릿에 대한 인간의 선택 데이터를 수집하고 모델을 훈련하지만, **도메인 외(Out-of-Domain) 시나리오에서의 일반화 능력이 제한적**으로 평가된다.

정리하면, 기존 지표들의 문제는 다음 세 가지로 요약된다:

1. **저수준 지표 (LPIPS, SSIM, PSNR):** 픽셀/패치 수준만 비교, 의미적 내용 미반영
2. **고수준 지표 (CLIP, DINO):** 의미적 압축 과다, 외관 세부사항 손실
3. **학습 기반 지표 (DreamSim):** 도메인 외 일반화 능력 부족

---

### 2-2. 제안하는 방법 (수식 포함)

#### (A) 방법론의 세 가지 핵심 통찰

첫째, U-Net의 **자기-어텐션(Self-Attention)** 및 **교차-어텐션(Cross-Attention)** 레이어는 외관 및 스타일 유사도를 평가하는 데 필요한 시각적 특징을 효과적으로 보존한다.

둘째, **ReferenceNet**은 U-Net을 사용하여 참조 이미지에서 특징을 추출하고 이를 디노이징 U-Net의 자기-어텐션 레이어에서 Key(K) 맵 및 Value(V) 맵으로 직접 연결함으로써 외관 유사성을 효과적으로 유지한다.

셋째, **Custom Diffusion**은 교차-어텐션 레이어의 `to K`와 `to V` 행렬이 Stable Diffusion 모델이 개념을 학습하는 핵심 모듈임을 입증한다.

#### (B) Aligned Attention Score (AAS) — 핵심 메커니즘

CLIP 및 DINO와 달리 Stable Diffusion U-Net의 특징은 공간적 정보가 밀집되어 있어 **픽셀 수준에서 정렬 불일치가 발생**한다. 따라서 특징 맵 간 단순한 MSE나 코사인 계산은 비실용적이며, 이는 이후 실험에 의해 검증된다.

이를 해결하기 위해 AAS(Aligned Attention Score)를 도입하며, 이는 **U-Net의 자기-어텐션 레이어에서 이미지 A와 B의 특징을 정렬하기 위해 어텐션 메커니즘을 혁신적으로 활용**하고, 이후 정렬된 특징 간의 코사인 거리를 계산한다.

AAS의 수식을 단계별로 구성하면 다음과 같다:

**Step 1: 두 이미지의 특징 추출**

이미지 $A$와 $B$를 각각 동일한 Diffusion U-Net에 통과시켜 지정된 레이어 $l$에서 특징 맵을 추출한다:

$$
F_A = \text{UNet}^{(l)}(A), \quad F_B = \text{UNet}^{(l)}(B)
$$

**Step 2: Query, Key, Value 생성 (Self-Attention)**

$$
Q_A = F_A W^Q, \quad K_A = F_A W^K, \quad V_A = F_A W^V
$$
$$
Q_B = F_B W^Q, \quad K_B = F_B W^K, \quad V_B = F_B W^V
$$

**Step 3: Aligned Attention — A의 쿼리로 B의 특징을 정렬**

이미지 $A$의 Query로 이미지 $B$의 Key와 Value를 어텐션 연산하여 $B$의 특징을 $A$의 공간 구조에 맞게 정렬:

$$
\tilde{F}_{B \to A} = \text{Softmax}\!\left(\frac{Q_A K_B^\top}{\sqrt{d}}\right) V_B
$$

여기서 $d$는 Key의 차원 수(스케일링 인자)이다.

**Step 4: 정렬된 특징 간 코사인 유사도 계산**

```math
\text{AAS}(A, B) = \frac{1}{N} \sum_{i=1}^{N} \frac{F_{A,i} \cdot \tilde{F}_{B \to A,i}}{\|F_{A,i}\| \cdot \|\tilde{F}_{B \to A,i}\|}
```

여기서 $N$은 공간적 위치(spatial position)의 수이다.

**최종 DiffSim 유사도 점수:**

```math
\text{DiffSim}(A, B) = \text{AAS}(A, B) \quad \text{at layer } l^*, \text{ timestep } t^*
```

최적 레이어 $l^\*$와 타임스텝 $t^*$는 평가 태스크의 목적(스타일 vs. 의미적 유사도)에 따라 달리 설정된다.

#### (C) 레이어·타임스텝 선택 전략

**얕은 레이어와 높은 디노이징 타임스텝**은 저수준 및 스타일 유사도 평가에 적합하며, **깊은 레이어와 낮은 타임스텝**은 의미적 유사도 평가에 뛰어나다. 이는 DiffSim이 단순한 설정 조정만으로도 다양한 유사도 측정을 달성할 수 있음을 의미한다.

---

### 2-3. 모델 구조 (DiffSim-S와 DiffSim-C)

DiffSim은 두 가지 구현 방식을 제공한다:
- **DiffSim-S (Self-Attention 기반):** U-Net이 두 이미지에서 특징을 추출하여 지정된 레이어에서 AAS를 계산한다.
- **DiffSim-C (Cross-Attention 기반):** IP-Adapter Plus와 U-Net을 통해 이미지 입력을 교환하여 특징을 추출한다.

**두 가지 버전의 비교:**

| | **DiffSim-S** | **DiffSim-C** |
|---|---|---|
| Attention 유형 | Self-Attention | Cross-Attention + IP-Adapter |
| 특징 추출 | 단일 U-Net | IP-Adapter Plus + U-Net |
| 강점 | 전반적 우수 성능 | 인스턴스 유사도에 약간 우위 |
| 백본 | SD 1.5 / SD-XL | SD 1.5 |

전반적으로 **DiffSim-S SD1.5**가 우수한 성능을 보이며, 인스턴스 수준 유사도 평가(CUTE 데이터셋)에서는 DiffSim-C SD1.5가 약간 더 나은 성능을 보인다.

---

### 2-4. 벤치마크 및 성능 향상

#### 새로 도입된 벤치마크

**Sref Benchmark:** 508가지 스타일을 수집하였으며, 각 스타일은 인간 아티스트가 직접 선별하고 Midjourney의 Sref 모드로 생성된 4개의 주제별 참조 이미지를 포함한다.

**IP Benchmark:** 299개의 IP 캐릭터 이미지를 수집하고, 고급 Flux 모델과 IP-Adapter를 사용하여 각 캐릭터의 다양한 변형본을 서로 다른 일관성 가중치로 생성하였다.

**NIGHTS Dataset:** 인간 지각 유사도 점수가 포함된 20,019개의 이미지 트리플릿으로 구성된 데이터셋이다. 각 트리플릿은 참조 이미지와 두 가지 왜곡 이미지로 구성되며, 본 논문에서는 2,120개의 이미지 트리플릿으로 구성된 테스트 세트를 활용한다.

#### 성능 결과

여러 벤치마크에서의 평가 결과 (대표 지표):

| 지표 | NIGHTS (Human-align) | Dreambench++ | CUTE (Instance) | IP | TID2013 (Low-level) | Sref (Style) | InstantStyle |
|---|---|---|---|---|---|---|---|
| LPIPS | 71.13% | 62.33% | 63.17% | 84.01% | **94.50%** | 87.85% | 93.15% |
| CLIP | (낮음) | (중간) | (중간) | (중간) | (낮음) | (중간) | (중간) |
| DINO v2 | (중간) | (중간) | (중간) | (중간) | (낮음) | (중간) | (중간) |
| **DiffSim** | **최고** | **최고** | **최고 수준** | **최고** | 우수 | **최고** | **최고** |

DiffSim은 추가적인 파인튜닝이나 감독 없이도 CLIP과 DINO v2를 능가하며, DiffSim의 이미지 유사도 평가는 인간 판단과 매우 일관되어 두 가지 인간 일관성 벤치마크에서 최상위를 기록한다.

저수준 유사도 평가(TID2013)에서는 **단순히 디노이징 타임스텝만 수정하여 뛰어난 성능**을 달성하고, CLIP과 DINO v2를 크게 능가하며 LPIPS에 필적하는 수준을 보인다.

---

### 2-5. 한계점

DiffSim은 **배경 민감성(Background Sensitivity)** 문제가 있다. 예를 들어 고양이 이미지를 사용하여 비슷한 배경을 가진 강아지 이미지를 검색하는 문제가 발생하며, 피사체를 크롭(Cropping)하면 이 문제를 완화할 수 있다.

또한 AAS 특징 정렬을 사용하지 않을 경우 **결과가 크게 저하**되는 것이 실험에서 확인되며, AAS 메커니즘에 대한 높은 의존성이 드러난다.

추가적으로 다음과 같은 한계가 있다:

- **계산 비용:** Diffusion U-Net을 매번 포워드 패스로 실행해야 하므로 CLIP·DINO 대비 추론 시간이 더 길다.
- **모델 의존성:** Stable Diffusion 1.5 또는 SD-XL 등 특정 백본에 의존한다.
- **자연 이미지 도메인 외 일반화:** 생성 이미지를 중심으로 설계되어 있어 의료 영상 등 전문 도메인에서의 성능 검증이 부족하다.

---

## 3. 모델의 일반화 성능 향상 가능성

DiffSim의 일반화 가능성은 크게 세 가지 측면에서 두드러진다.

### 3-1. AAS의 CLIP·DINO로의 범용 적용

AAS 기법을 **CLIP과 DINO 등 다른 아키텍처에도 일반화하여 적용**할 수 있음을 발견하였으며, CLIP AAS 메트릭과 DINO AAS 메트릭을 도입함으로써 특정 태스크에서 성능이 크게 향상되었다.

이는 AAS가 Diffusion 모델에만 국한된 기법이 아니라, **어텐션 메커니즘을 보유한 모든 Vision 백본에 적용 가능한 범용 정렬 기법**임을 의미한다.

### 3-2. 타임스텝·레이어 조정을 통한 멀티-그레이뉼러 유사도 평가

얕은 레이어와 높은 디노이징 타임스텝은 저수준·스타일 유사도 평가에, 깊은 레이어와 낮은 타임스텝은 의미적 유사도 평가에 뛰어나며, 이는 DiffSim이 **단순한 설정 조정만으로 다양한 유사도 측정을 달성**할 수 있음을 의미한다.

이러한 유연성은 동일한 모델을 다양한 다운스트림 태스크(스타일 전이 평가, 캐릭터 일관성, 영상 생성 평가 등)에 설정 변경만으로 재활용할 수 있음을 뜻하며, 높은 실용적 일반화 가능성을 시사한다.

### 3-3. 앙상블 모델을 통한 성능 보완

앙상블 모델은 CLIP, DINO v2, DiffSim의 예측을 취합하여 다수결 원칙에 따라 최종 분류를 결정하며, 이를 통해 원래 세 가지 방법 모두를 능가하는 결과를 보인다.

이는 DiffSim이 CLIP·DINO와 **상호 보완적(Complementary)인 정보**를 제공함을 의미하며, 앙상블 전략을 통해 일반화 성능을 더욱 높일 수 있음을 보여준다.

### 3-4. 비디오 도메인으로의 확장

**시간적 외관 일관성(Temporal Appearance Consistency)**은 비디오 생성 및 비디오-이미지 모델 평가에 매우 중요하다. 이상적인 지표는 객체 위치와 레이아웃 변화에 영향을 받지 않으면서 동일 피사체의 프레임 간 안정적인 유사도 점수를 보장해야 한다.

또한 이 기술이 **다른 아키텍처를 향상시키는 방향으로 일반화될 수 있음**을 발견하였다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 연도 | 특징 | 수준 | 한계 |
|---|---|---|---|---|
| **LPIPS** | 2018 | VGG 기반 지각 손실 | 저수준 (픽셀/패치) | 의미적 내용 미포착 |
| **CLIP-Score** | 2021 | 텍스트-이미지 정렬 | 고수준 (의미) | 외관 세부사항 압축 |
| **DINO/DINOv2** | 2021/2023 | 자기지도학습 시각 특징 | 고수준 (의미) | 외관 압축, 공간 정렬 부재 |
| **DreamSim** | 2023 | 인간 판단 정렬 학습 | 중간 수준 | OOD 일반화 제한 |
| **DiffSim (본 논문)** | 2024 | Diffusion U-Net + AAS | **다계층 (저↔고)** | 배경 민감성, 계산 비용 |

**DreamSim**은 "저수준" 지표(LPIPS, PSNR, SSIM)와 "고수준" 측정(CLIP) 사이의 간극을 메우는 새로운 지각 이미지 유사도 지표로, CLIP, OpenCLIP, DINO 임베딩을 연결하여 훈련된 모델이다.

단, 이러한 인간 정렬 학습 방법들은 도메인 외 시나리오에서 **일반화 능력이 제한적**으로 평가된다.

DiffSim이 DreamSim 대비 갖는 핵심 우위는 다음과 같다:

- **추가 학습 불필요 (Zero-shot):** 사전 학습된 Diffusion 모델을 그대로 활용
- **다계층 평가:** 레이어·타임스텝 조정으로 저수준~고수준 유사도를 모두 커버
- **특징 정렬:** AAS를 통해 공간적 불일치 문제를 근본적으로 해결

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5-1. 연구에 미치는 영향

**① Diffusion 모델의 역할 확장**

Diffusion 모델이 생성 분야를 혁신적으로 변환한 가운데, 이 논문은 **사전 학습된 Diffusion 모델을 평가 지표(Metric)로 활용하는 새로운 패러다임**을 제시한다. 이는 Diffusion 모델이 단순 생성 도구를 넘어, 생성된 결과물의 품질 평가 도구로서도 기능할 수 있음을 보여주는 중요한 전환점이다.

**② AAS의 범용 어텐션 정렬 원리 확산**

AAS의 원리(어텐션으로 공간적 불일치를 보정하고 코사인 유사도를 계산)는 향후 다양한 멀티모달 표현 정렬, 비디오 프레임 일관성 평가, 3D 객체 유사도 측정 등에 응용될 수 있다.

**③ 평가 지표 연구의 새 기준 제시**

Sref 및 IP 벤치마크의 도입은 **스타일 수준 및 인스턴스 수준에서 시각적 유사도를 체계적으로 평가하는 새로운 기준**을 제시하며, 향후 관련 연구들은 이 벤치마크를 기준으로 비교 평가될 것이다.

**④ 생성 모델 커스터마이제이션 연구와의 시너지**

DiffSim은 **생성 모델에서 시각적 일관성을 측정하기 위한 강력한 도구**로서 종합적 평가를 통해 SOTA 성능을 달성하며, 향후 IP-Adapter, InstantStyle, DreamBooth 등 커스터마이제이션 연구의 정량 평가에 직접 활용될 것으로 예상된다.

---

### 5-2. 앞으로 연구 시 고려할 점

**① 계산 효율성 개선**

DiffSim은 U-Net 포워드 패스가 필요하여 CLIP·DINO 대비 추론 비용이 높다. 향후 연구에서는 **경량화된 Diffusion 인코더** 또는 **증류(Distillation) 기법**을 통해 추론 속도를 개선해야 한다.

**② 배경 민감성 문제 해결**

배경이 포함된 이미지에서 배경 정보에 의해 유사도가 왜곡되는 문제가 있으며, 피사체 크롭이 완화책으로 제시되지만 자동화된 해결책이 필요하다. 향후 **사전 분할(Segmentation)과의 연계** 또는 **전경-배경 분리 후 유사도 계산** 방법이 연구될 필요가 있다.

**③ 도메인 일반화 실험 강화**

현재 DiffSim은 주로 **생성 이미지 도메인**에서 검증되었다. 의료 영상, 위성 이미지, 산업 검사 이미지 등 전문 도메인에서의 일반화 성능 검증이 추가로 필요하다.

**④ 비디오·3D로의 확장 연구**

시간적 외관 일관성은 비디오 생성 및 비디오-이미지 모델 평가에 매우 중요한 과제이므로, DiffSim의 AAS 원리를 **시간 축(Temporal Axis)으로 확장**한 비디오 일관성 평가 지표 연구가 유망한 방향이다.

**⑤ 앙상블 및 멀티모달 통합**

CLIP, DINO v2, DiffSim의 앙상블이 개별 방법을 능가한다는 결과는, 앞으로 이들의 상호 보완적 정보를 **학습 가능한 가중치**로 동적으로 결합하는 연구(Meta-Metric Learning)의 가능성을 시사한다.

**⑥ 더 강력한 Diffusion 백본 적용**

현재 SD 1.5와 SD-XL을 사용하지만, **FLUX, SD3, Stable Cascade** 등 더 강력한 Diffusion 백본으로의 확장과 이를 통한 성능 향상 가능성이 탐색되어야 한다.

---

## 📚 참고 자료 (출처)

| # | 제목 / 출처 |
|---|---|
| 1 | **arXiv 원문:** [DiffSim: Taming Diffusion Models for Evaluating Visual Similarity (arXiv:2412.14580)](https://arxiv.org/abs/2412.14580) |
| 2 | **ICCV 2025 Open Access 논문 PDF:** [openaccess.thecvf.com](https://openaccess.thecvf.com/content/ICCV2025/papers/Song_DiffSim_Taming_Diffusion_Models_for_Evaluating_Visual_Similarity_ICCV_2025_paper.pdf) |
| 3 | **arXiv HTML 전문:** [arxiv.org/html/2412.14580v1](https://arxiv.org/html/2412.14580v1) |
| 4 | **arXiv PDF:** [arxiv.org/pdf/2412.14580](https://arxiv.org/pdf/2412.14580) |
| 5 | **GitHub 공식 코드:** [github.com/showlab/DiffSim](https://github.com/showlab/DiffSim) |
| 6 | **ICCV 2025 포스터:** [iccv.thecvf.com/virtual/2025/poster/52](https://iccv.thecvf.com/virtual/2025/poster/52) |
| 7 | **Papers with Code:** [paperswithcode.com/paper/diffsim-taming-diffusion-models-for](https://paperswithcode.com/paper/diffsim-taming-diffusion-models-for) |
| 8 | **ResearchGate:** [researchgate.net/publication/387264489](https://www.researchgate.net/publication/387264489_DiffSim_Taming_Diffusion_Models_for_Evaluating_Visual_Similarity) |
| 9 | **NASA ADS Abstract:** [ui.adsabs.harvard.edu/abs/2024arXiv241214580S](https://ui.adsabs.harvard.edu/abs/2024arXiv241214580S/abstract) |
| 10 | **비교 연구 - DreamSim (NeurIPS 2023):** [github.com/ssundaram21/dreamsim](https://github.com/ssundaram21/dreamsim) |
| 11 | **Moonlight Literature Review:** [themoonlight.io/en/review/diffsim-taming-diffusion-models-for-evaluating-visual-similarity](https://www.themoonlight.io/en/review/diffsim-taming-diffusion-models-for-evaluating-visual-similarity) |

> ⚠️ **정확도 관련 주의사항:** AAS의 세부 수식(Step 1~4)은 논문의 개념적 기술을 바탕으로 표준적인 어텐션 수식 형태로 재구성한 것입니다. 논문 PDF 내 정확한 수식 표기는 [arXiv 원문](https://arxiv.org/pdf/2412.14580) 또는 [ICCV 공식 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Song_DiffSim_Taming_Diffusion_Models_for_Evaluating_Visual_Similarity_ICCV_2025_paper.pdf)를 직접 참조하시기를 권장합니다.
