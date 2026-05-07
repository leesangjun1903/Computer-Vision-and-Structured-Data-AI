# Random Conditioning with Distillation for Data-Efficient Diffusion Model Compression

> **대상 논문**: Dohyun Kim, Sehwan Park, Geonhee Han, Seung Wook Kim, Paul Hongsuck Seo, *"Random Conditioning with Distillation for Data-Efficient Diffusion Model Compression"*, **CVPR 2025** (arXiv:2504.02011, 2025년 4월). 고려대학교 CSE & NVIDIA. CVPR 2025 proceedings 논문 번호 18607–18618.

---

## 1. 핵심 주장과 주요 기여 요약

본 논문은 조건부 확산모델(conditional diffusion model)을 **이미지프리(image-free) 방식**으로 효율적으로 지식증류(knowledge distillation, KD)하기 위한 **Random Conditioning** 기법을 제안한다. 핵심 통찰은 다음과 같다.

1. **새로운 통찰 (novel insight)**: 인식(recognition) 모델의 KD에서는 학생(student)이 학습 중 보지 못한 클래스도 일반화하여 인식할 수 있음이 잘 알려져 있으나(Hinton et al. 2015의 MNIST '3' 사례), **조건부 확산모델에서는 학생 모델이 학습 데이터의 페어드(paired) 조건에 포함되지 않은 개념을 생성하지 못한다**는 사실을 명확하게 지적·실험적으로 입증한다(Fig. 2).
2. **Random Conditioning 기법**: 노이즈가 추가된 이미지 $\mathbf{x}_t^n$을 그것의 원래 텍스트 $c^n$이 아닌, **무관할 수 있는 임의의 텍스트 $\tilde{c}\in\mathcal{C}$와 페어링**하여 증류 손실을 계산한다. 페어링 확률은 timestep $t$에 의존하는 함수 $p(t)$가 결정한다.
3. **이미지프리·데이터 효율적 증류**: 이 기법을 이용하면, **(i) 페어드 이미지를 거의 사용하지 않고도** 광범위한 텍스트 조건 공간을 탐색할 수 있고, **(ii) 학생 모델이 학습 이미지에 한 번도 등장하지 않은 개념(예: 동물)을 생성**할 수 있게 되며, **(iii) 자원 제약 환경에서 BK-SDM 등 SOTA 압축 기법을 능가**한다.

본 논문은 위 세 가지를 ‘three-fold contributions’로 명시하고 있다. 즉 (a) 조건부 확산모델의 KD 일반화 한계를 발견한 통찰, (b) 이를 해결하는 random conditioning 기법, (c) 그것을 활용한 image-free 압축 파이프라인 설계.

---

## 2. 자세한 논문 분석

### 2.1 문제 정의: 조건부 확산모델의 압축과 이미지프리 설정의 어려움

목표는 대규모로 학습된 교사(teacher) 확산모델 $\mathcal{T}$의 지식을 (잠재적으로 다른 구조와) 훨씬 적은 매개변수를 가진 학생 모델 $\mathcal{S}$에게 이식하는 것이다. 본 논문에서는 텍스트→이미지 모델인 Stable Diffusion v1.4를 실증 사례로 다룬다. 이 작업은 일반적인 **diffusion acceleration (sampling step 감소)**과는 다른 차원의 문제이다. 후자는 동일 모델이 더 적은 step으로 추론하도록 만드는 데 초점을 두는 반면, 본 논문은 *모델 크기 자체*를 줄이는 것이 목표이며, 둘은 상호 보완적이다.

표준 KD는 각 timestep $t$에서 노이즈된 입력 $\mathbf{x}\_t$를 가지고 교사가 예측한 노이즈 ${\epsilon}_\mathcal{T}(\mathbf{x}_t,c,t)$를 학생이 모방하도록 한다. 그런데 이미지프리 설정에서는 **$\mathbf{x}_t$를 만들기 위한 원본 이미지 $\mathbf{x}_0$가 없으므로 $t \neq T$에 대해 직접적인 KD가 불가능**하다. $t=T$일 때 $\mathbf{x}_T \approx \mathcal{N}(\mathbf{0},\mathbf{I})$이므로 단지 그 시점에서만 손실 계산이 가능하지만, 이는 확산 모델의 본질적 학습이 다중 timestep에 걸친 점진적 디노이징이라는 점에서 매우 불충분하다.

또한 조건부 확산모델의 매핑은 **의미 조건공간 $\mathcal{C}$ → 매우 큰 이미지 공간**으로 이뤄지는 일대다 함수에 가까우며, 출력 노이즈는 *현재 입력에 매우 특화*되어 있어, 인식 모델처럼 클래스 간 관계가 풍부하게 인코딩된 soft target과는 달리, **조건 간 관계 정보가 거의 노출되지 않는다**. 그래서 학생이 보지 못한 조건을 일반화하기 어려운 것이다.

### 2.2 Naïve Baseline Approach

가장 단순한 접근은, 텍스트 프롬프트 집합으로부터 교사 모델을 사용해 **모든 프롬프트에 대한 이미지 $\mathbf{x}^n$을 생성·캐싱**하여 페어드 데이터셋 $\mathcal{D}=\{(\mathbf{x}^n,c^n)\}_{n=1}^{N}$을 구축한 뒤, 표준 KD 손실로 학습하는 것이다.

출력 단(output-level) 손실은 다음과 같이 정의된다:

$$
\mathcal{L}_{\text{out}} \;=\; \mathbb{E}_{(\mathbf{x}_t,c)\in\mathcal{D},\,t}\!\left[\,\big\lVert {\epsilon}_{\mathcal{T}}(\mathbf{x}_t,c,t)-{\epsilon}_{\mathcal{S}}(\mathbf{x}_t,c,t)\big\rVert_2^2 \,\right].
$$

여기에 BK-SDM과 같이 중간 특징(feature-level) 정합 손실을 결합한다:

$$
\mathcal{L}_{\text{feat}} \;=\; \mathbb{E}_{(\mathbf{x}_t,c)\in\mathcal{D},\,t}\!\left[\,\sum_{l}\big\lVert \mathbf{f}_{\mathcal{T}}^{l}(\mathbf{x}_t,c,t) - \mathbf{f}_{\mathcal{S}}^{l}(\mathbf{x}_t,c,t)\big\rVert_2^2 \,\right].
$$

여기서 $l$은 U-Net의 각 블록 출력층 인덱스이다. 학생-교사가 채널 차원이 다를 경우, 학습 시 임시 1×1 convolution 형태의 projection 모듈을 부착하여 차원을 맞추고, 증류 후에는 폐기한다.

이 단순 접근의 한계는 명백하다.

- **이미지 생성 비용**: 조건공간 $\mathcal{C}$가 매우 크기 때문에 모든 프롬프트에 대해 이미지를 생성하는 것은 비현실적이다. SD-v1.4는 50-step 샘플링으로 한 장 생성 시 수 초가 걸리며, 수십~수백만 프롬프트로 확장하면 비용이 기하급수적으로 증가한다.
- **조건공간 미커버 문제**: 만약 일부 조건만 페어드 이미지로 커버되면, 학생은 *나머지 조건에 대해 사실상 학습되지 않는다*(Fig. 2의 MNIST 실험에서 ‘3’이 빠지면 ‘3’을 절대 생성하지 못함).

### 2.3 Random Conditioning: 핵심 수식 및 직관

Random conditioning은 이 두 한계를 동시에 해소한다. 매우 큰 텍스트 집합 $\mathcal{C}$($M$개)와 작은 페어드 집합 $\mathcal{D}=\{(\mathbf{x}^n,c^n)\}$($N\!\ll\!M$)이 주어졌을 때, 학습 시 다음과 같이 *조건을 무작위로 교체*한다:

```math
\hat{c} \;=\;
\begin{cases}
c^{n}, & \text{with probability } 1-p(t), \\[2pt]
\tilde{c} \in \mathcal{C}, & \text{with probability } p(t),
\end{cases}
```

여기서 $\tilde{c}$는 $\mathcal{C}$에서 균등하게 샘플링된 임의 텍스트이다. 이렇게 만들어진 $\hat{c}$를 사용해 $\mathcal{L}\_{\text{out}}$, $\mathcal{L}_{\text{feat}}$를 계산한다. 즉:

```math
\mathcal{L} = \mathbb{E}_{(\mathbf{x}^n,c^n)\sim\mathcal{D},\,t,\,\hat{c}\sim q(\hat{c}\mid c^n,t)}\left[\big\lVert {\epsilon}_{\mathcal{T}}(\mathbf{x}_t,\hat{c},t) - {\epsilon}_{\mathcal{S}}(\mathbf{x}_t,\hat{c},t)\big\rVert_2^2 + \sum_l \big\lVert \mathbf{f}_{\mathcal{T}}^{l}-\mathbf{f}_{\mathcal{S}}^{l}\big\rVert_2^2\right].
```

### 2.4 $p(t)$ 함수 설계

$p(t)$의 선택은 임상적으로 매우 중요하다. 저자들은 상수, 시그모이드, 선형, 지수형 등을 비교했다고 보고하며(부록 D), 본문 실험에서는 **지수함수 형태**를 채택했다. 직관적으로 다음의 두 한계 영역에서의 거동이 핵심이다.

| Timestep 구간 | 직관 | $p(t)$ 권장값 |
|---|---|---|
| $t \to 0$ (저잡음) | $\mathbf{x}_t$가 거의 $\mathbf{x}_0$이며 *조건은 무시되고 디노이징에 집중*됨 → 이미지·조건 정합이 중요해도 큰 영향이 적음 | 작거나 중간 |
| 중간 timestep | 이미지 정체성과 조건이 혼합되어 작동, **페어링이 가장 중요**한 영역 | $p(t)$가 **상대적으로 낮을수록 성능 개선** |
| $t \to T$ (고잡음) | $\mathbf{x}\_t$가 사실상 노이즈, 모델은 거의 전적으로 $c$ 에 의존 | 1에 가까움 |

저자들이 $p(t)$를 지수형으로 채택한 핵심 이유는 (i) 큰 $t$에서 $\mathbf{x}_t \to \mathcal{N}(0,\mathbf{I})$이므로 $p(\mathbf{x}_t\mid c^n)$와 $p(\mathbf{x}_t\mid\tilde{c})$ 분포가 거의 일치하고, (ii) 작은 $t$에서는 모델이 조건에 약하게 의존하므로 random condition에 대해서도 학생이 교사의 미세 디노이징을 모방할 수 있기 때문이다(Fig. 3, 5). 즉 두 극한에서 random conditioning이 무해하거나 유익하며, 중간에서만 페어링을 강하게 유지하면 충분하다.

상수 $p(t)\!=\!1$을 사용하면 성능이 차선이라는 점은 중간 timestep에서의 정합이 학습 안정성에 필수적임을 시사한다.

### 2.5 Observations and Motivation

저자들의 motivation은 두 관찰에 기반한다.

- **Fig. 3 (MNIST/MS-COCO)**: 동일 $\mathbf{x}_0$로부터 forward로 $\mathbf{x}_t$를 만든 뒤, 원래 라벨이 아닌 다른 조건을 부여해 reverse하면, 작은 $t$에서는 출력이 *원래 이미지 라벨*에 정렬되고, 큰 $t$에서는 *부여된 조건*에 정렬된다. 중간 어떤 좁은 구간에서만 아티팩트가 생긴다. 이는 조건과 입력이 *항상 정합되어야 할 필요는 없다*는 것을 정량적으로 보여준다.
- **Fig. 5 (2D toy)**: $t$가 커질수록 $p(\mathbf{x}_t\!\mid\!c^n)$과 $p(\mathbf{x}_t\!\mid\!\tilde{c})$ 분포가 점진적으로 겹쳐, $t\to T$에서는 동일 가우시안으로 수렴한다. 즉 random conditioning은 $t$가 클수록 *통계적으로 무손실*에 가깝다.

이 두 관찰이 random conditioning이 단순한 트릭이 아니라 확산 과정의 본질적 통계 구조에 부합하는 합리적 설계임을 뒷받침한다.

### 2.6 모델 구조: Teacher = SD-v1.4, Student = BK-SDM 계열 + 채널 압축형 4종

본 연구에서 사용된 학생 구조는 두 가지 압축 전략으로 설계된다.

| 모델 | 압축 방식 | 매개변수 수(전체) | 교사 가중치 초기화 | 비고 |
|---|---|---|---|---|
| B-Base | UNet 블록 제거 (BK-SDM-Base와 동일 구조) | 0.76B | 가능 | BK-SDM 기준선과 1:1 비교 |
| B-Small | UNet 블록 제거 (BK-SDM-Small과 동일) | 0.66B | 가능 | mid-stage 추가 제거 |
| B-Tiny | UNet 블록 제거 (BK-SDM-Tiny와 동일) | 0.50B | 가능 | 가장 적극적인 블록 제거 |
| C-Base | 채널 폭 축소(layer 수 유지) | 0.73B | 불가(채널 dim 불일치) | 모든 layer 보존 |
| C-Small | 채널 폭 축소 | 0.61B | 불가 | |
| C-Tiny | 채널 폭 축소 | 0.49B | 불가 | |
| C-Micro | 채널 폭 매우 적극적 축소 | **0.40B** | 불가 | B-Tiny보다 30% 더 작음 |

채널 압축형은 BK-SDM 류의 블록 제거형이 가지지 못하는 *연속적 압축률 제어*가 가능하다는 장점이 있으나, 교사와 채널 차원이 달라 가중치 초기화를 사용할 수 없다. 본 논문은 이 핸디캡에도 불구하고 random conditioning이 *교사 가중치 초기화 없이도* 강한 성능을 낼 수 있음을 실증한다.

### 2.7 실험 결과 분석

#### (a) Random Conditioning의 핵심 효과 (Table 1, MS-COCO 30K)

| # | Rand Cond | Teacher Init | Real Image | FID↓ | IS↑ | CLIP↑ |
|---|---|---|---|---|---|---|
| 1 | ✗ | ✗ | ✗ | 18.13 | 31.84 | 0.2728 |
| 2 | ✗ | ✓ | ✗ | 18.15 | 33.81 | 0.2864 |
| 3 | ✗ | ✓ | ✓ | 15.76 | 33.79 | 0.2878 |
| 4 | ✓ | ✗ | ✗ | **15.46** | **34.48** | 0.2834 |
| 5 | ✓ | ✓ | ✗ | 15.76 | 36.03 | 0.2895 |
| 6 | ✓ | ✓ | ✓ | **15.00** | **36.14** | **0.2933** |

핵심 관찰:
- Row 1 vs 4: random conditioning만 추가해도 FID 14.72% 감소, IS 8.29% 증가.
- Row 4 vs 5/6: 교사 초기화 추가 효과는 점진적·작음. random conditioning이 가장 큰 단일 요인.
- Row 5 vs 6: real image 사용 여부의 차이가 매우 적다 — **이미지 없이도 거의 동등한 성능**을 달성.
- 특히 Row 3은 BK-SDM-Base와 동일한 설정인데, Row 5(이미지를 쓰지 않는 우리 모델)이 이를 모든 metric에서 능가.

#### (b) Unseen Concept (Animal) 실험 (Table 2)

24K 동물 관련 이미지를 LAION 212K 학습셋에서 제거한 188K 비-동물 학습셋으로 학생을 학습한 뒤, *이미지 없이 텍스트만* 추가한 random conditioning 효과를 검증.

| # | Rand Cond | Additional Texts | Seen FID↓ | Seen IS↑ | Seen CLIP↑ | Unseen FID↓ | Unseen IS↑ | Unseen CLIP↑ |
|---|---|---|---|---|---|---|---|---|
| Teacher | — | — | 13.29 | 32.47 | 0.2954 | 22.53 | 18.63 | 0.3035 |
| 1 | ✗ | None | 15.24 | 28.11 | 0.2801 | **37.86** | 17.73 | 0.2478 |
| 2 | ✓ | 24K animal-related | 14.42 | 27.86 | 0.2788 | **23.26** | 17.18 | 0.2833 |
| 3 | ✓ | 24K + 20M LAION | 15.37 | 30.27 | 0.2879 | 24.71 | 17.39 | **0.2913** |

특징:
- Row 1 → Row 2: 단지 *이미지 없는 텍스트 24K*를 random conditioning으로 추가하기만 해도 동물(unseen) FID가 **37.86 → 23.26**으로 38.6% 개선되며, 교사(22.53)에 거의 근접.
- 텍스트 데이터를 20M로 늘리면 CLIP score가 unseen에서 0.2913으로, seen에서도 0.2879로 둘 다 향상.
- Seen 카테고리에서도 random conditioning은 항상 성능을 끌어올려, unseen만이 아니라 *전반적 생성 품질의 정규화 효과*가 있음을 시사.

#### (c) 데이터프리 (LLM-합성 캡션) 실험 (Table 3)

| # | Rand Cond | Data Source | FID↓ | IS↑ | CLIP↑ |
|---|---|---|---|---|---|
| Teacher | — | — | 13.05 | 36.76 | 0.2958 |
| 1 | ✗ | LAION | 18.15 | 33.81 | 0.2864 |
| 2 | ✓ | LAION + 20M extra | 15.76 | 36.03 | 0.2896 |
| 3 | ✓ | **GPT-generated 2.2M** | **14.98** | **36.70** | **0.2952** |

LLM이 생성한 가상 캡션만 사용해도(즉 실 텍스트조차 없이도), 페어드 LAION 텍스트로 학습한 경우와 동등하거나 더 우수. 이는 데이터 저작권/프라이버시가 극단적으로 제약된 상황에서도 본 기법이 적용 가능함을 보여준다.

#### (d) 다른 SOTA 모델과의 비교 (Table 4, MS-COCO 30K)

| 모델 | #Params | #Real Images | FID↓ | IS↑ | CLIP↑ |
|---|---|---|---|---|---|
| SDM-v1.4 (Teacher) | 1.04B | >2000M | 13.05 | 36.76 | 0.2958 |
| Small SD | 0.76B | 229M | 12.76 | 32.33 | 0.2851 |
| BK-SDM-Base | 0.76B | 0.22M | 15.76 | 33.79 | 0.2878 |
| BK-SDM-Small | 0.66B | 0.22M | 16.98 | 31.68 | 0.2677 |
| BK-SDM-Tiny | 0.50B | 0.22M | 17.12 | 30.09 | 0.2653 |
| **B-Base (Ours)** | 0.76B | **0** | 14.47 | 36.50 | 0.2932 |
| **B-Small (Ours)** | 0.66B | **0** | 16.22 | 35.99 | 0.2804 |
| **B-Tiny (Ours)** | 0.50B | **0** | 16.71 | 35.46 | 0.2782 |
| C-Base (Ours) | 0.73B | 0 | 14.45 | 34.92 | 0.2904 |
| C-Small (Ours) | 0.61B | 0 | 14.43 | 34.58 | 0.2888 |
| C-Tiny (Ours) | 0.49B | 0 | **13.90** | 33.18 | 0.2860 |
| **C-Micro (Ours)** | **0.40B** | **0** | **13.42** | 32.64 | 0.2813 |
| GLIDE | 3.5B | 250M | 12.24 | — | — |
| LDM-KL-8-G | 1.45B | 400M | 12.63 | 30.29 | — |
| DALL·E-2 | 5.2B | 250M | 10.39 | — | — |
| SnapFusion | 0.99B | >100M | ≈13.6 | — | ≈0.295 |
| SDXL-Base-1.0 | 3.5B | — | 12.15 | 35.12 | 0.3199 |

핵심 발견:
- **C-Micro (0.40B, real image 0장)** 이 BK-SDM-Small (0.66B, 0.22M images)를 모든 metric에서 능가.
- B-Base는 BK-SDM-Base와 동일 구조, 0개 실 이미지 사용에도 FID 14.47 (vs 15.76), IS 36.50 (vs 33.79), CLIP 0.2932 (vs 0.2878)로 모두 우위.
- **FID 13.42**의 C-Micro는 SnapFusion(>100M images)이나 LDM-KL-8-G(400M)와 견줄 만한 수준.

> **주의**: FID는 표본분포·평가 코드 차이에 민감하다. 저자들은 부록 F에서 FID의 비단조적 거동을 언급하고 있어, 모든 metric을 종합적으로 보아야 한다는 점을 명시한다.

### 2.8 한계점

- **단일 teacher 검증**: 주 실험이 SD-v1.4에 한정. SDXL/SD3에 대한 검증은 부록 H의 KOALA 기반 추가 실험에 그치며, 더 큰 모델군 전반에 대한 일반성은 추가 검증이 필요.
- **모달리티 제한**: 텍스트→이미지에 한정. 비디오·오디오·3D 확산 모델로의 확장은 미실증.
- **$p(t)$ 의존성**: 함수형(지수형)의 선택이 휴리스틱이며, 자동 탐색이나 데이터 적응형 스케줄링은 미연구.
- **CLIP score 트레이드오프**: 일부 채널 압축 변형(C-Tiny, C-Micro)에서 CLIP가 BK-SDM 대비 큰 이득은 아니며, 텍스트-이미지 정렬 측면에서는 모델 크기를 줄일수록 성능 압박이 존재.
- **추론 가속과의 결합 미검증**: LCM/CTM 같은 step distillation과의 합성 실험은 향후 과제.

---

## 3. 일반화 성능 향상 가능성에 대한 중점 논의

본 연구의 가장 의미 있는 기여는 *압축 자체를 넘어*, **조건부 확산모델의 KD 일반화 메커니즘을 새롭게 제시했다**는 점이다.

### 3.1 Unseen Concept 일반화의 메커니즘

Random conditioning이 unseen 일반화를 가능하게 하는 이유는 다음과 같이 구조화할 수 있다.

1. **조건공간 탐색의 확장**: 학생이 받는 학습 신호는 $(\mathbf{x}\_t,\hat{c})$ 페어이며, $\hat{c}$가 $\mathcal{C}$ 전체에서 균등 샘플링되므로, 학생은 *교사의 조건 임베딩 ↔ 출력 노이즈 매핑*을 모든 조건에 걸쳐 학습한다.
2. **고잡음 영역의 ‘무손실’ 신호**: 큰 $t$에서 $\mathbf{x}\_t$는 거의 가우시안이므로 ${\epsilon}_\mathcal{T}(\mathbf{x}_t,\tilde{c},t)$가 통계적으로 *합법적*인 교사 출력이며, 학생은 이로부터 unseen 조건에 대한 거의 정확한 KD 신호를 얻는다.
3. **저잡음 영역의 ‘디노이징 보존’ 신호**: 작은 $t$에서는 모델이 조건을 거의 무시하므로, $\hat{c}$의 부정합이 학습에 큰 해가 되지 않는다.
4. **인식 모델 KD와의 본질적 차이를 보완**: 인식 모델은 soft target이 클래스 간 관계를 인코딩해 unseen 일반화를 자연스럽게 제공하지만, 확산 모델의 노이즈 출력은 입력 특정적이므로 그런 자연 일반화가 일어나지 않는다. Random conditioning은 *조건 다양성 자체를 학습 분포에 강제로 주입*해 이 격차를 메운다.

### 3.2 적은 이미지로 더 좋은 성능

Table 4와 Table 1의 비교에 따르면, 188~212K 이미지 캐시(BK-SDM과 동일 또는 더 적은 양)으로 학습된 본 모델이 BK-SDM과 동등 이상 성능을 달성한다. 부록 A의 데이터 효율성 분석은 이미지 수를 더 줄이는(예: 10K~50K) 실험을 보여주며, **random conditioning은 캐시 사이즈가 줄수록 baseline 대비 상대적 개선폭이 커진다**고 보고된다(부록 자료에 따른 정성적 추세이며, 정확한 수치는 부록 표를 참고). 이는 **캐시(이미지) 자원이 줄수록 random conditioning이 더 큰 데이터 증강 효과**를 발휘함을 시사한다.

### 3.3 다른 아키텍처에서의 적용 가능성

저자들은 **부록 H에서 KOALA(SDXL 기반 압축)에 random conditioning을 결합**한 실험을 보고하며, 부록 I에서는 **SLIM(spectral diffusion) 구조에도 적용**하여 이득을 확인한다(논문 부록 기술). 즉 random conditioning은 *학생 아키텍처와 무관한 일반적 학습 기법*으로, U-Net 블록 제거형(BK-SDM), 채널 축소형(본 논문 C-시리즈), self-attention KD형(KOALA), 주파수 분해형(SLIM) 모두에 호환된다.

### 3.4 다른 모달리티로의 확장 가능성

Random conditioning의 핵심 가정은 **(i) 조건이 noised input과 항상 정합될 필요 없음, (ii) forward 분포가 timestep에 따라 condition independent에 수렴**이라는 두 점인데, 이는 다음 모달리티에서 모두 성립한다.

| 모달리티 | 적용성 | 고려사항 |
|---|---|---|
| 비디오 확산 (text→video) | 매우 높음 | 시간 차원이 추가되어 캐시 비용이 더 큼 → 더 큰 효용 기대 |
| 오디오 확산 (text→audio) | 높음 | 텍스트 조건공간이 좁을 수 있음 |
| 3D / NeRF 확산 | 중간 | 3D ground truth가 희소하므로 image-free의 가치 큼 |
| 클래스 조건 이미지 확산 (CIFAR/ImageNet) | 직접 적용 가능 | 조건이 카디널리티 작음(10~1000), random conditioning의 조건공간 탐색 효과는 줄어듦 |
| ControlNet 류 (구조 조건) | 비자명 | 구조적 조건은 입력 정합성이 더 강함 → $p(t)$ 설계 재검토 필요 |

논문 결론부에서 저자들은 "future works include extending this approach to diffusion models for other modalities"라고 명시하고 있다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려사항

### 4.1 자원 제약 환경에서의 확산 모델 활용 확대

본 논문은 *모바일/엣지·소규모 연구실 환경에서 SOTA 확산 모델을 재현·압축하는 표준 워크플로우*를 제공한다. SnapFusion, BK-SDM, KOALA가 보여준 ‘작은 학생, 큰 교사’ 구도에 random conditioning을 결합하면, 데이터 수집 비용이 가장 큰 제약일 때(예: 의료, 위성, 산업 도메인) 매우 유용하다.

### 4.2 데이터 프라이버시·저작권 회피

LAION 등 대규모 웹 크롤링 데이터셋의 저작권 문제가 점점 대두되는 상황에서, **이미지를 일체 사용하지 않고 텍스트만으로 SOTA 교사의 능력을 이전**할 수 있다는 점은 산업적으로 큰 의미를 갖는다. Table 3의 결과는 **캡션조차 LLM으로 합성**해 fully data-free 설정으로 갈 수 있음을 보여주며, 이는 확산 모델 라이선스의 경계를 새롭게 정의할 가능성을 시사한다.

### 4.3 LLM 합성 캡션의 잠재력

GPT-생성 캡션 2.2M으로도 LAION 20M 텍스트 보강과 거의 동등 성능이 나오는 것은, **LLM이 효과적인 ‘조건공간 sampler’ 역할**을 한다는 강력한 증거다. 이는 향후 다음 방향을 시사한다:
- 도메인 특화 LLM 캡션을 통한 *target style 강제 증류* (예: 만화, 의료 영상).
- 다양성 제어가 가능한 *prompt distribution learning*과의 결합.

### 4.4 다른 도메인 적용 시 고려사항

- **조건 정합성 강도**: ControlNet과 같이 입력과 조건이 강한 의존을 가지는 경우, random conditioning은 적절한 $p(t)$ 스케줄 재설계가 필요.
- **교사 품질 한계**: 교사 자체가 빈약하면 random conditioning이 보강 가능한 정보가 적다. 교사 중심 평가가 선행되어야 한다.

### 4.5 $p(t)$ 함수의 자동 학습

향후 핵심 과제 중 하나는 $p(t)$를 **메타학습/베이즈 최적화로 자동 탐색**하는 것이다. 예를 들면, $p(t)$를 신경망으로 매개변수화하고, validation FID를 메타-목적함수로 하는 differentiable scheduling을 시도할 수 있다.

### 4.6 One-step / Few-step distillation과의 결합

본 연구는 명시적으로 *모델 크기 압축*에 한정되며, LCM(Latent Consistency Model), DMD, SiD, SwiftBrush 등 *step 압축*과 직교한다. 결합 가능성은 매우 높다:

- **Random conditioning + DMD/SiD**: 학생을 DMD 손실로 1-step 생성기로 만들면서, 동시에 random conditioning으로 더 많은 텍스트 조건을 커버하는 multi-stage distillation.
- **Random conditioning + Consistency Distillation**: $f(\mathbf{x}\_t,t,c)\to f(\mathbf{x}_{t'},t',c)$ self-consistency 학습 시, $c$를 random conditioning으로 다양화하면 일반화 강화 가능.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5-A. Diffusion 모델 압축 / 지식증류

| 방법 | 발표 (회의/연도) | 압축 대상 | 핵심 기법 | 사용 데이터 | Real Image | 모델 크기 | MS-COCO FID | 강점 | 약점 |
|---|---|---|---|---|---|---|---|---|---|
| **BK-SDM** (Kim et al.) | ECCV 2024 | SD v1.4/v2.1 | Block pruning + feature KD + output KD | 0.22M LAION | ✓ | 0.50–0.76B | 15.76(Base) | 단일 stage 학습, 13 A100 days, 적은 data | unseen 일반화 약함, real image 필요 |
| **KOALA-700M/1B** (Lee et al.) | NeurIPS 2024 | SDXL | layer-wise removal + **self-attention KD** | 공개 데이터 | ✓ | 0.78B/1.16B | (SDXL 비교) | self-attn 분석으로 큰 압축률(54–69%), step-distilled teacher 활용 | SDXL 의존, 데이터 사용 |
| **SnapFusion** (Li et al.) | NeurIPS 2023 | SD v1.5 | efficient UNet + **step distillation** + decoder distillation | >100M | ✓ | ≈13.6 | 모바일 2초 미만 추론, on-device 최초 | 강한 데이터/연산 요구 |
| **SLIM (Spectral Diffusion)** (Yang et al.) | CVPR 2023 | LDM (unconditional/conditional) | wavelet gating + **spectrum-aware distillation** | 표준 데이터 | ✓ | 도메인별 | 8–18× 연산 감소, 주파수 편향 분석 | T2I 대규모 미검증 |
| **Diff-Pruning** (Fang et al.) | NeurIPS 2023 | DDPM 등 | Taylor expansion **over pruned timesteps** | 일부 | ✓ | — | re-training 거의 불필요, 50% FLOPs 감소 | 텍스트 조건부 SOTA SD에 직접 적용 어려움 |
| **DKDM** (Xiang et al.) | arXiv 2024 / CVPR 2025 | DM (any architecture) | data-free, **dynamic iterative distillation** | 0 (data-free) | ✗ | — | architecture-agnostic, 경우에 따라 전체 데이터 학습 모델 능가 | 텍스트 조건부 대규모 모델 검증 부족 |
| **Random Conditioning (본 논문)** | CVPR 2025 | SD v1.4 | output+feature KD + **random text condition swapping with $p(t)$** | 0.22M generated images + 20M texts | ✗ (image-free) | 0.40–0.76B | **13.42 (C-Micro)** ~ 14.47 (B-Base) | image-free, unseen 일반화, 데이터프리(LLM 캡션)도 가능 | SD v1.4 위주, $p(t)$ 휴리스틱 |

**관찰**: BK-SDM은 본 논문의 직계 baseline이며, 동일 구조에서 random conditioning만 추가해 모든 metric을 개선한다. KOALA는 SDXL 압축의 SOTA로서 self-attention KD가 핵심이라는 다른 통찰을 제공하며, 본 논문과 결합 가능성이 높다(부록 H 검증). DKDM은 ‘image-free’를 또 다른 방식(dynamic iterative distillation)으로 추구한 동시기 작업이며, 본 논문과 함께 *data-free diffusion KD*의 최신 흐름을 형성한다.

### 5-B. Diffusion Acceleration (sampling step 감소)

| 방법 | 발표 | 단계 수 | 핵심 기법 | Image data 필요 | 1-step FID (COCO/IN) | 강점 | 약점 |
|---|---|---|---|---|---|---|---|
| **DDIM** (Song et al.) | ICLR 2021 | 10–50 | 결정론적 ODE 솔버, training-free | — | — | 학습 불필요 | 1-step 어려움 |
| **DPM-Solver** (Lu et al.) | NeurIPS 2022 | 10–25 | 고차 ODE 솔버 | — | — | training-free, fast | 매우 적은 step에서 품질 저하 |
| **Progressive Distillation** (Salimans & Ho) | ICLR 2022 | 1–8 | 반복적 절반 step 증류 | ✓ | (CIFAR) | 안정적 학습 | 다단계, 데이터 필요 |
| **Consistency Models** (Song et al.) | ICML 2023 | 1–4 | 동일 trajectory 일관성 매칭 | ✓ | — | 1-step, 통일된 프레임워크 | training-from-scratch 비용 |
| **Latent Consistency Model (LCM)** (Luo et al.) | 2023 | 2–4 | LDM에 consistency distillation 적용 | ✓ | (도메인별) | SD 기반 빠른 1–4 step | 추가 데이터 필요 |
| **CTM (Consistency Trajectory Model)** (Kim et al.) | ICLR 2024 | 1+ | consistency + score 통합 | ✓ | SOTA on CIFAR | 일관성·점수 학습 통합 | 텍스트 T2I 적용 추가 작업 |
| **Rectified Flow / InstaFlow** (Liu et al.) | 2022/2024 | 1 | flow-straightening + reflow | ✓ | (COCO) | 직진 확률 흐름 | 다단계 reflow 비용 |
| **DMD / DMD2** (Yin et al.) | CVPR 2024 / NeurIPS 2024 | 1 | distribution matching (KL via score difference) | ✓ (real image regression) | **11.49 (COCO-30k, SD-v1.5)** | 1-step + 고품질 | 두 score net + real image |
| **SDXL-Turbo (ADD)** (Sauer et al.) | 2023 | 1–4 | adversarial diffusion distillation | ✓ | — | 빠른 SDXL 1-step | 데이터·discriminator 의존 |
| **SDXL-Lightning** (Lin et al.) | 2024 | 1–8 | adversarial + progressive | ✓ | — | open-weight SDXL 가속 | 데이터 의존 |
| **SwiftBrush** (Nguyen & Tran) | CVPR 2024 | 1 | **image-free** variational score distillation (VSD) | **✗** | **16.67 (COCO-30k)** | image-free 1-step T2I | base model 그대로 (압축 X), 대규모 텍스트 필요 |
| **SiD / Guided SiD** (Zhou et al.) | ICML 2024 / arXiv 2024 / ICLR 2025(workshop) | 1 | **data-free** score identity distillation | **✗** | (CIFAR/ImageNet/AFHQ SOTA; SD-v1.5에서도 우수) | 지수적 FID 감소, 일부에서 교사 능가 | T2I 1.5/2.1까지는 검증, 대규모 SDXL은 추가 작업(SiD-LSG) |

**비교 결론**: 본 논문은 *모델 크기 압축* 라인에 위치하며, *step 압축* 라인과 직교적·상보적이다. 흥미로운 점은 SwiftBrush와 SiD가 **‘image-free’/‘data-free’ acceleration**을 추구하는 반면, 본 논문은 **‘image-free’ size compression**을 추구한다는 점이다. 이 둘을 결합하면 *image-free size compression + image-free step distillation*이라는 가장 자원 효율적 파이프라인이 가능하다.

### 5-C. 데이터 효율적 / 이미지프리 지식증류

| 방법 | 발표 | 데이터 | Image-free | Compression target | 핵심 손실 | 일반화 검증 | 강점 |
|---|---|---|---|---|---|---|---|
| **BOOT** (Gu et al.) | ICML 2023 Workshop | data-free | ✓ | step compression (1–few) | bootstrapping between consecutive steps | 일부 T2I | 데이터 없는 step distillation |
| **SwiftBrush** | CVPR 2024 | text-only | ✓ | step (1-step) | VSD (text-to-3D 영감) | COCO-30K | image-free 1-step, 16.67 FID |
| **SiD / Guided SiD** | ICML 2024 / 2024 | data-free | ✓ | step (1-step) | model-based Fisher divergence (3 score identities) | CIFAR/ImageNet/SD1.5 | 데이터 없이 교사 능가 가능 |
| **DKDM** | 2024/CVPR 2025 | data-free | ✓ | size (any architecture) | dynamic iterative distillation | 픽셀·잠재 공간 | architecture-agnostic, 새로운 패러다임 |
| **BK-SDM** (의 데이터 효율성 측면) | ECCV 2024 | 0.22M LAION | ✗ | size | output + feature KD | T2I (Seen) | 데이터 0.1% 미만으로 SD 모방 |
| **General Data-Free KD (예: DAFL, ZSKD, DeepInversion)** | 2019–2022 | data-free | ✓ | recognition (분류) | model-inversion, GAN-style synth | CIFAR/ImageNet | 분류 분야 패러다임 정립 |
| **Random Conditioning (본 논문)** | CVPR 2025 | text-only or LLM-synth | ✓ | size | output + feature KD with random text swap | **unseen 개념 직접 검증** | 이미지 없이 unseen 조건 일반화, $p(t)$로 timestep-aware 정규화 |

**핵심 차별점**:
- **BOOT/SwiftBrush/SiD**는 모두 *step 압축*에 초점 → 학생 아키텍처는 보통 교사와 동일 크기.
- **DKDM**은 본 논문과 동시기 *size 압축 image-free* 작업이지만, 픽셀/잠재 공간 unconditional/class-conditional DPM이 주된 검증 대상이며, T2I 대규모 모델에 대한 검증은 제한적.
- **본 논문**은 T2I SD-v1.4에 대해 image-free, **unseen 개념 일반화를 정량적으로 입증**한 최초 사례 중 하나.

---

## 6. 종합 평가 및 향후 전망

본 논문은 “*조건부 확산모델의 KD에서 학생이 unseen condition을 처리하지 못한다*”는 거의 다뤄지지 않은 문제를 명확히 정의하고, 그 원인을 조건공간 탐색의 부재로 규명한 뒤, **timestep-aware random conditioning**이라는 단순하지만 이론적·실증적으로 견고한 해결책을 제시한다. 그 결과 다음을 동시에 달성한다.

1. **이미지프리 학습**: BK-SDM과 동일 구조에서 0.22M의 *생성된 이미지*만으로(즉, 실 이미지 0장으로) BK-SDM을 모든 metric에서 능가.
2. **Unseen 일반화**: 동물 이미지를 학습에서 완전 배제해도, 이미지 없이 텍스트만 추가한 random conditioning만으로 동물 FID를 37.86→23.26으로 회복.
3. **데이터프리 가능성**: LLM-합성 캡션만으로도 LAION 텍스트와 동등 성능 달성.
4. **압축 한계 확장**: C-Micro(0.40B)가 BK-SDM-Small(0.66B)을 능가, 50% 더 작은 UNet에서도 SOTA-급 품질 유지.

향후 연구는 다음 축에서 진행될 것으로 보인다:
- **$p(t)$의 학습화·메타학습화**.
- **다른 모달리티(video/audio/3D) 적용**과 모달리티별 $p(t)$ 디자인 원칙 정립.
- **Step distillation(LCM, DMD, SiD, SwiftBrush)와의 결합**으로 *image-free, compact, 1-step* 모델 달성.
- **SDXL/SD3 등 차세대 모델로의 확장** 및 ControlNet 류 구조 조건과의 호환성 분석.
- **이론적 분석**: random conditioning이 minimize하는 손실의 *암묵적 분포 정규화*(implicit regularization)를 정보이론·분포정합 관점에서 정형화.

요컨대 본 연구는 *image-free conditional diffusion distillation*이라는 하위 분야를 확립한 ‘기준점’ 논문이라 평가할 수 있으며, 데이터 라이선스/저작권/프라이버시 이슈가 첨예해지는 2025–2027년의 연구·산업 환경에서 그 영향력은 더욱 확대될 것이다.

---

## 참고 문헌 / 출처

본 논문 (대상)
- Kim, Dohyun; Park, Sehwan; Han, Geonhee; Kim, Seung Wook; Seo, Paul Hongsuck. **Random Conditioning with Distillation for Data-Efficient Diffusion Model Compression**. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025*, pp. 18607–18618. arXiv:2504.02011. https://arxiv.org/abs/2504.02011 ; 프로젝트 페이지: https://dohyun-as.github.io/Random-Conditioning ; 코드: https://github.com/dohyun-as/Random-Conditioning

압축 / 지식증류 (Section 5-A)
- Kim, Bo-Kyeong et al. **BK-SDM: A Lightweight, Fast, and Cheap Version of Stable Diffusion**. *ECCV 2024*. arXiv:2305.15798. https://arxiv.org/abs/2305.15798 (Nota-NetsPresso/BK-SDM, GitHub).
- Lee, Youngwan; Park, Kwanyong; Cho, Yoorhim; Lee, Yong-Ju; Hwang, Sung Ju. **KOALA: Empirical Lessons Toward Memory-Efficient and Fast Diffusion Models for Text-to-Image Synthesis** (a.k.a. *KOALA: Self-Attention Matters in Knowledge Distillation of Latent Diffusion Models for Memory-Efficient and Fast Image Synthesis*). *NeurIPS 2024*. arXiv:2312.04005.
- Li, Yanyu et al. **SnapFusion: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds**. *NeurIPS 2023*. arXiv:2306.00980.
- Yang, Xingyi; Zhou, Daquan; Feng, Jiashi; Wang, Xinchao. **Diffusion Probabilistic Model Made Slim** (Spectral Diffusion). *CVPR 2023*. arXiv:2211.17106.
- Fang, Gongfan; Ma, Xinyin; Wang, Xinchao. **Structural Pruning for Diffusion Models** (Diff-Pruning). *NeurIPS 2023*. arXiv:2305.10924.
- Xiang, Qianlong; Zhang, Miao; Shang, Yuzhang; Wu, Jianlong; Yan, Yan; Nie, Liqiang. **DKDM: Data-Free Knowledge Distillation for Diffusion Models with Any Architecture**. *CVPR 2025* (arXiv:2409.03550, 2024).

Diffusion Acceleration (Section 5-B)
- Song, Jiaming; Meng, Chenlin; Ermon, Stefano. **Denoising Diffusion Implicit Models (DDIM)**. *ICLR 2021*.
- Lu, Cheng et al. **DPM-Solver**. *NeurIPS 2022*.
- Salimans, Tim; Ho, Jonathan. **Progressive Distillation for Fast Sampling of Diffusion Models**. *ICLR 2022*.
- Song, Yang et al. **Consistency Models**. *ICML 2023*.
- Luo, Simian et al. **Latent Consistency Models**. arXiv:2310.04378, 2023.
- Kim, Dongjun et al. **Consistency Trajectory Models (CTM)**. *ICLR 2024*.
- Liu, Xingchao et al. **Flow Straight and Fast / Rectified Flow**, **InstaFlow**.
- Yin, Tianwei; Gharbi, Michaël; Zhang, Richard; Shechtman, Eli; Durand, Frédo; Freeman, William T.; Park, Taesung. **One-step Diffusion with Distribution Matching Distillation (DMD)**. *CVPR 2024*. arXiv:2311.18828. (후속 DMD2 also referenced.)
- Sauer, Axel et al. **Adversarial Diffusion Distillation (SDXL-Turbo)**. 2023.
- Lin, Shanchuan et al. **SDXL-Lightning**. 2024.
- Nguyen, Thuan Hoang; Tran, Anh. **SwiftBrush: One-Step Text-to-Image Diffusion Model with Variational Score Distillation**. *CVPR 2024*. arXiv:2312.05239.
- Zhou, Mingyuan; Zheng, Huangjie; Wang, Zhendong; Yin, Mingzhang; Huang, Hai. **Score identity Distillation (SiD): Exponentially Fast Distillation of Pretrained Diffusion Models for One-Step Generation**. *ICML 2024*. arXiv:2404.04057. 후속 *Long and Short Guidance in SiD for One-Step Text-to-Image Generation*, arXiv:2406.01561 (Guided SiD).

데이터 효율적 / 이미지프리 KD (Section 5-C)
- Gu, Jiatao; Zhai, Shuangfei; Zhang, Yizhe; Liu, Lingjie; Susskind, Josh. **BOOT: Data-free Distillation of Denoising Diffusion Models with Bootstrapping**. *ICML 2023 Workshop on SPIGM*. arXiv:2306.05544.
- (BK-SDM, DKDM, SwiftBrush, SiD, Guided SiD: 위 항목 재인용)
- 일반 data-free KD 분야 (인식): DAFL, ZSKD, DeepInversion 등 (2019–2022) — 본 분석에서는 비교 패러다임 참조.

기타 기반 모델
- Rombach, Robin et al. **Stable Diffusion v1.4 / Latent Diffusion Models**, *CVPR 2022* (LDM-KL-8-G).
- Podell, Dustin et al. **SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis**, 2023.
- Hinton, Geoffrey; Vinyals, Oriol; Dean, Jeff. **Distilling the Knowledge in a Neural Network**, NeurIPS Workshop 2015 (MNIST '3' 사례).
- Stable Diffusion 3 / Stable Diffusion 3.5 (Stability AI, 2024–2025).
- Würstchen-v2, Pixart-alpha 등 비교 모델은 본 논문 Table 4의 보고치 인용.

> 표에 보고된 일부 수치(특히 SnapFusion FID ≈ 13.6, SDXL FID 12.15 등)는 본 대상 논문 Table 4에서 직접 인용한 값이며, 평가 프로토콜 차이로 출처 논문 원문 수치와 미세 차이가 있을 수 있다. 또한 FID는 평가 코드·이미지 해상도·통계 sample 수에 민감하므로 절대 비교에는 주의가 필요하다(본 논문 부록 F가 명시).
