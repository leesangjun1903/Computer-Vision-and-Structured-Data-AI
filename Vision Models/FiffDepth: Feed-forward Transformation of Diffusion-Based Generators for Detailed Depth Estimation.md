
# FiffDepth: Feed-forward Transformation of Diffusion-Based Generators for Detailed Depth Estimation 

> **논문 정보**
> - **저자**: Yunpeng Bai, Qixing Huang (The University of Texas at Austin)
> - **arXiv**: [2412.00671](https://arxiv.org/abs/2412.00671) (v1: 2024.12.01, v2: 2025.03.13)
> - **게재**: ICCV 2025 (OpenAccess 확인)
> - **프로젝트 페이지**: GitHub 공개

---

## 1. 핵심 주장과 주요 기여 요약

### ✅ 핵심 주장

단안 깊이 추정(Monocular Depth Estimation, MDE)은 3D 장면 재구성, 자율 주행, AI 콘텐츠 생성 등 다양한 응용 분야에서 핵심적인 3D 비전 문제이나, 실세계 레이블 데이터 부족과 합성 데이터셋과 실데이터 간의 분포 격차(domain gap)로 인해 강인하고 일반화 가능한 MDE는 여전히 도전적인 과제이며, 기존 방법들은 낮은 효율, 정확도 저하, 세부 묘사 부족 문제를 보인다.

이에 대응하여, FiffDepth는 확산 사전(diffusion prior)을 활용하며, 확산 기반 이미지 생성기를 피드-포워드(feedforward) 아키텍처로 변환하는 프레임워크를 제안한다.

### ✅ 주요 기여 (4가지)

논문의 기여를 정리하면: 1) 확산 모델 궤적(diffusion model trajectory)을 안정적인 피드-포워드 방식으로 활용하여 생성 모델을 밀집 예측 모델로 변환하는 개선된 접근법 제안; 2) DINOv2와 같은 모델의 강력한 일반화 능력을 확산 백본으로 전이하는 새로운 증류(distillation) 방법 도입; 3) 생성 모델 기반의 다른 접근법 대비 더 높은 안정성, 정확도, 효율성 달성; 4) 다른 FFN 모델 대비 더 세밀한 예측 결과 달성.

---

## 2. 해결하고자 하는 문제 / 제안 방법 / 모델 구조 / 성능 및 한계

### 🔴 2-1. 해결하고자 하는 문제

효율성, 정확도, 다양한 실세계 데이터에서의 일반화라는 근본적 도전이 남아 있다. 이는 실세계 깊이 데이터셋이 노이즈가 많고, 합성 데이터와 다양한 실세계 데이터 간 도메인 격차가 존재하기 때문이다.

사전 학습 모델 중 생성 네트워크(generative network)는 DINOv2와 같은 피드-포워드 네트워크(FFN)보다 복잡한 이미지 세부 사항을 더 효과적으로 보존하여 밀집 예측 모델에 더 큰 가능성을 지니지만, 생성 모델은 세부 묘사는 풍부하더라도 합성-실세계 전이에서 제한된 일반화 능력으로 인해 부족함을 보인다.

구체적으로, 기존 확산 기반 MDE 방법(예: Marigold)은 반복적 디노이징 과정을 수행하기 때문에 **추론 속도가 느리고**, 합성 데이터만으로 파인튜닝된 Stable Diffusion 모델은 실세계 데이터에 적용 시 세부 묘사를 보존하면서도 정확한 깊이를 항상 표현하지 못하며, $t=0$에서의 궤적이 부정확하고 혼돈스러운 깊이 예측으로 이어져 제한된 일반화 능력을 드러낸다.

---

### 🔵 2-2. 제안하는 방법 (수식 포함)

FiffDepth의 핵심 아이디어는 **확산 모델의 역방향 궤적(reverse trajectory)을 깊이 도메인으로 확장**하여 단일 순전파(single forward pass)로 깊이 예측을 수행하는 것이다.

#### 📌 확산 모델 기본 프레임워크

표준 DDPM(Denoising Diffusion Probabilistic Model)의 역방향 과정은 다음과 같이 정의된다:

$$
p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \Sigma_\theta(\mathbf{x}_t, t))
$$

여기서 $\mathbf{x}\_t$는 타임스텝 $t$에서의 노이즈가 포함된 잠재 표현이고, $\mu_\theta$는 학습된 평균이다.

#### 📌 FiffDepth의 궤적 확장 (Trajectory Extension)

FiffDepth는 이미지 확산 모델의 궤적을 깊이 도메인으로 확장하는 방법을 핵심으로 하며, 이는 생성 모델 기반 깊이 추정의 정확성과 효율성 모두에서 중요한 발전을 나타낸다. 본 연구는 MDE 태스크에 적합하도록 확산 궤적의 활용을 최적화하는 데 집중하며, 깊이 예측을 위해 파인튜닝될 때 모델이 세부적인 생성 특징을 더 잘 보존할 수 있도록 원래 생성 학습 과정을 보존하면서 원래 생성 모델의 궤적을 최대한 유지하는 것을 목표로 한다.

논문에서 제안하는 궤적 수정 과정(Eq. 4 기반, arXiv HTML v1 참조)은 다음과 같이 표현된다:

$$
\mathbf{d}_{-1} = f_\theta(\mathbf{x}_0, \mathbf{d}_0^{\text{DINO}})
$$

여기서:
- $\mathbf{x}_0$: 이미지 인코더로부터 얻은 잠재 표현
- $\mathbf{d}_0^{\text{DINO}}$: DINOv2가 생성한 pseudo-label 깊이 맵
- $\mathbf{d}_{-1}$: 확장된 궤적의 최종 깊이 출력 ($t=-1$ 단계에 해당)
- $f_\theta$: 파인튜닝된 확산 U-Net 백본

이와 같이 궤적을 수정하기 위해, FiffDepth는 DINOv2로부터 얻은 덜 세밀하지만 일반화된 깊이 도메인으로 궤적을 추가 확장한다.

#### 📌 학습 목표 함수

FiffDepth의 전체 학습 손실은 크게 두 가지 구성 요소로 이루어진다 (논문 본문 기반 재구성):

**① 생성 보존 손실 (Generative Preservation Loss)**:

$$
\mathcal{L}_{\text{gen}} = \mathbb{E}_{t, \mathbf{x}_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(\mathbf{x}_t, t)\|^2\right]
$$

- 원래 확산 모델의 생성 능력과 세부 묘사를 보존하기 위한 표준 노이즈 예측 손실.

**② DINOv2 증류 손실 (DINOv2 Distillation Loss)**:

$$
\mathcal{L}_{\text{DINO}} = \left\|\mathbf{d}_{-1} - \mathbf{d}_0^{\text{DINO}}\right\|_2^2
$$

- DINOv2의 일반화 능력을 확산 백본으로 전이하기 위한 손실.

**전체 손실**:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{gen}} + \lambda \cdot \mathcal{L}_{\text{DINO}}
$$

여기서 $\lambda$는 두 손실 사이의 균형을 조절하는 하이퍼파라미터이다.

> ⚠️ **주의**: 위 수식은 논문의 HTML 버전 및 본문 내용을 기반으로 재구성한 것이며, 논문에서의 정확한 수식 표기와 다소 다를 수 있습니다. 정확한 수식은 [arXiv PDF](https://arxiv.org/pdf/2412.00671)를 직접 참조하시기 바랍니다.

---

### 🟢 2-3. 모델 구조

#### 전체 파이프라인

FiffDepth는 사전 학습된 확산 모델을 피드-포워드 방식의 깊이 예측으로 변환하며, $t=-1$에서의 결과만을 활용하고, DINOv2가 생성한 pseudo-label을 감독에 활용한다.

주요 구성 요소:

| 구성 요소 | 역할 |
|---|---|
| **Stable Diffusion U-Net (backbone)** | 이미지 특징 추출 및 깊이 예측 |
| **DINOv2 (teacher)** | 일반화된 pseudo-label 생성 |
| **Trajectory Extension Module** | 확산 궤적을 깊이 도메인으로 확장 |
| **Feed-forward Decoder** | 단일 순전파로 깊이 맵 출력 |

생성 네트워크는 DINOv2와 같은 피드-포워드 네트워크보다 복잡한 이미지 세부 사항을 더 효과적으로 보존하여 밀집 예측 모델에 더 큰 가능성을 지닌다.

이 하이브리드 접근법은 생성 모델의 세부 보존 능력과 DINOv2의 강력한 일반화 성능을 결합하여 MDE 모델의 강건성과 정확도를 향상시킨다.

---

### 🟡 2-4. 성능 향상

어파인-불변(affine-invariant) 깊이 평가를 위해 Marigold와 동일한 데이터셋 및 평가 프로토콜을 사용하였으며, 이는 NYUv2, ScanNet, KITTI, ETH3D, DIODE를 포함한다. FiffDepth는 제로샷 일반화 능력을 주장하는 14가지 방법과 비교되었다.

FiffDepth는 대부분의 테스트 시나리오에서 최고 또는 최신 기술과 비견되는 결과를 달성하였다. 특히 본 방법은 상대적 깊이 관계를 정확하게 예측할 뿐 아니라 매우 미세한 객체의 깊이 식별 및 예측에도 탁월한 성능을 보인다.

또한 Depth Anything v2가 도입한 DA-2K 벤치마크에서도 평가를 수행하였다.

비교 대상 방법으로는 **Marigold**, **DepthFM**, **Lotus**, **Depth Anything v2**, **DINOv2 기반 방법** 등이 포함된다.

---

### 🔴 2-5. 한계

기존 방법들은 낮은 효율, 제한된 일반화, 불충분한 세부 묘사 보존 문제가 남아 있으며, FiffDepth 역시 다음과 같은 잠재적 한계를 지닌다:

1. **DINOv2 pseudo-label 의존성**: 증류 과정이 DINOv2의 출력 품질에 의존하므로, DINOv2가 실패하는 장면(예: 야간, 특수 조명)에서 성능 저하 가능성이 있다.
2. **합성 데이터 기반 학습**: 현재 MDE 연구는 주로 고품질 어노테이션과 제어된 환경 덕분에 합성 데이터에 의존하지만, 합성 데이터셋의 규모와 다양성은 포괄적인 학습에 여전히 불충분하다.
3. **절대 깊이(metric depth) 예측 불가**: 어파인-불변 상대 깊이만을 예측하는 구조로, 실제 스케일 복원이 필요한 응용에는 추가 처리가 필요하다.

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

### 핵심 전략: DINOv2 궤적 증류

생성 모델이 다양한 실세계 환경에서 강건성을 유지하는 데 한계가 있음을 인식하여, FiffDepth의 접근법은 궤적을 DINOv2가 예측한 깊이 도메인까지 확장하여 DINOv2의 탁월한 일반화 능력을 통합한다.

DINOv2는 합성 데이터로 훈련되었음에도 실세계 이미지에 효과적으로 일반화할 수 있음이 관찰된다.

이 전략의 핵심 메커니즘을 수식으로 표현하면:

$$
\mathbf{d}_{-1} \xrightarrow{\text{supervision}} \mathbf{d}_0^{\text{DINO}} \quad \text{(DINOv2 pseudo-label)}
$$

이 과정에서 중요한 점은 **세부 묘사 특징과 일반화 능력의 분리(decoupling)**이다:
- $t=0$: 확산 모델의 세부 묘사 풍부한 표현 보존
- $t=-1$: DINOv2의 일반화 능력 주입 (추가 확장 단계)

DINOv2와 같은 모델의 강력한 일반화 능력을 통합함으로써, FiffDepth는 향상된 정확도, 안정성, 세밀한 세부 묘사를 달성하여 다양한 실세계 시나리오에서 MDE 성능의 유의미한 향상을 제공한다.

### 왜 DINOv2가 일반화에 효과적인가?

DINOv2는 대규모 자기-지도 시각 기반 모델로, 수작업 어노테이션 없이 강건하고 전이 가능한 시각 특징을 생성하며, Vision Transformer(ViT) 백본 기반으로 1억 4,200만 개의 다양한 이미지로 구성된 큐레이팅된 데이터셋에서 학습되어 이미지-레벨 및 패치-레벨 목표를 결합한다.

### 일반화 향상의 수학적 관점

FiffDepth의 일반화 향상은 다음의 **편향-분산 트레이드오프(bias-variance tradeoff)** 관점에서 이해할 수 있다:

$$
\text{GeneralizationError} = \underbrace{\text{Bias}^2}_{\text{DINOv2로 교정}} + \underbrace{\text{Variance}}_{\text{확산 모델로 감소}} + \sigma^2
$$

- **확산 모델**은 세부 묘사 보존으로 분산(variance)을 낮추지만 합성→실세계 편향(bias)이 크다.
- **DINOv2**는 편향(bias)을 줄이지만 세부 묘사(detail) 손실이 있다.
- **FiffDepth**는 두 모델의 장점을 상호 보완적으로 결합한다.

---

## 4. 향후 연구에 미치는 영향과 고려 사항

### 📌 4-1. 연구에 미치는 영향

#### (A) 확산 모델의 피드-포워드 변환 패러다임 정립
확산 기반 이미지 생성기를 피드-포워드 아키텍처로 변환하여, DINOv2와 같은 모델의 강력한 일반화 능력을 보존하는 방향성을 제시한다. 이는 반복적 디노이징 없이도 생성 모델의 강점을 활용할 수 있음을 보여주는 중요한 패러다임 전환이다.

#### (B) 지식 증류의 새로운 활용
FiffDepth는 DINOv2의 강건한 일반화 능력을 확산 백본으로 전이하는 새로운 증류 방법을 도입하였다. 이 접근법은 깊이 추정을 넘어 표면 법선 추정, 광학 흐름, 의미론적 분할 등 다른 밀집 예측 태스크에도 확장 적용 가능하다.

#### (C) 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 연도 | 핵심 특징 | FiffDepth와의 차이점 |
|---|---|---|---|
| **Marigold** | 2024 | 확산 모델 기반 MDE, 반복 디노이징 | 느린 추론 속도, 세부 묘사 강하지만 일반화 제한 |
| **DepthFM** | 2024 | Flow Matching 기반 MDE | 최소 2회 함수 평가 필요, 직선 궤적으로 확산 기반 대비 빠름 |
| **Lotus** | 2024 | 확산 기반 시각 기반 모델 | 고품질 밀집 예측에 중점, FiffDepth는 일반화+세부 묘사 동시 강조 |
| **Depth Anything v2** | 2024 | FFN 기반, 대규모 레이블 데이터 | 세부 묘사에서 FiffDepth보다 부족 |
| **DINOv2 MDE** | 2023 | 자기-지도 특징 + MDE | 일반화 강하지만 생성적 세부 묘사 부족 |
| **FiffDepth** | 2024 | 확산 궤적 + DINOv2 증류 | 위 모두의 장점 결합, 단일 순전파 |

---

### 📌 4-2. 향후 연구 시 고려할 점

#### ① 절대 깊이(Metric Depth) 확장
현재 FiffDepth는 어파인-불변 상대 깊이만을 예측한다. 자율 주행 등 실용 응용을 위해서는 **스케일 복원 모듈** 또는 **메트릭 깊이 헤드** 추가가 필요하다.

$$
d_{\text{metric}} = s \cdot d_{\text{affine}} + t, \quad s, t \in \mathbb{R}
$$

#### ② 더 강력한 Teacher 모델 활용
DINOv2는 혁신적인 이중 목표 손실과 스케일링된 Vision Transformer 아키텍처를 사용하는 자기-지도 시각 기반 모델로, 효율적인 어텐션 커널과 확률적 깊이(stochastic depth) 등의 고급 학습 기법을 사용한다. 향후에는 DINOv2보다 더 강력한 기반 모델(예: SAM2, CLIP 기반 모델)을 teacher로 활용하는 연구가 유망하다.

#### ③ 동영상/시간적 일관성 확장
현재는 단일 이미지 기반 추론에 집중하고 있어, 비디오 깊이 추정에서의 **시간적 일관성** 유지가 중요한 과제이다.

#### ④ 효율적 백본 탐색
확산 U-Net은 파라미터가 많아 경량화된 생성 백본(예: LDM lite, DiT-small 등)을 활용한 효율적 버전 개발이 필요하다.

#### ⑤ 다중 모달 확장
깊이 + 표면 법선 + 의미론적 분할 등 **다중 밀집 예측 태스크의 통합** 모델로의 확장이 유망한 연구 방향이다.

---

## 📚 참고 자료 및 출처

| # | 제목 / 출처 | URL |
|---|---|---|
| 1 | **FiffDepth** (arXiv) | https://arxiv.org/abs/2412.00671 |
| 2 | **FiffDepth** (arXiv HTML v2) | https://arxiv.org/html/2412.00671v2 |
| 3 | **FiffDepth** (arXiv HTML v1) | https://arxiv.org/html/2412.00671v1 |
| 4 | **FiffDepth** (arXiv PDF) | https://arxiv.org/pdf/2412.00671 |
| 5 | **FiffDepth** (HuggingFace) | https://huggingface.co/papers/2412.00671 |
| 6 | **FiffDepth** (OpenReview, ICCV 2025) | https://openreview.net/forum?id=reqF08spwl |
| 7 | **FiffDepth** (ICCV 2025 OpenAccess) | https://openaccess.thecvf.com/content/ICCV2025/... |
| 8 | **FiffDepth** (ResearchGate) | https://www.researchgate.net/publication/386373796 |
| 9 | **DepthFM** (arXiv:2403.13788) | https://arxiv.org/html/2403.13788v1 |
| 10 | **DINOv2** (arXiv:2304.07193) | https://arxiv.org/abs/2304.07193 |
| 11 | Marigold — "Repurposing diffusion-based image generators for monocular depth estimation" | Ke et al., 2024 |
| 12 | Lotus — "Diffusion-based visual foundation model for high-quality dense prediction" (arXiv:2409.18124) | He et al., 2024 |
| 13 | "Progressive distillation for fast sampling of diffusion models" (arXiv:2202.00512) | Salimans & Ho, 2022 |

> ⚠️ **정확도 안내**: 본 답변은 arXiv 공개 논문 전문 및 관련 웹 소스를 기반으로 작성되었습니다. 일부 수식(특히 전체 손실 함수의 정확한 형태)은 논문 본문 내용을 기반으로 재구성된 것이므로, 정확한 수식은 반드시 **arXiv PDF 원문**을 직접 확인하시기 바랍니다.
