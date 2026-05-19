
# NormalCrafter: Learning Temporally Consistent Normals from Video Diffusion Priors

> **논문 정보**
> - 저자: Yanrui Bin et al.
> - arXiv: [2504.11427](https://arxiv.org/abs/2504.11427) (April 15, 2025)
> - 학회: ICCV 2025 (accepted)
> - GitHub: [Binyr/NormalCrafter](https://github.com/Binyr/NormalCrafter)
> - 프로젝트 페이지: [normalcrafter.github.io](https://normalcrafter.github.io/)

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

Surface Normal 추정은 컴퓨터 비전의 핵심 기반 기술이지만, 정적 이미지 기반 연구들은 많이 존재한 반면, **비디오 기반의 시간적 일관성(Temporal Coherence) 확보**는 매우 어려운 문제로 남아 있었다. NormalCrafter는 단순히 기존 방법에 시간적 요소를 추가하는 대신, **Video Diffusion Model의 내재적 시간적 Prior(Temporal Prior)를 활용**한다.

### 주요 기여 (3가지)

| # | 기여 항목 | 설명 |
|---|-----------|------|
| 1 | **NormalCrafter 프레임워크** | 임의 길이의 오픈 월드 비디오에서 고품질·시간 일관 Normal 시퀀스 생성 |
| 2 | **Semantic Feature Regularization (SFR)** | DINO 인코더의 의미론적 표현과 확산 특성을 정렬하여 세밀한 Normal 추정 |
| 3 | **Two-Stage Training Protocol** | 잠재 공간 + 픽셀 공간 학습을 결합한 2단계 훈련 전략 |

NormalCrafter는 기존 접근법들을 상당한 격차로 능가하며, 임의 길이 오픈 월드 비디오에서 세밀하고 시간적으로 일관된 Normal 시퀀스를 생성하는 새로운 프레임워크이다.

이 논문은 **ICCV 2025**에 채택되었다.

---

## 2. 해결하고자 하는 문제, 제안 방법(수식), 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

조명 변화, 카메라 움직임, 장면 역학 등 다양한 요인으로 인해, **다양하고 제약 없는 비디오에서 고충실도 및 시간 일관적인 Normal을 추정하는 것은 매우 어려운 과제**이다.

기존 정적 이미지 방법들은 비디오의 시간적 다이나믹스를 처리하지 못해 프레임 간 불일치 또는 깜박임(Flickering)을 야기한다. 또한, 비디오 확산 모델을 단순히 Normal 추정에 적용하면 **과도한 평활화(Over-smoothing)** 등 최적화되지 않은 결과가 발생한다.

판별적(Discriminative) 접근법은 훈련 데이터의 규모와 품질 한계로 **Zero-Shot 일반화에서 성능이 저하**되며, 생성적(Generative) 방법은 사전 학습된 확산 Prior를 활용해 뛰어난 성능을 달성하지만 정적 이미지에만 설계되어 있다는 한계가 있다.

---

### 2-2. 제안 방법 및 수식

#### (A) 기반 모델: Stable Video Diffusion (SVD) 재목적화

NormalCrafter는 **Stable Video Diffusion(SVD)을 Normal Map 예측을 위해 재목적화**하여, RGB 생성 대신 시간 구조를 유지하면서 Normal을 예측한다.

비디오 Normal 추정 작업을 입력 RGB 프레임에 조건화된 Video Diffusion Model로 모델링한다.

비디오 확산 모델의 기본 학습 목표인 **Diffusion Score Matching (DSM)** Loss는 다음과 같이 정의된다:

$$\mathcal{L}_{DSM} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \left\| \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}) - \boldsymbol{\epsilon} \right\|^2 \right]$$

- $\mathbf{x}_t$: 시간 스텝 $t$에서의 노이즈가 추가된 Normal 잠재 표현
- $\boldsymbol{\epsilon}_\theta$: U-Net 기반 노이즈 예측 네트워크
- $\mathbf{c}$: 입력 RGB 프레임 조건
- $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$: 추가된 가우시안 노이즈

---

#### (B) Semantic Feature Regularization (SFR)

SVD를 단순히 Normal 추정에 적용하면 SVD 특성 내 고수준 의미론적 단서의 부족으로 **과도하게 평활화된 예측**이 생성된다. SFR은 확산 특성을 DINO와 정렬시켜 더욱 세밀하고 정확한 Normal 예측을 달성한다.

SFR은 확산 모델의 **중간 특성(Intermediate Features)을 DINO 의미론적 임베딩과 정렬**시켜, 추론 시 추가 비용 없이 세밀한 기하학적 세부 사항을 향상시킨다.

SFR의 정규화 손실 $\mathcal{L}_{reg}$는 다음과 같이 정의된다:

$$\mathcal{L}_{reg} = \left\| \phi_{diff}(\mathbf{x}_t, t, \mathbf{c}) - \phi_{DINO}(\mathbf{c}) \right\|^2$$

- $\phi_{diff}(\cdot)$: U-Net의 중간 확산 특성
- $\phi_{DINO}(\cdot)$: DINO 인코더에서 추출된 의미론적 표현

Stage 1의 전체 학습 손실은 두 손실의 결합이다:

$$\mathcal{L}_{Stage1} = \mathcal{L}_{DSM} + \lambda \cdot \mathcal{L}_{reg}$$

- $\lambda$: SFR 항의 균형을 조절하는 하이퍼파라미터

---

#### (C) Two-Stage Training Protocol

학습 프로토콜은 두 단계로 구성된다: **1) 확산 스코어 매칭( $\mathcal{L}\_{DSM}$ )과 SFR( $\mathcal{L}_{reg}$ )을 포함한 잠재 공간에서의 전체 U-Net 학습; 2) 각도 손실(Angular Loss)을 포함한 픽셀 공간에서의 공간 레이어만 파인튜닝**이다.

Two-Stage Training은 장기 시간 문맥 모델링과 고정밀 공간 충실도를 균형 있게 유지한다. Stage 1(잠재 공간 학습): 전체 모델을 잠재 공간에서 훈련하여 장기 시간 문맥을 효과적으로 포착한다.

Stage 2의 Angular Loss는 다음과 같이 정의된다:

$$\mathcal{L}_{ang} = \mathbb{E} \left[ 1 - \hat{\mathbf{n}} \cdot \mathbf{n}_{gt} \right]$$

- $\hat{\mathbf{n}}$: 예측된 단위 법선 벡터
- $\mathbf{n}_{gt}$: Ground Truth 단위 법선 벡터

---

### 2-3. 모델 구조

NormalCrafter의 핵심은 **Stable Video Diffusion(SVD)을 Normal Map 예측을 위해 적응**시켜, RGB 생성 대신 시간 구조를 유지하는 것이다.

모델 구조의 핵심 구성 요소:

| 구성 요소 | 설명 |
|-----------|------|
| **Base Model** | Stable Video Diffusion (SVD) U-Net |
| **시간 모듈** | Temporal Convolution + Temporal Attention 레이어 |
| **SFR 모듈** | DINO 인코더 → 중간 확산 특성 정렬 |
| **Stage 1** | 전체 U-Net을 잠재 공간에서 훈련 (시간 일관성 학습) |
| **Stage 2** | 공간 레이어만 픽셀 공간에서 파인튜닝 (공간 정밀도 향상) |

잠재 확산 모델에서 시작하여 시간 합성곱 및 어텐션 레이어를 추가하는 방식으로 발전했으며, Stable Video Diffusion(SVD)은 이를 더욱 정제하여 NormalCrafter를 포함한 다양한 비디오 관련 작업의 모델 Prior로 활용된다.

---

### 2-4. 성능

NormalCrafter는 StableNormal과 Marigold-E2E-FT 대비 **공간적으로 더 정확하고 시간적으로 더 일관된 Normal Map**을 생성하며, y-t 슬라이스의 더 매끄러운 시간 프로파일로 이를 확인할 수 있다.

Marigold-E2E-FT와 비교했을 때, NormalCrafter의 결과는 더 높은 공간 충실도와 시간 일관성을 보여준다.

---

### 2-5. 한계

NormalCrafter는 **고주파 의미론적 세부 사항(High-Frequency, Semantics-Driven Details)** 보존이 필요한 Normal 추정의 어려움에 특화되어 있다. 논문에서 명시적으로 서술된 추가 한계점:

- **확산 모델 기반 추론 속도 문제**: 반복적 디노이징 과정으로 인한 추론 비용
- **합성 데이터 의존성**: 훈련 시 합성 데이터에 의존하는 부분이 있어 실제 도메인 갭 발생 가능성
- **극단적 조명 및 동적 장면의 도전**: 매우 빠른 움직임이나 급격한 조명 변화가 있는 비디오에서 성능 저하 가능

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 생성 Prior의 Zero-Shot 일반화 능력

**판별적(Discriminative) 접근법**은 훈련 데이터 규모와 품질의 한계로 최적이 아닌 Zero-Shot 일반화 성능을 보이는 반면, **생성적(Generative) 방법**은 사전 학습된 확산 Prior를 활용함으로써 합성 훈련 데이터만으로도 오픈 월드 이미지에서 최첨단 성능을 달성한다.

→ NormalCrafter는 SVD의 대규모 비디오 사전 학습 지식을 활용하므로, **훈련 데이터의 도메인 제약을 크게 넘어서는 일반화 능력**을 보유한다.

### 3-2. SFR의 의미론적 정렬이 일반화에 기여하는 이유

SFR은 외부 인코더(DINO)에서 추출된 의미론적 단서와 확산 특성을 정렬함으로써, 모델이 **장면의 내재적 의미론(Intrinsic Scene Semantics)**에 집중하도록 유도한다.

- DINO는 ImageNet 규모 이상의 대규모 데이터로 사전 학습된 Self-Supervised ViT 인코더로, 특정 도메인에 편향되지 않은 강건한 의미론적 특성을 제공함
- 이로 인해 NormalCrafter는 학습 데이터에 없는 장면 유형(Sora 생성 비디오 등)에서도 일반화 성능 발휘

### 3-3. 오픈 월드 비디오 처리 능력

NormalCrafter는 **제약 없는 오픈 월드 비디오**에서 임의 길이의 시퀀스를 처리할 수 있는 시간 일관 Normal 시퀀스 생성 기능을 제공한다.

DAVIS 데이터셋과 **Sora로 생성된 비디오** 모두에서 정성적 비교가 수행되었으며, y-t 슬라이스가 프레임 간 시간 일관성을 확인하는 데 사용되었다.

→ 특히 AI 생성 비디오에서도 일반화 성능이 검증되었다는 점이 중요하다.

### 3-4. 일반화 성능 향상을 위한 잠재적 방향

| 방향 | 설명 |
|------|------|
| **더 강력한 의미론적 Prior 활용** | DINO v2, CLIP 등 더 강력한 외부 인코더로 SFR 강화 |
| **도메인 적응 학습** | 의료 영상, 위성 영상 등 특수 도메인에 대한 파인튜닝 |
| **스케일 확장** | 더 대규모의 합성+실사 복합 데이터셋 활용 |
| **광학 흐름(Optical Flow) 통합** | 동적 장면의 움직임 정보를 명시적으로 활용 |

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4-1. 연구 계보

```
판별적(Discriminative) 방법
  ├── [2024] DSINE: 픽셀별 레이 방향 + 이웃 Normal 관계 모델링
  └── CNN 기반 방법들 (한계: 데이터 품질에 종속, Zero-Shot 일반화 취약)

생성적(Generative) 방법 (2023~)
  ├── [2023~2024] Marigold: SD(Stable Diffusion) 기반 밀집 예측 파인튜닝
  ├── [2024] GeoWizard: SD 기반 깊이+Normal 동시 추정
  ├── [2024] GenPercept: 단일 스텝 디노이징 (빠르나 세밀도 저하)
  ├── [2024] StableNormal: YOSO + SG-DRN 거친→세밀 추정 (이미지 특화)
  └── [2025] NormalCrafter: SVD 기반 비디오 시간 일관 Normal 추정 ★
```

판별적 Surface Normal 추정은 수작업 특성에서 딥러닝 방법으로 발전했으며, DSINE은 픽셀별 레이 방향을 통합하고 인접 Surface Normal 간의 관계를 모델링하여 강력한 기준선을 제공한다.

확산 기반 Surface Normal 추정은 현재 최첨단을 대표하며, Marigold는 이미지에 조건화된 밀집 예측 작업을 위해 Stable Diffusion을 파인튜닝하고, GeoWizard는 SD를 파인튜닝하여 깊이와 Normal Map 모두를 출력한다.

Lotus는 세부 사항 향상을 위한 이미지 재구성 목표를 추가했으며, StableNormal은 더 선명한 결과를 위해 거친→세밀(Coarse-to-Fine) 방식을 사용했다.

### 4-2. 방법론 비교표

| 방법 | 연도 | 유형 | 기반 모델 | 비디오 처리 | 시간 일관성 | 특이점 |
|------|------|------|-----------|-------------|-------------|--------|
| DSINE | 2024 | 판별적 | CNN | ✗ | ✗ | 픽셀별 레이 방향 |
| Marigold | 2023~2024 | 생성적 | SD | ✗ | ✗ | 합성 데이터 학습 |
| GeoWizard | 2024 | 생성적 | SD | ✗ | ✗ | 깊이+Normal 공동 추정 |
| StableNormal | 2024 | 생성적 | SD | 제한적 | ✗ | YOSO+SG-DRN |
| **NormalCrafter** | **2025** | **생성적** | **SVD** | **✓** | **✓** | **SFR + 2단계 학습** |

생성적 방법인 StableNormal은 사전 학습된 확산 Prior를 활용해 오픈 월드 이미지에서 SOTA 성능을 달성하지만, 이러한 방법들은 정적 이미지를 위해 설계되어 비디오에 적용 시 **시간적 불일관성이나 깜박임**을 발생시킨다.

StableNormal은 확산 프로세스를 Normal 추정과 같은 결정론적 작업에 적용할 때의 본질적 충돌을 효과적으로 해결하며, YOSO와 SG-DRN의 결합으로 세밀하고 안정적인 표면 재구성을 가능하게 한다.

---

## 5. 향후 연구에 미치는 영향 및 고려해야 할 점

### 5-1. 연구에 미치는 영향

#### (1) 비디오 기반 3D 기하학 추정 패러다임의 전환
Surface Normal은 3D 재구성, 재조명(Relighting), 비디오 편집, 혼합 현실 등 광범위한 응용 분야의 기반이 된다. NormalCrafter는 비디오 전체에서 일관된 Normal을 제공함으로써, 이러한 응용 분야의 품질과 안정성을 비약적으로 향상시킬 수 있는 기반을 마련한다.

#### (2) Video Diffusion Prior의 기하학적 이해 활용 가능성 증명
NormalCrafter는 이미지 기반 추정기에 단순히 시간 레이어를 증분 추가하는 대신, Video Diffusion Model의 내재적 시간적 Prior를 활용하는 보다 강건한 접근 방식을 입증했다. 이는 **깊이 추정, 광학 흐름, 의미론적 분할 등 다른 비디오 밀집 예측 작업**에도 동일한 패러다임을 적용할 수 있다는 가능성을 열어준다.

#### (3) DINO 기반 의미론적 정렬의 범용성
SFR의 DINO 기반 확산 특성 정렬이 더욱 선명하고 세밀한 Normal 예측을 가능하게 한다는 것은, 의미론적 Self-Supervised 특성이 기하학적 이해를 개선하는 데 효과적임을 증명하며, 다른 생성 모델 기반 연구에도 영향을 줄 것이다.

---

### 5-2. 향후 연구 시 고려해야 할 점

#### ① 추론 속도 최적화
확산 기반 방법의 본질적 한계인 반복 디노이징의 계산 비용을 줄이기 위해:
- Consistency Model이나 Flow Matching 기반의 단일/소수 스텝 추론 적용
- Knowledge Distillation을 통한 경량화 모델 개발

$$\mathcal{L}_{distill} = \mathbb{E} \left[ \left\| f_\theta(\mathbf{x}) - f_{teacher}(\mathbf{x}) \right\|^2 \right]$$

#### ② 더 긴 비디오 시퀀스 처리
NormalCrafter는 잠재 공간과 픽셀 공간 학습을 모두 활용하는 2단계 훈련 프로토콜로 공간 정확도를 보존하며 긴 시간 문맥을 유지하지만, 매우 긴 비디오에서의 일관성 유지 메커니즘(슬라이딩 윈도우, 메모리 효율적 어텐션 등)을 고려해야 한다.

#### ③ 도메인 특화 일반화
장면 배치 변화, 조명 변화, 카메라 움직임, 장면 역학 등의 변동에 대한 일반화는 여전히 과제이며, 이를 위해:
- 다양한 실세계 도메인(수중, 야간, 안개 등 비정상적 환경)에 대한 도메인 적응
- Synthetic-to-Real 갭을 줄이는 훈련 전략 필요

#### ④ 멀티모달 조건 통합
- 텍스트, 카메라 포즈, 깊이 정보 등 보조 조건을 통합한 조건부 Normal 추정
- 이는 특히 3D 재구성 파이프라인과의 결합에서 중요

#### ⑤ 평가 지표의 확장
- 기존 **Mean Angular Error(MAE)**만이 아닌, **시간 일관성 지표(Temporal Consistency Metric)**의 표준화 필요
- 예: $T_{consistency} = \frac{1}{T-1}\sum_{t=1}^{T-1}\cos(\hat{\mathbf{n}}\_t, \hat{\mathbf{n}}_{t+1})$

#### ⑥ 하류 작업(Downstream Tasks)에서의 검증
3D 재구성, 재조명, 비디오 편집, 혼합 현실 등 실제 응용 분야에서의 종단간(End-to-End) 검증을 통해 Normal 품질 향상이 실제 작업 성능에 미치는 영향을 정량화해야 한다.

---

## 참고 자료 (출처)

| # | 제목/출처 | URL |
|---|-----------|-----|
| 1 | **NormalCrafter 논문 (arXiv 2504.11427)** | https://arxiv.org/abs/2504.11427 |
| 2 | **NormalCrafter 논문 PDF** | https://arxiv.org/pdf/2504.11427 |
| 3 | **NormalCrafter HTML (arXiv)** | https://arxiv.org/html/2504.11427v1 |
| 4 | **NormalCrafter 프로젝트 페이지** | https://normalcrafter.github.io/ |
| 5 | **NormalCrafter GitHub** | https://github.com/Binyr/NormalCrafter |
| 6 | **OpenCV Blog: NormalCrafter** | https://opencv.org/blog/normalcrafter/ |
| 7 | **Moonlight Literature Review: NormalCrafter** | https://www.themoonlight.io/en/review/normalcrafter-learning-temporally-consistent-normals-from-video-diffusion-priors |
| 8 | **AIModels.fyi: NormalCrafter** | https://www.aimodels.fyi/papers/arxiv/normalcrafter-learning-temporally-consistent-normals-from-video |
| 9 | **DEV.to: NormalCrafter 분석** | https://dev.to/aimodels-fyi/video-normals-from-diffusion-detail-consistency-for-open-world-footage-oee |
| 10 | **StableNormal 논문 (arXiv 2406.16864)** | https://arxiv.org/html/2406.16864v1 |
| 11 | **Marigold 논문 (arXiv 2505.09358)** | https://arxiv.org/html/2505.09358v1 |
| 12 | **Moonlight: StableNormal** | https://www.themoonlight.io/en/review/stablenormal-reducing-diffusion-variance-for-stable-and-sharp-normal |
| 13 | **ResearchGate: Rethinking Inductive Biases for Surface Normal Estimation** | https://www.researchgate.net/publication/384237434 |
| 14 | **Consensus: NormalCrafter** | https://consensus.app/papers/details/13531cbfc0c45e11994c5a96b8aef2d9/ |

> ⚠️ **정확도 주의 사항**: 수식 중 $\mathcal{L}_{reg}$의 구체적인 구현 형태(코사인 유사도 vs. MSE 등)와 하이퍼파라미터 $\lambda$의 구체적 값은 논문 원문의 PDF 본문에서 확인이 필요합니다. 논문 원문 PDF(https://arxiv.org/pdf/2504.11427)를 직접 열람하여 Section 3의 수식을 검증하실 것을 권장합니다.
