
# Scalable Autoregressive Monocular Depth Estimation (DAR) 

---

## 1. 핵심 주장 및 주요 기여 요약

### 🔑 핵심 주장

이 논문은 **오토회귀(Autoregressive, AR) 모델이 효과적이고 확장 가능한 단안 깊이 추정기(Monocular Depth Estimator)**임을 입증합니다.

저자들은 지식이 허락하는 한, 단안 깊이 추정(MDE)을 위한 **최초의 오토회귀 모델인 DAR**을 제안하며, 핵심 통찰은 MDE의 두 가지 순서 속성인 **깊이 맵 해상도(Resolution)**와 **깊이 세분화(Granularity)**를 오토회귀 목적함수로 변환하는 데 있습니다.

### 📌 주요 기여 (3가지)

① 기존 인코더-디코더 모델의 저수준·고수준 특징 융합 과정을 **저해상도→고해상도 오토회귀 목적함수**로 재정의하고, 패치 단위 인과 마스크(patch-wise causal mask)를 사용하는 새로운 **Depth Autoregressive Transformer**를 도입합니다.

② MDE 태스크를 오토회귀 빈(bin) 시퀀스 예측 태스크로 변환하는 새로운 빈 분할 전략인 **MTBin(Multiway Tree Bins)**을 제안합니다.

③ 이 두 오토회귀 목적함수를 결합하여 **KITTI 및 NYU Depth v2에서 새로운 SOTA**를 달성하며, 모델을 **2.0B 파라미터까지 확장**하여 KITTI에서 RMSE 1.799를 기록하였습니다(기존 SOTA Depth Anything 대비 5% 향상).

---

## 2. 해결 문제, 제안 방법, 모델 구조, 성능 및 한계

### 🔴 2.1 해결하고자 하는 문제

**오토회귀(AR) 아키텍처**는 다양한 태스크에서 강력한 일반화 능력과 확장성을 보여 왔으나, 기존 MDE는 이러한 패러다임을 적극적으로 활용하지 못했습니다. AR 모델은 스케일링 법칙에 따라 모델 크기 확장을 통해 최적 성능을 달성할 수 있는 유연성을 제공합니다.

이에 자연스럽게 제기되는 질문은 "**AR 모델을 MDE 태스크에 적용할 수 있는가?**"이지만, 오토회귀 모델링은 각 단계의 예측이 이전 단계와 논리적으로 연결되는 잘 정돈된 순차적 데이터 형식에 의존한다는 도전이 있습니다.

---

### 🟡 2.2 제안하는 방법 (수식 포함)

저자들은 MDE의 두 가지 "순서" 속성을 발견하고 이를 오토회귀 목적함수로 변환합니다.

#### (a) 해상도 오토회귀 목적함수 (Resolution Autoregressive Objective)

깊이 맵 생성은 **저해상도에서 고해상도 순서**를 따를 수 있습니다. 해상도 오토회귀 과정의 각 단계에서 Transformer는 이전 모든 토큰 맵에 조건부로 다음 고해상도 토큰 맵을 예측합니다.

수식으로 표현하면, $K$개의 해상도 단계에서 $k$번째 해상도의 깊이 토큰 맵 $r_k$에 대한 조건부 확률은:

$$P(r_1, r_2, \ldots, r_K \mid I) = \prod_{k=1}^{K} P(r_k \mid r_1, r_2, \ldots, r_{k-1}, I)$$

여기서 $I$는 입력 RGB 이미지이며, 패치 단위 인과 마스크(patch-wise causal mask)를 통해 자기회귀 순서를 보장합니다.

#### (b) 세분화 오토회귀 목적함수 (Granularity Autoregressive Objective)

깊이 값의 범위는 0에서 특정 최댓값까지 순서가 있습니다. 세분화 오토회귀 과정의 각 단계에서 **빈(bin)의 수를 지수적으로 증가(예: 2배)**시키고, 이전 예측을 활용하여 더 정밀하고 세분화된 깊이를 예측합니다.

**MTBin(Multiway Tree Bins)** 전략의 수식은 다음과 같습니다. $k$번째 단계에서 빈 후보 $c_k$는 이전 예측 $\tilde{D}_{k-1}$으로부터 재귀적으로 생성됩니다:

$$\tilde{D}_k = \sum_{b=1}^{B_k} p_{k,b} \cdot c_{k,b}$$

여기서 $B_k = 2^k \cdot B_0$는 $k$번째 단계의 빈 수, $p_{k,b}$는 소프트맥스 확률, $c_{k,b}$는 $k$번째 빈 후보 값입니다. 이는 **순서형 회귀(ordinal regression)** 방식으로 학습됩니다.

#### (c) 두 목적함수의 결합

출력 잠재 토큰은 **ConvGRU 모듈**로 전달되어 이전 예측 $\tilde{D}_{k-1}$로부터 MTBin이 생성한 새로운 정제된 빈 후보 $c_k$의 프롬프트를 주입받아 세분화 가이던스를 제공하고, 다음 해상도 토큰 맵 $r_k$를 생성합니다. 새로운 깊이 맵 $\tilde{D}_k$는 다음 세분화 빈 후보 $c_k$와 소프트맥스 확률 $p_k$의 **선형 결합**으로 생성되며, 해상도 및 세분화 오토회귀가 동시에 발전합니다.

최종 손실 함수는 해상도 손실과 세분화 손실의 합으로 구성됩니다:

$$\mathcal{L}_{\text{DAR}} = \mathcal{L}_{\text{resolution}} + \lambda \cdot \mathcal{L}_{\text{granularity}}$$

여기서 $\mathcal{L}\_{\text{resolution}}$은 패치 단위 인과 마스크 기반 크로스 엔트로피, $\mathcal{L}_{\text{granularity}}$는 순서형 회귀 손실입니다.

---

### 🟢 2.3 모델 구조

DAR은 다음 세 가지 핵심 컴포넌트로 구성됩니다:


① **저수준→고수준 해상도 오토회귀 목적함수**로 인코더-디코더 특징 융합을 재정의  
② **패치 단위 인과 마스크**를 사용하며 점진적으로 증가하는 해상도에서 깊이 맵을 생성하는 **Depth Autoregressive Transformer**  
③ 세분화 오토회귀 목적함수를 위해 MDE를 오토회귀 빈 시퀀스 예측 태스크로 변환하는 **MTBin(Multiway Tree Bins) 전략**


모델 백본은 다음과 같이 구성됩니다:

```
입력 RGB 이미지 I
      ↓
[인코더: ViT 기반 Backbone (Depth Anything 기반)]
      ↓
[Depth Autoregressive Transformer]
  - patch-wise causal mask
  - 저해상도 → 고해상도 순차 예측
      ↓
[ConvGRU (Bin Injection Module)]
  ↑ MTBin으로 생성된 bin candidates ck
      ↓
[출력: 고해상도 정밀 깊이 맵 D̃K]
```

---

### 🔵 2.4 성능 향상

각 서브 AR 목적함수와 구성 요소가 베이스라인 성능을 향상시키며, 모델 크기를 **2.0B로 확장하면 모든 메트릭에서 최고 성능**을 달성함으로써 DAR의 강력한 확장성을 입증합니다.

| 데이터셋 | 지표 | DAR (2.0B) | 기존 SOTA | 향상률 |
|---|---|---|---|---|
| KITTI | RMSE ↓ | **1.799** | 1.896 (Depth Anything) | **~5%** |
| NYU Depth v2 | 다수 지표 | **SOTA** | — | 명확한 마진 |

특히 객체 경계에서의 깊이 추정이 더 일관적이고 부드러우며, 이는 이전 예측을 다음 단계의 더 정밀한 예측에 활용하는 **오토회귀 점진적 패러다임** 덕분입니다. 또한 의자 다리와 같은 **작고 얇은 객체나 원거리 소형 객체**의 깊이 추정에서도 더 정확한 결과를 보여줍니다.

---

### 🔴 2.5 한계


**① 경계 블러 문제**: 다단계 점진적 패러다임을 적용하면 깊이 맵이 더 부드럽고 연속적이지만, **경계가 흐릿해지고 선명도가 감소**할 수 있습니다.  
**② 높은 파라미터 수**: 오토회귀 Transformer를 사용하기 때문에 DAR의 파라미터 수가 상대적으로 높으며, 특히 모델 크기를 확장할 때 더욱 그러합니다. 그러나 대형 모델 지식 증류 또는 경량 AR 파운데이션 모델 설계 기법을 통해 복잡도-정확도 트레이드오프를 개선할 수 있다고 저자들은 언급합니다.


---

## 3. 모델의 일반화 성능 향상 가능성 🌐

### 3.1 제로샷 일반화 능력

**DAR은 미지의(unseen) 데이터셋에서 제로샷 일반화 능력을 보여줍니다.**

오토회귀(AR) 아키텍처는 이전에 보지 못한 데이터셋에서 탁월한 제로샷·퓨샷 성능을 달성하고, 다양한 다운스트림 태스크에 적응하는 데 있어 상당한 유연성을 제공합니다.

### 3.2 스케일링을 통한 일반화 강화

DAR은 강력한 확장성을 보여주며, 최첨단 방법들 사이에서 더 나은 성능-효율 트레이드오프를 달성합니다.

스케일링 법칙($N$: 파라미터 수, $D$: 학습 데이터 크기)에 따른 일반화 성능:

$$\text{Loss}(N, D) \propto \left(\frac{N_0}{N}\right)^{\alpha} + \left(\frac{D_0}{D}\right)^{\beta}$$

이 법칙에 의해 **모델이 클수록 일반화 성능도 향상**되는 경향이 있으며, DAR의 2.0B 규모 확장이 이를 실증합니다.

### 3.3 AR 기반 구조의 본질적 일반화 이점

DAR은 **기존 오토회귀 기반 파운데이션 모델에 오토회귀 깊이 추정을 통합하는 유망한 방법**을 제시합니다.

이 결과들은 DAR이 오토회귀 예측 패러다임으로 우수한 성능을 달성하며, **GPT-4o와 같은 현대적 오토회귀 대형 모델에 깊이 추정 능력을 부여하는 유망한 접근법**임을 시사합니다.

### 3.4 일반화 성능 향상을 위한 추가 가능성

DAR은 MDE를 위한 AR 기반 프레임워크에 확장성과 일반화를 도입하며, MDE 태스크를 두 가지 병렬 오토회귀 목적함수인 **해상도와 세분화**로 변환하는 것이 핵심 아이디어입니다.

이는 다음과 같은 일반화 향상 가능성을 내포합니다:

| 방향 | 설명 |
|---|---|
| **모델 스케일 확장** | 2.0B → 7B+ 수준으로의 확장 시 추가 일반화 기대 |
| **멀티모달 융합** | 언어 프롬프트·텍스트 조건부 깊이 추정과의 결합 |
| **대규모 비지도 사전학습** | 인터넷 스케일 이미지 데이터로 AR 사전학습 후 파인튜닝 |
| **다중 도메인 학습** | 실내/실외/의료/위성 등 도메인 통합 학습 |

---

## 4. 앞으로의 연구에 미치는 영향 및 고려 사항 🔭

### 4.1 미래 연구에 미치는 영향

**① MDE의 패러다임 전환**

이 결과들은 DAR이 오토회귀 예측 패러다임으로 우수한 성능을 달성하며, GPT-4o와 같은 현대적 오토회귀 대형 모델에 깊이 추정 능력을 부여하는 유망한 접근법임을 시사합니다.

**② 멀티모달 대형 모델과의 통합 가능성**

DAR Transformer와 Multiway Tree Bins 전략을 통해 두 오토회귀 목적함수를 달성하고, Bin Injection 모듈로 이를 연결함으로써, 여러 벤치마크에서 기존 SOTA 방법을 큰 차이로 능가함을 입증하였습니다.

**③ 스케일링 법칙의 컴퓨터 비전 적용 확장**

스케일링 법칙이 시사하듯, AR 모델은 다양한 실용적 응용에서 최적 성능을 위한 유연한 모델 크기 확장을 허용합니다.

### 4.2 앞으로 연구 시 고려해야 할 점

#### 🔶 기술적 고려사항

1. **추론 속도 최적화**  
   오토회귀 Transformer를 사용하기 때문에 DAR의 파라미터 수가 상대적으로 높다는 점을 감안하여, 지식 증류(Knowledge Distillation)나 효율적 어텐션 메커니즘(Flash Attention 등)을 결합한 경량화 연구가 필요합니다.

2. **경계 선명도 개선**  
   다단계 점진적 패러다임이 경계를 흐릿하게 만들고 선명도를 줄일 수 있다는 한계를 극복하기 위해, 경계 인식 손실 함수(Boundary-aware Loss) 또는 고주파 세부 정보 보존 기법의 통합이 중요합니다.

3. **메트릭 깊이 추정으로의 확장**  
   ZoeDepth의 도입이 제로샷 메트릭 깊이 추정에서 중요한 도약을 나타내듯, DAR을 절대 스케일의 메트릭 깊이 추정으로 확장하는 연구가 필요합니다.

4. **비디오 시퀀스로의 확장**  
   AR 모델의 순차적 특성을 활용하여 시간적 일관성(temporal consistency)을 갖춘 비디오 깊이 추정으로 자연스럽게 확장할 수 있습니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석 📊

### 5.1 주요 방법론별 비교표

단안 깊이 추정 모델은 크게 두 범주로 나뉩니다: **상대적 깊이 모델**(DPT, Depth Anything, Marigold)과 **메트릭 깊이 모델**(ZoeDepth, Metric3D, UniDepth).

| 모델 | 연도 | 패러다임 | 핵심 특징 | KITTI RMSE |
|---|---|---|---|---|
| **MiDaS** | 2020 | Discriminative | 다중 데이터셋 혼합 학습 | — |
| **ZoeDepth** | 2023 | Discriminative | 상대·메트릭 깊이 결합, 제로샷 | — |
| **Marigold** | 2024 | Diffusion | 확산 모델 기반, 고세밀도 | — |
| **Depth Anything** | 2024 | Discriminative | 대규모 비지도 데이터 활용 | 1.896 |
| **Depth Pro** | 2024 | Discriminative | 0.3초 고해상도 메트릭 추정 | — |
| **DAR (제안)** | 2024 | **Autoregressive** | 해상도+세분화 AR, 2.0B 확장 | **1.799** |

### 5.2 각 방법론의 특징 분석

**Depth Anything (2024)**
Depth Anything은 **반지도 자기학습 전략** 기반의 대규모 데이터 확장 접근법을 도입하여, 약 6,200만 개의 자기 레이블 샘플을 생성함으로써 다양한 시나리오에서 모델의 일반화 능력을 크게 향상시킵니다.

**Marigold (2024)**
Marigold는 깊이 추정에 확산 모델을 적용한 선구자로, 전통적인 판별 모델에 비해 **우수한 구조적 일관성과 세밀도**를 가진 깊이 맵을 생성합니다. 특히 반사체나 투명한 영역과 같은 어려운 영역을 처리하는 데 강점이 있습니다.

**Depth Pro (2024)**
Depth Pro는 V100 GPU에서 **0.3초 만에 225만 픽셀 해상도의 제로샷 메트릭 깊이 맵**을 생성합니다.

**DepthART (관련 후속 연구)**
오토회귀 생성 접근법, 특히 Visual AutoRegressive 모델링이 조건부 이미지 합성에서 확산 모델보다 우수한 결과를 보이면서 더 빠른 추론 시간을 제공한다는 점에서, DAR과 유사하게 Visual Autoregressive Transformer(VAR)를 단안 깊이 추정에 적용하는 연구들이 등장하고 있습니다.

**VAR-Depth (2024)**
이 연구는 오토회귀 사전(prior)을 기하학 인식 생성 모델의 보완적 패밀리로 확립하며, **데이터 확장성과 3D 비전 태스크 적응성에서의 이점**을 강조합니다.

### 5.3 연구 트렌드

명확한 연구 트렌드는 **미지 도메인에 대한 제로샷 일반화**입니다. ZoeDepth와 UniDepth 같은 접근법들은 아키텍처 혁신과 대규모 학습을 통해 유망한 전이 가능성을 보여주며, 앞으로의 우선순위는 **계산 효율성 향상, 다중 뷰 설정에서의 기하학적 일관성 강화, 도메인 적응 발전**이 될 것입니다.

2024~2025년에 이르러 연구는 **메트릭 정확도와 생성적 적응성을 동시에 추구**하는 UniDepth, Marigold와 같은 범용적이고 확산 기반의 MMDE 모델 개발로 수렴하고 있습니다.

---

## 📚 참고 자료 및 출처

| # | 제목 | 출처 |
|---|---|---|
| 1 | **Scalable Autoregressive Monocular Depth Estimation** (Wang et al., 2024) | [arXiv:2411.11361](https://arxiv.org/abs/2411.11361) |
| 2 | **Scalable Autoregressive Monocular Depth Estimation** (CVPR 2025) | [CVPR 2025 Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Scalable_Autoregressive_Monocular_Depth_Estimation_CVPR_2025_paper.pdf) |
| 3 | **Scalable Autoregressive Monocular Depth Estimation** - 프로젝트 페이지 | [depth-ar.github.io](https://depth-ar.github.io/) |
| 4 | **Scalable Autoregressive Monocular Depth Estimation** - IEEE Xplore | [IEEE Xplore](https://ieeexplore.ieee.org/document/11094590/) |
| 5 | **Visual Autoregressive Modelling for Monocular Depth Estimation** (El-Ghoussani et al., 2024) | [arXiv:2512.22653](https://arxiv.org/abs/2512.22653) |
| 6 | **DepthART: Monocular Depth Estimation as Autoregressive Refinement Task** (Gabdullin et al., 2024) | [arXiv:2409.15010](https://arxiv.org/abs/2409.15010) |
| 7 | **Survey on Monocular Metric Depth Estimation** (Zhang et al., 2025) | [arXiv:2501.11841](https://arxiv.org/html/2501.11841v1) |
| 8 | **Survey on Monocular Metric Depth Estimation** (MDPI Computers, 2025) | [MDPI](https://www.mdpi.com/2073-431X/14/11/502) |
| 9 | **Depth Anything V2** (Yang et al., 2024) | [arXiv HTML](https://arxiv.org/html/2406.09414v1) |
| 10 | **Depth Pro: Sharp Monocular Metric Depth in Less Than a Second** (Bochkovskiy et al., 2024) | [arXiv:2410.02073](https://arxiv.org/html/2410.02073v1) |
| 11 | **Zero-shot Monocular Metric Depth Estimation via Test-time Adaptation** (NeurIPS 2024) | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12922624/) |
| 12 | **Video Depth Anything** (2025) | [arXiv:2501.12375](https://arxiv.org/html/2501.12375v3) |

> ⚠️ **정확도 주의 사항**: 본 답변에서 수식은 논문의 공개된 내용 및 오토회귀 MDE의 일반적 원리를 바탕으로 재구성하였습니다. 논문 원문에 명시적으로 표기되지 않은 일부 수식 표현(예: 손실 함수 $\lambda$ 가중치 등)은 개념적 설명을 위한 것이며, 정확한 구현 세부사항은 [arXiv:2411.11361](https://arxiv.org/abs/2411.11361) 원문 및 [공식 코드](https://depth-ar.github.io/)를 참조하시기 바랍니다.
