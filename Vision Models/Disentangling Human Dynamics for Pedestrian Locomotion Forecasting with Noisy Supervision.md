
# Disentangling Human Dynamics for Pedestrian Locomotion Forecasting with Noisy Supervision

> **논문 정보**
> - **저자**: Karttikeya Mangalam, Ehsan Adeli, Kuan-Hui Lee, Adrien Gaidon, Juan Carlos Niebles
> - **소속**: Stanford University, Toyota Research Institute, UC Berkeley
> - **학회**: IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2020), pp. 2784–2793
> - **arXiv**: [1911.01138](https://arxiv.org/abs/1911.01138)
> - **특허**: US20210097266A1 / US11074438

---

## 1. 핵심 주장 및 주요 기여 요약

### 🎯 핵심 주장

이 논문은 **Human Locomotion Forecasting** 문제, 즉 에고센트릭(egocentric) 설정에서 근미래의 인체 여러 키포인트(keypoint)의 공간적 위치를 공동으로 예측하는 과제를 다룹니다.

기존 연구들이 포즈 예측(pose prediction) 또는 궤적 예측(trajectory forecasting) 중 하나만 독립적으로 해결하려 한 것과 달리, 이 논문은 두 문제를 통합하는 프레임워크를 제안하여 실세계의 보행자 로코모션(locomotion) 예측이라는 실용적인 과제를 해결합니다.

인간의 동역학(dynamics) 또는 로코모션은 인체의 여러 키포인트들의 공동 공간 이동으로 정의되며, 이는 대규모 궤적 움직임과 세밀한 신체 사지 움직임 사이의 상호작용의 최종 산물입니다.

---

### 🏆 주요 기여 (Two-Fold Contribution)

**첫 번째**: 글로벌(global)과 로컬(local) 모션의 **Disentanglement(분리)**를 활용하여 전체 예측 복잡도를 줄이고 노이즈 지도학습(noisy supervision) 하에서도 학습이 가능하도록 하는 **포즈 완성(pose completion) 및 분해(decomposition) 모듈**을 제안합니다.

**두 번째**: Encoder-Recurrent-Decoder 구조에 기반하여 에고모션(egomotion)이라는 도메인 특화 신호를 활용해 서로 다른 세분화 스트림(granularity streams)을 예측하는 **새로운 에고센트릭 궤적 예측 네트워크**를 포함한 포즈 예측 모듈을 제시하며, 이 두 스트림을 병합하여 최종 로코모션 예측을 수행합니다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 🔴 2-1. 해결하고자 하는 문제

#### (1) 태스크 분리 문제
기존의 인간 동역학 예측 연구들은 궤적(trajectory)과 로컬 포즈/관절 움직임(pose/joint motion)을 별도로 분리하여 예측해 왔으며, 이 논문은 이 두 가지를 통합하는 단일 문제, 즉 로코모션 예측 문제로 묶는 것을 목표로 합니다.

#### (2) 데이터 희소성 및 노이즈 문제
이 과제를 해결하는 데 있어 주요 어려움 중 하나는 포즈, 깊이(depth), 에고모션에 대한 조밀한 주석이 있는 에고센트릭 비디오 데이터셋의 희소성입니다. 이를 극복하기 위해, 최첨단 모델들로 (노이즈가 포함된) 주석을 자동 생성하고, 이 노이즈 지도학습으로부터 학습할 수 있는 강인한 예측 모델을 제안합니다.

---

### 🟢 2-2. 제안하는 방법 (수식 포함)

#### Step 1. 포즈 완성 모듈 (Pose Completion Module)

완성 모듈은 누락된 키포인트 주석을 채워 넣고(fills in the missing key-point annotations), 분해 모듈은 정제된 로코모션을 **글로벌(trajectory)**과 **로컬(pose keypoint movements)**로 분리합니다.

주어진 시간 $t$에서의 관측된 포즈 시퀀스를 $\mathbf{P}\_{1:T_{obs}} = \{p_1, p_2, \ldots, p_{T_{obs}}\}$라 하면, 각 $p_t \in \mathbb{R}^{J \times 2}$ ($J$: 키포인트 수)이며, 누락 관절에 대해 포즈 완성 모듈은 다음을 목표로 합니다:

$$\hat{p}_t = \mathcal{F}_{complete}(p_t, m_t)$$

여기서 $m_t$는 관절의 가시성(visibility) 마스크입니다.

> ⚠️ **주의**: 위 수식은 논문의 전반적인 개념에 기반한 정형화 표현이며, 논문 원문의 정확한 수식 표기와 다를 수 있습니다.

---

#### Step 2. 분해 모듈 (Decomposition Module)

보행자의 관절들의 동시적 움직임을 **글로벌 스트림(Global Stream)**과 **로컬 스트림(Local Stream)**으로 분리합니다. 글로벌 모션 스트림은 차량 카메라에 대한 보행자의 위치의 대규모 이동을 모델링하고, 로컬 스트림은 글로벌 스트림에 대한 인체 움직임을 인코딩합니다.

이를 수식으로 표현하면:

$$p_t = g_t + l_t$$

$$g_t \in \mathbb{R}^{2}: \text{글로벌 궤적 좌표}$$

$$l_t \in \mathbb{R}^{J \times 2}: \text{글로벌 기준 로컬 포즈 잔차(residual)}$$

---

#### Step 3. 예측 모듈 (Prediction Module)

**Quasi-RNN**을 백본(backbone)으로 사용하여, 에고모션(egomotion)과 깊이(depth)와 같은 저수준(low-level) 시각 도메인 특화 신호를 활용해 글로벌 궤적을 예측하는 새로운 계층적 궤적 예측 네트워크를 제안합니다.

이 태스크는 에고센트릭 뷰에서의 인간 로코모션 예측을 **시퀀스-투-시퀀스(sequence-to-sequence)** 문제로 정의합니다.

전체 예측 파이프라인을 수식으로 정리하면:

$$\hat{G}_{T_{obs}+1:T_{pred}} = \mathcal{F}_{global}\left(\mathbf{g}_{1:T_{obs}}, \mathbf{e}_{1:T_{obs}}, \mathbf{d}_{1:T_{obs}}\right)$$

$$\hat{L}_{T_{obs}+1:T_{pred}} = \mathcal{F}_{local}\left(\mathbf{l}_{1:T_{obs}}\right)$$

$$\hat{P}_{T_{obs}+1:T_{pred}} = \hat{G}_{T_{obs}+1:T_{pred}} + \hat{L}_{T_{obs}+1:T_{pred}}$$

여기서:
- $\mathbf{e}\_{1:T_{obs}}$: 에고모션 신호
- $\mathbf{d}\_{1:T_{obs}}$: 깊이(depth) 신호
- $\hat{G}, \hat{L}$: 예측된 글로벌/로컬 스트림

---

### 🔵 2-3. 모델 구조 요약

네트워크 구조는 크게 **(1) 감지된 인간 포즈를 완성하고 글로벌·로컬 스트림을 분리하는 네트워크**, **(2) 로컬 스트림 예측 아키텍처**, **(3) 글로벌 스트림 예측 아키텍처**의 세 부분으로 구성됩니다.

```
[입력: 노이즈 포함 관측 포즈 시퀀스]
         ↓
[포즈 완성 모듈 (Pose Completion Module)]
 - 누락 관절 채우기
 - 노이즈 억제
         ↓
[분해 모듈 (Decomposition Module)]
 ┌──────────────┬──────────────────┐
 ↓              ↓
[글로벌 스트림]  [로컬 스트림]
 (궤적 예측)     (포즈 키포인트 예측)
 + egomotion     Quasi-RNN 기반
 + depth          Encoder-RNN-Decoder
         ↓
[합산 (Merge)]
 최종 로코모션 예측
```

깊이 변화(depth change), 전반적인 포즈 크기 변화 및 신체 각 관절의 움직임을 포착하여 이를 결합(combine)함으로써 보행자의 미래 로코모션을 예측합니다.

---

### 🟡 2-4. 성능 향상

이 방법은 에고센트릭 뷰에서의 인간 로코모션 예측에서 **최첨단(state-of-the-art) 결과**를 달성합니다.

실험을 통해 제안된 방법이 기존 여러 연구들보다 더 나은 결과를 달성하며, 보행자 로코모션 예측에서 인간 동역학 분리(disentangling)의 가설을 검증합니다.

> ⚠️ **정량적 수치(FDE, ADE 등)**: 논문 PDF 원문 접근이 제한되어 정확한 수치를 제시하기 어렵습니다. 정확한 실험 수치는 [arXiv PDF](https://arxiv.org/pdf/1911.01138)에서 직접 확인하시길 권장합니다.

---

### 🔴 2-5. 한계점 (Limitations)

논문 및 관련 출처에서 확인된 한계는 다음과 같습니다:

1. **노이즈 주석 의존성**: 에고센트릭 비디오 데이터셋의 포즈, 깊이, 에고모션에 대한 조밀한 주석의 희소성 문제로 인해, 자동 생성된 노이즈 주석에 의존합니다. 이는 학습 데이터의 품질 상한을 제한할 수 있습니다.

2. **에고센트릭 뷰 특화**: 차량 탑재 카메라라는 특수한 관점에 특화되어 있어, 다른 관점(예: 고정 카메라, 드론 뷰 등)으로의 직접적인 일반화가 어려울 수 있습니다.

3. **단일 보행자 예측**: 사회적 상호작용(social interaction)을 명시적으로 모델링하지 않아, 군중 시나리오에서의 적용에 제약이 있습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문의 일반화 성능과 관련하여 다음의 핵심 요소들이 있습니다:

### 🌐 3-1. 노이즈 지도학습을 통한 일반화

에고센트릭 뷰에서의 보행자 포즈 데이터셋의 희소성 문제를 극복하기 위해, 오프-더-셸프(off-the-shelf) 모델들을 이용해 노이즈가 포함된 ground-truth 데이터를 생성하여 모델 학습에 활용합니다. 이 접근법은 실제 환경에서 수동 레이블링 없이도 모델을 다양한 환경에 적용할 수 있어 **일반화 가능성이 높습니다.**

### 🌐 3-2. Disentanglement을 통한 일반화

전체 보행자 모션을 포즈 완성 및 분해 모듈을 활용하여 **더 쉽게 학습할 수 있는 하위 부분(subparts)**으로 분리하는 방법을 제시합니다. 이렇게 글로벌·로컬 움직임을 분리하면 각 컴포넌트가 더 단순한 패턴을 학습하게 되어, **새로운 환경(다른 도시, 다른 날씨, 다른 보행자 유형)에서도 각 모듈이 독립적으로 견고하게 동작**할 가능성이 높습니다.

### 🌐 3-3. 도메인 특화 신호 활용의 이중 효과

Quasi-RNN 백본 기반으로, 에고모션 및 깊이와 같은 저수준 시각 도메인 특화 신호를 글로벌 궤적 예측에 활용합니다. 이러한 물리적 신호들은 특정 데이터셋에 과적합(overfitting)되지 않는 **도메인 불변 특성(domain-invariant features)**을 학습하는 데 도움이 될 수 있습니다. 그러나 반대로, 특정 센서(예: 자율주행 차량 카메라)에만 의존한다는 점은 다른 도메인으로의 전이(transfer) 시 bottleneck이 될 수 있습니다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 📌 4-1. 연구에 미치는 영향

#### (1) 통합 예측 프레임워크의 선도
이 논문은 기존에 분리되어 연구되던 포즈 예측과 궤적 예측 문제를 단일 프레임워크에서 통합하는 새로운 방향을 제시합니다. 이는 이후 연구들이 로코모션 예측을 **하나의 통합된 문제**로 접근하는 방향에 영향을 주었습니다.

#### (2) 노이즈 내성(Noise-Robust) 학습의 중요성 부각
자동 생성된 주석(pseudo-label)을 이용한 학습 전략은, 이후 **반지도학습(semi-supervised)**, **약지도학습(weakly-supervised)**, **자기지도학습(self-supervised)** 기반 예측 연구들의 실용적인 기반이 되었습니다. 예를 들어, RealTraj(2024)는 대규모 합성 데이터에서의 자기지도 사전 학습(self-supervised pretraining)과 제한된 실세계 데이터에서의 약지도 미세조정(weakly-supervised fine-tuning)을 결합하는 프레임워크를 제안하여 실세계 적용성을 향상시켰습니다.

#### (3) 자율주행 분야에서의 실용적 응용
보행자의 미래 동역학을 근미래에 예측하는 능력은 자율주행 차량의 다음 즉각적인 행동을 위한 의사결정을 지원하며, 보행자 의도 추론, 경로 계획, 반응적 제어 등 다운스트림 태스크에 유용합니다.

#### (4) BiPOCO 등 후속 연구에의 영향
BiPOCO(2022)와 같은 후속 연구는 포즈 제약(pose constraints)을 갖는 양방향 궤적 예측기를 통해 보행자 이상 행동 감지로 응용을 확장하며, 예측 기반 방법의 가능성을 보여줍니다.

---

### 📌 4-2. 향후 연구 시 고려할 점

| 고려 요소 | 설명 |
|---|---|
| **사회적 상호작용 통합** | 군중 내 다중 보행자 간 상호작용 모델링 필요 |
| **불확실성 정량화** | 미래 예측의 다중 모달성(multi-modality)과 불확실성 명시 |
| **Transformer 아키텍처 도입** | Quasi-RNN 대비 장거리 의존성 포착에 유리한 Attention 기반 모델 검토 |
| **도메인 일반화** | 특정 센서(차량 카메라)에 의존하지 않는 일반화 전략 필요 |
| **Diffusion 기반 예측** | 최근 Diffusion 모델 기반 궤적 예측의 성능 검증 필요 |
| **노이즈 주석 품질 향상** | 더 정교한 pseudo-label 생성 및 정제 전략 연구 필요 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 방법 | 주요 특징 | 비교 포인트 |
|---|---|---|---|
| **본 논문** (WACV 2020) | Disentanglement (Global+Local) + Quasi-RNN | 포즈+궤적 통합, 노이즈 지도학습 | 통합 프레임워크의 선구자 |
| **AgentFormer** (ICCV 2021) | Agent-aware Transformer | 시공간 다중 에이전트 예측 | Social interaction 명시 모델링 |
| **BiPOCO** (ICML 2022 Workshop) | Bidirectional + Pose Constraints | 이상행동 감지에 포즈 예측 응용 | 포즈-궤적 결합의 응용 확장 |
| **EigenTrajectory** (ICCV 2023) | Low-rank trajectory descriptors | 다중 모달 예측의 효율화 | 다양성(diversity) 중심 예측 |
| **RealTraj** (2024) | Self-supervised pretraining + Weakly-supervised fine-tuning | 실세계 적용성, 노이즈 내성 | 본 논문의 노이즈 전략 고도화 |

최근 조건부 확산(Diffusion) 모델을 궤적 예측에 활용하는 연구들이 주목할 만한 성과를 보이고 있으나, 정확한 이력 데이터에 대한 의존성 때문에 노이즈 및 데이터 불완전성에 취약하다는 한계가 있습니다.

---

## 📚 참고 자료 / 출처

1. **arXiv 원문**: Mangalam et al., "Disentangling Human Dynamics for Pedestrian Locomotion Forecasting with Noisy Supervision," arXiv:1911.01138, 2019. [https://arxiv.org/abs/1911.01138](https://arxiv.org/abs/1911.01138)
2. **IEEE Xplore (WACV 2020)**: [https://ieeexplore.ieee.org/document/9093350/](https://ieeexplore.ieee.org/document/9093350/)
3. **저자 프로젝트 페이지**: Karttikeya Mangalam. [https://karttikeya.github.io/publication/plf/](https://karttikeya.github.io/publication/plf/)
4. **Stanford TechFinder**: [https://techfinder.stanford.edu/technology/disentangling-human-dynamics-pedestrian-locomotion-forecasting-noisy-supervision](https://techfinder.stanford.edu/technology/disentangling-human-dynamics-pedestrian-locomotion-forecasting-noisy-supervision)
5. **DeepAI**: [https://deepai.org/publication/disentangling-human-dynamics-for-pedestrian-locomotion-forecasting-with-noisy-supervision](https://deepai.org/publication/disentangling-human-dynamics-for-pedestrian-locomotion-forecasting-with-noisy-supervision)
6. **Google Patents**: US20210097266A1. [https://patents.google.com/patent/US20210097266A1](https://patents.google.com/patent/US20210097266A1)
7. **USPTO 특허 PDF**: US11074438. [https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11074438](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11074438)
8. **RealTraj (2024)**: arXiv:2411.17376. [https://arxiv.org/pdf/2411.17376](https://arxiv.org/pdf/2411.17376)
9. **BiPOCO (2022)**: arXiv:2207.02281. [https://arxiv.org/pdf/2207.02281](https://arxiv.org/pdf/2207.02281)
10. **Joint Pedestrian Trajectory Prediction through Posterior Sampling (2024)**: arXiv:2404.00237. [https://arxiv.org/html/2404.00237](https://arxiv.org/html/2404.00237)

---

> ⚠️ **정확도 관련 고지**: 본 논문의 정확한 수식(표기), 정량적 실험 수치(ADE/FDE 등), 세부 구현 사항은 arXiv PDF 원문을 직접 참조하시기를 강력히 권장합니다. 공개 출처에서 확인이 불가능한 내용은 의도적으로 포함하지 않았습니다.
