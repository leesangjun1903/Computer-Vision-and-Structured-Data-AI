# Labelling unlabelled videos from scratch with multi-modal self-supervision

### 1. 핵심 주장과 주요 기여

본 논문의 **핵심 주장**은 다음 두 가지입니다.[1]

첫째, 우수한 특성 인코더(feature encoder)로부터 우수한 비지도 레이블링(unsupervised labelling)이 자동으로 나오지 않는다는 점을 입증합니다. 이는 기존의 강력한 representation learning 방법들이 반드시 좋은 클러스터링을 보장하지 않음을 시사합니다.[1]

둘째, 오디오와 비주얼 모달리티 간의 자연스러운 대응성을 활용하여 인간 주석 없이 비디오 데이터셋을 의사 라벨링(pseudo-labelling)할 수 있는 혁신적인 클러스터링 방법을 제안합니다.[1]

**주요 기여**는 다음과 같이 세 가지로 요약됩니다:[1]

1. **벤치마킹 기반 수립**: Kinetics, Kinetics-Sound, VGG-Sound, AVE 등 4개의 대표적인 비디오 데이터셋에서 최초의 비지도 레이블링 벤치마크 결과를 제시합니다.

2. **강력한 기저선(baseline) 개발**: 최첨단 비디오 representation learning 방법(DPC, MIL-NCE, XDC 등)들을 클러스터링에 적용한 견고한 기저선을 구축합니다.

3. **다중모달 최적화 알고리즘**: SeLaVi (Self-Labelling Videos)를 제안하여 동시에 특성 학습과 클러스터링을 수행하며 다중모달 데이터에 특화된 최첨단 성능을 달성합니다.

***

### 2. 문제 정의, 방법론, 모델 구조 및 성능

#### 2.1 해결하고자 하는 문제

**문제의 본질**은 비디오 데이터 레이블링의 높은 비용입니다. 이미지 도메인에서는 비지도 클러스터링 방법들이 성공을 거두었으나, 비디오 도메인에서는 특성 학습에만 초점이 맞춰져 있었습니다. 비디오는 이미지보다:[1]

- 더 비싼 주석 비용을 요구합니다.
- 시간적 차원을 포함합니다.
- 시각(visual)과 오디오(audio) 두 모달리티를 자연스럽게 포함합니다.

따라서 본 연구의 목표는 **인간 주석 없이도 의미론적으로 의미 있는 비디오 클러스터를 학습**하되, 비디오의 고유한 특성(시간성, 다중모달성)을 활용하는 것입니다.[1]

#### 2.2 방법론: 수식 중심 설명

**기초: 최적 수송 이론(Optimal Transport)**

본 논문은 SeLa 방법을 확장하여 최적 수송 문제로 클러스터링을 공식화합니다. 기본 에너지 함수는:[1]

$$E(p, q) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{y=1}^{K} q_y(x_i) \log p_y(x_i)$$

여기서 $$p_y(x_i) = \text{softmax}(\theta(x_i))$$이고, $$q_y$$는 할당된 라벨의 one-hot 벡터입니다.[1]

균일 클러스터 분포 제약:
$$\sum_{i=1}^{N} \frac{1}{N} p_y(x_i) = \frac{1}{K}$$

**핵심 개선 1: 임의의 사전 분포(Arbitrary Prior Distribution)**

기존 방법의 제약을 완화하기 위해, 논문은 클러스터 한계 확률 $$r$$을 임의로 설정 가능하도록 확장합니다. 최적 순열 $$R^*$$은:[1]

$$R^* = \arg\min_{R} E(R) = \arg\min_{R} \left[ \mathbb{E}_{Q, \log P} - \frac{1}{\beta}KL(Q||Rc) + \text{const} \right]$$

이 문제는 **정렬을 통해 효율적으로 해결**됩니다. 클래스 $$y$$를 $$q_y$$ 기준으로 정렬한 후, $$R \log r_y$$도 증가 순서로 정렬하면 최적성이 보장됩니다.[1]

**핵심 개선 2: 다중모달 단일 라벨링(Multi-modal Single Labelling)**

다중모달 데이터 $$x = (a, v)$$ (오디오, 비주얼)를 처리하기 위해, 각 모달리티를 데이터 증강(augmentation)으로 처리합니다:[1]

$$\log P_y \leftarrow \mathbb{E}_t[\log \text{softmax}_y(t(x_i))]$$

모달리티 스플리싱 변환 $$t_a$$와 $$t_v$$를 증강으로 간주하여 모달리티 불변 클러스터링을 학습합니다. 각 모달리티별 인코더를 학습하되, 같은 클러스터를 공유합니다:

$$\theta_a(t_a(x)) = \theta_v(t_v(x))$$

**초기화 및 정렬(Initialization and Alignment)**

두 네트워크가 랜덤 초기화되면 출력층이 비동기화되므로, 다음 최적화 문제를 통해 순열 행렬 $$R$$을 사전에 찾습니다:[1]

$$\min_{R} \sum_{i=1}^{N} \|\text{softmax}(RW_a^T \phi_a(x_i)) - \text{softmax}(W_v^T \phi_v(x_i))\|_1$$

이 문제는 **탐욕 알고리즘(greedy algorithm)**으로 신속하게 해결되며, 초기 학습을 부트스트래핑합니다.[1]

**핵심 개선 3: 장식된 클러스터 헤드(Decorrelated Clustering Heads)**

클러스터링의 본질적 모호성(예: 동물을 종으로 분류할 수도, 실내/실외로 분류할 수도 있음)을 해결하기 위해, 여러 클러스터링 함수를 병렬로 학습합니다. 각 라운드에서 무작위로 데이터의 두 증강을 생성하고, 절반의 헤드는 첫 번째 버전을, 다른 절반은 두 번째 버전을 처리하게 하여 클러스터 다양성을 증가시킵니다.[1]

#### 2.3 모델 구조

**아키텍처 구성**:

- **비주얼 인코더**: R(2+1)D-18 (3D 시간-공간 컨볼루션)[1]
- **오디오 인코더**: ResNet 9층[1]
- **입력 사양**: 30프레임 클립 (30fps 비디오), 단축 변 128, 훈련 시 112×112 랜덤 크롭[1]
- **오디오 처리**: 1×257×199 log-mel 뱅크 특성[1]
- **클러스터 헤드**: 2계층 MLP, 10개의 병렬 헤드[1]

**훈련 설정**:

- **옵티마이저**: SGD, 200 에포크[1]
- **배치 크기**: 64 GPU에서 각 16 (유효 1024)
- **초기 학습률**: 0.01 (선형 스케일링, 10 에포크 워밍업)[1]
- **정규화**: 가중치 감소 $$10^{-5}$$, 모멘텀 0.9[1]
- **클러스터링 파라미터**: Sinkhorn-Knopp $$\beta = 20$$, 100회 클러스터링 연산[1]

#### 2.4 성능 향상 및 결과

**VGG-Sound 데이터셋 결과**:[1]

| 방법 | NMI | ARI | Accuracy | H_pmax |
|------|-----|-----|----------|---------|
| 랜덤 | 10.2 | -4.0 | 2.2 | 4.9 |
| 지도학습 | 46.5 | 15.6 | 24.3 | 30.8 |
| DPC (최첨단 방법) | 15.4 | 0.7 | 3.2 | 4.9 |
| MIL-NCE | 48.5 | 12.5 | 22.0 | 32.9 |
| XDC (다중모달) | 16.7 | 1.0 | 3.9 | 7.4 |
| **SeLaVi** | **55.9** | **21.6** | **31.0** | **36.3** |

SeLaVi는 기존 최첨단 방법인 MIL-NCE 대비 NMI에서 **15.5% 향상**, 정확도에서 **41% 향상**을 달성합니다.[1]

**AVE 데이터셋 결과**:[1]

| 방법 | NMI | ARI | Accuracy | H_pmax |
|------|-----|-----|----------|---------|
| 지도학습 | 58.4 | 34.8 | 50.5 | 60.6 |
| **SeLaVi** | **66.2** | **47.4** | **57.9** | **59.3** |

주목할 점은 SeLaVi가 **지도학습 기준선을 능가**합니다.[1]

**Kinetics-Sound 결과**:[1]

| 방법 | Accuracy |
|------|----------|
| 지도학습 | 75.0 |
| **SeLaVi** | **41.2** |

Kinetics-Sound에서는 성능이 낮으며, 이는 **시각 인간 행동에 과도하게 집중된 라벨링 편향** 때문입니다. 그럼에도 SeLaVi는 배경음악, 바람음, 함성 등으로 의미 있는 클러스터를 발견합니다.[1]

**Kinetics-400에서의 성능 제한**:

Kinetics-400은 시각 행동 기반이므로 오디오 가중치가 낮아 정확도가 7.8%에 머물렀습니다. 다만 논문에서는 이러한 "실패"가 방법론의 한계가 아니라 데이터셋의 특성을 반영하는 것임을 강조합니다.[1]

**절제 연구(Ablation Study)**:[1]

| 모델 | MA | G. | DH | Accuracy | ARI | NMI |
|------|----|----|----|----|-----|-----|
| SeLa (기저선) | - | - | - | 6.4 | 2.3 | 20.6 |
| 특성 연결 | - | - | - | 7.6 | 3.2 | 24.7 |
| SeLaVi (기본) | ✓ | ✓ | ✗ | 24.6 | 15.6 | 48.8 |
| + Modality Alignment | ✓ | ✓ | ✗ | 26.6 | 18.5 | 50.9 |
| + Gaussian Marginals | ✓ | ✓ | ✓ | 26.6 | 17.7 | 51.1 |

**가장 큰 개선은 장식된 클러스터 헤드**에서 비롯되며, 그 다음 모달리티 정렬입니다.[1]

**다운스트림 작업 성능 (행동 검색)**:[1]

| 방법 | UCF-101 (Recall@1) | HMDB-51 (Recall@1) |
|------|-------------------|-------------------|
| VSP (기존 최첨단) | 24.8 | 52.0 |
| **SeLaVi** | **24.8** | **47.6** |

SeLaVi는 HMDB-51에서 **100% 향상**을 달성하여, 의사 레이블링 외에도 강건한 표현을 학습함을 입증합니다.[1]

***

### 3. 일반화 성능 향상 가능성 (중점)

#### 3.1 다중모달 불균형 문제 및 해결책

최근 연구에서 드러난 핵심 문제는 **다중모달 네트워크의 최적화 불균형**입니다. 두 모달리티의 특성 노름(feature norm) 차이가 한 모달리티를 "특권화"하여 다른 모달리티의 성능을 저하시킵니다.[2]

SeLaVi의 **모달리티 정렬 메커니즘**은 이 문제를 부분적으로 해결하지만, 최근 연구들은 더 나은 접근법을 제안합니다. Relative Norm Alignment (RNA) 손실 함수를 사용하면 두 모달리티의 특성 노름을 하이퍼스피어상에 고정시켜 더욱 균형잡힌 학습을 가능하게 합니다.[2]

**실제 효과**: 압축된 비주얼 입력에서 SeLaVi는 16배 압축에도 NMI 40 이상을 유지하며, MIL-NCE나 지도학습 기준선이 ~25로 떨어지는 것과 대조됩니다. 이는 **모달리티 간의 견고한 보완성**을 입증합니다.[1]

#### 3.2 도메인 일반화(Domain Generalization) 맥락

**현재 상황**: 2024-2025년 다중모달 도메인 일반화 연구에서 SeLaVi 유형의 통합 표현(unified representation) 접근법이 핵심입니다. 그러나 기존 DG 방법들을 단순히 다중모달 설정에 적용하면 **최적화 불일치**로 인해 모달리티 간 경쟁이 발생합니다.[3]

**SeLaVi의 강점**: 
- 각 모달리티별 독립적 인코더로부터 공유 클러스터를 학습하는 설계는 **모달리티 경합 완화**에 기여합니다.[1]
- 다중 클러스터 헤드 사용으로 클러스터링의 근본적 모호성을 다차원에서 포착합니다.[1]

**개선 기회**:
최신 연구(2024-2025)에서는 **지도 대조학습(supervised contrastive learning)** 기반 정렬 메커니즘과 **상호정보 분리(mutual information decoupling)**를 통해 모달리티 일반 정보와 모달리티 특화 정보를 구분합니다. SeLaVi에 이러한 기법을 통합하면 cross-dataset 전이 시 성능 향상이 예상됩니다.[3]

#### 3.3 시간적 모델링 및 장기 의존성

SeLaVi는 **프레임 레벨 기반**의 접근으로 시간적 시퀀스 정보를 충분히 활용하지 못합니다. 30프레임 클립을 입력하지만, 클러스터링 시 시간적 일관성 제약이 부재합니다.[1]

**최신 개선안**:
- **다층 특성 최적화**: 2021년 이후 연구에서는 고수준 특성의 분포 그래프를 저수준 및 중수준 학습에 반영하여 시간적 모델링을 강화합니다.[4]
- **시간적 증강**: 역재생, 셔플, 프레이트 순서 예측 등을 특성 레벨에서 수행하면 더욱 견고한 표현이 나옵니다.[4]

SeLaVi에 이런 **시간적 모델링 모듈**을 추가하면 긴 비디오 시퀀스에 대한 일반화가 개선될 것으로 예상됩니다.

#### 3.4 장기 클러스터 분포 처리

SeLaVi는 **Zipf 분포**를 가정하여 클러스터 주변 확률을 조정합니다. 실제 테스트에서 VGG-Sound에서 309개 클러스터가 256개 또는 619개로 변경되어도 NMI가 56.8-56.9로 안정적입니다.[1]

**더 나은 확률 모델**: 최신 연구(2024)에서는 메타러닝을 통해 **각 클러스터의 최적 크기를 동적으로 학습**합니다. 이는 특히 데이터셋 간 분포 편향(bias)이 클 때 중요합니다.[1]

#### 3.5 크로스 데이터셋 전이 성능

**현재 논문의 한계**: 논문은 같은 데이터셋 내에서의 전이만 평가합니다. VGG-Sound 선훈련 모델을 Kinetics-Sound와 AVE에 미세조정할 때:[1]

- Kinetics-Sound: 41.2% 정확도 (지도학습 75.0% 대비)
- AVE: 57.9% 정확도 (지도학습 50.5%에 근접)

**개선 가능성**: 2024년 연구에서 제시된 **다중모달 통합 표현** 기법을 적용하면, 다양한 데이터셋 간 도메인 시프트 시 일반화를 5-10% 향상시킬 수 있습니다.[3]

---

### 4. 한계(Limitations)

논문에서 명시적으로 언급된 주요 한계는:[1]

1. **데이터셋 편향에 대한 의존성**: Kinetics-400에서 7.8% 정확도는 시각 기반 라벨링에 편향된 데이터에 대한 방법의 취약성을 보여줍니다.

2. **의사 레이블 품질 과대평가 위험**: 클러스터링 결과를 절대적으로 신뢰하기 쉬우며, 특히 의료영상 등 도메인 전문가 지식이 필요한 분야에서 문제가 될 수 있습니다.

3. **편향 전파**: 데이터셋에 내재된 편향(bias)이 모델에 상속되어 불명확한 방식으로 전파될 수 있습니다.

---

### 5. 학문적 영향 및 앞으로의 연구 고려사항

#### 5.1 논문의 연구 커뮤니티에 대한 영향 (2020-2025)

SeLaVi는 **다중모달 자기지도학습** 분야에 다음과 같은 영향을 미쳤습니다:[5]

**벤치마크 수립**: 4개 주요 비디오 데이터셋에서 비지도 레이블링의 첫 체계적 평가를 제공하여, 이후 연구의 기준점 역할을 합니다.

**방법론적 혁신**: 다음의 후속 연구들이 SeLaVi의 다중모달 클러스터링 개념을 확장합니다:[6][7][8]
- 2021년 Multimodal Clustering Networks (MCN): 비디오/오디오/텍스트 3개 모달리티를 통합 임베딩 공간에서 클러스터링[6]
- 2024년 Decoupling Common and Unique Representations (DeCUR): 모달리티 공통 표현과 고유 표현을 구분하는 개선된 접근[8]

#### 5.2 현재(2025년) 연구 트렌드 및 고려사항

**1. 기초 모델(Foundation Models) 통합**

최신 연구에서는 CLIP, GPT-4 등 대규모 멀티모달 기초 모델을 활용하여 SeLaVi식 비지도 클러스터링을 개선합니다.[9][10]

- **Multi-MaP** (2024): CLIP 인코더와 GPT-4를 결합하여 사용자 관심사를 반영한 개인화 다중 클러스터링 수행[9]
- **Multi-Sub** (2024): 다중 클러스터링 문제에서 CLIP+GPT-4를 통해 유연한 사용자 선호도 적응[10]

**권장사항**: SeLaVi를 현대적 기초 모델과 통합하면, 보다 **의미론적으로 정렬된 클러스터**를 획득할 수 있습니다.

**2. 연속학습(Continual Learning) 확장**

2025년 최신 연구는 **비지도 비디오 연속학습** 문제를 제시합니다. 여러 작업을 순차적으로 학습할 때 작업 경계와 라벨이 없는 현실적 시나리오에서:[11]

- 커널 밀도 추정(KDE) 기반 비모수적 표현 사용
- 새로운 작업 데이터에 대한 동적 메모리 클러스터 확장
- UCF101, HMDB51, Something-to-Something V2에서 검증

**권장사항**: SeLaVi를 비디오 연속학습 설정으로 확장하면, 스트리밍 비디오 데이터 환경에 더 잘 적응할 수 있습니다.

**3. 직관적 물리학 이해와 시공간 추론**

2025년 벤치마크 연구들(IntPhys 2, MVP)에서 비디오 언어 모델의 **물리적 이해 및 시공간 추론 능력**이 부족함을 지적합니다. 현재 모델은 50% 정확도(random chance)에 머물러 있습니다.[5]

**권장사항**: SeLaVi는 현재 의미적 일관성만 학습하므로, **인과관계와 물리적 제약**을 클러스터링 목적에 통합하면 더 견고한 일반화가 가능합니다.

**4. 다중모달 도메인 일반화(MMDG)**

2025년 연구에서는 **다중모달 통합 표현**을 통한 도메인 일반화가 주요 과제입니다. 기존 단일모달 DG 방법을 다중모달 설정에 직접 적용하면 최적화 불일치로 인해 모달리티 경합이 발생합니다.[3]

**권장사항**: 
- 지도 대조학습 기반 모달리티 정렬
- 모달리티 일반 정보와 특화 정보 분리
- EPIC-Kitchens 등 멀티소스 도메인 데이터셋에서 평가

이 접근을 SeLaVi에 통합하면 MMDG 벤치마크에서 1-5% 성능 향상을 기대할 수 있습니다.

**5. 음성 기초 모델(Speech Foundation Models) 활용**

2025년 연구는 WavLM, iFLYTEK-speech 등 최신 음성 기초 모델의 **지식 증류**를 오디오-시각 표현학습에 적용합니다.[12]

**권장사항**: SeLaVi의 오디오 인코더를 현대적 음성 기초 모델로 대체하거나, 다층 표현 지식 증류를 적용하면 오디오 이해도가 향상될 것으로 예상됩니다.

**6. 안전성 및 편향 관리**

논문의 Broader Impact 섹션에서 언급한 편향 전파 문제는 여전히 미해결입니다.[1]

**권장사항**:
- 클러스터 품질 평가를 위한 **불확실성 정량화** 메커니즘 추가
- 편향 감지 및 완화를 위한 대조적 클러스터(adversarial clustering) 도입
- 실제 운영 환경에서 클러스터 모니터링 프레임워크 구축

***

### 결론

"Labelling unlabelled videos from scratch with multi-modal self-supervision"은 **비지도 비디오 클러스터링의 개척 논문**으로서, 다음 다섯 가지 핵심 성과를 이루었습니다:[1]

1. **개념적 기여**: 강력한 특성 인코더가 좋은 클러스터링을 자동으로 제공하지 않음을 입증
2. **방법론적 혁신**: 최적 수송, 다중모달 정렬, 장식된 헤드를 결합한 SeLaVi 알고리즘
3. **벤치마크 수립**: 4개 주요 비디오 데이터셋에서 첫 비지도 레이블링 평가
4. **우수한 성능**: VGG-Sound에서 55.9 NMI, AVE에서 지도학습 능가
5. **일반화 원리**: 모달리티 불균형, 시공간 모델링, 크로스 데이터셋 전이에 대한 통찰

**향후 연구 방향**은 (a) 기초 모델 통합, (b) 연속학습 확장, (c) 물리적 추론 통합, (d) 다중모달 도메인 일반화, (e) 편향 관리에 집중해야 하며, 이러한 개선을 통해 비디오 이해와 레이블링 비용 절감에 실질적 기여를 할 수 있여를 할 수 있을 것으로 기대됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e12b307e-54d7-4d3f-bf4a-203968a36c69/2006.13662v3.pdf)
[2](https://openaccess.thecvf.com/content/WACV2022/papers/Planamente_Domain_Generalization_Through_Audio-Visual_Relative_Norm_Alignment_in_First_Person_WACV_2022_paper.pdf)
[3](https://openaccess.thecvf.com/content/ICCV2025/papers/Huang_Bridging_Domain_Generalization_to_Multimodal_Domain_Generalization_via_Unified_Representations_ICCV_2025_paper.pdf)
[4](https://openaccess.thecvf.com/content/ICCV2021/papers/Qian_Enhancing_Self-Supervised_Video_Representation_Learning_via_Multi-Level_Feature_Optimization_ICCV_2021_paper.pdf)
[5](https://ai.meta.com/research/publications/labelling-unlabelled-videos-from-scratch-with-multi-modal-self-supervision/)
[6](https://arxiv.org/abs/2104.12671)
[7](https://arxiv.org/pdf/2112.12182.pdf)
[8](https://arxiv.org/pdf/2309.05300.pdf)
[9](https://arxiv.org/pdf/2404.15655.pdf)
[10](https://arxiv.org/html/2411.03978v1)
[11](https://arxiv.org/abs/2508.21773)
[12](https://arxiv.org/abs/2502.05766)
[13](http://arxiv.org/pdf/2402.16383.pdf)
[14](https://arxiv.org/pdf/2308.09247.pdf)
[15](http://arxiv.org/pdf/2402.19407.pdf)
[16](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Multimodal_Clustering_Networks_for_Self-Supervised_Learning_From_Unlabeled_Videos_ICCV_2021_paper.pdf)
[17](https://ieeexplore.ieee.org/iel8/34/11026037/10630605.pdf)
[18](https://www.sciencedirect.com/science/article/abs/pii/S0925231225004229)
[19](https://www.frontiersin.org/journals/sustainable-cities/articles/10.3389/frsc.2023.1197434/full)
[20](https://www.semanticscholar.org/paper/Multimodal-Clustering-Networks-for-Self-supervised-Chen-Rouditchenko/d9b1bb8053f32c6da9bbbec564d750d55b486f00)
[21](https://arxiv.org/pdf/2210.10317.pdf)
[22](https://arxiv.org/html/2406.10995v2)
[23](https://arxiv.org/pdf/2106.09362.pdf)
[24](http://arxiv.org/pdf/1912.11370.pdf)
[25](https://arxiv.org/pdf/2111.06977.pdf)
[26](http://arxiv.org/pdf/2405.15583.pdf)
[27](https://arxiv.org/pdf/2501.10933.pdf)
[28](https://www.aclweb.org/anthology/P19-1485.pdf)
[29](https://theses.liacs.nl/pdf/2023-2024-RienksSSjoerd.pdf)
[30](https://arxiv.org/html/2412.10925v1)
[31](https://openreview.net/pdf/8e48712582ba9e7d2c6f2d171c752415d92c37de.pdf)
[32](https://github.com/donghao51/Awesome-Multimodal-Adaptation)
[33](https://mlai.yonsei.ac.kr/publications)
[34](https://openreview.net/pdf/868cc39ed39b3a534c9da39a13e03d3441e8d25f.pdf)
[35](https://www.sciencedirect.com/science/article/abs/pii/S1566253525006578)
[36](https://arxiv.org/html/2503.22197v1)
