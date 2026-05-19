
# Segment Any Motion in Videos (SegAnyMo)

> **논문 정보**
> - **제목**: Segment Any Motion in Videos
> - **저자**: Nan Huang, Wenzhao Zheng, Chenfeng Xu, Kurt Keutzer, Shanghang Zhang, Angjoo Kanazawa, Qianqian Wang (UC Berkeley, Peking University)
> - **학회**: CVPR 2025 (pp. 3406–3416)
> - **arXiv**: [2503.22268](https://arxiv.org/abs/2503.22268)
> - **프로젝트 페이지**: https://motion-seg.github.io/
> - **코드**: https://github.com/nnanhuang/SegAnyMo

---

## 1. 핵심 주장 및 주요 기여 요약

### 🎯 핵심 주장

Moving Object Segmentation(MOS)은 시각적 장면의 고수준 이해를 위한 핵심 태스크이며, 기존 연구들은 주로 optical flow에 의존해 모션 단서를 추출했지만, 이는 부분적 모션, 복잡한 변형, 모션 블러, 배경 방해 요소 등으로 인해 불완전한 예측을 초래했다.

이에 대해 본 논문(SegAnyMo)은 세 가지 핵심 아이디어를 결합하여 이 문제를 해결합니다:

**장거리 궤적 모션 단서(long-range trajectory motion cues)** 와 **DINO 기반 의미론적 특징(DINO-based semantic features)** 을 결합하고, **SAM2** 를 활용한 반복적 프롬프팅 전략(iterative prompting strategy)을 통해 픽셀 수준의 마스크 밀화(mask densification)를 수행한다. 또한 모델은 **Spatio-Temporal Trajectory Attention** 과 **Motion-Semantic Decoupled Embedding** 을 채용하여 모션을 우선시하면서 의미론적 지원을 통합한다.

### ✅ 주요 기여 요약

| 기여 항목 | 설명 |
|-----------|------|
| Long-range Trajectory Cues | Optical flow 대신 장거리 포인트 트랙 활용 |
| Spatio-Temporal Trajectory Attention | 궤적 간 공간·시간 관계 포착 |
| Motion-Semantic Decoupled Embedding | 모션과 의미 정보의 효과적 분리·통합 |
| Iterative Prompting + SAM2 | 희소 포인트 마스크 → 픽셀 수준 세그멘테이션 |
| SOTA 성능 | 다양한 벤치마크에서 최첨단 성능 달성 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

본 논문은 태스크를 **Moving Object Segmentation(MOS)** 으로 정의하며, 이는 비디오 내에서 실제로 관측 가능한 움직임을 보이는 객체를 분할하는 것이다. 이는 Video Object Segmentation(정지 상태의 잠재적 이동 객체 포함)이나 motion segmentation(배경 흐름 포함)과는 다르다. 본 태스크는 카메라 모션과 객체 모션의 구분, 변형·폐색·빠른 움직임에도 강건한 추적, 정밀한 마스크 생성을 암묵적으로 요구하기 때문에 매우 도전적이다.

SAM2는 MOS를 네이티브로 처리할 수 없는데, 어떤 객체가 움직이는지 감지하는 메커니즘이 없기 때문이다.

### 2.2 제안 방법 및 파이프라인

모델은 오프더셀프(off-the-shelf) 모델이 생성한 **2D 트랙과 깊이 맵(depth maps)** 을 입력으로 받아, 모션 인코더가 모션 패턴을 포착하고 featured tracks를 생성한다. 그 다음 트랙 디코더가 DINO 특징을 통합하여 모션과 의미 정보를 분리(decouple)하고, 최종적으로 동적 궤적(dynamic trajectories)을 얻는다. 이후 SAM2를 사용해 동일 객체에 속하는 동적 트랙을 그룹화하고, 세밀한 이동 객체 마스크를 생성한다.

#### 🔧 핵심 모듈 1: Spatio-Temporal Trajectory Attention

**Spatio-Temporal Trajectory Attention** 은 입력 트랙의 장기적 특성을 고려하여, 서로 다른 궤적들 사이의 관계를 포착하는 **공간 어텐션(spatial attention)** 과 개별 궤적 내의 시간적 변화를 모니터링하는 **시간 어텐션(temporal attention)** 을 통합한다.

이를 수식으로 표현하면:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- **공간 어텐션**: 서로 다른 궤적 $\{T_1, T_2, \ldots, T_N\}$ 사이의 관계를 포착

$$
\mathbf{A}^{spatial}_{ij} = \text{softmax}\left(\frac{\mathbf{q}_i \cdot \mathbf{k}_j^T}{\sqrt{d_k}}\right)
$$

- **시간 어텐션**: 단일 궤적 $T_i = \{p_i^1, p_i^2, \ldots, p_i^L\}$ 내 시간적 변화를 포착

$$
\mathbf{A}^{temporal}_{t} = \text{softmax}\left(\frac{\mathbf{q}^t \cdot (\mathbf{k}^{1:L})^T}{\sqrt{d_k}}\right)
$$

- **최종 결합**:

$$
\mathbf{F}_{track} = \mathbf{A}^{spatial} \oplus \mathbf{A}^{temporal}
$$

#### 🔧 핵심 모듈 2: Motion-Semantic Decoupled Embedding

주요 목표가 MOS이므로 모션 단서를 강조하면서 의미 정보를 부차적 지원으로 사용한다. 이 두 정보를 효과적으로 균형잡기 위해 두 가지 특수 모듈을 제안하며, 그 중 **Motion-Semantic Decoupled Embedding** 은 모션 패턴을 우선시하고 의미론적 특징을 보조 경로(supplementary pathway)로 처리하는 특수 어텐션 메커니즘을 구현한다.

단순히 모션 단서만으로는 이동 객체를 구분하기 어려운데, 높은 추상화 수준의 궤적에서 객체 모션과 카메라 모션을 구분하는 것이 어렵기 때문이다. 여기에 텍스처, 외관, 의미 정보를 제공하면 어떤 객체가 이동하거나 이동될 가능성이 있는지 이해하는 데 도움이 되어 이 태스크를 단순화할 수 있다.

수식으로는:

$$
\mathbf{F}_{final} = \alpha \cdot \mathbf{F}_{motion} + (1 - \alpha) \cdot \mathbf{F}_{semantic}
$$

여기서 $\alpha$는 모션 우선도를 제어하는 학습 가능한 가중치 파라미터이며, $\mathbf{F}\_{motion}$은 궤적 기반 모션 특징, $\mathbf{F}_{semantic}$은 DINO로부터 추출한 의미 특징이다.

#### 🔧 핵심 모듈 3: Sparse-to-Dense Iterative Prompting (SAM2 활용)

핵심 통찰은 장거리 트랙이 모션 패턴을 포착할 뿐 아니라 프롬프터블 시각 세그멘테이션에 필수적인 장거리 프롬프트를 제공한다는 것이다. 따라서 장거리 포인트 트랙을 모션 단서로 활용하며, 공간-시간 어텐션을 적용해 컨텍스트 인식 특징을 추출한다. 동적 트랙이 식별되면, SAM2와 결합한 **Iterative Prompting** 을 통해 희소한 포인트 수준 마스크를 픽셀 수준 세그멘테이션으로 변환하는 **sparse-to-dense mask densification** 전략을 적용한다.

프로세스를 수식화하면:

$$
\mathcal{M}^{pixel} = \text{SAM2}\left(\mathcal{I}, \mathcal{P}_{dynamic}\right)
$$

$$
\mathcal{P}_{dynamic} = \text{Group}\left(\{T_i \mid y_i = 1\}\right)
$$

여기서 $y_i \in \{0, 1\}$ 은 궤적의 동적(dynamic) 여부를 나타내는 이진 레이블, $\mathcal{I}$ 는 입력 프레임이다.

### 2.3 모델 전체 구조

```
[입력]
  2D Long-range Tracks (off-the-shelf tracker)
  Depth Maps (off-the-shelf depth estimator)
       │
       ▼
[Motion Encoder]
  - 모션 패턴 인코딩
  - Featured Tracks 생성
       │
       ▼
[Tracks Decoder]
  ┌──────────────────────────────────────┐
  │ Spatio-Temporal Trajectory Attention │
  │  - Spatial Attention (궤적 간 관계)  │
  │  - Temporal Attention (시간 변화)    │
  └──────────────────────────────────────┘
       │
  ┌──────────────────────────────────────┐
  │ Motion-Semantic Decoupled Embedding  │
  │  - DINO Feature 통합                 │
  │  - 모션/의미 정보 분리               │
  └──────────────────────────────────────┘
       │
       ▼
[Dynamic Track Classification]
  각 트랙에 대한 동적/정적 레이블 예측
       │
       ▼
[SAM2 + Iterative Prompting]
  동적 트랙 → 그룹화 → SAM2 프롬프트
  Sparse → Dense Pixel-level Mask
       │
       ▼
[출력]
  Per-Object Fine-grained Moving Object Masks
```

SegAnyMo는 Segment Anything Model(SAM)을 기반으로 하되 비디오 모션 처리를 위해 확장되었으며, 아키텍처는 **모션 패턴 인코딩**, **궤적별 모션 예측**, **모션 디코더 모듈** 이라는 세 가지 주요 구성요소로 이루어진다.

### 2.4 성능 향상

다양한 데이터셋에 대한 광범위한 테스트에서 최첨단(state-of-the-art) 성능을 입증했으며, 특히 도전적인 시나리오와 다중 객체의 세밀한 분할에서 뛰어난 성능을 보인다.

본 방법은 관절 구조(articulated structures), 그림자 반사(shadow reflections), 동적 배경 모션, 급격한 카메라 움직임을 포함한 도전적인 시나리오를 처리하면서 객체 수준의 세밀한 이동 객체 마스크를 생성할 수 있다.

여러 데이터셋에서 다수의 SOTA 방법들과 비교했을 때, 특히 복잡한 모션이나 모션 단서만으로 구분 가능한 유사해 보이는 객체가 포함된 도전적인 케이스에서 일관되게 우수한 성능을 보였다.

Motion-Semantic Decoupled Embedding은 전통적인 어피니티 행렬(affinity matrix) 기반 접근법과 다른 방식으로 궤적에 대한 모션 레이블을 얻으며, 여러 벤치마크에서의 광범위한 결과는 특히 세밀한 이동 객체 분할에서 효과성을 입증한다.

### 2.5 한계점

모델의 성능은 포인트 궤적의 품질에 크게 의존한다. 광학 흐름 추정이 부정확한 경우(예: 저조도 환경이나 매우 빠른 움직임)에는 세그멘테이션 결과가 저하될 수 있다.

두 번째로, 매우 느리게 움직이는 객체에 어려움을 겪는데, 이러한 객체의 모션 패턴이 배경 노이즈나 카메라 움직임과 구분하기 어렵기 때문이다. 이는 많은 모션 세그멘테이션 접근법에서 공통적인 도전 과제다.

또한 폐색된 객체 처리 방법이 충분히 다루어지지 않는데, 이동하는 객체가 다른 객체 뒤를 지날 때 발생하는 시나리오는 궤적 기반 방법에 독특한 도전을 제시한다. 객체가 겹칠 때 궤적이 끊기거나 혼동될 수 있기 때문이다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화 강점

모델은 합성 및 실제 데이터셋으로 학습되었으며, 강력한 일반화 성능을 가진다.

SegAnyMo는 카테고리별 훈련 없이 다양한 모션 유형에 걸쳐 작동한다. 즉, 특정 객체 카테고리에 무관하게(category-agnostic) 모션 자체에 집중하기 때문에 일반화 성능의 기반이 된다.

이 시스템은 비디오 내의 모든 이동 객체를 그것이 무엇인지와 관계없이 식별하고 윤곽을 잡을 수 있다. 전통적 시스템이 특정 유형의 객체(자동차, 사람 등)에 대해 특별히 훈련이 필요했던 것과 달리, SegAnyMo는 객체 유형이 아닌 모션 자체에 집중하는 다른 접근법을 취한다.

### 3.2 일반화 성능 향상의 핵심 메커니즘

#### (1) Long-range Trajectory의 역할

장거리 트랙은 단순히 모션 패턴을 포착하는 것을 넘어, 프롬프터블 시각 세그멘테이션에 필수적인 장거리 프롬프트를 제공한다는 점이 핵심 통찰이다.

장기적으로 이동 객체가 폐색이나 조명 변화와 같은 요소를 경험할 때, 이는 광학 흐름 기반 방법의 추적 성능에 부정적 영향을 미칠 수 있다. 반면 장거리 트랙은 이러한 단기적 방해 요소에 더 강건하다.

#### (2) DINO Feature 통합의 역할

- DINOv2와 같은 사전학습 비전 트랜스포머의 풍부한 의미론적 특징은 도메인 전반에 걸쳐 일반화된 표현을 제공
- 객체 카테고리에 무관한 시각적 특징 추출이 가능하여 새로운 도메인에서도 강건하게 동작

#### (3) SAM2 활용의 역할

- SAM2는 대규모 데이터로 사전학습된 범용 세그멘테이션 모델로, 이를 백본으로 사용함으로써 뛰어난 일반화 성능을 물려받음
- 동적 트랙을 SAM2 프롬프트로 변환하는 전략은 픽셀 수준의 세밀한 마스크를 다양한 도메인에서 생성 가능하게 함

$$
\text{Generalization} \propto f(\underbrace{\text{Long-range Tracks}}_{\text{모션 강건성}}, \underbrace{\text{DINO features}}_{\text{의미론적 일반화}}, \underbrace{\text{SAM2}}_{\text{픽셀 수준 일반화}})
$$

#### (4) 합성+실제 데이터 혼합 학습

모델이 합성 데이터와 실제 데이터 모두에서 학습하는 것은, 도메인 갭을 줄이고 실세계 일반화 성능을 높이는 데 중요한 역할을 한다.

### 3.3 추가 일반화 향상 가능성

| 방향 | 설명 |
|------|------|
| 3D 궤적 활용 | 2D 트랙에 depth 정보를 완전히 통합하면 3D 공간에서의 모션 구분 강화 |
| 도메인 적응(DA) | 의료 영상, 위성 영상 등 특수 도메인에 대한 fine-tuning 전략 |
| 자기지도 학습 | 레이블 없이 대규모 비디오에서 사전학습하여 제로샷 성능 향상 |
| 다모달 융합 | RGB + 이벤트 카메라 + IMU 등 다양한 센서 데이터 결합 |

---

## 4. 최신 관련 연구 비교 분석 (2020년 이후)

### 4.1 연구 계보 분류

#### 📌 Optical Flow 기반 접근법

| 논문 | 방법 | 한계 |
|------|------|------|
| RAFT (ECCV 2020) | 반복적 광학 흐름 추정 | 단기 프레임 간 흐름에 한정 |
| FlowI-SAM / FlowP-SAM (ACCV 2024) | SAM + 광학 흐름 조합 | 다중 상호작용 객체에서 성능 저하 |

FlowI-SAM은 SAM의 이동 객체 분할 능력을 광학 흐름 필드의 뚜렷한 텍스처와 경계를 활용해 활용하지만, 다수의 상호작용 객체에서 발생하는 광학 흐름 장면에서는 흐름이 이들을 분리하기 위한 제한적 정보만을 포함하기 때문에 성능이 저하된다.

#### 📌 Trajectory/Track 기반 접근법

두 프레임의 광학 흐름과 달리, 수백 프레임에 걸친 포인트 궤적은 서로 다른 객체를 분리하는 것을 방해하는 단기적 변동에 덜 취약하다. 긍정적 부작용으로, 결과적인 그룹화는 전체 비디오 샷에 걸쳐 시간적으로 일관성을 갖는데, 이는 대다수 기존 접근법에서 번거로운 후처리를 요구하는 속성이다.

#### 📌 SAM 기반 접근법의 진화

| 연구 | 연도 | 특징 |
|------|------|------|
| Segment Anything (SAM) | 2023 | 정적 이미지 범용 세그멘테이션 |
| SAM2 | 2024 | 비디오 객체 세그멘테이션으로 확장 |
| **SegAnyMo** | **2025** | **SAM2 + 장거리 궤적 + DINO로 MOS 해결** |

#### 📌 관련 동시대(Concurrent) 연구

본 논문과 유사한 문제를 다루는 동시대 연구들로는 **RoMo: Robust Motion Segmentation Improves Structure from Motion** 과 **Learning segmentation from point trajectories** 가 있다.

MATNet(2020)은 객체 외관과 모션 정보를 통합하려 시도했고, Isomer(2023)는 트랜스포머의 장거리 의존성 모델링의 이점을 활용했으며, 2024년에는 RGB, 광학 흐름, 깊이, 색출지도를 입력으로 사용하는 다중 소스 예측기가 제안되었다.

### 4.2 종합 비교표

| 방법 | 연도 | 모션 표현 | 의미 정보 | 세그멘테이션 | 일반화 |
|------|------|-----------|----------|------------|--------|
| Optical Flow 기반 | ~2022 | 단기 광학 흐름 | ❌ | 프레임별 | 낮음 |
| MATNet | 2020 | 광학 흐름 | ✅ | 프레임별 | 중간 |
| FlowP-SAM | 2024 | 광학 흐름 | ❌ | SAM 기반 | 중간 |
| **SegAnyMo** | **2025** | **장거리 궤적** | **✅ DINO** | **SAM2 반복 프롬프팅** | **높음** |

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 연구에 미치는 영향

#### 🔬 패러다임 전환: Optical Flow → Long-range Trajectory

기존 연구들이 광학 흐름에 의존해 왔던 한계를 명확히 지적하고, 장거리 궤적 기반의 새로운 패러다임을 제시함으로써 이후 연구들이 더 긴 시간적 컨텍스트를 활용하도록 방향을 제시한다.

#### 🔬 Foundation Model의 MOS 적용 가능성 개척

SAM2와 같은 Foundation Model이 MOS를 네이티브로 처리하지 못하는 한계를 궤적 기반 프롬프트로 극복하는 방법을 제시함으로써, Foundation Model을 특정 도메인에 적용하는 새로운 방법론적 기반을 마련한다.

#### 🔬 다운스트림 태스크에의 파급력

액션 인식, 자율주행, 4D 재건 등 다양한 응용 분야에서의 이동 객체 분할의 중요성을 고려하면, 본 연구는 이들 분야의 성능 향상에 직접적으로 기여할 수 있다.

선택적 모션 동결(selective motion freezing), 인터랙티브 편집(interactive editing) 등의 응용도 가능하다.

### 5.2 향후 연구 시 고려할 점

#### ⚠️ 1. 궤적 품질 의존성 해결

시스템 성능이 포인트 궤적의 품질에 크게 의존하므로, 저조도나 빠른 모션 환경에서의 궤적 추적 품질 향상이 선행되어야 한다.

→ **연구 방향**: 이벤트 카메라(event camera)나 깊이 센서와 결합한 강건한 궤적 추적기 개발

#### ⚠️ 2. 느린 객체 및 폐색 처리

매우 느리게 움직이는 객체에서의 어려움은 배경 노이즈나 카메라 움직임과의 구분이 어렵기 때문이며, 이는 근본적인 해결이 필요한 문제다.

→ **연구 방향**: 상대적 모션(relative motion) 기반의 정규화, 카메라 ego-motion 추정과의 연동

#### ⚠️ 3. 실시간 처리 가능성

- 현재 파이프라인은 off-the-shelf tracker + depth estimator + motion encoder + SAM2의 복합 구조로, 실시간 처리에 병목이 될 수 있음
→ **연구 방향**: 경량화 모델 설계, 엔드-투-엔드 학습 구조 통합

#### ⚠️ 4. 레이블 효율적 학습 (Label-Efficient Learning)

합성 및 실제 데이터셋 혼합 학습이 일반화에 중요하지만, 고품질 레이블 확보의 비용이 높다.
→ **연구 방향**: 자기지도학습(Self-Supervised Learning), 약지도학습(Weakly-Supervised)을 통한 레이블 효율화

#### ⚠️ 5. 특수 도메인 일반화

- 의료 영상(내시경), 위성 영상, 수중 영상 등 특수 도메인에서의 도메인 적응 전략 필요
- 현재 DINO와 SAM2가 자연 이미지 중심으로 사전학습된 한계 존재

#### ⚠️ 6. 3D 공간 활용 심화

- 현재는 2D 트랙과 깊이 맵을 분리하여 사용하지만, 진정한 3D 포인트 트랙 기반의 통합이 이루어지면 더 정교한 카메라 모션 분리가 가능

$$
\mathbf{T}_{3D} = \pi^{-1}(\mathbf{T}_{2D}, d) \quad \Rightarrow \quad \Delta\mathbf{T}_{obj} = \mathbf{T}_{3D} - \mathbf{T}_{camera}
$$

여기서 $\pi^{-1}$는 역투영(back-projection), $d$는 깊이, $\mathbf{T}_{camera}$는 카메라 ego-motion 성분을 나타낸다.

---

## 📚 참고 자료 및 출처

| # | 제목 | 링크/출처 |
|---|------|----------|
| 1 | **Segment Any Motion in Videos** (주 논문) | arXiv:2503.22268, CVPR 2025, pp.3406–3416 |
| 2 | 프로젝트 페이지 | https://motion-seg.github.io/ |
| 3 | GitHub 코드 (nnanhuang/SegAnyMo) | https://github.com/nnanhuang/SegAnyMo |
| 4 | arXiv 논문 PDF | https://arxiv.org/pdf/2503.22268 |
| 5 | arXiv HTML | https://arxiv.org/html/2503.22268v1 |
| 6 | CVPR 2025 Open Access PDF | https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_Segment_Any_Motion_in_Videos_CVPR_2025_paper.pdf |
| 7 | HuggingFace Paper Page | https://huggingface.co/papers/2503.22268 |
| 8 | AI Models FYI 분석 | https://www.aimodels.fyi/papers/arxiv/segment-any-motion-videos |
| 9 | **Moving Object Segmentation: All You Need Is SAM (and Flow)** (ACCV 2024) | https://arxiv.org/abs/2404.12389 |
| 10 | **Instance-Level Moving Object Segmentation from a Single Image with Events** (2025) | https://arxiv.org/html/2502.12975v1 |
| 11 | RoMo: Robust Motion Segmentation Improves Structure from Motion | Concurrent work (언급됨) |
| 12 | Learning segmentation from point trajectories | Concurrent work (언급됨) |

> ⚠️ **정확도 관련 고지**: 본 논문의 구체적인 수치 성능(예: J&F 점수, IoU 등 구체적 벤치마크 결과)은 현재 검색된 정보에서 확인되지 않아 포함하지 않았습니다. 또한 수식 일부는 논문에 명시된 형태가 아닌 개념적 재구성이 포함되어 있으므로, 정확한 수식은 원문 PDF를 직접 참고하시기를 권장합니다.
