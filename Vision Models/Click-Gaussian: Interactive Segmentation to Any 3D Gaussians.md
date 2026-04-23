
# Click-Gaussian: Interactive Segmentation to Any 3D Gaussians

> **논문 정보**
> - **저자**: Seokhun Choi, Hyeonseop Song, Jaechul Kim, Taehyeong Kim, Hoseok Do (LG Electronics, Seoul National University)
> - **발표**: ECCV 2024
> - **arXiv**: [2407.11793](https://arxiv.org/abs/2407.11793)
> - **프로젝트 페이지**: https://seokhunchoi.github.io/Click-Gaussian/

---

## 1. 핵심 주장 및 주요 기여 요약

3D Gaussian Splatting의 실시간 렌더링 능력 덕분에 3D Gaussian의 인터랙티브 세그멘테이션은 3D 씬의 실시간 조작을 위한 훌륭한 기회를 제공한다. 그러나 기존 방법들은 노이즈가 많은 세그멘테이션 출력을 처리하기 위해 시간이 많이 걸리는 후처리(post-processing)에 의존하고 있다. 또한 3D 씬의 세밀한 조작에 중요한 정밀한 세그멘테이션을 제공하는 데 어려움을 겪는다.

이에 Click-Gaussian이 제안하는 **핵심 기여**는 다음 세 가지로 요약된다:

| 핵심 기여 | 내용 |
|---|---|
| **Two-Level Granularity Feature Fields** | Coarse/Fine 두 단계의 세분화 특징 필드 학습 |
| **Global Feature-guided Learning (GFL)** | 뷰 간 비일관성 문제를 전역적 클러스터링으로 해결 |
| **실시간 인터랙티브 세그멘테이션** | 클릭당 10ms의 추론 속도 달성 |

Click-Gaussian은 2D 세그멘테이션 마스크를 두 레벨의 세분성(granularity)을 갖는 3D 특징 필드로 변환함으로써, 사전 학습된 3D Gaussian에 대해 인터랙티브하고 세밀한 세그멘테이션을 가능하게 하는 신속하고 정밀한 방법이다.

---

## 2. 해결하고자 하는 문제

### 2.1 기존 방법의 문제점

기존 방법들은 노이즈가 많은 세그멘테이션 출력을 처리하기 위한 시간 소모적인 후처리 문제를 안고 있으며, 3D 씬의 세밀한 조작에 중요한 정밀한 세그멘테이션을 제공하는 데 어려움을 겪는다.

3D 씬으로부터 독립적으로 획득된 2D 세그멘테이션으로 인해 일관성 없이 학습된 특징 필드에서 발생하는 도전들을 다루며, 여러 뷰에 걸친 2D 세그멘테이션 결과(3D 세그멘테이션의 핵심 단서)가 충돌할 때 3D 세그멘테이션 정확도가 저하된다.

보다 구체적으로, 문제는 두 가지 차원에서 발생한다:

1. **노이즈가 많은 후처리 의존성**: 이러한 방법들은 씬에서 구분 가능한 특징 필드를 학습하는 데 어려움을 겪어, 명확한 세그멘테이션을 달성하기 위한 광범위한 후처리가 필요하다. 이러한 시간 소모적인 후처리에 대한 의존성은 3DGS의 효율성 이점을 크게 저해한다.

2. **뷰 간 일관성 없는 특징 필드**: 이 과정에서 중요한 장애물은 서로 다른 뷰에서의 2D 마스크 불일관성으로, 이는 일관되고 구분 가능한 의미론적 특징의 훈련을 방해한다.

---

## 3. 제안하는 방법 (수식 포함)

### 3.1 전체 파이프라인 개요

방법의 개요: i) 사전 학습된 3D Gaussian을 두 레벨의 세분성 특징 $\mathbf{f}_i$로 보강한다. ii) 이 특징들은 대조 학습(contrastive learning)을 통해 훈련되며, 2D 렌더링된 특징 맵 $\mathbf{F}$와 그에 대응하는 SAM이 생성한 마스크 $M$을 활용한다. iii) 뷰 간 마스크 신호의 불일관성을 해결하기 위해 Global Feature-guided Learning 방식을 도입한다.

### 3.2 3D Gaussian 표현 및 특징 분할

각 3D Gaussian $g_i$의 파라미터는 다음과 같이 정의된다:

$$g_i = \{p_i, s_i, q_i, o_i, c_i\}$$

여기서 $p_i \in \mathbb{R}^3$는 각 Gaussian의 중심 위치이다. 스케일링 인수 $s_i \in \mathbb{R}^3$와 쿼터니언 $q_i \in \mathbb{R}^4$는 각 Gaussian의 3D 공분산을 표현하는 데 사용된다.

Click-Gaussian은 각 Gaussian에 세그멘테이션용 특징 벡터 $\mathbf{f}_i \in \mathbb{R}^D$를 추가하여 보강된 Gaussian을 생성한다:

$$\tilde{g}_i = g_i \cup \{\mathbf{f}_i\}$$

특징 벡터 $\mathbf{f}_i$는 Granularity Prior에 기반하여 Coarse와 Fine 두 레벨로 분할된다:

$$\mathbf{f}_i = [\mathbf{f}_i^c \| \bar{\mathbf{f}}_i^c]$$

- $\mathbf{f}_i^c \in \mathbb{R}^{D^c}$: **Coarse-level** 특징
- $\bar{\mathbf{f}}_i^c \in \mathbb{R}^{D - D^c}$: **Fine-level** 특징 (coarse 특징과 concatenate하여 사용)

이는 실제 세계에서 두 레벨 간의 내재적 의존성(예: 두 객체 $b \subset B$가 자연적으로 다른 경우)으로부터 동기 부여된 Granularity Prior에 기반하며, Fine-level 특징 학습을 보다 효과적으로 만든다. 실험적 설정에서 $D = 24$로 설정하고 특징을 제외한 Gaussian의 다른 파라미터는 고정(freeze)한다.

### 3.3 2단계 마스크 생성

SAM 마스크 생성을 위해 공식 코드의 자동 마스크 생성 모듈을 활용하며, 레벨을 구분하지 않고 마스크를 추출하여 이미지에서 가장 높은 신뢰도의 세그먼트만 얻는다. 이 세그먼트들은 면적에 따라 두 마스크로 할당되는데, 여러 세그먼트가 하나의 픽셀에 할당될 경우 Coarse-level 마스크는 더 큰 세그먼트의 식별자를 우선시하고, Fine-level 마스크는 더 작은 세그먼트를 우선시한다. 이 방식은 각 레벨에서 픽셀당 단일 마스크 식별자를 할당하여 안정적인 대조 학습을 가능하게 한다.

### 3.4 대조 학습 손실 함수 (Contrastive Learning)

코사인 유사도 기반 대조 학습을 사용하여 두 레벨의 마스크 세트로 구분 가능한 특징을 훈련한다.

2D 렌더링 특징 맵 $\mathbf{F}^l$ ($l \in \{f, c\}$)에서의 대조 손실을 마스크 $M$에 기반하여 설계한다. 동일한 마스크 영역에 속하는 픽셀들은 **positive pair**, 서로 다른 마스크에 속하는 픽셀들은 **negative pair**를 형성한다.

코사인 유사도 기반 대조 손실:

$$\mathcal{L}_{seg} = \mathcal{L}_{seg}^c + \mathcal{L}_{seg}^f$$

$$\mathcal{L}_{seg}^l = -\sum_{(i,j) \in \mathcal{P}^l} \log \frac{\exp(\text{sim}(\mathbf{f}_i^l, \mathbf{f}_j^l) / \tau)}{\sum_{k} \exp(\text{sim}(\mathbf{f}_i^l, \mathbf{f}_k^l) / \tau)}$$

여기서 $\text{sim}(\cdot, \cdot)$은 코사인 유사도, $\tau$는 온도 파라미터, $\mathcal{P}^l$은 레벨 $l$에서의 positive pair 집합을 의미한다.

### 3.5 Global Feature-guided Learning (GFL)

이 문제를 극복하기 위해 Global Feature-guided Learning (GFL)을 제안한다. GFL은 여러 뷰에 걸쳐 노이즈가 많은 2D 세그먼트로부터 전역적 특징 후보들의 클러스터를 구성하며, 3D Gaussian의 특징을 훈련할 때 노이즈를 스무딩한다.

이와 대조적으로, 제안된 GFL 방법은 순차적인 이미지 입력을 가정하지 않고 씬 전체에 걸쳐 전역적으로 집계된 특징 후보를 활용함으로써 뷰 일관성 있는 학습 신호를 보장한다.

GFL의 핵심 아이디어를 수식으로 나타내면:

$$\hat{\mathbf{f}}^l_k = \frac{1}{|S_k^l|} \sum_{i \in S_k^l} \mathbf{F}^l(\mathbf{x}_i)$$

여기서 $S_k^l$은 레벨 $l$에서 전역 클러스터 $k$에 속하는 픽셀 집합, $\hat{\mathbf{f}}^l_k$는 $k$번째 전역 특징 후보(Global Feature Candidate)를 나타낸다. 이 전역 특징 후보들을 기반으로 대조 학습을 수행하여 뷰 간 일관성을 확보한다.

### 3.6 추론: 클릭 기반 세그멘테이션

3D Gaussian 추출 작업을 위해, 사용자가 제공한 포인트 프롬프트에 해당하는 렌더링된 특징과 두 레벨의 전역 특징 후보 간의 코사인 유사도를 사용하여 클릭된 객체를 효율적으로 검색한다.

$$k^* = \arg\max_k \, \text{sim}(\mathbf{F}^l(\mathbf{x}_{click}), \hat{\mathbf{f}}^l_k)$$

---

## 4. 모델 구조

```
[사전 학습된 3D Gaussian 씬]
          ↓
[SAM 자동 마스크 생성]
          ↓
[Two-Level 마스크 분리 (Coarse/Fine by Area)]
          ↓
[특징 필드 학습 with Granularity Prior]
  - 각 Gaussian에 f_i ∈ R^D 특징 부착
  - Coarse feature ‖ Fine feature로 분할
          ↓
[Global Feature-guided Learning (GFL)]
  - 전체 뷰의 2D 세그먼트 클러스터링
  - 전역 특징 후보 생성
  - 뷰 일관성 대조 학습
          ↓
[훈련 완료된 Click-Gaussian]
          ↓
[사용자 클릭 → 코사인 유사도 기반 매칭]
          ↓
[Coarse/Fine 두 레벨 세그멘테이션 결과 (10ms)]
```

SAM의 자동 마스크 생성 모듈을 씬의 모든 훈련 뷰에 활용하고, 생성된 마스크를 세그먼트 면적에 따라 정리하여 각 이미지에 대한 Coarse 및 Fine 레벨 마스크를 도출한다. 이 두 레벨 마스크의 정보는 Granularity Prior를 사용하여 각 Gaussian의 특징 공간을 분할함으로써 3D Gaussian에 통합되어 두 레벨의 세부 사항 표현을 용이하게 한다.

---

## 5. 성능 향상 및 한계

### 5.1 성능 향상

제안된 방법은 클릭당 10ms로 실행되며, 이는 이전 방법들보다 15배에서 130배 빠른 속도이고, 동시에 세그멘테이션 정확도도 크게 향상되었다.

제안된 방법은 Coarse 및 Fine 두 레벨 모두에서 우수한 세그멘테이션 능력을 보여준다. 빨간색과 노란색 박스는 각각 노이즈가 많은 결과와 과소 세그멘테이션 결과를 나타내며, 제안된 방법은 다른 베이스라인들보다 최대 130배 빠른 속도로 더 상세하고 깔끔한 Gaussian 추출을 수행한다.

제안된 방법은 Coarse 및 Fine 두 레벨 모두에서 우수한 세그멘테이션 능력을 보여준다.

구체적인 비교:

| 방법 | 속도 (클릭당) | 특징 |
|---|---|---|
| **Click-Gaussian** | **10 ms** | 후처리 불필요, 두 레벨 세분화 |
| SAGA | ~150 ms | 후처리 필요, scale-gated 특징 |
| Gaussian Grouping | ~1300 ms | 후처리 필요, SAM 기반 |

GFL 기법은 개별 2D 세그멘테이션 마스크에 내재된 모호성의 영향을 완화하여 특징 학습의 견고성과 신뢰성을 향상시킨다. Click-Gaussian의 효과는 복잡한 실제 씬에서의 포괄적인 실험을 통해 입증되었으며, 세그멘테이션 정확도와 계산 효율성 모두를 평가하였다.

또한 시간에 따라 변화하는 4D Gaussian도 세그멘테이션할 수 있다.

### 5.2 한계점

잠재적인 한계점으로는 세그멘테이션 과정을 초기화하기 위한 사용자 클릭에 대한 의존성이 있다. Click-Gaussian 방법이 직관적이고 사용하기 쉽지만, 의미론적 또는 인스턴스 수준 세그멘테이션과 같이 관심 있는 Gaussian을 식별하는 보다 자동화된 방식을 탐색하는 것이 유익할 수 있다.

추가적인 한계:
- **씬별 재학습 필요**: 3D Gaussian 특징 학습이 각 씬에 종속적이므로, 새로운 씬에는 처음부터 다시 학습해야 한다.
- **SAM 의존성**: SAM의 자동 마스크 생성 모듈을 씬의 모든 훈련 뷰에 활용하기 때문에 SAM의 품질 저하가 전체 파이프라인에 영향을 준다.
- **실내/정적 씬에 최적화**: 동적 씬이나 대규모 야외 환경에서의 성능 검증이 제한적이다.

---

## 6. 모델의 일반화 성능 향상 가능성 (중점 분석)

### 6.1 현재의 일반화 제약

Click-Gaussian의 현재 설계는 **씬별 최적화(per-scene optimization)** 패러다임을 따르므로, 학습된 특징 필드는 특정 씬에 특화되어 있다. 이는 다음과 같은 일반화 제약을 야기한다:

1. **새로운 씬에 대한 제로샷(zero-shot) 세그멘테이션 불가**
2. **다양한 도메인(예: 의료 영상, 위성 영상)에 대한 범용성 부족**
3. **훈련 뷰 수에 따른 특징 품질 의존성**

### 6.2 일반화 성능 향상의 잠재적 방향

#### (a) SAM2 통합을 통한 동적 씬 대응
Click-Gaussian의 훈련이 완료되면 사용자는 이전 방법들보다 더 신속하게 Coarse 및 Fine 레벨에서 원하는 객체를 선택할 수 있다. 이러한 향상된 능력은 다양한 애플리케이션에서 효율적이고 정밀한 3D 환경 수정을 개선할 잠재력을 가지고 있다.

더 강력한 기반 모델(SAM2 등)과 통합하면 동적 씬에서의 일반화를 강화할 수 있다.

#### (b) GFL의 일반화 가능성
GFL은 여러 훈련 뷰에 걸쳐 전역 특징 후보들을 체계적으로 집계하여 3D 특징 필드의 발전을 일관성 있게 안내하는 새로운 전략이다. GFL 기법은 개별 2D 세그멘테이션 마스크에 내재된 모호성의 영향을 완화하여 특징 학습의 견고성과 신뢰성을 향상시킨다.

GFL의 전역적 클러스터링 전략은 씬의 종류에 무관하게 적용 가능하며, 이를 메타러닝(meta-learning) 프레임워크로 확장하면 씬 간 일반화가 가능하다.

#### (c) 다중 세분성 레벨로의 확장
방법은 SAM의 세 가지 레벨 마스크(whole, part, subpart)를 두 가지 방식(three-level-score 및 three-level-area)으로 채택할 수 있다. 각 접근법은 각 레벨에 대해 가장 높은 점수의 세그먼트와 가장 작은 세그먼트를 각각 우선시한다. 이 경우 $\mathbf{f}_i \in \mathbb{R}^{24}$를 세 가지 세분성 레벨로 분할한다.

이는 3단계 이상의 계층적 특징 표현으로의 자연스러운 확장 가능성을 보여주며, 더 세밀한 씬 이해를 통해 다양한 타겟 도메인에서의 적응력을 높일 수 있다.

#### (d) 언어 기반 일반화 (Open-Vocabulary)
향후 연구는 더욱 세밀한 세그멘테이션 세부 사항을 위한 두 레벨 이상의 세분성 개념 확장을 탐색할 수 있다. 또한 Click-Gaussian을 콘텐츠 생성을 위한 고급 디퓨전 모델과 통합하면 3D 씬 합성의 충실도와 다양성 모두를 향상시킬 수 있다.

CLIP/DINO와의 결합을 통해 텍스트 기반 쿼리로 세그멘테이션하는 Open-Vocabulary 확장이 가능하다. 이 경우 특징 필드를 언어 공간에 정렬하여 학습하면 된다:

$$\mathbf{f}_i^{aligned} = \text{MLP}(\mathbf{f}_i) \in \mathbb{R}^{D_{CLIP}}$$

---

## 7. 2020년 이후 관련 최신 연구 비교 분석

### 7.1 NeRF 기반 세그멘테이션 계열

| 연구 | 연도 | 특징 | 한계 |
|---|---|---|---|
| **NeRF-SOS** | 2023 | 자기지도 학습 기반 객체 세그멘테이션 | 느린 렌더링 속도 |
| **SA3D (Cen et al.)** | 2023 | SAM + NeRF 결합, 단일 뷰 프롬프트 | NeRF의 높은 계산 비용 |
| **GARField** | 2024 (CVPR) | 물리적 3D 스케일 기반 다중 세분성 | 추론 속도 저하, 암시적 특징 필드 |

GARField의 암시적 특징 필드 의존성은 다른 스케일에서의 세그멘테이션에 반복적인 쿼리가 필요하여 효율성이 저하된다. 이와 대조적으로 SAGA의 scale-gate 메커니즘은 3D-GS와 직접 통합되어 추가 계산 없이 효율성을 향상시킨다.

### 7.2 3DGS 기반 세그멘테이션 계열

여러 주목할 만한 연구들이 이 분야를 발전시켰다: Feature3DGS, Gaussian Grouping, OmniSeg3D, SAGA, 그리고 Click-Gaussian. 이러한 방법들은 일반적으로 SAM을 사용하여 2D 마스크나 특징을 추출하고, 대조 학습이나 증류를 통해 정보를 3D 공간으로 변환한 다음, 결과적인 3D 세그멘테이션 특징을 2D로 다시 투영하여 새로운 시점에서 세그멘테이션을 가능하게 하는 유사한 파이프라인을 따른다. 그러나 각 방법은 SAM에서 파생된 2D 정보를 처리하는 방식이 다르며 고유한 한계에 직면한다.

상세 비교표:

| 연구 | 연도/학회 | 핵심 기법 | 속도 | 주요 특징 |
|---|---|---|---|---|
| **Feature3DGS** | 2024 (CVPR) | SAM/LSeg 증류 | 느림 | 최초 3DGS 특징 필드 |
| **Gaussian Grouping** | 2024 (ECCV) | SAM 기반 객체 연관 | 느림 | 오픈월드 재구성+편집 |
| **SAGA** | 2024 (AAAI-25) | Scale-gated 대조 학습 | 4ms | 멀티 세분성, SAM 증류 |
| **OmniSeg3D** | 2024 | 계층적 대조 학습 | 중간 | 전체 씬 동시 세그멘테이션 |
| **Click-Gaussian** | 2024 (ECCV) | Two-level GFL + 대조 학습 | **10ms** | 후처리 없음, 두 단계 세분성 |

SAGA는 3D Gaussian Splatting(3D-GS) 기반의 고효율 3D 프롬프터블 세그멘테이션 방법으로, 2D 시각적 프롬프트가 주어지면 4ms 내에 3D Gaussian으로 표현된 해당 3D 타겟을 세그멘테이션할 수 있다. 이는 각 3D Gaussian에 scale-gated affinity feature를 부착하여 다중 세분성 세그멘테이션을 향한 새로운 속성을 부여함으로써 달성된다.

Gaussian Grouping은 Gaussian Splatting을 확장하여 2D SAM을 활용한 오픈월드 3D 씬의 공동 재구성 및 세그멘테이션을 수행하며, 다양한 3D 씬 편집 작업을 효율적으로 지원한다.

### 7.3 Click-Gaussian의 차별점

Click-Gaussian은 이중 레벨 특징 필드와 Global Feature Guidance Learning(GFL)을 사용하여 뷰 간에 일관된 3D Gaussian 의미론적 표현을 구축한다.

**SAGA와의 차이**: SAGA는 3D 물리적 스케일을 scale-gate 파라미터로 사용하는 반면, Click-Gaussian은 면적 기반으로 SAM 마스크를 두 레벨로 분류하고 GFL로 뷰 일관성을 명시적으로 확보한다. 또한 Click-Gaussian은 **후처리 없이** 구분 가능한 특징 필드를 직접 학습하는 반면, SAGA는 여전히 일부 후처리에 의존한다.

### 7.4 최신 후속 연구 트렌드 (2024~2025)

증류 기반 방법은 2D 기반 모델의 특징을 3D 프리미티브로 전달하여 특징이 Gaussian 표현에 정렬되면 제로샷 3D 세그멘테이션의 초기 시연을 가능하게 한다. 대조 학습 프레임워크는 잠재적으로 불일관한 멀티뷰 2D 마스크를 입력으로 받아 3D 임베딩 공간에서 뷰 간 일관성을 강화하여 일관된 새 시점 세그멘테이션을 가능하게 한다. 인터랙션 기반 방법들은 Gaussian의 기하학 인식 구조에서 이점을 얻어 클릭 또는 프롬프트 기반 작업을 지원하기 위해 직접 Gaussian에서 작동한다.

---

## 8. 앞으로의 연구에 미치는 영향 및 연구 시 고려할 점

### 8.1 연구에 미치는 영향

#### (a) 3D 씬 편집 분야
이러한 기술의 발전은 3D 씬과의 실시간 상호작용을 크게 향상시켜 3D 객체 조작 작업에서 더 직관적이고 반응적인 경험을 가능하게 할 잠재력이 있다. 이러한 발전은 3D 씬 편집 능력을 향상시킬 뿐만 아니라 다양한 분야에서 3D 씬 표현의 실용적인 응용을 넓힐 잠재력을 가지고 있다.

사전 학습된 3DGS에 Click-Gaussian을 적용하여 사용자는 크기 조정, 이동, 텍스트 기반 편집 등 원하는 수정 작업을 유연하게 수행할 수 있다.

#### (b) 로보틱스 및 자율 주행
이 접근법은 정밀하고 신속한 3D 씬 조작을 위한 유망한 솔루션을 제공하며, 다양한 도메인의 애플리케이션을 촉진할 잠재력이 있다.

실시간으로 10ms 이내에 3D 세그멘테이션을 제공하는 Click-Gaussian의 능력은 로봇 팔 제어, 장면 이해, 자율 주행 시스템에서의 객체 분리에 활용 가능하다.

#### (c) 증강현실/혼합현실(AR/MR)
Click-Gaussian은 인터랙티브 3D 세그멘테이션의 어렵고 도전적인 측면을 극복하는 데 잘 정립된 접근법을 제시한다. Click-Gaussian의 효율성, 정확도, 실시간 능력의 조합은 주목할 만한 발전을 나타내며, 다양한 응용 분야에서 더 직관적이고 반응적인 3D 객체 조작을 위한 길을 열어준다.

### 8.2 향후 연구 시 고려할 점

#### (1) Granularity 레벨 확장
향후 연구는 더욱 세밀한 세그멘테이션 세부 사항을 위한 두 레벨 이상의 세분성 개념 확장을 탐색할 수 있다.

두 레벨(Coarse/Fine)을 넘어 three-level 또는 계층적 세분성으로 확장하는 방향이 유망하며, SAM의 three-level 출력(whole/part/subpart)을 더 효과적으로 활용하는 방법 연구가 필요하다.

#### (2) 씬 독립적(Scene-agnostic) 일반화
현재의 씬별 특징 학습을 넘어, 사전 학습된 범용 3D 특징 인코더를 구축하면 새로운 씬에 대한 zero-shot 또는 few-shot 세그멘테이션이 가능해질 것이다. 이를 위해 **대규모 3D 씬 데이터셋**과 **메타러닝** 전략의 결합이 필요하다.

#### (3) 동적 씬으로의 확장
시간에 따라 변화하는 4D Gaussian도 세그멘테이션할 수 있다는 가능성이 시사되었으나, 동적 씬에서의 뷰 일관성 학습은 더 어려운 문제다. SAM2와 같은 비디오 기반 마스크 전파 모델의 통합이 핵심 연구 방향이 될 것이다.

#### (4) 언어 지시 기반 세그멘테이션과 결합
Click-Gaussian을 콘텐츠 생성을 위한 고급 디퓨전 모델과 통합하면 3D 씬 합성의 충실도와 다양성 모두를 향상시킬 수 있다.

LangSplat, LERF 등과 결합하여 클릭 기반이 아닌 자연어 기반 세그멘테이션으로 발전시키면 더 범용적인 인터페이스를 제공할 수 있다.

#### (5) 3D 기하학 정보 활용 강화
기존 방법들은 주로 2D 마스크나 특징을 3D 공간으로 변환하는 데 집중하여, 3D 기하학에 내재된 공간적 정보를 충분히 활용하지 못하고 있다.

법선 벡터, 깊이 불연속성 등의 기하학적 단서를 GFL의 클러스터링 과정에 통합하면 경계 정확도를 높일 수 있다.

#### (6) 훈련 효율성 개선
GARField, SAGA, Feature3DGS와 같은 전통적인 오프라인 방법들은 상당한 "cold-start" 문제를 겪는다. 이 접근법들은 차단(blocking)이 발생하여 사용자들은 세그멘테이션 결과가 나오기 전에 30분에서 45분의 필수 훈련 기간을 기다려야 한다.

Click-Gaussian도 훈련 시간 단축이 필요하며, 증류 기반 초기화나 온라인 학습 전략을 결합하면 실용성이 크게 향상될 것이다.

---

## 📚 참고 자료 및 출처

| # | 출처 | 링크 |
|---|---|---|
| 1 | **Click-Gaussian arXiv (2407.11793)** | https://arxiv.org/abs/2407.11793 |
| 2 | **Click-Gaussian ECCV 2024 논문 (ECVA PDF)** | https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00406.pdf |
| 3 | **Click-Gaussian 공식 프로젝트 페이지** | https://seokhunchoi.github.io/Click-Gaussian/ |
| 4 | **Springer ECCV 2024 출판본** | https://link.springer.com/chapter/10.1007/978-3-031-72646-0_17 |
| 5 | **Semantic Scholar 논문 페이지** | https://www.semanticscholar.org/paper/2d81e6af252b16238b10cf6009514a61ed60d77a |
| 6 | **Hugging Face Paper 페이지** | https://huggingface.co/papers/2407.11793 |
| 7 | **SAGA (Segment Any 3D Gaussians) arXiv 2312.00860** | https://arxiv.org/abs/2312.00860 |
| 8 | **Gaussian Grouping GitHub (ECCV 2024)** | https://github.com/lkeab/gaussian-grouping |
| 9 | **SAGA GitHub (AAAI-25)** | https://github.com/Jumpat/SegAnyGAussians |
| 10 | **A Survey on 3D GS Applications (arXiv 2508.09977)** | https://arxiv.org/html/2508.09977v1 |
| 11 | **SAGOnline (arXiv 2508.08219)** | https://arxiv.org/html/2508.08219 |
| 12 | **Seg-Wild (arXiv 2507.07395)** | https://arxiv.org/html/2507.07395 |
| 13 | **PointGauss (arXiv 2508.00259)** | https://arxiv.org/html/2508.00259 |
| 14 | **EmergentMind 논문 분석** | https://www.emergentmind.com/papers/2407.11793 |
| 15 | **ECCV 2024 공식 포스터 페이지** | https://eccv.ecva.net/virtual/2024/poster/568 |
