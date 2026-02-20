# 3D Highlighter: Localizing Regions on 3D Shapes via Text Descriptions

---

## 1. 핵심 주장과 주요 기여 요약

**3D Highlighter**는 텍스트 설명만으로 3D 메시 표면의 의미적(semantic) 영역을 자동으로 지역화(localization)하는 기법이다. 핵심 주장은 다음과 같다:

1. **Out-of-domain 지역화 능력**: 기하학적 신호가 부재한 형상에서도 의미적으로 관련된 영역을 추론할 수 있다 (예: 말 위에 '목걸이', 촛대 위에 '모자' 배치). 이를 저자들은 **"hallucinated highlighting"**이라 명명한다.
2. **3D 데이터셋/어노테이션 불필요**: 사전 학습된 CLIP 모델의 비전-언어 임베딩 공간을 활용하여, 어떠한 3D 데이터셋이나 3D 사전 학습 없이도 동작한다.
3. **Neural Field 기반 확률 예측**: MLP 기반 neural field가 메시 표면의 각 점을 하이라이트 확률로 매핑하며, 확률 가중 블렌딩을 통해 부드럽고 일관된 지역화를 달성한다.
4. **높은 유연성과 일반성**: 특정 카테고리나 파트 클래스에 제한되지 않으며, 다양한 메시 해상도와 삼각화(triangulation)에 대해 전이(transfer) 가능하다.

**주요 기여**:
- 텍스트 기반 3D 형상 의미 지역화의 최초 프레임워크 제시
- Hallucinated highlighting이라는 새로운 과제 정의
- Neural field + 확률 블렌딩 + CLIP 가이던스의 조합 설계
- 선택적 편집, 조합적 스타일화, 다중 클래스 분할 등 다양한 응용 시연

---

## 2. 상세 분석: 문제, 방법, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

기존 3D 분할(segmentation) 방법들은 두 가지 근본적 한계를 가진다:

- **기하학 기반 방법**: 곡률, 볼록성 등 저수준 기하학적 특징에 의존하여, 기하학적 신호가 없는 의미적 영역(예: 옷을 입지 않은 인체 모델에서 "셔츠 영역")을 찾을 수 없다.
- **데이터 기반 방법**: 대규모 3D 어노테이션 데이터셋과 고정된 파트 클래스가 필요하여, 새로운 카테고리나 out-of-domain 개념에 대해 일반화하기 어렵다.

본 논문은 **"텍스트 설명만으로 임의의 3D 메시 위에 의미적 영역을 지역화하는 문제"**를 정의하고, 3D 데이터 없이 해결하는 것을 목표로 한다.

### 2.2 제안하는 방법

#### (a) Neural Highlighter (신경 하이라이터)

메시 $M$은 정점 $V \in \mathbb{R}^{n \times 3}$과 면 $F \in \{1, \ldots, n\}^{m \times 3}$으로 표현된다. Neural highlighter는 neural field로서, MLP $F_\theta$가 3D 좌표를 하이라이트 확률로 매핑한다:

$$F_\theta(\mathbf{x}_v) = p_v, \quad \mathbf{x}_v = (x, y, z), \quad p_v \in [0, 1]$$

여기서 $p_v$는 정점 $v$가 텍스트로 지정된 영역에 속할 확률이다.

**MLP의 spectral bias**: MLP는 저주파 해에 대한 편향(spectral bias)을 가지므로, 연속적이고 경계가 명확한 지역화를 자연스럽게 생성하며 노이즈를 억제한다. 이 때문에 **positional encoding을 사용하지 않는다**.

**MLP 구조**: 6개의 선형 레이어, 각 폭 256, 처음 5개 레이어 후에 ReLU + Layer Norm 적용, 마지막 레이어 후에 Softmax 적용 (2-class: 타겟 영역 / 비-타겟 영역).

#### (b) Mesh Color Blending (메시 색상 블렌딩)

각 정점의 색상 $C_v$는 하이라이트 색상 $H$와 배경 회색 $G$의 확률 가중 선형 결합으로 결정된다:

$$C_v = p_v \cdot H + (1 - p_v) \cdot G$$

- **초기화**: 모든 정점의 확률을 약 0.5로 초기화하여 전체 메시가 중간 색상으로 시작
- **연속적 그래디언트 제공**: argmax 기반 이산적 색상 할당 대비 부드러운 최적화 지형(landscape) 생성
- **명시적 분할 표현**: 직접 색상을 최적화하는 방식과 달리, 어떤 정점이 하이라이트 영역에 속하는지에 대한 명시적 확률 정보를 제공

#### (c) Unsupervised Guidance (비지도 가이던스)

CLIP의 비전-언어 임베딩 공간을 활용하여 최적화를 유도한다.

**타겟 텍스트 설계**: 
$$T = \text{"a gray [object] with highlighted [region]"}$$

**렌더링 및 증강**: 미분 가능 렌더러로 다중 뷰 렌더링 후, 2D 원근 증강(perspective augmentation) $\phi$를 적용한다.

**이미지 임베딩 집계**: 각 뷰 $\psi$에 대해 렌더링된 이미지 $I_\psi$를 CLIP 이미지 인코더 $E_I$로 인코딩하고 평균한다:

$$\mathbf{e}_I = \frac{1}{n} \sum_{\psi} E_I(\phi(I_\psi)) \in \mathbb{R}^{768}$$

**텍스트 임베딩**:

$$\mathbf{e}_T = E_T(T) \in \mathbb{R}^{768}$$

**손실 함수**: 이미지 임베딩과 텍스트 임베딩 간의 음의 코사인 유사도를 최소화한다:

$$\arg\min_{\theta} \mathcal{L}(\theta) = -\frac{\mathbf{e}_I \cdot \mathbf{e}_T}{|\mathbf{e}_I| \cdot |\mathbf{e}_T|}$$

**Primary View 선택**: 360° 뷰를 렌더링하여 타겟 텍스트와의 CLIP 유사도가 가장 높은 뷰를 primary view로 자동 선택하고, 이를 중심으로 가우시안 분포에서 $n$개의 뷰를 샘플링한다.

### 2.3 모델 구조 요약

| 구성 요소 | 상세 |
|----------|------|
| 입력 | 3D 좌표 $(x, y, z)$ |
| 네트워크 | MLP, 6 선형 레이어, 폭 256 |
| 활성화 | ReLU + LayerNorm (1-5층), Softmax (6층) |
| 출력 | 2-class 확률 벡터 (하이라이트 / 비-하이라이트) |
| 렌더러 | 미분 가능 렌더러 [Chen et al., 2019] |
| CLIP 모델 | ViT-L/14, 224×224 |
| 최적화 | Adam, lr= $1 \times 10^{-4}$ , 2500 iterations |
| 소요 시간 | ~5분 (Nvidia A40 GPU) |

### 2.4 성능 평가

#### 정량적 평가: CLIP R-Precision

3D 텍스트 기반 지역화에 대한 기존 벤치마크가 없으므로, Dream Fields에서 영감을 받은 **CLIP R-Precision** 메트릭을 설계하였다:

| 방법 | ViT-L/14 ↑ | ViT-B/16 ↑ |
|------|-----------|-----------|
| LSeg [20] | 18.75 | 6.25 |
| Text2LIVE [2] | 43.75 | 31.25 |
| **3D Highlighter (Ours)** | **81.25** | **43.75** |

- **LSeg**: 2D 시맨틱 분할 기법으로, 장면 내 전체 객체 분할에 특화되어 있으나 단일 객체 내 파트 분할에 취약
- **Text2LIVE**: 2D 이미지 편집 마스크를 추론하나, 하이라이팅에 필요한 날카로운 분할 경계 생성에 부적합

#### Ablation Study

| 설정 | CLIP Score |
|------|-----------|
| **Full (ours)** | **0.332** |
| Direct optimization | 0.319 |
| No probability blend | 0.297 |
| No 2D augmentations | 0.287 |

- **Neural field 제거 (direct)**: 노이즈가 많은 얼룩 패턴 발생
- **확률 블렌딩 제거 (no blend)**: 불안정한 그래디언트, 노이즈 하이라이트
- **2D 증강 제거 (no augs)**: 퇴화된(degenerate) 해 생성

### 2.5 한계

1. **텍스트-형상 호환성 요구**: 객체 설명이 실제 기하학과 호환되어야 한다 (예: 낙타를 "의자"로 기술하면 의미 있는 결과 생성 불가, Fig. 13)
2. **CLIP 편향에 종속**: CLIP의 시각-의미 연관이 인간의 직관과 일치하지 않는 경우 부정확한 지역화 (예: 토끼의 "귀"를 머리 옆으로 지역화, Fig. 19)
3. **최적화 민감성**: 일부 메시-프롬프트 조합에서 seed에 따라 결과가 변동 (Fig. 20)
4. **Per-shape 최적화**: 각 메시-텍스트 쌍마다 별도의 최적화가 필요 (~5분)하여, 대규모 적용 시 효율성 문제
5. **단일 클래스 지역화**: 기본적으로 한 번에 하나의 영역만 지역화 가능 (다중 클래스는 후처리 필요)

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화 성능의 강점

**3D Highlighter의 일반화 능력은 세 가지 핵심 설계에서 기인한다:**

1. **CLIP의 오픈 보캐블러리(open-vocabulary) 특성**: 특정 파트 클래스나 카테고리에 제한되지 않고, CLIP이 학습한 광범위한 시각-언어 지식을 활용한다. 이로 인해 humanoid, 동물, 제조 객체 등 다양한 도메인의 메시에서 'shoes', 'hat', 'necklace' 등 다양한 속성을 지역화할 수 있다.

2. **Neural field 표현의 형상 독립성**: Neural highlighter가 좌표 공간의 연속 함수로 학습되므로, 동일 객체의 다른 삼각화(re-meshed, subdivided)에 대해 **추가 최적화 없이** 지역화를 전이할 수 있다 (Fig. 9).

3. **Hallucinated highlighting**: 기하학적 단서가 없는 영역도 의미적으로 추론 가능하다 — 이는 CLIP의 cross-modal transfer 능력에 의해 가능하다.

### 3.2 일반화 한계 및 개선 방향

#### (A) Shape 간 일반화 (Cross-shape Generalization)

**현재 한계**: 각 메시-텍스트 쌍에 대해 개별적으로 최적화해야 하며, 한 형상에서 학습된 "shoes" 지역화가 다른 형상으로 자동 전이되지 않는다.

**개선 방향**:
- **Meta-learning 접근**: MAML 또는 Reptile 같은 메타 학습 프레임워크로 초기화를 학습하여, 새로운 메시-텍스트 쌍에 대해 소수의 최적화 스텝만으로 수렴하도록 할 수 있다.
- **Feed-forward 예측**: 3D 인코더(예: PointNet++, PointTransformer)와 텍스트 인코더를 결합한 feed-forward 네트워크를 대규모 합성 데이터로 학습하여, 추론 시 최적화 없이 즉시 지역화를 예측하는 방식.

#### (B) 텍스트 프롬프트에 대한 일반화

**현재 한계**: "highlighted"라는 특정 용어에 의존하며, CLIP의 binding 문제(속성-객체 결합 오류)에 취약하다.

**개선 방향**:
- **프롬프트 앙상블**: 동일한 의미를 다양한 텍스트로 표현하여 앙상블하면 robustness 향상 가능
- **대형 비전-언어 모델 활용**: CLIP 대신 더 강력한 모델(예: SigLIP, OpenCLIP, BLIP-2)을 사용하여 binding 능력과 fine-grained understanding을 개선

#### (C) 뷰포인트 일반화

**현재 강점**: Primary view 선택이 다양한 뷰에서 강건한 결과를 보인다 (Fig. 6).

**추가 개선**:
- **360° 균일 커버리지**: 구면 상의 균등 분포 샘플링과 중요도 기반 가중치를 결합하면 가시성이 낮은 영역에 대한 지역화 정확도를 더 높일 수 있다.

#### (D) Foundation Model로의 확장

- **Segment Anything Model (SAM)** 등의 2D foundation model을 multi-view에서 활용하여 3D consistency를 강화하는 방식이 유망하다.
- **3D foundation model** (예: OpenScene, LERF)을 사전 학습된 feature extractor로 활용하여, per-shape 최적화의 부담을 줄이면서도 일반화 성능을 유지할 수 있다.

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 연구에 미치는 영향

1. **Zero-shot 3D 이해의 새 패러다임**: 3D 데이터 없이 2D 비전-언어 모델의 지식을 3D로 전이하는 "lifting" 패러다임을 3D 파트 지역화에 적용한 선구적 사례로, 이후 수많은 후속 연구에 영감을 제공하였다.

2. **Hallucinated Highlighting이라는 새로운 과제**: 기하학적 단서 없이 의미적 영역을 추론하는 과제를 공식화함으로써, 3D 이해의 범위를 확장하였다.

3. **3D 편집/스타일화의 제어 가능성 향상**: 전역적 스타일화의 한계를 극복하고, 의미 기반 지역 선택을 통한 조합적(compositional) 편집의 가능성을 시연하였다.

4. **평가 프로토콜 제안**: 3D 텍스트 기반 지역화에 대한 CLIP R-Precision 기반 평가 전략을 제안하여 후속 연구의 벤치마킹 기반을 마련하였다.

### 4.2 향후 연구 시 고려할 점

1. **효율성**: Per-shape 최적화(~5분/쌍)는 대규모 적용에 병목이 된다. Feed-forward 방식 또는 amortized optimization 연구가 필요하다.

2. **Multi-part 동시 지역화**: 현재는 단일 영역만 지역화하며 다중 클래스는 후처리에 의존한다. End-to-end 다중 영역 지역화가 필요하다.

3. **3D Consistency 강화**: 2D 렌더링 기반 CLIP 감독은 뷰 간 불일치를 야기할 수 있다. 3D-aware loss 또는 multi-view consistency constraint의 도입이 필요하다.

4. **CLIP 편향 완화**: CLIP의 문화적, 시각적 편향이 지역화 결과에 직접 영향을 미치므로, debiasing 기법이나 인간 피드백(RLHF) 기반 fine-tuning을 고려해야 한다.

5. **파트 대응(Part Correspondence)**: 저자들이 미래 연구로 언급한 바와 같이, 위상적으로 다르지만 의미적으로 관련된 형상 간의 파트 대응 확보가 중요한 후속 과제이다.

6. **정량적 평가의 한계**: CLIP R-Precision은 CLIP 자체의 편향을 반영할 수 있으므로, 인간 평가(human study)나 GT 어노테이션 기반 IoU 등 보완적 평가가 필요하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 특징 | 3D Highlighter와의 비교 |
|------|------|----------|----------------------|
| **CLIP** (Radford et al.) | 2021 | 대규모 이미지-텍스트 대조 학습, 오픈 보캐블러리 시각 인식 | 3D Highlighter의 핵심 감독 신호 제공. CLIP 없이는 본 방법이 성립하지 않음 |
| **Text2Mesh** (Michel et al.) | 2022 (CVPR) | CLIP 가이드 3D 메시 스타일화, 텍스트 기반 텍스처/변위 생성 | 전역 스타일화에 특화, 파트 단위 선택 불가. 3D Highlighter는 이를 보완하여 지역 편집 가능 |
| **DreamFusion** (Poole et al.) | 2022 | SDS(Score Distillation Sampling)로 2D diffusion 모델을 3D 생성에 활용 | 전체 형상 생성에 초점. 3D Highlighter는 기존 형상의 파트 지역화에 초점 |
| **LSeg** (Li et al.) | 2022 (ICLR) | Language-driven semantic segmentation (2D) | 장면 내 전체 객체 분할에 강하나, 단일 객체 내 파트 분할에 약함 (R-Precision: 18.75 vs 81.25) |
| **Text2LIVE** (Bar-Tal et al.) | 2022 | 텍스트 기반 2D 이미지/비디오 레이어 편집 | 소프트 마스크 추론 가능하나, 날카로운 분할 경계 생성에 부적합 |
| **LERF** (Kerr et al.) | 2023 (ICCV) | Language Embedded Radiance Fields, 3D 장면에 CLIP 특징을 NeRF에 증류 | 3D 장면 내 오픈 보캐블러리 쿼리 가능하나, 메시가 아닌 NeRF 표현에 특화. 3D Highlighter는 메시 직접 처리 |
| **OpenScene** (Peng et al.) | 2023 (CVPR) | 3D 포인트 클라우드에 2D 비전-언어 특징을 증류하여 오픈 보캐블러리 3D 이해 | 대규모 3D 장면에 적용 가능하나, hallucinated highlighting과 같은 out-of-domain 지역화 능력은 불명 |
| **SAM (Segment Anything)** (Kirillov et al.) | 2023 (ICCV) | 프롬프트 기반 범용 2D 분할 모델 | 2D에서 강력한 분할 능력, 3D 확장 시 multi-view consistency 문제. SA3D 등의 확장 연구 존재 |
| **SA3D** (Cen et al.) | 2023 | SAM을 NeRF와 결합한 3D 분할 | NeRF 표현에 한정, 메시 기반이 아님. 기하학 부재 영역에 대한 hallucinated 지역화 능력 부족 |
| **PartSLIP** (Liu et al.) | 2023 (CVPR) | GLIP을 활용한 zero-shot 3D 파트 분할 | 2D 객체 검출기 기반으로 3D 파트 분할을 달성하나, 사전 정의된 파트 개념에 더 의존적 |
| **3D-OVS** (Liu et al.) | 2023 | Open-vocabulary 3D scene understanding | NeRF 기반 장면 이해에 초점, 단일 메시의 fine-grained 파트 지역화와는 다른 문제 설정 |
| **CLIP-Fields** (Shafiullah et al.) | 2023 | 3D 장면에 CLIP 특징을 neural field로 저장 | 장면 수준의 오픈 보캐블러리 쿼리, 객체 내 파트 수준 지역화에는 한계 |

### 비교 분석의 핵심 통찰

1. **NeRF vs. Mesh**: LERF, SA3D 등은 NeRF 표현에 특화되어 있어, 메시 기반 워크플로우와의 호환성이 낮다. 3D Highlighter는 메시를 직접 처리하여 3D 모델링 파이프라인과의 통합이 용이하다.

2. **Scene-level vs. Part-level**: OpenScene, CLIP-Fields 등은 장면 수준의 객체 분할에 강하나, 단일 객체 내 fine-grained 파트 분할에서는 3D Highlighter가 우위를 보인다.

3. **In-domain vs. Out-of-domain**: 대부분의 후속 연구는 기하학적으로 존재하는 파트의 분할에 초점을 맞추는 반면, 3D Highlighter의 hallucinated highlighting은 기하학적으로 부재한 개념의 지역화라는 고유한 능력을 제공한다.

4. **Foundation Model 시대의 방향**: 2023년 이후 SAM, DINOv2, SigLIP 등 더 강력한 foundation model이 등장함에 따라, CLIP 대신 이들을 감독 신호로 활용하거나 결합하면 3D Highlighter의 정확도와 일반화 성능을 크게 향상시킬 수 있다. 특히 **diffusion model 기반 SDS loss**와 결합하면 CLIP의 제한된 fine-grained 이해를 보완할 수 있다.

---

## 결론

3D Highlighter는 텍스트만으로 3D 메시의 의미적 영역을 지역화하는 최초의 프레임워크로서, neural field 기반 확률 예측과 CLIP 가이던스의 결합을 통해 높은 유연성과 일반성을 달성하였다. 특히 hallucinated highlighting이라는 독창적 능력은 기존 3D 분할 방법론의 근본적 한계를 극복한다. 다만 per-shape 최적화의 비효율성, CLIP 편향에의 종속, 다중 클래스 지역화의 제한 등이 남아있으며, feed-forward 아키텍처, 더 강력한 foundation model 활용, 3D consistency 강화 등의 방향으로 향후 연구가 진행되어야 한다. 이 연구는 2D 비전-언어 모델의 지식을 3D 형상 이해로 전이하는 패러다임의 중요한 이정표로서, 이후 LERF, OpenScene, PartSLIP 등 다수의 후속 연구에 영감을 제공하였다.
