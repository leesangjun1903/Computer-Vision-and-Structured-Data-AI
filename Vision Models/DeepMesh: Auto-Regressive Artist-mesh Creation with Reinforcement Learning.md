
# DeepMesh: Auto-Regressive Artist-mesh Creation with Reinforcement Learning

---

## 📌 1. 핵심 주장 및 주요 기여 요약

삼각형 메쉬(Triangle Mesh)는 3D 응용 프로그램에서 효율적인 조작과 렌더링을 위해 핵심적인 역할을 담당하지만, 기존의 자동회귀 방법은 이산 버텍스 토큰(discrete vertex tokens)을 예측하는 방식으로 구조적 메쉬를 생성하며 **제한된 면(face) 수와 메쉬 불완전성**의 문제를 안고 있었습니다.

이를 해결하기 위해 DeepMesh는 두 가지 핵심 혁신을 통해 메쉬 생성을 최적화하는 프레임워크를 제안합니다: **(1) 새로운 토큰화(tokenization) 알고리즘을 포함한 효율적인 사전학습 전략**, 그리고 **(2) Direct Preference Optimization(DPO)을 통한 인간 선호도 정렬을 위한 강화학습(RL)의 3D 메쉬 생성에의 최초 도입**.

이 연구는 칭화대학교(Tsinghua University), 난양공과대학교(Nanyang Technological University), ShengShu 소속 연구자들에 의해 개발되었습니다.

논문은 ICCV 2025에 채택되었으며, arXiv 프리프린트 번호는 2503.15265입니다.

---

## 📌 2. 문제 정의, 제안 방법(수식 포함), 모델 구조, 성능 및 한계

---

### 2.1 해결하고자 하는 문제

자동회귀 모델을 메쉬 생성에 적용하는 기존 시도들은 다음의 핵심 한계를 가집니다:
- **긴 토큰 시퀀스(Long Token Sequences)**: 메쉬 표현에 계산 비용이 큰 긴 시퀀스가 필요
- **학습 불안정성(Training Instability)**: 저품질 메쉬가 포함된 학습 데이터로 인한 손실 스파이크 및 수렴 장애
- **인간 선호도와의 정렬 부재(Limited Alignment with Human Preferences)**: 생성된 메쉬가 아티스트가 자연스럽게 구현하는 미적 품질 및 토폴로지 최적화를 반영하지 못함

메쉬는 아티스트가 수작업으로 제작하거나, Neural Radiance Fields(NeRF) 또는 Signed Distance Fields(SDF) 같은 볼류메트릭 필드에 Marching Cubes를 적용해 자동으로 생성될 수 있으나, 전자가 위상적(topological)으로 훨씬 최적화된 결과를 산출합니다.

---

### 2.2 제안하는 방법

#### ① 개선된 메쉬 토큰화 알고리즘

DeepMesh의 개선된 메쉬 토큰화 알고리즘은 고해상도에서 메쉬를 효율적으로 이산화하며 **기하학적 세부 정보를 손실 없이 약 72%의 압축률**을 달성합니다.

토큰화 프로세스는 계층적 시스템을 채택합니다:
- **로컬 패치(Local Patches)**: 연결성 기반으로 메쉬 면을 로컬 패치로 분할하여 중복을 최소화
- **인덱싱 전략**: 좌표계를 다중 레벨 블록으로 분할하여 블록 내 오프셋으로 버텍스 위치 표현
- **양자화(Quantization)**: 버텍스를 유한한 수의 빈(bin)으로 양자화 (해상도 $r = 512$)

주요 메커니즘으로는 **로컬 연결성을 보존하는 면 순회(Local-aware Face Traversal)**, **블록 분할 기반 계층적 좌표 표현(Coordinate Scaling and Merging)**, 그리고 **어휘 크기를 줄이기 위한 좌표 양자화(Quantization)**가 있으며, 이를 통해 기존 방법 대비 시퀀스 길이를 **72% 단축**하여 고해상도 메쉬 생성이 가능해집니다.

메쉬 토큰 시퀀스의 자동회귀 모델링은 다음의 조건부 확률의 곱으로 표현됩니다:

$$p(\mathbf{T}) = \prod_{t=1}^{N} p(T_t \mid T_1, T_2, \ldots, T_{t-1}, \mathbf{c})$$

여기서 $\mathbf{T} = (T_1, T_2, \ldots, T_N)$은 메쉬 토큰 시퀀스, $\mathbf{c}$는 포인트 클라우드 또는 이미지 컨디션 입력, $N$은 전체 토큰 수입니다.

#### ② 강화학습 기반 인간 선호도 정렬 (DPO)

DeepMesh는 사전 학습된 모델을 사용해 쌍(pairwise) 학습 데이터를 생성하고, 인간 평가와 3D 기하학 메트릭으로 이를 주석 처리한 후, 해당 선호도 레이블링 샘플로 강화학습을 통해 모델을 파인튜닝합니다.

이를 위해 3D 메트릭과 인간 평가를 결합한 채점 기준을 설계하고, 5,000개의 선호도 쌍을 주석 처리한 후 DPO를 통해 모델을 사후 학습(post-train)하여 인간 선호도에 정렬시킵니다.

DPO(Direct Preference Optimization)의 목적함수는 다음과 같이 정의됩니다:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

여기서:
- $\pi_\theta$: 학습 중인 정책 모델 (DeepMesh)
- $\pi_{\text{ref}}$: 사전 학습된 참조 모델
- $y_w$: 선호(preferred) 샘플
- $y_l$: 비선호(rejected) 샘플
- $\beta$: KL 발산 정규화 강도를 조절하는 하이퍼파라미터

선호도 쌍의 주석은 생성된 메쉬 쌍을 비교하여 "geometry completeness(기하학적 완전성)", 세부 표현 수준, 그리고 시각적 품질을 기반으로 이루어집니다.

---

### 2.3 모델 구조

DeepMesh의 핵심 구조는 각 레이어에 크로스-어텐션(cross-attention)과 셀프-어텐션(self-attention) 레이어, 그리고 피드포워드 네트워크(FFN)를 포함하는 **자동회귀 트랜스포머**입니다. 포인트 클라우드 조건부 생성을 위해 Michelangelo 기반의 공동 학습된 **Perceiver Encoder**를 사용하며, 컨디션된 포인트 클라우드 특성은 크로스-어텐션을 통해 통합됩니다. 학습 가속화를 위해 **Hourglass Transformer**를 채택하여 성능을 유지하면서 메모리를 50% 절약합니다.

공개된 사전학습 가중치의 모델 크기는 **0.5B(5억 파라미터)**입니다.

훈련 데이터셋은 약 **500,000개**의 메쉬로 구성되어 있으며, 메쉬당 평균 face 수는 약 8,000개입니다.

전체 학습 파이프라인은 다음과 같이 표현됩니다:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pretrain}} + \lambda \cdot \mathcal{L}_{\text{DPO}}$$

- **Phase 1 (Pre-training)**: 개선된 토큰화 알고리즘으로 생성된 이산 메쉬 토큰에 대해 자동회귀 언어 모델링 손실 $\mathcal{L}_{\text{pretrain}}$ 최소화
- **Phase 2 (Post-training with DPO)**: 인간 선호 레이블 기반 선호도 최적화 $\mathcal{L}_{\text{DPO}}$ 적용

---

### 2.4 성능 향상

이러한 개선을 통해 DeepMesh 프레임워크는 **양자화 해상도 512에서 최대 30,000개의 면(face)**을 가진 다양하고 고품질의 아티스트 수준 메쉬를 생성할 수 있습니다.

DeepMesh는 생성된 기하학 품질과 세밀한 디테일 보존 모두에서 기존 베이스라인을 능가하며, 생성된 메쉬의 면 수가 기타 방법에 비해 현저히 많습니다.

강화학습이 효율성 보상과 삼각형 배치 최적화를 통해 모델을 개선하며, 인간 평가자 연구에서 DeepMesh의 출력이 경쟁 모델보다 선호되었고, 더 적은 수의 삼각형을 사용하면서도 중요한 세부 사항을 유지합니다.

추론 코드 최적화를 통해 생성 시간이 **50% 단축**되었습니다.

---

### 2.5 한계

사전 학습 모델은 고품질 메쉬를 생성할 수 있지만, 때때로 비미적인 외관과 불완전한 기하학의 문제가 발생합니다.

시스템이 현재 단일 객체만 처리하며 인간 얼굴이나 손과 같은 복잡한 구조에서는 어려움을 겪어, 애니메이션용 캐릭터 모델링 등 즉각적인 실용적 응용에 한계가 있습니다.

논문에서 계산 요구사항에 대한 논의가 부족하며, 자동회귀 생성 방식은 본질적으로 순차적이어서 실시간 응용에 속도 제약이 생길 수 있습니다.

---

## 📌 3. 모델의 일반화 성능 향상 가능성

### 3.1 토큰화 개선을 통한 일반화

새로운 토큰화 알고리즘은 핵심 기하학적 디테일을 잃지 않으면서 토큰 시퀀스를 **72% 압축**하는데, 이는 학습 중 계산 부하를 줄일 뿐만 아니라 **학습 안정성을 향상**시킵니다.

또한 특화된 데이터 큐레이션 전략을 통해 특정 기하학적 및 미적 기준을 충족하지 못하는 메쉬를 필터링하는데, 이는 저품질 메쉬로 인한 손실 스파이크를 완화하여 **훨씬 안정적인 학습 프로세스**를 이끌어냅니다.

### 3.2 대규모 학습 데이터와 Perceiver Encoder를 통한 일반화

포인트 클라우드 조건부 생성을 위해 Michelangelo 기반의 공동 학습된 **Perceiver Encoder**를 채택하며, 포인트 클라우드 특성이 크로스-어텐션으로 통합됩니다. 이 구조는 다양한 형태의 3D 입력에 대한 일반화 가능성을 높입니다.

토큰화 알고리즘의 약 72% 압축률은 기하학적 디테일을 손실 없이 고해상도 메쉬 이산화를 가능하게 하여, 복잡하고 다양한 형상을 학습 시 포함할 수 있는 폭이 넓어집니다.

### 3.3 DPO 기반 선호도 정렬의 일반화 기여

학습은 아티스트 제작 메쉬로부터의 **모방 학습(Imitation Learning)**, 삼각형 배치 최적화를 위한 **강화학습(RL)**, 그리고 **인간 피드백 파인튜닝**의 세 단계로 이루어지며, 이러한 다단계 학습 구조가 다양한 형상 입력에 대한 견고성을 높입니다.

선호도 기반 파인튜닝은 특정 학습 도메인에 과적합되지 않고, 인간이 선호하는 일반적 기하학 원칙(완전성, 세부 표현, 미적 수준)을 학습하게 하여 **도메인 외 형상에 대한 일반화 가능성을 향상**시킵니다.

---

## 📌 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 모델 | 연도 | 최대 면 수 | 주요 방법 | 특징 |
|------|------|-----------|-----------|------|
| **PolyGen** | 2020 | ~300 | Transformer (vertex/face 분리) | 최초 신경망 기반 메쉬 생성 |
| **MeshGPT** | 2023 | ~800 | VQ-VAE + Auto-regressive Transformer | 면 정렬·압축 선구적 도입 |
| **MeshAnything** | 2024 | ~800 | Autoregressive Transformer | 아티스트 스타일 메쉬 최초 도전 |
| **MeshAnything V2** | 2024 | ~1,600 | AMT(Adjacent Mesh Tokenization) | 인접 면 기반 압축 개선 |
| **EdgeRunner** | 2024 | ~4,000 | Auto-regressive Auto-encoder | 유연한 양방향 순회 허용 |
| **BPT (Meshtron)** | 2024 | ~5,000–16,000 | Block-wise Partitioning + Hourglass | 극히 긴 시퀀스 처리 가능 |
| **DeepMesh** | 2025 | ~30,000 | AR Transformer + DPO (RL) | 최대 면 수, 인간 선호도 정렬 최초 |

기존 방법들은 메쉬 토큰화의 계산 비용으로 인해 1,000개 미만의 면으로 제한되었으며 일반화 능력도 제한적이었습니다. MeshAnything V2는 개선된 토큰화 기법을 도입해 최대 면 수를 1,600개까지 확장했습니다.

이후 MeshAnything V2와 EdgeRunner가 최대 면 수를 각각 1.6k, 4k로 확장했으며, BPT와 TreeMeshGPT가 5k–8k를 지원했습니다. Meshtron은 Hourglass Transformer와 절단 학습(truncation training)으로 최대 16k 면 생성을 가능하게 했으며, **DeepMesh는 DPO를 통한 강화학습을 적용한 최초의 3D 메쉬 생성 모델**입니다.

자동회귀 모델은 대형 언어 모델의 성공에 영감을 받아 메쉬 생성에 적용되기 시작했으나, 긴 토큰 시퀀스, 학습 불안정성, 인간 선호도와의 정렬 부재라는 근본적인 한계를 공통적으로 안고 있습니다.

---

## 📌 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5.1 연구에 미치는 영향

3D 콘텐츠가 다양한 산업에서 점점 더 중요해짐에 따라, DeepMesh는 품질 기준을 유지하면서 제작 프로세스를 가속화하는 귀중한 도구를 제공합니다. 이 연구는 3D 생성 분야를 발전시킬 뿐만 아니라, AI 생성 콘텐츠를 인간의 미적 선호도와 정렬하기 위해 강화학습 기술을 성공적으로 적용했음을 입증합니다.

**구체적 영향:**

1. **3D 메쉬 생성에 RLHF 적용의 선례 마련**: NLP 분야에서 검증된 RLHF/DPO 패러다임을 3D 구조적 생성에 성공적으로 이전함으로써, 이후 3D 생성 모델들의 후처리 정렬 연구에 핵심 참조점이 됩니다.

2. **고해상도 메쉬 생성의 새로운 기준 설정**: 양자화 해상도 512에서 최대 30,000개의 면을 생성하는 능력은 이전 방법 대비 획기적으로 높은 복잡도의 메쉬 생성을 가능하게 합니다.

3. **게임·애니메이션 파이프라인 혁신 가능성**: DeepMesh는 단일 뷰 이미지 및 텍스트 프롬프트 조건 모두에서 동작하여, 고품질 3D 애니메이션 모델 자동 생성 응용으로 이어질 수 있습니다.

---

### 5.2 앞으로 연구 시 고려할 점

| 고려 항목 | 세부 내용 |
|-----------|-----------|
| **복잡한 구조 대응** | 인간 얼굴, 손, 다중 객체 등에 대한 일반화 |
| **추론 속도 개선** | 자동회귀의 순차적 특성을 병렬화 혹은 캐싱으로 가속 |
| **선호도 데이터 확장** | 5,000쌍의 DPO 데이터 규모를 대폭 확장하여 더 다양한 선호도 반영 |
| **텍스처 및 UV 통합** | 현재 순수 기하학에 집중하므로 텍스처 연계 생성 연구 필요 |
| **물리 기반 제약** | 실제 물리 시뮬레이션(FEM 등)에 활용 가능하도록 메쉬 품질 제약 추가 |
| **평가 지표 표준화** | 기하학 품질과 미적 품질을 동시에 측정하는 통합 벤치마크 필요 |
| **Scaling Law 연구** | 파라미터 수와 학습 데이터 확장에 따른 성능 변화 체계적 분석 필요 |
| **다중 모달 조건 통합** | 텍스트·이미지·포인트 클라우드를 통합 조건으로 처리하는 통합 프레임워크 |

자동회귀 생성 방식은 본질적으로 순차적이어서 실시간 응용에 속도 제약이 생길 수 있으며, 이에 대한 병렬화 혹은 증류(distillation) 기반 가속 연구가 필요합니다.

기존 방법들이 출력과 인간 미적 선호도를 정렬하는 데 어려움을 겪어 기하학적 부정확성과 예술적 완성도 부족 문제가 있었다는 점에서, 앞으로의 연구는 더 정교한 선호도 모델과 더 넓은 인간 평가자 풀을 활용한 데이터 수집 방법론 개선이 중요한 과제가 됩니다.

---

## 📚 참고 자료 및 출처

| 번호 | 제목 / 출처 |
|------|-------------|
| 1 | **[주 논문]** Zhao, R. et al., "DeepMesh: Auto-Regressive Artist-mesh Creation with Reinforcement Learning," arXiv:2503.15265, 2025. — https://arxiv.org/abs/2503.15265 |
| 2 | **[공식 프로젝트 페이지]** https://zhaorw02.github.io/DeepMesh/ |
| 3 | **[공식 GitHub]** https://github.com/zhaorw02/DeepMesh |
| 4 | **[Hugging Face Paper Page]** https://huggingface.co/papers/2503.15265 |
| 5 | **[alphaXiv 개요]** https://www.alphaxiv.org/overview/2503.15265v1 |
| 6 | **[AI Models FYI 분석]** https://www.aimodels.fyi/papers/arxiv/deepmesh-auto-regressive-artist-mesh-creation-reinforcement |
| 7 | **[Moonlight 리뷰]** https://www.themoonlight.io/en/review/deepmesh-auto-regressive-artist-mesh-creation-reinforcement-learning |
| 8 | **[HF Daily Paper Review]** https://deep-diver.github.io/ai-paper-reviewer/paper-reviews/2503.15265/ |
| 9 | **[관련 연구]** Tang, J. et al., "EdgeRunner: Auto-regressive Auto-encoder for Artistic Mesh Generation," arXiv:2409.18114, 2024. — https://arxiv.org/html/2409.18114v1 |
| 10 | **[관련 연구]** "Scaling Mesh Generation via Compressive Tokenization," arXiv:2411.07025. — https://arxiv.org/html/2411.07025v1 |
| 11 | **[관련 연구]** Chen, Y. et al., "MeshAnything: Artist-created Mesh Generation with Autoregressive Transformers," arXiv:2406.10163, 2024. |
| 12 | **[관련 연구]** "Auto-Regressive Mesh Generation as Weaving Silk," arXiv:2507.02477. — https://arxiv.org/pdf/2507.02477 |

> ⚠️ **정확도 참고**: 본 답변에서 DPO 목적함수 및 자동회귀 확률 분해 수식은 DeepMesh 논문의 방법론에 기반하여 표준적인 형식으로 작성되었습니다. 논문 원문의 세부 수식 기호가 일부 다를 수 있으므로, 원문 PDF 확인을 권장합니다.
