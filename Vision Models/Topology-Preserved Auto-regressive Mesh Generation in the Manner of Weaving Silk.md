
# Topology-Preserved Auto-regressive Mesh Generation in the Manner of Weaving Silk (Mesh Silksong)

> **논문 정보**
> - **제목(OpenReview/arXiv 최신 버전):** *Topology-Preserved Auto-regressive Mesh Generation in the Manner of Weaving Silk* (별칭: **Mesh Silksong**)
> - **arXiv ID:** [2507.02477](https://arxiv.org/abs/2507.02477)
> - **저자:** Gaochao Song 외 5인
> - **상태:** Under Review (Preprint, 최신판 v3: 2026년 3월 2일 업데이트)
> - **출처:** arXiv, OpenReview (iFPUEBwwuT), ResearchGate, Moonlight Literature Review

---

## 1. 핵심 주장 및 주요 기여 요약

기존의 자기회귀(auto-regressive) 메시 생성 방법들은 효과적인 위상(topology) 보존에 실패하고 있으며, 이는 이전 메시 토큰화 방법들이 메시를 단순히 동등한 삼각형들의 집합으로 취급하여 생성 시 전체적인 위상 구조를 인식하지 못하기 때문이다.

이 논문의 핵심 주장과 주요 기여는 다음과 같다:

본 논문은 **Mesh Silksong**을 소개한다. 이는 비단 짜기(silk weaving)에 유사한 자기회귀 방식으로 폴리곤 메시를 생성하기 위해 설계된 컴팩트하고 효율적인 메시 표현 방법이다. 기존 메시 토큰화 방법들은 항상 반복된 정점 토큰을 포함하는 토큰 시퀀스를 생성하여 네트워크 역량을 낭비하였다. 이에 반해, 본 방법은 각 메시 정점을 오직 한 번만 접근하여 토큰화함으로써 토큰 시퀀스 중복을 50% 줄이고, 약 22%의 최신 압축률을 달성한다. 또한 Mesh Silksong은 다양한 기하학적 특성(매니폴드 위상, 방수성(watertight) 감지, 일관된 면 법선)을 갖춘 폴리곤 메시를 생성하며, 이는 실용적 응용에 매우 중요하다.

**주요 기여 4가지 요약:**

| 기여 항목 | 내용 |
|---|---|
| ① 새로운 메시 토큰화 알고리즘 | 정점 레이어링 및 정렬을 통한 위상 보존 |
| ② 압축 효율 SOTA 달성 | Compression Ratio 0.22 (BPT 대비 개선) |
| ③ 온라인 비-매니폴드 데이터 처리 | 수동 데이터 큐레이션 없이 훈련 데이터 확장 |
| ④ 훈련 재샘플링 전략 | 데이터 분포 불균형 해소 및 일반화 성능 향상 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

트리-순회(tree-traversal) 방법들은 반-엣지(half-edge) 자료 구조를 활용해 매니폴드 위상을 보존하려 하지만, 삼각형 순회 중 오류 누적으로 인해 메시 생성이 불안정하다. 반면 국소 패치(local patch) 방법들은 독립적인 삼각형 패치를 처리하여 더 안정적인 메시를 생성하지만, 이러한 메시는 종종 비-매니폴드(non-manifold)로, UV 언래핑, 3D 프린팅, 물리 시뮬레이션 같은 응용에는 부적합하다.

또한 기존 방법들은 **글로벌 구조 인식 부재**의 문제가 있어서, 생성 중 모든 삼각형이 동등하게 취급되어 눈처럼 작지만 중요한 연결 컴포넌트를 간과하게 된다.

구체적으로, 기존 방법의 두 가지 주요 한계가 있다:

1. **중복 정점 압축(Redundant Vertex Compression):** 비-경계 정점을 따라 순회 시 시작 정점이 두 번 인코딩되는 중복이 발생한다. 이러한 중복은 인접 삼각형 누락을 피하기 위해 비-경계 정점에서 보편적으로 나타나며, BPT, TreeMeshGPT, MeshAnythingv2 등에서도 유사한 문제가 관찰된다.

2. **글로벌 구조 인식 부재 및 비-매니폴드 토폴로지**

---

### 2.2 제안하는 방법 (알고리즘 + 수식 포함)

#### 📌 알고리즘 파이프라인 (4단계)

메시 토큰화 알고리즘은 다음 4단계로 구성된다:
(1) **전처리(Preprocessing):** 입력 메시가 매니폴드 위상 요건을 엄격히 따르도록 경량의 비-매니폴드 엣지 처리 알고리즘을 개발한다.
(2) **정점 레이어링 및 정렬(Vertex Layering and Sorting):** 주어진 반-엣지에서 시작하여, 모든 정점을 등고선 유사 여러 레이어로 분류하고 로컬 순서 기반으로 정렬한다.
(3) **레이어 인접 행렬 압축(Layer Adjacency Matrices Compression):** 각 레이어에 대해 2-레이어 인접 행렬을 계산한 후 토큰으로 압축한다.
(4) **토큰 패킹(Token Packing):** 정점 토큰과 인접 행렬에서 압축된 위상 토큰을 자기회귀 생성에 최적화된 통합 포맷으로 정리한다.

#### 📌 토큰 시퀀스 구조

레이어 $L$의 각 정점 $V_i^L$에 대해 세 가지 유형의 토큰이 생성된다: 정점 토큰 $V(L,i)$ (BPT를 따라 양자화된 $x$ - $y$ - $z$를 위한 2개의 서브토큰), 자기-레이어 위상 토큰 $S(L,i)$, 그리고 레이어 간 위상 토큰 $B(L,i)$이다. 시퀀스는 각 연결 컴포넌트에 대한 제어 토큰 'C'로 시작한다. 하나의 레이어 내 정점들의 토큰은 레이어 순서 $(V(L,i), S(L,i), B(L,i))$에 따라 순차적으로 패킹된다. 레이어들은 "업레이어" 토큰 'U'로 구분되며, 연결 컴포넌트는 'E' 토큰으로 종료된다.

한 정점 당 토큰 구성:

$$\text{Token}(V_i^L) = \left[ V(L,i),\ S(L,i),\ B(L,i) \right]$$

전체 토큰 시퀀스 구조:

$$\mathbf{T} = \left[ C,\ \{V(1,i), S(1,i), B(1,i)\}_{i=1}^{n_1},\ U,\ \{V(2,i), S(2,i), B(2,i)\}_{i=1}^{n_2},\ U,\ \ldots,\ E \right]$$

#### 📌 손실 함수

자기회귀 모델 학습을 위해 표준 교차 엔트로피 손실 함수를 사용하며, 이는 예측된 토큰 로짓과 실제 토큰 시퀀스 간의 차이를 최소화한다.

$$\mathcal{L} = -\sum_{t} \log P_\theta(S_{t+1} \mid S_1, S_2, \ldots, S_t)$$

여기서 $S_{t+1}$은 다음 타임스텝의 원-핫 인코딩된 실제 토큰이며, $P_\theta$는 파라미터 $\theta$로 모델링된 조건부 확률이다.

#### 📌 압축률

기존 방법들과 비교하여, 본 방법은 각 정점을 한 번만 압축하여 약 50%의 중복 정점 정보를 줄인다. BPT가 제안한 블록 단위 인덱싱(정점 당 3 토큰 → 2 토큰)과 결합하면, 최종 압축률이 BPT의 0.26에서 0.22로 개선되어, 더 효율적인 메시 압축 및 동일한 토큰 길이 내에서 더 세밀한 기하학적 세부 표현이 가능해진다.

$$\text{Compression Ratio} = \frac{\text{Token Sequence Length}}{\text{Number of Faces}} \approx 0.22$$

#### 📌 매니폴드 위상 보장 원리

알고리즘은 생성 과정에서 엣지 연결이 비단 짜기처럼 동일 레이어 또는 인접 레이어의 정점들 사이에서만 존재하도록 자연스럽게 강제한다. 디코딩 과정에서 삼각형은 같은 방식으로 레이어별로 동적으로 채워지며, 이를 통해 메시가 레이어별 구성 내내 매니폴드 위상을 엄격히 유지한다.

---

### 2.3 모델 구조

모델은 표준 교차 엔트로피 손실로 자기회귀적으로 훈련된 **디코더 전용 트랜스포머(decoder-only transformer)**이다. 포인트 클라우드 조건부 생성의 경우, Michelangelo로부터의 특징이 **크로스-어텐션(cross-attention)**을 통해 주입된다.

모델 구성 요약:

| 구성 요소 | 내용 |
|---|---|
| **백본** | Decoder-only Transformer |
| **조건화 방식** | Point Cloud → Cross-Attention (Michelangelo encoder) |
| **토큰 어휘 크기** | 최대 10,267 (위상 토큰 포함) |
| **훈련 데이터셋** | gObjaverse, ShapeNetV2, 3D-FUTURE, Toys4K 혼합 |
| **훈련 하드웨어** | H800, 약 15일 소요 |
| **옵티마이저** | AdamW |

---

### 2.4 성능 향상

정량적 지표로는 Chamfer Distance ($\text{CD} \downarrow$), Hausdorff Distance ($\text{HD} \downarrow$), Normal Consistency ($\text{NC} \uparrow$), 절댓값 Normal Consistency ($|\text{NC}| \uparrow$), Face Number Ratio ($\text{FR} \uparrow$)를 사용하며, EdgeRunner*, TreeMeshGPT, BPT와 같은 기준 모델 대비 우수한 기하학적 정확도와 면 생성 능력을 보인다.

정성적으로, 생성된 메시들은 강인한 구조, 더 많은 세부 사항, 엄격한 매니폴드 위상을 보이는 반면, BPT와 DeepMesh 같은 국소 패치 방법들은 비-매니폴드 결과를 생성한다. 압축률은 약 0.22로 검증되어 BPT의 0.26 및 EdgeRunner의 0.47보다 현저히 낮으며, TreeMeshGPT의 0.22와 동등하면서도 다른 지표에서는 더 나은 성능을 보인다.

**정량적 비교 요약 (논문 Table 1 기준):**

| 방법 | CD ↓ | HD ↓ | NC ↑ | \|NC\| ↑ | FR ↑ | Comp. Ratio ↓ |
|---|---|---|---|---|---|---|
| EdgeRunner* | 0.140 | 0.296 | 0.322 | 0.586 | 1.222 | 0.47 |
| TreeMeshGPT | 0.083 | 0.165 | 0.483 | 0.629 | 1.300 | 0.22 |
| **Mesh Silksong (Ours)** | **최고** | **최고** | **최고** | **최고** | **최고** | **0.22** |

---

### 2.5 한계

한계로는 위상 토큰으로 인한 더 큰 어휘 크기(최대 10,267)와, 레이어 당 최대 정점 수를 미리 정의해야 한다는 점($m=200$)이 있으며, 이는 데이터셋 필터링을 필요로 하고 일반화 성능에 영향을 줄 수 있다.

또한 미래 연구 방향으로 사각형(quadrilateral)과 같은 다른 폴리곤 유형에 대한 압축 방법 탐색이 필요하다.

---

## 3. 모델의 일반화 성능 향상 가능성

일반화 성능에 관련된 내용은 크게 세 가지 측면으로 분석된다.

### 3.1 온라인 비-매니폴드 데이터 처리 알고리즘

논문은 훈련 데이터셋의 규모를 확장하고 비용이 많이 드는 수동 데이터 큐레이션을 피하기 위해, **온라인 비-매니폴드 데이터 처리 알고리즘**과 **훈련 재샘플링 전략**을 도입한다.

이는 인터넷에서 수집된 노이즈가 많은 메시 데이터도 훈련에 활용 가능하게 하므로, 더 다양하고 광범위한 데이터로부터 학습할 수 있게 된다.

### 3.2 연결 컴포넌트 인식(Connected Component Awareness)

본 방법은 알고리즘이 연결 컴포넌트별로 동작하고 특수 토큰을 활용하여 마킹하기 때문에, 메시의 연결 컴포넌트에 민감하다. 이는 생성 모델이 전체적인 기하학적 위상에 집중하도록 유도하여 작고 종종 간과되는 연결 컴포넌트를 효과적으로 포착한다. 매니폴드 기반 메시 표현으로서, 방수성 감지 및 표면 법선 일관성에 대한 강력한 제약을 지원한다.

### 3.3 점진적 균형 재샘플링 전략(Progressively-Balanced Sampling)

훈련 중 **점진적 균형 샘플링 전략(progressively-balanced sampling strategy)**을 사용하여 데이터셋의 메시 크기의 롱테일 분포를 처리한다.

이를 수식으로 표현하면, 학습 배치에서 면 수 $f$를 가진 메시의 샘플링 확률 $p(f)$를:

$$p(f) \propto \frac{1}{\sqrt{N(f)}}$$

의 형태로 조정하여 희귀한 크기(대형 메시 등)의 과소표현 문제를 완화한다. *(이 수식은 일반적인 균형 샘플링 형식으로, 논문에서 정확한 형식을 직접 명시하지 않음을 유의)*

### 3.4 다양한 데이터셋 학습

gObjaverse, ShapeNetV2, 3D-FUTURE, Toys4K 등 다양한 데이터셋에 대한 실험이 Mesh Silksong의 효과를 입증한다.

이처럼 다양한 도메인의 데이터셋에서 훈련됨으로써 모델의 분포 외(out-of-distribution) 일반화 능력이 향상된다.

### 3.5 일반화의 잠재적 한계와 개선 방향

레이어 당 최대 정점 수($m=200$)를 미리 정의해야 하며, 이로 인한 데이터셋 필터링이 일반화에 영향을 줄 수 있다.

이를 극복하기 위한 잠재적 방향:
- **동적 레이어 크기 적응:** $m$을 고정하지 않고, 메시 복잡도에 따라 동적으로 조절
- **더 큰 어휘 처리:** 계층적 어휘 구조(hierarchical vocabulary)를 통한 메모리 효율 개선
- **텍스트/이미지 조건부 생성 확장:** 멀티모달 조건화로 다양한 생성 시나리오 지원

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구 영향

**① 위상 보존 메시 생성의 새로운 기준 제시**

알고리즘은 생성 과정에서 엣지 연결이 비단 짜기처럼 동일 레이어 또는 인접 레이어의 정점 사이에만 존재하도록 강제하며, 디코딩 과정에서 삼각형이 레이어별로 동적으로 채워져 매니폴드 위상이 철저히 유지된다. 이러한 매니폴드 위상의 엄격한 보존은 메시 디코딩 과정에서 적용되는 구성적 제약에서 근본적으로 비롯된다.

이는 UV 언래핑, 3D 프린팅, 물리 시뮬레이션 등 실용적 응용에서 즉시 활용 가능한 메시 생성의 새로운 방향을 제시한다.

**② 확장 가능한 데이터 파이프라인 구축**

수동 큐레이션 없이 대규모 비-매니폴드 데이터도 훈련에 활용할 수 있는 온라인 처리 알고리즘은, 향후 더 대규모의 모델 학습을 위한 데이터 파이프라인 구축에 직접적인 영향을 준다.

**③ 선행 연구 계보와의 연결**

MeshGPT는 VQ-VAE를 사용해 메시를 토큰으로 인코딩하는 방법을 선도하였고, MeshXL은 훈련 가능한 VQ-VAE 없이 삼각형 정점 토큰 양자화가 데이터셋 확장의 핵심임을 밝혔다. 이후 MeshAnything v2와 EdgeRunner는 컴팩트한 토큰화 알고리즘을 도입하여 최대 면 수를 800에서 1.6k 및 4k로 늘렸다. BPT와 TreeMeshGPT는 기하 압축과 정점 압축을 모두 최적화하여 5k~8k 면을 지원하며, Meshtron은 아워글래스 트랜스포머와 절단 학습으로 최대 16k 면 생성을 가능하게 했다. LLaMAMesh는 사전학습된 LLM을 텍스트-메시 생성에 활용하였고, DeepMesh는 강화학습(DPO)을 적용해 메시 미학을 개선하였다.

Mesh Silksong은 이 계보에서 **위상 보존 + 최고 압축률**이라는 두 목표를 동시에 달성하는 방향으로 연구 계보를 확장한다.

---

### 4.2 향후 연구 시 고려할 점

| 고려 항목 | 내용 |
|---|---|
| **어휘 크기 최적화** | 위상 토큰으로 인한 어휘 크기 증가(최대 10,267)는 메모리 및 추론 속도에 부담. 어휘 압축 또는 계층 구조 필요 |
| **레이어 크기 동적 처리** | $m=200$ 고정 제약 완화 → 다양한 복잡도의 메시 처리 가능 |
| **쿼드메시/혼합 폴리곤 확장** | 삼각형 이외의 폴리곤 유형(사각형 등)으로의 확장 가능성 탐색 |
| **멀티모달 조건화** | 텍스트, 이미지, 스케치 등 다양한 조건으로의 조건부 생성 확장 |
| **추론 속도 개선** | 자기회귀 특성상 긴 시퀀스에서 느린 추론 → 추측 디코딩(speculative decoding) 등 가속 기법 연구 필요 |
| **오픈 세계 일반화** | 학습 데이터에 없는 객체 범주에 대한 제로샷/퓨샷 일반화 성능 검증 |
| **강화학습과의 결합** | DeepMesh처럼 DPO 기반 사람 선호도 정렬을 Mesh Silksong에 결합하여 미적 품질 향상 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

주요 자기회귀 메시 생성 방법들의 발전 흐름을 정리하면, MeshGPT(2024)는 VQ-VAE + 디코더 전용 트랜스포머로 최초 접근하였고, MeshXL(2024)은 훈련 가능한 VQ-VAE 없이 삼각형 정점 토큰을 직접 양자화하는 방식이 데이터셋 확장의 핵심임을 확인하였다. MeshAnything v2와 EdgeRunner는 컴팩트한 토큰화 알고리즘을 도입하여 최대 면 수를 800에서 1.6k 및 4k로 늘렸으며, BPT와 TreeMeshGPT는 기하 및 정점 압축을 최적화하여 5k~8k 면을 지원한다.

| 방법 | 연도 | 핵심 특징 | 위상 보존 | 압축률 |
|---|---|---|---|---|
| **MeshGPT** | 2024 | VQ-VAE + AR Transformer | △ (불안정) | 낮음 |
| **MeshXL** | 2024 | VQ-VAE 없는 정점 양자화 | △ | 중간 |
| **MeshAnythingV2** | 2024 | Adjacent Mesh Tokenization | △ | 중간 |
| **EdgeRunner** | 2025 | EdgeBreaker 기반 | ○ (트리-순회) | 0.47 |
| **BPT** | 2025 | 패치 기반, 로컬 패치 | △ (비-매니폴드) | 0.26 |
| **TreeMeshGPT** | 2025 | 트리 시퀀싱 | ○ | 0.22 |
| **DeepMesh** | 2025 | DPO + 강화학습 | △ (비-매니폴드) | 낮음 |
| **Mesh Silksong (Ours)** | 2025 | 레이어 기반 + 위상 보존 | ✅ (매니폴드 완전 보장) | **0.22** |

트리-순회 방법인 EdgeRunner와 TreeMeshGPT는 생성 메시에 대한 매니폴드 위상이 보장되지만, 본 방법은 더 강인한 생성 능력과 더 많은 세부 사항을 생성한다. 반면 로컬 패치 방법인 BPT와 DeepMesh와는 시각적으로 비견할만한 결과를 달성하지만, 이들 방법은 매니폴드 위상의 메시를 생성할 수 없으며 작은 컴포넌트도 생성하지 못해 실용적 응용을 방해한다. BPT와 DeepMesh에서는 비-매니폴드 엣지가 빨간색으로 표시된다.

---

## 📚 참고 자료 및 출처

1. **Gaochao Song et al.**, "Topology-Preserved Auto-regressive Mesh Generation in the Manner of Weaving Silk (Mesh Silksong)", arXiv:2507.02477, 2025. https://arxiv.org/abs/2507.02477
2. **OpenReview 버전:** https://openreview.net/pdf?id=iFPUEBwwuT
3. **ResearchGate:** https://www.researchgate.net/publication/393379406
4. **Moonlight Literature Review:** https://www.themoonlight.io/en/review/mesh-silksong-auto-regressive-mesh-generation-as-weaving-silk
5. **Siddiqui et al.** (2024), "MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers", CVPR 2024.
6. **Chen et al.** (2024), "MeshXL: Neural Coordinate Field for Generative 3D Foundation Models", NeurIPS 2024.
7. **Weng et al.** (2025), "BPT: Binary Partition Tree for 3D Mesh Generation", 2025.
8. **Lionar et al.** (2025), "TreeMeshGPT: Artistic Mesh Generation with Autoregressive Tree Sequencing", arXiv:2503.11629, 2025.
9. **Zhao et al.** (2025), "DeepMesh: Auto-Regressive Artist-mesh Creation with Reinforcement Learning", ICCV 2025. https://arxiv.org/abs/2503.15265
10. **Tang et al.** (2025), "EdgeRunner: Auto-Regressive Auto-Encoder for Artistic Mesh Generation", ICLR 2025.
11. **XSpecMesh** (2025), "Quality-Preserving Auto-Regressive Mesh Generation Acceleration via Multi-Head Speculative Decoding", arXiv:2507.23777.
12. **Mesh RAG** (2025), "Retrieval Augmentation for Autoregressive Mesh Generation", arXiv:2511.16807.
