
# Meshtron: High-Fidelity, Artist-Like 3D Mesh Generation at Scale

> **논문 정보**
> - **제목:** Meshtron: High-Fidelity, Artist-Like 3D Mesh Generation at Scale
> - **저자:** Zekun Hao, David W. Romero, Tsung-Yi Lin, Ming-Yu Liu (NVIDIA)
> - **arXiv:** [2412.09548](https://arxiv.org/abs/2412.09548) (2024년 12월 12일)
> - **제출처:** ICLR 2025 (OpenReview)

---

## 1. 핵심 주장 및 주요 기여 요약

### 🎯 핵심 주장

Mesh는 3D 표면의 근본적인 표현 방식이지만, 고품질 메쉬를 제작하는 것은 3D 모델링에 상당한 시간과 전문성을 요구하는 노동집약적인 작업이다.

정밀한 객체를 정확하게 모델링하려면 종종 $10^4$개 이상의 face가 필요하지만, 기존의 artist-like mesh 생성 시도들은 $1.6\text{K}$개 face와 정점 좌표의 과도한 이산화(discretization)로 제한되어 있었다. 따라서 최대 face 수와 정점 좌표 해상도를 동시에 확장하는 것이 현실적이고 복잡한 3D 객체의 고품질 메쉬를 생성하는 데 매우 중요하다.

이에 Meshtron은 최대 $64\text{K}$ face를 $1024$-level 좌표 해상도로 생성할 수 있는 새로운 자기회귀(autoregressive) 메쉬 생성 모델을 제시하며, 이는 현재 최신 기법보다 face 수에서 10배 이상, 좌표 해상도에서 $8\times$ 높은 수준이다.

---

### 🏆 주요 기여 (4가지 핵심 컴포넌트)

Meshtron의 확장성은 네 가지 핵심 구성 요소로 이루어진다: (1) Hourglass 신경망 아키텍처, (2) Truncated Sequence Training, (3) Sliding Window Inference, (4) 메쉬 시퀀스의 순서를 강제하는 Robust Sampling Strategy.

이를 통해 기존 연구 대비 훈련 메모리를 50% 이상 절감하고, $2.5\times$ 빠른 처리량(throughput)과 더 나은 일관성을 달성한다.

---

## 2. 해결 문제 · 제안 방법 · 모델 구조 · 성능 · 한계

### 📌 2-1. 해결하고자 하는 문제

#### (a) 기존 방법의 한계

Meshtron은 확장성 문제의 근본 원인인 Vanilla Transformer의 이차(quadratic) 비용 문제를 해결하는 데 집중한다. 기존 방법들은 전역 자기-어텐션(global self-attention)에 크게 의존하여, 세밀한 객체에 필요한 긴 메쉬 시퀀스를 처리할 때 비용이 지나치게 커진다.

#### (b) Iso-surfacing 방법의 한계

DMTet, FlexiCubes와 같은 iso-surfacing 방법들은 많은 수의 face를 가진 메쉬를 생성할 수 있지만, 지나치게 조밀한 테셀레이션(tessellation), bumpy artifacts, 과도한 스무딩, 불충분한 기하학적 세부 표현이라는 문제를 가지고 있어 artist-created mesh와 현저히 다르다. 반면 Meshtron은 높은 기하학적 디테일과 잘 구조화된 테셀레이션을 갖춘 고품질 토폴로지의 메쉬를 생성한다.

---

### 📐 2-2. 제안하는 방법 (수식 포함)

#### (a) 자기회귀 메쉬 생성 (Autoregressive Mesh Generation)

자기회귀 생성에서, 인과적(causal) 신경망(일반적으로 Transformer)은 조건부 분포를 학습하는 데 사용된다:

$$p(c_i \mid \mathbf{c}_{ < i})$$

여기서 $c_i$는 시퀀스의 $i$번째 토큰(좌표값)이며, $\mathbf{c}_{<i}$는 이전 모든 토큰을 의미한다.

그러나 이차(quadratic) 연산 복잡도와 선형(linear) 메모리 요구량으로 인해, 긴 시퀀스를 처리하는 것은 매우 빠르게 비용이 크게 증가한다.

전체 메쉬 $M$의 생성 확률은 다음과 같이 분해된다:

$$p(M) = \prod_{i=1}^{N} p(c_i \mid \mathbf{c}_{ < i}, \mathbf{z})$$

여기서 $\mathbf{z}$는 포인트 클라우드로부터 인코딩된 전역 조건 벡터(global conditioning vector)이다.

#### (b) 메쉬 시퀀스 표현

메쉬의 face $f_k$는 3개의 정점 $v^{(1)}, v^{(2)}, v^{(3)}$으로 구성되며, 각 정점은 3개의 좌표 토큰 $(x, y, z)$로 이루어진다. 1024-level 양자화(quantization)를 사용하므로, 각 좌표는 다음과 같이 정수화된다:

$$\hat{c} = \left\lfloor \frac{c - c_{\min}}{c_{\max} - c_{\min}} \cdot (Q-1) \right\rceil, \quad Q = 1024$$

하나의 삼각형 face는 총 9개의 토큰으로 표현되며, $32\text{K}$개의 face를 가진 메쉬는 약 300K 토큰에 달하는 매우 긴 시퀀스를 형성한다.

직접 이러한 메쉬 토큰을 생성하는 것은 비용이 크다. $32\text{K}$개 face를 가진 메쉬는 300K 토큰이 필요하며, 일반적인 자기회귀 모델에서는 이를 생성하는 데 극도로 많은 시간과 비용이 소요된다.

---

### 🏗️ 2-3. 모델 구조 (Architecture)

#### (1) Hourglass 신경망 아키텍처

Meshtron은 내부적으로 토큰을 병합해 시퀀스 길이를 줄임으로써 연산 및 메모리를 절약하는 Hourglass Transformer 아키텍처를 채택한다. 구체적으로, 모델은 세 단계로 구성되며 각 단계가 시퀀스 길이를 3배씩 줄인다: 첫 번째 단계는 모든 토큰(좌표 단위)을 처리하고, 두 번째 단계는 정점(vertex) 단위로 처리하여(3배 감소), 세 번째 단계는 face 단위로 처리한다(9배 감소).

$$\text{토큰 수: } N_{\text{coord}} \xrightarrow{\times 1/3} N_{\text{vertex}} \xrightarrow{\times 1/3} N_{\text{face}}$$

Hourglass 네트워크의 각 단계는 메쉬의 좌표-정점-face 의미론(semantics)과 정확하게 일치한다. 이는 모델링 효율을 높일 뿐 아니라, 생성하기 어려운 토큰에 더 많은 연산을 배분하는 장점도 가진다. 삼각형 내에서 첫 번째 정점은 생성하기 쉬운 반면, 마지막 정점은 가장 어렵다.

각 단축(shortened) 단계의 토큰들은 메쉬 시퀀스의 정점 및 face에 정렬되어 메쉬 모델링을 위한 좋은 귀납적 편향(inductive bias)을 제공한다.

이 설계가 소규모에서 검증된 후, Meshtron은 1.1B 파라미터 규모와 최대 64K face 및 1024-level 좌표 양자화로 대규모 데이터셋에서 확장된다.

#### (2) Truncated Sequence Training

적절한 컨디셔닝을 통해, 메쉬 생성은 훈련 중 전체 메쉬 시퀀스에 접근할 필요가 없다. 대신, 잘린(truncated) 메쉬 시퀀스로 훈련하고 추론 시에는 슬라이딩 윈도우 방식으로 완전한 메쉬 시퀀스를 생성할 수 있다. 이는 훈련 중 연산 및 메모리 비용을 크게 줄이고 추론도 가속화한다.

$300\text{K}+$ 토큰을 생성하는 메쉬의 경우, 훈련은 전역 컨텍스트와 함께 랜덤 길이의 시퀀스 세그먼트(예: 8K 토큰)를 사용하여, 메모리를 50% 이상 절감하고 배치 처리량을 증가시킨다.

#### (3) Sliding Window Inference

기존 Transformer 모델은 컨텍스트 길이가 시퀀스 길이에 따라 커지며, 시퀀스가 길어질수록 연산이 이차적으로, 메모리가 선형적으로 증가한다. 이는 훈련과 생성 모두에서 긴 시퀀스로 인한 심각한 속도 저하를 야기한다. 이에 반해 Meshtron은 8,192개 face의 고정 길이 컨텍스트 윈도우를 유지한다.

#### (4) Cross-Attention 기반 조건부 생성

MeshXL, MeshAnything 같은 기존 연구들은 포인트 클라우드 임베딩을 메쉬 시퀀스의 시작 부분에 붙이는 방식으로 조건부 생성을 수행한다. 그러나 Meshtron의 확장 전략이 절단된 메쉬 시퀀스 훈련을 포함하기 때문에, 이 방식은 조건부 신호가 일부 메쉬 세그먼트에만 전달되거나 복잡한 연결 전략이 필요하다. 이를 극복하기 위해 Meshtron은 **cross-attention**을 사용하여, 시퀀스 내 위치와 무관하게 모든 메쉬 세그먼트를 전역 조건부 신호에 연결한다. 이를 통해 훈련과 추론 모두에서 로컬 및 전역 정보를 효과적으로 결합하여 낮은 리소스 사용으로 정확한 예측이 가능하다.

#### (5) Robust Sampling Strategy

이후 좌표가 메쉬 시퀀스에서 사전에 정의된 순서를 반드시 따르도록 하는 robust sampling strategy를 도입한다. 이는 생성된 메쉬 시퀀스가 현실적인 구조를 유지하도록 보장하여, 더 일관적이고 신뢰할 수 있는 메쉬 생성을 가능하게 한다.

#### (6) 전체 모델 구성

Meshtron은 Hourglass 아키텍처와 슬라이딩 윈도우 어텐션을 기반으로 하는 자기회귀 메쉬 생성기이다.

인코더는 소규모 모델에서는 8-layer Transformer를, 풀스케일 실험에서는 12-layer Transformer를 사용한다. 입력 포인트 클라우드는 소규모 실험에서 8,192개, 풀스케일 실험에서 16,384개의 포인트를 사용한다.

Meshtron은 포인트 클라우드, face 수, 쿼드(quad) 비율, 창의성 수준과 같은 입력을 받아 독립적인 리메셔(remesher)로 사용하거나 text-to-3D 또는 image-to-3D 모델과 결합하여 사용할 수 있는 높은 제어 가능성을 갖춘다.

---

### 📊 2-4. 성능 향상

| 지표 | Meshtron | 기존 최신 기법 |
|---|---|---|
| 최대 Face 수 | **64K** | 1.6K (MeshAnythingV2) |
| 좌표 해상도 | **1024-level** | 128-level |
| 훈련 메모리 절감 | **50% 이상** | - |
| 처리량(Throughput) | **2.5× 빠름** | - |

Meshtron은 최대 64K 삼각형 face와 1024-level 양자화를 달성하며, 정량적 지표 및 정성적(시각적, 전문가 평가) 표면 세부 표현에서 선행 연구(MeshGPT ≤800 face, MeshAnythingV2 ≤1600 face)를 크게 능가한다.

크게 확장된 컨텍스트 길이 덕분에 Meshtron은 MeshAnythingV2보다 복잡한 형태를 현저히 더 잘 처리한다. 또한 정점 좌표 양자화에서 $8\times$ 높은 해상도는 더 부드러운 메쉬를 생성하게 한다.

---

### ⚠️ 2-5. 한계

Meshtron이 제공하는 발전에도 불구하고 여러 개선 영역이 남아 있다. 첫째, Meshtron의 크게 향상된 효율성에도 불구하고 대형 메쉬 생성에는 상당한 시간이 필요하며, 가장 큰 모델은 140 토큰/초의 속도로 추론한다. 효율적인 자기회귀 모델, speculative decoding, 고급 추론 시스템 등의 발전이 이 과정을 가속화하는 데 도움이 될 수 있다.

둘째, Meshtron은 인상적인 생성 능력을 보이지만 포인트 클라우드 조건화의 저수준(low-level) 특성에 의해 제한된다. 그 결과, marching cube text-to-3D 생성기의 열화된(degraded) 3D 형상에 상당한 디테일을 추가하는 데 어려움을 겪는다. 텍스트와 같은 고수준(higher-level) 조건화 신호나 고해상도 법선 맵과 같은 추가 가이던스를 통합하면 능력을 더욱 향상시킬 수 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 🔬 3-1. 분포 외(Out-of-Distribution) 데이터에서의 일반화

Meshtron은 artist-created, 3D-scanned, 온라인 text-to-3D 서비스로 생성된 세 가지 다른 출처의 메쉬를 기반으로 리메싱 도구로 평가된다.

크게 확장된 컨텍스트 길이 덕분에 Meshtron은 MeshAnythingV2보다 복잡한 형태를 현저히 더 잘 처리한다. 정점 좌표 양자화에서 $8\times$ 높은 해상도는 더 부드러운 메쉬를 이끌어낸다. 분포 외의 non-artist 메쉬에서 포인트 클라우드를 컨디셔닝할 때, Meshtron은 입력 형태를 충실하게 재현한다.

### 🔬 3-2. 다양한 데이터로 훈련 시 일반화 향상

8K face까지의 데이터로 훈련된 모델을 분석한 결과, 목표 시퀀스 길이를 초과하는 경우를 포함한 추가 훈련 데이터가 더 짧은 시퀀스의 성능도 향상시키는 것이 확인됐다. 이 결과는 잘린 시퀀스로의 훈련이 더 짧은 시퀀스를 생성하는 것이 목표일 때도 성능 향상을 위한 실행 가능한 전략임을 시사한다.

> 💡 **의미:** 데이터 다양성 증가(더 긴 시퀀스, 더 다양한 카테고리)가 전반적인 메쉬 품질 향상에 직결됨을 보여준다. 이는 대규모 3D 데이터셋 구축이 일반화 성능 향상의 핵심 요인임을 시사한다.

### 🔬 3-3. Cross-Attention 기반 조건화의 확장성

Meshtron의 제어 입력은 cross-attention으로 구현되며, 이는 높은 확장성을 가지고 이미지, 추가 제어 신호 등 다른 유형의 입력에도 쉽게 적용될 수 있다.

거의 모든 3D 표현이 포인트 클라우드로 변환될 수 있으므로, Meshtron은 독립적인 리메셔로 사용되어 기존 메쉬의 품질을 향상시키거나, text-to-3D 또는 image-to-3D 모델과 함께 사용되어 처음부터 artist-grade 메쉬를 생성할 수 있다.

### 🔬 3-4. Hourglass 구조의 귀납적 편향

이전 연구들은 Hourglass 아키텍처가 메쉬 생성에 더 나은 귀납적 편향(inductive bias)을 제공하며, 적절한 전역 조건화 및 시퀀스 순서 정렬로 메쉬 생성이 성능 저하 없이 로컬 정보에 효과적으로 의존할 수 있음을 확인하였다.

> 💡 **의미:** 이러한 구조적 귀납적 편향은 훈련 데이터에 없는 새로운 형태의 객체에 대해서도 안정적인 생성을 가능하게 하는 핵심 요소다.

---

## 4. 최신 관련 연구 비교 분석 (2020년 이후)

| 모델 | 연도 | 방법 | 최대 Face 수 | 특징 |
|---|---|---|---|---|
| **PolyGen** | 2020 | Autoregressive (Transformer) | ~300 | 최초 Transformer 기반 메쉬 생성 |
| **MeshDiffusion** | 2022 | Score-based Diffusion | 제한적 | 확률적 생성, 형태 다양성 우수 |
| **MeshGPT** | 2023 | Autoregressive + VQ-VAE | **800** | 기하학 어휘(triangle vocabulary) 학습 |
| **PolyDiff** | 2023 | Diffusion (triangle soup) | 제한적 | 확산 모델 기반 폴리곤 메쉬 |
| **MeshAnything V1** | 2024 | Autoregressive + VQ-VAE | **800** | Shape-conditioned, point cloud 입력 |
| **MeshAnythingV2** | 2024 | Autoregressive + AMT | **1,600** | Adjacent Mesh Tokenization |
| **BPT** | 2024 | Block-wise + Patch | **8,000** | 좌표 공간 블록 분할, 시퀀스 74% 압축 |
| **Meshtron** | 2024 | Autoregressive + Hourglass | **64,000** | 1024-level 해상도, 슬라이딩 윈도우 |
| **FlashMesh** | 2025 | Speculative Decoding (Meshtron 기반) | 10,000+ | Meshtron 위에서 병렬 토큰 예측 |

MeshGPT는 잔차 벡터 양자화(residual vector quantization)를 사용해 메쉬 그래프에서 기하학적 "삼각형 어휘"를 학습함으로써 훨씬 짧은 토큰 시퀀스와 직접적인 삼각형 디코딩을 달성한다.

MeshAnything은 MeshGPT에 포인트 클라우드 인코더를 포함하여 조건부 생성 능력을 추가하며, MeshAnythingV2는 VQ-VAE를 더 효율적인 무손실 메쉬 압축 알고리즘으로 대체하여 최대 1.6K face까지 지원을 확장한다.

FlashMesh는 효율적이고 고품질의 합성을 위한 predict-correct-verify 패러다임을 기반으로 하는 새로운 자기회귀 메쉬 생성 프레임워크로, Hourglass Transformer에 맞춤화된 계층적 투기적 디코딩(speculative decoding) 전략을 제안한다.

PolyGen은 두 개의 자기회귀 Transformer 모델로 정점 좌표와 face 인덱스를 별도로 생성하고, MeshXL은 명시적 정점 좌표 시퀀스를 직접 자기회귀적으로 처리하며, MeshGPT는 기하학 구조를 학습된 어휘로 토큰화하고 자기회귀 모델로 디코딩한다. PolyDiff는 메쉬를 양자화된 삼각형 수프(triangle soup)로 표현하고 확산 모델로 노이즈를 제거한다.

---

## 5. 미래 연구에의 영향 및 고려 사항

### 🚀 5-1. 미래 연구에 미치는 영향

**① 3D 콘텐츠 파이프라인의 혁신**

Meshtron은 전례 없는 수준의 해상도와 충실도로 복잡한 3D 객체의 메쉬를 생성하며, 전문 아티스트가 제작한 것과 유사한 메쉬를 만들어 애니메이션, 게이밍, 가상 환경을 위한 더 현실적인 3D 에셋 생성의 문을 열고 있다.

**② 리메싱(Remeshing) 패러다임 전환**

3D 스캐닝이나 text-to-3D 도구로 생성된 메쉬는 보통 매우 조밀하고 품질이 낮은 토폴로지를 가지는데, Meshtron은 이러한 메쉬의 테셀레이션(tessellation)을 개선하는 리메싱 도구로 사용될 수 있다.

**③ 후속 연구 촉발**

Meshtron은 Hourglass Transformer와 절단(truncation) 훈련을 제안하여 최대 16K face 생성을 가능하게 하였고, LLaMAMesh는 text-to-mesh 생성을 위해 사전 훈련된 LLM을 활용하는 방향으로, DeepMesh는 DPO를 통한 강화 학습을 적용하는 방향으로 후속 연구가 진행되고 있다.

### 🔍 5-2. 앞으로 연구 시 고려할 점

**① 고수준 조건화 신호 통합**

텍스트와 같은 고수준 조건화 신호나 고해상도 표면 법선 맵과 같은 추가 가이던스를 통합하면 능력을 더욱 향상시킬 수 있다.

**② 추론 속도 최적화**

효율적인 자기회귀 모델(Mamba 등), speculative decoding, 고급 추론 시스템이 대형 메쉬 생성의 속도를 가속화하는 데 도움이 될 수 있다.

**③ 메쉬 토큰화 효율화**

기존 방법들 대비 중복된 정점 압축을 줄여 약 50%의 중복 정점 정보를 압축하는 방향, 그리고 BPT의 블록 단위 인덱싱과 결합하여 최종 압축 비율을 개선하는 방향이 더 효율적인 메쉬 표현을 가능하게 한다.

**④ 일반화를 위한 대규모 다양한 데이터셋 구축**

훈련 데이터가 더 긴 시퀀스를 포함하면 더 짧은 시퀀스 성능도 향상됨을 고려할 때, 다양한 카테고리와 복잡도를 포함한 대규모 3D 데이터셋 구축이 일반화의 핵심이다.

**⑤ 메쉬 편집 및 부분 생성으로의 확장**

MeshAnything, MeshAnythingV2, EdgeRunner, Meshtron 등의 후속 연구들은 토큰화, 어텐션 메커니즘, 형태 범위에서 각각의 개선을 제안하며 직접 메쉬 생성의 한계를 밀어붙이고 있다. PolyDiff는 다른 방향으로 폴리곤 메쉬 합성에 확산 백본을 사용한다. 상당한 진전에도 불구하고 이 방법들은 전체 형태 생성에 집중하여 로컬 편집 작업을 쉽게 다루지 못한다.

---

## 📚 참고 자료 (출처)

| 번호 | 제목 / 출처 |
|---|---|
| 1 | Hao, Z., Romero, D. W., Lin, T.-Y., & Liu, M.-Y. (2024). *Meshtron: High-Fidelity, Artist-Like 3D Mesh Generation at Scale.* arXiv:2412.09548 — [arxiv.org/abs/2412.09548](https://arxiv.org/abs/2412.09548) |
| 2 | NVIDIA Technical Blog. *High-Fidelity 3D Mesh Generation at Scale with Meshtron* — [developer.nvidia.com/blog](https://developer.nvidia.com/blog/high-fidelity-3d-mesh-generation-at-scale-with-meshtron/) |
| 3 | OpenReview (ICLR 2025 Submission). *Meshtron: High-Fidelity, Artist-Like 3D Mesh Generation at Scale* — [openreview.net](https://openreview.net/forum?id=mhzDv7UAMu) |
| 4 | HuggingFace Papers. *Paper page - Meshtron* — [huggingface.co/papers/2412.09548](https://huggingface.co/papers/2412.09548) |
| 5 | Chen, Y. et al. (2024). *MeshAnything: Artist-Created Mesh Generation with Autoregressive Transformers.* arXiv:2406.10163 |
| 6 | Chen, Y. et al. (2024). *MeshAnything V2: Artist-Created Mesh Generation with Adjacent Mesh Tokenization.* arXiv:2408.02555 |
| 7 | Siddiqui, Y. et al. (2023/2024). *MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers.* CVPR 2024. arXiv:2311.15475 |
| 8 | Alliegro, A. et al. (2023). *PolyDiff: Generating 3D Polygonal Meshes with Diffusion Models.* arXiv:2312.11417 |
| 9 | FlashMesh (2025). *FlashMesh: Faster and Better Autoregressive Mesh Synthesis via Structured Speculation.* arXiv:2511.15618 — [arxiv.org/html/2511.15618](https://arxiv.org/html/2511.15618) |
| 10 | AI Models FYI. *Meshtron Paper Details* — [aimodels.fyi](https://www.aimodels.fyi/papers/arxiv/meshtron-high-fidelity-artist-like-3d-mesh) |
| 11 | Emergent Mind. *MeshGPT: Transformer-Based Mesh Generation* — [emergentmind.com](https://www.emergentmind.com/topics/meshgpt) |
| 12 | NVIDIA Research Labs. *Meshtron Project Page* — [research.nvidia.com/labs/cosmos-lab/meshtron](https://research.nvidia.com/labs/cosmos-lab/meshtron/) |
