
# OctGPT: Octree-based Multiscale Autoregressive Models for 3D Shape Generation

> **논문 정보**
> - **제목**: OctGPT: Octree-based Multiscale Autoregressive Models for 3D Shape Generation
> - **저자**: Si-Tong Wei, Rui-Huan Wang, Chuan-Zhi Zhou, Baoquan Chen, Peng-Shuai Wang (Peking University)
> - **출처**: arXiv:2504.09975 (2025년 4월), SIGGRAPH 2025 Technical Paper
> - **DOI**: https://doi.org/10.1145/3721238.3730601
> - **코드**: https://github.com/octree-nn/octgpt

---

## 1. 핵심 주장 및 주요 기여 요약

자기회귀(Autoregressive) 모델은 다양한 분야에서 놀라운 성공을 거두었지만, 3D 형상 생성에서의 성능은 Diffusion 모델에 비해 현저히 뒤처졌다. OctGPT는 이 격차를 해소하기 위한 새로운 다중 스케일 자기회귀 모델로, 기존 3D 자기회귀 접근법의 효율성과 성능을 극적으로 향상시키며 최첨단 Diffusion 모델과 경쟁하거나 이를 능가한다.

### 주요 기여 (Key Contributions)

| # | 기여 항목 | 설명 |
|---|-----------|------|
| 1 | **직렬화 옥트리 표현** | 3D 형상을 다중 스케일 이진 시퀀스로 변환 |
| 2 | **효율적 트랜스포머** | Octree 기반 어텐션 + 3D RoPE + 병렬 생성 |
| 3 | **속도 향상** | 학습 시간 13배↓, 생성 시간 69배↓ |
| 4 | **다양한 태스크** | 텍스트·스케치·이미지 조건부 생성, 씬 합성 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

자기회귀 모델을 3D 형상 생성에 적용하는 것은 고유한 난제를 안고 있다. 텍스트나 이미지와 달리, 3D 형상에는 자연스러운 순차적 순서가 없어 자기회귀 예측에 적합한 1차원 시퀀스로 변환하는 과정이 필요하다.

또한 3D 형상은 복잡한 기하학과 위상을 포착하기 위해 많은 수의 토큰이 필요하여 학습과 추론 모두 계산 집약적이 된다. 이를 해결하기 위한 선행 연구들은 메시 기반 표현과 저차원 토큰화 방식을 탐색했지만, 토큰 수를 약 1k 수준으로 줄임에도 불구하고 여전히 표현력에 한계가 있고 세밀한 디테일의 고품질 3D 형상 생성에 어려움을 겪는다.

직렬화 옥트리 표현의 토큰 길이는 50k를 초과할 수 있어, 이차 시간 복잡도를 가진 단순 자기회귀 모델에 큰 부담이 된다. 이를 극복하기 위해 OctGPT는 옥트리 기반 트랜스포머를 개선하여 시간 복잡도를 선형으로 줄이고 13배의 속도 향상을 달성한다.

---

### 2.2 제안하는 방법 (수식 포함)

#### (1) 직렬화 옥트리 표현 (Serialized Octree Representation)

OctGPT의 핵심 혁신은 새로운 직렬화 옥트리 표현이다. 옥트리는 3D 형상의 계층적 구조를 자연스럽게 포착하면서, 자기회귀 예측에 적합한 지역성 보존 순서를 제공한다.

3D 형상은 다중 스케일로 직렬화된 옥트리로 인코딩되며, 조대한(coarse) 구조는 옥트리 계층에서 유도된 다중 스케일 이진 분할 신호로 표현되고, 세밀한(fine-grained) 디테일은 옥트리 기반 VQVAE의 이진화된 잠재 코드로 포착된다. 이 이진 토큰들은 teacher-forcing 마스크와 함께 자기회귀 학습을 위한 트랜스포머에 입력된다.

옥트리의 각 노드에서 분할 신호 $s_i$는 다음과 같은 이진 값을 가집니다:

$$s_i \in \{0, 1\}^8, \quad i = 1, 2, \ldots, N$$

여기서 각 비트는 8개의 자식 노드 중 해당 노드가 점유(occupied)인지를 나타냅니다.

전체 3D 형상은 다중 스케일 시퀀스 $\mathbf{T}$로 표현됩니다:

$$\mathbf{T} = \left[ \underbrace{s_1^{(1)}, \ldots, s_{N_1}^{(1)}}_{\text{scale 1 (coarse)}}, \underbrace{s_1^{(2)}, \ldots, s_{N_2}^{(2)}}_{\text{scale 2}}, \ldots, \underbrace{z_1^{(L)}, \ldots, z_{N_L}^{(L)}}_{\text{scale L (fine, VQVAE)}} \right]$$

여기서 $z_k^{(L)}$는 VQVAE 코드북으로부터 양자화된 잠재 코드입니다.

#### (2) VQVAE를 통한 세밀한 디테일 표현

조대한 기하학은 옥트리 구조로 인코딩되고, 세밀한 디테일은 벡터 양자화 변분 오토인코더(VQVAE)를 통해 생성된 이진 토큰으로 표현되며, 3D 형상을 자기회귀 예측에 적합한 컴팩트한 다중 스케일 이진 시퀀스로 변환한다.

VQVAE의 코드북 학습 목표는 다음과 같습니다:

$$\mathcal{L}_{\text{VQVAE}} = \mathcal{L}_{\text{recon}} + \|\text{sg}[z_e(x)] - e\|_2^2 + \beta \|z_e(x) - \text{sg}[e]\|_2^2$$

여기서 $\text{sg}[\cdot]$는 stop-gradient 연산, $z_e(x)$는 인코더 출력, $e$는 코드북 벡터입니다.

#### (3) 자기회귀 예측 목표

자기회귀 모델의 학습 목표(cross-entropy loss)는:

$$\mathcal{L}_{\text{AR}} = -\sum_{t=1}^{T} \log P\left(x_t \mid x_1, x_2, \ldots, x_{t-1}; \theta\right)$$

여기서 $x_t$는 $t$번째 토큰, $\theta$는 트랜스포머 파라미터입니다.

#### (4) 3D Rotary Positional Encoding (RoPE3D)

1D RoPE를 3D 공간으로 확장하고, 각 토큰에 대한 스케일별 위치 인코딩을 도입하여 직렬화된 옥트리 표현에서 모델이 다중 스케일 이진 신호를 구별하는 능력을 향상시킨다.

3D 공간에서 위치 $(x, y, z)$에 있는 토큰의 RoPE3D 인코딩은 각 축에 독립적으로 적용됩니다:

$$\text{RoPE3D}(\mathbf{q}, x, y, z) = \text{RoPE}_x(\mathbf{q}_x, x) \oplus \text{RoPE}_y(\mathbf{q}_y, y) \oplus \text{RoPE}_z(\mathbf{q}_z, z)$$

스케일 $l$에 대한 스케일별 임베딩 $e_l$을 추가하여:

$$\mathbf{h}_i^{(l)} = \mathbf{h}_i + e_l$$

#### (5) 다중 토큰 병렬 생성 (Token-Parallel Generation)

깊이별 teacher-forcing 마스크를 이용한 다중 토큰 생성 전략을 채택하여, 계층적 의존성을 유지하면서 여러 토큰을 병렬로 예측한다.

Depth-wise teacher-forcing 마스크 $\mathbf{M}$은 다음과 같이 정의됩니다:

$$M_{ij} = \begin{cases} 1 & \text{if } \text{depth}(j) < \text{depth}(i) \\ 0 & \text{otherwise} \end{cases}$$

이를 통해 같은 깊이(depth)의 토큰들은 병렬로 예측되고, 상위 계층(coarse) 토큰들만 조건으로 사용됩니다.

#### (6) Octree 기반 어텐션 (Efficient Attention)

OctGPT는 효율적인 자기 어텐션 계산을 위해 토큰을 고정 크기 윈도우로 나누고, 교차 윈도우 상호작용을 가능하게 하는 팽창 옥트리 어텐션(dilated octree attention)과 시프트 윈도우 어텐션(shifted window attention)을 번갈아 사용한다.

---

### 2.3 모델 구조

전체 파이프라인은 **두 단계(Two-Stage)**로 구성됩니다:

```
[Stage 1: VQVAE 학습]
3D Shape → Octree Encoder → Codebook (이진 잠재 코드 z) → Octree Decoder → 재구성

[Stage 2: Autoregressive Transformer 학습]
Octree 이진 분할 신호 + VQVAE 코드
→ 직렬화 다중 스케일 시퀀스 T
→ OctGPT Transformer (Octree Attention + RoPE3D + Scale Embedding)
→ 자기회귀 토큰 예측
→ VQVAE 디코더 → 최종 3D 형상
```

추론 시, 트랜스포머는 토큰 시퀀스를 점진적으로 예측하여 옥트리와 잠재 코드를 재구성하며, 조대한 것부터 세밀한 방향으로(coarse-to-fine) 3D 형상을 생성한다. 이 시퀀스는 VQVAE에 의해 디코딩되어 최종 3D 형상을 생성한다.

---

### 2.4 성능 향상

옥트리 기반 트랜스포머에 3D 회전 위치 인코딩, 스케일별 임베딩, 토큰 병렬 생성 방식을 통합하여 학습 시간을 13배, 생성 시간을 69배 단축하며, $1024^3$ 해상도의 고해상도 3D 형상을 단 4개의 NVIDIA 4090 GPU로 며칠 내에 효율적으로 학습할 수 있게 한다.

AutoSDF, 3DILG, MeshGPT와 같은 기존 자기회귀 방법들과 비교하여 OctGPT는 FID 성능에서 상당한 향상을 이루었다. 특히, OctGPT는 이전 최고 성능의 자기회귀 3D 형상 생성 모델인 3DILG보다 FID 점수에서 평균 42.84 향상을 달성하였다.

OctGPT는 평균적으로 최고 성능을 달성한다. OctGPT는 다른 자기회귀 방법들을 지속적으로 능가하며 일부 카테고리에서는 최첨단 확산 기반 방법들도 초월한다. 기존 자기회귀 접근법과 비교하여 FID 성능에서 상당한 향상을 달성하면서 훨씬 더 효율적이다.

이 접근법을 통해 직접 좌표 예측에 비해 훨씬 빠른 수렴과 높은 생성 품질을 달성한다. 예를 들어, 제안 방법은 단 10 에폭 학습 후 고품질 3D 형상을 생성한 반면, 좌표 예측 방식은 100 에폭 학습 후에도 만족스러운 결과를 내지 못했다.

---

### 2.5 한계 (Limitations)

두 단계 파이프라인(VQVAE 이후 자기회귀 트랜스포머)이 엔드-투-엔드(end-to-end)로 학습 가능하지 않아 전체적인 성능을 제한할 수 있다.

추가적으로 확인된 한계점들:

1. **텍스처 부재**: 논문은 주로 기하학적 형상(geometry)에 집중하며, 텍스처(texture) 생성을 통합하지 않음.
2. **대규모 씬 처리의 한계**: 씬 수준의 생성은 5K 씬을 포함하는 Synthetic Rooms 데이터셋에서 학습되며, 대규모 실세계 씬으로의 확장성은 추가 검증 필요.
3. **이진 토큰의 표현 제한**: VQVAE의 이진화된 잠재 코드는 연속적 표현 대비 미세한 디테일 표현에 한계 가능성.

---

## 3. 모델의 일반화 성능 향상 가능성

OctGPT는 스케치와 이미지로부터 3D 형상을 생성할 수 있어 일반화 능력을 보여준다. 뷰 정보에 의존하지 않고도 입력 스케치나 이미지와 일관된 고품질 형상을 생성한다.

스케치 조건부 생성을 통해 OctGPT의 일반화 능력과 다양성을 입증한다. 스케치 입력은 사전 학습된 DINOv2로 인코딩된 후 OctGPT 모델에 통합된다. 주목할 점은 뷰 정보에 의존하지 않고, LAS-Diffusion의 뷰 인식 어텐션 메커니즘을 표준 크로스 어텐션 모듈로 대체함에도 불구하고, 입력 스케치와 일관된 고품질 형상을 생성한다는 것이다.

OctGPT는 텍스트-, 스케치-, 이미지 조건부 생성, 여러 객체가 포함된 씬 수준 합성을 포함한 다양한 태스크에 걸쳐 탁월한 다목적성을 보여준다.

**일반화 성능 향상 가능성의 주요 메커니즘:**

1. **계층적 다중 스케일 표현**: 직렬화된 옥트리 표현을 통해 3D 형상의 계층적·공간적 구조를 효율적으로 포착하며, 조대한 기하학은 옥트리로 인코딩되고 세밀한 디테일은 VQVAE로 생성된 이진 토큰으로 표현되어 자기회귀 예측에 적합한 컴팩트한 다중 스케일 이진 시퀀스로 변환된다.

2. **LLM/이미지 모델과의 정렬 가능성**: OctGPT는 Diffusion 모델과 구별되는 3D 콘텐츠 생성의 새로운 패러다임을 제시하며, 대형 언어 및 이미지 모델과의 정렬 또는 미세 조정을 통한 멀티모달 모델 개발을 촉진할 수 있다고 연구진은 믿는다.

3. **다양한 해상도 학습**: OctGPT는 훨씬 더 긴 시퀀스를 처리할 수 있어 더 세밀한 디테일을 포착하고 더 높은 품질의 형상을 생성한다.

4. **멀티모달 확장 방향**: 멀티모달리티 학습 탐색은 다양한 입력 모달리티를 처리할 수 있도록 하여 OctGPT의 적용 가능성을 넓힐 수 있는 흥미로운 방향이며, 이는 3D 형상 생성을 다른 모달리티와 통합하는 새로운 가능성을 열어줄 것이다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

PolyGen (ICML 2020)은 트랜스포머 기반 3D 메시 생성 프레임워크를 도입하였다. 버텍스 모델과 페이스 모델의 두 개의 순차적 트랜스포머 네트워크를 사용한다.

| 모델 | 연도 | 표현 | 방법 | 주요 한계 |
|------|------|------|------|-----------|
| **PolyGen** | 2020 | 메시(vertex+face) | Autoregressive Transformer | 복잡한 토폴로지 제한 |
| **ShapeFormer** | 2022 | VQDIF (희소 암묵 함수) | Autoregressive Transformer | 낮은 해상도, 짧은 시퀀스 |
| **Octree Transformer** | 2023 | 옥트리 | Autoregressive | 세밀한 디테일 부재 |
| **MeshGPT** | 2024 | 메시 | Autoregressive | 복잡 형상 제한 |
| **OctFusion** | 2024 | 옥트리 + 잠재 특징 | Diffusion | 고비용 샘플링 |
| **OctGPT** | 2025 | 직렬화 옥트리 + VQVAE | Multiscale Autoregressive | Two-stage (비end-to-end) |

3D 형상 생성의 맥락에서 여러 연구들이 자기회귀 모델링을 탐색했다. PointGrow는 트랜스포머 유사 자기 어텐션 메커니즘을 사용하여 이전에 생성된 포인트들에 조건화하면서 단계적으로 포인트를 자기회귀적으로 예측한다. 이는 장거리 의존성을 포착하고 다양한 형상을 생성하지만, 샘플링이 느리다는 본질적 단점이 있다.

ShapeFormer는 여러 완성 결과의 분포를 샘플링할 수 있으며, 각각은 그럴듯한 형상 디테일을 보이면서 입력에 충실하다. 트랜스포머를 위해 공간적 희소성을 활용하여 짧은 이산 변수 시퀀스로 3D 형상을 근사하는 VQDIF 표현을 도입했다.

Octree Transformer는 옥트리를 순회 순서로 시퀀스화할 수 있는 컴팩트한 계층적 형상 표현으로 사용한다. 시퀀스 길이를 크게 줄이는 적응적 압축 방식을 도입하여 완전 자기회귀 샘플링과 병렬 학습을 허용하면서 트랜스포머로 효과적인 생성을 가능하게 한다.

**OctGPT의 차별점**: 기존 Octree Transformer 대비 **VQVAE 기반 세밀한 디테일 표현**과 **3D RoPE + 스케일 임베딩 + 병렬 토큰 생성**을 결합하여 훨씬 높은 해상도($1024^3$)에서 실용적 학습을 가능하게 한 것이 핵심 혁신입니다.

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5.1 앞으로의 연구에 미치는 영향

OctGPT는 수렴을 가속화하고 기존 자기회귀 방법보다 생성 품질을 향상시켜 고품질의 확장 가능한 3D 콘텐츠 생성을 위한 새로운 패러다임을 제시한다.

1. **자기회귀 vs. Diffusion 패러다임 재정립**: OctGPT는 기존에 Diffusion 모델이 독주하던 3D 생성 분야에 자기회귀 모델이 강력한 경쟁자가 될 수 있음을 입증하였습니다.

2. **멀티모달 3D 생성 LLM의 가능성**: OctGPT는 LLM 및 이미지 모델과의 정렬/파인튜닝을 통한 멀티모달 모델 개발을 촉진할 수 있으며, 이는 GPT-4V와 같은 대형 모델에 3D 이해 능력을 부여하는 연구 방향으로 이어질 수 있습니다.

3. **씬 규모 생성 연구 촉진**: 무조건부 생성, 카테고리·텍스트·이미지 조건부 생성, 대규모 씬 합성 등 다양한 시나리오에서 고품질 3D 형상을 생성할 수 있음을 보여줌으로써, 이 분야 연구의 기준점(baseline)을 새롭게 설정하였습니다.

4. **계층적 이진 토큰화의 일반성**: 옥트리 기반 이진 시퀀스라는 아이디어는 포인트 클라우드, NeRF, Gaussian Splatting 등 다른 3D 표현에도 응용될 수 있습니다.

### 5.2 앞으로 연구 시 고려할 점

1. **End-to-End 학습**: 두 단계 파이프라인(VQVAE + 자기회귀 트랜스포머)이 end-to-end로 학습 불가한 점은 전체 성능을 제한하므로, 통합 학습 프레임워크 개발이 중요한 연구 과제입니다.

2. **텍스처 통합**: 현재 OctGPT는 기하학 생성에 집중하며 텍스처를 다루지 않습니다. 텍스처 정보를 옥트리 잠재 코드에 통합하거나, 기하학 생성 후 텍스처 생성 모델을 연결하는 연구가 필요합니다.

3. **멀티모달 학습 확장**: 다양한 입력 모달리티를 처리할 수 있도록 하는 멀티모달리티 학습 탐색은 3D 형상 생성을 다른 모달리티와 통합하는 새로운 가능성을 열어줄 것이다.

4. **실세계 데이터 일반화**: 현재 실험은 ShapeNet, Objaverse, Synthetic Rooms 등 합성 데이터셋 중심으로 진행되었습니다. 실세계 스캔 데이터로의 도메인 일반화(domain generalization) 연구가 중요합니다.

5. **모델 스케일링 법칙 연구**: GPT 시리즈에서 확인된 스케일링 법칙(Scaling Law)이 3D 자기회귀 모델에도 적용되는지, 즉 모델 크기·데이터 크기와 생성 품질의 관계를 규명하는 연구가 필요합니다.

6. **물리 기반 제약 통합**: 생성된 3D 형상의 물리적 타당성(structural integrity, 제조 가능성)을 보장하는 제약 조건을 모델에 통합하는 방향도 중요한 응용 연구 주제입니다.

---

## 📚 참고 자료

| # | 자료 | 출처 |
|---|------|------|
| 1 | **OctGPT** (arXiv 2504.09975) | https://arxiv.org/abs/2504.09975 |
| 2 | **OctGPT** (SIGGRAPH 2025, ACM DL) | https://dl.acm.org/doi/10.1145/3721238.3730601 |
| 3 | **OctGPT** (arXiv HTML 전문) | https://arxiv.org/html/2504.09975v1 |
| 4 | **OctGPT** (arXiv PDF) | https://arxiv.org/pdf/2504.09975 |
| 5 | **OctGPT** (HuggingFace Papers) | https://huggingface.co/papers/2504.09975 |
| 6 | **OctGPT** (aimodels.fyi) | https://www.aimodels.fyi/papers/arxiv/octgpt-octree-based-multiscale-autoregressive-models-3d |
| 7 | **OctGPT** (ResearchGate) | https://www.researchgate.net/publication/390772327 |
| 8 | **Octree Transformer** (arXiv 2111.12480) | https://arxiv.org/pdf/2111.12480 |
| 9 | **ShapeGPT** (arXiv 2311.17618) | https://arxiv.org/abs/2311.17618 |
| 10 | **3D Shape Generation: A Survey** (arXiv 2506.22678) | https://arxiv.org/html/2506.22678v2 |
| 11 | **PolyGen** (ICML 2020) | Nash et al., 2020 |
| 12 | **ShapeFormer** (CVPR 2022) | Yan et al., 2022 |
| 13 | **GET3D** (NeurIPS 2022) | ResearchGate |
| 14 | **OctGPT GitHub** | https://github.com/octree-nn/octgpt |
