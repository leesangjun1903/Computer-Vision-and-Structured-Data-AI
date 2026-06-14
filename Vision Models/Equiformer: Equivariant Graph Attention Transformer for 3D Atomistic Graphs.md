# Equiformer: Equivariant Graph Attention Transformer for 3D Atomistic Graphs

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장
Equiformer는 **Transformer 아키텍처가 적절한 귀납적 편향(inductive bias)을 갖추면 3D 원자 그래프에서도 효과적으로 일반화**될 수 있음을 실증적으로 보여줍니다. 기존 Transformer들이 3D 원자 그래프 도메인에서 저조한 성능을 보였던 근본적 이유는 $SE(3)/E(3)$-등변성(equivariance) 부재였으며, 이를 비가약 표현(irreducible representations, irreps)을 통해 해결합니다.

### 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **등변 Transformer 아키텍처** | 기존 Transformer 연산을 등변 대응물로 교체 + 텐서곱 추가 |
| **등변 그래프 어텐션** | MLP 어텐션 + 비선형 메시지 패싱 |
| **대규모 검증** | QM9, MD17, OC20 데이터셋에서 경쟁력 있는 성능 달성 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**문제 1: Transformer의 3D 도메인 일반화 실패**

Vision Transformer(ViT)가 ImageNet 단독 학습 시 CNN보다 성능이 낮았듯이, 기존 등변 Transformer들(SE(3)-Transformer, TorchMD-NET, EQGAT)은 모든 데이터셋에서 일관되게 우수한 성능을 보이지 못했습니다.

**문제 2: 기존 등변 네트워크들의 한계**

- **TFN, NequIP**: 선형 메시지만 사용 → 표현력 제한
- **SEGNN**: 비선형 메시지 도입했으나 어텐션 메커니즘 부재
- **SE(3)-Transformer**: 내적(dot product) 어텐션 + 선형 메시지 → 표현력 부족, 메모리 비효율적
- **TorchMD-NET, EQGAT**: $L_{\max} = 1$ 제한 → 고차 텐서 상호작용 불가

### 2.2 제안하는 방법 (수식 포함)

#### (1) E(3) 등변성의 수학적 정의

함수 $f: X \to Y$가 변환 그룹 $G$에 대해 등변적(equivariant)이려면:

$$f(D_X(g)x) = D_Y(g)f(x), \quad \forall x \in X, \; g \in G$$

여기서 $D_X(g)$, $D_Y(g)$는 각각 입력/출력 공간에서 $g$로 매개변수화된 변환 행렬입니다.

#### (2) 비가약 표현(Irreps)과 Wigner-D 행렬

$SO(3)$의 임의의 군 표현은 **비가약 표현들의 직합(direct sum)**으로 분해됩니다:

$$D(g) = P^{-1} \left( \bigoplus_i D_{l_i}(g) \right) P$$

차수(degree) $L$의 type- $L$ 벡터는 $(2L+1)$차원이며, 스칼라($L=0$), Euclidean 벡터($L=1$), 텐서($L=2, 3, \ldots$)를 통합합니다.

**구형 조화 함수(Spherical Harmonics):**

$$f^{(L)} = Y^{(L)}\!\left(\frac{\vec{r}}{\|\vec{r}\|}\right), \quad D_L(g)f^{(L)} = Y^{(L)}\!\left(\frac{D_1(g)\vec{r}}{\|D_1(g)\vec{r}\|}\right)$$

#### (3) 텐서곱 (Tensor Product)

Type- $L_1$ 벡터 $f^{(L_1)}$과 Type- $L_2$ 벡터 $g^{(L_2)}$의 텐서곱으로 Type- $L_3$ 벡터 $h^{(L_3)}$를 생성:

$$h^{(L_3)}_{m_3} = \left(f^{(L_1)} \otimes g^{(L_2)}\right)_{m_3} = \sum_{m_1=-L_1}^{L_1} \sum_{m_2=-L_2}^{L_2} C^{(L_3, m_3)}_{(L_1, m_1)(L_2, m_2)} f^{(L_1)}_{m_1} g^{(L_2)}_{m_2}$$

$C^{(L_3, m_3)}_{(L_1, m_1)(L_2, m_2)}$: Clebsch-Gordan 계수, $|L_1 - L_2| \leq L_3 \leq |L_1 + L_2|$일 때만 비영(non-zero).

**O(3) 텐서곱 (패리티 포함):**

$$h^{(L_3, p_3)}_{m_3} = \sum_{m_1, m_2} C^{(L_3,m_3)}_{(L_1,m_1)(L_2,m_2)} f^{(L_1,p_1)}_{m_1} g^{(L_2,p_2)}_{m_2}, \quad p_3 = p_1 \times p_2$$

#### (4) 등변 그래프 어텐션 (핵심 기여)

**메시지 정의:**

$$m_{ij} = a_{ij} \times v_{ij}$$

**콘텐츠 및 기하 정보 통합:**

$$x_{ij} = \text{Linear}_{\text{dst}}(x_i) + \text{Linear}_{\text{src}}(x_j)$$

$$x'_{ij} = x_{ij} \otimes^{DTP}_{w(\|\vec{r}_{ij}\|)} \text{SH}(\vec{r}_{ij}), \quad f_{ij} = \text{Linear}(x'_{ij})$$

여기서 $w(\|\vec{r}_{ij}\|)$는 상대 거리로 조건화된 깊이별 텐서곱(DTP) 가중치.

**MLP 어텐션 (Multi-Layer Perceptron Attention):**

$$z_{ij} = a^\top \text{LeakyReLU}\!\left(f^{(0)}_{ij}\right), \quad a_{ij} = \text{softmax}_j(z_{ij}) = \frac{\exp(z_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(z_{ik})}$$

$a$는 학습 가능한 벡터, $f^{(0)}_{ij}$는 스칼라 특징(방향 정보 인코딩).

**비선형 메시지 패싱:**

$$\mu_{ij} = \text{Gate}\!\left(f^{(L)}_{ij}\right), \quad v_{ij} = \text{Linear}\!\left([\mu_{ij} \otimes^{DTP}_w \text{SH}(\vec{r}_{ij})]\right)$$

**Gate 활성화 함수:**
- Type-0 벡터: SiLU/sigmoid 적용 가능
- Type- $L$ ($L>0$) 벡터: 스칼라에서 비선형 변환된 가중치를 곱함

$$\text{Gate}(x)^{(L)}_{c} = \sigma(s_c) \cdot x^{(L)}_c, \quad L > 0$$

**레이어 정규화 (등변 버전):**

$$\text{LN}(x) = \frac{x}{\text{RMS}_C(\text{norm}(x))} \circ \gamma$$

### 2.3 모델 구조

```
입력 3D 그래프
    ↓
[임베딩 모듈]
  - 원자 임베딩: 원자 종류 one-hot → Linear
  - 엣지-차수 임베딩: 기하 정보 집계
    ↓
[Transformer 블록 × N]
  ┌─────────────────────────────────┐
  │  LayerNorm                      │
  │  ↓                              │
  │  등변 그래프 어텐션              │
  │  - MLP Attention                 │
  │  - 비선형 메시지 패싱            │
  │  - Multi-Head Attention          │
  │  ↓                              │
  │  Residual Connection            │
  │  LayerNorm                      │
  │  Feed Forward Network           │
  │  (Linear → Gate → Linear)       │
  │  Residual Connection            │
  └─────────────────────────────────┘
    ↓
[출력 헤드]
  - FFN → 스칼라 변환 → 합산 집계
```

**주요 하이퍼파라미터 (QM9 기준):**

| 파라미터 | 값 |
|----------|-----|
| Transformer 블록 수 | 6 |
| 임베딩 차원 $d_{\text{embed}}$ | [(128,0),(64,1),(32,2)] |
| 어텐션 헤드 수 $h$ | 4 |
| 차단 반경 | 5 Å |
| 최대 차수 $L_{\max}$ | 2 (또는 3) |

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 등변성이 일반화에 기여하는 이유

등변 신경망은 **대칭성을 귀납적 편향으로 내재화**하므로, 데이터 증강 없이 회전/평행이동에 대한 불변성을 보장합니다. 논문의 부록(Sec. A.2)에서 명시적으로 언급:

> *"Group equivariant neural networks are guaranteed to make equivariant predictions on data transformed by a group. Additionally, they are found to be data-efficient and generalize better than non-symmetry-aware and invariant methods."*

### 3.2 OOD(Out-of-Distribution) 성능

OC20 데이터셋은 **분포 외 일반화를 직접 측정**하는 4개의 서브 분할을 제공합니다:

| 서브셋 | 의미 |
|--------|------|
| **ID** | 분포 내 흡착물 & 촉매 |
| **OOD-Ads** | 분포 외 흡착물 |
| **OOD-Cat** | 분포 외 촉매 |
| **OOD-Both** | 분포 외 흡착물 & 촉매 |

Equiformer는 **모든 4개 서브셋에서 SEGNN, SphereNet 대비 일관되게 낮은 MAE** 달성:

$$\text{평균 MAE (Equiformer)}: 0.5858 \text{ eV} < \text{SEGNN}: 0.6101 \text{ eV}$$

### 3.3 소규모 데이터셋에서의 일반화 (MD17)

MD17은 분자당 **950개 훈련 샘플**만 사용하는 극도로 데이터 효율적인 설정입니다. Equiformer ($L_{\max}=2$)가 NequIP ($L_{\max}=3$, 더 많은 파라미터)보다 전반적으로 낮은 MAE 달성:

> 이는 **MLP 어텐션이 선형 메시지보다 더 표현력 있는 어텐션 패턴**을 학습하여 소량 데이터에서도 효과적임을 시사.

### 3.4 일반화 성능 향상의 메커니즘 분석

**① 고차 텐서($L_{\max}$ 증가)의 영향:**

$L_{\max}$를 2에서 3으로 증가시키면 대부분의 분자에서 성능 향상(단, 벤젠은 과적합 발생):

$$\text{Equiformer}(L_{\max}=2) \xrightarrow{L_{\max} \uparrow} \text{Equiformer}(L_{\max}=3): \text{MAE 개선 (대부분)}$$

**② MLP 어텐션 vs. 내적 어텐션:**

MLP는 **보편 근사 정리(Universal Approximation Theorem)**에 의해 임의의 어텐션 패턴을 이론적으로 근사:

$$\text{MLP Attention} \supset \text{Dot Product Attention (표현력 측면)}$$

OC20처럼 다양한 원자 종류와 대형 그래프에서 MLP 어텐션이 더 명확한 개선 보임 → **복잡하고 다양한 데이터에서 일반화 우위**.

**③ 비선형 메시지 패싱의 효과:**

두 데이터셋 모두에서 비선형 메시지가 선형 메시지 대비 일관된 성능 향상:

| 모델 | QM9 ($\alpha$) | OC20 평균 MAE |
|------|---------------|--------------|
| 선형 메시지 + MLP 어텐션 | 0.051 | 0.5555 |
| **비선형 메시지 + MLP 어텐션** | **0.046** | **0.5489** |

**④ IS2RS 보조 작업(Auxiliary Task)의 역할:**

노드 레벨 보조 작업(IS2RS) 추가 시 OOD 성능 대폭 향상:

$$\text{Equiformer (IS2RE only)}: 0.4657 \text{ eV} \xrightarrow{+\text{IS2RS}} 0.4410 \text{ eV (검증셋)}$$

이는 **구조 예측 보조 작업이 원자 간 상호작용에 대한 더 풍부한 표현 학습**을 유도함을 보여줌.

**⑤ Noisy Nodes 데이터 증강:**

$$\text{Equiformer + IS2RS}: 0.4410 \xrightarrow{+\text{Noisy Nodes}} 0.4344 \text{ (검증셋 평균 MAE)}$$

---

## 4. 성능 향상 및 한계

### 4.1 성능 향상 요약

**QM9 (12개 작업 전체에서 최고 성능):**

| 작업 | Equiformer | 차순위 (EQGAT) |
|------|-----------|--------------|
| $\varepsilon_{\text{HOMO}}$ | **15 meV** | 20 meV |
| $\varepsilon_{\text{LUMO}}$ | **14 meV** | 16 meV |
| $\Delta\varepsilon$ | **30 meV** | 32 meV |

**OC20 IS2RE (보조 작업 포함, 테스트셋):**

$$\text{Graphormer (372 GPU-days)}: 0.4722 \text{ eV} \quad \text{vs.} \quad \text{Equiformer (24 GPU-days)}: \mathbf{0.4660} \text{ eV}$$

훈련 효율: **2.3× ~ 15.5× 향상**

### 4.2 한계 (논문 Sec. G 기반)

**① 계산 비용:**
- 고차 $L$: 특징 차원 $(2L+1)$ 증가 → 메모리 부담
- 텐서곱: GPU에 최적화되지 않은 커널 → 비선형 메시지 시 텐서곱 2배 증가

**② 작업/데이터셋 의존성:**
- QM9(작은 분자, 단순 구성): MLP 어텐션 vs. 내적 어텐션 차이 미미
- OC20(대형/다양한 시스템): MLP 어텐션 명확한 우위

**③ 추가 집계 비용:**
- Softmax 연산 → 추가 합산 집계 필요 → 일반 메시지 패싱 대비 오버헤드

**④ 다른 도메인 적용 제약:**
- 어텐션 복잡도: 채널 수 × 엣지 수에 비례
- 컴퓨터 비전 등에서는 픽셀/노드 수 기준 복잡도와 불일치 → 추가 수정 필요

---

## 5. 2020년 이후 최신 연구 비교 분석

### 5.1 Equiformer 이전/동시대 주요 연구

| 모델 | 연도 | 특징 | Equiformer 대비 한계 |
|------|------|------|---------------------|
| **NequIP** (Batzner et al., 2022) | 2022 | E(3)-등변 그래프 컨볼루션, 선형 메시지 | 선형 메시지만 → 표현력 제한 |
| **SEGNN** (Brandstetter et al., 2022) | 2022 | 비선형 메시지, 물리량 활용 | 어텐션 메커니즘 없음 |
| **TorchMD-NET** (Thölke & Fabritiis, 2022) | 2022 | 등변 Transformer, $L_{\max}=1$ | 고차 텐서 미지원, 내적 어텐션 |
| **EQGAT** (Le et al., 2022) | 2022 | 등변 그래프 어텐션 | $L_{\max}=1$ 제한 |
| **MACE** (Musaelian et al., 2022) | 2022 | 로컬 등변 표현, 효율적 | 어텐션 메커니즘 없음 |
| **Graphormer** (Shi et al., 2022) | 2022 | 불변 Transformer, OC20 SOTA | 등변성 없음 → 훈련 효율 낮음 |

### 5.2 Equiformer 이후 발전 연구

> **주의:** 아래 내용은 논문 출판(ICLR 2023) 이후의 연구이므로, 원 논문에 직접 언급되지 않았습니다. 알려진 후속 연구를 근거 기반으로 서술합니다.

**EquiformerV2** (Liao et al., 2023, arXiv:2306.12059):
- Equiformer의 직접적 후속작
- SO(2) 컨볼루션을 활용하여 고차 텐서($L_{\max}$) 확장 시 계산 비용 대폭 절감
- OC20에서 당시 SOTA 달성

**주요 연구 트렌드 비교:**

| 측면 | Equiformer (2023) | 이후 연구 방향 |
|------|-------------------|---------------|
| 등변성 | SE(3)/E(3) | 동일 방향 유지 |
| 어텐션 | MLP 어텐션 | 더 효율적인 변형 탐색 |
| 텐서곱 | DTP (깊이별) | SO(2) 컨볼루션으로 대체 |
| 스케일 | 소/중형 | 대규모 원자 시스템 확장 |

---

## 6. 앞으로의 연구에 미치는 영향과 고려할 점

### 6.1 연구에 미치는 영향

**① Transformer와 물리적 대칭성의 통합 패러다임 확립**

Equiformer는 *"도메인 특화 귀납적 편향 + Transformer 강점"* 결합의 성공 사례를 제시합니다. 이는 단백질 구조 예측, 재료 설계, 약물 발견 등 분야로의 확장을 촉진합니다.

**② 등변 어텐션 메커니즘 연구 가속화**

MLP 어텐션이 내적 어텐션보다 대형/다양한 그래프에서 우월함을 실증함으로써, **등변 어텐션 설계**가 독립적 연구 분야로 부상했습니다.

**③ 계산 화학과 ML의 가교**

OC20 같은 대규모 촉매 데이터셋에 등변 Transformer 적용 가능성을 보여줌으로써 **계산 효율적인 ab initio 대안** 연구를 자극합니다.

**④ 소프트웨어 인프라 발전 촉진**

e3nn 라이브러리 기반 구현이 공개(GitHub: atomicarchitects/equiformer)되어, 후속 연구자들의 진입 장벽 낮춤.

### 6.2 앞으로 연구 시 고려할 점

**① 계산 효율성 개선 (최우선 과제)**

텐서곱의 GPU 최적화가 필수적입니다:
- **경로 가지치기(Path Pruning)**: 중요하지 않은 $(L_1, L_2) \to L_3$ 경로 제거
- **SO(2) 컨볼루션 활용**: EquiformerV2 방향 → $O(L_{\max}^3)$에서 $O(L_{\max}^2)$로 복잡도 감소
- **혼합 정밀도 훈련**: 불필요한 메모리 최소화

**② $L_{\max}$ 자동 선택 및 적응적 설정**

현재 $L_{\max}$는 수동 튜닝이 필요합니다. 작업/데이터 특성에 따라 최적 $L_{\max}$가 다르므로:
- 학습 중 차수의 중요도를 측정하는 **적응적 $L_{\max}$ 선택** 메커니즘 연구
- 물리적 직관(예: 특정 성질은 $L \leq 2$로 충분)과 결합

**③ 더 큰 시스템으로의 확장 (스케일링)**

OC20 대비 더 복잡한 시스템(단백질, 결정 구조 등)에 적용 시:
- **로컬 vs. 글로벌 어텐션** 균형: 현재 국소 이웃만 고려 → 장거리 상호작용 포착을 위한 계층적 어텐션 탐색
- **그래프 군집화**와 결합하여 메모리 효율 개선

**④ 일반화 메커니즘의 이론적 분석**

등변성이 왜 일반화 성능을 높이는지에 대한 이론적 보장이 부족합니다:
- **PAC-Learning 프레임워크**와 결합한 등변 네트워크의 샘플 복잡도 분석
- OOD 일반화와 대칭 군의 관계를 수학적으로 규명

**⑤ 사전 훈련 및 전이 학습 (Pre-training)**

현재 Equiformer는 각 데이터셋을 독립적으로 훈련합니다:
- **대규모 사전 훈련 + 미세 조정** 패러다임 도입 (예: 분자 동역학 데이터로 사전 훈련 → 특정 성질 예측 미세 조정)
- 자기 지도 학습(Self-supervised Learning)과 등변 GNN 결합

**⑥ 불확실성 정량화**

과학적 응용에서는 예측 불확실성이 중요합니다:
- 베이지안 등변 네트워크 또는 앙상블 기반 불확실성 추정
- 능동 학습(Active Learning)과 결합하여 데이터 효율성 극대화

**⑦ E(3) vs SE(3) 등변성 선택 기준**

논문에서 E(3)-Equiformer가 OC20에서 SE(3)-Equiformer보다 미소하게 열등함을 보임:
- 역전(inversion) 대칭이 중요한 작업(예: 카이랄 분자, 광학 이성질체 구분) vs. 불필요한 작업에 대한 체계적 분석 필요
- 패리티가 물리적으로 의미 있는 경우에만 E(3) 적용하는 **조건부 등변성** 연구

---

## 참고 자료 (출처)

1. **원 논문**: Yi-Lun Liao, Tess Smidt. "Equiformer: Equivariant Graph Attention Transformer for 3D Atomistic Graphs." *ICLR 2023*. arXiv:2206.11990v2.

2. **코드 저장소**: https://github.com/atomicarchitects/equiformer

3. **관련 인용 논문** (원 논문 내 참고문헌):
   - Batzner et al. (2022). "E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials." *Nature Communications*.
   - Brandstetter et al. (2022). "Geometric and physical quantities improve E(3) equivariant message passing." *ICLR 2022*.
   - Fuchs et al. (2020). "SE(3)-Transformers: 3D rototranslation equivariant attention networks." *NeurIPS 2020*.
   - Thölke & Fabritiis (2022). "Equivariant transformers for neural network based molecular potentials." *ICLR 2022*.
   - Chanussot* et al. (2021). "Open Catalyst 2020 (OC20) dataset and community challenges." *ACS Catalysis*.
   - Vaswani et al. (2017). "Attention is all you need." *NeurIPS 2017*.
   - Brody et al. (2022). "How attentive are graph attention networks?" *ICLR 2022*.
   - Thomas et al. (2018). "Tensor field networks." arXiv:1802.08219.
   - Geiger et al. (2022). "e3nn/e3nn: 2022-04-13." Zenodo. https://doi.org/10.5281/zenodo.6459381.

4. **후속 연구** (원 논문 외):
   - Liao et al. (2023). "EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations." arXiv:2306.12059. *(원 논문에 미포함, 후속 연구로 별도 확인 필요)*
