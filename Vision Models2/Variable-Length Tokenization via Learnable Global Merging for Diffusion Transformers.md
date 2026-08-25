# Variable-Length Tokenization via Learnable Global Merging for Diffusion Transformers

---

## 1. Executive Summary (10문장 이내)

본 논문은 Latent Diffusion Models(LDMs)에서 단일 모델로 품질-연산 트레이드오프를 유연하게 제어하기 위한 **가변 길이 토크나이저(Variable-Length Tokenizer, VLT)** 를 제안한다.  
기존 VLT는 주로 **Nested Dropout** 방식으로 토큰 수를 조절하는데, 이는 순서화된 토큰 시퀀스를 잘라내어 토큰 길이에 따라 유사도 구조(similarity structure)가 달라지는 **cross-length shift** 문제를 유발한다.  
이 분포 불일치(distribution shift)는 단일 가변 길이 확산 모델의 학습을 어렵게 만든다.  
저자들은 이를 해결하기 위해 **토큰 병합(token merging)** 기반 길이 조절을 제안하며, 유사한 토큰을 병합하면 길이 간 표현 정렬(representational alignment)이 가능함을 수학적으로 보인다.  
핵심 도전은 기존 병합 방법이 데이터 의존적이어서 생성 시 사용 불가능하다는 것인데, 이를 **학습 가능한 전역 병합(Learnable Global Merging, LGM)** 으로 해결한다.  
LGM은 데이터와 독립적인 학습 가능 임베딩과 Agglomerative Clustering, Straight-Through Trick을 조합하여 그래디언트 전파 문제를 극복한다.  
생성 시에는 **비례 어텐션(Proportional Attention)** 과 **병합 위치 임베딩(Merged Positional Embedding)** 을 통해 DiT와 호환성을 확보한다.  
ImageNet 256×256 실험에서 기존 VLT 대비 gFID-연산량 트레이드오프에서 우수한 성능을 달성한다.  
경량 길이별 LoRA 파인튜닝을 추가하면 최소한의 오버헤드로 성능을 추가 향상시킬 수 있다.

> **💡 용어 설명**
> - **Latent Diffusion Model (LDM)**: 픽셀 공간이 아닌 압축된 잠재 공간에서 확산 과정을 수행하는 생성 모델 (예: Stable Diffusion)
> - **Nested Dropout**: 학습 시 토큰 시퀀스의 끝부분을 무작위로 잘라내어 중요 정보가 앞쪽 토큰에 집중되도록 유도하는 기법
> - **gFID (generation FID)**: 생성 이미지와 실제 이미지 분포 간의 거리를 측정하는 지표. 낮을수록 좋음

### 1-1. 연구의 목적과 필요성

**목적**: 단일 가변 길이 확산 모델이 다양한 토큰 수에서 효과적으로 동작할 수 있도록, 길이 간 표현 정렬을 보장하는 새로운 VLT를 설계한다.

**필요성 (3단 논리)**:

| 단계 | 내용 |
|---|---|
| ① 현실적 수요 | 고화질 생성과 저자원 환경 모두를 단일 모델로 지원해야 하는 다양한 배포 시나리오 존재 |
| ② 기존 방법의 한계 | Nested Dropout 기반 VLT는 cross-length shift로 인해 단일 모델의 공동 학습(joint training)이 어렵고, 더 많은 토큰을 써도 성능이 개선되지 않는 역설 발생 (Fig. 1) |
| ③ 제안 방법의 기여 | 토큰 병합으로 길이 간 직접 정렬이 가능해져 단일 가변 길이 DiT 학습이 효과적으로 이루어짐 |

> **💡 용어 설명**
> - **Cross-length shift**: 토큰 수가 달라질 때 잠재 표현들 사이의 유사도 구조가 변하는 현상. 확산 모델이 학습해야 할 score function이 토큰 수마다 달라져서 공동 학습이 어려워짐
> - **Score function**: 확산 모델이 학습하는 잠재 분포의 기울기 정보. 분포가 달라지면 score function도 달라짐

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 | 위치 |
|---|---|---|---|
| ① | Nested Dropout은 cross-length 표현 정렬을 심각하게 훼손한다 | CKNNA 측정: 독립 학습 토크나이저(0.51) 대비 Nested Dropout은 0.13~0.34에 불과 | Fig. 1, p.3 |
| ② | 이 정렬 훼손이 공동 학습 DiT의 성능 저하를 유발한다 | 공동 학습 vs 독립 학습 gFID 차이: 최대 1.7 (Nested Dropout), 최소 0.3 미만 (Ours) | Table 1, p.7 |
| ③ | 유사 토큰 병합 → 표현 정렬 가능 | $\|\boldsymbol{W}\boldsymbol{z} - \boldsymbol{z}\|^2$가 유사 토큰 병합 시 최소화됨을 수식으로 증명 | Eq. 4-5, p.4 |
| ④ | 데이터 독립적 LGM으로 생성 시 호환성 확보 | Straight-through trick + Agglomerative clustering으로 그래디언트 전파 가능 | Sec. 3.2, p.5 |
| ⑤ | LGM은 데이터 의존적 병합(ToMe)에 비해 생성 품질 크게 향상 | ToMe 방식 gFID: 5.34~4.89, LGM: 5.24~4.27 → 학습 가능 임베딩+정렬 손실 추가 시 3.19~2.79 | Table 3, p.8 |
| ⑥ | 기존 VLT 대비 gFID-연산량 트레이드오프 우수 | Semanticist, FlexTok 대비 동일 TFLOPs에서 낮은 gFID 달성 | Fig. 4, Table 2, p.7-8 |
| ⑦ | LoRA 파인튜닝으로 독립 학습 수준 성능 근접 | LoRA 적용 시 독립 학습 모델과 0.05 gFID 이내 성능 차이 | Table 1, p.7 |
| ⑧ | 512×512에서도 동일 효과 확인 | 512×512 rFID, CKNNA, gFID 모두 Nested Dropout 대비 향상 | Appendix C, Fig. 8-9, Table 5 |

> **💡 용어 설명**
> - **CKNNA (Centered Kernel Nearest-Neighbor Alignment)**: 두 표현 공간 간 유사도 구조의 일치도를 측정하는 지표. 1에 가까울수록 두 표현 간 유사도 구조가 일치함
> - **LoRA (Low-Rank Adaptation)**: 기존 모델 가중치는 고정하고 저랭크 행렬만 학습하는 경량 파인튜닝 기법
> - **ToMe (Token Merging)**: Bolya et al. (2023)이 제안한 데이터 의존적 토큰 병합 기법으로, 각 이미지마다 병합 패턴이 다름

---

## 2-1. 상세 설명

### 2-1-A. 해결하고자 하는 문제

기존 VLT(주로 Nested Dropout 방식)는 토큰 수가 줄어들 때 **cross-length similarity structure shift**가 발생한다.

수식적으로, 길이 $N$인 전체 잠재 표현 $\boldsymbol{z} \in \mathbb{R}^{N \times D}$에서 Nested Dropout으로 $K$개 토큰을 제거하면:

$$\tilde{\boldsymbol{z}} = \text{Trunc}[\boldsymbol{z}, K] = (z_1, z_2, \ldots, z_{N-K}) \in \mathbb{R}^{(N-K) \times D}$$

이때 초기 토큰은 고수준 의미(semantics), 후기 토큰은 저수준 디테일을 인코딩하므로, 길이가 달라지면 **데이터포인트 간 유사도 구조** 자체가 근본적으로 변한다. 이는 확산 모델이 학습해야 할 score function이 토큰 수마다 달라지는 문제를 야기한다.

**Fig. 1 상단**: Nested Dropout의 CKNNA는 0.13~0.51로 매우 낮음 (독립 학습 토크나이저 간 CKNNA 0.51보다도 낮은 경우 존재)

**Fig. 1 하단**: 공동 학습(Joint) DiT가 독립 학습(Separate) DiT보다 gFID가 현저히 나쁨 (예: 32토큰에서 5.21 vs 4.35)

> **💡 용어 설명**
> - **Similarity Structure**: 여러 데이터 포인트들 사이의 쌍별(pairwise) 유사도 패턴. 이 구조가 토큰 길이에 따라 일관되어야 하나의 DiT가 여러 길이에서 잘 동작할 수 있음

---

### 2-1-B. 제안하는 방법 및 수식

#### ① 병합 기반 길이 조절 (Sec. 3.1)

**할당 행렬(Assignment Matrix)** $\Gamma \in \{0,1\}^{N \times (N-K)}$가 주어질 때 (여기서 $\Gamma_{ij}=1$은 $i$번째 토큰이 $j$번째 클러스터에 할당됨을 의미):

$$\tilde{\boldsymbol{z}} = \bar{\Gamma}^\top \boldsymbol{z}, \quad \text{where} \quad \bar{\Gamma}_{ij} = \frac{\Gamma_{ij}}{\sum_k \Gamma_{kj}} \tag{Eq. 2}$$

- $\tilde{\boldsymbol{z}} \in \mathbb{R}^{(N-K) \times D}$: 병합된 토큰 (각 클러스터의 평균)
- 병합 토큰 크기: $m_j = \sum_i \Gamma_{ij}$ (각 병합 토큰이 포함하는 원본 토큰 수)

> **💡 용어 설명**
> - **할당 행렬 (Assignment Matrix)**: 어떤 원본 토큰이 어떤 클러스터(병합 그룹)에 속하는지를 나타내는 이진 행렬

**비례 어텐션(Proportional Attention)**:

$$\boldsymbol{A} = \text{Softmax}\!\left(\frac{\boldsymbol{Q}\boldsymbol{K}^\top}{\sqrt{d}} + \log \boldsymbol{m}\right)$$

- $\boldsymbol{Q}, \boldsymbol{K}$: 쿼리, 키 행렬
- $d$: 어텐션 헤드 차원
- $\log \boldsymbol{m}$: 각 병합 토큰의 유효 크기에 따른 보정 항

> **💡 용어 설명**
> - **Proportional Attention**: 병합된 토큰이 실제로 몇 개의 원본 토큰을 대표하는지 ($m_j$)를 어텐션 점수에 반영하여 가중치를 조정하는 변형 어텐션 메커니즘

**동치 관계**: 비례 어텐션으로 $N-K$개 병합 토큰을 처리하는 것은, 투영 행렬 $\boldsymbol{W} \in \mathbb{R}^{N \times N}$으로 정의되는 전체 길이 동치 표현(full-length equivalent representation) $\boldsymbol{W}\boldsymbol{z} \in \mathbb{R}^{N \times D}$에 표준 어텐션을 적용하는 것과 동치:

$$W_{ij} = \begin{cases} \frac{1}{|C_i|} & \text{if } j \in C_i \\ 0 & \text{otherwise} \end{cases} \tag{Eq. 3}$$

- $C_i$: 토큰 $i$와 같은 클러스터에 속한 토큰들의 집합
- 이 동치 덕분에 원본 $\boldsymbol{z}$와 $\boldsymbol{W}\boldsymbol{z}$가 동일한 차원 $N$을 가져 **직접 정렬 손실** 계산 가능

**표현 정렬 손실 (Representation Alignment Loss)**:

$$\|\boldsymbol{W}\boldsymbol{z} - \boldsymbol{z}\|^2 = \sum_{i=1}^{N} \|z_i - (\boldsymbol{W}\boldsymbol{z})_i\|^2 \tag{Eq. 4}$$

$$= \frac{1}{2} \sum_{i=1}^{N} \frac{1}{|C_i|} \sum_{j \in C_i} \|z_i - z_j\|^2 \tag{Eq. 5}$$

**핵심 통찰**: Eq. 5는 같은 클러스터 내 토큰들이 서로 유사할수록 최소화된다. 즉, **유사한 토큰을 병합하면 표현 정렬이 자동으로 달성**된다.

> **💡 용어 설명**
> - **Full-length equivalent representation ($\boldsymbol{W}\boldsymbol{z}$)**: 병합된 토큰들을 원래 토큰 수 $N$으로 "펼쳐놓은" 등가 표현. 같은 클러스터에 속하는 위치들은 동일한 값(클러스터 평균)을 가짐

---

#### ② 학습 가능한 전역 병합 (Learnable Global Merging, Sec. 3.2)

데이터 독립적 학습 임베딩 $\boldsymbol{e} = \{e_1, \ldots, e_N\} \in \mathbb{R}^{N \times D}$를 도입하여 병합 패턴을 결정:

$$\Gamma = \text{Agglomerative}(\boldsymbol{e}) \in \{0,1\}^{N \times (N-K)} \tag{Eq. 6}$$

$$\tilde{\boldsymbol{z}} = \bar{\Gamma}^\top \boldsymbol{z} \quad \text{where} \quad \bar{\Gamma}_{ij} = \frac{\Gamma_{ij}}{\sum_k \Gamma_{kj}} \tag{Eq. 7}$$

- $\boldsymbol{e}$: 위치별 학습 가능 임베딩 (데이터와 무관)
- $\text{Agglomerative}(\cdot)$: 코사인 유사도 기반 계층적 클러스터링

> **💡 용어 설명**
> - **Agglomerative (Hierarchical) Clustering**: 가장 가까운 두 클러스터를 반복적으로 병합하는 계층적 군집화 알고리즘. 이산적(discrete) 연산이므로 직접 역전파 불가
> - **데이터 독립적(Data-Independent)**: 병합 패턴이 입력 이미지에 의존하지 않고, 오직 학습된 위치 임베딩에만 의존. 따라서 생성 시에도 동일한 패턴 사용 가능

**Straight-Through Trick** (이산 연산의 그래디언트 문제 해결):

클러스터 중심: $\boldsymbol{c} = \bar{\Gamma}^\top \boldsymbol{e}$ $\tag{Eq. 8}$

소프트 할당: $\Gamma^{\text{soft}} = \text{softmax}\!\left(\frac{\boldsymbol{e}\boldsymbol{c}^\top}{\tau}\right)$ $\tag{Eq. 9}$

스트레이트-스루 적용: $\Gamma \to [\Gamma - \Gamma^{\text{soft}}]_{\text{sg}} + \Gamma^{\text{soft}}$ $\tag{Eq. 10}$

$$\tilde{\boldsymbol{z}} = \left([\Gamma - \Gamma^{\text{soft}}]_{\text{sg}} + \Gamma^{\text{soft}}\right)^\top \boldsymbol{z} \tag{Eq. 11}$$

- $[\cdot]_{\text{sg}}$: stop-gradient 연산자 (역전파 차단)
- $\tau$: 스케일링 파라미터 (소프트맥스 온도)
- **포워드 패스**: 이산 $\Gamma$ 사용 (정확한 병합)
- **백워드 패스**: 연속 $\Gamma^{\text{soft}}$ 통해 그래디언트 전파

> **💡 용어 설명**
> - **Straight-Through Trick (Bengio et al., 2013)**: 이산(discrete) 연산의 역전파 불가 문제를 해결하기 위해, 포워드 패스에는 이산 값을 쓰고 백워드 패스에는 연속 근사값의 그래디언트를 사용하는 기법
> - **Stop-gradient ($[\cdot]_{\text{sg}}$)**: 해당 텐서를 통한 그래디언트 역전파를 차단하는 연산

**정렬 손실 (Alignment Loss)**:

$$\mathcal{L}_{\text{align}} = \sum_{i,j} \text{ReLU}\!\left(\left\lvert \left[\frac{z_i \cdot z_j}{\|z_i\|\|z_j\|}\right]_{\text{sg}} - \frac{e_i \cdot e_j}{\|e_i\|\|e_j\|} \right\rvert - \delta\right)$$

- $z_i \cdot z_j / (\|z_i\|\|z_j\|)$: 잠재 토큰 $i, j$ 간 코사인 유사도 (stop-gradient 적용)
- $e_i \cdot e_j / (\|e_i\|\|e_j\|)$: 학습 임베딩 $i, j$ 간 코사인 유사도
- $\delta$: 과도한 정규화 방지 마진 (margin)
- **역할**: 학습 임베딩의 유사도 구조가 잠재 토큰의 유사도 구조를 반영하도록 유도하여, 결과적으로 유사한 토큰끼리 병합되게 함

**최종 학습 목적 함수**:

$$\min_{\boldsymbol{e}, \mathcal{E}, \mathcal{D}} \mathcal{L}_{\text{total}} + \lambda_{\text{align}} \mathcal{L}_{\text{align}} \tag{Eq. 12}$$

- $\mathcal{L}_{\text{total}}$: 재구성 + 지각 + 적대적 + 정규화 손실 (Eq. 1)
- $\lambda_{\text{align}}$: 정렬 손실 가중치
- $\mathcal{E}, \mathcal{D}$: 인코더, 디코더

---

#### ③ 확산 트랜스포머 적용 (Sec. 3.3)

**확산 학습 목적 함수**:

$$\mathcal{L}_{\text{diff}}(\psi) = \mathbb{E}_{K, \tilde{\boldsymbol{z}}} \|G_\psi(\tilde{\boldsymbol{z}}_t, t) - \epsilon_t\|^2 \tag{Eq. 13}$$

- $G_\psi$: 확산 트랜스포머 (DiT) 파라미터 $\psi$
- $\tilde{\boldsymbol{z}}_t$: 시간 $t$에서 노이즈가 추가된 병합 잠재 표현
- $\epsilon_t$: 추가된 노이즈
- $K$: 무작위 샘플링된 토큰 감소량

**병합 위치 임베딩**:

$$\tilde{\boldsymbol{pe}} = \bar{\Gamma}^\top \boldsymbol{pe} \tag{Eq. 14}$$

- $\boldsymbol{pe} \in \mathbb{R}^{N \times D}$: 학습 가능한 위치 임베딩
- 잠재 토큰과 동일한 방식으로 병합하여 병합 토큰의 위치 정보를 효과적으로 집계

---

### 2-1-C. 모델 구조

```
[입력 이미지 x ∈ R^{H×W×3}]
        ↓
[ViT 인코더 (SoftVQ 기반)]
  - 이미지 패치 + N개 학습 가능 잠재 토큰
  - 소프트 양자화 (코드북 C)
        ↓
[잠재 토큰 z ∈ R^{N×D}]  ←── 학습 가능 임베딩 e ∈ R^{N×D}
        ↓                              ↓
[Agglomerative Clustering (e 기반)]
        ↓
[할당 행렬 Γ + Straight-Through Trick]
        ↓
[병합 토큰 z̃ ∈ R^{(N-K)×D}]
        ↓
┌─────────────┬─────────────┐
↓             ↓             
[ViT 디코더]  [LightningDiT-XL]
[비례 어텐션]  [비례 어텐션 + 병합 위치 임베딩]
[재구성 x̂]   [가변 길이 확산 생성]
              ↓
         [길이별 LoRA 파인튜닝 (선택)]
```

**주요 하이퍼파라미터**:
- 패치 크기: 16, 잠재 차원: 32, 최대 토큰 수: 256
- 토크나이저: ViT-B 기반, 배치 256, 25 에폭
- DiT: LightningDiT-XL (675M), AdamW lr= $2\times10^{-4}$ , 배치 1024

---

### 2-1-D. 성능 향상 및 한계

**성능 향상**:

| 측면 | 수치 |
|---|---|
| CKNNA (표현 정렬) | Nested Dropout 평균 0.34 → Ours 0.76 (Fig. 3) |
| 공동학습 gFID 갭 | Nested Dropout 0.8~1.7 → Ours <0.3 (Table 1) |
| vs Semanticist (25스텝, 32토큰) | 유사 TFLOPs에서 더 낮은 gFID (Fig. 4) |
| 512×512 32토큰 gFID | Nested 5.65 → Ours 3.99 (Table 5) |

**한계**:
1. 1D와 2D 토크나이저 간 생성 품질 격차 존재 (Table 2에서 VA-VAE+LightningDiT-XL이 gFID 1.35로 더 우수)
2. LGM은 데이터 독립적이므로 이미지별 최적 병합(ToMe 방식) 대비 재구성 품질 소폭 열세 (Fig. 6)
3. ImageNet 256×256 단일 벤치마크에만 주요 실험 집중
4. 학습 가능 임베딩 수 $N$이 고정되어 있어 임의의 토큰 수 조합에 대한 확장성 미검증

---

## 3. 각 주장에 위치 표시

| 주장 | 위치 |
|---|---|
| Nested Dropout이 CKNNA를 훼손함 | p.3, Fig. 1 (상단) |
| 공동 학습 DiT의 gFID 저하 | p.3, Fig. 1 (하단), Table 1 (p.7) |
| 병합 기반 정렬의 수학적 정당화 | p.4, Eq. 4-5, Sec. 3.1 |
| LGM의 필요성 (데이터 의존적 병합의 문제) | p.5, Sec. 3.2 도입부 |
| Straight-Through Trick 적용 | p.5, Eq. 8-11 |
| 정렬 손실 정의 | p.5, Alignment loss 절 |
| 비례 어텐션과 병합 위치 임베딩 | p.4, p.6, Sec. 3.1, 3.3 |
| CKNNA 향상 (0.76 vs 0.34) | p.7, Fig. 3 |
| 공동학습 gFID 갭 감소 (<0.3) | p.7, Table 1 |
| 기존 VLT 대비 우수한 트레이드오프 | p.7-8, Fig. 4, Table 2 |
| 소거 연구 (ablation) | p.8, Table 3, Fig. 6 |
| 병합 토큰 유사도 분석 | p.9, Table 4 |
| 512×512 확장 결과 | Appendix C, Fig. 8-9, Table 5 |

---

## 4. 저자 보고 결과 vs. 독립 해석 분리

### 저자가 직접 보고한 결과

**연구 주제**: 토큰 병합을 통한 가변 길이 토크나이저가 표현 정렬을 개선하여 확산 트랜스포머의 가변 길이 공동 학습을 용이하게 함

**방법 (수식)**:
- 병합 조작: $\tilde{\boldsymbol{z}} = \bar{\Gamma}^\top \boldsymbol{z}$ (Eq. 2)
- 전체 길이 동치: $\boldsymbol{W}\boldsymbol{z}$, 정렬 가능성의 핵심 (Eq. 3)
- 표현 이동 측정: $\|\boldsymbol{W}\boldsymbol{z} - \boldsymbol{z}\|^2 = \frac{1}{2}\sum_i \frac{1}{|C_i|}\sum_{j \in C_i}\|z_i - z_j\|^2$ (Eq. 5)
- LGM + Straight-Through (Eq. 6-11)
- 정렬 손실 $\mathcal{L}_{\text{align}}$

**저자 보고 결과**:
- CKNNA: Nested Dropout 평균 0.34 → Ours 0.76 (Fig. 3, p.7)
- gFID 갭: Nested 0.8~1.7 → Ours <0.3 (Table 1)
- LoRA 적용 시 독립학습 대비 0.05 이내 (Table 1)
- 시스템 비교: 25스텝/32토큰 gFID 2.89, 128토큰 gFID 2.03; 100스텝/32토큰 gFID 2.54 (Table 2)
- LGM merged pair 유사도 0.669~0.858 (vs 전체 평균 0.142) (Table 4)
- 512×512 32토큰 gFID: Nested 5.65 → Ours 3.99 (Table 5)

---

### 독자적 해석

1. **이론적 우아함 vs. 근사 최적성**: Eq. 5의 최소화는 이론적으로 각 이미지에 맞는 최적 클러스터링을 요구하지만, LGM은 전역적으로 고정된 패턴을 학습한다. 저자들도 "최적 해는 아님(does not provide an optimal solution)"임을 인정(p.5)한다. 실제로 Table 4에서 ToMe(이미지별 최적 병합)가 LGM보다 높은 유사도를 보이는 것이 이를 확인시켜 준다.

2. **CKNNA의 한계**: CKNNA는 배치 내 최근접 이웃 기반 지표로, 배치 크기와 구성에 민감할 수 있다. 저자들이 이 민감도에 대한 분석을 별도로 제시하지 않아, 수치의 절대적 해석에는 주의가 필요하다.

3. **비교의 불완전성**: Table 2의 시스템 비교는 토크나이저 아키텍처, 생성 모델 계열, 샘플링 예산이 모두 상이하다. 저자들도 "tokenizer-only ablation이 아님"을 명시했으나(p.8), 독자가 이를 순수 방법론 비교로 오해할 여지가 있다.

4. **LoRA의 역할 재해석**: 저자들은 LoRA를 "경미한 오버헤드의 추가 이득"으로 제시하지만, LoRA 적용 후에야 독립 학습 모델 수준에 근접한다는 점에서 LGM 단독의 효과는 Table 1의 "Ours (✓, Joint)" 행이 기준이 되어야 한다.

5. **의의 재해석**: 이 논문의 핵심 기여는 "병합을 통한 전체 길이 동치 표현"이라는 개념적 프레임워크로, 이는 가변 길이 생성 모델 설계의 일반적 원칙으로 확장 가능하다. 단순히 기존 VLT를 개선하는 것 이상으로, 표현 정렬을 위한 새로운 설계 공간을 열었다고 평가한다.

---

## 5. 통계적으로 취약한 부분 및 비교 불가능한 수치

| 항목 | 문제점 | 위치 |
|---|---|---|
| ⚠️ **단일 데이터셋** | ImageNet 256×256 (및 512×512 appendix)에만 실험. 다른 도메인(텍스트-이미지, 의료 이미지 등) 일반화 여부 미검증 | 전체 실험부 |
| ⚠️ **단일 시드 보고** | gFID 수치에 표준편차/신뢰구간 없음. FID는 샘플 수 및 참조 통계에 따라 변동 가능 | Table 1, 2, 3, 5 |
| ⚠️ **Table 2 시스템 비교 불가** | 토크나이저 종류(ViT vs CNN vs Diffusion), 생성 모델 계열(Diffusion vs MAR vs AR), 샘플링 스텝이 모두 다름. 저자도 "quality-compute comparison이지 tokenizer ablation이 아님" 명시 | Table 2, p.8 |
| ⚠️ **CKNNA 배치 의존성** | CKNNA는 배치 내 최근접 이웃 기반으로 배치 크기·구성에 민감. 세부 조건(배치 크기, 이미지 수) 미보고 | Fig. 1, Fig. 3 |
| ⚠️ **FlexTok과의 FLOPs 비교** | FlexTok은 AR 생성 모델(1.33B~1.4B 파라미터), Ours는 Diffusion(675M). 모델 크기 차이로 인해 FLOPs 비교가 공정하지 않을 수 있음 | Fig. 4, Table 2 |
| ⚠️ **Semanticist와의 throughput 비교** | Semanticist-L의 throughput이 0.015 imgs/s로 매우 낮은 이유가 MAR 생성기 특성인지 다른 요인인지 불명확 | Table 2 |
| ⚠️ **LoRA 오버헤드의 상대적 기준** | "2.5% 학습 오버헤드, 2.4~3.4% LoRA 파라미터"는 특정 설정(길이 3개)에 한정된 수치 | p.6, 각주 4 |
| ⚠️ **512×512 실험의 제한성** | 512×512는 DiT-B 모델로만 검증. 메인 실험과 동일한 DiT-XL 수준의 검증 없음 | Appendix C |
| ⚠️ **$\lambda_{\text{align}}$ 민감도 미보고** | 정렬 손실 가중치 $\lambda_{\text{align}}$ 값 및 민감도 분석 없음 | Sec. 3.2 |
| ⚠️ **마진 $\delta$ 설정 근거 부재** | $\mathcal{L}_{\text{align}}$의 마진 $\delta$ 값과 그 선택 기준 미제시 | Sec. 3.2 |

---

## 6. 문서가 답하지 않는 질문

1. **텍스트 조건부 생성에서의 성능**: 클래스 조건부 생성(ImageNet)만 검증. 텍스트-이미지 생성이나 다른 모달리티로의 확장 가능성 및 성능 미보고.

2. **최적 토큰 수 결정 기준**: 어떤 이미지가 몇 개의 토큰을 사용해야 하는지에 대한 자동 또는 콘텐츠 적응적 결정 메커니즘이 없음 (연산 예산은 사용자가 수동으로 지정).

3. **$\lambda_{\text{align}}$과 $\delta$ 하이퍼파라미터 민감도**: 정렬 손실 가중치와 마진 값의 선택이 최종 성능에 얼마나 영향을 미치는지 분석 없음.

4. **학습 가능 임베딩 $\boldsymbol{e}$의 시각화**: 학습된 병합 패턴이 실제로 의미론적으로 일관된 토큰 그룹을 형성하는지 시각적 분석 부재.

5. **더 많은 토큰 길이 수의 확장성**: 현재 {32, 64, 96, 128, 256} 토큰 수 범위만 실험. 더 넓은 범위나 연속적 길이 조절에서의 동작 미검증.

6. **비디오 생성으로의 확장**: 시간 축이 추가되는 비디오 생성에서 가변 길이 토크나이저가 동일하게 유효한지 불명확.

7. **토크나이저와 DiT의 공동 학습(end-to-end training)**: 현재는 토크나이저와 DiT를 분리 학습. 공동 최적화 시 추가 이득 가능성 미탐색.

8. **LGM의 클러스터링 알고리즘 선택 민감도**: Agglomerative clustering 외 K-means 등 다른 클러스터링 방법과의 비교 없음.

9. **매우 낮은 토큰 수(예: 8, 16)에서의 성능**: 극단적 압축 시나리오에 대한 실험 없음.

10. **1D vs 2D 토크나이저 품질 격차의 근본 원인**: 1D 토크나이저가 2D 대비 gFID가 낮은 근본적 이유와 해결 방향에 대한 심층 분석 없음.

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.3): Nested Dropout의 문제 정량화

**구성**: 상단 - CKNNA 막대 그래프, 하단 - gFID 비교

**해석**:
- **상단**: Nested Dropout 토크나이저의 전체 길이(N=256) 잠재 표현과 단축 표현(M<256) 사이의 CKNNA가 32토큰에서 0.13, 256토큰(자기 자신)에서 1.0으로 급격히 감소함. 독립 학습 토크나이저 간 CKNNA(0.51, 검정 점선)보다도 낮은 값이 다수. **이는 Nested Dropout이 같은 토크나이저의 서로 다른 길이 표현들을 오히려 독립적으로 학습된 별개 토크나이저들보다도 더 이질적으로 만든다는 충격적 결과.**
- **하단**: 파란 막대(독립 학습)는 더 많은 토큰일수록 gFID 감소(품질 향상). 반면 빨간 막대(공동 학습)는 단조감소하지 않고 들쑥날쑥하며, 256토큰에서조차 독립학습 32토큰보다 나쁜 경우(5.14 vs 4.35)까지 발생. **이는 표현 정렬 실패가 더 많은 토큰을 써도 오히려 성능이 나빠지는 역설을 야기함을 보여줌.**

> **연구 동기의 핵심 증거**: 이 그림이 논문 전체의 문제 설정을 정당화하는 가장 중요한 실험적 근거.

---

### Figure 2 (p.4): 방법론 전체 개요도

**구성**: 인코더-LGM-디코더/DiT의 파이프라인 도식

**해석**:
- 왼쪽: ViT 인코더가 이미지를 N개의 초기 잠재 토큰으로 변환
- 중앙(3.2 박스): 학습 가능 임베딩 $\boldsymbol{e}$에서 Agglomerative Clustering → Straight-Through Trick 적용 → 할당 행렬 $\Gamma$ 생성
- **핵심**: 학습 임베딩이 이미지와 무관하게 병합 패턴을 결정하므로, **생성 시에도 동일한 패턴 $\Gamma$를 사전에 알 수 있음**
- 오른쪽: 인코딩된 잠재를 병합 → Mask Token과 함께 디코더로 재구성, 또는 DiT로 생성

> **직관적 이해**: LGM은 "위치 기반 고정 병합 패턴"을 학습하는 것. 마치 이미지 패치에서 공간적으로 인접하고 유사한 토큰들을 미리 정해진 규칙으로 묶는 것과 유사하지만, 그 규칙이 데이터 분포로부터 최적화됨.

---

### Figure 3 (p.7): CKNNA 비교 (Ours vs Nested Dropout)

**구성**: 4개의 서브플롯, 각각 다른 기준 길이(256, 128, 64, 32)에서 다른 길이와의 CKNNA 비교

**해석**:
- 파란 선(Ours): 기준 길이와 다른 길이 간 CKNNA가 대부분 0.75 이상으로 매우 높음. 특히 **독립 학습 토크나이저 기준선(수평 점선)을 지속적으로 상회**함. 이는 LGM이 단순히 같은 아키텍처 내부에서의 일관성을 넘어, **서로 다른 길이의 잠재 공간이 실질적으로 정렬됨**을 의미.
- 빨간 선(Nested Dropout): 같은 길이에서는 1.0이지만, 길이가 달라지면 0.4 이하로 급격히 감소. 기준선을 하회하는 경우도 다수.
- **평균 CKNNA: Ours 0.76 vs Nested 0.34** - 2배 이상 차이

> **통계적 주의**: CKNNA 수치가 측정 배치 구성에 의존할 수 있으며, 배치 크기 등 측정 조건이 명시되지 않음. 추세는 명확하나 절대 수치의 해석에 주의 필요.

---

### Figure 4 (p.7): gFID-연산량 트레이드오프 비교

**구성**: (a) Throughput vs gFID, (b) TFLOPs vs gFID

**해석**:
- Ours(초록)는 25스텝, 32/64/128 토큰 세 점을 커버하며 Pareto 프론티어를 형성
- **Semanticist-L**(보라 삼각형): throughput이 매우 낮음(0.015 imgs/s). MAR 생성기 특성상 느린 것으로 보이나, 원인이 명시되지 않음 ⚠️
- **FlexTok-d18**(빨간 역삼각형): FLOPs 면에서는 경쟁적이나, Ours 대비 gFID가 높음
- **SoftVQ-B-32**(파란 사각형): 고정 길이이므로 토큰 수 조절 불가. 단일 점으로 표시됨

> **비교 가능성 주의**: ⚠️ FlexTok은 AR 모델(1.33B+), Ours는 DiT(675M)으로 모델 복잡도가 다름. 단순한 FLOPs 비교가 공정한 효율성 지표인지 불확실.

---

### Figure 5 (p.8): 재구성 품질 비교 (rFID, PSNR, SSIM)

**구성**: 세 지표 모두 토큰 수(32~256)에 따른 변화

**해석**:
- **Ours(다이아몬드)**: 세 지표 모두에서 전반적으로 최상위권. 특히 고압축(32~64 토큰)에서 두드러진 우위
- **Semanticist(원)**: rFID는 32토큰에서 Ours와 유사하나, PSNR과 SSIM은 매우 낮음. **이유**: Semanticist는 diffusion 기반 디코더를 사용해 정확한 픽셀 재구성이 아닌 지각적으로 그럴듯한 이미지를 생성함. rFID가 좋아도 PSNR/SSIM이 나쁜 것은 이 때문
- **One-D-Piece(사각형)**: 전반적으로 Ours보다 낮은 성능, 특히 고압축에서 차이 두드러짐
- **Nested-dropout(삼각형)**: 같은 아키텍처에서도 LGM 대비 낮은 성능

> **핵심 교훈**: rFID만으로 재구성 품질을 평가하면 diffusion decoder를 쓰는 방법이 유리하게 보일 수 있으나, PSNR/SSIM로 정확한 픽셀 재구성 능력을 함께 평가해야 공정한 비교가 가능.

---

## 8. 결론 및 후속 연구

### 8-1. 저자가 제시한 시사점 및 후속 연구

**저자 제시 시사점**:
1. 토큰 병합이 가변 길이 토크나이저에서 표현 정렬을 보장하는 원리적으로 타당한 접근임을 수학적으로 정당화
2. LGM은 데이터 독립적 병합 패턴 학습을 통해 확산 모델과의 호환성 확보
3. 경량 LoRA 파인튜닝이 추가 오버헤드 없이 성능을 효과적으로 향상시킬 수 있음
4. 1D VLT와 2D 토크나이저 간 품질 격차가 여전히 존재하며, 하이브리드 아키텍처 탐색이 유망

**저자 명시 후속 연구 방향** (p.8, 9):
- 1D와 2D 토크나이저의 장점을 결합한 **하이브리드 VLT 아키텍처** 탐색
- 다른 생성 모델(AR, MAR 등)과의 결합 가능성 언급(Impact Statement)

---

### 8-1-A. 모델의 일반화 성능 향상 가능성

**현재 일반화의 한계**:
- ImageNet 단일 데이터셋, 클래스 조건부 생성만 검증
- 고정된 토큰 수 집합 {32, 64, 96, 128, 256}만 사용
- 도메인 특이적 시각적 복잡도 차이를 반영하지 않음

**일반화 성능 향상을 위한 구체적 방향**:

**① 콘텐츠 적응적 토큰 할당 (Content-Adaptive Token Allocation)**

현재 LGM은 모든 이미지에 동일한 병합 패턴을 적용. 이미지 복잡도에 따라 토큰 수를 동적으로 결정하는 메커니즘 추가 시 일반화 향상 기대:

$$K^* = \arg\min_K \left[\alpha \cdot \text{complexity}(\boldsymbol{x}) + \beta \cdot K\right]$$

여기서 $\text{complexity}(\boldsymbol{x})$는 이미지 복잡도 추정기 (예: gradient magnitude, semantic entropy).

**② 도메인 일반화를 위한 메타 학습**

ImageNet 이외 데이터셋에서 학습 가능 임베딩 $\boldsymbol{e}$가 도메인 독립적인 구조를 학습하도록, 다중 도메인 데이터로 메타 학습(MAML 류) 적용 가능.

**③ 표현 정렬 손실의 강화**

현재 $\mathcal{L}_{\text{align}}$은 토큰 쌍별 코사인 유사도만 비교. 더 풍부한 구조 정보(예: 그래프 구조, 위상적 특징)를 포함하는 정렬 손실 설계:

$$\mathcal{L}_{\text{align}}^{\text{enhanced}} = \mathcal{L}_{\text{align}} + \gamma \cdot \mathcal{L}_{\text{topological}}$$

**④ 연속적 토큰 수 조절 (Continuous Length Modulation)**

현재 이산적인 토큰 수 집합 대신, 연속적인 압축률 파라미터 $r \in [0,1]$로 조절하는 방식으로 확장하면 임의의 계산 예산에 대응 가능.

**⑤ 멀티모달 확장**

LGM의 데이터 독립적 병합 원리는 텍스트-이미지, 비디오, 3D 포인트 클라우드 등 다양한 모달리티에 원칙적으로 적용 가능. 표현 정렬 프레임워크가 크로스 모달 일관성에도 기여할 수 있음.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

#### 관련 연구 계보

| 연구 | 연도 | 핵심 기여 | 본 논문과의 관계 |
|---|---|---|---|
| VQGAN (Esser et al.) | 2021 | 트랜스포머 기반 이미지 토크나이저 표준화 | 기반 토크나이저 아키텍처 |
| LDM (Rombach et al.) | 2022 | 잠재 공간 확산 모델 표준화 | LDM 프레임워크 채택 |
| Matryoshka RL (Kusupati et al.) | 2022 | 정렬된 표현 학습 (Nested 원리) | Nested Dropout의 기반 원리 |
| ToMe (Bolya et al.) | 2023 | ViT 추론 가속을 위한 토큰 병합 | LGM의 핵심 아이디어 출발점 (단, 데이터 의존적) |
| TiTok (Yu et al.) | 2024 | 1D 이미지 토크나이저 (32토큰) | 1D 토크나이저 아키텍처 참조 |
| ATC (Haurum et al.) | 2024 | Agglomerative Token Clustering | LGM의 클러스터링 방법 채택 |
| SoftVQ (Chen et al.) | 2025 | 효율적 1D 연속 토크나이저 | 본 논문의 토크나이저 백본 |
| LightningDiT (Yao et al.) | 2025 | 재구성-생성 정렬 최적화된 DiT | 본 논문의 DiT 아키텍처 |
| REPA (Yu et al.) | 2025 | DiT 학습을 위한 표현 정렬 | 표현 정렬 개념 공유, 단 고정 길이 |
| One-D-Piece (Miwa et al.) | 2025 | Nested Dropout 기반 VLT | 주요 비교 대상 |
| Semanticist (Wen et al.) | 2025 | PCA 기반 VLT | 주요 비교 대상 |
| FlexTok (Bachmann et al.) | 2025 | 유연한 길이의 1D AR 토크나이저 | 주요 비교 대상 |
| **Ours (Lee & Hong)** | **2026** | **LGM 기반 VLT for DiT** | **본 논문** |

> **⚠️ 참고**: 상기 표의 "2025" 논문들은 본 논문(ICML 2026)이 인용한 동시대 연구들로, 이 중 일부는 본 논문 심사 과정에서 출판된 최신 연구임. 독자적 검증 없이 논문 인용 정보에만 기반함.

#### 본 논문의 차별점 분석

**vs. ToMe (Bolya et al., 2023)**: ToMe는 ViT 추론 가속을 위한 데이터 의존적 병합. 본 논문은 이를 생성 모델에 적용하기 위해 데이터 독립적으로 재설계. 단순 가속 도구를 표현 정렬 프레임워크로 격상시킨 개념적 도약.

**vs. REPA (Yu et al., 2025)**: REPA는 고정 길이 DiT 학습 시 표현 정렬 손실 적용. 본 논문은 가변 길이 시나리오에서 차원이 다른 표현들 간의 정렬 문제를 해결.

**vs. Semanticist/One-D-Piece**: 두 방법 모두 Nested Dropout 기반으로 cross-length shift 문제 미해결. 본 논문은 이를 원론적으로 해결.

#### 앞으로의 연구에 미치는 영향

1. **VLT 설계 패러다임 전환**: "잘라내기(truncation)"에서 "병합(merging)"으로의 패러다임 전환을 제시. 향후 VLT 연구가 병합 기반 접근을 검토할 가능성 높음.

2. **표현 정렬의 수학적 프레임워크**: $\boldsymbol{W}\boldsymbol{z}$라는 전체 길이 동치 표현 개념은 가변 길이 표현 학습 전반에 활용 가능한 일반적 도구.

3. **멀티스케일 생성 모델**: 단일 모델로 다양한 품질-속도 트레이드오프를 제어하는 연구 촉진.

4. **효율적 배포**: 엣지 디바이스부터 서버까지 동일 모델로 대응하는 적응형 생성 시스템 설계에 기여.

#### 앞으로 연구 시 고려할 점

1. **벤치마크 다양화**: ImageNet을 넘어 COCO, LAION 등 다양한 도메인에서 검증 필요. 특히 자연스럽게 복잡도가 다양한 데이터셋에서 VLT의 이점이 극대화될 가능성.

2. **공정한 비교 기준 수립**: VLT 연구의 공정한 비교를 위해 동일 백본·동일 생성 모델·동일 학습 예산 하에서의 표준 벤치마크 필요.

3. **이론적 수렴 분석**: LGM의 학습 안정성과 수렴 특성에 대한 이론적 분석이 부재. 특히 Straight-Through Trick의 편향(bias) 문제가 실제 학습에 미치는 영향 분석 필요.

4. **확산 모델 이외 생성 모델**: MAR, AR, Flow Matching 등 다른 생성 패러다임과의 결합 가능성 탐색. 특히 토큰 수에 독립적으로 동작하는 모델(예: 마스크 기반 생성)과의 시너지 가능.

5. **동적 토큰 예산 할당**: 입력 이미지 복잡도에 따라 토큰 수를 자동으로 결정하는 메커니즘 개발 시, 본 논문의 프레임워크가 자연스러운 확장 기반이 됨.

---

## 참고 자료

**논문 자체 (Primary Source)**:
- Lee, D. H., & Hong, S. (2026). "Variable-Length Tokenization via Learnable Global Merging for Diffusion Transformers." *Proceedings of the 43rd International Conference on Machine Learning (ICML)*. arXiv:2606.20076v1.

**논문 내 인용 참고문헌 (본 분석에 활용)**:
- Bolya, D., et al. (2023). "Token Merging: Your ViT but Faster." *ICLR 2023*.
- Rombach, R., et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." *CVPR 2022*.
- Kusupati, A., et al. (2022). "Matryoshka Representation Learning." *NeurIPS 2022*.
- Chen, H., et al. (2025b). "SoftVQ-VAE: Efficient 1-Dimensional Continuous Tokenizer." *CVPR 2025*.
- Yao, J., et al. (2025). "Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models." *CVPR 2025*.
- Yu, S., et al. (2025). "Representation Alignment for Generation: Training Diffusion Transformers is Easier Than You Think." *ICLR 2025*.
- Haurum, J. B., et al. (2024). "Agglomerative Token Clustering." *ECCV 2024*.
- Wen, X., et al. (2025). "'Principal Components' Enable a New Language of Images." *ICCV 2025*.
- Bachmann, R., et al. (2025). "FlexTok: Resampling Images into 1D Token Sequences of Flexible Length." *ICML 2025*.
- Miwa, K., et al. (2025). "One-D-Piece: Image Tokenizer Meets Quality-Controllable Compression." *Tokenization Workshop 2025*.
- Bengio, Y., et al. (2013). "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation." arXiv:1308.3432.
- Huh, M., et al. (2024). "The Platonic Representation Hypothesis." *ICML 2024*.
- Yu, Q., et al. (2024). "An Image is Worth 32 Tokens for Reconstruction and Generation." *NeurIPS 2024*.
- Heusel, M., et al. (2017). "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium." *NeurIPS 2017*.

> **⚠️ 정확도 고지**: 본 분석은 제공된 PDF 원문에 기반하며, 논문이 ICML 2026 출판 예정(arXiv 기준 2026년 6월 제출)임을 감안할 때, 일부 비교 대상 논문(2025년 발표)은 공개 접근이 제한될 수 있습니다. 논문 내 인용 정보 이상의 해당 비교 논문 세부 내용은 직접 검증하지 않았으므로, 비교 분석 부분은 원문 저자 서술에 의존함을 명시합니다.
