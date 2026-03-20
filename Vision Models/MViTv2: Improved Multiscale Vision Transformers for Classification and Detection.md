# MViTv2: Improved Multiscale Vision Transformers for Classification and Detection

## 종합 분석 보고서

---

## 1. 핵심 주장과 주요 기여 요약

MViTv2는 MViTv1(Multiscale Vision Transformers)을 개선한 범용 비전 백본 아키텍처로, **이미지 분류, 객체 검출, 비디오 인식**이라는 세 가지 시각 인식 과제에서 통합적으로 사용될 수 있도록 설계되었다. 핵심 기여는 다음과 같다:

1. **분해된 상대 위치 임베딩(Decomposed Relative Positional Embeddings):** 절대 위치 임베딩의 이동 불변성(shift-invariance) 부재 문제를 해결하여, 토큰 간 상대적 거리만을 기반으로 위치 정보를 인코딩하며, 시공간 축으로 분해하여 계산 효율성을 확보.

2. **잔차 풀링 연결(Residual Pooling Connection):** 풀링된 쿼리 텐서를 어텐션 출력에 잔차 연결하여 풀링 스트라이드에 의한 정보 손실을 보상하고 학습 안정성을 향상.

3. **하이브리드 윈도우 어텐션(Hybrid Window Attention):** 풀링 어텐션과 로컬 윈도우 어텐션을 결합한 효율적 설계로 객체 검출에서 정확도/계산량 트레이드오프 개선.

4. **세 가지 도메인에서의 SOTA 성능:** ImageNet 분류 88.8%, COCO 객체 검출 58.7 AP $^{\text{box}}$, Kinetics-400 비디오 분류 86.1%.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

ViT(Vision Transformer)는 이미지 분류에서 뛰어난 성능을 보여주었으나, 다음과 같은 근본적 한계를 가진다:

- **고해상도 입력의 계산 복잡도:** Self-attention의 복잡도가 토큰 수에 대해 $O(N^2)$으로 증가하여, 고해상도 객체 검출과 비디오 이해에 적용이 어려움.
- **단일 스케일 특징 표현의 한계:** ViT는 네트워크 전체에서 고정 해상도를 사용하여 다중 스케일 특징이 필요한 검출 태스크에 부적합.
- **이동 불변성 부재:** MViTv1의 절대 위치 임베딩은 두 패치 간의 상대적 위치가 동일하더라도 절대 위치에 따라 상호작용이 달라지는 문제를 야기.
- **풀링에 의한 정보 손실:** MViTv1에서 Key, Value에 대한 큰 풀링 스트라이드가 정보 흐름을 제한.

### 2.2 제안하는 방법 (수식 포함)

#### (A) 풀링 어텐션 (Pooling Attention) — MViTv1 기반

입력 시퀀스 $X \in \mathbb{R}^{L \times D}$에 대해 선형 프로젝션과 풀링 연산자 $\mathcal{P}$를 적용:

$$Q = \mathcal{P}_Q(X W_Q), \quad K = \mathcal{P}_K(X W_K), \quad V = \mathcal{P}_V(X W_V)$$

여기서 $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$이고, $Q \in \mathbb{R}^{\tilde{L} \times D}$의 길이 $\tilde{L}$은 $\mathcal{P}_Q$에 의해 축소 가능. 풀링된 self-attention은:

$$Z := \text{Attn}(Q, K, V) = \text{Softmax}\left(QK^\top / \sqrt{D}\right) V$$

#### (B) 분해된 상대 위치 임베딩 (Decomposed Relative Position Embedding)

두 입력 원소 $i$와 $j$ 사이의 상대 위치를 임베딩 $R_{p(i),p(j)} \in \mathbb{R}^d$로 인코딩하고, self-attention에 통합:

$$\text{Attn}(Q, K, V) = \text{Softmax}\left((QK^\top + E^{(\text{rel})}) / \sqrt{d}\right) V$$

$$\text{where} \quad E^{(\text{rel})}_{ij} = Q_i \cdot R_{p(i),p(j)}$$

전체 상대 위치 임베딩 $R_{p(i),p(j)}$의 가능한 수가 $O(TWH)$로 방대하므로, **시공간 축 분해**를 적용:

$$R_{p(i),p(j)} = R^{\text{h}}_{h(i),h(j)} + R^{\text{w}}_{w(i),w(j)} + R^{\text{t}}_{t(i),t(j)}$$

여기서 $R^{\text{h}}, R^{\text{w}}, R^{\text{t}}$는 각각 높이, 너비, 시간 축의 위치 임베딩이며, $h(i), w(i), t(i)$는 토큰 $i$의 각 축 위치. 이 분해를 통해 학습 파라미터 수가 $O(T + W + H)$로 대폭 감소한다. 시간 축 임베딩 $R^{\text{t}}$는 비디오 태스크에서만 사용.

**효과:** Joint 상대 위치 대비 학습 속도 약 3.9배 향상 (COCO 기준), 정확도 동등 유지 (Table 6).

#### (C) 잔차 풀링 연결 (Residual Pooling Connection)

풀링된 쿼리 텐서를 어텐션 출력에 직접 더하는 잔차 연결:

$$Z := \text{Attn}(Q, K, V) + Q$$

출력 시퀀스 $Z$는 풀링된 쿼리 $Q$와 동일한 길이를 가지며, Key/Value의 큰 풀링 스트라이드에 의한 정보 손실을 보상한다. 추가 계산 비용이 거의 없으면서도 ImageNet에서 +0.3%, COCO에서 +1.4 AP $^{\text{box}}$ 개선 (Table 7).

#### (D) 하이브리드 윈도우 어텐션 (Hybrid Window Attention, Hwin)

- FPN에 연결되는 마지막 3개 스테이지의 **마지막 블록에서만 글로벌 어텐션**을 수행
- 나머지 블록에서는 로컬 윈도우 내 어텐션 수행
- Swin의 shifted window와 달리, 간단한 구조로 cross-window 연결을 확보
- Swin 대비 ImageNet에서 +1.7%, COCO에서 일관된 우위 달성

### 2.3 모델 구조

MViTv2는 4단계 계층적(multiscale) 구조를 채택하며, 각 스테이지별로:
- **채널 폭(D)이 점진적으로 증가**: [96 → 192 → 384 → 768] (Base 기준)
- **해상도(시퀀스 길이 L)가 점진적으로 감소**: [ $56^2 → 28^2 → 14^2 → 7^2$ ]

| 변형 | 채널 | 블록 수 | 헤드 수 | FLOPs (G) | Param (M) |
|------|------|--------|--------|-----------|-----------|
| MViT-T | [96-192-384-768] | [1-2-5-2] | [1-2-4-8] | 4.7 | 24 |
| MViT-S | [96-192-384-768] | [1-2-11-2] | [1-2-4-8] | 7.0 | 35 |
| MViT-B | [96-192-384-768] | [2-3-16-3] | [1-2-4-8] | 10.2 | 52 |
| MViT-L | [144-288-576-1152] | [2-6-36-4] | [2-4-8-16] | 39.6 | 218 |
| MViT-H | [192-384-768-1536] | [4-8-60-8] | [3-6-12-24] | 120.6 | 667 |

**객체 검출 통합:** 4-스테이지의 다중 스케일 특징 맵이 FPN(Feature Pyramid Network)에 자연스럽게 연결되어 Mask R-CNN, Cascade Mask R-CNN 등의 검출 프레임워크와 결합.

**비디오 인식 확장:** (1) 패치화 스템에서 시공간 큐브 프로젝션, (2) 시공간 풀링 연산자, (3) 시공간 상대 위치 임베딩의 세 가지 변경만으로 비디오 태스크 적용. ImageNet 사전학습 가중치로부터 inflation 초기화 적용.

### 2.4 성능 향상

#### ImageNet 분류
- **MViTv2-B:** 84.4% (MViTv1-B-24 83.4% 대비 +1.0%, 더 적은 FLOPs/파라미터)
- **MViTv2-S:** 83.6% (MViTv1-B-16 83.0% 대비 +0.6%, 10% 적은 FLOPs)
- **MViTv2-L ↑384²:** 86.3% (IN-1K only, SOTA)
- **MViTv2-H ↑512² (IN-21K):** 88.8%
- Swin-B 대비: +1.1% 정확도, 33% 이상 적은 FLOPs/파라미터

#### COCO 객체 검출
- **MViTv2-B (Mask R-CNN):** 51.0 AP $^{\text{box}}$ (Swin-B 48.5 대비 +2.5)
- **MViTv2-L (Cascade Mask R-CNN, SoftNMS+멀티스케일):** 58.7 AP $^{\text{box}}$ (Swin-L 58.0 대비 +0.7, 더 단순한 검출기 사용)

#### 비디오 인식
- **K400:** MViTv2-S 81.0% (+2.6% over MViTv1), MViTv2-L 86.1%
- **K600:** MViTv2-L 87.9% (SOTA)
- **K700:** MViTv2-L 79.4% (+7.1% over prior best)
- **SSv2:** MViTv2-L 73.3%

#### 런타임 비교 (Table 8)
MViTv2-S는 Swin-B 대비 ImageNet에서 +0.3% 높은 정확도와 23.5% 높은 처리량(341 vs. 276 im/s)을 보이며, COCO에서도 더 빠른 학습 속도(2.7 vs. 2.5 iter/s)와 적은 메모리(5.2G vs. 6.3G)를 달성.

### 2.5 한계

논문에서 명시적으로 언급한 한계:
1. **하이퍼파라미터 최적화 미비:** 각 태스크에 대해 기존 커뮤니티의 표준 레시피를 기반으로 경량 튜닝만 수행하여, 변형별 하이퍼파라미터가 최적이 아닐 수 있음.
2. **스케일링 방향:** 모바일 환경을 위한 더 작은 모델이나, 대규모 데이터 시나리오를 위한 더 큰 모델로의 확장은 미래 과제로 남김.
3. **대형 모델의 과적합 문제:** MViTv2-L은 Kinetics에서 from scratch 학습 시 심각한 과적합이 발생하여 ImageNet 사전학습이 필수적 (Table 14에서 scratch 81.4% vs. IN-21K 84.5%).

---

## 3. 모델의 일반화 성능 향상 가능성

MViTv2의 설계에서 일반화 성능 향상과 직결되는 핵심 요소들을 다음과 같이 분석한다:

### 3.1 분해된 상대 위치 임베딩과 이동 불변성

절대 위치 임베딩은 두 패치의 상호작용이 절대 좌표에 의존하게 만들어, 학습 시 보지 못한 위치 조합에 대한 일반화가 어려웠다. 상대 위치 임베딩은 **이동 불변성(shift-invariance)**이라는 비전의 근본 원리를 Transformer에 도입하여:

- **다양한 해상도에의 전이:** 사전학습 해상도(224²)와 다른 해상도(384², 512²)로의 전이 시 위치 임베딩을 보간(interpolation)하여 적용 가능. 이는 검출 태스크에서 가변 크기 입력을 처리하는 데 핵심적.
- **도메인 간 전이:** 이미지에서 학습된 공간 상대 위치 임베딩을 비디오의 공간 차원에 직접 초기화하고, 시간 축은 0으로 초기화하여 효과적으로 전이.
- **정량적 효과:** 절대 위치 대비 ImageNet +0.1%, COCO +0.6 AP $^{\text{box}}$ 향상 (Table 6). 특히 COCO 같은 전이 학습 시나리오에서 더 큰 이득.

### 3.2 풀링 어텐션의 글로벌 수용 영역

윈도우 어텐션(Swin)이 로컬 영역 내에서만 self-attention을 수행하는 반면, MViTv2의 풀링 어텐션은 **글로벌 self-attention 계산을 유지**하면서 풀링으로 복잡도를 줄인다. 이는:

- 모든 토큰이 전체 특징 맵과 상호작용하므로, 객체의 문맥적(contextual) 관계를 더 잘 포착
- ViT-B 기반 실험에서 풀링 어텐션이 full attention 대비 38% 적은 FLOPs로 유사한 정확도 달성 (Table 4a)
- COCO에서는 오히려 full attention을 능가 (+0.6 AP $^{\text{box}}$, Table 4b)

### 3.3 다중 스케일 계층 구조의 일반화 이점

CNN의 고전적 성공 요인인 다중 스케일 특징 계층을 Transformer에 도입:
- FPN과의 자연스러운 결합으로 단일 스케일 대비 +2.9 AP $^{\text{box}}$ 향상 (Table 9, MViTv2-S)
- ViT-B의 FPN 이득(+1.5 AP $^{\text{box}}$)보다 훨씬 큰 이득은 **네이티브 계층적 설계의 효과**를 입증

### 3.4 사전학습의 효과와 규모별 일반화

| 모델 | Scratch | IN-1K | IN-21K |
|------|---------|-------|--------|
| MViTv2-S (K400) | 81.2 | 82.2 (+1.0) | 82.6 (+1.4) |
| MViTv2-B (K400) | 82.9 | 83.3 (+0.4) | 84.3 (+1.4) |
| MViTv2-L (K400) | 81.4 | 83.4 (+2.0) | 84.5 (+3.1) |

- 소형 모델은 scratch 학습으로도 경쟁력 있는 성능 달성 가능 (일반화 용이)
- 대형 모델은 사전학습 없이 과적합되므로, **대규모 사전학습이 일반화의 필수 조건**
- IN-21K → COCO 전이에서도 대형 모델일수록 이득이 큼 (Table A.5: MViTv2-L +0.9 AP $^{\text{box}}$)

### 3.5 정규화 기법과 학습 안정성

MViTv2는 다양한 정규화를 활용하여 일반화 성능을 확보:
- **Stochastic Depth (Drop Path):** 모델 크기에 따라 0.1~0.8까지 적응적 설정
- **Label Smoothing, Mixup, CutMix, Random Erasing, RandAugment** 등의 데이터 증강
- 잔차 풀링 연결 자체가 정보 흐름을 원활하게 하여 학습 안정성을 제공

### 3.6 다중 태스크 일반화

**동일한 아키텍처 패밀리**가 최소한의 수정으로 3개 도메인(이미지 분류, 객체 검출, 비디오 인식)에서 모두 SOTA를 달성한 것은 MViTv2의 **범용 백본으로서의 일반화 능력**을 강력히 입증한다.

---

## 4. 연구 영향과 향후 고려사항

### 4.1 연구에 미치는 영향

1. **통합 비전 백본의 가능성 입증:** 이미지/비디오/검출에 걸쳐 단일 아키텍처가 SOTA를 달성할 수 있음을 실증적으로 보여주어, "하나의 백본으로 다양한 비전 태스크를 해결"하는 방향의 연구를 가속화.

2. **풀링 어텐션 vs. 윈도우 어텐션 논쟁에 기여:** Swin의 shifted window 방식이 아닌, 글로벌 풀링 기반 어텐션이 동등하거나 우월할 수 있음을 체계적 비교를 통해 입증. 이후 어텐션 메커니즘 설계에 중요한 참조점 제공.

3. **상대 위치 인코딩의 중요성 재확인:** 분해된 상대 위치 임베딩의 효율성과 효과성을 다중 도메인에서 검증하여, 후속 Transformer 설계에서의 표준적 선택지로 자리매김.

4. **효율적 스케일링 레시피 제공:** 5가지 모델 변형(T/S/B/L/H)과 상세 학습 레시피를 공개하여, 실무 연구자들이 다양한 계산 예산에 맞춰 활용할 수 있는 실용적 가이드 제공.

### 4.2 향후 연구 시 고려할 점

1. **모바일/엣지 디바이스로의 경량화:** 현재 가장 작은 MViTv2-T도 4.7G FLOPs로 모바일 배포에는 과대. MobileViT, EfficientFormer 등의 경량 설계와의 결합이나 지식 증류(Knowledge Distillation) 적용이 필요.

2. **자기지도 학습(Self-supervised Learning)과의 결합:** MViTv2는 지도 학습 기반으로만 평가되었으나, MAE(Masked Autoencoder)나 DINO 같은 자기지도 사전학습과 결합 시 일반화 성능의 추가 향상 가능성이 높음.

3. **대규모 모델의 효율적 학습:** MViTv2-H(667M 파라미터)의 학습은 막대한 계산 자원을 요구하며, 더 효율적인 학습 전략(예: progressive resizing, mixed precision training 고도화)이 필요.

4. **세그멘테이션 등 추가 태스크 확장:** 논문에서 인스턴스 세그멘테이션은 다루었으나, 시맨틱 세그멘테이션, 파노프틱 세그멘테이션 등으로의 체계적 확장과 평가가 필요.

5. **주의 메커니즘의 해석 가능성:** 풀링 어텐션이 왜 윈도우 어텐션보다 효과적인지에 대한 이론적 분석과 시각화 연구가 부족하며, 이에 대한 후속 연구가 필요.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 주요 비교 대상 모델

| 모델 | 발표 | 핵심 접근법 | ImageNet Top-1 | COCO AP $^{\text{box}}$ |
|------|------|-----------|---------------|----------------------|
| **ViT** (Dosovitskiy et al., 2020) | ICLR 2021 | 단일 스케일 Transformer, 절대 위치 임베딩 | 85.2% (ViT-L, IN-21K) | - |
| **DeiT** (Touvron et al., 2020) | ICML 2021 | 효율적 학습 레시피, 지식 증류 | 81.8% (DeiT-B) | - |
| **Swin Transformer** (Liu et al., 2021) | ICCV 2021 | Shifted window attention, 계층적 구조 | 87.3% (Swin-L, IN-21K) | 58.0 (HTC++) |
| **PVT/PVTv2** (Wang et al., 2021) | ICCV/arXiv 2021 | Spatial-reduction attention, 피라미드 구조 | 83.8% (PVTv2-B5) | - |
| **CSWin Transformer** (Dong et al., 2021) | arXiv 2021 | Cross-shaped window attention | 87.5% (CSWin-L, IN-21K) | - |
| **CoAtNet** (Dai et al., 2021) | NeurIPS 2021 | Convolution + attention 하이브리드 | 88.4% (CoAtNet-4, IN-21K) | - |
| **ViViT** (Arnab et al., 2021) | ICCV 2021 | Factorized 시공간 attention (비디오) | 81.3% (K400) | - |
| **Video Swin** (Liu et al., 2021) | CVPR 2022 | 3D shifted window (비디오) | 84.9% (K400, IN-21K) | - |
| **MViTv2 (본 논문)** | CVPR 2022 | 풀링 어텐션, 분해 상대 위치, 잔차 풀링 | **88.8%** (MViTv2-H, IN-21K) | **58.7** (Cascade) |

### 5.2 핵심 비교 분석

#### (1) MViTv2 vs. Swin Transformer

| 비교 항목 | MViTv2 | Swin |
|---------|--------|------|
| 어텐션 방식 | 풀링 기반 글로벌 어텐션 | 로컬 윈도우 + shifted window |
| 수용 영역 | 전역(global) | 로컬(점진적으로 확장) |
| 계산 효율 | 풀링 스트라이드로 제어 | 윈도우 크기로 제어 |
| IN-1K (L 규모) | 86.3% (218M) | 86.3% (197M) — 동등 |
| IN-21K (L ↑384²) | **88.4%** | 87.3% — MViTv2 +1.1% |
| COCO (시스템급) | **58.7** (Cascade) | 58.0 (HTC++) — 더 단순한 검출기로 우위 |
| 런타임 | MViTv2-S: 341 im/s | Swin-B: 276 im/s — MViTv2 23.5% 빠름 |

**핵심 차이:** MViTv2는 글로벌 어텐션을 유지하면서 풀링으로 효율성을 확보하는 반면, Swin은 로컬 어텐션을 기반으로 shifted window로 cross-window 연결을 시도. MViTv2의 접근이 특히 객체 검출과 비디오에서 우위.

#### (2) MViTv2 vs. CoAtNet

CoAtNet은 convolution과 attention을 단계적으로 결합한 하이브리드 모델로, CoAtNet-4(↑512²)가 88.4%를 달성. MViTv2-H(↑512²)는 88.8%로 이를 상회하나, CoAtNet은 275M 파라미터로 MViTv2-H의 667M보다 효율적. 이는 **파라미터 효율성 측면에서 하이브리드 접근의 강점**을 시사.

#### (3) MViTv2 vs. 비디오 Transformer들

- **ViViT:** Factorized encoder로 시공간 분해 처리. K400에서 81.3% (IN-21K). MViTv2-L은 86.1%로 +4.8% 우위.
- **Video Swin:** K400에서 84.9% (IN-21K, Swin-L ↑384²). MViTv2-L(40×3)은 86.1%로 +1.2% 우위. 특히 FLOPs도 유사한 수준에서 달성.
- **핵심:** MViTv2의 다중 스케일 풀링 어텐션이 시공간 모델링에서 shifted window보다 효과적.

### 5.3 후속 연구 동향 (2022년 이후)

MViTv2 이후 등장한 주요 모델들과의 관계:

1. **Hiera (Ryali et al., 2023):** MViTv2의 후속으로, MAE 사전학습과 결합하여 풀링 어텐션의 효과를 극대화. 상대 위치 임베딩 등의 수동 설계를 제거하고 MAE의 마스킹이 위치 정보를 자연스럽게 학습하도록 함. MViTv2의 설계 철학을 계승하면서 더 단순화.

2. **InternImage (Wang et al., 2023), ConvNeXt V2 (Woo et al., 2023):** CNN 기반 접근의 부활로, Transformer와 CNN의 설계 공간 탐색이 계속되고 있음. MViTv2의 풀링 어텐션과 CNN의 로컬 집계 간의 유사성이 주목받음.

3. **EVA/EVA-02 (Fang et al., 2023):** 대규모 사전학습과 결합한 ViT 기반 모델로, MViTv2가 보여준 "사전학습 규모에 따른 일반화 향상" 트렌드를 더 극단적으로 확장.

---

## 참고자료

1. **Li, Y., Wu, C.-Y., Fan, H., Mangalam, K., Xiong, B., Malik, J., & Feichtenhofer, C.** (2022). "MViTv2: Improved Multiscale Vision Transformers for Classification and Detection." *Proc. CVPR 2022.* — 본 논문 원문.

2. **Fan, H., Xiong, B., Mangalam, K., Li, Y., Yan, Z., Malik, J., & Feichtenhofer, C.** (2021). "Multiscale Vision Transformers." *Proc. ICCV 2021.* — MViTv1 원 논문.

3. **Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B.** (2021). "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." *Proc. ICCV 2021.* — 핵심 비교 대상.

4. **Dosovitskiy, A. et al.** (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR 2021.* — ViT 기초 논문.

5. **Shaw, P., Uszkoreit, J., & Vaswani, A.** (2018). "Self-Attention with Relative Position Representations." *arXiv:1803.02155.* — 상대 위치 임베딩 기초.

6. **Dai, Z., Liu, H., Le, Q. V., & Tan, M.** (2021). "CoAtNet: Marrying Convolution and Attention for All Data Sizes." *NeurIPS 2021.* — 하이브리드 모델 비교.

7. **Liu, Z., Ning, J., Cao, Y., Wei, Y., Zhang, Z., Lin, S., & Hu, H.** (2021). "Video Swin Transformer." *CVPR 2022.* — 비디오 도메인 비교.

8. **Ryali, C. et al.** (2023). "Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles." *ICML 2023.* — MViTv2 후속 연구.

9. **Touvron, H. et al.** (2020). "Training Data-Efficient Image Transformers & Distillation through Attention." *ICML 2021.* — DeiT 레시피.

10. **Lin, T.-Y. et al.** (2017). "Feature Pyramid Networks for Object Detection." *Proc. CVPR 2017.* — FPN 기초.

11. MViTv2 공식 코드 저장소: https://github.com/facebookresearch/mvit
