# Token Merging: Your ViT But Faster

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

**Token Merging (ToMe)**는 기존 Vision Transformer (ViT) 모델을 **재학습 없이** 처리 속도를 향상시킬 수 있는 간단하고 범용적인 방법이다. 토큰을 제거(pruning)하는 대신 **유사한 토큰들을 병합(merging)** 함으로써, 정보 손실을 최소화하면서 속도를 높인다.

### 주요 기여

| 기여 | 내용 |
|------|------|
| **학습 불필요** | 기존 ViT 모델에 off-the-shelf 적용 가능 |
| **이분 소프트 매칭** | 빠르고 정확한 신규 토큰 병합 알고리즘 제안 |
| **비례 어텐션** | 병합 후 토큰 크기를 반영한 소프트맥스 보정 |
| **다중 모달리티** | 이미지, 비디오, 오디오 모두에 코드 변경 없이 적용 |
| **훈련 가속** | 학습 시 적용 시 MAE fine-tuning 속도 최대 2× 향상 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

ViT는 강력한 성능을 가지지만 **연산 비용이 높다**. 기존 토큰 감소 방법들(token pruning)은 다음 문제를 가진다:

- 정보 손실로 인한 정확도 저하
- 재학습 필수 (일부는 추가 파라미터 필요)
- 훈련 속도 향상 불가능 (훈련 시 마스킹 필요)
- 동적 토큰 수로 인해 배치 추론 불가능

### 2.2 제안하는 방법

#### 2.2.1 전략 (Strategy)

각 트랜스포머 블록에서 $r$개의 토큰을 병합하여, $L$개 레이어에 걸쳐 총 $rL$개 토큰을 점진적으로 줄인다. $r$은 비율이 아닌 **개수**이며, 입력 내용과 무관하게 고정된다(→ 배치 추론 가능).

#### 2.2.2 토큰 유사도 (Token Similarity)

중간 feature 공간이 과잉 파라미터화되어 있어 feature 값 그대로 유사도를 계산하면 노이즈에 취약하다. 대신, QKV self-attention의 **Key 행렬** $K$를 활용하여 코사인 유사도를 계산한다:

$$\text{similarity}(i, j) = \frac{K_i \cdot K_j}{\|K_i\| \|K_j\|}$$

#### 2.2.3 이분 소프트 매칭 (Bipartite Soft Matching)

알고리즘:

1. 토큰을 두 집합 $\mathbb{A}$, $\mathbb{B}$로 분할 (교번 방식)
2. $\mathbb{A}$의 각 토큰에서 $\mathbb{B}$ 내 가장 유사한 토큰으로 엣지 연결
3. 유사도 상위 $r$개의 엣지만 유지
4. 연결된 토큰 쌍을 병합 (가중 평균)
5. 두 집합을 다시 연결(concatenate)

이 알고리즘은 이분 그래프를 형성하여 연결 요소(connected component) 탐색이 $O(1)$이며, **$r$에 대해 상수 시간 복잡도**를 가진다.

#### 2.2.4 비례 어텐션 (Proportional Attention)

토큰이 병합되면 하나의 토큰이 여러 패치를 대표한다. 이를 소프트맥스 어텐션에 반영하기 위해:

$$\boldsymbol{A} = \text{softmax}\!\left(\frac{\boldsymbol{Q}\boldsymbol{K}^\top}{\sqrt{d}} + \log \boldsymbol{s}\right) $$

여기서 $\boldsymbol{s}$는 각 토큰이 대표하는 패치 수(토큰 크기)를 담은 행 벡터이다. $\log \boldsymbol{s}$를 더함으로써, 마치 해당 키를 $s$번 복사한 것과 동일한 효과를 낸다.

병합 시 가중 평균:

$$\text{merged token} = \frac{\sum_{i \in \text{cluster}} s_i \cdot x_i}{\sum_{i \in \text{cluster}} s_i}$$

#### 2.2.5 병합 스케줄 (Merging Schedule)

$$\text{Constant Schedule: } x \text{ per layer} \quad \rightarrow \quad r_x $$

$$\text{Decreasing Schedule: } 2x \to 0 \text{ per layer} \quad \rightarrow \quad r_x \searrow $$

두 스케줄 모두 총 $rL$개의 토큰을 제거하지만, decreasing schedule은 초반에 많이 제거하여 더 높은 처리량을 달성한다.

#### 2.2.6 학습 중 병합 (Training with Merging)

토큰 병합을 **평균 풀링**으로 취급하여 역전파를 통해 학습한다. Gumbel-softmax 등의 gradient trick이 필요 없으며, 기존 ViT 학습 레시피를 그대로 사용할 수 있다.

### 2.3 모델 구조

```
Input Tokens
     ↓
[Attention Module]
     ↓
[ToMe: Bipartite Soft Matching] ← K matrix에서 유사도 계산
     ↓
[MLP Module]
     ↓
Output (reduced tokens)
```

- ToMe는 **어텐션과 MLP 사이**에 삽입됨 (블록 시작 부분에 삽입하는 기존 연구와 차별화)
- 이 위치 선택이 정확도 향상에 기여 (어텐션 후 K 행렬 활용 가능)

### 2.4 성능 향상

#### 이미지 (ImageNet-1k)

| 모델 | 설정 | 처리량 향상 | 정확도 하락 |
|------|------|------------|------------|
| ViT-L/16 @ 512 (SWAG) | off-the-shelf | $2\times$ | $0.3\%$ |
| ViT-H/14 @ 518 (SWAG) | off-the-shelf | $2\times$ | $0.3\%$ |
| ViT-H/14 (MAE) | 학습 적용 | $2\times$ | $0.4\%$ |
| ViT-L/16 (MAE) | 학습 적용 | $2\times$ | $0.6\%$ |

#### 비디오 (Kinetics-400)

| 모델 | 설정 | 처리량 향상 | 정확도 하락 |
|------|------|------------|------------|
| ViT-L (Spatiotemporal MAE) | $r=65$, constant | $2.2\times$ | $0.2\%$ |
| ViT-L (MAE) | $r=65$, constant | 훈련 시간 $0.5\times$ | 무시 가능 |

#### 오디오 (AudioSet-2M)

| 모델 | 설정 | 처리량 향상 | mAP 하락 |
|------|------|------------|----------|
| ViT-B (Audio MAE) | $r=40$, 학습 적용 | $\approx 2\times$ | $0.4\%$ |

#### 토큰 프루닝 대비 비교 (ViT-S, DeiT 기준)

| 방법 | 정확도 | 처리량(im/s) | 훈련 속도 |
|------|--------|------------|----------|
| DeiT-S (baseline) | 79.8% | 930 | $1\times$ |
| DynamicViT | 79.3% | 1505 | $1\times$ |
| SP-ViT | 79.3% | — | $1\times$ |
| **ToMe DeiT** $r_{13}\rightarrow$ | **79.4%** | **1552** | **$1.5\times$** |

### 2.5 한계

1. **소형 모델에서 정확도 하락 큼**: ViT-B, S, Ti 등 작은 모델은 $2\times$ 속도에서 4~5% 정확도 하락 발생
2. **레이어 깊이 의존성**: 더 깊은(larger) 모델에서 효과가 크고, 얕은 모델에서는 효과 감소
3. **콘텐츠 무관 고정 병합 수**: 동적 방법보다 정확도 측면에서 이론적으로 불리
4. **조밀한 예측 태스크 미검증**: 분류 태스크 위주로 검증; 객체 탐지, 세그멘테이션 등은 미래 과제
5. **프리트레이닝과��� 정합성**: MAE 모델은 proportional attention 없이도 잘 작동하지만, 지도학습 모델은 필요 → 프리트레이닝 방식에 따른 최적 설정이 다름
6. **멀티클립 보상 효과**: 비디오에서 여러 클립 평가가 정보 손실을 일부 보상할 수 있어, 실제 단일 추론 성능은 다를 수 있음

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 멀티모달 일반화

ToMe의 가장 강력한 일반화 특성은 **코드 변경 없이** 이미지, 비디오, 오디오에 모두 적용 가능하다는 점이다. 이는 모달리티 특화 귀납적 편향(inductive bias)이 없기 때문이다.

```math
\text{ToMe} \xrightarrow{\text{동일 코드}} \begin{cases} \text{Image (ImageNet-1k)} \\ \text{Video (Kinetics-400)} \\ \text{Audio (AudioSet-2M)} \end{cases}
```

### 3.2 학습 없이도 일반화 가능

**Off-the-shelf 적용**: 재학습 없이 기존 모델에 바로 적용 가능하다는 것은, ToMe가 모델 가중치에 내재된 표현 능력에 의존하며 **추가적인 도메인 특화 학습 없이** 일반화됨을 시사한다.

### 3.3 대형 모델에서의 일반화 향상

실험 결과, **모델이 크고 깊을수록** ToMe의 정확도 하락이 작다:

$$\text{ViT-Ti: } \Delta\text{acc} \approx -4\% \quad \xrightarrow{\text{모델 크기 증가}} \quad \text{ViT-H: } \Delta\text{acc} \approx -0.3\% \quad (\text{at } 2\times)$$

저자들은 이 현상을 "깊은 모델일수록 레이어당 feature 변화가 점진적이어서 병합의 영향이 줄어든다"고 설명한다. 이는 **스케일링 법칙(scaling law)과 ToMe의 시너지** 가능성을 시사한다.

### 3.4 Re-evaluation을 통한 일반화

흥미로운 발견: $r=5$로 학습된 ViT-L 모델을 $r=0$(병합 없음)으로 재평가하면 기존 baseline보다 **정확도가 향상**된다(85.7% → 85.8%). 이는 ToMe 학습이 일종의 **정규화(regularization)** 효과를 가지며, 모델의 일반화 성능을 향상시킬 수 있음을 의미한다.

$$\mathcal{M}_{r=5}^{\text{train}} \xrightarrow{r=0 \text{ 재평가}} \text{acc: } 85.8\% > 85.7\% = \mathcal{M}_{r=0}^{\text{baseline}}$$

### 3.5 시각적 일반화: 부분 분할 및 객체 추적

ToMe의 토큰 병합은 **부분 분할(part segmentation)**과 유사한 패턴을 자연스럽게 학습한다. 이미지에서는 객체의 의미론적 부분들이 같은 토큰으로 병합되고, 비디오에서는 동일 객체가 여러 프레임에 걸쳐 추적된다. 이는 ToMe가 의미론적 표현의 일반화를 촉진할 수 있음을 시사한다.

### 3.6 MAE와의 시너지를 통한 일반화

MAE는 프리트레이닝 시 토큰을 마스킹하여 제거하는데, ToMe의 병합 방식이 이와 유사한 효과를 내어 **fine-tuning 시 일반화 성능**을 향상시킬 수 있다. 특히 fine-tuning 시 원래 레시피를 그대로 사용할 수 있다는 점은 ToMe가 학습 동역학을 크게 변형하지 않음을 보여준다.

---

## 4. 최신 연구 비교 분석 (2020년 이후)

### 4.1 토큰 프루닝 계열

| 논문 | 방법 | 장점 | 단점 vs ToMe |
|------|------|------|-------------|
| **DynamicViT** (Rao et al., NeurIPS 2021) | 어텐션 기반 동적 프루닝 | 높은 정확도 | 재학습 필요, 배치 추론 불가, 훈련 속도 향상 없음 |
| **A-ViT** (Yin et al., CVPR 2022) | 적응적 토큰 프루닝 | 입력 적응적 | DeiT 체크포인트 fine-tuning 필요, 배치 불가 |
| **SP-ViT** (Kong et al., ECCV 2022) | 소프트 토큰 프루닝 | 높은 정확도 | 추가 파라미터, 배치 불가 |
| **EViT** (Liang et al., ICLR 2022) | 비중요 토큰 집약 | 정보 보존 | 재학습 필요 |
| **ToMe** (Bolya et al., ICLR 2023) | 이분 소프트 매칭 병합 | **학습 불필요, 배치 가능, 훈련 가속** | 소형 모델 정확도 하락 |

### 4.2 효율적 ViT 아키텍처 계열

| 논문 | 방법 | vs ToMe |
|------|------|---------|
| **Swin Transformer** (Liu et al., ICCV 2021) | 시프트 윈도우 어텐션 | 도메인 특화 설계 필요, 재학습 필요 |
| **MViTv2** (Li et al., CVPR 2022) | 멀티스케일 풀링 | 비디오 특화, 재학습 필요 |
| **LeViT** (Graham et al., ICCV 2021) | Conv 모듈 혼합 | 비ViT 아키텍처 |
| **ToMe** | 플러그인 병합 | **기존 ViT 재사용, 코드 수정 최소화** |

### 4.3 토큰 병합/풀링 계열

| 논문 | 방법 | vs ToMe |
|------|------|---------|
| **Token Pooling** (Marin et al., 2021) | k-means 클러스터링 | 느림(순차적), 학습 없이 10~40% 정확도 하락 |
| **GroupViT** (Xu et al., CVPR 2022) | 크로스 어텐션 그루핑 | 효율성 목적 아님, 세그멘테이션 특화 |
| **TokenLearner** (Ryoo et al., NeurIPS 2021) | MLP 기반 토큰 감소 | 추가 파라미터 필요 |
| **ToMe** | 이분 소프트 매칭 | **추가 파라미터 없음, 병렬화 가능, off-the-shelf** |

### 4.4 이후 ToMe의 영향을 받은/관련된 연구 방향

ToMe 발표(2022~2023) 이후, 다음과 같은 연구 방향들이 활발히 진행되었다:

- **ToMe for Stable Diffusion**: ToMe 개념을 디퓨전 모델의 U-Net에 적용하여 생성 속도 향상 시도 (Bolya & Hoffman, 2023)
- **동적 병합 스케줄 최적화**: 입력 콘텐츠에 따른 적응적 병합 전략 연구
- **LLM 토큰 감소**: 언어 모델에서의 유사 아이디어 적용

---

## 5. 향후 연구에 미치는 영향과 고려 사항

### 5.1 향후 연구에 미치는 영향

#### 5.1.1 효율적 ViT 연구 패러다임 전환

ToMe는 기존의 **"더 작은 아키텍처 설계"** 패러다임에서 **"기존 대형 모델의 효율적 배포"** 패러다임으로의 전환을 촉진한다. 재학습 없이 대형 모델을 효율화할 수 있다는 점은 산업 현장에서의 활용성을 크게 높인다.

#### 5.1.2 자연적 계층 모델로서의 ViT

저자들이 언급하듯, ToMe는 ViT를 Swin이나 MViT와 유사한 **자연적 계층 구조** 모델로 변환한다. 이는 기존 계층적 아키텍처와 순수 ViT 사이의 간극을 메우는 새로운 관점을 제공한다.

#### 5.1.3 대규모 모델 훈련 가속

훈련 시 최대 $2\times$ 속도 향상은, 이전에는 불가능했던 **더 큰 모델의 훈련**을 가능하게 한다. 이는 스케일링 연구에 직접적인 영향을 미친다.

#### 5.1.4 멀티모달 AI 연구

코드 변경 없이 이미지, 비디오, 오디오에 적용 가능하다는 점은, **멀티모달 파운데이션 모델**의 효율화에 직접 적용 가능하며, 향후 멀티모달 연구의 중요한 기반이 될 수 있다.

### 5.2 향후 연구 시 고려할 점

#### 5.2.1 조밀 예측 태스크로의 확장

현재 ToMe는 분류 태스크에서 검증되었다. **객체 탐지, 인스턴스 세그멘테이션, 깊이 추정** 등 공간 정보가 중요한 태스크에서는 토큰 병합이 위치 정보를 손상시킬 수 있다. 병합 후에도 공간 정보를 보존하는 방법이 필요하다.

#### 5.2.2 비병합 토큰의 선택적 보호

현재 알고리즘은 유사한 토큰을 병합하지만, **의미론적으로 중요한 토큰**(예: 희귀 객체, 작은 세부 사항)을 보호하는 메커니즘이 없다. 중요도 점수와 유사도를 결합하는 방법을 고려할 수 있다.

#### 5.2.3 적응적 $r$ 값 선택

현재 $r$은 고정값이다. **입력 복잡도에 따른 동적 $r$ 선택** (단, 배치 추론 가능성 유지)은 정확도-속도 트레이드오프를 개선할 수 있다. 예를 들어, 학습 가능한 경량 predictor를 통해 레이어별 $r$을 결정하는 방법이 있다.

#### 5.2.4 소형 모델에서의 정확도 하락 문제

ViT-Ti/S/B 등 소형 모델에서는 정확도 하락이 크다. 이를 해결하기 위해:

- ToMe를 고려한 **처음부터의 학습(training from scratch)** 전략
- **Knowledge Distillation**과의 결합
- **더 정교한 병합 기준** (단순 코사인 유사도 외 의미론적 정보 활용)

등을 고려할 수 있다.

#### 5.2.5 LLM/디퓨전 모델로의 확장

ToMe의 핵심 아이디어는 어텐션 기반 모델 전반에 적용 가능하다. **Large Language Model (LLM)** 에서의 KV-cache 최적화나 **Diffusion Model**의 U-Net 가속에 적용하는 연구가 진행되고 있으며, 이 방향에서의 이론적 분석이 필요하다.

#### 5.2.6 병합의 이론적 이해

왜 Key 행렬 기반 코사인 유사도가 가장 효과적인지, 병합이 왜 정규화 효과를 가지는지에 대한 **이론적 분석**이 부족하다. 정보 이론적 관점에서의 분석이 향후 연구의 방향을 제시할 수 있다.

#### 5.2.7 하드웨어 최적화와의 결합

ToMe는 현재 소프트웨어 레벨의 최적화이다. **FlashAttention, Sparse Attention** 등 하드웨어 친화적 최적화와의 결합을 통해 추가적인 속도 향상이 가능할 것이다.

---

## 참고 자료

**주요 논문 (직접 인용)**

1. **Bolya, D., Fu, C.-Y., Dai, X., Zhang, P., Feichtenhofer, C., & Hoffman, J. (2023).** "Token Merging: Your ViT But Faster." *ICLR 2023.* arXiv:2210.09461v3.

**논문 내 참조 문헌 (주요)**

2. Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR 2021.*
3. He, K., et al. (2022). "Masked Autoencoders Are Scalable Vision Learners." *CVPR 2022.*
4. Rao, Y., et al. (2021). "DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification." *NeurIPS 2021.*
5. Yin, H., et al. (2022). "A-ViT: Adaptive Tokens for Efficient Vision Transformer." *CVPR 2022.*
6. Kong, Z., et al. (2022). "SPViT: Enabling Faster Vision Transformers via Soft Token Pruning." *ECCV 2022.*
7. Liang, Y., et al. (2022). "Not All Patches Are What You Need: Expediting Vision Transformers via Token Reorganizations." *ICLR 2022.*
8. Liu, Z., et al. (2021). "Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows." *ICCV 2021.*
9. Li, Y., et al. (2022). "MViTv2: Improved Multiscale Vision Transformers for Classification and Detection." *CVPR 2022.*
10. Feichtenhofer, C., et al. (2022). "Masked Autoencoders as Spatiotemporal Learners." *NeurIPS 2022.*
11. Huang, P.-Y., et al. (2022). "Masked Autoencoders That Listen." *NeurIPS 2022.*
12. Marin, D., et al. (2021). "Token Pooling in Vision Transformers." arXiv:2110.03860.
13. Singh, M., et al. (2022). "Revisiting Weakly Supervised Pre-Training of Visual Perception Models." *CVPR 2022.*
14. Steiner, A., et al. (2022). "How to Train Your ViT? Data, Augmentation, and Regularization in Vision Transformers." *TMLR 2022.*
15. Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS 2017.*

**공개 코드**
- GitHub: [facebookresearch/ToMe](http://github.com/facebookresearch/ToMe)
