# Segment Any 3D Gaussians

---

# 1. 핵심 주장 및 주요 기여 (요약)

SAGA(Segment Any 3D GAussians)는 3D Gaussian Splatting(3D-GS) 기반의 고효율 3D promptable segmentation 방법으로, 2D 시각 프롬프트 입력 시 해당 3D 타겟을 3D Gaussian으로 4 ms 이내에 분할할 수 있다.

**주요 기여:**

1. 각 3D Gaussian에 **scale-gated affinity feature**를 부착하여 다중 세분화(multi-granularity) 분할 능력을 부여한다.
2. **Scale-aware contrastive training** 전략을 제안하여 (1) SAM의 2D 분할 능력을 affinity feature로 증류(distill)하고, (2) soft scale gate 메커니즘으로 3D 물리적 스케일에 따라 각 feature 채널의 크기를 조정하여 다중 세분화 모호성(multi-granularity ambiguity)을 처리한다.
3. 3D-GS에서 promptable segmentation을 다룬 최초의 방법 중 하나로, 단순성과 효과성을 통해 해당 분야의 향후 발전 기반을 마련하였다.

---

# 2. 상세 분석

## 2.1 해결하고자 하는 문제

2D 분할 기초 모델(SAM 등)의 발전에도 불구하고, **3D promptable segmentation**은 3D 데이터의 부족과 높은 어노테이션 비용으로 인해 상대적으로 미개척 영역이었다.

SAGA가 해결해야 하는 두 가지 핵심 과제는: (1) 3D-GS의 높은 효율성을 유지하면서 각 3D Gaussian에 분할 능력을 부여하는 효율적 방법을 찾는 것, (2) 단일 3D Gaussian이 다양한 세분화 수준에서 서로 다른 부분/객체에 속할 수 있는 **다중 세분화 모호성(multi-granularity ambiguity)** 문제를 해결하는 것이다.

기존 방법들은 세밀한 다중 세분화 분할을 달성하거나 상당한 계산 오버헤드를 처리하는 데 어려움을 겪어 실시간 상호작용을 저해했다.

## 2.2 제안하는 방법 (수식 포함)

### (A) 전체 파이프라인

사전 학습된 3DGS 모델과 학습 세트가 주어지면, 모델 내 각 Gaussian에 저차원 3D feature를 부착한다.

3D-GS에서 각 Gaussian $g_i$는 기존 속성(위치 $\boldsymbol{\mu}_i$, 공분산 $\boldsymbol{\Sigma}_i$, 색상 $\mathbf{c}_i$, 불투명도 $\alpha_i$) 외에 새로운 **affinity feature** $\mathbf{f}_i \in \mathbb{R}^d$를 갖게 된다.

### (B) 3D-GS 렌더링 (Feature Splatting)

3D-GS의 래스터화는 선형(linear)이므로 feature를 색상과 동일하게 렌더링할 수 있다. 픽셀 $\mathbf{p}$에서의 렌더링된 feature:

$$
\hat{\mathbf{f}}(\mathbf{p}) = \sum_{i \in \mathcal{N}} \mathbf{f}_i \, \alpha_i \, T_i, \quad \text{where} \quad T_i = \prod_{j=1}^{i-1}(1 - \alpha_j)
$$

여기서 $\mathcal{N}$은 해당 픽셀에 기여하는 Gaussian 집합, $T_i$는 transmittance이다.

### (C) SAM-guidance Loss

SAM으로 자동 추출한 2D 마스크 $\{M_k\}$를 이용하여 affinity feature를 학습한다. 각 마스크 $M_k$에 대해 마스크 내부 픽셀들의 평균 feature를 프로토타입 $\bar{\mathbf{f}}_k$로 정의하고, 각 픽셀이 해당 마스크에 속할 확률을 softmax로 계산한다:

$$
P(M_k \mid \mathbf{p}) = \frac{\exp\bigl(\text{sim}(\hat{\mathbf{f}}(\mathbf{p}),\; \bar{\mathbf{f}}_k)\bigr)}{\sum_{k'} \exp\bigl(\text{sim}(\hat{\mathbf{f}}(\mathbf{p}),\; \bar{\mathbf{f}}_{k'})\bigr)}
$$

SAM-guidance loss는 cross-entropy 형태:

$$
\mathcal{L}_{\text{SAM}} = -\sum_{\mathbf{p}} \sum_{k} \mathbf{1}[\mathbf{p} \in M_k] \log P(M_k \mid \mathbf{p})
$$

### (D) Correspondence Loss

SAM-guidance loss만으로는 학습된 feature가 충분히 compact하지 않아 다양한 프롬프트에 대한 분할 품질이 저하된다. 이에 contrastive correspondence distillation에서 영감을 받아 correspondence loss를 도입한다.

두 픽셀 $p_1, p_2$가 속하는 마스크 집합의 IoU를 기반으로 mask correspondence를 정의하고, 이를 이용한 대조 학습 손실:

$$
\mathcal{L}_{\text{corr}} = \sum_{(p_1, p_2)} \bigl\| \text{sim}(\hat{\mathbf{f}}(p_1), \hat{\mathbf{f}}(p_2)) - \text{IoU}(S_{p_1}, S_{p_2}) \bigr\|^2
$$

여기서 $S_p$는 픽셀 $p$가 속하는 마스크 집합이다.

### (E) Scale-Gated Affinity Feature

soft scale gate 메커니즘을 사용하여 원하는 3D 물리적 스케일에 따라 각 feature 채널의 크기를 조정함으로써 다중 세분화 분할 모호성을 처리한다.

각 SAM 마스크 $M_k$의 3D 물리적 스케일 $s_k$를 계산한 후, scale gate $\mathbf{g}(s) \in \mathbb{R}^d$를 통해 feature를 변조한다:

$$
\tilde{\mathbf{f}}_i = \mathbf{f}_i \odot \mathbf{g}(s)
$$

여기서 $\mathbf{g}(s)$는 스케일 $s$에 따라 soft하게 각 채널을 활성화/비활성화하는 게이트이며, $\odot$은 element-wise product이다. 이 게이트는 MLP 등으로 구현되어 스케일에 따라 학습된다.

### (F) 최종 손실 함수

SAM-guidance loss와 correspondence loss를 결합하여 최종 손실 함수를 구성한다:

$$
\mathcal{L} = \mathcal{L}_{\text{SAM}} + \lambda \, \mathcal{L}_{\text{corr}}
$$

### (G) 추가 정규화 기법

local feature smoothing과 feature norm regularization을 도입하여 노이즈 아웃라이어를 제거하고 3D Gaussian feature의 정렬을 향상시킨다.

- **Local Feature Smoothing**: 아웃라이어는 주로 local feature smoothing을 통해 제거된다.
- **Feature Norm Regularization**: 3D feature가 2D feature와 완벽히 정렬되지 않으므로, feature norm 정규화를 통해 같은 ray 위 Gaussian들의 affinity feature를 같은 방향으로 정렬시킨다.

## 2.3 모델 구조

SAGA의 전체 파이프라인은 다음과 같다:

```
[사전 학습된 3D-GS] → [각 Gaussian에 affinity feature 부착]
        ↓
[SAM으로 multi-view 2D 마스크 자동 추출 + 마스크 스케일 계산]
        ↓
[Scale-aware Contrastive Training]
  ├── SAM-guidance Loss
  ├── Correspondence Loss  
  └── Scale Gate Mechanism
        ↓
[추론 시: 2D 프롬프트 → query feature 생성 → feature matching → 3D Gaussian 검색 → 후처리]
```

학습은 SAM으로 자동 추출된 마스크를 기반으로 3D Gaussian feature를 훈련하며, 추론 시에는 입력 프롬프트로 query를 생성하고 효율적 feature matching으로 해당 Gaussian을 검색한다.

기존 radiance field와 달리, 명시적(explicit) 3D Gaussian 구조는 3D 분할의 이상적 기반이 되어 추가적인 대형 분할 모듈 없이도 내재적 속성으로 분할 능력을 통합할 수 있다.

## 2.4 성능

SAGA는 3D-OVS 데이터셋에서 open-vocabulary semantic segmentation 기준으로 최고 수준의 mIoU 96.0%를 달성하여, N2F2(93.9%), LangSplat(93.4%), 3D-OVS(86.8%), LERF(54.8%)를 모두 크게 능가한다.

SAGA는 실시간 다중 세분화 분할을 최첨단 방법과 비교 가능한 품질로 달성한다.

| 방법 | mIoU (3D-OVS) | 비고 |
|------|:---:|------|
| LERF | 54.8% | 언어 기반 |
| 3D-OVS | 86.8% | NeRF 기반 |
| LangSplat | 93.4% | 3D-GS + 언어 |
| N2F2 | 93.9% | 계층적 feature field |
| **SAGA** | **96.0%** | 3D-GS + contrastive |

**속도:** 2D 프롬프트 입력 시 4 ms 이내에 3D 분할을 완료한다.

## 2.5 한계점

SAGA는 SAM이 추출한 multi-view 2D 마스크로부터 affinity feature를 학습하므로, 이 마스크에 나타나지 않는 객체는 분할하기 어렵다. 이 한계는 특히 관심 대상이 작은 경우에 두드러진다.

3DGS로 학습된 Gaussian은 기하학적 제약 없이 모호하며, 단일 Gaussian이 여러 객체에 대응할 수 있어 feature matching을 통한 정확한 분할이 복잡해진다.

SAM이 자동 추출한 마스크는 다중 세분화 특성의 부산물로 일정 수준의 노이즈를 보이며, 후처리 단계가 의미론적으로 무관(semantic-agnostic)하여 false positive를 유발할 수 있다.

SAGA의 feature matching 방식은 경계(boundary)에서 불완전한 분할 문제가 있다.

---

# 3. 모델의 일반화 성능 향상 가능성

## 3.1 현재의 일반화 한계

논문은 3D Gaussian Splatted 씬과 같은 특정 시나리오에서 SAGA를 평가했으며, 보다 다양하고 복잡한 3D 환경에서의 성능 평가가 필요하다.

핵심 일반화 한계 요인:

1. **SAM 의존성**: SAM이 인식하지 못한 대상에 대해서는 affinity feature 학습이 불가능
2. **장면별 학습(scene-specific training)**: 2D 정보를 3D로 통합하려면 기존 모델의 추가 학습이 불가피하여, 3D 씬 획득 후 즉시 분할 작업을 시작할 수 없다.
3. **경계 모호성**: 3D Gaussian의 기하학적 제약 부재로 인한 경계부 분할 부정확성

## 3.2 일반화 향상을 위한 방향

1. **Scale Gate의 범용성**: scale gate 메커니즘을 GARField 등 다른 radiance field에 적용하여 경쟁력 있는 결과를 달성할 수 있어 SAGA의 다양한 radiance field에 대한 잠재력을 보여준다.

2. **SAM 2 등 발전된 Foundation Model 활용**: SAM 2와 같은 비디오 foundation model을 활용하면 합성된 뷰 간 마스크 전파가 가능해지고, 실시간 학습 불요 분할이 가능하다.

3. **기하학적 제약 강화**: 3DGS 표현 자체의 향후 발전으로 이 문제가 완화될 수 있다.

4. **다중 모달 프롬프트 통합**: 사용자 피드백 통합과 반복적 분할 정제를 통해 대화형 3D 모델링 및 편집에서의 사용성을 더욱 향상시킬 수 있다.

5. **Cross-scene 일반화**: WildSeg3D와 같이 임의의 씬에서 장면별 학습 없이도 강건한 일반화를 보여주는 방법론의 통합이 유망하다.

---

# 4. 향후 연구에 미치는 영향 및 고려사항

## 4.1 연구적 영향

1. **3D-GS 기반 분할 패러다임 확립**: 3D-GS에서 promptable segmentation을 다룬 최초의 방법 중 하나로서 해당 분야 후속 연구의 토대를 마련하였다.

2. **Contrastive Learning + 3D-GS 조합의 표준화**: SAGA에서 영감을 받은 후속 연구들이 contrastive learning 접근법을 채택하여 Gaussian affinity feature를 훈련하는 방식을 따르고 있다.

3. **실시간 3D 이해의 실용성 입증**: 4 ms 이내의 분할 속도로 AR/VR, 로보틱스 등 실시간 응용의 가능성을 열었다.

4. **Open-vocabulary 3D 분할의 가능성 확대**: CLIP 등과의 결합으로 텍스트 기반 3D 분할까지 확장될 수 있음을 시사한다.

## 4.2 향후 연구 시 고려할 점

| 고려 사항 | 세부 내용 |
|-----------|----------|
| **경계 정밀도** | SAGD의 Gaussian Decomposition 같은 기법으로 경계 품질 향상 필요 |
| **Multi-view 일관성** | Gaga의 3D-aware Memory Bank처럼 뷰 간 일관된 분할 보장 메커니즘 |
| **학습 비용 감소** | Training-free 또는 few-shot 적응 방법 개발 |
| **Foundation Model 진화** | SAM 2, Grounded-SAM 등 차세대 모델과의 시너지 |
| **대규모 씬 확장** | 도시 규모, 야외 환경 등 대규모 복잡한 씬으로의 확장성 |
| **동적 씬 처리** | 정적 씬 한정이 아닌 동적 3D-GS에서의 분할 |

---

# 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 학회/저널 | 3D 표현 | 핵심 특징 | SAGA 대비 차별점 |
|------|------|----------|---------|-----------|----------------|
| **NeRF** (Mildenhall et al.) | 2020 | ECCV | Implicit (MLP) | Neural radiance field | 분할 기능 없음, 느린 렌더링 |
| **Semantic-NeRF** (Zhi et al.) | 2021 | ICCV | NeRF | 의미론 전파 | Closed-set, 느린 속도 |
| **LERF** (Kerr et al.) | 2023 | ICCV | NeRF | CLIP feature field | 정밀 분할 부족 (mIoU 54.8%) |
| **SAM** (Kirillov et al.) | 2023 | ICCV | 2D | Promptable 2D 분할 | 2D 전용, 3D 확장 필요 |
| **3D-GS** (Kerbl et al.) | 2023 | ACM TOG | Explicit Gaussian | 실시간 렌더링 | 분할 미지원 |
| **SA3D** (Cen et al.) | 2023 | NeurIPS | NeRF | 반복적 파이프라인으로 SAM과 3D 마스크 그리드를 정제 | 느린 반복적 처리, 단일 객체 |
| **GARField** (Kim et al.) | 2024 | CVPR | NeRF/3D-GS | 계층적 그룹핑 파이프라인, SAM 마스크 기반 contrastive loss | 41단계 계층 구조, 단일 이미지에 ~20분 소요 |
| **OmniSeg3D** (Ying et al.) | 2024 | CVPR | NeRF/3D-GS | 계층적 contrastive learning으로 multi-view 2D 마스크에서 자동 분할 학습 | 뷰 간 고유 마스크 라벨 미제공 |
| **Gaussian Grouping** (Ye et al.) | 2024 | ECCV | 3D-GS | Identity Encoding 부착, SAM 2D 마스크와 비디오 트래킹으로 감독 | 유사 객체 혼동, 큰 카메라 시점 변화에 취약 |
| **SAGD** (Hu et al.) | 2024 | arXiv | 3D-GS | Gaussian Decomposition으로 경계 Gaussian을 분해하여 분할 정확도 향상 | Training-free지만 장면별 처리 필요 |
| **Gaga** (Lyu et al.) | 2024 | arXiv | 3D-GS | 3D-aware Memory Bank로 비일관적 2D 마스크를 공간 정보 기반 연관 | 뷰 연속성 가정 불요, 다양한 2D 소스 호환 |
| **SAGA** (Cen et al.) | 2025 | AAAI | 3D-GS | Scale-gated affinity + Contrastive training | **4 ms 분할, 96.0% mIoU (3D-OVS)** |
| **SAGOnline** | 2025 | arXiv | 3D-GS | SAM 2 활용, NVOS 92.7%, SPIn-NeRF 95.2% mIoU 달성 | Training-free, GPU 가속 |
| **DCSEG** | 2024 | arXiv | 3D-GS | NeRF 기반 baseline 대비 대부분의 지표에서 우수, 모듈화된 설계 | 마스크 제안과 분류를 분리(decoupled) |
| **Contrastive Gaussian Clustering** | 2024 | ECCV | 3D-GS | contrastive clustering loss로 비일관적 2D 마스크에서 일관된 3D 분할 feature 학습, 전처리 최소화 | spatial-similarity regularization 추가 |

---

# 참고자료 및 출처

1. **Cen, J. et al.** "Segment Any 3D Gaussians," *AAAI 2025* (arXiv:2312.00860). — [arXiv](https://arxiv.org/abs/2312.00860), [Project Page](https://jumpat.github.io/SAGA/), [GitHub](https://github.com/Jumpat/SegAnyGAussians)
2. **Kerbl, B. et al.** "3D Gaussian Splatting for Real-Time Radiance Field Rendering," *ACM TOG 2023*.
3. **Kirillov, A. et al.** "Segment Anything," *ICCV 2023*.
4. **Kim, C. M. et al.** "GARField: Group Anything with Radiance Fields," *CVPR 2024*.
5. **Ying, H. et al.** "OmniSeg3D: Omniversal 3D Segmentation via Hierarchical Contrastive Learning," *CVPR 2024*.
6. **Ye, M. et al.** "Gaussian Grouping: Segment and Edit Anything in 3D Scenes," *ECCV 2024*. — [GitHub](https://github.com/lkeab/gaussian-grouping)
7. **Hu, X. et al.** "SAGD: Boundary-Enhanced Segment Anything in 3D Gaussian via Gaussian Decomposition," *arXiv:2401.17857*.
8. **Lyu, W. et al.** "Gaga: Group Any Gaussians via 3D-aware Memory Bank," *arXiv:2404.07977*.
9. **Bhalgat, Y. et al.** "Contrastive Lift: 3D Object Instance Segmentation by Slow-Fast Contrastive Fusion," *NeurIPS 2023*.
10. **Bhalgat, Y. et al.** "N2F2: Hierarchical Scene Understanding with Nested Neural Feature Fields," *arXiv:2403.10997*.
11. **Kerr, J. et al.** "LERF: Language Embedded Radiance Fields," *ICCV 2023*.
12. **DCSEG**: "Decoupled 3D Open-Set Segmentation using Gaussian Splatting," *arXiv:2412.10972*.
13. **Contrastive Gaussian Clustering** (Springer, ECCV 2024 workshop). — [Link](https://link.springer.com/chapter/10.1007/978-3-031-78347-0_8)
14. **SAGOnline**: "Segment Any Gaussians Online," *arXiv:2508.08219*.
15. **AI Models Review**: [aimodels.fyi/papers/arxiv/segment-any-3d-gaussians](https://www.aimodels.fyi/papers/arxiv/segment-any-3d-gaussians)
16. **Liner Quick Review**: [liner.com/review/segment-any-3d-gaussians](https://liner.com/review/segment-any-3d-gaussians)

> **주의**: 본 분석에서 제시된 수식은 논문의 공식적 표기를 기반으로 하되, arXiv HTML 버전(v3)의 구체적 수식 번호와 표기가 일부 간략화되어 있을 수 있습니다. 정확한 수식 확인을 위해 원 논문 PDF를 참조하시기 바랍니다.
