# Semi-DETR: Semi-Supervised Object Detection with Detection Transformers

논문의 핵심은, (1) DETR 계열을 그대로 SSOD에 쓰면 “헝가리안 one-to-one 매칭 + 노이즈 많은 pseudo box” 때문에 학습이 비효율적이고, (2) 쿼리–출력 간 일대일 대응이 없어 기존 consistency regularization을 적용하기 어렵다는 한계를 분석한 뒤, 이를 해결하는 Semi-DETR 프레임워크(하이브리드 매칭 + cross-view query consistency + cost 기반 pseudo-label mining)를 제안해 COCO/VOC 전 설정에서 기존 SOTA SSOD를 크게 능가한다는 것입니다.[^1_1][^1_2]

***

## 핵심 주장과 기여

- DETR 기반 SSOD에서 헝가리안 one-to-one 매칭은 부정확한 pseudo box에 대해 “틀린 proposal 하나를 무조건 양성으로” 만들고, 나머지(더 좋은) proposal들을 음성으로 밀어내 학습 효율을 크게 떨어뜨린다는 것을 체계적으로 분석합니다.[^1_2][^1_1]
- DETR는 학습 과정에서 object query와 최종 예측 사이의 대응이 계속 바뀌기 때문에, Teacher–Student 간 출력 일관성을 직접 맞추는 기존 SSOD의 consistency loss를 그대로 적용하기 어렵다는 구조적 한계를 지적합니다.[^1_1][^1_2]
- 이를 해결하기 위해 다음 세 모듈로 구성된 Semi-DETR를 제안합니다.[^1_2][^1_1]

1. **Stage-wise Hybrid Matching (SHM)**: 1단계에서 one-to-many 매칭으로 여러 양성 proposal을 학습에 활용해 pseudo label 품질과 수렴 속도를 높이고, 2단계에서 다시 one-to-one 매칭으로 전환해 최종 NMS-free DETR 구조를 유지.
2. **Cross-view Query Consistency (CQC)**: 두 뷰의 RoI feature로 만든 cross-view query를 서로의 디코더에 넣어, 쿼리–출력의 “의미적 불변성”에 대해 consistency loss를 부여하는 새로운 DETR용 consistency 규제.
3. **Cost-based Pseudo Label Mining (CPM)**: matching cost 분포에 Gaussian Mixture Model(GMM)을 맞춰 “신뢰도 높은 pseudo box”를 동적으로 더 많이 뽑아 consistency 학습에 활용.
- Deformable DETR와 DINO에 모두 적용해 COCO-Partial/Full, Pascal VOC 전 설정에서 PseCo, Dense Teacher 등 기존 SOTA보다 명확한 마진으로 향상(mAP +2∼8 수준)을 보이며, DETR 계열 SSOD의 일반적 설계 패턴을 제시합니다.[^1_1][^1_2]

***

## 해결하려는 문제

### DETR one-to-one 매칭의 SSOD 비효율성

- Teacher–Student SSOD에서 unlabeled 이미지에 대한 teacher의 pseudo box는 초기에 특히 부정확합니다.[^1_2][^1_1]
- DETR의 전통적인 헝가리안 one-to-one 매칭은 각 (pseudo) GT마다 **단 하나의 proposal**을 양성으로 강제하고 나머지는 전부 음성으로 두기 때문에, pseudo box가 잘못된 경우 “틀린 proposal 1개를 끝까지 positive로 밀고, 실제 물체에 더 가까운 proposal들을 negative로 학습시키는” 악영향이 생깁니다.[^1_1][^1_2]
- 라벨이 적을수록 pseudo label 품질이 더 나빠져 이 현상이 심해지고, SSOD에서 DETR 계열이 잘 안 먹히는 이유가 됩니다.[^1_2][^1_1]


### DETR 쿼리 특성과 consistency regularization의 충돌

- Faster R-CNN/YOLO 계열은 anchor나 grid와 같이 입력 feature 위치와 출력 box 간의 대응이 비교적 결정적이라, “같은 이미지의 weak/strong augmentation에 대해 출력 box를 일치시키는” consistency loss를 적용하기 쉽습니다.[^1_1][^1_2]
- 반면 DETR 계열은 **학습 가능한 object query**를 입력으로 쓰고, self-/cross-attention으로 query feature와 대응 box가 학습 중 계속 바뀝니다.[^1_2][^1_1]
- 따라서 “teacher의 i번째 box ↔ student의 j번째 box” 같은 안정적 매칭을 전제한 기존 consistency 스킴(예: Soft Teacher, Dense Teacher, PseCo 등)을 그대로 쓰기 어렵고, DETR 전용 consistency 설계가 필요합니다.[^1_1][^1_2]

***

## 제안 방법과 주요 수식

### Stage-wise Hybrid Matching (SHM)

#### 1단계: one-to-one 매칭 (기본 형식)

기존 DETR 스타일의 pseudo label–prediction 간 one-to-one 매칭은 헝가리안 알고리즘으로 다음과 같이 정의됩니다.[^1_2][^1_1]

$$
\hat{\sigma}_{\text{o2o}} 
= \arg\min_{\sigma \in \Xi_N} 
\sum_{i=1}^{N} \mathcal{C}_{\text{match}}\left(\hat{y}^{t}_{i}, \hat{y}^{s}_{\sigma(i)}\right),
$$

여기서 $\hat{y}^t_i$는 teacher가 생성한 pseudo box, $\hat{y}^s_j$는 student의 예측, $\Xi_N$은 길이 $N$인 순열 집합입니다.[^1_1][^1_2]

#### 1단계: one-to-many 매칭

초기에는 pseudo box가 부정확하므로, 각 pseudo box에 대해 **여러 개의 양성 proposal**을 할당하는 one-to-many 매칭을 사용합니다.[^1_2][^1_1]

```math
\{\boldsymbol{\sigma}_i\}_{\text{o2m}}
=
\left\{
\arg\min_{\boldsymbol{\sigma}_i \in C^M_N}
\sum_{j=1}^{M} 
\mathcal{C}_{\text{match}}\left(\hat{y}^{t}_{i}, \hat{y}^{s}_{\boldsymbol{\sigma}_i(j)}\right)
\right\}_{i=1}^{|\hat{y}^t|},
```

여기서 $C^M_N$은 $N$개 중 $M$개를 고르는 조합, 즉 각 pseudo box마다 $M$개의 proposal을 positive로 고르는 것을 의미합니다.[^1_1][^1_2]

매칭 score는 분류 score $s$와 IoU $u$의 고차 조합으로 정의합니다.[^1_2][^1_1]

$$
m = s^{\alpha} u^{\beta},
$$

논문에서는 $\alpha = 1, \beta = 6$을 사용하며, 각 pseudo box에 대해 $m$이 큰 상위 $M$개 proposal을 positive로 선택합니다.[^1_1][^1_2]

#### 1단계 손실 (o2m)

양성/음성 수가 크게 달라지는 one-to-many 세팅에 맞춰 classification/regression loss를 가중합니다.[^1_2][^1_1]

```math
\mathcal{L}^{\text{o2m}}_{\text{cls}}
=
\sum_{i=1}^{N_{\text{pos}}}
|\hat{m}_i - s_i|^{\gamma}
\text{BCE}(s_i, \hat{m}_i)
+
\sum_{j=1}^{N_{\text{neg}}}
s_j^{\gamma}
\text{BCE}(s_j, 0),
```

```math
\mathcal{L}^{\text{o2m}}_{\text{reg}}
=
\sum_{i=1}^{N_{\text{pos}}}
\hat{m}_i \mathcal{L}_{\text{GIoU}}(b_i, \hat{b}_i)
+
\sum_{i=1}^{N_{\text{pos}}}
\hat{m}_i \mathcal{L}_{L_1}(b_i, \hat{b}_i),
```

```math
\mathcal{L}^{\text{o2m}}
=
\mathcal{L}^{\text{o2m}}_{\text{cls}}
+
\mathcal{L}^{\text{o2m}}_{\text{reg}}.
```

여기서 $\hat{m}_i$는 해당 proposal의 매칭 score 정규화 버전, $\gamma$는 Focal Loss와 유사한 조절 파라미터(기본값 2)입니다.[^1_1][^1_2]

핵심 아이디어는 “pseudo box 근처에 여러 좋은 proposal이 있을 때, 그 중 하나만 positive로 두지 말고, 상위 다수를 모두 positive로 최적화에 참여시켜서 **수렴을 빠르게 하고 pseudo label 품질도 끌어올리자**”는 것입니다.[^1_2][^1_1]

#### 2단계: 다시 one-to-one으로 전환

1단계에서 학습이 진행되면 Teacher의 pseudo label 품질이 높아지므로, 이후에는 **원래 DETR의 one-to-one 매칭으로 되돌아가 NMS-free DETR의 장점**을 회복합니다.[^1_1][^1_2]

2단계 손실은 DINO 등 기존 DETR 계열과 동일한 형태의 one-to-one 기반 손실 $\mathcal{L}^{\text{o2o}}$를 사용합니다.[^1_2][^1_1]

***

### Cross-view Query Consistency (CQC)

CQC는 DETR에 맞춘 “쿼리 수준 consistency”입니다.[^1_1][^1_2]

#### RoI 기반 cross-view query 생성

unlabeled 이미지에 대해 teacher/student에 서로 다른 augmentation(weak/strong)을 적용하여 feature map $F_t, F_s$를 얻고, pseudo box $b$에 대해 RoIAlign으로 지역 feature를 추출합니다.[^1_2][^1_1]

$$
\begin{aligned}
c_t &= \text{MLP}(\text{RoIAlign}(F_t, b)), \\
c_s &= \text{MLP}(\text{RoIAlign}(F_s, b)).
\end{aligned}
$$

이 두 벡터를 **cross-view query embedding**으로 보고, teacher/ student 디코더에 서로 교차로 주입합니다.[^1_1][^1_2]

$$
\begin{aligned}
\hat{o}_t, o_t &= \text{Decoder}_t([c_s, q_t], E_t \mid A), \\
\hat{o}_s, o_s &= \text{Decoder}_s([c_t, q_s], E_s \mid A),
\end{aligned}
$$

여기서 $q_t, q_s$는 원래 object query, $E_t, E_s$는 인코더 출력, $A$는 DN-DETR 스타일의 attention mask입니다.[^1_2][^1_1]

#### consistency loss

이렇게 얻은 cross-view query 출력 $\hat{o}_t, \hat{o}_s$는 서로 같은 객체를 다른 view에서 본 표현이므로, MSE 기반 consistency loss를 줍니다.[^1_1][^1_2]

$$
\mathcal{L}_c 
= \text{MSE}\bigl(\hat{o}_s, \text{detach}(\hat{o}_t)\bigr).
$$

teacher 쪽은 gradient를 끊어(student만 업데이트), teacher는 EMA로만 갱신되는 전형적인 Mean Teacher 스타일을 따릅니다.[^1_3][^1_2][^1_1]

이 구조 덕분에 “쿼리 인덱스와 box 간의 결정적 매핑” 없이도, 동일한 pseudo box 기반 cross-view query가 teacher와 student 양쪽에서 의미적으로 일관된 feature를 내도록 학습할 수 있습니다.[^1_2][^1_1]

***

### Cost-based Pseudo Label Mining (CPM)

CQC에 사용할 pseudo box는 많을수록 좋지만, 품질이 너무 나쁘면 consistency가 노이즈에 맞춰지는 문제가 생깁니다.  CPM은 **matching cost 분포를 이용해 더 많은 ‘신뢰 가능한’ pseudo box를 동적으로 뽑는 모듈**입니다.[^1_1][^1_2]

#### matching cost 정의

초기 pseudo box(예: score threshold $\tau_s = 0.4$로 필터링)를 teacher에서 얻은 뒤, student 예측과 다시 bipartite matching을 수행하고, 각 pseudo box–prediction 쌍 $(i, j)$에 대해 다음의 cost를 계산합니다.[^1_2][^1_1]

```math
C_{ij}
= 
\lambda_1 C_{\text{Cls}}(p_i, \hat{p}_j)
+
\lambda_2 C_{\text{GIoU}}(b_i, \hat{b}_j)
+
\lambda_3 C_{L_1}(b_i, \hat{b}_j),
```

여기서 $p_i, b_i$는 i번째 prediction의 class/box, $\hat{p}_j, \hat{b}_j$는 j번째 pseudo box의 class/box입니다.[^1_1][^1_2]

#### GMM 기반 신뢰도 추정

- batch 내 모든 pseudo box에 대한 cost $C$를 모아 보면 대략 **이봉(bimodal) 분포**를 보이므로, 이를 두 개의 Gaussian으로 이루어진 GMM으로 모델링합니다.[^1_2][^1_1]
- 하나의 Gaussian은 “신뢰할 수 있는 pseudo box(낮은 cost)”, 다른 하나는 “불신뢰 pseudo box(높은 cost)”에 대응하는 것으로 해석합니다.[^1_1][^1_2]

GMM 파라미터 $\theta$에 대해, pseudo box가 “reliable cluster”에 속할 확률을 최대로 하는 cost $\tau_c$를 threshold로 삼고, $C_{ij} \le \tau_c$인 pseudo box만 CQC에 사용합니다.[^1_2][^1_1]

$$
\tau_c = \arg\max_c P_{\text{reliable}}(c \mid c, \theta).
$$

이렇게 하면, 단순 score threshold보다 **precision–recall 트레이드오프가 좋은 pseudo box 집합**을 얻어 consistency 학습에 더 많은 유효 예시를 공급할 수 있습니다.[^1_1][^1_2]

***

### 전체 손실 함수

최종 손실은 다음과 같이 정의됩니다.[^1_2][^1_1]

$$
\begin{aligned}
\mathcal{L}
&=
\mathbb{I}(t \le T_1)
\cdot
\bigl(
\mathcal{L}^{\text{o2m}}_{\text{sup}}
+
w_u \mathcal{L}^{\text{o2m}}_{\text{unsup}}
\bigr)
\\
&\quad+
\mathbb{I}(t > T_1)
\cdot
\bigl(
\mathcal{L}^{\text{o2o}}_{\text{sup}}
+
w_u \mathcal{L}^{\text{o2o}}_{\text{unsup}}
\bigr)
+
w_c \mathcal{L}_c,
\end{aligned}
$$

여기서 $t$는 현재 iteration, $T_1$은 1단계(one-to-many) 학습 기간, $w_u, w_c$는 unlabeled loss 및 consistency loss의 가중치입니다(기본값 $w_u=4, w_c=1$).[^1_1][^1_2]

***

## 모델 구조

- 기본 프레임워크는 Soft Teacher류와 유사한 **Teacher–Student DETR 아키텍처**입니다.[^1_3][^1_2][^1_1]
- Detector는 Deformable DETR 또는 DINO 기반의 encoder–decoder 구조에 ResNet-50 backbone을 사용합니다.[^1_2][^1_1]
- 학습 루프 구조:
    - Labeled 데이터: 학생 모델에 supervision($\mathcal{L}^{\text{o2m/o2o}}_{\text{sup}}$).
    - Unlabeled 데이터:
        - Teacher(EMA 업데이트)가 weak augmentation 입력에 대해 pseudo box 생성.
        - Student가 strong augmentation 입력에 대해 예측.
        - Stage-wise Hybrid Matching으로 pseudo box ↔ student prediction을 매칭해 SSOD loss($\mathcal{L}^{\text{o2m/o2o}}_{\text{unsup}}$).
        - CPM으로 선별된 pseudo box를 사용해 CQC 모듈에서 consistency loss($\mathcal{L}_c$).
- Inference 시에는 Teacher 없이 **Student DETR만 사용하는, 완전히 end-to-end NMS-free detector**입니다(단, 1단계에서만 NMS 사용).[^1_1][^1_2]

***

## 성능 향상 및 한계

### 성능 향상 (정량)

COCO-Partial(1/5/10% 라벨)에서 Semi-DETR는 다음과 같이 기존 SSOD 대비 큰 폭의 향상을 보여줍니다.[^1_2][^1_1]

- Deformable DETR 기반 (COCO-Partial, COCO-style mAP):
    - Sup only: 11.0 / 23.7 / 29.2
    - 단순 DETR+SSOD baseline: 19.4 / 31.1 / 34.8
    - Dense Teacher: 22.38 / 33.01 / 37.13
    - **Semi-DETR**: 25.2 / 34.5 / 38.1 (Dense Teacher 대비 +2.82 / +1.49 / +0.97 mAP).[^1_1][^1_2]
- DINO 기반 (동일 설정):
    - Sup only: 18.0 / 29.5 / 35.0
    - DINO SSOD baseline: 28.4 / 38.0 / 41.6
    - Omni-DETR(DINO): 27.6 / 37.7 / 41.3
    - **Semi-DETR(DINO)**: 30.5 / 40.1 / 43.5 (baseline 대비 +2.1 / +2.1 / +1.9 mAP).[^1_2][^1_1]

COCO-Full(100% 라벨 + COCO unlabeled)에서도, Deformable DETR 기준 PseCo, Dense Teacher가 46.1 mAP인 반면 Semi-DETR는 47.2 mAP, DINO 기반에서는 50.4 mAP로 더 큰 차이(+4.3 mAP)를 보입니다.[^1_1][^1_2]

Pascal VOC에서도 Deformable DETR 기준 AP50 74.5 → 83.5, AP50:95 46.2 → 57.2, DINO 기준 AP50 81.2 → 86.1, AP50:95 59.6 → 65.2로 크게 향상됩니다.[^1_2][^1_1]

### 성능 향상 (정성/설계 관점)

- SHM: 라벨 비율이 1%일 때 특히 큰 이득을 주며, one-to-one만 쓸 때보다 수렴 속도와 최종 성능이 모두 개선됩니다.[^1_1][^1_2]
- CQC + CPM: pseudo label 필터링 전략(고정 threshold, Top-K, mean+std) 간 비교에서, CPM(GMM)을 사용할 때 mAP와 pseudo label precision–recall이 가장 좋은 균형을 보입니다.[^1_2][^1_1]


### 한계

- **one-to-many만 쓰는 경우가 성능은 더 높음**: 1, 2단계 모두 one-to-many 전략(즉, 완전 NMS 기반)으로 유지하면 mAP가 더 오르지만, 이때는 DETR의 NMS-free 장점을 잃습니다.[^1_1][^1_2]
- CPM pseudo box를 consistency가 아닌 **회귀/분류 supervisory label**로 쓰면 오히려 성능이 떨어지는 등, pseudo box 용도에 따라 민감한 거동을 보입니다.[^1_2][^1_1]
- 논문 실험은 COCO/VOC 같은 범용 도메인에 국한되어 있고, 실제 도메인 시프트(예: 의료, 자율주행, 산업 검사) 상황에서의 일반화는 직접 평가되지 않았습니다.[^1_1]

***

## 일반화 성능 향상 가능성 (중점)

### 멀티 detector 일반화

- Semi-DETR는 Deformable DETR와 DINO라는 서로 다른 DETR 계열 백본 모두에서 **일관되게 큰 향상**을 보여, SSOD 개선이 특정 백본에 한정되지 않음을 시사합니다.[^1_2][^1_1]
- COCO-Partial/Full, VOC 모든 설정에서 baseline 대비 1.8∼14.2 mAP까지 향상되는 점은, unlabeled 데이터가 많은 저라벨 환경에서 특히 일반화 성능을 강하게 끌어올릴 수 있음을 보여줍니다.[^1_1][^1_2]


### pseudo-label 품질과 일반화

- SHM의 one-to-many 1단계는 “실제 GT에 더 가까운 proposal도 함께 positive로 학습”하게 하므로, 잘못된 pseudo box에 과도하게 맞추는 overfitting을 완화합니다.[^1_1]
- CPM은 classification/regression cost를 통합한 matching cost를 기준으로 pseudo box를 re-select함으로써, 높은 score지만 박스 위치가 틀어진 pseudo box를 걸러낼 수 있고, 이는 localization 일반화에 유리합니다.[^1_1]
- CQC는 **feature-level 의미 불변성**을 학습하므로, 작은 변형/증강(스케일·회전·컬러)에 대해서도 안정된 표현을 유도해, 실환경 변동(조명·뷰포인트 등)에 대한 generalization을 이론적으로 뒷받침합니다.[^1_2][^1_1]


### 후속 연구에서의 일반화 검증

- 2024년 Sparse Semi-DETR는 쿼리 품질과 pseudo-label filtering을 개선해 작은/가려진 객체에 대한 성능을 더 끌어올렸고, Semi-DETR류가 **어려운 샘플(소형·occlusion)에 대한 일반화**에도 적합한 틀임을 보였습니다.[^1_4][^1_5][^1_6]
- 2024년 Sized L1 Loss 논문은 Semi-DETR에 크기 정규화 회귀 손실을 도입해 특히 small object mAP를 추가로 향상시켜, Semi-DETR 기반 SSOD가 손실 설계 변경만으로도 다양한 크기에 일반화 가능한 구조임을 입증했습니다.[^1_7]
- 2024년 이후 SSOD survey 및 실증 연구들은 Semi-DETR가 MixPL, Consistent-Teacher 등과 함께 라벨 효율·성능·모델 크기 사이에서 좋은 트레이드오프를 제공한다고 보고하며, 소수 라벨(low-data regime)에서 일관된 일반화 성능을 보인다는 점을 강조합니다.[^1_8][^1_9]

요약하면, Semi-DETR 자체 실험과 후속 논문을 종합할 때, **“라벨이 매우 적고 unlabeled가 많은 환경, 복잡한 배경과 다양한 스케일의 객체”에 대한 일반화**에 특히 강점이 있는 프레임워크로 볼 수 있습니다.[^1_4][^1_8][^1_1]

***

## 2020년 이후 관련 최신 연구 비교

### 대표 SSOD 방법(2020+)과의 위치

아래 표는 2020년 이후 주요 SSOD 연구 일부와 Semi-DETR의 위치를 정성적으로 비교한 것입니다 (모두 arXiv/CFV open access).[^1_10][^1_5][^1_9][^1_8][^1_4][^1_3][^1_2][^1_1]


| 방법 | 연도 / Backbone | Detector 유형 | 주요 아이디어 | DETR 기반 여부 |
| :-- | :-- | :-- | :-- | :-- |
| STAC | 2020, Faster R-CNN | 2-stage | multi-stage pseudo labeling + consistency | X |
| Soft Teacher | 2021, Faster R-CNN, Swin | 2-stage | end-to-end Teacher–Student, soft-weighted cls loss, box jittering | X |
| Omni-DETR | 2022, Deformable DETR | DETR | omni-supervised (weak/strong labels 혼합), 간단 pseudo filtering | △ (SSOD 전용은 아님) |
| Dense Teacher | 2022, RetinaNet 계열 | 1-stage | dense feature-level pseudo label, threshold free | X |
| DSL | 2022, FCOS | 1-stage | dense learning + adaptive filtering + uncertainty regularization | X |
| PseCo | 2022, Faster R-CNN | 2-stage | feature pyramid scale-consistency, 개선된 pseudo labeling | X |
| **Semi-DETR** | 2023, Deformable DETR/DINO | DETR | SHM + CQC + CPM으로 DETR 전용 SSOD 설계 | **O (2D DETR 최초 SSOD)** |
| Sparse Semi-DETR | 2024, DETR | DETR | query refinement + 개선된 pseudo filtering로 small/occluded 개선 | O |
| Consistent-Teacher | 2023, CNN | 2-stage | 개선된 consistency regularization (Soft Teacher 계열) | X |
| 비교 실증 연구 (MixPL, Semi-DETR, Consistent-Teacher) | 2026 | - | 모델 크기·지연·few-shot 세팅에서 세 방법을 체계 비교 | Semi-DETR 포함[^1_8] |

주요 관찰점:[^1_9][^1_10][^1_8][^1_4][^1_3][^1_2]

- CNN 기반 SSOD(Soft Teacher, Dense Teacher, DSL, PseCo 등)는 anchor·NMS에 의존하고 구조가 복잡한 반면, Semi-DETR는 손수 설계 컴포넌트를 제거한 **end-to-end DETR 계열**로서 보다 단순한 파이프라인을 제안합니다.
- Omni-DETR는 DETR 기반이지만 목표가 omni-supervised(weak label 포함)이며, SSOD 관점의 consistency 설계나 matching 전략은 Semi-DETR만큼 특화되어 있지 않습니다.[^1_11][^1_10][^1_2]
- Sparse Semi-DETR는 Semi-DETR의 한계(소형·occluded object, query 품질 등)를 인식하고 Query Refinement / Reliable Pseudo-Label Filtering을 도입하여 Semi-DETR 계열 발전 방향을 제시합니다.[^1_5][^1_6][^1_4]
- 2024 survey 및 2026 실증 비교 논문은 Semi-DETR를 “DETR 기반 SSOD의 사실상 표준 베이스라인” 중 하나로 취급하며, 다양한 라벨 비율·도메인에서 성능–비용 트레이드오프가 우수하다고 보고합니다.[^1_8][^1_9]

***

## 앞으로의 연구에 미치는 영향과 고려할 점

### 영향

1. **DETR 전용 SSOD 설계 패턴 제시**
    - SHM + CQC + CPM 조합은 단순 pseudo labeling + consistency를 넘어, **DETR 구조적 특성(쿼리–출력 비결정성, one-to-one 매칭)을 전제로 한 SSOD 설계**의 출발점을 제공합니다.[^1_2][^1_1]
    - 이후 3D Semi-3DETR, Sparse Semi-DETR 등 2D·3D DETR 기반 SSOD 후속 연구들이 대부분 “혼합 매칭 전략 + 쿼리 수준 consistency + pseudo label filtering”을 공통 모티브로 채택하고 있습니다.[^1_12][^1_5][^1_4]
2. **라벨 효율 및 저라벨 시나리오 연구의 기준선**
    - COCO 1% 라벨에서 큰 향상을 보이는 Semi-DETR는, 향후 few-shot / low-resource 객체 검출 실험에서 “SSOD + DETR” 조합 평가 시 자주 인용되는 strong baseline이 되었습니다.[^1_8][^1_1]
3. **일반화와 손실/매칭 설계 연구 촉진**
    - Sized L1 loss, 다양한 hybrid matching 변형(예: SimOTA vs Max-IoU vs ATSS 비교) 등, DETR 기반 SSOD에서 **손실과 매칭 전략이 일반화·소형 객체 성능에 미치는 영향**을 후속 연구가 더 정밀하게 탐구하게 되는 계기를 제공했습니다.[^1_7][^1_1]

### 향후 연구 시 고려할 점

1. **도메인 시프트와 open-set/long-tail 환경**
    - Semi-DETR는 일반 도메인(COCO/VOC)에서는 우수하지만, 개체 분포가 편향된 real-world(open-set, long-tail, 도메인 시프트) 상황에서는 pseudo label noise 특성이 달라질 수 있습니다.[^1_13][^1_14][^1_1]
    - 향후 연구에서는 CPM을 domain-aware하게 수정하거나, open-set SSOD (예: SS-OWFormer 등)와 결합해 **알려지지 않은 클래스 및 분포 이동**을 처리하는 방향이 유망합니다.[^1_15][^1_14][^1_9]
2. **소형·occluded 객체와 복잡한 배경**
    - Sparse Semi-DETR가 보여주듯, query 품질 향상과 더 정교한 pseudo-label filtering은 특히 작은/가려진 객체에서 중요합니다.[^1_5][^1_4]
    - Semi-DETR를 기반으로, multi-scale feature 강화, super-resolution, occlusion-aware consistency 등을 결합하면 일반화 성능을 더 끌어올릴 수 있습니다.
3. **계산 비용·지연과 실시간 적용**
    - DETR 계열은 본질적으로 heavy한 구조이므로, 실시간/엣지 환경에서는 MixPL, YOLO 기반 SSOD보다 불리할 수 있습니다.[^1_16][^1_8]
    - 향후 연구에서는 Semi-DETR 아이디어를 경량 DETR(Efficient DETR 등)과 결합하거나, 쿼리 pruning·규모 축소를 통한 **latency–성능 트레이드오프 최적화**가 필요합니다.
4. **3D·멀티모달 확장**
    - Semi-3DETR 등 3D DETR 기반 SSOD는 2D Semi-DETR 아이디어를 3D로 확장하면서, 쿼리 alignment와 pseudo label denoising, hybrid matching을 함께 사용합니다.[^1_12]
    - LiDAR·RGB-fusion, multi-view 카메라 등 멀티모달 환경에서도 SHM·CQC·CPM류 모듈을 어떻게 확장할 것인지가 중요한 개방 문제입니다.
5. **이론적 분석과 최적성**
    - SHM의 one-to-many → one-to-one 전환 시점 $T_1$, pseudo label threshold $\tau_s$, GMM 기반 threshold $\tau_c$ 등은 현재 경험적으로 설정되어 있어, 이들의 **이론적 최적성·일반화 bound**에 대한 분석이 앞으로의 연구 과제입니다.[^1_1]

***

## 참고 문헌 및 링크 (모두 오픈 액세스)

- Jiacheng Zhang et al., “Semi-DETR: Semi-Supervised Object Detection with Detection Transformers,” CVPR 2023.[^1_17][^1_2][^1_1]
- Mengde Xu et al., “End-to-End Semi-Supervised Object Detection with Soft Teacher,” ICCV 2021.[^1_3]
- Pei Wang et al., “Omni-DETR: Omni-Supervised Object Detection with Transformers,” CVPR 2022.[^1_10][^1_11]
- Hongyu Zhou et al., “Dense Teacher: Dense Pseudo-Labels for Semi-Supervised Object Detection,” arXiv 2022.[^1_2]
- Binghui Chen et al., “DSL: Dense Learning Based Semi-Supervised Object Detection,” CVPR 2022.[^1_2]
- Gang Li et al., “PseCo: Pseudo Labeling and Consistency Training for Semi-Supervised Object Detection,” arXiv 2022.[^1_2]
- Tahira Shehzadi et al., “Sparse Semi-DETR: Sparse Learnable Queries for Semi-Supervised Object Detection,” CVPR 2024.[^1_6][^1_4][^1_5]
- (익명) “Practical Insights into Semi-Supervised Object Detection Approaches,” 2026 preprint (MixPL, Semi-DETR, Consistent-Teacher 비교).[^1_8]
- (익명) “Semi-Supervised Object Detection: A Survey on Progress from CNN to Transformers,” 2024 survey.[^1_9]
- (익명) “Co-Learning: Towards Semi-Supervised Object Detection with Road-side Cameras,” 2024.[^1_13]
- (익명) “Semi-supervised object detection with uncurated unlabeled data for real-world scenarios,” 2024.[^1_14]
- (익명) “Semi-3DETR: Semi-Supervised Detection Transformer for 3D Object Detection,” ICLR 2026 submission.[^1_12]
- (익명) “A Normalized Regression Loss for DETR-based Object Detection in Fully- and Semi-Supervised Settings (Sized L1 Loss),” 2024.[^1_7]
- JCZ404/Semi-DETR 공식 구현 GitHub 저장소.[^1_18]
<span style="display:none">[^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29]</span>

<div align="center">⁂</div>

[^1_1]: 2307.08095v1.pdf

[^1_2]: https://ar5iv.labs.arxiv.org/html/2307.08095

[^1_3]: https://arxiv.org/abs/2106.09018

[^1_4]: https://arxiv.org/html/2404.01819v1

[^1_5]: https://arxiv.org/pdf/2404.01819.pdf

[^1_6]: https://cvpr.thecvf.com/virtual/2024/poster/30138

[^1_7]: https://www.arxiv.org/pdf/2410.22638.pdf

[^1_8]: https://arxiv.org/pdf/2601.13380.pdf

[^1_9]: https://arxiv.org/html/2407.08460v2

[^1_10]: https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Omni-DETR_Omni-Supervised_Object_Detection_With_Transformers_CVPR_2022_paper.pdf

[^1_11]: https://www.semanticscholar.org/paper/Omni-DETR:-Omni-Supervised-Object-Detection-with-Wang-Cai/59232131a251e19a03cb45f593196b56d2661c86

[^1_12]: https://openreview.net/forum?id=N1OG2t1OvX

[^1_13]: https://arxiv.org/html/2411.19143v1

[^1_14]: https://www.sciencedirect.com/science/article/pii/S1569843224001687

[^1_15]: https://arxiv.org/html/2402.16013v1

[^1_16]: https://opencodepapers-b7572d.gitlab.io/benchmarks/semi-supervised-object-detection-on-coco-1.html

[^1_17]: https://arxiv.org/pdf/2307.08095.pdf

[^1_18]: https://github.com/JCZ404/Semi-DETR

[^1_19]: https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Semi-DETR_Semi-Supervised_Object_Detection_With_Detection_Transformers_CVPR_2023_paper.pdf

[^1_20]: https://openaccess.thecvf.com/content/CVPR2023/supplemental/Zhang_Semi-DETR_Semi-Supervised_Object_CVPR_2023_supplemental.pdf

[^1_21]: https://arxiv.org/abs/2307.08095

[^1_22]: https://www.semanticscholar.org/paper/Semi-DETR:-Semi-Supervised-Object-Detection-with-Zhang-Lin/4c0d2894571b37da2b44bb4dab0a562e32ae66fd

[^1_23]: https://www.amazon.science/publications/omni-detr-omni-supervised-object-detection-with-transformers

[^1_24]: https://liner.com/ko/review/omnidetr-omnisupervised-object-detection-with-transformers

[^1_25]: https://cvpr.thecvf.com/virtual/2023/poster/23144

[^1_26]: https://cvpr.thecvf.com/media/cvpr-2024/Slides/30138.pdf

[^1_27]: https://github.com/microsoft/SoftTeacher

[^1_28]: https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Semi-DETR_Semi-Supervised_Object_Detection_With_Detection_Transformers_CVPR_2023_paper.html

[^1_29]: https://liner.com/ko/review/semidetr-semisupervised-object-detection-with-detection-transformers

