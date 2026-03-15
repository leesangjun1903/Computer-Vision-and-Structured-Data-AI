# A Strong and Reproducible Object Detector with Only Public Datasets

---

## 1. 핵심 주장 및 주요 기여 요약

본 논문은 **Focal-Stable-DINO**라는 객체 탐지 모델을 제안하며, 핵심 주장은 다음과 같다:

> **대규모 비공개(private) 데이터, 복잡한 사전학습 기법(masked image modeling, image-text contrastive learning), 테스트 시간 증강(TTA) 없이도, 오직 공개 데이터셋만으로 COCO 벤치마크에서 SOTA(State-of-the-Art)에 준하는 성능을 달성할 수 있다.**

### 주요 기여

| # | 기여 내용 |
|---|---------|
| 1 | **공개 자원만으로 재현 가능한 강력한 검출기 구축**: FocalNet-Huge 백본 + Stable-DINO 검출 헤드 조합 |
| 2 | **COCO val2017에서 64.6 AP, test-dev에서 64.8 AP** 달성 (TTA 없음, 마스크 어노테이션 미사용) |
| 3 | 비공개 데이터·복잡한 학습 파이프라인에 의존하는 기존 SOTA 모델들에 대한 **재현 가능한 대안** 제시 |
| 4 | COCO 데이터셋 어노테이션의 **오류, 누락, 비일관성** 문제를 체계적으로 분석하여 향후 개선 방향 제시 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

최근 객체 탐지 분야에서 SOTA 성능을 달성한 모델들(EVA, BEiT-3, InternImage 등)은 다음과 같은 문제를 가진다:

1. **비공개 데이터 의존**: ImageNet-22K-ext (70M 비공개 이미지), 대규모 image-text 쌍 등 접근이 제한된 데이터 활용
2. **복잡한 학습 파이프라인**: Masked Image Modeling (MAE, BEiT), contrastive learning 등 정교한 사전학습 기법 필요
3. **재현성 부족**: 위의 요인들로 인해 연구 커뮤니티에서 결과를 재현하기 어려움
4. **과도한 파라미터**: 일부 모델은 1B~3B 파라미터를 사용

### 2.2 제안하는 방법

Focal-Stable-DINO는 두 가지 공개 가용 구성 요소를 결합하는 **단순하지만 효과적인 전략**을 채택한다:

#### (A) 백본: FocalNet-Huge

FocalNet [26]은 Focal Modulation 메커니즘을 사용하는 비전 백본으로, self-attention 대신 **focal modulation**을 통해 다양한 스케일의 컨텍스트를 집약한다. FocalNet-Huge는 689M 파라미터를 가지며 ImageNet-22K에서만 사전학습되었다.

Focal Modulation의 핵심 연산은 다음과 같다. 입력 특징 $\mathbf{x}$에 대해:

$$\mathbf{y} = \text{FocalModulation}(\mathbf{x}) = q(\mathbf{x}) \odot h\left(\sum_{l=1}^{L} g_l(z_l(\mathbf{x})) + g_0(\mathbf{x})\right)$$

여기서:
- $q(\mathbf{x})$: query 프로젝션
- $z_l(\mathbf{x})$: 레벨 $l$에서의 focal contextualization (계층적 컨텍스트 집약)
- $g_l(\cdot)$: 게이팅 함수
- $h(\cdot)$: 최종 프로젝션
- $\odot$: element-wise 곱셈
- $L$: focal 레벨 수

#### (B) 검출 헤드: Stable-DINO

Stable-DINO [16]는 DINO [30] 기반의 DETR 변형으로, 디코더 레이어 간 **매칭 불안정성(matching instability)** 문제를 해결한다. 핵심 기법은 두 가지이다:

**1) Position-Supervised Loss (위치 감독 손실)**

DINO에서는 각 디코더 레이어마다 독립적인 Hungarian matching을 수행하는데, 이로 인해 서로 다른 레이어에서 동일 쿼리가 서로 다른 GT에 매칭되는 **multi-optimization paths** 문제가 발생한다. Stable-DINO는 이를 해결하기 위해 position-supervised loss를 도입한다:

$$\mathcal{L}_{\text{pos}} = \sum_{i=1}^{N_q} \mathbb{1}[\sigma_i^{(k)} \neq \sigma_i^{(K)}] \cdot \mathcal{L}_{\text{box}}(\hat{b}_i^{(k)}, b_{\sigma_i^{(K)}})$$

여기서:
- $\sigma_i^{(k)}$: 레이어 $k$에서 쿼리 $i$에 대한 매칭 인덱스
- $\sigma_i^{(K)}$: 마지막 디코더 레이어 $K$에서의 매칭 인덱스 (기준)
- $\hat{b}_i^{(k)}$: 레이어 $k$에서 쿼리 $i$의 예측 바운딩 박스
- $b_{\sigma_i^{(K)}}$: 마지막 레이어 매칭에 의한 GT 박스
- $\mathcal{L}_{\text{box}}$: 바운딩 박스 회귀 손실 (L1 + GIoU)

이를 통해 모든 디코더 레이어에서 마지막 레이어의 매칭을 따르도록 유도하여 **단일 최적화 경로(single optimization path)**를 보장한다.

**2) Position-Modulated Cost (위치 조절 비용)**

Hungarian matching 시 위치 정보를 활용하여 매칭 비용을 조절:

$$\mathcal{C}_{\text{match}}^{(k)} = \mathcal{C}_{\text{cls}} + \mathcal{C}_{\text{box}} + \lambda_{\text{pos}} \cdot \mathcal{C}_{\text{pos}}^{(k)}$$

여기서 $\mathcal{C}_{\text{pos}}^{(k)}$는 이전 레이어의 위치 예측을 기반으로 한 추가 비용 항이다.

**전체 학습 손실:**

$$\mathcal{L}_{\text{total}} = \sum_{k=1}^{K} \left( \lambda_{\text{cls}} \mathcal{L}_{\text{cls}}^{(k)} + \lambda_{\text{box}} \mathcal{L}_{\text{box}}^{(k)} + \lambda_{\text{giou}} \mathcal{L}_{\text{giou}}^{(k)} \right) + \lambda_{\text{pos}} \mathcal{L}_{\text{pos}} + \mathcal{L}_{\text{dn}}$$

여기서:
- $\mathcal{L}_{\text{cls}}$: 분류 손실 (Focal Loss)
- $\mathcal{L}_{\text{box}}$: L1 바운딩 박스 회귀 손실
- $\mathcal{L}_{\text{giou}}$: Generalized IoU 손실
- $\mathcal{L}_{\text{dn}}$: De-noising 손실 (DN-DETR [11]에서 도입)
- $\lambda_{\text{cls}} = 6.0$ (논문에서 명시)

### 2.3 모델 구조

```
┌─────────────────────────────────────────┐
│              Focal-Stable-DINO           │
├─────────────────────────────────────────┤
│  Input Image                             │
│      ↓                                   │
│  FocalNet-Huge Backbone (689M params)    │
│  - ImageNet-22K pretrained               │
│  - Focal Modulation (multi-scale)        │
│  - 4-stage hierarchical features         │
│      ↓                                   │
│  Multi-scale Feature Maps                │
│      ↓                                   │
│  Stable-DINO Detector                    │
│  - Deformable Transformer Encoder        │
│  - Deformable Transformer Decoder        │
│  - Position-Supervised Loss              │
│  - Position-Modulated Matching Cost      │
│  - 1000 De-noising Queries               │
│      ↓                                   │
│  Detection Outputs (class + bbox)        │
└─────────────────────────────────────────┘
```

**학습 파이프라인:**
1. **사전학습**: Objects365 (1.7M 이미지, 공개 데이터셋)에서 검출 사전학습
2. **미세조정**: COCO (118K 학습 이미지)에서 미세조정
3. **해상도**: 미세조정 시 $1.5\times$ 해상도 사용

### 2.4 성능 향상

#### 주요 성능 결과 (COCO)

| 모델 | 백본 | 파라미터 | 사전학습 데이터 | TTA | val AP | test AP |
|------|------|---------|------------|-----|--------|---------|
| DINO | Swin-L | 218M | IN-22K + O365 | ✓ | 63.2 | 63.3 |
| Stable-DINO | Swin-L | 218M | IN-22K + O365 | ✗ | 63.7 | 63.8 |
| FocalNet-DINO | Focal-Huge | 689M | IN-22K + O365 | ✗ | 64.0 | - |
| EVA-01 | EVA | 1.0B | merged-30M | ✗ | 64.2 | 64.4 |
| InternImage | InternImage-G | 602M | IN-22K + O365 | ✓ | 64.2 | 64.3 |
| **Focal-Stable-DINO** | **Focal-Huge** | **689M** | **IN-22K + O365** | **✗** | **64.6** | **64.8** |

**세부 성능 (COCO val2017):**

$$\text{AP} = 64.6, \quad \text{AP}_{50} = 81.5, \quad \text{AP}_{75} = 71.4$$

$$\text{AP}_S = 50.4, \quad \text{AP}_M = 68.5, \quad \text{AP}_L = 78.5$$

**성능 향상의 원천:**
- Stable-DINO (Swin-L) → Focal-Stable-DINO: **+0.9 AP** (val), **+1.0 AP** (test-dev) — 백본 교체 효과
- DINO → Stable-DINO: **+0.5 AP** (test-dev) — 매칭 안정화 효과
- TTA 없이도 TTA를 사용한 대부분의 모델을 능가

### 2.5 한계

1. **소형 객체 탐지 성능**: $\text{AP}_S = 50.4$로 중·대형 객체 대비 현저히 낮음 (약 18~28 AP 차이)
2. **카테고리별 성능 편차**: "Book" (AP: 35.9), "Banana" (AP: 40.5) 등 일부 카테고리에서 성능 저하
3. **배경 혼동(Background Confusion)**: 저성능 카테고리에서 배경 false positive가 주요 오류 원인
4. **위치 정확도 문제**: 고성능 카테고리에서도 localization error가 주요 개선 여지
5. **어노테이션 품질 의존**: COCO 데이터셋의 잘못된 어노테이션, 누락, 비일관적 레이블링 기준이 성능과 일반화에 부정적 영향
6. **파라미터 규모**: 689M 파라미터로 실시간 추론에는 부적합

---

## 3. 모델의 일반화 성능 향상 가능성

본 논문은 일반화 성능과 관련하여 여러 중요한 시사점을 제공한다:

### 3.1 공개 데이터만을 활용한 강건한 일반화

Focal-Stable-DINO의 핵심 강점은 **비공개 데이터 없이도 높은 일반화 성능**을 달성했다는 점이다:

- **Objects365** (365개 카테고리, 1.7M 이미지)에서 사전학습 → COCO (80개 카테고리)로 전이
- 이는 Objects365의 풍부한 카테고리 다양성이 일반적인 시각적 특징 학습에 충분함을 시사

### 3.2 백본의 전이 학습 능력

FocalNet-Huge의 **Focal Modulation** 메커니즘은 다양한 스케일의 컨텍스트를 효과적으로 포착:

$$z_l = \text{DepthwiseConv}_{k_l}(z_{l-1}), \quad l = 1, \ldots, L$$

이 계층적 컨텍스트 집약은 다양한 크기와 형태의 객체에 대한 **스케일 불변(scale-invariant) 특징 표현**을 가능하게 하여 일반화에 기여한다.

### 3.3 매칭 안정성과 일반화의 관계

Stable-DINO의 position-supervised loss는 학습 안정성을 개선하여 다음과 같은 일반화 이점을 제공:

1. **단일 최적화 경로 보장**: 디코더 레이어 간 일관된 매칭으로 학습 시 gradient 충돌 방지
2. **안정적 수렴**: 대규모 모델(689M)에서도 안정적으로 수렴하여 overfitting 위험 감소
3. **강건한 위치 예측**: position-modulated cost가 쿼리의 위치 일관성을 유지

### 3.4 일반화 향상을 위한 잠재적 방향

논문의 분석에서 도출되는 일반화 개선 가능성:

| 방향 | 설명 | 기대 효과 |
|------|------|---------|
| **어노테이션 품질 개선** | 오류, 누락, 비일관적 레이블 수정 | 올바른 데이터 분포 학습으로 일반화 향상 |
| **소형 객체 전용 메커니즘** | 고해상도 특징 강화, FPN 개선 | $\text{AP}_S$ 개선으로 전체 일반화 향상 |
| **배경 혼동 해결** | Hard negative mining, contrastive loss 강화 | 저성능 카테고리의 일반화 향상 |
| **다양한 공개 데이터셋 활용** | OpenImages, LVIS 등 추가 공개 데이터 | 카테고리 다양성 확대로 일반화 강화 |
| **데이터 증강 전략** | Copy-paste, Mosaic, MixUp 등 | 학습 데이터 다양성 증가 |

### 3.5 어노테이션 비일관성이 일반화에 미치는 영향

논문에서 구체적으로 지적한 세 가지 문제:

1. **잘못된 데이터 분포 학습 불가**: 비일관적 레이블링("banana"를 묶음 vs. 개별 표기)으로 모델이 올바른 데이터 분포를 학습하지 못함
2. **바운딩 박스 위치 모호성**: 동일 카테고리에 대한 상이한 기준으로 위치 예측의 불확실성 증가
3. **학습 불안정**: 상충하는 어노테이션이 gradient에 노이즈를 유발하여 수렴 방해

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 연구 커뮤니티에 미치는 영향

#### (1) 재현성(Reproducibility) 패러다임 전환
본 논문은 **"공개 데이터만으로도 충분히 강력한 모델을 만들 수 있다"**는 메시지를 명확히 전달한다. 이는 연구의 접근성과 공정성을 높이며, 비공개 데이터에 의존하는 연구 트렌드에 대한 건전한 반론을 제시한다.

#### (2) 모듈형 설계의 효용성 입증
백본(FocalNet-Huge)과 검출 헤드(Stable-DINO)를 독립적으로 발전시키고 조합하는 전략이 효과적임을 보여주어, 향후 **plug-and-play 방식의 모듈형 객체 탐지 연구**를 촉진한다.

#### (3) 벤치마크 공정성 문제 제기
기존 SOTA 모델들의 성능 비교 시 데이터 규모, 접근성, 학습 기법의 복잡성을 공정하게 고려해야 함을 강조한다.

### 4.2 향후 연구 시 고려할 점

1. **효율성과 성능의 균형**: 689M 파라미터는 여전히 대규모이며, 모델 압축(knowledge distillation, pruning, quantization)을 통한 효율화 연구 필요
2. **다양한 도메인으로의 확장**: COCO 외 다른 도메인(의료, 위성, 자율주행)에서의 일반화 성능 검증 필요
3. **공개 데이터셋의 품질 관리**: Objects365, COCO 등의 어노테이션 품질 개선이 성능 향상의 중요한 축
4. **학습 효율성**: Objects365 사전학습 + COCO 미세조정의 학습 비용 분석 및 최적화
5. **Open-vocabulary / Zero-shot 확장**: 공개 데이터만으로 어떻게 open-vocabulary 탐지까지 확장할 수 있는지 탐구

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 DETR 계열 발전사

| 시기 | 모델 | 핵심 기여 | COCO AP |
|------|------|---------|---------|
| 2020 | DETR (Carion et al.) | Transformer 기반 end-to-end 객체 탐지 도입 | 43.3 |
| 2021 | Deformable DETR (Zhu et al.) | Deformable attention으로 수렴 속도 개선 | 46.2 |
| 2022 | DAB-DETR [15] | Dynamic anchor box를 query로 활용 | 45.7 |
| 2022 | DN-DETR [11] | Query denoising으로 학습 가속 | 48.6 |
| 2022 | DINO [30] | Denoising + contrastive denoising + mixed query | 63.3 (test-dev, Swin-L) |
| 2023 | Stable-DINO [16] | Position-supervised loss로 매칭 안정화 | 63.8 (test-dev) |
| 2023 | **Focal-Stable-DINO** | FocalNet-Huge + Stable-DINO 조합 | **64.8** (test-dev) |
| 2023 | Co-DETR [32] | Collaborative hybrid assignments | 64.5 (test-dev) |
| 2023 | Group-DETR-v2 [2] | Encoder-decoder pretraining | 64.5 (test-dev) |

### 5.2 대규모 비전 백본 비교

| 모델 | 파라미터 | 사전학습 | 핵심 메커니즘 | 데이터 접근성 |
|------|---------|---------|------------|-----------|
| Swin-V2-G [18] | 3.0B | IN-22K-ext (비공개) | Shifted window attention | ❌ 비공개 데이터 |
| EVA [6] | 1.0B | merged-30M (부분 공개) | Masked image modeling + CLIP | △ 부분 공개 |
| EVA-02 [5] | 304M | merged-38M (부분 공개) | MIM + distillation | △ 부분 공개 |
| InternImage-G [22] | 3.0B | IN-22K + O365 | Deformable convolutions | ✅ 공개 |
| BEiT-3 [23] | 1.9B | merged data (부분 비공개) | Multimodal MIM | ❌ 비공개 데이터 |
| **FocalNet-Huge [26]** | **689M** | **IN-22K** | **Focal modulation** | **✅ 완전 공개** |

### 5.3 최신 연구와의 심층 비교

#### (A) Co-DETR (2022) vs. Focal-Stable-DINO

Co-DETR [32]는 collaborative hybrid assignments training을 제안하여 DETR의 학습 효율성을 높였다.

- Co-DETR: 64.5 AP (test-dev, TTA 사용)
- Focal-Stable-DINO: **64.8 AP** (test-dev, TTA 미사용)
- **차별점**: Focal-Stable-DINO는 TTA 없이도 Co-DETR+TTA보다 높은 성능

#### (B) EVA / EVA-02 (2022-2023) vs. Focal-Stable-DINO

EVA 계열은 masked image modeling과 CLIP 기반 representation learning을 활용:

$$\mathcal{L}_{\text{EVA}} = \mathcal{L}_{\text{MIM}} + \lambda \mathcal{L}_{\text{CLIP}}$$

- EVA-02: 64.5 AP (test-dev, 마스크 어노테이션 사용)
- Focal-Stable-DINO: **64.8 AP** (test-dev, 마스크 어노테이션 미사용)
- **차별점**: EVA는 복잡한 MIM+CLIP 파이프라인과 merged 데이터를 사용하지만, Focal-Stable-DINO는 단순한 파이프라인만으로 동등 이상의 성능

#### (C) RT-DETR (2023, Zhao et al.) — 실시간 DETR

RT-DETR는 실시간 객체 탐지를 목표로 한 DETR 변형:
- Focal-Stable-DINO와 상보적: 성능 vs. 속도 트레이드오프의 다른 지점
- 향후 Focal-Stable-DINO의 경량화에 RT-DETR의 기법 적용 가능

#### (D) Grounding DINO (2023, Liu et al.) — Open-set 탐지

Grounding DINO는 텍스트-이미지 grounding을 통한 open-set 객체 탐지:
- Focal-Stable-DINO와 같은 DINO 기반이나 목적이 다름
- **향후 연구 방향**: Focal-Stable-DINO의 강력한 closed-set 성능을 open-vocabulary로 확장

#### (E) DINO v2 (2023, Meta) — Self-supervised 비전 백본

DINOv2는 자기지도 학습 기반의 범용 비전 백본:
- FocalNet-Huge 대신 DINOv2 백본을 Stable-DINO에 결합하는 실험이 흥미로운 연구 방향
- 단, DINOv2의 사전학습 데이터(LVD-142M)의 접근성 확인 필요

### 5.4 종합 비교 테이블

| 모델 | 연도 | AP (test-dev) | 파라미터 | TTA | 비공개 데이터 | 복잡한 사전학습 | 재현성 |
|------|------|-------------|---------|-----|---------|------------|------|
| DINO (Swin-L) | 2022 | 63.3 | 218M | ✓ | ✗ | ✗ | ✅ 높음 |
| Stable-DINO (Swin-L) | 2023 | 63.8 | 218M | ✗ | ✗ | ✗ | ✅ 높음 |
| EVA-01 | 2022 | 64.7 | 1.0B | ✓ | △ | ✓ (MIM+CLIP) | △ 중간 |
| EVA-02 | 2023 | 64.5 | 304M | ✗ | △ | ✓ (MIM) | △ 중간 |
| InternImage-G | 2022 | 65.8 | 3.0B | ✓ | ✗ | ✓ (MIM) | △ 중간 |
| Co-DETR | 2022 | 64.5 | ~1.0B | ✓ | ✗ | ✓ | ○ |
| **Focal-Stable-DINO** | **2023** | **64.8** | **689M** | **✗** | **✗** | **✗** | **✅ 매우 높음** |

---

## 6. 결론

Focal-Stable-DINO는 **"단순함과 재현성"**이라는 가치를 중심으로, 공개 데이터와 공개 모델만으로 경쟁력 있는 객체 탐지 성능을 달성할 수 있음을 증명한 의미 있는 연구이다. 특히:

1. **재현성**: 모든 자원(코드, 데이터, 사전학습 가중치)이 공개되어 누구나 결과를 재현 가능
2. **일반화**: 공개 데이터만으로도 비공개 대규모 데이터를 활용한 모델에 필적하는 성능 달성
3. **실용성**: 복잡한 학습 파이프라인 없이 조합 가능한 모듈형 접근법 제시

향후 연구는 (1) 어노테이션 품질 개선, (2) 소형 객체 탐지 강화, (3) 모델 효율화, (4) open-vocabulary 확장, (5) 다양한 도메인으로의 전이 등의 방향으로 발전할 수 있을 것이다.

---

## 참고자료

1. **Ren, T., Yang, J., Liu, S., Zeng, A., et al.** "A Strong and Reproducible Object Detector with Only Public Datasets." *arXiv preprint arXiv:2304.13027*, 2023.
2. **Yang, J., Li, C., Dai, X., & Gao, J.** "Focal Modulation Networks." *Advances in Neural Information Processing Systems (NeurIPS)*, 35:4203–4217, 2022.
3. **Liu, S., Ren, T., Chen, J., et al.** "Detection Transformer with Stable Matching." *arXiv preprint arXiv:2304.04742*, 2023.
4. **Zhang, H., Li, F., Liu, S., et al.** "DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection." *arXiv preprint arXiv:2203.03605*, 2022.
5. **Fang, Y., et al.** "EVA: Exploring the Limits of Masked Visual Representation Learning at Scale." *arXiv preprint arXiv:2211.07636*, 2022.
6. **Fang, Y., et al.** "EVA-02: A Visual Representation for Neon Genesis." *arXiv preprint arXiv:2303.11331*, 2023.
7. **Wang, W., et al.** "InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions." *arXiv preprint arXiv:2211.05778*, 2022.
8. **Shao, S., et al.** "Objects365: A Large-scale, High-quality Dataset for Object Detection." *ICCV*, 2019.
9. **Li, F., et al.** "DN-DETR: Accelerate DETR Training by Introducing Query DeNoising." *CVPR*, 2022.
10. **Zong, Z., Song, G., & Liu, Y.** "DETRs with Collaborative Hybrid Assignments Training." *arXiv preprint arXiv:2211.12860*, 2022.
11. **Chen, Q., et al.** "Group DETR v2: Strong Object Detector with Encoder-Decoder Pretraining." *arXiv preprint arXiv:2211.03594*, 2022.
12. **Wang, W., et al.** "Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks." *arXiv preprint arXiv:2208.10442*, 2022.
13. GitHub: https://github.com/microsoft/FocalNet
14. GitHub: https://github.com/IDEA-Research/Stable-DINO

> **주의**: Stable-DINO의 position-supervised loss에 대한 수식은 원 논문 [16] (arXiv:2304.04742)의 내용을 기반으로 재구성한 것이며, 본 Technical Report에는 수식이 직접 포함되어 있지 않습니다. 정확한 수식은 Stable-DINO 원 논문을 참조하시기 바랍니다.
