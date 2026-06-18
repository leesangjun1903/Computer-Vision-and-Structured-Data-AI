
# ADELE: Adaptive Early-Learning Correction for Segmentation from Noisy Annotations

> **논문 기본 정보**
> - **저자**: Sheng Liu*, Kangning Liu*, Weicheng Zhu, Yiqiu Shen, Carlos Fernandez-Granda
> - **소속**: NYU Center for Data Science / NYU Courant Institute of Mathematical Sciences
> - **발표**: CVPR 2022 (Oral Presentation), pp. 2606–2616
> - **arXiv**: 2110.03740

---

## 1. 핵심 주장과 주요 기여 요약

### 🔑 핵심 주장

딥러닝에서 노이즈가 포함된 어노테이션 문제는 분류(classification) 분야에서는 광범위하게 연구되어 왔으나, 세그멘테이션(segmentation) 분야에서는 훨씬 덜 연구되었다. 이 논문은 부정확하게 어노테이션된 데이터로 학습된 딥 세그멘테이션 네트워크의 학습 동역학을 연구한다.

핵심 발견: 네트워크는 "Early-Learning" 단계에서 먼저 클린(clean)한 픽셀 레벨 레이블에 맞추려 하다가, 결국 잘못된 어노테이션을 암기(memorize)하게 된다. 그러나 분류와 달리, 세그멘테이션에서의 암기 현상은 모든 의미론적 카테고리에서 동시에 발생하지 않는다.

### 🏆 주요 기여

| 기여 항목 | 내용 |
|---|---|
| 현상 발견 | 세그멘테이션에서 카테고리별 암기 시점이 다름을 최초 발견 |
| 적응적 레이블 보정 | 카테고리별로 암기 시작 시점을 탐지하여 레이블 수정 |
| 스케일 일관성 정규화 | 다양한 스케일에서의 예측 일관성을 강화하는 정규화 항 추가 |
| 성능 달성 | PASCAL VOC 2012에서 SOTA 달성 및 의료 영상 세그멘테이션에서 우수 성능 |

---

## 2. 상세 설명

### 2-1. 🔴 해결하고자 하는 문제

세그멘테이션 네트워크는 early-learning 단계에서 클린한 픽셀 레벨 레이블을 먼저 학습하고, 이후 잘못된 어노테이션을 암기한다. 그런데 분류와 달리 세그멘테이션에서 암기 현상은 모든 의미론적 카테고리에서 동시에 발생하지 않는다.

이는 다음과 같은 두 가지 핵심 문제를 발생시킨다:

1. **카테고리별 암기 시점 불일치**: 분류 문제에서의 단일 Early-Learning Correction 적용 방식을 그대로 세그멘테이션에 적용할 수 없음
2. **픽셀 수준 노이즈의 복잡성**: 의료 영상 어노테이션 오류, 약지도(weakly-supervised) 어노테이션 등 다양한 형태의 픽셀 레이블 노이즈

---

### 2-2. 🟡 제안하는 방법

저자들은 두 가지 핵심 요소를 가진 새로운 방법을 제안한다: (1) 학습 중 카테고리별로 암기 단계의 시작을 탐지하여 어노테이션을 적응적으로 수정하고, (2) 어노테이션 노이즈에 대한 강건성을 높이기 위해 스케일 간 일관성을 강화하는 정규화 항을 포함한다.

#### ① 적응적 레이블 보정 (Adaptive Label Correction)

레이블 보정에는 foreground/background 신뢰도 임계값 $\tau_{fg}$, $\tau_{bg}$를 사용하며, 특정 카테고리가 보정될 시점을 제어하기 위한 곡선 피팅(curve fitting) 임계값 $r$을 사용한다.

**카테고리별 암기 탐지** 과정은 각 카테고리 $c$에 대한 학습 손실 또는 예측 정확도 곡선의 변곡점을 추적함으로써 이루어진다:

$$
t^*_c = \arg\min_{t} \; \Delta \mathcal{A}_c(t), \quad \Delta \mathcal{A}_c(t) = \mathcal{A}_c(t) - \mathcal{A}_c(t-1)
$$

여기서 $\mathcal{A}_c(t)$는 epoch $t$에서 카테고리 $c$에 대한 모델의 픽셀 수준 예측 정확도(또는 이와 유사한 지표)를 나타낸다. 암기 시작 시점 $t^*_c$ 이후, 카테고리 $c$의 픽셀 레이블을 모델 예측값으로 교체(레이블 보정)한다:

$$
\hat{y}_{i,c} = \begin{cases} f_\theta(x_i)_c & \text{if } \max_c f_\theta(x_i)_c \geq \tau \text{ and } t > t^*_c \\ \tilde{y}_{i,c} & \text{otherwise} \end{cases}
$$

여기서:
- $f_\theta(x_i)_c$: 모델 예측값(softmax 확률)
- $\tilde{y}_{i,c}$: 원래 노이즈 어노테이션
- $\tau$: 신뢰도 임계값 (foreground/background에 따라 $\tau_{fg}$, $\tau_{bg}$ 적용)

#### ② 스케일 일관성 정규화 (Scale Consistency Regularization)

저자들은 어노테이션 노이즈에 대한 강건성을 높이기 위해 스케일 간 일관성을 강화하는 정규화 항을 추가한다.

이 정규화 항은 Jensen-Shannon Divergence (JSD)를 사용한다:

$$
\mathcal{L}_{\text{cons}} = \lambda \cdot \text{JSD}\left(f_\theta(x^{(s_1)}), f_\theta(x^{(s_2)})\right)
$$

$$
\text{JSD}(p \| q) = \frac{1}{2} D_{\text{KL}}(p \| m) + \frac{1}{2} D_{\text{KL}}(q \| m), \quad m = \frac{p+q}{2}
$$

여기서:
- $x^{(s_1)}, x^{(s_2)}$: 서로 다른 스케일로 리사이즈된 동일 입력 이미지
- $\lambda$: 일관성 강도 하이퍼파라미터 (`jsd-lambda`, 기본값 1)
- 신뢰도 임계값 $\rho$ 이상인 예측에만 일관성 정규화를 적용

일관성 강도 $\lambda$ 는 `jsd-lambda`로 제어되며, 0으로 설정 시 일관성 정규화가 적용되지 않는다.

**전체 학습 손실 함수**:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(\hat{y}, f_\theta(x)) + \lambda \cdot \mathcal{L}_{\text{cons}}
$$

$$
\mathcal{L}_{\text{CE}} = -\sum_{i} \sum_{c} \hat{y}_{i,c} \log f_\theta(x_i)_c
$$

---

### 2-3. 🔵 모델 구조

ADELE는 PASCAL VOC 실험에서 ImageNet 사전학습된 ResNet38 모델을 백본(backbone)으로 사용한다.

```
[입력 이미지 x]
       │
       ▼
┌─────────────────┐
│  ResNet38       │  ← ImageNet Pretrained Backbone
│  (Backbone)     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Segmentation Head              │
│  (픽셀별 카테고리 확률 출력)       │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  ADELE Module                               │
│  ① 카테고리별 암기 시점 탐지 (곡선 피팅)       │
│  ② 레이블 보정 (τ_fg, τ_bg 기반)            │
│  ③ 스케일 일관성 정규화 (JSD, λ)            │
└─────────────────────────────────────────────┘
```

> **주의**: ADELE는 특정 백본에 종속되지 않은 플러그인(plugin) 방식의 모듈로, 다양한 세그멘테이션 네트워크에 결합 가능하다. 의료 영상 실험(SegTHOR 데이터셋)에서는 별도의 3D 세그멘테이션 모델이 사용된다.

---

### 2-4. 🟢 성능 향상

본 방법은 인간 어노테이션 오류를 모사한 노이즈가 합성된 의료 영상 세그멘테이션 과제에서 표준 방법을 능가하며, 약지도 의미론적 세그멘테이션에서 현실적인 노이즈 어노테이션에 강건성을 제공하여 PASCAL VOC 2012에서 최첨단(SOTA) 결과를 달성한다.

| 데이터셋 | 방법 | mIoU |
|---|---|---|
| PASCAL VOC 2012 | Baseline (SEAM) | ~55% (약) |
| PASCAL VOC 2012 | SEAM + ADELE | SOTA 달성 |
| SegTHOR | Baseline | - |
| SegTHOR | + ADELE | 표준 방법 대비 우수 |

기준 방법 SEAM과 제안된 ADELE를 결합한 결과의 시각화에서, ADELE는 세그멘테이션 품질을 개선한다.

### 2-5. 🔴 한계

검색 결과 및 논문의 공개된 정보를 바탕으로 확인된 한계:

1. **하이퍼파라미터 민감성**: 레이블 보정 신뢰도 임계값($\tau_{fg}$, $\tau_{bg}$)과 곡선 피팅 임계값 $r$ 등 여러 하이퍼파라미터 설정이 필요하며, 논문에서 이 값들을 실험 전체에서 단순화를 위해 동일하게 설정한다.

2. **암기 탐지의 정확성**: 카테고리별 암기 시작 시점 탐지가 곡선 피팅에 기반하므로, 데이터셋 특성에 따라 탐지 정확도가 달라질 수 있다.

3. **계산 비용**: 학습에 NVIDIA Quadro RTX 8000 GPU 2장이 필요하며, 입력 해상도에 따라 메모리 문제가 발생할 수 있다.

4. **도메인 일반화 검증 부족**: 실험이 의료 영상(SegTHOR)과 자연 이미지(PASCAL VOC 2012)에 한정되어 있어, 다른 도메인(예: 위성 이미지, 산업 검사 등)으로의 일반화 여부가 불확실하다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 플러그인 방식의 모듈성

ADELE의 가장 중요한 일반화 가능성은 **플러그인(plug-in) 모듈** 방식에 있다. SEAM, DeepLab 등 다양한 기존 세그멘테이션 백본에 결합할 수 있으며, 이는 다양한 아키텍처로의 확장성을 의미한다.

### 3-2. 스케일 일관성 정규화의 기여

스케일 일관성 정규화 ($\mathcal{L}_{\text{cons}}$)는 모델이 입력 이미지의 스케일 변화에 강건하게 만든다:

$$
\mathcal{L}_{\text{cons}} = \lambda \cdot \text{JSD}\left(f_\theta(x^{(s_1)}), f_\theta(x^{(s_2)})\right)
$$

이 정규화는 데이터 증강의 일종으로 볼 수 있으며, **다음과 같은 일반화 효과**를 기대할 수 있다:
- 다양한 해상도/크기의 입력에 대한 강건성 향상
- Overfitting 방지를 통한 테스트 도메인 성능 유지
- 약지도 학습 환경에서 부정확한 레이블에 의한 과적합 억제

### 3-3. 카테고리별 적응적 보정의 일반화 이점

분류의 노이즈 레이블 문제와 달리, 모든 의미론적 카테고리가 동일한 학습 동역학을 공유하지 않기 때문에, ADELE는 카테고리별로 노이즈 레이블을 분리하여 보정한다.

이 방식은:
- **클래스 불균형(class imbalance)** 문제가 있는 데이터셋에서 특히 유리
- 희귀 카테고리의 early-learning 단계를 독립적으로 보호
- 의료 영상처럼 배경 클래스와 전경 클래스 간 픽셀 수 차이가 극심한 경우에 효과적

### 3-4. 약지도 학습(Weakly-Supervised Learning)으로의 확장

ADELE는 노이즈가 합성된 의료 영상 세그멘테이션에서 표준 방법을 능가하며, PASCAL VOC 2012에서 약지도 의미론적 세그멘테이션에 존재하는 현실적인 노이즈 어노테이션에도 강건성을 제공하여 SOTA를 달성한다.

이는 ADELE가 **합성 노이즈와 실제 노이즈 모두에 대해 일반화**됨을 보여준다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려사항

### 4-1. 연구에 미치는 영향

#### 🔶 세그멘테이션 노이즈 학습 연구의 촉발

ADELE는 세그멘테이션에서 모든 카테고리가 동기화된 학습 동역학을 공유하지 않는다는 점을 체계적으로 보여주며, 이후 연구인 Dynamic Loss Decay 방법 등에 영향을 미쳤다. DLD(Dynamic Loss Decay)는 ADELE와 달리, 잘못된 카테고리 어노테이션의 영향을 동적 손실 감쇠를 통해 완화하는 보완적 방향으로 발전하였다.

#### 🔶 약지도 학습과 노이즈 강건 학습의 융합

ADELE는 약지도 학습(weakly-supervised segmentation)과 노이즈 강건 학습(noise-robust learning)의 경계를 허물었다. 이는 아래와 같은 연구 방향에 직접적으로 영향을 준다:

- **Semi-supervised segmentation + noisy labels**: Peixia Li 등의 "Semi-supervised semantic segmentation under label noise via diverse learning groups" (ICCV 2023)는 레이블 노이즈 하의 반지도 의미론적 세그멘테이션을 다룬 후속 연구이다.

- **Instance Segmentation으로의 확장**: DivideMix 등 분류 기반의 노이즈 레이블 방법들이 세그멘테이션으로 확장되는 흐름이 가속화되었다.

#### 🔶 의료 영상 AI에서의 실용적 기여

현실 의료 현장에서 전문가 어노테이션은 비용이 높고 오류가 불가피하다. 의료 및 원격 탐지 영역에서 노이즈 어노테이션은 어노테이션 피로와 전문 지식 요구 때문에 불가피하게 존재하며, 의료 세그멘테이션 방법들은 인간 어노테이션 오류 모델링, 정규화 항 채택 등을 통해 강건성을 향상시킨다.

---

### 4-2. 앞으로 연구 시 고려할 점

#### ① 동적 임계값 자동화
현재 $\tau_{fg}$, $\tau_{bg}$, $r$ 등의 하이퍼파라미터는 수동 설정이 필요하다. **베이지안 최적화(Bayesian Optimization)** 또는 **메타 러닝(Meta-Learning)** 기반의 자동 임계값 탐색 연구가 필요하다:

$$
\tau^* = \arg\max_{\tau} \; \mathbb{E}_{(x, y) \sim \mathcal{D}_{\text{val}}} \left[ \text{mIoU}(f_\theta(x; \tau), y) \right]
$$

#### ② Instance Segmentation 및 Panoptic Segmentation으로 확장
ADELE는 Semantic Segmentation에 집중하고 있다. Instance Segmentation 및 Panoptic Segmentation 분야에서도 카테고리별 암기 시점 탐지 프레임워크 적용 가능성을 연구할 필요가 있다.

#### ③ Transformer 기반 아키텍처와의 결합
현재 ADELE는 CNN 기반 ResNet38을 백본으로 사용한다. **SegFormer, Mask2Former, SAM(Segment Anything Model)** 등 Transformer 기반 모델에서의 학습 동역학 차이와 ADELE 적용 가능성을 탐구해야 한다.

#### ④ Open-vocabulary / Foundation Model 시대의 확장
SAM, CLIP 등 Foundation Model이 약지도 세그멘테이션에 적극 활용되면서, 이들이 생성하는 **pseudo-label의 노이즈 패턴**이 기존 방법과 어떻게 다른지 분석하고, ADELE-류 방법의 적용 범위를 검토해야 한다.

#### ⑤ 노이즈 유형의 다양화
현재 실험에서는 경계 침식/팽창 등 특정 형태의 합성 노이즈만 다룬다. **체계적인 노이즈 유형 분류 및 벤치마크** 구축이 필요하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 연도 | 학회 | 대상 | 핵심 접근 | ADELE와의 차이 |
|---|---|---|---|---|---|
| **DivideMix** | 2020 | ICLR | 분류 | 두 네트워크를 동시에 훈련하며 co-divide, label co-refinement, co-guessing을 통해 강건성을 달성 | 분류 전용, 세그멘테이션 미적용 |
| **ELR (Early-Learning Regularization)** | 2020 | NeurIPS | 분류 | 손실 함수에 early-learning 정규화 항 추가 | ELR은 손실 함수에 early-learning 정규화 항을 추가하고, ADELE는 early-learning 단계에서 고신뢰도 오류 레이블을 수정 |
| **ADELE** | 2022 | CVPR Oral | 세그멘테이션 | 카테고리별 암기 탐지 + JSD 정규화 | 본 논문 |
| **DLG (Diverse Learning Groups)** | 2023 | ICCV | 세그멘테이션 | 레이블 노이즈 하의 반지도 의미론적 세그멘테이션을 다양한 학습 그룹으로 처리 | 반지도 학습 프레임워크와 통합 |
| **DLD (Dynamic Loss Decay)** | 2024 | - | 원격탐지 객체탐지 | ADELE와 달리 동적 손실 감쇠 메커니즘으로 잘못된 카테고리 어노테이션의 영향을 완화 | 손실 함수 관점의 접근 |

---

## 📚 참고 자료 및 출처

1. **arXiv (원문)**: Sheng Liu et al., "Adaptive Early-Learning Correction for Segmentation from Noisy Annotations," arXiv:2110.03740, 2022.
   - https://arxiv.org/abs/2110.03740

2. **CVPR 2022 공식 논문**: Liu et al., CVPR 2022, pp. 2606–2616
   - https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Adaptive_Early-Learning_Correction_for_Segmentation_From_Noisy_Annotations_CVPR_2022_paper.html

3. **IEEE Xplore**: https://ieeexplore.ieee.org/document/9879317/

4. **GitHub 공식 구현 (Kangning Liu)**: https://github.com/Kangningthu/ADELE

5. **GitHub 공식 구현 (Sheng Liu)**: https://github.com/shengliu66/ADELE-1

6. **NSF Public Access Repository**: https://par.nsf.gov/biblio/10331906

7. **DeepAI**: https://deepai.org/publication/adaptive-early-learning-correction-for-segmentation-from-noisy-annotations

8. **관련 연구 - DivideMix (Li et al., ICLR 2020)**: https://arxiv.org/pdf/2002.07394

9. **관련 연구 - Dynamic Loss Decay (arXiv 2405.09024)**: https://arxiv.org/html/2405.09024v1

10. **관련 연구 - Benchmarking Label Noise in Instance Segmentation (arXiv 2406.10891)**: https://arxiv.org/html/2406.10891v2

---

> ⚠️ **정확도 관련 주의사항**: 논문의 세부 수식(암기 탐지 수식, 레이블 보정 수식 등)은 공개된 arXiv 전문 및 GitHub 구현 코드를 바탕으로 재구성하였습니다. 일부 수식 표현은 원문에서 직접 인용된 것이 아니라, 공개 코드와 방법론 설명을 기반으로 표준적인 표기로 정리한 것임을 밝힙니다. 정확한 원문 수식은 반드시 [CVPR 2022 논문 PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Adaptive_Early-Learning_Correction_for_Segmentation_From_Noisy_Annotations_CVPR_2022_paper.pdf)를 직접 참조하시기 바랍니다.
