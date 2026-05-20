
# Quantile Rendering: Efficiently Embedding High-dimensional Feature on 3D Gaussian Splatting

> **📌 논문 정보**
> - **제목:** Quantile Rendering: Efficiently Embedding High-dimensional Feature on 3D Gaussian Splatting
> - **arXiv ID:** 2512.20927 (2025년 12월 24일)
> - **저자:** Yoonwoo Jeong 외 4인

---

## 1. 핵심 주장 및 주요 기여 요약

최근 컴퓨터 비전 분야에서 3D Gaussian Splatting(3D-GS)을 활용해 Open-Vocabulary Segmentation(OVS)을 3D 도메인으로 확장하는 연구가 활발히 진행되고 있다. 그러나 open-vocabulary 쿼리에 필요한 고차원 피처를 효율적으로 렌더링하는 것은 여전히 중요한 과제이다.

기존 방법들은 코드북(codebook) 또는 피처 압축(feature compression)을 사용하여 정보 손실을 초래하고, 이로 인해 세그멘테이션 품질이 저하되는 문제가 있다.

이 논문의 핵심 주장과 기여는 다음 두 가지로 요약된다:

**① Q-Render (Quantile Rendering):**
위 한계를 해결하기 위해 **Quantile Rendering(Q-Render)**을 제안하며, 이는 3D Gaussian에서 고차원 피처를 고충실도로 효율적으로 처리하는 새로운 렌더링 전략이다. 기존의 볼륨 렌더링이 각 광선과 교차하는 모든 3D Gaussian을 밀집 샘플링하는 것과 달리, Q-Render는 광선을 따라 지배적인 영향을 미치는 Gaussian만을 희소(sparse) 샘플링한다.

**② GS-Net (Gaussian Splatting Network):**
Q-Render를 일반화 가능한 3D 신경망에 통합하여, Gaussian 피처를 일반화된 방식으로 예측하는 **Gaussian Splatting Network(GS-Net)**을 제안한다.

저자들은 **(1) ScanNet**과 **(2) LeRF-OVS** 두 가지 open-vocabulary 3D 시맨틱 세그멘테이션 벤치마크에서 방법을 검증하였으며, Q-Render는 두 경우 모두 최첨단 성능을 달성하여, 2D 기반 모델과 3D Gaussian 표현 간의 확장 가능한 다리로서의 가치를 입증했다.

---

## 2. 해결하고자 하는 문제, 제안 방법(수식 포함), 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

컴퓨터 비전의 최신 발전은 3D-GS를 활용하여 OVS를 3D 도메인으로 성공적으로 확장했다. 그러나 open-vocabulary 쿼리에 필요한 고차원 피처를 효율적으로 렌더링하는 것은 여전히 중요한 과제이다. 기존 방법들은 코드북 또는 피처 압축을 사용하여 정보 손실을 유발하고, 세그멘테이션 품질을 저하시킨다.

즉, 기존 방법의 문제점은 다음과 같이 정리된다:

| 문제 유형 | 설명 |
|---|---|
| **정보 손실** | 코드북/피처 압축으로 인한 품질 저하 |
| **계산 비용** | 모든 Gaussian을 밀집 샘플링하는 비효율성 |
| **일반화 부재** | 장면별 최적화(scene-specific) 모델의 한계 |

---

### 2-2. 제안 방법 및 수식

#### ■ 기존 볼륨 렌더링 (Volume Rendering in 3D-GS)

기존 3D-GS에서의 피처 렌더링은 광선 $r$ 과 교차하는 정렬된 Gaussian 집합 $\{g_i\}_{i=1}^{N}$ 에 대해 다음과 같이 정의된다:

$$\tilde{f}_{VR} = \sum_{i=1}^{N} \alpha_i \cdot T_i \cdot f_i$$

$$T_i = \prod_{j=1}^{i-1}(1 - \alpha_j)$$

여기서:
- $\alpha_i$: $i$번째 Gaussian의 불투명도(opacity)
- $T_i$: $i$번째 Gaussian까지의 누적 투과율(transmittance)
- $f_i$: $i$번째 Gaussian의 피처 벡터

이 방식은 $N$ 개의 모든 Gaussian을 계산해야 하므로 고차원 피처에서는 계산 비용이 매우 크다.

#### ■ Q-Render: Quantile Rendering

모든 Gaussian이 영향력이 있는 것은 아니며, 일부 Gaussian만이 광선을 따라 고차원 피처 렌더링에 의미 있는 영향을 미친다는 관찰에서 출발한다. 이 관찰이 투과율 인식(transmittance-aware) 효율적 렌더링 알고리즘인 Q-Render를 동기부여한다. 모든 래스터화된 3D Gaussian을 밀집 누적하는 대신, Q-Render는 광선의 투과율 분포를 지배하는 소수의 **분위수 Gaussian(quantile Gaussians)**을 적응적으로 선택하여 이들만 렌더링한다.

Q-Render의 최종 공식은 검색된 결과에서 다음과 같이 확인된다:

$$\tilde{f}_Q = \text{QuantileRender}(\mathcal{G}, \mathcal{F}, K, I)$$

Q-Render는 원본 3D-GS의 투과율 분포를 효과적으로 근사한다.

여기서:
- $\mathcal{G}$: 3D Gaussian 집합
- $\mathcal{F}$: 피처 집합
- $K$: 선택할 분위수 Gaussian의 수 (하이퍼파라미터)
- $I$: 래스터화된 3D Gaussian의 인덱스

이 분위수 기반 선택은 중복 계산을 줄이고, 고차원 피처 맵 렌더링이 필요한 다운스트림 작업을 위해 원본 신호를 근사한다.

**근사 오차 경계:**

희소 분위수 선택은 계산 오버헤드를 크게 줄이면서 고충실도 피처 매핑에 충분하며, 근사 오차는 $O(1/K)$로 바운드된다.

즉:

$$\|\tilde{f}_Q - \tilde{f}_{VR}\| = O\left(\frac{1}{K}\right)$$

이는 $K$가 증가할수록 오차가 감소하고, 충분히 큰 $K$에서 원본 볼륨 렌더링을 임의로 잘 근사함을 보장한다.

#### ■ Top-K 방법 대비 비교

Q-Render는 볼륨 렌더링의 투과율 경향을 잘 근사하는 반면, Top- $K$ 샘플링 전략은 다른 경향을 보인다. Q-Render는 오버헤드를 피하면서도 Top- $K$보다 성능이 뛰어나다.

---

### 2-3. 모델 구조: GS-Net

Q-Render를 3D 신경망에 통합하여 3D Gaussian을 입력으로 받아 Gaussian 피처를 예측하는 **GS-Net(Gaussian Splatting Network)**을 구축한다. Q-Render는 2D 지도(supervision)와 3D 신경망 사이의 효율적인 다리 역할을 하며, 이미지 공간 손실에서 3D 신경망의 예측으로 역전파 그래디언트를 흘려보낸다.

전체 프레임워크의 두 핵심 구성요소는 다음과 같다:
- **Quantile Rendering**: 각 광선을 따라 가장 대표적인 Gaussian만을 선택하는 희소, 투과율 가이드 샘플링 전략
- **Gaussian Splatting Network**: 최적화된 3D Gaussian으로부터 고차원 피처 Gaussian을 예측하는 3D 신경망으로, Q-Render가 2D 기반 모델로부터 효율적이고 효과적인 피처 증류를 가능하게 함

GS-Net의 전체 파이프라인을 도식화하면:

```
입력 3D Gaussians (𝒢)
        ↓
  [3D Neural Network]  ← GS-Net 본체 (Sparse Conv 기반)
        ↓
  고차원 Gaussian 피처 (ℱ) 예측
        ↓
  [Q-Render: Quantile Sampling]
        ↓
  2D Feature Map (f̃_Q)
        ↓
  [CLIP 기반 Image-Space Loss]  ← 2D 지도 신호
        ↓
  역전파 → GS-Net 가중치 업데이트
```

---

### 2-4. 성능 향상

ScanNet과 LeRF 벤치마크에서의 광범위한 실험을 통해 제안 프레임워크가 최첨단 방법들을 능가하며, 근사 실시간 렌더링을 가능하게 함을 보였다.

전반적으로 Q-Render는 다른 피처 렌더링 알고리즘들에 비해 우수하거나 비슷한 수준의 성능을 달성한다.

---

### 2-5. 한계점

검색된 결과에서 명시적으로 확인된 한계점은 다음과 같다:

1. **K 파라미터 민감성:** 성능은 $K$가 증가함에 따라 수렴하는 경향이 있다. 적절한 $K$ 값을 선택해야 하는 하이퍼파라미터 튜닝 문제가 남아있다.

2. **CLIP 의존성:** GS-Net이 CLIP 임베딩에 의존하는 구조이므로, CLIP의 사전 학습 목적 및 규모의 한계로 인해 세밀한 카테고리나 복잡한 텍스트 설명을 추론하는 능력이 제한될 수 있으며, 이는 open-vocabulary 인식의 주요 병목이 된다.

3. **3D-GS 의존성:** Q-Render는 3D-GS 기반 표현에 특화되어 있어, NeRF 등 다른 3D 표현 방식에 직접 적용하기 어렵다.

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문에서 일반화(Generalization)는 핵심 목표 중 하나이다.

### 3-1. GS-Net을 통한 장면 간 일반화

Q-Render를 일반화 가능한 3D 신경망에 통합함으로써, **Gaussian 피처를 일반화된 방식으로 예측하는 GS-Net**을 제안한다.

기존 3D-GS는 장면별 최적화를 수행하여 새로운 장면에 대한 일반화가 불가능했다. GS-Net은 이를 극복하기 위한 핵심 설계를 갖추고 있다:

Q-Render는 2D 지도(supervision)와 3D 신경망 사이의 효율적인 다리 역할을 하며, 이미지 공간 손실에서 3D 신경망의 예측으로 역전파 그래디언트를 흘려보낸다.

또한 Q-Render의 희소 샘플링은 3D 신경망의 귀납적 편향(inductive bias)과 시너지를 이룬다: 3D 신경망은 공간적으로 매끄러운(smooth) Gaussian 피처를 예측하는 경향이 있어, 모든 Gaussian을 밀집 샘플링하는 것이 불필요해진다.

### 3-2. 2D 기반 모델과의 연동을 통한 일반화

Q-Render는 2D 기반 모델과 3D Gaussian 표현 간의 확장 가능한 다리(scalable bridge)로서 작동한다.

이는 다음과 같은 일반화 가능성을 시사한다:

- **다양한 2D VLM 연동:** CLIP 외에 DINO, LLaVA 등 다른 2D 비전-언어 모델로부터 피처를 증류하는 데 확장 가능
- **다양한 3D 신경망 아키텍처:** Sparse Convolution(Choy et al., 2019), 트랜스포머 기반 3D 네트워크 등으로 교체 가능한 모듈형 설계
- **새로운 장면 일반화:** 장면별 재최적화 없이 새로운 환경에서 Gaussian 피처 예측 가능

### 3-3. 일반화의 핵심 메커니즘

| 메커니즘 | 역할 |
|---|---|
| **Q-Render의 희소 샘플링** | 계산 효율 확보 → 대규모 데이터 학습 용이 |
| **투과율 기반 분위수 선택** | 장면 독립적인 선택 기준 → 도메인 강건성 |
| **GS-Net의 귀납적 편향** | 공간 매끄러움 → 새로운 장면에서도 안정적 예측 |
| **2D 지도 신호 활용** | 대규모 2D 사전학습 지식 흡수 → 제로샷 일반화 |

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4-1. 향후 연구에 미치는 영향

**① 3D 장면 이해의 새로운 패러다임:**
Q-Render는 3D Gaussian에서 고차원 피처를 고충실도로 처리하는 새로운 렌더링 전략을 제시하며, 기존 밀집 샘플링의 비효율을 해결한다. 이는 향후 3D open-vocabulary 이해, 3D 편집, 3D-LLM 등 다양한 분야에 영향을 미칠 것이다.

**② 효율적 피처 렌더링 연구 촉진:**
$O(1/K)$ 오차 바운드를 이론적으로 제시함으로써, 희소 샘플링 기반 렌더링 이론 연구의 토대를 마련한다.

**③ 2D-3D 지식 증류 연구:**
GS-Net과 Q-Render의 결합은 2D 기반 모델의 지식을 3D Gaussian에 효과적으로 증류하는 새로운 프레임워크를 제시하며, 이 방향의 후속 연구를 촉진할 것이다.

**④ 실시간 3D 세그멘테이션의 실용화:**
근사 실시간 렌더링을 가능하게 하는 성능은 자율주행, 로봇공학, AR/VR 등 실시간 3D 이해가 필요한 응용 분야에서 실용적 배포 가능성을 높인다.

---

### 4-2. 향후 연구 시 고려할 점

**① K 값의 적응적 결정:**
현재 $K$는 고정 하이퍼파라미터로 설정되나, 장면의 복잡도나 Gaussian 밀도에 따라 $K$를 동적으로 결정하는 **적응형 분위수 선택** 방법 연구가 필요하다.

**② 다양한 2D 기반 모델과의 호환성:**
CLIP 외에 SAM(Segment Anything Model), DINO-v2, 최신 LLM 기반 비전 모델 등과의 통합을 통해 일반화 성능을 더욱 향상시킬 수 있다.

**③ 동적 장면(Dynamic Scene)으로의 확장:**
현재 프레임워크는 정적 장면을 가정하고 있어, 움직이는 객체가 포함된 동적 환경에서의 적용 가능성 연구가 필요하다.

**④ 대규모 야외 장면:**
ScanNet, LeRF-OVS 등 실내 중심 벤치마크에서 검증되었으므로, 대규모 야외 장면(outdoor large-scale)에서의 성능 검증이 필요하다.

**⑤ 이론적 근사 오차의 실용적 한계 분석:**
$O(1/K)$ 오차 바운드가 실제 다운스트림 작업(세그멘테이션 정확도 등)에 미치는 영향을 더욱 정밀하게 분석할 필요가 있다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 연도 | 핵심 아이디어 | 한계 | Q-Render와의 차이 |
|---|---|---|---|---|
| **NeRF** (Mildenhall et al.) | 2020 | MLP 기반 암시적 신경 복사장 | 느린 렌더링, 장면별 최적화 | Q-Render는 명시적 Gaussian 기반으로 실시간 가능 |
| **3D-GS** (Kerbl et al.) | 2023 | 명시적 3D Gaussian으로 실시간 렌더링 | 고차원 피처 렌더링 비효율 | Q-Render가 이를 직접 해결 |
| **LERF** (Kerr et al.) | 2023 | NeRF에 CLIP 피처 증류 | NeRF 기반으로 속도 한계 | Q-Render는 3D-GS 기반으로 속도 대폭 개선 |
| **LangSplat** | 2024 | 3D-GS + CLIP, 오토인코더 압축 | 피처 압축으로 정보 손실 | Q-Render는 압축 없이 원본 고차원 피처 직접 처리 |
| **Feature 3DGS** | 2024 | 3D-GS에 임의의 피처 필드 결합 | 모든 Gaussian 밀집 렌더링 | Q-Render의 희소 분위수 샘플링으로 효율 대폭 향상 |
| **Q-Render / GS-Net** (본 논문) | 2025 | 투과율 기반 희소 분위수 샘플링 + 일반화 가능한 3D 신경망 | K 파라미터 민감성, CLIP 의존성 | 정보 손실 없이 고차원 피처 효율 렌더링, 장면 간 일반화 |

---

## 📚 참고 자료 출처

1. **arXiv 논문 원본 (Abstract)**
   - Yoonwoo Jeong et al., *"Quantile Rendering: Efficiently Embedding High-dimensional Feature on 3D Gaussian Splatting"*, arXiv:2512.20927, December 2025.
   - URL: https://arxiv.org/abs/2512.20927

2. **arXiv HTML 전문**
   - URL: https://arxiv.org/html/2512.20927

3. **arXiv PDF**
   - URL: https://arxiv.org/pdf/2512.20927

4. **OpenReview (논문 리뷰 페이지)**
   - URL: https://openreview.net/forum?id=BncqszFjCb

5. **Hugging Face Papers 페이지**
   - URL: https://huggingface.co/papers/2512.20927

---

> ⚠️ **정확도 관련 안내:** 본 답변은 검색을 통해 확인된 내용을 기반으로 작성되었습니다. 수식의 세부 표기(특히 분위수 선택의 정확한 알고리즘 수식)는 검색 결과에서 일부만 확인되어, 공개된 arXiv 정보를 기반으로 서술하였습니다. 더 정확한 수식은 arXiv PDF 원문을 직접 참조하시기 바랍니다.
