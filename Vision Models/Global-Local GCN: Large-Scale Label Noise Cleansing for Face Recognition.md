
# Global-Local GCN: Large-Scale Label Noise Cleansing for Face Recognition 

> **논문 정보**
> - **저자**: Yaobin Zhang, Weihong Deng, Mei Wang, Jiani Hu, Xian Li, Dongyue Zhao, Dongchao Wen
> - **발표**: CVPR 2020 (pp. 7731–7740)
> - **IEEE Xplore**: DOI 10.1109/CVPR42600.2020.00776

---

## 1. 핵심 주장 및 주요 기여 요약

얼굴 인식 분야에서 대규모 웹 수집 데이터셋은 판별적 표현 학습에 필수적이지만, **아웃라이어(outlier)**와 **레이블 플립(label flip)**과 같은 노이즈 레이블 문제를 안고 있다. 이러한 레이블 노이즈를 자동으로 제거하는 것이 인식 정확도 향상에 중요하지만, 기존 클렌징 방법들은 실제 환경에서 노이즈를 정확하게 식별하지 못한다.

이를 해결하기 위해, 저자들은 얼굴 인식 데이터셋을 위한 효과적인 자동 레이블 노이즈 클렌징 프레임워크인 **FaceGraph**를 제안한다. FaceGraph는 두 개의 연쇄된 그래프 컨볼루션 네트워크(GCN)를 사용하여 **글로벌→로컬 판별(global-to-local discrimination)** 방식으로 노이즈 환경에서 유용한 데이터를 선별한다. CASIA-WebFace, VGGFace2, MegaFace2, MS-Celeb-1M 등 광범위한 데이터셋에 대한 실험에서 ArcFace 등 최신 표현 학습 방법의 인식 성능이 향상됨을 확인하였다.

**주요 기여 4가지**:

1. 두 개의 연쇄 GCN을 활용한 글로벌→로컬 판별 기반 데이터 선별 프레임워크 FaceGraph 제안
2. 대규모 자체 수집 유명인 데이터셋인 **MillionCelebs**(1,880만 장, 636K 신원)를 클렌징하여 공개하였고, 이를 기반으로 ArcFace가 IJB-C 벤치마크에서 **1e-5 FPR 기준 95.62% TPR**을 달성하여 당시 최고 성능을 기록
3. 레이블 노이즈의 두 가지 유형(아웃라이어, 레이블 플립)을 모두 처리하는 통합 프레임워크 설계
4. 기존 데이터셋(CASIA-WebFace, MS-Celeb-1M 등)에도 적용 가능한 **플러그인 방식의 클렌징 파이프라인** 제시

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

얼굴 인식에서 웹 수집 데이터셋은 판별적 표현 학습에 필수적이지만, 아웃라이어와 레이블 플립이라는 노이즈 레이블 문제를 갖고 있으며, 이를 자동으로 클렌징하는 것이 인식 정확도 향상에 유익함에도 불구하고 기존 방법들은 실환경에서 정확한 노이즈 식별에 실패한다.

노이즈의 두 가지 유형:

- **아웃라이어(Outlier)**: 해당 클래스(신원)와 관계없는 무관한 이미지 (예: 배경, 사물 이미지)
- **레이블 플립(Label Flip)**: 다른 사람의 신원으로 잘못 부여된 레이블

---

### 2-2. 제안 방법: FaceGraph 프레임워크

#### (1) 전체 파이프라인

FaceGraph는 크게 다음 순서로 작동합니다:

1. **특징 추출 (Feature Extraction)**: 사전 학습된 얼굴 인식 모델(예: ArcFace)로 각 이미지의 임베딩 벡터 추출
2. **그래프 구성 (Graph Construction)**: 코사인 유사도 기반 $k$-최근접 이웃($k$-NN) 그래프 생성
3. **Global GCN**: 전체 그래프에서 아웃라이어 제거
4. **Local GCN**: 클래스 내부 그래프에서 레이블 플립 제거

#### (2) 그래프 구성

각 이미지 $i$를 노드 $v_i$로 표현하고, 특징 벡터 $\mathbf{f}_i$에 대해 코사인 유사도로 엣지를 구성합니다.

$$s_{ij} = \frac{\mathbf{f}_i \cdot \mathbf{f}_j}{\|\mathbf{f}_i\| \|\mathbf{f}_j\|}$$

$k$-NN 기반 인접 행렬 $\mathbf{A}$:

$$A_{ij} = \begin{cases} s_{ij} & \text{if } j \in \mathcal{N}_k(i) \\ 0 & \text{otherwise} \end{cases}$$

정규화된 인접 행렬:

$$\hat{\mathbf{A}} = \tilde{\mathbf{D}}^{-\frac{1}{2}} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-\frac{1}{2}}, \quad \tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$$

여기서 $\tilde{\mathbf{D}}$는 $\tilde{\mathbf{A}}$의 대각 차수 행렬(degree matrix)입니다.

#### (3) GCN 레이어 전파 수식

표준 GCN 레이어 업데이트:

$$\mathbf{H}^{(l+1)} = \sigma\!\left(\hat{\mathbf{A}} \mathbf{H}^{(l)} \mathbf{W}^{(l)}\right)$$

- $\mathbf{H}^{(l)}$: $l$번째 레이어의 노드 특징 행렬
- $\mathbf{W}^{(l)}$: 학습 가능한 가중치 행렬
- $\sigma$: 활성화 함수 (예: ReLU)

#### (4) Global GCN — 아웃라이어 탐지

전체 데이터셋을 하나의 그래프로 구성하여 각 노드(이미지)에 대해 **이진 분류**를 수행합니다:

$$p_i^{(\text{global})} = \text{Sigmoid}\!\left(\mathbf{h}_i^{(\text{global})} \mathbf{w}_{\text{cls}}\right)$$

이진 교차 엔트로피 손실:

$$\mathcal{L}_{\text{global}} = -\sum_{i \in \mathcal{S}_{\text{labeled}}} \left[ y_i \log p_i^{(\text{global})} + (1 - y_i) \log (1 - p_i^{(\text{global})}) \right]$$

$y_i = 1$: clean 샘플, $y_i = 0$: noisy 샘플.

> ⚠️ **주의**: 위 수식은 논문에서 사용하는 GCN 기반 이진 분류의 일반적 형태를 기반으로 기술한 것이며, 논문 원문의 정확한 수식 표기와 일부 차이가 있을 수 있습니다.

#### (5) Local GCN — 레이블 플립 탐지

Global GCN 이후 남은 샘플들을 **클래스별 서브그래프**로 재구성합니다. 클래스 $c$에 속하는 이미지들만으로 로컬 그래프 $\mathcal{G}_c$를 구성한 후, 동일 클래스 내에서 실제로 다른 신원인 레이블 플립 샘플을 탐지합니다:

$$p_i^{(\text{local})} = \text{Sigmoid}\!\left(\mathbf{h}_i^{(\text{local})} \mathbf{w}'_{\text{cls}}\right)$$

$$\mathcal{L}_{\text{local}} = -\sum_{c} \sum_{i \in \mathcal{G}_c} \left[ y_i \log p_i^{(\text{local})} + (1 - y_i) \log (1 - p_i^{(\text{local})}) \right]$$

#### (6) 최종 데이터 선택

$$\mathcal{D}_{\text{clean}} = \{x_i \mid p_i^{(\text{global})} > \tau_g \;\wedge\; p_i^{(\text{local})} > \tau_l\}$$

여기서 $\tau_g$, $\tau_l$은 각각 글로벌, 로컬 임계값입니다.

---

### 2-3. 모델 구조 (Architecture)

FaceGraph는 글로벌-로컬 GCN을 이진 분류기로 배치하여 $k$-NN 그래프 위에서 신호(clean)와 노이즈를 분류한다.

```
[Raw Dataset]
     ↓
[Feature Extractor (Pre-trained ArcFace / ResNet)]
     ↓
[k-NN Graph Construction (cosine similarity)]
     ↓
[Global GCN] ──→ Outlier 제거 → [Filtered Dataset]
     ↓
[Class-wise Local Graph Construction]
     ↓
[Local GCN]  ──→ Label Flip 제거 → [Clean Dataset]
     ↓
[Train Face Recognition Model (e.g., ArcFace)]
```

두 GCN은 별도의 소규모 **라벨이 있는 참조 데이터**(simulation set)로 학습됩니다.

---

### 2-4. 성능 향상 및 주요 결과

저자들은 자체 수집한 MillionCelebs 데이터(18.8M 이미지, 636K 신원)를 클렌징하여 공개하였으며, 이 클린 데이터로 ArcFace를 학습한 결과 IJB-C 벤치마크에서 **1e-5 FPR 기준 95.62% TPR**로 당시 최고 성능을 상회하는 결과를 달성하였다.

CASIA-WebFace, VGGFace2, MegaFace2, MS-Celeb-1M 등 광범위하게 사용되는 데이터셋을 클렌징하면 ArcFace 등 최신 표현 학습 방법의 인식 성능을 향상시킬 수 있음을 실험으로 검증하였다.

---

### 2-5. 한계점

FaceGraph와 같은 GCN 기반 클렌징 방법의 핵심 문제는 클렌징 대상 데이터가 일반적으로 레이블이 없기 때문에, 클렌징 모델을 **별도의 레이블 데이터(소스 도메인)**로 학습해야 한다는 점이다. 소스 도메인과 비레이블 타겟 데이터 간의 **도메인 갭**으로 인해, 학습된 모델이 타겟 데이터의 분포에 적응하는 데 어려움이 있다.

추가적인 한계:
- **계산 비용**: 수백만 규모의 그래프를 구성하고 GCN 추론을 수행하는 데 상당한 메모리와 연산이 필요
- **참조 데이터 의존성**: 소수의 clean 레이블 데이터가 반드시 필요
- **클래스 불균형**: 클래스 수가 매우 많고 클래스 간 이미지 수 불균형이 클 경우 Local GCN 성능 저하 가능

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 일반화 향상에 기여하는 요소

**① 데이터 품질 향상을 통한 일반화**
노이즈 레이블로 학습된 모델은 노이즈 패턴을 암기(memorization)하는 경향이 있어 일반화 성능이 저하됩니다. FaceGraph로 클렌징된 데이터로 학습 시, 모델이 진정한 신원 특징(identity-discriminative feature)에 집중할 수 있어 일반화 성능이 향상됩니다.

**② MillionCelebs의 다양성**
MillionCelebs는 Freebase 유명인 이름 목록을 기반으로 인터넷 이미지 검색 엔진에서 수집되었으며, 전처리 후 100만 신원의 8,700만 장 이미지가 구성되었다. 배우에 편중된 기존 데이터셋과 달리 다양한 직업군의 신원을 포함한다. 이러한 다양성은 모델의 분포 외 일반화(out-of-distribution generalization)를 강화합니다.

**③ 글로벌-로컬 이중 클렌징의 의의**
- **Global GCN**: 클래스에 무관한 아웃라이어 제거 → 전체 특징 공간 정화
- **Local GCN**: 클래스 내 레이블 플립 제거 → 클래스 내 경계 명확화

이중 단계 클렌징은 학습 데이터의 결정 경계를 정확하게 하여, 보지 못한 데이터(unseen data)에 대한 일반화 능력을 높입니다.

**④ 플러그인 특성**
FaceGraph는 특정 인식 모델에 종속되지 않는 **모델-어그노스틱(model-agnostic)** 전처리 파이프라인으로, ArcFace 외 다른 손실 함수(CosFace, SphereFace 등)와 결합 시에도 일반화 이점을 제공할 수 있습니다.

### 3-2. 일반화를 제약하는 요소

클렌징 대상 데이터가 레이블이 없어 별도 레이블 데이터로 학습한 모델을 전이해야 하며, 소스-타겟 도메인 갭으로 인해 타겟 데이터 분포에 적응하기 어렵다는 문제가 있다.

이를 극복하기 위한 방향:
- 도메인 어댑테이션 기법의 통합
- 메타-러닝 기반의 적응적 임계값 조정 (후속 연구에서 제시됨)

---

## 4. 후속 연구에 미치는 영향 및 고려 사항

### 4-1. 후속 연구에 미치는 영향

**① 메타-러닝 기반 클렌징으로의 발전**

ICCV 2021에서 발표된 AMC(Adaptive Meta-Supervision Cleaning)는 FaceGraph의 도메인 갭 문제를 해결하기 위해 **메타-러닝 기반 적응형 레이블 노이즈 클렌징 알고리즘**을 제안하였다. 이 방법은 레이블된 노이즈 데이터에서 신뢰할 수 있는 클렌징 지식을 먼저 학습한 후, 메타-수퍼비전으로 타겟 데이터에 점진적으로 전이하며, 전이 학습의 드리프트 문제를 해결하는 임계값 어댑터 모듈도 제안하였다.

AMC는 IJB-C 벤치마크에서 클렌징된 WebFace 기준으로 FaceGraph보다 0.85% 높은 1e-5 FPR 기준 73.59% TPR을 달성하였다.

**② AACN — 이웃 집계 방식 개선**

AACN(Adaptive Aggregation of Clean Neighbors)은 GCN에 그래프를 입력하기 전 두 단계의 사전 처리를 통해 노드가 GCN 모듈을 통해 더 강건한 특징을 학습할 수 있도록 한다.

**③ MillionCelebs 데이터셋의 광범위한 활용**

MillionCelebs는 풍부한 통계 정보로 얼굴 인식의 인종 편향 및 클래스 불균형 연구에 활용되고 있으며, FaceGraph 논문과 함께 최초 공개된 후 1년에 걸쳐 추가적으로 정제되었다.

**④ 대규모 정제 데이터셋 구축의 표준 제시**

FaceGraph 이후 WebFace260M(4M 신원/2.6억 얼굴)과 같이 더 대규모로 정제된 학습 데이터를 구축하는 연구 방향이 활성화되었다.

**⑤ GCN의 얼굴 클러스터링/노이즈 탐지 적용 확산**

FaceGraph가 GCN을 얼굴 인식 작업의 레이블 노이즈 클렌징에 배치한 이후, 여러 후속 연구가 GCN을 얼굴 이미지 클러스터링 등에도 활용하게 되었다.

---

### 4-2. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 학회 | 핵심 방법 | FaceGraph 대비 특징 |
|---|---|---|---|
| **FaceGraph (본 논문)** | CVPR 2020 | Global-Local GCN 이진 분류 | 최초의 대규모 GCN 기반 얼굴 노이즈 클렌징 |
| **AMC** (Zhang et al.) | ICCV 2021 | 메타-러닝 기반 적응적 클렌징 | 도메인 갭 문제 해결, FaceGraph +0.85% TPR |
| **AACN** | - | 깨끗한 이웃 적응적 집계 + GCN | GCN 입력 전 2단계 전처리로 강건성 향상 |
| **WebFace42M** (Zhu et al.) | CVPR 2021 | 대규모 정제 데이터 구축 | 42M 정제 얼굴로 확장된 데이터 패러다임 |
| **DivideMix** (Li et al.) | ICLR 2020 | 반지도학습 기반 노이즈 학습 | 일반 노이즈 레이블 학습, 얼굴 특화 아님 |

AMC는 메타-러닝 기반으로 클렌징 대상 데이터의 분포를 학습하고 클래스 차이에 기반한 자동 조정을 수행하는 적응적 레이블 노이즈 클렌징 알고리즘으로, FaceGraph의 도메인 갭 문제를 직접적으로 해결한다.

---

### 4-3. 향후 연구 시 고려할 점

1. **도메인 갭 최소화**: 참조 데이터와 클렌징 대상 데이터 간의 분포 차이를 줄이기 위한 도메인 적응 또는 메타-러닝 기법 통합이 필수적입니다.

2. **계산 효율성**: 수천만 이미지 규모의 그래프를 처리할 때의 메모리/시간 복잡도 문제를 해결하기 위한 **미니배치 GCN**, **클러스터링 기반 서브그래프 분해** 등의 효율적 구현이 필요합니다.

3. **자기지도 학습(Self-supervised Learning) 통합**: 레이블 데이터 없이 노이즈를 탐지할 수 있는 방향으로, SSL 기반 특징 표현과 결합한 클렌징 방법론 탐구가 유망합니다.

4. **페어니스(Fairness) 및 편향 제거**: MillionCelebs가 인종 편향 및 클래스 불균형 문제 연구에 활용되고 있듯이, 클렌징 과정에서 특정 인종/성별 그룹이 과다 제거되지 않도록 공정성 제약 조건을 추가하는 연구가 필요합니다.

5. **프라이버시 보호 학습**: 대규모 유명인 데이터 수집 및 활용에 대한 윤리적·법적 문제가 증가하고 있어, 연합학습(Federated Learning) 또는 차등 프라이버시(Differential Privacy)와의 결합이 중요한 연구 방향이 됩니다.

6. **노이즈 유형 확장**: 현재 아웃라이어와 레이블 플립만 다루지만, **다중 레이블 노이즈**, **부분 레이블(partial label)**, **의존 노이즈(instance-dependent noise)** 등 더 복잡한 노이즈 유형으로의 확장이 필요합니다.

---

## 📚 참고 자료 및 출처

1. **Zhang, Y., Deng, W., et al. (2020)** — "Global-Local GCN: Large-Scale Label Noise Cleansing for Face Recognition," *CVPR 2020*, pp. 7731–7740.
   - IEEE Xplore: https://ieeexplore.ieee.org/document/9156958/
   - CVF Open Access: https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Global-Local_GCN_Large-Scale_Label_Noise_Cleansing_for_Face_Recognition_CVPR_2020_paper.html

2. **Zhang, Y., Deng, W., et al. (2021)** — "Adaptive Label Noise Cleaning with Meta-Supervision for Deep Face Recognition," *ICCV 2021*.
   - IEEE Xplore: https://ieeexplore.ieee.org/document/9710478/

3. **Semantic Scholar** — FaceGraph 논문 페이지: https://www.semanticscholar.org/paper/Global-Local-GCN:-Large-Scale-Label-Noise-Cleansing-Zhang-Deng/865cab74e0c9b32698a4972266a5261f7a144b1c

4. **Papers with Code** — FaceGraph: https://paperswithcode.com/paper/global-local-gcn-large-scale-label-noise

5. **MillionCelebs Dataset 공식 페이지**: https://buptzyb.github.io/MillionCelebs/

6. **ResearchGate** — FaceGraph 논문 페이지: https://www.researchgate.net/publication/343454442

7. **PaperTalk** — FaceGraph 발표 영상: https://papertalk.org/papertalks/15385

8. **GitHub - gorkemalgan/deep_learning_with_noisy_labels_literature**: https://github.com/gorkemalgan/deep_learning_with_noisy_labels_literature

> ⚠️ **정확도 관련 고지**: 본 답변의 **세부 수식**(GCN 레이어 구성, 손실 함수 표기 등)은 검색을 통해 확인된 FaceGraph의 구조적 설명(Global GCN + Local GCN + k-NN 그래프, 이진 분류기)과 일반적인 GCN 수식 관례를 조합하여 재구성한 것으로, **논문 원문의 정확한 수식과 일부 차이가 있을 수 있습니다**. 가장 정확한 수식 확인을 위해서는 CVF Open Access PDF 원문을 직접 참조하시기 바랍니다.
