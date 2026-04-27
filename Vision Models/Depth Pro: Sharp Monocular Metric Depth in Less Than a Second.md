
# Depth Pro: Sharp Monocular Metric Depth in Less Than a Second

> **논문 정보**
> - 저자: Aleksei Bochkovskii, Amaël Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R. Richter, Vladlen Koltun (Apple)
> - 발표: arXiv:2410.02073 (2024) / ICLR 2025 게재 확정
> - 공식 코드: https://github.com/apple/ml-depth-pro
> - Apple ML Research: https://machinelearning.apple.com/research/depth-pro

---

## 1️⃣ 핵심 주장 및 주요 기여 요약

Depth Pro는 **Zero-Shot 메트릭 단안 깊이 추정(Zero-Shot Metric Monocular Depth Estimation)** 을 위한 파운데이션 모델로, 카메라 내재 파라미터(intrinsics) 등의 메타데이터 없이도 절대 스케일을 갖는 메트릭 깊이 맵을 생성하며, 표준 GPU에서 **2.25 메가픽셀 깊이 맵을 0.3초** 만에 생성합니다.

주요 기술 기여는 다음 네 가지입니다:
1. **효율적인 멀티스케일 ViT 기반 아키텍처** 설계 (글로벌 컨텍스트 + 고해상도 세부 구조 포착)
2. **새로운 경계 정확도 메트릭** 도입 (매팅 데이터셋 기반)
3. **손실 함수 + 훈련 커리큘럼** 설계 (실세계 + 합성 데이터를 조합한 날카로운 경계 추정)
4. **Zero-Shot 초점 거리 추정** 모듈 (단일 이미지로부터 SOTA 성능 달성)

---

## 2️⃣ 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 🔴 해결하고자 하는 문제

기존 단안 깊이 추정 연구의 한계는 크게 세 가지입니다:

**① 스케일 모호성 (Scale Ambiguity)**

일부 기존 방법들은 우수한 일반화 성능을 보여주었지만, 그 예측이 스케일과 시프트에서 모호하여 정확한 형태, 크기, 거리가 필요한 다운스트림 응용에는 사용할 수 없었습니다.

**② 카메라 내재 파라미터 의존성**

가장 광범위한 'in-the-wild' 적용을 위해서는, 이미지에 카메라 내재 파라미터(예: 초점 거리)가 제공되지 않더라도 절대 스케일을 갖는 메트릭 깊이 맵을 생성해야 합니다.

**③ 해상도 및 경계 선명도 부족**

단안 깊이 추정기는 고해상도로 동작하며 머리카락, 털, 기타 미세 구조와 같은 이미지 세부 사항을 정밀하게 추적하는 세밀한 깊이 맵을 생성해야 합니다. 복잡한 세부 사항을 정확하게 추적하는 선명한 깊이 맵 생성의 한 이점은 뷰 합성 같은 응용에서 이미지 품질을 저하시킬 수 있는 "flying pixel"의 제거입니다.

---

### 🟢 제안 방법 (수식 포함)

#### (A) Canonical Inverse Depth (정규화된 역 깊이)

Depth Pro는 깊이 $d$를 직접 예측하지 않고, 카메라의 수평 시야각(Field of View, $\text{FoV}$)으로 스케일된 **역 깊이(Inverse Depth)** 형태로 예측합니다.

훈련 과정은 카메라의 시야각에 의해 스케일된 역 깊이 이미지로부터 메트릭 깊이 맵을 예측하는 것에 집중합니다.

정규화된 역 깊이 $\tilde{d}$는 다음과 같이 정의됩니다:

$$\tilde{d} = \frac{f}{d}$$

여기서 $f$는 초점 거리(focal length), $d$는 실제 깊이(metric depth)입니다.

최종 메트릭 깊이 $d$는 추정된 초점 거리 $\hat{f}$를 이용해 복원합니다:

$$\hat{d} = \frac{\hat{f}}{\tilde{d}}$$

#### (B) 손실 함수 (Loss Functions)

핵심 손실 함수는 깊이 예측을 위한 **MAE**와 경계 세부 사항을 정제하기 위한 **기울기 손실(Gradient Loss)** 및 **라플라시안 손실(Laplace Loss)** 을 포함합니다. 훈련은 특히 합성 데이터에서 선명도를 향상시키기 위해 1차 및 2차 도함수 손실의 조합을 활용합니다.

전체 손실 함수는 다음과 같이 구성됩니다:

$$\mathcal{L} = \mathcal{L}_{\text{MAE}} + \lambda_1 \mathcal{L}_{\text{grad}} + \lambda_2 \mathcal{L}_{\text{lap}}$$

**MAE Loss (Mean Absolute Error):**

$$\mathcal{L}_{\text{MAE}} = \frac{1}{N} \sum_{i=1}^{N} \left| \tilde{d}_i - \hat{\tilde{d}}_i \right|$$

**Gradient Loss (1차 도함수 손실 — 경계 정보 강화):**

$$\mathcal{L}_{\text{grad}} = \frac{1}{N} \sum_{i} \left( \left| \nabla_x \tilde{d}_i - \nabla_x \hat{\tilde{d}}_i \right| + \left| \nabla_y \tilde{d}_i - \nabla_y \hat{\tilde{d}}_i \right| \right)$$

**Laplacian Loss (2차 도함수 손실 — 경계 선명도 강화):**

$$\mathcal{L}_{\text{lap}} = \frac{1}{N} \sum_{i} \left| \nabla^2 \tilde{d}_i - \nabla^2 \hat{\tilde{d}}_i \right|$$

#### (C) 초점 거리 추정 (Focal Length Estimation)

네트워크는 초점 거리 추정 헤드로 보완됩니다. 소형 합성곱 헤드(convolutional head)가 깊이 추정 네트워크의 고정된 특징과 별도의 ViT 이미지 인코더에서 얻은 작업별 특징을 입력으로 받아 수평 시야각(angular FoV)을 예측합니다.

L2 손실 함수를 적용하여 훈련함으로써 모델은 이러한 특징 세트를 기반으로 초점 거리를 정확하게 추정할 수 있습니다.

초점 거리 추정을 위한 L2 손실:

$$\mathcal{L}_{\text{fov}} = \left\| \widehat{\text{FoV}} - \text{FoV}_{\text{gt}} \right\|_2^2$$

추정된 초점 거리 $\hat{f}$는 수평 FoV $\widehat{\theta}$ 및 이미지 너비 $W$로부터 다음과 같이 계산됩니다:

$$\hat{f} = \frac{W}{2 \tan\left(\frac{\widehat{\theta}}{2}\right)}$$

#### (D) 훈련 커리큘럼 (Two-Stage Training Curriculum)

두 단계의 훈련 커리큘럼이 적용됩니다: 첫 번째 단계는 모든 레이블 데이터셋의 혼합을 사용하여 도메인 간 일반화에 집중하고, 두 번째 단계는 합성 데이터셋으로 파인튜닝하여 경계를 선명하게 하고 더 세밀한 세부 정보를 포착합니다.

---

### 🔵 모델 구조 (Architecture)

Depth Pro의 아키텍처는 글로벌 이미지 컨텍스트 포착과 미세 구조 보존의 균형을 맞추도록 설계된 멀티스케일 Vision Transformer(ViT)를 중심으로 합니다. 일반적인 트랜스포머와 달리, Depth Pro는 여러 스케일에서 플레인 ViT 백본을 적용하고 예측을 단일 고해상도 출력으로 융합하며, ViT 사전 훈련의 지속적인 발전으로부터 이점을 얻습니다.

DINOv2 인코더를 공유하는 멀티스케일 ViT 기반 아키텍처를 채택하며, 이미지를 다운샘플링하고 패치로 분할한 후 처리합니다. 추출된 패치 수준 특징은 병합, 업샘플링되고 DPT 방식의 융합 단계를 통해 정제되어 정밀한 깊이 추정을 가능하게 합니다.

구조를 도식화하면 다음과 같습니다:

```
입력 이미지
    │
    ├──► [멀티스케일 패치 분할] ──► [Patch Encoder (DINOv2 ViT 공유)] ──┐
    │                                                                    │
    └──► [전체 이미지 다운샘플 384×384] ──► [Image Encoder (ViT)] ──────┤
                                                                         ↓
                                                            [DPT 융합 디코더]
                                                                         ↓
                                                      [1536×1536 깊이 맵 출력]
                                                                         │
                                          [초점 거리 추정 헤드 (Conv Head)] ◄── frozen features
```

Depth Pro는 ViT 인코더를 사용하여 여러 스케일에서 이미지 패치를 처리하고 예측을 단일 고해상도 깊이 맵으로 융합합니다. 두 개의 ViT 인코더를 사용하는데, 하나는 스케일 불변 학습을 위한 패치 인코더이고 다른 하나는 전역 컨텍스트를 위한 이미지 인코더입니다.

플레인 ViT 인코더를 사용함으로써 Depth Pro는 다양한 사전 훈련된 백본을 활용할 수 있어 모델 성능을 향상시킵니다. 패치 기반 처리는 멀티헤드 자기 주의(self-attention)와 같이 입력 픽셀 수에 따라 이차적으로 증가하는 계산 부하를 줄입니다.

Depth Pro는 504M 파라미터 모델로 6GiB VRAM을 갖춘 노트북에도 쉽게 탑재할 수 있습니다.

---

### 🟡 성능 향상

Depth Pro는 모든 기존 단안 깊이 추정 모델을 능가하여 Zero-Shot 메트릭 깊이 정확도에서 최고 평균 정확도와 경계 리콜(R)에서 최고 F1 점수를 달성합니다. 논문의 Table 1에서 Depth Pro는 평균 순위 2.5(낮을수록 좋음)로 모든 데이터셋에서 통합적으로 우수한 성능을 보입니다.

Depth Pro는 경계 정확도와 지연 시간에서 특히 뛰어나며, Marigold, Depth Anything v2, Metric3D v2와 같은 SOTA 모델들을 크게 능가하는 미세 구조 및 경계 추적 정밀도를 보입니다.

예를 들어, PPR10K 인물 데이터셋에서 Depth Pro의 초점 거리 예측의 64.6%가 상대 오차 25% 이하를 달성했는데, 이는 차선책인 SPEC의 34.6%보다 크게 우수합니다.

**평가 메트릭 (Boundary F1 / Recall):**

$$\text{F1}_{\text{boundary}} = \frac{2 \cdot \text{Precision}_b \cdot \text{Recall}_b}{\text{Precision}_b + \text{Recall}_b}$$

스케일 불변 경계 리콜(SI Boundary Recall):

$$\text{SI-Recall} = \frac{|\{p \in \partial \hat{d} : \exists q \in \partial d_{\text{gt}}, \|p - q\| < \tau\}|}{|\partial d_{\text{gt}}|}$$

여기서 $\tau$는 허용 거리 임계값입니다.

---

### 🔶 한계

단안 깊이 추정 모델들은 일반적으로 거울, 물, 유리와 같은 복잡한 조명 및 굴절 특성을 가진 **투명하고 반사적인 표면**에 어려움을 겪습니다. 모델은 장면을 정확하게 이해하고 물체의 고유한 특성을 추론하여 외부 반사에 의한 오예측 없이 정밀한 깊이 정보를 제공해야 합니다.

야생 동물 모니터링 벤치마크 실험에서 Depth Anything V2가 MAE 0.454m, 상관관계 0.962로 최고 전반적 정확도를 달성한 반면, Metric3D는 상관관계 0.974로 최고 깊이 구조 보존력을 보였습니다. 두 방법 모두 이 야외 야생 환경 설정에서 ML Depth Pro와 ZoeDepth를 크게 능가했습니다. → **특정 야외 환경(극단적인 자연 환경)에서는 한계가 있음을 시사**합니다.

평가 표는 다양한 모델과 도메인에 걸쳐 상당한 성능 변동을 보여주는데, 이는 단안 메트릭 깊이 추정 모델이 강건한 일반화를 달성하는 데 여전히 상당한 도전에 직면해 있음을 시사합니다.

---

## 3️⃣ 모델의 일반화 성능 향상 가능성

### ✅ 현재 일반화 강점

대부분의 단안 깊이 추정 모델이 실내 또는 실외 환경 중 하나에 국한된 특정 데이터셋에 과적합되는 경향이 있는 반면, Depth Pro는 역동적인 환경에서 in-the-wild 이미지에 대해 지속적으로 뛰어난 성능을 보이며 Zero-Shot 깊이 추정기로서 가장 선호되는 선택이 되고 있습니다.

AbsRel, Log10, δ2, δ3 등의 기타 메트릭은 Depth Anything 및 Metric3D 같은 일부 모델의 도메인 편향을 확인했는데, 이 모델들은 Zero-Shot 전제를 위반하는 도메인별 모델이나 크롭 크기에 의존합니다. 반면 Depth Pro는 강력한 일반화 성능을 보여주며 데이터셋 전체에서 지속적으로 상위 접근법 중 하나로 랭크됩니다.

### 🚀 일반화 성능 향상을 위한 가능성

#### (1) 다양한 ViT 사전훈련 백본 활용

플레인 ViT 인코더를 사용함으로써 Depth Pro는 다양한 사전 훈련된 백본을 활용할 수 있어 모델 성능을 향상시킬 수 있는 잠재력을 갖습니다. → 예를 들어, DINOv2 Large/Giant, EVA, SAM 인코더 등을 백본으로 교체하면 더 강력한 시각적 표현 학습이 가능합니다.

#### (2) 합성-실세계 데이터 혼합 전략

맞춤형 손실 함수와 특수한 훈련 체계의 조합이 날카로운 깊이 추정을 보장합니다. 이 커리큘럼은 경계 근방에 거칠고 부정확한 감독을 제공하는 실세계 데이터셋과 현실감은 낮지만 픽셀 정밀 정답을 제공하는 합성 데이터셋 간의 학습 균형을 맞춥니다.

#### (3) 초점 거리 추정의 독립화를 통한 일반화

Depth Pro의 성공은 초점 거리 훈련을 깊이 추정으로부터 분리하는 독특한 네트워크 아키텍처와 훈련 방식에 기인하며, 이것이 전반적인 정밀도를 향상시킵니다. → 초점 거리 추정을 별도로 훈련함으로써, 미지의 카메라 설정에 대한 로버스트성이 크게 향상됩니다.

#### (4) 도메인 특화 파인튜닝 없이 Zero-Shot 성능

Depth Pro는 임의의 이미지로부터 도메인별 데이터에 대한 추가 훈련 없이 Zero-Shot 조건에서 절대 스케일을 갖는 메트릭 깊이 맵을 생성함으로써 전통적인 방법의 간극을 메우고자 합니다.

#### (5) 확산 기반 모델과의 비교 관점

Marigold와 같은 확산 기반 깊이 모델들은 이미지에 대한 풍부한 구조적 지식을 가져오는 풍부한 잠재 공간 덕분에 탁월한 일반화 성능을 갖지만, 여러 번의 노이즈 제거 반복 사이클로 인해 느립니다. 반면 Depth Pro는 날카로운 경계를 위해 확산 사전 감독이나 복잡한 다단계 작업별 모듈이 필요하지 않습니다.

---

## 4️⃣ 앞으로의 연구에 미치는 영향 및 고려할 점

### 📌 연구에 미치는 영향

**① 파운데이션 모델로서의 단안 깊이 추정**

Depth Pro는 단안 깊이 추정의 중요한 발전을 나타내며, 혁신적인 아키텍처 설계와 훈련 방법론을 통해 높은 정확도와 효율성을 달성합니다. Zero-Shot 능력은 카메라 하드웨어 사양의 제한 없이 광범위한 실세계 시나리오에서의 적용 가능성을 확장합니다. 경계 정확도와 메트릭 깊이 추출 문제를 계산적으로 효율적인 방식으로 해결함으로써 컴퓨터 비전 및 관련 분야의 미래 연구와 응용을 위한 강력한 기반을 마련했습니다.

**② 평가 메트릭 패러다임 변화**

새로운 메트릭 집합을 도출하여 단안 깊이 맵에서 경계 추적 정확도를 정량화하기 위해 고정밀 매팅 데이터셋을 활용할 수 있게 합니다. → 이는 향후 경계 중심의 깊이 평가 프레임워크 발전에 핵심적인 영향을 줍니다.

**③ 다운스트림 애플리케이션에의 파급 효과**

Depth Pro는 깊이 네트워크의 특징으로부터 초점 거리를 직접 추정하여 다양한 실세계 응용에서의 다용성을 향상시킵니다. 이를 통해 메타데이터 없이도 임의의 이미지에서 특정 거리를 지정한 뷰 합성이 가능합니다.

**④ 2020년 이후 관련 최신 연구와의 비교 분석**

| 모델 | 연도 | 특징 | 스케일 | Zero-Shot | 경계 정확도 | 속도 |
|---|---|---|---|---|---|---|
| **MiDaS** | 2020 | 다중 데이터셋 혼합 학습 | 상대적 | ✅ | 낮음 | 빠름 |
| **Metric3D v1** | ICCV 2023 | 카메라 캐노니컬 공간 | **메트릭** | 부분적 | 중간 | 중간 |
| **ZoeDepth** | 2023 | 상대→메트릭 변환 | **메트릭** | 부분적 | 낮음 | 빠름 |
| **Marigold** | CVPR 2024 | Stable Diffusion 기반 | 상대적 | ✅ | 높음 | **매우 느림** |
| **Depth Anything v2** | 2024 | 대규모 데이터 학습 | 상대적 | ✅ | 중간 | 빠름 |
| **Metric3D v2** | 2024 | 기하 파운데이션 모델 | **메트릭** | ✅ | 중간 | 중간 |
| **UniDepth** | CVPR 2024 | 카메라 표현 분리 | **메트릭** | ✅ | 중간 | 중간 |
| **PatchFusion** | CVPR 2024 | 타일 기반 고해상도 | 상대적 | ✅ | 높음 | **느림** |
| **Depth Pro** | ICLR 2025 | 멀티스케일 ViT | **메트릭** | ✅ | **최고** | **0.3초** |

Metric3D v2와 Depth Anything v2 모델의 경쟁적인 메트릭 정확도가 날카로운 경계를 의미하지는 않습니다. Depth Pro는 머리카락, 털과 같은 얇은 구조에 대한 일관적으로 높은 리콜을 보이고 더 날카로운 경계를 생성합니다. 이는 수십억 개의 실세계 이미지로 훈련된 사전 모델을 활용하는 확산 기반 Marigold 및 가변 해상도로 동작하는 PatchFusion과 비교해도 마찬가지입니다.

Metric3D는 카메라 파라미터 접근이 필요하여 적용 가능성을 제한합니다. Depth Anything은 유연한 프레임워크를 제공하지만 현재 성능이 Zero-Shot 일반화 요건을 완전히 충족하지 못합니다. 또한 평가 표는 모델과 도메인에 걸쳐 상당한 성능 변동을 보여주어 여전히 강건한 일반화 달성에 상당한 도전이 있음을 나타냅니다.

---

### 📌 앞으로 연구 시 고려할 점

**① 야외·극단적 환경 일반화 강화**

특히 메트릭 깊이에서 실내와 야외 장면 간의 스케일링 차이로 인해 모든 데이터셋에 대해 단일 깊이 추정 모델을 훈련하면 성능이 저하되는 경향이 있습니다. 야생 환경, 의료 영상, 수중 등 특수 도메인에서의 일반화 연구가 중요합니다.

**② 투명·반사 표면 처리**

단안 깊이 추정 모델들은 일반적으로 복잡한 조명 및 굴절 특성을 가진 투명하고 반사적인 표면 처리에 어려움을 겪습니다. 이는 향후 반드시 극복해야 할 과제입니다.

**③ 비디오 일관성 (Temporal Consistency)**

Depth Anything은 단안 깊이 추정에서 뛰어난 일반화 능력으로 놀라운 성공을 거두었지만, 비디오에서 시간적 일관성 문제가 있어 실제 응용을 방해합니다. Depth Pro 역시 이미지 단위 모델로서, 비디오 시퀀스에서의 일관성 연구가 필요합니다.

**④ 더 강력한 사전 훈련 백본 통합**

최신 깊이 파운데이션 모델의 광범위한 채택은 실내 및 야외 시나리오 모두에서 상당한 개선으로 이어졌으며, 웹 규모 데이터 가용성의 결정적인 역할을 다시 한번 증명합니다. CLIP, LLaVA 등 거대 멀티모달 모델의 특징을 깊이 추정에 융합하는 연구가 유망합니다.

**⑤ 효율적 온디바이스 추론**

2.25 메가픽셀 깊이 맵을 표준 V100 GPU에서 단 0.3초 만에 생성하여 이미지 편집, 가상 현실, 증강 현실 같은 실시간 응용에서의 실용성을 보여줍니다. 그러나 모바일 기기에서의 실시간 추론을 위한 모델 경량화(Quantization, Pruning, Distillation) 연구가 요구됩니다.

---

## 📚 참고 자료

| # | 출처 | 링크 |
|---|---|---|
| 1 | **Depth Pro 논문 (arXiv:2410.02073)** | https://arxiv.org/abs/2410.02073 |
| 2 | **ICLR 2025 게재 논문 (proceedings.iclr.cc)** | https://proceedings.iclr.cc/paper_files/paper/2025/file/bc8b2058fd96978a4146f18298cb2d39-Paper-Conference.pdf |
| 3 | **Apple Machine Learning Research 공식 페이지** | https://machinelearning.apple.com/research/depth-pro |
| 4 | **GitHub 공식 코드 (apple/ml-depth-pro)** | https://github.com/apple/ml-depth-pro |
| 5 | **Hugging Face 모델 페이지 (apple/DepthPro-hf)** | https://huggingface.co/apple/DepthPro-hf |
| 6 | **Hugging Face Transformers 문서 (depth_pro.md)** | https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/depth_pro.md |
| 7 | **Medium 리뷰 (Andrew Lukyanenko)** | https://artgor.medium.com/paper-review-depth-pro-sharp-monocular-metric-depth-in-less-than-a-second-3f3cb7bea39a |
| 8 | **Medium — Apple Depth Pro 실습 (Nivitus)** | https://medium.com/@Nivitus./estimating-depth-and-focal-length-with-apple-depth-pro-e49a19392b47 |
| 9 | **LearnOpenCV 상세 설명** | https://learnopencv.com/depth-pro-monocular-metric-depth/ |
| 10 | **MarkTechPost 분석 기사** | https://www.marktechpost.com/2024/10/08/apple-ai-releases-depth-pro-a-foundation-model-for-zero-shot-metric-monocular-depth-estimation/ |
| 11 | **Synced Review 기사** | https://syncedreview.com/2024/10/07/instant-3d-vision-apples-depth-pro-delivers-high-precision-depth-maps-in-0-3-seconds/ |
| 12 | **The Moonlight Literature Review** | https://www.themoonlight.io/en/review/depth-pro-sharp-monocular-metric-depth-in-less-than-a-second |
| 13 | **Survey on Monocular Metric Depth Estimation (arXiv:2501.11841)** | https://arxiv.org/html/2501.11841v3 |
| 14 | **Metric3D v2 논문 (arXiv:2404.15506)** | https://arxiv.org/html/2404.15506v2 |
| 15 | **The Fourth Monocular Depth Estimation Challenge (arXiv:2504.17787)** | https://arxiv.org/html/2504.17787 |
| 16 | **Benchmark on Monocular Metric Depth in Wildlife Setting (arXiv:2510.04723)** | https://arxiv.org/html/2510.04723v1 |
| 17 | **Awesome-Monocular-Depth (GitHub 큐레이션 목록)** | https://github.com/choyingw/Awesome-Monocular-Depth |
