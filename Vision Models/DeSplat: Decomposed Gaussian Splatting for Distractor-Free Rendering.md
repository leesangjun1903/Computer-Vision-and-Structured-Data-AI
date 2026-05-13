
# DeSplat: Decomposed Gaussian Splatting for Distractor-Free Rendering 

---

## 📌 1. 핵심 주장 및 주요 기여 요약

Gaussian Splatting은 정적 3D 환경에서 빠른 Novel View Synthesis(새로운 시점 합성)를 가능하게 하지만, 실세계 환경을 재구성할 때 **distractors(방해 요소)** 또는 **occluders(폐색 물체)** 가 다시점 일관성(multi-view consistency) 가정을 위반하여 정확한 3D 재구성을 방해한다는 문제가 있습니다.

기존 대부분의 방법들은 사전 학습된 모델에서 외부 의미론적 정보(semantic information)에 의존하여 전처리 단계 또는 최적화 과정에서 추가 계산 비용이 발생합니다. 이에 DeSplat은 **오직 Gaussian 프리미티브의 볼륨 렌더링만을 기반으로** distractors와 정적 장면 요소를 직접 분리하는 새로운 방법을 제안합니다.

### ✅ 핵심 기여 3가지

| 기여 | 설명 |
|---|---|
| ① 외부 모델 불필요 | 사전 학습 모델 없이 Gaussian 볼륨 렌더링만으로 분리 |
| ② 명시적 장면 분해 | 정적 요소와 distractor를 명시적으로 분리 |
| ③ 렌더링 속도 유지 | 기존 방법 대비 속도 희생 없이 성능 달성 |

DeSplat은 정적 요소와 distractors의 명시적 장면 분리를 실현하며, 렌더링 속도를 희생하지 않고 기존 distractor-free 방법들과 비교 가능한 성능을 달성합니다.

---

## 📐 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

DeSplat은 특히 distractors(정적 환경의 정확한 재구성을 방해하는 동적 객체 또는 요소)를 포함하는 동적 장면을 처리할 때 전통적인 3D Gaussian Splatting의 문제를 해결하려 합니다.

이 방법은 외부 의미론적 정보나 사전 학습된 모델에 대한 의존성을 배제함으로써 계산 비용을 줄이고 전처리 단계를 제거합니다.

---

### 2-2. 제안하는 방법 및 수식

#### (1) 3DGS 기초: Gaussian 표현

전통적인 Gaussian Splatting에서 장면은 이산 폴리곤이나 메시가 아닌 3D Gaussian 함수의 집합으로 모델링됩니다. 각 Gaussian은 다음 파라미터로 공간의 한 점을 표현합니다: 위치 $\boldsymbol{\mu}$, 공분산 행렬 $\boldsymbol{\Sigma}$, 불투명도 $o$, 구면 조화 함수로 파라미터화된 색상 $c$.

3D Gaussian은 다음과 같이 정의됩니다:

$$
G(\mathbf{x}) = e^{-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}
$$

공분산 행렬은 스케일 행렬 $\mathbf{S}$와 회전 행렬 $\mathbf{R}$로 분해됩니다:

$$
\boldsymbol{\Sigma} = \mathbf{R} \mathbf{S} \mathbf{S}^\top \mathbf{R}^\top
$$

#### (2) 기존 Alpha Compositing (3DGS 렌더링)

기존 3DGS의 픽셀 색상 $\hat{C}$는 카메라 뷰를 따라 정렬된 $N$개의 Gaussian에 대해 다음과 같이 계산됩니다:

$$
\hat{C} = \sum_{i=1}^{N} c_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)
$$

여기서 $\alpha_i = o_i \cdot G_i^{2D}(\mathbf{x})$는 2D 투영된 Gaussian의 불투명도입니다.

#### (3) DeSplat의 핵심: 분해된 Alpha Compositing

DeSplat은 **정적 장면 요소와 동적 distractors를 분리하는 이중 Gaussian 표현(dual-Gaussian representation)** 을 통해 문제를 해결합니다. 구체적으로 두 종류의 Gaussian 포인트 세트를 초기화합니다: **정적 Gaussians $\mathcal{G}_s$**: 장면의 불변 요소를 표현, **Distractor Gaussians $\mathcal{G}_d$**: 뷰별로 모델링되어 순간적인 객체를 포착합니다.

Distractor Gaussians는 각 카메라 시점 앞에 초기화되어 많은 동적 요소의 시점 의존적 특성을 고려합니다. 이는 distractors가 프레임 간에 외형 또는 위치가 바뀔 수 있어 서로 다른 시점마다 별도의 Gaussian 표현이 필요하기 때문입니다.

DeSplat의 분해된 렌더링 수식은 다음과 같이 표현됩니다:

**Static scene 렌더링:**

$$
\hat{C}_s = \sum_{i \in \mathcal{G}_s} c_i \alpha_i^s \prod_{j < i}(1 - \alpha_j^s)
$$

**Distractor 렌더링 (per-view):**

$$
\hat{C}_d = \sum_{i \in \mathcal{G}_d} c_i \alpha_i^d \prod_{j < i}(1 - \alpha_j^d)
$$

**Alpha Compositing을 통한 최종 픽셀 색상:**

$$
\hat{C} = \hat{C}_d + (1 - A_d)\hat{C}_s
$$

여기서 $A_d = \sum_{i \in \mathcal{G}\_d} \alpha_i^d \prod_{j < i}(1 - \alpha_j^d)$ 는 distractor의 누적 투명도(accumulated transmittance)입니다.

DeSplat은 각 카메라 뷰의 distractor와 다시점 비일관성을 뷰별 Gaussians로 모델링하고, alpha compositing을 분해하여 방해 요소(occluder)와 배경 정적 3D 장면을 명시적으로 분리합니다.

#### (4) 손실 함수

두 파라미터 세트 ($\mathcal{G}_s$와 $\mathcal{G}_d$)는 표준 광측정 손실(photometric loss)과 일관된 동일한 손실 하에 공동으로 최적화됩니다.

전체 학습 손실은 다음과 같은 photometric loss를 기반으로 합니다:

$$
\mathcal{L} = \mathcal{L}_{\text{rgb}} + \lambda_{\text{SSIM}} \mathcal{L}_{\text{SSIM}}
$$

$$
\mathcal{L}_{\text{rgb}} = \|\hat{C} - C_{gt}\|_1
$$

---

### 2-3. 모델 구조

DeSplat은 정적 장면과 뷰별 distractors를 명시적으로 모델링하도록 3DGS를 분해합니다.

모델 구조는 다음 세 구성 요소로 이루어집니다:

```
입력 이미지 (다시점)
        │
        ▼
┌─────────────────────────┐
│  Static Gaussians (Gs)  │  ← 전역 3D 장면 표현
│  - 3D 공간에서 초기화    │
│  - 최적화 후 공유        │
└─────────────────────────┘
        +
┌─────────────────────────┐
│ Distractor Gaussians(Gd)│  ← 뷰별(per-view) 표현
│  - 각 카메라 앞에 초기화 │
│  - 2D 평면상 배치        │
└─────────────────────────┘
        │
        ▼
   Alpha Compositing
   (분해된 볼륨 렌더링)
        │
        ▼
  Distractor-Free Rendering
```

장면 분해 시각화 결과로, 최적화된 distractor Gaussians, 정적 3D 장면, 결합된 alpha-composited 이미지를 확인할 수 있습니다.

DeSplat은 NerfStudio 코드베이스 위에 구현되었습니다.

---

### 2-4. 성능 향상

DeSplat은 정적 요소와 distractors의 명시적 장면 분리를 실현하며, 렌더링 속도를 희생하지 않고 기존 distractor-free 방법들에 필적하는 성능을 달성하였습니다. 세 가지 벤치마크 데이터셋에서 효과가 입증되었습니다.

평가 데이터셋은 다음과 같습니다:
- **RobustNeRF Dataset** — 실내외 distractor-filled 장면 포함
- **NeRF On-the-go Dataset** — 야외 casual capture 장면 포함
- **추가 벤치마크 1개**

RobustNeRF 데이터셋은 distractor가 있는 훈련 분할과 distractor-free 훈련 분할을 포함하는 네 개의 장면으로 구성되어, 방해 요소가 있는 모델과 깨끗한 이미지로 학습된 'clean' 모델의 비교를 허용합니다.

---

### 2-5. 한계점

현재 구현은 여러 움직이는 객체를 포함하는 복잡한 장면에서 어려움을 겪을 수 있습니다. 또한 부분 폐색이 발생하거나 객체가 매우 느리게 움직이는 경우 정적 요소와 동적 요소 사이의 분리가 덜 신뢰할 수 있게 됩니다.

미래 연구는 여러 움직이는 객체가 있는 더 동적인 장면을 처리하고, 의도적인 움직임과 바람에 흔들리는 식물 같은 환경적 요인을 구별하는 시스템 능력을 향상시키는 방향을 탐색할 수 있습니다.

---

## 🔍 3. 모델의 일반화 성능 향상 가능성

DeSplat의 일반화 성능과 관련된 핵심 논점은 다음과 같습니다.

### 3-1. DeSplat의 일반화 강점

DeSplat의 핵심 장점 중 하나는 **외부 모델(Diffusion, DINO 등)에 의존하지 않는다**는 점입니다. 기존 방법들은 사전 학습된 모델의 외부 의미론적 정보에 의존하여 추가 계산 비용이 발생하는 반면, DeSplat은 Gaussian 프리미티브의 볼륨 렌더링만으로 distractors를 직접 분리합니다. 이는 특정 도메인의 semantic model에 종속되지 않아 다양한 환경에서의 일반화 가능성을 시사합니다.

### 3-2. 동시 연구들과의 비교 (Generalization 관점)

HybridGS는 distractors와 정적 요소를 2DGS와 3DGS로 각각 명시적으로 분리한다는 DeSplat과 유사한 아이디어를 도입한 동시 연구입니다.

T-3DGS는 입력 비디오에서 distractors를 제거하여 3DGS로 재구성하며, Segment Anything Model(SAM)을 사용해 순간 마스크를 정제합니다.

### 3-3. DGGS — 일반화에 특화된 후속 연구

DeSplat의 한계를 발전시킨 연구로 DGGS(Distractor-free Generalizable 3D Gaussian Splatting)가 있습니다:

DGGS는 Distractor-free Generalizable 3D Gaussian Splatting이라는 이전에 탐구되지 않았던 과제를 해결하는 새로운 프레임워크로, **훈련과 추론 단계 모두에서** distractor가 있는 데이터에 대한 일반화 가능한 3DGS를 강화하면서 기존 distractor-free 방법에 크로스씬(cross-scene) 적응 능력을 성공적으로 확장합니다.

이를 위해 DGGS는 훈련 단계에서 장면에 구애받지 않는(scene-agnostic) 참조 기반 마스크 예측 및 정제 방법론과 훈련 뷰 선택 전략을 도입하여 distractor 예측 정확도와 훈련 안정성을 향상시킵니다. 또한 추론 단계에서 distractor로 인한 공백과 아티팩트를 해결하기 위해 더 나은 참조 선택을 위한 2단계 추론 프레임워크를 제안합니다.

### 3-4. Sparse View 조건에서의 일반화 도전

Transient 객체 문제를 해결하기 위해 distractor-free 3DGS 방법들이 등장하여 **밀집된 이미지 촬영이 가능할 때는** 좋은 성능을 보이지만, **희소 입력 조건에서는 성능이 크게 저하됩니다.** 이는 색상 잔차 휴리스틱(color residual heuristics)에 대한 의존이 관측이 제한될 때 신뢰할 수 없어지기 때문입니다.

---

## 📊 4. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 연도 | 기반 | 핵심 전략 | 외부 모델 의존 | 속도 |
|---|---|---|---|---|---|
| **NeRF-W** | 2021 | NeRF | 불확실성 필드로 distractors 모델링 | ❌ | 느림 |
| **RobustNeRF** | 2023 | NeRF | 광측정 잔차 기반 강건 손실 | ❌ | 느림 |
| **NeRF On-the-go** | 2024 | NeRF | DINOv2 불확실성 예측기 | ✅ DINOv2 | 느림 |
| **WildGaussians** | 2024 | 3DGS | DINO feature + 외형 모델링 | ✅ DINOv2 | 빠름 |
| **SpotLessSplats** | 2024/25 | 3DGS | Stable Diffusion feature 기반 클러스터링 | ✅ Diffusion | 중간 |
| **DeSplat** | 2024/25 | 3DGS | 순수 Gaussian 볼륨 렌더링 분해 | ❌ | **빠름** |
| **HybridGS** | 2024 | 2D+3DGS | 2DGS(distractor)+3DGS(static) | ❌ | 빠름 |
| **DGGS** | 2024 | 3DGS | Scene-agnostic 일반화 | 부분적 | 중간 |
| **DualSplat** | 2025 | 3DGS | Pseudo-mask 부트스트래핑 | 부분적 | 빠름 |

RobustNeRF는 광측정 잔차로부터 잘린 마스크를 도출하는 강건 추정 관점을 취하고, NeRF On-the-go는 DINOv2 특성에 대한 불확실성 예측기를 훈련하여 픽셀별 재구성 손실을 조절합니다.

SpotLessSplats는 순수 색상 잔차 전략과 달리, 사전 학습된 확산 모델에서 의미론적 특성을 활용하여 특성 맵을 훈련 이전에 오프라인으로 추출한 후 클러스터링 또는 경량 MLP를 사용하여 특성 공간에서 구조화된 이상치를 분리합니다.

WildGaussians는 강건한 DINO 특성을 활용하고 외형 모델링 모듈을 3DGS 내에 통합함으로써 폐색 및 외형 변화를 처리하여 최첨단 결과를 달성합니다.

DeGauss는 분리된 동적-정적 Gaussian Splatting 설계에 기반한 간단하고 강건한 자기지도(self-supervised) 프레임워크로, 전경 Gaussians로 동적 요소를, 배경 Gaussians로 정적 콘텐츠를 모델링하며, 확률적 마스크를 사용하여 구성을 조율하고 독립적이면서도 보완적인 최적화를 가능하게 합니다.

DeGauss는 임시 이미지 컬렉션부터 길고 동적인 에고센트릭 비디오까지 폭넓은 실세계 시나리오에서 복잡한 휴리스틱이나 광범위한 지도 학습 없이 강건하게 일반화됩니다.

---

## 🚀 5. 앞으로의 연구에 미치는 영향과 고려 사항

### 5-1. DeSplat이 미치는 영향

#### ① 외부 의존성 없는 분해 패러다임 확립
DeSplat의 가장 큰 기여는 **외부 사전 학습 모델 없이** Gaussian 렌더링 자체의 구조적 분해만으로 distractor 제거가 가능함을 증명한 것입니다. 이는 향후 3DGS 기반 강건 재구성 연구에서 lightweight 솔루션의 가능성을 제시합니다.

#### ② 명시적 장면 분해의 활용
DeSplat은 정적 장면과 뷰별 distractors를 **명시적으로** 모델링하도록 3DGS를 분해합니다. 이 명시적 분해 아이디어는 이후 HybridGS, DualSplat, DeGauss 등 여러 후속 연구에서 변형 및 발전되었습니다.

#### ③ Sparse View / Generalizable 방향으로의 확장

희소 시점 조건에서 distractor-free 3DGS를 개선하기 위해 풍부한 사전 정보를 통합하는 프레임워크가 후속 연구로 제안되고 있습니다. DeSplat은 이러한 방향성의 출발점 역할을 합니다.

---

### 5-2. 향후 연구 시 고려할 점

#### 🔷 기술적 한계 극복
복잡한 장면에서 여러 움직이는 객체가 있거나, 부분 폐색이 발생하거나, 객체가 매우 느리게 움직이는 경우 정적/동적 분리의 신뢰성이 저하됩니다. 향후 연구에서는:

- **Inpainting 모델과의 결합**: 광범위한 상호 폐색 아래에서의 성능 저하는 불가피하므로, 예측된 마스크를 기반으로 inpainting 모델을 통합하는 방식으로 이 한계를 보완할 수 있습니다.
- **느린 객체 처리 개선**: 바람에 흔들리는 식물 같은 환경적 요인과 의도적 움직임을 구별하는 능력 향상이 필요합니다.

#### 🔷 일반화 성능 강화 (Cross-Scene)
DGGS의 실험 결과는 장면에 구애받지 않는 마스크 추론이 장면별로 훈련된 방법과 비교 가능한 정확도를 달성함을 보여줍니다. DeSplat 기반에 이러한 scene-agnostic 접근을 결합하는 것이 중요한 발전 방향입니다.

#### 🔷 희소 시점(Sparse View)에서의 강건성
기존 distractor-free 3DGS 방법들은 밀집된 이미지 촬영이 가능할 때는 유망한 결과를 보이지만, 희소 입력 조건에서는 성능이 크게 저하됩니다. 소수의 이미지만으로도 강건하게 작동하는 시스템 개발이 실용적으로 매우 중요합니다.

#### 🔷 비디오 및 동적 장면으로의 확장
DeGauss처럼 NeRF-on-the-go, ADT, AEA, Hot3D, EPIC-Fields 등 다양한 벤치마크에서 실험함으로써 일반화된 distractor-free 3D 재구성을 위한 강한 베이스라인을 확립하는 것이 필요합니다.

#### 🔷 렌더링 속도와 품질의 균형
WildGaussians는 3DGS의 실시간 렌더링 속도를 달성하면서 in-the-wild 데이터 처리에서 3DGS와 NeRF 베이스라인을 모두 능가합니다. 속도와 품질의 균형을 유지하면서 강건성을 높이는 것이 핵심 과제입니다.

---

## 📚 참고 자료 및 출처

| # | 제목 | 출처/링크 |
|---|---|---|
| 1 | **DeSplat: Decomposed Gaussian Splatting for Distractor-Free Rendering** | [arXiv:2411.19756](https://arxiv.org/abs/2411.19756) |
| 2 | DeSplat 공식 프로젝트 웹사이트 | [aaltoml.github.io/desplat](https://aaltoml.github.io/desplat/) |
| 3 | DeSplat CVPR 2025 Open Access | [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_DeSplat_Decomposed_Gaussian_Splatting_for_Distractor-Free_Rendering_CVPR_2025_paper.html) |
| 4 | DeSplat GitHub 코드 저장소 | [github.com/AaltoML/desplat](https://github.com/AaltoML/desplat) |
| 5 | **SpotLessSplats: Ignoring Distractors in 3D Gaussian Splatting** | [arXiv:2406.20055](https://arxiv.org/html/2406.20055v1) / ACM Transactions on Graphics |
| 6 | **WildGaussians: 3D Gaussian Splatting in the Wild** (NeurIPS 2024) | [wild-gaussians.github.io](https://wild-gaussians.github.io/) |
| 7 | **Distractor-free Generalizable 3D Gaussian Splatting (DGGS)** | [arXiv:2411.17605](https://arxiv.org/abs/2411.17605) |
| 8 | **DeGauss: Dynamic-Static Decomposition with Gaussian Splatting** (ICCV 2025) | [batfacewayne.github.io/DeGauss](https://batfacewayne.github.io/DeGauss.io/) |
| 9 | **DualSplat: Robust 3D Gaussian Splatting via Pseudo-Mask Bootstrapping** | [arXiv:2604.21631](https://arxiv.org/html/2604.21631) |
| 10 | **Sparse View Distractor-Free Gaussian Splatting** | [arXiv:2603.01603](https://arxiv.org/abs/2603.01603) |
| 11 | DeSplat Literature Review (Moonlight AI) | [themoonlight.io](https://www.themoonlight.io/en/review/desplat-decomposed-gaussian-splatting-for-distractor-free-rendering) |
| 12 | DeSplat AI Models Summary | [aimodels.fyi](https://www.aimodels.fyi/papers/arxiv/desplat-decomposed-gaussian-splatting-distractor-free-rendering) |
| 13 | **ForestSplats: Deformable transient field for Gaussian Splatting in the Wild** | [arXiv:2503.06179](https://arxiv.org/html/2503.06179v3) |

> ⚠️ **정확도 고지**: DeSplat의 세부 수식 중 일부(특히 분해된 alpha compositing 수식)는 논문 원문의 공개된 내용과 Gaussian Splatting의 표준 수식을 기반으로 재구성한 것입니다. 모델 내부의 정확한 수식 표기는 [원문 논문 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_DeSplat_Decomposed_Gaussian_Splatting_for_Distractor-Free_Rendering_CVPR_2025_paper.pdf)를 직접 참조하시기 바랍니다.
