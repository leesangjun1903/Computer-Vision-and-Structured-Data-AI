
# No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views

> **논문 정보**
> - **제목:** No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views
> - **저자:** Ranran Huang, Krystian Mikolajczyk
> - **arXiv:** [2508.01171](https://arxiv.org/abs/2508.01171) (2025년 8월)
> - **학회:** ICCV 2025 **Highlight**
> - **프로젝트 페이지:** https://ranrhuang.github.io/spfsplat/
> - **코드:** https://github.com/ranrhuang/SPFSplat

---

## 1. 핵심 주장 및 주요 기여 요약

SPFSplat은 희소한 다시점 이미지로부터 3D Gaussian Splatting을 수행하는 효율적인 프레임워크로, 훈련 및 추론 시 정답(ground-truth) 카메라 포즈를 전혀 필요로 하지 않는다.

핵심 기여는 세 가지다: (1) 정답 포즈 없이 희소 뷰에서 3D Gaussian과 카메라 포즈를 Canonical Space에서 동시에 예측하는 SPFSplat 프레임워크 제안, (2) 이 방법이 극단적인 시점 변화 및 제한된 이미지 오버랩 상황에서도 SOTA Pose-Required 및 Supervised Pose-Free NVS 방법을 능가하는 최초의 자기 지도(self-supervised) 포즈-프리 방법임, (3) 포즈 감독 없이도 기하 Prior에 의존하는 최신 방법보다 우수한 상대적 포즈 추정 성능 달성.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 방법들의 문제를 세 파이프라인으로 분류할 수 있다: (a) **Pose-Required 방법**은 3D 장면 복원과 타깃 뷰 렌더링 모두에 정답 포즈를 필요로 하고, (b) **Supervised Pose-Free 방법**은 복원에는 포즈가 불필요하지만 렌더링 손실 계산에 여전히 정답 포즈를 사용하며, (c) **Self-Supervised Pose-Free(본 논문)** 방법은 타깃 포즈를 추정하여 비포즈 이미지로부터 3D 장면 복원을 최적화한다.

이 접근에는 본질적 도전이 존재한다: 렌더링 손실이 3D 장면 기하와 카메라 포즈의 학습을 내재적으로 결합하므로, 포즈 오류가 복원 품질을 저하시키고 이것이 다시 포즈 추정을 방해하는 악순환이 발생한다. 기존의 Supervised Pose-Free 방법들은 장면 복원과 포즈 추정을 별개의 모듈로 처리해 일관된 특징 표현 학습을 방해하기 때문에 이 문제를 완화하기 어렵다.

---

### 2.2 제안하는 방법 (수식 포함)

#### ① 전체 훈련 패러다임

SPFSplat은 **인코더(Encoder)**, **디코더(Decoder)**, **포즈 헤드(Pose Head)**, **Gaussian 예측 헤드(Gaussian Prediction Heads)** 의 네 가지 주요 구성 요소로 이루어진다. 이 전문화된 헤드들은 공유된 ViT 백본에 통합되어, 비포즈 이미지로부터 Canonical Space에서 Gaussian 중심, 추가적 Gaussian 파라미터, 카메라 포즈를 동시에 예측하며, 첫 번째 입력 뷰가 레퍼런스로 사용된다. 컨텍스트-온리 브랜치(위)는 추론 시에만 사용되고, 컨텍스트-위드-타깃 브랜치(아래)는 훈련 중 타깃 포즈 추정에만 사용되어 렌더링 손실 감독에 활용된다. 추가적으로, Reprojection Loss가 두 브랜치의 추정된 컨텍스트 포즈를 사용해 Gaussian 중심과 대응 픽셀 간의 정렬을 강제한다. 이 방법은 3D Gaussian과 포즈를 공동으로 최적화하여 기하학적 일관성과 복원 품질을 향상시킨다.

#### ② 인코더-디코더 구조

각 입력 뷰에 대해 RGB 이미지는 먼저 패치화(patchify)되고 이미지 토큰 시퀀스로 평탄화된다. 스케일 모호성을 완화하기 위해, 카메라 내부 파라미터(intrinsic)를 선형 레이어를 통해 추가 토큰으로 인코딩하고 이미지 토큰과 공간 차원으로 연결한다. 이 연산은 선택적(optional)이다.

내부 파라미터를 주입하지 않는 경우에도, SPFSplat은 기존 방법들을 능가하는 성능을 보인다.

#### ③ 포즈 추정 (Pose Head)

포즈 헤드 $f_\phi$는 뷰 $I_v$에서 레퍼런스 뷰 $I_1$으로의 상대적 변환을 추정한다. 추정된 포즈는 다음과 같이 표현된다:

$$P^{v \to 1} = [R^{v \to 1} \mid T^{v \to 1}]$$

여기서 $R^{v \to 1} \in SO(3)$은 회전 행렬, $T^{v \to 1} \in \mathbb{R}^3$은 이동 벡터이다.

SfM 방법과 유사하게, 이 접근 방법은 3D 포인트와 카메라 포즈를 공동으로 예측한다. 훈련 중 사용되는 이미지 렌더링 및 Reprojection Loss는 추정된 장면 표현과 포즈를 공동으로 정제하고 정렬하는 번들 조정(bundle adjustment)의 형태로 해석될 수 있다.

#### ④ 손실 함수 (Loss Functions)

**전체 훈련 손실**은 다음 세 항의 가중합으로 구성된다:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{render}} + \lambda_{\text{LPIPS}} \cdot \mathcal{L}_{\text{LPIPS}} + \lambda_{\text{reproj}} \cdot \mathcal{L}_{\text{reproj}}$$

**렌더링 손실 (Rendering Loss):**

추정된 타깃 포즈 $\hat{P}^{t \to 1}$를 사용하여 타깃 뷰를 렌더링하고, 실제 RGB 이미지 $I_t^{\text{gt}}$와의 $L_2$ 손실 및 LPIPS를 계산한다:

$$\mathcal{L}_{\text{render}} = \| \hat{I}_t - I_t^{\text{gt}} \|_2^2$$

$$\mathcal{L}_{\text{LPIPS}} = \text{LPIPS}(\hat{I}_t, I_t^{\text{gt}})$$

LPIPS와 Reprojection Loss의 손실 가중치는 각각 0.05와 0.001로 설정된다.

**Reprojection Loss:**

Reprojection Loss는 예측된 Gaussian을 대응하는 이미지 픽셀과 명시적으로 정렬하여, 더 강한 기하학적 제약을 부과하여 훈련 안정성을 향상시킨다.

3D Gaussian 중심 $\mathbf{G}_p^{(v)}$를 컨텍스트 포즈 $\hat{P}^{v \to 1}$를 사용해 이미지 평면으로 투영할 때:

$$\hat{\mathbf{p}}_i = \pi\!\left(K \cdot [R^{v \to 1} \mid T^{v \to 1}] \cdot \mathbf{G}_p^{(v)}\right)$$

Reprojection Loss는 예측된 3D Gaussian 중심과 대응하는 2D 픽셀 사이의 Reprojection 오차를 최소화하여 기하학적 일관성을 강제한다.

$$\mathcal{L}_{\text{reproj}} = \sum_{i} \| \hat{\mathbf{p}}_i - \mathbf{p}_i^{\text{gt}} \|_2$$

---

### 2.3 모델 구조

```
입력: 비포즈 희소 이미지 {I_1, I_2, ..., I_N}
                     │
          ┌──────────┴──────────┐
          │   공유 ViT 백본     │   (MASt3R 사전학습 초기화)
          │  (인코더 + 디코더)  │
          └──────────┬──────────┘
           ┌─────────┼─────────┐
           │         │         │
    Gaussian     Gaussian    포즈 헤드
    Center Head  Param Head  (Pose Head)
           │         │         │
      3D Gaussian 중심  추가 파라미터  P^{v→1}
     (위치, 공분산 등)  (색상, 불투명도)  (R, T)
           └─────────┴─────────┘
                     │
          ┌──────────┴──────────┐
    [훈련]                  [추론]
Context+Target Branch    Context-only Branch
  (타깃 포즈 추정)         (직접 Gaussian 출력)
  → Rendering Loss        → Novel View 렌더링
  → Reprojection Loss
```

이 통합된 아키텍처는 계산 효율성을 높일 뿐만 아니라 장면 복원과 포즈 추정을 위한 공동 특징 학습을 촉진하여 기하학적 일관성을 개선하고 불안정한 피드백 루프를 완화한다. 이는 3D 기하가 정확한 컨텍스트-인식 카메라 정렬로부터 이점을 얻고, 포즈 예측이 전역 장면 컨텍스트를 활용하게 함으로써 상호 강화(mutual reinforcement) 형태로 달성된다.

---

### 2.4 성능 향상 및 한계

#### 성능 향상

포즈 감독이 전혀 없음에도 불구하고, SPFSplat은 상당한 시점 변화와 제한된 이미지 오버랩 하에서도 Novel View Synthesis에서 SOTA 성능을 달성한다.

평가 시 포즈 정렬(EPA) 없이도, SPFSplat은 NoPoSplat을 능가하는데, 이는 추정된 포즈가 Gaussian과 잘 정렬되어 있음을 보여준다.

SPFSplat은 같은 A6000 GPU에서 두 장의 256×256 이미지로부터 3D Gaussian을 0.044초 만에 복원하며 높은 효율성을 달성한다. 이는 feed-forward 네트워크를 사용해 Canonical Space에서 Gaussian을 직접 구성하기 때문이다.

또한 포즈 추정에서도, 기하학적 Prior로 훈련된 최신 방법들을 능가한다.

#### 한계점

- 이 방법에는 불안정한 훈련으로 이어지는 오버피팅 문제가 존재한다. 구체적으로, 네트워크는 3D Gaussian 공간이 카메라 좌표계로 정의된 첫 번째 컨텍스트 뷰의 렌더링 품질 개선을 우선시하게 되어, 다른 컨텍스트 뷰에서의 Gaussian 중심을 이동시키고 카메라 포즈를 조정함으로써 해당 뷰의 기여도를 억제하는 방향으로 학습이 진행될 수 있다.

- 일부 실패 사례에서 정렬 오류로 인한 블렌딩 아티팩트와 고스팅 효과가 관찰되며, 특히 극단적인 시점 변화와 텍스처리스(textureless) 영역(예: 창문), 장면 기하(예: 다리), 미세 디테일(예: 수영장) 처리에서 어려움을 보인다.

---

## 3. 모델의 일반화 성능 향상 가능성

일반화 성능은 이 논문의 핵심 강점 중 하나이다.

### 3.1 Zero-Shot 크로스 도메인 일반화

RE10K에서만 훈련된 모델을 ACID에서 평가하여 일반화 성능을 측정한다. Table 4의 크로스 데이터셋 일반화 결과에서, SPFSplat은 RE10K에서 훈련하고 ACID 및 DTU에서 제로샷(zero-shot) 방식으로 평가할 때 SOTA 방법들을 능가하며, ACID에서 학습한 NoPoSplat보다도 우수한 성능을 보인다.

### 3.2 일반화를 가능하게 하는 핵심 설계 요소

| 요소 | 역할 |
|---|---|
| **Canonical Space 예측** | 포즈 오류의 전파 억제 |
| **공유 ViT 백본** | 포즈·기하 학습의 상호 강화 |
| **Reprojection Loss** | 기하학적 일관성 강제 |
| **카메라 내부 파라미터 임베딩** | 스케일 모호성 해소 |

Canonical Space 기반 방법들에서 영감을 받아, SPFSplat은 레퍼런스 뷰를 기준으로 3D Gaussian 프리미티브를 직접 예측하여 포즈 오류가 장면 기하에 미치는 영향을 줄인다.

### 3.3 포즈-프리 학습의 데이터 확장성

정답 포즈에 대한 의존성을 제거함으로써, 이 방법은 더 크고 다양한 데이터셋을 활용하는 확장성(scalability)을 제공한다.

후속 연구 SPFSplatV2에서 이 가능성을 실증했는데, DL3DV를 훈련에 통합하면 두 변형 모두 서로 다른 벤치마크에서 NoPoSplat을 포함한 모든 이전 방법을 능가하며, 정답 포즈 감독 없이도 프레임워크의 확장성과 효과성을 입증한다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 연도 | 포즈 필요 (훈련) | 포즈 필요 (추론) | 핵심 접근 | 학회 |
|---|---|:---:|:---:|---|---|
| **NeRF** | 2021 | ✅ | ✅ | 암묵적 신경 복사장 | ECCV |
| **pixelSplat** | 2023 | ✅ | ✅ | 이미지 쌍에서 3DGS 예측 | CVPR |
| **MVSplat** | 2024 | ✅ | ✅ | Cost Volume 기반 3DGS | ECCV |
| **DUSt3R/MASt3R** | 2024 | ❌ | ❌ | 포인트맵 회귀 | CVPR/ECCV |
| **Splatt3R** | 2024 | ❌ | ❌ | MASt3R 기반 3DGS | - |
| **NoPoSplat** | 2024 | ✅ (포즈) | ❌ | Canonical Space, 포토메트릭 손실만 | - |
| **PF3plat** | 2024 | ❌ (지도학습 free) | ❌ | 단안 깊이 + 시각적 대응 활용 | ICML |
| **SelfSplat** | 2024 | ❌ | ❌ | Pose-free & 3D prior-free | CVPR |
| **InstantSplat** | 2024 | ❌ | ❌ | MASt3R 초기화 + 최적화 | - |
| **SPFSplat (본 논문)** | 2025 | ❌ | ❌ | Self-supervised, 공유 ViT 백본 | ICCV Highlight |

#### 주요 비교 방법 설명

**NoPoSplat (arXiv:2410.24207, 2024)**
NoPoSplat은 비포즈 희소 다시점 이미지로부터 3D Gaussian으로 파라미터화된 3D 장면을 복원할 수 있는 feed-forward 모델로, 포토메트릭 손실만으로 훈련되어 추론 시 실시간 3D Gaussian 복원을 달성한다.

NoPoSplat은 두 단계로 포즈를 추정한다: 먼저 예측된 Gaussian 중심을 활용해 RANSAC이 포함된 PnP로 입력 뷰 간 상대 포즈를 초기화하고, Gaussian 파라미터를 고정한 채 포토메트릭 손실과 SSIM 손실의 조합을 최소화하여 포즈를 정제한다. 이 두 번째 단계 최적화는 3D Gaussian Splatting을 루프에 통합하므로 계산 비용이 높고 실시간 응용에 적합하지 않다.

**PF3plat (arXiv:2410.22128, 2024)**
PF3plat은 서로 다른 뷰에서 잘못 정렬된 3D Gaussian이 불안정한 훈련의 원인임을 파악하여, 사전학습된 단안 깊이 추정 및 시각적 대응 모델을 사용해 3D Gaussian의 거친 정렬을 달성하고, 경량 학습 모듈을 도입하여 깊이와 포즈 추정을 정제한다.

**InstantSplat (arXiv:2403.20309, 2024)**
InstantSplat은 feed-forward 모델의 풍부한 기하학적 Prior와 3DGS 표현을 결합하여 SfM의 필요성을 제거하며, 미분 가능한 신경 렌더링을 통해 2D 픽셀을 3D로 비투영(unproject)하고 정렬하여 3D 장면 표현과 카메라 포즈를 공동으로 업데이트하는 자기 지도 최적화 프레임워크를 채택한다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려 사항

### 5.1 연구에 미치는 영향

**① 포즈-프리 학습 패러다임의 실용화**

포즈-프리 훈련 패러다임과 효율적인 1단계 feed-forward 설계는 SPFSplat을 실제 응용에 매우 적합하게 만든다. 이는 라벨링 비용이 높은 정답 포즈 데이터 없이도 고품질 3D 복원이 가능함을 입증해, 자율주행, AR/VR, 로보틱스 등 분야에서의 활용 가능성을 열어준다.

**② 대규모 데이터 확장의 길 개척**

정답 포즈로부터의 독립성은 대규모 및 다양한 실세계 데이터셋으로의 확장성을 강조하며, 이는 확장 가능하고 일반화 가능한 3D 복원의 미래 발전을 위한 길을 열어준다.

**③ 공동 최적화 아키텍처의 우수성 입증**

통합 백본은 계산 효율성을 향상시키고 장면 복원과 포즈 추정을 위한 공동 특징 학습을 촉진하여 기하학적 일관성을 향상시킨다. 이는 3D 기하가 컨텍스트-인식 카메라 정렬로부터 이익을 얻고 포즈 예측이 전역 장면 컨텍스트를 활용할 수 있도록 함으로써 달성된다.

**④ 후속 연구 SPFSplatV2의 등장**

SPFSplatV2는 SPFSplat을 기반으로 masked attention을 통해 계산 비용을 줄이고 잠재적 포즈 정렬 오류를 완화하며, 학습 가능한 포즈 토큰을 통해 포즈 추정을 향상시키고, 다시점 드롭아웃(multi-view dropout) 전략을 추가하여 전반적인 성능 및 다양한 입력 구성에 대한 일반화를 개선한다.

---

### 5.2 앞으로 연구 시 고려할 점

#### (1) 훈련 안정성 문제 해결
불안정한 훈련으로 인한 오버피팅 문제가 여전히 존재한다. 네트워크가 첫 번째 컨텍스트 뷰의 렌더링 품질 개선을 우선시하면서 다른 뷰의 기여를 억제하는 경향이 있다. 이를 해결하기 위한 정규화 전략 또는 학습 커리큘럼 설계가 중요한 연구 방향이다.

#### (2) 더 많은 입력 뷰로의 확장

현재 평가는 주로 2뷰 설정이나 제한된 다시점으로 수행되었다. RealEstate10K에서의 10뷰 평가가 일부 제시되었지만, 다양한 뷰 수에 대한 확장성 연구가 필요하다.

#### (3) 고해상도 및 대규모 장면 처리

모든 실험이 256×256 해상도에서 수행되었다. 실제 응용에서의 고해상도 지원 및 도시 스케일 장면으로의 확장은 중요한 미래 연구 방향이다.

#### (4) 동적 장면 및 객체 처리
현재 SPFSplat은 정적 장면을 가정한다. 동적 객체나 움직이는 요소가 포함된 장면에서의 포즈 추정 및 Gaussian 예측의 강건성 향상이 필요하다.

#### (5) 다양한 데이터 도메인 활용

이 프레임워크는 대규모 비포즈 데이터셋으로 효율적으로 확장되며, 훈련 데이터가 증가할수록 성능이 향상된다. 웹 규모의 비디오 데이터, 위성 이미지, 의료 이미징 등 다양한 비포즈 데이터 소스를 활용하는 연구가 기대된다.

#### (6) 텍스처리스 및 반사 영역 처리

텍스처리스 영역(예: 창문)이나 미세 디테일 처리에서 실패 사례가 관찰된다. 이를 위한 Depth 추정 정확도 향상이나 의미론적(semantic) Prior 활용 방안이 연구되어야 한다.

---

## 📚 참고 자료 및 출처

| # | 제목 | 저자 | 출처 |
|---|---|---|---|
| 1 | **No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views (SPFSplat)** | Ranran Huang, Krystian Mikolajczyk | [arXiv:2508.01171](https://arxiv.org/abs/2508.01171), ICCV 2025 Highlight |
| 2 | **SPFSplat Project Page** | Ranran Huang | https://ranrhuang.github.io/spfsplat/ |
| 3 | **SPFSplat GitHub Repository** | ranrhuang | https://github.com/ranrhuang/SPFSplat |
| 4 | **SPFSplat ICCV 2025 Poster** | - | https://iccv.thecvf.com/virtual/2025/poster/2493 |
| 5 | **SPFSplat PDF (ICCV 2025 Open Access)** | - | https://www.openaccess.thecvf.com/content/ICCV2025/papers/Huang_No_Pose_at_All... |
| 6 | **SPFSplat HTML (arXiv)** | - | https://arxiv.org/html/2508.01171v1 |
| 7 | **SPFSplatV2: Efficient Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views** | Ranran Huang, Krystian Mikolajczyk | [arXiv:2509.17246](https://arxiv.org/html/2509.17246) |
| 8 | **HuggingFace Paper Page (SPFSplat)** | - | https://huggingface.co/papers/2508.01171 |
| 9 | **No Pose, No Problem: Surprisingly Simple 3D Gaussian Splats from Sparse Unposed Images (NoPoSplat)** | Ye et al. | [arXiv:2410.24207](https://noposplat.github.io/), 2024 |
| 10 | **PF3plat: Pose-Free Feed-Forward 3D Gaussian Splatting** | Hong et al. | [arXiv:2410.22128](https://arxiv.org/pdf/2410.22128), ICML 2025 |
| 11 | **InstantSplat: Sparse-view Gaussian Splatting in Seconds** | Fan et al. | [arXiv:2403.20309](https://instantsplat.github.io/), 2024 |
| 12 | **SPFSplatV2 emergentmind summary** | - | https://www.emergentmind.com/papers/2509.17246 |
| 13 | **ResearchGate - No Pose at All** | - | https://www.researchgate.net/publication/394292804 |
