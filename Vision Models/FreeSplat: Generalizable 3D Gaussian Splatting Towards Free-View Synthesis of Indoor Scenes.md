
# FreeSplat: Generalizable 3D Gaussian Splatting Towards Free-View Synthesis of Indoor Scenes

> **논문 정보**
> - **저자**: Yunsong Wang, Tianxin Huang, Hanlin Chen, Gim Hee Lee (National University of Singapore)
> - **게재**: NeurIPS 2024
> - **arXiv**: [2405.17958](https://arxiv.org/abs/2405.17958)
> - **공식 코드**: [github.com/wangys16/FreeSplat](https://github.com/wangys16/FreeSplat)

---

## 1. 핵심 주장 및 주요 기여 요약

3D Gaussian Splatting(3DGS)에 일반화 능력을 부여하는 것은 매력적인 연구 방향이다. 그러나 기존의 일반화 가능한 3DGS 방법들은 무거운 백본으로 인해 스테레오 이미지 간의 좁은 범위 보간에 크게 국한되어, 3D Gaussian을 정확히 위치화하고 넓은 시점 범위에서 자유 시점 합성을 지원하는 능력이 부족하였다.

이에 대응하여 FreeSplat은 세 가지 핵심 기여를 제안한다:

| 기여 | 모듈명 |
|------|--------|
| 효율적인 다중 뷰 특징 집계 | **Low-cost Cross-View Aggregation (LCVA)** |
| 중복 Gaussian 제거 및 융합 | **Pixel-wise Triplet Fusion (PTF)** |
| 임의 뷰 수에 대한 강인한 학습 | **Free-View Training (FVT)** |

FreeSplat의 핵심 목표는 다중 뷰 픽셀 정렬 3D Gaussian을 점진적으로 융합하여 대규모 장면의 사실적인 재구성을 위한 일반화 가능한 3DGS 프레임워크를 제안하는 것이다.

---

## 2. 상세 설명

### 2-1. 해결하고자 하는 문제

**① 좁은 시야 범위의 보간 문제**

3DGS에 일반화 능력을 부여하려는 다양한 시도들이 있었으나, 이 방법들은 좁은 범위의 씬 수준 뷰 보간 및 객체 중심 합성에 국한되어 있다. 그 근본적인 이유는 기존 방법들이 다중 뷰 이미지 간의 조밀한 뷰 매칭에 트랜스포머를 사용하여 Gaussian 프리미티브를 예측하기 때문인데, 이는 긴 시퀀스에서 계산적으로 불가능해지며 좁은 범위의 보간 뷰에만 supervision을 제한한다.

**② 잘못 위치화된 Gaussian (Floater) 문제**

그림 3에서 보여주듯이, 좁은 범위의 보간 뷰에 의한 supervision은 종종 poorly localized 3D Gaussian을 만들어내고, 이는 외삽 뷰에서 렌더링될 때 floater가 된다. 또한 기존 방법들이 다중 뷰 3D Gaussian을 단순 concatenation으로 병합하여 중복 영역에서 눈에 띄는 중복 문제가 발생한다.

**③ 긴 시퀀스 처리 불가 문제**

기존 일반화 가능한 3DGS 방법들은 무거운 백본으로 인해 좁은 범위 보간에 국한되어 정확한 3D Gaussian 위치화에 한계가 있다. 이 방법들은 넓은 시점 범위에서의 자유 시점 합성에 어려움을 겪으며 긴 시퀀스에서는 계산적으로 다루기 어렵다. 현재 방법들은 또한 단순 concatenation을 통해 다중 뷰 Gaussian을 병합하여 중복 영역에서 상당한 중복을 야기한다.

---

### 2-2. 제안하는 방법 및 수식

#### 전체 파이프라인 개요

입력 희소 이미지 시퀀스가 주어지면, 인접 뷰 간에 cost volume을 구성하고 depth map 및 대응하는 feature map을 예측한 후, 3D 위치를 가진 Gaussian triplets으로 unprojection한다. 그런 다음 Pixel-aligned Triplet Fusion(PTF) 모듈을 통해 픽셀 단위 정렬을 기반으로 local/global Gaussian triplets을 점진적으로 집계하고 업데이트한다. 글로벌 Gaussian triplets은 이후 Gaussian 파라미터로 디코딩된다.

---

#### ① 3D Gaussian Splatting 기본 수식

각 3D Gaussian은 다음 파라미터로 구성된다:

$$\mathcal{G} = \{\boldsymbol{\mu}, \boldsymbol{\Sigma}, \boldsymbol{\alpha}, \mathbf{c}\}$$

- $\boldsymbol{\mu} \in \mathbb{R}^3$: 3D 위치 (mean)
- $\boldsymbol{\Sigma} \in \mathbb{R}^{3\times3}$: 공분산 행렬 (covariance)
- $\boldsymbol{\alpha}$: 불투명도 (opacity)
- $\mathbf{c}$: 색상 (color, SH 계수)

카메라 보정에서 생성된 희소 포인트에서 시작하여, 씬을 3D Gaussian으로 표현하며, 공분산의 비등방성 최적화를 포함하여 씬의 정확한 표현을 달성한다.

2D 투영 시 Gaussian의 공분산은 다음과 같이 변환된다:

$$\boldsymbol{\Sigma}' = \mathbf{J} \mathbf{W} \boldsymbol{\Sigma} \mathbf{W}^T \mathbf{J}^T$$

여기서 $\mathbf{W}$는 카메라 변환 행렬, $\mathbf{J}$는 투영 변환의 야코비안이다.

볼륨 렌더링은 다음과 같이 수행된다:

$$\mathbf{C} = \sum_{i=1}^{N} \mathbf{c}_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)$$

---

#### ② Low-cost Cross-View Aggregation (LCVA)

FreeSplat은 특징 추출을 위한 저비용 2D 백본과 다중 뷰 집계를 위한 cost volume을 활용하는 일반화 가능한 3DGS 방법이다.

인접 뷰 $i$와 $j$ 사이의 cost volume $\mathbf{C}_{ij}$는 각 깊이 가설 $d$에 대해 다음과 같이 정의된다:

$$\mathbf{C}_{ij}(d) = \frac{1}{K}\sum_{k=1}^{K} \langle \mathbf{F}_i^k, \mathbf{F}_{j \to i}^k(d) \rangle$$

- $\mathbf{F}_i^k$: 뷰 $i$의 $k$번째 스케일 특징
- $\mathbf{F}_{j \to i}^k(d)$: 깊이 $d$로 뷰 $j$에서 뷰 $i$로 워핑된 특징
- $\langle \cdot, \cdot \rangle$: 내적 (dot product)

이를 통해 인접 뷰 간의 적응적 cost volume을 구성하고 멀티스케일 구조로 특징을 집계하는 Low-cost Cross-View Aggregation을 소개한다.

깊이 예측은 소프트맥스로 가중합된다:

$$\hat{d}_i = \sum_{d} d \cdot \text{softmax}(\mathbf{C}_{ij}(d))$$

픽셀 $p$의 3D 위치는 역투영(unprojection)으로 계산된다:

$$\boldsymbol{\mu}_p = \hat{d}_p \cdot \mathbf{K}^{-1} \tilde{\mathbf{p}}$$

여기서 $\mathbf{K}$는 카메라 내부 파라미터 행렬, $\tilde{\mathbf{p}} = [u, v, 1]^T$는 픽셀의 동차 좌표이다.

---

#### ③ Pixel-wise Triplet Fusion (PTF)

Pixel-wise Triplet Fusion은 중첩 뷰 영역에서 3D Gaussian의 중복을 제거하고 다중 뷰에서 관찰된 특징을 집계한다.

각 픽셀 위치에서 예측된 Gaussian을 "triplet" $(\boldsymbol{\mu}, \mathbf{f}, w)$로 표현한다:
- $\boldsymbol{\mu}$: 3D 위치
- $\mathbf{f}$: 잠재 특징 벡터
- $w$: 가중치

뷰 $i$의 Gaussian triplet을 새 뷰 $j$의 이미지 평면에 투영한 후, 픽셀 정렬로 다음과 같이 융합한다:

$$\mathbf{f}^{global} = \frac{\sum_{i} w_i \cdot \mathbf{f}_i^{local}}{\sum_{i} w_i}$$

PTF 모듈은 Gaussian 중복을 약 55.0% 감소시키며, 이는 긴 시퀀스 재구성에 매우 중요하다.

---

#### ④ Free-View Training (FVT) 전략

임의의 뷰 수에 관계없이 더 넓은 시점 범위에서 강인한 뷰 합성을 보장하는 간단하지만 효과적인 free-view 학습 전략을 제안한다.

학습 손실은 광도 측정 손실(Photometric Loss)을 사용한다:

$$\mathcal{L} = \lambda_1 \mathcal{L}_{MSE} + \lambda_2 \mathcal{L}_{LPIPS}$$

$$\mathcal{L}_{MSE} = \frac{1}{|\mathcal{V}_{target}|} \sum_{v \in \mathcal{V}_{target}} \| \hat{I}_v - I_v \|_2^2$$

- $\hat{I}_v$: 렌더링된 이미지
- $I_v$: 실제 GT 이미지
- $\mathcal{V}_{target}$: FVT 전략으로 선택된 다양한 시점 범위의 타깃 뷰 집합

특정 입력 뷰 수로 학습하면 다른 뷰 수로 추론할 때 성능이 눈에 띄게 감소한다. 하지만 free-view 학습 전략은 이러한 상관관계를 완화하여 임의의 수의 입력 뷰에 적합할 수 있게 한다.

---

### 2-3. 모델 구조

```
입력: 희소 RGB 이미지 시퀀스 {I_1, ..., I_N}
  │
  ├─ [2D CNN Backbone] Feature Extraction
  │   └─ 멀티스케일 특징 맵 {F_i^k}
  │
  ├─ [LCVA] Low-cost Cross-View Aggregation
  │   ├─ 인접 뷰 간 Adaptive Cost Volume 구성
  │   ├─ 멀티스케일 특징 집계
  │   └─ 깊이 맵 예측 → Gaussian Triplet Unprojection
  │
  ├─ [PTF] Pixel-wise Triplet Fusion
  │   ├─ Local Gaussian Triplet → 다음 뷰에 투영
  │   ├─ 픽셀 단위 정렬 기반 점진적 융합
  │   └─ Global Gaussian Triplet 생성
  │
  ├─ [Decoder] Gaussian Parameter Prediction
  │   └─ {μ, Σ, α, c} 예측
  │
  └─ [3DGS Rasterizer] Novel View Rendering
```

FreeSplat은 임의의 수의 입력 뷰를 수용하고 글로벌 3D Gaussian을 사용하여 자유 시점 합성을 수행하도록 설계된 일반화 가능한 3DGS 모델이다. 긴 입력 시퀀스를 효율적으로 처리하는 능력을 향상시키는 Low-cost Cross-View Aggregation 파이프라인을 개발하였다. 또한 중복 픽셀 정렬 3D Gaussian을 효과적으로 줄이고 다중 뷰 Gaussian 잠재 특징을 병합하는 Pixel-wise Triplet Fusion 모듈을 고안하였다.

---

### 2-4. 성능 향상

FreeSplat-spec은 2-views 및 3-views 설정에서 pixelSplat 및 MVSplat과 비교하여 렌더링 품질과 효율성을 지속적으로 향상시킨다. NeuRay와 비교하면 SSIM에서 약간 낮지만, PSNR 및 LPIPS에서 상당한 향상을 보이며 300배 빠른 추론 속도를 달성한다.

긴 시퀀스(10 뷰)에서 FreeSplat-3views는 뷰 보간과 외삽 모두에서 pixelSplat 및 MVSplat을 크게 능가한다. FVT 전략은 긴 시퀀스에 대한 FreeSplat-fv의 성능을 더욱 향상시킨다.

FreeSplat은 pixelSplat 및 MVSplat보다 훨씬 우수한 정확한 위치화된 3D Gaussian 예측에서 27.0%까지 도달하며, 이는 새로운 뷰에서 비지도 깊이 추정을 가능하게 한다. FreeSplat-fv의 향상된 깊이 추정 정확도는 더 넓은 시점 범위에서 자유 시점 합성을 지원하는 깊이 추정의 중요성을 강조한다.

---

### 2-5. 한계점

Replica 데이터셋에서의 zero-shot 전이 결과를 평가한 결과, 뷰 보간 및 새로운 뷰 깊이 추정 결과에서는 기존 방법을 능가했다. 하지만 긴 시퀀스 결과는 부정확한 깊이 추정과 도메인 갭으로 인해 성능이 저하되었으며, 이는 zero-shot 전이에서의 깊이 추정 개선이 미래 연구 과제임을 시사한다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. Free-View Training(FVT)을 통한 뷰 수 일반화

free-view 학습 전략은 뷰 수와의 상관관계를 완화하여 임의의 수의 입력 뷰에 적합하게 한다. 10-views 설정에서도 FreeSplat-fv는 FreeSplat-8views와 유사한 성능을 발휘하며, 다양한 수의 입력 뷰를 사용하는 것이 긴 시퀀스 재구성 결과에 해를 끼치지 않음을 보여준다.

### 3-2. Cost Volume 기반 기하학적 정확도

Cost volume은 3D Gaussian을 정확히 위치화하는 데 필수적이며, PTF 모듈은 Gaussian의 사후 집계를 통해 렌더링 품질과 깊이 추정 결과에 지속적으로 기여한다.

이 결과는 뷰 보간 작업에서 실내 데이터셋 전반에 걸친 FreeSplat의 일반화 능력을 나타낸다.

### 3-3. Depth Prior 없는 Feed-forward 재구성

FreeSplat은 깊이 prior 없이 feed-forward 글로벌 Gaussian 재구성을 가능하게 하면서 색상 이미지 품질과 깊이 맵 정확도 모두에서 새로운 뷰 렌더링의 충실도를 지속적으로 향상시킨다.

### 3-4. FreeSplat++로의 발전 가능성

이 연구는 대규모 재구성을 위한 일반화 가능한 3DGS에서 선구적인 노력을 나타낸다. 알려진 한, 실내 전체 씬 재구성에 일반화 가능한 3DGS를 효과적으로 적용한 최초의 프레임워크이다.

FreeSplat의 전체 씬 재구성 중 한계를 해결하기 위해, FreeSplat++는 foreground floater를 줄이기 위한 융합 모듈을 개선하고 cross-view floater를 효율적으로 제거하는 효과적인 floater 제거 모듈을 설계하였다. 또한 unprojection을 위한 저해상도 맵을 사용하여 효율성을 향상시키고, feed-forward 3DGS 초기화를 유지하면서 기하학적 정확도를 보존하는 depth-regularized 파인튜닝 프로세스를 설계하였다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 학회/연도 | 핵심 기법 | 뷰 범위 | 실내 씬 지원 | 깊이 Prior |
|------|---------|---------|--------|------------|----------|
| NeRF (Mildenhall et al.) | ECCV 2020 | 암묵적 복사장 | 제한적 | ✓ | ✗ |
| pixelSplat | CVPR 2024 | Epipolar Transformer | 좁은 범위 | △ | ✗ |
| MVSplat | ECCV 2024 | Plane-sweep Cost Volume | 좁은 범위 | △ | ✗ |
| GPS-Gaussian | CVPR 2024 | 픽셀 정렬 Gaussian | 좁은 범위 | ✗ (인체) | ✓ |
| **FreeSplat** | **NeurIPS 2024** | **LCVA + PTF + FVT** | **넓은 범위** | **✓** | **✗** |
| FreeSplat++ | arXiv 2025 | FreeSplat + Floater 제거 | 전체 씬 | ✓✓ | ✗ |

pixelSplat과 MVSplat은 기하학적으로 일관된 글로벌 3D Gaussian 재구성에 실패하는 반면, FreeSplat은 긴 시퀀스 입력에서 3D Gaussian을 정확히 위치화하고 자유 시점 합성을 지원하도록 제안되었다.

MVSplat은 최신 pixelSplat 대비 10배 적은 파라미터와 2배 이상 빠른 추론 속도로 더 높은 외관 및 기하 품질과 더 나은 크로스 데이터셋 일반화를 제공한다.

NeRF의 주요 단점은 수많은 3D 포인트의 계산 집약적인 레이 기반 볼륨 렌더링으로 인한 느린 렌더링 속도이다.

FreeSplat++는 일반화 가능한 3DGS를 대규모 실내 전체 씬 재구성의 대안적 접근법으로 확장하는 데 초점을 맞추고 있으며, 재구성 속도를 크게 가속화하고 기하학적 정확도를 향상시킬 가능성이 있다.

---

## 5. 향후 연구에 미치는 영향 및 고려 사항

### 5-1. 연구에 미치는 영향

**① 대규모 씬 재구성의 새로운 패러다임 제시**

FreeSplat은 더 효율적으로 추론하고 중복 Gaussian을 효과적으로 줄이며, 깊이 prior 없이 feed-forward 대규모 씬 재구성의 가능성을 제시한다.

**② 실용적 응용 범위 확대**

3DGS의 효율적이고 제어 가능한 명시적 표현의 특성으로 인해, 그 응용은 VR/AR의 몰입형 환경 향상, 로봇 공학 및 자율 시스템에서의 공간 인식 개선, 도시 계획 및 건축 등 다양한 분야로 확장된다.

**③ FreeSplat++ 후속 연구로의 직접적 기여**

FreeSplat++는 전체 씬 재구성 특히 기존 일반화 가능한 3DGS 방법을 크게 능가하며, per-scene 파인튜닝 결과는 재구성 정확도의 실질적인 향상과 기존 per-scene 최적화 3DGS 대비 훈련 시간의 현저한 감소를 보여준다.

### 5-2. 향후 연구 시 고려할 점

**① 도메인 갭 및 Zero-shot 전이 개선**
Replica 데이터셋에서의 뷰 보간 및 새로운 뷰 깊이 추정 결과는 강한 일반화 능력을 보이지만, 긴 시퀀스 zero-shot 전이에서의 성능은 부정확한 깊이 추정과 도메인 갭으로 인해 저하되며, 이 영역에서 추가 연구가 필요하다.

**② 야외·비정형 씬 확장성**
현재 FreeSplat은 실내 씬 중심으로 설계되어 있어, 야외 대규모 씬이나 동적 씬에 대한 일반화 방법론이 추가로 연구되어야 한다. FVT 방식과 PTF 모듈의 동적 객체에 대한 확장이 필요하다.

**③ Floater 문제의 근원적 해결**
FreeSplat의 전체 씬 재구성에서의 도전 과제를 해결하기 위해 fusion 메커니즘을 개선하고 floater를 추가로 제거하는 가중 floater 제거 전략을 제안한다. 이는 아직 완전히 해결되지 않은 과제로, 예측된 깊이 맵의 노이즈 억제를 위한 더 강인한 방법이 필요하다.

**④ 실시간 및 메모리 효율성**
긴 시퀀스에서의 메모리 효율 최적화, 특히 모바일 기기 및 엣지 컴퓨팅 환경에서의 경량화가 실용화를 위한 핵심 과제이다.

**⑤ 동적 씬 및 시간적 일관성**
최근 3DGS에 효율적인 feed-forward 방식의 통합이 활발히 탐구되고 있으나, 대부분의 방법은 작은 영역의 희소 뷰 재구성에 집중되어 있어 품질과 효율성 측면에서 전체 씬 재구성 결과를 도출하지 못하고 있다. 동적 요소가 포함된 씬에서의 시간적 일관성 유지가 중요한 차기 연구 방향이다.

---

## 참고 자료

1. **FreeSplat 논문 (arXiv)**: Wang, Y., Huang, T., Chen, H., Lee, G.H. "FreeSplat: Generalizable 3D Gaussian Splatting Towards Free-View Synthesis of Indoor Scenes." *NeurIPS 2024*. https://arxiv.org/abs/2405.17958
2. **FreeSplat 공식 프로젝트 페이지**: https://wangys16.github.io/FreeSplat-project/
3. **FreeSplat 공식 GitHub 코드**: https://github.com/wangys16/FreeSplat
4. **NeurIPS 2024 포스터 페이지**: https://nips.cc/virtual/2024/poster/93734
5. **OpenReview (NeurIPS 2024)**: https://openreview.net/forum?id=ml01XyP698
6. **NeurIPS 2024 공식 논문 PDF**: https://proceedings.neurips.cc/paper_files/paper/2024/file/c2166d01fe4bcd694aba89f608737678-Paper-Conference.pdf
7. **FreeSplat++ (후속 연구, arXiv 2025)**: Wang et al. "FreeSplat++: Generalizable 3D Gaussian Splatting for Efficient Indoor Scene Reconstruction." https://arxiv.org/abs/2503.22986
8. **pixelSplat (CVPR 2024)**: Charatan et al. "pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction." https://github.com/dcharatan/pixelsplat
9. **MVSplat (ECCV 2024)**: Chen et al. "MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images." https://donydchen.github.io/mvsplat/
10. **3DGS 원본 (SIGGRAPH 2023)**: Kerbl et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering." https://arxiv.org/abs/2308.04079
11. **3DGS Survey**: "3D Gaussian Splatting: Survey, Technologies, Challenges, and Opportunities." https://arxiv.org/abs/2407.17418
12. **eFreeSplat (epipolar-free 변형)**: "Epipolar-Free 3D Gaussian Splatting for Generalizable Novel View Synthesis." https://arxiv.org/abs/2410.22817
