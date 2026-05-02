
# DUSt3R: Geometric 3D Vision Made Easy

> **논문 정보**
> - **제목:** DUSt3R: Geometric 3D Vision Made Easy
> - **저자:** Shuzhe Wang (Aalto University), Vincent Leroy, Yohann Cabon, Boris Chidlovskii, Jerome Revaud (NAVER Labs Europe)
> - **게재:** CVPR 2024 (pp. 20697–20709)
> - **arXiv:** [arXiv:2312.14132](https://arxiv.org/abs/2312.14132)
> - **공식 코드:** [github.com/naver/dust3r](https://github.com/naver/dust3r)

---

## 1. 핵심 주장 및 주요 기여 (Executive Summary)

DUSt3R는 임의의 이미지 컬렉션에 대해 카메라 보정(calibration) 정보나 시점(viewpoint) 포즈 없이 동작하는 **Dense and Unconstrained Stereo 3D Reconstruction**의 완전히 새로운 패러다임을 제시합니다.

이 모델의 출력인 3D 포인트맵(pointmap)은 풍부한 속성을 동시에 내포하며, (a) 장면의 기하 구조, (b) 픽셀과 장면 포인트 간의 관계, (c) 두 시점 간의 관계를 한꺼번에 표현합니다.

### 주요 기여 4가지

| # | 기여 내용 |
|---|----------|
| 1 | 카메라 파라미터 없이 동작하는 새로운 MVS 패러다임 제시 |
| 2 | 포인트맵(pointmap) 표현 방식 도입 |
| 3 | 다중 뷰를 위한 글로벌 정렬(global alignment) 최적화 절차 도입 |
| 4 | 단안/다중 뷰 깊이 추정 및 포즈 추정에서 SoTA 달성 |

포인트맵 표현은 네트워크가 정규 프레임(canonical frame) 내에서 3D 형상을 예측하면서 픽셀과 장면 간의 암묵적 관계를 보존할 수 있게 하며, 이는 일반적인 투영 카메라 모델의 많은 제약 조건들을 제거합니다.

---

## 2. 해결하고자 하는 문제

### 2-1. 기존 방법의 한계

기존의 Multi-view Stereo(MVS) 복원은 카메라의 내부/외부 파라미터(intrinsic/extrinsic)를 먼저 추정해야 하는데, 이는 매우 번거롭고 까다로운 작업임에도 3D 공간에서 픽셀을 삼각 측량(triangulation)하기 위해 반드시 필요한 요소입니다.

전통적인 SfM 파이프라인은 여러 이미지 간의 키포인트 매칭(keypoint matching)으로부터 얻은 픽셀 대응 관계를 시작으로 카메라 파라미터와 3D 좌표를 공동 최적화하는 번들 조정(bundle adjustment)으로 이어지는 복잡한 과정을 거칩니다.

### 2-2. DUSt3R의 핵심 관점 전환

DUSt3R는 이미지 내용으로부터 카메라 파라미터를 직접 추론하는 방식으로, 명시적인 카메라 보정의 필요성을 완전히 제거합니다. 포인트맵에 대한 회귀(regression) 접근 방식을 채택하여 다중 뷰 깊이 추정을 위한 풍부한 기하 세부 정보를 디코딩합니다.

---

## 3. 제안하는 방법 (수식 포함)

### 3-1. 포인트맵(Pointmap) 정의

입력 이미지 $I_1, I_2 \in \mathbb{R}^{W \times H \times 3}$가 주어졌을 때, 각 이미지에 대해 **포인트맵** $X \in \mathbb{R}^{W \times H \times 3}$을 예측합니다.

- $X_1^{1,1} \in \mathbb{R}^{W \times H \times 3}$: 이미지 $I_1$의 각 픽셀이 뷰 1의 좌표계에서 대응하는 3D 점
- $X_2^{2,1} \in \mathbb{R}^{W \times H \times 3}$: 이미지 $I_2$의 각 픽셀이 뷰 1의 좌표계에서 대응하는 3D 점

이를 간략하게 아래와 같이 표현할 수 있습니다.

$$
(X_1^{1,1},\ C_1^{1,1}),\ (X_2^{2,1},\ C_2^{2,1}) = f_\theta(I_1, I_2)
$$

여기서 $C \in \mathbb{R}^{W \times H}$는 각 포인트에 대한 **신뢰도 맵(confidence map)**입니다.

### 3-2. 학습 목표(Training Objective)

회귀 손실은 **신뢰도 가중 3D 회귀 손실**로 정의됩니다. 예측된 포인트맵 $\hat{X}$와 정답 포인트맵 $X^*$ 사이의 손실은 다음과 같이 표현됩니다.

```math
\mathcal{L} = \sum_{(i,j)} C_{ij} \cdot \left\| \frac{\hat{X}_{ij}}{\|\hat{X}_{ij}\|} - \frac{X^*_{ij}}{\|X^*_{ij}\|} \right\| - \alpha \log C_{ij}
```

- 분모의 정규화는 **스케일 불변성(scale invariance)**을 부여합니다.
- $-\alpha \log C_{ij}$ 항은 신뢰도 값이 0으로 퇴화하는 것을 방지하는 정규화 역할을 합니다.
- 네트워크는 불확실한 영역에 낮은 신뢰도를 할당하는 것을 학습합니다.

### 3-3. 카메라 파라미터 복원

포인트맵으로부터 카메라 내부 파라미터(초점 거리 $f$, 주점 $(c_x, c_y)$ )를 다음과 같이 닫힌 형태(closed-form)로 복원합니다.

$$
f = \frac{W}{2\tan(\theta/2)}, \quad P = K^{-1} \cdot p_\text{pixel}
$$

여기서 $K$는 내부 행렬, $p_\text{pixel}$은 픽셀 좌표이며, 포인트맵의 3D 구조로부터 직접 추정됩니다.

### 3-4. 글로벌 정렬(Global Alignment)

기존 SfM과 달리 DUSt3R는 2D 재투영 오차를 최소화하는 번들 조정(BA)을 사용하지 않고, 3D 공간에서 직접 오차를 최소화하는 방식으로 카메라 포즈와 장면 기하 구조를 조정하는 새로운 글로벌 정렬 전략을 도입합니다.

$N$개의 이미지가 주어질 때 글로벌 정렬 최적화 목표는:

$$
\min_{\{P_k\},\ \{s_{ij}\}} \sum_{(i,j) \in \mathcal{E}} \sum_{v \in \{i,j\}} \sum_{p} C_{vp}^{ij} \cdot \left\| P_k(p) - s_{ij} \cdot R_k X_{vp}^{ij} - t_k \right\|^2
$$

- $P_k$: 각 뷰 $k$의 월드 좌표 포인트맵
- $s_{ij}$: 페어 $(i,j)$의 스케일 팩터
- $R_k, t_k$: 카메라 회전 및 이동
- $\mathcal{E}$: 이미지 쌍의 그래프(scene graph)

쌍별 상대 포즈 추정을 계산하고 최적의 매칭 쌍을 신뢰도 점수를 기준으로 선택한 후, 모든 쌍별 포인트맵을 반복적으로 공통 기준 프레임에 정렬하는 방식으로, 이는 BA와 유사하지만 더 빠르며 적은 수의 이미지에서 빠르게 수렴합니다.

---

## 4. 모델 구조 (Architecture)

DUSt3R의 네트워크 구조는 두 입력 이미지 $I_1, I_2 \in \mathbb{R}^{W \times H \times 3}$를 패치(patch)로 분할한 뒤 공유 가중치(shared weights)를 가진 ViT(Vision Transformer) 인코더로 처리하고, 이후 두 개의 별도 Transformer 디코더(Decoder1, Decoder2)를 통해 포인트맵 $X_1$, $X_2$와 신뢰도 $C_1$, $C_2$를 출력합니다.

```
[Image I₁] ──► [ViT Encoder (공유)] ──► [Transformer Decoder 1] ──► Head1 ──► (X₁,₁, C₁)
                        ↕ (Information Exchange)
[Image I₂] ──► [ViT Encoder (공유)] ──► [Transformer Decoder 2] ──► Head2 ──► (X₂,₁, C₂)
```

DUSt3R의 아키텍처는 CroCo v1과 유사한 구조이나, 공유 디코더 대신 두 개의 분리된 디코더를 사용하는 **비대칭 설계(asymmetric design)**를 채택합니다.

디코더는 어텐션(attention), 크로스 어텐션(cross-attention), MLP 레이어를 포함한 12개의 CrossBlock으로 구성되며, 두 디코더 간에 지속적으로 정보가 공유되어 모델이 적절히 정렬된 포인트맵을 출력하도록 유도합니다.

DUSt3R와 MASt3R 모두 동일 팀이 개발한 사전 학습 전략인 **CroCo(Cross-View Completion)**를 기반으로 하며, CroCo는 이 모델들의 강력한 기초 아키텍처를 가능하게 하는 핵심 요소입니다.

### 디코더 세부 사양 (ViT-Large 기준)

| 구성 요소 | 세부 사항 |
|---|---|
| 인코더 | ViT-Large (출력 차원 1024) |
| 디코더 임베딩 | Linear(1024 → 768) |
| 디코더 블록 | 12 × CrossBlock (Attention + CrossAttention + MLP) |
| 출력 헤드 | DPT (Dense Prediction Transformer) |
| 최종 출력 채널 | 4채널 (xyz + confidence) per pixel |

---

## 5. 성능 향상

DUSt3R는 3D 장면 모델과 깊이 정보를 직접 제공할 뿐 아니라 픽셀 매칭, 상대/절대 카메라 포즈까지 복원할 수 있으며, 단안/다중 뷰 깊이 추정과 상대 포즈 추정에서 새로운 SoTA를 달성합니다.

글로벌 정렬을 적용한 DUSt3R는 두 데이터셋에서 최고 성능을 달성하며 SoTA인 PoseDiffusion을 크게 능가하고, PnP를 사용한 DUSt3R 역시 기존 학습 기반 및 구조 기반 방법 대비 우월한 성능을 보입니다.

아키텍처는 DTU, Tanks and Temples, ETH-3D 등의 벤치마크에서 단일 파이프라인으로 다양한 3D 비전 태스크를 통합하면서 SoTA 성능을 달성합니다.

### 벤치마크 요약

| 태스크 | 데이터셋 | 성과 |
|---|---|---|
| 단안 깊이 추정 | NYUv2, KITTI, ScanNet | SoTA (Zero-shot transfer) |
| 다중 뷰 깊이 추정 | DTU, Tanks&Temples, ETH-3D | SoTA |
| 상대 포즈 추정 | Co3Dv2, RealEstate10K | SoTA (PoseDiffusion 압도) |
| 밀집 3D 복원 | MegaDepth, CO3Dv2 | SoTA |

---

## 6. 한계

| 한계 | 설명 |
|---|---|
| **스케일 모호성** | 출력 포인트맵은 미지의 스케일까지만 회귀되며, 즉 깊이는 상대적이지 절대적(metric)이지 않습니다. |
| **이미지 해상도 제한** | DUSt3R와 MASt3R는 Transformer 아키텍처 기반으로, 2025년 기준 주류 GPU에서 최대 512픽셀 해상도의 이미지로 제한됩니다. |
| **쌍별 처리의 이차 복잡도** | DUSt3R는 이미지 쌍을 처리하여 로컬 3D 복원을 회귀하는 구조로, 이미지 쌍의 수가 이차적(quadratically)으로 증가하며 이는 대규모 이미지 컬렉션에서 강건하고 빠른 최적화에 있어 근본적인 한계입니다. |
| **동적 장면 처리 미흡** | 정적 장면을 가정하기 때문에 움직이는 객체가 있는 동적 환경에서는 성능이 저하됩니다. |
| **텍스처 없는 영역** | DUSt3R/MASt3R/VGGT가 처리하는 관점 변화가 적고 텍스처가 없는 영역(예: 항공 촬영 대규모 장면)에서는 성능이 제한될 수 있습니다. |

---

## 7. 일반화 성능 향상 가능성

DUSt3R의 일반화 능력은 여러 측면에서 확인되고 향상 가능성이 있습니다.

### 7-1. 현재의 일반화 강점

DUSt3R의 가장 중요한 특징은 전통적으로 별도로 처리되던 다양한 3D 비전 태스크를 하나의 단순화된 파이프라인으로 통합하는 능력이며, 강력한 기하학적·형상적 사전 지식을 학습하는 완전 데이터 기반 접근 방식을 활용합니다.

논문이 시각화한 장면들은 학습 중 한 번도 본 적 없는 장면들이며 임의로 선택된 것으로, 일반화 능력을 보여줍니다.

DUSt3R는 이미지 간 시각적 내용이 거의 겹치지 않는 극단적인 경우(예: stop sign, motorcycle)에서도 뚜렷한 문제 없이 극적인 시점 변화를 처리합니다.

### 7-2. 일반화 성능 향상을 위한 방향

DUSt3R, MASt3R, VGGT와 같은 파운데이션 모델들은 매우 희박한 이미지 중첩을 처리하고 다양한 장면에 일반화하는 능력으로 주목받고 있습니다.

**다양한 데이터셋 학습:** 모델을 더 다양한 환경(실내/야외, 항공/지상, 동적/정적)의 데이터로 훈련하면 새로운 도메인에 대한 제로샷(zero-shot) 일반화 성능이 향상될 수 있습니다.

**사전 지식 통합 (Pow3R):** 기존 DUSt3R와 MASt3R의 단점 중 하나는 이미지만 입력으로 받는다는 것인데, Pow3R는 카메라 내부 파라미터, 카메라 포즈, 깊이 센서(LIDAR)의 밀집/희박 깊이 데이터 등 보조 정보를 함께 입력으로 받을 수 있어 일반화 성능을 높입니다.

**도메인 파인튜닝:** AerialMegaDepth 데이터셋으로 파인튜닝된 DUSt3R는 원본 버전보다 강력한 성능을 보이며, 이처럼 도메인 특화 파인튜닝이 일반화 성능 향상에 기여합니다.

---

## 8. 최신 후속 연구 비교 분석 (2020년 이후)

### 8-1. DUSt3R 계보 연구

DUSt3R가 3D 비전 태스크를 통합하는 새로운 패러다임을 도입한 데 이어, **MASt3R**는 매칭 및 메트릭 포인트맵을 위한 추가 헤드(head)와 더 확장 가능한 글로벌 정렬을 통해 이를 확장했습니다.

| 모델 | 기여 | 게재 |
|---|---|---|
| **CroCo** | DUSt3R의 사전학습 기반, 교차 뷰 완성 방식 | NeurIPS 2022 |
| **DUSt3R** | 포인트맵 회귀 기반 통합 3D 비전 | CVPR 2024 |
| **MASt3R** | 밀집 로컬 특징 헤드 추가, 매트릭 포인트맵, 향상된 매칭 | ECCV 2024 |
| **MASt3R-SfM** | 완전 통합 비제약 SfM 솔루션 | 3DV 2025 |
| **MUSt3R** | 다중 뷰 직접 처리, 대칭 아키텍처, 대규모 컬렉션 | CVPR 2025 |
| **MV-DUSt3R+** | 단일 스테이지 다중 뷰 재구성, 2초 내 처리 | CVPR 2025 Oral |
| **Pow3R** | 카메라/깊이 사전 지식 통합, 고해상도 재구성 | CVPR 2025 |

MASt3R는 밀집 로컬 특징을 출력하는 새로운 헤드를 DUSt3R 네트워크에 추가하여 추가적인 매칭 손실로 훈련됩니다. 다만 DUSt3R는 내부적으로 이미지 쌍을 처리하여 글로벌 좌표계에 정렬이 필요한 로컬 3D 복원을 회귀하는 구조이며, 이미지 쌍의 수가 이차적으로 증가하는 것이 대규모 이미지 컬렉션 처리에서 본질적인 한계가 됩니다.

MUSt3R는 DUSt3R 아키텍처를 대칭화(symmetric)하고 모든 뷰에 대해 공통 좌표계에서 직접 3D 구조를 예측하도록 확장합니다.

MV-DUSt3R는 단일 스테이지 피드포워드 네트워크로, 임의 수의 뷰에 걸쳐 하나의 기준 뷰를 고려하면서 정보를 교환하는 다중 뷰 디코더 블록을 핵심으로 사용하며, MV-DUSt3R+는 서로 다른 기준 뷰 선택 전반에 걸쳐 정보를 융합하는 크로스-기준뷰 블록을 추가합니다.

VGGT는 DUSt3R가 사용하는 비용이 많이 드는 반복적 후처리 최적화를 제거하는 피드포워드 신경망을 통해 파이프라인을 더욱 발전시키며, 결과적으로 속도와 품질 모두에서 DUSt3R와 MASt3R를 능가할 수 있습니다.

### 8-2. 응용 확장 연구

DUSt3R 기반으로 Splatt3R(비보정 이미지 쌍으로부터의 3D Gaussian Splatting), SLAM3R(RGB 비디오의 실시간 밀집 장면 재구성), Align3R(동적 비디오의 단안 깊이 추정) 등 다양한 다운스트림 응용 연구들이 등장했습니다.

---

## 9. 앞으로의 연구에 미치는 영향 및 고려할 점

### 9-1. 연구에 미치는 영향

DUSt3R는 3D 기하 비전 분야의 주요 발전을 대표하며, 카메라 파라미터를 추정하고 보정하는 번거롭고 세밀한 단계 없이 다양한 3D 비전 과제를 처리하는 기존 방법에 비해 상당한 단순화를 달성함으로써 향후 발전 가능성을 보여주었습니다.

1. **파운데이션 모델 패러다임 확산:** DUSt3R의 핵심인 다양한 3D 비전 태스크를 하나의 단순화된 파이프라인으로 통합하는 능력은 3D 비전 분야의 파운데이션 모델 연구를 촉진시킵니다.

2. **3D 표현의 재정의:** 포인트맵이라는 새로운 3D 표현 방식은 Gaussian Splatting, NeRF 등 다른 3D 표현 방식과의 융합 연구를 촉진합니다.

3. **엔드투엔드 SfM/MVS 대체:** 포인트맵의 글로벌 정렬 최적화 절차는 기존 SfM 및 MVS 파이프라인의 통상적인 중간 출력물 모두를 손쉽게 추출할 수 있으며, 이는 전통적인 두 단계 파이프라인의 대체를 가속화합니다.

### 9-2. 앞으로 연구 시 고려할 점

| 고려 사항 | 상세 설명 |
|---|---|
| **절대 스케일(Metric Depth)** | 현재 출력이 상대적 깊이이므로, 절대 스케일 복원이나 LiDAR 등과의 융합 연구가 필요합니다. |
| **동적 장면 처리** | 움직이는 물체가 포함된 동적 장면에서의 포인트맵 분리 연구(예: D²USt3R, Easi3R)가 요구됩니다. |
| **고해상도 처리** | 현재 512픽셀 해상도 한계를 극복하기 위한 효율적인 고해상도 처리 방법 연구가 필요합니다. |
| **이차 복잡도 해소** | 이미지 쌍의 수가 이차적으로 증가하는 것이 글로벌 정렬에서 중대한 과제이므로, 이를 해소하는 효율적 알고리즘 연구가 중요합니다. |
| **도메인 일반화** | 항공 촬영, 의료 이미징 등 특수 도메인으로의 파인튜닝 및 전이 학습 연구가 유망합니다. |
| **실시간 처리** | SLAM, 자율주행 등 실시간 응용을 위한 경량화 연구(예: SLAM3R, Fast3R)가 필요합니다. |
| **불확실성 정량화** | 신뢰도 맵을 넘어선 더 정밀한 불확실성 추정으로 안전-크리티컬 응용을 지원해야 합니다. |

---

## 참고 자료

1. **Wang, S. et al.** "DUSt3R: Geometric 3D Vision Made Easy," *CVPR 2024*, pp. 20697–20709. [arXiv:2312.14132](https://arxiv.org/abs/2312.14132)
2. **CVPR 2024 Open Access** — [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_DUSt3R_Geometric_3D_Vision_Made_Easy_CVPR_2024_paper.pdf)
3. **NAVER LABS Europe 공식 페이지** — [europe.naverlabs.com/research/publications/dust3r-geometric-3d-vision-made-easy](https://europe.naverlabs.com/research/publications/dust3r-geometric-3d-vision-made-easy/)
4. **NAVER LABS Europe 블로그** — [europe.naverlabs.com/blog/3d-reconstruction-models-made-easy](https://europe.naverlabs.com/blog/3d-reconstruction-models-made-easy/)
5. **NAVER LABS Europe 3D Foundation Models 페이지** — [europe.naverlabs.com/research/3d-foundation-models/](https://europe.naverlabs.com/research/3d-foundation-models/)
6. **GitHub 공식 코드** — [github.com/naver/dust3r](https://github.com/naver/dust3r)
7. **LearnOpenCV 설명 아티클** — [learnopencv.com/dust3r-geometric-3d-vision](https://learnopencv.com/dust3r-geometric-3d-vision/)
8. **Awesome-DUSt3R (후속 연구 큐레이션)** — [github.com/ruili3/awesome-dust3r](https://github.com/ruili3/awesome-dust3r)
9. **MASt3R GitHub** — [github.com/naver/mast3r](https://github.com/naver/mast3r)
10. **MV-DUSt3R+ 프로젝트 페이지** — [mv-dust3rp.github.io](https://mv-dust3rp.github.io/)
11. **Wu et al.** "An Evaluation of DUSt3R/MASt3R/VGGT 3D Reconstruction on Photogrammetric Aerial Blocks," *arXiv:2507.14798*, 2025.
12. **DUSt3R CVPR 2024 Slides** — [europe.naverlabs.com/wp-content/uploads/2024/09/DUSt3R-slides.pdf](https://europe.naverlabs.com/wp-content/uploads/2024/09/DUSt3R-slides.pdf)
13. **HuggingFace Paper Page** — [huggingface.co/papers/2312.14132](https://huggingface.co/papers/2312.14132)
