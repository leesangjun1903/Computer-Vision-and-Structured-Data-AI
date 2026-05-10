# Reloc3r: Large-Scale Training of Relative Camera Pose Regression for Generalizable, Fast, and Accurate Visual Localization

> **논문 정보**: Siyan Dong, Shuzhe Wang, Shaohui Liu, Lulu Cai, Qingnan Fan, Juho Kannala, Yanchao Yang, "Reloc3r: Large-Scale Training of Relative Camera Pose Regression for Generalizable, Fast, and Accurate Visual Localization", arXiv:2412.08376v2 (2025년 3월 21일), CVPR 2025 채택. [코드: https://github.com/ffrivera0/reloc3r]

---

## 1. 핵심 주장과 주요 기여 요약

Reloc3r는 **시각적 위치 추정(Visual Localization)** 분야에서 기존의 **상대적 카메라 포즈 회귀(Relative Pose Regression, RPR)** 방법들이 가진 세 가지 한계, 즉 **(i) 새로운 장면에 대한 일반화 부족, (ii) 추론 효율성 저하, (iii) 포즈 추정 정확도 부족**을 동시에 해결하는 단순하지만 강력한 프레임워크입니다.

**핵심 기여**는 다음과 같이 요약됩니다.

1. **완전 대칭(fully symmetric)** ViT 기반 상대 포즈 회귀 네트워크를 제안하고, 이를 **최소화된 모션 평균(motion averaging) 모듈**과 결합한 단순한 파이프라인을 구성했습니다.
2. 약 **800만 개의 자세(pose) 정보가 있는 이미지 쌍**을 7개의 공개 데이터셋에서 수집·정제하여 **포즈 회귀 분야 최초의 파운데이션 모델급 대규모 학습**을 수행했습니다.
3. 기존 RPR 방법뿐 아니라 일부 **APR(Absolute Pose Regression)·구조 기반 방법까지 능가**하며, 학습되지 않은 6개 벤치마크에서 일관된 SOTA를 달성했습니다 (논문 Sec. 4 및 Tables 2–11).

---

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

데이터베이스 $D = \{I_{d_n} \in \mathbb{R}^{H \times W \times 3} \mid n=1, \dots, N\}$의 자세 정보가 있는 이미지들과 동일 장면의 질의(query) 이미지 $I_q$가 주어졌을 때, 질의 이미지의 6-DoF 카메라 포즈 $P \in \mathbb{R}^{3\times 4}$를 추정하는 것이 목표입니다. 여기서 $P$는 회전 $R \in \mathbb{R}^{3\times 3}$과 평행이동 $t \in \mathbb{R}^3$로 표현됩니다.

기존 접근법의 trade-off는 다음과 같습니다.
- **구조 기반(SfM + matching)**: 정확하지만 추론이 느리고 시스템이 복잡함.
- **장면 좌표 회귀(SCR)**: 일반화 부족, 장면별 학습 필요.
- **APR**: 빠르나 장면-특화이며 정확도 부족.
- **RPR**: 일반화 가능성은 있으나 정확도가 APR/구조 기반에 못 미침.

### 2.2 제안 방법 — 모델 구조 (Sec. 3.1)

전체 파이프라인은 두 모듈로 구성됩니다 (Figure 2 in paper):

#### (a) 상대 포즈 회귀 네트워크

DUSt3R 백본을 채택하되, **두 분기(branch)가 가중치를 완전히 공유하는 대칭 구조**로 변경했습니다.

**ViT 인코더**: 입력 이미지를 토큰화하여 RoPE 위치 임베딩과 함께 $m=24$개의 인코더 블록을 통과시킵니다.

$$F_i^{(T \times d)} = \mathrm{Encoder}\bigl(\mathrm{Patchify}(I_i^{(H \times W \times 3)})\bigr), \quad i = 1, 2$$

**ViT 디코더**: $n=12$개 블록에서 self-attention과 feed-forward 사이에 cross-attention을 끼워 두 토큰 집합 간 상호작용을 학습합니다.

$$G_1^{(T \times d)} = \mathrm{Decoder}\bigl(F_1^{(T \times d)}, F_2^{(T \times d)}\bigr)$$
$$G_2^{(T \times d)} = \mathrm{Decoder}\bigl(F_2^{(T \times d)}, F_1^{(T \times d)}\bigr)$$

**포즈 회귀 헤드**: $h=2$개의 feed-forward 층 + 평균 풀링으로 회전(9D 표현 후 SVD로 SO(3) 직교화)과 평행이동(방향 단위 벡터)을 산출합니다.

$$\hat{P}_{I_1, I_2}^{(3\times 4)} = \mathrm{Head}(G_1^{(T\times d)}), \quad \hat{P}_{I_2, I_1}^{(3\times 4)} = \mathrm{Head}(G_2^{(T\times d)})$$

**지도 신호(loss)**: 회전과 평행이동을 모두 **각도(angle)** 단위로 다루어 두 항의 가중치 균형 문제와 데이터셋 간 metric 스케일 불일치 문제를 회피합니다.

$$\mathcal{L} = \ell_R + \ell_t$$

$$\ell_R = \arccos\!\Bigl(\frac{\mathrm{tr}(\hat{R}^{-1} R) - 1}{2}\Bigr), \quad \ell_t = \arccos\!\Bigl(\frac{\hat{t}\cdot t}{\|\hat{t}\|\,\|t\|}\Bigr)$$

여기서 $\mathrm{tr}(\cdot)$은 행렬의 트레이스이며 $\hat{R}, \hat{t}$가 예측값, $R, t$가 ground-truth입니다.

#### (b) 모션 평균(Motion Averaging) 모듈 (Sec. 3.2)

질의 $I_q$에 대해 NetVLAD로 상위 $K=10$개의 데이터베이스 이미지를 검색한 뒤, 각 쌍에서 얻은 상대 포즈를 결합합니다. **학습 가능한 파라미터는 없습니다**.

- **회전 평균**: 각 쌍에서 $\hat{R}\_q = R_{d_i} \hat{R}_{q, d_i}$ 계산 후, 쿼터니언 표현 기반의 **중앙값(median) 회전**을 최종 추정치로 사용 (mean보다 노이즈에 강건함).
- **카메라 중심 삼각측량(triangulation)**: 두 쌍 이상의 결과로부터 카메라 중심까지의 평행이동 방향선들과의 거리 제곱합을 최소화하는 SVD 기반 최소제곱 해를 사용:

$$\mathbf{c}^{*} = \arg\min_{\mathbf{c}} \sum_{i} \bigl\| (\mathbf{c} - \mathbf{c}_{d_i}) - \bigl((\mathbf{c} - \mathbf{c}_{d_i})\cdot \hat{\mathbf{u}}_i \bigr)\hat{\mathbf{u}}_i \bigr\|^2$$

여기서 $\hat{\mathbf{u}}_i$는 각 쌍에서 추정된 평행이동 단위 벡터입니다 (논문에서는 SVD를 통한 표준 직선 교차 공식을 사용한다고 기술).

### 2.3 학습 데이터 (Sec. 4 및 Table 1)

| 데이터셋 | 장면 유형 | 이미지 쌍 수 |
|---|---|---|
| CO3Dv2 | 객체 중심 | ~1M |
| ScanNet++ | 실내 | ~850K |
| ARKitScenes | 실내 | ~2.1M |
| BlendedMVS | 야외 | ~1M |
| MegaDepth | 야외 | ~1.8M |
| DL3DV | 실내·야외 | ~1.1M |
| RealEstate10K | 실내·야외 | ~100K |

총 **약 800만 쌍**, 다양한 도메인을 포괄합니다.

### 2.4 성능 향상 (Sec. 4.1, 4.2)

논문이 주장하는 정량 결과는 다음과 같습니다 (모두 학습 시 보지 못한 데이터셋·장면).

- **ScanNet1500 (Pair-wise)**: AUC@5 = **34.79**, AUC@20 = **75.56**. DUSt3R 대비 AUC@20 약 +13%p, ROMA·MASt3R도 능가하면서 **추론 ~25 ms (50배 이상 빠름)** (Table 3).
- **CO3Dv2 (Multi-view)**: RRA@15 = 95.8%, RTA@15 = 93.7%, mAA@30 = 82.9%로 PR/Non-PR 부문 모두 SOTA (Table 2).
- **7 Scenes (실내 absolute pose)**: 평균 0.04 m / 1.02° — **장면별 학습 없이도** APR 일부 모델과 동등하거나 더 우수 (Table 4).
- **Cambridge Landmarks (야외)**: 평균 0.55 m / 0.56° — 기존 RPR 대비 **오차 절반**, 회전 오차는 모든 APR 평균보다 우수 (Table 5).
- **모델 크기**: 0.43B 파라미터, RTX 4090에서 실시간 추론 (Sec. C of supplementary).

### 2.5 한계 (Sec. 4.3, 부록 C)

1. **공선 퇴화(collinear degeneracy)**: 질의 이미지와 검색된 데이터베이스 이미지들이 모두 한 직선 위에 있을 때 모션 평균으로 metric 스케일을 복원할 수 없음.
2. **초점 거리 변화에 취약**: 3–4× zoom in/out과 같은 큰 intrinsic 변화에서 평행이동 추정 부정확 (Figure 5). 이는 본질적으로 two-view geometry의 *scale-distance ambiguity*와 동일한 구조적 한계.
3. **카메라 내부 파라미터 미사용**: 5-point algorithm처럼 ground-truth intrinsic을 활용하지 않아, 일부 실패 사례 발생.
4. **MegaDepth1500과 같이 intrinsic 변화가 극심한 데이터셋**에서는 여전히 매칭 기반 SOTA(ROMA, Efficient LoFTR)에 못 미침 (Table 11).
5. **점-매칭 부재로 post-optimization 불가**: LazyLoc 같은 방법의 후처리 정제를 적용하기 어려움.

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

이 논문은 RPR 분야에서 **"규모(scale)와 다양성(diversity)이 일반화의 핵심"**임을 실증한 대표 사례입니다. 논문이 제시하는 일반화 메커니즘은 다음과 같이 정리할 수 있습니다.

### 3.1 일반화의 구조적 토대

(a) **대칭 아키텍처**가 이미지 순서 편향(order bias)을 제거하여 학습 분포 외 데이터에서도 안정적입니다. 비대칭 변형(Reloc3r-512 asymmetric)은 ScanNet1500 AUC@20에서 75.56 → 74.63으로 떨어지고, 파라미터 수도 약 28% 더 많습니다 (Table 6).

(b) **Metric 스케일을 학습하지 않는 결정**이 핵심입니다. 회전과 평행이동 모두 각도 단위로 다루어 데이터셋별 절대 스케일 차이로 인한 학습 충돌을 회피합니다. Metric 버전(Reloc3r-512 metric)은 AUC@5 25.70으로 기본 모델 34.79보다 크게 떨어집니다 (Table 6, 9).

(c) **800만 쌍의 멀티 도메인 학습**: Table 10 분석에서 ScanNet++ 단일 학습 모델은 ScanNet1500 AUC@20=68.70에 그치지만, 전체 데이터로 학습 시 75.56까지 향상됩니다. RE10K, ACID 같은 분포 외 데이터셋에서는 더 큰 폭으로 향상됩니다.

### 3.2 사전학습 가중치의 중요성

부록 Table 8에 따르면 초기화 전략별 ScanNet1500 AUC@5는 다음과 같습니다.
- 무작위 초기화 (224): 3.74
- DUSt3R 인코더만 초기화: 17.83
- CroCo v2 (full): 22.44
- MASt3R (full): 32.62
- **DUSt3R-512 (full): 34.79**

이는 **3D 기하 사전학습으로 얻은 기하 인식 능력이 포즈 회귀의 일반화에 결정적**이라는 점을 시사합니다.

### 3.3 흥미로운 emergent 현상 — 암묵적 매칭 학습

논문 Figure 4, 6, 7과 Sec. C에서 중요한 관찰을 제시합니다. **포즈 supervision만으로 학습했음에도 디코더의 cross-attention 맵이 패치 단위 대응(matching)을 자발적으로 학습**합니다. 이는 다음을 의미합니다.

- 포즈 회귀 네트워크가 단순 회귀가 아니라 **암묵적 기하 추론**을 수행함.
- Sattler et al. 2019 (CVPR)가 지적한 "APR은 본질적으로 image retrieval과 유사하다"는 한계를 RPR + 대규모 학습이 부분적으로 극복함.
- 무작위 초기화에서도 cross-attention이 어느 정도 매칭을 학습하나, DUSt3R 초기화 시 더 정확하고 집중된 응답이 나타남 → ground-truth correspondence supervision을 추가하면 더 큰 향상 가능성.

### 3.4 In-the-wild 일반화

부록 F (Figure 8, 9)에서는 **회화·스케치·실사 이미지 간**, 그리고 **서로 다른 인물 얼굴 간**에서도 합리적인 상대 포즈를 추정합니다. 이는 단순한 도메인 적응을 넘어 매우 강한 OOD(out-of-distribution) 일반화를 시사합니다.

### 3.5 일반화의 잔존 한계

- **카메라 intrinsic 변화에 약함**: 학습 분포 내 일반적인 초점거리 범위를 벗어나면 실패 (Figure 5).
- **공선 구도에서의 metric 복원 불가**: 데이터 분포 자체로 해결 불가능한 기하적 퇴화.
- **객체-중심 데이터(CO3Dv2)를 빼면** 객체 도메인 정확도만 떨어지고 다른 도메인에는 영향 미미 (Table 10) → 도메인별 데이터의 marginal benefit은 도메인 내부에 한정됨.

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 향후 연구에 미치는 영향

1. **포즈 회귀의 패러다임 전환**: 그동안 RPR은 "정확도가 부족한 경량 대안"으로 여겨졌으나, Reloc3r는 **DUSt3R 계열 3D 파운데이션 모델 + 대규모 학습**이라는 처방으로 RPR을 SOTA 영역까지 끌어올렸습니다. 이는 후속 GeLoc3r(arXiv:2509.23038, 2025) 등 RPR 개선 연구의 출발점이 되고 있습니다.

2. **3R 생태계 형성에 기여**: DUSt3R, MASt3R, MASt3R-SfM, MonST3R, CUT3R 등과 함께 **"3R" 계열 기하 파운데이션 모델 생태계**의 시각적 위치 추정 분야 대표 모델로 자리매김했습니다 (3D-Vision-World/All-3R-SLAM-in-this-Repo 정리 기준).

3. **Metric-scale-free 학습 전략의 유효성 입증**: "모든 것을 직접 회귀하지 말고, 학습 가능한 부분과 기하 알고리즘이 잘 푸는 부분을 분리하라"는 hybrid design 원칙을 강하게 지지합니다.

4. **Cross-attention의 emergent matching 발견**: 포즈 supervision만으로 매칭이 emergent하게 학습된다는 관찰은, **dense correspondence supervision 없이도 기하 인식 능력을 학습할 수 있다는 가능성**을 보여 후속 연구의 학습 신호 설계에 영향을 줄 수 있습니다.

### 4.2 향후 연구 시 고려할 점

1. **공선 퇴화 해결**: Metric scale을 신경망과 모션 평균이 hybrid로 추정하거나, IMU·단안 깊이·learned scale prior 등 보조 신호 통합 연구.
2. **카메라 intrinsic embedding**: NoPoSplat의 intrinsic token 임베딩처럼, focal length 변화에 강건하도록 intrinsic을 명시적으로 입력에 포함하는 방안.
3. **Essential matrix 회귀**: 5-point algorithm이 처리하는 기하 제약을 학습 출력에 반영하면 scale-distance 모호성 완화 가능 (논문 자체에서 제안).
4. **Cross-attention에 매칭 supervision 추가**: 학습 수렴 가속과 정확도 향상에 직접 기여할 가능성 (Sec. C 시사).
5. **점 매칭 출력 부가**: 후처리 단계(LazyLoc 류 robust estimator, post-optimization) 적용을 위한 매칭 출력 헤드 결합.
6. **CNN 백본의 한계 확인**: 본 논문이 시도한 ResUNet 기반 확장은 0.1B 파라미터에서도 underfit/overfit 문제를 보였으므로, 향후 RPR scaling은 **Transformer 기반이 사실상 표준**이 될 가능성이 큼.
7. **동적 장면 일반화**: 본 모델은 정적 장면 가정. MonST3R, CUT3R처럼 동적/시간적 일관성을 결합하는 방향 필요.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연도 | 방법 | 카테고리 | 핵심 아이디어 | 강점 | 약점 |
|---|---|---|---|---|---|
| 2020 | Zhou et al. *EssNet/NC-EssNet* (ICRA) | RPR | Essential matrix 분해 기반 학습 | 단순 | 미보유 장면 정확도 큰 폭 하락 (7 Scenes 평균 0.48 m / 32.97°) |
| 2021 | Sarlin et al. *PixLoc* / Brachmann et al. *DSAC*\* | Hybrid / SCR | Pixel-level/scene-coord 회귀 | 정확 | 장면별 학습, 느림 |
| 2021 | Turkoglu et al. *Relpose-GNN* (3DV) | RPR(Seen) | GNN으로 다중 쌍 결합 | RPR seen에서 우수 | unseen 일반화 제약 |
| 2022 | Arnold et al. *Map-free* (ECCV) | RPR | 단일 이미지 metric pose, 약 523K 쌍 학습 | 진정한 unseen | 실내·야외 별도 모델 필요, 정확도 한계 |
| 2023 | Brachmann et al. *ACE* (CVPR) | SCR | 분 단위 장면별 학습 | 빠른 적응 | 장면별 학습 필요 |
| 2024 | Wang et al. *DUSt3R* (CVPR) | 3D foundation | Pointmap 회귀, 두 비대칭 분기 | 다목적, 강한 일반화 | 비대칭 → 매칭에 직접 최적화 X, 추론 느림 (~441 ms) |
| 2024 | Leroy et al. *MASt3R* | 3D foundation + matching | DUSt3R + 매칭 헤드, InfoNCE | 매칭/포즈 정확도 향상 | 여전히 매칭 기반, 느림 (~294 ms) |
| 2024 | Wang et al. *VGGSfM* (CVPR) | Differentiable SfM | End-to-end SfM | CO3Dv2 강력 | 본질적으로 SfM 비용 |
| 2024 | Chen et al. *Marepo* (CVPR) | APR-like | Pose regressor + scene-specific map, 15분/장면 | 빠른 적응, 정확 | 장면별 학습 필요 |
| 2024 | Ye et al. *NoPoSplat* (ICLR'25 Oral) | Pose-free 3DGS | Canonical 공간 Gaussian + intrinsic token | NVS+pose 통합, 높은 정확도 | 추론 >2000 ms, 학습 비용 큼 |
| 2025 | **Reloc3r** (CVPR'25) | RPR (대규모) | 대칭 ViT, 800만 쌍, 모션 평균, scale-free | 일반화·속도·정확도 모두 SOTA | 공선 퇴화, intrinsic 변화 약함 |
| 2025 | *GeLoc3r* (arXiv:2509.23038) | RPR + 기하 일관성 | Reloc3r 기반에 geometric consistency regularization | 추가 정확도 향상 | Reloc3r에 의존, 평가 초기 단계 |

위 비교에서 보듯, 2020–2022년의 RPR은 **"빠르지만 부정확"**, 2023–2024년의 매칭 기반 3R 모델(DUSt3R/MASt3R)은 **"정확하지만 느림"**, 그리고 2024–2025년의 NoPoSplat 같은 pose-free 3DGS는 **"정확하지만 매우 무거움"**이라는 trade-off가 있었습니다. **Reloc3r는 이 세 축의 균형점**(Figure 1: ScanNet1500에서 24 FPS + AUC@5 ≈ 35)을 처음으로 잘 잡은 모델로 평가됩니다.

---

## 참고자료 (References)

**1차 자료 (논문 본문)**
- Dong, Wang, Liu, Cai, Fan, Kannala, Yang. *Reloc3r: Large-Scale Training of Relative Camera Pose Regression for Generalizable, Fast, and Accurate Visual Localization*. arXiv:2412.08376v2, 2025. (CVPR 2025) [업로드된 PDF 본문 및 부록]

**검색을 통해 확인한 외부 자료**
- Leroy, Cabon, Revaud. *Grounding Image Matching in 3D with MASt3R*. arXiv:2406.09756 — <https://www.emergentmind.com/papers/2406.09756>
- Wang, Leroy, Cabon, Chidlovskii, Revaud. *DUSt3R: Geometric 3D Vision Made Easy*. arXiv:2312.14132 — <https://arxiv.org/abs/2312.14132>
- Ye et al. *No Pose, No Problem: Surprisingly Simple 3D Gaussian Splats from Sparse Unposed Images (NoPoSplat)*. arXiv:2410.24207 (ICLR 2025 Oral) — <https://arxiv.org/abs/2410.24207>, <https://noposplat.github.io/>
- *GeLoc3r: Enhancing Relative Camera Pose Regression with Geometric Consistency Regularization*. arXiv:2509.23038, 2025 — <https://arxiv.org/html/2509.23038>
- 3D-Vision-World GitHub repository, *All-3R-SLAM-in-this-Repo* — <https://github.com/3D-Vision-World/All-3R-SLAM-in-this-Repo> (Reloc3r의 CVPR 2025 채택 확인)
- NAVER LABS Europe 블로그, *MASt3R: Matching And Stereo 3D Reconstruction* — <https://europe.naverlabs.com/blog/mast3r-matching-and-stereo-3d-reconstruction/>
- LearnOpenCV, *DUSt3R: Geometric 3D Vision Made Easy — Explanation & Results* — <https://learnopencv.com/dust3r-geometric-3d-vision/>

**참고 부언**: 논문에서 인용한 [4] Map-free, [7] RelocNet, [49] Relative PN, [101] Relpose-GNN, [111] DUSt3R, [116] ExReNet, [122] NoPoSplat, [127] RelPose, [128] RayDiffusion 등의 비교 수치(Table 2–11)는 모두 Reloc3r 본문 표에서 직접 가져온 값이며, 표 외의 정량 수치(예: 다른 데이터셋에서의 별도 수치)는 본 답변에 포함하지 않았습니다.

논문에 명시되지 않은 일부 세부 사항(예: 카메라 중심 삼각측량의 SVD 풀이 정확한 형태)은 표준 multi-view geometry 기법(Hartley & Sturm 1997)으로 알려져 있어 수식을 일반적 형태로 제시했으며, 논문은 "least-squares method that minimizes the sum of squared distances from the camera center to each translation direction"이라고 기술합니다 (Sec. 3.2).
