# MUSt3R: Multi-view Network for Stereo 3D Reconstruction

> **논문**: Cabon et al., "MUSt3R: Multi-view Network for Stereo 3D Reconstruction" (arXiv:2503.01661v1, March 2025, NAVER LABS Europe, CVPR 2025 채택)

---

## 1. 핵심 주장 및 주요 기여 (간결 요약)

MUSt3R는 DUSt3R [Wang et al., CVPR 2024]가 가지는 **쌍별(pairwise) 처리의 이차 복잡도 문제**와 **국소 좌표계 기반 후처리 정렬(Global Alignment)의 비효율성**을 근본적으로 해결하기 위해 제안된 다시점(multi-view) 3D 재구성 네트워크입니다.

세 가지 핵심 기여는 다음과 같습니다:

1. **DUSt3R 아키텍처의 대칭화 및 N-view 일반화**: 비대칭 이중 디코더를 가중치 공유 단일 Siamese 디코더로 대체하여 메트릭(metric) 공간에서 N개 뷰의 포인트맵을 동일 좌표계로 직접 예측합니다.
2. **다층 메모리(Memory) 메커니즘 도입**: 오프라인 SfM과 온라인 SLAM/VO 시나리오를 단일 네트워크로 처리할 수 있게 하는 반복적 메모리 업데이트 구조를 추가했습니다.
3. **무캘리브레이션(uncalibrated) 환경에서 SOTA 달성**: FoV 추정, 카메라 포즈, 절대 스케일, 3D 재구성, 다시점 깊이 추정 등 여러 다운스트림 태스크에서 추론 속도 손실 없이 최첨단 성능을 보였습니다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

DUSt3R는 두 입력 이미지 $\{I_i\}\_{i=1,2}$에 대해 첫 번째 카메라 좌표계에서 정의되는 포인트맵 $X_{i,1} \in \mathbb{R}^{H \times W \times 3}$을 예측하지만, $N$개 이미지가 주어지면 $\binom{N}{2} = \mathcal{O}(N^2)$ 쌍을 처리해야 하며 각 예측이 서로 다른 국소 좌표계에 존재하므로 전역 정렬(GA) 후처리가 필수입니다. 이는 다음 문제를 야기합니다:

- 대규모 컬렉션에서 계산 복잡도 폭증
- 실시간(온라인) 처리 불가
- 정렬 최적화의 견고성·수렴성 문제

### 2.2 제안하는 방법 (수식 포함)

#### (1) 손실 함수 (DUSt3R로부터 계승)

기본 픽셀 단위 회귀 손실:

$$\ell_{\text{regr}}(i, j) = \sum_{p \in I_i} \left\| \frac{1}{z}\mathbf{X}_{i,j}[p] - \frac{1}{\hat{z}}\hat{\mathbf{X}}_{i,j}[p] \right\|$$

여기서 GT가 메트릭(metric)일 때는 MASt3R [Leroy et al., ECCV 2024]을 따라 $z := \hat{z}$로 두어 메트릭 예측을 강제합니다.

#### (2) 대규모 장면을 위한 로그 공간 변환 (논문의 핵심 개선)

먼 점들의 학습 안정성을 위해 다음 비선형 변환을 도입합니다:

$$f: x \mapsto \frac{x}{\|x\|}\log(1 + \|x\|)$$

$$\mathbf{X}'_{i,j}[p] = f\!\left(\tfrac{1}{z}\mathbf{X}_{i,j}[p]\right), \quad \hat{\mathbf{X}}'_{i,j}[p] = f\!\left(\tfrac{1}{\hat{z}}\hat{\mathbf{X}}_{i,j}[p]\right)$$

$$\ell_{\text{regr}}(i, j) = \sum_{p \in I_i} \left\| \mathbf{X}'_{i,j}[p] - \hat{\mathbf{X}}'_{i,j}[p] \right\|$$

#### (3) 대칭 디코더 + 학습 가능한 참조 임베딩

기준 이미지 $I_1$의 좌표계를 식별하기 위해 학습 가능한 임베딩 $\mathbf{B}$를 비참조 뷰에 더합니다:

$$\mathbf{D}_2^0 = \text{Lin}(\mathbf{E}_2) + \mathbf{B}$$

또한 cross-attention의 RoPE [Su et al., 2021]는 제거됩니다(부록 A.2 ablation).

#### (4) N-view Cross-Attention

$$\mathbf{M}_n^l = \text{Cat}_N(\mathbf{D}_1^l, \dots, \mathbf{D}_n^l), \quad \mathbf{M}_{n,-i}^l = \text{Cat}_N(\mathbf{D}_1^l, \dots, \mathbf{D}_{i-1}^l, \mathbf{D}_{i+1}^l, \dots, \mathbf{D}_n^l)$$

$$\mathbf{D}_i^l = \text{Dec}^l(\mathbf{D}_i^{l-1}, \mathbf{M}_{n,-i}^{l-1})$$

#### (5) 듀얼 포인트맵 헤드 (빠른 상대 포즈 추정)

각 뷰에 대해 전역 좌표계의 $\mathbf{X}\_{i,1}$과 자기 좌표계의 $\mathbf{X}_{i,i}$를 동시에 예측:

$$(\mathbf{X}_{i,1}, \mathbf{X}_{i,i}, \mathbf{C}_i) = \text{Head}^{3D}(\mathbf{D}_i^L), \quad i \in \{1,\dots,n\}$$

이 설계 덕분에 PnP 대신 **Procrustes alignment**으로 카메라 포즈를 추정할 수 있어 한 자릿수 빠른 추론이 가능합니다 (32.9 FPS vs 4.1 FPS, Tab. 7).

#### (6) 반복적 메모리 업데이트 (KV-cache 유사 구조)

새로운 이미지 $I_{n+1}$이 입력되면:

$$\mathbf{D}_{n+1}^l = \text{Dec}^l(\mathbf{D}_{n+1}^{l-1}, \mathbf{M}_n^{l-1})$$

이후 $\mathbf{M}\_{n+1}^l = [\mathbf{M}\_n^l; \mathbf{D}_{n+1}^l]$로 메모리를 확장합니다.

#### (7) Global 3D Feedback (Inj3D)

말단(terminal) 메모리 층의 정보를 초기 층으로 주입:

$$\bar{\mathbf{D}}_i^l = 
\begin{cases}
\mathbf{D}_i^l + \text{Inj3D}(\mathbf{D}_i^{L-1}), & \forall l < L-1 \;\text{and}\; i \in \mathcal{P} \\
\mathbf{D}_i^l, & l = L-1 \;\text{or}\; i \in \mathcal{N}
\end{cases}$$

여기서 $\text{Inj3D}$는 LayerNorm + 2-layer MLP(은닉 차원 4배 확장)로 구성됩니다. Ablation에 따르면 이 모듈이 메모리 크기 확장 시 성능 유지에 결정적입니다.

#### (8) 학습 손실

$N=10$ 뷰로 학습하며, 메모리에 $n$ 뷰를 채운 후 모든 뷰를 렌더링하여 손실을 계산:

$$\mathcal{L} = \sum_{i=1}^{n+N} \ell_{\text{regr}}(i, 1) + \ell_{\text{regr}}(i, i)$$

토큰 드롭아웃은 224 해상도 0.05, 512 해상도 0.15로 적용되며, 첫 이미지 토큰은 항상 보호됩니다.

### 2.3 모델 구조 개요

| 구성요소 | 역할 |
|---|---|
| Siamese ViT 인코더 | 각 이미지를 패치 임베딩으로 변환 (CroCo v2 [Weinzaepfel et al., ICCV 2023]에서 초기화, 학습 시 freeze) |
| 가중치 공유 디코더 ($L=12$) | self-attention + 메모리 cross-attention + MLP, $\text{Inj3D}$로 전역 3D 피드백 |
| 듀얼 헤드 $\text{Head}^{3D}$ | $(\mathbf{X}\_{i,1}, \mathbf{X}_{i,i}, \mathbf{C}_i)$ 동시 예측 |
| 메모리 관리 | 온라인: 발견율(discovery rate) 기반 KDTree 휴리스틱 / 오프라인: ASMK [Tolias et al., IJCV 2016] + farthest point sampling |

### 2.4 성능 향상 (주요 벤치마크 결과)

논문 Table 1, 2, 5, 6, 7, 8에서 인용:

- **TUM-RGBD VO (11개 시퀀스, ATE RMSE)**: MUSt3R = 5.5 cm로 무캘리브레이션 군에서 최고. 캘리브레이션을 사용하는 GlORIE-VO(9.3 cm), DROID-VO(11.4 cm)를 능가합니다.
- **Vertical FoV 오차**: MUSt3R = 4.32° vs Spann3R = 12.06°
- **3D Reconstruction (7-Scenes/NRGBD/DTU)**: MUSt3R-512가 DUSt3R-224 대비 5배 가벼우면서도 한 자릿수 빠른 FPS와 동등 이상의 정확도.
- **Relative Pose (CO3Dv2 mAA@30)**: MUSt3R-512 = 84.1 vs DUSt3R-512-GA = 76.7
- **속도**: Procrustes 변형 시 32.9 FPS 달성

### 2.5 한계 (논문 §5.5 명시)

- 첫 번째 뷰로부터 시점이 너무 많이 벗어난 경우(긴 궤적, 큰 회전) 좌표 드리프트가 발생합니다. ETH3D의 일부 대규모 시퀀스에서 RMSE가 여전히 큽니다 (Tab. 12).
- 메모리는 휴리스틱으로 관리되므로 전역적 일관성을 보장하는 학습 가능한 메모리 선택 전략은 없습니다.
- 동적 장면(dynamic scene)에 대한 명시적 모델링이 없어 MonST3R [Zhang et al., 2024]와 같이 별도 처리가 필요합니다.

---

## 3. 일반화 성능 향상 가능성

MUSt3R가 일반화 측면에서 주목받는 이유와 향후 가능성을 다음과 같이 정리합니다.

**(a) 학습-테스트 시점 수의 비대칭 일반화 (Empirical Evidence)**
부록 A.3과 Fig. 8에 따르면, 모델은 $N=10$ 뷰로만 학습되었음에도 메모리에 $n=50$ 뷰까지 축적했을 때 회귀 오차가 단조감소하며 그 이후로도 안정적으로 유지됩니다. 또한 $s=1$로 학습되었지만 추론에서 $s=10$으로 동시 처리해도 성능 저하가 미미합니다 — 이는 KV-cache 유사 메모리가 토큰 위치보다 어텐션 분포 자체에 의해 작동하기 때문으로 해석됩니다.

**(b) 좌표 표현의 도메인 일반화**
로그 공간 변환 $f(x) = \frac{x}{\|x\|}\log(1+\|x\|)$은 실내(수 미터)와 야외(수십~수백 미터) 사이 스케일 차이를 압축하여, 14개 이종 데이터셋(Habitat, MegaDepth, ScanNet++, ARKitScenes, MapFree, Virtual KITTI 등) 혼합 학습에서 발생하는 분포 불균형을 완화합니다. 이는 메트릭 예측 모델의 zero-shot 적용 범위를 의미 있게 확장하는 요인입니다.

**(c) 인코더 동결(freeze) 전략의 영향**
MUSt3R는 인코더를 동결하고 디코더만 학습하므로, CroCo v2의 self-supervised 사전학습이 보존되어 미학습 도메인(예: Aachen Day-Night, MIP-360)에서도 강건한 특징 표현을 유지합니다 (Fig. 5, 6 정성 결과).

**(d) 향후 일반화 강화 방향 (논문이 직접 명시하지 않은 추론적 가능성)**
- 학습 시 더 많은 뷰($N \gg 10$)로 확장 시, 어텐션 메커니즘의 길이 일반화(length generalization) 한계가 추가 연구 과제가 될 수 있습니다.
- 동적 장면, 비강체 변형, 수중/저조도 환경에 대한 일반화는 추가 도메인 데이터와 조건부 표현(예: time-aware tokens)이 필요해 보입니다.
- 현재 인코더 동결로 얻은 이점은 동시에 인코더 도메인 적응의 한계로도 작용할 수 있어, LoRA류 파라미터 효율 미세조정이 잠재적 개선 경로입니다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

DUSt3R 패러다임 이후 폭발적 후속 연구가 진행되었으며, MUSt3R의 위치는 다음과 같습니다.

| 모델 | 발표 | 핵심 아이디어 | 처리 방식 | MUSt3R 대비 |
|---|---|---|---|---|
| **DUSt3R** [Wang et al., CVPR 2024] | 2024 | Pointmap 회귀 패러다임 창시 | Pairwise + GA | 기반 모델, $\mathcal{O}(N^2)$ 복잡도 |
| **MASt3R** [Leroy et al., ECCV 2024] | 2024 | Descriptor head + InfoNCE matching loss | Pairwise | 매칭 강화, MUSt3R는 다시점·메모리 강조 |
| **MASt3R-SfM** [Duisterhof et al., 2024] | 2024 | ASMK 기반 Sparse SfM | Pairwise + 효율 GA | MUSt3R는 GA 자체 제거, 온라인 가능 |
| **Spann3R** [Wang & Agapito, 2024] | 2024 | 외부 spatial memory | Sequential pairs (2-pass) | MUSt3R는 진정한 N-view, 더 정확/빠름 |
| **MonST3R** [Zhang et al., 2024] | 2024 | 동적 장면 확장 | Pairwise | 보완 관계, 정적 가정 보완 |
| **MV-DUSt3R+** [Tang et al., CVPR 2025] | 2025 | 단일 단계 sparse view 재구성 (~2초) | Multi-view feed-forward | 유사 목표, MUSt3R는 메모리로 임의 확장성 |
| **Fast3R** [Yang et al., CVPR 2025] | 2025 | 단일 forward pass에 1000+ 이미지 | Parallel multi-view (FlashAttention) | MUSt3R는 인과적 온라인 처리 가능, Fast3R는 병렬 배치 강점 |
| **VGGT** [Wang et al., CVPR 2025] | 2025 | 대형 transformer로 글로벌 attention | Feed-forward | 모델 규모 확장, MUSt3R는 효율·메모리 관리 강조 |
| **MASt3R-SLAM** | 2025 | DUSt3R 패러다임 SLAM 통합 | Online | MUSt3R는 단일 네트워크로 SfM/SLAM 통합 |
| **STREAM3R** [arXiv:2508.10893] | 2025 | Causal transformer + scalable sequential | Streaming | MUSt3R의 인과적 메모리 컨셉 계승·확장 |
| **VGGT-Long, FastVGGT, Test3R, Point3R** | 2025 후반 | VGGT 한계 극복(긴 시퀀스, 효율) | 다양 | DUSt3R 계열 진화의 가속화 |

핵심 차별화: MUSt3R는 (1) **인과적 메모리 + 듀얼 헤드 + Procrustes**로 단일 네트워크가 SfM과 VO를 모두 처리, (2) 무캘리브레이션 시나리오에서 절대 스케일까지 회복하는 거의 유일한 모델, (3) Spann3R 대비 1.5-2배 빠르면서 더 정확합니다.

---

## 5. 향후 연구에 미치는 영향과 고려할 점

### 5.1 영향

1. **단일 네트워크 다중 시나리오 통합의 사례 제공**: 동일 가중치로 SfM과 VO/SLAM을 전환 없이 처리한다는 점은 신경 SLAM 시스템 설계에 중요한 기준점이 됩니다. 실제로 STREAM3R, MASt3R-SLAM 등이 유사한 통합 방향으로 진화하고 있습니다.
2. **메모리 기반 인과 트랜스포머의 3D 비전 도입**: KV-cache 유사 구조가 대규모 시퀀스 처리에 유효함을 입증하여, NLP에서 3D 비전으로의 추론 패러다임 이전을 가속화합니다.
3. **무캘리브레이션 메트릭 예측의 표준화**: 카메라 매개변수 없이도 절대 스케일까지 회복 가능함을 보임으로써, 모바일·웨어러블·로봇 응용의 진입 장벽을 낮춥니다.

### 5.2 향후 연구 시 고려할 점

- **장기 드리프트와 루프 클로저(loop closure)**: 현재 모델은 첫 뷰로부터의 좌표 누적 오차에 취약합니다. 학습 가능한 글로벌 토큰, 또는 그래프 기반 메모리 구조와의 결합이 유망합니다.
- **메모리 선택 학습화**: 휴리스틱(KDTree + discovery rate)을 학습 기반 키프레임 선택으로 대체하면 더 견고할 수 있습니다.
- **동적/비강체 장면 대응**: MonST3R, DAS3R, Geo4D 등과의 융합으로 시간 차원 일반화가 필요합니다.
- **모델 규모 vs 효율의 균형**: VGGT가 대형 모델로 정확도를 끌어올린 반면 MUSt3R는 효율을 강조했습니다. Fast3R, FastVGGT 등이 보여주듯 두 축의 동시 최적화가 향후 연구의 큰 과제가 될 것입니다.
- **불확실성 정량화**: 현재 confidence map은 픽셀 단위 출력에 그치며, 시스템 수준의 불확실성(예: 포즈 공분산)을 출력하면 다운스트림 SLAM 백엔드와의 통합이 용이해집니다.
- **3D Gaussian Splatting과의 결합**: InstantSplat 등 후속 연구처럼 MUSt3R 출력을 곧바로 3DGS 초기화로 사용하는 파이프라인이 자연스러운 응용 경로입니다.

---

## 참고 자료 (출처)

논문 본문 인용:

1. **MUSt3R 원논문**: Cabon et al., "MUSt3R: Multi-view Network for Stereo 3D Reconstruction," arXiv:2503.01661v1, March 2025. (https://arxiv.org/abs/2503.01661, https://arxiv.org/html/2503.01661v1, https://github.com/naver/must3r)

논문 내 핵심 인용 [원문 참고문헌 번호]:

2. [71] Wang et al., "DUSt3R: Geometric 3D Vision Made Easy," CVPR 2024.
3. [32] Leroy, Cabon, Revaud, "Grounding Image Matching in 3D with MASt3R," ECCV 2024.
4. [67] Wang & Agapito, "3D Reconstruction with Spatial Memory (Spann3R)," arXiv:2408.16061, 2024.
5. [19] Duisterhof et al., "MASt3R-SfM: A Fully-Integrated Solution for Unconstrained Structure-from-Motion," arXiv:2409.19152, 2024.
6. [80] Zhang et al., "MonST3R," arXiv:2410.03825, 2024.
7. [73] Weinzaepfel et al., "CroCo: Self-Supervised Pre-training for 3D Vision Tasks by Cross-View Completion," NeurIPS 2022.
8. [74] Weinzaepfel et al., "CroCo v2," ICCV 2023.
9. [59] Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," arXiv:2104.09864, 2021.
10. [44] Pope et al., "Efficiently Scaling Transformer Inference," MLSys 2023.
11. [63] Tolias, Avrithis, Jégou, "Image Search with Selective Match Kernels (ASMK)," IJCV 2016.
12. [79] Zhang et al., "GlORIE-SLAM: Globally Optimized RGB-only Implicit Encoding Point Cloud SLAM," arXiv:2403.19549, 2024.
13. [62] Teed, Lipson, Deng, "Deep Patch Visual Odometry (DPVO)," NeurIPS 2023.
14. [61] Teed & Deng, "DROID-SLAM," NeurIPS 2021.

웹 검색 출처 (2020년 이후 관련 최신 연구):

15. "awesome-dust3r" curated repository — Rui Li, GitHub (https://github.com/ruili3/awesome-dust3r). DUSt3R 계열 후속 연구 (Human3R, Rig3R, SAIL-Recon, FastVGGT, VGGT-Long, Test3R, Point3R, π³, MoGe-2, STream3R 등)의 종합 큐레이션.
16. Yang et al., "Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass," CVPR 2025 (https://fast3r-3d.github.io/, arXiv:2501.13928).
17. Tang et al., "MV-DUSt3R+: Single-Stage Scene Reconstruction from Sparse Views In 2 Seconds," CVPR 2025 Oral (https://mv-dust3rp.github.io/).
18. Wang et al., "VGGT: Visual Geometry Grounded Transformer," CVPR 2025 (출처: MLWires 리뷰 https://www.mlwires.com/metas-vggt-reconstructs-3d-scenes-in-seconds-cvpr-2025/).
19. "STREAM3R: Scalable Sequential 3D Reconstruction with Causal Transformer," arXiv:2508.10893, 2025.
20. "MASt3R & MASt3R-SfM for Image Matching and 3D Reconstruction," LearnOpenCV, April 2025 (https://learnopencv.com/mast3r-sfm-grounding-image-matching-3d/).
21. "An Evaluation of DUSt3R/MASt3R/VGGT 3D Reconstruction on Photogrammetric Aerial Blocks," Tandfonline, December 2025 (https://www.tandfonline.com/doi/full/10.1080/10095020.2025.2597491).
22. Zhang et al., "Review of Feed-forward 3D Reconstruction: From DUSt3R to VGGT," J. Artif. Intell. Control Syst., 2025 (https://www.researchgate.net/publication/393655961).

---

**참고 사항 (정확도)**: 본 분석은 (i) 업로드된 MUSt3R 논문(arXiv:2503.01661v1) 본문과 (ii) 위 웹 검색 결과에서 확인한 후속/관련 연구 정보를 기반으로 작성되었습니다. 논문 내부 결과(수식, 표 수치)는 원문에서 직접 인용했으며, 후속 연구 비교(Fast3R, VGGT, MV-DUSt3R+ 등)는 검색된 출처에서 확인한 내용으로만 한정하였습니다. 5절에서 일부 향후 방향(예: LoRA 미세조정, 3DGS 결합)은 논문이 직접 명시한 내용이 아닌 합리적 추론임을 명시합니다.
