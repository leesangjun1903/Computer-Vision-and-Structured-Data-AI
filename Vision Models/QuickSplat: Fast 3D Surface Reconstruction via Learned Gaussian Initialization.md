# QuickSplat: Fast 3D Surface Reconstruction via Learned Gaussian Initialization

> ⚠️ **답변의 근거**: 아래 분석은 사용자가 업로드한 논문 PDF("QuickSplat: Fast 3D Surface Reconstruction via Learned Gaussian Initialization", arXiv:2505.05591v2, 2025년 8월 10일판)의 본문 내용에 한정하여 작성했습니다. 논문에 명시되지 않은 외부 정보는 추가하지 않았으며, 비교 연구도 해당 논문이 인용한 참고문헌 범위 내에서만 다룹니다.

---

## 1. 핵심 주장 및 주요 기여 요약

QuickSplat은 **대규모 실내 장면(indoor scene)에 대한 3D 표면 재구성(surface reconstruction)** 문제에서, 기존 2D Gaussian Splatting(2DGS) 기반 per-scene 최적화의 **느린 수렴 속도**와 **텍스처 없는 영역(textureless wall 등)에서의 품질 저하**를 동시에 해결하는 **데이터 기반 학습 prior** 프레임워크다.

핵심 기여는 세 가지다.

1. **학습된 초기화(Initializer) 네트워크**: SfM 점군을 단순히 입력하지 않고, sparse 3D CNN으로 dense한 Gaussian 분포를 예측해 **textureless 영역의 holes를 메움**.
2. **Densifier–Optimizer 결합 루프**: 휴리스틱(gradient threshold 기반 split/clone) 없이, 렌더링 그래디언트로부터 **새 Gaussian의 위치(densifier)** 와 **기존 Gaussian의 업데이트 벡터(optimizer)** 를 동시에 학습.
3. **8× 빠른 런타임 + 48% 더 낮은 깊이 오차**: ScanNet++ 기준 124초로 2DGS(1796s), GS2Mesh(973s)를 압도하면서도 정확도 우위.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 2DGS / 3DGS 기반 surface reconstruction의 두 가지 한계:

- **속도**: 한 장면당 30K iteration의 SGD 최적화 → 30분 이상 소요.
- **품질**: 광도 손실(photometric loss)에만 의존 → SfM 점이 거의 없는 흰 벽 등 textureless 영역에서 floating artifact, bent wall 발생.

### 2.2 표면 표현 (2DGS 기반)

QuickSplat은 2DGS를 채택. 각 Gaussian $g_i \in \mathbb{R}^{14}$ 는 위치, scale, 회전, opacity, RGB로 매개변수화되며, alpha-blending으로 픽셀 색상을 렌더링한다:

$$c(x) = \sum_{i \in \mathcal{N}} c_i \alpha_i T_i, \quad \text{with } T_i = \prod_{j=1}^{i-1}(1-\alpha_j)$$

여기서 $\alpha_i = o_i \cdot g_i^{2D}(\mathbf{u}(x))$ 는 opacity와 2D Gaussian 값의 곱이다.

렌더링 손실은 L1 + SSIM 결합:

$$\mathcal{L}_c(x) = 0.8\,\|c(x) - C\|_1 + 0.2\,(1 - \mathrm{SSIM}(c(x), C))$$

### 2.3 모델 구조

전체 파이프라인은 **Initializer → (Densifier ↔ Optimizer) × T 반복** 구조 (논문 Fig. 2).

#### (a) Initializer 네트워크 $\theta_I$

- **백본**: SGNN [Dai et al., 2020] 기반 sparse 3D CNN encoder-decoder. 보틀넥의 dense CNN block과 upsampling layer로 voxel 밀도를 점진적으로 증가.
- **출력**: 각 voxel을 64-d latent feature $\mathbf{f}$ 로 표현하고 MLP로 $v_g = 2$ 개 Gaussian 디코딩.
- **위치 디코딩** (PixelSplat의 reparameterization trick 응용):

$$\mathbf{g}_c = \mathbf{v}_c + R\,(2\sigma(x) - 1)$$

여기서 $R = 4 v_d$, $v_d = 4\text{cm}$, $\sigma$ 는 sigmoid. opacity는 마지막 upsampling layer의 occupancy로 정의되어 rendering loss가 backprop 가능.

- **손실 함수**: occupancy BCE + depth L1 + normal cosine + distortion 결합

$$\mathcal{L}_n = 1 - \mathbf{n}_g^T \mathbf{n}_m$$

$$\mathcal{L}(\theta_I) = \mathcal{L}_c + \mathcal{L}_d + \mathcal{L}_{occ} + 0.01\,\mathcal{L}_n + 10\,\mathcal{L}_{dist}$$

#### (b) Optimizer 네트워크 $\theta_O$

G3R [Chen et al., 2024]에서 영감을 받아, 현재 Gaussian과 그래디언트를 입력으로 업데이트 벡터를 예측:

$$f_{\theta_O}(\mathcal{G}_t, \nabla \mathcal{G}_t, t) = \Delta \mathcal{G}_t, \qquad \mathcal{G}_{t+1} = \mathcal{G}_t + \Delta \mathcal{G}_t$$

출력은 $[-1, 1]$ 로 정규화하여 overshoot 방지. Sparse 3D UNet [Choy et al., 2019] 사용.

#### (c) Densifier 네트워크 $\theta_D$

Optimizer는 voxel 중심 주변에서만 이동 가능 → 빈 공간의 holes 못 메움. 이를 보완하는 새 Gaussian 생성기:

$$f_{\theta_D}(\mathcal{G}_t, \nabla \mathcal{G}_t, t) = \hat{\mathcal{G}}_t$$

특징:
- 보틀넥에 dense block 미사용 → **기존 voxel의 이웃에만 새 voxel 배치**.
- **Importance sampling**: occupancy ≈ opacity 해석을 활용해 "solid"한 voxel을 우선 샘플링.
- 시간 t에 따라 추가 voxel 수가 줄어듦:

$$n(t) = \frac{s}{2^t}, \quad s = 20\text{K}$$

#### (d) Densification–Optimization 루프

매 iteration $t$:

$$\bar{\mathcal{G}}_t = \mathcal{G}_t \cup \hat{\mathcal{G}}_t, \quad \bar{\nabla \mathcal{G}}_t = \nabla \mathcal{G}_t \cup \mathbf{0}$$

$$\mathcal{G}_{t+1} = \bar{\mathcal{G}}_t + f_{\theta_O}(\bar{\mathcal{G}}_t, \bar{\nabla \mathcal{G}}_t, t)$$

총 $T = 5$ 회 반복. End-to-end 학습 시 G3R 방식대로 **timestep 간 그래디언트는 detach**.

### 2.4 성능 향상 (Tab. 1, ScanNet++ 20개 테스트 장면)

| Method | Abs err ↓ | Acc(2cm) ↑ | Chamfer ↓ | Time ↓ |
|---|---|---|---|---|
| 2DGS | 0.1127 | 0.4021 | 0.2420 | 1796s |
| GS2Mesh | 0.1212 | 0.4028 | 0.2012 | 973s |
| MonoSDF | 0.0569 | 0.5774 | 0.1450 | >10h |
| **Ours (w/o opt)** | 0.0732 | 0.5263 | 0.1461 | **26s** |
| **Ours (w/ opt)** | **0.0578** | **0.5783** | **0.1347** | 124s |

특히 Ablation(Tab. 2)에서 **initializer만으로도** Chamfer 0.137 달성, densifier 추가 시 fine detail 개선 확인.

### 2.5 한계

논문 4.3절에 명시:
1. **거울 반사**: 광도 손실이 반사된 기하를 거울 뒤편에 재구성 → 노이즈 발생.
2. **정적 장면 가정**: 사람 등 동적 객체 처리 불가.
3. **비실시간**: SLAM 통합은 향후 과제 [SplaTAM, GS-SLAM 등 인용].

---

## 3. 일반화 성능 향상 가능성 (중점 분석)

이 부분은 논문의 **Appendix B**가 핵심이다.

### 3.1 Cross-dataset 일반화 (Tab. 4, Fig. 7–8)

ScanNet++로만 학습한 모델을 **fine-tuning 없이** 두 가지 외부 데이터셋에 직접 적용:

- **ARKitScenes**: 모바일 RGB-D 캡처. RGB FoV가 ScanNet++보다 좁음.
- **Mip-NeRF 360 (Room)**: COLMAP 점군의 절대 스케일을 monocular metric depth(Depth Anything v2)로 복원.

ARKitScenes 5장면 결과:

| Method | Abs err ↓ | Acc(10cm) ↑ | Chamfer ↓ | Time |
|---|---|---|---|---|
| 2DGS | 0.6978 | 0.3590 | 0.6015 | 1780s |
| **Ours** | **0.1775** | **0.7698** | **0.4301** | **111s** |

도메인 갭이 존재함에도 2DGS 대비 **Abs err 약 4배 감소**, **Chamfer 28% 개선**.

### 3.2 일반화가 가능한 구조적 이유

논문 본문에서 직접 언급하거나 합리적으로 추론할 수 있는 요인들:

1. **Sparse 3D convolution 기반**: 입력 이미지 수에 무관하게 voxel grid에서 동작하므로 다중 방 같은 **대규모 장면**에도 확장 가능 (Fig. 9).
2. **Gaussian representation space에서의 prior 학습**: GS2Mesh처럼 외부 stereo estimator를 끼워 넣지 않고 표현 공간 안에서 학습 → **렌더링 노이즈에 강건**.
3. **Initializer 모듈성** (Appendix C, Tab. 7): SAGS [Ververas et al., 2024]에 QuickSplat initializer를 끼워 넣자, SfM 초기화 대비 Abs err가 0.1292 → 0.0692로 감소. 즉 **다른 GS 변종에도 plug-in 가능**한 일반화.

### 3.3 일반화의 잠재적 제약

논문이 직접 다루지는 않지만 표 4와 Fig. 7에서 추론 가능한 부분:
- ScanNet++의 ground-truth mesh(레이저 스캔)에 강하게 종속된 학습 → **실외 장면, 비-맨해튼 구조, 비정형 환경**에서의 성능은 검증되지 않음.
- 4cm voxel 해상도 가정 → 매우 미세한 구조나 얇은 객체에서는 한계 가능.

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 영향

1. **휴리스틱-free densification 패러다임 제시**: 3DGS의 고질적 문제였던 "언제, 어디서 split/clone?" 질문을 학습으로 대체. Mini-Splatting [Fang & Wang, 2024], Taming 3DGS [Mallick et al., 2024], Revising Densification [Rota Bulò et al., 2024] 등 휴리스틱 개선 계열 연구의 **상위 대안 가능성**.
2. **Meta-learning of optimization**의 3D 재구성 적용 확대: G3R [Chen et al., 2024]의 LiDAR 의존성을 제거하고 **순수 다중뷰 RGB**로 일반화한 사례.
3. **Feed-forward 방법과 per-scene 최적화의 가교**: PixelSplat [Charatan et al., 2024], MVSplat [Chen et al., 2024], GS-LRM [Zhang et al., 2024] 등 sparse-view feed-forward 모델과, NeuS [Wang et al., 2021]·Neuralangelo [Li et al., 2023] 같은 per-scene 최적화 사이의 **하이브리드 설계 표본**.

### 4.2 향후 연구 시 고려할 점

1. **동적 장면 / 거울 / 유리** 처리 — 논문의 한계 그대로.
2. **SLAM 통합** — 저자들이 직접 제시한 방향 (SplaTAM, MonoGS, GS-SLAM 인용).
3. **학습 prior의 도메인 의존성 평가**: 실외, 합성-실사 갭, 저텍스처 외에도 **고반사/투명 객체, 식생, 비정형 구조**에 대한 벤치마크 필요.
4. **Voxel 해상도 자동화**: 현재 $v_d = 4\text{cm}$ 고정. Adaptive voxel size 또는 Octree 기반 확장이 fine-detail 한계를 극복할 수 있음.
5. **GT mesh 의존성 완화**: $\mathcal{L}_{occ}$, $\mathcal{L}_n$, $\mathcal{L}_d$ 모두 ScanNet++의 laser scan mesh를 사용. Self-supervised 또는 pseudo-GT (예: monocular normal/depth) 만으로 학습 가능한지 탐구 필요.
6. **Initializer의 transferability 추가 검증**: SAGS 외에 GOF [Yu et al., 2024], PGSR 등에도 적용 시 어떤 보편적 이점이 있는지 정량화.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ 아래 비교는 **QuickSplat 논문이 직접 인용한 참고문헌**에 한정합니다. 외부 검증을 거치지 않은 정보는 포함하지 않았습니다.

| 분류 | 대표 연구 | 핵심 아이디어 | QuickSplat과의 차이 |
|---|---|---|---|
| **Implicit NeRF/SDF** | NeRF [Mildenhall et al., 2020/21], Instant-NGP [Müller et al., 2022], Neuralangelo [Li et al., 2023], NeuS [Wang et al., 2021], MonoSDF [Yu et al., 2022] | MLP/grid 기반 implicit field를 per-scene 최적화 | MonoSDF는 정확도 유사하지만 **>10시간** 소요. QuickSplat은 124초. |
| **3DGS 계열 (NVS)** | 3DGS [Kerbl et al., 2023], 3DGS-LM [Höllein et al., 2024], FlashGS [Feng et al., 2024], Taming 3DGS [Mallick et al., 2024] | 명시적 Gaussian + 휴리스틱 densification | QuickSplat은 **휴리스틱을 학습으로 대체**하고 surface 품질에 초점. |
| **GS 기반 Surface** | SuGaR [Guédon & Lepetit, 2024], 2DGS [Huang et al., 2024], PGSR [Chen et al., 2024], GS2Mesh [Wolf et al., 2024], GOF [Yu et al., 2024] | GS를 표면에 정렬하거나 stereo depth 후처리 | GS2Mesh는 외부 stereo estimator 의존 → 렌더링 노이즈에 취약. QuickSplat은 GS 표현 공간 내에서 prior 학습. |
| **Sparse-view Feed-forward GS** | PixelSplat [Charatan et al., 2024], MVSplat [Chen et al., 2024], LaRa [Chen et al., 2024], GS-LRM [Zhang et al., 2024], DepthSplat [Xu et al., 2024], Long-LRM [Ziwen et al., 2024] | 2~수 장의 이미지에서 단일 forward pass로 GS 예측 | QuickSplat은 **임의 다수 이미지 + 반복 최적화** → 대규모 장면에 유리. |
| **Meta-learning 최적화** | Learning to learn [Andrychowicz et al., 2016], MAML [Finn et al., 2017], MetaSDF [Sitzmann et al., 2020], Learned Init [Tancik et al., 2021], G3R [Chen et al., 2024] | 신경망이 최적화 step 자체를 학습 | QuickSplat은 G3R의 idea를 GS에 적용하면서 **initializer + densifier**를 추가. |
| **Scene Completion** | SGNN [Dai et al., 2020] | Sparse 3D CNN으로 빈 공간 점군 채움 | QuickSplat의 initializer 백본으로 직접 차용. |
| **Structure-aware GS** | SAGS [Ververas et al., 2024], Scaffold-GS [Lu et al., 2024] | 구조적 anchor에 GS 부착 | Appendix C에서 SAGS에 QuickSplat initializer를 결합 가능함을 실증. |

핵심 포지셔닝: QuickSplat은 (i) **per-scene 최적화의 품질 우위**와 (ii) **feed-forward 모델의 속도 우위**를 모두 흡수한 **hybrid prior + iterative refinement** 형태로, 위 분류에서 어디에도 완벽히 속하지 않는 **새 카테고리**를 점한다.

---

## 6. 참고자료 (사용한 출처)

본 답변은 사용자가 제공한 PDF 한 편만을 1차 자료로 사용했습니다. 아래는 QuickSplat 논문이 본문에서 직접 인용한 참고문헌 중 본 답변에 등장한 것만 정리한 목록입니다 (논문 번호는 원문 표기 그대로).

- [4] Barron et al., *Mip-NeRF 360* (CVPR 2022)
- [5] Baruch et al., *ARKitScenes* (NeurIPS 2021 D&B)
- [7] Charatan et al., *pixelSplat* (CVPR 2024)
- [9] Chen et al., *LaRa* (ECCV 2024)
- [10] Chen et al., *PGSR* (TVCG 2024)
- [11] Chen et al., *G3R* (ECCV 2024)
- [12] Chen et al., *MVSplat* (arXiv 2024)
- [13] Choy et al., *Minkowski CNN* (CVPR 2019)
- [15] Dai et al., *SG-NN* (CVPR 2020)
- [17] Eftekhar et al., *Omnidata* (ICCV 2021)
- [18] Fang & Wang, *Mini-Splatting* (ECCV 2024)
- [19] Feng et al., *FlashGS* (arXiv 2024)
- [20] Finn et al., *MAML* (ICML 2017)
- [23] Guédon & Lepetit, *SuGaR* (CVPR 2024)
- [25] Höllein et al., *3DGS-LM* (arXiv 2024)
- [26] Huang et al., *2D Gaussian Splatting* (SIGGRAPH 2024)
- [27] Keetha et al., *SplaTAM* (CVPR 2024)
- [28] Kerbl et al., *3D Gaussian Splatting* (SIGGRAPH 2023)
- [30] Li et al., *Neuralangelo* (CVPR 2023)
- [31] Lu et al., *Scaffold-GS* (CVPR 2024)
- [32] Mallick et al., *Taming 3DGS* (arXiv 2024)
- [33] Matsuki et al., *Gaussian Splatting SLAM* (CVPR 2024)
- [34] Mildenhall et al., *NeRF* (Comm. ACM 2021)
- [35] Müller et al., *Instant-NGP* (TOG 2022)
- [38] Rota Bulò et al., *Revising Densification in GS* (ECCV 2024)
- [40] Sitzmann et al., *MetaSDF* (NeurIPS 2020)
- [42] Tancik et al., *Learned Initializations* (CVPR 2021)
- [44] Ververas et al., *SAGS* (arXiv 2024)
- [46] Wang et al., *NeuS* (arXiv 2021)
- [48] Wolf et al., *GS2Mesh* (ECCV 2024)
- [49] Xu et al., *DepthSplat* (arXiv 2024)
- [51] Yan et al., *GS-SLAM* (CVPR 2024)
- [52] Yang et al., *Depth Anything v2* (arXiv 2024)
- [55] Yeshwanth et al., *ScanNet++* (ICCV 2023)
- [57] Yu et al., *MonoSDF* (NeurIPS 2022)
- [58] Yu et al., *Gaussian Opacity Fields* (TOG 2024)
- [59] Zhang et al., *GS-LRM* (ECCV 2024)
- [61] Zhao et al., *DLNR* (CVPR 2023)
- [62] Ziwen et al., *Long-LRM* (arXiv 2024)

> 📌 **불확실성 표시**: 논문에 정량 결과가 명시되지 않은 비교(예: GS-LRM, MVSplat과 QuickSplat의 직접 정량 비교)는 위 표에서 **개념적 차이**로만 기술했고, 수치 비교는 의도적으로 하지 않았습니다. QuickSplat 논문 자체가 이러한 직접 비교 실험을 보고하지 않기 때문입니다.
