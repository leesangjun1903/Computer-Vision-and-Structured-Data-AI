# No Pose, No Problem: Surprisingly Simple 3D Gaussian Splats from Sparse Unposed Images

이 논문은 카메라 포즈가 전혀 주어지지 않은 매우 적은 수의 이미지만으로도, **하나의 feed-forward 네트워크로 3D Gaussian splat 장면을 직접 재구성하고 NVS와 포즈 추정을 동시에 잘 할 수 있다**는 것을 보이는 것이 핵심 주장입니다. 특히, 기존처럼 “포즈 추정 → 3D 재구성”의 순차 파이프라인이 아니라 **정렬 기준이 되는 canonical 공간에서 바로 3D Gaussian을 예측**함으로써, 소수 뷰·작은 overlap에서도 포즈 기반 SOTA보다 뛰어난 품질과 일반화 성능을 달성했다고 주장합니다.[^1][^2][^3]

***

## 1. 핵심 주장과 주요 기여

- 저자들은 NoPoSplat이라는 모델이 **포즈가 주어지지 않은 sparse multi-view 이미지(2장 수준)와 카메라 내파라미터만으로** canonical 공간에서 3D Gaussian들을 직접 예측할 수 있고, 이로부터 고품질 novel view synthesis(NVS)와 상대 포즈 추정을 동시에 수행할 수 있다고 주장합니다.[^2][^3][^1]
- 이 과정에서 **SfM/COLMAP 기반 포즈 계산이나 깊이 supervision 없이, 순수 photometric loss(MSE+LPIPS)만으로 대규모 비디오 데이터에서 학습이 가능**하다는 점을 중요한 기여로 내세웁니다.[^3][^1][^2]
- 주요 기여로는 (1) canonical Gaussian space 기반의 완전 pose-free 3DGS 재구성, (2) scene scale ambiguity를 해결하기 위한 간단하지만 효과적인 카메라 intrinsic token 설계, (3) 예측된 Gaussians를 이용한 두 단계 pose estimation 파이프라인, (4) RE10K/ACID/DTU/ScanNet++ 등에서 pose-free/pose-required SOTA를 상회하는 NVS 및 포즈 추정 성능과 강한 zero-shot 일반화가 제시됩니다.[^1][^2][^3]

***

## 2. 해결하고자 하는 문제

- 현존 NeRF·3DGS 기반 sparse‑view 재구성은 **정확한 카메라 포즈(대개 dense 비디오+SfM)와 상당한 뷰 overlap, 그리고 기하 priors(코스트 볼륨, epipolar geometry 등)에 강하게 의존**하기 때문에, 실제 사용자 입력처럼 “몇 장의 임의 사진만 있는” 상황에서는 잘 동작하지 않습니다.[^2][^3][^1]
- 최근 pose-free 방법들은 포즈 추정과 재구성을 통합하거나 반복적으로 번갈아 최적화하지만, **포즈 추정 오차가 scene 재구성 품질을 떨어뜨리고, 다시 나빠진 재구성이 포즈 추정에 악영향을 주는 error compounding 문제**로 인해 여전히 pose-required 방법보다 성능이 떨어지는 한계를 보였습니다.[^4][^5][^3][^1][^2]
- 이 논문이 노리는 문제는 “**포즈를 전혀 추정하지 않고도** 소수의 unposed 이미지에서 3DGS를 일반화 가능하게 재구성하고, 기존 pose-required SOTA 대비 동등 이상 성능과 더 나은 일반화(특히 작은 overlap·out-of-distribution)를 달성할 수 있는가?”입니다.[^3][^1][^2]

***

## 3. 제안하는 방법 개요

- 입력은 뷰 수 $V$의 **unposed RGB 이미지와 각 이미지의 카메라 내파라미터** $\{(I_v, k_v)\}_{v=1}^V$이고, 출력은 canonical 공간에 놓인 픽셀 단위 3D Gaussian primitive들의 집합입니다.[^1][^2][^3]
- 첫 번째 뷰의 카메라 좌표계를 canonical(world) 좌표계로 고정하고, **모든 뷰의 픽셀별 Gaussian center, scale, orientation, opacity, SH 계수 등을 이 canonical 좌표계 기준으로 직접 회귀**하도록 학습합니다.[^2][^3][^1]
- 학습은 렌더링된 novel view와 GT 이미지 사이의 **MSE + 0.05·LPIPS**로만 수행하며, 그 외 깊이·포인트맵·feature matching supervision은 사용하지 않고, 학습된 Gaussian 장(field)을 이용해 별도의 두 단계 pose estimation(PnP→photometric refinement)을 정의합니다.[^3][^1][^2]

***

## 4. 수식 관점에서의 정리

### 4.1 입력–출력 매핑

모델은 다음과 같은 매핑을 학습합니다.[^1]

$$
f_\theta : \{(I_v, k_v)\}_{v=1}^V \;\longrightarrow\; \bigcup_{v=1}^V \bigcup_{j=1}^{HW} \{ \mu_{vj}, \alpha_{vj}, r_{vj}, s_{vj}, c_{vj} \},
$$

여기서 각 Gaussian primitive는
$\mu\in\mathbb{R}^3$ (center), $\alpha\in\mathbb{R}$ (opacity), $r\in\mathbb{R}^4$ (quaternion rotation), $s\in\mathbb{R}^3$ (scale), $c\in\mathbb{R}^k$ (SH 계수)로 구성됩니다.[^1]

canonical 공간에서의 예측을 강조하기 위해, 뷰 $v$의 픽셀 $p_j$에 대응하는 Gaussian 파라미터를 $\{\mu^{v\to 1}_j, r^{v\to1}_j, c^{v\to1}_j, \alpha_j, s_j\}$와 같이 **“뷰 1 좌표계 기준”**으로 표기합니다.[^1]

### 4.2 카메라 intrinsic 임베딩

카메라 내파라미터 $k = [f_x, f_y, c_x, c_y]$를 임베딩하는 세 가지 방식을 제안하며, 기본 선택은 **global intrinsic token concat**입니다.[^1]

- Global-add:

$$
e_k = W k,\quad \tilde{x}_{v,j} = x_{v,j} + e_k
$$
- Global-concat(기본):

$$
X_v = [t_k, x_{v,1}, x_{v,2}, \dots, x_{v,HW}],
$$

여기서 $t_k = W k$는 하나의 token으로, image patch 토큰 시퀀스에 concat 됩니다.[^1]
- Dense embedding: 각 픽셀에 대해

$$
r_j = K^{-1} p_j
$$

를 구한 뒤, 이를 SH 임베딩해 RGB와 concat하여 입력으로 사용합니다.[^1]

실험적으로는 **intrinsic token concat 방식이 PSNR/SSIM/LPIPS에서 가장 좋은 성능을 보여 기본 설정으로 채택됩니다.**[^1]

### 4.3 학습 손실

렌더링 연산자를 $\mathcal{R}(\cdot; T)$ (입력 Gaussians와 카메라 포즈 $T$를 받아 이미지를 생성)라 하고, target 이미지 $I^{\text{gt}}$, 렌더링 결과 $\hat{I}$에 대해 손실은

$$
\hat{I} = \mathcal{R}( \{\mu,\alpha,r,s,c\}; T_{\text{target}} ),
$$

$$
\mathcal{L}_{\text{rgb}} = \| \hat{I} - I^{\text{gt}} \|_2^2 + 0.05 \cdot \text{LPIPS}(\hat{I}, I^{\text{gt}}),
$$

와 같이 정의되며, 모든 supervision은 이 photometric 손실만으로 주어집니다.[^1]

### 4.4 포즈 추정 단계

포즈 추정은 두 단계로 구성됩니다.[^1]

1. **PnP 초기화**: 추정된 Gaussian center $\{\mu_j\}$와 대응되는 관측 픽셀 $\{p_j\}$를 이용하여 PnP+RANSAC으로 상대 포즈 $\hat{T}_0$를 계산합니다.[^1]
2. **photometric refinement**: Gaussian 파라미터는 고정하고, 카메라 포즈 $T$만을 변수로 하여

$$
\mathcal{L}_{\text{pose}}(T) = \mathcal{L}_{\text{rgb}}(\mathcal{R}(\{\mu,\alpha,r,s,c\}; T), I^{\text{gt}}) + \lambda \, (1-\text{SSIM}_{\text{struct}}(\hat{I}, I^{\text{gt}}))
$$

를 gradient descent로 최적화합니다.[^1]

이로써 **ground-truth depth, explicit matching loss 없이**도 강력한 relative pose estimation 성능을 얻었다고 보고합니다.[^1]

***

## 5. 모델 구조

- 백본은 **pure Vision Transformer encoder–decoder 구조**이며, encoder는 ViT-L, decoder는 ViT-B로 구성되고, 모든 뷰에 대해 encoder는 weight를 공유하며 각 뷰의 토큰 시퀀스는 decoder에서 cross-attention으로 상호 작용합니다.[^3][^1]
- Gaussian parameter head는 DPT 스타일 두 개의 head로 구성되어, 하나는 decoder feature만을 사용해 center를, 다른 하나는 decoder feature + RGB shortcut을 함께 사용해 나머지 파라미터($\alpha, r, s, c$)를 예측하는데, RGB shortcut이 없으면 텍스처가 풍부한 영역에서 블러가 증가하는 것으로 보고됩니다.[^1]
- 가장 중요한 구조적 특징은 **local-to-global transform-then-fuse가 아니라, 처음부터 canonical 공간에서 바로 Gaussians를 예측**한다는 점으로, 이를 통해 포즈 입력 필요성 제거와 동시에, multi-view 정보가 하나의 일관된 글로벌 표현으로 직접 aggregation된다는 장점을 가집니다.[^1]

***

## 6. 성능 향상 및 한계

- RE10K/ACID에서, NoPoSplat은 **기존 pose-free 방법(DUSt3R, MASt3R, CoPoNeRF 등)보다 PSNR·SSIM에서 큰 폭으로 우수**하며, 소형 overlap 구간에서는 pixelSplat, MVSplat 같은 pose-required sparse-view 3DGS보다도 높은 PSNR/SSIM과 더 낮은 LPIPS를 기록합니다.[^1]
- DTU, ScanNet++와 같은 out-of-distribution 데이터셋에서도, **RE10K만으로 학습한 모델이 pixelSplat·MVSplat 등 pose-required 방법보다 높은 PSNR/SSIM·더 낮은 LPIPS를 보이며**, 포즈 추정에서도 RoMa, DUSt3R, CoPoNeRF를 포함한 SOTA보다 높은 AUC(@5°, 10°, 20°)를 달성합니다.[^1]
- 한편 한계로는 (1) **카메라 내파라미터가 필요**하다는 점(실험에서는 EXIF 또는 휴리스틱으로 근사), (2) feed-forward 비생성 구조라 **입력에서 보이지 않는 영역의 geometry/texture hallucination 능력이 제한**된 점, (3) 학습 데이터(RE10K, ACID, DL3DV)에 편중되어 있어 매우 다양한 in-the-wild 장면에 대한 완전한 일반화는 아직 제한적이라는 점 등이 명시됩니다.[^1]

***

## 7. 일반화 성능 향상 가능성에 대한 분석

- 저자들은 NoPoSplat이 **기하 priors 없이도 순수 ViT backbone + photometric loss만으로 다양한 장면 분포에 잘 generalize**한다고 강조하며, 특히 DL3DV와 같은 다양한 카메라 모션을 포함한 데이터셋을 추가하면 포즈 추정 AUC가 꾸준히 향상된다는 실험 결과를 제시합니다.[^1]
- canonical Gaussian space와 pose-free 설계 덕분에 **기존의 포즈 estimation 오차 누적 문제로부터 자유롭고, transform-then-fuse에서 발생하던 misalignment·ghosting·non-overlap 영역 기하 붕괴 현상이 줄어들어** out-of-distribution generalization에서도 안정적인 geometry를 유지할 수 있음을 시각적·정량적으로 보여줍니다.[^1]
- 더 나아가, **모바일 사진과 Sora 생성 비디오 프레임에 대한 qualitative 결과에서 별도 fine-tuning 없이도 plausible한 3D 재구성과 novel view를 생성**하는 예시를 통해, 실제 사용자 시나리오(텍스트/이미지→multi-view→3DGS)에서의 실용적 generalization 가능성을 시사합니다.[^3][^1]

***

## 8. 2020년 이후 관련 최신 연구 비교 분석

아래는 NoPoSplat과 직접적으로 연관된, 2020년 이후의 대표적인 open-access 연구들입니다(제목·저자·링크·2줄 요약).

### 8.1 포즈 기반 NeRF/3DGS 및 일반화 NeRF

- **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis** – Mildenhall et al., ECCV 2020.[^1]
    - 수백 장의 포즈가 알려진 이미지를 필요로 하는 per-scene 최적화 기반 NeRF로, 이후 모든 신경 렌더링 연구의 출발점입니다.[^1]
- **pixelNeRF: Neural Radiance Fields from One or Few Images** – Yu et al., CVPR 2021.[^1]
    - 카테고리 수준의 generalizable NeRF로, 몇 장의 포즈가 주어진 이미지만으로 unseen scene의 radiance field를 추정하며, pose-required generalization의 초기 형태를 제시합니다.[^1]
- **MVSNeRF / MuRF / IBRNet / MVSplat / pixelSplat** – Chen et al. 2021–2024, Charatan et al. 2024 등.[^2][^3][^1]
    - cost volume, epipolar geometry, multi-baseline aggregation 등 다양한 기하 priors를 사용한 sparse-view generalizable NeRF/3DGS 계열로, **정확한 포즈와 충분한 image overlap을 강하게 가정**한다는 점에서 NoPoSplat의 직접적인 비교 대상입니다.[^2][^3][^1]


### 8.2 포즈-프리 NeRF/재구성

- **NeRF--: Neural Radiance Fields without Known Camera Parameters** – Wang et al., 2021.[^1]
    - rough pose 초기값을 필요로 하면서도 포즈와 NeRF를 joint optimization하는 방식으로, 포즈 없이도 scene 재구성이 가능함을 초기에 보여줍니다.[^1]
- **DBARF / BARF 계열** – Chen \& Lee 2023 등.[^4][^1]
    - bundle-adjusting NeRF로, pose refinement와 NeRF 학습을 통합하지만, 여전히 optimization-heavy하고 generalization보다는 단일 장면 안정화에 초점이 있습니다.[^1]
- **FlowCam, CoPoNeRF: Unifying Correspondence, Pose and NeRF for Generalized Pose-Free Novel View Synthesis** – Hong et al., CVPR 2024.[^5][^6][^7][^8][^4][^1]
    - correspondence, pose estimation, NeRF를 하나의 통합 프레임워크에서 학습하여 pose-free NVS를 달성하지만, 여전히 pose estimation 모듈과 NeRF 사이의 상호 의존성으로 인한 error compounding 문제가 존재하며, NoPoSplat이 같은 설정에서 더 높은 NVS 성능을 보여줍니다.[^2][^3][^1]


### 8.3 DUSt3R / MASt3R 계열 (포인트맵 기반)

- **DUSt3R: Geometric 3D Vision Made Easy** – Wang et al., CVPR 2024.[^9][^10][^11][^12]
    - 포인트맵(pointmap)을 회귀하여 카메라 보정 없이도 dense stereo 3D reconstruction, depth, pose estimation을 통합적으로 수행하는 방법으로, pose-free 3D vision의 중요한 이정표입니다.[^10][^12][^9]
- **MASt3R: Grounding Image Matching in 3D with MASt3R** – Leroy et al., 2024.[^1]
    - DUSt3R의 아이디어를 기반으로 feature matching을 강화한 모델로, NoPoSplat의 ViT 및 Gaussian center head 초기화에 사용되며, Splatt3R 등 후속 3DGS에서도 핵심 역할을 합니다.[^1]
- **연관성**: DUSt3R/MASt3R는 **3DGS가 아니라 포인트맵/포인트클라우드 기반**이지만, pose-free 3D reconstruction의 핵심 베이스라인으로 NoPoSplat과 모든 task(깊이, 포즈, NVS)에서 비교됩니다.[^12][^9][^10][^1]


### 8.4 Pose-free 3D Gaussian Splatting 계열

- **Colmap-free 3D Gaussian Splatting** – Fu et al., CVPR 2024.[^1]
    - COLMAP 없이도 3DGS를 학습하는 방법이지만, 비디오 시퀀스 입력과 특정 제약이 존재하며, canonical space에서 직접 Gaussians를 예측하는 NoPoSplat과는 접근 방식이 다릅니다.[^1]
- **Splatt3R: Zero-shot Gaussian Splatting from Uncalibrated Image Pairs** – Smart et al., 2024.[^1]
    - frozen MASt3R에서 얻은 포인트를 기반으로 Gaussian center를 정하는 방식으로, 여전히 depth supervision과 MASt3R의 한계(뷰 간 내용 merge 문제)에 의존하며, NoPoSplat보다 NVS 성능과 generalization이 떨어지는 것으로 보고됩니다.[^1]
- **PF3plat: Pose-Free Feed-Forward 3D Gaussian Splatting for Novel View Synthesis** – 2024.[^13]
    - sparse unposed 이미지에서 feed-forward로 3DGS를 예측한다는 점에서 NoPoSplat과 매우 유사한 목표를 가진 동시대 연구로, 자세한 구현은 다르지만 pose-free 3DGS의 feed-forward화라는 공통 방향성을 가집니다.[^13]
- **SPFSplat: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views** – Huang \& Mikolajczyk, 2025.[^14]
    - novel-view 예측 포즈와 reprojection loss를 결합한 self-supervised pose-free 3DGS로, canonical space에서 3D Gaussian과 포즈를 동시에 예측하며, NoPoSplat 이후 self-supervised, pose-free 방향으로의 확장을 보여줍니다.[^14]
- **SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting** – Kang et al., CVPR 2025.[^15][^16]
    - explicit 3D priors 없이도 self-supervised depth·pose estimation을 내장한 3DGS를 제안하여, NoPoSplat이 사용하는 pre-trained MASt3R 초기화/3D priors 의존성을 줄이려는 방향의 후속 연구로 볼 수 있습니다.[^16][^15]

이들 연구와 비교했을 때, NoPoSplat의 특징은

1) **canonical Gaussian space에서의 직접 예측**,
2) **photometric loss only** (깊이·matching supervision 없음),
3) **일반 ViT backbone + 간단한 intrinsic token으로도 pose-required SOTA를 뛰어넘는 NVS/pose 성능**을 보였다는 점이며, 이는 이후 PF3plat, SPFSplat, SelfSplat 등이 보다 self-supervised·prior-free 방향으로 확장하는 데 중요한 출발점이 됩니다.[^15][^16][^14][^13][^3][^2][^1]

***

## 9. 앞으로의 연구에 미치는 영향과 고려할 점

- NoPoSplat은 “**포즈 추정 자체를 모델 외부에서 별도로 하지 않고, canonical 3D 표현을 바로 학습하면 더 잘 된다**”는 실증을 통해, 앞으로 sparse-view 3DGS/NeRF 연구에서 **pose-free canonical representation learning**이 핵심 연구 축이 될 수 있음을 보여줍니다.[^3][^2][^1]
- 또한 photometric loss만으로도 강한 generalization이 가능함을 보였지만, 여전히 MASt3R 등의 3D-pretrained 모델 초기화에 의존하고 있으므로, **SelfSplat, SPFSplat처럼 완전 self-supervised·3D-prior-free pose-free 3DGS**로 나아가기 위한 중간 단계로 이해할 수 있고, 향후에는 더 대규모·다양한 데이터셋에서의 self-supervised scaling이 중요한 과제가 됩니다.[^16][^14][^15][^1]
- 실용적 관점에서는 (1) 카메라 intrinsics 추정의 자동화·robustification, (2) 동적 장면·비강체 객체를 다루는 dynamic pose-free 3DGS, (3) text/video diffusion 모델과의 end-to-end 결합을 통한 “텍스트→비디오→3DGS” 파이프라인 최적화, (4) 소수 뷰에서의 강력한 geometry hallucination 능력을 갖춘 생성적 pose-free 3DGS 설계 등이 앞으로 연구 시 특히 고려해야 할 중요한 방향으로 보입니다.[^14][^15][^16][^2][^3][^1]
<span style="display:none">[^17][^18][^19][^20][^21][^22][^23][^24]</span>

<div align="center">⁂</div>

[^1]: 2410.24207v1.pdf

[^2]: https://arxiv.org/html/2410.24207v1

[^3]: https://www.research-collection.ethz.ch/server/api/core/bitstreams/5c49ef7e-e478-4ed4-b7cc-eafbe54e0ad6/content

[^4]: https://arxiv.org/abs/2312.07246

[^5]: https://openaccess.thecvf.com/content/CVPR2024/papers/Hong_Unifying_Correspondence_Pose_and_NeRF_for_Generalized_Pose-Free_Novel_View_CVPR_2024_paper.pdf

[^6]: https://arxiv.org/html/2312.07246v2

[^7]: https://ku-cvlab.github.io/CoPoNeRF/

[^8]: https://pure.kaist.ac.kr/en/publications/unifying-correspondence-pose-and-nerf-for-generalized-pose-free-n-2/

[^9]: https://arxiv.org/abs/2312.14132

[^10]: https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_DUSt3R_Geometric_3D_Vision_Made_Easy_CVPR_2024_paper.pdf

[^11]: https://arxiv.org/html/2312.14132v2

[^12]: https://huggingface.co/papers/2312.14132

[^13]: https://arxiv.org/html/2410.22128v2

[^14]: https://arxiv.org/html/2508.01171v1

[^15]: https://arxiv.org/html/2411.17190v5

[^16]: https://openaccess.thecvf.com/content/CVPR2025/papers/Kang_SelfSplat_Pose-Free_and_3D_Prior-Free_Generalizable_3D_Gaussian_Splatting_CVPR_2025_paper.pdf

[^17]: https://arxiv.org/abs/2410.24207

[^18]: https://arxiv.org/html/2503.01661v1

[^19]: https://www.semanticscholar.org/paper/No-Pose,-No-Problem:-Surprisingly-Simple-3D-Splats-Ye-Liu/9f23a9901473784b1d7546524bd4b7eaf244fe92

[^20]: https://www.semanticscholar.org/paper/DUSt3R:-Geometric-3D-Vision-Made-Easy-Wang-Leroy/5f82a81766cb78395a55b8fc697c2421a20f4a9e

[^21]: https://github.com/KU-CVLAB/CoPoNeRF

[^22]: https://liner.com/review/no-pose-no-problem-surprisingly-simple-3d-gaussian-splats-from

[^23]: https://research.aalto.fi/en/publications/dust3r-geometric-3d-vision-made-easy/

[^24]: https://noposplat.github.io

