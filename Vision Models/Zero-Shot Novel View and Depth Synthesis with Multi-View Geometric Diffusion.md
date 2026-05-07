# MVGD: Zero-Shot Novel View and Depth Synthesis with Multi-View Geometric Diffusion

> **출처**: Guizilini, V., Irshad, M.Z., Chen, D., Shakhnarovich, G., Ambrus, R. (2025). *Zero-Shot Novel View and Depth Synthesis with Multi-View Geometric Diffusion*. CVPR 2025, pp. 764–776. arXiv:2501.18804v1.

---

## 1. 핵심 주장과 주요 기여 (간결 요약)

MVGD는 **NeRF, 3D Gaussian Splatting과 같은 중간 3D 표현(intermediate 3D representation)을 사용하지 않고**, 디퓨전 모델만으로 새로운 시점의 RGB 이미지와 Depth 맵을 *픽셀 레벨에서 직접* 생성하는 새로운 프레임워크입니다.

**주요 기여 3가지**:

1. **Multi-task 디퓨전 아키텍처**: 임의 개수(2~수천 장)의 posed 입력 이미지로부터 멀티뷰 일관성을 갖는 이미지와 깊이 맵을 동시에 합성
2. **60M+ 다중 도메인 데이터셋 학습 기법**: 19개의 이질적(indoor/outdoor/synthetic/dynamic) 데이터셋을 다루기 위한 Scene Scale Normalization(SSN), Task Embedding, 픽셀 레벨 디퓨전 등을 제안
3. **Incremental Fine-Tuning 전략**: 작은 모델의 latent token을 복제·확장하여 큰 모델로 점진적으로 학습. 처음부터 학습할 때 대비 **약 70% 학습 시간 단축**

---

## 2. 자세한 설명: 문제, 방법(수식), 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

기존 generalizable novel view synthesis는 다음 한계를 가집니다:
- **NeRF/3DGS 기반**: 새로운 장면 일반화 능력 제한, 입력 뷰가 많아질수록 성능 저하
- **Diffusion 기반(ZeroNVS, ReconFusion, CAT3D)**: 멀티뷰 일관성을 위해 보조적인 3D 재구성 파이프라인이 필요. 또한 깊이(geometry)는 부산물에 그침
- **공통 문제**: scale ambiguity, dynamic object, 다양한 카메라 캘리브레이션을 통합 학습하기 어려움

→ MVGD는 "암묵적(implicit) 분포에서 직접 샘플링"하여 이 문제를 우회합니다.

### 2.2 디퓨전 기본 수식

표준 forward process:

$$x_t = \sqrt{\alpha_t}\, x_0 + \sqrt{1-\alpha_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

여기서 $\alpha_t = \prod_{s=1}^{t}(1-\beta_s)$. MVGD가 학습하는 조건부 분포는:

$$f_\theta \sim p\!\left(\hat{I}_t,\, \hat{D}_t \mid \mathcal{C}_t,\, \mathcal{I}_C\right)$$

즉 입력 뷰 집합 $\mathcal{I}\_C = \{I, \mathcal{C}\}_{n=1}^{N}$과 타겟 카메라 $\mathcal{C}_t$가 주어지면, 타겟 이미지 $\hat{I}_t$와 깊이 $\hat{D}_t$를 함께 생성합니다.

### 2.3 Scene Scale Normalization (SSN) — 핵심 기여

서로 다른 데이터셋(LiDAR 메트릭 vs. COLMAP 임의 스케일)을 통합 학습하기 위한 핵심 기법입니다.

1. 모든 conditioning extrinsic을 타겟 기준 상대 좌표로 변환:
$$\tilde{T}_c^n = T_c^n \, T_t^{-1}$$

2. Scene scale을 카메라 translation의 최대 절댓값으로 정의:
$$s = \max\{ |\tilde{x}|, |\tilde{y}|, |\tilde{z}|\}_{n=1}^{N}$$

3. translation과 ground-truth depth를 정규화:
$$\tilde{t}_c^n = [\tfrac{x}{s},\, \tfrac{y}{s},\, \tfrac{z}{s}]^T, \quad \tilde{D}_t = D_t / s$$

4. 추론 시 예측된 depth는 다시 $s$를 곱해 conditioning camera와 동일 스케일로 복원

이를 통해 **scale-aware하고 multi-view consistent한** 깊이 예측이 가능해집니다(Ablation D, Table 5에서 SSN 제거 시 큰 성능 저하).

### 2.4 Depth 파라미터화 (log-scale)

$$P_D = 2 \cdot \frac{\log\!\left(\dfrac{D}{s \cdot d_{\min}}\right)}{\log\!\left(\dfrac{d_{\max}}{d_{\min}}\right)} - 1$$

$$\hat{D} = \exp\!\left(\dfrac{(2\hat{P}_D + 1)\log(d_{\max}/d_{\min})}{2}\right) \cdot d_{\min} \cdot s$$

$d_{\min}=0.1$, $d_{\max}=200$으로 실내/실외 모두 커버합니다.

### 2.5 모델 구조

```
[Conditioning Views] → Image Encoder (EfficientViT-SAM-L2) → E^I
                    → Ray Encoder (Fourier raymap) → E^R
                                                           → Scene Tokens E_c
[Target Camera]     → Ray Encoder → E^R_t
                    + Task Embedding E^task (image/depth)
                    + State Embedding S_t (noisy)
                                                           → Prediction Tokens

         RIN (Recurrent Interface Networks) Diffusion
            • Latent tokens Z ∈ R^{L×D} (256 ~ 2048)
            • Cross-attn(Z↔X) + Self-attn(Z) 반복
                    ↓
              denoise → ÎI_t, D̂_t
```

**핵심 설계**:
- **RIN(Recurrent Interface Networks)**: Self-attention의 $O(N^2)$ 복잡도를 $O(L^2)$ ($L \ll N$)로 낮춰 픽셀 레벨 디퓨전을 가능하게 함 (auto-encoder 불필요)
- **Raymap conditioning**: 픽셀 $p_{ij}$마다 origin $t_{ijk}$ + view direction $r_{ijk} = (K_k R_k)^{-1}[u_{ij}, v_{ij}, 1]^T$ 를 Fourier 인코딩
- **Task Embedding** $E^{\text{task}} \in \mathbb{R}^{D_{\text{task}}}$: 동일한 latent를 image 또는 depth 출력으로 분기. 깊이 라벨 없는 데이터셋도 RGB 학습에 활용 가능

### 2.6 성능 향상

| 벤치마크 | 입력 뷰 | 비교 SOTA | MVGD |
|---|---|---|---|
| RealEstate10K | 2-view PSNR | MVSplat 26.39 | **28.41** |
| RealEstate10K | 3-view PSNR | CAT3D 26.78 | **28.70** |
| MipNeRF360 | 9-view PSNR | CAT3D 18.67 | **21.18** |
| ScanNet (Stereo Depth) | AbsRel | GRIN 0.088 | **0.065** |
| ScanNet (Video Depth, 10view) | AbsRel | NeuralRecon 0.047 | **0.041** |

추가로 **Incremental Fine-Tuning**: 256→2048 latent로 늘릴 때 파라미터는 +0.5%만 증가하지만 성능은 최대 +20% 개선.

### 2.7 한계 (논문 명시)

1. **SSN의 구조적 한계**: 타겟 카메라가 항상 원점이라고 가정 → 여러 viewpoint에서 동시에 생성 불가
2. **Dynamic object 명시적 모델링 부재**: 암묵적 dynamics 학습은 일부 관찰됨(Fig. 6, DDAD 데이터)
3. **추론 속도**: 2048 latent + 1250 conditioning views 시 약 20초/생성
4. **카메라 정확도 의존**: posed image를 가정하므로 SfM이 부정확하면 성능 저하 우려
5. 대규모 dynamic 멀티뷰 데이터셋 부족이 future foundation model의 병목

---

## 3. 일반화 성능 향상 가능성 (중점 분석)

MVGD가 zero-shot 일반화에 강한 4가지 메커니즘:

### (1) 데이터 스케일과 다양성
60M+ 샘플(19개 데이터셋)을 driving/indoors/robotics/synthetic 전반에 걸쳐 통합 학습. Ablation E(dynamic 데이터 제거)에서 **in-domain 벤치마크에서도 성능 저하**가 관찰되어, 다양성이 일반화에 직접 기여함을 입증합니다.

### (2) Scene Scale Normalization → 데이터셋 간 호환성 확보
서로 다른 캘리브레이션 출처(LiDAR, IMU, COLMAP)에서 온 데이터를 단일 분포로 통합. 이는 ZeroNVS의 "depth-scale ambiguity" 해결책과 유사한 동기지만, 더 단순하고 데이터셋 메타데이터에 비의존적입니다.

### (3) RIN의 입력 비의존성
계산 복잡도가 conditioning view 수에 거의 독립적($O(L^2)$, $L$ 고정). **2~5뷰로 학습한 모델이 100+ 뷰까지 직접 일반화** (Fig. 7). 이는 transformer 기반 SOTA가 갖지 못한 강력한 inductive 일반화입니다.

### (4) Multi-task Joint Learning이 Geometric Prior 강화
Ablation에서 image-only 모델은 multi-task 모델보다 PSNR이 크게 낮음 → "**geometric reasoning이 NVS 품질의 핵심 요소**"임을 정량 입증. Depth 합성을 강제함으로써 모델이 implicit 3D 일관성을 학습.

### (5) Incremental Fine-Tuning이 가져오는 Scaling Property
처음부터 2048 latent로 학습한 모델보다 **256→2048 점진적 확장 모델이 더 우수**. 작은 모델의 prior가 큰 모델로 효율적으로 전이됨을 시사하며, 향후 foundation model 스케일업에 적용 가능한 일반 전략입니다.

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 영향
- **"3D representation-free" 패러다임**의 강력한 검증. NeRF/3DGS 의존 없이도 SOTA가 가능함을 입증하여, 이후 Stable Virtual Camera(SEVA), 4DiM, ViewCrafter, MEt3R 등 후속 연구의 기준점이 됨
- **Geometric Foundation Model**의 가능성 제시: 단일 모델이 NVS + Stereo depth + Video depth 모두 SOTA
- **Incremental fine-tuning** 전략은 LLM 분야의 "weight expansion" 기법과 유사하게 비전 분야에서도 일반화 가능

### 4.2 향후 연구 시 고려할 점
1. **Dynamic Scene 명시적 모델링**: 시간 토큰 또는 motion token 추가 (저자가 STORM[2] 인용). 자율주행/4D 콘텐츠 생성에 필수
2. **Pose-free 확장**: 현재는 posed image 가정. 향후 SfM-free 방향(예: DUSt3R, MASt3R 계열)과의 결합이 필요
3. **추론 속도 개선**: distillation, consistency model, flow matching 적용 가능
4. **다중 타겟 동시 생성**: SSN의 "타겟=원점" 가정 완화 필요
5. **공간-시간 일관성 평가 메트릭**: MEt3R 같은 새로운 평가 프레임워크와의 결합
6. **Scale의 정확성 검증**: SSN은 conditioning camera scale을 그대로 따름. 이 scale 자체가 부정확할 때 MVGD 깊이도 함께 부정확해짐 → 메트릭 깊이 응용에서는 추가 보정 필요
7. **Long-context 효율화**: 현재 incremental conditioning은 모든 과거 뷰를 사용하므로, near-view selection heuristic 등 효율화 여지가 큼

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 모델 (연도) | 접근 | 중간 3D 표현 | 멀티뷰 일관성 보장 | 깊이 동시 생성 | 일반화 |
|---|---|---|---|---|---|
| **PixelNeRF** (2021) | Feed-forward NeRF | NeRF | △ | ✗ | 약함 |
| **Zero-1-to-3** (2023) | Conditional Diffusion (1-view) | 없음 | ✗ | ✗ | 객체 중심 |
| **ZeroNVS** (CVPR 2024) | SDS + 3D-aware Diffusion | NeRF (SDS) | △ | ✗ | scene-level zero-shot |
| **PixelSplat** (CVPR 2024) | Feed-forward 3DGS | 3D Gaussians | ✓ | ✗ (간접) | 2-view 한정 |
| **MVSplat** (ECCV 2024) | Cost volume + 3DGS | 3D Gaussians | ✓ | ✗ (간접) | 2-view 한정 |
| **ReconFusion** (CVPR 2024) | Diffusion + PixelNeRF | NeRF | ✓ | ✗ | 3~9 view |
| **CAT3D** (NeurIPS 2024) | Multi-view Latent Diffusion | NeRF (사후) | ✓ | ✗ | 1~9 view |
| **MVGD** (CVPR 2025) | **RIN Pixel Diffusion** | **없음** | ✓ (암묵적) | **✓** | **2~수천 view, zero-shot** |
| **Stable Virtual Camera/SEVA** (2025) | Video Diffusion 기반 | 없음 | ✓ | ✗ | 강함 |

**핵심 차별점**:
- PixelSplat/MVSplat은 깊이/3D Gaussian으로 일관성을 강제하지만 **뷰 수 확장성이 약함**
- ReconFusion/CAT3D는 멀티뷰 latent diffusion이지만 **여전히 NeRF를 사후 학습**으로 사용
- MVGD는 두 계열의 단점(중간 표현 의존, 뷰 수 한계)을 모두 우회하면서 **깊이까지 동시 생성**하는 유일한 방법
- 정량적으로 MVGD는 RealEstate10K 3-view에서 PSNR 28.70으로 CAT3D(26.78)을 2 dB 능가

---

## 참고자료 / 출처

1. **원 논문**: Guizilini et al., *Zero-Shot Novel View and Depth Synthesis with Multi-View Geometric Diffusion*, CVPR 2025. arXiv:2501.18804v1. https://arxiv.org/abs/2501.18804
2. **프로젝트 페이지**: https://mvgd.github.io/
3. **CVPR Open Access**: https://openaccess.thecvf.com/content/CVPR2025/html/Guizilini_Zero-Shot_Novel_View_and_Depth_Synthesis_with_Multi-View_Geometric_Diffusion_CVPR_2025_paper.html
4. **MarkTechPost 해설**: "MVGD from Toyota Research Institute: Zero Shot 3D Scene Reconstruction" (2025-03)
5. **CAT3D**: Gao et al., *CAT3D: Create Anything in 3D with Multi-View Diffusion Models*, NeurIPS 2024. https://cat3d.github.io/, arXiv:2405.10314
6. **ReconFusion**: Wu et al., *ReconFusion: 3D Reconstruction with Diffusion Priors*, CVPR 2024
7. **ZeroNVS**: Sargent et al., *ZeroNVS: Zero-Shot 360-Degree View Synthesis from a Single Real Image*, CVPR 2024 (OpenReview ID: cIgfXQBExO)
8. **PixelSplat**: Charatan et al., *PixelSplat: 3D Gaussian Splats from Image Pairs*, CVPR 2024. arXiv:2312.12337
9. **MVSplat**: Chen et al., *MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-view Images*, ECCV 2024. arXiv:2403.14627. https://donydchen.github.io/mvsplat/
10. **Stable Virtual Camera (SEVA)**: 2025, Generative View Synthesis with Diffusion Models (비교 표 참조)
11. **RIN**: Jabri, Fleet, Chen, *Scalable Adaptive Computation for Iterative Generation*, ICML 2023
12. **EfficientViT-SAM**: Cai, Gan, Han, *EfficientViT*, arXiv:2205.14756

---

> **신뢰도 관련 알림**: 위 비교 표의 정량 수치(PSNR 등)와 모델별 발표 시점은 원 논문 Table 2–7 및 검색된 공식 자료에서 확인 가능한 범위에서만 인용했습니다. 일부 후속 연구(예: Stable Virtual Camera, MEt3R, Pointmap-Conditioned Diffusion 등)의 세부 수치는 빠르게 업데이트되고 있으므로, 최종 비교 시 각 논문의 최신 버전과 공식 리더보드를 직접 확인하시길 권장합니다.
