# DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos

## 1. 핵심 주장 및 주요 기여 (요약)

DepthCrafter는 카메라 포즈, 광학 흐름(optical flow) 등 **추가 정보 없이** open-world 비디오에 대해 시간적으로 일관되고 세밀한 디테일을 갖춘 **장기(long) depth 시퀀스**를 생성하는 방법입니다. 사전 학습된 image-to-video diffusion 모델을 video depth estimation에 활용하면서 open-world 비디오에 대한 일반화 성능을 유지하도록 적응시킨 것이 핵심 아이디어입니다.

주요 기여를 정리하면 다음과 같습니다:

- **변동 길이 장기 컨텍스트 지원**: 한 번에 최대 110프레임까지 가변적인 길이의 depth 시퀀스를 생성 가능 (기존 SVD는 25프레임, ChronoDepth는 10프레임)
- **3-stage training 전략**: 현실적(realistic) 데이터셋과 합성(synthetic) 데이터셋의 장점을 모두 흡수
- **세그먼트 기반 추론(inference) 전략**: 110프레임을 초과하는 매우 긴 비디오도 매끄럽게 stitching하여 처리
- 4개 비디오 데이터셋(Sintel, ScanNet, KITTI, Bonn)에서 zero-shot SOTA 성능 달성

---

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

Open-world 비디오는 외형, 콘텐츠 움직임, 카메라 움직임, 길이가 매우 다양합니다. 기존 접근 방식들의 한계는:

1. **이미지 기반 단일 프레임 depth 모델(Depth-Anything, Marigold 등)**을 비디오에 직접 적용하면 시간적 비일관성(flickering)이 발생
2. **기존 비디오 depth 방법들**은 카메라 포즈나 optical flow에 의존하거나 동적 콘텐츠 비율에 민감하여 open-world에서 성능 저하
3. **Diffusion 기반 비디오 모델**은 일반적으로 고정된 짧은 프레임 수만 생성 가능

저자들이 정의한 세 가지 핵심 도전 과제:
1. 일반화를 위한 비디오 콘텐츠의 포괄적 이해
2. 전체 depth 분포를 정확히 배치하기 위한 길고 가변적인 temporal context
3. 매우 긴 비디오 처리 능력

### 2.2 제안 방법 (수식 포함)

문제를 **조건부 diffusion 생성 문제** $p(\mathbf{d} \mid \mathbf{v})$로 정식화합니다. 여기서 $\mathbf{v} \in \mathbb{R}^{T \times H \times W \times 3}$는 입력 비디오, $\mathbf{d} \in \mathbb{R}^{T \times H \times W}$는 depth 시퀀스입니다.

**Diffusion 과정 (EDM 프레임워크 기반)**:

$$\mathbf{x}_t = \mathbf{x}_0 + \sigma_t^2 \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

**Denoising Score Matching 학습 목적함수**:

$$\mathbb{E}_{\mathbf{x}_t \sim p(\mathbf{x}; \sigma_t),\, \sigma_t \sim p(\sigma)} \left[ \lambda_{\sigma_t} \left\| D_\theta(\mathbf{x}_t; \sigma_t; \mathbf{c}) - \mathbf{x}_0 \right\|_2^2 \right]$$

**EDM 프리컨디셔닝(preconditioning)**:

$$D_\theta(\mathbf{x}_t; \sigma_t; \mathbf{c}) = c_{\text{skip}}(\sigma_t)\,\mathbf{x}_t + c_{\text{out}}(\sigma_t)\,F_\theta\bigl(c_{\text{in}}(\sigma_t)\,\mathbf{x}_t;\, c_{\text{noise}}(\sigma_t);\, \mathbf{c}\bigr)$$

구체적으로 각 함수는 다음과 같이 정의됩니다:

$$c_{\text{in}}(\sigma_t) = \frac{1}{\sqrt{1 + \sigma_t^2}}, \quad c_{\text{out}}(\sigma_t) = -\frac{\sigma_t}{\sqrt{1 + \sigma_t^2}}$$

$$c_{\text{skip}}(\sigma_t) = \frac{1}{1 + \sigma_t^2}, \quad c_{\text{noise}}(\sigma_t) = 0.25 \cdot \log(\sigma_t)$$

손실 가중치 및 학습 시 노이즈 분포:

$$\lambda_{\sigma_t} = \frac{1}{c_{\text{out}}(\sigma_t)^2}, \quad \ln(\sigma_t) \sim \mathcal{N}(0.7,\, 1.6^2)$$

**Latent space 변환 (LDM 구조)**:

$$\mathbf{z}^{(\mathbf{x})} = \mathcal{E}(\mathbf{x}), \quad \hat{\mathbf{x}} = \mathcal{D}(\mathbf{z}^{(\mathbf{x})})$$

여기서 depth 시퀀스의 경우 3채널 입력 형식을 맞추기 위해 3번 복제 후 디코더 출력 평균을 취합니다. 중요한 점은 프레임별 정규화가 아닌 **전체 비디오에 걸쳐 동일한 scale과 shift**를 사용한다는 것입니다(시간적 일관성 유지의 핵심).

### 2.3 모델 구조

기본적으로 **Stable Video Diffusion (SVD)** 기반의 U-Net을 사용하며, 다음 두 가지로 비디오 조건부를 구성합니다:

1. **Latent concatenation**: 비디오 프레임 latent $\mathbf{z}^{(\mathbf{v})}$를 노이즈가 포함된 depth latent $\mathbf{z}_t^{(\mathbf{d})}$에 **프레임별로** concatenate (원래 SVD는 첫 프레임만 사용)
2. **CLIP cross-attention**: 비디오 프레임의 CLIP 임베딩을 frame-to-frame 방식으로 cross-attention을 통해 주입

### 2.4 3단계 학습 전략 (Three-stage Training)

| 단계 | 학습 대상 | 데이터셋 | 시퀀스 길이 |
|------|-----------|----------|-------------|
| Stage 1 | 전체 U-Net (spatial + temporal) | Realistic (~200K) | $T \in [1, 25]$ |
| Stage 2 | Temporal layers만 | Realistic | $T \in [1, 110]$ |
| Stage 3 | Spatial layers만 | Synthetic (~3K) | $T = 45$ |

- **Stage 1**: video-to-depth task 적응
- **Stage 2**: temporal layer만 fine-tune → 메모리 절감과 동시에 긴 시퀀스 학습
- **Stage 3**: 합성 데이터의 정밀한 depth로 spatial detail 강화

### 2.5 매우 긴 비디오를 위한 추론 전략

- 비디오를 **overlapped segment**들로 분할(최대 110프레임씩)
- 이전 segment의 denoised latent에 노이즈를 추가하여 다음 segment의 overlap 영역을 초기화 → scale/shift 정렬
- Mortise-and-tenon 스타일의 **선형 보간 stitching**: $w_i$가 1에서 0으로 선형 감소하면서 두 segment의 latent를 보간

### 2.6 성능 (Zero-shot, 4개 비디오 데이터셋)

| 데이터셋 | 메트릭 | DepthCrafter | Depth-Anything-V2 | 향상 |
|----------|--------|-----------------|-------------------|------|
| Sintel | AbsRel↓ | **0.270** | 0.367 | 26.4% |
| Sintel | δ₁↑ | **0.697** | 0.554 | – |
| KITTI | AbsRel↓ | **0.104** | 0.140 | 25.7% |
| ScanNet | AbsRel↓ | **0.123** | 0.135 | 5.4% |
| Bonn | AbsRel↓ | **0.071** | 0.106 | 33.0% |

추론 속도: 1024×576 해상도, A100 GPU 기준 프레임당 465.84 ms (Marigold 1070.29 ms보다 빠르지만 Depth-Anything-V2의 180.46 ms보다는 느림).

### 2.7 한계

1. **계산/메모리 비용**: 110프레임 segment는 약 24GB GPU 메모리 필요(40프레임 segment로는 12GB로 감소 가능)
2. **추론 속도**: Diffusion의 반복적 denoising 때문에 discriminative 모델보다 느림 (다만 5 step만 필요)
3. **NYU-v2(단일 이미지)**에서는 Depth-Anything 시리즈보다 약간 낮은 성능 (0.072 vs 0.042 AbsRel) — 비디오 시퀀스에 최적화된 결과

---

## 3. 일반화 성능 향상 가능성 (중점 분석)

DepthCrafter가 open-world 일반화를 달성하는 메커니즘은 **세 가지 시너지**에 있습니다:

### 3.1 비디오 Diffusion Prior의 활용

SVD는 대규모로 큐레이션된 비디오 데이터셋에서 학습되어 다양한 콘텐츠, 모션, 카메라 움직임에 대한 풍부한 prior를 보유합니다. 이를 video-to-depth task로 **재용도화(repurpose)**함으로써, 작은 video depth 데이터셋만으로도 강한 일반화가 가능해집니다. 이는 Marigold가 image diffusion prior로 단일 이미지 depth에서 강한 일반화를 달성한 것과 같은 철학입니다.

### 3.2 Realistic + Synthetic 듀얼 데이터셋 전략

- **Realistic dataset (~200K)**: BiDAStereo로 생성한 stereo 기반 depth → **콘텐츠 다양성** 확보
- **Synthetic dataset (~3K, DynamicReplica + MatrixCity)**: 픽셀 단위 정확한 GT → **세밀한 디테일** 확보

이 두 데이터셋을 단순히 섞지 않고 **3단계로 분리 학습**한 것이 핵심입니다. Synthetic 데이터에 과적합되어 photorealism 일반화가 깨지는 것을 방지하면서도, 마지막 단계에서 spatial layer만 fine-tune하여 디테일을 추가합니다.

### 3.3 가변 길이 학습 ($T \in [1, 110]$)

학습 시 길이를 무작위 샘플링함으로써, 단일 이미지(T=1)부터 긴 비디오까지 동일 모델이 처리 가능합니다. 표 1에서 NYU-v2(1프레임)에서도 경쟁력 있는 성능(0.072 AbsRel)을 보이는 것이 이 전략의 효과를 입증합니다.

### 3.4 일반화 측면의 제약

- Synthetic 데이터가 매우 작아(~3K) 합성 도메인 편향 가능성 존재
- Stereo 기반 의사 GT(BiDAStereo)의 오류가 학습에 누적될 위험
- Diffusion의 stochastic sampling으로 인해 **결정론적 일관성(deterministic consistency)이 보장되지 않음** — 후속 연구인 DVD가 지적한 한계

---

## 4. 향후 연구 영향 및 고려사항

### 4.1 후속 연구에 미친 영향

DepthCrafter는 발표 직후 video depth estimation 분야의 강력한 베이스라인으로 자리잡았으며, 다양한 후속 연구를 촉발시켰습니다:

**(1) Video Depth Anything (CVPR 2025 Highlight)**: Depth Anything V2에 효율적인 spatial-temporal head를 통합하여 임의 길이의 비디오에 대한 일관된 depth estimation을 달성. KITTI에서 δ₁ 0.944로 DepthCrafter(0.753)와 Depth-Anything-V2-L(0.815)을 모두 능가했습니다. Diffusion 기반 모델 대비 더 빠른 추론 속도, 더 적은 파라미터, 더 일관된 depth 정확도를 제공합니다.

**(2) Depth Any Video (ICLR 2025)**: SVD에서 파생되어 다양하고 고품질의 합성 데이터로 fine-tuning. 5초 길이의 40,000개 비디오 클립을 갖는 확장 가능한 합성 데이터 파이프라인 개발. Rotary position encoding과 flow matching 같은 고급 기법 통합. 가변 길이 비디오와 다른 frame rate를 다루는 mixed-duration training 전략 도입.

**(3) ChronoDepth (concurrent work)**: Replacement trick(이전 depth 프레임에 노이즈를 추가하여 overlap 초기화)이 노이즈 변동으로 인해 일관성 문제를 야기한다고 지적하고, 학습 시 각 프레임에 다른 노이즈 레벨을 적용해 노이즈 추가 없이 이전 depth 프레임을 컨텍스트로 사용하는 전략을 제안.

**(4) DepthSync (2025)**: DepthCrafter의 diffusion inversion 기반 overlap 초기화 방식을 분석하고, scale 및 geometry consistency를 더욱 향상시키기 위한 diffusion guidance 기반 동기화 기법을 제안.

**(5) DVD (Deterministic Video Depth)**: DepthCrafter류의 generative 모델은 stochastic sampling으로 인한 시간적 불확실성과 geometric hallucination 문제를 안고 있고, VDA류의 discriminative ViT 모델은 효율성과 결정론적 출력을 제공하나 일반화에서 한계가 있다고 지적하며 이 둘의 장점을 결합하려는 흐름이 형성되었습니다.

**(6) GeometryCrafter, Online VDA (oVDA, 2025)**: Video Depth Anything의 batch-processing 한계를 극복하기 위해 LLM 기법(latent feature caching, frame masking at training)을 적용하여 온라인/저메모리 환경에서 실시간 추론을 가능하게 함.

### 4.2 향후 연구 시 고려할 점

1. **결정론성 vs 일반화의 trade-off**: Generative 접근(DepthCrafter)은 강한 prior를, Discriminative 접근(VDA)은 빠른 결정론적 추론을 제공. 하이브리드 구조 연구 가치가 큼

2. **Metric depth로의 확장**: DepthCrafter는 affine-invariant relative depth만 제공. 3D 재구성, 로봇, AR/VR 응용에는 metric scale이 필요 → GeometryCrafter, Depth Pro, Metric3D V2 같은 metric depth 방향과의 결합 필요

3. **추론 효율성**: 5-step denoising도 ViT 기반 feedforward(Depth-Anything-V2 ~180ms vs DepthCrafter ~466ms)보다 느림. Distillation, quantization, consistency model 적용 여지

4. **온라인/스트리밍 처리**: 현재는 사전에 전체 비디오 segment 분할이 필요한 batch 모드 → 자율주행, 로봇 등 실시간 응용에서는 oVDA류의 streaming 구조가 필수

5. **합성 데이터 품질과 규모**: Stage 3의 합성 데이터셋(~3K)이 너무 작음. Depth Any Video의 40K 합성 클립 같은 대규모 데이터 파이프라인이 향후 표준이 될 가능성

6. **장기간 일관성 평가 메트릭**: AbsRel/δ₁만으로는 시간적 일관성을 충분히 평가하기 어려움. Video Depth Anything이 제안한 TAE(Temporal Alignment Error) 같은 메트릭을 활용해 평가를 보강할 필요

7. **카메라 모션과 동적 객체의 명시적 모델링**: 현재는 implicit하게 처리됨. SLAM/SfM과 결합한 카메라 포즈 인지형(camera-aware) diffusion 방향이 유망

---

## 5. 2020년 이후 관련 최신 연구 비교

| 모델 | 연도 | 백본/접근 | 최대 길이 | 카메라 포즈 | 주요 특징 |
|------|------|-----------|-----------|-------------|-----------|
| Consistent Video Depth (Luo et al.) | 2020 | Test-time opt. | 가변 | 필요 | 비디오별 최적화, 느림 |
| MiDaS | 2020 | ViT, 단일 이미지 | 1 | 불필요 | Affine-invariant, 시간적 비일관 |
| NVDS | ICCV 2023 | Plug-and-play stabilizer | 가변 | 불필요 | Flickering 지속 |
| Marigold | CVPR 2024 | Stable Diffusion (단일 이미지) | 1 | 불필요 | Image diffusion prior 활용 |
| Depth-Anything V1/V2 | CVPR/NeurIPS 2024 | DINOv2 + 대규모 라벨 | 1 | 불필요 | Image SOTA, 시간 비일관 |
| ChronoDepth | 2024 (arXiv 2406.01493) | SVD prior | 10 frames | 불필요 | 가변 노이즈 레벨 학습 |
| **DepthCrafter** | **CVPR 2024** | **SVD + 3-stage** | **110 frames** | **불필요** | **본 논문** |
| Depth Any Video | ICLR 2025 | SVD + flow matching + RoPE | 150 frames | 불필요 | 40K 합성 데이터, mixed-duration |
| Video Depth Anything (VDA) | CVPR 2025 Highlight | DAv2 + ST-head | "임의 길이" | 불필요 | Diffusion 없이 super-long video, 30 FPS |
| GeometryCrafter | 2025 | Diffusion prior | 장기 | 불필요 | Point map (geometry) 직접 추정 |
| DepthSync | 2025 | Diffusion guidance | 장기 | 불필요 | Scale/geometry 일관성 강화 |
| oVDA | 2025 (arXiv 2510.09182) | LLM 기법 + VDA | 무제한(스트리밍) | 불필요 | 온라인, 저메모리 |
| DVD | 2025 | Generative + 결정론적 | 장기 | 불필요 | Generative prior + deterministic 출력 |

비교 핵심 관찰:
- **Diffusion 계열(DepthCrafter, ChronoDepth, Depth Any Video)**: 풍부한 prior로 강한 일반화, 그러나 추론 속도와 결정론성 면에서 약점
- **Discriminative 계열(VDA)**: 빠른 추론과 임의 길이 처리, 그러나 합성/현실 광역 일반화에서는 diffusion 계열이 여전히 우위
- 2025년 이후 트렌드는 **두 패러다임의 융합**과 **온라인/실시간 처리**, 그리고 **affine-invariant에서 metric/geometry로 확장**

---

## 참고자료 출처

논문 본문 및 보충자료는 사용자가 업로드한 PDF (`2409_02095v2.pdf`, Hu et al., DepthCrafter, arXiv:2409.02095v2, CVPR 2024)에서 직접 인용하였습니다.

후속 및 관련 연구 비교를 위해 참고한 외부 자료:

1. Chen, S. et al., **"Video Depth Anything: Consistent Depth Estimation for Super-Long Videos"**, CVPR 2025 (arXiv:2501.12375) — https://arxiv.org/abs/2501.12375, https://videodepthanything.github.io/
2. Yang, H. et al., **"Depth Any Video with Scalable Synthetic Data"**, ICLR 2025 (arXiv:2410.10815) — https://arxiv.org/pdf/2410.10815
3. Shao, J. et al., **"ChronoDepth: Learning Temporally Consistent Video Depth from Video Diffusion Priors"**, 2024 (arXiv:2406.01493) — https://arxiv.org/abs/2406.01493
4. **"DepthSync: Diffusion Guidance-Based Depth Synchronization for Scale- and Geometry-Consistent Video Depth Estimation"**, 2025 (arXiv:2507.01603) — https://arxiv.org/html/2507.01603
5. **"Online Video Depth Anything (oVDA)"**, 2025 (arXiv:2510.09182) — https://arxiv.org/abs/2510.09182
6. **"DVD: Deterministic Video Depth Estimation with Generative Priors"**, 2025 — https://dvd-project.github.io/
7. Roboflow Blog, **"Best Depth Estimation Models: Depth Anything V2 & More"** (2025/2026) — https://blog.roboflow.com/depth-estimation-models/
8. Liner Reviews, **"DepthCrafter Quick Review"** & **"Video Depth Anything Quick Review"** — https://liner.com/review/

> **주의 사항**: 비교표의 일부 수치(예: VDA의 KITTI δ₁ 0.944 등)는 외부 리뷰/논문에서 인용한 것으로 원논문 평가 프로토콜이 DepthCrafter의 것과 완전히 동일하지 않을 수 있으므로, 정량 비교 시 각 논문의 평가 셋팅(시퀀스 길이, alignment 방식, max depth 캡 등)을 직접 확인할 것을 권장합니다. DepthCrafter 자체에 대한 수치는 모두 원논문의 Table 1, S1, S2, S3에서 직접 인용하였습니다.
