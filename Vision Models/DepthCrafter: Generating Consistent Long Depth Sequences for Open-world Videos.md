# DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos

## 1. 핵심 주장과 주요 기여 (간결한 요약)

**DepthCrafter**는 카메라 포즈나 옵티컬 플로우 같은 부가 정보 없이도, **오픈월드(open-world) 비디오**에 대해 **시간적으로 일관된(temporally consistent)** **장기 깊이 시퀀스(long depth sequence)**를 생성하는 새로운 비디오 깊이 추정(video depth estimation) 방법입니다.

**핵심 기여 3가지**

1. **사전학습된 이미지-투-비디오(image-to-video) 디퓨전 모델 (SVD)** 을 비디오-투-깊이(video-to-depth) 모델로 전환하여, 오픈월드 비디오에 대한 강한 일반화 성능 확보.
2. **3단계 학습 전략(three-stage training strategy)** 을 통해 한 번에 최대 **110프레임**의 가변 길이 깊이 시퀀스를 생성 가능하며, 합성·실사 데이터셋의 장점을 모두 흡수.
3. **세그먼트 단위 추론(segment-wise inference) + 잠재 보간 스티칭(latent interpolation stitching)** 으로 수백~수천 프레임의 매우 긴 비디오 처리 가능.

---

## 2. 해결하려는 문제, 방법, 모델 구조, 성능 및 한계

### 2.1 해결하려는 문제

오픈월드 비디오 깊이 추정의 3가지 핵심 난점:

- **콘텐츠 다양성**: 외관, 객체 움직임, 카메라 움직임, 길이가 매우 다양해 일반화가 어려움.
- **장기 시간적 일관성(temporal consistency)**: 단일 이미지 모델을 프레임별로 적용하면 깜빡임(flickering)이 발생.
- **카메라 포즈/옵티컬 플로우 의존**: 기존 비디오 깊이 추정법(테스트 타임 최적화 계열)은 카메라 포즈나 플로우가 필요하며, 동적 콘텐츠가 많거나 매우 긴 비디오에서는 이를 얻기가 비현실적.

### 2.2 디퓨전 모델 기반 정식화

논문은 비디오 깊이 추정을 **조건부 디퓨전 생성 문제** $p(\mathbf{d} \mid \mathbf{v})$ 로 정식화합니다. 여기서 $\mathbf{v} \in \mathbb{R}^{T \times H \times W \times 3}$ 는 입력 비디오, $\mathbf{d} \in \mathbb{R}^{T \times H \times W}$ 는 깊이 시퀀스입니다.

**전방(forward) 디퓨전 과정** — 데이터 $\mathbf{x}_0 \sim p(\mathbf{x})$ 에 i.i.d. $\sigma_t^2$-분산 가우시안 노이즈를 추가:

$$\mathbf{x}_t = \mathbf{x}_0 + \sigma_t^2 \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

**Denoiser $D_\theta$ 학습 목적함수 (denoising score matching)**:

$$\mathbb{E}_{\mathbf{x}_t \sim p(\mathbf{x};\sigma_t),\, \sigma_t \sim p(\sigma)} \left[ \lambda_{\sigma_t} \left\| D_\theta(\mathbf{x}_t; \sigma_t; \mathbf{c}) - \mathbf{x}_0 \right\|_2^2 \right]$$

**EDM 프레임워크의 전제조건화(preconditioning)** [Karras et al., 2022]:

$$D_\theta(\mathbf{x}_t; \sigma_t; \mathbf{c}) = c_{\text{skip}}(\sigma_t)\mathbf{x}_t + c_{\text{out}}(\sigma_t)F_\theta\bigl(c_{\text{in}}(\sigma_t)\mathbf{x}_t;\, c_{\text{noise}}(\sigma_t);\, \mathbf{c}\bigr)$$

**구체적 전제조건 함수** (보충자료 Eq. 5):

$$c_{\text{in}}(\sigma_t) = \frac{1}{\sqrt{1+\sigma_t^2}},\quad c_{\text{out}}(\sigma_t) = \frac{-\sigma_t}{\sqrt{1+\sigma_t^2}}$$

$$c_{\text{skip}}(\sigma_t) = \frac{1}{1+\sigma_t^2},\quad c_{\text{noise}}(\sigma_t) = 0.25 \cdot \log(\sigma_t)$$

**손실 가중치**: $\lambda_{\sigma_t} = 1/c_{\text{out}}(\sigma_t)^2$

**노이즈 레벨 분포** (학습 시 샘플링): $\ln(\sigma_t) \sim \mathcal{N}(0.7, 1.6^2)$

### 2.3 모델 구조

**잠재 공간 변환 (LDM 기반)**: VAE 인코더 $\mathcal{E}$와 디코더 $\mathcal{D}$를 사용해 비디오·깊이를 모두 잠재공간으로 변환:

$$\mathbf{z}^{(\mathbf{x})} = \mathcal{E}(\mathbf{x}),\quad \hat{\mathbf{x}} = \mathcal{D}(\mathbf{z}^{(\mathbf{x})})$$

깊이 시퀀스는 3채널로 복제하여 SVD VAE에 입력하고, 출력은 평균하여 단일 채널 깊이로 복원합니다. 깊이는 affine-invariant 표현으로 $[0,1]$ 범위로 정규화하되, **per-frame이 아닌 시퀀스 전체에 동일한 scale/shift**를 사용 — 시간적 일관성 유지의 핵심 포인트입니다.

**비디오 조건화 메커니즘 (SVD 변형)**:
- 원래 SVD는 첫 프레임만 조건으로 받지만, DepthCrafter는 **모든 프레임의 비디오 잠재 $\mathbf{z}^{(\mathbf{v})}$ 를 잡음 깊이 잠재 $\mathbf{z}_t^{(\mathbf{d})}$ 와 프레임별 채널 결합(concatenation)**.
- **CLIP 임베딩**도 프레임별로 cross-attention에 주입.

### 2.4 3단계 학습 전략

| 단계 | 학습 대상 | 데이터셋 | 시퀀스 길이 $T$ |
|---|---|---|---|
| Stage 1 | U-Net 전체 | 실사(realistic, ~200K 쌍) | $T \in [1, 25]$ |
| Stage 2 | 시간(temporal) 레이어만 | 실사 | $T \in [1, 110]$ |
| Stage 3 | 공간(spatial) 레이어만 | 합성(synthetic, ~3K, DynamicReplica + MatrixCity) | $T = 45$ |

**핵심 아이디어**: Stage 1에서 영상-깊이 변환을 학습하고, Stage 2에서 시간 레이어만 학습해 메모리를 절약하며 장기 컨텍스트를 확장하고, Stage 3에서 공간 레이어를 합성 데이터의 정밀한 깊이로 미세조정해 시간 일관성을 해치지 않으면서 디테일을 향상.

### 2.5 매우 긴 비디오를 위한 추론 전략

- 비디오를 최대 110프레임의 **겹치는 세그먼트(overlapped segments)** 로 분할.
- 다음 세그먼트의 겹치는 프레임 잠재는 **이전 세그먼트의 디노이즈된 잠재에 노이즈 추가**로 초기화 → scale/shift 일관성(anchor) 확보.
- **mortise-and-tenon (장부맞춤) 스타일 잠재 보간**: 겹치는 프레임 $o_i$ 에 대해 가중치 $w_i$ 와 $1-w_i$ ($w_i$ 는 1에서 0으로 선형 감소) 으로 보간.

### 2.6 성능 향상 (Tab. 1, zero-shot)

| 데이터셋 | 지표 | Depth-Anything V2 | DepthCrafter | 개선율 |
|---|---|---|---|---|
| Sintel (~50f) | AbsRel ↓ | 0.367 | **0.270** | **26.4%** |
| Sintel | $\delta_1$ ↑ | 0.554 | **0.697** | +14.3%p |
| KITTI (110f) | AbsRel ↓ | 0.140 | **0.104** | **25.7%** |
| Scannet (90f) | AbsRel ↓ | 0.135 | **0.123** | 8.9% |
| Bonn (110f) | AbsRel ↓ | 0.106 | **0.071** | **33.0%** |

특히 **카메라/객체 운동이 큰 Sintel·KITTI**에서 큰 폭의 향상이 두드러집니다.

### 2.7 한계 (논문 §4.5)

- **연산·메모리 비용**: 1024×576 해상도에서 프레임당 약 465.84ms (Depth-Anything V2 의 ~2.6배), 110프레임 세그먼트 기준 약 24GB GPU 메모리 필요. 40프레임으로 줄이면 12GB로 완화 가능.
- **반복적 디노이징** 으로 인한 비결정론적(non-deterministic) 출력.
- **Sora급 매우 긴 비디오/극단적 조명/투명 객체** 등에서는 오류 가능성 (후속 연구에서 지적됨).

---

## 3. 일반화 성능(Open-world Generalization) 향상 가능성 중점 분석

DepthCrafter의 일반화 능력은 다음 4가지 설계가 결합한 결과입니다.

### (1) 사전학습 비디오 디퓨전의 강력한 사전(prior) 활용
SVD는 대규모 잘 정제된 비디오로 학습되어 **자연스러운 시공간 패턴**을 이미 알고 있습니다. DepthCrafter는 이를 깊이 회귀에 그대로 전이하여, 라벨이 부족한 비디오 깊이 도메인의 한계를 우회합니다.

### (2) 실사 + 합성 데이터의 상보적 결합
- **실사 데이터(~200K, BiDAStereo로 의사 라벨)**: 콘텐츠 다양성 확보 → 도메인 일반화.
- **합성 데이터(~3K, DynamicReplica + MatrixCity)**: 정밀한 GT 깊이 → 디테일 학습.
이 둘을 단순 혼합하지 않고, **3단계 학습으로 단계별로 분리 학습** 하여 각각의 장점이 충돌하지 않게 한 점이 핵심입니다.

### (3) 가변 길이 + 장기 컨텍스트 학습
$T \in [1, 110]$ 범위에서 학습된 모델은 **단일 이미지(T=1)에서 장기 비디오까지 동일 모델로 처리** 가능. 실제로 NYU-v2 단일 이미지 평가에서도 경쟁력 있는 성능($\delta_1 = 0.948$)을 보여, 이미지·비디오 도메인 통합 일반화 능력을 입증합니다.

### (4) 공유 scale/shift 정규화
프레임별 정규화 대신 **시퀀스 전체에 단일 scale/shift** 를 적용해, 모델이 "전역 깊이 분포"를 학습하도록 강제 → 새로운 도메인에서도 시간 일관성을 유지합니다.

### 향후 일반화 향상의 잠재 방향
- **모델 증류(distillation) / 양자화** 로 추론 비용을 줄이면서 일반화 유지.
- **메트릭 깊이(metric depth)** 추정으로 확장 시, 카메라 내부 파라미터를 추가 조건으로 주는 설계 가능.
- **합성 데이터 스케일 업** (Depth Any Video[ICLR 2025]는 6M, Video Depth Anything[CVPR 2025]은 10M+ 프레임 사용) 이 일반화에 결정적임이 후속 연구에서 입증됨.

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 연구 영향

DepthCrafter는 **"비디오 디퓨전 사전학습 모델을 비디오 dense prediction 작업으로 재사용한다"** 는 패러다임을 명확히 제시했습니다. 발표 이후 다음과 같은 연구들이 영향을 받았거나 직접 비교군이 되었습니다.

- **Video Depth Anything (CVPR 2025 Highlight)**: DepthCrafter의 인용 및 직접 비교 대상이 되었으며, 디퓨전 기반의 무거운 추론을 피하기 위해 Depth Anything V2 위에 시공간 헤드를 추가한 비-디퓨전 접근.
- **Depth Any Video (ICLR 2025)**: 디퓨전 + 혼합 길이 학습 + 회전 위치 인코딩으로 DepthCrafter를 개선.
- **StereoDiff (2025)**: 정적 영역은 스테레오 매칭, 동적 영역만 비디오 디퓨전을 쓰는 2단계 하이브리드 — DepthCrafter의 "전 영역 디퓨전" 접근의 한계(연산량, 정적 영역의 글로벌 일관성 약화)를 지적.
- **DKT (2025)**: DepthCrafter의 발상을 **투명/반사 객체** 라는 어려운 도메인에 확장.
- **Online Video Depth Anything (2025)**: DepthCrafter의 오프라인·세그먼트 스티칭 방식의 한계(메모리, 실시간성)를 보완하는 온라인 스트리밍 모델.

### 4.2 향후 연구 시 고려할 점

1. **추론 효율성**
   디퓨전 기반은 본질적으로 무겁습니다. **단일 스텝 디퓨전(consistency model, distillation, flow matching)** 이나 **회귀 헤드 변환**을 통한 가속이 필수 연구 주제. Video Depth Anything은 디퓨전을 아예 제거하는 방향을 택했고, Depth Any Video는 flow matching을 사용했습니다.

2. **메트릭 깊이 추정 확장**
   DepthCrafter는 affine-invariant 상대 깊이만 다룹니다. 자율주행·로보틱스 응용을 위해서는 **메트릭 깊이** 가 필요하며, 카메라 내부 파라미터를 조건화하는 후속 연구(MoGe, Metric3D v2 계열)와의 결합이 유망합니다.

3. **온라인/스트리밍 추론**
   DepthCrafter는 오프라인 세그먼트 스티칭에 의존하여 **저지연 실시간 처리에는 부적합**. 인과적(causal) 시간 어텐션, KV 캐싱, Mamba류 상태공간 모델로 대체하는 연구가 진행 중.

4. **동적/정적 영역의 분리 처리**
   StereoDiff가 지적한 대로, 정적 배경은 다중 시점 기하학으로 더 강한 일관성을 얻을 수 있는데 DepthCrafter는 이를 활용하지 않습니다. **기하학(geometry) + 디퓨전 prior 의 결합**이 명확한 다음 단계입니다.

5. **장기 비디오에서의 drift**
   Video Depth Anything 논문에서는 DepthCrafter가 긴 비디오에서 깊이 드리프트(depth drift) 를 보인다고 지적합니다. 세그먼트 간 전역 정합을 위한 더 강한 제약(글로벌 정렬 모듈, 키프레임 앵커링 등)이 필요합니다.

6. **데이터 스케일과 품질의 균형**
   3K 합성 데이터로 디테일을 학습하는 전략은 우아하지만, **합성 데이터를 더 크고 다양하게**(예: 게임 엔진 기반 수백만 프레임) 만들면 추가 향상 가능성이 큽니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연도 | 방법 | 핵심 아이디어 | DepthCrafter와의 차이 / 위치 |
|---|---|---|---|
| 2020 | **MiDaS** [Ranftl et al., TPAMI] | Affine-invariant 깊이 + 다중 데이터 혼합 | 단일 이미지 전용 |
| 2020 | **Consistent Video Depth** [Luo et al., SIGGRAPH] | 테스트 타임 최적화 + COLMAP 카메라 포즈 | 카메라 포즈 필수, 동적 장면 약함 |
| 2023 | **NVDS** [Wang et al., ICCV] | Plug-and-play 시간 안정화 네트워크 | DepthCrafter 비교군, AbsRel 대폭 열세 |
| 2024 | **Marigold** [Ke et al., CVPR] | Stable Diffusion → 단일 이미지 깊이 | 이미지만, 시간 일관성 없음 |
| 2024 | **Depth Anything V2** [Yang et al., NeurIPS] | 대규모 비라벨 데이터 + 강한 백본 | 단일 이미지 SOTA, 비디오 깜빡임 |
| 2024 | **ChronoDepth** [Shao et al.] | SVD 기반 비디오 깊이 (10프레임) | DepthCrafter와 동시기, 컨텍스트 매우 짧음 |
| **2024.09** | **DepthCrafter (본 논문)** | SVD + 3단계 학습 + 110 프레임 + 스티칭 | — |
| 2025.01 | **Video Depth Anything** (CVPR 2025 Highlight) | Depth Anything V2 기반 비-디퓨전, 임의 길이 비디오 | Sintel에서는 DepthCrafter가 더 정확하나, 장기 비디오·속도에서 우위 |
| 2025 (ICLR) | **Depth Any Video** [Yang et al.] | 6M 합성 프레임 + flow matching + 회전 위치 인코딩 | 더 큰 데이터, 가변 길이 학습 |
| 2025 (CVPR) | **RollingDepth** ("Video depth without video models") | 비디오 모델 없이 이미지 디퓨전을 롤링 윈도우로 활용 | 비디오 디퓨전 사전학습 사용 안 함 |
| 2025 | **StereoDiff** | 정적 영역=스테레오 매칭, 동적 영역=비디오 디퓨전의 2단계 시너지 | DepthCrafter의 "단일 디퓨전" 한계를 명시적으로 비판하며 개선 |
| 2025.10 | **Online Video Depth Anything** | 저메모리·온라인 스트리밍 깊이 추정 | DepthCrafter의 오프라인 세그먼트 한계 보완 |
| 2025.12 | **DKT (Diffusion Knows Transparency)** | 투명/반사 객체에 특화된 비디오 디퓨전 깊이 | DepthCrafter가 약한 도메인을 직접 겨냥 |

**전체 흐름**

오픈월드 비디오 깊이 추정은 (a) **단일 이미지 디퓨전(2024 Marigold)** → (b) **비디오 디퓨전으로 확장(2024 ChronoDepth, DepthCrafter)** → (c) **데이터·길이 스케일업(2025 Depth Any Video, Video Depth Anything)** → (d) **하이브리드·실시간·특수 도메인(2025 StereoDiff, Online VDA, DKT)** 으로 빠르게 분화 중입니다. DepthCrafter는 이 전이의 **결정적 분기점** 역할을 했으며, 후속 연구 대부분이 명시적 비교군 또는 출발점으로 삼고 있습니다.

---

## 참고자료 출처

1. **본 논문**: Hu, Wenbo et al., "DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos" (CVPR 2025), arXiv:2409.02095v2 — 사용자 업로드 PDF.
2. **DepthCrafter 프로젝트 페이지**: https://depthcrafter.github.io/
3. **Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models" (EDM)**, NeurIPS 2022 — 본 논문 [31]번 참조.
4. **Stable Video Diffusion (SVD)**: Blattmann et al., arXiv:2311.15127 — 본 논문 [3]번 참조.
5. **Video Depth Anything (CVPR 2025 Highlight)**: https://videodepthanything.github.io/ , GitHub: https://github.com/DepthAnything/Video-Depth-Anything , CVPR 논문 PDF: https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Video_Depth_Anything_Consistent_Depth_Estimation_for_Super-Long_Videos_CVPR_2025_paper.pdf
6. **Depth Any Video (ICLR 2025)**: OpenReview https://openreview.net/forum?id=gWqFbnKsqR , arXiv:2410.10815
7. **StereoDiff (2025)**: arXiv:2506.20756 — https://arxiv.org/abs/2506.20756
8. **DKT — Diffusion Knows Transparency (2025)**: arXiv:2512.23705 — https://arxiv.org/abs/2512.23705
9. **Online Video Depth Anything (2025)**: arXiv:2510.09182 — https://arxiv.org/html/2510.09182v1
10. **"Towards Depth Foundation Model" 서베이 (2025)**: arXiv:2507.11540 — https://arxiv.org/html/2507.11540v1

> **정확도에 대한 공지**: 본 답변에서 DepthCrafter 자체에 관한 모든 수식·표·수치는 업로드된 논문 PDF에 직접 근거합니다. 후속 연구(2025년 이후) 비교 부분은 위 출처의 공개 정보에 기반했으나, **각 후속 모델의 세부 수치(예: VDA-L의 KITTI $\delta_1$ 등)는 해당 원논문 본문 검증을 권장**합니다. 추가로 매우 새로운 일부 논문(특히 2025년 후반)에 대해서는 분야 전체 동향을 보여주기 위한 비교 위치 표시 정도로만 사용했음을 밝힙니다.
