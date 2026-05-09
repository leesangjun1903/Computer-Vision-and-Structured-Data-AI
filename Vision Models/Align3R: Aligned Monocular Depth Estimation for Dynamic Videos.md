# Align3R: Aligned Monocular Depth Estimation for Dynamic Videos

## 1. 핵심 주장 및 주요 기여 요약

**Align3R**(Lu et al., 2024, CVPR 2025 Highlight)은 동적(dynamic) 단안 비디오로부터 **시간적으로 일관된 비디오 깊이(video depth)**, **동적 점군(dynamic point clouds)**, **카메라 포즈**를 동시에 추정하는 방법입니다.

핵심 주장은 다음과 같이 정리할 수 있습니다.

- **단안 깊이 추정기(Depth Pro, Depth Anything V2 등)**는 단일 프레임에서는 고품질 결과를 내지만 프레임 간 일관성을 유지하지 못함.
- **DUSt3R**는 두 프레임 간 점지도(point map)를 정합(align)할 수 있지만, 정적(static) 장면 위주로 학습되어 동적 장면에서는 부정확하고 세부 디테일이 부족함.
- 두 모델의 강점을 결합하면 ① 학습이 비디오 디퓨전 모델보다 훨씬 가볍고, ② DUSt3R 단독보다 디테일이 풍부하며, ③ 카메라 포즈까지 동시에 추정 가능.

주요 기여:

1. ControlNet 영감의 **Pre-combination(사전 결합) 전략** — 단안 깊이를 3D 점지도로 unproject한 뒤 별도 ViT로 인코딩하고, **zero-convolution**을 통해 DUSt3R 디코더에 주입.
2. 동적 장면용 5개 데이터셋(SceneFlow, VKITTI, TartanAir, Spring, PointOdyssey)으로 DUSt3R **fine-tuning**.
3. 긴 비디오 처리를 위한 **계층적 최적화(Hierarchical Optimization)** — 메모리 24GB → 5.9GB, 시간 2.9분 → 1.1분으로 단축.
4. 6개 데이터셋(Sintel, Bonn, TUM, PointOdyssey, FlyingThings3D, DAVIS)에서 SOTA 달성.

---

## 2. 문제 정의 · 제안 방법 · 모델 구조 · 성능 및 한계

### 2.1 해결하고자 하는 문제

입력으로 $N$개의 프레임 $\{I_k \in \mathbb{R}^{H \times W \times 3} \mid k = 1, \dots, N\}$이 주어졌을 때, 다음을 동시에 추정:

$$\{D_k \in \mathbb{R}^{H \times W}\}_{k=1}^{N}, \quad \{\pi_k \in SE(3)\}_{k=1}^{N}$$

기존 방법의 세 가지 핵심 난제:

- **단안 깊이 추정기(monocular depth estimator)**: 프레임마다 스케일이 달라 비디오 깊이로 사용 시 깜빡임(flickering) 발생.
- **비디오 디퓨전 기반 방법(DepthCrafter, ChronoDepth)**: 학습 비용 막대, 고정 클립 길이 제약, 카메라 포즈 추정 불가.
- **DUSt3R 원본**: 정적 장면 가정, 디테일 부족.

### 2.2 DUSt3R 복습 (논문 §3.1)

DUSt3R는 두 이미지 $I_n, I_m$에서 점지도 $X_n^e, X_m^e \in \mathbb{R}^{H \times W \times 3}$ ($e = (m,n)$ 좌표계는 $n$ 기준)와 신뢰도 $C_n^e, C_m^e$를 예측합니다. 그리고 **글로벌 정렬(global alignment)** 단계에서 다음을 최적화합니다:

$$\arg\min_{D, \pi, \sigma} \sum_{e \in E} \sum_{v \in e} C_v^e \, \big\| D_v - \sigma_e \, P_e(\pi_v, X_v^e) \big\|_2^2$$

여기서 $\sigma_e$는 엣지별 스케일 인자, $P_e(\pi_v, X_v^e)$는 $X_v^e$를 카메라 $\pi_v$로 투영해 얻는 깊이맵입니다.

### 2.3 제안 방법: 단안 깊이의 사전 결합(Pre-combination)

**Step 1 — Depth → 3D Point map (Unproject)**
Depth Pro 또는 Depth Anything V2가 추정한 깊이 $\hat{D}_n, \hat{D}_m$을 카메라 내부 파라미터(예측된 초점거리 또는 고정값)로 3D 공간에 unproject:

$$\hat{X}_i = \text{Unproject}(\hat{D}_i, K), \quad i \in \{n, m\}$$

수치 안정성을 위해 각 축 $(x, y, z)$를 $[-1, 1]$로 정규화.

**Step 2 — Point Map ViT**
점지도를 patch embedding 후 self-attention으로 다중 레벨 특징 $\hat{F}_i^{(1)}, \dots, \hat{F}_i^{(s)}$ 추출:

$$\hat{X}_i' = \text{PatchEmbed}(\hat{X}_i)$$

**Step 3 — Zero Convolution을 통한 디코더 주입 (ControlNet 방식)**

$$\hat{E}_i^{(l)} = \text{ZeroConv}(\hat{F}_i^{(l)}) + E_i^{(l)}, \quad l = 1, 2, \dots, s$$

여기서 $E_i^{(l)}$는 DUSt3R 디코더의 $l$번째 레이어 출력이고, ZeroConv는 초기에 0으로 초기화되어 **사전학습된 분포를 보존**합니다.

**Step 4 — Fine-tuning Loss (DUSt3R 원본 손실 사용)**

$$\mathcal{L}_{dust3r} = \left\| \frac{1}{z} X_v^e - \frac{1}{\overline{z}} \overline{X}_v^e \right\|_2$$

$z, \overline{z}$는 예측/실측 점지도 정규화 인자, $X, \overline{X}$는 예측/실측 점지도. 인코더는 동결, 디코더 + 새 ViT만 학습.

추론 시에는 MonST3R의 **Flow Loss**(RAFT 광학흐름 기반)를 보조 손실로 추가하여 카메라 포즈 정확도를 개선.

### 2.4 모델 구조 (Figure 2 요약)

학습 시 두 갈래의 ViT 인코더 — (1) RGB 이미지용 (DUSt3R 원본), (2) unprojected point map용 (새로 추가) — 가 병렬로 작동하며, 이들의 특징이 두 개의 Transformer Decoder에서 information interaction을 거쳐 Head를 통해 점지도가 출력됩니다. 추론 시에는 글로벌 정렬을 통해 비디오 전체에 일관된 깊이/포즈 도출.

### 2.5 계층적 최적화 (Hierarchical Optimization)

긴 비디오(30+ 프레임)에서 메모리 폭주 문제를 해결하기 위해:

1. 비디오를 길이 $M = 10$ 또는 $20$의 $K$개 클립으로 분할.
2. 각 클립의 키프레임을 모아 글로벌 정렬 → 키프레임의 깊이/포즈/초점거리 초기화.
3. 각 클립 내부에서 로컬 정렬 수행.

### 2.6 성능 향상 (논문 Table 2, 3)

| 데이터셋 | 지표 | DUSt3R | MonST3R | **Align3R (Depth Pro)** |
|---|---|---|---|---|
| Sintel | Abs Rel ↓ | 0.422 | 0.335 | **0.263** |
| PointOdyssey val | Abs Rel ↓ | 0.184 | 0.089 | **0.077** |
| FlyingThings3D | Abs Rel ↓ | 0.140 | 0.132 | **0.102** |
| Bonn (5 scenes) | Abs Rel ↓ | 0.154 | 0.082 | **0.068** |
| TUM dynamics | ATE ↓ | 0.093 | 0.020 | **0.012** |
| Sintel | RRE ↓ | 11.426 | 0.780 | **0.432** |

대부분의 동적 데이터셋에서 SOTA, 카메라 포즈에서도 RTE/RRE 기준 일관된 우위.

### 2.7 한계점

저자들이 직접 인정한 한계와 추가로 식별 가능한 한계:

- **Bonn, TUM처럼 카메라 움직임이 작고 동적 객체가 적은 단순한 실내 장면**에서는 Depth Pro 단독 결과가 이미 충분히 좋아 글로벌 정렬이 일부 디테일을 손실시킴.
- 동적 데이터셋이 대부분 합성(synthetic) — 실세계 동적 영역에 대한 일반화 검증은 제한적.
- Depth Anything V2처럼 초점거리 예측이 없는 경우 고정값을 써야 해서 unprojection이 부정확.
- 합성곱이 아닌 ViT 기반이라 해상도 변경·도메인 시프트에 여전히 민감.
- Depth filtering(>400m 제거)은 실외 자율주행처럼 원거리가 중요한 시나리오에서 정보 손실 가능.

---

## 3. 모델 일반화 성능 향상 가능성 (집중 분석)

이 부분이 Align3R의 가장 중요한 설계 철학과 맞닿아 있습니다.

### 3.1 일반화에 유리한 설계 결정들

**(1) 인코더 동결 + 디코더만 fine-tuning**
저자들의 ablation(Table 4)에 따르면 전체 모델 fine-tuning은 성능을 오히려 떨어뜨립니다. 이는 DUSt3R 인코더가 다양한 데이터로 사전학습되어 있기에 **그 특징 분포를 보존**하는 것이 일반화에 결정적임을 보여줍니다.

| Setting | Sintel Abs Rel ↓ | TUM ATE ↓ |
|---|---|---|
| F.t. all | 0.310 | 0.025 |
| F.t. last 4 layers | 0.319 | 0.016 |
| F.t. decoder only | 0.306 | 0.017 |
| **ViT encoder + zero-conv (Ours)** | **0.263** | **0.012** |

**(2) Zero-Convolution 주입 (ControlNet 패러다임)**
$\text{ZeroConv}$의 가중치가 0으로 초기화되므로 학습 초기에는 원본 DUSt3R와 동일한 출력을 내고, 점진적으로 깊이 정보를 통합합니다. 이는 **catastrophic forgetting 방지**의 강력한 메커니즘이며, 다른 도메인의 prior(예: 표면 법선, 시맨틱 맵)도 같은 방식으로 결합 가능함을 시사합니다.

**(3) Concat 대신 별도 ViT를 사용한 이유**
RGB와 깊이를 단순히 채널 축으로 concatenate하면 사전학습된 인코더의 입력 분포를 망가뜨려 성능이 급락(Sintel Abs Rel: 0.263 → 0.399). 별도 인코더로 분리한 것은 **모달리티 불일치 시에도 일반화를 보장**하는 일반적 원칙으로 확장 가능합니다.

**(4) 단안 깊이 모델의 Plug-and-Play 호환성**
Depth Pro와 Depth Anything V2 모두에 적용 가능하며, 둘 모두 비슷한 품질의 결과를 냅니다. 이는 **외부 단안 깊이 모델의 발전을 그대로 흡수**할 수 있는 모듈식 구조라는 강한 장점이 있습니다. 향후 더 강력한 단안 깊이 모델이 등장할수록 Align3R의 성능도 함께 향상될 가능성이 큽니다.

### 3.2 일반화 성능에 대한 잠재적 우려

- **합성 데이터 편향**: 학습 데이터 5개 중 4개가 합성(SceneFlow, VKITTI, Spring, PointOdyssey). TartanAir만 정적 실세계 데이터셋. 실세계 동적 데이터(예: Bonn 같은 RGB-D)는 fine-tuning에 사용되지 않음. → **합성-실세계 도메인 갭** 가능성.
- **Depth filtering(>400m)**: 자율주행·드론·우주 영상처럼 매우 먼 거리가 중요한 시나리오에서 일반화 한계.
- **카메라 모션 패턴 편향**: 학습 데이터의 모션 분포가 실제 사용자(휴대폰) 영상과 다를 경우 글로벌 정렬이 발산할 가능성.
- **계층적 최적화의 키프레임 선택**: 클립 분할이 fixed length 기반이라 빠른 모션 영역에서는 키프레임 간 overlap이 부족할 수 있음.

### 3.3 일반화를 더 강화할 향후 방향

1. **PromptDA를 활용한 원해상도 복원** (저자 GitHub 공식 TODO에 언급) — 패치 다운샘플링에서 잃어버린 디테일 복구.
2. **실세계 동적 영상(예: Stereo4D, Waymo Dynamic)으로 fine-tuning 확장**.
3. **단안 깊이 모델의 불확실성(uncertainty) 추정**을 zero-conv에 함께 주입하여 신뢰도 가중 결합.
4. **메트릭(metric) 깊이 추정**까지의 확장 — 현재는 affine-invariant 평가만 수행.

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 영향

**(A) "Foundation Model + Light Adapter" 패러다임의 확립**
Align3R은 DUSt3R 같은 대형 3D 기반 모델에 ControlNet 스타일 어댑터로 새 모달리티를 주입하는 설계가 효과적임을 보였고, 이는 후속 연구에서 광범위하게 채택되었습니다. 예컨대 **G-CUT3R**(2025, arXiv:2508.11379)는 명시적으로 같은 zero-convolution 메커니즘으로 카메라/깊이 prior를 CUT3R에 주입합니다.

**(B) 동적 비디오 3D 비전의 새 벤치마크 정립**
Sintel + Bonn + TUM dynamics + PointOdyssey 조합이 동적 비디오 깊이 평가의 사실상 표준 벤치마크로 자리잡는 데 기여.

**(C) 4D Reconstruction 파이프라인의 입력 모듈로의 활용**
Mosca, Shape-of-Motion 같은 후속 4D Gaussian Splatting 작업들이 Align3R/MonST3R/MegaSaM 출력을 초기화로 활용.

### 4.2 향후 연구 시 고려할 점

1. **Feed-forward vs. Optimization-based의 트레이드오프**
Align3R은 여전히 추론 시 글로벌 정렬 최적화(약 1~2분)에 의존합니다. VGGT(CVPR 2025), CUT3R, π³ 같은 완전 feed-forward 모델은 수 초 내 결과를 제공하지만 일관성이 다소 떨어질 수 있습니다. 두 패러다임의 융합이 중요한 연구 방향.

2. **광학흐름(flow) 의존성 재검토**
저자들도 인정했듯 카메라 포즈 정확도는 RAFT flow loss에 크게 의존합니다. 고속 모션·블러·저조도에서 flow 추정 자체가 무너지면 전체 시스템이 흔들립니다. **End-to-end flow-free 동적 정렬** 연구가 필요.

3. **메트릭 스케일 복원(metric scale recovery)**
Align3R은 affine-invariant 평가만 수행. 로보틱스·AR/VR에서 실용성을 위해서는 메트릭 단위가 필요하며, MegaSaM처럼 메트릭 prior를 함께 학습하는 방법이 유망.

4. **장기 일관성(long-term consistency)**
계층적 최적화로 메모리는 해결했지만, 수천 프레임의 loop-closure는 미해결. VGGT-Long, MASt3R-SLAM이 이 방향에서 발전 중.

5. **동적 객체 분할(dynamic mask)과의 통합**
DAS3R, VGGT4D는 명시적 동적 마스크를 학습합니다. Align3R은 이를 분리하지 않아 매우 큰 동적 영역에서 글로벌 정렬이 편향될 위험.

6. **Open-world 일반화 평가**
대부분 평가가 학습 분포와 유사한 도메인(synthetic indoor/outdoor). 진짜 in-the-wild 영상(예: 핸드헬드 휴대폰, 동물 다큐)에 대한 정량적 벤치마크 부족.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 연도 | 카테고리 | 핵심 아이디어 | 카메라 포즈 | 일관성 | 추론 속도 |
|---|---|---|---|---|---|---|
| Robust-CVD | 2021 | Optimization-based | flow + pose 기반 단안 깊이 정렬, deformation spline | △ (입력 필요 무) | 중 | 매우 느림(수 시간) |
| CasualSAM | 2022 | Optimization-based | NeRF + mono-depth 공동 최적화 | ○ | 중 | 매우 느림 |
| ChronoDepth | 2024 | Video Diffusion | Latent Video Diffusion 기반 깊이 생성 | ✕ | 중상 | 중 |
| DepthCrafter | 2024 | Video Diffusion | SVD 기반, 긴 시퀀스 깊이 생성 | ✕ | 상 | 중 |
| Depth Pro | 2024 | Single-frame | 0.3초 메트릭 단안 깊이, 초점거리 예측 | ✕ | 하 | 매우 빠름 |
| DUSt3R | CVPR 2024 | Pointmap-based | 두 이미지 → 점지도 직접 회귀 + 글로벌 정렬 | ○ (정적) | 정적 SOTA | 보통 |
| MonST3R | 2024 (concurrent) | Pointmap-based | DUSt3R를 동적 데이터로 단순 fine-tuning + flow loss | ○ | 상 | 보통 |
| **Align3R** | CVPR 2025 Highlight | Pointmap + Mono-depth fusion | 단안 깊이 unproject → ViT + zero-conv 주입 | ○ | **최상** | 1.1~1.8분 |
| MegaSaM | CVPR 2025 | Deep Visual SLAM | DROID-SLAM에 mono-depth/focal 사전정보 통합 + uncertainty BA | ○ (메트릭) | 상 | 빠름 |
| CUT3R | CVPR 2025 | Streaming pointmap | persistent state로 온라인 4D 인지 | ○ | 상 | 빠름 |
| VGGT | CVPR 2025 | Foundation feed-forward | 단일 transformer가 깊이/포즈/포인트 한 번에 회귀 | ○ | 상 | 매우 빠름(수 초) |
| StreamVGGT | 2025 | Streaming feed-forward | VGGT의 스트리밍 버전 | ○ | 상 | 매우 빠름 |
| ViPE | 2025 (NVIDIA) | Hybrid (BA + 학습) | 고전 BA + 학습 컴포넌트 | ○ (메트릭) | 상 | 빠름 |
| G-CUT3R | 2025 | Pointmap + Prior | CUT3R에 카메라/깊이 prior를 zero-conv로 주입 (Align3R 패러다임 계승) | ○ | 상 | 보통 |

핵심 흐름:
- 2020~2022: **테스트 시 최적화** 중심 (Robust-CVD, CasualSAM) — 느리고 brittle.
- 2023~2024 상반기: **비디오 디퓨전**으로 일관성 확보 시도 (DepthCrafter, ChronoDepth) — 학습 비용 막대, 포즈 없음.
- 2024 하반기: **DUSt3R 계열**(MonST3R, Align3R)이 동시 등장 — 점지도 회귀로 가벼운 학습, 포즈+깊이 통합.
- 2025: **Foundation feed-forward**(VGGT, CUT3R, π³)와 **Hybrid BA**(MegaSaM, ViPE)가 양대 축으로 확산.

Align3R의 위치는 "단안 깊이 prior를 사전학습 모델에 비파괴적으로 주입한다"는 설계 철학으로, **G-CUT3R 등 후속 연구의 직접적 영감**이 되었습니다.

---

## 참고자료 출처

본 분석에서 인용/참조한 자료들의 제목과 출처:

1. **Align3R 원논문 PDF** (사용자 업로드, arXiv:2412.03079v2, Dec 2024): Lu et al., "Align3R: Aligned Monocular Depth Estimation for Dynamic Videos"
2. **Align3R arXiv 페이지** — https://arxiv.org/abs/2412.03079
3. **Align3R 공식 GitHub** (CVPR 2025 Highlight 표기) — https://github.com/jiah-cloud/Align3R
4. **Emergent Mind 요약** — https://www.emergentmind.com/papers/2412.03079
5. **MonST3R 원논문**: Zhang et al., "MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion", arXiv:2410.03825
6. **DUSt3R 원논문**: Wang et al., "DUSt3R: Geometric 3D Vision Made Easy", CVPR 2024
7. **Depth Pro**: Bochkovskii et al., arXiv:2410.02073, 2024
8. **Depth Anything V2**: Yang et al., arXiv:2406.09414, 2024
9. **DepthCrafter**: Hu et al., arXiv:2409.02095, 2024
10. **ChronoDepth**: Shao et al., arXiv:2406.01493, 2024
11. **MegaSaM** (CVPR 2025) — https://mega-sam.github.io/ , https://arxiv.org/abs/2412.04463
12. **VGGT** (CVPR 2025): "Meta's VGGT reconstructs 3D scenes in seconds" — https://www.mlwires.com/metas-vggt-reconstructs-3d-scenes-in-seconds-cvpr-2025/
13. **StreamVGGT** (arXiv:2507.11539, 2025) — https://wzzheng.net/StreamVGGT/
14. **G-CUT3R** (arXiv:2508.11379, 2025) — https://huggingface.co/papers/2508.11379
15. **ViPE: Video Pose Engine** (NVIDIA, 2025) — https://research.nvidia.com/labs/toronto-ai/vipe/
16. **VGGT4D** (arXiv:2511.19971, 2025)
17. **All-3R-SLAM-in-this-Repo** (커뮤니티 정리) — https://github.com/3D-Vision-World/All-3R-SLAM-in-this-Repo
18. **ControlNet**: Zhang et al., "Adding Conditional Control to Text-to-Image Diffusion Models", ICCV 2023 (Align3R의 zero-convolution 영감원)
19. **Robust-CVD**: Kopf et al., CVPR 2021
20. **CasualSAM**: Zhang et al., ECCV 2022

> **정확도 관련 주의**: Align3R 본 논문의 수치, 손실식, 구조 설명은 업로드된 PDF에 직접 근거합니다. 후속 연구들의 비교 표 항목 중 일부 정성적 기술(예: "느림/빠름")은 각 논문의 자체 보고에 기반한 상대적 표현이며, 동일 하드웨어·동일 영상에서의 직접 비교 결과는 아닐 수 있습니다. 각 연도와 출판 venue는 인용한 공식 페이지 기준입니다.
