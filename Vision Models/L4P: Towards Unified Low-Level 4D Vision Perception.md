# L4P: Towards Unified Low-Level 4D Vision Perception

L4P 논문은 대규모 비디오 MAE(VideoMAE v2) 백본 하나와 가벼운 태스크 전용 헤드를 조합해, 깊이·광류·모션 분할 같은 **dense** 태스크와 2D/3D 포인트 트래킹 같은 **sparse** 태스크를 단일 피드포워드 모델로 동시에 풀면서도, 각 태스크 특화 SOTA와 비슷한 수준의 성능과 강한 제로샷 일반화를 보인다는 것을 핵심 주장으로 합니다.[^1][^2]

***

## 1. 핵심 주장과 주요 기여

- 서로 다른 저수준 4D 인지(깊이, 광류, 2D/3D 트래킹, 모션 세그멘테이션, 카메라 포즈)를 **하나의 VideoMAE 기반 백본 + 태스크별 경량 헤드**로 통합하는 일반 목적 아키텍처를 제안합니다.[^2][^1]
- 트래킹을 “쿼리 포인트에 대한 2D 확률 히트맵 + 깊이 + 가시성”으로 정식화하고, 메모리 메커니즘을 추가해 긴 비디오에서의 온라인 포인트 추적을 지원합니다.[^1]
- 4개의 합성 데이터셋만으로 멀티태스크 파인튜닝을 수행하면서도, ScanNet, KITTI, Spring, TAPVid-3D 등 다양한 **실세계 벤치마크에서 강한 제로샷 일반화와 SOTA급 성능**을 달성합니다.[^1]
- 비디오 인코더를 고정한 채 새로운 헤드만 추가해 모션 기반 세그멘테이션, 카메라 포즈 추정 등 태스크를 손쉽게 확장할 수 있음을 보입니다.[^1]

***

## 2. 해결하고자 하는 문제

저자들이 보는 근본 문제는, 비디오 픽셀 간 시공간 관계가 많은 저수준 4D 태스크의 공통 기반인데도, 현재 방법들이 깊이·광류·트래킹 등을 전부 **서로 다른 전용 네트워크**로 풀고 있다는 점입니다.[^2][^1]
특히 (i) 밀집 2D 맵(깊이, 플로우)과 (ii) 희소 2D/3D 트랙(포인트 트래킹)을 동시에 지원할 수 있는 공통 표현과 백본을 설계하는 것이 어렵고, (iii) 장길이 비디오에서 온라인으로 추적을 유지하는 것도 VideoMAE의 고정된 윈도우 길이 때문에 도전적입니다.[^1]

이 논문은 “**강한 비디오 표현(백본)** + **태스크별 가벼운 디코더**”를 통해 이 분절을 해소하고, 다양한 저수준 4D 태스크를 하나의 모델로 처리하면서도, 기존 태스크 특화 모델과 유사한 정확도와 효율성을 달성하는 것을 목표로 합니다.[^2][^1]

***

## 3. 제안된 통합 아키텍처 개요

입력 비디오는 $T \times H \times W$ 크기의 RGB 시퀀스로, VideoMAEv2 인코더에 의해 큐브 패치($t \times h \times w$) 단위로 토큰화됩니다.[^1]
이를 통해 비디오는 $P$개의 비디오 토큰 시퀀스 $S \in \mathbb{R}^{P \times C}$로 임베딩되며, 이 인코더는 40층 ViT로 구성되고, dense 태스크 헤드는 중간 레이어(14, 21, 28, 36층), sparse 헤드는 마지막 레이어(39층)의 토큰만 사용해 상호 간섭을 줄입니다.[^1]

이 공통 표현 위에 두 종류의 헤드가 얹힙니다.[^1]

- **Dense DPT 기반 헤드**: 깊이, 광류, 모션 세그멘테이션, 카메라 rays(→ 포즈)를 출력하도록 Video용 3D DPT로 확장.
- **Sparse SAM‑style 헤드**: 쿼리 포인트 토큰과 출력 토큰(heatmap, depth, visibility)이 Video 토큰과 2‑way attention으로 상호작용하여 3D 트랙을 복원.

***

## 4. 수식 관점에서의 방법 (태스크 정식화 및 손실)

### 4.1 3D 포인트 트래킹 정식화

하나의 쿼리 포인트는 $(t_i, x_i, y_i)$로 주어지며, 모델은 전체 비디오 길이 $S$에 대해 다음 궤적을 예측합니다.[^1]

```math
T_i = \left\{ \hat x_i(t), \hat y_i(t), \hat d_i(t), \hat v_i(t) \right\}_{t=0}^{S-1} \quad [^1]
```

여기서 $(\hat x_i(t), \hat y_i(t))$는 프레임 $t$에서의 2D 위치, $\hat d_i(t)$는 깊이, $\hat v_i(t)$는 가시성(0–1)을 의미합니다.[^1]
트래킹은 곧 $T \times H \times W$ 크기의 2D 확률 히트맵을 예측한 뒤, 각 프레임에서 2D soft‑argmax로 $(\hat x_i(t), \hat y_i(t))$를 얻고, 평균 풀링+exp/sigmoid를 통해 깊이와 가시성을 얻는 형태로 구현됩니다.[^1]

### 4.2 깊이·트랙 깊이의 스케일 불변 손실 (SILog)

깊이와 3D 트랙 깊이에 대해 저자들은 Eigen et al.의 스케일 불변 로그(SILog) 손실을 사용합니다.[^1]

```math
L(y, y^*) = \frac{1}{N} \sum_{i=1}^{N} \left( \log y_i - \log y_i^* + \alpha(y, y^*) \right)^2 \quad [^3]
```

```math
\alpha(y, y^*) = \frac{1}{N} \sum_{i=1}^{N} \left( \log y_i^* - \log y_i \right) \quad [^4]
```

여기서 $y$, $y^*$는 예측/GT 깊이(또는 트랙 깊이)이고, $e^{\alpha}$는 예측을 GT에 최적으로 맞추는 전역 스케일입니다.[^1]
저자들은 먼저 깊이에 대해 $\alpha_{\text{depth}}$를 구한 뒤, 3D 트랙 깊이 및 포즈 손실도 이 스케일을 공유하도록 강제해 깊이·트랙·포즈 사이의 **공통 스케일 일관성**을 확보합니다.[^1]

### 4.3 멀티태스크 손실 구조

전체 손실은 깊이, 광류, 2D 트랙, 트랙 깊이, 트랙 가시성, 모션 세그멘테이션, 카메라 포즈에 대한 손실을 가중합한 형태로 주어집니다.[^1]

```math
L_{\text{total}} = \lambda_{\mathrm{depth}} L_{\mathrm{depth}}
+ \lambda_{\mathrm{flow}} L_{\mathrm{flow}}
+ \lambda_{\mathrm{track2D}} L_{\mathrm{track2D}}
+ \lambda_{\mathrm{trackD}} L_{\mathrm{trackD}}
+ \lambda_{\mathrm{vis}} L_{\mathrm{vis}}
+ \lambda_{\mathrm{seg}} L_{\mathrm{seg}}
+ \lambda_{\mathrm{pose}} L_{\mathrm{pose}} \quad [^5]
```

각 항은 SILog(깊이·트랙 깊이), $L_1$(광류·트랙 2D·Plücker ray), BCE(가시성·세그멘테이션)로 정의되며, 가중치는 손실 크기를 같은 오더로 맞춘 뒤 소규모 검색으로 튜닝했다고 보고합니다.[^1]

***

## 5. 모델 구조 상세

### 5.1 VideoMAEv2 백본

- 입력: $16 \times 224 \times 224$ 비디오 클립, 패치 크기 $2 \times 14 \times 14$.[^1]
- 출력: $P = 2048$개의 토큰, 임베딩 차원 $C = 1408$, 40층 ViT 인코더.[^1]
- VideoMAEv2는 1.35M 비디오 클립에 대해 masked autoencoding으로 사전학습되어, 액션 인식 등 고수준 태스크에서 강한 표현력을 보인 백본입니다.[^1]


### 5.2 Dense DPT‑기반 헤드

DPT는 원래 단일 이미지 깊이 추정을 위해 설계된 Transformer 기반 디코더로, 여러 계층의 토큰을 점진적으로 업샘플링·결합해 고해상도 dense 맵을 생성합니다.[^6][^1]
L4P에서는 Video 토큰을 3D feature map으로 리쉐이프하고, 2D convolution을 3D convolution으로 교체하여 시간축 정보를 처리하도록 확장한 뒤, 마지막 레이어에서 깊이(1채널), 광류(2채널), 모션 마스크(1채널), 카메라 rays(16×16×6 Plücker 좌표) 등을 출력합니다.[^1]

### 5.3 Sparse SAM‑style 헤드와 메모리

Sparse 헤드는 SAM의 prompt‑encoding + mask‑decoding 구조를 비디오 및 3D 트랙에 맞게 확장한 것입니다.[^7][^1]

- 입력 토큰: 쿼리 포인트 토큰 $P$ (3D positional encoding + learnable embedding)와 트랙 특징 토큰 $F$.[^1]
- 출력 토큰: heatmap $H$, depth $D$, visibility $V$ 각각에 대응하는 learnable 토큰.[^1]
- 두‑방향 attention: 위 입력/출력 토큰이 Video 토큰 $S$와 2‑layer two‑way attention을 통해 상호작용한 뒤, 3D conv 기반 mask decoder로 $T \times H \times W$ 맵을 생성.[^1]

장길이 비디오($S > T$)를 위해, 인접 윈도우 간 겹치는 프레임에서의 **트랙 특징 토큰 $F$**와 **디코딩된 Video 토큰**을 다음 윈도우로 전달하는 메모리 메커니즘을 도입해, 단순 체이닝보다 드리프트와 누락을 크게 줄입니다.[^1]

***

## 6. 학습 커리큘럼과 데이터

### 6.1 합성 데이터 기반 멀티태스크 학습

모델은 완전히 합성 데이터만으로 파인튜닝되며, 네 가지 주요 데이터셋을 사용합니다.[^1]

- Kubric: 다수 객체 상호작용, 깊이·광류·2D/3D 트랙 GT 제공.[^8][^1]
- PointOdyssey: 긴 3D 트랙을 포함한 사람/객체 상호작용 비디오, 깊이·포즈·트랙 GT 제공.[^1]
- DynamicReplica: 동적 stereo 장면, 깊이·광류·포즈·트랙 GT 제공.[^9][^1]
- TartanAir: 다양한 실내/실외 시뮬레이션 장면, 깊이·광류·포즈 GT 제공.[^1]

실제 벤치마크(ScanNet, KITTI, Spring, TAPVid‑3D 등)는 파인튜닝에 사용하지 않아, 제로샷 일반화 성능을 평가할 수 있습니다.[^1]

### 6.2 3‑단계 커리큘럼

저자들은 VideoMAE 사전학습 표현을 최대한 활용하기 위해 3단계 커리큘럼을 사용합니다.[^1]

1. **Stage 1**: Kubric만 사용, 깊이·광류·트래킹을 단일 16‑프레임 윈도우에서 end‑to‑end 학습.[^1]
2. **Stage 2**: 네 데이터셋을 모두 사용, 여전히 단일 윈도우에서 모든 태스크를 멀티태스크로 학습 (배치 내에서 각 데이터셋을 균형 있게 샘플링하여 모든 태스크에 항상 신호가 가도록 구성).[^1]
3. **Stage 3**: 온라인 트래킹·메모리 학습을 위해 길이 40 프레임, stride 8의 unrolled 윈도우 학습을 수행하되, VideoMAE의 상위 3층(37–39층)과 sparse 헤드만 미세 조정.[^1]

이 전략 덕분에 메모리 메커니즘으로 장거리 트래킹을 개선하면서도 깊이·광류 성능은 유지할 수 있음을 ablation에서 보여줍니다.[^1]

***

## 7. 성능 및 일반화 능력

### 7.1 깊이 추정 (Video Depth)

DepthCrafter, MonST3R, ChronoDepth, NVDS, Marigold, DepthAnything 등과 비교해, L4P는 ScanNet, KITTI, Sintel, Bonn 등 네 개의 비디오 데이터셋에서 일관되게 더 낮은 AbsRel과 더 높은 $\delta_1$를 기록합니다.[^1]

예를 들어, ScanNet에서 L4P는 AbsRel 0.071, $\delta_1 = 0.953$으로 DepthCrafter(0.123, 0.856)를 크게 상회하며, KITTI에서도 AbsRel 0.084, $\delta_1 = 0.935$로 MonST3R 및 DepthCrafter보다 우수한 상대 깊이 추정을 보입니다.[^1]
NYUv2 단일 이미지에서는 비디오 문맥이 부족해 성능이 뒤처지지만, 이는 비디오 전용 학습 때문이라고 명시합니다.[^1]

### 7.2 광류 (Optical Flow)

Spring, Sintel, Virtual KITTI에 대한 16‑프레임 입력(224×224)으로 RAFT(2‑프레임), MemFlow(멀티프레임)와 비교했을 때, L4P는 모든 데이터셋에서 EPE와 EPE\<1 비율 측면에서 경쟁력 있는 성능을 보입니다.[^10][^1]
예를 들어 Spring에서 L4P는 EPE 0.09, EPE\<1 98.7%로 RAFT(0.13, 98.4%)와 MemFlow(0.11, 98.7%)를 소폭 상회합니다.[^1]

### 7.3 2D/3D 트래킹 (TAPVid‑3D)

TAPVid‑3D 전체 평가에서, L4P는 3D tracking 기준(3D‑AJ, APD, OA)에서 SpaTracker 및 TAPIR‑3D, COLMAP 기반 파이프라인들을 평균적으로 능가합니다.[^11][^1]
예를 들어 3D‑AJ/전체 평균에서 SpaTracker는 9.0, 15.5, 83.7(OA)인 반면, L4P는 12.0, 19.0, 88.5로 더 높은 기하 정확도와 가시성 예측을 달성합니다.[^1]

2D 트래킹에서는 BootsTAPIR, CoTracker 등에 비해 다소 낮은 성능을 보이지만, 해상도를 높이거나 2D 트래킹 전용으로 파인튜닝하면 상당 부분 격차를 줄일 수 있음을 “L4P (2D Only)” 실험으로 보여줍니다.[^12][^1]

### 7.4 추가 태스크 (모션 세그멘테이션, 포즈)

RigidMask와 비교한 모션 기반 세그멘테이션에서, L4P는 Virtual KITTI와 Spring 둘 다에서 foreground IoU를 크게 향상시키며, 특히 VKITTI에서 RigidMask‑Drive(36.5) 대비 56.0 IoU를 기록합니다.[^1]
카메라 포즈에서는 DUSt3R, Spann3R, CUT3R, DROID‑SLAM 등과 비교해, feedforward 계열 중에서는 ATE/RPE 기준으로 competitive 또는 더 나은 성능을 보이며, 특히 Sintel·ScanNet에서 end‑to‑end fine‑tuning 버전이 강력한 성능을 달성합니다.[^13][^1]

***

## 8. 일반화 성능 향상 가능성에 대한 분석

### 8.1 사전학습 비디오 표현의 효과 (Scaling 4D representations와의 연결)

L4P는 VideoMAEv2 백본을 단순히 “고정 특징 추출기”로 쓰지 않고, 멀티태스크 셋업에서 end‑to‑end로 추가 파인튜닝할 때 성능이 크게 향상됨을 ablation으로 보입니다.[^1]
예를 들어 깊이 AbsRel은 “from scratch” 모델의 0.274에서, VideoMAE를 고정한 경우 0.140, 전체를 태스크별로 학습하면 0.108, 멀티태스크 학습(L4P, no‑memory)에서 0.103까지 개선되며, optical flow와 3D tracking에서도 유사한 패턴을 보입니다.[^1]

이는 대규모 비디오 MAE를 4D 태스크에 맞춰 적절히 scale하고 파인튜닝하면, Carreira et al.의 “Scaling 4D Representations”에서 보고된 것처럼, 포즈·트래킹·깊이 같은 비‑시맨틱 4D 태스크에 대해 모델 크기와 학습 데이터가 증가할수록 성능이 계속 향상될 수 있음을 시사합니다.[^14][^15]

### 8.2 합성 멀티도메인 학습과 제로샷 실세계 일반화

L4P는 Kubric, PointOdyssey, DynamicReplica, TartanAir 등 합성 데이터만으로 학습하고도, ScanNet·KITTI·Spring·TAPVid‑3D 같은 실세계 데이터에 제로샷으로 일반화합니다.[^8][^9][^1]
이는 DepthAnything V2처럼 대규모 합성 데이터를 중심으로 학습하여 실세계에 제로샷 일반화하는 최근 추세와 맞닿아 있지만, L4P는 **단일 백본에서 여러 태스크를 동시에 학습**했다는 점이 다릅니다.[^16][^17]

### 8.3 멀티태스크 학습이 일반화에 미치는 영향

Ablation에서 “각 태스크별 독립 모델”과 “멀티태스크 모델(L4P, no‑memory)”의 성능이 거의 동일함을 보여주며, 이는 하나의 백본을 공유하면서도 특정 태스크 성능이 크게 희생되지 않음을 의미합니다.[^1]
또한 메모리 메커니즘을 추가한 최종 L4P는 3D 트래킹 성능(2D‑AJ/3D‑AJ)을 유의미하게 향상시키면서, 깊이·광류 성능은 유지하여, **장거리 시계열 정보를 추가적으로 활용해도 다른 태스크의 일반화 성능이 저하되지 않음**을 보여줍니다.[^1]

***

## 9. 한계점

- **입력 해상도 224×224**: VideoMAE 사전학습 구성에 맞춘 이 해상도는 많은 실제 응용(예: 자율주행, 고해상도 비디오 편집)에 필요한 세밀한 공간 정보를 충분히 활용하지 못합니다.[^1]
저자들은 DINOv2 스타일의 고해상도 파인튜닝이나 convex upsampling layer 도입으로 완화 가능하다고 언급합니다.[^18][^1]
- **단일 이미지 깊이 약점**: L4P는 비디오 전용 학습 때문에 NYUv2 같은 단일 이미지 데이터셋에서 DepthAnything, Marigold 등 diffusion 기반/이미지 기반 방법들보다 성능이 떨어집니다.[^19][^16][^1]
- **실시간 제약**: A6000 기준 16프레임당 약 300ms(프레임당 ~19ms, 윈도우 중첩까지 고려하면 ~28ms)로, 다수의 실시간 시스템에는 여전히 빡빡할 수 있습니다.[^1]
- **2D 전용 트래킹 SOTA와의 격차**: BootsTAPIR, CoTracker 등 2D 트래킹 특화 모델과 비교하면 2D‑AJ 기준 소폭 열세이며, 이는 낮은 해상도와 task‑specific trick 부재에 기인한다고 분석합니다.[^20][^12][^1]

***

## 10. 2020년 이후 관련 최신 연구 비교

L4P를 중심으로, 2020년 이후 저수준 4D 인지·재구성 분야의 주요 공개 연구들을 간단히 비교하면 다음과 같습니다.

### 10.1 대표 논문 비교 표

| 논문 (연도) | 주요 태스크 | 핵심 아이디어/표현 | L4P와의 관계 |
| :-- | :-- | :-- | :-- |
| **L4P (2025)**[^1][^2][^21] | 깊이, 광류, 2D/3D 트랙, 모션 세그멘트, 포즈 | VideoMAEv2 백본 + DPT/ SAM‑style 헤드, dense·sparse 4D 태스크 통합 | 합성 데이터 기반 멀티태스크 4D perception의 통합 아키텍처를 제시 |
| **DepthCrafter (CVPR 2025)**[^22][^23][^24][^13] | 비디오 깊이 | 이미지→비디오 diffusion 모델을 video‑to‑depth로 전이, 장길이 비디오를 세그먼트 단위로 추정·스티칭 | 깊이만을 위한 diffusion 기반 SOTA; L4P의 비디오 깊이 성능 비교 기준 |
| **Depth Anything V2 (NeurIPS 2024)**[^25][^16][^17][^26] | 단일 이미지 깊이 | 대규모 합성+실세계 pseudo‑label로 단일 이미지 depth foundation model을 구축 | 이미지 기반 depth foundation에 초점, L4P는 비디오 기반·멀티태스크에 초점 |
| **DUSt3R (CVPR 2024)**[^11][^18][^27] | 포인트맵 기반 3D 재구성, 깊이, 상대 포즈 | 두 이미지에서 pointmap을 직접 회귀해 다양한 3D 태스크를 통합 | 정적 장면 중심의 3D 통합 프레임워크로, L4P의 카메라 포즈 평가 기준 중 하나 |
| **MonST3R (2024)**[^28][^29] | 동적 장면에서의 비디오 깊이·포즈 | DUSt3R의 pointmap 표현을 동적 장면에 확장, timestep별 pointmap 추정 | 동적 장면에 대한 geometry‑first 접근으로, L4P와 비디오 깊이·포즈에서 비교됨 |
| **MemFlow (CVPR 2024)**[^10][^30][^31] | 광류 추정·예측 | 메모리 버퍼를 활용한 실시간 optical flow + 미래 flow 예측 | L4P의 multi‑frame flow와 대비, L4P는 보다 general‑purpose 백본을 사용 |
| **Scaling 4D Representations (2024)**[^14][^32][^15] | 포즈, 트래킹, 깊이 (4D representation) | 대규모 비디오 MAE 모델(최대 22B)로 비시맨틱 4D 태스크의 스케일링 효과 분석 | L4P가 사용하는 VideoMAE 계열이 4D 태스크에 잘 스케일링된다는 이론적·실험적 근거 제공 |
| **D4RT (2025)**[^33][^6][^34] | 4D 재구성·트래킹 | Scene representation transformer 기반 encoder + point‑query decoder로, 시공간 점 질의를 통해 4D 정보 추출 | L4P처럼 단일 feedforward 모델로 깊이·카메라·트래킹을 통합하나, decoder 인터페이스를 완전히 “point query”로 통합 |
| **Flow4R (2026)**[^35][^36][^37] | 4D 재구성 및 트래킹 | 카메라 공간 scene flow를 중심 표현으로 사용, 2‑view 입력에서 3D 위치·scene flow·pose weight를 예측 | 4D 표현을 “scene flow”로 중심화한 통합 프레임워크로, L4P의 “VideoMAE feature + 헤드” 설계와 대비되는 설계 철학 제시 |

요약하면, L4P는 DepthCrafter/Depth Anything이 주로 깊이 하나에 초점을 맞춘 것과 달리, VideoMAE 기반 강력한 비디오 표현을 공유 백본으로 사용해 **여러 저수준 4D 태스크를 동시에 해결**하려는 점에서 DUSt3R, D4RT, Flow4R과 더 직접적으로 비교할 수 있는 “multi‑task 4D perception” 계열 연구입니다.[^22][^33][^36][^14][^1]

***

## 11. 향후 연구에의 영향과 연구 시 고려할 점

### 11.1 비디오 foundation model과 4D perception의 접점

L4P는 VideoMAE 같은 비디오 self‑supervised 모델이 액션 인식 등 고수준 태스크뿐 아니라, 깊이·광류·트래킹 같은 **저수준 4D perception에도 효과적으로 전이**될 수 있음을 실증합니다.[^14][^1]
이는 향후 VLM(예: VideoChat, InternVideo2 등)에서 저수준 4D 인지 능력을 강화하기 위해, L4P 스타일의 low‑level 4D 학습을 백본 사전학습에 포함시키는 방향을 뒷받침합니다.[^14][^1]

### 11.2 설계 관점에서의 시사점

- **공통 백본 + 태스크별 헤드 분리**: dense/sparse 태스크를 한 모델에서 처리하되, 헤드 수준에서만 특화하는 설계가 멀티태스크 성능과 학습 안정성 측면에서 유효함을 보여줍니다.[^1]
- **heatmap 기반 트래킹 정식화**: sparse 트래킹을 2D heatmap 회귀 문제로 캐스팅해 dense 태스크와 representation을 공유하는 아이디어는, 향후 “segment/track anything” 계열 모델 설계에도 직접적으로 활용될 수 있습니다.[^38][^20][^1]
- **메모리·윈도우 기반 온라인 처리**: 고정 길이 VideoMAE에 메모리 토큰과 cross‑window 피처 전달을 추가하는 방식은, 긴 시퀀스를 다루는 다른 비디오 foundation model에도 일반적으로 적용 가능한 패턴입니다.[^38][^10][^1]


### 11.3 향후 연구 시 구체적인 고려 사항

1. **해상도 및 효율성 스케일링**
    - 224×224 해상도 한계를 고려하면, DINOv2 스타일의 고해상도 파인튜닝, multi‑scale tokenization, 혹은 convex upsampling layer 도입 등으로 high‑res 4D perception을 다루는 연구가 필요합니다.[^16][^18][^1]
    - D4RT, Flow4R처럼 디코더를 sparse/point‑query 형태로 설계하면, 고해상도에서도 효율성을 유지할 수 있는 대안이 될 수 있습니다.[^33][^36]
2. **합성·실세계 데이터 혼합 전략**
    - L4P, Depth Anything V2, DepthCrafter 모두 대규모 합성/비지도 데이터에 강하게 의존하고 있어, 합성→실세계 도메인 갭을 체계적으로 분석·완화하는 연구(예: domain randomization, test‑time adaptation, self‑training)가 중요합니다.[^25][^24][^1]
3. **4D foundation과 다운스트림(로보틱스·생성 모델) 연계**
    - MoSca, ST‑VLA, MMPhysVideo, Free4D, 4D‑RGPT 등은 4D 인지를 로봇 조작, 4D 비디오 생성, MLLM으로 연결하려는 흐름을 보이고 있어, L4P 스타일의 4D perception 모듈을 이들 시스템의 저수준 perception 프런트엔드로 통합하는 방향이 유망합니다.[^39][^40][^28][^20][^1]
4. **표현 중심의 통합 설계 비교 연구**
    - L4P(“VideoMAE feature + 태스크 헤드”), DUSt3R/MonST3R(“pointmap 중심”), Flow4R(“scene flow 중심”), D4RT(“scene representation + point query”)처럼 서로 다른 중심 표현을 사용하는 통합 4D 모델들이 등장하고 있습니다.[^28][^36][^11][^33]
    - 향후 연구에서는 이러한 표현 선택이 일반화, 데이터 효율, 다운스트림 전이(예: 로봇 조작, 4D 생성)에 미치는 영향을 체계적으로 비교하는 것이 중요합니다.

***

## 참고한 공개 논문·자료 목록 (제목 기준)

- Abhishek Badki et al., **“L4P: Towards Unified Low-Level 4D Vision Perception”**, arXiv:2502.13078 / 3DV 2026 (oral).[^21][^41][^7][^2][^1]
- Wenbo Hu et al., **“DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos”**, CVPR 2025 / arXiv:2409.02095.[^23][^24][^22][^13]
- Lihe Yang et al., **“Depth Anything V2”**, NeurIPS 2024 / arXiv:2406.09414.[^17][^26][^25][^16]
- Shuzhe Wang et al., **“DUSt3R: Geometric 3D Vision Made Easy”**, CVPR 2024 / arXiv:2312.14132.[^27][^11][^18]
- Jiahao Shao et al., **“Learning Temporally Consistent Video Depth from Video Diffusion Priors (ChronoDepth)”**, arXiv:2406.01493.[^42][^1]
- Junyi Zhang et al., **“MonST3R: A Simple Approach for Estimating Geometry in the Presence of Motion”**, arXiv:2410.03825.[^29][^28]
- Qiaole Dong and Yanwei Fu, **“MemFlow: Optical Flow Estimation and Prediction with Memory”**, CVPR 2024 / arXiv:2404.04808.[^30][^31][^10]
- João Carreira et al., **“Scaling 4D Representations”**, arXiv:2412.15212.[^32][^15][^14]
- “Efficiently Reconstructing Dynamic Scenes One D4RT at a Time (D4RT)”**, arXiv:2512.08924.[^34][^33][^6]
- Shenhan Qian et al., **“Flow4R: Unifying 4D Reconstruction and Tracking with Scene Flow”**, arXiv:2602.14021.[^35][^36][^37]
- Zhan Tong et al., **“VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training”**, NeurIPS 2022.[^31][^9][^1]
- René Ranftl et al., **“Vision Transformers for Dense Prediction (DPT)”**, ICCV 2021.[^6][^1]
- Alexander Kirillov et al., **“Segment Anything”**, ICCV 2023.[^7][^1]
- Skanda Koppula et al., **“TAPVid-3D: A Benchmark for Tracking Any Point in 3D”**, arXiv:2404.xxxx.[^11][^1]

위 자료들만을 근거로 설명을 구성했으며, 명시되지 않은 수식·구조·수치는 추가로 가정하지 않았습니다.
<span style="display:none">[^43][^44][^45][^46][^47][^48][^49][^50][^51][^52][^53][^54][^55][^56][^57][^58][^59][^60][^61][^62][^63][^64]</span>

<div align="center">⁂</div>

[^1]: 2502.13078v3.pdf

[^2]: https://arxiv.org/html/2502.13078v3

[^3]: https://arxiv.org/abs/2502.13078

[^4]: https://arxiv.org/html/2512.01383v1

[^5]: https://arxiv.org/html/2512.17012v4

[^6]: https://arxiv.org/abs/2512.08924

[^7]: https://research.nvidia.com/labs/lpr/l4p/

[^8]: https://www.semanticscholar.org/paper/d095fe4ed26763eb6023fa5cc1e91debf1670dd1

[^9]: https://taek-guen.tistory.com/51

[^10]: https://arxiv.org/abs/2404.04808

[^11]: https://arxiv.org/abs/2312.14132

[^12]: https://jglobal.jst.go.jp/en/detail?JGLOBAL_ID=202502213222563607

[^13]: https://huggingface.co/papers/2409.02095

[^14]: https://arxiv.org/abs/2412.15212

[^15]: https://huggingface.co/papers/2412.15212

[^16]: https://arxiv.org/html/2406.09414v2

[^17]: https://depth-anything-v2.github.io

[^18]: https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_DUSt3R_Geometric_3D_Vision_Made_Easy_CVPR_2024_paper.pdf

[^19]: http://paperreading.club/page?id=285368

[^20]: https://arxiv.org/html/2604.02817v1

[^21]: http://arxiv.org/pdf/2502.13078.pdf

[^22]: https://arxiv.org/html/2409.02095v2

[^23]: https://openaccess.thecvf.com/content/CVPR2025/html/Hu_DepthCrafter_Generating_Consistent_Long_Depth_Sequences_for_Open-world_Videos_CVPR_2025_paper.html

[^24]: https://arxiv.org/abs/2409.02095

[^25]: https://arxiv.org/abs/2406.09414

[^26]: https://github.com/DepthAnything/Depth-Anything-V2

[^27]: https://arxiv.org/html/2312.14132v2

[^28]: https://arxiv.org/abs/2410.03825

[^29]: https://arxiv.org/html/2410.03825v2

[^30]: https://openaccess.thecvf.com/content/CVPR2024/papers/Dong_MemFlow_Optical_Flow_Estimation_and_Prediction_with_Memory_CVPR_2024_paper.pdf

[^31]: https://cvpr.thecvf.com/virtual/2024/poster/30462

[^32]: https://arxiv.org/html/2412.15212v1

[^33]: https://arxiv.org/html/2512.08924v1

[^34]: https://huggingface.co/papers/2512.08924

[^35]: https://arxiv.org/html/2602.14021v1

[^36]: https://arxiv.org/abs/2602.14021

[^37]: https://www.arxiv.org/abs/2602.14021

[^38]: https://arxiv.org/html/2512.13684v2

[^39]: https://arxiv.org/html/2512.17012v3

[^40]: https://www.semanticscholar.org/paper/59400851a87d90eae1c2f29ed1e33f025c4356f1

[^41]: https://openreview.net/forum?id=QjXSTzq6AZ

[^42]: https://linkinghub.elsevier.com/retrieve/pii/S0010482525015707

[^43]: https://arxiv.org/html/2502.13078v1

[^44]: https://arxiv.org/html/2512.08924v2

[^45]: https://arxiv.org/html/2512.07821v1

[^46]: https://arxiv.org/pdf/2602.14021.pdf

[^47]: https://arxiv.org/html/2502.13078v2

[^48]: https://arxiv.org/pdf/2502.13078.pdf

[^49]: https://arxiv.org/html/2412.15212v2

[^50]: https://www.semanticscholar.org/paper/34a96fb3052d35c058860ae8ec4efdf47201f590

[^51]: https://ieeexplore.ieee.org/document/10422325/

[^52]: https://arxiv.org/abs/2205.14332

[^53]: https://arxiv.org/html/2503.20785v1

[^54]: http://arxiv.org/pdf/2502.00721.pdf

[^55]: https://www.themoonlight.io/de/review/l4p-low-level-4d-vision-perception-unified

[^56]: https://www.themoonlight.io/en/review/l4p-low-level-4d-vision-perception-unified

[^57]: https://openreview.net/pdf/c1a046f8958d7fce568aa952968f9f73cb4b10a9.pdf

[^58]: https://ieeexplore.ieee.org/document/11092953/

[^59]: https://dl.acm.org/doi/10.1145/3763330

[^60]: https://arxiv.org/abs/2602.01661

[^61]: https://arxiv.org/abs/2509.20624

[^62]: https://ieeexplore.ieee.org/document/11112569/

[^63]: https://ieeexplore.ieee.org/document/10655581/

[^64]: https://arxiv.org/abs/2509.07676

