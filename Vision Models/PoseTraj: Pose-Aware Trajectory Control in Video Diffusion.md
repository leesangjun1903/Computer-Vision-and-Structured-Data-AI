# PoseTraj: Pose-Aware Trajectory Control in Video Diffusion

---

## 1. 핵심 주장 및 주요 기여 요약

**PoseTraj**는 2D 트래젝토리 입력만으로 객체의 6D 포즈(위치 + 방향) 변화를 인지하여 3D 정합된 영상을 생성하는 **포즈 인지(pose-aware) 비디오 드래깅 모델**입니다. 기존 DragNUWA, DragAnything 등이 2D 픽셀 공간에서의 평행 이동(translation)에는 강하지만 회전(rotation)을 동반한 트래젝토리에 대해서는 객체 붕괴(entity collapse)나 카메라 무빙으로 회피하는 한계를 가졌던 점을 지적합니다.

**주요 기여 3가지:**

| 기여 | 내용 |
|---|---|
| ① 두 단계 포즈 인지 사전학습 | 3D 바운딩 박스를 **중간 감독 신호(intermediate supervision)** 로 활용하여 6D 포즈 이해 능력을 주입 |
| ② PoseTraj-10K 합성 데이터셋 | Objaverse 기반 2,000개 객체 × 5개 트래젝토리 = 10,000개 영상, 회전 모션 + 3D bbox 동시 어노테이션 |
| ③ 카메라 분리(Camera-disentangled) 파인튜닝 | 실제 영상의 카메라 모션과 객체 모션을 분리 학습, 일반화 성능 향상 |

---

## 2. 문제 정의, 제안 방법, 모델 구조

### 2.1 해결하고자 하는 문제

기존 트래젝토리 기반 비디오 생성 모델의 한계는 두 가지 근본 원인에서 비롯됩니다:

1. **데이터 편향**: 수집된 실세계 영상 데이터셋에서 회전 모션은 드물고, 자동 어노테이션이 어려움
2. **본질적 모호성(ill-posed)**: 2D 트래젝토리만으로 3D 회전 정보를 추론하는 것은 정보 부족 문제

이를 수식적으로 표현하면, 입력 이미지 $I \in \mathbb{R}^{H \times W \times 3}$ 와 트래젝토리 $\mathbf{tr} = \{(x_i, y_i)\}\_{i=1}^{L}$ 가 주어졌을 때, 객체가 트래젝토리를 정확히 따르면서 일관된 외형을 유지하는 비디오 시퀀스 $\{f_i\}_{i=1}^{L}$ 를 생성하는 것이 목표입니다.

### 2.2 모델 구조

베이스 모델로 **SVD (Stable Video Diffusion)** 를 채택하고, 다음 구성요소로 구성됩니다:

- **Latent Diffusion Model** $\epsilon_\theta$ (3D U-Net) — latent noise 디노이징
- **Encoder/Decoder** $(\mathcal{E}, \mathcal{D})$ — latent 공간 압축 및 복원
- **Traj-ControlNet** — SVD 인코더 블록의 학습 가능한 복사본 (ControlNet 아이디어 차용)
- **Trajectory Guider** — 3D ConvNet으로 트래젝토리 이미지 시퀀스 인코딩
- **Camera MLP + Projection Layer** — 카메라 외부 파라미터를 융합

### 2.3 핵심 수식

**(1) MSE 손실 (모든 단계 공통)**

$$\mathcal{L}_{\text{MSE}} = \mathbb{E}_{x_t, \epsilon} \left[ \sum_{i=1}^{L} \left\| \epsilon - \epsilon_\theta(x_t, t, C^i) \right\|^2_{2} \right]$$

**(2) 단계별 조건부 입력 $C^i$**

$$C^i = \begin{cases} \{I_{tr}^i,\ \mathbf{I}_{bbox}\}, & \text{stage one (3D bbox 감독)} \\ \{I_{tr}^i,\ \mathbf{I}\}, & \text{stage two (외형 정제)} \\ \{I_{tr}^i,\ \mathbf{I},\ \text{Cam}^i\}, & \text{finetuning (카메라 분리)} \end{cases}$$

여기서 Stage 1에서는 **bbox-증강 비디오 프레임**이 재구성 타깃이 됩니다.

**(3) 공간 강화 손실 (Spatial Enhancement Loss)**

대형 회전 모션에서 객체 붕괴를 막기 위해 단일 프레임 재구성 보조 과제를 도입:

$$\mathcal{L}_{\text{SPA}} = \|\epsilon_j - \epsilon_\theta(x_{t,j}, t, C^j)\|^2_2,\quad j \in (1 \sim L)$$

역전파 시 **공간 레이어만 업데이트**되어 프레임별 공간 정확도에 집중합니다.

**(4) 통합 손실**

$$\mathcal{L}_{\text{all}} = \mathcal{L}_{\text{MSE}} + \lambda_{\text{SPA}} \cdot \mathcal{L}_{\text{SPA}}$$

### 2.4 학습 파이프라인 (3단계)

1. **Stage 1 — 3D bbox 가이드 위치 추정**: 합성 데이터에서 객체와 함께 3D bbox를 픽셀 공간에 그려 동시 생성하도록 학습. 객체 위치/회전 인지 능력 주입.
2. **Stage 2 — 객체 중심 재구성**: bbox 감독 제거, 외형 디테일 정제.
3. **카메라 분리 파인튜닝**: VIPSeg 실세계 데이터에서 카메라 외부 파라미터 $\{\text{Cam}\_i\}_{i=1}^{L}$ 를 MLP로 임베딩, 50% 확률로 드롭아웃하여 추론 시 카메라 정보 없이도 동작 가능하게 함.

### 2.5 성능 결과

| 데이터셋 / 해상도 | 모델 | ObjMC ↓ | FID ↓ | FVD ↓ |
|---|---|---|---|---|
| VIPSeg 256×256 | DragAnything | 100.23 | 61.69 | 410.70 |
| VIPSeg 256×256 | **Ours** | **87.56** | **46.60** | **384.41** |
| VIPSeg 320×576 | DragAnything | 91.12 | 39.29 | 275.93 |
| VIPSeg 320×576 | **Ours** | **77.48** | **38.41** | **267.33** |
| DAVIS 320×576 (OOD) | DragAnything | 47.01 | 50.83 | 771.78 |
| DAVIS 320×576 (OOD) | **Ours** | **29.92** | 51.48 | **729.16** |

ObjMC 기준 각각 **13%, 15%, 36%** 개선되었으며, 특히 OOD 환경(DAVIS)에서 36%의 큰 폭 개선이 일반화 성능을 시사합니다.

### 2.6 한계점 (저자 자술)

1. **동적 객체의 광범위 회전 처리 미흡**: 자동차/비행기는 잘 되지만 사람/동물 등 자체적으로 동적 변형이 있는 객체에 약함 (학습 데이터에 없음)
2. **카메라 제어 정밀도 부족**: 분리 모듈은 객체 모션 정확도 향상에는 기여하나, 정밀한 카메라 제어 능력은 제한적
3. **대형 모션에서의 배경 흐림(blur)**: SVD 베이스 모델 자체 한계 + 학습 데이터 품질 영향

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

이 논문에서 일반화 성능에 직접적으로 기여하는 설계 요소는 다음과 같습니다.

### 3.1 합성→실세계 도메인 갭 극복 전략

PoseTraj는 합성 데이터(Blender 렌더링)에서 학습한 3D 인지 능력을 실세계로 전이하기 위해 **2단계 분리(decoupling) 전략**을 사용합니다:

- **Stage 1**의 bbox 시각화는 본질적으로 합성 데이터 특유의 **시각적 아티팩트**(체크무늬 바닥, 단순 조명)를 학습하지 않도록 설계됨. 모델이 외형보다는 **위치-포즈 매핑**에 집중하도록 강제.
- **Stage 2**에서 bbox를 제거함으로써 추론 시 추가 신호 추정 오류로 인한 **inference-stage mismatch**를 회피. 저자가 강조하는 **injection-by-reconstruction** 철학이 핵심.

### 3.2 카메라 분리 모듈의 일반화 기여

실세계 비디오에서 객체 모션과 카메라 모션은 강하게 결합되어 있습니다. 카메라 외부 파라미터 $\text{Cam}^i$ 를 명시적 조건으로 추가하고 학습 시 50% 확률로 드롭아웃함으로써:

$$p(\text{video}|\text{traj}, I, \text{Cam}) \approx p(\text{video}|\text{traj}, I)\quad \text{when}\ \text{Cam=}\emptyset$$

이러한 dropout-based marginalization은 추론 시 카메라 정보 없이도 자연스러운 영상을 생성할 수 있게 합니다. DAVIS OOD 데이터셋에서 **36% ObjMC 개선** (47.01 → 29.92)이 이 전략의 효과를 정량적으로 입증합니다.

### 3.3 Ablation을 통한 일반화 요인 분해

| 변형 | ObjMC | FVD | 의미 |
|---|---|---|---|
| Full method | 77.48 | 267.33 | — |
| No pretrain | 145.72 | 486.84 | 사전학습 제거 시 ObjMC 88% 악화 → 가장 큰 일반화 기여 |
| No SPA-loss | 137.26 | 436.56 | 공간 강화 손실 제거 → 77% 악화 |
| No bbox stage | 81.36 | 275.40 | bbox 감독만 제거 → 5% 악화 (시각적 collapse는 큼) |
| No Cam-disen | 83.22 | 279.15 | 카메라 분리 제거 → 7% 악화 |

이는 **합성 데이터 사전학습 자체**가 일반화의 가장 큰 동력이며, bbox 감독은 정량 메트릭보다는 **시각적 안정성** 확보에 기여한다는 점을 보여줍니다.

### 3.4 데이터 스케일링 ablation (보충자료)

| 영상 수 | 객체 수 | ObjMC ↓ | FVD ↓ |
|---|---|---|---|
| 1,000 | 200 | 0.1987 | 190.35 |
| 2,000 | 400 | 0.2065 | 187.12 |
| 5,000 | 1,000 | 0.1960 | 185.47 |
| 10,000 | 2,000 | 0.1895 | 186.01 |

**5,000개 이상에서 성능 saturation**이 관찰됩니다. 이는 합성 데이터를 무한히 늘리는 것이 능사가 아니며, 다양성이 양보다 중요함을 시사합니다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 학계에 미치는 영향

1. **3D 인지 신호의 "주입-by-재구성" 패러다임**: depth map처럼 추론 시 추가 추정이 필요한 신호 대신, 학습 시에만 사용하고 추론 시 제거 가능한 중간 감독 신호 활용은 다른 controllable generation 분야(예: 3D 인지 이미지 편집, 4D 생성)에도 확장 가능한 아이디어입니다.
2. **합성-실세계 하이브리드 학습 프로토콜**: 정밀 어노테이션이 필요한 분야에서 합성 데이터로 pretraining + 실세계로 fine-tuning이라는 설계 청사진을 제시.
3. **pose 이해의 명시적 평가 프레임워크 부재 노출**: 현재 ObjMC/FID/FVD는 모두 2D 메트릭이므로, 3D pose alignment를 직접 평가할 메트릭의 부재는 후속 연구의 기회입니다.

### 4.2 후속 연구 시 고려사항

| 고려사항 | 구체 방향 |
|---|---|
| **동적 객체 회전 부재** | Mixamo, Animal3D, SMPL-X 기반 합성 데이터 추가 (저자 명시) |
| **DiT 기반 모델로의 확장** | Tora처럼 DiT 아키텍처에 동일 전략 적용 시 재훈련 필요. CogVideoX, HunyuanVideo, Wan 등에서 동일 전략의 효용 검증 필요 |
| **카메라 제어 정밀도** | RealEstate10K 등 카메라 어노테이션 데이터셋과 조합. CameraCtrl/CineMaster 류 모듈 융합 검토 |
| **3D 평가 메트릭 부재** | 생성 영상에서 3D pose를 추출(예: FoundPose)하여 ground-truth와 비교하는 정량 평가 필요 |
| **Multi-object interaction** | 현재는 객체별 독립 트래젝토리. 충돌·접촉을 포함한 물리 일관성 학습 필요 |
| **물리적 plausibility** | 회전이 트래젝토리와 정합되더라도 중력, 관성 등 물리법칙 준수 보장 안됨 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

PoseTraj와 직접 비교 가능한 관련 연구를 시간순으로 정리합니다.

### 5.1 트래젝토리/모션 제어 계열

| 연도 | 모델 | 핵심 차별점 | PoseTraj 대비 |
|---|---|---|---|
| 2021 | **IPoke** [Blattmann et al.] | 인간 도메인 한정 stochastic 비디오 합성 | 도메인 협소 |
| 2023 | **MCDiff** [Chen et al.] | 스켈레톤 기반 모션 조건 | 도메인 협소 |
| 2023.08 | **DragNUWA** [Yin et al.] | 오픈도메인 첫 트래젝토리 드래깅, optical flow 기반 | 회전 미지원, PoseTraj가 ObjMC 41.8% 우위 |
| 2023.12 | **MotionCtrl** [Wang et al.] | 카메라/객체 모션 분리 첫 시도 | PoseTraj가 카메라 분리 + 3D 인지 모두 보유 |
| 2024.03 | **DragAnything** [Wu et al.] | 객체 마스크에서 entity representation 추출 | 회전 시 객체 붕괴, PoseTraj 직접 비교 우위 |
| 2024.07 | **Tora** [Zhang et al.] | DiT 기반, 모션 VAE로 trajectory 임베딩 | DiT 기반이라 아키텍처 다름, 회전 미해결 |
| 2024.07 | **PuppetMaster** [Li et al.] | 파트 단위 합성 데이터로 part-level 애니메이션 | 객체 일부 제어, PoseTraj와 상보적 |
| 2024.11 | **TrailBlazer** [Ma et al.] | bbox로 모션 가이드 (학습 불필요) | 2D bbox만 사용, 3D 인지 없음 |

### 5.2 3D 인지 비디오 생성 계열 (PoseTraj와 직접 경쟁)

검색 결과 PoseTraj 전후로 유사한 동기를 가진 연구들이 등장했습니다:

- **3DTrajMaster** (Fei et al., 2024.12): 3D 트래젝토리 직접 입력, 다중 엔티티 모션 제어. PoseTraj는 2D 입력에서 3D를 추론하는 차이.
- **Free-Form Motion Control / SynFMC** "FMC can control the 6D poses of objects and camera independently or simultaneously, producing high-fidelity videos." (Shuai et al., 2025.01): 6D pose 직접 어노테이션된 합성 데이터셋 SynFMC를 제안. PoseTraj보다 직접적으로 6D pose를 다루지만 입력 인터페이스가 더 복잡함.
- **CineMaster** (Wang et al., 2025): 3D bbox를 직접 입력 조건으로 사용하여 영화적 카메라 + 객체 모션 동시 제어. PoseTraj는 bbox를 **감독 신호로만** 쓰고 추론 시 제거.
- **Diffusion as Shader (DaS)** (2025): 3D point tracking 비디오를 condition으로 사용하는 일반화된 제어. 추론 시 3D tracking 입력 필요라는 점에서 PoseTraj와 차별화.

### 5.3 PoseTraj의 차별화 포지션

$$\text{PoseTraj} = \underbrace{\text{2D 입력 단순성}}_{\text{DragAnything 계열}} + \underbrace{\text{3D 인지 사전학습}}_{\text{CineMaster/3DTrajMaster 계열}} + \underbrace{\text{Inference-time 단순성}}_{\text{추가 신호 불필요}}$$

즉, **사용자 인터페이스 단순성과 3D 정확성의 절충점**을 제시한 것이 PoseTraj의 독자적 위치입니다.

---

## 6. 답변의 정확도 및 한계 명시

- **PDF 본문 기반 내용** (방법, 수식, 표 1·2·3, 그림 설명): 업로드된 논문 텍스트에서 직접 추출하여 **높은 신뢰도**.
- **2025년 이후 후속 연구 비교** (3DTrajMaster, FMC, CineMaster, DaS): 웹 검색 결과 기반이며, 각 논문의 세부 구현 차이는 원문 직접 확인이 필요합니다.
- **저자가 명시하지 않은 추론 부분** (예: dropout-based marginalization 해석)은 제 분석이며, 저자 의도와 정확히 일치하지 않을 수 있습니다.

---

## 참고 자료

1. **본 논문**: Ji, L., Zhong, L., Wei, P., Li, C. (2025). *PoseTraj: Pose-Aware Trajectory Control in Video Diffusion*. arXiv:2503.16068. https://arxiv.org/abs/2503.16068 (프로젝트: https://robingg1.github.io/Pose-Traj/)
2. **DragNUWA**: Yin et al. (2023). *DragNUWA: Fine-grained Control in Video Generation by Integrating Text, Image, and Trajectory*. arXiv:2308.08089
3. **DragAnything**: Wu et al. (2024). *DragAnything: Motion Control for Anything Using Entity Representation*. ECCV 2024. arXiv:2403.07420
4. **Tora**: Zhang et al. (2024). *Tora: Trajectory-oriented Diffusion Transformer for Video Generation*. CVPR 2025. https://github.com/alibaba/Tora
5. **MotionCtrl**: Wang et al. (2023). *MotionCtrl: A Unified and Flexible Motion Controller for Video Generation*. arXiv:2312.03641
6. **SVD (베이스 모델)**: Blattmann et al. (2023). *Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets*. arXiv:2311.15127
7. **Free-Form Motion Control (SynFMC)**: Shuai et al. (2025). arXiv:2501.01425. https://arxiv.org/abs/2501.01425
8. **3DTrajMaster**: Fei et al. (2024). *Mastering 3D Trajectory for Multi-Entity Motion in Video Generation*. arXiv:2412.07759
9. **CineMaster**: A 3D-Aware and Controllable Framework for Cinematic Text-to-Video Generation. SIGGRAPH 2025
10. **TrailBlazer**: Ma et al. (2024). *TrailBlazer: Trajectory Control for Diffusion-Based Video Generation*
11. **Objaverse**: Deitke et al. (2024). *Objaverse-XL: A Universe of 10M+ 3D Objects*. NeurIPS 2024
12. **VIPSeg**: Miao et al. (2022). *Large-scale Video Panoptic Segmentation in the Wild*. CVPR 2022
13. **DAVIS**: Perazzi et al. (2016). CVPR 2016
