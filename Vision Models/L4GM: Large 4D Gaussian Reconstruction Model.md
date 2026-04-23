# L4GM: Large 4D Gaussian Reconstruction Model

---

## 📌 참고 자료 (출처)

- **주요 논문**: Ren, J., Xie, K., Mirzaei, A., Liang, H., Zeng, X., Kreis, K., Liu, Z., Torralba, A., Fidler, S., Kim, S.W., Ling, H. "L4GM: Large 4D Gaussian Reconstruction Model." *arXiv:2406.10324v1* [cs.CV], 14 Jun 2024. https://arxiv.org/abs/2406.10324
- **기반 모델**: Tang, J., et al. "LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation." *arXiv:2402.05054*, 2024.
- **3D Gaussian Splatting**: Kerbl, B., et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering." *ACM Transactions on Graphics*, 42(4):1–14, 2023.
- **LRM**: Hong, Y., et al. "LRM: Large Reconstruction Model for Single Image to 3D." *arXiv:2311.04400*, 2023.
- **Consistent4D**: Jiang, Y., et al. "Consistent4D: Consistent 360° Dynamic Object Generation from Monocular Video." *arXiv:2311.02848*, 2023.
- **Objaverse**: Deitke, M., et al. "Objaverse: A Universe of Annotated 3D Objects." *CVPR*, 2023.
- **ImageDream**: Wang, P., Shi, Y. "ImageDream: Image-Prompt Multi-View Diffusion for 3D Generation." *arXiv:2312.02201*, 2023.
- **HexPlane**: Cao, A., Johnson, J. "HexPlane: A Fast Representation for Dynamic Scenes." *CVPR*, 2023.
- **DreamFusion**: Poole, B., et al. "DreamFusion: Text-to-3D using 2D Diffusion." *ICLR*, 2023.
- **4D Gaussian Splatting**: Wu, G., et al. "4D Gaussian Splatting for Real-Time Dynamic Scene Rendering." *arXiv:2310.08528*, 2023.

---

## 1. 핵심 주장과 주요 기여 요약

### 1.1 핵심 주장

L4GM은 **단일 시점(monocular) 비디오로부터 단 한 번의 피드포워드(feed-forward) 패스**만으로 고품질 4D 동적 객체를 초 단위(~1초)로 재구성하는 **최초의 4D Large Reconstruction Model**이다. 기존 최적화 기반 방법들(수십 분~수 시간 소요)과 비교해 **100~1,000배 빠른 추론 속도**를 달성하면서 품질 지표에서도 최고 성능을 기록하였다.

### 1.2 주요 기여

| 기여 항목 | 내용 |
|-----------|------|
| **최초 4D LRM** | 단일 시점 비디오 → 4D 3D Gaussian Splatting 시퀀스를 feed-forward로 생성 |
| **Objaverse-4D 데이터셋** | 44K 객체, 110K 애니메이션, 48 뷰포인트, 12M 비디오, 총 300M 프레임 규모의 신규 합성 4D 데이터셋 구축 |
| **Temporal Self-Attention 도입** | 기존 3D LGM에 시간축 자기-어텐션 레이어를 추가하여 시간적 일관성 확보 |
| **4D 보간 모델** | 낮은 FPS 재구성 결과를 더 높은 FPS로 업샘플링하는 4D 보간 모델 설계 |
| **강력한 일반화** | 합성 데이터만으로 훈련되었음에도 Sora, ActivityNet 등 실제(in-the-wild) 비디오에 강한 일반화 성능 시연 |

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

#### 기존 방법들의 한계

**문제 1: 데이터 요구 조건의 제약**
- 기존 4D 재구성 방법들(Dynamic 3D Gaussians 등)은 다시점(multiview) 비디오를 요구하여 데이터 수집이 매우 비용이 높음

**문제 2: 속도 문제 (Score Distillation 기반 방법)**
- 4D-Fy, Align Your Gaussians, STAG4D 등은 Video Score Distillation Sampling(SDS)을 사용
- 객체 하나당 수십 분~수 시간의 최적화 시간 필요
- 프롬프트에 민감하고 결과가 불안정

**문제 3: 단일 시점 비디오에서의 ill-posed 문제**
- 단안 비디오에서의 4D 재구성은 정보가 불충분한 역문제(ill-posed problem)
- 깊이 정보, 다중 객체 처리, 폐색 영역 복원 등이 어려움

**L4GM의 목표**: 단일 시점 비디오 $\mathcal{I} = \{I_t\}\_{t=1}^{T}$ 로부터 각 타임스텝의 3D Gaussian 집합 $\mathcal{P} = \{P_t\}_{t=1}^{T}$ 를 단 한 번의 피드포워드로 빠르게 재구성

---

### 2.2 제안하는 방법

#### 2.2.1 핵심 설계 통찰 (Two Key Insights)

**통찰 1: 3D 사전 지식의 활용 (3D Pretraining 전략)**

비디오 생성 모델(예: Stable Video Diffusion)이 이미지 사전학습 후 비디오로 파인튜닝하는 전략처럼, L4GM도 정적 3D 데이터로 대규모 사전학습된 **LGM(Large Multi-View Gaussian Model)** 을 기반으로 시작한다.

**통찰 2: 초기 타임스텝의 멀티뷰 이미지만으로 충분**

전 타임스텝에 걸친 멀티뷰 비디오 대신, **초기 프레임 $t=1$ 에서의 멀티뷰 이미지 하나**만으로 충분하다. 이후 타임스텝의 3D 정보는 Temporal Self-Attention이 전파·적응시킨다.

#### 2.2.2 멀티뷰 이미지 생성: ImageDream 활용 (Section 4.1)

입력 비디오의 첫 프레임 $I_1$을 조건으로 **ImageDream** 멀티뷰 확산 모델을 사용해 4개의 직교 뷰 이미지 $\mathcal{J}_1$을 생성한다.

방위각 정렬(Azimuth Alignment) 문제를 해결하기 위해:

$$\theta_{\text{align}} = \arg\min_{\theta} \| f(P_{\text{static}}, \theta) - I_1 \|_2^2$$

여기서 $f$는 Gaussian 볼륨 렌더링 함수. 최종 사용 뷰:

$$\mathcal{J}_1 = \{ f(P_{\text{static}}, \theta_{\text{align}} + \Delta\theta) \}_{\Delta\theta \in \{0°, 90°, 180°, 270°\}}$$

#### 2.2.3 모델 구조: 3D LGM → 4D 재구성 모델 (Section 4.2)

**입력 구성**: 생성된 멀티뷰 이미지 $\mathcal{J}_1$을 모든 타임스텝 $t \neq 1$에 복사하여 $T \times V$ 입력 그리드 구성

**카메라 임베딩**: Plücker ray embedding으로 카메라 포즈 인코딩 후 RGB 채널에 concatenation

**U-Net 처리 과정**: 입력을 $(B \cdot T \cdot V) \times H \times W \times C$ 형태로 reshape하여 비대칭 U-Net 처리

**핵심 추가 레이어: Temporal Self-Attention**

각 Cross-View Self-Attention 레이어 뒤에 추가되며, 시간 축을 배치 차원으로 처리:

$$\mathbf{x} = \text{rearrange}(\mathbf{x},\ (B\ T\ V)\ H\ W\ C \rightarrow (B\ V)\ (T\ H\ W)\ C) \tag{1}$$

$$\mathbf{x} = \mathbf{x} + \text{TempSelfAttn}(\mathbf{x}) \tag{2}$$

$$\mathbf{x} = \text{rearrange}(\mathbf{x},\ (B\ V)\ (T\ H\ W)\ C \rightarrow (B\ T\ V)\ H\ W\ C) \tag{3}$$

**출력**: U-Net 출력은 $B \times T \times V \times H_{\text{out}} \times W_{\text{out}} \times 14$ 형태의 14채널 특징 맵

각 픽셀의 14개 파라미터가 하나의 3D Gaussian Ellipsoid를 정의:

$$\text{Gaussian 파라미터}: \quad \underbrace{\mathbf{z} \in \mathbb{R}^3}_{\text{center}} \oplus \underbrace{\mathbf{s} \in \mathbb{R}^3}_{\text{scale}} \oplus \underbrace{\mathbf{q} \in \mathbb{R}^4}_{\text{rotation}} \oplus \underbrace{\alpha \in \mathbb{R}}_{\text{opacity}} \oplus \underbrace{\mathbf{c} \in \mathbb{R}^3}_{\text{color}}$$

뷰 차원 $V$를 따라 Gaussian을 concat하여 타임스텝 $t$별 단일 Gaussian 집합 $P_t$ 구성, 최종 4D 표현:

$$\mathcal{P} = \{P_t\}_{t=1}^{T}$$

각 프레임당 출력 Gaussian 수: $128 \times 128 \times 4 = 65{,}536$개

#### 2.2.4 손실 함수 (Loss Functions)

RGB 이미지 손실:

$$\mathcal{L}_{\text{RGB}} = \sum_{t=1}^{T} \sum_{O \in \mathcal{O} \cup \mathcal{O}_{\text{sup}}} \| I_t^O - f(P_t, O) \|_2^2 + \lambda \mathcal{L}_{\text{LPIPS}}(I_t^O,\ f(P_t, O)) \tag{4}$$

마스크 손실:

$$\mathcal{L}_{\text{Mask}} = \sum_{t=1}^{T} \sum_{O \in \mathcal{O} \cup \mathcal{O}_{\text{sup}}} \| \alpha_t^O - g(P_t, O) \|_2^2 \tag{5}$$

총 손실:

$$\mathcal{L} = \mathcal{L}_{\text{RGB}} + \mathcal{L}_{\text{Mask}} \tag{6}$$

여기서 $f, g$는 각각 RGB와 알파 마스크에 대한 Gaussian 볼륨 렌더링 함수, $\mathcal{O}$는 입력 카메라 포즈, $\mathcal{O}_{\text{sup}}$는 추가 감독 카메라 포즈이다.

#### 2.2.5 오토리그레시브 재구성 (Section 4.3)

긴 비디오 처리를 위한 청크 단위 순차 처리:

$$\mathcal{J}_T = \{ f(P_T,\ \Delta\theta) \}_{\Delta\theta \in \{0°, 90°, 180°, 270°\}}$$

마지막 Gaussian $P_T$를 4개 직교 방향으로 렌더링한 이미지를 다음 청크의 멀티뷰 입력으로 사용. 실험적으로 10회 이상 반복해도 품질 저하가 미미함.

#### 2.2.6 4D 보간 모델 (4D Interpolation Model)

프레임 간 Gaussian 추적(tracking)이 불가하므로 직접 궤적 보간 대신, L4GM 위에 파인튜닝된 **별도 보간 모델** 사용:

- 입력: 두 타임스텝의 멀티뷰 이미지 $\mathcal{J}\_i$, $\mathcal{J}_{i+3}$
- 중간 프레임 생성: 두 멀티뷰 이미지의 RGB 픽셀 가중 평균으로 중간 뷰 생성
- 출력: 중간 타임스텝의 Gaussian 집합 (프레임레이트 3배 업샘플링)
- 처리 속도: 프레임 쌍당 약 0.065초

---

### 2.3 Objaverse-4D 데이터셋 (Section 5)

| 항목 | 수치 |
|------|------|
| 총 객체 수 | 44K (Objaverse 1.0의 800K 중 애니메이션 있는 것) |
| 총 애니메이션 수 | 110K (객체당 다수 가능) |
| 렌더링 뷰포인트 | 48개 (고정 16개 + 랜덤 32개) |
| 총 비디오 수 | 12M (광학흐름 필터링 후) |
| 총 프레임 수 | 300M |
| 원본 프레임레이트 | 24 FPS |
| 훈련용 다운샘플링 | 8 FPS |

**모션 필터링**: 광학흐름(optical flow) 크기 < 0.15인 클립 약 50% 제거하여 12M 비디오 확보

---

### 2.4 성능 향상

#### 정량적 비교 (Consistent4D 벤치마크)

| 방법 | LPIPS↓ | CLIP↑ | FVD↓ | 시간↓ |
|------|--------|-------|------|-------|
| Consistent4D | 0.16 | 0.87 | 1133.44 | 2 hr |
| 4DGen | 0.13 | 0.89 | - | 1 hr |
| GaussianFlow | 0.14 | 0.91 | - | - |
| STAG4D | 0.13 | 0.91 | 992.21 | 1 hr |
| DG4D | 0.16 | 0.87 | - | 10 min |
| Efficient4D | 0.14 | 0.92 | - | 6 min |
| **L4GM (Ours)** | **0.12** | **0.94** | **691.87** | **3s** |

**모든 품질 지표에서 SOTA 달성, 추론 속도는 100~1,000배 빠름**

#### 사용자 연구 (User Study, Table 2)

L4GM이 모든 평가 기준에서 가장 선호됨:
- 전체 품질(Overall Quality): DG4D 대비 65.4% vs 25.0%
- 3D 외관(3D Appearance): DG4D 대비 67.1% vs 25.8%
- 3D 정렬: DG4D 대비 61.3% vs 26.3%

---

### 2.5 한계 (Limitations)

**논문에서 명시한 한계점:**

1. **운동 모호성(Motion Ambiguity)**: 보행 동작에서 입력 뷰와는 일치하지만 다른 뷰에서 다리 동작이 부자연스러운 경우 발생
2. **다중 객체 처리 불가**: 객체들이 서로 폐색(occlusion)될 때 재구성 실패
3. **에고센트릭 뷰포인트 불가**: 모델이 $0°$ elevation 입력 뷰를 가정하므로 Ego4D와 같은 1인칭 시점 비디오에서 실패
4. **합성 데이터 의존성**: 순전히 합성 데이터로만 훈련 → 실제 비디오의 미묘한 분포 차이 존재
5. **배경 처리 불가**: 단일 객체 중심으로 설계됨 (배경 분리 필요)
6. **오토리그레시브 품질 저하**: 재구성 횟수가 증가할수록 점진적 품질 감소

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 합성→실제 도메인 갭(Sim-to-Real Gap) 극복 메커니즘

L4GM은 **순수 합성 데이터(Objaverse-4D)만으로 훈련**되었음에도 실제(in-the-wild) 비디오에 강한 일반화를 보인다. 이 성공의 핵심 메커니즘은 다음과 같다:

#### 3.1.1 대규모 3D 사전학습의 역할

Ablation Study(Figure 6a)에서 확인:

$$\text{3D 사전학습 없음} \rightarrow \text{모델 수렴 실패}$$
$$\text{3D 사전학습 있음 (LGM)} \rightarrow \text{정상 수렴, 높은 PSNR}$$

LGM이 Objaverse의 방대한 3D 다양성을 통해 획득한 **형태·외관에 대한 일반적 3D 이해**가 4D 파인튜닝 시 강력한 초기화 역할을 한다.

#### 3.1.2 렌더링 전략의 일반화 배려

- **0° elevation 고정 입력 뷰** 가정: 실제 세계 비디오 대부분이 0° 고도 카메라로 촬영된다는 사실을 반영
- **랜덤 elevation 감독 뷰** ([-5°, 60°]): 다양한 각도에서 올바른 3D 구조를 학습하도록 강제
- 이 비대칭 전략이 입력 도메인 적합성과 3D 완결성을 동시에 확보

#### 3.1.3 모션 필터링을 통한 학습 효율화

광학흐름 기반 필터링으로 의미 있는 동적 콘텐츠만 학습:

$$\text{유지 조건}: \overline{\|\text{optical flow}\|} > 0.15$$

정적/거의 정적인 장면을 제거하여 모델이 실제 동적 패턴을 더 효과적으로 학습

#### 3.1.4 ImageDream를 통한 뷰 완성

단일 시점 비디오의 정보 불완전성을 멀티뷰 확산 모델로 보완. ImageDream 자체가 대규모 3D 데이터로 사전학습되어 있어 **일반화된 3D 구조 추론** 능력을 제공

### 3.2 일반화 성능 향상을 위한 향후 방향

#### 방향 1: 실제 데이터 포함 혼합 훈련

현재 Objaverse-4D는 100% 합성 데이터. 실제 비디오 데이터(ActivityNet, Ego4D 등)와 혼합 훈련 시 도메인 갭 추가 감소 가능. 단, 실제 4D Ground Truth 레이블 확보가 핵심 병목.

#### 방향 2: 도메인 적응 기법 적용

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda_{\text{domain}} \mathcal{L}_{\text{domain adaptation}}$$

Style transfer, domain randomization 등의 기법으로 합성→실제 도메인 갭을 추가로 축소할 수 있음.

#### 방향 3: 카메라 포즈 추정 통합

현재 모델은 $0°$ elevation 가정. 카메라 포즈 추정 모듈과 통합하면 에고센트릭 뷰, 드론 영상 등 다양한 촬영 조건에 적용 가능.

#### 방향 4: 대규모 비디오-4D 데이터셋 확장

Objaverse-4D를 넘어 Objaverse-XL, ShapeNet-Animated 등 더 많은 합성 데이터셋 활용, 혹은 비디오 생성 모델(Sora 등)로 pseudo 4D 데이터 생성.

#### 방향 5: 다중 객체 및 장면 수준 확장

현재 단일 객체 중심 → 장면 수준 4D 재구성으로 확장 시 픽셀스플랫(PixelSplat), MVSplat 등 장면 기반 Gaussian 모델과의 통합 필요.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 연구 흐름 분류

```
4D 재구성/생성 연구 흐름
├── 최적화 기반 (Optimization-based)
│   ├── D-NeRF (CVPR 2021) - 동적 NeRF
│   ├── HexPlane (CVPR 2023) - 동적 장면 표현
│   ├── Consistent4D (2023) - 단안 비디오→4D
│   ├── STAG4D (2024) - 공간-시간 앵커 Gaussian
│   └── GaussianFlow (2024) - 광학흐름 기반 4D Gaussian
├── Score Distillation 기반 (SDS-based)
│   ├── Text-to-4D (2023) - 텍스트→4D
│   ├── 4D-Fy (2023) - 하이브리드 SDS
│   ├── Align Your Gaussians (2023) - 텍스트→4D Gaussian
│   └── DreamGaussian4D (2023) - 이미지→4D Gaussian
├── 2단계 파이프라인 (Two-stage)
│   ├── Efficient4D (2024) - 멀티뷰 샘플링+최적화
│   └── Diffusion² (2024) - 직교 확산 모델 합성
└── 피드포워드 (Feed-forward) ← L4GM이 여기에 해당
    ├── LRM (2023) - 단일 이미지→3D NeRF
    ├── LGM (2024) - 멀티뷰→3D Gaussian
    └── L4GM (2024) - 단안 비디오→4D Gaussian
```

### 4.2 상세 비교표

| 논문 | 연도 | 입력 | 표현 | 방법 | 속도 | 일반화 | LPIPS |
|------|------|------|------|------|------|--------|-------|
| D-NeRF | 2021 | 멀티뷰 비디오 | Dynamic NeRF | 최적화 | 매우 느림 | 낮음 | 0.51 |
| HexPlane | 2023 | 멀티뷰 비디오 | 6-plane | 최적화 | 느림 | 낮음 | 0.38 |
| Consistent4D | 2023 | 단안 비디오 | NeRF | SDS | 2 hr | 중간 | 0.16 |
| DG4D | 2023 | 단안 비디오 | 4D Gaussian | SDS | 10 min | 중간 | 0.16 |
| STAG4D | 2024 | 단안 비디오 | 4D Gaussian | SDS+앵커 | 1 hr | 중간 | 0.13 |
| GaussianFlow | 2024 | 단안 비디오 | 4D Gaussian | SDS+광학흐름 | - | 중간 | 0.14 |
| Efficient4D | 2024 | 단안 비디오 | 4D Gaussian | 2단계 | 6 min | 중간 | 0.14 |
| **L4GM** | **2024** | **단안 비디오** | **4D Gaussian** | **Feed-forward** | **3s** | **높음** | **0.12** |

### 4.3 패러다임 전환 관점에서의 위치

**L4GM은 4D 재구성에서 "최적화 패러다임 → 피드포워드 패러다임"의 전환점**을 표지하는 연구이다.

3D 재구성에서 LRM/LGM이 보여준 것처럼, 대규모 데이터+대용량 모델+피드포워드 추론의 조합이 4D 영역에서도 최적화 기반을 압도할 수 있음을 입증하였다.

---

## 5. 향후 연구에 미치는 영향과 고려할 점

### 5.1 앞으로의 연구에 미치는 영향

#### 영향 1: 피드포워드 4D 재구성의 표준화

L4GM은 최적화 기반 4D 방법의 실용적 대안으로 **피드포워드 4D 재구성 패러다임**을 확립하였다. 이후 연구들은 더 빠르고 정확한 피드포워드 4D 모델 개발을 목표로 할 것이다.

#### 영향 2: 합성 4D 데이터셋의 중요성 부각

Objaverse-4D와 같은 대규모 고품질 합성 4D 데이터셋 구축이 이 분야 발전의 핵심 병목임을 명확히 하였다. **더 다양하고 현실적인 합성 4D 데이터 생성** 연구가 활성화될 것이다.

#### 영향 3: 3D→4D 점진적 학습 전략의 확산

"3D 사전학습 + 4D 파인튜닝" 전략은 향후 4D 생성 모델들의 표준 학습 레시피가 될 가능성이 높다. 비디오 생성 분야의 "이미지 사전학습 → 비디오 파인튜닝" 전략과 정확히 평행한다.

#### 영향 4: 4D 콘텐츠 제작 파이프라인의 민주화

3초 수준의 추론 속도는 게임, 영화, VR/AR 산업에서 4D 에셋 생성을 실용화할 수 있는 임계점을 넘는다. 텍스트→비디오→4D, 이미지→비디오→4D와 같은 파이프라인이 현실적으로 가능해진다.

#### 영향 5: 더 큰 스케일 모델로의 확장 유인

현재 LGM 기반의 모델 크기는 ViT-B 수준. 더 큰 기반 모델(LGM-XL, GRM 등)로 확장 시 추가 성능 향상이 기대된다.

### 5.2 앞으로 연구 시 고려할 점

#### 고려점 1: 실제 데이터 레이블링의 어려움

**문제**: 4D Ground Truth(시간에 따른 다시점 비디오)는 구축 비용이 극도로 높음

**해결 방향**:
- 비디오 생성 모델(Sora, Emu 등)로 합성 비디오 생성 후 pseudo GT 활용
- 능동 학습(active learning)으로 고가치 실제 데이터 선별 레이블링
- 자기-지도(self-supervised) 4D 학습 프레임워크 개발

#### 고려점 2: 시간적 일관성의 정량적 평가 메트릭

**문제**: 기존 LPIPS, CLIP, FVD는 개별 프레임 품질이나 비디오 품질을 측정하지만, **4D 재구성의 시공간 일관성**을 종합적으로 평가하는 메트릭이 부재

**고려 방향**: 새로운 4D 특화 평가 메트릭 설계 (예: 3D 일관성 + 시간 일관성 + 모션 충실도를 종합하는 지표)

#### 고려점 3: 모션 모호성(Motion Ambiguity) 처리

**문제**: 단안 비디오에서 뒷면 다리 동작 등 정면에서 보이지 않는 부분의 자연스러운 모션 생성 어려움

**고려 방향**:
- 물리 기반 제약(physics-informed constraints) 추가
- 인체 골격 모델(SMPL 등) 또는 동물 골격 사전 정보 통합
- 생성 모델로부터의 소프트 가이던스 활용

#### 고려점 4: 다중 객체·장면 수준으로의 확장

**문제**: 현재 단일 전경 객체로 제한, 배경/다중 객체 처리 불가

**고려 방향**:
- 장면 분해(scene decomposition) 모듈 통합
- 배경 처리를 위한 환경 맵 또는 NeRF 하이브리드 접근
- 객체 간 상호작용 모델링

#### 고려점 5: 카메라 가정의 완화

**문제**: 현재 정적 카메라($0°$ elevation) 가정이 실제 비디오 적용 범위를 크게 제한

**고려 방향**:
- 카메라 포즈 추정 모델(COLMAP, DUSt3R 등)과의 통합
- 카메라 움직임과 객체 움직임을 동시에 분리하는 학습 목표 설계
- 에고센트릭, 드론, 핸드헬드 카메라 시나리오 지원

#### 고려점 6: 편집 가능성과 사용자 제어

**문제**: 현재 모델은 입력 비디오를 충실히 재구성하는 것이 목표이나, 전문 사용자를 위한 **4D 편집 인터페이스**가 없음

**고려 방향**:
- 재구성된 Gaussian 시퀀스의 사후(post-hoc) 편집 툴
- 텍스트 지시 기반 모션 편집 (예: "더 빠르게 걷게 하기")
- 부분적 재구성/수정 기능

#### 고려점 7: 모델 압축 및 온디바이스 추론

**문제**: 현재 16G VRAM 요구 → 모바일/엣지 디바이스 배포 불가

**고려 방향**:
- Gaussian 수 최적화(dynamic pruning)
- 지식 증류(knowledge distillation)로 경량 4D 모델 개발
- 단계적 해상도 처리(coarse-to-fine)

---

## 종합 요약

L4GM은 4D 재구성 분야에서 **피드포워드 대용량 모델 패러다임**을 개척한 선구적 연구로, 다음 세 가지 측면에서 특히 중요하다:

1. **실용성**: 3초 이내 추론으로 실제 산업 응용 가능성을 열었다
2. **일반화**: 합성 데이터만으로 훈련했음에도 실제 비디오에 강력하게 일반화된다
3. **확장성**: 대규모 사전학습 + 간단한 아키텍처 수정이라는 스케일러블한 설계 원칙을 제시한다

동시에 모션 모호성, 다중 객체 처리, 카메라 가정 등의 한계는 향후 연구의 풍부한 방향을 제시하며, 4D 콘텐츠 생성 연구의 중요한 이정표로 기능할 것이다.
