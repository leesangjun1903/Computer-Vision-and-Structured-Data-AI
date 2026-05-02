# LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

LVSM(Large View Synthesis Model)은 기존 Novel View Synthesis(NVS) 방법들이 의존해온 **3D 귀납적 편향(3D Inductive Bias)**—NeRF, 3DGS 같은 3D 표현, Epipolar Projection, Plane-Sweep Volume 등의 구조적 설계—을 최소화하고, **순수 데이터 기반(fully data-driven)** 접근으로 NVS를 해결할 수 있다는 것을 보인다.

### 주요 기여

| 기여 항목 | 내용 |
|---|---|
| **두 가지 아키텍처 제안** | Encoder-Decoder LVSM + Decoder-Only LVSM |
| **3D 귀납적 편향 제거** | NeRF/3DGS 표현식, Epipolar line, Plane-sweep 구조 없이 학습 |
| **SOTA 성능 달성** | GS-LRM 대비 1.5~3.5 dB PSNR 향상 |
| **Zero-shot 일반화** | 훈련 시 4-view로 학습하고, 테스트 시 1~16+ view에서도 동작 |
| **컴퓨팅 효율성** | 1~2개의 A100 GPU로도 기존 64-GPU 모델 능가 |
| **완전 Transformer 기반** | Bidirectional self-attention만 사용, 특수한 attention mask 없음 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

기존 NVS 방법들은 다음과 같은 두 가지 수준의 3D 귀납적 편향을 가진다:

1. **표현 수준 편향**: NeRF(연속 체적 필드), 3DGS(가우시안 프리미티브) 등 특정 3D 표현 구조 의존
2. **아키텍처 수준 편향**: Epipolar projection, Plane-sweep cost volume, 특정 렌더링 방정식 등

이러한 편향들은 모델의 **유연성**을 제한하고, 사전에 정의된 구조에 맞지 않는 복잡한 시나리오(복잡한 재질, 세밀한 구조, 투명체 등)에서 **일반화를 저해**한다.

---

### 2.2 제안하는 방법 (수식 포함)

#### 입력 표현

$N$개의 Sparse 입력 이미지와 카메라 포즈/내부 파라미터가 주어진다:

$$\{(\mathbf{I}_i, \mathbf{E}_i, \mathbf{K}_i) \mid i = 1, \ldots, N\}$$

각 입력 이미지는 $\mathbb{R}^{H \times W \times 3}$ 크기를 가진다.

#### Plücker Ray Embedding

카메라 포즈와 내부 파라미터로부터 픽셀별 Plücker ray embedding을 계산한다:

$$\mathbf{P}_i \in \mathbb{R}^{H \times W \times 6}, \quad i = 1, \ldots, N$$

#### 입력 토큰화 (ViT 스타일 Patchify)

이미지 패치 $\mathbf{I}\_{ij} \in \mathbb{R}^{p \times p \times 3}$와 Plücker ray 패치 $\mathbf{P}_{ij} \in \mathbb{R}^{p \times p \times 6}$를 연결하여 1D 벡터로 변환한 뒤 선형 레이어로 토큰 생성:

$$\mathbf{x}_{ij} = \text{Linear}_{input}([\mathbf{I}_{ij}, \mathbf{P}_{ij}]) \in \mathbb{R}^{d} $$

여기서 $d$는 잠재 차원, $[\cdot, \cdot]$은 연결(concatenation)을 의미하며, 패치 크기 $p=8$, $d=768$을 사용한다.

#### 타겟 뷰 토큰화

목표 뷰의 카메라 포즈로부터 Plücker ray를 계산하여 쿼리 토큰 생성:

$$\mathbf{q}_j = \text{Linear}_{target}(\mathbf{P}^t_j) \in \mathbb{R}^{d} $$

입력 토큰 시퀀스 길이: $l_x = NHW/p^2$, 타겟 쿼리 토큰 길이: $l_q = HW/p^2$

#### Transformer 모델을 통한 Novel View 합성

$$y_1, \ldots, y_{l_q} = M(q_1, \ldots, q_{l_q} \mid x_1, \ldots, x_{l_x}) $$

#### 출력 픽셀 회귀

$$\hat{\mathbf{I}}^t_j = \text{Sigmoid}(\text{Linear}_{out}(y_j)) \in \mathbb{R}^{3p^2} $$

#### 손실 함수

$$\mathcal{L} = \text{MSE}(\hat{\mathbf{I}}^t, \mathbf{I}^t) + \lambda \cdot \text{Perceptual}(\hat{\mathbf{I}}^t, \mathbf{I}^t) $$

$\lambda$는 Perceptual loss(Johnson et al., 2016) 가중치로, Object-level: 1.0, Scene-level: 0.5 사용

---

### 2.3 모델 구조

#### (1) Encoder-Decoder LVSM

학습 가능한 잠재 토큰 $\{e_k \in \mathbb{R}^d \mid k=1, \ldots, l\}$ (3072개)을 사용:

$$x'_1, \ldots, x'_{l_x}, z_1, \ldots, z_l = \text{Transformer}_{Enc}(x_1, \ldots, x_{l_x}, e_1, \ldots, e_l) $$

$$z'_1, \ldots, z'_l, y_1, \ldots, y_{l_q} = \text{Transformer}_{Dec}(z_1, \ldots, z_l, q_1, \ldots, q_{l_q}) $$

- 인코더 12개 레이어 + 디코더 12개 레이어 (총 24 레이어)
- 중간 잠재 표현(latent scene representation) 생성 후 디코딩
- **장점**: 고정 길이 잠재 토큰으로 렌더링 속도 일정 (input view 수에 무관)
- **단점**: 압축에 의한 정보 손실로 상한선 존재

#### (2) Decoder-Only LVSM

중간 표현 없이 단일 스트림 Transformer로 직접 매핑:

$$x'_1, \ldots, x'_{l_x}, y_1, \ldots, y_{l_q} = \text{Transformer}_{Dec\text{-}only}(x_1, \ldots, x_{l_x}, q_1, \ldots, q_{l_q}) $$

- 24개 full self-attention 레이어
- 모든 입력/출력 토큰이 서로 양방향 self-attention
- **장점**: 더 높은 품질, 확장성, zero-shot 일반화
- **단점**: 입력 뷰 증가 시 시퀀스 길이가 선형 증가 → 연산량 이차적 증가

#### 핵심 설계 선택: Bidirectional Self-Attention

- QK-Norm (Henry et al., 2020) 적용으로 학습 안정화
- FlashAttention-v2 (Dao, 2023), Gradient Checkpointing, BFloat16 혼합 정밀도 학습

---

### 2.4 성능 향상

#### 객체 수준 (Object-Level) 비교

| 방법 | ABO PSNR↑ | GSO PSNR↑ |
|---|---|---|
| Triplane-LRM (Li et al., 2023) | 27.50 | 26.54 |
| GS-LRM (Zhang et al., 2024) | 29.09 | 30.52 |
| **Ours Encoder-Decoder** | 29.81 | 29.32 |
| **Ours Decoder-Only** | **32.10** | **32.36** |

(모두 Res-512 기준)

#### 장면 수준 (Scene-Level, RealEstate10K) 비교

| 방법 | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|
| pixelSplat (Charatan et al., 2024) | 26.09 | 0.863 | 0.136 |
| MVSplat (Chen et al., 2024) | 26.39 | 0.869 | 0.128 |
| GS-LRM (Zhang et al., 2024) | 28.10 | 0.892 | 0.114 |
| **Ours Encoder-Decoder** | 28.58 | 0.893 | 0.114 |
| **Ours Decoder-Only** | **29.67** | **0.906** | **0.098** |

#### 소규모 컴퓨팅에서의 성능

- **1 GPU (LVSM-small, 6 레이어)**: 27.66 dB PSNR → MVSplat 대비 +1.3 dB
- **2 GPU (12 레이어)**: 28.56 dB PSNR → GS-LRM(64 GPU) 능가

---

### 2.5 한계

1. **미관측 영역 처리 불가**: 결정론적 모델 특성상 보이지 않는 영역을 생성(hallucination)하지 못하며, 노이즈/깜빡임 아티팩트 발생
2. **훈련 해상도/종횡비 의존성**: 훈련 시 사용한 종횡비($512 \times 512$)와 다른 입력($512 \times 960$)에서 경계 영역 품질 저하
3. **Decoder-Only의 이차적 연산 복잡도**: 입력 뷰 수 증가 시 $O(N^2)$ 복잡도로 렌더링 속도 감소
4. **Encoder-Decoder의 정보 손실**: 고정 길이 잠재 토큰으로의 압축 과정에서 정보 손실 발생

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 Zero-Shot 일반화 (입력 뷰 수 변화)

LVSM의 가장 주목할 만한 일반화 특성은 **훈련 시 사용하지 않은 입력 뷰 수에 대한 zero-shot 일반화**이다.

- **훈련 조건**: Object-level에서 4개 입력 뷰, Scene-level에서 2개 입력 뷰만 사용
- **테스트 조건**: 1개~16개 이상의 뷰에서 zero-shot 적용

실험 결과(Figure 5):
- **Decoder-Only LVSM**: 입력 뷰 수가 증가할수록 PSNR이 지속적으로 향상 → 확장성 입증
- **Encoder-Decoder LVSM**: 8개 이상의 뷰에서 성능 저하 발생 (고정 길이 잠재 표현의 한계)
- **GS-LRM**: Encoder-Decoder와 유사한 패턴으로 8개 이상에서 성능 저하

이는 3D 귀납적 편향 최소화가 **테스트 시 입력 조건 변화에 대한 강건성**을 높인다는 것을 입증한다.

### 3.2 단일 입력 뷰 일반화

모델을 멀티뷰로 학습했음에도 불구하고, 단일 입력 이미지에서도 의미 있는 3D 추론(깊이 이해 등)이 가능하며, 4-view 기준 일부 베이스라인보다 높은 성능을 보인다.

### 3.3 일반화를 높이는 요인 분석

**① 3D 귀납적 편향 제거의 효과**

3D 귀납적 편향이 없으므로 모델이 특정 3D 표현 구조(예: NeRF의 체적 밀도, 가우시안의 형태 가정)에 제약받지 않아:
- **투명/반사 재질** 처리 개선 (ABO 데이터셋에서의 큰 성능 향상이 이를 입증)
- **얇은 구조물, 복잡한 기하학** 처리 개선
- **고주파 텍스처 디테일** 보존 향상

**② Bidirectional Full Self-Attention의 역할**

Ablation study(Table 3)에 따르면:
- **Pure cross-attention decoder** (SRT 방식): -3.47 dB PSNR 하락
- **Per-patch prediction** (패치 간 독립 예측): -1.80 dB PSNR 하락
- **Latent 업데이트 비활성화**: -1.46 dB PSNR 하락

이는 잠재 토큰과 출력 패치 토큰 간의 **전방향 정보 교환**이 일반화에 핵심임을 보여준다.

**③ 스케일링에 따른 일반화 향상 (Decoder-Only)**

모델 크기 Ablation(Table 2):

| Decoder-Only 레이어 수 | GSO PSNR↑ | RE10K PSNR↑ |
|---|---|---|
| 6 레이어 (43M) | 24.15 | 27.62 |
| 12 레이어 (86M) | 26.11 | 28.61 |
| 18 레이어 (128M) | 26.81 | 28.77 |
| 24 레이어 (171M) | 27.04 | 28.89 |

레이어 수가 증가할수록 성능이 단조 증가하여 **Scaling Law가 성립**함을 확인. 이는 향후 더 큰 모델로의 확장이 일반화 성능을 지속적으로 향상시킬 수 있음을 시사한다.

**④ 학습 데이터 다양성**

Objaverse(730K 객체)와 RealEstate10K(80K 영상)와 같은 대규모 다양한 데이터로 학습하여 다양한 장면/객체 유형에 대한 일반화 프라이어를 학습한다.

**⑤ Encoder-Decoder의 생성 모델과의 통합 가능성**

1D 잠재 토큰 공간은 생성 모델(Generative Model)과의 통합에 적합하며, 이를 통해 미관측 영역에 대한 일반화를 생성적으로 처리하는 방향이 가능하다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 방법론적 계보 비교

```
3D Inductive Bias 정도 (높음 → 낮음)
NeRF/3DGS ── LRM 계열 ── LVSM
(표현+아키텍처) (표현 편향만) (최소화)
```

### 4.2 주요 방법별 상세 비교

#### NeRF 기반 방법

| 방법 | 연도 | 특징 | LVSM 대비 한계 |
|---|---|---|---|
| NeRF (Mildenhall et al.) | 2020 | 연속 체적 필드 + 볼륨 렌더링 | Per-scene 최적화 필요, 일반화 불가 |
| Mip-NeRF (Barron et al.) | 2021 | 멀티스케일 안티앨리어싱 | 동일 |
| Instant-NGP (Müller et al.) | 2022 | 해시 인코딩 기반 빠른 NeRF | 동일 |
| PixelNeRF (Yu et al.) | 2021 | 일반화 가능 NeRF | Epipolar 편향, PSNR 20.43 vs 29.67 |

#### 3DGS 기반 방법

| 방법 | 연도 | 특징 | LVSM 대비 한계 |
|---|---|---|---|
| 3D Gaussian Splatting (Kerbl et al.) | 2023 | 가우시안 프리미티브 + 스플래팅 | Per-scene 최적화 |
| pixelSplat (Charatan et al.) | 2024 | 이미지 쌍에서 3DGS 예측 | Epipolar 편향, PSNR 26.09 vs 29.67 |
| MVSplat (Chen et al.) | 2024 | 멀티뷰 3DGS | 3DGS 표현 편향, PSNR 26.39 vs 29.67 |
| GS-LRM (Zhang et al.) | 2024 | 대형 Transformer + 3DGS | 3DGS 표현 편향, PSNR 28.10 vs 29.67 |
| LGM (Tang et al.) | 2024 | 멀티뷰 가우시안 | 고정 포즈 가정, PSNR 21.44 vs 32.36 |

#### LRM 계열 방법

| 방법 | 연도 | 특징 | LVSM 대비 한계 |
|---|---|---|---|
| LRM (Hong et al.) | 2024 | Triplane NeRF + 대형 Transformer | Triplane 표현 편향 |
| Instant3D (Li et al.) | 2023 | Text-to-3D + LRM | Triplane NeRF 표현 편향 |
| MeshLRM (Wei et al.) | 2024 | 메시 표현 + LRM | 메시 표현 편향 |

#### 3D 편향 최소화 선행 연구

| 방법 | 연도 | 특징 | LVSM와의 차이 |
|---|---|---|---|
| SRT (Sajjadi et al.) | 2022 | Transformer + 1D 잠재 표현 | CNN 토크나이저, Cross-attention 디코더로 성능 열위 |
| Geometry-free NVS (Rombach et al.) | 2021 | 3D 없는 뷰 합성 | 모델 용량/확장성 부족 |
| ViewFormer (Kulhánek et al.) | 2022 | NeRF-free Transformer | 확장성 부족, 고주파 디테일 불량 |
| Light Field Networks (Sitzmann et al.) | 2021 | 단일 평가 렌더링 | 확장성 부족 |

#### 생성 모델 기반 방법 (결정론적 LVSM과 근본적 차이)

| 방법 | 연도 | 특징 | LVSM 대비 차이 |
|---|---|---|---|
| Zero-1-to-3 (Liu et al.) | 2023 | Diffusion 기반 NVS | 확률론적, 멀티뷰 비일관성 |
| CAT3D (Gao et al.) | 2024 | 멀티뷰 Diffusion | 확률론적, 미관측 영역 생성 가능하나 일관성 보장 어려움 |
| SV3D (Voleti et al.) | 2025 | 비디오 Diffusion 기반 3D | 확률론적 |

### 4.3 핵심 비교 차트

```
                    아키텍처 편향  표현 편향  Zero-shot  확장성  속도
PixelNeRF                 높음      높음       제한적    낮음   느림
GS-LRM                    낮음      높음(3DGS)   제한적   중간   빠름
SRT                       중간      없음        중간     낮음   중간
LVSM Enc-Dec              최소      없음(학습)   강함     중간   빠름
LVSM Dec-Only             최소      없음        최강     높음   느려짐
```

---

## 5. 해당 논문이 앞으로의 연구에 미치는 영향과 고려할 점

### 5.1 연구에 미치는 영향

#### (1) 3D 귀납적 편향 패러다임 전환

LVSM은 NVS 분야에서 **"3D 구조를 명시적으로 설계해야 한다"**는 오랜 패러다임에 도전한다. 대규모 데이터와 Transformer 스케일링만으로도 NeRF, 3DGS 등의 정교한 물리 기반 렌더링 시스템을 능가할 수 있음을 증명함으로써, 향후 연구의 방향성을 재정의할 가능성이 높다.

#### (2) Scaling Law의 NVS 적용 가능성 확인

언어 모델의 Scaling Law가 NVS에도 적용됨을 실험적으로 입증했다. 이는 향후 더 큰 데이터셋과 더 큰 모델로의 확장 연구를 촉진시킬 것이다.

#### (3) 통합 멀티태스크 학습 방향 제시

입력 토큰과 타겟 포즈 토큰만으로 모든 정보를 처리하는 단순한 구조는 NVS를 **Sequence-to-Sequence 변환** 문제로 재정의하며, 이 프레임워크가 깊이 추정, 3D 재구성, 장면 이해 등 다양한 비전 태스크와 통합되는 **범용 비전 모델**로 발전할 가능성을 시사한다.

#### (4) 생성 모델과의 결합 연구 촉진

Encoder-Decoder의 1D 잠재 공간은 VAE, 확산 모델 등 생성 모델과 결합하여 **3D 콘텐츠 생성** 분야에 적용될 수 있다. 이는 미관측 영역 생성(hallucination)이라는 현재의 한계를 극복하는 방향으로 연구를 이끌 것이다.

#### (5) 소규모 연구환경에서의 접근성 향상

1~2개의 GPU로도 대규모 방법들을 능가함으로써, 아카데믹 환경에서의 3D 비전 연구의 **접근 장벽을 낮추는 데** 기여한다.

---

### 5.2 앞으로 연구 시 고려할 점

#### (1) 미관측 영역 처리 및 불확실성 모델링

현재 LVSM은 결정론적 모델로, 보이지 않는 영역에서 **노이즈/깜빡임 아티팩트** 발생 문제를 안고 있다. 향후 연구에서는:
- LVSM의 1D 잠재 공간에 **확산 모델(Diffusion Model)**이나 **Flow Matching**을 적용하여 불확실한 영역의 합리적인 생성 가능성 탐구
- 결정론적 모델과 생성 모델의 **하이브리드 프레임워크** 설계

#### (2) 연산 효율성 개선

Decoder-Only 모델은 입력 뷰 수 $N$ 증가 시 시퀀스 길이가 $O(N)$으로 늘어 Self-Attention의 $O(N^2)$ 복잡도 문제가 발생한다. 연구 방향:
- **Linear Attention** 또는 **State Space Model(SSM, Mamba)** 기반 대체
- **계층적 처리**: 먼저 뷰 내 정보 압축 후 뷰 간 정보 교환
- **동적 토큰 수 조절**: 중요도 기반 토큰 프루닝

#### (3) 다양한 해상도 및 종횡비에 대한 일반화

현재 훈련 종횡비 이외의 입력에서 성능 저하가 발생한다. 고려할 사항:
- **Resolution-agnostic 학습 전략**: RoPE(Rotary Position Embedding)의 확장, 다양한 종횡비 데이터 증강
- **Dynamic Plücker Ray 정규화**: 다양한 카메라 내부 파라미터에 대한 강건성 확보

#### (4) 카메라 포즈 추정과의 통합

현재 LVSM은 **알려진 카메라 포즈**를 전제로 한다. 포즈 추정이 불필요한 방향으로의 확장:
- Pose-free 학습 프레임워크 (PF-LRM, LEAP 등의 접근법 통합)
- 포즈 추정과 NVS를 공동 학습하는 End-to-End 모델

#### (5) 동적 장면 및 시간적 일관성

현재는 정적 장면에 집중되어 있다. 향후:
- 비디오 입력을 통한 **동적 장면** NVS (시간 축 추가)
- 물리 기반 시뮬레이션과의 결합

#### (6) 더 다양하고 대규모의 학습 데이터

현재 Object-level: Objaverse, Scene-level: RealEstate10K를 사용하나:
- **인터넷 규모 비디오 데이터** 활용 (자동 SfM 파이프라인과 결합)
- **합성+실제 데이터** 혼합 학습으로 도메인 갭 해소
- **의료 영상, 위성 영상** 등 특수 도메인으로의 확장 가능성

#### (7) 해석 가능성(Interpretability) 연구

Decoder-Only 모델이 내부적으로 어떤 3D 표현을 암묵적으로 학습하는지, Self-Attention 맵 분석을 통해 **모델의 내부 표현을 이해**하는 연구가 필요하다. 이는 모델의 신뢰성 향상과 디버깅에 도움이 된다.

---

## 참고 자료

**주요 논문 (본 리뷰 대상)**
- Jin, H., Jiang, H., Tan, H., et al. "LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias." *arXiv:2410.17242v2*, 2025. https://arxiv.org/abs/2410.17242

**비교 대상 방법들**
- Mildenhall, B., et al. "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis." *arXiv:2003.08934*, 2020.
- Kerbl, B., et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering." *ACM ToG*, 2023.
- Zhang, K., et al. "GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting." *arXiv:2404.19702*, 2024.
- Charatan, D., et al. "pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction." *CVPR*, 2024.
- Chen, Y., et al. "MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images." *arXiv:2403.14627*, 2024.
- Sajjadi, M. S. M., et al. "Scene Representation Transformer: Geometry-Free Novel View Synthesis through Set-Latent Scene Representations." *CVPR*, 2022.
- Hong, Y., et al. "LRM: Large Reconstruction Model for Single Image to 3D." *arXiv:2311.04400*, 2024.
- Li, J., et al. "Instant3D: Fast Text-to-3D with Sparse-View Generation and Large Reconstruction Model." *arXiv:2311.06214*, 2023.
- Tang, J., et al. "LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation." *arXiv:2402.05054*, 2024.
- Yu, A., et al. "pixelNeRF: Neural Radiance Fields from One or Few Images." *arXiv:2012.02190*, 2021.
- Sitzmann, V., et al. "Light Field Networks: Neural Scene Representations with Single-Evaluation Rendering." *NeurIPS*, 2021.
- Kulhánek, J., et al. "ViewFormer: NeRF-Free Neural Rendering from Few Images Using Transformers." *ECCV*, 2022.
- Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *arXiv:2010.11929*, 2020.
- Vaswani, A., et al. "Attention Is All You Need." *NeurIPS*, 2017.
- Henry, A., et al. "Query-Key Normalization for Transformers." *arXiv:2010.04245*, 2020.
- Dao, T. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." *arXiv:2307.08691*, 2023.
- Johnson, J., et al. "Perceptual Losses for Real-Time Style Transfer and Super-Resolution." *ECCV*, 2016.
- Zhou, T., et al. "Stereo Magnification: Learning View Synthesis Using Multiplane Images." *SIGGRAPH*, 2018. (RealEstate10K 데이터셋)
- Collins, J., et al. "ABO: Dataset and Benchmarks for Real-World 3D Object Understanding." *CVPR*, 2022.
- Downs, L., et al. "Google Scanned Objects: A High-Quality Dataset of 3D Scanned Household Items." *arXiv:2204.11918*, 2022.
- Deitke, M., et al. "Objaverse: A Universe of Annotated 3D Objects." *CVPR*, 2023.
- Wei, X., et al. "MeshLRM: Large Reconstruction Model for High-Quality Mesh." *arXiv:2404.12385*, 2024.
- Rombach, R., et al. "Geometry-Free View Synthesis: Transformers and No 3D Priors." *ICCV*, 2021.
- Suhail, M., et al. "Generalizable Patch-Based Neural Rendering." *ECCV*, 2022.
- Barron, J. T., et al. "Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields." *ICCV*, 2021.
- Müller, T., et al. "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding." *ACM ToG*, 2022.
