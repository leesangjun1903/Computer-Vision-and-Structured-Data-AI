# Implicit Gaussian Splatting with Efficient Multi-Level Tri-Plane Representation

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

**IGS(Implicit Gaussian Splatting)**는 3D Gaussian Splatting(3DGS)의 대용량 저장 문제를 해결하기 위해, **명시적(explicit) 포인트 클라우드**와 **암묵적(implicit) 특징 임베딩**을 결합한 하이브리드 표현 방식을 제안합니다. 핵심 주장은 다음과 같습니다:

> "멀티레벨 트리-플레인 구조를 통해 Gaussian 프리미티브 간의 공간적 상관관계를 명시적으로 모델링함으로써, 저장 효율성과 렌더링 품질을 동시에 달성할 수 있다."

### 주요 기여 (4가지)

| 기여 | 내용 |
|------|------|
| ① 멀티레벨 트리-플레인 아키텍처 | 다해상도 2D 특징 그리드로 Gaussian 속성을 암묵적으로 인코딩 |
| ② 레벨 기반 점진적 학습 스킴 | 부트스트래핑 + 순차적 레벨 활성화로 안정적 최적화 |
| ③ 공간 정규화 | TV 손실 + 희소성 손실로 공간 상관관계 강화 |
| ④ 맞춤형 압축 파이프라인 | 포인트 클라우드(PNG+Morton 정렬) + 특징 플레인(HEIC) 압축 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

3DGS는 수백 MB에 달하는 명시적 Gaussian 속성 데이터를 저장해야 하며, 다음의 근본적 문제를 가집니다:

1. **높은 데이터 엔트로피**: Gaussian 프리미티브가 위치와 속성에서 독립적이고 불규칙하여 공간적 상관관계가 없음
2. **압축의 어려움**: 상관관계 부재로 인해 효율적인 압축이 어려움
3. **품질-저장 트레이드오프**: 기존 압축 방법들은 렌더링 품질 손실을 초래

### 2.2 제안하는 방법 및 수식

#### 2.2.1 3DGS 기반 (Preliminaries)

각 Gaussian 커널은 다음과 같이 정의됩니다:

$$G(\mathbf{x}) = \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})\right) \tag{1}$$

공분산 행렬의 분해:

$$\boldsymbol{\Sigma} = \mathbf{R}\mathbf{S}\mathbf{S}^\top\mathbf{R}^\top$$

여기서 $\mathbf{S} \in \mathbb{R}^{3\times3}$는 대각 스케일링 행렬, $\mathbf{R} \in \mathbb{R}^{3\times3}$는 회전 행렬입니다.

#### 2.2.2 트리-플레인 특징 추출

3D 포인트 $\mathbf{p}$에 대한 특징 임베딩:

$$\mathbf{f}_\mathbf{p} = \|_{i \in \{xy, xz, yz\}} \psi(\mathbf{F}_i, \pi_i(\mathbf{p})) \tag{2}$$

- $\pi_i$: 투영 함수 (각 축 평면으로 투영)
- $\psi$: 바이리니어 보간 함수
- $\|$: 연결(concatenation) 연산자
- $\mathbf{F}_i \in \mathbb{R}^{w \times h \times m}$: $m$채널의 특징 플레인

#### 2.2.3 멀티레벨 트리-플레인에서의 Gaussian 속성 예측

쿼리된 3D 포인트 $\mathbf{p}$의 Gaussian 속성 계산:

$$\alpha, \mathbf{s}, \mathbf{q}, \mathbf{h} = \sum_{l=1}^{3} \Phi^l(\mathbf{f}^l_\mathbf{p}) \tag{3}$$

- $\alpha$: opacity (불투명도)
- $\mathbf{s}$: scaling matrix $\mathbf{S}$의 대각 원소
- $\mathbf{q}$: 회전행렬 $\mathbf{R}$의 쿼터니언 표현
- $\mathbf{h}$: Spherical Harmonics(SH) 계수
- $\Phi^l$: $l$번째 레벨의 MLP 디코더
- $\mathbf{f}^l_\mathbf{p}$: $l$번째 레벨에서의 특징 벡터

#### 2.2.4 공간 정규화 손실

**Total Variation(TV) 정규화**:

$$\mathcal{L}_{\text{spatial}} = \frac{1}{|\mathcal{P}|} \sum_{\mathbf{F} \in \mathcal{F}} \sum_{\mathbf{p} \in \mathcal{P}} \left[\|\Delta_u(\mathbf{F}, \mathbf{p})\|_1 + \|\Delta_v(\mathbf{F}, \mathbf{p})\|_1\right] \tag{4}$$

- $\Delta_u(\mathbf{F}, \mathbf{p})$: 픽셀 $\mathbf{p} := (u,v)$와 $(u+1, v)$의 특징 벡터 차이
- $\Delta_v(\mathbf{F}, \mathbf{p})$: 픽셀 $\mathbf{p} := (u,v)$와 $(u, v+1)$의 특징 벡터 차이

**희소성(Sparsity) 손실**:

$$\mathcal{L}_{\text{sparsity}} = \sum_{\mathbf{F} \in \mathcal{F}} \|\mathbf{F}\|_1 \tag{5}$$

**최종 학습 손실**:

$$\mathcal{L} = \mathcal{L}_{\text{render}} + \sum_{l=1}^{3} \lambda_l \left[\mathcal{L}^l_{\text{sparsity}} + \lambda_t \mathcal{L}^l_{\text{spatial}}\right] \tag{6}$$

- $\lambda_l$: 각 레벨별 가중치
- $\lambda_t$: 공간 정규화 항 가중치
- $\mathcal{L}_{\text{render}}$: 원본 3DGS의 렌더링 손실 (D-SSIM + L1)

### 2.3 모델 구조

```
IGS 전체 구조
├── 포인트 클라우드 (명시적 위치 정보)
│   └── 3D 좌표 (x, y, z)
├── 멀티레벨 트리-플레인 (암묵적 속성)
│   ├── Level 1: 저해상도 트리-플레인 → 거친(coarse) 속성 예측
│   ├── Level 2: 중해상도 트리-플레인 → 잔차(residual) 예측
│   └── Level 3: 고해상도 (2500×2500) 트리-플레인 → 세밀한 잔차 예측
│       각 레벨: 5채널, 레벨 간 해상도 비율 = 1:2
└── MLP 디코더 (레벨별 독립)
    └── 3층 FC 네트워크, 은닉층 크기=168, ReLU 활성화
        출력: opacity(1) + scaling(3) + rotation(4) + SH coeffs
```

#### 학습 진행 과정

```
Iteration:    0        16,000    20,000         35,000    50,000
              |---------|---------|---------------|---------|
              부트스트래핑   Level2   Level3 활성화        수렴
              (PC 최적화)  활성화
```

#### 압축 파이프라인

**포인트 클라우드 압축:**
1. 3D 좌표 정규화 (bounding box 기준)
2. **Morton 정렬**로 3D→2D 공간 매핑 (공간 지역성 보존)
3. 단일 채널 2D 이미지로 변환 후 **PNG 포맷**(무손실) 압축

**특징 플레인 압축:**
1. 모든 채널의 2D 맵을 단일 채널 이미지로 통합
2. **양자화 적응(Quantization Adaptation)**: 학습 중 균일 노이즈 $[-Q, Q]$ 추가
3. **HEIC(High Efficiency Image Coding)** 손실 압축
   - 하위 레벨: 높은 품질 파라미터 (중요 정보 보존)
   - 상위 레벨: 낮은 품질 파라미터 (희소 정보 강압축)

### 2.4 성능 향상

#### 정량적 결과 (Table 1, 2 기반)

**Synthetic-NeRF 데이터셋 (단일 객체):**

| 방법 | PSNR↑ | SSIM↑ | LPIPS↓ | Size(MB)↓ |
|------|-------|-------|--------|-----------|
| 3DGS | 33.80 | 0.970 | 0.031 | 68.5 |
| HAC-High | 33.73 | 0.968 | 0.033 | 3.16 |
| **Ours-High** | **34.18** | **0.975** | **0.032** | 2.72 |
| **Ours-Low** | 33.36 | 0.971 | 0.036 | **1.85** |

**DeepBlending 데이터셋:**

| 방법 | PSNR↑ | Size(MB)↓ |
|------|-------|-----------|
| 3DGS | 29.42 | 664 |
| HAC-High | 30.21 | 7.58 |
| **Ours-High** | **32.33** | 7.74 |

> ✅ Synthetic-NeRF에서 3DGS 대비 **25배 이상 저장 절감** + PSNR **+0.38dB** 향상

#### Ablation Study 결론

| 구성 | 효과 |
|------|------|
| Full (멀티레벨 + 정규화) | 최고 성능 + 최소 저장 |
| w/o 멀티레벨 (단일 레벨) | 동일 품질 대비 저장량 증가 |
| w/o 공간 정규화 | 렌더링 품질 저하 + 압축률 저하 |
| w/o 둘 다 | 아티팩트 발생 + 최대 저장량 |

### 2.5 한계점

1. **무한 장면(unbounded scenes)에서의 성능 저하**: Space contraction으로 인한 왜곡이 특징 플레인의 표현력을 약화시켜 HAC 대비 일부 성능 열세
2. **긴 학습 시간**: 단일 객체 장면 약 30분, 무한 장면 약 70분 (V100 GPU 기준), 3DGS 대비 학습 시간 증가
3. **고정된 장면 표현**: 동적 장면에 대한 확장성 미검증
4. **트리-플레인의 축 정렬 편향**: $xy, xz, yz$ 평면에 투영하므로 특정 방향 세부 사항 표현에 불리할 수 있음

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 IGS의 일반화 관련 강점

#### (1) 연속 공간 도메인 표현
IGS는 Gaussian 속성을 이산적 변수가 아닌 **연속 공간 함수**로 표현합니다:

$$\mathbf{f}_\mathbf{p} = \|_{i \in \{xy, xz, yz\}} \psi(\mathbf{F}_i, \pi_i(\mathbf{p}))$$

이 연속성은 학습 데이터에 없는 시점(novel view)에서도 부드러운 보간을 가능하게 합니다.

#### (2) 공간 정규화를 통한 과적합 방지

$\mathcal{L}\_{\text{spatial}}$과 $\mathcal{L}_{\text{sparsity}}$는 특징 플레인을 **평활화(smooth)**하고 **희소화(sparse)**합니다. 이는:
- 지역 내 Gaussian 속성의 일관성 향상
- 특정 학습 뷰에 과적합되는 고주파 노이즈 억제
- 압축 후에도 일반화된 특징 유지

#### (3) 멀티레벨 표현의 계층적 일반화

```
Level 1 (저해상도): 장면의 전반적 구조 → 높은 일반화
Level 2 (중해상도): 중간 세부 사항
Level 3 (고해상도): 세밀한 디테일 → 낮은 일반화
```

저해상도 레벨이 coarse한 구조를 학습하고, 고해상도 레벨이 잔차를 담당하는 구조는 Feature Pyramid Network(FPN) [Lin et al., CVPR 2017]의 설계 철학과 일치하며, 다양한 스케일에서의 일반화를 지원합니다.

#### (4) 부트스트래핑을 통한 초기화 강건성

부트스트래핑 단계(0~16,000 iteration)에서 명시적 3DGS 방식으로 포인트 클라우드를 초기화함으로써, 암묵적 표현 학습 시작 시 이미 **물리적으로 유의미한 위치 정보**를 보유합니다. 이는 수렴 안정성 및 일반화에 기여합니다.

### 3.2 일반화 성능 향상의 추가 가능성

#### (1) Cross-scene 일반화 (현재 미지원 → 잠재력 높음)

현재 IGS는 장면별(per-scene) 최적화 방식입니다. 그러나 멀티레벨 트리-플레인 구조는 다음과 같은 발전 가능성을 가집니다:

- **사전학습된 특징 인코더 통합**: InstantNGP [Müller et al., 2022]나 ENeRF 스타일의 일반화 NeRF처럼, 이미지 특징을 트리-플레인에 초기화하는 방식으로 확장 가능
- **메타러닝(Meta-learning) 적용**: MAML 등의 방식으로 빠른 새 장면 적응 가능

#### (2) 희소 뷰(Sparse-view) 일반화

공간 정규화가 이미 부드러운 특징 분포를 강제하므로, 학습 뷰 수를 줄였을 때의 일반화 성능이 기대됩니다:

$$\mathcal{L}_{\text{spatial}}^{l} \propto \sum_{\mathbf{F}} \sum_{\mathbf{p}} \|\nabla \mathbf{F}(\mathbf{p})\|_1$$

이 TV 정규화는 적은 관측에서도 공간적으로 일관된 특징 학습을 유도합니다.

#### (3) 동적 장면으로의 확장

논문의 공저자가 발표한 **TeTriRF** [Wu et al., CVPR 2024]는 트리-플레인을 시간 축으로 확장한 사례입니다. IGS의 멀티레벨 구조를 4D 시공간으로 확장하면:

$$\mathbf{f}_\mathbf{p}^t = \|_{i \in \{xy, xz, yz, xt, yt, zt\}} \psi(\mathbf{F}_i^l, \pi_i(\mathbf{p}, t))$$

형태의 동적 장면 표현이 가능합니다.

#### (4) 조건부 생성 모델과의 결합

트리-플레인 구조는 **EG3D** [Chan et al., CVPR 2022]처럼 생성 모델(GAN, Diffusion)과 결합하기 용이합니다. 이를 통해 단일 이미지나 텍스트로부터 IGS 표현을 생성하는 일반화된 3D 생성이 가능합니다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 방법론 분류별 비교

#### 카테고리 1: 순수 명시적 방법 (Pure Explicit)

| 논문 | 연도 | 저장 | PSNR | 특징 |
|------|------|------|------|------|
| **3DGS** [Kerbl et al., ACM TOG 2023] | 2023 | ~700MB | 27.49 | 기준선, 실시간 렌더링 |
| **Scaffold-GS** [Lu et al., CVPR 2024] | 2024 | ~250MB | 27.50 | 앵커 기반 구조화 |
| **OctreeGS** [Ren et al., arXiv 2024] | 2024 | - | - | LOD 렌더링 |

#### 카테고리 2: 벡터 양자화 기반 압축

| 논문 | 연도 | 저장 | PSNR | 특징 |
|------|------|------|------|------|
| **Compact3D** [Navaneet et al., arXiv 2023] | 2023 | 50.3MB | 27.16 | VQ + PNG 압축 |
| **LightGaussian** [Fan et al., arXiv 2023] | 2023 | 44.5MB | 27.00 | 프루닝 + VQ |
| **EAGLES** [Girish et al., arXiv 2023] | 2023 | 68.9MB | 27.15 | 경량 인코딩 |
| **Niedermayr et al.** [CVPR 2024] | 2024 | 28.8MB | 26.98 | 가속 합성 |

#### 카테고리 3: 앵커 기반 + 암묵적 표현

| 논문 | 연도 | 저장 | PSNR | 특징 |
|------|------|------|------|------|
| **HAC** [Chen et al., arXiv 2024] | 2024 | 15~25MB | 27.53~27.84 | 해시그리드 컨텍스트 |
| **CompGS** [Liu et al., arXiv 2024] | 2024 | 11~16MB | 26.79~27.26 | 레이트-왜곡 최적화 |
| **IGS (Ours)** | 2024 | **12.5~25MB** | **27.33~27.62** | 트리-플레인 + 공간 정규화 |

#### 카테고리 4: NeRF 기반 압축 (비교 참고)

| 논문 | 연도 | 특징 |
|------|------|------|
| **TensoRF** [Chen et al., ECCV 2022] | 2022 | CP/VM 분해로 컴팩트 NeRF |
| **K-Planes** [Fridovich-Keil et al., CVPR 2023] | 2023 | 트리-플레인의 직접 선행 연구 |
| **InstantNGP** [Müller et al., ACM TOG 2022] | 2022 | 멀티해상도 해시 인코딩 |
| **VideoRF** [Wang et al., CVPR 2024] | 2024 | 2D 비디오 코덱 활용 |
| **TeTriRF** [Wu et al., CVPR 2024] | 2024 | 시간적 트리-플레인 NeRF |

### 4.2 IGS의 포지셔닝 분석

```
저장 효율성 (낮을수록 좋음)
    │
    │  CompGS-Low ●
    │              HAC-Low ●
    │                       IGS-Low ●
    │                                IGS-High ●  HAC-High ●
    │
    └─────────────────────────────────────────────────────▶
                                               렌더링 품질 (높을수록 좋음)
```

**IGS의 차별점:**
1. **명시적 공간 정규화**: HAC, CompGS가 암묵적으로 상관관계를 모델링하는 반면, IGS는 TV 손실로 명시적 정규화 수행
2. **트리-플레인의 연속성**: 이산적 코드북(VQ 기반) 대비 연속적 공간 보간 가능
3. **레벨별 압축 파라미터**: 엔트로피 특성을 고려한 적응적 압축 (하위 레벨 고품질, 상위 레벨 저품질)

---

## 5. 미래 연구에 미치는 영향 및 고려 사항

### 5.1 미래 연구에 미치는 영향

#### (1) 하이브리드 표현의 패러다임 정착

IGS는 "명시적 위치 + 암묵적 속성"의 분리 원칙을 명확히 보여줍니다. 이는 앞으로 3DGS 기반 연구에서 **속성 표현의 암묵화**가 표준 접근법으로 자리 잡을 가능성을 시사합니다.

#### (2) 트리-플레인의 3DGS 적용 가능성 입증

트리-플레인은 원래 NeRF (TensoRF, K-Planes) 맥락에서 발전했으나, IGS는 이를 **비정형 포인트 클라우드 기반 렌더링**에 성공적으로 통합하였습니다. 이 결과는 격자(grid) 기반 특징과 포인트 기반 렌더링의 결합 연구를 촉진합니다.

#### (3) 공간 정규화의 중요성 재확인

Ablation study에서 공간 정규화가 **렌더링 품질과 압축률을 동시에 향상**시킴을 보여주었습니다. 이는 Gaussian 속성의 지역 일관성 유지가 품질 향상의 핵심임을 실증하며, 향후 연구에서 정규화 설계에 더 많은 관심을 기울여야 함을 시사합니다.

#### (4) 압축 파이프라인의 재활용성

Morton 정렬 + PNG + HEIC의 조합은 기존 소프트웨어/하드웨어 인프라를 활용하여 극도로 실용적입니다. 이 접근법은 다른 3D 표현 압축 연구에서도 직접 활용 가능합니다.

### 5.2 앞으로 연구 시 고려할 점

#### 기술적 개선 방향

**(A) 무한 장면에서의 공간 왜곡 문제 해결**

Space contraction으로 인한 표현력 손실을 극복하기 위해:
- **구형(spherical) 트리-플레인**: 먼 배경을 위한 구면 좌표계 기반 특징 플레인
- **적응적 해상도 할당**: 카메라 가까운 영역에 더 많은 해상도 할당

$$\mathbf{f}_\mathbf{p}^{\text{far}} = \|_{i \in \{\theta\phi, \theta r, \phi r\}} \psi(\mathbf{F}_i^{\text{spherical}}, \pi_i^{\text{sph}}(\mathbf{p}))$$

**(B) 학습 속도 개선**

- **Progressive Point Culling**: 현재 매 이터레이션마다 frustum culling 수행 → 더 정교한 조기 종료 전략 필요
- **분산 학습(Distributed Training)**: 대형 장면에서 트리-플레인 분할 병렬 처리
- **증류(Distillation)**: 빠른 초기화를 위한 사전학습 모델 활용

**(C) 적응적 멀티레벨 구조**

현재 레벨 수(3개)와 해상도 비율(1:2)이 고정되어 있습니다:
- **장면 복잡도 기반 동적 레벨 결정**
- **비등방성(anisotropic) 해상도**: 장면의 방향별 복잡도에 따른 차별적 해상도

**(D) 동적 장면으로의 확장**

$$\mathcal{L}^{\text{temporal}} = \sum_{t} \|\mathbf{F}^t - \mathbf{F}^{t-1}\|_1 \cdot \lambda_{\text{temporal}}$$

시간적 일관성 손실을 추가하여 동적 장면에서의 시간적 트리-플레인 학습 가능.

#### 연구 방향성 고려 사항

**(E) 일반화 가능한 IGS (Generalizable IGS)**

현재 장면별 최적화 → 다음을 고려해야 함:
1. **인코더-디코더 구조 도입**: 이미지를 입력받아 트리-플레인 특징을 직접 예측하는 인코더 학습
2. **Few-shot 적응**: 2~3장의 이미지로 새 장면을 IGS 표현으로 빠르게 적응

$$\mathbf{F}^{\text{new}} = \text{Encoder}(I_1, I_2, \ldots, I_k; \theta_{\text{enc}})$$

**(F) 학습 데이터 다양성 확보**

일반화 성능 향상을 위해서는:
- 실내/실외/객체/도시 등 다양한 도메인 데이터 필요
- 다양한 조명 조건 포함 (IGS의 SH 표현 한계 존재)
- 합성 데이터와 실제 데이터 간 도메인 갭 해소

**(G) 평가 지표 다양화**

현재 PSNR/SSIM/LPIPS 중심 평가에서:
- **렌더링 속도** (FPS): 압축 후 속성 디코딩 시간 포함 측정
- **메모리 사용량**: 추론 시 RAM/VRAM 소비량
- **견고성(Robustness)**: 포인트 클라우드 품질 저하 시 복원력

---

## 참고 자료 및 출처

**주요 참고 논문 (논문 내 인용 기준):**

1. **Wu, M., & Tuytelaars, T.** (2024). *Implicit Gaussian Splatting with Efficient Multi-Level Tri-Plane Representation*. arXiv:2408.10041v1. ← **본 분석 대상 논문**

2. **Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G.** (2023). *3D Gaussian Splatting for Real-Time Radiance Field Rendering*. ACM Transactions on Graphics, 42(4).

3. **Mildenhall, B., et al.** (2021). *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*. Communications of the ACM, 65(1):99–106.

4. **Chen, Y., Wu, Q., Cai, J., Harandi, M., & Lin, W.** (2024). *HAC: Hash-Grid Assisted Context for 3D Gaussian Splatting Compression*. arXiv:2403.14530.

5. **Liu, X., et al.** (2024). *CompGS: Efficient 3D Scene Representation via Compressed Gaussian Splatting*. arXiv:2404.09458.

6. **Lu, T., et al.** (2024). *Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering*. CVPR 2024.

7. **Fridovich-Keil, S., et al.** (2023). *K-Planes: Explicit Radiance Fields in Space, Time, and Appearance*. CVPR 2023.

8. **Lin, T.-Y., et al.** (2017). *Feature Pyramid Networks for Object Detection*. CVPR 2017.

9. **Fan, Z., et al.** (2023). *LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS*. arXiv:2311.17245.

10. **Wu, M., Wang, Z., Kouros, G., & Tuytelaars, T.** (2024). *TeTriRF: Temporal Tri-Plane Radiance Fields for Efficient Free-Viewpoint Video*. CVPR 2024.

11. **Müller, T., Evans, A., Schied, C., & Keller, A.** (2022). *Instant Neural Graphics Primitives with a Multiresolution Hash Encoding*. ACM Trans. Graph., 41(4).

12. **Chen, A., et al.** (2022). *TensoRF: Tensorial Radiance Fields*. ECCV 2022.

13. **Barron, J.T., et al.** (2022). *Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields*. CVPR 2022.

14. **Lee, J.C., et al.** (2024). *Compact 3D Gaussian Representation for Radiance Field*. CVPR 2024.

15. **Morton, G.M.** (1966). *A Computer Oriented Geodetic Data Base and a New Technique in File Sequencing*.

> ⚠️ **정확도 고지**: 본 답변은 제공된 PDF 원문(arXiv:2408.10041v1)을 직접 분석한 결과입니다. 최신 연구 비교에서 IGS 이후 발표된 논문들(2024년 하반기 이후)에 대한 비교는 제공된 문서의 범위를 벗어나므로 포함하지 않았습니다. 일반화 성능 관련 미래 연구 방향은 논문의 내용을 바탕으로 한 분석적 제안입니다.
