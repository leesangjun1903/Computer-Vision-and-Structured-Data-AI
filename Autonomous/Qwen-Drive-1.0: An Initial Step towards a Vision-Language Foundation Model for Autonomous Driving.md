# Qwen-Drive-1.0: An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving

> **⚠️ 주의사항**: 이 논문은 arXiv:2609.00111v1로, 출판 날짜가 2026년 8월 31일로 표기되어 있습니다. 현재(2025년 기준) 미래 날짜의 preprint이므로, 일부 비교 대상 연구들도 동일하게 미래 날짜를 가집니다. 본 분석은 제공된 PDF 원문에만 근거하며, 외부 인터넷 검색 결과는 포함하지 않습니다.

---

## 1. Executive Summary (10문장 이내)

Qwen-Drive-1.0은 사전 학습된 VLM(Vision-Language Model) 아키텍처를 변경하지 않으면서, 자율주행을 위한 3D 인식·시각 질의응답(VQA)·모션 플래닝을 단일 프레임워크에 통합한 최초의 비전-언어 파운데이션 모델이다.  
기존 VLA 방식은 텍스트 VQA 감독만으로는 3D 공간 정보를 직접 제약하지 못하고, 광범위한 도메인 적응 과정에서 사전 학습 지식의 **Catastrophic Forgetting(치명적 망각)**이 발생하는 두 가지 한계를 가진다.

> 💡 **Catastrophic Forgetting(치명적 망각)**: 신경망이 새로운 태스크를 학습할 때 기존에 학습한 지식을 급격히 잊어버리는 현상.

이를 해결하기 위해 외부 BEV(Bird's-Eye-View) 인식 헤드를 추가하여 3D 객체 검출·의미론적 점유 예측·BEV 맵 분할을 동시에 수행하고, Planning Expert가 flow matching을 통해 미래 자아 궤적을 생성한다.  
4단계 학습 레시피는 인식·VQA·플래닝 목표를 순차적으로 도입하며, 일반 VL 데이터를 혼합하여 도메인 적응과 망각 억제를 동시에 달성한다.  
nuScenes에서 43.95 mAP, NAVSIM에서 PDMS 90.7, WOD-E2E 테스트 분할에서 RFS 7.91을 달성하며 경쟁력 있는 성능을 입증하였다.  
또한 일반 VL 벤치마크에서 기반 모델인 Qwen3.5-4B와 비교해 1점 이내의 성능 유지에 성공, 코크핏-주행 통합 플랫폼에서 단일 모델 배포 가능성을 제시한다.

### 1-1. 연구의 목적과 필요성

| 문제 | 설명 | 출처 |
|------|------|------|
| VQA 감독의 3D 공간 한계 | 텍스트 감독만으로는 3D 레이아웃·깊이·점유를 직접 제약 불가 | p.2, §Introduction |
| Catastrophic Forgetting | 도메인 적응 시 OOD 상황 대처에 필요한 사전 지식 손실 | p.2, §Introduction |
| 코크핏-주행 통합 배포 요구 | 단일 컴퓨팅 플랫폼에서 대화·인식·플래닝 모두 처리 필요 | p.2, §Introduction |
| 이종 데이터 통합 부재 | 데이터셋별 레이블 체계 불일치로 공동 학습 어려움 | p.8, §2.3 |

---

## 2. 핵심 주장과 근거 표

| 핵심 주장 | 제안 방법 | 정량적 근거 | 위치 |
|-----------|-----------|------------|------|
| VLM 아키텍처 불변 유지로 일반 VL 능력 보존 | 외부 모듈(BEV Head, Planning Expert) 부착 | General VQA Avg. 66.41 (Qwen3.5-4B: 67.40, 차이 <1점) | Tab.3, p.17 |
| 외부 BEV 헤드로 명시적 3D 인식 추가 | 깊이 기반 뷰 변환 + 쿼리 기반 BEV 트랜스포머 | nuScenes 43.95 mAP, BEVFormerV2* 대비 +2.01 mAP | Tab.1, p.13 |
| Flow Matching 기반 플래닝 전문가 | 32층 Diffusion Transformer, 10-step Euler 적분 | NAVSIM PDMS 90.7 (RL 후), WOD-E2E RFS 7.91 | Tab.6, Tab.4, p.18-20 |
| CoC 추론 감독으로 인과 이해 향상 | Chain-of-Causation 데이터 자동 구축 및 감사 | PAI-AV-CoC Overall Acc. 41.26 (2위 Gemma4-12B: 4.01) | Tab.2, p.15 |
| 이종 데이터 통합 학습 레시피 | 레이블 통합 + 공간 정렬 + 일관성 필터링 | OpenScene NDS 44.16 (nuScenes-only head: 16.50) | Tab.1, p.12-13 |
| RL로 태스크 수준 보상 최적화 | 그룹 상대적 어드밴티지 + 저주파 궤적 탐색 | off-road rate 24%→12% (AlpaSim), PDMS +2.5점 | Tab.7, Tab.6, p.20 |

---

## 2-1. 상세 방법론 설명

### 해결하고자 하는 문제

1. **3D 공간 감독 부재**: 텍스트 VQA만으로는 깊이·점유·3D 레이아웃을 명시적으로 학습 불가 (p.2)
2. **Catastrophic Forgetting**: 주행 도메인 적응 시 OOD 일반화를 위한 사전 지식 손실 (p.2)
3. **이종 데이터셋 통합**: nuScenes/OpenScene의 레이블 체계·좌표계·해상도 불일치 (p.8-9)
4. **모방 학습의 한계**: 기록된 단일 궤적 재현은 다양한 안전 행동을 패널티화 (p.6-7)

---

### 제안하는 방법 (수식 포함)

#### (1) 자기회귀 텍스트 생성 목표 (Eq. 1, p.4)

$$\mathcal{L}_{\text{ntp}} = -\sum_{t=1}^{T} \log p(y_t \mid \mathbf{x}, y_{<t})$$

- $\mathbf{x}$: 직렬화된 멀티모달 입력 (이미지 토큰 + 텍스트 프롬프트)
- $y_t$: $t$번째 응답 토큰
- $y_{<t}$: $t$ 이전의 모든 응답 토큰
- 주행 및 일반 VL 샘플 모두 동일한 목표 함수 사용

> 💡 **자기회귀(Autoregressive)**: 이전 출력을 조건으로 다음 출력을 순차 생성하는 방식.

---

#### (2) BEV 복셀 특징 계산 (Eq. 2, p.4)

$$\mathbf{V}(\mathbf{p}) = \sum_{i \in \Omega(\mathbf{p})} \mathbf{D}_i(u_i, v_i, d_i) \, \mathbf{F}^v_i(u_i, v_i)$$

- $\mathbf{V}(\mathbf{p})$: 3D 공간 위치 $\mathbf{p}$에서의 복셀 특징
- $\mathbf{D}_i(u_i, v_i, d_i)$: 카메라 $i$에서 픽셀 $(u_i, v_i)$의 깊이 빈 $d_i$에 대한 예측 확률 (범주형 분포)
- $\mathbf{F}^v_i(u_i, v_i)$: 비전 인코더의 2D 특징 (VLM 입력 전의 저수준 외관 특징)
- $\Omega(\mathbf{p})$: 3D 점 $\mathbf{p}$가 유효하게 투영되는 카메라 뷰 집합
- $\mathbf{P}_i$: 카메라 $i$의 캘리브레이션 행렬 (카메라→이미지 좌표 변환)

> 💡 **BEV (Bird's-Eye-View, 조감도)**: 차량 위에서 내려다보는 시점으로 3D 장면을 2D 평면에 표현.
> 💡 **깊이 기반 뷰 변환 (Depth-based View Transform)**: 2D 이미지 특징을 깊이 예측을 통해 3D 공간으로 투영하는 기법 (Lift-Splat-Shoot 방식).

---

#### (3) 인식 손실 함수 (Eq. 3-5, p.4-5)

$$\mathcal{L}_{\text{perc}} = \mathcal{L}_{\text{det}} + \mathcal{L}_{\text{occ}} + \mathcal{L}_{\text{map}}$$

**3D 검출 손실 (Eq. 4)**:

```math
\mathcal{L}_{\text{det}} = \sum_{l=1}^{L} \left( 2\mathcal{L}^{(l)}_{\text{focal}} + 0.75\mathcal{L}^{(l)}_{\ell_1} \right)
```

- $L$: 디코더 레이어 수 (deep supervision)
- $\mathcal{L}^{(l)}_{\text{focal}}$: $l$번째 레이어의 focal loss (클래스 불균형 처리)
- $\mathcal{L}^{(l)}_{\ell_1}$: $l$번째 레이어의 $\ell_1$ 회귀 손실 (박스 좌표 회귀)
- Hungarian 알고리즘으로 예측-GT 매칭

**점유 예측 손실 (Eq. 5)**:

```math
\mathcal{L}_{\text{occ}} = 100\mathcal{L}_{\text{focal}} + \mathcal{L}_{\text{geo}} + \mathcal{L}_{\text{sem}} + \mathcal{L}_{\text{lov}}
```

- $\mathcal{L}_{\text{focal}}$: 클래스 균형 focal loss
- $\mathcal{L}\_{\text{geo}}, \mathcal{L}_{\text{sem}}$: MonoScene의 기하학적·의미론적 장면 유사도 손실
- $\mathcal{L}_{\text{lov}}$: Lovász-softmax loss (IoU 최적화를 위한 미분 가능 대리 손실)
- 맵 분할: $\mathcal{L}\_{\text{map}} = 100\mathcal{L}\_{\text{focal}} + \mathcal{L}_{\text{lov}}$

> 💡 **Focal Loss**: 쉬운 샘플의 기여를 줄이고 어려운 샘플에 집중하는 손실 함수.
> 💡 **Lovász-softmax**: IoU(교집합/합집합 비율)를 직접 최적화하기 위한 미분 가능한 대리 손실.

---

#### (4) 궤적 예측 조건부 생성 (Eq. 6, p.5)

$$\boldsymbol{\tau} \sim p(\boldsymbol{\tau} \mid \mathbf{s}, \ell, \boldsymbol{\tau}_{\text{hist}}, \mathbf{n}, \mathbf{e}, \mathbf{r}), \quad \boldsymbol{\tau} = \{(x_k, y_k, \theta_k)\}_{k=1}^{50}$$

- $\mathbf{s}$: 차량 센서 입력 (카메라 이미지)
- $\ell$: 직렬화된 카메라 레이아웃
- $\boldsymbol{\tau}_{\text{hist}}$: 과거 자아 궤적 (역사적 위치/방향)
- $\mathbf{n}$: 내비게이션 지시 (직진/좌회전/우회전 등)
- $\mathbf{e}$: 현재 자아 상태 (속도, 가속도 등)
- $\mathbf{r}$: 텍스트 플래닝 추론 (없으면 $\varnothing$)
- $x_k, y_k$: 웨이포인트 $k$에서의 종/횡 위치 (현재 자아 프레임 기준)
- $\theta_k$: 웨이포인트 $k$에서의 방향각 (현재 방향 대비)
- 스케일 정규화: $x_k$는 165m, $y_k$는 25m, $\theta_k$는 $\pi/2$ rad으로 나눔

---

#### (5) Flow Matching 학습 (Eq. 7-8, p.5)

선형 보간 경로:
$$\boldsymbol{\tau}_t = (1-t)\boldsymbol{\tau}_0 + t\boldsymbol{\tau}_1$$

전체 플래닝 손실:

```math
\mathcal{L}_{\text{plan}} = \mathcal{L}_{\text{fm}} + 2\times10^{-4}\mathcal{L}_{\Delta_1} + 2\times10^{-5}\mathcal{L}_{\Delta_2}
```

- $\boldsymbol{\tau}_1$: 정규화된 정답 궤적 (clean trajectory)
- $\boldsymbol{\tau}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$: 동일 형태의 가우시안 노이즈
- $t \sim \text{Beta}(1.5, 1.0)$, $t = \min\{\tilde{t}, 0.9\}$ (안정성 보장)
- $\mathcal{L}_{\text{fm}}$: 유도 흐름 속도와 목표 흐름 속도($\boldsymbol{\tau}_1 - \boldsymbol{\tau}_0$) 간의 제곱 오차
- $\mathcal{L}_{\Delta_j}$ ($j \in \{1,2\}$): $j$차 시간 차분에 대한 Huber 패널티 (궤적 지터 방지)
- 추론 시: 가우시안 노이즈 초기화 후 10-step Euler solver로 적분

> 💡 **Flow Matching**: 노이즈에서 신호로 가는 확률적 흐름을 학습하는 생성 모델 패러다임. Diffusion 모델보다 단순하고 효율적.

---

#### (6) 강화학습: 점수 보정 및 전이 (Eq. 9-14, p.7-8)

**점수 보정 (Eq. 9)**:
$$s_\theta\left(\boldsymbol{\tau}^{(k)}, t_k\right) = -\frac{\boldsymbol{\tau}^{(k)} - t_k\hat{\boldsymbol{\tau}}^{(k)}_1}{(1-t_k)^2}$$

**전이 평균 (Eq. 10)**:
$$\boldsymbol{\mu}^{(k)} = \boldsymbol{\tau}^{(k)} + v_\theta\left(\boldsymbol{\tau}^{(k)}, t_k\right)\Delta t + \frac{\sigma_k^2}{2}s_\theta\left(\boldsymbol{\tau}^{(k)}, t_k\right)$$

**저주파 탐색 (Eq. 11)**:
$$\boldsymbol{\tau}^{(k+1)} = \boldsymbol{\mu}^{(k)} + \sigma_k \boldsymbol{\Phi} \mathbf{Z}_k$$

- $\boldsymbol{\Phi} \in \mathbb{R}^{N \times M}$: $N=50$ 웨이포인트에 대한 $M=6$개 코사인 기저 벡터 (저주파 모드)
- $\mathbf{Z}_k \in \mathbb{R}^{M\times3}$: 표준 가우시안 난수 행렬
- $\sigma_k = 0.03$ ($k \in \mathcal{W} = \{7,8,9\}$, 마지막 3 스텝에만 확률적 탐색)

**로그 가능도 대리 (Eq. 12)**:
$$\log \pi_\theta\left(\boldsymbol{\tau}^{(k+1)} \mid \boldsymbol{\tau}^{(k)}\right) = -\frac{1}{2\sigma_k^2}\left\|\boldsymbol{\Phi}^\top\left(\boldsymbol{\tau}^{(k+1)} - \boldsymbol{\mu}^{(k)}\right)\right\|_F^2 + \text{const}$$

**그룹 상대 어드밴티지 (Eq. 13)**:
$$A_i = \frac{R_i - \bar{R}}{\sigma_R + \epsilon_R}, \quad \bar{R} = \frac{1}{G}\sum_{j=1}^G R_j$$

- $G=8$: 그룹 내 롤아웃 수
- $R_i$: $i$번째 롤아웃 보상
- $\sigma_R$: 그룹 내 보상의 표준편차, $\epsilon_R = 10^{-8}$ (수치 안정성)

**할인 정책 그래디언트 (Eq. 14)**:

```math
\mathcal{L}_{\text{rl}} = -\frac{1}{GW}\sum_{i=1}^G\sum_{w=0}^{W-1} \gamma^{W-1-w} A_i \log\pi_\theta\left(\boldsymbol{\tau}^{(k_w+1)}_i \mid \boldsymbol{\tau}^{(k_w)}_i\right)
```

- $W = |\mathcal{W}| = 3$: 확률적 전이 스텝 수
- $\gamma = 0.6$: 할인율 (출력에 가까운 스텝에 더 큰 크레딧)

> 💡 **그룹 상대 어드밴티지**: 학습된 가치 함수 없이 그룹 내 보상의 상대적 비교로 정책 개선 신호를 도출 (GRPO 방식, DeepSeekMath에서 유래).

---

### 모델 구조 (Fig. 2, 3)

```
입력 이미지 (단일뷰/멀티뷰/비디오/일반 이미지)
        ↓
   [Vision Encoder] (SigLIP-Qwen 기반)
   ├── F^v_i (저수준 특징, BEV Head로)
   └── 이미지 토큰 → [Qwen3.5 Language Model (4B)]
                         ↓                    ↓
              텍스트 응답 생성            K, V 캐시
              (L_ntp)              (→ Planning Expert)
        ↓
   [BEV Perception Head]          [Planning Expert]
   ├── 깊이 예측 네트워크           ├── 32층 Diffusion Transformer
   ├── 뷰 변환 (F^v → 3D 볼륨)    ├── AdaLN (시간, 내비게이션, 상태 주입)
   ├── BEV Transformer            └── Flow Matching → 50 웨이포인트
   ├── 3D 검출 (DETR 스타일)
   ├── 점유 예측 (3D UNet)
   └── 맵 분할 (UNet 헤드)
```

> 💡 **AdaLN (Adaptive Layer Normalization)**: 조건 신호(시간, 상태 등)를 레이어 정규화의 스케일·편향으로 주입하는 기법.
> 💡 **DETR (Detection Transformer)**: 헝가리안 매칭 기반의 end-to-end 객체 검출 트랜스포머.

---

### 4단계 학습 레시피 (Fig. 4, p.6)

| 단계 | 훈련 모듈 | 고정 모듈 | 목표 함수 |
|------|-----------|-----------|-----------|
| Stage 1: BEV Head Pretraining | BEV Head | Vision Encoder, VLM | $\mathcal{L}_{\text{perc}}$ |
| Stage 2: Perception & VQA Joint | BEV Head, Vision Encoder, VLM | - | $\mathcal{L}\_{\text{perc}} + \mathcal{L}_{\text{ntp}}$ |
| Stage 3: Planning Expert Pretraining | Planning Expert | Vision Encoder, VLM | $\mathcal{L}_{\text{plan}}$ |
| Stage 4: Reinforcement Learning | Planning Expert | Vision Encoder, VLM | $\mathcal{L}_{\text{rl}}$ |

---

### 성능 향상

| 영역 | 주요 결과 | 비교 기준 |
|------|-----------|-----------|
| 3D 검출 (nuScenes) | 43.95 mAP | BEVFormerV2* (SigLIP): 41.94 mAP |
| 3D 검출 (OpenScene) | 43.45 mAP | Head-only (joint): 40.57 mAP |
| 맵 분할 (nuScenes) | 60.99 mIoU | PETRv2 (SigLIP): 57.62 mIoU |
| Driving VQA 평균 | 69.43 | Qwen3.5-4B: 63.52 |
| CoC Overall Acc. | 41.26 | Gemma4-12B: 4.01 (2위) |
| NAVSIM PDMS (RL) | 90.7 | ExploreVLA: 90.4 |
| WOD-E2E RFS (test) | 7.91 | MindVLA-U1 RL: 7.87 |
| General VQA 평균 (a) | 66.41 | Qwen3.5-4B: 67.40 (차이 <1점) |

---

### 한계

| 한계 | 설명 | 위치 |
|------|------|------|
| 인과 구조 불안정 | 다른 시간 스케일의 원인들이 공존할 때 지배 원인 식별 불안정 | p.24, §5 |
| 궤적-추론 불일치 | 생성된 궤적이 텍스트 추론과 항상 일치하지 않음 | p.24-25, §5 |
| 시간 샘플링 한계 | 0.5s 간격의 희소 이력이 AlpaSim의 단기 재계획 응답성 제한 | p.20, §3.3 |
| 점유 크로스데이터셋 | OpenScene의 기계 생성 복셀 레이블이 joint adaptation 효과 제한 | p.12-13, §3.1 |
| 크로스 태스크 전이 한계 | 3개 태스크의 입력 형식·해상도·시간 맥락 불일치 | p.25, §5 |

---

## 3. 각 주장에 페이지/Figure/Table 번호 표시

| 주장 | 근거 위치 |
|------|-----------|
| VLM 아키텍처 불변 유지 | p.2 (§Introduction), p.3 (§2.1), Fig.2 |
| BEV Head가 3D 인식을 추가 | p.4 (§2.1), Fig.3(a), Tab.1 |
| Stage 2 joint adaptation의 필요성 | p.12-13 (§3.1), Tab.1, Tab.8 |
| Flow matching 기반 Planning Expert | p.5 (§2.1), Fig.3(b), Eq.7-8 |
| RL이 안전성과 선호도 개선 | p.19-21 (§3.3), Tab.6, Tab.7, Fig.10 |
| 일반 VL 능력 보존 | p.17 (§3.2.2), Tab.3 |
| 인과 추론 능력 | p.15 (§3.2.1), Tab.2, Fig.8 |
| 데이터 스케일 효과 | p.23 (§3.4), Fig.12 |
| 미지의 카메라 리그 전이 | p.23-24 (§3.4), Fig.13 |
| RL 보상 설계 분석 | p.22-23 (§3.4), Fig.11 |

---

## 4. 저자 보고 결과 vs. 나의 해석 분리

### 저자가 직접 보고한 결과

**3D 인식** (Tab.1, p.13):
- nuScenes: 43.95 mAP, 42.83 NDS, 60.99 map mIoU, 19.82 Occ mIoU
- OpenScene: 43.45 mAP, 44.16 NDS, 71.27 map mIoU, 19.84 Occ mIoU

**Driving VQA** (Tab.2, p.15):
- LingoQA 77.80, Ego3D RMSE 7.78↓, VLAD 66.52, SURDS 66.13, WaymoQA All 74.47, Avg. 69.43
- PAI-AV-CoC Overall 41.26, IH 71.00, Causal Avg. 58.30

**General VQA** (Tab.3, p.17):
- Group(a) Avg. 66.41 (Qwen3.5-4B: 67.40), Group(b) Avg. 53.96 (Qwen3.5-4B: 52.99)

**Motion Planning**:
- NAVSIM PDMS: SFT w/ reasoning 88.2, RL 90.7, RL best-of-6 91.4 (Tab.6, p.19)
- WOD-E2E test RFS: SFT w/ reasoning 7.78, RL 7.91 (Tab.4b, p.18)
- AlpaSim: at-fault close encounter 11%, off-road 12%, at-fault score 0.37 (Tab.7, p.20)

### 나의 해석 및 추가 분석

1. **CoC 성능 격차의 해석**: PAI-AV-CoC에서 2위(Gemma4-12B: 4.01)와 약 10배 차이는 인상적이지만, 이 벤치마크가 저자가 직접 구축한 내부 벤치마크임을 감안할 때, 훈련 데이터와의 분포 유사성이 과도하게 반영되었을 가능성을 배제하기 어렵다.

2. **WOD-E2E 검증 분할 RFS 8.45의 해석**: 검증 분할 RFS 8.45가 인간 운전자(8.13)를 초과하지만, 저자 스스로 "이 RFS 어노테이션이 RL 보상으로 사용되므로 훈련 시나리오에 대한 선호도 정렬 최적화"라고 명시 (p.19)—즉 in-sample 결과로 과도한 해석은 금물.

3. **AlpaSim 점수 해석의 복잡성**: DriveWAM의 at-fault 점수 0.53이 높아 보이지만 progress 35%로 거의 정지 상태—알파심 점수는 전진/안전 트레이드오프를 함께 봐야 함. 저자도 이를 명시 (p.20-21).

4. **크로스데이터셋 점유 예측의 한계**: OpenScene 점유 mIoU 19.84는 nuScenes only head (11.50)보다 높지만, BEVFormerV2* (OpenScene에서 평가 불가)와 직접 비교 불가.

5. **데이터 스케일 포화 미도달**: Fig.12에서 1.38M에서도 ADE/FDE 감소세 지속—더 많은 데이터로의 성능 향상이 예상되나, 실제 포화 지점은 미확인.

---

## 5. 통계적으로 취약한 부분과 비교 불가능한 수치

| 항목 | 취약점/비교 불가 이유 |
|------|----------------------|
| ⚠️ PAI-AV-CoC 벤치마크 | **저자가 직접 구축한 내부 벤치마크** (p.14). 훈련 데이터와 분포 유사성 불명확 |
| ⚠️ WOD-E2E val RFS 8.45 (RL) | **검증 분할이 RL 보상 소스** (p.19). in-sample 최적화로 일반화 근거 약함 |
| ⚠️ IH (In-House) 벤치마크 | 내부 중국어 주행 결정 벤치마크, 외부 검증 불가 (p.14) |
| ⚠️ LingoQA 평가 프로토콜 변경 | 공식 LingoJudge 대신 Qwen-Plus 사용 (p.14, 각주1). 공식 점수(79.4)와 불일치 |
| ⚠️ AlpaSim 비교 | DriveWAM/SimWAM은 생성적 미래 감독 포함—아키텍처 차이로 단순 비교 부적절 (p.20) |
| ⚠️ PAI-AV 644-example split | 공식 구성에 의해 공개 훈련 데이터와 중복 (p.18). 저자가 700-frame 누출 없는 부분집합 제시하나 방법론 차이 |
| ⚠️ OpenScene 점유 비교 | BEVFormerV2*이 6-cam 누스씬 리그에 고정된 카메라 임베딩 사용—8-cam 오픈씬 평가 불가 (Tab.1) |
| ⚠️ 통계적 유의성 검증 부재 | 모든 성능 비교에 신뢰구간/표준편차/p-value 없음 |
| ⚠️ AlpaSim 916 시나리오 | 표본 수가 상대적으로 적어 통계적 안정성 한계 |
| ⚠️ NAVSIM PDMS 포화 구간 | 저자 스스로 "상위 구간에서의 추가 이득이 점수 함수 적응 반영 가능" 언급 (p.20) |

---

## 6. 문서가 답하지 않는 질문

1. **LiDAR 센서 미사용의 한계**: 카메라 전용 시스템으로 악천후(폭우, 눈, 야간)에서의 강건성이 어느 정도인지 정량적 평가 없음.

2. **실시간 추론 지연 (Latency)**: 4B VLM + BEV Head + Planning Expert의 총 추론 시간 및 온보드 하드웨어 요구사항 미제시.

3. **RL 훈련 안정성**: Stage 4 RL의 훈련 불안정성(발산, 리워드 해킹) 발생 여부 및 완화 방법 상세 미기재.

4. **장기 시계열 플래닝**: 5초 이상의 장기 경로 계획 능력에 대한 평가 없음.

5. **멀티 에이전트 상호작용**: 주변 에이전트의 의도 예측을 명시적으로 모델링하는지 여부.

6. **VLM 백본 스케일링 효과**: 4B보다 큰 VLM (예: 7B, 72B)을 백본으로 사용할 때의 성능 변화.

7. **실제 차량 배포 결과**: 시뮬레이션이 아닌 실제 도로 주행 테스트 결과.

8. **CoC 추론의 인과성 검증**: Chain-of-Causation 추론이 실제로 올바른 인과 메커니즘을 학습했는지 인과적 개입(causal intervention) 실험 부재.

9. **다국어/다지역 일반화**: 중국 외 도로 환경(미국, 유럽)에서의 성능 일반화 정도.

10. **BEV Head와 VLM 표현의 상호 강화 메커니즘**: 두 모듈이 서로의 표현을 어떻게 향상시키는지 기제 분석 부재.

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.1): 성능 개요 레이더 차트

두 개의 레이더 차트가 Qwen-Drive-1.0의 다차원 성능을 시각화한다.
- **왼쪽**: 주행 VQA (LingoQA, UnGoQA 등)와 일반 VQA (HRBench, AnsWorth 등)에서 유사 크기/더 큰 모델들과 경쟁력 있는 성능 표시. Qwen3.5-4B (베이스라인)보다 주행 VQA에서 우수하면서 일반 VQA를 거의 유지.
- **오른쪽**: 3D 인식 (nuScenes/OpenScene mAP, BEV mIoU)과 모션 플래닝 (NAVSIM PDMS, WOD-E2E RFS)에서 BEVFormerV2, PETR, PETRv2, DriveWAM 등 전문화된 방법들과 비교하여 경쟁력 있음을 시각적으로 제시.
- **해석**: 단일 모델이 네 가지 이질적 능력 영역에서 동시에 경쟁력을 유지한다는 핵심 주장을 효과적으로 시각화. 그러나 레이더 차트의 축 스케일이 다르고 일부 방법은 일부 벤치마크에서만 비교되므로 직접 대비에 주의 필요.

---

### Figure 2 (p.3): 통합 아키텍처

모델의 전체 데이터 흐름을 명확히 도식화:
- **입력 다양성**: 주행 비디오, 단일뷰, 멀티뷰, 일반 이미지 모두 지원—단일 아키텍처의 유연성 강조.
- **공유 경로**: Vision Encoder와 Qwen3.5 LM이 모든 입력에 공유—VLM 아키텍처 불변성의 핵심.
- **외부 모듈**: BEV Head (좌측, 인식)와 Planning Expert (우측, 플래닝)가 VLM의 K,V 캐시를 활용—VLM 수정 없이 기능 확장하는 핵심 설계 원칙.
- **해석**: 아키텍처 설계의 모듈성과 확장성이 명확히 드러남. 단, BEV Head와 Planning Expert가 VLM 표현에 어떻게 역방향 신호를 보내는지 (Stage 2에서만)의 비대칭성은 그림에서 충분히 표현되지 않음.

---

### Figure 4 (p.6): 4단계 학습 레시피

불꽃 아이콘(훈련 가능)과 눈꽃 아이콘(고정)으로 각 단계별 훈련/고정 모듈을 직관적으로 표시:
- **Stage 1→2**: BEV Head만 먼저 초기화(Stage 1), 이후 VLM 전체를 함께 적응(Stage 2)—head-only 학습의 한계를 극복하기 위한 2단계 전략.
- **Stage 3→4**: Planning Expert를 고정된 VLM 표현에서 먼저 모방 학습(Stage 3), 이후 RL로 정제(Stage 4)—궤적 학습과 VL 표현 변화를 분리하는 핵심 설계.
- **해석**: 각 단계의 목적이 명확히 분리되어 있고, Stage 2의 인식+VQA 혼합이 표현 품질 향상의 핵심임을 이해하게 함 (Tab.8 ablation과 연결). RL 단계에서만 Planning Expert가 적응되고 VLM은 고정된다는 점이 일반 능력 보존의 핵심.

---

### Figure 7 (p.13): 3D 인식 질적 결과

nuScenes와 OpenScene 검증 분할에서 4가지 장면 유형의 3D 검출, 의미론적 점유, BEV 맵 분할 결과를 나란히 비교:
- **주목할 점 (b)**: OpenScene의 기계 생성 레이블에 존재하는 road surface 위의 부유 복셀(floating voxels)을 모델 예측이 억제하고 올바른 도로 표면 의미론을 복원—훈련 레이블보다 나은 예측의 흥미로운 사례.
- **멀티뷰 시각화**: 6~8개 카메라 뷰에서의 예측과 BEV 뷰에서의 GT 비교—카메라 가시 영역 내에서 경쟁력 있는 3D 구조 이해 확인.
- **해석**: 질적 결과가 양적 지표를 보완하며, 특히 점유 예측이 GT 레이블의 아티팩트보다 더 물리적으로 합리적인 경우도 있음을 보여줌. 단, 카메라 비가시 영역의 성능은 시각화하기 어려운 한계.

---

### Figure 9 (p.21): 모션 플래닝 질적 결과

오픈루프(WOD-E2E, PAI-AV)와 클로즈드루프(AlpaSim) 플래닝 결과:
- **(a) 오픈루프**: 여우 횡단 감속, 우회전 전용 차선 준수, 정지 트럭 측방 통과 등 다양한 장면 인식 기반 계획 생성. PAI-AV 예측이 기록된 미래와 근접—추론 기반 계획의 질적 타당성 입증.
- **(b) 클로즈드루프 (AlpaSim)**: 18초에 걸쳐 교통 신호 변화(초록→빨강)에 따른 실시간 계획 업데이트, 추월 차량에 대한 측방 조정—동적 환경에서의 반응성 확인.
- **해석**: 클로즈드루프 결과가 누적 오류 하에서도 일관된 내비게이션 명령 준수를 보이며, RL이 보수적이지만 안전한 행동으로 수렴하는 경향을 시각적으로 확인. 단, 선택된 성공 사례로만 구성되어 실패 사례는 포함되지 않음.

---

## 8. 결론 및 후속 연구 방향

### 8-1. 저자 제시 시사점 및 후속 연구 계획

**저자 제시 시사점** (p.24, §4-5):
1. 사전 학습된 VLM에 외부 모듈만으로 3D 인식 및 궤적 생성을 추가하면서 일반 능력을 보존할 수 있음을 입증
2. 통합 궤적 감독 + 보상 기반 최적화의 조합이 다양한 평가 수준(오픈/의사 클로즈드/클로즈드루프)에서 효과적
3. 코크핏-주행 통합 플랫폼에서 단일 모델 배포의 실현 가능성 제시

**저자 제시 후속 연구 방향** (p.24-25, §5):
1. **다중 시간 스케일 인과 모델링**: 빨간 신호(20m 앞, 점진적 감속 필요) vs. 갑작스러운 장애물(5m, 즉각 반응)의 공존 상황 처리
2. **추론-궤적 일관성 감독**: 생성된 텍스트 추론과 실제 궤적 간의 명시적 정합성 학습
3. **크로스 태스크 전이 강화**: 3개 태스크의 입력 형식·해상도·시간 맥락 통합으로 상호 표현 강화

### 8-1. 모델 일반화 성능 향상 가능성

**현재 일반화 능력의 근거**:
- 일반 VQA에서 Qwen3.5-4B 대비 1점 이내 유지 (Tab.3, p.17)
- 훈련 데이터 미포함 중국어 내부 벤치마크 IH에서 71.00 달성 (p.15)
- 미학습 카메라 리그(WOD-E2E 8-cam, PAI-AV 6-cam)에서 질적으로 일관된 인식 출력 (Fig.13, p.23-24)
- 계획 성능이 훈련 데이터 스케일 증가에 따라 포화 없이 향상 (Fig.12)

**일반화 향상을 위한 제안 방향**:

1. **도메인 무관 표현 학습 강화**: 현재 Stage 2의 일반 VL 데이터 혼합 비율(26%)을 적응적으로 조정하고, 도메인 불변 특징을 명시적으로 학습하는 대조 학습 방식 도입.

2. **합성 데이터 활용**: 자율주행 세계 모델(예: DriveWAM, DriveDreamer)로 생성한 반사실적(counterfactual) 장면—드문 날씨, 비정상 교통 상황—을 증강 데이터로 활용하여 long-tail OOD 일반화 개선.

3. **메타 학습 기반 빠른 적응**: 새로운 카메라 리그나 지역 도로 환경에 소량의 레이블 데이터만으로 적응하는 few-shot/meta-learning 프레임워크 도입.

4. **시간적 일반화**: 현재 4-프레임(1.5초) 이력 입력을 더 긴 시간 창으로 확장하고, 가변 길이 시간 시퀀스를 처리하는 위치 인코딩 개선.

5. **지식 蒸溜 (Knowledge Distillation)**: 대형 교사 VLM의 표현을 소형 학생 모델에 전달하여, 소형 모델의 일반화 능력을 향상시키는 방향.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> **⚠️ 주의**: 아래 비교는 본 논문의 참고문헌 목록에 명시된 연구들에만 근거합니다. 논문 외부의 추가 최신 연구에 대한 직접적 검색 결과는 포함하지 않습니다.

#### 주요 계보별 비교

| 연구 계열 | 대표 연구 (참고문헌 기준) | Qwen-Drive와의 차별점 |
|-----------|--------------------------|----------------------|
| **모듈형 파이프라인** | UniAD (Hu et al., 2023, CVPR) | 계획 지향적이나 언어 이해 제한적; Qwen-Drive는 VLM 기반 통합 |
| **VLA 기반 주행** | DriveLM (Sima et al., 2024, ECCV) | VQA 그래프 구조, 3D 인식 모듈 부재 |
| **추론 강화 주행** | Alpamayo-R1/1.5 (Wang et al., 2025d) | 10B 파라미터로 규모 더 크나, 일반 VL 능력 저하됨 (Tab.3) |
| **확산 기반 플래닝** | DiffusionDrive (Liao et al., 2025, CVPR) | NAVSIM 88.1 달성하나 VQA/인식 통합 없음 |
| **세계 모델 접근** | DriveWAM (Shi et al., 2026) | 미래 생성 감독 포함이나 AlpaSim progress 35%로 과보수적 |
| **스트리밍 아키텍처** | MindVLA-U1 (Huang et al., 2026) | 스트리밍 처리에 강점이나 일반 VL 평균 미제시 |
| **코스모스 계열** | Cosmos-Reason2-32B (NVIDIA, 2025a) | 32B 파라미터에도 Driving QA Avg. 46.62로 Qwen-Drive-4B의 69.43에 못 미침 |

#### 이 논문이 앞으로의 연구에 미치는 영향

1. **VLM 불변 설계 원칙의 확산**: VLM 아키텍처를 수정하지 않고 외부 모듈로 기능을 확장하는 설계 패턴이 자율주행 외 로봇공학, 구현 AI 분야로 확산될 가능성.

2. **이종 데이터 통합 방법론**: 레이블 통합·공간 정렬·일관성 필터링의 3단계 데이터 파이프라인이 멀티소스 학습의 표준 방법론으로 자리잡을 가능성.

3. **Flow Matching의 자율주행 적용**: Diffusion 모델 대신 Flow Matching으로 궤적을 생성하는 방식이 효율성과 품질의 균형에서 우수함을 입증—후속 연구의 플래닝 모듈 설계에 영향.

4. **인과 추론(CoC) 데이터 자동 구축**: VLM 기반의 계획 추론 데이터 자동 생성 및 다단계 감사 파이프라인이 데이터 구축 비용을 낮추는 방법론으로 주목받을 것.

5. **코크핏-주행 통합 배포 패러다임**: 단일 모델이 두 도메인을 동시에 서비스하는 배포 방식이 업계 표준으로 확산될 경우, 일반 능력을 보존하면서 도메인 능력을 추가하는 연구 수요 증가.

#### 앞으로 연구 시 고려할 점

1. **평가 프로토콜 표준화**: 서로 다른 판사 모델, 서로 다른 평가 프로토콜로 인한 비교 불가능성 문제를 해결하기 위해 커뮤니티 표준 평가 프레임워크 필요.

2. **인과적 검증 강화**: CoC 추론의 질을 단순 정확도가 아닌 인과적 개입 실험(interventional experiment)으로 검증하는 방법론 개발.

3. **안전-진보 트레이드오프 정량화**: AlpaSim에서 RL이 안전성을 개선하지만 진행 속도(progress)를 희생—이 트레이드오프를 명시적으로 제어하는 다목적 최적화 방법.

4. **배포 환경의 이질성**: 실제 차량 플랫폼별로 다른 카메라 리그·캘리브레이션·센서 노이즈에 대한 강건성을 사전에 설계에 반영하는 연구 필요.

5. **데이터 효율성**: Alpamayo-1.5가 80,000시간 데이터를 사용하는 반면 Qwen-Drive는 훨씬 적은 데이터—데이터 효율적 학습(semi-supervised, active learning)이 중요한 연구 방향.

6. **시간 연속성과 재계획 빈도**: 현재 0.5s 간격 4프레임 이력이 단기(0.1s 재계획) 응답에 부족—가변 주파수 재계획에 적응하는 아키텍처 설계 고려.

---

## 참고 자료

**원문 논문**:
- Qwen Team & Huazhong University of Science and Technology. "Qwen-Drive-1.0: An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving." arXiv:2609.00111v1, 31 Aug 2026.

**논문 내 주요 인용 문헌 (분석에 활용된 것)**:
- Lipman et al. (2023). "Flow Matching for Generative Modeling." ICLR 2023.
- Li et al. (2022c). "Unifying Voxel-based Representation with Transformer for 3D Object Detection." NeurIPS 2022.
- Philion & Fidler (2020). "Lift, Splat, Shoot." ECCV 2020.
- Zhu et al. (2021). "Deformable DETR." ICLR 2021.
- Lin et al. (2017). "Focal Loss for Dense Object Detection." ICCV 2017.
- Berman et al. (2018). "The Lovász-Softmax Loss." CVPR 2018.
- Cao & De Charette (2022). "MonoScene." CVPR 2022.
- Yu et al. (2023). "FlashOcc." arXiv:2311.12058.
- Shao et al. (2024). "DeepSeekMath." arXiv:2402.03300.
- Dauner et al. (2024). "NAVSIM." NeurIPS 2024.
- Hu et al. (2023). "Planning-Oriented Autonomous Driving (UniAD)." CVPR 2023.
- Yang et al. (2023). "BEVFormerV2." CVPR 2023.
- Sima et al. (2024). "DriveLM." ECCV 2024.
- Caesar et al. (2020). "nuScenes." CVPR 2020.
- OpenScene Contributors (2023). OpenScene Dataset.
- Liao et al. (2025). "DiffusionDrive." CVPR 2025.
- Wang et al. (2025d). "Alpamayo-R1." arXiv:2511.00088.
