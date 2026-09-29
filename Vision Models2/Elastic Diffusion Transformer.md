# Elastic Diffusion Transformer

---

## 1. Executive Summary (10문장 이내)

Elastic Diffusion Transformer(E-DiT)는 대규모 Diffusion Transformer(DiT) 모델의 추론 비용을 적응적으로 줄이기 위한 가속화 프레임워크이다.  
기존 pruning(가지치기)·distillation(지식 증류) 기반 방법들은 고정된 모델 구조를 사용해 모든 입력에 동일한 연산량을 할당하는 정적(static) 설계의 한계를 가진다.  
E-DiT는 DiT 생성 과정에 **샘플 의존적 희소성(sample-dependent sparsity)**이 존재한다는 관찰에서 출발한다.  
각 Transformer 블록에 경량 **라우터(router)**를 장착하여, 블록 스킵 여부와 MLP 너비를 입력 잠재(latent) 표현에 따라 동적으로 결정한다.  
세 가지 핵심 메커니즘—적응적 블록 스킵, 적응적 MLP 너비 축소, 블록 단위 특성 캐싱—을 결합하여 추론을 가속한다.  
학습 목표는 생성 품질 손실(flow-matching loss)과 효율성 정규화 손실의 합으로 구성된다.  
Qwen-Image(20B), FLUX, Hunyuan3D-3.0에 걸쳐 실험한 결과, 품질 손실을 최소화하면서 약 **2배의 추론 속도 향상**을 달성한다.  
블록 단위 캐싱은 별도 재학습 없이(training-free) 추가적인 가속을 제공한다.  
비적응적(non-adaptive) 정적 모델 대비 생성 품질이 현저히 우수하며, 다양한 모달리티(이미지·3D)에 범용 적용이 가능하다.

---

### 1-1. 연구 목적과 필요성

**문제 배경:**
DiT 기반 대형 생성 모델(Qwen-Image, FLUX, Hunyuan3D 등)은 강력한 생성 품질을 보이지만, 수십 회의 노이즈 제거(denoising) 단계마다 거대한 Transformer 연산이 요구되어 실제 배포가 어렵다.

**기존 방법의 한계 (p.1~2):**
- Pruning/Distillation 기반 방법(TinyFusion, PPCL 등): **고정된** 소형 구조를 모든 샘플에 균일 적용 → 충분하지 않은 가속 또는 품질 저하
- 동적 방법(DyDiT, Dense2MoE 등): 백본(backbone) 변경 시 구조적 재설계 필요, 입력 복잡도 무관하게 고정 파라미터 활성화

**필요성:**
생성 과정의 희소성이 **샘플마다 다름**을 실증적으로 확인하고(Figure 2), 이를 동적으로 활용하는 범용 가속 프레임워크가 필요하다.

> 💡 **용어 설명**
> - **Pruning(가지치기)**: 모델에서 중요도가 낮은 가중치나 레이어를 제거하여 모델을 압축하는 기법
> - **Distillation(지식 증류)**: 큰 모델(교사)의 지식을 작은 모델(학생)에게 전달하여 경량화하는 기법
> - **Denoising(노이즈 제거)**: 확산 모델이 순수 노이즈에서 시작하여 점진적으로 노이즈를 제거하며 샘플을 생성하는 과정
> - **Latent(잠재 표현)**: 입력 데이터를 저차원의 압축된 특징 공간으로 변환한 표현

---

## 2. 핵심 주장과 근거 표

| # | 핵심 주장 | 근거 (Figure/Table/Page) | 결과 수치 |
|---|-----------|--------------------------|-----------|
| 1 | DiT 생성 과정은 샘플 의존적 희소성을 가짐 | Figure 2 (p.2) | 동일 블록 제거 시 샘플마다 품질 영향 상이 |
| 2 | 경량 라우터로 블록 스킵 여부를 동적 예측 가능 | Figure 3(a), Eq.(4)~(9) (p.4) | 라우터 파라미터 $H_r \ll D$로 오버헤드 최소 |
| 3 | 적응적 MLP 너비 축소로 추가 가속 가능 | Figure 3(c), Eq.(11)~(16) (p.5) | 너비 비율 $S=\{1/4, 1/2, 3/4, 1\}$ 동적 선택 |
| 4 | 블록 단위 캐싱으로 훈련 없이 추가 가속 가능 | Algorithm 1, Figure 6 (p.5~8) | E-DiT-turbo: 1800ms → 1283ms |
| 5 | E-DiT가 기존 pruning 방법 대비 우수한 품질-효율 트레이드오프 달성 | Table 1, Table 2 (p.6) | Qwen-Image: GenEval 0.893 @ 1702ms |
| 6 | 비적응적 정적 모델 대비 적응적 라우팅이 필수적 | Table 4 (p.7) | Non-adaptive: DPG 83.7 vs Ours: 87.6 |
| 7 | 전체 용량 초기화(full-capacity init)가 학습 안정성 개선 | Table 5 (p.7) | Random init: GenEval 0.801 vs Full-cap: 0.853 |
| 8 | 3D 생성에서도 2배 가속 달성 | Table 3, Figure 5 (p.7~8) | 5012ms → 2587ms, ULIP-I 유지 |
| 9 | 라우터가 블록 중요도를 자연스럽게 학습 | Figure 7 (p.8~9) | 첫 2개·마지막 3개 블록은 거의 스킵 안 됨 |

---

### 2-1. 상세 설명

#### ① 해결하고자 하는 문제

기존 DiT 가속 방법들이 **정적 구조**를 채택함으로써 발생하는 두 가지 문제:
1. 샘플 복잡도에 무관하게 동일 연산량 할당 → 쉬운 샘플에서 연산 낭비
2. 특정 백본에 종속된 구조적 변경 필요 → 범용성 부족

#### ② 제안하는 방법 (수식 포함)

**[사전 지식: Rectified Flow]** *(p.3)*

$$dx_t = v(\mathbf{x}_t, t)\,dt, \quad t \in [0, 1] $$

- $\mathbf{x}_t$: 시각 $t$에서의 잠재 변수
- $v$: 속도장(velocity field), 네트워크 $\epsilon_\theta$로 파라미터화
- 학습 목표:

$$\min_\theta \mathbb{E}_{t \sim \mathcal{U}(0,1)}\left[\|(\mathbf{x}_1 - \mathbf{x}_0) - \epsilon_\theta(\mathbf{x}_t, t)\|^2\right] $$

- $\mathbf{x}_0 \sim \pi_0$: 데이터 분포, $\mathbf{x}_1 \sim \pi_1$: 노이즈 분포

> 💡 **Rectified Flow**: 데이터와 노이즈 사이를 직선 경로로 연결하는 생성 모델 학습법. DDPM보다 더 적은 샘플링 단계로 고품질 생성 가능

**[표준 MLP 구조]** *(p.3, Eq.3)*

$$\text{MLP}(\mathbf{z}) = \sigma(\mathbf{z}\mathbf{W}_1)\mathbf{W}_2 $$

- $\mathbf{z} \in \mathbb{R}^{L \times D}$: 입력 ($L$=시퀀스 길이, $D$=특징 차원)
- $\mathbf{W}_1 \in \mathbb{R}^{D \times H}$, $\mathbf{W}_2 \in \mathbb{R}^{H \times D}$: 선형 투영 행렬
- $\sigma(\cdot)$: 비선형 활성화 함수 (예: GELU)
- $H/D = 4$: MLP 너비 비율 (대부분의 모델에서 기본값)

**[라우터 설계]** *(p.4, Eq.4~6)*

타임스텝 조건부 변조(modulation):

$$\tilde{\mathbf{x}}_t^i = (1 + \gamma(t)) \odot \text{LN}(\mathbf{x}_t^i) + \delta(t) $$

- $\mathbf{x}_t^i \in \mathbb{R}^{L \times D}$: 블록 $B^i$의 확산 단계 $t$에서의 입력 잠재
- $\gamma(t), \delta(t) \in \mathbb{R}^D$: 타임스텝 임베딩 $E(t)$의 선형 투영으로 얻은 스케일·시프트 파라미터
- $\odot$: 원소별 곱(element-wise multiplication)
- $\text{LN}(\cdot)$: Layer Normalization

비선형 활성화 후 은닉 표현 계산:

$$\mathbf{h} = \sigma(\tilde{\mathbf{x}}_t^i \mathbf{W}) \in \mathbb{R}^{L \times H_r} $$

- $\mathbf{W} \in \mathbb{R}^{D \times H_r}$, $H_r \ll D$: 라우터를 경량화하기 위한 축소 차원

두 개의 헤드(head)로 출력:

$$\ell_t^i = \frac{1}{L}\sum_{j=1}^{L}\mathbf{h}[j,:]\mathbf{W}_g, \quad \mathbf{u}_t^i = \frac{1}{L}\sum_{j=1}^{L}\mathbf{h}[j,:]\mathbf{W}_w $$

- $\mathbf{W}_g \in \mathbb{R}^{H_r \times 1}$: 블록 스킵 여부를 결정하는 게이팅 헤드 파라미터
- $\mathbf{W}_w \in \mathbb{R}^{H_r \times 4}$: MLP 너비 선택을 위한 너비 헤드 파라미터
- $\ell_t^i$: 블록 스킵 여부를 나타내는 스칼라 로짓(logit)
- $\mathbf{u}_t^i \in \mathbb{R}^4$: MLP 너비 선택을 위한 4차원 벡터

> 💡 **Layer Normalization (LN)**: 각 샘플 내의 특징 차원에 걸쳐 정규화를 수행하는 기법으로, 학습 안정성 향상에 기여
> 💡 **Logit**: 확률로 변환되기 전의 원시 점수값. Sigmoid 함수를 통해 [0,1] 범위의 확률로 변환됨

**[적응적 블록 스킵]** *(p.4, Eq.7~9)*

게이트 확률:

$$p_t^i = \sigma(\ell_t^i) \in [0, 1]$$

Straight-Through Estimator(STE)를 통한 비미분 가능 연산의 기울기 전파:

$$g_t^i = \mathbb{1}[p_t^i \geq \tau] + p_t^i - \text{StopGrad}(p_t^i) $$

- $\mathbb{1}[\cdot]$: 지시 함수 (조건이 참이면 1, 거짓이면 0)
- $\tau$: 블록 스킵 임계값 (실험에서 0.5로 설정)
- $\text{StopGrad}(\cdot)$: 해당 항의 역전파를 중단하는 연산

블록 출력:

$$\mathbf{x}_t^{i+1} = \mathbf{x}_t^i + g_t^i \cdot (B^i(\mathbf{x}_t^i) - \mathbf{x}_t^i) $$

게이팅 정규화 손실:

$$\mathcal{L}_{\text{gating}} = (\bar{p} - \rho_g)^2 $$

- $\bar{p} = \frac{1}{n}\sum_{i=1}^n p_t^i$: 전체 블록의 평균 게이트 확률
- $\rho_g \in (0,1)$: 목표 블록 활성화 비율 (E-DiT-base: 0.6, turbo: 0.5)

> 💡 **Straight-Through Estimator (STE)**: 역전파 시 비미분 가능한 연산(예: 이진 결정)을 우회하는 기법. 순전파에서는 이진 결정을 사용하고, 역전파에서는 연속 확률값의 기울기를 그대로 통과시킴

**[적응적 MLP 너비 축소]** *(p.5, Eq.11~16)*

너비 확률 계산:

$$\mathbf{q}_t^i = \text{softmax}(\mathbf{u}_t^i) \in \mathbb{R}^4 $$

최적 너비 선택:

$$k = \arg\max_j \mathbf{q}_t^i[j], \quad \hat{s}_t^i = \mathcal{S}[k] $$

- $\mathcal{S} = \{1/4, 1/2, 3/4, 1\}$: 사전 정의된 너비 축소 비율 집합
- $\hat{s}_t^i$: 선택된 MLP 너비 비율

훈련 시 마스킹으로 미분 가능성 유지:

$$\text{MLP}_{\text{adapt}}(\mathbf{z}) = \left(\sigma(\mathbf{z}\mathbf{W}_1) \odot \mathbf{m}(\hat{s}_t^i)\right)\mathbf{W}_2 $$

- $\mathbf{m}(\hat{s}_t^i) \in \{1, 0\}^H$: 은닉 차원의 앞 $\hat{s}_t^i \cdot H$ 부분만 1로 유지하는 마스크

평균 너비 축소 비율:

$$\bar{r} = \frac{\sum_{i=1}^n \mathbb{1}[p_t^i \geq \tau]\, r_t^i}{\sum_{i=1}^n \mathbb{1}[p_t^i \geq \tau]} $$

- $r_t^i = \sum_{j=1}^4 \mathbf{q}_t^i[j]\,s[j]$: 블록 $B^i$의 기대 너비 비율

너비 정규화 손실:

$$\mathcal{L}_{\text{width}} = (\bar{r} - \rho_w)^2 $$

- $\rho_w \in (0,1)$: 목표 평균 MLP 너비 비율

추론 시 명시적 행렬 슬라이싱:

$$\text{MLP}_{\hat{s}_t^i}(\mathbf{z}) = \sigma(\mathbf{z}\widetilde{\mathbf{W}}_1)\widetilde{\mathbf{W}}_2 $$

- $\widetilde{\mathbf{W}}_1 = \mathbf{W}_1[:, : H \cdot \hat{s}_t^i]$, $\widetilde{\mathbf{W}}_2 = \mathbf{W}_2[: H \cdot \hat{s}_t^i, :]$: 선택된 너비에 해당하는 부분 행렬

**[전체 학습 목표]** *(p.5, Eq.17)*

$$\mathcal{L} = \mathcal{L}\_{\text{perf}} + \lambda \mathcal{L}_{\text{eff}} $$

- $\mathcal{L}_{\text{perf}}$: Eq.(2)의 flow-matching 목적 함수 (생성 품질 보존)
- $\mathcal{L}\_{\text{eff}} = \mathcal{L}\_{\text{gating}} + \mathcal{L}_{\text{width}}$: 효율성 정규화
- $\lambda = 1$: 두 항의 균형 가중치

**[블록 단위 캐싱]** *(p.6, Eq.18~19)*

잔차 업데이트 계산 및 저장:

$$\Delta^i = B^i(\mathbf{x}_t^i, \mathbf{q}_t^i) - \mathbf{x}_t^i $$

캐시된 잔차로 업데이트:

$$\mathbf{x}_{\tilde{t}}^{i+1} = \mathbf{x}_{\tilde{t}}^i + \Delta^i $$

- $\Delta^i$: 블록 $B^i$의 잔차 업데이트 (특성 은행 $\mathcal{C}^i$에 저장)
- 조건: $p_{\tilde{t}}^i \in [\tau, \tau + \delta]$ (경계 영역) AND $\mathcal{C}^i \neq \emptyset$ AND $k^i < K$
- $\delta$: 경계 영역 마진 (E-DiT-base: 0.1)
- $K$: 최대 재사용 횟수 (E-DiT-base: 5)

> 💡 **잔차(Residual)**: 블록 출력과 입력의 차이. DiT는 잔차 연결(skip connection) 구조이므로, 이전 타임스텝의 잔차를 재사용하면 전체 블록 계산을 우회할 수 있음

#### ③ 모델 구조

```
E-DiT 전체 구조 (Figure 3, p.4)
├── 입력 잠재 x^i
│   └── 각 DiT 블록 B^i에 경량 라우터 R^i 장착
│       ├── Router R^i
│       │   ├── 타임스텝 변조 (Eq.4)
│       │   ├── 선형+활성화 (Eq.5)
│       │   ├── 게이팅 헤드 → p^i_t (블록 스킵 확률, Sigmoid)
│       │   └── 너비 헤드 → q^i_t (MLP 너비 분포, SoftMax)
│       └── E-DiT 블록 B^i
│           ├── Modulation + Attention (변경 없음)
│           └── Adaptive-width MLP (너비 동적 조정)
└── 추론 시: 블록 단위 캐싱 추가 적용
```

#### ④ 성능 향상 및 한계

**성능 향상:**
| 모델 | 지연시간 감소 | 품질 지표 변화 |
|------|--------------|----------------|
| Qwen-Image (E-DiT-base) | 2431→1702ms (~30%) | GenEval: 0.870→0.893 (↑) |
| Qwen-Image (E-DiT-turbo) | 2431→1283ms (~47%) | GenEval: 0.870→0.853 (소폭↓) |
| FLUX | 715→374ms (~48%) | GenEval: 0.665→0.671 (유지) |
| Hunyuan3D-3.0 | 5012→2587ms (~48%) | ULIP-I: 0.1446→0.1473 (↑) |

**한계:**
- 3D 생성에서는 Hunyuan3D-3.0 단일 모델만 비교 (다른 3D 모델과의 비교 없음)
- 비디오 생성 모달리티에 대한 검증 없음
- 하이퍼파라미터($\rho_g, \rho_w, \delta, K$)를 모델별로 수동 조정 필요
- 학습 데이터(BLIP3o-60K + ShareGPT-4o ≈ 100K)가 제한적

---

## 3. 각 주장의 위치 표시

| 주장 | 위치 |
|------|------|
| 샘플 의존적 희소성 존재 | p.2, Figure 2 (a)(b)(c) |
| 라우터 설계 | p.4, Figure 3(a), Eq.(4)-(6) |
| 적응적 블록 스킵 | p.4, Eq.(7)-(9) |
| 추론 시 블록 스킵 | p.5, Eq.(10) |
| 적응적 MLP 너비 축소 | p.5, Eq.(11)-(16) |
| 전체 학습 목표 | p.5, Eq.(17) |
| 블록 단위 캐싱 | p.6, Eq.(18)-(19), Algorithm 1 |
| Qwen-Image 정량 결과 | p.6, Table 1 |
| FLUX 정량 결과 | p.6, Table 2 |
| 3D 생성 결과 | p.7, Table 3 |
| 컴포넌트 ablation | p.7, Table 4 |
| 초기화 ablation | p.7, Table 5 |
| 캐싱 ablation | p.8, Figure 6 |
| 라우터 예측 시각화 | p.8~9, Figure 7 |

---

## 4. 저자 보고 결과 vs. 해석 분리

### 저자가 직접 보고한 결과

| 항목 | 저자 보고 내용 |
|------|---------------|
| Qwen-Image 속도 | E-DiT-base: 1702ms, turbo: 1283ms vs. 기본: 2431ms (Table 1) |
| FLUX 속도 | 374ms vs. 기본: 715ms (Table 2) |
| 3D 생성 속도 | 2587ms vs. 기본: 5012ms (Table 3) |
| GenEval (E-DiT-base) | **0.893** (기본 0.870보다 높음) |
| Ablation (Table 4) | 블록스킵+너비축소+캐싱 조합 시 1283ms, DPG 85.4, GenEval 0.853 |
| 초기화 비교 (Table 5) | Full-capacity init: DPG 85.4, GenEval 0.853 vs. Random: 78.6, 0.801 |
| 3D 품질 지표 | ULIP-I: 0.1473 (기본 0.1446보다 높음), Uni3D-I: 0.4332 (기본 0.4334, 유사) |

### 본 분석자의 해석

> ⚠️ 이하는 논문 내용에 기반한 해석이며, 저자의 직접 주장이 아닙니다.

1. **E-DiT-base가 기본 모델보다 GenEval에서 높은 점수를 기록**한 것은 흥미롭지만, 이는 소규모 파인튜닝(fine-tuning) 효과로 해석될 가능성이 있으며, 통계적 유의성이 불명확하다.

2. **3D 생성에서 ULIP-I가 소폭 향상**된 것 역시 측정 오차 범위 내일 수 있으며, 단순히 가속이 3D 품질을 보존한다는 수준의 해석이 더 타당하다.

3. 라우터가 "자연스럽게 블록 중요도를 학습"한다는 주장(Figure 7)은 정성적 해석이며, 학습된 라우터 결정이 실제 블록 중요도와 일치하는지에 대한 정량적 검증은 논문에서 제공되지 않는다.

4. **비교 베이스라인 중 PPCL과 DyDiT는 원 논문의 수치를 그대로 인용**하고 있어(p.6), 동일 하드웨어·조건에서 측정된 수치가 아닐 수 있다.

---

## 5. 통계적 취약점 및 비교 불가능 수치

| 항목 | 문제점 | 위험도 |
|------|--------|--------|
| PPCL, DyDiT 수치 | 원 논문에서 직접 인용, 하드웨어/설정이 다를 수 있음 (p.6) | ⚠️ 높음 |
| E-DiT-base GenEval 0.893 > 기본 0.870 | 가속 후 성능이 향상되는 것은 직관에 반하며, 100K 파인튜닝 효과와 혼재 가능 | ⚠️ 높음 |
| 3D 생성 비교 | 오직 Hunyuan3D-3.0과 E-DiT 두 가지만 비교 (p.7, Table 3) | ⚠️ 높음 |
| 3D ULIP-I 향상 (0.1446→0.1473) | 표준편차 미기재, 통계적 유의성 불명확 | ⚠️ 중간 |
| 학습 데이터 규모 | ~100K 이미지 파인튜닝 → 원 모델 수백만 대비 극히 소규모 | ⚠️ 중간 |
| 추론 H20 단일 GPU 지연시간 | 배치 크기, 해상도, 시퀀스 길이 등 측정 조건 미상세 | ⚠️ 중간 |
| Figure 1 그래프 | 점이 1~2개뿐이어서 경향성 해석 주의 필요 | ⚠️ 낮음 |

---

## 6. 문서가 답하지 않는 질문

1. **라우터 오버헤드**: 라우터 자체($H_r$, 레이어 수)의 파라미터 수와 이로 인한 지연시간 증가량은 얼마인가?

2. **비디오 생성 적용 가능성**: Wan2.1, CogVideoX 등 비디오 DiT에 E-DiT를 적용했을 때의 성능은?

3. **학습 데이터 의존성**: 100K 파인튜닝 데이터의 도메인이 다를 경우(예: 의료, 과학 이미지) 라우터 일반화 성능은 유지되는가?

4. **적응형 배치 추론**: 배치(batch) 내 서로 다른 샘플이 다른 블록을 활성화할 때, GPU 병렬성(parallelism)은 어떻게 처리되는가?

5. **$\rho_g, \rho_w$ 선택 기준**: 모델별 하이퍼파라미터 튜닝 가이드라인이 있는가, 아니면 수동 탐색인가?

6. **오차 누적(error accumulation)**: 캐싱의 최대 재사용 횟수 $K$ 제한이 충분한가? 장시간(많은 스텝) 생성 시 품질 저하는?

7. **FLUX.1-dev와 FLUX.1-schnell 구분**: 실험에서 사용한 FLUX 버전이 명확히 명시되지 않은 부분이 있음.

8. **E-DiT와 step distillation 결합**: 동시에 적용했을 때 상호작용(interaction) 효과는?

---

## 7. 가장 중요한 그림 5개 해석

### Figure 1 (p.1) — E-DiT 전체 성능 요약

**해석:** Qwen-Image와 Hunyuan3D-3.0에서 E-DiT가 경쟁 방법 대비 동일 지연시간에서 높은 품질을 달성함을 산점도로 보여준다. 파레토 프론티어(Pareto frontier) 상에서 E-DiT가 우월한 위치를 차지하고 있어, 품질-효율 트레이드오프가 개선되었음을 시각적으로 주장한다.

> ⚠️ **주의**: PPCL 등 일부 비교점은 원 논문 수치를 인용하므로, 실제 동일 조건 비교인지 불명확하다.

> 💡 **파레토 프론티어**: 하나의 지표를 희생하지 않고는 다른 지표를 개선할 수 없는 최적해의 집합. 그래프에서 좌상단(낮은 지연시간, 높은 품질)에 가까울수록 우수

---

### Figure 2 (p.2) — 샘플 의존적 희소성의 세 가지 증거

**해석:** 이 논문에서 가장 핵심적인 동기를 제공하는 그림이다.
- **(a)**: 동일한 블록 집합(예: 30,35,40,45,50번)을 제거해도 일부 이미지(풍선 이미지)는 큰 영향 없지만, 다른 이미지(텍스트 포함 이미지)는 심각하게 손상됨 → 블록 중요도는 샘플별로 다름
- **(b)**: 타임스텝 건너뛰기도 샘플별로 영향이 다름 → 타임스텝 중요도도 샘플 의존적
- **(c)**: 20B 모델이 필요한 샘플과 10B로 충분한 샘플이 존재 → 계산량 요구가 샘플 복잡도에 따라 달라짐

이 세 가지 관찰이 E-DiT의 설계 철학 전체를 정당화한다.

---

### Figure 3 (p.4) — E-DiT 전체 파이프라인

**해석:** 세 개의 서브그림이 E-DiT의 계층적 설계를 보여준다.
- **(a) 라우터**: 입력 잠재에서 Sigmoid 기반 게이팅 확률 $p^i$와 Softmax 기반 너비 분포 $q^i$를 출력하는 이중 헤드 구조
- **(b) E-DiT 전체**: 블록 스킵 여부($p^i > \tau$)에 따라 일부 블록(B^i)은 실행, 나머지는 건너뜀
- **(c) E-DiT 블록 내부**: MLP의 너비($q^i$에 따라 D, 2D, 3D, 4D 중 선택)가 동적으로 조정됨

이 구조가 기존 DiT 아키텍처에 최소한의 수정으로 적용 가능함을 보여주는 것이 핵심 기여다.

---

### Figure 6 (p.8) — 블록 단위 캐싱 Ablation

**해석:** 블록 단위 캐싱의 설계 결정을 검증한다.
- **직접 스킵(Direct Skip, 1633ms)**: 경계 블록을 단순 제거하면 품질(텍스트 렌더링)이 눈에 띄게 저하됨
- **캐시 사용(1319ms)**: 이전 타임스텝의 잔차를 재사용하면 속도와 품질의 균형이 개선됨
- **$\delta$ 증가**: 캐싱 적용 범위를 넓히면($\delta=0.03$) 속도는 1223ms로 증가하지만 품질이 저하됨
- **$K$ 증가($K=15$)**: 최대 재사용 횟수를 늘려도 품질 손실은 미미하고 속도 향상도 제한적

이는 $\delta$가 $K$보다 훨씬 중요한 하이퍼파라미터임을 보여준다.

---

### Figure 7 (p.8~9) — 라우터 예측 시각화

**해석:** x축은 블록 ID(0~60), y축은 디노이징 타임스텝(0~30)을 나타낸다. 빨간 픽셀은 블록 활성화($p \geq 0.5$), 파란 픽셀은 스킵을 나타낸다.

- **Hard Sample(복잡한 텍스트 이미지)**: 전반적으로 빨간 픽셀이 많아 대부분의 블록이 활성화됨 → 높은 연산량 필요
- **Easy Sample(단순 앵무새 이미지)**: 파란 픽셀이 많아 상당수 블록이 스킵됨 → 낮은 연산량으로 충분
- **공통 패턴**: 첫 2~3블록과 마지막 2~3블록은 양쪽 모두 빨간색 → 이 블록들은 생성에 항상 필수적
- **중간 타임스텝**: 초기·후기 타임스텝보다 중간 단계에서 더 많은 블록이 스킵되는 경향

이 시각화는 E-DiT의 적응적 라우팅이 실제로 의미 있는 중요도 패턴을 학습했음을 정성적으로 지지한다.

---

## 8. 결론 요약, 후속 연구, 시사점

### 8-0. 저자들이 제시한 시사점

저자들은 다음을 결론으로 제시한다 (p.9):
- E-DiT는 범용 DiT 가속 프레임워크로, 블록 스킵 + MLP 너비 축소 + 블록 캐싱의 삼중 구조가 효과적
- 약 2배 가속을 이미지 및 3D 두 모달리티에서 달성
- 라우터가 학습을 통해 블록 중요도를 자연스럽게 포착

**저자들이 명시한 후속 연구 계획**: 논문에는 구체적인 후속 연구 계획이 명시되지 않았음.

---

### 8-1. 모델 일반화 성능 향상 가능성

**현재의 일반화 범위:**
- 2D 이미지: Qwen-Image(20B), FLUX(MMDiT 계열)
- 3D 자산: Hunyuan3D-3.0

**일반화 성능 향상을 위한 방향:**

1. **비디오 DiT로 확장**: CogVideoX, HunyuanVideo, Wan2.1 등에 E-DiT를 적용할 경우, 시간(temporal) 차원의 희소성까지 고려한 4D 라우팅이 필요. 시간적으로 인접한 프레임 간 잔차 캐싱과의 시너지 가능성이 높음.

2. **도메인 이동(domain shift) 강건성**: 현재 라우터는 100K 자연 이미지로 파인튜닝됨. 의료 영상, 위성 이미지 등 특수 도메인에서 라우터의 희소성 예측이 부정확할 수 있음. **도메인 적응형 라우터 재조정(router re-calibration)** 기법이 필요.

3. **조건(conditioning) 유형 다양화**: 현재는 텍스트 조건 생성에 집중. 클래스 조건, 이미지 조건, 레이아웃 조건 등 다양한 조건 유형에서 라우터의 일반화 성능 검증이 필요.

4. **해상도 스케일링**: 고해상도(예: 4K) 이미지 생성 시 시퀀스 길이 $L$이 크게 증가하므로, 라우터의 글로벌 평균풀링이 충분한 정보를 포착하는지 불명확. 계층적(hierarchical) 라우터 설계가 필요할 수 있음.

5. **조합 가능성**: 저자들이 언급한 대로 DyDiT와의 결합이 가능. 또한 step distillation(예: Consistency Models)과 결합하면 스텝 수와 블록별 연산량을 동시에 줄이는 시너지 효과 기대.

---

### 8-2. 2020년 이후 관련 최신 연구 비교 분석

> ⚠️ 아래 비교는 본 논문의 참고문헌 및 공개된 정보에 기반하며, 직접 실험 비교는 아닙니다.

| 연구 | 연도 | 핵심 방법 | E-DiT와 차이 |
|------|------|-----------|-------------|
| DDPM (Ho et al.) | 2020 | 확산 생성 모델 기초 | E-DiT의 가속 대상 |
| Rectified Flow (Liu et al.) | 2022 | 직선 경로 생성 모델 | E-DiT의 기반 프레임워크 |
| DiT (Peebles & Xie) | 2023 | Transformer 기반 확산 모델 | E-DiT의 가속 대상 아키텍처 |
| Token Merging (Bolya et al.) | 2022 | 토큰 병합으로 ViT 가속 | 정적 방법, 샘플 적응성 없음 |
| FLUX (Black Forest Labs) | 2024 | MMDiT 대규모 이미지 생성 | E-DiT 적용 대상 |
| DynamicViT (Rao et al.) | 2021 | 동적 토큰 희소화 | 비생성 모델, 이미지 인식 중심 |
| DyDiT (Zhao et al.) | 2024 | DiT 동적 어텐션 헤드/토큰 선택 | E-DiT와 보완 관계, 결합 가능 |
| Dense2MoE (Zheng et al.) | 2025 | DiT를 MoE 구조로 변환 | 고정 전문가 그룹화, 유연성 낮음 |
| TinyFusion (Fang et al.) | 2025 | 얕은 DiT 학습 | 정적 pruning, 샘플 적응성 없음 |
| PPCL (Ma et al.) | 2025 | 플러그형 pruning + 지식 증류 | 정적 방법 |
| TeaCache/FoRA (Selvaraju et al.) | 2024 | 특성 캐싱 기반 가속 | 고정 기준, 라우터 예측 미활용 |
| Qwen-Image (Wu et al.) | 2025 | 20B 이미지 생성 모델 | E-DiT 적용 대상 |

**E-DiT의 차별점 요약:**
1. **샘플 적응적 + 경량 라우터**: 기존 동적 방법들(DyDiT)은 특정 아키텍처에 종속적이나, E-DiT는 다양한 백본에 플러그인 가능
2. **캐싱의 원리적 기준**: 기존 캐싱(TeaCache, FoRA)은 경험적 기준을 사용하나, E-DiT는 학습된 라우터 확률을 캐싱 지표로 활용
3. **다중 모달리티**: 이미지와 3D를 모두 검증한 드문 가속 연구

**앞으로의 연구에 미치는 영향:**

1. **샘플 의존적 희소성 개념의 확산**: DiT 가속 연구에서 "고정 구조가 아닌 입력 적응적 구조"라는 패러다임을 강화할 것으로 예상
2. **라우터 기반 캐싱의 일반화**: 라우터 예측을 캐싱 결정에 활용하는 아이디어는 비디오, 음성, 멀티모달 등 다른 확산 모델로 확장될 수 있음
3. **효율성-품질 트레이드오프 연구의 벤치마크**: E-DiT의 Pareto curve가 새로운 비교 기준이 될 수 있음

**앞으로 연구 시 고려할 점:**

1. **배치 추론 비효율 문제**: 서로 다른 블록 경로를 가진 샘플들을 배치로 처리할 때, 조건 분기로 인한 GPU 활용률 저하 문제 해결 필요 (dynamic sparse computation 라이브러리 필요)

2. **하드웨어 인식 설계**: 현재 행렬 슬라이싱 방식은 실제 GPU에서 메모리 접근 패턴이 비효율적일 수 있음. FlashAttention처럼 하드웨어 최적화된 커스텀 커널 필요

3. **라우터의 학습 비용**: E-DiT 라우터 학습에 32개의 H20 GPU가 사용되었다는 점은 소규모 연구팀에는 접근이 어려울 수 있음. 더 적은 데이터와 연산으로 라우터를 학습하는 **효율적 라우터 학습법** 연구 필요

4. **공정한 비교를 위한 표준화**: PPCL, DyDiT 등의 수치를 원 논문에서 인용하는 현행 비교 방식은 공정성 문제가 있음. 통일된 벤치마크 환경 구축 필요

---

## 참고자료

**본 답변 작성에 참고한 자료:**

1. **주요 분석 대상 논문:**
   - Wang, J. et al. "Elastic Diffusion Transformer." arXiv:2602.13993v1, February 2026.

2. **논문 내 인용 핵심 참고문헌:**
   - Peebles, W. & Xie, S. "Scalable Diffusion Models with Transformers." ICCV 2023.
   - Liu, X. et al. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." arXiv:2209.03003, 2022.
   - Ho, J. et al. "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
   - Bengio, Y. et al. "Estimating or Propagating Gradients through Stochastic Neurons for Conditional Computation." arXiv:1308.3432, 2013. *(STE)*
   - Zhao, W. et al. "Dynamic Diffusion Transformer." arXiv:2410.03456, 2024. *(DyDiT)*
   - Zheng, Y. et al. "Dense2MoE." ICCV 2025.
   - Ma, J. et al. "Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers (PPCL)." arXiv:2511.16156, 2025.
   - Fang, G. et al. "TinyFusion." CVPR 2025.
   - Wimbauer, F. et al. "Cache Me If You Can." 2024.
   - Wu, C. et al. "Qwen-Image Technical Report." arXiv:2508.02324, 2025.
   - Labs, B.F. "FLUX." GitHub, 2024.
   - Team, H. "Hunyuan3D-3.0." 2025.
   - Ghosh, D. et al. "GenEval." NeurIPS 2023.
   - Hu, X. et al. "ELLA / DPG-Bench." arXiv:2403.05135, 2024.
   - Huang, K. et al. "T2I-CompBench++." TPAMI 2025.
   - Xue, L. et al. "ULIP." CVPR 2023.
   - Zhou, J. et al. "Uni3D." arXiv:2310.06773, 2023.

> ⚠️ **면책 고지**: 본 답변은 제공된 PDF 원문(arXiv:2602.13993v1)을 기반으로 작성되었습니다. 섹션 8-2의 비교 분석 일부는 논문 내 인용 정보와 공개된 연구 동향을 종합한 것으로, 직접 실험 비교 결과가 아닙니다. 확인이 불가능한 내용은 포함하지 않았습니다.
