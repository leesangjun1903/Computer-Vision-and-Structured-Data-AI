# Gen4U: Unifying Video Generation and Understanding via Diffusion 

> **⚠️ 투명성 고지**: 본 답변은 제공된 논문 PDF(arXiv:2607.06856v1)를 직접 분석한 내용에 기반합니다. 불확실한 부분은 명시적으로 표기하겠습니다.

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

> *"대규모 비디오 확산 모델(Video Diffusion Model)은 고수준 의미론(high-level semantics)을 포착하지 못한다"*는 기존 통념을 반박하고, **동결된(frozen) 생성 모델의 중간 표현(intermediate representations)만으로도 다양한 비디오 이해 작업을 경쟁력 있게 수행할 수 있음**을 최초로 체계적으로 증명한다.

### 주요 기여 3가지

| 기여 | 내용 |
|------|------|
| **(a) 잠재 공간 분석** | Mutual-kNN 정렬 메트릭을 통해 네트워크 깊이(depth)와 노이즈 레벨(noise level)에 따른 확산 표현의 구조적 진화 매핑 |
| **(b) 의미론 및 시간 역학** | SSv2 비디오 분류에서 SOTA 달성(72.6%), 이미지/비디오 캡셔닝에서 만족스러운 성능 |
| **(c) 기하학적 이해** | 단안 깊이 추정(monocular depth estimation) 및 카메라 자세 추정(camera pose estimation)에서 강한 성능 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

현재 시각 표현 학습 패러다임의 근본적 딜레마:

```
MAE 계열 (VideoMAE, 4DS, D4RT)
  → 우수: 저수준 기하학(geometry), 국소 움직임
  → 취약: 광범위한 의미론적 일반화

대조 학습 계열 (CLIP, SigLIP, DINO)
  → 우수: 고수준 의미론(semantics)
  → 취약: 정밀한 시공간적 세부사항

기존 확산 모델 연구 (V-WALT 등)
  → 저수준 기하학만 포착 가능하다는 결론
  → 고수준 의미론 포착 실패로 범용 인코더 부적합 판정
```

**Gen4U의 핵심 질문**: *최신 대규모 비디오 확산 모델은 정말 고수준 의미론을 포착하지 못하는가?*

---

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 배경: 잠재 확산 모델 (LDM)

입력 비디오 $x \in \mathbb{R}^{F \times H \times W \times C}$ 에 대해, 인코더 $E$가 잠재 코드를 생성:

$$z_0 = E(x)$$

**DDPM 계열 (Veo3)**: 노이즈 레벨 $t \in [0, T]$에서 손상된 잠재:

$$z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

학습 목표 (표준 재가중 변분 하한):

$$\mathcal{L}_{LDM} = \mathbb{E}_{z_0, \epsilon, t} \left[ \| \epsilon - f_\theta(z_t, t, c) \|_2^2 \right]$$

**Flow Matching 계열 (Wan 2.2)**: 연속 시간 $t \in [0, 1]$에서:

$$z_t = (1-t)z_0 + t\epsilon$$

속도 벡터장(velocity vector field) 예측:

$$f_t = \frac{dz_t}{dt} = \epsilon - z_0$$

학습 목표 (Flow Matching):

$$\mathcal{L}_{FM} = \mathbb{E}_{z_0, \epsilon, t} \left[ \| f_\theta(z_t, t, c) - (\epsilon - z_0) \|_2^2 \right]$$

백본 $f_\theta$의 $l$번째 레이어에서 노이즈 스텝 $t$의 중간 시공간 활성화:

$$h_t^{(l)} \in \mathbb{R}^{F' \times H' \times W' \times D}$$

---

#### 2.2.2 Mutual k-NN (MkNN) 정렬 메트릭

비디오 $N$개에 대해 확산 인코더로부터 임베딩 추출 (전역 평균 풀링):

$$\mathbf{x}_i = \text{AvgPool}\left(h_t^{(l)}(v_i)\right) \in \mathbb{R}^D \tag{1}$$

참조 인코더(텍스트 또는 독립적으로 학습된 비디오 인코더):

$$\mathbf{y}_i = E_{\text{ref}}(c_i) \in \mathbb{R}^{D'}, \quad Y \in \mathbb{R}^{N \times D'} \tag{2}$$

$k$-최근접 이웃 그래프를 나타내는 이진 지시 행렬:

$$M_{ij}^X = \begin{cases} 1 & \text{if } j \in \mathcal{N}_k^X(i) \\ 0 & \text{otherwise} \end{cases} \tag{3}$$

**MkNN 정렬 점수** (핵심 평가 메트릭):

$$\mathcal{A}_{\text{MkNN}}(X, Y) = \frac{1}{kN} \sum_{i=1}^{N} \sum_{j=1}^{N} \left(M^X \odot M^Y\right)_{ij} = \frac{1}{N} \sum_{i=1}^{N} \frac{|\mathcal{N}_k^X(i) \cap \mathcal{N}_k^Y(i)|}{k} \tag{4}$$

최적 레이어 쌍 탐색:

$$\left(l^\star, l'^\star\right) = \arg\max_{l, l'} \mathcal{A}_{\text{MkNN}}\left(X^{(l)}, Y^{(l')}\right) \tag{5}$$

실험에서 $k=10$, $N=1024$ 사용.

---

#### 2.2.3 다운스트림 작업별 경량 디코더

**비디오 분류 (선형 프로브)**:
$$\hat{y} = W \cdot \text{GlobalPool}(h_t^{(l)}) + b$$

**비디오 분류 (어텐션 프로브)**: 태스크별 쿼리를 사용한 크로스-어텐션 헤드 (1-블록)

**깊이 추정**: Dense Prediction Transformer (DPT) 헤드 (~23M 파라미터), Scale-Invariant Log (SiLog) 손실 사용

**캡셔닝**: 크로스-어텐션 어댑터(3블록) → 32개 학습 가능 쿼리 토큰 → Gemma-2 (2B) LLM

**결합 손실 함수들 (부록 B)**:

- 분류용 교차 엔트로피:

$$\mathcal{L}_{CE} = -\frac{1}{B}\sum_{i=1}^{B}\sum_{c=1}^{C} \mathbf{1}[y_i = c] \log \frac{\exp(\hat{y}_{i,c})}{\sum_{c'=1}^{C} \exp(\hat{y}_{i,c'})}$$

- 교차 모달 정렬용 Multi-positive InfoNCE:

$$\mathcal{L}_{InfoNCE} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\sum_{j \in \mathcal{P}(i)} \exp(\text{sim}(\mathbf{v}_i, \mathbf{v}_j)/\tau)}{\sum_{k=1}^{B} \exp(\text{sim}(\mathbf{v}_i, \mathbf{v}_k)/\tau)}$$

---

### 2.3 모델 구조

```
입력 비디오 x
      ↓
  VAE 인코더 E
      ↓
  잠재 코드 z₀
      ↓ (노이즈 주입, 고정 seed)
손상된 잠재 z_t (단일 노이즈 레벨 t 선택)
      ↓
  확산 트랜스포머 f_θ (Frozen)
  [Veo3: DiT 기반, 독점 모델]
  [Wan 2.2: Flow Matching 기반, 오픈소스]
      ↓
  최적 중간 활성화 h_t^(l) 추출
  (depth ≈ 70-80%, noise ≈ 30-60%)
      ↓
  ┌─────────────────────────────────┐
  │     경량 태스크별 디코더들        │
  │  (학습 가능, 백본은 완전 동결)   │
  ├─────────────────────────────────┤
  │ • 비디오 분류: 선형/어텐션 프로브 │
  │ • 깊이 추정: DPT 헤드            │
  │ • 카메라 자세: 어텐션 디코더     │
  │ • 캡셔닝: 크로스어텐션 + Gemma2  │
  └─────────────────────────────────┘
      ↓
  생성 경로: Decoder D → 고품질 비디오 (보존)
```

**핵심 설계 원칙**: 단일 순전파(single forward pass)만 수행 → 반복적 디노이징 불필요 → 표준 판별 인코더와 동일한 계산 효율성

---

### 2.4 성능 향상

#### SSv2 비디오 분류 (Table 1)

| 모델 | 사전학습 방법 | 파라미터(M) | Top-1 정확도(%) |
|------|-------------|------------|----------------|
| VideoMAEv2-g | MAE | 1,013 | 65.6 |
| VideoPrism-g | MAE + 대조 | 1,113 | 65.4 |
| 4DS-j | MAE | 21,495 | 68.2 |
| InternVideo2 | MAE + 대조 + 캡셔닝 | 6,000 | 67.7 |
| V-JEPA-H | 마스크 특징 예측 | 635 | 72.2 |
| V-Walt | 확산 | 1,900 | 59.7 |
| **Gen4U (ours)** | **확산** | **-** | **71.3** |
| **Gen4U + 데이터 증강** | **확산** | **-** | **72.6 (SOTA)** |

#### 기하학 이해 (ScanNet)

- **깊이 추정**: AbsRel = **0.075**, $\delta_1$ = 0.952 (4DS 기준선 0.084 대비 10.7% 상대 개선)
- **카메라 자세 추정**: EPE = **1.10** (DINOv2 기준선 1.08과 동등)

#### 캡셔닝 (Table 2, 일부 발췌)

| 모델 | SSv2 CIDEr | COCO CIDEr | Vatex CIDEr |
|------|-----------|-----------|------------|
| SigLIP-so400m/14 | 204.5 | 118.5 | 66.0 |
| Gen4U @ 30% | **289.5** | 54.9 | 44.8 |
| Gen4U + Noise Aug. | 280.4 | 69.3 | 56.7 |
| Gen4U + Noise Aug. + High res. | - | 102.0 | - |

**관찰**: SSv2(시간적 이해)에서 SigLIP 압도, COCO/Vatex(정적 이미지/범용 캡셔닝)에서는 열세 → 생성 모델이 시간 역학 이해에 특화

---

### 2.5 한계

논문이 직접 명시한 한계:

1. **재현성 제한**: 주요 실험이 독점 모델인 Veo3에서 수행됨 → 외부 연구자가 완전히 재현 불가
2. **캡셔닝 성능 격차**: COCO와 Vatex에서 대규모 시각-언어 데이터로 최적화된 SigLIP 기반 모델에 열세
3. **이중모드 패턴 미해명**: Veo3에서 발견된 bimodal 정렬 패턴의 이론적 설명 부재 → 향후 연구 과제로 남김
4. **데이터 증강 의존성**: 소규모 데이터셋(COCO, Vatex)에서는 과적합 방지를 위해 노이즈 레벨 증강 필요

---

## 3. 일반화 성능 향상 가능성 (중점 분석)

### 3.1 일반화 성능의 핵심 메커니즘

#### (1) 60% 노이즈 의미론적 병목 (Semantic Bottleneck)

두 이질적인 모델(Veo3, Wan 2.2) 모두에서 $t = 60\%$ 노이즈 레벨이 텍스트 및 시각 인코더와의 정렬을 극대화하는 보편적 sweet spot임이 발견됨.

이는 **특정 노이즈 레벨이 모델 아키텍처와 무관하게 범용 의미론적 표현을 인코딩**한다는 것을 시사:

$$t^\star \approx 60\%, \quad l^\star \approx 70\text{-}80\% \text{ (depth)}$$

#### (2) 생성 능력과 표현 품질의 상관관계

논문의 중요한 발견:
- V-WALT (이전 세대) < Wan 2.2 < Veo3 순으로 정렬 점수가 향상
- **"더 강한 생성 모델 → 더 풍부한 내부 표현"** 법칙이 성립
- 이는 스케일링 법칙이 생성뿐만 아니라 표현 학습에도 적용됨을 시사

#### (3) 선형 어댑터의 우월한 일반화

부록 B 결과에서 중요한 발견:

| 어댑터 | 학습 데이터 | 학습 목표 | SSv2 정확도 |
|--------|-----------|---------|------------|
| 최적 단일 블록 (기준선) | - | - | 17.04% |
| **선형** | VATEX | 텍스트 정렬 | **21.4%** |
| 선형 | SSv2 | 텍스트 정렬 | 21.9% |
| 크로스-어텐션 | SSv2 | 분류 | 24.9% |

**핵심 관찰**: VATEX로 학습한 선형 어댑터가 SSv2 out-of-domain 분류로 제로샷 일반화 성공 → **학습 도메인과 무관한 범용 표현 존재 확인**

#### (4) 분류 학습이 텍스트 정렬을 동시에 개선

역설적 발견:
> 교차 엔트로피 분류 손실로 학습한 어댑터가 텍스트 정렬을 *명시적으로* 최대화하도록 학습한 어댑터보다 **더 높은 텍스트-모달 정렬 점수**를 보임.

저자들은 이를 다음과 같이 해석:
- 분류의 이산적 레이블 신호가 다중 양성 대조 정렬의 노이즈 그래디언트보다 더 안정적인 학습 신호를 제공
- 즉, 잠재 공간 내 의미론적 구조가 이미 충분히 형성되어 있어서 분류 신호만으로도 활성화 가능

#### (5) 다중 노이즈 레벨 데이터 증강의 일반화 효과

```math
\text{학습}: t \in \{10\%, 30\%, 60\%, 90\%\}, \quad \text{추론}: t = 30\%
```

- COCO: +32.7% CIDEr 향상 (단일 노이즈 레벨 대비)
- 다양한 노이즈 레벨이 서로 다른 추상화 수준의 정보를 제공하여 표현의 다양성 증가 → **일반화 성능 향상**

#### (6) 어텐션 프로브의 공간적 일반화

낮은 노이즈 레벨에서 의미론적 정보가 패치별로 공간적으로 산재됨:

$$t \approx 30\%, l \approx 80\% \rightarrow \text{어텐션 프로브 최적}$$
$$t \approx 60\%, l \approx 70\% \rightarrow \text{선형 프로브 최적}$$

어텐션 프로브는 동적 시공간 풀링 메커니즘으로 작동하여 산재된 세밀한 정보를 집약 → **다양한 다운스트림 태스크에 대한 적응적 일반화 가능**

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 패러다임별 비교

| 연구 | 방법 | 특징 | SSv2 | 한계 |
|------|------|------|------|------|
| **VideoMAE** (NeurIPS'22) | MAE 시간 마스킹 | 데이터 효율적 | 65.6% | 의미론 약함 |
| **VideoMAEv2** (CVPR'23) | Dual masking 스케일링 | 1B 파라미터 | 65.6% | 생성 불가 |
| **VideoPrism** (ICML'24) | MAE + 대조학습 혼합 | 범용 비디오 인코더 | 65.4% | 복잡한 파이프라인 |
| **InternVideo2** (ECCV'24) | MAE+대조+캡셔닝 | 멀티태스크 | 67.7% | 6B 파라미터, 과중 |
| **V-JEPA** (TMLR'24) | 잠재 공간 예측 | 픽셀 재구성 회피 | 72.2% | 생성 미지원 |
| **V-JEPA 2** (arXiv'25) | 예측+계획 통합 | 행동 계획 | - | 생성 미지원 |
| **V-WALT** (ICCV'25) | 확산 표현 분석 | 저수준만 포착 | 59.7% | 고수준 의미론 실패 |
| **4DS** (arXiv'25) | MAE 4D 스케일링 | 21B 파라미터 | 68.2% | 엄청난 모델 크기 |
| **Transfusion** (ICLR'25) | 생성+이해 공동학습 | 단일 모델 | - | 처음부터 공동학습 필요 |
| **Gen4U (Ours)** | 확산 표현 재활용 | 동결 모델 활용 | **72.6%** | 독점 모델 의존 |

### 주목할 만한 비교 포인트

**vs. V-JEPA (72.2%)**: Gen4U가 동등 혹은 미세하게 우월하며, 추가로 **고품질 비디오 생성 능력**을 유지함. V-JEPA는 생성 미지원.

**vs. 4DS (68.2%, 21.5B 파라미터)**: Gen4U가 명시적 파라미터 수를 밝히지 않았으나 (Veo3 파라미터 수 미공개), 확산 표현 품질이 대규모 MAE 스케일링을 능가함.

**vs. V-WALT (59.7%)**: 동일한 확산 계열이지만 12.6%p 차이. 이는 **생성 모델의 능력(세대)이 표현 품질에 직접 영향**을 미친다는 증거.

**vs. Transfusion**: Transfusion은 처음부터 생성+이해를 공동학습하는 반면, Gen4U는 **기존 생성 모델을 수정 없이 재활용** → 실용성 측면에서 차별화.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 앞으로의 연구에 미치는 영향

#### (1) 패러다임 전환: "생성 = 이해"의 가능성 제시

Gen4U는 생성과 이해를 **별도 모델로 유지하던 기존 관행에 근본적 의문**을 제기한다. 하나의 동결된 생성 모델이 다양한 이해 태스크를 지원할 수 있다면, 미래의 AI 시스템은 단일 비디오 파운데이션 모델로 통합될 수 있다.

#### (2) 확산 표현 연구의 활성화

- 이미지 확산(Stable Diffusion 계열)에서도 유사한 의미론적 표현이 존재하는가?
- 노이즈 레벨과 의미론적 추상화 수준의 이론적 관계는 무엇인가?
- Veo3의 bimodal 패턴의 이론적 설명 → 대형 트랜스포머의 정보 라우팅 메커니즘 연구로 연결

#### (3) 스케일링 법칙의 확장

**"더 강한 생성 → 더 나은 표현"** 법칙은 생성 AI 연구의 투자가 이해 AI에도 간접적으로 기여함을 의미한다. 이는 생성 모델 스케일링의 정당성을 강화하는 추가적 근거가 된다.

#### (4) 실용적 응용: 단일 모델 배포

현재 많은 시스템이 생성용 모델과 이해용 모델을 별도로 운영한다. Gen4U가 제안하는 통합 패러다임은:
- 인프라 비용 절감
- 모델 유지관리 단순화
- 생성과 이해 간 정보 공유 가능성

---

### 5.2 앞으로 연구 시 고려할 점

#### ⚠️ 주의해야 할 방법론적 과제

**1. 재현성 문제 해결**

논문 자체가 명시한 가장 큰 한계. 후속 연구는:
- 완전 오픈소스 확산 모델(Wan 2.2, CogVideoX, Open-Sora 등)에서 동일 실험 재현 필요
- Veo3와 Wan 2.2 간의 성능 격차(Figure 3)가 시사하는 "생성 능력 스케일링" 효과를 오픈소스 모델 규모에서 체계적으로 검증

**2. 단일 고정 노이즈 레벨의 한계**

현재 Gen4U는 단일 $(t, l)$ 쌍에서 표현을 추출하지만, 부록 B는 여러 노이즈 레벨 조합이 보완적 정보를 포함함을 시사한다. 향후 연구는:
- 효율적인 다중 레벨 표현 융합 방법론 개발
- 태스크 적응적 최적 $(t, l)$ 선택 자동화

**3. 텍스트 컨디셔닝 편향**

Gen4U는 *"A video of a scene"*이라는 중립적 프롬프트를 사용하지만, 의미론적 이해 능력이 텍스트 컨디셔닝 학습에서 기인하는지, 아니면 순수 시각 생성 능력에서 기인하는지 분리 분석이 필요하다.

**4. 고정 시드(Fixed Seed) 의존성**

확산 과정의 확률성을 제어하기 위해 고정 시드를 사용했는데, 이것이 실제 배포 환경에서의 강건성에 미치는 영향을 평가해야 한다.

**5. 태스크별 최적 설정의 이론화**

현재는 실험적으로 발견된 $(t \approx 60\%, l \approx 70\text{-}80\%)$ 규칙이 왜 성립하는지에 대한 이론적 설명이 부재하다. 이는 정보 이론적 관점 또는 표현 기하학(representation geometry) 연구로 접근 가능하다.

**6. 동적 비디오 이해의 한계**

SSv2에서는 강하지만 COCO(정적 이미지)에서 상대적으로 약한 성능은, **확산 모델의 표현이 시간적 동역학에 편향**되어 있을 가능성을 시사한다. 정적 장면에 대한 일반화 방법론 연구가 필요하다.

**7. 윤리적 고려사항**

Veo3와 같은 대규모 독점 모델을 기반으로 한 연구는:
- 접근성 불평등 문제
- 모델 내부 편향의 이해 태스크로의 전파 가능성
- 대규모 모델의 탄소 발자국 문제를 함께 고려해야 함

---

## 참고문헌 (논문에서 직접 인용된 주요 문헌)

본 답변은 다음 자료를 직접 분석하여 작성되었습니다:

**주요 출처**:
- King, M. et al. "Gen4U: Unifying Video Generation and Understanding via Diffusion." arXiv:2607.06856v1, 7 Jul 2026.

**논문 내 인용 주요 참고문헌**:
- Vélez, P. et al. "From image to video: An empirical study of diffusion representations." ICCV, 2025.
- Zhu, T. et al. "Dynamic reflections: Probing video representations with text alignment." ICLR, 2026.
- Huh, M. et al. "Position: The platonic representation hypothesis." ICML, 2024.
- Bardes, A. et al. "Revisiting feature prediction for learning visual representations from video (V-JEPA)." TMLR, 2024.
- Carreira, J. et al. "Scaling 4d representations (4DS)." arXiv:2412.15212, 2025.
- Wang, L. et al. "VideoMAE v2: Scaling video masked autoencoders with dual masking." CVPR, 2023.
- Zhao, L. et al. "VideoPrism: A foundational visual encoder for video understanding." ICML, 2024.
- Wang, Y. et al. "InternVideo2: Scaling foundation models for multimodal video understanding." ECCV, 2024.
- Oquab, M. et al. "DINOv2: Learning robust visual features without supervision." TMLR, 2024.
- Zhou, C. et al. "Transfusion: Predict the next token and diffuse images with one multi-modal model." ICLR, 2025.
- Gupta, A. et al. "Photorealistic video generation with diffusion models (WALT)." ECCV, 2023.
- Google DeepMind. "Veo 3 technical report." Technical report, 2025.
- Wan Team. "Wan: Open and advanced large-scale video generative models." arXiv:2503.20314, 2025.
- Luo, G. et al. "Diffusion hyperfeatures: Searching through time and space for semantic correspondence." NeurIPS, 2023.
- Valeriani, L. et al. "The geometry of hidden representations of large transformer models." NeurIPS, 2023.
- Dieleman, S. "Diffusion is spectral autoregression." sander.ai, 2024.
