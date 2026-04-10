# MambaIRv2: Attentive State Space Restoration

---

## 1. 핵심 주장과 주요 기여 요약

### 1.1 핵심 주장

MambaIRv2는 기존 Mamba 기반 영상 복원 모델이 갖는 **인과적 상태공간 모델링(causal state-space modeling)의 한계**를 해결하기 위해, ViT(Vision Transformer)의 비인과적(non-causal) 처리 능력을 Mamba에 통합한 **"어텐티브 상태공간 복원(Attentive State Space Restoration)"** 모델을 제안한다.

### 1.2 세 가지 핵심 기여

| 기여 | 설명 |
|------|------|
| **① Attentive State-space Equation (ASE)** | 프롬프트 학습을 통해 스캔되지 않은 시퀀스의 픽셀도 참조 가능하게 함 → 단일 방향 스캔으로도 전역 정보 활용 |
| **② Semantic Guided Neighboring (SGN)** | 의미적으로 유사한 픽셀을 1D 시퀀스 내 공간적으로 가깝게 재구성 → 장거리 감쇄(long-range decay) 완화 |
| **③ MambaIRv2 통합 아키텍처** | ASSM(Attentive State Space Module) + 윈도우 MHSA를 결합한 로컬-글로벌 계층적 모델링 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 Mamba 기반 영상 복원 방법(MambaIR 등)은 세 가지 근본적 문제를 안고 있다:

**① 제한된 인과적 인식 (Constrained Causal Perception)**
- $i$번째 픽셀은 오직 이전 $i-1$개의 픽셀에만 의존
- 아직 스캔되지 않은 픽셀은 전혀 활용 불가

**② 다방향 스캔의 비효율성 및 중복성**
- 이를 보완하기 위해 4방향 스캔을 사용하지만, 각 방향 스캔 간 코사인 유사도가 0.7 이상으로 매우 높은 중복성 존재
- 연산량이 크게 증가함

**③ 장거리 감쇄 (Long-range Decay)**
- 제어 행렬 $\overline{\mathbf{A}}$의 값이 통계적으로 1보다 작음
- 두 픽셀 간 거리 $k$가 클수록 상호작용이 $\overline{\mathbf{A}}^k$에 비례하여 지수적으로 감소

---

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 Mamba의 기본 상태공간 방정식

$$h_i = \overline{\mathbf{A}}h_{i-1} + \overline{\mathbf{B}}x_i$$

$$y_i = \mathbf{C}h_i + \mathbf{D}x_i $$

여기서:
- $\overline{\mathbf{A}} = \exp(\Delta \mathbf{A})$: 제어 행렬 (control matrix)
- $\overline{\mathbf{B}} \approx \Delta \mathbf{B}$: 입력 행렬 (input matrix)
- $\mathbf{C}$: 출력 행렬 (output matrix) — **어텐션의 Query에 해당**
- $\mathbf{D}$: 스킵 커넥션 항

#### 2.2.2 Attention과 State Space의 수학적 연결

**인과 선형 어텐션 (Causal Linear Attention):**

$$y_i = \frac{\mathbf{Q}_i \left(\sum_{j=1}^{i} \mathbf{K}_j^\top \mathbf{V}_j\right)}{\mathbf{Q}_i \left(\sum_{t=1}^{i} \mathbf{K}_t^\top\right)} $$

$\mathbf{S}\_i = \sum_{j=1}^{i} \mathbf{K}\_j^\top \mathbf{V}\_j$, $\mathbf{Z}\_i = \sum_{t=1}^{i} \mathbf{K}_t^\top$로 표기하면:

$$y_i = \mathbf{Q}_i \mathbf{S}_i / \mathbf{Q}_i \mathbf{Z}_i $$

**공통 형식으로 정리:**

```math
\mathbf{S}_i = \mathbf{I}\mathbf{S}_{i-1} + \mathbf{K}_i^\top \mathbf{V}_i, \quad y_i = \mathbf{Q}_i\mathbf{S}_i/\mathbf{Q}_i\mathbf{Z}_i + \mathbf{O}x_i
```

**상태공간 방정식 공통 형식:**

$$h_i = \overline{\mathbf{A}}h_{i-1} + \mathbf{B}(\Delta x_i), \quad y_i = \mathbf{C}h_i/\mathbf{I} + \mathbf{D}x_i $$

**핵심 대응 관계:**

$$h_i \sim \mathbf{S}_i, \quad \mathbf{B} \sim \mathbf{K}^\top, \quad \mathbf{C} \sim \mathbf{Q}$$

> ✅ **통찰**: $\mathbf{C}$가 어텐션의 Query에 해당하므로, $\mathbf{C}$에 미스캔 픽셀의 정보를 주입하면 비인과적 쿼리가 가능해진다.

#### 2.2.3 Attentive State-space Equation (ASE)

**프롬프트 풀 구성 (의미론적 분리, Semantic Decoupling):**

$$\mathcal{P} = \mathbf{M}\mathbf{N}, \quad \mathbf{M} \in \mathbb{R}^{T \times r}, \quad \mathbf{N} \in \mathbb{R}^{r \times d} $$

- $T$: 프롬프트 풀의 크기
- $r$: 내부 랭크 ($r \ll \min\{T, d\}$)
- $\mathbf{N}$: 블록 간 공유 (공통 특징 공간)
- $\mathbf{M}$: 블록별 특화 (조합 계수)

**인스턴스별 프롬프트 선택 과정:**

1. 입력 특징 $\mathbf{x}' \in \mathbb{R}^{L \times C}$에 선형 레이어 적용 → 채널 $C \to T$ 투영
2. LogSoftmax로 각 프롬프트의 샘플링 확률 예측
3. Gumbel-Softmax 트릭으로 미분 가능한 원-핫 라우팅 행렬 $\mathbf{R} \in \mathbb{R}^{L \times T}$ 획득
4. $\mathbf{P} = \mathbf{R}\mathcal{P}$로 인스턴스별 프롬프트 $\mathbf{P} \in \mathbb{R}^{L \times d}$ 생성

**ASE 최종 수식:**

$$h_i = \overline{\mathbf{A}}h_{i-1} + \overline{\mathbf{B}}x_i$$

$$y_i = (\mathbf{C} + \mathbf{P})h_i + \mathbf{D}x_i $$

> $\mathbf{C}$에 $\mathbf{P}$를 잔차 덧셈으로 통합함으로써, 스캔되지 않은 픽셀의 정보를 활용하는 비인과적 쿼리가 가능해짐

#### 2.2.4 장거리 감쇄의 수학적 증명

상태공간 방정식을 반복 전개하면:

$$y_k = \mathbf{C}\overline{\mathbf{A}}^k\overline{\mathbf{B}}x_0 + \mathbf{C}\overline{\mathbf{A}}^{k-1}\overline{\mathbf{B}}x_1 + \cdots + \mathbf{C}\overline{\mathbf{B}}x_k + \mathbf{D}x_k $$

- $x_0$이 $y_k$에 기여하는 가중치는 $\mathbf{C}\overline{\mathbf{A}}^k\overline{\mathbf{B}}$로, $\overline{\mathbf{A}}^k$에 비례
- $\overline{\mathbf{A}}$의 평균값이 통계적으로 1 미만이므로, $k$가 클수록 $\overline{\mathbf{A}}^k \to 0$
- **→ 장거리 픽셀 간 상호작용이 지수적으로 감쇄됨**

#### 2.2.5 Semantic Guided Neighboring (SGN)

- **SGN-unfold**: $i$번째 프롬프트 카테고리의 픽셀들을 같은 의미 그룹으로 묶어 1D 시퀀스로 재구성
- ASE를 통한 상태공간 모델링 수행
- **SGN-fold**: SGN-unfold의 역변환으로 다시 공간적 특징 맵으로 복원

> 💡 **핵심**: 공간적으로 멀리 있지만 의미적으로 유사한 픽셀들이 1D 시퀀스에서 인접하게 되어, 장거리 감쇄 문제를 구조적으로 완화

---

### 2.3 모델 구조

```
LQ Image
    ↓
Conv 3×3 (Shallow Feature Extraction)
    ↓
[Attentive State Space Group (ASSG)] × N
    ├── [Attentive State Space Block (ASSB)] × M
    │       ├── Norm → Window MHSA → Norm → FFN  (Local)
    │       └── Norm → ASSM → Norm → FFN          (Global)
    │               └── ASSM:
    │                   ├── Positional Encoding
    │                   ├── SGN-unfold
    │                   ├── ASE (Prompt Pool + Gumbel Routing)
    │                   └── SGN-fold → Linear
    ↓
Task-specific Reconstruction
(PixelShuffle for SR / Conv for Denoising)
    ↓
HQ Image
```

**모델 변형:**
- **MambaIRv2-light**: 경량 SR용 (~774K params)
- **MambaIRv2-S**: 소형 클래식 SR (9.6M params)
- **MambaIRv2-B**: 기본 클래식 SR (22.9M params)
- **MambaIRv2-L**: 대형 클래식 SR (34.2M params)

---

### 2.4 성능 향상

#### 경량 SR (Lightweight SR)

| 방법 | Scale | #Params | Urban100 PSNR | Manga109 PSNR |
|------|-------|---------|---------------|---------------|
| SRFormer-light | 2× | 853K | 32.91 | 39.28 |
| **MambaIRv2-light** | 2× | **774K** | **33.26 (+0.35)** | **39.35** |
| MambaIR-4방향 | 2× | 1.36M | 32.92 | 39.31 |
| **MambaIRv2** | 2× | **774K** | **33.26 (+0.34)** | **39.39** |

> ✅ 파라미터 9.3% 감소 + PSNR 0.35dB 향상

#### 클래식 SR (Classic SR)

| 방법 | Scale | Urban100 PSNR | Manga109 PSNR |
|------|-------|---------------|---------------|
| HAT | 2× | 34.45 | 40.26 |
| **MambaIRv2-B** | 2× | 34.49 | **40.42 (+0.16)** |
| **MambaIRv2-L** | 2× | **34.60** | 40.55 |

#### 계산 복잡도 비교 (2× SR, 출력 256×256)

| 모델 | #Params | MACs | Urban100 | Manga109 |
|------|---------|------|----------|----------|
| HAT | 20.8M | 514.9G | 34.45 | 40.26 |
| MambaIRv2-B | 22.9M | **445.8G (-13.4%)** | 34.49 | 40.42 |

#### 이미지 디노이징 (σ=15)

| 방법 | Urban100 PSNR |
|------|---------------|
| Restormer (U-Net) | 35.13 |
| MambaIR | 35.37 |
| **MambaIRv2** | **35.42 (+0.29 vs Restormer)** |

---

### 2.5 한계점

논문이 직접 명시한 한계(Section F):

1. **해석가능성 부족**: Mamba와 ViT가 영상 복원 과정에서 실제로 무엇을 학습하는지에 대한 깊은 해석적 분석이 부재
2. **제한적 태스크 커버리지**: 디블러링(deblurring), 디헤이징(dehazing), 디레이닝(deraining) 등 다른 복원 태스크 미적용
3. **U-Net 구조 미적용**: 디노이징에서 U-Net이 유리하다고 알려져 있으나, 아키텍처 일관성을 위해 사용하지 않음 → U형 MambaIRv2 적용 가능성 미탐색
4. **초기 단계**: Mamba 기반 영상 복원 연구는 아직 초기 단계로, 더 많은 연구가 필요

---

## 3. 모델의 일반화 성능 향상 가능성

MambaIRv2는 여러 측면에서 일반화 성능 향상에 기여하는 구조적 특성을 갖고 있다.

### 3.1 다중 태스크 일반화 능력 검증

논문은 단일 아키텍처로 세 가지 이질적 태스크에서 SOTA를 달성하여 범용 백본으로서의 가능성을 입증했다:

| 태스크 | 결과 |
|--------|------|
| 이미지 초해상화 (Classic SR) | 5개 벤치마크 전반에서 최고 성능 |
| JPEG 압축 아티팩트 제거 (JPEG CAR) | Classic5, LIVE1 전체 품질 인자에서 최고 성능 |
| 가우시안 컬러 디노이징 | CBSD68, Kodak24, McMaster, Urban100 전체에서 최고 성능 |

### 3.2 비인과적 모델링을 통한 일반화

**이미지 복원은 본질적으로 비인과적 태스크**이다. 기존 Mamba의 인과적 모델링은 이 태스크와 구조적 불일치를 초래한다. MambaIRv2의 ASE는:

- 전체 이미지 픽셀에 대한 어텐션 유사 쿼리를 가능하게 함
- 특정 스캔 방향에 의존하지 않으므로 다양한 이미지 구조에 더 강건함

### 3.3 의미론적 프롬프트 풀의 범용성

$$\mathcal{P} = \mathbf{M}\mathbf{N}, \quad \mathbf{N}: \text{공유}, \mathbf{M}: \text{블록별}$$

- **$\mathbf{N}$ (공유 공간)**: 서로 다른 블록이 공통의 시각적 특징 공간을 학습 → 다양한 이미지 패턴에 대한 일반화된 표현 학습
- **$\mathbf{M}$ (블록별)**: 각 레이어 수준에서 특화된 표현 학습 가능
- 이 구조는 **다양한 훈련 데이터 분포에 대해 안정적인 적응**을 가능하게 한다

### 3.4 SGN의 데이터 독립적 의미론적 그룹화

SGN은 데이터의 의미론적 구조를 명시적으로 활용하여 픽셀 간 상호작용을 재정의한다:

- 공간적 위치보다 **의미론적 유사성**에 기반한 시퀀스 구성
- 이는 훈련 데이터셋에서 보지 못한 새로운 이미지 패턴에도 일반화 가능한 구조를 제공
- 장거리 유사 픽셀 간 상호작용이 강화되어, 다양한 텍스처와 구조를 가진 이미지에서도 효과적

### 3.5 스케일링 능력 (Scaling Law)

| 모델 | 4× Urban100 PSNR |
|------|-----------------|
| MambaIRv2-S (9.8M) | 27.73 |
| MambaIRv2-B (23.1M) | 27.89 |
| MambaIRv2-L (34.2M) | **28.07 (+0.18)** |

모델 크기를 늘림에 따라 일관된 성능 향상이 관찰되어, **스케일링에 따른 일반화 성능 향상 가능성**이 확인된다.

### 3.6 전역 수용 영역 (Global Receptive Field)

LAM(Local Attribution Map) 시각화에서 MambaIRv2의 Diffusion Index(DI)가 다른 방법들보다 높게 나타났으며, ERF(Effective Receptive Field) 시각화에서도 전역적 인식 능력이 확인되었다. 이는 새로운 이미지에서도 넓은 컨텍스트를 활용할 수 있음을 의미한다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 방법론 비교표

| 방법 | 연도 | 백본 | 수용 영역 | 복잡도 | 비인과성 |
|------|------|------|----------|--------|---------|
| SwinIR | 2021 | Swin-Transformer | 윈도우 로컬 | $O(HW \cdot w^2)$ | ✅ |
| Restormer | 2022 | Transformer | 채널 전역 | $O(HW \cdot C^2)$ | ✅ |
| CAT-A | 2022 | Transformer | 혼합 | 중간 | ✅ |
| DAT | 2023 | Dual-Attention | 글로벌+로컬 | 중간 | ✅ |
| HAT | 2023 | HAB+Swin | 활성화 확장 | 높음 | ✅ |
| MambaIR | 2024 | Mamba (4방향) | 글로벌 | $O(HW)$ | ❌ |
| ATD | 2024 | Transformer+Dict | 글로벌 | 중간 | ✅ |
| SRFormer | 2023 | Permuted-Attn | 글로벌 윈도우 | 중간 | ✅ |
| **MambaIRv2** | **2025** | **Mamba+ASE+SGN** | **글로벌** | **$O(HW)$** | **✅** |

### 4.2 세대별 발전 흐름

```
CNN 시대 (2014-2020)
SRCNN → EDSR → RCAN → RDN → SAN
[로컬 수용 영역, 제한적 장거리 의존성]
         ↓
Transformer 시대 (2021-2023)
IPT → SwinIR → Restormer → HAT → SRFormer → ATD
[비인과적, 글로벌, but 이차 복잡도]
         ↓
Mamba 시대 (2024-2025)
MambaIR → VmambaIR → MambaIRv2
[선형 복잡도 + 비인과성 도전]
```

### 4.3 주요 방법별 핵심 차별점

**SwinIR (2021)** vs MambaIRv2:
- SwinIR은 이동 윈도우로 제한적 글로벌성만 달성
- MambaIRv2는 단일 스캔으로 완전한 전역 픽셀 참조 가능
- 2× Urban100: SwinIR 33.81 → MambaIRv2-B 34.49 (+0.68 dB)

**HAT (2023)** vs MambaIRv2:
- HAT: 복잡한 하이브리드 어텐션, 높은 연산량(514.9G MACs)
- MambaIRv2-B: 13.4% 적은 MACs로 더 높은 성능
- HAT는 윈도우 어텐션 기반으로 비인과적이나, 연산 비용 큼

**ATD (2024)** vs MambaIRv2:
- 공통점: 프롬프트/토큰 딕셔너리 활용
- 차이점: ATD는 윈도우 어텐션의 제한된 수용 영역 극복이 목적, MambaIRv2는 Mamba의 인과성 극복이 목적
- ATD는 어텐션 맵으로 암묵적 카테고리 결정, MambaIRv2는 명시적 라우팅 모듈 사용
- 2× 클래식 SR Urban100: ATD 34.70 vs MambaIRv2-B 34.49 (ATD 우위) / Manga109: ATD 40.37 vs MambaIRv2-B 40.42 (MambaIRv2 우위)

**MambaIR (2024)** vs MambaIRv2:
- MambaIR: 4방향 스캔, 인과적 한계 존재, 1.36M 파라미터
- MambaIRv2: 단일 방향 스캔, 비인과적, 774K 파라미터
- 파라미터 43% 감소, 연산량 50% 감소, PSNR 0.34dB 향상

---

## 5. 앞으로의 연구에 미치는 영향과 고려 사항

### 5.1 연구에 미치는 영향

#### 5.1.1 Mamba의 비인과화 패러다임 확립
MambaIRv2는 Mamba를 비인과적 태스크에 적용하는 방법론적 프레임워크를 제시했다. **" $\mathbf{C}$ 행렬 = Query"라는 수학적 통찰**은 향후 다양한 비인과적 Mamba 응용의 이론적 기반이 될 것으로 보인다.

#### 5.1.2 프롬프트 기반 전역 정보 통합의 일반화
ASE의 프롬프트 풀 메커니즘은 영상 복원 외에도 세그멘테이션, 객체 탐지 등 다른 비인과적 비전 태스크에 적용 가능한 범용 설계 원칙을 제공한다.

#### 5.1.3 의미론적 시퀀스 재구성의 가능성
SGN이 제시한 "공간적 근접성" 대신 "의미론적 근접성"으로 시퀀스를 구성하는 아이디어는 향후 Mamba 기반 모델의 시퀀스 설계에 영향을 미칠 것으로 예상된다.

#### 5.1.4 효율성-성능 트레이드오프의 새로운 기준
단일 스캔으로 다방향 스캔보다 높은 성능을 달성하면서 연산량을 절반으로 줄인 결과는, 향후 경량 모델 설계에 있어 방향성을 제시한다.

### 5.2 앞으로 연구 시 고려할 점

#### 5.2.1 해석가능성 및 이론적 기반 강화
- Mamba와 ViT가 영상 복원 과정에서 각각 무엇을 학습하는지에 대한 깊은 이론적 분석 필요
- ASE에서 학습된 프롬프트의 의미론적 해석에 대한 추가 연구 필요
- $\overline{\mathbf{A}}^k$ 감쇄가 특정 태스크에서 실제로 얼마나 문제가 되는지 정량화 필요

#### 5.2.2 U-Net 구조와의 결합
- 디노이징 및 고해상도 복원 태스크에서 U-Net이 유리하다는 것이 알려져 있음
- MambaIRv2의 ASSM을 U-Net 형태로 구현하면 추가적 성능 향상 가능성 존재

#### 5.2.3 더 다양한 복원 태스크로의 확장
- 디블러링, 디헤이징, 디레이닝 등에서의 성능 검증 필요
- 의료 영상, 위성 영상 등 특수 도메인에서의 일반화 성능 검증 중요

#### 5.2.4 더 정교한 의미론적 그룹화 전략
- 현재 SGN은 프롬프트 라우팅 결과를 재활용하는 비교적 단순한 방식
- 사전 학습된 세그멘테이션 모델이나 더 정교한 클러스터링 기법 활용 가능성 탐색 필요

#### 5.2.5 Mamba2 및 후속 SSM과의 통합
- 2024년 이후 등장한 Mamba2 등 개선된 SSM과의 결합 가능성
- 선택적 상태 공간 설계와 비인과적 ASE의 시너지 탐색

#### 5.2.6 프롬프트 수 T와 랭크 r의 최적화
- 논문의 Ablation(표 A.2)에서 $T \times r = 32 \times 64$를 최적으로 선택했으나, 이는 태스크 및 데이터셋에 따라 달라질 수 있음
- 적응적 하이퍼파라미터 선택 방법 연구 필요

#### 5.2.7 고해상도 이미지에서의 확장성
- 표 A.1에 따르면 1024×1024에서는 HAT보다 낮은 MACs를 달성하지만, 여전히 절대적 연산량(1823G)이 큰 편
- 슬라이딩 윈도우 방식이나 계층적 처리 방식과의 결합 필요

#### 5.2.8 다중 열화 통합 복원 (All-in-one Restoration)
- 현재 MambaIRv2는 각 태스크별로 별도 학습
- 단일 모델로 다양한 열화 유형을 동시에 처리하는 범용 복원 모델로의 발전 가능성 탐색

---

## 📋 최종 요약

```
MambaIRv2의 핵심 기여
├── 문제: Mamba의 인과적 한계 (미스캔 픽셀 인식 불가, 다방향 스캔 중복, 장거리 감쇄)
├── 해법1: ASE - C 행렬에 프롬프트 주입 → 비인과적 전역 쿼리 + 단일 스캔
├── 해법2: SGN - 의미론적 픽셀 재배열 → 장거리 감쇄 완화
├── 성능: SRFormer 대비 +0.35dB (파라미터 9.3% 감소), HAT 대비 +0.16dB (MACs 13.4% 감소)
└── 의의: Mamba를 비인과적 비전 태스크에 적용하는 최초의 체계적 방법론 제시
```

---

## 📌 참고 자료

- **주요 논문**: Hang Guo et al., "MambaIRv2: Attentive State Space Restoration," arXiv:2411.15269v2, 2025. (본 문서에 첨부된 PDF)
- **MambaIR**: Hang Guo et al., "MambaIR: A Simple Baseline for Image Restoration with State-Space Model," ECCV 2025.
- **Mamba**: Albert Gu and Tri Dao, "Mamba: Linear-time sequence modeling with selective state spaces," arXiv:2312.00752, 2023.
- **SwinIR**: Jingyun Liang et al., "SwinIR: Image Restoration Using Swin Transformer," ICCVW, 2021.
- **HAT**: Xiangyu Chen et al., "Activating More Pixels in Image Super-Resolution Transformer," CVPR 2023.
- **Restormer**: Syed Waqas Zamir et al., "Restormer: Efficient Transformer for High-Resolution Image Restoration," CVPR 2022.
- **SRFormer**: Yupeng Zhou et al., "SRFormer: Permuted Self-Attention for Single Image Super-Resolution," arXiv:2303.09735, 2023.
- **ATD**: Leheng Zhang et al., "Transcending the Limit of Local Window: Advanced Super-Resolution Transformer with Adaptive Token Dictionary," CVPR 2024.
- **Demystify Mamba**: Dongchen Han et al., "Demystify Mamba in Vision: A Linear Attention Perspective," arXiv:2405.16605, 2024.
