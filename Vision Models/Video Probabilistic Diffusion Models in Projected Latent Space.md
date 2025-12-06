
# Video Probabilistic Diffusion Models in Projected Latent Space

## 1. 핵심 주장과 주요 기여

**PVDM(Projected Latent Video Diffusion Model)**은 비디오 생성에서 계산 효율성과 성능을 획기적으로 향상시킨 첫 번째 잠재 확산 모델입니다. 기존 비디오 확산 모델들이 직접 픽셀 공간에서 작동하여 계산 및 메모리 비효율을 겪었던 반면, PVDM은 비디오의 복잡한 3D 구조를 세 개의 2D 이미지 같은 잠재 벡터로 인수분해함으로써 **계산 복잡도를 $O(SHW)$에서 $O(HW + SW + SH)$로 감소**시켰습니다.[1]

주요 기여는 다음과 같습니다:

1. **혁신적인 잠재 공간 설계**: 비디오를 시간축 공통 콘텐츠($z_s$)와 모션 성분($z_h$, $z_w$)으로 분해하여 정보 손실을 최소화하면서도 계산 효율 획기적 개선[1]

2. **효율적인 아키텍처**: 기존 3D CNN 대신 2D 컨볼루션 기반 U-Net 확산 모델을 설계하여 메모리 효율을 3.5배, 계산 효율을 17.6배 향상[1]

3. **장기 비디오 생성**: 단일 모델로 무조건부와 조건부 분포를 결합 학습하는 새로운 기법으로 임의 길이의 연속 비디오 생성 가능[1]

4. **최고 성능**: UCF-101 128프레임 벤치마크에서 FVD 점수를 1773.4에서 639.7로 개선(64% 향상)[1]

***

## 2. 해결하는 문제 및 제안 방법

### 2.1 핵심 문제

기존 비디오 확산 모델의 세 가지 주요 한계:

- **고차원성**: 비디오 데이터는 $(S \times H \times W \times 3)$의 3D RGB 배열로 $O(SHW)$의 계산 복잡도
- **메모리 비효율**: 확산 프로세스가 고차원 입력 공간에서 반복적으로 작동하므로 고해상도, 장시간 비디오 생성 불가능[1]
- **시간적 일관성**: 높은 계산 비용으로 인해 시간적으로 일관된 장기 비디오 생성 어려움

### 2.2 제안하는 방법

#### **Stage 1: 투영 오토인코더(Projected Autoencoder)**

비디오 $x \in \mathbb{R}^{3 \times S \times H \times W}$를 세 개의 2D 잠재 벡터로 변환:

$$u := f^{shw}_{\phi_{shw}}(x), \quad u = [u_{shw}] \in \mathbb{R}^{C \times S \times H' \times W'}$$

각 축에 대한 3D→2D 프로젝션:

$$z^s_{hw} := f^s_{\phi_s}(u^1_{hw}, \ldots, u^S_{hw}), \quad z^s \in \mathbb{R}^{C \times H' \times W'}$$

$$z^h_{sw} := f^h_{\phi_h}(u^s_{1w}, \ldots, u^s_{H'w}), \quad z^h \in \mathbb{R}^{C \times S \times W'}$$

$$z^w_{sh} := f^w_{\phi_w}(u^s_{h1}, \ldots, u^s_{hW'}), \quad z^w \in \mathbb{R}^{C \times S \times H'}$$

**설계 원리**:
- $z_s$: 시간축 공통 요소(배경, 정적 요소) 캡처
- $z_h, z_w$: 공간축 방향 모션 인코딩
- 결과적으로 **잠재 코드 차원이 $O(SHW)$에서 $O(HW + SW + SH)$로 축소**[1]

디코더는 이 세 벡터로부터 3D 잠재를 재구성:

$$\mathbf{v} = (v_{shw}) \in \mathbb{R}^{3C \times S \times H' \times W'}, \quad v_{shw} := [z^h_{hw}, z^s_{hw}, z^w_{sh}]$$

#### **Stage 2: 확산 모델**

기본 확산 프로세스는 Markov 체인으로 정의되며:

$$q(z_t|z_{t-1}) := \mathcal{N}(z_t; \sqrt{1-\beta_t}z_{t-1}, \beta_t I)$$

$$q(z_t|z_0) = \mathcal{N}(z_t; \sqrt{\bar{\alpha}_t}z_0, (1-\bar{\alpha}_t)I), \quad \bar{\alpha}_t := \prod_{i=1}^{t}(1-\beta_i)$$

역 프로세스는 노이즈 예측 목표로 학습:

$$\mathbb{E}_{z_0,\epsilon,t}\left[\|\epsilon - \epsilon_\theta(z_t, t)\|_2^2\right] \quad \text{where } z_t = \sqrt{\bar{\alpha}_t}z_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$$

디코딩 단계에서 생성 프로세스:

$$p_\theta(z_{t-1}|z_t) := \mathcal{N}\left(z_{t-1}; z_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(z_t, t), \sigma_t^2\right)$$

**아키텍처**: 2D 컨볼루션 기반 U-Net으로 각 $z_s, z_h, z_w$를 공유 파라미터로 처리하되, 세 벡터 간 의존성을 주의(attention) 레이어로 모델링[1]

#### **Stage 3: 장기 비디오 생성**

단일 모델로 무조건부와 조건부 분포 결합 학습:

$$\mathbb{E}_{(x^1_0,x^2_0),\epsilon,t}\left[\lambda\|\epsilon - \epsilon_\theta(z^2_t, z^1_0, t)\|_2^2 + (1-\lambda)\|\epsilon - \epsilon_\theta(z^2_t, 0, t)\|_2^2\right]$$

여기서 $z^1_0$는 이전 클립의 잠재, $z^2_t$는 현재 클립의 노이즈 처리 잠재입니다.[1]

생성 시 알고리즘:

$$\text{for } \ell = 1 \text{ to } L:$$
$$\quad \text{Sample } z_T^\ell \sim \mathcal{N}(0, I)$$
$$\quad \text{for } t = T \text{ to } 1:$$
$$\quad\quad \text{if } \ell = 1: \epsilon_t = \epsilon_\theta(z_t^\ell, 0, t)$$
$$\quad\quad \text{else: } \epsilon_t = \epsilon_\theta(z_t^\ell, z_0^{\ell-1}, t)$$
$$\quad\quad z_{t-1}^\ell = \frac{1}{\sqrt{1-\beta_t}}\left(z_t^\ell - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_t\right) + \sigma_t\epsilon$$
$$\quad x^\ell = g_\psi(z_0^\ell)$$

이는 임의 길이의 연속 비디오 생성을 가능하게 합니다.[1]

***

## 3. 모델 구조 상세 분석

### 3.1 Autoencoder 구조

**Encoder 구성**:
- **Video-to-3D**: TimeSformer를 사용하여 비디오를 3D 잠재로 인코딩
- **3D-to-2D Projections**: 4층 Transformer (헤드 4개, 숨김 차원 384)로 각 축별 프로젝션 수행[1]

**Encoder 훈련 목표**:
$$L(\phi, \psi) := L_{pixel}(\phi, \psi) + \lambda_1 L_{LPIPS}(\phi, \psi) + \lambda_2 \max_h L_{GAN}(\phi, \psi)$$

- $L_{pixel}$: ℓ1 재구성 손실
- $L_{LPIPS}$: LPIPS 지각적 유사성 (음수)
- $L_{GAN}$: 판별기 $h$를 포함한 대적 목표
- 하이퍼파라미터: $\lambda_1 = 1$, $\lambda_2 = 0$ (수렴 전) → 0.25 (수렴 후)[1]

**Decoder 구성**:
- 3D 잠재 그리드 $v \in \mathbb{R}^{3C \times S \times H' \times W'}$로부터 재구성
- TimeSformer 기반 디코더

### 3.2 Diffusion 모델 아키텍처

**2D 컨볼루션 기반 U-Net**:
- 기존 3D CNN 대신 2D 컨볼루션으로 계산 효율성 확보
- 각 수준에서 업/다운샘플 잔여 블록 (공유)
- 주의 레이어로 세 잠재 벡터 간 정보 흐름 모델링[1]

**모델 크기**:
- PVDM-S: 베이스 채널 128, 반복 400k
- PVDM-L: 베이스 채널 256, 반복 850k[1]

**계산 복잡도 비교**:

| 방식 | 잠재 코드 차원 | 자기주의 복잡도 |
|-----|--------|---------|
| 기존 3D CNN | $O(SHW)$ | $O((SHW)^2)$ |
| **PVDM** | $O(HW + SW + SH)$ | $O((HW + SW + SH)^2)$ |

이는 구체적으로 다음과 같이 번역됩니다:
- 코드 차원: $32 \times 32 + 16 \times 32 + 16 \times 32 = 1,536$ (PVDM) vs $32 \times 32 \times 16 = 16,384$ (기존)[1]

***

## 4. 성능 향상 및 비교

### 4.1 정량적 성능

**Inception Score (IS) - UCF-101 16프레임**:

| 방법 | IS 점수 |
|-----|-------|
| MoCoGAN | 12.42 |
| ProgressiveVGAN | 14.56 |
| VideoGPT | 24.69 |
| TGANv2 | 28.87 |
| DIGAN | 29.71 |
| VDM | 57.00 |
| TATS | 57.63 |
| **PVDM-L** | **74.40** |

PVDM은 기존 최고 성능(TATS: 57.63)을 **29% 향상**시켰습니다.[1]

**Fréchet Video Distance (FVD) - UCF-101**:

| 방법 | FVD16 | FVD128 |
|-----|--------|---------|
| StyleGAN-V | 1431.0 | 1773.4 |
| **PVDM-S** | 457.4 | 902.2 |
| **PVDM-L** | 398.9 | **639.7** |

128프레임 생성에서 PVDM은 기존 대비 **64% 개선**을 달성했습니다.[1]

**SkyTimelapse 데이터셋**:

| 메트릭 | PVDM-L |
|-------|--------|
| FVD16 | 61.70 |
| FVD128 | 137.2 |

### 4.2 효율성 비교

**메모리 및 계산 효율 (NVIDIA 3090Ti 24GB GPU)**:

| 방법 | 훈련 배치 크기 | 16프레임 생성 시간 | 메모리 | 128프레임 생성 가능 |
|-----|---------|----------|-------|----------|
| VDM | 0 | >113초 | 11.1GB | ✗ |
| TATS | 0 | 84.8초 | 18.7GB | ✗ |
| VideoGPT | 0 | 139초 | 15.2GB | ✗ |
| **PVDM-L** | **2** | **20.4초** | **5.22GB** | **✓** |

PVDM은 계산 효율 **17.6배** 향상을 달성했습니다.[1]

### 4.3 재구성 품질

**Autoencoder 성능**:

| 메트릭 | 훈련 | 테스트 |
|-------|------|-------|
| R-FVD (UCF-101) | 25.87 | 32.26 |
| PSNR (UCF-101) | 27.34 | 26.99 |
| R-FVD (SkyTimelapse) | 7.37 | 36.52 |
| PSNR (SkyTimelapse) | 34.33 | 32.68 |

높은 재구성 성능은 제안된 잠재 표현이 비디오 정보를 효과적으로 보존함을 보여줍니다.[1]

***

## 5. 일반화 성능 향상 가능성

### 5.1 다중 데이터셋 강화 성능

PVDM은 특성이 매우 다른 두 데이터셋에서 동시에 우수한 성능을 달성합니다:

**UCF-101 (복잡한 멀티클래스 행동 인식)**:
- 101개 서로 다른 행동 클래스
- 높은 공간-시간 변동성
- 카메라 움직임과 객체 상호작용

**SkyTimelapse (일관된 시간경과 영상)**:
- 하늘 시간경과 비디오
- 단조로운 모션, 제한된 카메라 움직임
- 일관된 배경과 점진적 변화

두 데이터셋 모두에서 SOTA 달성한 이유:

#### (1) **유연한 잠재 공간의 적응성**

시간축 공통 콘텐츠 분해($z_s$)가 서로 다른 배경과 정적 요소를 효과적으로 학습합니다:
- UCF-101: 복잡한 배경을 $z_s$로 압축 학습
- SkyTimelapse: 일관된 하늘 배경을 $z_s$에 특화

모션 벡터($z_h$, $z_w$)는 다양한 동작 범위에 적응합니다:
- 큰 움직임(UCF-101 액션): 높은 특성 강도
- 작은 움직임(SkyTimelapse 시간경과): 세밀한 변화 포착

#### (2) **2D 프로젝션의 정보 보존**

$$\text{정보 손실} = \text{3D 구조 복잡도} - \text{2D 프로젝션 표현력}$$

공간과 시간을 축 단위로 분해함으로써:
- 각 축의 패턴을 독립적으로 학습
- 중복성을 최소화하면서 정보 보존
- 교차 축 상호작용은 주의 메커니즘으로 모델링[1]

#### (3) **계산 효율성이 가능하게 하는 확대 학습**

$$\text{학습 가능 데이터량} \propto \frac{\text{GPU 메모리}}{\text{계산 복잡도}}$$

PVDM의 효율성으로:
- 더 큰 배치 크기 가능 (7 vs 1)
- 더 높은 해상도 학습 (256×256 vs 128×128)
- 더 긴 시퀀스 처리 (128프레임 vs 불가능)

이를 통해 더 다양한 데이터 샘플에 대한 학습이 가능하여 일반화 성능 향상[1]

### 5.2 도메인 간 전이 가능성

#### **공간 표현의 보편성**

2D 컨볼루션 기반 설계로 이미지 도메인의 강력한 사전(prior) 활용:
- 이미지 생성의 성숙한 기법 (U-Net 아키텍처) 직접 적용 가능
- 이미지 확산 모델의 최적화 기법 재활용 가능
- 새로운 비디오 도메인에 대한 빠른 적응 가능[1]

#### **시간 일관성의 보편성**

시간축 차원 분리($z_s$)가 도메인 특성과 독립적:
- 빠른 모션의 도메인 (스포츠): $z_s$ 역할 최소화, $z_h, z_w$ 강조
- 느린 모션의 도메인 (풍경): $z_s$ 역할 극대화
- 자동으로 데이터 통계에 적응[1]

### 5.3 미검증 도메인 시나리오

제시된 실험에서 성공하지만, 다음 시나리오에서의 성능은 미지수입니다:

1. **극단적 해상도**: 
   - 매우 높은 해상도 (2K, 4K 이상)
   - 극도로 낮은 해상도 (64×64 이하)

2. **극단적 길이**:
   - 매우 긴 비디오 (>1000프레임)
   - 누적 오류의 가능성

3. **도메인 시프트**:
   - 애니메이션 vs 실사
   - 흑백 vs 컬러
   - 실내 vs 실외

4. **조건화 시나리오**:
   - 텍스트 조건부 (논문에서 미지원)
   - 매우 특정한 제어 신호

***

## 6. 모델의 한계

### 6.1 명시적 한계 (논문 저자 언급)

#### (1) **생성 품질의 근본적 한계**

논문에서 언급: *"실제 비디오와 생성 비디오 간 여전히 간격 존재"*

이유:
- 확산 모델의 일반적 한계 (모드 평균화)
- 세밀한 텍스처와 작은 객체 표현 어려움
- 매우 장기 비디오에서 누적 오류[1]

#### (2) **데이터셋 스케일 제한**

실험 규모:
- UCF-101: 9,357개 훈련 비디오 (소규모)
- 최근 모델들(Stable Video Diffusion): 수백만 개 비디오로 훈련

저자 언급: *"대규모 비디오 데이터셋에 대한 실험 미수행"*

이는 일반화 성능의 검증 부족을 의미합니다.[1]

#### (3) **텍스트-투-비디오 미지원**

PVDM은 **무조건부 생성**에만 설계됨:
- 텍스트 조건화 미포함
- 실제 응용의 핵심 기능 부재

저자: *"향후 텍스트-투-비디오로 확장 예상"*[1]

### 6.2 잠재적 기술 한계

#### (1) **프로젝션의 정보 손실**

3D 구조의 2D 축 프로젝션 시:
$$\text{정보 손실} = \text{3D 교차 축 상호작용} - \text{주의 메커니즘 모델링}$$

구체적 예시:
- 복잡한 3D 회전: 2개 평면으로는 완전히 표현 불가
- 광각 카메라의 비선형 왜곡: 축 분해로 손상
- 깊이 관계: 공간 평면에서 손실[1]

#### (2) **압축 비율의 정량화 한계**

계산 복잡도는 감소하지만, 표현력 손실:
$$O(SHW) \to O(HW + SW + SH)$$

예시: 
- 256×256×16 비디오: 1M 차원 → 48K 차원 (97.5% 압축)
- 이 정도 압축에서 미세 디테일 손실 불가피
- 높은 신호 대 잡음비(SNR)를 요구하는 의료/과학 영상에 부적합[1]

#### (3) **시간축 모션 표현의 한계**

$z_h$와 $z_w$로 모션을 표현:
- 시간 축과 공간 축의 완전한 상호작용 포착 불가
- 예: 비디오 길이 S에 따라 모션 표현 능력 변화
- 초고속 움직임과 극저속 변화를 동시에 표현 어려움

#### (4) **장기 비디오의 누적 오류**

조건부 생성으로 임의 길이 비디오 생성하지만:

$$\text{오류}_\ell = \sum_{i=1}^{\ell} \text{오류}_i \quad (\text{누적 오류})$$

- 각 클립 생성 오류가 다음 클립 조건으로 전파
- 수십 개 클립(>1000프레임) 생성 시 누적 오류 심화
- 실제 구현에서 프레임 보간/수정 필요[1]

***

## 7. 앞으로의 연구에 미치는 영향

### 7.1 학술적 영향

#### (1) **새로운 패러다임 제시**

PVDM은 3D 확산 모델이 아닌 **2D 잠재 확산** 접근이 타당함을 입증:
- 이전 가정: 시간축 모델링을 위해 3D CNN 필수
- PVDM의 발견: 효율적인 2D 표현으로 우수한 성능 가능
- **영향**: 후속 연구의 아키텍처 설계에 근본적 변화[1]

#### (2) **인수분해 표현의 중요성**

비디오를 공간-시간으로 인수분해하는 개념의 확대:
- ProAV-DiT (2025): 오디오-비디오 멀티모달 잠재에 유사 개념 적용
- TempoMaster (2025): 계층적 시간 생성에서 유사 분해 활용
- 향후 연구: 더 나은 인수분해 방식 탐색 촉발[1]

#### (3) **효율성-성능 트레이드오프의 새로운 기준**

PVDM 이전: 높은 성능을 위해 계산 비용 감수 필요
PVDM 이후: **효율성과 성능의 동시 달성 가능함 증명**

이는 향후 모든 비디오 생성 연구의 평가 기준 변경:
- 계산 효율을 핵심 평가 지표로 정착
- 개인용 GPU에서 고해상도 비디오 생성의 가능성 제시[1]

### 7.2 기술 발전 방향

#### (1) **Autoencoder 설계의 진화**

PVDM의 Triplane 개념을 넘어서는 가능성:
- **적응형 프로젝션**: 데이터셋 특성에 따라 프로젝션 축 동적 선택
- **계층적 분해**: 다중 수준의 시공간 계층 구조
- **명시적 오토인코더 없음**: 확산 과정 자체에서 학습 (전체-모델 학습)

#### (2) **장기 비디오 생성의 개선**

조건부 생성 기법의 한계 극복:
- **글로벌 일관성 모듈**: 전체 비디오 통일성 보장
- **역방향 확산**: 양방향 모델링으로 누적 오류 감소
- **흐름 기반 예측**: 광학 흐름/잠재 흐름으로 프레임 연결성 강화[1]

#### (3) **멀티모달 조건화의 확장**

현재 무조건부 → 텍스트, 오디오, 제어 신호 등으로 확장:
- LaVie (2023): Cascaded 구조로 텍스트-투-비디오 성공
- Make-A-Video 계열: 텍스트 임베딩 기반 조건화
- 미래 방향: 약한 레이블, 자기 감독 학습 결합[1]

#### (4) **도메인 적응 및 전이 학습**

PVDM의 효율성이 도메인 적응 연구 촉발:
- 새로운 도메인에 대한 빠른 파인튜닝 가능
- 데이터 부족 시나리오에서의 성능 검증 필요
- Few-shot, Zero-shot 비디오 생성 탐색[1]

### 7.3 산업 응용 가능성

#### (1) **실시간 비디오 생성**

PVDM의 계산 효율로:
- 개인용 GPU에서 256×256 고해상도 생성 가능
- 모바일 디바이스로의 확산 가능성
- 인터랙티브 비디오 편집 도구 개발[1]

#### (2) **비디오 인페인팅 및 복원**

조건부 생성 기능 활용:
- 손상된 비디오 프레임 복원
- 비디오 초고해상도 업스케일링
- 프레임 보간 및 중간 프레임 생성[1]

#### (3) **생성적 콘텐츠 제작**

이미지 생성(Stable Diffusion) 성공의 비디오 버전:
- 크리에이티브 산업의 워크플로우 변화
- 영상 VFX 파이프라인 자동화
- 맞춤형 배경/환경 생성[1]

***

## 8. 향후 연구 시 고려할 점

### 8.1 기술적 고려사항

#### (1) **표현 능력 검증**

프로젝션된 2D 잠재 공간의 완전성 증명:
$$\text{복원 오차} = \|x - \text{decode}(\text{encode}(x))\|$$

- 기존 3D 표현 vs 2D 프로젝션의 정보 이론적 비교
- 다양한 비디오 특성(동작 범위, 텍스처, 조명 변화)에 따른 재구성 오류 분석
- 임계값 이상의 복잡한 장면에서의 성능 저하 연구[1]

#### (2) **스케일링 특성 분석**

모델 크기, 데이터 크기에 따른 성능 변화:
$$\text{성능} = f(\text{모델 파라미터 수}, \text{훈련 데이터 크기})$$

- 현재: PVDM-S/L만 검증
- 미래: 초소형(~100M 파라미터)과 초대형(~10B 파라미터) 모델 테스트
- 수렴 곡선: 추가 데이터/파라미터로 얼마나 성능 개선되는가?[1]

#### (3) **도메인 강건성 평가**

다양한 도메인 간 성능 안정성:
- 의료 영상: 극도의 일관성 필요
- 애니메이션: 고도의 추상화 가능
- 실사: 높은 시각적 충실도 필요
- 각 도메인에서의 일반화 성능 정량화[1]

#### (4) **조건화 메커니즘 확장**

무조건부 → 다중 조건화:
- 텍스트 조건화: 언어 이해 능력 통합
- 제어 신호: 카메라 경로, 객체 궤적 등 명시적 제어
- 약한 지도: 태그, 카테고리 정보 활용
- 모달리티 융합: 텍스트+오디오+스케치 결합[1]

### 8.2 이론적 고려사항

#### (1) **일반화 이론**

확산 모델의 일반화 경계:
$$\text{일반화 오류} \leq \text{훈련 오류} + \text{복잡도 페널티} + \text{표현 오류}$$

- PVDM의 2D 프로젝션이 표현 오류에 어떻게 영향하는가?
- 모델 용량과 데이터 크기의 최적 비율은?
- 과적합 위험은 기존 3D 모델과 비교해 어떠한가?[1]

#### (2) **정보 이론적 분석**

시공간 인수분해의 정보량:
$$I(z_s, z_h, z_w) = I(x) + \text{손실}$$

- 세 벡터의 상호 정보량(Mutual Information) 분석
- 각 벡터가 비디오 정보의 몇 %를 보존하는가?
- 주의 메커니즘이 복구하는 교차 축 정보량 정량화[1]

#### (3) **계산 복잡도의 한계**

현재 복잡도 분석:
$$O(HW + SW + SH) \text{ vs } O(SHW)$$

이 외에 고려해야 할 요소:
- 프로젝션 연산의 숨겨진 복잡도
- 주의 레이어의 실제 계산 시간 (이론적 복잡도 vs 실측)
- 배치 병렬화의 효율성
- 메모리 접근 패턴(캐시 효율)[1]

### 8.3 실증적 고려사항

#### (1) **인간 평가**

정량 메트릭 외 인간 선호도:
- 비디오 품질 MOS (Mean Opinion Score)
- 시간 일관성 평가
- 아티팩트(깜빡임, 테일러링 등) 감지
- 현실성 평가[1]

#### (2) **속성 분석 (Ablation Study)**

제안 요소의 중요도:
- 3D→2D 프로젝션 vs 기존 3D 잠재 (각 축별 기여도)
- 공유 U-Net vs 별도 U-Net (파라미터 효율 vs 성능)
- 주의 레이어의 필요성 (제거 시 성능 저하)
- Null-frame 트릭의 효과성[1]

#### (3) **음성 사례 분석**

PVDM이 실패하는 상황:
- 극도로 복잡한 비디오 (많은 객체, 빠른 모션)
- 매우 긴 비디오에서의 누적 오류 시각화
- 도메인 시프트 시나리오에서의 성능 붕괴
- 개선 방안 도출[1]

### 8.4 실제 구현 시 고려사항

#### (1) **메모리 최적화**

- 그래디언트 체크포인팅으로 훈련 메모리 감소
- 양자화로 추론 메모리 감소
- 프로그레시브 생성으로 장기 비디오 메모리 감소

#### (2) **추론 가속**

- DDIM 샘플링 단계 수 최적화
- 조기 종료 (Early Stopping) 기법
- 배치 동적 구성으로 최대 처리량 달성

#### (3) **배포 시스템**

- 클라우드 기반 API 설계
- 엣지 디바이스 최적화
- 실시간 스트리밍 생성 아키텍처[1]

***

## 9. 결론

PVDM은 비디오 생성 분야에서 **효율성과 성능의 새로운 패러다임**을 제시했습니다. 기본 가정의 전환(3D CNN 필수 → 효율적 2D 표현으로 충분)을 통해 개인용 GPU에서도 고해상도, 시간적으로 일관된 장기 비디오 생성을 가능하게 했습니다.[1]

다만, **텍스트-투-비디오 미지원**, **대규모 데이터셋 미검증**, **극단적 장기 비디오의 누적 오류** 등의 한계도 명확합니다. 그럼에도 불구하고 CVPR 2023에서 256회 이상 인용된 높은 영향도는 이 연구가 비디오 생성 분야의 향후 발전 방향을 크게 바꾸었음을 입증합니다.[1]

향후 연구는 다음 방향으로 진행될 것으로 예상됩니다: **(1) 멀티모달 조건화 통합**, **(2) 초대형 모델 확장**, **(3) 도메인 적응 강화**, **(4) 실시간 생성 기술**. 이러한 발전을 통해 비디오 생성은 이미지 생성처럼 대중적 도구로 자리잡을 것입니다.

***

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/39065612-7ff4-4d33-824a-0b5985000c25/2302.07685v2.pdf)
[2](https://arxiv.org/abs/2310.07771)
[3](https://link.springer.com/10.1007/s11263-024-02295-1)
[4](https://link.springer.com/10.1007/s11263-024-02271-9)
[5](https://arxiv.org/abs/2304.08477)
[6](https://arxiv.org/abs/2309.00398)
[7](https://ieeexplore.ieee.org/document/10203078/)
[8](https://arxiv.org/abs/2211.11018)
[9](https://arxiv.org/abs/2311.15127)
[10](https://ieeexplore.ieee.org/document/10204290/)
[11](https://arxiv.org/abs/2306.17203)
[12](https://arxiv.org/abs/2501.00103)
[13](https://arxiv.org/pdf/2309.15103.pdf)
[14](https://arxiv.org/pdf/2311.11325.pdf)
[15](http://arxiv.org/pdf/2410.01594.pdf)
[16](http://arxiv.org/pdf/2408.12590.pdf)
[17](https://arxiv.org/html/2412.09551v1)
[18](https://arxiv.org/abs/2304.11603)
[19](https://lilianweng.github.io/posts/2024-04-12-diffusion-video/)
[20](https://milvus.io/ai-quick-reference/what-techniques-help-improve-the-generalization-of-diffusion-models)
[21](https://arxiv.org/html/2511.12072v1)
[22](https://www.emergentmind.com/topics/latent-video-diffusion-model)
[23](https://openaccess.thecvf.com/content/ICCV2025/papers/He_Boosting_Domain_Generalized_and_Adaptive_Detection_with_Diffusion_Models_Fitness_ICCV_2025_paper.pdf)
[24](https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_Video_Probabilistic_Diffusion_Models_in_Projected_Latent_Space_CVPR_2023_paper.pdf)
[25](https://openaccess.thecvf.com/content/CVPR2023/papers/Blattmann_Align_Your_Latents_High-Resolution_Video_Synthesis_With_Latent_Diffusion_Models_CVPR_2023_paper.pdf)
[26](https://www.siam.org/publications/siam-news/articles/generalization-of-diffusion-models-principles-theory-and-implications/)
[27](https://scholarworks.sjsu.edu/cgi/viewcontent.cgi?article=1922&context=etd_projects)
[28](https://www.marvik.ai/blog/diffusion-models-for-video-generation)
[29](https://arxiv.org/abs/2312.11752)
[30](https://arxiv.org/abs/2502.00336)
[31](https://www.semanticscholar.org/paper/8818cf3b27cc65acca184f6ae070255d378067e4)
[32](https://arxiv.org/abs/2206.08265)
[33](https://www.semanticscholar.org/paper/2e758828b32dcad917c48f08fc3ec651f05b6edd)
[34](https://ieeexplore.ieee.org/document/10592434/)
[35](https://arxiv.org/abs/2402.06121)
[36](https://www.semanticscholar.org/paper/aecbe351822b77cb36d22e8a43b4fe2bda6ab998)
[37](https://arxiv.org/abs/2209.12104)
[38](https://www.semanticscholar.org/paper/c8c5ba8ce06bb1feca1977751bb5ec0afd692296)
[39](https://arxiv.org/pdf/2211.03595.pdf)
[40](https://academic.oup.com/jrsssb/advance-article-pdf/doi/10.1093/jrsssb/qkae005/56173808/qkae005.pdf)
[41](http://arxiv.org/pdf/2002.00107.pdf)
[42](https://arxiv.org/pdf/2502.03435.pdf)
[43](https://arxiv.org/pdf/2302.07400.pdf)
[44](https://arxiv.org/pdf/2401.15604.pdf)
[45](https://arxiv.org/abs/2407.07998)
[46](https://gwern.net/doc/ai/nn/diffusion/2011-vincent.pdf)
[47](https://www.sciencedirect.com/science/article/abs/pii/S0097849325002250)
[48](https://sander.ai/2022/01/31/diffusion.html)
[49](https://nvlabs.github.io/eg3d/)
[50](https://arxiv.org/html/2504.12027v1)
[51](https://velog.io/@yhyj1001/Generative-Modeling-by-Estimating-Gradients-of-the-Data-Distribution-1-A-Connection-Between-Score-Matching-Denoising-Autoencoders)
[52](https://arxiv.org/abs/2505.16535)
[53](https://arxiv.org/abs/2312.00845)
[54](https://blog.si-analytics.ai/49)
[55](https://www.mdpi.com/2078-2489/16/11/990)
[56](https://arxiv.org/abs/2509.11165)
[57](https://arxiv.org/pdf/1912.03716.pdf)
[58](https://arxiv.org/pdf/1710.06924.pdf)
[59](https://arxiv.org/html/2411.12832)
[60](https://arxiv.org/pdf/2203.08321.pdf)
[61](https://arxiv.org/pdf/2403.02714.pdf)
[62](https://arxiv.org/pdf/2303.10452.pdf)
[63](https://arxiv.org/pdf/2109.00522.pdf)
[64](https://arxiv.org/html/2501.18592)
[65](https://www.sciencedirect.com/science/article/abs/pii/S0031320325007198)
[66](https://www.nature.com/articles/s41598-025-85602-1)
[67](https://papers.neurips.cc/paper_files/paper/2022/file/944618542d80a63bbec16dfbd2bd689a-Paper-Conference.pdf)
[68](https://arxiv.org/html/2505.24346v1)
[69](https://arxiv.org/html/2511.12578v2)
[70](https://openaccess.thecvf.com/content/CVPR2023/papers/Ni_Conditional_Image-to-Video_Generation_With_Latent_Flow_Diffusion_Models_CVPR_2023_paper.pdf)
[71](https://dl.acm.org/doi/full/10.1145/3679010)
[72](https://www.guoyongcs.com/publication/turbo2k/)
[73](https://arxiv.org/abs/2507.15269)
[74](https://www.nature.com/articles/s41597-024-03951-4)
