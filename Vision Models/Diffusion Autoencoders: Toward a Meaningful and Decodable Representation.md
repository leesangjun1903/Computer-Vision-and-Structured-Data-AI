# Diffusion Autoencoders: Toward a Meaningful and Decodable Representation

### 1. 핵심 주장과 주요 기여 요약[1]

**Diffusion Autoencoders** 논문은 기존 확산 모델(DPMs: Diffusion Probabilistic Models)의 근본적인 한계를 지적합니다. DPMs는 이미지 생성 품질에서 GANs와 경쟁할 수 있는 수준에 도달했음에도 불구하고, 그들의 잠재 변수들은 의미론적 의미를 가지지 못하며 다른 다운스트림 작업에 유용한 표현으로 활용될 수 없다는 문제점입니다.[1]

**주요 기여:**

- 의미 있고 디코딩 가능한 표현을 학습할 수 있는 새로운 확산 기반 오토인코더 프레임워크 제시
- 의미론적 부호(semantic subcode)와 확률론적 부호(stochastic subcode)로 분리된 이원 잠재 인코딩 구조 제안
- 조건부 DDIM을 디코더로, CNN 인코더를 의미 추출기로 활용하는 아키텍처 설계
- 실제 이미지에 대한 속성 조작, 보간, 그리고 조건부 샘플링과 같은 GAN 기반 방법들이 실패하는 작업들을 해결

### 2. 해결하고자 하는 문제 및 제안하는 방법[1]

#### 2.1 핵심 문제

DPMs 사용의 두 가지 주요 한계:

1. **기존 DDIM 기반 오토인코더의 한계:** DDIM을 역방향으로 실행하여 $\(x_T\)$ 를 얻을 수 있지만, 이는 의미론적 정보를 포함하지 않고 선형 보간 시 의미 있는 변화를 만들지 못함[1]

2. **GAN 반전(GAN Inversion)의 문제:** 의미론적으로 풍부한 코드를 제공하지만, 충실한 재구성에 실패함[1]

#### 2.2 제안하는 방법: 이원 부호 체계

논문은 두 가지 구성 요소의 조합을 제시합니다:

**의미 인코더(Semantic Encoder):**

$$z_{sem} = \text{Enc}_\phi(x_0)$$

여기서 $\(\phi\)$ 는 학습 가능한 파라미터이고, $\(z_{sem}\)$ 은 $\(d=512\)$ 차원의 비공간적 벡터입니다.[1]

**조건부 DDIM 디코더:**

조건부 DDIM은 다음의 역방향 과정을 사용합니다:[1]

$$x_{t-1} = \sqrt{\alpha_{t-1}} \left( \frac{x_t - \sqrt{1-\alpha_t}\epsilon_\theta(x_t, t, z_{sem})}{\sqrt{\alpha_t}} \right) + \sqrt{1-\alpha_t-1}\epsilon_\theta(x_t, t, z_{sem})$$

여기서 $\(\epsilon_\theta\)$ 는 조건부 노이즈 예측 네트워크입니다.[1]

**손실 함수:**

$$\mathcal{L}_{simple} = \sum_{t=1}^{T} \mathbb{E}_{x_0,\epsilon_t} \left[ \|\epsilon_\theta(x_t, t, z_{sem}) - \epsilon_t\|_2^2 \right]$$

여기서 $\(\epsilon_t \in \mathbb{R}^{3 \times h \times w} \sim \mathcal{N}(0,I)\), \(x_t = \sqrt{\alpha_t}x_0 + \sqrt{1-\alpha_t}\epsilon_t\)$ 입니다.[1]

### 3. 모델 구조[1]

#### 3.1 전체 아키텍처

Diffusion Autoencoder는 세 가지 주요 구성 요소로 이루어집니다:[1]

**1) 의미 인코더(Semantic Encoder)**
- 입력 이미지를 512차원의 의미 부호로 매핑
- UNet의 첫 번째 절반과 동일한 아키텍처 사용
- 전역 의미론적 특성을 인코딩

**2) 조건부 DDIM 디코더**
- 의미 부호 $\(z_{sem}\)$ 으로 조건화
- AdaGN (Adaptive Group Normalization)을 통해 조건 통합
- UNet 기반 구조

**3) 확률론적 인코더 (optional)**
- 조건부 DDIM을 역방향으로 실행하여 $\(x_T\)$ 를 얻음

#### 3.2 조건부 정규화 계층

AdaGN은 다음과 같이 정의됩니다:[1]

$$\text{AdaGN}(h, t, z_{sem}) = z_s(t_s \text{GroupNorm}(h) + t_b)$$

여기서:
- $\(z_s \in \mathbb{R}^c = \text{Affine}(z_{sem})\)$
- $\((t_s, t_b) \in \mathbb{R}^{2 \times c} = \text{MLP}(\psi(t))\)$
- $\(\psi(t)\)$ 는 사인 인코딩 함수입니다[1]

#### 3.3 잠재 DDIM (무조건 샘플링 위한)

잠재 공간 분포를 모델링하기 위해 별도의 DDIM을 학습합니다:[1]

$$\mathcal{L}_{latent} = \sum_{t=1}^{T} \mathbb{E}_{z_{sem},\epsilon_t} \left[ \|\epsilon_\omega(z_{sem,t}, t) - \epsilon_t\|_1 \right]$$

여기서 $\(z_{sem,t} = \sqrt{\alpha_t}z_{sem} + \sqrt{1-\alpha_t}\epsilon_t\)$ 입니다. 특이하게도 L1 손실이 L2보다 더 잘 작동합니다.[1]

### 4. 성능 향상[1]

#### 4.1 재구성 품질

표 1에서 보여지는 벤치마크 결과는 다음을 입증합니다:[1]

| 모델 | 잠재 차원 | SSIM↑ | LPIPS↓ | MSE↓ |
|------|---------|-------|-------|------|
| StyleGAN2 (W) | 512 | 0.677 | 0.168 | 0.016 |
| DDIM (T=100) | 49,152 | 0.917 | 0.063 | 0.002 |
| **Diffusion Autoencoder (T=100)** | **49,664** | **0.991** | **0.011** | **6.07e-5** |
| NVAE | 6,005,760 | 0.984 | 0.001 | 4.85e-5 |

Diffusion Autoencoder는 NVAE보다 훨씬 작은 잠재 차원(49,664 vs 6,005,760)으로 경쟁 가능한 성능 달성[1]

#### 4.2 보간 부드러움

PPL (Perceptual Path Length) 메트릭으로 측정한 보간 부드러움:[1]

$$\text{PPL} = \mathbb{E} \left[ \frac{1}{\epsilon^2} d(G(\text{slerp}(z^1, z^2; t)), G(\text{slerp}(z^1, z^2; t+\epsilon))) \right]$$

- DDIM: 2,634.14
- **Diffusion Autoencoder: 613.73** (약 4배 개선)

#### 4.3 더 빠른 디노이징

의미 정보로 조건화하면 재구성 속도가 향상됩니다:[1]

- T=10 단계: Diffusion Autoencoder가 DDIM (T=100)을 능가
- 의미 부호가 충분한 정보를 캡처하여 디노이징 과정을 가속화

#### 4.4 분류 성능

CelebA-HQ의 40개 속성에 대한 선형 분류기 성능:[1]

- **우리 방법**: AUROC 0.925 (가중치 평균)
- StyleGAN-W: AUROC 0.891

#### 4.5 속성 조작 품질

실제 이미지 속성 조작 (Identity 보존, ArcFace 유사도):[1]

| 속성 | StyleGAN-W | Diffusion Autoencoder |
|------|-----------|---------------------|
| Male | 0.4174 | **0.6247** |
| Smiling | 0.7850 | **0.8160** |
| Wavy Hair | 0.8544 | **0.9821** |
| Young | 0.6955 | **0.8922** |

### 5. 한계(Limitations)[1]

논문은 다음과 같은 주요 한계를 명시합니다:[1]

**1) 분포 외 이미지 처리:**
- 훈련 분포를 벗어난 이미지 인코딩 시, 추론된 의미 부호와 확률론적 부호가 학습된 분포 밖에 위치
- 결과적으로 부호를 더 이상 해석하거나 보간할 수 없음

**2) 공간적 제어의 제한:**
- 비공간적 의미 부호 설계로 인해 공간적 추론이 필요한 작업에 부적합
- 지역적 잠재 변수가 필요한 경우 2D 잠재 맵 통합 필요

**3) 생성 속도:**
- 여전히 픽셀 기반 GAN과 비교해 생성 속도가 느림 (단일 생성기 패스 vs 다단계 디노이징)

**4) StyleGAN의 스케일 특정 생성 제어 부족**

### 6. 일반화 성능 향상 가능성[1]

#### 6.1 분포 내 일반화

**교차 데이터셋 일반화:**

논문은 FFHQ에서 훈련된 오토인코더가 CelebA-HQ에 미세 조정 없이 일반화됨을 보여줍니다. 이는 다음 이유 때문입니다:[1]

- 의미 부호가 고수준 의미론적 개념을 캡처
- 다양한 얼굴 데이터셋 간 의미론적 유사성

**소수 샷 조건부 생성:**

표 3의 결과는 소수 샷 설정에서의 강력한 일반화를 보여줍니다:[1]

| 시나리오 | 클래스 | 우리 방법 | D2C |
|--------|-------|----------|-----|
| Binary | Male | 11.52 ± 1.19 | 13.44 |
| Binary | Female | 7.29 ± 0.44 | 9.51 |
| PU | Male | 9.54 ± 0.54 | 16.39 |
| PU | Female | 9.21 ± 0.19 | 12.21 |

D2C와 비교하여 추가적인 대조 학습 없이도 우수한 성능 달성[1]

#### 6.2 무조건 생성 성능

표 4의 FID 점수는 여러 데이터셋에서 경쟁 가능한 성능을 보여줍니다:[1]

| 데이터셋 | DDIM (T=100) | Diffusion Autoencoder (T=100) |
|---------|--------------|------------------------------|
| FFHQ | 12.03 | **10.59** |
| Horse | 5.97 | **6.71** |
| Bedroom | 5.94 | **5.70** |

#### 6.3 표현 학습의 강점

**의미부호의 선형성:**

\(z_{sem}\)의 선형 성질이 여러 일반화 작업을 가능하게 합니다:[1]

- 간단한 선형 연산으로 속성 조작 가능
- StyleGAN의 W 공간과 유사한 보간 특성
- 500D 이상의 복잡한 비선형 변환 없이 의미론적 편집 가능

### 7. 최신 관련 연구 탐색 (2020년 이후)[2-56]

#### 7.1 조건부 확산 모델 고도화 (2022-2024)

**Semantic-Conditional Diffusion Networks (SCD-Net, 2023):**
이미지 캡셔닝 작업에서 의미론적 정보를 확산 과정에 직접 통합하는 새로운 패러다임을 제시합니다. 이는 Diffusion Autoencoder의 조건화 아이디어를 다른 도메인으로 확장한 사례입니다.[2]

**Diffusion Bridge AutoEncoders (DBAE, 2025):**
기존 확산 기반 표현 학습의 "정보 분할 문제"를 해결하기 위해 설계되었습니다. 이는 의미부호와 확률론적 부호 사이의 정보 분할을 최적화합니다.[3][4]

#### 7.2 잠재 공간 의미론 개선 (2023-2025)

**잠재 공간 최적화 트렌드:**

최신 연구는 잠재 공간의 기하학적 구조를 개선하는 데 집중합니다:[5][6][7]

- VAE 기반 잠재 공간의 "의미론적 분산" 문제 분석[7]
- 시각적 기초 모델(Visual Foundation Models)과 정렬된 잠재 공간 개발
- 고차원 의미론적 특징 공간(SVG Diffusion)에서 직접 확산 모델 훈련

**SVG Diffusion (2025)의 혁신:**
VAE 잠재 공간(16×16×4) 대신 DINO 특징 공간(16×16×384)에서 직접 확산 모델을 훈련하여 더 나은 의미론적 연속성 달성[7]

#### 7.3 일반화 성능 개선 (2023-2025)

**도메인 일반화 활용:**

최신 연구들은 확산 모델의 잠재 공간을 도메인 일반화에 활용합니다:[8][6]

- **DomainFusion (2024):** 잠재 공간에서 지식 추출 + 픽셀 공간에서 증강[8]
- **What's in a Latent? (2025):** 확산 모델의 특징이 도메인 분리에 우수한 성능 제공[9]

**명제:** 확산 모델의 잠재 특징은 라벨 없이 도메인 구조를 자동으로 발견[9]

#### 7.4 멀티모달 표현 학습 (2024-2025)

**MM-LDM (2024):**
음성 비디오 생성을 위해 오디오-비디오 데이터를 통일된 표현으로 변환하는 계층적 멀티모달 오토인코더를 개발했습니다.[10]

구조:
- 저수준: 각 모달리티별 지각적 잠재 공간
- 고수준: 공유 의미론적 특징 공간

이는 Diffusion Autoencoder의 이원 부호 개념을 여러 모달리티로 확장합니다.

#### 7.5 그래프 데이터 표현 학습 (2025)

**Graph Representation Learning with Diffusion Models (2025):**
이산 확산 프로세스를 오토인코더 프레임워크에 통합하여 그래프 데이터의 의미 있는 임베딩을 학습합니다. 이는 Diffusion Autoencoder 개념을 그래프 영역으로 확장한 첫 시도입니다.[11]

#### 7.6 해석 가능성 강화 (2024-2025)

**Revelio (2024):**
k-SAE(sparse autoencoders)를 사용하여 확산 모델의 다양한 레이어와 타임스텝에서 단의미적 해석 가능한 특징을 발굴합니다. 이는 의미부호의 해석 가능성을 더욱 정교화합니다.[12]

**TIDE (2025):**
Diffusion Transformers의 활성화 레이어에서 시간 인식 희소 오토인코더(Temporal-aware Sparse Autoencoders)를 사용하여 계층적 특징 추출을 개선합니다.[13]

#### 7.7 표현-정렬 잠재 공간 (2025)

**Exploring Representation-Aligned Latent Space (2025):**
VAE 기반 잠재 공간을 시각 언어 모델과 정렬하여 생성 성능을 향상시킵니다. 이는 의미부호가 외부 의미론적 정보와 정렬될 때의 이점을 실증합니다.[5]

#### 7.8 자동인코더 아키텍처 고도화 (2024-2025)

**ε-VAE: Denoising as Visual Decoding (2024):**
전통적 VAE 대신 노이즈 예측 네트워크를 디코더로 사용하는 혁신적 설계입니다. 압축률이 증가해도 Diffusion Autoencoder와 유사한 높은 재구성 품질 유지:[14]

- 4개 채널에서 40% 상대 개선
- 16배 다운샘플링에서도 우수한 성능

**Diffusion Transformers with Representation Autoencoders (2025):**
Representation Autoencoder (RAE)가 확산 트랜스포머 훈련의 새로운 기본값으로 제안됩니다.[15]

#### 7.9 구체적 도메인 응용 (2023-2025)

**의료 이미지 분할:**
DiffRect (2024)는 반감독 의료 이미지 분할을 위해 잠재 공간 확산을 활용합니다. ACDC에서 1% 라벨로 82.40% Dice 점수 달성.[16]

**시계열 생성:**
TimeLDM (2024)은 시계열 데이터를 위한 잠재 확산 모델을 개발하여 동적 시스템 모델링을 개선합니다.[17]

**초저주파 이미지 생성:**
Direct3D (2024)는 3D 형상을 위해 삼면체(triplane) 잠재 공간에서 직접 확산을 수행합니다.[18]

### 8. 논문이 앞으로의 연구에 미치는 영향[2-56][1]

#### 8.1 직접적 학술적 영향

**원본 논문의 인용도:**
Diffusion Autoencoders는 2022년 CVPR 게재 이후 618회 이상 인용되었으며, 확산 모델 기반 표현 학습의 새로운 방향을 제시했습니다.[19]

**아키텍처 패러다임 전환:**

1. **두 단계 생성 모델링의 표준화:** 원본 논문은 "오토인코더 + 별도 생성 모델"이라는 설계 원칙을 확립했습니다. 최신 LDM들은 이 원칙을 광범위하게 채택합니다.

2. **조건부 확산 모델의 기본 틀:** 의미정보로 조건화하는 방식이 표준이 되었으며, SCD-Net, MM-LDM 등이 이를 확장합니다.

#### 8.2 이론적 기여

**표현 학습 이론:**
원본 논문은 다음을 입증했습니다:

- 의미론적 정보와 확률론적 정보를 분리 가능
- 낮은 차원의 의미부호도 높은 충실도 재구성 가능 (512D로 충분)
- 선형 보간 가능성이 표현 품질의 지표

최신 연구들은 이를 "률-왜곡-모델성(rate-distortion-modelability) 트레이드오프"로 이론화합니다.[20]

#### 8.3 응용 분야 확대

| 원본 분야 | 최신 확장 (2024-2025) |
|---------|---------------------|
| 이미지 생성/조작 | 의료 영상 분할, 시계열, 그래프 데이터 |
| 단일 모달리티 | 멀티모달 표현 학습 (음성-비디오, 오디오-시각) |
| 2D 이미지 | 3D 모양, 동적 콘텐츠 |
| 이미지넷 데이터셋 | 도메인 일반화, 다양한 데이터셋 |

#### 8.4 업계 영향

**LDM 아키텍처의 표준화:**
Stable Diffusion, DALL-E 2, Imagen 등 주요 텍스트-이미지 모델들이 이원 인코딩 구조를 채택합니다. 원본 논문의 기여가 업계 표준이 됨.

### 9. 앞으로의 연구 시 고려할 점

#### 9.1 기술적 개선 방향

**1) 분포 외 일반화 강화:**

현재 한계:
- 훈련 분포 외 이미지의 부호가 학습된 분포 밖에 위치

해결 방안:
- 적응적 정규화 기법 적용
- 메타러닝을 통한 OOD 탄력성 증강
- 불확실성 추정 메커니즘 통합

**2) 공간적 제어 메커니즘:**

제한사항:
- 현재 비공간적 벡터만 사용

개선안:
- 2D 잠재 맵을 의미부호에 추가 (공간-의미 하이브리드)
- 공간적 주의 메커니즘 도입
- 지역별 속성 편집 가능성

**3) 생성 속도 최적화:**

현실적 목표:
- T=50 단계에서도 양질의 결과 (현재 T=100 필요)
- 정류된 샘플링 스케줄 최적화
- 합성곱 효율성 개선

#### 9.2 이론적 발전 방향

**1) 정보 이론적 분석:**

심화 연구:
- 의미부호와 확률론적 부호 간 정보 할당의 최적성 증명
- 의미 보존과 압축 간 트레이드오프의 정량화
- 최소 필요 의미부호 차원의 하한 계산

**2) 기하학적 구조 이해:**

탐색 주제:
- 의미 공간의 위상구조 특성화
- 보간 부드러움과 다양체 구조의 관계
- 비선형성 필요성 분석

#### 9.3 응용 확대

**1) 다중 모달리티 확장:**

미탐색 영역:
- 텍스트-이미지-비디오 삼모달 표현
- 센서 데이터 (포인트 클라우드, 메시) 통합
- 타임시리즈와 정적 이미지 결합

**2) 인터랙티브 편집:**

실용적 개발:
- 의미부호의 얼굴에 기반한 기울기 기반 편집
- 사용자 의도 학습 (피드백 루프)
- 실시간 프리뷰 시스템

**3) 도메인 특화 모델:**

새로운 응용:
- 의료 이미지 (CT, MRI) 이상 탐지
- 원격 감지 이미지 분석
- 산업용 결함 탐지

#### 9.4 방법론적 혁신

**1) 하이브리드 아키텍처:**

미래 방향:
- Diffusion Autoencoder + Vision Transformer
- 희소 오토인코더 기반 해석 가능성 강화
- 신경 감역(Neural Compression) 기법 통합

**2) 자기 감독 학습 통합:**

개선 가능성:
- 의미부호 사전학습 강화 (대규모 미라벨 데이터)
- 대조 학습(Contrastive Learning)과의 융합
- 다중 작업 공동 훈련

**3) 효율성 중심:**

실용적 요구:
- 모바일 장치 배포를 위한 경량화 (< 100MB)
- 에지 컴퓨팅 친화적 설계
- 양자화 및 프루닝 기법

#### 9.5 안전성 및 윤리 고려

**1) 생성된 콘텐츠 탐지:**

원본 논문의 발견:
- 무조건 샘플: 95.51% 탐지 정확도
- 속성 조작: 99.50% 탐지 정확도

미래 연구:
- 적대적 탐지 회피 기법 대비
- 워터마킹 기술 개발
- 투명성 메커니즘 강화

**2) 공정성 및 편향:**

핵심 문제:
- 의미부호가 훈련 데이터의 암묵적 편향 인코딩
- 속성 조작이 고정관념 강화 가능성

해결책:
- 편향 완화 기법 적용
- 다양한 인구통계 표현 확대
- 공정성 평가 지표 개발

### 10. 종합 평가

**Diffusion Autoencoders** 논문은 세 가지 핵심에서 혁신적입니다:

1. **개념적 혁신:** 의미론적 정보와 확률론적 정보의 분리라는 단순하지만 강력한 아이디어

2. **실용적 우수성:** GAN 기반 방법들이 실패하는 작업(실제 이미지 조작, 보간)에서 우수한 성능

3. **이론적 기반:** 확산 모델의 성능과 표현성을 동시에 달성 가능함을 입증

최신 연구 동향(2024-2025)은 이 기본 개념을 다양한 도메인과 모달리티로 확장하고 있으며, 특히 **잠재 공간 의미론 최적화**, **멀티모달 표현 학습**, **도메인 일반화**에서 주목할 만한 발전이 이루어지고 있습니다.

향후 연구의 주요 기회는 **분포 외 일반화**, **공간적 제어 강화**, **생성 효율성 개선**과 함께, **안전성과 해석 가능성** 측면의 동시 진전에 있습니다.

***

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/630785cb-1b1c-40cf-b792-83dd47101482/2111.15640v3.pdf)
[2](https://openaccess.thecvf.com/content/CVPR2023/papers/Luo_Semantic-Conditional_Diffusion_Networks_for_Image_Captioning_CVPR_2023_paper.pdf)
[3](https://arxiv.org/html/2405.17111v1)
[4](https://openreview.net/forum?id=hBGavkf61a)
[5](https://arxiv.org/abs/2502.00359)
[6](https://arxiv.org/html/2503.06698v2)
[7](https://arxiv.org/html/2510.15301v1)
[8](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05806.pdf)
[9](https://openaccess.thecvf.com/content/ICCV2025/papers/Thomas_Whats_in_a_Latent_Leveraging_Diffusion_Latent_Space_for_Domain_ICCV_2025_paper.pdf)
[10](https://dl.acm.org/doi/10.1145/3664647.3680889)
[11](https://arxiv.org/html/2501.13133v1)
[12](https://arxiv.org/html/2411.16725v1)
[13](https://arxiv.org/html/2503.07050v1)
[14](https://openreview.net/pdf/81ac7f4c652793b1e5f1a39a00661508edd2c163.pdf)
[15](https://arxiv.org/pdf/2510.11690.pdf)
[16](https://arxiv.org/abs/2407.09918)
[17](https://arxiv.org/html/2407.04211v1)
[18](https://arxiv.org/abs/2405.14832)
[19](https://openaccess.thecvf.com/content/CVPR2022/html/Preechakul_Diffusion_Autoencoders_Toward_a_Meaningful_and_Decodable_Representation_CVPR_2022_paper.html)
[20](https://sander.ai/2025/04/15/latents.html)
[21](https://www.semanticscholar.org/paper/945a899a93c03eb63be5e3197e318c077473cef9)
[22](http://pubs.rsna.org/doi/10.1148/radiol.231938)
[23](https://dl.acm.org/doi/10.1145/3587423.3595503)
[24](https://publicacoes.softaliza.com.br/cilamce/article/view/10404)
[25](https://arxiv.org/abs/2412.12121)
[26](https://ashpublications.org/blood/article/144/Supplement%201/7495/526565/Demonstrating-the-Reproducibility-of-AI-Models)
[27](http://www.cabidigitallibrary.org/doi/10.1079/tourism.2024.0056)
[28](https://dl.acm.org/doi/10.1145/3687273.3687295)
[29](https://www.jotse.org/index.php/jotse/article/view/2928)
[30](https://ejournal.polraf.ac.id/index.php/JIRA/article/view/663)
[31](https://arxiv.org/html/2503.06132v1)
[32](https://arxiv.org/abs/2111.15640)
[33](https://arxiv.org/pdf/2310.19789.pdf)
[34](http://arxiv.org/pdf/2201.00308.pdf)
[35](https://npg.copernicus.org/articles/31/409/2024/)
[36](https://www.nature.com/articles/s41598-024-61040-3)
[37](https://arxiv.org/html/2406.14862v7)
[38](https://ernestryu.com/courses/FM/diffusion4.pdf)
[39](https://liner.com/ko/review/generative-human-motion-stylization-in-latent-space)
[40](https://milvus.io/ai-quick-reference/what-does-it-mean-for-a-diffusion-model-to-be-conditional)
[41](https://onlinelibrary.wiley.com/doi/10.1002/tee.24254)
[42](https://ieeexplore.ieee.org/document/10896580/)
[43](https://ieeexplore.ieee.org/document/10657869/)
[44](https://ieeexplore.ieee.org/document/10208651/)
[45](https://ieeexplore.ieee.org/document/10687922/)
[46](https://onepetro.org/armaigs/proceedings/IGS24/IGS24/ARMA-IGS-2024-0455/632580)
[47](https://arxiv.org/abs/2410.19324)
[48](http://arxiv.org/pdf/2404.06760.pdf)
[49](https://arxiv.org/html/2411.04873)
[50](http://arxiv.org/pdf/2310.09213.pdf)
[51](https://arxiv.org/html/2410.21314)
[52](https://arxiv.org/abs/2303.11073)
[53](https://arxiv.org/html/2405.14857v1)
[54](https://neurips.cc/virtual/2023/73957)
[55](https://www.nature.com/articles/s41598-024-51400-4)
[56](https://github.com/WeilunWang/semantic-diffusion-model)
