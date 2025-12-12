# Pastiche Master: Exemplar-Based High-Resolution Portrait Style Transfer

### 1. 논문의 핵심 주장 및 주요 기여

본 논문은 **DualStyleGAN**이라는 새로운 GAN 구조를 제안하여 **exemplar 기반 고해상도 초상화 스타일 전이**(1024×1024)를 실현한다. 논문의 핵심 주장은 기존 StyleGAN이 학습된 전체 분포를 변환하기만 하는 반면, 사용자가 선호하는 예시 이미지에 기반한 유연한 스타일 제어가 불가능하다는 점을 극복한다는 것이다.[1]

**세 가지 주요 기여:**

1. **Intrinsic & Extrinsic 듀얼 스타일 경로**: 원본 이미지의 스타일을 제어하는 intrinsic 경로와 목표 영역의 스타일을 모델링하는 extrinsic 경로를 명확히 분리하여, 색상과 구조적 스타일에 대한 계층적 제어 실현[1]

2. **원리적인 외재적 스타일 경로 설계**: 세밀한 튜닝 동작 분석을 기반으로 Modulative Residual Blocks (ModRes)를 통해 StyleGAN의 사전학습된 생성 공간을 보존하면서도 구조적 스타일 변조 가능[1]

3. **점진적 세밀한 튜닝(Progressive Fine-Tuning) 방법론**: 커리큘럼 학습 개념을 적용하여 3단계에 걸쳐 네트워크를 안정적으로 학습, 수백 개의 스타일 예시만으로도 효과적인 전이 학습 달성[1]

***

### 2. 해결하고자 하는 문제 및 제안 방법

#### 2.1 문제 정의

기존 StyleGAN 기반 접근법의 한계:[1]
- **도메인 레벨 스타일 전이**: StyleGAN을 새로운 도메인으로 세밀한 튜닝하면, 고정된 스타일만 생성 가능
- **예시 기반 제어 불가**: 사용자가 원하는 구체적인 예시 이미지의 스타일을 적용할 수 없음
- **구조적 스타일 전이 실패**: 단순 레이어 스왑(feature swapping)은 도메인 간 정렬 부족으로 구조적 스타일(만화의 추상화, 캐리커처의 변형) 전이 불가

#### 2.2 제안 방법: DualStyleGAN의 구조

**네트워크 구조:**[1]

$$G(I, S, \mathbf{w}) = \text{DualStyleGAN}(\text{Content}, \text{Style}, \text{Weight})$$

여기서 $$I$$는 입력 이미지, $$S$$는 스타일 예시 이미지, $$\mathbf{w} \in \mathbb{R}^{18}$$는 두 경로의 가중치 벡터이다.

**두 가지 스타일 경로의 역할:**[1]

1. **Intrinsic 스타일 경로 (원본 StyleGAN)**
   - 입력: $$z \in \mathbb{R}^{1 \times 512}$$ (단위 가우시안 노이즈) 또는 $$\mathbf{z}_i^+ = E(I)$$ (이미지 임베딩)
   - 역할: 원본 얼굴 도메인의 스타일 제어
   - 18개 레이어에서 AdaIN을 통해 스타일 변조

2. **Extrinsic 스타일 경로 (새로운 설계)**
   - 입력: $$\mathbf{z}_e^+ = E(S)$$ (스타일 이미지 임베딩)
   - 역할: 목표 도메인의 색상과 구조적 스타일 모델링

**계층적 스타일 분해:**[1]

- **구조 제어 (Coarse-resolution 레이어, 1~7)**: 
  - ModRes (Modulative Residual) 블록 사용
  - 구조 변환 블록 $$T_s$$를 통해 도메인 특정 구조적 스타일 학습
  
- **색상 제어 (Fine-resolution 레이어, 8~18)**:
  - 색상 변환 블록 $$T_c$$로 구성된 완전 연결 계층
  - 스타일 코드 $$\mathbf{z}_e^+$$를 매핑 네트워크 $$f$$를 통해 처리

#### 2.3 세밀한 튜닝 행동의 시뮬레이션

**ModRes 블록의 설계 이유:**[1]

세밀한 튜닝 후 StyleGAN의 합성곱 레이어가 가장 많이 변한다는 관찰에 기반하여, 다음 세 가지 방식을 비교 실험:

$$\text{Output} = \text{Input} + \text{Residual}(\text{Input})$$

- **AdaIN (채널별)**: 불충분 (Fig. 5d)
- **Diagonal Attention (공간별)**: 불충분 (Fig. 5e)
- **ResBlock (요소별)**: 최적 근사 (Fig. 5c) ✓

#### 2.4 페이셜 Destylization 방법

**문제**: 예술적 초상화에서 현실적인 얼굴-초상화 쌍을 얻기 위한 감독 데이터 부족

**3단계 해결책:**[1]

**Stage I: Latent 초기화**

$$\mathbf{z}_e^+ = E(S) \in \mathbb{R}^{18 \times 512}$$

인코더 $$E$$를 통해 예술적 초상화를 $$Z^+$$ 공간에 임베딩

**Stage II: Latent 최적화**

$$\hat{\mathbf{z}}_e^+ = \arg \min_{\mathbf{z}^+} \mathcal{L}_{\text{perc}}(g'(\mathbf{z}^+), S) + \lambda_{\text{ID}} \mathcal{L}_{\text{ID}}(g'(\mathbf{z}^+), S) + \|\sigma(\mathbf{z}^+)\|_1$$

여기서:
- $$\mathcal{L}_{\text{perc}}$$: 지각적 손실 (Perceptual Loss)
- $$\mathcal{L}\_{\text{ID}}$$: 신원 보존 손실 ($$\lambda_{\text{ID}} = 0.1$$)
- $$\sigma(\mathbf{z}^+)$$: 18개 벡터의 표준편차 ($$\mathbb{Z}$$ 공간으로 정규화)

**Stage III: 이미지 임베딩**

$$\mathbf{z}_i^+ = E(g(\hat{\mathbf{z}}_e^+))$$

이 방식은 현실성이 떨어지는 세부 사항을 제거하여 구조적 변형의 의도를 드러낸다.

***

### 3. 모델의 일반화 성능 향상 가능성

#### 3.1 점진적 세밀한 튜닝 전략

**3단계 과정으로 task 난이도를 점진적으로 증가:**[1]

**Stage I: 원본 도메인에서의 색상 전이**

$$G(z_1, z_2, 1) \approx g(z^+) \text{ (Style Mixing)}$$

- ModRes 블록의 가중치를 거의 0에 가깝게 초기화
- 색상 변환 블록의 FC 레이어를 항등 행렬로 초기화
- 결과: 원본 StyleGAN의 생성 공간 유지 ✓

**Stage II: 원본 도메인에서의 구조 전이**

$$\min_G \max_D \lambda_{\text{adv}} \mathcal{L}_{\text{adv}} + \lambda_{\text{perc}} \mathcal{L}_{\text{perc}}(G(z_1, z_2, 1), g(\mathbf{z}_{1:l}))$$

여기서 $$l$$을 7에서 5로 점진적으로 감소시켜 구조 정보 비중을 증가.

**Stage III: 목표 도메인에서의 스타일 전이**

$$\min_G \max_D \lambda_{\text{adv}} \mathcal{L}_{\text{adv}} + \lambda_{\text{perc}} \mathcal{L}_{\text{perc}} + \mathcal{L}_{\text{sty}} + \mathcal{L}_{\text{con}}$$

여기서:

$$\mathcal{L}_{\text{sty}} = \lambda_{\text{cx}} \mathcal{L}_{\text{cx}}(G(z, z_e^+, 1), S) + \lambda_{\text{FM}} \mathcal{L}_{\text{FM}}(G(z, z_e^+, 1), S')$$

- $$\mathcal{L}_{\text{cx}}$$: Contextual Loss (예시 기반 스타일 일치)
- $$\mathcal{L}_{\text{FM}}$$: Feature Matching Loss

**내용 손실:**

$$\mathcal{L}_{\text{con}} = \lambda_{\text{ID}} \mathcal{L}_{\text{ID}}(G(z, z_e^+, 1), g(z)) + \lambda_{\text{reg}} \|\mathbf{W}\|_2$$

- $$\mathcal{L}_{\text{ID}}$$: 신원 보존 (얼굴 특징 유지)
- $$\|\mathbf{W}\|_2$$: ModRes 가중치 정규화 (원본 구조 보존)

#### 3.2 페이셜 Destylization을 통한 다양성 학습

**Mode Collapse 방지 메커니즘:**[1]

기존 무조건 세밀한 튜닝 (Unconditional Fine-tuning)의 문제:
- StyleGAN의 생성 공간이 목표 도메인으로 전체적으로 이동
- 다양한 스타일 학습 불가 → Mode Collapse

**DualStyleGAN의 장점:**
- Extrinsic 경로만 학습하고 사전학습된 StyleGAN 보존
- 원래의 다양한 얼굴 특징 유지
- 감시 신호 (얼굴-초상화 쌍) 제공으로 다양한 구조적 변형 학습

#### 3.3 계층적 스타일 제어의 일반화

**구조적 의미의 계층적 분해 (Fig. 11 기반):**[1]

- **4×4 + 8×8 레이어**: 전체 얼굴 모양 조정
- **16×16 레이어**: 입과 눈 같은 얼굴 부위 과장
- **32×32 레이어**: 주름 같은 세부 형태 초점

이러한 계층적 설계는:
1. 서로 다른 스타일의 특성을 각 해상도에 자동으로 할당
2. 새로운 도메인에 대한 전이 학습 시 더 효율적인 학습 경로 제공
3. 세밀하게 조정된 스타일 가중치 벡터 $$\mathbf{w}$$로 유연한 제어 가능

***

### 4. 성능 향상 및 한계

#### 4.1 성능 향상

**정량적 평가 - 사용자 선호도 (Table 1):**[1]

| 방법 | 카툰 | 캐리커처 | 애니메 | 평균 |
|------|------|---------|--------|------|
| **DualStyleGAN** | **0.93** | **0.79** | **0.78** | **0.83** |
| UI2I-style | 0.05 | 0.15 | 0.14 | 0.11 |
| StarGANv2 | 0.01 | 0.00 | 0.04 | 0.02 |
| GNR | 0.01 | 0.06 | 0.04 | 0.04 |

**비교 대상 방법들의 한계:**[1]
- **Toonify, FS-Ada, U-GAT-IT**: 도메인 레벨 스타일 전이만 가능 (예시 기반 제어 불가)
- **UI2I-style**: 색상 제어만 효과적, 구조적 특징 혼합 실패
- **StarGANv2, GNR**: 심한 데이터 불균형으로 오버피팅

**정성적 결과:**[1]
- 색상과 구조적 스타일 모두에서 우수한 전이
- 고해상도(1024×1024) 생성으로 세밀한 디테일 보존
- 카툰(추상화), 캐리커처(변형), 애니메(독특한 특징) 등 다양한 스타일 처리

#### 4.2 Ablation Study 결과

**페이셜 Destylization의 효과 (Fig. 10a):**[1]
- 감독 신호 없음: 초상화 과적합, 입력 얼굴 구조 무시
- 감독 신호 포함: 얼굴-초상화 간 구조적 관계 학습, 합리적 결과

**정규화의 효과 (Fig. 10b):**[1]

$$\lambda_{\text{reg}} = 0.005 \text{(최적값)}$$

- $$\lambda_{\text{reg}} = 0$$: 헤어 스타일 과적합
- $$\lambda_{\text{reg}} = 0.01$$: 입 모양 과도하게 보존

**점진적 세밀한 튜닝의 중요성 (Fig. 10c):**[1]
- Stage I만으로: 생성 공간 심각하게 변경 → 완전 실패
- Stage II+III: 불완전한 세밀한 튜닝
- Stage I+II+III (완전): 유일하게 효과적인 스타일 전이 달성

#### 4.3 한계점

**데이터 편향 문제 (Fig. 16):**[1]

1. **비얼굴 영역 손실**: 모자, 배경 텍스처 같은 얼굴 외 지역 세부사항 손실
2. **색상-구조 트레이드오프**: 원본 색상 보존 시 애니메 스타일의 추상적 코 불자연스러움
3. **헤어 스타일 편향**: 애니메 데이터의 직선 머리 및 앞머리 편향 → 곱슬머리 처리 실패
4. **드문 스타일 미처리**: 극단적으로 큰 눈 같은 비정상적 스타일 모방 불가

**미학습 스타일에 대한 일반화 (Fig. 15):**[1]
- 학습 데이터 외 스타일: 덜 일관성 있는 결과 생성
- Latent 최적화로 개선 가능하나 인공물 발생 가능
- 향후 연구 과제

**계산 비용:**[1]
- Destylization: ~5시간 (100개 이미지 당)
- Stage II: ~0.5시간
- Stage III: ~0.75시간 (평균, 스타일별 다름)
- 추론: ~0.13초/이미지 (실시간 응용에 충분)

***

### 5. 모델의 일반화 성능 향상 가능성에 대한 심층 분석

#### 5.1 현재 설계의 일반화 강점

**1) 도메인-불가지론적(Domain-Agnostic) 구조:**[1]
- Extrinsic 경로의 설계가 특정 스타일에 제한되지 않음
- 인코더 $$E$$가 다양한 예술적 초상화 특징 추출 가능
- 새로운 도메인 학습 시 3단계 과정만 반복하면 적용 가능

**2) 계층적 특징 분해의 이점:**[1]
- 구조 vs 색상의 명확한 분리 (Fig. 11)
- 각 해상도 레이어가 의미 있는 시각적 특징을 학습
- 다양한 스타일 도메인에 쉽게 적용 가능

**3) Extrinsic 경로 고립의 이점:**[1]
$$G(\mathbf{z}, \mathbf{z}_e^+, \mathbf{w}=0) \approx g(\mathbf{z})$$
- 사전학습된 StyleGAN의 생성 공간 보존
- 전이 학습의 안정성 증가
- Mode collapse 방지

#### 5.2 일반화 향상을 위한 잠재적 개선 방향

**1) 데이터 불균형 해결:**[1]
논문에서 제시한 미래 계획:
- 데이터 증강을 통한 드문 스타일 처리 개선
- 더 다양한 헤어 스타일, 얼굴 각도 데이터셋 확보

**2) 주의 메커니즘 통합:**
최신 연구 동향 (2024-2025):[2][3]
- Domain-Aware 어댑터로 도메인 간 대응 강화
- Semantic 기반 스타일 전이로 구조적 일치도 개선

**3) Diffusion 모델과의 통합:**[3][2]
최근 2024-2025년 연구:
- Diffusion 기반 접근이 더 나은 일반화 성능 보임
- 예: "Domain Generalizable Portrait Style Transfer"는 Pre-trained diffusion 모델과 Semantic Adapter로 여러 도메인에서 우수한 성능 달성[3]
- 구조: Diffusion features로 밀집 의미론적 대응(Dense Semantic Correspondence) 수립

#### 5.3 전이 학습 안정성의 이론적 근거

**Semantic Alignment 보존:[1]
- 미세 튜닝 전후 StyleGAN의 잠재 공간이 의미론적으로 정렬됨 (Prior work: StyleAlign )[4]
- DualStyleGAN의 ModRes 설계가 이러한 정렬 유지
- 결과: 새로운 도메인 학습 시에도 콘텐츠-스타일 분리 유지 가능

**Progressive 전략의 이점:[1]
- Curriculum Learning 원리 적용
- 간단한 작업(Stage I: 색상)에서 복잡한 작업(Stage III: 구조 + 색상)으로 점진적 난이도 증가
- 각 단계에서의 학습이 다음 단계의 수렴 촉진

***

### 6. 2020년 이후 관련 최신 연구 탐색

#### 6.1 StyleGAN 기반 연구의 진화

**2020-2021: StyleGAN 기반 초상화 생성의 기초 확립**[5]
- **"Fine-tuning StyleGAN2 for Cartoon Face Generation"** (2021): Cartoon-StyleGAN
  - 제한된 데이터에서 Style GAN2 세밀한 튜닝 기술 확립
  - Adaptive Discriminator Augmentation (ADA) 도입으로 학습 안정성 향상

- **"Encoding in Style: A StyleGAN Encoder"** (2021, pSp encoder)[6]
  - StyleGAN 잠재 공간 반전(Inversion) 기술 고도화
  - 실제 이미지를 $$W^+$$ 공간에 임베딩하는 효율적 인코더 개발
  - 본 논문의 Destylization에서 활용됨

**2022-2023: 초상화 특정 작업의 고도화**[7][8][9]

- **"Pastiche Master: Exemplar-Based High-Resolution Portrait Style Transfer"** (2022, 본 논문)[10]
  - 최초로 exemplar 기반 고해상도 스타일 전이 달성
  - Dual 경로 설계로 색상과 구조 스타일 명확히 분리

- **"Face Generation and Editing with StyleGAN: A Survey"** (2023)[7]
  - StyleGAN 기반 얼굴 생성 및 편집 최신 동향 총괄
  - 학습 메트릭, 잠재 표현, GAN 반전, 얼굴 편집, 크로스도메인 스타일화 포괄

- **"StyleIPSB: Identity-Preserving Semantic Basis of StyleGAN"** (2023)[11]
  - StyleGAN의 의미론적 기저(Semantic Basis) 분석
  - 신원 보존 얼굴 교환 및 고충실도 초상화 편집 달성

- **"ChildGAN: Large Scale Synthetic Child Facial Data"** (2023)[9]
  - StyleGAN2 기반 아동 얼굴 합성 데이터 생성
  - 도메인 적응을 통한 효과적인 전이 학습 시연

#### 6.2 Diffusion 모델 기반의 새로운 패러다임 (2023-2025)

**Diffusion 모델의 부상:**[12][2][3]

최근 2024-2025년 연구들은 GAN을 넘어 Diffusion 모델로 패러다임 전환:

**"Domain Generalizable Portrait Style Transfer"** (2025, 최신)[3]
- **핵심 혁신**: Pre-trained Diffusion 모델 + Semantic Adapter로 보다 나은 도메인 일반화
- **방법**:
  1. Diffusion 특징으로 밀집 의미론적 대응(Dense Semantic Correspondence) 수립
  2. 참조 초상화를 내용과 의미론적으로 정렬된 상태로 워핑
  3. ControlNet으로 구조 가이드, Image Adapter로 스타일 가이드 제공
  4. AdaIN-Wavelet 변환으로 스타일라이제이션과 내용 보존 균형

- **성능**: 여러 도메인에서 StyleGAN 방식보다 우수한 일반화 성능
- **장점**:
  - Diffusion 모델의 풍부한 의미론적 정보 활용
  - 구조적 변이가 큰 초상화 간 스타일 전이 성능 향상
  - 다양한 도메인에 대한 강화된 적응력

**"Style Transfer with Diffusion Models for Synthetic-to-Real"** (2025)[13]
- Class-wise Adaptive Instance Normalization (AdaIN) 기법
- Semantic consistency를 유지하면서 합성 이미지를 실사 도메인으로 전환

**"Diffusion model for one-shot text-image style transfer"** (2025)[2]
- Diff-TST 모델로 한 장의 텍스트 이미지만으로 스타일 전이
- 다국어 지원으로 일반화 확대

#### 6.3 스타일 제어의 세분화 (2021-2024)

**"Style Fader Generative Adversarial Networks"** (2022)[14]
- **스타일 정도 제어**: Style Scaling Injection (SSI) + Style Degree Interpretation (SDI)
- 기존 GAN 기반 모델에 쉽게 통합 가능
- 실시간 스타일 정도 조정으로 유연성 증대

**"Domain-Aware Universal Style Transfer"** (2021)[15]
- 예술 스타일과 사진 현실주의 스타일을 모두 처리하는 통합 프레임워크
- Domain-aware skip connection으로 의미 정보와 구조 세부사항 보존

#### 6.4 3D 인식 얼굴 생성 및 편집 (2023-2025)

**"Efficient 3D-Aware Facial Image Editing"** (2024)[16]
- StyleGAN의 표현력과 분리된 잠재 공간 활용
- 다양한 목표 포즈에서 속성별 제어 가능

**"Toonify3D: StyleGAN-based 3D Stylized Face Generator"** (2024)[17]
- StyleGAN 기반 전체 두상(Full-head) 3D 스타일화 아바타 생성
- GAN 기반 3D 얼굴 표정 편집 지원

**"Text-to-Face Generation with StyleGAN2"** (2022)[18]
- BERT 임베딩으로 텍스트를 StyleGAN2 잠재 공간에 매핑
- 1024×1024 고해상도 얼굴 생성 (57% 유사도)

#### 6.5 특수 응용 분야

**의료/임상 응용:**[19]
- **"CleftGAN: Transfer Learning for Cleft Face Generation"** (2025)
  - 514개 구순구개열 환자 이미지로 StyleGAN3 세밀한 튜닝
  - Frechet Inception Distance (FID) 기반 성능 평가
  - 희귀 질환 합성 데이터 생성으로 임상 모델 훈련 지원

**다중 도메인 스타일 전이:**[20]
- **"GRA-GAN: Gender, Race, Age Image Transformation"** (2022)
  - 향상된 CycleGAN 기반으로 성별, 인종, 나이 변환
  - Channel-wise 및 Element-wise 곱셈으로 향상된 다양성

#### 6.6 평가 메트릭의 발전

최신 연구들이 도입한 평가 메트릭들:[3]
- **Gram Loss**: 스타일 일치도 측정 (스타일 손실의 전통적 방식)
- **LPIPS (Learned Perceptual Image Patch Similarity)**: 지각적 유사성
- **ID (Identity Distance)**: 신원 보존도 평가 (ArcFace 기반)
- **FID (Fréchet Inception Distance)**: 생성 이미지의 분포 유사성
- **PPL (Perceptual Path Length)**: 의미론적 보간 부드러움

***

### 7. 해당 논문이 앞으로의 연구에 미치는 영향과 고려사항

#### 7.1 학술적 영향

**1) 건축학적 혁신:**
DualStyleGAN의 설계 철학은 이후 많은 스타일 전이 연구에 영향:
- Intrinsic/Extrinsic 경로 분리의 개념이 후속 연구들의 지침
- 계층적 스타일 제어 개념이 diffusion 모델 기반 연구로 확장

**2) Exemplar-Based 접근의 확립:**
- 이 논문이 최초로 exemplar 기반 고해상도 초상화 스타일 전이 달성
- 후속 연구들이 이를 기반으로 domain generalization 강화

**3) 점진적 학습(Curriculum Learning) 재평가:**
- 복잡한 스타일 전이 작업에 curriculum learning의 효과 실증
- Diffusion 모델 기반 연구에서도 유사한 단계적 접근 채택

#### 7.2 기술적 발전 방향과 고려사항

**1) 데이터 효율성의 한계와 개선:**

현재 한계:[1]
- 수백 개의 예술적 초상화 필요
- 새로운 도메인 적용 시 수 시간의 학습 시간

개선 방향:[1]
- Few-shot 또는 one-shot 학습으로 데이터 요구량 감소
- 최신 연구: Diffusion 기반 접근이 더 나은 few-shot 성능 보임[2][3]

**2) 일반화 성능의 질적 도약:**

논문의 한계:
- 미학습 스타일에 대한 일반화 능력 부족 (Fig. 15)
- 비얼굴 영역(배경, 옷) 처리 미흡

최신 해결책 (2025):[3]
- Semantic Adapter를 통한 밀집 의미론적 대응으로 도메인 간 구조 차이 극복
- Diffusion 기반 방식이 구조적 변이가 큰 도메인에서 더 우수한 성능

**3) 모드 붕괴(Mode Collapse) 문제의 해결:**

DualStyleGAN의 접근:
- Extrinsic 경로만 학습하면서 원본 다양성 보존
- 페이셜 Destylization으로 다양한 구조적 변형 감시

향후 고려사항:
- 더 복잡한 도메인(예: 추상화 미술)에서의 모드 붕괴 재발 가능성
- 정규화 강도 $$\lambda_{\text{reg}}$$ 와 다양성 간 트레이드오프 조정 필요

#### 7.3 응용 분야 확대 전망

**1) 의료 이미징:**[19]
- 희귀 질환의 합성 데이터 생성으로 의료 AI 훈련 지원
- 클리니컬 이미지 도메인 적응

**2) 크로스 도메인 얼굴 작업:**[9][7]
- 애니메이션 캐릭터 생성 확대
- 3D 아바타 생성과의 결합 (예: Toonify3D)[17]

**3) 비디오 스타일 전이:**[21]
- 프레임 단위의 일관성 있는 스타일 전이
- 임시 코히어런스(Temporal Coherence) 유지 메커니즘 개발 필요

#### 7.4 향후 연구 시 고려할 핵심 기술적 요소

**1) 모델 아키텍처 관점:**
- ✓ 이원 경로(Dual-Path) 설계의 일반성 검증
- ✓ ModRes 외 다른 잔차 메커니즘 탐색 필요
- ✓ Attention 메커니즘과의 결합 가능성[22]

**2) 학습 전략 관점:**
- ✓ 점진적 세밀한 튜닝의 단계 수 최적화
- ✓ 손실 함수의 가중치 자동 조정 메커니즘
- ✓ Meta-learning 을 통한 빠른 도메인 적응

**3) 평가 및 검증:**
- ✓ 사용자 연구의 통계적 엄밀성 강화 (현재: 27명)
- ✓ 다양한 문화권의 초상화 스타일에 대한 cross-cultural 평가
- ✓ 객관적 메트릭과 주관적 평가 간 상관관계 분석

**4) 에너지 효율성:**
- 현재: ~6시간(Destylization + 세밀한 튜닝) 소요
- 향후 고려: 경량 모델(Mobile StyleGAN 등) 개발, Distillation 기법 적용

#### 7.5 Diffusion 모델 시대의 성찰

**논문 발표 이후의 패러다임 변화:**

2024-2025년 최신 트렌드:[12][2][3]
- GAN 중심에서 Diffusion 모델 중심으로 패러다임 전환
- 이유: Diffusion 모델의 우수한 일반화 능력과 안정적 훈련

**DualStyleGAN의 지속적 가치:**
1. **개념적 기여**: Intrinsic/Extrinsic 경로 분리는 여전히 유효
2. **실용성**: GAN의 빠른 추론(0.13초/이미지)은 실시간 응용에 우수
3. **모듈화**: 다른 생성 모델과의 결합 가능성 (예: Diffusion 기반 extrinsic 경로)

**향후 방향:**
- 혼합 모델: GAN의 빠른 생성 + Diffusion의 우수한 일반화
- 예시: Diffusion 기반 초상 스타일 전이는 ControlNet(GAN 유사 구조) 활용[3]

***

### 결론

"Pastiche Master"는 **exemplar 기반 고해상도 초상화 스타일 전이**의 새로운 기준을 제시한 중요한 논문이다. DualStyleGAN의 이원 경로 설계와 점진적 세밀한 튜닝 전략은 학술적, 실무적으로 큰 영향을 미쳤다.

특히 **일반화 성능 측면에서** 다음과 같은 특징이 있다:

1. **강점**: 사전학습된 StyleGAN 보존으로 안정적인 전이 학습, 계층적 스타일 제어로 의미론적 분리
2. **한계**: 미학습 도메인 일반화 부족, 데이터 편향 문제
3. **개선 방향**: Diffusion 모델과의 결합, Semantic Adapter 도입, 더 정교한 도메인 적응 메커니즘

2024-2025년의 최신 연구들은 Diffusion 모델 기반 접근으로 이러한 한계들을 극복하고 있으며, DualStyleGAN의 설계 철학(이원 경로, 계층적 제어)은 새로운 모델들의 기초 개념으로 계속 활용되고 있다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3ae60261-f942-4a2e-8b5e-e6c2e06c4fd0/2203.13248v1.pdf)
[2](https://www.sciencedirect.com/science/article/abs/pii/S0957417424026149)
[3](https://arxiv.org/html/2507.04243v1)
[4](https://academic.oup.com/ijpp/article/32/Supplement_2/ii56/7885445)
[5](https://arxiv.org/pdf/2106.12445.pdf)
[6](https://openaccess.thecvf.com/content/CVPR2021/papers/Richardson_Encoding_in_Style_A_StyleGAN_Encoder_for_Image-to-Image_Translation_CVPR_2021_paper.pdf)
[7](https://arxiv.org/abs/2212.09102)
[8](https://arxiv.org/html/2409.00345v1)
[9](https://arxiv.org/pdf/2307.13746.pdf)
[10](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Pastiche_Master_Exemplar-Based_High-Resolution_Portrait_Style_Transfer_CVPR_2022_paper.pdf)
[11](https://openaccess.thecvf.com/content/CVPR2023/papers/Jiang_StyleIPSB_Identity-Preserving_Semantic_Basis_of_StyleGAN_for_High_Fidelity_Face_CVPR_2023_paper.pdf)
[12](https://www.nature.com/articles/s41598-025-95819-9)
[13](https://arxiv.org/html/2505.16360v2)
[14](https://www.ijcai.org/proceedings/2022/0693.pdf)
[15](https://openaccess.thecvf.com/content/ICCV2021/papers/Hong_Domain-Aware_Universal_Style_Transfer_ICCV_2021_paper.pdf)
[16](https://arxiv.org/html/2406.04413)
[17](https://dl.acm.org/doi/10.1145/3641519.3657480)
[18](https://arxiv.org/ftp/arxiv/papers/2205/2205.12512.pdf)
[19](https://www.nature.com/articles/s41598-025-86588-6)
[20](https://www.sciencedirect.com/science/article/pii/S0957417422002512)
[21](https://ieeexplore.ieee.org/document/11257596/)
[22](https://peerj.com/articles/cs-2332)
[23](https://www.mdpi.com/2227-7390/13/11/1861)
[24](https://ieeexplore.ieee.org/document/10879474/)
[25](https://ieeexplore.ieee.org/document/11213614/)
[26](https://www.worldscientific.com/doi/10.1142/S012915642540779X)
[27](https://ph01.tci-thaijo.org/index.php/ecticit/article/view/260297)
[28](https://ieeexplore.ieee.org/document/9356828/)
[29](https://ieeexplore.ieee.org/document/9259251/)
[30](https://ieeexplore.ieee.org/document/9107081/)
[31](https://ieeexplore.ieee.org/document/9036960/)
[32](https://arxiv.org/abs/2308.10601)
[33](https://arxiv.org/pdf/1702.06762.pdf)
[34](https://arxiv.org/pdf/1904.02296.pdf)
[35](http://thesai.org/Downloads/Volume14No11/Paper_32-Enhancing_Style_Transfer_with_GANs.pdf)
[36](https://downloads.hindawi.com/journals/mpe/2020/9453586.pdf)
[37](http://arxiv.org/pdf/2501.01106.pdf)
[38](http://arxiv.org/pdf/2408.12673.pdf)
[39](https://arxiv.org/html/2506.19278v1)
[40](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yi_APDrawingGAN_Generating_Artistic_Portrait_Drawings_From_Face_Photos_With_Hierarchical_CVPR_2019_paper.pdf)
[41](https://en.wikipedia.org/wiki/Generative_adversarial_network)
[42](https://liner.com/ko/review/apdrawinggan-generating-artistic-portrait-drawings-from-face-photos-with-hierarchical)
[43](https://github.com/neverbiasu/Awesome-Portraits-Style-Transfer)
[44](https://www.nature.com/articles/s41598-025-30170-7)
[45](https://github.com/happy-jihye/Cartoon-StyleGAN)
[46](https://biss.pensoft.net/article/140428/)
[47](https://papers.academic-conferences.org/index.php/ecel/article/view/3035)
[48](https://openaccess.cms-conferences.org/publications/book/978-1-964867-10-6/article/978-1-964867-10-6_9)
[49](https://open-publishing.org/publications/index.php/APUB/article/view/1335)
[50](https://ojs.bonviewpress.com/index.php/IJCE/article/view/2702)
[51](https://invergejournals.com/index.php/ijss/article/view/90)
[52](https://academic.oup.com/humrep/article/doi/10.1093/humrep/deae108.292/7703349)
[53](https://mspsss.org.ua/index.php/journal/article/view/972)
[54](https://iopscience.iop.org/article/10.1149/MA2024-022267mtgabs)
[55](https://arxiv.org/html/2310.04194)
[56](https://arxiv.org/html/2409.11010)
[57](http://arxiv.org/pdf/1812.04948.pdf)
[58](https://github.com/MingkunLei/Awesome-Style-Transfer-with-Diffusion-Models)
[59](https://breckon.org/toby/publications/papers/abarghouei20domain-transfer.pdf)
[60](https://www.sciencedirect.com/science/article/pii/S0925231225020016)
[61](https://arxiv.org/abs/2507.04243v1)
[62](https://liner.com/ko/review/inversionbased-style-transfer-with-diffusion-models)
[63](https://artilects.net/manuscripts/oys-prmu202010.pdf)
