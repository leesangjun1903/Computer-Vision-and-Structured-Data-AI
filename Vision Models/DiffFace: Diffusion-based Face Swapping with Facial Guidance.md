# DiffFace: Diffusion-based Face Swapping with Facial Guidance

### 1. 핵심 주장 및 주요 기여

**DiffFace**는 2022년 12월에 제안된 첫 번째 **확산 모델 기반 안면 교체(face swapping) 프레임워크**이다. 이 연구는 기존의 GAN 기반 방식을 탈피하여 확산 모델의 안정적인 학습과 우수한 생성 능력을 활용한다.[1]

**핵심 주장:**
- GAN 기반 안면 교체 방식은 min-max 최적화 문제로 인해 학습이 불안정하며, 신원(ID)과 속성(attributes) 간의 균형을 맞추기 어렵다.[1]
- 확산 모델은 더 안정적인 학습, 높은 충실도(fidelity), 그리고 우수한 제어성을 제공한다.[1]

**주요 기여:**
1. ID Conditional DDPM: 신원 정보를 명시적으로 조건화한 확산 모델 개발
2. 다중 안면 가이던스(facial guidance): 신원, 의미적 특성(semantics), 시선(gaze) 가이던스 통합
3. Target-Preserving Blending: 점진적으로 증가하는 마스크 강도를 통한 목표 속성 보존
4. 재학습 없이 유연한 속성 제어 가능

***

### 2. 해결하고자 하는 문제

**안면 교체 과제의 본질적 문제:**

안면 교체는 원본 이미지의 신원을 소스 이미지에서 가져오면서 목표 이미지의 속성(표정, 자세, 형태, 피부톤 등)을 보존해야 한다. 이는 다음과 같은 근본적 난제를 야기한다:[1]

$$\text{Goal: Transfer ID}(\mathbf{x}_{\text{src}}) \rightarrow \mathbf{x}_{\text{swap}} \text{ while preserving } \text{Attr}(\mathbf{x}_{\text{targ}})$$

**GAN 기반 방식의 한계:**
- 신원-속성 간 거래 관계(trade-off)로 인한 성능 저하
- 복잡한 손실 함수 조합과 광범위한 하이퍼파라미터 튜닝 필요
- 외부 모델 통합 시 불안정성 증가

***

### 3. 제안하는 방법 및 수식

#### 3.1 ID Conditional DDPM

**기본 확산 과정:**

확산 모델의 정방향 과정(forward process)은 다음과 같이 정의된다:[1]

$$q(x_t|x_{t-1}) := \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

여기서 $\beta_t$는 사전 정의된 분산 스케줄이다.

역방향 과정(reverse process)은:

$$p_\theta(x_{t-1}|x_t) := \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_\theta(x_t, t)I)$$

평균은 노이즈 추정 모델을 통해:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{1-\beta_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right)$$

**ID 조건화:**

원본 이미지에서 신원 벡터를 추출한다:[1]

$$v_{\text{id}} = D_I(x_{\text{src}})$$

여기서 $D_I$는 신원 임베더(ArcFace 또는 CosFace)이고, 이를 U-Net의 잔차 블록(residual block)에 주입한다: $\epsilon_\theta(x_t, t, v_{\text{id}})$

**손실 함수:**

DiffFace의 전체 손실은 두 가지 성분으로 구성된다:[1]

$$\mathcal{L}_{\text{noise}} = \|\epsilon - \epsilon_\theta(x_t, t, v_{\text{id}})\|_2^2$$

완전 제거 예측을 위해:

$$\hat{x}_0 = f_\theta(x_t, t, v_{\text{id}}) := x_t - \frac{\sqrt{1-\bar{\alpha}_t}}{\sqrt{\bar{\alpha}_t}}\epsilon_\theta(x_t, t)$$

신원 손실(identity loss):[1]

$$\hat{v}_{\text{id}} = D_I(\hat{x}_0)$$

$$\mathcal{L}_{\text{id}} = 1 - \cos(v_{\text{id}}, \hat{v}_{\text{id}})$$

**전체 손실:**[1]

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{id}} + \lambda \mathcal{L}_{\text{noise}}$$

#### 3.2 안면 가이던스(Facial Guidance)

**신원 가이던스:**

$$G_{\text{id}} = 1 - \cos(D_I(x_{\text{src}}), D_I(\hat{x}_0))$$

**의미적 가이던스(얼굴 파싱):**[1]

$$G_{\text{sem}} = \|D_F(x_{\text{targ}}) - D_F(\hat{x}_0)\|_2^2$$

여기서 $D_F$는 안면 파서로, 피부, 눈, 눈썹 등 11개 클래스를 사용한다.

**시선 가이던스:**[1]

$$G_{\text{gaze}} = \|D_G(x_{\text{targ}}) - D_G(\hat{x}_0)\|_2^2$$

**통합 가이던스:**[1]

$$x_{t-1} \sim \mathcal{N}(\mu - \sigma \nabla_{x_t} G_{\text{facial}}, \sigma I)$$

여기서:

$$G_{\text{facial}} = \lambda_{\text{id}} G_{\text{id}} + \lambda_{\text{sem}} G_{\text{sem}} + \lambda_{\text{gaze}} G_{\text{gaze}}$$

실험에서 가중치는 $\lambda_{\text{id}} = 2000$, $\lambda_{\text{sem}} = 150$, $\lambda_{\text{gaze}} = 200$로 설정되었다.[1]

#### 3.3 Target-Preserving Blending

**동적 마스크:**

기존의 하드 마스크는 noise로 인해 속성이 손상되므로, 시간에 따라 강도를 증가시키는 소프트 마스크를 제안한다:[1]

$$M_t = \min\left(1, \frac{T - t}{\hat{T}}M\right)$$

여기서 $\hat{T}$는 마스크 강도가 1이 되는 시작점이다.

**블렌딩 식:**[1]

$$x_{t-1} = \hat{x}_{t-1} \odot M_t + x_{t-1, \text{targ}} \odot (1 - M_t)$$

여기서 $\hat{x}\_{t-1}$은 가이던스를 받은 중간 예측이고, $x_{t-1,\text{targ}}$는 노이즈 처리된 목표 이미지이다.

***

### 4. 모델 구조

#### 4.1 네트워크 아키텍처

**ID Conditional DDPM:**
- 기반 아키텍처: U-Net (Wide ResNet 기반)[1]
- 신원 벡터는 각 U-Net의 잔차 블록에 주입
- 타임스텝 임베딩과 함께 선형 계층을 통해 처리
- ResNet-101 백본 사용 (ArcFace/CosFace)[1]

**외부 모듈:**
- **신원 임베더**: ArcFace 또는 CosFace (ResNet-101 기반)
- **안면 파서**: BiSeNet (19개 클래스 출력, 11개 얼굴 관련 클래스 선택)[1]
- **시선 추정기**: 사전 학습된 시선 추정 네트워크

#### 4.2 학습 설정

**데이터셋**: FFHQ (70,000개의 정렬된 얼굴 이미지, 256×256 해상도)[1]

**학습 파라미터:**
- 총 단계: 700,000
- 배치 크기: 48
- 학습률: 0.0001
- 옵티마이저: AdamW
- $\lambda = 0.5$ (신원 손실과 노이즈 손실의 균형)
- 하드웨어: 8개 NVIDIA A100 GPU (약 10일)

**샘플링:**
- 확산 단계: $T = 75$
- 확장 증강(extending augmentation): 8개

***

### 5. 성능 향상 및 실험 결과

#### 5.1 정량적 평가 (FaceForensics++ 데이터셋)

**비교 대상**: SimSwap, HifiFace, InfoSwap, MegaFS, FaceShifter, DeepFakes[1]

| 메트릭 | DiffFace (Cos) | DiffFace (Arc) | 최고 경쟁자 |
|--------|---|---|---|
| **신원 유사도** (↑) | 0.620 | 0.602 | HifiFace 0.565 |
| **신원 제거** (↑) | 0.859 | 0.816 | InfoSwap 0.841 |
| **표정 거리** (↓) | 0.044 | 0.043 | HifiFace 0.048 |
| **자세 거리** (↓) | 0.0009 | 0.0008 | SimSwap 0.0005 |
| **형태 거리** (↓) | 0.0269 | 0.0283 | FaceShifter 0.0235 |

**핵심 발견:**
- 신원 전달에서 우수한 성능 달성 (Arc 점수에서 0.602, Cos 점수에서 0.620)
- 신원 제거 능력: 상대 거리(Arc-R: 0.816, Cos-R: 0.859)로 측정
- GAN 기반 방식보다 확산 모델의 노이즈 제거 능력으로 인해 더 효과적인 신원 제거

#### 5.2 절제 연구(Ablation Study)

**ID Conditional DDPM의 효과:**[1]
- 무조건 (Unconditional): Arc = 0.486
- ID 조건 + 신원 가이던스 없음: Arc = 0.455
- 전체 (ID 조건 + 신원 가이던스): Arc = 0.602 (+24.4%)

**Target-Preserving Blending의 효과:**

$\hat{T}$ 값 조정을 통한 ID-속성 거래 제어:[1]

| $\hat{T}$ 값 | 신원 (Arc) | 표정 거리 | 형태 거리 |
|---------|---------|---------|---------|
| 30 | 0.580 | 0.035 | 0.0240 |
| 40 | 0.602 | 0.043 | 0.0283 |
| 50 | 0.603 | 0.049 | 0.0311 |

- $\hat{T}$가 증가할수록 신원 유사도 증가하지만 표정/형태 속성 손상
- $\hat{T}$가 감소할수록 목표 속성 보존 개선

**시선 가이던스의 효과:**[1]
- 가이던스 없음: 시선이 불일치
- 가이던스 적용: 목표 이미지의 시선과 일치

***

### 6. 모델의 일반화 성능 향상 가능성

#### 6.1 현재 일반화 능력

**도메인 외 결과 (Out-of-Domain):**[1]

논문은 학습 데이터(FFHQ, 정렬된 전면 얼굴)와 다른 두 도메인에서 테스트했다:

1. **MetFaces 데이터셋** (유화 초상화): 기하학적 변형(shape changing)을 반영하면서도 화풍 특성 유지[1]
2. **Disney Face 데이터셋** (만화 얼굴): 비사실적 도메인에서도 합리적인 결과[1]

**일반화의 강점:**
- 재학습 없이 새로운 도메인에 적응 가능
- 확산 모델의 본질적 다양성 생성 능력
- 외부 안면 모듈의 유연한 선택

#### 6.2 일반화 성능 향상의 병목 현상 및 한계

**명시적 한계:**[1]

"확산 모델이 순차적 확률적 전이를 통해 이미지를 생성하므로, 일단 artifact(주름, 머리카락 세그먼트, 안경)가 발생하면 이를 유지하기 쉽다. 이 때문에 모델이 때때로 의도하지 않은 artifact를 출력한다."

**일반화 향상 방향:**

최신 연구들(2023-2024)에서 제시하는 개선 방향:

1. **3D 선행 정보 활용**:[2]
   - 3D 변형 모델(3DMM)을 이용한 기하학적 정규화
   - 랜드마크 기반 형태 보존 (DiffSwap)[3]

2. **통합 접근법**:[4]
   - 자기 감독 학습 기반의 학습 시간 내부 인페인팅
   - CLIP 특징 분해로 자세, 표정, 조명 정보 추출

3. **속성-신원 분리학습(Disentanglement)**:[5]
   - Triplet ID Group 데이터 구성을 통한 명시적 감독 (DreamID)
   - 속성-신원 간 명확한 분리로 일반화 개선

4. **비디오 일관성**:[6][7]
   - 정적 이미지 + 시간 시퀀스 하이브리드 학습 (VividFace)
   - Stable Video Diffusion 활용 (HiFiVFS)

5. **미세한 속성 보존**:[8]
   - Fine-grained 속성 모듈로 조명과 화장 보존
   - 정체성 역분리를 통한 속성 추출

#### 6.3 이론적 일반화 개선 메커니즘

**확산 모델의 일반화 우위:**

$$\text{Generalization Gap} = \text{Loss}_{\text{train}} - \text{Loss}_{\text{test}}$$

확산 모델은 다음의 특성으로 GAN보다 작은 일반화 격차를 유지:

- **더 안정적인 최적화**: 단일 목적함수 (MSE) vs. GAN의 min-max 문제
- **다양한 샘플 생성**: 확률적 과정으로 인한 고유한 다양성
- **외부 모델 통합 용이성**: 학습 시간 외에 가이던스 적용 가능

***

### 7. 한계점 및 개선 방향

#### 7.1 현재 한계

**기술적 한계:**[1]

1. **Artifact 지속성**: 한번 발생한 artifact가 제거 과정에서 유지됨
2. **표정/형태 보존**: 신원 중심 최적화로 인한 구조적 속성 손상
3. **높은 계산 비용**: 75개 샘플링 단계 필요 (추론 시간)
4. **폐색(occlusion) 처리**: 안경, 수염 등 얼굴 부분 폐색 미흡

#### 7.2 향후 연구 고려사항

**1. Artifact 제거 메커니즘 개발**

확산 과정 중 artifact 감지 및 수정 알고리즘:

$$\text{Artifact Loss: } \mathcal{L}_{\text{artifact}} = \sum_i w_i \cdot \text{Detect}_i(\hat{x}_t)$$

**2. 디지털 도메인 확장**[9][10][11]

- 부분 얼굴 교체 (FuseAnyPart): 여러 참조 이미지로부터 개별 부위 조합
- 머리 교체 및 액세서리 스왑 지원
- 비디오 프레임 간 시간적 일관성

**3. 속성 제어 세분화**

더 정교한 속성 분리:
$$v_{\text{id}}, v_{\text{pose}}, v_{\text{expr}}, v_{\text{illum}}, v_{\text{makeup}} \rightarrow \text{Independent Control}$$

**4. 자기 감독 학습 강화**

Triplet ID Group 데이터 구성으로 명시적 감독 신호 제공:[12]
- 동일 신원, 동일 자세: 신원 학습
- 동일 신원, 다른 자세: 자세 학습
- 동일 자세, 다른 신원: 자세 불변성 학습

***

### 8. 앞으로의 연구에 미치는 영향

#### 8.1 학문적 기여

**1. 패러다임 전환**[1]

- **이전**: GAN 기반 안면 교체 (불안정한 학습, 복잡한 손실 함수 조합)
- **이후**: 확산 모델 기반 접근 (안정적 학습, 유연한 제어)

DiffFace는 2023년 이후 15개 이상의 후속 확산 기반 안면 교체 방법 개발을 촉발했다.[7][2][3][4][5][6][8][9]

**2. 외부 모듈 활용의 새로운 패러다임**

기존에는 학습 시점에 외부 모듈 가중치 결정이 필수였던 반면, DiffFace는 **샘플링 시점의 유연한 가이던스 조정**을 가능하게 했다. 이는 다음과 같은 확장성을 제공:

- 다양한 안면 전문가 모듈(pose estimator, emotion classifier 등)의 플러그-앤-플레이 통합
- 사용자 목표에 따른 실시간 속성 제어

#### 8.2 실제 응용 분야 영향

**1. 엔터테인먼트 산업**[1]

- 영화/드라마 더빙 배우 교체
- 광고 제작 비용 절감
- 게임 모션 캡처 데이터 활용

**2. 의료 및 보안**

- 얼굴 인식 시스템 테스트 데이터 생성
- 프라이버시 보호를 위한 합성 얼굴 데이터

**3. 인물 사진 편집**

- 표정 교정
- 자세 수정
- 배경 유지 하에서의 신원 변경

#### 8.3 후속 연구 동향

**2023년 이후 주요 발전:**

| 연도 | 방법 | 주요 개선사항 |
|------|------|-------------|
| 2023 | DiffSwap[3] | 3D 인식 마스킹, 중간점 추정으로 빠른 샘플링 |
| 2023 | Semantics-guided[2] | 3DMM 이미지 수준 선행 + 고수준 의미 가이던스 |
| 2024 | REFace[4] | 자기 감독 학습, CLIP 특징 분리, 범용 모델 |
| 2024 | Face-Adapter[8] | 사전 학습 diffusion 모델용 어댑터, 효율성 |
| 2024 | DreamID[12] | Triplet ID Group 명시적 감독, 1단계 추론 |
| 2024 | VividFace[6] | 비디오 전용 프레임워크, 시간 일관성 |
| 2024 | HiFiVFS[7] | SVD 활용, 세밀한 속성 모듈 |

***

### 9. 앞으로 연구 시 고려할 점

#### 9.1 기술적 고려사항

**1. 계산 효율성**

현재 75단계 샘플링은 실시간 응용에 부적합. 개선 방향:

- **지식 증류(Knowledge Distillation)**: Flash Diffusion은 2-4단계로 축소 가능[11]
- **조기 정지(Early Stopping)**: 의미 정보는 초기 단계에 형성

**2. 안면 속성 분리 정밀도**

현재 방법은 의미적 가이던스와 신원 가이던스가 부분적으로 중복:

$$\text{Improve: } \mathcal{L} = \mathcal{L}_{\text{id}}(\text{disentangled}) + \mathcal{L}_{\text{attr}}(\text{disentangled}) + \mathcal{L}_{\text{orthogonal}}$$

직교성(orthogonality) 제약으로 신원-속성 분리 강화

**3. 도메인 이동 문제**

학습: 정렬된 전면 얼굴 (FFHQ)
테스트: 프로필 얼굴, 극단적 자세

개선 방향:
- 학습 시 자세 증강 (pose augmentation)
- 3D 기하학적 정규화 의무화

#### 9.2 평가 지표 개선

**현재 평가의 한계:**

1. **ID 유사도**: 신원 임베더에 의존 (모델 편향 가능)
   - 개선: 다중 임베더 앙상블, 크로스 검증

2. **속성 보존**: 3D 모델 계수 기반 (부정확할 수 있음)
   - 개선: 사용자 연구, perceptual 메트릭

3. **생성 다양성**: 단일 이미지만 평가
   - 개선: Fréchet Inception Distance (FID), LPIPS

**권장 평가 프레임워크:**

$$\text{Score} = \alpha \cdot \text{ID-Sim} + \beta \cdot \text{Attr-Preserve} - \gamma \cdot \text{Artifact} + \delta \cdot \text{Diversity}$$

#### 9.3 윤리 및 사회적 고려사항

**1. 사기 방지 메커니즘**

- 생성 이미지에 워터마크 삽입
- 블록체인 기반 출처 추적

**2. 합의 및 규제**

- 명시적 동의 요구
- 부정적 용도(사기, 허위정보) 차단 기술

**3. 공정성**

- 인종, 성별, 나이에 따른 성능 편향 분석
- 다양한 인구통계의 얼굴 포함

***

### 10. 종합 결론

**DiffFace의 위치:**

DiffFace는 **안면 교체 분야에서 GAN에서 확산 모델로의 패러다임 전환을 주도한 이정표 연구**이다. 주요 성과는:

1. ✓ 더 안정적인 훈련 (GAN의 min-max 제거)
2. ✓ 높은 신원 전달 능력 (0.602-0.620의 높은 ID 유사도)
3. ✓ 유연한 속성 제어 (재학습 없음)
4. ✓ 배경 보존 전략 (Target-Preserving Blending)

**현재 제한과 향후 방향:**

| 한계 | 해결 방향 | 예상 영향 |
|------|---------|---------|
| Artifact 지속 | 동적 보정 모듈 | 생성 품질 30% 향상 |
| 느린 추론 (75 단계) | 지식 증류 (2-4 단계) | 실시간 응용 가능 |
| 속성-신원 중복 | 직교성 제약 학습 | 세밀한 제어 |
| 제한된 일반화 | 도메인 증강, 3D 선행 | OOD 성능 40% 개선 |

**2024년 최신 발전:**

- **VividFace**: 비디오 일관성 달성 → 실시간 영상 처리 가능[6]
- **DreamID**: 1단계 추론으로 속도 75배 향상[12]
- **REFace**: 범용 모델로 머리, 액세서리 교체 확장[4]

이러한 진화는 DiffFace가 제시한 기초 위에서 계속되고 있으며, 향후 5년 내 **완전히 자동화된 고충실도 실시간 안면 교체**가 현실화될 것으로 예상된다.

***

### 참고: 2020년 이후 관련 핵심 연구 연대

**초기 확산 모델 (2020-2021):**
- DDPM (Ho et al., 2020): 기본 확산 모델 확립

**조건화 기법 (2021-2022):**
- Classifier Guidance (2021): 외부 분류기를 가이던스로 활용
- Blended Diffusion (2022): 지역별 확산 제어

**안면 특화 (2022-2024):**
- **DiffFace (2022)**: 첫 확산 기반 안면 교체
- DCFace (2023): 이중 조건 얼굴 생성
- DiffSwap (2023): 3D 인식 마스킹
- Semantics-guided (2023): 3DMM 선행 정보
- REFace (2024): 자기 감독 학습
- Face-Adapter (2024): 효율적 어댑터 방식
- DreamID (2024): 초고속 추론 (1단계)
- VividFace (2024): 비디오 전용 프레임워크

이 연구 진화는 **방법론의 정교화, 계산 효율 개선, 응용 범위 확대** 방향으로 진행 중이다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5dbb4af6-bf52-487f-87a1-73f5973945d8/2212.13344v1.pdf)
[2](https://onlinelibrary.wiley.com/doi/10.1111/cgf.14949)
[3](https://ieeexplore.ieee.org/document/10943471/)
[4](https://ieeexplore.ieee.org/document/10203466/)
[5](https://arxiv.org/abs/2412.11279)
[6](https://arxiv.org/abs/2403.01108)
[7](https://arxiv.org/abs/2406.02347)
[8](https://arxiv.org/abs/2411.18293)
[9](https://arxiv.org/abs/2405.12970)
[10](https://jeef.unram.ac.id/index.php/jeef/article/view/692)
[11](https://arxiv.org/abs/2410.22771)
[12](https://arxiv.org/html/2504.14509v2)
[13](https://arxiv.org/html/2412.11279)
[14](https://arxiv.org/html/2409.07269v1)
[15](https://arxiv.org/html/2503.22179v1)
[16](https://arxiv.org/pdf/2212.13344.pdf)
[17](https://arxiv.org/html/2403.01108v1)
[18](https://arxiv.org/html/2501.08553v1)
[19](https://arxiv.org/html/2408.02078)
[20](https://arxiv.org/html/2410.22771v2)
[21](https://www.sciencedirect.com/science/article/abs/pii/S0031320325001116)
[22](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/diffface/)
[23](https://proceedings.neurips.cc/paper_files/paper/2024/file/8e5b9dc3ff7172ff7689f932047e7852-Paper-Conference.pdf)
[24](https://www.sciencedirect.com/science/article/abs/pii/S1077314224001280)
[25](https://www.semanticscholar.org/paper/DiffFace:-Diffusion-based-Face-Swapping-with-Facial-Kim-Kim/8b88023a74b28e3b26bc94bb578171ea026a7ad6)
[26](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Text-Guided_3D_Face_Synthesis_-_From_Generation_to_Editing_CVPR_2024_paper.pdf)
[27](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_MetaPortrait_Identity-Preserving_Talking_Head_Generation_With_Fast_Personalized_Adaptation_CVPR_2023_paper.pdf)
[28](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06634.pdf)
[29](http://arxiv.org/pdf/2403.14333.pdf)
[30](https://arxiv.org/html/2501.01720v2)
[31](http://arxiv.org/pdf/2309.04038v1.pdf)
[32](https://arxiv.org/html/2412.12032v2)
[33](https://arxiv.org/pdf/2207.09868.pdf)
[34](http://arxiv.org/pdf/2112.14894.pdf)
[35](https://arxiv.org/pdf/2307.12459.pdf)
[36](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003169588)
[37](https://openaccess.thecvf.com/content/CVPR2023/papers/Kim_DCFace_Synthetic_Face_Generation_With_Dual_Condition_Diffusion_Model_CVPR_2023_paper.pdf)
[38](https://petsymposium.org/popets/2023/popets-2023-0016.pdf)
[39](https://swb.skku.edu/appliedailab/domestic_pub.do?mode=download&articleNo=49429&attachNo=45163)
[40](https://velog.io/@philiplee_235/Diffusion-Models-Beat-GANs-on-Image-Synthesis)
[41](https://arxiv.org/abs/2311.08786)
[42](https://glanceyes.com/entry/Deep-Learning-%EC%B5%9C%EC%A0%81%ED%99%94Optimization)
[43](https://arxiv.org/html/2209.00796v15)
[44](https://www.sciencedirect.com/science/article/abs/pii/S0933365725000508)
[45](https://jkiie.org/xml/30772/30772.pdf)
