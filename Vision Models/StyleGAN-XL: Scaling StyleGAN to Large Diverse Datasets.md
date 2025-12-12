# StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets

### 1. 핵심 주장 및 주요 기여

**StyleGAN-XL**은 기존 StyleGAN 기반 모델들이 ImageNet과 같은 **대규모 비정형 데이터셋에서 성능 저하**를 보이는 근본적인 문제를 해결한 논문이다. 저자들의 핵심 주장은 이전에 StyleGAN의 **제한적인 아키텍처 자체**가 다양한 데이터셋에 부적합하다고 생각했지만, 실제 문제는 **훈련 전략**에 있다는 것이다.[1]

주요 기여는 다음과 같다:[1]

- Projected GAN 패러다임을 활용한 StyleGAN3의 성공적인 확장
- 진행적 성장(progressive growing) 전략의 재도입으로 안정적인 고해상도 학습 달성
- CNN과 ViT를 결합한 다중 특성 네트워크의 효과 입증
- 분류기 유도(classifier guidance) 기법의 GAN 적용
- ImageNet에서 1024×1024 해상도의 최초 생성 달성
- BigGAN과 확산 모델을 능가하는 최신 수준의 성능 달성

***

### 2. 해결하고자 하는 문제 및 제안 방법

#### 2.1 핵심 문제

StyleGAN은 고해상도 얼굴 이미지 생성에서는 뛰어났지만, ImageNet 같은 다양한 데이터셋에서는 심각한 성능 저하를 보였다. 기존 연구에서는 StyleGAN의 **제한적인 아키텍처**가 근본 원인으로 의심되었으나, 저자들은 체계적 분석을 통해 **훈련 전략**의 문제임을 규명했다.[1]

#### 2.2 Projected GAN 기반 접근

제안된 기본 목표 함수는 다음과 같다:[1]

$$\min_G \max_{\{D_l\}} \sum_{l \in L} \left[ \mathbb{E}_x[\log D_l(P_l(x))] + \mathbb{E}_z[\log(1 - D_l(P_l(G(z))))] \right]$$

여기서:
- $\{P_l\}$: 특성 프로젝터 집합
- $F$: 사전학습된 특성 네트워크
- $D_l$: 다양한 특성 표현에 작동하는 독립적 판별자들
- CCM(Cross-Channel Mixing)과 CSM(Cross-Scale Mixing): 모드 붕괴 방지 메커니즘

#### 2.3 주요 구성 요소

**A. 정규화 및 아키텍처 적응**[1]

- StyleGAN3-T(변환 등변) 레이어 선택
- 스타일 믹싱과 경로 길이 정규화 비활성화
- 경로 길이 정규화는 모델이 충분히 학습한 후(200k 이미지)만 적용
- 초기 200k 이미지 동안 σ=2 픽셀의 가우시안 필터로 이미지 블러 처리

**B. 저차원 잠재 공간**[1]

StyleGAN의 고차원 잠재 코드 크기를 개선:

$$z \in \mathbb{R}^{64} \text{ (기존: } \mathbb{R}^{512}) $$
$$w \in \mathbb{R}^{512} \text{ (유지)}$$

이 감소는 자연 이미지의 내재적 차원(ImageNet ≈ 40)을 반영하며, 매핑 네트워크의 초기 적응 속도 향상을 가능하게 한다.

**C. 사전학습된 클래스 임베딩**[1]

클래스 붕괴 방지를 위해:

$$e_c = \text{Linear}(\text{PooledFeatures}(c))$$

EfficientNet-lite0에서 추출한 클래스별 평균 특성을 임베딩 초기값으로 사용하고, GAN 훈련 중 최적화하여 다양성 향상(recall: 0.004 → 0.15).

#### 2.4 진행적 성장 재도입

progressively growing 전략의 재설계:[1]

- 시작 해상도: 16² 픽셀, 11개 레이어
- 각 단계에서 2개 레이어 제거, 7개 레이어 추가
- 최종 단계(1024²): 총 39개 레이어
- 고정된 일정이 아니라 FID 수렴 기준으로 단계 전환
- 배치 크기: 낮은 해상도(16², 32²)에서 2048, 고해상도에서 128~256

이 방식은 StyleGAN3의 등변성 특성을 활용하여 앨리어싱 방지:

$$\text{EQ-T}_{Config-C} = 55 \text{ vs } \text{EQ-T}_{Config-D} = 48$$

(높을수록 등변성 우수, 앨리어싱 없음 조건: EQ-T ∼ 15 이상)

#### 2.5 다중 특성 네트워크 활용

CNN과 ViT의 상보적 특성 활용:[1]

| 특성 네트워크 | FID ↓ | IS ↑ |
|:---:|:---:|:---:|
| EfficientNet만 | 19.51 | 35.74 |
| EfficientNet + ResNet50 | 16.16 | 49.13 |
| EfficientNet + DeiT-M | 12.43 | 56.72 |

CNN은 지역적 특성을, ViT는 전역적 의존성을 포착하여 상호 보완적 효과 창출.

#### 2.6 분류기 유도

확산 모델에서 도입된 기법을 GAN에 적용:[1]

$$\mathcal{L}_{CE} = -\sum_{i=0}^{C} c_i \log(\text{CLF}(x_i))$$

$$\mathcal{L}_{total} = \mathcal{L}_{GAN} + \lambda \mathcal{L}_{CE}$$

여기서 $\lambda = 8$ (경험적으로 최적값), 32×32 이상의 해상도에서만 적용하여 모드 붕괴 방지.

결과: IS 56.72 → 86.21 (큰 폭의 샘플 품질 향상)

***

### 3. 모델 구조 및 성능 향상

#### 3.1 StyleGAN-XL 아키텍처

**매핑 네트워크 ($G_m$)**[1]
- 입력: 잠재 코드 $z \in \mathbb{R}^{64}$, 클래스 임베딩 $e_c$
- 출력: 스타일 코드 $w \in \mathbb{R}^{512}$
- 구조: 8계층 MLP with ReLU 활성화

**합성 네트워크 ($G_s$)**[1]
- Fourier 특성 기반 공간 맵 초기화
- 14개의 컨볼루션 레이어(기본), 고해상도 시 최대 39개
- 각 레이어마다 스타일 조정 (AdaIN)
- 앨리어싱 방지 필터 설계:
  - 컷오프 및 스톱밴드 주파수: 네트워크 깊이에 따라 기하급수적 진행
  - 마지막 2개 레이어: 임계 샘플링 (컷오프 = 대역폭)

**특성 프로젝터**[1]
- $F_1$: EfficientNet-lite0 (분류 사전학습)
- $F_2$: DeiT-base (분류 사전학습)
- CCM: 1×1 컨볼루션 기반 무작위 채널 혼합
- CSM: 3×3 컨볼루션과 이중선형 업샘플링 기반 다중 스케일 혼합
- 8개의 독립 판별자: 4개 해상도 수준에서 각각 2개 연산

#### 3.2 성능 지표 비교

**ImageNet 1024² 해상도에서의 최종 성능**[1]

| 모델 | FID ↓ | IS ↑ | sFID ↓ | rFID ↓ | Precision ↑ | Recall ↑ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| BigGAN | 8.43 | 177.90 | 8.13 | 312.00 | 0.88 | 0.29 |
| ADM-G-U | 3.85 | 221.72 | 5.86 | 210.83 | 0.84 | 0.53 |
| **StyleGAN-XL** | **2.41** | **267.75** | **4.06** | **51.54** | **0.77** | **0.52** |

주요 성과:
- FID: 37% 개선 (ADM-G-U 대비)
- IS: 20.8% 개선
- 미계획 특성 공간에서의 성능(rFID): 큰 폭의 개선
- 공간 구조 평가(sFID): 최고 성능

#### 3.3 훈련 효율성

StyleGAN-XL은 5122 해상도에서 ADM 대비 **478배 빠른 훈련** 달성:[1]

- 현재 state-of-the-art 도달: 2 V100-days
- ADM 도달: 1914 V100-days
- 매개변수: StyleGAN3의 **3배 증가**에도 불구하고 더 효율적

**추론 속도 비교** (배치 크기 1, V100 기준)[1]

| 해상도 | ADM | StyleGAN-XL | 속도 향상 |
|:---:|:---:|:---:|:---:|
| 128² | 27.07s | 0.05s | **541배** |
| 256² | 40.26s | 0.07s | **575배** |
| 512² | 91.54s | 0.10s | **916배** |

***

### 4. 모델 일반화 성능 향상 가능성

#### 4.1 모드 다양성 향상

**클래스당 다양성 개선**[1]

Config-C의 문제점: 클래스 임베딩 붕괴 → recall: 0.004 (심각한 모드 붕괴)

사전학습된 임베딩 도입 후: recall: 0.15

이는 다음과 같은 메커니즘을 통해 달성:[1]

1. EfficientNet-lite0 특성 추출
2. 클래스별 평균 풀링
3. 선형 투영으로 $z$와 크기 균형
4. GAN 훈련 중 임베딩 및 투영 최적화

결과: 클래스당 **다양한 이미지 생성** 가능 (삽입도: 0.004 → 0.15, 37.5배 향상)

#### 4.2 고해상도 확장성

진행적 성장 전략의 핵심 이점:[1]

- **단계적 난이도 증가**: 저해상도(16²)에서 고해상도(1024²)로 점진적 확장
- **과적합 방지**: 각 단계에서 충분히 수렴할 때까지 학습
- **앨리어싱 최소화**: 레이어 추가 시 기하급수적 필터 설계 조정

1024² 단일 V100-day로 FID 2.8 달성 (이전 최고 기록 5122에서 3.85 vs StyleGAN-XL 2.41)

#### 4.3 도메인 외 이미지 적응

**Pivotal Tuning Inversion (PTI) 결합**[1]

초기 반전: 잠재 공간 최적화로 기본 재구성

$$\mathcal{L}_{inversion} = \|I_{target} - G_s(w_{opt})\|_{LPIPS}$$

PTI 정규화: 생성기 미세조정으로 더 정확한 반전

$$\mathcal{L}_{PTI} = \|I_{target} - G'(w)\| + \lambda_{reg} \cdot \|G' - G\|$$

결과: ImageNet 검증 세트에서 **PSNR 13.5, FID 21.7** 달성
- BigGAN 대비: PSNR 10.8, FID 47.5
- 픽셀 재구성과 의미적 근접도 모두 우수

#### 4.4 다양한 데이터셋에서의 일반화

**단일 모달 데이터셋 성능**[1]

| 데이터셋 | 모델 | FID | 주요 특성 |
|:---:|:---:|:---:|:---|
| FFHQ 10² | StyleGAN-XL | 2.02 | 얼굴 도메인에서도 우수 |
| Pokémon 10² | StyleGAN-XL | 25.47 | 예술적/만화 이미지도 처리 |

StyleGAN-XL은:
- 고도로 정제된 얼굴 데이터셋(FFHQ): StyleGAN3 대비 28% FID 개선
- 예술적 데이터셋(Pokémon): 기존 방법 대비 27% 개선

이는 아키텍처가 **다양한 데이터 도메인에 본질적으로 호환성 있음**을 시사한다.

#### 4.5 편집 가능성과 조작

**잠재 공간 편집 기능**[1]

1. **이미지 반전**: 기본 잠재 최적화로 만족스러운 결과
2. **의미론적 조작**: GANspace 기법으로 발견된 방향 활용
3. **스타일 혼합**: 두 개 이미지의 스타일 코드를 다양한 레이어에서 혼합
4. **반사실적 이미지 생성**: 서로 다른 클래스 간 스타일 혼합

제한사항: StyleGAN3의 등변성 추구로 인한 **편집 가능성 감소** (StyleGAN2 대비)
- 해결책: StyleSpace 및 StyleMC 기법과 결합으로 고품질 편집 달성

***

### 5. 모델의 한계

#### 5.1 아키텍처적 한계[1]

1. **모델 크기**: StyleGAN3의 3배 증가 → 파인튜닝 시작점으로 사용할 때 계산 오버헤드 증가
   - 해결책: GAN 증류(distillation) 기법 활용

2. **편집 가능성 감소**: StyleGAN3의 등변성을 위한 설계가 W 공간에서의 편집 어려움 초래
   - 더 높은 편집 품질이 필요한 경우: W+ 공간 사용 필요 (편집 가능성↑ vs 재구성 정확도↓)

#### 5.2 데이터셋 제약[1]

- 더 크고 다양한 고해상도 데이터셋 부족
- 현재 대규모 고해상도 데이터셋은 단일 객체 클래스 또는 반복적 이미지로 제한
- 메가픽셀 규모의 완전히 다양한 데이터셋 부재

#### 5.3 성능 메트릭 제약[1]

- Truncation trick이 높은 다양성 데이터에서 precision 향상에 미치는 영향 미미
- 새로운 truncation 방법 개발 필요

***

### 6. 2020년 이후 관련 최신 연구 탐색

#### 6.1 StyleGAN 기반 확장 연구

**StyleGAN-T (2023)**[2]
- 대규모 텍스트-이미지 합성을 위한 StyleGAN 확장
- StyleGAN2 백본 채택 (텍스트 조건화 개선)
- BigGAN, Imagen 등 대규모 데이터셋에서 경쟁력 있는 성능
- 추론 속도: 초당 10프레임(FPS) 달성

**GigaGAN (2023)**[3][4]
- StyleGAN을 10억 매개변수 규모로 확장
- LAION 2B 같은 대규모 인터넷 데이터에서 학습
- 512×512 해상도에서 0.13초 내 생성 (확산 모델보다 **수백배 빠름**)
- 16메가픽셀(4096×4096) 이미지 3.66초 내 생성
- **다중 스케일 훈련**: 저해상도 생성 블록의 파라미터 효율성 향상
- 텍스트-이미지 정렬 및 저주파 디테일 개선 스킴 도입

특징: StyleGAN-XL의 문제점(다양한 데이터에서의 불안정성)을 **다중 스케일 훈련**으로 해결

#### 6.2 Vision Transformer 기반 생성 모델

**DiffiT (2023)**[5]
- Diffusion Vision Transformer: ViT 기반 확산 모델
- 시간 의존적 멀티헤드 자기주의(TMSA) 메커니즘
- ImageNet 256 해상도에서 **FID 1.73** 달성
- 파라미터 효율: MDT 대비 19.85% 감소, DiT 대비 16.88% 감소

**Latte (ViT 기반 비디오 생성)**[6]
- 효율적인 장기 문맥 모델링: 3D 컨볼루션의 메모리 병목 회피
- FaceForensics: FVD 27.08, UCF101: FVD 333.61 (SOTA)
- 선형 확장성: 더 큰 모델(Latte-XL, 673M 파라미터)이 일관된 품질 향상

#### 6.3 확산 모델의 우위

**현재 상황 (2024-2025)**[7]
- 확산 모델이 이미지 생성의 새로운 표준으로 확립
- DALL-E 3, Stable Diffusion, Midjourney 등 주류 모델
- 장점: 높은 다양성, 제어 가능성
- 단점: 느린 추론 (GAN의 수백배)

StyleGAN-XL/GigaGAN의 **고속 추론**이 여전히 강점:
- 실시간 응용(인터랙티브 편집, 제어되는 생성)에서의 우위

#### 6.4 최신 GAN 개선 연구

**텍스트-이미지 합성 안정화**[8][9][10]
- EfficientCLIP-GAN: CLIP 통합으로 100배 빠른 생성 실현
- 자기조절화 특성 융합 GAN: 단일 생성기-판별자로 안정적 훈련
- DSE-GAN, MARS: 동적 의미론 진화로 단계별 텍스트 특성 적응

**생성 모델의 일반화**[11]
- 잠재 공간의 모델가능성(modelability) vs 재구성 품질 트레이드오프
- 대역폭-왜곡-모델가능성 3원 트레이드오프 확립
- 효율적 잠재 표현: 높은 해상도 생성을 위한 공간적 다운샘플링 증대 추세(32× → 64×)

#### 6.5 수렴성 및 안정성 연구

**GAN 훈련 안정화 기법 (2022-2025)**[12]
- 두 시간대 업데이트 규칙(TTUR): 생성기와 판별자에 다른 학습률
- Spectral Normalization: 판별자 기울기 조절
- Gradient Penalty (R1): 실데이터에서만 적용하여 국소 수렴 보장
- Differentiable Augmentation: 데이터 효율성 및 안정성 향상

**최신 수렴 분석 (2025)**[13]
- 공진화 생성 모델의 수렴 동역학
- 안정화 전략의 수학적 분석 및 증명

***

### 7. 해당 논문이 앞으로의 연구에 미치는 영향

#### 7.1 GAN 재평가와 복권

StyleGAN-XL은 **GAN의 확장성 한계**를 깨뜨린 이정표이다.[3][1]

**이전 신념**: StyleGAN은 본질적으로 단일/제한된 도메인에만 적합
**현실**: 훈련 전략 개선으로 대규모 다양한 데이터셋 처리 가능

이를 통해:
1. GigaGAN(2023)으로 이어지는 **"GAN 재부활" 열풍** 촉발
2. GANs vs 확산 모델의 **공존적 에코시스템** 형성
   - GANs: 고속 추론, 실시간 응용
   - 확산 모델: 높은 품질, 제어 가능성

#### 7.2 기울기 흐름 최적화 원리 확립

논문의 핵심 발견들이 **이후 GAN 설계의 원칙**으로 정립:[12][1]

| 원칙 | StyleGAN-XL의 적용 | 의의 |
|:---:|:---|:---|
| 저차원 잠재 공간 | z ∈ ℝ⁶⁴ (대 512) | 초기 학습 안정성 ↑ |
| 진행적 성장 | 단계별 해상도 증가 | 고해상도 학습 가능 |
| 다중 판별자 | 8개 독립 판별자 | 모드 붕괴 방지 |
| 사전학습 특성 | CNN+ViT 결합 | 상보적 정보 활용 |

#### 7.3 "훈련 > 아키텍처" 패러다임 확산

이 논문은 **생성 모델 개발의 철학적 전환** 가져옴:[3][1]

**기존 사고**: 더 나은 결과 = 더 나은 아키텍처
**새로운 사고**: 더 나은 결과 = 더 나은 훈련 전략

이 인식이 2023년 이후 연구에 반영:
- GigaGAN의 "다중 스케일 훈련" 도입
- 확산 모델의 "캐스케이드 두 단계 모델" 채택
- 메타러닝, 강화학습 기반 하이퍼파라미터 자동 조정 연구

#### 7.4 다중 특성 네트워크의 표준화

CNN과 ViT의 상보성 발견:[1]

이 결과는:
1. **멀티모달 특성 추출**의 중요성 확립
2. DiffiT(2023)에서 ViT 기반 확산 모델로 확대
3. Latte(ViT 기반 비디오)에서 계속 활용
4. 향후 대규모 생성 모델의 **표준 구성**으로 채택

***

### 8. 앞으로의 연구 시 고려할 점

#### 8.1 아키텍처-훈련 코-설계의 중요성

**제언**: 새로운 생성 모델 개발 시 다음 순서 권장:[3][1]

1. **기본 아키텍처** 설계: 이론적 근거 기반
2. **훈련 전략** 최적화: 안정성, 수렴성, 효율성 중심
3. **하이퍼파라미터** 튜닝: 체계적 탐색
4. 아키텍처 개선이 효과 미미할 때만 고려

StyleGAN-XL은 step 2에서 큰 이득을 얻었고, 이것이 GigaGAN의 설계에도 반영됨.

#### 8.2 다중 손실 함수의 균형

StyleGAN-XL이 사용한 손실 함수:[1]

$$\mathcal{L}_{total} = \mathcal{L}_{GAN} + \lambda_{CE} \mathcal{L}_{CE} + \lambda_{path} \mathcal{L}_{path}$$

**주의점**:
- 각 손실의 가중치 $\lambda$는 **데이터셋 특성에 민감**
- 경로 길이 정규화는 **모델이 충분히 학습한 후**에만 도입 (초기에는 모드 붕괴 초래)
- 분류기 유도는 **일정 해상도 이상**에서만 효과적

#### 8.3 진행적 성장의 재평가

StyleGAN3 논문에서 버려진 기법이 다시 부활:[1]

**핵심 조건**:
1. **앨리어싱 방지 필터**: 고정 필터보다 깊이에 따라 조정되는 필터 필수
2. **단계별 종료 조건**: 고정 스케줄보다 FID 수렴 기준
3. **다중 스케일 훈련**: 각 해상도에서 특성 네트워크의 전체 용량 활용

앞으로의 연구에서:
- 다른 생성 모델(자동회귀, 확산)에도 **진행적 학습** 원리 적용 검토
- 비전 트랜스포머 기반 모델에서 "계층적 학습" 재설계

#### 8.4 잠재 공간 설계의 재고

저차원 잠재 공간의 발견은 중요한 통찰:[1]

$$\text{차원 감소: } 512 \to 64$$

**후속 고려사항**:
1. 최적 잠재 공간 차원의 **데이터셋 의존성** 분석
2. 내재적 차원(Intrinsic Dimension)과 학습 안정성의 관계 규명
3. 적응적 차원 조정: 훈련 진행에 따라 동적으로 차원 증가

Latent Space 이론에 따르면:
- **재구성 품질** vs **모델가능성** 트레이드오프 존재
- StyleGAN-XL의 저차원 선택은 **모델가능성 최적화** 전략

#### 8.5 도메인 일반화를 위한 데이터 전처리

ImageNet 1024² 생성을 위한 슈퍼해상도 전처리:[1]

**고려점**:
1. 전처리 모델 선택의 영향 분석
2. 원본 저해상도 특성 보존 vs 복원 품질 트레이드오프
3. 합성-실제 이미지 도메인 갭 분석

현재 한계:
- SwinIR을 통한 업샘플링이 50배 이상 느림
- 생성 모델 자체의 상향 확대 역량 강화 필요

#### 8.6 평가 지표의 재검토

StyleGAN-XL에서 제안한 random-FID (rFID)의 의의:[1]

$$rFID = \text{Fréchet Distance in random feature space}$$

**배경**: FID, IS가 사전학습 분류기에 의존 → 편향 가능성

**개선 방향**:
1. 모델 무관(model-agnostic) 평가 지표 개발
2. 인간 평가와의 상관성 검증
3. 다양한 작업(편집, 반전)에 특화된 지표 정의

#### 8.7 편집 가능성의 새로운 접근

StyleGAN-XL의 제약: 등변성으로 인한 **W 공간 편집 어려움**[1]

**해결책 탐색**:
1. StyleSpace, StyleMC: 더 높은 차원의 공간에서 편집
2. **의미론적 벡터 학습**: 사전학습 모델(CLIP 등)의 의미론적 방향 활용
3. 단계별 편집: 일부 레이어만 수정하는 **로컬 편집** 강화

#### 8.8 실제 응용을 위한 고려사항

고속 추론의 장점을 활용한 응용:[8][1]

1. **실시간 인터랙티브 이미지 생성**: 사용자 입력에 즉시 반응
2. **모바일/엣지 배포**: 추론 속도로 인해 CPU에서도 가능 (EfficientCLIP-GAN)
3. **스트리밍 콘텐츠 생성**: 비디오 프레임 실시간 합성

***

### 결론

**StyleGAN-XL**은 단순한 성능 개선을 넘어 생성 모델 연구의 **패러다임 전환**을 가져왔다. 아키텍처 제약으로 생각했던 문제가 실은 훈련 전략의 문제였다는 발견, Projected GAN과 진행적 성장의 재조명, CNN-ViT 다중 특성의 상보성 활용 등은 모두 후속 연구에 직접적 영향을 미쳤다.

2023년 이후 GigaGAN의 등장, DiffiT와 Latte의 ViT 기반 확산 모델, EfficientCLIP-GAN의 효율적 설계 등은 StyleGAN-XL이 열어준 방향을 다양하게 확대하고 있다. 특히 **"훈련 > 아키텍처"** 원칙과 **"다중 특성 네트워크"** 개념은 이제 생성 모델의 표준적 설계 철학으로 자리잡았다.

다만 더 큰 고해상도 데이터셋의 부족, 다양성-품질 트레이드오프, 편집 가능성 제약 등의 문제는 여전히 해결해야 할 과제로 남아 있으며, 이들을 해결하기 위해서는 본 논문에서 시사한 **체계적인 훈련 전략 설계**, **잠재 공간의 신중한 설계**, 그리고 **다중 모달리티 통합**의 원칙이 지속적으로 적용되어야 할 것이다.

***

### 참고

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c069bdbe-72d3-4917-a141-1375082383b6/2202.00273v2.pdf)
[2](https://proceedings.mlr.press/v202/sauer23a/sauer23a.pdf)
[3](https://ieeexplore.ieee.org/document/10205294/)
[4](https://openaccess.thecvf.com/content/CVPR2023/papers/Kang_Scaling_Up_GANs_for_Text-to-Image_Synthesis_CVPR_2023_paper.pdf)
[5](https://arxiv.org/abs/2312.02139)
[6](https://blog.roboflow.com/vision-transformers/)
[7](https://learnopencv.com/image-generation-using-diffusion-models/)
[8](https://ieeexplore.ieee.org/document/10973999/)
[9](https://ieeexplore.ieee.org/document/10526149/)
[10](https://arxiv.org/abs/2407.07614)
[11](https://sander.ai/2025/04/15/latents.html)
[12](https://davidleonfdez.github.io/gan/2022/05/17/gan-convergence-stability.html)
[13](https://arxiv.org/abs/2503.08117)
[14](https://www.mdpi.com/2076-0817/14/6/551)
[15](http://pubs.rsna.org/doi/10.1148/radiol.233529)
[16](https://www.scivisionpub.com/pdfs/efficacy-of-cicatricort-in-the-prevention-of-keloids-in-postoperative-plastic-surgery-a-retrospective-comparative-study-20232025-3904.pdf)
[17](https://ieeexplore.ieee.org/document/11147740/)
[18](https://www.rusjel.ru/jour/article/view/2513)
[19](http://pubs.rsna.org/doi/10.1148/radiol.250617)
[20](https://link.springer.com/10.1007/s11207-024-02307-w)
[21](https://ojs.zu.edu.pk/pjmd/article/view/3324)
[22](https://arxiv.org/abs/2503.18711)
[23](https://journals.lww.com/10.4103/jpcs.jpcs_49_25)
[24](http://arxiv.org/pdf/2301.09515.pdf)
[25](https://arxiv.org/abs/2112.02236v3)
[26](https://arxiv.org/abs/2104.04767)
[27](http://arxiv.org/pdf/2202.12211.pdf)
[28](https://arxiv.org/pdf/2107.09700.pdf)
[29](https://arxiv.org/html/2409.11010)
[30](https://arxiv.org/html/2410.06104v1)
[31](https://arxiv.org/ftp/arxiv/papers/2205/2205.12512.pdf)
[32](https://digitaldefynd.com/IQ/pros-cons-of-stylegan/)
[33](https://viso.ai/deep-learning/vision-transformer-vit/)
[34](https://www.mathworks.com/help/deeplearning/ug/generate-images-using-diffusion.html)
[35](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/stylegan-t/)
[36](https://arxiv.org/abs/2303.07909)
[37](https://www.emergentmind.com/topics/stylegan3)
[38](https://www.nature.com/articles/s41598-023-41484-9)
[39](https://ieeexplore.ieee.org/document/10118052/)
[40](https://ieeexplore.ieee.org/document/10767714/)
[41](https://ieeexplore.ieee.org/document/10313436/)
[42](https://dl.acm.org/doi/10.1145/3587828.3587852)
[43](https://ieeexplore.ieee.org/document/10423780/)
[44](https://arxiv.org/pdf/2209.01339.pdf)
[45](https://arxiv.org/pdf/2406.18547.pdf)
[46](http://arxiv.org/pdf/1812.04948.pdf)
[47](https://arxiv.org/abs/2303.05511)
[48](https://arxiv.org/pdf/2204.07513.pdf)
[49](https://arxiv.org/abs/1710.10916)
[50](https://arxiv.org/pdf/2305.15421.pdf)
[51](https://www.sciencedirect.com/science/article/pii/S0006349524001395)
[52](https://www.peakermap.com/blogs/news/stabilizing-and-converging-gan-training-a-technological-challenge)
[53](https://www.ibm.com/think/topics/latent-space)
[54](https://www.sciencedirect.com/science/article/abs/pii/S009784932300064X)
[55](https://en.wikipedia.org/wiki/Latent_space)
[56](https://liner.com/review/scaling-up-gans-for-texttoimage-synthesis)
