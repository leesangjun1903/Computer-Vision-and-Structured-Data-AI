# Pivotal Tuning for Latent-based Editing of Real Images

### 1. 핵심 주장 및 주요 기여

**Pivotal Tuning Inversion (PTI)** 논문의 핵심 주장은 **StyleGAN의 고정된 생성기를 약간 조정함으로써 실사 이미지 편집 시 발생하는 왜곡-편집성(distortion-editability) 트레이드오프를 해결할 수 있다**는 것입니다.[1]

주요 기여는 다음과 같습니다:[1]

1. **문제 해결**: StyleGAN의 고유한 특성인 왜곡-편집성 트레이드오프를 극복하여, 도메인 외(out-of-domain) 이미지에서도 높은 품질의 신원 보존 편집을 가능하게 함

2. **혁신적 방법론**: 초기 역변환(inversion)된 잠재 코드를 중심점(pivot)으로 사용하여 생성기를 미세 조정하는 **Pivotal Tuning**이라는 간단하면서도 효과적인 기법 제시

3. **정규화 기법**: 지역성 정규화(locality regularization)를 통해 조정의 영향을 국소적으로 제한하여, 다른 신원에 미치는 부정적 영향을 최소화

4. **다중 신원 처리**: 단일 생성기로 여러 신원을 동시에 처리할 수 있음을 입증

5. **정성적·정량적 검증**: 기존 방법들(SG2, SG2 W+, e4e)을 능가하는 reconstruction 및 editing 품질을 달성

***

### 2. 상세한 기술 설명

#### 2.1 해결하고자 하는 문제

StyleGAN 기반 이미지 편집은 두 가지 주요 문제를 직면합니다:[1]

**문제의 근본 원인**: StyleGAN의 잠재 공간은 다음과 같은 특성을 가집니다:
- **W 공간**: 높은 편집성(editability)을 제공하지만, W+ 공간보다 표현력이 부족하여 실사 이미지 역변환 시 왜곡(distortion)이 발생
- **W+ 공간**: 확장된 표현력으로 왜곡을 최소화하지만, W 공간보다 편집성이 낮음

**트레이드오프의 현상**:[1]
- W 기반 역변환: 신원이 바뀌거나 부자연스러운 모양 발생
- W+ 기반 역변환: 신원은 보존되지만 편집 효과가 미약함 (회전이 작고 스마일 변화가 제한적)

***

#### 2.2 제안하는 방법 및 수식

**2단계 방식의 핵심 개념:**[1]

**Step 1: 역변환 (Inversion)**

초기 역변환은 다음과 같이 정의됩니다:

$$w_p, n = \arg\min_{w,n} \mathcal{L}_{LPIPS}(x, G(w, n; \theta)) + \lambda_n \mathcal{L}_n(n)$$

여기서:
- $x$: 입력 이미지
- $w_p$: 중심점(pivot) 잠재 코드 (W 공간)
- $n$: 노이즈 벡터
- $G$: 생성기 (고정된 가중치 $\theta$)
- $\mathcal{L}_{LPIPS}$: 지각 손실(perceptual loss)
- $\mathcal{L}_n$: 노이즈 정규화항

**Step 2: Pivotal Tuning**

생성기를 미세 조정하기 위한 손실함수:

$$\mathcal{L}_{pt} = \mathcal{L}_{LPIPS}(x, x_p) + \lambda_{L2} \mathcal{L}_{L2}(x, x_p)$$

여기서:
- $x_p = G(w_p; \theta^*)$: 조정된 생성기로 생성된 이미지
- $\theta^*$: 미세 조정된 생성기 가중치
- $w_p$: 고정된 중심점 코드

**다중 신원 확장:**

N개의 이미지에 대해:

```math
\mathcal{L}_{pt}=\frac{1}{N}\sum _{i=1}^{N}\left(\mathcal{L}_{LPIPS}(x_{i},x_{p_{i}})+\lambda _{L2}\mathcal{L}_{L2}(x_{i},x_{p_{i}})\right)
```

여기서 $\(x_{p,i}=G(w_{i};\theta ^{*})\)$

**지역성 정규화 (Locality Regularization):**[1]

핵심 아이디어는 중심점 근처의 코드만 영향을 받도록 제한하는 것입니다.

임의의 정규 분포에서 샘플링한 $z$에 대해:
- $w_z = f(z)$ (매핑 네트워크)
- 보간된 코드: $w_r = w_p + \alpha(w_z - w_p)$

정규화 손실:

$$\mathcal{L}_R = \mathcal{L}_{LPIPS}(X_r, \tilde{x}_r) + \lambda_R^{L2} \mathcal{L}_{L2}(X_r, \tilde{x}_r)$$

여기서:
- $X_r = G(w_r; \theta_0)$: 원래 생성기로 생성한 이미지
- $\tilde{x}_r = G(w_r; \theta^*)$: 조정된 생성기로 생성한 이미지

**최종 최적화 목표:**[1]

$$\theta^* = \arg\min_{\theta^*} \mathcal{L}_{pt} + \lambda_R \mathcal{L}_R$$

여기서 $\lambda_R$는 정규화 강도를 조절하는 하이퍼파라미터입니다.

***

#### 2.3 모델 구조

**전체 아키텍처 개요:**[1]

```
입력 이미지
    ↓
[Step 1: W-공간 역변환]
  - 최적화 기반 방식 (Karras et al. 2020)
  - 500ms 소요
    ↓
중심점 잠재 코드 (w_p)
    ↓
[Step 2: Pivotal Tuning]
  - LPIPS + L2 손실로 생성기 미세조정
  - 지역성 정규화 적용
  - 1-2분 소요
    ↓
조정된 생성기 (θ*)
    ↓
[Step 3: 편집]
  - GAN-Space, InterfaceGAN, StyleClip 등 활용
  - 편집 벡터 적용
    ↓
편집된 이미지 출력
```

**핵심 설계 선택:**[1]

1. **W 공간 선택**: 편집성이 높으므로 W+ 대신 W 공간에서 역변환
2. **중심점 고정**: 역변환된 코드를 고정하면 생성기만 조정하면 되어 최적화 효율 증대
3. **이중 손실 함수**: LPIPS(지각 손실)와 L2(픽셀 손실) 조합으로 재구성 품질 극대화
4. **적응적 정규화**: 임의 샘플링과 보간을 통해 중심점 근처 영향 제한

***

#### 2.4 성능 향상

**정량적 결과:**[1]

| 지표 | PTI (제안) | e4e | SG2 | SG2 W+ |
|------|----------|-----|-----|--------|
| LPIPS ↓ | 0.09 | 0.4 | 0.4 | 0.34 |
| MSE ↓ | 0.014 | 0.05 | 0.08 | 0.043 |
| MS SSIM ↓ | 0.21 | 0.38 | 0.38 | 0.3 |
| ID Similarity ↑ | 0.9 | 0.75 | 0.8 | 0.85 |

**편집 품질 평가:**[1]

포즈 편집에서:
- 편집 크기: 14.86도 (PTI) vs. 14.6 (e4e) vs. 15 (SG2) vs. 11.15 (SG2 W+)
- 단일 편집 후 신원 보존: 0.90 (PTI) vs. 0.79 (e4e) vs. 0.82 (SG2) vs. 0.85 (SG2 W+)
- 순차 편집 후 신원 보존: 0.82 (PTI) vs. 0.73 (e4e) vs. 0.78 (SG2) vs. 0.81 (SG2 W+)

**질적 결과:**[1]
- 무거운 메이크업, 복잡한 헤어스타일, 모자 등 도메인 외 이미지에서 우수한 성능
- 유명 인물 이미지에서 신원 보존하면서 자연스러운 편집 달성
- 다중 신원 처리에서 간섭 최소화

***

#### 2.5 한계점

**계산 비용:**[1]
- W 공간 역변환: 약 1분
- Pivotal Tuning: 약 1분 (정규화 포함 시 2분)
- 총 소요 시간: 약 3분/이미지
- e4e (인코더 기반)의 <1초 추론 시간에 비해 느림

**적응성 제한:**[1]
- 각 새로운 신원마다 개별 최적화 필요
- 실시간 애플리케이션에는 부적합
- 학습된 매퍼로 대체하는 것이 향후 과제

**생성기 수정의 위험성:**[1]
- 생성기 조정 시 대역폭 제약 존재
- 극단적인 도메인 외 이미지에는 여전히 한계

***

### 3. 모델의 일반화 성능 향상 가능성

#### 3.1 현재의 일반화 성능

**도메인 외(Out-of-Domain) 이미지 처리:**[1]

논문은 다음과 같은 도메인 외 사례에서 성공을 보여줍니다:
- **실제 유명인**: Serena Williams, Robert Downey Jr. 등
- **특수 메이크업**: 무거운 페이스페인팅, 색조 분장
- **특이한 헤어스타일**: 아프로, 비전통적 색상
- **헤드웨어**: 모자, 왕관, 선글라스
- **극단적 조명**: 그림자, 역광 상황
- **예술 작품**: 초상화, 조각 (MetFace 데이터셋)

**일반화 성능의 핵심 이유:**[1]

1. **지역성 정규화의 효과**: 중심점 근처의 변화만 허용하여 StyleGAN의 기본 구조 유지
2. **중심점 선택의 최적성**: 역변환으로 얻은 중심점이 거의 최적에 가까움
3. **미세한 조정**: 생성기의 대부분을 유지하면서 특정 신원만 조정

#### 3.2 일반화 성능 향상 가능성

**1. 다중 신원 동시 적응 (Multi-ID Personalization):**[1]

논문이 입증한 내용:
- 정규화 없이: 12개 신원 처리 시 심각한 아티팩트 발생
- 정규화 적용: LPIPS로 측정했을 때 0.03~0.05 수준의 부작용만 남음
- 편집 품질은 유지되면서 관계없는 신원에 미치는 영향 최소화

**향후 개선 방향:**
- 더 많은 신원을 동시에 처리할 수 있도록 정규화 강화
- 신원 간 간섭 최소화 메커니즘 개발

**2. 인코더 기반 빠른 근사:**[1]

현재 한계: 각 이미지마다 3분 최적화 필요

향후 가능성:
- HyperNetwork 기반 접근 (HyperStyle): 생성기 가중치를 예측하는 네트워크 학습[2]
- 경량 어댑터 네트워크: 이미지 특성에 따른 동적 가중치 조정
- 메타-러닝: 새로운 신원에 빠르게 적응하는 학습 기법

**3. 다른 생성기로의 확장:**[1]

현재: StyleGAN2에만 검증

가능한 확장:
- BigGAN, Vision Transformers (ViT) 기반 생성기
- 최근 Diffusion Models를 활용한 이미지 편집으로 전환
- 3D 생성기 (예: EG3D)에 적용

**4. 전이 학습(Transfer Learning) 가능성:**[1]

- 한 도메인(예: 얼굴)에서 학습한 정규화 전략을 다른 도메인(예: 고양이, 자동차)에 적용
- 도메인별 최적 정규화 파라미터 자동 탐색

***

### 4. 논문의 영향과 향후 연구 고려사항

#### 4.1 현재의 연구적 영향

**1. 개념적 기여:**
- StyleGAN의 생성 능력을 활용하되, 고정 생성기의 한계를 극복
- "생성기 조정" 관점의 재평가: 기존에는 회피했던 방식이 효과적임을 증명

**2. 실무적 영향:**
- 영화/광고 제작: 여러 배우의 이미지를 단일 맞춤형 생성기로 처리 가능
- 얼굴 인식 회피: 도메인 외 이미지(메이크업, 소품)에서도 높은 품질 유지

**3. 후속 연구 촉진:**
- HyperStyle: Pivotal Tuning의 속도 문제를 해결하기 위한 하이퍼네트워크 접근[2]
- ReGANIE: 두 단계 프레임워크로 편집 - 재구성 트레이드오프 극복[3]
- 최근 확산 모델(Diffusion Models) 기반 편집[32-50]: GAN을 넘어 새로운 생성 패러다임

***

#### 4.2 향후 연구 시 고려할 점

**1. 계산 효율성 개선:**

**현재 상황**: 3분/이미지
- 장점: 높은 품질
- 단점: 대규모 배치 처리 어려움, 실시간 앱 불가능

**개선 방안**:
- **경량 어댑터**: LoRA(Low-Rank Adaptation) 기법 적용[2]
- **캐시 최적화**: 공통 계산 단계 재사용
- **GPU 병렬화**: 다중 신원 동시 처리

**2. 정규화 메커니즘 고도화:**

**현재 지역성 정규화의 문제점**:
- 보간 파라미터 $\alpha = 30$은 경험적으로 결정됨
- 신원별, 도메인별 최적값이 다를 수 있음

**개선 방안**:
- **적응적 정규화**: 신원 특성에 따라 자동으로 조정
- **계층적 정규화**: 생성기 층별로 다른 강도 적용
- **의미론적 정규화**: 편집하려는 속성과 관련 없는 영역만 보호

**3. 도메인 일반화 (Domain Generalization):**

**도메인 변화**:
- 얼굴 → 동물(고양이, 개)
- 얼굴 → 물체(자동차, 건물)
- 사진 → 미술 작품

**고려사항**:
- 각 도메인의 StyleGAN 특성 분석 필요
- 도메인 간 생성기 가중치 전이 메커니즘 개발
- 도메인 특화 정규화 항 설계

**4. 편집 다양성 확대:**

**현재**: 얼굴 속성(나이, 포즈, 표정, 미소)만 검증

**향후 가능성**:
- **기하학적 편집**: 3D 회전, 변형
- **재조명(Relighting)**: 조명 조건 변경
- **질감 편집**: 표면 재질 변경
- **언어 조건 편집**: StyleClip과의 결합 확장

**5. 이론적 분석 강화:**

**현재 논문의 이론적 근거**:
- StyleGAN의 disentanglement 특성에 기반
- 경험적 검증이 주를 이룸

**강화 방향**:
- **라플라시안 분석**: 생성기 야코비안 분석으로 조정 효과 예측
- **손실 곡면 분석**: 왜 중심점 근처 정규화가 효과적인가
- **일반화 경계**: 이론적 일반화 한계 규명

**6. 안전성 및 윤리 고려:**

**잠재 위험**:
- 얼굴 합성으로 인한 거짓 신원 생성
- 신원 추적 회피(adversarial deepfakes)
- 개인정보 보호 침해

**대응 방안**:
- 조작 감지 기법 개발
- 수정 이력 추적 메커니즘
- 동의 기반 사용 정책

***

### 5. 2020년 이후 관련 최신 연구 탐색

#### 5.1 StyleGAN 역변환 분야의 최근 진전

**1. 확장된 잠재 공간 탐색 (2023-2024)**

**Revisiting Latent Space of GAN Inversion (2023-2024):**[4][5]
- StyleGAN의 초기 Z 공간(hyperspherical prior)을 재조명
- $\mathcal{F}/\mathcal{Z}^+$ 복합 공간 제시: 충실도와 편집성 동시 달성
- 기존 W, W+, S 공간을 능가하는 성능

**Delving StyleGAN Inversion (2023):**[6]
- 기초 잠재 공간 W의 중요성 강조
- 대조 학습(contrastive learning)으로 W-S 정렬
- 크로스-어텐션 인코더로 W → W+ → F 변환

**2. 생성기 미세조정 기반 접근 (2022-2023)**

**HyperStyle (2022):**[2]
- Pivotal Tuning의 느린 속도 해결
- 하이퍼네트워크로 생성기 가중치 조정
- 인코더 기반이면서 높은 품질 유지

**ReGANIE (2023):**[3]
- 두 단계 프레임워크: W-공간 편집 + 재구성 수정
- 편집성과 재구성 품질의 완전한 분리
- 보지 못한 편집 타입에도 일반화 가능

**3. 다중 신원 처리 진전 (2021-2023)**

**MambaStyle (2025):**[7]
- Vision State-Space Models (VSSM) 활용
- 인코더 기반 단일 단계 접근
- 계산 복잡도 대폭 감소, 실시간 처리 가능

***

#### 5.2 확산 모델(Diffusion Models) 기반 이미지 편집의 부상

**1. 확산 모델의 우위성 (2022-2024)**

**Latent Diffusion Models (2022):**[8]
- 픽셀 공간이 아닌 잠재 공간에서 확산 수행
- StyleGAN 대비 안정적 학습, 높은 다양성
- 텍스트 조건화, 인페인팅 우수

**DragDiffusion (2023):**[9]
- DRAGGAN을 확산 모델로 확장
- 점 기반 인터랙티브 편집
- 더 강한 사전 정보(pretrained large-scale models)

**2. 역변환 문제 해결 (2023-2024)**

**Accelerated Iterative Diffusion Inversion (2023):**[10]
- DDIM 역변환의 불안정성 극복
- 혼합 안내(blended guidance) 기법
- 10-20 단계 빠른 편집 가능

**Guided Newton-Raphson Inversion (2023):**[11]
- 수치 해석 기반 고속 역변환
- Stable Diffusion, SDXL-Turbo, Flux 지원
- 0.4초 내 역변환 가능

**3. 의미론적 편집 고도화 (2023-2024)**

**PFB-Diff (2023):**[12]
- 다층 특징 혼합으로 텍스트-이미지 편집
- 배경 편집, 다중 객체 교체 지원
- 미세조정 없이 학습-불필요(training-free)

**Prompt Tuning Inversion (2023):**[13]
- 텍스트 조건 확산 모델용 역변환
- 학습 가능 조건부 임베딩
- 고충실도 + 높은 편집성

***

#### 5.3 클립(CLIP) 기반 다중 모달 편집

**1. 언어-이미지 정렬 활용 (2021-2023)**

**StyleCLIP (2021):**[14]
- CLIP으로 텍스트 기반 StyleGAN 편집
- 세 가지 편집 모드: 전역 텍스트, 로컬 편집, 매퍼 기반
- 라벨 없이 의미론적 방향 발견

**CLIPInverter (2023):**[15]
- 텍스트 조건 가벼운 어댑터
- 다중 속성 변화 효율화
- CLIP 임베딩 기반 정제

**2. 멀티모달 확산 모델 (2023-2024)**

**PAIR-Diffusion (2024):**[16]
- 객체 레벨 다중 모달 편집
- 참고 이미지, 텍스트, 마스크 결합
- 신원 보존 및 구조 제어

***

#### 5.4 도메인 특화 편집 (2020-2025)

**1. 3D 생성 모델 역변환 (2023-2024)**

**EG3D 역변환 (2023):**[17]
- 3D GAN 역변환을 위한 기하학-인식 인코더
- 정규 잠재 공간 개념 도입
- 높은 3D 충실도 + 텍스처 상세도

**2. 비얼굴 도메인 편집 (2023-2024)**

**EditGAN (2022):**[18]
- 고정밀 의미론적 부품 편집
- 마스크 기반 최적화
- 자동차, 동물 등 복합 객체 지원

**LASPA (2024):**[19]
- 확산 모델 기반 학습-불필요 편집
- 공간 안내를 통한 세부 보존
- 텍스트 기반 실시간 편집

***

#### 5.5 일반화 및 효율성 개선 (2023-2025)

**1. 효율성 혁신**

| 방법 | 연도 | 시간 | 주요 특성 |
|------|------|------|---------|
| PTI | 2021 | 3분/이미지 | 높은 품질, 느린 속도 |
| HyperStyle | 2022 | 0.1초 | 하이퍼네트워크 기반 |
| MambaStyle | 2025 | 0.05초 | VSSM 기반, 최소 파라미터 |

**2. 일반화 성능 향상 (2023-2025)**

**ReGANIE의 일반화:**[3]
- 보지 못한 편집 타입에 강건
- 도메인 외(out-of-distribution) 이미지 처리
- 생성기 재훈련 없음

**Diffusion 기반 접근의 일반화:**[20][21][16]
- 대규모 학습 데이터로 더 강한 사전 정보
- 다양한 도메인에 사전 학습된 모델 가용
- 미세조정으로 새 도메인 빠른 적응

***

#### 5.6 미래 방향성 (2025 이후 예상)

**1. 하이브리드 접근:**
- StyleGAN의 고속성 + Diffusion의 안정성
- 적응형 선택: 이미지 특성에 따라 모델 선택

**2. 메타-러닝 및 퓨샷 적응:**
- 새로운 도메인에 몇 개 예제로 빠른 적응
- 도메인 일반화 이론 기반 설계

**3. 원리 기반(First-principles) 이해:**
- 생성 모델의 선형성 분석
- 조건 번들(condition bundles) 개념화
- 편집 방향의 수학적 특성화

**4. 안전하고 해석 가능한 편집:**
- 편집 추적성(editability traceability)
- 조작 감지 기법 발전
- 규제 준수 메커니즘

***

## 결론

**"Pivotal Tuning for Latent-based Editing of Real Images"**는 단순하면서도 효과적인 아이디어로 StyleGAN 기반 이미지 편집의 한계를 극복했습니다. 왜곡-편집성 트레이드오프를 해결하기 위해 **생성기를 약간 조정하되, 지역성 정규화로 부작용을 제한**하는 전략은 이후 많은 후속 연구의 토대가 되었습니다.

다만 **3분의 계산 시간**은 현재 기술(2025년)에서 큰 한계로, 이를 극복하기 위해 HyperStyle(하이퍼네트워크), ReGANIE(두 단계 분리), MambaStyle(상태 공간 모델) 등 다양한 접근이 제시되었습니다. 동시에 **확산 모델의 부상**으로 GAN 기반 편집이 보다 안정적이고 일반화된 대안으로 이동하는 중입니다.

향후 연구는 다음에 집중할 것으로 예상됩니다:
1. **실시간성**: 밀리초 단위 편집
2. **일반화**: 도메인별 미세 조정 최소화
3. **안전성**: 조작 감지 및 윤리 보호
4. **다중성**: 텍스트, 마스크, 참고 이미지 등 다양한 조건

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f9acc7fe-77bd-4ce5-8f6c-4afa19f11248/2106.05744v1.pdf)
[2](http://arxiv.org/abs/2111.15666)
[3](https://arxiv.org/abs/2301.13402)
[4](https://ieeexplore.ieee.org/document/10483826/)
[5](https://arxiv.org/abs/2307.08995)
[6](https://ieeexplore.ieee.org/document/10204400/)
[7](https://arxiv.org/abs/2505.15822)
[8](http://arxiv.org/pdf/2112.10752.pdf)
[9](https://ieeexplore.ieee.org/document/10655542/)
[10](https://ieeexplore.ieee.org/document/10378330/)
[11](https://www.semanticscholar.org/paper/46214f1ba1eb3a1c56773bd5c1727b04dc13f627)
[12](https://arxiv.org/abs/2306.16894)
[13](https://ieeexplore.ieee.org/document/10377418/)
[14](https://arxiv.org/abs/1907.10786)
[15](https://dl.acm.org/doi/10.1145/3610287)
[16](https://arxiv.org/html/2303.17546v3)
[17](https://openaccess.thecvf.com/content/ICCV2023/papers/Yuan_Make_Encoder_Great_Again_in_3D_GAN_Inversion_through_Geometry_ICCV_2023_paper.pdf)
[18](https://wandb.ai/geekyrakshit/editgan/reports/EditGAN-High-Precision-Semantic-Image-Editing--VmlldzoxNzc1MDYw)
[19](https://arxiv.org/pdf/2403.12585.pdf)
[20](https://ieeexplore.ieee.org/document/10377708/)
[21](https://ieeexplore.ieee.org/document/10208651/)
[22](https://arxiv.org/abs/2306.00241)
[23](https://dl.acm.org/doi/10.1145/3617695.3617701)
[24](https://ieeexplore.ieee.org/document/10377247/)
[25](https://arxiv.org/abs/2310.15081)
[26](http://arxiv.org/pdf/2104.14754.pdf)
[27](https://arxiv.org/pdf/2307.15033.pdf)
[28](https://arxiv.org/abs/2211.11448)
[29](https://arxiv.org/pdf/2304.14403.pdf)
[30](https://arxiv.org/pdf/2208.12408.pdf)
[31](http://arxiv.org/pdf/2306.00241.pdf)
[32](https://arxiv.org/html/2312.08256)
[33](https://www.youtube.com/watch?v=36hLx1CtKr4)
[34](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_L2M-GAN_Learning_To_Manipulate_Latent_Space_Semantics_for_Facial_Attribute_CVPR_2021_paper.pdf)
[35](https://blog.neuralwork.ai/a-survey-on-image-generation-and-generative-image-editing/)
[36](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Delving_StyleGAN_Inversion_for_Image_Editing_A_Foundation_Latent_Space_CVPR_2023_paper.pdf)
[37](https://pmc.ncbi.nlm.nih.gov/articles/PMC10602338/)
[38](https://pmc.ncbi.nlm.nih.gov/articles/PMC12620437/)
[39](https://openaccess.thecvf.com/content/WACV2024/papers/Katsumata_Revisiting_Latent_Space_of_GAN_Inversion_for_Robust_Real_Image_WACV_2024_paper.pdf)
[40](https://arxiv.org/html/2412.09656v1)
[41](https://happy-jihye.github.io/gan/gan-23/)
[42](https://interpretable-ml-class.github.io/slides/Lecture_18_Latent_Space_GAN.pdf)
[43](https://proceedings.mlr.press/v162/subramanyam22a/subramanyam22a.pdf)
[44](https://ieeexplore.ieee.org/document/10376878/)
[45](https://arxiv.org/abs/2312.06680)
[46](https://arxiv.org/abs/2311.12066)
[47](https://arxiv.org/pdf/2210.11427.pdf)
[48](https://aclanthology.org/2023.findings-emnlp.646.pdf)
[49](http://arxiv.org/pdf/2309.00613.pdf)
[50](https://arxiv.org/abs/2405.00313)
[51](https://arxiv.org/html/2312.02548)
[52](https://openaccess.thecvf.com/content/WACV2025W/ImageQuality/papers/Wu_LatentPS_Image_Editing_Using_Latent_Representations_in_Diffusion_Models_WACVW_2025_paper.pdf)
[53](https://papers.cool/arxiv/2502.18116v2)
[54](https://arxiv.org/html/2504.13226v1)
[55](http://vinesmsuic.github.io/paper-gan-inversion/index.html)
[56](https://research.nvidia.com/labs/toronto-ai/semanticGAN/resources/SemanticGAN_supp.pdf)
[57](https://arxiv.org/pdf/2504.15723.pdf)
[58](https://hyoseok-personality.tistory.com/entry/Paper-Review-Style-Transformer-Style-Transformer-for-Image-Inversion-and-Editing)
[59](https://openaccess.thecvf.com/content/WACV2024/papers/Zhang_Text-to-Image_Editing_by_Image_Information_Removal_WACV_2024_paper.pdf)
[60](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/imagic/)
