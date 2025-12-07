
# Cross-Domain Compositing with Pretrained Diffusion Models

## 1. 핵심 주장 및 주요 기여

**"Cross-Domain Compositing with Pretrained Diffusion Models"** 논문의 핵심 주장은 **사전 학습된 확산 모델(pretrained diffusion models)을 임의적 조정 없이 직접 활용하여 서로 다른 시각적 도메인에 속하는 이미지 요소들을 자연스럽게 합성할 수 있다**는 것입니다.[1]

주요 기여는 다음과 같습니다.[1]

- **추론 시간 조건화 기법**: 훈련이 필요 없는 반복적 국소화 정제 방식(localized iterative refinement scheme)을 제안하여, 배경 장면에서 파생된 맥락 정보로 삽입된 객체를 주입합니다.
- **광범위한 응용 분야**: 이미지 블렌딩, 객체 몰입(object immersion), 텍스처 교체, CG2Real 변환 및 스타일화 등 다양한 교차 도메인 합성 작업을 단일 프레임워크로 처리합니다.
- **무주석 학습**: 추가적인 주석이나 도메인 특화 학습 없이 기성 확산 모델을 활용합니다.
- **실용적 응용**: 단일 뷰 3D 재구성(SVR)의 데이터 증강에서 시뮬레이션-현실 간(sim-to-real) 도메인 갭을 효과적으로 해소함을 증명합니다.

## 2. 해결하고자 하는 문제 및 제안하는 방법

### 문제 정의

교차 도메인 합성(cross-domain compositing)은 다음과 같은 도전 과제를 제시합니다.[1]

서로 다른 시각적 도메인에서 나온 요소들(예: 사진을 회화에 삽입, 3D 렌더링 객체를 사진으로 변환)을 합성할 때, 단순한 객체 조화(object harmonization)를 넘어 도메인 스타일 매칭까지 동시에 달성해야 합니다. 이는 기존의 이미지 합성 방법들이 동일 도메인 내에서만 최적화되었기 때문에 해결이 어렵습니다.[1]

### 제안하는 방법: Masked ILVR

저자들은 ILVR(Iterative Latent Variable Refinement)을 확장한 **Masked ILVR** 기법을 제안합니다.[2][1]

#### 기본 원리

ILVR의 핵심은 매 디노이징 단계에서 저주파 정보를 참조 이미지에서 주입하는 것입니다:[1]

$$x_{t-1} = \phi(y_{t-1}) + (I-\phi)(x'_{t-1})$$

여기서:
- $\(x'\_{t-1} \sim p_\theta(x'_{t-1}|x_t)\)$ : 확산 모델의 예측
- $\(y_{t-1} \sim q(y_{t-1}|y_0)\)$ : 잡음이 섞인 참조 이미지
- $\(\phi\)$ : 저주파 필터(bilinear downsampling/upsampling으로 구현)

이 과정은 $\(T_{stop}\)$ 까지 반복되며, 이후 조건화 없이 디노이징을 계속합니다.[1]

#### 국소화 제어 (Localized Control)

나이브하게 ILVR을 마스크 영역 내에 적용하면 경계에서 앨리어싱 아티팩트가 발생합니다. 이를 해결하기 위해 저자들은 다음을 도입합니다:[1]

**혼합 필터 연산자**:

$$\phi(x; M_b) = M_b\phi_{in}(x) + (1-M_b)\phi_{out}(x)$$

여기서 $\(M_b\)$ 는 혼합 마스크이고, $\(\phi_{in}\)$ 과 $\(\phi_{out}\)$ 은 마스킹된 영역 내외에서 다른 스케일 $\(N_{in}, N_{out}\)$ 을 가집니다.[1]

**시간 마스크 도입**:

$$x_{t-1} = x'_{t-1} + M_t(\phi(y_{t-1}) - \phi(x'_{t-1}))$$

여기서 $\(M_t\)$ 는 영역별 조건화 중단 시간을 제어합니다:[1]

$$M_T = (1-T_{in})T \cdot M + (1-T_{out})T \cdot (1-M)$$

$$M^{(i,j)}_t(t) = \begin{cases} 0 & : t < M^{(i,j)}_T \\ 1 & : t \geq M^{(i,j)}_T \end{cases}$$

이를 통해 사용자는 $\(T_{in}, T_{out}, N_{in}, N_{out}\)$ 파라미터로 지표/현실 간 트레이드오프를 영역별로 독립적으로 제어할 수 있습니다.[1]

#### 앨리어싱 아티팩트 해결

두 가지 완화 방법을 제안합니다:[1]

**1) $\(\hat{x}_0\)$ -공간 블렌딩**: 

$$\hat{x}_0 = \frac{x_t}{\sqrt{\alpha_t}} - \frac{\sqrt{1-\alpha_t}}{\sqrt{\alpha_t}}\epsilon_\theta(x_t, t)$$

ILVR 블렌딩을 $\(\hat{x}_0\)$ -공간에서 수행하여 저주파 필터는 이미지에만, 잡음은 사후 추가됩니다. 이는 잡음 맵에 날카로운 경계가 생기는 것을 방지합니다.[1]

**2) 마스크 평활화**: 
$$b(M) \text{적용}: p_{blend} = \max(N_{in}, N_{out})$$

BlurOutwards 연산자를 통해 마스크 경계를 평활화하여 전환 영역에서 잡음 수준 차이를 제거합니다.[1]

#### 단계 반복 (Step Repetitions)

정보 전달 속도가 신경망의 수용 영역(receptive field)에 제한되므로, RePaint의 아이디어를 차용하여 일부 단계를 반복합니다. 이는 수용 영역을 효과적으로 확장하고 교차 도메인 정보 확산을 촉진합니다.[1]

**재샘플링 스케줄**:
매개변수 $\(R\)$ 은 재샘플링을 시작하는 상대적 타임스텝을 제어하며, 이는 배경에서 객체로 전파되는 의미론적 및 스타일 세부사항의 양을 조절합니다.[1]

## 3. 모델 구조 및 성능 향상

### 모델 구조

논문에서는 두 가지 사전 학습된 확산 모델을 활용합니다:[1]

| 모델 | 특징 | 응용 분야 |
|------|------|---------|
| **Guided Diffusion** | 픽셀 공간 DDPM (FFHQ 훈련) | 얼굴 수정, 국소 편집 |
| **Stable Diffusion** | 잠재 확산 모델 (대규모 데이터) | 다중 도메인 합성, 데이터 증강 |

#### 아키텍처의 특성

- **사전 학습**: CLIP 텍스트 인코더 및 cross-attention 메커니즘 활용[1]
- **유연한 조건화**: 텍스트, 마스크, 이미지 참조 조건을 동시에 지원[1]
- **계층적 처리**: U-Net 기반 노이즈 예측기 $\(\epsilon_\theta(x_t, t, y)\)$ [1]

### 성능 향상

#### 평가 지표

저자들은 다음 지표를 제시합니다:[1]

**충실도 (Fidelity)**:
- LPIPS: 사전/사후 합성 전경 유사성
- PSNR: 픽셀 수준 유사성

**현실성 (Realism)**:
$$\text{CLIP}_{dir} = \frac{\Delta I \cdot \Delta T}{|\Delta I||\Delta T|}$$

여기서 $\(\Delta I = E_I(I_{pred}) - E_I(I_{ref})\), \(\Delta T = E_T(t_{target}) - E_T(t_{source})\)$ [1]

$$\text{CLIP}_{SI} = \frac{E_I(I_{pred}) \cdot E_T(t_{target})}{E_I(I_{ref}) \cdot E_I(I_{target})}$$

**배경 보존**:
- LPIPS/PSNR (배경 영역)

#### 정량적 결과

**스크리블 기반 편집** (Table 1):[1]
- 제안 방법: CLIP_dir = 0.093, CLIP_SI = 1.074
- 마스킹된 SDEdit: CLIP_dir = 0.076, CLIP_SI = 1.061
- 배경 PSNR: 29.526 (최고)

**객체 몰입** (Table 1, Figure 6):[1]
- 사용자 연구: 제안 방법이 Deep Image Blending (29.57% vs 12.67%)과 Poisson Blending (87.33% vs -)에서 선호
- CLIP_SI: 1.214 (최고, Paint-by-Example 대비)
- 배경 보존: LPIPS 0.159 (Paint-by-Example 0.208)

**단일 뷰 3D 재구성 (SVR) 증강** (Table 2):[1]

| 데이터셋 | 원본 | 사본-붙여넣기 | T_in=0.25 | T_in=0.5 | T_in=0.75 |
|---------|------|------------|----------|----------|----------|
| 소파 | 0.535 | 0.486 | 0.650 | 0.623 | 0.554 |
| 의자 | 0.434 | 0.301 | 0.529 | 0.564 | 0.474 |

2D IoU 메트릭 사용: 제안 방법이 기존 기준선을 10-26% 상회

#### 파라미터 절감 연구 (Figure 8)

세 가지 주요 파라미터의 효과:[1]

- **T_in** (ILVR 강도): 낮을수록 현실성 증가, 높을수록 충실도 증가
- **N_in** (저주파 필터 스케일): 증가 시 다양성 감소, 구조 손상 위험
- **R** (재샘플링 타임스텝): 증가 시 의미론적/스타일 세부사항 전파 증가

## 4. 모델의 일반화 성능 향상 가능성

### 일반화 능력 분석

**핵심 강점**:[1]

1. **훈련 불필요**: 사전 학습된 대규모 모델을 직접 활용하므로, 특정 도메인이나 작업에 대한 재훈련이 필요 없습니다. 이는 완전히 새로운 도메인 조합에도 즉시 적용 가능함을 의미합니다.[1]

2. **다중 도메인 지원**: Stable Diffusion은 LAION 데이터셋에서 훈련되어 광범위한 시각적 도메인(사진, 회화, 스케치, 만화 등)에 대한 풍부한 사전 지식을 보유합니다.[3]

3. **영역별 독립적 제어**: $\(T_{in}, T_{out}, N_{in}, N_{out}\)$ 파라미터를 마스킹된 영역별로 다르게 설정할 수 있어, 다양한 특성의 객체에 대한 유연한 적응이 가능합니다.[1]

4. **반복 이미지 재샘플링**: 재샘플링 단계를 통해 배경의 맥락이 점진적으로 객체로 확산되므로, 도메인 간 의미론적 거리에 관계없이 조화가 가능합니다.[1]

### 한계점 및 개선 방향

**주요 한계**:[1]

1. **작은 객체 처리**: 잠재 확산 모델의 저해상도 잠재 공간에서 미세한 세부사항이 손실됩니다. 이를 위해 저자들은 바운딩 박스를 2배 확대하여 처리한 후 원본 크기로 복원하는 전처리 방법을 제안합니다.[1]

2. **의미론적 오류**: 확산 모델이 삽입된 객체를 잘못된 의미 클래스로 인식할 수 있습니다. 특히 비표준적 렌더링이나 어려운 뷰(예: ShapeNet 비행기)에서 이 문제가 두드러집니다.[1]

3. **그림자 생성 실패**: 방법은 조명 및 음영을 어느 정도 생성하지만, 사용자 지도 없이는 적절한 그림자를 생성하지 못합니다.[1]

4. **충실도-현실성 트레이드오프**: 모든 파라미터 조합이 모든 입력에 대해 만족스러운 결과를 생성하지 않으므로, 사용자가 여러 설정을 시도하고 수동으로 최적의 결과를 선택해야 합니다. 이를 완화하기 위해 저자들은 준자동 채점 방법을 제안합니다:[1]

$$f(s_{fidelity}, s_{realism}) = \frac{1}{1 + \frac{1}{\lambda}(s_{fidelity} + \lambda s_{realism})}$$

$(\(\lambda = 2\)$ , 현실성을 약간 선호)

### 일반화 향상의 실증적 증거

**SVR 데이터 증강 실험**이 일반화 능력의 강력한 증거입니다:[1]

- **도메인 갭 해소**: 합성 데이터로 훈련한 SVR 모델이 현실 이미지에서 기존 대비 10-26% 성능 향상
- **강건성**: 다양한 $\(T_{in}\)$ 값에서 일관된 개선을 보여주며, $\(T_{in}=0.5\)$ 에서 최적 성능 달성
- **확장성**: 소파, 의자, 테이블 등 서로 다른 객체 카테고리에서 모두 효과적

### 향후 일반화 개선 가능성

**1. 프롬프트 기반 적응**: Paint-by-Word나 Prompt-to-Prompt와 같은 기법을 통해 주의 메커니즘을 수정하여 객체-마스크 정렬을 개선할 수 있습니다.[1]

**2. 다중 참조 확산**: 여러 참조 이미지를 활용하여 의미론적 일관성을 높일 수 있습니다.[4]

**3. 하이브리드 도메인 학습**: 약간의 도메인 특화 미세조정(minimal fine-tuning)을 통해 특정 도메인 쌍의 성능을 추가로 개선할 수 있습니다.[1]

## 5. 한계 및 향후 연구 방향

### 기술적 한계

1. **계산 비용**: 최대 54초/이미지 (R=0.8, RTX 2080 Ti 기준)로, 실시간 응용에는 부적합합니다.[1]

2. **비디오 확장의 어려움**: 공간적 일관성은 달성하지만 시간적 일관성 유지가 매우 도전적입니다.[1]

3. **패치 불일치**: 겹치는 패치 영역에서 불일치 문제 발생 가능성[1]

4. **자동화 부재**: 파라미터 선택이 반자동이므로 실제 사용 편의성이 제한됩니다.[1]

### 향후 연구 시 고려할 점

**1. 확산 모델 개선**

- 더 빠른 샘플링: DDIM, DPM-Solver 같은 고급 스케줄러 활용[5][6]
- 지연 공간 최적화: 더 높은 해상도 잠재 공간 개발
- 더 나은 의미론적 이해: 객체 인식도를 높이기 위한 사전 학습 개선

**2. 조건화 메커니즘 강화**

- **다중 조건 통합**: 텍스트, 레이아웃, 깊이 정보 동시 활용[7][6]
- **적응형 가이던스**: 입력 특성에 따른 동적 가이던스 강도 조정[8]
- **계층적 조건화**: 저주파에서 고주파로 진행하는 명시적 계층 구조

**3. 일반화 능력 확대**

- **교차 도메인 관계 학습**: 특정 도메인 쌍에 대한 최소한의 미세조정[9]
- **소수 샘플 적응**: 극소수의 참조 이미지로 새 도메인에 빠르게 적응
- **도메인 역전 기법**: DIDEX와 같이 의사 타겟 도메인을 생성하여 도메인 일반화 개선[10]

**4. 응용 확대**

- **3D 표면 텍스처링**: 메시 기하학과 함께 고려한 3D 객체 합성[11]
- **동적 장면 합성**: 비디오나 애니메이션 시퀀스 생성[12]
- **의료 영상 응용**: 도메인 간 합성을 통한 의료 데이터 증강[13]

**5. 평가 방법론**

- **더 정교한 지표**: 신경망 기반 객체 인식, 조화도 지수 개발[14]
- **인간 평가 표준화**: 사용자 연구 프로토콜 고도화[1]
- **비디오 일관성 메트릭**: 시간적 연속성 평가 방법

## 6. 2020년 이후 최신 관련 연구 동향

### A. 확산 모델 기반 합성 및 편집 (2023-2025)

**직접 관련 작업:**

1. **TF-ICON & TALE** (2023-2024)[15][16]
   - 훈련 불필요 교차 도메인 이미지 합성
   - CLIP 이미지 임베딩 분해를 통한 스타일 강도 조절
   - 배경 보존 개선

2. **FreeCompose** (2024)[17][18]
   - 확산 모델의 생성 사전(generative prior)을 활용한 일반화 이미지 합성
   - 저밀도 영역(unnatural composition areas) 자동 감지
   - 마스크 기반 손실 함수로 의미론적 합성 지원

3. **RefPaint** (2023)[19][20]
   - 참조 기반 회화 인페인팅
   - CLIP 이미지 임베딩 분해로 의미론과 스타일 정보 분리
   - 큰 도메인 갭(photorealistic → artistic) 극복

4. **ControlCom** (2023)[21]
   - 제어 가능한 이미지 합성
   - 전경 속성 및 ID 보존에 중점
   - 다양한 조건 유형 통합

### B. 도메인 적응 및 일반화 (2023-2025)

1. **DIDEX: Generalization by Adaptation** (2024)[10][9]
   - 확산 모델을 이용한 의사 타겟 도메인 생성
   - 비지도 도메인 적응과 결합하여 일반화 달성
   - SOTA 결과 (ResNet: 7.4% 절대 향상, Transformer: 11.8% 향상)

2. **Cross-Domain Ensemble Distillation** (2022)[22]
   - 평탄한 극값(flat minima) 추구로 도메인 불변 특성 학습
   - 스타일 정규화를 통한 도메인 갭 감소
   - 다중 출처 및 단일 출처 도메인 일반화 지원

3. **Zero-Shot Depth-Aware Image Editing (DAEdit)** (2024)[6]
   - 깊이 기반 계층 분해 + 확산 모델 활용
   - 특징 공간에서의 계층화 지도(FeatGLaC)
   - 깊이 순서 보존으로 자연스러운 합성

### C. 고급 인페인팅 및 조건화 기법 (2023-2025)

1. **Paint by Example** (2022)[23][24]
   - 이미지-조건 확산 기반 편집
   - 자기 감독 학습으로 자명한 솔루션(trivial copying) 방지
   - 내용 병목 및 강력한 증강 활용

2. **RAD: Region-Aware Diffusion** (2024)[5]
   - 픽셀별 잡음 스케줄로 비동기 영역 생성
   - 전역 맥락 고려하며 국소 제어 달성
   - 추가 성분 불필요

3. **Lanpaint** (2025)[25]
   - Langevin 역학 기반 정확한 조건부 추론
   - 훈련 불필요, ODE 기반 샘플러 호환
   - 효율적이고 정확한 인페인팅

4. **Uni-Paint** (2023)[26]
   - 다중 모드 이미지 인페인팅
   - 텍스트, 스케치, 이미지 참조 동시 활용
   - 사용자 정의 모양/색상/배치 제어

### D. 스타일 전이 및 미적 스타일링 (2023-2025)

1. **LSAST: Layer & Step-aware Artistic Style Transfer** (2024)[27]
   - 단계 및 계층 인식 프롬프트 공간
   - 콘텐츠 구조 보존 + 현실적 스타일화
   - 예술 작품 생성 최적화

2. **DreamStyler** (2023)[28]
   - 텍스트와 스타일 참조 결합
   - 다단계 텍스트 임베딩 최적화
   - 예술 제품 창작 지원

3. **Painterly Image Harmonization** (2024)[14]
   - 동적 커널 기반 양방향 변환
   - 전경/배경 특징 맵 정렬
   - 회화 스타일 조화

### E. 이론적 진전 및 메커니즘 이해 (2023-2025)

1. **Critical Windows in Diffusion** (2024)[29]
   - 확산 과정의 시간적 특성 이해
   - 특징 출현의 좁은 시간 구간 식별
   - 공정성/개인정보보호 진단 도구로 활용

2. **Creativity in Attention-Based Diffusion** (2025)[30]
   - 자기 주의의 역할 이론화
   - 글로벌 이미지 일관성 메커니즘 설명
   - 생성적 다양성의 원천 규명

3. **Theoretical Justification for Diffusion Inpainting** (2023)[31]
   - 선형 모델 설정에서 정리적 정당화
   - 재훈련 없이 미보이 마스크에 대한 일반화 증명
   - 확산 기반 인페인팅의 견고성 분석

### F. 특화된 응용 (2023-2025)

1. **Zero-Shot Image Translation** (2024)[32]
   - 자기 주의 층 편집
   - 참조 이미지 기반 스타일 조정
   - 최적화/미세조정 불필요

2. **MaskMedPaint** (2024)[13]
   - 의료 영상 도메인 적응
   - 가짜 상관관계 완화
   - 의료 데이터 정렬

3. **CFDiffusion: Controllable Foreground Relighting** (2024)[33]
   - 조화 및 그림자 생성 동시 수행
   - 전경 재조명 제어
   - 조명 일관성 개선

### G. 인간 선호도 정렬 (2024)[34]

**PrefPaint: 강화 학습 기반 정렬**
- 확산 모델을 인간 미적 기준과 정렬
- 보상 모델 정확도 한계 도출
- 신뢰도 인식 정렬 프로세스

## 7. 향후 연구에 미치는 영향 및 고려사항

### 학술적 영향

1. **확산 모델의 보편적 도구화**: 이 논문은 사전 학습된 확산 모델을 **작업 특화 모델 없이** 직접 활용할 수 있음을 증명하여, 확산 모델이 이미지 처리의 "기본 연산"이 될 수 있음을 시사합니다.

2. **훈련 불필요 패러다임의 확립**: 최근 3년간의 연구 동향(TF-ICON, TALE, FreeCompose, RefPaint 등)이 이 방향성을 강하게 지지하며, 산업 응용의 장벽을 크게 낮춥니다.[16][15][17][19]

3. **국소화 제어 기법의 확산**: 마스킹된 ILVR의 핵심 아이디어(영역별 독립적 파라미터, 시간 기반 제어)가 RAD, FeatGLaC 등에서 진화되고 있습니다.[6][5]

### 실무적 함의

1. **데이터 증강 혁신**: SVR 실험이 보여주듯이, 교차 도메인 합성이 **sim-to-real 도메인 갭 해소의 강력한 도구**임을 입증합니다. 이는 로봇공학, 자율주행, 3D 컴퓨터 비전 분야에 직접 적용 가능합니다.[1]

2. **창의 산업으로의 확대**: 예술 작품 생성(회화에 객체 삽입), 게임/영화 프로덕션 자동화, 패션 디자인 등에서 실질적 수요를 창출합니다.[1]

3. **접근성 향상**: 훈련과 미세조정이 불필요하므로, 전문 지식 없는 사용자도 고품질 이미지 조작이 가능해집니다.

### 기술적 과제

1. **계산 효율성**: 54초/이미지는 여전히 실시간 응용에 부족합니다. 향후 연구는:
   - 잠재 공간 해상도 개선[25][3]
   - 더 빠른 샘플링 스케줄 활용[6]
   - 증류(distillation) 기법 적용

2. **의미론적 견고성**: 비표준 입력(이상한 뷰, 텍스트 렌더링)에서의 실패를 개선하려면:
   - 주의 메커니즘 조정(Paint-by-Word, Prompt-to-Prompt)[35][1]
   - 객체 인식도 강화[1]
   - 더 나은 사전 학습 데이터

3. **사용자 경험**: 반자동 파라미터 선택 개선:
   - 메타학습을 통한 자동 파라미터 탐색
   - 입력별 최적 파라미터 예측 신경망
   - 대화형 미세조정 인터페이스

### 향후 연구 로드맵

**단기 (1-2년)**
- 비디오 확장: 시간적 일관성 메커니즘 개발[12]
- 계산 최적화: DDIM, DPM-Solver 고급 통합
- 3D 확대: 메시 기하학 인식 합성[11]

**중기 (2-4년)**
- 멀티모달 조건화: 깊이, 의미 세그멘테이션, 레이아웃 동시 활용[6]
- 적응형 가이던스: 입력 특성별 동적 조건 강도
- 도메인 자동 감지: 입력 도메인 자동 식별 및 최적 파라미터 추천

**장기 (4년+)**
- **일반화된 시각 생성 엔진**: 모든 2D 이미지 처리 작업(편집, 합성, 번역, 향상)을 단일 프레임워크로 통합
- **3D-2D 상호작용**: 3D 장면 이해와 2D 이미지 생성의 밀접한 결합
- **지속적 학습**: 새 도메인에 대한 온라인 적응 능력

***

## 결론

**"Cross-Domain Compositing with Pretrained Diffusion Models"**는 사전 학습된 확산 모델의 **무훈련, 범용 이미지 합성 도구로서의 잠재성**을 명확히 입증한 중요한 연구입니다. 

**핵심 기여:**
- Masked ILVR을 통한 국소화된 반복적 정제
- 영역별 독립적 충실도-현실성 트레이드오프 제어
- 데이터 증강을 통한 실증적 일반화 향상 증명

**향후 영향:**
- 훈련 불필요 패러다임의 강화 (TF-ICON, TALE, FreeCompose 등)
- 도메인 적응 일반화 기법의 진화 (DIDEX 등)
- 비디오, 3D, 의료 영상 등 다양한 도메인으로 확대

**연구자들이 고려할 점:**
1. 계산 효율성과 의미론적 견고성의 동시 개선
2. 자동화된 파라미터 선택 메커니즘
3. 시간적/기하학적 일관성이 필요한 도메인으로의 확장
4. 이론적 일반화 분석(왜 사전 학습이 충분한가?)

이 논문의 방법론과 통찰은 앞으로 확산 모델 기반의 모든 이미지 처리 연구의 기초가 될 것으로 예상됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/f64ec86d-6796-4b66-9d4f-ebc465eafd6d/2302.10167v2.pdf)
[2](https://www.semanticscholar.org/paper/ILVR:-Conditioning-Method-for-Denoising-Diffusion-Choi-Kim/cda3fbbac6734b603bee363b0938e9baa924aa78)
[3](https://arxiv.org/pdf/2311.01090.pdf)
[4](https://arxiv.org/html/2508.12784v1)
[5](https://arxiv.org/html/2412.09191v1)
[6](https://rishubhpar.github.io/DAEdit/)
[7](https://openaccess.thecvf.com/content/ICCV2025/papers/Parihar_Zero-Shot_Depth_Aware_Image_Editing_with_Diffusion_Models_ICCV_2025_paper.pdf)
[8](https://theaisummer.com/classifier-free-guidance/)
[9](https://openaccess.thecvf.com/content/WACV2024/papers/Niemeijer_Generalization_by_Adaptation_Diffusion-Based_Domain_Extension_for_Domain-Generalized_Semantic_Segmentation_WACV_2024_paper.pdf)
[10](https://elib.dlr.de/202784/1/WACV_DIDEX.pdf)
[11](https://github.com/zju-pi/Awesome-Conditional-Diffusion-Models)
[12](https://arxiv.org/html/2501.12267v1)
[13](https://arxiv.org/html/2411.10686v1)
[14](https://bmva-archive.org.uk/bmvc/2024/papers/Paper_100/paper.pdf)
[15](https://arxiv.org/html/2408.03637v1)
[16](http://arxiv.org/pdf/2307.12493.pdf)
[17](http://arxiv.org/pdf/2407.04947.pdf)
[18](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02529.pdf)
[19](https://arxiv.org/pdf/2307.10584.pdf)
[20](https://arxiv.org/abs/2307.10584)
[21](https://arxiv.org/pdf/2308.10040.pdf)
[22](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850001.pdf)
[23](https://ieeexplore.ieee.org/document/10204542/)
[24](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Paint_by_Example_Exemplar-Based_Image_Editing_With_Diffusion_Models_CVPR_2023_paper.pdf)
[25](https://arxiv.org/html/2502.03491v1)
[26](https://arxiv.org/abs/2310.07222)
[27](https://www.ijcai.org/proceedings/2024/0865.pdf)
[28](https://arxiv.org/abs/2309.06933)
[29](https://arxiv.org/abs/2403.01633)
[30](https://arxiv.org/abs/2506.17324)
[31](http://arxiv.org/pdf/2302.01217.pdf)
[32](https://www.sciencedirect.com/science/article/abs/pii/S0952197625020561)
[33](https://dl.acm.org/doi/10.1145/3664647.3681283)
[34](https://proceedings.neurips.cc/paper_files/paper/2024/file/3658e78b56268b7fd089e3165843086b-Paper-Conference.pdf)
[35](http://arxiv.org/pdf/2403.11929.pdf)
[36](https://link.springer.com/10.1007/s10489-025-06673-1)
[37](https://www.semanticscholar.org/paper/945a899a93c03eb63be5e3197e318c077473cef9)
[38](https://dl.acm.org/doi/10.1145/3707292.3707367)
[39](https://arxiv.org/abs/2507.20478)
[40](https://ieeexplore.ieee.org/document/11147740/)
[41](https://link.springer.com/10.1007/s00261-025-05164-8)
[42](http://pubs.rsna.org/doi/10.1148/radiol.250617)
[43](http://pubs.rsna.org/doi/10.1148/radiol.242969)
[44](http://arxiv.org/pdf/2303.11916.pdf)
[45](http://arxiv.org/pdf/2112.10752.pdf)
[46](https://arxiv.org/html/2501.00944v1)
[47](https://arxiv.org/html/2308.09388v2)
[48](https://www.sciencedirect.com/science/article/abs/pii/S0957417425026764)
[49](https://openaccess.thecvf.com/content/WACV2025/papers/Ko_Text-to-Image_Synthesis_for_Domain_Generalization_in_Face_Anti-Spoofing_WACV_2025_paper.pdf)
[50](https://icml.cc/media/icml-2024/Slides/34089_UGCUCsq.pdf)
[51](https://arxiv.org/abs/2402.17307)
[52](https://www.semanticscholar.org/paper/9e73a3beffc299ccabedc98512b3dc234d2b0350)
[53](https://link.springer.com/10.1007/s10851-024-01175-0)
[54](https://azbuki.bg/uncategorized/linguistic-models-of-mass-media-genres-stylistic-diffusion-in-the-communicative-space-of-ukraine-and-bulgaria/)
[55](http://pubs.rsna.org/doi/10.1148/radiol.240288)
[56](http://pubs.rsna.org/doi/10.1148/radiol.240343)
[57](https://www.mdpi.com/2076-3417/14/17/7704)
[58](https://arxiv.org/pdf/2401.13795.pdf)
[59](https://arxiv.org/html/2412.01223v1)
[60](https://github.com/zengyh1900/Awesome-Image-Inpainting)
[61](https://ieeexplore.ieee.org/document/10613576/)
[62](https://blog.outta.ai/201)
[63](https://ieeexplore.ieee.org/document/10377423/)
[64](https://ieeexplore.ieee.org/document/10541481/)
[65](https://ieeexplore.ieee.org/document/10660873/)
[66](https://arxiv.org/abs/2501.07922)
[67](https://link.springer.com/10.1007/s11263-024-02292-4)
[68](https://ieeexplore.ieee.org/document/10992882/)
[69](https://dl.acm.org/doi/10.1145/3746027.3754926)
[70](https://arxiv.org/abs/2211.13227)
[71](https://arxiv.org/html/2401.01456v2)
[72](https://arxiv.org/html/2406.07865)
[73](https://arxiv.org/html/2502.20904v1)
[74](https://arxiv.org/abs/2311.02343)
[75](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/paint-by-example/)
[76](https://www.sciencedirect.com/science/article/abs/pii/S0950705125015503)
[77](https://blog.outta.ai/362)
[78](https://www.emergentmind.com/topics/universal-guidance-algorithm-for-diffusion-models)
[79](https://ostin.tistory.com/134)
