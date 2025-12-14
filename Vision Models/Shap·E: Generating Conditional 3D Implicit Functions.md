# Shap·E: Generating Conditional 3D Implicit Functions

### 1. 핵심 주장 및 주요 기여

**ShapE**는 OpenAI에서 발표한 조건부 3D 생성 모델로, 기존 3D 생성 연구와의 근본적인 차이를 제시합니다. 핵심 주장은 다음과 같습니다.[1]

**주요 기여:**

1. **다중 표현 지원**: ShapE는 텍스처 메시를 기준으로 하는 서명 거리 함수(SDF)와 신경 방사 필드(NeRF) 모두로 동시에 렌더링 가능한 암묵함수 매개변수를 직접 생성합니다. 이는 기존 방법들이 단일 출력 표현만 생성하던 것과 대조적입니다.[1]

2. **효율적인 두 단계 접근**: 명시적 3D 표현(점 구름)에서 암묵함수 매개변수로의 매핑을 학습하는 인코더를 먼저 훈련한 후, 인코더 출력에 대한 조건부 확산 모델을 훈련합니다.[1]

3. **성능 우위**: PointE(명시적 점 구름 생성 모델)와 동일한 아키텍처, 데이터셋, 조건화 메커니즘을 사용하면서도 더 빠른 수렴과 동등하거나 우수한 샘플 품질을 달성합니다.[1]

***

### 2. 해결하고자 하는 문제

#### 2.1 기본 문제

3D 자산 생성은 다음과 같은 본질적인 어려움을 갖습니다:[1]

- **표현 선택의 복잡성**: 점 구름, 복셀, 메시 등 다양한 3D 표현이 각각의 한계를 가짐
- **데이터 부족**: 3D 훈련 데이터의 스케일이 이미지 데이터에 비해 훨씬 작음
- **생성 효율성**: 최적화 기반 방법들(DreamFusion)은 샘플당 수 시간이 소요

#### 2.2 암묵 신경 표현(INR)의 기존 도전과제

암묵함수는 해상도 독립성과 미분 가능성의 장점을 제공하지만:[1]

- 각 샘플마다 INR을 획득하는 과정이 계산 비용이 높음
- INR 매개변수의 수가 매우 많아 직접 생성 모델 훈련이 어려움
- 여러 표현(NeRF와 메시)을 동시에 지원하는 통합 프레임워크 부재

***

### 3. 제안하는 방법론 (수식 포함)

#### 3.1 인코더 아키텍처

ShapE의 인코더는 두 가지 입력을 처리합니다:[1]

- 점 구름: 16,384개의 RGB 점
- 다중뷰 영상: 20개 각도의 256×256 렌더링

**인코더 구조 (Algorithm 1):**

$$h = \text{CatPointConv}(p, h_l)$$

$$h = \text{CrossAttend}(h, \text{Proj}(p))$$

$$h = \text{CrossAttend}(h, \text{PatchEmb}(m))$$

$$h = \text{Transformer}(h)$$

$$h = h_{len(h)}^{\text{tail}}$$

$$h = \tanh(h)$$

$$h = \text{DiffusionNoise}(h)$$

$$\theta = \text{Proj}(h)$$

여기서 $h$는 잠재 표현, $\theta$는 MLP 매개변수입니다.[1]

#### 3.2 NeRF 렌더링 손실

NeRF 기반의 신경 방사 필드는 다음과 같이 정의됩니다:[1]

$$F(x, d) = (\sigma(x), c(x, d))$$

여기서 $\sigma(x)$는 밀도, $c(x, d)$는 RGB 색상입니다.

**렌더링 손실:**

$$\mathcal{L}_{\text{RGB}} = \mathbb{E}_{r \in R} \| C_c(r) - C_r \|_1 + \| C_f(r) - C_r \|_1$$

$$\mathcal{L}_T = \mathbb{E}_{r \in R} \| T_c(r) - T_r \|_1 + \| T_f(r) - T_r \|_1$$

$$\mathcal{L}_{\text{NeRF}} = \mathcal{L}_{\text{RGB}} + \mathcal{L}_T$$

여기서 $C_c(r)$과 $C_f(r)$은 각각 거친 렌더링과 세밀한 렌더링의 색상이며, $T_c(r)$과 $T_f(r)$은 투과율입니다.[1]

#### 3.3 서명 거리 함수(SDF) 및 텍스처 필드 렌더링

SDF 기반 렌더링은 메시 생성을 가능하게 합니다:[1]

**증류 손실:**

$$\mathcal{L}_{\text{distill}} = \mathcal{L}_{\text{NeRF}} + \mathbb{E}_{x \sim U[1][2]} \left[ \| \text{SDF}(x) - \text{SDF}_{\text{regression}}(x) \|_1 + \| \text{RGB}(x) - \text{RGB}_{\text{NN}}(x) \|_1 \right]$$

**STF 렌더링 손실:**

$$\mathcal{L}_{\text{STF}} = \frac{1}{Ns^2} \sum_{i=1}^{N} \| \text{Render}(\text{Mesh}_i) - \text{Image}_i \|_2^2$$

여기서 $N$은 이미지 수, $s$는 해상도입니다.[1]

**최종 미세 조정 손실:**

$$\mathcal{L}_{\text{FT}} = \mathcal{L}_{\text{NeRF}} + \mathcal{L}_{\text{STF}}$$

#### 3.4 확산 모델

잠재 확산 모델은 PointE와 동일한 트랜스포머 아키텍처를 사용하지만, 점 구름 대신 잠재 벡터를 모델링합니다.[1]

**확산 목표:**

$$\mathcal{L}_{x_0} = \mathbb{E}_{x_0 \sim q(x_0), \epsilon \sim N(0,I), t \sim U(1,T)} \| \hat{x}_\theta(x_t, t) - x_0 \|_2^2$$

여기서 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$입니다.[1]

**분류기 없는 안내(Classifier-free guidance):**

$$\hat{x}_\theta(x_t, t | y) = \hat{x}_\theta(x_t, t) + s(\hat{x}_\theta(x_t, t | y) - \hat{x}_\theta(x_t, t))$$

여기서 $s$는 안내 스케일, $y$는 조건(텍스트 또는 이미지)입니다.[1]

**확산 노이즈 스케줄:**

$$\alpha_t = e^{-\frac{1}{2}t^5}$$

이는 의미적 정보를 점진적으로 파괴하도록 설계되었습니다.[1]

***

### 4. 모델 구조

#### 4.1 전체 구조

ShapE는 세 가지 주요 구성 요소로 이루어집니다:[1]

```
┌─────────────────────────────────────────────────┐
│         3D 자산 입력 (점 구름 + 렌더링)           │
└────────────────┬────────────────────────────────┘
                 │
        ┌────────▼─────────┐
        │  3D 인코더        │
        │  (PointConv +    │
        │   Transformer)   │
        └────────┬─────────┘
                 │
        ┌────────▼─────────────────────┐
        │  MLP 매개변수 (잠재 표현)      │
        │  Shape: 1024×1024             │
        └────────┬─────────────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
┌───▼───┐  ┌────▼────┐  ┌───▼────┐
│ NeRF  │  │  SDF    │  │Texture │
│Head   │  │ Head    │  │Head    │
└───┬───┘  └────┬────┘  └───┬────┘
    │           │           │
└───┴───────────┴───────────┘
         │
    ┌────▼─────┐
    │확산 모델  │
    │(Diffusion)
    └─────┬────┘
          │
    ┌────▼────────────────────┐
    │ 텍스처 메시 & NeRF 생성  │
    │ (텍스트/이미지 조건)      │
    └─────────────────────────┘
```

#### 4.2 인코더 세부 구조

인코더는 다음 단계로 구성됩니다:[1]

1. **점 구름 처리**: PointConv를 사용하여 16,384개 점을 1,024개 임베딩으로 다운샘플
2. **교차 주의 (Cross-Attention)**: 점 구름에 대한 교차 주의
3. **다중뷰 처리**: 패치 임베딩된 다중뷰 영상에 대한 교차 주의
4. **트랜스포머**: 시퀀스 처리를 위한 트랜스포머 백본
5. **잠재 병목**: tanh 활성화로 [-1, 1] 범위로 클램핑
6. **MLP 생성**: 1024개의 256×256 MLP 가중치 행렬 생성

#### 4.3 두 단계 훈련 전략

**1단계: NeRF 사전 훈련 (600K 반복)**
- NeRF 렌더링 목표만 사용
- 더 안정적한 최적화

**2단계: SDF/텍스처 증류 및 미세 조정 (50K + 65K 반복)**
- PointE의 SDF 회귀 모델에서 SDF 증류
- RGB 점 구름의 최근접 이웃으로 색상 증류
- NeRF와 STF 렌더링을 함께 미세 조정

***

### 5. 성능 향상 및 실험 결과

#### 5.1 인코더 성능 평가

**표 1: 훈련 단계별 인코더 평가**[1]

| 단계 | NeRF PSNR (dB) | STF PSNR (dB) | NeRF CLIP R-Precision | STF CLIP R-Precision |
|------|-----------------|---------------|-----------------------|----------------------|
| 사전훈련 300K | 33.2 | - | 44.3 | - |
| 사전훈련 600K | 34.5 | - | 45.2 | - |
| 증류 | 32.9 | 23.9 | 42.6 | 41.1 |
| 미세 조정 | 35.4 | 31.3 | 45.3 | 44.0 |

증류는 NeRF 품질을 일시적으로 감소시키지만, 미세 조정으로 NeRF 품질이 회복되고 STF 품질이 대폭 향상됩니다.[1]

#### 5.2 PointE와의 비교

**이미지-조건부 생성:**
- ShapE와 PointE는 유사한 최종 평가 성능 달성
- CLIP R-Precision에서 ShapE가 약간의 이점
- CLIP Score에서 PointE가 약간의 이점[1]

**텍스트-조건부 생성:**
- ShapE는 300M 모델에서 PointE 300M을 능가
- CLIP R-Precision: ShapE 37.8 vs PointE 33.6 (ViT-B32)
- CLIP Score: ShapE 40.9 vs PointE 35.5 (ViT-B32)
- ShapE는 PointE 1B과 비교 가능한 성능 달성[1]

#### 5.3 다른 방법과의 비교

**표 2: COCO 검증 프롬프트에서의 3D 생성 기법 비교**[1]

| 방법 | ViT-B32 | ViT-L14 | 지연시간 |
|------|---------|---------|---------|
| DreamFields | 78.6 | 82.9 | 200 V100-hr |
| DreamFusion | 75.1 | 79.7 | 12 V100-hr |
| PointE 300M (텍스트만) | 33.6 | 35.5 | 25 V100-sec |
| ShapE 300M (텍스트만) | 37.8 | 40.9 | 13 V100-sec |
| ShapE 300M (이미지) | 41.1 | 46.4 | 1.0 V100-min |

ShapE는 최적화 기반 방법보다 훨씬 빠르면서도 합리적인 품질을 달성합니다.[1]

#### 5.4 생성 속도

- 단일 NVIDIA V100 GPU에서 약 13초에 샘플 생성
- DreamFusion (200시간)이나 DreamFields (200시간)보다 수백 배 빠름
- PointE의 업샘플 모델이 필요 없어 더 효율적[1]

***

### 6. 모델의 일반화 성능 향상 가능성

#### 6.1 현재 일반화 성능

**강점:**
- **다중뷰 일관성**: 이미지 조건부 생성에서 ShapE와 PointE가 유사한 실패 사례를 보이는 것은, 훈련 데이터, 모델 아키텍처, 조건화 메커니즘이 출력 표현 선택보다 더 중요함을 시사합니다.[1]
- **암묵 vs 명시적 표현**: ShapE의 CLIP R-Precision이 PointE와 비슷하면서도 CLIP Score는 더 낮다는 것은 ShapE가 일부 프롬프트에 대해 질적으로 다른 샘플을 생성함을 의미합니다.[1]
- **얇은 특징 처리**: 벤치에서 ShapE가 점 구름의 맹점으로 인해 놓치기 쉬운 얇은 슬릿과 같은 세부사항을 더 잘 포착함을 보여줍니다.[1]

#### 6.2 일반화 성능 향상을 위한 제안

**1) 데이터 수집 개선:**

ShapE는 텍스트 조건부 모델에서 개념 합성과 속성 바인딩에 제약을 가집니다.[1]

$$\text{해결책: 더 많은 쌍을 이룬 3D-텍스트 데이터 수집}$$

- 현재 120K개 수동 캡션만 사용
- 더 큰 데이터셋은 복합 개념 이해 능력 향상
- 범주 간 일반화 개선[1]

**2) 인코더 개선:**

현재 인코더가 세부사항 손실을 야기합니다. 예: 선인장의 줄무늬 손실[1]

$$\text{개선된 인코더} \Rightarrow \text{더 나은 재구성 품질} \Rightarrow \text{더 나은 생성 품질}$$

**3) 최적화 기반 방법과의 결합:**

논문은 ShapE가 DreamFusion의 초기화로 사용될 수 있음을 제안합니다.[1]

$$\text{ShapE 샘플} \rightarrow \text{DreamFusion 최적화} \rightarrow \text{더 높은 품질}$$

평균적으로 12-40분 처리 시간에서 1.5시간으로 증가하지만, 품질 향상은 가능합니다.[1]

**4) 이미지 공간 안내:**

ShapE 샘플링을 DreamFusion 기반 이미지 공간 안내로 개선할 수 있습니다.[1]

**Appendix D의 이미지 공간 안내:**

$$\nabla_t \mathcal{L} = \frac{\partial I}{\partial x_0} \frac{\partial x_0}{\partial x_t}$$

여기서 $I$는 이미지 공간 손실(예: DreamFusion)입니다.

***

### 7. 모델의 한계

#### 7.1 주요 한계

**1) 개념 합성 부족:**[1]
- "초록색 의자와 빨간색 다리가 있는 의자" 같은 복합 프롬프트 이해 실패
- 여러 물체의 개수를 정확히 생성하지 못함 (2개 이상 요청 시 부정확)
- **원인**: 제한된 쌍을 이룬 훈련 데이터

**2) 세부사항 부족:**[1]
- 생성된 샘플이 거친 외관 또는 세밀한 디테일 부족
- 특히 텍스처 디테일이 손실됨
- **원인**: 인코더 자체가 상세 텍스처를 완전히 보존하지 못함 (예: 선인장의 줄무늬)

**3) 샘플 품질 제한:**[1]
- 최적화 기반 방법(DreamFusion, DreamFields)보다 낮은 샘플 품질
- 속성 바인딩 및 복합 개념 처리의 어려움

#### 7.2 데이터 편향

**편향 분석:**[1]

Figure 11에서 성별 편향 발견:
- "의사" 프롬프트 → 남성 인물 생성 경향
- "간호사" 프롬프트 → 여성 인물 생성 경향

**현실적 위험:**[1]
- 생성된 3D 객체가 3D 프린팅 등으로 현실화될 경우 충분한 검증 없이 사용되면 문제 발생 가능
- ShapE는 매우 사실적이지 않아 deepfake 관련 우려는 덜하지만, 산업 응용에는 위험

***

### 8. 2020년 이후 관련 최신 연구 비교 분석

#### 8.1 주요 3D 생성 방법의 진화

**표 3: 3D 생성 방법 분류 및 진화**

| 시기 | 방법 | 특징 | 장점 | 한계 |
|------|------|------|------|------|
| **2020-2021** | NeRF (Mildenhall et al., 2020) | 신경 방사 필드 제안 | 사진 현실적 렌더링 | 훈련 시간 오래 걸림 |
| **2021-2022** | 암묵 표현 (DeepSDF) | 부호 거리 함수 기반 | 메시 추출 용이 | 텍스처 처리 어려움 |
| **2022** | DreamFusion (Poole et al.) | Score Distillation Sampling | 3D 훈련 데이터 불필요 | 최적화 시간 매우 오래 (12-200시간) |
| **2022** | PointE (Nichol et al.) | 명시적 점 구름 확산 | 빠른 생성 (1-2분) | 점 구름 품질 제한 |
| **2022** | GET3D (Gao et al.) | GAN 기반 메시 생성 | 고품질 메시와 텍스처 | 복잡한 위상 제한 |
| **2023** | ShapE (Jun & Nichol) | **암묵 함수 확산** | **다중 표현, 빠른 생성** | **개념 합성 부족** |
| **2023** | Text2NeRF (Zhang et al.) | NeRF 기반 장면 생성 | 사실적 장면 생성 | 수렴 시간 오래 |
| **2023** | Michelangelo (Liu et al.) | 정렬된 잠재 공간 VAE | 다중 모달 생성 | 복잡한 구조 |
| **2023** | 3D-LDM (Zhao et al.) | 암묵 표현 잠재 확산 | 높은 품질 표면 | 메모리 요구량 높음 |
| **2023-2024** | LION (Zeng et al.) | 계층 잠재 점 확산 | 유연한 조작 | 여전히 점 구름 기반 |
| **2023-2024** | OctFusion (Xiong et al.) | 옥트리 기반 확산 | 고해상도, 빠른 생성 | 위상 제약 |
| **2024** | DIRECT-3D (Kim et al.) | 트라이플레인 확산 | 잡음 많은 데이터 처리 | 복잡한 최적화 |
| **2024** | CraftsMan (Liu et al.) | 다중뷰 네이티브 확산 | 고충실도 메시 | 2단계 프로세스 |
| **2024** | Direct3D (Zhang et al.) | 3D 잠재 확산 트랜스포머 | 큰 규모 사전학습 | 여전히 발전 중 |
| **2025** | TripoSG (Tang et al.) | 정류 흐름 + 표면 법선 | 초고해상도 | 계산 비용 높음 |

#### 8.2 ShapE의 위치와 기여

**ShapE의 혁신성:**[3][4][5][6][7][1]

1. **암묵 함수 생성의 선구자**:
   - Meta-learning 기반 접근(Dupont et al., 2022)을 스케일 업
   - Chen & Wang(2022)의 트랜스포머 인코더 개념 확장
   - 직접 확산 생성으로 전환

2. **다중 렌더링 경로 지원**:
   - NeRF 렌더링: 고충실도 뷰 합성
   - STF 렌더링: 실시간 메시 처리
   - 동일 모델로 두 경로 모두 지원하는 최초

3. **효율성과 품질의 균형**:
   - PointE와 유사한 계산 비용으로 암묵 표현 생성
   - 암묵과 명시 표현의 성능 차이 최소화

#### 8.3 최신 연구(2023-2025)와의 비교

**관련 핵심 논문들:**[8][9][10][11][12]

**1) Michelangelo (Liu et al., 2023)**[8]
- 접근: 정렬된 다중모달 잠재 공간
- 장점: 더 나은 의미론적 일관성
- ShapE와의 차이: ShapE는 더 단순한 2단계 접근

**2) OctFusion (Xiong et al., 2024)**[9]
- 접근: 옥트리 기반 잠재 표현
- 속도: 2.5초 생성 (ShapE와 유사)
- 품질: ShapeNet에서 최고 수준
- 차별점: 여러 스케일에서 가중치 공유

**3) LION (Zeng et al., 2022)**[12]
- 접근: 계층 VAE + 계층 확산
- 유연성: 다양한 조건부 작업 지원
- 한계: 여전히 점 구름 기반

**4) Direct3D (Zhang et al., 2024)**[10]
- 접근: 3D 트라이플레인 확산 트랜스포머
- 혁신: 반연속 표면 샘플링 감독
- 일반화: 매우 큰 규모로 사전학습
- ShapE와의 비교: Direct3D가 더 높은 확장성

**5) DDMI (Zeng et al., 2024)**[13]
- 혁신: 적응형 위치 임베딩 생성 (고정 위치 임베딩 대신)
- 성과: 4개 모달리티에서 SOTA
- ShapE와의 차이: 더 일반적인 암묵 표현 접근

#### 8.4 기술 트렌드 분석

**2020-2023: 기초 수립 단계**
- NeRF, DreamFusion: 3D 생성의 기초 확립
- Point-E: 확산 모델의 3D 적용
- ShapE: 암묵 함수 + 확산의 결합

**2023-2024: 스케일 확대 단계**
- Michelangelo, 3D-LDM: 더 나은 잠재 공간 설계
- OctFusion, LION: 효율적 표현 개발
- DIRECT-3D: 대규모 데이터 처리

**2024-2025: 정제 및 특화 단계**
- TripoSG: 초고해상도 달성
- CraftsMan: 다중뷰 네이티브 생성
- Direct3D, DDMI: 더 나은 일반화 성능

#### 8.5 ShapE의 후속 영향

ShapE 이후 3D 생성 연구의 발전:

1. **인코더 개선** (2023-2024):
   - DDMI: 고정 위치 임베딩의 한계 극복
   - Direct3D: 더 강력한 감독 신호 (이미지가 아닌 기하학 직접)

2. **다양한 표현 탐색** (2024):
   - TripoSG: SDF + 표면 법선 감독
   - OctFusion: 옥트리 표현
   - SALAD: 부분 레벨 암묵 표현

3. **일반화 개선** (2024-2025):
   - Direct3D: 범용 3D 기초 모델 방향
   - 이종 데이터 처리 개선
   - 범주 간 일반화 향상

***

### 9. 앞으로의 연구에 미치는 영향 및 고려사항

#### 9.1 학문적 영향

**1) 암묵 함수 생성의 새로운 패러다임:**

ShapE는 다음을 입증했습니다:

$$\text{암묵함수 생성} \approx \text{명시적 표현 생성 (효율성 측면)}$$

이는 연구자들이 더 표현력 있는 암묵 표현에 집중하도록 격려했습니다.

**2) 다중 표현 통합의 가능성:**

단일 모델이 여러 렌더링 방식을 지원할 수 있음을 보여줌으로써, 후속 연구가 더 다양한 표현을 결합하도록 영감을 주었습니다.

**3) 잠재 확산의 3D 적용:**

2D의 성공을 3D에 적용하는 실증적 사례를 제공하여, 많은 후속 작업(OctFusion, LION, Direct3D 등)의 기초가 되었습니다.

#### 9.2 실무 응용 시 고려사항

**1) 데이터 전처리:**

논문이 보여준 대로, 점 구름 생성에 충분한 뷰 수(20개)가 필요합니다:

$$\text{충분한 뷰} \rightarrow \text{완전한 점 구름} \rightarrow \text{더 나은 인코딩}$$

**2) 인코더의 품질 평가:**

Table 1에서 보듯이, 인코더의 CLIP R-Precision이 최종 생성 품질에 영향을 미칩니다:

$$\text{평가 기준: PSNR (기하학) + CLIP R-Precision (의미론)}$$

**3) 가이던스 스케일 선택:**

텍스트 조건부 모델은 매우 높은 가이던스 스케일(20.0)을 선호하며, 이미지 조건부 모델보다 다릅니다.[1]

**4) 렌더링 방식 선택:**

- NeRF: 고품질 뷰 합성 필요 시
- STF: 메시 추출 및 실시간 렌더링 필요 시
- 두 방식 모두: 작업에 따라 선택 가능

#### 9.3 향후 연구 방향

**1) 개념 합성 개선:**

현재의 한계(Figure 7)를 극복하기 위해:

- 더 큰 규모의 쌍을 이룬 3D-텍스트 데이터셋 구축
- 프롬프트 구조를 명시적으로 처리하는 모델 개발
- 장면 그래프 기반 생성 탐색

**2) 인코더 성능 향상:**

$$\text{개선 목표: PSNR} > 36 \text{dB}, \text{CLIP R-Precision} > 0.46$$

방법:
- 더 강력한 인코더 아키텍처 (Vision Transformer 등)
- 다른 감독 신호 (예: 표면 법선)
- 계층적 인코딩

**3) 최적화 기반 방법과의 하이브리드:**

Appendix D의 이미지 공간 안내를 개선하여:

- 생성 시간: 13초 → 30초 (추가)
- 품질: 현저히 향상
- 사용자 제어: 더 정밀한 조정 가능

**4) 도메인 특화 모델:**

현재 ShapE는 일반적인 3D 객체에 학습되었지만:

- 인간 모델 (avatars)
- 장면 생성
- 기계 부품 등

특정 도메인에 특화된 모델의 개발 필요

**5) 일반화 성능 연구:**

이론적 분석이 필요합니다:

$$\text{일반화 오차} = \text{근사 오차} + \text{추정 오차}$$

- 근사 오류: 인코더가 타겟을 근사할 수 있는가
- 추정 오류: 확산 모델이 잠재 분포를 학습할 수 있는가

#### 9.4 산업 적용 시 주의사항

**1) 편향 및 공정성:**

Figure 11에서 보듯이 성별 편향이 존재하므로:

- 생성된 모델이 고정관념을 강화하지 않도록 검증
- 다양한 인구통계를 반영하는 훈련 데이터 필요

**2) 안전성:**

Figure 12의 3D 인쇄 위험처럼:

- 생성된 객체가 물리적 실현 전 검증 필요
- 사용 목적에 따른 안전 기준 설정

**3) 품질 보증:**

현재 ShapE의 한계:

- 복합 개념: 신뢰할 수 없음
- 세부사항: 부분적으로 부정확할 수 있음
- 응용: 고품질 요구 시 최적화 기반 방법 추천

***

### 10. 결론

ShapE는 3D 생성 분야에서 중요한 진전을 나타냅니다. 암묵함수 매개변수를 직접 생성하면서도 PointE와 비슷한 속도를 유지하면서 동등하거나 우수한 품질을 달성했습니다. 이는 다음을 시사합니다:[1]

1. **표현 선택의 유연성**: 명시적과 암묵적 표현 간의 성능 차이가 생각보다 작음
2. **확산 모델의 일반성**: 확산 모델은 다양한 3D 표현에 적용 가능
3. **실용성의 중요성**: 초당 초의 생성 속도는 실무 응용에 필수적

그러나 여전히 **개념 합성 부족**, **세부사항 손실**, **샘플 품질 한계** 등의 문제가 남아있으며, 이는 후속 연구의 주요 과제입니다. 2023-2025년의 최신 연구들(OctFusion, DDMI, Direct3D 등)은 ShapE의 기초 위에서 이러한 한계를 극복하기 위한 노력을 계속하고 있습니다.

***

### 참고문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0e0c2dd1-b788-4dea-aeaa-5dec9f82857f/2305.02463v1.pdf)
[2](https://arxiv.org/pdf/2209.08725.pdf)
[3](https://arxiv.org/abs/2305.02463)
[4](https://arxiv.org/abs/2209.14988)
[5](https://ieeexplore.ieee.org/document/10203601/)
[6](https://ieeexplore.ieee.org/document/10378460/)
[7](https://arxiv.org/abs/2211.10440)
[8](https://arxiv.org/abs/2306.17115)
[9](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70198)
[10](https://arxiv.org/abs/2405.14832)
[11](https://ieeexplore.ieee.org/document/10376744/)
[12](https://arxiv.org/pdf/2210.06978.pdf)
[13](https://arxiv.org/abs/2401.12517)
[14](https://www.nature.com/articles/s41598-024-54861-9)
[15](https://link.springer.com/10.1007/s11263-024-02270-w)
[16](https://ieeexplore.ieee.org/document/10656208/)
[17](https://ieeexplore.ieee.org/document/10377009/)
[18](https://www.mdpi.com/2072-4292/16/10/1772)
[19](https://ieeexplore.ieee.org/document/10612788/)
[20](https://journals.sagepub.com/doi/10.1177/14780771231168233)
[21](https://arxiv.org/abs/2309.16110)
[22](https://arxiv.org/pdf/2305.02463.pdf)
[23](https://arxiv.org/abs/2212.00842)
[24](http://arxiv.org/pdf/1812.02822.pdf)
[25](https://arxiv.org/html/2402.16994v1)
[26](http://arxiv.org/pdf/2405.20853.pdf)
[27](https://arxiv.org/abs/2110.15678)
[28](https://arxiv.org/html/2311.01714)
[29](https://www.sciencedirect.com/science/article/abs/pii/S0010448525000995)
[30](https://www.youtube.com/watch?v=mMdCMaqgdtk)
[31](https://github.com/wyysf-98/CraftsMan3D)
[32](https://openaccess.thecvf.com/content/CVPR2025/papers/Hu_Turbo3D_Ultra-fast_Text-to-3D_Generation_CVPR_2025_paper.pdf)
[33](https://pubmed.ncbi.nlm.nih.gov/38315587/)
[34](https://www.youtube.com/watch?v=Dzdu4cQlS2k)
[35](https://arxiv.org/html/2501.17547v1)
[36](https://www.youtube.com/watch?v=G3xiabqcv3E)
[37](https://openaccess.thecvf.com/content/CVPR2023/papers/Shim_Diffusion-Based_Signed_Distance_Fields_for_3D_Shape_Generation_CVPR_2023_paper.pdf)
[38](https://dl.acm.org/doi/10.1145/3737902.3768358)
[39](https://arxiv.org/html/2502.06608v1)
[40](https://arxiv.org/html/2210.00379v6)
[41](https://arxiv.org/html/2509.12501v1)
[42](https://arxiv.org/pdf/2402.01166.pdf)
[43](https://ar5iv.labs.arxiv.org/html/2305.11588)
[44](https://arxiv.org/abs/2408.14732)
[45](https://arxiv.org/pdf/2403.12034.pdf)
[46](https://arxiv.org/abs/2305.11588)
[47](https://arxiv.org/abs/2210.06978)
[48](https://photonics.pl/PLP/index.php/letters/article/view/14-20)
[49](https://pnas.org/doi/10.1073/pnas.2219263119)
[50](https://ieeexplore.ieee.org/document/10657779/)
[51](https://ieeexplore.ieee.org/document/10376998/)
[52](https://link.springer.com/10.1134/S1054661824700962)
[53](https://arxiv.org/abs/2406.10000)
[54](https://arxiv.org/abs/2303.11938)
[55](https://arxiv.org/pdf/2312.08754.pdf)
[56](http://arxiv.org/pdf/2306.12422.pdf)
[57](https://arxiv.org/pdf/2212.14704.pdf)
[58](https://arxiv.org/html/2402.02972v2)
[59](https://arxiv.org/html/2406.18581)
[60](https://arxiv.org/abs/2403.14966)
[61](https://arxiv.org/html/2405.11252v1)
[62](https://www.semanticscholar.org/paper/DreamFusion:-Text-to-3D-using-2D-Diffusion-Poole-Jain/4c94d04afa4309ec2f06bdd0fe3781f91461b362)
[63](https://dagshub.com/blog/point-e/)
[64](https://www.reddit.com/r/AR_MR_XR/comments/xnl09t/nvidia_get3d_a_generative_model_of_high_quality/)
[65](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/dreamfusion/)
[66](https://www.youtube.com/watch?v=jbr6CwykkZ0)
[67](https://proceedings.neurips.cc/paper_files/paper/2022/file/cebbd24f1e50bcb63d015611fe0fe767-Paper-Conference.pdf)
[68](https://www.youtube.com/watch?v=Z6dB1zIfwr4)
[69](https://arxiv.org/abs/2212.08751)
[70](https://research.nvidia.com/labs/toronto-ai/GET3D/)
[71](https://dreamfusion3d.github.io)
[72](https://arxiv.org/pdf/2405.1824.pdf)
[73](https://openaccess.thecvf.com/content/CVPR2023/papers/Wei_TAPS3D_Text-Guided_3D_Textured_Shape_Generation_From_Pseudo_Supervision_CVPR_2023_paper.pdf)
[74](https://ar5iv.labs.arxiv.org/html/2209.14988)
[75](https://arxiv.org/html/2506.17074v1)
[76](https://openaccess.thecvf.com/content/ICCV2023/papers/Zuo_DG3D_Generating_High_Quality_3D_Textured_Shapes_by_Learning_to_ICCV_2023_paper.pdf)
[77](https://arxiv.org/html/2408.02993v1)
[78](https://arxiv.org/html/2404.16510v1)
[79](https://arxiv.org/html/2511.19512v1)
[80](https://www.reddit.com/r/MachineLearning/comments/zrfy75/n_pointe_a_new_dallelike_model_that_generates_3d/)
[81](https://syncedreview.com/2022/12/27/openais-point%C2%B7e-generating-3d-point-clouds-from-complex-prompts-in-minutes-on-a-single-gpu/)
[82](https://ieeexplore.ieee.org/document/10687563/)
[83](https://ieeexplore.ieee.org/document/10480311/)
[84](https://arxiv.org/abs/2510.17137)
[85](https://arxiv.org/abs/2506.19820)
[86](https://arxiv.org/html/2503.08737v1)
[87](http://arxiv.org/pdf/2201.00308.pdf)
[88](https://arxiv.org/html/2412.17808v3)
[89](https://arxiv.org/html/2405.14832v1)
[90](https://arxiv.org/pdf/2503.14325.pdf)
[91](https://arxiv.org/html/2503.10403v1)
[92](https://arxiv.org/html/2502.14247v2)
[93](https://research.nvidia.com/labs/toronto-ai/LION/)
[94](https://openaccess.thecvf.com/content/ICCV2023/papers/Erkoc_HyperDiffusion_Generating_Implicit_Neural_Fields_with_Weight-Space_Diffusion_ICCV_2023_paper.pdf)
[95](https://openreview.net/pdf?id=JIMZsqE8bA)
[96](https://www.launchpad.ai/blog/lion-latent-point-diffusion-models-for-3d-shape-generation)
[97](https://openreview.net/pdf?id=5KUiMKRebi)
[98](https://aair-lab.github.io/genplan25/papers/21.pdf)
[99](https://arxiv.org/abs/2207.06283)
[100](https://www.cvc.uab.es/blog/2025/06/19/advancing-transfer-learning-and-control-of-generative-image-models/)
[101](https://arxiv.org/abs/2503.08737)
[102](https://arxiv.org/html/2512.09923v1)
[103](https://arxiv.org/html/2503.00655v1)
[104](https://arxiv.org/html/2509.22407v1)
[105](https://arxiv.org/html/2511.21787v1)
[106](https://arxiv.org/html/2510.03075v1)
[107](https://arxiv.org/html/2511.20924v1)
[108](https://arxiv.org/html/2403.17869v1)
[109](https://proceedings.neurips.cc/paper_files/paper/2022/file/40e56dabe12095a5fc44a6e4c3835948-Paper-Conference.pdf)
