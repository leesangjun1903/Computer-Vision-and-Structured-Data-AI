
# DDP: Diffusion Model for Dense Visual Prediction

## 1. 논문의 핵심 주장과 주요 기여

**DDP(Diffusion Model for Dense Visual Prediction)**는 조건부 확산 모델(conditional diffusion model)을 밀집 예측 작업에 효율적으로 적용하는 프레임워크입니다. 논문의 핵심 주장은 단순하지만 강력합니다: "노이즈-투-맵(noise-to-map)" 생성 패러다임을 통해 무작위 가우시안 분포의 노이즈를 점진적으로 제거하여 밀집 예측을 수행할 수 있다는 것입니다.[1]

### 주요 기여점

1. **밀집 시각 예측의 조건부 노이징 프로세스 공식화**: DDP는 밀집 예측 작업을 일반적인 조건부 노이징 프로세스로 재구성하여, 작업별 맞춤 설계 없이 의미론적 분할, 깊이 추정, BEV 맵 분할 등 다양한 작업에 적용 가능합니다.

2. **효율적인 아키텍처 설계**: 이미지 인코더와 맵 디코더의 분리를 통해, 이미지 인코더는 한 번만 실행되고 확산 프로세스는 경량 디코더 헤드에서만 수행됩니다. 이는 기존의 무거운 U-Net 기반 방식과 근본적으로 다릅니다.

3. **다중 추론 및 불확실성 인식**: 단일 스텝 판별 방법과 달리, DDP는 동적 추론(computational cost와 예측 품질의 트레이드오프)과 자연스러운 예측 불확실성 추정을 제공합니다.

## 2. 해결하고자 하는 문제

DDP는 다음과 같은 구체적인 문제들을 해결합니다:

### 밀집 예측의 기본 문제
밀집 예측 작업은 입력 이미지 $\(x \in \mathbb{R}^{3 \times h \times w}\)$ 의 모든 픽셀에 대해 이산 레이블 또는 연속값 $\(y\)$ 를 예측해야 합니다. 기존의 판별 기반 방법들은 단일 순방향 단계로 예측하여 빠르지만, 생성 모델의 우수한 표현 학습 능력을 충분히 활용하지 못합니다.

### 기존 확산 기반 방법의 한계
선행 연구들(SegDiff, MedSegDiff 등)은 무거운 U-Net 기반 확산 모델을 적용하여 다음의 문제가 발생했습니다:
- **낮은 효율성**: 원본 이미지에 여러 번 모델을 적용하여 계산 오버헤드 증대
- **느린 수렴**: 복잡한 아키텍처 설계로 인한 학습 어려움
- **준최적 성능**: 밀집 예측 작업의 특성을 충분히 반영하지 못함

## 3. 제안하는 방법 및 수식

### 3.1 조건부 확산 모델의 기초

조건부 확산 모델은 순방향 노이징 프로세스와 역방향 디노이징 프로세스로 구성됩니다.

**순방향 프로세스 (Forward Diffusion):**

$$q(z_t | z_0) = \mathcal{N}\left(z_t; \sqrt{\bar{\alpha}_t}z_0, (1-\bar{\alpha}_t)\mathbf{I}\right) \quad (1)$$

여기서:
- $\(z_0 = y\)$ (지면 진실 맵)
- $\(\bar{\alpha}\_t := \prod_{s=0}^{t} \alpha_s = \prod_{s=0}^{t} (1-\beta_s)\)$ 는 누적 알파 값
- $\(\beta_s\)$ 는 노이즈 스케줄

**역방향 프로세스 (Reverse Diffusion):**

$$p_\theta(z_{0:T} | x) = p(z_T) \prod_{t=1}^{T} p_\theta(z_{t-1} | z_t, x) \quad (2)$$

추론 단계에서 무작위 노이즈 $\(z_T \sim \mathcal{N}(0, \mathbf{I})\)$ 로부터 시작하여, 조건 이미지 $\(x\)$ 의 안내를 받아 점진적으로 예측을 정제합니다.

### 3.2 라벨 인코딩

의미론적 분할의 이산 레이블을 처리하기 위해 DDP는 클래스 임베딩 방식을 사용합니다:

```math
\text{map\_enc} = (\sigma(\text{map\_enc}) \times 2 - 1) \times \text{scale} \quad (3)
```

여기서 $\(\sigma\)$ 는 시그모이드 함수이고, scale은 신호-대-노이즈 비율(SNR)을 제어하는 하이퍼파라미터입니다. 실험에서 최적 scale은 0.01입니다.

### 3.3 맵 손상 (Map Corruption)

훈련 중에 인코딩된 지면 진실에 가우시안 노이즈를 추가합니다:

```math
y_t = \sqrt{\bar{\alpha}_t} \cdot \text{map\_enc} + \sqrt{1-\bar{\alpha}_t} \cdot \epsilon \quad (4)
```

여기서 $\(\epsilon \sim \mathcal{N}(0, \mathbf{I})\)$ 입니다. 코사인 스케줄이 선형 스케줄보다 더 우수한 성능을 보입니다.

### 3.4 샘플링 규칙 (DDIM 업데이트)

추론 단계에서 DDP는 DDIM(Denoising Diffusion Implicit Models) 업데이트 규칙을 사용합니다:

```math
\epsilon = \frac{1}{\sqrt{1-\alpha_{\text{now}}}} \left(\text{map}_t - \sqrt{\alpha_{\text{now}}} \cdot \text{map\_enc}\right)
```

```math
\text{map}_{\text{next}} = \sqrt{\alpha_{\text{next}}} \cdot \text{map\_pred} + \sqrt{1-\alpha_{\text{next}}} \cdot \epsilon \quad (5)
```

## 4. 모델 구조

DDP의 아키텍처는 세 가지 주요 구성요소로 이루어집니다:

### 4.1 이미지 인코더 (Image Encoder)

**구조**: 입력 이미지를 받아 다양한 해상도에서 다중 스케일 특징을 추출합니다.

- **백본 네트워크**: Swin Transformer, ConvNeXt 등의 현대적 구조 지원
- **FPN(Feature Pyramid Network)**: 4개의 해상도 수준에서 특징 융합
- **특징 맵 크기**: $256 \times \frac{h}{4} \times \frac{w}{4}$

핵심 이점은 이미지 인코더가 **한 번만 실행**되어 계산 효율성을 크게 향상시킵니다.

### 4.2 맵 디코더 (Map Decoder)

**구조**: 노이징된 맵과 이미지 인코더의 조건부 특징을 결합하여 디노이징을 수행합니다.

- **구성**: 6개 레이어의 변형 어텐션(deformable attention)으로 이루어진 경량 구조
- **파라미터**: 8.4M (K-Net의 41.5M, UperNet의 31.5M과 비교)
- **입력**: 노이징된 맵 \(y_t\)와 이미지 특징의 연결(concatenation)
- **출력**: 픽셀별 분류 또는 회귀 예측

```math
y_0 = f_\theta(y_t \oplus \text{img\_feat}, t) \quad (6)
```

여기서 $\(\oplus\)$ 는 연결 연산입니다.

### 4.3 훈련 알고리즘 (Algorithm 1)

```
def train(images, maps):
    img_enc = image_encoder(images)  # 이미지 인코딩
    map_enc = encoding(maps)          # 지면 진실 인코딩
    map_enc = (sigmoid(map_enc) * 2 - 1) * scale
    
    # 지면 진실 손상
    t, eps = uniform(0, 1), normal(mean=0, std=1)
    map_crpt = sqrt(alpha_cumprod(t)) * map_enc +
               sqrt(1 - alpha_cumprod(t)) * eps
    
    # 예측 및 역전파
    map_pred = map_decoder(map_crpt, img_enc, t)
    loss = objective_func(map_pred, maps)
    return loss
```

### 4.4 추론 알고리즘 (Algorithm 2)

```
def sample(images, steps, td=1):
    img_enc = image_encoder(images)
    map_t = normal(0, 1)  # [b, 256, h/4, w/4]
    
    for step in range(steps):
        # 시간 간격
        t_now = 1 - step / steps
        t_next = max(1 - (step + 1 + td) / steps, 0)
        
        # map_0을 map_t로부터 예측
        map_pred = map_decoder(map_t, img_enc, t_now)
        
        # t_next에서의 map_t 추정
        map_t = ddim(map_t, map_pred, t_now, t_next)
    
    return map_pred
```

## 5. 성능 향상 및 주요 실험 결과

DDP는 세 가지 대표적인 밀집 예측 작업에서 최첨단 또는 경쟁력 있는 성능을 달성합니다:

### 5.1 의미론적 분할 (Semantic Segmentation)

#### ADE20K 데이터셋

| 백본 | 파라미터 | FLOPs | mIoU (Step 1) | mIoU (Step 3) |
|------|---------|-------|---------------|---------------|
| Swin-T | 40M | 113G | 46.1 | 47.0 |
| Swin-S | 61M | 136G | 48.4 | 48.7 |
| Swin-B | 99M | 173G | 49.2 | 49.4 |
| Swin-L | 207M | 285G | 53.1 | 53.2 |

DDP는 기존 비확산 기준선(44.9 mIoU)을 1.2포인트 상회하며, 3 스텝에서 추가로 0.9포인트 향상됩니다.

#### Cityscapes 데이터셋

**최고 성능**: ConvNeXt-L 백본으로 **83.9 mIoU** 달성
- Step 1: 82.95 mIoU
- Step 3: 83.21 mIoU

Mask2Former(83.30 mIoU)와 유사한 수준의 성능을 보이면서도, DDP는 동적 추론과 불확실성 추정의 추가 이점을 제공합니다.

### 5.2 BEV 맵 분할 (Bird's Eye View Segmentation)

#### nuScenes 데이터셋

| 모드 | 방법 | mIoU |
|------|------|------|
| 카메라 전용 | DDP (Step 1) | 59.3 |
| 카메라 전용 | DDP (Step 3) | 59.4 |
| 다중모달 (C+L) | DDP (Step 1) | 70.3 |
| 다중모달 (C+L) | DDP (Step 3) | **70.6** |

이전 최고 성능인 X-Align(58.0, 65.7)을 카메라 전용에서 1.3포인트, 다중모달에서 4.6포인트 초과합니다.

### 5.3 깊이 추정 (Monocular Depth Estimation)

#### KITTI 데이터셋

| 방법 | REL ↓ | RMSE ↓ | RMSE log ↓ |
|------|-------|--------|-----------|
| DepthFormer | 0.052 | 2.143 | 0.079 |
| DepthGen | 0.064 | 2.985 | 0.100 |
| **DDP (Step 3)** | **0.050** | **2.072** | **0.076** |

DDP는 동시대 확산 기반 방법인 DepthGen을 RMSE 메트릭에서 1.4포인트 상회합니다.

### 5.4 성능 향상의 기여 인자

**자기 정렬 디노이징 (Self-Aligned Denoising)**

샘플링 드리프트(sampling drift) 문제를 해결하기 위해, 훈련의 마지막 5K 반복에서 모델의 자체 예측을 사용하여 \(y_t\)를 구성합니다:

$$y_t^{\text{self}} = \sqrt{\bar{\alpha}_t} \cdot f_\theta(y_t^{\text{random}}, t) + \sqrt{1-\bar{\alpha}_t} \cdot \epsilon$$

이는 훈련과 테스트 데이터 분포를 정렬하여 성능 저하를 방지합니다.

**노이즈 스케줄 영향**

코사인 스케줄이 선형 스케줄보다 우수: 47.0 mIoU (코사인) vs 45.1 mIoU (선형)

## 6. 모델의 일반화 성능 향상 가능성

### 6.1 동적 추론 (Dynamic Inference)

DDP의 가장 주목할 만한 특징 중 하나는 계산 비용과 예측 품질 간의 트레이드오프를 조정할 수 있다는 것입니다:

**성능 궤적 (Cityscapes, ConvNeXt-T)**:

| 샘플링 스텝 | mIoU | FLOPs | FPS |
|-----------|------|-------|-----|
| 1 | 82.33 | 883G | 18 |
| 2 | 82.48 | 1.3T | 15 |
| 3 | 82.60 | 1.99T | 13 |
| 4 | 82.61 | 2.5T | 11 |

추가 스텝을 통해 지속적인 성능 향상이 가능하며, 3-5 스텝에서 포화점에 도달합니다. 이는 이미지 생성 작업(10-50 스텝)과 대조적입니다.

**이유**: 생성 작업은 점진적으로 정보를 축적하는 반면, 인식 작업은 결정에 필요한 핵심 정보를 소수의 스텝 내에 획득할 수 있습니다.

### 6.2 불확실성 인식 (Uncertainty Awareness)

DDP는 자연스럽게 픽셀 수준의 불확실성 맵을 생성합니다:

```math
\text{Uncertainty}(i,j)=\frac{\sum _{t=1}^{T}\mathbb{I}(\text{prediction\ change(pixels) \ at\ step\ }t)}{T}
```

기존 방법(베이지안 네트워크)과 달리 복잡한 모델링이 필요 없습니다.

**실증적 검증**: 불확실성 맵의 높은 반응 영역은 오류 맵의 오분류 지점과 높은 양의 상관관계를 보입니다.

### 6.3 일반화 성능을 위한 핵심 설계

**크로스도메인 강건성**

DDP는 다양한 백본 아키텍처(Swin, ConvNeXt)와 모델 크기에 일관되게 적용 가능하며, 다양한 작업(분할, BEV, 깊이)에 강건한 성능을 보입니다.

**확률론적 특성**

확산 모델의 다중 샘플링 특성은 자연스럽게 다음을 제공합니다:
- 예측 신뢰도 추정
- 앙상블 효과
- 적응적 추론 조정 가능성

## 7. 논문의 한계

### 7.1 명시적 한계

1. **계산 오버헤드**: 다중 스텝 추론의 계산 비용. 3 스텝 추론은 1 스텝 대비 약 2.2배의 FLOPs를 요구합니다.

2. **의미론적 분할에서의 경쟁력**: Mask2Former(83.30 mIoU)와 비교하여 DDP는 약간 낮은 성능을 보입니다. 분할 특화 확산 프레임워크 설계가 향후 과제입니다.

3. **도메인 특이성**: 현재 프레임워크는 시각적 밀집 예측 작업에 초점화되어 있으며, 다른 도메인으로의 확장 가능성은 검증되지 않았습니다.

### 7.2 기술적 한계

1. **샘플링 드리프트**: 훈련과 테스트 분포의 불일치로 인한 성능 저하. 자기 정렬 디노이징으로 부분적으로 완화되지만 완전히 해결되지 않습니다.

2. **라벨 인코딩 전략**: 이산 레이블에 대한 여러 인코딩 전략(원-핫, 아날로그 비트, 클래스 임베딩) 중 최적 선택이 필요하며, 각 작업 및 데이터셋에 대해 재조정이 필요할 수 있습니다.

3. **시간 간격 하이퍼파라미터**: td=1이 최적이지만, 이는 데이터셋 특이적일 수 있습니다.

## 8. 관련 최신 연구 비교 분석 (2020년 이후)

### 8.1 초기 확산 기반 분할 연구

**SegDiff (2021년)** - Wolleb et al.[2]
- **접근**: 의료 이미지 분할을 위한 확산 확률 모델 적용
- **특징**: 단순 U-Net 기반 확산 프로세스
- **한계**: 계산 비용 높음, 두 개의 파라미터 집약적 U-Net 필요

**MedSegDiff (2022년)** - Wu et al.
- **접근**: 의료 이미지 분할을 위한 확산 모델
- **성능**: 기존 방법보다 우수하나 계산 효율성 제한

### 8.2 DDP와 동시대 연구

**DepthGen (2023년)** - Saxena et al.
- **초점**: 단안 깊이 추정
- **방법**: 확산 모델 기반 깊이 예측
- **비교**: DDP는 DepthGen보다 RMSE에서 13% 개선 (2.072 vs 2.985)
- **핵심 차이**: DDP는 경량 디코더 사용으로 효율성 우월

**Pix2Seq-D (2022년)** - Chen et al.[2]
- **방법**: 비트 확산 모델을 사용한 전신 분할
- **한계**: 아날로그 비트 인코딩의 복잡성

### 8.3 이후 발전 방향 (2023년 이후)

#### 생성 모델 사전정보 활용

**DMP: Exploiting Diffusion Prior for Generalizable Dense Prediction (2023년)**[3]
- **혁신**: 사전 학습된 텍스트-이미지 확산 모델을 밀집 예측에 활용
- **방법**: 확산 프로세스를 보간 시퀀스로 재구성하여 결정론적 매핑 생성
- **강점**: 
  - 도메인 간 강건한 일반화
  - 제한된 도메인 내 학습 데이터로도 우수한 성능
- **범위**: 5개 작업 (3D 속성 추정, 의미론적 분할, 본질적 이미지 분해)

**GenPercept: What Matters When Repurposing Diffusion Models (2024년)**[4]
- **핵심 발견**:
  1. 고품질 미세조정 데이터의 중요성
  2. 확산 모델의 확률적 특성이 결정론적 인식 작업에 약간의 부정적 영향
  3. 잠재 공간 감독만이 아닌 이미지 수준 감독의 필수성
- **혁신**: 확산 모델의 다중 스텝을 제거한 결정론적 원스텝 패러다임
- **성능**: 이전 다중 스텝 방법과 비교하여 훨씬 빠른 추론 속도

**D³-Predictor: Noise-Free Deterministic Diffusion (2025년)**[5]
- **핵심 개선**: 확산 모델에서 확률적 노이즈 제거
- **방법**: 자기 감독 방식으로 노이즈 없는 완전한 확산 사전 집계
- **성능**:
  - 추론: 단일 스텝으로 경합성 성능 달성
  - 훈련: 이전보다 절반 이하의 데이터 필요
  - 제로샷 일반화: 5개 벤치마크에서 우수 성능

#### 도메인 특화 확산 모델

**FreeSeg-Diff: Training-Free Open-Vocabulary Segmentation (2024년)**[6]
- **특징**: 훈련 불필요한 제로샷 오픈어휘 분할
- **파이프라인**: 
  1. BLIP-2로부터 이미지 캡션 생성
  2. Stable Diffusion으로부터 시각 특징 추출
  3. CLIP을 사용한 텍스트 클래스 매핑
- **성능**: Pascal VOC/COCO에서 다양한 훈련 기반 방법 상회

**의료 이미지 분할 확산 모델들** (2024-2025년)
- **CMDiff**: 공간 주의와 조건부 지도를 포함한 뇌종양 분할
- **MedSegLatDiff**: VAE와 결합한 잠재 공간 확산 모델
- **Diff-UNet**: 경계 예측 분기를 포함한 3D 의료 이미지 분할

#### 통합 모델 접근

**Unified Dense Prediction of Video Diffusion (2025년)**[7]
- **혁신**: 비디오 생성과 동시에 분할 및 깊이 맵 생성
- **특징**: 단일 모델에서 생성과 밀집 예측을 통합
- **강점**: 효율성 증대, 계산 비용 증가 최소화

**Merge: Unifying Generation and Depth (2025년)**[8]
- **개념**: 고정된 사전 학습 텍스트-이미지 모델에서 출발
- **파라미터**: 추가 훈련 가능 파라미터는 12%만
- **성능**: NYU-v2에서 5.9 REL, 95.4% δ₁ 달성

### 8.4 DDP와 최신 방법의 비교 분석

| 특징 | DDP (2023) | GenPercept (2024) | D³-Predictor (2025) | DMP (2024) |
|------|-----------|-----------------|------------------|-----------|
| **아키텍처** | 경량 디코더 | 원스텝 결정론적 | 노이즈 없는 확산 | 사전 학습 T2I |
| **추론 스텝** | 3-5 | 1 | 1 | 다중 |
| **훈련 데이터** | 표준 | 표준 | 절반 이하 | 제한적 |
| **일반화** | 좋음 | 우수 | 우수 | 우수 |
| **불확실성 추정** | 자연스러움 | 제한적 | 제한적 | 제한적 |
| **동적 추론** | 지원 | 미지원 | 미지원 | 지원 |

### 8.5 연구 진화 궤적

1. **초기 단계 (2021-2022)**: 표준 확산 모델을 밀집 예측에 직접 적용 → 계산 효율성 문제

2. **최적화 단계 (2023)**: 
   - DDP: 경량 아키텍처 제안
   - DepthGen: 깊이 특화 확산 모델
   - DMP: 사전 학습 모델 활용

3. **효율성 개선 단계 (2024)**:
   - GenPercept: 원스텝 결정론적 모델
   - 도메인 특화 모델 (의료, 어휘 제한 분할)

4. **통합 및 극단적 효율화 (2025)**:
   - D³-Predictor: 노이즈 제거로 단일 스텝 달성
   - 통합 생성-예측 모델
   - 저데이터 시나리오 최적화

## 9. 논문의 앞으로의 연구에 미치는 영향

### 9.1 패러다임 변화

DDP는 밀집 예측 연구에 **생성 모델 기반 접근의 타당성**을 입증했습니다. 이전에는 판별 기반 방법이 표준이었지만, DDP는 확산 모델이 효율적이고 강력한 대안이 될 수 있음을 보여주었습니다.

### 9.2 구체적인 영향

**1. 아키텍처 설계 철학의 변화**
- 이전: 단일 종단간(end-to-end) 네트워크
- 현재: 인코더-디코더 분리, 조건부 프로세스 활용
- 미래: 더 정교한 조건부 메커니즘 개발

**2. 다중 스텝 추론의 정당성**
- 이전: 추론 속도 최우선
- 현재: 속도-정확도 트레이드오프의 명시적 관리
- 미래: 적응적 스텝 선택 메커니즘

**3. 불확실성 추정의 통합**
- 이전: 부가적 모듈 필요
- 현재: 자연스럽게 포함됨
- 미래: 불확실성 기반 의사결정 시스템

### 9.3 새로운 연구 방향 개척

DDP의 성공은 다음과 같은 새로운 연구 라인을 개척했습니다:

1. **사전 학습 모델 활용**: T2I 확산 모델의 강력한 표현을 밀집 예측에 직접 활용

2. **효율성 최적화**: 원스텝 결정론적 모델, 노이즈 없는 확산으로 진화

3. **크로스도메인 적응**: 제한된 학습 데이터로도 우수한 일반화

4. **멀티태스크 통합**: 비디오 생성 + 분할 + 깊이 추정의 통합

## 10. 향후 연구 시 고려할 점

### 10.1 기술적 고려사항

**1. 노이즈 스케줄 최적화**

```math
\text{현재}: \alpha_t = \cos\left(\frac{t + n_s}{1 + d_s} \times \frac{\pi}{2}\right)^{-2}
```

다양한 작업과 데이터셋에 대해 최적의 스케줄을 찾는 메타 학습 접근이 필요합니다.

**2. 조건부 메커니즘 개선**
- 현재: 간단한 연결 기반 조건화
- 미래: 계층적 조건화, 적응적 가중치 메커니즘
- 목표: 더 나은 특징 통합과 정보 흐름

**3. 샘플링 드리프트 완전 해결**
- 현재: 자기 정렬 디노이징 (부분적 해결)
- 미래: 훈련-테스트 분포 완전 정렬, 확률론적 보정

### 10.2 확장성 고려사항

**1. 다양한 도메인으로의 확장**
- 비정상 탐지 (anomaly detection)
- 광학 흐름 추정 (optical flow)
- 경계 추출 (edge detection)
- 초해상도 처리

**2. 3D 및 비디오 작업**
- 현재 프레임워크는 2D 이미지 중심
- 3D 볼륨 데이터에의 적용
- 시간적 일관성을 유지하는 비디오 예측

**3. 저리소스 환경 대응**
- 엣지 디바이스에 맞는 경량화
- 양자화 및 프루닝 기법 통합
- 단일 스텝 추론의 추가 최적화

### 10.3 이론적 고려사항

**1. 수렴성 분석**
- 조건부 확산 모델의 수렴 특성
- 다양한 스케줄 하에서의 수렴 속도

**2. 일반화 이론**
- 왜 DDP가 다양한 작업에 일반화되는가?
- 최소 필요 훈련 데이터 크기

**3. 불확실성 이론**
- DDP의 불확실성 추정이 보정되었는가?
- 신뢰성 있는 신뢰도 구간

### 10.4 실용적 고려사항

**1. 하이퍼파라미터 자동화**
- 최적의 scale, 스텝 수, td 값을 자동으로 선택하는 메커니즘
- 데이터셋별 설정 가이드라인 개발

**2. 계산 비용 추정**
- 정확도 목표에 따른 필요 스텝 수 예측
- 비용-효율 최적화 전략

**3. 도메인 적응**
- 소스 도메인에서의 사전 학습과 타겟 도메인으로의 적응
- 미세조정 전략 개발

## 결론

**DDP: Diffusion Model for Dense Visual Prediction**는 세 가지 측면에서 밀집 시각 예측 연구에 중요한 기여를 합니다:

1. **효율적 구조 설계**: 이미지 인코더와 맵 디코더의 분리를 통해 기존 확산 기반 방법의 계산 비효율을 극복

2. **새로운 패러다임**: "노이즈-투-맵" 생성 패러다임이 단순하면서도 강력하며 작업 간 일반화 가능

3. **다원적 이점**: 동적 추론과 불확실성 인식이라는 판별 방법에 부족한 특성 제공

이후 연구들(GenPercept, D³-Predictor, DMP 등)이 효율성과 일반화 성능을 더욱 개선했지만, DDP는 이러한 발전의 기초를 제공했습니다. 특히 조건부 확산 모델의 효율적 적용 방식에 대한 인사이트는 향후 다양한 시각 인식 작업에 계속 영향을 미칠 것으로 예상됩니다.

향후 연구자들은 DDP의 기본 원리를 바탕으로 노이즈 스케줄 최적화, 샘플링 드리프트 해결, 크로스도메인 일반화, 그리고 저리소스 환경으로의 확장에 중점을 두어야 할 것입니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/807bef15-797d-4ee9-9796-ae14c58155af/2303.17559v2.pdf)
[2](https://arxiv.org/pdf/2112.00390.pdf)
[3](https://ieeexplore.ieee.org/document/10656754/)
[4](https://www.semanticscholar.org/paper/a03ca8c7997c528d7756b5c7411209ea9ddf99c8)
[5](https://arxiv.org/html/2512.07062v1)
[6](https://ieeexplore.ieee.org/document/11227428/)
[7](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_Unified_Dense_Prediction_of_Video_Diffusion_CVPR_2025_paper.pdf)
[8](https://arxiv.org/html/2510.23574v1)
[9](https://ieeexplore.ieee.org/document/10376747/)
[10](https://arxiv.org/abs/2402.17319)
[11](https://arxiv.org/abs/2407.07853)
[12](https://www.semanticscholar.org/paper/945a899a93c03eb63be5e3197e318c077473cef9)
[13](https://ieeexplore.ieee.org/document/10913186/)
[14](https://journal.stemfellowship.org/doi/10.17975/sfj-2024-004)
[15](https://arxiv.org/abs/2311.07421)
[16](http://arxiv.org/pdf/2403.20105.pdf)
[17](https://arxiv.org/abs/2303.17559)
[18](http://arxiv.org/pdf/2306.09004.pdf)
[19](https://arxiv.org/pdf/2311.18832.pdf)
[20](https://arxiv.org/pdf/2210.17408.pdf)
[21](http://arxiv.org/pdf/2405.16947.pdf)
[22](https://arxiv.org/pdf/2303.10326.pdf)
[23](https://www.nature.com/articles/s41598-025-25137-7)
[24](https://pmc.ncbi.nlm.nih.gov/articles/PMC10490601/)
[25](https://www.sciencedirect.com/science/article/pii/S0957417425042460)
[26](https://www.sciencedirect.com/science/article/abs/pii/S1361841525002014)
[27](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Generative_Semantic_Segmentation_CVPR_2023_paper.pdf)
[28](https://arxiv.org/html/2312.08768v3)
[29](https://openaccess.thecvf.com/content/ICCV2023/papers/Ji_DDP_Diffusion_Model_for_Dense_Visual_Prediction_ICCV_2023_paper.pdf)
[30](https://openaccess.thecvf.com/content/CVPR2021/papers/Hoyer_Three_Ways_To_Improve_Semantic_Segmentation_With_Self-Supervised_Depth_Estimation_CVPR_2021_paper.pdf)
[31](https://onlinelibrary.wiley.com/doi/full/10.1155/jece/2935790)
[32](https://openaccess.thecvf.com/content/CVPR2024/papers/Lee_Exploiting_Diffusion_Prior_for_Generalizable_Dense_Prediction_CVPR_2024_paper.pdf)
[33](https://arxiv.org/html/2410.10105v3)
[34](https://arxiv.org/html/2512.04734v1)
[35](https://arxiv.org/html/2512.01292v1)
[36](https://arxiv.org/html/2505.15263v2)
[37](https://arxiv.org/html/2410.11439v3)
[38](https://arxiv.org/html/2503.09344v1)
[39](https://pmc.ncbi.nlm.nih.gov/articles/PMC11974562/)
[40](https://openreview.net/pdf/2855d25128f480c240e4539d67a0a701efd31491.pdf)
[41](https://www.nature.com/articles/s41598-025-90631-x)
