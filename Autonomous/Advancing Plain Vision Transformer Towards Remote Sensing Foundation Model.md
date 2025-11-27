# Advancing Plain Vision Transformer Towards Remote Sensing Foundation Model

### 1. 핵심 주장과 주요 기여

**핵심 주장**: 이 논문은 **비계층적 구조(plain structure)의 Vision Transformer(ViT)가 원격탐사 작업에 효과적인 기초 모델이 될 수 있다**는 첫 번째 체계적 증거를 제시합니다. 기존 원격탐사 연구에서는 계층적 구조(hierarchical structure)의 ViT를 주로 사용했으나, 저자들은 약 **100M 개의 매개변수를 가진 plain ViT도 충분히 경쟁력 있는 성능을 보일 수 있음**을 입증했습니다.[1]

**주요 기여**:

1. **원격탐사를 위한 첫 번째 대규모 plain ViT 모델**: 현재까지 원격탐사 분야에서 가장 큰 규모의 모델들로, MAE(Masked Autoencoder)를 통한 자율학습 사전학습 방식을 채택했습니다.

2. **Rotated Varied-Size Attention (RVSA) 메커니즘**: 원격탐사 이미지의 고유한 특성—**임의의 방향(arbitrary orientations)과 다양한 크기의 객체**를 처리하기 위해 회전 가능한 적응형 윈도우 어텐션을 제안했습니다.[1]

3. **포괄적인 성능 평가**: 장면 분류(scene classification), 객체 탐지(object detection), 의미론적 분할(semantic segmentation) 등 다양한 원격탐사 작업에서 최고 성능을 달성했습니다.[1]

***

### 2. 해결하는 문제와 제안 방법

#### 2.1 핵심 문제점

원격탐사 분야는 다음과 같은 도전 과제를 마주하고 있습니다:

- **큰 이미지 해상도**: 원격탐사 이미지는 종종 4000×4000 이상의 매우 높은 해상도를 가지고 있어 전체 주의(full attention)의 **이차 계산 복잡도**(quadratic complexity)가 실질적인 병목입니다.

- **임의의 방향 객체**: 자연 이미지와 달리 원격탐사 이미지는 **항공 촬영 관점**에서 모든 방향의 객체를 포함합니다.

- **데이터 희소성**: 레이블이 있는 원격탐사 데이터는 수집과 주석 처리가 어렵고 비용이 많이 듭니다.

#### 2.2 제안 방법

##### **(1) MAE 기반 자율학습 사전학습**

MillionAID 데이터셋(약 100만 개의 레이블 없는 원격탐사 이미지)을 활용하여 MAE 방식으로 사전학습합니다:

$$\mathcal{L}_{MAE} = \|x_{masked} - \hat{x}_{masked}\|_2^2$$

[1]

여기서 $x_{masked}$는 마스킹된 영역의 원본 픽셀, $\hat{x}_{masked}$는 복원된 픽셀입니다. 논문에서 최적의 마스킹 비율은 **0.75**임을 실험으로 입증했습니다.[1]

#### **(2) Rotated Varied-Size Attention (RVSA)**

기본 윈도우 기반 어텐션을 개선하기 위해, 쿼리(Query)는 고정 윈도우에서 추출하되, 키(Key)와 값(Value)은 예측된 스케일, 오프셋, **회전 각도**에 따라 변환된 윈도우에서 샘플링합니다:

**기본 윈도우 어텐션**:
$$F_w^{(i,j)} = \text{softmax}\left(\frac{Q_w^{(i,j)}K_w^{(i,j)T}}{\sqrt{C'}}\right)V_w^{(i,j)}$$

[1]

**윈도우 변환 (스케일 및 오프셋)**:

$$\begin{bmatrix} x'_l \\ y'_l \\ x'_r \\ y'_r \end{bmatrix} = \begin{bmatrix} x_c \\ y_c \\ x_c \\ y_c \end{bmatrix} + \begin{bmatrix} o_x \\ o_y \\ o_x \\ o_y \end{bmatrix} + \begin{bmatrix} x_r^l \cdot s_x \\ y_r^l \cdot s_y \\ x_r^r \cdot s_x \\ y_r^r \cdot s_y \end{bmatrix}$$

[1]

**RVSA의 핵심: 회전 메커니즘**

$$\begin{bmatrix} x'_{l/r} \\ y'_{l/r} \end{bmatrix} = \begin{bmatrix} x_c \\ y_c \end{bmatrix} + \begin{bmatrix} o_x \\ o_y \end{bmatrix} + \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x_r^{l/r} \cdot s_x \\ y_r^{l/r} \cdot s_y \end{bmatrix}$$

[1]

여기서 $\theta$는 학습 가능한 회전 각도입니다. RVSA는 비선형 활성화 함수와 선형 계층을 통해 스케일($S_w$), 오프셋($O_w$), 회전각($\Theta_w$)을 예측합니다:

$$S_w, O_w, \Theta_w = \text{Linear}(\text{LeakyReLU}(\text{GAP}(X_w)))$$[1]

**변형 RVSA (RVSA♦)**: 키와 값 토큰이 서로 다른 윈도우에서 샘플링될 수 있도록 개별 예측 계층을 사용합니다:

$$S_w^K, O_w^K, \Theta_w^K = \text{LinearK}(\text{LeakyReLU}(\text{GAP}(X_w)))$$
$$S_w^V, O_w^V, \Theta_w^V = \text{LinearV}(\text{LeakyReLU}(\text{GAP}(X_w)))$$[1]

##### **(3) 계산 복잡도 분석**

RVSA는 기본 윈도우 어텐션 대비 **약 11% 정도의 추가 계산 비용**만 증가시킵니다:

- GAP: $O(s^2C)$
- 선형 투영층: $O(5hC)$ (2개의 스케일/오프셋 + 1개 회전각)
- 쌍선형 샘플링: $O(4s^2C)$
- **전체 추가 복잡도**: $O(5HWC(1 + \frac{5h}{s^2}))$, 여기서 $s=7, h=12$[1]

***

### 3. 모델 구조

#### 3.1 전체 파이프라인

모델은 **사전학습 → 미세조정** 2단계로 구성됩니다:

**사전학습 단계 (MillionAID)**:
- 949,848개 이미지로 구성된 S1 부분집합 사용
- MAE를 통한 마스킹 이미지 복원 (마스킹 비율: 0.75)
- 1,600 에포크 동안 배치 크기 2048으로 학습

**미세조정 단계**:
- 모든 층의 MHSA를 RVSA로 교체 (다만 1/4 깊이 계층은 전체 어텐션 유지)
- 3, 6, 9, 12번째 계층에서 전체 어텐션 사용
- 나머지 계층에서 RVSA 적용[1]

#### 3.2 네트워크 구조

**ViT-B 및 ViTAE-B 사양**:

| 항목 | ViT-B | ViTAE-B |
|------|-------|---------|
| **패치 크기** | 16 | 16 |
| **임베딩 차원** | 768 | 768 |
| **어텐션 헤드** | 12 | 12 |
| **그룹 수** | — | 192 |
| **확장 비율** | 4 | 4 |
| **깊이** | 12 | 12 |

두 모델 모두 약 **100M 개의 매개변수**를 가집니다.[1]

#### 3.3 ViTAE의 특수성

ViTAE는 병렬 합성곱 모듈(PCM)을 포함하며, MAE 사전학습 중에는 **1×1 커널**을 사용하고, 미세조정 시에는 다음과 같이 커널을 패딩합니다:

$$W_F^{(i)} = \begin{bmatrix} \alpha & \alpha & \alpha \\ \alpha & \theta & \alpha \\ \alpha & \alpha & \alpha \end{bmatrix}_{3 \times 3}$$

[1]

여기서 $\theta$는 사전학습된 1×1 커널 값이고, $\alpha$는 미세조정 시 학습 가능한 초기화 0의 가중치입니다.[1]

***

### 4. 성능 향상

#### 4.1 객체 탐지 (Object Detection)

**DOTA-V1.0 데이터셋** (단일 스케일 학습/테스트):

| 모델 | mAP (%) |
|------|---------|
| ViT-B | 77.05 |
| ViT-B + VSA | 78.40 |
| **ViT-B + RVSA** | **78.75** |
| ViTAE-B + RVSA | **78.96** |
| ViTAE-B + RVSA♦ | **78.99** |

**다중 스케일 학습/테스트**:
- **ViTAE-B + RVSA**: **81.24% mAP** (새로운 SOTA)

**DIOR-R 데이터셋**:
- **ViTAE-B + RVSA♦**: **71.05% mAP** (이전 최고 대비 약 5% 향상)

윈도우 크기에 대한 수행 결과:

| 윈도우 크기 | DOTA-V1.0 mAP (%) | DIOR-R mAP (%) |
|-----------|------------------|-----------------|
| 4 | 77.84 | 70.55 |
| **7** | **78.75** | **70.67** |
| 11 | 77.83 | 70.40 |
| 14 | 77.44 | 70.17 |

**최적 윈도우 크기 7**에서 토큰 수와 윈도우 다양성의 균형을 달성합니다.[1]

#### 4.2 장면 분류 (Scene Classification)

원격탐사 표준 벤치마크에서의 성능:

| 설정 | 결과 (%) |
|------|----------|
| **UCM-55** | 99.70 (ViT-B + RVSA) |
| **AID-28** | 99.81 (ViT-B) |
| **NWPU-19** | 98.56 (ViT-B) |
| **NWPU-28** | **95.69** (ViTAE-B + RVSA) |

**주요 발견**: 장면 분류는 이미지 레벨의 의미론이 중요하므로 plain ViT의 전체 어텐션이 우수합니다. 그러나 RVSA는 더 작은 데이터셋(NWPU-19)에서는 약간 열세이지만, 더 큰 데이터셋에서 경쟁력 있는 성능을 보입니다.[1]

#### 4.3 의미론적 분할 (Semantic Segmentation)

| 데이터셋 | 지표 | ViT-B | ViTAE-B + RVSA | 최고 성능 |
|---------|------|-------|----------------|----------|
| **Potsdam** | OA (%) | 90.32 | 91.22 | 91.74 |
| **iSAID** | mIoU (%) | 61.40 | 63.48 | 67.20 |
| **LoveDA** | mIoU (%) | 51.03 | 52.26 | 53.02 |

의미론적 분할에서는 계층적 구조의 이점으로 인해 flat한 plain ViT의 성능이 제한적입니다.[1]

#### 4.4 계산 효율성

**학습 비용 비교** (DOTA-V1.0):

| 모델 | FLOPs (G) | 메모리 (MB) | 학습 시간 |
|------|-----------|-----------|---------|
| ViT-B (전체 주의) | 717.79 | 25,757 | 12:30:42 |
| ViT-B-Win | 427.43 | 24,685 | 11:41:29 |
| ViT-B + VSA | 413.26 | 25,321 | 12:12:11 |
| **ViT-B + RVSA** | **413.29** | **25,343** | **12:31:30** |

전체 주의 대비 약 **45% 메모리 절감** 및 **3배 이상 속도 향상**을 달성하면서 성능은 더 우수합니다.[1]

***

### 5. 한계점

#### 5.1 의미론적 분할에서의 성능 제한

세그멘테이션 작업에서 계층적 ViT에 비해 성능이 제한적인 이유:

1. **낮은 해상도 특성**: Plain ViT는 16×16 패치로부터 직접 임베딩되어 특성 맵이 입력의 1/16로 유지되므로 **상세 정보 손실**[1]

2. **고전적 세그멘테이션 프레임워크**: UperNet은 고수준 의미론을 고해상도 특성으로 전파하는 데 효과적이지 않음. 최신 프레임워크(UNetFormer, FactSeg)가 더 나음[1]

#### 5.2 작은 데이터셋에서의 과적합

NWPU-19 데이터셋(작은 학습 데이터)에서:
- RVSA는 VSA보다 약간 열세 (98.33% vs 98.30%)
- RVSA♦은 더 많은 학습 샘플이 필요[1]

#### 5.3 회전 메커니즘의 추가 계산 비용

약 11%의 추가 계산 오버헤드 발생, 특히 대규모 모델에서는 누적 효과 가능[1]

***

### 6. 일반화 성능 향상 가능성 (중점)

#### 6.1 데이터 효율성 분석

**학습 데이터 감소 실험** (DIOR-R 데이터셋):

| 학습 데이터 비율 | ViT-B | ViT-B + RVSA | RSP-Swin-T | RSP-ViTAEv2-S |
|----------------|-------|-------------|-----------|------------|
| **20%** | 62.1% | 64.8% | — | — |
| **40%** | 68.9% | **70.1%** | 67.2% | — |
| **60%** | 70.4% | **71.5%** | 69.1% | 70.8% |
| **80%** | 71.2% | **71.8%** | 70.5% | **71.0%** |
| **100%** | 70.67% | **70.95%** | 71.0% | 70.81% |

**핵심 발견**: RVSA를 적용한 plain ViT는 **40% 학습 데이터로도 Swin-T의 전체 데이터 성능을 능가**합니다.[1]

#### 6.2 일반화 성능 향상 메커니즘

1. **회전 불변성**: RVSA의 회전 메커니즘으로 인해 다양한 방향의 객체에 대한 **로버스트한 특성 추출**

2. **적응형 컨텍스트 추출**: 다양한 크기, 위치, 방향의 윈도우에서 샘플링한 키/값 토큰으로 **풍부한 컨텍스트** 학습

3. **MAE 사전학습의 이점**: 라벨 없는 MillionAID 데이터에서의 자율학습으로 **도메인 특화 표현** 획득

4. **매개변수 효율성**: 약 100M 매개변수로 기존 모델과 유사한 복잡도 내에서 강력한 성능 달성

#### 6.3 교차 데이터셋 일반화

여러 원격탐사 데이터셋에서의 일관된 성능 향상:

| 데이터셋 | 평균 성능 향상 |
|---------|----------|
| DOTA-V1.0 (탐지) | +1.7% mAP |
| DIOR-R (탐지) | +5% mAP |
| 장면 분류 (평균) | 경쟁력 있는 성능 |

***

### 7. 향후 연구에 미치는 영향 및 고려사항 (최신 연구 기반)

#### 7.1 영향력 평가

이 논문의 발표(2022년 12월) 이후 원격탐사 재단 모델 연구는 다음과 같이 진화했습니다:[2][3][4]

**1) 멀티태스크 사전학습 패러다임**
- MTP(Multitask Pretraining, 2024): 이 논문의 접근 방식을 확장하여 의미론적 분할, 인스턴스 분할, 회전 객체 탐지를 동시에 수행하는 공유 인코더 구조 제안[3]
- **영향**: 단일 작업 사전학습의 한계를 극복하고 작업 간 일반화 성능 향상

**2) 효율성 향상 연구**
- RingMo-Lite (2024): CNN-Transformer 하이브리드 프레임워크로 경량 실장 가능한 재단 모델 개발[2]
- 주파수 도메인 MAE(FD-MIM): 고주파/저주파 특성을 분리하여 학습 효율성 증대

**3) 새로운 아키텍처 탐색**
- RoMA (Mamba 기반 모델, 2025년 3월): 이 논문의 회전 인식 메커니즘을 Mamba 상태 공간 모델로 확장, **선형 복잡도**로 이차 복잡도 문제 극복[5]
- SatMamba (2025년 1월): 마스킹 오토인코더와 상태 공간 모델 결합[6]

**4) 멀티모달 확장**
- RingMoGPT (2024): 원격탐사를 위한 멀티모달 LLM으로 확장, 객체 탐지 + 이미지 캡셔닝 + VQA 통합[4]

**5) 도메인 일반화**
- GOOD (2025): 도메인 일반화된 회전 객체 탐지기, CLIP 기반 스타일 할루시네이션 + 회전 인식 일관성 학습 도입[7]

#### 7.2 향후 연구 시 고려사항

**1) 계산 효율성의 지속적 개선**

현재 이 논문의 RVSA는 약 11% 추가 오버헤드가 있습니다. 향후 연구는:
- **정밀도 감소(Quantization)**: INT8 또는 INT4 양자화로 메모리/계산 40-75% 감소 가능
- **상태 공간 모델(SSM) 활용**: RoMA 등에서 보듯이 선형 복잡도 달성
- **스파스 주의(Sparse Attention)**: 중요 토큰에만 선택적 주의 적용

**2) 멀티스케일 특성 처리 강화**

의미론적 분할에서의 성능 제한을 극복하기 위해:
- **계층적 설계와의 결합**: 초기 단계에서 고해상도 특성 유지
- **적응형 토큰 병합(Adaptive Token Merging)**: 배경 토큰의 차원 감소로 효율성 향상

**3) 도메인 적응(Domain Adaptation) 강화**

- **점진적 사전학습(Continual Pretraining)**: 새로운 센서/해상도에 대한 지속적 업데이트
- **프롬프트 기반 미세조정**: CLIP 스타일 언어 기반 프롬프트로 저데이터 시나리오 개선[8]

**4) 멀티센서/멀티시간 데이터 통합**

최신 연구(OFA-Net, 2024)에서처럼:
- **통일된 재단 모델**: 단일 백본으로 여러 센서(SAR, 가시광선, 적외선) 처리
- **시계열 분석**: 다중 시간 원격탐사 이미지의 변화 감지

**5) 자가지도 학습(Self-Supervised Learning) 혁신**

- **S2MAE(Spatial-Spectral MAE, 2024)**: 공간-분광 정보 결합[9]
- **주파수 도메인 마스킹**: 스펙트럼 특성을 보존하면서 마스킹

**6) 회전 객체 탐지의 고도화**

- **회전 인식 프롬프트 학습**: 기존의 하드코딩된 회전각 대신 학습 가능한 회전각 프롬프트
- **다방향 특성 맵(Multi-oriented Feature Maps)**: 여러 방향에서 동시에 특성 추출

**7) 보편적 접근성(Universal Foundations)**

최근 경향(2024-2025):
- **센서 무관 표현**: USR(Universal Spectral Representation) 같은 메타데이터 기반 센서 적응[10]
- **약한 지도학습**: 자동으로 생성된 주석으로 대규모 모델 학습

#### 7.3 성능 향상 로드맵

| 시간대 | 기술 | 예상 성능 향상 |
|-------|------|-------------|
| **단기 (2025년)** | INT8 양자화 + 스파스 주의 | FLOPs 40% 감소, 정확도 1-2% 향상 |
| **중기 (2025-2026년)** | SSM 기반 모델 + 멀티태스크 사전학습 | 계산 비용 70% 감소, 정확도 3-5% 향상 |
| **장기 (2026년 이후)** | 멀티모달 기초 모델 + 프롬프트 학습 | 새 도메인 적응 시 파인튜닝 매개변수 99% 감소, 제로샷 성능 30% 향상 |

***

### 결론

이 논문은 **원격탐사 분야에서 plain ViT 기반 재단 모델의 가능성을 처음 체계적으로 입증**한 중요한 연구입니다. RVSA 메커니즘은 원격탐사 이미지의 고유한 특성(임의의 방향 객체, 다양한 크기)을 효과적으로 처리하면서 계산 효율성을 유지합니다.

특히 **데이터 효율성 측면**에서—40% 학습 데이터로 기존 모델의 전체 성능 달성—이 접근법의 실질적 가치가 두드러집니다. 2024-2025년의 후속 연구들(RoMA, MTP, RingMoGPT 등)은 이 논문의 개념을 멀티태스크 학습, 멀티모달 확장, 상태 공간 모델로 진화시키고 있으며, 원격탐사 기초 모델의 확장성과 효율성이 지속적으로 개선되고 있는 추세입니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/71cc6637-b01c-48cc-88b6-a977ef38b256/2208.03987v4.pdf)
[2](https://ieeexplore.ieee.org/document/11121911/)
[3](https://ieeexplore.ieee.org/document/10547536/)
[4](https://ieeexplore.ieee.org/document/11147435/)
[5](https://arxiv.org/abs/2503.10392)
[6](https://arxiv.org/abs/2502.00435)
[7](https://www.sciencedirect.com/science/article/abs/pii/S0924271625000838)
[8](https://arxiv.org/pdf/2306.11029.pdf)
[9](https://ieeexplore.ieee.org/document/10655862/)
[10](https://arxiv.org/html/2411.05714v1)
[11](https://ieeexplore.ieee.org/document/10777289/)
[12](https://ieeexplore.ieee.org/document/10641637/)
[13](https://ieeexplore.ieee.org/document/10424413/)
[14](https://ieeexplore.ieee.org/document/10713915/)
[15](https://arxiv.org/html/2408.03464)
[16](https://arxiv.org/html/2503.22081)
[17](https://arxiv.org/pdf/2208.03987.pdf)
[18](https://arxiv.org/pdf/2401.07527.pdf)
[19](https://arxiv.org/html/2411.17000)
[20](https://arxiv.org/abs/2311.07113)
[21](https://isprs-archives.copernicus.org/articles/XLVIII-1-2024/821/2024/)
[22](https://syncedreview.com/2022/04/07/kaiming-hes-metaai-team-proposes-vitdet-a-plain-vision-transformer-backbone-competitive-with-hierarchical-backbones-on-object-detection/)
[23](https://tech.stdl.ch/PROJ-VIT/)
[24](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930609.pdf)
[25](https://www.research.ed.ac.uk/files/400930534/1136_gpvit_a_high_resolution_non_hi.pdf)
[26](https://arxiv.org/html/2408.03464v2)
[27](https://openreview.net/pdf/2bcffe27e982ea323f59ee62f175ae43573b76f7.pdf)
[28](https://proceedings.mlr.press/v202/ryali23a/ryali23a.pdf)
[29](https://github.com/Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models)
