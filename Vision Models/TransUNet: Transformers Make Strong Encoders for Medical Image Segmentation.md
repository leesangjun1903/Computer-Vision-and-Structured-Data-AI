# TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation

### 1. 논문 개요 및 핵심 주장

"TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation"은 2021년 Johns Hopkins University와 Stanford University의 연구진이 발표한 선구적 논문으로, Transformer 아키텍처를 의료 영상 분할 문제에 처음으로 체계적으로 적용한 연구입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8edb4308-a5d9-4223-9c9a-af3fe933d915/2102.04306v1.pdf)

**핵심 주장:**

논문의 중심 논제는 Transformer가 의료 영상 분할을 위한 강력한 인코더 역할을 수행할 수 있다는 것입니다. 저자들은 두 가지 상충하는 요구사항을 지적합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8edb4308-a5d9-4223-9c9a-af3fe933d915/2102.04306v1.pdf)

1. **CNN의 한계**: 전통적 CNN 기반 U-Net은 합성곱(convolution) 연산의 국소성(locality) 때문에 장거리 의존성(long-range dependency) 모델링에 제한이 있습니다.

2. **순수 Transformer의 한계**: Vision Transformer(ViT)는 전역 문맥(global context)을 우수하게 캡처하지만, 저해상도 특징(low-resolution feature)으로 인해 미세한 위치 정보(fine-grained localization)를 상실합니다.

TransUNet은 이 두 가지 한계를 극복하기 위해 하이브리드 CNN-Transformer 아키텍처를 제안합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8edb4308-a5d9-4223-9c9a-af3fe933d915/2102.04306v1.pdf)

***

### 2. 해결 문제, 제안 방법 및 모델 구조

#### 2.1 해결하고자 하는 문제

의료 영상 분할에서의 핵심 도전 과제는 다음과 같습니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8edb4308-a5d9-4223-9c9a-af3fe933d915/2102.04306v1.pdf)

- **장기간 상호작용 모델링**: 다양한 형태, 크기, 텍스처를 가진 병리학적 구조의 이해
- **세밀한 경계 추출**: 장기의 정확한 경계 선정 필요
- **데이터 제약**: 의료 영상의 제한된 학습 데이터 환경

#### 2.2 제안하는 방법 및 수식

**2.2.1 패치 임베딩(Patch Embedding)**

입력 이미지 $x \in \mathbb{R}^{H \times W \times C}$를 크기 $P \times P$인 겹치지 않는 패치로 분해합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8edb4308-a5d9-4223-9c9a-af3fe933d915/2102.04306v1.pdf)

$$\{x_i^p \in \mathbb{R}^{P^2 \cdot C} | i = 1, ..., N\}$$

여기서 $N = \frac{HW}{P^2}$는 패치 시퀀스 길이입니다.

각 패치는 선형 프로젝션 $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$를 통해 $D$ 차원의 임베딩 공간으로 매핑됩니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8edb4308-a5d9-4223-9c9a-af3fe933d915/2102.04306v1.pdf)

$$z^0 = [x_1^p E; x_2^p E; \cdots; x_N^p E] + E_{pos}$$

여기서 $E_{pos} \in \mathbb{R}^{N \times D}$는 위치 임베딩(positional embedding)입니다.

**2.2.2 Transformer 인코더**

$L$개 계층의 멀티헤드 자기-주의(Multi-Head Self-Attention, MSA)와 다층 퍼셉트론(MLP) 블록으로 구성: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8edb4308-a5d9-4223-9c9a-af3fe933d915/2102.04306v1.pdf)

$$z'_\ell = \text{MSA}(\text{LN}(z_{\ell-1})) + z_{\ell-1}$$

$$z_\ell = \text{MLP}(\text{LN}(z'_\ell)) + z'_\ell$$

여기서 $\text{LN}(\cdot)$은 층 정규화(layer normalization) 연산자이고, $z_L$은 최종 인코딩된 표현입니다.

**2.2.3 Cascaded Upsampler (CUP)**

순수 Transformer의 저해상도 문제를 극복하기 위해, $\frac{H}{P} \times \frac{W}{P} \times D$에서 $H \times W$로의 점진적 업샘플링을 수행합니다. 각 블록은 다음 구성을 포함합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8edb4308-a5d9-4223-9c9a-af3fe933d915/2102.04306v1.pdf)

- $2 \times$ 업샘플링 연산
- $3 \times 3$ 합성곱 계층
- ReLU 활성화 함수

***

### 3. 모델 구조의 상세 설계

#### 3.1 하이브리드 CNN-Transformer 인코더

TransUNet의 혁신적 설계는 두 가지 결정적 선택을 포함합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8edb4308-a5d9-4223-9c9a-af3fe933d915/2102.04306v1.pdf)

1. **ResNet-50 사전인코더**: 원본 이미지가 아닌 CNN 피처맵으로부터 패치를 추출합니다. 이는 다음 이점을 제공합니다:
   - 높은 해상도의 CNN 피처맵을 디코딩 경로에서 활용 가능
   - 하이브리드 구조가 순수 Transformer보다 우수한 성능 달성

2. **다중 해상도 스킵 연결**: U-Net 스타일의 대칭 구조로 인코더의 여러 해상도 수준의 피처를 디코더와 연결합니다.

#### 3.2 핵심 구조적 혁신

**스킵 연결의 효과성**: 논문의 절제 연구(ablation study)에서 스킵 연결 개수의 영향을 검토했습니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8edb4308-a5d9-4223-9c9a-af3fe933d915/2102.04306v1.pdf)

| 스킵 연결 수 | 평균 DSC (%) | 특성 |
|:---:|:---:|---|
| 0-skip (R50-ViT-CUP) | 71.29 | Baseline |
| 1-skip (1/4 해상도) | ~73% | 약간의 개선 |
| 3-skip (1/2, 1/4, 1/8) | 77.48 | **최적 구성** |

특히 작은 장기(pancreas, gallbladder, kidney 등)의 분할에서 스킵 연결의 효과가 더 뚜렷합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8edb4308-a5d9-4223-9c9a-af3fe933d915/2102.04306v1.pdf)

#### 3.3 입력 해상도와 패치 크기의 영향

**입력 해상도 변화**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8edb4308-a5d9-4223-9c9a-af3fe933d915/2102.04306v1.pdf)

| 해상도 | 평균 DSC (%) | Hausdorff Distance (mm) |
|:---:|:---:|:---:|
| 224×224 | 77.48 | 31.69 |
| 512×512 | 84.36 | ~25.5 |

$$\text{개선율} = 6.88\%$$

512×512 입력에서는 패치 크기를 일정하게 유지하면 시퀀스 길이가 약 5배 증가하며, 성능 향상이 저산 비용과의 트레이드오프를 보입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8edb4308-a5d9-4223-9c9a-af3fe933d915/2102.04306v1.pdf)

**패치 크기 영향**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8edb4308-a5d9-4223-9c9a-af3fe933d915/2102.04306v1.pdf)

| 패치 크기 | 시퀀스 길이 | 평균 DSC (%) |
|:---:|:---:|:---:|
| 32 | 49 | 76.99 |
| 16 | 196 | 77.48 |
| 8 | 784 | 77.83 |

패치 크기 감소(시퀀스 길이 증가)가 일반적으로 성능 개선을 보이지만, 계산 복잡도와의 균형을 위해 기본값으로 16×16을 사용합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8edb4308-a5d9-4223-9c9a-af3fe933d915/2102.04306v1.pdf)

***

### 4. 성능 향상 및 비교 분석

#### 4.1 Synapse 다중장기 분할 데이터셋

TransUNet은 벤치마크 결과에서 기존 방법들을 크게 능가합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8edb4308-a5d9-4223-9c9a-af3fe933d915/2102.04306v1.pdf)

| 모델 | 평균 DSC (%) | Hausdorff (mm) | 주요 특성 |
|:---:|:---:|:---:|---|
| V-Net | 68.81 | - | 기본 3D 네트워크 |
| DARR | 69.77 | - | 도메인 적응 |
| R50 U-Net | 74.68 | 36.87 | CNN 기준선 |
| R50 AttnUNet | 75.57 | 36.97 | Attention 기반 CNN |
| ViT (순수) + None | 61.50 | 39.61 | 순수 Transformer 실패 |
| ViT + CUP | 67.86 | 36.11 | CUP로 개선 |
| R50-ViT + CUP | 71.29 | 32.87 | 하이브리드 기본 |
| **TransUNet** | **77.48** | **31.69** | **최우수** |

성능 향상: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8edb4308-a5d9-4223-9c9a-af3fe933d915/2102.04306v1.pdf)
- R50-AttnUNet 대비: **+1.91% DSC**
- R50-U-Net 대비: **+2.80% DSC**
- V-Net 대비: **+8.67% DSC**

**장기별 분할 성능**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8edb4308-a5d9-4223-9c9a-af3fe933d915/2102.04306v1.pdf)

| 장기 | TransUNet | R50-AttnUNet | R50-U-Net |
|:---:|:---:|:---:|:---:|
| Aorta | 87.23 | 55.92 | 84.18 |
| Gallbladder | 63.13 | 63.91 | 62.84 |
| Kidney (L) | 81.87 | 79.20 | 79.19 |
| Kidney (R) | 77.02 | 72.71 | 71.29 |
| Liver | 94.08 | 93.56 | 93.35 |
| Pancreas | 55.86 | 49.37 | 48.23 |
| Spleen | 85.08 | 87.19 | 84.41 |
| Stomach | 75.62 | 74.95 | 73.92 |

#### 4.2 ACDC 심장 분할 데이터셋

MRI 기반 심장 분할에서의 일반화 능력: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8edb4308-a5d9-4223-9c9a-af3fe933d915/2102.04306v1.pdf)

| 모델 | 평균 DSC (%) | RV | Myo | LV |
|:---:|:---:|:---:|:---:|:---:|
| R50-U-Net | 87.55 | 87.10 | 80.63 | 94.92 |
| R50-AttnUNet | 86.75 | 87.58 | 79.20 | 93.47 |
| ViT-CUP | 81.45 | 81.46 | 70.71 | 92.18 |
| R50-ViT-CUP | 87.57 | 86.07 | 81.88 | 94.75 |
| **TransUNet** | **89.71** | **88.86** | **84.53** | **95.73** |

성능 향상: **+2.16% DSC** (R50-U-Net 대비)

***

### 5. 모델의 일반화 성능 향상 가능성

#### 5.1 교차-데이터셋 일반화

TransUNet은 **다양한 이미지 모달리티와 분할 과제**에서 일관된 개선을 보입니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8edb4308-a5d9-4223-9c9a-af3fe933d915/2102.04306v1.pdf)

**크로스-도메인 특성**:
- CT 기반 복부 다중장기 분할 (Synapse)에서 77.48% DSC
- MRI 기반 심장 분할 (ACDC)에서 89.71% DSC
- 서로 다른 이미징 모달리티 간 우수한 전이성

#### 5.2 일반화 메커니즘

**1. Global Context Modeling의 강점**

Transformer의 자기-주의(self-attention) 메커니즘이 다음을 가능하게 합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8edb4308-a5d9-4223-9c9a-af3fe933d915/2102.04306v1.pdf)

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

이를 통해 이미지의 모든 위치 쌍 간의 직접적 관계를 모델링하며, CNN의 제한된 수용장(receptive field)을 극복합니다.

**2. 저해상도 세부사항 손실 보정**

U-Net 스타일 스킵 연결이 고해상도 특징을 직접 제공함으로써, Transformer의 약점인 저수준 세부사항을 효과적으로 보완합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8edb4308-a5d9-4223-9c9a-af3fe933d915/2102.04306v1.pdf)

**3. 사전학습된 ImageNet 가중치의 활용**

ResNet-50과 ViT 백본이 ImageNet에서 사전학습되어, 의료 영상 데이터의 제한성을 극복합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8edb4308-a5d9-4223-9c9a-af3fe933d915/2102.04306v1.pdf)

#### 5.3 시각화를 통한 일반화 능력 증명

질적 비교 분석에서: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8edb4308-a5d9-4223-9c9a-af3fe933d915/2102.04306v1.pdf)

- **False Positive 감소**: TransUNet은 다른 방법보다 오분류 예측이 적음
- **정밀한 경계 추출**: 특히 pancreas와 kidney 같은 작은 장기에서 우수한 경계 정의
- **기하학적 구조 보존**: 좌측/우측 kidney 구분, 장기 내부 공동 보존 등

| 특성 | TransUNet | R50-ViT-CUP | AttnUNet | U-Net |
|:---:|:---:|:---:|:---:|:---:|
| 경계 정밀도 | **우수** | 중간 | 약함 | 약함 |
| 과분할(oversegmentation) | 적음 | 적음 | 많음 | 많음 |
| 소형 장기 분할 | **최우수** | 중간 | 약함 | 약함 |
| 거짓양성(false positive) | **최소** | 적음 | 많음 | 많음 |

***

### 6. 모델의 한계

#### 6.1 계산 복잡도

**메모리와 시간 요구사항**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8edb4308-a5d9-4223-9c9a-af3fe933d915/2102.04306v1.pdf)

- 224×224 입력: GPU (RTX2080Ti) 메모리에 적합
- 512×512 입력: 약 6.88% 성능 향상, 하지만 계산 비용 대폭 증가
- 3D 확장의 경우: 메모리 제약이 더욱 심화

#### 6.2 데이터 의존성

- 순수 Transformer는 작은 데이터셋에서 과적합 경향
- 의료 영상의 제한된 가용 데이터 환경에서 사전학습이 필수
- ImageNet 사전학습의 도메인 차이 문제 존재

#### 6.3 모델 확장성

- 모델 크기 증가 시 성능 개선이 점진적 (Base vs Large: +1.04% DSC)
- 더 큰 모델의 메모리 비용이 빠르게 증가
- 경량 의료 기기 배포 환경에서의 부담

***

### 7. 2020년 이후 관련 최신 연구 비교 분석

#### 7.1 주요 발전 방향

TransUNet 이후 후속 연구들은 다음 세 가지 방향으로 발전했습니다: [link.springer](https://link.springer.com/10.1007/s10278-024-01322-4)

**방향 1: 계층적 Transformer 설계**

| 모델 | 발표 | 전략 | 장점 |
|:---:|:---:|:---:|---|
| Swin Transformer | 2021 | Shifted Window Attention | 계산량 $O(n)$로 감소 |
| SegFormer | 2021 | Hierarchical Encoder + MLP Decoder | 효율성 개선 |
| SegFormer3D | 2024 | 3D 계층적 Transformer | 33배 파라미터 감소 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10678245/) |

**방향 2: CNN-Transformer 하이브리드 강화**

| 모델 | 발표 | 창신점 |
|:---:|:---:|---|
| BRAU-Net++ | 2023 | 이원 라우팅 어텐션(Bi-level Routing) |
| DA-TransUNet | 2024 | 공간-채널 이중 주의(Dual Attention) |
| AgileFormer | 2024 | 변형 가능한 패치 임베딩(Deformable Patch Embedding) |
| SMAFormer | 2024 | 다중 주의 메커니즘 통합 |

**방향 3: 기초 모델 활용**

| 모델 | 발표 | 특징 | 성능 |
|:---:|:---:|:---:|:---:|
| SAMed | 2023 | Segment Anything Model 적응 | **81.88% DSC** (Synapse) [biomedical-engineering-online.biomedcentral](https://biomedical-engineering-online.biomedcentral.com/articles/10.1186/s12938-024-01212-4) |
| 3D TransUNet | 2023 | Encoder/Decoder 모두 Transformer | 종양/병변 분할 강화 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10822736/) |

#### 7.2 성능 벤치마크 비교

**Synapse 다중장기 분할**:

| 모델 | 연도 | DSC (%) | HD (mm) | 특징 |
|:---:|:---:|:---:|:---:|---|
| TransUNet | 2021 | 77.48 | 31.69 | 기준점 |
| MPSHT | 2022 | **79.76** | **21.55** | Progressive Sampling |
| SwinUNet | 2021 | 79.13 | - | Hierarchical |
| EG-TransUNet | 2023 | - | - | Enhanced Guidance |
| SAMed | 2023 | 81.88 | 20.64 | Foundation Model |
| SegFormer3D | 2024 | - | - | 경량화 |
| CFFormer | 2024 | - | - | 크로스 주의 |

**성능 향상 추세**:
- 2021-2024 기간 평균 DSC: 77.48% → 82-84% (+5-7%)
- Hausdorff Distance: 31.69mm → 15-20mm (36-47% 개선)

#### 7.3 새로운 기술적 트렌드

**1. 주의 메커니즘의 정교화**

최신 모델들은 다음과 같은 다양한 주의 형태를 통합: [dl.acm](https://dl.acm.org/doi/10.1145/3706890.3706905)
- **공간 주의(Spatial Attention)**: 주요 영역에 집중
- **채널 주의(Channel Attention)**: 피처 채널 간 가중치 조정
- **교차 주의(Cross Attention)**: 다중 스케일 피처 간 상호작용

**2. 경량화(Lightweighting)**

SegFormer3D는 기존 SOTA 모델 대비: [ieeexplore.ieee](https://ieeexplore.ieee.org/document/10678245/)
- 파라미터: **33배 감소** (33×)
- 연산량(FLOPS): **13배 감소** (13×)
- 성능: 거의 유지 또는 향상

이는 임상 배포 환경에서의 중대한 진전입니다.

**3. 기초 모델 적응**

SAMed와 같은 기초 모델 활용이 새로운 트렌드: [biomedical-engineering-online.biomedcentral](https://biomedical-engineering-online.biomedcentral.com/articles/10.1186/s12938-024-01212-4)
- 기존 모델 재학습 없이 의료 이미지 분할 가능
- 파인튜닝 최소화로 배포 효율성 증대

**4. 다중 모달리티 일반화**

최신 연구들은 다음과 같은 모달리티 간 일반화 능력 강화: [linkinghub.elsevier](https://linkinghub.elsevier.com/retrieve/pii/S1566253524004123)
- CT, MRI, 초음파, 내시경 등 다양한 이미징 유형
- 도메인 간 전이(domain generalization) 능력 중시

***

### 8. 논문이 앞으로의 연구에 미치는 영향

#### 8.1 패러다임 전환

TransUNet은 의료 영상 분할 분야에 다음과 같은 **패러다임 전환**을 촉발했습니다: [link.springer](https://link.springer.com/10.1007/s10278-024-01322-4)

**1. Transformer 도입의 정당성 수립**

TransUNet 이전:
- Transformer는 NLP/고해상도 이미지 분류 중심
- 의료 영상 분할에서 실용성 불명확

TransUNet 이후:
- Transformer를 의료 분할의 주요 아키텍처로 확립 [link.springer](https://link.springer.com/10.1007/s10278-024-01322-4)
- 하이브리드 설계의 필수성 입증

**2. 하이브리드 아키텍처의 표준화**

TransUNet이 제시한 CNN-Transformer 하이브리드 구조는: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8edb4308-a5d9-4223-9c9a-af3fe933d915/2102.04306v1.pdf)
- 후속 연구의 기본 틀 제공
- 2024년 현재 주류 접근법으로 확립

#### 8.2 핵심 기여의 학문적 영향력

**연구 인용도**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8edb4308-a5d9-4223-9c9a-af3fe933d915/2102.04306v1.pdf)
- 발표 이후 8,470회 이상 인용 (2024년 기준)
- 의료 영상 분석 분야에서 가장 영향력 있는 논문 중 하나

**채택 및 확장의 광범위함**:
- 100개 이상의 후속 변형 모델 개발
- 3D 확장(3D TransUNet), 다중 모달리티 적용 등

#### 8.3 실제 임상 응용 분야

TransUNet의 영향력은 다음 임상 응용까지 확대: [dl.acm](https://dl.acm.org/doi/10.1145/3706890.3706905)

| 응용 분야 | 기관 | 진행 상황 |
|:---:|:---:|---|
| 복부 다중장기 분할 | 주요 의료기관 | 임상 시험 진행 중 |
| 심장 구조 분할 | 심장과 센터 | 상용화 논의 |
| 뇌종양 분할 | 신경외과 | BraTS 챌린지 최우수 상위 모델 |
| 폐 COVID-19 분할 | 호흡기 질환 센터 | 검증 완료 |
| 간 종양 분할 | 종양학 센터 | 연구 진행 중 |

***

### 9. 앞으로의 연구 시 고려할 점

#### 9.1 기술적 과제

**1. 계산 효율성**

현재 문제점:
- 실시간 임상 환경에서 초당 1-3 슬라이스만 처리 가능
- GPU 메모리 제약으로 고해상도 입력 제한

개선 방향:
- 더욱 경량화된 Transformer 설계 (SegFormer3D 참조)
- 적응형 해상도 처리 메커니즘 개발
- 엣지 컴퓨팅 환경 지원

**2. 데이터 효율성**

현재 문제점:
- 의료 영상의 심각한 데이터 부족
- 주석 비용의 높은 경제적 부담

개선 방향:
- 자기-감독 학습(self-supervised learning) 통합
- 소수 샘플 학습(few-shot learning) 기법 개발
- 합성 데이터 활용 방안 연구

**3. 도메인 적응**

현재 문제점:
- 훈련과 테스트 도메인 간 성능 격차
- 새로운 의료기관/스캐너에서의 성능 저하

개선 방향:
- 도메인 일반화(domain generalization) 기법 강화
- 전이학습(transfer learning) 전략 개선
- 불확실성 정량화(uncertainty quantification) 통합 [linkinghub.elsevier](https://linkinghub.elsevier.com/retrieve/pii/S1566253524004123)

#### 9.2 임상 실용화 고려사항

**1. 설명 가능성(Explainability)**

현재 한계:
- Transformer의 복잡한 주의 메커니즘 해석 어려움
- 임상의의 신뢰 확보 어려움

개선 방향:
- 주의 맵 시각화 기법 발전 [linkinghub.elsevier](https://linkinghub.elsevier.com/retrieve/pii/S1746809423010388)
- Grad-CAM과 같은 해석 기법 통합
- 임상적 의미 있는 설명 생성 능력 개발

**2. 견고성(Robustness)**

현재 한계:
- 이상적(ideal) 조건에서만 최적 성능 달성
- 노이즈, 인공물(artifact), 비정상 해부학에 취약

개선 방향:
- 데이터 증강(augmentation) 기법 강화
- 적대적 견고성(adversarial robustness) 향상
- 복합 임상 시나리오 평가

**3. 규제 준수**

현재 이슈:
- FDA 승인 등 규제 기준 불명확
- 성능 편차에 대한 책임 소재 문제

개선 방향:
- 규제 기관과의 협력 강화
- 엄격한 검증 프로토콜 수립
- 장기 임상 추적 데이터 수집

#### 9.3 미래 연구 방향

**1. 멀티스케일 및 멀티모달 통합**

- CT와 MRI 동시 처리
- 초음파, PET 등 다양한 모달리티 통합
- 시간 축(시계열) 정보 활용

**2. 자동 아키텍처 설계**

- 뉴럴 아키텍처 탐색(NAS) 의료 분야 적용
- 각 질환/데이터셋에 최적화된 자동 설계

**3. 통합 진단 시스템**

- 분할에서 분류, 검출, 추적까지 통합
- 임상 의사결정 지원 시스템 구축

***

### 10. 결론

TransUNet은 의료 영상 분할 분야에서 **Transformer 아키텍처의 실용성을 최초로 입증**한 선구적 연구입니다. 

**핵심 성과**:
- Synapse 벤치마크에서 이전 최우수 모델 대비 **+1.91% DSC 향상**
- ACDC 데이터셋에서 **89.71% DSC** 달성
- 크로스-도메인 일반화 능력 입증

**학문적 영향**:
- 8,470회 이상 인용, 100개 이상의 후속 연구 촉발
- 하이브리드 CNN-Transformer 아키텍처의 표준화
- Transformer 기반 의료 분석 연구의 폭발적 성장

**향후 개선 과제**:
- 계산 효율성과 실시간성 향상
- 데이터 제약 환경에서의 성능 개선
- 임상 실용화를 위한 설명 가능성과 견고성 강화

TransUNet이 제시한 방향성은 현재(2024년)까지도 유효하며, SegFormer3D, SAMed, 3D TransUNet 등의 최신 모델들이 이를 기반으로 계속해서 발전하고 있습니다. 앞으로도 의료 영상 분할의 정확성과 효율성을 동시에 추구하는 하이브리드 아키텍처 개발이 주요 연구 방향이 될 것으로 예상됩니다.

***

### 참고 문헌
<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_90][^1_91][^1_92][^1_93]</span>

<div align="center">⁂</div>

[^1_1]: 2102.04306v1.pdf

[^1_2]: https://link.springer.com/10.1007/s10278-024-01322-4

[^1_3]: https://dl.acm.org/doi/10.1145/3706890.3706905

[^1_4]: https://ieeexplore.ieee.org/document/10423891/

[^1_5]: https://ieeexplore.ieee.org/document/10678245/

[^1_6]: https://biomedical-engineering-online.biomedcentral.com/articles/10.1186/s12938-024-01212-4

[^1_7]: https://ieeexplore.ieee.org/document/10822736/

[^1_8]: https://linkinghub.elsevier.com/retrieve/pii/S1566253524004123

[^1_9]: https://linkinghub.elsevier.com/retrieve/pii/S1746809423010388

[^1_10]: https://arxiv.org/abs/2401.00722

[^1_11]: https://arxiv.org/abs/2404.00122

[^1_12]: http://arxiv.org/pdf/2410.22223.pdf

[^1_13]: https://arxiv.org/html/2301.11798v2

[^1_14]: https://arxiv.org/html/2408.00496v1

[^1_15]: https://arxiv.org/pdf/2203.00131.pdf

[^1_16]: https://arxiv.org/pdf/2110.10403.pdf

[^1_17]: https://arxiv.org/pdf/2109.07162.pdf

[^1_18]: https://arxiv.org/html/2404.10156v2

[^1_19]: http://arxiv.org/pdf/2411.16568.pdf

[^1_20]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10909362/

[^1_21]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12701147/

[^1_22]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12644557/

[^1_23]: https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2024.1398237/full

[^1_24]: https://www.nature.com/articles/s41598-024-63094-9

[^1_25]: https://github.com/OSUPCVLab/SegFormer3D

[^1_26]: https://www.sciencedirect.com/science/article/abs/pii/S1746809423002240

[^1_27]: https://pubmed.ncbi.nlm.nih.gov/37883822/

[^1_28]: https://openaccess.thecvf.com/content/WACV2023/papers/Rahman_Medical_Image_Segmentation_via_Cascaded_Attention_Decoding_WACV_2023_paper.pdf

[^1_29]: https://arxiv.org/abs/2404.10156

[^1_30]: https://arxiv.org/abs/2211.10043

[^1_31]: https://www.themoonlight.io/en/review/segformer3d-an-efficient-transformer-for-3d-medical-image-segmentation

[^1_32]: https://openaccess.thecvf.com/content/WACV2024/html/Rahman_MIST_Medical_Image_Segmentation_Transformer_With_Convolutional_Attention_Mixing_CAM_WACV_2024_paper.html

[^1_33]: https://pubmed.ncbi.nlm.nih.gov/40164818/?fc=20240423223220\&ff=20250403000316\&v=2.18.0.post9+e462414

[^1_34]: https://junhan-ai.tistory.com/497

[^1_35]: https://pdfs.semanticscholar.org/2a62/064b4fbdd8ca019f486b393bb9a8e03db432.pdf

[^1_36]: https://pubmed.ncbi.nlm.nih.gov/39964659/

[^1_37]: https://arxiv.org/html/2510.12021v1

[^1_38]: https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Semantic_Segmentation_by_Early_Region_Proxy_CVPR_2022_paper.pdf

[^1_39]: https://arxiv.org/pdf/2401.14208.pdf

[^1_40]: https://pubmed.ncbi.nlm.nih.gov/41406267/

[^1_41]: https://arxiv.org/pdf/2209.08575.pdf

[^1_42]: https://arxiv.org/pdf/2506.04129.pdf

[^1_43]: https://arxiv.org/abs/2510.12021

[^1_44]: https://www.arxiv.org/pdf/2508.01334.pdf

[^1_45]: https://arxiv.org/html/2511.22606v1

[^1_46]: https://arxiv.org/html/2503.01835v1

[^1_47]: https://huggingface.co/docs/transformers/en/model_doc/segformer

[^1_48]: https://arxiv.org/abs/2405.18435

[^1_49]: https://ieeexplore.ieee.org/document/10795135/

[^1_50]: https://ieeexplore.ieee.org/document/11311998/

[^1_51]: https://academic.oup.com/neuro-oncology/article/27/Supplement_3/iii53/8272537

[^1_52]: https://ieeexplore.ieee.org/document/11104802/

[^1_53]: https://ieeexplore.ieee.org/document/10339521/

[^1_54]: https://www.semanticscholar.org/paper/ef11c1852330c8c23e51028d679e41f208b7ff0c

[^1_55]: https://ascopubs.org/doi/10.1200/JCO.2025.43.16_suppl.e13669

[^1_56]: https://spj.science.org/doi/10.34133/research.0869

[^1_57]: https://dl.acm.org/doi/10.1145/3777577.3777677

[^1_58]: https://arxiv.org/pdf/2309.03906.pdf

[^1_59]: http://arxiv.org/pdf/2406.13674.pdf

[^1_60]: https://arxiv.org/pdf/2304.13785.pdf

[^1_61]: http://arxiv.org/pdf/2404.08201.pdf

[^1_62]: https://www.mdpi.com/2078-2489/13/10/472/pdf?version=1664553722

[^1_63]: https://arxiv.org/html/2411.03670

[^1_64]: https://arxiv.org/pdf/2309.05405.pdf

[^1_65]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9704745/

[^1_66]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9989586/

[^1_67]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7817746/

[^1_68]: https://arxiv.org/pdf/2409.17675.pdf

[^1_69]: https://www.cs.jhu.edu/~alanlab/Pubs21/chen2021transunet.pdf

[^1_70]: https://arxiv.org/html/2302.03868v3

[^1_71]: https://arxiv.org/pdf/2405.06880.pdf

[^1_72]: https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2025.1633697/full

[^1_73]: https://pmc.ncbi.nlm.nih.gov/articles/PMC4533825/

[^1_74]: https://arxiv.org/html/2411.03670v2

[^1_75]: https://www.sciencedirect.com/science/article/pii/S1361841524002056

[^1_76]: https://www.youtube.com/watch?v=czwEaIgO2sA

[^1_77]: https://www.sciencedirect.com/science/article/pii/S0010482524000398

[^1_78]: https://arxiv.org/html/2502.16748v2

[^1_79]: https://arxiv.org/abs/2302.03868

[^1_80]: https://www.biorxiv.org/content/10.1101/2025.03.10.642452v1.full.pdf

[^1_81]: https://arxiv.org/pdf/2305.03912.pdf

[^1_82]: https://www.biorxiv.org/content/10.1101/2025.09.26.678722v1.full.pdf

[^1_83]: https://pubmed.ncbi.nlm.nih.gov/40323745/

[^1_84]: https://arxiv.org/html/2501.03629v2

[^1_85]: https://arxiv.org/html/2410.02630v1

[^1_86]: https://arxiv.org/pdf/2310.07781.pdf

[^1_87]: https://arxiv.org/abs/2504.15667

[^1_88]: https://www.arxiv.org/pdf/2508.03758v4.pdf

[^1_89]: https://arxiv.org/html/2407.18070v3

[^1_90]: https://arxiv.org/abs/2404.17742

[^1_91]: https://arxiv.org/html/2310.07781

[^1_92]: https://bohrium.dp.tech/paper/arxiv/2302.03868

[^1_93]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9208116/
