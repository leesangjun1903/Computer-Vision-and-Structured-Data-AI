
# Feature Pyramid Networks for Object Detection

## 1. 핵심 주장 및 주요 기여

**Feature Pyramid Network (FPN)**은 2017년 Lin et al.이 발표한 논문으로, 객체 검출에서 **효율성과 정확도의 근본적 균형**을 달성한 획기적 아키텍처입니다. 

### 1.1 핵심 주장

FPN의 근본적 주장은 다음과 같습니다:

> **"ConvNet의 본래 계층적 특성 구조를 활용하여, 비용 효율적인 다중 스케일 특성 표현을 구축할 수 있다"** [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/beaad79e-7b90-485b-9ee2-2a668429ac86/1612.03144v2.pdf)

전통적으로 객체 검출에서 다중 스케일 처리는 이미지 피라미드 방식으로만 가능했는데, 이는 메모리 및 계산 비용이 매우 높아 테스트 시에만 사용 가능했습니다. FPN은 이 문제를 네트워크 구조적 혁신으로 해결합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/beaad79e-7b90-485b-9ee2-2a668429ac86/1612.03144v2.pdf)

### 1.2 주요 기여

1. **구조적 혁신**: 상향식(bottom-up) 경로 + 하향식(top-down) 경로 + 측면 연결(lateral connections)의 삼원 구조
2. **메모리 효율성**: 훈련/테스트 시 일관된 구조로 학습 가능
3. **범용 특성 추출기**: ResNet 등 다양한 백본 네트워크와 호환
4. **성능 향상**: 소형 객체 검출 성능 대폭 개선

***

## 2. 해결하고자 하는 문제

### 2.1 문제 정의

**다중 스케일 객체 검출의 핵심 딜레마**:

| 문제 | 세부 사항 |
|------|---------|
| **이미지 피라미드의 한계** | 각 이미지 스케일에서 독립적으로 특성 계산 → 4배 이상 연산 증가 |
| **메모리 불가능성** | 훈련 시 이미지 피라미드 사용 불가능하므로 train/test 불일치 문제 |
| **저수준 특성의 의미론적 약점** | 고해상도 특성맵은 공간 정보는 풍부하나 의미론적 정보 부족 |
| **고수준 특성의 해상도 부족** | 낮은 해상도 특성맵은 의미론적으로 강하나 소형 객체 정보 손실 |

### 2.2 기존 방식의 문제점

- **Fast/Faster R-CNN**: 단일 스케일 특성맵 사용 → 정확도 제약
- **SSD**: 고층부터 피라미드 구축 → 고해상도 특성맵 미활용
- **이미지 피라미드**: 실용적이지 않은 계산 비용

***

## 3. 제안하는 방법론 (수식 포함)

### 3.1 구조적 구성

FPN의 핵심은 세 가지 주요 성분으로 구성됩니다:

#### (1) Bottom-up Pathway (특성 계산)

ResNet 백본에서 각 단계의 마지막 레이어 출력 사용:
$$\{C_2, C_3, C_4, C_5\} \text{ with strides } \{4, 8, 16, 32\}$$

여기서 $C_i$는 해상도가 원본의 $1/2^i$인 특성맵입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/beaad79e-7b90-485b-9ee2-2a668429ac86/1612.03144v2.pdf)

#### (2) Top-down Pathway와 Lateral Connections

가장 강한 의미론적 정보를 가진 높은 수준의 특성맵에서 시작하여 단계적으로 상향식 특성과 결합:

$$P_i = F(C_i) + \text{Upsample}(P_{i+1})$$

여기서:
- $P_i$: 최종 출력 특성맵 (피라미드 레벨 $i$)
- $F(C_i)$: $1 \times 1$ 합성곱으로 채널 차원 감소 (2048 → 256)
- $\text{Upsample}(P_{i+1})$: 최근접 이웃 보간으로 2배 업샘플링

최종적으로 $3 \times 3$ 합성곱 적용하여 앨리어싱 효과 감소:

$$P_i^{\text{final}} = \text{Conv}_{3 \times 3}(P_i)$$ [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/beaad79e-7b90-485b-9ee2-2a668429ac86/1612.03144v2.pdf)

#### (3) RoI 할당 전략 (Fast R-CNN 적용 시)

RoI의 크기에 따라 적절한 피라미드 레벨로 할당:

$$k = \left\lfloor k_0 + \log_2\left(\sqrt{\frac{wh}{224}}\right)\right\rfloor$$

여기서:
- $k$: 할당된 피라미드 레벨
- $w, h$: RoI의 너비와 높이
- $k_0 = 4$: 표준 레벨
- 224: ImageNet 기준 크기 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/beaad79e-7b90-485b-9ee2-2a668429ac86/1612.03144v2.pdf)

### 3.2 모델 구조 다이어그램

```
Bottom-up path (ResNet):  C₂ ← C₃ ← C₄ ← C₅
                           ↓    ↓    ↓    ↓
Lateral connections:    1×1 conv layers
                           ↓    ↓    ↓    ↓
Top-down path:         P₂ → P₃ → P₄ → P₅
                           ↓    ↓    ↓    ↓
Final feature maps:    {P₂, P₃, P₄, P₅} (d=256 channels)
```

### 3.3 공유된 분류기/회귀기

모든 피라미드 레벨에서 **공유 가중치** 사용:

- RPN: 각 레벨에 동일한 $3 \times 3$ 컨볼루션 헤드 적용
- 앵커 박스: 각 레벨에 단일 스케일 앵커 배정 (multi-scale 앵커 불필요)
- Fast R-CNN: RoI 풀링 + 2-FC 레이어 (공유)

이는 모든 피라미드 레벨이 **의미론적으로 유사**함을 의미합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/beaad79e-7b90-485b-9ee2-2a668429ac86/1612.03144v2.pdf)

***

## 4. 성능 향상 및 한계

### 4.1 성능 향상 결과

#### RPN 성능 (COCO minival):

| 메트릭 | FPN 없음 | FPN 적용 | 향상도 |
|--------|---------|---------|--------|
| AR₁₀₀ | 36.1 | 44.0 | **+7.9** |
| AR₁ₖ | 48.3 | 56.3 | **+8.0** |
| AR_s^{1k} (소형) | 32.0 | 44.9 | **+12.9** |
| AR_m^{1k} (중형) | 58.7 | 63.4 | **+4.7** |

#### Faster R-CNN 성능 (COCO minival):

| 메트릭 | ResNet-50 기준 | FPN 적용 | 향상도 |
|--------|--------------|---------|--------|
| AP@0.5 | 53.1 | 56.9 | **+3.8** |
| AP | 31.6 | 33.9 | **+2.3** |
| AP_s | 13.2 | 17.8 | **+4.6** |

#### COCO test-dev 최종 결과 (ResNet-101):

| 지표 | 점수 |
|-----|------|
| AP@0.5 | 59.1 |
| AP | 36.2 |
| AP_s | 18.2 |
| AP_m | 39.0 |
| AP_l | 48.2 |

**결론**: 당시 COCO 2016 챌린지 우승 모델(Faster R-CNN +++)을 단일 모델로 초과 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/beaad79e-7b90-485b-9ee2-2a668429ac86/1612.03144v2.pdf)

### 4.2 주요 절제 실험 (Ablation Study)

#### Top-down 경로의 중요성:

$$\text{Table 1(e)}: \text{Top-down 제거} \Rightarrow \text{AR}_{1k} \text{ 10점 하락}$$

**인사이트**: Bottom-up 피라미드만으로는 충분하지 않음. ResNet 특히 깊은 네트워크에서 계층 간 의미론적 차이가 큼 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/beaad79e-7b90-485b-9ee2-2a668429ac86/1612.03144v2.pdf)

#### 측면 연결의 중요성:

$$\text{Table 1(d)}: \text{Lateral 제거} \Rightarrow \text{AR}_{1k} \text{ 약 10점 하락}$$

**인사이트**: 정확한 위치 정보 전파의 핵심. 저해상도 특성을 반복 업샘플링하면 위치 정보 손실 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/beaad79e-7b90-485b-9ee2-2a668429ac86/1612.03144v2.pdf)

#### 피라미드 표현의 필요성:

P₂ 단일 레벨 사용: 33.4 AP (FPN 33.9 AP와 비교)

**인사이트**: 최고 해상도 특성만으로도 유사 성능이나, RPN이 이미 다중 스케일 제안 생성 중

### 4.3 운영 특성

| 항목 | 값 |
|-----|-----|
| 추론 시간 (GPU) | 0.148초/이미지 (ResNet-50) |
| 비교 기준선 | 0.32초 (단일 스케일 Faster R-CNN) |
| 속도 향상 | **약 2.16배** |
| 학습 시간 | ~10시간 (8 GPU COCO) |

### 4.4 방법의 한계

| 한계 | 설명 |
|-----|------|
| **고정된 채널 차원** | 모든 피라미드 레벨에서 256채널로 고정 (정보 손실 가능성) |
| **단순 특성 융합** | 요소 단위 덧셈만 사용 (가중치 없는 융합) |
| **큰 객체 성능** | AP_l 개선이 상대적으로 제한적 (소형 객체 중심) |
| **계산 오버헤드** | 추가 1×1, 3×3 합성곱 레이어 필요 |
| **앵커 설계 의존** | 앵커 박스 설계에 여전히 의존 |

***

## 5. 모델의 일반화 성능 향상 가능성

### 5.1 현황 분석

FPN의 핵심 강점은 **스케일 불변성(Scale Invariance)**입니다:

$$\text{작은 객체 크기 변화} \Leftrightarrow \text{피라미드 레벨에서의 위치 변화}$$

이는 다음과 같은 일반화 개선을 가능하게 합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/beaad79e-7b90-485b-9ee2-2a668429ac86/1612.03144v2.pdf)

### 5.2 일반화 성능 향상 메커니즘

#### (1) 크로스 도메인 전이 학습 (2020-2024 연구)

최근 연구 에 따르면: [ecva](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930609.pdf)
- **사전학습 중요성 증가**: ImageNet-21K, JFT-300M 등 대규모 데이터셋 사용
- **자기 지도 학습**: SSL은 도메인 간 특성 이전성 향상
- **Transformer 백본**: Vision Transformers는 CNN보다 분포 변화에 강함

#### (2) 도메인 일반화 (2023-2025 최신)

**도메인 일반화 (Domain Generalization)** 연구가 급속 확대 중: [arxiv](https://arxiv.org/html/2203.05294v5)

$$\mathcal{D}_{\text{train}} \Rightarrow \mathcal{D}_{\text{unseen}}$$

관련 방법들:
- **OA-DG** (2023): 객체 인식 데이터 증강 + 손실 함수 개선
- **MonoGDG** (2024): 기하학적 특성 기반 도메인 분리
- **스타일 일반화**: CLIP 기반 스타일 추상화

성능 개선: 기준선 대비 **4.9-7.9% mAP 향상** [arxiv](https://arxiv.org/html/2312.08875)

#### (3) 다중 모달 표현 학습

**CLIP 기반 방법** (2024-2025): [arxiv](https://arxiv.org/pdf/2504.14280.pdf)
- Zero-shot 검출 능력
- 언어-이미지 정렬을 통한 일반화
- 스타일, 조명, 날씨 변화에 강건

### 5.3 개선된 FPN 변형들 (2020년 이후)

#### EfficientDet의 BiFPN (2020):

$$P_i^{\text{in}} = \frac{w_i \cdot P_i^{\text{in}}}{\varepsilon + \sum_j w_j \cdot P_j^{\text{in}}}$$

**특성**: 가중치 학습으로 중요도 차별화 → **4점 AP 향상** [openaccess.thecvf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tan_EfficientDet_Scalable_and_Efficient_Object_Detection_CVPR_2020_paper.pdf)

#### PANet의 Path Aggregation (2018):

Bottom-up augmentation으로 정보 경로 단축:
$$\text{FPN: 100+ 레이어} \Rightarrow \text{PANet: <10 레이어}$$

**성능**: COCO 2017 1위 달성 [arxiv](https://arxiv.org/pdf/1803.01534.pdf)

#### 최신 개선 (2023-2025):

| 방법 | 핵심 아이디어 | 성능 |
|-----|-------------|------|
| **MSE-FPN** | 의미론적 강화 + 게이트 채널 안내 | +2.1 AP (ResNet-50) |
| **BAFPN** | 양방향 정렬 모듈 (SPAM) | 고정밀 위치 결정 |
| **HA-FPN** | 계층적 주의 메커니즘 | 계산 오버헤드 최소 |
| **LR-FPN** | 원격 감지 최적화 + 위치 정제 | DOTA 성능 향상 |

### 5.4 Transformer 기반 검출로의 진화

**Vision Transformer + FPN 조합** (2023-2025):

- **DETR**: 트랜스포머 기반 end-to-end 검출 (NMS 제거)
- **RT-DETR** (2024): 실시간 하이브리드 인코더 + 효율적 다중 스케일 처리
- **Deformable DETR**: 변형 가능한 주의로 공간 관계 개선

**일반화 장점**:
- 전역 컨텍스트 자동 포착
- 도메인 시프트에 더 강건
- Zero-shot 학습 가능성

***

## 6. 앞으로의 연구 영향 및 고려 사항

### 6.1 2020년 이후 FPN의 지속적 영향

#### (1) 기본 아키텍처 표준화

FPN은 현재 **객체 검출의 거의 모든 주류 방법의 필수 구성 요소**입니다:

- **YOLO 계열**: YOLOv3~v11 모두 FPN 또는 PANet 기반 다중 스케일 구조 사용
- **Faster R-CNN 파생**: Mask R-CNN, Cascade R-CNN 등
- **한 단계 검출기**: RetinaNet, EfficientDet, YOLOv4 등
- **Transformer 검출**: DETR, RT-DETR에서 멀티 스케일 특성 처리 기본

### 6.2 향후 주요 연구 방향

#### 1) **도메인 일반화/적응의 심화** (핫 토픽 2024-2025)

**핵심 연구 영역**:
- **단일 도메인 일반화 (SDG)**: 제한된 source 데이터로 unseen target에 일반화 [arxiv](https://arxiv.org/html/2312.12133v1)
- **지속 학습 (Continual Learning)**: 테스트 시 동적 도메인 변화 대응 [arxiv](https://arxiv.org/html/2312.08875)
- **기하학적 일반화**: 3D 객체 검출에서 기하 특성 보존 [ise.thss.tsinghua.edu](https://ise.thss.tsinghua.edu.cn/mig/2024-3.pdf)

$$\text{성능 개선}: \text{기준선 대비 } 4.9\%-7.9\% \text{ mAP}$$

#### 2) **효율화 (Lightweight & Edge)** 

**구체적 노력**:
- **경량 백본**: MobileNet, ShuffleNet, GhostNet 등과 FPN 조합
- **양자화 & 지식 증류**: 모바일 배포용 모델 경량화
- **신경 아키텍처 탐색 (NAS)**: NAS-FPN, OPANAS 등 자동 최적화

**현황**: 1-2MB 모델로도 실용적 성능 달성 가능 [onlinelibrary.wiley](https://onlinelibrary.wiley.com/doi/10.1002/mop.34012)

#### 3) **멀티 모달 학습과 통합**

**최신 동향 (2024-2025)**:
- **CLIP 기반 검출**: 언어-이미지 정렬로 zero-shot 일반화 [arxiv](https://arxiv.org/pdf/2504.14280.pdf)
- **텍스트 기반 검출**: 자연어 지시로 객체 검출 가능
- **3D 멀티뷰 통합**: 카메라 + LiDAR 기반 다중 센서 데이터 처리

#### 4) **작은 객체 검출의 지속적 개선**

**남은 도전**:
- 초소형 객체 ($<32$ 픽셀): 여전히 10-20 AP 이하 수준
- 조밀한 객체 장면: 군집 객체 분할 어려움
- 극저해상도 이미지: 정보 손실로 인한 성능 저하

**해결 방향**:
- **Denoising FPN** (2024): 노이즈 제거로 정확도 향상 [arxiv](https://arxiv.org/pdf/2406.05755.pdf)
- **고주파/공간 인식 FPN** (2024): 고주파 정보와 공간 인식 결합 [arxiv](https://arxiv.org/html/2412.10116v1)
- **스케일 시퀀스 특성**: 3D CNN으로 스케일 간 관계성 학습

### 6.3 구현 시 고려사항

#### 1) **특성 채널 차원 선택**

$$d = 256 \text{ (현재 표준)}$$

- 장점: 계산 효율과 성능 균형
- 향후 고려: 계층별 적응형 채널 크기

#### 2) **가중치 융합의 중요성**

EfficientDet 이후 표준 관행:

$$P_i = \text{Normalize}\left(\sum_j w_j \cdot P_j\right)$$

- 학습 가능한 $w_j$로 특성 중요도 자동 조절
- 기본 덧셈 대비 2-4% AP 향상

#### 3) **Back-bone 선택**

| 백본 | 특성 | 추천 |
|-----|------|------|
| **ResNet** | 표준, 안정적 | 기본 선택 |
| **Vision Transformer** | 전역 컨텍스트 우수 | 도메인 일반화 |
| **EfficientNet** | 효율성 우수 | 모바일/엣지 |
| **ConvNeXt** | 현대화 CNN | 고성능 (2024+) |

#### 4) **멀티 태스크 학습 고려**

최신 추세 (2024-2025):
- 단순 검출 + 추가 태스크 조합
- 의미론 분할, 인스턴스 분할, 깊이 추정 등
- **개선 효과**: 각각 2-5% 성능 향상

***

## 7. 2020년 이후 관련 최신 연구 비교 분석

### 7.1 특성 융합 방법 진화

| 시기 | 방법 | 핵심 혁신 | 성능 (COCO) |
|-----|-----|---------|-----------|
| **2017** | **FPN** | 상향식 + 하향식 + 측면 연결 | 36.2 AP |
| **2018** | **PANet** | Bottom-up 경로 증강 + 적응형 풀링 | 37.8 AP |
| **2019** | **NAS-FPN** | 신경 아키텍처 탐색 | 38.3 AP |
| **2020** | **EfficientDet** | BiFPN (가중치 융합) + 복합 스케일링 | 52.6 AP |
| **2021** | **A2-FPN** | 주의 기반 집계 | 40.5 AP |
| **2023** | **MSE-FPN** | 의미론적 강화 + 게이트 채널 | 42.2-43.4 AP |
| **2024** | **LR-FPN** | 위치 정제 (원격 감지) | DOTA 향상 |
| **2024** | **BAFPN** | 양방향 정렬 모듈 (SPAM) | 고정밀화 |
| **2024** | **DNTR** | 노이징 제거 FPN + Transformer | AI-TOD: 26.2 AP |

### 7.2 도메인 일반화/적응 (Hot topic 2023-2025)

| 접근법 | 핵심 원리 | 성과 |
|--------|---------|------|
| **OA-DG** (2023) | 객체 인식 혼합 + 손실 설계 | 4.9% mAP 향상 |
| **MonoGDG** (2024) | 기하 기반 특성 분리 | 단일 도메인 일반화 우수 |
| **CLIP 기반** (2024-25) | 언어-이미지 정렬 | Zero-shot 가능 |
| **연속 학습** (2024) | 테스트 시 적응 | 7.9% 향상 (SHIFT) |
| **Diffusion 기반** (2025) | 확산 모델 기반 정렬 | DA/DG 통합 우수 |

### 7.3 Real-time 검출 진화 (Transformer 영향)

| 모델 | 아키텍처 | 특성 | 성능 |
|-----|---------|------|------|
| **YOLOv5** (2020) | CNN + PANet | 모듈식 설계 | 48.5 AP |
| **YOLO-MS** (2023) | CNN + 적응형 다중 스케일 | 스케일 동적 선택 | 50.3 AP |
| **RT-DETR** (2024) | Transformer + 하이브리드 인코더 | 하이브리드, 효율적 | 48-52 AP |
| **YOLOE** (2025) | YOLO11 + 멀티 태스크 | 검출 + 분할 + 임베딩 | 53+ AP |

**주목**: Real-time 검출에서도 50 AP 이상 달성 가능해짐

### 7.4 소형/극소형 객체 검출 특화 (2023-2025)

| 방법 | 기법 | 개선 사항 |
|-----|-----|---------|
| **HS-FPN** | 고주파 + 공간 인식 FPN | 소형 객체 특화 |
| **DNTR** | 노이징 제거 + Transformer | 극소형 객체 (AI-TOD: 26.2 AP) |
| **SRD-YOLOv5** (2025) | 스케일 시퀀스 + MSFE | UAV: +10.6% AP (자전거) |
| **Octave-YOLO** (2024) | 옥타브 합성곱 | 정보 손실 감소 |

**성과**: 초소형 객체 AP가 2020년 대비 **50-100% 향상**

### 7.5 효율성 vs 정확도 트렌드

```
        정확도
          ↑
          │        YOLOE (2025)
          │        RT-DETR (2024)
          │    ╱────────╲
          │   ╱          ╲
        52├──────────────╲─── BiFPN/EfficientDet
          │              ╲
        48├────────────────╲─ YOLOv5 / PANet
          │                ╲
        36├──────────────────╲ FPN (2017)
          │
          └────────────────→ 효율성 (FPS/파라미터)
          
        2017 → 2024 발전 추세: 
        좌상향 이동 (같은 효율로 정확도 향상)
```

***

## 8. 결론

### 8.1 FPN의 지속적 유효성

**7년 이상이 지난 지금도 FPN은**:
1. **근본적 구조**: 모든 주류 검출기의 기본
2. **개선의 기초**: BiFPN, PANet, MSE-FPN 등 모두 FPN 개념 확장
3. **실용적 도구**: 경량부터 고성능까지 다양한 실현 가능

### 8.2 주요 발전 방향

1. **도메인 일반화**: 현실 배포의 핵심 → 4.9-7.9% 성능 향상 중
2. **효율화 + 정확도**: 경량 모델도 50+ AP 달성 가능
3. **멀티 모달 통합**: CLIP, 3D, 텍스트 등과 결합 중
4. **Transformer 통합**: 전역 컨텍스트 + 다중 스케일 처리 개선

### 8.3 앞으로의 고려 사항

**학술 연구 방향**:
- 단일 도메인 일반화의 근본적 해결
- 극소형/초밀도 객체 검출 개선
- 효율성과 정확도의 파레토 최적화

**실무 적용 시**:
- 도메인 적응 기법 우선 적용 (배포 환경 특성)
- 경량 모델 + 고성능 백본 선택 신중
- 멀티 태스크 학습으로 부가 이득 활용

***

## 참고 문헌

<span style="display:none">[^1_100][^1_101][^1_102][^1_103][^1_104][^1_105][^1_106][^1_107][^1_108][^1_109][^1_110][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_90][^1_91][^1_92][^1_93][^1_94][^1_95][^1_96][^1_97][^1_98][^1_99]</span>

<div align="center">⁂</div>

[^1_1]: 1612.03144v2.pdf

[^1_2]: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930609.pdf

[^1_3]: https://arxiv.org/html/2203.05294v5

[^1_4]: https://arxiv.org/html/2312.08875

[^1_5]: https://arxiv.org/pdf/2504.14280.pdf

[^1_6]: https://openaccess.thecvf.com/content_CVPR_2020/papers/Tan_EfficientDet_Scalable_and_Efficient_Object_Detection_CVPR_2020_paper.pdf

[^1_7]: https://arxiv.org/pdf/1803.01534.pdf

[^1_8]: https://arxiv.org/html/2312.12133v1

[^1_9]: https://ise.thss.tsinghua.edu.cn/mig/2024-3.pdf

[^1_10]: https://onlinelibrary.wiley.com/doi/10.1002/mop.34012

[^1_11]: https://www.mdpi.com/2079-9292/12/24/4936

[^1_12]: https://arxiv.org/pdf/2406.05755.pdf

[^1_13]: https://arxiv.org/html/2412.10116v1

[^1_14]: https://www.epidemvac.ru/jour/article/view/2017

[^1_15]: http://vestnik.mednet.ru/content/view/1700/30/lang,ru/

[^1_16]: http://www.cdc.gov/mmwr/volumes/73/rr/rr7305a1.htm?s_cid=rr7305a1_w

[^1_17]: https://jurnal.unismabekasi.ac.id/index.php/maslahah/article/view/4456

[^1_18]: https://biss.pensoft.net/article/112666/

[^1_19]: https://essd.copernicus.org/articles/16/3495/2024/

[^1_20]: https://photonics.pl/PLP/index.php/letters/article/view/16-27

[^1_21]: https://www.tandfonline.com/doi/full/10.1080/10106049.2024.2387786

[^1_22]: https://ieeexplore.ieee.org/document/10164213/

[^1_23]: https://journal.unj.ac.id/unj/index.php/jpud/article/view/43940

[^1_24]: https://arxiv.org/pdf/2103.10643.pdf

[^1_25]: https://arxiv.org/pdf/1911.09070.pdf

[^1_26]: https://arxiv.org/pdf/1904.07392.pdf

[^1_27]: http://arxiv.org/pdf/2404.01614.pdf

[^1_28]: https://arxiv.org/pdf/2412.01859.pdf

[^1_29]: https://www.mdpi.com/1424-8220/23/9/4508

[^1_30]: http://arxiv.org/pdf/1909.01122.pdf

[^1_31]: https://arxiv.org/pdf/2105.09464.pdf

[^1_32]: https://peerj.com/articles/cs-1824/

[^1_33]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0332408

[^1_34]: https://arxiv.org/html/2410.11301v1

[^1_35]: https://arxiv.org/pdf/1612.03144.pdf

[^1_36]: https://arxiv.org/html/2308.05480v2

[^1_37]: https://arxiv.org/pdf/2412.17325.pdf

[^1_38]: https://arxiv.org/html/2306.15988v1

[^1_39]: https://arxiv.org/html/2408.04326v1

[^1_40]: https://arxiv.org/html/2412.10116v2

[^1_41]: https://arxiv.org/pdf/2206.14098.pdf

[^1_42]: https://arxiv.org/html/2510.26641v1

[^1_43]: https://arxiv.org/html/2404.01614v1

[^1_44]: https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html

[^1_45]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0300120

[^1_46]: https://www.nature.com/articles/s41598-023-34277-7

[^1_47]: https://proceedings.mlr.press/v162/dimitrakopoulos22a/dimitrakopoulos22a.pdf

[^1_48]: https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf

[^1_49]: https://www.nature.com/articles/s41598-024-60897-8

[^1_50]: https://www.sciencedirect.com/science/article/abs/pii/S0893608026000079

[^1_51]: https://yai-yonsei.tistory.com/12

[^1_52]: https://www.igi-global.com/article/multi-scale-feature-aligned-for-object-detection/376935

[^1_53]: https://ieeexplore.ieee.org/iel8/6287639/10380310/10584528.pdf

[^1_54]: https://arxiv.org/abs/2012.00779

[^1_55]: https://www.sciencedirect.com/science/article/abs/pii/S0031320324007349

[^1_56]: https://wikidocs.net/162976

[^1_57]: https://velog.io/@davidlyoo/Feature-Pyramid-Network-Paper-Review-Enhancing-Object-Detection-with-Feature-Pyramids

[^1_58]: https://aclanthology.org/2020.semeval-1.142

[^1_59]: https://www.semanticscholar.org/paper/61e4d1f47b9553f2a6c38f78693033caa20c4f38

[^1_60]: https://link.springer.com/10.1007/s40012-020-00302-7

[^1_61]: https://ieeexplore.ieee.org/document/11084092/

[^1_62]: https://arxiv.org/abs/2510.03876

[^1_63]: https://www.mdpi.com/2076-3417/12/13/6600

[^1_64]: https://onlinelibrary.wiley.com/doi/10.1002/cpe.70375

[^1_65]: https://www.mdpi.com/2077-1312/13/10/1936

[^1_66]: https://downloads.hindawi.com/journals/cin/2022/2262549.pdf

[^1_67]: https://www.mdpi.com/1424-8220/21/8/2799/pdf

[^1_68]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8071535/

[^1_69]: https://arxiv.org/pdf/2302.06052.pdf

[^1_70]: https://www.mdpi.com/1424-8220/23/17/7619/pdf?version=1693801772

[^1_71]: https://www.mdpi.com/2072-4292/14/3/516/pdf?version=1643001744

[^1_72]: https://www.mdpi.com/1424-8220/22/15/5817/pdf?version=1659597008

[^1_73]: https://arxiv.org/pdf/2208.11533.pdf

[^1_74]: https://pdfs.semanticscholar.org/dfed/483726be78605b1586ae8d5825f9b57ba170.pdf

[^1_75]: https://www.arxiv.org/pdf/1911.09070v5.pdf

[^1_76]: https://openaccess.thecvf.com/content/CVPR2021/papers/Liang_OPANAS_One-Shot_Path_Aggregation_Network_Architecture_Search_for_Object_Detection_CVPR_2021_paper.pdf

[^1_77]: https://arxiv.org/html/2402.06784v2

[^1_78]: https://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Path_Aggregation_Network_CVPR_2018_paper.pdf

[^1_79]: https://arxiv.org/html/2503.06282v2

[^1_80]: https://arxiv.org/abs/1911.09070

[^1_81]: https://openaccess.thecvf.com/content/CVPR2021/papers/Hu_A2-FPN_Attention_Aggregation_Based_Feature_Pyramid_Network_for_Instance_Segmentation_CVPR_2021_paper.pdf

[^1_82]: https://arxiv.org/html/2502.02322v1

[^1_83]: https://openaccess.thecvf.com/content/WACV2021/papers/Gong_Effective_Fusion_Factor_in_FPN_for_Tiny_Object_Detection_WACV_2021_paper.pdf

[^1_84]: https://arxiv.org/html/2509.00351v1

[^1_85]: https://suminizz.github.io/panet/

[^1_86]: https://escholarship.org/uc/item/1q2683zm

[^1_87]: https://www.sciencedirect.com/science/article/abs/pii/S0924271625000838

[^1_88]: https://velog.io/@raziel/EfficientDet2020

[^1_89]: https://deep-learning-study.tistory.com/637

[^1_90]: https://hw-hk.tistory.com/354

[^1_91]: https://blog.naver.com/siniphia/221490098283

[^1_92]: https://arxiv.org/html/2402.06784v1

[^1_93]: https://velog.io/@hseop/2018-CVPR-Path-Aggregation-Network-for-Instance-Segmentation

[^1_94]: https://openreview.net/forum?id=fBlRnKDHEl

[^1_95]: https://ieeexplore.ieee.org/document/10568180/

[^1_96]: https://arxiv.org/abs/2410.22461

[^1_97]: https://ieeexplore.ieee.org/document/10769077/

[^1_98]: https://ieeexplore.ieee.org/document/10241054/

[^1_99]: https://ieeexplore.ieee.org/document/10873705/

[^1_100]: https://ieeexplore.ieee.org/document/10633799/

[^1_101]: https://ieeexplore.ieee.org/document/10379128/

[^1_102]: https://ieeexplore.ieee.org/document/10363633/

[^1_103]: https://ieeexplore.ieee.org/document/10773828/

[^1_104]: https://www.mdpi.com/2076-3417/14/18/8109

[^1_105]: http://arxiv.org/pdf/2107.13389.pdf

[^1_106]: https://arxiv.org/abs/1910.11319

[^1_107]: http://arxiv.org/pdf/2108.12612.pdf

[^1_108]: https://arxiv.org/pdf/2301.00371.pdf

[^1_109]: https://arxiv.org/html/2405.14497

[^1_110]: https://arxiv.org/abs/2311.10845작성 기준**: 2026년 1월 31일, 최신 논문 기준: 2020-2025년 발표 자료
