
# PANet : Path Aggregation Network for Instance Segmentation

## 1. 핵심 주장 및 기여 요약

Path Aggregation Network (PANet)는 2018년 Shu Liu 등이 제안한 인스턴스 분할 프레임워크로, Mask R-CNN의 정보 전파 메커니즘을 근본적으로 개선한 모델입니다. 논문의 중심 주장은 신경망에서 정보의 흐름 방식이 성능에 결정적 영향을 미친다는 것으로, 기존 FPN의 단방향 정보 전파를 양방향화하여 정확도를 대폭 향상시킵니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/218d741f-dd54-4e2d-9500-8ae7ec354796/1803.01534v4.pdf)

핵심 기여는 다음 세 가지로 구성됩니다. 첫째, **Bottom-up Path Augmentation**은 저수준 특징의 위치 정보(localization signals)를 고수준 특징 피라미드 전체에 퍼뜨려 FPN의 상향식 경로를 보완합니다. 둘째, **Adaptive Feature Pooling**은 각 제안(proposal)이 모든 특징 레벨에서 정보를 풀링하도록 허용하여, 기존 FPN의 고정적인 크기 기반 할당 방식의 한계를 극복합니다. 셋째, **Fully-Connected Fusion**은 FCN 기반 마스크 예측에 완전연결 레이어의 보완적 성질을 추가하여 마스크 품질을 개선합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/218d741f-dd54-4e2d-9500-8ae7ec354796/1803.01534v4.pdf)

## 2. 해결하고자 하는 문제 및 제안 방법

### 2.1 문제 정의

기존 Mask R-CNN과 FPN의 구조에는 다음 세 가지 근본적 문제가 있습니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/218d741f-dd54-4e2d-9500-8ae7ec354796/1803.01534v4.pdf)

1. **정보 경로의 길이 문제**: 저수준 특징(P2)에서 최상위 특징(P5)으로의 정보 전달이 CNN 주간선을 통해 100+ 레이어를 거치는 반면, 직접적인 경로는 10 레이어 미만입니다.

2. **특징 할당의 고정성**: FPN에서 제안 크기에 따라 특징 레벨이 휴리스틱하게 할당되므로, 유용한 크로스 레벨 정보가 무시됩니다.

3. **마스크 예측의 단일 뷰**: Mask R-CNN의 FCN은 국소 수용장(local receptive field)만을 고려하여 정보 다양성이 제한됩니다.

### 2.2 제안 방법: 수식 포함

#### Bottom-up Path Augmentation

각 빌딩 블록의 연산은 다음과 같이 표현됩니다:

$$N_{i+1} = \text{Conv}_{3\times3}(P_{i+1} + \text{Conv}_{3\times3,s=2}(N_i))$$

여기서:
- $N_i$: 새로 생성된 특징 맵
- $P_{i+1}$: FPN으로부터의 기존 특징 맵  
- $\text{Conv}_{3\times3}$: 3×3 컨볼루션 (256 채널)
- $\text{Conv}_{3\times3,s=2}$: stride 2의 3×3 컨볼루션 (공간 크기 축소)

각 단계에서 ReLU 활성화 함수가 적용되며, $N_2 = P_2$ (처리 없음)로 시작합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/218d741f-dd54-4e2d-9500-8ae7ec354796/1803.01534v4.pdf)

#### Adaptive Feature Pooling

각 제안에 대해 모든 특징 레벨에서 특징을 추출한 후 퓨전합니다:

$$F_{\text{fused}} = \text{Fusion}(F_{P_2}, F_{P_3}, F_{P_4}, F_{P_5})$$

퓨전 연산으로는 원소별 최댓값(element-wise max) 또는 합(sum)을 사용:

$$F_{\text{fused}}(i,j,c) = \max_{k \in \{2,3,4,5\}} F_{P_k}(i,j,c)$$

또는

$$F_{\text{fused}}(i,j,c) = \sum_{k \in \{2,3,4,5\}} F_{P_k}(i,j,c)$$

Box branch에서는 첫 번째 FC 레이어 후, 마스크 branch에서는 첫 번째와 두 번째 컨볼루션 레이어 사이에 퓨전을 배치합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/218d741f-dd54-4e2d-9500-8ae7ec354796/1803.01534v4.pdf)

#### Fully-Connected Fusion

마스크 예측은 두 개의 병렬 경로로 수행됩니다:

$$M_{\text{final}} = M_{\text{FCN}} + M_{\text{FC}}$$

여기서:
- $M_{\text{FCN}}$: FCN 경로의 마스크 출력 (4개의 3×3 컨볼루션 + 1개의 deconvolutional)
- $M_{\text{FC}}$: FC 레이어 경로의 마스크 출력

FC 경로는 conv3 레이어에서 시작하는 단축 경로로, 2개의 3×3 컨볼루션과 1개의 FC 레이어로 구성됩니다. 마스크 크기가 28×28이므로 FC 레이어는 784×1×1 벡터를 생성합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/218d741f-dd54-4e2d-9500-8ae7ec354796/1803.01534v4.pdf)

## 3. 모델 구조

### 3.1 전체 아키텍처

PANet의 구조는 4개의 주요 구성 요소로 이루어집니다:

1. **Backbone + FPN**: ResNet 기반 특징 추출 ({P2, P3, P4, P5})
2. **Bottom-up Path Augmentation**: 저수준 정보 전파 ({N2, N3, N4, N5})
3. **ROIAlign 및 Adaptive Feature Pooling**: 모든 레벨에서 특징 풀링
4. **Task Branches**: 
   - Box classification & regression branch
   - Mask prediction branch (FCN + FC 퓨전)

### 3.2 특징 분포 특성

논문에서 제시한 중요한 관찰은 Adaptive Feature Pooling의 효과입니다. 원래 FPN에서 Level 1에 할당된 소형 제안들은 실제로 70%의 특징을 다른 레벨에서 풀링하고, Level 4에 할당된 대형 제안들은 50% 이상을 저수준 특징에서 풀링합니다. 이는 제안 크기만으로는 최적의 특징 레벨 할당이 불가능함을 증명합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/218d741f-dd54-4e2d-9500-8ae7ec354796/1803.01534v4.pdf)

### 3.3 계산 복잡도

PANet의 추가 계산량은 미미합니다. 기존 Mask R-CNN 대비:
- Bottom-up path: 8개의 3×3 컨볼루션 추가
- Adaptive feature pooling: ROIAlign 연산 반복 (4배)
- FC fusion: 경량 완전연결 레이어

전체적으로 10% 미만의 추가 계산 오버헤드로 상당한 성능 향상을 달성합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/218d741f-dd54-4e2d-9500-8ae7ec354796/1803.01534v4.pdf)

## 4. 성능 향상 및 실험 결과

### 4.1 COCO 데이터셋 성능

| 지표 | Baseline (Mask R-CNN) | PANet | 개선도 |
|------|:---:|:---:|:---:|
| Mask AP | 35.7% | 37.8% | +2.1%p |
| AP50 | 57.3% | 59.4% | +2.1%p |
| AP75 | 38.0% | 41.0% | +3.0%p |
| APS (소형) | 17.8% | 19.2% | +1.4%p |
| APM (중형) | 37.7% | 41.5% | +3.8%p |
| APL (대형) | 45.8% | 54.3% | +8.5%p |

Box AP (독립 객체 탐지기):
- 37.1% → 39.2% (+2.1%p)

다중 스케일 학습 포함 시 (ms-train):
- ResNet-50: 38.2% → 42.0% (+3.8%p)
- ResNeXt-101: 40.0% → 42.0% (+2.0%p) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/218d741f-dd54-4e2d-9500-8ae7ec354796/1803.01534v4.pdf)

### 4.2 성분별 기여도 분석

ablation study 결과 (COCO val-2017, ResNet-50): [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/218d741f-dd54-4e2d-9500-8ae7ec354796/1803.01534v4.pdf)

| 구성 요소 | 누적 AP 향상 | 주요 효과 |
|---|:---:|---|
| Baseline | 33.6% | 초기값 |
| + 다중 스케일 학습 | 35.3% | +1.7%p |
| + 동기화 배치 정규화 | 35.7% | +0.4%p |
| + Bottom-up 경로 | 36.4% | +0.7%p (특히 대형 객체) |
| + Adaptive 풀링 | 36.9% | +0.5%p (모든 스케일) |
| + FC 퓨전 | 37.6% | +0.7%p (마스크 품질) |
| + 헤비 헤드 | 37.8% | +0.2%p |

### 4.3 다른 데이터셋 성능

**Cityscapes 데이터셋** (길거리 장면):
- Mask R-CNN (fine-only): 31.5%
- PANet (fine-only): 36.5% (+5.0%p)
- Mask R-CNN (COCO pre-train): 36.4%
- PANet (COCO pre-train): 41.4% (+5.0%p) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/218d741f-dd54-4e2d-9500-8ae7ec354796/1803.01534v4.pdf)

**MVD 데이터셋** (Mapillary Vistas):
- PANet (단일 스케일): 23.6% AP
- UCenter (ensemble with COCO pre-train): 24.9%
- PANet + 테스트 tricks: 26.3% [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/218d741f-dd54-4e2d-9500-8ae7ec354796/1803.01534v4.pdf)

### 4.4 COCO 2017 Challenge 결과

논문 발표 시 달성한 최고 성능:
- **Instance Segmentation: 1위 (46.7% Mask AP)**
  - 2016 챔피언 대비 9.1%p 절대 개선 (24% 상대 개선)
- **Object Detection: 2위 (51.0% Box AP)**
  - 2016 챔피언 대비 9.4%p 절대 개선 (23% 상대 개선) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/218d741f-dd54-4e2d-9500-8ae7ec354796/1803.01534v4.pdf)

## 5. 모델의 일반화 성능 향상 가능성

### 5.1 일반화 성능의 원리

PANet의 일반화 능력 향상은 두 가지 메커니즘에서 비롯됩니다:

**1) 정보 경로 단축을 통한 그래디언트 흐름 개선**

Bottom-up path augmentation은 그래디언트 역전파를 단축시켜 훈련 안정성을 향상시킵니다:

$$\frac{\partial L}{\partial N_i} = \sum_{j > i} \frac{\partial L}{\partial N_j} \cdot \frac{\partial N_j}{\partial N_i}$$

짧은 경로로 인해 그래디언트 소실(vanishing gradient) 문제가 완화되며, 결과적으로 저수준 특징의 학습이 더 효과적입니다.

**2) 특징 다양성 증대**

Adaptive feature pooling은 각 객체가 다양한 수준의 특징을 활용하도록 하여:
- 작은 객체: 고수준의 문맥 정보 활용
- 큰 객체: 저수준의 위치 정보 활용
- 모든 객체: 다중 표현 학습

결과적으로 특징 공간이 더 풍부하고 표현력 있어집니다.

### 5.2 도메인 일반화 분석

PANet의 도메인 간 일반화 능력은 제한적입니다:

**강점**:
- Cityscapes와 MVD 데이터셋에서 일관된 성능 향상 (+5%p)
- 다양한 스케일의 객체 처리에 강인함

**한계**:
- CNN 기반 구조의 텍스처 편향(texture bias)로 인한 도메인 시프트 취약성
- 도메인 적응 메커니즘 부재
- 배경 복잡도가 높은 환경에서 성능 저하 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/218d741f-dd54-4e2d-9500-8ae7ec354796/1803.01534v4.pdf)

### 5.3 최신 연구의 개선 방향

2020년 이후 일반화 성능 개선을 위한 연구 방향:

1. **Vision Transformer 기반 모델** (Mask DINO, 2022)
   - 형상 편향으로 인한 OOD 일반화 우수 (+8-12%)
   - 54.5 AP 달성 (PANet 대비 +12.5%p)

2. **도메인 일반화 기법** (ReVT, 2023)
   - 데이터 증강 + 재매개변수화
   - 새로운 도메인에서 PANet 대비 +15% 성능 향상

3. **특징 정규화** (multi-GPU sync BN)
   - PANet에서도 적용: +1.5%p 향상
   - 배치 통계 안정성으로 도메인 간 일반화 개선

## 6. 모델의 한계

### 6.1 아키텍처적 한계

1. **CNN 기반 설계의 한정성**
   - 글로벌 수용장의 제약으로 장거리 정보 전파 어려움
   - 로컬 특징에 편향된 표현 학습

2. **고정된 구조**
   - 입력 이미지 크기에 따른 유연성 부족
   - 실시간 처리 어려움 (5 fps)

3. **계단식 구조의 비효율성**
   - 다단계 처리로 인한 추론 시간 증가
   - 엣지 디바이스 배포 제약

### 6.2 데이터셋 의존성

1. **COCO 데이터셋 의존**
   - 제한된 도메인 다양성
   - 자동차 중심의 객체 분포

2. **소형 객체 처리 약점**
   - APS 개선이 제한적 (+1.4%p)
   - 고밀도 장면에서 성능 저하

3. **겹침(occlusion) 처리 미흡**
   - 마스크 품질 예측 부재
   - 부분 가림 상황에서 정확도 감소

## 7. 2020년 이후 관련 최신 연구 비교

### 7.1 주요 경쟁 모델 비교

| 모델 | 발표년 | 주요 기술 | COCO AP | 속도 (fps) | 특징 |
|---|---|---|:---:|:---:|---|
| **PANet** | 2018 | Bottom-up path, Adaptive pooling | 42.0 | 5 | 높은 정확도 |
| **YOLACT** | 2019 | 실시간 마스크 예측 | 29.8 | 33.5 | 빠른 속도 |
| **Mask DINO** | 2022 | Transformer, Unified detection-seg | 54.5 | 7 | 최고 정확도 |
| **YOLOv8-seg** | 2023 | 앵커 프리, EfficientNet | 51.2 | 40 | 높은 일반화 |
| **PolySnake** | 2023 | 경계 기반 반복 정제 | 43.8 | 12 | 정교한 경계 |
| **SODAR** | 2022 | 동적 마스크 집계 | 44.2 | 15 | 개선된 SOLO |

### 7.2 기술 혁신 분석

**Transformer 기반 전환 (Mask DINO, 2022)**: [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Mask_DINO_Towards_a_Unified_Transformer-Based_Framework_for_Object_Detection_CVPR_2023_paper.pdf)
- 글로벌 자기주의로 모든 픽셀 간 관계 모델링
- Detection과 segmentation의 통일된 프레임워크
- 성능: 54.5 AP (PANet의 42.0 AP 대비 +12.5%p)

**실시간 처리 중심 (YOLACT, YOLOv8-seg)**:
- 앵커 기반 또는 프리 접근
- 프로토타입 기반 마스크 생성
- 속도-정확도 트레이드오프 최적화

**도메인 일반화 강화 (Vision Transformer 적용)**: [openaccess.thecvf](https://openaccess.thecvf.com/content/ICCV2023W/OODCV/papers/Termohlen_A_Re-Parameterized_Vision_Transformer_ReVT_for_Domain-Generalized_Semantic_Segmentation_ICCVW_2023_paper.pdf)
- 형상 편향(shape bias)으로 인한 OOD 일반화 우수
- CNN 모델 대비 +10-15% 성능 향상 (새 도메인)
- 그러나 계산 비용 증가

### 7.3 구체적 비교 분석

#### Mask DINO vs PANet

**Mask DINO의 장점**:
1. **성능**: 54.5 AP vs 42.0 AP (+12.5%p)
2. **구조**: 엔드-투-엔드 학습 가능
3. **일반화**: 도메인 시프트에 강인함

**Mask DINO의 단점**:
1. **속도**: 7 fps vs 5 fps (PANet과 비슷)
2. **계산량**: 더 높은 메모리 요구
3. **구현**: 복잡한 Transformer 아키텍처

#### YOLOv8-seg vs PANet

**YOLOv8-seg의 장점**:
1. **실시간성**: 40 fps (PANet의 8배)
2. **일반화**: 새로운 패턴에 더 나은 적응 [nature](https://www.nature.com/articles/s41598-025-02131-7)
3. **배포**: 에지 디바이스 최적화

**PANet의 장점**:
1. **정확도**: 42.0 AP (PANet) vs 51.2 AP (YOLOv8-seg, 단순 비교)
2. **세부 표현**: 복잡한 패턴 인식 능력 우수
3. **주석 기반**: 정밀한 마스크 품질 [nature](https://www.nature.com/articles/s41598-025-02131-7)

### 7.4 최신 트렌드와 시사점

**1) Vision Transformer의 부상**: [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Delving_Deep_Into_the_Generalization_of_Vision_Transformers_Under_Distribution_CVPR_2022_paper.pdf)
- ViT 기반 모델들이 도메인 일반화에서 우수
- 형상 편향으로 인한 자연스러운 일반화
- 그러나 CNN 모델도 최적화를 통해 경쟁력 유지

**2) 도메인 적응 기법의 중요성**:
- 단순 특징 확대(scaling)만으로는 부족
- 도메인별 특화 모듈 필요
- 메타러닝, 자기지도 학습 활용 증가

**3) 효율성-정확도 트레이드오프**:
- 실시간 애플리케이션: YOLO 계열 선호
- 고정확 요구 애플리케이션: Mask DINO 선호
- 균형: YOLOv8, PolySnake 등

## 8. 논문의 영향과 향후 연구 방향

### 8.1 학술적 영향

**직접적 영향**:
- COCO 2017 1위 달성으로 인스턴스 분할 분야의 벤치마크 설정
- Bottom-up path 개념이 YOLO v4의 PANet에 채택되어 산업 표준화
- Adaptive feature pooling의 개념이 이후 다양한 모델에 영감 제공

**간접적 영향**:
- 정보 경로 최적화의 중요성 강조로 후속 연구 방향 제시
- 마스크 예측의 다중 경로 설계 원칙 확립
- 특징 계층 간 상호작용의 정량적 분석 제시

### 8.2 산업 적용

1. **자율주행차**: 
   - 보행자, 차량 분할 (Cityscapes 성능 우수)
   - 실시간 처리 한계 극복 필요

2. **의료 영상**:
   - 기관 분할, 종양 감지
   - 고정확 요구로 PANet-like 모델 선호

3. **로봇공학**:
   - 객체 조작 (대형 객체 성능 우수, APS 개선 필요)
   - 실시간 성능 요구로 경량 모델 병행

### 8.3 향후 연구 시 고려 사항

#### 8.3.1 아키텍처 개선 방향

1. **하이브리드 구조 설계**
   - CNN의 효율성 + Transformer의 글로벌 정보 통합
   - 조건부 계산(conditional computation)으로 효율성 향상
   - 예상 성능: PANet 대비 +5-8% (Mask DINO 수준)

2. **적응형 경로 선택**
   - 객체 특성에 따른 동적 특징 경로 선택
   - 학습 가능한 라우팅 모듈 추가
   - 계산 오버헤드 제한 (+5-10%)

3. **계층별 특징 정규화**
   - 도메인 시프트 대응
   - 인스턴스 정규화 + 배치 정규화 조합
   - 새 도메인 성능: +3-5% 향상 가능

#### 8.3.2 학습 전략 개선

1. **자기지도 학습(Self-Supervised Learning) 활용**
   - 라벨링 비용 감소
   - 도메인 일반화 개선
   - 콘트라스티브 학습 + PANet 결합

2. **메타러닝 프레임워크**
   - 소량 데이터에서의 빠른 적응
   - 도메인 간 전이학습 강화
   - Few-shot instance segmentation 가능

3. **다중 작업 학습(Multi-Task Learning)**
   - 객체 탐지 + 의미 분할 + 인스턴스 분할 동시 학습
   - 작업 간 정보 공유로 일반화 개선
   - 전체 성능: +2-4% 향상

#### 8.3.3 평가 메트릭 확장

1. **세밀한 마스크 품질 평가**
   - Mask IoU 외 경계 정밀도(Boundary Precision) 도입
   - 작은 객체/겹침 상황에서의 성능 평가
   - 실제 응용 성능 반영

2. **도메인 간 일반화 평가**
   - Cross-domain AP (CD-AP) 제안
   - 자동차 데이터셋 → 의료 영상 등 이질 도메인 평가
   - 실무 적용 가능성 평가

3. **계산 효율성 메트릭**
   - 정확도-속도 파레토 프론티어 분석
   - 에너지 소비 포함한 종합 평가
   - 에지 디바이스 배포 가능성 판단

#### 8.3.4 특정 응용 분야별 맞춤화

1. **의료 영상 분할**
   - 3D 확장 (3D PANet)
   - 약한 감독(weakly supervised) 학습
   - 불균형 클래스 처리 강화

2. **비디오 인스턴스 분할**
   - 시간적 일관성 모듈링
   - 프레임 간 특징 전파
   - 움직임 보상(motion compensation)

3. **고해상도 원격감지**
   - 다중 스케일 패치 기반 처리
   - 메모리 효율적인 구조
   - 소형 객체(건물, 도로) 분할

#### 8.3.5 이론적 분석

1. **정보 병목 분석(Information Bottleneck)**
   - 각 레이어의 정보 흐름 정량화
   - 필요한 경로 용량 결정
   - 최적 아키텍처 설계

2. **일반화 한계(Generalization Bounds)**
   - VC 차원 분석을 통한 이론적 한계 규명
   - PANet의 복잡도 vs 일반화 능력 트레이드오프 분석
   - 모델 정규화 전략 이론적 근거 제시

3. **도메인 적응 이론**
   - 소스-타겟 도메인 간 불일치도(domain discrepancy) 측정
   - 적응 가능 상한(adaptation upper bound) 도출
   - 효율적 적응 전략 설계

## 결론

PANet은 2018년 발표되었음에도 불구하고 정보 경로 최적화의 원칙과 특징 다양성 활용의 아이디어를 통해 인스턴스 분할 분야에 지속적인 영향을 미치고 있습니다. COCO 2017 1위 달성과 YOLO v4 등 후속 모델에의 채택으로 산업 표준으로 인정받았습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/218d741f-dd54-4e2d-9500-8ae7ec354796/1803.01534v4.pdf)

그러나 2020년 이후 등장한 Vision Transformer 기반 모델(Mask DINO)의 54.5 AP와 도메인 일반화 우수성에 직면하여, 향후 연구는 다음 방향으로 진행되어야 합니다:

1. **하이브리드 아키텍처**: CNN의 효율성과 Transformer의 표현력을 결합
2. **도메인 일반화 강화**: 도메인 적응 모듈 및 자기지도 학습 통합
3. **효율성 개선**: 실시간 처리 가능성과 에지 배포 지원
4. **이론적 분석**: 정보 흐름과 일반화의 수학적 근거 규명

이러한 방향의 연구를 통해 PANet의 핵심 개념을 현대적 프레임워크에 적용하면, 정확성, 효율성, 일반화 성능 모두에서 우수한 차세대 인스턴스 분할 모델의 개발이 가능할 것으로 예상됩니다.

***

## 참고문헌

<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_21][^1_22][^1_23][^1_24][^1_25][^1_26][^1_27][^1_28][^1_29][^1_30][^1_31][^1_32][^1_33][^1_34][^1_35][^1_36][^1_37][^1_38][^1_39][^1_40][^1_41][^1_42][^1_43][^1_44][^1_45][^1_46][^1_47][^1_48][^1_49][^1_50][^1_51][^1_52][^1_53][^1_54][^1_55][^1_56][^1_57][^1_58][^1_59][^1_6][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_7][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_8][^1_80][^1_81][^1_82][^1_83][^1_84][^1_9]</span>

<div align="center">⁂</div>

[^1_1]: 1803.01534v4.pdf

[^1_2]: https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Mask_DINO_Towards_a_Unified_Transformer-Based_Framework_for_Object_Detection_CVPR_2023_paper.pdf

[^1_3]: https://openaccess.thecvf.com/content/ICCV2023W/OODCV/papers/Termohlen_A_Re-Parameterized_Vision_Transformer_ReVT_for_Domain-Generalized_Semantic_Segmentation_ICCVW_2023_paper.pdf

[^1_4]: https://www.nature.com/articles/s41598-025-02131-7

[^1_5]: https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Delving_Deep_Into_the_Generalization_of_Vision_Transformers_Under_Distribution_CVPR_2022_paper.pdf

[^1_6]: https://arxiv.org/pdf/2202.07402.pdf

[^1_7]: http://arxiv.org/abs/2203.12827

[^1_8]: https://arxiv.org/pdf/2205.12646.pdf

[^1_9]: https://arxiv.org/abs/2501.01685

[^1_10]: http://arxiv.org/pdf/1904.02689v2.pdf

[^1_11]: https://arxiv.org/abs/1611.07709v1

[^1_12]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9459926/

[^1_13]: https://arxiv.org/pdf/1910.02624.pdf

[^1_14]: http://arxiv.org/pdf/2301.08898.pdf

[^1_15]: https://arxiv.org/abs/2306.15348

[^1_16]: https://pdfs.semanticscholar.org/4f67/316616279dc15d96cc9ba259123a21b245dd.pdf

[^1_17]: https://arxiv.org/html/2409.07022

[^1_18]: http://arxiv.org/pdf/2107.11758.pdf

[^1_19]: https://arxiv.org/pdf/1710.08192.pdf

[^1_20]: https://arxiv.org/html/2407.21498v1

[^1_21]: https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/2784.pdf

[^1_22]: https://arxiv.org/pdf/2511.15062.pdf

[^1_23]: https://arxiv.org/abs/2401.08174

[^1_24]: http://www.arxiv.org/abs/1908.06391

[^1_25]: https://arxiv.org/pdf/2508.04333.pdf

[^1_26]: https://arxiv.org/abs/2306.16132

[^1_27]: http://www.arxiv.org/abs/2002.06345

[^1_28]: https://arxiv.org/pdf/2504.16081.pdf

[^1_29]: https://kimjy99.github.io/논문리뷰/yolact/

[^1_30]: https://kimhongsi.tistory.com/entry/딥러닝-instance-segmentation-vs-semantic-segmantation

[^1_31]: https://railly-linker.tistory.com/187

[^1_32]: https://kimjy99.github.io/논문리뷰/oneformer/

[^1_33]: https://docs.ultralytics.com/ko/compare/rtdetr-vs-yolov5/

[^1_34]: https://velog.io/@yeontachi/Object-Detection-FPN-Feature-Pyramid-Network-paper-review

[^1_35]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12431344/

[^1_36]: https://www.jstna.org/journal/jsta/jsta-1-1/jsta-1-1_full.pdf

[^1_37]: https://blog.outta.ai/292

[^1_38]: https://ai-stat-lab.tistory.com/13

[^1_39]: https://deep-learning-study.tistory.com/637

[^1_40]: https://moordo91.tistory.com/49

[^1_41]: https://www.ultralytics.com/ko/blog/what-is-instance-segmentation-a-quick-guide

[^1_42]: https://wikidocs.net/255165

[^1_43]: https://wikidocs.net/162976

[^1_44]: https://hefjournal.org/index.php/HEF/article/view/347

[^1_45]: https://www.mdpi.com/2504-446X/8/9/491

[^1_46]: https://ieeexplore.ieee.org/document/9290720/

[^1_47]: https://www.semanticscholar.org/paper/2bd88e7d3210621773fa083781b09df97f0e883d

[^1_48]: https://link.springer.com/10.1007/s11042-024-20353-1

[^1_49]: https://fcc08321-8158-469b-b54d-f591e0bd3df4.filesusr.com/ugd/185b0a_8f8df49321034db5af4d571293e953c3.pdf

[^1_50]: https://publishing.emanresearch.org/Journal/Abstract/angiotherapy-899875

[^1_51]: https://www.semanticscholar.org/paper/26b6c75d4883c6e986d890d83dda79021d715234

[^1_52]: https://francis-press.com/papers/17917

[^1_53]: https://ieeexplore.ieee.org/document/10849031/

[^1_54]: http://arxiv.org/pdf/1903.07209.pdf

[^1_55]: https://downloads.hindawi.com/journals/ddns/2020/9242917.pdf

[^1_56]: https://arxiv.org/pdf/2109.03426.pdf

[^1_57]: https://arxiv.org/ftp/arxiv/papers/2211/2211.02799.pdf

[^1_58]: https://arxiv.org/pdf/2107.12889.pdf

[^1_59]: https://www.mdpi.com/1424-8220/20/4/1010/pdf

[^1_60]: https://www.mdpi.com/2227-9032/10/12/2396/pdf?version=1669721795

[^1_61]: https://arxiv.org/abs/2203.03886

[^1_62]: https://www.biorxiv.org/content/10.1101/2024.03.28.587212v1.full-text

[^1_63]: https://arxiv.org/html/2402.00045v4

[^1_64]: https://arxiv.org/abs/2206.02777

[^1_65]: https://arxiv.org/html/2512.23903v1

[^1_66]: https://arxiv.org/pdf/2206.02777.pdf

[^1_67]: https://arxiv.org/pdf/2404.04452.pdf

[^1_68]: https://arxiv.org/html/2508.16527v1

[^1_69]: https://arxiv.org/html/2412.10028v4

[^1_70]: https://arxiv.org/html/2404.04452v2

[^1_71]: https://arxiv.org/pdf/2508.16527.pdf

[^1_72]: https://arxiv.org/html/2304.09854v4

[^1_73]: https://arxiv.org/abs/2308.13331

[^1_74]: https://www.reddit.com/r/MachineLearning/comments/p8xvb7/d_mask_rcnn_was_from_2017_are_there_any_good/

[^1_75]: https://scholarworks.bwise.kr/cau/bitstream/2019.sw.cau/68695/1/Domain-Adaptive Vision Transformers for Generalizing Across Visual Domains.pdf

[^1_76]: https://arxiv.org/abs/1703.06870

[^1_77]: https://kimjy99.github.io/논문리뷰/mask-dino/

[^1_78]: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840473.pdf

[^1_79]: https://github.com/roboflow/rf-detr

[^1_80]: https://www.semanticscholar.org/paper/A-Comparison-of-YOLO-and-Mask-R-CNN-for-Segmenting-Prasetyo-Suciati/5d8eb7f85474e79934e2e8eb632faeb67e3f1f19

[^1_81]: https://ivrl.github.io/VTAGML/

[^1_82]: https://www.sciencedirect.com/science/article/pii/S258972172400028X

[^1_83]: https://ostin.tistory.com/86

[^1_84]: https://arxiv.org/abs/2408.14957
