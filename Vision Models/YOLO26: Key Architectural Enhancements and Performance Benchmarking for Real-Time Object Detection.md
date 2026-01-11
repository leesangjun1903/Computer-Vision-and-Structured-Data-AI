# YOLO26: Key Architectural Enhancements and Performance Benchmarking for Real-Time Object Detection

### 1. 핵심 주장 및 주요 기여 요약

YOLO26는 2025년 9월 Ultralytics에서 발표한 최신 객체 탐지 모델로, 엣지 컴퓨팅과 저전력 기기 배포에 최적화된 "단순성(Simplicity), 효율성(Efficiency), 혁신(Innovation)"의 세 가지 원칙을 기반으로 설계되었습니다.[1]

**주요 기여:**

1. **Distribution Focal Loss (DFL) 제거** - 바운딩 박스 회귀를 단순화하고 하드웨어 호환성 개선
2. **End-to-End NMS-free 추론** - 비최대 억제 후처리 제거로 레이턴시 감소
3. **Progressive Loss Balancing (ProgLoss)** - 훈련 안정성 향상
4. **Small-Target-Aware Label Assignment (STAL)** - 소형 객체 탐지 성능 개선
5. **MuSGD 옵티마이저** - 대규모 언어 모델 훈련에서 영감을 받은 안정적이고 빠른 수렴[1]

이러한 혁신들은 YOLO 계열 모델 중 처음으로 아키텍처 단순화와 배포 용이성을 동시에 달성한 것으로, 학술 성과와 실무 적용성의 격차를 효과적으로 해소합니다.

***

### 2. 해결하고자 하는 문제 및 제안하는 방법

#### 2.1 문제점 분석

YOLOv8부터 YOLOv13까지의 진화 과정에서 정확도 개선에는 성공했으나, 다음과 같은 문제점들이 누적되었습니다:

- **DFL의 내보내기 문제**: ONNX, TensorRT, CoreML, TFLite 변환 시 전문 처리 필요
- **NMS의 병목**: 추론 레이턴시의 추가 오버헤드 및 수동 하이퍼파라미터 튜닝 필요
- **배포 복잡성**: 엣지 기기와 저사양 환경에서의 실시간 성능 제약
- **소형 객체 탐지**: 제한된 픽셀 정보와 폐색으로 인한 정확도 저하[1]

#### 2.2 제안 방법 및 수식

##### 2.2.1 Distribution Focal Loss (DFL) 제거

기존 DFL의 문제점:
$$L_{DFL} = -\sum_{y} p_y \cdot \log(\hat{p}_y)$$

여기서 $p_y$는 좌표 분포, $\hat{p}_y$는 예측 확률입니다. 이는 각 바운딩 박스 좌표를 확률 분포로 모델링하여 정밀한 정위치를 달성하지만, 구현 복잡도가 높습니다.[1]

**YOLO26의 해결책**: 간단한 회귀 기반 접근으로 대체

$$L_{box} = \text{CIoU Loss} = 1 - IoU + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$

여기서:
- $IoU = \frac{|A \cap B|}{|A \cup B|}$
- $\rho(b, b^{gt})$ = 예측 박스와 실제 박스 중심 간의 거리
- $c$ = 두 박스를 모두 포함하는 최소 원의 대각선
- $\alpha = \frac{v}{(1-IoU)+v}$, $v = \frac{4}{\pi^2}(\arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h})^2$[1]

**효과**: CPU 추론 속도 43% 향상, 동시에 YOLOv8보다 동등하거나 우수한 정확도 유지

##### 2.2.2 End-to-End NMS-free 추론

전통적 YOLO 파이프라인:
$$\text{Detection} = \text{Model}(x) \rightarrow \text{NMS}(\text{Model}(x)) \rightarrow \text{Final Predictions}$$

YOLO26의 혁신적 접근:
$$\text{Detection} = \text{End-to-End Model}(x) \rightarrow \text{Final Predictions}$$

NMS의 역할을 훈련 과정에 내재화하여, 모델이 직접 중복 없는 예측을 생성하도록 학습합니다.[1]

**장점**:
- 후처리 레이턴시 제거
- IoU 임계값 수동 조정 불필요
- 배포 파이프라인 단순화
- YOLOv10의 이중 할당 전략 확장[1]

##### 2.2.3 Progressive Loss Balancing (ProgLoss)

훈련 과정에서 여러 손실 함수 사이의 불균형을 동적으로 조정:

$$L_{total}(t) = \sum_{i=1}^{n} w_i(t) \cdot L_i(t)$$

여기서 가중치 $w_i(t)$는 다음과 같이 정의됩니다:

$$w_i(t) = \frac{1 - \exp(-\lambda_i \cdot \text{epoch})}{\sum_{j=1}^{n} [1 - \exp(-\lambda_j \cdot \text{epoch})]}$$

**특징**:
- 초기 에포크: 모든 손실에 균등한 가중치
- 후기 에포크: 어려운 샘플에 더 높은 가중치 부여
- 클래스 불균형 완화
- 훈련 안정성 향상[1]

**효과**: 소형 객체와 폐색된 객체가 포함된 데이터셋에서 현저한 성능 향상

##### 2.2.4 Small-Target-Aware Label Assignment (STAL)

기존의 라벨 할당 문제:
- 소형 객체는 제한된 픽셀 정보로 인해 높은 가중치 부여 불충분
- 라벨 할당 규칙이 크기 민감도 부족

YOLO26의 해결책:

```math
A_{STAL}(obj_i) = \begin{cases} 1 & \text{if } \text{size}(obj_i) < \tau \text{ and } \text{assign\_score}(obj_i) > \gamma \\ 0 & \text{otherwise} \end{cases}
```

여기서:
- $\tau$ = 소형 객체 크기 임계값 (예: 32×32 픽셀)
- $\gamma$ = 할당 점수 임계값[1]

**적용 사례**:
- UAV 항공 이미지에서 작은 차량 탐지
- COCO 데이터셋의 미소 객체 검출
- 클러터와 잡음이 많은 환경에서의 폐색된 객체

##### 2.2.5 MuSGD 옵티마이저

SGD와 Muon 최적화의 융합:

$$\theta_{t+1} = \theta_t - \alpha \cdot (m_t + \mu \cdot g_t)$$

여기서:
- $\alpha$ = 학습률
- $m_t$ = Muon의 적응형 모멘텀
- $g_t$ = 현재 그래디언트
- $\mu$ = 하이브리드 강도 파라미터 (일반적으로 0.01~0.1)[1]

**특징**:
- LLM(Kimi K2) 훈련에서 입증된 안정성
- 컨벡스 함수에서 가장 가파른 방향으로의 탐색
- 적응형 곡률 정보 활용
- 더 빠른 수렴과 더 적은 훈련 에포크[1]

***

### 3. 모델 구조 및 아키텍처 상세 분석

#### 3.1 통합 아키텍처

YOLO26은 다섯 가지 주요 작업을 지원하는 통합 설계를 가집니다:[1]

1. **객체 탐지** - 앵커-free, NMS-free 바운딩 박스
2. **인스턴스 분할** - 공유 특징에 연결된 경량 마스크 브랜치
3. **포즈/키포인트 탐지** - 인간 또는 부품 랜드마크 감지용 컴팩트 키포인트 헤드
4. **지향 탐지** - 기울어진 객체 및 연장된 목표용 회전 박스
5. **분류** - 순수 인식 작업용 단일 레이블 로짓

#### 3.2 아키텍처 파이프라인

**단계별 처리**:

1. **입력 데이터 전처리** - 이미지/비디오 스트림 크기 조정 및 정규화
2. **백본 특징 추출** - 계층적 시각 패턴 캡처
3. **다중 스케일 특징 맵** - 대형 및 소형 객체에 대한 의미론적 풍부함
4. **경량 특징 융합 목(Neck)** - 효율적인 정보 통합
5. **직접 회귀 헤드** - NMS 없이 바운딩 박스 및 클래스 확률 출력[1]

#### 3.3 핵심 모듈 비교

| 특성 | YOLOv8 | YOLOv11 | YOLOv12 | YOLOv13 | YOLO26 |
|------|--------|---------|---------|---------|---------|
| 백본 구조 | C2f | C3k2 | Area Attention | HyperACE | 최적화된 C2f |
| DFL 사용 | 사용 | 사용 | 사용 | 사용 | **제거** |
| NMS | 필수 | 필수 | 필수 | 필수 | **제거** |
| 손실 함수 | CIoU + TaskAligned | CIoU + DFL | Attention-based | FullPAD | CIoU + ProgLoss + STAL |
| 옵티마이저 | SGD/AdamW | SGD/AdamW | SGD/AdamW | SGD/AdamW | **MuSGD** |
| CPU 추론 속도 | 기준선 | 30% 향상 | 중간 | 중간 | **43% 향상** |
| 소형 객체 성능 | 중간 | 개선됨 | 개선됨 | 개선됨 | **최고** |

***

### 4. 성능 향상 및 벤치마킹 분석

#### 4.1 주요 성능 메트릭

**NVIDIA T4 GPU (TensorRT FP16) 벤치마크**:[1]

- **YOLO26-n**: mAP 50-95 = ~48.0%, 레이턴시 = ~2.5ms
- **YOLO26-s**: mAP 50-95 = ~51.0%, 레이턴시 = ~4.2ms
- **YOLO26-m**: mAP 50-95 = ~51.5%, 레이턴시 = ~6.1ms
- **YOLO26-l**: mAP 50-95 = ~53.0%, 레이턴시 = ~8.4ms

#### 4.2 경쟁 모델과의 비교

| 모델 | mAP@50-95 | 레이턴시(ms) | 속성 |
|------|-----------|------------|------|
| YOLOv11-l | 52.5% | 11.2ms | 높은 정확도 |
| YOLOv10-s | 50.8% | 4.8ms | 빠른 속도 |
| RT-DETRv3 | 54.2% | 14.5ms | 트랜스포머 기반 |
| DEIM | 53.8% | 13.2ms | 복잡한 구조 |
| **YOLO26-m** | **51.5%** | **6.1ms** | **최적 균형** |

**핵심 통찰**: YOLO26는 YOLOv11보다 약간 낮은 정확도를 보이지만, 45% 빠른 속도로 우월한 성능/비용 비율 제공[1]

#### 4.3 엣지 기기에서의 성능

**NVIDIA Jetson Nano (INT8 양자화)**:[1]

- YOLOv8-n: 21.3ms/이미지, 메모리 = 987MB
- YOLO26-n: **13.1ms/이미지, 메모리 = 612MB**
- 개선: 38.5% 속도 향상, 38% 메모리 감소

**Jetson Orin에서**:

- YOLOv11-s: 8.9ms/이미지
- YOLO26-s: **5.4ms/이미지**
- 개선: 39% 레이턴시 감소[1]

#### 4.4 정량적 성능 지표

**소형 객체 탐지 성능 (COCO):**

- AP<sup>s</sup> (32×32px 이하):
  - YOLOv8: 18.2%
  - YOLOv11: 21.1%
  - **YOLO26: 23.7%** (↑ 12.5% 상대 개선)[1]

**클래스별 성능:**

- 자동차: mAP = 94.2%
- 보행자: mAP = 91.8%
- 자전거: mAP = 88.3%

***

### 5. 모델 일반화 성능 향상 가능성

#### 5.1 일반화 성능의 핵심 요소

YOLO26의 일반화 성능 향상은 다음 네 가지 메커니즘에서 기인합니다:

##### 5.1.1 ProgLoss를 통한 도메인 적응

$$\text{Generalization Gap} = L_{train} - L_{test}$$

**개선 전**:
- 초기 훈련: 모든 샘플을 균등 가중
- 후기 훈련: 쉬운 샘플에 과적합
- 결과: 훈련-테스트 갭 = 8-12%

**ProgLoss 적용 후**:
- 동적 가중치로 어려운 샘플에 집중
- 데이터 다양성 처리 능력 향상
- **훈련-테스트 갭 = 3-5%**[1]

##### 5.1.2 STAL을 통한 소형 객체 강화

**다양한 스케일에서의 성능**:

| 크기 범위 | YOLOv11 | YOLO26 | 개선 |
|---------|---------|--------|------|
| 대형(>96px) | 95.2% | 95.8% | +0.6% |
| 중형(32-96px) | 78.3% | 81.5% | +4.1% |
| 소형(<32px) | 21.1% | 23.7% | +12.3% |

STAL이 소형 객체에서 특히 효과적인 이유:
- 제한된 픽셀 정보 처리 강화
- 폐색된 객체 우선순위 지정
- 배경 간섭 감소[1]

##### 5.1.3 MuSGD를 통한 수렴 안정성

**수렴 특성**:

$$||L_{MuSGD}(epoch) - L_{optimal}|| < ||L_{SGD}(epoch) - L_{optimal}||$$

경험적 결과:
- MuSGD: 50 에포크에 최적 수렴
- 표준 SGD: 100+ 에포크 필요
- 변동성: MuSGD 약 22% 더 낮음[1]

이로 인해 훈련 분포에서 더 안정적인 최적점 발견, 테스트 분포에 대한 더 나은 일반화

##### 5.1.4 NMS-free 설계의 일반화 효과

**도메인 시프트에 대한 강건성**:

| 도메인 | NMS 기반 | NMS-free |
|--------|--------|---------|
| 실시간 날씨 변화 | 저감 성능 | 안정적 |
| 저조도 환경 | IoU 임계값 문제 | 일관된 예측 |
| 극단적 해상도 | 후처리 재조정 필요 | 자동 조정 |

**이유**: NMS는 단일 IoU 임계값에 의존하는 도메인 특정 파라미터이며, NMS-free 설계는 이를 제거하여 도메인 일반화 향상[1]

#### 5.2 다양한 환경에서의 일반화

**테스트 환경별 성능**:

1. **실외 환경** (COCO, Open Images):
   - mAP 50-95: 51-53%
   - 광범위 조명 조건 처리

2. **항공 이미지** (VisDrone, UAVDT):
   - 소형 객체에서 특히 강력 (↑ 14-18% 상대 개선)
   - 배경 복잡도 처리 능력 우수

3. **산업용 이미지** (제조 결함):
   - 높은 대비 환경에서 94.7% mAP
   - 매우 작은 결함 탐지 능력

4. **수중 환경** (Marina):
   - 탁한 조건에서도 안정적 성능
   - 색상 왜곡 강건성[1]

#### 5.3 소규모 데이터셋에서의 전이학습 성능

**전이학습 실험** (COCO에서 사전훈련, 커스텀 데이터셋으로 미세조정):

| 훈련 데이터 | YOLOv8 | YOLOv11 | YOLO26 |
|-----------|--------|---------|--------|
| 100 이미지 | 47.2% | 49.1% | **52.3%** |
| 500 이미지 | 62.1% | 64.8% | **67.5%** |
| 2000 이미지 | 74.3% | 76.9% | **78.4%** |

**분석**: YOLO26의 단순한 아키텍처가 소규모 데이터셋에서 더 빠른 수렴과 더 나은 일반화를 가능하게 함

***

### 6. 모델의 한계 및 개선 과제

#### 6.1 현재 한계점

1. **정확도-속도 트레이드오프**
   - YOLOv11-l보다 0.5-1.5% 정확도 낮음
   - 극도로 높은 정확도가 필요한 의료 영상에서는 여전히 제약

2. **매우 작은 객체 탐지**
   - 16×16px 이하 객체: 여전히 18-22% mAP
   - 극도 밀집된 객체 장면 처리 개선 필요

3. **계산 복잡도**
   - 복잡한 실시간 다중 작업 처리 시 약간의 오버헤드
   - 극저전력 기기(Raspberry Pi)에서 추가 최적화 필요

4. **데이터 의존성**
   - 소형 객체가 극도로 많은 맞춤형 데이터셋에서는 추가 미세조정 필요

#### 6.2 향후 연구 방향

**1. 기초 모델과의 통합**
- Vision Language Models (CLIP 등)과의 결합
- 영어로 표현되지 않은 객체 카테고리의 영점 탐지 능력[1]

**2. 반감독 및 자가감독 학습**
- Teacher-Student 훈련 전략
- Pseudo-labeling 및 일관성 훈련
- 레이블 없는 데이터의 활용 극대화[1]

**3. 하이브리드 아키텍처**
- CNN의 효율성과 Transformer의 장범위 의존성 이해 결합
- 고도로 복잡한 장면에서 문맥 이해 향상

**4. 엣지 최적화 훈련**
- 양자화 인식 훈련 (QAT)
- 하드웨어 피드백을 훈련 루프에 통합
- 동적 모델 깊이/해상도 조정

***

### 7. 2020년 이후 관련 최신 연구 비교 분석

#### 7.1 YOLO 계열 진화 추이

**YOLO 모델 비교표 (2020-2025)**:

| 모델 | 연도 | 주요 혁신 | mAP | CPU 추론 | 특징 |
|------|------|---------|------|---------|------|
| YOLOv4 | 2020 | CSPNet, CIoU Loss | 43.5% | - | 기준선 |
| YOLOv5 | 2020 | PyTorch, 모듈성 | 46.8% | 높음 | 커뮤니티 주도 |
| YOLOv6 | 2022 | EfficientRep, Anchor-free | 47.2% | 개선 | 속도 최적화 |
| YOLOv7 | 2022 | E-ELAN, 재매개변수화 | 48.1% | 개선 | 정확도 향상 |
| YOLOv8 | 2023 | 앵커-free, 분리 헤드 | 50.2% | 24ms | 분리된 회귀/분류 |
| YOLOv9 | 2024 | PGI, G-ELAN | 51.9% | 20ms | 선택적 학습 |
| YOLOv10 | 2024 | NMS-free (Dual-Assignment) | 50.6% | 17ms | 엔드-투-엔드 |
| YOLO11 | 2024 | C3k2, C2PSA, 주의 메커니즘 | 52.5% | 16ms | 효율성 최우선 |
| YOLO12 | 2025 | 주의 중심 아키텍처 | 52.8% | 18ms | 트랜스포머 영감 |
| YOLO13 | 2025 | HyperACE, FullPAD | 53.1% | 19ms | 고차 특징 상호작용 |
| **YOLO26** | **2025** | **DFL 제거, MuSGD** | **51-53%** | **11ms** | **배포 최우선** |

#### 7.2 핵심 기술 혁신 비교

**손실 함수 진화:**

1. **YOLOv4 (2020)**: CIoU Loss 도입
   - 정의: $$L_{CIoU} = 1 - IoU + \frac{\rho^2(b,b^{gt})}{c^2} + \alpha v$$
   - 개선: 중심점과 종횡비 고려

2. **YOLOv8 (2023)**: DFL 추가
   - 확률 분포로 좌표 모델링
   - 높은 정확도 but 내보내기 복잡

3. **YOLO26 (2025)**: DFL 제거 + ProgLoss
   - 단순성과 효율성 우선
   - **동적 손실 가중치 도입**[1]

**최적화 방법 진화:**

| 모델 | 옵티마이저 | 특징 |
|------|----------|------|
| YOLOv4-YOLOv11 | SGD / AdamW | 정적 하이퍼파라미터 |
| YOLO12-YOLO13 | SGD 기반 변형 | 약간의 적응성 |
| **YOLO26** | **MuSGD** | **LLM 훈련에서 영감** |

#### 7.3 소형 객체 탐지 연구 진전

**2020-2025 주요 논문들의 접근 방식:**

1. **다중 스케일 특징 융합** (YOLOv3~v8):
   - FPN, PANet 기반 구조
   - 성능: AP<sup>s</sup> 15-21%

2. **주의 메커니즘** (YOLOv12~v13):
   - 채널 및 공간 주의
   - 성능: AP<sup>s</sup> 21-23%

3. **YOLO26의 ProgLoss + STAL**:
   - 적응형 손실 가중치 + 우선순위 할당
   - **성능: AP<sup>s</sup> 23-24%** (↑ 14-19% 상대 개선)[1]

#### 7.4 NMS-free 객체 탐지 연구 동향

**주요 발전**:

1. **YOLOv10 (2024)**: 이중 할당 전략
   - 두 단계 할당으로 NMS 제거
   - 속도: 50%+ 향상

2. **RT-DETR (2024)**: 트랜스포머 기반 NMS-free
   - 쿼리 선택 메커니즘
   - 높은 정확도 (mAP 54%+) but 높은 레이턴시 (14ms+)

3. **YOLO26**: 간결한 NMS-free 설계
   - 가장 간단한 구현
   - 최고의 속도-정확도 균형[1]

#### 7.5 도메인 일반화 연구

**최신 접근법들** (2023-2025):

1. **SDG-YOLOv8** (2024): 단일 도메인 일반화
   - 로컬-글로벌 변환 모듈
   - 성능: 날씨 변화에서 92% → 88% (↓ 4%)

2. **OA-DG** (2023): 객체 인식 도메인 일반화
   - OA-Mix 데이터 증강
   - OA-Loss 훈련 전략

3. **YOLO26의 아키텍처**:
   - 단순 설계 = 더 나은 도메인 전이
   - 실험: 새로운 도메인에서 평균 3-6% 더 나은 성능[1]

#### 7.6 엣지 배포 연구 비교

**배포 효율성 비교** (2024-2025):

| 모델 | 양자화 | INT8 mAP | FP16 mAP | 배포 용이도 |
|------|--------|---------|---------|-----------|
| YOLOv8 | 가능 | 47.8% | 50.2% | 어려움 |
| YOLOv11 | 가능 | 50.1% | 52.5% | 중간 |
| RT-DETRv3 | 어려움 | 49.2% | 54.2% | 매우 어려움 |
| **YOLO26** | **우수** | **49.8%** | **51.5%** | **매우 쉬움** |

**DFL 제거의 영향:**
- YOLOv8의 INT8 양자화: 2.4% mAP 손실
- YOLO26의 INT8 양자화: 1.7% mAP 손실 (↓ 29% 손실 감소)[1]

***

### 8. 논문이 미치는 영향과 미래 연구 방향

#### 8.1 단기 임팩트 (2025-2026)

**산업 적용:**
- 로봇공학: 실시간 인지 능력 향상 (33% 레이턴시 감소)
- 제조업: 스마트 팩토리 결함 탐지 배포 가속화
- IoT/엣지: Jetson, ARM 기반 기기에서 실시간 성능 가능

**연구 커뮤니티:**
- DFL 제거 논리의 재검토 필요
- ProgLoss, STAL 메커니즘에 대한 추가 연구 자극
- NMS-free 설계 가속화[1]

#### 8.2 중기 영향 (2026-2028)

**아키텍처 진화:**

1. **기초 모델 통합**
   - CLIP, SAM과의 결합
   - 영어 표현 가능한 객체 = 자동으로 탐지 가능
   - 극적인 일반화 성능 향상 예상

2. **하이브리드 설계 표준화**
   - CNN의 효율성 + Transformer의 표현력
   - YOLO26를 기반으로 한 하이브리드 아키텍처 개발 가속화

3. **자동 머신러닝 (AutoML)**
   - 하드웨어별 최적 YOLO 변형 자동 생성
   - "YOLO 커스터마이저" 도구 등장

#### 8.3 장기 비전 (2028+)

**YOLO 계열의 최종 형태:**

$$f_{\text{future}}(x) = \text{Foundation Model}(x) + \text{YOLO Architecture}(x)$$

**특징:**
- 영어로 표현 가능한 모든 객체 탐지 가능
- 실시간 성능 유지
- 극단적 환경에서의 강건성 (악천후, 극저조도, 극고고도)[1]

***

### 9. 향후 연구 시 고려할 점

#### 9.1 기술적 고려사항

1. **아키텍처 설계**
   - 단순성과 효율성이 정확도만큼 중요
   - DFL 제거 논리를 다른 고급 손실에 적용 가능한지 검토

2. **손실 함수 설계**
   - ProgLoss의 일반화 확인 필요
   - 서로 다른 작업(분할, 포즈 추정)에서의 효과 검증

3. **최적화 방법**
   - MuSGD의 하이퍼파라미터 튜닝 가이드라인 필요
   - 다양한 데이터셋에서의 일반화 성능 평가

#### 9.2 실무적 고려사항

1. **배포 전략**
   - 양자화 인식 훈련 (QAT) 도입
   - 하드웨어별 최적화 가이드

2. **평가 메트릭**
   - 레이턴시, 에너지 소비, 메모리 사용을 포함한 종합 평가
   - 도메인 시프트 강건성 평가 프레임워크 개발

3. **데이터 고려사항**
   - 엣지 디바이스에서의 데이터 수집 및 라벨링
   - 소형 객체 풍부 데이터셋의 구축 필요성[1]

#### 9.3 이론적 개선

1. **일반화 이론**
   - ProgLoss가 일반화 갭을 감소시키는 이론적 증명
   - 신경망 이중 강하(Double Descent) 현상과의 관계

2. **최적성 분석**
   - MuSGD의 수렴 속도 분석
   - 비볼록 최적화에서의 성능 경계 도출

3. **복잡도 분석**
   - YOLO26의 계산 복잡도 감소 메커니즘 이론화
   - 정보 이론적 관점에서의 NMS 제거 영향

***

### 10. 결론

YOLO26은 **단순성, 효율성, 혁신**의 원칙을 기반으로 하는 차세대 객체 탐지 모델로, 다음과 같은 핵심 특징을 가집니다:

**주요 성과:**
- DFL 제거로 CPU 추론 43% 가속화
- NMS-free 설계로 배포 복잡도 대폭 감소
- ProgLoss + STAL로 소형 객체 탐지 23.7% AP 달성
- MuSGD 옵티마이저로 50 에포크 내 수렴[1]

**일반화 성능:**
- 도메인 시프트에서 3-5% 우수한 성능
- 소규모 데이터셋에서 YOLOv8 대비 5.1% 향상
- 다양한 환경(항공, 산업, 수중)에서 일관된 성능[1]

**미래 방향:**
- 기초 모델 통합으로 영어 표현 가능한 모든 객체 탐지
- 하이브리드 CNN-Transformer 설계 표준화
- 하드웨어별 자동 최적화 (AutoML) 개발

YOLO26는 학술 혁신과 실제 배포 요구의 완벽한 균형점을 제시하며, 이후 객체 탐지 연구의 새로운 패러다임을 제시할 것으로 기대됩니다.

***

### 참고 자료

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/c126fe53-8963-4051-b26a-29e1553bff3d/2509.25164v2.pdf)
[2](https://ieeexplore.ieee.org/document/10912942/)
[3](https://hightechjournal.org/index.php/HIJ/article/view/773)
[4](https://onlinelibrary.wiley.com/doi/10.1002/tee.24078)
[5](https://arxiv.org/abs/2410.19846)
[6](https://link.springer.com/10.1007/s11760-024-03538-x)
[7](https://www.joig.net/show-104-487-1.html)
[8](https://ieeexplore.ieee.org/document/11291929/)
[9](https://arxiv.org/abs/2509.12682)
[10](https://isprs-annals.copernicus.org/articles/X-2-W2-2025/173/2025/)
[11](https://ieeexplore.ieee.org/document/11088012/)
[12](https://arxiv.org/pdf/2501.13400v1.pdf)
[13](https://arxiv.org/pdf/2307.13901.pdf)
[14](https://arxiv.org/abs/2410.19869)
[15](https://arxiv.org/pdf/2410.22898.pdf)
[16](https://arxiv.org/html/2407.12040v7)
[17](https://arxiv.org/html/2502.14314v3)
[18](https://arxiv.org/html/2411.00201)
[19](https://arxiv.org/pdf/2411.18871.pdf)
[20](https://www.labellerr.com/blog/yolo11-vs-yolov8-model-comparison/)
[21](https://docs.ultralytics.com/models/rtdetr/)
[22](https://www.sciencedirect.com/science/article/abs/pii/S0957417423015385)
[23](https://docs.ultralytics.com/compare/yolov8-vs-yolo11/)
[24](https://www.sciencedirect.com/science/article/abs/pii/S0141938224003123)
[25](https://pmc.ncbi.nlm.nih.gov/articles/PMC10280595/)
[26](https://docs.ultralytics.com/compare/yolo11-vs-yolov8/)
[27](https://www.nature.com/articles/s41598-024-68115-1)
[28](https://arxiv.org/abs/2503.20516)
[29](https://www.joig.net/2025/JOIG-V13N5-515.pdf)
[30](https://openaccess.thecvf.com/content/CVPR2024/papers/Cheng_YOLO-World_Real-Time_Open-Vocabulary_Object_Detection_CVPR_2024_paper.pdf)
[31](https://www.sciencedirect.com/science/article/abs/pii/S0262885624001586)
[32](https://arxiv.org/html/2410.19869v3)
[33](https://dl.acm.org/doi/abs/10.1007/s11554-024-01572-z)
[34](https://arxiv.org/pdf/2503.20516.pdf)
[35](https://arxiv.org/pdf/2509.12682.pdf)
[36](https://arxiv.org/html/2503.06282v2)
[37](https://arxiv.org/html/2411.00201v1)
[38](https://www.arxiv.org/pdf/2508.02067.pdf)
[39](https://arxiv.org/html/2407.16424v1)
[40](https://arxiv.org/html/2507.10775v1)
[41](https://arxiv.org/html/2502.20622v2)
[42](https://arxiv.org/pdf/2505.00044.pdf)
[43](https://arxiv.org/pdf/2412.12349.pdf)
[44](https://openaccess.thecvf.com/content/ACCV2024/papers/Hu_A_Universal_Structure_of_YOLO_Series_Small_Object_Detection_Models_ACCV_2024_paper.pdf)
[45](https://arxiv.org/html/2503.07330v3)
[46](https://pubmed.ncbi.nlm.nih.gov/39021898/)
[47](http://www.proceedings.com/079017-0796.html)
[48](https://www.mdpi.com/2079-9292/13/23/4653)
[49](https://www.sciltp.com/journals/aim/2024/1/430)
[50](https://link.springer.com/10.1007/s11227-024-06773-8)
[51](https://www.semanticscholar.org/paper/c21b64ef9e6bb599b290d3826a807330fe401f2d)
[52](https://arxiv.org/abs/2511.07301)
[53](https://arxiv.org/abs/2509.10503)
[54](https://ieeexplore.ieee.org/document/11164935/)
[55](https://www.mdpi.com/2076-3417/15/20/10877)
[56](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70254)
[57](https://arxiv.org/html/2203.05294v5)
[58](https://arxiv.org/html/2312.12133v1)
[59](https://arxiv.org/html/2405.14497)
[60](http://arxiv.org/pdf/2404.07794.pdf)
[61](http://arxiv.org/pdf/2411.07392.pdf)
[62](http://arxiv.org/pdf/2211.02213.pdf)
[63](https://arxiv.org/pdf/2301.00371.pdf)
[64](https://arxiv.org/html/2310.19351)
[65](https://ise.thss.tsinghua.edu.cn/mig/2024-3.pdf)
[66](https://www.ijcai.org/proceedings/2025/0145.pdf)
[67](https://www.sciencedirect.com/science/article/abs/pii/S0957417424027726)
[68](https://www.nature.com/articles/s41598-025-96314-x)
[69](https://www.sciencedirect.com/science/article/abs/pii/S0924271625000838)
[70](https://arxiv.org/html/2502.05147v1)
[71](https://docs.ultralytics.com/compare/yolov10-vs-yolo11/)
[72](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d3b57e06e3fc45f077eb5c9f28156d4-Paper-Conference.pdf)
[73](https://dl.acm.org/doi/abs/10.1145/3746027.3754737)
[74](https://docs.ultralytics.com/compare/yolo11-vs-yolov10/)
[75](https://arxiv.org/html/2409.16538v1)
[76](https://www.sciencedirect.com/science/article/abs/pii/S0925231225014274)
[77](https://www.sciencedirect.com/science/article/pii/S2772375524002533)
[78](https://arxiv.org/html/2507.02798v1)
[79](https://arxiv.org/html/2504.04242v1)
[80](https://openaccess.thecvf.com/content/CVPR2024W/MAT/papers/Ruan_Fully_Test-time_Adaptation_for_Object_Detection_CVPRW_2024_paper.pdf)
[81](https://arxiv.org/html/2509.25164v1)
[82](https://arxiv.org/html/2510.11090v1)
[83](https://arxiv.org/pdf/2412.14633.pdf)
[84](https://arxiv.org/html/2504.18586v1)
[85](https://arxiv.org/html/2511.07301)
[86](https://arxiv.org/html/2502.05147v2)
[87](https://arxiv.org/html/2411.18871v1)
[88](https://arxiv.org/html/2505.19990v1)
[89](https://arxiv.org/html/2508.02067v1)
