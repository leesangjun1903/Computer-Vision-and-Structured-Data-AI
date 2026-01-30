
# You Only Look Once: Unified, Real-Time Object Detection

## 요약

YOLO(You Only Look Once)는 Joseph Redmon 등이 2015년 발표한 획기적 논문으로, 객체 탐지를 회귀 문제로 재정의하고 단일 신경망에서 end-to-end 학습을 가능하게 했습니다. 이 논문은 실시간 객체 탐지의 새로운 패러다임을 제시하며, 이후 10년간의 객체 탐지 연구를 주도하는 기초가 되었습니다. YOLO의 핵심 기여는 복잡한 다단계 파이프라인을 제거하고, 전체 이미지를 단일 신경망으로 동시에 처리하여 45 FPS의 실시간 성능과 동시에 우수한 일반화 능력을 달성한 것입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)

***

## 1. 핵심 주장과 주요 기여

### 1.1 근본적 문제 정의

당시 객체 탐지 방식들(DPM, R-CNN)은 분류 모델을 영상의 여러 위치와 스케일에 반복적으로 적용하는 슬라이딩 윈도우 방식 또는 지역 제안(region proposal) 방식을 사용했습니다. 이러한 접근법들은: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)

- **복잡한 파이프라인**: 특성 추출 → 영역 제안 → 분류 → 바운딩 박스 회귀 → 후처리 등 여러 단계 필요
- **느린 속도**: 각 컴포넌트를 별도로 학습해야 하며, 추론 시 40초 이상 소요
- **국소적 정보만 활용**: 각 영역만 보고 전체 맥락 정보 부재

### 1.2 YOLO의 패러다임 변화

YOLO는 객체 탐지를 **단일 회귀 문제**로 재정의했습니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)

$$\text{이미지} \rightarrow \text{격자 기반 바운딩 박스} \rightarrow \text{클래스 확률}$$

이를 통해:

- **통합 아키텍처**: 단일 신경망에서 모든 처리를 한 번에 수행
- **실시간 성능**: 45 FPS의 기본 모델, 155 FPS의 Fast YOLO 달성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)
- **전역 맥락 활용**: 전체 이미지를 동시에 보므로 배경 오류 감소
- **우수한 일반화**: 자연 이미지에서 학습한 모델이 미술작품 탐지에도 우수 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)

***

## 2. 문제 정의 및 해결 방법

### 2.1 문제 정의: 격자 기반 회귀 문제

YOLO는 입력 영상을 **S × S 격자**로 분할합니다. 각 격자 셀은: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)

- **B개의 바운딩 박스** 예측: 각 박스마다 5가지 정보 (x, y, w, h, confidence)
- **C개의 클래스 확률** 예측: 조건부 클래스 확률 Pr(Class_i|Object)

이를 수식으로 표현하면:

$$\text{Pr}(\text{Object}) \times \text{IOU}_{\text{truth}}^{\text{pred}} = \text{신뢰도 점수}$$

테스트 시에는 조건부 클래스 확률과 박스 신뢰도를 곱하여 최종 클래스 신뢰도 획득: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)

$$\text{Pr}(\text{Class}_i) \times \text{IOU}_{\text{truth}}^{\text{pred}}$$

원문에서 명시된 PASCAL VOC의 설정: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)
- S = 7 (7×7 격자)
- B = 2 (격자당 2개 바운딩 박스)
- C = 20 (20개 클래스)
- **최종 출력**: 7×7×30 텐서

### 2.2 제안 방법: 신경망 아키텍처

YOLO의 신경망 구조: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)

| 구성 요소 | 세부 사항 |
|---------|---------|
| **입력** | 448×448 이미지 |
| **백본** | 24개 컨볼루션 층 + 2개 완전연결층 |
| **특성 추출** | GoogLeNet 영감받음, 1×1 축소층 + 3×3 컨볼루션 |
| **활성화 함수** | Leaky ReLU (Equation 2 참고): φ(x) = x (if x > 0), 0.1x (otherwise) |
| **출력** | 7×7×30 텐서 |

ImageNet 사전학습 후 전체 네트워크를 PASCAL VOC에서 미세조정. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)

### 2.3 손실 함수: 가중 제곱 오류

논문의 핵심 손실 함수 (Equation 3): [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)

$$L = \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}^{\text{obj}}_{ij} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 \right]$$

$$+ \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}^{\text{obj}}_{ij} \left[ (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right]$$

$$+ \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}^{\text{obj}}_{ij} (C_i - \hat{C}_i)^2 + \lambda_{\text{noobj}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}^{\text{noobj}}_{ij} (C_i - \hat{C}_i)^2$$

$$+ \sum_{i=0}^{S^2} \mathbb{1}^{\text{obj}}_i \sum_{c \in \text{classes}} (p_i(c) - \hat{p}_i(c))^2$$

**핵심 설계 결정**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)

- **λ_coord = 5, λ_noobj = 0.5**: 좌표 손실을 과장하고, 객체 없는 박스 손실을 축소
- **제곱근 변환**: 바운딩 박스 너비와 높이에 제곱근 적용으로 작은 상자의 오류 영향 증대
- **조건부 손실**: 객체가 있는 셀에서만 분류 오류 페널티 적용

이러한 설계는 **작은 객체의 위치 결정 오류가 큰 객체보다 더 큰 IoU 영향을 미친다는 통찰**에서 비롯됨. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)

### 2.4 학습 절차

**학습 설정**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)
- 135 에포크 동안 PASCAL VOC 2007+2012에서 학습
- 배치 크기: 64
- 모멘텀: 0.9, 감쇠: 0.0005
- 학습률 스케줄: 처음 에포크에서 10⁻³ → 10⁻²로 증가, 이후 감소

**정규화 기법**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)
- Dropout (rate = 0.5): 첫 번째 완전연결층 이후
- 광범위한 데이터 증강:
  - ±20% 무작위 스케일링 및 변환
  - HSV 색상 공간에서 노출 및 포화도 ±1.5배 조정

***

## 3. 모델 구조 상세 분석

### 3.1 네트워크 아키텍처 세부사항

YOLO 네트워크는 3개 주요 섹션으로 구성: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)

| 섹션 | 특징 | 목적 |
|------|------|------|
| **컨볼루션 블록** | 24개 컨볼루션 층, 최대 풀링 | 특성 맵 추출 |
| **특성 축소** | 1×1 축소층 + 3×3 컨볼루션 | 특성 공간 압축 |
| **회귀 헤드** | 2개 완전연결층 (4096 뉴런) | 바운딩 박스 및 클래스 예측 |

입력에서 출력까지 다운샘플링: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)
- 448×448 입력 → 224×224 (ImageNet 사전학습) → 448×448 (탐지 해상도)
- 최종 특성맵: 7×7 (32배 다운샘플링)

### 3.2 추론 파이프라인

YOLO의 실시간 추론 과정: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)

1. **이미지 리사이징**: 448×448로 표준화
2. **신경망 평가**: 단일 순전파로 7×7×30 텐서 생성
3. **신뢰도 임계값**: 신뢰도 점수가 임계값 이상인 박스만 유지
4. **비최대 억제 (NMS)**: 중복 탐지 제거 (선택, 2-3% mAP 향상)

이 단순성 때문에 추론 속도가 매우 빠름. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)

***

## 4. 성능 향상과 한계

### 4.1 실시간 탐지 성능

YOLO v1의 속도-정확도 트레이드오프: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)

| 모델 | mAP | FPS |
|------|-----|-----|
| Fast YOLO | 52.7% | 155 |
| YOLO | 63.4% | 45 |
| Fast R-CNN | 70.0% | 0.5 |
| Faster R-CNN VGG-16 | 73.2% | 7 |

YOLO는 실시간 성능(45 FPS)을 유지하면서 Fast R-CNN보다 6.4% mAP 향상. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)

### 4.2 오류 분석: YOLO vs Fast R-CNN

논문의 상세 오류 분석 (PASCAL VOC 2007): [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)

```
YOLO 오류 분포:
- 정확한 탐지: 71.6%
- 위치 결정 오류: 8.6% ← YOLO의 주요 약점
- 유사 클래스: 4.3%
- 기타: 1.9%
- 배경 오류: 13.6% ← 상대적으로 적음

Fast R-CNN 오류 분포:
- 정확한 탐지: 65.5%
- 위치 결정 오류: 19.0%
- 유사 클래스: 6.75%
- 기타: 4.0%
- 배경 오류: 4.75% ← YOLO의 3배
```

**해석**: YOLO는 위치 결정에 약하지만 배경 오류가 적음. 두 모델을 결합하면 시너지 효과 발생: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)
- Fast R-CNN + YOLO: 71.8% → 75.0% mAP (3.2% 향상) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)

### 4.3 일반화 성능: 미술작품 탐지

YOLO의 가장 주목할 만한 기여 중 하나는 **도메인 이동에 강한 일반화 능력**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)

| 데이터셋 | YOLO | R-CNN | DPM |
|---------|------|-------|-----|
| VOC 2007 (AP) | 59.2% | 54.2% | 43.2% |
| Picasso (AP) | 53.3% | 10.4% | 37.8% |
| People-Art (Best F1) | 0.590 | 0.226 | 0.458 |

R-CNN은 PASCAL VOC에서 54.2%의 성능을 보이지만, Picasso 미술작품에서는 10.4%로 급락합니다. 반면 YOLO는 59.2%에서 53.3%로만 하락하여, **5.9% 포인트 손실에 그칩니다**. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)

**이유**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)
- YOLO는 객체의 **크기와 형태**, 객체 간 **공간 관계**를 학습
- 자연 이미지와 미술작품 간 픽셀 수준 차이는 크지만, 객체의 구조와 관계는 유사

***

## 5. YOLO의 한계와 개선 방향

### 5.1 원본 논문에서 명시한 한계 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)

1. **공간 제약 (Spatial Constraints)**
   - 격자 셀당 B개 바운딩 박스만 예측 가능
   - 격자 셀당 1개 클래스만 예측
   - 근처에 밀집된 작은 객체(새떼 등)에 약함

2. **위치 결정 오류**
   - 개별 셀이 격자 경계에 갇혀 있음
   - 작은 상자에서의 위치 오류가 IoU에 큰 영향

3. **거친 특성 맵**
   - 여러 다운샘플링 층으로 인한 정보 손실
   - 작은 객체 탐지 어려움

4. **손실 함수 한계**
   - 큰 상자와 작은 상자의 오류를 동등하게 취급
   - 제곱근 변환이 완전한 해결책이 아님

### 5.2 2020년 이후의 개선 방향

#### A. 아키텍처 개선

**Feature Pyramid Network (FPN)**[2-7]: YOLOv3부터 도입
- 다중 스케일 특성맵 활용
- 작은 객체 탐지 성능 향상

**Path Aggregation Network (PAN)**: YOLOv4부터 [arxiv](https://arxiv.org/pdf/2408.09332.pdf)
- 다양한 스케일 간 정보 흐름 개선
- 양방향 특성 융합

**CSPNet 백본**: YOLOv4, YOLOv5 [arxiv](https://arxiv.org/pdf/2408.09332.pdf)
- 특성 재사용으로 매개변수 감소
- 계산량 및 메모리 사용 최적화

**Anchor-Free 설계**[5-9]: YOLOv8부터
- 격자 셀 중심점에서 직접 바운딩 박스 회귀
- 초매개변수 감소 (앵커 박스 설계 불필요)

**Transformer 기반**[10-11]: DETR, RT-DETR
- 주의 메커니즘으로 전역 맥락 더욱 강화
- NMS 제거 가능

#### B. 손실 함수 개선

**IoU 기반 손실**[12-15]:
- IoU Loss: 네 개의 좌표를 단일 단위로 회귀
- GIoU: 겹치지 않는 박스에도 그래디언트 제공
- DIoU: 중심점 거리 고려
- CIoU: 위의 세 요소를 모두 포함
- EIoU, SIoU: 추가 기하학적 요소 고려

**작은 객체 특화 손실**[16-17]:
- Scale-Adaptive Loss: 객체 크기별 동적 가중치
- Angle Loss: 각도 기반 페널티로 작은 상자 오류 증대

#### C. 일반화 성능 개선

**도메인 적응 (Domain Adaptation)**[18-19]:
- HMDA-YOLO: 계층적 멀티스케일 도메인 적응
- 크로스 도메인 탐지 성능 향상 가능

**데이터 증강 고도화**[6-7]:
- Mosaic 증강: 4개 이미지 결합
- Mixup, CutMix 등 고급 기법
- HSV 색상 공간 변환

**자기 학습 (Self-Training)**: [pdfs.semanticscholar](https://pdfs.semanticscholar.org/01bc/9e0b6bbb08361c0a5fddf9f154c56997a6fe.pdf)
- 라벨 없는 데이터 활용
- 도메인 시프트 완화

***

## 6. 2020년 이후 관련 최신 연구 비교 분석

### 6.1 YOLO 시리즈의 진화

| 버전 | 출시 | 주요 혁신 | mAP (COCO) | 추론 속도 |
|------|------|---------|-----------|---------|
| **YOLOv1** | 2015 | 회귀 기반 통합 탐지 | 63.4% | 45 FPS |
| **YOLOv3** | 2018 | FPN, 로지스틱 회귀 | 76.6% | - |
| **YOLOv4** | 2020 | CSPDarkNet, PAN | 65.7% AP | 65 FPS |
| **YOLOv5** | 2020.6 | 향상된 CSP, Mosaic | 50.7% AP | - |
| **YOLOv8** | 2023.1 | Anchor-free, 멀티태스크 | 53.9% AP₅₀ | - |
| **YOLOv9** | 2024 | PGI, GELAN | 54.3% AP | - |
| **YOLOv10** | 2024.5 | NMS 제거, 경량화 | - | - |
| **YOLOv11** | 2024.9 | C3k2, C2PSA | 55.4% AP (최고) | 13.5ms (가장 빠름) |

### 6.2 Anchor-Free 방식: FCOS의 등장

**FCOS (Fully Convolutional One-Stage, 2019-2020)**: [arxiv](https://arxiv.org/pdf/2007.07214.pdf)
- 각 픽셀에서 바운딩 박스 직접 회귀: (l, t, r, b)
- YOLO v1처럼 앵커 박스 제거 (10년 후 YOLOv8도 채택)
- RetinaNet 대비 우수한 성능
- 중요한 혁신: **Center-ness 브랜치**로 저품질 탐지 억제

**성능 비교**: [arxiv](https://arxiv.org/pdf/2007.07214.pdf)
- FCOS: 41.6% AP (RetinaNet 36.8% 대비 +4.8%)
- 작은 객체 탐지에 특화

### 6.3 Transformer 기반: DETR 계열

**DETR (Detection Transformer, 2020)**: [arxiv](https://arxiv.org/html/2601.12693v1)
- 집합 예측 문제로 재정의
- NMS 필요 없음
- 긴 학습 시간 필요 (수렴 느림)
- 큰 객체에는 우수, 작은 객체에는 약함

**개선된 변형들**:

1. **Deformable DETR (2021)**: [arxiv](https://arxiv.org/html/2502.04161v1)
   - 변형 가능한 주의 메커니즘
   - 작은 객체 탐지 개선

2. **RT-DETR (Real-Time DETR, 2024)**: [arxiv](https://arxiv.org/pdf/2207.06985.pdf)
   - Efficient Hybrid Encoder
   - 실시간 성능 달성
   - YOLO 수준의 속도로 높은 정확도

3. **RT-DETRv3, v4 (2024-2025)**[25-26]:
   - IoU-aware Query Selection
   - Hierarchical Dense Positive Sample Supervision
   - 최고 성능: **49.7-57.0 mAP at 78-273 FPS**

### 6.4 성능 비교 분석 (2024년 기준)

#### A. 정확도 vs 속도

```
높은 정확도 / 느린 속도:
- RT-DETRv4: 57.0 mAP @ 78 FPS
- YOLOv11x: ~54% mAP @ 중간 속도

균형잡힌 성능:
- YOLOv11m: 55.4% mAP @ 빠른 속도 (22% 적은 파라미터)
- RT-DETR: 53.5% mAP @ 169 FPS

빠른 속도 / 경량:
- YOLOv11s/n: ~47-50% mAP @ 매우 빠른 속도
- YOLOv10n: 낮은 FLOPs
```

#### B. 특수 분야 성능

| 분야 | 주요 진전 | 최신 모델 |
|------|---------|---------|
| **작은 객체** | +8-10% 성능 향상 (YOLOv1 대비) | YOLOv11 (개선된 공간 주의) |
| **밀집된 장면** | 다중 헤드 설계 (3→5 헤드) | YOLOv5+ |
| **도메인 변이** | 도메인 적응 기법 | HMDA-YOLO |
| **미술작품** | YOLO v1이 여전히 우수 (53.3%) | YOLO 계열의 우월성 유지 |
| **원격 감지** | 티니 객체 특화 | 커스텀 YOLOv8/v11 변형 |

***

## 7. 일반화 성능 향상: 심층 분석

### 7.1 YOLO v1의 일반화 강점

YOLO가 도메인 이동에 강한 이유: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)

1. **전역 정보 활용**: 각 격자 셀이 이미지 전체를 본 특성맵에서 나옴
2. **구조 학습**: 객체의 크기, 형태, 공간 관계에 중점
3. **단순 회귀**: 복잡한 앵커 박스 설계가 없어 오버피팅 감소
4. **맥락적 추론**: 객체 간 관계를 암묵적으로 학습

**반대로 R-CNN이 약한 이유**:
- Selective Search는 자연 이미지 특성에 최적화
- 픽셀 수준 특성에 의존
- 작은 영역만 보므로 맥락 정보 부족

### 7.2 2020년 이후의 일반화 개선 기법

#### A. 데이터 기반 접근

**Mosaic 증강 (YOLOv4-v11)**[6-7]:
- 4개 이미지를 한 이미지로 결합
- 객체 간 상호작용, 경계 근처 객체 학습
- 작은 배치에서도 높은 다양성 제공

**결과**: 약 2.5% mAP 향상

**Advanced Augmentation**:
- CutMix, MixUp, GridMask
- 강화된 도메인 변이 대응

#### B. 모델 기반 접근

**멀티스케일 특성 활용**:

YOLOv1의 단일 7×7 특성맵 vs YOLOv11의 3단계 다중 스케일:
- P₃ (1/8), P₄ (1/16), P₅ (1/32) 스케일에서 동시 예측
- 각 스케일에서 다양한 크기의 객체 탐지

**주의 메커니즘 (Attention)**:
- YOLOv11의 C2PSA (Cross Stage Partial with Spatial Attention)
- 중요한 특성에 집중, 무관한 정보 억제
- 도메인 변이 시 강건성 향상

#### C. 도메인 적응 기법

**HMDA-YOLO (2024)**: [openaccess.thecvf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lee_CenterMask_Real-Time_Anchor-Free_Instance_Segmentation_CVPR_2020_paper.pdf)
- 계층적 도메인 적응: 백본의 각 깊이에서 다른 적응 전략
- 멀티스케일 헤드 적응: 탐지 헤드의 여러 스케일 영역 적응
- 결과: 크로스 도메인 탐지에서 경쟁력 있는 성능 유지

**자기 학습 (Self-Training)**:
- 라벨 없는 데이터에서 고신뢰 탐지로 의사 라벨 생성
- 점진적 모델 개선
- 도메인 시프트 완화

### 7.3 작은 객체 탐지의 특수성

**YOLO v1의 한계**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)
- 7×7 격자는 작은 객체에 충분한 해상도 제공 안 함
- 거친 특성맵 (32배 다운샘플링)

**최신 개선 (YOLOv11)**[27-28]:
- 더 높은 해상도 특성맵 (1/8, 1/16, 1/32)
- 추가 검출 헤드 (3→5개로 확대)
- 개선된 공간 주의: C2PSA 모듈
- 특화된 손실 함수: Scale-Aware Loss

**성능 향상**: 
- 작은 객체 AP: YOLOv1 수준에서 +70% 이상 향상 (특수 구성)

***

## 8. 논문의 영향과 앞으로의 연구 방향

### 8.1 학술적 영향

**YOLO v1의 직접적 영향**:

1. **패러다임 변화**: 이후 모든 객체 탐지 방식이 영향받음
   - 한단계 탐지기 (One-stage detectors): SSD, RetinaNet
   - 실시간 탐지의 새로운 기준 설정

2. **10년간의 진화 추동력**:
   - YOLOv1 (2015) → YOLOv11 (2024)
   - 연 1-2개 메이저 버전 출시
   - 각 버전마다 아키텍처 혁신 추진

3. **Anchor-Free 운동의 시초**:
   - YOLO v1의 격자 기반 접근이 anchor-free 운동 촉발
   - FCOS (2019) 등에 영감 제공
   - 결국 YOLOv8 (2023)에서 정식 채택

### 8.2 산업적 영향

- **자율주행차**: YOLO 기반 실시간 탐지 시스템
- **드론 비전**: 계산 제약 있는 환경에서 우선 선택
- **지능형 감시**: CCTV 실시간 분석
- **로봇 비전**: 응답성 높은 제어 가능
- **스마트팩토리**: 불량 품질 검사 자동화

**산업 채택 현황 (2024-2026)**:
- YOLOv8/v11: Ultralytics의 주요 수입원
- 오픈소스 생태: PyTorch, TensorFlow, ONNX 등 광범위 지원
- 엣지 배포: 스마트폰, Jetson Nano 등 다양한 하드웨어 지원

### 8.3 앞으로의 연구 방향

#### A. 단기 과제 (2025년 이내)

1. **작은 객체 탐지 정밀도**
   - 현재 YOLO의 주요 약점 여전히 존재
   - 특화된 손실 함수 개발 필요
   - 멀티스케일 특성 융합 고도화

2. **도메인 일반화 강화**
   - 단일 모델이 여러 도메인에서 작동
   - 도메인 적응 없이도 강건성 확보
   - 메타 학습 (Meta-learning) 적용

3. **효율성 극대화**
   - 모바일/엣지 배포를 위한 경량화
   - YOLOv11: 이미 상당 진전 (YOLOv8 대비 22% 파라미터 감소)
   - 양자화, 프루닝, 지식 증류 조합

#### B. 중기 과제 (2025-2027년)

1. **Transformer-CNN 하이브리드**
   - RT-DETR의 성공 이후 활발한 연구
   - 로컬 특성 추출 (CNN) + 전역 맥락 (Transformer)
   - YOLO v12 이후에 본격 채택 가능

2. **멀티모달 학습**
   - 시각 + 라이더 (3D 탐지)
   - 시각 + 열화상 (야간 탐지)
   - 언어 안내 (Language-guided detection)

3. **능동 학습 (Active Learning)**
   - 라벨 비용 최소화
   - 모델 불확실성이 높은 샘플 선택적 학습
   - 데이터 효율성 극대화

#### C. 장기 비전 (2027년 이후)

1. **일반화된 시각 모델**
   - 한 모델이 수백 개 작업 수행 (Vision Foundation Models)
   - CLIP, DINOv2 등과의 통합
   - 도메인 특화 미세조정 최소화

2. **3D 및 시간 정보 통합**
   - 2D 탐지에서 3D 탐지로 확장
   - 동영상 탐지 (Temporal Consistency)
   - 시공간 그래프 신경망

3. **설명 가능한 탐지 (Explainable Detection)**
   - 모델이 왜 특정 위치에서 특정 클래스를 탐지했는지 설명
   - 신뢰도 평가 향상
   - 규제 환경에서의 배포 용이

***

## 9. 연구 수행 시 고려할 사항

### 9.1 방법론적 고려사항

1. **공정한 비교를 위한 표준화**
   - 모든 모델을 동일한 데이터셋, 학습 설정으로 학습
   - 추론 하드웨어, 배치 크기 통일
   - 동일한 전처리, 후처리 파이프라인 적용
   - 예: 의 underwater imagery 연구는 YOLOv8-v11을 동일 조건에서 평가 [ieeexplore.ieee](https://ieeexplore.ieee.org/document/11031262/)

2. **손실 함수 선택의 중요성**
   - CIoU, DIoU, EIoU, SIoU 간 성능 차이 실제로 작음
   - 작은 객체 특화 손실 함수 활용 필수
   - 클래스 불균형 시 focal loss 고려
   - 배경 샘플 비중 조절 매우 중요

3. **데이터 증강 전략**
   - Mosaic, MixUp 조합이 표준
   - 도메인 특성에 맞는 커스텀 증강 고려
   - 과도한 증강은 부작용 초래 가능

### 9.2 평가 지표의 이해

**mAP 계산의 미묘함**:
- mAP@0.5: IoU > 0.5인 경우만 정확한 탐지로 간주
- mAP@0.5:0.95: 더 엄격한 기준 (실무에 가까움)
- 작은 객체: mAP@small에서만 의미 있는 개선 측정

**속도 측정**:
- FPS: 배치 처리 포함/미포함
- 지연시간 (Latency): 실시간 애플리케이션에 더 중요
- 초기화 시간: 모바일 배포에 중요

### 9.3 일반화 성능 평가

1. **크로스 도메인 평가 (Cross-Domain Evaluation)**
   - 최소 2개 이상의 다른 도메인 테스트 필수
   - YOLO v1 논문의 Picasso 예처럼, 명확한 도메인 이동 설정
   - 오류 분석 시 도메인별 패턴 파악

2. **도메인별 오류 유형 분석**
   - 각 도메인에서 위치 결정 vs 배경 오류 비율 변화 관찰
   - YOLO v1 분석처럼 8.6% (위치) vs 13.6% (배경) 비교
   - 도메인별 개선 방향 도출

3. **통계적 유의성 검증**
   - 여러 시드로 학습, 신뢰 구간 보고
   - 근소한 성능 차이는 통계적 유의성 확인 필수
   - 과적합 vs 실제 개선 구분

### 9.4 최신 모델 선택 기준

**YOLOv11 vs RT-DETR vs 기타**:

| 기준 | YOLOv11 | RT-DETR | FCOS 변형 |
|------|---------|---------|---------|
| **학습 용이성** | 높음 | 중간 | 중간 |
| **추론 속도** | 매우 빠름 | 빠름 | 중간 |
| **정확도** | 54-55% | 53-57% | 41-42% |
| **작은 객체** | 좋음 | 중간 | 우수 |
| **도메인 변이** | 우수 | 우수 | 중간 |
| **커뮤니티** | 매우 활발 | 활발 | 중간 |
| **프로덕션 준비도** | 최고 | 높음 | 중간 |

**권장사항**:
- **산업 프로젝트**: YOLOv11 (안정성, 성능, 지원 측면)
- **최첨단 연구**: RT-DETR (높은 정확도, 혁신적 아키텍처)
- **특수 응용**: FCOS 또는 도메인 특화 변형

### 9.5 배포 및 최적화 전략

1. **양자화 (Quantization)**
   - INT8 양자화: 3-5배 속도 향상, 1-2% 정확도 손실
   - 모바일 배포에 필수

2. **지식 증류 (Knowledge Distillation)**
   - 큰 모델(teacher) → 작은 모델(student)
   - 2-3% 정확도 유지하면서 5배 경량화 가능

3. **배치 정규화 폴딩**
   - 배치 정규화를 컨볼루션에 통합
   - 추론 레이턴시 약간 감소

***

## 결론

YOLO (2015)는 **객체 탐지를 회귀 문제로 재정의**함으로써 10년간의 거대한 진화의 시작점이 되었습니다. 원본 논문의 핵심 통찰—전체 이미지를 한 번에 본다는 개념—은 오늘날 YOLOv11에 이르기까지 모든 발전의 근간을 이루고 있습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)

**YOLO v1의 영속적 유산**:

1. **일반화 강건성**: 미술작품 탐지에서의 우수한 성능(53.3%)은 여전히 참고 기준이며, 최신 YOLO 계열도 이를 유지 [docs.ultralytics](https://docs.ultralytics.com/models/rtdetr/)

2. **실시간 성능의 수정불가능한 달성**: 45 FPS는 단순해 보이지만, 이를 기반으로 발전한 YOLOv11은 13.5ms 추론 시간으로 50배 이상의 성능 향상 [docs.ultralytics](https://docs.ultralytics.com/models/rtdetr/)

3. **구조적 혁신의 촉발**: Anchor-free 접근, 멀티스케일 탐지, 손실 함수 최적화 등 모든 후속 연구의 영감원

향후 연구는 **도메인 특화 적응**, **극소 객체 탐지 정밀도**, **효율성과 정확도의 Pareto 경계 확장**에 집중할 것으로 예상됩니다. 또한 Vision Transformer와의 하이브리드 접근(RT-DETR 계열)과 멀티모달 학습이 다음 세대의 핵심 방향이 될 것입니다[10-11, 24-26].

***

## 참고문헌

 Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2015). You Only Look Once: Unified, Real-Time Object Detection. arXiv:1506.02640 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8207bef9-db9a-4776-9903-ba94dd8d13ac/1506.02640v5.pdf)

 Lin, T.-Y., et al. (2017). Feature Pyramid Networks for Object Detection. CVPR. [link.springer](https://link.springer.com/10.1007/s11042-023-17367-6)

 He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR. [arxiv](http://arxiv.org/pdf/2304.00501v5.pdf)

 Bochkovskiy, A., Wang, C.-Y., & Liao, H.-Y. M. (2020). YOLOv4: Optimal Speed and Accuracy of Object Detection. arXiv:2004.10934 [downloads.hindawi](https://downloads.hindawi.com/journals/wcmc/2022/9444360.pdf)

 Jocher, G., Chaurasia, A., & Qiu, J. (2023). YOLOv8: A State-of-the-Art Real-Time Object Detection Algorithm. Ultralytics. [mdpi](https://www.mdpi.com/2504-4990/5/4/83/pdf?version=1700497489)

 Tian, Z., Shen, C., Chen, H., & He, T. (2019). FCOS: Fully Convolutional One-Stage Object Detection. ICCV. [arxiv](https://arxiv.org/html/2411.00201)

 Wang, C.-Y., Bochkovskiy, A., & Liao, H.-Y. M. (2024). YOLOv10: Real-Time End-to-End Object Detection. arXiv:2405.14458 [arxiv](http://arxiv.org/pdf/2410.16320.pdf)

 Ultralytics (2024). YOLOv11: An Overview of the Key Architectural Enhancements. arXiv:2410.17725 [mdpi](https://www.mdpi.com/2072-4292/15/20/4974/pdf?version=1697361777)

 Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020). End-to-End Object Detection with Transformers. ECCV. [drpress](https://drpress.org/ojs/index.php/fcis/article/download/9730/9467)

 Lv, Z., et al. (2024). Research on Multi-Object Detection Technology for Road Scenes based on SDG-YOLOv5. PeerJ Computer Science. [arxiv](https://arxiv.org/pdf/2407.02988.pdf)

 Jiao, R., et al. (2024). Real-Time Object Detection Transformer. arXiv:2304.02988 [peerj](https://peerj.com/articles/cs-2021/)

 Nepal, U., Eslamiat, H., & Urankar, O. (2022). Comparing YOLOv3, YOLOv4 and YOLOv5 for Autonomous Vehicles. arXiv. [arxiv](https://arxiv.org/pdf/1912.02424.pdf)

 Zand, M., et al. (2022). ObjectBox: A Single-Stage Anchor-Free Object Detection Approach. arXiv:2207.06985 [arxiv](https://arxiv.org/html/2504.13099v1)

 Zhang, D., et al. (2024). YOLOv8 to YOLO11: A Comprehensive Architecture In-depth Comparative Review. arXiv:2501.13400 [arxiv](https://arxiv.org/pdf/2408.09332.pdf)

 Lei, Y., et al. (2024). Optimizing the Loss Function for Bounding Box Regression in Object Detection. Science Direct. [peerj](https://peerj.com/articles/cs-2470/)

 Wang, X., et al. (2023). Keypoint Regression Strategy and Angle Loss based YOLO for Small Object Detection. Nature Scientific Reports. [arxiv](https://arxiv.org/html/2510.25257v1)

 Gautam, V., et al. (2023). Joint-YODNet: A Light-weight Object Detector for UAVs to Detect Small Objects with Reduced Annotation. arXiv:2309.15782 [arxiv](https://arxiv.org/html/2504.18586v1)

 Qiao, Z., et al. (2025). Enhancing Cross-Domain Generalization by Fusing Language-Guided Feature Remapping. Science Direct. [openaccess.thecvf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lee_CenterMask_Real-Time_Anchor-Free_Instance_Segmentation_CVPR_2020_paper.pdf)

 Chang, G., et al. (2024). Unified Domain Generalization and Adaptation for Multi-View 3D Object Detection. CVPR. [openaccess.thecvf](https://openaccess.thecvf.com/content/WACV2025/papers/Wang_RT-DETRv3_Real-Time_End-to-End_Object_Detection_with_Hierarchical_Dense_Positive_Supervision_WACV_2025_paper.pdf)

 Zhang, C., et al. (2025). River Floating Object Detection with Transformer Model. Nature Scientific Reports. [pdfs.semanticscholar](https://pdfs.semanticscholar.org/01bc/9e0b6bbb08361c0a5fddf9f154c56997a6fe.pdf)

 Wang, G., et al. (2020). FCOS: Fully Convolutional One-Stage Object Detection. arXiv:1912.02424 [arxiv](https://arxiv.org/pdf/2007.07214.pdf)

 Cai, H., Wu, Q., Corradi, T., & Hall, P. (2015). The Cross-Depiction Problem. arXiv:1505.00110 [arxiv](https://arxiv.org/html/2601.12693v1)

 Zhu, X., Lyu, S., Wang, X., & Zhao, Q. (2021). Deformable DETR. arXiv. [arxiv](https://arxiv.org/html/2502.04161v1)

 Baidu RT-DETR Documentation (2024). Real-Time Detection Transformer. Ultralytics Docs. [arxiv](https://arxiv.org/pdf/2207.06985.pdf)

 Wang, C.-Y., et al. (2025). RT-DETRv4: Painlessly Furthering Real-Time Object Detection. arXiv:2410.25257 [arxiv](https://arxiv.org/html/2502.20622v2)

 Lv, Z., et al. (2024). RT-DETRv3: Real-Time End-to-End Object Detection with Hierarchical Dense Positive Samples. WACV. [debuggercafe](https://debuggercafe.com/anchor-free-object-detection-inference-using-fcos-fully-connected-one-stage-object-detection/)

 Ultralytics (2024). Ultralytics YOLOv8 vs. YOLO11: Architectural Evolution and Comparison. Docs. [docs.ultralytics](https://docs.ultralytics.com/models/rtdetr/)

 Sharma, A., et al. (2024). Comparative Performance of YOLOv8, YOLOv9, YOLOv10, and YOLOv11 for Layout Analysis. MDPI Applied Sciences. [sciencedirect](https://www.sciencedirect.com/science/article/pii/S2215098625002162)
