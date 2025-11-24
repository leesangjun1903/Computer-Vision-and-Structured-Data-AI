# DiffusionDet: Diffusion Model for Object Detection
### 1. 핵심 주장과 주요 기여
**DiffusionDet**은 객체 탐지(object detection) 문제를 **노이즈에서 정제로의 확산 과정(denoising diffusion process)**으로 재정의한 획기적인 프레임워크이다. 이 논문의 핵심 주장은 기존의 고정된 학습 가능한 쿼리(learnable queries)나 휴리스틱 객체 후보(heuristic object priors)에 의존하지 않고, 완전히 무작위 박스에서 시작하여 점진적으로 정제하는 방식이 더 유연하고 효과적이라는 것이다.[1]

주요 기여는 다음과 같다:

- **첫 확산 모델 적용**: 객체 탐지에 확산 모델을 성공적으로 적용한 최초 연구[1]
- **유연한 아키텍처**: 학습 시와 평가 시의 박스 개수 분리, 반복적 평가 가능[1]
- **우수한 일반화 성능**: COCO에서 CROWDHUMAN으로의 제로샷 전이에서 5.3 AP 향상[1]
- **광범위한 벤치마크 검증**: COCO, LVIS, CrowdHuman 등 다중 데이터셋에서 우수한 성능 입증[1]

***

### 2. 해결 문제와 제안 방법
#### 2.1 문제 정의

기존의 DETR 이후 쿼리 기반 탐지 방법들은 다음의 한계를 지닌다:

- **고정 쿼리 의존성**: 학습 시 정해진 개수의 학습 가능한 쿼리에 의존
- **평가 유연성 부족**: 평가 시 다른 개수의 후보를 사용하면 성능 저하
- **복잡한 파이프라인**: 다양한 휴리스틱 요소(앵커, 프로포절 등)를 포함

#### 2.2 제안 방법

**확산 프로세스 공식화**:

DiffusionDet은 객체 탐지를 조건부 확산 모델로 정식화한다. 순방향 노이징 프로세스는:[1]

$$q(z_t | z_0) = \mathcal{N}(z_t | \sqrt{\bar{\alpha}_t} z_0, (1-\bar{\alpha}_t)I)$$

여기서 $$z_0 = b$$는 지정된 박스들의 집합이고, $$\bar{\alpha}\_t = \prod_{i=1}^t \alpha_i$$이며, $$\alpha_i$$는 분산 스케줄에 의해 제어된다.[1]

**역프로세스 학습**:

신경망 $$f_\theta(z_t, t, x)$$는 조건부 이미지 특성 $$x$$를 고려하여 $$z_0$$을 예측하도록 학습되며, 손실 함수는:

$$\mathcal{L} = \mathbb{E}_{t} \lVert f_\theta(z_t, t, x) - z_0 \rVert_2^2$$

**신호 스케일링**:

DiffusionDet은 이미지 생성 태스크와 달리, 박스 표현이 단 4개의 파라미터(중심 좌표 $$c_x, c_y$$, 너비 $$w$$, 높이 $$h$$)만 가지므로, 신호 대 노이즈 비(SNR)를 증가시켜 더 강한 학습 신호를 유지한다. 최적의 신호 스케일 값은 2.0으로 설정된다.[1]

***

### 3. 모델 구조
#### 3.1 아키텍처 구성

모델은 두 개의 주요 모듈로 구성된다:[1]

**이미지 인코더(Image Encoder)**:
- ResNet-50 또는 Swin Transformer 백본 사용
- Feature Pyramid Network(FPN)으로 다중 스케일 특성맵 생성
- 이미지 특성을 한 번만 추출하여 계산 효율성 극대화

**탐지 디코더(Detection Decoder)**:
- 6개의 캐스케이드 스테이지 구성
- 타임스텝 임베딩을 통해 확산 과정의 진행 단계 인코딩
- RoI Align을 사용하여 노이즈 박스로부터 RoI 특성 추출
- 반복적 평가 시 전체 디코더 헤드를 재사용 가능

#### 3.2 학습 프로세스

**알고리즘 1: DiffusionDet 학습**

입력: 이미지 배치 $$B \times H \times W \times 3$$, 지정 박스 $$B \times N \times 4$$

1. 이미지 인코더를 통해 특성 추출: $$\text{feats} = \text{ImageEncoder}(\text{images})$$

2. 지정 박스를 $$N$$개로 패딩(무작위 가우시안 박스 연결)

3. 신호 스케일링 적용: $$pb \leftarrow pb \cdot 2 - 1$$

4. 시간 스텝 $$t$$ 무작위 샘플링

5. 가우시안 노이즈 생성: $$\epsilon \sim \mathcal{N}(0, 1)$$

6. 노이징 박스 계산:
$$pb^{\text{crpt}} = \sqrt{\bar{\alpha}_t} pb + \sqrt{1-\bar{\alpha}_t} \epsilon$$

7. 탐지 디코더로 예측: $$pb^{\text{pred}} = \text{DetectionDecoder}(pb^{\text{crpt}}, \text{feats}, t)$$

8. 집합 예측 손실(Set Prediction Loss) 계산 및 역전파

#### 3.3 추론 프로세스

**알고리즘 2: DiffusionDet 샘플링**

입력: 이미지, 스테이 수 $$S$$, 총 시간 스텝 $$T$$

1. 이미지 특성 추출: $$\text{feats} = \text{ImageEncoder}(\text{images})$$

2. 초기 무작위 박스: $$pb_T \sim \mathcal{N}(0, 1)$$

3. 균등 샘플 스텝 크기 설정

4. 각 스텝에서:
   - 현재 박스에서 예측: $$pb_0 = \text{DetectionDecoder}(pb_t, \text{feats}, t_{\text{now}})$$
   - DDIM을 사용하여 다음 시간 스텝의 박스 추정
   - **박스 갱신(Box Renewal)**: 낮은 신뢰도의 박스를 새로운 무작위 박스로 교체

5. 최종 예측 반환

**박스 갱신 전략**:

박스 갱신은 학습 시 박스가 노이징 프로세스를 통해 생성되지만, 추론에서 원하지 않는 박스들은 임의로 분포하는 문제를 해결한다. 신뢰도 임계값 이하의 박스는 제거하고, 원하는 박스들과 새로운 무작위 박스를 연결한다.[1]

#### 3.4 손실 함수

집합 예측 손실(Set Prediction Loss)을 사용하며, 최적 수송(optimal transport)으로 매칭된 예측과 지정 객체 간의 손실을 계산한다:[1]

$$\mathcal{L}_{\text{total}} = \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{L1}} \mathcal{L}_{\text{L1}} + \lambda_{\text{giou}} \mathcal{L}_{\text{giou}}$$

여기서 $$\lambda_{\text{cls}} = 2.0, \lambda_{\text{L1}} = 5.0, \lambda_{\text{giou}} = 2.0$$[1]

***

### 4. 성능 향상 및 실험 결과
#### 4.1 COCO 데이터셋 성능

| Method | Backbone | AP | AP50 | AP75 |
|--------|----------|-----|------|------|
| RetinaNet | ResNet-50 | 38.7 | 58.0 | 41.5 |
| Faster R-CNN | ResNet-50 | 40.2 | 61.0 | 43.8 |
| Sparse R-CNN | ResNet-50 | 45.0 | 63.4 | 48.2 |
| **DiffusionDet (1 step, 300 boxes)** | **ResNet-50** | **45.8** | **64.1** | **50.4** |
| **DiffusionDet (4 steps, 500 boxes)** | **ResNet-50** | **46.8** | **65.3** | **51.8** |

ResNet-50 백본에서 DiffusionDet은 단일 스텝으로 45.8 AP를 달성하여 Sparse R-CNN(45.0 AP)을 능가하고, 4 스텝과 500개 박스로 46.8 AP를 달성한다.[1]

#### 4.2 동적 박스 개수 평가

DiffusionDet의 가장 눈에 띄는 특성 중 하나는 **평가 시 임의의 개수의 박스를 사용할 수 있다**는 점이다:[1]

| Number of Boxes | DETR (AP) | DiffusionDet (AP) |
|-----------------|-----------|-------------------|
| 50 | 31.0 | 38.4 |
| 100 | 34.9 | 38.4 |
| 300 | 38.8 | 45.8 |
| 500 | 36.5 | 46.3 |
| 1000 | 34.0 | 46.7 |
| 2000 | 30.2 | 46.8 |
| 4000 | 26.4 | 46.8 |

DETR은 300개의 쿼리에서 훈련되어 쿼리 개수가 변하면 성능이 저하되지만, DiffusionDet은 박스 개수가 증가해도 지속적으로 성능이 향상되거나 유지된다.[1]

#### 4.3 반복적 평가

| Iteration Steps | 100 boxes (AP) | 300 boxes (AP) | 500 boxes (AP) |
|-----------------|----------------|----------------|----------------|
| 1 | 41.9 | 45.8 | 46.3 |
| 2 | 44.5 | 46.5 | 46.8 |
| 3 | 45.2 | 46.6 | 46.9 |
| 4 | 45.8 | 46.6 | 47.0 |
| 8 | 46.1 | 46.8 | 46.9 |

반복 스텝이 증가함에 따라 성능이 개선되며, 100개 박스의 경우 1단계에서 4단계로 진행하면 4.2 AP 향상을 달성한다.[1]

#### 4.4 제로샷 전이 성능 - 일반화의 핵심

**COCO에서 CrowdHuman으로의 전이**:

| Method | COCO (AP) | CrowdHuman (300 boxes, AP) | CrowdHuman (2000 boxes/4 steps, AP) | Gain |
|--------|-----------|---------------------------|--------------------------------------|------|
| DETR | 61.3 | 61.3 | 61.3 | 0.0 |
| Sparse R-CNN | 66.6 | 66.5 | 66.5 | -0.1 |
| **DiffusionDet** | **66.6** | **69.0** | **71.9** | **5.3** |

이는 DiffusionDet의 가장 중요한 강점이다. CrowdHuman은 COCO보다 혼잡한 장면(평균 22.6명/이미지)을 포함하므로, DiffusionDet은 평가 시 박스 개수와 반복 스텝을 동적으로 조정하여 성능을 크게 향상시킨다.[1]

#### 4.5 LVIS 벤치마크 (장꼬리 분포)

| Method | Backbone | AP | AP_rare | AP_common | AP_freq |
|--------|----------|-----|---------|-----------|---------|
| Sparse R-CNN | ResNet-50 | 29.2 | 20.6 | 27.7 | 34.6 |
| **DiffusionDet (1 step, 1000 boxes)** | **ResNet-50** | **31.4** | **24.5** | **28.8** | **37.3** |
| **DiffusionDet (4 steps, 300 boxes)** | **ResNet-50** | **31.5** | **24.1** | **29.3** | **37.4** |

반복 평가는 COCO에서 0.8 AP 향상을 가져오지만, LVIS에서는 2.1 AP 향상을 달성한다. 이는 더 어려운 벤치마크일수록 DiffusionDet의 이점이 더 크다는 것을 시사한다.[1]

***

### 5. 일반화 성능 향상 분석 (중점)
#### 5.1 일반화 메커니즘

DiffusionDet의 뛰어난 일반화 성능은 다음 요인들에 의해 실현된다:

**1. 확률적 접근 방식**:
확산 모델의 확률적 특성은 다양한 박스 초기화와 점진적 정제를 통해 더 강건한 표현을 학습하게 한다. 고정된 학습 가능한 쿼리와 달리, 무작위 박스에서 시작하므로 훈련-평가 분포 불일치가 감소한다.[1]

**2. 유연한 박스 수 처리**:
동적 박스 개수 평가 메커니즘으로 인해 희소(sparse) 또는 혼잡(crowded) 장면에 자동으로 적응한다. 제로샷 전이에서 COCO(평균 7.7 객체/이미지)에서 CrowdHuman(평균 22.6 객체/이미지)로 전이할 때, 박스 개수를 동적으로 증가시켜 5.3 AP 향상을 달성한다.[1]

**3. 반복적 정제**:
다중 반복 스텝을 통한 점진적 정제는 모델이 예측 오류를 점차 수정하도록 한다. 이는 도메인 시프트 상황에서도 견고하게 작동한다.[1]

**4. 신호 스케일링 최적화**:
신호 스케일 2.0은 박스의 제한된 파라미터(4개)를 고려하여 더 강한 학습 신호를 유지한다. 이는 이미지 생성(신호 스케일 1.0) 또는 세그멘테이션(신호 스케일 0.1)과 다르며, 탐지 태스크에 맞춘 최적화이다.[1]

#### 5.2 통계적 안정성

무작위 박스 초기화로 인한 성능 변동성 분석:[1]
- 5개의 독립적 학습 인스턴스에서 45.7 AP 근처에 밀집된 분포
- 모델 인스턴스 간 성능 차이는 미미
- 10개의 다양한 무작위 시드로 평가해도 신뢰성 있는 결과

이는 DiffusionDet이 무작위 박스 초기화에 견고함을 입증한다.[1]

#### 5.3 도메인 특정 적응

LVIS의 장꼬리 분포에서 DiffusionDet의 우수한 성능:[1]
- 드물게 발생하는 클래스(rare class)에서 AP_rare 24.5 달성 (vs. Sparse R-CNN 20.6)
- 반복적 정제가 장꼬리 분포에 더 효과적
- 단일 모델로 다양한 클래스 분포에 자동 적응

***

### 6. 모델의 한계
#### 6.1 계산 효율성

**추론 속도 트레이드오프**:[1]
- 단일 스텝 (300 박스): 30 FPS - Sparse R-CNN과 유사
- 4 스텝 (300 박스): 20 FPS - 약 33% 속도 저하
- 4 스텝 (1000 박스): 24 FPS - 복잡한 장면에서 실시간성 감소

#### 6.2 기술적 한계

1. **DDIM 및 박스 갱신의 필요성**:
   - DDIM 없이는 반복 스텝이 증가해도 성능 개선 없음
   - 박스 갱신 전략이 추론 복잡도 증가

2. **고급 컴포넌트 부재**:
   - Deformable Attention 등 최신 기술 미적용
   - DINO와 같은 최고 성능 방법과 여전히 성능 차이

3. **신호 스케일 의존성**:
   - 신호 스케일 2.0이 최적값이지만, 다른 도메인에서는 재조정 필요 가능성

#### 6.3 학습 복잡도

- **박스 패딩 전략 필요**: 다양한 길이의 지정 박스 목록을 고정 크기로 패딩
- **최적 수송 할당**: 다대일 매칭으로 계산 복잡도 증가
- **하이퍼파라미터 민감성**: 신호 스케일, 타임스텝 스케줄 등 신중한 튜닝 필요

***

### 7. 최신 연구 기반 향후 전망 및 고려사항
#### 7.1 확산 모델 기반 탐지의 진화 방향

**최신 연구 추세**:[2][3][4][5][6][7][8]

1. **데이터 엔진으로서의 확산 모델**:[2]
   - DiffusionEngine은 확산 모델을 탐지용 합성 데이터 생성 엔진으로 활용
   - COCO에서 3.1% mAP, VOC에서 7.6% mAP 향상
   - **시사점**: DiffusionDet과 결합하면 데이터 부족 시나리오에서 큰 효과

2. **3D 탐지로의 확장**:[4][8][9]
   - 3DifFusionDet: LiDAR-Camera 퓨전의 강건한 확산 기반 3D 탐지
   - DiffRef3D: 포인트 클라우드 기반 3D 탐지에 확산 적용
   - Diff3Det: 무작위 3D 박스로부터 점진적 정제
   - **시사점**: 2D에서 3D로의 자연스러운 확장 가능

3. **특수 탐지 태스크로의 적용**:[3][10][11][12]
   - diffCOD: 위장된 객체 탐지(Camouflaged Object Detection)
   - DiffHOI: 인간-객체 상호작용 탐지
   - DiffusionTrack: 다중 객체 추적(MOT)
   - **시사점**: DiffusionDet 패러다임이 다양한 탐지 변종에 적용 가능

4. **도메인 일반화 강화**:[13][14][15]
   - 최신 논문(2025): "Mining Robust Features from Diffusion Models for Domain-Generalized Detection" - 14.0% mAP 향상[13]
   - 확산 모델의 멀티-스텝 중간 특성을 도메인 불변 표현으로 활용
   - 단일 도메인 일반화(SDG-DiffDet): 메모리 가이드 확산 모듈로 소스-타겟 분포 전이[15]
   - **시사점**: DiffusionDet의 일반화 성능을 더욱 극대화할 가능성

#### 7.2 향후 연구 시 고려할 점

**1. 속도 최적화**:
- Consistency Models나 다른 고속 샘플링 전략 적용[1]
- 적응형 스텝 개수 조정으로 정확도-속도 트레이드오프 최적화
- 고주파 정보 손실을 보완하는 경량화 방법

**2. 아키텍처 개선**:
- Deformable Attention, Wide Detection Head 등 최신 기법 통합
- 대규모 백본(예: Swin-Large)과의 결합
- 마일티-스케일 반복 정제 메커니즘

**3. 데이터 증강 및 합성**:
- DiffusionEngine과 DiffusionDet의 통합: 합성 데이터로 사전학습 후 탐지
- 도메인별 적응형 데이터 생성
- 라벨 부족 시나리오에서의 반자동 라벨링

**4. 이론적 심화**:
- 확산 모델이 객체 탐지 태스크에 왜 더 효과적인지에 대한 이론적 분석
- 신호 스케일과 성능의 관계에 대한 수학적 프레임워크
- 일반화 한계에 대한 엄밀한 분석

**5. 실제 응용 확대**:
- 자율주행, 감시, 의료 영상 등 특정 도메인 최적화
- 저전력 디바이스에서의 경량 버전 개발
- 실시간 성능 요구 시스템에 대한 적응형 스텝 조정

**6. 멀티모달 및 비디오 확장**:
- 텍스트-이미지 조건부 탐지
- 비디오 프레임 간 일관성 유지를 위한 확산 적용
- 시공간 정제 메커니즘

#### 7.3 실무적 함의

1. **제로샷 일반화 성능**: 새로운 도메인에 사전학습 모델을 직접 적용할 때, 기존 방법과 달리 박스 개수와 반복 스텝만 조정하면 추가 재학습 없이 성능 개선 가능[1]

2. **유연한 배포**: 동일한 모델 가중치로 다양한 속도-정확도 트레이드오프를 실현하므로, 단일 모델로 여러 응용에 대응 가능[1]

3. **혼잡 장면 강점**: CrowdHuman의 예시처럼 밀집된 객체 장면에서 성능 향상이 더 크므로, 군중 탐지, 교통 모니터링 등에 특히 유용[1]

***

### 8. 결론
DiffusionDet은 **객체 탐지 문제를 생성 모델 관점에서 재해석**한 혁신적 접근이다. 무작위 박스로부터 점진적 정제를 통해 학습과 평가의 유연성을 달성하고, 특히 **제로샷 도메인 전이에서 탁월한 일반화 성능**을 보여준다.[1]

최근 연구 동향을 보면, 확산 모델 기반 탐지는 단순한 한 가지 방법론을 넘어서 **데이터 생성, 특수 탐지 태스크, 3D 탐지, 도메인 일반화** 등으로 빠르게 확산되고 있다. 특히 2025년 발표된 최신 논문들은 확산 모델의 중간 특성 활용으로 도메인 일반화에서 14% 이상의 성능 향상을 달성하고 있다.[2][3][4][13]

향후 연구는 **계산 효율성 개선**, **최신 아키텍처 컴포넌트 통합**, **이론적 이해 심화**에 초점을 맞추면서, 동시에 **멀티모달 확장**과 **실제 응용 고려**를 통해 DiffusionDet 패러다임을 다양한 분야로 확산시킬 것으로 예상된다.

***

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/91a407ea-de7b-4717-82ae-e17d86caf86e/2211.09788v2.pdf)
[2](https://arxiv.org/abs/2309.03893)
[3](https://arxiv.org/abs/2308.00303)
[4](https://arxiv.org/abs/2312.02966)
[5](https://ieeexplore.ieee.org/document/10435420/)
[6](https://arxiv.org/abs/2307.02270)
[7](https://www.mdpi.com/2079-9292/12/24/4962)
[8](https://arxiv.org/abs/2311.03742)
[9](https://arxiv.org/abs/2309.02049)
[10](https://arxiv.org/abs/2305.12252)
[11](https://arxiv.org/abs/2305.17932)
[12](https://arxiv.org/abs/2308.09905)
[13](https://arxiv.org/abs/2503.02101)
[14](https://arxiv.org/abs/2412.13815)
[15](https://openaccess.thecvf.com/content/ICCV2025/papers/Jiang_Diffusion-based_Source-biased_Model_for_Single_Domain_Generalized_Object_Detection_ICCV_2025_paper.pdf)
[16](https://arxiv.org/pdf/2211.09788.pdf)
[17](http://arxiv.org/pdf/2312.11578.pdf)
[18](https://arxiv.org/pdf/2309.03893.pdf)
[19](http://arxiv.org/pdf/2310.16349.pdf)
[20](https://arxiv.org/html/2502.14891)
[21](https://arxiv.org/abs/2303.09813)
[22](https://arxiv.org/html/2408.12747v1)
[23](https://arxiv.org/abs/2211.09788)
[24](https://arxiv.org/html/2509.13214v1)
[25](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_DiffusionDet_Diffusion_Model_for_Object_Detection_ICCV_2023_paper.pdf)
[26](https://openaccess.thecvf.com/content/CVPR2023W/VAND/papers/Graham_Denoising_Diffusion_Models_for_Out-of-Distribution_Detection_CVPRW_2023_paper.pdf)
[27](https://viplab.snu.ac.kr/viplab/courses/mlvu_2023_1/projects/09.pdf)
[28](https://pmc.ncbi.nlm.nih.gov/articles/PMC11601717/)
[29](https://github.com/ShoufaChen/DiffusionDet)
