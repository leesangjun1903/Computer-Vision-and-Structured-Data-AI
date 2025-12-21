# YOLACT Real-time Instance Segmentation

### 1. 논문의 핵심 주장과 주요 기여

YOLACT(You Only Look At CoefficienTs)는 2019년 발표된 논문으로, **최초의 실시간(>30fps) 인스턴스 분할 알고리즘**을 제시합니다. 기존의 Mask R-CNN(8.6fps)과 FCIS(6.6fps)는 정확도에 초점을 맞춰 속도가 느렸던 반면, YOLACT는 병렬 구조와 경량 조립 과정을 통해 33.5fps에서 29.8 mAP를 달성합니다.[1]

논문의 핵심 기여는 다음과 같습니다.

**주요 기여:**
- 인스턴스 분할을 두 개의 독립적 병렬 작업으로 분해함으로써 구조적 단순성 확보[1]
- ROI pooling/align을 제거한 완전 컨볼루션 구조[1]
- 프로토타입 마스크의 emergent 행동 분석[1]
- Fast NMS 제안(12ms 속도 개선)[1]

***

### 2. 해결하고자 하는 문제와 제안 방법

#### 2.1 문제 정의

인스턴스 분할은 객체 검출과 의미론적 분할을 결합해야 하는 복잡한 작업입니다. 기존 방법들의 주요 문제:[1]

1. **두 단계 방법의 순차성**: Mask R-CNN은 ROI 추출 → 특성 국소화 → 마스크 예측의 순차적 처리로 인해 병렬화 어려움[1]
2. **특성 국소화의 의존성**: ROI pooling은 각 인스턴스마다 수행되어 계산량 증가[1]
3. **완전 컨볼루션 네트워크의 한계**: FCN은 translation invariant 특성으로 인해 위치 정보 학습 어려움[1]

#### 2.2 제안하는 방법: YOLACT 구조

YOLACT는 다음과 같이 구성됩니다.

**$$M = \sigma(P C^T)$$**

여기서:
- P: h × w × k 프로토타입 마스크 행렬 (ProtoNet의 출력)
- C: n × k 마스크 계수 행렬 (각 앵커의 k개 계수)
- σ: 시그모이드 활성화 함수
- n: NMS 후 남은 인스턴스 개수

**핵심 설계:**

**1) Prototype Generation (ProtoNet)**

프로토타입은 FCN으로 생성되며, 다음 특성을 가집니다:

$$P_{output} = \text{ReLU}(\text{FCN}(F))$$

- FPN의 P3(가장 깊고 큰 특성맵) 사용[1]
- 입력의 1/4 해상도로 업샘플링하여 138×138 해상도 달성[1]
- 명시적 손실 함수 없이, 최종 마스크 손실을 통해 감독[1]
- Unbounded output (ReLU 사용): 네트워크가 극단값 생성 가능[1]

**2) Mask Coefficient Prediction**

기존 object detection head를 확장:

$$C = \text{tanh}(\text{Head}(F))$$

- 표준 분류 분기와 상자 회귀 분기에 추가 분기 추가[1]
- k개의 마스크 계수 예측 (각 프로토타입에 대응)[1]
- tanh 활성화: 프로토타입의 뺄셈을 가능하게 함[1]

**3) Mask Assembly와 손실함수**

세 가지 손실을 가중치와 함께 사용:

$$L_{total} = L_{cls} + 1.5 L_{box} + 6.125 L_{mask}$$

여기서:

$$L_{mask} = \text{BCE}(M, M_{gt})$$

- M: 예측된 마스크 (선형 결합 후 시그모이드)
- M_gt: 지면 진실 마스크
- 학습 중 지면 진실 바운딩박스로 크롭, 손실을 박스 면적으로 정규화[1]

***

### 3. 모델 구조 및 성능 분석

#### 3.1 전체 아키텍처

[1]

YOLACT는 RetinaNet을 기반으로 다음 구성요소로 이루어집니다:[1]

| 구성요소 | 상세사항 |
|---------|---------|
| **Backbone** | ResNet-101 + FPN |
| **Feature Levels** | P3-P7 (다중 스케일) |
| **ProtoNet** | FCN 기반, 138×138 해상도 |
| **Detection Head** | 3×3 conv (공유) + 병렬 분기 |
| **Mask Assembly** | 행렬 곱셈 + 시그모이드 |
| **NMS** | Fast NMS (12ms 개선) |

#### 3.2 성능 비교

MS COCO 테스트셋 결과:[1]

| 방법 | Backbone | FPS | mAP | AP50 | AP75 |
|-----|----------|-----|-----|------|------|
| Mask R-CNN | R-101-FPN | 8.6 | 35.7 | 58.0 | 37.8 |
| FCIS | R-101-C5 | 6.6 | 29.5 | 51.5 | 30.2 |
| **YOLACT-550** | **R-101-FPN** | **33.5** | **29.8** | **48.5** | **31.2** |
| YOLACT-700 | R-101-FPN | 23.4 | 31.2 | 50.6 | 32.8 |

**성능의 특징:**

- **큰 객체에서 우수**: AP_L = 47.7 (Mask R-CNN: 52.4)
- **마스크 품질 우수**: AP95에서 1.6 (Mask R-CNN: 1.3) - repooling 제거의 효과[1]
- **작은 객체에서 약함**: AP_S = 9.9 (해상도 제약)[1]

#### 3.3 Prototype의 Emergent Behavior

YOLACT의 흥미로운 발견은 프로토타입이 명시적 지도 없이도 다양한 패턴을 학습합니다:[1]

1. **공간 분할 프로토타입**: 이미지의 일부 영역만 활성화 (예: 좌측/우측)
2. **객체 경계 검출**: 객체 간 경계선 활성화
3. **방향 맵**: FCIS의 position-sensitive 맵과 유사한 패턴
4. **배경/전경 분리**: 배경 영역의 선택적 활성화

이는 ResNet의 패딩이 translation variance를 제공하기 때문입니다.[1]

***

### 4. 모델의 일반화 성능과 한계

#### 4.1 일반화 강점

**1) 구조적 단순성**
- 선형 결합만으로 마스크 생성: 다양한 백본 적용 용이[1]
- 단일 GPU에서 학습 가능[1]

**2) 시간적 안정성**
- Repooling 제거로 프레임 간 일관성 높음
- 비디오 추적에서 자연스러운 마스크 변화[1]

**3) 마스크 품질**
- 해상도 손실 없음: 138×138 프로토타입
- 큰 객체의 경계 정확도 우수[1]

**4) 빠른 추론**
- 마스크 분기: ~5ms 추가 계산[1]
- 한 번의 GPU 행렬 곱셈으로 처리

#### 4.2 일반화 한계 및 오류 분석

**1) 국소화 실패 (Localization Failure)**

복잡한 장면에서 여러 객체가 밀집되면, 네트워크가 각 객체를 별도 프로토타입으로 분리하지 못함. 예: 비행기 아래 트럭이 제대로 분할되지 않음.[1]

**원인**: 프로토타입 개수의 제한 및 선형 결합의 표현력 한계

**2) Leakage 현상**

바운딩박스 외부의 노이즈가 크롭 후에도 마스크에 포함됨.[1]

**원인**: 
- 부정확한 바운딩박스 예측
- 네트워크가 먼 객체를 분리할 필요가 없다고 학습

**3) 검출 성능 의존성**

전체 성능의 대부분은 객체 검출 정확도에 의존:[1]

- 마스크 mAP: 29.8
- 박스 mAP: 32.3
- 차이: 2.5만으로 완벽한 마스크로도 몇 포인트만 개선

이는 Mask R-CNN과 동일한 패턴 (35.7 vs 38.2)이며, YOLACT의 마스크 생성 방식보다 검출기의 품질이 병목임을 의미합니다.

**4) 프로토타입 개수의 트레이드오프**

k값에 따른 성능:[1]

| k | AP | FPS |
|---|-----|------|
| 8 | 26.8 | 33.0 |
| 32 | 27.7 | 32.4 |
| 64 | 27.8 | 31.7 |
| 128 | 27.6 | 31.5 |
| 256 | 27.7 | 29.8 |

- k=32 최적점: 성능과 속도의 균형
- k 증가 시 성능 정체: 계수 예측의 난이도 증가[1]

***

### 5. 2020년 이후 관련 최신 연구 비교 분석

#### 5.1 주요 후속 연구들

**A. SOLOv2 (2020, NeurIPS)**[2]

| 항목 | YOLACT | SOLOv2 |
|------|--------|--------|
| 방식 | Prototype 기반 | 위치 기반 (grid) |
| 구조 | 원스테이지 | 원스테이지 |
| 특징 | 선형 결합 | Dynamic kernel |
| 성능 | 31.2 AP (700) | 37.1 AP |
| 속도 | 23.4 fps | 31.3 fps |

SOLOv2는 마스크 분기를 kernel 학습과 feature 학습으로 분리하여 더 나은 성능 달성.[2]

**B. CondInst (2020, ECCV)**[3]

| 항목 | YOLACT | CondInst |
|------|--------|----------|
| 방식 | Prototype 기반 선형 결합 | 조건부 동적 컨볼루션 |
| ROI 연산 | 제거 | 제거 |
| 마스크 헤드 | 고정 가중치 | 동적 가중치 |
| 성능 | 29.8 AP | ~40 AP |

CondInst는 인스턴스별 동적 필터로 마스크를 생성하여 더 강력한 표현력 제공.[4][3]

**C. Mask2Former (2021-2022, CVPR)**[5][6]

| 항목 | YOLACT | Mask2Former |
|------|--------|------------|
| 아키텍처 | CNN 기반 | Transformer 기반 |
| 작업 | 인스턴스 분할 | 범용 (panoptic/instance/semantic) |
| 성능 | 29.8 AP | 50.1 AP (COCO) |
| 속도 | 33.5 fps | ~15 fps |

Mask2Former는 Transformer의 강력한 표현 능력으로 SOTA 달성하되, 속도는 YOLACT보다 느림.[7][5]

**D. YOLACT++ (2020)**[8][9]

YOLACT의 직접적 후속으로, 다음 개선사항 포함:

1. **Deformable Convolution (DCN)**
   - C3~C5 레이어의 3×3 conv를 DCN으로 교체
   - 마스크 mAP: +1.8 개선[9][8]
   - 최적화: 간격을 두고 11개 레이어만 적용하여 속도 오버헤드 최소화

2. **개선된 앵커 스케일과 종횡비**
   - 더 나은 다중 스케일 표현

3. **Fast Mask Re-scoring 분기**
   - 마스크 신뢰도 개선

**최종 성과**: 34.1 mAP @ 33.5 fps (+5.3 mAP)[10][8]

**E. FastSAM (2023)**[11][12]

| 항목 | YOLACT | FastSAM |
|------|--------|---------|
| 기반 | 맞춤 설계 | SAM + YOLOv8-seg + YOLACT |
| 방식 | 직접 분할 | 두 단계: all-instance → prompt |
| 학습 데이터 | COCO (전체) | SA-1B (2% 샘플) |
| 일반화 | COCO 최적화 | 제로샷 능력 |
| 속도 | 33.5 fps | SAM 대비 50-170배 빠름 |

FastSAM은 YOLACT의 아이디어를 기초 모델과 결합하여 zero-shot 능력 획득.[12][13][11]

**F. FastInst (2023, CVPR)**[14]

| 항목 | YOLACT | FastInst |
|------|--------|----------|
| 아키텍처 | CNN + 선형 결합 | Transformer 기반 쿼리 |
| 기반 | RetinaNet | Mask2Former |
| 성능 | 29.8 AP | 40.5 AP |
| 속도 | 33.5 fps | 32.5 fps |

FastInst는 Transformer 기반 query 방식으로 더 높은 정확도 달성.[15][14]

#### 5.2 기술 패러다임의 진화

**2019년 (YOLACT)**: 
- Prototype 기반 선형 결합으로 실시간성 확보
- 원스테이지 구조의 새로운 패러다임[1]

**2020년 (SOLOv2, CondInst)**:
- 동적 네트워크 등장 (Dynamic kernel, Conditional conv)
- 더 강력한 표현력 추구
- 속도-정확도 트레이드오프 개선[3][2]

**2021-2022년 (Mask2Former)**:
- Transformer 기반 아키텍처로의 전환
- 범용 프레임워크 (panoptic/instance/semantic)
- 정확도 극대화[6][5]

**2023-2024년 (FastSAM, FastInst, etc.)**:
- 기초 모델(Foundation Models) 활용
- Query 기반 방식 주류화
- 경량화 및 실시간성 재강조[16][14]

***

### 6. 일반화 성능 향상 가능성

#### 6.1 도메인 변화에 대한 강건성

**현재 한계:**

YOLACT는 MS COCO에서 학습된 가중치가 다른 도메인으로 전이될 때 성능 저하가 큼. 특히:[1]

1. **의료 이미지**: Breast MRI 데이터셋에서 수정된 YOLACT 모델 98.5% 정확도 달성 (일반적 샘플)[17]
2. **수중 환경**: 0.377 mAP (TrashCAN 데이터셋)[18]
3. **산업 응용**: 개선 필요[1]

**개선 방안:**

1. **Multi-scale 학습**
   - 다양한 해상도로 학습하여 작은 객체 성능 개선
   - 현재: 단일 550×550 해상도

2. **Data Augmentation 강화**
   - Copy-Paste augmentation: +0.6 mAP[19]
   - 드문 범주(long-tail) 처리

3. **적응형 학습 (Domain Adaptation)**
   - Unsupervised 학습: FreeSOLO 방식 적용 가능[20]
   - 테스트 타임 적응

#### 6.2 구조 개선을 통한 일반화

**1) Prototype 개수의 적응적 선택**

현재: 고정 k=32

개선 방향:
- 이미지 내용에 따른 동적 prototype 개수
- 복잡도가 높은 이미지에서 k 증가

**2) Hierarchical Prototype**

프로토타입을 의미론적 계층으로 조직화:
- 저수준: 경계선, 텍스처
- 중수준: 부분 형태
- 고수준: 의미론적 특성

**3) 앙상블 기법**

여러 prototype assembly 방식 결합:
- 선형 결합 + 비선형 변환
- Attention-weighted assembly

***

### 7. 앞으로의 연구에 미치는 영향과 고려사항

#### 7.1 학문적 영향

**1) 실시간 분할의 새로운 방향 제시**

YOLACT 이전: 정확도 vs 속도의 근본적 트레이드오프 존재  
YOLACT 이후: **구조적 혁신으로 속도 문제 해결 가능함을 증명**

→ 이후 연구들(FastSAM, FastInst)이 이 패러다임을 기반으로 발전[14][11]

**2) Prototype-based Assembly의 가치**

- 간단함 (Occam's Razor): 선형 결합만으로 충분
- 효율성: O(n) 시간에 n개 인스턴스 생성
- 해석 가능성: 프로토타입의 역할 분석 가능[1]

**3) 다양한 응용 분야 개척**

- 의료 영상: 유방암 진단[17]
- 수중 환경: 해양 쓰레기 감지[18]
- 산업 검사: 콘크리트 균열, 댐 안전[21][14]
- 자율주행: 실시간 객체 분할[22]

#### 7.2 향후 연구 시 고려할 점

**1) 성능-속도-메모리 삼각형**

```
         Accuracy
           /  \
          /    \
         /      \
    Speed ---- Memory
```

YOLACT는 속도에 최적화. 향후 연구:
- Pareto 최적점 탐색 (NAS 활용)[23][24]
- 하드웨어별 최적화 (엣지 디바이스, TPU 등)

**2) 일반화 능력 강화**

- **도메인 일반화 (Domain Generalization)**
  - 학습 데이터와 다른 도메인에서의 성능
  - 현재 YOLACT의 주요 약점

- **Few-shot/Zero-shot 학습**
  - 주석 데이터 부족 상황
  - FastSAM의 성공 사례[11]

**3) 설명 가능성 (Interpretability)**

YOLACT의 프로토타입이 자연스럽게 해석 가능:
- 어떤 프로토타입이 어떤 객체에 기여하는지 분석
- 오류 분석 및 디버깅 용이
→ 규제 산업(의료, 자동차)에서 중요[17]

**4) 멀티태스크 확장**

현재: 인스턴스 분할만  
향후:
- Panoptic 분할 (stuff + thing)
- 3D 인스턴스 분할
- 영상 인스턴스 분할 (temporal consistency)[25]

**5) 경량화 및 양자화**

- 모바일 배포: 프로토타입 크기 축소
- 정수 양자화: ONNX, TensorRT 지원[26]
- Knowledge distillation: 경량 student 모델 학습

**6) 동적 환경 적응**

- 온라인 학습: 지속적 분포 변화에 대응
- Continual learning: 재앙적 망각 방지[27]
- 분포 외(OOD) 감지: 신뢰도 기반 선택적 처리

***

### 결론

YOLACT는 **원스테이지 구조와 prototype 기반 assembly라는 간단하지만 효과적인 아이디어로 실시간 인스턴스 분할의 새로운 방향을 제시**했습니다.[1]

**강점:**
- 33.5 fps의 획기적 속도 달성
- 구조의 단순성으로 인한 확장성
- 이후 연구의 기초 제공

**한계:**
- 정확도 (29.8 mAP)는 Mask R-CNN(35.7 mAP)보다 낮음
- 소객체 성능 약함
- 도메인 일반화 부족

**향후 방향:**
2020년 이후 SOLOv2, CondInst, Mask2Former, FastSAM 등의 연구들이 YOLACT의 패러다임을 기반으로 발전했으며, **쿼리 기반 Transformer 아키텍처(FastInst)가 실시간성과 정확도를 동시에 확보하는 새로운 표준**이 되고 있습니다.[14]

향후 연구의 초점은 (1) 도메인 일반화, (2) 멀티태스크 확장, (3) 경량화, (4) 기초 모델 활용에 있을 것으로 예상됩니다.[16][1]

***

### 참고문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2716dbc6-9721-48cc-8ffe-785721fceec8/1904.02689v2.pdf)
[2](https://www.semanticscholar.org/paper/fab853583c8465c01e2b8244debaa2bcd6be18d6)
[3](http://arxiv.org/abs/2003.05664)
[4](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460273.pdf)
[5](https://arxiv.org/abs/2112.01527)
[6](https://huggingface.co/docs/transformers/model_doc/mask2former)
[7](https://arxiv.org/pdf/2112.01527.pdf)
[8](https://www.cnblogs.com/VincentLee/p/12843489.html)
[9](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/yolact++/)
[10](https://3dvar.com/Bolya2022YOLACT.pdf)
[11](https://docs.ultralytics.com/models/fast-sam/)
[12](https://www.artificialmind.io/fastsam-vs-sam)
[13](https://arxiv.org/html/2507.15008v1)
[14](https://ieeexplore.ieee.org/document/10204244/)
[15](https://openaccess.thecvf.com/content/CVPR2023/papers/He_FastInst_A_Simple_Query-Based_Model_for_Real-Time_Instance_Segmentation_CVPR_2023_paper.pdf)
[16](https://www.ikomia.ai/blog/top-instance-segmentation-models)
[17](https://pmc.ncbi.nlm.nih.gov/articles/PMC10177566/)
[18](https://www.mdpi.com/2077-1312/11/8/1532)
[19](https://ieeexplore.ieee.org/document/9578639/)
[20](https://arxiv.org/pdf/2202.12181.pdf)
[21](https://ieeexplore.ieee.org/document/10705005/)
[22](https://linkinghub.elsevier.com/retrieve/pii/S1474034622002774)
[23](https://ieeexplore.ieee.org/document/10446079/)
[24](https://arxiv.org/html/2511.09554v1)
[25](https://ieeexplore.ieee.org/document/9607521/)
[26](https://tech-stack.com/blog/instance-segmentation-computer-vision/)
[27](https://arxiv.org/html/2512.08569)
[28](https://www.medscimonit.com/abstract/index/idArt/938872)
[29](https://ieeexplore.ieee.org/document/9879517/)
[30](https://www.imemo.ru/en/index.php?page_id=1650&article_id=10968)
[31](https://www.mdpi.com/2076-393X/10/9/1461)
[32](http://www.csam.or.kr/journal/view.html?doi=10.29220/CSAM.2022.29.5.499)
[33](https://clinlabdia.ru/article/analiz-otechestvennogo-rynka-naborov/)
[34](https://ijareeie.com/upload/2022/april/4_Machine.pdf)
[35](https://f1000research.com/articles/11-127/v4)
[36](https://f1000research.com/articles/11-127/v3)
[37](http://arxiv.org/abs/2203.12827)
[38](http://arxiv.org/pdf/1904.02689v2.pdf)
[39](https://www.mdpi.com/1424-8220/23/14/6446)
[40](https://arxiv.org/pdf/2202.07402.pdf)
[41](http://arxiv.org/pdf/1905.11358.pdf)
[42](https://arxiv.org/pdf/2205.12646.pdf)
[43](http://arxiv.org/pdf/1704.02386.pdf)
[44](https://arxiv.org/pdf/2311.06659.pdf)
[45](https://www.sciltp.com/journals/ijndi/articles/2504000070)
[46](https://www.sciencedirect.com/science/article/abs/pii/S0045790621004225)
[47](https://arxiv.org/html/2506.14096v2)
[48](https://www.koreascience.kr/article/JAKO202430943205221.page)
[49](https://openaccess.thecvf.com/content_ICCV_2019/papers/Bolya_YOLACT_Real-Time_Instance_Segmentation_ICCV_2019_paper.pdf)
[50](https://www.sciencedirect.com/science/article/pii/S131915782400315X)
[51](https://dl.acm.org/doi/10.1007/s00371-022-02537-8)
[52](https://arxiv.org/html/2512.11884v1)
[53](https://arxiv.org/pdf/2510.13590.pdf)
[54](https://arxiv.org/html/2501.17688v3)
[55](https://arxiv.org/html/2410.04960v1)
[56](https://arxiv.org/html/2508.00737v2)
[57](https://arxiv.org/html/2509.05144v1)
[58](https://arxiv.org/html/2410.04960v4)
[59](https://arxiv.org/html/2511.22606v1)
[60](https://arxiv.org/html/2510.15026v1)
[61](https://www.sciencedirect.com/science/article/abs/pii/S0952197625009583)
[62](https://www.semanticscholar.org/paper/d73795d03114e3ff80c1b42e9f7a1bb95872bea9)
[63](https://www.repository.cam.ac.uk/handle/1810/316307)
[64](https://www.semanticscholar.org/paper/9d546bbfcde0bfb180548884772ed8ec5d683822)
[65](https://www.semanticscholar.org/paper/2ef0310dfcad321c912f24bd5766c9cfe63e5c7c)
[66](https://www.semanticscholar.org/paper/3eee75f022095e375e57db3d77143b130dc9b10b)
[67](https://www.semanticscholar.org/paper/a1edaa77a72d29a09007c35144a33d47b66642b7)
[68](https://www.semanticscholar.org/paper/edd5c28df8bf1fafadaa912df846e2ae6d9b0805)
[69](https://www.semanticscholar.org/paper/28f14defa2910bae7f155024b17c7be4090540b3)
[70](https://arxiv.org/abs/2106.15947)
[71](https://arxiv.org/pdf/2112.11037.pdf)
[72](https://arxiv.org/pdf/1909.13226.pdf)
[73](https://arxiv.org/html/2406.18558v2)
[74](https://www.youtube.com/watch?v=L69fw3s63HU)
[75](https://proceedings.neurips.cc/paper/2020/file/cd3afef9b8b89558cd56638c3631868a-Review.html)
[76](https://huggingface.co/docs/transformers/en/model_doc/mask2former)
[77](https://github.com/WXinlong/SOLO)
[78](https://arxiv.org/abs/2003.05664)
[79](https://arxiv.org/abs/2003.10152)
[80](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_FreeSOLO_Learning_To_Segment_Objects_Without_Annotations_CVPR_2022_paper.pdf)
[81](https://openaccess.thecvf.com/content/CVPR2021/papers/Tian_BoxInst_High-Performance_Instance_Segmentation_With_Box_Annotations_CVPR_2021_paper.pdf)
[82](https://openaccess.thecvf.com/content/CVPR2023/papers/Ishtiak_Exemplar-FreeSOLO_Enhancing_Unsupervised_Instance_Segmentation_With_Exemplars_CVPR_2023_paper.pdf)
[83](https://openaccess.thecvf.com/content/CVPR2021/papers/He_DyCo3D_Robust_Instance_Segmentation_of_3D_Point_Clouds_Through_Dynamic_CVPR_2021_paper.pdf)
[84](https://arxiv.org/abs/1912.04488)
[85](https://openaccess.thecvf.com/content/CVPR2022/papers/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.pdf)
[86](https://ieeexplore.ieee.org/document/10588831/)
[87](https://link.springer.com/10.1007/s00530-023-01212-9)
[88](https://link.springer.com/10.1007/s10586-024-04373-y)
[89](https://dl.acm.org/doi/10.1145/3653804.3656278)
[90](https://www.mdpi.com/2076-3417/14/5/1999)
[91](https://arxiv.org/html/2405.13518)
[92](http://arxiv.org/pdf/1511.08250.pdf)
[93](https://www.sciencedirect.com/science/article/abs/pii/S0923596525001195)
[94](https://www.reddit.com/r/computervision/comments/16vpo0z/which_real_time_instance_segmentation_to_use_in/)
[95](https://www.basic.ai/blog-post/instance-segmentation-in-2024)
[96](https://www.semanticscholar.org/paper/Deep-Learning-Based-Modified-YOLACT-Algorithm-on-of-Wang-Wang/d75a99d0c42a313319df14978ef9bc1690148816)
[97](https://arxiv.org/html/2311.15707v2)
[98](https://arxiv.org/abs/1912.06218)
[99](https://arxiv.org/abs/2507.15008)
[100](https://arxiv.org/html/2403.03296v1)
[101](https://arxiv.org/html/2109.01123v3)
[102](https://arxiv.org/abs/2407.12658)
[103](https://arxiv.org/html/2501.17688v1)
