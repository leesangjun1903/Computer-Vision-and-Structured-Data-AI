# Parsing R-CNN for Instance-Level Human Analysis

### 1. 핵심 주장 및 주요 기여

**Parsing R-CNN**은 인스턴스 수준의 인간 분석을 위한 통합 엔드-투-엔드 파이프라인으로, 인간 부분 분할(human part segmentation), 밀집 자세 추정(dense pose estimation), 인간-물체 상호작용(human-object interactions) 등의 작업을 효과적으로 처리한다. 이 논문의 핵심 주장은 기존의 **Mask R-CNN** 기반 방법들이 클래스 불특정 마스크(class-agnostic mask)를 예측하도록 설계되어 인스턴스 수준 인간 분석의 세부 정보 캡처에 부족하다는 것이다.[1]

논문의 주요 기여는 다음 네 가지 측면에 집중한다:[1]

1. **제안 분리 샘플링(Proposals Separation Sampling, PSS)**: 피라미드 기반 특징 표현을 유지하면서 고해상도 세부 정보 추출
2. **RoI 해상도 확대(Enlarging RoI Resolution, ERR)**: 32×32 해상도로 확대하여 인간 신체 세부사항 보존
3. **기하학적 문맥 인코딩 모듈(Geometric and Context Encoding, GCE)**: 다중 스케일 수용장 및 인간 신체 부위 간의 기하학적 관계 학습
4. **분석 브랜치 분해(Parsing Branch Decoupling, PBD)**: 브랜치를 세 부분으로 분해하여 효율성과 정확성의 균형 달성

***

### 2. 문제 정의 및 해결 방법

#### 2.1 해결하고자 하는 문제

기존 두 단계 파이프라인인 Mask R-CNN은 물체 감지 및 인스턴스 분할에 효과적이지만, 인스턴스 수준 인간 분석에는 다음과 같은 결함이 있다:[1]

- 클래스 불특정 마스크(class-agnostic mask) 예측으로 인한 세부 정보 손실
- 인간 신체 부위/밀집 포인트 간의 기하학적 및 의미론적 관계 부재
- 대규모 인간 인스턴스 처리 시 기존 FPN 할당 전략의 비효율성

구체적으로, CIHP와 MHP v2.0 데이터셋에서는 각각 74%, 86%의 인간 인스턴스가 이미지의 10% 이상을 차지하지만, COCO 데이터셋에서는 20% 미만이다. 이는 표준 FPN 할당 전략이 인간 인스턴스를 저해상도 특징 맵에 할당하여 안경, 시계 등의 세부사항 인식에 어려움을 초래한다.[1]

#### 2.2 제안 방법

**방법 1: 제안 분리 샘플링(PSS)**

기존 방식은 RoI 크기에 따라 피라미드 레벨을 할당:

$$\text{Level} = \lfloor k + \log_2(\sqrt{wh}/224) \rfloor$$

여기서 \(w, h\)는 RoI의 너비와 높이이고, \(k=4\)는 기준 크기이다.[1]

Parsing R-CNN의 PSS 전략:
- **Bbox 브랜치**: 기존 FPN 할당 전략 유지 (P2-P5)
- **Parsing 브랜치**: 최고 해상도 특징 맵(P2)에서만 RoIAlign 수행[1]

이를 통해 객체 감지의 다중 스케일 표현 이점을 유지하면서 인간 신체의 세부사항을 보존한다.[1]

**방법 2: RoI 해상도 확대(ERR)**

기존 Mask R-CNN: \(14 \times 14\) RoI 크기 사용

Parsing R-CNN: \(32 \times 32\) 또는 \(64 \times 64\) 확대[1]

계산량 증가 문제 해결:
- 인스턴스 수준 작업의 배치 크기를 고정값(예: 32)으로 분리
- 이를 통해 훈련 속도 개선과 정확도 저하 회피[1]

**방법 3: 기하학적 문맥 인코딩(GCE) 모듈**

GCE는 ASPP(Atrous Spatial Pyramid Pooling)와 Non-local 연산을 결합한다:[1]

$$\text{GCE} = \text{ASPP} \oplus \text{Non-local}$$

여기서 \(\oplus\)는 특징 연결을 의미한다.[1]

**ASPP 부분 구성:**
- 1×1 합성곱 1개
- 팽창율 \(\text{rates} = (6, 12, 18)\)인 3×3 팽창 합성곱 3개
- 전역 평균 풀링 후 1×1 합성곱, 원래 32×32 차원으로 쌍선형 업샘플링[1]

**Non-local 부분:**
임베딩 가우시안 버전 사용, 배치 정규화 층 추가[1]

이 모듈은 다양한 척도의 수용장과 인간 신체 부위 간의 기하학적 관계를 학습한다.[1]

**방법 4: 분석 브랜치 분해(PBD)**

Parsing 브랜치를 세 부분으로 분해:[1]

$$\text{Parsing Branch} = \text{Semantic Transformation} \rightarrow \text{GCE Module} \rightarrow \text{Feature Refinement}$$

- **GCE 이전**: 특징을 해당 작업으로 변환
- **GCE 모듈**: 다중 스케일 정보 및 기하학적 관계 인코딩
- **GCE 이후**: 4개의 3×3 512-d 합성곱 층으로 의미론적 특징 정제[1]

***

### 3. 모델 구조

#### 3.1 전체 파이프라인

Parsing R-CNN은 Mask R-CNN 구조를 기반으로 하며, Feature Pyramid Network(FPN) 백본과 RoIAlign 연산을 채택한다:[1]

```
입력 이미지
    ↓
Backbone (FPN) → P2, P3, P4, P5, P6
    ↓
RPN (Region Proposal Network)
    ↓
RoIs
    ├─→ Bbox 브랜치 (표준 R-CNN 구조)
    └─→ Parsing 브랜치
         ├─→ RoIAlign (P2에서만)
         ├─→ Semantic Space Transformation
         ├─→ GCE 모듈
         └─→ Feature Refinement 합성곱
             ↓
         인스턴스 레벨 인간 분석 결과
```

**핵심 구성 요소:**

| 구성 요소 | 설명 | 특징 |
|---------|------|------|
| Backbone | ResNet-50-FPN 또는 ResNeXt-101 | 다중 스케일 특징 추출 |
| RPN | Region Proposal Network | RoI 후보 생성 |
| Bbox 브랜치 | 경계 상자 회귀 및 분류 | 표준 Faster R-CNN |
| Parsing 브랜치 | 인스턴스 수준 인간 분석 | PSS + ERR + GCE + PBD |
| RoIAlign | 관심 영역 정렬 | P2 특징 맵에만 적용 |

#### 3.2 GCE 모듈 상세 구조

GCE 모듈은 다음과 같이 구성된다:[1]

$$\text{Output} = \text{Concat}([\text{ASPP}_1, \text{ASPP}_2, \text{ASPP}_3, \text{Non-local}])$$

각 ASPP 분기:
- 서로 다른 팽창율의 합성곱으로 다중 스케일 정보 캡처
- 각 채널: 256차원

Non-local 분기:
- 장거리 의존성 학습
- 인간 신체 부위 간의 기하학적 관계 인코딩[1]

---

### 4. 성능 향상

#### 4.1 CIHP 데이터셋 결과

| 지표 | Baseline | PSS | ERR | GCE | PBD | +3x LR | +COCO | 향상도 |
|------|---------|-----|-----|-----|-----|--------|-------|--------|
| mIoU | 47.2 | 48.2 | 50.7 | 52.7 | 53.5 | 56.3 | 57.5 | **+10.3** |
| AP_50^p | 41.4 | 42.9 | 47.9 | 53.2 | 58.5 | 63.7 | 65.4 | **+24.0** |
| AP_vol^p | 45.4 | 46.0 | 47.6 | 49.7 | 51.7 | 53.9 | 54.6 | **+9.2** |
| PCP_50 | 44.3 | 45.5 | 49.7 | 52.6 | 56.5 | 60.1 | 62.6 | **+18.3** |

각 컴포넌트의 기여도:
- **PSS**: mIoU +1.0 향상[1]
- **ERR (32×32)**: mIoU +2.8, AP_50^p +5.0 향상[1]
- **GCE**: mIoU +2.0 향상 (ASPP+Non-local 결합의 효과)[1]
- **PBD (GCE 이후 4conv)**: mIoU +0.8, AP_50^p +5.3 향상[1]

#### 4.2 MHP v2.0 데이터셋 결과

| 지표 | Baseline | PSS | ERR | GCE | PBD | +3x LR | +COCO | 향상도 |
|------|---------|-----|-----|-----|-----|--------|-------|--------|
| mIoU | 28.7 | 29.8 | 32.3 | 33.7 | 34.3 | 36.2 | 37.0 | **+8.3** |
| AP_50^p | 10.1 | 10.6 | 14.0 | 17.4 | 20.0 | 24.5 | 26.6 | **+16.5** |

MHP v2.0의 58개 세밀한 의미론적 범주에서도 강력한 향상 달성[1]

#### 4.3 DensePose-COCO 결과

| 지표 | Backbone | Baseline | 최종 결과 | 향상도 |
|------|---------|---------|---------|--------|
| AP | ResNet50 | 48.9 | 58.3 | **+9.4** |
| AP_50 | ResNet50 | 84.9 | 90.1 | **+5.2** |
| AP_75 | ResNet50 | 50.8 | 66.9 | **+16.1** |
| AP | ResNeXt101 | 55.5 | 61.6 | **+6.1** |

**COCO 2018 Challenge 우승**: 64.1% AP 달성으로 2위(58%)보다 6 포인트 차이로 우승[1]

#### 4.4 최신 방법과의 비교

**인간 부분 분할:**
- CIHP: 61.1% mIoU (ResNeXt-101, 테스트 타임 증강) vs PGN 55.8%
- MHP v2.0: 41.8% mIoU vs NAN 25.1%[1]

***

### 5. 한계

논문에서 명시적으로 언급되지 않은 한계들:

1. **계산 비용**: 32×32 RoI 사용으로 인한 메모리 오버헤드 (9.1 fps vs baseline 10.4 fps)[1]

2. **데이터셋 의존성**: CIHP, MHP v2.0 등 특정 인간 분석 데이터셋에 최적화. 다른 도메인 성능 미평가

3. **작은 인스턴스 처리**: 논문에서 "작은 인스턴스는 정확하게 주석 처리되지 않는다"는 가정. 실제로 작은 인간 인스턴스에 대한 성능 저하 가능성[1]

4. **배치 크기 고정**: ERR 관련 배치 크기를 32로 고정하여 훈련 유연성 제약

5. **모델 복잡도**: GCE 모듈과 추가 컴포넌트로 인한 모델 매개변수 증가

***

### 6. 일반화 성능 향상 가능성

#### 6.1 논문 내 일반화 증거

**크로스 데이터셋 평가:**
- COCO 사전훈련을 통한 성능 개선: mIoU +2.4~4.0 (데이터셋 별)[1]
- ResNet50에서 ResNeXt101으로 백본 업그레이드 시 일관된 성능 향상[1]

**반복 횟수 증가의 효과:**
- 1x → 3x 학습 스케줄 전환: CIHP에서 mIoU +2.8 향상[1]

#### 6.2 최신 연구 기반 일반화 가능성

**Vision Transformer 기반 접근:**

최근 **ViTPose++** 등의 연구는 순수 Vision Transformer를 기반으로 100M에서 1B 파라미터까지 확장 가능하며, 뛰어난 일반화 능력을 보여준다. 이는 Parsing R-CNN의 ResNet-FPN 기반 구조를 ViT 기반 백본으로 대체할 경우:[2]

- **도메인 간 전이 학습**: Transformer의 자기 주의(self-attention) 메커니즘이 기하학적 관계 학습에 더 효과적일 가능성
- **확장성**: 더 큰 모델로의 확장을 통한 성능 개선

**약하게 감시되는 학습:**

**WISH** (Weakly Supervised Instance Segmentation) 등 2025년 최신 연구는 약한 라벨만으로 인스턴스 분할을 수행할 수 있음을 보여준다. 이는:[3]

- 주석 비용 감소로 더 많은 데이터 확보 가능
- 다양한 도메인에서의 모델 적응 용이성 향상

**도메인 적응 연구:**

Direct Dense Pose (DDP) 등은 탑-다운 방식의 Mask R-CNN 의존성을 제거하고, 하단-상향 방식의 대안 제시, 이는:[4]

- 탐지 오류 누적 문제 해결
- 여러 인물이 있는 이미지에서 더 견고한 성능

***

### 7. 앞으로의 연구 고려사항

#### 7.1 아키텍처 개선 방향

1. **Transformer 기반 백본 통합**
   - Vision Transformer를 FPN과 결합하여 더 강력한 특징 표현 학습
   - 자기 주의 메커니즘을 통한 효율적인 장거리 의존성 학습[5]

2. **멀티모달 학습**
   - RGB 이외의 추가 채널(깊이, 열화상) 통합
   - 약한 감시 신호와 강한 감시 신호의 혼합 활용[6]

3. **동영상 시퀀스 모델링**
   - 시간적 일관성 강제를 통한 비디오에서의 떨림 감소
   - 인접 프레임 간의 기하학적 관계 활용

#### 7.2 일반화 능력 강화

1. **도메인 적응 기법**
   - 원본 도메인과 목표 도메인 간의 분포 차이 최소화
   - 적대적 학습(adversarial training) 적용

2. **메타 학습**
   - 적은 수의 샘플로 빠른 적응 가능
   - 새로운 신체 형태나 자세에 대한 빠른 학습

3. **자기감시 학습**
   - 레이블 없는 데이터의 효율적 활용
   - 기하학적 불변성과 의미론적 일관성 보존

#### 7.3 응용 분야 확장

1. **인간-물체 상호작용 감지**
   - Parsing R-CNN 프레임워크를 직접 확장 가능
   - 인간 부위와 물체 간의 공간 관계 모델링

2. **3D 인간 메시 복원**
   - 밀집 자세 추정 결과를 3D 메시 복원에 활용
   - SMPL 모델과의 통합

3. **세밀한 신체 움직임 인식**
   - 손가락 움직임, 얼굴 표정 등의 세부 분석
   - 고해상도 예측을 통한 미묘한 움직임 포착

#### 7.4 실용적 고려사항

1. **계산 효율성**
   - 모바일/엣지 기기를 위한 경량화
   - 지식 증류(knowledge distillation) 활용
   - 동적 배치 크기 조정

2. **데이터 효율성**
   - 몇 샷 학습(few-shot learning) 연구
   - 합성 데이터 활용도 개선

3. **공정성과 편향**
   - 다양한 신체 타입, 피부색, 의류에 대한 성능 평가
   - 편향 감소 기법 적용

---

### 8. 결론

**Parsing R-CNN**은 인스턴스 수준 인간 분석을 위한 효과적이고 유연한 통합 파이프라인을 제시한다. 제안 분리 샘플링, RoI 해상도 확대, 기하학적 문맥 인코딩, 분석 브랜치 분해 등의 혁신적 기법들을 통해 기존 Mask R-CNN 대비 10~24 포인트의 성능 향상을 달성했다. 특히 COCO 2018 Challenge에서 우승을 차지하며 그 실효성을 입증했다.[1]

향후 연구에서는 **Vision Transformer 기반 백본 통합**, **약하게 감시되는 학습**, **도메인 적응 기법** 등을 통해 모델의 일반화 능력을 더욱 강화할 수 있을 것으로 예상된다. 또한 계산 효율성 개선과 실제 응용 분야 확대를 위한 연구가 지속적으로 필요하다.[2][3][5]

***

### 참고 문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/26519ff1-0db4-4fa9-900e-521c5466ea31/1811.12596v1.pdf)
[2](https://arxiv.org/pdf/2212.04246.pdf)
[3](https://openaccess.thecvf.com/content/CVPR2025/papers/Kweon_WISH_Weakly_Supervised_Instance_Segmentation_using_Heterogeneous_Labels_CVPR_2025_paper.pdf)
[4](https://arxiv.org/abs/2204.01263)
[5](https://openreview.net/pdf?id=6H2pBoPtm0s)
[6](https://arxiv.org/abs/2312.06988)
[7](http://arxiv.org/pdf/1811.12596.pdf)
[8](https://arxiv.org/pdf/2107.12889.pdf)
[9](http://arxiv.org/pdf/1907.06713.pdf)
[10](https://www.mdpi.com/2072-4292/13/1/39/pdf)
[11](http://arxiv.org/pdf/2404.16633.pdf)
[12](http://arxiv.org/pdf/1711.10370.pdf)
[13](https://arxiv.org/pdf/2305.01910.pdf)
[14](http://arxiv.org/pdf/1703.06870.pdf)
[15](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_Parsing_R-CNN_for_Instance-Level_Human_Analysis_CVPR_2019_paper.pdf)
[16](https://arxiv.org/abs/1811.12596)
[17](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570409.pdf)
[18](https://openaccess.thecvf.com/content_cvpr_2018/papers/Guler_DensePose_Dense_Human_CVPR_2018_paper.pdf)
[19](https://dl.acm.org/doi/10.1109/TMM.2023.3260631)
[20](http://ieeexplore.ieee.org/document/8953214/footnotes)
[21](https://ai.meta.com/research/publications/densepose-dense-human-pose-estimation-in-the-wild/)
[22](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cvi2.12222)
[23](https://www.sciencedirect.com/science/article/pii/S1361841524001683)
[24](https://arxiv.org/pdf/2205.05277.pdf)
[25](https://arxiv.org/pdf/2208.10211.pdf)
[26](https://arxiv.org/abs/2103.15320)
[27](https://arxiv.org/pdf/2311.13615.pdf)
[28](http://arxiv.org/pdf/2312.08344.pdf)
[29](https://arxiv.org/pdf/2201.07384.pdf)
[30](https://arxiv.org/pdf/2205.15448.pdf)
[31](https://debuggercafe.com/vitpose/)
[32](https://aclanthology.org/P19-1031/)
[33](https://aclanthology.org/2022.emnlp-main.749/)
[34](https://papers.nips.cc/paper_files/paper/2022/hash/fbb10d319d44f8c3b4720873e4177c65-Abstract-Conference.html)
[35](https://dl.acm.org/doi/10.1145/3581783.3612243)
[36](https://arxiv.org/abs/2210.12445)
[37](https://arxiv.org/abs/2212.04246)
