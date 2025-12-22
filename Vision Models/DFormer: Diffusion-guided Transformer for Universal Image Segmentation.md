# DFormer: Diffusion-guided Transformer for Universal Image Segmentation

---

### I. 핵심 주장 및 기여 요약
DFormer는 **확산 모델(Diffusion Model)을 활용한 통합 이미지 세그멘테이션(Universal Image Segmentation)** 프레임워크를 제안한다. 기존 세그멘테이션 방법들이 각 작업마다 특화된 아키텍처를 요구하는 반면, DFormer는 단일 신경망으로 의미론적(semantic), 인스턴스(instance), 파노프틱(panoptic) 세 가지 세그멘테이션 작업을 동시에 처리한다.

주요 혁신은 세그멘테이션을 **노이즈가 있는 마스크로부터의 점진적 노이즈 제거 프로세스**로 재정의하는 것이다. 이를 통해 DFormer는 최신 확산 기반 방법 대비 현저한 성능 향상을 달성한다:[1][2]

- **파노프틱 세그멘테이션**: Pix2Seq-D 대비 **+3.6% PQ** (51.1% vs 47.5% on MS COCO)
- **인스턴스 세그멘테이션**: DiffusionInst 대비 **+5.3% AP** (42.6% vs 37.3%)
- **의미론적 세그멘테이션**: DDP 대비 **+2.2% mIoU** (48.3% vs 46.1% on ADE20K)

***

### II. 해결하는 문제 및 방법론
#### A. 근본적 문제

기존 세그멘테이션 방법의 한계는 **작업 특화성**에 있다. 각 세그멘테이션 유형은 서로 다른 목표를 가지고 있다:[1]

- **의미론적 세그멘테이션**: 픽셀을 의미론적 카테고리로 분류 (FCN 계열)
- **인스턴스 세그멘테이션**: 개별 객체 인스턴스 분리 (SOLO/QueryInst)
- **파노프틱 세그멘테이션**: 객체(things) 인스턴스 + 영역(stuff) 의미론적 분할 (Panoptic FPN)

이러한 특화성으로 인해 동일 이미지에 대해 여러 모델을 순차적으로 실행해야 하는 비효율성이 발생한다.

#### B. 핵심 수식 및 이론

**1. 확산 모델 기초**

확산 과정은 원본 데이터 $x_0$에 점진적으로 가우시안 노이즈를 추가하는 마르코프 연쇄로 정의된다:[1]

$$q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$$

여기서 $\bar{\alpha}_t$는 1에서 0으로 단조 감소하는 누적 계수이며, 타임스텝 $t$가 증가할수록 노이즈 수준이 높아진다. 역확산 프로세스는 학습된 네트워크 $f(x_t, t)$가 원본 샘플 $x_0$를 예측하도록 최적화된다:[1]

$$L_{diff} = \frac{1}{2}||f(x_t, t) - x_0||^2$$

**2. 마스크 특성 집계**

DFormer의 핵심 설계는 노이즈가 있는 마스크 $M_n$을 이용하여 픽셀 임베딩 $F_{pixel}$을 동적으로 가중한다:[1]

$$F_m = f_{avg}(F_{pixel} \times f_{norm}(M_n))$$

이 수식에서 $f_{norm}$은 정규화 연산이며, $f_{avg}$는 전역 평균 풀링으로, 각 마스크 특성이 해당 마스크 영역의 대표 특성 벡터가 되도록 한다.

**3. 통합 손실 함수**

DFormer는 Hungarian 알고리즘을 기반으로 한 이분 매칭을 통해 예측과 실제값을 매칭한 후, 다음 손실을 최소화한다:[1]

$$L = \lambda_1 L_{cls} + \lambda_2 L_{ce} + \lambda_3 L_{dice}$$

여기서:
- $L_{cls}$: 분류 손실 (cross entropy)
- $L_{ce}$: 이진 마스크 손실 (binary cross-entropy)  
- $L_{dice}$: Dice 손실 (마스크 겹침도)
- 손실 가중치: $\lambda_1=2.0$, $\lambda_2=5.0$, $\lambda_3=5.0$

#### C. 모델 아키텍처

**1. 픽셀 레벨 모듈**

입력 이미지 $x \in \mathbb{R}^{H \times W \times 3}$에서 CNN 백본(ResNet-50 또는 Swin Transformer)이 저해상도 특성 $F_{low} \in \mathbb{R}^{H/64 \times W/64 \times C}$를 추출한다. 이후 픽셀 디코더가 이를 다양한 해상도의 피라미드 특성으로 변환한다:[1]

$$F_p^i \in \mathbb{R}^{H/S_i \times W/S_i \times C}, \quad i=1,2,3,4$$

여기서 스트라이드는 각각 $S_1=32, S_2=16, S_3=8, S_4=4$이다.

**2. 확산 기반 디코더**

디코더는 L=9개의 Transformer 디코더 레이어로 구성되며, 각 레이어는 다음을 포함한다:[1]

- **마스크 기반 교차 어텐션**: 예측된 마스크 영역 내에서만 주의 계산
  $$M_{att} = M_n > 0 \text{ (이진 임계값)}$$
- **자기 어텐션**: 특성 간 전역 상호작용
- **Feed-Forward Network**: 비선형 변환

첫 번째 레이어에서 노이즈 마스크 $M_n$이 주의 마스크로 직접 변환되고, 이후 레이어에서는 이전 디코더의 출력 마스크가 주의 마스크로 사용된다.[1]

***

### III. 성능 향상 및 일반화 분석
#### A. 세 가지 세그멘테이션 작업에서의 성능

**파노프틱 세그멘테이션** (MS COCO val2017, ResNet-50 백본):[1]

| 방법 | 파라미터 | Epoch | PQ | PQ_Th | PQ_St |
|------|---------|-------|-----|-------|-------|
| Panoptic FPN | - | 12 | 39.0 | 45.9 | 28.7 |
| MaskFormer | 45M | 50 | 46.5 | 51.0 | 39.8 |
| Mask2Former | 44M | 50 | 51.9 | 57.7 | 43.0 |
| Pix2Seq-D (확산) | 94.5M | 800 | 47.5 | 52.2 | 40.3 |
| **DFormer** | **44M** | **50** | **51.1** | **56.6** | **42.8** |

DFormer는 Pix2Seq-D 대비 파라미터 크기를 2.15배 감소시키면서 PQ를 3.6% 향상시켰다. 또한 학습 에포크를 16배 단축했다. Mask2Former과는 경쟁 수준의 성능을 유지하면서 확산 모델의 확률적 특성을 활용할 수 있다는 이점이 있다.[1]

**인스턴스 세그멘테이션** (MS COCO val2017):[1]

| 방법 | AP | AP50 | AP75 | APS | APM | APL |
|------|-----|------|------|-----|-----|-----|
| Mask R-CNN | 37.1 | 58.5 | 39.7 | 18.7 | 39.6 | 53.9 |
| Mask2Former | 43.7 | - | - | 23.4 | 47.2 | 64.8 |
| DiffusionInst | 37.3 | 60.3 | 39.3 | 18.9 | 40.1 | 54.7 |
| **DFormer** | **42.6** | **64.8** | **45.8** | **22.3** | **45.7** | **64.2** |

DFormer는 DiffusionInst 대비 +5.3% AP 절대 개선을 달성했다. 특히 중간 크기 객체(APM)에서 +5.6%의 현저한 개선을 보인다.[1]

**의미론적 세그멘테이션** (ADE20K val, Swin-T 백본):[1]

| 방법 | mIoU |
|------|------|
| Mask2Former | 47.7 |
| DDP (확산) | 46.1 |
| **DFormer** | **48.3** |

DFormer는 최신 의미론적 세그멘테이션 벤치마크에서 +2.2% mIoU 향상을 달성하며, Mask2Former을 미세하게 앞선다.[1]

#### B. 다중 작업 일반화 능력

DFormer의 가장 큰 강점은 **단일 모델로 세 가지 작업을 모두 처리**하면서 각각 경쟁 수준의 성능을 유지한다는 점이다. 이는 다음을 의미한다:[1]

1. **아키텍처 단순화**: 세 가지 별도 모델 대신 하나의 통합 모델 사용
2. **메모리 효율성**: 모델 크기 2.8배 감소 (예: 3개 모델 132M vs 단일 44M)
3. **배포 용이성**: 단일 가중치 파일로 모든 작업 지원

#### C. 아키텍처 설계 최적화 (제거 실험)

제거 실험을 통해 각 설계 선택의 중요성이 검증되었다:[1]

**디코더 레이어 수**:
- 3 레이어: AP 41.1%
- 6 레이어: AP 42.4%
- **9 레이어: AP 42.6%** ← 최적

레이어 9개 이후로는 성능 향상이 없다.[1]

**추론 스텝 수**:
- 1-스텝: AP 42.6%
- 2-스텝: AP 42.7%
- 4-스텝: AP 42.7%

놀랍게도 1-스텝 샘플링으로도 최대 성능에 도달하며, 이는 계산 효율성을 극대화한다.[1]

**노이즈 마스크 수**:
- 50개: AP 40.7%
- **100개: AP 42.6%** ← 최적
- 150개: AP 42.7% (성능 안정화)
- 200개: AP 42.5%

패딩 마스크 100개가 최적 성능을 제공한다.[1]

**마스크 인코딩 전략**:
- 이진 (b=1): AP 41.2%
- **이진 (b=0.1): AP 42.6%** ← 최적
- 무작위 (b=1): AP 42.3%
- 무작위 (b=0.1): AP 42.5%

작은 스케일 팩터(b=0.1)가 마스크 값을 [-0.1, 0.1] 범위로 정규화하여 최상의 성능을 제공한다.[1]

***

### IV. 모델의 일반화 성능 및 강건성
#### A. 다양한 도메인과 데이터셋에서의 성능

DFormer는 여러 대규모 벤치마크에서 일관된 성능을 보여준다:[1]

- **MS COCO (133개 카테고리)**: 객체 중심의 사진
- **ADE20K (150개 카테고리)**: 장면 이해 중심의 영상
- **YouTube-VIS 2019**: 비디오 인스턴스 세그멘테이션 (확장 평가)

각 도메인에서 기존 확산 기반 방법들을 상당한 마진으로 앞선다.

#### A. 백본 독립성

DFormer는 다양한 백본과 호환되며, 각각 성능 향상을 보인다:[1]

| 백본 | 인스턴스 AP | 의미론적 mIoU | 파노프틱 PQ |
|------|-----------|------------|----------|
| ResNet-50 | 42.6 | 46.7 | 51.1 |
| Swin-T | 44.4 | 48.3 | 52.5 |
| 성능 향상 | +1.8% | +1.6% | +1.4% |

Swin Transformer 기반 모델이 모든 작업에서 더 나은 성능을 보이며, 이는 계층적 비전 Transformer의 우수성을 반영한다.[1]

#### C. 확률적 추론의 이점

확산 모델의 고유한 특성으로, DFormer는 **단일 입력에 대한 다중 세그멘테이션 마스크 생성**이 가능하다. 이는:[1]

1. **불확실성 추정**: 예측의 신뢰도 평가
2. **앙상블 효과**: 여러 샘플의 투표 기반 최종 예측
3. **의료 영상 활용**: 모호한 경계 영역의 다양한 해석 제공 가능

***

### V. 한계점 분석 및 개선 방안
#### A. 확인된 한계

**1. 작은 객체 인식 성능**

인스턴스 세그멘테이션에서 작은 객체(APS) 점수는 22.3%로, Mask2Former의 23.4%보다 낮다. 이는 다음의 이유 때문이다:[1]

- 확산 프로세스의 점진적 노이즈 제거가 세밀한 경계 추출에 덜 효과적
- 초기 노이즈 마스크의 저해상도로 인한 정보 손실

**2. 추론 속도**

1-스텝 샘플링에도 불구하고 Mask2Former보다 느린 추론 속도를 보인다. 이는 다음 때문이다:

- 9개 Transformer 디코더 레이어의 순차 처리
- 각 레이어에서의 마스크 기반 어텐션 계산 오버헤드

**3. 학습 복잡성**

마스크 패딩, 노이즈 부패, 이분 매칭 등 여러 전처리 단계가 필요하다.[1]

#### B. 개선 가능성

**1. 작은 객체 개선 전략**

- **다중 스케일 확산**: 서로 다른 해상도의 마스크에 대한 독립적 확산
- **적응형 노이즈 스케줄**: 객체 크기에 따른 동적 노이즈 레벨 조정
- **지역 세분화**: 작은 객체 영역에 추가적 Transformer 레이어 적용

**2. 추론 속도 최적화**

- **잠재 공간 확산**: VAE를 사용한 16배 계산량 감소 가능
- **경량화 디코더**: 레이어 9개 → 4-5개 감소 가능 (제거 실험 기반)
- **ONNX/TensorRT 컴파일**: 10배 이상 속도 향상

**3. 도메인 확장**

최신 연구(2023-2024)에서 DFormer의 설계를 의료 영상에 적용한 여러 방법들이 제시되었다:[3][4][5]

| 방법 | 혁신 | 성능 |
|------|------|------|
| BerDiff (2023) | Bernoulli noise for binary masks | 의료 영상 SOTA |
| MedSegDiff-V2 (2023) | Transformer 강화 | 20개 의료 작업 SOTA |
| LSegDiff (2023) | 잠재 공간 확산 | 50배 빠른 추론 |
| LC-SegDiff (2023) | Label constraint | 1-스텝 안정적 추론 |

***

### VI. 2020년 이후 관련 최신 연구 비교
#### A. 통합 세그멘테이션 기술 진화
**기술 진화 타임라인**:[1][6][7][8][9][10][11]

**2020 - DETR (End-to-End Object Detection with Transformers)**
- 혁신: Transformer 도입, bipartite matching 기반 set prediction
- 파노프틱 PQ: 43.4%
- 영향: 이후 모든 통합 세그멘테이션의 기초

**2021 - MaskFormer (Per-Pixel Classification is Not All You Need)**
- 혁신: 마스크 분류 패러다임 전환 (픽셀별 분류 → 마스크 기반)
- 파노프틱 PQ: 46.5% (+3.1%)
- 핵심: 의미론적/인스턴스 작업 통합 가능성 증명

**2021 - K-Net (Towards Unified Image Segmentation)**
- 혁신: 학습 가능한 동적 커널 (kernel update strategy)
- 파노프틱 PQ: 47.1% (+0.6%)
- 특징: NMS-free, box-free 파노프틱 세그멘테이션

**2022 - Mask2Former (Masked-Attention Mask Transformer)**
- 혁신: 마스크 기반 교차 어텐션 정제
- 파노프틱 PQ: 51.9% (+4.8%)
- 효율성: 메모리 3배 감소 (18GB → 6GB)
- 영향: 통합 세그멘테이션의 실질적 SOTA 확립

**2022 - OneFormer (One Transformer To Rule Universal Segmentation)**
- 혁신: 진정한 멀티태스크 학습 (task-conditioned joint training)
- 특징: 단 한 번 학습으로 3개 작업 모두 최적화
- 기술: 작업 토큰, query-text contrastive loss
- 영향: 멀티태스크 학습의 새로운 표준 제시

**2023 - DFormer (Diffusion-guided Transformer)**
- 혁신: 생성 모델 기반 접근 (denoising process)
- 파노프틱 PQ: 51.1% (경쟁 수준)
- 기술: 마스크 노이즈-제거 프로세스
- 영향: 생성 모델의 차별적 작업 활용 증명

#### B. 성능 수렴의 의미

2022년 Mask2Former (PQ 51.9%)과 2023년 DFormer (PQ 51.1%) 사이의 성능 차이 감소는 중요한 시사점을 제공한다:[12][13][1]

1. **기술의 수렴**: 아키텍처 혁신의 한계에 도달
2. **작은 객체 문제의 구조적 한계**: +5% 이상의 개선에는 근본적 접근 필요
3. **다양한 기술 방향의 탐색**: 단순 성능 경쟁 → 특수 도메인 최적화, 효율성, 견고성 등으로 다양화

#### C. 확산 모델 활용의 확산

DFormer의 성공 이후, 의료 영상, 비디오, 3D 세그멘테이션 등 다양한 분야에서 확산 모델 기반 방법들이 빠르게 등장했다:[3][4][5][14][15][16][17]

- **의료 영상**: BerDiff, MedSegDiff-V2, Diff-SFCT, LC-SegDiff 등
- **효율성 개선**: LSegDiff (잠재 공간), DiffDIS (사전학습 모델 활용)
- **도메인 적응**: DGInStyle (자동 운전용 스타일 제어)

이는 DFormer가 단순한 성능 기록 경신을 넘어 **새로운 세그멘테이션 패러다임을 개척**했음을 의미한다.

***

### VII. 향후 연구에 미치는 영향 및 고려 사항
#### A. 학문적 영향

**1. 생성 모델의 차별적 작업 활용**

DFormer는 확산 모델이 원래 설계된 이미지 생성 작업뿐만 아니라 의미론적 이해(semantic understanding) 작업에도 효과적임을 증명했다. 이는 다음의 새로운 연구 방향을 열었다:[1]

- 다른 생성 모델(VAE, 정규화 흐름 등)의 차별적 작업 활용
- 생성-차별 모델의 하이브리드 구조
- 기하학적 구조를 보존하는 생성적 접근

**2. 확률적 세그멘테이션**

기존 방법은 각 입력에 대해 **단일 결정론적 마스크**를 출력한다. DFormer는 다양한 세그멘테이션 마스크를 생성할 수 있으므로:[1]

- **불확실성 정량화**: 예측의 신뢰도 추정
- **의료 영상 진단**: 모호한 경계 영역의 여러 해석 제공
- **의사결정 지원**: 불확실성에 기반한 추가 검사 제안

**3. 통합 아키텍처의 극한 탐색**

OneFormer, DFormer 등의 성공은 **단일 모델로 무제한 작업 처리** 가능성을 시사한다:

- 도메인 적응(domain adaptation) 문제로의 확장
- 메타학습(meta-learning) 기반 작업 적응
- 기초 모델(foundation model) 개발 방향

#### B. 실무 적용 시 고려사항

**1. 배포 환경 최적화**

- **추론 속도**: Mask2Former 대비 30-50% 느린 추론
  → 자동 운전 등 실시간 처리가 필요한 분야에서는 추가 최적화 필수
- **메모리 사용**: 단일 모델이므로 메모리 효율적 (44M 파라미터)
- **엣지 배포**: 모바일/IoT 기기에서는 경량화 필수

**2. 도메인 특화 파인튜닝 필요성**

DFormer는 일반적인 RGB 자연 이미지에 최적화되어 있으며, 특수 도메인에서는:[1]

| 도메인 | 과제 | 해결 방안 |
|--------|------|---------|
| 의료 영상 | 작은 종양/병변 인식 | 의료 특화 노이즈 스케줄 |
| 위성 영상 | 고해상도, 작은 객체 | 다중 스케일 확산 |
| 산업 검사 | 실시간 처리 요구 | 1-스텝 샘플링 + 양자화 |
| 자율 주행 | 견고성, 속도 | Latent diffusion + 경량화 |

**3. 데이터 준비의 복잡성**

마스크 패딩, 인코딩, 노이즈 부패 등 복잡한 전처리가 필요하므로:[1]

- 관련 코드 라이브러리 구축 필수
- 하이퍼파라미터 튜닝 가이드 필요
- 도메인별 최적 설정 연구

#### C. 향후 연구 우선순위

**1 순위: 추론 속도 개선** (실무 배포의 핵심)
- Latent diffusion model 적용 → 50배 계산량 감소 가능
- 경량 Transformer 디코더 설계
- 구조 가지치기 (pruning) 및 양자화

**2순위: 작은 객체 성능** (일반화의 핵심)
- 다중 스케일 확산 모듈
- 적응형 노이즈 스케줄
- 지역 세분화 메커니즘

**3순위: 도메인 확장** (응용의 핵심)
- 의료 영상 (BerDiff 등이 이미 진행 중)
- 3D 의료 볼륨 세그멘테이션
- 비디오 세그멘테이션 (시간축 일관성)

***

### 결론
DFormer는 **확산 모델이 차별적 세그멘테이션 작업에도 효과적일 수 있음**을 입증한 중요한 연구이다. 파노프틱 PQ 51.1%는 기존 확산 기반 방법 대비 현저한 향상이며, 단일 모델로 세 가지 작업을 처리하는 통합 아키텍처는 실무 적용의 효율성을 크게 높인다.[1]

그러나 작은 객체 인식(APS 22.3%)과 추론 속도 측면에서는 개선 여지가 있다. 향후 연구는 잠재 공간 확산을 통한 속도 개선, 적응형 노이즈 스케줄을 통한 작은 객체 성능 향상, 그리고 의료/위성 영상 등 특수 도메인으로의 확장이 중요할 것으로 예상된다.

최신 연구 동향에서 볼 때, DFormer의 확산 모델 기반 접근이 2023-2024년에 의료 영상 분야에서 급속히 확산되고 있으며, 이는 DFormer가 개척한 패러다임이 학문적으로 얼마나 영향력 있는지를 보여준다.

***

### 참고문헌 및 인용 논문

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/0275faf4-adfd-4e4b-9377-cbc53f67ee19/2306.03437v2.pdf)
[2](https://arxiv.org/pdf/2306.03437.pdf)
[3](https://link.springer.com/10.1007/978-3-031-43901-8_47)
[4](https://ieeexplore.ieee.org/document/10804854/)
[5](https://arxiv.org/abs/2305.09447)
[6](https://arxiv.org/abs/2112.01527)
[7](https://ieeexplore.ieee.org/document/10203147/)
[8](https://arxiv.org/abs/2211.06220)
[9](https://arxiv.org/pdf/2107.06278.pdf)
[10](https://proceedings.neurips.cc/paper/2021/file/55a7cf9c71f1c9c495413f934dd1a158-Paper.pdf)
[11](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460205.pdf)
[12](https://ieeexplore.ieee.org/document/9878483/)
[13](http://arxiv.org/pdf/2112.01527v2.pdf)
[14](https://ieeexplore.ieee.org/document/10708363/)
[15](https://dl.acm.org/doi/10.1145/3628797.3629010)
[16](https://ieeexplore.ieee.org/document/10385510/)
[17](https://ieeexplore.ieee.org/document/10385655/)
[18](https://ieeexplore.ieee.org/document/10635329/)
[19](https://arxiv.org/abs/2310.12868)
[20](https://ieeexplore.ieee.org/document/10424888/)
[21](https://arxiv.org/pdf/2112.00390.pdf)
[22](https://arxiv.org/html/2410.10105)
[23](https://arxiv.org/pdf/2312.03048.pdf)
[24](https://arxiv.org/html/2304.04429)
[25](https://arxiv.org/html/2407.12952v2)
[26](https://arxiv.org/html/2301.11798v2)
[27](https://arxiv.org/pdf/2303.10326.pdf)
[28](https://www.nature.com/articles/s41598-025-90631-x)
[29](https://www.digitalocean.com/community/tutorials/panoptic-segmentation)
[30](https://proceedings.mlr.press/v172/wolleb22a/wolleb22a.pdf)
[31](https://openaccess.thecvf.com/content/CVPR2022/papers/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.pdf)
[32](https://isprs-annals.copernicus.org/articles/X-1-W1-2023/605/2023/isprs-annals-X-1-W1-2023-605-2023.pdf)
[33](https://arxiv.org/abs/2407.03548)
[34](https://www.sciencedirect.com/science/article/pii/S1361841524002056)
[35](https://arxiv.org/pdf/2111.10250.pdf)
[36](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/dformer/)
[37](https://arxiv.org/html/2512.01292v1)
[38](https://openaccess.thecvf.com/content/CVPR2023/papers/Jain_OneFormer_One_Transformer_To_Rule_Universal_Image_Segmentation_CVPR_2023_paper.pdf)
[39](https://ar5iv.labs.arxiv.org/html/2104.03962)
[40](https://arxiv.org/html/2307.00773v1)
[41](https://arxiv.org/html/2408.16504v2)
[42](https://arxiv.org/html/2510.09681v1)
[43](https://arxiv.org/abs/2505.19795)
[44](https://arxiv.org/abs/2204.05370)
[45](https://arxiv.org/html/2310.12868v2)
[46](https://pmc.ncbi.nlm.nih.gov/articles/PMC11876438/)
[47](https://arxiv.org/abs/2209.07704)
[48](https://arxiv.org/abs/2404.14657)
[49](https://ieeexplore.ieee.org/document/10760750/)
[50](https://www.mdpi.com/1424-8220/23/2/581)
[51](https://www.mdpi.com/2227-7390/12/5/765)
[52](https://isprs-archives.copernicus.org/articles/XLVIII-1-W2-2023/203/2023/)
[53](https://ieeexplore.ieee.org/document/10334480/)
[54](https://arxiv.org/pdf/2303.07336.pdf)
[55](https://www.mdpi.com/1424-8220/23/2/581/pdf?version=1672826687)
[56](https://arxiv.org/html/2404.14657)
[57](https://arxiv.org/pdf/2402.19422.pdf)
[58](https://arxiv.org/abs/2112.10764)
[59](https://ar5iv.labs.arxiv.org/html/2106.14855)
[60](https://www.youtube.com/watch?v=utxbUlo9CyY)
[61](https://huggingface.co/blog/mask2former)
[62](https://papers.nips.cc/paper/2021/file/55a7cf9c71f1c9c495413f934dd1a158-Paper.pdf)
[63](https://pmc.ncbi.nlm.nih.gov/articles/PMC12252279/)
[64](https://huggingface.co/docs/transformers/model_doc/mask2former)
[65](https://www.semanticscholar.org/paper/K-Net:-Towards-Unified-Image-Segmentation-Zhang-Pang/262654ac1d13cf8d8b204594f4a88d3e04f3dd37)
[66](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Mr._DETR_Instructive_Multi-Route_Training_for_Detection_Transformers_CVPR_2025_paper.pdf)
[67](https://arxiv.org/pdf/2112.01527.pdf)
[68](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Video_K-Net_A_Simple_Strong_and_Unified_Baseline_for_Video_CVPR_2022_paper.pdf)
[69](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Semi-DETR_Semi-Supervised_Object_Detection_With_Detection_Transformers_CVPR_2023_paper.pdf)
[70](https://ar5iv.labs.arxiv.org/html/2305.01255)
[71](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_DA-DETR_Domain_Adaptive_Detection_Transformer_With_Information_Fusion_CVPR_2023_paper.pdf)
[72](https://arxiv.org/pdf/2506.05897.pdf)
[73](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_Hybrid_Proposal_Refiner_Revisiting_DETR_Series_from_the_Faster_R-CNN_CVPR_2024_paper.pdf)
[74](https://github.com/facebookresearch/Mask2Former)
