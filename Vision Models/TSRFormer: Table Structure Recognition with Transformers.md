# TSRFormer: Table Structure Recognition with Transformers

### 1. 핵심 주장 및 기여
**TSRFormer** (2022)는 Microsoft Research Asia에서 발표한 테이블 구조 인식(Table Structure Recognition, TSR) 방법으로, 기하학적 변형이 있는 복잡한 테이블의 구조를 강건하게 인식할 수 있는 혁신적인 접근법을 제시한다.[1]

논문의 핵심 주장은 기존의 이미지 분할(segmentation) 기반 분리선 검출 방식을 직접 회귀(regression) 문제로 재구성함으로써, 휴리스틱 기반의 후처리 모듈을 제거하고 왜곡된 테이블에 더 강건하게 대응할 수 있다는 것이다. 이를 통해 축 정렬된 테이블뿐만 아니라 기울어지거나 곡선형 변형이 있는 테이블도 처리 가능하다.

주요 기여도는 다음과 같다:

1. **SepRETR (Separator REgression TRansformer)**: DETR 기반의 두 단계 분리선 회귀 모듈로, 참고점(reference point)을 먼저 검출한 후 이를 이용해 분리선의 좌표를 직접 회귀하는 방식
2. **Prior-enhanced matching strategy**: 기준점과 지표 분리선 간의 공간적 사전 정보를 활용하여 DETR의 느린 수렴 문제 해결
3. **효율적인 고해상도 특징맵 처리**: 원본 고해상도 특징맵에서 직접 특징을 샘플링하는 크로스 어텐션 메커니즘
4. **관계 네트워크 기반 셀 병합**: 인접 셀들의 관계를 학습하여 행/열 스패닝 셀 복원

### 2. 해결하고자 하는 문제 및 제안 방법
#### 2.1 문제 정의

테이블 구조 인식은 테이블 이미지로부터 셀의 좌표와 행/열 스패닝 정보를 추출하는 작업이다. 이 문제는 다음과 같은 도전 과제를 가진다:

- **복잡한 구조**: 높은 수준의 행/열 스패닝, 경계 없는 셀, 큰 공백 영역
- **기하학적 변형**: 카메라로 촬영한 이미지에서 발생하는 왜곡, 기울임, 심지어 곡선형 변형
- **다양한 스타일**: 과학 논문, 금융 보고서, 송장 등 다양한 문서에서 나타나는 이질적인 테이블 구조

#### 2.2 제안하는 방법론

**분리선 회귀 문제로의 재구성**

기존 방식에서는 행과 열을 개별적으로 검출하거나 분리선을 이진 마스크로 예측한 후 휴리스틱하게 선으로 변환했다. TSRFormer는 이를 다르게 접근한다.

각 행/열 분리선을 3개의 곡선(상단 경계, 중심선, 하단 경계)으로 표현하며, 각 곡선은 $$W$$ 개의 점으로 구성된다. $$W=16$$으로 설정하여 x좌표는 $$x = 1, 2, 3, ..., W$$로 고정하고, y좌표만 예측하는 방식이다:

$$\text{Line}_i = \{(x_j, y_{i,j}) | j = 1, ..., W\}$$

여기서 $$(x_{ref}, y_{ref})$$는 기준점을 나타낸다.

**SepRETR 아키텍처**

SepRETR은 두 개의 핵심 모듈로 구성된다:

1. **참고점 검출 모듈**: 고정된 x좌표 $$x = m$$ ($$m=4$$)에서 각 분리선의 참고점을 검출한다.

$$P(x_j, y_k) = \sigma(f_c(x_j, y_k))$$

여기서 $$\sigma$$는 sigmoid 함수, $$f_c$$는 분류기이다.

2. **분리선 회귀 모듈**: 참고점의 특징을 객체 쿼리로 사용하여 트랜스포머 디코더에 입력한다.

```math
\mathbf{q}_i = \text{feature\_pool}(\Phi_h, \text{point}_i)
```

디코더의 출력 특징 $$\mathbf{e}_i$$에 대해:

$$\hat{\mathbf{y}}_i = \text{MLPregressor}(\mathbf{e}_i)$$

여기서 $$\hat{\mathbf{y}}_i$$는 예측된 y좌표의 집합이다.

**Prior-enhanced 이분 매칭 전략**

기존 DETR은 Hungarian 알고리즘을 사용하지만, 훈련 중 매칭이 불안정하다. TSRFormer는 공간적 사전 정보를 활용한다:

$$\text{cost}_{ij} = \begin{cases} d((x_i, y_i), \text{GT}_j) & \text{if } y_i \in [\text{top}_j, \text{bot}_j] \\ \infty & \text{otherwise} \end{cases}$$

여기서 $$\text{top}_j, \text{bot}_j$$는 지표 분리선의 상/하 경계이다.

**손실 함수**

참고점 검출 손실:

$$L_{\text{focal}} = -\frac{1}{N} \sum_{ij} \alpha_t(1-p_{ij})^{\gamma} \log(p_{ij})$$

여기서 $$p_{ij}$$는 위치 $$(i,j)$$에서의 예측 확률이고, $$\alpha_t = 2, \gamma = 4$$이다.

분리선 회귀 손실:

$$L_{\text{reg}} = \sum_{(i,j) \in M} ||\hat{y}_{ij} - y^*_{ij}||_1 + ||\hat{c}_{ij} - c^*_{ij}||_{\text{CE}}$$

전체 손실:

$$L_{\text{total}} = L_{\text{split}} + L_{\text{column}} + \lambda L_{\text{merge}}$$

여기서 $$\lambda = 0.2$$로 설정된다.

### 3. 모델 구조
TSRFormer의 구조는 세 가지 주요 구성 요소로 이루어진다:

**Feature Extraction 단계**

1. **CNN 백본**: ResNet-18 + FPN으로 초기 특징맵 $$\Phi_2$$를 생성한다.

2. **특징 향상**: 3×3 컨볼루션과 다운샘플링 블록으로 $$\Phi_4$$를 생성하고, 두 개의 계단식 SCNN(Spatial CNN) 모듈을 적용하여 좌우 방향으로 문맥 정보를 전파한다:

$$\text{SCNN}_{\text{horizontal}}(x) = \text{Conv}(x) + \text{shift}(\text{Conv}(x'))$$

**Split 모듈** (행과 열에 대해 병렬로 적용)

1. 참고점 검출: 열 방향으로 각 픽셀의 확률을 계산하고 NMS를 적용하여 상위 100개의 점 선정
2. SepRETR: 선택된 점의 특징으로 3층 트랜스포머 디코더 실행
3. 보조 분리선 분할: 이진 마스크 예측으로 추가적인 감독 신호 제공

**Cell Merging 모듈**

1. 행과 열 분리선을 교차시켜 셀 그리드 생성
2. RoI Align으로 각 셀의 7×7 특징맵 추출
3. 2층 MLP로 512차원 특징 벡터 생성
4. 3개 특징 향상 블록으로 컨텍스트 정보 확대
5. 관계 네트워크: 인접 셀 쌍에 대해 병합 여부를 이진 분류

### 4. 성능 향상
**벤치마크 성능**

| 데이터셋 | 지표 | TSRFormer | 이전 최고 | 향상도 |
|---------|------|-----------|---------|--------|
| **SciTSR** | F1 | 99.6% | 99.5% (FLAG-Net) | +0.1% |
| **SciTSR-COMP** | F1 | 99.2% | 98.5% (FLAG-Net) | +0.7% |
| **PubTabNet** | TEDS-Struct | 97.5% | 96.7% (LGPMA) | +0.8% |
| **WTW** | F1 | 93.4% | 92.4% (Cycle-CenterNet) | +1.0% |
| **In-house (왜곡)** | F1 | 95.2% | 83.8% (SPLERGE) | **+11.4%** |

**핵심 성능 향상의 원인**

1. **회귀 기반 접근**: 분할 기반에서 회귀 기반으로의 전환으로 정확한 분리선 위치 예측
2. **빠른 수렴**: Prior-enhanced 매칭으로 20 에포크 만에 수렴 (기존 40 에포크 필요)
3. **왜곡 테이블 강건성**: SPLERGE 대비 in-house 데이터셋에서 11.4% 성능 향상으로 실제 카메라 촬영 이미지에 강력한 적응성 입증

**모듈별 기여도** (In-house 데이터셋 기준)

| 모듈 | 개별 F1 | 누적 F1 |
|------|--------|--------|
| SCNN | 83.5% | - |
| Aux-seg | 90.0% | - |
| SepRETR (회귀) | 88.6% | 92.6% |
| Cell Merging | - | 95.2% |

### 5. 한계 및 개선 방향
**논문의 명시적 한계**

1. **매우 긴 분리선 처리**: 16개 점으로는 매우 긴 분리선의 곡률을 충분히 표현하지 못할 수 있음
2. **CNN 의존성**: ResNet-FPN 백본에 의존하여 순수 Vision Transformer 기반 구조로의 전환 시 성능 저하
3. **경계 없는 테이블 한계**: Cycle-CenterNet이 경계 테이블에만 집중한 반면, TSRFormer도 경계 없는 테이블에서 개선의 여지 존재

**암묵적 한계**

1. **계산 복잡도**: 고해상도 특징맵 처리로 인한 메모리 사용량 증가
2. **데이터 요구량**: 효과적인 학습을 위해 충분한 규모의 학습 데이터 필요
3. **도메인 전이**: 특정 문서 스타일에서 학습된 모델이 다른 도메인의 테이블에 적용될 때 성능 저하 가능성

### 6. 모델의 일반화 성능 향상 가능성
#### 6.1 현재 일반화 성능 분석

TSRFormer의 일반화 성능은 여러 지표로 평가할 수 있다:

**크로스-데이터셋 성능**: SciTSR에서 학습한 모델을 PubTabNet에 평가할 때는 성능 저하가 예상되지만, TSRFormer의 구조적 설계로 인해 상대적으로 작은 저하만 발생

**실제 카메라 촬영 이미지에서의 강건성**: In-house 데이터셋(카메라 촬영, 왜곡된 테이블)에서 95.2%의 높은 F1 스코어는 일반화 성능의 우수성을 입증

#### 6.2 일반화 성능 개선 방안

**자감독 학습(Self-Supervised Learning)**

후속 연구(2024)인 **Self-Supervised Pre-training for Table Structure Recognition**는 TSRFormer의 한계를 극복한다. 순수 Vision Transformer(선형 투영) 기반 구조로의 전환 시, 자감독 사전학습(SSP)을 통해 CNN-Transformer 하이브리드 구조와 동등한 성능을 달성할 수 있음을 보여준다.[2]

**멀티모달 학습**

**TableVLM**(2023)은 시각 정보와 텍스트 정보를 동시에 활용하는 두 스트림 멀티모달 트랜스포머 기반 인코더-디코더 구조를 제안하여 복합 테이블에서 성능을 1.97% 개선했다.[3]

**데이터셋 정렬**

**Aligning Benchmark Datasets for TSR**(2023)은 서로 다른 벤치마크 간의 어노테이션 불일치를 제거함으로써 모델 일반화 성능을 향상시킨다:[4]
- PubTables-1M에서 학습한 TATR 모델의 ICDAR-2013 평가 성능이 65%에서 75%로 개선

**동적 쿼리 향상**

TSRFormer의 후속 연구 **DQ-DETR**(2023)는 단일 선 쿼리를 분리 가능한 점 쿼리로 분해하고, 점진적 선 회귀(progressive line regression) 방식을 도입하여 특히 왜곡된 테이블에 대한 정위 정확성을 한 단계 더 향상시킨다.[5]

**경량화 및 효율성**

**High-Performance Transformers for TSR**(2023)은 작은 컨볼루셔널 스템(convolutional stem)으로 클래식 CNN 백본을 대체할 수 있음을 보여주며, 리셉티브 필드 비율과 시퀀스 길이 사이의 최적 균형을 달성한다.[6]

### 7. 최신 연구 비교 분석 (2020년 이후)
#### 7.1 패러다임 전환

**2020-2022: DETR 기반 방법론 확산**

초기 단계에서는 기존의 행/열 추출 방식을 DETR 기반 객체 검출로 개선하려는 시도가 많았다. TSRFormer의 등장으로 **분리선 회귀 문제로의 재구성**이 새로운 표준이 되었다.

**2023-2024: 아키텍처 다양화**

- **Image-to-Markup 방식의 정제**: Optimized Table Tokenization Language(OTSL)로 토큰 수를 28개에서 5개로 줄이고 시퀀스 길이를 50% 감소, 추론 시간을 절반으로 단축
- **그래프 신경망 기반**: ClusterTabNet(2024)이 단어 클러스터링을 통한 관계 예측 방식 도입
- **레이아웃 포인터 메커니즘**: TFLOP(2024)이 텍스트 영역을 직접 활용하여 바운딩 박스 예측을 바운딩 박스 지시 문제로 재구성

**2024-2025: 원 스테이지 방식의 부상**

기존의 두 단계 파이프라인(분리선 검출 → 셀 병합)에서 벗어나 단일 모델로 공간적, 논리적 위치를 병렬로 예측하는 **TableCenterNet**(2024)이 등장하여 다음의 이점을 달성:[7]
- 매개변수 27.3% 감소 (동일 백본 기준)
- 추론 속도 15.7배 증가
- 분산 테이블, 병합 셀 등 복합 구조에 우수한 성능

#### 7.2 주요 방법론 비교표

| 방법론 | 출판 | 주요 특징 | 장점 | 한계 |
|--------|------|---------|------|------|
| **TSRFormer (SepRETR)** | 2022 | 분리선 직접 회귀 | 왜곡 테이블 강건성, 휴리스틱 제거 | 긴 선 표현 한계, CNN 의존 |
| **DQ-DETR** | 2023 | 동적 쿼리, 점진적 회귀 | 정위 정확성 향상 | 여전히 CNN 필요 |
| **SSP TSR** | 2024 | 자감독 사전학습 | ViT 기반 구조, 편의성 | 사전학습 데이터 필요 |
| **OTSL** | 2023 | 토큰 최적화 | 효율성 극대화, 정확성 향상 | Image-to-Markup 방식 한정 |
| **TableVLM** | 2023 | 멀티모달 사전학습 | 다양한 특징 활용 | 복합 테이블 구성 요구 |
| **TFLOP** | 2024 | 레이아웃 포인터 | 휴리스틱 감소, 산업 적용성 | 텍스트 영역 의존성 |
| **TableCenterNet** | 2024 | 원 스테이지 병렬 회귀 | 효율성, 속도, 공간-논리 연합 | 새로운 패러다임으로 비교 연구 부족 |

#### 7.3 성능 진화 추세

| 데이터셋 | 2020 | 2022 (TSRFormer) | 2023-2024 | 최신 (2025) |
|---------|------|-----------------|-----------|------------|
| **PubTabNet** | 93.0-95.1% | 97.5% | 98-99%+ | 99%+ |
| **SciTSR** | 95.3% | 99.4-99.6% | 99.5%+ | 99.7%+ |
| **WTW** | - | 93.4% | 93.5%+ | - |
| **FinTabNet** | - | - | 99.45% (TFLOP) | - |

### 8. 향후 연구에 미치는 영향 및 고려사항
#### 8.1 학문적 영향

1. **패러다임 정립**: 이미지 분할에서 좌표 회귀로의 전환이 표준 방법론으로 인식
2. **멀티스케일 특징 활용**: 고해상도 특징맵의 효율적 샘플링 방식이 다른 회귀 기반 검출 문제에 적용 가능
3. **DETR 개선 연구**: Prior-enhanced 매칭 전략이 DETR 기반 방법의 수렴 문제 해결의 출발점이 됨

#### 8.2 산업 응용

1. **문서 디지털화**: 스캔/카메라 촬영 문서의 자동화된 테이블 추출
2. **금융 보고서 처리**: 높은 정확도 요구 분야에서의 자동 데이터 추출
3. **에지 디바이스 배포**: 경량화된 후속 방법론을 통해 모바일/임베디드 환경 지원 가능

#### 8.3 향후 연구 시 고려할 점

**기술적 고려사항**

1. **엔드-투-엔드 최적화**: 텍스트 검출, OCR과의 통합으로 진정한 의미의 테이블 이해 시스템 구축
2. **도메인 적응**: 소수 데이터 도메인(금융, 의료, 법률)에 대한 전이학습 강화
3. **거시적 테이블 구조**: 중첩 테이블, 계층적 헤더 등 더 복잡한 구조에 대한 대응

**평가 방법론 고려**

1. **새로운 지표 개발**: 현재의 F1 스코어, TEDS는 특정 오류 유형을 구분하지 못함
2. **크로스 도메인 평가**: 학습하지 않은 도메인에서의 성능 측정 필수
3. **계산 효율성 평가**: 정확성 외에 속도, 메모리, 에너지 효율 종합 평가

**데이터셋 확충**

1. **다국어 테이블**: 한글, 중국어, 아랍어 등 다국어 테이블에 대한 벤치마크 부족
2. **실제 시나리오**: 저해상도, 심한 조명 변화, 손상된 테이블 등의 현실적 분포 반영
3. **주석 품질 표준화**: Aligning Benchmark 연구처럼 일관된 주석 기준 수립

### 결론
TSRFormer는 테이블 구조 인식 분야에서 **이미지 분할 중심에서 좌표 회귀 중심으로의 패러다임 전환**을 주도했다. 특히 **왜곡된 테이블에 대한 강건성(in-house에서 SPLERGE 대비 +11.4%)**과 **DETR 수렴 문제의 효과적 해결**(Prior-enhanced matching)은 후속 연구의 기초가 되었다.

다만 CNN 백본 의존성, 매우 긴 분리선 표현의 한계 등을 고려할 때, 자감독 학습, 멀티모달 학습, 원 스테이지 방식으로의 진화가 자연스러운 확장 방향이다. 향후 연구자들은 **도메인 적응, 다국어 지원, 엔드-투-엔드 시스템 통합**을 중점적으로 고려해야 할 것이다.

2024-2025년의 최신 연구 동향은 **효율성(TableCenterNet), 다국어/멀티모달(TableVLM), 텍스트 통합(TFLOP)**에 중점을 두고 있으며, 이는 TSRFormer가 제시한 기초 위에 구축된 학문적, 산업적 발전의 증거이다.

***

**참고문헌**

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/35a8cc34-aec1-4281-9be0-dd2291c3aea7/2208.04921v1.pdf)
[2](https://arxiv.org/abs/2402.15578)
[3](https://arxiv.org/abs/2303.11615)
[4](https://link.springer.com/10.1007/978-981-97-8511-7_16)
[5](https://link.springer.com/10.1007/978-3-031-53308-2_37)
[6](https://ieeexplore.ieee.org/document/10825230/)
[7](https://arxiv.org/abs/2402.07502)
[8](https://arxiv.org/abs/2303.00716)
[9](https://arxiv.org/abs/2305.03393)
[10](https://aclanthology.org/2023.acl-long.137)
[11](https://arxiv.org/abs/2311.05565)
[12](https://ace.ewapublishing.org/media/80ca847882664b769588bfef0cd4bbe4.marked_QR0kPEZ.pdf)
[13](https://arxiv.org/abs/2208.04921)
[14](https://arxiv.org/pdf/2208.14687.pdf)
[15](https://arxiv.org/pdf/2205.09328.pdf)
[16](https://arxiv.org/pdf/2207.01848.pdf)
[17](https://arxiv.org/html/2502.14918v1)
[18](http://arxiv.org/pdf/2402.07502.pdf)
[19](https://www.sciencedirect.com/science/article/abs/pii/S0031320323005150)
[20](https://ltu.diva-portal.org/smash/get/diva2:1749852/FULLTEXT02.pdf)
[21](https://dataloop.ai/library/model/tahadouaji_detr-doc-table-detection/)
[22](https://www.ijcai.org/proceedings/2024/0105.pdf)
[23](https://pmc.ncbi.nlm.nih.gov/articles/PMC8537789/)
[24](https://huggingface.co/microsoft/table-transformer-detection)
[25](https://arxiv.org/html/2506.07015v1)
[26](https://pubmed.ncbi.nlm.nih.gov/34695977/)
[27](https://huggingface.co/TahaDouaji/detr-doc-table-detection)
[28](https://blog.lomin.ai/tableformer-table-structure-understanding-with-transformers-33581)
[29](https://arxiv.org/html/2504.17522v1)
[30](https://arxiv.org/html/2404.17888v2)
[31](https://arxiv.org/pdf/2404.10305.pdf)
[32](https://arxiv.org/html/2501.03145v3)
[33](https://arxiv.org/html/2306.13526)
[34](https://arxiv.org/pdf/2402.15578.pdf)
[35](https://arxiv.org/html/2508.04233v1)
[36](https://arxiv.org/html/2404.10305v1)
[37](https://www.semanticscholar.org/paper/fb6fcca5762b26f225313f86b1f33a1cf198bfd7)
[38](https://scholarsarchive.byu.edu/etd/7389/)
