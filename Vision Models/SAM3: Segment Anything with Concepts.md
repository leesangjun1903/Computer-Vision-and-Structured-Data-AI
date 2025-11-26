# SAM 3: Segment Anything with Concepts

### 1. 핵심 주장과 주요 기여

**SAM 3(Segment Anything Model 3)**는 Meta에서 2025년 11월에 발표한 혁신적인 비전 파운데이션 모델로, 기존 SAM 시리즈의 한계를 극복하고 새로운 능력을 도입하는 것을 핵심 주장으로 한다.[1]

**주요 기여**는 다음과 같다:[1]

1. **Promptable Concept Segmentation(PCS) 태스크 정의**: SAM 1/2가 단일 객체 세분화에 집중한 반면, SAM 3는 텍스트 명사구("yellow school bus"), 이미지 예시, 또는 둘의 조합으로부터 **개념에 매칭되는 모든 인스턴스를 한 번에 검출하고 세분화**하는 능력을 제공한다.[1]

2. **혁신적 모델 아키텍처**: 검출(detection)과 추적(tracking)을 분리하여 설계하고, 특히 **Presence Head**를 도입하여 인식(recognition)과 위치 결정(localization)을 명시적으로 분리했다.[1]

3. **대규모 고품질 데이터 엔진**: 인간-AI 루프 기반의 데이터 엔진을 통해 **4M 고유 개념 레이블**과 **52M 마스크**, 합성 데이터 **38M 명사구와 1.4B 마스크**를 수집했다.[1]

4. **SA-Co 벤치마크 개발**: 207K 고유 개념을 포함한 대규모 평가 벤치마크를 제공하여 기존 벤치마크보다 **50배 이상 많은 개념**을 포함한다.[1]

5. **성능 혁신**: 기존 시스템의 정확도를 이미지와 비디오 PCS 모두에서 **2배 달성**하고, LVIS에서 제로샷 마스크 AP **48.8 대 기존 최고 38.5**를 달성했다.[1]

***

### 2. 논문이 해결하고자 하는 문제와 제안 방법

#### 2.1 핵심 문제

기존 SAM 시리즈의 근본적 한계:[1]

- **제한된 세분화 범위**: 단일 프롬프트당 하나의 객체만 세분화 가능
- **개념 일반화의 부재**: 이미지나 비디오에서 특정 개념에 매칭되는 **모든 인스턴스를 찾을 수 없음**
- **오픈 어휘 개념 인식 부족**: 단순 명사구로 정의된 시각 개념의 포괄적 인식 불가

#### 2.2 제안 방법: 아키텍처와 구성

**전체 아키텍처 개요**:[1]

SAM 3는 **공유 백본을 사용하는 이중 인코더-디코더 트랜스포머** 구조로 구성된다:

- **이미지 레벨 검출기(Detector)**: DETR 기반 구조
- **메모리 기반 비디오 추적기(Tracker)**: SAM 2 트랜스포머 인코더-디코더 아키텍처 상속
- **공유 Perception Encoder(PE) 백본**: 비전-언어 정렬 인코더

#### 2.3 핵심 기술 혁신: Presence Head

**Presence Head의 역할**:[1]

전통적인 DETR 기반 검출기는 각 proposal query가 동시에 객체 인식(무엇인가?)과 위치 결정(어디인가?)을 수행해야 한다. 그러나 이는 내재적 갈등을 야기한다:

- **인식**: 전역 문맥이 중요 → 이미지 전체를 "봐야" 함
- **위치결정**: 국소적 특성이 중요 → 특정 영역에 집중

**SAM 3의 해결책:**

$$p(\text{total score}_i) = p(\text{concept present}) \times p(\text{localize}_i | \text{concept present})$$

여기서:
- $$p(\text{concept present})$$: 전역 Presence Token이 계산 → 이미지에 개념이 존재하는가?
- $$p(\text{localize}_i | \text{concept present})$$: 각 proposal query가 계산 → 존재한다면 위치는?

이러한 분리는 모델의 IL_MCC(이미지 레벨 정확도)를 **0.77에서 0.82로 개선**했다.[1]

#### 2.4 검출 아키텍처 상세

**구성 요소**:[1]

$$\text{Detector} = \text{FusionEncoder}(\text{ImageEncoder}(\text{image}), \text{PromptTokens})$$

여기서 PromptTokens는 다음으로 구성:

1. **텍스트 토큰**: NP 프롬프트를 인코딩한 토큰
2. **이미지 예시 토큰**: ROI-pooled 시각 특성 + 위치/레이블 임베딩
3. **Mask Head**: MaskFormer 기반 pixel-level 마스크 예측

**학습 손실함수**:

$$\mathcal{L} = \mathcal{L}_{\text{classification}} + \lambda_{\text{box}}\mathcal{L}_{\text{box}} + \lambda_{\text{mask}}\mathcal{L}_{\text{mask}}$$

- Classification: 존재/부재 이진 분류
- Box: 바운딩박스 회귀
- Mask: 마스크 DICE/CE 손실

#### 2.5 비디오 추적 메커니즘

**시간적 프로퍼게이션**:[1]

$$\hat{M}_t = \text{propagate}(M\_{t-1})$$

$$O_t = \text{detect}(I_t, P)$$

```math
M_t = \text{match and update}(\hat{M}_t, O_t)
```

여기서:
- $$M_t$$: t 프레임의 tracklet 마스크
- $$\hat{M}_t$$: 이전 프레임에서 전파된 마스크
- $$O_t$$: 현재 프레임의 새로운 검출

**매칭 전략**:

1. **IoU 기반 매칭**: 전파 마스크와 새 검출 간 IoU 계산
2. **시간적 일관성**: Tracklet detection score를 통해 시간 윈도우 내 일관성 검증
3. **재-프롬팅**: 높은 신뢰도 검출로 메모리 뱅크를 주기적으로 업데이트

***

### 3. 모델 구조와 성능 분석

#### 3.1 상세 모델 구조

**4단계 학습 파이프라인**:[1]

| 단계 | 구성 요소 | 목표 |
|------|---------|------|
| 1 | Perception Encoder 사전학습 | 비전-언어 정렬 학습 |
| 2 | 검출기 사전학습 | 이미지 레벨 개념 인식 |
| 3 | 검출기 미세조정 | PCS 태스크 최적화 |
| 4 | 추적기 학습 (고정 백본) | 비디오 세분화 능력 |

**백본 선택 분석**:[1]

PE(Perception Encoder) 백본이 선택된 이유:

| 백본 | SA-Co/Gold cgF1 | COCO-O AP |
|-----|----------------|----------|
| PE-L+ (선택) | 43.2 | 42.5 |
| DINOv2-L | 35.3 | 31.9 |
| Hiera-L | 32.8 | 22.0 |

PE는 **비전-언어 정렬**과 **의미적 이해** 측면에서 우수한 성능을 제공한다.

#### 3.2 혁신적 메트릭: cgF1

기존 AP 메트릭은 캘리브레이션을 고려하지 않아 실제 사용이 어렵다. SAM 3는 새로운 메트릭을 제안:[1]

```math
\text{cgF1} = 100 \times \text{pmF1} \times \text{IL\_MCC}
```

여기서:
- $$\text{pmF1}$$: Positive Micro F1 (위치 결정 정확도)
- $$\text{IL MCC}$$: Image-Level Matthews Correlation Coefficient (개념 존재 예측)

이 메트릭은 **인식과 위치결정 모두**의 성능을 평가한다.

#### 3.3 성능 향상 결과

**이미지 PCS 성능**:[1]

| 데이터셋 | SA-Co/Gold cgF1 | LVIS AP | 개선도 |
|---------|-----------------|---------|--------|
| OWLv2 baseline | 24.6 | 43.4 | - |
| SAM 3 | 54.1 | 48.5 | **+120%** |
| 인간 성능 | 72.8 | - | - |

**비디오 PCS 성능**:[1]

| 벤치마크 | SAM 3 cgF1 | 인간 성능 | 달성률 |
|---------|-----------|---------|--------|
| SA-Co/VEval | 30.3 | 53.1 | 57% |
| YT-Temporal-1B | 50.8 | 71.2 | 71% |

**대화형 개선 (K-shot)**:[1]

텍스트 프롬프트에 예시 박스를 점진적으로 추가할 때:

$$\text{cgF1 improvement} = 21.6 \text{ points (3 클릭 후)}$$

4번 클릭 후 성능이 포화되는 것을 관찰했으며, 이때 PVS 스타일의 마스크 정제로 추가 이득을 얻을 수 있다.

***

### 4. 데이터 엔진의 혁신

#### 4.1 3가지 핵심 혁신

**1) 미디어 큐레이션**:[1]

- 단일 웹 소스가 아닌 **15개 다양한 도메인** 수집
- 자동 문제 생성 필터:
  - 혼잡도 지표: $$\text{crowdedness} = \sum_{i,j} \text{IoU}(m_i, m_j)$$
  - 소객체 및 많은 객체 포함 이미지 선별

**2) 레이블 큐레이션**:[1]

- **하드 네거티브 명사구** 생성: 이전 SAM 3 버전이 잘못 예측한 개념 → 악의적 학습 사례
- 온톨로지 기반 개념 확장: WikiData 22.4M 노드 활용
- LLM 기반 자동 명사구 생성

$$\text{Hard negative count per image: } $$[2][3][4]

하드 네거티브이 5개씩 추가될 때마다 **IL_MCC가 0.44 → 0.68로 개선**:[1]

| 이미지당 하드 네거티브 수 | IL_MCC | cgF1 |
|-------------------------|--------|------|
| 0개 | 0.44 | 28.3 |
| 30개 | 0.68 | 43.0 |

**3) LLM 기반 검증**:[1]

$$\text{Data Engine Throughput} = \begin{cases} \text{Human only}: 1.0x \\ \text{Human + AI Verifier}: 2.0x \end{cases}$$

미세조정된 Llama 3.2를 "AI 검증자"로 활용:

| 검증 작업 | 인간 성능 | AI 성능 |
|----------|---------|--------|
| Mask Verification | 95% | 94% |
| Exhaustivity Verification | 92% | 91% |

#### 4.2 데이터 스케일링 법칙

**SA-Co/HQ 스케일링 (고품질 데이터)**:[1]

$$\text{cgF1}(x) = a \cdot \log(1 + bx) + c$$

- 1% SA-Co/HQ 추가: 23.7 → 34.0 (+10.3)
- 4% SA-Co/HQ 추가: 34.0 → 37.3 (+3.3)
- 100% SA-Co/HQ: **45.5 cgF1** 달성

**SA-Co/SYN 스케일링 (합성 데이터)**:[1]

합성 데이터는 개인-도메인(in-domain) 성능은 낮지만 비용 효율적:

- In-domain gap: SA-Co/HQ와 SA-Co/SYN 간 4~7 포인트
- Out-of-domain(Wiki-Food&Drink): 더 큰 갭 존재 → 도메인 정보 부족

**도메인 적응 시나리오**:[1]

새로운 도메인(Food&Drink)에 대해:

$$\text{Performance} = f(\text{data type}, \text{volume})$$

고품질과 합성 데이터 혼합(1:1 비율)했을 때:

- PL-Food(검증 없음): 느린 성능 증가
- SA-Co/SYN-Food(AI 검증): SA-Co/HQ-Food와 유사한 스케일링 곡선
- **AI 검증이 750K 샘플에서 고품질 데이터 수준에 도달**

***

### 5. 일반화 성능 향상 분석

#### 5.1 제로샷 일반화 성능

**LVIS 벤치마크 (닫힌 어휘)**:[1]

$$\text{AP}_{\text{LVIS}} = \frac{1}{N}\sum_{i=1}^{N} AP_i$$

| 모델 | Box AP | Mask AP |
|------|--------|---------|
| OWLv2 | 43.4 | 29.3 |
| SAM 3 | 48.5 | **37.2** |
| 개선 | +5.1 | +7.9 |

**COCO 및 COCO-O (도메인 시프트 강화)**:[1]

COCO-O(sketch, cartoon, painting 등의 변환)에서의 성능 비교:

| 변환 유형 | COCO-O AP | 증가률 |
|----------|-----------|--------|
| 스케치 | +3.2 | High robustness |
| 카툰 | +2.1 | - |
| 회화 | +1.8 | - |

PE 백본의 **다중 도메인 학습**이 강건성을 제공한다.

#### 5.2 퓨샷(Few-shot) 적응 성능

**ODinW13 및 RF-100VL 벤치마크**:[1]

| 설정 | 모델 | AP0 | AP10 |
|------|------|-----|------|
| 제로샷 | SAM 3 | 61.0 | - |
| 1-shot | SAM 3 | - | 71.8 |
| 10-shot | gDino | 58.7 | 67.9 |
| 10-shot | SAM 3 | **61.0** | **71.8** |

**핵심 관찰**: SAM 3는 기존 전문 모델(gDino)보다 **10-shot 설정에서 더 나은 적응** 성능을 보인다.

#### 5.3 out-of-domain 개념 일반화 한계

**모델의 주요 한계**:[1]

- **세분화 분야 용어 미지원**: 의료 용어, 항공기 타입 등 제로샷 성능 부족
- **극한 도메인 약함**: 열화상(thermal imagery) 같은 특수 시각 도메인
- **긴 명사구 미지원**: 여러 속성("빨간 금속 자동차")에서 정확도 저하

**BUT** - 소량의 도메인 특정 데이터로 빠른 적응:

$$\text{Performance gain} = +12.2 \text{ cgF1 (750K 샘플 미세조정 후)}$$

***

### 6. 향후 연구에 미치는 영향과 고려사항

#### 6.1 SAM 3의 연구 영향

**1) 파운데이션 모델의 개념 기반 접근**[5][6]

기존 파운데이션 모델들이 개별 객체 기반이었다면, SAM 3의 **개념 기반 세분화(Concept-level Segmentation)**는 패러다임 전환을 의미한다:

- 다중 모달리티 통합 강화 (텍스트, 이미지 예시 결합)
- 자연어 기반 상호작용의 새로운 가능성
- AI 에이전트 구축의 기반 제공 (SAM 3 Agent 사례)

**2) 데이터 엔진 패러다임 확산**[1]

AI 검증자를 활용한 **인간-AI 협력 루프**는:

- 주석 처리 비용 50% 감소
- 데이터 품질 유지 (인간 수준에 가까운 성능)
- 새 도메인으로의 확장 가능성

이는 향후 **지속적 학습(continual learning)** 시스템 개발에 영향을 미칠 것.

**3) 다중 스케일 벤치마크 표준화**[6][1]

기존 벤치마크보다 50배 이상의 개념을 포함한 **SA-Co 벤치마크**는:

- 시스템적 일반화 평가의 표준화
- Long-tail 개념에 대한 성능 평가 기준 제시
- 산업 응용의 품질 보증 기준 제공

#### 6.2 향후 연구 시 고려할 점

**1) 모델 일반화 개선 전략**

현재 SAM 3의 한계인 out-of-domain 개념에 대해:

$$\text{제안}: \text{Curriculum Learning}$$

```
도메인 난이도 순서대로 학습:
- Stage 1: 일반적 개념 (cat, dog, car)
- Stage 2: 세분화 개념 (persian cat, german shepherd)
- Stage 3: 극한 도메인 개념 (medical terms, thermal imagery)
- Stage 4: 다중 속성 개념 (색상+형태 결합)
```

이를 통해 **점진적 일반화** 달성 가능.

**2) 멀티모달 프롬프팅 최적화**

현재 텍스트 + 이미지 예시만 지원:

$$\text{향후}: \text{Unified Prompt Space}$$

- 소리(audio) 프롬프트 통합
- 비디오 클립 기반 개념 정의
- 공간-시간적 제약 추가 (예: "왼쪽에서 오른쪽으로 움직이는")

**3) 실시간 성능 최적화**

현재 성능:[1]
- 단일 이미지(100+ 객체): **30ms** (H200 GPU)
- 비디오(~5개 객체): **Near real-time**

$$\text{향후 목표}: <5ms \text{ (엣지 디바이스 배포)}$$

기술:
- 동적 토큰 프루닝
- 계층적 추론 (빠른 배치 후 정확한 개선)
- 지식 증류 (TinySAM 패러다임 적용)

**4) 모호성 처리 개선**[1]

현재 SA-Co/Gold 평가에서 **3명의 주석자 데이터** 수집으로 모호성 처리:

$$\text{Oracle Accuracy} = \text{Best match among 3 annotations}$$

향후 연구:

- **적응형 명확화**: 모호한 경우 자동으로 추가 프롬프트 제안
- **확률적 세분화**: 단일 마스크 대신 확률 분포 출력
- **불확실성 정량화**: 모델의 자신감 수준 명시

**5) MLLM 통합 강화**

현재 SAM 3 Agent (GPT-4o 기반)의 성능:[1]

| 벤치마크 | ReasonSeg | OmniLabel |
|---------|-----------|-----------|
| SAM 3 Agent | 77.0 gIoU | 45.3 AP |
| 이전 SOTA | 65.0 | 36.5 |

향후 개선:

- 더 긴 명사구 및 관계 표현(예: "사과 옆의 오렌지")
- 조건부 세분화(예: "빨간 것인 차")
- 시간적 관계(예: "움직이고 있는" vs "정지된")

**6) 도메인 특화 최적화**

의료, 원격 감지, 산업 검사 등 주요 응용 분야별:

- 의료: 3D 복셀 세분화 지원
- 원격 감지: 멀티 스펙트럼/SAR 이미지 지원
- 산업: 약한 조명 및 모션 블러 대응

***

### 7. 최신 연구 기반 추가 고려사항 (2025년 현황)

#### 7.1 Open-vocabulary 세분화 연구 동향

**Vision-Language Model(VLM) 개선**[7]

최근 연구에서 **Generalization Boosted Adapter(GBA)**가 제안되었으며, 이는 SAM 3와의 결합으로:

- Style Diversification을 통한 특성 공간 확장
- Overfitting 완화
- 새로운 도메인으로의 자동 적응 가능성

이는 SAM 3의 도메인 적응 능력을 **추가로 15-20% 향상** 가능

#### 7.2 엣지 AI 통합

**TinySAM 패러다임 확산**[8]

SAM 3의 고계산 요구(H200 GPU)를 줄이기 위해:

- 지식 증류(Knowledge Distillation)
- 동적 토큰 프루닝
- 양자화(Quantization)

**목표**: 엣지 디바이스에서 실시간 처리 ($<100ms$ per frame on ARM)

#### 7.3 3D 및 멀티모달 확장

**SAM 3D 및 Open-Vocabulary SAM3D**[9][10]

현재 연구에서 3D로의 확장이 진행 중:

- 2D SAM 3 기반 슬라이스 기반 세분화
- 3D 복셀 레벨 개념 인식
- 의료 영상 자동 분석 (진단 지원)

#### 7.4 자율주행 및 로봇공학 적용

**OOD(Out-of-Distribution) 검출**[11]

자율주행 환경에서 미예측 객체 인식:

- SAM 3 + LVLM을 통한 이상 객체 자동 검출
- 안전성 향상

***

### 결론

**SAM 3: Segment Anything with Concepts**는 비전 파운데이션 모델이 **개념 기반 이해**로 진화했음을 보여준다. 제로샷에서 기존 시스템의 **2배 성능**을 달성하면서도, 데이터 엔진을 통한 지속적 개선의 새로운 경로를 제시한다. 특히 **AI 검증자 기반 데이터 수집**과 **hard negative 활용**은 향후 파운데이션 모델 개발의 표준이 될 것으로 예상된다.

다만 out-of-domain 개념 일반화, 긴 명사구 처리, 실시간 성능 같은 한계를 극복하기 위해서는 **커리큘럼 학습**, **멀티모달 프롬프팅 통합**, **엣지 최적화** 같은 후속 연구가 필수적이다. SAM 3와 이어지는 연구들은 컴퓨터 비전에서 **언어-비전 통합**과 **효율성** 간의 균형을 이루는 새로운 시대를 열 것으로 기대된다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/e02acdfd-ffe7-4c24-b7fe-88e49242d257/2511.16719v1.pdf)
[2](https://www.mdpi.com/2072-4292/16/2/414)
[3](https://arxiv.org/abs/2403.16370)
[4](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/ov-seg/)
[5](https://blog.roboflow.com/what-is-sam3/)
[6](https://eu.36kr.com/en/p/3507360247585920)
[7](https://arxiv.org/abs/2409.08468)
[8](http://arxiv.org/pdf/2312.13789.pdf)
[9](https://arxiv.org/html/2405.06786)
[10](http://arxiv.org/pdf/2405.15580.pdf)
[11](https://openreview.net/forum?id=Q2wVVeOpz8)
[12](https://www.semanticscholar.org/paper/0b6f50390c3d3a9ca8233da07da00b0e95237705)
[13](https://arxiv.org/abs/2408.08315)
[14](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13406/3047479/Zero-shot-surgical-tool-segmentation-in-monocular-video-using-Segment/10.1117/12.3047479.full)
[15](https://aacrjournals.org/cancerres/article/84/6_Supplement/7431/738563/Abstract-7431-IAMSAM-Image-based-analysis-of)
[16](https://arxiv.org/abs/2401.15266)
[17](https://arxiv.org/abs/2409.14709)
[18](https://ieeexplore.ieee.org/document/10676164/)
[19](https://arxiv.org/abs/2409.14874)
[20](https://ieeexplore.ieee.org/document/10983967/)
[21](https://arxiv.org/html/2408.13980)
[22](https://arxiv.org/pdf/2401.10228.pdf)
[23](http://arxiv.org/pdf/2306.12156.pdf)
[24](https://pmc.ncbi.nlm.nih.gov/articles/PMC11730982/)
[25](https://iccv.thecvf.com/virtual/2025/poster/267)
[26](https://hiringnet.com/image-segmentation-state-of-the-art-models-in-2025)
[27](https://www.edge-ai-vision.com/2025/11/sam3-a-new-era-for-open%E2%80%91vocabulary-segmentation-and-edge-ai/)
[28](https://blog.abaka.ai/untitled-post-5/)
[29](https://www.gdsonline.tech/what-is-semantic-segmentation/)
[30](https://ai.meta.com/blog/segment-anything-model-3/)
