# ReConPatch : Contrastive Patch Representation Learning for Industrial Anomaly Detection

### 1. 핵심 주장과 주요 기여도 요약[1]

**ReConPatch**는 사전학습된 모델에서 추출한 패치 피처를 선형 변환(linear modulation)으로 조정하여 산업용 이상 탐지의 성능을 향상시키는 방법이다. 이 논문의 핵심 기여는 다음과 같다:

**주요 기여:**
- **데이터 증강 없이도 높은 성능 달성**: 기존 방법(PatchCore)은 임의의 크롭, 회전, 색상 조정 등 신중하게 설계된 입력 증강이 필요하지만, ReConPatch는 이를 최소화하면서도 우수한 성능 유지[1]
- **대조학습 기반 특징 표현 학습**: 쌍별(pairwise) 유사도와 문맥(contextual) 유사도라는 두 가지 의사 레이블(pseudo-label)을 활용하여 비지도 메트릭 학습 수행[1]
- **계산 효율성**: 전체 신경망을 학습하지 않고 선형 변환만 학습하여 실용적인 솔루션 제공[1]
- **최첨단 성능**: MVTec AD 데이터셋에서 99.72 AUROC 달성[1]

***

### 2. 문제 정의, 제안 방법 및 모델 구조[1]

#### 2.1 해결하고자 하는 문제[1]

산업용 이미지에서 이상 탐지는 다음의 어려움을 직면하고 있다:
- **분포 편향(Distribution Shift)**: ImageNet으로 사전학습된 모델은 자연 이미지에 최적화되어 있어 산업용 이미지와의 분포 차이 발생
- **제한된 학습 데이터**: 이상 샘플이 극히 드물어 모델 학습이 어려움
- **미세한 결함 탐지**: 부품 오정렬, 손상 등 미묘한 이상을 구별하기 위해서는 매우 판별력 있는 표현 필요

#### 2.2 제안 방법[1]

**ReConPatch의 핵심 메커니즘:**

ReConPatch는 **쌍별 유사도와 문맥 유사도를 결합한 완화된 대조손실(Relaxed Contrastive Loss)**을 사용한다.

**쌍별 유사도(Pairwise Similarity):**

$$Pairwise_{ij} = e^{-\frac{\|z_i - z_j\|_2^2}{2\sigma^2}} - 1$$

여기서 $z_i = g(f(p_i))$이고, $\sigma$는 가우시안 커널의 대역폭(bandwidth)이다.[1]

**문맥 유사도(Contextual Similarity):**

문맥 유사도는 특징의 k-최근접 이웃들의 교집합을 고려한다:[1]

$$Contextual_{ij} = \begin{cases} \frac{|N_i^k \cap N_j^k|}{|N_i^k|}, & \text{if } j \in N_i^k \\ 0, & \text{otherwise} \end{cases}$$

여기서 $N_i^k = \{j | d_{ij} \leq d_{il}, l은 k번째 최근접 이웃\}$[1].

**쿼리 확장(Query Expansion)을 통한 개선:**

$$R_i^k = \{j | j \in N_i^k \text{ and } i \in N_j^k\}$$

$$Contextual_{ij} = \frac{1}{|R_i^k|^2} \sum_{l \in R_i^k} Contextual_{lj}$$

최종적으로 양방향 유사도를 평균하면:

$$Contextual_{ij} = \frac{1}{2}(Contextual_{ij} + Contextual_{ji})$$

**통합 유사도:**

$$\rho_{ij} = \alpha \cdot Pairwise_{ij} + (1-\alpha) \cdot Contextual_{ij}, \quad \alpha \in $$[1]

**완화된 대조손실:**

$$L_{RC}(z) = \frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{N} \rho_{ij}^2 \rho_{ij} + (1-\rho_{ij})\max(m-\rho_{ij}, 0)^2$$

여기서 $\rho_{ij} = \frac{\|z_i - z_j\|\_2^2}{1 + \frac{1}{N}\sum_{n=1}^{N}\|z_i - z_n\|_2^2}$는 정규화된 거리이고, $m$은 격퇴 마진(repelling margin)이다[1].

**지수이동평균(EMA) 업데이트:**

유사도 계산 네트워크는 표현 학습 네트워크의 EMA로 천천히 업데이트되어 안정적인 학습을 보장한다:[1]

$$\theta_{f',g'} \leftarrow \tau\theta_{f',g'} + (1-\tau)\theta_{f,g}$$

여기서 $\tau$는 모멘텀 초매개변수(hyperparameter)이다.[1]

#### 2.3 모델 구조[1]

ReConPatch는 **두 개의 네트워크 시스템**으로 구성된다:

**1) 표현 학습 네트워크 (하단)**
- **특징 표현 층(f)**: 선형 변환으로 사전학습 특징을 목표 지향적 표현으로 변환
- **사영 층(g)**: 표현을 투영 공간으로 변환

**2) 유사도 계산 네트워크 (상단, $f'$, $g'$)**
- 쌍별 및 문맥 유사도 계산용
- EMA로 업데이트되어 학습 안정성 확보

**추론 단계:**

이상점수는 메모리 뱅크에 저장된 대표 특징(coreset)과의 거리로 계산된다:[1]

$$r^* = \arg\min_{r \in M} D(f(p_t), r)$$

$$s_t = \frac{1}{e^{s_t} + \sum_{r \in N_b^r} e^{-D(f(p_t),r)}} + D(f(p_t), r^*)$$

여기서 $N_b^r$는 메모리 뱅크에서 $r$의 b-최근접 이웃 집합이다.[1]

***

### 3. 성능 향상 및 한계

#### 3.1 성능 향상[1]

| 데이터셋 | 단일 모델 | 앙상블 모델 | 검출 vs 분할 |
|---------|---------|----------|----------|
| **MVTec AD** | 99.56 AUROC | 99.72 AUROC | 99.56 검출 / 98.07 분할 |
| **BTAD** | - | - | 95.8 검출 / 97.5 분할 |

**비교 분석:**
- **PatchCore 대비**: 단일 모델에서 동등 수준, 데이터 증강에 대한 견고성 현저히 우수[1]
- **CFA 대비**: 99.56 vs 99.3 AUROC로 상위 성능
- **PNI 대비**: WideResNet-101 사용 시 99.62와 비슷한 수준의 성능

#### 3.2 데이터 증강 견고성[1]

회전, 색상 변화, 가우시안 블러 등을 적용했을 때:
- **PatchCore**: 99.1 → 95.48 AUROC (3.62 저하)
- **ReConPatch**: 99.56 → 98.56 AUROC (1.0 저하)

**ReConPatch가 3배 이상 견고함을 증명**[1]

#### 3.3 차원 축소 성능[1]

표 2에서 보이듯이, ReConPatch는 차원을 현저히 줄이면서도 성능 유지:
- **1024 차원**: ReConPatch 99.49 vs PatchCore 99.1
- **64 차원**: ReConPatch 99.14 vs PatchCore 97.75 (1.39 포인트 향상)

이는 더 효율적인 메모리 사용과 빠른 추론을 가능하게 한다.[1]

#### 3.4 한계[1]

1. **픽셀 수준 이상 분할 성능**: 검출 성능(99.56)에 비해 분할 성능(98.07)이 낮은 편
2. **개별 클래스 편차**: 일부 클래스(금속 너트, 약제, 가죽)에서 성능이 상대적으로 낮음
3. **이미지 해상도 의존성**: 480×480 이미지 크기에서는 최고 성능을 달성하지만, 표준화된 224×224에서는 PNI에 미치지 못함
4. **인접 특징 간 상관성 미고려**: 논문 결론에서 "이웃한 특징들 간의 상관관계를 고려하여 픽셀 수준 이상 탐지 개선 기대"라고 명시[1]

***

### 4. 모델의 일반화 성능 향상 가능성 (중점)

#### 4.1 현재 일반화 성능의 강점[1]

**1) 분포 이동에 대한 견고성**

표 4의 데이터 증강 실험에서:
- 기존 대조학습의 약점인 "명목 인스턴스 내 변동성 모델링 부족"을 **문맥 유사도**로 해결[1]
- 같은 그룹에 속하는 특징들을 함께 끌어당겨서 정상 범위 내 변동을 학습

**2) 목표 지향적 표현 학습**

선형 변환 $f$는 이미지 공간의 기하학적 정보를 보존하면서 산업 이미지에 맞게 피처를 재배열:[1]
- UMAP 시각화(그림 3)에서 공간적으로 유사한 위치의 특징들이 자동으로 모임
- 이는 위치 정보의 암묵적 학습을 시사[1]

**3) 차원 축소를 통한 일반화 개선**

고차원 피처 공간의 과적합 위험을 낮추면서도:
- 64 차원에서도 99.14 AUROC 달성
- 더 효율적인 메모리 사용으로 더 많은 정상 샘플 저장 가능 → 더 견고한 메모리 뱅크 구성

#### 4.2 MVTEC AD 크로스 클래스 성능 분석[1]

| 클래스 유형 | 평균 성능 | 특징 |
|----------|---------|------|
| **객체 클래스** | 99.44 검출 | 기하학적 특징이 중요 (결함 위치 변화가 적음) |
| **텍스처 클래스** | 99.81 검출 | 미세한 색상/패턴 변화 탐지에 우수 |

ReConPatch가 **목표 지향적 특징을 학습함**으로써 다양한 유형의 결함에 적응[1]

#### 4.3 향후 일반화 성능 향상 방안[1]

**논문 제안:**
> "이웃 특징들 간의 상관관계를 고려하여 픽셀 수준 이상 탐지 성능 개선 기대"[1]

**추가 개선 가능성:**

1. **공간 구조 정보 활용**: 현재는 개별 패치 특징만 고려하지만, 인접 패치 간의 공간적 연관성 모델링으로 논리적 이상(missing/excess elements) 탐지 능력 향상

2. **적응형 문맥 유사도**: $k$ 값을 고정하지 않고 데이터셋의 특성에 따라 동적으로 조정

3. **다중 스케일 문맥 유사도**: 다양한 이웃 크기($k_1, k_2, ...$)에서 문맥 유사도를 계산하여 다양한 규모의 변동 포착

***

### 5. 최신 관련 연구 비교 분석 (2020년 이후)[2][3][4][5][6][7][8][9]

#### 5.1 주요 경쟁 방법들의 비교[8][1]

| 방법 | 발표년 | 아이디어 | MVTec AUROC | 특징 | 한계 |
|------|-------|--------|-----------|------|------|
| **PaDiM[8]** | 2020 | 패치별 가우시안 분포 모델링 | 95.3 | 확률 모델링으로 추론 시 확장성 우수 | 상관성 미고려 |
| **PatchCore[10]** | 2022 | Locally aware 패치 + Coreset 샘플링 | 99.1 | 간단하고 효과적 | 데이터 증강 필수 |
| **CFLOW-AD[11]** | 2022 | 정규화 흐름 + 조건부 흐름 | 98.26 | 조건부 정규화 흐름 사용 | 외부값에 취약 |
| **CFA[12]** | 2022 | 초구(hypersphere) 기반 특징 적응 | 99.3 | 하이퍼스피어 정렬 | 복잡한 학습 과정 |
| **PNI[2]** | 2023 | 위치 및 이웃 정보 활용 | 99.62 | 위치 정보를 명시적으로 활용 | 세밀한 처리 망 필요 |
| **ReConPatch(본 논문)** | 2023 | 쌍별+문맥 유사도 기반 대조학습 | 99.72 | 데이터 증강 불필요, 차원 축소 가능 | 픽셀 수준 분할 성능 상대적 저하 |

#### 5.2 2020-2024 연도별 연구 동향[3][5][6][7][9][2]

**2020-2021: 기초 확립 단계**
- **PaDiM (2020)**: 사전학습 CNN으로부터 패치별 가우시안 분포 모델링 제시, 확장성과 성능의 균형 달성[8]
- **SPADE (2020)**: 계층적 특징 비교 메커니즘 도입

**2022: 고성능 방법 경쟁 시대**
- **PatchCore**: 메모리 효율적인 coreset 샘플링으로 높은 성능 달성[1]
- **CFLOW-AD**: 정규화 흐름을 활용한 확률적 접근
- **CFA**: 초구 기하학 활용[1]

**2023-2024: 특수화 및 보완 단계**
- **PNI (2023)**: 위치와 이웃 정보의 명시적 활용으로 99.62 달성[1]
- **ULSAD (2024)**: 구조적 이상(structural)과 논리적 이상(logical) 동시 탐지[5][7]
- **AnomalyDINO (2024)**: DINOv2 백본을 활용한 소량-샷(few-shot) 학습에서 96.6% 달성[9]
- **M3DM-NR (2024)**: RGB-3D 멀티모달 노이즈 견고 탐지[4]
- **DBAD (2024)**: 이중 분기(dual branch) 재구성을 통한 일반화 능력 제어[3]

#### 5.3 기술적 혁신 분석

**문맥 인식의 진화:**
- PatchCore: 메모리 뱅크 활용
- ReConPatch: **문맥 유사도로 이웃 관계 명시화** ← 새로운 기여[1]
- PNI: 위치 인코딩 + 이웃 정보
- ULSAD: 논리적 이상까지 포함한 통합 모델

**표현 학습 방법의 진화:**
- 초기: 거리 기반 접근(PaDiM, PatchCore)
- 중기: 기하학적 제약(CFA, 초구)
- **현재: 대조학습 기반 목표 지향적 표현(ReConPatch, Self-Supervised Models)**

***

### 6. 향후 연구에 미치는 영향 및 고려사항

#### 6.1 학문적 영향[6][1]

**1) 대조학습의 산업 적용 확대**

ReConPatch의 성공은 **대조학습을 명목(normal) 샘플 분포의 미세 구조를 모델링하는 도구**로 재정의:
- 기존 대조학습: 긍정 쌍 vs 부정 쌍의 이진 구분
- ReConPatch: **이웃 정보를 활용한 부드러운 의사 레이블** → 더 정교한 표현 학습

**2) 무레이블 메트릭 학습의 발전**

완화된 대조손실(Relaxed Contrastive Loss)은 다음 분야로 확장 가능:
- 이상 이미지 검색(Anomaly Image Retrieval)
- 원샷/소량-샷 학습(One-shot/Few-shot Learning)
- 도메인 적응(Domain Adaptation)

#### 6.2 실제 응용 관점[6][1]

**1) 산업 4.0 친화적 설계**

- **데이터 증강 최소화**: 제조 환경에서 도메인 전문가의 데이터 증강 설계 부담 제거
- **실시간 배포 용이성**: 선형 변환만 학습하므로 경량 모델로 가능
- **적응형 재학습**: 새로운 제품 클래스에 대해 빠른 미세조정(fine-tuning) 가능

**2) 메모리 효율성**

64 차원에서도 99.14 AUROC 달성 → 엣지 디바이스(edge device)에서도 배포 가능

#### 6.3 후속 연구 시 고려할 점[6][1]

**1) 픽셀 수준 이상 분할 성능 개선**

논문의 명시적 제안:
- **공간 상관성 모델링**: 인접 패치 특징 간의 상관관계를 손실 함수에 포함
- **다중 해상도 특징**: 다양한 수준의 공간 정보 통합

**추천 구현:**
$$L_{spatial} = L_{RC} + \lambda \sum_{i} \sum_{j \in N_i} \|p_i - p_j\|_2^2$$

여기서 $N_i$는 패치 $i$의 공간적 이웃 집합[1]

**2) 다양한 산업 도메인 검증**

현재: MVTec AD(15개 클래스), BTAD(3개 클래스)
- **향후**: 의료 영상, 반도체 검사, 식품 검사 등 도메인별 성능 검증
- **도메인 적응**: 소수 대상 도메인 샘플로 제로-샷 또는 소량-샷 적응[7]

**3) 강건성 평가 강화**

표 4의 데이터 증강은 단순한 기하학적 변환만 포함:
- **조명 변화**: 자동 노출 조정, 섀도우 추가
- **초점 흐림**: 심도 변화
- **격자 무늬(Moiré) 현상**: 카메라 센서 특성
- **노이즈**: 실제 센서 노이즈 (가우시안만이 아닌 다양한 타입)

**4) 계산 복잡도 분석**

현재 논문(표 S6):
- PatchCore: 37.89 ms
- ReConPatch: 38.09 ms

향후 연구 방향:
- 메모리 접근 최적화로 추론 시간 단축
- 문맥 유사도 계산의 병렬화[1]

***

### 7. 최신 동향과 ReConPatch의 위치 (2023-2025)[2][4][5][7][9][3][6]

#### 7.1 멀티모달 접근[4]

**M3DM-NR (2024)**: RGB + 3D 포인트 클라우드 + 텍스트 활용
- ReConPatch는 RGB 이미지만 고려
- **확장 가능성**: 깊이 정보를 추가 채널로 포함하여 3D 이상 탐지 강화

#### 7.2 구조적 vs 논리적 이상 분류[5][7]

**ULSAD (2024)**:
- **구조적 이상**: 스크래치, 들여쓰기 (ReConPatch가 잘 탐지)
- **논리적 이상**: 부품 누락, 초과 (ReConPatch의 약점)

**ReConPatch 개선 방안**:
전역 컨텍스트 정보를 통합한 하이브리드 손실:
$$L_{hybrid} = L_{RC} + \lambda L_{global}$$

여기서 $L_{global}$은 이미지 전체의 부품 배치를 모델링[5]

#### 7.3 소량-샷 및 제로-샷 학습[9]

**AnomalyDINO (2024)**: DINOv2 + Few-shot
- 원샷(1-shot): 93.1% → 96.6%

**ReConPatch와의 비교**:
- ReConPatch는 충분한 정상 샘플이 필요 (지도 학습 없이도 많은 데이터 활용)
- 향후: 소량의 정상 샘플로도 학습 가능하도록 메타러닝(Meta-learning) 적용 고려[9]

***

### 결론

**ReConPatch**는 산업용 이상 탐지 분야에서 **대조학습과 문맥 정보의 결합**으로 새로운 기준을 제시했다. 특히 **데이터 증강 불필요성**과 **차원 축소 내성**은 실제 산업 배포에 중대한 장점이다.

**향후 연구의 핵심 과제:**
1. 공간 상관성을 통한 픽셀 수준 분할 성능 개선
2. 논리적 이상 탐지로의 확장
3. 멀티모달 정보 통합
4. 소량-샷 학습으로의 적응

이러한 방향들이 해결된다면, ReConPatch는 **Industry 5.0 시대의 지능형 품질 검사 시스템**의 핵심 기술로 자리 잡을 것으로 예상된다.[6][1]

***

### 참고문헌 (논문 내 인용 기준)

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/49607972-5053-4e6c-935c-b3b288e78138/2305.16713v3.pdf)
[2](https://ieeexplore.ieee.org/document/10710816/)
[3](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ell2.13289)
[4](https://ieeexplore.ieee.org/document/11091585/)
[5](https://arxiv.org/abs/2410.16255)
[6](https://arxiv.org/html/2501.11310v1)
[7](https://arxiv.org/html/2410.16255v1)
[8](https://arxiv.org/pdf/2011.08785.pdf)
[9](https://arxiv.org/html/2405.14529v3)
[10](https://pdfs.semanticscholar.org/6aba/5e4cc448dcf72c1031779eb59a2229f3e836.pdf)
[11](https://www.mdpi.com/1424-8220/23/3/1310/pdf?version=1674534674)
[12](https://premierscience.com/wp-content/uploads/2025/10/5-pjs-25-1320.pdf)
[13](https://ieeexplore.ieee.org/document/9136901/)
[14](https://wjps.uowasit.edu.iq/index.php/wjps/article/view/598)
[15](https://ieeexplore.ieee.org/document/10737508/)
[16](https://ieeexplore.ieee.org/document/10391270/)
[17](https://link.springer.com/10.1007/s10010-024-00765-z)
[18](https://arxiv.org/abs/2405.08349)
[19](http://arxiv.org/pdf/2404.17925.pdf)
[20](https://arxiv.org/pdf/2201.07284.pdf)
[21](http://arxiv.org/pdf/2404.18525.pdf)
[22](https://arxiv.org/pdf/2301.11514.pdf)
[23](https://www.mdpi.com/1424-8220/24/10/3244/pdf?version=1716200396)
[24](https://www.sciencedirect.com/science/article/abs/pii/S2352467725000219)
[25](https://github.com/M-3LAB/awesome-industrial-anomaly-detection)
[26](https://ffighting.net/deep-learning-paper-review/anomaly-detection/patchcore/)
[27](https://www.sciencedirect.com/science/article/abs/pii/S0950705125003958)
[28](https://www.sciencedirect.com/science/article/abs/pii/S0263224125017361)
[29](https://ffighting.net/deep-learning-paper-review/anomaly-detection/padim/)
[30](https://www.ijcai.org/proceedings/2022/0330.pdf)
[31](https://www.lgresearch.ai/blog/view?seq=401)
[32](https://pubmed.ncbi.nlm.nih.gov/41184405/)
[33](https://arxiv.org/html/2507.22659v1)
[34](https://arxiv.org/pdf/2403.14233.pdf)
[35](https://arxiv.org/html/2404.11269v4)
[36](https://arxiv.org/html/2508.12230v1)
[37](https://arxiv.org/html/2511.05245v1)
[38](https://arxiv.org/html/2501.05130v5)
[39](https://arxiv.org/html/2501.09239v1)
[40](https://proceedings.neurips.cc/paper_files/paper/2023/file/1700ad4e6252e8f2955909f96367b34d-Paper-Conference.pdf)
[41](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1508821/full)
[42](https://arxiv.org/abs/2011.08785)
[43](https://www.scribd.com/document/785068024/2011-08785v1)
[44](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Unilaterally_Aggregated_Contrastive_Learning_with_Hierarchical_Augmentation_for_Anomaly_Detection_ICCV_2023_paper.pdf)
