# VOLO: Vision Outlooker for Visual Recognition

### 1. 핵심 주장과 주요 기여

#### 1.1 핵심 주장

VOLO(Vision Outlooker)의 핵심 주장은 **Vision Transformers(ViTs)가 미세 수준의 특징(fine-level features)을 토큰 표현에 인코딩하는 효율성이 낮다**는 것이다. 이를 해결하기 위해 본 논문은 두 가지 주요 관찰을 제시한다:[1]

1. **ViT의 성능 제약 요인**: 기존 ViT는 자기주의(self-attention)를 통해 전역 종속성을 모델링하는 데 초점을 맞추지만, 이는 거시 수준의 특징 인코딩에만 효과적이며 **미세 수준의 공간 정보**를 충분히 활용하지 못한다.

2. **Outlook Attention의 제안**: 자기주의의 쿼리-키 행렬 곱셈 대신, **선형 투영과 재형성 연산**을 통해 인접한 토큰들의 가중 평균을 효율적으로 계산하는 새로운 주의 메커니즘을 도입한다.[1]

#### 1.2 주요 기여도

- **최초 87% 이상 정확도 달성**: ImageNet-1K 분류에서 추가 학습 데이터 없이 87.1% 정확도를 달성한 첫 번째 모델[1]
- **효율성 우수성**: 296M 파라미터로 NFNet-F6(438M) 대비 적은 매개변수로 더 우수한 성능 달성[1]
- **강력한 일반화 성능**: ImageNet-ReaL(90.6%), ImageNet-V2(78.0%)에서 기존 모델 초과[1]
- **다운스트림 작업 전이 우수성**: Cityscapes(84.3%), ADE20K(54.3%)에서 기존 SOTA 모델 초과[1]

***

### 2. 해결하고자 하는 문제와 근본 원인 분석

#### 2.1 문제의 정의

Vision Transformer의 등장으로 CNN의 지배에서 벗어났음에도, **추가 학습 데이터 없이 최신 CNN 모델을 능가하지 못하는 성능 격차**가 존재했다:[1]

- CaiT-M48: 86.5% (ImageNet-1K)
- NFNet-F5: 86.8% (SAM, augmult 사용)
- **성능 격차: 약 0.3~0.8%**

#### 2.2 근본 원인 분석

논문이 규명한 ViT의 근본적 한계:[1]

1. **토큰화 해상도 문제**: 16×16 패치를 사용하여 14×14 토큰만 생성되어 미세한 공간 정보 손실
2. **Self-attention의 비효율성**: 자기주의는 전역적 종속성에 최적화되었으나, 미세 수준의 특징 인코딩에는 부적절
3. **계산 복잡성 증가**: 더 작은 패치(예: 8×8)를 사용할 경우 토큰 수가 4배 증가하여 self-attention 복잡도가 제곱 배 증가

***

### 3. 제안하는 방법 및 수식

#### 3.1 Outlook Attention 메커니즘

Outlook Attention은 각 공간 위치에서 **인접한 토큰들의 가중 평균**을 계산하는 방식으로 미세 수준의 특징을 인코딩한다.[1]

**Step 1: 값 및 가중치 임베딩**[1]

$$V = W^V X, \quad A = W^A X$$

여기서:
- $X \in \mathbb{R}^{H \times W \times C}$: 입력 토큰 표현
- $W^V \in \mathbb{R}^{C \times C}$: 값 투영 행렬
- $W^A \in \mathbb{R}^{C \times K^4}$: Outlook 가중치 투영 행렬

**Step 2: 로컬 윈도우 내 값 추출**[1]

$$V^{\Delta_{i,j}} = \{V_{i+p-\lfloor K/2 \rfloor, j+q-\lfloor K/2 \rfloor}\}, \quad 0 \leq p, q < K$$

위치 $(i, j)$ 중심의 $K \times K$ 윈도우 내 모든 값을 추출한다.

**Step 3: Outlook 가중치 재형성 및 소프트맥스**[1]

$$\hat{A}_{i,j} = \text{Reshape}(A_{i,j}) \in \mathbb{R}^{K^2 \times K^2}$$

$$Y^{\Delta_{i,j}} = \text{MatMul}(\text{Softmax}(\hat{A}_{i,j}), V^{\Delta_{i,j}})$$

Outlook 가중치 벡터를 $K^2 \times K^2$ 행렬로 재형성하고 소프트맥스를 적용한다.

**Step 4: Dense Aggregation (중첩 윈도우 합산)**[1]

$$\tilde{Y}_{i,j} = \sum_{0 \leq m,n < K} Y_{i,j}^{\Delta_{i+m-\lfloor K/2 \rfloor, j+n-\lfloor K/2 \rfloor}}$$

각 위치에서 자신을 중심으로 하는 모든 $K \times K$ 윈도우로부터 출력을 수집하여 합산한다.

#### 3.2 계산 복잡도 비교

**Self-Attention (SA) 계산량:**[1]

$$\text{M-Adds(SA)} \approx 4HWC^2 + 2(HW)^2C$$

**Local Self-Attention (LSA) 계산량:**[1]

$$\text{M-Adds(LSA)} \approx 4HWC^2 + 2HWK^2C$$

**Outlook Attention (OA) 계산량:**[1]

$$\text{M-Adds(OA)} \approx HWC(2C + NK^4) + HWK^2C$$

일반적 경우($C = 384, K = 3, N = 6$): $NK^4 = 486 < 2C = 768$이므로 **OA가 LSA보다 계산 효율적**이다.[1]

#### 3.3 Outlooker 블록

$$\tilde{X} = \text{OutlookAtt}(\text{LN}(X)) + X \quad (1)$$

$$Z = \text{MLP}(\text{LN}(\tilde{X})) + \tilde{X} \quad (2)$$

Layer Normalization을 거친 입력에 Outlook Attention을 적용하고 잔여 연결을 추가한다.[1]

***

### 4. 모델 구조

#### 4.1 두 단계 아키텍처[1]

**Stage 1: Fine-level 특징 인코딩 (28×28 해상도)**
- 입력: 224×224 이미지
- 패치 임베딩: 8×8 패치 → 28×28 토큰
- 주요 모듈: Outlooker 블록 다층 스택
- 목표: 미세 수준의 공간 정보 인코딩

**Stage 2: Global 정보 집계 (14×14 해상도)**
- 다운샘플링: 2×2 패치 임베딩
- 주요 모듈: Transformer 블록 스택
- 목표: 전역 종속성 모델링

#### 4.2 아키텍처 설계 원칙[1]

1. **1:3 비율 유지**: Outlooker와 Transformer의 비율을 1:3으로 유지
2. **계층적 구조**: CNN과 유사한 다단계 구조로 계산 부하 분산
3. **Class Attention**: 최종 단계에 2개의 클래스 어텐션 층 추가
4. **숨은 차원 설정**: Outlooker의 숨은 차원을 Transformer의 절반으로 설정

***

### 5. 성능 향상 분석

#### 5.1 ImageNet-1K 분류 성능[1]

| 모델 | 파라미터 | FLOPs | Top-1 |
|------|---------|-------|-------|
| CaiT-M48 | 356M | 330B | 86.5% |
| NFNet-F5 | 377M | 290B | 86.8% |
| **VOLO-D5** | **296M** | **304B** | **87.1%** |

**주요 성과:**
- 이전 SOTA 대비 0.3% 향상 (87.1% vs 86.8%)[1]
- 26% 적은 파라미터 (296M vs 438M for NFNet-F6)[1]
- ImageNet-ReaL: 90.6%[1]
- ImageNet-V2: 78.0%[1]

#### 5.2 Outlooker 기여도 분석[1]

| 단계 | 변경 사항 | Top-1 | 향상도 |
|------|---------|-------|--------|
| 기준 | LV-ViT-S (16×16 패치) | 83.3 | - |
| +1 | 2개 Outlooker 추가 | 83.7 | +0.4% |
| +2 | 4개 Outlooker 추가 | 84.0 | +0.7% |
| +3 | Transformer 헤드 증가 | 84.2 | +0.9% |
| +4 | 해상도 확대 (224→384) | 85.2 | +1.9% |

**비교 분석:**
- **Outlook Attention vs Local Self-Attention**: 84.2% vs 83.8% (+0.4%)[1]
- **Outlook Attention vs Convolution**: 84.2% vs 83.8% (+0.4%)[1]

***

### 6. 일반화 성능 향상 분석

#### 6.1 ImageNet-ReaL 벤치마크[1]

ImageNet-ReaL은 라벨 오류를 수정한 벤치마크로 모델의 진정한 일반화 능력을 평가한다:

| 모델 | Top-1 (Original) | Top-1 (ReaL) |
|------|-----------------|-------------|
| VOLO-D5 | 87.1 | 90.6 |
| CaiT-M48 | 86.5 | 90.2 |

VOLO는 **더 강력한 일반화 능력**을 보유한다.[1]

#### 6.2 ImageNet-V2 성능 (분포 이동 저항)[1]

ImageNet-V2는 새롭게 수집된 이미지로 분포 이동 저항성을 평가한다:

| 모델 | ImageNet-1K | ImageNet-V2 |
|------|------------|-----------|
| VOLO-D3 | 85.4 | 75.6 |
| VOLO-D5 | 87.1 | 78.0 |

VOLO는 **분포 이동에 강건**하며, 미세 수준의 특징 인코딩이 일반화 성능 향상에 기여한다.[1]

#### 6.3 일반화 성능 향상의 기제

1. **미세 수준 특징의 역할**: Outlooker를 통한 8×8 패치 수준의 공간 정보 포착이 지역 구조 학습 강화
2. **Dense Aggregation**: 중첩 윈도우를 통한 밀집 집계가 공간 연속성 학습에 도움
3. **계층적 구조**: 두 단계 설계로 미세와 전역 정보를 균형 있게 활용

***

### 7. 모델의 한계

#### 7.1 계산 효율성 한계[1]

1. **학습 시 높은 계산 비용**: VOLO-D5 학습에 2개 노드(16-32 GPU) 필요, 300 에포크 학습에 상당한 시간 소요
2. **메모리 요구량**: 배치 크기 1024 유지로 인한 고메모리 사용, 대규모 모델 확장 시 메모리 초과 위험

#### 7.2 구조적 한계[1]

1. **패치 크기 고정**: 8×8 패치로 고정되어 다른 해상도에 대한 적응성 부족
2. **Outlooker 헤드 수 포화**: 표 8에서 보듯이 헤드 수 증가 시 성능 포화 (6개 이상 효과 미미)
3. **Two-stage 설계의 유연성 부족**: Stage 비율(1:3)이 경험적으로 최적화되었을 뿐 일반성 부족

#### 7.3 데이터 의존성[1]

1. **Token Labeling 필요성**: Token Labeling이 성능 향상에 필수적
2. **Stochastic Depth 의존**: 큰 모델에서 과적합 방지를 위해 높은 stochastic depth 필요

***

### 8. 최신 연구에 미치는 영향 (2022-2025)

#### 8.1 VOLO의 기여 및 영향

**아키텍처 하이브리드 트렌드 확립:**[2][3]
- CNN의 강점(지역 특징 추출)과 Transformer의 강점(전역 의존성)을 결합하는 하이브리드 아키텍처 설계의 중요성 입증
- CMT, Next-ViT 등 후속 연구에 영향

**계층적 구조의 재조명:**[4][5]
- ConvNeXt(2022)에서 VOLO의 다단계 설계 원리를 참고하여 현대식 CNN 구현
- Swin Transformer와 유사한 계층적 접근 강화

**효율성 중시 흐름 형성:**[6][7]
- Trio-ViT, Adaptive Token Sampling, LF-ViT 등에서 VOLO의 효율적 주의 메커니즘 참고

#### 8.2 후속 연구 방향 (2022-2025)

**효율성 개선 연구:**

- **EfficientViT (2023)**: Cascaded Group Attention 도입하여 MHSA 층의 계산 중복성 제거[8]
- **LightViT (2022)**: Convolution 없는 경량 ViT 설계, Outlook과 유사한 효율적 집계 방식 도입[9]
- **Trio-ViT (2024)**: Softmax-free 효율적 Vision Transformer 제시[6]

**Hybrid 아키텍처 발전:**

- **CMT (2022)**: CNN의 Convolutional Stem과 Transformer Block 결합[3]
- **Next-ViT (2022)**: ViT와 CNN 하이브리드로 실제 산업 배포 고려[2]
- **ConvNeXt (2022)**: VOLO의 계층적 구조에서 영감받아 현대식 CNN 구현, ImageNet: 87.8% 달성[5][4]

**Robustness 및 일반화 연구:**

- **Towards Robust Vision Transformer (2022)**: ViT의 구성 요소 분석으로 견고성 향상[10]
- **Vision-Language Models (2023-2024)**: VOLO와 같은 효율적 백본을 기반으로 다중모달 학습 확대[11][12]

***

### 9. 향후 연구 시 고려할 점

#### 9.1 기술적 개선 사항

1. **동적 해상도 처리**: 고정 8×8 패치 대신 적응형 패치 크기 설계
2. **Adversarial Robustness 강화**: Outlook Attention의 견고성 검증 필요
3. **해석성 개선**: Outlook Attention의 시각화 기법 개발

#### 9.2 효율성 최적화

1. **컴파일 및 양자화**: INT8/INT4 양자화 연구 (Trio-ViT 참고)
2. **토큰 프루닝**: Adaptive Token Sampling 활용
3. **분산 학습 최적화**: 효율적인 그래디언트 동기화

#### 9.3 응용 분야 확대

1. **고해상도 이미지 처리**: 4K, 8K 이미지 처리 능력 향상
2. **비디오 인식**: 시간 정보 통합 설계
3. **의료/과학 분야**: 의료 이미지의 세밀한 특징 포착[13]

***

### 10. 결론

VOLO는 **Vision Transformer의 한계를 규명하고, 간단하면서도 효과적인 해결책을 제시**함으로써 비전 인식 분야에 중대한 기여를 했다.[1]

**학술적 의의:**
- 아키텍처 설계의 새로운 패러다임: CNN과 Transformer 하이브리드의 타당성 입증
- 효율성의 중요성: 계산 복잡도 감소가 성능 유지와 양립 가능함을 보여줌
- 미세 수준 특징의 가치: 거시 수준 모델링만큼이나 중요함을 강조

**향후 전망:**
- **단기(2024-2025)**: 효율성 극대화, 경량화, 모바일 배포
- **중기(2025-2027)**: 멀티모달 통합, 비디오 처리, 의료 응용 확대
- **장기(2027+)**: 통합 비전 파운데이션 모델의 핵심 기술로 자리매김

Outlook Attention은 이후 연구의 **효율성 개선 및 하이브리드 아키텍처 설계의 중요한 참고사항**이 되었으며, 비전 인식 기술의 지속적 발전을 견인하고 있다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b007e70e-38b8-437c-857d-0e14d96f5a94/2106.13112v2.pdf)
[2](http://arxiv.org/pdf/2207.05501.pdf)
[3](https://arxiv.org/pdf/2107.06263.pdf)
[4](https://beelinekim.tistory.com/90)
[5](https://velog.io/@yellofi/2022-CVPR-A-ConvNet-for-the-2020s-ConvNeXt)
[6](http://arxiv.org/pdf/2405.03882.pdf)
[7](http://arxiv.org/pdf/2111.15667.pdf)
[8](https://velog.io/@softwarerbfl/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-EfficientViT-Memory-Efficient-Vision-Transformer-With-Cascaded-Group-Attention)
[9](https://arxiv.org/pdf/2207.05557.pdf)
[10](https://blog.outta.ai/151)
[11](https://velog.io/@sksmslhy/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-Multimodal-Learning-with-Transformers-A-Survey)
[12](https://arxiv.org/html/2403.09394v1)
[13](https://pmc.ncbi.nlm.nih.gov/articles/PMC10157024/)
[14](https://arxiv.org/pdf/2106.13112.pdf)
[15](https://pmc.ncbi.nlm.nih.gov/articles/PMC11443922/)
[16](https://pmc.ncbi.nlm.nih.gov/articles/PMC11071258/)
[17](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362238/)
[18](https://pmc.ncbi.nlm.nih.gov/articles/PMC8640772/)
[19](https://www.mdpi.com/2075-4418/14/2/121/pdf?version=1704423724)
[20](https://www.mdpi.com/2072-4292/17/1/162)
[21](https://ar5iv.labs.arxiv.org/html/2106.13112)
[22](https://discuss.pytorch.kr/t/2025-07-14-20-ai-ml/7287)
[23](https://arxiv.org/abs/2106.13112)
[24](https://rahites.tistory.com/373)
[25](https://pubmed.ncbi.nlm.nih.gov/36094970/)
[26](https://hyunseo-fullstackdiary.tistory.com/419)
[27](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE09874346)
[28](https://kalelpark.tistory.com/206)
[29](https://www.kisdi.re.kr/report/fileView.do?key=m2101113025377&arrMasterId=4333446&id=1150337)
[30](http://arxiv.org/pdf/2205.13535.pdf)
[31](https://arxiv.org/pdf/2402.00033.pdf)
[32](https://blog.outta.ai/113)
[33](https://www.emergentmind.com/topics/hybrid-cnn-transformer-backbone-architecture)
[34](https://velog.io/@conel77/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0CMT-Convolutional-Neural-Networks-Meet-Vision-Transformers)
[35](https://sjkoding.tistory.com/77)
[36](https://lifestyleimformation.tistory.com/entry/%EB%94%A5%EB%9F%AC%EB%8B%9D%EC%9D%98-%EC%A0%95%EC%88%98-CNN-RNN-%EA%B7%B8%EB%A6%AC%EA%B3%A0-Transformer%EC%9D%98-%EB%B9%84%EA%B5%90-%EB%B6%84%EC%84%9D)
[37](https://wikidocs.net/236136)
[38](https://dlgari33.tistory.com/28)
[39](https://www.themoonlight.io/ko/review/d-trattunet-toward-hybrid-cnn-transformer-architecture-for-generic-and-subtle-segmentation-in-medical-images)
