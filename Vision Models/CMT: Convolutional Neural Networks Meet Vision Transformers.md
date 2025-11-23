# CMT: Convolutional Neural Networks Meet Vision Transformers

### 1. 핵심 주장 및 주요 기여

**CMT 논문의 핵심 주장**은 **CNN의 지역 특징 추출 능력과 Vision Transformer의 장거리 의존성 모델링 능력을 결합하면, 순수한 ViT나 고성능 CNN 모델보다 우수한 성능을 달성할 수 있다**는 것입니다.[1]

**주요 기여:**

- **하이브리드 아키텍처 제안**: CNN과 Transformer의 강점을 조합한 CMT 아키텍처 개발
- **효율성 개선**: CMT-S는 DeiT 대비 14배, EfficientNet 대비 2배 적은 FLOPs로 더 높은 성능 달성 (83.5% vs 79.8%, 80.0%)
- **다양한 스케일 지원**: CMT-Ti, CMT-XS, CMT-S, CMT-B 등 4개 모델 제공
- **다중 작업 일반화**: 이미지 분류, 객체 검출, 인스턴스 세그멘테이션 등 다양한 작업에서 강력한 성능 증명

***

### 2. 해결하고자 하는 문제 및 제안 방법

**문제점 분석:**

Vision Transformer의 성능이 CNN에 미치지 못하는 이유를 세 가지로 분석:[1]

1. **Patch 기반 입력의 한계**: 이미지의 2D 구조와 로컬 정보를 무시하고, Patch 내 구조 정보를 제대로 모델링하지 못함
2. **다중 스케일 특징 추출 어려움**: 고정된 Patch 크기로 인한 저해상도 및 다중 스케일 특징 추출의 부재
3. **높은 계산 복잡도**: Self-Attention의 $$O(N^2C)$$ 복잡도 vs CNN의 $$O(NC^2)$$ 복잡도

**제안 방법:**

#### (1) **Local Perception Unit (LPU)**

Depth-wise Convolution을 사용하여 로컬 정보 추출:[1]

$$\text{LPU}(X) = \text{DWConv}(X) + X$$

여기서 $$X \in \mathbb{R}^{H \times W \times d}$$이고, $$H$$, $$W$$는 해상도, $$d$$는 특징 차원입니다.

#### (2) **Lightweight Multi-head Self-Attention (LMHSA)**

Depth-wise Convolution을 활용한 Key, Value 축소:[1]

$$\text{LightAttn}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + B\right)V$$

여기서:
- $$K = \text{DWConv}_K(K) \in \mathbb{R}^{n/k^2 \times d_k}$$
- $$V = \text{DWConv}_V(V) \in \mathbb{R}^{n/k^2 \times d_v}$$
- $$B \in \mathbb{R}^{n \times n/k^2}$$는 학습 가능한 상대 위치 편향

계산 복잡도는 $$O(2nd^2(1 + \frac{1}{k^2}) + \frac{2n^2d}{k^2})$$로 감소합니다.

#### (3) **Inverted Residual Feed-Forward Network (IRFFN)**

MobileNetV2의 역잔차 블록 개념을 활용:[1]

$$\text{IRFFN}(X) = \text{Conv}(F(\text{Conv}(X, \theta_1), \theta_2))$$

여기서:
$$F(X) = \text{DWConv}(X) + X$$

***

### 3. 모델 구조

**CMT 블록의 구성:**[1]

$$Y_i = \text{LPU}(X_{i-1})$$
$$Z_i = \text{LMHSA}(\text{LN}(Y_i)) + Y_i$$
$$X_i = \text{IRFFN}(\text{LN}(Z_i)) + Z_i$$

**전체 아키텍처:**

1. **Convolutional Stem**: 3×3 Conv 3개로 초기 특징 추출 (해상도 1/4 감소)
2. **4단계 구조 (Stage-wise)**: CNN과 유사하게 4개 단계로 점진적 해상도 감소 및 채널 확장
   - Stage 1: 해상도 56×56, 채널 46-76
   - Stage 2: 해상도 28×28, 채널 92-152
   - Stage 3: 해상도 14×14, 채널 184-304 (최대 블록 수 16)
   - Stage 4: 해상도 7×7, 채널 368-608
3. **Global Average Pooling**: ViT의 Class Token 대체
4. **분류 헤드**: 1000-way Softmax 분류층

**Stage 간 연결:**
$$\text{Patch Embedding} = \text{LN}(\text{Conv}_{2 \times 2, \text{stride}=2}(X))$$

***

### 4. 성능 향상 및 한계

**성능 향상:**

| 데이터셋 | 모델 | 정확도 | FLOPs | 비고 |
|---------|------|-------|-------|------|
| ImageNet[1] | CMT-S | 83.5% | 4.0B | DeiT-S 대비 +3.7%, EfficientNet-B4 대비 +0.6% |
| CIFAR-10[1] | CMT-S | 99.2% | 4.0B | 우수한 전이학습 성능 |
| CIFAR-100[1] | CMT-S | 91.7% | 4.0B | - |
| COCO Detection[1] | CMT-S | 44.3 mAP | 231B | PVT 대비 +3.9 mAP |

**Ablation Study 결과:**[1]
- Stem 추가: +0.5% (81.9% → 81.4%)
- LPU 추가: +0.8% (82.7%)
- Shortcut 제거: -0.7% (82.0%)
- IRFFN 추가: +0.6% (83.3%)
- Projection + Normalization: +0.2% (83.5%)

**한계:**

1. **배치 정규화와 계층 정규화의 혼합**: 모든 LN을 BN으로 대체하면 모델이 수렴하지 않으며, 모든 BN을 LN으로 대체하면 성능이 83.0%로 저하됨[1]
2. **스케일링 전략의 한계**: 깊이만 확장하면 오히려 성능이 악화 (83.4%)되어 균형잡힌 스케일링이 필수적
3. **상대 위치 편향의 전이성**: 다른 해상도로 미세조정 시 Bicubic 보간이 필요하여 추가 복잡성 발생
4. **메모리 효율성**: 여전히 고해상도 입력 처리 시 메모리 부담 존재

***

### 5. 일반화 성능 향상 분석

**CMT의 일반화 우수성의 원인:**

#### (1) **로컬-글로벌 특징의 균형**

DWConv를 통한 로컬 특징 추출과 Self-Attention의 글로벌 모델링 결합으로, 과적합 위험 감소

#### (2) **다중 스케일 특징 학습**

Stage-wise 구조를 통해 여러 해상도에서 특징 학습:
- Stage 1: 미세한 로컬 특징 (56×56)
- Stage 3: 중간 단계 특징 (14×14)  
- Stage 4: 추상적 글로벌 특징 (7×7)

#### (3) **전이학습에서의 강점**

전이 학습 결과:[1]

| 데이터셋 | CMT-S | EfficientNet-B7 | DeiT-B |
|---------|-------|-----------------|--------|
| Flowers | 98.7% | 98.8% | 98.4% |
| Pets | 95.2% | 95.4% | - |
| Cars | 94.4% | 94.7% | 92.1% |

CMT는 **9배 적은 FLOPs**로 EfficientNet-B7과 유사한 성능 달성

#### (4) **위치 정보 보존**

- ViT: 고정된 절대 위치 인코딩 (평행이동에 불변성 위반)
- CMT: 상대 위치 편향 $$B$$를 사용하여 평행이동-불변성 유지

***

### 6. 최신 연구 동향 및 영향 (2024-2025)

**CMT 이후의 연구 발전:**

#### (1) **하이브리드 모델의 일반화**

2024-2025년 현재, CMT와 유사한 하이브리드 아키텍처가 주류로 자리잡음:[2][3]
- **CoAtNet**: 상대 주의 메커니즘으로 Conv와 Attention 통합
- **ConvNeXt V2**: CNN 아키텍처에 계층 정규화, 상대 위치 편향 추가
- **Swin Transformer**: 윈도우 기반 Attention으로 계층적 특징 학습

#### (2) **일반화 성능 강화 기법**

최근 연구 트렌드:
- **도메인 일반화**: 여러 소스 도메인에서 학습하여 보지 못한 도메인에 대한 강건성 개선
- **자기 지도 학습**: DINO, SimSiam 등으로 표현력 향상 및 과적합 감소
- **적응 메커니즘**: AdaptFormer 등으로 다양한 다운스트림 작업에 대한 효율적 적응

#### (3) **메모리 및 계산 효율성 개선**

- **동적 토큰 정규화 (DTN)**: Transformer의 학습 안정성 및 성능 향상
- **공간 중복 제거 (LF-ViT)**: 고해상도 이미지 처리 시 계산 비용 60% 감소
- **양자화 및 압축**: Trio-ViT 등으로 모바일/엣지 디바이스 배포 실현

#### (4) **멀티모달 및 통합 모델**

2025년 주요 트렌드:[4]
- **비전 파운데이션 모델 (VFM)**: Florence-2, SAM 2 등으로 다양한 작업을 통합
- **동적 아키텍처**: 입력 특성에 따라 계산량 동적 조절
- **3D 공간 이해**: 2D 이미지에서 3D 구조 추론 (NeRF, Point Cloud 기반 모델)

***

### 7. 앞으로의 연구 방향 및 고려사항

**향후 연구 시 고려할 점:**

#### (1) **메모리 효율성 극대화**

- Long-range attention을 유지하면서 Sparse Attention 패턴 개발
- 적응형 토큰 병합으로 불필요한 계산 제거

#### (2) **도메인 특화 설계**

- 의료 영상, 위성 영상 등 특정 도메인의 특성 반영
- 회전/스케일 불변성을 위한 구조적 개선

#### (3) **강건성 및 설명 가능성**

- 적대적 공격에 강건한 모델 설계
- Attention 맵의 시각화를 통한 해석 가능성 개선

#### (4) **온디바이스 배포**

- 경량 버전 개발 (CMT-Ti 수준)
- 양자화, 지식 증류 등 모델 압축 기법 강화

#### (5) **스케일링 법칙 규명**

- 비전 모델의 명확한 스케일링 법칙 정립 필요
- 파라미터 수, 데이터셋 크기, 계산량 간의 관계 규명

#### (6) **통합 아키텍처로의 전환**

2025년 트렌드인 작업별 통합 모델 개발:
- 분류, 탐지, 세그멘테이션을 하나의 모델로 처리
- I/O 표준화를 통한 일관된 인터페이스 제공

***

### 결론

**CMT 논문의 의의:**

CMT는 단순히 CNN과 Transformer를 결합한 것이 아니라, **각 아키텍처의 고유한 강점을 최대한 활용하고 약점을 보완하는 설계 원칙**을 제시했습니다. 특히 Local Perception Unit과 경량 Multi-head Self-Attention 등의 혁신은 이후 수많은 하이브리드 모델의 설계 철학이 되었습니다.

**2025년 현재의 시사점:**

- **하이브리드 아키텍처의 우월성 확립**: 순수 CNN도, 순수 Transformer도 아닌 하이브리드 모델이 성능과 효율성의 최적점
- **일반화 성능의 중요성**: 단순 정확도뿐 아니라 다양한 도메인과 작업에 대한 전이 가능성이 핵심
- **효율성과 성능의 균형**: 대규모 모델도 좋지만, 제한된 리소스에서 최적의 성능을 내는 모델 설계 기법의 가치 증대

CMT의 설계 철학은 현재의 VFM(비전 파운데이션 모델) 연구와 엣지 디바이스 최적화 연구의 토대가 되고 있으며, 앞으로도 컴퓨터 비전 분야의 핵심 기술로 자리잡을 것으로 예상됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/120e4340-91fe-435f-b19e-c767ff7e86e5/2107.06263v3.pdf)
[2](https://arxiv.org/ftp/arxiv/papers/2402/2402.02941.pdf)
[3](https://calib.tistory.com/entry/2025%EB%85%84-%EC%BB%B4%ED%93%A8%ED%84%B0-%EB%B9%84%EC%A0%84-%EA%B8%B0%EC%88%A0-%EB%8F%99%ED%96%A5-YOLO%EB%B6%80%ED%84%B0-Transformer%EA%B9%8C%EC%A7%80)
[4](https://blog-ko.superb-ai.com/vision-foundation-model-technical-challenges-future-trends/)
[5](https://arxiv.org/pdf/2107.06263.pdf)
[6](http://arxiv.org/pdf/2207.05501.pdf)
[7](https://pmc.ncbi.nlm.nih.gov/articles/PMC11243949/)
[8](https://arxiv.org/pdf/2302.09462.pdf)
[9](https://linkinghub.elsevier.com/retrieve/pii/S2215016124000098)
[10](https://pmc.ncbi.nlm.nih.gov/articles/PMC10381782/)
[11](http://arxiv.org/pdf/2410.11428.pdf)
[12](https://rahites.tistory.com/373)
[13](https://deep-learning-study.tistory.com/829)
[14](https://www.etnews.com/20250915000019)
[15](https://blog.naver.com/ziippy/222783603848)
[16](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12041791)
[17](https://deep-learning-study.tistory.com/?page=15)
[18](https://www.hellot.net/news/article.html?no=96069)
[19](https://www.jaenung.net/tree/18915)
[20](http://arxiv.org/pdf/2205.13535.pdf)
[21](https://arxiv.org/html/2403.09394v1)
[22](https://arxiv.org/pdf/2112.02624.pdf)
[23](https://arxiv.org/pdf/2207.05557.pdf)
[24](https://arxiv.org/pdf/2112.09747.pdf)
[25](http://arxiv.org/pdf/2405.03882.pdf)
[26](https://arxiv.org/pdf/2402.00033.pdf)
[27](https://wikidocs.net/255172)
[28](https://cdn.hanbit.co.kr/examples/2068/answer.pdf)
[29](https://velog.io/@heomollang/DeiT-%EA%B4%80%EB%A0%A8-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-04-Training-data-efficient-image-transformers-distillation-through-attentionDeiT)
[30](https://www.youtube.com/watch?v=CJoWBr8jjGQ)
[31](https://faculty.unist.ac.kr/sunghoonlim/wp-content/uploads/sites/393/2020/10/Guidebook_%EA%B3%B5%EA%B8%89%EB%A7%9D-%EC%B5%9C%EC%A0%81%ED%99%94-AI-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%85%8B.pdf)
[32](https://hyunseo-fullstackdiary.tistory.com/422)
[33](https://ki-it.com/xml/41150/41150.pdf)
[34](https://translate.google.com/translate?u=https%3A%2F%2Fwww.quora.com%2FIf-AI-has-no-creativity-how-did-it-solve-protein-folding&hl=ko&sl=en&tl=ko&client=srp)
[35](https://blog.naver.com/yongsulkwan_seoul/223818673181)
[36](https://www.themoonlight.io/ko/review/vision-transformers-in-domain-adaptation-and-domain-generalization-a-study-of-robustness)
[37](https://arxiv.org/pdf/2106.04803.pdf)
[38](https://arxiv.org/pdf/2103.14030.pdf)
[39](https://www.mdpi.com/2073-4409/11/15/2394/pdf?version=1659531956)
[40](https://pubs.aip.org/aip/pof/article-pdf/doi/10.1063/5.0160755/18071374/085108_1_5.0160755.pdf)
[41](http://arxiv.org/pdf/2412.01944.pdf)
[42](https://arxiv.org/pdf/2105.04553.pdf)
[43](https://arxiv.org/pdf/2208.11247.pdf)
[44](http://arxiv.org/html/2409.04734)
[45](https://blog.outta.ai/149)
[46](https://www.youtube.com/watch?v=V0VYsplgO5Q)
[47](https://swb.skku.edu/appliedailab/domestic_pub.do?mode=download&articleNo=49429&attachNo=45163)
[48](https://www.reddit.com/r/MachineLearning/comments/1iocgvg/d_need_suggestions_for_image_classification/)
[49](https://www.youtube.com/watch?v=hU7gP3u-tLQ)
[50](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE10546319)
[51](https://kr.linkedin.com/pulse/cnns-vs-vision-transformers-modern-comparison-cost-amit-kharche-8wnxf?tl=ko)
[52](https://hoya012.github.io/blog/Vision-Transformer-1/)
[53](https://www.youtube.com/watch?v=kocvskz2tJU)
[54](https://translate.google.com/translate?u=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FVision_transformer&hl=ko&sl=en&tl=ko&client=srp)
