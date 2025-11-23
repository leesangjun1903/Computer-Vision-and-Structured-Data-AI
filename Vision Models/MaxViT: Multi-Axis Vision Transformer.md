# MaxViT: Multi-Axis Vision Transformer

### 1. 핵심 주장 및 주요 기여

MaxViT(Multi-Axis Vision Transformer)의 **핵심 주장**은 비전 트랜스포머의 확장성 문제를 해결하기 위해 다축(multi-axis) 주의 메커니즘을 제안하는 것입니다. 기존 Vision Transformer(ViT)는 이미지 크기에 대한 자기 주의(self-attention)의 이차 복잡도로 인해 실제 응용이 제한되었으며, 창 기반 주의(window-based attention)를 사용하는 Swin Transformer는 지역성을 과도하게 강조하여 전역 상호작용을 놓치고 있습니다.[1]

MaxViT의 **주요 기여**는 다음과 같습니다:[1]

- **다축 주의 모듈(Max-SA)**: 차단 국소 주의와 확장 전역 주의로 구성되어 선형 복잡도로 전역-국소 공간 상호작용을 가능하게 함
- **범용 강력한 백본**: 네트워크의 모든 단계(심지어 초기 고해상도 단계에서도)에서 전역 지각이 가능한 하이브리드 비전 트랜스포머 설계
- **광범위한 성능 검증**: 이미지 분류, 객체 검출, 이미지 미학 평가, 이미지 생성 등 다양한 시각 작업에서 최첨단 성능 달성

ImageNet-1K에서 미세 조정 시 **86.5%의 상위 1 정확도**를 달성했으며, ImageNet-21K 사전학습 시 **88.7%**의 정확도를 기록했습니다.[1]

***

### 2. 해결하는 문제, 제안 방법 및 모델 구조

#### 2.1 해결하는 주요 문제

기존 비전 트랜스포머들의 한계:[1]

1. **ViT의 문제**: 완전 자기 주의는 $$O(N^2)$$ 복잡도($$N$$은 패치 수)로 인해 고해상도 이미지에서 계산상 불가능
2. **Swin Transformer의 문제**: 창 기반 주의로 국소성을 강화했지만 비지역성 손실로 인해 큰 규모 데이터셋(ImageNet-21K, JFT)에서 성능 저하
3. **스케일 문제**: 계산 효율성과 모델 용량의 균형 달성 어려움

#### 2.2 제안하는 방법: 다축 주의(Multi-Axis Attention)

MaxViT는 공간축을 분해하여 두 가지 희소 주의 형태를 도입합니다:[1]

**차단 주의(Block Attention)**:
$$X \in \mathbb{R}^{H \times W \times C}$$에 대해, 고정 크기 $$P \times P$$의 겹치지 않는 창으로 분할합니다:

$$\text{Block}: (H, W, C) \rightarrow \left(\frac{H}{P} \times \frac{W}{P}, P \times P, C\right) \rightarrow \left(\frac{HW}{P^2}, P^2, C\right)$$

각 창 내에서 상대 주의를 적용하여 국소 상호작용을 수행합니다:

$$\text{RelAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + B\right)V$$

여기서 $$B \in \mathbb{R}^{(2H-1)(2W-1)}$$는 학습된 위치 편향 행렬입니다.[1]

**격자 주의(Grid Attention)**:

이미지를 균일한 $$G \times G$$ 격자로 분할하여 각 격자 셀이 적응형 크기 $$\frac{H}{G} \times \frac{W}{G}$$를 가집니다:

$$\text{Grid}: (H, W, C) \rightarrow \left(G \times \frac{H}{G}, G \times \frac{W}{G}, C\right) \rightarrow \left(G^2, \frac{HW}{G^2}, C\right)$$

격자 축($$G \times G$$)에 자기 주의를 적용하여 희소하지만 전역적인 공간 혼합을 수행합니다.[1]

**복잡도 분석**:
- 차단 주의: $$O(HW \cdot P^2)$$
- 격자 주의: $$O(HW \cdot (H/G)^2)$$
- 총합: $$O(HW)$$ (선형 복잡도)

#### 2.3 모델 구조

**MaxViT 블록 구성**:[1]

MaxViT 블록은 순차적으로 다음을 포함합니다:

$$\text{Block} = \text{MBConv} + \text{BlockAttention} + \text{GridAttention}$$

각 구성 요소:

1. **MBConv 층** (선택적 하향 샘플링):
$$x \leftarrow x + \text{Proj}(\text{SE}(\text{DWConv}(\text{Conv}(\text{Norm}(x)))))$$

2. **차단 주의**:
$$x \leftarrow x + \text{Unblock}(\text{RelAttention}(\text{Block}(\text{LN}(x))))$$
$$x \leftarrow x + \text{MLP}(\text{LN}(x))$$

3. **격자 주의**:
$$x \leftarrow x + \text{Ungrid}(\text{RelAttention}(\text{Grid}(\text{LN}(x))))$$
$$x \leftarrow x + \text{MLP}(\text{LN}(x))$$

**계층적 구조**:[1]

MaxViT는 ResNet 스타일의 4단계 계층 구조를 따릅니다:

| 단계 | 해상도 | 역할 | 블록 수 |
|------|---------|------|---------|
| S0 (Stem) | 1/2 | 특성 추출 | Conv 2×2 |
| S1 | 1/4 | 초기 특성 | L₁ (2~2) |
| S2 | 1/8 | 중간 특성 | L₂ (2~6) |
| S3 | 1/16 | 심화 특성 | L₃ (5~14) |
| S4 | 1/32 | 최종 특성 | L₄ (2~2) |

***

### 3. 성능 향상

#### 3.1 이미지 분류 성능[1]

**ImageNet-1K (224×224)**:
- MaxViT-T: 83.62% (31M 매개변수, 5.6G FLOPs)
- MaxViT-S: 84.45% (69M 매개변수, 11.7G FLOPs)
- MaxViT-B: 84.95% (120M 매개변수, 23.4G FLOPs)
- MaxViT-L: **85.17%** (212M 매개변수, 43.9G FLOPs) - 추가 학습 전략 없이 달성

**ImageNet-1K (384×384 미세 조정)**:
- MaxViT-L: **86.40%** 상위 1 정확도

**ImageNet-1K (512×512 미세 조정)**:
- MaxViT-L: **86.70%** 상위 1 정확도 (새로운 최첨단 성능)

#### 3.2 대규모 사전학습 성능[1]

**ImageNet-21K → ImageNet-1K**:
- MaxViT-B: 88.24% (119M 매개변수)
- MaxViT-L: 88.32% (212M 매개변수)
- MaxViT-XL: **88.70%** (475M 매개변수) - 512×512에서 미세 조정

**JFT-300M → ImageNet-1K**:
- MaxViT-XL: **89.53%** (475M 매개변수, 535.2G FLOPs)

#### 3.3 다운스트림 작업 성능

**COCO 객체 검출 (AP)**:[1]
- MaxViT-S: 53.1 AP (595G FLOPs)
- MaxViT-B: 53.4 AP (856G FLOPs)
- 비교: Swin-B는 51.9 AP (982G FLOPs)

**AVA 이미지 미학 평가**:[1]
- MaxViT-T (512×512): PLCC 0.745, SRCC 0.708

**ImageNet 이미지 생성 (128×128)**:[1]
- FID: 30.77 (18.6M 매개변수)
- IS: 22.58 (HiT의 21.64 대비)

***

### 4. 일반화 성능 향상 가능성

#### 4.1 일반화 메커니즘

MaxViT가 우수한 일반화 성능을 달성하는 핵심 요인:[1]

1. **전역-국소 상호작용**: 네트워크의 모든 단계에서 전역 지각으로 인해 고해상도 단계에서도 장거리 의존성 포착
2. **선형 복잡도**: 계산 효율로 인해 모든 해상도에서 주의 메커니즘 적용 가능
3. **하이브리드 설계**: MBConv의 귀납적 편향과 주의의 유연성 결합

#### 4.2 분포 외(OOD) 성능

비전 트랜스포머의 일반화 특성에 관한 최신 연구에 따르면:[2]

- **형태 편향**: 트랜스포머는 CNN과 달리 형태에 더 강하게 편향되어 텍스처 변환에 더 잘 대응
- **규모 상관관계**: 더 큰 ViT 모델이 분포 외 데이터에서 더 나은 성능 달성
- **매개변수 민감성**: ViT는 CNN보다 하이퍼파라미터에 더 민감

MaxViT의 경우, 다양한 데이터 영역에서의 강인성이 입증되었습니다:[1]
- ImageNet-1K 사전학습 모델이 COCO, AVA, ImageNet 생성 작업에서 우수한 전이 성능
- 서로 다른 입력 해상도(224×224, 384×384, 512×512)에 대한 강인한 성능 유지

#### 4.3 앙상블과 특성 다양성

최신 연구(2023-2025)에서 시사하는 개선 방향:[2]

1. **특성 다양성 강화**: 주의 헤드의 직교성을 격려하여 서로 다른 특성 학습 유도
2. **헤드 가지치기**: 테스트 시간에 허위 특성에 해당하는 헤드 제거로 OOD 성능 향상

***

### 5. 한계 및 제약

#### 5.1 현재 한계[1]

1. **메모리 효율성**: 격자 주의 시 고해상도에서 여전히 상당한 메모리 필요
2. **학습 복잡성**: 다양한 데이터 영역에서의 하이퍼파라미터 튜닝 필요
3. **순차적 설계의 최적성**: 블록-격자 순서가 분류에는 최적이지만, 생성 작업에서는 격자-블록 순서 필요
4. **규모 한계**: 10억 매개변수 규모의 매우 큰 모델과 행성 규모 데이터셋(JFT-3B)에서의 실험 미수행

#### 5.2 기술적 제약

1. **위치 편향 보간**: 높은 해상도로 미세 조정 시 상대 위치 편향 행렬의 쌍선형 보간이 필요[1]
2. **순서 의존성**: MBConv-BlockAttention-GridAttention 순서가 효과적이지만, 다른 작업에서는 다른 순서 필요[1]
3. **병렬 vs 순차 설계**: 순차 설계가 병렬 설계보다 우수하지만 매개변수/FLOPs 오버헤드 존재[1]

***

### 6. 최신 연구 동향 (2024-2025)

#### 6.1 MaxViT 기반 확장 연구

**의료 이미지 분할**:
MSA-MaxNet(2024)은 MaxViT 블록을 인코더-디코더 구조에 통합하여 Synapse 데이터셋에서 DSC 92.58% 달성. 다중 척도 주의 강화 및 MCBAM을 통해 스킵 연결 최적화.[3]

**초고해상도(Super-Resolution)**:
MaxSR(2023)는 MaxViT를 이미지 초고해상도에 적용하여 자기 유사성 선행 활용.[4]

**자궁경부암 진단**:
MaxCerVixT(2024)는 MaxViT를 개조하여 Pap smear 이미지 분류에 99.02% 정확도 달성.[5]

**교각 손상 분류**:
GCN+MaxViT(2024)는 다중 레이블 손상 분류에 mAP 99.37% 달성.[6]

#### 6.2 일반화 성능 개선 추세

최신 연구(2024-2025)의 일반화 성능 강화 방향:[7][2]

1. **고주파 성분 강화(HAT)**: 이미지의 고주파 성분에 대한 대적 학습으로 ViT 성능 1-1.2% 향상
2. **다중 척도 특성**: 비전-언어 작업에서 MaxViT는 다중 척도 특성 추출로 CoAtNet보다 추론 중심 작업에서 우수
3. **가벼운 네트워크 개발**: 모바일 배포를 위한 경량 MaxViT 변형 개발 (예: MaxCerVixT)

***

### 7. 향후 연구 시 고려할 점

#### 7.1 기술적 개선 방향

1. **효율성 최적화**:
   - 더 효율적인 격자 분할 알고리즘 개발
   - 메모리 효율적인 주의 계산 기법
   - 양자화 및 프루닝 기법 적용

2. **구조적 혁신**:
   - 적응형 윈도우/격자 크기 동적 조정
   - 작업별 최적 순서 자동 결정
   - 다양한 해상도에서의 위치 인코딩 개선

3. **전이 학습 강화**:
   - 다중 작업 학습 프레임워크 개발
   - 도메인 특화 사전학습 전략
   - 몇 샷 학습(few-shot learning) 성능 개선

#### 7.2 응용 분야 확장

1. **비디오 이해**: 시간-공간 다축 주의로 확장[1]
2. **멀티모달 학습**: 비전-언어 작업에서 MaxViT의 계층적 특성 활용[7]
3. **의료 이미징**: 3D 의료 이미지 분할에 MaxViT 적용[3]
4. **고해상도 이미지 처리**: 4K, 8K 이미지 처리 효율성 개선

#### 7.3 이론적 분석

1. **일반화 한계 증명**: 다축 주의의 일반화 오류에 대한 이론적 분석
2. **귀납적 편향 연구**: 하이브리드 설계의 귀납적 편향 체계적 분석
3. **최적화 동역학**: 다축 주의 학습의 수렴 특성 연구

#### 7.4 실무적 고려사항

1. **배포 효율성**:
   - 엣지 디바이스 지원 경량 버전 개발
   - 추론 시간 최적화 및 배치 처리 효율성
   - 온디바이스 학습 가능성 탐색

2. **공정성과 해석성**:
   - 편향 완화 메커니즘 개발
   - 주의 맵 해석성 연구
   - 설명 가능한 AI(XAI) 기법 적용

3. **대규모 모델 확장**:
   - 10억+ 매개변수 모델의 학습 및 배포 전략
   - 분산 학습 최적화
   - 메모리 및 계산 효율성 균형

***

### 결론

MaxViT는 **다축 주의** 메커니즘을 통해 비전 트랜스포머의 확장성 문제를 우아하게 해결하며, 이미지 분류, 객체 검출, 생성 작업 등 광범위한 시각 작업에서 최첨단 성능을 달성합니다. 특히 **선형 복잡도**와 **전역 지각 능력**의 결합으로 인해 우수한 일반화 성능을 보여줍니다.

향후 연구는 **의료 이미징, 멀티모달 학습, 비디오 이해** 등 다양한 분야로의 확장, **효율성 개선**, **이론적 기초 강화**에 중점을 두어야 할 것으로 예상됩니다. 2024-2025년의 최신 연구들은 MaxViT 기반의 특화된 구조들(MSA-MaxNet, MaxCerVixT 등)이 실제 응용에서 뛰어난 성능을 입증하고 있으며, 이는 MaxViT의 범용성과 적응성을 강력하게 뒷받침합니다.

***

### 참고문헌

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/022d705c-35af-4eed-9ca8-e6b0795cc618/2204.01697v4.pdf)
[2](https://ieeexplore.ieee.org/document/10914740/)
[3](http://medrxiv.org/lookup/doi/10.1101/2024.11.02.24316635)
[4](https://www.mdpi.com/2076-3417/15/22/11882)
[5](https://www.mdpi.com/2306-5354/12/7/693)
[6](https://isjem.com/download/colorectal-cancer-detection-using-deep-and-transfer-learning/)
[7](https://www.ijraset.com/best-journal/aidriven-selfhealing-automated-ui-testing-framework-with-visual-proof)
[8](https://ieeexplore.ieee.org/document/11203536/)
[9](https://ieeexplore.ieee.org/document/10557525/)
[10](http://pubs.rsna.org/doi/10.1148/radiol.240775)
[11](https://www.ewadirect.com/proceedings/ace/article/view/25643)
[12](https://arxiv.org/pdf/2103.15358.pdf)
[13](https://arxiv.org/pdf/2307.07240.pdf)
[14](http://arxiv.org/pdf/2204.01697.pdf)
[15](http://arxiv.org/pdf/2205.13535.pdf)
[16](https://www.mdpi.com/1424-8220/23/7/3447/pdf?version=1680001445)
[17](https://arxiv.org/pdf/2206.09959.pdf)
[18](https://arxiv.org/pdf/2207.07268.pdf)
[19](https://arxiv.org/pdf/2206.01191.pdf)
[20](http://arxiv.org/abs/2204.01697)
[21](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840001.pdf)
[22](https://pmc.ncbi.nlm.nih.gov/articles/PMC11661918/)
[23](https://www.sciencedirect.com/science/article/abs/pii/S0950705124001175)
[24](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Delving_Deep_Into_the_Generalization_of_Vision_Transformers_Under_Distribution_CVPR_2022_paper.pdf)
[25](https://www.sciencedirect.com/science/article/pii/S1574013724001047)
[26](https://www.iaeng.org/IJCS/issues_v52/issue_8/IJCS_52_8_36.pdf)
[27](https://arxiv.org/abs/2308.16274)
[28](https://arxiv.org/html/2305.08396v5)
[29](https://www.semanticscholar.org/paper/MaxViT-UNet:-Multi-Axis-Attention-for-Medical-Image-Khan-Khan/70d59e24719a3e29f94916476d005dc59167a1a4)
