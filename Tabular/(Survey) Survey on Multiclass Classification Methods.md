# Survey on Multiclass Classification Methods

### 1. 논문의 핵심 주장과 주요 기여

이 조사 논문(Mohamed Aly, 2005)의 핵심 주장은 **멀티클래스 분류 문제를 해결하기 위한 다양한 기법들이 존재하며, 이들은 크게 세 가지 패러다임으로 구분된다**는 것입니다.[1]

주요 기여는 다음과 같습니다:

1. **체계적인 분류 체계 제시**: 기존의 이진 분류 알고리즘을 멀티클래스로 확장하는 방식과 이진 분류 문제로 분해하는 방식, 그리고 계층적 접근법을 명확히 구분

2. **포괄적인 방법론 비교**: 신경망, 결정 트리, k-NN, Naive Bayes, SVM 등 자연 확장이 가능한 알고리즘부터 One-versus-All(OVA), All-versus-All(AVA), Error-Correcting Output Coding(ECOC) 등 분해 기법, 그리고 계층적 분류까지 일목요연하게 정리

3. **실무적 통찰**: "성능이 최상인 코딩 전략은 문제에 따라 다르다"는 결론으로, 특정 상황에서는 더 단순한 기법(예: OVA)도 적절히 튜닝되면 복잡한 기법만큼 효과적일 수 있음을 제시

***

### 2. 문제 정의, 제안 방법, 모델 구조 및 성능

#### 2.1 해결하고자 하는 문제

논문의 기본 문제 설정은 다음과 같습니다. 훈련 데이터셋 $$\{(x_i, y_i) : x_i \in \mathbb{R}^n, y_i \in \{1, \ldots, K\}\}$$이 주어질 때, 새로운 예제에 대해 올바른 클래스 레이블을 예측하는 학습 모델 $$H$$를 찾는 것입니다. 여기서 $$K$$는 클래스 개수입니다.[1]

**핵심 도전**: 대부분의 성공적인 이진 분류 알고리즘이 멀티클래스 문제에 직접 적용되지 않거나, 적용 시 비효율적이 되는 문제를 해결하는 것입니다.

#### 2.2 제안하는 방법들

논문은 세 가지 주요 접근 방식을 제시합니다:

**A. 자연 확장(Extensible Algorithms)**

신경망의 경우, 출력층의 뉴런 구조를 개선하여 멀티클래스 문제를 해결합니다:

- **일대일 코딩(One-per-class Coding)**: 클래스 $$k$$에 대해 $$K$$개의 뉴런이 필요하며, $$k$$번째 뉴런만 1, 나머지는 0의 출력을 가집니다. 테스트 시 최대 출력값을 제공하는 뉴런의 클래스가 선택됩니다.[1]

- **분산 코딩(Distributed Coding)**: 각 클래스에 0에서 $$2^N - 1$$ 범위의 고유 이진 코드워드를 할당합니다. 테스트 시 출력 코드워드와 각 클래스 코드워드 사이의 **Hamming 거리**를 계산하여 최소 거리를 가진 클래스를 선택합니다.[1]

결정 트리, k-NN, Naive Bayes, SVM도 자연스럽게 멀티클래스로 확장 가능합니다.

Naive Bayes의 경우, 최대 사후 확률(MAP) 원리를 적용하여:

$$c^* = \arg\max_c P(C=c|x_1, \ldots, x_N) = \arg\max_c P(C=c) \prod_{i=1}^{N} P(x_i|C=c)$$

조건부 독립 가정 하에서 $$K$$개 클래스에 대해 직접 최적 클래스를 선택합니다.[1]

**B. 이진 분해 방법(Decomposition into Binary Classification)**

$$K$$개의 클래스를 여러 이진 분류 문제로 변환합니다:

1. **일대다(OVA)**: $$K$$개의 이진 분류기를 학습합니다. $$k$$번째 분류기는 클래스 $$k$$를 양성, 나머지 $$K-1$$개를 음성으로 취급합니다. 테스트 시 최대 출력값을 제공하는 분류기를 선택합니다.[1]

   수식: 분류기 $$f_k(x)$$ (k=1,...,K)에 대해 $$\hat{y} = \arg\max_k f_k(x)$$

2. **일대일(AVA)**: $$\frac{K(K-1)}{2}$$개의 이진 분류기를 구성하여 모든 클래스 쌍을 구분합니다. 테스트 시 투표 방식으로 가장 많은 표를 얻은 클래스를 선택합니다.[1]

3. **오류 정정 출력 코딩(ECOC)**: $$N$$개의 이진 분류기를 학습하며, 각 클래스 $$k$$에 길이 $$N$$의 코드워드를 할당합니다. 이진 행렬 $$M$$의 각 열이 하나의 분류기에 대응됩니다. 테스트 시 출력 코드워드와 $$K$$개 클래스 코드워드 간 Hamming 거리를 계산하여 최소값을 선택합니다.[1]

4. **일반화 코딩(Generalized Coding)**: Allwein et al.이 제안한 확장으로, 코딩 행렬 $$M$$의 원소를 $$\{-1, 0, +1\}$$으로 허용합니다.[1]

   - $$M(k,n) = +1$$: 클래스 $$k$$의 예제를 분류기 $$n$$에 양성
   - $$M(k,n) = -1$$: 음성
   - $$M(k,n) = 0$$: 무시

   마진 기반 분류기의 경우 거리 함수:
   $$d(M(k), f(x)) = \sum_{i=1}^{N} L(M(k,i)f_i(x))$$
   
   여기서 $$L(\cdot)$$는 손실 함수입니다.[1]

   코드워드 길이는 **밀집 방법**으로 $$\lceil 10 \log_2 K \rceil$$, **희소 방법**으로 $$\lceil 15 \log_2 K \rceil$$입니다.[1]

**C. 계층적 분류(Hierarchical Classification)**

클래스를 이진 트리 구조로 배열하여 단계적으로 분류합니다:

- **Binary Hierarchical Classifier(BHS)**: $$K-1$$개의 이진 분류기를 이진 트리로 배열하며, 각 노드에서 Fisher Discriminant를 이용한 최적 이진 분할을 수행합니다.[1]

- **Hierarchical SVM(HSVM)**: Kullback-Leibler 거리 기반 그래프와 최대 절단(max-cut) 알고리즘으로 클래스를 분할합니다.[1]

- **Divide-By-2(DB2)**: k-means 또는 클래스 평균 비교를 통해 이진 분할하며, 각 노드에서 이진 SVM을 사용합니다.[1]

#### 2.3 성능 분석

논문의 주요 성능 발견:

| 방법 | 장점 | 한계 | 
|------|------|------|
| **OVA** | 간단함, 적절히 튜닝되면 우수 성능 | 대체로 다른 방법보다 성능 낮음 |
| **AVA** | OVA보다 일반적으로 우수 | 분류기 개수 증가로 계산 복잡도 증가 |
| **ECOC** | ECOC \& 일반화 코딩보다 우수 | 오류 정정 코드 설계 필요 |
| **일반화 코딩** | 위 방법들의 일반화, 유연함 | 문제 의존적 성능 |
| **계층적 분류** | 학습/테스트 시간 효율적 | 계층 구조 결정이 중요 |

***

### 3. 일반화 성능 향상 및 이론적 기반

#### 3.1 논문의 일반화 성능 인사이트

이 논문은 명시적인 일반화 이론을 제시하지는 않지만, 다음의 중요한 관찰을 제공합니다:

1. **코드워드 설계의 중요성**: ECOC의 성공은 클래스 간 충분한 Hamming 거리를 유지하는 코드워드 설계에 달려있습니다. 이는 분류 오류의 오류 정정 능력을 향상시킵니다.[1]

2. **분해 방식의 선택**: 문제마다 최적의 분해 전략이 다르다는 관찰은, 일반화 성능이 데이터의 클래스 구조(manifold structure)에 의존함을 시사합니다.[1]

3. **계층적 구조**: 클래스 간 유사성을 고려한 계층적 배열이 일반화를 개선할 수 있음을 시사합니다.[1]

#### 3.2 최신 연구 기반 일반화 성능 분석

최근 멀티클래스 분류의 일반화 성능 관련 연구들이 더욱 심화되었습니다:

**일반화 이론의 최신 발전**:

1. **깊은 신경망의 일반화**: Ledent et al.(2021)의 연구에서 보면, 깊은 멀티클래스 신경망의 일반화 오차 경계는 **클래스 개수에 대한 명시적 의존성을 제거**할 수 있습니다. 이는 Norm-based bounds를 통해 달성되며:[2]
   
   $$\text{Generalization Error} \propto \|W\|_{\text{spectral}} \cdot \sqrt{\frac{\log K}{n}}$$
   
   여기서 $$K$$는 클래스 수, $$n$$은 샘플 수입니다. 기존의 $$K$$에 선형 의존하는 경계보다 훨씬 나은 스케일링을 제공합니다.[2]

2. **과잉 모수화 아래의 멀티클래스 분류**: Subramanian et al.(2022)은 최소 노름 보간(Minimum-Norm Interpolator)을 사용한 멀티클래스 분류 시, 이진 분류와 달리 **클래스 수가 천천히 증가할 때에도 일반화가 가능**함을 증명했습니다. 이는 멀티클래스 문제가 이진 문제보다 "더 어렵지만" 여전히 관리 가능함을 시사합니다.[3]

**실용적 일반화 개선 기법들**:

1. **클래스 불균형 처리**: 최근 리뷰(Yang et al., 2024)에서 SMOTE, ADASYN, GMM-SMOTE 등의 오버샘플링 기법이 멀티클래스 불균형 데이터에서 일반화를 크게 개선함을 보여줍니다. 특히 GMM-SMOTE는 Mahalanobis 거리를 이용하여 소수 클래스의 공분산 구조를 보존함으로써 오버피팅을 감소시킵니다.[4][5]

2. **정규화 기법**: L2 정규화와 Dropout은 여전히 멀티클래스 신경망의 오버피팅 방지에 효과적입니다. 또한 조기 종료(early stopping)는 검증 데이터 성능을 기준으로 학습을 조정합니다.[6][7]

3. **앙상블 방법**: 최신 연구(Iwendi et al., 2020)에 따르면, 이기종(heterogeneous) 앙상블은 멀티클래스 불균형 문제에서 개별 분류기보다 **최소 10% 이상 정확도를 개선**할 수 있습니다.[8]

4. **전이 학습(Transfer Learning)**: 사전학습된 ResNet/EfficientNet을 특징 추출기로 사용하고, 데이터 증강을 통해 클래스 균형을 맞추는 방식이 5-클래스 문제에서 97% 이상의 정확도를 달성하고 있습니다.[9]

5. **Transformer 기반 방법**: Vision Transformer와 Sequential Transformer의 결합(ViT-TST)이 멀티클래스 MRI 분류에서 98.77% 정확도를 달성하여, CNN 기반 방법(97.64%)을 능가하고 있습니다. 이는 self-attention 메커니즘이 클래스 간 섬세한 구분 경계를 학습하는 데 우수함을 시사합니다.[10]

---

### 4. 한계와 개선 방향

#### 4.1 원 논문의 한계

1. **이론적 보장 부재**: 논문은 경험적 비교에 중점을 두며, 각 방법의 일반화 오차에 대한 이론적 경계를 제시하지 않습니다.

2. **계산 복잡도 분석 부족**: AVA와 ECOC의 분류기 개수 증가에 따른 실행 시간 영향을 충분히 분석하지 않습니다. DB2는 학습 시간에서 AVA와 유사하지만 테스트 시간이 우수하다는 언급에 그칩니다.[1]

3. **불균형 데이터 처리 미흡**: 논문이 작성된 2005년 당시 클래스 불균형 문제에 대한 논의가 제한적입니다.

4. **마진 개념의 제한**: SVM의 마진(margin) 개념이 멀티클래스 문제에 직접 확장되지 않음을 충분히 다루지 않습니다.

#### 4.2 최신 연구 기반 개선 사항 및 고려 사항

1. **클래스 구조 활용**: 최근 연구는 클래스 간의 의미론적 유사성을 활용하여 계층 구조를 자동으로 학습하는 방법을 제시합니다. 이는 Kullback-Leibler 거리나 학습된 임베딩 공간에서의 거리를 기반으로 합니다.[11]

2. **약한 감독(Weak Supervision) 활용**: Multi-class Classification without Multi-class Labels 연구(2019)에서는 쌍별 유사성 정보만으로 멀티클래스 분류를 학습할 수 있음을 보여줍니다. 이는 레이블링 비용이 높은 실제 응용에서 일반화 성능을 향상시킵니다.[12]

3. **동적 네트워크 구조**: QADM-Net(2025) 같은 최신 방법은 샘플 품질에 따라 네트워크 깊이와 매개변수를 동적으로 조정하여 멀티모달 멀티클래스 분류에서 신뢰성을 개선합니다.[13]

4. **정규화 강화**:
   - **그래프 기반 정규화**: 클래스 관계를 그래프로 표현하고 GCN(Graph Convolutional Network)을 통해 정규화
   - **메타 학습(Meta-Learning)**: 분류기 학습 과정 자체를 학습하여 새로운 클래스 적응에 최적화
   - **확률적 정규화**: Bayesian Neural Networks를 통한 불확실성 정량화

5. **다중 관점 학습**: 멀티모달 데이터에서 서로 다른 모드에서 나온 분류기들을 앙상블하면, 단일 모드 분류기보다 **강건한 일반화**를 달성합니다.[14]

6. **스케일링 문제 해결**: 클래스 개수 $$K$$가 매우 클 때(예: 수천 개 클래스), 계층적 분류(Hierarchical Classification)와 ECOC의 조합이 필수적입니다. 최근 연구에서는 학습 가능한 트리 구조(Learnable Hierarchies)를 제안하고 있습니다.[15]

***

### 5. 시사점 및 향후 연구 고려 사항

#### 5.1 이론과 실제의 간극

원 논문의 경험적 결론—"최적 전략은 문제 의존적"—은 여전히 유효하지만, 최근 일반화 이론의 발전으로 **어떤 조건에서 어느 방법이 우수한지**를 더욱 정확히 예측할 수 있게 되었습니다. 예를 들어:

- **클래스 개수가 적고 데이터 충분**: 모든 방법이 유사하게 동작하며, 계산 효율을 고려해 OVA 추천
- **클래스 개수 많고 데이터 제한적**: ECOC 또는 계층적 분류로 표현 차원 감소
- **심한 클래스 불균형**: GMM-SMOTE + 앙상블 + 가중 손실 함수 조합

#### 5.2 향후 연구 중점 분야

1. **적응형 멀티클래스 전략**: 데이터의 특성(클래스 수, 불균형도, 차원 등)을 자동으로 감지하고 최적 분류 전략을 선택하는 메타학습

2. **온라인 멀티클래스 학습**: 클래스가 동적으로 추가되거나 개념이 드리프트되는 상황에서의 일반화 유지

3. **설명 가능성**: 블랙박스 모델이 $$K$$개 클래스 중 특정 클래스를 선택한 근거를 설명하는 방법(특히 Transformer와 같은 대규모 모델)

4. **분산 학습의 멀티클래스 최적화**: 대규모 클래스를 분산 환경에서 효율적으로 처리하기 위한 통신 효율적 알고리즘

5. **불완전 정보 환경**: 일부 클래스가 충분히 표현되지 않거나 오류로 레이블된 상황에서의 강건한 멀티클래스 분류

***

## 결론

Mohamed Aly의 2005년 Survey는 멀티클래스 분류의 기초적 틀을 제공했으며, 20년이 지난 현재에도 **OVA, AVA, ECOC와 계층적 분류의 근본적 아이디어는 여전히 유효**합니다. 그러나 최신 연구의 진전은:

1. **이론적 뒷받침**: 일반화 오차 경계와 class number scalability 연구로 각 방법의 장점을 정량화
2. **실무적 고도화**: 불균형 데이터, 약한 감독, 동적 적응을 포함한 현실적 상황 대응
3. **아키텍처 혁신**: Transformer와 메타학습으로 기존 방법론의 한계 극복

이러한 진보들이 결합되면, 단순히 "문제에 따라 다르다"에서 "이러한 조건에서는 이 방법을 추천"으로의 전환이 가능하게 되었습니다. 특히 **깊은 신경망 기반의 일반화 이론**과 **실시간 데이터 처리 환경**에서의 응용이 향후 핵심 연구 주제가 될 것으로 예상됩니다.[3][14][2][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/149cf2ce-5488-4c3b-a602-6cc42ff3c9ec/document.pdf)
[2](https://ml.cs.rptu.de/publications/2021/AAAI_norm.pdf)
[3](https://openreview.net/pdf?id=ikWvMRVQBWW)
[4](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2024.1430245/full)
[5](https://internationalpubls.com/index.php/cana/article/view/4260)
[6](https://towardsdatascience.com/regularization-avoiding-overfitting-in-machine-learning-bb65d993e9cc/)
[7](https://sol.sbc.org.br/index.php/eniac/article/download/9335/9237/)
[8](https://pmc.ncbi.nlm.nih.gov/articles/PMC7249012/)
[9](https://www.frontiersin.org/journals/bioinformatics/articles/10.3389/fbinf.2025.1567219/full)
[10](https://www.nature.com/articles/s41598-024-59578-3)
[11](https://www.nature.com/articles/s41598-025-13929-w)
[12](https://arxiv.org/pdf/1901.00544.pdf)
[13](http://arxiv.org/pdf/2412.14489.pdf)
[14](https://www.sciencedirect.com/science/article/abs/pii/S0957417424017007)
[15](https://arxiv.org/abs/2308.03005)
[16](https://arxiv.org/pdf/1609.00085.pdf)
[17](https://arxiv.org/pdf/2205.15860.pdf)
[18](https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-18/issue-1/Multiclass-classification-for-multidimensional-functional-data-through-deep-neural-networks/10.1214/24-EJS2229.pdf)
[19](http://arxiv.org/pdf/1310.1949.pdf)
[20](https://arxiv.org/pdf/1601.01121.pdf)
[21](http://arxiv.org/pdf/2306.06517.pdf)
[22](https://arxiv.org/html/2507.17121v2)
[23](https://ieeexplore.ieee.org/document/10578143/)
[24](https://arxiv.org/pdf/2501.12554.pdf)
[25](https://www.nature.com/articles/s41598-025-05585-x)
[26](http://arxiv.org/pdf/2502.03417.pdf)
[27](https://pmc.ncbi.nlm.nih.gov/articles/PMC11487558/)
[28](https://www.frontiersin.org/articles/10.3389/fncom.2024.1404623/full)
[29](https://arxiv.org/pdf/2210.02476.pdf)
[30](https://www.mdpi.com/1424-8220/25/5/1487)
[31](http://arxiv.org/pdf/2311.04157.pdf)
[32](https://arxiv.org/pdf/2111.12993.pdf)
[33](https://norma.ncirl.ie/7921/1/sharanyaneelakanti.pdf)
[34](https://ruslanmv.com/blog/Multiclass-Classification-with-Ensemble-Models)
[35](https://developers.google.com/machine-learning/crash-course/overfitting/regularization)
[36](https://kr.mathworks.com/help/stats/ensemble-algorithms.html)
[37](https://arxiv.org/html/2509.16542v1)
[38](https://www.nature.com/articles/s41598-023-49080-7)
