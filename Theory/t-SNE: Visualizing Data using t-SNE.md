# t-SNE: Visualizing Data using t-SNE

## 핵심 주장과 주요 기여

t-SNE(t-distributed Stochastic Neighbor Embedding)는 Laurens van der Maaten과 Geoffrey Hinton이 2008년 발표한 혁신적인 고차원 데이터 시각화 기법입니다. 이 논문의 핵심 주장과 주요 기여는 다음과 같습니다:[1]

**핵심 주장:**
- 기존의 SNE(Stochastic Neighbor Embedding)의 두 가지 주요 문제인 "crowding problem"과 최적화의 어려움을 해결[1]
- 고차원 데이터의 local structure와 global structure를 동시에 보존하는 2차원 또는 3차원 지도 생성[1]
- Student t-distribution을 사용하여 저차원 공간에서의 heavy-tailed distribution 특성 활용[1]

**주요 기여:**
1. **Symmetric SNE**: 기존 SNE의 비대칭 조건부 확률을 대칭적 결합 확률로 변환하여 gradient 계산 단순화[1]
2. **t-distribution 사용**: 저차원 공간에서 Student t-distribution(자유도 1)을 사용하여 crowding problem 해결[1]
3. **Random walk 기반 확장**: 대용량 데이터셋 처리를 위한 landmark 기반 접근법 개발[1]
4. **우수한 성능**: MNIST, Olivetti faces, COIL-20 등 다양한 데이터셋에서 기존 기법들(Sammon mapping, Isomap, LLE 등) 대비 뛰어난 시각화 성능 입증[1]

## 해결하고자 하는 문제

### 1. Crowding Problem
기존 차원축소 기법들의 핵심 문제는 **crowding problem**입니다. 고차원 공간에서는 구의 부피가 반지름의 m제곱에 비례하여 증가하는데(m은 차원수), 이를 2차원으로 축소할 때 중간 거리의 데이터 포인트들을 배치할 충분한 공간이 없어 중앙에 밀집되는 현상이 발생합니다.[1]

### 2. SNE의 최적화 어려움
원래 SNE는 simulated annealing이 필요하고 매개변수 선택이 까다로워 여러 번 실행해야 하는 문제가 있었습니다.[1]

### 3. Local-Global Structure 보존
기존 기법들은 local structure 또는 global structure 중 하나만 잘 보존하는 한계가 있었습니다.[1]

## 제안하는 방법

### 1. High-dimensional Space에서의 확률 정의
고차원 공간에서 데이터 포인트 $$x_i$$를 중심으로 한 가우시안 분포를 사용하여 조건부 확률을 정의합니다:

$$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$$

대칭적 결합 확률은 다음과 같이 정의됩니다:

$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$$

### 2. Low-dimensional Space에서의 t-distribution 사용
저차원 공간에서는 Student t-distribution(자유도 1)을 사용하여 결합 확률을 정의합니다:

$$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}$$

### 3. Cost Function과 Gradient
Kullback-Leibler divergence를 최소화합니다:

$$C = KL(P||Q) = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

Gradient는 다음과 같습니다:

$$\frac{\partial C}{\partial y_i} = 4\sum_j (p_{ij} - q_{ij})(y_i - y_j)(1 + \|y_i - y_j\|^2)^{-1}$$

### 4. 최적화 기법
- **Early exaggeration**: 초기 단계에서 $$p_{ij}$$를 4배로 확대하여 클러스터 분리 촉진[1]
- **Adaptive learning rate**: Jacobs(1988)의 적응적 학습률 사용[1]
- **Momentum**: 초기 250회는 0.5, 이후 0.8로 설정[1]

## 모델 구조

t-SNE의 구조는 다음과 같습니다:

1. **입력**: 고차원 데이터셋 $$X = \{x_1, x_2, ..., x_n\}$$
2. **전처리**: PCA로 30차원으로 축소 (계산 효율성)[1]
3. **확률 계산**: Perplexity 기반으로 $$\sigma_i$$ 결정 후 $$p_{ij}$$ 계산
4. **초기화**: $$N(0, 10^{-4}I)$$에서 저차원 포인트 샘플링
5. **반복 최적화**: 1000회 gradient descent 수행
6. **출력**: 2D/3D 시각화 맵 $$Y = \{y_1, y_2, ..., y_n\}$$

### Random Walk 확장
대용량 데이터를 위한 landmark 기반 접근법:
- Neighborhood graph 구성
- Landmark 포인트에서 시작하는 random walk 수행
- 확률적 전이를 통한 유사도 계산[1]

## 성능 향상

### 정량적 성과
1. **MNIST 데이터셋**: 6,000개 숫자 이미지에서 거의 완벽한 클러스터 분리 달성[1]
2. **분류 성능**: 1-nearest neighbor 분류기에서 원본 784차원 데이터(5.75% 오류율) 대비 2차원 t-SNE 표현(5.13% 오류율)에서 더 나은 성능[1]
3. **계산 효율성**: Random walk 버전으로 60,000개 MNIST 데이터 처리 시 1시간 소요[1]

### 기존 기법 대비 우위
- **Sammon mapping**: "공" 형태의 제한적 분리 vs t-SNE의 명확한 클러스터 분리[1]
- **Isomap/LLE**: 클래스 간 큰 중복 vs t-SNE의 우수한 분리[1]
- **PCA**: 선형 변환의 한계 vs t-SNE의 비선형 매니폴드 보존[1]

## 모델의 일반화 성능 향상 가능성

### 장점
1. **Scale Invariance**: Student t-distribution의 특성으로 스케일 변화에 강건[1]
2. **Long-range Forces**: 초기에 분리된 유사 클러스터를 다시 결합시키는 장거리 힘 존재[1]
3. **Robust Optimization**: Simulated annealing 없이도 양질의 local optima 발견[1]
4. **Multi-scale Structure**: 다양한 스케일에서 구조 보존[1]

### 일반화 한계
1. **3차원 이상의 제한**: 3차원 이상 축소 시 t-distribution의 heavy tail이 문제가 될 수 있음[1]
2. **고유 차원의 저주**: 매우 높은 고유 차원(~100차원)의 데이터에서는 local linearity 가정 위반 가능[1]
3. **비볼록 최적화**: 초기값과 매개변수에 따라 결과가 달라질 수 있음[1]

## 한계점

### 1. 차원축소 범용성
t-SNE는 주로 2-3차원 시각화에 특화되어 있어 일반적인 차원축소 작업(d > 3)에서의 성능은 불분명합니다.[1]

### 2. 계산 복잡도
O(n²)의 시간 및 메모리 복잡도로 대용량 데이터 처리에 제한이 있습니다.[1]

### 3. 매개변수 민감성
Perplexity, learning rate 등 여러 매개변수의 선택이 결과에 영향을 미칩니다.[1]

## 연구에 미치는 영향과 고려사항

### 미래 연구에 미치는 영향

1. **시각화 표준화**: t-SNE는 고차원 데이터 시각화의 사실상 표준이 되어 생물정보학, 자연어처리, 컴퓨터 비전 등 다양한 분야에서 활용[2][3]

2. **후속 연구 촉진**: UMAP, LargeVis 등 t-SNE의 한계를 극복한 새로운 기법들의 개발 동기 제공[4]

3. **딥러닝과의 결합**: 신경망의 feature 시각화, 임베딩 공간 분석 등에서 핵심 도구로 활용[3]

4. **매니폴드 학습 발전**: 비선형 차원축소 연구의 새로운 방향 제시[1]

### 향후 연구 시 고려사항

1. **대용량 데이터 처리**: 선형 시간 복잡도를 갖는 근사 알고리즘 개발 필요[4]

2. **매개변수 자동 선택**: Perplexity 등 핵심 매개변수의 자동 최적화 방법 연구[1]

3. **해석 가능성**: 시각화 결과의 신뢰성과 해석 방법에 대한 추가 연구 필요[1]

4. **다차원 확장**: 3차원 이상의 효과적인 차원축소를 위한 개선된 분포 선택[1]

5. **Parametric 버전**: 새로운 데이터에 대한 일반화를 위한 parametric t-SNE 개발[1]

t-SNE는 데이터 시각화 분야에 혁신을 가져온 중요한 연구로, 현재까지도 널리 활용되며 관련 연구의 기반이 되고 있습니다. 특히 복잡한 고차원 데이터의 구조를 직관적으로 이해할 수 있게 해주는 도구로서의 가치가 매우 높습니다.

[1] https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf
[2] https://wikidocs.net/201525
[3] https://learnopencv.com/t-sne-for-feature-visualization/
[4] https://umap-learn.readthedocs.io/en/latest/benchmarking.html
[5] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/d13e58ec-2263-4c50-8da6-b715e33a32b5/vandermaaten08a.pdf
[6] http://papers.neurips.cc/paper/2276-stochastic-neighbor-embedding.pdf
[7] https://wikidocs.net/201649
[8] https://www.flowjo.com/docs/flowjo11/platforms-2-2/tsne-2-2
[9] https://www.cs.toronto.edu/~fritz/absps/sne.pdf
[10] https://www.appsilon.com/post/r-tsne
[11] https://pmc.ncbi.nlm.nih.gov/articles/PMC11578865/
[12] https://devhwi.tistory.com/20
[13] https://lovit.github.io/nlp/representation/2018/09/28/tsne/
[14] https://amt.copernicus.org/articles/13/2995/2020/
[15] https://papers.nips.cc/paper/2276-stochastic-neighbor-embedding
[16] https://www.sciencedirect.com/science/article/pii/S156625351930377X
[17] https://gaussian37.github.io/ml-concept-t_sne/
[18] https://woochan-autobiography.tistory.com/559
[19] https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2021.646936/full
[20] https://3months.tistory.com/571
[21] https://pubs.aip.org/aip/pof/article/35/7/073322/2903032/Comparison-and-evaluation-of-dimensionality
