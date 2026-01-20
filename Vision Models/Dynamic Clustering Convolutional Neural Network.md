# Dynamic Clustering Convolutional Neural Network

### 1. 핵심 주장과 주요 기여

**Dynamic Clustering Convolutional Neural Network (DCCNeXt)**는 CNNs의 근본적인 한계를 혁신적인 방식으로 극복하는 새로운 아키텍처입니다. 논문의 핵심 주장은 다음과 같습니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

**문제점 인식:** 기존 CNN 아키텍처, 특히 ConvNeXt의 7×7 깊이별 분리 컨볼루션도 고정된 지역 윈도우 내에서만 동작하므로, 이미지 내 물체의 장거리 의존성(long-range dependencies)을 포착하기 어렵다는 점을 지적합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

**혁신적 해결책:** DCCNeXt는 전역 클러스터링(global clustering)을 통해 의미론적으로 유사한 이미지 패치들을 동적으로 그룹화하고, 공유 컨볼루션 커널을 사용하여 각 클러스터에 대해 컨볼루션을 수행합니다. 이를 통해 CNNs도 Vision Transformers(ViTs)와 같은 전역적 수용야를 달성할 수 있음을 보입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

**주요 기여:**
- Vision Transformers, MLPs, GNNs, Vision Mambas 등 주류 아키텍처 대비 우월한 성능 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)
- 계산 복잡도 문제를 해결하기 위한 부분벡터 다운샘플링(subvector sampling) 제안 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)
- 이미지 분류, 객체 감지, 인스턴스/의미론적 분할 등 다양한 시각 작업에서의 일반화 가능성 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

***

### 2. 해결하고자 하는 문제 및 제안 방법

#### 2.1 문제 정의

현대 CNNs의 핵심 제약은 **지역적 수용야(local receptive field)**입니다. AlexNet 이후 3×3 커널로 표준화된 이후, 최근 ConvNeXt(2022)도 7×7로 확장했지만, 여전히 이미지의 작은 부분만을 고려합니다. RepLKNet(31×31)과 SLaK(51×51)도 성능 포화 문제와 높은 계산 비용으로 인해 제한적입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

대조적으로 Vision Transformers는 자기주의(self-attention) 메커니즘을 통해 이미지의 모든 패치 간 상호작용을 모델링하여 진정한 전역 수용야를 달성합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

#### 2.2 제안 방법: Dynamic Clustering Convolution (DCConv)

**이미지를 클러스터로 변환:**

이미지의 H×W 패치 각각을 클러스터 중심으로 취급합니다. 먼저 패치들을 특징 벡터로 변환합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

$$X = \{x_1, x_2, ..., x_n\}, \quad x_i \in \mathbb{R}^D$$

여기서 D는 특징 차원입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

**동적 클러스터링:**

각 클러스터 중심과 다른 패치 간의 L2-노름 거리를 계산합니다:

$$\text{distance}(x, y) = (x - y)^2$$

여기서 x는 클러스터 중심, y는 다른 패치입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

거리 행렬 $M \in \mathbb{R}^{n \times n}$에서 Top-K 알고리즘을 사용하여 가장 가까운 K-1개 패치를 선택합니다:

$$\text{idx} = \text{TopkIndex}(M)$$

여기서 $\text{idx} \in \mathbb{R}^{n \times k}$입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

선택된 패치들의 특징 벡터를 추출합니다:

$$X' = \text{IndexSelect}(\text{idx}, X), \quad X' \in \mathbb{R}^{n \times k \times d}$$

**클러스터에 대한 컨볼루션:**

공유 컨볼루션 커널을 사용하여 각 클러스터에 대해 그룹화 컨볼루션을 수행합니다:

$$Y_{i,j,c} = \sum_{k=0}^{K-1} X'_{i*w+j,k,c} \cdot W_{k,c} + b_c$$

여기서:
- Y는 출력 특징
- W는 컨볼루션 커널의 가중치
- b는 편향
- w는 특징맵의 너비입니다 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

**효율적인 동적 클러스터링:**

전체 동적 클러스터링의 계산 복잡도는 이차 복잡도입니다:

$$\Omega(DC) = h^2w^2C + 2hwC$$

여기서 h, w는 특징맵의 높이와 너비, C는 특징 차원입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

고해상도 이미지에서 이 복잡도는 처리 불가능하므로, 부분벡터를 사용합니다:

$$V_{\text{sub}} = \{a_1, a_{1+d}, a_{1+2d}, ..., a_c\}, \quad V_{\text{sub}} \in \mathbb{R}^{C//d}$$

이렇게 하면 효율적 동적 클러스터링의 복잡도는:

$$\Omega(EDC) = h^2w^2\frac{C}{d} + 2hw\frac{C}{d}$$

여기서 d는 샘플링 간격(논문에서는 8로 설정)입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

**Convolution FFN (피드포워드 네트워크):**

국소 정보 추출 능력을 강화하기 위해 FFN에 3×3 깊이별 분리 컨볼루션을 삽입합니다:

$$Y = \text{DWConv}(\text{Linear}_{C \to 4C}(Y))$$

$$Y = \text{Linear}_{4C \to C}(\text{GELU}(Y))$$

여기서 DWConv는 깊이별 컨볼루션, GELU는 활성화 함수입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

***

### 3. 모델 구조

#### 3.1 네트워크 아키텍처

DCCNeXt는 4단계 계층적 구조를 채택합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

**Stage 1-2 (지역 특징 추출):** 7×7 깊이별 분리 컨볼루션 사용
**Stage 3-4 (전역 특징 추출):** Dynamic Clustering Convolution 사용

각 블록은 DCConv(또는 일반 컨볼루션) + FFN + Layer Normalization(LN)으로 구성됩니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

| 모델 | 채널 구성 | 블록 수 | 파라미터(M) | FLOPs(G) |
|------|---------|--------|-----------|----------|
| DCCNeXt-B0 |  [ietresearch.onlinelibrary.wiley](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cvi2.70028) |  [semanticscholar](https://www.semanticscholar.org/paper/5ccf39381dc4d1b80587cfbd4816233d88e4dc2d) | 5.5 | 0.8 |
| DCCNeXt-B1 |  [arxiv](https://arxiv.org/pdf/2508.10057.pdf) |  [dx.plos](https://dx.plos.org/10.1371/journal.pone.0318264) | 11.4 | 1.6 |
| DCCNeXt-B2 |  [arxiv](https://arxiv.org/pdf/2401.09417.pdf) |  [semanticscholar](https://www.semanticscholar.org/paper/5ccf39381dc4d1b80587cfbd4816233d88e4dc2d) | 26.7 | 4.0 |
| DCCNeXt-B3 |  |  [semanticscholar](https://www.semanticscholar.org/paper/5ccf39381dc4d1b80587cfbd4816233d88e4dc2d) | 56.8 | 13.0 |
| DCCNeXt-B4 |  |  [dl.acm](https://dl.acm.org/doi/10.1145/3757324) | 86.7 | 15.7 |

**패치 임베딩:** 입력 RGB 이미지를 7×7 컨볼루션(스트라이드 4)으로 패치로 변환하며, 각 패치의 특징 차원은 모델 크기에 따라 결정됩니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

**Down-sampling 레이어:** 인접한 패치들을 새로운 패치로 병합하여 특징맵을 원래 크기의 1/4로 감소시키고 특징 차원을 증가시킵니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

***

### 4. 성능 향상 및 한계

#### 4.1 성능 평가 결과

**이미지 분류 (ImageNet-1K)**

DCCNeXt-B2는 82.8%의 Top-1 정확도를 달성하여 ConvNeXt V2-T(82.5%)를 능가하면서도 파라미터(26.7M vs 28.6M)와 FLOPs(4.0G vs 4.5G) 측면에서 더 효율적입니다. 더 큰 모델들도 일관된 우월성을 보입니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)
- DCCNeXt-B3: 84.3% (ConvNeXt V2-B 84.3%와 동등)
- DCCNeXt-B4: 84.5% (ConvNeXt V2-B 84.3% 대비 우수) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

클러스터링 기반 아키텍처와의 비교에서도 FEC-Large(28.3M, 6.5G, 81.2%)를 크게 능가합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

**객체 감지 및 인스턴스 분할 (COCO 2017)**

RetinaNet 기반 객체 감지: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)
- DCCNeXt-B2: 43.0 AP (ConvNeXt V2-T 41.7 AP 대비 +2.1%)
- 특히 중형 및 대형 객체 감지에서 우수한 성능 (47.4 APₘ vs 45.2 APₘ)

Mask R-CNN 기반 인스턴스 분할: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)
- DCCNeXt-B2: 44.6 APᵇ, 40.6 APᵐ (ConvNeXt V2-T 42.5 APᵇ, 38.8 APᵐ 대비)
- 2.1 APᵇ, 1.8 APᵐ의 향상 달성

**의미론적 분할 (ADE20K)**

DCCNeXt-B2가 44.5% mIOU를 기록하여 ConvNeXt V2-T(43.7%)와 다른 모델들을 능가합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

#### 4.2 제거 실험 (Ablation Studies)

**컨볼루션 커널 크기의 영향:**

커널 크기 K를 4~18로 조정한 결과, K=12와 K=18(3단계와 4단계에 각각 사용)에서 최적 성능(78.7% Top-1)을 달성합니다. K=4는 너무 작아 불충분한 수용야(78.1%), K=16은 포화(78.5%)를 보입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

**DCCNeXt의 주요 모듈:**

| 구성 | 파라미터(M) | FLOPs(224×224) | FLOPs(1280×800) | ImageNet Top-1(%) |
|------|-----------|----------------|-----------------|------------------|
| DCConv 없음 | 11.2 | 1.52G | 31.0G | 75.3 |
| +DCConv | 11.3 | 1.65G | 78.9G | 77.4 |
| ++Convolution FFN | 11.4 | 1.67G | 79.5G | 78.7 |
| +++Subvector Sampling | 11.4 | 1.58G | 39.2G | 78.7 |

DCConv 제거 시 성능이 크게 하락(75.3% → 77.4%, 차이 2.1%)하여 전역 특징 추출의 중요성을 입증합니다. Convolution FFN은 1.3% 성능 개선을 제공하고, Subvector Sampling은 고해상도 이미지에서 계산 비용을 50% 감소시킵니다(79.5G → 39.2G). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

#### 4.3 시각화 및 해석성

**클러스터 분포 분석:**

Figure 3에서 보여지듯이, DCCNeXt-B2는 7번째 블록에서 국소적 클러스터링을 수행하고, 16번째 블록(깊은 층)에서는 의미론적으로 유사한 패치들을 전역적으로 그룹화합니다. 빨간 패치(클러스터 중심)와 주황색 패치(클러스터 내 다른 패치)의 분포 패턴은 모델이 의미론적 유사성을 효과적으로 학습함을 보입니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

**컨볼루션 커널 가중치 열맵:**

Figure 4는 클러스터 내 패치들의 위치별 가중치 분포를 보여줍니다. 클러스터 중심에 가까운 패치들의 컨볼루션 가중치가 더 크고, 거리가 멀어질수록 감소합니다. 이는 모델이 클러스터 내에서 패치들의 기여도를 거리에 따라 가중치화하고 있음을 의미합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

#### 4.4 논문의 한계

1. **컨볼루션 크기 최적화:** 커널 크기 K의 선택이 경험적이며, 다양한 작업별로 최적 값이 다를 수 있습니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

2. **클러스터링 기하 이해:** Top-K 알고리즘만 사용하므로, 더 정교한 클러스터링 방법(가우시안 혼합 모델, 스펙트럼 클러스터링 등)의 효과는 미탐구입니다.

3. **메모리 효율성:** 고해상도 이미지 처리 시 부분벡터 다운샘플링에도 불구하고, 추가적인 메모리 사용량 분석이 부족합니다.

4. **소규모 데이터셋 일반화:** ImageNet-1K과 같은 대규모 데이터셋에서는 우수하지만, 소규모 데이터셋(예: CIFAR-10)에서의 성능 평가가 제시되지 않습니다.

***

### 5. 모델의 일반화 성능 향상 가능성

#### 5.1 일반화의 정의와 중요성

머신러닝에서 **일반화**는 훈련 데이터에서 학습한 모델이 미지의 테스트 데이터에 얼마나 잘 적용되는지를 측정합니다. CNNs와 Vision Transformers 간의 근본적인 차이는 이 일반화 능력에 있습니다. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12522997/)

#### 5.2 DCCNeXt의 일반화 메커니즘

**전역 모델링의 이점:**

DCCNeXt는 초기 층부터 전역 의존성을 모델링하므로, Vision Transformers(ViTs)처럼: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)
- 거리가 먼 객체 간의 관계를 초기에 포착 가능 [aicompetence](https://aicompetence.org/vision-transformers-vs-cnns/)
- 고정된 커널 크기의 제약 없이 유연한 특징 추출

**계층별 수용야 확대:**

CNN은 여러 층을 거쳐야 전역 수용야를 달성하지만, DCCNeXt는 3-4단계부터 전역 클러스터링을 도입하여 수렴 속도를 개선합니다. 이는 특히 다운스트림 작업에서 두드러집니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

#### 5.3 다운스트림 작업에서의 일반화 우위

**객체 감지에서의 향상:**

이미지 분류보다 객체 감지에서 더 큰 성능 향상(+2.1% AP)을 보입니다. 이는 물체의 위치와 크기 변화에 대한 강건성이 개선되었음을 의미합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

$$\text{DCCNeXt 향상도} = \text{탐지 성능} - \text{분류 성능 향상도}$$

분류에서 0.3% 향상 대비, 탐지에서 2.1% 향상은 전역 모델링이 공간 정보 보존에 특히 유효함을 보여줍니다.

**의미론적 분할에서의 강건성:**

픽셀 수준의 미세한 예측이 필요한 의미론적 분할에서도 우수한 성능을 유지합니다. 이는 DCCNeXt가 국소적 디테일(Convolution FFN)과 전역 문맥(DCConv)을 균형있게 통합함을 시사합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

#### 5.4 일반화 향상을 위한 설계 원리

1. **하이브리드 특징 추출:**
   - Stage 1-2: 국소 특징 (3×3 깊이별 분리 컨볼루션)
   - Stage 3-4: 전역 특징 (Dynamic Clustering Convolution)
   
   이 설계는 CNNs의 귀납 편향(inductive bias) 이점과 Transformers의 전역 모델링을 결합합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

2. **적응형 클러스터링:**
   
   각 샘플마다 동적으로 클러스터 구성이 변경되므로, 입력 데이터의 특성에 맞춘 학습이 가능합니다. 고정된 컨볼루션 커널과 달리 데이터 분포에 반응합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

3. **Convolution FFN의 역할:**
   
   식 (7)-(8)의 Convolution FFN은 Transformer의 MLP 블록처럼 비선형 변환을 수행하면서도, 3×3 DWConv를 통해 국소적 정보 집약을 강화합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

***

### 6. 최신 관련 연구와의 비교 (2020년 이후)

| 아키텍처 | 연도 | 핵심 기여 | 제약점 | DCCNeXt와의 관계 |
|---------|------|---------|-------|------------------|
| **ConvNeXt** | 2022 | 7×7 깊이별 분리 컨볼루션 | 고정 지역 수용야 | ConvNeXt 기반 설계, 전역 모델링으로 개선 |
| **RepLKNet** | 2022 | 31×31 재매개변수화 커널 | 포화 + 높은 계산 비용 | 희소 커널 대신 동적 클러스터링으로 효율화 |
| **SLaK** | 2023 | 51×51 희소 분해 | 최소 성능 향상 | 고정 희소성 대신 데이터 기반 클러스터링 |
| **Vision Transformer (ViT)** | 2020 | 전역 자기주의 | 데이터 요구량 많음 | 같은 전역성 + CNN 효율성 |
| **Swin Transformer** | 2021 | 윈도우 기반 자기주의 | 복잡한 시프팅 | 더 간단한 전역 클러스터링 |
| **Context-Cluster (CoC)** | 2023 | 이미지를 점 집합으로 모델링 | 지역 윈도우 제약 | 전역 클러스터링으로 확장 |
| **FEC (Feature Extraction with Clustering)** | 2024 | 신경망 클러스터링 기반 표현학습 | 단순 평균 집약 | 가중치 기반 집약, 더 나은 성능 |
| **Vision Mamba (ViM)** | 2024 | 상태 공간 모델의 선형 복잡도 | 시퀀스 모델링에 특화 | CNN 구조 유지 + 전역 모델링 |
| **UniRepLKNet** | 2024 | 다중 모달리티 대형 커널 ConvNet | 모달리티 간 일반화 | 단일 시각 모달리티 최적화 |

#### 6.1 주요 경쟁 아키텍처 분석

**ConvNeXt V2 vs DCCNeXt:**
- ConvNeXt V2-T: 82.5% (28.6M params, 4.5G FLOPs) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)
- DCCNeXt-B2: 82.8% (26.7M params, 4.0G FLOPs) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

DCCNeXt는 더 적은 파라미터로 더 높은 정확도를 달성하면서도, 다운스트림 작업에서 더 큰 이점을 보입니다.

**FEC vs DCCNeXt:**

FEC(2024)도 클러스터링 기반이지만: [github](https://github.com/guikunchen/FEC/)
- 클러스터링이 지역 윈도우에 제약됨 [github](https://github.com/guikunchen/FEC/)
- 각 패치가 한 번에 한 클러스터에만 할당됨 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)
- 특징 집약이 단순 평균 방식 [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Chen_Neural_Clustering_based_CVPR_2024_supplemental.pdf)

반면 DCCNeXt: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)
- 전역 클러스터링으로 더 먼 패치 간 상호작용 가능
- 각 패치가 여러 클러스터 중심의 영향을 받음 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)
- 거리 기반 가중치 집약으로 더 정교한 특징 추출

**Context-Cluster (CoC) vs DCCNeXt:**

CoC(2023): [arxiv](https://arxiv.org/pdf/2303.01494.pdf)
- 이미지를 정돈되지 않은 점 집합으로 모델링
- 국소 지역 내에서만 클러스터링 [arxiv](https://arxiv.org/pdf/2303.01494.pdf)

DCCNeXt: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)
- H×W 패치 모두를 클러스터 중심으로 취급
- 전역 L2-노름 거리 기반 클러스터링 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

***

### 7. 앞으로의 연구에 미치는 영향

#### 7.1 패러다임 전환의 의의

**CNN의 재혁신:**

Vision Transformers의 등장 이후 많은 연구자들이 CNNs의 쇠퇴를 예상했습니다. 하지만 DCCNeXt는: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)
- CNNs도 전역 모델링을 효과적으로 구현 가능함을 입증
- "It is time for CNNs to fight back"이라는 논문의 기조처럼, CNN 아키텍처의 새로운 가능성을 제시 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

이는 2023-2025년 간 CNNs의 재평가 추세와 일치합니다. [aicompetence](https://aicompetence.org/vision-transformers-vs-cnns/)

#### 7.2 하이브리드 아키텍처 설계의 지침

DCCNeXt의 설계 원리는 향후 하이브리드 아키텍처 개발에 다음 지침을 제공합니다: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

1. **계층별 특징 추출 전략:** 초기 층에서 귀납 편향, 후기 층에서 전역 모델링
2. **동적 구조 적응:** 고정된 컨볼루션 커널 대신 데이터 기반 클러스터링
3. **효율성과 표현력의 균형:** 부분벡터 샘플링으로 계산 복잡도 감소

#### 7.3 다양한 도메인으로의 확장 가능성

**의료 영상 분석:**

의료 이미지는 높은 해상도와 세밀한 구조 인식이 필요합니다. DCCNeXt의 전역 모델링 + 국소 특징 추출 조합은: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)
- 장기 전체 구조 파악 (전역 클러스터링)
- 미세 병변 감지 (Convolution FFN)

에 동시에 효과적일 수 있습니다.

**자율 주행:**

실시간 객체 감지에서 DCCNeXt는: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)
- ViTs의 높은 계산 비용 문제 회피
- CNNs의 효율성 유지 + Transformers 성능

을 달성하여 엣지 디바이스 배포에 유리합니다.

**3D 비전:**

최근 점 구름(point cloud) 처리 연구와의 연계 가능성: [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0893608025003831)
- DCCNeXt의 "이미지를 점 집합으로 보는" 개념 (Figure 1의 Flatten 단계) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)
- 3D 점 구름 처리 네트워크와의 직접적 통합 가능

#### 7.4 이론적 진전

**일반화 이론:**

현재 연구에서 부족한 부분:
1. DCCNeXt와 ViTs 간 일반화 성능의 이론적 비교
2. 동적 클러스터링이 VC-차원(VC-dimension) 또는 라데마허 복잡도(Rademacher complexity)에 미치는 영향

이런 분석이 진행되면 CNN의 일반화 경계에 대한 새로운 이해가 가능합니다.

***

### 8. 향후 연구 시 고려할 점

#### 8.1 방법론적 개선

**1. 고급 클러스터링 알고리즘 통합:**

현재 Top-K 기반 단순 클러스터링 대신: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)
- K-means++ 초기화로 더 안정적인 클러스터 생성
- 가우시안 혼합 모델(GMM)로 확률적 클러스터 할당
- 스펙트럼 클러스터링으로 비유클리드 구조 포착

이런 고도화는 성능을 2-3% 추가 향상시킬 수 있습니다. [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Chen_Neural_Clustering_based_CVPR_2024_supplemental.pdf)

**2. 적응형 커널 크기:**

현재 고정 K 값(논문에서 최적은 12, 18) 대신: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)
- 각 층의 특징 맵 특성에 따른 동적 K 설정
- 훈련 가능한 K 학습 메커니즘

$$K_l = f_\theta(\text{feature statistics}_l)$$

**3. 멀티-헤드 클러스터링:**

현재 단일 클러스터링 대신: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)
- 여러 부분공간에서 병렬 클러스터링
- 앙상블 방식 클러스터 중심 결합

이는 서로 다른 의미론적 수준의 특징을 동시에 포착합니다.

#### 8.2 실험 설계 개선

**1. 소규모 데이터셋 평가:**

ImageNet-1K 외에:
- CIFAR-10/100: 작은 모델의 일반화 능력 검증
- 의료 이미지 데이터셋 (PathImageNet, MedMNIST): 도메인 특화 성능
- 롱-테일 데이터셋: 클래스 불균형 환경에서의 강건성

**2. 계산 효율성 벤치마킹:**

현재 FLOPs 기준 외에: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)
- 실제 메모리 사용량 (peak memory, memory throughput)
- 다양한 하드웨어(CPU, GPU, TPU, 모바일 가속기)에서의 지연 시간
- 에너지 소비량 (탄소 발자국)

**3. 견고성(Robustness) 평가:**

- 적대적 공격 저항성 (adversarial examples)
- 분포 시프트 하에서의 성능 (domain adaptation, out-of-distribution detection)
- 입력 노이즈, 가우시안 블러, 기하학적 변환에 대한 저항성

#### 8.3 이론적 분석

**1. 동적 클러스터링의 수렴 속성:**

클러스터링이 훈련 과정에서 어떻게 진화하는지:
- 초기 에포크 vs 후기 에포크의 클러스터 안정성
- 수렴 속도에 미치는 영향

**2. 클러스터링과 정규화(Regularization) 효과:**

동적 클러스터링이 암시적 정규화 역할을 하는지:
- Dropout과의 유사성
- 일반화 경계에 대한 이론적 분석

**3. 특징 학습 역학:**

$$\frac{d\mathbf{x}_i}{dt} = -\frac{\partial \mathcal{L}}{\partial \mathbf{x}_i}$$

클러스터링이 특징 진화에 미치는 영향을 분석합니다.

#### 8.4 응용 확장

**1. 다중 모달리티 학습:**

DCCNeXt의 원리를 비전-언어, 비전-오디오 결합 작업으로 확장:
- 이미지 패치와 텍스트 토큰 간 동적 클러스터링
- 크로스 모달 의미론적 정렬 개선

**2. 효율적인 배포:**

- 지식 증류(knowledge distillation): 큰 DCCNeXt → 작은 CNN/ViT
- 모델 프루닝: 불필요한 클러스터링 연산 제거
- 양자화: 저정밀 클러스터링 계산

**3. 자기감독학습(Self-Supervised Learning):**

DCCNeXt의 동적 클러스터링을 비감독 학습에 활용:
- 클러스터 일관성을 보존하는 손실 함수 설계
- 심 네트워크(siamese networks)와의 결합

***

### 결론

**Dynamic Clustering Convolutional Neural Network (DCCNeXt)**는 CNNs의 근본적 제약을 극복하는 혁신적 접근법입니다. Vision Transformers의 전역 모델링 능력을 CNNs의 효율성 이점과 결합하여, 이미지 분류, 객체 감지, 인스턴스/의미론적 분할 등 다양한 시각 작업에서 우월한 성능을 달성합니다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

특히 **일반화 성능 측면에서**, DCCNeXt는:
1. 초기 층부터 전역 의존성을 모델링하여 Vision Transformers와 유사한 일반화 이점 제공 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)
2. 국소 및 전역 특징을 균형있게 통합하여 다양한 스케일의 구조 포착 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)
3. 다운스트림 작업에서 더 큰 성능 향상을 보여, 전이학습 가능성이 우수함 [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

향후 고급 클러스터링 알고리즘의 통합, 소규모 데이터셋에서의 일반화 검증, 그리고 다중 모달리티 확장은 이 아키텍처의 영향력을 더욱 확대할 것으로 예상됩니다.

***

### 참고문헌 (References)

 Li, T., Zhang, B., Lyu, J., Zheng, X., Guo, G., & Jin, T. (2025). Dynamic Clustering Convolutional Neural Network. Proceedings of the Thirty-Ninth AAAI Conference on Artificial Intelligence (AAAI-25). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/b7dab59d-08e7-4a44-824d-6501e39dd625/34030-Article-Text-38098-1-2-20250410.pdf)

 Vision Transformers Vs CNNs: Who Leads Vision In 2025? (2025). AI Competence. [aicompetence](https://aicompetence.org/vision-transformers-vs-cnns/)

 Takahashi, S., et al. (2024). Comparison of Vision Transformers and Convolutional Neural Networks. PMC, 2024. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC12522997/)

 Gai, Y., et al. (2025). Interpretable unsupervised neural network structure for clustering. Neurocomputing. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0893608025003831)

 Ma, X., Zhou, Y., Wang, H., Qin, C., Sun, B., Liu, C., & Fu, Y. (2023). Image as Set of Points. ICLR. [arxiv](https://arxiv.org/pdf/2303.01494.pdf)

 Chen, G., Li, X., Yang, Y., & Wang, W. (2024). Neural clustering based visual representation learning. CVPR, 5714-5725. [github](https://github.com/guikunchen/FEC/)

 Chen, G., et al. (2024). Neural Clustering based Visual Representation Learning (Supplementary). CVPR 2024. [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Chen_Neural_Clustering_based_CVPR_2024_supplemental.pdf)
