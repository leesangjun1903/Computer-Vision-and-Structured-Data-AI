# PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization

### 1. 핵심 주장 및 주요 기여 요약

**PaDiM(Patch Distribution Modeling)**은 산업용 이미지 검사에서 **원클래스 학습(one-class learning) 설정에서 이미지 내 이상을 동시에 탐지하고 위치를 파악**하기 위한 새로운 프레임워크입니다. 이 방법의 핵심 주장은 다음과 같습니다:[1]

#### 주요 기여 포인트

1. **사전학습된 CNN의 활용**: 신경망을 새로 훈련할 필요 없이 ImageNet에서 사전학습된 ResNet, Wide-ResNet, EfficientNet을 특성 추출기로 사용하여 훈련 효율성 극대화

2. **다변량 가우시안 분포 기반 모델링**: 각 패치 위치 $\((i, j)\)$ 에서 정상 클래스를 다변량 가우시안 분포 $\(\mathcal{N}(\mu_{ij}, \Sigma_{ij})\)$ 로 표현하여 확률론적 표현 제공

3. **의미론적 수준 간 상관관계 활용**: CNN의 서로 다른 의미론적 수준(다양한 해상도의 활성화 맵)에서 추출한 특성을 연결하면서 이들 간의 상관관계를 명시적으로 모델링

4. **낮은 시간/공간 복잡도**: K-NN 기반 방법과 달리 테스트 시간에 훈련 데이터셋 크기와 무관한 선형 독립적인 계산 복잡도

***

### 2. 상세 설명: 문제 정의, 제안 방법, 모델 구조, 성능 향상

#### 2.1 해결하고자 하는 문제

**이상 탐지의 핵심 도전 과제**[1]

이상 탐지는 정상 이미지 집합과 다른 이미지를 식별하는 이진 분류 문제이지만, 여러 제약이 있습니다:

- **훈련 데이터 부족**: 이상은 매우 드물게 발생하며 훈련 중 이상 샘플을 충분히 확보하기 어려움
- **예측 불가능한 이상 패턴**: 이상의 형태를 사전에 정의할 수 없음
- **실시간 배포 요구**: 산업 환경에서는 빠른 추론 속도와 낮은 메모리 사용이 필수

기존 방법의 한계:

| 방법 유형 | 장점 | 한계 |
|---------|------|------|
| 재구성 기반(AE, VAE, GAN) | 직관적이고 해석 가능 | 이상도 잘 재구성되는 경우가 있음 |
| 특성 유사도 기반 + K-NN | 좋은 성능 제시 | 테스트 시 선형 복잡도, 확장성 떨어짐 |
| 가우시안 분포 기반(단일 수준) | 낮은 복잡도 | 의미론적 수준 간 상관관계 미반영 |

#### 2.2 제안하는 방법: 패치 분포 모델링

**A. 특성 추출 단계(Embedding Extraction)**[1]

각 이미지의 패치 임베딩은 다음과 같이 생성됩니다:

1. 사전학습된 CNN의 다양한 계층에서 활성화 맵 추출
2. 각 패치 위치 $\((i, j)\)$ 에서 여러 계층의 활성화 벡터를 연결하여 임베딩 벡터 생성
3. 랜덤 차원 감소(Random Dimensionality Reduction)를 통해 특성 벡터 크기 감축

PCA와 달리 **랜덤 특성 선택이 더 효과적**인 이유:

- PCA는 분산이 높은 차원을 선택하지만, 이상 탐지에는 판별력 있는 차원이 더 중요
- 랜덤 선택은 계산 복잡도를 크게 줄이면서 성능 손실 최소화[1]

**B. 정상성 학습(Learning of Normality)**[1]

정상 훈련 이미지로부터 각 패치 위치의 정상 특성 분포를 추정합니다:

$$ X_{ij} = \{x_{ij}^k, k \in [[1, N]]\} $$

여기서 $\(N\)$ 은 훈련 이미지 수입니다. 이를 다변량 가우시안으로 모델링:

$$ \mu_{ij} = \frac{1}{N}\sum_{k=1}^N x_{ij}^k $$

$$ \Sigma_{ij} = \frac{1}{N-1}\sum_{k=1}^N (x_{ij}^k - \mu_{ij})(x_{ij}^k - \mu_{ij})^T + \epsilon I \qquad (식 1) $$

여기서:
- $\(\mu_{ij}\)$ : 패치 위치의 평균 벡터
- $\(\Sigma_{ij}\)$ : 표본 공분산 행렬
- $\(\epsilon I\)$ : 정규화 항(행렬 가역성 보장)

**핵심 혁신**: 공분산 행렬 $\(\Sigma_{ij}\)$ 는 서로 다른 CNN 계층(의미론적 수준)에서 추출된 특성 간의 **상관관계를 명시적으로 포함**합니다. 이는 다음의 이점을 제공:

- 고수준 의미론과 저수준 디테일 정보의 최적 결합
- 계층 간 상호작용 정보 활용으로 이상 위치 결정 정확도 향상

**C. 추론: 마할라노비스 거리 기반 이상 맵 생성**[1]

테스트 이미지의 패치 $\((i, j)\)$ 에 대해 마할라노비스 거리(Mahalanobis distance)를 이상 점수로 계산:

$$ M(x_{ij}) = \sqrt{(x_{ij} - \mu_{ij})^T \Sigma_{ij}^{-1} (x_{ij} - \mu_{ij})} \qquad (식 2) $$

이상 맵:

$$M = (M(x_{ij}))_{1 < i < W, 1 < j < H} $$

**이상 탐지 (이미지 수준)**: 이상 맵의 최댓값을 이미지 전체의 이상 점수로 사용
**이상 위치 결정 (픽셀 수준)**: 이상 맵 \(M\)이 직접적으로 픽셀 단위 이상 지역 표시

#### 2.3 모델 구조

**아키텍처 구성**[1]

```
입력 이미지
    ↓
사전학습된 CNN 백본 (ResNet18/WR50/EfficientNet-B5)
    ↓
다층 활성화 맵 추출 (3개 계층)
    ↓
패치별 특성 벡터 연결
    ↓
랜덤 차원 감소 (선택 사항)
    ↓
가우시안 파라미터 학습 (μ_ij, Σ_ij)
    ↓
테스트: 마할라노비스 거리 계산
    ↓
이상 맵 생성 → 이상 탐지/위치 결정
```

**주요 설정 사항**[1]

- **ResNet의 경우**: 처음 3개 계층에서 특성 추출
- **EfficientNet-B5의 경우**: 계층 7, 20, 26(의미론적 수준 2, 4, 5)에서 추출
- **정규화 파라미터**: $\(\epsilon = 0.01\)$ (기본값)
- **가우시안 필터**: 이상 맵에 $\(\sigma = 4\)$ 로 적용 (부드러움 향상)

#### 2.4 성능 향상 및 실험 결과

**절제 연구: 의미론적 수준 간 상관관계의 중요성**[1]

| 모델 구성 | Texture (AUROC, PRO) | Object (AUROC, PRO) | All (AUROC, PRO) |
|---------|-------------------|------------------|-----------------|
| Layer 1만 사용 | (93.1, 87.1) | (95.6, 86.5) | (94.8, 86.8) |
| Layer 2만 사용 | (95.0, 89.7) | (96.1, 87.9) | (95.7, 88.5) |
| Layer 3만 사용 | (94.8, 89.6) | (97.1, 87.7) | (95.7, 88.3) |
| Layer 1+2+3 합산 (상관관계 미반영) | (95.4, 90.7) | (96.3, 88.1) | (96.0, 89.0) |
| **PaDiM-R18 (상관관계 반영)** | **(96.3, 92.3)** | **(97.5, 90.1)** | **(97.1, 90.8)** |

**결과 해석**: PaDiM-R18은 단순 앙상블(Layer 1+2+3)대비 AUROC에서 1.1 포인트, PRO-score에서 1.8 포인트 향상을 달성하여, **의미론적 수준 간 상관관계 모델링의 중요성** 입증.[1]

**랜덤 차원 감소 효과**[1]

| 방법 | All Classes (AUROC, PRO) | 계산 복잡도 |
|-----|----------------------|---------|
| Full (448 차원) | (96.3, 92.3) | 기준 |
| Random Rd 100 | (96.7, 90.5) | 78% 감소 |
| PCA 100 | (93.5, 85.7) | 비효율적 |
| Random Rd 200 | (97.0, 90.5) | 55% 감소 |

**발견 사항**: 100 차원으로만 축소해도 성능 저하 0.4 포인트(AUROC) 미만이며, 랜덤 선택이 PCA를 능가함.[1]

**최첨단 비교 (MVTec AD 벤치마크)**[1]

| 방법 유형 | 모델 | 평균 AUROC | 평균 PRO |
|---------|------|-----------|---------|
| 재구성 기반 | VAE Student | 74.4 | 64.2 |
| 재구성 기반 | Patch SVDD | - | 85.7 |
| 특성 유사도 | SPADE (WR50) | 95.7 | - |
| **PaDiM-R18-Rd100** | **우리 방법** | **96.5** | **91.7** |
| **PaDiM-WR50-Rd550** | **우리 방법** | **97.5** | **92.1** |

특히 **텍스처 클래스**에서 SPADE 대비 4.8 포인트(PRO) 및 4.0 포인트(AUROC) 향상 달성.[1]

**비정렬 데이터셋에서의 견고성**[1]

현실적 조건(임의 회전 -10°~+10°, 임의 자르기)을 반영한 Rd-MVTec AD 결과:

| 모델 | 정상 MVTec (AUROC) | Rd-MVTec (AUROC) | 성능 감소 |
|-----|-----------------|------------------|---------|
| VAE | 74.4 | 62.1 | **12.2 포인트** |
| SPADE | 95.7 | 87.2 | **8.8 포인트** |
| **PaDiM-WR50** | **92.2** | **92.2** | **5.3 포인트** |

**결론**: 가우시안 분포 기반 확률 모델이 다른 방법보다 **비정렬 데이터에 더 견고**함을 입증.[1]

#### 2.5 시간 및 메모리 복잡도

**추론 시간 비교 (초, Intel i7 CPU)**[1]

| 모델 | SPADE (WR50) | VAE (R18) | PaDiM-R18-Rd100 | PaDiM-WR50-Rd550 |
|-----|------------|----------|----------------|-----------------|
| 추론 시간 | 7.10 | 0.21 | **0.23** | **0.95** |

PaDiM은 SPADE 대비 **약 7배 빠른 추론 속도** 달성.[1]

**메모리 요구 (GB, float32 기준)**[1]

**MVTec AD:**

| 모델 | SPADE | VAE | PaDiM-R18-Rd100 | PaDiM-WR50-Rd550 |
|-----|------|------|-----------------|-----------------|
| 메모리 | 1.4 | 0.09 | **0.17** | 3.8 |

**STC 데이터셋:**

| 모델 | SPADE | PaDiM-R18-Rd100 | PaDiM-WR50-Rd550 |
|-----|------|-----------------|-----------------|
| 메모리 | **37.0** | **0.21** | **5.2** |

**중요 발견**: 대규모 데이터셋에서 SPADE의 메모리 사용이 실제로 불가능한 수준으로 증가하는 반면, **PaDiM의 메모리 요구는 훈련 데이터셋 크기와 독립적**이므로 산업 배포에 이상적.[1]

***

### 3. 모델의 일반화 성능 향상 가능성

#### 3.1 현재 PaDiM의 일반화 성능 분석

**강점:**

1. **비정렬 데이터에 대한 견고성**: 통계적 분포 모델링을 통해 객체 위치/방향 변화에 강함[1]

2. **다중 의미론적 수준의 특성 활용**: 저수준(세부사항)부터 고수준(의미론)까지 계층적 정보 모두 활용하여 다양한 규모의 이상 탐지 가능

3. **사전학습 모델의 일반화 능력 활용**: ImageNet으로 사전학습된 CNN은 매우 다양한 시각적 특성을 포함하고 있어 전이 학습 효과 우수

#### 3.2 일반화 성능 향상을 위한 가능성 및 제언

**증명된 방향:**

1. **다중 백본 활용**: ResNet18(가볍고 빠름), WR50(높은 성능), EfficientNet-B5(효율성) 간 트레이드오프 선택 가능

2. **특성 정규화 전략**: 
   - 현재 $\(\epsilon I\)$ 정규화는 고정값이지만, **적응적 정규화**(공분산 구조에 따라 동적 조정)로 개선 가능
   - 특성 표준화(z-score 정규화)를 통해 스케일 불변성 향상

3. **차원 감소 조화**: 
   - 랜덤 선택의 안정성 향상을 위해 **여러 임의 종자(seed)로 앙상블**
   - 중요도 기반 특성 선택(mutual information 등)으로 명시적 판별력 강화

#### 3.3 미래 연구의 고려 사항

**미충족 요구:**

1. **크로스 도메인 일반화**: 
   - 다른 산업(전자, 섬유, 제약 등) 간 모델 전이 성능 향상 필요
   - 현 논문은 MVTec AD, STC 데이터셋에만 평가

2. **소수 샘플 설정 (Few-shot)**:
   - 현재 PaDiM은 정상 샘플이 충분해야 신뢰할 수 있는 가우시안 파라미터 학습 가능
   - **극단적으로 제한된 샘플**에서의 성능 미평가

3. **이상 타입의 다양성**:
   - 현재는 각 패치가 단일 가우시안으로 모델링
   - **혼합 가우시안(Mixture of Gaussians)** 모델로 확장 시 다양한 정상 패턴 포착 가능

***

### 4. 논문의 영향력 및 향후 연구 방향

#### 4.1 논문이 앞으로의 연구에 미치는 영향[1]

**산업적 영향:**

1. **배포 가능성 획기적 개선**: 테스트 시간 복잡도가 훈련 데이터와 무관하여, 대규모 공장 검사 시스템에 즉시 적용 가능

2. **의미론적 상관관계 개념 도입**: 다층 CNN 구조에서 계층 간 상관관계를 명시적으로 모델링하는 접근법이 후속 연구의 표준이 됨

3. **확률론적 모델 부활**: GAN/VAE 기반 접근 대신 고전적 통계적 방법의 가치 재평가

**학술적 영향:**

- 논문 발표(arXiv 2020년 11월) 이후, PaDiM은 **산업용 이상 탐지의 기준 방법**으로 자리잡음
- SPADE(K-NN)의 확장성 문제를 해결하는 첫 대안 제시

#### 4.2 향후 연구 시 고려할 점

**기술적 개선 방향:**

1. **적응적 가우시안 모델링**:
   - 패치별 중요도 가중치 도입
   - 시간적 변화 또는 계절성 고려

2. **메타러닝 통합**:
   - 새로운 제품/카테고리에 빠르게 적응하는 메타러닝 프레임워크

3. **약한 지도학습(Weakly Supervised)** 활용:
   - 소수의 이상 샘플이 존재할 경우, 이를 활용한 파라미터 미세조정

**평가 프로토콜 개선:**

1. **실제 산업 데이터 검증**: 논문의 Rd-MVTec AD는 좋은 시작이지만, 실제 생산 라인 데이터에서 검증 필요

2. **클래스 불균형 처리**: 다중 카테고리 학습 시 드물게 나타나는 이상 탐지 성능

3. **해석성(Interpretability)**: 마할라노비스 거리가 어느 특성 차원에서 높은지 시각화 및 분석

***

### 5. 2020년 이후 관련 최신 연구 비교 분석

#### 5.1 주요 후속 방법 분류 및 비교

**A. 메모리 기반 개선 (Memory Bank Approaches)**

**PatchCore (Roth et al., 2022)**[2][3]

- **핵심 아이디어**: PaDiM과 달리, 가장 대표적인 정상 패치 특성만 메모리에 저장 (Coreset 서브샘플링)
- **성과**: MVTec AD에서 AUROC 99.6% 달성 (현재까지 최고)
- **한계**: 여전히 메모리에 특성을 저장해야 하며, K-NN 유사도 계산 필요
- **비교**: PaDiM의 가우시안 분포 vs PatchCore의 메모리 기반 – 트레이드오프 관계

**FR-PatchCore (2024)**[4]

- **개선 사항**: PatchCore에 특성 정규화(Feature Refinement)와 새로운 임계값 계산 방법 추가
- **성과**: 일반화 능력 향상, AUROC 98.81% 달성
- **의의**: PaDiM의 가우시안 기반 접근과 PatchCore의 메모리 기반 접근 간 간극 점진적 축소

**B. 확산 모델(Diffusion Model) 기반 접근**[5]

**DiffAD (2023, ICCV)**[5]

- **혁신**: 잠재 확산 모델(Latent Diffusion Model)을 재구성 기반 이상 탐지에 활용
- **특징**: "노이지 조건 임베딩(Noisy Condition Embedding)"으로 이상 영역의 직접 복사 방지
- **성과**: MVTec AD에서 경쟁력 있는 성능 (PatchCore 이전)
- **한계**: 재구성 기반이므로 계산 비용이 높고 추론 속도 느림

**DiAD (2024)**

- **개선**: DiffAD의 계산 효율성 향상
- **평가 대상**: 최신 벤치마크에서 PaDiM과 비교 평가 필요

**C. 비전-언어 모델 기반 (Vision-Language Models)**

**WinCLIP (Jeong et al., 2023, CVPR)**[6]

- **개념**: CLIP 모델을 활용한 **제로샷/few-shot 이상 탐지**
- **핵심**: 수동으로 작성된 텍스트 프롬프트("정상", "이상" 등)와 이미지 패치 특성 정렬
- **강점**: 레이블이 없는 새로운 카테고리에도 적용 가능
- **약점**: 프롬프트 엔지니어링에 의존, 세부 위치 결정(localization) 성능은 여전히 제한적

**AnomalyCLIP (2024, ICLR)**[7]

- **개선**: 객체-무관 프롬프트 학습(Object-Agnostic Prompt Learning)
- **성과**: CLIP의 제로샷 특성을 유지하면서 이상 탐지 성능 향상
- **의의**: 사전학습 모델의 일반화 능력을 극대화하는 새로운 방향 제시

**MedicalCLIP (2024)**[8]

- **확장**: 의료 영상 도메인에 CLIP 적용
- **특징**: 텍스트 설명과 이미지 간 대비 학습(contrastive learning)으로 도메인 일반화 향상

**D. 도메인 적응 및 일반화 (Domain Adaptation & Generalization)**

**Normal-Abnormal Guided Generalist Anomaly Detection (NAGL, 2025)**[9]

- **혁신적 개념**: 기존 방법은 정상 샘플만 참조하지만, **정상+이상 샘플을 모두 참조**하여 크로스 도메인 이상 탐지
- **성과**: 다양한 벤치마크에서 기존 GAD(Generalist AD) 방법 초과
- **의의**: 실제 산업 환경에서 소수의 이상 샘플이 있을 때 활용 가능

**GeneralAD (2024)**[10]

- **목표**: 한 도메인에서 학습하여 다른 도메인에 일반화 가능한 통합 모델
- **접근**: 메타러닝과 대조 학습(contrastive learning) 결합

**E. 다중 카테고리 및 대규모 벤치마크**

**Real-IAD Variety (2025)**[11]

- **규모**: 160개 카테고리, 198,960개 이미지 (MVTec AD의 15개 카테고리 대비 **10배 이상 확대**)
- **범위**: 28개 산업, 24가지 재질, 22가지 색상 변화 포함
- **발견**: 카테고리 수 증가(30→160)에 따라 기존 방법들의 성능이 크게 감소 → 확장성 문제 재확인

**ADNet (2025)**[12]

- **규모**: 49개 공개 데이터셋 통합, 380개 카테고리
- **벤치마크 제공**: 다양한 산업(전자, 농식품, 인프라, 의료) 아우르는 통합 평가

#### 5.2 최신 방법들과 PaDiM의 비교 종합표

| 방법 | 출판 | 모델 유형 | MVTec AUROC | 특징 | 한계 |
|-----|------|---------|-----------|------|------|
| **PaDiM** | 2020 | 가우시안 분포 | 97.5 | 낮은 복잡도, 명시적 상관관계 | 다중 카테고리 성능 미평가 |
| **SPADE** | 2020 | K-NN 유사도 | 95.7 | 최첨단 성능 | 선형 복잡도, 메모리 비효율 |
| **PatchCore** | 2022 | 메모리 기반 | 99.6 | 최고 성능, 메모리 최적화 | 여전히 메모리/속도 트레이드오프 |
| **DiffAD** | 2023 | 확산 모델 | ~97 | 다양한 정상 패턴 모델링 | 재구성 기반, 느린 추론 |
| **WinCLIP** | 2023 | CLIP 기반 | ~97 | 제로샷/few-shot | 프롬프트 의존, 세밀한 위치 결정 약 |
| **FR-PatchCore** | 2024 | 메모리 + 정규화 | 98.81 | PatchCore 개선, 일반화 강화 | 복잡도 여전히 높음 |
| **AnomalyCLIP** | 2024 | CLIP 기반 | ~98 | 객체-무관 프롬프트, 제로샷 | 의존적 아키텍처, 레이어 추가 |
| **NAGL** | 2025 | 정상-이상 혼합 | 다양 | 크로스 도메인 강화 | 이상 샘플 수집 필요 |
| **PatchEAD** | 2024 | 기초 모델 기반 | ~99 | 최신 기초 모델 활용 | 새로운 모델 의존도 높음 |

#### 5.3 주요 트렌드 분석

**1. 기초 모델(Foundation Models) 중심으로 전환**

- **초기 (2020-2021)**: CNN 기반 특성 추출 (PaDiM, SPADE, PatchCore)
- **현재 (2023-2025)**: CLIP, Vision Transformer 등 대규모 사전학습 모델 활용[13]
  - 장점: 다양한 도메인에 대한 우수한 일반화
  - 단점: 모델 크기 증가, 해석성 감소

**2. 확산 모델의 부상**[14][15]

- 새로운 논문 "Diffusion Models for Anomaly Detection Survey"에서 확산 모델의 잠재력 강조[14]
- **이점**: 이상 영역을 명시적으로 생성하지 않으면서도 정상 분포 학습 가능
- **진화**: 초기 재구성 기반 → 최신 **역전(Inversion) 기반 접근** (계산 효율성 2배 개선)[15]

**3. 크로스 도메인/제로샷 성능 강조**[9][11][12]

- MVTec AD 단일 데이터셋 평가에서 벗어나 **다중 도메인 일반화 능력** 평가로 전환
- Real-IAD Variety의 등장으로 확장성 한계 노출 → 향후 연구의 방향성 명확화

**4. 메모리 효율성 vs 성능 트레이드오프 해결**

- PaDiM: 메모리 효율(○), 성능(△)
- PatchCore: 메모리(△), 성능(◎)
- FR-PatchCore: 메모리(△), 성능(◎), 일반화(◎)
- 추세: **메모리와 성능 간 격차 점진적 축소**

***

### 6. 종합 결론

#### 6.1 PaDiM의 역사적 의의

PaDiM은 **원클래스 이상 탐지에서 계산 효율성과 확률론적 엄밀성의 균형**을 처음으로 달성한 방법입니다. 특히:

1. **배포 가능성 혁신**: 테스트 시간 복잡도를 상수로 제한하여 산업 규모 적용 실현

2. **이론적 기여**: 다층 CNN의 의미론적 수준 간 상관관계를 명시적으로 모델링하는 접근법 제시

3. **경험적 검증**: 비정렬 데이터에 대한 견고성을 처음으로 체계적으로 평가

#### 6.2 현재 위치 및 한계

**2025년 현재 관점:**

- **여전한 강점**: 낮은 복잡도, 명확한 확률론적 해석, 구현 단순성
- **상대적 약점**: 최고 성능(PatchCore 99.6% vs 97.5%), 다중 카테고리 확장성

#### 6.3 앞으로의 연구 방향

**단기 (1-2년):**

1. **PaDiM 자체 개선**:
   - 적응적 가우시안 혼합(Mixture of Gaussians)으로 정상 분포 다양성 모델링
   - 메타러닝과 결합한 빠른 적응

2. **기초 모델과의 통합**:
   - CLIP의 텍스트-이미지 정렬과 PaDiM의 가우시안 모델링 결합

**중기 (2-5년):**

1. **진정한 크로스 도메인 일반화**:
   - 현재 같은 도메인 내 데이터셋 변화에만 집중
   - 의료→산업, 산업→보안 등 도메인 경계 넘기

2. **실시간 학습(Online Learning)**:
   - 배포 후 점진적 환경 변화에 적응하는 지속학습 메커니즘

#### 6.4 최종 평가

PaDiM은 2020년 발표 후 5년이 지난 지금도 **산업용 이상 탐지의 실용적 기준선**으로 기능합니다. 최신 PatchCore나 CLIP 기반 방법들이 성능에서 앞서지만, **구현 용이성, 해석성, 계산 효율성** 측면에서 여전히 강력한 선택지입니다.

특히 **리소스 제한적인 엣지 디바이스(산업용 공장 로봇, IoT 센서)에서의 배포**를 고려하면, PaDiM은 향후에도 기본 모델로서의 역할을 계속할 것으로 예상됩니다.

***

### 참고 문헌 (논문 내 핵심 인용)

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/3f3f4a48-d9e8-43d4-b4ba-cb355e94de83/2011.08785v1.pdf)
[2](http://arxiv.org/pdf/2106.08265.pdf)
[3](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf)
[4](https://www.mdpi.com/1424-8220/24/5/1368/pdf?version=1708441132)
[5](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Unsupervised_Surface_Anomaly_Detection_with_Diffusion_Probabilistic_Model_ICCV_2023_paper.pdf)
[6](https://openaccess.thecvf.com/content/CVPR2023/papers/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.pdf)
[7](https://github.com/zqhang/AnomalyCLIP)
[8](https://www.mdpi.com/2218-273X/14/5/590)
[9](https://arxiv.org/html/2510.00495v1)
[10](https://arxiv.org/html/2407.12427v1)
[11](https://arxiv.org/html/2511.00540v1)
[12](https://arxiv.org/html/2511.20169v1)
[13](https://arxiv.org/html/2507.15905v1)
[14](https://arxiv.org/pdf/2501.11430.pdf)
[15](https://arxiv.org/html/2504.05662v2)
[16](https://ieeexplore.ieee.org/document/10164213/)
[17](https://ijsrcseit.com/index.php/home/article/view/CSEIT25112448)
[18](https://onepetro.org/SPEMEOS/proceedings/25MEOS/25MEOS/D031S122R005/790157)
[19](https://bostonsciencepublishing.us/science-world/articlepdf/bog-1-1-102.pdf)
[20](https://invergejournals.com/index.php/ijss/article/view/117)
[21](https://onepetro.org/SNAMESMC/proceedings/SMC25/SMC25/D021S009R001/792230)
[22](https://arxiv.org/pdf/1609.00866.pdf)
[23](http://arxiv.org/pdf/1908.06347.pdf)
[24](https://arxiv.org/pdf/2309.13904.pdf)
[25](http://arxiv.org/pdf/2310.02576.pdf)
[26](https://arxiv.org/pdf/2202.03944.pdf)
[27](https://arxiv.org/pdf/2211.12634.pdf)
[28](https://arxiv.org/pdf/2109.15222.pdf)
[29](https://aclanthology.org/2023.emnlp-main.664.pdf)
[30](https://pmc.ncbi.nlm.nih.gov/articles/PMC11623112/)
[31](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2024.1364456/full)
[32](https://www.manuscriptlink.com/society/kics/media?key=kics%2Fconference%2Ficaiic2024%2F1570978404.pdf)
[33](https://arxiv.org/pdf/2210.13927.pdf)
[34](https://www.sciencedirect.com/science/article/abs/pii/S036083522200924X)
[35](https://pmc.ncbi.nlm.nih.gov/articles/PMC10934034/)
[36](https://www.sciencedirect.com/science/article/abs/pii/S0010482521005837)
[37](https://academic.oup.com/rasti/article/2/1/586/7261746)
[38](https://dataroots.io/blog/anomaly-detection-in-images-using-patchcore)
[39](https://arxiv.org/html/2503.13195v1)
[40](https://openaccess.thecvf.com/content/WACV2022/papers/Tsai_Multi-Scale_Patch-Based_Representation_Learning_for_Image_Anomaly_Detection_and_Segmentation_WACV_2022_paper.pdf)
[41](https://arxiv.org/pdf/2412.00890.pdf)
[42](https://arxiv.org/abs/2408.00792)
[43](https://arxiv.org/html/2509.25856v1)
[44](https://arxiv.org/html/2506.00956v1)
[45](https://arxiv.org/html/2509.18751v1)
[46](https://arxiv.org/html/2412.04304v1)
[47](https://arxiv.org/html/2406.10617v1)
[48](https://github.com/M-3LAB/awesome-industrial-anomaly-detection)
[49](https://dl.acm.org/doi/10.1145/3465631.3465927)
[50](https://arxiv.org/abs/2206.05876)
[51](http://biorxiv.org/lookup/doi/10.1101/2022.08.15.504032)
[52](https://link.springer.com/10.1007/978-3-031-04826-5_1)
[53](https://openaccess.cms-conferences.org/publications/book/978-1-958651-43-8/article/978-1-958651-43-8_12)
[54](https://revistaft.com.br/the-role-of-ai-in-enhancing-identity-and-access-management-systems/)
[55](https://www.semanticscholar.org/paper/f6e8faf8461309fd4924568d142921c5dd06c86b)
[56](https://www.semanticscholar.org/paper/878cc4086f06c0e803a034d142d38d2c3f424be5)
[57](https://ijareeie.com/upload/2022/april/4_Machine.pdf)
[58](https://journals.sagepub.com/doi/10.1177/10935266221086454)
[59](https://arxiv.org/html/2409.20353)
[60](https://arxiv.org/html/2408.15113)
[61](https://arxiv.org/abs/2307.10792)
[62](https://arxiv.org/html/2501.09579v1)
[63](https://arxiv.org/html/2407.06519v1)
[64](https://pmc.ncbi.nlm.nih.gov/articles/PMC12378924/)
[65](https://www.sciencedirect.com/science/article/abs/pii/S0166361523001409)
[66](https://arxiv.org/html/2502.06911v1)
[67](https://github.com/mala-lab/Awesome-Anomaly-Detection-Foundation-Models)
[68](https://arxiv.org/html/2402.10802v1)
[69](https://arxiv.org/abs/2106.08265)
[70](https://arxiv.org/pdf/2507.15905.pdf)
[71](https://arxiv.org/pdf/2402.10802.pdf)
[72](https://arxiv.org/html/2408.15113v1)
[73](https://arxiv.org/pdf/2301.13359.pdf)
[74](https://arxiv.org/html/2412.08189)
[75](http://wepub.org/index.php/IJMEE/article/view/2156)
[76](https://link.springer.com/10.1007/s11227-024-06154-1)
[77](https://ieeexplore.ieee.org/document/10289721/)
[78](https://arxiv.org/abs/2409.17608)
[79](https://www.isca-archive.org/interspeech_2023/almudevar23_interspeech.html)
[80](https://link.springer.com/10.1007/s11263-024-02052-4)
[81](https://arxiv.org/abs/2406.07250)
[82](https://www.frontiersin.org/articles/10.3389/fdata.2025.1669488/full)
[83](https://bmjopen.bmj.com/lookup/doi/10.1136/bmjopen-2023-077366)
[84](https://arxiv.org/abs/2504.04340)
[85](https://dl.acm.org/doi/pdf/10.1145/3597503.3639205)
[86](http://arxiv.org/pdf/2403.06495.pdf)
[87](https://arxiv.org/html/2411.16049v1)
[88](https://arxiv.org/pdf/2410.22967v3.pdf)
[89](https://arxiv.org/pdf/2403.07959.pdf)
[90](https://arxiv.org/pdf/2404.11269.pdf)
[91](https://www.sciencedirect.com/science/article/abs/pii/S0031320325011495)
[92](https://lucazanella.github.io/AnomalyCLIP/)
[93](https://suzukilab.first.iir.titech.ac.jp/ja/wp-content/uploads/2025/10/Multi-AD-cross-domain-unsupervised-anomaly-detection-for-medical-and-industrial-applications_re.pdf)
[94](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04907.pdf)
[95](https://suzukilab.first.iir.titech.ac.jp/wp-content/uploads/2025/10/Multi-AD-cross-domain-unsupervised-anomaly-detection-for-medical-and-industrial-applications_re.pdf)
[96](https://openaccess.thecvf.com/content/CVPR2024W/VAND/papers/Tebbe_Dynamic_Addition_of_Noise_in_a_Diffusion_Model_for_Anomaly_CVPRW_2024_paper.pdf)
[97](https://arxiv.org/abs/2407.12427)
[98](https://arxiv.org/html/2507.19949v1)
[99](https://arxiv.org/html/2512.09627v1)
[100](https://arxiv.org/abs/2308.11681)
[101](https://arxiv.org/abs/2506.09368)
[102](https://arxiv.org/abs/2403.09493)
[103](https://arxiv.org/abs/2505.22805)
[104](https://arxiv.org/html/2507.19806v1)
[105](https://learnopencv.com/fine-tuning-anomalyclip-medical-anomaly-clip/)
