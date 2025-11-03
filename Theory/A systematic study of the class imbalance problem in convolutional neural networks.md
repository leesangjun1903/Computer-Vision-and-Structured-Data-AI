# A systematic study of the class imbalance problem in convolutional neural networks

### 1. 핵심 주장 및 주요 기여 (간결 요약)

이 논문은 **CNN에서 클래스 불균형이 분류 성능에 미치는 영향을 처음으로 체계적으로 연구**했습니다. 논문의 핵심 기여는 다음과 같습니다.[1]

**주요 발견**

- **불균형의 해로운 영향**: 클래스 불균형이 분류 성능을 명백하게 저하시킨다는 것을 실증적으로 증명했습니다.[1]
- **오버샘플링의 우월성**: 비교된 모든 방법 중에서 **오버샘플링이 대부분의 시나리오에서 가장 효과적**임을 보여주었습니다.[1]
- **CNN의 특이성**: 기존 머신러닝과 달리 **CNN은 오버샘플링으로 인한 과적합이 발생하지 않음**을 실증했습니다.[1]
- **실용적 권고사항**: 불균형 정도에 따른 구체적인 메서드 선택 기준을 제시했습니다.[1]

***

### 2. 문제 정의 및 해결 방법

#### 2.1 해결하고자 하는 문제

CNN이 의료 진단, 사기 탐지, 컴퓨터 비전 등 현실의 많은 응용 분야에서 널리 사용되는 반면, **클래스 불균형은 여전히 미해결 문제**로 남아있었습니다. 예를 들어, 질병 클래스가 정상 클래스보다 1000배 적을 수 있으며, 이는 모델이 다수 클래스를 선호하는 편향된 예측을 하도록 유도합니다.[1]

#### 2.2 제안하는 방법들

논문에서 비교한 7가지 방법은 크게 데이터 레벨과 알고리즘 레벨로 분류됩니다:[1]

**데이터 레벨 방법 (Data-level Methods)**

1. **Random Minority Oversampling (무작위 소수 오버샘플링)**
   - 소수 클래스의 표본을 무작위로 반복 추출하여 데이터 수를 증가시킵니다.[1]
   - 구현이 간단하지만 정보 손실이 없다는 장점이 있습니다.

2. **Random Majority Undersampling (무작위 다수 언더샘플링)**
   - 다수 클래스의 일부 표본을 제거하여 클래스 균형을 맞춥니다.[1]
   - 계산 비용 감소는 장점이지만, 정보 손실이 발생합니다.

**알고리즘 레벨 방법 (Classifier-level Methods)**

3. **Thresholding (임계값 조정)**
   - 신경망 출력의 클래스 확률을 조정하여 사전 클래스 확률을 보정합니다.[1]

다음 공식을 통해 바이어스를 수정합니다:[1]

$$
y_i(x) = p(i|x) = \frac{p(i) \cdot p(x|i)}{p(x)}
$$

여기서 $$p(i) = \frac{|i|}{\sum_k |k|}$$ 는 클래스 $i$의 사전 확률입니다.[1]

**하이브리드 방법 (Hybrid Methods)**

4-7. **Two-Phase Training 및 조합 방법들**
   - 균형 데이터에서 먼저 사전 학습한 후, 원본 데이터에서 출력층을 미세 조정합니다.[1]
   - 오버샘플링/언더샘플링과 임계값 조정을 조합합니다.

#### 2.3 평가 메트릭

기존 정확도(accuracy) 대신 **ROC AUC (Area Under Receiver Operating Characteristic Curve)**를 주평가 메트릭으로 사용했습니다. 이는 불균형 데이터에서 정확도의 한계(예: 다수 클래스 편향)를 극복하기 위함입니다.[1]

***

### 3. 모델 구조 및 실험 설계

#### 3.1 사용된 CNN 모델

| 데이터셋 | 모델 | 이유 |
|---------|------|------|
| MNIST | LeNet-5 | 간단한 작업, 빠른 학습 |
| CIFAR-10 | All-CNN | 중간 복잡도 |
| ImageNet | ResNet-10 | 높은 복잡도, 배치 정규화 포함 |

#### 3.2 불균형의 정의

논문은 두 가지 대표적인 불균형 형태를 정의했습니다:[1]

**Step Imbalance (계단식 불균형)**

소수 클래스와 다수 클래스가 명확하게 구분되는 형태입니다. 두 개의 파라미터로 정의됩니다:

$$
\mu = \frac{|\{i \in \{1, \ldots, N\} : C_i \text{ is minority}\}|}{N}
$$

$$
\rho = \frac{\max_i |C_i|}{\min_i |C_i|}
$$

예: 10개 클래스 중 5개가 500개 표본, 5개가 5,000개 표본이면 $$\rho = 10, \mu = 0.5$$[1]

**Linear Imbalance (선형 불균형)**

클래스 간 표본 수가 선형적으로 변하는 형태입니다. 하나의 파라미터 $$\rho$$로만 정의됩니다.[1]

#### 3.3 실험 규모

- **MNIST**: 4,500개의 불균형 데이터셋 생성, 22,500개 신경망 학습[1]
- **CIFAR-10**: 640개 네트워크 학습[1]
- **ImageNet**: 15개 ResNet-10 네트워크 학습[1]

***

### 4. 성능 향상 및 주요 결과

#### 4.1 불균형의 영향 분석

실험 결과는 **불균형의 영향이 작업의 복잡도에 따라 달라진다**는 것을 보여줍니다.[1]

- **MNIST**: 불균합 비율이 100배 증가했을 때 ROC AUC 약 5% 감소[1]
- **CIFAR-10**: 같은 불균합 비율에서 **약 100배 더 크게 성능 저하** (약 5% 감소는 훨씬 작은 불균합에서 발생)[1]
- **ImageNet**: 극단적 불균합에서 ROC AUC 99.41에서 90.74로 **급격한 저하**[1]

이는 모델의 복잡도가 높을수록 불균합에 더 취약함을 의미합니다.

#### 4.2 방법 간 비교 분석

**오버샘플링의 우월성**

- 거의 모든 시나리오에서 **최고 성능**을 달성했습니다.[1]
- 기준선(do-nothing) 대비 일관되게 **성능 개선**을 보였습니다.[1]

**언더샘플링의 한계**

- 대부분 시나리오에서 기준선보다 **나쁜 성능**을 보였습니다.[1]
- 다수 클래스에서 정보를 버리기 때문에, 소수 클래스의 비율이 높을 때만 약간의 이점이 있었습니다.[1]

**Two-Phase Training의 성능**

- 일반적으로 기준선과 오버/언더샘플링 사이의 **중간 성능**을 보였습니다.[1]
- 오버샘플링이 이미 우수할 때 미세조정은 성능을 **오히려 악화**시켰습니다.[1]

#### 4.3 오버샘플링의 최적화 수준

흥미로운 발견은 **오버샘플링의 최적 수준**입니다:[1]

- MNIST (불균합 비율 1,000)의 경우:
  - 완전 균형화(완전 오버샘플링)가 최고 성능
  - 부분 오버샘플링은 일관되게 더 나쁜 성능

언더샘플링의 경우, **최적 수준은 불균합 정도에 따라 달라집니다**.[1]

***

### 5. 일반화 성능 향상 가능성 (중점 분석)

#### 5.1 오버샘플링과 과적합: 반직관적 발견

**고전 머신러닝의 통설 vs. CNN의 현실**

전통적인 머신러닝에서는 오버샘플링이 과적합을 유발한다는 것이 알려져 있습니다. 그러나 이 논문의 핵심 발견 중 하나는:[1]

> "오버샘플링은 CNN에서 과적합을 유발하지 않는다"

이를 검증하는 증거는 CIFAR-10 실험에서 확인됩니다.[1]

**수렴 분석**

논문은 훈련과 테스트 정확도 차이를 비교했습니다:

- **기준선**: 훈련과 테스트 사이의 **큰 격차** (과적합 징후)
- **오버샘플링**: 훈련과 테스트 **간격이 일정**하게 유지 (과적합 없음)
- **언더샘플링**: 오버샘플링만큼 효과적은 아니지만 기준선보다 **나은 일반화**

**왜 이런 차이가 발생하는가?**

CNN의 특성이 이를 설명합니다:[1]

1. **자동 특성 추출**: CNN은 계층적으로 특징을 학습하여 인공적 반복에 덜 민감함
2. **정규화의 영향**: 신경망의 가중치 감소와 드롭아웃 같은 내재적 정규화 메커니즘이 과적합을 방지
3. **스케일의 효과**: 합성곱 필터의 특성상 정확한 픽셀 반복은 학습에 미치는 영향이 제한적

#### 5.2 임계값 조정의 역할

**ROC AUC vs. 정확도**

이 논문은 중요한 구분을 제시합니다:[1]

- **ROC AUC**: 분류기의 진정한 **판별 능력**을 측정
- **정확도**: 클래스 사전 확률의 영향을 받음

임계값 조정(thresholding)은 **정확도를 개선**하지만 ROC AUC는 변경하지 않습니다. 이는 다음 공식으로 설명됩니다:[1]

ROC 곡선은 양의 실수에 의한 의사결정 변수의 곱셈에 불변이므로, 임계값 이동은 ROC 곡선을 변경하지 않습니다.[1]

#### 5.3 불균합 수준에 따른 일반화 성능

**중등도 불균합 ($$\rho = 10 \sim 50$$)**

- 오버샘플링이 일관되게 최고의 일반화 성능 제공
- 완전 균형화 후 약 1-2% 성능 향상

**극단적 불균합 ($$\rho = 100 \sim 1,000$$)**

- ImageNet의 극단적 경우 ($$\rho = 100$$)에서 오버샘플링 성능이 약간 저하
- 저자들은 이를 **신중하게 해석**: 하이퍼파라미터 변동성과 제한된 데이터 규모 때문일 수 있음[1]

---

### 6. 한계 및 제약 사항

이 논문에는 몇 가지 명확한 한계가 있습니다:[1]

| 한계 | 설명 |
|-----|------|
| **ImageNet 분석의 제한성** | 극단적 불균합에 대한 제한된 실험. 최대 10% 데이터만 사용하여 최적 하이퍼파라미터 불일치 가능성 |
| **SMOTE 미포함** | 고급 오버샘플링 기법(SMOTE, Borderline-SMOTE 등) 미포함 |
| **앙상블 방법 미포함** | 계산 비용 등의 이유로 앙상블 메서드 미포함 |
| **비지도 및 준지도 학습 미포함** | 레이블이 있는 분류 문제만 다룸 |
| **시각 도메인 한정** | 자연어처리나 다른 도메인으로의 일반화 정도 불명확 |

***

### 7. 앞으로의 연구 영향 및 고려사항 (최신 연구 기반)

#### 7.1 이 논문의 학술적 영향

이 논문은 출판 이후 **매우 높은 인용도**를 기록했으며, 불균형 데이터 처리의 **기준이 되는 연구**로 인식되고 있습니다. 최근 연구들은 이 논문의 발견을 토대로 발전했습니다.

#### 7.2 최신 연구 동향 (2023-2025)

**1. 손실함수 기반 접근 (Loss Function-based Approaches)**

전통적 오버샘플링을 넘어 **특화된 손실함수**가 제안되고 있습니다:[2][3][4]

- **Focal Loss**: 어려운 샘플에 더 큰 가중치를 부여합니다. 2023년 연구에서 의료 영상에서 **ROC AUC 99.41에서 95.06으로 개선**을 보였습니다.[2]

- **Dual Focal Loss (DFL)**: Focal Loss의 개선판으로, 양의 클래스와 음의 클래스 모두에 대해 손실을 고려하여 **소실 기울기 문제 해결**.[3][4]

- **AutoBalance**: 정확도와 공정성을 균형있게 최적화하는 **쌍단계 최적화 프레임워크**.[5]

**2. 고급 데이터 증강 기법 (Advanced Augmentation Techniques)**

단순 오버샘플링을 넘어, 더 정교한 증강 방법이 개발되었습니다:[6][7][8][9][10]

- **SMOTE 확장**: Borderline-SMOTE, K-means SMOTE, DBSCAN-SMOTE 등이 **원본 특성 분포를 더 잘 보존**합니다.[6]

- **MixUp/CutMix**: 선형 보간이나 이미지 영역 혼합을 통한 증강. **MixUp과 CutMix가 CutOut보다 우수**한 것으로 보고되었습니다.[11][10]

- **Intra-Class CutMix**: 소수 클래스의 **클래스 내 샘플들을 혼합**하여 결정 경계 수정.[9]

**3. 대조학습 기반 방법 (Contrastive Learning)**

최근 강력한 추세는 **대조학습**을 활용한 접근입니다:[12][13][14]

- **Targeted Supervised Contrastive Learning (TSC)**: 사전 정의된 균형 클래스 중심점을 목표로 학습하여 **특성 공간의 균형을 보장**합니다.[12]

- **Global Contrastive Learning (CoGloAT)**: k-positive 학습의 한계를 개선하여 **머리 클래스와 꼬리 클래스 모두의 학습 능력 향상**.[13]

- **Subclass-balancing Contrastive Learning (SBCL)**: 클래스를 부분클래스로 분해하여 **인스턴스 균형과 부분클래스 균형 동시 달성**.[14]

**4. 도메인 적응 및 일반화**

불균합 데이터에서의 **도메인 간 일반화**가 새로운 연구 방향입니다:[15]

- **Domain Generalization Networks**: Mixup 기반 증강과 도메인 불일치 메트릭을 결합하여 **분포 변화에 강건한 모델 개발**.[15]

- **이론적 토대**: 2025년 연구에서 **H-consistency 증명과 Rademacher 복잡도 기반 학습 보장**을 제시하는 이론적 프레임워크 제안.[16]

**5. 하이브리드 및 엔드-투-엔드 방법**

**AutoSMOTE** (2025): 합성 데이터 생성을 분류기와 **함께 최적화**하는 엔드-투-엔드 프레임워크로, 매개변수 탐색 공간 감소 및 과적합 완화.[17]

#### 7.3 앞으로의 연구 시 고려할 점

**의료 영상 등 도메인 특화 연구의 필요성**

최근 2024년 연구는 **의료 데이터의 특이성**을 강조합니다:[18]

> "SMOTE가 의료 데이터에서 임상적 유효성이 의심스럽다. 20개의 원본 환자에서 62개의 합성 환자 생성이 실제 변동성을 반영하지 못할 수 있다."[18]

이는 도메인에 따라 **다른 불균합 처리 전략이 필요**함을 시사합니다.

**이론적 토대의 강화**

근래 연구들이 **비용-민감 학습의 비일관성**(non-Bayes consistency)을 지적하고 있습니다. 따라서:[16]

- 기존 경험적 방법에서 **수학적 보장이 있는 방법**으로의 전환
- **클래스 민감 복잡도** 개념을 통한 일반화 경계 도출

**극단적 불균합에서의 해결책**

이 논문이 다루지 못한 **매우 극단적인 불균합** ($$\rho > 1,000$$)에 대해:

- **한 클래스 분류** (One-class classification)로의 전환
- **생성 모델** (GAN) 기반 균형화
- **강화학습** 기반 샘플링 전략

**평가 메트릭의 발전**

단순 ROC AUC를 넘어:

- **클래스별 F1 점수의 가중 평균**
- **비용 행렬을 고려한 맞춤형 메트릭**
- **신뢰도 보정** (Calibration)을 포함한 평가

**계산 효율성과 확장성**

- 대규모 데이터셋에서의 **오버샘플링 계산 비용** 문제
- **온라인 학습** 시나리오에서의 불균합 처리
- **분산 학습** 환경에서의 샘플링 전략

---

### 결론

Buda, Maki, Mazurowski의 이 연구는 **심층학습 시대의 불균합 문제를 처음으로 체계적으로 분석**한 선구적 논문입니다. 오버샘플링의 우월성, CNN에서의 과적합 부재 등의 발견은 실무적으로 매우 중요한 가이드라인을 제공했습니다.[1]

그러나 최근 7년간의 발전을 보면, **단순 오버샘플링을 넘어 고급 손실함수, 대조학습, 도메인 적응** 등이 새로운 표준으로 부상하고 있습니다. 특히 의료 영상이나 극단적 불균합 등 특정 응용 분야에서는 **도메인 특화 해결책의 필요성**이 강조되고 있습니다. 향후 연구는 이론적 토대 강화와 함께 실세계 데이터의 복잡성을 더욱 반영하는 방향으로 진행될 것으로 예상됩니다진행될 것으로 예상됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/58b0d132-b2db-4dca-90f5-3ac39d1052cb/1710.05381v2.pdf)
[2](https://pmc.ncbi.nlm.nih.gov/articles/PMC10289178/)
[3](https://www.sciencedirect.com/science/article/abs/pii/S0925231221011310)
[4](https://www.arxiv.org/pdf/1909.11932v2.pdf)
[5](https://arxiv.org/pdf/2201.01212.pdf)
[6](https://pubs.rsc.org/en/content/articlehtml/2025/sc/d5sc00270b)
[7](https://arxiv.org/html/2304.02858)
[8](https://ieeexplore.ieee.org/document/10722815/)
[9](https://www.semanticscholar.org/paper/Intra-Class-Cutmix-for-Unbalanced-Data-Augmentation-Zhao-Lei/e53b5687897178cda08207665adac33a9b214e55)
[10](https://arxiv.org/pdf/2309.09970.pdf)
[11](https://openaccess.thecvf.com/content/ICCV2023W/CVAMD/papers/Rao_Studying_the_Impact_of_Augmentations_on_Medical_Confidence_Calibration_ICCVW_2023_paper.pdf)
[12](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Targeted_Supervised_Contrastive_Learning_for_Long-Tailed_Recognition_CVPR_2022_paper.pdf)
[13](https://openreview.net/forum?id=xWrtiJwJj5)
[14](https://openaccess.thecvf.com/content/ICCV2023/papers/Hou_Subclass-balancing_Contrastive_Learning_for_Long-tailed_Recognition_ICCV_2023_paper.pdf)
[15](https://www.nature.com/articles/s41598-024-75088-8)
[16](https://arxiv.org/pdf/2502.10381.pdf)
[17](https://arxiv.org/pdf/2502.06878.pdf)
[18](https://www.mdpi.com/2504-4990/6/2/39)
[19](https://www.mdpi.com/2504-4990/7/3/105)
[20](https://bmccancer.biomedcentral.com/articles/10.1186/s12885-025-14876-5)
[21](http://biorxiv.org/lookup/doi/10.1101/2025.04.27.649481)
[22](https://aca.pensoft.net/article/151406/)
[23](https://arxiv.org/pdf/2411.05733.pdf)
[24](https://arxiv.org/pdf/1710.05381.pdf)
[25](http://arxiv.org/pdf/1807.06538.pdf)
[26](https://arxiv.org/pdf/2207.06080.pdf)
[27](http://arxiv.org/pdf/2412.12984.pdf)
[28](https://arxiv.org/pdf/2312.02517.pdf)
[29](https://arxiv.org/pdf/2412.20656.pdf)
[30](https://arxiv.org/pdf/2404.15593.pdf)
[31](https://www.ijcesen.com/index.php/ijcesen/article/view/3367)
[32](https://www.sciencedirect.com/science/article/pii/S2666827024000732)
[33](https://www.nature.com/articles/s41598-025-04952-y)
[34](https://www.sciencedirect.com/science/article/pii/S2666202725002502)
[35](https://ieeexplore.ieee.org/document/10872126/)
[36](https://www.semanticscholar.org/paper/b1e60c2515d2431a2dd2cc03791b8c0ccc855c36)
[37](https://www.opastpublishers.com/open-access-articles/addressing-challenges-in-data-quality-and-model-generalization-for-malaria-detection.pdf)
[38](https://ieeexplore.ieee.org/document/10957672/)
[39](https://ieeexplore.ieee.org/document/10498918/)
[40](https://ieeexplore.ieee.org/document/10796751/)
[41](https://ieeexplore.ieee.org/document/10708720/)
[42](https://ieeexplore.ieee.org/document/10922797/)
[43](https://arxiv.org/pdf/2102.09554.pdf)
[44](https://arxiv.org/pdf/2103.01550.pdf)
[45](https://arxiv.org/html/2308.08638v2)
[46](https://arxiv.org/pdf/2205.12070.pdf)
[47](https://arxiv.org/pdf/1905.09872.pdf)
[48](https://isi-web.org/sites/default/files/2024-02/Handling-Data-Imbalance-in-Machine-Learning.pdf)
[49](https://openreview.net/pdf?id=DLqPhQxgYu)
