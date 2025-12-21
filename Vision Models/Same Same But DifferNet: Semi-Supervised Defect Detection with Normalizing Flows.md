
# Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows

### **1. 핵심 주장 및 주요 기여 요약**

이 논문은 결함 데이터 없이 정상 데이터만으로 학습하는 **준지도(Semi-supervised) 결함 탐지** 방법론인 **DifferNet**을 제안합니다.
*   **핵심 주장:** 이미지의 고차원 픽셀을 직접 모델링하는 대신, CNN으로 추출한 특징(Features)의 밀도(Density)를 **Normalizing Flows(NF)**로 추정하면 결함을 효과적으로 탐지할 수 있다.
*   **주요 기여:**
    1.  **Multi-scale Feature & NF:** CNN 특징 추출과 Normalizing Flow를 결합하여 고차원 이미지에서도 안정적인 밀도 추정을 가능하게 함.
    2.  **Multi-transform Robustness:** 입력 이미지에 다양한 변환(회전, 스케일 등)을 적용하여 학습 및 추론함으로써 모델의 견고성(Robustness)을 크게 향상.
    3.  **Pixel-wise Localization:** 별도의 픽셀 단위 레이블 없이도, 결함 점수의 기울기(Gradient)를 역전파(Backpropagation)하여 결함 위치를 시각화하는 방법 제안.
    4.  **Few-shot Efficiency:** 단 16장의 정상 이미지만으로도 높은 탐지 성능을 달성.

***

### **2. 상세 설명: 문제, 방법, 구조, 성능 및 한계**

#### **2.1. 해결하고자 하는 문제 (Problem Statement)**
제조업 등에서 발생하는 결함(Defect)은 매우 드물고 그 형태가 예측 불가능합니다. 따라서 결함 데이터를 확보하여 학습하는 지도 학습(Supervised Learning)은 불가능합니다. 이 논문은 **정상 데이터만으로 학습**하여, 정상 범주에서 벗어난 샘플을 결함으로 분류하는 **이상 탐지(Anomaly Detection)** 문제를 다룹니다.

#### **2.2. 제안하는 방법 (Proposed Method)**
DifferNet은 사전 학습된 CNN으로 특징을 추출하고, Normalizing Flow를 통해 이 특징들이 정규분포(Latent Space)에 매핑되도록 학습합니다.

*   **특징 추출 (Feature Extraction):** 입력 이미지 $x$로부터 사전 학습된 CNN(AlexNet 등)을 통해 특징 $y = f_{ex}(x)$를 추출합니다. 이때 다양한 스케일(Multi-scale)의 특징을 연결(Concatenation)하여 사용합니다.
*   **밀도 추정 (Density Estimation via NF):** 추출된 특징 $y$를 Normalizing Flow $f_{NF}$를 통해 잠재 벡터 $z$로 변환합니다. $z$는 표준 정규분포 $p_Z(z) \sim \mathcal{N}(0, I)$를 따르도록 강제됩니다.
*   **손실 함수 (Loss Function):** 데이터의 음의 로그 우도(Negative Log-Likelihood, NLL)를 최소화합니다.


   $$\mathcal{L}(\theta) = - \mathbb{E}_{x} \left[ \log p_Z(f_{NF}(y)) + \log \left| \det \frac{\partial f_{NF}}{\partial y} \right| \right] $$
    
여기서 $\log p_Z(z) = -\frac{\|z\|_2^2}{2} - \text{const}$ 이므로, 결국 $z$의 L2 Norm을 최소화하고 야코비안(Jacobian) 행렬식의 로그 값을 최대화하는 방향으로 학습됩니다.

*   **결함 점수 (Anomaly Score):** 테스트 시, 입력 이미지에 여러 변환 $T_i$ (회전, 크기 조절 등)를 적용한 후 평균 우도를 계산하여 점수화합니다. 점수가 높을수록 결함일 확률이 높습니다.

    $$\mathcal{A}(x) = \frac{1}{K} \sum_{i=1}^{K} \left( \frac{\| z_i \|\_2^2}{2} - \log \left| \det \frac{\partial f_{NF}}{\partial y_i} \right| \right) $$

여기서 $z_i = f_{NF}(f_{ex}(T_i(x)))$ 입니다.

#### **2.3. 모델 구조 (Model Structure)**
1.  **Fixed Feature Extractor:** ImageNet으로 사전 학습된 AlexNet을 사용하여 가중치를 고정(Freeze)하고 특징만 추출합니다.
2.  **Normalizing Flow (Real-NVP):** 추출된 특징을 입력으로 받아 가역적(Invertible) 변환을 수행하는 Coupling Layer들을 쌓아 구성됩니다. 이 네트워크만이 학습 대상입니다.
3.  **Evaluation Head:** 출력된 $z$ 벡터의 L2 Norm을 기반으로 최종 점수를 산출하고, 이를 역전파하여 입력 이미지 상의 결함 위치(Gradient Map)를 생성합니다.

#### **2.4. 성능 향상 및 한계**
*   **성능:** 발표 당시 MVTec AD 벤치마크에서 평균 **94.9% AUC** (일부 카테고리)를 기록하며 기존 GAN, Autoencoder 기반 방법론을 상회했습니다. 특히 텍스처와 객체 결함 모두에서 균형 잡힌 성능을 보였습니다.
*   **한계:**
    1.  **Global vs Local:** 이미지 전체의 특징을 하나의 벡터로 압축하여 처리하므로, 아주 미세한 픽셀 단위의 결함 위치를 정확히 찾아내는 데에는 패치(Patch) 기반 방법론보다 정밀도가 떨어질 수 있습니다.
    2.  **Gradient Map의 모호성:** 역전파를 통한 시각화는 결함의 대략적인 위치는 알려주지만, 정확한 세그멘테이션(Segmentation) 마스크를 제공하지는 않습니다.

***

### **3. 모델의 일반화 성능 향상 가능성 (Generalization)**

이 논문에서 가장 강조하는 일반화 전략은 **"Multi-transform Strategy"**입니다.

1.  **변환을 통한 분포 확장:** 학습 데이터가 적을 때(Few-shot), 단순히 원본 이미지만 학습하면 과적합(Overfitting)되기 쉽습니다. DifferNet은 이미지를 회전, 스케일링하여 입력함으로써 모델이 정상 데이터의 다양한 변동성을 학습하게 합니다. 이는 모델이 보지 못한 정상 변동(예: 약간의 각도 틀어짐)을 결함으로 오판하는 것을 방지하여 일반화 성능을 높입니다.
2.  **특징 공간의 강건함:** 픽셀 공간이 아닌 사전 학습된 CNN의 특징 공간(Feature Space)에서 밀도를 추정하므로, 조명 변화나 미세한 노이즈 같은 불필요한 정보에 덜 민감하며, 이는 새로운 환경에서의 일반화 능력을 높여줍니다.
3.  **잠재력:** 이 구조는 특징 추출기(Backbone)를 더 강력한 모델(예: WideResNet, Vision Transformer)로 교체하는 것만으로도 성능이 비약적으로 향상될 수 있는 유연한 구조를 가지고 있습니다.

***

### **4. 향후 연구 영향 및 2020년 이후 최신 연구 비교**

**영향 (Impact):** DifferNet은 "Feature Extraction + Density Estimation"이라는 패러다임을 정립했습니다. 이후 연구들은 복잡한 생성 모델(GAN)을 학습하는 대신, **잘 학습된 특징을 어떻게 잘 분포시킬 것인가**에 집중하게 되었습니다.

**2020년 이후 최신 연구 비교 분석:**
DifferNet(2020) 이후, 결함 탐지 성능은 비약적으로 발전하여 현재는 MVTec AD 기준 **99% 이상의 AUC**를 달성하고 있습니다.

| 비교 항목 | **DifferNet (2020)** | **FastFlow / CS-Flow (2021-2022)** | **PatchCore (2022)** | **SimpleNet (2023)** |
| :--- | :--- | :--- | :--- | :--- |
| **핵심 접근** | Global Features + NF (1D) | **2D** Features + NF | Memory Bank (Patch Retrieval) | Feature Adaptation + Gaussian Noise |
| **특징 처리** | 이미지를 하나의 벡터로 압축 | 위치 정보($H \times W$) 유지한 채 흐름 학습 | 이미지 패치들을 그대로 DB에 저장 | CNN 특징에 노이즈를 섞어 분류기 학습 |
| **장점** | 구조가 단순하고 빠름 | 위치 정보 보존으로 **세그멘테이션 성능 우수** | **학습 불필요**, SOTA 성능 (99%+) | 구조가 매우 간단하며 추론 속도 빠름 |
| **단점** | 미세 결함 위치 특정 어려움 | 모델이 상대적으로 무거움 | 메모리 사용량이 큼 | 학습 데이터 설정에 민감할 수 있음 |
| **성능 (AUC)** | ~93-95% | ~99.4% | ~99.6% | ~99.6%+ |

*   **FastFlow:** DifferNet이 특징을 1차원 벡터로 뭉갠 것과 달리, 2D 공간 구조를 유지한 채로 Normalizing Flow를 적용하여 위치 정확도를 획기적으로 개선했습니다.
*   **PatchCore:** 학습을 아예 하지 않고 정상 이미지의 패치들을 메모리 뱅크에 저장한 뒤, 테스트 이미지 패치와의 거리를 측정하는 방식으로 최고 성능을 달성했습니다. 이는 DifferNet의 "밀도 추정" 방식에서 "거리 기반(Nearest Neighbor)" 방식으로의 전환을 보여줍니다.

**앞으로 연구 시 고려할 점:**
1.  **Feature Adaptation:** 단순히 사전 학습된 특징을 쓰는 것을 넘어, 도메인(산업 현장)에 맞게 특징을 어떻게 적응(Adaptation)시킬 것인가? (SimpleNet 등의 접근)
2.  **Local vs Global:** 전체적인 맥락(Context)과 국소적인 결함(Local Defect)을 동시에 잡기 위해 Transformer 구조를 활용하는 연구가 필요합니다.
3.  **Foundation Models:** 최근에는 Segment Anything Model (SAM)이나 DINOv2 같은 거대 모델을 활용하여 별도 학습 없이 결함을 찾는 Zero-shot/Few-shot 연구로 흐름이 이동하고 있습니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/5dcd6945-ba63-453e-be35-e16e0f8319d8/2008.12577v1.pdf)
[2](https://arxiv.org/abs/2401.08686)
[3](https://ieeexplore.ieee.org/document/9423203/)
[4](https://ieeexplore.ieee.org/document/10651547/)
[5](https://link.springer.com/10.1007/s10586-025-05343-8)
[6](https://ieeexplore.ieee.org/document/10208830/)
[7](https://link.springer.com/10.1007/s00521-024-10172-8)
[8](https://arxiv.org/abs/2507.07579)
[9](https://ieeexplore.ieee.org/document/11229407/)
[10](https://www.semanticscholar.org/paper/11709bfadfd6bbb371f4077bccb7c26d93c39cdd)
[11](https://arxiv.org/abs/2308.15366)
[12](http://arxiv.org/pdf/2210.14485.pdf)
[13](https://arxiv.org/abs/2503.21622)
[14](https://pmc.ncbi.nlm.nih.gov/articles/PMC10154284/)
[15](https://arxiv.org/pdf/2206.08826.pdf)
[16](https://www.degruyter.com/document/doi/10.1515/biol-2022-0859/html)
[17](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2024.1332188/pdf)
[18](https://www.mdpi.com/2306-5354/11/12/1191)
[19](https://www.frontiersin.org/articles/10.3389/fnagi.2024.1434589/full)
[20](https://www.tnt.uni-hannover.de/papers/data/1464/Same_Same_But_DifferNet_final.pdf)
[21](https://www.mathworks.com/help/vision/ug/getting-started-with-anomaly-detection-using-deep-learning.html)
[22](https://openreview.net/forum?id=YwHOH3ROel)
[23](https://openaccess.thecvf.com/content/WACV2021/papers/Rudolph_Same_Same_but_DifferNet_Semi-Supervised_Defect_Detection_With_Normalizing_Flows_WACV_2021_paper.pdf)
[24](https://papers.cool/arxiv/2503.21622)
[25](https://pmc.ncbi.nlm.nih.gov/articles/PMC12349016/)
[26](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/datasets/mvtec_ad.pdf)
[27](https://dippingtodeepening.tistory.com/112)
[28](https://chatpaper.com/paper/162889)
[29](https://openaccess.thecvf.com/content/CVPR2023W/VAND/papers/Heckler_Exploring_the_Importance_of_Pretrained_Feature_Extractors_for_Unsupervised_Anomaly_CVPRW_2023_paper.pdf)
[30](https://ar5iv.labs.arxiv.org/abs/2202.12759)
[31](https://ar5iv.labs.arxiv.org/html/2111.07677)
[32](https://arxiv.org/html/2506.16890v1)
[33](https://arxiv.org/pdf/2307.06534.pdf)
[34](https://arxiv.org/html/2410.11591v1)
[35](https://www.semanticscholar.org/paper/Same-Same-But-DifferNet:-Semi-Supervised-Defect-Rudolph-Wandt/b1464ca857593c049873421db2f37bf2d0ff676d)
[36](https://arxiv.org/pdf/2205.14852.pdf)
[37](https://arxiv.org/html/2503.23451v1)
[38](https://openaccess.thecvf.com/content/WACV2021/html/Rudolph_Same_Same_but_DifferNet_Semi-Supervised_Defect_Detection_With_Normalizing_Flows_WACV_2021_paper.html)
[39](https://pmc.ncbi.nlm.nih.gov/articles/PMC11121878/)
[40](https://arxiv.org/abs/2008.12577)
[41](https://dataroots.io/blog/anomaly-detection-in-images-using-patchcore)
