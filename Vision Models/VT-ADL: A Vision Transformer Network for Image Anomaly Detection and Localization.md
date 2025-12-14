# VT-ADL: A Vision Transformer Network for Image Anomaly Detection and Localization

## 1. 핵심 주장 및 주요 기여 요약

**VT-ADL (Vision Transformer for Anomaly Detection and Localization)**은 이미지 이상 탐지 분야에 Vision Transformer(ViT)를 도입하여 기존 CNN 기반 방식의 한계를 극복하고자 한 연구입니다.

*   **핵심 주장:** 기존의 재구성(Reconstruction) 기반 방식은 이상 부위의 위치를 정확히 파악하는 데 한계가 있습니다. 이를 해결하기 위해 **Vision Transformer(ViT)를 사용하여 이미지 패치(Patch)의 공간 정보를 보존**하고, 잠재 공간(Latent Space)에서 **Gaussian Mixture Density Network(GMDN)**를 통해 정상 데이터의 분포를 모델링하면 이상 탐지 및 위치 추정 성능을 동시에 높일 수 있습니다.
*   **주요 기여:**
    1.  **ViT와 GMDN의 결합:** 위치 정보를 보존하는 ViT의 특성을 활용하여 이상 부위 위치 추정(Localization) 성능을 극대화한 하이브리드 모델 제안.
    2.  **새로운 산업용 데이터셋 공개:** 실제 제조 현장의 데이터를 담은 **BTAD (beanTech Anomaly Detection)** 데이터셋을 공개하여 관련 연구 활성화에 기여.
    3.  **성능 입증:** MNIST 및 MVTec AD 벤치마크에서 당시 SOTA(State-of-the-Art) 모델들과 대등하거나 우수한 성능을 달성.

***

## 2. 상세 분석: 문제 정의, 제안 방법, 모델 구조, 한계

### 2.1 해결하고자 하는 문제 (Problem Definition)
산업 현장의 이상 탐지(Anomaly Detection)는 대부분 정상 데이터만으로 학습하여 비정상(결함)을 찾아내야 하는 **비지도(Unsupervised) 또는 준지도(Semi-supervised)** 학습 문제입니다.
기존 CNN 기반 Autoencoder 방식은 이미지를 압축했다가 복원하는 과정에서 **"미세한 결함까지 정상처럼 복원해버리거나(Generalization too well)", "공간적 위치 정보를 소실"**하여 결함의 정확한 위치를 파악하기 어려운 문제가 있었습니다.

### 2.2 제안하는 방법 (Proposed Method)
이 논문은 **재구성 오차(Reconstruction Error)**와 **잠재 특징의 확률 밀도 추정(Density Estimation)**을 결합한 방식을 제안합니다.

**1) 입력 처리 (Input Embedding)**
이미지를 $P \times P$ 크기의 패치로 분할하여 시퀀스로 만듭니다. 이때 각 패치의 위치 정보를 유지하기 위해 **Positional Embedding**을 더합니다.

$$ Z_0 = [X_1 E; X_2 E; ...; X_N E] + E_{pos} $$

($N$: 패치 수, $E$: 선형 투영 매트릭스, $E_{pos}$: 위치 임베딩)

**2) 특징 추출 (Encoder)**
Transformer Encoder를 사용하여 이미지의 글로벌한 문맥(Context)과 로컬 정보를 동시에 학습합니다.

$$ Z'_l = \text{MSA}(\text{LN}(Z_{l-1})) + Z_{l-1} $$

$$ Z_l = \text{MLP}(\text{LN}(Z'_l)) + Z'_l $$

**3) 밀도 추정 (Gaussian Mixture Density Network)**
Transformer가 추출한 잠재 특징 벡터($x$)가 정상 데이터 분포에 속하는지 판단하기 위해 **GMM(Gaussian Mixture Model)**을 신경망으로 모델링합니다. 특정 패치가 정상 분포에서 벗어나면 이상으로 간주합니다.
조건부 확률 밀도 $\hat{p}(y|x)$는 다음과 같이 정의됩니다.

$$ \hat{p}(y|x) = \sum_{k=1}^{K} w_k(x; \theta) \mathcal{N}(y | \mu_k(x; \theta), \sigma_k^2(x; \theta)) $$

여기서 $w_k, \mu_k, \sigma_k^2$는 신경망이 예측하는 GMM의 매개변수(가중치, 평균, 분산)입니다.

**4) 손실 함수 (Loss Function)**
모델은 재구성 성능과 밀도 추정 성능을 동시에 최적화합니다.

$$ L(X) = -LL + \lambda_1 \text{MSE}(X, \hat{X}) + \lambda_2 \text{SSIM}(X, \hat{X}) $$

*   **$-LL$ (Negative Log-Likelihood):** 잠재 특징이 GMM 분포를 잘 따르도록 학습 (정상 데이터의 특징 분포 학습).
*   **MSE & SSIM:** 이미지를 원본과 유사하게 복원하도록 학습 (픽셀 단위 및 구조적 유사도).

### 2.3 모델 구조 (Model Architecture)
*   **Transformer Encoder:** 이미지를 패치 단위로 처리하여 위치 정보가 포함된 특징 벡터를 생성합니다.
*   **Decoder:** 잠재 특징을 다시 이미지로 복원하는 CNN 기반 디코더입니다. (Autoencoder 역할)
*   **GMDN Branch:** Encoder의 출력(잠재 특징)을 입력으로 받아 해당 특징이 정상 분포에 속할 확률(Likelihood)을 계산합니다. 이 값이 낮을수록 이상(Anomaly)일 확률이 높습니다.

### 2.4 성능 향상 및 한계
*   **성능 향상:** Transformer의 Self-Attention 메커니즘 덕분에 CNN보다 **글로벌한 상관관계**를 잘 파악하며, 패치 단위 처리를 통해 **결함 위치 추정(Localization)** 성능이 크게 향상되었습니다. 특히 BTAD와 같은 복잡한 텍스처 데이터셋에서 강점을 보였습니다.
*   **한계:**
    *   **계산 복잡도:** ViT 특성상 CNN보다 연산량이 많고 학습 데이터가 적을 때 과적합(Overfitting) 위험이 있습니다.
    *   **GMM 튜닝:** GMM의 구성 요소(Gaussian 개수 등)를 데이터셋에 맞춰 세밀하게 튜닝해야 최적의 성능이 나옵니다.
    *   **메모리 사용량:** 모든 패치에 대해 Attention을 계산하므로 고해상도 이미지 처리 시 메모리 부담이 큽니다.

***

## 3. 모델의 일반화 성능 향상 가능성 (Generalization)

이 논문은 모델이 학습 데이터(정상 이미지)에 과적합되지 않고, **보지 못한 이상(Unseen Anomalies)**을 잘 탐지할 수 있도록 하는 '일반화' 전략에 대해 중요한 통찰을 제공합니다.

1.  **노이즈 주입 (Noise Injection)을 통한 정규화:**
    논문에서는 Transformer가 추출한 특징 벡터를 GMDN에 입력하기 전, **가우시안 노이즈 $\mathcal{N}(0, 0.2)$를 추가**하는 기법을 사용했습니다.
    *   **효과:** 이는 데이터 증강(Data Augmentation)과 유사한 효과를 내어, 모델이 정상 데이터의 '완벽한' 패턴만 외우는 것을 방지하고 분포의 경계를 부드럽게(Smoothing) 만듭니다. 결과적으로 약간의 변형이 있는 정상 데이터는 정상으로, 확실한 이상은 이상으로 구분하는 **일반화 성능(Generalization Performance)이 향상**되었습니다. (실험 결과 노이즈 추가 시 PRO 점수가 0.807에서 0.897로 대폭 상승)

2.  **Transformer의 강건함:**
    ViT는 CNN의 Inductive Bias(지역적 특성 우선)가 적어 데이터가 충분할 경우 더 유연한 특징을 학습합니다. 이는 다양한 형태의 정상 패턴을 아우르는 **Global Context**를 학습하는 데 유리하여, 국소적인 노이즈나 변화에 민감하게 반응하지 않고 전체적인 맥락에서 이상을 판단하는 능력을 부여합니다.

***

## 4. 향후 연구 영향 및 고려사항

### 4.1 향후 연구에 미치는 영향
*   **ViT의 이상 탐지 적용 가속화:** 이 연구는 이상 탐지 분야에 Transformer를 성공적으로 적용한 초기 사례 중 하나로, 이후 **AnoViT**, **InTra** 등 다양한 ViT 기반 이상 탐지 모델이 등장하는 기폭제가 되었습니다.
*   **벤치마크 데이터셋 확장:** 함께 공개된 **BTAD 데이터셋**은 MVTec AD와 함께 산업용 이상 탐지 모델을 평가하는 표준 벤치마크 중 하나로 자리 잡았습니다. 최신 논문들(예: WinCLIP, EfficientAD 등)도 성능 평가 시 BTAD를 필수적으로 포함합니다.

### 4.2 연구 시 고려할 점
*   **Inference 속도 최적화:** 현장 적용을 위해서는 ViT의 느린 추론 속도를 개선해야 합니다. (예: 경량화된 Transformer, CNN-ViT 하이브리드 구조 고려)
*   **Few-shot/Zero-shot 확장:** VT-ADL은 정상 데이터 학습이 필요합니다. 최근 트렌드는 학습 데이터가 거의 없거나 아예 없는 상황(Zero-shot)을 가정하므로, **CLIP과 같은 사전 학습된 거대 모델(Foundation Model)과의 결합**을 고려해야 합니다.

***

## 5. 2020년 이후 관련 최신 연구 비교 분석

VT-ADL(2021) 이후 이상 탐지 기술은 **특징 임베딩(Embedding)**과 **메모리 뱅크(Memory Bank)**, 그리고 **대규모 언어-비전 모델(VLM)**을 활용하는 방향으로 진화했습니다.

| 구분 | VT-ADL (2021) | PatchCore (2022) | SimpleNet (2023) | WinCLIP / AnomalyGPT (2023~2024) |
| :--- | :--- | :--- | :--- | :--- |
| **핵심 접근** | **Reconstruction + Density**<br>(ViT로 복원 및 밀도 추정) | **Embedding + Memory Bank**<br>(정상 패치 특징 저장 후 비교) | **Feature Adaptation**<br>(특징에 노이즈 추가 후 구분) | **Zero/Few-shot**<br>(VLM의 일반 상식 활용) |
| **Backbone** | Vision Transformer (Trainable) | Pre-trained CNN (ResNet 등) | Pre-trained CNN (MobileNet 등) | CLIP, LLM (Frozen) |
| **학습 필요성** | 정상 데이터 학습 필수 | 정상 특징만 저장 (학습 거의 없음) | 간단한 Feature Adapter 학습 | 학습 불필요 (Zero-shot) 또는 소량 학습 |
| **성능 (MVTec)** | 준수함 (90% 중반) | **매우 우수 (99% 이상)** | **SOTA 급 (99.7% 등)** | 상황에 따라 다르나 범용성 최강 |
| **장점** | 위치 추정 정확, 공간 정보 보존 | 학습 빠름, 성능 최고 수준 | 구조 단순, 추론 속도 매우 빠름 | 별도 학습 없이 바로 사용 가능, 대화형 가능 |
| **단점** | 느린 학습/추론, 데이터 많이 필요 | 메모리 사용량 큼 (Coreset 필요) | 대규모 데이터셋 의존적일 수 있음 | 미세한 산업 특화 결함에는 약할 수 있음 |

### 최신 트렌드와의 비교 요약
*   **PatchCore (CVPR 2022):** VT-ADL처럼 복잡하게 네트워크를 학습시키는 대신, 사전 학습된 네트워크(ImageNet)의 중간 특징들을 **메모리 뱅크(Memory Bank)**에 저장하고 최근접 이웃(Nearest Neighbor) 방식으로 이상을 탐지합니다. 이 방식이 VT-ADL보다 **학습 속도가 훨씬 빠르고 성능(AUC 99%+)도 더 뛰어납니다.**
*   **SimpleNet (CVPR 2023):** 복잡한 구조 없이 사전 학습된 특징에 **노이즈를 섞은 뒤 이를 구분하는 간단한 Discriminator**만 학습시켜 SOTA 성능을 달성했습니다. VT-ADL의 GMDN보다 훨씬 단순한 구조로 더 높은 성능을 냅니다.
*   **WinCLIP / AnomalyGPT:** VT-ADL은 특정 제품(Bottle, Cable 등)의 정상 데이터를 학습해야만 작동하지만, 최신 **WinCLIP**이나 **AnomalyGPT**는 학습 없이 텍스트 프롬프트("손상된 병", "긁힌 케이블")만으로 이상을 탐지하는 **Zero-shot/Few-shot** 단계로 넘어갔습니다. 이는 모델의 일반화(Generalization) 개념을 '데이터 분포 학습'에서 '사전 지식 활용'으로 확장시킨 것입니다.

결론적으로 **VT-ADL**은 ViT를 이상 탐지에 도입한 선구적인 연구이지만, 최신 연구들은 **재구성(Reconstruction)보다는 특징 비교(Feature Matching)**나 **초거대 모델(Foundation Model) 활용** 쪽으로 패러다임이 이동하고 있습니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/7f070baf-6f69-427e-94a3-8451935aa98e/2104.10036v1.pdf)
[2](https://ieeexplore.ieee.org/document/9576231/)
[3](https://link.springer.com/10.1007/s44196-023-00328-0)
[4](https://link.springer.com/10.1007/s40032-025-01169-w)
[5](https://www.mdpi.com/2076-3417/15/15/8330)
[6](https://www.journaljenrr.com/index.php/JENRR/article/view/434)
[7](http://link.springer.com/10.1007/s10479-019-03508-4)
[8](https://www.ahajournals.org/doi/10.1161/res.133.suppl_1.P3139)
[9](https://aapc.khu.ac.ir/article-1-1165-en.html)
[10](https://dl.acm.org/doi/10.1145/3383583.3398531)
[11](https://link.springer.com/10.1007/s11192-023-04886-0)
[12](https://arxiv.org/abs/2104.10036)
[13](http://arxiv.org/pdf/2412.00890.pdf)
[14](https://www.mdpi.com/1424-8220/21/24/8501/pdf)
[15](http://arxiv.org/pdf/2411.09558.pdf)
[16](http://arxiv.org/pdf/2306.10239.pdf)
[17](https://arxiv.org/pdf/2303.07557.pdf)
[18](http://arxiv.org/pdf/2106.05410v2.pdf)
[19](https://www.mdpi.com/1424-8220/22/5/1951/pdf)
[20](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08405.pdf)
[21](https://arxiv.org/html/2507.15905v1)
[22](https://pmc.ncbi.nlm.nih.gov/articles/PMC10934034/)
[23](https://openaccess.thecvf.com/content/CVPR2023/papers/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.pdf)
[24](https://github.com/CASIA-IVA-Lab/AnomalyGPT)
[25](https://www.koreascience.kr/article/CFKO202419334901956.page)
[26](https://thescipub.com/pdf/jcssp.2025.1613.1620.pdf)
[27](https://pmc.ncbi.nlm.nih.gov/articles/PMC12565623/)
[28](https://www.semanticscholar.org/paper/AnomalyGPT:-Detecting-Industrial-Anomalies-using-Gu-Zhu/f2ec0182c6646d3128afa5100f37d9de7b533463)
[29](https://arxiv.org/html/2510.00495v1)
[30](https://openaccess.thecvf.com/content/WACV2022/papers/Tsai_Multi-Scale_Patch-Based_Representation_Learning_for_Image_Anomaly_Detection_and_Segmentation_WACV_2022_paper.pdf)
[31](https://arxiv.org/html/2507.13378v1)
[32](https://arxiv.org/pdf/2104.10036.pdf)
[33](https://arxiv.org/html/2506.06836v2)
[34](https://openaccess.thecvf.com/content/WACV2024/papers/Hyun_ReConPatch_Contrastive_Patch_Representation_Learning_for_Industrial_Anomaly_Detection_WACV_2024_paper.pdf)
[35](https://ar5iv.labs.arxiv.org/html/2303.14814)
[36](https://arxiv.org/html/2508.10681v1)
[37](https://ar5iv.labs.arxiv.org/html/2104.10036)
[38](https://dataroots.io/blog/anomaly-detection-in-images-using-patchcore)
[39](https://www.sciencedirect.com/science/article/abs/pii/S0143816624004354)
[40](https://arxiv.org/html/2507.19949v1)
[41](https://github.com/IHPCRits/IAD-Survey)
