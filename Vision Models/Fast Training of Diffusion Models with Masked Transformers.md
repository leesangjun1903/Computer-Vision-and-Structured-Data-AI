# Fast Training of Diffusion Models with Masked Transformers

### 1. 핵심 주장 및 주요 기여 요약

**MaskDiT**는 기존 트랜스포머 기반 확산 모델(Diffusion Transformer, DiT)의 막대한 학습 비용 문제를 해결하기 위해 **마스킹(Masking)** 기법을 도입한 새로운 학습 프레임워크입니다.

*   **핵심 주장:** 이미지 픽셀 간에는 중복성이 높으므로, 학습 시 전체 패치의 약 50%를 마스킹(제거)하고 나머지 패치만으로 확산 과정을 학습해도 충분하며, 이를 통해 학습 속도를 획기적으로 높일 수 있다.
*   **주요 기여:**
    1.  **비대칭 인코더-디코더 구조**를 제안하여, 인코더는 마스킹되지 않은 패치만 처리해 연산량을 절반으로 줄이고, 가벼운 디코더가 전체 이미지를 복원하도록 설계했습니다.
    2.  **보조 재구성 손실(Auxiliary Reconstruction Loss)**을 도입하여 모델이 마스킹된 영역의 전역적 문맥(Global Context)을 학습하도록 유도, 생성 성능을 향상시켰습니다.
    3.  기존 DiT 대비 **학습 시간을 약 30% 수준으로 단축**하면서도 동등하거나 더 우수한 이미지 생성 성능(FID)을 달성했습니다.

***

### 2. 논문 상세 분석

#### 2.1 해결하고자 하는 문제 (Problem Statement)
최근 확산 모델의 백본(Backbone)이 U-Net에서 ViT(Vision Transformer) 기반의 DiT로 전환되면서 성능은 향상되었으나, **학습 비용(계산 자원 및 시간)이 기하급수적으로 증가**했습니다. 기존 DiT는 모든 이미지 패치를 연산에 포함하므로 고해상도 이미지 학습 시 비효율적입니다. 이 논문은 **"생성 성능을 저하시키지 않으면서 학습 효율성을 극대화할 수 있는가?"**라는 질문에 답하고자 합니다.

#### 2.2 제안하는 방법 및 수식 (Proposed Method)
MaskDiT는 MAE(Masked Autoencoder)의 개념을 확산 모델에 접목하되, 생성 모델의 특성에 맞게 수정했습니다.

**1. 마스킹 전략 (Masking Strategy):**
학습 시 입력 이미지 $x$를 패치 단위로 나누고, 약 50%의 패치를 무작위로 제거합니다.
$$x_{masked} = \text{Mask}(x, r=0.5)$$

**2. 손실 함수 (Loss Function):**
단순히 노이즈를 예측하는 기존 확산 손실($L_{DSM}$)에 마스킹된 패치를 복원하는 재구성 손실($L_{recon}$)을 결합하여 최적화합니다.

$$L = L_{DSM} + \lambda L_{recon}$$

*   **$$L_{DSM}$$ (Denoising Score Matching Loss):**
    마스킹되지 않은(Visible) 패치들에 대해서만 노이즈(혹은 점수, Score)를 예측합니다. 이는 연산량을 직접적으로 줄여줍니다.

$$L_{DSM} = \mathbb{E}_{t, x_0, \epsilon} [ \| \epsilon - \epsilon_\theta(x_t^{vis}, t) \|^2 ]$$

*   **$$L_{recon}$$ (Masked Reconstruction Loss):**
    디코더가 출력한 마스킹된 위치의 픽셀 값과 원본 픽셀 값 간의 차이(MSE)를 최소화합니다. 이는 모델이 보이지 않는 영역을 추론하며 전체적인 이미지 구조를 이해하도록 돕습니다.

$$L_{recon} = \mathbb{E} [ \| x_{masked} - \text{Decoder}(z_{masked}) \|^2 ]$$

#### 2.3 모델 구조 (Model Architecture)
**비대칭 인코더-디코더 (Asymmetric Encoder-Decoder)** 구조가 핵심입니다.
*   **인코더 (Encoder):** 표준 DiT 블록을 사용하지만, **마스킹되지 않은 패치(Visible patches)**만 입력으로 받습니다. 패치 수가 절반으로 줄어들어 연산량(FLOPs)이 크게 감소합니다.
*   **디코더 (Decoder):** 인코더 출력에 **학습 가능한 마스크 토큰(Mask tokens)**을 추가하여 원래 이미지 크기로 복원한 뒤, 가벼운(Lightweight) DiT 블록을 통과시킵니다. 여기서 전역적인 픽셀 복원과 노이즈 예측이 동시에 수행됩니다.

#### 2.4 성능 향상 및 한계 (Performance & Limitations)
*   **성능 향상:** ImageNet 256x256 및 512x512 벤치마크에서 DiT 대비 학습 속도가 **약 3배 빨라졌으며**, FID(Frechet Inception Distance) 점수는 동등하거나 소폭 우수했습니다.
*   **한계점:**
    1.  **학습-추론 간 괴리 (Gap):** 학습 때는 마스킹을 하지만 추론(이미지 생성) 때는 전체 패치를 사용해야 하므로 분포 차이가 발생합니다. 이를 해결하기 위해 학습 마지막 단계에 마스킹 없이 짧게 추가 학습하는 **"Unmasking Tuning"** 과정이 필수적입니다.
    2.  **Unconditional 생성 약점:** 클래스 조건(Class-conditional)이 없는 순수 생성에서는 마스킹 학습만으로 좋은 성능을 내기 어려워 튜닝 의존도가 높습니다.

---

### 3. 모델의 일반화 성능 향상 가능성 (Generalization Capability)

이 논문에서 가장 흥미로운 점은 마스킹이 단순한 속도 향상 도구가 아니라, **모델의 일반화 능력을 강화하는 학습 기제**로 작용한다는 것입니다.

1.  **전역적 문맥 학습 (Global Context Learning):**
    기존 확산 모델은 픽셀 간의 지역적 상관관계(Local correlation)에 과적합되기 쉽습니다. 그러나 MaskDiT는 이미지의 절반이 가려진 상태에서 나머지 반을 예측해야 하므로, 모델이 단순히 주변 픽셀을 베끼는 것이 아니라 **이미지 전체의 의미론적 구조(Semantic Structure)와 사물 간의 관계**를 파악해야만 합니다. 이는 모델이 더 강력한 **표현 학습(Representation Learning)**을 수행하게 만듭니다.

2.  **재구성 보조 작업의 효과:**
    수식 $$L_{recon}$$을 통한 재구성 작업은 확산 모델이 노이즈 제거(Denoising)라는 생성 작업 외에도, 이미지의 **내재적 구조(Structure)**를 이해하도록 강제합니다. 실험 결과, 이 보조 작업이 없는 경우($\lambda=0$)보다 있는 경우에 FID가 훨씬 빠르게 개선되었으며, 이는 모델이 보지 못한 데이터에 대해서도 더 강건하게(Robust) 작동할 수 있음을 시사합니다.

***

### 4. 향후 연구 영향 및 고려사항

#### 향후 연구에 미치는 영향
*   **대규모 모델의 민주화:** 수백 개의 GPU가 필요한 DiT 학습 비용을 1/3로 줄임으로써, 더 적은 자원으로도 고성능 생성 모델(Sora 등 비디오 모델 포함)을 연구할 수 있는 길을 열었습니다.
*   **멀티모달 학습으로의 확장:** 마스킹 기법은 텍스트(BERT), 이미지(MAE)에 이어 생성 모델에서도 유효함이 입증되었습니다. 이는 텍스트-이미지, 비디오-오디오 등 멀티모달 생성 모델의 통합 학습 아키텍처로 발전할 가능성이 큽니다.

#### 연구 시 고려할 점
*   **Unmasking Tuning의 자동화:** 현재는 수동으로 튜닝 단계를 두어야 합니다. 학습 과정에서 마스킹 비율을 점진적으로 줄이는 **Curriculum Learning** 방식을 도입하여 별도의 튜닝 없이 학습을 완료하는 연구가 필요합니다.
*   **생성 품질 vs 다양성:** 마스킹 학습이 구조적 일관성은 높여주지만, 창의적이고 다양한 디테일 생성(High-frequency details)에 어떤 영향을 미치는지에 대한 심층 분석이 필요합니다.

***

### 5. 2020년 이후 관련 최신 연구 비교 분석

| 연도 | 모델 / 연구명 | 핵심 아키텍처 | 주요 특징 및 MaskDiT와의 차이 |
| :--- | :--- | :--- | :--- |
| **2020** | **DDPM** (Ho et al.) | **U-Net** (CNN) | 확산 모델의 르네상스를 연 연구. CNN 기반으로, 전역적 문맥 파악에 한계가 있으며 연산량이 이미지 해상도에 비례하여 큼. |
| **2021** | **ADM** (Dhariwal & Nichol) | Optimized U-Net | 모델 구조 최적화로 GAN을 능가하는 성능 달성. 여전히 픽셀 공간 전체 연산 수행. |
| **2022** | **DiT** (Peebles & Xie) | **Transformer** | U-Net을 ViT로 대체. 확장성(Scalability)은 좋으나, 모든 패치를 처리하므로 학습 비용이 매우 높음 ($O(N^2)$ attention). |
| **2023** | **MDT** (Gao et al.) | Masked DiT | MaskDiT와 유사하게 마스킹을 사용하나, 마스킹된 잠재 표현(Latent)의 문맥 학습에 더 집중. (MaskDiT와 동시대 경쟁 연구) |
| **2023** | **MaskDiT** (본 논문) | **Asymmetric DiT** | **효율성(Efficiency)**에 초점. 인코더 입력을 50% 줄여 속도 3배 향상 및 전역 문맥 학습 강화. |
| **2024~** | **FasterDiT / SiT** | Improved DiT | MaskDiT의 아이디어를 발전시켜, 학습 단계별로 마스킹 비율을 조절하거나(Curriculum), 보간(Interpolant) 기반 학습으로 수렴 속도를 더욱 가속화함. |
| **2025~** | **MaskGWM** (Video) | Video DiT | 마스킹 아이디어를 **비디오(3D)**로 확장. 시간축(Temporal) 마스킹을 통해 비디오 생성의 막대한 연산량을 제어하고 일관성을 유지함. |

**요약하자면**, MaskDiT는 2022년 DiT가 증명한 "확산 모델의 트랜스포머화" 흐름 위에서, 2023년 "마스킹을 통한 효율화 및 문맥 학습 강화"라는 새로운 패러다임을 제시한 중요한 분기점이 되는 연구입니다.

[1](https://arxiv.org/abs/2306.09305)
[2](https://arxiv.org/pdf/2306.09305v1.pdf)
[3](https://arxiv.org/pdf/2306.11363.pdf)
[4](https://arxiv.org/pdf/2403.17004.pdf)
[5](http://arxiv.org/pdf/2407.01425.pdf)
[6](http://arxiv.org/pdf/2502.11663.pdf)
[7](https://arxiv.org/pdf/2304.07313.pdf)
[8](http://arxiv.org/pdf/2404.10445.pdf)
[9](https://aclanthology.org/2023.acl-long.248.pdf)
[10](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/11278.pdf)
[11](https://arxiv.org/html/2306.09305v1)
[12](https://www.reddit.com/r/ArtificialInteligence/comments/1jl0t50/mask%C2%B2dit_dualmasked_diffusion_transformer_for/)
[13](https://openreview.net/pdf?id=L4uaAR4ArM)
[14](https://openaccess.thecvf.com/content/ICCV2023/papers/Gao_Masked_Diffusion_Transformer_is_a_Strong_Image_Synthesizer_ICCV_2023_paper.pdf)
[15](https://huggingface.co/hzzheng/MaskDiT/blob/340bb574a82fd442dbde83734bf07d615a8ff744/README.md)
[16](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/mdt/)
[17](https://www.emergentmind.com/topics/masked-diffusion-models)
[18](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/maskdit/)
[19](https://fugumt.com/fugumt/paper_check/2312.07231v1_enmode)
[20](https://arxiv.org/html/2509.21565v1)
[21](https://arxiv.org/pdf/2406.04329.pdf)
[22](https://arxiv.org/html/2306.09305v2)
[23](https://www.semanticscholar.org/paper/FasterDiT:-Towards-Faster-Diffusion-Transformers-Yao-Cheng/3e77725656549a19f0e8a1b75befc556f3a1af70)
[24](https://arxiv.org/html/2303.14389v2)
[25](https://arxiv.org/html/2507.16579v1)
[26](https://arxiv.org/html/2410.10356v2)
[27](https://arxiv.org/html/2504.10188v1)
[28](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/997fbea5-36fb-477b-b35d-2ffbccb83a26/2306.09305v2.pdf)
