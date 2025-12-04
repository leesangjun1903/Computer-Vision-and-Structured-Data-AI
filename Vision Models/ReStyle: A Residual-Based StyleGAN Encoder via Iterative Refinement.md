# ReStyle: A Residual-Based StyleGAN Encoder via Iterative Refinement

# **1. 핵심 주장 및 주요 기여 요약 (Abstract)**

**ReStyle**은 기존 StyleGAN Inversion(역변환) 기법의 한계인 '속도와 정확도 간의 트레이드오프'를 해결하기 위해 제안되었습니다. 이 논문의 핵심 주장은 **"단일 패스(One-shot) 추론 대신, 잔차(Residual)를 예측하여 반복적으로 잠재 코드(Latent Code)를 수정하는 것이 훨씬 효율적이고 정확하다"**는 것입니다.

**주요 기여 (Key Contributions):**
1.  **Iterative Refinement Scheme:** 인코더가 한 번에 정답을 맞히는 것이 아니라, 현재 추정치와 목표 이미지 간의 차이(Residual)를 단계적으로 줄여나가는 새로운 인버전 프레임워크 제안.
2.  **ReStyle Encoder:** 반복 수행을 전제로 설계되어, 파라미터 수가 적은 단순한 구조로도 기존 SoTA(pSp, e4e) 모델보다 우수한 재구성 품질(Reconstruction Quality) 달성.
3.  **Quality-Time Tradeoff 개선:** 최적화(Optimization) 기반 방식에 근접한 품질을 보이면서도 추론 속도는 수십 배 빠름.

***

## **2. 상세 분석: 문제 해결부터 한계까지**

### **2.1. 해결하고자 하는 문제 (Problem Statement)**
StyleGAN의 잠재 공간(Latent Space)으로 실제 이미지를 매핑하는 **GAN Inversion**에는 두 가지 주류가 있었습니다.
*   **Optimization-based:** 이미지마다 수천 번의 경사 하강법을 수행하여 $w$를 찾음. 품질은 높으나 시간이 매우 오래 걸림(수 분 소요).
*   **Encoder-based:** 신경망(Encoder)을 학습시켜 한 번의 전파(Forward pass)로 $w$를 예측. 속도는 빠르지만(수 ms), 디테일한 재구성 품질이 떨어짐.

$\rightarrow$ **ReStyle은 이 두 방식의 간극을 메우기 위해, 인코더를 여러 번 통과시키는 '절충안'을 통해 최적화 수준의 품질과 인코더 수준의 속도를 동시에 잡고자 했습니다.**

### **2.2. 제안하는 방법 (Proposed Method)**
ReStyle은 **Residual Learning(잔차 학습)**과 **Iterative Refinement(반복 정제)**를 결합했습니다.

**수식적 정의:**
단계 $t$에서의 입력 이미지 $x$에 대한 잠재 코드 추정치를 $w_t$, 이를 생성자가 복원한 이미지를 $y_t = G(w_t)$라고 합시다. ($w_0$는 평균 잠재 코드 $w_{avg}$로 초기화)

인코더 $E$는 원본 이미지 $x$와 현재 복원된 이미지 $y_t$를 채널 차원에서 결합(Concatenation)하여 입력받고, 잠재 코드의 **변화량(Residual)** $\Delta_t$를 예측합니다.

$$ \Delta_t = E(x \oplus y_t) $$

이후 잠재 코드는 다음과 같이 업데이트됩니다.

$$ w_{t+1} = w_t + \Delta_t $$

최종적으로 $N$번의 단계(Step)를 거쳐 최종 잠재 코드 $w_N$을 얻습니다. 학습 시에는 각 단계마다의 재구성 손실 함수(Loss Function)를 합산하여 인코더를 업데이트합니다.

### **2.3. 모델 구조 (Model Architecture)**
기존의 pSp(pixel2style2pixel)나 e4e(encoder4editing)는 FPN(Feature Pyramid Network) 기반의 복잡한 구조를 사용하여 여러 해상도에서 스타일 벡터를 추출했습니다.

반면, **ReStyle은 반복 구조 덕분에 인코더 자체를 경량화**했습니다.
*   **Simplified Encoder:** ResNet Backbone의 마지막 $16 \times 16$ 특징 맵(Feature Map)에서만 스타일 벡터를 추출합니다.
*   **Weight Sharing:** 모든 반복 단계($t=1 \dots N$)에서 동일한 인코더 가중치를 공유합니다. 이는 마치 RNN처럼 동작하며, 모델의 크기를 키우지 않고도 표현력을 높입니다.

### **2.4. 성능 향상 (Performance Improvement)**
*   **Coarse-to-Fine 수렴:** 초기 단계에서는 포즈나 얼굴형 같은 큰 특징(Coarse feature)을 수정하고, 후반 단계로 갈수록 눈매, 머리카락 결 등 미세한 특징(Fine detail)을 보정하는 경향을 보입니다.
*   **품질:** L2 Loss 및 LPIPS(지각 손실) 지표에서 기존 단일 패스 인코더들보다 우수한 수치를 기록했습니다.
*   **속도:** 5~10회 반복만으로 수렴하며, 이는 최적화 기반 방식보다 훨씬 빠릅니다 (약 0.5초 내외).

### **2.5. 한계점 (Limitations)**
*   **실시간성 부족:** 단일 패스 인코더보다는 $N$배의 연산이 필요하므로, 실시간(Real-time) 비디오 처리에 적용하기에는 여전히 부담이 될 수 있습니다.
*   **초기 추정 의존성:** 첫 번째 단계($t=1$)의 결과물은 기존 단일 인코더보다 품질이 낮을 수 있습니다. (반복을 통해 고치도록 학습되었기 때문)

***

## **3. 모델의 일반화 성능 (Generalization Capabilities)**

ReStyle의 가장 강력한 특징 중 하나는 특정 도메인이나 태스크에 국한되지 않는 **일반화 가능성**입니다.

1.  **다양한 도메인 적용:** 사람 얼굴(FFHQ)뿐만 아니라 자동차(Stanford Cars), 교회(LSUN Church), 야생 동물(AFHQ) 등 구조가 복잡하거나 비정형적인 데이터셋에서도 안정적인 인버전 성능을 보였습니다.
2.  **Encoder Bootstrapping (Toonification):**
    *   ReStyle은 "잔차를 예측한다"는 특성 덕분에, 다른 도메인으로의 변환에도 유리합니다.
    *   예를 들어, 실사 얼굴을 '디즈니 캐릭터(Toon)' 스타일로 바꿀 때, 실사 인코더로 얻은 $w$를 초기값($w_0$)으로 사용하고, ReStyle 인코더가 Toon 도메인에 맞는 $\Delta$를 예측하게 함으로써 훨씬 자연스러운 스타일 변환(Style Transfer)을 수행했습니다.
    *   이는 모델이 **"현재 상태에서 목표로 가기 위해 무엇을 고쳐야 하는가"**를 학습했기 때문에 가능한 일반화 능력입니다.

***

## **4. 향후 연구 영향 및 고려할 점 (Future Impact & Considerations)**

### **학계에 미친 영향 (Impact)**
이 논문은 GAN Inversion 분야에서 **"Iterative Refinement(반복 정제)"를 표준 방법론 중 하나로 정착**시켰습니다.
*   **패러다임 전환:** 단순히 인코더 구조를 깊게 쌓는 것보다, 추론 과정을 반복하는 것이 효율적이라는 점을 증명했습니다.
*   **Diffusion Model과의 연결:** 최근 생성 모델의 대세인 Diffusion Model 또한 노이즈를 반복적으로 제거(Refinement)하는 방식입니다. ReStyle은 GAN에서도 이러한 반복적 접근이 유효함을 보여준 선구적 연구로 평가받습니다.

### **연구 시 고려할 점 (Considerations for Future Research)**
1.  **Feedback Mechanism의 고도화:** 현재는 단순히 이미지($y_t$)를 채널에 붙여 넣는 방식입니다. Transformer의 Cross-attention 등을 활용해 인코더가 '어디를 고쳐야 할지' 더 명확하게 인지하도록 개선할 여지가 있습니다.
2.  **Editability 보존:** 반복해서 원본과 똑같이 만들수록($\Delta$를 계속 더할수록), Latent Code가 분포를 벗어나 편집(Editing)이 안 되는 문제가 발생할 수 있습니다. (Distortion-Editability Tradeoff). 이를 제어하는 정규화(Regularization) 기법 연구가 필요합니다.

***

## **5. 2020년 이후 관련 최신 연구 탐색 (Latest Research after 2020)**

ReStyle 이후, 이 아이디어를 발전시키거나 보완한 최신 연구들은 다음과 같습니다.

*   **HyperStyle (CVPR 2022) & RefineStyle (2024)**
    *   ReStyle이 잠재 코드($w$)만 수정했다면, 이 연구들은 **Generator의 가중치(Weights) 자체를 미세 조정(Hypernetwork 활용)**하여 더 완벽한 복원을 수행합니다.
*   **Style Transformer (CVPR 2022) / Stiles (2023)**
    *   ReStyle의 반복 구조 대신 Transformer를 사용하여 이미지의 특징을 잠재 공간으로 매핑하는 방식이 연구되었습니다.
*   **Gradual Residuals Alignment (2024)**
    *   ReStyle의 coarse-to-fine 접근법을 계승하여, 디테일 보존을 위한 Dual-stream 프레임워크를 제안했습니다.
*   **Diffusion-based GAN Inversion (2023-2024)**
    *   최근에는 GAN 인버전 자체를 Diffusion Process로 해석하거나, Diffusion 모델을 활용해 GAN의 잠재 공간을 탐색하는 하이브리드 연구가 활발합니다.

### **결론**
ReStyle은 **"반복적 잔차 학습"**이라는 직관적이고 강력한 아이디어로 GAN Inversion의 효율성을 획기적으로 높인 연구입니다. 최신 연구를 수행하실 때, **단순한 구조의 반복 추론이 복잡한 단일 모델보다 강력할 수 있음**을 염두에 두시면 좋은 성과가 있을 것입니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/03cb49fb-cf02-4716-b154-83e2747b19f6/2104.02699v2.pdf)
[2](https://arxiv.org/html/2410.06104v1)
[3](https://arxiv.org/pdf/2312.11422.pdf)
[4](https://arxiv.org/abs/2102.02766)
[5](https://arxiv.org/html/2402.14398v1)
[6](https://arxiv.org/pdf/2307.15033.pdf)
[7](http://arxiv.org/pdf/2004.14367.pdf)
[8](https://arxiv.org/html/2205.14377)
[9](https://www.arxiv.org/pdf/1912.04958.pdf)
[10](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Alaluf_ReStyle_A_Residual-Based_ICCV_2021_supplemental.pdf)
[11](https://www.scitepress.org/Papers/2024/123710/123710.pdf)
[12](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750457-supp.pdf)
[13](https://openaccess.thecvf.com/content/ICCV2021/papers/Alaluf_ReStyle_A_Residual-Based_StyleGAN_Encoder_via_Iterative_Refinement_ICCV_2021_paper.pdf)
[14](https://openaccess.thecvf.com/content/WACV2024/papers/Katsumata_Revisiting_Latent_Space_of_GAN_Inversion_for_Robust_Real_Image_WACV_2024_paper.pdf)
[15](https://woo-niverse.tistory.com/130)
[16](https://www.semanticscholar.org/paper/ReStyle:-A-Residual-Based-StyleGAN-Encoder-via-Alaluf-Patashnik/44c0446bb53e951cca8df07af91f1dea96045aea)
[17](https://arxiv.org/html/2505.15822v1)
[18](https://ieeexplore.ieee.org/iel7/9709627/9709628/09711424.pdf)
[19](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/restyle/)
