# Towards Practical Plug-and-Play Diffusion Models

***

### **1. 핵심 주장 및 주요 기여 요약**

이 논문은 기존의 **"Plug-and-Play" (PnP)** 확산 모델 가이던스 방식이 실제로는 노이즈가 섞인 중간 생성물에 대해 외부 모델(Guidance Model)이 제대로 작동하지 않아 '실용적이지 않다'는 문제를 지적합니다. 이를 해결하기 위해 **(1) 다중 전문가(Multi-Experts) 전략**, **(2) 파라미터 효율적 미세조정(Parameter-Efficient Fine-Tuning)**, **(3) 데이터 프리 지식 전이(Data-Free Knowledge Transfer)**를 결합한 **PPAP (Practical Plug-and-Play)** 프레임워크를 제안합니다. 핵심 기여는 **레이블이 지정된 데이터셋 없이도** 일반적인 기학습 모델(Classifier, Depth Estimator 등)을 확산 모델의 가이던스로 즉시 활용할 수 있게 하여, 진정한 의미의 범용적이고 효율적인 제어 생성을 가능하게 한 점입니다.

***

### **2. 상세 분석: 문제, 방법, 구조, 성능 및 한계**

#### **2.1. 해결하고자 하는 문제 (Problem Definition)**
기존 연구들은 사전 학습된 분류기(Off-the-shelf classifier)를 사용하여 확산 모델의 생성 과정을 제어하려 했습니다. 그러나 확산 과정의 중간 이미지($x_t$)는 심한 노이즈를 포함하고 있어, 깨끗한 이미지($x_0$)로 학습된 일반 모델은 이를 인식하지 못합니다.
*   **기존의 한계:** 이를 해결하기 위해 노이즈 섞인 데이터로 모델을 재학습(Fine-tuning)시키려 했으나, **(1) 하나의 모델이 모든 노이즈 레벨(Timestep $0 \sim T$)을 커버하기 어렵고**, **(2) 학습을 위한 대규모 레이블 데이터셋이 필요**하다는 문제가 있어 확장성이 떨어졌습니다.

#### **2.2. 제안하는 방법 (Proposed Method)**

저자들은 노이즈 레벨에 따라 모델이 주목하는 특징이 다르다는 점(초기: 구조/형태, 후기: 텍스처/세부사항)에 착안하여 **PPAP 프레임워크**를 제안합니다.

**1) 다중 전문가 전략 (Multi-Experts Strategy)**
전체 타임스텝 $T$를 $N$개의 구간으로 나누고, 각 노이즈 구간에 특화된 $N$개의 '전문가' 모델을 둡니다.
특정 타임스텝 $t$에서의 가이던스는 해당 구간을 담당하는 전문가 $f_{\phi_n^*}$가 수행합니다.

$$
x_{t-1} = \frac{1}{\sqrt{1-\beta_t}}(x_t - \frac{\beta_t}{\sqrt{1-\alpha_t}}\epsilon_\theta(x_t, t)) + \sigma_t z - s\sigma_t \nabla_{x_t} \mathcal{L}_{guide}(f_{\phi_n^*}(x_t), y_{target})
$$

**2) 파라미터 효율적 미세조정 (Parameter-Efficient Fine-Tuning)**
$N$개의 모델을 전부 저장하는 것은 비효율적이므로, **LoRA (Low-Rank Adaptation)**와 같은 기법을 사용하여 기존 모델(Backbone)은 고정하고 아주 적은 수의 파라미터(예: 전체의 <5%)만 각 전문가별로 학습시킵니다.

**3) 데이터 프리 지식 전이 (Data-Free Knowledge Transfer)**
외부 데이터셋 없이 학습하기 위해, 확산 모델이 생성한 깨끗한 이미지( $\tilde{x}\_0$ )를 입력으로 하여 원본 모델( Teacher, $f_\phi$ )이 예측한 결과를 정답(Pseudo-label)으로 사용합니다. 이를 통해 전문가 모델($f_{\phi_n^*}$)은 노이즈가 섞인 이미지($\tilde{x}_t$)를 보고도 원본 모델과 같은 출력을 내도록 학습됩니다.

*   **지식 전이 손실 함수 (Knowledge Transfer Loss):**

$$
    \mathcal{L}_{KT} = \mathbb{E}_{t \sim \text{unif}} [\mathcal{L}(\texttt{sg}(f_\phi(\tilde{x}_0)), f_{\phi_n^*}(\tilde{x}_t))]
    $$

여기서 $\texttt{sg}$는 stop-gradient 연산이며, 분류 문제의 경우 $\mathcal{L}$은 KL-divergence가 됩니다.

#### **2.3. 모델 구조 (Model Structure)**
*   **Generative Model:** 학습된 Unconditional Diffusion Model (가중치 고정).
*   **Guidance Model:** 사전 학습된 Off-the-shelf Model (예: ResNet50, MiDaS, DeepLabV3) + $N$개의 경량화된 어댑터(LoRA layers).
*   **Process:** Inference 시 타임스텝 $t$에 따라 적절한 어댑터 파라미터만 스위칭하여 가이던스 신호(Gradient)를 계산합니다.

#### **2.4. 성능 향상 및 한계**
*   **성능:** ImageNet 조건부 생성 실험에서, 기존의 '단일 노이즈 학습 모델(Single Noise-aware)' 대비 **FID(낮을수록 좋음)는 약 10점 감소(30.42 → 19.98)**, **IS(높을수록 좋음)는 대폭 상승(43.05 → 74.78)**하여 레이블 데이터 없이도 지도 학습 모델을 능가하는 성능을 보였습니다.[1]
*   **한계:**
    1.  **데이터 분포 불일치:** 가이던스 모델이 학습하지 못한 도메인의 데이터를 확산 모델이 생성하는 경우(예: 만화 스타일), 가이던스가 부정확할 수 있습니다.
    2.  **입력 제한:** 현재는 단일 이미지 입력을 받는 모델(분류기 등)에 최적화되어 있어, 텍스트나 오디오 등 멀티모달 입력 모델로의 확장은 추가 연구가 필요합니다.

***

### **3. 모델의 일반화 성능 향상 가능성**

이 논문에서 가장 주목할 만한 점은 **"일반화(Generalization)"**에 대한 접근 방식입니다.

1.  **Task Generalization (작업의 확장성):**
    PPAP는 **'Data-Free'** 방식이므로, 새로운 태스크(예: 깊이 추정, 세그멘테이션)를 위한 가이던스를 구축할 때 해당 태스크를 위한 **노이즈 데이터셋을 따로 수집할 필요가 없습니다.** 단순히 확산 모델로 이미지를 생성하고, 그 이미지에 대해 사전 학습된 모델(Teacher)의 출력을 흉내 내도록 전문가를 학습시키면 됩니다.
    *   *실증:* 논문에서는 ImageNet Classifier 외에도 **MiDaS(Depth)**, **DeepLabV3(Segmentation)** 모델을 별도의 데이터 수집 없이 그대로 가져와 GLIDE 모델을 제어하는 데 성공했습니다.

2.  **Domain Generalization (도메인 확장성):**
    ImageNet으로 학습된 분류기를 사용하여 GLIDE(더 광범위한 데이터로 학습됨)를 제어할 때, **ResNet이 본 적 없는 '만화 스타일'이나 '스케치'** 이미지에 대해서도 클래스 가이던스가 작동함을 보였습니다. 이는 전문가 모델이 확산 모델이 생성하는 다양한 분포(Distribution) 위에서 지식 전이를 수행했기 때문에, 원본 모델보다 더 넓은 도메인에 대해 강건함(Robustness)을 갖게 되었음을 시사합니다.

***

### **4. 향후 연구에 미치는 영향 및 고려할 점**

*   **영향 (Impact):**
    *   이 연구는 "확산 모델의 제어(Control)"를 위해 **거대 모델을 처음부터 다시 학습하거나(Retraining)**, **비싼 레이블 데이터를 구축**해야 하는 진입 장벽을 제거했습니다.
    *   이후 연구들(Universal Guidance 등)이 "Training-Free" 혹은 "Lightweight Adaptation" 방향으로 나아가는 데 중요한 기틀을 마련했습니다.
*   **연구 시 고려할 점:**
    *   **Inference Cost:** 다중 전문가를 사용하지만 파라미터 스위칭 방식이므로 메모리 오버헤드는 적습니다. 그러나 실시간 애플리케이션에서는 어댑터 로딩 시간 최적화가 필요할 수 있습니다.
    *   **Guidance Scale 민감도:** 가이던스 강도($s$)에 따라 이미지 품질(Fidelity)과 다양성(Diversity)의 트레이드오프가 심하므로, 이를 동적으로 조절하는 스케줄링 기법이 고려되어야 합니다.

***

### **5. 2020년 이후 관련 최신 연구 탐색**

PPAP 이후 확산 모델의 제어 가능성(Controllability)을 높이기 위한 연구가 활발히 진행되었습니다.

1.  **Universal Guidance for Diffusion Models (CVPR 2023, Bansal et al.)**[2][3]
    *   **내용:** PPAP와 달리 **추가 학습(Fine-tuning)을 아예 배제**하고, Forward/Backward Guidance 알고리즘만으로 모든 Off-the-shelf 모델(CLIP, Segmentation, FaceID 등)을 가이던스로 사용하는 방법을 제안했습니다.
    *   **비교:** PPAP보다 더 범용적이나, 최적화 과정에서 계산 비용이 높고 가이던스 성공률이 다소 불안정할 수 있습니다.

2.  **Training-Free Guidance (TFG) & FreeDoM (2023-2024)**[2]
    *   **내용:** 에너지 기반 모델(EBM) 관점에서 가이던스를 해석하여, 별도의 학습 없이 다양한 조건(Face landmark, Style 등)을 부여하는 기법들입니다. "Time-Travel" 전략 등을 사용하여 가이던스 품질을 높입니다.

3.  **Efficient Diffusion Models & Mixture-of-Experts (2024-2025)**[4][5]
    *   **내용:** 최근(2025년)에는 확산 모델 자체의 효율성을 위해 **MoE (Mixture-of-Experts)** 구조를 도입하는 연구가 등장했습니다. PPAP가 가이던스 모델에 MoE를 썼다면, 최신 연구는 생성 모델 자체에 DeepSeek-V3 스타일의 MoE를 적용하여 학습 및 추론 효율을 극대화하고 있습니다.

4.  **InverseBench & Scientific PnP (2025)**[6]
    *   **내용:** Plug-and-Play 확산 모델을 의료 영상(MRI), 블랙홀 이미징 등 **과학적 역문제(Inverse Problems)** 해결에 적용하고 벤치마킹하는 연구로 확장되고 있습니다.

이러한 흐름은 **"거대 생성 모델은 고정하고(Frozen), 가벼운 모듈이나 알고리즘만으로 원하는 목적에 맞게 제어한다"**는 방향으로 수렴하고 있습니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/cea2ed98-24cc-4794-8276-2c525087bf73/2212.05973v2.pdf)
[2](https://www.emergentmind.com/topics/universal-guidance-algorithm-for-diffusion-models)
[3](https://openaccess.thecvf.com/content/CVPR2023W/GCV/papers/Bansal_Universal_Guidance_for_Diffusion_Models_CVPRW_2023_paper.pdf)
[4](https://arxiv.org/html/2502.06805v1)
[5](https://arxiv.org/html/2512.01252v1)
[6](https://openreview.net/forum?id=U3PBITXNG6)
[7](https://ieeexplore.ieee.org/document/10204687/)
[8](https://www.semanticscholar.org/paper/d530b0d4f22770016002b0d8ca798d916cc5f329)
[9](https://www.semanticscholar.org/paper/e41da959be53fd3ad9d9dd0bf3a28545238fd7ac)
[10](https://arxiv.org/abs/2212.05973v2)
[11](http://arxiv.org/pdf/2410.11795.pdf)
[12](http://arxiv.org/pdf/2412.17162.pdf)
[13](https://arxiv.org/html/2404.07771v1)
[14](https://arxiv.org/html/2405.17401)
[15](https://arxiv.org/html/2311.09262v2)
[16](https://arxiv.org/pdf/2401.08740.pdf)
[17](https://openaccess.thecvf.com/content/CVPR2023/papers/Go_Towards_Practical_Plug-and-Play_Diffusion_Models_CVPR_2023_paper.pdf)
[18](https://arxiv.org/html/2510.22835v1)
[19](https://scholar.google.com/citations?user=ZVzn5Y8AAAAJ&hl=ko)
[20](https://ai4d3.github.io/2025/papers/49_Monte_Carlo_Tree_Diffusion_.pdf)
[21](https://dblp.org/pid/218/7548)
[22](https://arxiv.org/html/2510.00430v1)
