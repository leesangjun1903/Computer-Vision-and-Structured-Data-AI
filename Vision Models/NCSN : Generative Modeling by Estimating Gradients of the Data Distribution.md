# NCSN : Generative Modeling by Estimating Gradients of the Data Distribution

# 1. 핵심 주장과 주요 기여 (간결 요약)

이 논문의 핵심 아이디어는 “**데이터 분포의 그래디언트(스코어)를 추정해서 Langevin dynamics로 샘플링하면, GAN 없이도 고품질 생성이 가능하다**”는 것이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)

구체적으로, 주요 기여는 다음 네 가지로 요약할 수 있다.

1. **Noise Conditional Score Network (NCSN)** 제안  
   여러 수준의 가우시안 노이즈로 데이터 분포를 퍼트린 뒤, 각 노이즈 수준에 대한 “로그 밀도의 그래디언트(스코어)”를 하나의 네트워크 $\(s_\theta(x,\sigma)\)$ 가 동시에 추정하도록 학습하는 프레임워크를 제안한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)

2. **Annealed Langevin Dynamics**  
   높은 노이즈 수준에서 시작해 점점 노이즈를 줄여가며, 각 단계에서 해당 노이즈 수준의 스코어를 이용해 Langevin dynamics로 샘플링하는 “어닐링된” MCMC 알고리즘을 제안한다. 이는 다봉분포에서 모드 간 이동이 어려운 기존 Langevin의 혼합 문제를 크게 완화한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)

3. **이론적·실증적으로 제시한 두 난점과 해결책**  
   - 데이터가 저차원 매니폴드 위에 있을 때 스코어가 정의되기 어렵고 score matching이 불안정하다는 **매니폴드 문제**  
   - 저밀도 영역에서 데이터가 부족해 스코어 추정이 부정확하고 Langevin 혼합이 매우 느려지는 **저밀도 영역 문제**  
   이를 “여러 수준의 가우시안 노이즈로 분포를 두껍게 만들고(풀 서포트), 저밀도 영역에도 샘플을 뿌린 뒤, 그 전체에 대한 조건부 스코어를 학습”하는 방식으로 해결한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)

4. **GAN 급의 이미지 품질과 일반적 유틸리티 시연**  
   - CIFAR-10 무조건 생성에서 Inception score 8.87, FID 25.32로 당시 SNGAN 등과 경쟁 가능하고, IS는 SOTA를 달성했다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)
   - MNIST, CelebA에서도 GAN에 필적하는 샘플을 보이며, 임의 마스크 인페인팅 등 다양한 역문제에도 잘 적용됨을 보인다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)

***

# 2. 논문의 기술적 내용: 문제, 방법, 구조, 성능 및 한계

## 2.1 해결하고자 하는 문제

저자들은 “스코어 기반 생성”이라는 매우 직관적인 아이디어에서 출발한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)

1. **아이디어**  
   데이터 분포 $\(p_{\text{data}}(x)\)$ 의 로그 밀도 그래디언트
   $$\nabla_x \log p_{\text{data}}(x)$$
   는 고밀도 방향을 가리키는 벡터 필드이다. 이를 근사하는 네트워크
   $$s_\theta(x) \approx \nabla_x \log p_{\text{data}}(x)$$
   를 학습한 뒤, Langevin dynamics
   $$x_{t} = x_{t-1} + \frac{\epsilon}{2} s_\theta(x_{t-1}) + \sqrt{\epsilon}\,z_t,\quad z_t\sim\mathcal N(0,I)$$
   로 이동하면, 랜덤 초기화에서 데이터 분포의 고밀도 영역으로 수렴할 수 있다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)

2. **하지만 두 가지 근본적인 난점이 존재**

   1) **매니폴드 가설 문제**  
   - 실제 고차원 데이터(이미지 등)는 저차원 매니폴드 위에 집중해 있다고 보는 “매니폴드 가설” 하에서,  
     - $\(\log p_{\text{data}}(x)\)$ 는 주변공간 전체에 대한 밀도가 아니라 매니폴드 상의 밀도이므로, 주변공간에서 스코어 $\(\nabla_x \log p_{\text{data}}(x)\)$ 자체가 잘 정의되지 않는다.  
     - Hyvärinen의 score matching 이론은 분포가 전체 공간을 지지(support)할 때만 일관 추정을 보장하는데, 매니폴드 위에 국한되면 이 조건이 깨진다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)

   2) **저밀도 영역에서의 스코어 추정 및 혼합 문제**  
   - score matching 손실은 데이터 분포 평균

$$\frac12\mathbb{E}_{p_{\text{data}}}\big[\|s_\theta(x) - \nabla_x \log p_{\text{data}}(x)\|_2^2\big]$$

를 최소화하므로, 데이터가 거의 없는 저밀도 영역에서는 스코어 추정이 거의 규제되지 않는다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)
   - Langevin dynamics는 보통 저밀도 영역(잡음 초기화)에서 시작하기 때문에, 스코어 오차가 크면 제대로 고밀도 영역으로 이동하지 못한다.  
   - 또한 모드 사이에 저밀도 영역이 놓인 혼합분포에서는, 스코어가 각 모드 내에서만 정의되어 모드 간 상대 질량(혼합비)을 반영하지 못해, 매우 많은 스텝 없이는 올바른 모드 비율을 재현하기 어렵다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)

요약하면, **“원 데이터 분포의 스코어를 직접 추정해서 Langevin으로 샘플링하자”는 아이디어는 매니폴드·저밀도 문제 때문에 그대로는 잘 작동하지 않는다**는 것이 이 논문이 분석하는 출발점이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)

***

## 2.2 제안 방법: Noise Conditional Score Network와 학습/샘플링 알고리즘

### 2.2.1 스코어 매칭의 기본

원래 Hyvärinen의 score matching 목적은 다음과 같다. [papers.baulab](https://papers.baulab.info/papers/also/Ho-2022.pdf)

```math
\min_\theta \frac12 \mathbb{E}_{p_{\text{data}}(x)}\Big[\big\|s_\theta(x) - \nabla_x \log p_{\text{data}}(x)\big\|_2^2\Big].
```

이는 적분 by parts를 통해(경계 조건 가정하에) 다음과 동치이다.

$$
\(\mathcal{J}(\theta )=\mathbb{E}_{p_{\text{data}}(x)}\left[\mathrm{tr}\>\left(\nabla _{x}s_{\theta }(x)\right)+\frac{1}{2}\|s_{\theta }(x)\|_{2}^{2}\right]\)
$$

여기서 $\(\nabla_x s_\theta(x)\)$ 는 Jacobian이며, $\(\mathrm{tr}\ > (\cdot )\)$ 계산이 고차원 딥넷에서는 매우 비싸다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)

이를 완화하기 위해 두 가지 변형을 사용한다. [ewadirect](https://www.ewadirect.com/proceedings/ace/article/view/17684)

1. **Denoising Score Matching (Vincent)**  
   - 노이즈 분포 $\(q_\sigma(\tilde x|x) = \mathcal N(\tilde x\mid x,\sigma^2 I)\)$ 로 노이즈를 주입하고,  
   - 노이즈가 섞인 분포
     $$q_\sigma(\tilde x) = \int q_\sigma(\tilde x|x) p_{\text{data}}(x)\,dx$$
     의 스코어를 학습한다. [arxiv](https://arxiv.org/html/2405.13540v1)
   - 목적함수는

$$
     \ell(\theta;\sigma)
     = \frac12 \mathbb{E}_{p_{\text{data}}(x)} 
     \mathbb{E}_{\tilde x\sim\mathcal N(x,\sigma^2 I)}
     \Big[
     \big\|s_\theta(\tilde x,\sigma) - \nabla_{\tilde x}\log q_\sigma(\tilde x|x)\big\|_2^2
     \Big].
     $$
   
   - 가우시안 조건부의 스코어는 $\(\nabla_{\tilde x}\log q_\sigma(\tilde x|x) = -(\tilde x - x)/\sigma^2\)$ 이므로:

$$
     \ell(\theta;\sigma)
     = \frac12 \mathbb{E}_{p_{\text{data}}(x)} 
     \mathbb{E}_{\tilde x\sim\mathcal N(x,\sigma^2 I)}
     \left[
     \left\| s_\theta(\tilde x,\sigma) + \frac{\tilde x - x}{\sigma^2}\right\|_2^2
     \right].
     $$

2. **Sliced Score Matching (SSM)**  
   - 무작위 방향 $\(v\sim p_v\)$ 를 뽑아 trace term를 1차 미분으로 근사한다. [ewadirect](https://www.ewadirect.com/proceedings/ace/article/view/17684)
   - 목적함수:

$$
     \mathcal{J}_{\text{SSM}}(\theta)
     = \mathbb{E}_{p_v}\mathbb{E}_{p_{\text{data}}(x)}\big[
       v^\top \nabla_x s_\theta(x) v
       + \tfrac12\|s_\theta(x)\|_2^2
     \big].
     $$

이 논문에서는 **학습에는 denoising score matching을, 분석에는 SSM을** 사용한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)

***

### 2.2.2 Noise Conditional Score Network (NCSN)

핵심 발상은 다음 두 단계이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)

1. **다양한 크기의 가우시안 노이즈를 사용해 데이터 분포를 퍼트림**  
   - 표준편차 $\(\{\sigma_i\}_{i=1}^L\)$ (기하수열)를 선택:

$$\sigma_1 > \sigma_2 > \cdots > \sigma_L.$$

   - 각 $\(\sigma\)$ 에 대해

$$
     q_\sigma(x)
     = \int p_{\text{data}}(t)\,\mathcal N(x\mid t,\sigma^2 I)\,dt
     $$
     
  를 정의하면,  
     - $\(\sigma_1\)$ 이 충분히 크면 $\(q_{\sigma_1}\)$ 은 전체 공간을 지지 → 매니폴드 문제 해소  
     - 큰 $\(\sigma\)$ 에서는 원래 분포의 저밀도 영역도 상당한 질량을 가지게 되어, 그 영역의 스코어도 학습 가능.

2. **하나의 네트워크가 (x, σ)에 조건부로 모든 노이즈 수준의 스코어를 예측**  
   - Noise Conditional Score Network:

$$
     s_\theta(x,\sigma) \approx \nabla_x \log q_\sigma(x)
     \quad \text{for all } \sigma\in\{\sigma_i\}_{i=1}^L.
     $$

이를 위해 denoising score matching을 모든 $\(\sigma_i\)$ 에 대해 결합한 목적함수를 사용한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)

#### 통합 학습 목적함수

각 $\(\sigma_i\)$ 에 대한 손실

$$
\ell(\theta;\sigma_i)
= \frac12 \mathbb{E}_{p_{\text{data}}(x)} 
\mathbb{E}_{\tilde x\sim\mathcal N(x,\sigma_i^2 I)}
\left[
\left\| s_\theta(\tilde x,\sigma_i) + \frac{\tilde x - x}{\sigma_i^2}\right\|_2^2
\right].
$$

이를 가중합해 최종 목적함수:

$$
L(\theta)
= \frac1L \sum_{i=1}^L \lambda(\sigma_i)\,\ell(\theta;\sigma_i).
$$

논문에서는 ** $\(\lambda(\sigma) = \sigma^2\)$ ** 를 추천한다. 경험적으로 최적 근처에서 $\(\|s_\theta(x,\sigma)\|_2 \propto 1/\sigma\)$ 라 가정하면,  
$\(\lambda(\sigma)\ell(\theta;\sigma)\)$ 의 규모가 $\(\sigma\)$ 에 크게 의존하지 않도록 균형 잡히기 때문이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)

요약하면, 학습은 “**여러 노이즈 수준 $\(\{\sigma_i\}\)$ 에 대해, 그때의 노이즈가 섞인 분포 $\(q_{\sigma_i}\)$ 의 스코어를 동시에 근사하도록 $\(s_\theta(x,\sigma)\)$ 를 훈련**”하는 과정이다.

***

### 2.2.3 Annealed Langevin Dynamics

학습된 $\(s_\theta(x,\sigma)\)$ 를 이용해 샘플을 생성할 때, 논문은 **노이즈 수준을 고에서 저로 점진적으로 줄이는 Langevin chain**을 제안한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)

노이즈 스케줄 $\(\{\sigma_i\}_{i=1}^L\)$ , 가장 작은 노이즈 기준 스텝 사이즈 $\(\epsilon\)$ , 각 스케일당 스텝 수 $\(T\)$ 를 두고, 알고리즘은 다음과 같다.

1. 초기화: $\(\tilde x_0\sim \pi(x)\)$ (예: Uniform 잡음)
2. for $\(i=1\)$ to $\(L\)$ : (노이즈 수준 큰 것 → 작은 것 순)
   - 스텝 크기 설정:

$$
     \alpha_i = \epsilon\,\frac{\sigma_i^2}{\sigma_L^2}.
     $$
   
   - Langevin dynamics $\(T\)$ step 반복:

$$
     \tilde x_t = \tilde x_{t-1}
       + \frac{\alpha_i}{2} s_\theta(\tilde x_{t-1},\sigma_i)
       + \sqrt{\alpha_i}\, z_t,\quad z_t\sim\mathcal N(0,I).
     $$
   
   - 다음 스케일의 초기값으로 $\(\tilde x_0 \leftarrow \tilde x_T\)$ .

직관은 다음과 같다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)

- $\(\sigma_1\)$ 이 매우 크면, $\(q_{\sigma_1}\)$ 는 모드가 덜 분리되고 저밀도 영역이 메워진 **“평탄한” 분포**가 되어  
  - 스코어 추정이 쉽고,  
  - Langevin 혼합도 빠르다.
- 이 분포에서 얻은 샘플을 $\(\sigma_2\)$ 수준의 초기값으로 사용하면, $\(q_{\sigma_2}\)$ 에서도 이미 비교적 고밀도 영역 근처에서 시작하게 되어 혼합이 쉬워진다.
- 이런 식으로 점차 $\(\sigma_i\)$ 를 줄여가다, 마지막 $\(\sigma_L\)$ 이 충분히 작으면 $\(q_{\sigma_L}\approx p_{\text{data}}\)$ 가 되어, 최종 샘플이 원 분포와 거의 일치한다.

스텝 크기 $\(\alpha_i\propto\sigma_i^2\)$ 선택은 시그널 대 노이즈 비율

$$
\frac{\alpha_i s_\theta}{2\sqrt{\alpha_i}z}
$$

의 규모를 $\(\sigma\)$ 에 의존하지 않게 유지해, 각 스케일에서 비슷한 “업데이트 강도”를 갖도록 하는 경험적 설계이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)

***

## 2.3 모델 구조 (네트워크 아키텍처)

NCSN의 네트워크 아키텍처는 **픽셀 단위 dense prediction 문제(semantic segmentation 등)**에서 성능이 검증된 구조를 적극적으로 차용한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)

주요 요소:

1. **U-Net / RefineNet 기반 구조**  
   - 인코더–디코더 형태에 skip connection을 가진 U-Net 계열(RefineNet 변형) [kimjy99.github](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/ldm/)
   - 이러한 구조는
     - 로컬 디테일(엣지, 텍스처)과
     - 글로벌 구조(형태, 레이아웃)를 동시에 포착하는 데 유리하다.

2. **Dilated (Atrous) Convolution**  
   - 대부분의 다운샘플링 레이어 대신 dilated conv를 사용해 해상도를 유지하면서 receptive field를 크게 가져간다. [bmcpsychology.biomedcentral](https://bmcpsychology.biomedcentral.com/articles/10.1186/s40359-022-00877-7)
   - 이는 고해상도 출력에서 위치 정보를 잃지 않으면서 넓은 문맥을 보는 데 도움이 된다.

3. **Conditional Instance Normalization++ (CondInstanceNorm++)**  
   - $\(\sigma\)$ 마다 다른 scale/shift 파라미터를 사용해 **노이즈 수준을 명시적으로 조건부로 주입**하는 instance normalization 변형을 사용한다. [arxiv](https://arxiv.org/pdf/2304.08291.pdf)
   - 기존 instance norm은 채널별 평균 $\(\mu_k\)$ 를 제거해 색상 shift를 잃는 문제가 있어, 논문은 $\(\mu_k\)$ 의 통계와 learnable parameter $\(\alpha\)$ 를 결합해 색조 정보를 부분적으로 복원하는 확장 버전을 제안한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)

4. **ResNet 블록 + ELU 활성화**  
   - pre-activation residual blocks와 ELU를 사용하며,  
   - 모든 conv와 pooling 앞에 CondInstanceNorm++를 붙이는 식으로 설계된다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)

이 아키텍처는 후속 diffusion/score 기반 모델(예: DDPM/ADM, EDM, Stable Diffusion의 UNet 등)이 채택한 설계의 전신 역할을 한다. [arxiv](http://arxiv.org/pdf/2112.10752.pdf)

***

## 2.4 성능 향상 및 한계

### 2.4.1 성능: GAN에 필적하는 고품질 이미지 생성

실험은 MNIST, CelebA, CIFAR-10에서 수행되며, 주요 결과는 CIFAR-10 무조건 생성이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)

- **CIFAR-10 (unconditional)**  
  - Inception score: 8.87 ± 0.12 (당시 무조건 모델 최고 기록)  
  - FID: 25.32 (SNGAN의 21.7 등과 비슷한 수준) [arxiv](https://arxiv.org/pdf/2207.12598.pdf)

- **MNIST, CelebA**  
  - 정량지표는 다양해 비교가 애매하여 생략하지만, 시각적으로 현대 GAN, likelihood-based 모델에 필적하는 품질을 시연한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)

또한, **임의 마스크 인페인팅(inpainting)** 실험을 통해,  
모델이 의미론적으로 그럴듯한 구조를 복원하고, 한 마스크에 대해 다양한 합리적 보완을 생성할 수 있음을 보여준다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)

이는 **분포 수준의 표현 학습과 일반화**가 일정 수준 이뤄졌음을 시각적으로 뒷받침한다.

### 2.4.2 장점 요약

- **훈련 시 MCMC/Adversarial이 필요 없음**  
  - 순수 score matching 기반으로, 훈련 단계에서 샘플링이 필요 없고, no adversarial training → 안정적이고 구현이 단순하다. [papers.baulab](https://papers.baulab.info/papers/also/Ho-2022.pdf)
- **모델 구조 자유도**  
  - 정규화된 확률모델(autoregressive, flow)처럼 Jacobian determinant 계산이 필요 없으므로, 아키텍처 제약이 적다. [dl.acm](https://dl.acm.org/doi/10.1145/3610542.3626152)
- **명시적인 학습 목적**  
  - 동일 데이터셋에서 여러 모델의 성능(스코어 추정 정확도)을 비교할 수 있는 명시적 손실 $\(L(\theta)\)$ 를 제공한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)

### 2.4.3 한계 및 문제점

1. **샘플링 속도**  
   - Annealed Langevin은 noise level마다 수백 스텝씩, 전체 수천 스텝이 필요해 샘플 하나를 생성하는 데 매우 느리다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)
   - 이후 DDPM/DDIM, SDE 기반 기법들이 이 “many-step MCMC” 병목을 핵심 해결 과제로 삼게 된다. [openreview](https://openreview.net/pdf/ef0eadbe07115b0853e964f17aa09d811cd490f1.pdf)

2. **정확한 likelihood 계산 불가**  
   - NCSN은 **명시적 density 모델이 아니라 score field만 추정**하므로,
     - 로그우도 평가,
     - 정확한 likelihood 기반 비교  
     가 어렵다.  
   - 이후 SDE 기반 score 모델은 probability flow ODE를 통해 exact(?) 혹은 tractable한 likelihood 평가를 가능하게 한다. [emergentmind](https://www.emergentmind.com/topics/score-based-generative-modeling)

3. **고해상도/복잡 도메인 확장**  
   - 논문은 $\(32\times32\)$ 해상도 중심이다.  
   - 이후 연구에서는
     - latent space로 이동(LSGM, Latent Diffusion), [openaccess.thecvf](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf)
     - 더 효율적인 sampler(EDM, consistency models, few-step methods) [arxiv](http://arxiv.org/pdf/2310.04378.pdf)
     를 통해 고해상도와 다중 모달리티로 확장한다.

4. **하이퍼파라미터(노이즈 스케줄, step 수) 민감성**  
   - 노이즈 스케줄 $\(\{\sigma_i\}\), \(\epsilon\), \(T\)$ 선택이 품질과 속도에 크게 영향을 미친다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)
   - Karras et al.의 EDM은 이러한 design space(노이즈 분포, 스케줄링, preconditioning)를 체계적으로 분석하고 최적화하여 동일 스코어 네트워크에서 훨씬 빠르게 SOTA FID를 달성한다. [openreview](https://openreview.net/pdf?id=k7FuTOWMOc7)

***

# 3. 일반화 성능 관점에서 본 NCSN의 의미

질문에서 특별히 강조한 “모델의 일반화 성능 향상 가능성”을 이 논문 중심으로 정리하면 다음과 같다.

## 3.1 매니폴드·저밀도 문제 완화가 곧 일반화 품질 향상

이 논문이 분석한 두 난점(매니폴드·저밀도)은, 사실상 **“학습 분포 전역에서 스코어를 얼마나 잘 일반화해서 추정할 수 있는가”**의 문제이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)

1. **여러 노이즈 수준 훈련이 주는 효과**  
   - 큰 $\(\sigma\)$ 에서의 분포 $\(q_\sigma\)$ 는 원 분포의 저밀도 영역도 충분한 질량을 부여하므로,  
     - 훈련 데이터가 거의 없는 영역에서도 스코어 추정이 강하게 regularize된다.  
   - 이는 **“저밀도/미관측 영역에서의 과도한 불확정성”을 완화하고, 샘플링 초기 단계에서 합리적인 방향으로 이동**하게 한다.

2. **연속적인 noise 스케일 간 공유 파라미터**  
   - 하나의 $\(s_\theta(x,\sigma)\)$ 가 여러 $\(\sigma\)$ 를 공유함으로써,
     - 거친 스케일(큰 $\(\sigma\)$ )에서 학습한 구조적 정보가  
     - 미세 스케일(작은 $\(\sigma\)$ )로 전이된다.  
   - 이는 multi-scale regularization으로 볼 수 있으며, **고주파 세부 구조에 대한 과적합을 완화하고, 분포 전역에 걸친 smoother한 score field를 학습**하는 데 기여한다.

3. **인페인팅 및 다양한 조건부 샘플링에서의 일반화**  
   - 논문은 인페인팅처럼 “부분 관측 + 자유도 많은 영역”을 가진 문제에서도,  
     - 의미론적으로 일관된 다양한 샘플을 생성할 수 있음을 보인다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)
   - 이는 단순한 “훈련 이미지 암기”가 아니라,
     - 학습된 score field가 관측된 부분과 상충되지 않는,  
     - 다양한 가능성을 포괄하는 분포를 내포하고 있음을 시사한다.

이러한 아이디어는 이후 **SDE 기반 score 모델**과 **diffusion 모델 전반의 일반화 특성**을 이해하는 이론·실증 연구의 출발점이 되었다. [iclr](https://iclr.cc/virtual/2021/oral/3402)

## 3.2 후속 이론 연구와의 연결

2020년 이후 여러 이론 연구가 “score-based/diffusion 모델이 왜 이렇게 잘 일반화하는가”를 분석하고 있다.

- **SDE 기반 완전한 프레임워크 (Song et al., 2020)**  
  - 정·역 확산 과정을 확률 미분방정식(SDE)로 기술하고,  
  - score 네트워크가 잘 학습되면 역 SDE/ODE를 따라 노이즈에서 데이터로 이동하는 것이 이론적으로 정당화됨을 보인다. [emergentmind](https://www.emergentmind.com/topics/score-based-generative-modeling)
  - NCSN의 다중 노이즈 학습은 이 SDE 프레임워크에서 “time-discretized” VE/VP SDE의 특수한 경우로 해석된다.

- **저차원 구조에 적응하는 수렴 속도 이론**  
  - DDPM/SGM sampler가 데이터의 “내재 차원”에 맞춰 샘플링 복잡도가 $\(O(k/\varepsilon)\)$ 수준으로 적응할 수 있음을 보이는 이론이 등장했다. [arxiv](https://arxiv.org/abs/2501.12982)
  - 이는 원 논문에서 직관적으로 제기한 “매니폴드+저밀도 문제를 노이즈 스케일링으로 완화한다”는 아이디어를 수학적으로 뒷받침한다.

- **score matching + 샘플링 전체 파이프라인의 비정상(non-asymptotic) 분석**  
  - Wang et al.은 denoising score matching의 gradient descent 수렴과 variance exploding 모델의 샘플링 오차를 통합 분석해,  
    - 어떤 노이즈 분포, 손실 가중, 시간/분산 스케줄이 이론적으로 바람직한지 제시한다. [arxiv](https://arxiv.org/abs/2406.12839)
  - 이 결과는 NCSN/EDM 계열에서 사용되는 노이즈 스케줄 설계가 우연이 아니라 이론적으로도 타당한 선택임을 보여 준다.

- **Diffusion 일반화 메커니즘 직접 분석**  
  - 최근 연구는 사전 학습된 diffusion 모델과 “경험적 최적 denoiser”를 비교하여,  
    - 네트워크가 **지역적 denoising 연산**을 통해 score matching objective를 넓은 영역에서 잘 근사하는 것이 일반화의 핵심임을 관찰한다. [arxiv](https://arxiv.org/html/2411.19339v2)
  - NCSN의 multi-scale 노이즈 훈련은 이러한 “지역적, multi-scale denoising 인덕티브 바이어스”를 강화해, 일반화에 유리한 조건을 만든다.

요약하면, **NCSN의 설계(다중 노이즈 스케일 + 조건부 스코어 학습 + 어닐링 샘플링)는 이후 이론이 규명한 diffusion/score 모델의 일반화 메커니즘과 잘 맞물린 선구적 구조**라고 볼 수 있다.

***

# 4. 2020년 이후 관련 최신 연구 비교 분석

이제 NCSN 이후의 대표적 연구들을, 특히 “구조·훈련 원리·일반화·성능” 관점에서 비교하겠다.

## 4.1 DDPM과의 관계: 변분 관점 vs score matching 관점

**Denoising Diffusion Probabilistic Models (DDPM)** 은 Song & Ermon 계열과 거의 동시에 제안된 또 하나의 축이다. [proceedings.neurips](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)

- **전방 과정**:  
  데이터에 점진적으로 가우시안 노이즈를 추가하는 Markov chain \(q(x_t|x_{t-1})\).
- **역방 과정**:  
  노이즈를 제거하는 $\(p_\theta(x_{t-1}|x_t)\)$ 를 학습.
- 핵심 기여:
  - 이 과정을 **변분 하한(ELBO)** 관점에서 정의하고,  
  - 특정 파라미터화(ε-parameterization)를 사용하면 훈련 손실이
    - 다중 노이즈 수준에서의 denoising score matching과 사실상 동일함을 보였다. [proceedings.neurips](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)
  - 즉, DDPM은 **score matching + Langevin dynamics와 등가 구조**를 변분적·확률모형 형태로 재표현한 셈이다. [hojonathanho.github](https://hojonathanho.github.io/diffusion/)

즉,

- **NCSN**:  
  - 명시적 density 없이 **스코어만 추정**, 샘플링은 Langevin dynamics.  
- **DDPM**:  
  - 명시적 잠재 변수 모델로,  
  - 역 조건부 $\(p_\theta(x_{t-1}|x_t)\)$ 를 학습하고,  
  - 로그우도 평가 가능.  
  - 그러나 학습 목적은 본질적으로 다중 노이즈 수준 denoising score matching과 동일.

두 계열은 이후 Song et al.의 SDE 프레임워크에서 완전히 통합된다. [openreview](https://openreview.net/pdf/ef0eadbe07115b0853e964f17aa09d811cd490f1.pdf)

## 4.2 Score-Based Generative Modeling through SDEs (Song et al., 2020)

이 연구는 **NCSN, DDPM, Langevin 기반 score 모델, 확산 모델을 하나의 SDE/ODE 프레임워크로 통합**한다. [iclr](https://iclr.cc/virtual/2021/oral/3402)

- 전방 과정:  
  - Variance Exploding(VE), Variance Preserving(VP) 등 다양한 SDE로 데이터를 노이즈로 보낸다.
- 역 과정:  
  - time-dependent score 네트워크 $\(s_\theta(x,t)\approx\nabla_x\log p_t(x)\)$ 를 학습하고,  
  - 역 SDE 또는 probability flow ODE를 수치적으로 풀어 샘플링.  

이 프레임워크에서

- **NCSN**은 VE-SDE의 특정 시간 스케줄을 몇 개의 이산 노이즈 수준으로 근사한 모델이며, [openreview](https://openreview.net/pdf/ef0eadbe07115b0853e964f17aa09d811cd490f1.pdf)
- **DDPM**은 VP-SDE의 다른 이산화 및 파라미터화 버전으로 해석할 수 있다. [proceedings.neurips](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)

이 통합은

- **역문제(인페인팅, 컬러라이제이션, 초해상도 등)**에서의 일반화된 formulation을 제공하고, [openreview](https://openreview.net/pdf/ef0eadbe07115b0853e964f17aa09d811cd490f1.pdf)
- exact/tractable likelihood 평가, ODE 기반 빠른 샘플링, 다양한 데이터 도메인(그래프, 포인트 클라우드 등) 확장에 큰 역할을 했다. [dl.acm](https://dl.acm.org/doi/10.1145/3610542.3626152)

## 4.3 Diffusion Models Beat GANs (Dhariwal & Nichol, 2021)

이 논문은 **적절한 아키텍처 및 훈련 개선, classifier guidance를 적용한 diffusion 모델이 GAN을 능가하는 샘플 품질을 달성함**을 보였다. [proceedings.nips](https://proceedings.nips.cc/paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf)

- ImageNet 256×256에서 FID 3.94, Inception Score 7.72 등 GAN 대비 우수한 지표를 기록. [semanticscholar](https://www.semanticscholar.org/paper/Diffusion-Models-Beat-GANs-on-Image-Synthesis-Dhariwal-Nichol/64ea8f180d0682e6c18d1eb688afdb2027c02794)
- 중요한 점:
  - **Classifier guidance**로, class-conditional 모델에서  
    $$\nabla_x \log p_\theta(x|y) + \gamma \nabla_x \log p_\phi(y|x)$$  
    형태의 스코어를 사용, 샘플 품질–다양성 trade-off를 조절한다. [proceedings.nips](https://proceedings.nips.cc/paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf)
  - 이는 결국 “**스코어를 조정해 분포의 특정 부분(고품질 모드)을 강조**”하는 기법이며,  
    NCSN이 제안한 스코어 기반 샘플링의 유연성을 잘 보여주는 사례이다.

여기서부터 diffusion/score 기반 모델은 이미지 생성의 주류로 부상한다.

## 4.4 Classifier-Free Guidance (Ho & Salimans, 2021/2022)

Classifier guidance는 추가 classifier 훈련이 필요하다는 단점이 있다. **Classifier-Free Guidance(CFG)** 는 이를 해결한다. [arxiv](https://arxiv.org/abs/2207.12598)

- 하나의 diffusion/score 모델이 **조건부와 무조건 스코어를 동시에 학습**:
  - $\(s_\theta(x|y)\)$ : 조건부
  - $\(s_\theta(x)\)$ : 무조건
- guidance 시, 두 스코어를 혼합:

$$
  s_{\text{CFG}}(x|y)
  = s_\theta(x) + w\big(s_\theta(x|y) - s_\theta(x)\big),
  $$
  
  여기서 $\(w\)$ 는 guidance 강도. [arxiv](https://arxiv.org/abs/2207.12598)

이는 실질적으로 “**조건부 vs 무조건 score의 차이**”를 증폭해 조건에 강하게 부합하는 샘플을 생성하게 하며, DALL·E 2, Imagen, Stable Diffusion 등 현대 텍스트–이미지 모델의 필수 구성요소가 되었다. [ommer-lab](https://ommer-lab.com/research/latent-diffusion-models/)

NCSN에서 이미 “조건부 score 네트워크” 개념이 있었음을 고려하면, CFG는 **조건부 스코어를 활용한 샘플링 제어**라는 방향의 자연스러운 확장으로 볼 수 있다.

## 4.5 Latent Space로의 확장: LSGM, Latent Diffusion (Stable Diffusion)

고해상도 이미지 및 다양한 모달리티에 대한 확장에서는, **latent space에서 score/diffusion을 수행**하는 방법이 등장했다.

1. **Latent Score-Based Generative Model (LSGM)** [proceedings.neurips](https://proceedings.neurips.cc/paper/2021/file/5dca4c6b9e244d24a30b4c45601d9720-Paper.pdf)
   - VAE encoder–decoder로 latent 공간을 만들고, 그 latent 상에서 score-based generative model을 학습한다.  
   - 장점:
     - 차원 축소로 **샘플링 속도 및 안정성**을 크게 향상.
     - 비연속 데이터(텍스트, 그래프 등)도 latent 연속공간으로 매핑해 활용 가능. [proceedings.neurips](https://proceedings.neurips.cc/paper/2021/file/5dca4c6b9e244d24a30b4c45601d9720-Paper.pdf)

2. **Latent Diffusion Models (LDM, Stable Diffusion)** [arxiv](https://arxiv.org/abs/2112.10752)
   - 고성능 autoencoder로 이미지 → latent로 압축 후,  
   - latent 공간에서 diffusion/detnoising UNet을 학습한다.
   - cross-attention을 통해 텍스트, bounding box 등 다양한 conditioning을 받는다.
   - Stable Diffusion 계열이 바로 이 구조 위에서 구현된다.  
   - LDM은 pixel-space diffusion 대비 수십 배 적은 연산으로 고해상도 이미지를 생성하면서도 SOTA급 품질을 달성한다. [arxiv](http://arxiv.org/pdf/2112.10752.pdf)

LSGM/LDM은 **“NCSN의 score 기반 generative modeling을 latent 공간으로 옮긴 것”**이라는 점에서, NCSN의 구조적 아이디어가 대규모/다중 모달 diffusion에 직접적으로 이어진 사례라 할 수 있다.

## 4.6 EDM: 설계 공간의 명시화와 SOTA FID

**Elucidating the Design Space of Diffusion-Based Generative Models (EDM)** 은 diffusion/score 모델의 설계 요소를 체계적으로 분해·최적화한 작업이다. [github](https://github.com/NVlabs/edm/tree/main)

- 노이즈 분포, 손실 가중, 데이터 preconditioning, sampler 스케줄 등  
  다양한 선택지를 통합적으로 분석하고,  
- 적절한 설계를 통해 CIFAR-10에서  
  - FID 1.79 (class-conditional), 1.97 (unconditional)와 같이  
    매우 적은 스텝(35 NFE 내외)으로 SOTA를 달성한다. [semanticscholar](https://www.semanticscholar.org/paper/Elucidating-the-Design-Space-of-Diffusion-Based-Karras-Aittala/2f4c451922e227cbbd4f090b74298445bbd900d0)

이는 본질적으로 **NCSN/score-matching 기반 모델의 설계 자유도를 “블랙박스 기법”에서 “정교한 엔지니어링 대상”으로 격상**시킨 연구이며,  
NCSN이 연 기초 이론–구조를 실용적인 대규모 시스템으로 밀어붙인 형태로 볼 수 있다.

## 4.7 Few-step & Consistency 계열

긴 chain이라는 한계를 해결하기 위해, **few-step 샘플링**이 가능한 모델들이 등장했다.

- **Consistency Models (CM)** 및 **Latent Consistency Models (LCM)**:  
  - 역 확산/ODE 궤적의 fixed point를 한 번에 예측해,  
    1–2 step 내에 고품질 샘플을 생성하는 consistency-style objective를 사용한다. [arxiv](https://arxiv.org/html/2510.21857v1)
- **Directly Denoising Diffusion Model (DDDM)** 등:  
  - multi-step과 few-step 샘플링을 동시에 지원하는 구조를 도입한다. [arxiv](https://arxiv.org/html/2405.13540v1)

이 역시 **스코어/denoiser를 다중 노이즈 수준에서 학습한다는 NCSN의 기본 아이디어를, 더 강력한 regularization 및 distillation으로 재해석한 것**이라 볼 수 있다.

***

# 5. 향후 연구에 미치는 영향과 앞으로 고려할 점

## 5.1 영향 요약

이 논문이 이후 연구에 남긴 핵심 영향은 다음과 같다.

1. **“스코어를 직접 추정 → 그래디언트 기반 샘플링”이라는 패러다임 확립**  
   - NCSN은 스코어 매칭을 현대 딥러닝 세팅(고차원 이미지, ResNet/U-net 아키텍처)에 성공적으로 이식해,  
     - DDPM, SDE 기반 SGM, EDM, LDM 등 현재 diffusion 세대 모델의 이론적 전신을 제공했다. [openreview](https://openreview.net/pdf?id=k7FuTOWMOc7)

2. **다중 노이즈/스케일 학습의 중요성 제시**  
   - “여러 노이즈 수준에서 joint로 denoising/score 학습”이라는 개념은  
     - DDPM의 다중 timestep 훈련,  
     - SDE score 모델의 continuous-time score estimation,  
     - EDM의 노이즈 분포/가중 최적화,  
     - Consistency/LCM의 multi-σ distillation  
     모두에 깊게 스며들어 있다. [arxiv](http://arxiv.org/pdf/2310.04378.pdf)

3. **inverse problem 및 일반화된 조건부 생성으로의 확장 가능성 입증**  
   - 인페인팅 실험은 score 기반 생성이  
     - 관측 조건을 만족시키면서도 다양한 해를 제시하는,  
     - 강력한 **universal prior**로 활용될 수 있음을 보여주었다. [arxiv](https://arxiv.org/html/2412.04339v1)
   - 이후 SGM/DM은 의료영상 재구성, PET/MRI, HEP unfolding 등 다양한 과학·공학 inverse problem에 적용되고 있다. [arxiv](https://arxiv.org/html/2308.14190v2)

4. **일반화 이론 연구의 출발점 제공**  
   - 매니폴드·저밀도 분석과 multi-scale 노이즈 설계는  
     - score/diffusion 모델의 수렴·일반화 이론과 [arxiv](http://arxiv.org/pdf/2406.01320.pdf)
     - “how do denoisers generalize?” 논의의 중요한 직관적 기반이 되었다.

## 5.2 앞으로 연구 시 고려할 점

NCSN 및 후속 연구를 바탕으로, 앞으로 연구를 진행할 때 고려해야 할 주요 포인트를 정리하면 다음과 같다.

### 5.2.1 일반화와 표현 학습

- **로컬/멀티스케일 denoising 관점에서의 해석**  
  - 스코어 네트워크는 결국 “노이즈가 섞인 데이터에서 clean signal을 복원”하는 로컬 연산을 multi-scale로 수행하는 구조다. [arxiv](https://arxiv.org/html/2411.19339v2)
  - 향후 연구에서는
    - 이러한 로컬 denoising 연산을 더 직접적으로 정규화하거나,
    - 레이어별로 어떤 노이즈 수준/주파수 대역을 담당하는지 해석하는 방향이 중요하다.

- **표현 학습 vs 생성 품질의 trade-off**  
  - 인페인팅, representation transfer 등을 고려하면,
    - 단순 FID/IS뿐 아니라,  
    - downstream 태스크에서의 성능(분류, 검출, RL, 과학 데이터 분석 등)을 함께 최적화하는 목적 설계가 필요하다. [arxiv](https://arxiv.org/html/2308.14190v2)

### 5.2.2 효율성과 설계 공간 탐색

- **샘플링 비용 감소**  
  - Langevin/역 SDE chain의 길이를 줄이기 위한
    - 고차 적분기(EDM),  
    - few-step consistency/flow matching,  
    - hybrid MCMC–ODE 방안  
    을 더욱 체계적으로 결합하는 연구가 요구된다. [github](https://github.com/NVlabs/edm/tree/main)

- **노이즈 스케줄 및 가중치의 이론적 설계**  
  - Wang et al.의 분석처럼, 노이즈 분포와 loss weighting이 수렴/일반화에 미치는 영향을 이론–실험적으로 동시에 최적화하는 방향이 중요하다. [arxiv](https://arxiv.org/abs/2406.12839v2)
  - 특히 NCSN 계열에서는 $\(\{\sigma_i\}\), \(\lambda(\sigma)\), \(\alpha_i\)$ (step size) 설계가 성능을 크게 좌우하므로,  
    - 데이터 도메인별로 자동 튜닝 또는 원리 기반 설계를 개발할 필요가 있다.

### 5.2.3 도메인·데이터 구조 확장

- **비유클리드/구조적 데이터로의 확장**  
  - 그래프, 시계열, 3D 포인트 클라우드, 분자 구조 등에 대한 score/diffusion 모델이 이미 등장했지만,  
    - 해당 도메인의 기하학, 대칭성(group equivariance)을 반영한 score network 설계가 필수다. [arxiv](https://arxiv.org/abs/2302.04313)

- **latent space 설계**  
  - LSGM/LDM처럼 latent 공간에서 score/diffusion을 수행할 때,
    - latent의 geometry, regularization, disentanglement가  
      일반화와 제어 가능성(컨트롤러빌리티)에 큰 영향을 준다. [arxiv](https://arxiv.org/pdf/2206.05895.pdf)
  - 앞으로는 “어떤 latent space가 어떤 태스크/도메인에 가장 적합한가?”를 이론·실험적으로 규명하는 연구가 중요하다.

### 5.2.4 안전성·견고성·평가

- **견고성 및 OOD 일반화**  
  - diffusion/score 모델은 종종 놀라운 OOD generalization을 보이지만,  
    - 특정 도메인에서 초기 seed, conditioning 변화에 극도로 민감한 brittle behavior도 보고된다. [arxiv](https://arxiv.org/abs/2312.11473)
  - 안전한 적용을 위해
    - OOD 감지,
    - uncertainty quantification,
    - adversarial noise/shift에 대한 견고성  
    을 체계적으로 평가·보강하는 것이 필요하다.

- **평가지표 다변화**  
  - FID/IS만으로는
    - 분포 지원 coverage,
    - 미세한 모드 붕괴,  
    - downstream utility  
    를 충분히 포착하지 못한다는 비판이 커지고 있다. [semanticscholar](https://www.semanticscholar.org/paper/Diffusion-Models-Beat-GANs-on-Image-Synthesis-Dhariwal-Nichol/64ea8f180d0682e6c18d1eb688afdb2027c02794)
  - 특히 과학 및 의료 도메인에서는
    - likelihood-based 지표,
    - 태스크별 수치 지표,
    - domain expert 평가  
    를 함께 고려하는 종합적인 evaluation 체계가 필요하다.

***

# 6. 정리

- 이 논문은 **데이터 분포의 그래디언트를 직접 추정하고, 이를 이용해 Langevin dynamics로 샘플을 생성하는 score-based generative modeling**을,  
  - 매니폴드·저밀도 문제를 분석하고  
  - 다중 노이즈 수준의 NCSN과 annealed Langevin dynamics를 통해 실제 고차원 이미지에 성공적으로 적용한,  
  **현대 diffusion/score 모델의 시발점 중 하나**이다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/8ce645b9-c8de-41df-ad7c-d9259842d148/1907.05600v3.pdf)

- 특히,
  - multi-scale 노이즈에 대한 조건부 스코어 학습,
  - 샘플링 단계에서의 어닐링 전략,
  - U-Net 계열 아키텍처의 결합  
  은 이후 DDPM, SDE SGM, EDM, Stable Diffusion, Consistency Models 등에 그대로 계승·확장되며,  
  현재의 **“diffusion/score 기반 생성이 GAN을 능가하는 시대”**를 여는 데 결정적 기여를 했다. [arxiv](http://arxiv.org/pdf/2112.10752.pdf)

- “모델의 일반화 성능” 관점에서 보면,  
  - 이 논문은 저밀도 영역과 매니폴드 경계 근처에서도 유효한 스코어 필드를 학습하기 위해  
    multi-noise denoising을 사용하는 아이디어를 제시했고,  
  - 이는 나중에 이론적으로도 정당화되며, [arxiv](https://arxiv.org/abs/2501.12982)
    score/diffusion 모델이 다양한 도메인과 inverse problem에서 강한 일반화 성능을 보이는 중요한 구조적 원인이 되었다.

향후 연구에서는,  
이 논문이 제시한 **score-based generative modeling의 기본 철학(“denoise to generate”)**을 유지하되,

- 효율성(샘플링/훈련),
- 일반화 이론,
- 도메인 특화 설계,
- 안전성과 평가 체계

를 통합적으로 개선하는 것이, diffusion/score 기반 생성의 다음 단계 과제로 볼 수 있다.

<span style="display:none">[^1_57][^1_58][^1_59][^1_60][^1_61][^1_62][^1_63][^1_64][^1_65][^1_66][^1_67][^1_68][^1_69][^1_70][^1_71][^1_72][^1_73][^1_74][^1_75][^1_76][^1_77][^1_78][^1_79][^1_80][^1_81][^1_82][^1_83][^1_84][^1_85][^1_86][^1_87][^1_88][^1_89][^1_90][^1_91][^1_92][^1_93]</span>

<div align="center">⁂</div>

[^1_1]: 1907.05600v3.pdf

[^1_2]: https://papers.baulab.info/papers/also/Ho-2022.pdf

[^1_3]: https://www.ewadirect.com/proceedings/ace/article/view/17684

[^1_4]: https://arxiv.org/html/2405.13540v1

[^1_5]: https://kimjy99.github.io/논문리뷰/ldm/

[^1_6]: https://arxiv.org/html/2306.15324

[^1_7]: https://bmcpsychology.biomedcentral.com/articles/10.1186/s40359-022-00877-7

[^1_8]: https://arxiv.org/html/2412.14422

[^1_9]: https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf

[^1_10]: https://arxiv.org/pdf/2304.08291.pdf

[^1_11]: https://ommer-lab.com/research/latent-diffusion-models/

[^1_12]: http://arxiv.org/pdf/2406.01320.pdf

[^1_13]: http://arxiv.org/pdf/2112.10752.pdf

[^1_14]: https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf

[^1_15]: https://arxiv.org/abs/2112.10752

[^1_16]: https://arxiv.org/pdf/2112.10752.pdf

[^1_17]: https://www.semanticscholar.org/paper/High-Resolution-Image-Synthesis-with-Latent-Models-Rombach-Blattmann/c10075b3746a9f3dd5811970e93c8ca3ad39b39d

[^1_18]: https://proceedings.nips.cc/paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf

[^1_19]: https://openreview.net/pdf?id=k7FuTOWMOc7

[^1_20]: https://github.com/NVlabs/edm/tree/main

[^1_21]: https://arxiv.org/pdf/2207.12598.pdf

[^1_22]: https://dl.acm.org/doi/10.1145/3610542.3626152

[^1_23]: https://arxiv.org/pdf/2212.09462.pdf

[^1_24]: https://arxiv.org/html/2309.14068v3

[^1_25]: https://openreview.net/pdf/ef0eadbe07115b0853e964f17aa09d811cd490f1.pdf

[^1_26]: https://iclr.cc/virtual/2021/oral/3402

[^1_27]: https://www.semanticscholar.org/paper/014576b866078524286802b1d0e18628520aa886

[^1_28]: https://www.emergentmind.com/topics/score-based-generative-modeling

[^1_29]: https://proceedings.neurips.cc/paper/2021/file/5dca4c6b9e244d24a30b4c45601d9720-Paper.pdf

[^1_30]: http://arxiv.org/pdf/2310.04378.pdf

[^1_31]: https://arxiv.org/html/2510.21857v1

[^1_32]: https://www.semanticscholar.org/paper/Diffusion-Models-Beat-GANs-on-Image-Synthesis-Dhariwal-Nichol/64ea8f180d0682e6c18d1eb688afdb2027c02794

[^1_33]: https://www.semanticscholar.org/paper/Elucidating-the-Design-Space-of-Diffusion-Based-Karras-Aittala/2f4c451922e227cbbd4f090b74298445bbd900d0

[^1_34]: https://arxiv.org/abs/2406.12839

[^1_35]: https://arxiv.org/abs/2406.12839v2

[^1_36]: https://arxiv.org/html/2411.19339v2

[^1_37]: https://arxiv.org/abs/2501.12982

[^1_38]: https://hojonathanho.github.io/diffusion/

[^1_39]: https://arxiv.org/abs/2006.11239

[^1_40]: https://arxiv.org/pdf/2006.11239.pdf

[^1_41]: https://www.semanticscholar.org/paper/Denoising-Diffusion-Probabilistic-Models-Ho-Jain/5c126ae3421f05768d8edd97ecd44b1364e2c99a

[^1_42]: https://arxiv.org/abs/2105.05233

[^1_43]: https://ar5iv.labs.arxiv.org/html/2105.05233

[^1_44]: https://arxiv.org/abs/2207.12598

[^1_45]: https://openreview.net/pdf?id=qw8AKxfYbI

[^1_46]: https://sander.ai/2022/05/26/guidance.html

[^1_47]: https://theaisummer.com/classifier-free-guidance/

[^1_48]: https://ar5iv.labs.arxiv.org/html/2207.12598

[^1_49]: https://arxiv.org/abs/2206.00364

[^1_50]: https://arxiv.org/html/2412.04339v1

[^1_51]: https://arxiv.org/html/2308.14190v2

[^1_52]: http://arxiv.org/pdf/2406.01507.pdf

[^1_53]: https://arxiv.org/abs/2302.04313

[^1_54]: https://arxiv.org/pdf/2206.05895.pdf

[^1_55]: https://arxiv.org/abs/2312.11473

[^1_56]: https://openaccess.thecvf.com/content/ICCV2025/papers/Jiang_Moderating_the_Generalization_of_Score-based_Generative_Model_ICCV_2025_paper.pdf

[^1_57]: https://arxiv.org/abs/2207.13038

[^1_58]: https://arxiv.org/abs/2211.13095

[^1_59]: https://link.springer.com/10.1007/s11263-025-02526-z

[^1_60]: https://www.semanticscholar.org/paper/1cada6a84e29ec5b27939e5cef977fa654779e3c

[^1_61]: https://www.mdpi.com/2673-8112/2/12/119

[^1_62]: https://doi.apa.org/doi/10.1037/xge0001290

[^1_63]: https://arxiv.org/abs/2509.19276

[^1_64]: https://arxiv.org/pdf/2208.14699.pdf

[^1_65]: https://arxiv.org/html/2406.14815

[^1_66]: https://arxiv.org/html/2411.04873

[^1_67]: https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html

[^1_68]: https://yang-song.net/blog/2021/score/

[^1_69]: https://arxiv.org/pdf/2512.04985.pdf

[^1_70]: https://arxiv.org/abs/2304.08818

[^1_71]: https://arxiv.org/html/2505.19210v2

[^1_72]: https://www.semanticscholar.org/paper/5c126ae3421f05768d8edd97ecd44b1364e2c99a

[^1_73]: http://pubs.rsna.org/doi/10.1148/ryai.2020200007

[^1_74]: https://www.semanticscholar.org/paper/2aab1a79341e4967e31b8efab4dfaf1f96596b74

[^1_75]: https://www.semanticscholar.org/paper/91b32fc0a23f0af53229fceaae9cce43a0406d2e

[^1_76]: https://www.semanticscholar.org/paper/d7d9ec048cc0a320a4fc0f88da16b2ad19ad8873

[^1_77]: https://ieeexplore.ieee.org/document/9887996/

[^1_78]: https://arxiv.org/pdf/2107.03006.pdf

[^1_79]: https://www.ewadirect.com/proceedings/ace/article/view/17684/pdf

[^1_80]: https://liner.com/review/diffusion-models-beat-gans-on-image-synthesis

[^1_81]: https://learnopencv.com/denoising-diffusion-probabilistic-models/

[^1_82]: https://kimjy99.github.io/논문리뷰/dmbg/

[^1_83]: https://www.youtube.com/watch?v=OYiQctx7kDE

[^1_84]: https://letter-night.tistory.com/207

[^1_85]: https://openreview.net/forum?id=AAWuCvzaVt

[^1_86]: https://github.com/NVlabs/edm/blob/main/README.md

[^1_87]: https://arxiv.org/pdf/2402.04384.pdf

[^1_88]: https://openaccess.thecvf.com/content/WACV2024/papers/Stypulkowski_Diffused_Heads_Diffusion_Models_Beat_GANs_on_Talking-Face_Generation_WACV_2024_paper.pdf

[^1_89]: https://arxiv.org/pdf/2312.04370.pdf

[^1_90]: https://arxiv.org/abs/2107.03006

[^1_91]: https://arxiv.org/pdf/2507.18534.pdf

[^1_92]: https://arxiv.org/pdf/2105.05233.pdf

[^1_93]: https://arxiv.org/html/2406.12839v1
