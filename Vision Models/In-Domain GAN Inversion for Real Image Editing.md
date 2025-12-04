# In-Domain GAN Inversion for Real Image Editing

1. 핵심 주장과 주요 기여 (간결 요약)  
***

- **핵심 주장**:  
  기존 GAN inversion은 **픽셀 재구성 오차 최소화**에만 초점을 맞춰, 얻어진 잠재 코드가 GAN이 학습한 **semantic manifold(의미 공간)** 안에 놓이지 않아 “편집(editability)”이 크게 떨어진다는 문제를 가진다. 이 논문은 **“in-domain”이라는 개념**을 도입하여,  
  1) 입력 이미지를 고충실도로 복원하면서도  
  2) 잠재 코드가 GAN의 원래 latent 공간의 의미 구조를 따르도록  
  하는 **In-Domain GAN Inversion(IDInvert)** 방법을 제안한다.[1]

- **주요 기여**:[1]
  1) **Domain-guided encoder**:  
     - 단순히 $z$를 복원하는 encoder가 아니라,  
       $$x_{\text{real}} \to E(x_{\text{real}}) = z \to G(z) \approx x_{\text{real}}$$  
       을 직접 학습하며, VGG perceptual loss와 GAN discriminator를 활용해 **“GAN이 학습한 의미 도메인 안의 코드(in-domain code)”**를 출력하도록 유도.  
  2) **Domain-regularized optimization**:  
     - inference 시 encoder 출력 코드를 초기값으로 두고,  
       - 픽셀/퍼셉추얼 재구성  
       - **encoder-consistency 정규화(term)**  
       를 동시에 최적화하는 새로운 objective를 제안해, **재구성-편집 성능을 동시에 향상**.  
  3) **정량·정성 평가로 검증**:  
     - FID, SWD, MSE 기준으로 기존 Image2StyleGAN 등 대비 더 나은 재구성 품질 및 속도,  
     - InterFaceGAN 기반 latent classifier로 평가 시 더 높은 semantic alignment,  
     - 다양한 real image editing(보간, 속성 편집, semantic diffusion)에서 **월등한 편집 품질**을 보여줌.[1]

***

2. 논문 상세 설명  
***

### 2.1 해결하고자 하는 문제

**기본 배경**  
- 잘 학습된 GAN(특히 StyleGAN)은 latent space에서 선형 방향 이동만으로 나이, 성별, 안경, 포즈 등 다양한 **고수준 의미(semantics)** 를 조작할 수 있음.[1]
- 하지만 **real image**를 편집하려면, 먼저 real image를 GAN의 latent space로 **inversion** 해야 한다.  

**기존 방법의 한계**[1]
1) **Learning-based encoder inversion**  
   - $z \sim p(z)$ 로부터 이미지 $x_{\text{syn}} = G(z)$를 생성하고,  
     $$x_{\text{syn}} \xrightarrow{E} \hat{z} \approx z$$  
     를 학습(예: IDInvert 등).  
   - 문제: supervision이 $\|z - E(G(z))\|$ 뿐이라 **“ $z$ 복구”에만 초점**이고,  
     - real image에 대한 일반화가 약하며,  
     - encoder가 생성물 분포에만 맞춰져 실제 데이터 도메인에서 **semantic alignment**가 보장되지 않는다.[1]

2) **Optimization-based inversion (Image2StyleGAN 등)**  
   - 개별 이미지 $x$에 대해  
     $$z^* = \arg\min_z \mathcal{L}_{\text{recon}}(x, G(z))$$  
     를 직접 gradient descent로 최적화.  
   - 픽셀/퍼셉추얼 수준 재구성은 우수하지만  
     - 잠재 코드가 **“GAN이 의도한 latent 도메인” 바깥(out-of-domain)** 으로 나가므로,  
     - 기존에 발견된 semantic 방향(예: InterFaceGAN boundary)을 따라 조작해도 원하는 속성만 바뀌지 않고, 아티팩트·정체성 붕괴 등이 발생.[1]

**논문의 문제 정의**  
- GAN inversion은 단순히  

$$G(z_{\text{inv}}) \approx x$$  만 만족해서는 안 되고,  

$$z_{\text{inv}} \in \mathcal{S}$$  

(여기서 $\mathcal{S}$는 GAN이 학습한 **semantic domain**/latent manifold) 를 만족해야 한다고 주장.  
- 즉, **“재구성 fidelity + semantic in-domain 제약”을 동시에 만족하는 inversion** 방법을 제안하는 것이 목표.[1]

***

### 2.2 제안 방법 개요

제안 방법은 크게 **두 단계**로 구성된다.[1]

1) **Domain-guided encoder $E$ 학습 (distribution-level)**  
   - real image $x_{\text{real}}$ 에 대해,  
     $$x_{\text{real}} \xrightarrow{E} z \xrightarrow{G} \hat{x} \approx x_{\text{real}}$$  
   - 이때 GAN의 **generator $G$와 discriminator $D$를 고정**하고,  
     - 픽셀 L2 loss  
     - VGG perceptual loss  
     - adversarial loss  
     를 활용해 $E$를 학습, encoder 출력 코드들이 **항상 in-domain** 이 되도록 함.  

2) **Domain-regularized optimization (instance-level)**  
   - 테스트 시, real image $x$에 대해  
     - 먼저 $z_0 = E(x)$ 로 초기화,  
     - 이후 $z$를 직접 최적화하되,  
       - 재구성 손실 + perceptual loss + **encoder consistency 정규화 $\|z - E(G(z))\|^2$** 를 동시에 최소화.  
   - 이를 통해  
     - $z$가 encoder가 정의하는 in-domain 영역에서 크게 벗어나지 않게 유지(semantic alignment),  
     - 동시에 개별 이미지에 대한 재구성 fidelity도 높임.  

요약하면,  
> **“Encoder가 정의하는 in-domain manifold에 projection + 그 주변에서의 instance-level refinement”**  
라는 2단계 구조로, 재구성과 편집성의 trade-off를 동시에 잡으려는 설계이다.[1]

***

### 2.3 수식 중심 설명

#### 2.3.1 기존 encoder 학습 (비교 대상)

기존 deterministic encoder 기반 inversion은 보통 다음 objective로 학습된다.[1]

$$
\min_{\Theta_E} L_E 
= \left\| z_{\text{sam}} - E(G(z_{\text{sam}})) \right\|_2^2
$$

- $z_{\text{sam}} \sim p(z)$, $x_{\text{syn}} = G(z_{\text{sam}})$  
- encoder는 **생성된 이미지** $G(z_{\text{sam}})$를 입력으로 받아 latent code를 복원하도록 학습.  
- 문제:  
  - supervision이 latent space 상의 L2 뿐이고,  
  - real image distribution을 거의 보지 못하며,  
  - $G$의 gradient를 활용하지 못해 GAN이 학습한 semantics를 충분히 반영하지 못한다.[1]

#### 2.3.2 Domain-guided encoder objective

IDInvert가 제안하는 **domain-guided encoder**는 다음 objective로 학습된다.[1]

Encoder loss:

$$
\begin{aligned}
\min_{\Theta_E} L_E 
&= \left\| x_{\text{real}} - G(E(x_{\text{real}})) \right\|_2^2 \\
&\quad + \lambda_{\text{vgg}} \left\| F(x_{\text{real}}) - F(G(E(x_{\text{real}}))) \right\|_2^2 \\
&\quad - \lambda_{\text{adv}} \, \mathbb{E}_{x_{\text{real}}\sim P_{\text{data}}} \left[ D(G(E(x_{\text{real}}))) \right],
\end{aligned}
$$

Discriminator loss:

```math
\begin{aligned}
\min_{\Theta_D} L_D 
&= \mathbb{E}_{x_{\text{real}}\sim P_{\text{data}}} \left[ D(G(E(x_{\text{real}}))) \right] 
 - \mathbb{E}_{x_{\text{real}}\sim P_{\text{data}}} \left[ D(x_{\text{real}}) \right] \\
&\quad + \frac{\gamma}{2} \, \mathbb{E}_{x_{\text{real}}\sim P_{\text{data}}} \left[ \left\| \nabla_x D(x_{\text{real}}) \right\|_2^2 \right].
\end{aligned}
```

여기서  
- $F(\cdot)$: VGG 네트워크의 중간 feature (perceptual loss)[1]
- $\lambda_{\text{vgg}}$, $\lambda_{\text{adv}}$, $\gamma$: 하이퍼파라미터  

핵심 포인트:  
- **loss가 latent space가 아니라 image space에서 정의**되어,  
  - encoder 출력 $z = E(x_{\text{real}})$가 반드시 $G$를 통해 **real-like 이미지를 생성**해야만 loss가 작아진다.  
- GAN의 $G$와 $D$ 양쪽의 도메인 지식을 적극 활용함으로써,  
  - encoder가 출력하는 모든 $z$가 **“GAN이 학습한 도메인 안의 의미 있는 코드(in-domain code)”**가 되도록 유도한다.[1]

#### 2.3.3 Domain-regularized optimization objective

테스트 단계에서, 주어진 이미지 $x$에 대해 다음을 최적화한다.[1]

```math
\begin{aligned}
z_{\text{inv}} = \arg\min_{z} \ \ &
\left\| x - G(z) \right\|_2^2 
+ \lambda_{\text{vgg}} \left\| F(x) - F(G(z)) \right\|_2^2 \\
&\quad + \lambda_{\text{dom}} \left\| z - E(G(z)) \right\|_2^2.
\end{aligned}
```

- 첫 두 항: reconstruction (pixel + perceptual)  
- 마지막 항: **domain regularizer**  
  - $z$가 생성 이미지 $G(z)$를 다시 encoder에 넣었을 때의 코드 $E(G(z))$와 크게 다르지 않도록 한다.  
  - 직관적으로,  
    - $E$가 학습한 in-domain manifold에 대한 **projection-like 제약** 역할을 한다.  
    - $z$가 out-of-domain 방향으로 멀리 나가면, $E(G(z))$와 차이가 커지며 penalty 증가.  

실험에서 $\lambda_{\text{dom}}$ 값을 바꾸며  
- $\lambda_{\text{dom}} = 0$: 재구성은 가장 좋지만, 코드가 out-of-domain이 되어 편집성이 떨어짐.  
- $\lambda_{\text{dom}}$ 이 커질수록: semantic alignment는 좋아지지만, 픽셀 재구성은 다소 희생.  
라는 **재구성–편집성 trade-off**를 정량·정성적으로 보여준다.[1]

#### 2.3.4 잠재 공간 선택 ($\mathcal{W}$ vs $\mathcal{Z}$)

- StyleGAN의 **$\mathcal{W}$ space**가 $\mathcal{Z}$보다 의미 disentanglement가 높고, semantic linear separability가 좋다는 기존 결과를 따라,[2]
  본 논문도 **$\mathcal{W}$를 inversion 대상 공간**으로 선택한다.[1]
- 이는 편집성(semantic control) 관점에서 일반화 능력을 높이는 설계 선택이라고 볼 수 있다.

#### 2.3.5 Semantic manipulation 수식

InterFaceGAN 스타일의 방향 벡터 $n$ (예: 나이, 성별, 안경, 포즈)을 찾은 뒤,  
inverted code $z_{\text{inv}}$를 선형 이동하여 편집한다.[1]

$$
x_{\text{edit}} = G\left( z_{\text{inv}} + a n \right),
$$

- $a$는 조작 강도(step size).  
- 이러한 조작에서 IDInvert의 코드는  
  - identity, 배경, 기타 속성을 잘 유지하면서  
  - 목표 속성만 자연스럽게 변화시키는 결과를 보여,  
  - **in-domain code**가 실제로 semantic editing에 적합함을 입증했다.[1]

***

### 2.4 모델 구조

**구성 요소**[1]
1) **고정된 GAN (StyleGAN 계열)**  
   - Generator $G$: FFHQ, LSUN Tower, LSUN Bedroom 등 데이터셋으로 사전 학습.  
   - Discriminator $D$: encoder 학습 시 adversarial loss로만 사용.  

2) **Domain-guided encoder $E$**  
   - ResNet 계열 backbone 위에 StyleGAN의 style vector dimension에 맞는 fully connected head들로 구성.  
   - 출력은 layer-wise $\mathcal{W}$ 코드(StyleGAN의 각 레이어에 주입).  

3) **Inference pipeline**  
   - 입력 $x$에 대해  
     1) $z_0 = E(x)$  
     2) 위의 domain-regularized objective를 $z$에 대해 GD로 최적화  
   - 최종 $z_{\text{inv}}$를 편집에 사용.  

**학습 세팅**[1]
- Encoder 학습:  
  - Generator 고정, $E$와 보조 $D$만 업데이트.  
  - Loss weight: $\lambda_{\text{vgg}} = 5\times 10^{-5}$, $\lambda_{\text{adv}} = 0.1$, gradient penalty $\gamma = 10$.  
- Domain-regularized optimization:  
  - $\lambda_{\text{dom}} = 2$ 사용(재구성과 편집의 균형점)  

***

### 2.5 성능 향상

#### 2.5.1 Semantic alignment (일반화된 의미 표현력)

- InterFaceGAN을 이용해 latent space 내 나이, 성별, 안경, pose 등 **semantic boundary**를 찾고,  
  - real face 이미지 7,000장을 invert하여 그 latent code들이 해당 boundary의 어느 쪽에 있는지로 attribute classifier를 구성.  
- Image2StyleGAN vs IDInvert 비교에서,  
  - **Precision–Recall 곡선이 전 attribute에서 크게 개선**되었음을 보고.  
  - 즉, inverted code만으로 attribute를 분류했을 때 IDInvert가 훨씬 잘 맞는다 ⇒  
    **“inverted code가 실제 semantic 정보를 잘 보존”** 하고 있다는 의미.[1]

이는 **“semantic generalization”의 핵심 증거**로 볼 수 있다.  
- 특정 real image에서만 잘 동작하는 것이 아니라,  
- 전체 real 데이터 분포 상에서 latent code와 attribute 간의 관계가 더 선형·안정적으로 유지된다.

#### 2.5.2 재구성 품질 및 속도

- FID, SWD, MSE 기준 비교(500장 이미지):[1]

  - Face 데이터셋에서  
    - 전통 encoder < Image2StyleGAN(최적화) < Domain-guided encoder(ours) < In-domain inversion(ours)  
    - FID, SWD, MSE 모두에서 **제안 방법이 최고**.  
  - Tower에서도 유사한 패턴.  

- 속도 측면:[1]
  - 순수 optimization 기반(Image2StyleGAN): 한 장당 ≈ 290초  
  - 제안 in-domain inversion: ≈ 8초 (≈ 35배 빠름)  
  - pure encoder: ≈ 0.017초 수준으로 real-time inference 가능  

⇒ **편집 가능한 inversion을 상당히 빠르게 제공**한다는 점에서 practical generalization(실사용 가능성)도 높다.

#### 2.5.3 Real image editing 사례

- **Interpolation**:  
  - 두 inverted code 사이 선형 보간 시  
    - 얼굴: identity/표정/성별이 매끄럽게 변화, 중간 샘플도 자연스러운 얼굴 유지  
    - 타워: 형태가 달라도 중간 샘플 모두 “그럴듯한 타워”로 유지 ⇒ in-domain property.[1]
- **Attribute manipulation**:  
  - 포즈 회전, 안경 추가/삭제, 표정 변화, 구름/해, 실내 조명, 목재 재질 등  
  - Image2StyleGAN 대비  
    - 주 target 속성만 깔끔하게 변하고  
    - 나머지 속성(identity, 배경, 구조)이 잘 보존됨.[1]
- **Semantic diffusion**:  
  - target 얼굴을 다른 context 얼굴에 자연스럽게 “이식”하는 새로운 태스크 제안.  
  - encoder 기반 초기화 + 마스크된 optimization으로  
    - target identity를 유지하면서  
    - context의 스타일과 주변을 자연스럽게 융합.[1]

이러한 실험들은 “**in-domain inversion이야말로 ‘편집 가능한’ inversion**” 이라는 논문의 주장을 설득력 있게 뒷받침한다.

***

### 2.6 한계

논문과 후속 분석에서 확인되는 한계는 다음과 같다.[3][1]

1) **Domain 제한성** (진짜 의미의 OOD 일반화는 약함)  
   - face StyleGAN으로 cat face, bedroom 등을 invert하면  
     - IDInvert는 여전히 “사람 얼굴 도메인” 안에서 best effort를 내기 때문에  
       - 고양이나 침실이 아니라, **얼굴 비슷한 구조**로 재구성됨.[1]
   - 반대로 Image2StyleGAN은 pixel overfitting으로 bedroom을 “얼핏 비슷하게” 그리지만,  
     - 이 코드는 face semantics와 무관해 편집이 거의 불가능.  
   - 논문은 이것을 **“의도된 특성”** 으로 보는데,  
     - semantic editing을 목표로 한다면, OOD 이미지를 face 도메인으로 강제 mapping하는 것이 더 의미 있다고 주장.[3][1]
   - 하지만 **다양한 real 도메인을 한 번에 다루고자 할 때의 일반화** 는 여전히 미해결 과제.

2) **재구성–편집 trade-off**  
   - $\lambda_{\text{dom}}$를 늘리면 편집성은 좋아지지만, 재구성 MSE는 커지는 trade-off가 존재.[1]
   - downstream task에 따라 적정 값을 별도로 조율해야 하는 실용적인 부담이 있다.

3) **StyleGAN 의존성**  
   - generator prior에 강하게 의존하기 때문에  
     - StyleGAN이 잘 못 학습한 영역/속성에 대해서는 inversion/편집도 자연히 한계가 있다.  

***

3. “모델의 일반화 성능 향상 가능성” 관점에서의 해석  
***

논문이 말하는 “in-domain” 개념은 일반적인 **“데이터 분포 밖으로의 성능 일반화”** 라기보다는,  
> **GAN이 학습한 semantic manifold 위에서의 robust & editable representation 일반화**  
에 가깝다.

이를 몇 가지 관점에서 해석할 수 있다.

### 3.1 Latent manifold 상의 일반화

- Domain-guided encoder + domain-regularized optimization은  
  - $G$가 학습한 manifold $\mathcal{S}$ 에 대한 **근사 projection 연산**으로 작동한다.  
- 결과적으로, real image $x$가 약간 노이즈가 있거나 부분 가려짐, pose 변화가 있더라도  
  - inversion된 코드 $z_{\text{inv}}$는 항상 **“GAN이 표현 가능한 semantic 영역”** 안으로 떨어지게 되어  
  - 동일한 editing 방향 $n$ (예: “안경 추가”)이 **다양한 입력 이미지에 대해 일관되게 작동**한다.[1]

이는 **“편집 연산의 일반화”** 라는 측면에서 중요한 장점이다.  

### 3.2 Encoder-regularizer를 통한 안정성

- Optimization-based inversion만 사용하는 경우,  
  - local minimum에 빠지거나  
  - generator의 비선형 구조를 따라 이상한 방향으로 $z$가 나갈 수 있다.  
- 제안 방법은  
  - $z_0 = E(x)$ 로 시작함으로써 **좋은 초기화** 제공,  
  - $\|z - E(G(z))\|^2$ term으로 in-domain 제약을 걸어  
    - 최적화 경로 전반에서 $z$가 manifold 밖으로 과도하게 벗어나는 것을 막는다.[1]

이로 인해  
- **convergence의 안정성**,  
- **입력 변화에 대한 inversion 결과의 일관성** (robustness)이 향상되며,  
이는 넓은 의미의 **일반화 성능** 향상으로 볼 수 있다.

### 3.3 그러나, 데이터 분포 전체에 대한 일반화는?

- 앞서 본 것처럼, face GAN으로 bedroom을 invert하면,  
  - IDInvert는 과감하게 “bedroom을 포기”하고 “가장 가까운 얼굴”을 찾아간다.[1]
- 이는 “**semantic editing이라는 목표에 최적화된 일반화**” 이지,  
  - 이미지 복원/복원(fidelity) 관점에서의 일반화는 아니다.  

따라서 연구 관점에서 이 논문은  
- “**무조건 fidelity가 좋은 inversion이 좋은 것은 아니다**”  
- “**downstream task(semantic editing)에 맞는 일반화 개념을 다시 정의해야 한다**”  
는 중요한 문제 제기를 했다고 볼 수 있다.  

후속 연구들도 이 관점을 받아들여  
- reconstruction–editability trade-off,  
- in-domain vs out-of-domain inversion,  
- latent space 선택과 정규화  
를 핵심 연구 방향으로 다루고 있다.[4][5][6][7][8]

***

4. 2020년 이후 관련 최신 연구 동향 (GAN inversion & real image editing)  
***

2020년 이후 GAN inversion은 매우 활발히 연구되었고, IDInvert의 문제의식(“재구성 + 편집성 동시 확보”)을 계승·확장하는 흐름이 뚜렷하다. 아래는 대표적 방향과 논문들이다.

### 4.1 종합 서베이

- **GAN Inversion: A Survey (2021)**[7][4]
  - encoder 기반, optimization 기반, hybrid 기반 inversion 모두를 정리.  
  - **재구성–편집 trade-off** 와  
    - in-domain / out-of-domain 이슈,  
    - latent space 선택($\mathcal{Z}$, $\mathcal{W}$, $\mathcal{W}^+$, feature space 등)  
    를 핵심 논점으로 정리.  
  - IDInvert류 방법을 “semantic-aware / in-domain inversion” 카테고리로 소개하며, 이후 연구의 기준점이 됨.

### 4.2 Encoder 기반 고속 inversion

1) **pSp (pixel2style2pixel, 2020)**  
   - style-based encoder로 다양한 도메인(face, car, church 등)에 대해 high-fidelity inversion을 제공.  
   - 이후 많은 encoder 기반 방법의 baseline이 됨.[9][10]

2) **e4e (Encoding for Editing, 2021)**  
   - “encoder가 너무 reconstruction에 치우치면 편집성이 망가진다”는 점을 지적하고,  
   - explicit하게 **editability-friendly encoder**를 설계.[9]
   - IDInvert의 문제의식을 encoder 설계에 직접 반영한 예.  

3) **ReStyle (ICCV 2021)**[11][9]
   - single-shot encoder 대신, residual을 반복적으로 refine하는 **iterative encoder**.  
   - pSp, e4e 위에 ReStyle을 씌워  
     - 재구성과 편집성을 동시에 개선.  

4) **E2Style (2021)**[12]
   - 효율적인 backbone과 multi-stage refinement, multi-layer identity/face parsing loss로  
   - encoder 기반 inversion의 재구성 품질과 속도를 동시에 끌어올림.  

5) **High-fidelity style-based encoder (ECCV 2022)**[10]
   - StyleGAN2/ADA에 특화된 encoder로  
   - FFHQ, AFHQ 등에서 거의 “near perfect” 수준의 inversion과 높은 편집성을 동시에 달성.  

이들 모두 **“encoder 출력이 in-domain이면서도 재구성도 잘 되도록”** 하는 데 초점을 두며, IDInvert의 domain-guided 개념과 매우 유사한 철학을 갖는다.

### 4.3 Hybrid / fine-tuning 기반 inversion

1) **PTI (Pivotal Tuning Inversion, 2021)**[13][14]
   - 먼저 $x$를 $\mathcal{W}$ space로 invert한 pivot code $w_p$를 찾고,  
   - 그 주변(local region)에서 generator weight를 **미세 튜닝(pivotal tuning)**.  
   - 이렇게 하면  
     - pivot code는 여전히 잘 편집되면서도  
     - 해당 real image에 대한 재구성 fidelity는 크게 향상.  
   - IDInvert가 latent space에만 정규화를 두는 반면, PTI는 **generator parameter space에 local adaptation**을 도입하여 일반화를 확장하는 방식.

2) **BDInvert (2021)**[5]
   - out-of-range(기하학적으로 misaligned) real 이미지에 대한 inversion을 위한 방법.  
   - alternative latent space와 regularized inversion으로  
     - OOD geometry(예: 자세가 크게 다른 얼굴)에 대해서도 semantic editing을 가능하게 함.  
   - IDInvert가 주로 “도메인 안(in-range)인 이미지”에 초점을 둔 것과 달리,  
     - **기하학적 OOD generalization**을 직접 다루는 연구.

3) **MAGEC (2021)**[15]
   - optimization 기반 inversion에 **latent consistency loss** 를 도입,  
   - 재구성 fidelity와 semantic editability를 함께 높이려는 시도.  
   - IDInvert의 domain-regularizer와 개념적으로 비슷한 방향(semantic-aware regularization).  

4) **Near Perfect GAN Inversion (2022)**[16]
   - encoder/optimization의 한계를 넘는, **거의 완벽한 재구성**을 달성하는 알고리즘 제안.  
   - editability 유지 문제를 함께 논의하며,  
   - 재구성–편집 trade-off의 한계선을 어디까지 밀어붙일 수 있는지 보여줌.

### 4.4 Latent space 재설계와 robust inversion

1) **SalS-GAN (2021)**[17]
   - Affine 파라미터를 공간적으로 확장한 **spatially-adaptive latent space**를 도입해  
   - 세밀한 디테일까지 reconstruct하면서도 semantic disentanglement를 유지.  

2) **Spatially-Adaptive Multilayer Selection (CVPR 2022)**[18]
   - inversion 시 layer별로 latent/feature 공간을 선택적으로 사용하는 방식으로  
   - 기존 $\mathcal{W}^+$, PTI보다 좋은 재구성 및 편집 품질 보고.  

3) **Robust GAN inversion (2023)**[6]
   - 기존 $\mathcal{W}^+$ 기반 inversion이  
     - reconstruction–editability trade-off에서 한계를 보이는 점을 분석하고,  
   - native $\mathcal{W}$ 공간을 활용하는 새로운 inversion으로  
     - 낮은 distortion과 높은 editability를 동시에 달성.  

4) **Revisiting Latent Space of GAN Inversion (WACV 2024)**[8]
   - 기존 $\mathcal{Z}$ space를 확장한 $\mathcal{Z}^+$ 등 새로운 latent space를 제안,  
   - reconstruction과 editing을 둘 다 만족하는 **F/Z+** 공간을 설계.  

이 흐름은 IDInvert가 던진  
> “어떤 latent 공간에서, 어떤 정규화를 걸어야 in-domain & editable code를 얻을 수 있는가?”  
라는 질문에 다양한 답을 제시하고 있는 셈이다.

### 4.5 Out-of-domain editing, inpainting, 고차원 편집

- **Editing Out-of-domain GAN Inversion via Differential Activation (ECCV 2022)**[19]
  - composition–decomposition paradigm으로 OOD 편집 시 ghosting을 제거하고  
  - differential activation module로 semantic change를 정확히 추출.  
- **Diverse Inpainting and Editing with GAN Inversion (2023)**[20]
  - 지워진 영역(inpainting)까지 포함하는 더 어려운 inversion 문제에서  
  - 다수의 plausible 결과와 편집을 가능하게 하는 방법 제안.  
- **Gradual Residuals Alignment (2024)**[21]
  - dual-stream 구조로 residual alignment를 점진적으로 수행해  
  - inversion과 attribute editing을 통합적으로 개선.  

이들도 모두 “GAN이 학습한 prior를 안전하게 reuse하면서, 다양한 real 이미지에 대해 일반화된 편집”을 목표로 한다.

***

5. 앞으로의 연구에 미치는 영향과 연구 시 고려할 점  
***

### 5.1 이 논문의 영향

1) **“편집 가능성(editability)”을 inversion 평가의 1급 시민으로 격상**  
   - 이전에는 inversion 품질을 거의 **MSE / LPIPS / FID**로만 평가하는 경향이 강했다.  
   - 이 논문은  
     - semantic classifier(PR curve),  
     - latent manipulation 결과,  
     - interpolation·semantic diffusion 같은 downstream task를 통해  
     **“inverted code의 semantic 의미 보존 능력”**을 체계적으로 평가할 필요성을 제기했다.[1]
   - 이후 서베이 및 많은 논문에서 reconstruction–editability 둘 다를 표준 지표로 쓰게 된 데 큰 영향을 미침.[7]

2) **“In-domain”이라는 개념의 보편화**  
   - encoder 출력/최적화 결과가 generator의 statistical prior 안에 있도록 정규화하는 아이디어는  
     - encoder 설계(e4e, ReStyle, E2Style, High-fidelity encoder 등),[10][9]
     - latent regularization(MAGEC, Robust GAN inversion),[15][6]
     - latent space 재설계(F/Z+ 등)[8]
     등으로 확장되었다.  

3) **Domain-regularized optimization이라는 hybrid 패턴 확립**  
   - encoder initialization + regularized optimization 구조는  
     - 이후 hybrid inversion, PTI류 접근,[14][13]
     - video/연속 이미지 기반 inversion[22][23]
     등에서 일반적인 패턴으로 자리잡았다.  

### 5.2 앞으로 연구 시 고려할 점

연구자로서 이 논문을 기반으로 후속 연구를 진행할 때 고려할 핵심 포인트는 다음과 같다.

1) **목표 task에 맞는 “일반화” 정의하기**  
   - real image editing인지,  
   - pure reconstruction인지,  
   - inpainting/segmentation/3D consistent editing인지에 따라  
     - in-domain 제약의 강도,  
     - latent space 선택($\mathcal{W}$, $\mathcal{W}^+$, feature space, F/Z+, SA space 등),  
     - regularization 형태가 달라져야 한다.[7][8]

2) **재구성–편집 trade-off를 명시적으로 설계**  
   - IDInvert에서 $\lambda_{\text{dom}}$를 조절하듯이,[1]
   - loss 설계 시  
     - reconstruction term과  
     - semantic/domain regularization term  
     의 균형을 **이론적으로/실험적으로 분석**하는 것이 중요하다.  
   - encoder 기반 연구에서는  
     - identity/attribute 유지 loss,  
     - latent smoothness,  
     - adversarial loss  
     등을 어떻게 조합하는지가 핵심.[12][6][9][10]

3) **Out-of-domain generalization 전략**  
   - 한 도메인(face)에서만 잘 작동하는 알고리즘과  
   - 여러 도메인(동물, 실내, 풍경 등)에 동시에 일반화되는 알고리즘은 설계 철학이 다를 수밖에 없다.  
   - BDInvert, OOD editing, inpainting 연구처럼  
     - composition–decomposition,  
     - alternative latent space,  
     - local generator fine-tuning(PTI)  
     를 적절히 결합해,  
     **“semantic editing이 가능한 범위”를 어떻게 넓힐지** 고민해야 한다.[5][20][19][14]

4) **평가 프로토콜 정교화**  
   - 서베이와 IDInvert에서 보듯,[7][1]
     - attribute classifier 기반 semantic alignment,  
     - latent 방향 조작 후 attribute consistency,  
     - interpolation smoothness,  
     - downstream task 성능(예: face recognition robustness, gaze tracking 개선 등)[24]
     을 포함한 종합적인 평가가 필요하다.  

5) **GAN → Diffusion 시대에도 남는 아이디어**  
   - 최근 diffusion 모델 기반 inversion·editing이 활발하지만,  
   - “pretrained generative prior를 그대로 두고, inversion에서 semantic을 보존하도록 정규화한다”는 이 논문의 핵심 아이디어는  
     - diffusion latent space inversion,  
     - score distillation,  
     - text-guided editing  
     등에서도 그대로 응용 가능하다.  
   - 즉, **“in-domain inversion + semantic-aware regularization”** 이라는 추상적 개념은 앞으로도 유효한 연구 방향이다.

***

요약하면,  
> 이 논문은 GAN inversion을 “픽셀 복원 문제”에서 “semantic-preserving in-domain representation learning 문제”로 재정의했고,  
> 이를 구현하는 구체적 방법(domain-guided encoder + domain-regularized optimization)을 제안함으로써,  
> 이후 수년간의 GAN inversion/real image editing 연구의 기준점을 마련했다.

앞으로 관련 연구를 진행할 때는  
- **어떤 latent 도메인을 대상으로**,  
- **어떤 의미 공간(semantics)을 보존·조작하고 싶은지**,  
- **어디까지를 in-domain으로 받아들일 것인지**  
를 먼저 명확히 정의한 뒤, IDInvert와 이후 방법들(e4e, ReStyle, PTI, Robust inversion, F/Z+ 등)의 설계 선택을 비교·조합하는 방향이 바람직하다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/32e33993-63e6-4984-92a5-0bad79c74eaf/2004.00049v3.pdf)
[2](https://arxiv.org/pdf/1611.06355.pdf)
[3](https://everyday-deeplearning.tistory.com/entry/%EC%B4%88-%EA%B0%84%EB%8B%A8-%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-In-domain-GAN-Inversion-for-Real-Image-Editing)
[4](https://ieeexplore.ieee.org/document/9792208/)
[5](https://ieeexplore.ieee.org/document/9711252/)
[6](https://arxiv.org/abs/2308.16510)
[7](https://arxiv.org/pdf/2101.05278.pdf)
[8](https://openaccess.thecvf.com/content/WACV2024/papers/Katsumata_Revisiting_Latent_Space_of_GAN_Inversion_for_Robust_Real_Image_WACV_2024_paper.pdf)
[9](https://openaccess.thecvf.com/content/ICCV2021/papers/Alaluf_ReStyle_A_Residual-Based_StyleGAN_Encoder_via_Iterative_Refinement_ICCV_2021_paper.pdf)
[10](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136750579.pdf)
[11](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/restyle/)
[12](https://ieeexplore.ieee.org/document/9760266/)
[13](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/pti/)
[14](https://arxiv.org/abs/2106.05744)
[15](https://www.bmva.org/bmvc/2021/conference/papers/paper_1394.html)
[16](https://arxiv.org/abs/2202.11833)
[17](https://dl.acm.org/doi/10.1145/3474085.3475633)
[18](https://openaccess.thecvf.com/content/CVPR2022/papers/Parmar_Spatially-Adaptive_Multilayer_Selection_for_GAN_Inversion_and_Editing_CVPR_2022_paper.pdf)
[19](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770001.pdf)
[20](https://arxiv.org/pdf/2307.15033.pdf)
[21](https://arxiv.org/html/2402.14398v1)
[22](https://ieeexplore.ieee.org/document/9710678/)
[23](http://arxiv.org/pdf/2107.13812.pdf)
[24](https://ieeexplore.ieee.org/document/9755402/)
[25](https://www.semanticscholar.org/paper/984032793f179d8fbaa4954862417aa828acb6da)
[26](https://www.semanticscholar.org/paper/d831e9bdbbe3f321750cc2b99a55d1b5d3f25fa1)
[27](https://ieeexplore.ieee.org/document/9506059/)
[28](https://arxiv.org/pdf/2304.14403.pdf)
[29](https://arxiv.org/html/2309.13956)
