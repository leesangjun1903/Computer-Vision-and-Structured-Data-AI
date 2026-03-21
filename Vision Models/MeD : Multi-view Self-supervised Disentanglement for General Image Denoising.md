# Multi-view Self-supervised Disentanglement for General Image Denoising

---

# 1. 핵심 주장 및 주요 기여 (요약)

기존 딥러닝 기반 디노이징 방법들은 학습에 사용된(seen) 노이즈 분포에서는 좋은 성능을 보이지만, 학습하지 않은(unseen) 노이즈 유형이나 실제(real) 노이즈에 대한 일반화(generalisation)에 취약합니다. 이 논문은 동일한 클린 이미지의 서로 다른 손상(corrupted) 버전들이 공통 잠재 공간(common latent space)을 공유한다는 직관적 가정 하에, 노이즈 이미지를 분리(disentangle)하는 방법을 학습하도록 제안합니다.

클린 이미지를 전혀 보지 않고도(self-supervised) 목표를 달성하는 학습 프레임워크를 제안합니다. 두 가지 서로 다른 손상 버전의 이미지를 입력으로 받아, **MeD(Multi-view Self-supervised Disentanglement)** 접근법이 잠재 클린 특징(latent clean features)을 손상(corruptions)으로부터 분리해 내고, 결과적으로 클린 이미지를 복원합니다.

**주요 기여:**
1. 다중 뷰 자기지도 분리(disentanglement) 프레임워크 제안
2. 클린 이미지 없이도 잠재 표현 학습 가능
3. 합성 노이즈와 실제 노이즈 모두에서 기존 자기지도 접근법보다 우수한 성능을 보이며, 특히 학습하지 않은 새로운 노이즈 유형에서 뛰어나고, 실제 노이즈에서는 지도학습(supervised) 방법을 3 dB 이상 능가합니다.

---

# 2. 상세 분석

## 2.1 해결하고자 하는 문제

딥러닝 패러다임이 현대 이미지 디노이저의 표준이 되었지만, 기존 접근법들은 학습에 사용된 노이즈 분포에서만 좋은 성능을 보이며, 학습하지 않은 노이즈 유형이나 실제 노이즈로의 일반화에 어려움을 겪습니다. 이는 모델이 노이즈 이미지에서 클린 이미지로의 쌍(paired) 매핑을 학습하도록 설계되어 있기 때문입니다.

핵심적으로 두 가지 문제를 해결합니다:
- **일반화 부족**: 특정 노이즈 분포(예: Gaussian $\sigma=25$)로 학습한 모델이 다른 유형(Salt & Pepper, Poisson 등)에서 성능 급감
- **클린 이미지 의존성**: 지도학습 방법은 노이즈-클린 쌍이 필요하나, 실제 환경에서 확보가 어려움

## 2.2 제안하는 방법 (수식 포함)

### 기본 가정 및 모델링

노이즈 이미지 $\mathbf{y}$를 다음과 같이 모델링합니다:

$$\mathbf{y} = \mathbf{x} + \mathbf{n}$$

여기서 $\mathbf{x}$는 잠재 클린 이미지, $\mathbf{n}$은 노이즈입니다.

동일한 클린 이미지의 서로 다른 손상 버전들이 공통 잠재 공간을 공유한다는 가정 하에, 두 개의 서로 다른 손상 버전을 입력으로 받아 잠재 클린 특징을 노이즈로부터 분리합니다.

두 개의 노이즈 뷰(view)를 다음과 같이 정의합니다:

$$\mathbf{y}_1 = \mathbf{x} + \mathbf{n}_1, \quad \mathbf{y}_2 = \mathbf{x} + \mathbf{n}_2$$

여기서 $\mathbf{n}_1$과 $\mathbf{n}_2$는 독립적인 노이즈 실현(realization)입니다.

### 분리(Disentanglement) 프레임워크

MeD는 인코더-디코더 구조를 사용하여 각 노이즈 뷰에서 **장면(scene) 특징** $\mathbf{z}_s$와 **손상(corruption) 특징** $\mathbf{z}_c$를 분리합니다:

$$\mathbf{z}_{s,k} = E_s(\mathbf{y}_k), \quad \mathbf{z}_{c,k} = E_c(\mathbf{y}_k), \quad k \in \{1, 2\}$$

여기서 $E_s$는 장면 인코더(scene encoder), $E_c$는 손상 인코더(corruption encoder)입니다.

핵심 가정은 **장면 특징이 뷰 간에 공유**된다는 것입니다:

$$\mathbf{z}_{s,1} \approx \mathbf{z}_{s,2}$$

### Mix 연산자

논문에서는 Mix 연산자를 다음과 같이 정의합니다:

$$\text{Mix}_p(\mathbf{m}, \mathbf{n}) \triangleq \mathbf{b}_f \odot \mathbf{m} + (1 - \mathbf{b}_f) \odot \mathbf{n}$$

여기서 $\mathbf{b}_f$는 이진 마스크, $\odot$는 원소별 곱(element-wise product)입니다. 이 연산자를 통해 두 뷰의 특징을 혼합하면서 자기지도 학습 신호를 생성합니다.

### 손실 함수(Loss Function)

MeD의 전체 손실 함수는 여러 구성 요소로 이루어집니다:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda_1 \mathcal{L}_{\text{consist}} + \lambda_2 \mathcal{L}_{\text{indep}}$$

- **재구성 손실(Reconstruction Loss)** $\mathcal{L}_{\text{recon}}$: 분리된 장면 특징과 손상 특징을 재결합했을 때 원래 노이즈 이미지를 복원할 수 있는지 확인

$$\mathcal{L}_{\text{recon}} = \sum_{k} \| D(\mathbf{z}_{s,k}, \mathbf{z}_{c,k}) - \mathbf{y}_k \|^2$$

- **일관성 손실(Consistency Loss)** $\mathcal{L}_{\text{consist}}$: 서로 다른 뷰에서 추출한 장면 특징이 동일해야 함

$$\mathcal{L}_{\text{consist}} = \| \mathbf{z}_{s,1} - \mathbf{z}_{s,2} \|^2$$

- **독립성 손실(Independence Loss)** $\mathcal{L}_{\text{indep}}$: 장면 특징과 손상 특징 사이의 상호 정보를 최소화하여 분리를 보장

## 2.3 모델 구조

MeD의 아키텍처는 다음 구성 요소로 이루어집니다:

| 구성 요소 | 역할 |
|---|---|
| **Scene Encoder** ($E_s$) | 클린 장면의 잠재 특징 추출 |
| **Corruption Encoder** ($E_c$) | 노이즈/손상 관련 특징 추출 |
| **Decoder** ($D$) | 장면 + 손상 특징으로부터 이미지 재구성 |

백본(backbone) 네트워크로 **Swin Transformer (Swin-Tx)**를 사용하며, 기존 LIR 방법의 U-Net 백본을 Swin-Tx로 교체하여 평가의 일관성을 유지합니다. 이 결과 Gaussian 디노이징에서 약 1 dB의 평균 PSNR 향상을 달성하였으며, 모든 실험에서 DIV2K 데이터셋만을 사용하여 학습합니다.

추론(inference) 시에는 장면 인코더와 디코더만 사용하여 클린 이미지를 복원합니다:

$$\hat{\mathbf{x}} = D(E_s(\mathbf{y}), \mathbf{0})$$

## 2.4 성능 향상

| 실험 항목 | 결과 |
|---|---|
| 합성 Gaussian 노이즈 | 기존 자기지도 방법 대비 우수 |
| **미학습 노이즈 유형** | 기존 자기지도 방법 대비 현저한 우월성 |
| **실제 노이즈** | 지도학습 방법 대비 **3 dB 이상** 능가 |
| **사전학습(Pre-training)** | MeD를 사전학습 모델로 사용 시 N2C, N2N, LIR에 대해 최대 2 dB 향상, 평균 약 0.5 dB의 성능 이득 |

## 2.5 한계

논문 및 관련 분석에서 도출되는 한계점:

1. **다중 뷰 필요성**: 학습 시 동일 장면의 두 개 이상 노이즈 버전이 필요하며, 실제 환경에서는 항상 확보하기 어려울 수 있음
2. **공간적으로 상관된 노이즈**: 많은 자기지도 방법들이 공간적으로 독립이거나 분석적(analytical)인 노이즈를 가정하는데, 이는 실제 세계의 구조적이고 복잡한 노이즈와 괴리가 있습니다.
3. **계산 비용**: Swin Transformer 기반 구조로 인해 경량 모델 대비 추론 시간이 증가할 수 있음
4. 이러한 disentanglement 방법들은 종종 대량의 노이즈 이미지를 필요로 하며 데이터 효율성이 부족할 수 있습니다.

---

# 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

MeD의 핵심 강점이자 가장 큰 혁신은 **일반화 성능**에 있습니다.

## 3.1 일반화 메커니즘

기존 방법이 $f: \mathbf{y} \rightarrow \mathbf{x}$ (특정 노이즈 분포에 대한 직접 매핑)을 학습하는 반면, MeD는:

$$\text{Disentangle}: \mathbf{y} \rightarrow (\mathbf{z}_s, \mathbf{z}_c)$$

를 학습합니다. 이 분리 학습의 핵심 이점은:

- **노이즈 유형에 불가지론적(agnostic)**: 장면 특징 $\mathbf{z}_s$는 노이즈 유형과 무관하게 클린 신호만 포착하도록 학습
- **전이 가능성**: 특정 노이즈로 학습한 모델의 장면 인코더가 다른 노이즈 유형에도 적용 가능

## 3.2 사전학습(Pre-training)으로서의 MeD

MeD 사전학습 모델을 사용한 접근법들(N2C, N2N, LIR)이 자체 전이(self-transfer) 모델을 최대 2 dB까지 능가하며, 평균적으로 약 0.5 dB의 성능 이득을 보여, MeD가 강력한 사전학습 전략이 될 수 있음을 입증합니다.

이는 MeD가 학습한 장면 표현이:

$$\mathbf{z}_s = E_s(\mathbf{y}) \in \mathcal{Z}_s$$

노이즈에 독립적인 **범용적 클린 이미지 표현**임을 시사합니다.

## 3.3 미학습 노이즈에 대한 실험적 증거

Gaussian $\sigma = 25$로만 학습된 모든 방법에 대해 미학습 노이즈 유형에서의 정성적 디노이징 결과를 비교한 결과, MeD가 일관되게 우수한 성능을 보입니다.

기존 방법들과 달리, MeD는 다중 정적 관측을 동시에 사용하여 여러 개별 뷰가 공유하는 클린 장면의 잠재 표현을 학습합니다.

## 3.4 일반화 향상을 위한 향후 방향

1. **더 많은 뷰 활용**: 논문은 더 많은 뷰를 통합하여 성능에 미치는 영향을 연구합니다. 뷰 수 $K$가 증가하면:

$$\mathcal{L}_{\text{consist}} = \frac{1}{\binom{K}{2}} \sum_{i < j} \| \mathbf{z}_{s,i} - \mathbf{z}_{s,j} \|^2$$

으로 확장되어 더 견고한 장면 표현 학습이 가능

2. **다른 복원 태스크로의 확장**: 논문은 이미지 초해상도(super-resolution)와 인페인팅(inpainting) 등 다른 태스크에도 적용하여 범용성을 입증합니다.

---

# 4. 연구 영향 및 향후 고려사항

## 4.1 연구에 미치는 영향

1. **패러다임 전환**: 직접 매핑( $\mathbf{y} \rightarrow \mathbf{x}$) 대신 분리(disentanglement, $\mathbf{y} \rightarrow (\mathbf{z}_s, \mathbf{z}_c)$ )라는 새로운 방향을 제시하여, 노이즈 유형에 불가지론적인 디노이징의 가능성을 열었습니다.

2. **사전학습 전략**: MeD의 장면 인코더를 사전학습 모델로 활용하는 것이 하류 디노이징 태스크의 성능을 향상시킬 수 있다는 점은 foundation model 패러다임과 유사한 방향성을 제시합니다.

3. **자기지도 학습의 새로운 벤치마크**: 클린 이미지 없이도 지도학습에 필적하거나 능가하는 성능을 달성함으로써, 데이터 제약 환경(의료, 천문 등)에서의 적용 가능성을 크게 확장했습니다.

4. 후속 연구들이 disentangled representation learning을 이미지 디노이징에 적극 활용하는 방향으로 발전하고 있습니다.

## 4.2 향후 연구 시 고려할 점

| 고려사항 | 설명 |
|---|---|
| **단일 이미지 적용** | 다중 뷰 없이 단일 노이즈 이미지만으로 분리 학습을 수행할 수 있는 방법 연구 필요 |
| **공간 상관 노이즈** | 실제 카메라 노이즈의 공간적 상관성을 고려한 분리 메커니즘 개발 |
| **경량화** | Swin Transformer 기반 구조의 계산 비용 절감을 위한 지식 증류(knowledge distillation) 등 |
| **비정상(non-stationary) 노이즈** | 이미지 영역에 따라 노이즈 특성이 변하는 경우의 적응적 분리 |
| **다른 도메인 확장** | 3D 의료영상, 비디오 디노이징, 위성영상 등으로의 적용 |

---

# 5. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 연도 | 학회 | 핵심 접근법 | MeD와의 차이 |
|---|---|---|---|---|
| **Noise2Void / BSN** | 2019–2021 | CVPR | Blind-spot 네트워크로 자기지도 학습 | 단일 뷰, 공간 독립 노이즈 가정; 일반화 제한 |
| **R2R (Recorrupted-to-Recorrupted)** | 2021 | CVPR | 재손상(recorrupted) 이미지를 이용한 비지도 딥러닝 디노이징 | 단일 이미지 기반이나, 노이즈 모델 가정 필요 |
| **CVF-SID** | 2022 | CVPR | Cyclic multi-Variate Function (CVF) 모듈과 자기지도 이미지 분리(SID) 프레임워크 기반으로, 입력을 여러 분해 변수로 출력하고 이를 순환적으로 입력으로 사용 | 단일 이미지, signal-dependent/independent 노이즈 분리; MeD는 다중 뷰 활용 |
| **AP-BSN / LG-BPN** | 2022–2023 | CVPR | Pixel-shuffle와 masked convolution을 BS 네트워크에 통합하여 실제 노이즈에 대응 | 구조적 제약이 큼; MeD는 더 유연한 분리 기반 |
| **MeD (본 논문)** | 2023 | ICCV | 다중 뷰 자기지도 분리 | 뷰 간 공유 잠재 공간 학습으로 일반화 극대화 |
| **TBSN** | 2024 | AAAI | Transformer 기반 Blind-Spot Network; dilated BSN 원리를 따르며 공간 및 채널 self-attention 레이어를 통합 | BSN 프레임워크의 한계 내에서의 개선; 분리 기반 아님 |
| **Diffusion Priors for Denoising** | 2024 | arXiv | 실제 촬영 및 생의학 이미지에서 발생하는 신호 의존적이고 공간 상관된 복잡한 노이즈를 확산(diffusion) 사전을 활용하여 제거 | 확산 모델 기반; 데이터 효율성 높으나 다른 패러다임 |

### 핵심 비교 수식

| 방법 | 학습 목표 |
|---|---|
| N2C (Noise2Clean) | $\min_\theta \mathbb{E}\[\|\hat{f}_\theta(\mathbf{y}) - \mathbf{x}\|^2\]$ |
| N2N (Noise2Noise) | $\min_\theta \mathbb{E}\[\|\hat{f}_\theta(\mathbf{y}_1) - \mathbf{y}_2\|^2\]$ |
| BSN (Blind-Spot) | $\min_\theta \mathbb{E}\[\|\hat{f}\_\theta(\mathbf{y}_{\setminus i}) - y_i\|^2\]$ |
| **MeD** | $\min_{\theta_s, \theta_c, \theta_d} \mathcal{L}\_{\text{recon}} + \lambda_1 \mathcal{L}\_{\text{consist}} + \lambda_2 \mathcal{L}_{\text{indep}}$ |

MeD는 직접적 매핑이 아닌 **잠재 공간에서의 분리**를 통해 노이즈 유형에 대한 불변성(invariance)을 구조적으로 확보하는 점에서 근본적으로 다릅니다.

---

## 참고 자료 및 출처

1. **Chen, H., Qu, C., Zhang, Y., Chen, C., & Jiao, J.** (2023). "Multi-view Self-supervised Disentanglement for General Image Denoising." *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 12281–12291.
   - OpenAccess PDF: https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Multi-view_Self-supervised_Disentanglement_for_General_Image_Denoising_ICCV_2023_paper.pdf
   - arXiv: https://arxiv.org/abs/2309.05049
   - 프로젝트 페이지: https://chqwer2.github.io/MeD/
   - GitHub: https://github.com/chqwer2/Multi-view-Self-supervised-Disentanglement-Denoising
   - IEEE Xplore: https://ieeexplore.ieee.org/document/10377551/
   - University of Birmingham: https://research.birmingham.ac.uk/en/publications/multi-view-self-supervised-disentanglement-for-general-image-deno

2. **Neshatavar, R., Yavartanoo, M., Son, S., & Lee, K. M.** (2022). "CVF-SID: Cyclic Multi-Variate Function for Self-Supervised Image Denoising by Disentangling Noise from Image." *CVPR 2022*, pp. 17583–17591.
   - arXiv: https://arxiv.org/abs/2203.13009
   - OpenAccess: https://openaccess.thecvf.com/content/CVPR2022/html/Neshatavar_CVF-SID_Cyclic_Multi-Variate_Function_for_Self-Supervised_Image_Denoising_by_Disentangling_CVPR_2022_paper.html

3. **TBSN** (2024). "Transformer-Based Blind-Spot Network for Self-Supervised Image Denoising."
   - arXiv: https://arxiv.org/html/2404.07846v2

4. **Diffusion Priors for Variational Likelihood Estimation and Image Denoising** (2024).
   - arXiv: https://arxiv.org/html/2410.17521v1

5. **Semantic Scholar**: https://www.semanticscholar.org/paper/186f7e71f2a878c42f1f19e712ab7786345a9614

> **주의**: 위 분석에서 손실 함수의 구체적인 수식 형태(특히 $\mathcal{L}_{\text{indep}}$의 정확한 수학적 형태)는 논문의 전체 PDF를 정밀하게 참조한 일반적인 형태를 기반으로 기술하였습니다. 정확한 계수, 하이퍼파라미터 값 등은 원 논문의 본문 및 보충 자료를 직접 참조하시기를 권장합니다.
