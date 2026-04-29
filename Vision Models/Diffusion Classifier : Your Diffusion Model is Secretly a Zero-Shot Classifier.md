
# Your Diffusion Model is Secretly a Zero-Shot Classifier

> **논문 정보**
> - **저자**: Alexander C. Li, Mihir Prabhudesai, Shivam Duggal, Ellis Brown, Deepak Pathak (Carnegie Mellon University)
> - **학술대회**: ICCV 2023 (pp. 2206–2217)
> - **arXiv**: [arXiv:2303.16203](https://arxiv.org/abs/2303.16203)
> - **프로젝트 페이지**: [diffusion-classifier.github.io](https://diffusion-classifier.github.io)
> - **공식 코드**: [github.com/diffusion-classifier/diffusion-classifier](https://github.com/diffusion-classifier/diffusion-classifier)

---

## 1. 핵심 주장 및 주요 기여 요약

대규모 텍스트-이미지 확산 모델(Diffusion Model)은 방대한 종류의 프롬프트에 대해 사실적인 이미지를 생성할 수 있으며, 인상적인 구성적 일반화 능력을 갖추고 있습니다. 그러나 지금까지 거의 모든 활용 사례는 단순 샘플링(sampling)에만 집중되어 있었고, 이미지 생성 이외의 작업에 유용한 **조건부 밀도 추정(conditional density estimation)** 능력은 활용되지 않았습니다.

이 논문의 핵심 주장은 다음과 같습니다:

**대규모 텍스트-이미지 확산 모델(예: Stable Diffusion)의 밀도 추정값을 활용하면, 추가 학습 없이도 제로샷(zero-shot) 분류를 수행할 수 있다**는 것입니다.

**주요 기여:**

1. **Diffusion Classifier**: 텍스트-이미지 모델(Stable Diffusion)에서 제로샷 분류기를, 클래스 조건부 모델(DiT)에서 표준 분류기를 추가 학습 없이 추출하는 방법론을 제안합니다.

2. 제로샷 인식 과제에서 생성적 접근법과 판별적 접근법 사이에 여전히 격차가 존재하지만, **확산 기반 접근법은 경쟁하는 판별적 접근법보다 훨씬 강력한 멀티모달 구성적 추론 능력**을 보여줍니다.

3. ImageNet으로 학습된 클래스 조건부 확산 모델에서 표준 분류기를 추출하며, 이 모델은 **약한 데이터 증강만으로도 강력한 분류 성능**을 달성하고, 분포 변이(distribution shift)에 대해 질적으로 더 나은 "유효 강건성(effective robustness)"을 보입니다.

4. **전체적인 결과는 하위 과제에서 판별적 모델 대신 생성적 모델을 활용하는 방향으로의 한 걸음**을 의미합니다.

---

## 2. 해결하고자 하는 문제, 제안 방법(수식), 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

NLP 분야에서 생성적 사전 학습(generative pre-training)은 널리 활용되고 있으나, 시각적 파운데이션 모델은 대조 학습(contrastive learning)과 같은 다른 방법론을 주로 사용합니다. 이 논문은 **생성적 사전 학습이 비전-언어 과제에서 설득력 있는 대안이 될 수 있음을 입증**하고자 합니다.

기존의 확산 모델 기반 분류 시도는 (1) 합성 데이터로 별도 분류기를 학습하거나 (2) 확산 모델의 특징(feature)을 추출하는 방식이었는데, **Diffusion Classifier는 다양한 벤치마크에서 강력한 결과를 달성하고, 확산 모델로부터 지식을 추출하는 대안 방법들을 능가합니다.**

---

### 2-2. 제안하는 방법 (수식 포함)

#### ① 베이즈 정리 기반 분류

조건부 생성 모델을 이용한 분류는 레이블 $\{\mathbf{c}_i\}$에 대한 균등 사전 분포(uniform prior) $p(\mathbf{c}_i)$와 베이즈 정리를 통해 다음과 같이 수행됩니다:

$$p_\theta(\mathbf{c}_i \mid \mathbf{x}) = \frac{p(\mathbf{c}_i)\, p_\theta(\mathbf{x} \mid \mathbf{c}_i)}{\sum_j p(\mathbf{c}_j)\, p_\theta(\mathbf{x} \mid \mathbf{c}_j)}$$

균등 사전 분포($p(\mathbf{c}_i) = \frac{1}{N}$)를 가정하면 $p(\mathbf{c})$ 항이 모두 소거됩니다.

#### ② ELBO를 통한 조건부 로그 우도 근사

확산 모델에서는 $\log p_\theta(\mathbf{x} \mid \mathbf{c})$의 직접 계산이 불가능(intractable)하므로, 이를 **ELBO(Evidence Lower BOund)**로 근사합니다:

$$\log p_\theta(\mathbf{x} \mid \mathbf{c}) \geq \mathbb{E}_{t, \boldsymbol{\epsilon}} \left[ -\| \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \mathbf{c}) - \boldsymbol{\epsilon} \|^2 \right] + \text{const.}$$

여기서 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x} + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}$이고, $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$, $t \sim \text{Uniform}[1, T]$입니다.

#### ③ 몬테카를로 추정 및 분류 규칙

**Diffusion Classifier의 개요**: 입력 이미지 $\mathbf{x}$와 가능한 조건 입력들(텍스트 또는 클래스 인덱스) 집합이 주어졌을 때, 확산 모델을 사용하여 이미지에 가장 잘 맞는 조건을 선택합니다. 이 방법은 확산 모델의 변분적(variational) 관점을 통해 이론적으로 동기화되며, ELBO를 사용해 $\log p_\theta(\mathbf{x} \mid \mathbf{c})$를 근사합니다. Diffusion Classifier는 **입력 이미지에 추가된 노이즈를 가장 잘 예측하는 조건 $\mathbf{c}$를 선택**합니다.

이를 몬테카를로 추정으로 표현하면:

$$\hat{c} = \arg\min_{c} \; \mathbb{E}_{t, \boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, c) - \boldsymbol{\epsilon}\|^2\right]$$

#### ④ 효율적인 Diffusion Classifier 알고리즘

기본 알고리즘은 모든 클래스에 대해 평가를 수행하므로 계산 비용이 높습니다. 이를 해결하기 위해 **적응적으로 어떤 클래스를 계속 평가할지를 선택하는 효율적인 Diffusion Classifier 알고리즘(Algorithm 2)**을 제안합니다.

가능성이 낮은 클래스들을 약한 판별 분류기(예: CLIP ResNet-50)로 **가지치기(pruning)**하면 정확도를 높이고 추론 시간을 줄일 수 있습니다.

---

### 2-3. 모델 구조

#### 제로샷 분류: Stable Diffusion 기반

Diffusion Classifier는 LAION-5B의 필터링된 서브셋으로 학습된 텍스트-이미지 잠재 확산 모델(latent diffusion model)인 **Stable Diffusion** 위에 구축됩니다.

- 텍스트 인코더로 CLIP/OpenCLIP 인코더 활용
- 노이즈 $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$ 샘플링 후 $t \in [1, T]$ 샘플링
- 각 클래스 텍스트 프롬프트 $c$에 대해 노이즈 예측 오차를 계산하여 최소 오차 클래스를 예측

#### 표준 분류기 추출: DiT (Diffusion Transformer) 기반

Diffusion Classifier를 사용하여 사전 학습된 DiT(Diffusion Transformer) 모델로부터 ImageNet에 대한 표준 **1000-way 분류기**를 획득합니다.

DiT는 ImageNet-1k에서만 학습된 클래스 조건부 확산 모델로, 무작위 수평 뒤집기만 사용하며 정규화가 없습니다.

---

### 2-4. 성능 향상

**제로샷 분류**에서 Diffusion Classifier의 제로샷 분류 방법은 CLIP과 경쟁력 있으며, 합성 Stable Diffusion 데이터로 분류기를 학습하는 제로샷 확산 모델 기준선을 **크게 능가**합니다. 또한 특히 ImageNet과 같은 복잡한 데이터셋에서 Stable Diffusion 특징 기반 기준선을 일반적으로 능가합니다.

표준 분류 설정에서 Diffusion Classifier는 **ImageNet에서 79.1% top-1 정확도**를 달성하며, 이는 ResNet-101 및 ViT-L/32보다 우수한 성능입니다.

무엇보다 **이 접근법은 ImageNet 정확도를 고도로 경쟁력 있는 판별적 분류기와 비교할 수 있는 수준으로 달성한 최초의 생성적 모델링 접근법**으로, 판별 모델들이 세밀하게 조정된 학습률 스케줄, 증강 전략 및 정규화를 사용해 학습되었음을 감안할 때 특히 인상적입니다.

**Winoground 벤치마크 (구성적 추론):**
Diffusion Classifier는 CLIP 및 OpenCLIP 두 대조 학습 기준선을 **크게 능가**합니다. Stable Diffusion이 OpenCLIP ViT-H/14와 동일한 텍스트 인코더를 사용함을 고려하면, 이 향상은 **개념과 이미지 간의 더 나은 교차 모달 바인딩(cross-modal binding)**에서 비롯된 것입니다.

---

### 2-5. 한계

**① 높은 추론 비용:**
단일 이미지를 분류하는 데 데이터셋에 따라 **18초(Pets)에서 1000초(ImageNet)**까지 소요됩니다.

**② 프롬프트 튜닝 부재:**
논문에서는 수동 프롬프트 튜닝을 전혀 수행하지 않고 단순히 CLIP 저자들이 사용한 프롬프트를 그대로 사용했습니다. Stable Diffusion 학습 분포에 맞게 프롬프트를 조정하면 인식 능력이 향상될 것으로 예상됩니다.

**③ 학습 데이터 분포의 제한:**
Stable Diffusion 분류 정확도는 더 넓은 학습 분포로 개선될 수 있을 것으로 예상합니다. Stable Diffusion은 저해상도, 잠재적 NSFW 또는 미적으로 좋지 않은 이미지를 제거하기 위해 **공격적으로 필터링된 LAION-5B의 서브셋**으로 학습되었습니다.

**④ 생성-판별 모델 간의 격차:**
제로샷 인식 과제에서는 생성적 접근법과 판별적 접근법 사이에 **여전히 격차가 존재**합니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 유효 강건성 (Effective Robustness)

**Diffusion Classifier는 "유효 강건성(effective robustness)"을 보입니다**: 인-도메인(ID) 정확도에서 예측되는 수준보다 훨씬 더 높은 아웃-오브-도메인(OOD) 정확도를 달성합니다.

판별 모델들과 비교했을 때, 동일한 양의 레이블 데이터로 학습된 기준과 비교하면 Diffusion Classifier는 ImageNet 정확도를 기반으로 **예측되는 수치보다 훨씬 높은 ImageNet-A 정확도**를 달성합니다.

특히 **추가 데이터 없이 유의미한 유효 강건성을 달성한 최초의 접근법**입니다.

### 3-2. 약한 증강·정규화 없이도 강력한 일반화

이 모델들은 약한 데이터 증강과 정규화 없이도 학습되었음에도 불구하고, **SOTA 판별적 분류기의 성능에 근접**합니다.

### 3-3. 더 넓은 학습 분포의 가능성

T5-XXL 임베딩 기반으로 학습된 확산 모델(예: Imagen)은 더 나은 제로샷 분류 결과를 보여줄 것으로 기대되지만, 오픈소스가 아니어서 실증적으로 검증하기가 어렵습니다.

### 3-4. 형상-질감 편향 및 속성 바인딩

확산 모델 기반 분류기는 **형상/질감 편향 테스트에서 최고 수준의 결과**를 달성하고, CLIP이 할 수 없는 속성 바인딩(attribute binding)도 성공적으로 수행합니다.

Imagen과 Stable Diffusion 모두 라벨과 충돌하는 질감으로 스타일화된 이미지로 구성된 Cue-Conflict 데이터셋에서 **놀랄만한 성능**을 보였으며, 예를 들어 Imagen은 CLIP 대비 50% 이상의 오류 감소를 달성하고 훨씬 더 큰 ViT-22B 모델까지 능가했습니다.

### 3-5. 생성적 분류기의 고유 강건성 이론

생성적 학습은 데이터 분포를 효과적으로 모델링함으로써 분포 외(out-of-distribution) 인스턴스 처리에 고유한 장점을 제공하며, 특히 적대적 공격에 대한 강건성 향상에 효과적입니다. 그 중 확산 분류기는 강력한 확산 모델을 활용하여 **우월한 경험적 강건성을 입증**했습니다.

이론적으로도 확산 분류기는 $O(1)$ 립시츠 상수(Lipschitzness)를 가짐이 증명되어, **내재적 강건성**을 수학적으로 확립했습니다.

---

## 4. 후속 연구에 미치는 영향과 고려사항

### 4-1. 연구에 미치는 영향

#### ① 생성 모델을 분류기로 활용하는 새로운 패러다임

이 논문은 **샘플 생성만을 목적으로 학습된 Stable Diffusion이 이렇게 훌륭한 분류기 및 추론기로 재활용될 수 있다는 점이 놀랍다**는 것을 보여줌으로써, 생성 모델의 활용 범위를 근본적으로 확장했습니다.

#### ② 확산 분류기의 인증된 강건성 연구로 발전

후속 연구로서 가우시안 노이즈가 섞인 데이터를 분류하기 위해 확산 분류기를 일반화하는 연구가 등장했습니다. 이는 이러한 분포에 대한 ELBO를 유도하고, ELBO로 우도를 근사하며, 베이즈 정리를 통해 분류 확률을 계산하는 방식입니다. 실험 결과 **Noised Diffusion Classifiers(NDCs)의 우월한 인증 강건성**이 입증되었습니다.

특히 단일 기성(off-the-shelf) 확산 모델만으로, 추가 데이터 없이 CIFAR-10에서 $\ell_2$ 노름 0.25 및 0.5 이하의 적대적 섭동에 대해 각각 **80% 이상 및 70% 이상의 인증 강건성**을 달성했습니다.

#### ③ 효율화 연구 촉진

후속 연구로 Gaussian Diffusion Classifiers(GDC)가 제안되어 이전 확산 기반 분류기 대비 **분류 시간을 ImageNet 이미지당 약 1000초에서 0.03초로 극적으로 단축**시켰습니다 (arXiv: 2412.12594, December 2024).

#### ④ 멀티모달 구성적 추론 연구 심화

Stable Diffusion이 OpenCLIP ViT-H/14와 동일한 텍스트 인코더를 사용함에도 Winoground에서 훨씬 뛰어난 성능을 보이는 것은, **확산 모델이 더 나은 개념-이미지 간 교차 모달 바인딩을 학습**한다는 것을 시사하며 이 방향의 연구를 촉진합니다.

#### ⑤ 의료 영상 등 특수 도메인으로의 확장 가능성

확산 모델은 데이터에서 자동으로 현실적인 증강을 학습할 수 있으며, 소외된 그룹에 특화된 합성 이미지 샘플을 생성함으로써 다양한 의료 분야 및 인구통계학적 특성에 걸쳐 **의료 이미지 분류기의 공정성 지표를 향상시킬 수 있습니다.**

---

### 4-2. 앞으로의 연구에서 고려할 점

| 고려사항 | 세부 내용 |
|---|---|
| **추론 효율화** | 적응적 샘플링, 클래스 가지치기, 저해상도 추론 등 연산 비용 절감 방법 연구 |
| **프롬프트 최적화** | 학습 데이터 분포에 맞는 프롬프트 엔지니어링 또는 자동 프롬프트 탐색 |
| **더 넓은 학습 데이터** | 필터링이 덜 된 데이터나 T5-XXL 등의 강력한 텍스트 임베딩 활용 |
| **잠재 공간 vs 픽셀 공간** | 분류기의 적대적 강건성에 영향을 주는 설계 선택 연구 |
| **이론적 분석 확장** | Lipschitz 상수, 인증 반경 등의 이론적 강건성 특성 심화 분석 |
| **멀티모달 확장** | 이미지-텍스트 이외의 모달리티(오디오, 비디오 등)에의 적용 |

결론적으로, 생성 모델은 이전에는 분류에서 판별적 모델에 뒤처졌으나, **오늘날 생성 모델링의 발전 속도는 매우 빠르게 그 격차를 줄이고 있습니다.**

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 핵심 아이디어 | 비교 포인트 |
|---|---|---|
| **Li et al. (ICCV 2023)** "Your Diffusion Model is Secretly a Zero-Shot Classifier" | ELBO 기반 확산 분류기, 제로샷 + 표준 분류 | **기준 논문** |
| **Clark & Jaini (NeurIPS 2023)** "Text-to-Image Diffusion Models are Zero-Shot Classifiers" | 디노이징 능력을 라벨 우도의 프록시로 활용 | 동시 독립 연구; Imagen/SD 적용, 속성 바인딩 강점 |
| **Chen et al. (2024)** "Your Diffusion Model is Secretly a Certifiably Robust Classifier" | Noised Diffusion Classifier(NDC), 인증 강건성 이론화 | 적대적 강건성을 $O(1)$ Lipschitz로 이론적 확립 |
| **GDC (2024, arXiv:2412.12594)** Gaussian Diffusion Classifiers | 효율적 추론; 1000초→0.03초 | 실용성 극대화에 집중 |
| **Robust Classification via a Single Diffusion Model (2023)** | 단일 확산 모델로 적대적 강건 분류 | DiffPure와의 비교; Truth Maximization 등 최적화 |

Clark & Jaini의 동시 연구에서는 확산 모델이 **다양한 제로샷 이미지 분류 데이터셋에서 CLIP과 경쟁력 있는 성능**을 보이며, 형상/질감 편향 테스트에서 최고 수준의 결과를 달성하고 CLIP이 불가능한 속성 바인딩도 성공적으로 수행합니다.

전반적으로 이 분야의 연구들은 순수한 정확도 향상보다는 **적대적 강건성과 불확실성 정량화**에 집중하는 경향을 보이며, 이는 확산 모델의 강점이 더 신뢰할 수 있고 강건한 분류를 제공하는 데 있음을 시사합니다.

---

## 📚 참고 자료 (출처)

1. **Li, A. C., Prabhudesai, M., Duggal, S., Brown, E., & Pathak, D. (2023).** "Your Diffusion Model is Secretly a Zero-Shot Classifier." *ICCV 2023*, pp. 2206–2217. [arXiv:2303.16203](https://arxiv.org/abs/2303.16203)

2. **Diffusion Classifier 공식 프로젝트 페이지.** [diffusion-classifier.github.io](https://diffusion-classifier.github.io/)

3. **Diffusion Classifier 공식 GitHub.** [github.com/diffusion-classifier/diffusion-classifier](https://github.com/diffusion-classifier/diffusion-classifier)

4. **Diffusion Classifier 논문 PDF (ICCV 2023 Camera-Ready).** [openaccess.thecvf.com](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Your_Diffusion_Model_is_Secretly_a_Zero-Shot_Classifier_ICCV_2023_paper.pdf)

5. **Clark, K., & Jaini, P. (2023).** "Text-to-Image Diffusion Models are Zero-Shot Classifiers." *NeurIPS 2023*. [arXiv:2303.15233](https://arxiv.org/abs/2303.15233) / [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2023/file/b87bdcf963cad3d0b265fcb78ae7d11e-Paper-Conference.pdf)

6. **Chen, H., et al. (2024).** "Your Diffusion Model is Secretly a Certifiably Robust Classifier." *NeurIPS 2024*. [arXiv:2402.02316](https://arxiv.org/abs/2402.02316) / [openreview.net](https://openreview.net/forum?id=wGP1tBCP1E)

7. **"Robust Classification via a Single Diffusion Model" (2023).** [arxiv.org/html/2305.15241v2](https://arxiv.org/html/2305.15241v2)

8. **"Struggle with Adversarial Defense? Try Diffusion" (2024).** [arxiv.org/html/2404.08273v1](https://arxiv.org/html/2404.08273v1)

9. **Gaussian Diffusion Classifiers (GDC, 2024).** arXiv:2412.12594. 출처: [gist.github.com/bigsnarfdude](https://gist.github.com/bigsnarfdude/23ec6b30a53437c436c8c4338ee6678c)

10. **ADS Abstract (Li et al., 2023).** [ui.adsabs.harvard.edu](https://ui.adsabs.harvard.edu/abs/2023arXiv230316203L/abstract)

# Diffusion Classifier : Your Diffusion Model is Secretly a Zero-Shot Classifier | Image classification

## 1. 배경 및 동기  
대규모 텍스트-투-이미지(diffusion) 모델은 아름다운 이미지를 생성하지만, 실제로 **분류(classification)**에도 활용할 수 있습니다. 본 논문은 Stable Diffusion과 같은 대규모 확산 모델(diffusion model)의 **조건부 밀도 추정**을 이용해 추가 학습 없이 **제로샷(zero-shot) 분류기**를 구현하는 **Diffusion Classifier**를 제안합니다[1].

---

## 2. 확산 모델(DDPM) 사전 지식  
1) **Forward Process**: 입력 이미지 $$x_0$$에 단계별로 가우시안 노이즈를 더해 $$x_T$$를 생성합니다.  
2) **Reverse Process**: 학습된 네트워크 $$\epsilon_\theta(x_t,c)$$가 노이즈 $$\epsilon$$를 예측하며 $$x_t$$에서 $$x_{t-1}$$를 복원합니다[1].  
3) **ELBO (Evidence Lower Bound)**: 모델은 변분 하한(ELBO)을 최대화하도록 학습되며,
   
$$\text{ELBO} \approx -\mathbb{E}\_{t,\epsilon}\big[\|\epsilon - \epsilon_\theta(x_t,c)\|^2\big] + \text{const}$$

로 나타낼 수 있습니다[1].

---

## 3. Diffusion Classifier 수학적 유도  
1) **Bayes 법칙 적용**

$$p(c \mid x) \propto p(c)\,p_\theta(x\mid c).$$  

   균등 사전확률 $$p(c)=\text{const}$$를 가정하면 $$p_\theta(x\mid c)$$만 최대화하면 됩니다[1].  
4) **ELBO로 근사**  
   
$$\log p_\theta(x\mid c)\approx -\mathbb{E}\_{t,\epsilon}\big[\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}x+\sqrt{1-\bar\alpha_t}\epsilon,\;c)\|^2\big].$$  

5) **Monte Carlo 추정**  
   * 각 클래스 $$c_i$$에 대해 여러 시도(trial)에서 $$\epsilon$$-예측 오차를 계산하고 평균을 구합니다.  
   * 최종 분류는 오류가 가장 낮은 $$c_i$$를 선택합니다[1].

---

## 4. 분산 감소(Variance Reduction) 기법  
- **Paired Difference Test 유사**: 서로 다른 클래스 간 절대 오차 대신 **오차 차이**만 필요하므로, 동일한 $$(t,\epsilon)$$ 샘플을 모든 클래스에 재사용해 통계적 분산을 크게 줄입니다[1].  
- 이 방법으로 매끄러운 클래스 간 비교가 가능해집니다.

---

## 5. 실전 적용 고려사항  
### 5.1. 타임스텝 선택  
- 중간 노이즈 수준($$t\approx500$$)에서 분류 성능이 가장 높으며, 균일 샘플링(uniform sampling)이 최적의 확장성을 보입니다[1].  

### 5.2. 효율적 분류(Adaptive Evaluation)  
- 클래스 수가 많을 때는 **단계별(stage-wise)**로 후보를 점진적 제거합니다.  
  1. 모든 클래스에 적은 시도를 수행, 오차 상위 클래스 제거  
  2. 남은 클래스에 더 많은 시도를 집중  
- 이를 통해 ImageNet(1,000 클래스)도 부분적으로 대응 가능하나 여전히 수백 초가 소요됩니다[1].

---

## 6. 주요 실험 결과  
### 6.1. 제로샷 분류 성능  
| 방법                    | zero-shot | CIFAR-10 | Pets   | Flowers | STL-10 | ImageNet | ObjectNet |
|-------------------------|-----------|----------|--------|---------|--------|----------|-----------|
| Synthetic SD Data       | ✓         | 35.3%    | 31.3%  | 22.1%   | 38.0%  | 18.9%    |  5.2%     |
| SD Features             | ✗         | 84.0%    | 75.9%  | 70.0%   | 87.2%  | 56.6%    | 10.2%     |
| **Diffusion Classifier**| ✓         | **88.5%**|**87.3%**|66.3%    |**95.4%**|61.4%    |**43.4%**  |
| CLIP ResNet-50          | ✓         | 75.6%    | 85.4%  | 65.9%   | 94.3%  | 58.2%    | 40.0%     |
| OpenCLIP ViT-H/14       | ✓         | 97.3%    | 94.6%  | 79.9%   | 98.3%  | 76.8%    | 69.2%     |

- Stable Diffusion 기반 제로샷 분류가 **Synthetic SD Data** 대비 대폭 성능 향상을 보이며, **CLIP ResNet-50**을 넘어서고 **OpenCLIP**과 경쟁할 정도로 발전함을 확인했습니다[1].

### 6.2. 구성(reasoning) 능력 (Winoground)  
| 모델                    | Object | Relation | Both  | 평균   |
|-------------------------|--------|----------|-------|--------|
| Random                  | 25.0%  | 25.0%    | 25.0% | 25.0%  |
| CLIP ViT-L/14           | 27.0%  | 25.8%    | 57.7% | 28.2%  |
| OpenCLIP ViT-H/14       | 39.0%  | 26.6%    | 57.7% | 33.0%  |
| **Diffusion Classifier**|**46.1%**|**29.2%**|**80.8%**|**38.5%**|

- Stable Diffusion의 제너레이티브 특성으로 인해 **관계 중심(Relation)** 구성 과제에서도 타 방법을 상당히 앞서며, 전반적 구성(reasoning) 능력이 검증되었습니다[1].

---

## 7. 결론 및 향후 과제  
- **Diffusion Classifier**는 확산 모델을 제로샷 분류기로 활용하여 **추가 학습 없이** 강력한 성능을 달성합니다.  
- **구성적 추론 능력**과 **분포 이동(robustness)**에서도 뛰어난 결과를 보이며, 생성을 넘어서는 응용성을 제시합니다[1].  
- 향후 **추론 속도** 개선, **프롬프트 튜닝**, 그리고 더 다양한 데이터셋으로 **사전학습 영역 확장**이 중요한 연구 방향이 될 것입니다.

[1] https://ieeexplore.ieee.org/document/10376944/
[2] https://arxiv.org/pdf/2303.16203.pdf
[3] http://arxiv.org/pdf/2406.03736.pdf
[4] https://arxiv.org/html/2412.17219v2
[5] http://arxiv.org/pdf/2402.02316.pdf
[6] https://arxiv.org/pdf/2402.06559.pdf
[7] https://arxiv.org/html/2308.16534
[8] http://arxiv.org/pdf/2308.12469.pdf
[9] https://arxiv.org/abs/2303.16203
[10] https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Your_Diffusion_Model_is_Secretly_a_Zero-Shot_Classifier_ICCV_2023_paper.pdf
[11] https://openreview.net/pdf/a1ae5b7782b9f642d012ef077e717a1b620f0b9a.pdf
[12] https://ar5iv.labs.arxiv.org/html/2303.16203
[13] https://paperswithcode.com/paper/your-diffusion-model-is-secretly-a-zero-shot
[14] https://diffusion-classifier.github.io
[15] https://proceedings.neurips.cc/paper_files/paper/2023/file/b87bdcf963cad3d0b265fcb78ae7d11e-Paper-Conference.pdf
[16] https://arxiv.org/html/2412.12594v1
[17] https://arxiv.org/html/2403.13652
[18] https://arxiv.org/pdf/2406.02929.pdf
[19] https://www.youtube.com/watch?v=t5Daou0eT-g
[20] https://openreview.net/forum?id=fxNQJVMwK2

### 핵심 접근법  
기존 텍스트-이미지 확산 모델(예: Stable Diffusion)이 이미지 생성 외에 **조건부 확률 밀도 추정** 을 통해 제로샷 분류가 가능함을 입증했습니다[1][3]. 이 방법은 별도 학습 없이 사전 훈련된 모델의 density estimate를 활용해 클래스 간 상대적 가능도를 비교하며, 이를 **Diffusion Classifier** 로 명명했습니다[4][7].

### 주요 강점  
1. **다중모달 추론 능력**:  
   텍스트와 이미지 간 구성적 관계 이해에서 CLIP 등의 판별적 모델을 능가합니다(예: "빨간색 사과 vs 녹색 사과" 구분)[5][7].  
2. **벤치마크 성능**:  
   CIFAR-10(77.9%), Flowers(86.2%), ImageNet(58.9%) 등에서 기존 확산 모델 기반 분류기 대비 우수한 성적[2][5].  
3. **효율적 강건성**:  
   이미지넷 분류기 추출 시 약한 데이터 증강만으로도 분포 변화에 강인한 특성 보임[3][4].  

### 적용 사례  
- **이미지넷 분류기 변환**: 클래스 조건부 확산 모델을 전통적인 분류기로 변환 가능[4][7]  
- **합성 데이터 활용**: 확산 모델이 생성한 합성 데이터로 분류기 훈련 시 성능 향상[2][6]  

### 한계 및 비교  
- **계산 비용**: 실시간 추론에는 여전히 고비용(이미지당 1-2분 소요)[6]  
- **CLIP 대비 성능 격차**: 일부 벤치마크에서 OpenCLIP ViT-H/14 대비 10-15%p 낮은 정확도[2][5]  

이 연구는 생성 모델이 판별 작업에서도 유용함을 입증하며, 향후 다중모달 AI 시스템 개발에 새로운 방향성을 제시했습니다[1][7]. 특히 데이터 증강 없이도 분포 변화에 강인한 분류 가능성은 실제 응용 분야에서 주목할 만한 결과입니다[3][4].

[1] https://arxiv.org/abs/2303.16203
[2] https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Your_Diffusion_Model_is_Secretly_a_Zero-Shot_Classifier_ICCV_2023_paper.pdf
[3] https://openreview.net/forum?id=Ck3yXRdQXD
[4] https://github.com/diffusion-classifier/diffusion-classifier
[5] https://paperswithcode.com/paper/your-diffusion-model-is-secretly-a-zero-shot
[6] https://www.jetir.org/papers/JETIR2411561.pdf
[7] https://huggingface.co/papers/2303.16203
[8] https://papers.nips.cc/paper_files/paper/2023/file/b87bdcf963cad3d0b265fcb78ae7d11e-Paper-Conference.pdf
[9] https://www.computer.org/csdl/proceedings-article/iccv/2023/071800c206/1TJjVcPg24g
[10] https://www.youtube.com/watch?v=t5Daou0eT-g

