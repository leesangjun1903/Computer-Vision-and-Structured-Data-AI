
# Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models

> **저자:** Tuomas Kynkäänniemi, Miika Aittala, Tero Karras, Samuli Laine, Timo Aila, Jaakko Lehtinen (Aalto University & NVIDIA)
> **발표:** NeurIPS 2024 | arXiv: 2404.07724

---

## 1️⃣ 핵심 주장 및 주요 기여 요약

### 🔑 핵심 주장

기존에는 이미지 샘플링 체인 전체에 걸쳐 일정한 가이던스 가중치(constant guidance weight)를 적용해왔다. 그러나 이 논문은 **가이던스가 샘플링 체인의 시작 부분(높은 노이즈 레벨)에서는 명백히 해롭고, 끝 부분(낮은 노이즈 레벨)에서는 대체로 불필요하며, 오직 중간 구간에서만 유익하다**는 것을 보인다.

### 🏆 주요 기여

| 기여 | 내용 |
|------|------|
| **Limited Guidance Interval** | 가이던스를 특정 노이즈 레벨 구간에만 제한 적용 |
| **FID 기록 갱신** | ImageNet-512에서 FID 1.81 → 1.40으로 향상 |
| **범용성 입증** | 다양한 아키텍처/데이터셋/샘플러에서 유효 |
| **하이퍼파라미터 제안** | Guidance Interval을 모든 확산 모델에서 노출할 것을 권고 |

이 방법은 가이던스를 특정 노이즈 레벨 범위로 제한함으로써 **추론 속도와 결과 품질을 동시에 향상**시킨다.

---

## 2️⃣ 상세 분석: 문제, 방법, 구조, 성능, 한계

### 🔴 해결하고자 하는 문제

확산 모델에서 표준 샘플링만으로는 고품질 이미지가 보장되지 않아, 가이던스 방법이 생성 품질을 높이는 데 필요하다. **Classifier Guidance** (Dhariwal & Nichol, 2021)는 노이즈가 추가된 이미지에 학습된 분류기의 그래디언트를 활용하여 클래스 가능성을 높이는 방식으로 이 개념을 도입하였다. 이후 **Classifier-Free Guidance (CFG)** (Ho & Salimans, 2022)는 명시적인 분류기 없이 동일한 효과를 달성할 수 있게 하였다.

그러나 기존 방법의 핵심 문제는, 샘플링 체인 전체에 동일한 가이던스 가중치를 적용한다는 것이며, 이는 높은 노이즈 단계에서는 명백히 유해하고, 낮은 노이즈 단계에서는 불필요하다.

또한 CFG의 주요 문제점으로는 a) 색상 과포화(intensity oversaturation), b) 매우 큰 가중치에서 분포 이탈(out-of-distribution) 샘플 생성, c) 단순한 배경과 같은 쉬운 샘플로 인한 다양성 제한이 있다.

---

### 🟢 제안하는 방법 (수식 포함)

#### ① Classifier-Free Guidance (CFG) 기본 수식

CFG에서 수정된 스코어 함수는 다음과 같이 정의된다:

$$\tilde{\epsilon}_\theta(\mathbf{x}_t, c, t) = \epsilon_\theta(\mathbf{x}_t, t) + w \cdot \left(\epsilon_\theta(\mathbf{x}_t, c, t) - \epsilon_\theta(\mathbf{x}_t, t)\right)$$

또는 스코어 함수(score function) 형태로:

$$s_{t,\gamma}(\mathbf{x}, c) = \gamma \nabla_{\mathbf{x}_t} \log q_t(\mathbf{x}_t | c) + (1-\gamma) \nabla_{\mathbf{x}_t} \log q_t(\mathbf{x}_t)$$

조건부 샘플링만으로는 합성 샘플이 시각적으로 비일관적인 결과를 낳기 때문에, CFG는 사전 설정된 가중치 $\gamma$를 사용하여 조건부 스코어 함수와 비조건부 스코어 함수 사이의 보간(interpolation)을 수행한다.

#### ② Limited Guidance Interval 적용

본 논문에서 제안하는 핵심 방법은, 가이던스를 노이즈 레벨 구간 $[\sigma_{\min}^{\text{guide}}, \sigma_{\max}^{\text{guide}}]$ 로 제한하는 것이다:

$$\tilde{\epsilon}_\theta(\mathbf{x}_t, c, t) = \begin{cases} \epsilon_\theta(\mathbf{x}_t, t) + w \cdot \left(\epsilon_\theta(\mathbf{x}_t, c, t) - \epsilon_\theta(\mathbf{x}_t, t)\right) & \text{if } \sigma_t \in [\sigma_{\min}^{\text{guide}}, \sigma_{\max}^{\text{guide}}] \\ \epsilon_\theta(\mathbf{x}_t, c, t) & \text{otherwise} \end{cases}$$

즉, 노이즈 레벨 $\sigma_t$가 정의된 구간 내에 있을 때만 가이던스를 적용하고, 그 외의 단계(매우 높은 노이즈 또는 매우 낮은 노이즈)에서는 일반 조건부 디노이징을 수행한다.

가이던스는 샘플링 체인의 시작(높은 노이즈)에서는 명백히 해롭고, 끝(낮은 노이즈)에서는 불필요하며, 중간에서만 이로우므로, 이를 특정 노이즈 레벨 범위로 제한하여 추론 속도와 결과 품질 모두를 향상시킨다.

#### ③ 하이퍼파라미터로서의 Guidance Interval

저자들은 **가이던스를 사용하는 모든 확산 모델에서 가이던스 인터벌을 하이퍼파라미터로 노출할 것을 제안**한다.

---

### 🏗️ 모델 구조

이 논문은 새로운 네트워크 아키텍처를 제안하는 것이 아니라, **기존 확산 모델의 샘플링 프로세스를 개선**하는 방법을 제안한다. 실험에 사용된 모델/환경은 다음과 같다:

| 모델/환경 | 내용 |
|-----------|------|
| **EDM2-XXL** | ImageNet-512 정량적 실험의 주 모델 |
| **Stable Diffusion XL** | 대규모 텍스트-이미지 생성 검증 |
| **다양한 샘플러** | DDPM, DDIM, DPM-Solver 등 범용 검증 |

이 방법은 서로 다른 샘플러 파라미터, 네트워크 아키텍처, 데이터셋에 걸쳐 정량적·정성적으로 이점이 있으며, Stable Diffusion XL의 대규모 설정에서도 유효함을 보인다.

---

### 📈 성능 향상

제안된 Limited Guidance Interval은 **ImageNet-512에서 FID 기록을 1.81에서 1.40으로 크게 향상**시켰다.

추가적으로, 가이던스를 일부 단계에서 생략함으로써:
- **추론 속도 향상**: 가이던스가 불필요한 단계에서 조건부/비조건부 두 번의 forward pass를 한 번으로 줄일 수 있어 연산 비용 절감
- **품질 향상**: 고노이즈 단계에서의 유해한 영향 제거로 전체적인 생성 품질 개선

---

### ⚠️ 한계

검색된 정보를 바탕으로 확인 가능한 한계:

1. **최적 구간의 수동 탐색**: 최적의 $[\sigma_{\min}^{\text{guide}}, \sigma_{\max}^{\text{guide}}]$는 모델/데이터셋마다 다를 수 있어 별도의 하이퍼파라미터 탐색이 필요하다.
2. **이론적 근거의 미흡**: 왜 중간 노이즈 구간에서만 가이던스가 유익한지에 대한 이론적 증명보다는 경험적 분석에 의존한다.
3. **범용적 최적 구간 부재**: 서로 다른 샘플러 파라미터, 네트워크 아키텍처, 데이터셋에서 정량적·정성적으로 이로움을 보이지만, 단일 최적 구간이 모든 설정에서 동일하게 적용되지는 않을 수 있다.

---

## 3️⃣ 모델의 일반화 성능 향상 가능성

이 연구는 서로 다른 **샘플러 파라미터, 네트워크 아키텍처, 데이터셋**에 걸쳐 정량적·정성적으로 효과가 있음을 보이며, Stable Diffusion XL과 같은 대규모 설정까지 검증한다. 이는 본 방법의 강력한 **일반화 가능성**을 시사한다.

일반화 성능 향상 가능성을 다음 네 측면에서 정리할 수 있다:

### ① 아키텍처 독립성 (Architecture-Agnostic)
본 방법은 특정 네트워크 구조에 종속되지 않고 **샘플링 알고리즘 수준**에서 동작한다. UNet 기반(SDXL), Transformer 기반(DiT), EDM 기반 모델 모두에 적용 가능하다.

### ② 도메인 독립성 (Domain Generalization)
ImageNet뿐만 아니라 대규모 텍스트-이미지 생성 모델(Stable Diffusion XL)에서도 이점이 확인되어, 클래스 조건부 생성과 텍스트 조건부 생성 모두에서의 일반화 가능성을 보여준다.

### ③ 과가이던스(Over-guidance) 억제를 통한 분포 커버리지 향상
기존 CFG의 큰 문제 중 하나는 높은 가이던스 가중치에서의 분포 이탈과 다양성 감소인데, Limited Guidance Interval은 높은 노이즈 단계에서 가이던스를 제거함으로써 생성 분포가 실제 데이터 분포를 보다 충실히 커버하게 돕는다. 이는 **분포 품질(Distribution Quality)** 지표(FID, recall 등) 향상에 직결된다.

### ④ 다양한 모달리티로의 확장 가능성
확산 모델은 이미지 외에도 비디오, 3D 형상, 오디오 등 다른 모달리티로도 쉽게 확장되는데, Limited Guidance Interval 전략은 이러한 모달리티에도 동일하게 적용될 가능성이 높다.

---

## 4️⃣ 향후 연구에 미치는 영향 및 고려 사항

### 🌱 향후 연구에 미치는 영향

#### A. 가이던스 스케줄링 연구의 촉진
CFG의 불안정성을 해결하기 위한 가이던스 스케일 스케줄러 연구가 활발해지고 있으나, 이러한 스케줄은 대체로 수동으로 설계되고 서로 상반된 휴리스틱에 기반한다. 본 논문은 이 흐름에 이론적·실증적 기반을 제공하여, **자동화된 또는 학습 기반의 가이던스 스케줄 탐색** 연구를 자극할 것으로 예상된다.

#### B. 가이던스 구간의 적응적 최적화 연구
현재는 가이던스 구간을 수동으로 설정하는 방식이지만, 이를 **모델이나 입력 조건에 따라 동적으로 결정**하는 메타러닝 또는 강화학습 기반 접근법 연구가 촉진될 것이다.

#### C. 다른 가이던스 기법과의 결합 연구
CFG는 수정된 분포 $p_\omega(x_0|c) \propto p(x_0)^{-\omega}p(x_0|c)^{1+\omega}$에서 샘플링하는 방법으로 정당화되었지만, 이후 연구에서 표준 CFG 샘플링이 실제로는 이 목표 분포에서 샘플을 생성하지 않는다는 것이 밝혀졌으며, 오히려 원래 조건부 분포 $p(x_0|c)$의 모드 방향으로 샘플을 이동시킨다는 것이 알려져 있다. Limited Guidance Interval은 Rectified CFG, Inner CFG 등 여러 개선된 가이던스 방법과 직교적(orthogonal)으로 결합될 수 있다.

#### D. 비디오/3D/오디오 생성으로의 확장
확산 모델은 비디오, 3D 형상, 오디오 등 다른 모달리티에서도 널리 사용되므로, 이러한 영역에서 최적 가이던스 구간에 대한 체계적인 분석 연구가 기대된다.

---

### 🔬 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 주요 아이디어 | 본 논문과의 관계 |
|------|------|----------------|-----------------|
| **DDPM** (Ho et al.) | 2020 | 확산 모델 기초 확립 | 기반 프레임워크 |
| **Diffusion Beats GANs** (Dhariwal & Nichol) | 2021 | Classifier Guidance 도입 | 가이던스 개념의 시초 |
| **CFG** (Ho & Salimans) | 2022 | 분류기 없는 가이던스 | 본 논문이 개선 대상으로 삼는 방법 |
| **Stable Diffusion** (Rombach et al.) | 2022 | Latent Diffusion Models | 본 논문의 검증 플랫폼 |
| **Rethinking CFG** | 2024 | 타임스텝 정보 기반 내부 가이던스 | 가이던스 개선의 병렬 연구 |
| **Rectified Diffusion Guidance** | 2024 | 가이던스의 이론적 교정 | 이론적 근거 보완 |
| **Navigating with Annealing Guidance Scale** | 2026 | CFG 불안정성 해결을 위한 스케줄러 연구 | 본 논문 이후 나온 스케줄링 연구 |

본 논문의 핵심 기여는 **Classifier-Free Guidance를 제한된 구간에서만 적용함으로써 확산 모델의 정량적·정성적 결과를 향상**시키는 것이다.

---

### ⚠️ 앞으로 연구 시 고려할 점

1. **최적 구간의 자동 탐색**: $[\sigma_{\min}^{\text{guide}}, \sigma_{\max}^{\text{guide}}]$를 모델 및 데이터셋에 따라 자동으로 결정하는 방법론 개발이 필요하다.

2. **이론적 분석 강화**: 왜 중간 노이즈 레벨에서만 가이던스가 유효한지에 대한 정보이론적 또는 확률론적 이론 분석이 필요하다.

3. **가이던스 강도와 구간의 동시 최적화**: 구간과 가이던스 가중치 $w$를 함께 최적화하는 joint optimization 접근법 검토가 필요하다.

4. **다양한 조건부 생성 태스크로의 적용**: 텍스트-이미지, 이미지 편집, 인페인팅 등 다양한 downstream task에서의 최적 구간 분석이 요구된다.

5. 저자들이 제안하듯, 가이던스를 사용하는 **모든 확산 모델에서 가이던스 인터벌을 하이퍼파라미터로 노출**하는 관행을 채택하는 것이 권고된다.

---

## 📚 참고 자료 및 출처

| # | 자료 | URL |
|---|------|-----|
| 1 | **arXiv 논문 원문** (2404.07724) | https://arxiv.org/abs/2404.07724 |
| 2 | **arXiv PDF** | https://arxiv.org/pdf/2404.07724 |
| 3 | **NeurIPS 2024 공식 페이지** | https://proceedings.neurips.cc/paper_files/paper/2024/hash/dd540e1c8d26687d56d296e64d35949f-Abstract-Conference.html |
| 4 | **OpenReview** | https://openreview.net/forum?id=nAIhvNy15T |
| 5 | **Hugging Face Papers** | https://huggingface.co/papers/2404.07724 |
| 6 | **GitHub 코드** | https://github.com/kynkaat/guidance-interval |
| 7 | **Aalto University Research Portal** | https://research.aalto.fi/en/publications/applying-guidance-in-a-limited-interval-improves-sample-and-distr |
| 8 | **Semantic Scholar** | https://www.semanticscholar.org/paper/Applying-Guidance-in-a-Limited-Interval-Improves-in-Kynk%C3%A4%C3%A4nniemi-Aittala/68735af959504b0c812f591a56e10254be271928 |
| 9 | **Rethinking CFG** (비교 연구) | https://arxiv.org/pdf/2407.02687 |
| 10 | **Rectified Diffusion Guidance** (비교 연구) | https://arxiv.org/pdf/2410.18737 |
| 11 | **Navigating with Annealing Guidance Scale** (후속 연구) | https://arxiv.org/html/2506.24108 |

> ⚠️ **정확도 관련 주의**: 논문의 세부 수식(Limited Guidance Interval 조건 분기) 중 일부는 검색된 정보에서 직접 확인되지 않아, CFG의 일반 수식과 논문의 개념을 기반으로 표준적인 방식으로 재구성하였습니다. 정확한 수식 및 세부 실험 결과는 반드시 [원문 PDF](https://arxiv.org/pdf/2404.07724)를 직접 확인하시길 권장합니다.
