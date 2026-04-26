
# WildGaussians: 3D Gaussian Splatting in the Wild 

> **논문 정보**
> - **제목:** WildGaussians: 3D Gaussian Splatting in the Wild
> - **저자:** Jonas Kulhánek, Songyou Peng, Zuzana Kukelova, Marc Pollefeys, Torsten Sattler
> - **소속:** Czech Technical University in Prague / ETH Zurich
> - **학회:** NeurIPS 2024
> - **arXiv:** [2407.08447](https://arxiv.org/abs/2407.08447)

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

3D 장면 재구성 분야는 NeRF(Neural Radiance Fields)가 주도하고 있었으나, 3D Gaussian Splatting(3DGS)이 실시간 렌더링 속도와 유사한 품질로 주목받기 시작했다. 그러나 두 방법 모두 통제된 환경에서는 탁월하지만, **가려짐(occlusions), 동적 객체, 조명 변화**를 가진 In-the-wild 데이터에서는 여전히 도전적이다. NeRF는 per-image embedding 벡터를 통해 이러한 조건에 쉽게 적응할 수 있지만, 3DGS는 명시적 표현 방식과 공유 파라미터의 부재로 인해 어려움을 겪는다.

이를 해결하기 위해, 저자들은 3DGS에서 가려짐과 외관 변화를 처리하는 새로운 방법인 **WildGaussians**를 소개한다. DINO 특징을 활용하고 3DGS 내에 외관 모델링 모듈을 통합하여 최첨단 결과를 달성한다.

### 1.2 주요 기여 (Contributions)

WildGaussians는 3DGS의 실시간 렌더링 속도를 유지하면서 In-the-wild 데이터 처리에서 3DGS와 NeRF 기준 모델 모두를 능가하며, 이를 단순한 아키텍처 프레임워크 내에서 달성한다.

구체적인 기여는 다음 두 가지이다:

1. **외관 모델링(Appearance Modeling):** Per-Gaussian 및 per-image 임베딩이 외관 MLP에 입력되어, Gaussian의 뷰 의존적 색상에 적용되는 아핀 변환의 파라미터를 출력한다.

2. **불확실성 모델링(Uncertainty Modeling):** GT 이미지의 DINO 특징을 학습된 변환을 통해 불확실성 추정치를 획득하며, 불확실성 학습을 위해 DINO 코사인 유사도를 사용한다.

---

## 2. 해결하고자 하는 문제, 제안 방법(수식 포함), 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

In-the-wild 데이터는 가려짐, 동적 객체, 다양한 조명으로 특징지어지며, NeRF는 per-image embedding 벡터로 적응이 쉽지만, 3DGS는 명시적 표현 방식과 공유 파라미터의 부재로 인해 어려움을 겪는다.

장면에 가리는 물체가 있으면 Gaussian Splatting이 장면을 올바르게 표현하지 못해 과도한 양의 floater가 생성된다. 이 방법은 DINO 기반 불확실성 예측기를 사용하여 가리는 물체를 제거할 수 있다.

---

### 2.2 제안하는 방법 (수식 포함)

#### (A) 기본 3DGS 표현

3DGS는 장면을 3D 가우시안 집합으로 표현한다. 각 가우시안 $G_i$는 다음으로 파라미터화된다:

$$G_i = \{\mu_i \in \mathbb{R}^3,\ \Sigma_i \in \mathbb{R}^{3\times3},\ \alpha_i \in \mathbb{R},\ c_i(\mathbf{d}) \in \mathbb{R}^3\}$$

여기서 $\mu_i$는 위치, $\Sigma_i$는 공분산(형태/크기/방향), $\alpha_i$는 불투명도, $c_i(\mathbf{d})$는 뷰 방향 $\mathbf{d}$에 의존하는 색상(구면 조화 함수로 표현)이다.

렌더링된 색상은 타일 기반 래스터라이저를 통해 다음과 같이 계산된다:

$$\hat{C}(\mathbf{r}) = \sum_{i \in \mathcal{N}} c_i \cdot \alpha_i \prod_{j < i}(1-\alpha_j)$$

---

#### (B) 외관 모델링 (Appearance Modeling)

In-the-wild 장면 처리를 위해 두 가지 핵심 구성 요소를 제안한다. (1) **외관 모델링**은 픽셀 색상이 시점뿐 아니라 촬영 시간, 날씨 등 조건에도 의존한다는 사실을 처리한다. 다양한 조건에서 촬영된 이미지로부터 장면을 재구성하는 NeRF 기반 방식을 따라 각 훈련 이미지에 대한 외관 임베딩을 학습하며, 각 가우시안별 외관 임베딩도 학습하여 조명 효과 등 국소적 효과를 모델링한다.

각 가우시안 $G_i$에는 훈련 가능한 임베딩 벡터 $\mathbf{g}_i$가 할당된다. 또한 각 훈련 이미지 $j$에는 고유한 외관 임베딩 $\mathbf{e}_j$가 있다. MLP $f$는 이 임베딩들과 가우시안의 기본 색상을 입력으로 받아 파라미터를 생성한다.

외관 MLP는 다음과 같이 아핀 변환 파라미터 $(\mathbf{A}, \mathbf{b})$를 출력한다:

$$(\mathbf{A}_i^{(j)}, \mathbf{b}_i^{(j)}) = f_\theta(\mathbf{g}_i,\ \mathbf{e}_j)$$

최종 외관 조건부 색상은:

$$\tilde{c}_i^{(j)}(\mathbf{d}) = \mathbf{A}_i^{(j)} \cdot c_i(\mathbf{d}) + \mathbf{b}_i^{(j)}$$

본 방법은 각 가우시안에 직접 외관 벡터를 임베딩하는 더 단순하고 확장 가능한 전략을 채택한다. 이 설계는 아키텍처를 단순화할 뿐 아니라, 외관이 고정된 후 학습된 표현을 다시 3DGS로 '베이킹(baking)'할 수 있게 하여 효율성과 적응성을 높인다.

---

#### (C) 불확실성 모델링 (Uncertainty Modeling via DINOv2)

불확실성 모델링은 보행자나 차량과 같은 가려짐을 훈련 중에 식별하고 무시하는 데 도움을 준다. 연구자들은 기존 방법과 비교하여 조명 변화에 더 강건한 사전 학습된 DINOv2 특징에 의존한다.

DINOv2 특징이 훈련 이미지에서 추출된다. 이 특징들에 대해 학습된 아핀 변환이 픽셀당 불확실성 인코딩을 예측하여 정적 영역과 가리는 물체를 구분하는 데 도움을 준다.

DINO 기반 불확실성 손실 함수는 다음과 같이 정의된다:

$$\mathcal{L}_{\text{dino}}(\tilde{D}, D) = \min\!\left(1,\ 2 - \frac{2\,\tilde{D} \cdot D}{\|\tilde{D}\|_2 \,\|D\|_2}\right)$$

여기서 $\tilde{D}$는 렌더링된 이미지의 DINO 특징, $D$는 GT 이미지의 DINO 특징이다.

이 결과는 모델이 가리는 물체를 효과적으로 무시하고 정적 장면 구조에 집중할 수 있도록 한다.

---

#### (D) 전체 학습 손실 함수

전체 학습 손실은 기존 3DGS 재구성 손실과 불확실성 가중 항을 조합하여 구성된다. 불확실성 맵 $\beta$를 통해 가려짐 픽셀에 낮은 가중치를 부여한다:

$$\mathcal{L}_{\text{total}} = \sum_p \left[(1 - \beta_p)\,\mathcal{L}_{\text{rgb}}(p) + \lambda_{\text{dino}}\,\mathcal{L}_{\text{dino}}(p)\right]$$

---

### 2.3 모델 구조

방법론의 핵심은 전통적인 3DGS 기법을 향상시키기 위해 **외관 모델링**과 **불확실성 모델링**이라는 두 가지 주요 구성 요소를 도입하는 것을 중심으로 한다.

| 구성 요소 | 입력 | 출력 | 역할 |
|---|---|---|---|
| **Appearance MLP** | $\mathbf{g}_i$ (per-Gaussian), $\mathbf{e}_j$ (per-image) | 아핀 변환 파라미터 $(A, b)$ | 조명·날씨 등 외관 변화 처리 |
| **Uncertainty Predictor** | GT 이미지의 DINOv2 특징 | 픽셀당 불확실성 맵 $\beta$ | 가려짐(Occluder) 마스킹 |
| **3DGS Renderer** | 변환된 가우시안 색상 | 렌더링 이미지 | 실시간 래스터라이제이션 |

렌더러는 3DGS 및 Mip-Splatting 위에 구축된다.

---

### 2.4 성능 향상

WildGaussians는 두 가지 도전적인 데이터셋, 즉 다양한 수준의 가려짐이 있는 **NeRF On-the-go 데이터셋**과 다양한 조건에서 촬영된 유명 랜드마크의 사용자 제공 이미지가 포함된 **Photo Tourism 데이터셋**에서 평가되었다. 새로운 방법은 대부분의 예시에서 현재 최첨단 방법의 품질을 능가하는 동시에, NVIDIA RTX 4090 GPU에서 초당 117 이미지의 실시간 렌더링을 가능하게 했다.

방법론은 특히 높은 가려짐 시나리오와 다양한 외관 조건에서 WildGaussians가 어떻게 성능이 우수한지 강조하면서, 기존 방법들과 광범위하게 평가되었다. 평가는 PSNR, SSIM, LPIPS 등의 지표를 사용한다.

---

### 2.5 한계 (Limitations)

본 방법은 실시간 렌더링과 외관 모델링을 가능하게 하지만, 현재로서는 **물체의 하이라이트(specular highlights)를 캡처하지 못한다**. 또한 불확실성 모델링이 MSE나 SSIM보다 강건하지만, 일부 도전적인 시나리오에서는 여전히 어려움을 겪는다. 이를 처리하는 한 가지 방법은 사전 학습된 확산 모델(diffusion models)과 같은 추가적인 사전 지식을 통합하는 것으로, 향후 연구 과제로 남겨둔다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재의 일반화 강점

WildGaussians 모델은 서로 다른 시간이나 계절에 걸쳐 다양한 비율의 가리는 물체와 함께 촬영된 이미지를 포함하는 **비제어 In-the-wild 설정**으로 Gaussian Splatting을 확장한다. 성공의 핵심은 3DGS에 맞게 특별히 설계된 새로운 외관 및 불확실성 모델링으로, 고품질 실시간 렌더링을 보장한다. 이 방법은 노이즈가 많은 크라우드 소싱 데이터 소스에서 강건하고 다목적 포토리얼리스틱 재구성을 향한 한 걸음이라고 믿는다.

### 3.2 일반화를 가능하게 하는 설계 전략

본 방법은 각 가우시안에 직접 외관 벡터를 임베딩하는 더 단순하고 확장 가능한 전략을 채택한다. 이 설계는 아키텍처를 단순화할 뿐 아니라, 외관이 고정된 후 학습된 표현을 3DGS로 '베이킹(baking)'할 수 있게 하여 **효율성과 적응성**을 높인다.

이 'baking' 전략은 훈련된 외관 MLP를 추론 시에는 완전히 제거하고 구운(baked) 색상을 직접 사용함으로써 추론 속도를 표준 3DGS 수준으로 유지하면서도 일반화 능력을 확보하는 핵심 메커니즘이다.

### 3.3 DINOv2 특징의 일반화 기여

사전 계산된 DINO와 같은 비전 특징의 사용은 다양한 비전 태스크에 일반화하는 능력을 보여주었다.

DINOv2 기반 불확실성 예측은 훈련 이미지의 특정 가리는 물체에 과적합되지 않고, 의미론적으로 다른 객체 유형(보행자, 차량, 계절 변화 등)에 강건하게 일반화된다는 것이 핵심 강점이다.

### 3.4 최신 후속 연구에서의 일반화 확장

WildSplatter(2025)는 알 수 없는 카메라 파라미터와 다양한 조명 조건을 가진 비제어 이미지를 위한 피드포워드(feed-forward) 3DGS 모델을 제안하며, 3DGS는 일반적으로 반복적 최적화와 일관된 조명 하의 다시점 이미지를 필요로 하지만, WildSplatter는 비제어 사진 컬렉션으로 학습되어 3D 가우시안과 외관 임베딩을 입력 이미지에 조건화하여 공동 학습한다. 이러한 설계는 조명 및 외관의 상당한 변화를 표현하기 위해 가우시안 색상을 유연하게 조정할 수 있도록 한다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 관련 연구 계보

| 연구 | 연도/학회 | 표현 방식 | 외관 모델링 | 가려짐 처리 | 특징 |
|---|---|---|---|---|---|
| **NeRF-W** (Martin-Brualla et al.) | 2021 CVPR | NeRF(암시적) | Per-image 임베딩 | 불확실성 모델링 | 최초 In-the-wild NeRF |
| **RobustNeRF** | 2023 | NeRF | - | IRLS 기반 아웃라이어 제거 | 정적 장면 재구성 집중 |
| **NeRF On-the-go** | 2024 CVPR | NeRF | - | DINOv2+MLP 불확실성 | WildGaussians 비교 대상 |
| **SWAG** (Dahmani et al.) | 2024 ECCV | 3DGS | hash-grid 기반 MLP | 비지도 transient 가우시안 | 첫 3DGS In-the-wild 확장 중 하나 |
| **GS-W** (Zhang et al.) | 2024 ECCV | 3DGS | CNN 기반 intrinsic/dynamic 특징 분리 | 2D visibility map | 물리적 외관 분리 |
| **WildGaussians** | **2024 NeurIPS** | **3DGS** | **Per-Gaussian + Per-image MLP** | **DINOv2 기반 불확실성** | **단순·확장 가능·실시간** |
| **SpotlessSplats** | 2024 | 3DGS | - | Diffusion 특징 기반 마스킹 | 가려짐에 특화 |
| **WildSplatter** | 2025 | 3DGS (피드포워드) | 입력 이미지 조건부 임베딩 | - | 카메라 파라미터 불필요 |

### 4.2 SWAG와의 비교

SWAG는 외부 해시 그리드 기반 암시적 필드에 외관 데이터를 저장하는 방식으로 이 문제를 해결한다. 이에 반해 WildGaussians는 각 가우시안 내에 외관 벡터를 직접 임베딩하는 더 단순하고 확장 가능한 전략을 채택한다.

SWAG의 경우 3DGS의 PSNR을 평균 5.01 dB 향상시킨다. SWAG의 hash-grid 쿼리 방식은 렌더링 속도를 약 15 FPS로 저하시키는 단점이 있는 반면, WildGaussians는 훨씬 높은 실시간 속도를 유지한다.

### 4.3 SpotlessSplats와의 비교

기존 방법들(SWAG, WildGaussians 포함)은 주로 밝기 변화와 같은 전역 외관 변화에 집중하며, 이 연구를 위해 큐레이션된 **무작위 촬영 데이터셋의 가려짐**에는 집중하지 않는다. SpotlessSplats는 이 부분을 확산 모델 특징으로 보완하는 방향을 취한다.

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5.1 연구에 미치는 영향

1. **3DGS In-the-Wild 연구의 기준점 제시:** WildGaussians는 3D 장면 재구성 분야, 특히 비제어 환경에서 고품질 렌더링과 동적 요소 및 조명 변화 처리 모두를 달성하는 중요한 발전을 제공한다. 이 기여는 전통적인 Gaussian 스플래팅 프레임워크를 실질적으로 향상시켜 실제 시나리오에 더 적용 가능하고 강건하게 만든다.

2. **'Baking' 패러다임의 전파:** per-Gaussian 임베딩 후 표현을 순수 3DGS로 '굽는' 전략은 이후 연구들이 속도 손실 없이 외관 모델링을 통합하는 공통 패턴으로 자리잡았다.

3. **피드포워드 모델로의 영향:** WildSplatter(2025)는 비제어 이미지를 위한 외관 모델링을 갖춘 최적화 기반 3DGS 방법인 WildGaussians와 비교하여 자신의 방법을 위치시킨다.

4. **NerfBaselines 통합:** WildGaussians는 이미 NerfBaselines의 최신 릴리스에 포함되어 있어, 연구 재현성과 비교 분석의 기준으로 널리 활용될 수 있다.

### 5.2 향후 연구 시 고려할 점

#### ① 미해결 기술 문제

- **Specular Highlight(정반사 하이라이트):** 현재 방법은 실시간 렌더링과 외관 모델링을 가능하게 하지만, 현재로서는 물체의 하이라이트를 캡처하지 못한다. 이를 해결하기 위한 물리 기반 렌더링(PBR) 통합이 중요 연구 방향이다.

- **극단적 가려짐 시나리오:** 향후 연구는 극단적 가려짐 및 외관 변화를 처리하기 위한 더 정교한 모델 통합을 제안한다.

- **확산 모델 사전 지식 통합:** 이를 처리하는 한 가지 방법은 사전 학습된 확산 모델과 같은 추가적인 사전 지식을 통합하는 것이며, 저자들은 이를 향후 연구 과제로 남겨두었다.

#### ② 일반화 확장 방향

- **카메라 파라미터 비의존성:** 현재 WildGaussians는 COLMAP 등으로 사전 추정된 카메라 파라미터가 필요하다. WildSplatter처럼 알 수 없는 카메라 파라미터를 가진 희소 입력 뷰에서 1초 이내에 3D 가우시안을 재구성하고, 다양한 조명 조건에서 외관 제어를 가능하게 하는 방향으로 발전할 수 있다.

- **동적 장면으로의 확장:** 현재 방법은 정적 배경 재구성에 초점을 맞추며, 진정한 동적 객체(지속적으로 움직이는 요소)에 대한 모델링은 여전히 도전 과제이다.

- **대규모 장면 확장성:** 수천~수만 장의 In-the-wild 이미지를 처리할 수 있는 확장성 연구가 필요하다.

#### ③ 평가 방법론

평가는 NeRF On-the-go와 Photo Tourism 등 다양한 데이터셋에 걸쳐 PSNR, SSIM, LPIPS 지표를 활용하는 것이 중요하다. 향후에는 이러한 지표를 넘어 사용자 연구(user study) 및 실제 응용 시나리오에서의 평가도 고려되어야 한다.

---

## 📚 참고 자료 및 출처

| # | 제목 | 출처 |
|---|---|---|
| 1 | WildGaussians: 3D Gaussian Splatting in the Wild (공식 프로젝트 페이지) | https://wild-gaussians.github.io/ |
| 2 | WildGaussians arXiv 논문 (arXiv:2407.08447) | https://arxiv.org/abs/2407.08447 |
| 3 | WildGaussians NeurIPS 2024 공식 논문 PDF | https://proceedings.neurips.cc/paper_files/paper/2024/ |
| 4 | WildGaussians GitHub 공식 코드 | https://github.com/jkulhanek/wild-gaussians/ |
| 5 | WildGaussians OpenReview (NeurIPS 2024) | https://openreview.net/forum?id=NU3tE3lIqf |
| 6 | WildGaussians arXiv HTML 전문 | https://arxiv.org/html/2407.08447v1 |
| 7 | [Literature Review] WildGaussians (Moonlight) | https://www.themoonlight.io/en/review/wildgaussians-3d-gaussian-splatting-in-the-wild |
| 8 | Wild Gaussians (The Decoder 기사) | https://the-decoder.com/wild-gaussians-new-ai-method-enables-3d-reconstruction-from-user-captured-web-photos/ |
| 9 | SWAG: Splatting in the Wild images with Appearance-conditioned Gaussians (ECCV 2024) | https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09801.pdf |
| 10 | GS-W: Gaussian in the Wild (ECCV 2024) | https://eastbeanzhang.github.io/GS-W/ |
| 11 | SpotlessSplats: Ignoring Distractors in 3D Gaussian Splatting | https://arxiv.org/html/2406.20055v2 |
| 12 | Splatfacto-W: A Nerfstudio Implementation of Gaussian Splatting for Unconstrained Photo Collections | https://arxiv.org/html/2407.12306v1 |
| 13 | WildSplatter: Feed-forward 3D Gaussian Splatting with Appearance Control from Unconstrained Images (2025) | https://arxiv.org/html/2604.21182 |
| 14 | NeRF On-the-go: Exploiting Uncertainty for Distractor-free NeRFs in the Wild (CVPR 2024) | https://openaccess.thecvf.com/content/CVPR2024/papers/ |
| 15 | WildGaussians Hugging Face 모델 페이지 | https://huggingface.co/jkulhanek/wild-gaussians |
