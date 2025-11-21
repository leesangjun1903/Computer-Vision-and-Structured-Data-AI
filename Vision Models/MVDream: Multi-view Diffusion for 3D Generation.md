
# MVDream: Multi-view Diffusion for 3D Generation

## 1. 핵심 주장 및 주요 기여

MVDream은 ByteDance와 UC San Diego 연구팀이 제시한 혁신적인 다중 시점 확산 모델입니다. 이 논문의 핵심 주장은 **텍스트 프롬프트로부터 일관된 다중 시점 이미지를 동시에 생성할 수 있는 확산 모델이 이미 3D의 형태에 대한 암묵적인 선험지식을 내포하고 있다**는 것입니다.[1]

주요 기여는 다음과 같습니다:

- **2D와 3D 데이터의 결합 학습**: 대규모 2D 이미지-텍스트 쌍(LAION)과 3D 렌더링 데이터(Objaverse)를 함께 학습하여, 2D 확산 모델의 일반화 능력과 3D 렌더링의 일관성을 동시에 확보했습니다.[1]

- **멀티 페이스 야누스 문제 해결**: 기존의 2D-리프팅 방법들이 고통받던 "한 객체가 여러 얼굴을 가지는" 문제와 시점 간 콘텐츠 드리프트 문제를 획기적으로 완화했습니다.[1]

- **표현-무관한 3D 선험**: 제안하는 모델은 특정 3D 표현(NeRF, mesh 등)에 독립적으로 작동하는 범용적인 3D 선험으로 기능합니다.[1]

- **Score Distillation Sampling(SDS) 적용**: 다중 시점 감독을 통해 기존 방법 대비 현저히 안정적인 3D 생성 결과를 달성했습니다.[1]

***

## 2. 해결하는 문제와 제안 방법

### 2.1 문제점 분석

기존 3D 생성 방법들은 세 가지 유형으로 분류됩니다:[1]

1. **템플릿 기반 파이프라인**: 제한된 3D 모델로 인해 단순한 위상 구조의 객체만 생성 가능
2. **3D 생성 모델**: 3D 데이터 부족으로 일반화 성능 제한
3. **2D-리프팅 방법**: 2D 확산 모델의 단일 이미지 관점에 의존하여 다중 시점 일관성 부족

2D-리프팅 방법의 구체적인 문제점:[1]

- **멀티 페이스 야누스 문제**: 같은 객체가 다른 각도에서 여러 얼굴을 가짐
- **콘텐츠 드리프트**: 시점 변경 시 콘텐츠가 일관되지 않게 변화

### 2.2 제안 방법론

#### 2.2.1 팽창된 3D 자기주의 메커니즘

기존 2D 자기주의를 3D로 확장하는 혁신적 접근:[1]

$$\text{Attention} = \text{softmax}\left(\frac{Q_i \cdot K_j^T}{\sqrt{d_k}}\right) V_j$$

여기서 입력 텐서의 형태를 $B \times F \times H \times W \times C$에서 $B \times FHW \times C$로 재구성합니다. 이렇게 하면 모든 시점의 특징이 자기주의 연산에서 상호작용하게 되어 강력한 다중 시점 일관성을 달성합니다.[1]

시간적 주의(temporal attention)는 작동하지 않습니다. 이는 대응하는 픽셀이 서로 다른 시점에서 멀리 떨어져 있을 수 있기 때문입니다.[1]

#### 2.2.2 카메라 임베딩

카메라 매개변수는 2층 MLP를 통해 임베딩되어 시간 임베딩에 잔차로 추가됩니다:[1]

$$e_{\text{camera}} = \text{MLP}(c) + e_{\text{time}}$$

여기서 $c \in \mathbb{R}^{F \times 16}$는 외부 카메라 매개변수입니다. 이 설계는 카메라 정보가 텍스트 조건과 덜 얽히게 하여 더욱 강력한 결과를 제공합니다.[1]

#### 2.2.3 훈련 손실 함수

다중 시점 확산 손실:[1]

$$\mathcal{L}_{\text{MV}}(\theta, X, X_{\text{mv}}) = \mathbb{E}_{x, y, c, t, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t; y, c, t)\|_2^2\right]$$

여기서:
- $x_t$: 랜덤 잡음으로부터 생성된 잡음이 포함된 이미지
- $y$: 텍스트 조건
- $c$: 카메라 조건 (2D 데이터의 경우 비어있음)
- $\epsilon_\theta$: 다중 시점 확산 모델

훈련 중 30% 확률로 3D 주의와 카메라 임베딩을 비활성화하고 2D 텍스트-이미지 모델로 작동하도록 하여 일반화를 유지합니다.[1]

### 2.3 텍스트-3D 생성

#### Score Distillation Sampling 수정

기존 SDS 손실은 다음과 같이 표현됩니다:[1]

$$\nabla_\phi \mathcal{L}_{\text{SDS}}(\theta, x = g(\phi)) = \mathbb{E}_{t, c, \epsilon}\left[w(t)(\epsilon_\theta(x_t; y, c, t) - \epsilon)\frac{\partial x}{\partial \phi}\right]$$

제안하는 $\hat{x}_0$ 재구성 손실:[1]

$$\mathcal{L}_{\text{SDS}}(\phi, x = g(\phi)) = \mathbb{E}_{t, c, \epsilon}\left[\|x - \hat{x}_0\|_2^2\right]$$

여기서 $\hat{x}_0$은 다음과 같이 추정됩니다:[1]

$$\hat{x}_0 = x_t - \sigma_t \frac{\epsilon_\theta}{\alpha_t} = x + \frac{\sigma_t}{\alpha_t}(\epsilon - \epsilon_\theta)$$

이는 원래 SDS와 동등하며, $w(t) = 2\sigma_t/\alpha_t$입니다. 이 공식화는 동적 임계값(dynamic thresholding) 또는 CFG 재조정 같은 트릭을 직접 $\hat{x}_0$에 적용할 수 있게 합니다.[1]

#### Multi-view DreamBooth

DreamBooth 손실 함수:[1]

$$\mathcal{L}_{\text{DB}}(\theta, X_{\text{id}}) = \mathcal{L}_{\text{LDM}}(X_{\text{id}}) + \lambda \frac{\|\theta - \theta_0\|_1}{N_\theta}$$

여기서:
- $\theta_0$: 원래 다중 시점 확산 모델의 매개변수
- $N_\theta$: 총 매개변수 개수
- $\lambda = 1$: 균형 매개변수

***

## 3. 모델 구조

### 3.1 아키텍처 개요

MVDream은 Stable Diffusion v2.1을 기반으로 합니다. 핵심 수정 사항은 두 가지입니다:[1]

1. **자기주의를 2D에서 3D로 변환**: 모든 시점의 특징을 단일 자기주의 연산에서 상호작용하도록 합니다.
2. **카메라 임베딩 추가**: 각 시점에 대한 카메라 정보를 명시적으로 인코딩합니다.

### 3.2 훈련 파이프라인

**이중 모드 훈련 전략:**[1]

- **이미지 모드** (30% 확률): 2D 자기주의, 카메라 임베딩 비활성화 → LAION 데이터에 대한 일반화
- **다중 시점 모드** (70% 확률): 3D 자기주의, 카메라 임베딩 활성화 → Objaverse 렌더링 데이터

이 이중 학습 전략이 일반화와 일관성의 균형을 맞추는 핵심입니다.[1]

***

## 4. 성능 향상 및 실험 결과

### 4.1 다중 시점 이미지 생성 평가

**정량적 결과 (Table 1):**[1]

| 모델 | FID ↓ | IS ↑ | CLIP ↑ |
|------|-------|------|--------|
| 검증 세트 | N/A | 12.90 ± 0.66 | 30.12 ± 3.15 |
| 3D 데이터만 | 40.38 | 12.33 ± 0.63 | 29.69 ± 3.36 |
| 3D + LAION 2D 데이터 | 39.04 | 12.97 ± 0.60 | 30.38 ± 3.50 |

3D와 2D 데이터 결합이 일반화 성능을 크게 향상시킵니다.[1]

### 4.2 3D 생성 (NeRF) 성능

**사용자 연구 결과:**[1]

- 제안 방법: **78%** 선호
- Prolific Dreamer: 11%
- Text Mesh-IF: 2%
- Magic3D-IF-SD: 8%
- DreamFusion-IF: 1%

사용자는 38명이 40개 프롬프트에 대해 914개 피드백을 제공했으며, 제안 방법이 가장 높은 선호도를 기록했습니다.[1]

### 4.3 주의 메커니즘 비교

**Figure 4의 세 가지 주의 설계:**[1]

- **시간적 주의 (Temporal Attention)**: 콘텐츠 드리프트 심각
- **추가 3D 자기주의 모듈**: 이미지 품질 급격히 하락
- **팽창된 3D 자기주의** (제안): 최고의 일관성과 품질 달성

### 4.4 다중 시점 수에 따른 영향

**Figure 8 결과:**[1]

- 1-view 모델: 심각한 야누스 문제
- 2-view 모델: 멀티 페이스 문제 부분적 감소
- 4-view 모델 (제안): 거의 완벽한 다중 시점 일관성

***

## 5. 일반화 성능 향상 가능성

### 5.1 핵심 일반화 메커니즘

**데이터 다양성 결합의 효과:**[1]

대규모 2D 텍스트-이미지 쌍(LAION)과 제한된 3D 데이터의 결합 학습이 일반화의 핵심 요소입니다. 이는 확산 모델이 2D 데이터에서 의미론적 다양성을 학습하고, 3D 데이터에서 기하학적 일관성을 학습할 수 있게 합니다.[1]

### 5.2 임의 시점 수 확장 가능성

**Figure 11의 발견:**[1]

무작위 시점으로 훈련한 모델은 훈련 시 4개 시점만 사용했음에도 불구하고 추론 시 64개 이상의 시점을 생성할 수 있습니다. 이는 모델이 학습한 기하학적 선험이 매우 일반화 가능함을 시사합니다.

### 5.3 새로운 개념 학습

**Multi-view DreamBooth:**[1]

소수의 2D 이미지(약 4-5개)로부터 새로운 개념을 학습하면서도 다중 시점 일관성을 유지합니다. 이는 기존 DreamBooth3D 대비 훨씬 간단한 단일 단계 프로세스입니다.

### 5.4 스타일 일반화의 한계

논문은 생성된 스타일(조명, 텍처)이 렌더링 데이터셋의 영향을 받는다고 지적합니다. 더 다양하고 현실적인 렌더링이 필요하며, 이는 향후 개선의 주요 방향입니다.[1]

***

## 6. 한계

### 6.1 해상도 제약

현재 모델은 256×256 해상도에서만 작동합니다. 원래 Stable Diffusion은 512×512이므로 이는 기존 모델 대비 감소입니다. 이는 더 큰 확산 모델(SDXL 등)로 이전하여 해결 가능합니다.[1]

### 6.2 기반 모델의 한계

모델의 일반화 성능은 기반이 되는 Stable Diffusion v2.1 자체의 능력에 의해 제한됩니다. 더 강력한 기반 모델 사용이 개선의 관건입니다.[1]

### 6.3 렌더링 다양성

생성되는 스타일이 Objaverse 데이터셋의 렌더링 특성에 영향을 받습니다. 더 다양하고 현실적인 3D 렌더링 데이터가 필요합니다.[1]

***

## 7. 최신 연구 기반 향후 전망 및 고려사항

### 7.1 후속 연구 동향

**다중 시점 확산의 발전 (2024-2025):**

**MVGenMaster (CVPR 2025):** 메트릭 깊이와 카메라 포즈로 워핑된 3D 선험을 활용하여 일반화와 3D 일관성을 대폭 향상시켰습니다. 최대 100개의 새로운 시점을 생성할 수 있으며, MvD-1M이라는 160만 개 장면의 대규모 데이터셋을 소개했습니다.[2]

**Hunyuan3D 1.0 (January 2025):** 2단계 접근으로 4초 내에 다중 시점 RGB를 생성한 후 3D 재구성하는 방식으로, MVDream의 아이디어를 발전시키며 추론 속도를 획기적으로 개선했습니다.[3]

**SV3D (ECCV 2024):** 이미지-비디오 확산 모델을 다중 시점 합성에 적응시켜 포즈 제어 가능성, 다중 시점 일관성, 그리고 강력한 일반화를 동시에 달성했습니다.[4]

**VFusion3D (July 2024):** 비디오 확산 모델의 다중 시점 생성 능력을 활용하여 약 300만 개의 합성 다중 시점 데이터를 생성하고, 이를 통해 대규모 피드포워드 3D 생성 모델을 훈련했습니다.[5]

### 7.2 일반화 성능 향상의 핵심 방향

**대규모 다양한 데이터:** MVGenMaster의 성공은 160만 개 장면의 대규모 정렬된 다중 시점 데이터셋의 구성에 크게 의존합니다. 이는 향후 연구가 더 큰 규모의 3D 데이터 수집에 집중해야 함을 시사합니다.[6]

**3D 기하학적 선험 통합:** 메트릭 깊이, 에피폴라 기하학, 렐레이맵 조건화 등 명시적인 3D 기하학적 선험을 모델에 통합하는 것이 일관성과 일반화를 동시에 향상시킵니다.[7][2]

**계산 효율성:** 확산 모델의 반복적 특성과 고차원 3D 데이터로 인한 계산 비용이 주요 병목입니다. 잠재 공간 확산(latent space diffusion)이나 3D 가우시안 스플래팅 같은 명시적 표현으로의 전환이 효율성을 개선합니다.[8][9]

**다중 모드 융합:** 텍스트, 이미지, 스케치, 깊이 맵 등 다양한 조건을 통합하는 멀티모달 확산 모델이 더 강력한 일반화를 제공합니다.[9]

### 7.3 3D 구조 일관성 개선

**깊이 기반 다중 시점 확산:** MVDD는 다중 시점 깊이 표현을 도입하여 생성 공간의 차원을 감소시키고 에피폴라 직선 분할 주의를 통해 3D 일관성을 강제합니다.[10]

**기하학-텍스처 분리:** RichDreamer는 표면의 기하학과 텍스처, 조명을 분리하여 생성 품질을 개선하고, MVDream의 다중 시점 일관성과 결합하여 더욱 강력한 결과를 달성합니다.[11]

### 7.4 실무적 고려사항

**렌더링 리얼리즘:** Objaverse의 렌더링 스타일이 최종 생성 결과의 스타일을 크게 영향을 미치므로, 더욱 사실적이고 다양한 렌더링 데이터의 수집이 중요합니다.[1]

**해상도 확장:** SDXL 같은 고해상도 기반 모델로 전환하여 최대 1024×1024 해상도에서 고충실도 3D 자산 생성이 가능합니다.[12]

**실시간 생성:** Hunyuan3D의 4초 생성 시간은 산업 응용을 위한 필수 요구사항이며, 향후 연구는 더욱 빠른 추론을 목표로 합니다.[3]

***

## 8. 결론

MVDream은 **텍스트 프롬프트로부터 일관된 다중 시점 이미지를 생성하는 첫 번째 실용적 확산 모델**로서, 3D 생성 분야에 패러다임 변화를 가져왔습니다. 팽창된 3D 자기주의, 카메라 임베딩, 이중 모드 훈련 전략의 조합은 2D 모델의 일반화 능력과 3D 렌더링의 일관성을 성공적으로 통합합니다.[1]

논문의 주요 강점은 기존 2D-리프팅 방법의 "멀티 페이스 야누스" 문제를 획기적으로 해결하고, 매우 간단한 재구성 손실 함수만으로도 안정적인 3D 생성이 가능함을 보여준 점입니다.[1]

향후 연구는 **대규모 다양한 3D 데이터**, **명시적 3D 기하학적 선험 통합**, **계산 효율성 개선**, **해상도 확장**에 집중할 것으로 예상됩니다. MVGenMaster, Hunyuan3D 같은 최신 연구들은 MVDream의 기본 아이디어를 바탕으로 더욱 강력한 성능을 달성하고 있으며, 이는 이 분야의 지속적인 발전을 시사합니다.[2][9][3]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2c637ad3-3e75-43b8-95c4-5e37555604fc/2308.16512v4.pdf)
[2](https://arxiv.org/html/2411.16157)
[3](https://arxiv.org/html/2411.02293)
[4](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00150.pdf)
[5](http://arxiv.org/pdf/2403.12034.pdf)
[6](https://arxiv.org/abs/2411.16157)
[7](http://arxiv.org/abs/2501.18804)
[8](https://arxiv.org/html/2403.12019v2)
[9](https://arxiv.org/html/2410.04738v3)
[10](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02081.pdf)
[11](https://openaccess.thecvf.com/content/CVPR2024/papers/Qiu_RichDreamer_A_Generalizable_Normal-Depth_Diffusion_Model_for_Detail_Richness_in_CVPR_2024_paper.pdf)
[12](https://arxiv.org/html/2503.21694)
[13](https://arxiv.org/html/2308.16512v4)
[14](https://arxiv.org/html/2404.17419)
[15](https://arxiv.org/pdf/2311.14494.pdf)
[16](https://arxiv.org/pdf/2311.07885.pdf)
[17](https://arxiv.org/html/2405.03894v1)
[18](http://arxiv.org/pdf/2404.03656.pdf)
[19](https://jang-inspiration.com/mv-dream)
[20](https://dreamfusion3d.github.io)
[21](https://proceedings.iclr.cc/paper_files/paper/2024/file/adbe936993aa7cf41e45054d8b72f183-Paper-Conference.pdf)
[22](https://arxiv.org/html/2502.19716v1)
[23](https://pure.kaist.ac.kr/en/publications/let-2d-diffusion-model-know-3d-consistency-for-robust-text-to-3d-)
[24](https://velog.io/@deepdiv/MVDream-Multi-view-Diffusion-for-3D-Generation)
[25](http://openaccess.thecvf.com/content/CVPR2025/papers/Wang_MEAT_Multiview_Diffusion_Model_for_Human_Generation_on_Megapixels_with_CVPR_2025_paper.pdf)
[26](https://arxiv.org/abs/2412.05929)
[27](https://arxiv.org/abs/2308.16512)
[28](https://arxiv.org/html/2503.06136v1)
[29](https://arxiv.org/abs/2402.14253)
[30](https://openreview.net/forum?id=MN3yH2ovHb)
[31](https://pmc.ncbi.nlm.nih.gov/articles/PMC12473764/)
[32](https://github.com/ewrfcas/MVGenMaster)
[33](https://arxiv.org/html/2311.16918v2)
[34](https://cg.cs.tsinghua.edu.cn/papers/CVMJ-2025-diffusion.pdf)
