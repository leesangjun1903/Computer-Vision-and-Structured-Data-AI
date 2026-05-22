
# ReconViaGen: Towards Accurate Multi-view 3D Object Reconstruction via Generation

> **논문 정보**
> - 저자: Jiahao Chang, Chongjie Ye, Yushuang Wu, Yuantao Chen, Yidan Zhang, Zhongjin Luo, Chenghong Li, Yihao Zhi, Xiaoguang Han
> - arXiv: [2510.23306](https://arxiv.org/abs/2510.23306) (2025년 10월 27일)
> - 발표: **ICLR 2026** (GitHub 기준)
> - 프로젝트 페이지: https://jiahao620.github.io/reconviagen/
> - GitHub: https://github.com/GAP-LAB-CUHK-SZ/ReconViaGen

---

## 1. 핵심 주장과 주요 기여 요약

### 🔑 핵심 주장

기존 멀티뷰 3D 객체 복원 방법들은 입력 뷰 간의 충분한 중첩(overlap)에 크게 의존하며, 실제 환경에서의 가림(occlusion)이나 희소한 뷰 커버리지는 심각한 복원 불완전성을 초래한다.

이를 해결하기 위해 diffusion 기반 3D 생성 기술이 잠재적 해결책으로 주목받고 있지만, 추론 과정의 확률적(stochastic) 특성이 생성 결과의 정확성과 신뢰성을 제한하여 기존 복원 프레임워크가 이러한 3D 생성 사전(prior)을 통합하기 어렵게 한다.

ReconViaGen은 **정확하고 완전한 멀티뷰 객체 복원**을 위해 강력한 복원 사전(reconstruction prior)을 diffusion 기반 3D 생성기에 통합한 **최초의 프레임워크**이다.

### 🏆 주요 기여

핵심 기여는 복원 사전이 풍부한 이미지 특징을 **멀티뷰 인식(multi-view-aware) diffusion 조건**으로 집약하는 설계에 있다.

생성은 **Coarse-to-Fine 패러다임**을 채택하며, 전역(global) 및 지역(local) 복원 기반 조건을 활용하여 기하(geometry)와 텍스처 모두에서 정확한 결과를 생성하고, 추론 단계에서 지역 잠재 표현(local latent representation)의 denoising 경로를 제약하는 **Rendering-Aware Velocity Compensation** 메커니즘을 제안한다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

diffusion 기반 3D 생성 방법이 높은 일관성을 달성하지 못하는 이유를 세 가지로 분석한다:
**(a)** 멀티뷰 이미지 특징을 조건으로 추출할 때 뷰 간 크로스-뷰 연결 구축 부족,
**(b)** 전역 거친 구조(coarse structure) 생성이 초기 노이즈에 취약,
**(c)** 지역 디테일 생성 시 반복적 denoising의 낮은 제어성으로 입력과 불일치한 지역 디테일 발생.

기존 순수 복원 방법들은 불완전한 결과만 생성할 수 있고, 생성 기반 방법들은 그럴듯하지만 입력 이미지와 강한 불일치를 보이는 완전한 결과를 생성한다.

---

### 2.2 제안하는 방법 및 모델 구조

#### ① 전체 파이프라인

ReconViaGen 프레임워크는 복원(reconstruction)과 생성(generation)을 동시에 수행하며 두 가지 사전 지식을 상호 보완적으로 활용한다. 특히 TRELLIS를 기반으로 생성 사전을 통해 보이지 않는 부분을 그럴듯하게 합성하며, Coarse-to-Fine 복원 파이프라인을 채택한다.

**Stage 1**: 사전 학습된 **VGGT**를 사용하여 전역(global) 및 지역(local) 수준의 복원 기반 멀티뷰 조건을 제공한다. **Stage 2**: 전역 기하 조건(GGC)과 지역 뷰별 조건(PVC)을 각각 **SS(Sparse Structure) Flow Transformer**와 **SLAT(Structured Latent) Flow Transformer**에 입력하여 멀티뷰 인식 생성을 수행한다. 이후 VGGT의 카메라 포즈 추정값을 생성 결과를 이용해 정제(refine)하고, 픽셀 수준 정렬 제약을 추론 단계에서만 적용한다.

#### ② 핵심 구성 요소

**[A] 전역 기하 조건 (Global Geometry Condition, GGC)**

GGC는 Coarse 구조의 예측 정확도를 크게 향상시키며, 거의 모든 지표에서 실질적인 성능 향상을 이끈다.

VGGT에서 추출된 전역 포인트 클라우드 및 기하 특징을 SS Flow Transformer의 조건으로 활용하는 구조이다.

**[B] 뷰별 지역 조건 (Per-View Condition, PVC)**

복원 사전이 풍부한 이미지 특징을 멀티뷰 인식 diffusion 조건으로 집약하는 것이 핵심 설계이다.

VGGT의 멀티뷰 aware 특징맵을 지역 수준 조건으로 SLAT Flow Transformer에 주입한다.

**[C] Rendering-Aware Velocity Compensation (RAVC)**

RAVC는 ReconViaGen 내에서 추론 시 입력 뷰와의 픽셀 수준 정렬을 위해 **지역 잠재 표현의 denoising 경로를 제약**하는 메커니즘이다.

이미지 매칭 기반의 포즈 정제는 TRELLIS의 생성 사전에서 나온 초기 포즈 예측을 효과적으로 수정하여 높은 정확도를 달성하며, 정제된 포즈는 입력 뷰로부터의 픽셀 단위 제약을 가능하게 하여 디테일 정렬을 지원한다.

RAVC의 핵심 아이디어를 수식으로 표현하면, 일반적인 Flow Matching의 velocity field $v_\theta$에 렌더링 기반 보정항 $\Delta v$를 더하는 방식으로 이해할 수 있다:

$$
\tilde{v}_\theta(\mathbf{z}_t, t) = v_\theta(\mathbf{z}_t, t) + \Delta v_{\text{render}}(\mathbf{z}_t, t, \mathcal{I})
$$

여기서:
- $\mathbf{z}_t$: 시각 $t$에서의 latent 표현
- $v_\theta$: 학습된 velocity field (TRELLIS SLAT Flow Transformer)
- $\Delta v_{\text{render}}$: 입력 이미지 $\mathcal{I}$와의 렌더링 차이에서 계산된 보정 속도항
- $\mathcal{I}$: 입력 멀티뷰 이미지 집합

> ⚠️ **주의**: 위 수식 표현은 논문의 아이디어를 기반으로 개념적으로 재구성한 것입니다. 논문 본문의 정확한 수식 표기는 [arXiv PDF](https://arxiv.org/pdf/2510.23306)를 직접 확인하시기 바랍니다.

#### ③ 카메라 포즈 정제

VGGT로부터 추정된 거친 포즈를 2단계 생성 결과를 이용해 정제하며, 새로운 Rendering-Aware Velocity Compensation 메커니즘으로 입력 뷰와의 픽셀 정렬을 강제한다. 입력 이미지와 추정된 카메라 포즈가 결합되어 사용된다.

포즈 정제에는 **PnP 솔버** (Lepetit et al., 2009)와 **RANSAC** (Fischler & Bolles, 1981)이 활용된다.

---

### 2.3 성능 향상

예를 들어, GGC를 기준선(TRELLIS-M)에 통합하면 Dora-bench 데이터셋에서 PSNR이 16.706 → **20.462**, SSIM이 0.882 → **0.894**, F-score가 0.843 → **0.941**로 향상되고, LPIPS는 0.111 → 0.102, CD는 0.144 → 0.093으로 감소한다.

Dora-bench 및 OmniObject3D 데이터셋에서의 광범위한 실험을 통해 전역 형상 정확도, 완전성, 지역 디테일 충실도에서 **최첨단(SOTA) 성능**을 입증한다.

후속 연구인 Mix3R와 비교했을 때도 ReconViaGen은 비견할만한 성능을 보이며, 다른 방법들은 일반적으로 기하 또는 텍스처 왜곡을 겪는다.

---

### 2.4 한계

논문에서 명시적으로 기술된 한계는 다음과 같이 파악된다:

- 후속 연구 Mix3R의 분석에 따르면, ReconViaGen은 VGGT 특징을 생성 모델에 주입하는 **단방향(one-way) 과정**이어서, 두 사전 지식 간의 상호 이익이 제한된다는 점이 지적된다.

- 추론 시 16장의 이미지를 처리할 때 **최소 18GB~24GB의 VRAM**이 필요하여, 경량화된 환경에서의 활용에 제한이 있다.

- TRELLIS라는 특정 생성 모델에 아키텍처적으로 의존하고 있어, 다른 생성 프레임워크로의 이전성(transferability)이 아직 불분명하다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 위한 설계적 강점

Stage 1에서 **사전 학습된 VGGT**를 그대로 활용하여 복원 기반 특징을 추출함으로써, 다양한 객체 카테고리와 입력 뷰 수에 걸쳐 강건한 기하 정보를 제공한다.

입력 이미지 수가 달라지는 상황에서도 ReconViaGen의 복원 품질 스케일링을 평가하여, 다양한 뷰 개수 조건에서의 일반화 가능성을 검증하고 있다.

### 3.2 Wild 환경에서의 일반화

in-the-wild 멀티뷰 이미지에 대한 정성적(qualitative) 성능을 다양한 객체 샘플에서 비교 평가하여, 실제 환경 적용 가능성을 탐색하고 있다.

### 3.3 확장성: TRELLIS.2와의 결합

TRELLIS.2에 대한 효과적인 멀티뷰 융합 전략을 제안하고, ReconViaGen과 TRELLIS.2를 결합하여 **고해상도 메쉬 및 PBR 재질 생성**이 가능하도록 확장함으로써, 일반화 능력의 향상 가능성을 보여준다.

### 3.4 일반화의 이론적 근거

VGGT와 같이 대규모 멀티뷰 데이터로 사전 학습된 모델을 복원 prior로 사용하기 때문에, 학습 데이터에 포함되지 않은 객체 카테고리나 희소 뷰 조건에서도 강건한 특징 추출이 기대된다. 다만 이를 체계적으로 검증한 실험(예: zero-shot 카테고리 일반화)은 현재 논문 수준에서는 명시적으로 제시되어 있지 않다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

| 관점 | 내용 |
|------|------|
| **패러다임 전환** | 복원(Reconstruction)과 생성(Generation)을 단순 후처리 관계가 아닌 하나의 프레임워크 내에서 상호 보완적으로 결합하는 새로운 패러다임 제시 |
| **후속 연구 촉진** | Mix3R는 ReconViaGen의 단방향 특징 주입의 한계를 극복하기 위해 feed-forward 복원 모델과 3D 생성 모델을 상호 이익적으로 결합하는 **Mixture-of-Transformers(MoT)** 구조를 제안한다. |
| **응용 확장** | VR, AR, 3D 모델링 등 멀티뷰 3D 복원의 광범위한 응용 분야에 직접적인 기여 가능성이 있다. |
| **인용 및 영향력** | ForeHOI 등 다수의 후속 연구에서 이미 핵심 구성 요소로 활용되고 있다. |

### 4.2 앞으로 연구 시 고려할 점

1. **상호적 prior 융합 설계**
   Mix3R가 제안한 것처럼, Feed-forward 복원 모델과 3D 생성 모델 간 상호 정보 교환을 통해 두 가지 prior가 상호 보완적으로 작동하도록 하는 설계가 중요하다.

2. **Diffusion 기반 생성 모델의 결정론적 제어**
   RAVC 메커니즘이 Inference-time에만 작동한다는 점에서, 학습 단계에서부터 일관성을 내재화하는 방법 연구가 필요하다.

3. **경량화 및 효율성**
   추론 시 최소 18GB VRAM을 소요하는 구조는 실용적 배포에 장벽이 되므로, 모델 경량화(distillation, pruning) 및 추론 속도 개선 연구가 필요하다.

4. **PBR 재질 및 고해상도 지원**
   TRELLIS.2와의 결합을 통해 고해상도 메쉬 및 PBR 재질 생성 가능성이 열려있으며, 이를 보다 체계적으로 통합하는 연구가 유망하다.

5. **Zero-shot 일반화 및 Out-of-distribution 강건성**
   학습 데이터 분포 밖(out-of-distribution)의 객체에 대한 일반화 실험이 추가적으로 필요하며, 특히 생물체, 유연 물체 등 비정형 객체에서의 성능 검증이 요구된다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 방법 | 특징 | 한계 |
|------|------|------|------|
| **NeRF** (Mildenhall et al., 2020) | Neural Radiance Field | 고품질 Novel View Synthesis | 밀집 뷰 필요, 느린 최적화 |
| **3DGS** (Kerbl et al., 2023) | 3D Gaussian Splatting | 실시간 렌더링, 빠른 최적화 | 희소 뷰에서 아티팩트 발생 |
| **VGGT** (Wang et al., 2024) | Feed-forward Reconstruction | 빠른 추론, 카메라 포즈 추정 | 보이지 않는 영역 복원 불가 |
| **TRELLIS** (Xiang et al., 2024) | Diffusion-based 3D Generation | 완전한 3D 생성 | 입력과의 일관성 부족 |
| **ReconViaGen** (Chang et al., 2025) | Recon Prior + Diffusion Generation | 완전성 + 정확성 동시 달성 | 단방향 prior 주입, 높은 VRAM |
| **Mix3R** (2025) | MoT: Recon ↔ Generation 상호 융합 | 복원과 생성 prior 간 상호 정보 교환으로 상호 이익적 정렬 달성 | 더 복잡한 설계 |

---

## 참고 자료 (출처)

1. **논문 원문 (arXiv)**: Chang et al., "ReconViaGen: Towards Accurate Multi-view 3D Object Reconstruction via Generation," arXiv:2510.23306, 2025. https://arxiv.org/abs/2510.23306
2. **논문 HTML 전문 (arXiv)**: https://arxiv.org/html/2510.23306v1
3. **프로젝트 공식 페이지**: https://jiahao620.github.io/reconviagen/
4. **GitHub 공식 코드**: https://github.com/GAP-LAB-CUHK-SZ/ReconViaGen
5. **HuggingFace 모델 페이지**: https://huggingface.co/Stable-X/trellis-vggt-v0-1
6. **OpenReview (ICLR 2026)**: https://openreview.net/forum?id=z0QLeooEEf
7. **Liner.com Quick Review**: https://liner.com/review/reconviagen-towards-accurate-multiview-3d-object-reconstruction-via-generation
8. **Mix3R (비교 후속 연구)**: arXiv:2605.03359, https://arxiv.org/html/2602.06226
9. **ForeHOI (인용 사례)**: arXiv:2602.06226, https://arxiv.org/html/2602.06226

> ⚠️ **정확도 관련 고지**: RAVC의 구체적 수식, GGC/PVC의 세부 구현 수식은 arXiv 전문 PDF에 상세히 기술되어 있으나, 본 답변에서 웹 검색으로 확인된 수식 표현은 개념적 재구성이 포함될 수 있습니다. 완전한 정확도를 위해서는 [논문 PDF 원문](https://arxiv.org/pdf/2510.23306)을 직접 참조하시기를 권장합니다.
