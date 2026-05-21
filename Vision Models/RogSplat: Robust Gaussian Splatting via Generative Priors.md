
# RogSplat: Robust Gaussian Splatting via Generative Priors

---

## 📌 참고 출처 (Reference)

| # | 출처 |
|---|------|
| [1] | Kong, H., Yang, X., Wang, X. **"RogSplat: Robust Gaussian Splatting via Generative Priors"**, ICCV 2025, pp. 25735–25745. https://openaccess.thecvf.com/content/ICCV2025/html/Kong_RogSplat_Robust_Gaussian_Splatting_via_Generative_Priors_ICCV_2025_paper.html |
| [2] | ICCV 2025 Poster page: https://iccv.thecvf.com/virtual/2025/poster/238 |
| [3] | Hanyang Kong 저자 홈페이지: https://hyokong.github.io/ |
| [4] | Kerbl et al. **"3D Gaussian Splatting for Real-Time Radiance Field Rendering"**, SIGGRAPH 2023. |
| [5] | Sabour et al. **"RobustNeRF: Ignoring Distractors with Robust Losses"**, CVPR 2023. |
| [6] | Ren et al. **"NeRF On-the-go: Exploiting Uncertainty for Distractor-free NeRFs in the Wild"**, CVPR 2024. |
| [7] | Kulhanek & Sattler. **"WildGaussians: 3D Gaussian Splatting in the Wild"**, arXiv 2024. https://arxiv.org/pdf/2407.08447 |
| [8] | Xiao et al. **"RoGSplat: Learning Robust Generalizable Human Gaussian Splatting from Sparse Multi-View Images"**, CVPR 2025. https://arxiv.org/abs/2503.14198 *(주의: 이는 별개의 논문)* |
| [9] | Schöps et al. **"Robust 3D Gaussian Splatting for Novel View Synthesis in Presence of Distractors"**, arXiv 2408.11697, 2024. |

---

## 1. 핵심 주장 및 주요 기여 (요약)

3D Gaussian Splatting(3DGS)은 고품질 3D 재구성 및 렌더링을 위한 효율적인 표현으로 주목받아 왔으나, 입력 이미지들 간의 기하학적 일관성(geometric consistency) 가정에 크게 의존한다. 실세계에서는 폐색(occlusion), 동적 객체(dynamic objects), 카메라 블러(camera blur) 등으로 인해 이 가정이 위반되어 재구성 아티팩트 및 렌더링 부정확성이 발생한다.

이에 대응하여 **RogSplat**이 제안되었으며, 핵심 기여는 다음과 같습니다:

| 기여 | 설명 |
|------|------|
| **강건한 프레임워크** | 생성 모델(generative models)을 활용하여 3DGS의 신뢰성을 향상시키는 강건한 프레임워크 RogSplat을 제안 |
| **아웃라이어 탐지** | 제안된 Fused Features를 사용하여 아웃라이어(outlier) 영역을 먼저 탐지 |
| **RF-Refiner** | 탐지된 영역을 RF-Refiner가 정확하게 인페인팅(inpainting)하여, 폐색된 영역의 신뢰할 수 있는 재구성을 보장하면서 가시 영역의 무결성을 보존 |
| **실험 성능** | RobustNeRF 및 NeRF-on-the-go 데이터셋에서 state-of-the-art 재구성 품질을 달성하며, 동적 객체가 포함된 도전적인 실세계 시나리오에서 기존 방법을 크게 능가 |

---

## 2. 해결 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

3DGS는 뛰어난 렌더링 품질과 속도를 자랑하지만, 입력 이미지들 간의 기하학적 일관성 가정에 과도하게 의존한다. 실세계에서 이 가정이 위반되는 경우(폐색, 동적 객체, 카메라 블러 등)에는 재구성 아티팩트와 렌더링 부정확성으로 이어진다.

구체적으로 해결해야 할 문제는 다음 세 가지입니다:

1. **동적 객체(Dynamic Objects)**: 장면 최적화 시 움직이는 물체가 Gaussian 표현을 오염
2. **폐색(Occlusion)**: 특정 시점에서 보이지 않는 영역의 잘못된 재구성
3. **카메라 블러(Camera Blur)**: 흔들린 이미지로 인한 기하학적 불일치

---

### 2-2. 제안 방법

#### 3DGS 기본 렌더링 공식 (배경 지식)

3DGS에서 각 Gaussian은 다음 파라미터로 정의됩니다:

$$\mathcal{G} = \{\mu_i, \Sigma_i, \alpha_i, c_i\}_{i=1}^{N}$$

여기서 $\mu_i \in \mathbb{R}^3$은 평균(위치), $\Sigma_i \in \mathbb{R}^{3\times3}$은 공분산 행렬(형태), $\alpha_i$는 불투명도(opacity), $c_i$는 색상(SH 계수)입니다.

3DGS의 2D 투영 렌더링은 다음과 같이 이루어집니다:

$$C(\mathbf{r}) = \sum_{i \in \mathcal{N}} c_i \cdot \alpha_i \cdot \prod_{j < i}(1-\alpha_j)$$

표준 재구성 손실은 다음과 같습니다:

```math
\mathcal{L}_{3DGS} = (1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{D\text{-}SSIM}
```

**그러나** 동적 객체나 폐색이 존재할 경우, 특정 픽셀의 렌더링 결과 $\hat{C}(\mathbf{r})$와 실제 관측값 $C_{gt}(\mathbf{r})$ 사이에 체계적인 오차가 발생하여 위 손실이 Gaussian을 잘못된 방향으로 최적화하게 됩니다.

#### RogSplat의 핵심 파이프라인

RogSplat은 비정형 장면(unstructured scenes)의 최적화 과정에서 폐색 영역을 탐지하고 수정하며, 아웃라이어 영역을 먼저 탐지한 후 정확하게 인페인팅한다.

**Step 1 — Fused Feature 기반 아웃라이어 탐지:**

렌더링 잔차(residual)와 특징 맵을 융합하여 아웃라이어 마스크 $\mathbf{M}$을 추정합니다:

$$\mathbf{M} = f_{\text{detect}}\left(\mathbf{F}_{\text{render}}, \mathbf{F}_{\text{semantic}}\right)$$

여기서 $\mathbf{F}\_{\text{render}}$는 렌더링 기반 특징, $\mathbf{F}_{\text{semantic}}$는 사전학습된 특징 추출기(예: DINO, CLIP 등)에서 추출한 의미론적 특징입니다 *(구체적 구조는 논문 전문 미공개로 확인 불가)*.

**Step 2 — RF-Refiner를 통한 생성적 인페인팅:**

스마트한 아웃라이어 정리(Smart Outlier Cleanup): 손상된 영역을 탐지하고 생성적 리파이너(generative refiner)로 인페인팅한다.

탐지된 마스크 영역에 대해 생성 모델이 신뢰할 수 있는 픽셀 값을 생성:

$$\hat{I}_{\text{clean}} = f_{\text{RF-Refiner}}\left(I_{\text{corrupted}}, \mathbf{M}\right)$$

**Step 3 — 정제된 이미지로 3DGS 재최적화:**

$$\mathcal{L}_{\text{RogSplat}} = \sum_{\mathbf{r}} (1-\mathbf{M}(\mathbf{r})) \cdot \mathcal{L}_{\text{pixel}}(\hat{C}(\mathbf{r}), C_{gt}(\mathbf{r})) + \mathbf{M}(\mathbf{r}) \cdot \mathcal{L}_{\text{pixel}}(\hat{C}(\mathbf{r}), \hat{I}_{\text{clean}}(\mathbf{r}))$$

*(위 수식은 논문의 컨셉을 구현 방향에 따라 표현한 것이며, 실제 논문의 수식과 다를 수 있음을 명시합니다.)*

---

### 2-3. 모델 구조

공개된 정보를 기반으로 파악된 모델 구성 요소:

```
입력: 다중 뷰 이미지 (+ 카메라 파라미터)
    ↓
[3DGS 초기 최적화]
    ↓
[Fused Feature 기반 아웃라이어 탐지 모듈]
  - 렌더링 잔차 맵 계산
  - 의미론적 특징 융합
  - 이진 마스크 M 생성
    ↓
[RF-Refiner (Generative Inpainting)]
  - 마스크 영역을 생성 모델로 채움
  - 가시 영역 무결성 보존
    ↓
[정제된 이미지로 3DGS 재최적화]
    ↓
출력: 강건한 3D Gaussian 표현
```

RogSplat은 폐색, 모션 블러 등 실세계 문제를 수정하며 실제 환경(in the Wild)에서 3DGS가 작동할 수 있도록 한다.

---

### 2-4. 성능 향상

광범위한 실험을 통해 RogSplat은 RobustNeRF 및 NeRF-on-the-go 데이터셋에서 state-of-the-art 재구성 품질을 달성하며, 동적 객체를 포함한 도전적인 실세계 시나리오에서 기존 방법을 크게 능가함을 증명했다.

비교 대상으로는 다음 방법들이 포함된 것으로 파악됩니다:

| 비교 방법 | 특징 |
|-----------|------|
| Vanilla 3DGS (Kerbl et al., 2023) | 기준선 (baseline) |
| RobustNeRF (Sabour et al., 2023) | IRLS 기반 NeRF 강건화 |
| NeRF-on-the-go (Ren et al., 2024) | DINO 특징 기반 불확실성 추정 |
| WildGaussians (Kulhanek, 2024) | 3DGS + 외관 모델링 |

---

### 2-5. 한계 (논문 전문 미공개로 추정 가능한 수준 기술)

논문 전문이 공개되지 않아 명시된 한계를 직접 인용하기 어렵습니다. 그러나 유사 방법들의 공통적 한계와 방법론적 특성으로부터 다음과 같은 한계를 유추할 수 있습니다:

1. **추가적인 생성 모델 의존성**: RF-Refiner라는 별도의 생성 모델이 필요하여 전체 파이프라인의 복잡성이 증가
2. **계산 비용 증가**: 인페인팅 단계가 추가되어 순수 3DGS 대비 학습 시간이 증가할 가능성
3. **극심한 동적 장면**: 대부분의 픽셀이 동적인 장면에서는 마스크 추정 자체가 어려울 수 있음
4. **정적 장면 가정**: 최종 렌더링 대상은 여전히 정적 장면으로, 동적 장면 렌더링 자체가 목적이 아님

---

## 3. 모델의 일반화 성능 향상 가능성

RogSplat의 일반화 성능 향상 가능성은 다음 측면에서 분석됩니다:

### 3-1. 생성 모델(Generative Prior)의 일반화 기여

생성 모델을 활용하여 3DGS의 신뢰성을 향상시키는 접근 방식은 사전학습된 생성 모델이 다양한 유형의 폐색 패턴에 대한 시각적 사전 지식(visual prior)을 인코딩하고 있음을 활용합니다. 이는 특정 데이터셋에 과적합되지 않고 다양한 실세계 시나리오에 적용될 수 있는 가능성을 제공합니다.

$$\text{일반화 능력} \propto \text{생성 모델의 학습 데이터 다양성}$$

### 3-2. Fused Feature의 일반화 기여

아웃라이어 영역을 Fused Features로 탐지하고 RF-Refiner로 인페인팅하여, 폐색 영역의 신뢰할 수 있는 재구성을 보장하면서 가시 영역의 무결성을 보존한다.

다양한 모달리티의 특징을 융합함으로써, 단일 특징 공간에 의존하는 방법보다 다양한 장면 유형에 강건할 수 있습니다.

### 3-3. 비교 관점: WildGaussians와의 차이

3DGS와 NeRF 모두 잘 통제된 3D 장면에서는 탁월하지만, 폐색·동적 객체·조명 변화를 특징으로 하는 실세계 데이터는 여전히 도전적이다. NeRF는 이미지별 임베딩 벡터를 통해 쉽게 적응할 수 있지만, 3DGS는 명시적 표현과 공유 파라미터의 부재로 어려움을 겪는다.

RogSplat은 이 문제를 **생성적 인페인팅**으로 해결하려는 방향으로, WildGaussians가 외관 모델링 모듈을 별도로 학습하는 것과 대조됩니다.

| 방법 | 일반화 전략 |
|------|-------------|
| RobustNeRF | IRLS robust loss → 장면별 최적화, 일반화 제한 |
| NeRF On-the-go | DINO v2 특징으로 불확실성 예측, 긴 훈련 시간 |
| WildGaussians | DINO 특징 + 외관 모델링 모듈로 SOTA 달성 |
| **RogSplat** | 생성 모델(RF-Refiner)로 폐색 영역 직접 복원, 일반화 기대 |

### 3-4. 일반화 한계와 가능성

**가능성:**
- 대규모 사전학습 생성 모델(예: diffusion model)을 RF-Refiner로 사용할 경우, 다양한 장면 유형과 폐색 패턴에 대한 광범위한 일반화 기대
- 데이터셋-독립적인 탐지 모듈 설계로 임의의 비정형 장면에 적용 가능

**한계:**
- 생성 모델이 훈련된 도메인 외의 장면(예: 의료 영상, 위성 영상)에서는 RF-Refiner 성능이 저하될 가능성
- 장면 최적화(per-scene optimization) 기반이므로, 새로운 장면마다 재최적화 필요

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 연구 | 연도/학회 | 핵심 방법 | 강점 | 약점 |
|------|-----------|-----------|------|------|
| NeRF (Mildenhall et al.) | 2020, ECCV | MLP + volume rendering | 암묵적 표현, 고품질 | 느린 학습/렌더링 |
| 3DGS (Kerbl et al.) | 2023, SIGGRAPH | 3D Gaussian + rasterization | 실시간 렌더링 | 기하학적 일관성 가정 |
| RobustNeRF | 2023, CVPR | IRLS robust loss | 분산자(distractor) 처리 | NeRF 기반, 느림 |
| NeRF-on-the-go | 2024, CVPR | DINO v2 불확실성 | 복잡한 장면 처리 | 긴 학습 시간 |
| Robust 3DGS (Schöps et al., 2024) | 2024, arXiv | SAM 마스크 + neural boundary | 3DGS 대비 +1.9dB, RobustNeRF 대비 +4.3dB PSNR | SAM 의존, 분산자 유형 제한 |
| WildGaussians (Kulhanek, 2024) | 2024, arXiv | DINO 특징 + 외관 모델링 | SOTA 결과, 빠른 최적화 | 조명 변화에 특화 |
| **RogSplat (Kong et al.)** | **2025, ICCV** | Fused Features + RF-Refiner | 생성 prior 활용, 폐색 인페인팅 | 추가 생성 모델 필요 |

---

## 5. 앞으로의 연구에 미치는 영향 및 연구 시 고려할 점

### 5-1. 연구에 미치는 영향

**① 생성 모델과 3DGS의 융합 패러다임 확립**

RogSplat은 생성적 Sparse-View Gaussian Splatting 등 동일 연구 그룹의 후속 연구와 함께, 생성 모델(diffusion model, inpainting model)을 3DGS 파이프라인에 통합하는 새로운 연구 방향을 제시합니다. 이는 "3DGS + generative prior"의 조합이 다양한 도전적 장면 재구성 문제에 적용될 수 있음을 시사합니다.

**② 실세계(in-the-wild) 3DGS 연구 활성화**

실세계 문제(폐색, 모션 블러)를 해결하는 RogSplat의 접근 방식은 자율주행, AR/VR, 로봇 내비게이션 등 실세계 응용 분야에서 3DGS 활용 가능성을 넓히는 중요한 발판이 됩니다.

**③ 탐지-복원 (Detect-and-Inpaint) 패러다임**

아웃라이어를 먼저 탐지하고 생성 모델로 복원하는 two-stage 접근법은, 단순히 손실 함수를 강건하게 만드는 기존 방법(RobustNeRF)에 비해 명시적이고 해석 가능한 방향을 제시합니다.

---

### 5-2. 앞으로 연구 시 고려할 점

**① 생성 모델의 일관성 문제**
- RF-Refiner의 인페인팅 결과가 다중 뷰 간에 3D 기하학적으로 일관성을 유지하는지가 핵심 과제입니다. 단순 2D 인페인팅은 각 뷰에서 다른 결과를 생성하여 3DGS 최적화를 방해할 수 있습니다.
- 다중 뷰 일관성을 보장하는 $\mathcal{L}_{\text{multiview-consistency}}$ 같은 손실 설계가 필요합니다.

**② 계산 효율성**
- 생성 모델 추론 비용을 줄이기 위한 경량화 RF-Refiner 설계 또는 단계적(progressive) 적용 전략이 요구됩니다.

**③ 동적 장면으로의 확장**
- 현재는 정적 장면에서 동적 객체를 *제거*하는 방향이나, 동적 객체 *자체*를 함께 재구성하는 방향(4D Gaussian Splatting)으로의 확장이 중요한 후속 연구가 될 것입니다.

**④ 마스크 탐지의 정밀도**
- Fused Feature 기반 탐지가 정밀하지 않을 경우, 실제 정적 배경을 잘못 마스킹하여 재구성 품질이 저하될 수 있습니다. 적응형 임계값(adaptive threshold) 설계가 중요합니다.

**⑤ 벤치마크 다변화**
- RobustNeRF, NeRF-on-the-go 데이터셋 외에도 대규모 야외 장면, 의료 영상, 위성 영상 등 다양한 도메인에서의 평가가 일반화 성능 검증에 필요합니다.

**⑥ 생성 모델 사전학습 도메인 의존성**
- RF-Refiner가 어떤 생성 모델로 초기화되었는지에 따라 특정 도메인에 편향될 수 있으므로, 도메인-범용 생성 prior의 선택이 중요합니다.

---

> ⚠️ **정확도 고지**: 본 답변은 ICCV 2025 공식 abstract, 저자 홈페이지, 관련 연구 논문의 공개 정보에 기반합니다. RogSplat (ICCV 2025) 논문의 **전문(full paper PDF)이 공개 접근되지 않아**, 구체적인 수식, 정량적 수치(PSNR/SSIM), 모델 세부 구조는 확인된 범위에서만 기술하였으며 일부는 방법론적 추론으로 작성되었습니다. 수식은 논문의 공식 수식이 아닌 컨셉 수준의 표현임을 명시합니다. 논문 전문 공개 후 재확인을 강력히 권장합니다.
