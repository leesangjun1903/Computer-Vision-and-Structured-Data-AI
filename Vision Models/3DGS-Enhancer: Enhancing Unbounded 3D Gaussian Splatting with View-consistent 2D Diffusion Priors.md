
# 3DGS-Enhancer: Enhancing Unbounded 3D Gaussian Splatting with View-consistent 2D Diffusion Priors

> **논문 정보**: Liu, Xi, Chaoyi Zhou, and Siyu Huang. "3DGS-Enhancer: Enhancing Unbounded 3D Gaussian Splatting with View-consistent 2D Diffusion Priors." *NeurIPS 2024 (Spotlight)*.
> **arXiv**: [2410.16266](https://arxiv.org/abs/2410.16266) | **프로젝트 페이지**: https://xiliu8006.github.io/3DGS-Enhancer-project/ | **GitHub**: https://github.com/xiliu8006/3DGS-Enhancer

---

## 1. 핵심 주장 및 주요 기여 요약

### 🎯 핵심 주장

Novel-View Synthesis(NVS)에서 3D Gaussian Splatting(3DGS)은 사실적인 렌더링에서 큰 성공을 거뒀으나, sparse input views와 같은 도전적인 환경에서는 언더샘플링 영역의 정보 부족으로 인해 심각한 아티팩트가 발생한다.

3DGS-Enhancer는 비디오 확산 모델(video diffusion model)이라는 뷰 일관성(view-consistent) 2D 생성 프라이어를 활용해 3D Gaussian Splatting의 렌더링 품질을 향상시키는 방법을 제안한다.

### 🏆 주요 기여 (Contributions)

지식 범위 내에서, 이 논문은 **실용적 3DGS 응용에서 널리 발생하는 저품질 3DGS 렌더링 결과를 향상시키는 문제를 최초로 다룬 연구**이다.

3DGS-Enhancer는 **3D 일관성 이미지 복원 작업을 시간적으로 일관된 비디오 생성 문제로 재구성**함으로써, 고품질이고 3D 일관성 있는 이미지를 생성하는 데 강력한 비디오 LDM을 활용한다.

실험에서는 수백 개의 언바운드 장면에 기반한 DL3DV 데이터셋을 활용해 저품질-고품질 이미지 쌍으로 대규모 데이터셋을 생성하여, 새롭게 정의된 3DGS 향상 문제를 포괄적으로 평가한다.

---

## 2. 해결하고자 하는 문제 / 제안 방법 / 모델 구조 / 성능 및 한계

### 2-1. 해결하고자 하는 문제

기존 뷰포인트에서 멀리 떨어진 novel view를 고품질로 렌더링하는 것은 sparse-view 환경에서 매우 어렵다. 입력 뷰가 3개뿐일 때 타원형(ellipsoid-like)이나 텅 빈(hollow) 아티팩트가 뚜렷하게 나타난다. 실제 환경에서 이러한 저품질 렌더링 결과가 흔하게 발생하기 때문에, 실용적 활용을 위해 3DGS의 품질을 향상시키는 것이 필수적이다.

기존 sparse-view NVS 방법들과 달리, 이 접근법은 깊이 추정 네트워크를 통한 depth regularization에 의존하지 않는다. 대신, 저품질 3DGS 모델로부터 렌더링된 이미지를 향상시키기 위해 비디오 확산 모델을 활용하는 순수 2D 시각적 방법을 취한다.

---

### 2-2. 제안하는 방법 및 수식

#### ① 3DGS 기본 표현

장면은 중심 위치(center position), 스케일링 인자(scaling factors), 회전(rotation), 색상(color), 불투명도(opacity) 등의 파라미터로 특성화된 이방성 3D 가우시안 구의 집합으로 표현된다.

각 Gaussian $G_i$의 파라미터:

$$G_i = \{\mu_i \in \mathbb{R}^3,\; r_i \in \mathbb{R}^4,\; s_i \in \mathbb{R}^3,\; \eta_i \in \mathbb{R},\; c_i \in \mathbb{R}^3\}$$

- $\mu_i$: 3D 공간상의 위치
- $r_i$: 쿼터니언 회전(rotation)
- $s_i$: 스케일(scale)
- $\eta_i$: 불투명도(opacity)
- $c_i$: 구면 조화 함수(SH)로 표현된 뷰 의존적 색상

3DGS의 렌더링(alpha compositing):

$$\hat{C}(p) = \sum_{i=1}^{M} c_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)$$

- $\alpha_i$: 픽셀 $p$에서 $i$번째 Gaussian의 불투명도 기여
- $M$: 픽셀에 투영된 Gaussian의 수 (깊이 순 정렬)

#### ② 문제 정의 (Enhancement Problem Formulation)

주어진 입력: reference 이미지 집합 $\{I^{ref}\_1, I^{ref}\_2, \ldots, I^{ref}\_{N_{ref}}\}$ 및 대응하는 카메라 포즈 $\{\mathbf{p}^{ref}\_1, \mathbf{p}^{ref}\_2, \ldots, \mathbf{p}^{ref}\_{N_{ref}}\}$

목표는 초기 3DGS 모델 $\mathcal{G}^{init}$으로부터 렌더링된 저품질 novel view 이미지 $\{\hat{I}^{nov}\_k\}\_{k=1}^{N_{nov}}$를 고품질 이미지 $\{I^{enh}\_k\}\_{k=1}^{N_{nov}}$로 복원하는 것이다:

$$\mathcal{F}_\theta : \{\hat{I}^{nov}_k, I^{ref}_j\}_{k,j} \;\longrightarrow\; \{I^{enh}_k\}_{k=1}^{N_{nov}}$$

#### ③ Video LDM 기반의 뷰 일관성 복원

3D 뷰 일관성 문제를 비디오 생성 프로세스 내의 **시간적 일관성(temporal consistency) 달성 문제로 재구성**하여, 2D 비디오 확산 프라이어를 활용한다.

DDPM 기반 확산 모델의 역방향 프로세스(denoising):

$$p_\theta(\mathbf{z}_{t-1} | \mathbf{z}_t, \mathbf{c}) = \mathcal{N}\!\left(\mathbf{z}_{t-1};\; \mu_\theta(\mathbf{z}_t, t, \mathbf{c}),\; \Sigma_\theta(\mathbf{z}_t, t)\right)$$

- $\mathbf{z}_t$: 시간 $t$에서의 노이즈 잠재 특징(latent feature)
- $\mathbf{c}$: reference 뷰 컨디셔닝 정보
- $\mu_\theta, \Sigma_\theta$: 학습된 평균 및 분산

#### ④ Confidence-Aware 3DGS Fine-tuning

복원된 뷰의 약간의 부정확성이 fine-tuning 과정에서 증폭될 수 있기 때문에, reference 뷰에 더 의존하는 방식을 제안한다. 이를 위해 **confidence-aware 3D Gaussian splatting**을 제안하며, 이는 이미지 수준(image level)과 픽셀 수준(pixel level)의 두 가지 신뢰도를 포함한다.

Fine-tuning 손실 함수:

$$\mathcal{L}_{total} = \lambda^{ref} \mathcal{L}_{ref} + \sum_k w_k^{img} \cdot w_k^{pix} \cdot \mathcal{L}_{enh}(I^{enh}_k)$$

- $\lambda^{ref}$: reference 뷰에 대한 가중치 (항상 높게 유지)
- $w_k^{img}$: **이미지 수준 신뢰도** — reference 뷰로부터 멀수록 높은 신뢰도 부여
- $w_k^{pix}$: **픽셀 수준 신뢰도** — 잘 재구성된 영역의 소부피 Gaussian 밀도 기반

먼 뷰포인트일수록 아티팩트가 적을 가능성이 높으므로, reference 뷰와의 거리를 $[0, 1]$로 정규화하여 더 먼 뷰포인트에 비디오 확산 결과에 대한 높은 신뢰도를 부여한다.

픽셀 수준 신뢰도는 잘 재구성된 영역에서의 소부피 Gaussian의 밀도를 기반으로 하며, 색상 렌더링 파이프라인을 사용해 볼륨을 계산한다. 두 픽셀 및 이미지 수준 신뢰도 전략은 각각 독립적으로 결과를 개선하며, 이 둘의 조합이 최고의 성능을 달성한다.

---

### 2-3. 모델 구조 (Architecture)

3DGS-Enhancer의 핵심은 다음 세 요소로 구성된 **Video LDM**이다:
1. **Image Encoder**: 렌더링된 뷰의 잠재 특징을 인코딩
2. **Video-based Diffusion Model**: 시간적으로 일관된 잠재 특징을 복원
3. **Spatial-Temporal Decoder (STD)**: 원본 렌더링 이미지의 고품질 정보와 복원된 잠재 특징을 효과적으로 통합

비디오 확산 모델은 대부분의 아티팩트를 제거하고, STD 모듈은 세밀하고 고주파 텍스처를 향상시켜 실제에 더 가까운 생생한 novel view 렌더링 결과를 생성한다.

**Spatial-Temporal Decoder**: 색상 이동(color shift) 및 흐림(blurriness)과 같은 문제를 해결하는 기법을 사용하여 향상된 잠재 특징을 출력 이미지로 정제하며, 원본 렌더링 이미지를 효과적으로 통합해 출력 뷰의 품질을 향상시킨다.

전체 파이프라인 흐름:

```
[초기 3DGS 모델] 
      ↓ (Trajectory Interpolation)
[저품질 Novel View 렌더링]
      ↓ (Image Encoder)
[잠재 특징 추출]
      ↓ (Video Diffusion Model + Reference Views)
[시간적으로 일관된 잠재 특징 복원]
      ↓ (Spatial-Temporal Decoder)
[고품질 Enhanced Views]
      ↓ (Confidence-Aware Fine-tuning)
[향상된 3DGS 모델]
```

제안된 3DGS-Enhancer는 **trajectory-free** 방식으로도 적용 가능하여, sparse 뷰로부터 언바운드 장면을 재구성하고 두 알려진 뷰 사이의 보이지 않는 영역에 대한 자연스러운 3D 표현을 생성할 수 있다.

---

### 2-4. 성능 향상

언바운드 장면의 대규모 데이터셋에 대한 광범위한 실험을 통해, 3DGS-Enhancer가 최신(state-of-the-art) 방법들에 비해 우수한 재구성 성능과 고충실도 렌더링 결과를 달성함을 실증하였다.

DL3DV 기반의 수백 개의 언바운드 장면에서 저품질-고품질 이미지 쌍의 대규모 데이터셋으로 실험하였으며, 다양한 도전적인 장면에서 우수한 재구성 성능과 더욱 선명하고 생생한 렌더링 결과를 달성하였다.

이미지 및 픽셀 수준의 신뢰도를 3DGS fine-tuning과 결합함으로써, NVS 향상에서 state-of-the-art 성능을 달성하였다.

---

### 2-5. 한계점

이 방법은 연속적인 보간을 위해 인접 뷰에 의존하기 때문에, **단일 뷰 3D 모델 생성에는 쉽게 적용하기 어렵다**. 또한, confidence-aware 3DGS fine-tuning 전략이 상대적으로 단순하고 직관적이라는 한계가 있다.

3DGS-Enhancer는 두 고품질 reference 뷰 사이의 중간 렌더링 프레임 시퀀스를 향상시키는 작업으로 구성되었으나, **단일 카메라 구성과 순차적 생성 파이프라인은 다중 뷰 또는 다중 카메라 시뮬레이션 환경으로의 확장성을 제한**한다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 현재 일반화 강점

이 방법은 다양한 데이터셋에서 견고한 성능을 보여주며, 다양한 조건에 대한 적응성을 강조하고 few-shot novel view synthesis에 사용되는 기존 방법론들을 효과적으로 개선한다.

V3D와 같은 동시대 연구가 단일 이미지에서 객체 수준의 3DGS 모델 생성에 집중하는 것과 달리, 3DGS-Enhancer는 **기존의 모든 3DGS 모델을 향상시키는 데 초점을 맞추어 언바운드 장면과 같이 더 일반화된 장면에 적용될 수 있다**.

### 3-2. 일반화 성능 향상을 위한 미래 방향

미래에는 confidence map을 비디오 생성 모델에 직접 통합하여 후처리 없이도 실제 3D 세계에 더 부합하는 이미지를 생성하는 방향이 흥미롭다. 동시에, 3DGS의 효율적인 데이터 생성 능력을 활용해 비디오 생성 모델을 위한 대규모 데이터셋을 구축하는 것이 모델의 3D 일관성을 향상시키는 좋은 기회가 될 것이다.

### 3-3. 일반화 관련 후속 연구 동향

SetDiff와 같은 후속 연구는 3DGS-Enhancer의 한계를 보완하는 방향으로, 다양한 카메라 포즈와 타임스텝에서 렌더링된 임의 길이·임의 순서의 이미지 집합을 처리할 수 있는 **기하학 기반 확산 향상기(geometry-grounded diffusion enhancer)**를 설계하였다. 이 모델은 뷰포인트 전반에 걸쳐 보완 정보를 어텐션하고 집계하는 **순서 무관 집합-믹서 어텐션 메커니즘(orderless set-mixer attention mechanism)**을 활용한다.

3DEnhancer는 MVDream과 같은 모델이 생성한 다중 뷰 이미지나 NeRF·3DGS 같은 조악한 3D 표현에서 렌더링된 이미지와 호환된다. 카메라 포즈와 함께 저품질 다중 뷰 이미지가 주어지면, DiT 프레임워크 내에서 행 어텐션(row attention)과 에피폴라 집계(epipolar aggregation) 모듈을 사용하여 다중 뷰 정보를 집계함으로써 시각적 품질과 일관성을 동시에 향상시킨다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 방법 | 연도 | 접근법 | 주요 특징 | 한계 |
|---|---|---|---|---|
| **NeRF** (Mildenhall et al.) | 2020 | Volume Rendering | 고품질 NVS의 기초 확립 | 느린 렌더링 속도 |
| **3DGS** (Kerbl et al.) | 2023 | Gaussian Splatting | 실시간 렌더링 | Sparse view 아티팩트 |
| **DiffusionNeRF** | 2023 | Diffusion + NeRF | RGBD 패치 프라이어 학습 | NeRF 속도 한계 |
| **GaussianSR** | 2024 | SDS + 3DGS | 저해상도 입력으로 고해상도 NVS | SDS의 불안정성 |
| **3DGS-Enhancer** (본 논문) | 2024 | Video LDM + 3DGS | 뷰 일관성, confidence-aware fine-tuning | 단일 뷰 미지원 |
| **3DEnhancer** | 2024 | DiT + Multi-view | 다중 뷰 동시 복원, 텍스트 편집 지원 | 복잡한 아키텍처 |
| **SetDiff** | 2025 | Set-based Diffusion | 다중 카메라·다중 시간 지원 | 초기 단계 연구 |

기존 연구들은 NeRF 향상에 초점을 맞추었으며, NeRF-SR과 Refsr-nerf는 슈퍼 해상도 네트워크를 사용해 훈련 뷰 이미지를 업스케일하여 더 높은 해상도로 novel view를 합성한다.

일부 접근법은 2D 확산 프라이어를 3D 재구성에 통합한다. 예를 들어, DiffusionNeRF는 확산 모델을 활용해 RGBD 패치 프라이어의 로그 기울기를 학습하고, Nerfbusters는 확산 프라이어를 사용해 3D Gaussian의 유령 같은 아티팩트를 제거한다.

GaussianSR은 Score Distillation Sampling(SDS)을 통해 2D 지식을 3D로 증류하는 방식으로 오프더셸프 2D 확산 프라이어를 활용하지만, SDS를 직접 적용하면 생성 프라이어의 무작위성으로 인해 불필요하고 중복된 3D Gaussian 기본 요소가 발생하는 문제가 있다.

---

## 5. 향후 연구에 미치는 영향 및 고려사항

### 5-1. 향후 연구에 미치는 영향

**① 새로운 패러다임 제시**

3DGS-Enhancer가 제시한 "**3D 일관성 복원 문제를 시간적으로 일관된 비디오 생성으로 재구성**"하는 패러다임은, 이후 3DGS 향상 연구들이 비디오 생성 모델의 강력한 사전 지식을 활용하는 방향으로 발전하는 데 핵심적인 영향을 미쳤다.

**② 후속 연구의 기반 형성**

3DGS-Enhancer와 GenFusion은 아티팩트가 있는 렌더링을 수정하고 3D 표현으로 다시 증류하기 위해 fine-tuned 비디오 확산 모델을 통합하는 방식을 공유하며, 이러한 접근법이 이후 연구에서 광범위하게 채택되었다.

**③ 데이터셋 기여**

DL3DV 기반의 수백 개의 언바운드 장면에서 저품질-고품질 이미지 쌍의 대규모 데이터셋을 생성함으로써, 3DGS 향상 문제를 포괄적으로 평가하는 벤치마크를 제공하였다.

### 5-2. 앞으로 연구 시 고려해야 할 점

**① 단일 뷰 확장성**

현재 방법이 연속적인 보간을 위해 인접 뷰에 의존하므로 단일 뷰 3D 모델 생성에는 쉽게 적용할 수 없다는 한계를 극복하기 위한 연구가 필요하다.

**② 더 정교한 Confidence 설계**

Confidence-aware 3DGS fine-tuning 전략이 현재로서는 상대적으로 단순하고 직관적이므로, 더 정교한 신뢰도 추정 및 적용 방법이 고려되어야 한다.

**③ 다중 카메라 환경 대응**

단일 카메라 구성 및 순차적 생성 파이프라인이 다중 뷰 또는 다중 카메라 시뮬레이션 환경으로의 확장성을 제한하므로, 이를 해결하는 연구가 향후 중요한 방향이 될 것이다.

**④ 동적 장면으로의 확장**

현재 3DGS-Enhancer는 정적 장면에 최적화되어 있으므로, 동적 장면(dynamic scene)에서도 view-consistent enhancement를 달성하는 연구가 필요하다.

**⑤ 대규모 데이터셋 활용**

3DGS의 효율적인 데이터 생성 능력을 활용해 비디오 생성 모델을 위한 대규모 데이터셋을 구축하는 것이 모델의 3D 일관성을 향상시키는 유망한 기회가 될 것이다.

---

## 📚 참고자료 및 출처

| # | 제목 / 출처 | 링크 |
|---|---|---|
| 1 | **3DGS-Enhancer** (arXiv 원문) | https://arxiv.org/abs/2410.16266 |
| 2 | **3DGS-Enhancer** (NeurIPS 2024 공식) | https://proceedings.neurips.cc/paper_files/paper/2024/hash/f0b42291ddab77dcb2ef8a3488301b62-Abstract-Conference.html |
| 3 | **3DGS-Enhancer** (프로젝트 페이지) | https://xiliu8006.github.io/3DGS-Enhancer-project/ |
| 4 | **3DGS-Enhancer** (GitHub) | https://github.com/xiliu8006/3DGS-Enhancer |
| 5 | **3DGS-Enhancer** (HTML 전문) | https://arxiv.org/html/2410.16266v1 |
| 6 | **3DGS-Enhancer** (OpenReview) | https://openreview.net/forum?id=P4s6FUpCbG |
| 7 | **3DGS-Enhancer** (HuggingFace) | https://huggingface.co/papers/2410.16266 |
| 8 | **3DGS-Enhancer** (NeurIPS 포스터) | https://neurips.cc/virtual/2024/poster/95333 |
| 9 | **3DEnhancer: Consistent Multi-View Diffusion for 3D Enhancement** (arXiv 2412.18565) | https://arxiv.org/html/2412.18565v1 |
| 10 | **GaussianSR: 3D Gaussian Super-Resolution with 2D Diffusion Priors** (arXiv 2406.10111) | https://arxiv.org/html/2406.10111v1 |
| 11 | **SetDiff: Enhancing Novel View Synthesis via Geometry Grounded Set Diffusion** (arXiv 2601.07540) | https://arxiv.org/html/2601.07540 |
| 12 | **FixingGS: Enhancing 3D Gaussian Splatting** (arXiv 2509.18759) | https://arxiv.org/pdf/2509.18759 |
| 13 | **Moonlight Literature Review — 3DGS-Enhancer** | https://www.themoonlight.io/en/review/3dgs-enhancer-... |
| 14 | **MrNeRF Twitter/X 해설** (NeurIPS Spotlight 언급) | https://x.com/janusch_patas/status/1848570145976754682 |

> ⚠️ **정확도 주의**: 본 답변에서 수식의 세부 표기(예: 손실 함수 가중치의 구체적 수식 형태)는 논문 HTML 전문에서 확인 가능한 범위 내에서 작성하였으며, 일부 세부 수식은 논문의 실제 표기와 다를 수 있습니다. 정확한 수식은 반드시 [원문 arXiv](https://arxiv.org/html/2410.16266v1)를 직접 확인하시기 바랍니다.
