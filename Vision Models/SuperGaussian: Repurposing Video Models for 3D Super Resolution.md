
# SuperGaussian: Repurposing Video Models for 3D Super Resolution

> **논문 정보**
> - **저자**: Yuan Shen, Duygu Ceylan, Paul Guerrero, Zexiang Xu, Niloy J. Mitra, Shenlong Wang, Anna Frühstück
> - **학회**: ECCV 2024 (European Conference on Computer Vision)
> - **arXiv**: [2406.00609](https://arxiv.org/abs/2406.00609)
> - **프로젝트 페이지**: [supergaussian.github.io](https://supergaussian.github.io/)
> - **DOI**: 10.1007/978-3-031-73397-0_13

---

## 1. 핵심 주장 및 주요 기여 (Executive Summary)

SuperGaussian은 저해상도(coarse) 3D 모델에 기하학적·외관적 세부사항을 추가하여 업샘플링하는 단순하고 모듈화된 범용(generic) 방법을 제안합니다. 현재 생성형 3D 모델은 이미지·비디오 도메인에 비해 품질이 낮다는 한계가 있는데, 이 논문은 기존 사전학습된(pretrained) 비디오 모델을 3D 초해상도(super-resolution)에 직접 재활용함으로써 고품질 3D 학습 데이터 부족 문제를 우회할 수 있음을 증명합니다.

### 주요 기여 요약

| 기여 | 설명 |
|---|---|
| **비디오 모델 재활용** | 사전학습된 비디오 업샘플러를 3D SR에 재사용 |
| **3D 일관성 보장** | 비-3D-일관 비디오 모델 + 3D 통합(consolidation) 결합 |
| **다양한 입력 형식 지원** | NeRF, Gaussian Splat, 노이즈 스캔, text-to-3D, 저해상도 메시 등 |
| **카테고리 비종속(agnostic)** | 특정 객체 카테고리에 제한되지 않음 |
| **Gaussian Splat 출력** | 효율적 렌더링 가능한 고품질 3DGS 출력 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

현재 생성형 3D 모델은 이미지·비디오 도메인과 달리 품질이 충분히 높지 않습니다. 이는 고품질 3D 학습 데이터의 대규모 저장소가 부족하기 때문인데, 기존의 사전학습 비디오 모델을 3D 초해상도에 직접 재사용하면 이 데이터 부족 문제를 우회할 수 있습니다.

핵심 과제는 3D 일관성(3D consistency)을 보장하는 것입니다. 비디오 모델은 시간적으로 부드럽기는 하지만 3D 일관성이 보장되지 않습니다.

정리하면, 문제는 크게 두 가지입니다:
1. **3D 품질의 낮음**: 생성형 3D 모델이 이미지/비디오 대비 낮은 품질
2. **3D 데이터 부족**: 고품질 3D 학습 데이터 레포지토리의 절대적 부족

---

### 2.2 제안하는 방법

#### 파이프라인 개요

SuperGaussian은 두 단계로 진행됩니다: 먼저 샘플링된 뷰 궤적을 기반으로 저해상도 3D 입력에서 비디오를 렌더링하고, 두 번째로 이를 사전학습된 비디오 업샘플러(선택적으로 도메인 특화 아티팩트에 맞게 파인튜닝 가능)로 업샘플링합니다.

보다 구체적으로:

주어진 저해상도 3D 표현에서 부드러운 카메라 궤적(smooth camera trajectory)을 샘플링하여 중간 저해상도 비디오를 렌더링합니다. 이 비디오를 기존의 비디오 업샘플러로 업샘플링하여 더 선명하고 생동감 있는 세부사항을 가진 고해상도 3D 표현을 얻습니다. 이후 3D 최적화를 수행하여 기하학적·텍스처 세부사항을 개선하며, SuperGaussian은 최종적으로 고해상도 Gaussian Splat 형태의 3D 표현을 생성합니다.

#### 수식 표현

**3D Gaussian Splatting의 기본 표현:**

각 Gaussian은 위치 $\boldsymbol{\mu} \in \mathbb{R}^3$, 공분산 행렬 $\boldsymbol{\Sigma} \in \mathbb{R}^{3\times3}$, 불투명도 $\alpha$, 색상(Spherical Harmonics) $\mathbf{c}$로 파라미터화됩니다:

$$G(\mathbf{x}) = e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})}$$

**공분산 행렬의 분해 (학습 안정성):**

$$\boldsymbol{\Sigma} = \mathbf{R}\mathbf{S}\mathbf{S}^\top\mathbf{R}^\top$$

여기서 $\mathbf{R}$은 회전 행렬(Rotation), $\mathbf{S}$는 스케일 행렬(Scale)입니다.

**Alpha-compositing을 통한 이미지 렌더링:**

$$\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} \mathbf{c}_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)$$

여기서 $\mathbf{c}_i$는 색상, $\alpha_i$는 불투명도입니다.

**SuperGaussian의 목표 함수 (3D 최적화):**

업샘플링된 고해상도 비디오 프레임 $\{\hat{I}_1, \hat{I}_2, \ldots, \hat{I}_T\}$를 감독 신호로 삼아, 3DGS 파라미터 $\theta$를 다음 손실로 최적화합니다:

$$\mathcal{L} = \mathcal{L}_1(\hat{I}, I_{render}) + \lambda_{\text{SSIM}} \mathcal{L}_{\text{D-SSIM}}(\hat{I}, I_{render})$$

여기서:
- $\hat{I}$: 비디오 업샘플러로 생성된 고해상도 프레임
- $I_{render}$: 현재 3DGS 파라미터 $\theta$로 렌더링한 이미지
- $\mathcal{L}_1$: L1 픽셀 손실
- $\mathcal{L}_{\text{D-SSIM}}$: 구조적 유사도 손실

**비디오 업샘플링 (VideoGigaGAN 기반):**

저해상도 비디오 $V_{low} \in \mathbb{R}^{T \times H \times W \times 3}$에서 고해상도 비디오 $V_{high} \in \mathbb{R}^{T \times sH \times sW \times 3}$로의 변환:

$$V_{high} = f_{\phi}(V_{low})$$

여기서 $f_\phi$는 사전학습된 비디오 업샘플러(VideoGigaGAN), $s$는 업샘플 배율(e.g., $s=4$ for $4\times$ upsampling)입니다.

---

### 2.3 모델 구조

SuperGaussian에서는 VideoGigaGAN을 비디오 업샘플러로 사용하며, 이는 기존 GigaGAN 아키텍처를 재사용하면서 시간적 특징 추출 및 전파를 위한 BasicVSR++ 레이어를 추가하여 저해상도 비디오 입력 프레임을 처리합니다.

모델 크기는 피처 차원 조정 후 이미지 기반 사전 모델보다 약간 작습니다. 최종적으로 이미지 업샘플러와 비디오 업샘플러 모두 MVImgNet 데이터셋에서 수렴할 때까지 파인튜닝됩니다.

비디오 업샘플러의 사전 지식을 활용하는 것 외에도, SuperGaussian은 도메인 특화된 저해상도 비디오(즉, 저해상도 3D 표현에서 렌더링된 비디오)에 대해 파인튜닝을 수행합니다. 이를 통해 다양한 3D 캡처·생성 과정에서 발생하는 복잡한 열화(degradation)를 처리할 수 있습니다. 또한 프레임워크의 각 컴포넌트는 고도로 모듈화되어 있어 다른 SOTA 비디오 방법으로 쉽게 교체할 수 있습니다.

파이프라인 구조를 정리하면:

```
[저해상도 3D 입력 ψ_low]
    (NeRF / 3DGS / Mesh / noisy scan / text-to-3D)
         ↓
[카메라 궤적 샘플링 → 비디오 렌더링 V_low]
         ↓
[VideoGigaGAN 비디오 업샘플링 (+ MVImgNet 파인튜닝)]
         ↓
[고해상도 비디오 V_high]
         ↓
[3D 최적화 (3DGS 피팅, L1 + D-SSIM 손실)]
         ↓
[고해상도 Gaussian Splat 출력 ψ_high]
```

---

### 2.4 성능 향상

기준 방법들과의 4× 업샘플링 비교에서 SuperGaussian은 저해상도 Gaussian Splats을 포함한 다양한 입력 유형에 대해 비교를 수행하며, 일반적(generic)임에도 불구하고 일관되게 최고의 정량적 결과를 달성합니다.

두 가지 사전 모델(이미지 사전, 비디오 사전) 모두 업샘플링 후 생성적(generative) 행동을 보이며, 비디오 사전(prior)은 업샘플링된 프레임 전반에 걸쳐 뛰어난 시간적 일관성(temporal consistency)을 나타냅니다.

SuperGaussian은 최대 16× 업샘플링 결과를 저해상도 입력에 적용하여 보여줍니다.

#### 비교 대상 베이스라인 (4× 업샘플링 기준):
- **NeRF-SR** (Wang et al., MM 2022)
- **FastSR-NeRF** (Lin et al., WACV 2024)
- **Instruct-NeRF2NeRF** (Haque et al., ICCV 2023)
- **이미지 업샘플러 기반 GigaGAN 베이스라인**

---

### 2.5 한계 (Limitations)

SuperGaussian이 매우 크거나 복잡한 3D 장면으로 어떻게 확장될 수 있는지가 불명확합니다. Gaussian splatting 과정이 계산 집약적으로 될 수 있습니다.

또한 논문은 입력 3D 데이터의 품질 및 특성 변화에 대한 방법의 민감도를 충분히 탐색하지 않습니다.

비디오를 활용하는 접근 방식이지만, 이미지는 비디오보다 훨씬 쉽게 얻을 수 있기 때문에, 기존의 2D 이미지 모델이 비디오 모델보다 더 우월한 특징 표현 능력을 제공할 수 있다는 반론도 있습니다.

추가 한계 정리:
- **실내/대규모 장면** 처리에 대한 검증 미흡 (객체 중심 장면에 초점)
- **동적 장면(dynamic scenes)** 미지원
- **3D 일관성** 비디오 모델의 태생적 한계를 완전히 해결하지 못함 (3D consolidation으로 보완하지만 완벽하지 않음)

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화를 가능하게 하는 설계 요소

SuperGaussian의 핵심 관찰은 어떤 3D 표현이든 부드러운 궤적을 따라 여러 시점에서 렌더링하여 중간적인 범용 비디오 표현으로 변환할 수 있다는 것입니다. 따라서 기존의 비디오 모델을 3D 업샘플링 또는 초해상도 작업에 재사용하는 것이 가능합니다. 이러한 비디오 모델들은 대규모 비디오 데이터로 학습되었기 때문에, 다양한 일반적 시나리오에 적용 가능한 강력한 사전(prior)을 제공합니다.

SuperGaussian은 NeRF, Gaussian Splat, iPhone 같은 노이즈 RGB-D 센서에서 얻은 재구성 결과, 최근 text-to-3D 방법으로 생성된 3D 결과, 저해상도 메시 등 다양한 입력 유형을 처리할 수 있습니다.

중간 비디오 표현이 이러한 모든 입력 유형에서 렌더링될 수 있기 때문에, SuperGaussian은 범용적인 3D 초해상도 프레임워크를 제공합니다.

### 3.2 반복 적용을 통한 추가 향상

다양한 수준의 업샘플링을 반복적으로 실행함으로써 달성할 수 있습니다.

즉, 아래와 같은 순환 적용이 가능합니다:

$$\psi^{(0)}_{low} \xrightarrow{\text{SuperGaussian}} \psi^{(1)} \xrightarrow{\text{SuperGaussian}} \psi^{(2)} \xrightarrow{\cdots} \psi^{(n)}_{high}$$

### 3.3 모듈화 설계로 인한 확장성

SuperGaussian의 방법은 카테고리에 무관(category-agnostic)하며 기존 3D 워크플로우에 쉽게 통합될 수 있습니다.

GitHub에서는 VideoGigaGAN의 대안으로 Upscale-a-Video 비디오 사전(video prior)을 SuperGaussian 프레임워크에서 사용해 볼 것을 권장하고 있으며, 이는 RealBasicVSR보다 훨씬 우수한 성능을 제공할 것으로 예상됩니다.

### 3.4 파인튜닝 전략

비디오 업샘플러의 사전 지식을 활용하는 것 외에도, SuperGaussian은 도메인 특화 저해상도 비디오에 대한 파인튜닝을 수행합니다. 이를 통해 SuperGaussian은 다양한 3D 캡처 및 생성 프로세스로 인해 발생하는 복잡한 열화를 처리할 수 있습니다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 앞으로의 연구에 미치는 영향

#### (1) 2D 사전(Prior)의 3D 전이 패러다임 정립

이 논문은 기존의 사전학습된 비디오 모델을 3D 초해상도에 직접 재사용하여 고품질 3D 학습 데이터 부족 문제를 우회할 수 있다는 것을 증명하였습니다. 이는 2D(이미지/비디오) 도메인의 강력한 생성 사전을 3D 표현으로 전이하는 일반적인 패러다임을 열어줍니다.

#### (2) 모듈화된 3D 향상 워크플로우

이 접근법은 특정 3D 객체 카테고리에 대한 학습 없이도 고품질 3D 업샘플링을 가능하게 하며, 의료 스캔, 3D 지도, 제품 디자인 등 더 넓은 범위의 3D 데이터에 적용 가능합니다.

#### (3) 후속 연구 자극

이 논문은 다양한 후속 연구를 직접 자극했습니다:

- 3DSR은 SuperGaussian과 같은 VSR 기반 방법에 비해 비디오 모델을 파인튜닝하지 않고도 더 명시적으로 3D 일관성을 장려하는 방법을 제안합니다.
- SRGS는 동결된(frozen) 단일 이미지 SR 모델로 생성한 초해상도 뷰와 LR 정답 이미지 모두를 이용해 Gaussian 파라미터를 공동 최적화합니다.
- SuperGS는 두 단계 coarse-to-fine 프레임워크를 설계하여, 잠재 특징 필드(latent feature field)로 저해상도 장면을 표현하고 초해상도 최적화의 초기값으로 사용하며, 변분 잔차 특징(variational residual features)과 다중 뷰 공동 학습을 도입합니다.

---

### 4.2 앞으로 연구 시 고려할 점

#### (1) 3D 일관성의 근본적 해결
비디오 모델은 시간적으로 부드럽지만 3D 일관성이 보장되지 않습니다. SuperGaussian은 이미지 기반 방법보다 시간적 일관성을 크게 개선하지만, 3D 통합(consolidation) 단계에서도 완전한 3D 일관성을 보장하기는 어렵습니다. 향후 연구는 3D 일관성을 보다 명시적으로 강제하는 방법(예: 에피폴라 제약, depth consistency loss)을 통합하는 방향으로 발전해야 합니다.

#### (2) 대규모·실내 장면으로의 확장
매우 크거나 복잡한 3D 장면으로 SuperGaussian 방법이 어떻게 확장되는지 불명확합니다. 향후 연구는 장면 단위 분할(tiling), 스트리밍 방식 처리, 또는 계층적 3DGS(LoD 기반)와의 결합을 검토해야 합니다.

#### (3) 동적 장면 처리
현재 SuperGaussian은 정적 장면에만 적용됩니다. 동적 3D Gaussian Splatting(예: 4D-GS, Dynamic 3DGS)과의 결합을 통해 동적 장면의 초해상도로 확장하는 것이 중요한 연구 방향입니다.

#### (4) 더 강력한 비디오/확산 사전과의 결합
3DSR처럼 확산 기반 2D 초해상도 모델을 활용하는 방향도 유망합니다. 기존 이미지·비디오 SR 방법들이 뷰 종속 아티팩트를 겪거나 비디오 모델을 파인튜닝해야 하는 반면, 확산 모델 사전과 3D 일관 렌더링 파이프라인을 결합하면 여러 시점에서 구조적 일관성을 보장할 수 있습니다.

#### (5) 데이터 및 파인튜닝 전략

MVImgNet과 같은 멀티뷰 데이터셋 외에도 더 다양하고 대규모인 3D 데이터셋(예: Objaverse-XL)에서의 파인튜닝 또는 어댑터(adapter) 기반 경량 도메인 적응 전략이 일반화 성능을 더욱 높일 수 있습니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 핵심 전략 | 입력 | 출력 | 3D 일관성 | 카테고리 독립성 | 발표 |
|---|---|---|---|---|---|---|
| **NeRF-SR** | NeRF 슈퍼샘플링 | NeRF | NeRF | 중간 | O | MM 2022 |
| **FastSR-NeRF** | 경량 SR 파이프라인 | NeRF | NeRF | 중간 | O | WACV 2024 |
| **SuperGaussian** | 비디오 모델 재활용 + 3DGS | NeRF/3DGS/Mesh 등 | 3DGS | 비디오 prior + 3D 통합 | **O** | ECCV 2024 |
| **SRGS** | 단일이미지 SR + 3DGS 공동 최적화 | 저해상도 멀티뷰 | 3DGS | 지오메트리 기반 | O | 2024 |
| **SuperGS** | Coarse-to-fine + 잠재 특징 필드 | 저해상도 멀티뷰 | 3DGS | 불확실성 모델링 | O | 2024 |
| **S2Gaussian** | Sparse-view + Gaussian Shuffle Split | 희소뷰 저해상도 | 3DGS | 비일관성 모델링 모듈 | O | CVPR 2025 |
| **SplatSuRe** | 선택적(geometry-aware) SR | 저해상도 멀티뷰 | 3DGS | 기하학적 선택적 적용 | O | 2025 |
| **3DSR** | 확산 기반 SR + 3DGS 명시적 일관성 | 멀티뷰 | 3DGS | **명시적 강제** | O | 2025 |

3DSR은 비디오 모델 파인튜닝 없이 SuperGaussian 대비 더 명시적인 3D 일관성을 장려하는 방법을 제안합니다.

S2Gaussian은 희소 뷰 재구성에 초점을 맞추며, SR 이미지에서의 비일관성을 줄이기 위한 장면별(per-scene) 비일관성 모델링 모듈을 제안합니다.

3DGS 훈련 파이프라인에 SR을 통합하는 자연스러운 전략은 각 이미지를 독립적으로 향상시키는 것이지만, 이는 멀티뷰 비일관성을 초래하여 흐릿한 렌더링으로 이어집니다. 이전 방법들은 학습된 신경 컴포넌트, 시간적으로 일관된 비디오 사전, 또는 LR과 SR 뷰에 대한 공동 최적화를 통해 이러한 비일관성을 완화하려 하지만, 모두 모든 이미지에 SR을 균일하게 적용합니다.

---

## 참고 자료 (References)

1. **arXiv 원문**: Shen, Y. et al., "SuperGaussian: Repurposing Video Models for 3D Super Resolution", arXiv:2406.00609, 2024. https://arxiv.org/abs/2406.00609
2. **ECCV 2024 공식 논문 (ECVA)**: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04210.pdf
3. **Springer ECCV 2024 챕터**: https://link.springer.com/chapter/10.1007/978-3-031-73397-0_13
4. **프로젝트 페이지**: https://supergaussian.github.io/
5. **Adobe Research 페이지**: https://research.adobe.com/publication/supergaussian-repurposing-video-models-for-3d-super-resolution/
6. **GitHub (공식 코드)**: https://github.com/adobe-research/SuperGaussian
7. **SplatSuRe (비교 연구)**: arXiv:2512.02172, https://splatsure.github.io/
8. **3DSR (비교 연구)**: arXiv:2508.04090, https://consistent3dsr.github.io/
9. **SuperGS (비교 연구)**: arXiv:2410.02571
10. **3D Gaussian Splatting 원논문**: Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", ACM Trans. Graph. 42(4), 2023. https://github.com/graphdeco-inria/gaussian-splatting
11. **ECCV 2024 포스터 페이지**: https://eccv.ecva.net/virtual/2024/poster/1441
12. **Illinois Experts**: https://experts.illinois.edu/en/publications/supergaussian-repurposing-video-models-for3d-super-resolution

> ⚠️ **정확도 안내**: 본 답변에서 수식(특히 손실 함수의 세부 가중치, 파인튜닝 하이퍼파라미터)은 원 논문의 공개된 HTML/PDF 버전 및 관련 문헌에서 추론한 것입니다. 논문의 구체적인 실험 수치(PSNR, LPIPS 등 정량 결과)는 논문 원문(PDF) 또는 프로젝트 페이지에서 직접 확인하시기 바랍니다.
