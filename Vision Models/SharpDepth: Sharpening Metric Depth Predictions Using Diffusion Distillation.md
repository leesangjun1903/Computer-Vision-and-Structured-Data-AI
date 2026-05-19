
# SharpDepth: Sharpening Metric Depth Predictions Using Diffusion Distillation

> **논문 정보**
> - **저자**: Duc-Hai Pham, Tung Do, Phong Nguyen, Binh-Son Hua, Khoi Nguyen, Rang Nguyen (VinAI Research, Vietnam / Trinity College Dublin)
> - **발표**: arXiv:2411.18229 (2024.11.27), **CVPR 2025** (pp. 17060–17069)
> - **프로젝트 페이지**: https://sharpdepth.github.io/

---

## 1. 핵심 주장 및 주요 기여 요약

### 🎯 핵심 주장

SharpDepth는 Metric3D, UniDepth와 같은 판별적(discriminative) 깊이 추정 방법의 **메트릭 정확도**와, Marigold, Lotus와 같은 생성적(generative) 방법이 달성하는 **세밀한 경계 선명도**를 결합한 단안 메트릭 깊이 추정(monocular metric depth estimation)의 새로운 접근법입니다.

### 🔑 주요 기여

논문의 주요 기여는 다음과 같습니다:
1. **SharpDepth**: 제로샷(zero-shot) 메트릭 깊이를 고충실도(high-fidelity) 세부 정보와 함께 생성할 수 있는 새로운 확산 기반 깊이 선명화 모델
2. **이미지만으로 학습 가능**: 두 가지 노이즈 인식(noise-aware) 모듈 덕분에 GT(Ground-Truth) 없이 이미지만으로 학습

전체 학습 이미지 수는 기존 단안 깊이 추정 방법들보다 약 **100~150배 더 적게** 필요합니다.

SharpDepth의 학습 과정은 ground-truth-free로, 실제 데이터에서 사전 학습된 깊이 모델만 활용하여 필요한 학습 이미지 수를 크게 줄입니다.

---

## 2. 문제 정의 / 제안 방법 / 모델 구조 / 성능 및 한계

### 📌 2.1 해결하고자 하는 문제

실제 데이터의 희소한(sparse) GT 깊이로 학습된 기존 판별 모델들은 메트릭 깊이를 정확히 예측하지만, 종종 **과도하게 평활화(over-smoothed)되거나 저세부(low-detail) 깊이 맵**을 생성합니다.

반면 생성 모델들은 밀도 높은(dense) GT를 가진 합성 데이터로 학습되어 날카로운 경계의 깊이 맵을 생성하지만, **낮은 정확도의 상대적 깊이(relative depth)만 제공**합니다.

단안 메트릭 깊이 추정은 자율주행, 로보틱스 등 다양한 응용 분야에 중요하지만, 스케일 모호성(scale ambiguity)과 제한된 깊이 단서로 인해 zero-shot 환경에서 상당한 도전 과제에 직면합니다.

---

### ⚙️ 2.2 제안하는 방법 (수식 포함)

SharpDepth는 세 가지 핵심 구성 요소로 이루어집니다.

#### (A) Noise-aware Gating 메커니즘

**Noise-aware Gating**은 메트릭 깊이 예측과 affine-invariant 깊이 예측 간의 **차이 맵(difference map)** 에서 불확실한 영역으로 식별된 부분에 깊이 확산 모델이 더 정확하게 집중하도록 안내하는 메커니즘입니다.

차이 맵에서 높은 차이(밝은 영역)는 노이즈에 의해 심하게 왜곡되고, 낮은 차이(어두운 영역)는 일부 정보를 인식 가능한 상태로 유지합니다.

차이 맵 $\mathbf{D}$는 다음과 같이 정의됩니다:

$$\mathbf{D} = \left| d_{\text{metric}} - \hat{d}_{\text{affine}} \right|$$

여기서 $d_{\text{metric}}$은 UniDepth 등 판별 모델의 예측, $\hat{d}_{\text{affine}}$은 Lotus 등 생성 모델의 affine-invariant 예측(스케일 정렬 후)입니다. 이 차이 맵을 기반으로 노이즈 강도를 픽셀별로 조절하여 선택적 잠재 맵(selectively noisy latent map)을 생성합니다.

---

#### (B) Score Distillation Sampling (SDS) Loss

**Score Distillation Sampling (SDS)** 는 3D 자산 합성에 적용되는 증류(distillation) 기법입니다.

SDS Loss는 학습 중 생성적 깊이 모델로부터 선명화 모델로 지식을 증류하는 역할을 하며, 메트릭 정확도를 유지하면서 세부적인 깊이 예측 생성을 촉진합니다.

SDS Loss의 그래디언트는 다음과 같이 근사됩니다:

$$\nabla_\phi \mathcal{L}_{\text{SDS}} = \mathbb{E}_{t, \epsilon}\left[ w(t) \left( \hat{\epsilon}_\theta(\mathbf{z}_t; y, t) - \epsilon \right) \frac{\partial \mathbf{z}}{\partial \phi} \right]$$

여기서:
- $\phi$: SharpDepth 모델의 파라미터
- $\mathbf{z} = \mathcal{E}(d_\phi)$: 깊이 예측의 잠재 인코딩
- $\mathbf{z}_t$: 타임스텝 $t$에서 노이즈가 추가된 잠재 변수
- $\hat{\epsilon}_\theta$: 사전 학습된 확산 모델(Lotus)의 노이즈 예측
- $\epsilon \sim \mathcal{N}(0, \mathbf{I})$: 추가된 가우시안 노이즈
- $w(t)$: 타임스텝 가중치 함수

U-Net Jacobian 항을 제거함으로써, 확산 모델 U-Net을 통한 역전파 없이 최적화가 가능합니다.

---

#### (C) Noise-aware Reconstruction Loss

**Noise-aware Reconstruction Loss**는 확산 기반 모델의 스케일 인식 부재를 보완하기 위해 적용되며, 최종 예측이 초기 깊이 추정치에 근접하게 유지되도록 하는 정규화기(regularizer)로 작동하여, 원래 깊이 스케일에서 벗어나지 않고 메트릭 정확도를 유지합니다.

Noise-aware Reconstruction Loss는 차이 맵에서 식별된 높은 차이 영역에 집중함으로써 초기 메트릭 깊이 추정치에서의 드리프트를 방지합니다.

이를 수식으로 나타내면:

$$\mathcal{L}_{\text{recon}} = \mathbb{E}_{\mathbf{x}} \left[ \mathbf{M}_{\text{low}} \odot \left\| d_\phi(\mathbf{x}) - d_{\text{metric}}(\mathbf{x}) \right\|_1 \right]$$

여기서:
- $\mathbf{M}_{\text{low}}$: 차이 맵 $\mathbf{D}$에서 **낮은 차이 영역**을 나타내는 마스크 (메트릭 정확도가 신뢰할 수 있는 영역)
- $d_\phi(\mathbf{x})$: SharpDepth의 예측 깊이
- $d_{\text{metric}}(\mathbf{x})$: 판별 메트릭 모델의 예측 깊이

#### 최종 학습 목적 함수 (Total Loss)

$$\mathcal{L}_{\text{total}} = \lambda_{\text{SDS}} \cdot \mathcal{L}_{\text{SDS}} + \lambda_{\text{recon}} \cdot \mathcal{L}_{\text{recon}}$$

이 이중 목적(dual-objective) 접근 방식은 SharpDepth가 전체 메트릭 스케일을 손상시키지 않으면서 불확실한 영역을 선택적으로 정제할 수 있게 합니다.

---

### 🏗️ 2.3 모델 구조

SharpDepth 프레임워크는 **확산 기반 추정기(Lotus)**와 **메트릭 깊이 추정기(UniDepth)** 를 활용하여 각각 affine-invariant 깊이 맵과 메트릭 깊이 맵을 생성합니다. Noise-Aware Gating 메커니즘이 선택적 노이즈 잠재 맵을 생성하고, 이를 SharpDepth 모델에 입력합니다. 학습 파이프라인은 SDS Loss와 Noise-Aware Reconstruction Loss를 사용하여 정확도를 개선하고 세부 정보를 향상시킵니다.

전체 아키텍처를 도식화하면:

```
Input RGB Image
      │
      ├──────────────────────────┐
      ▼                          ▼
[UniDepth (판별 모델)]      [Lotus (생성 모델)]
 d_metric (메트릭 깊이)      d_affine (상대 깊이)
      │                          │
      └──────────┬───────────────┘
                 ▼
        [Difference Map 생성]
         D = |d_metric - d_affine|
                 │
                 ▼
    [Noise-aware Gating 메커니즘]
     → 불확실 영역: 높은 노이즈 주입
     → 확실 영역: 낮은 노이즈 주입
     → Selectively Noisy Latent Map
                 │
                 ▼
        [SharpDepth 모델]
      (Diffusion U-Net 기반)
                 │
        ┌────────┴─────────┐
        ▼                  ▼
  [SDS Loss]      [Noise-aware Recon. Loss]
  (세부 선명도)    (메트릭 스케일 유지)
        └────────┬─────────┘
                 ▼
     최종 출력: 선명 + 메트릭 정확 깊이 맵
```

---

### 📊 2.4 성능 향상

표준 깊이 추정 벤치마크에서의 광범위한 제로샷 평가는 SharpDepth의 효과를 확인하며, 높은 깊이 정확도와 세부 표현을 동시에 달성하는 능력을 보여주어 다양한 실세계 환경에서 고품질 깊이 인식이 필요한 응용 분야에 적합함을 입증합니다.

SharpDepth는 모든 데이터셋에서 단순히 affine-invariant 깊이를 메트릭 깊이로 정렬한 naive UniDepth-aligned Lotus 베이스라인을 일관되게 능가하며, 이는 단순한 정렬이 차선책임을 보여줍니다.

다양한 제로샷 데이터셋에서의 실험은 SharpDepth의 정확도가 판별 모델과 경쟁력 있으면서도 생성 모델의 고세부 출력을 포함한다는 것을 보여줍니다.

---

### ⚠️ 2.5 한계점

논문 및 관련 리뷰에서 확인된 한계점은 다음과 같습니다:

1. **기반 모델 의존성**: 프레임워크는 UniDepth와 같은 메트릭 깊이 추정기와 Lotus와 같은 확산 모델을 통합하므로, 두 기반 모델의 성능에 종속적입니다. 메트릭 깊이 모델은 신뢰할 수 있는 전역 깊이 추정치를 제공하고, 생성 모델은 시각적으로 더 선명한 출력을 제공합니다.

2. **추론 속도**: 확산 모델 기반의 생성 과정을 포함하므로, 순수 판별 모델 대비 추론 시간이 증가할 수 있습니다. 확산 방법들은 여전히 계산 비용이 높지만, 다중 스텝 추론의 최적화가 발전하고 있습니다.

3. **학습 파이프라인 복잡성**: 두 개의 사전 학습된 모델(메트릭 + 생성)을 동시에 활용해야 하므로, 단일 모델 대비 파이프라인 구성이 복잡합니다.

---

## 3. 일반화 성능 향상 가능성

SharpDepth의 일반화 성능 향상과 관련된 핵심 요소들을 중점적으로 분석합니다.

### 3.1 Ground-Truth-Free 학습의 일반화 기여

SharpDepth의 학습 과정은 ground-truth-free로, 실제 데이터에서 사전 학습된 깊이 모델만 활용하며, 필요한 학습 이미지 수를 크게 줄입니다.

이는 GT 레이블이 존재하지 않는 새로운 도메인(의료 영상, 수중 환경, 위성 이미지 등)에도 적용 가능함을 의미하며, 데이터 수집 비용 없이 다양한 분야로 확장될 수 있는 잠재력을 가집니다.

### 3.2 Zero-Shot 평가를 통한 일반화 검증

광범위한 제로샷 평가가 다양한 실세계 환경에서 고품질 깊이 인식이 필요한 응용 분야에의 적합성을 확인합니다.

Metric3D, UniDepth와 같은 돌파구적인 연구들은 카메라 내인수(camera intrinsics)를 명시적으로 고려함으로써 교차 카메라 일반화를 달성했으며, 알려진 내인수를 통해 입력을 정규화하거나 내인수에 따라 네트워크를 조건화하여 카메라 기하학과 장면 내용을 분리하는 전략을 사용합니다.

### 3.3 Noise-aware Gating의 적응적 일반화

차이 맵이 Noise-aware Gating 메커니즘을 안내하여 신뢰할 수 있는 픽셀에 대해서만 정렬을 수행하고, 메트릭과 affine-invariant 예측 간에 중요한 불일치가 있는 영역에 선명화기(sharpener)를 집중시킵니다.

이는 도메인이 달라져도 **데이터 기반으로 불확실 영역을 적응적으로 감지**할 수 있어, 특정 도메인에 고정된 경계 정보 없이도 일반화된 선명화가 가능합니다.

### 3.4 기반 메트릭 모델의 일반화 능력 계승

ZoeDepth, UniDepth와 같은 접근 방식들은 구조적 혁신과 대규모 학습을 통해 유망한 이식성(transferability)을 보여줍니다.

SharpDepth는 이러한 강력한 제로샷 일반화 능력을 가진 판별 모델 위에서 동작하므로, 기반 모델의 일반화 능력을 그대로 계승하면서 추가적인 선명도를 더합니다.

### 3.5 데이터 효율성과 일반화의 상관관계

SharpDepth는 판별 모델에 일반적으로 사용되는 데이터셋의 단 **1%** 만 활용하여 학습하도록 설계되었습니다.

소량의 데이터로도 학습이 가능하다는 것은 과적합(overfitting)의 위험을 낮추고, 다양한 새로운 도메인에 빠르게 적응 가능한 일반화 강인성을 나타냅니다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 모델 | 연도 | 유형 | 메트릭 깊이 | 경계 선명도 | Zero-Shot | GT 필요 |
|------|------|------|------------|------------|-----------|---------|
| **AdaBins** | 2021 | 판별 | ✅ | ❌ (보통) | ❌ | ✅ |
| **ZoeDepth** | 2023 | 판별 | ✅ | ❌ (보통) | △ | ✅ |
| **Metric3D** | 2023 | 판별 | ✅ | ❌ | ✅ | ✅ |
| **Marigold** | 2024 | 생성(확산) | ❌ (상대) | ✅ | ✅ | ❌ |
| **UniDepth** | 2024 | 판별 | ✅ | ❌ | ✅ | ✅ |
| **Lotus** | 2024 | 생성(확산) | ❌ (상대) | ✅ | ✅ | ❌ |
| **Depth Pro** | 2024 | 판별+확산 | ✅ | ✅ | ✅ | ✅ |
| **SharpDepth** | 2024 | 혼합(증류) | ✅ | ✅ | ✅ | **❌** |

Marigold, GeoWizard와 같은 생성 확산 모델들은 고주파 세부 정보와 복잡한 기하학을 복원하는 데 강한 잠재력을 보여주지만, DMD는 시야각(field-of-view) 조건화와 로그 스케일 매개변수화를 도입하여 적응성을 개선했습니다.

Metric3D, UniDepth, Depth Pro와 같은 접근 방식들은 초점 거리 모호성을 해결하고 고주파 세부 정보를 보존하는 데 집중합니다.

판별 모델들은 $\ell_1/\ell_2$ 회귀 손실 하에서 깊이 불연속성 전반에서 평균화되는 경향이 있고, 생성 방법들은 전형적으로 구조적 세부 정보를 손상시키는 낮은 차원의 잠재 표현(VAE 병목)에 의존합니다. Depth Pro와 Pixel-Perfect Depth 같은 최근 연구들은 경계 인식 손실과 픽셀 공간 확산으로 이 문제를 명시적으로 다룹니다.

**SharpDepth의 차별점**: 기존 방법들이 GT 데이터를 통해 직접 선명도를 학습하거나, 선명도와 메트릭 정확도 중 하나를 포기하는 반면, SharpDepth는 **GT 없이 확산 증류를 통해 두 장점을 모두 획득**합니다.

---

## 5. 미래 연구에 미치는 영향 및 고려 사항

### 🔮 5.1 앞으로의 연구에 미치는 영향

#### (1) GT-Free 학습 패러다임의 확산

학습 손실들의 또 다른 이점은 추가적인 GT 없이 오직 사전 학습된 깊이 모델을 사용하여 실제 데이터로 개선 모델을 학습할 수 있다는 것입니다.

이는 **레이블 획득 비용이 높은 도메인**(수술 영상, 수중 로봇 등)에서의 깊이 추정 연구에 새로운 방향을 제시합니다.

#### (2) 확산 증류의 새로운 응용

SDS를 깊이 선명화에 적용한 것은 **표면 법선 추정, 광학 흐름(optical flow), 3D 재구성** 등 다른 기하학적 비전 태스크에도 확산 증류를 적용할 수 있는 가능성을 열어줍니다.

#### (3) 판별-생성 하이브리드 연구 트렌드 가속화

SharpDepth는 판별 방법의 메트릭 정확도와 생성 방법의 경계 선명도를 통합하여 고품질 깊이 예측을 제공하는 방향으로, 앞으로의 연구에서 두 패러다임의 장점을 결합하는 하이브리드 접근법이 더욱 주목받을 것입니다.

#### (4) 모듈화된 깊이 개선 파이프라인의 발전

SharpDepth의 플러그인(plug-in) 방식 — 기존 메트릭 모델 위에 선명화 모듈을 얹는 구조 — 은 **모듈화된 깊이 추정 파이프라인** 연구를 자극하여, 임의의 메트릭 깊이 추정기와 조합 가능한 범용 선명화 모듈 개발로 이어질 수 있습니다.

---

### 🧭 5.2 앞으로 연구 시 고려할 점

#### (1) 추론 속도 최적화

확산 기반 파이프라인의 다중 스텝 추론은 자율주행 등 실시간 응용에서 병목이 됩니다. 확산 방법들은 여전히 계산 비용이 높지만, 다중 스텝 추론의 최적화가 진전되고 있으므로, **일관성 모델(consistency model)** 이나 **플로우 매칭(flow matching)** 등 빠른 샘플링 기법과의 결합이 필요합니다.

#### (2) 비디오 및 시간적 일관성 확장

정적 이미지에서 동적 비디오로의 전환은 비자명(non-trivial)하며, 프레임별 정밀한 기하학적 추론뿐만 아니라 엄격한 시간적 일관성도 요구합니다. SharpDepth의 프레임 단위 처리를 시간적 일관성을 고려한 비디오 깊이 추정으로 확장하는 연구가 필요합니다.

#### (3) 다양한 도메인 적용 가능성 검증

향후 우선 과제는 계산 효율성 개선, 다시점(multi-view) 환경에서의 기하학적 일관성 강화, 도메인 적응 발전을 포함합니다. SharpDepth가 평가한 벤치마크 외의 도메인(내시경, 드론 촬영, 야간 영상 등)에 대한 성능 검증이 필요합니다.

#### (4) 차이 맵 품질의 의존성 극복

차이 맵이 Noise-aware Gating 메커니즘에 필수적인데, 이는 신뢰할 수 있는 픽셀에 대해서만 정렬을 수행하고 메트릭과 affine-invariant 예측 간에 중요한 불일치가 있는 영역에 집중하게 합니다. 만약 기반 메트릭 모델과 생성 모델의 예측이 모두 부정확하다면 차이 맵 자체의 신뢰도가 낮아질 수 있으므로, **차이 맵 품질을 보장하는 메커니즘** 연구가 필요합니다.

#### (5) 스케일 일관성의 장기적 보장

확산 모델의 본질적인 확률적 특성(stochastic nature)은 동일 입력에 대한 다중 추론 시 깊이 스케일의 변동을 야기할 수 있습니다. 생성 모델들은 확률적 기하학적 환상(stochastic geometric hallucinations)과 스케일 드리프트(scale drift)의 문제를 가지고 있기 때문에, SharpDepth의 Reconstruction Loss가 이를 얼마나 견고하게 방지하는지에 대한 추가 분석이 필요합니다.

#### (6) 기반 모델의 발전과의 동기화

SharpDepth의 성능은 UniDepth, Lotus 등 기반 모델에 종속됩니다. 따라서 **더 강력한 메트릭 모델(Metric3D v2, Depth Pro 등)을 기반으로 SharpDepth를 재학습**했을 때의 성능 향상 가능성을 탐구하는 연구가 가치 있습니다.

---

## 📚 참고 자료 및 출처

| # | 제목 / 출처 | 링크 |
|---|------------|------|
| 1 | **SharpDepth: Sharpening Metric Depth Predictions Using Diffusion Distillation** (arXiv:2411.18229) — Pham et al., 2024 | https://arxiv.org/abs/2411.18229 |
| 2 | **CVPR 2025 포스터 페이지** — SharpDepth | https://cvpr.thecvf.com/virtual/2025/poster/35055 |
| 3 | **SharpDepth 공식 프로젝트 페이지** | https://sharpdepth.github.io/ |
| 4 | **HuggingFace Papers** — SharpDepth (arXiv:2411.18229) | https://huggingface.co/papers/2411.18229 |
| 5 | **Semantic Scholar** — SharpDepth (CVPR 2025, pp. 17060–17069) | https://www.semanticscholar.org/paper/SharpDepth.../dd9a15c9 |
| 6 | **Liner Quick Review** — SharpDepth | https://liner.com/review/sharpdepth-... |
| 7 | **Moonlight Literature Review** — SharpDepth | https://www.themoonlight.io/en/review/sharpdepth-... |
| 8 | **Survey on Monocular Metric Depth Estimation** (arXiv:2501.11841) — Zhang, 2025 | https://arxiv.org/pdf/2501.11841 |
| 9 | **UniDepth: Universal Monocular Metric Depth Estimation** (CVPR 2024) — Piccinelli et al. | https://arxiv.org/html/2403.18913v1 |
| 10 | **IEEE Xplore** — SharpDepth | https://ieeexplore.ieee.org/document/11093832/ |
| 11 | **Awesome-Monocular-Depth** GitHub (관련 연구 목록) | https://github.com/choyingw/Awesome-Monocular-Depth |

> ⚠️ **주의**: 본 답변의 수식 중 SDS 그래디언트, Reconstruction Loss, 차이 맵 정의 등 일부는 논문의 전체 원문이 공개된 arXiv PDF(arxiv.org/pdf/2411.18229)의 내용과 공개된 리뷰들을 기반으로 재구성한 것입니다. 정확한 수식 표기는 원문 논문 PDF를 직접 확인하시기를 권장합니다.
