
# RadSplat: Radiance Field-Informed Gaussian Splatting for Robust Real-Time Rendering with 900+ FPS

> **논문 정보:**
> - **저자:** Michael Niemeyer, Fabian Manhardt, Marie-Julie Rakotosaona, Michael Oechsle, Daniel Duckworth, Rama Gosula, Keisuke Tateno, John Bates, Dominik Kaeser, Federico Tombari (Google)
> - **발표:** arXiv:2403.13806 (2024.03), International Conference on 3D Vision (3DV) 2025
> - **공식 페이지:** https://m-niemeyer.github.io/radsplat/
> - **arXiv:** https://arxiv.org/abs/2403.13806

---

## 1. 핵심 주장 및 주요 기여 요약

### 🎯 핵심 주장

Radiance Field 기반 방법(NeRF 계열)은 어려운 환경(야외 촬영, 대규모 장면)에서 최고 품질을 달성하지만 볼류메트릭 렌더링으로 인해 과도한 연산량을 요구하는 반면, Gaussian Splatting 기반 방법은 래스터화를 사용해 실시간 렌더링을 달성하지만 복잡한 장면에서 불안정한 최적화 휴리스틱으로 인해 성능이 저하된다.

RadSplat의 핵심 아이디어는 Neural Field의 안정적인 최적화 및 품질을 **prior**와 **supervision signal**로 활용하여 point-based scene representation의 최적화를 개선하는 것이다.

### 📌 세 가지 주요 기여 (Threefold Contributions)


1. **Radiance Field as Prior & Supervision:** 방사장(radiance field)을 point-based scene representation 최적화의 prior 및 supervision signal로 활용하여 품질 향상 및 강건한 최적화 달성
2. **Novel Pruning Technique:** 전체 point 수를 줄이면서 고품질을 유지하는 새로운 pruning 기법으로 더 작고 컴팩트한 scene representation 구현
3. **Test-Time Filtering:** 렌더링을 더욱 가속화하고 대규모 house-sized 장면까지 확장 가능한 test-time filtering 방식 제안


---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

3DGS는 어려운 최적화 환경과 제한 없는 모델 크기 문제를 가진다. Gaussian primitive의 수를 사전에 알 수 없고, 만족스러운 결과를 위해 세심하게 조정된 merging, splitting, pruning 휴리스틱이 필요하다. 이러한 휴리스틱의 불안정성은 노출 변화, 모션 블러, 움직이는 객체가 불가피한 대규모 장면에서 특히 두드러지며, primitive 수가 증가할수록 메모리 사용량이 관리 불가능해지고 렌더링 속도가 저하되어 대규모 장면에서의 모델 품질에 심각한 제약이 생긴다.

### 2.2 제안하는 방법 (수식 포함)

#### (A) Radiance Field-Based Supervision

Radiance Field는 조명 변화, 노출 변화 등 실제 세계 촬영 환경에서 뛰어난 성능을 보인다. RadSplat은 이 강점을 활용하여 데이터의 복잡성과 노이즈를 제거하고, 손상될 수 있는 입력 이미지 대신 더 깨끗한 supervision signal을 제공한다.

RadSplat의 학습 손실 함수는 다음과 같이 구성된다:

$$
\mathcal{L} = \mathcal{L}_{\text{color}} + \lambda \mathcal{L}_{\text{NeRF}}
$$

구체적으로, NeRF 모델로 모든 입력 이미지를 렌더링하고, $\lambda = 0.2$를 사용하여 NeRF rendered image를 supervision으로 활용한다.

여기서 각 손실 항은 다음과 같이 정의된다:

$$
\mathcal{L}_{\text{color}} = \| \hat{C}_{\text{GS}}(\mathbf{r}) - C_{\text{GT}}(\mathbf{r}) \|_2^2
$$

$$
\mathcal{L}_{\text{NeRF}} = \| \hat{C}_{\text{GS}}(\mathbf{r}) - \hat{C}_{\text{NeRF}}(\mathbf{r}) \|_2^2
$$

- $\hat{C}_{\text{GS}}(\mathbf{r})$: Gaussian Splatting으로 렌더링된 색상
- $C_{\text{GT}}(\mathbf{r})$: 실제 GT 이미지의 색상
- $\hat{C}_{\text{NeRF}}(\mathbf{r})$: 사전 학습된 NeRF로 렌더링된 색상 (supervision signal)

#### (B) NeRF-Informed Initialization

3DGS는 SfM(Structure from Motion) 포인트 클라우드로 초기화되지만, RadSplat은 NeRF의 **density field**를 이용하여 초기화 품질을 개선한다.

NeRF density에서 3D Gaussian의 초기 위치를 샘플링하는 과정:

$$
\mathbf{x}_i \sim p(\mathbf{x}) \propto \sigma_{\text{NeRF}}(\mathbf{x})
$$

- $\sigma_{\text{NeRF}}(\mathbf{x})$: NeRF의 위치 $\mathbf{x}$에서의 volume density

이 NeRF 기반 초기화 및 supervision은 3DGS에 비해 더 안정적이고 고품질의 뷰 합성을 가능하게 한다.

#### (C) Fisheye / 복잡한 렌즈 지원

이 접근 방식의 또 다른 실용적인 장점은 NeRF의 유연한 ray casting으로 인해 임의의 카메라 렌즈 타입에서도 학습이 가능하다는 것이다. 반면 3DGS의 gradient 공식은 pinhole 카메라 모델을 가정하며 fisheye 또는 더 복잡한 렌즈 타입으로의 효율적인 확장이 불분명하다.

#### (D) Novel Pruning Strategy (Importance Score 기반)

각 Gaussian primitive의 학습 뷰에 대한 기여도를 평가하는 새로운 **importance score**를 통해, 기여도가 낮은 점들을 최적화 과정에서 제거한다.

Importance Score는 다음과 같이 정의된다:

$$
s_i = \sum_{j \in \mathcal{V}_i} \alpha_{i,j}
$$

- $s_i$: $i$번째 Gaussian의 importance score
- $\mathcal{V}_i$: Gaussian $i$가 영향을 미치는 학습 뷰 집합
- $\alpha_{i,j}$: $j$번째 뷰에서 해당 Gaussian의 불투명도(opacity) 기여도

Pruning 조건:

$$
\text{Remove Gaussian } i \iff s_i < t_{\text{prune}}
$$

$t_{\text{prune}}$은 장면 표현에 사용되는 point 수를 제어하는 메커니즘으로, 실험에서는 기본 모델과 경량 변형에 대해 두 가지 값을 정의한다.

이 pruning 전략은 Gaussian primitive 수를 최대 10배까지 줄이면서도 품질과 렌더링 속도를 향상시킨다.

#### (E) Test-Time Visibility Filtering

더 크고 복잡한 장면(집 전체 또는 아파트 규모)으로의 확장을 위해, 고전적인 occlusion culling에서 영감을 받아 품질 저하 없이 테스트 시 렌더링 속도를 높이는 새로운 viewpoint 기반 filtering을 post-processing 단계로 도입한다.

입력 카메라 뷰를 클러스터링하고 visibility filtering 단계를 적용하여, 주어진 시점에서 관련 있는 point들의 집합을 동적으로 조정한다. 이 접근 방식은 특히 더 크고 복잡한 장면으로 확장할 때 상당한 FPS 향상을 가져온다.

필터링 수식:

$$
\mathcal{G}_{\text{visible}}(\mathbf{v}) = \{ g_i \mid g_i \in \mathcal{G},\ \mathbf{v} \in \text{cluster}(g_i) \}
$$

- $\mathbf{v}$: 현재 시점(viewpoint)
- $\mathcal{G}$: 전체 Gaussian 집합
- $\text{cluster}(g_i)$: $g_i$가 속한 카메라 클러스터

### 2.3 전체 파이프라인 (모델 구조)

```
[Phase 1: NeRF Pre-training]
  입력 이미지 (wild-type, fisheye 포함)
       ↓
  ZipNeRF / Mip-NeRF360 등 학습
       ↓
  NeRF density & rendered images 획득

[Phase 2: RadSplat Optimization]
  NeRF density → 3D Gaussian 초기화 (위치 샘플링)
       ↓
  NeRF-rendered images를 supervision signal로 활용
  (L_color + λ·L_NeRF)
       ↓
  Importance Score 기반 Pruning (최대 10× 감소)
       ↓
  최적화된 Compact 3D Gaussian Scene

[Phase 3: Test-Time Rendering]
  입력 카메라 클러스터링
       ↓
  Visibility Filtering (viewpoint-adaptive)
       ↓
  실시간 Rasterization → 900+ FPS 렌더링
```

### 2.4 성능 향상

MipNeRF360 데이터셋에서 RadSplat은 품질 지표(SSIM, PSNR, LPIPS)에서 경쟁 방법들을 능가하며, 경량 변형의 경우 평균 907 FPS, 기본 설정에서는 410 FPS의 렌더링 속도를 달성한다.

특히 Mip-NeRF360 벤치마크에서 ZipNeRF보다 높은 SSIM과 LPIPS를 달성하면서 3000배 빠르게 렌더링한다.

Visibility Filtering post-processing을 통해 mip-NeRF 360 장면에서 최대 10% FPS 증가, 더 큰 ZipNeRF 장면에서는 최대 45% 렌더링 속도 향상을 달성하면서도 품질을 유지한다.

| 방법 | 품질 (PSNR/SSIM) | 렌더링 속도 | 모델 크기 |
|---|---|---|---|
| ZipNeRF | SOTA | ~0.3 FPS | 크다 |
| 3DGS | 낮음 | ~200 FPS | 크다 |
| **RadSplat (default)** | **ZipNeRF급** | **~410 FPS** | **컴팩트** |
| **RadSplat (light)** | **경쟁력 있음** | **~907 FPS** | **매우 컴팩트** |

### 2.5 한계

대규모 장면에서 ZipNeRF에 비해 작은 성능 격차가 존재하며, 이는 향후 연구에서 조사할 예정이다.

추가적인 한계:
- RadSplat의 pruning 전략은 동적이거나 움직이는 장면에서 제한이 있을 수 있다.
- 2단계 학습 구조(NeRF 사전학습 → 3DGS 최적화)로 인해 전체 학습 시간이 길다. 약 2시간의 학습 시간이 소요된다.
- NeRF 사전학습이 필수적이어서 단독 3DGS 대비 파이프라인이 복잡하다.
- 3DGS 표현은 래스터화 덕분에 효율적으로 렌더링되지만, 실시간 성능은 여전히 강력한 GPU를 필요로 하며 모든 플랫폼에서 달성되지는 않는다.

---

## 3. 모델의 일반화 성능 향상 가능성

RadSplat이 일반화 성능을 향상시키는 핵심 메커니즘은 다음과 같다:

### 3.1 NeRF Prior를 통한 In-the-Wild 일반화

Radiance field는 조명 변화 및 노출 변화가 포함된 실제 야외 촬영에서도 뛰어난 성능을 보인다. RadSplat은 이 강점을 활용하여 데이터의 복잡성과 노이즈를 제거하고, 손상될 수 있는 입력 이미지 대신 더 깨끗한 supervision signal을 제공한다.

### 3.2 다양한 카메라 모델에 대한 일반화

NeRF의 유연한 ray casting 덕분에 임의의 카메라 렌즈 타입에서도 학습 가능하며, fisheye나 더 복잡한 렌즈 타입으로의 확장이 가능하다. 이는 자율주행, 360° 카메라 등 다양한 실제 응용 환경에서의 일반화를 크게 향상시킨다.

### 3.3 대규모 장면으로의 확장성 일반화

새로운 test-time filtering 방식은 렌더링을 더욱 가속화하고 더 큰 house-sized 장면까지 확장을 가능하게 하며, 900+ FPS에서 복잡한 촬영물의 최고 수준 합성을 가능하게 한다.

### 3.4 단일 모델 학습을 통한 평가 일반화

기존 방법들은 시각화용과 정량 비교용 두 개의 분리된 모델을 학습시켜야 했으나, RadSplat은 radiance field prior 덕분에 강건한 단일 모델을 학습한다. 평가 시에는 잠재적인 색상 이동을 맞추고 공정한 비교를 보장하기 위해 원본 이미지 데이터에 미세 조정만 수행한다.

### 3.5 일반화 향상을 위한 향후 방향

- **동적 장면:** Gaussian을 시간 축으로 확장(4D-GS 등과 결합)하여 동적 장면에서의 일반화
- **Cross-scene generalization:** 단일 장면 최적화 방식에서 벗어나 여러 장면에 공유되는 generalized prior network와의 결합
- **도메인 적응:** NeRF supervision을 다양한 도메인(의료, 위성, 산업 검사 등)으로 확장

---

## 4. 미래 연구에 미치는 영향 및 고려할 점

### 4.1 앞으로의 연구에 미치는 영향

#### 🔬 NeRF-GS 하이브리드 패러다임의 확립
RadSplat은 실시간 3D 렌더링 분야에서 선구적 기여로, radiance field의 섬세한 세부 포착 능력과 Gaussian Splatting의 효율성·실시간 능력을 결합하는 방향을 제시했다.

#### 🔬 Pruning 기반 경량화 연구의 방향 제시
Novel pruning 기법이 point 수를 크게 줄이면서도 더 컴팩트한 장면 표현과 개선된 품질을 가능하게 함을 입증함으로써, 이후 3DGS 경량화 연구(LightGaussian, Compact3DGS 등)에 방향을 제시했다.

#### 🔬 대규모 장면 렌더링 연구에의 기여
Visibility filtering post-processing 단계는 특히 더 복잡한 대규모 장면으로 확장할 때 중요함을 입증하여, 도시 규모, 실내 대형 공간 등의 렌더링 연구에 기초를 제공했다.

#### 🔬 실용적 AR/VR 응용에의 기여
RadSplat의 adaptive test-time filtering 방식은 스트리밍 응용이나 AR/VR 환경으로의 확장 가능성을 보여주며, 이러한 실용적 응용 분야의 연구를 촉진한다.

### 4.2 관련 최신 연구 비교 분석 (2020년 이후)

| 연구 | 연도 | 방법 | 렌더링 속도 | 품질 | 특징 |
|---|---|---|---|---|---|
| **NeRF** (Mildenhall et al.) | 2020 | MLP + Volume Rendering | 매우 느림 (~분) | 높음 | 최초 implicit neural radiance field |
| **Mip-NeRF** (Barron et al.) | 2021 | Anti-aliased NeRF | 느림 | 더 높음 | Multi-scale 처리 |
| **Instant-NGP** (Müller et al.) | 2022 | Hash encoding | 실시간 (~60 FPS) | 높음 | Hash grid로 대폭 가속 |
| **ZipNeRF** (Barron et al.) | 2023 | Zip+Mip NeRF | 느림 | 최고 | 대규모 장면 SOTA |
| **3DGS** (Kerbl et al.) | 2023 | Gaussian Splatting | 실시간 (~200 FPS) | 중간 | 래스터화 기반 |
| **LightGaussian** (Fan et al.) | 2023 | 3DGS + pruning | 200+ FPS | 중간 | 15× 압축 |
| **4D-GS** | 2023 | 시공간 Gaussian | ~82 FPS | 중간 | 동적 장면 |
| **RadSplat** (Niemeyer et al.) | 2024 | NeRF+GS hybrid | **900+ FPS** | **ZipNeRF급** | **NeRF prior 활용** |

### 4.3 앞으로 연구 시 고려할 점

1. **학습 비용 vs. 추론 속도 트레이드오프**
   - 2단계 파이프라인(NeRF 사전학습 필요)의 학습 비용을 줄이기 위한 단일 단계 통합 방법 연구 필요

2. **동적 장면 적용**
   - RadSplat의 pruning 전략은 동적이거나 움직이는 장면에서 제한이 있으므로, 시간적 일관성(temporal consistency)을 고려한 확장 필요

3. **엣지 디바이스 최적화**
   - 실시간 렌더링은 여전히 강력한 GPU가 필요하므로, 모바일·임베디드 환경을 위한 경량화 추가 연구 필요

4. **일반화 가능한 NeRF Prior**
   - 현재 각 장면별로 독립적인 NeRF를 학습하는 방식에서 벗어나, 범용 prior를 사전학습하여 새로운 장면에 빠르게 적응하는 방향(Few-shot / Zero-shot generalization) 연구

5. **다중 모달 센서 통합**
   - LiDAR, depth sensor 등 다중 모달 정보를 NeRF supervision에 통합하여 기하 정확도와 일반화 성능 추가 향상 가능

6. **평가 프로토콜의 표준화**
   - RadSplat이 제시한 단일 모델 학습 및 공정한 평가 방법론을 표준화하여 후속 연구들의 재현성 확보

---

## 📚 참고 자료 및 출처

1. **[주 논문]** Niemeyer et al., "RadSplat: Radiance Field-Informed Gaussian Splatting for Robust Real-Time Rendering with 900+ FPS", arXiv:2403.13806, International Conference on 3D Vision (3DV) 2025. https://arxiv.org/abs/2403.13806
2. **[공식 프로젝트 페이지]** https://m-niemeyer.github.io/radsplat/
3. **[논문 HTML 버전]** https://arxiv.org/html/2403.13806v1 / v2
4. **[Semantic Scholar]** https://www.semanticscholar.org/paper/RadSplat:-Radiance-Field-Informed-Gaussian-for-with-Niemeyer-Manhardt/9942392f8f7cd20019495942cee2eb657c4402bf
5. **[EmergentMind 리뷰]** https://www.emergentmind.com/papers/2403.13806
6. **[RadianceFields.com 해설]** https://radiancefields.com/radsplat
7. **[IEEE Xplore]** https://ieeexplore.ieee.org/abstract/document/11125567
8. **[HuggingFace Papers]** https://huggingface.co/papers/2403.13806
9. **[관련 배경 논문]** Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023. https://arxiv.org/abs/2308.04079
10. **[관련 배경 논문]** Barron et al., "ZipNeRF: Anti-Aliased Grid-Based Neural Radiance Fields", ICCV 2023.
11. **[관련 연구 비교]** Fan et al., "LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS", 2023.

> ⚠️ **주의:** 본 답변에서 수식 중 importance score의 세부 formulation 및 NeRF initialization 수식 일부는 논문의 공개된 HTML 버전 기반으로 재구성하였으며, 논문 내 정확한 수식 표기와 완전히 동일하지 않을 수 있습니다. 정확한 수식은 공식 PDF(https://m-niemeyer.github.io/radsplat/static/pdf/niemeyer2024radsplat.pdf)를 직접 확인하시기 바랍니다.
