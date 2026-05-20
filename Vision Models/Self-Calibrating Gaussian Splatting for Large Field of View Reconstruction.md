
# Self-Calibrating Gaussian Splatting for Large Field of View Reconstruction

> **논문 정보**
> - **제목**: Self-Calibrating Gaussian Splatting for Large Field-of-View Reconstruction
> - **저자**: Youming Deng 외 6인
> - **arXiv**: [2502.09563](https://arxiv.org/abs/2502.09563) (2025년 2월 제출)
> - **학회**: **ICCV 2025** (발표 확정)
> - **프로젝트 페이지**: https://denghilbert.github.io/self-cali/
> - **코드**: https://github.com/denghilbert/Self-Cali-GS

---

## 1. 핵심 주장 및 주요 기여 요약

이 논문은 카메라 파라미터, 렌즈 왜곡, 3D Gaussian 표현을 **동시에 최적화(jointly optimize)**하는 **자기 교정(self-calibrating) 프레임워크**를 제시하며, 특히 광각 렌즈로 촬영한 **대형 FOV(Field of View) 영상으로부터 더 적은 수의 이미지만으로도 고품질 장면 재구성**을 가능하게 한다.

### 주요 기여 3가지

| 기여 | 내용 |
|:--|:--|
| **① Hybrid Distortion Field** | Invertible ResNet + Explicit Grid 결합 |
| **② Cubemap 기반 리샘플링** | 대형 FOV에서 해상도 손실 없이 렌더링 |
| **③ 완전 미분 가능한 파이프라인** | 포즈, 내부 파라미터, 왜곡, 3DGS 동시 최적화 |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

NeRF와 3DGS 같은 기존 방법들은 고품질 novel-view synthesis에서 놀라운 성능을 보였지만, **조밀한 이미지 촬영, 좁은 FOV, SfM 기반의 정밀한 카메라 포즈 추정**을 필요로 한다.

실제 응용에서는 피쉬아이 렌즈와 같은 **대형 FOV 렌즈**가 로보틱스, VR 분야에 광범위하게 사용되는데, 이는 더 적은 이미지로 장면의 더 넓은 영역을 한 번에 촬영할 수 있기 때문이다.

기존 재구성 파이프라인들은 대형 FOV 입력 데이터를 투시(perspective) 이미지로 변환하는 방식을 사용하여 이미지 전체를 활용하지 못하고, 렌즈 주변부에서 실제 피쉬아이 렌즈를 정확히 근사하지 못하는 모델로 캘리브레이션을 수행한다.

기존 방법들은 그리드 기반 레이 오프셋이나 invertible neural network를 사용해 왔는데, **그리드 기반 모델**은 속도는 빠르지만 광도 손실(photometric loss)만으로 지도 학습 시 불안정하고 노이즈가 많은 결과를 내며, **신경망 기반 접근**은 정규화는 우수하지만 각각의 Gaussian에 적용할 경우 계산 비용이 지나치게 크다.

---

### 2-2. 제안하는 방법 및 수식

#### (A) 3D Gaussian Splatting 기본 모델

3D Gaussian Splatting에서 각 Gaussian은 다음과 같이 정의된다:

$$\mathcal{G}(\mathbf{x}) = e^{-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}$$

여기서 $\boldsymbol{\mu}$는 3D 공간에서의 평균(위치), $\boldsymbol{\Sigma} = \mathbf{R}\mathbf{S}\mathbf{S}^\top\mathbf{R}^\top$은 공분산 행렬이며, $\mathbf{R}$은 회전 행렬, $\mathbf{S}$는 스케일 행렬이다.

#### (B) 기존 렌즈 왜곡 모델 (Brown–Conrady)

기존의 방사형(radial) 렌즈 왜곡은 다음과 같이 파라메트릭하게 표현된다:

$$D(r(\mathbf{x}_n)) = 1 + k_1 r^2 + k_2 r^4 + k_3 r^6 + \cdots$$

이 표준 다항식 모델은 피쉬아이 렌즈의 주변부 왜곡을 정밀하게 표현하지 못하는 한계가 있다.

#### (C) Hybrid Distortion Field (핵심 제안)

이 한계를 극복하기 위해 논문은 **두 가지 접근법의 장점을 결합하는 하이브리드 신경 필드**를 제안한다. 구체적으로 **Invertible Residual Networks (iResNet)**를 사용하여 정규화된 희소 그리드(sparse grid)에서 변위(displacement)를 예측하고, 이후 **쌍선형 보간(bilinear interpolation)**을 적용하여 연속적인 왜곡 필드를 생성한다.

수식으로 표현하면:

$$\mathbf{d}(\mathbf{p}_c) = \text{Bilinear}\left(\mathcal{F}_{\text{iResNet}}(\mathbf{P}_c)\right)$$

여기서:
- $\mathbf{p}_c$: 정규화된 이미지 좌표
- $\mathbf{P}_c$: 희소 제어점(control points) 그리드
- $\mathcal{F}_{\text{iResNet}}$: Invertible ResNet이 예측하는 변위 벡터 필드
- $\mathbf{d}(\mathbf{p}_c)$: 연속 왜곡 필드

왜곡이 적용된 최종 픽셀 좌표:

$$\hat{\mathbf{x}} = \mathbf{x} + \mathbf{d}(\mathbf{x})$$

Invertible neural network가 제공하는 **전단사(bijective) 매핑**은 렌즈 왜곡을 정확히 모델링하면서 계산의 실행 가능성을 보장하는 데 필수적이다.

이 아키텍처의 장점은 비용이 큰 ResNet 순전파/역전파 계산을 희소 그리드의 위치에서만 수행하면 되고, 이는 장면 내 Gaussian의 수와 무관하게 **그리드 해상도에만 비례하여 스케일**된다는 점이다.

#### (D) 공동 최적화 손실 함수

시스템은 **원본 입력 픽셀에 직접 측정된 손실(loss)**을 사용하여 렌즈 왜곡, 카메라 내부 파라미터, 카메라 포즈, 장면 표현을 동시에 최적화한다.

전체 손실 함수는 다음과 같이 구성된다:

$$\mathcal{L} = \mathcal{L}_{\text{photo}} + \lambda_1 \mathcal{L}_{\text{smooth}} + \lambda_2 \mathcal{L}_{\text{reg}}$$

- $\mathcal{L}_{\text{photo}}$: 렌더링 이미지와 입력 이미지 간의 광도 손실 (L1 + D-SSIM)
- $\mathcal{L}_{\text{smooth}}$: 왜곡 필드의 매끄러움 정규화
- $\mathcal{L}_{\text{reg}}$: 카메라 파라미터 정규화

---

### 2-3. 모델 구조

Self-Calibrating Gaussian Splatting은 **하이브리드 렌즈 왜곡 필드를 갖춘 미분 가능한 래스터화 파이프라인**으로, 보정되지 않은 광각 사진으로부터 고품질 novel view synthesis 결과를 생성할 수 있다.

전체 파이프라인 구조:

```
[입력: 미보정 광각 이미지]
        ↓
[SfM 초기화 (COLMAP 등)]
        ↓
┌──────────────────────────────────┐
│   Self-Calibrating 최적화 루프   │
│                                  │
│  ┌─────────────────────────┐    │
│  │  Hybrid Distortion Field │    │
│  │  iResNet + Sparse Grid  │    │
│  │  → Bilinear Interp.     │    │
│  └──────────┬──────────────┘    │
│             ↓                    │
│  ┌─────────────────────────┐    │
│  │  Cubemap Projection     │    │
│  │  (6면 × 90° 렌더링)    │    │
│  └──────────┬──────────────┘    │
│             ↓                    │
│  ┌─────────────────────────┐    │
│  │  3D Gaussian Splatting  │    │
│  │  (μ, Σ, opacity, SH)   │    │
│  └──────────┬──────────────┘    │
│             ↓                    │
│     [Photometric Loss]           │
│       ↑ 역전파(backprop) ↑       │
│  카메라 포즈 / 내부 파라미터 갱신 │
└──────────────────────────────────┘
        ↓
[출력: 고품질 3D 재구성 + 새 시점 합성]
```

#### Cubemap 기반 리샘플링

대형 FOV 이미지 렌더링을 위해 논문은 단일 평면 투영에서 **큐브맵(cubemap) 투영**으로의 전환을 제안한다. 큐브맵의 각 면은 장면의 90°씩을 투영하며, FOV 전체에 걸쳐 샘플링을 보다 균일하게 분산시켜 픽셀 스트레칭과 왜곡 아티팩트를 크게 줄인다. 이 방법론을 통해 프레임워크는 **180°를 초과하는 FOV**도 지원할 수 있다.

단일 평면 투영은 FOV가 180°에 가까워질수록 주변부에서 업샘플링 비율이 급격히 증가하는 수학적 한계가 있기 때문에, 큐브맵 투영으로 확장하는 것이 필요하다.

---

### 2-4. 성능 향상 및 한계

#### 성능 향상

본 방법은 Gaussian Splatting의 빠른 래스터화와 호환되며, 다양한 카메라 렌즈 왜곡에 적응 가능하고, **합성 및 실제 데이터셋 모두에서 최첨단(SOTA) 성능**을 달성한다.

FisheyeNeRF 데이터셋에서의 비교 실험에서, Vanilla 3DGS는 렌즈 왜곡 모델링 부재로 인해 수많은 플로터(floaters)가 발생하며 제대로 된 재구성에 실패하고, Fisheye-GS는 왜곡이 심한 주변부에서 어려움을 겪는 반면, 본 방법은 특히 고도로 왜곡된 영역에서 더 나은 재구성을 산출한다.

Fisheye-GS 같은 기존 방법은 고정된 전통적 파라메트릭 왜곡 모델 때문에 복잡한 렌즈 왜곡 처리에 실패하는 반면, 본 방법은 특히 주변부 영역에서 대형 왜곡을 정확하게 모델링하여 고도로 왜곡된 원시 이미지 전체를 재구성에 활용한다.

큐브맵 리샘플링 전략은 FOV 전반에 걸쳐 일관된 픽셀 밀도를 유지하며, 하이브리드 왜곡 필드와 결합되어 심각한 왜곡이나 픽셀 스트레칭 없이 주변부 영역을 활용할 수 있다. 또한 본 방법은 **최대 180°의 FOV**도 처리 가능하다.

#### 한계 (논문에서 명시하지 않은 부분은 추론)

- **SfM 초기화 의존성**: 여전히 초기 카메라 포즈 추정을 위해 COLMAP 등의 SfM에 의존
- **큐브맵 경계 아티팩트**: 래스터화 시 큐브맵 면 경계에서 강도 불일치를 방지하기 위해 거리 기반 정렬 조정이 필요하다.
- **학습 시간 증가**: iResNet 최적화로 인한 훈련 시간 증가 가능성

---

## 3. 모델의 일반화 성능 향상 가능성

일반화 성능 향상은 이 논문의 핵심 강점 중 하나이다.

### 3-1. 카메라 모델에 대한 일반화

시스템은 3D Gaussian 파라미터와 카메라 파라미터를 **동시에(in tandem) 최적화**하여, 광범위한 사전 캘리브레이션 없이도 다양한 카메라 설정에 대한 높은 유연성과 적응성을 가능하게 한다.

본 방법은 **다양한 종류의 카메라 렌즈 왜곡에 적응 가능**하도록 설계되어, 특정 카메라 모델에 종속되지 않는 일반화된 접근법을 제공한다.

투영 레이어의 모듈화된 설계는 파노라마, 큐브맵 등 다른 카메라 모델로의 손쉬운 확장을 가능하게 하여, 소나 융합(sonar fusion) 및 LiDAR를 포함한 하이브리드 센서 파이프라인과의 통합도 용이하다.

### 3-2. 데이터셋에 대한 일반화

합성 데이터셋과 실제 데이터셋 모두에서 진행된 포괄적인 평가는 기존 프레임워크인 Fisheye-GS 및 3DGS 대비 제안된 방법의 우월성을 입증하며, **도메인 간 일반화 가능성**을 보여준다.

### 3-3. 일반화를 촉진하는 구조적 요소

하이브리드 왜곡 모델링은 기존 렌즈 왜곡 모델의 한계—특히 피쉬아이 렌즈에서—를 해결하기 위해 **iResNet과 명시적 그리드 시스템을 결합**한다. iResNet은 정규화된 희소 그리드에 대한 변위 필드를 예측하고, 이후 쌍선형 보간을 통해 연속적인 왜곡 필드를 생성하며, 이 결합은 **표현력과 계산 효율성 사이의 균형**을 잡는다.

- **표현 유연성**: iResNet의 데이터 기반(data-driven) 학습이 사전 정의된 파라메트릭 모델보다 더 다양한 렌즈 특성을 포착
- **정규화 효과**: 희소 그리드 + 보간의 구조가 과적합을 방지하고 매끄러운 왜곡 필드 생성
- **미분 가능성**: 전체 파이프라인이 미분 가능하여 end-to-end 최적화 및 새로운 손실 함수 적용이 용이

---

## 4. 최신 관련 연구 비교 분석 (2020년 이후)

| 연구 | 연도 | 방법 | 카메라 모델 | 렌즈 왜곡 처리 | 한계 |
|:--|:--|:--|:--|:--|:--|
| **NeRF** (Mildenhall et al.) | 2020 | 음함수 신경 방사장 | 핀홀 | 레이 샘플링으로 유연 처리 | 느린 학습/추론 |
| **Mip-NeRF 360** (Barron et al.) | 2022 | Anti-aliased NeRF | 무한 장면 | 제한적 | 좁은 FOV 중심 |
| **3DGS** (Kerbl et al.) | 2023 | 명시적 Gaussian 래스터화 | 핀홀 | 미지원 | 광각 렌즈 불가 |
| **Fisheye-GS** | 2024 | 3DGS + 피쉬아이 투영 | Equidistant | 고정 파라메트릭 모델 | 주변부 정확도 부족 |
| **Self-Cali-GS (본 논문)** | 2025 | 자기 교정 + Hybrid Field | 다양한 광각 | iResNet + Grid 하이브리드 | SfM 의존성 일부 잔재 |

NeRF 기반 접근법은 피쉬아이 카메라와 같은 비핀홀 카메라 모델을 레이 샘플링을 통해 다룰 수 있지만, 대규모 평가는 여전히 제한적이다.

피쉬아이 이미지를 novel view synthesis에 통합하는 것은 내재적인 곡률과 왜곡 때문에 여러 과제를 제기한다. NeRF는 레이 샘플링 접근법을 통해 피쉬아이 렌즈의 왜곡을 다룰 수 있지만, 3DGS는 원근 투영에 의존하기 때문에 피쉬아이 영상의 극단적인 곡률과 넓은 FOV를 처리하는 데 적합하지 않아 렌더링의 정확성과 효율성 부족으로 이어진다.

피쉬아이 처리와 결합된 명시적 3DGS는 대규모 재구성에서 NeRF 스타일의 암묵적 모델보다 정량적으로나 기하학적 정확도 측면 모두에서 우수한 성능을 보인다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려사항

### 5-1. 앞으로의 연구에 미치는 영향

#### ① 산업 응용 확장
피쉬아이 기반 3DGS는 효율적인 광각 및 전방위 3D 재구성을 가능하게 하며, 자율주행, 로보틱스, 몰입형 VR/AR, 신속한 전체 장면 커버리지가 필요한 대규모 매핑 작업에 응용될 수 있다.

#### ② 캘리브레이션 패러다임 전환
카메라 파라미터와 3D 장면 표현을 함께 최적화하는 **자기 교정 기술**은 별도의 캘리브레이션 타겟 없이 카메라를 교정하는 방법에 뿌리를 두며, 방사장(radiance field) 및 3D Gaussian 프레임워크에 통합되어 내·외부 파라미터를 광도 손실로 정제하는 방향으로 발전해왔다.

#### ③ Gaussian Splatting 확장성

번들 조정(bundle adjustment) 프로세스를 카메라 포즈와 함께 렌즈 파라미터까지 최적화하도록 확장하는 자기 교정 접근법은 향후 다양한 재구성 시스템에서 표준 구성 요소가 될 가능성이 높다.

### 5-2. 앞으로 연구 시 고려할 점

#### ① SfM 의존성 극복
캘리브레이션 타겟 없이 카메라를 보정하는 것은 신뢰할 수 있는 대응점(correspondences)을 확립하기 위해 장면 구조와 기하학적 사전 지식에 대한 강한 가정에 의존하기 때문에 특히 어렵다.
→ SfM 없이 초기화하는 방법(ex. monocular depth estimation 활용)에 대한 연구 필요

피쉬아이 Gaussian Splatting은 무거운 전처리 없이 실현 가능하며, 단안 깊이(monocular depth)가 왜곡이 심한 장면에서 SfM의 실용적인 대안이 될 수 있고, 향후 연구는 단안 3D 추정기로부터 직접 Gaussian 파라미터를 회귀하거나 더 대규모 장면으로 확장하는 방향을 탐색할 수 있다.

#### ② 큐브맵 경계 처리
래스터화 시 큐브맵 면 경계에서의 강도 불일치 문제는 아직 완전히 해결되지 않았으며, 무결절(seamless) 큐브맵 렌더링 연구가 필요하다.

#### ③ 실시간 처리
3DGS는 이미지 공간에서 직접 래스터화되는 비등방성(anisotropic) Gaussian으로 장면을 표현하여 실시간 렌더링을 가능하게 한다. 하이브리드 왜곡 필드 추가로 인한 지연(latency)을 최소화하는 경량화 연구가 필요하다.

#### ④ 다중 카메라 및 센서 융합
모듈화된 투영 레이어 설계는 LiDAR, 소나(sonar) 등 다른 센서와의 하이브리드 파이프라인 통합에도 적합하므로, 자율주행 등의 분야에서 이종 센서 융합 연구로 확장 가능하다.

#### ⑤ 동적 장면으로의 확장
현재 본 논문은 정적 장면에 초점을 맞추고 있으며, 대형 FOV 카메라로 촬영된 **동적 장면(dynamic scene)**으로의 확장이 중요한 후속 연구 방향이다.

---

## 📚 참고 자료 및 출처

| 번호 | 제목 / 출처 |
|:--:|:--|
| 1 | **[Primary] Self-Calibrating Gaussian Splatting for Large Field-of-View Reconstruction** — arXiv:2502.09563, Youming Deng et al. https://arxiv.org/abs/2502.09563 |
| 2 | **[ICCV 2025 공식 페이퍼]** OpenAccess ICCV 2025: https://openaccess.thecvf.com/content/ICCV2025/papers/Deng_Self-Calibrating_Gaussian_Splatting_for_Large_Field-of-View_Reconstruction_ICCV_2025_paper.pdf |
| 3 | **[프로젝트 페이지]** https://denghilbert.github.io/self-cali/ |
| 4 | **[공식 코드 GitHub]** Self-Cali-GS [ICCV 2025]: https://github.com/denghilbert/Self-Cali-GS |
| 5 | **[ICCV 2025 포스터]** https://iccv.thecvf.com/virtual/2025/poster/438 |
| 6 | **[Literature Review]** Moonlight AI Review: https://www.themoonlight.io/en/review/self-calibrating-gaussian-splatting-for-large-field-of-view-reconstruction |
| 7 | **[관련 연구] Fisheye-GS** — arXiv:2409.04751 (2024): https://arxiv.org/abs/2409.04751 |
| 8 | **[관련 연구] 3D Gaussian Splatting** — Kerbl et al., SIGGRAPH 2023 |
| 9 | **[관련 연구] NeRF** — Mildenhall et al., ECCV 2020 |
| 10 | **[관련 연구] Evaluating Fisheye-Compatible 3DGS Methods** — arXiv:2508.06968: https://arxiv.org/abs/2508.06968 |
| 11 | **[관련 연구] Fisheye-Based 3D Gaussian Splatting (Survey)** — EmergentMind: https://www.emergentmind.com/topics/fisheye-based-3d-gaussian-splatting |

> ⚠️ **정확도 관련 안내**: 본 논문의 구체적인 정량적 수치(PSNR/SSIM 수치 등)는 공개된 검색 결과에서 충분히 확인되지 않아 기재를 생략하였습니다. 정확한 수치 비교는 arXiv 원문 또는 ICCV 2025 공식 논문을 직접 참조하시기를 권장합니다.
