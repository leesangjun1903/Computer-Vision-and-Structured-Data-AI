
# DN-Splatter: Depth and Normal Priors for Gaussian Splatting and Meshing

> **논문 정보**
> - **저자**: Matias Turkulainen, Xuqian Ren, Iaroslav Melekhov, Otto Seiskari, Esa Rahtu, Juho Kannala
> - **발표**: WACV 2025 (IEEE/CVF Winter Conference on Applications of Computer Vision)
> - **arXiv**: [arXiv:2403.17822](https://arxiv.org/abs/2403.17822) (v3: Nov 7, 2024)
> - **GitHub**: [maturk/dn-splatter](https://github.com/maturk/dn-splatter)

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

실내 씬의 고품질 3D 재건은 VR/AR 응용에서 매우 중요하다. 3D Gaussian Splatting(3DGS)은 빠른 렌더링 속도와 낮은 학습 시간으로 최신 novel view synthesis 결과를 달성하지만, 최적화 과정에서 **기하학적 제약(geometric constraint)의 부재**로 인해 실내 데이터셋에서의 성능이 저하된다.

이 논문은 쉽게 접근 가능한 기하학적 단서(geometric cues)를 활용하여 도전적이고 비정형적이며 텍스처가 없는(textureless) 씬에서의 Gaussian Splatting 최적화를 향상시키고, 3DGS를 깊이(depth)와 법선(normal) 단서로 확장하여 효율적인 메시 추출 기법을 제시함을 핵심 주장으로 삼는다.

### 1.2 주요 기여 (4가지)

| 기여 | 내용 |
|------|------|
| ① Depth Prior 도입 | 센서 깊이 및 단안(monocular) 깊이 추정 네트워크를 활용한 깊이 정규화 |
| ② Normal Prior 도입 | Omnidata 등 off-the-shelf 모노큘러 네트워크로 Gaussian의 법선 방향 정렬 |
| ③ Adaptive Depth Loss | 컬러 이미지의 그래디언트 기반 적응적 깊이 손실 함수 제안 |
| ④ Mesh 추출 | 정규화된 Gaussian 표현으로부터 Poisson surface reconstruction을 통한 직접적 메시 추출 |

컬러 이미지의 그래디언트에 기반한 적응적 깊이 손실(adaptive depth loss)을 제안하여 다양한 베이스라인 대비 깊이 추정과 novel view synthesis 결과를 개선하였으며, 이 단순하면서도 효과적인 정규화 기법은 Gaussian 표현으로부터 직접 메시를 추출할 수 있게 하여 실내 씬의 물리적으로 더 정확한 재건을 가능하게 한다.

---

## 2. 해결하고자 하는 문제 / 제안 방법 / 모델 구조 / 성능 향상 및 한계

### 2.1 해결하고자 하는 문제

전통적인 광도 측정(photometric) 방식은 텍스처가 없는 환경이나 기하학적 제약이 최소화된 환경에서 아티팩트 및 부정확한 표면 표현과 같은 문제를 일으키는 경우가 많다.

구체적으로는 다음 세 가지 핵심 문제를 다룬다:
1. **Textureless 영역** (벽, 천장, 바닥 등): 광도 손실만으로는 기하 구조 복원 불가
2. **Gaussian의 기하 비정렬**: 3DGS는 최적화 중 기하 제약이 없어 표면과 맞지 않는 Gaussian이 생성됨
3. **메시 추출의 어려움**: 기존 NeRF/SDF 기반 메시 추출법은 느리고 후처리(post-refinement)가 필요

3DGS는 차별화된 가역 렌더링(differentiable rendering) 접근 방식으로 실시간 렌더링 능력을 자랑하지만, 최적화 중 3D 및 표면 제약이 없어 기하학적 모호성과 아티팩트에 취약하다.

---

### 2.2 제안하는 방법 (수식 포함)

#### (A) 3D Gaussian의 기본 렌더링

각 Gaussian $i$의 불투명도(opacity) 기여는 다음과 같다:

$$\alpha_i = o_i \cdot \exp\left(-\frac{1}{2}(\mathbf{p} - \boldsymbol{\mu}_i)^\intercal \boldsymbol{\Sigma}_i^{-1}(\mathbf{p} - \boldsymbol{\mu}_i)\right)$$

여기서 $o_i$는 학습 가능한 불투명도, $\boldsymbol{\mu}_i$는 Gaussian 중심, $\boldsymbol{\Sigma}_i$는 공분산 행렬이다.

---

#### (B) 깊이 예측 (Depth Prediction)

모노큘러 깊이 추정값 $D_\text{mono}$에 대해 SfM 포인트를 카메라 뷰에 투영한 희소 깊이 맵 $D_\text{sparse}$와 스케일을 맞추며, 이미지별 스케일 $a$와 시프트 $b$ 파라미터를 폐쇄형 선형 회귀로 다음과 같이 구한다:

$$\hat{a}, \hat{b} = \arg\min_{a,b} \sum_{ij} \left\| (a \cdot D_{\text{mono},ij} + b) - D_{\text{sparse},ij} \right\|_2^2$$

---

#### (C) Gradient-Aware (Sensor) Depth Loss

RGB 이미지 기반의 그래디언트 인식 깊이 손실(gradient-aware depth loss)을 제안하며, 엣지를 나타내는 큰 이미지 그래디언트 영역에서 깊이 손실을 낮춤으로써 텍스처 없는 평탄한 영역에 정규화를 더 강하게 적용한다.

실험을 통해 로그 페널티(logarithmic penalty)가 선형 또는 이차 페널티보다 더 부드러운 재건을 결과한다는 것을 발견하였다.

Edge-aware log depth loss는 다음과 같이 정의된다:

$$\mathcal{L}_{\hat{D}} = \frac{1}{|\hat{D}|} \sum_{ij} e^{-\nabla C_{ij}} \cdot \left| \log \hat{D}_{ij} - \log D_{ij} \right|$$

여기서 $\nabla C_{ij}$는 픽셀 $(i,j)$에서의 컬러 이미지 그래디언트이다.

---

#### (D) 법선 예측 및 법선 손실 (Normal Prediction & Loss)

최적화 과정에서 Gaussian은 납작한 디스크(disc) 형태로 수렴하도록 유도되며, 하나의 스케일 축이 나머지 두 축보다 훨씬 작아지고, 이 작은 스케일 축이 법선 방향의 근사값으로 사용된다.

렌더링된 깊이 맵의 그래디언트에서 유사 정답 법선 맵을 추정하는 방법 대신, Omnidata 네트워크로부터 얻은 단안 단서를 이용해 예측 법선을 지도학습하며, 이는 훨씬 부드러운 법선 추정을 제공한다.

법선 손실은 다음과 같이 정의된다:

$$\mathcal{L}_{\hat{N}} = \frac{1}{|\hat{N}|} \sum \left\| \hat{\mathbf{N}} - \mathbf{N} \right\|_1$$

평활도 손실(smoothness loss):

$$\mathcal{L}_{\text{smooth}} = \sum_{i,j} \left( |\nabla_i \hat{\mathbf{N}}_{i,j}| + |\nabla_j \hat{\mathbf{N}}_{i,j}| \right)$$

---

#### (E) 전체 최적화 손실 함수 (Total Loss)

전체 최적화 손실은 여러 구성 요소를 통합하며, 각 항은 광도 손실, 깊이 정규화, 스케일 제약, 법선 평활도에 각각 대응하여 균형 있는 학습 과정을 가능하게 한다:

$$\mathcal{L} = \mathcal{L}_{\hat{C}} + \lambda_d \mathcal{L}_{\hat{D}} + \mathcal{L}_{\text{scale}} + \underbrace{\lambda_n \mathcal{L}_{\hat{N}} + \lambda_s \mathcal{L}_{\text{smooth}}}_{\mathcal{L}_{\text{normal}}}$$

실험에서는 $\lambda_d = 0.2$, $\lambda_n = 0.1$, $\lambda_s = 0.1$로 설정한다.

---

### 2.3 모델 구조

제안 방법은 PyTorch와 gsplat(v1.0.0)으로 구현되었으며, 모든 모델은 30,000 iteration 동안 학습된다. 단안 법선 단서를 얻기 위해 사전 학습된 Omnidata 모델에 RGB 이미지를 통과시키며, 학습 데이터셋의 센서 깊이로부터 역투영된 1M 포인트로 Gaussian 씬을 초기화한다.

전체 파이프라인은 아래와 같이 구성된다:

```
[입력] RGB Images + (Sensor Depth / Monocular Depth)
        ↓
[SfM 초기화] 1M 역투영 포인트로 3DGS 초기화
        ↓
[Off-the-shelf 네트워크] Omnidata → 모노큘러 법선 맵 생성
        ↓
[3DGS 최적화]
  - 광도 손실 L_C
  - Edge-aware 깊이 손실 L_D
  - 법선 정렬 손실 L_N
  - 평활도 손실 L_smooth
  - 스케일 손실 L_scale
        ↓
[메시 추출] Poisson Surface Reconstruction (깊이 + 법선 맵 역투영)
        ↓
[출력] Novel View Images + Depth Maps + Normal Maps + Triangle Mesh
```

표면 추출을 위해 최적화된 Gaussian 표현에 Poisson 재건 기법을 적용하며, 정렬된 깊이 및 필터링된 법선을 활용하여 일관된 메시 기하 구조를 생성한다.

---

### 2.4 성능 향상

Splatfacto, SuGaR, 2DGS 등 Gaussian 기반 방법들은 오직 광도 손실로만 학습되어 낮은 텍스처 환경에서 씬 기하 구조 파악에 심각하게 어려움을 겪지만, DN-Splatter로 깊이 및 법선 지도학습을 추가하면 재건 품질이 크게 향상된다.

DN-Splatter는 novel view 및 메시 재건 메트릭 모두에서 더 높은 수치를 달성하며, 2DGS는 더 부드러운 깊이 렌더를 얻는다.

- **평가 지표**: PSNR, SSIM, LPIPS(컬러 이미지) 및 공통 깊이 메트릭을 보고하며, 지면 진실(sensor) 데이터가 있는 데이터셋의 깊이 품질 분석에 사용한다.
- **비교 베이스라인**: 암묵적 NeRF·SDF 기반 표현(Nerfacto, Depth-Nerfacto, Neusfacto, MonoSDF)과 명시적 Gaussian 기반 방법(Splatfacto, SuGaR, 2DGS)을 폭넓게 비교한다.

---

### 2.5 한계점

베이스라인 대비 개선을 달성했지만, 카메라가 상대적으로 정지한 밀집 캡처(densely captured) 씬에 초점을 맞추었으며, 모션 블러 및 기타 캡처 아티팩트가 있는 더 어렵고 희소한 데이터 캡처에 대한 연구가 향후 필요하다.

또한, Gaussian 씬 파라미터와 메시 품질을 동시에 최적화하는 더 나은 메싱 기법이 필요하며, Poisson surface reconstruction은 법선 추정보다 포인트의 위치 추정에 더 민감하다.

이 방법은 깊이와 법선 프라이어가 모호성을 해소하기에 충분하지 않은 매우 복잡하거나 혼잡한(cluttered) 씬에서 어려움을 겪을 수 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

DN-Splatter의 일반화 성능과 관련된 내용을 핵심적으로 분석한다.

### 3.1 Off-the-shelf 네트워크를 통한 일반화

단안 깊이 및 법선 추정 네트워크에서의 발전과 ToF·깊이 센서가 탑재된 모바일 디바이스의 풍부함에 힘입어, 이러한 기하학적 프라이어를 이용한 3DGS 정규화를 탐구하며, 이를 통해 광도 재현성과 표면 재건을 향상시킨다.

즉, Omnidata, DepthAnything 등 **사전 학습된 범용 네트워크를 플러그-앤-플레이(plug-and-play)**로 사용하기 때문에, 특정 씬에 국한되지 않고 다양한 도메인으로의 확장이 용이하다.

### 3.2 스마트폰 데이터로의 일반화

이 저장소는 스마트폰 데이터(아이폰)를 사용하여 개선된 novel-view synthesis 및 메시 재건을 위한 Gaussian splatting 모델의 깊이·법선 지도학습 연구 논문들을 구현한다.

센서 깊이 측정값이 포함된 데이터셋에 대해 예측 깊이 맵에 직접 깊이 정규화를 적용하며, iPhone 등 소비자 기기에서 발견되는 저해상도 상용 깊이 센서는 물체 경계에서 매끄럽지 않은 엣지를 생성하고 부정확한 판독값을 제공하는 경우가 많다.

이 점에 착안하여 **gradient-aware loss**를 통해 노이즈가 많은 저가 센서 데이터에서도 안정적인 학습이 가능하도록 설계되었으며, 이는 실용적 일반화 성능 향상에 기여한다.

### 3.3 센서 깊이 없는 환경에서의 일반화 (Monocular Prior)

이 정규화 전략은 깊이 데이터가 없는 데이터셋에 대해 센서로부터 얻거나 단안 깊이 추정 네트워크를 통해 추론된 깊이 프라이어를 활용하며, 실내 씬의 텍스처가 없거나 관측이 부족한 영역에서의 모호성 감소에 유익하다.

센서 깊이 데이터가 없는 경우 모노큘러 깊이 지도학습(`PearsonDepth` 손실)이 권장되며, 센서 깊이 지도학습에는 `EdgeAwareLogL1` 손실이 권장된다.

### 3.4 일반화의 한계

다양한 조명 조건을 포함하는 씬에서는 DN-Splatter 결과에 더 많은 노이즈가 발생할 수 있다.

이는 Omnidata 등 법선 추정 네트워크가 복잡한 조명 변화에 취약한 경우, DN-Splatter의 법선 프라이어의 품질 자체가 저하될 수 있음을 의미한다.

---

## 4. 후속 연구에 미치는 영향 및 연구 시 고려할 점

### 4.1 후속 연구에 미치는 영향

#### (1) 기하학적 프라이어의 필요성 공식화
DN-Splatter는 도전적인 실내 데이터셋에서 광도 재현성과 기하학적 정확성을 위한 3DGS의 깊이·법선 정규화 방법으로서, 이 단순하면서도 효과적인 전략이 공통 novel-view RGB 메트릭을 향상시키고 Gaussian 씬 표현에서 추출된 표면 품질을 크게 개선함을 보였으며, 도전적인 실내 씬에서 더 기하학적으로 유효하고 일관된 재건을 위해 프라이어 정규화가 필수적임을 입증하였다.

이는 이후의 **GaussianRoom, PGSR, VCR-GauS, 2DGS-Room** 등 수많은 후속 연구에서 기하 프라이어 도입의 필요성을 정당화하는 기반이 되었다.

#### (2) AGS-Mesh로의 직접적 발전
2024년 11월에 공개된 AGS-Mesh 방법은 DN-Splatter를 기반으로, 새로운 깊이·법선 필터링 전략과 octree 기반 등가면(isosurface) 추출 방법으로 메시 재건을 향상시킨다.

#### (3) 로보틱스 및 AR/VR 응용으로의 파급
FusionSense는 로봇 촉각(tactile) 응용을 위한 희소 설정(sparse settings)에서 DN-Splatter를 개선한다.

#### (4) 모노큘러 네트워크와의 결합 촉진
학습 기반 기법을 통합하여 프라이어를 데이터에서 학습하는 방향으로의 개선이 가능하며, 깊이·법선 프라이어를 Gaussian splatting 이외의 3D 재건 파이프라인에 통합하는 것도 유익한 연구 방향이다.

---

### 4.2 향후 연구 시 고려할 점

| 고려사항 | 내용 |
|---------|------|
| **희소 뷰 설정** | 현재는 밀집 캡처에 제한되어 있으므로, 모션 블러나 기타 아티팩트가 있는 더 어렵고 희소한 데이터 캡처에 대한 연구가 필요하다. |
| **메시 품질의 동시 최적화** | Gaussian 씬 파라미터와 메시 품질을 동시 최적화하는 더 나은 메싱 기법 개발이 필요하다. |
| **조명 변화 강인성** | DN-Splatter는 다양한 조명 조건을 포함하는 씬에서 더 많은 노이즈가 발생할 수 있으므로, 조명 불변(illumination-invariant) 프라이어 통합이 필요하다. |
| **법선 추정 네트워크 선택** | 서로 다른 모노큘러 네트워크가 서로 다른 카메라 좌표계를 사용할 수 있으므로, 좌표계 변환 및 네트워크 호환성에 주의해야 한다. |
| **SDF와의 결합** | GaussianRoom은 Signed Distance Field(SDF)의 연속성에서 영감을 받아 신경 SDF와 3DGS를 통합하는 통합 최적화 프레임워크를 제시하며, 학습 가능한 신경 SDF 필드가 Gaussian의 밀집화·가지치기를 가이드한다. 이러한 SDF와의 결합은 DN-Splatter의 기하 정확성을 더욱 향상시킬 수 있다. |
| **일반화 vs. 씬별 최적화 트레이드오프** | DN-Splatter는 씬별 최적화(per-scene optimization)에 의존하므로, 새로운 씬에서 재사용 가능한 피드포워드(feed-forward) 일반화 모델로의 발전이 필요하다. |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 연도 | 표현 방식 | 기하 프라이어 | 메시 추출 | 특징 |
|------|------|----------|------------|---------|------|
| **NeRF** (Mildenhall et al.) | 2020 | 암묵적 MLP | ✗ | Marching Cubes | Novel view synthesis 기반 |
| **MonoSDF** | 2022 | SDF+NeRF | 단안 깊이·법선 | Marching Cubes | 높은 재건 품질, 느린 학습(~18시간) |
| **3DGS** (Kerbl et al.) | 2023 | 명시적 3D Gaussian | ✗ | 어려움 | 실시간 렌더링, 낮은 기하 품질 |
| **SuGaR** | 2024 | 3DGS 변형 | 표면 밀도 | Poisson | 메시 재건에 초점, 기하 정규화 부족 |
| **2DGS** (Huang et al.) | 2024 | 2D Gaussian surfel | ✗ | TSDF | 기하 정확성 향상, 광택 표면 취약 |
| **DN-Splatter** (본 논문) | 2024 | 3DGS + Depth/Normal | 깊이+법선 프라이어 | Poisson | 스마트폰 데이터, 적응적 손실함수 |
| **GaussianRoom** | 2024 | 3DGS + SDF | 단안 법선+엣지 | - | SDF 가이드 Gaussian 분포 |
| **PGSR** | 2024 | Planar 3DGS | 멀티뷰 기하 제약 | - | 평면 기반 표면 정확성 |
| **AGS-Mesh** | 2025 | DN-Splatter 확장 | 깊이·법선 필터링 | Octree Isosurface | DN-Splatter의 직접적 후속 |

이전 연구는 NeRF 표현에서 메시를 추출하는 데 일부 성공을 거두었지만, 이러한 방법들은 종종 비싼 후처리 단계에 의존하며, 대부분의 최신 기법들은 SDF나 점유(occupancy) 표현과 marching cubes를 결합하여 세부 사항을 달성하나 밀집한 3D 볼륨을 쿼리·평가해야 하므로 일반적으로 학습이 느리다.

DN-Splatter의 차별점은:
1. **학습 속도**: SDF 기반 방법(MonoSDF: ~18시간) 대비 빠른 학습
2. **접근성**: 스마트폰 데이터만으로 고품질 재건 가능
3. **확장성**: off-the-shelf 네트워크 교체(Omnidata → DepthAnything, DSINE 등)로 손쉬운 업그레이드 가능

---

## 참고 자료 및 출처

1. **arXiv 논문 원문**: [arXiv:2403.17822](https://arxiv.org/abs/2403.17822) — Turkulainen et al., "DN-Splatter: Depth and Normal Priors for Gaussian Splatting and Meshing" (2024)
2. **WACV 2025 공식 논문 PDF**: [WACV 2025 Open Access](https://openaccess.thecvf.com/content/WACV2025/papers/Turkulainen_DN-Splatter_Depth_and_Normal_Priors_for_Gaussian_Splatting_and_Meshing_WACV_2025_paper.pdf)
3. **프로젝트 페이지**: [maturk.github.io/dn-splatter](https://maturk.github.io/dn-splatter/)
4. **GitHub 코드**: [github.com/maturk/dn-splatter](https://github.com/maturk/dn-splatter)
5. **IEEE Xplore**: [DOI:10.1109/WACV61041.2025.00241](https://ieeexplore.ieee.org/document/10943388/)
6. **Aalto University Research Portal**: [research.aalto.fi](https://research.aalto.fi/en/publications/dn-splatter-depth-and-normal-priors-for-gaussian-splatting-and-me/)
7. **GaussianRoom (비교 연구)**: [arXiv:2405.19671](https://arxiv.org/html/2405.19671v1)
8. **3D Gaussian Splatting 최신 연구 리뷰**: Springer Computational Visual Media, "Recent advances in 3D Gaussian splatting" (2024)
9. **AGS-Mesh (후속 연구)**: [arXiv:2411.19271](https://arxiv.org/abs/2411.19271) — Ren et al., 3DV 2025
