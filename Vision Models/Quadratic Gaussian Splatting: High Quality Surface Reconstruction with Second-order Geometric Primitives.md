
# Quadratic Gaussian Splatting: High Quality Surface Reconstruction with Second-order Geometric Primitives (QGS)

> **논문 정보**
> - **제목**: Quadratic Gaussian Splatting: High Quality Surface Reconstruction with Second-order Geometric Primitives
> - **저자**: Ziyu Zhang, Binbin Huang, Hanqing Jiang, Liyang Zhou, Xiaojun Xiang, Shunhan Shen
> - **소속**: CASIA, The University of Hong Kong, SenseTime Research
> - **게재**: ICCV 2025
> - **arXiv**: [2411.16392](https://arxiv.org/abs/2411.16392)
> - **공식 코드**: [https://github.com/QuadraticGS/QGS](https://github.com/QuadraticGS/QGS)

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

QGS는 기존의 정적 프리미티브를 변형 가능한 이차 곡면(예: 타원, 포물면)으로 대체하는 새로운 표현 방식을 제안합니다. 기존 연구들이 변형 하에서 표면 기하학과 정렬되지 않는 유클리드 거리(Euclidean distance)를 밀도 모델링에 사용하는 것과 달리, QGS는 **측지 거리(geodesic distance) 기반 밀도 분포**를 도입합니다. 이를 통해 프리미티브가 곡률 변화에 따라 밀도 가중치를 본질적으로 적응시키며, 형상 변화(평면 디스크 → 곡선 포물면) 중에도 일관성을 유지합니다.

### 주요 기여 (3가지)

① **이차 곡면 기반 장면 프리미티브**: 더 강력한 기하학적 피팅 능력을 갖춘 새로운 미분 가능한(differentiable) 표현 방식인 QGS를 제안합니다. ② **측지 거리 기반 Gaussian 분포의 최초 도입**: Gaussian Splatting 분야에서 최초로 측지 거리를 이용해 표면 위에 Gaussian 분포를 수립, 프리미티브가 더 복잡한 텍스처를 피팅할 수 있도록 합니다. 이를 통해 이차 곡면 위에 Gaussian 분포를 수립하고 end-to-end 최적화를 가능하게 합니다. ③ **볼륨 렌더링 순서 문제 해결**: 2DGS의 정렬 방법을 개선하여 StopThePop의 정렬 기법을 통합함으로써 정밀한 기하학적 재구성과 고품질 렌더링을 달성합니다.

---

## 2. 해결하고자 하는 문제 / 제안 방법 / 모델 구조 / 성능 / 한계

### 2-1. 해결하고자 하는 문제

최근 3D Gaussian Splatting(3DGS)은 NeRF 대비 우수한 렌더링 품질과 속도로 주목받고 있습니다. 3DGS의 표면 표현 한계를 해결하기 위해 2D Gaussian Splatting(2DGS)이 디스크를 장면 프리미티브로 도입하여 멀티뷰 이미지로부터 기하학을 재구성하고 뷰 일관성 있는 기하학을 제공하였습니다. 그러나 **디스크의 1차 선형 근사(first-order linear approximation)는 과도하게 매끄러운(over-smoothed) 결과**를 자주 초래합니다.

NeRF와 같은 전통적 방법들은 렌더링 속도와 기하학적 정확도에서 종종 부족함을 보이며, 기존 Gaussian 기반 방법들은 표면을 효과적으로 모델링하지 못해 특히 복잡한 기하학에서 **과도 평활화(over-smoothing)** 문제**가 발생합니다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### (a) 이차 곡면 모델 (Quadratic Surface Model)

QGS의 핵심은 **Quadratic Gaussian Model**로, 동차 좌표계(homogeneous coordinate system)를 이용하여 이차 곡면을 정의합니다. 이 이차 형식(quadratic forms)은 타원체와 포물면 같은 형상을 포착할 수 있어 기하학적 유형 간의 미분 가능한 전환을 허용합니다. 핵심 수학적 표현은 다음과 같습니다:

$$f(x, y, z) = Ax^2 + 2Bxy + 2Cxz + 2Dx + Ey^2 + 2Fyz + 2Gy + Hz^2 + J = 0$$

동차 좌표 $\mathbf{x} = [x, y, z, 1]^T \in \mathbb{R}^4$ 로 표현하면, 이차 곡면은 $4 \times 4$ 대칭 행렬 $\mathbf{Q}$ 를 이용해 다음과 같이 간결하게 나타낼 수 있습니다:

$$\mathbf{x}^T \mathbf{Q} \mathbf{x} = 0$$

#### (b) 포물면으로의 제한 (Restriction to Paraboloid)

표면 모델링을 목표로 하므로 이차 표현을 포물면(paraboloid)으로 제한합니다. 여기서 평면 타원은 퇴화 케이스(degenerate case)로 포함됩니다. 이 공식화는 포인트 기반 렌더링 효율성을 유지하면서 적응형 곡률 모델링을 가능하게 하며, 단일 포물면이 여러 평면 서펠이 필요했던 복잡한 기하학을 근사하여 유사한 프리미티브 수로 더 높은 충실도를 달성합니다.

로컬 좌표계에서 포물면은 다음과 같이 정의됩니다:

$$z = s_3 \cdot (x^2 + y^2), \quad s_3 \in \mathbb{R}$$

여기서 $s_3 \to 0$ 이면 평면 디스크(2DGS의 프리미티브)로 수렴합니다.

#### (c) 측지 거리 기반 Gaussian 분포

표면 위의 임의의 점에서 Gaussian 가중치를 계산하기 위해 먼저 표면 위에 측도(measure)를 정의하여 Gaussian 분포를 수립합니다. 단순한 접근법은 유클리드 거리를 이용해 Gaussian 분포를 구성하는 것이지만, 이는 표면이 변형될 때 불일관성을 유발하여 고르지 않은 스플래팅 재구성 아티팩트로 이어집니다. 이를 해결하기 위해 **측지 거리(geodesic distance)** 를 사용하여 표면 변형에 일관되게 가중치를 적응시킵니다.

포물면 위 점 $P$에서 중심 $O$까지의 측지 거리 $l$은 다음 적분으로 정의됩니다:

$$l(\rho) = \int_0^{\rho} \sqrt{1 + 4s_3^2 r^2} \, dr$$

이를 통해 Gaussian 가중치는 다음과 같이 정의됩니다:

$$G(P) = \exp\left(-\frac{l(\rho)^2}{2\sigma^2}\right)$$

여기서 $\rho$는 프리미티브 로컬 좌표의 반경 거리이며, $\sigma$는 분산 파라미터입니다.

이는 2DGS를 QGS의 특수한 퇴화 케이스(degenerate case)로 볼 수 있음을 의미하며, QGS의 보다 일반화된 특성이 고곡률 영역을 효과적으로 피팅할 수 있게 합니다.

#### (d) Ray-Splat Intersection (광선-스플랫 교차)

서펠 기반 표현과 2DGS에서 영감을 받아 Ray-splat 교차(ray-splat intersection)를 사용하여 스플래팅하며, 멀티뷰 일관성과 원근 정확성을 유지합니다.

카메라 원점 $\hat{\mathbf{o}}$와 광선 방향 $\hat{\mathbf{d}}$에 대해, 이차 방정식으로 교차점을 구합니다:

$$a t^2 + b t + c = 0$$

$$a = \hat{d}_3^2 \cdot s_3, \quad b = \hat{o}_3 \hat{d}_3 s_3 - \frac{1}{2}, \quad c = \hat{o}_3^2 s_3 - \hat{o}_3$$

#### (e) 손실 함수 (Loss Function)

QGS의 최적화는 2DGS의 손실 설계를 확장하여 **곡률 정보(curvature)** 를 추가 정규화 항으로 활용합니다:

$$\mathcal{L} = \mathcal{L}_{\text{color}} + \lambda_1 \mathcal{L}_{\text{depth}} + \lambda_2 \mathcal{L}_{\text{normal}} + \lambda_3 \mathcal{L}_{\text{dist}}$$

2차 표면 근사로서, QGS는 법선 일관성 항을 안내하기 위한 공간 곡률(spatial curvature)을 렌더링하여 과도 평활화를 효과적으로 줄입니다.

#### (f) 렌더링 순서 개선

대부분의 GS 방법에서 사용되는 볼륨 렌더링 순서는 팝핑 아티팩트(popping artifacts)를 유발하며, 이는 신규 뷰 합성과 기하학적 재구성 모두에 영향을 미칩니다. 이를 해결하기 위해 StopThePop의 정렬 기준을 채택하고, 이를 이차 곡면 구조에 맞게 재구성하여 이차 곡면의 복잡한 교차를 더 잘 처리합니다.

---

### 2-3. 모델 구조

```
입력: 멀티뷰 RGB 이미지
     ↓
[초기화] SfM(COLMAP) 포인트 클라우드 → 이차 Gaussian 프리미티브 초기화
     ↓
[QGS 프리미티브] 각 프리미티브 파라미터:
  - 위치 μ ∈ ℝ³
  - 방향 (rotation) R
  - 곡률 파라미터 s₃ (convex/concave 전환)
  - 스케일 (s₁, s₂)
  - 불투명도 α
  - 색상(spherical harmonics)
     ↓
[Splatting]
  - Ray-Quadric Intersection (광선-이차 교차)
  - 측지 거리 기반 Gaussian 가중치 계산
  - Per-tile 정렬 + Per-pixel 재정렬
     ↓
[렌더링 출력]
  - 색상(RGB) 맵
  - 깊이(Depth) 맵
  - 법선(Normal) 맵
  - 곡률(Curvature) 맵
  - 깊이 왜곡(Depth Distortion) 맵
     ↓
[손실 계산 및 역전파]
  L = L_color + λ₁L_depth + λ₂L_normal + λ₃L_dist
     ↓
[메쉬 추출] TSDF Fusion → 고품질 메쉬 모델
```

QGS의 래스터라이저는 3DGS 프레임워크 위에 커스텀 CUDA 커널로 구현되었으며, 깊이 왜곡 맵, 깊이 맵, 법선 맵, 곡률 맵을 출력하도록 렌더러를 확장합니다.

---

### 2-4. 성능 향상

DTU, Tanks and Temples, MipNeRF360 데이터셋 실험에서 SOTA 표면 재구성을 달성하였으며, QGS는 DTU 데이터셋에서 **2DGS 대비 챔퍼 거리(chamfer distance)를 33% 감소**, **GOF 대비 27% 감소**시켰습니다. 특히 QGS는 경쟁력 있는 외관 품질을 유지하여 로봇공학 및 몰입형 현실 애플리케이션을 위한 기하학적 정밀도와 시각적 충실도 간의 격차를 해소합니다.

| 지표 | QGS vs 2DGS | QGS vs GOF |
|------|------------|-----------|
| Chamfer Distance (DTU) ↓ | **33% 감소** | **27% 감소** |
| F1-Score (TNT) | 2DGS 대비 향상 | SOTA 경쟁력 |

렌더링 과정에서는 Gaussian 스플래팅 방법에서 흔한 볼륨 렌더링 관련 팝핑 아티팩트를 완화하기 위한 고급 정렬 기술을 활용합니다. Per-tile 정렬과 Per-pixel 재정렬 방식이 통합되어 Gaussian들이 올바르게 블렌딩되도록 합니다.

---

### 2-5. 한계점

곡률 관련 손실(curvature distortion loss, curvature flatten loss) 실험을 진행하였으나, 아쉽게도 그 성능은 만족스럽지 않았습니다.

포물면의 비볼록(non-convex) 특성과 측지 거리 함수의 역함수 부재로 인해, 이미지 경계 박스 계산에 직사각형 절단 및 근사를 사용해야 했습니다.

QGS는 오목한 케이스를 포함하며, 측지 거리로 인해 전처리 시 Gaussian 프리미티브의 렌더링 부분을 해석적으로 결정하는 것이 복잡해집니다. 측지 함수는 초등 역함수(elementary inverse)가 없어 수치적 또는 근사 해법이 필요합니다.

추가적으로:
- 대규모 장면(aerial/street-view)에서는 이미지 수에 따른 학습 반복 횟수 조정이 별도로 필요합니다.
- 이차 곡면의 비볼록 특성으로 인한 렌더링 오버헤드가 존재합니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 일반화 가능성의 핵심 근거

QGS는 3D Gaussian 프리미티브를 일반화하고 2D Gaussian Splatting을 특수 케이스로 포함하며, **고차 표면 적응**(higher-order surface adaptation)을 허용하면서 형상 커널, 텍스처 빌보드, 가중치 분포와 같은 커널 수준의 혁신을 통합합니다.

이차 곡면에서의 측지 거리를 폐쇄형(closed form)으로 풀어냄으로써 QGS는 표면 인식 스플래팅을 가능하게 하며, 단일 프리미티브가 이전에 수십 개의 평면 서펠이 필요했던 복잡한 곡률을 표현할 수 있어, 빠른 광선-이차 교차를 통해 효율적인 렌더링을 유지하면서 메모리 사용량을 줄일 가능성이 있습니다.

### 3-2. 다양한 씬 유형으로의 일반화

QGS는 Gaussian Splatting에 이차 곡면을 최초로 도입하고, 비유클리드 공간에서 Gaussian 분포를 정의하여 피팅을 개선하고 2차 곡률을 포착합니다. 이를 통해 다양한 실내외 데이터셋에서 SOTA 기하학적 재구성 및 경쟁력 있는 렌더링 결과를 달성합니다.

2차 표면 근사로서 QGS는 법선 일관성 항을 안내하기 위한 공간 곡률을 렌더링하여 과도 평활화를 효과적으로 줄입니다. 더욱이 QGS는 2DGS의 일반화 버전으로서, DTU와 TNT 실험으로 검증된 것처럼 더 정확하고 세밀한 재구성을 달성합니다.

### 3-3. 일반화를 위한 향후 방향

향후 연구 방향은 QGS의 기능을 확장하고 다른 접근법과의 통합 가능성을 탐구하여 다양한 환경에서 더 강인한 재구성을 가능하게 하는 것입니다.

- **Sparse-view 일반화**: Gaussian Splatting을 장면 기하학에 최적화하는 최근 발전은 이미지로부터 세밀한 표면의 효율적인 재구성을 가능하게 했습니다. 그러나 입력 뷰가 희소할 때는 과적합(overfitting)에 취약하여 최적이 아닌 재구성 품질로 이어집니다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 방법 | 출처 | 핵심 프리미티브 | 기하학 정확도 | 특징 |
|------|------|--------------|-------------|------|
| NeRF (2020) | ECCV 2020 | 암시적 신경망 | 보통 | 느린 학습/추론 |
| 3DGS (2023) | SIGGRAPH 2023 | 3D 타원 Gaussian | 낮음(표면 부정확) | 빠른 렌더링 |
| 2DGS (2024) | SIGGRAPH 2024 | 2D 평면 디스크 | 중간(과도 평활화) | 뷰 일관성 |
| **QGS (2024/ICCV'25)** | arXiv 2411.16392 | **이차 포물면** | **SOTA** | 고곡률 포착 |
| SuGaR (2023) | arXiv 2311.12775 | 3D Gaussian + 메쉬 정렬 | 중간 | 빠른 메쉬 추출 |
| SparseSurf (2025) | arXiv 2511.14633 | 평면화 Gaussian | 중간 | Sparse-view 대응 |
| SurfaceSplat (2025) | arXiv 2507.15602 | SDF + 3DGS 하이브리드 | 높음 | 글로벌 일관성 |

후속 표면 재구성 방법으로서, GOF와 RadeGS와 같은 볼륨 방법들은 광선-스플랫 교차 기술을 활용하여 SOTA 재구성 품질을 달성하지만 법선과 깊이의 일관성을 제한합니다. 이와 대조적으로 2DGS는 평면 디스크 내에 2D Gaussian 분포를 정의하여 멀티뷰 간 일관된 법선과 깊이를 제공합니다. 그러나 디스크는 표면의 1차 근사에 불과하여 2DGS에서 과도하게 매끄러운 재구성 결과를 초래합니다.

SDF 기반 방법들은 세밀한 디테일에서 어려움을 겪는 반면, 3DGS 기반 접근법들은 전역 기하학 일관성이 부족합니다. SurfaceSplat은 두 접근법의 강점을 결합한 하이브리드 방법을 제안합니다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5-1. 연구에 미치는 영향

**① 기하학적 프리미티브 설계 패러다임 전환**

QGS는 GS 계열 방법의 변형으로서 정확한 장면 기하학 재구성과 더 세밀한 디테일 복원을 위해 설계되었습니다. QGS는 **Gaussian Splatting에 이차 곡면을 최초로 도입**하여 비유클리드 공간에서 Gaussian 분포를 정의함으로써 피팅을 향상시키고 2차 곡률을 포착합니다. 이는 Gaussian 기반 표현의 표현력 한계를 극복하는 새로운 방향을 제시합니다.

**② 측지 거리의 3D 표현 학습 적용 가능성**

이 논문은 비유클리드 거리 메트릭을 Gaussian 분포 설계에 활용할 수 있음을 실증하였으며, 이는 향후 메쉬 기반 학습, 포인트 클라우드 처리, 동적 씬 표현 등 다양한 분야에 영향을 줄 수 있습니다.

**③ 로봇공학 및 AR/VR 응용**

DTU, Tanks and Temples, MipNeRF360 데이터셋 실험에서 QGS는 SOTA 표면 재구성을 달성하였으며, **로봇공학 및 몰입형 현실(immersive reality)**과 같은 응용 분야를 위해 기하학적 정밀도와 시각적 충실도 간의 격차를 해소합니다.

**④ 대규모 씬 재구성에의 영향**

2025년 8월 기준 원래 직사각형 경계 박스를 더 컴팩트한 절단 원뿔형(truncated cone-shaped) 경계 박스로 교체하여 유효하지 않은 렌더링 영역을 크게 줄이고 **2배의 속도 향상**을 달성하였습니다. 이는 대규모 장면 처리 연구에 실용적인 기반을 제공합니다.

---

### 5-2. 향후 연구 시 고려할 점

**① 곡률 손실 함수 설계 개선**

곡률 왜곡 손실(curvature distortion loss)과 곡률 평탄화 손실(curvature flatten loss) 등의 곡률 관련 손실을 실험하였으나, 아쉽게도 그 성능이 만족스럽지 않았습니다. 따라서 곡률 정규화를 보다 효과적으로 활용할 수 있는 새로운 손실 함수 설계가 중요한 연구 방향입니다.

**② Sparse-view 일반화 강화**

기존 접근법들은 평탄화된 Gaussian 프리미티브와 깊이 정규화를 결합하지만, 평탄화 Gaussian의 증가된 이방성(anisotropy)은 희소 뷰 시나리오에서 과적합을 악화시켜 정확한 표면 피팅과 신규 뷰 합성 성능을 저하시킵니다. QGS 프리미티브를 희소 뷰 시나리오에서도 안정적으로 최적화하는 연구가 필요합니다.

**③ 동적 씬(Dynamic Scene)으로의 확장**

현재 QGS는 정적 씬에 초점을 맞추고 있으므로, 포물면 프리미티브를 시간 축으로 확장하여 동적 객체를 처리하는 연구가 필요합니다.

**④ 더 높은 차수의 표면 표현**

QGS가 2차(quadratic) 표현을 도입했다면, 3차(cubic) 또는 NURBS 기반의 더 고차 표현과의 비교 및 적용 가능성 탐색이 향후 연구 방향이 될 수 있습니다.

**⑤ 대규모 씬 및 실시간 처리**

대규모 씬으로의 확장은 높은 계산 요구량과 실외 환경의 복잡한 동적 외관으로 인해 여전히 도전적입니다. 이는 항공 측량 및 자율 주행 적용을 방해합니다.

**⑥ 일반화 성능을 위한 데이터 다양성**

QGS의 성능이 DTU, TNT, MipNeRF360에서 입증되었지만, 의료 영상, 수중 환경, 야간 조건 등 더 다양한 도메인에서의 일반화 성능 검증이 필요합니다.

---

## 참고 자료 및 출처

| # | 자료 | 링크 |
|---|------|------|
| 1 | **[주 논문] arXiv:2411.16392v4** (ICCV 2025) | https://arxiv.org/abs/2411.16392 |
| 2 | **arXiv HTML 전문 (v3)** | https://arxiv.org/html/2411.16392v3 |
| 3 | **arXiv HTML 전문 (v1)** | https://arxiv.org/html/2411.16392v1 |
| 4 | **공식 프로젝트 페이지** | https://quadraticgs.github.io/QGS/ |
| 5 | **공식 GitHub (QuadraticGS)** | https://github.com/QuadraticGS/QGS |
| 6 | **공식 GitHub (will-zzy)** | https://github.com/will-zzy/QGS |
| 7 | **ICCV 2025 포스터 페이지** | https://iccv.thecvf.com/virtual/2025/poster/2697 |
| 8 | **ICCV 2025 보충 자료 (PDF)** | https://www.openaccess.thecvf.com/content/ICCV2025/supplemental/Zhang_Quadratic_Gaussian_Splatting_ICCV_2025_supplemental.pdf |
| 9 | **[리뷰] themoonlight.io 리뷰** | https://www.themoonlight.io/en/review/quadratic-gaussian-splatting-for-efficient-and-detailed-surface-reconstruction |
| 10 | **[비교] SuGaR (arXiv:2311.12775)** | https://arxiv.org/abs/2311.12775 |
| 11 | **[비교] SparseSurf (arXiv:2511.14633)** | https://arxiv.org/abs/2511.14633 |
| 12 | **[비교] SurfaceSplat (arXiv:2507.15602)** | https://arxiv.org/abs/2507.15602 |
| 13 | **ResearchGate PDF** | https://www.researchgate.net/publication/386111681 |
| 14 | **alphaXiv 개요** | https://www.alphaxiv.org/overview/2411.16392v4 |
