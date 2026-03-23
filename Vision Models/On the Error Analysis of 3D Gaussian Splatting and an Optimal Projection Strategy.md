# On the Error Analysis of 3D Gaussian Splatting and an Optimal Projection Strategy

**저자:** Letian Huang, Jiayang Bai, Jie Guo, Yuanqi Li, Yanwen Guo
**발표:** ECCV 2024 (Computer Vision – ECCV 2024, Springer, pp. 247–263)

---

## 1. 핵심 주장과 주요 기여 요약

3D Gaussian Splatting(3D-GS)은 포인트 클라우드 저장, 성능, 희소 시점에서의 강건성 등 다양한 한계가 제기되어 왔지만, splatting 자체에 내재된 **local affine approximation이 유발하는 투영 오차(projection error)**라는 근본적인 문제에 대해서는 주목이 부족했다.

본 논문은 투영 함수의 1차 Taylor 전개에서 발생하는 잔여 오차(residual error)를 분석하여, 이 오차와 Gaussian 평균 위치(mean position) 사이의 상관관계를 밝히고 있다. 이어서 함수 최적화 이론을 활용하여 오차 함수의 극솟값을 분석하고, 다양한 카메라 모델에 적용 가능한 **Optimal Gaussian Splatting**이라는 최적 투영 전략을 제안한다.

### 주요 기여 (3가지):


1. **오차 분석:** 3D-GS의 투영 과정에서 아티팩트를 유발하고 렌더링 품질을 저하시키는 오차에 대한 철저한 분석을 수행하며, 이 오차와 Gaussian 위치 사이의 상관관계를 규명
2. **수학적 기대값 유도 및 극값 분석:** 오차 함수의 수학적 기대값을 유도하고, 함수 최적화 방법론을 통해 극값 조건을 분석
3. **최적 투영 제안:** 각 Gaussian의 평균에서 카메라 중심 방향으로의 접선 평면(tangent plane) 투영을 사용하여, 기존의 단일 평면 투영 대신 다양한 카메라 모델에 적응 가능한 최적 투영 방식 제안


---

## 2. 상세 분석: 문제, 방법론, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

Gaussian 함수는 아핀 변환(affine transformation) 하에서 Gaussian 성질을 유지하지만, **투영 변환(projective transformation)** 하에서는 반드시 그렇지 않다. 따라서 3D-GS는 투영 함수를 Taylor 전개의 처음 두 항으로 근사하는 **local affine approximation**을 채택하는데, 이 근사는 렌더링 이미지에서 아티팩트를 유발하는 오차를 수반한다.

특히, 화각(FOV)이 확장되고 초점 거리(focal length)가 감소할수록 3D-GS의 투영 오차가 급격히 증가하며, 이는 바늘 형태(needle-like)의 늘어난 Gaussian이나 구름 형태(cloud-like)의 Gaussian 아티팩트를 더 많이 발생시켜 전체 이미지 품질을 심각하게 저하시킨다.

### 2.2 수학적 배경 및 제안 방법

#### (a) 3D Gaussian의 정의

3D Gaussian은 평균(mean) $\boldsymbol{\mu}$와 공분산 행렬(covariance matrix) $\boldsymbol{\Sigma}$로 정의되며, 각 Gaussian은 투명도 $\alpha$와 방향별 색상을 나타내는 구면 조화 함수(SH)를 가진다:

```math
G(\mathbf{x}) = \exp\left\{-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})\right\}
```

#### (b) 뷰 변환 및 Local Affine Approximation

월드 좌표에서 카메라 좌표로의 변환 후, 3D Gaussian을 이미지 평면으로 투영할 때 Gaussian 함수가 투영 변환 하에서 Gaussian 성질을 유지하지 않으므로, Taylor 1차 전개(local affine approximation)를 사용하여 투영 함수를 근사한다. 이 근사는 렌더링 이미지에서 아티팩트, 왜곡, 부정확성을 유발하는 오차를 필연적으로 도입한다.

뷰 변환 후의 Gaussian은 다음과 같이 표현된다:

```math
G'(\mathbf{x}) = \exp\left\{-\frac{1}{2}(\mathbf{W}\mathbf{x} - \mathbf{W}\boldsymbol{\mu})^\top (\mathbf{W}\boldsymbol{\Sigma}\mathbf{W}^{\top})^{-1}(\mathbf{W}\mathbf{x} - \mathbf{W}\boldsymbol{\mu})\right\}
```

여기서 $\mathbf{W}$는 뷰 변환 행렬이다.

#### (c) 투영 오차 함수 분석

본 논문은 Taylor 나머지 항(remainder term)을 분석하여, 3D-GS의 오차와 Gaussian 평균 위치 사이의 관계를 밝힌다. 또한, 오차 함수의 극값(extremum)을 결정하여 오차가 최소화되는 조건을 식별한다.

투영 오차의 수학적 기대값 $\epsilon$은 Gaussian 평균의 구면 좌표 $(\theta_\mu, \phi_\mu)$의 함수로 표현된다:

$$
\epsilon = \epsilon(\theta_\mu, \phi_\mu)
$$

임계점(critical points)을 구하면, **Gaussian 평균의 투영이 카메라 중심의 평면 상 투영과 일치할 때** 오차 함수가 최솟값을 갖는다는 것이 밝혀진다.

#### (d) Optimal Gaussian Splatting (최적 투영 전략)

오차 함수의 극값 분석을 기반으로 3D-GS의 투영 오차를 최소화하는 최적 투영 방법을 제안한다. 이 방법은 소수의 코드 수정만 필요하고, 실시간 렌더링 성능에 영향을 주지 않으면서도 렌더링 품질에서 유의미한 향상을 달성한다. 구체적으로, Gaussian 평균에서 카메라 중심 방향으로 투영하여, 투영 평면이 이 연결선에 접하는 접선 평면(tangent plane)이 되도록 한다.

렌더링 파이프라인은 다음 단계로 구성된다:


1. **월드→카메라 좌표 변환:** 기존 3D-GS와 동일
2. **Optimal Projection:** $z=1$ 평면 대신, 각 Gaussian을 그 평균과 카메라 중심을 잇는 선을 따라 방사형으로 투영하여, 단위 구(unit sphere)에 접하고 이 선에 수직인 접선 평면 위에 투영
3. **알파 블렌딩:** 단위 구 위의 점들에 대해 접선 평면 상의 2D Gaussian을 알파 블렌딩하여 색상 계산
4. **이미지 생성:** 각 픽셀에 대해 단위 구에 레이를 캐스팅하여 해당 픽셀의 색상을 검색


핀홀(pinhole) 카메라 모델에서 Gaussian 위치에 기반한 카메라 좌표 계산:

```math
\mathbf{x}_{2D} = \begin{bmatrix} \dfrac{t_x}{\mu_x t_x + \mu_y t_y + \mu_z t_z} \\[6pt] \dfrac{t_y}{\mu_x t_x + \mu_y t_y + \mu_z t_z} \\[6pt] \dfrac{t_z}{\mu_x t_x + \mu_y t_y + \mu_z t_z} \end{bmatrix}
```

이미지 좌표에서 접선 평면(tangent plane)으로의 좌표 변환:

$$
\varphi_p\left(\begin{bmatrix} (u - c_x)/f_x \\ (v - c_y)/f_y \\ 1 \end{bmatrix}\right)
$$

접선 평면 위에서의 2D Gaussian:

```math
G_{2D}(\mathbf{x}_{2D}) = \exp\left\{-\frac{1}{2}\left(\mathbf{x}_{2D} - \varphi_p(\boldsymbol{\mu}')\right)^\top \left(\mathbf{J}_p \boldsymbol{\Sigma}' \mathbf{J}_p^\top\right)^{-1}\left(\mathbf{x}_{2D} - \varphi_p(\boldsymbol{\mu}')\right)\right\}
```

여기서 $\mathbf{J}_p$는 접선 평면 투영의 야코비안(Jacobian) 행렬이다.

### 2.3 모델 구조

| 구성 요소 | 기존 3D-GS | Optimal Gaussian Splatting |
|---|---|---|
| **투영 대상** | 모든 Gaussian을 $z=1$ 평면에 투영 | 각 Gaussian을 개별 접선 평면에 투영 |
| **투영 방향** | 카메라 광축 방향 | Gaussian 평균→카메라 중심 방사형 |
| **카메라 모델** | 핀홀(pinhole) 전용 | 이미지 공간에서 카메라 공간으로의 변환을 수정함으로써 다양한 카메라 모델(파노라마, 어안 렌즈 등)에 적응 가능 |
| **렌더링 방식** | 타일 기반 래스터화 | 레이 캐스팅 + 접선 평면 교차 |

### 2.4 성능 향상

투영 오차를 최소화하는 오차 분석을 통해 원래 3D-GS 대비 렌더링 이미지 품질에서 향상을 달성하였다.

정량적 비교에서는 Mip-NeRF360이 제안한 train/test split 방식을 따라, 매 8번째 사진을 테스트용으로 보류하고, PSNR, LPIPS, SSIM 등 표준 메트릭으로 평가하였다.

기존 방법과 비교하여 제안 방법은 더 큰 사실감(realism)과 강건성(robustness)을 보이며, 투영 오차로 인한 흐림(blurriness)과 아티팩트를 크게 감소시켰다.

Mip-Splatting과 Scaffold-GS가 여전히 기존 3D-GS의 전통적인 투영 방식을 사용하므로, 이들 대비 오차가 더 작다.

### 2.5 한계

논문에서 명시적으로 언급된 한계로, **현재 CUDA 구현이 느려서 최적화가 필요하다**는 점이 있다. 이는 접선 평면 투영 및 레이 캐스팅 과정이 기존의 단순 타일 기반 래스터화보다 계산 비용이 높기 때문이다.

---

## 3. 모델의 일반화 성능 향상 가능성

본 논문은 여러 측면에서 **일반화 성능 향상**에 기여한다:

### 3.1 다양한 카메라 모델 지원

제안된 Optimal Gaussian Splatting은 다양한 카메라 모델을 수용할 수 있다. 실제로 핀홀(pinhole), 파노라마(panorama), 어안(fisheye) 카메라에 대한 래스터화 서브모듈이 구현되어 있다. 기존 3D-GS는 핀홀 카메라에 주로 한정되었던 것과 대비된다.

### 3.2 FOV 변화에 대한 강건성

초점 거리가 감소하면 화각이 확장되어, 더 많은 Gaussian이 투영 중심에서 이탈하고 전체 투영 오차가 증가한다. 이 경우 기존 3D-GS는 바늘 형태나 구름 형태의 아티팩트가 더 많이 나타나 전체 이미지 품질이 크게 저하되지만, 중심 방사형 투영(central radial projection)을 사용하는 제안 방법은 이러한 결함이 발생하지 않는다.

### 3.3 이론적 일반화 근거

투영 오차가 Gaussian 평균 위치의 함수임을 수학적으로 증명하고, 이 오차를 최소화하는 최적 투영 지점을 도출함으로써, 임의의 장면이나 시점 구성에서도 **이론적으로 보장된 오차 하한(error lower bound)**을 제공한다. 이는 특정 데이터셋에 종속되지 않는 일반적인 개선이다.

### 3.4 넓은 FOV/짧은 초점 거리 시나리오

특히 VR/AR 응용에서 중요한 넓은 화각 렌더링에서 기존 방법 대비 큰 폭의 품질 향상이 기대된다. 이는 제안된 접선 평면 투영이 투영 중심에서 멀리 떨어진 Gaussian에 대해서도 오차를 효과적으로 억제하기 때문이다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구 영향

투영 및 기하학적 왜곡(특히 광각 또는 주변 시야에서의 Gaussian 왜곡)은 3DGS의 아티팩트를 유발하는 미묘한 원인이며, 이 논문의 1차 오차 분석은 Gaussian 공분산 텐서를 뷰 접선 평면에 정렬하는 개선된 투영 체계를 도출하여 체계적 왜곡을 줄였다.

이러한 연구들은 Gaussian 프리미티브 기하학과 투영 기하학 사이의 정렬이 고품질 novel-view synthesis에 핵심적임을 강조한다. 이러한 수학적 탐구들은 3DGS의 이론적 기반을 깊이 있게 만들고 있다.

후속 연구인 360-GS가 이 저장소를 기반으로 3DV 2025에 채택되었다.

### 4.2 향후 연구 시 고려할 점

1. **실시간 성능 최적화:** 현재 CUDA 구현의 속도 문제 해결이 필수. 타일 기반 래스터화의 효율성을 유지하면서 접선 평면 투영을 통합하는 하이브리드 접근이 필요
2. **고차 오차 분석:** 1차 Taylor 잔여항을 넘어 고차 항까지 고려하면 더 정밀한 오차 모델 가능
3. **동적 장면 확장:** 4D Gaussian Splatting이나 동적 장면에서의 시간 변화에 따른 투영 오차 분석
4. **압축과의 통합:** HAC++, Mini-Splatting 등의 압축 기법과 최적 투영의 결합
5. **스케일 적응성:** Mip-Splatting의 주파수 필터링과 결합하여 다중 스케일 렌더링에서의 일반화 강화

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문/방법 | 연도 | 핵심 접근 | 이 논문과의 비교 |
|---|---|---|---|
| **NeRF** (Mildenhall et al.) | 2020 | MLP 기반 암시적 장면 표현 | 실시간 렌더링 불가, 3D-GS가 명시적 표현으로 대체 |
| **3D Gaussian Splatting** (Kerbl et al.) | 2023 | MLP 대신 Gaussian 함수를 이용한 명시적 표현으로 볼륨 렌더링의 밀집 샘플링 비용을 회피하여 실시간 성능 달성 | 본 논문이 해결하는 투영 오차 문제의 원본 방법 |
| **Mip-Splatting** (Yu et al., CVPR 2024 Best Student Paper) | 2024 | 3D 주파수 제약 부재와 2D dilation 필터 사용이 문제의 원인임을 밝히고, 3D smoothing filter와 2D Mip filter 도입 | **스케일/앨리어싱** 문제에 초점. 여전히 기존 3D-GS의 투영 방식을 사용하므로 투영 오차는 본 논문 대비 크다 |
| **Scaffold-GS** (Lu et al., 2024) | 2024 | 앵커 포인트를 사용하여 로컬 3D Gaussian을 분배하고, 뷰 방향과 거리에 따라 속성을 실시간 예측하여 중복 Gaussian 감소 | **밀도 제어**에 초점. 투영 자체는 기존 방식 유지 |
| **2DGS** (Huang et al., SIGGRAPH 2024) | 2024 | 2D Gaussian 디스크로 기하학적 정확도 향상 | 기하학적 표면 정밀도에 초점, 투영 오차 분석과는 상호보완적 |
| **Mipmap-GS** (2024) | 2024 | 스케일 적응형 Gaussian으로 멀티 스케일 렌더링, pseudo-GT 기반 최적화 | 멀티 스케일 일반화에 초점. 투영 전략 자체는 변경 안 함 |
| **3DGUT** (Wu et al., CVPR 2025) | 2025 | 왜곡 카메라와 2차 광선을 위한 렌더링 조정 | 투영 왜곡 보정이라는 유사한 목표를 공유하지만 접근 방식이 상이 |
| **FlashGS** (CVPR 2025) | 2025 | 4K 대규모 장면의 실시간 렌더링을 위해 래스터화 파이프라인을 재설계 | 성능/확장성에 초점 |

### 주요 차별점 정리

본 논문의 가장 핵심적인 차별점은 **투영 오차의 이론적 분석에서 출발하여 최적 투영 전략을 도출한 것**이다. 대부분의 후속 연구들이 밀도 제어, 압축, 앨리어싱, 스케일 적응 등 실용적 개선에 집중한 반면, 이 연구는 3D-GS의 근본적인 수학적 한계를 분석하고 이론적으로 정당화된 해결책을 제시했다는 점에서 **이론적 기여도**가 높다.

---

## 참고자료

1. **arXiv 원문:** Huang, L., Bai, J., Guo, J., Li, Y., & Guo, Y. (2024). "On the Error Analysis of 3D Gaussian Splatting and an Optimal Projection Strategy." arXiv:2402.00752 — https://arxiv.org/abs/2402.00752
2. **ECCV 2024 공식 PDF:** https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02588.pdf
3. **Springer 출판본:** Lecture Notes in Computer Science, vol 15075, pp. 247–263 — https://link.springer.com/chapter/10.1007/978-3-031-72643-9_15
4. **프로젝트 페이지:** https://letianhuang.github.io/op43dgs/
5. **GitHub 코드 저장소:** https://github.com/LetianHuang/op43dgs
6. **arXiv HTML 버전 (v4):** https://arxiv.org/html/2402.00752v4
7. **Semantic Scholar:** https://www.semanticscholar.org/paper/ae002bbeeb8cdec0013bd4555b27542fc9ea5be2
8. **Mip-Splatting (Yu et al., CVPR 2024):** https://arxiv.org/abs/2311.16493
9. **"The Impact and Outlook of 3D Gaussian Splatting" (2025 Survey):** https://arxiv.org/html/2510.26694v1
10. **AI Models 분석:** https://www.aimodels.fyi/papers/arxiv/error-analysis-3d-gaussian-splatting-optimal-projection
