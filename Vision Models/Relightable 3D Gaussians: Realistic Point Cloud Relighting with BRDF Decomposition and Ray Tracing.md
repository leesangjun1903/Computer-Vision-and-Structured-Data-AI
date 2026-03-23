# Relightable 3D Gaussians: Realistic Point Cloud Relighting with BRDF Decomposition and Ray Tracing

---

## 1. 핵심 주장 및 주요 기여 요약

**Relightable 3D Gaussians (R3DG)**는 3D Gaussian Splatting (3DGS) 프레임워크를 확장하여, 다시점 이미지로부터 **재조명 가능한(relightable)** 3D 장면을 복원하는 새로운 미분 가능한 포인트 기반 렌더링 파이프라인을 제안한다. 핵심 주장은 다음과 같다:

1. **BRDF 분해 및 조명 모델링**: 각 3D Gaussian에 법선 벡터, BRDF 파라미터(albedo, roughness), 입사광 정보를 부여하여 물리 기반 렌더링(PBR)을 수행한다.
2. **포인트 기반 레이 트레이싱**: Bounding Volume Hierarchy (BVH) 기반의 새로운 레이 트레이싱 방법을 도입하여, 이산적 포인트 표현에서도 정확한 가시성(visibility)을 효율적으로 사전 계산하고 사실적인 그림자 효과를 구현한다.
3. **통합 그래픽스 파이프라인**: 편집(editing), 레이 트레이싱(ray tracing), 재조명(relighting)을 모두 지원하는 완전한 포인트 기반 그래픽스 파이프라인을 시연한다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 3DGS [Kerbl et al., 2023]는 실시간 고품질 Novel View Synthesis(NVS)를 달성하지만, 다음과 같은 근본적 한계를 가진다:

- **재조명 불가**: 장면의 외형(appearance)을 Spherical Harmonics로 직접 인코딩하기 때문에, 조명 조건 변경 시 새로운 렌더링이 불가능하다.
- **레이 트레이싱 부재**: 포인트 기반 표현에서는 그림자, 반사 등 광선 추적 기반 효과를 구현하기 어렵다.
- **재질-조명 모호성(ambiguity)**: 다시점 이미지만으로 BRDF와 조명을 분리하는 것은 본질적으로 ill-posed 문제이다.

### 2.2 제안 방법 (수식 포함)

#### (A) 3DGS 기초 (Preliminary)

3D Gaussian은 다음과 같이 정의된다:

$$G(\boldsymbol{x}) = \exp\left(-\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^\top \Sigma^{-1}(\boldsymbol{x}-\boldsymbol{\mu})\right)$$

여기서 $\boldsymbol{\mu}$는 공간 평균, $\Sigma$는 공분산 행렬이다. 각 Gaussian은 불투명도 $o$와 뷰 의존 색상 $\boldsymbol{c}$를 가진다. 픽셀 색상은 front-to-back alpha blending으로 합성된다:

$$\mathcal{C} = \sum_{i \in N} T_i \alpha_i \boldsymbol{c}_i, \quad T_i = \prod_{j=1}^{i-1}(1-\alpha_j)$$

#### (B) 기하학 향상 (Geometry Enhancement)

**법선 추정**: 각 3D Gaussian에 법선 속성 $\boldsymbol{n}$을 부여하고 역전파로 최적화한다. 깊이와 법선은 가중 합으로 렌더링된다:

$$\{\mathcal{D}, \mathcal{N}\} = \sum_{i \in N} w_i \{d_i, \boldsymbol{n}_i\}$$

여기서 $w_i = T_i \alpha_i / \sum_{i \in N} T_i \alpha_i$이다. 렌더링된 법선 $\mathcal{N}$과 깊이로부터 계산한 의사 법선 $\tilde{\mathcal{N}}$ 사이의 일관성을 강제한다:

$$\mathcal{L}_n = \|\mathcal{N} - \tilde{\mathcal{N}}\|_2$$

**깊이 분포 제약**: 정확한 표면을 가정하여 깊이 불확실성을 최소화한다:

$$\mathcal{L}_u = \mathcal{D}_{sq} - \mathcal{D}^2$$

여기서 $\mathcal{D}\_{sq} = \sum_{i \in N} w_i d_i^2$이다.

**법선 기울기 기반 밀도화 (Normal Gradient Based Densification)**: 법선 기울기가 임계값 $T_n$을 초과하는 Gaussian을 추가 밀도화하여 얇은 구조의 법선 복원을 개선한다.

**오브젝트 마스크 제약**:

$$\mathcal{L}_O = -M \log O - (1-M) \log(1-O)$$

#### (C) BRDF 및 조명 모델링

**렌더링 방정식**:

$$L_o(\boldsymbol{\omega_o}, \boldsymbol{x}) = \int_{\Omega} f(\boldsymbol{\omega_o}, \boldsymbol{\omega_i}, \boldsymbol{x}) L_i(\boldsymbol{\omega_i}, \boldsymbol{x})(\boldsymbol{\omega_i} \cdot \boldsymbol{n}) d\boldsymbol{\omega_i}$$

**BRDF 매개변수화**: 간소화된 Disney BRDF 모델 [Burley, 2012]을 채택한다. BRDF는 diffuse 항 $f_d = \frac{\boldsymbol{b}}{\pi}$ (albedo $\boldsymbol{b} \in [0,1]^3$)과 specular 항으로 분리된다:

$$f_s(\boldsymbol{\omega}_o, \boldsymbol{\omega}_i) = \frac{D(\boldsymbol{h}; r) \cdot F(\boldsymbol{\omega}_o, \boldsymbol{h}) \cdot G(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o, h; r)}{(\boldsymbol{n} \cdot \boldsymbol{\omega}_i) \cdot (\boldsymbol{n} \cdot \boldsymbol{\omega}_o)}$$

여기서 $\boldsymbol{h} = (\boldsymbol{\omega}_i + \boldsymbol{\omega}_o)/2$은 반벡터(half vector), $D$, $F$, $G$는 각각 법선 분포 함수, Fresnel 항, 기하 항이다.

**입사광 모델링**: 입사광을 전역 직접광(global environment map)과 개별 Gaussian별 간접광으로 분리한다:

$$L_i(\boldsymbol{\omega}_i) = V(\boldsymbol{\omega}_i) \cdot L_{direct}(\boldsymbol{\omega}_i) + L_{indirect}(\boldsymbol{\omega}_i)$$

- $V(\boldsymbol{\omega}_i)$: 가시성 항 (BVH 기반 레이 트레이싱으로 계산)
- $L_{direct}$: 16×32 환경 맵 $\boldsymbol{l}^{env}$로 매개변수화
- $L_{indirect}$: 3차 Spherical Harmonics $\boldsymbol{l}$로 매개변수화

**PBR 색상 계산**: 각 Gaussian에 대해 반구 위 $N_s = 64$ 방향의 Fibonacci 샘플링으로 수치 적분:

$$\boldsymbol{c}'(\boldsymbol{\omega}_o) = \sum_{i=0}^{N_s} (f_d + f_s(\boldsymbol{\omega}_o, \boldsymbol{\omega}_i)) L_i(\boldsymbol{\omega}_i) (\boldsymbol{\omega}_i \cdot \boldsymbol{n}) \Delta \boldsymbol{\omega}_i$$

#### (D) 정규화 (Regularizations)

**조명 정규화** (자연광이 백색에 가깝다는 가정):

$$\mathcal{L}_{light} = \sum_c \left(L_c - \frac{1}{3}\sum_c L_c\right), \quad c \in \{R, G, B\}$$

**평활도 사전(Smoothness Prior)**:

$$\mathcal{L}_{s,r} = \|\nabla R\| \exp(-\|\nabla C_{gt}\|)$$

roughness, normal, albedo에 대해 유사한 평활도 제약이 적용된다.

#### (E) 포인트 기반 레이 트레이싱

BVH 기반으로 레이-Gaussian 교차를 효율적으로 계산한다. 반투명 Gaussian과 레이의 **등가 교차점(equivalent intersection point)**은 Gaussian 기여가 최대인 점으로 근사된다:

$$\boldsymbol{r_x} = \boldsymbol{r_o} + t_j \boldsymbol{r_d}$$

$$t_j = \frac{(\boldsymbol{\mu} - \boldsymbol{r_o})^T \Sigma \boldsymbol{r_d}}{\boldsymbol{r_d}^T \Sigma \boldsymbol{r_d}}$$

투과율(transmittance)은 순서에 무관하게 누적된다:

$$T_i = (1-\alpha_{i-1}) T_{i-1}, \quad T_1 = 1$$

임계값 $T_{min}$ 이하로 투과율이 떨어지면 조기 종료(early termination)한다.

### 2.3 모델 구조

전체 파이프라인은 **2단계(two-stage)** 최적화로 구성된다:

| 단계 | 목표 | 최적화 대상 | 반복 횟수 |
|------|------|------------|-----------|
| Stage 1 | 기하학 + 법선 최적화 | $\{\boldsymbol{\mu}, \boldsymbol{q}, \boldsymbol{s}, o, \boldsymbol{c}, \boldsymbol{n}\}$ | 30,000 |
| (중간) | BVH 기반 가시성 사전 계산 | $V(\boldsymbol{\omega}_i)$ | - |
| Stage 2 | 재질 + 조명 최적화 (기하학 고정) | $\{\boldsymbol{b}, r, \boldsymbol{l}, \boldsymbol{l}^{env}\}$ | 10,000 |

**Stage 1 손실 함수**:

$$\mathcal{L} = \lambda_1 \mathcal{L}_1 + \lambda_{ssim} \mathcal{L}_{ssim} + \lambda_n \mathcal{L}_n + \lambda_{s,n} \mathcal{L}_{s,n} + \lambda_O \mathcal{L}_O + \lambda_u \mathcal{L}_u$$

**Stage 2 손실 함수**:

$$\mathcal{L} = \lambda_1 \mathcal{L}_1 + \lambda_{ssim} \mathcal{L}_{ssim} + \lambda_l \mathcal{L}_l + \lambda_{s,b} \mathcal{L}_{s,b} + \lambda_{s,r} \mathcal{L}_{s,r}$$

각 Gaussian $\mathcal{P}_i$는 최종적으로 $\{\boldsymbol{\mu}_i, \boldsymbol{q}_i, \boldsymbol{s}_i, o_i, \boldsymbol{c}_i, \boldsymbol{n}_i, \boldsymbol{b}_i, r_i, \boldsymbol{l}_i\}$로 매개변수화된다.

### 2.4 성능 향상

#### NeRF Synthetic Dataset (NVS)

| Method | Geometry | Relightable | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|----------|-------------|-------|-------|--------|
| 3DGS [26] | point | ✘ | 33.88 | 0.970 | 0.031 |
| Nvdiffrec [33] | mesh | ✔ | 29.05 | 0.939 | 0.081 |
| NeILF++ [49] | neural | ✔ | 26.37 | 0.911 | 0.091 |
| **R3DG (Ours)** | point | ✔ | **31.22** | **0.959** | **0.039** |

#### Synthetic4Relight Dataset (Relighting)

| Method | NVS PSNR↑ | Relight PSNR↑ | Time (hours) |
|--------|-----------|---------------|--------------|
| TensoIR [23] | 35.80 | 29.69 | 3.24 |
| **R3DG (Ours)** | **36.80** | **31.00** | **0.90** |

R3DG는 relightable 방법 중 **최고 성능**을 달성하면서, 학습 시간은 기존 방법 대비 **3.6~53배 빠르다**.

### 2.5 한계

논문에서 명시적으로 언급된 한계:

1. **정적 장면에 한정**: 동적 장면이나 움직이는 물체에는 적용 불가.
2. **대규모 장면의 확장성 문제**: 포인트 밀도가 높아지면 각 포인트에서의 PBR 연산(레이 샘플링)으로 인해 최적화 속도가 저하된다.
3. **간접광의 제한적 표현**: 3차 SH로 간접광을 표현하므로, 복잡한 inter-reflection이나 고주파 간접 조명을 완전히 포착하기 어렵다.
4. **재질-조명 모호성의 완전한 해소 불가**: 정규화를 통해 완화하지만, 본질적으로 ill-posed 문제이다.
5. **기하학 정확도**: 3DGS의 "소프트" 포인트 특성 상, 정밀한 표면 복원에 한계가 있으며, MVS 통합이 필요하다고 언급한다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재의 일반화 능력

R3DG는 다음 측면에서 일반화 가능성을 보여준다:

- **합성 데이터 → 실세계 데이터**: NeRF Synthetic에서 학습/검증 후 Mip-NeRF 360 실세계 장면에서도 설득력 있는 재조명 결과를 시연하였다 (Fig. 6).
- **다중 객체 조합(Multi-object Composition)**: 개별 객체를 복원한 후, 새로운 장면으로 조합하고 복잡한 상호 차폐(inter-object occlusion)를 BVH 레이 트레이싱으로 처리하여 사실적 그림자를 생성한다 (Fig. 1).
- **명시적 표현의 이점**: 포인트 기반 명시적 표현은 장면 편집, 객체 조합, 조명 교체 등에서 NeRF 기반 암시적 표현보다 유연한 일반화를 가능하게 한다.

### 3.2 일반화 성능 향상을 위한 핵심 방향

#### (1) 대규모/실외 장면으로의 확장
- 현재 파이프라인은 포인트 단위 PBR 연산으로 인해 대규모 장면에서 병목이 발생한다.
- **Deferred rendering** 기법 (논문에서 직접 언급)을 도입하면, PBR 연산을 이미지 공간(screen space)에서 수행하여 포인트 수에 대한 의존성을 줄일 수 있다.
- 최근 연구인 **GS-IR** [Liang et al., 2024]과 **3DGS-DR** [Ye et al., 2024]은 deferred rendering을 3DGS에 적용하여 이 문제를 부분적으로 해결하였다.

#### (2) 동적 장면으로의 확장
- 가시성 사전 계산이 정적 장면을 가정하므로, 동적 장면에서는 매 프레임 또는 키프레임마다 BVH를 재구축해야 한다.
- Deformable 3DGS [Yang et al., 2024] 등과 결합하여 시간 변화를 모델링할 수 있는 가능성이 있다.

#### (3) 복잡한 재질 모델로의 확장
- 현재 simplified Disney BRDF (albedo + roughness)만 사용하며, metallic, subsurface scattering, anisotropy 등은 모델링하지 않는다.
- 보다 일반적인 BRDF 모델 또는 학습 가능한 BRDF 표현을 도입하면 다양한 재질에 대한 일반화가 향상될 수 있다.

#### (4) 조명 모델의 고도화
- 3차 SH 간접광은 저주파 간접 조명만 포착 가능하다.
- **Path tracing** 기반 간접 조명 계산 [NeFII, Wu et al., 2023] 또는 **screen-space global illumination** 기법을 통합하면 일반화된 재조명이 가능하다.

#### (5) Few-shot / Sparse View 일반화
- 현재 방법은 dense multi-view 입력을 가정한다.
- 사전 학습된 모노큘러 깊이/법선 추정 모델 [MonoSDF, Yu et al., 2022]을 prior로 통합하면 sparse view에서의 일반화 가능성이 향상될 수 있다.

#### (6) 기하학적 일반화
- 논문에서 MVS (Multi-View Stereo) 통합이 더 정확한 표현을 위해 유망하다고 직접 언급하였다.
- 최근 **2DGS** [Huang et al., 2024]는 3D Gaussian을 2D surfel로 제한하여 더 정확한 표면 복원을 달성하며, 이러한 기하학적 개선이 BRDF 분해 품질에도 직접적으로 기여한다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구 영향

1. **포인트 기반 그래픽스 파이프라인의 패러다임 전환**: R3DG는 기존 메쉬 기반 그래픽스 파이프라인(모델링→렌더링→레이 트레이싱)을 포인트 기반으로 완전히 대체할 수 있는 가능성을 최초로 제시하였다. 이후 GaussianShader [Jiang et al., 2024], GS-IR [Liang et al., 2024], GaussianEditor 등 후속 연구가 이 프레임워크를 기반으로 발전하였다.

2. **3DGS의 응용 범위 확장**: 3DGS를 NVS 전용 도구에서 inverse rendering, relighting, material editing이 가능한 종합적 표현 방식으로 확장하였다.

3. **실시간 렌더링과 물리 기반 렌더링의 통합**: 포인트 기반 표현에서 BVH 레이 트레이싱과 PBR을 결합한 것은 실시간 응용(게임, VR/AR, 디지털 트윈)에 직접적 영향을 미친다.

4. **편집 가능한 재구성(Editable Reconstruction)**: 명시적 포인트 표현의 편집 용이성과 물리 기반 재조명의 결합은 콘텐츠 제작 파이프라인에 실질적 영향을 준다.

### 4.2 향후 연구 시 고려사항

| 고려사항 | 세부 내용 |
|----------|----------|
| **확장성(Scalability)** | 대규모 장면에서의 포인트 수 증가에 따른 연산 비용 관리 (deferred rendering, LOD 등) |
| **재질-조명 분해 정확도** | 더 강력한 물리적 사전(prior) 또는 학습 기반 분해 방법 필요 |
| **동적 장면** | 시간에 따른 BVH 재구축, 가시성 업데이트의 효율성 |
| **글로벌 일루미네이션** | 단순 SH 기반 간접광을 넘어, 다중 반사(multi-bounce) 간접 조명 모델링 |
| **기하학 품질** | 3DGS의 소프트 포인트 한계를 극복할 표면 정합 기법 필요 |
| **일반 BRDF** | 금속성, 반투명, 이방성 재질 등 다양한 재질 유형 지원 |
| **벤치마크 표준화** | 재조명 평가를 위한 표준화된 벤치마크 및 메트릭 필요 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 표현 | 핵심 특징 | R3DG와의 차이점 |
|------|------|------|----------|----------------|
| **NeRF** [Mildenhall et al.] | 2020 | Implicit (MLP) | 뷰 합성의 기초, 재조명 불가 | R3DG는 명시적 표현 + 재조명 |
| **PhySG** [Zhang et al.] | 2021 | Neural (SG) | Spherical Gaussians 기반 재조명 | R3DG가 NVS/relighting 모두 우세 |
| **NeRFactor** [Zhang et al.] | 2021 | Implicit | 형상/반사 분해, >48시간 학습 | R3DG가 0.9시간으로 53배 빠르고 성능도 우수 |
| **NeRV** [Srinivasan et al.] | 2021 | Implicit | 반사/가시성 필드 | R3DG는 명시적 가시성 계산 |
| **Nvdiffrec** [Munkberg et al.] | 2022 | Mesh (DMTet) | 메쉬 기반 역렌더링 | R3DG는 포인트 기반, NVS PSNR +2.17dB |
| **TensoIR** [Jin et al.] | 2023 | Neural (Tensor) | 텐서 분해 기반 역렌더링 | R3DG가 NVS/relighting PSNR 모두 우수, 3.6배 빠름 |
| **InvRender** [Zhang et al.] | 2022 | Neural | 간접 조명 모델링 | R3DG가 relighting PSNR +2.33dB, 15.9배 빠름 |
| **NeILF++** [Zhang et al.] | 2023 | Neural (VolSDF) | 상호 반사 광장 | R3DG가 NVS에서 PSNR +4.85dB |
| **3DGS** [Kerbl et al.] | 2023 | Point (Gaussian) | 실시간 NVS, 재조명 불가 | R3DG는 재조명 기능 추가 (NVS에서 -2.66dB 트레이드오프) |
| **GaussianShader** [Jiang et al.] | 2024 | Point (Gaussian) | 간소화된 셰이딩 함수 | R3DG는 완전한 PBR + 레이 트레이싱 |
| **GS-IR** [Liang et al.] | 2024 | Point (Gaussian) | Deferred shading + baking | R3DG의 per-point PBR과 대비되는 deferred 접근 |
| **2DGS** [Huang et al.] | 2024 | 2D Surfel | 2D Gaussian으로 표면 정확도 개선 | R3DG의 기하학적 한계를 보완하는 방향 |
| **Relightable 3DGS (ICL)** [Saito et al.] | 2024 | Point (Gaussian) | All-frequency relighting | R3DG의 확장으로 고주파 재조명 처리 |

### 핵심 비교 인사이트

- **속도-품질 트레이드오프**: R3DG는 NeRF 기반 역렌더링(TensoIR, InvRender) 대비 학습 시간을 대폭 단축하면서 동등 이상의 성능을 달성한다.
- **NVS vs. Relighting 트레이드오프**: vanilla 3DGS 대비 NVS 품질은 약간 하락하지만(-2.66dB), 재조명 기능이라는 근본적 이점을 확보한다.
- **명시적 vs. 암시적 표현**: 명시적 포인트 표현은 편집/조합에서 우월하지만, 연속적 표면 표현과 글로벌 조명 모델링에서는 암시적 표현에 비해 한계가 있다.
- **후속 연구 동향**: R3DG 이후 GS-IR, GaussianShader 등이 deferred rendering, screen-space 기법 등으로 확장성 문제를 해결하려는 시도가 활발히 진행 중이다.

---

## 참고자료

1. **Gao, J., Gu, C., Lin, Y., Li, Z., Zhu, H., Cao, X., Zhang, L., & Yao, Y.** (2024). "Relightable 3D Gaussians: Realistic Point Cloud Relighting with BRDF Decomposition and Ray Tracing." *arXiv:2311.16043v2* [cs.CV].
2. **Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G.** (2023). "3D Gaussian Splatting for Real-Time Radiance Field Rendering." *ACM Transactions on Graphics (TOG)*.
3. **Munkberg, J., Hasselgren, J., Shen, T., Gao, J., Chen, W., Evans, A., Müller, T., & Fidler, S.** (2022). "Extracting Triangular 3D Models, Materials, and Lighting from Images." *CVPR 2022*.
4. **Jin, H., Liu, I., Xu, P., Zhang, X., Han, S., Bi, S., Zhou, X., Xu, Z., & Su, H.** (2023). "TensoIR: Tensorial Inverse Rendering." *CVPR 2023*.
5. **Zhang, Y., Sun, J., He, X., Fu, H., Jia, R., & Zhou, X.** (2022). "Modeling Indirect Illumination for Inverse Rendering." *CVPR 2022* (InvRender).
6. **Zhang, J., Yao, Y., Li, S., Liu, J., Fang, T., McKinnon, D., Tsin, Y., & Quan, L.** (2023). "NeILF++: Inter-Reflectable Light Fields for Geometry and Material Estimation." *ICCV 2023*.
7. **Zhang, K., Luan, F., Wang, Q., Bala, K., & Snavely, N.** (2021). "PhySG: Inverse Rendering with Spherical Gaussians." *CVPR 2021*.
8. **Zhang, X., Srinivasan, P.P., Deng, B., Debevec, P., Freeman, W.T., & Barron, J.T.** (2021). "NeRFactor: Neural Factorization of Shape and Reflectance." *ACM TOG 2021*.
9. **Burley, B.** (2012). "Physically-Based Shading at Disney." *SIGGRAPH 2012*.
10. **Yao, Y., Zhang, J., Liu, J., Qu, Y., Fang, T., McKinnon, D., Tsin, Y., & Quan, L.** (2022). "NeILF: Neural Incident Light Field for Physically-Based Material Estimation." *ECCV 2022*.
11. **Huang, B., Yu, Z., Chen, A., Geiger, A., & Gao, S.** (2024). "2D Gaussian Splatting for Geometrically Accurate Radiance Fields." *SIGGRAPH 2024*.
12. **Liang, Z., Zhang, Q., Feng, Y., Shan, Y., & Jia, K.** (2024). "GS-IR: 3D Gaussian Splatting for Inverse Rendering." *CVPR 2024*.
13. **Jiang, Y., Tu, J., Liu, Y., Gao, X., Long, X., Wang, W., & Ma, Y.** (2024). "GaussianShader: 3D Gaussian Splatting with Shading Functions for Reflective Surfaces." *CVPR 2024*.
14. **Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., & Ng, R.** (2020). "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis." *ECCV 2020*.
15. **Karras, T.** (2012). "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees." *Proceedings of the Fourth ACM SIGGRAPH/Eurographics Conference on High-Performance Graphics*.
16. **Keselman, L. & Hebert, M.** (2023). "Flexible Techniques for Differentiable Rendering with 3D Gaussians." *arXiv preprint*.
17. **Wu, H., Hu, Z., Li, L., Zhang, Y., Fan, C., & Yu, X.** (2023). "NeFII: Inverse Rendering for Reflectance Decomposition with Near-Field Indirect Illumination." *CVPR 2023*.
18. **Kajiya, J.T.** (1986). "The Rendering Equation." *Proceedings of the 13th Annual Conference on Computer Graphics and Interactive Techniques*.
