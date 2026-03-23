# GaussianShader: 3D Gaussian Splatting with Shading Functions for Reflective Surfaces

---

## 1. 핵심 주장과 주요 기여 요약

**GaussianShader**는 3D Gaussian Splatting(3DGS) 프레임워크에 **간소화된 셰이딩 함수(simplified shading function)**를 결합하여, 반사 표면(reflective surfaces)을 포함한 장면에서도 **실시간 렌더링 속도를 유지하면서 고품질 뉴럴 렌더링**을 달성하는 방법을 제안한다.

### 주요 기여 (3가지)
1. **간소화된 셰이딩 함수**: 렌더링 방정식을 명시적으로 근사하는 셰이딩 모델을 3D Gaussian에 적용하여, 특히 고반사(specular) 표면에서의 사실감을 크게 향상시킴.
2. **새로운 노멀 추정 프레임워크**: 3D Gaussian의 최단축(shortest axis) 방향을 기반으로 한 노멀 표현과, 노멀-기하 일관성(normal-geometry consistency) 정규화 손실을 통해 정밀한 노멀 추정을 가능하게 함.
3. **실시간 렌더링 유지**: Gaussian Splatting의 효율성을 활용하여 97 FPS의 실시간 렌더링을 유지하면서, Ref-NeRF 대비 학습 시간을 약 40배 단축 (23h → 0.58h).

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 3D Gaussian Splatting [Kerbl et al., 2023]은 구형 조화 함수(SH)만으로 외형을 모델링하여 **빛-표면 상호작용(light-surface interaction)을 명시적으로 고려하지 않는다**. 이로 인해:
- **강한 반사(specular highlights)**를 포착하지 못하고,
- 시점에 따른 급격한 색상 변화를 표현하기 어려움.

반면 Ref-NeRF [Verbin et al., 2022]나 ENVIDR [Liang et al., 2023]은 반사 표면을 처리할 수 있지만:
- 최적화 시간이 매우 길고 (23h, 6h),
- 렌더링 속도가 느리며 (0.03 FPS, 1.33 FPS),
- ENVIDR는 SDF 기반이라 복잡한 장면에서 성능이 저하됨.

**핵심 과제**: 3D Gaussian Splatting의 효율성을 유지하면서, 반사 표면을 정확히 렌더링할 수 있는 셰이딩 함수를 어떻게 결합할 것인가?

### 2.2 제안하는 방법 (수식 포함)

#### (A) 간소화된 셰이딩 함수 (Sec. 3.1)

각 Gaussian sphere의 렌더링 색상 $\mathbf{c}$는 관찰 방향 $\omega_o$에 대해 다음과 같이 계산된다:

$$\mathbf{c}(\omega_o) = \gamma\left(\mathbf{c}_d + \mathbf{s} \odot L_s(\omega_o, \mathbf{n}, \rho) + \mathbf{c}_r(\omega_o)\right) \tag{3}$$

여기서:
- $\gamma$: 감마 톤 매핑 함수 (sRGB 변환)
- $\mathbf{c}_d \in [0,1]^3$: **확산(diffuse) 색상** — 시점에 무관한 일정한 색
- $\mathbf{s} \in [0,1]^3$: **스펙큘러 틴트(specular tint)** — 표면 고유 반사색
- $L_s(\omega_o, \mathbf{n}, \rho)$: **직접 스펙큘러 조명(direct specular light)**
- $\mathbf{n}$: Gaussian sphere의 법선 벡터
- $\rho \in [0,1]$: **거칠기(roughness)**
- $\mathbf{c}_r(\omega_o): \mathbb{R}^3 \to \mathbb{R}^3$: **잔차 색상(residual color)** — 3차 구형 조화 함수(SH)로 파라미터화
- $\odot$: 원소별 곱셈

**Ref-NeRF와의 차이점**: Ref-NeRF는 잔차 색상 항 $\mathbf{c}_r(\omega_o)$이 없어 간접 조명, 산란 등 복잡한 반사를 처리하기 어렵지만, GaussianShader는 이를 보상하여 더 다양한 반사 외형을 효율적으로 표현할 수 있다.

#### (B) 스펙큘러 조명 계산 (Sec. 3.2)

스펙큘러 조명 $L_s$는 GGX Normal Distribution Function [Walter et al., 2007]을 사용하여 입사 방사를 적분한다:

$$L_s(\omega_o, \mathbf{n}, \rho) = \int_{\Omega} L(\omega_i) D(\mathbf{r}, \rho)(\omega_i \cdot \mathbf{n}) \, d\omega_i \tag{4}$$

여기서:
- $\Omega$: 상반구(upper hemisphere)
- $\omega_i$: 입사 방향
- $D(\mathbf{r}, \rho)$: 스펙큘러 로브를 특성화하는 GGX NDF (거칠기 $\rho$가 작으면 로브가 좁고, 크면 넓음)
- $\mathbf{r} = 2(\omega_o \cdot \mathbf{n})\mathbf{n} - \omega_o$: 반사 방향

환경 조명 $L(\omega_i)$는 **학습 가능한 $6 \times 64 \times 64$ 큐브맵**으로 표현되며, 사전 필터링된 다중 mip map을 통해 효율적으로 적분값을 보간한다. 이는 Ref-NeRF의 MLP 기반 integrated directional encoding보다 학습 효율이 높다.

#### (C) 노멀 추정 (Sec. 3.3)

**최단축 방향(Shortest axis direction)**: 최적화 과정에서 3D Gaussian의 축 비율(aspect ratio)이 점진적으로 증가하여 평판 형태로 수렴하는 현상을 관찰. 이에 기반하여 최단축 방향 $\mathbf{v}$를 근사 법선으로 사용한다.

**예측 노멀 잔차(Predicted normal residual)**: 최단축의 방향 모호성(내/외 방향)을 처리하기 위해 두 개의 학습 가능한 잔차를 도입:

$$\mathbf{n} = \begin{cases} \mathbf{v} + \Delta\mathbf{n}_1 & \text{if } \omega_o \cdot \mathbf{v} > 0, \\ -(\mathbf{v} + \Delta\mathbf{n}_2) & \text{otherwise.} \end{cases} \tag{5}$$

노멀 잔차의 과도한 편차를 방지하기 위한 정규화:

$$\mathcal{L}_{\text{reg}} = \|\Delta\mathbf{n}\|^2 \tag{6}$$

**노멀-기하 일관성 손실(Normal-geometry consistency)**: 개별 Gaussian의 노멀이 주변 Gaussian이 형성하는 국소 기하와 일관되도록 강제한다:

$$\mathcal{L}_{\text{normal}} = \|\bar{\mathbf{n}} - \hat{\mathbf{n}}\|^2 \tag{7}$$

여기서 $\bar{\mathbf{n}}$은 렌더링된 노멀 맵, $\hat{\mathbf{n}}$은 렌더링된 깊이 맵에 Sobel 유사 연산자를 적용하여 얻은 깊이 기울기 법선이다. KNN 검색 없이도 다수 Gaussian의 국소 기하 정보를 간접적으로 활용하는 효율적 방법이다.

#### (D) 전체 손실 함수 (Sec. 3.4)

$$\mathcal{L} = \mathcal{L}_{\text{color}} + \lambda_n \mathcal{L}_{\text{normal}} + \lambda_s \mathcal{L}_{\text{sparse}} + \lambda_r \mathcal{L}_{\text{reg}} \tag{9}$$

여기서:

$$\mathcal{L}_{\text{color}} = \|\mathbf{C} - \mathbf{C}_{\text{gt}}\|^2 \tag{2}$$

$$\mathcal{L}_{\text{sparse}} = \frac{1}{|\alpha|} \sum_{\alpha_i} [\log(\alpha_i) + \log(1 - \alpha_i)] \tag{8}$$

하이퍼파라미터: $\lambda_n = 0.01$, $\lambda_s = 0.001$, $\lambda_r = 0.001$.

### 2.3 모델 구조

GaussianShader의 각 3D Gaussian sphere는 두 가지 범주의 속성을 갖는다:

| 범주 | 속성 |
|------|------|
| **형상 속성 (Shape Attributes)** | 공분산 $\Sigma = RSS^TR^T$ , 불투명도 $\alpha$, 위치 $\mathbf{p}$ |
| **셰이딩 속성 (Shading Attributes)** | 확산색 $\mathbf{c}\_d$ , 스펙큘러 틴트 $\mathbf{s}$, 거칠기 $\rho$, 노멀 잔차 $\Delta\mathbf{n}_{1,2}$, 잔차 색상 SH 계수 |

렌더링 파이프라인:
1. SfM으로 생성된 초기 점군으로 Gaussian 초기화
2. 각 Gaussian에서 셰이딩 속성 및 환경 조명 큐브맵 학습
3. 카메라 파라미터에 따라 2D로 투영 및 타일 기반 래스터화
4. $\alpha$-블렌딩으로 최종 픽셀 색상 합산:

$$\mathbf{C} = \sum_{i \in N} \mathbf{c}_i \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j) \tag{1}$$

### 2.4 성능 향상

| 메트릭 | 비교 대상 | 결과 |
|--------|---------|------|
| **PSNR** (Shiny Blender 평균) | 3DGS → Ours | 30.37 → 31.94 (+1.57 dB) |
| **PSNR** (Glossy Synthetic 평균) | 3DGS → Ours | 26.26 → 27.36 (+1.10 dB) |
| **PSNR** (NeRF Synthetic 평균) | 3DGS → Ours | 33.30 → 33.38 (유사) |
| **학습 시간** | Ref-NeRF vs. Ours | 23h → 0.58h (~40× 가속) |
| **FPS** | Ref-NeRF vs. Ours | 0.03 → 97 (~3200× 가속) |
| **평균 PSNR** (전체) | 기존 최고 대비 | **32.76** (Ours) vs. 32.05 (3DGS), 31.73 (Ref-NeRF) |

Ablation study 결과 (Shiny Blender, 1/2 해상도):

| 설정 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| w/o $\mathcal{L}_{\text{sparse}}$ | 31.79 | 0.952 | 0.056 |
| w/o $\mathcal{L}_{\text{normal}}$ | 30.93 | 0.941 | 0.060 |
| w/o $\mathbf{c}_r$ | 31.49 | 0.948 | 0.060 |
| w/o $\mathbf{v}$ (naive normal) | 31.47 | 0.951 | 0.058 |
| MLP Lighting | 29.73 | 0.936 | 0.075 |
| **Full Model** | **32.09** | **0.953** | **0.054** |

### 2.5 한계

논문에서 직접적으로 명시한 한계와 분석을 통해 도출 가능한 한계점:

1. **확산 표면에서의 제한적 향상**: Tanks and Temples (주로 diffuse 객체) 데이터셋에서 3DGS 대비 개선폭이 미미 (29.54 → 29.73 PSNR). 셰이딩 함수의 이점이 반사 표면에 집중됨.
2. **ENVIDR/Ref-NeRF 대비 반사 객체 품질**: Shiny Blender에서 ENVIDR (32.88 PSNR)이나 Ref-NeRF (32.32 PSNR)에 비해 GaussianShader (31.94 PSNR)가 약간 낮음. SDF 기반의 연속 표면이 더 매끄러운 노멀을 자연스럽게 생성하기 때문.
3. **간접 조명의 근사적 처리**: 잔차 색상 $\mathbf{c}_r$이 간접 조명을 보상하지만, 물리적으로 정확한 간접 조명 모델링은 아님.
4. **렌더링 속도 감소**: 3DGS의 274 FPS 대비 97 FPS로 약 2.8배 느려짐 (여전히 실시간이지만 추가 오버헤드 존재).
5. **단일 환경 조명 가정**: 큐브맵 기반 단일 환경 조명을 사용하므로, 공간적으로 변하는 조명이나 가까운 광원에 의한 국소 조명은 정확히 모델링하기 어려움.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재의 일반화 성능

GaussianShader는 다양한 데이터셋에서 일반화 능력을 평가하였다:

- **일반 객체 (NeRF Synthetic)**: 3DGS와 동등한 성능 유지 (33.38 vs. 33.30 PSNR) — 셰이딩 함수 추가가 일반 객체 렌더링을 해치지 않음을 입증.
- **반사 객체 (Shiny Blender, Glossy Synthetic)**: 3DGS 대비 유의미한 개선.
- **대규모 실외 장면 (Tanks and Temples)**: 소폭 개선.
- **리라이팅 (Relighting)**: 다양한 조명 환경에서의 리라이팅 결과가 사실적임을 보여 조명-재질 분리의 일반화 가능성을 시사.

### 3.2 일반화 성능 향상을 위한 잠재적 방향

1. **물리 기반 BRDF의 확장**: 현재 간소화된 셰이딩 모델을 Cook-Torrance 등 더 정교한 BRDF로 대체하면, 금속/비금속, 이방성(anisotropic) 반사 등 다양한 재질에 대한 일반화가 가능할 것임. 다만 효율성과의 트레이드오프를 고려해야 함.

2. **간접 조명의 명시적 모델링**: 현재 잔차 색상 $\mathbf{c}_r$로 근사하는 간접 조명을 명시적으로 모델링하면 (예: 다중 바운스 레이 트레이싱의 근사), 복잡한 실내 환경이나 상호 반사가 있는 장면에서의 일반화가 개선될 수 있음.

3. **공간 가변 조명(Spatially-varying illumination)**: 단일 환경맵 대신 공간적으로 변하는 조명 모델을 도입하면, 대규모 실외 장면이나 다중 광원 환경에 대한 일반화가 향상될 수 있음.

4. **노멀 추정의 멀티스케일 접근**: 현재 Sobel 연산자 기반의 깊이-노멀 일관성은 단일 스케일에서 동작하므로, 멀티스케일 노멀 일관성을 도입하면 다양한 기하학적 디테일 수준에서의 노멀 품질이 향상될 수 있음.

5. **사전 학습(Pretraining) 및 크로스 도메인 전이**: 대규모 다양한 재질 데이터셋으로 셰이딩 속성의 초기값을 사전 학습하면, 새로운 장면에 대한 수렴 속도와 일반화 성능이 모두 향상될 가능성이 있음.

6. **동적 장면으로의 확장**: 현재는 정적 장면에 한정되어 있으나, 시간 변화하는 조명과 움직이는 객체를 처리할 수 있도록 확장하면 실용적 일반화가 크게 향상될 것임.

---

## 4. 연구 영향 및 향후 연구 시 고려사항

### 4.1 연구 영향

1. **Gaussian Splatting 생태계의 확장**: GaussianShader는 3DGS 프레임워크에 물리 기반 렌더링 개념을 최초로 효과적으로 통합한 연구 중 하나로, 이후 GS 기반 inverse rendering, relighting, material editing 연구의 기초를 마련함.

2. **효율성-품질 트레이드오프의 새로운 기준점**: 실시간 렌더링을 유지하면서 반사 표면 처리를 가능하게 한 것은 AR/VR, 게임, 디지털 트윈 등 실시간 응용에 직접적인 영향.

3. **노멀 추정 방법론**: 이산적(discrete) 표현에서의 노멀 추정이라는 근본적 문제에 대한 실용적 해법(최단축 + 잔차 + 깊이 기울기 일관성)을 제시하여, 이후 포인트 기반 및 Gaussian 기반 렌더링 연구에 영향.

4. **연구 후속 논문들의 기반**: 이 논문 이후 3DGS + shading을 결합하는 다양한 후속 연구(GaussianShader의 개선, Relightable 3D Gaussians 등)가 등장.

### 4.2 향후 연구 시 고려할 점

1. **물리적 정확성 vs. 효율성**: 더 정교한 BRDF 모델이나 다중 바운스 렌더링을 도입하면 품질은 향상되지만 실시간성이 저하될 수 있으므로, 적절한 근사 수준을 설계해야 함.

2. **노멀 추정의 근본적 한계**: Gaussian sphere의 이산적 특성상, 날카로운 엣지나 불연속적인 기하 변화에서의 노멀 추정이 어려움. 이를 해결하기 위한 적응적 분할(adaptive splitting) 또는 하이브리드 표현 연구가 필요.

3. **평가 프로토콜의 표준화**: 반사 표면 렌더링의 품질을 PSNR/SSIM/LPIPS만으로 평가하는 것은 한계가 있으며, 반사 방향 정확도, 재질 분리 품질, 리라이팅 일관성 등 추가적인 평가 메트릭이 필요.

4. **실제 환경 데이터에서의 검증**: 논문의 주요 실험은 합성 데이터셋 중심이며, 실제 환경(in-the-wild)에서의 복잡한 조명과 다양한 재질에 대한 검증이 더 필요.

5. **메모리 효율성**: 각 Gaussian에 셰이딩 속성을 추가하면 메모리 사용량이 증가하므로, 대규모 장면에서의 확장성을 위한 메모리 최적화 전략(속성 양자화, 공유 등)이 고려되어야 함.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 표현 방식 | 반사 표면 처리 | 학습 시간 | 렌더링 FPS | 핵심 차이점 |
|------|------|---------|------------|---------|----------|---------|
| **NeRF** [Mildenhall et al.] | 2020/2021 | 암시적 (MLP) | SH로 제한적 | ~수시간 | <1 | 반사 표면 미처리, 느린 렌더링 |
| **Mip-NeRF** [Barron et al.] | 2021 | 암시적 (MLP) | 제한적 | ~수시간 | <1 | 안티앨리어싱 개선, 반사 미고려 |
| **Ref-NeRF** [Verbin et al.] | 2022 | 암시적 (MLP) | 반사 방향 파라미터화 | **23h** | **0.03** | 반사 방향 재파라미터화, 잔차 색상 없음 |
| **NVDiffRec** [Munkberg et al.] | 2022 | 메시 + SDF | BRDF 분리 | 수시간 | ~수 FPS | Inverse rendering, 메시 추출 |
| **NVDiffRecMC** [Hasselgren et al.] | 2022 | 메시 + SDF | Monte Carlo 렌더링 | 수시간 | ~수 FPS | MC 기반 더 정확한 렌더링 |
| **ENVIDR** [Liang et al.] | 2023 | SDF + 암시적 | 환경 조명 + SDF 기반 노멀 | **6h** | **1.33** | SDF의 연속성 활용, 복잡 장면에서 성능 저하 |
| **NeRO** [Liu et al.] | 2023 | SDF + 암시적 | BRDF + 간접 조명 | 수시간 | <1 | 간접 조명 모델링, 느린 속도 |
| **3D Gaussian Splatting** [Kerbl et al.] | 2023 | 명시적 (3D Gaussian) | SH만 사용 (미처리) | **0.25h** | **274** | 최고 속도, 반사 표면 실패 |
| **GaussianShader** (본 논문) | 2023 | 명시적 (3D Gaussian) + 셰이딩 | 간소화 셰이딩 함수 + 환경맵 | **0.58h** | **97** | 속도와 품질의 최적 균형 |

### 핵심 비교 분석

**vs. Ref-NeRF**: GaussianShader는 잔차 색상 $\mathbf{c}_r$을 도입하여 간접 조명 등 복잡한 반사를 보상하는 반면, Ref-NeRF는 직접 반사만 모델링. 그러나 Ref-NeRF는 연속 표현의 이점으로 일부 반사 장면에서 더 높은 PSNR을 달성. 속도 면에서 GaussianShader가 압도적 우위 (~3200× FPS 향상).

**vs. ENVIDR**: SDF의 연속 표면 특성으로 ENVIDR이 매끄러운 반사 표면에서 약간 높은 PSNR을 보이나, 복잡한 기하(그림자, 세밀한 구조)에서는 실패. GaussianShader는 일반 객체와 반사 객체 모두에서 균형 잡힌 성능 제공.

**vs. 3D Gaussian Splatting**: 동일 프레임워크 기반이므로 직접 비교 가능. 반사 객체에서 +1.57 dB (Shiny Blender), +1.1 dB (Glossy Synthetic) 개선. 일반 객체에서는 동등. 학습 시간은 2.3× 증가, FPS는 2.8× 감소하지만 여전히 실시간.

**이후 등장한 관련 연구들** (2023-2024, 논문 발표 이후):
- **Relightable 3D Gaussians** (R3DG, 2023): 3DGS에 point-based ray tracing을 결합하여 더 정확한 간접 조명 처리.
- **GS-IR** (2024): 3DGS 기반 inverse rendering으로 BRDF 분리 및 리라이팅 강화.
- **3DGS-DR** (2024): Deferred rendering 기반의 3DGS 셰이딩.

이들 후속 연구는 GaussianShader가 개척한 "3DGS + 물리 기반 셰이딩"이라는 연구 방향을 확장·개선한 것으로, GaussianShader의 학술적 영향을 입증한다.

---

## 참고자료

1. Jiang, Y., Tu, J., Liu, Y., Gao, X., Long, X., Wang, W., & Ma, Y. (2023). "GaussianShader: 3D Gaussian Splatting with Shading Functions for Reflective Surfaces." *arXiv:2311.17977v1*.
2. Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). "3D Gaussian Splatting for Real-Time Radiance Field Rendering." *ACM Transactions on Graphics (ToG)*, 42(4), 1–14.
3. Verbin, D., Hedman, P., Mildenhall, B., Zickler, T., Barron, J. T., & Srinivasan, P. P. (2022). "Ref-NeRF: Structured View-Dependent Appearance for Neural Radiance Fields." *CVPR 2022*.
4. Liang, R., Chen, H., Li, C., Chen, F., Panneer, S., & Vijaykumar, N. (2023). "ENVIDR: Implicit Differentiable Renderer with Neural Environment Lighting." *arXiv:2303.13022*.
5. Liu, Y., Wang, P., Lin, C., Long, X., Wang, J., Liu, L., Komura, T., & Wang, W. (2023). "NeRO: Neural Geometry and BRDF Reconstruction of Reflective Objects from Multiview Images." *arXiv:2305.17398*.
6. Munkberg, J., Hasselgren, J., Shen, T., et al. (2022). "Extracting Triangular 3D Models, Materials, and Lighting from Images." *CVPR 2022*.
7. Mildenhall, B., Srinivasan, P. P., Tancik, M., et al. (2021). "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis." *Communications of the ACM*, 65(1), 99–106.
8. Walter, B., Marschner, S. R., Li, H., & Torrance, K. E. (2007). "Microfacet Models for Refraction through Rough Surfaces." *Eurographics Symposium on Rendering*.
9. Barron, J. T., Mildenhall, B., Tancik, M., Hedman, P., Martin-Brualla, R., & Srinivasan, P. P. (2021). "Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields." *ICCV 2021*.
