# Deformable Beta Splatting

논문 출처: Liu, R., Sun, D., Chen, M., Wang, Y., & Feng, A. (2025). *Deformable Beta Splatting*. SIGGRAPH 2025 Conference Papers. arXiv:2501.18630v2. (USC / Institute for Creative Technologies)

---

## 1. 핵심 주장과 주요 기여 요약

DBS는 3D Gaussian Splatting(3DGS)의 두 가지 본질적 한계 — (i) 가우시안 커널의 매끄럽고 무한한 꼬리(unbounded tail)로 인한 평면·날카로운 경계 표현 한계, (ii) 저차 구면조화함수(Spherical Harmonics, SH)로 인한 specular highlight 표현 한계 — 를 해결하기 위해 **Beta 분포 기반의 변형 가능한(deformable) 커널**을 제안한 연구입니다.

핵심 기여 세 가지는 다음과 같습니다.

1. **Beta Kernel (기하 표현용)**: 유계(bounded) 지지구간을 가지며, 단일 파라미터 $b$로 평탄한 표면부터 고주파 디테일까지 적응적으로 표현 가능한 커널.
2. **Spherical Beta (색상 인코딩용)**: SH degree 3 대비 31% 파라미터로 diffuse·specular 분리 및 sharp specular highlight 표현. Phong reflection model에서 영감.
3. **Kernel-Agnostic MCMC**: 3DGS-MCMC(Kheradmand et al., NeurIPS 2024)를 일반화하여, opacity 정규화만으로 분포 보존 densification이 성립함을 수학적으로 증명. 임의의 splatting 커널에 적용 가능.

성과 요약: **3DGS-MCMC 대비 파라미터 45%, 렌더링 속도 1.5배**, Mip-NeRF360/Tanks&Temples/Deep Blending/NeRF Synthetic 4개 벤치마크에서 SoTA 달성.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

3DGS(Kerbl et al., 2023)는 명시적(explicit) 표현으로 실시간 렌더링을 가능하게 했으나, 다음 이유로 NeRF 계열(특히 Zip-NeRF, Barron et al., 2023) 대비 photorealism이 떨어집니다.

- **가우시안 커널의 고정성**: $f(x) = e^{-x^2/2}$는 무한 지지구간(long tail)을 갖기 때문에 평면 표면이나 sharp edge를 표현할 때 인위적인 cut-off가 필요하고, 이는 artifact를 유발합니다.
- **SH의 다항식 증가**: $N$차 SH의 파라미터 수는 $3(N+1)^2$로 급증하여, 실시간 렌더링을 위해 보통 $N=3$으로 제한되며, 그 결과 sharp specular reflection을 표현하지 못합니다.
- **Densification 휴리스틱 의존**: 기존의 클로닝/스플리팅 전략은 가우시안 분산 특성에 강하게 의존하여, 다른 형태의 커널에 일반화되지 않습니다.

### 2.2 제안 방법 (수식 포함)

#### (a) Beta Kernel 유도

Beta 분포는 다음과 같이 정의됩니다(Johnson et al., 1995):

$$
f(x; \alpha, \beta) = \frac{1}{B(\alpha, \beta)} x^{\alpha-1} (1-x)^{\beta-1}, \quad x \in [0, 1]
$$

저자들은 multi-view consistency를 보장하는 종 모양(bell shape)을 위해 $\alpha=1$로 고정하고 정규화 항을 제거합니다:

$$
\mathcal{B}(x; \beta) = (1 - x)^{\beta}, \quad x \in [0,1], \, \beta \in (0, \infty)
$$

여기서 핵심은 **inverse Abel transform을 사용해 multi-view consistency를 만족하는 splatting kernel임을 수학적으로 증명**한 점입니다(논문 Appendix A).

직접 $\beta$ 최적화는 저주파 편향을 일으키므로, exponential 활성화로 reparameterize하고 $b=0$일 때 가우시안과 거의 일치하도록 상수 4를 선택:

$$
\mathcal{B}(x; b) = (1-x)^{\beta(b)}, \quad \beta(b) = 4e^{b}, \quad b \in \mathbb{R}
$$

이때 $b<0$은 평탄한 표면+sharp cutoff, $b>0$은 sharp peak(고주파 디테일)을 표현합니다.

#### (b) 3D Ellipsoidal Beta Primitive

각 primitive의 파라미터 집합:

$$
B = \{\boldsymbol{\mu}, o, \boldsymbol{q}, \boldsymbol{s}, b, \boldsymbol{f}\}
$$

여기서 $\boldsymbol{\mu} \in \mathbb{R}^3$ (위치), $o \in [0,1]$ (불투명도), $\boldsymbol{q}$ (쿼터니언 회전), $\boldsymbol{s}$ (스케일), $b \in \mathbb{R}$ (Beta shape), $\boldsymbol{f} \in \mathbb{R}^d$ (color feature). 기존 3DGS와의 차이는 **$b$와 $\boldsymbol{f}$**.

픽셀까지의 Mahalanobis 거리:

$$
r_i(\boldsymbol{x}) = \sqrt{(\boldsymbol{x} - \boldsymbol{\mu}_i')^\top {\boldsymbol{\Sigma}_i'}^{-1} (\boldsymbol{x} - \boldsymbol{\mu}_i')}
$$

알파 합성:

$$
\boldsymbol{C}(\boldsymbol{x}) = \sum_{i=1}^{N} c_i \, o_i \, \mathcal{B}(r_i(\boldsymbol{x})^2; b_i) \prod_{j=1}^{i-1} \big(1 - o_j \, \mathcal{B}(r_j(\boldsymbol{x})^2; b_j)\big)
$$

#### (c) Spherical Beta (SB) — Color encoding

Phong reflection model 기반 — ambient/diffuse를 base color $c_0$로 합치고, specular lobe만 학습 가능한 Beta로 표현:

$$
c(\hat{V}) = c_0 + \sum_{m \in \mathcal{M}} \mathcal{B}(1 - \hat{R}_m \cdot \hat{V}; b_m) \, c_m
$$

피처 차원은 $3 + 6M$으로 reflection lobe 수 $M$에 선형. SH degree 3의 48차원 대비, $M=2$일 때 15차원으로 약 31%의 파라미터만 사용. **Spherical Gaussian과 달리 입력이 $[0,1]$ 범위로 유계여서 truncation artifact가 없음**이 강점입니다.

#### (d) Kernel-Agnostic MCMC

3DGS-MCMC의 densification은 가우시안 함수 형태에 의존(스케일 조정 포함)하기에 deformable kernel에 직접 적용 불가능. 저자들은 정규화된 작은 $o$에 대해 Taylor 전개를 사용:

$$
o' = 1 - \sqrt[N]{1 - o} \approx \frac{o}{N}
$$

그리고 binomial 근사로:

$$
1 - \left(1 - \frac{o}{N} f(x)\right)^N \approx 1 - \big(1 - o f(x)\big) + \mathcal{O}(o^2) = o f(x) + \mathcal{O}(o^2)
$$

즉 **opacity가 작게 정규화되어 있다면, 복제 횟수 $N$이나 커널 형태 $f(x)$에 무관하게 분포가 보존됨**을 증명. 이를 위해 손실 함수에 opacity regularizer 추가:

$$
\mathcal{L} = (1-\lambda_{\text{SSIM}})\mathcal{L}_1 + \lambda_{\text{SSIM}}\mathcal{L}_{\text{SSIM}} + \lambda_o \sum_i |o_i| + \lambda_\Sigma \sum_i \sum_j \sqrt{|\mathrm{eig}_j(\Sigma_i)|}
$$

또한 noise term을 logit이 아닌 Beta 함수로 재정의:

$$
\boldsymbol{\mu} \leftarrow \boldsymbol{\mu} - \lambda_{\text{lr}} \nabla_{\boldsymbol{\mu}}\mathcal{L} + \lambda_\epsilon \cdot \epsilon, \quad \epsilon = \lambda_{\text{lr}} \cdot \mathcal{B}(o_i; b') \cdot \Sigma_\eta
$$

여기서 $b' = \ln(25)$는 원래 logit fall-off를 모사.

### 2.3 모델 구조

DBS는 GSplat 라이브러리(Ye et al., 2024) 기반으로, 3DGS-MCMC 파이프라인을 다음과 같이 수정:

1. **Primitive 정의**: 가우시안 → 3D Ellipsoidal Beta Primitive ($b$ 추가)
2. **Color**: SH 계수 → Spherical Beta lobes ($M=2$ 기본)
3. **Rasterization**: CUDA로 Beta 함수와 SB 함수 구현 (fully explicit & differentiable)
4. **Optimization**: Kernel-Agnostic MCMC + flexible early stopping (patience 10k)

### 2.4 성능 향상

논문 Table 1, 2의 핵심 수치:

- **Mip-NeRF360**: PSNR 28.75 (DBS full) vs 28.54 (Zip-NeRF), 28.29 (3DGS-MCMC)
- **Tanks&Temples**: 24.85 vs 24.29 (3DGS-MCMC); LPIPS 0.140 vs 0.190
- **NeRF Synthetic**: 34.66 vs 33.80 (3DGS-MCMC), 33.10 (Zip-NeRF)
- **효율성**: DBS 356 MB / 123 FPS / 22분 학습 vs 3DGS-MCMC 733 MB / 82 FPS / 31분 (RTX 6000 Ada)
- **압축률**: 파라미터 45% / SB는 SH-3 대비 31%

Ablation에서는 가우시안 → Beta 커널 자체 변경(+0.16 dB), Kernel-Agnostic MCMC(+0.12 dB), SB lobe 수 $M=2$가 sweet spot임을 보여줍니다.

### 2.5 한계 (논문 명시)

1. **Popping artifact**: 래스터라이제이션 기반이므로 depth sorting 부정확성에 따른 popping 발생 가능.
2. **Mirror/anisotropic specular**: Spherical Beta가 거울 반사나 비등방성 specular 모델링에는 한계.
3. **원거리 배경 처리**: Frustum 모델 특성상 먼 배경에서 Beta kernel이 평탄 기하로 최적화되어 분포가 왜곡.

---

## 3. 모델의 일반화 성능 향상 가능성 (중점)

DBS의 **generalization-favoring 설계 요소**는 다음과 같습니다.

### (1) 커널 표현력의 일반화 (Representational Generalization)

Beta Kernel은 $b=0$에서 가우시안과 거의 동일하므로 **가우시안 커널의 strict superset**입니다. 따라서 학습 초기에는 안정적인 가우시안 동작을 보장하다가, scene 복잡도에 따라 자동으로 sharp edge / flat surface / fine detail로 변형됩니다. 이는 **다양한 장면 통계(scene statistics)에 대해 단일 framework로 적응**할 수 있다는 점에서 기존의 fixed-function 커널보다 일반화 측면에서 유리합니다. 실제로 NeRF Synthetic(랜덤 초기화), Mip-NeRF360(unbounded outdoor), Deep Blending(indoor) 등 성격이 매우 다른 데이터셋 모두에서 SoTA를 달성한 사실이 이를 뒷받침합니다.

### (2) 초기화 견고성 (Initialization Robustness)

NeRF Synthetic 실험에서 **랜덤 초기화**로도 PSNR 34.66을 달성. 이는 Kernel-Agnostic MCMC가 SfM point cloud 의존성을 완화함을 시사합니다. 3DGS 계열의 큰 약점이었던 "초기화 품질에 따른 성능 변동"을 줄이는 효과가 있어, **새로운 도메인(예: 의료영상 3D, 위성 영상, 수중 환경 등 SfM이 부정확한 영역)으로의 전이 가능성**이 있습니다.

### (3) 수학적으로 증명된 Kernel-Agnostic 성질

저자들이 증명한 분포 보존 densification 정리는 **Beta 커널뿐 아니라 임의의 splatting 커널 $f(x)$에 대해 성립**합니다. 즉, 미래에 더 표현력 있는 새로운 커널이 제안되더라도(예: Linear Kernel Splatting, 3D Convex Splatting 등) 동일한 최적화 파이프라인을 재사용할 수 있어, 방법론 자체가 long-term generalization을 갖습니다.

### (4) 색상 표현의 모듈화

Spherical Beta는 diffuse/specular를 명시적으로 분리하므로, **물리 기반 렌더링(BRDF estimation), relighting, material editing** 등으로 일반화 가능. 논문 Fig. 4, 6의 geometry/lighting decomposition 결과는 downstream task로의 전이 가능성을 보여줍니다.

### (5) 실측되지 않은 일반화 측면(주의)

다만 논문 자체는 **정적(static) 장면, 단일 도메인 학습** 평가만 수행했으며, 다음은 **검증되지 않았습니다**:
- Cross-domain 전이 (예: DTU → KITTI, indoor → outdoor zero-shot)
- Few-shot/sparse-view 일반화
- 동적 장면(temporal generalization)
- Generative prior 결합 시 일반화

이 부분은 후속 연구가 필요합니다(저자들의 후속 작업 *Universal Beta Splatting*이 N-차원 anisotropic Beta로 spatial/angular/temporal까지 통합 일반화를 시도 중인 것으로 알려져 있음. 출처: rongliu-leo.github.io/beta-splatting).

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 영향

1. **Splatting kernel design의 새로운 패러다임 제시**: 가우시안에 매여 있던 splatting 분야에 "kernel deformability"라는 축을 본격 도입. 이미 Deformable Radial Kernel(Huang et al., CVPR 2025), 3D Convex Splatting(Held et al., 2024) 등 동시기 연구들이 같은 흐름을 형성.
2. **MCMC framework의 일반화 이론**: opacity-only regularization으로 distribution preservation이 성립한다는 증명은 후속 splatting 최적화 연구의 기반 정리(foundational lemma)로 활용 가능.
3. **컴팩트 표현(compact representation)**: 메모리 45% / SH 31%만으로 SoTA를 달성하므로, **모바일/AR/VR/엣지 디바이스 실시간 렌더링** 응용을 가속.
4. **물리 기반 분해**: Spherical Beta의 diffuse/specular 분리는 inverse rendering, relighting, material capture 분야에 직접 응용 가능.

### 4.2 향후 연구 시 고려할 점

1. **Mirror reflection / anisotropic specular**: SB의 한계를 극복하기 위해 Spec-Gaussian(Yang et al., 2024a)이나 Ref-NeRF(Verbin et al., 2022)와의 hybrid 검토 필요.
2. **Popping artifact**: depth sorting의 근본적 부정확성을 해결하기 위해 ray-tracing 기반 splatting(EVER, GaussianTracer)과의 통합 또는 sort-free rendering 방향 탐색.
3. **동적 장면 / temporal generalization**: DBS는 정적 장면이며, 동적 확장 시 deformation field(예: Deformable 3D Gaussians, Yang et al., CVPR 2024)와의 결합이 필요.
4. **Few-view / sparse-view 시나리오**: 현재 평가는 dense view 가정. 데이터 효율성 검증 필요.
5. **하이퍼파라미터 민감도**: $\lambda_{\text{SSIM}}, \lambda_o, \lambda_\Sigma, b' = \ln(25)$ 등 상수의 도메인 의존성 분석 필요.
6. **이론적 한계 재검토**: opacity 근사 $o' \approx o/N$은 작은 $o$ 가정에 의존. 대용량 장면에서 $o$가 일정 수준 이상 누적될 때의 행동 분석.
7. **압축/스트리밍**: 3DGS.zip(Bagdasarian et al., 2025) 등 압축 기법과 결합 시 시너지 — 저자들도 Appendix E에서 언급.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연도 | 연구 | 핵심 아이디어 | DBS 대비 위치 |
|---|---|---|---|
| 2020 | NeRF (Mildenhall et al., ECCV 2020) | MLP 기반 implicit volumetric | DBS의 비교 origin; rendering quality target |
| 2021 | Mip-NeRF / Mip-NeRF 360 (Barron et al.) | Anti-aliasing, unbounded scene | DBS가 Mip-NeRF360 데이터셋에서 상회 |
| 2022 | Instant-NGP (Müller et al.) | Multi-resolution hash encoding | DBS가 PSNR/SSIM 모두 상회 |
| 2023 | **3DGS** (Kerbl et al., SIGGRAPH 2023) | Explicit Gaussian splatting, real-time | DBS의 직접 비교 baseline |
| 2023 | Zip-NeRF (Barron et al., ICCV 2023) | Anti-aliased grid-based NeRF | 이전까지 NeRF 계열 SoTA; DBS가 Mip-NeRF360에서 PSNR 상회 |
| 2024 | 2DGS (Huang et al., SIGGRAPH 2024) | 2D 평면 splatting, 정확한 표면 | 표면 정확도 강점; rendering quality는 DBS가 우위 |
| 2024 | GES (Hamdi et al., CVPR 2024) | Generalized Exponential 함수 | Sharp edge에 fewer primitives; DBS와 motivation 유사하나 deformability/MCMC 통합은 DBS만 |
| 2024 | Mip-Splatting (Yu et al., CVPR 2024) | Anti-aliasing for 3DGS | Aliasing 한정; DBS가 종합 품질 우위 |
| 2024 | Spec-Gaussian (Yang et al., 2024) | Anisotropic SG로 specular | DBS의 SB와 motivation 유사, 다만 SB는 bounded support로 truncation artifact 없음 |
| 2024 | **3DGS-MCMC** (Kheradmand et al., NeurIPS 2024 Spotlight) | SGLD 기반 densification | DBS가 직접 확장; Kernel-Agnostic 일반화 |
| 2024 | Revising Densification (Bulò et al., 2024) | densification 휴리스틱 개선 | DBS의 opacity 정규화 증명에 인용된 기반 |
| 2024 | Scaffold-GS (Lu et al., CVPR 2024), RadSplat (Niemeyer et al.) | 효율적 placement | DBS는 kernel 자체를 deform |
| 2024 | 3D Half Gaussian (Li et al.) | 반공간 가우시안 | 단순한 수정; DBS는 general parametric family |
| 2024 | 3D Convex Splatting (Held et al.) | Convex primitive | 동시기 대안; sharp edge에 강점이나 SH 의존 유지 |
| 2025 | Deformable Radial Kernel (DRK, Huang et al., CVPR 2025) | 학습 가능한 radial basis로 2DGS 확장 | 가장 직접적 동시기 경쟁; 2D 기반 vs DBS의 3D ellipsoidal |

핵심 차별점: 동시기 deformable kernel 연구들(GES, DRK, 3D Convex)이 **기하 표현**에만 집중한 반면, DBS는 (a) 기하(Beta), (b) 색상(SB), (c) 최적화(Kernel-Agnostic MCMC)를 **하나의 일관된 수학적 framework**로 통합한 점이 가장 두드러집니다.

---

## 참고 자료 (출처)

1. **본 논문**: Liu, R., Sun, D., Chen, M., Wang, Y., & Feng, A. (2025). *Deformable Beta Splatting*. arXiv:2501.18630v2 [cs.CV]. https://arxiv.org/abs/2501.18630 — 모든 수식과 실험 수치의 일차 출처
2. **프로젝트 페이지**: https://rongliu-leo.github.io/beta-splatting/ (interactive demo, 후속 *Universal Beta Splatting* 언급 확인)
3. **공식 GitHub**: https://github.com/RongLiu-Leo/beta-splatting (SIGGRAPH 2025 official implementation)
4. **SIGGRAPH 2025 ACM 출판**: Liu et al. (2025). *Deformable Beta Splatting*. SIGGRAPH Conference Papers '25, Article No. 101. DOI: 10.1145/3721238.3730716. https://dl.acm.org/doi/10.1145/3721238.3730716
5. **USC ICT 발표 자료**: https://ict.usc.edu/news/ict-paper-on-deformable-beta-splatting-accepted-at-siggraph-2025/ 및 https://ict.usc.edu/news/geospatial-research-lab-to-present-novel-rendering-technique-at-siggraph-2025/
6. **3DGS-MCMC**: Kheradmand, S. et al. (2024). *3D Gaussian Splatting as Markov Chain Monte Carlo*. NeurIPS 2024 Spotlight. arXiv:2404.09591. https://github.com/ubc-vision/3dgs-mcmc
7. **3DGS 원논문**: Kerbl, B. et al. (2023). *3D Gaussian Splatting for Real-Time Radiance Field Rendering*. SIGGRAPH 2023. arXiv:2308.04079
8. **3DGS Survey**: Bagdasarian, M. T. et al. (2025). *3DGS.zip: A survey on 3D Gaussian Splatting Compression Methods*. arXiv:2407.09510
9. **Deformable Radial Kernel Splatting**: Huang, Y.-H. et al. (2025). CVPR 2025. arXiv:2412.11752. https://github.com/VAST-AI-Research/Deformable-Radial-Kernel-Splatting
10. **3DGS Survey (general)**: arXiv:2401.03890 (https://arxiv.org/html/2401.03890v8) — 동적 장면 deformable 3DGS 비교 맥락 확인
11. **SIGGRAPH 2025 프로그램**: https://www.realtimerendering.com/kesen/sig2025.html — 채택 확인

---

**참고 - 답변 정확도에 관한 주의사항**: 위 분석에서 (1) 수식·실험 수치·아키텍처 세부사항·한계는 모두 본 논문(arXiv:2501.18630v2)에서 직접 확인한 내용이며, (2) 일반화 성능에 관한 논의 중 cross-domain 전이/few-shot 등은 **논문에서 직접 평가되지 않았다는 점을 명시**하였습니다. 후속 작업 *Universal Beta Splatting*은 프로젝트 페이지에 언급되어 있으나, 저자가 해당 별도 논문 본문을 직접 확인하지 않았으므로 세부 주장은 단정하지 않았습니다.
