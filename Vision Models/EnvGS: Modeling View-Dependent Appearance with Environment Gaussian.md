
# EnvGS: Modeling View-Dependent Appearance with Environment Gaussian

> **논문 정보**
> - **제목**: EnvGS: Modeling View-Dependent Appearance with Environment Gaussian
> - **저자**: Tao Xie, Xi Chen, Zhen Xu, Yiman Xie, Yudong Jin, Yujun Shen, Sida Peng, Hujun Bao, Xiaowei Zhou (Zhejiang University 등)
> - **학회**: CVPR 2025, pp. 5742–5751
> - **arXiv**: [arXiv:2412.15215](https://arxiv.org/abs/2412.15215)
> - **코드**: [https://zju3dv.github.io/envgs](https://zju3dv.github.io/envgs) / [GitHub](https://github.com/zju3dv/EnvGS)

---

## 1. 핵심 주장 및 주요 기여 요약

2D 이미지로부터 실세계 장면의 복잡한 반사를 복원하는 것은 사실적인 Novel View Synthesis(NVS)를 위해 필수적이다. 그러나 기존의 환경 맵(environment map) 기반 방법들은 고주파수 반사 디테일 재현에 어려움을 겪으며, 근거리(near-field) 반사를 제대로 처리하지 못한다.

이러한 한계를 극복하기 위해 EnvGS가 제안하는 핵심 주장은 다음 세 가지이다:

**① 환경 Gaussian 표현**: Gaussian primitive 집합을 명시적(explicit) 3D 표현으로 활용하여 환경의 반사를 캡처하며, 이 환경 Gaussian primitives를 기반 Gaussian(base Gaussian)과 통합하여 장면 전체의 외관을 모델링한다. **② 레이 트레이싱 기반 렌더러**: GPU의 RT core를 활용하는 ray-tracing 기반 렌더러를 개발하여 빠른 렌더링을 구현했다. **③ 실시간 고품질 렌더링**: 이를 통해 고품질 복원과 실시간 렌더링 속도를 동시에 달성한다.

**주요 기여 정리**

| 기여 항목 | 내용 |
|---|---|
| Environment Gaussian 표현 | 반사를 위한 새로운 명시적 3D Gaussian 표현 |
| Ray-tracing 기반 렌더러 | GPU RT core 활용 차분 가능한 렌더러 |
| 근거리 반사 모델링 | Near-field 반사 정확 처리 |
| 공동 최적화 | Base Gaussian + Environment Gaussian 동시 최적화 |
| 실시간 성능 | 고품질 반사 + 실시간 렌더링 속도 달성 |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

GaussianShader나 3DGS-DR과 같은 기존 연구들은 환경 맵과 셰이딩 함수를 3DGS에 통합하여 반사 모델링 능력을 향상시키려 했다. 그러나 이 방법들은 두 가지 이유로 복잡한 정반사(specular reflection)를 정확히 재현하는 데 어려움을 겪는다. 첫째, 환경 맵의 원거리 조명(distant lighting) 가정으로 인해 근거리 반사를 합성하기 어렵다. 둘째, 이 표현은 본질적으로 고주파수 반사 디테일을 포착하기에 표현력이 부족하다.

또한 2DGS는 구면 조화 함수(Spherical Harmonics, SH)의 제한된 표현 능력에 의존하여 정반사와 같은 강한 시점 의존적 효과를 모델링하지 못하고, 결과적으로 품질 저하 및 "foggy" geometry 문제가 발생한다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### (A) 전체 렌더링 파이프라인

렌더링 프로세스는 Base Gaussian을 래스터라이즈(rasterize)하여 픽셀별 법선(normal), 기본 색상(base color), 혼합 가중치(blending weight)를 획득하는 것으로 시작된다. 다음으로, 반사 방향(reflection direction)으로 Environment Gaussian을 ray-tracing 기반 렌더러로 렌더링하여 반사 색상을 캡처한다. 마지막으로, 반사 색상과 기본 색상을 결합하여 최종 출력을 생성하며, 단안 법선(monocular normal)과 실제 이미지를 감독으로 사용하여 Environment Gaussian과 Base Gaussian을 공동 최적화한다.

최종 출력 색상 $\mathbf{C}$는 기반 색상 $\mathbf{c}\_\text{base}$와 반사 색상 $\mathbf{c}_\text{ref}$의 혼합으로 표현된다:

$$\mathbf{C} = (1 - w) \cdot \mathbf{c}_\text{base} + w \cdot \mathbf{c}_\text{ref}$$

여기서 $w$는 Base Gaussian에서 예측되는 픽셀별 블렌딩 가중치이다.

#### (B) 2D Gaussian의 기하학적 표현 (Base Gaussian)

논문에서 활용하는 **2D Gaussian Splatting (2DGS)** 기반의 2D Gaussian primitive $k$는 다음의 변환 행렬 $\mathbf{H}$로 정의된다 (논문 수식):

```math
\mathbf{H} = \begin{bmatrix} s_u \mathbf{t}_u & s_v \mathbf{t}_v & 0 & \mathbf{p}_k \\ 0 & 0 & 0 & 1 \end{bmatrix}
```

여기서 $s_u, s_v$는 스케일 파라미터, $\mathbf{t}_u, \mathbf{t}_v$는 탄젠트 벡터, $\mathbf{p}_k$는 Gaussian 중심 위치이다.

3D Gaussian에 비해 2D Gaussian은 표면 표현으로서 뚜렷한 장점이 있다. 첫째, 2DGS가 채택하는 ray-splat 교차 방법은 다중 시점 깊이 불일치를 방지한다. 둘째, 2D Gaussian은 본질적으로 잘 정의된 법선을 제공하며, 이는 고품질 표면 복원 및 정확한 반사 계산에 필수적이다.

#### (C) Volume Rendering (논문 수식)

볼륨 렌더링 방정식은 다음과 같이 정의된다: $\sum_{i=1}^{N} T_i \alpha_i \mathbf{c}\_i$, where $\alpha\_i = \sigma\_i \mathcal{G}\_i$, $T_i = \prod_{j=1}^{i-1}(1-\alpha_j)$, 여기서 $\mathcal{G}(\cdot)$는 표준 2D Gaussian 값 평가 함수이다.

LaTeX 형식으로 정리하면:

$$\mathbf{C} = \sum_{i=1}^{N} T_i \alpha_i \mathbf{c}_i, \quad \alpha_i = \sigma_i \mathcal{G}_i, \quad T_i = \prod_{j=1}^{i-1}(1-\alpha_j)$$

#### (D) 반사 방향 계산

반사 방향 $\mathbf{r}$은 뷰 방향 $\mathbf{d}$와 표면 법선 $\mathbf{n}$으로부터 다음과 같이 계산된다:

$$\mathbf{r} = 2(\mathbf{d} \cdot \mathbf{n})\mathbf{n} - \mathbf{d}$$

Environment Gaussian은 표면점에서 표면 법선 주위로 시점 방향의 반사 방향으로 렌더링되어 반사 색상을 캡처한다.

#### (E) Densification 전략

학습 중에는 3DGS의 적응적 Gaussian 제어 전략과 3DGS-DR에서 도입된 normal propagation 및 color sabotage를 적용한다. Gaussian tracer가 3D 공간에서 Gaussian 속성을 직접 통합하기 때문에 densification 기준으로 사용되는 투영된 2D 중심에 대한 유효한 그래디언트가 없다. 이를 위해 3D 공간 그래디언트를 누적하여 유사한 효과를 달성하며, 누적된 각 그래디언트는 원거리 영역의 under-densification을 방지하기 위해 교차 깊이의 절반으로 스케일링된다.

#### (F) 손실 함수

최종 학습 손실은 다음과 같이 구성된다:

$$\mathcal{L} = \mathcal{L}_{1} + \lambda_\text{ssim} \mathcal{L}_\text{SSIM} + \lambda_\text{lpips} \mathcal{L}_\text{LPIPS} + \lambda_\text{normal} \mathcal{L}_\text{normal}$$

이 방법은 단안 법선 제약(monocular normal constraint)과 지각 손실(perceptual loss)을 포함하는 포괄적인 최적화 과정을 통합하여 학습 안정성을 높이고 기하 및 반사 복원 품질을 크게 향상시킨다.

---

### 2-3. 모델 구조

```
입력 이미지 (다중 시점)
       │
       ▼
[Base Gaussian (2DGS 기반)]
 - 위치(p), 스케일(s), 회전(r), 불투명도(σ)
 - Spherical Harmonics (기반 외관)
 - 예측: per-pixel 법선, 기반 색상, 블렌딩 가중치(w)
       │
       ├──── 법선(n) + 시점(d) → 반사 방향 r
       │
       ▼
[Environment Gaussian]
 - 명시적 3D Gaussian primitives (near/far 반사 캡처)
 - GPU RT Core 기반 ray-tracing 렌더러
 - 반사 색상(c_ref) 예측
       │
       ▼
[최종 색상 혼합]
 C = (1-w)·c_base + w·c_ref
       │
       ▼
[공동 최적화]
 - Ground Truth 이미지 감독
 - 단안 법선 감독 (monocular normal loss)
 - Perceptual loss (LPIPS)
```

Base Gaussian과 Environment Gaussian의 공동 최적화를 반사 렌더링 단계에서 분리하면 정확한 기하를 복원하지 못하고 열등한 반사 복원 및 렌더링 품질로 이어진다.

---

### 2-4. 성능 향상

EnvGS는 복잡한 정반사를 가진 실제 장면에서 정량 지표(PSNR, SSIM, LPIPS)를 기준으로 3DGS, 2DGS, GaussianShader, 3DGS-DR과 같은 기존 실시간 명시적 방법들을 크게 능가하며, 실시간 기법 중 가장 높은 렌더링 품질을 달성한다. Ref-Real 및 NeRF-Casting 데이터셋에서 일관적으로 높은 PSNR, SSIM과 낮은 LPIPS 값을 보이며 특히 고주파수 반사 디테일과 근거리 반사 캡처에서 우수하다.

비실시간 암시적 방법인 NeRF-Casting과 비교할 때에도 경쟁력 있는 렌더링 품질을 달성하면서 약 100배 빠른 속도를 보인다. 이는 EnvGS가 고품질 암시적 방법과 실시간 명시적 방법 사이의 간극을 메우는 역할을 함을 보여준다.

실시간 성능 측면에서 복잡한 장면에서 평균 26.221 FPS의 렌더링 속도를 기록하여 인터랙티브 애플리케이션에 적합하다.

---

### 2-5. 한계점

반투명하면서도 반사성인 표면(예: 창문)은 이 방법에 도전적인 케이스이며, 이러한 복잡한 투명 반사 표면을 처리하는 능력이 제한된다.

EnvGS는 GS 특화(GS-specific) 렌더링 파이프라인에 한정되어 있다는 한계가 존재한다.

추가적으로 논문 자체에서 확인되는 한계:
- **장면별 최적화 (Per-scene optimization)**: 각 장면마다 별도의 최적화가 필요하여 새로운 장면에 대한 즉각적인 일반화 불가
- **GPU 의존성**: 모든 실험은 단일 NVIDIA RTX 4090 GPU에서 수행되며, RT Core를 지원하는 고사양 GPU가 필수적
- **동적 장면 미지원**: 현재 정적 장면에만 적용 가능

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 현재 일반화의 강점

EnvGS는 보다 일반적이며, 3DGS-DR에서 최적화 실패를 방지하기 위해 필수적인 전경 객체의 수동 추정 바운딩 박스에 의존하지 않는다.

실세계 데이터에 대한 견고성을 위해 설계되었음에도 불구하고, 원거리 반사를 정확하게 재현하며 환경 맵 조명 시나리오를 위해 특별히 설계된 GaussianShader 및 3DGS-DR과 동등하거나 이를 능가하는 성능을 발휘한다. 또한 근거리 반사 캡처에서는 이들 방법을 크게 앞선다.

이 방법은 근거리 및 원거리 광원으로부터의 반사를 모두 통합된 Gaussian primitive 집합으로 모델링하여, 거리에 상관없이 모든 반사를 효과적으로 표현한다.

EnvGS는 복잡한 시점 의존적 효과를 특징으로 하는 실제 장면에 초점을 맞춘 다양한 데이터셋에서 학습 및 평가되었다.

### 3-2. 일반화 성능 향상을 위한 미래 방향

현재 EnvGS는 **장면별(per-scene) 최적화 방식**으로, 진정한 의미의 Cross-scene 일반화는 아직 미해결 과제이다. 다음의 확장 가능성이 존재한다:

| 확장 방향 | 가능성 | 근거 |
|---|---|---|
| **Generalizable 3DGS와 결합** | 높음 | pixelSplat, MVSplat 등과 결합 가능 |
| **동적 장면 확장** | 중간 | Environment Gaussian의 시간적 변화 모델링 |
| **Diffusion Prior 활용** | 높음 | StableNormal 이미 사용 중 |
| **피지컬 기반 분해 (BRDF)** | 높음 | 조명 변경 및 재조명(relighting) 지원 가능 |

단안 법선 손실(monocular normal loss)은 근거리 상호 반사를 모델링하는 데 필수적이며, 이는 사전학습된 단안 추정 모델(StableNormal 등)을 활용한 방향으로 확장 가능성을 시사한다.

---

## 4. 최신 관련 연구 비교 분석 (2020년 이후)

### 4-1. 주요 관련 연구 계보

```
NeRF (2020) → Ref-NeRF (2022) → NeRF-Casting (2024)
                    │
3DGS (2023) ────────┤
                    │
2DGS (2024) ────────┤
                    │
GaussianShader ─────┤──→ EnvGS (CVPR 2025)
3DGS-DR ────────────┘
```

### 4-2. 비교 표

| 방법 | 연도/학회 | 표현 방식 | 실시간 | 근거리 반사 | 고주파 반사 | 비고 |
|---|---|---|---|---|---|---|
| **NeRF** | 2020 ECCV | Implicit MLP | ❌ | △ | △ | 기초 연구 |
| **Ref-NeRF** | 2022 CVPR | NeRF + 반사 방향 인코딩 | ❌ | △ | △ | 원거리 조명 가정 |
| **3DGS** | 2023 SIGGRAPH | 3D Gaussian + SH | ✅ | ❌ | ❌ | 반사 취약 |
| **2DGS** | 2024 SIGGRAPH | 2D Gaussian | ✅ | ❌ | ❌ | 법선 정확, 반사 약함 |
| **GaussianShader** | 2024 CVPR | 3DGS + 환경맵 | ✅ | ❌ | △ | 원거리 반사만 |
| **3DGS-DR** | 2024 SIGGRAPH | 3DGS + deferred shading | ✅ | ❌ | △ | BBox 필요 |
| **NeRF-Casting** | 2024 arXiv | NeRF + ray casting | ❌ | ✅ | ✅ | 비실시간, 고품질 |
| **EnvGS** | **2025 CVPR** | **2DGS + Env Gaussian** | **✅** | **✅** | **✅** | **제안 방법** |

GaussianShader와 3DGS-DR은 환경 맵과 셰이딩 함수를 통합하여 3DGS의 반사 모델링 능력을 향상시켰으나, 여전히 두 가지 요인으로 복잡한 정반사를 정확히 재현하는 데 어려움을 겪는다.

Ref-NeRF는 반사 시점 방향을 이용해 나가는 방사를 인코딩하여 원거리 조명 조건에서 향상된 결과를 보인다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려사항

### 5-1. 앞으로의 연구에 미치는 영향

**① 명시적 3D 반사 표현의 새로운 패러다임**

EnvGS는 Gaussian primitives를 이용하여 고주파수 반사 디테일을 캡처하며, 명시적 3D 반사 표현이 원거리 조명 가정을 불필요하게 만들어 근거리 반사의 정확한 모델링을 가능하게 한다는 것을 증명했다. 이는 환경 맵 기반 패러다임에서 **명시적 공간 Gaussian 표현**으로의 패러다임 전환을 이끌 수 있다.

**② 레이 트레이싱 + Gaussian Splatting 융합 연구 촉진**

다수의 실제 및 합성 데이터셋 결과는 실시간 Novel View Synthesis에서 최고의 렌더링 품질을 달성함을 보여주며, 이는 GPU RT Core를 활용한 Gaussian 기반 레이 트레이싱 연구의 활성화를 자극할 것이다.

**③ 동적 장면 반사 연구로의 확장**

EnvGS를 확장하여 동적 환경 Gaussian 표현 $G_\text{env}$를 도입하려는 후속 연구들이 이미 등장하고 있으며, 이는 EnvGS가 동적 반사성 장면 연구의 기반이 됨을 의미한다.

**④ 관련 후속 연구들**

EnvGS 이후, 2D Gaussian이 deferred shading 중 다중 시점 일관성 있는 재질 맵을 생성하도록 강제하고 2DGS를 통한 레이 트레이싱으로 환경 모델링 전략을 도입하는 후속 연구들이 등장하고 있다.

### 5-2. 앞으로 연구 시 고려사항

**① 일반화 성능 (Generalizability)**

현재 EnvGS는 장면별 최적화(per-scene optimization)를 수행하므로, **feed-forward 일반화 가능한 버전**이 중요한 연구 과제이다. 특히 pixelSplat, MVSplat처럼 단일 포워드 패스로 다양한 장면에 적용 가능한 generalizable EnvGS 연구가 필요하다.

**② 반투명 표면 처리**

반투명하면서도 반사성을 가진 표면(예: 창문)은 이 방법에 도전적인 케이스이다. 이를 해결하기 위한 **투명도-반사율 분리 모델링** 연구가 중요한 방향이다.

**③ BRDF 분해와의 통합**

현재 EnvGS는 물리 기반 재질(BRDF) 분해를 명시적으로 수행하지 않는다. 재조명(relighting) 및 재질 편집을 위해서는 **Diffuse/Specular 분리 + BRDF 파라미터 추정**과의 통합이 필요하다.

**④ 동적 반사 장면**

정적 장면에만 적용되는 현재 방법을 **시간적으로 일관된 동적 반사 장면**으로 확장하는 연구, 특히 자율주행이나 VR/AR 응용에서의 활용이 유망하다.

**⑤ 메모리 및 확장성**

Novel View Synthesis는 VR/AR, 자율주행 등 다양한 응용을 가능하게 하며, 대규모 실외 장면(예: 도시 스케일)에서도 Environment Gaussian을 효율적으로 관리하는 **계층적 또는 스트리밍 기반 접근법**이 필요하다.

**⑥ 데이터셋 다양성**

Shiny Blender 데이터셋처럼 원거리 조명 가정 하에 렌더링된 환경에서도 실험이 수행되었으나, 더욱 다양한 조명 조건, 재질 유형, 실내/실외 복합 시나리오에 대한 벤치마크 구축이 중요하다.

---

## 참고 자료 및 출처

1. **[주 논문]** Tao Xie et al., "EnvGS: Modeling View-Dependent Appearance with Environment Gaussian," *CVPR 2025*, pp. 5742–5751. — [arXiv:2412.15215](https://arxiv.org/abs/2412.15215)
2. **[CVPR 2025 Open Access]** [https://openaccess.thecvf.com/content/CVPR2025/papers/Xie_EnvGS_...pdf](https://openaccess.thecvf.com/content/CVPR2025/papers/Xie_EnvGS_Modeling_View-Dependent_Appearance_with_Environment_Gaussian_CVPR_2025_paper.pdf)
3. **[공식 프로젝트 페이지]** [https://zju3dv.github.io/envgs/](https://zju3dv.github.io/envgs/)
4. **[GitHub 코드]** [https://github.com/zju3dv/EnvGS](https://github.com/zju3dv/EnvGS)
5. **[IEEE Xplore]** [https://ieeexplore.ieee.org/document/11093185](https://ieeexplore.ieee.org/iel8/11091818/11091608/11093185.pdf)
6. **[Semantic Scholar]** [https://www.semanticscholar.org/paper/f3c39ed9ba33b939901cc0f508ac411e44e87603](https://www.semanticscholar.org/paper/f3c39ed9ba33b939901cc0f508ac411e44e87603)
7. **[ResearchGate]** [https://www.researchgate.net/publication/387264434](https://www.researchgate.net/publication/387264434_EnvGS_Modeling_View-Dependent_Appearance_with_Environment_Gaussian)
8. **[Quick Review - Liner]** [https://liner.com/review/envgs-modeling-viewdependent-appearance-with-environment-gaussian](https://liner.com/review/envgs-modeling-viewdependent-appearance-with-environment-gaussian)
9. **[관련 논문] GaussianShader** — Jiang et al., CVPR 2024. [https://openaccess.thecvf.com/content/CVPR2024/html/Jiang_GaussianShader](https://openaccess.thecvf.com/content/CVPR2024/html/Jiang_GaussianShader_3D_Gaussian_Splatting_with_Shading_Functions_for_Reflective_Surfaces_CVPR_2024_paper.html)
10. **[관련 논문] 3DGS-DR (Deferred Reflection)** — [arXiv:2404.18454](https://arxiv.org/html/2404.18454v2)
11. **[CVPR 2025 Poster]** [https://cvpr.thecvf.com/virtual/2025/poster/33066](https://cvpr.thecvf.com/virtual/2025/poster/33066)
