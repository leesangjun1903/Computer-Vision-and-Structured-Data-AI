
# Progressive Radiance Distillation for Inverse Rendering with Gaussian Splatting

> **논문 정보**
> - **저자**: Keyang Ye, Qiming Hou, Kun Zhou
> - **발표**: arXiv:2408.07595, August 2024
> - **DOI**: https://doi.org/10.48550/arXiv.2408.07595

---

## 1. 핵심 주장 및 주요 기여 요약

이 논문은 **Progressive Radiance Distillation**이라는 역 렌더링(inverse rendering) 방법을 제안하며, **Distillation Progress Map**을 활용하여 물리 기반 렌더링(PBR)과 Gaussian 기반 Radiance Field 렌더링을 결합합니다.

저자 Keyang Ye, Qiming Hou, Kun Zhou는 다시점 이미지로부터 기하(geometry), 조명(light), 재질(material) 특성을 분리 복원하는 기술에 집중하며, 물리 기반 렌더링 모델과 Gaussian 기반 Radiance Field 렌더링을 점진적 Distillation 과정으로 통합합니다.

### 주요 기여 (Key Contributions)

| # | 기여 항목 | 설명 |
|---|----------|------|
| ① | Progressive Distillation 메커니즘 | 사전 학습된 Radiance Field에서 출발하여 PBR 파라미터를 점진적으로 추출 |
| ② | 하이브리드 렌더링 모델 | Radiance Field + PBR을 선형 보간(linear interpolation)으로 결합 |
| ③ | 국소 최솟값(local minima) 회피 | 초기 학습 불안정성 문제 해결 |
| ④ | 일반화 가능성 | Gaussian Splatting에 국한되지 않고 메시 기반 방법으로도 확장 가능 |

최신 방법들과 비교하여, 본 논문의 하이브리드 렌더링 모델과 progressive distillation 메커니즘은 Novel View Synthesis(NVS)와 리라이팅(relighting) 양쪽 모두에서 우수한 결과를 보여 역 렌더링에 가치 있는 기여를 합니다.

---

## 2. 해결하고자 하는 문제, 제안 방법(수식 포함), 모델 구조, 성능 향상 및 한계

### 2-1. 해결하고자 하는 문제

**역 렌더링(Inverse Rendering)의 근본적 난제:**

미지의 조명 조건 아래에서 조명·재질·기하가 복잡하게 상호작용하기 때문에 문제가 본질적으로 under-constrained(비결정적)합니다.

NeRF는 신규 시점 합성에 뛰어난 성능을 보였으나, MLP 기반 방법들은 표현력 한계와 높은 연산 비용으로 품질과 효율의 균형을 맞추기 어렵습니다.

기존 일부 연구들은 emissive radiance 항을 물리 모델과 결합하여 간접 조명을 근사하지만, 이들을 직접 합산하여 처음부터 공동 최적화(jointly optimize from scratch)하면 모호성(ambiguity)이 증가할 수 있습니다.

또한 구면 조화 함수(spherical harmonics)는 정반사(specular reflection)를 정확히 표현하기 위한 방향 해상도가 부족하며, Gaussian splatting 및 복제(cloning) 과정에서 부유 아티팩트(floating artifacts)가 발생할 수 있습니다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### ① 하이브리드 렌더링 모델

최종 렌더링 모델에서 $I(x, \boldsymbol{\omega}\_o)$는 공간 위치에 대해 매개변수화된 최종 Radiance이며, $I_\text{phy}(x, \boldsymbol{\omega}\_o)$는 물리 기반 항(physically-based term), $I_\text{raw}(x, \boldsymbol{\omega}_o)$는 distillation 과정에서 미세 조정되는 Radiance Field 항입니다.

이를 수식으로 표현하면:

$$I(x, \boldsymbol{\omega}_o) = \alpha(x) \cdot I_\text{phy}(x, \boldsymbol{\omega}_o) + (1 - \alpha(x)) \cdot I_\text{raw}(x, \boldsymbol{\omega}_o)$$

여기서 $\alpha(x) \in [0, 1]$은 학습 가능한 **Distillation Progress Map**입니다.

#### ② Distillation Progress Map 초기화 및 진행

Distillation Progress Map은 작은 값으로 초기화되어 초기에는 Radiance Field 렌더링을 선호합니다. 초기 반복(early iterations)에서 조명·재질 파라미터가 수렴되지 않은 상태일 때, Radiance Field fallback이 이미지 손실(gradient) 기울기의 건전성을 보장하고 under-fit 상태로 끌어당기는 국소 최솟값(local minima)을 회피합니다.

파라미터가 수렴함에 따라 물리 모델이 점차 주도권을 가지며 distillation progress가 증가합니다. 물리 모델로 표현되지 않는 광 경로(light paths)가 있을 경우, 해당 픽셀의 distillation progress는 완료되지 않아 학습된 Radiance Field가 최종 렌더링에 남아 있게 됩니다.

#### ③ 물리 기반 항 (PBR Term)

물리 기반 항의 기반이 되는 렌더링 방정식:

$$I_\text{phy}(x, \boldsymbol{\omega}_o) = \int_\Omega \rho(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o;\, \mathbf{c}, r, m) \cdot L(\boldsymbol{\omega}_i) \cdot V(x, \boldsymbol{\omega}_i) \cdot (\mathbf{n} \cdot \boldsymbol{\omega}_i)^+ \, d\boldsymbol{\omega}_i$$

여기서 $\rho(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o;\, \mathbf{c}, r, m)$은 microfacet BRDF(Bidirectional Reflectance Distribution Function)이며, $L(\boldsymbol{\omega}_i)$는 환경 맵(environment map)으로 구현됩니다.

#### ④ Specular 항: Cook-Torrance 모델

Specular 항 $I_\text{spec}$을 계산하기 위해 **Cook-Torrance microfacet specular shading model**을 채택하며, $D$, $G$, $F$는 각각 GGX 법선 분포 함수(NDF), 기하학적 감쇠 함수(geometric attenuation function), 근사 Fresnel 항입니다.

$$f_\text{spec}(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) = \frac{D(\mathbf{h},\, r) \cdot G(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o,\, r) \cdot F(\boldsymbol{\omega}_o, \mathbf{h},\, \mathbf{c}_0)}{4(\boldsymbol{\omega}_o \cdot \mathbf{n})(\boldsymbol{\omega}_i \cdot \mathbf{n})}$$

- $\mathbf{h} = \frac{\boldsymbol{\omega}_o + \boldsymbol{\omega}_i}{\|\boldsymbol{\omega}_o + \boldsymbol{\omega}_i\|}$: Halfway vector
- $r$: Roughness
- $m$: Metallic
- $\mathbf{c}$: Albedo(base color)

Diffuse 항은 Lambertian 모델로:

$$f_\text{diff} = \frac{\mathbf{c}_\text{albedo}}{\pi}$$

따라서 전체 BRDF:

$$\rho(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) = (1 - m) \cdot f_\text{diff} + f_\text{spec}$$

#### ⑤ Diffuse 및 Visibility 처리

Diffuse 항의 적분은 구면 조화(SH) triple product로 근사하며, $L(\boldsymbol{\omega}_i)$를 SH에 투영합니다. Visibility $V(x, \boldsymbol{\omega}_i)$는 정규 그리드 위에서 사전 계산되며, 각 셀은 splatted opacity cube-map을 투영하여 SH 계수를 계산합니다. 학습 과정에서 $V$는 초기화 단계에서 한 번만 계산되고 전체 반복 동안 고정(frozen)됩니다.

#### ⑥ Specular 항 적분 (Split-Sum 근사)

적분의 첫 번째 부분은 사전 필터링된 환경 맵(pre-filtered environment maps)으로 참조되며, roughness가 높을수록 낮은 해상도를 사용하는 mip-map 피라미드로 구현됩니다. 두 번째 부분은 조명과 독립적인 BRDF 적분 맵(BRDF integration map)으로 룩업 테이블(lookup table)에 사전 계산됩니다.

$$I_\text{spec} \approx \underbrace{\int_\Omega L(\boldsymbol{\omega}_i) D(\mathbf{h}, r) d\boldsymbol{\omega}_i}_{\text{Pre-filtered Env. Map}} \cdot \underbrace{\int_\Omega f_\text{spec} \cdot (\mathbf{n} \cdot \boldsymbol{\omega}_i) d\boldsymbol{\omega}_i}_{\text{BRDF Integration Map (LUT)}}$$

---

### 2-3. 모델 구조

```
[Stage 1: Pre-training]
  Multi-view Images → 3D Gaussian Splatting (3DGS)
       ↓
  Pre-trained Radiance Field (I_raw) 수렴 완료

[Stage 2: Progressive Distillation]
  I_raw (frozen/fine-tunable) ←→ I_phy (PBR 파라미터 학습)
         ↓ 선형 보간 (α 가중치)
  I_final = α · I_phy + (1-α) · I_raw
         ↓
  이미지 손실(L_image) 역전파 → α, c(albedo), r(roughness), m(metallic), L(env) 업데이트

[Deferred Shading]
  Gaussian→픽셀 매핑 (splatting) → Screen-space 파라미터 합산
         ↓
  최종 픽셀 색상 (deferred shading 방식으로 계산)
```

모든 구성 요소가 결합되면 Gaussian-to-pixel 매핑(선형 함수)이 완성되어 Gaussian 셰이딩 파라미터를 스크린 공간에 splatting하고 blending하며, 최종 픽셀 색상은 deferred shading을 통해 계산됩니다.

구체적으로, 각 학습 이미지에 대해 Radiance Field로 렌더링된 이미지와 PBR 파라미터로 렌더링된 이미지를 각각 생성합니다. 손실 함수는 학습 가능한 distillation progress map을 가중치로 사용하여 두 렌더링을 선형 보간한 최종 예측 이미지에 대해 계산되며, 이는 Radiance 분포 최적화를 통해 기본 Radiance baseline과 동등하거나 더 나은 이미지 품질을 보장합니다.

---

### 2-4. 성능 향상

물리 모델의 제약에 대한 허용성(tolerance)이 설계되어, 모델링되지 않은 색상 성분이 조명·재질 파라미터로 누출(leaking)되는 것을 방지하여 리라이팅 아티팩트를 완화합니다. 남은 Radiance Field는 물리 모델의 한계를 보완하여 고품질 Novel View Synthesis를 보장합니다.

실험 결과, 본 방법은 Novel View Synthesis와 리라이팅 양쪽에서 최신 기법 대비 품질 면에서 크게 능가합니다.

본 방법은 더 매끄러운 표면 법선을 생성하며, 차량 재질을 거의 완전한 specular로 추론하여 새로운 조명 하의 색상 변화를 더 충실하게 재현합니다.

### 2-5. 한계

복잡한 조명·재질 효과(광택이나 경면 반사 등)의 처리가 개선 여지로 남아 있으며, 이러한 시각적으로 도전적인 현상을 정확하게 포착하기 위해 추가 개선이 필요할 수 있습니다.

또한, Gaussian Splatting 기반 역 렌더링에서 간접 조명(indirect illumination)과 같은 복잡한 전역 조명 효과로부터 재질 특성을 정확히 분리하는 것은 여전히 주요 도전 과제입니다.

사전 학습된 Gaussian 프리미티브들은 제한된 학습 시점에 대해서만 감독(supervised)되므로, 관측되지 않은 시점에서의 간접 Radiance 모델링을 위한 감독 신호가 부족합니다.

---

## 3. 모델의 일반화 성능 향상 가능성

Progressive Radiance Distillation의 아이디어는 3D Gaussian Splatting에만 국한되지 않으며, NeRF 표현으로도 일반화할 가능성이 있습니다. 저자들은 메시 기반 역 렌더링 방법인 NDR을 적용하여 Radiance Distillation을 통합함으로써 이 가능성을 탐구했습니다.

NDR은 SDF에서 메시를 추출하고, 본 논문과 동일한 PBR 모델을 역 렌더링에 사용합니다.

Progressive Radiance Distillation 아이디어는 Gaussian Splatting에만 한정되지 않으며, 메시 기반 역 렌더링 방법에 적용했을 때도 **두드러지게 specular한 장면에서 긍정적인 효과**를 보입니다.

수렴된 raw radiance에서 출발함으로써 초기 단계의 local minima를 회피하며, 고도로 specular한 물체에 대해 distillation이 현저히 더 합리적으로 이루어집니다.

Progressive Radiance Distillation은 역 렌더링 분야에서 획기적인 전환을 나타내며, 학습된 Radiance Field의 이점과 물리 기반 모델링의 엄밀함을 효과적으로 균형 잡습니다. 그 다용성(versatility)도 부각되며, Gaussian Splatting을 넘어 메시 기반 렌더링 시스템으로 확장 가능성을 제시해 그래픽스 및 시각 컴퓨팅의 광범위한 응용을 위한 길을 개척합니다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 논문 | 표현 방식 | 접근 방법 | 특징 |
|------|-----------|-----------|------|
| **NeRF** (2020) | MLP | 체적 렌더링 | NVS 혁신, 역 렌더링 기반 마련 |
| **3DGS** (Kerbl et al., 2023) | 3D Gaussian | Splatting | 실시간 렌더링, >30fps@1080p |
| **NDR** | SDF+Mesh | PBR | 메시 기반 역 렌더링 |
| **GaussianShader** | 3D Gaussian | PBR+잔차항 | Specular 분리, 잔차로 간접 조명 근사 |
| **DeferredGS** | 3D Gaussian | Deferred Shading | 리라이팅 아티팩트 감소 |
| **PRD-GS (본 논문)** | 3D Gaussian | Progressive Distillation | PBR + RF 하이브리드, 일반화 가능 |
| **RTR-GS** (2025) | 3D Gaussian | Radiance Transfer + Reflection | 임의 반사율 물체 처리 |

NeRF는 신규 시점 합성에서 뛰어난 성공을 거두며 역 렌더링의 기반을 마련했으나, MLP 기반 방법들은 제한된 표현력과 높은 연산 비용으로 품질과 효율 균형에 어려움이 있습니다.

3DGS는 최신 시각 품질을 유지하면서 경쟁력 있는 학습 시간과 1080p 해상도에서 고품질 실시간($\geq$ 30fps) NVS를 가능하게 하는 세 가지 핵심 요소를 도입했습니다.

GaussianShader는 Specular 성분을 분리하고 잔차 항을 도입하여 2차 조명 효과를 포착하며, PRD-GS는 Progressive Radiance Distillation을 도입합니다.

RTR-GS는 임의의 반사율 특성을 가진 물체의 렌더링, BRDF 및 조명 분해, 신뢰할 수 있는 리라이팅 결과를 제공하는 새로운 역 렌더링 프레임워크로, 전향 렌더링(radiance transfer)과 지연 렌더링(deferred rendering for reflections)을 결합한 하이브리드 렌더링 모델로 기하 구조를 복원합니다.

Gaussian Splatting 기반 역 렌더링은 빠르게 발전하고 있으나, 간접 조명과 같은 복잡한 전역 조명 효과로부터 재질 특성을 정확히 분리하는 것은 여전히 주요 과제로 남아 있으며, 기존 방법들은 종종 Gaussian 프리미티브로부터 간접 Radiance를 질의합니다.

---

## 5. 향후 연구에 미치는 영향과 고려할 점

### 5-1. 향후 연구에 미치는 영향

이 연구는 역 렌더링 및 3D 장면 이해 분야에 중요한 기여를 하며, Gaussian Splatting의 혁신적 활용과 점진적 distillation 과정은 사실적이고 대화형(interactive) 3D 시각화 및 편집을 위한 새로운 가능성을 열어 줍니다.

점진적 distillation 과정과 하이브리드 Radiance Field 표현을 포함하는 핵심 혁신들은 다양한 3D 시각화 및 편집 작업에 가치 있는 도구가 될 수 있는 잠재력을 보여 주며, 역 렌더링의 경계를 탐구하는 연구 커뮤니티에 중요한 통찰을 제공하고 이 분야의 발전을 위한 길을 열어 줍니다.

### 5-2. 향후 연구 시 고려할 점

1. **간접 조명 모델링 강화**
관측되지 않은 시점에 대한 감독을 제공하는 물리 기반 제약(예: radiometric consistency loss)을 도입하여 물리 기반 렌더링과 NVS 양쪽에서 감독 신호를 제공하는 자기 교정(self-correcting) 피드백 루프를 구성할 수 있습니다.

2. **Specular/Glossy 장면 처리 개선**
복잡한 조명·재질 효과(광택 및 경면 반사 등)의 처리가 개선 여지로 남아 있으며, 이러한 시각적으로 도전적인 현상을 더 정확히 포착하기 위한 추가 연구가 필요합니다.

3. **다양한 표현 방식으로의 확장**
Progressive Radiance Distillation은 역 렌더링 분야의 변혁적 단계를 의미하며, 그 다용성이 부각되어 Gaussian Splatting을 넘어 메시 기반 렌더링 시스템으로의 확장을 시사하며 그래픽스 및 시각 컴퓨팅 분야에서 더 넓은 응용의 길을 열어 줍니다.

4. **기존 접근 방법과의 결합**
본 방법과 기존 접근 방식의 결합이 미치는 영향을 완전히 탐구하는 것이 흥미로운 연구 방향이 될 것입니다.

5. **Gaussian 기하 정확도 개선**
복잡한 외관에서 재질 및 조명을 분리하기 위해서는 정확한 기하가 매우 중요하며, 고주파 세부 묘사가 과적합(overfitting)으로 이어져 물리적으로 매끄러운 표면에서 벗어난 부유 아티팩트를 유발할 수 있습니다.

---

## 📚 참고 자료 및 출처

| # | 제목 | 출처 |
|---|------|------|
| 1 | **Progressive Radiance Distillation for Inverse Rendering with Gaussian Splatting** (Ye et al., 2024) | [arXiv:2408.07595](https://arxiv.org/abs/2408.07595) |
| 2 | **Progressive Radiance Distillation** — HTML 전문 | [arxiv.org/html/2408.07595v1](https://arxiv.org/html/2408.07595v1) |
| 3 | **Literature Review: Progressive Radiance Distillation** | [themoonlight.io](https://www.themoonlight.io/en/review/progressive-radiance-distillation-for-inverse-rendering-with-gaussian-splatting) |
| 4 | **RTR-GS: 3D Gaussian Splatting for Inverse Rendering with Radiance Transfer and Reflection** (arXiv:2507.07733) | [arxiv.org/abs/2507.07733](https://arxiv.org/abs/2507.07733) |
| 5 | **Radiometrically Consistent Gaussian Surfels for Inverse Rendering** (arXiv:2603.01491) | [arxiv.org/html/2603.01491](https://arxiv.org/html/2603.01491) |
| 6 | **3D Gaussian Splatting for Real-Time Radiance Field Rendering** (Kerbl et al., arXiv:2308.04079) | [arxiv.org/abs/2308.04079](https://arxiv.org/abs/2308.04079) |
| 7 | **ResearchGate: Progressive Radiance Distillation** | [researchgate.net](https://www.researchgate.net/publication/383120143) |
| 8 | **HuggingFace Paper Page: 2408.07595** | [huggingface.co/papers/2408.07595](https://huggingface.co/papers/2408.07595) |
| 9 | **RTR-GS (ACM MM 2025)** | [dl.acm.org/doi/10.1145/3746027.3755197](https://dl.acm.org/doi/10.1145/3746027.3755197) |
| 10 | **BibSonomy: arXiv:2408.07595** | [bibsonomy.org](https://www.bibsonomy.org/bibtex/10068453eb5b4cf44268098674475ae0e) |

> ⚠️ **정확도 안내**: 본 답변의 수식 일부(특히 세부 NDF/GGX 표현식)는 논문 HTML 원문의 수식 렌더링이 LaTeX가 아닌 MathML/Unicode 형태로 제공되어, 논문의 서술 및 PBR 표준 공식에 기반하여 재구성하였습니다. 정확한 수식 확인을 위해서는 [arXiv PDF 원문](https://arxiv.org/pdf/2408.07595)을 직접 참조하시기 바랍니다.
