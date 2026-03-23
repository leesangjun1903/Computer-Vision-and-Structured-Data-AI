# Ref-NeRF: Structured View-Dependent Appearance for Neural Radiance Fields

---

## 1. 핵심 주장 및 주요 기여 (요약)

Neural Radiance Fields (NeRF)는 장면을 연속적인 체적 함수로 표현하여 MLP가 각 위치에서의 볼륨 밀도와 시점 의존적 방출 복사를 제공하는 뷰 합성 기법이다. 그러나 NeRF 기반 기법은 광택 표면(glossy surface)의 외관을 정확히 캡처하고 재현하는 데 종종 실패한다.

Ref-NeRF는 NeRF의 시점 의존적 출사 복사(outgoing radiance) 매개변수화를 **반사 복사(reflected radiance)**의 표현으로 대체하고, 공간적으로 변하는 장면 속성의 집합을 사용하여 이 함수를 구조화한다.

**주요 기여 3가지:**

1. 반사 방향 재매개변수화(Reflection Direction Reparameterization): 시점 벡터 대신 법선 벡터에 대한 시점 벡터의 반사를 MLP 입력으로 사용하여, 정반사(specular) 외관의 보간을 크게 개선한다.
2. Integrated Directional Encoding (IDE): von Mises-Fisher 분포 하에서 구면 조화 함수(spherical harmonics)의 기댓값을 사용하여 물체의 거칠기를 명시적으로 모델링하며, 서로 다른 거칠기를 가진 점들 간에 발산 함수를 공유할 수 있게 한다.
3. 법선 벡터에 대한 정규화기(regularizer)를 도입하여 정반사 반사의 사실성과 정확도를 크게 향상시키며, 모델의 내부 표현이 해석 가능하고 장면 편집에 유용하다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 NeRF는 단순한 장면에서도 실제 복사 함수가 시점 방향에 따라 급격히 변하며, 특히 정반사 하이라이트 주변에서 더욱 심하다. 결과적으로 NeRF는 훈련 이미지에서 관찰된 특정 시점 방향에서만 정확히 렌더링할 수 있고, 새로운 시점에서의 광택 외관 보간이 불량하다. 또한, NeRF는 표면의 시점 의존적 복사 대신 물체 내부의 등방성 발광체(isotropic emitters)를 사용하여 정반사 반사를 "위조(fake)"하는 경향이 있어, 반투명하거나 "안개낀(foggy)" 셸을 가진 물체가 생성된다.

### 2.2 제안하는 방법 및 수식

#### (A) 반사 방향 재매개변수화

기존 NeRF가 시점 방향 $\hat{\omega}_o$를 직접 MLP에 입력하는 것과 달리, Ref-NeRF는 **반사 방향** $\hat{\omega}_r$을 사용한다:

$$\hat{\omega}_r = 2(\hat{\omega}_o \cdot \hat{n})\hat{n} - \hat{\omega}_o$$

여기서 $\hat{n}$은 해당 지점의 표면 법선 벡터이다. 반사 방향을 사용하면 발산 함수(emittance function)의 학습과 보간이 현저히 쉬워져 결과가 크게 개선된다.

#### (B) Integrated Directional Encoding (IDE)

IDE는 공간적으로 변하는 집중 매개변수(concentration parameter) $\kappa$를 가진 von Mises-Fisher(vMF) 분포 하에서 구면 조화 함수 집합의 기댓값을 사용하여 물체 거칠기를 명시적으로 모델링한다.

vMF 분포는 다음과 같이 정의된다:

$$\text{vMF}(\hat{\omega}; \hat{\omega}_r, \kappa) = c(\kappa) \exp(\kappa \hat{\omega}_r^\top \hat{\omega})$$

여기서 $c(\kappa)$는 정규화 상수, $\kappa$는 집중도(roughness의 역)이다.

IDE의 핵심 수식으로, vMF 분포 하에서 구면 조화 함수 $Y_\ell^m$의 기댓값은 다음과 같다:

$$\mathbb{E}_{\hat{\omega} \sim \text{vMF}(\hat{\omega}_r, \kappa)}[Y_\ell^m(\hat{\omega})] = A_\ell(\kappa) \cdot Y_\ell^m(\hat{\omega}_r)$$

여기서 감쇠 함수(attenuation function) $A_\ell(\kappa)$는:

$$A_\ell(\kappa) = \frac{\kappa}{2\sinh\kappa} \int_{-1}^{1} P_\ell(u) e^{\kappa u} du$$

$P_\ell$은 르장드르 다항식(Legendre polynomial)이다.

거칠기가 적은(매끈한) 위치는 고주파 인코딩을 받고, 거친 영역은 고주파가 감쇠된 인코딩을 받아, 서로 다른 거칠기를 가진 위치들 간에 조명 정보가 공유되고 반사율 편집이 가능하다.

#### (C) 확산/정반사 분리 (Diffuse-Specular Decomposition)

Ref-NeRF는 출사 복사를 명시적인 확산(diffuse) 및 정반사(specular) 성분으로 분리한다. Spatial MLP가 확산 색상(diffuse color)과 정반사 틴트(specular tint)를 출력하고, 이를 Directional MLP의 정반사 색상과 결합한다.

최종 색상 $\mathbf{c}$는 다음과 같이 합성된다:

$$\mathbf{c} = \mathbf{c}_d + s \cdot \mathbf{c}_s$$

여기서:
- $\mathbf{c}_d$: spatial MLP에서 출력하는 확산 색상
- $s$: 정반사 틴트 (specular tint)
- $\mathbf{c}\_s = F_{\theta}(\text{IDE}(\hat{\omega}_r, \kappa))$: directional MLP에서 출력하는 정반사 색상

#### (D) 법선 벡터 정규화 (Normal Regularization)

Ref-NeRF는 볼륨 밀도에 대한 새로운 정규화기를 도입하여, 노이즈가 있는 법선 벡터와 안개낀 체적 기하를 개선하고, 밀도를 표면 주위에 집중시킨다.

Orientation loss는 다음과 같이 정의된다:

$$\mathcal{L}_o = \sum_i w_i \max(0, \hat{n}_i \cdot \hat{d})^2$$

여기서 $w_i$는 렌더링 가중치, $\hat{n}_i$는 예측된 법선, $\hat{d}$는 광선 방향이다. 이는 법선이 카메라를 향하도록 유도한다.

법선 예측 손실:

$$\mathcal{L}_n = \sum_i w_i \|\hat{n}_i - \hat{n}_i^{\text{grad}}\|^2$$

여기서 $\hat{n}_i^{\text{grad}} = -\nabla \sigma / \|\nabla \sigma\|$는 밀도 필드의 그래디언트에서 도출된 법선이다.

### 2.3 모델 구조

Ref-NeRF는 두 개의 MLP로 구성된다:

| 구성요소 | 입력 | 출력 |
|---------|------|------|
| **Spatial MLP** | 위치 $\mathbf{x}$ (positional encoding) | 밀도 $\sigma$, 확산 색상 $\mathbf{c}_d$, 정반사 틴트 $s$, 거칠기 $\kappa$, 예측 법선 $\hat{n}$, 병목 벡터 |
| **Directional MLP** | IDE($\hat{\omega}_r, \kappa$), 병목 벡터 | 정반사 색상 $\mathbf{c}_s$ |

볼륨 렌더링은 기존 NeRF와 동일한 적분 방식을 따른다:

$$\hat{C}(\mathbf{r}) = \sum_i T_i \alpha_i \mathbf{c}_i, \quad T_i = \prod_{j<i}(1-\alpha_j)$$

### 2.4 성능 향상

Ref-NeRF는 mip-NeRF 등 기존 최첨단 방법을 크게 능가한다. 예를 들어, "Shiny Blender" 데이터셋에서 Ref-NeRF는 PSNR 47.7dB, 평균 각도 오차(MAE) 1.6°를 달성한 반면, mip-NeRF는 PSNR 23.2dB, MAE 96.6°로, 시각적 사실성과 기하학적 충실도 모두에서 극적인 향상을 보인다.

### 2.5 한계

Ref-NeRF는 이전 최첨단 방법 대비 크게 향상되지만, 계산량이 증가한다: IDE 평가가 표준 positional encoding보다 약간 느리고, 법선 벡터 계산을 위해 spatial MLP의 그래디언트를 역전파하므로 mip-NeRF보다 약 25% 느리다.

출사 복사의 반사 방향 재매개변수화는 상호 반사(interreflection)나 비원거리 조명(non-distant illumination)을 명시적으로 모델링하지 않으므로, 이러한 경우에서의 성능 향상이 감소한다.

또한 Ref-NeRF는 주로 환경 맵 조명 조건하의 물체 수준 재구성에 적합하며, 근거리 조명(near-field lighting)에서는 성능이 떨어진다. 이는 환경 맵이 공간적으로 변하는 근거리 조명 조건에서 잘 작동하지 않기 때문이다.

---

## 3. 모델의 일반화 성능 향상 가능성

Ref-NeRF의 일반화 성능과 관련하여 다음과 같은 핵심 포인트가 있다:

### 3.1 구조화된 표현의 일반화 이점

Ref-NeRF의 핵심 통찰은 NeRF의 시점 의존적 외관 표현을 구조화하면 기저 함수가 단순해지고 보간이 쉬워진다는 것이다. 이는 곧 **학습되지 않은 시점(unseen viewpoint)**에서의 렌더링 품질, 즉 일반화 성능을 직접적으로 향상시킨다.

### 3.2 IDE의 구 위의 정상성(Stationarity on the Sphere)

이론적으로 IDE 인코딩은 구 위에서 정상성을 가지며, 이는 NeRF의 positional encoding이 유클리드 공간에서 갖는 정상성과 유사하다. 이 속성은 반사 방향의 회전에 대해 인코딩이 불변(equivariant)임을 의미하여, **다양한 반사 환경에서의 일반화**를 촉진한다.

### 3.3 거칠기 간 정보 공유

IDE를 통해 directional MLP는 연속적으로 변하는 거칠기를 가진 재질의 반사 복사 함수를 효율적으로 표현한다. 각 인코딩 성분은 vMF 분포의 집중 매개변수 $\kappa$로 컨볼브된 구면 조화 함수이며, 거칠기가 적은 위치는 고주파 인코딩, 거친 영역은 감쇠된 고주파를 받는다. 이를 통해 서로 다른 거칠기를 가진 위치들 간에 조명 정보가 공유된다.

### 3.4 일반화의 한계 및 개선 방향

- **근거리 조명**: 기존 연구들은 Ref-NeRF 등의 방법이 실시간 이미지 기반 조명(IBL) 기법에서 영감을 받은 휴리스틱 모듈을 활용하지만, 원거리 조명을 가정한 물체 수준 재구성에 국한되어 환경 맵이 공간적으로 변하는 근거리 조명 환경에서는 성능이 떨어진다고 지적한다.
- **상호 반사**: 다중 반사 경로(multi-bounce reflection)를 모델링하지 않아, 복잡한 실내 장면에서의 일반화가 제한된다.
- **Relighting/편집**: Ref-NeRF는 재질과 환경 조명을 분해하지 않고 단순화된 함수로 셰이딩하여, 릴라이팅 및 편집 같은 다운스트림 작업에서의 적용성이 제한된다.

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 4.1 연구 영향

1. **반사 방향 재매개변수화의 패러다임 확립**: Ref-NeRF의 반사 방향 조건부 시점 의존적 외관 모델링은 이후의 NeRF 개선 연구에서 핵심 원리로 자리잡았으며, 이 재매개변수화가 기저 장면 함수를 단순화하여 광택 물체에 대한 더 나은 기하 및 뷰 보간 품질을 이끌어낸다.

2. **표면 기반 확장 (Ref-NeuS 등)**: Ref-NeuS는 Ref-NeRF 위에 SDF를 추정하고, SDF를 활용하여 가시성 매개변수에 접근하여 반사 영역의 기여도를 조정하는 방식으로 발전하였다.

3. **장면 편집 가능성**: Ref-NeRF의 구조화된 표현은 훈련 후 재질 속성(거칠기, 정반사/확산 비율, 확산 색상 등)을 직관적으로 변경할 수 있는 장면 편집을 가능하게 한다.

4. **후속 인코딩 방법론 영감**: Ref-NeRF의 IDE는 NeAI의 Integrated Lobe Encoding (ILE), SpecNeRF의 Gaussian Directional Encoding 등 새로운 인코딩 방법론에 직접적인 영감을 제공하였다.

### 4.2 향후 연구 시 고려할 점

| 고려사항 | 설명 |
|----------|------|
| **근거리 조명** | 공간적으로 변하는 조명 환경에서의 반사 모델링 필요 |
| **상호 반사** | Multi-bounce path tracing 또는 근사 기법 통합 |
| **계산 효율성** | IDE 및 법선 그래디언트 역전파의 오버헤드 감소 |
| **비등방성 반사** | Brushed metal 등 비등방성 재질에 대한 확장 |
| **Gaussian Splatting 통합** | NeRF의 체적 표현에서 3DGS 기반 표현으로의 전환 |
| **재질-조명 분해** | Inverse rendering을 위한 물리 기반 분해 강화 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 방법론 | Ref-NeRF 대비 차별점 |
|------|------|------------|---------------------|
| **Mip-NeRF** (Barron et al.) | 2021 | 원뿔 추적, 안티앨리어싱 | 광택 표면에 대한 특별한 처리 없음 |
| **Mip-NeRF 360** (Barron et al.) | 2022 | 비제한 장면으로 확장 | 반사 모델링은 제한적 |
| **Ref-NeRF** (Verbin et al.) | 2022 | 반사 방향 + IDE | 본 논문 |
| **Ref-NeuS** (Ge et al.) | 2023 | Ref-NeRF에 SDF 기반 표면 추정 추가 | 표면 재구성 정확도 향상 |
| **SpecNeRF** (Ma et al.) | 2023 | 3D Gaussian Directional Encoding | 근거리 조명 조건에서의 정반사 모델링 개선 |
| **Neural Directional Encoding (NDE)** (Wu et al.) | 2024 | 반사 방향이 아닌 공간적 큐브맵/볼륨 기반 인코딩 | 간접 조명을 포함하되 계산 비용이 낮음 |
| **3DGS** (Kerbl et al.) | 2023 | 3D Gaussian 프리미티브와 타일 기반 미분 래스터라이저로 실시간 렌더링 | SH 함수의 방향 주파수가 제한되어 정반사 반사 모델링이 어려움 |
| **GaussianShader** (Jiang et al.) | 2024 | 물리 기반 BRDF + 잔여 색상항, Ref-NeRF에 없는 간접 반사 모델링 | 실시간 3DGS 기반이나 노이즈 발생 가능 |
| **3DGS-DR (Deferred Reflection)** (Ye et al.) | 2024 | 스크린 스페이스 맵에 법선/반사 강도를 베이킹 후 환경맵 쿼리 | 픽셀 레벨 반사 계산으로 효율성 유지 |
| **NeRF-Casting** (Verbin et al.) | 2024 | 반사 광선을 NeRF 표현을 통해 추적하여 특징 벡터를 렌더링 | 근거리 반사를 포함한 실세계 장면에서 사실적인 정반사 외관 합성이 가능한 유일한 기존 NeRF 방법 |
| **Normal-NeRF** (2025) | 2025 | 투과도 그래디언트를 spatial MLP로 정제하여 정확하고 정밀한 법선 예측 | Ref-NeRF가 고반사 표면에서 반투명 표면과 노이즈 법선맵을 생성하는 모호성 문제 해결 |
| **Ref-GS** (Zhang et al.) | 2025 | 2D Gaussian Splatting 기반 반사 장면 재구성, IDE를 GS에 적용 | GS 프레임워크 내 반사 모델링 |
| **Reflective Gaussian Splatting** | 2025 | 3DGS에 IDE 적용하여 반사면 모델링 | 환경 조명 분해 가능하여 릴라이팅 지원 |

---

## 6. 참고자료 및 출처

1. Verbin, D., Hedman, P., Mildenhall, B., Zickler, T., Barron, J.T., Srinivasan, P.P. — *"Ref-NeRF: Structured View-Dependent Appearance for Neural Radiance Fields"*, CVPR 2022 — [arXiv:2112.03907](https://arxiv.org/abs/2112.03907)
2. Ref-NeRF 프로젝트 페이지 — [dorverbin.github.io/refnerf](https://dorverbin.github.io/refnerf/)
3. Ref-NeRF CVPR 2022 Open Access — [thecvf.com](https://openaccess.thecvf.com/content/CVPR2022/html/Verbin_Ref-NeRF_Structured_View-Dependent_Appearance_for_Neural_Radiance_Fields_CVPR_2022_paper.html)
4. Ref-NeRF Supplementary Material — [CVPR 2022 Supplemental PDF](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Verbin_Ref-NeRF_Structured_View-Dependent_CVPR_2022_supplemental.pdf)
5. ar5iv HTML version — [ar5iv.labs.arxiv.org/html/2112.03907](https://ar5iv.labs.arxiv.org/html/2112.03907)
6. Ma, L. et al. — *"SpecNeRF: Gaussian Directional Encoding for Specular Reflections"*, 2023 — [arXiv:2312.13102](https://arxiv.org/html/2312.13102v3)
7. Wu, L. et al. — *"Neural Directional Encoding for Efficient and Accurate View-Dependent Appearance Modeling"*, CVPR 2024
8. Jiang et al. — *"GaussianShader: 3D Gaussian Splatting with Shading Functions for Reflective Surfaces"*, CVPR 2024
9. Ye et al. — *"3D Gaussian Splatting with Deferred Reflection"*, SIGGRAPH 2024
10. Verbin, D. et al. — *"NeRF-Casting: Improved View-Dependent Appearance with Consistent Reflections"*, SIGGRAPH Asia 2024
11. *"Normal-NeRF: Ambiguity-Robust Normal Estimation for Highly Reflective Scenes"*, 2025 — [arXiv:2501.09460](https://arxiv.org/html/2501.09460)
12. Zhang et al. — *"Ref-GS: Directional Factorization for 2D Gaussian Splatting"*, CVPR 2025
13. *"Reflective Gaussian Splatting"*, ICLR 2025
14. Semantic Scholar — [Ref-NeRF page](https://www.semanticscholar.org/paper/40c8c8d8a41c16a0e017cc0d059fae9d346795f0)
15. liner.com — *"Quick Review: Ref-NeRF"* — [liner.com/review/refnerf](https://liner.com/review/refnerf-structured-viewdependent-appearance-for-neural-radiance-fields)
16. *"Material transforms from disentangled NeRF representations"*, 2024 — [arXiv:2411.08037](https://arxiv.org/html/2411.08037v1)
17. *"Neural Radiance Fields for the Real World: A Survey"*, 2025 — [arXiv:2501.13104](https://arxiv.org/html/2501.13104v1)

---

> **참고**: 위 분석에서 제시한 수식은 원 논문 및 보충 자료에 기반하되, 표기를 일관되게 정리한 것입니다. IDE의 감쇠 함수 $A_\ell(\kappa)$에 대한 근사식 및 재귀 관계 등의 세부 수학적 증명은 CVPR 2022 보충 자료에서 확인할 수 있습니다.
