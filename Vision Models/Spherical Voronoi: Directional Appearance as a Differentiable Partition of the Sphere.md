
# Spherical Voronoi: Directional Appearance as a Differentiable Partition of the Sphere

> **논문 정보**: Di Sario, F., Rebain, D., Verbin, D., Grangetto, M., & Tagliasacchi, A. (2025). *Spherical Voronoi: Directional Appearance as a Differentiable Partition of the Sphere*. arXiv:2512.14180.
> **소속**: University of Torino, Simon Fraser University, University of British Columbia, University of Toronto, **Google DeepMind**

---

## 1. 핵심 주장 및 주요 기여 (요약)

이 논문은 **Spherical Voronoi(SV)** 를 3D Gaussian Splatting의 외형 표현을 위한 **통합 프레임워크**로 제안하며, SV는 방향성 도메인을 **부드러운 경계를 가진 학습 가능한 영역**으로 분할하여 시점 의존적 효과에 대한 직관적이고 안정적인 파라미터화를 제공한다.

### 주요 기여 요약

| 기여 항목 | 내용 |
|---|---|
| ① 새로운 구형 표현 | 미분 가능한 Voronoi 분할 기반 구형 함수 표현 |
| ② SH 대체 | 3DGS에서 SH를 SV로 교체하는 완전 명시적(explicit) 모델 |
| ③ Voronoi Light Probes | 공간적으로 변하는 반사 효과 처리 |
| ④ 최적화 안정성 | SG보다 안정적이고 SH보다 표현력이 높음 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

---

### 2.1 해결하고자 하는 문제

Radiance field 방법들(예: 3D Gaussian Splatting)은 새로운 시점 합성을 위한 강력한 패러다임으로 부상했지만, 외형 모델링은 Spherical Harmonics(SH)에 의존하는 경우가 많아 근본적인 한계를 지닌다. SH는 고주파 신호를 표현하는 데 어려움을 겪고, **Gibbs 링잉 아티팩트**를 나타내며, 현실적인 렌더링의 핵심인 **정반사(specular reflections) 표현에 실패**한다. Spherical Gaussians(SG) 등의 대안은 개선을 제공하지만, 최적화 복잡도를 크게 증가시킨다.

기존 방법들의 문제점을 수식으로 정리하면:

**① Spherical Harmonics (SH)**

SH는 $f_{\text{SH}}(\omega; \mathbf{c}) = \sum_{l=0}^L \sum_{m=-l}^l c_{lm} Y_l^m(\omega)$ 형태로 전개되는데, 수치적으로 안정적이고 전역 지지를 가지지만, **대역 제한적(band-limited)** 특성으로 인해 고주파 신호를 표현하려면 매우 많은 계수 $L$이 필요하며, 이로 인해 큰 파라미터 수와 Gibbs 링잉 아티팩트가 발생한다.

$$f_{\text{SH}}(\omega; \mathbf{c}) = \sum_{l=0}^{L} \sum_{m=-l}^{l} c_{lm} Y_l^m(\omega)$$

**② Spherical Gaussians (SG)**

Spherical Gaussian과 Spherical Beta는 국소적으로 지지되는 함수와 불연속성을 표현할 수 있지만, 최적화가 안정적이지 않고 **로컬 미니마(local minima)에 빠지기 쉽다**.

---

### 2.2 제안하는 방법 (수식 포함)

Spherical Voronoi는 **고주파 신호를 효과적으로 모델링**하고, 구형 도메인의 **적응적 분해**를 제공하며, **최적화가 더 쉬운** 새로운 명시적 표현이다.

#### 핵심 수식: Soft Voronoi 함수

SV의 핵심은 구 위에 $K$개의 **학습 가능한 사이트(sites)** $\{s_k\}\_{k=1}^{K}$를 배치하고, 각 사이트에 대응하는 **색상 벡터** $\{c_k\}_{k=1}^{K}$를 학습하는 것이다. 임의의 방향 $\omega$에 대한 함수 값은 다음 softmax 가중 합으로 정의된다:

$$f_{\text{SV}}(\omega; \{s_k, c_k\}) = \sum_{k=1}^{K} w_k(\omega) \cdot c_k$$

여기서 가중치 $w_k$는 거리 기반 softmax로 계산된다:

$$w_k(\omega) = \frac{\exp\left( \tau \cdot \langle \omega, s_k \rangle \right)}{\sum_{j=1}^{K} \exp\left( \tau \cdot \langle \omega, s_j \rangle \right)}$$

- $\langle \omega, s_k \rangle$: 방향 벡터와 사이트 사이의 **내적(코사인 유사도)**
- $\tau$: **온도(temperature) 파라미터** — 분할의 날카로움(sharpness)을 제어

온도 파라미터는 **각도 선명도(angular sharpness)를 제어**하여, 낮은 값에서는 부드럽고 넓은 응답을, 높은 값에서는 날카로운 Voronoi형 구면 분할을 만들어낸다.

#### 반사 방향 파라미터화

Ref-NeRF를 따라 반사 방향에서 방향성 외형을 평가함으로써 광택 반사를 파라미터화한다. SV 함수를 통해 반사 복사량 $f(\omega_r)$을 모델링하며, **날카로운 정반사 로브, 다중 피크, 심지어 불연속성까지 표현**하면서도 완전한 미분 가능성을 유지한다.

반사 방향 $\omega_r$는 다음과 같이 정의된다:

$$\omega_r = 2(\omega \cdot \mathbf{n})\mathbf{n} - \omega$$

여기서 $\omega$는 시점 방향, $\mathbf{n}$은 표면 법선 벡터이다.

---

### 2.3 모델 구조

#### ① SV for Radiance (SH 대체)

SH를 3D Gaussian Splatting에서 SV로 교체함으로써 **완전히 명시적이고(fully explicit), 미분 가능하며, 더욱 표현력 있는 시점 방향 모델**을 구성한다.

#### ② Voronoi Light Probes (VLP)

이 표현을 **Voronoi Light Probes**를 통해 공간적으로 변하는 반사 처리로 확장한다. 이는 신경망 디코더 없이 근거리 효과를 포착하는 **완전 명시적 공식화**이며, 지연 쉐이딩(deferred shading)과 프로브 보간(probe interpolation)에 의존하여 반사 벤치마크에서 최고 수준의 품질을 달성한다.

#### ③ 가속화 기법 (Cubemap 인덱싱)

핵심 아이디어는 단위 구를 **저해상도 큐브맵(cubemap)** 으로 분할하고, 각 텍셀에 고정된 후보 사이트 집합을 미리 할당하는 것이다. 런타임 시 softmax는 전체 사이트가 아닌 이 후보 집합으로만 제한된다.

이를 통해 다음 계산 병목을 해결한다:

반사 모델링처럼 많은 수의 사이트가 필요한 경우 나이브 평가는 비실용적이 된다. 특히 수천 개의 사이트를 사용할 때 모든 픽셀에서 방향성 외형을 평가하는 렌더링 중 심각한 병목이 된다. 이를 완화하기 위해, 어떤 방향에 대해서도 소수의 사이트만이 관련됨을 이용하는 간단한 가속 방법을 도입한다.

---

### 2.4 성능 향상

SV는 레이디언스 필드 기반 재구성에서 방향성 외형을 모델링하기 위한 새로운 명시적 표현으로 도입되었다. **구면의 적응적 분해와 안정적인 최적화 동작 덕분에, SV는 레이디언스 전용 환경에서 SH, SG, SB 등 기존 기저 함수들을 일관되게 능가**하면서, 기반이 되는 렌더링 백본과 유사한 런타임을 유지한다.

확산(diffuse) 외형의 경우 SV는 기존 대안보다 더 단순한 최적화를 유지하면서 경쟁력 있는 결과를 달성한다. 반사의 경우(SH가 실패하는 영역) SV를 학습 가능한 반사 프로브로 활용하여 고전적 그래픽스 원리를 따라 반사 방향을 입력으로 사용한다. 이 공식화는 **합성 및 실세계 데이터셋 모두에서 최고 수준의 결과를 달성**하며, SV가 원칙적이고, 효율적이며, 일반적인 외형 모델링 솔루션임을 입증한다.

SV는 **일관되게 더 나은 재구성으로 수렴**한다.

반사 모델링 실험에서 제안된 방법은 **모든 데이터셋에서 최고 수준의 성능을 일관되게 달성**한다.

---

### 2.5 한계

$f(\omega_r)$만을 평가하는 것은 **원거리 조명(far-field illumination)** 을 가정하는데, 이는 기하학이나 광원이 표면 가까이에 있을 때 **근거리 효과에서는 한계**를 보인다.

추가적으로 논문에서 확인된 한계:

- 반사 모델링 시 많은 수의 사이트가 필요하여, 픽셀당 방향성 외형을 평가할 때 렌더링 중 **심각한 병목**이 발생할 수 있다.
- 공간적으로 변하는 조명이 있는 장면에서 **공간 불변 모델**을 사용할 경우, 상충되는 측정치가 평균화되어 **흐릿한(blurry) 재구성**이 발생한다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 명시적 표현의 구조적 일반화

지연 쉐이딩과 프로브 보간에 의존함으로써 반사 벤치마크에서 최고 수준의 품질을 달성한다. 전체적인 결과는 **신중하게 설계된 명시적 표현이 신경망 기반 외형 모델에 대한 강력하고 효율적인 대안으로 기능**할 수 있음을 보여주며, 고품질의 시점 의존적 효과, 해석 가능성(interpretability), 실용적인 렌더링을 가능하게 한다.

즉, 신경망 디코더가 없기 때문에:
- **과적합(overfitting) 위험 감소**: MLP 파라미터에 의존하지 않으므로 훈련 시점 수 부족에 대한 민감도 완화
- **도메인 이전(domain transfer) 잠재력**: 명시적 사이트-색상 표현은 장면 간 이전 학습 가능성 제공

### 3.2 적응적 분해를 통한 일반화

SV의 학습 가능한 사이트는 훈련 데이터의 반사 분포에 따라 **자동으로 적응적 배치**되므로:

- **다양한 반사 특성을 가진 장면**(확산~고광택)에 모두 적용 가능
- 온도 파라미터 $\tau$를 통한 표현 복잡도 제어로 장면 유형별 모델 적응성 향상

SV가 "합성 및 실세계 데이터셋 모두에서 최고 수준의 결과를 달성"한 것은 다양한 환경에서의 **일반화 능력**을 직접 보여주는 증거이다.

### 3.3 Voronoi Light Probes의 공간 일반화

공간적으로 변하는 반사로의 확장인 **Voronoi Light Probes**는 신경망 디코더 없이 복잡한 근거리 효과를 포착하는 완전 명시적 공식화이며, 이는 다양한 공간 조명 변화가 있는 실세계 장면에서도 일반화를 가능하게 한다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

| 영역 | 영향 |
|---|---|
| 3D Gaussian Splatting | SH의 실질적 대안으로 SV 도입 가능성 확립 |
| 역 렌더링(Inverse Rendering) | 명시적 조명 프로브 학습에 적용 가능 |
| 실시간 렌더링 | 가속 cubemap 기법으로 실시간 환경 확대 |
| 해석 가능한 AI | 신경망 비의존 명시적 표현 패러다임 강화 |

SV 모델은 기존 색상 파라미터화를 일관되게 능가하며, 주목할 만하게도 **신경 필드가 아닌 공식화 중 처음으로** 이를 달성한 방법이다.

### 4.2 관련 최신 연구 비교 분석 (2020년 이후)

시점 의존적 색상과 불확실성은 구형 조화 함수(SH) 전개나 SG-Splatting 및 Spec-Gaussian에서처럼 구형 가우시안(SG) 또는 비등방성 구형 가우시안(ASG)의 혼합을 통해 효율적으로 모델링될 수 있으며, 이는 단위 구에서 날카롭고 비등방성을 가진 하이라이트를 포착한다.

**Dual SH for 3DGS** (SIGGRAPH 2025)는 확산 성분에는 시점 방향, 정반사 성분에는 반사 방향이라는 서로 다른 방향성 파라미터화를 사용하여 확산-정반사 구성 요소를 분리 모델링하는 이중 SH 분해 프레임워크를 제시하며, 반사 표면의 현실적 렌더링과 Gaussian Splatting의 계산 효율성을 동시에 유지한다.

SV와 관련 연구의 비교:

| 방법 | 표현 유형 | 고주파 | 최적화 | 반사 | 설명 가능성 |
|------|----------|--------|--------|------|------------|
| **SH** (3DGS 기본) | 암시적/글로벌 | ❌ Gibbs | ✅ 안정 | ❌ 실패 | ✅ |
| **SG** (SG-Splatting) | 명시적/국소 | ✅ | ⚠️ 불안정 | ⚠️ 부분적 | ✅ |
| **ASG** (Spec-Gaussian) | 명시적/비등방 | ✅ | ⚠️ 복잡 | ✅ | ✅ |
| **Dual SH** | 암시적/이중 | ⚠️ | ✅ 안정 | ✅ | ✅ |
| **SV (본 논문)** | 명시적/적응적 | ✅ | ✅ 안정 | ✅ | ✅ |

### 4.3 앞으로 연구 시 고려할 점

1. **동적 장면(Dynamic Scenes) 확장**:
   현재 SV는 정적 장면에 초점을 맞추고 있다. Global Neural Texture Splatting과 같이 공간적 위치, 시점, 시간에 조건화된 신경 디코더를 활용하는 방향으로 동적 SV 확장 연구가 필요하다.

2. **근거리 조명 처리 개선**:
   반사 방향만 평가하는 것은 원거리 조명을 가정하는 것으로, 기하학이나 광원이 표면 가까이에 있을 때 성능이 저하된다. 근거리 효과를 더 정밀하게 처리하는 VLP의 고도화가 필요하다.

3. **사이트 수(K) 결정 전략**:
   최적의 사이트 수 $K$를 자동으로 결정하는 적응적 알고리즘 연구(예: 성장·가지치기 전략)가 유망하다.

4. **물리 기반 렌더링(PBR)과의 통합**:
   SV를 BRDF 분해 파이프라인과 결합하여 조명과 재질을 분리하는 역 렌더링 연구가 중요한 방향이다.

5. **대규모 장면으로의 확장**:
   대규모 야외 장면에서 Voronoi Light Probes의 효율적 배치와 보간 전략에 대한 연구가 필요하다.

---

## 📚 참고 자료 / 출처

1. **논문 원문 (arXiv)**: Di Sario et al. (2025). *Spherical Voronoi: Directional Appearance as a Differentiable Partition of the Sphere*. arXiv:2512.14180. https://arxiv.org/abs/2512.14180

2. **논문 PDF (arXiv)**: https://arxiv.org/pdf/2512.14180

3. **논문 HTML (arXiv)**: https://arxiv.org/html/2512.14180v1

4. **공식 프로젝트 페이지**: https://sphericalvoronoi.github.io/

5. **공식 GitHub 저장소**: https://github.com/sphericalvoronoi/sphericalvoronoi

6. **문헌 리뷰 (Moonlight)**: https://www.themoonlight.io/en/review/spherical-voronoi-directional-appearance-as-a-differentiable-partition-of-the-sphere

7. **관련 연구 — Dual SH for 3DGS (SIGGRAPH 2025)**: https://dl.acm.org/doi/10.1145/3756863.3769709

8. **관련 연구 — SG-Splatting (arXiv 2025)**: https://arxiv.org/pdf/2501.00342

9. **관련 연구 — SpecGaussian (ACM MM 2024)**: https://arxiv.org/pdf/2409.05868

10. **관련 연구 — Reflective Gaussian Splatting (ICLR 2025)**: https://proceedings.iclr.cc/paper_files/paper/2025/file/abf3682c9cf9245a0294a4bebe4544ff-Paper-Conference.pdf
