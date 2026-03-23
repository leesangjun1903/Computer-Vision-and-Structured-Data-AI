
# Gaussian Splatting with NeRF-based Color and Opacity

**논문 정보:**  
- 저자: Dawid Malarz, Weronika Smolak-Dyżewska, Jacek Tabor, Sławomir Konrad Tadeja, Przemysław Spurek  
- 발표: arXiv:2312.13729 (2023.12), *Computer Vision and Image Understanding* (CVIU, 2024)  
- 코드: [GitHub - gmum/ViewingDirectionGaussianSplatting](https://github.com/gmum/ViewingDirectionGaussianSplatting)

---

## 1. 핵심 주장 및 주요 기여 (요약)

본 논문은 NeRF와 GS 두 모델의 단점을 보완하기 위해 **Viewing Direction Gaussian Splatting (VDGS)**라는 하이브리드 모델을 제안한다. VDGS는 GS 기반의 3D 객체 형상 표현과 NeRF 기반의 색상 및 불투명도 인코딩을 결합한다.

### 주요 기여:
1. NeRF와 GS를 모두 활용하는 하이브리드 아키텍처를 제안하였으며, 기존 GS 대비 색상과 불투명도가 시점 방향(viewing direction)에 더 민감하게 반응한다.
2. 추가적인 텍스처나 조명 컴포넌트 없이도 그림자, 빛 반사, 3D 객체의 투명성을 더 잘 묘사할 수 있다.
3. VDGS는 GS가 NeRF 방식으로 신경망에 의해 조건화(condition)될 수 있음을 보여준다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

NeRF는 신경망 가중치에 형상과 색상 정보를 인코딩하여 3D 객체의 복잡성을 포착하는 데 뛰어난 잠재력을 보여주었다. 반면, Gaussian Splatting (GS)은 신경망 없이도 유사한 렌더링 품질을 더 빠른 학습 및 추론으로 제공하지만, 약 수십만 개의 Gaussian 컴포넌트가 필요하기 때문에 조건화(conditioning)가 어렵다.

정리하면:

| 모델 | 장점 | 단점 |
|------|------|------|
| **NeRF** | 정교한 뷰 합성, 조건화 용이 | 느린 학습/추론 시간 |
| **GS** | 빠른 학습/추론, 실시간 렌더링 | 조건화 어려움, 반사/투명성 처리 미흡 |

### 2.2 제안하는 방법 (VDGS)

#### (a) 기본 NeRF 수식

NeRF는 3D 위치 $\mathbf{x}$와 시점 방향 $\mathbf{d}$를 입력으로 받아 색상 $\mathbf{c}$와 밀도 $\sigma$를 출력하는 MLP로 표현된다:

$$F_{\text{NeRF}}(\mathbf{x}, \mathbf{d}; \Theta) = (\mathbf{c}, \sigma)$$

이 과정에서 이미지를 통과하는 레이(ray)를 생성하고, 신경망이 표현하는 3D 객체와 상호작용하여 관찰된 색상과 깊이를 사용해 학습한다.

#### (b) 기본 3D Gaussian Splatting 수식

3DGS에서 각 Gaussian 컴포넌트 $g_i$는 다음과 같이 정의된다:

$$g_i = \{\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i, o_i, \mathbf{c}_i\}$$

여기서 $\boldsymbol{\mu}_i \in \mathbb{R}^3$은 평균 위치, $\boldsymbol{\Sigma}_i \in \mathbb{R}^{3 \times 3}$는 공분산 행렬, $o_i \in \mathbb{R}$은 불투명도, $\mathbf{c}_i$는 색상(구면 조화함수, SH)이다.

렌더링 시 알파 블렌딩을 통해 픽셀 색상 $\mathbf{C}$를 계산한다:

$$\mathbf{C} = \sum_{i \in \mathcal{N}} \mathbf{c}_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)$$

여기서 $\alpha_i$는 2D Gaussian의 평가값과 불투명도의 곱이다.

#### (c) VDGS의 핵심 제안

VDGS는 Gaussian 분포의 학습 가능한 위치(평균), 형상(공분산), 색상(구면 조화함수), 불투명도에 더해, Gaussian의 위치와 시점 방향을 입력으로 받아 불투명도의 변화량을 출력하는 신경망을 사용한다. VDGS는 색상이나 색상과 불투명도 모두를 갱신할 수 있으나, ablation study 결과 **불투명도만 변경**하는 것이 가장 일관되게 높은 품질을 보였다.

VDGS의 핵심 수식은 다음과 같이 표현할 수 있다. 신경망 $F_{\text{NN}}$이 Gaussian 파라미터와 시점 방향을 입력받아 불투명도 변화량 $\Delta o$를 출력한다:

$$\Delta o_i = F_{\text{NN}}(\boldsymbol{\mu}_i, \mathbf{d}; \Phi)$$

최종 불투명도는 곱셈(multiplication) 방식으로 조합된다:

$$o_i^{\text{VDGS}}(\mathbf{d}) = o_i \cdot \sigma(\Delta o_i)$$

여기서 $\sigma(\cdot)$는 시그모이드 함수, $o_i$는 원래의 학습 가능한 불투명도이다.

논문에서는 PSNR 기준으로 실제 장면 데이터셋에서 가장 많은 최고 점수를 얻었기 때문에 **VDGS Opacity Multiply**를 최종 모델로 선택하였다.

### 2.3 모델 구조

VDGS의 전체 구조를 정리하면:

```
입력: SfM 포인트 클라우드에서 초기화된 3D Gaussian 집합
       + 각 카메라의 시점 방향 d

각 Gaussian 컴포넌트:
  ├── 학습 가능 파라미터: μ (위치), Σ (공분산/형상), c (SH 색상), o (불투명도)
  └── 신경망(MLP): F_NN(μ, d; Φ) → Δo (불투명도 변화량)

최종 렌더링:
  o_i^{VDGS}(d) = o_i · σ(Δo_i)
  C = Σ c_i · α_i^{VDGS} · Π(1 - α_j^{VDGS})
```

최근 연구에서 VDGS는 MLP modulation을 통해 불투명도에 대한 view-dependence를 확장하는 것으로 분류되며, 이는 실시간 성능과 견고한 기하학적 복원을 제공하지만, 구면 조화함수 기반의 색상 모델링은 고주파 view-dependent 효과에는 여전히 한계가 있다.

### 2.4 성능 향상

VDGS는 NeRF Synthetic, Tanks and Temples, Mip-NeRF 360, Deep Blending, Shiny Blender 등 다양한 데이터셋에서 평가되었다.

VDGS는 합성 데이터와 실제 장면 모두에서 유리 표면과 반사 객체를 정확하게 모델링할 수 있으며, 특정 시점 방향에서 나타나는 아티팩트를 제거하고 배경을 더 정확하게 표현할 수 있다.

빠른 학습과 추론이 관찰되었으며, NeRF 기반 모델과 유사한 품질로 그림자, 빛 반사, 투명성을 모델링할 수 있다. 실험 결과 VDGS가 NeRF와 GS 모델 모두보다 우수한 결과를 보였다.

VDGS는 GS와 동일한 학습 및 추론 시간을 가진다. 시점 방향 신경망을 사용하기 때문에 약간 더 긴 학습 및 추론 시간이 소요되지만, 실제로 추론 시 실시간으로 작동할 수 있다.

### 2.5 한계

VDGS는 유리 표면 렌더링, 그림자, 아티팩트 제거 등 다양한 작업에서 기존 GS를 능가하지만, 범용적(general) 모델이기 때문에 정확한 빛 반사 등 물리적 속성을 시뮬레이션하여 특정 틈새 문제(niche problems)를 해결하도록 설계된 다른 전용 솔루션의 결과를 재현하지 못한다.

구면 조화함수 기반은 고주파 view-dependent 효과나 specular 효과 모델링에 한계가 남아 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능의 핵심 메커니즘

VDGS의 일반화 성능이 향상되는 핵심 원리는 다음과 같다:

1. **시점 의존 불투명도 조절:** 불투명도의 변화는 간접적으로 색상도 변화시킨다. 다른 Gaussian들이 노출되어 렌더에 보이게 되기 때문이다. 이로 인해 단일 신경망으로도 색상과 불투명도 양쪽의 view-dependent 효과를 모델링할 수 있다.

2. **NeRF-GS 하이브리드의 장점:** 물리적 텍스처/조명 모델을 사용하지 않으므로 전용 도구가 아닌 **범용 솔루션**으로서 많은 응용에 사용할 수 있다. GS 컴포넌트의 색상과 불투명도를 효과적으로 제어하여 조건화할 수 있으며, 이러한 접근법은 NeRF in the Wild, 동적 장면, 생성 모델 등 다양한 응용에 활용 가능하다.

3. **다양한 데이터셋에서의 일관된 성능:** 합성 데이터(NeRF Synthetic), 대규모 실세계 장면(Tanks and Temples, Mip-NeRF 360, Deep Blending), 반사 객체(Shiny Blender) 등 다양한 유형에서 검증되었다.

### 3.2 일반화 성능 향상을 위한 확장 가능성

- **조건부 생성 모델과의 결합:** NeRF 방식 신경망으로 GS를 조건화하는 VDGS의 패러다임은 다양한 생성적 응용(동적 장면, 텍스트 기반 3D 생성 등)으로 확장 가능하다.
- **물리 기반 모델과의 결합:** 범용 모델의 한계를 극복하기 위해, 향후 물리 기반 BRDF/shading 모델과 결합하면 특정 도메인에서의 일반화 성능을 더욱 향상시킬 수 있다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구 영향

1. **하이브리드 패러다임 확립:** VDGS는 "명시적 표현(GS) + 암묵적 신경망(NeRF)"의 하이브리드 방식이 유효함을 입증하였다. 이는 후속 연구에서 유사한 결합 패러다임을 시도하는 기반이 되었다.

2. **후속 연구들의 영감원:**
   - VoD-3DGS (Nowak et al., 2025)는 학습 가능한 대칭 행렬을, VDGS (Malarz et al., 2025)는 MLP 변조를 통해 불투명도에 view-dependence를 확장하였다.
   - GENIE는 NeRF의 사실적 렌더링 품질과 GS의 편집 가능한 구조적 표현을 결합하여, 기하학 기반 편집과 신경 렌더링 간의 격차를 해소하는 하이브리드 모델이다.
   - NeRF-GS는 NeRF와 3DGS를 공동 최적화하는 프레임워크로, 잔차 벡터 최적화를 통해 3DGS의 개인화 능력을 향상시킨다.

3. **조건화(Conditioning) 가능성 확장:** VDGS는 GS가 NeRF 기반 방식으로 신경망에 의해 조건화될 수 있음을 보여주었다. 이는 텍스트 기반 3D 생성, 아바타 애니메이션 등 다운스트림 태스크에 중요한 시사점을 갖는다.

### 4.2 향후 연구 시 고려할 점

| 고려사항 | 상세 설명 |
|----------|-----------|
| **고주파 효과 모델링** | SH 기반 색상의 한계를 극복하기 위해 양자 임베딩, 해시 인코딩 등 대안적 방향 인코딩 탐색 필요 |
| **물리 기반 렌더링 통합** | 그림자·반사에 대한 일반적 해결을 넘어, BRDF 등 물리 기반 모델과의 통합 검토 |
| **확장성(Scalability)** | 대규모/복잡한 장면에서의 계산 효율성 및 Gaussian 수 관리 전략 |
| **동적 장면 확장** | 시간적 변화를 모델링하는 4D Gaussian Splatting과의 결합 가능성 |
| **압축 및 효율화** | 실시간 모바일/VR 응용을 위한 모델 압축 기법 연구 |
| **Few-shot/Sparse View** | 소수 뷰에서의 견고한 복원을 위한 정규화 전략 연구 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 특징 | VDGS와의 비교 |
|------|------|-----------|---------------|
| **NeRF** (Mildenhall et al.) | 2020 | MLP 기반 암묵적 표현, 볼류메트릭 렌더링 | VDGS의 NeRF 컴포넌트의 원형; 느린 학습/추론 |
| **Plenoxels** (Fridovich-Keil et al.) | 2022 | 신경망 없는 radiance fields | GS 이전의 명시적 표현; VDGS는 신경망 추가 |
| **Instant-NGP** (Müller et al.) | 2022 | 해시 인코딩으로 NeRF 가속 | NeRF 가속 접근; VDGS는 GS+NeRF 하이브리드 |
| **3DGS** (Kerbl et al.) | 2023 | Gaussian 기반 실시간 렌더링 | VDGS의 기반 모델; 반사/투명성 처리 미흡 |
| **Ref-NeRF** (Verbin et al.) | 2022 | 반사 방향 파라미터화 | 반사 전용 설계; VDGS는 범용적 |
| **GaussianShader** (Jiang et al.) | 2024 | GS에 shading function 통합, BRDF 기반 | 물리 기반 반사 모델링; VDGS보다 반사에 특화 |
| **Mip-Splatting** (Yu et al.) | 2024 | 앨리어싱 제거를 위한 3DGS 개선 | 앨리어싱에 집중; VDGS는 view-dependent opacity에 집중 |
| **VoD-3DGS** (Nowak et al.) | 2025 | 학습 가능 대칭 행렬로 불투명도 view-dependence | VDGS의 MLP 대신 행렬 기반; 유사한 동기 |
| **QuantumGS** | 2026 | 양자 임베딩으로 고주파 효과 모델링 | QuantumGS는 Hyper-Quantum 파이프라인으로 PSNR 33.98, SSIM 0.970을 달성하여 3DGS 및 VDGS를 포함한 모든 기준선을 능가한다. |

### 핵심 트렌드 분석

최근 신경 렌더링 연구는 NeRF와 같은 암묵적 좌표 기반 표현에서 명시적 점 기반 방법으로 이동하고 있다. 3DGS는 비등방성 3D Gaussian을 모델링하여 복잡한 장면의 실시간 렌더링을 달성하지만, 저차 구면 조화함수 의존으로 인해 날카로운 specular 하이라이트, 광택 표면, 투명도 변화 등 고주파 view-dependent 효과에 어려움을 겪는다.

3DGS 도입 이후 3D 장면 표현의 판도가 급격히 변화하였으며, 효율성, 확장성, 실세계 적용 가능성을 향상시키는 광범위한 후속 연구가 이어지고 있다.

---

## 참고자료 출처

1. **[arXiv:2312.13729]** Malarz et al., "Gaussian Splatting with NeRF-based Color and Opacity", 2023 — https://arxiv.org/abs/2312.13729
2. **[CVIU 2024]** 동일 논문의 저널 버전, *Computer Vision and Image Understanding* — https://www.sciencedirect.com/science/article/abs/pii/S1077314224003540
3. **[arXiv HTML v4/v5]** 논문 전체 HTML 버전 — https://arxiv.org/html/2312.13729v5
4. **[GitHub]** VDGS 공식 코드 — https://github.com/gmum/ViewingDirectionGaussianSplatting
5. **[Semantic Scholar]** 논문 인용 및 관련 연구 — https://www.semanticscholar.org/paper/f9165d51b39626ec351f58cc66d13c26c9fcf481
6. **[ResearchGate]** 논문 전문 — https://www.researchgate.net/publication/376990011
7. **[QuantumGS, arXiv:2602.05047]** Quantum Encoding Framework for Gaussian Splatting, 2026 — https://arxiv.org/html/2602.05047
8. **[GaussianShader, CVPR 2024]** Jiang et al. — https://openaccess.thecvf.com/content/CVPR2024/papers/Jiang_GaussianShader_CVPR_2024_paper.pdf
9. **[A Survey on 3D Gaussian Splatting, arXiv:2401.03890]** Chen et al. — https://arxiv.org/abs/2401.03890
10. **[The Impact and Outlook of 3DGS]** — https://arxiv.org/html/2510.26694v1
11. **[Wikipedia: Gaussian splatting]** — https://en.wikipedia.org/wiki/Gaussian_splatting
12. **[ACM DL]** Gaussian Splatting with NeRF-based color and opacity — https://dl.acm.org/doi/10.1016/j.cviu.2024.104273

> ⚠️ **참고:** 본 분석에서 제시한 수식의 세부 표기(예: opacity multiply 연산의 정확한 활성화 함수 선택)는 논문 원문의 설명을 바탕으로 구성한 것이며, 정확한 수식 번호와 표기는 원문 PDF를 직접 확인하시기 바랍니다. ablation study의 구체적 수치(PSNR/SSIM/LPIPS)는 논문의 Table 1~6을 참고해 주세요.
