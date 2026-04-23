# Geometry Transfer for Stylizing Radiance Fields

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

기존 3D 스타일 전이(Style Transfer) 연구들은 **색상과 텍스처** 전이에만 집중하고, 기하학적(Geometric) 측면을 간과해왔다. 본 논문은 스타일의 정체성을 정의하는 데 있어 **형태(Shape)와 기하학적 패턴**이 본질적으로 중요하다는 점을 강조하며, 깊이 맵(Depth Map)을 스타일 가이드로 활용하여 Radiance Fields의 **기하 구조 자체를 스타일화**하는 최초의 방법론을 제안한다.

### 주요 기여 (4가지)

| 기여 | 설명 |
|------|------|
| **Geometry Transfer** | 깊이 맵(Depth Map)을 스타일 가이드로 사용하여 Radiance Fields의 기하 구조를 스타일화하는 최초의 방법 |
| **Deformation Fields의 새로운 활용** | 형태와 외관(Appearance)의 정합성 있는 스타일화를 위한 변형 그리드 $\mathcal{G}_\Delta$ 도입 |
| **RGB-D 스타일화 기법** | Geometry-aware Matching, Patch-wise Optimization, Perspective Style Augmentation 등 새로운 기법 제안 |
| **Panoptic Radiance Fields와의 통합** | 부분적(Partial) 스타일화를 위한 Panoptic Lifting과의 호환성 제공 |

---

## 2. 상세 기술 설명

### 2.1 해결하고자 하는 문제

#### 문제 1: 기하학적 스타일의 부재
기존 방법들(SNeRF, ARF, Ref-NPR)은 **색상, 붓터치, 텍스처**만 전이하며, 스타일 이미지의 기하학적 형태는 전혀 반영하지 못한다.

- 예시: 고사리(fern) 잎을 스타일화할 때, 잎이 좁고 날카로운 형태를 유지한 채로는 스타일 이미지의 패턴을 완전히 표현할 수 없음

#### 문제 2: Radiance Fields에서의 외관-형태 불일치
Radiance Fields의 고유한 구조적 분리( $\mathcal{G}\_c$: 색상 그리드, $\mathcal{G}_\sigma$: 밀도 그리드)로 인해, 밀도를 직접 최적화하면 **색상 필드가 업데이트된 형태와 불일치**하는 문제가 발생한다.

$$\text{문제: } \mathcal{G}_\sigma \text{가 변화하더라도 } \mathcal{G}_c \text{는 원래 표면 위치와 정렬되어 있음}$$

→ 전경 객체의 새로운 위치에 배경 색상이 할당되는 아티팩트 발생

---

### 2.2 제안하는 방법과 수식

#### Step 1: 사전 학습 (TensoRF 기반 재구성)

$$\text{Input: } \{I_i\}_{i=1}^N, \{p_i\}_{i=1}^N \rightarrow \text{학습: } \mathcal{G}_c, \mathcal{G}_\sigma$$

TensoRF를 사용하여 색상 그리드 $\mathcal{G}\_c$와 밀도 그리드 $\mathcal{G}_\sigma$를 저랭크(Low-rank) 분해 형태로 학습한다.

---

#### Step 2: Geometry Transfer (깊이 맵 기반 스타일 손실)

기존 ARF의 RGB 스타일 손실을 기반으로, 깊이 맵 $\mathcal{S}_D$를 스타일 가이드로 대체한다.

**기본 스타일 손실 (ARF 방식, Eq. 1):**

$$L_{style} = \frac{1}{N} \sum_{i,j} \min_{i',j'} D(F^{rgb}_{\mathcal{I}}(i,j), F^{rgb}_{\mathcal{S}}(i',j'))$$

여기서 $D(\cdot, \cdot)$는 두 정규화된 특징 벡터 사이의 **코사인 거리**이다.

$$D(\mathbf{u}, \mathbf{v}) = 1 - \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$

깊이 스타일화 시, RGB 이미지 대신 렌더링된 깊이 맵 $D_{p_i}$와 스타일 깊이 $\mathcal{S}\_D$ 사이의 손실로 $\mathcal{G}_\sigma$를 최적화한다.

---

#### Step 3: Deformation Fields 모델링

**핵심 아이디어:** $\mathcal{G}\_\sigma$를 고정한 채, 추가적인 변형 그리드 $\mathcal{G}_\Delta$를 최적화하여 각 3D 점 $\mathbf{x}_i$에 대한 변위 벡터 $\Delta \mathbf{x}_i \in \mathbb{R}^3$를 예측한다.

$$\mathbf{x}_i \rightarrow \mathbf{x}_i + \Delta \mathbf{x}_i \quad \text{(정준 공간으로의 매핑)}$$

**렌더링 과정:**

$$\sigma_i = \mathcal{G}_\sigma(\mathbf{x}_i), \quad c_i = \mathcal{G}_c(\mathbf{x}_i + \Delta \mathbf{x}_i)$$

이를 통해 형태가 변형되더라도 **원래 표면의 색상 정보를 일관성 있게 참조**할 수 있다. 즉, 기하 변형 후에도 색상과 형태가 함께 정합성 있게 업데이트된다.

---

#### Step 4: RGB-D 스타일화

**4-1. Geometry-aware Nearest Matching (Eq. 2, 3)**

RGB와 깊이 두 모달리티를 동시에 고려하여 최근접 이웃을 탐색한다:

$$j = \arg\min_{i'} D\left([F^{rgb}_{\mathcal{I}}(i), F^{D}_{\mathcal{I}}(i)], [F^{rgb}_{\mathcal{S}}(i'), F^{D}_{\mathcal{S}}(i')]\right)$$

최적 매칭 쌍 $j$를 찾은 후, 각 모달리티별로 손실을 계산한다:

$$L(i) = D(F^{rgb}_{\mathcal{I}}(i), F^{rgb}_{\mathcal{S}}(j)) + D(F^{D}_{\mathcal{I}}(i), F^{D}_{\mathcal{S}}(j))$$

$$L_{style} = \frac{1}{N} \sum_i L(i)$$

**4-2. Patch-wise Optimization (Eq. 4, 5)**

개별 픽셀 기반 매칭의 한계를 극복하기 위해, $k \times k$ 패치 단위로 매칭을 수행한다:

$$L_{\mathcal{SP}} = \frac{1}{|\mathcal{P}_{\mathcal{I}}|} \sum_i \min_j D^{\mathcal{P}}(\mathcal{P}^i_{\mathcal{I}}, \mathcal{P}^j_{\mathcal{S}})$$

패치 내 코사인 거리의 합:

$$D^{\mathcal{P}}(\mathcal{P}_1, \mathcal{P}_2) = \sum_i^{k^2} D(F^i_1, F^i_2)$$

팽창률(Dilation rate) $r$을 하이퍼파라미터로 사용하여 계산량 증가 없이 수용 영역(Receptive Field)을 확장한다.

**4-3. Perspective Style Augmentation**

장면의 깊이에 따라 스타일 패턴 크기를 다르게 적용한다:

$$s_i = \frac{C_1}{C_i}$$

여기서 $C_i$는 $i$번째 깊이 빈(Bin)의 중심값이다. 가까운 표면에는 큰 패턴, 먼 표면에는 작은 패턴을 적용하여 **원근감**을 향상시킨다.

---

### 2.3 모델 구조

```
입력: 다시점 이미지 {Iᵢ} + RGB-D 스타일 쌍 (S_rgb, S_D)
         ↓
[Phase 1: 재구성]
TensoRF 학습 → G_c (색상 그리드), G_σ (밀도 그리드)
         ↓
[Phase 2: RGB-D 스타일화]
  ┌─────────────────────────────────────┐
  │  G_σ: 고정 (변경 없음)              │
  │  G_Δ: 변형 그리드 최적화           │
  │  G_c: 외관 그리드 최적화           │
  └─────────────────────────────────────┘
         ↓ (각 반복마다)
  1. 뷰포인트 pᵢ 선택 → 렌더링: Î_{pᵢ}, D_{pᵢ}
  2. VGG 특징 추출: F^rgb, F^D
  3. Geometry-aware Matching
  4. Patch-wise 손실 계산
  5. Perspective Augmentation 적용
  6. G_c, G_Δ 업데이트
         ↓
출력: 스타일화된 3D 장면 (형태 + 외관 모두 스타일화)
```

---

### 2.4 성능 향상

#### 정량적 비교 (SIFID ↓, 낮을수록 좋음)

| 방법 | trex RGB | trex Gray | trex Depth | fern RGB | fern Gray | fern Depth |
|------|----------|-----------|------------|----------|-----------|------------|
| SNeRF | 1.62 | 0.81 | 0.59 | 1.32 | 0.64 | 0.40 |
| ARF | 1.54 | 0.64 | 0.51 | 1.11 | 0.48 | 0.36 |
| Ref-NPR | 1.59 | 0.72 | 0.61 | 1.75 | 0.79 | 0.41 |
| **Ours** | **1.43** | **0.58** | **0.44** | **0.81** | **0.37** | **0.28** |

#### 사용자 연구 (평균 순위 ↓)

| 방법 | Ours | Ref-NPR | ARF | SNeRF |
|------|------|---------|-----|-------|
| 평균 순위 | **1.55** | 3.17 | 2.58 | 2.70 |

22명 참가자, 12개 장면, 264개 응답 중 162회(61.4%) 1위 선정

---

### 2.5 한계점

1. **TensoRF 기반 한계:** 360° 무경계(Unbounded) 장면 처리에 제약이 있음
2. **단일 스타일 이미지의 ill-posed 문제:** 하나의 스타일 이미지로 다양한 시점에서 일관된 기하 패턴을 전이하는 것은 근본적으로 과소 결정(Under-determined) 문제임
3. **최적화 기반 방식의 속도:** 매 장면마다 재최적화가 필요하여 실시간 적용에 한계

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화의 구조적 강점

#### (1) Panoptic Radiance Fields와의 통합
논문은 방법을 Panoptic Lifting에 통합하여 **부분 스타일화(Partial Stylization)**를 구현한다:

$$\mathbf{x}_i \rightarrow (c_i, \sigma_i, \kappa_i(k), \pi_i(j))$$

변형 필드가 정준 공간(Canonical Space)에서의 샘플링을 유지하므로, 스타일화 후에도 **의미론적 예측이 새로운 형태에 자동으로 적응**한다. 이는 일반화 가능성의 핵심 메커니즘이다.

#### (2) Zero-shot 깊이 추정 활용
ZoeDepth(Bhat et al., 2023)를 활용하여 **임의의 RGB 스타일 이미지에서 자동으로 깊이 맵 생성**이 가능하다. 이를 통해 사용자가 깊이 정보 없이 RGB 이미지만 제공해도 RGB-D 스타일화가 가능하다.

#### (3) Perspective Style Augmentation의 도메인 불변성
원근 규칙 기반의 스케일 조정:

$$s_i = \frac{C_1}{C_i}$$

이 수식은 물리적 원근 법칙에 기반하여 **다양한 카메라 구성과 장면 규모에 자동으로 적응**한다.

### 3.2 일반화 향상을 위한 잠재적 방향

#### 방향 1: 다시점 스타일 가이드로의 확장
현재는 단일 스타일 이미지에 의존하지만, **다시점 스타일 가이드(Multi-view Style Guides)** 또는 **3D 스타일 가이드**를 도입하면 360° 장면에서의 일관성 있는 스타일화가 가능해질 것이다.

#### 방향 2: Feed-forward 네트워크와의 결합
현재 최적화 기반(Per-scene Optimization) 방식을 **하이퍼네트워크(Hypernetwork)** 또는 **피처 변환(Feature Transformation)** 방식과 결합하면 임의의 장면에 대한 제로샷 일반화가 가능해진다.

$$\theta_{stylized} = \mathcal{H}(\theta_{content}, \phi_{style})$$

#### 방향 3: 3D Gaussian Splatting으로의 확장
TensoRF 대신 **3D Gaussian Splatting(3DGS)**을 기반으로 하면 더 빠른 렌더링과 넓은 장면 처리가 가능하며, 변형 필드와의 통합도 기하학적으로 직관적이다.

#### 방향 4: 생성 모델과의 결합
**Diffusion Model**이나 **CLIP** 기반 텍스트 가이드를 결합하면, 단일 스타일 이미지 없이도 텍스트로부터 기하 스타일을 추출하여 일반화된 스타일화가 가능해질 수 있다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 연도 | 주요 기법 | 기하 변형 | 스타일 가이드 | 한계 |
|------|------|-----------|-----------|----------------|------|
| **NeRF** (Mildenhall et al.) | ECCV 2020 | 체적 렌더링 기반 장면 표현 | ✗ | - | 스타일화 불가 |
| **SNeRF** (Nguyen-Phuoc et al.) | TOG 2022 | 렌더링-2D스타일화 교차 반복 | △ (제한적) | RGB | 기하 스타일 미반영 |
| **ARF** (Zhang et al.) | ECCV 2022 | 최근접 이웃 매칭 손실 | ✗ | RGB | 기하 변형 없음 |
| **StyleRF** (Liu et al.) | CVPR 2023 | 제로샷, 피처 변환 | ✗ | RGB | 기하 미변형 |
| **Ref-NPR** (Zhang et al.) | CVPR 2023 | 참조 스타일화 시점 활용 | ✗ | RGB+참조뷰 | 기하 미변형 |
| **Locally Stylized NeRF** (Pang et al.) | ICCV 2023 | 해시 인코딩 + 의미 매칭 | ✗ | RGB | 기하 미변형 |
| **Instruct-NeRF2NeRF** (Haque et al.) | ICCV 2023 | 확산 모델 기반 편집 | △ | 텍스트 | 세밀 제어 어려움 |
| **Geo-SRF (Ours)** | arXiv 2024 | 깊이 맵 스타일 가이드 + 변형 필드 | **✓** | **RGB-D** | 360° 장면, 속도 |

### 핵심 차별점 분석

```
기존 방법들의 스타일화 공간:
  색상 공간 ████████████████████ (100%)
  기하 공간 ░░░░░░░░░░░░░░░░░░░░ (0%)

본 논문의 스타일화 공간:
  색상 공간 ████████████ (60%)
  기하 공간 ████████ (40%)
```

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5.1 연구에 미치는 영향

#### 영향 1: 3D 스타일 전이의 패러다임 전환
본 논문은 **"스타일 = 색상/텍스처"라는 기존 관념을 "스타일 = 색상 + 기하"로 확장**했다. 이는 3D 콘텐츠 생성 연구에서 기하학적 스타일을 독립적인 연구 변수로 다루는 새로운 방향을 제시한다.

#### 영향 2: 변형 필드의 스타일 전이 응용
D-NeRF에서 시간적 변형(Temporal Deformation)에 사용되던 **변형 필드를 스타일 전이에 창의적으로 재활용**한 아이디어는, 다른 신경 렌더링 기반 편집 작업(예: 형태 편집, 표정 전이)에도 영향을 미칠 것이다.

#### 영향 3: RGB-D 다중 모달 스타일화
RGB-D 쌍을 스타일 가이드로 사용하는 프레임워크는 **깊이 카메라(LiDAR, RGB-D 센서)와 결합한 실용적 응용**으로 확장될 가능성이 높다. AR/VR 콘텐츠 생성 파이프라인에 직접적으로 기여할 수 있다.

### 5.2 앞으로 연구 시 고려할 점

#### 고려사항 1: 표현 기반의 일반화
TensoRF 대신 **3D Gaussian Splatting** 또는 **Instant-NGP** 등 더 빠른 표현 방식을 기반으로 변형 필드를 구현하면 속도와 품질을 동시에 향상시킬 수 있다.

$$\Delta \mathbf{x}_i = f_\theta(\mathbf{x}_i, \phi_{style}) \quad \text{(스타일 조건부 변형 네트워크)}$$

#### 고려사항 2: 깊이 맵의 품질 의존성
ZoeDepth로 추정된 깊이의 품질이 기하 스타일화에 직접적인 영향을 미친다. **단안 깊이 추정의 불확실성**을 고려한 강건한 손실 함수 설계가 필요하다.

#### 고려사항 3: 멀티뷰 일관성 강화
단일 스타일 이미지로 다시점 일관성을 보장하는 것은 근본적 한계가 있다. **3D 스타일 프리미티브(Style Primitives)**나 **구형 조화 함수(Spherical Harmonics)** 기반의 방향성 스타일 표현이 해결책이 될 수 있다.

#### 고려사항 4: 평가 지표의 한계
SIFID는 스타일 유사도를 측정하지만, **인간의 지각적 품질**이나 **콘텐츠 보존도**와의 균형을 함께 평가하는 복합 지표 개발이 필요하다.

#### 고려사항 5: 대규모 및 동적 장면으로의 확장
현재는 정적 장면에만 적용 가능하다. **D-NeRF나 HyperNeRF** 등 동적 장면 표현과 결합하여 **시간에 따라 변하는 기하 스타일화**로의 확장이 향후 중요한 연구 방향이 될 것이다.

---

## 참고 자료

1. **Jung, H., Nam, S., Sarafianos, N., Yoo, S., Sorkine-Hornung, A., & Ranjan, R. (2024).** "Geometry Transfer for Stylizing Radiance Fields." *arXiv:2402.00863v3* — **본 논문 (첨부 PDF)**

2. **Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., & Ng, R. (2020).** "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis." *ECCV 2020.* (논문 내 참고문헌 [53])

3. **Chen, A., Xu, Z., Geiger, A., Yu, J., & Su, H. (2022).** "TensoRF: Tensorial Radiance Fields." *ECCV 2022.* (논문 내 참고문헌 [6])

4. **Zhang, K., Kolkin, N., Bi, S., Luan, F., Xu, Z., Shechtman, E., & Snavely, N. (2022).** "ARF: Artistic Radiance Fields." *ECCV 2022.* (논문 내 참고문헌 [83])

5. **Nguyen-Phuoc, T., Liu, F., & Xiao, L. (2022).** "SNeRF: Stylized Neural Implicit Representations for 3D Scenes." *ACM TOG 41.* (논문 내 참고문헌 [55])

6. **Zhang, Y., He, Z., Xing, J., Yao, X., & Jia, J. (2023).** "Ref-NPR: Reference-based Non-Photorealistic Radiance Fields for Controllable Scene Stylization." *CVPR 2023.* (논문 내 참고문헌 [86])

7. **Liu, K., Zhan, F., Chen, Y., Zhang, J., Yu, Y., El Saddik, A., Lu, S., & Xing, E. P. (2023).** "StyleRF: Zero-shot 3D Style Transfer of Neural Radiance Fields." *CVPR 2023.* (논문 내 참고문헌 [45])

8. **Pumarola, A., Corona, E., Pons-Moll, G., & Moreno-Noguer, F. (2021).** "D-NeRF: Neural Radiance Fields for Dynamic Scenes." *CVPR 2021.* (논문 내 참고문헌 [61])

9. **Bhat, S. F., Birkl, R., Wofk, D., Wonka, P., & Müller, M. (2023).** "ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth." *arXiv:2302.12288.* (논문 내 참고문헌 [3])

10. **Siddiqui, Y., Porzi, L., Rota Buló, S., Müller, N., Nießner, M., Dai, A., & Kontschieder, P. (2023).** "Panoptic Lifting for 3D Scene Understanding with Neural Fields." *CVPR 2023.* (논문 내 참고문헌 [68])
