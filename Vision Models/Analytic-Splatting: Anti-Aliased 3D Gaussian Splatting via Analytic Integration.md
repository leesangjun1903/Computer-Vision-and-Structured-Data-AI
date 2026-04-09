# Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic Integration

---

## 1. 핵심 주장과 주요 기여 (간결한 요약)

### 핵심 주장

**3DGS(3D Gaussianf Splatting)는 각 픽셀을 면적이 아닌 단일 점(point)으로 취급하기 때문에 앨리어싱(aliasing)이 발생한다.** 이를 해결하기 위해, 픽셀 창(window) 영역에 대한 2D 가우시안 신호의 적분을 해석적으로(analytically) 근사하는 방법을 제안한다.

### 주요 기여

1. **알리어싱의 원인 재분석**: 3DGS에서 앨리어싱이 발생하는 이유를 신호 창 응답(window response) 관점에서 분석
2. **해석적 근사법 도출**: 1D 가우시안 신호의 CDF를 조건부 로지스틱 함수로 근사하고, 이를 2D 픽셀 셰이딩에 확장한 **Analytic-Splatting** 제안
3. **성능 우위 입증**: 다양한 벤치마크(Multi-scale Blender, Mip-NeRF 360)에서 앤티앨리어싱 및 디테일 보존 측면에서 SOTA 달성

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

**3DGS의 앨리어싱 문제**는 픽셀 처리 방식에서 기인한다.

기존 3DGS의 픽셀 셰이딩:

$$g^{\text{2D}}(\boldsymbol{u} | \hat{\boldsymbol{\mu}}, \hat{\boldsymbol{\Sigma}}) = \exp\left(-\frac{a}{2}\hat{u}_x^2 - \frac{c}{2}\hat{u}_y^2 - b\hat{u}_x\hat{u}_y\right)$$

여기서 $\hat{u}_x = u_x - \hat{\mu}_x$, $\hat{u}_y = u_y - \hat{\mu}_y$

이 식은 픽셀 중심점의 가우시안 값만을 강도 응답으로 사용한다. 결과적으로:

- **해상도 변화 시 픽셀 풋프린트(footprint)** 변화에 둔감
- 줌인/아웃 시 블러링 또는 재기(jaggies) 아티팩트 발생
- 샘플링 대역폭이 제한되어 나이퀴스트 기준 이하로 떨어질 때 앨리어싱 발생

**기존 접근법의 한계:**
- **슈퍼 샘플링(3DGS-SS)**: 계산 부담이 크게 증가
- **Mip-Splatting**: 픽셀 창을 가우시안 저역통과 필터로 근사 → 고주파 성분(세부 디테일) 억제로 과도한 스무딩 발생

$$\mathcal{I}_g \approx g \circledast g_w \quad (\text{Mip-Splatting의 근사})$$

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: 1D 가우시안 신호의 창 적분

창 영역 $[u - \frac{1}{2}, u + \frac{1}{2}]$에서의 적분:

$$\mathcal{I}_g(u) = \int^{u + \frac{1}{2}}_{u - \frac{1}{2}} g(x)\,dx = G\!\left(u + \frac{1}{2}\right) - G\!\left(u - \frac{1}{2}\right)$$

여기서 $G(x)$는 표준 가우시안 분포의 누적 분포 함수(CDF):

$$G(x) = \int^x_{-\infty} g(x)\,dx, \quad g(x) = \frac{1}{\sqrt{2\pi}}\exp\!\left(-\frac{x^2}{2}\right)$$

그런데 $G(x)$는 오차함수(erf)로 닫힌 형태(closed-form)가 아니므로 해석적 근사가 필요하다.

#### Step 2: 조건부 로지스틱 함수로 CDF 근사 (Definition 1)

$$S(x) = \frac{1}{1 + \exp(-1.6 \cdot x - 0.07 \cdot x^3)}$$

이 함수는 표준편차 $\sigma = 1$인 가우시안 CDF $G(x)$의 해석적 근사이며, 미분 가능하다.

표준편차 $\sigma \neq 1$인 경우:

$$S_\sigma(x) = S\!\left(\frac{x}{\sigma}\right) = \frac{1}{1 + \exp\!\left(-1.6 \cdot \frac{x}{\sigma} - 0.07 \cdot \left(\frac{x}{\sigma}\right)^3\right)}$$

따라서 1D 창 적분의 근사:

$$\mathcal{I}_g(u) \approx S_\sigma\!\left(u + \frac{1}{2}\right) - S_\sigma\!\left(u - \frac{1}{2}\right)$$

#### Step 3: 2D 픽셀 셰이딩으로 확장

픽셀 $\boldsymbol{u} = [u_x, u_y]^\top$에 대한 2D 가우시안 적분:

$$\mathcal{I}_g^{\text{2D}}(\boldsymbol{u}) = \int^{u_x + \frac{1}{2}}_{u_x - \frac{1}{2}} \int^{u_y + \frac{1}{2}}_{u_y - \frac{1}{2}} \exp\!\left(-\frac{a}{2}(x-\hat{\mu}_x)^2 - \frac{c}{2}(y-\hat{\mu}_y)^2 - \underbrace{b(x-\hat{\mu}_x)(y-\hat{\mu}_y)}_{\text{correlation term}}\right) dx\,dy$$

상관 항(correlation term)을 제거하기 위해 **공분산 행렬 $\hat{\boldsymbol{\Sigma}}$의 고유값 분해(eigendecomposition)** 수행:

$$\lambda_1 = \frac{\text{Tr}(\hat{\boldsymbol{\Sigma}}) + \sqrt{\text{Tr}(\hat{\boldsymbol{\Sigma}})^2 - 4\det(\hat{\boldsymbol{\Sigma}})}}{2}, \quad \lambda_2 = \frac{\text{Tr}(\hat{\boldsymbol{\Sigma}}) - \sqrt{\text{Tr}(\hat{\boldsymbol{\Sigma}})^2 - 4\det(\hat{\boldsymbol{\Sigma}})}}{2}$$

고유벡터 $\{\boldsymbol{v}_1, \boldsymbol{v}_2\}$로 구성된 새 좌표계에서 픽셀 중심 변환:

$$\tilde{\boldsymbol{u}} = \begin{bmatrix}\tilde{u}_x \\ \tilde{u}_y\end{bmatrix} = \begin{bmatrix}-\, \boldsymbol{v}_1\, -\\ -\, \boldsymbol{v}_2\, -\end{bmatrix}(\boldsymbol{u} - \hat{\boldsymbol{\mu}})$$

대각화 후 2D 가우시안을 두 독립적인 1D 가우시안의 곱으로 표현:

$$g^{\text{2D}}(\boldsymbol{u}) = \exp\!\left(-\frac{1}{2\lambda_1}\tilde{u}_x^2\right)\exp\!\left(-\frac{1}{2\lambda_2}\tilde{u}_y^2\right)$$

적분 영역을 회전하여 2D 적분을 두 1D 적분의 곱으로 근사:

$$\mathcal{I}_g^{\text{2D}}(\boldsymbol{u}) \approx 2\pi\sigma_1\sigma_2 \underbrace{\left[S_{\sigma_1}\!\left(\tilde{u}_x + \frac{1}{2}\right) - S_{\sigma_1}\!\left(\tilde{u}_x - \frac{1}{2}\right)\right]}_{\mathcal{I}_{\sigma_1}} \underbrace{\left[S_{\sigma_2}\!\left(\tilde{u}_y + \frac{1}{2}\right) - S_{\sigma_2}\!\left(\tilde{u}_y - \frac{1}{2}\right)\right]}_{\mathcal{I}_{\sigma_2}}$$

여기서 $\sigma_1 = \sqrt{\lambda_1}$, $\sigma_2 = \sqrt{\lambda_2}$

#### Step 4: 최종 볼륨 렌더링

$$\boldsymbol{C}(\boldsymbol{u}) = \sum_{i \in N} T_i \mathcal{I}_{g\text{-}i}^{\text{2D}}(\boldsymbol{u} | \hat{\boldsymbol{\mu}}_i, \hat{\boldsymbol{\Sigma}}_i)\alpha_i \boldsymbol{c}_i$$

$$T_i = \prod^{i-1}_{j=1}\left(1 - \mathcal{I}_{g\text{-}j}^{\text{2D}}(\boldsymbol{u} | \hat{\boldsymbol{\mu}}_j, \hat{\boldsymbol{\Sigma}}_j)\alpha_j\right)$$

기존 3DGS의 $g^{\text{2D}}$를 $\mathcal{I}_g^{\text{2D}}$로 교체함으로써 투과율(transmittance) 계산에도 픽셀 풋프린트 변화가 반영된다.

### 2.3 모델 구조

Analytic-Splatting은 3DGS 프레임워크 위에 **셰이딩 모듈**을 교체하는 방식으로 구현된다.

```
[3DGS 기반 구조]
  ↓
[3D Gaussian 파라미터 학습]
  - 위치 μ, 공분산 Σ, 불투명도 α, 구면 조화 색상 c
  ↓
[2D 투영] (기존 3DGS와 동일)
  - μ̂ = KT[μ,1]ᵀ, Σ̂ = JTΣTᵀJᵀ
  ↓
[Analytic Shading Module] ← 핵심 변경 부분
  1. Σ̂ 고유값 분해 → {λ₁, λ₂}, {v₁, v₂}
  2. 픽셀 좌표 변환 → ũ
  3. 조건부 로지스틱 함수로 CDF 근사
  4. 2D 창 적분 계산 → I²ᴰ_g
  ↓
[Volume Rendering] (I²ᴰ_g 기반 transmittance 계산)
  ↓
[최종 픽셀 색상 C(u)]
```

**CUDA 커스텀 확장**으로 구현되며, 역전파(backward)는 체인 룰(chain rule)을 이용하여 $\hat{\boldsymbol{\mu}}$, $\hat{\boldsymbol{\Sigma}}$에 대한 그라디언트를 유도한다.

### 2.4 성능 향상

#### Multi-scale Blender Synthetic Dataset (MTMT)

| 방법 | PSNR (Avg.) ↑ | SSIM (Avg.) ↑ | LPIPS (Avg.) ↓ |
|------|--------------|--------------|----------------|
| 3DGS | 29.77 | 0.960 | 0.040 |
| Mip-Splatting | 34.56 | 0.979 | 0.019 |
| **Ours** | **35.03** | **0.979** | **0.018** |

#### Multi-scale Mip-NeRF 360 Dataset (MTMT)

| 방법 | PSNR (Avg.) ↑ | SSIM (Avg.) ↑ | LPIPS (Avg.) ↓ |
|------|--------------|--------------|----------------|
| 3DGS | 27.63 | 0.853 | 0.156 |
| Mip-Splatting | 29.12 | 0.883 | 0.134 |
| **Ours** | **29.51** | **0.887** | **0.123** |

- **Zip-NeRF**(실시간 렌더링 불가)에 비해서는 소폭 낮지만, **실시간 렌더링이 가능한 방법 중 최고 성능**
- 특히 **1/8 해상도**에서 3DGS 대비 큰 폭의 성능 향상 (PSNR: 27.98 → 36.00 on Blender)
- 단일 해상도 학습/테스트 설정(single-scale)에서도 Mip-Splatting 대비 동등하거나 우수한 성능

### 2.5 한계

1. **계산 오버헤드**: 더 많은 제곱근(root) 및 지수 연산을 포함하여 계산 부담 증가 → 프레임 레이트가 Mip-Splatting 대비 약 **10% 감소**
2. **적분 도메인 회전 오류**: 2D 픽셀 창을 고유벡터 방향으로 회전할 때 근사 오류가 추가로 발생 (회전각이 클수록 오류 증가, 단 여전히 다른 방법보다 우수)
3. **3D 가우시안 정규화 미포함**: Mip-Splatting과 달리 3D 스무딩 필터를 기본적으로 적용하지 않음 (단, 선택적으로 결합 가능)

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 해상도 일반화

Analytic-Splatting의 가장 핵심적인 일반화 향상은 **픽셀 풋프린트 변화에 대한 민감도**에 있다.

$$\mathcal{I}_g^{\text{2D}}(\boldsymbol{u}) = 2\pi\sigma_1\sigma_2\left[S_{\sigma_1}\!\left(\tilde{u}_x + \frac{1}{2}\right) - S_{\sigma_1}\!\left(\tilde{u}_x - \frac{1}{2}\right)\right]\left[S_{\sigma_2}\!\left(\tilde{u}_y + \frac{1}{2}\right) - S_{\sigma_2}\!\left(\tilde{u}_y - \frac{1}{2}\right)\right]$$

이 수식에서 $\sigma_1, \sigma_2$가 픽셀 풋프린트에 따라 자연스럽게 스케일링되므로, **다양한 해상도에서 일관된 응답**을 제공한다.

**구체적 근거:**
- **소해상도(저해상도, 풋프린트 큼)**: $\sigma_1, \sigma_2$가 크면 $\mathcal{I}_g^{\text{2D}}$가 픽셀 창 내 가우시안 신호 전체를 통합 → 블러링 방지
- **고해상도(풋프린트 작음)**: $\sigma_1, \sigma_2$가 작으면 세밀한 고주파 디테일 보존 → 저역통과 필터링의 과도한 스무딩 문제 없음

Mip-Splatting은 픽셀 창을 가우시안 저역통과 필터로 모델링하므로 고주파 신호를 억제하지만, Analytic-Splatting은 **고주파 성분도 손실 없이** 처리 가능하다.

### 3.2 슈퍼 해상도(Super-Resolution) 일반화

논문의 **슈퍼 해상도 실험**(2× Res., Mip-NeRF 360)에서:

| 방법 | PSNR Avg. | SSIM Avg. | LPIPS Avg. |
|------|-----------|-----------|------------|
| 3DGS | 25.95 | 0.747 | 0.358 |
| Mip-Splatting | 26.46 | 0.764 | 0.329 |
| **Ours** | **26.90** | **0.774** | **0.324** |

학습 해상도보다 **높은 해상도로 렌더링**하는 상황에서도 가장 우수한 성능을 보여, **단순 보간 이상의 고주파 디테일** 복원 능력을 입증한다.

### 3.3 다양한 장면 유형에서의 일반화

단일 해상도 학습/테스트(Single-Scale Training & Testing):
- **Mip-NeRF 360**: PSNR 27.58 (Mip-Splatting 27.57과 동등)
- **Tanks&Temples**: PSNR 23.84 (Mip-Splatting 23.78 대비 우세)
- **Deep Blending**: PSNR 29.75 (Mip-Splatting 29.69 대비 우세)

이는 Analytic-Splatting이 단순히 다중 해상도 시나리오에만 특화된 것이 아니라, **다양한 실내외 장면**에서도 일반화됨을 보여준다.

### 3.4 고주파 신호에 대한 강건성

근사 오류 분석에서 **표준편차 $\sigma$가 작을수록**(고주파 신호) 다른 방법들에 비해 Analytic-Splatting의 우위가 더 명확해진다:

$$\mathcal{E}_{\text{Int}}(x) = \left|\left(S\!\left(x+\frac{1}{2}\right) - S\!\left(x-\frac{1}{2}\right)\right) - \left(G\!\left(x+\frac{1}{2}\right) - G\!\left(x-\frac{1}{2}\right)\right)\right|$$

오류 스케일이 $10^{-4}$ 수준으로 매우 작으며, 이는 **얇은 구조물, 가는 선, 잎사귀 등 세밀한 장면 요소**에서 특히 중요한 일반화 능력이다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 NeRF 계열 (Backward Mapping)

| 연구 | 핵심 특징 | 앨리어싱 처리 | 실시간 여부 |
|------|----------|-------------|-----------|
| **NeRF** (Mildenhall et al., ECCV 2020) | 암시적 신경 표현, 레이 마칭 | ✗ | ✗ |
| **Mip-NeRF** (Barron et al., ICCV 2021) | 원뿔 캐스팅, 통합 위치 인코딩 | ✓ (사전 필터링) | ✗ |
| **Mip-NeRF 360** (Barron et al., CVPR 2022) | 비제한 장면, 비선형 파라미터화 | ✓ | ✗ |
| **Zip-NeRF** (Barron et al., ICCV 2023) | 해시 인코딩 + 다중 샘플링 | ✓ | ✗ |
| **Tri-MipRF** (Hu et al., ICCV 2023) | 삼중 평면 Mip 표현 | ✓ | △ |

**Mip-NeRF의 핵심 아이디어**: 픽셀을 원뿔(cone)로 모델링하고 각 샘플 구간을 원뿔 절두체(frustum)의 적분으로 근사:

$$\text{IPE}(\boldsymbol{\mu}, \boldsymbol{\Sigma}) = \int p(\boldsymbol{x}|\boldsymbol{\mu}, \boldsymbol{\Sigma})\gamma(\boldsymbol{x})d\boldsymbol{x}$$

이는 backward-mapping 기반으로 실시간 렌더링이 불가능하다는 근본적 한계가 있다.

### 4.2 3DGS 계열 (Forward Mapping)

| 연구 | 핵심 특징 | 앨리어싱 처리 | 디테일 보존 |
|------|----------|-------------|-----------|
| **3DGS** (Kerbl et al., ACM ToG 2023) | 명시적 3D 가우시안, 타일 기반 래스터화 | ✗ | △ |
| **Mip-Splatting** (Yu et al., 2023) | 2D/3D 가우시안 사전 필터링 (저역통과) | ✓ | △ (과도한 스무딩) |
| **Analytic-Splatting** (Liang et al., 2024) | 픽셀 창 적분의 해석적 근사 | ✓ | ✓ |

### 4.3 방법별 핵심 비교 요약

```
[앨리어싱 처리 전략 비교]

3DGS:         픽셀 중심 샘플링만 사용
              → 앨리어싱 발생
              → O(1) 연산

3DGS-SS:      2× 해상도 렌더링 후 평균 풀링
              → 부분적 앨리어싱 해소
              → O(4) 연산 (약 4배 느림)

Mip-Splatting: 2D 픽셀 창을 가우시안 저역통과 필터로 근사
              g ⊛ g_w (σ_w = 0.1)
              → 앨리어싱 해소 BUT 고주파 억제
              → 비슷한 속도

Analytic-Splatting (본 논문):
              픽셀 창 적분을 CDF로 정확히 근사
              I²ᴰ_g = 2πσ₁σ₂[S_σ₁(ũₓ+½)-S_σ₁(ũₓ-½)][S_σ₂(ũᵧ+½)-S_σ₂(ũᵧ-½)]
              → 앨리어싱 해소 AND 고주파 보존
              → Mip-Splatting 대비 약 10% 느림
```

### 4.4 실시간 렌더링 여부에 따른 분류

**실시간 가능(30+ FPS)**:
- 3DGS, Mip-Splatting, **Analytic-Splatting** (≈ Mip-Splatting의 90% 속도)

**실시간 불가**:
- NeRF, Mip-NeRF, Mip-NeRF 360, Zip-NeRF (특히 Zip-NeRF는 렌더링 시 슈퍼샘플링 추가 사용)

**수치 비교** (Multi-scale Mip-NeRF 360, MTMT):
- Zip-NeRF: PSNR 30.58 (최고, 비실시간)
- **Analytic-Splatting: PSNR 29.51** (실시간 중 최고)
- Mip-Splatting: PSNR 29.12
- 3DGS: PSNR 27.63

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5.1 앞으로의 연구에 미치는 영향

#### (1) 3DGS 파생 연구의 앤티앨리어싱 표준화

Analytic-Splatting은 3DGS 생태계에서 앤티앨리어싱을 처리하는 **원칙적(principled) 접근법**을 제시한다. 인간/아바타 모델링, 역렌더링, 표면 재구성 등 3DGS 기반 응용 연구들은 이 방법을 기반으로 삼아 고품질 렌더링을 달성할 수 있다.

#### (2) 신호 처리 관점의 도입 촉진

가우시안 신호의 창 적분을 해석적으로 근사한다는 아이디어는, 신경 렌더링 분야에서 **신호 처리 이론(CDF, 적분 근사)**을 더 적극적으로 활용하는 방향을 제시한다.

#### (3) 고주파 보존 렌더링 패러다임

기존의 저역통과 필터링 접근(Mip-Splatting 등)과 달리, 고주파 성분을 손실 없이 처리하면서도 앨리어싱을 제거하는 방법론의 가능성을 보여줌으로써, **슈퍼 해상도, HDR 렌더링, 디테일 보존 렌더링** 연구에 방향성을 제공한다.

#### (4) 실시간 앤티앨리어싱 렌더링의 실용화

Zip-NeRF 수준의 품질에 근접하면서도 실시간 렌더링이 가능하다는 점에서, **VR/AR, 게임, 실시간 시각화** 분야에서의 활용 가능성이 높다.

### 5.2 앞으로 연구 시 고려할 점

#### (1) 계산 효율성 최적화

현재 Mip-Splatting 대비 약 10% 프레임 레이트 감소가 있다. 향후 연구에서:
- **CUDA 커널 최적화**를 통해 고유값 분해 및 로지스틱 함수 계산 병렬화
- **근사 정밀도와 속도 간의 트레이드오프** 조절 가능한 파라미터 도입

#### (2) 적분 도메인 회전 오류 최소화

픽셀 창을 고유벡터 방향으로 회전할 때 발생하는 근사 오류:
- 회전각 $\theta$가 클수록 오류 증가
- **비직사각형 픽셀 창** 또는 **타원형 픽셀 창**에 대한 더 정확한 근사법 연구 필요

#### (3) 3D 가우시안 정규화와의 결합

논문에서 3D 스무딩 필터(Mip-Splatting의 3D 필터)를 선택적으로 결합할 수 있음을 보였다 (Ours + 3D filter). 그러나 단일 해상도 설정에서만 소폭 개선 효과가 있고, 다중 해상도 설정에서는 큰 차이가 없었다. **최적의 3D 정규화 전략** 탐색이 필요하다.

#### (4) 동적 장면 및 비정형 구조로의 확장

현재 3DGS 기반 동적 장면 렌더링(e.g., Gaussian Splashing, Drivable 3D Gaussian Avatars) 연구들이 활발히 진행 중이다. 이들 방법에 Analytic-Splatting의 픽셀 창 적분 기법을 통합할 때:
- 시간적 일관성(temporal consistency) 유지
- 동적 픽셀 풋프린트 변화에 대한 적응

#### (5) 비등방성(anisotropic) 픽셀에 대한 고려

현재 방법은 픽셀 창을 $1 \times 1$ 크기의 정사각형으로 가정한다. 비표준 이미지 센서나 비등방성 샘플링 환경에서는:
- 비정사각형 픽셀 창 $[u_x - \frac{w}{2}, u_x + \frac{w}{2}] \times [u_y - \frac{h}{2}, u_y + \frac{h}{2}]$에 대한 일반화
- 다양한 카메라 모델 지원

#### (6) 더 많은 벤치마크 검증

현재 Blender Synthetic, Mip-NeRF 360, Tanks&Temples, Deep Blending에서 검증되었으나, **의료 영상, 위성 영상, 현미경 영상** 등 특수 도메인에서의 일반화 성능 검증이 필요하다.

#### (7) 다른 렌더링 파이프라인과의 통합

Analytic-Splatting의 핵심 아이디어(CDF 기반 창 적분 근사)는 3DGS뿐 아니라 **포인트 기반 렌더링, 메시 기반 렌더링** 등 다른 래스터화 파이프라인에도 적용 가능성이 있다.

---

## 참고 자료

**주요 논문 (본 문서에서 직접 인용)**

1. **Analytic-Splatting** (본 논문): Liang et al., "Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic Integration," arXiv:2403.11056v2, 2024.

2. **3DGS**: Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G., "3D Gaussian Splatting for Real-Time Radiance Field Rendering," ACM Transactions on Graphics 42(4), 2023.

3. **Mip-Splatting**: Yu, Z., Chen, A., Huang, B., Sattler, T., Geiger, A., "Mip-Splatting: Alias-Free 3D Gaussian Splatting," arXiv:2311.16493, 2023.

4. **NeRF**: Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis," ECCV 2020.

5. **Mip-NeRF**: Barron, J.T., Mildenhall, B., Tancik, M., Hedman, P., Martin-Brualla, R., Srinivasan, P.P., "Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields," ICCV 2021.

6. **Mip-NeRF 360**: Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P., "Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields," CVPR 2022.

7. **Zip-NeRF**: Barron, J.T., Mildenhall, B., Verbin, D., Srinivasan, P.P., Hedman, P., "Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields," ICCV 2023.

8. **Tri-MipRF**: Hu, W., Wang, Y., Ma, L., Yang, B., Gao, L., Liu, X., Ma, Y., "Tri-MipRF: Tri-Mip Representation for Efficient Anti-Aliasing Neural Radiance Fields," ICCV 2023.

9. **Instant-NGP**: Müller, T., Evans, A., Schied, C., Keller, A., "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding," ACM ToG 41(4), 2022.

10. **Plenoxels**: Fridovich-Keil, S., et al., "Plenoxels: Radiance Fields without Neural Networks," CVPR 2022.
