# Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields

---

## 1. 핵심 주장 및 주요 기여 요약

**Mip-NeRF 360**은 기존 mip-NeRF를 확장하여, 카메라가 360도 회전하며 장면 콘텐츠가 임의의 거리에 존재하는 **비한정(unbounded) 장면**에서도 고품질의 안티앨리어싱된 뷰 합성(view synthesis)을 수행하는 모델이다.

### 주요 기여 (3가지)
1. **비선형 장면 파라미터화 (Non-linear Scene Parameterization):** 유클리드 공간의 가우시안을 수축(contraction) 함수와 칼만 필터(Kalman filter) 방식으로 변환하여 비한정 장면을 유한한 공간에 매핑
2. **온라인 증류 (Online Distillation):** 작은 "Proposal MLP"와 큰 "NeRF MLP"를 동시에 학습하여, 모델 용량을 ~15배 키우면서도 학습 시간은 ~2배만 증가
3. **왜곡 기반 정규화 (Distortion-based Regularizer):** "floater" 및 "background collapse" 아티팩트를 억제하는 새로운 정규화 손실 함수 제안

**핵심 성과:** mip-NeRF 대비 평균 제곱 오차(MSE)를 **57% 감소**시키고, 복잡한 실세계 비한정 장면에서 사실적인 렌더링과 상세한 깊이 맵(depth map)을 생성한다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 NeRF/mip-NeRF를 비한정 360도 장면에 적용할 때 발생하는 세 가지 핵심 문제:

| 문제 | 설명 |
|------|------|
| **파라미터화 (Parameterization)** | 비한정 장면은 유클리드 공간에서 임의로 큰 영역을 차지하지만, mip-NeRF는 유한한 좌표 공간을 요구 |
| **효율성 (Efficiency)** | 크고 상세한 장면은 더 많은 네트워크 용량과 레이당 더 많은 샘플을 필요로 하여 학습 비용이 급증 |
| **모호성 (Ambiguity)** | 비한정 장면의 콘텐츠는 소수의 레이에서만 관측되어, 2D→3D 복원의 본질적 모호성이 심화됨 |

### 2.2 사전 지식: mip-NeRF 기초

레이 $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$에서 구간 $T_i = [t_i, t_{i+1})$에 대해 원뿔 절두체(conical frustum)의 평균·공분산 $(\boldsymbol{\mu}, \boldsymbol{\Sigma}) = \mathbf{r}(T_i)$를 계산하고, 통합 위치 부호화(Integrated Positional Encoding, IPE)를 적용한다:

```math
\gamma(\boldsymbol{\mu}, \boldsymbol{\Sigma}) = \left\{\begin{bmatrix} \sin(2^\ell \boldsymbol{\mu}) \exp\left(-2^{2\ell-1}\text{diag}(\boldsymbol{\Sigma})\right) \\ \cos(2^\ell \boldsymbol{\mu}) \exp\left(-2^{2\ell-1}\text{diag}(\boldsymbol{\Sigma})\right) \end{bmatrix}\right\}_{\ell=0}^{L-1}
```

MLP는 밀도 $\tau$와 색상 $\mathbf{c}$를 출력한다:

$$\forall T_i \in \mathbf{t}, \quad (\tau_i, \mathbf{c}_i) = \text{MLP}(\gamma(\mathbf{r}(T_i)); \Theta_{\text{NeRF}}) $$

볼륨 렌더링은 알파 합성 가중치 $w_i$를 사용한다:

$$\mathbf{C}(\mathbf{r}, \mathbf{t}) = \sum_i w_i \mathbf{c}_i $$

$$w_i = \left(1 - e^{-\tau_i(t_{i+1} - t_i)}\right) e^{-\sum_{i' < i} \tau_{i'}(t_{i'+1} - t_{i'})} $$

---

### 2.3 제안 방법 1: 장면 및 레이 파라미터화

#### 가우시안에 대한 좌표 변환

임의의 매끄러운 좌표 변환 $f(\mathbf{x}): \mathbb{R}^n \to \mathbb{R}^n$에 대해, 가우시안 $(\boldsymbol{\mu}, \boldsymbol{\Sigma})$를 다음과 같이 변환한다 (확장 칼만 필터와 동일):

$$f(\mathbf{x}) \approx f(\boldsymbol{\mu}) + \mathbf{J}_f(\boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu}) $$

$$f(\boldsymbol{\mu}, \boldsymbol{\Sigma}) = \left(f(\boldsymbol{\mu}),\; \mathbf{J}_f(\boldsymbol{\mu})\boldsymbol{\Sigma}\mathbf{J}_f(\boldsymbol{\mu})^\top\right) $$

여기서 $\mathbf{J}_f(\boldsymbol{\mu})$는 $f$의 야코비안(Jacobian)이다.

#### 수축(Contraction) 함수

$$\text{contract}(\mathbf{x}) = \begin{cases} \mathbf{x} & \|\mathbf{x}\| \leq 1 \\ \left(2 - \frac{1}{\|\mathbf{x}\|}\right)\frac{\mathbf{x}}{\|\mathbf{x}\|} & \|\mathbf{x}\| > 1 \end{cases} $$

- $\|\mathbf{x}\| \leq 1$인 근거리 영역은 영향 없음 (유클리드 공간 유지)
- $\|\mathbf{x}\| > 1$인 원거리 점들은 반지름 2인 구 안으로 수축됨
- 원거리 점들을 **역거리(disparity)에 비례**하여 분포시킴 → NDC의 전방향 확장

#### 디스패리티 기반 레이 샘플링

유클리드 거리 $t$와 정규화 거리 $s$ 사이의 가역 매핑:

$$s \triangleq \frac{g(t) - g(t_n)}{g(t_f) - g(t_n)}, \quad t \triangleq g^{-1}(s \cdot g(t_f) + (1 - s) \cdot g(t_n)) $$

$g(x) = 1/x$로 설정하면 **디스패리티에 선형인 샘플링**이 되어, 수축 함수와 결합 시 비한정 장면이 NeRF 원본의 유한 공간 설정과 유사한 형태가 된다.

---

### 2.4 제안 방법 2: 온라인 증류 (Coarse-to-Fine Online Distillation)

#### 구조
- **Proposal MLP** ($\Theta_{\text{prop}}$): 4층, 256 히든 유닛 → 밀도만 예측 (색상 X), 2회 평가, 각 64 샘플
- **NeRF MLP** ($\Theta_{\text{NeRF}}$): 8층, 1024 히든 유닛 → 밀도 + 색상 예측, 1회 평가, 32 샘플

Proposal MLP가 생성한 가중치 $\hat{\mathbf{w}}$로부터 재샘플링하여 NeRF MLP에 구간을 제공한다.

#### Proposal 손실 함수

먼저 "bound" 함수를 정의한다:

$$\text{bound}\left(\hat{\mathbf{t}}, \hat{\mathbf{w}}, T\right) = \sum_{j:\, T \cap \hat{T}_j \neq \varnothing} \hat{w}_j $$

Proposal 손실:

$$\mathcal{L}_{\text{prop}}\left(\mathbf{t}, \mathbf{w}, \hat{\mathbf{t}}, \hat{\mathbf{w}}\right) = \sum_i \frac{1}{w_i} \max\left(0,\; w_i - \text{bound}\left(\hat{\mathbf{t}}, \hat{\mathbf{w}}, T_i\right)\right)^2 $$

- **비대칭 손실**: proposal 가중치가 NeRF 가중치를 **과소추정**할 때만 페널티 (과대추정은 허용)
- $w_i$로 나누어 bound가 0일 때 그래디언트가 상수가 되도록 보장
- **Stop-gradient**: NeRF MLP의 출력 $\mathbf{t}, \mathbf{w}$에 적용하여 NeRF가 "선도"하고 proposal이 "추종"하도록 함

**효과:** 전체 모델 용량이 mip-NeRF의 ~15배이지만, 학습 시간은 ~2배 증가에 그침.

---

### 2.5 제안 방법 3: 왜곡 기반 정규화 (Distortion Regularizer)

정규화 거리 $s$와 가중치 $\mathbf{w}$에 대해 정의:

$$\mathcal{L}_{\text{dist}}(\mathbf{s}, \mathbf{w}) = \iint_{-\infty}^{\infty} \mathbf{w}_\mathbf{s}(u)\,\mathbf{w}_\mathbf{s}(v)\,|u - v|\,du\,dv $$

여기서 $\mathbf{w}\_\mathbf{s}(u) = \sum_i w_i \mathbf{1}\_{[s_i, s_{i+1})}(u)$이다. 닫힌 형태로 효율적 계산이 가능하다:

$$\mathcal{L}_{\text{dist}}(\mathbf{s}, \mathbf{w}) = \sum_{i,j} w_i w_j \left|\frac{s_i + s_{i+1}}{2} - \frac{s_j + s_{j+1}}{2}\right| + \frac{1}{3}\sum_i w_i^2(s_{i+1} - s_i) $$

- 첫 번째 항: 모든 구간 중점 쌍의 가중 거리 최소화 → 가중치를 **공간적으로 집중**
- 두 번째 항: 각 구간의 가중 크기 최소화 → **얇은 표면** 선호
- **Floater** 억제: 반투명한 부유 영역을 제거
- **Background collapse** 방지: 원거리 표면이 근거리로 잘못 모델링되는 현상 억제

---

### 2.6 전체 최적화 목적 함수

$$\mathcal{L}_{\text{recon}}(\mathbf{C}(\mathbf{t}), \mathbf{C}^*) + \lambda\,\mathcal{L}_{\text{dist}}(\mathbf{s}, \mathbf{w}) + \sum_{k=0}^{1} \mathcal{L}_{\text{prop}}\left(\mathbf{s}, \mathbf{w}, \hat{\mathbf{s}}^k, \hat{\mathbf{w}}^k\right) $$

- $\lambda = 0.01$ (모든 실험에서 고정)
- $\mathcal{L}_{\text{recon}}$: Charbonnier loss $\sqrt{(x - x^*)^2 + \epsilon^2}$, $\epsilon = 0.001$
- 250K iteration, batch size $2^{14}$, Adam ($\beta_1=0.9, \beta_2=0.999$)
- 학습률: $2 \times 10^{-3} \to 2 \times 10^{-5}$ log-linear annealing

---

### 2.7 모델 구조 요약

| 구성 요소 | Proposal MLP | NeRF MLP |
|----------|-------------|----------|
| 층 수 | 4 | 8 |
| 히든 유닛 | 256 | 1024 |
| 활성 함수 | ReLU + softplus(밀도) | ReLU + softplus(밀도) |
| 출력 | 밀도 $\tau$만 | 밀도 $\tau$ + 색상 $\mathbf{c}$ |
| 평가 횟수 | 2회, 각 64 샘플 | 1회, 32 샘플 |
| 전체 파라미터 | ~9.9M (합산) | |

추가 구성 요소:
- **Off-Axis IPE**: 이중 분할 정이십면체(twice-tessellated icosahedron) 정점을 기저 $\mathbf{P}$로 사용하여 비등방 가우시안을 구분
- **Annealing**: Schlick's bias 함수로 proposal 가중치의 거듭제곱을 점진적으로 1로 증가
- **Dilation**: proposal 히스토그램의 구간을 약간 확장하여 회전 앨리어싱 억제
- **랜덤 배경색**: 학습 시 $[0,1]^3$에서 랜덤 RGB 배경색 → 불투명 배경 복원 유도

---

### 2.8 성능 향상

#### 주 결과 (저자 제공 360 데이터셋, 9개 장면)

| 모델 | PSNR↑ | SSIM↑ | LPIPS↓ | 학습 시간 | 파라미터 |
|------|-------|-------|--------|---------|--------|
| mip-NeRF | 24.04 | 0.616 | 0.441 | 3.17h | 0.7M |
| NeRF++ | 25.11 | 0.676 | 0.375 | 9.45h | 2.4M |
| mip-NeRF (큰 MLP) | 26.19 | 0.748 | 0.285 | 22.71h | 9.0M |
| NeRF++ (큰 MLPs) | 26.39 | 0.750 | 0.293 | 19.88h | 9.0M |
| SVS | 25.33 | 0.771 | **0.211** | - | - |
| **Mip-NeRF 360** | **27.69** | **0.792** | 0.237 | **6.89h** | 9.9M |

- mip-NeRF 대비 **MSE 57% 감소**, 학습 시간 2.17배 증가
- 큰 MLP를 사용한 mip-NeRF/NeRF++ 대비 ~3배 빠른 학습 + 더 높은 정확도
- SVS보다 PSNR/SSIM에서 우위, LPIPS에서만 약간 열위 (SVS는 perceptual loss로 학습)

#### Ablation Study (bicycle 장면)

| 변형 | PSNR↑ | SSIM↑ | LPIPS↓ |
|-----|-------|-------|--------|
| No $\mathcal{L}_{\text{prop}}$ | 20.49 | 0.406 | 0.573 |
| No $\mathcal{L}_{\text{dist}}$ | 24.41 | 0.687 | 0.300 |
| No Contraction | 23.77 | 0.642 | 0.347 |
| Small NeRF MLP (256) | 22.80 | 0.515 | 0.480 |
| **Complete Model** | **24.37** | **0.687** | **0.300** |

모든 구성 요소가 최종 성능에 기여함을 확인.

---

### 2.9 한계 (Limitations)

1. **미세 구조 복원 어려움**: 자전거 바퀴살, 잎맥 등 매우 얇은 구조 누락 가능
2. **장면 중심으로부터 먼 카메라**: 카메라가 장면 중심에서 크게 벗어나면 품질 저하
3. **학습 시간**: 여전히 가속기에서 수 시간의 학습 필요 → 온-디바이스 학습 불가
4. **장면별 최적화**: 각 장면마다 별도 학습이 필요하며, 다른 장면으로의 일반화 불가 (per-scene optimization)

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 모델의 일반화 한계

Mip-NeRF 360은 **per-scene optimization** 패러다임에 해당한다. 즉, 각 장면마다 독립적으로 MLP를 처음부터 학습해야 하며, 학습한 모델은 해당 장면에만 유효하다. 이는 NeRF 계열 모델의 본질적 한계이다.

### 3.2 일반화 성능 향상에 기여하는 설계 요소

논문의 세 가지 핵심 기여는 **장면 간 일반화**가 아닌 **장면 내 일반화 (novel view generalization within a scene)**를 향상시킨다:

1. **수축 함수 + 디스패리티 샘플링**: 임의의 비한정 장면에 대해 일관된 좌표 체계를 제공하므로, 다양한 규모·구조의 장면에 동일한 하이퍼파라미터로 적용 가능
2. **왜곡 정규화 ($\mathcal{L}_{\text{dist}}$)**: 관측되지 않은 뷰에서의 아티팩트(floater, background collapse)를 억제하여 **novel view에서의 품질**을 향상 → 장면 내 일반화 강화
3. **Proposal MLP의 온라인 증류**: 효율적인 샘플링으로 높은 용량의 NeRF MLP 사용이 가능해져, 복잡한 장면의 세부 사항을 더 잘 캡처 → 관측되지 않은 뷰에서도 더 나은 복원

### 3.3 장면 간 일반화를 위한 향후 방향

- **GLO (Generative Latent Optimization)** 임베딩 실험 (Table 1): 4차원 외관 임베딩을 사용하여 조명 변화에 대한 불변성 확보 → 제한적이나 photometric 일반화의 가능성 시사
- 논문의 파라미터화 및 정규화 기법은 **generalizable NeRF** (예: pixelNeRF, IBRNet, GNT 등)에 통합될 수 있는 **장면 독립적(scene-agnostic) 구성 요소**임
- Off-axis IPE의 비등방 가우시안 구분 능력은 다양한 스케일의 장면에서 **features의 표현력**을 높여, 장면 간 전이(transfer) 시에도 유리할 수 있음

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 연구에 미치는 영향

1. **비한정 장면 표준 설정 확립**: Mip-NeRF 360이 제안한 데이터셋(9개 360도 장면)과 평가 프로토콜은 후속 연구(3D Gaussian Splatting, Zip-NeRF 등)의 **표준 벤치마크**가 됨
2. **Proposal Network 패러다임**: 온라인 증류를 통한 proposal–NeRF 분리 아키텍처는 Zip-NeRF, Nerfacto 등 후속 모델에 널리 채택됨
3. **수축 함수의 보편적 사용**: Equation (10)의 contraction은 비한정 장면 처리의 **사실상 표준(de facto standard)**이 됨
4. **정규화 전략**: 왜곡 손실은 NeRF의 기하학적 품질 향상을 위한 정규화 연구의 기초가 됨

### 4.2 앞으로 연구 시 고려할 점

| 측면 | 고려 사항 |
|------|----------|
| **학습 속도** | 여전히 수 시간 소요 → 해시 그리드, 가우시안 스플래팅 등 명시적 표현과의 결합 필요 |
| **실시간 렌더링** | 학습된 MLP의 실시간 렌더링은 baking/distillation 없이 어려움 |
| **장면 간 일반화** | Per-scene optimization의 한계 → feed-forward generalizable 모델과의 통합 필요 |
| **동적 장면** | 정적 장면만 다룸 → 시간 축 확장 (D-NeRF, Nerfies 등) 필요 |
| **photometric 변동** | GLO는 부분적 해결 → NeRF-W 수준의 transient/appearance 분리가 추가로 필요 |
| **메모리/컴퓨팅** | 1024 히든 유닛의 큰 NeRF MLP → 경량화 및 효율적 추론 연구 필요 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 차이점 | Mip-NeRF 360과의 관계 |
|------|------|-----------|---------------------|
| **NeRF** (Mildenhall et al.) | 2020 | 점 기반 PE, bounded/front-facing 장면 | Mip-NeRF 360의 기초 모델 |
| **Mip-NeRF** (Barron et al.) | 2021 | 원뿔 기반 IPE, 안티앨리어싱 | 직접적 선행 연구, 본 논문이 확장 |
| **NeRF++** (Zhang et al.) | 2020 | 두 개 MLP(inside/outside)로 비한정 장면 처리 | 유사한 문제 해결, 다른 파라미터화 접근 |
| **DONeRF** (Neff et al.) | 2021 | 로그 간격 샘플링, depth oracle | 비한정 파라미터화의 대안, 본 논문에서 비교/열위 |
| **Instant-NGP** (Müller et al.) | 2022 | Multi-resolution hash encoding → 학습 수초~수분 | 학습 속도를 극적으로 개선하나 비한정 장면 처리는 본 논문의 기법 필요 |
| **Zip-NeRF** (Barron et al.) | 2023 | Mip-NeRF 360 + Instant-NGP의 결합 | Mip-NeRF 360의 파라미터화·proposal network를 hash grid와 결합 → 품질·속도 모두 향상 |
| **3D Gaussian Splatting (3DGS)** (Kerbl et al.) | 2023 | 명시적 가우시안 표현, 래스터화 기반 실시간 렌더링 | NeRF와 완전히 다른 표현이지만 Mip-NeRF 360 데이터셋을 벤치마크로 사용; 실시간 렌더링 가능하나 안티앨리어싱에서 열위 |
| **Mip-Splatting** (Yu et al.) | 2024 | 3DGS에 3D 스무딩 + 2D Mip 필터 추가 → 안티앨리어싱 | Mip-NeRF 360의 안티앨리어싱 철학을 가우시안 스플래팅에 적용 |
| **Nerfacto** (Nerfstudio, Tancik et al.) | 2023 | Mip-NeRF 360의 proposal network + Instant-NGP의 hash grid + 다양한 기법의 모듈형 조합 | 본 논문의 여러 구성 요소를 실용적 프레임워크로 통합 |
| **NeRF in the Wild** (Martin-Brualla et al.) | 2021 | 외관 임베딩 + transient 분리 → "in the wild" 사진 컬렉션 처리 | 본 논문에서 GLO 변형으로 부분 채택; photometric 변동 처리에 보완적 |
| **Ref-NeRF** (Verbin et al.) | 2022 | 반사 방향 기반 외관 모델링 → 광택 표면 개선 | 동일 저자 그룹의 후속 연구, Mip-NeRF 360 위에 반사 모델링 추가 가능 |

### 핵심 트렌드 분석

1. **Mip-NeRF 360의 구성 요소가 후속 연구의 표준이 됨**: contraction, proposal network, distortion loss 등이 Zip-NeRF, Nerfacto 등에 직접 채택
2. **MLP → 명시적 표현으로의 전환**: 3DGS, Instant-NGP 등이 학습/렌더링 속도를 혁신적으로 개선하였으나, Mip-NeRF 360의 안티앨리어싱 및 비한정 장면 처리 기법은 여전히 필수적 구성 요소
3. **일반화 성능**: pixelNeRF, IBRNet, GNT, MVSNeRF 등 generalizable NeRF 연구가 활발하나, Mip-NeRF 360 수준의 품질에는 미치지 못함 → 두 패러다임의 결합이 핵심 연구 방향

---

## 참고자료

1. **Barron, J. T., Mildenhall, B., Verbin, D., Srinivasan, P. P., & Hedman, P.** (2022). "Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields." *CVPR 2022*. arXiv:2111.12077v3.
2. **Barron, J. T., et al.** (2021). "Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields." *ICCV 2021*.
3. **Mildenhall, B., et al.** (2020). "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis." *ECCV 2020*.
4. **Zhang, K., Riegler, G., Snavely, N., & Koltun, V.** (2020). "NeRF++: Analyzing and Improving Neural Radiance Fields." arXiv:2010.07492.
5. **Neff, T., et al.** (2021). "DONeRF: Towards Real-Time Rendering of Compact Neural Radiance Fields using Depth Oracle Networks." *Computer Graphics Forum*.
6. **Müller, T., Evans, A., Schied, C., & Keller, A.** (2022). "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding." *ACM TOG (SIGGRAPH)*.
7. **Barron, J. T., et al.** (2023). "Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields." *ICCV 2023*.
8. **Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G.** (2023). "3D Gaussian Splatting for Real-Time Radiance Field Rendering." *ACM TOG (SIGGRAPH)*.
9. **Yu, Z., et al.** (2024). "Mip-Splatting: Alias-free 3D Gaussian Splatting." *CVPR 2024*.
10. **Tancik, M., et al.** (2023). "Nerfstudio: A Modular Framework for Neural Radiance Field Development." *SIGGRAPH 2023*.
11. **Martin-Brualla, R., et al.** (2021). "NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections." *CVPR 2021*.
12. **Verbin, D., Hedman, P., Mildenhall, B., Zickler, T., Barron, J. T., & Srinivasan, P. P.** (2022). "Ref-NeRF: Structured View-Dependent Appearance for Neural Radiance Fields." *CVPR 2022*.
13. **Riegler, G. & Koltun, V.** (2021). "Stable View Synthesis." *CVPR 2021*.
14. **Kalman, R. E.** (1960). "A New Approach to Linear Filtering and Prediction Problems." *Journal of Basic Engineering*.
