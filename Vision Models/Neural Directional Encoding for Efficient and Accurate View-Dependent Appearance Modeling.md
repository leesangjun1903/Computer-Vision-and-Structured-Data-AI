
# Neural Directional Encoding for Efficient and Accurate View-Dependent Appearance Modeling
---

## 📌 논문 기본 정보

| 항목 | 내용 |
|---|---|
| **제목** | Neural Directional Encoding for Efficient and Accurate View-Dependent Appearance Modeling |
| **저자** | Liwen Wu (UC San Diego), Sai Bi, Zexiang Xu, Fujun Luan, Kai Zhang, Iliyan Georgiev, Kalyan Sunkavalli (Adobe Research), Ravi Ramamoorthi (UC San Diego) |
| **학회** | CVPR 2024 |
| **arXiv** | arXiv:2405.14847 |
| **코드/프로젝트** | [lwwu2.github.io/nde](https://lwwu2.github.io/nde/) |

---

## 1. 핵심 주장 및 주요 기여 요약

광택 금속, 유광 페인트 등 정반사(specular) 물체의 Novel-view Synthesis는 여전히 중요한 도전 과제이며, 반짝이는 외관뿐 아니라 환경 내 다른 물체의 반사를 포함하는 글로벌 조명 효과까지 재현해야 한다.

이 논문은 **Neural Directional Encoding (NDE)** 를 제안하며, 이는 specular 물체 렌더링을 위한 NeRF의 시점 의존적(view-dependent) 외관 인코딩이다. NDE는 특징 그리드(feature-grid) 기반 공간 인코딩 개념을 **각도 영역(angular domain)** 으로 전이함으로써, 고주파 각도 신호(high-frequency angular signals) 모델링 능력을 획기적으로 향상시킨다.

기존 방법들이 오직 각도 입력만을 사용하는 인코딩 함수와 달리, NDE는 추가적으로 **공간 특징을 콘 추적(cone-trace)** 하여 공간적으로 변하는(spatially varying) 방향 인코딩을 획득함으로써, 어려운 **상호반사(interreflection) 효과** 를 처리한다.

합성 데이터셋과 실제 데이터셋 모두에서의 광범위한 실험을 통해, NDE를 탑재한 NeRF 모델이 (1) specular 물체의 뷰 합성(view synthesis)에서 최신 기술을 능가하고, (2) 소형 네트워크에서도 동작하여 실시간(real-time) 추론을 가능하게 함을 보였다.

---

## 2. 해결하고자 하는 문제

### 2.1 기존 방법의 한계

기존 방법은 대형 MLP(multi-layer perceptron)를 필요로 하며, 분석적(analytical) 방향 인코딩 함수를 사용할 경우 수렴 속도가 느리다는 문제를 보인다.

Ref-NeRF 같은 방법들은 대형 MLP에서 뷰잉 방향을 분석적 함수로 인코딩하여, **복잡한 반사를 모델링하는 데 실패**한다.

반사 방향(reflected direction)을 활용하는 IDE(Integrated Directional Encoding)는 환경 맵 유사 함수를 학습하고 거친 반사 효과를 위해 사전 필터링(pre-filtered)되지만, NDE는 유사한 일반 뷰 의존적 외관을 더 **작은 연산 비용**으로 모델링할 수 있다.

핵심 문제를 정리하면:
- ❌ 고주파 정반사 효과를 모델링하기 위해 **대형 MLP** 가 필요
- ❌ 방향 인코딩이 각도 입력만 사용 → **위치 의존적(spatially varying) 반사** 처리 불가
- ❌ 상호반사(interreflection) 처리 어려움
- ❌ 실시간 렌더링 불가

---

## 3. 제안하는 방법 (수식 포함)

### 3.1 파이프라인 개요

NDE 파이프라인은 원거리(far-field) 반사를 큐브맵(cubemap)으로, 근거리(near-field) 상호반사를 볼륨(volume)으로 인코딩한다. 두 표현 모두 방향을 인코딩하기 위한 **학습 가능한 특징 벡터(learnable feature vectors)** 를 저장하고, 거친 반사를 처리하기 위해 **밉맵(mip-mapped)** 처리된다. 반사 광선이 주어지면, 표면 거칠기에 비례한 콘(cone)을 추적하여 공간 특징을 집계하고 큐브맵 특징을 배경으로 혼합한다.

### 3.2 렌더링 방정식 (Volume Rendering)

표준 NeRF의 볼륨 렌더링 방정식은 다음과 같다:

$$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\, \sigma(\mathbf{r}(t))\, \mathbf{c}(\mathbf{r}(t), \mathbf{d})\, dt$$

여기서:

$$T(t) = \exp\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s))\, ds\right)$$

- $\sigma$: 볼륨 밀도(volume density)
- $\mathbf{c}$: 시점 의존적 색상(view-dependent color)
- $\mathbf{d}$: 뷰 방향(viewing direction)

### 3.3 NDE의 스페큘러 색상 모델링

NDE는 색상을 diffuse와 specular로 분리한다:

$$C = c_d + k_s \cdot c_s(\mathbf{H}_n, \mathbf{H}_c)$$

- $c_d$: 난반사(diffuse) 색상
- $k_s$: 스페큘러 강도 스케일
- $c_s$: 스페큘러 색상 (NDE로부터 디코딩)
- $\mathbf{H}_n$: 근거리 볼륨으로부터의 특징
- $\mathbf{H}_c$: 큐브맵으로부터의 특징

### 3.4 Spatio-Spatial 재파라미터화 (핵심 인사이트)

공간-공간 인코딩(spatio-spatial encoding)은 거울 반사의 일반적인 공간-각도 인코딩(spatio-angular encoding)과 등가이면서, 동시에 서로 다른 $\mathbf{x}$에 걸친 $\mathbf{x}'$의 변화를 포착한다. 이 아이디어는 반사 콘으로 덮이는 밉맵 공간 특징을 콘 추적(cone tracing)함으로써 거친 반사를 모델링하는 데까지 확장된다.

저자들의 핵심 통찰은, **공간-각도 반사를 현재 및 다음 반사 위치의 공간-공간 함수**로도 파라미터화할 수 있다는 것이다.

즉, 시점 의존적 반사를 다음과 같이 재파라미터화한다:

$$c_s(\mathbf{x}, \boldsymbol{\omega}_r) \quad\longrightarrow\quad c_s(\mathbf{x}, \mathbf{x}')$$

여기서 $\mathbf{x}' = \mathbf{x} + t \cdot \boldsymbol{\omega}_r$ 는 다음 반사 위치(second bounce location)이고, $\boldsymbol{\omega}_r$는 반사 방향이다.

### 3.5 콘 추적(Cone Tracing)을 통한 근거리 특징 집계

거친 반사(rough reflection)에 대한 특징 집계:

$$\mathbf{H}_n = \int_0^{\infty} T(t)\,\sigma_n(t)\,h_n(\mathbf{x}+\boldsymbol{\omega}_r t, \lambda_i)\,dt$$

여기서 밉 레벨은:

$$\lambda_i = \log_2(2r_i)$$

- $r_i$: 콘의 반지름(cone radius) → 표면 거칠기 $\rho$에 비례
- $h_n$: 밉맵된 공간 특징
- $\sigma_n$: 밉맵된 밀도(별도 최적화)

거친 반사의 경우, 반사 로브(reflection lobe) 아래에서 평균된 두 번째 반사 특징을 콘 추적[9]으로 집계하며, 이는 반사 광선 $\mathbf{x} + \boldsymbol{\omega}_r t$를 따라 mip 레벨 $\lambda_i = \log_2(2r_i)$에서 밉맵된 공간 특징 $h_n$을 밉맵된 밀도 $\sigma_n$으로 볼륨 렌더링한다.

### 3.6 최적화 손실 함수 (Charbonnier Loss)

표현은 색조 매핑(tone-mapped) 공간에서 정답 픽셀 색상 $C_{gt}$와 렌더링 $C$ 사이의 **Charbonnier 손실**로 최적화된다.

$$\mathcal{L} = \sum_{\mathbf{r}} \sqrt{\|C(\mathbf{r}) - C_{gt}(\mathbf{r})\|^2 + \epsilon^2}$$

### 3.7 빠른 공간 인코딩

기하 최적화에는 VolSDF의 위치 인코딩 MLP를 사용하여 SDF를 출력하며, 계산 비용 감소를 위해 **해시 그리드(hash grid)** 를 활용하여 다른 공간 특징들($c_d, k_s, \rho, f$)을 인코딩하고 나머지 MLP들은 모두 소형으로 유지된다.

---

## 4. 모델 구조

### 4.1 전체 구조

NDE는 크게 두 개의 방향 인코딩 모듈과 하나의 작은 디코더 MLP로 구성된다:

| 구성 요소 | 역할 |
|---|---|
| **Far-field 큐브맵** | 원거리 환경 조명 (학습 가능한 특징 벡터, 밉맵) |
| **Near-field 특징 볼륨** | 근거리 상호반사 (3D 공간 특징 그리드, 밉맵) |
| **콘 트레이서** | 거칠기 $\rho$에 비례한 반사 콘으로 특징 집계 |
| **소형 디코더 MLP** | $\mathbf{H}_n + \mathbf{H}_c$ 를 입력받아 스페큘러 색상 출력 |
| **SDF Network (VolSDF)** | 정확한 표면 형상 추정 |
| **Hash Grid** | 공간 특징 ($c_d, k_s, \rho, f$) 고속 인코딩 |

큐브맵 기반 특징 인코딩을 통해 단 **2 레이어, 64 폭의 소형 MLP** 만으로도 IDE(8 레이어, 256 폭 MLP)에 필적하는 거울 반사 세부 사항을 모델링할 수 있으며, 소형 MLP에서 IDE가 실패하는 경우에도 제대로 동작한다.

### 4.2 실시간 버전 (NDE-RT)

SDF를 marching cubes를 통해 메시(mesh)로 변환하고 $c_d, k_s, \rho, f$를 메시 정점(vertex)에 베이킹(baking)하여 실시간 모델을 생성할 수 있다.

NDE는 효율적이며 **실시간 웹 렌더링**을 지원한다.

---

## 5. 성능 향상

### 5.1 정량적 비교

합성 장면에 대한 정량적 비교에서 NDE는 specular 물체 뷰 합성에서 다른 방법들과 비교하여 **최고 또는 2위** 성능을 기록하였다.

전반적으로 NDE는 합성 장면에서 최고의 렌더링 품질을 제공하며, 이는 NDE가 **원거리 반사와 상호반사 모두를 가장 세밀하게 모델링**하기 때문이며, 이는 기하 재구성 품질 향상에도 기여한다.

ENVIDR의 SSIM이 일부 장면에서 NDE보다 약간 높지만, NDE는 **2dB 이상 더 높은 PSNR**과 더 높은 LPIPS 점수를 달성한다.

NDE는 찻주전자의 세부 사항 및 핑크 공의 다중 반사 반사까지 성공적으로 재구성하며, **NVIDIA 3090 GPU에서 75 FPS** 의 낮은 계산 오버헤드를 달성한다.

### 5.2 효율성 비교

NDE는 렌더링 품질을 저하시키지 않고도 실질적으로 더 작은 MLP가 필요하기 때문에 평가에 **수 분의 1초** 밖에 걸리지 않는다. 반면 다른 기준선들은 렌더링 품질을 유지하기 위해 대형 MLP가 필요하여 실시간 시각화가 불가능하다.

### 5.3 기하 재구성 향상

Ref-NeRF는 상호반사를 흉내 내기 위해 잘못된 기하를 사용하는 경향이 있다. 반면, NDE는 상호반사를 모델링할 충분한 용량을 갖추고 있어 **더 정확한 법선(normal) 추정**이 가능하다.

---

## 6. 한계 (Limitations)

논문에서 언급되거나 구조적으로 추론 가능한 한계는 다음과 같다:

| 한계 | 설명 |
|---|---|
| **장면별 최적화** | NDE는 여전히 scene-specific 최적화가 필요하며 범용 사전 학습(pre-trained) 모델로의 직접 일반화는 한계가 있음 |
| **오목한 기하 처리** | SDF는 구 받침대(sphere base)와 같은 **오목한 기하(concave geometry)** 를 모델링하는 데 비효율적이어서, Materials 장면에서의 PSNR이 Ref-NeRF보다 낮은 원인이 됨 |
| **실제 장면 평가** | 실제 장면 평가에서 캡처 기기의 반사를 제거하기가 어려워, 전경 줌인 부분에 대해서만 정량적 비교가 가능하다 |
| **빠른 추론 한계** | 콘 트레이싱 기반 근거리 특징 평가가 각 1차 광선 샘플에 대한 반사 광선 추적을 필요로 해 메모리/연산 오버헤드 존재 |
| **동적 장면** | 정적 장면만을 대상으로 하며, 동적 물체/조명 변화에 대한 처리가 부재 |

---

## 7. 모델 일반화 성능 향상 가능성

### 7.1 현재 일반화 구조

NDE는 단순화된 조명이나 반사를 가정하지 않고 일반적인 시점 의존적 외관을 모델링할 수 있다. 이는 IDE와 달리 물리적 가정(예: 완전 거울 표면, lambertian 반사)을 강요하지 않는다는 점에서 구조적 일반화 능력이 높다.

### 7.2 일반화 확장 가능 방향

**① 학습 가능한 특징 그리드의 전이 가능성:**
NDE의 큐브맵 및 볼륨 특징 그리드는 장면 간 공유 조명 사전(lighting prior)으로 활용될 수 있다. 유사한 조명 환경의 장면들 간에 특징 그리드를 초기화하거나 공유 학습하면 Few-shot 일반화가 가능하다.

**② 확장된 재료 다양성 처리:**
해시 그리드로 인코딩되는 공간 특징들($c_d, k_s, \rho, f$)은 재료 정보를 명시적으로 인수분해(factorize)하며, 이러한 물리 기반 분리 표현은 학습되지 않은 재료 유형으로의 **재조명(relighting) 및 소재 편집** 일반화를 지원한다.

**③ 데이터 확장을 통한 일반화:**
LIRM과 같은 후속 연구에서는 Transformer 아키텍처가 뷰 의존적 효과와 함께 형상, 재료, 복사 필드를 1초 이내에 공동 재구성하는 새로운 신경 방향 임베딩 메커니즘을 개발하였다. NDE의 방향 인코딩 개념을 대규모 사전 학습 모델에 통합하면 Zero-shot 일반화가 가능하다.

**④ 3DGS와의 결합:**
Ref-GS는 Gaussian Splatting의 지연 렌더링(deferred rendering)을 기반으로 방향 인코딩을 지연 렌더링된 표면에 적용하여, 방향과 뷰잉 각도 사이의 모호성을 효과적으로 줄이고 다양한 오픈 월드 장면에서 우수한 사실적 렌더링을 달성한다. NDE의 방향 인코딩을 3DGS에 적용하면 더 빠른 일반화가 가능하다.

**⑤ 정규화를 통한 일반화 강화:**
SpecNeRF에서는 형상-복사 모호성(shape-radiance ambiguity) 완화를 돕는 데이터 기반 기하 사전(geometry prior)을 도입하였으며, 이러한 방향 인코딩과 기하 사전의 결합이 도전적인 정반사 반사 모델링을 크게 향상시킨다. NDE도 유사한 사전 정보를 통합하면 일반화 성능이 향상될 수 있다.

---

## 8. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 연도/학회 | 핵심 기법 | 장점 | 단점 |
|---|---|---|---|---|
| **NeRF** (Mildenhall et al.) | ECCV 2020 | Positional Encoding + MLP | 최초 신경 복사 필드 | 정반사 처리 미흡, 느린 속도 |
| **Ref-NeRF** (Verbin et al.) | CVPR 2022 | IDE, 반사 방향 파라미터화 | 정반사 구조화 표현 | 대형 MLP 필요, 복잡한 반사 모델링 실패 |
| **NeRO** (Liu et al.) | SIGGRAPH 2023 | Split-sum 근사 + BRDF | 반사 물체 기하 재구성 | 느린 속도 (0.02 FPS) |
| **ENVIDR** (Liang et al.) | ICCV 2023 | 신경 환경 조명 | 글로벌 조명 고려 | SDF 표현으로 인해 자세한 기하(얇은 벽, 림 등)에서 실패 |
| **SpecNeRF** (Ma et al.) | CVPR 2024 | Gaussian 방향 인코딩 | 근거리 조명 처리 | 단안 법선 감독(monocular normal supervision) 필요 |
| **NDE (본 논문)** | CVPR 2024 | 큐브맵+볼륨 특징 그리드 + 콘 추적 | 고속(75 FPS), 소형 MLP | 정적 장면만, 오목 기하 처리 제한 |
| **3DGS-DR** (Ye et al.) | SIGGRAPH 2024 | Gaussian Splatting + 지연 반사 | 실시간, 법선 전파 | 반사 장면에만 특화, 응용 범위 제한 |
| **NeRF-Casting** (Verbin et al.) | SIGGRAPH Asia 2024 | 일관된 반사 캐스팅 | 시점 일관성 향상 | 복잡도 증가 |

SpecNeRF의 Gaussian 방향 인코딩은 근거리 조명의 공간 변화 특성을 포착하고 다양한 거칠기 계수를 가진 3D 위치에서의 사전 합성(pre-convolved) 스페큘러 색상을 효율적으로 평가하는 데 탁월하다. 이는 NDE와 상호 보완적인 관계에 있다.

---

## 9. 해당 논문이 앞으로의 연구에 미치는 영향 및 고려 사항

### 9.1 미래 연구에 미치는 영향

**① 방향 인코딩 패러다임 전환:**
NDE는 특징 그리드 기반 인코딩을 방향 영역(directional domain)으로 가져와, 학습 가능한 특징 벡터를 글로벌 환경 맵에 저장하여 원거리 광원의 반사를 표현한다. 이는 향후 모든 view-dependent NeRF 연구에서 참조될 패러다임이다.

**② 실시간 신경 렌더링의 실용화:**
NDE는 합성 및 실제 데이터셋 모두에서 최신 기술을 뛰어넘고, 소형 네트워크로 동작하여 실시간 추론을 허용하며, 실시간 웹 렌더링을 지원한다. 이는 메타버스, AR/VR, 게임 등 실용적 응용 연구를 가속화한다.

**③ Gaussian Splatting과의 융합:**
LIRM(Large Inverse Rendering Model)과 같은 후속 연구는 NDE에서 영감을 받은 신경 방향 임베딩 메커니즘으로 고품질 형상, 재료, 복사 필드를 1초 이내에 공동 재구성한다.

**④ 역 렌더링(Inverse Rendering) 연구로의 파급:**
NDE의 물리 기반 분리(diffuse/specular 분리, 재질 파라미터)는 역 렌더링 연구에서 더 정확한 BRDF 추정 및 재조명으로 이어질 수 있다.

### 9.2 향후 연구 시 고려할 점

**① 동적 장면으로의 확장:**
현재 NDE는 정적 장면에만 적용된다. 동적 물체나 변화하는 조명을 처리하기 위해서는 시간적 일관성(temporal consistency)을 유지하는 4D 방향 인코딩 확장이 필요하다.

**② 오목 기하 처리:**
SDF 기반 기하 표현은 오목 기하에서 비효율적임이 확인되었다. NeRF 기반 밀도 표현이나 Gaussian-based 표현과의 결합 연구가 필요하다.

**③ 일반화 능력(Cross-scene Generalization):**
현재 방법은 장면별 최적화를 수행한다. 여러 장면에 걸쳐 사전 학습(pre-training)된 모델에 NDE를 통합하여 Few-shot 또는 Zero-shot 렌더링이 가능하도록 하는 연구가 중요하다.

**④ 더 깊은 다중 반사(Multi-bounce) 처리:**
현재는 주로 1~2 반사 바운스만 처리한다. 다중 바운스 상호반사를 효율적으로 처리하기 위한 재귀적 콘 추적(recursive cone tracing) 또는 경로 추적(path tracing) 통합이 필요하다.

**⑤ 비등방성(Anisotropic) 반사 처리:**
NDE의 큐브맵 방향 특징은 등방성(isotropic) 분포를 가정한다. 금속 브러시 처리면 등 비등방성 반사를 처리하기 위한 방향성 있는 특징 구조가 요구된다.

**⑥ 비교 평가 기준 강화:**
실제 장면에서는 캡처 기기의 반사 제거가 어려워 공정한 평가에 어려움이 있다. 향후 연구에서는 반사 마스크나 편광 카메라 등 추가 입력을 활용하는 방향을 고려해야 한다.

---

## 📚 참고 자료 (출처)

| # | 자료 |
|---|---|
| 1 | **[주 논문]** Wu, L., Bi, S., et al., "Neural Directional Encoding for Efficient and Accurate View-Dependent Appearance Modeling," *CVPR 2024*. arXiv:2405.14847. |
| 2 | **[공식 프로젝트 페이지]** https://lwwu2.github.io/nde/ |
| 3 | **[공식 논문 PDF (CVPR OA)]** https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Neural_Directional_Encoding_CVPR_2024_paper.pdf |
| 4 | **[저자 홈페이지 PDF]** https://cseweb.ucsd.edu/~ravir/liwenwu_cvpr.pdf |
| 5 | **[IEEE Xplore]** https://ieeexplore.ieee.org/document/10657008/ |
| 6 | **[GitHub 코드]** https://github.com/lwwu2/nde |
| 7 | **[Semantic Scholar]** https://www.semanticscholar.org/paper/Neural-Directional-Encoding.../37894ad3 |
| 8 | **[비교 논문] Ref-NeRF**: Verbin et al., "Ref-NeRF: Structured View-Dependent Appearance for Neural Radiance Fields," *CVPR 2022*. |
| 9 | **[비교 논문] SpecNeRF**: Ma et al., "SpecNeRF: Gaussian Directional Encoding for Specular Reflections," *CVPR 2024*. arXiv:2312.13102. |
| 10 | **[비교 논문] NeRO**: Liu et al., "NeRO: Neural Geometry and BRDF Reconstruction of Reflective Objects from Multiview Images," *SIGGRAPH 2023*. |
| 11 | **[비교 논문] ENVIDR**: Liang et al., "ENVIDR: Implicit Differentiable Renderer with Neural Environment Lighting," *ICCV 2023*. |
| 12 | **[비교 논문] 3DGS-DR**: Ye et al., "3D Gaussian Splatting with Deferred Reflection," *SIGGRAPH 2024*. |
| 13 | **[후속 연구] Reflective Gaussian Splatting**, *ICLR 2025*. |
| 14 | **[후속 연구] NeRF-Casting**: Verbin et al., "NeRF-Casting: Improved View-Dependent Appearance with Consistent Reflections," *SIGGRAPH Asia 2024*. |

> ⚠️ **정확도 주의**: 본 답변에서 논문 내부의 세부 수식 번호(Eq. 1, 5, 9 등)는 원문 PDF의 맥락을 기반으로 재구성되었으며, 수식의 정확한 세부 파라미터는 반드시 원문 PDF(참고자료 3 또는 4)를 통해 직접 확인하시기를 권장합니다.
