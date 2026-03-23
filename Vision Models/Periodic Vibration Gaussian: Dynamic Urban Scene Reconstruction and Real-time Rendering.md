
# Periodic Vibration Gaussian: Dynamic Urban Scene Reconstruction and Real-time Rendering

---

## 1. 핵심 주장 및 주요 기여 요약

동적이고 대규모인 도시 장면을 모델링하는 것은 매우 복잡한 기하학적 구조와 시공간적으로 제약 없는 역학 때문에 어려운 과제입니다. 기존 방법들은 고수준의 아키텍처 사전지식(architectural priors)을 사용하여 정적·동적 요소를 분리함으로써, 이들 간의 시너지적 상호작용을 최적으로 포착하지 못했습니다.

이 문제를 해결하기 위해, 저자들은 **Periodic Vibration Gaussian (PVG)**이라는 통합 표현 모델을 제안합니다. PVG는 원래 정적 장면 표현을 위해 설계된 효율적인 3D Gaussian Splatting 기법에 주기적 진동 기반 시간 역학(periodic vibration-based temporal dynamics)을 도입하여, 동적 도시 장면의 다양한 객체와 요소 특성을 우아하고 균일하게 표현할 수 있습니다.

### 주요 기여 (3가지)

**(i)** 대규모 동적 도시 장면 재구성을 위한 최초의 통합 표현 모델 PVG 도입. 기존 NeRF 기반 솔루션과 달리, 3D Gaussian Splatting 패러다임을 채택하고 주기적 진동 기반 시간 역학을 통합하여 동적 장면을 우아하게 표현합니다.

**(ii)** 표현의 시간적 연속성을 강화하기 위한 새로운 temporal smoothing mechanism과, 제한 없는 도시 장면을 위한 position-aware adaptive control strategy를 개발했습니다.

**(iii)** KITTI와 Waymo 두 대규모 벤치마크에서의 광범위한 실험을 통해, PVG가 novel view synthesis에서 기존 모든 최첨단 대안들을 능가함을 입증했습니다.

특히, PVG는 수동 라벨링된 객체 바운딩 박스나 비용이 많이 드는 optical flow 추정에 의존하지 않으면서 이를 달성하며, 최선의 대안 대비 렌더링 속도에서 900배의 가속을 보여줍니다.

---

## 2. 상세 분석: 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

도시 공간(거리, 도시)의 기하학적 재구성은 디지털 맵, 자동 내비게이션, 자율주행 등의 응용에서 핵심적인 역할을 합니다. 우리 세계는 본질적으로 공간적·시간적으로 역동적이고 복잡합니다. NeRF와 같은 장면 표현 기법의 발전에도 불구하고, 이들은 주로 정적 장면에 초점을 맞추어 더 어려운 동적 요소를 간과합니다.

SUDS는 optical flow를 사용하여 객체 라벨링의 엄격한 요구사항을 완화했고, EmerNeRF는 자기지도(self-supervised) 방법으로 optical flow 의존성을 줄였습니다. 그러나 이러한 방법들은 암시적 NeRF 표현을 채택하여 훈련과 렌더링 모두에서 낮은 효율성을 겪고 있으며, 이는 대규모 장면 렌더링과 재구성에 심각한 병목현상을 초래합니다. 또한, 구성 요소를 수동으로 분리하는 것은 설계 복잡성을 도입하고 본질적 상관관계와 상호작용을 포착하는 능력을 제한합니다.

### 2.2 제안 방법 (수식 포함)

#### 2.2.1 기본 3D Gaussian Splatting 배경

기존 3DGS에서 각 Gaussian은 다음과 같이 정의됩니다:

$$G(\mathbf{x}) = e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})}$$

여기서 $\boldsymbol{\mu}$는 평균(위치), $\boldsymbol{\Sigma}$는 공분산 행렬입니다. 각 Gaussian 포인트는 $\{\boldsymbol{\mu}, \boldsymbol{\Sigma}, o, \mathbf{c}\}$ (위치, 공분산, 불투명도, 색상)로 파라미터화됩니다.

3D Gaussian Splatting 모델은 장면의 정적 포인트를 표현하는 것으로, 시간에 따른 동적 변화를 포착하는 능력이 부족하여 동적 도시 장면 모델링에 필수적인 요소가 결여되어 있습니다. 이 한계를 해결하기 위해 Periodic Vibration Gaussian 모델이 제안되었습니다.

#### 2.2.2 Periodic Vibration Gaussian (PVG)의 핵심 수식

**① Life Peak 및 시간 의존적 불투명도 (Temporal Opacity Decay)**

PVG 모델은 'life peak' 개념을 도입합니다. $\tau$로 표기되며, 이는 포인트의 시간에 따른 최대 두드러짐의 순간을 나타냅니다. 이 개념의 동기는 각 Gaussian 포인트에 고유한 수명(lifespan)을 할당하여, 그것이 언제 그리고 어느 정도로 능동적으로 기여하는지를 정의하는 것입니다.

시간 $t$에서의 진동 불투명도(vibrating opacity)는 다음과 같이 정의됩니다:

$$\tilde{o}(t) = o \cdot \exp\left(-\frac{(t - \tau)^2}{2\beta^2}\right)$$

여기서:
- $o$: 기본 불투명도 (base opacity)
- $\tau$: life peak (최대 불투명도를 가지는 시간)
- $\beta$: 시간적 수명 범위를 제어하는 스케일링 파라미터 (lifespan)

이 수식의 핵심 아이디어: **정적 객체**는 $\beta \to \infty$로 학습되어 모든 시간에서 일정한 불투명도를 가지며, **동적 객체**는 유한한 $\beta$를 가져 특정 시간 창에서만 활성화됩니다.

동적 및 정적 요소는 서로 다른 수명(lifespan)을 가진 Periodic Vibration Gaussian 포인트를 통해 통합적으로 표현됩니다. 모델은 동적과 정적을 스스로 구별하도록 학습될 수 있습니다.

**② 주기적 진동 (Periodic Vibration)**

위치의 시간 종속적 변화를 위해, 전통적인 3D Gaussian의 평균(mean)을 life peak $\tau$를 기반으로 수정합니다. 이 적응은 모델이 동적 움직임을 효과적으로 포착할 수 있도록 하여, 각 포인트가 시간적 변화에 따라 조정될 수 있게 합니다.

시간 $t$에서의 Gaussian 위치는:

$$\boldsymbol{\mu}(t) = \boldsymbol{\mu}_0 + \mathbf{v} \cdot (t - \tau)$$

여기서:
- $\boldsymbol{\mu}_0$: 기준 위치 (life peak 시점의 위치)
- $\mathbf{v}$: 학습 가능한 속도 벡터 (velocity)
- $\tau$: life peak

각 PVG 포인트는 확장된 속성 집합으로 정의됩니다:

$$\mathcal{P} = \{\boldsymbol{\mu}_0, \boldsymbol{\Sigma}, o, \mathbf{c}, \tau, \beta, \mathbf{v}\}$$

**③ Scene Flow 기반 Temporal Smoothing Mechanism**

자율주행의 동적 장면을 재구성하는 것은, 뷰와 타임스탬프 모두에서 희소한 데이터와 프레임 간 제약 없는 변이로 인해 상당한 도전을 제기합니다. 구체적으로, PVG에서 개별 포인트는 좁은 시간 창만을 포함하여 학습 데이터가 제한적이고 과적합에 대한 취약성이 증가합니다. 이를 해결하기 위해, PVG의 고유한 동적 속성을 활용하여 연속 관측 상태 간의 연결을 수립합니다.

평균 속도 메트릭은 다음과 같이 정의됩니다:

$$\bar{\mathbf{v}} = \frac{\int_{-\infty}^{\infty} \mathbf{v}(t) \cdot \tilde{o}(t) \, dt}{\int_{-\infty}^{\infty} \tilde{o}(t) \, dt}$$

이 직관은 불투명도 감쇄로 가중된 평균 속도에서 비롯되며, 명시적 해를 구하기 어렵습니다. 즉, $\lim_{\rho \to \infty} \bar{\mathbf{v}} = \mathbf{0}$으로, 정적 포인트의 속도는 자연스럽게 0이 됩니다.

Temporal smoothing loss는 인접 시간 $t$와 $t + \Delta t$에서의 렌더링 일관성을 보장합니다:

$$\mathcal{L}_{\text{smooth}} = \left\| \hat{I}(t + \Delta t) - \text{warp}(\hat{I}(t), \mathbf{F}_{t \to t+\Delta t}) \right\|_1$$

여기서 $\mathbf{F}_{t \to t+\Delta t}$는 scene flow를 기반으로 한 warping field입니다.

**④ Position-Aware Adaptive Control Strategy**

대규모 비제한적(unbounded) 도시 장면에서 Gaussian 포인트의 밀도 제어를 위해, 카메라로부터의 거리에 따라 적응적으로 densification과 pruning을 수행합니다:

$$\text{grad threshold}(d) = \text{grad base} \cdot f(d)$$

여기서 $f(d)$는 거리 $d$에 따른 적응 함수로, 원거리의 포인트에 대해서는 더 관대한 threshold를, 근거리에서는 더 엄격한 threshold를 적용합니다.

### 2.3 모델 구조 개요

PVG의 전체 파이프라인은 다음과 같이 구성됩니다:

```
입력: 다중 뷰 이미지 + LiDAR 포인트 클라우드 + 타임스탬프
    ↓
[1] LiDAR 기반 초기 포인트 클라우드 생성
    ↓
[2] PVG 포인트 파라미터화: {μ₀, Σ, o, c, τ, β, v}
    ↓
[3] 시간 t에서의 불투명도 계산: õ(t) = o · exp(-(t-τ)²/(2β²))
    ↓
[4] 시간 t에서의 위치 계산: μ(t) = μ₀ + v·(t-τ)
    ↓
[5] Differentiable Splatting Rendering
    ↓
[6] Loss 계산 + Temporal Smoothing
    ↓
[7] Position-Aware Adaptive Densification/Pruning
    ↓
출력: 재구성된 동적 장면 + 실시간 렌더링
```

### 2.4 성능 향상

Waymo Open Dataset과 KITTI 벤치마크에서의 광범위한 실험은 PVG가 동적 및 정적 장면 모두에서 재구성과 novel view synthesis에서 최첨단 대안들을 능가함을 입증했습니다.

정량적 성능 (DeSiRe-GS 논문에서 보고된 비교 데이터):
Waymo Open Dataset에서 PVG는 reconstruction PSNR 32.46dB, SSIM 0.910을 달성했으며, 이는 EmerNeRF의 PSNR 28.11dB, SSIM 0.786을 크게 상회합니다.

더불어, PVG는 훈련에서 50배, 렌더링에서 6000배의 놀라운 가속을 제공합니다.

### 2.5 한계

PVG는 동적 장면 관리에 탁월하지만, 높은 적응성 설계로 인해 정밀한 기하학적 표현에서 한계를 보입니다. 향후 작업은 기하학적 정확도 향상과 도시 장면의 복잡성을 정확하게 묘사하는 모델의 능력을 더욱 정련하는 데 초점을 맞출 것입니다.

후속 연구(Bézier Curve GS)에서 지적한 바와 같이, PVG는 주기적 진동 패턴으로 긴 궤적을 세그먼트로 이어 구성하는데, 이 주기적 진동 패턴과 불투명도 감쇄는 실세계 운동과 정확히 일치하지 않으며, 궤적을 세그먼트화하면 단일 객체의 시간에 걸친 일관성을 완전히 활용하기 어렵습니다.

UrbanGS 논문에서는 PVG와 같은 방법이 정적·동적 요소 간의 명시적 구분이 부족하여, 정지 객체에 대한 불필요한 업데이트가 발생하고 이로 인해 재구성 품질이 저하될 수 있다고 지적했습니다.

DeSiRe-GS에서는 PVG와 같은 GS 기반 방법이 NeRF 기반 방법보다 이미지 렌더링에서는 우수하지만, 명시적 GS 방법이 이미지에 과적합하여 깊이 렌더링에서는 성능이 저하되는 경향이 있다고 보고합니다.

---

## 3. 모델의 일반화 성능 향상 가능성

PVG의 일반화 성능과 관련하여, 다음과 같은 핵심적인 설계 요소들이 기여하고 있으며, 동시에 개선 가능성이 존재합니다:

### 3.1 일반화에 기여하는 설계 요소

**① 통합 표현의 장점:**
PVG는 정적·동적 요소를 하나의 단일 수식(single formulation)으로 통합하여 표현합니다. 이는 별도의 분기 아키텍처 없이도 다양한 장면에 적용될 수 있는 범용성을 제공합니다.

**② 자기지도 학습:**
PVG는 주기적 진동 기반 Gaussian 속성을 도입하여, 3D 바운딩 박스 없이 자기지도를 통해 최적화됩니다. 각 Gaussian은 진동 방향, 수명(life span), life peak를 포함하는 최적화 가능한 속성을 통해 시간에 따른 동적 변화를 모델링합니다. 이러한 annotation-free 접근은 새로운 데이터셋에 대한 일반화를 크게 용이하게 합니다.

**③ Temporal Smoothing의 정규화 효과:**
희소한 학습 데이터로부터 시간적으로 일관된 표현 학습을 향상시키기 위해, 새로운 flow 기반 temporal smoothing mechanism과 position-aware adaptive control strategy를 도입합니다. 이는 과적합을 방지하고 시간적 일관성을 보장하는 정규화 역할을 합니다.

### 3.2 일반화 성능의 한계 및 향상 방향

**한계 1: 기하학적 정밀도의 부족**

PVG의 유연한 설계(매 포인트에 독립적인 $\tau, \beta, \mathbf{v}$)는 강력한 표현력을 제공하지만, 동시에 과도한 자유도로 인해 기하학적 정밀도가 떨어질 수 있습니다. 이를 개선하려면:

$$\mathcal{L}_{\text{geo}} = \lambda_{\text{depth}} \mathcal{L}_{\text{depth}} + \lambda_{\text{normal}} \mathcal{L}_{\text{normal}}$$

형태의 기하학적 감독 손실을 추가하여, depth 및 normal 일관성을 강화할 수 있습니다.

**한계 2: 장면 유형 전이(Domain Transfer)**

PVG는 주로 Waymo/KITTI의 도로 주행 장면에서 평가되었으며, 다른 유형의 도시 환경(실내, 보행자 밀집 지역 등)에의 일반화는 검증되지 않았습니다.

**한계 3: 시맨틱 정보 미활용**

UrbanGS에서 제안된 것처럼, 각 Gaussian의 시맨틱 속성을 활용하여 적응적으로 속성을 조정하면, 정적 요소는 시간에 걸쳐 변하지 않게 유지하고, 저텍스처 영역에서의 일관성을 강화하며, 잠재적으로 동적인 객체를 4D 표현을 통해 직관적으로 포착할 수 있습니다.

### 3.3 일반화 향상을 위한 구체적 제안

| 개선 방향 | 접근 방법 | 기대 효과 |
|-----------|----------|-----------|
| 시맨틱 가이던스 | 2D 시맨틱 맵 통합 | 정적/동적 구분 정확도 향상 |
| 기하학적 정규화 | Depth/Normal supervision | 깊이 렌더링 품질 개선 |
| 궤적 모델링 개선 | Bézier 곡선 등 물리적 궤적 모델 | 장기 시간 일관성 향상 |
| 다중 해상도 학습 | Level-of-Detail 전략 | 대규모 장면 확장성 |
| Cross-dataset 학습 | 도메인 적응 기법 | 새로운 환경에 대한 전이 능력 |

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 학술적·산업적 영향

**① 패러다임 전환의 촉매:**
PVG는 동적 장면 표현에서 NeRF 기반 접근에서 3DGS 기반 접근으로의 패러다임 전환을 가속화했습니다. "정적/동적 분리" 대신 "통합 표현"이라는 새로운 방향을 제시하여, 후속 연구(UrbanGS, SplatFlow, DeSiRe-GS 등)에 직접적 영감을 주었습니다.

**② 자율주행 시뮬레이션:**
도시 공간의 기하학적 재구성은 디지털 맵, 자동 내비게이션, 자율주행 등의 응용에서 핵심적인 역할을 합니다. PVG의 실시간 렌더링 능력은 자율주행 시뮬레이터의 실용성을 크게 높입니다.

**③ Annotation-Free 접근의 확산:**
PVG는 수동 라벨링된 3D 바운딩 박스에 의존하지 않고 동적 주행 장면을 모델링하는 Periodic Vibration Gaussian 메커니즘을 제안했습니다. 이는 대규모 데이터셋 활용을 실용적으로 만들었습니다.

### 4.2 후속 연구 시 고려할 핵심 사항

**① 물리적 일관성 강화:**
주기적 진동 패턴과 불투명도 감쇄가 실세계 운동과 정확히 일치하지 않는 문제를 해결하기 위해, 물리 기반 운동 모델(예: 운동학적 제약 조건)의 통합이 필요합니다.

**② 장기 시간 일관성:**
단일 객체의 전체 궤적에 걸친 일관성을 보장하는 것이 중요하며, 이를 위해 Bézier 곡선이나 스플라인 기반 궤적 모델링이 유효한 대안이 될 수 있습니다.

**③ 다중 모달리티 융합:**
LiDAR, 카메라, IMU 데이터의 더 긴밀한 융합을 통해 기하학적 정밀도와 시간적 일관성을 동시에 향상시킬 수 있습니다.

**④ 확장성 (Scalability):**
도시 규모의 장면(수 km 이상)에서의 효율적 학습과 렌더링을 위한 계층적 분할 전략이 필요합니다.

**⑤ 편집 가능성 (Editability):**
PVG는 동적 장면을 재구성할 뿐 아니라 동적 구성 요소를 효율적으로 분리하여 동적 장면 요소의 제거와 같은 유연한 조작을 가능하게 합니다. 향후 더 정교한 편집 기능(객체 삽입, 경로 변경 등)의 연구가 기대됩니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 연도 | 표현 방식 | 동적 모델링 | 감독 | 주요 특징 | 렌더링 속도 |
|------|------|-----------|------------|------|-----------|------------|
| **NeRF** (Mildenhall et al.) | 2020 | Implicit (MLP) | 없음 (정적) | Multi-view | 암시적 체적 렌더링 | 매우 느림 |
| **NSG** (Ost et al.) | 2021 | Neural Scene Graph | Scene graph decomposition | 3D Bbox | 장면 그래프 기반 분해 | 느림 |
| **SUDS** (Turki et al.) | 2023 | Multi-branch NeRF | 3-branch 분리 | Optical Flow + Labels | 정적/동적/환경 분리 | 느림 |
| **EmerNeRF** (Yang et al.) | 2023 | Hash grid NeRF | Self-supervised flow | Self-supervised | 시공간 분해 자기지도 | 0.05 FPS |
| **3DGS** (Kerbl et al.) | 2023 | Explicit (Gaussian) | 없음 (정적) | Multi-view | 실시간 렌더링 | 매우 빠름 |
| **4D-GS** (Wu et al.) | 2024 | 4D Gaussian | Hexplane + MLP | Multi-view | 실시간 동적 렌더링 | ~82 FPS |
| **PVG** (Chen et al.) | 2023 | Periodic Vibration GS | 주기적 진동 + opacity decay | Self-supervised | **통합 표현, annotation-free** | **~50-59 FPS** |
| **Street Gaussians** (Yan et al.) | 2024 | Explicit GS + Tracking | 추적 포즈 + 4D SH | 3D Bbox + Tracker | 객체별 분리 재구성 | ~135 FPS |
| **UrbanGS/Urban4D** | 2024 | Semantic-guided 4DGS | 시맨틱 분해 + MLP | 2D Semantic | 시맨틱 가이드 분해 | 빠름 |
| **SplatFlow** (Sun et al.) | 2025 | Neural Motion Flow + GS | Continuous flow field | Self-supervised | 연속적 모션 플로우 필드 | 빠름 |
| **DeSiRe-GS** (Peng et al.) | 2025 | 4D Street GS | 정적-동적 분해 + 표면 재구성 | Self-supervised | 깊이/법선 정밀도 향상 | ~36-41 FPS |
| **Bézier Curve GS** | 2025 | Bézier trajectory GS | 학습가능 Bézier 곡선 | Annotation 활용 | 궤적 오류 자동 보정 | 빠름 |

### 핵심 비교 포인트

**PVG vs. EmerNeRF:**
Waymo에서 PVG(PSNR 32.46)는 EmerNeRF(PSNR 28.11)를 크게 상회하며, 렌더링 속도에서도 압도적 우위를 보입니다.

**PVG vs. Street Gaussians:**
Street Gaussians는 동적 도시 장면을 시맨틱 로짓과 3D Gaussian이 장착된 포인트 클라우드 세트로 표현하며, 전경 차량의 동역학을 최적화 가능한 추적 포즈와 4D 구면 조화 모델로 모델링합니다. 이는 PVG보다 빠르지만(135 FPS), 3D Bbox 및 tracker에 의존합니다.

**PVG vs. Urban4D (UrbanGS):**
Urban4D는 novel view synthesis에서 26.56 PSNR, 0.814 SSIM을 달성하여 PVG를 0.64 PSNR, 0.016 SSIM만큼 능가합니다. 모든 방법이 novel view synthesis에서 image reconstruction 대비 성능 저하를 겪지만, Urban4D가 가장 적은 성능 저하를 보여 더 나은 장면 이해와 기하학 모델링을 시사합니다.

**PVG vs. SplatFlow:**
SplatFlow는 Street GS가 요구하는 3D Bbox의 필요성을 제거하고, PVG 대비 렌더링 품질을 향상시킵니다.

**PVG vs. DeSiRe-GS:**
PVG와 같은 GS 기반 방법은 EmerNeRF 같은 NeRF 기반 방법보다 이미지 렌더링에서 일반적으로 우수하지만, 명시적 GS 방법은 이미지에 과적합하여 깊이 렌더링에서 저조한 성능을 보이는 경향이 있습니다. DeSiRe-GS는 이 문제를 표면 재구성 기법으로 해결합니다.

---

## 참고 자료 및 출처

1. **Chen, Y., Gu, C., Jiang, J., Zhu, X., & Zhang, L.** (2023). "Periodic Vibration Gaussian: Dynamic Urban Scene Reconstruction and Real-time Rendering." *arXiv:2311.18561* → IJCV 2026. [https://arxiv.org/abs/2311.18561](https://arxiv.org/abs/2311.18561)

2. **PVG 공식 프로젝트 페이지:** [https://fudan-zvg.github.io/PVG/](https://fudan-zvg.github.io/PVG/)

3. **PVG GitHub 공식 구현:** [https://github.com/fudan-zvg/PVG](https://github.com/fudan-zvg/PVG)

4. **Springer IJCV 출판:** [https://link.springer.com/article/10.1007/s11263-026-02740-3](https://link.springer.com/article/10.1007/s11263-026-02740-3)

5. **Yan, Y. et al.** (2024). "Street Gaussians: Modeling Dynamic Urban Scenes with Gaussian Splatting." *ECCV 2024.* [https://arxiv.org/abs/2401.01339](https://arxiv.org/abs/2401.01339)

6. **UrbanGS (Urban4D):** "Semantic-Guided Gaussian Splatting for Urban Scene Reconstruction." [https://arxiv.org/html/2412.03473](https://arxiv.org/html/2412.03473)

7. **Sun et al.** (2025). "SplatFlow: Self-Supervised Dynamic Gaussian Splatting in Neural Motion Flow Field." *CVPR 2025.* [https://openaccess.thecvf.com/content/CVPR2025/](https://openaccess.thecvf.com/content/CVPR2025/)

8. **Peng et al.** (2025). "DeSiRe-GS: 4D Street Gaussians for Static-Dynamic Decomposition and Surface Reconstruction for Urban Driving Scenes." *CVPR 2025.* [https://arxiv.org/html/2411.11921](https://arxiv.org/html/2411.11921)

9. **Bézier Curve Gaussian Splatting** (2025). "Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting." [https://arxiv.org/html/2506.22099v2](https://arxiv.org/html/2506.22099v2)

10. **Scene reconstruction techniques for autonomous driving: a review of 3D Gaussian splatting.** *Artificial Intelligence Review*, Springer (2024). [https://link.springer.com/article/10.1007/s10462-024-10955-4](https://link.springer.com/article/10.1007/s10462-024-10955-4)

11. **Semantic Scholar - PVG 인용 및 관련 연구:** [https://www.semanticscholar.org/paper/3d3b85d1d1829bf951d5368670fb56ac096760c2](https://www.semanticscholar.org/paper/3d3b85d1d1829bf951d5368670fb56ac096760c2)

---

> **⚠️ 참고 사항:** 본 분석에서 제시한 수식은 논문의 핵심 아이디어를 수학적으로 표현한 것입니다. 일부 수식(특히 temporal smoothing loss와 position-aware adaptive control의 세부 수식)은 논문의 공개된 정보와 공식 코드를 바탕으로 재구성한 것이며, 원문의 정확한 notation과 약간 차이가 있을 수 있습니다. 정확한 수식은 원본 논문(arXiv:2311.18561)의 PDF를 직접 참조하시기를 권장합니다.
