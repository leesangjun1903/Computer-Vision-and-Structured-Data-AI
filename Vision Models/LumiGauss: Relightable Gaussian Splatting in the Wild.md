
# LumiGauss: Relightable Gaussian Splatting in the Wild

> **논문 정보**
> - **제목**: LumiGauss: Relightable Gaussian Splatting in the Wild
> - **저자**: Joanna Kaleta, Kacper Kania, Tomasz Trzcinski, Marek Kowalski
> - **게재**: WACV 2025
> - **arXiv**: [2408.04474](https://arxiv.org/abs/2408.04474)
> - **코드**: https://github.com/joaxkal/lumigauss

---

## 1. 핵심 주장 및 주요 기여 요약

비구속적(unconstrained) 사진 컬렉션에서 조명과 기하 구조를 분리하는 것은 매우 어려운 문제이며, 기존의 많은 연구들은 출력 충실도를 희생시키는 방식으로 이를 해결하려 했다.

LumiGauss는 이 문제를 새로운 방식으로 접근한다.

2D Gaussian Splatting을 통해 장면의 3D 재구성과 환경 조명을 동시에 다루며, 고품질 장면 재구성과 새로운 환경 맵 하에서 사실적인 조명 합성을 가능하게 한다. 또한 구면 조화 함수(spherical harmonics) 특성을 활용하여 야외 장면에서 흔히 나타나는 그림자의 품질을 향상시키는 방법을 제안하며, 게임 엔진과의 원활한 통합 및 고속 사전 계산된 라디언스 전송(PRT)의 사용을 가능하게 한다.

### 주요 기여(Contributions) 요약

① 야생(in-the-wild) 환경의 역 그래픽스 파이프라인을 위해 2D Gaussian Splatting을 재목적화하여 고품질 알베도와 환경 맵을 복원하고, ② 각 2D 스플랫에 대해 구면 조화 함수로 표현되는 라디언스 전달 함수(radiance transfer function)를 학습하여 그림자 모델링을 가능하게 하며, ③ 재구성된 환경 맵을 그래픽 엔진 내 임의의 객체를 리라이팅하는 데 활용할 수 있음을 시연한다.

---

## 2. 해결 문제 · 제안 방법(수식 포함) · 모델 구조 · 성능 및 한계

### 2-1. 해결하고자 하는 문제

LumiGauss가 다루는 핵심 문제는 비구속 사진 컬렉션을 이용해 야외 장면에서 조명과 기하 구조를 분리하는 것이다. 이 과제는 계산 사진학과 게임 디자인에서 중대한 의미를 지니며, 현실적인 3D 자산 생성에는 방대한 수동 작업이 요구된다.

3DGS는 고품질 이미지를 생성하지만 내재적 표면 표현이 노이즈가 많아 리라이팅 시나리오에서의 적용성이 제한된다. 이에 2D Gaussian을 사용하면 정확한 2D 서펠 투영 덕분에 부드럽고 일관된 메시를 만들 수 있다.

### 2-2. 제안하는 방법 및 수식

#### (a) 기본 렌더링 방정식

컴퓨터 그래픽스에서 색상은 알베도와 환경 맵으로 분리된다:

$$
\mathbf{c} = \rho \cdot \int_{\Omega} L_i(\boldsymbol{\omega}) \, D(\boldsymbol{\omega}) \, (\mathbf{n} \cdot \boldsymbol{\omega}) \, d\boldsymbol{\omega}
$$

여기서 $L_i(\cdot)$는 조명의 강도와 색상을 나타내며, $D(\cdot)$는 그림자 또는 다른 표면으로부터의 반사를 고려하는 항이다.

#### (b) 구면 조화 함수(SH) 기반 조명 표현

조명은 환경 맵과 라디언스 전달 함수의 조합으로 모델링된다. 환경 맵은 주어진 서펠(surfel)을 어느 방향에서 어떻게 조명하는지를 나타내며, 이 두 요소 모두 구면 조화 함수로 표현된다. 이 접근법은 그림자 모델링을 가능하게 하며, 다른 물체에서 반사된 빛의 표현 가능성도 가진다.

SH 계수 기반 환경 맵 $L$과 전달 함수 $\mathbf{d}_k$의 내적으로 irradiance를 근사할 수 있다:

$$
E_k = \sum_{l=0}^{L} \sum_{m=-l}^{l} l_m \cdot d_m^{(k)}
$$

- $l_m$: 환경 맵의 SH 계수
- $d_m^{(k)}$: $k$번째 Gaussian의 전달 함수 SH 계수

#### (c) Unshadowed 모델 (그림자 없음)

각 Gaussian $G_k$의 색상은 알베도 $\rho_k$, 법선 $\mathbf{n}_k$, 환경 SH 계수를 활용하여 다음과 같이 계산된다:

$$
c_k = \rho_k \cdot \sum_{l,m} l_m \cdot Y_{lm}(\mathbf{n}_k)
$$

리라이팅을 위해 접근법은 학습 이미지마다 2차 구면 조화 함수로 표현된 환경 조명을 예측하고, 비그림자 시나리오에서는 표면 법선 방향으로 반구에 걸쳐 빛을 적분하는 방정식을 따른다.

#### (d) Shadowed 모델 (그림자 있음)

그림자를 효과적으로 포착하기 위해 Gaussian의 출력 색상을 재정의한다. 전달 함수 $\mathbf{d}_k$를 통해 음영을 추가한다:

$$
\tilde{c}_k = \rho_k \cdot \sum_{l,m} l_m \cdot d_m^{(k)} \cdot Y_{lm}(\mathbf{n}_k)
$$

라디언스 전달 $D_k$ 함수는 0에서 1 사이의 범위로 제한되며, 0은 완전한 그림자, 1은 빛에 완전히 노출된 상태를 의미한다.

#### (e) 그림자 시각화

LumiGauss는 그림자를 명시적으로 예측하지 않기 때문에, 비그림자(Eq. 10)와 그림자(Eq. 11) 아이러디언스(irradiance) 출력의 그레이스케일 차이로 그림자 효과를 시각화하며, 환경 맵의 어두운 조명과 그림자를 구별한다.

수식으로 표현하면:

$$
\text{Shadow Map} = \text{Grayscale}(E_{\text{unshadowed}} - E_{\text{shadowed}})
$$

#### (f) 물리 기반 제약 손실 함수

2DGS의 정규화는 Gaussian을 표면에 가깝게 유지하고 국소적으로 부드럽게 하며, 이는 리라이팅 시나리오에서 매우 중요하다. 이에 더해 LumiGauss는 최적화가 퇴화된(degenerate), 리라이팅 불가능한 상태에 도달하지 않도록 물리적 광 특성에 기반한 새로운 손실 항을 제안한다.

그림자가 unshadowed보다 밝아지지 않도록 강제하는 손실 함수:

$$
\mathcal{L}_{\text{shadow}} = \sum_k \max(0, \, \tilde{c}_k - c_k)
$$

전체 손실 함수는 다음과 같은 형태:

$$
\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{render}} + \lambda_2 \mathcal{L}_{\text{shadow}} + \lambda_3 \mathcal{L}_{\text{reg}} + \cdots
$$

### 2-3. 모델 구조

LumiGauss의 파이프라인은 비구속 사진 컬렉션에서 리라이팅 가능한 2D Gaussian 표현을 학습한다. $k$개의 각 Gaussian은 법선 $\mathbf{n}_k$, 알베도 $\rho_k$, 학습 가능한 전달 함수 $d_k$를 보유하며, 그림자/비그림자의 두 가지 모드로 Gaussian을 합성한다.

기존 2DGS와 달리 각 Gaussian에 알베도 $\rho$와 전달 함수 $\mathbf{d}$의 SH 파라미터를 추가하며, 다양한 조명 조건으로 특성화되는 in-the-wild 이미지에는 환경을 인코딩하는 학습 가능한 잠재 코드 $\{e_c\}$를 부여한다.

환경 맵 예측을 위해 크기 64의 완전 연결 레이어 3개로 구성된 MLP를 사용하며, 모든 모델은 50,000번 반복으로 학습된다. 첫 번째 학습 단계는 30,000번, MLP 및 임베딩의 학습률은 0.002로 설정하며 첫 번째 단계 이후 0.0001로 감소한다. Gaussian 구면 조화 함수의 학습률은 0.002이다.

**모델 구조 요약 다이어그램:**

```
[비구속 사진 컬렉션 (in-the-wild)]
         ↓
[COLMAP SfM → 초기 포인트 클라우드]
         ↓
[2D Gaussian Splatting (2DGS)]
    각 Gaussian Gk:
    ├── 법선 nk
    ├── 알베도 ρk
    └── 전달 함수 dk (SH 파라미터)
         ↓
[MLP (3 FC layers, size 64)] → 환경 맵 (SH 표현)
[잠재 코드 ec] → 각 이미지의 조명 조건 인코딩
         ↓
  ┌──────────────────┐
  │  Unshadowed 렌더링 │  ← ck = ρk · Σ lm · Ylm(nk)
  └──────────────────┘
  ┌──────────────────┐
  │  Shadowed 렌더링  │  ← c̃k = ρk · Σ lm · dm(k) · Ylm(nk)
  └──────────────────┘
         ↓
[물리 기반 제약 손실] + [2DGS 정규화 손실]
         ↓
[Novel View Synthesis + Relighting + Game Engine 통합]
```

### 2-4. 성능 향상

NeRF-OSR 데이터셋에서 기준 방법 대비 우수한 성능을 검증하였으며, LumiGauss는 학습에 사용되지 않은 새로운 환경 맵에 대해서도 사실적인 이미지를 합성할 수 있다.

SSIM 수치를 보면, 비그림자 및 그림자 재현 모두에서 SSIM 0.800을 달성하여 합성 이미지의 구조적 일관성을 유지하는 데 탁월한 성능을 보인다.

본 방법은 그럴듯한 리라이팅 결과를 달성하면서 학습과 추론 모두에서 기준 방법들 대비 수십 배 빠른 속도를 달성한다.

이 방법은 렌더링 시 신경망 실행이 필요 없는 NeRF-OSR의 방식과 달리, 그래픽 엔진과의 통합을 간소화하고 실시간 응용을 용이하게 한다.

### 2-5. 한계

LumiGauss의 한계로는, 경도(hard)의 그림자가 빈번히 나타나는 시나리오에서 표면 알베도와 법선이 그림자를 시뮬레이션하려는 경향이 있으며, 이것이 그림자 학습을 방해하고, 여러 훈련 이미지에서 그림자가 보일 때 표면 법선의 정확한 표현을 방해할 수 있다.

환경 조명 및 그림자에 대한 사전 지식(prior)을 통합하면 분리 및 광 전송 모델링을 더욱 향상시킬 수 있다. 또한 확산(diffuse) 알베도만을 가정하기 때문에, 창문과 같은 반사 표면에서 그림자가 부자연스럽게 나타날 수 있다. 별도의 배경 최적화는 넓은 하늘 영역이 있는 장면 합성을 향상시킬 수 있다.

LumiGauss의 전방향(omnidirectional) 조명 가정은 선명한 그림자와 같은 고주파 효과를 제한하며, PRT 오버피팅이 종종 구워진(baked-in) 아티팩트를 초래한다. 또한 학습 가능한 이미지 기반 큐브맵의 메모리 집약적 특성은 야외 촬영에서 일반적인 다수의 조명 조건 처리에 비효율적이다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 현재 일반화 능력

LumiGauss의 출력은 학습 시 사용된 것 이외의 환경 맵을 사용한 novel view synthesis와 리라이팅에 활용될 수 있다.

구면 조화 함수의 특성 덕분에, 게임 엔진에 적용했을 때 빠른 사전 계산된 라디언스 전송과 쉽게 결합할 수 있다.

2DGS를 적용하여 물체의 표면을 정확하게 재구성하고, 올바르게 광 특성을 분리하는 학습 구성 요소를 활용함으로써 기준 방법보다 더 나은 재구성 결과를 달성한다.

### 3-2. 일반화 성능 향상의 가능성과 방향

미래 발전 방향으로, 더 복잡한 조명 상호작용과 세밀한 그림자 세부 사항을 캡처하기 위한 고차 구면 조화 함수의 도입, 동적으로 변화하는 장면과 비정적 환경에 적응하기 위한 동적 장면 적응(Dynamic Scene Adaptation), 그리고 실시간 조정과 장면 편집을 위한 상호작용성과 사용성 향상이 제안된다.

환경 조명과 그림자에 대한 사전 지식(prior)을 통합하면 분리(disentanglement)와 광 전송 모델링을 더욱 향상시킬 수 있다. NeuSky와 같은 접근법이 이 방향을 보여준다.

**일반화 성능 향상을 위한 구체적 전략:**

| 전략 | 내용 | 기대 효과 |
|---|---|---|
| 고차 SH 도입 | 2차 → 더 높은 차수로 확장 | 고주파 조명 효과, 선명한 그림자 |
| 하늘/전경 분리 표현 | 하늘 Gaussian을 독립적으로 모델링 | 깊이 추정 개선, 하늘-전경 경계 렌더링 향상 |
| 반사(specular) 표면 지원 | Diffuse 가정 완화, BRDF 확장 | 창문 등 반사 표면의 리라이팅 정확도 향상 |
| 동적 장면 지원 | 시간 축 잠재 코드 확장 | 계절, 날씨 변화에 강건 |
| 사전 학습 모델 활용 | 환경 조명 prior 사전 학습 | 한정된 데이터에서도 일반화 강화 |

---

## 4. 관련 최신 연구 비교 분석 및 앞으로의 영향

### 4-1. 2020년 이후 관련 연구 비교

| 방법 | 기반 | 조명 표현 | 그림자 | 속도 | 특징 |
|---|---|---|---|---|---|
| **NeRF-OSR** (2022) | NeRF | SH | 제한적 | 느림 | 야외 리라이팅 NeRF 선구자 |
| **SR-TensoRF** | TensoRF | 잠재 벡터 | - | 중간 | 동기화된 시간 활용 |
| **NeuSky** (2024) | NeRF | 사전 학습 모델 | ✓ | 매우 느림(14h/scene) | 강력한 sky prior 활용 |
| **LumiGauss** (2024) | **2DGS** | **SH (2차)** | **✓ (PRT)** | **빠름** | 게임 엔진 통합, WACV2025 |
| **R3GW** (2025) | 3DGS | SH + Split-sum | 제한적 | 빠름(2h/scene) | Cook-Torrance BRDF, 하늘-전경 분리 |
| **ROS-GS** (2025) | 2DGS | 하이브리드(Sun+Sky) | ✓ (Mesh-based) | 빠름 | 2단계 파이프라인, 고주파 조명 |
| **GaRe** (2025) | 3DGS | Sun+Sky+간접광 | ✓ (Ray-tracing) | - | Ray-trace 기반 가시성 쿼리 |

LumiGauss는 관련 NeRF 기반 기술보다 빠른 학습과 렌더링을 제공하면서 고품질 재구성 및 리라이팅 결과를 달성하지만, 확산(diffuse) 반사만 모델링하고 하늘과 전경을 단일 Gaussian 컬렉션으로 표현하는 한계가 있다.

NeRF 기반 접근법과 비교하면, R3GW는 NeRF-OSR 및 SR-TensoRF보다 높은 평균 PSNR과 SSIM을 달성하지만 PSNR에서는 NeuSky에 약간 미치지 못한다. 그러나 NeuSky는 장면당 약 14시간의 학습이 필요한 반면, R3GW는 약 2시간에 불과하다.

ROS-GS는 LumiGauss와 달리 분리된 알베도 맵에서 더 자연스러운 색상과 적은 그림자 아티팩트를 보이며, 고주파 조명 효과에서 더 우수한 조명 표현력을 보인다.

### 4-2. 앞으로의 연구에 미치는 영향

**① 역 렌더링(Inverse Rendering) 분야 가속화**

LumiGauss는 in-the-wild 이미지에서 역 렌더링 과정을 통해 출력 충실도를 희생하지 않고 고품질 장면 특성을 재구성하는 방향을 개척했다는 점에서 의의가 있다.

**② 게임 엔진 및 실시간 응용의 새로운 기준 제시**

LumiGauss는 사전 계산된 라디언스 전송을 활용하여 그래픽 엔진과의 원활한 통합을 가능하게 하고, 정확한 그림자 모델링을 통해 생성 렌더링의 사실감을 크게 향상시킨다.

**③ 후속 연구의 직접적 기반**

LumiGauss가 제안한 확산 반사 모델과 2차 SH 기반 환경 조명 표현 방식은 이후 연구들이 직접 비교 기준(baseline)으로 삼는 표준이 되었다.

### 4-3. 앞으로 연구 시 고려할 점

1. **더 높은 차수의 SH 혹은 대안 조명 표현**
   고차 구면 조화 함수를 통합하면 더 복잡한 조명 상호작용을 캡처할 수 있지만, 잠재적인 계산 트레이드오프를 신중히 고려해야 한다.

2. **하늘-전경 분리 표현 도입**
   하늘과 전경의 결합 표현은 물질 특성이 없는 하늘 Gaussian에 material을 할당하는 문제를 일으킬 수 있으므로, 이를 분리하는 설계가 필요하다.

3. **Specular/반사 BRDF 지원**
   현재의 확산 알베도 가정은 대부분의 야외 사례에 유효하지만, 창문과 같은 반사 표면에서 그림자가 부자연스럽게 나타날 수 있어 반사 BRDF의 통합이 필요하다.

4. **동적/transient 객체 처리**
   야외 in-the-wild 데이터에는 보행자, 차량 등의 동적 객체가 포함되어 있어, 이를 효과적으로 처리하는 메커니즘(예: transient mask 학습)이 필요하다.

5. **일반화를 위한 사전 학습 모델 활용**
   환경 조명 및 그림자에 대한 사전 지식의 통합은 분리 및 광 전송 모델링을 더욱 향상시킬 수 있으며, NeuSky와 같은 접근법이 이미 이 방향을 보여주고 있다.

6. **벤치마크 데이터셋 다양화**
   현재는 NeRF-OSR 데이터셋 위주로 평가가 이루어지고 있어, 더 다양한 야외 환경(열대우림, 야간 등)에서의 일반화 능력 검증이 필요하다.

---

## 📚 참고 자료 및 출처

| # | 제목/출처 | URL |
|---|---|---|
| 1 | **LumiGauss: Relightable Gaussian Splatting in the Wild** (arXiv) | https://arxiv.org/abs/2408.04474 |
| 2 | **LumiGauss HTML 논문 전문** (arXiv HTML v2) | https://arxiv.org/html/2408.04474v2 |
| 3 | **LumiGauss 공식 GitHub** (WACV2025) | https://github.com/joaxkal/lumigauss |
| 4 | **LumiGauss 프로젝트 페이지** | https://lumigauss.github.io/ |
| 5 | **LumiGauss WACV2025 CVF 공식 페이지** | https://openaccess.thecvf.com/content/WACV2025/html/Kaleta_LumiGauss_... |
| 6 | **LumiGauss IEEE Xplore** | https://ieeexplore.ieee.org/document/10943575/ |
| 7 | **ResearchGate: LumiGauss 초기 버전** | https://www.researchgate.net/publication/382971305 |
| 8 | **ar5iv: LumiGauss High-Fidelity Outdoor Relighting** | https://ar5iv.labs.arxiv.org/html/2408.04474 |
| 9 | **Sano Centre 연구 소개** | https://sano.science/research/lumigauss-relightable-gaussian-splatting-in-the-wild/ |
| 10 | **AI Models FYI 분석** | https://www.aimodels.fyi/papers/arxiv/lumigauss-relightable-gaussian-splatting-wild |
| 11 | **Emergent Mind 분석** | https://www.emergentmind.com/papers/2408.04474 |
| 12 | **R3GW: Relightable 3D Gaussians for Outdoor Scenes in the Wild** (arXiv) | https://arxiv.org/html/2603.02801 |
| 13 | **ROS-GS: Relightable Outdoor Scenes With Gaussian Splatting** (arXiv) | https://arxiv.org/html/2509.11275 |
| 14 | **GaRe: Relightable 3D Gaussian Splatting for Outdoor Scenes** (arXiv PDF) | https://arxiv.org/pdf/2507.20512 |
