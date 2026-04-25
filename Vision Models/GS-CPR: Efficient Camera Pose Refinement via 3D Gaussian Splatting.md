
# GS-CPR: Efficient Camera Pose Refinement via 3D Gaussian Splatting

> **논문 정보**
> - **제목:** GS-CPR: Efficient Camera Pose Refinement via 3D Gaussian Splatting
> - **저자:** Changkun Liu, Shuai Chen, Yash Sanjay Bhalgat, Siyan Hu, Ming Cheng, Zirui Wang, Victor Adrian Prisacariu, Tristan Braud
> - **학회:** ICLR 2025 (The Thirteenth International Conference on Learning Representations)
> - **arXiv:** [2408.11085](https://arxiv.org/abs/2408.11085)
> - **공식 코드:** [GitHub - XRIM-Lab/GS-CPR](https://github.com/XRIM-Lab/GS-CPR)

---

## 1. 핵심 주장 및 주요 기여 요약

### 🔑 핵심 주장

GS-CPR은 3D Gaussian Splatting(3DGS)을 장면 표현으로 활용한 새로운 테스트 타임(test-time) 카메라 포즈 정제(Camera Pose Refinement, CPR) 프레임워크입니다.

느린 수렴 속도, 제한적 정확도, 커스텀 feature descriptor 훈련의 필요성이라는 기존의 문제점들을 해결하기 위해 GS-CPR이 제안되었습니다.

### 📌 주요 기여

| 기여 항목 | 내용 |
|---|---|
| 장면 표현 | 3DGS 기반 고품질 렌더링 |
| 매칭 방식 | MASt3R 파운데이션 모델 활용 |
| 노출 적응 | ACT(Affine Color Transformation) 모듈 |
| 효율성 | One-shot 포즈 정제 |

구체적으로:

1. GS-CPR은 3DGS를 장면 표현에 활용하여 고품질 고속 Novel View Synthesis(NVS) 기능으로 이미지와 깊이 맵을 렌더링하고, 초기 포즈 추정치를 기반으로 쿼리 이미지와 렌더링 이미지 간 2D-3D 대응 관계를 효율적으로 구축합니다.

2. GS-CPR은 feature extractor나 descriptor를 별도로 훈련할 필요 없이 RGB 이미지에 직접 작동하며, 3D 파운데이션 모델 MASt3R을 정밀 2D 매칭에 활용합니다.

3. 야외의 도전적인 환경에서 모델의 강인성을 향상시키기 위해 3DGS 프레임워크 내에 노출 적응(exposure-adaptive) 모듈을 통합했습니다.

4. GS-CPR은 단일 RGB 쿼리와 APR/SCR 메서드의 대략적인 초기 포즈 추정치만으로 one-shot 포즈 정제를 가능하게 하며, 실내외 다양한 시각적 위치 추정 벤치마크에서 기존 NeRF 기반 최적화 방법보다 정확도와 런타임 모두에서 우수한 성능을 보입니다.

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

비주얼 로컬라이제이션(Visual Localization)에서 카메라 포즈 추정은 핵심 과제입니다. 기존의 접근법인 APR(Absolute Pose Regression)과 SCR(Scene Coordinate Regression)은 실용적이지만 포즈 정확도에 한계가 있으며, 이를 정제하기 위한 방법들도 다음과 같은 문제를 가지고 있었습니다:

대부분의 정제 방법은 특정 descriptor나 전용 네트워크에 의존하는 2D-3D 대응 관계를 사용하며, 다른 descriptor를 사용하려면 장면 재구성 혹은 네트워크 전체 재학습이 필요합니다. 일부 최신 방법들은 feature 유사도로 포즈를 추정하지만, 기하학적 제약의 부재로 정확도가 떨어집니다.

NeRF 기반 방법들도 **느린 수렴 속도**와 **높은 연산 비용**으로 실용성이 제한되었습니다.

---

### 2-2. 제안하는 방법 (수식 포함)

GS-CPR의 처리 파이프라인은 다음과 같습니다:

#### Step 1: 3DGS 렌더링

3DGS 모델은 추정된 초기 포즈 $\hat{p}$를 기반으로 합성 이미지 $\hat{I}_r$와 대응 깊이 맵 $\hat{I}_d$를 렌더링합니다. 이때 ACT 모듈이 적용되어 렌더링된 이미지가 쿼리 이미지의 노출 특성을 모방하도록 합니다.

각 3D Gaussian은 다음의 속성으로 정의됩니다:

$$\mathcal{G}_k = \{\mu_k,\ \Sigma_k,\ \alpha_k,\ c_k\}$$

- $\mu_k$: 3D 평균(위치)
- $\Sigma_k$: 공분산 행렬 (형태/크기/방향)
- $\alpha_k$: 불투명도
- $c_k$: 색상 (Spherical Harmonics)

3DGS의 렌더링은 alpha-compositing 방식으로:

$$\hat{I}_r(u,v) = \sum_{k \in \mathcal{N}} c_k \cdot \alpha_k \prod_{j < k}(1 - \alpha_j)$$

깊이 맵 $\hat{I}_d$도 동일한 방식으로 렌더링되어 픽셀별 3D 좌표 맵 $X_r^d \in \mathbb{R}^{H \times W \times 3}$을 생성합니다.

#### Step 2: 노출 적응 변환 (ACT 모듈)

노출 적응 Affine Color Transformation(ACT) 모듈은 쿼리 이미지의 노출 특성에 맞게 렌더링 이미지를 조정하여 야외 환경에서의 강인성을 향상시킵니다.

ACT 변환은 다음과 같이 표현됩니다:

$$\hat{I}_r^{\text{ACT}} = a \cdot \hat{I}_r + b$$

여기서 $a$, $b$는 쿼리 이미지에 맞춰 MLP를 통해 학습된 affine 변환 파라미터입니다.

#### Step 3: 2D-2D 매칭 → 2D-3D 대응 수립

MASt3R 매처를 사용하여 쿼리 이미지 $I_q$와 렌더링 이미지 $\hat{I}_r$ 사이의 dense 2D-2D 대응 관계를 구축합니다. 깊이 맵 $\hat{I}_d$는 3D 좌표 맵 $X_r^d$를 생성하여 이 대응 관계를 3D 장면과 연결합니다.

매칭 결과: 픽셀 집합 $\{(u_i^q, v_i^q)\} \leftrightarrow \{(u_i^r, v_i^r)\}$

깊이 맵을 통한 3D 리프팅:

$$P_i^{3D} = X_r^d(u_i^r, v_i^r) \in \mathbb{R}^3$$

따라서 2D-3D 대응 집합:

$$\mathcal{C} = \{(u_i^q, v_i^q,\ P_i^{3D})\}_{i=1}^{N}$$

#### Step 4: PnP + RANSAC 포즈 정제

구축된 2D-2D 대응 관계와 3D 좌표 맵으로부터 Perspective-n-Point(PnP) 알고리즘과 RANSAC을 결합하여 outlier를 필터링하고 최종 정제 포즈 $\hat{p}'$를 계산합니다.

$$\hat{p}' = \arg\min_{\mathbf{R}, \mathbf{t}} \sum_{i \in \mathcal{I}_{\text{inlier}}} \left\| \pi\bigl(\mathbf{R} \cdot P_i^{3D} + \mathbf{t}\bigr) - \begin{pmatrix} u_i^q \\ v_i^q \end{pmatrix} \right\|^2$$

- $\mathbf{R} \in SO(3)$: 회전 행렬
- $\mathbf{t} \in \mathbb{R}^3$: 이동 벡터
- $\pi(\cdot)$: 카메라 투영 함수

GS-CPR은 3DGS 모델로부터 깊이 맵을 렌더링하고 PnP solver와 RANSAC을 적용하여 반복적 역전파 없이 one-shot 포즈 정제를 수행합니다.

---

### 2-3. 모델 구조

GS-CPR은 사전 학습된 포즈 추정기(pose estimator)와 3DGS 모델을 전제합니다. 쿼리 이미지가 입력되면 먼저 포즈 추정기로부터 초기 추정 포즈를 얻으며, 최종적으로 정제된 6DoF 카메라 포즈를 출력합니다.

전체 아키텍처 구성요소:

| 구성 요소 | 역할 | 비고 |
|---|---|---|
| **Scaffold-GS** | 3D 장면 표현 및 렌더링 | COLMAP SfM으로 초기화 |
| **ACT MLP** | 노출 적응 색상 변환 | 야외 도메인 시프트 완화 |
| **MASt3R** | Dense 2D-2D 매칭 | 3D 파운데이션 모델 |
| **PnP + RANSAC** | 포즈 계산 및 outlier 제거 | IPPE/P3P solver |
| **APR/SCR** | 초기 포즈 추정 | DFNet, ACE 등 플러그인 |

GS-CPR 프로젝트는 Scaffold-GS, MASt3R, DUSt3R, NeFeS, ACE, Depth for 3DGS 등 여러 뛰어난 저장소를 기반으로 개발되었습니다.

---

### 2-4. 성능 향상

#### 런타임 성능

NVIDIA GeForce RTX 4090 GPU 기준으로, 3DGS 렌더링은 7Scenes 데이터셋에서 평균 3.7ms, Cambridge Landmarks 데이터셋에서 12ms가 소요됩니다. MASt3R 상대 포즈 추정에 71ms, MASt3R 매칭에 추가 42ms, PnP+RANSAC에 52ms가 소요됩니다.

Cambridge Landmarks 데이터셋에서 MCLoc은 80회 반복으로 쿼리당 평균 2.4초가 필요한 반면, ACE+GS-CPR의 one-shot 최적화는 쿼리당 0.19초만 소요됩니다.

#### 로컬라이제이션 정확도

GS-CPR은 APR 및 SCR 메서드의 대략적인 초기 포즈 추정치를 활용한 one-shot 포즈 정제를 가능하게 하며, 다양한 실내외 시각적 로컬라이제이션 벤치마크에서 기존 NeRF 기반 최적화 방법들보다 정확도와 런타임 모두에서 우수하여, 두 개의 실내 데이터셋에서 새로운 최고 수준의 정확도를 달성합니다.

빠른 변형 GS-CPR $\text{rel}$ 은 속도와 정확도 간의 트레이드오프를 제공합니다. GS-CPR $\text{rel}$ 은 DFNet과 같은 APR 방법을 7Scenes에서 5cm, 5° 기준 recall이 43.1%에서 80.5%로 크게 향상시키지만, 이미 높은 정확도를 가진 ACE와 같은 SCR 방법은 97.1%에서 79.9%로 성능이 저하될 수 있습니다.

#### 정리

| 데이터셋 | 측정 지표 | 개선 효과 |
|---|---|---|
| 7Scenes (실내) | 중앙값 병진/회전 오차 | 최고 수준 달성 |
| 12Scenes (실내) | 중앙값 오차 | 최고 수준 달성 |
| Cambridge Landmarks (야외) | 런타임 | MCLoc 대비 ~12배 빠름 |

---

### 2-5. 한계점

1. **장면별 3DGS 사전 학습 필요**: 각 장면에 대한 3DGS 모델 훈련 시 COLMAP으로 생성된 훈련 프레임의 희소 포인트 클라우드를 초기화로 사용합니다. 즉, 새로운 장면마다 3DGS를 새로 학습해야 합니다.

2. **GS-CPR $\text{rel}$ 의 성능 불일치**: GS-CPR $\text{rel}$ 은 초기 포즈 추정이 거칠고 계산 효율성이 중요한 시나리오에 가장 적합하며, 모든 초기 포즈 품질에서 정확도를 극대화하려면 GS-CPR을 사용하는 것이 바람직합니다.

3. **야외 환경의 동적 객체**: GS-CPR의 노출 적응 ACT 모듈과 temporal object filtering이 조명 변화와 움직이는 객체가 있는 환경에서의 도전적인 조건을 처리하는 데 기여합니다. 그러나 동적 장면에서의 일반화는 여전히 도전적입니다.

4. **MASt3R 매칭 병목**: 매칭 단계(71ms+42ms)가 전체 파이프라인에서 가장 큰 시간 비중을 차지하여, 실시간 적용에 제약이 있습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 현재 일반화 관련 설계

GS-CPR은 feature extractor나 descriptor를 별도로 훈련할 필요 없이 RGB 이미지에 직접 작동하며, 3D 파운데이션 모델 MASt3R을 정밀 2D 매칭에 활용합니다. 이는 장면별 재학습 없이도 다양한 매칭 환경에 대응할 수 있는 기반을 제공합니다.

3DGS 모델에 노출 적응 모듈을 통합하여 쿼리 이미지와 렌더링 이미지 간의 도메인 시프트에 대한 강인성을 향상시킵니다.

### 3-2. 포즈 추정기에 대한 Plug-and-Play 일반화

이 프레임워크는 최첨단 절대 포즈 회귀(APR) 및 장면 좌표 회귀(SCR) 방법들의 로컬라이제이션 정확도를 향상시킵니다.

GS-CPR은 포즈 추정기에 구애받지 않는(agnostic) 플러그인 방식으로, DFNet, ACE 등 다양한 APR/SCR 방법과 조합 가능합니다. 이는 미래 등장하는 새로운 포즈 추정기에도 손쉽게 적용할 수 있음을 의미합니다.

### 3-3. 후속 연구에서 본 논문을 기반으로 한 개선 사례

GS-CPR의 한계인 "포즈 prior 불확실성과 기하학적 불확실성"을 해결하기 위해 후속 연구가 이미 등장했습니다:

GS-CPR 기반의 연구에서는 3D Gaussian Splatting 기반 포즈 정제에서 포즈 prior와 기하학적 불확실성의 영향을 다루며, 이를 완화하기 위해 Monte Carlo 샘플링과 Fisher Information 기반 PnP 최적화를 결합한 확률론적 재로컬라이제이션 프레임워크를 제안합니다. 이 설계는 재학습이나 장면별 조정 없이 불확실성을 처리합니다.

또한 GS-SMC라는 방법도 등장했습니다:

GS-SMC는 기존 3DGS 모델을 활용하여 새로운 뷰를 렌더링하므로, 추가 학습이나 fine-tuning 없이 다양한 장면에 직접 적용할 수 있는 경량 솔루션을 제공합니다. 쿼리와 여러 렌더링 이미지들 간의 epipolar 기하학적 제약을 활용하여 카메라 포즈를 반복적으로 정제합니다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 방법 | 연도 | 표현 방식 | 정제 방식 | 특징 | 한계 |
|---|---|---|---|---|---|
| **NeFeS** (Chen et al.) | 2024 | NeRF | 반복 역전파 | SCR 정확도 높음 | 느린 수렴 |
| **CrossFire** (Moreau et al.) | 2023 | NeRF | Self-supervised feature | 암묵적 표현 | 정확도 제한 |
| **MCLoc** (Trivigno et al.) | 2024 | 3DGS | 80회 반복 최적화 | 표현 무관 | 2.4초/쿼리 |
| **GS-CPR (본 논문)** | 2024/2025 | 3DGS+Scaffold | MASt3R+PnP+RANSAC | One-shot, 0.19초/쿼리 | 장면별 3DGS 필요 |
| **UGS-Loc** | 2026 | 3DGS | Monte Carlo+Fisher PnP | 불확실성 처리 | GS-CPR 기반 확장 |
| **GS-SMC** | 2025 | 3DGS | Epipolar+반복 정제 | 학습 불필요, 범용 적용 | 다중 렌더 필요 |

반복 적용 실험에서 GS-CPR을 여러 번 반복하면 병진 및 회전 오차가 점진적으로 감소하지만 첫 번째 반복 후 빠르게 포화됩니다. 이에 비해 확률론적 방법(UGS-Loc)은 일관된 수렴과 더 낮은 최종 오차를 달성하며, 이 개선이 단순 반복 최적화가 아닌 프레임워크 자체에서 비롯됨을 보여줍니다.

---

## 5. 앞으로의 연구에 미치는 영향 및 연구 시 고려할 점

### 5-1. 연구에 미치는 영향

**① 3DGS의 비주얼 로컬라이제이션 표준화 촉진**

GS-CPR은 고급 3D 렌더링 기술과 강력한 매칭 알고리즘을 효과적으로 결합하여 간단한 초기 추정치로부터 포즈 추정 정확도를 향상시키는 카메라 포즈 정제 분야의 중요한 진보를 나타냅니다. 그 효과성, 효율성, 실용적인 응용에의 적합성을 입증하며 시각적 로컬라이제이션 분야에 새로운 기준을 세웁니다.

**② 파운데이션 모델과 3DGS의 융합 연구 방향 제시**

MASt3R(DUSt3R 계열)와 같은 3D 비전 파운데이션 모델을 3DGS 기반 파이프라인에 통합하는 패러다임을 제시하여, 이후 연구들이 비슷한 조합을 탐구하는 데 영감을 줍니다.

**③ Test-time Refinement 패러다임의 확산**

별도의 재학습 없이 테스트 시점에서 포즈를 정제하는 패러다임이 다양한 도메인(로봇공학, AR/VR, 자율주행)으로 확산될 것으로 기대됩니다.

**④ 후속 연구의 출발점**

GS-CPR을 기반으로, 3D Gaussian Splatting 기반 포즈 정제에서 포즈 prior와 기하학적 불확실성의 영향을 탐구하는 연구들이 활발히 등장하고 있으며, Monte Carlo 샘플링과 Fisher Information 기반 PnP 최적화를 결합한 확률론적 접근법이 재학습이나 장면별 조정 없이 불확실성을 처리할 수 있음을 보여줍니다.

---

### 5-2. 향후 연구 시 고려할 점

**① 동적 장면 대응 강화**

현재 GS-CPR은 정적 3DGS 표현을 사용합니다. 동적 객체(사람, 차량 등)가 많은 환경에서의 일반화를 위해 **Dynamic 3DGS** 또는 **Deformable Gaussian** 기법과의 융합을 고려해야 합니다.

**② 장면별 3DGS 사전 학습의 효율화**

현재 방식은 각 장면마다 COLMAP + Scaffold-GS 학습이 필요합니다. 향후 연구에서는 **generalizable NeRF/3DGS** 방법을 통해 새로운 장면에 대한 zero-shot 혹은 few-shot 3DGS 구성을 탐구할 필요가 있습니다.

**③ 불확실성 추정 통합**

MASt3R은 강력한 맥락적 추론과 dense feature aggregation 덕분에 약간 더 높은 정확도를 달성하는 경향이 있지만, 불확실성 인식 정제 방법은 경량 희소 매처로도 비슷한 로컬라이제이션 정확도를 달성할 수 있습니다. 이는 불확실성 추정의 통합이 매처 선택의 유연성을 높일 수 있음을 시사합니다.

**④ 실시간 처리를 위한 경량화**

MASt3R 매칭(113ms)이 병목임을 고려하여, **경량 매처(SuperPoint+LightGlue)** 또는 **희소 매칭** 전략을 결합한 실시간 파이프라인 개발이 필요합니다.

**⑤ 대규모 장면으로의 확장**

현재 7Scenes, 12Scenes, Cambridge Landmarks 등 소~중규모 장면에서 검증되었습니다. **NeRF In the Wild** 수준의 대규모 야외 장면이나 도시 규모(city-scale) 로컬라이제이션으로의 확장 연구가 필요합니다.

**⑥ 멀티모달 입력 통합**

RGB 이미지만을 입력으로 사용하는 현재 방식에서 **Depth, IMU, LiDAR** 등 멀티모달 센서 정보를 통합하면 열악한 조명 환경이나 특징이 부족한 장면에서의 강인성을 높일 수 있습니다.

---

## 📚 참고 자료 및 출처

1. **arXiv 원문**: [GS-CPR: Efficient Camera Pose Refinement via 3D Gaussian Splatting (arXiv:2408.11085)](https://arxiv.org/abs/2408.11085)
2. **ICLR 2025 OpenReview**: [GS-CPR OpenReview](https://openreview.net/forum?id=mP7uV59iJM)
3. **공식 프로젝트 페이지**: [xrim-lab.github.io/GS-CPR](https://xrim-lab.github.io/GS-CPR/)
4. **GitHub 공식 코드**: [XRIM-Lab/GS-CPR](https://github.com/XRIM-Lab/GS-CPR)
5. **ICLR 2025 포스터 슬라이드**: [iclr.cc/media/iclr-2025/Slides/28467](https://iclr.cc/media/iclr-2025/Slides/28467_LEQRMgp.pdf)
6. **관련 후속 논문 (GS-SMC)**: [arXiv:2508.17876 - Camera Pose Refinement via 3D Gaussian Splatting](https://arxiv.org/abs/2508.17876)
7. **관련 후속 논문 (UGS-Loc)**: [arXiv:2603.16538 - Rethinking Pose Refinement in 3D Gaussian Splatting under Pose Prior and Geometric Uncertainty](https://arxiv.org/html/2603.16538)
8. **Quick Review**: [liner.com GS-CPR Review](https://liner.com/review/gscpr-efficient-camera-pose-refinement-via-3d-gaussian-splatting)
9. **Literature Review**: [themoonlight.io GS-CPR Review](https://www.themoonlight.io/en/review/gs-cpr-efficient-camera-pose-refinement-via-3d-gaussian-splatting)
10. **HKUST Research Portal**: [researchportal.hkust.edu.hk](https://researchportal.hkust.edu.hk/en/publications/gs-cpr-efficient-camera-pose-refinement-via-3d-gaussian-splatting)

> ⚠️ **주의**: 본 분석에서 수식 일부(3DGS 렌더링 공식, ACT 수식의 세부 파라미터, PnP 비용 함수 등)는 논문의 공개된 arXiv HTML 및 관련 문헌에서 확인 가능한 범위에서 작성되었으며, 논문 원문의 전체 수식 체계와 일부 차이가 있을 수 있습니다. 정확한 수식 확인은 원문 PDF를 참조하시기 바랍니다.
