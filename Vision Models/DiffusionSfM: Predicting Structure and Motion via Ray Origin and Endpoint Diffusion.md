
# DiffusionSfM: Predicting Structure and Motion via Ray Origin and Endpoint Diffusion

> **논문 정보**
> - **제목:** DiffusionSfM: Predicting Structure and Motion via Ray Origin and Endpoint Diffusion
> - **저자:** Qitao Zhao, Amy Lin, Jeff Tan, Jason Y. Zhang, Deva Ramanan, Shubham Tulsiani (Carnegie Mellon University)
> - **발표:** CVPR 2025 (IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6317–6326)
> - **arXiv:** [2505.05473](https://arxiv.org/abs/2505.05473) (2025년 5월 8일)
> - **프로젝트 페이지:** [qitaozhao.github.io/DiffusionSfM](https://qitaozhao.github.io/DiffusionSfM)
> - **공식 코드:** [github.com/QitaoZhao/DiffusionSfM](https://github.com/QitaoZhao/DiffusionSfM)

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

현재 Structure-from-Motion(SfM) 방법들은 일반적으로 학습된 혹은 기하학적 쌍별(pairwise) 추론과 이후의 전역 최적화(global optimization) 단계를 결합하는 **두 단계 파이프라인**을 따른다. 이에 반해, DiffusionSfM은 멀티뷰 이미지로부터 **3D 장면 기하 구조와 카메라 포즈를 직접 추론**하는 데이터 기반 멀티뷰 추론 접근 방식을 제안한다.

DiffusionSfM 프레임워크는 장면 기하 구조와 카메라를 **글로벌 프레임 내 픽셀별 레이 오리진(ray origin) 및 엔드포인트(ray endpoint)**로 파라미터화하고, 트랜스포머 기반 디노이징 확산 모델(denoising diffusion model)을 활용하여 멀티뷰 입력으로부터 이를 예측한다.

### 1.2 주요 기여 (Contributions)

| 기여 항목 | 설명 |
|---|---|
| **통합 표현(Unified Representation)** | 카메라 포즈와 3D 기하 구조를 레이 오리진+엔드포인트로 단일화 |
| **확산 모델 적용** | SfM에 트랜스포머 기반 DDPM 적용 |
| **결측 데이터 처리** | GT Mask Conditioning 도입 |
| **무한 좌표계 처리** | 동차 좌표계(Homogeneous Coordinates) 적용 |
| **불확실성 모델링** | 확산 프로세스를 통한 자연스러운 불확실성 표현 |

전통적인 SfM 파이프라인이 쌍별 추론과 전역 최적화를 두 단계로 분리한 것과 달리, 이 접근 방식은 두 단계를 **단일 엔드투엔드 멀티뷰 추론 프레임워크**로 통합한다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

#### 기존 방법의 한계

기존 SfM 방법들은 학습된 혹은 기하학적 쌍별 추론 후 전역 최적화를 수행하는 두 단계 파이프라인을 채택하고 있다. DiffusionSfM은 대신 멀티뷰 이미지로부터 카메라와 3D 기하를 **직접** 추론하는 데이터 기반 멀티뷰 추론 접근을 제안한다.

RayDiffusion이 이미지 패치당 (depth-agnostic한) 레이만 추론하거나, DUSt3R이 픽셀당 3D 포인트만 추론하는 것과 달리, DiffusionSfM은 **레이 오리진과 엔드포인트를 픽셀별로 함께 예측**하여 장면 기하(엔드포인트)와 일반화된 카메라(레이)를 동시에 직접 보고한다.

DUSt3R 대비 DiffusionSfM은 $N$개 뷰에 대해 직접 구조 및 모션을 예측하므로, **메모리 집약적인 전역 정렬(global alignment)이 불필요**하다.

#### 훈련 시의 두 가지 핵심 과제

확산 모델을 훈련하는 데 두 가지 핵심 문제가 있다. 첫째, 확산 모델은 훈련을 위해 (노이즈가 포함된) 정답(ground truth)을 입력으로 요구하는데, 기존 실제 데이터셋은 멀티뷰 스테레오에서 깊이 정보 누락으로 인해 **모든 픽셀의 엔드포인트를 알 수 없다**.

두 번째 문제는 **무제한 장면 좌표(unbounded scene coordinates)** 문제이다. DiffusionSfM의 확산 기반 접근법은 고정된 표준편차 1의 가우시안 노이즈를 훈련에 사용하므로 입력 데이터가 합리적인 범위 내에 있다고 가정하는데, 구성 요소 간 스케일 차이가 큰 훈련 장면은 모델의 학습 과정을 방해한다.

---

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 레이 오리진 & 엔드포인트 파라미터화

카메라 모델을 일반화하여, 각 픽셀 $p$에 대한 레이를 다음과 같이 정의한다:

$$\mathbf{r}_p(t) = \mathbf{o}_p + t \cdot (\mathbf{e}_p - \mathbf{o}_p), \quad t \in [0, 1]$$

여기서:
- $\mathbf{o}_p \in \mathbb{R}^3$: 레이 오리진 (카메라 중심)
- $\mathbf{e}_p \in \mathbb{R}^3$: 레이 엔드포인트 (3D 장면 점)

#### 2.2.2 동차 좌표계 (Homogeneous Coordinates) — 핵심 메커니즘

동차 좌표계(homogeneous coordinates)를 사용하면 입력 데이터를 단위 노름(unit norm)으로 정규화할 수 있으며, 이는 훈련을 안정화하고 수렴을 용이하게 한다.

레이 오리진 $\mathbf{o}_p$와 엔드포인트 $\mathbf{e}_p$를 각각 동차 좌표로 표현하면:

$$\tilde{\mathbf{o}}_p = \frac{[\mathbf{o}_p; 1]}{\|[\mathbf{o}_p; 1]\|} \in \mathbb{S}^3, \quad \tilde{\mathbf{e}}_p = \frac{[\mathbf{e}_p; 1]}{\|[\mathbf{e}_p; 1]\|} \in \mathbb{S}^3$$

레이 오리진과 엔드포인트에 동차 좌표를 사용하는 것은 안정적인 모델 훈련에 **매우 중요**하다. 이를 표준 3D 좌표 $\mathbb{R}^3$으로 대체하면 훈련이 어렵고 수렴에 실패한다.

#### 2.2.3 DDPM 기반 디노이징 목표 함수

기본 DDPM(Denoising Diffusion Probabilistic Models, Ho et al., 2020)의 훈련 목표:

**순방향 프로세스 (Forward Process):**

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$$

여기서 $\bar{\alpha}\_t = \prod_{s=1}^{t}(1 - \beta_s)$이고 $\beta_t$는 노이즈 스케줄.

**역방향 프로세스 훈련 손실 ( $x_0$ -파라미터화):**

$$\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \| \mathbf{x}_0 - \hat{\mathbf{x}}_\theta(\mathbf{x}_t, t, \mathcal{I}) \|^2 \right]$$

여기서:
- $\mathbf{x}_0$: 정답 레이 오리진 및 엔드포인트 맵
- $\hat{\mathbf{x}}_\theta$: DiT 기반 신경망 예측값
- $\mathcal{I}$: 멀티뷰 이미지 조건

DiffusionSfM은 **$x_0$-파라미터화**를 사용하여 레이 오리진과 엔드포인트 맵의 클린 샘플을 모델 출력으로 예측하며, 100개의 확산 디노이징 타임스텝을 사용한다.

#### 2.2.4 GT Mask Conditioning (결측 데이터 처리)

불완전한 정답(GT) 처리를 위해 모델은 **GT 마스크를 조건으로** 훈련된다. 추론 시에는 GT 마스크를 모두 1로 설정하여 모든 픽셀에 대한 오리진 및 엔드포인트 예측이 가능하도록 한다.

마스킹된 손실 함수:

$$\mathcal{L}_{\text{masked}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \mathbf{M} \odot \| \mathbf{x}_0 - \hat{\mathbf{x}}_\theta(\mathbf{x}_t, t, \mathcal{I}, \mathbf{M}) \|^2 \right]$$

여기서 $\mathbf{M} \in \{0, 1\}^{H \times W}$은 유효 깊이 픽셀 마스크.

---

### 2.3 모델 구조

#### 전체 파이프라인

```
멀티뷰 이미지 입력 (N장)
        ↓
[DINOv2 패치 임베딩] → 이미지 특징 추출
        ↓
[다운샘플링 Conv 레이어] → 노이즈 레이 오리진/엔드포인트 인코딩
        ↓
[Diffusion Transformer (DiT)] → 멀티뷰 교차 어텐션 기반 디노이징
        ↓
[DPT Conv Head] → 풀해상도 레이 오리진/엔드포인트 복원
        ↓
후처리: 카메라 외재/내재 파라미터 + 깊이 맵 복원
```

각 이미지에 대해 **DINOv2**로 패치 임베딩을 계산하고, 단일 다운샘플링 합성곱 레이어로 노이즈가 있는 레이 오리진과 엔드포인트를 잠재 공간(latent)으로 인코딩하여 이미지 임베딩의 공간적 풋프린트(spatial footprint)와 정렬한다.

**Diffusion Transformer(DiT)** 아키텍처를 구현하여 노이즈 샘플로부터 클린한 레이 오리진과 엔드포인트를 예측하며, 합성곱 DPT 헤드가 **풀해상도** 디노이즈된 결과물을 출력한다.

예측된 레이 오리진과 엔드포인트는 3D로 직접 시각화하거나 후처리를 통해 **카메라 외재 파라미터(extrinsics), 내재 파라미터(intrinsics), 멀티뷰 일관성 있는 깊이 맵**으로 변환할 수 있다.

#### Sparse-to-Dense 훈련 전략

고해상도 모델(dense model)을 처음부터 훈련하면 성능이 저하되기 때문에, **스파스-투-덴스(sparse-to-dense) 전략**을 사용하여 모델을 훈련한다.

스파스 모델은 각 이미지 패치에 대한 레이 오리진과 엔드포인트를 예측하지만, 이는 장면의 세밀한 디테일을 포착하는 데 한계가 있다.

#### 조기 종료(Early Stopping) 추론 최적화

DiffusionSfM은 **타임스텝 $T=90$에서 가장 정확한 클린 샘플 예측**을 달성하며, 이는 최종 디노이징 단계가 아닌 초기 타임스텝에서 최적의 성능이 나오는 흥미로운 현상이다. 이 관찰은 여러 입력 이미지 수에서 일관되게 나타나며, 이를 활용하여 **추론을 10 디노이징 스텝으로 제한**하고 $T=90$에서의 $x_0$ 예측을 최종 출력으로 사용하여 추론 시간을 크게 단축한다.

---

### 2.4 성능 향상

DiffusionSfM은 합성 및 실제 데이터셋 모두에서 실험적으로 검증되었으며, **클래식 및 학습 기반 접근법 모두를 능가**하면서 자연스러운 불확실성 모델링을 제공한다.

DiffusionSfM은 카메라 중심 정확도에서 **모든 다른 방법들을 능가**하며, 동등한 데이터로 훈련된 모든 방법들 중 회전 정확도에서도 우수한 성능을 보인다.

DiffusionSfM은 회전 정확도에서 DUSt3R과 대등한 반면, **카메라 중심(center) 정확도에서는 DUSt3R을 일관되게 능가**한다.

또한 DiffusionSfM은 도전적인 실제 세계 데이터에서 기존 클래식 및 학습 기반 방법들보다 향상된 성능을 보이면서, **불확실성을 자연스럽게 모델링**하고 추론 시 외부 가이던스를 통합할 수 있다.

---

### 2.5 한계

DiffusionSfM은 최신 T2I 생성 시스템이 채택한 잠재 공간 모델과 달리 **픽셀 공간 확산 모델**을 사용한다. 픽셀 공간에서 동작하면 더 큰 모델 용량이 필요하지만 현재 모델은 상대적으로 작아서, 물체 경계를 따라 **노이즈 패턴**이 관찰될 수 있다.

또한 멀티뷰 트랜스포머에서 연산 요구사항은 입력 이미지 수에 따라 **이차적으로(quadratically) 증가**하므로, 대규모 입력 이미지 집합에 배포하려면 마스크드 어텐션이 필요하다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화 성능

DiffusionSfM은 도전적인 입력에서도 강건한 성능을 보이며, DUSt3R이 일관된 방식으로 이미지를 등록하는 데 실패하는 경우에도 **일관된 전역 예측**을 생성한다.

반면 RayDiffusion은 CO3D 데이터만으로 훈련되어 장면 이미지가 **분포 외(out-of-distribution)**로 취급된다.

### 3.2 일반화를 가능하게 하는 설계 요소

| 설계 요소 | 일반화에 미치는 영향 |
|---|---|
| **동차 좌표계** | 스케일 변동에 불변한 표현으로 다양한 장면 스케일 처리 가능 |
| **GT Mask Conditioning** | 불완전한 데이터셋에서도 강건한 학습 가능 |
| **DINOv2 백본** | 강력한 범용 시각 특징 제공 |
| **확산 기반 불확실성** | 모호한 입력에 대한 다중 가설 제공 가능 |

GT 마스크 컨디셔닝으로 학습 중 결측 데이터를 표시하면 **기하 품질이 향상**된다.

### 3.3 미래 일반화 향상 방향

레이 오리진 및 엔드포인트에 대한 표현력 있는 잠재 공간을 **VAE 학습**으로 습득하는 것이 향후 연구의 유망한 방향이 될 수 있다.

또한 이 접근법은 SfM(오리진과 엔드포인트가 미지인 이미지), 등록(일부 이미지는 알려진 오리진/엔드포인트를 가지고 다른 것들은 그렇지 않은 경우), 매핑(알려진 레이이지만 미지의 엔드포인트), 뷰 합성(알려진 레이에 대한 미지의 픽셀 값) 등의 **관련 기하 태스크 전반에 걸쳐 공통 시스템을 훈련**하는 데 발전될 수 있다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 관련 연구 타임라인

| 연도 | 연구 | 핵심 방법 | 주요 특징 |
|---|---|---|---|
| 2020 | **DDPM** (Ho et al., NeurIPS) | Denoising Diffusion Probabilistic Models | 확산 모델 기반 확률적 생성 프레임워크 |
| 2022 | **RelPose** (Zhang et al., ECCV) | Energy-based 상대 회전 분포 | 단일 오브젝트 상대 카메라 회전 추론 |
| 2023 | **PoseDiffusion** | 확산 프레임워크 내 카메라 포즈 모델링 | SfM 문제를 확률적 확산 프레임워크 내에서 정식화하여 카메라 포즈의 조건부 분포를 모델링하는 방식 |
| 2024 | **RayDiffusion** (Zhang et al., ICLR) | 레이 기반 희소 뷰 포즈 추정 | 패치 단위 레이 예측, 포즈만 추정 |
| 2024 | **DUSt3R** (Wang et al., CVPR) | Dense Unconstrained Stereo 3D Reconstruction | 포인트맵 예측, 쌍별 추론 + 전역 정렬 |
| 2024 | **MASt3R** (Leroy et al., ECCV) | DUSt3R 확장 | 3D 이미지 매칭 결합 |
| 2024 | **RelPose++** (Lin et al., 3DV) | 희소 뷰 6D 포즈 복원 | 6DoF 완전 포즈 추정 |
| **2025** | **DiffusionSfM** (Zhao et al., CVPR) | 레이 오리진+엔드포인트 확산 | 카메라+기하 통합 추론, 불확실성 모델링 |

### 4.2 DiffusionSfM vs. 주요 경쟁 방법 비교

DiffusionSfM은 RayDiffusion 대비 구조를 직접 예측하며 패치 단위가 아닌 **픽셀 단위**의 더 세밀한 스케일로 동작한다. DUSt3R 대비 모션과 구조를 모두 직접 예측하며, $N$개의 뷰에 대해 메모리 집약적인 전역 정렬 없이 처리한다.

```
방법 비교:
         포즈 추정   기하 복원   불확실성   N뷰 직접처리   전역정렬 필요
COLMAP       ✓          ✓          ✗           ✗             ✓
DUSt3R       ✓          ✓          ✗           △             ✓
RayDiffusion ✓          ✗          ✓           ✓             ✗
DiffusionSfM ✓          ✓          ✓           ✓             ✗
```

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 연구에 미치는 영향

**① SfM 패러다임 전환**
DiffusionSfM은 멀티뷰 기하 태스크를 위한 **통합 접근 방식의 가능성**을 강조하며, 관련 기하 태스크 전반에 걸친 공통 시스템 구축의 발판이 될 것이다.

**② 확산 모델의 3D 비전 확장**
확산 모델이 불확실성을 자연스럽게 모델링하고 **추론 시 외부 가이던스를 통합**할 수 있다는 점에서, 3D 비전 전반에 걸쳐 확산 기반 기하 추론 연구를 가속화할 것이다.

**③ 표현의 통일성**
카메라와 기하 구조를 레이 오리진·엔드포인트로 통합 표현하는 방식은 향후 뷰 합성, 장면 편집, 로봇 내비게이션 등 다양한 하위 태스크에서 재사용 가능한 범용 3D 표현 연구를 촉진할 것이다.

### 5.2 앞으로 연구 시 고려할 점

#### (1) 잠재 공간 확산으로의 전환
현재 픽셀 공간 확산 모델은 더 큰 모델 용량을 필요로 하며 물체 경계에서 노이즈 패턴 문제가 있다. **VAE를 통한 레이 오리진·엔드포인트의 잠재 공간 학습**은 유망한 미래 방향이다.

#### (2) 연산 효율화
멀티뷰 트랜스포머의 연산 요구량이 입력 이미지 수에 따라 이차적으로 증가하므로, 대규모 이미지 집합 배포를 위한 **마스크드 어텐션(masked attention)** 메커니즘 도입이 필요하다.

#### (3) 멀티 데이터셋 훈련
CO3D만으로 훈련된 모델(*)과 다중 데이터셋으로 훈련된 모델 간 성능 차이가 보고되어 있으므로, **다양한 도메인의 대규모 데이터 훈련**이 일반화 성능에 핵심적이다.

#### (4) 조기 종료(Early Stopping) 최적 타임스텝 탐색
최적 타임스텝은 데이터셋에 따라 달라지며($T=90$ 또는 $T=85$ 등), 타임스텝 적응 전략 연구가 필요하다.

#### (5) 통합 기하 학습 프레임워크 구축
SfM(미지의 오리진·엔드포인트), 등록(일부 알려진 경우), 매핑(알려진 레이·미지의 엔드포인트), 뷰 합성(알려진 레이·미지의 픽셀 값) 등 **관련 기하 태스크 전체를 아우르는 통합 시스템 연구**가 DiffusionSfM의 핵심 미래 방향이다.

---

## 참고 자료 (References)

1. **DiffusionSfM 논문 (arXiv):** Qitao Zhao et al., "DiffusionSfM: Predicting Structure and Motion via Ray Origin and Endpoint Diffusion," arXiv:2505.05473, 2025. https://arxiv.org/abs/2505.05473
2. **DiffusionSfM 프로젝트 페이지:** https://qitaozhao.github.io/DiffusionSfM
3. **DiffusionSfM 공식 코드 (GitHub):** https://github.com/QitaoZhao/DiffusionSfM
4. **DiffusionSfM CVPR 2025 논문 (CVF OpenAccess):** https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_DiffusionSfM_Predicting_Structure_and_Motion_via_Ray_Origin_and_Endpoint_CVPR_2025_paper.pdf
5. **DiffusionSfM Semantic Scholar:** https://www.semanticscholar.org/paper/DiffusionSfM:-Predicting-Structure-and-Motion-via-Zhao-Lin/be9c7130fd8ee7dd818e928bf9117a8929b46531
6. **DiffusionSfM CVPR 2025 Poster (CVF Virtual):** https://cvpr.thecvf.com/virtual/2025/poster/34891
7. **DiffusionSfM arXiv HTML 전문:** https://arxiv.org/html/2505.05473v1
8. **DDPM:** Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." NeurIPS.
9. **DUSt3R:** Wang, S., et al. (2024). "DUSt3R: Geometric 3D Vision Made Easy." CVPR.
10. **RayDiffusion:** Zhang, J. Y., et al. (2024). "Cameras as Rays: Sparse-view Pose Estimation via Ray Diffusion." ICLR.
11. **MASt3R:** Leroy, V., Cabon, Y., & Revaud, J. (2024). "Grounding Image Matching in 3D with MASt3R." ECCV.
12. **RelPose++:** Lin, A., Zhang, J. Y., Ramanan, D., & Tulsiani, S. (2024). "RelPose++: Recovering 6D Poses from Sparse-view Observations." 3DV.
13. **DiT:** Peebles, W., & Xie, S. (2023). "Scalable Diffusion Models with Transformers." ICCV.
14. **DINOv2:** Oquab, M., et al. (2024). "DINOv2: Learning Robust Visual Features without Supervision." TMLR.
