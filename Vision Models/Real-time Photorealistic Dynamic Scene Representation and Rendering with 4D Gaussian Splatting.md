# Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting

> **논문 정보**: Yang, Zeyu, Yang, Hongye, Pan, Zijie, & Zhang, Li. *ICLR 2024*. Fudan University, University of Surrey, University of Oxford.

---

## 1. 핵심 주장과 주요 기여 (요약)

2D 이미지로부터 동적(dynamic) 3D 장면을 재구성하고 시간에 따라 다양한 뷰를 생성하는 것은, 장면의 복잡성과 시간적 역학 때문에 매우 어려운 과제이다. 기존의 neural implicit 모델에는 두 가지 한계가 있다: (i) 부적절한 장면 구조 — 복잡한 6D 플렌옵틱 함수를 직접 학습하는 것은 동적 장면의 공간·시간적 구조를 드러내기 어렵고, (ii) 변형 모델링의 확장성 — 장면 요소의 변형을 명시적으로 모델링하는 것은 복잡한 역학에서는 비현실적이다. 이를 해결하기 위해, 시공간(spacetime)을 하나의 전체로 간주하고 4D 프리미티브(primitive)의 집합을 최적화하여 동적 장면의 시공간 4D 볼륨을 근사하는 방법을 제안한다.

### 핵심 기여 3가지:

**(i)** 공간과 시간 차원의 일관된 통합 모델링을 위해 **편향 없는(unbiased) 4D Gaussian 프리미티브**와 전용 splatting 기반 렌더링 파이프라인을 제안한다.

**(ii)** 동적 장면에서 뷰 의존적 색상의 시간적 진화를 모델링하기 위한 **4D Spherindrical Harmonics(4DSH)**를 도입하였으며 이것이 유용하고 해석 가능하다.

**(iii)** 합성 및 실제, 단안(monocular) 및 다시점(multi-view) 데이터셋에서의 광범위한 실험을 통해 시각적 품질과 효율성 모두에서 기존의 모든 방법을 능가한다.

---

## 2. 상세 분석: 문제 정의, 방법론, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

기존 neural implicit 모델의 한계는 크게 두 가지이다: (i) **부적절한 장면 구조** — 복잡한 6D 플렌옵틱 함수를 직접 학습하면서 동적 장면의 시공간 구조를 파악하지 못하고, (ii) **변형 모델링의 확장 문제** — 복잡한 역학에서 장면 요소의 변형을 명시적으로 모델링하는 것이 비현실적이다.

기존 deformation 기반 방법들은 동적 장면이 고정된 3D Gaussian 집합에 의해 생성된다고 가정하고 장면을 구성하는 요소가 항상 가시적이라고 전제한다. 반면, 본 연구는 새로운 4D 장면 프리미티브를 정식화하여 이러한 가정들을 폐기하고 모호하고 복잡한 매핑을 유지할 필요를 회피한다.

### 2.2 제안 방법론 (수식 포함)

#### (A) 3D Gaussian Splatting 기초

3D Gaussian은 평균 $\boldsymbol{\mu} = (\mu_x, \mu_y, \mu_z)$와 공분산 행렬 $\boldsymbol{\Sigma}$로 정의된다:

$$G(\mathbf{x}) = \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)$$

공분산 행렬은 스케일 행렬 $\mathbf{S}$와 회전 행렬 $\mathbf{R}$로 분해되어 $\boldsymbol{\Sigma} = \mathbf{R}\mathbf{S}\mathbf{S}^T\mathbf{R}^T$로 표현되며, $\mathbf{S} = \text{diag}(s_x, s_y, s_z)$이고 $\mathbf{R}$은 단위 쿼터니언 $\mathbf{q}$로부터 구성된다. 또한 각 3D Gaussian은 뷰 의존적 색상을 위한 구면조화함수(SH) 계수와 불투명도 $\alpha$를 포함한다.

#### (B) 4D Gaussian 프리미티브로의 확장

본 논문의 핵심은 3D Gaussian을 **네이티브 4D Gaussian**으로 확장하는 것이다.

4D Gaussian의 평균은 $\boldsymbol{\mu} = (\mu_x, \mu_y, \mu_z, \mu_t)$인 4개의 스칼라로 표현되어 완전한 4D Gaussian 표현에 도달한다.

4D Gaussian은 다음과 같이 정의된다:

$$G_{4D}(\mathbf{x}, t) = \exp\left(-\frac{1}{2}\begin{pmatrix}\mathbf{x}-\boldsymbol{\mu}_{xyz}\\ t-\mu_t\end{pmatrix}^T \boldsymbol{\Sigma}_{4D}^{-1} \begin{pmatrix}\mathbf{x}-\boldsymbol{\mu}_{xyz}\\ t-\mu_t\end{pmatrix}\right)$$

여기서 4D 공분산 행렬은 다음과 같이 분해된다:

$$\boldsymbol{\Sigma}_{4D} = \mathbf{R}_{4D}\mathbf{S}_{4D}\mathbf{S}_{4D}^T\mathbf{R}_{4D}^T$$

4D 회전은 Gaussian이 4D 매니폴드에 적합하고 장면의 본질적인 운동을 포착할 수 있도록 한다. 비등방적 타원체가 공간과 시간에서 임의로 회전할 수 있다.

#### (C) 렌더링 파이프라인: 조건부·주변 분해

렌더링 파이프라인에서, 주어진 시간 $t$와 뷰 $\mathcal{I}$에 대해 각 4D Gaussian은 먼저 **조건부(conditional) 3D Gaussian**과 **주변(marginal) 1D Gaussian**으로 분해된다. 이후 조건부 3D Gaussian은 2D splat으로 투영된다. 최종적으로 평면 조건부 Gaussian, 1D 주변 Gaussian, 그리고 시간 진화하는 뷰 의존적 색상을 통합하여 뷰를 렌더링한다.

수학적으로, 4D Gaussian $G_{4D}(\mathbf{x}, t)$를 시간 $t$에서 조건부로 분해하면:

$$G_{4D}(\mathbf{x}, t) = G_{3D|t}(\mathbf{x} \mid t) \cdot p(t)$$

여기서 $p(t)$는 시간 축의 주변 1D Gaussian 분포이고, $G_{3D|t}$는 시간 $t$에서의 조건부 3D Gaussian이다. 조건부 3D Gaussian은 다시 카메라 투영을 통해 2D splat으로 변환된다:

$$\boldsymbol{\Sigma}' = \mathbf{J}\mathbf{W}\boldsymbol{\Sigma}_{3D|t}\mathbf{W}^T\mathbf{J}^T$$

최종 픽셀 색상은 알파 블렌딩으로 계산된다:

$$C(\mathbf{u}) = \sum_{i \in \mathcal{N}} c_i \cdot \alpha_i \cdot p_i(t) \cdot \prod_{j=1}^{i-1}(1-\alpha_j \cdot p_j(t))$$

여기서 $c_i$는 각 Gaussian의 색상, $\alpha_i$는 불투명도, $p_i(t)$는 시간 주변 분포 값이다.

#### (D) 4D Spherindrical Harmonics (4DSH)

구면조화함수(SH)의 4D 확장을 활용하여 각 Gaussian의 외관의 시간적 진화를 직접 표현한다. 색상은 $c_i(\mathbf{d}, \Delta t)$로 조작되며, 여기서 $\mathbf{d} = (\theta, \phi)$는 구면좌표계에서의 정규화된 뷰 방향이고 $\Delta t$는 시간 차이이다. 4DSH를 SH와 다양한 1D 기저 함수를 결합하여 구성하되, 계산 편의를 위해 푸리에 급수를 1D 기저 함수로 채택한다.

$$\text{4DSH}_{l,n}^{m}(\theta, \phi, \Delta t) = Y_l^m(\theta, \phi) \cdot F_n(\Delta t)$$

여기서 $Y_l^m$은 3D 구면조화함수이고, $n$은 푸리에 급수의 차수이다.

최종 색상:

$$c_i(\mathbf{d}, \Delta t) = \sum_{l=0}^{L}\sum_{m=-l}^{l}\sum_{n=0}^{N} a_{l,n}^{m} \cdot Y_l^m(\theta, \phi) \cdot F_n(\Delta t)$$

#### (E) 최적화 및 손실 함수

최적화에서는 렌더링 손실만을 감독 신호로 사용한다. 대부분의 경우 기본 학습 스케줄만으로 만족스러운 결과를 산출한다.

$$\mathcal{L} = \lambda_1 \mathcal{L}_1 + (1 - \lambda_1)\mathcal{L}_{\text{D-SSIM}}$$

여기서 $\mathcal{L}\_1 = \|\hat{I} - I\|\_1$은 렌더링된 이미지와 GT 이미지 간의 L1 손실, $\mathcal{L}_{\text{D-SSIM}}$은 구조적 유사도 손실이다.

추가적인 정규화(regularizer)나 모션 사전(prior)이 필요하지 않다. 모든 기하, 외관, 운동이 end-to-end로 학습되며 동적 분할(splitting)과 가지치기(pruning)가 자동 모델 적응을 제공한다.

### 2.3 모델 구조 개요

```
┌─────────────────────────────────────────────────────────┐
│             4D Gaussian Splatting (4DGS)                │
│                                                         │
│  입력: 2D 이미지 시퀀스 + 카메라 파라미터 + 시간 t      │
│                                                         │
│  ┌──────────────────────────────────────────────┐       │
│  │ 4D Gaussian 프리미티브                        │       │
│  │  μ = (μ_x, μ_y, μ_z, μ_t)                   │       │
│  │  Σ_{4D} = R_{4D} S_{4D} S_{4D}^T R_{4D}^T  │       │
│  │  4D Spherindrical Harmonics (4DSH)           │       │
│  │  불투명도 α                                   │       │
│  └──────────────────────────────────────────────┘       │
│                      ↓                                  │
│  ┌──────────────────────────────────────────────┐       │
│  │ 시간 t에서의 조건부 분해                       │       │
│  │  G_{4D} → G_{3D|t}(conditional) × p(t)(marginal)│    │
│  └──────────────────────────────────────────────┘       │
│                      ↓                                  │
│  ┌──────────────────────────────────────────────┐       │
│  │ Tile-Based 래스터라이저 (2D Splatting)         │       │
│  │  조건부 3D Gaussian → 2D Splat 투영           │       │
│  │  알파 블렌딩 + 시간 주변분포 가중              │       │
│  └──────────────────────────────────────────────┘       │
│                      ↓                                  │
│  출력: 렌더링된 이미지 (임의의 시점·시간)               │
└─────────────────────────────────────────────────────────┘
```

### 2.4 성능 향상

4DGS 모델은 Plenoptic Video 데이터셋에서 기존 SOTA 방법들을 정량적으로 크게 능가하며, 우수한 PSNR, DSSIM, LPIPS 점수를 달성하는 동시에 상당한 속도 향상을 보여주어 이 벤치마크에서 고품질 동적 새로운 뷰 합성을 위한 실시간 렌더링이 가능한 유일한 방법이다.

D-NeRF 합성 장면(monocular, under-constrained)에서 **PSNR = 34.09**를 실시간 프레임 레이트로 달성한다.

이 접근법은 복잡한 동적 장면에서 체적 효과(volumetric effects)와 가변적인 조명 조건을 포함한 고해상도 사실적 새로운 뷰의 end-to-end 학습과 실시간 렌더링을 지원하는 최초의 모델이다.

### 2.5 한계점

더 극적인 변화가 있는 장면에서는 temporal flickering과 jitter 같은 문제가 관찰된다. 이는 비최적 샘플링 기법에서 발생할 수 있다.

Real-Time4DGS는 수백만 개의 4D Gaussian을 생성하여 1GB 이상의 메모리를 소비하며, 각 프리미티브가 시간에 따라 진화하는 고차원 속성을 갖기 때문에 저장 요구량이 실용적 한계를 자주 초과한다. 특히 실시간 제약 하에서, 모바일 디바이스에서, 또는 스트리밍 시나리오에서 이러한 오버헤드는 다양한 다운스트림 작업을 복잡하게 만든다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 본래적 일반화 강점

이 접근법은 end-to-end 학습과 복잡한 동적 장면의 실시간 렌더링을 지원하는 최초의 모델이며, 제안된 표현은 해석 가능하고 공간 및 시간 차원 모두에서 고도로 확장 가능하고 적응적이다.

최적화 과정은 완전히 end-to-end이며, 전체 비디오를 처리할 수 있고, 기존의 프레임별 또는 다단계 훈련 접근법과 달리 임의의 시간과 뷰에서 샘플링할 수 있다.

이 접근법은 가변 길이 비디오와 end-to-end 학습에 대한 단순성, 유연성, 그리고 효율적 실시간 렌더링을 제공하여 복잡한 동적 장면 운동을 포착하기에 적합하다.

### 3.2 일반화 향상을 위한 구체적 설계

| 설계 요소 | 일반화 기여 |
|-----------|------------|
| **네이티브 4D 프리미티브** | 별도의 변형 필드나 프레임별 복제 없이 시공간을 명시적 4D Gaussian 집합으로 표현함으로써 모든 시공간 상관관계(운동, 시간적 가림, 외관 변화)가 네이티브하게 인코딩된다. |
| **4DSH** | Spherindrical Harmonics가 고주파 뷰·시간 효과를 처리하기 위한 간결하면서도 표현력 있는 기저를 제공하여 사실감을 확보한다. |
| **시공간 밀도 제어** | 4DGS 프레임워크는 공간적·시간적 그래디언트를 모두 사용하여 Gaussian 밀도를 적응적으로 제어하는 새로운 밀도화(densification) 전략을 포함한다. |
| **최소한의 가정** | 이 새로운 관점은 명시적 표현을 도입할 뿐 아니라, 운동이 어떻게 구성되는지에 대한 최소한의 가정만을 하여 다재다능한 동적 장면 학습 프레임워크를 위한 길을 연다. |

### 3.3 일반화 향상을 위한 향후 방향

**희소 카메라 입력을 위한 기하 일관성 확장**: 다시점 스테레오 사전(prior)을 통합하여 희소 카메라 입력에서의 일반화를 개선할 수 있다.

**모델 압축**: 가지치기(pruning), 양자화(quantization), 엔트로피 인식 인코딩을 통한 공격적 모델 압축으로 엣지 배포를 가능하게 한다.

**하이브리드 3D–4D 체계**: 정적 배경(순수 3D Gaussian)과 동적 요소를 분리하여 표현하는 방법도 연구되고 있다.

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 학술적·산업적 영향

4D Gaussian Splatting은 정적 또는 프레임 기반 접근법을 넘어 시공간 상관관계를 직접 활용하는 통합적이고 명시적이며 고효율의 프레임워크를 구성하며, 표현, 특징 분해, 경량 시간적 변형의 혁신을 통해 동적 장면 재구성 및 합성에서 최첨단 렌더링 속도와 메모리 효율을 달성한다.

명시적이고 조작 가능한 Gaussian 표현은 고급 편집, 추적, 실시간 제어 응용—로보틱스 및 동적 환경 이해에 관련된—에 대한 가능성을 가진다.

자유 시점 비디오, 자율 주행 시뮬레이션, VR/AR 등 시공간적 일관성과 실시간 렌더링이 중요한 광범위한 응용에 문을 연다.

### 4.2 향후 연구 시 고려할 점

1. **메모리·저장 효율성**: 현재 4D Gaussian 표현은 상당한 계산 비용과 메모리 사용량을 수반하며, 이 오버헤드가 다양한 다운스트림 작업을 복잡하게 하므로 시각적 충실도와 렌더링 속도를 유지하면서 효과적인 저장 감소 기법이 필요하다.

2. **시간적 안정성**: 극적인 변화가 있는 장면에서의 temporal flickering 및 jitter 문제를 해결해야 하며, 사전 정규화 대신 시간 배치 샘플링이 효과적인 것으로 밝혀졌다.

3. **도시·대규모 장면**: 희소 관측과 넓은 범위를 가진 도시 장면에서의 적용 가능성과 재구성 품질에 대한 연구가 필요하다.

4. **프레임워크 통합**: 변형(deformation) 기반 방법은 MLP를 변형 네트워크로 삽입하여 프레임 간 3D Gaussian의 변형을 예측하고, 4D 프리미티브 기반 방법은 시간 차원을 통합하여 특정 타임스탬프에서 4D → 3D로 샘플링하고 정규화를 통해 시간적 일관성을 보장하는데, 두 접근의 장점을 결합하는 연구가 유망하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연도 | 방법 | 핵심 접근 | 장점 | 한계 |
|------|------|----------|------|------|
| 2020 | **NeRF** (Mildenhall et al.) | MLP 기반 암시적 복사장 | 고품질 뷰 합성의 시초 | 학습·렌더링 매우 느림 |
| 2020 | **D-NeRF** (Pumarola et al.) | Canonical space + deformation field | 단안 동적 장면 처리 | 실시간 불가, 복잡한 움직임에 약함 |
| 2022 | **TiNeuVox** (Fang et al., SIGGRAPH Asia) | 시간 인식 neural voxels | 빠른 동적 복사장 학습 | 해상도·품질 제한 |
| 2023 | **HexPlane** / **K-Planes** (Fridovich-Keil et al.) | 분해된 시공간 평면 | 효율적 4D 특징 표현 | 실시간 렌더링 미달성 |
| 2023 | **3D Gaussian Splatting** (Kerbl et al., SIGGRAPH) | 명시적 3D Gaussian + tile-based rasterizer | 정적 장면 실시간 고품질 | 동적 장면 미지원 |
| 2024 | **4DGS (Yang et al., ICLR)** ⬅ 본 논문 | 네이티브 4D Gaussian + 4DSH | 실시간 + end-to-end + 고품질 | 메모리 소비 큼, 극적 변화 시 flickering |
| 2024 | **4D-GS (Wu et al., CVPR)** | Canonical 3D Gaussian + Gaussian deformation field로 운동·형태 변화 표현, 시공간 구조 인코더와 극소형 multi-head 디코더 사용. 하나의 canonical 3D Gaussian 집합만 유지하며 각 타임스탬프마다 새 위치·형태로 변환 | 빠른 학습 (8분), 82 FPS | deformation field 의존 |
| 2024 | **4DRotorGS** (Duan et al.) | 비등방적 4D XYZT Gaussian으로 동적 장면을 표현하며 temporal slicing으로 각 타임스탬프의 동적 3D Gaussian을 구성 | RTX 3090에서 최대 277 FPS, RTX 4090에서 583 FPS | 특정 장면 유형에 최적화 |
| 2025 | **OMG (Optimized Minimal 4DGS)** | Real-Time4DGS 위에 구축, Gaussian 샘플링·가지치기·병합·속성 압축을 통한 최소 수의 Gaussian으로 고충실도 동적 장면 표현 | 저장 공간 대폭 절감 | 품질-압축 트레이드오프 |

### 패러다임 변화 요약

```
NeRF (2020, 암시적/느림) 
    → D-NeRF/HexPlane (2020-2023, 동적 확장/여전히 느림)
        → 3D-GS (2023, 명시적/실시간/정적만)
            → 4DGS 본 논문 (2024, 네이티브 4D/실시간/동적)
                → OMG/압축 방법 (2025, 효율화)
```

동적 장면의 렌더링과 재구성은 가상현실 및 컴퓨터 그래픽스에서 핵심적이며, 따라서 정적에서 동적 장면으로의 neural rendering 방법 확장이 중요한 연구 영역이 되었다.

---

## 참고 자료 및 출처

1. **Yang, Z., Yang, H., Pan, Z., & Zhang, L.** (2024). "Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting." *ICLR 2024*. [arXiv:2310.10642](https://arxiv.org/abs/2310.10642)
2. **Yang, Z., Pan, Z., Zhu, X., Zhang, L., et al.** (2024). "4D Gaussian Splatting: Modeling Dynamic Scenes with Native 4D Primitives." *arXiv extended version*. [arXiv:2412.20720](https://arxiv.org/html/2412.20720v2)
3. **Wu, G., Yi, T., Fang, J., et al.** (2024). "4D Gaussian Splatting for Real-Time Dynamic Scene Rendering." *CVPR 2024*. [CVPR PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_4D_Gaussian_Splatting_for_Real-Time_Dynamic_Scene_Rendering_CVPR_2024_paper.pdf)
4. **Duan, Y., Wei, F., et al.** (2024). "4D-Rotor Gaussian Splatting: Towards Efficient Novel View Synthesis for Dynamic Scenes." [Project Page](https://weify627.github.io/4drotorgs/)
5. **ICLR 2024 공식 논문 PDF**: [proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2024/file/26230ff5299de0929d03ed3576c3bbf9-Paper-Conference.pdf)
6. **OpenReview**: [Real-time 4DGS Review](https://openreview.net/forum?id=WhgB5sispV)
7. **프로젝트 페이지**: [fudan-zvg.github.io/4d-gaussian-splatting](https://fudan-zvg.github.io/4d-gaussian-splatting/)
8. **GitHub 공식 구현**: [github.com/fudan-zvg/4d-gaussian-splatting](https://github.com/fudan-zvg/4d-gaussian-splatting)
9. **Emergent Mind 분석**: [4D Gaussian Splatting (4D-GS)](https://www.emergentmind.com/topics/4d-gaussian-splatting-4d-gs)
10. **OMG (2025)**: "Optimized Minimal 4D Gaussian Splatting." [arXiv:2510.03857](https://arxiv.org/html/2510.03857v1)
11. **Dynamic Scene Reconstruction Survey (2025)**: [arXiv:2503.08166](https://arxiv.org/html/2503.08166v1)
12. **A Survey on 3D Gaussian Splatting**: Chen et al. [arXiv:2401.03890](https://arxiv.org/html/2401.03890v7)
13. **Liner Quick Review**: [liner.com/review](https://liner.com/review/realtime-photorealistic-dynamic-scene-representation-and-rendering-with-4d-gaussian)

> **참고**: 본 분석은 위 출처들을 종합하여 작성되었으며, 논문의 구체적인 수치 결과(Table 등)는 원 논문 PDF를 직접 확인하시기 바랍니다. 수식의 일부 세부 표기는 원문의 notation과 약간 다를 수 있으나 의미적으로 동일합니다.
