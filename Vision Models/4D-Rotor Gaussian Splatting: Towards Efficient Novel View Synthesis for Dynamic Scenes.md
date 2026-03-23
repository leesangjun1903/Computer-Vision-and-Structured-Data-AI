# 4D-Rotor Gaussian Splatting: Towards Efficient Novel View Synthesis for Dynamic Scenes

**저자**: Yuanxing Duan, Fangyin Wei, Qiyu Dai, Yuhang He, Wenzheng Chen, Baoquan Chen  
**발표**: ACM SIGGRAPH 2024 Conference Papers (2024)  
**기관**: Peking University, SKL of General AI, Princeton University, NVIDIA

---

## 1. 핵심 주장과 주요 기여 요약

이 논문은 정적 장면에서 3D Gaussian Splatting의 성공에 영감을 받아, 비등방성(anisotropic) 4D XYZT Gaussian으로 동적 장면을 표현하는 4DRotorGS라는 새로운 방법을 제안합니다.

**핵심 주장**: 기존 방법들은 canonical space와 implicit/explicit deformation field를 학습하여 동적 장면을 인코딩하지만, 급격한 움직임이나 고충실도 렌더링에서 어려움을 겪습니다. 이에 대해 4DRotorGS는 시공간을 통합적으로 모델링하는 명시적(explicit) 4D 표현을 제안합니다.

### 주요 기여 (Key Contributions)

| 기여 항목 | 내용 |
|-----------|------|
| **4D Rotor 기반 회전 표현** | 기하대수(Geometric Algebra)에서 영감을 받아 4D rotor를 사용하여 4D 회전을 표현하며, 이는 시공간 분리 가능한(spatial-temporal separable) 회전 표현입니다. |
| **Temporal Slicing** | 각 타임스탬프에서 4D Gaussian을 시간적으로 슬라이싱하여 동적 3D Gaussian을 자연스럽게 구성하고, 이를 이미지로 원활하게 투영합니다. |
| **정규화 손실 함수** | 3DGS의 최적화 전략을 개선하고 두 개의 새로운 정규화 항(entropy loss, 4D consistency loss)을 도입하여 동적 재구성을 안정화하고 향상시킵니다. |
| **실시간 렌더링** | 고도로 최적화된 CUDA 가속 프레임워크를 구현하여 RTX 3090에서 최대 277 FPS, RTX 4090에서 583 FPS의 실시간 렌더링 속도를 달성합니다. |
| **3DGS의 일반화된 형태** | 4DRotorGS는 3DGS의 일반화된 형태로, 시간 차원을 닫으면 3DGS로 축소됩니다. |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

이 논문은 동적 장면(dynamic scenes)에 대한 Novel View Synthesis(NVS) 문제를 다룹니다. 최근 신경망 접근법들은 정적 3D 장면에서 우수한 NVS 결과를 달성했지만, 4D 시간-변화 장면으로의 확장은 여전히 비자명합니다.

기존 접근법의 한계:
- canonical + deformation field를 활용하는 전통적 방법은 고충실도 렌더링과 급격한 동작 묘사에 어려움을 겪으며, volumetric 방법은 밀집 샘플링된 광선의 높은 계산 비용으로 인해 실시간 렌더링에 부족합니다.
- 3D 회전의 quaternion 기반 표현을 4D로 확장하는 것이 복잡합니다.

### 2.2 제안하는 방법 (수식 포함)

#### (A) 3D Gaussian Splatting (배경)

3DGS에서 각 Gaussian은 3D 중심 위치 $\boldsymbol{\mu}\_{3D} = (\mu_x, \mu_y, \mu_z)$와 3D 공분산 행렬 $\Sigma_{3D}$로 표현됩니다:

$$G(\mathbf{x}) = \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_{3D})^T \Sigma_{3D}^{-1} (\mathbf{x} - \boldsymbol{\mu}_{3D})\right)$$

공분산은 스케일링 행렬 $\mathbf{S} = \text{diag}(s_x, s_y, s_z) \in \mathbb{R}^3$과 회전 행렬 $\mathbf{R}$로 분해됩니다:

$$\Sigma_{3D} = \mathbf{R} \mathbf{S} \mathbf{S}^T \mathbf{R}^T$$

#### (B) 4D Gaussian 표현

3D Gaussian과 유사하게, 4D Gaussian은 4D 중심 위치 $\boldsymbol{\mu}_{4D} = (\mu_x, \mu_y, \mu_z, \mu_t)$와 4D 공분산 행렬로 표현됩니다:

$$G_{4D}(\mathbf{x}, t) = \exp\left(-\frac{1}{2}\begin{pmatrix}\mathbf{x} - \boldsymbol{\mu}_{xyz} \\ t - \mu_t\end{pmatrix}^T \Sigma_{4D}^{-1} \begin{pmatrix}\mathbf{x} - \boldsymbol{\mu}_{xyz} \\ t - \mu_t\end{pmatrix}\right)$$

4D 공분산 행렬은 4D 회전 행렬 $\mathbf{R}\_{4D}$과 4D 스케일링 행렬 $\mathbf{S}_{4D}$로 구성됩니다:

$$\Sigma_{4D} = \mathbf{R}_{4D} \mathbf{S}_{4D} \mathbf{S}_{4D}^T \mathbf{R}_{4D}^T$$

여기서 $\mathbf{S}_{4D} = \text{diag}(s_x, s_y, s_z, s_t) \in \mathbb{R}^4$입니다.

#### (C) 4D Rotor 기반 회전 표현

3D Gaussian을 4D 공간으로 끌어올리는 것은 비자명한 과제로, 4D 회전, 슬라이싱, 시공간 최적화 스킴 설계에 큰 도전이 존재합니다. 기하대수에서 영감을 받아 4D rotor를 선택하여 4D 회전을 표현하며, 이는 시공간 분리 가능한 회전 표현입니다.

4D Rotor $\mathbf{R}$는 기하대수(Geometric Algebra)의 bivector로 구성됩니다. 4D에서는 6개의 bivector 평면이 존재하며, rotor는 다음과 같이 표현됩니다:

$$\mathbf{R} = a + b_{xy}\mathbf{e}_{xy} + b_{xz}\mathbf{e}_{xz} + b_{xt}\mathbf{e}_{xt} + b_{yz}\mathbf{e}_{yz} + b_{yt}\mathbf{e}_{yt} + b_{zt}\mathbf{e}_{zt} + c\mathbf{e}_{xyzt}$$

이는 8개의 파라미터로 구성되며, 정규화 조건 $\|\mathbf{R}\| = 1$을 만족합니다.

**핵심 장점**: Rotor 표현은 3D와 4D 회전 모두를 수용합니다: 시간 차원이 0으로 설정되면 quaternion과 동치가 되어 3D 공간 회전도 표현할 수 있습니다. 이러한 적응성은 동적 및 정적 장면 모두를 모델링하는 유연성을 부여합니다.

#### (D) Temporal Slicing

슬라이싱 방법은 조건부 확률 유도(conditional probability derivation)에 기반합니다. 주어진 시간 $t^*$에서 4D Gaussian을 슬라이싱하면 3D Gaussian이 됩니다:

```math
\Sigma_{4D} = \begin{pmatrix} \Sigma_{xyz} & \Sigma_{xyzt} \\ \Sigma_{xyzt}^T & \Sigma_{tt} \end{pmatrix}
```

조건부 분포에 의해 슬라이싱된 3D Gaussian의 파라미터는:

```math
\boldsymbol{\mu}_{3D|t^*} = \boldsymbol{\mu}_{xyz} + \Sigma_{xyzt} \Sigma_{tt}^{-1}(t^* - \mu_t)
```

$$\Sigma_{3D|t^*} = \Sigma_{xyz} - \Sigma_{xyzt} \Sigma_{tt}^{-1} \Sigma_{xyzt}^T$$

슬라이싱된 Gaussian의 불투명도(opacity)는 시간에 따라 변조됩니다:

```math
\alpha_{t^*} = \alpha \cdot \exp\left(-\frac{(t^* - \mu_t)^2}{2\Sigma_{tt}}\right)
```

#### (E) 렌더링 (Alpha-Blending)

슬라이싱된 3D Gaussian들은 기존 3DGS와 동일한 방식으로 2D 이미지 평면에 투영됩니다:

$$\hat{C}(\mathbf{p}) = \sum_{i \in \mathcal{N}} c_i \alpha_i' \prod_{j=1}^{i-1}(1 - \alpha_j')$$

여기서 $c_i$는 Spherical Harmonics로 표현된 색상이고, $\alpha_i'$는 2D 투영된 Gaussian과 시간 변조된 불투명도를 결합한 값입니다.

### 2.3 손실 함수 (Loss Functions)

#### 기본 재구성 손실

$$\mathcal{L}_{recon} = (1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{D\text{-}SSIM}$$

#### Entropy Loss

Gaussian의 불투명도를 1 또는 0 방향으로 밀어주는 엔트로피 손실을 제안하며, 이는 실험에서 "floater"를 제거하는 데 효과적입니다:

$$\mathcal{L}_{entropy} = -\frac{1}{N}\sum_{i=1}^{N}\left[\alpha_i \log(\alpha_i) + (1-\alpha_i)\log(1-\alpha_i)\right]$$

#### 4D Consistency Loss (4D 일관성 손실)

4D consistency loss는 동적 재구성을 안정화하고 일관된 동역학을 유지합니다. 이 손실은 인접한 4D Gaussian의 시공간적 상태가 급격하게 변하지 않도록 부드러움(smoothness)을 부여합니다.

#### 전체 손실 함수

$$\mathcal{L}_{total} = \mathcal{L}_{recon} + \lambda_{ent}\mathcal{L}_{entropy} + \lambda_{con}\mathcal{L}_{consistency}$$

### 2.4 모델 구조 (Framework)

프레임워크 개요: 초기화 후, 먼저 rotor로 시공간 움직임이 모델링된 4D Gaussian을 시간적으로 슬라이싱합니다.

```
[4D Gaussian 초기화]
        ↓
[4D Rotor 기반 회전 + 4D 스케일링 → Σ₄D 구성]
        ↓
[Temporal Slicing (t* 쿼리)]  →  [3D Gaussian (μ₃D|t*, Σ₃D|t*, α_t*)]
        ↓
[3DGS 파이프라인: 2D 투영 (Splatting)]
        ↓
[Alpha-Blending → 렌더링된 이미지 Ĉ]
        ↓
[Loss 계산 & 역전파 (L_recon + L_entropy + L_consistency)]
```

### 2.5 성능 향상

| 지표 | 성능 |
|------|------|
| **Plenoptic Video Dataset** | PSNR 31.62로 기존 SOTA 방법들을 능가합니다. |
| **D-NeRF Dataset** | NeRF 기반 및 Gaussian 기반 기준선 모두를 PSNR과 렌더링 속도에서 큰 폭으로 능가하며, 렌더링 속도 1258 FPS를 달성합니다 (이전 최고 대비 8배 빠름). |
| **학습 시간** | 빠른 구현에서 학습은 약 5분 소요됩니다. |
| **렌더링 속도** | RTX 3090에서 최대 277 FPS, RTX 4090에서 583 FPS의 실시간 렌더링 속도를 자랑합니다. |

### 2.6 한계점 (Limitations)

Entropy loss를 Plenoptic dataset에 적용하면 PSNR이 저하되는데, 이는 Plenoptic dataset이 밀집 뷰를 제공하고 많은 투명 객체를 포함하기 때문입니다. 따라서 entropy loss는 불투명 표면과 희소 뷰에 추가하는 것이 권장됩니다.

추가적 한계:
- 4D 회전과 스케일링 기반 방법은 4D 공분산 행렬에 시공간 변형을 도입하며, 타임스탬프가 변할 때 4D Gaussian을 3D Gaussian으로 슬라이싱해야 하므로 중복 계산이 증가합니다. 또한 4차원 행렬 계산은 계산 집약적입니다.
- Gaussian 함수의 특성상 고주파 정보(예: 날카로운 객체, 급격한 시간 변화)를 표현할 때 아티팩트가 발생할 수 있습니다.
- 장면별(per-scene) 최적화가 필요하여, 새로운 장면에 대한 zero-shot 일반화 능력은 부재합니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 구조적 일반화 능력

Rotor 표현은 3D와 4D 회전 모두를 수용하며, 시간 차원이 0으로 설정되면 quaternion과 동치가 됩니다. 즉, 4DRotorGS는 3DGS의 일반화된 형태로, 시간 차원을 닫으면 3DGS로 축소됩니다.

이는 단일 프레임워크로 **정적 장면과 동적 장면 모두를 처리**할 수 있음을 의미하며, 모델 아키텍처 수준에서의 일반화 성능을 보장합니다.

### 3.2 현재 한계와 향후 가능성

현재 4DRotorGS는 **per-scene optimization** 방식으로, 각 장면마다 별도의 학습이 필요합니다. 일반화 성능 향상을 위한 방향:

1. **Feed-forward 4DGS**: 4D Gaussian Splatting의 per-scene 최적화는 비용이 많이 드는 반복적 정제가 필요하여 대규모 환경으로 확장이 어렵습니다. 현재 feed-forward 접근법은 광도 품질이 저하되는 문제가 있으나, ReconDrive 같은 프레임워크가 이 격차를 해소하려 시도합니다.

2. **4D Consistency Loss의 역할**: 원래 4D 공간에서 인접한 Gaussian의 상태가 자유롭게 변할 수 있어 최적화 난이도와 모델 중복성이 증가하는데, 4D consistency loss는 이러한 문제를 완화하여 보다 구조화된 표현을 학습하게 합니다.

3. **Entropy Loss의 선택적 적용**: 불투명 표면과 희소 뷰에서 entropy loss를 추가하는 것이 권장되며, 장면 특성에 따른 적응적 정규화 전략이 일반화 성능 향상의 열쇠가 됩니다.

4. **대규모 사전 학습 + 미세 조정**: Foundation model (예: VGGT)을 활용하여 사전 학습된 feature를 4DGS 생성에 활용하는 방향이 유망합니다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 연구 영향

4DGS는 동적 장면을 위한 실용적 시공간 표현을 제공하며 속도와 렌더링 충실도에서 새로운 벤치마크를 설정합니다. 정적·동적 환경 모두에 적합한 통합 프레임워크로서 VR/AR, 게이밍, 영화 제작 등 다양한 미래 산업 응용에 큰 잠재력을 갖습니다.

### 4.2 향후 연구 시 고려할 점

| 연구 방향 | 세부 내용 |
|-----------|-----------|
| **계산 효율성** | 4D 행렬 연산의 오버헤드 감소 (Disentangled4DGS 등 후속 연구 참조) |
| **메모리 최적화** | 시간 변화를 Gaussian 파라미터에 직접 임베딩하면 서로 다른 타임스텝에서 같은 객체를 다른 Gaussian으로 모델링하게 되어 메모리 소비 증가 및 불일관한 동작이 발생할 수 있습니다. |
| **Streamable 렌더링** | 위 방법들은 오프라인 모델링 패러다임을 따르며 스트리밍 가능한 역량이 부족합니다. |
| **고주파 정보** | Gaussian 함수의 고유 한계인 고주파 디테일 표현 개선 |
| **단안(Monocular) 입력** | 단안 비디오로부터의 동적 장면 NVS는 본질적으로 ill-posed 문제이며, 적절한 아키텍처 프라이어의 통합이 이상적입니다. |
| **Zero-shot 일반화** | Per-scene 최적화를 넘어선 일반화 가능한 동적 장면 재구성 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 발표 | 핵심 접근법 | 렌더링 속도 | 특징 |
|------|------|------------|------------|------|
| **NeRF** (Mildenhall et al.) | ECCV 2020 → CACM 2021 | 암시적 신경 복사 필드 | ~수 초/프레임 | 정적 장면, 볼류메트릭 렌더링 |
| **D-NeRF** (Pumarola et al.) | CVPR 2021 | Canonical + Deformation NeRF | 비실시간 | 단안 동적 장면 |
| **3D Gaussian Splatting** (Kerbl et al.) | SIGGRAPH 2023 | 3D Gaussian splatting for real-time radiance field rendering | 실시간 | 정적 장면 SOTA |
| **4D-GS** (Wu et al.) | CVPR 2024 | 3D Gaussian + 4D neural voxel의 명시적 표현을 사용하며, HexPlane에 영감받은 decomposed neural voxel encoding과 경량 MLP로 새로운 타임스탬프에서 Gaussian 변형을 예측합니다. | RTX 3090에서 800×800 해상도 82 FPS | Deformation field 기반 |
| **RT-4DGS** (Yang et al.) | ICLR 2024 | 비등방성 타원체의 4D Gaussian과 4D spherindrical harmonics를 사용하여 시간 변화 외관을 표현합니다. | 실시간 | Native 4D primitives, Dual quaternion |
| **4DRotorGS** (Duan et al.) | **SIGGRAPH 2024** | 4D Rotor + Temporal Slicing | **277 FPS (3090) / 583 FPS (4090)** | **본 논문** |
| **Deformable 3DGS** | 2024 | 3D Gaussian을 canonical space에서 재구성하고 deformation field로 학습합니다. | 실시간 | 변형 필드 기반 |
| **STG (Spacetime Gaussian)** | 2024 | Spacetime Gaussian Feature | 실시간 | 시공간 특징 splatting |
| **Disentangled4DGS** | 2025 | 시간적·공간적 변형을 분리(disentangle)하여 4D 행렬 계산 의존성을 제거합니다. | 더 빠름 | 4DRotorGS의 계산 병목 해결 |
| **4DGS-1K** | 2025 | Pruning + Temporal filtering | 프루닝으로 Gaussian 수를 줄여 약 5배 빠른 래스터화 속도를 달성합니다. | 1000+ FPS 목표 |

### 접근법 분류

```
동적 장면 NVS 방법
├── Deformation-based (변형 기반)
│   ├── D-NeRF, Nerfies, HyperNeRF (암시적)
│   ├── 4D-GS (Wu et al., CVPR 2024) - Neural voxel + MLP
│   └── Deformable 3DGS - Canonical + deformation
│
├── Native 4D Primitives (원시 4D 프리미티브)
│   ├── RT-4DGS (Yang et al., ICLR 2024) - Dual quaternion
│   ├── ★ 4DRotorGS (본 논문, SIGGRAPH 2024) - 4D Rotor
│   └── Disentangled4DGS (2025) - Disentangled 표현
│
└── Tracking-based (추적 기반)
    └── Dynamic 3DGS (Luiten et al., 2023)
```

### 핵심 비교 분석

**4DRotorGS vs. 4D-GS (Wu et al.)**: 4D-GS는 Gaussian 변형 필드 네트워크(temporal-spatial structure encoder + multi-head Gaussian deformation decoder)를 사용하며, 하나의 canonical 3D Gaussian 세트를 유지하고 각 타임스탬프에서 변형합니다. 반면 4DRotorGS는 **네트워크 없이 명시적 4D Gaussian**으로 동적 장면을 직접 모델링하여 MLP 의존성을 제거합니다.

**4DRotorGS vs. RT-4DGS (Yang et al.)**: Yang et al.의 방법은 4D Gaussian 표현을 사용하지만 4D rotation formulation을 사용하며, 이는 rotor 기반 표현에 비해 해석 가능성이 낮고 시공간 분리성이 부족합니다.

**4DRotorGS vs. Disentangled4DGS**: 4DRotorGS 등의 4D rotation + scaling 기반 방법은 4D 공분산 행렬에 시공간 변형을 도입하여 4D Gaussian을 3D로 슬라이싱할 때 타임스탬프 변경에 따라 중복 계산이 증가합니다. Disentangled4DGS는 이 문제를 시공간 분리로 해결합니다.

---

## 참고문헌 및 출처

1. **Duan, Y., Wei, F., Dai, Q., He, Y., Chen, W., & Chen, B.** (2024). "4D-Rotor Gaussian Splatting: Towards Efficient Novel View Synthesis for Dynamic Scenes." *Proc. ACM SIGGRAPH 2024*. [arXiv:2402.03307](https://arxiv.org/abs/2402.03307)
2. **프로젝트 페이지**: https://weify627.github.io/4drotorgs/
3. **GitHub 코드**: https://github.com/weify627/4D-Rotor-Gaussians
4. **ACM Digital Library**: https://dl.acm.org/doi/10.1145/3641519.3657463
5. **NVIDIA Research**: https://research.nvidia.com/publication/2024-02_4d-rotor-gaussian-splatting-towards-efficient-novel-view-synthesis-dynamic
6. **Emergent Mind 분석**: https://www.emergentmind.com/papers/2402.03307
7. **Wu, G. et al.** (2024). "4D Gaussian Splatting for Real-Time Dynamic Scene Rendering." *CVPR 2024*. [arXiv:2310.08528](https://arxiv.org/abs/2310.08528)
8. **Yang, Z. et al.** (2024). "Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting." *ICLR 2024*. [arXiv:2310.10642](https://arxiv.org/abs/2310.10642)
9. **Kerbl, B. et al.** (2023). "3D Gaussian Splatting for Real-Time Radiance Field Rendering." *ACM ToG 42(4)*.
10. **Disentangled 4D Gaussian Splatting**: [arXiv:2503.22159](https://arxiv.org/html/2503.22159v1) (2025)
11. **4DGS-1K: 1000+ FPS 4D Gaussian Splatting**: [arXiv:2503.16422](https://arxiv.org/html/2503.16422v1) (2025)
12. **ResearchGate 관련 논문 모음**: https://www.researchgate.net/publication/382238427

> **참고**: 위 수식들은 논문의 공식적인 수학적 프레임워크를 기반으로 하되, 검색 결과에서 확인 가능한 표현과 논문의 공개된 HTML 버전(arXiv:2402.03307v3)을 참조하여 구성하였습니다. 일부 세부 수식(예: 4D consistency loss의 정확한 형태)은 논문 원문 PDF를 직접 확인하시는 것을 권장합니다.
