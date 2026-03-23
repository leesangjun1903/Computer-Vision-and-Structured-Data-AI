# Mesh-based Gaussian Splatting for Real-time Large-scale Deformation

**논문 정보**: Lin Gao, Jie Yang, Bo-tao Zhang, Jia-mu Sun, Yu-jie Yuan, Hongbo Fu, Yu-Kun Lai  
**게재**: ACM Transactions on Graphics (SIGGRAPH Asia 2024), arXiv:2402.04796 (2024.02)  
**코드**: https://github.com/IGLICT/GaussianMesh

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장
Neural implicit representations(NDF, NeRF)은 복잡한 기하학 및 토폴로지를 가진 표면 재구성과 새로운 뷰 생성에서 뛰어난 성능을 보이지만, 실시간으로 대규모 변형을 직접 수행하기가 어렵다. 또한 Gaussian Splatting(GS)은 명시적 기하학 표현으로 정적 장면의 실시간 고품질 렌더링에 유리하지만, 이산적(discrete) Gaussian 분포의 사용과 명시적 토폴로지의 부재로 인해 쉽게 변형할 수 없다.

### 1.2 주요 기여

| 기여 | 설명 |
|------|------|
| **Mesh-based GS 표현** | 핵심 아이디어는 혁신적인 mesh 기반 GS 표현을 설계하여 Gaussian 학습과 조작에 통합하는 것이다. |
| **양방향 바인딩** | 3D Gaussians는 명시적 메시 위에 정의되며, 양방향으로 결합된다: 3D Gaussian의 렌더링이 메시 면 분할(adaptive refinement)을 안내하고, 메시 면 분할이 3D Gaussian의 분할을 지시한다. |
| **정규화 효과** | 명시적 메시 제약은 Gaussian 분포를 정규화하여, 정렬 불량 Gaussian이나 길고 좁은 형태의 Gaussian 등의 불량 Gaussian을 억제함으로써 시각적 품질을 향상시키고 변형 시 아티팩트를 방지한다. |
| **대규모 변형 기법** | 연관된 메시의 조작에 따라 3D Gaussian의 파라미터를 변경하는 대규모 Gaussian 변형 기법을 도입한다. |
| **데이터 기반 변형** | 기존 메시 변형 데이터셋을 활용하여 보다 사실적인 데이터 기반 Gaussian 변형이 가능하다. |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

Neural implicit representations의 실시간 대규모 변형의 어려움과, Gaussian Splatting이 이산적 Gaussian 사용 및 명시적 토폴로지 부재로 인해 쉽게 변형될 수 없는 문제를 해결하고자 한다.

구체적으로 다음 문제들을 다룬다:
1. **토폴로지 부재 문제**: 기존 3DGS는 개별 Gaussian들 사이에 연결 구조가 없어 일관된 변형이 불가능
2. **변형 시 아티팩트**: Gaussian들이 표면에 정렬되지 않거나 비정상적 형태를 가질 때 변형 후 시각적 아티팩트 발생
3. **실시간성 확보**: 대규모 변형을 수행하면서도 실시간 렌더링 속도 유지

### 2.2 제안하는 방법 (수식 포함)

#### (A) 3D Gaussian Splatting 기초

3DGS에서 각 3D Gaussian은 평균 위치 $\boldsymbol{\mu} \in \mathbb{R}^3$와 공분산 행렬 $\boldsymbol{\Sigma} \in \mathbb{R}^{3\times3}$로 정의된다:

$$G(\mathbf{x}) = e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})}$$

공분산 행렬은 회전 행렬 $\mathbf{R}$과 스케일 행렬 $\mathbf{S}$로 분해된다:

$$\boldsymbol{\Sigma} = \mathbf{R}\mathbf{S}\mathbf{S}^T\mathbf{R}^T$$

각 Gaussian은 속성 벡터 $\{\boldsymbol{\mu}, \mathbf{q}, \mathbf{s}, \alpha, \mathbf{c}\}$를 가진다 (위치, 쿼터니언 회전, 스케일, 불투명도, 색상/SH 계수).

렌더링 시 색상 $\mathbf{C}$는 depth-sorting 후 alpha-compositing으로 합성된다:

$$\mathbf{C} = \sum_{i=1}^{N} \mathbf{c}_i \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j)$$

#### (B) Mesh-based Gaussian 표현

핵심 아이디어는 mesh 기반 GS 표현을 설계하여 Gaussian 학습과 조작에 통합하는 것으로, 3D Gaussians가 명시적 메시 위에 정의되며 양방향으로 결합된다.

**메시-가우시안 바인딩**: 메시의 삼각형 면 $f_k$에 하나 이상의 Gaussian이 대응된다. 삼각형 면의 꼭짓점 $\{\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3\}$에 대해, 대응되는 Gaussian의 위치 $\boldsymbol{\mu}$가 barycentric 좌표 $(u, v, w)$로 표현된다:

$$\boldsymbol{\mu} = u \cdot \mathbf{v}_1 + v \cdot \mathbf{v}_2 + w \cdot \mathbf{v}_3, \quad u+v+w=1$$

**메시 제약에 의한 정규화**: Gaussian의 스케일 $\mathbf{s}$와 회전 $\mathbf{q}$는 해당 삼각형 면의 법선 벡터 $\mathbf{n}_f$와 접선 방향으로 제약된다. 스케일의 법선 방향 성분 $s_n$이 제한되어 Gaussian이 표면 근처에 밀착된다:

$$s_n \leq \epsilon \cdot \bar{s}_t$$

여기서 $\bar{s}_t$는 접선 방향 스케일의 평균이고, $\epsilon$은 소규모 상수이다.

#### (C) Adaptive Mesh-Gaussian Splitting

메시는 Gaussian splitting과 함께 적응적으로 세분화(refine)된다. 구체적으로:

- **Gaussian → Mesh**: 렌더링 gradient가 높은 영역의 Gaussian이 분할될 때, 해당 메시 면도 함께 분할
- **Mesh → Gaussian**: 메시 면이 분할되면, 대응되는 Gaussian도 해당 새로운 면들에 맞게 분할

분할 조건은 기존 3DGS의 adaptive density control을 따르되, 메시 기반 제약이 추가된다:

$$\nabla_{\boldsymbol{\mu}} \mathcal{L} > \tau_{\text{grad}} \implies \text{Split}(G_i, f_k)$$

여기서 $\tau_{\text{grad}}$는 gradient 임계값이다.

#### (D) 대규모 변형 (Large-scale Deformation)

메시 변형 시, 연관된 메시의 조작에 따라 3D Gaussian의 파라미터를 변경하는 대규모 Gaussian 변형 기법을 도입한다.

메시 삼각형 $f_k$가 변형 전 꼭짓점 $\{\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3\}$에서 변형 후 $\{\mathbf{v}'_1, \mathbf{v}'_2, \mathbf{v}'_3\}$로 변환될 때, 각 삼각형에 대한 로컬 아핀 변환 행렬 $\mathbf{T}_k$를 추출한다:

$$\mathbf{T}_k = \mathbf{V}'_k \mathbf{V}_k^{-1}$$

여기서 $\mathbf{V}_k$, $\mathbf{V}'_k$는 각각 변형 전후의 에지 벡터 행렬이다.

변형된 Gaussian 파라미터는 다음과 같이 갱신된다:

**위치 갱신** (barycentric 좌표 보존):
$$\boldsymbol{\mu}' = u \cdot \mathbf{v}'_1 + v \cdot \mathbf{v}'_2 + w \cdot \mathbf{v}'_3$$

**회전 갱신** ($\mathbf{T}_k$의 극분해로부터):
$$\mathbf{T}_k = \mathbf{R}_k \mathbf{U}_k \quad \implies \quad \mathbf{q}' = \mathbf{R}_k \cdot \mathbf{q}$$

**스케일 갱신** (로컬 스트레칭 반영):
$$\mathbf{s}' = \mathbf{U}_k \cdot \mathbf{s}$$

여기서 $\mathbf{R}_k$는 회전 성분, $\mathbf{U}_k$는 스트레치(대칭) 성분이다.

이 방법은 ARAP (As-Rigid-As-Possible) 변형의 일관성을 유지하면서도 3DGS의 고품질과 효율성을 보존한다. ARAP 에너지 함수는 다음과 같다:

$$E_{\text{ARAP}} = \sum_i \sum_{j \in \mathcal{N}(i)} w_{ij} \|(\mathbf{p}'_i - \mathbf{p}'_j) - \mathbf{R}_i(\mathbf{p}_i - \mathbf{p}_j)\|^2$$

여기서 $\mathbf{R}_i$는 최적화하고자 하는 회전 행렬이며, $\mathbf{p}_i$와 $\mathbf{p}'_i$는 각각 최적화 전후의 꼭짓점 위치이다.

#### (E) 학습 손실 함수

총 손실 함수는 다음과 같이 구성된다:

$$\mathcal{L} = \mathcal{L}_{\text{photo}} + \lambda_1 \mathcal{L}_{\text{mesh}} + \lambda_2 \mathcal{L}_{\text{reg}}$$

- **$\mathcal{L}_{\text{photo}}$**: Photometric loss (L1 + D-SSIM)

$$\mathcal{L}_{\text{photo}} = (1-\lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}}$$

- **$\mathcal{L}_{\text{mesh}}$**: 메시-가우시안 정합 손실 (Gaussian이 메시 표면 가까이 유지되도록)
- **$\mathcal{L}_{\text{reg}}$**: 정규화 항 (Gaussian 분포의 비정상적 형태 억제)

### 2.3 모델 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    전체 파이프라인 개요                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [Multi-view Images] → [SDF-based Mesh Reconstruction]       │
│           │                        │                         │
│           ▼                        ▼                         │
│  [3D Gaussian Initialization]  [Proxy Mesh M]                │
│           │                        │                         │
│           └──────────┬─────────────┘                         │
│                      ▼                                       │
│         [Mesh-based Gaussian Learning]                       │
│         ├── Barycentric Binding                              │
│         ├── Adaptive Mesh-Gaussian Split                     │
│         ├── Mesh Constraint Regularization                   │
│         └── Photometric Optimization                         │
│                      │                                       │
│                      ▼                                       │
│         [Trained Mesh-Gaussian Representation]                │
│                      │                                       │
│           ┌──────────┴──────────┐                            │
│           ▼                     ▼                            │
│    [Novel View Synthesis]  [Mesh Deformation]                │
│    (65 FPS real-time)     ├── ARAP / ACAP                    │
│                           ├── Data-driven                    │
│                           └── Parameter Update               │
│                                  │                           │
│                                  ▼                           │
│                    [Deformed GS Rendering]                    │
│                    (Real-time, artifact-free)                 │
└─────────────────────────────────────────────────────────────┘
```

이 방법은 surface-aware 재구성을 도입하여, 먼저 SDF(Sign Distance Field) 기반 방법으로 메시를 구성하고, 이 메시를 사용하여 Gaussian Splatting 재구성 과정을 제약하며, 로컬 강성(local rigidity)과 글로벌 비강성(global non-rigidity) 제한을 통합하여 Gaussian 변형을 안내한다.

### 2.4 성능 향상

광범위한 실험 결과, 이 접근법은 고품질 재구성과 효과적인 변형을 달성하면서, 단일 상용 GPU에서 높은 프레임 레이트(평균 65 FPS)로 유망한 렌더링 결과를 유지한다.

| 성능 지표 | 결과 |
|----------|------|
| **렌더링 FPS** | 평균 65 FPS (단일 GPU) |
| **렌더링 품질** | 기존 3D Gaussian Splatting보다 높은 품질의 novel view synthesis 달성, 대규모 변형에서도 유지 |
| **변형 품질** | 실시간으로 대규모 변형에서 고품질 변형 결과 생성 |
| **정규화 효과** | 명시적 메시 제약이 Gaussian 분포를 정규화하여 정렬 불량/길고 좁은 형태의 Gaussian 등을 억제, 시각적 품질 향상 및 변형 시 아티팩트 감소 |

### 2.5 한계점

1. **메시 품질 의존성**: ARAP을 이용한 고해상도 메시 변형이 어려우며, 메시를 10-20K 면으로 단순화할 필요가 있다. 메시에 구멍이나 자기교차(self-intersection) 같은 아티팩트가 있으면 ACAP이 작동하지 않는다.

2. **초기 메시 재구성 필요**: SDF 기반 방법 등으로 프록시 메시를 사전에 재구성해야 하므로, 전체 파이프라인이 복잡해진다.

3. **물리 기반 시뮬레이션 부족**: 현재 방법은 기하학적 변형에 초점을 맞추고 있으며, 재질 속성이나 물리적 시뮬레이션은 다루지 않는다.

4. **일반화 제약**: 메시가 대상 객체의 거의 모든 기하학적 요소를 커버할 수 있다고 가정하며, 이는 복잡한 unbounded 장면에서는 제한적이다.

5. **토폴로지 변화 처리 불가**: 메시 토폴로지가 고정된 상태에서의 변형만 지원하여, 절단·병합 등의 토폴로지 변화는 처리할 수 없다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화의 한계

명시적 메시 표현이 GS 모델의 다양한 객체 카테고리 및 복잡성 전반에 걸친 확장성과 일반화에 어떤 영향을 미치는지가 핵심 질문이다.

현재 모델의 일반화 한계는 다음과 같다:

- **장면 특화(scene-specific) 학습**: 각 장면마다 독립적으로 메시-가우시안 표현을 학습해야 한다.
- **메시 품질 의존**: 초기 메시의 품질이 최종 결과에 크게 영향을 미친다.
- **도메인 제한**: 3D GS가 per-Gaussian 변형을 모델링하여 동적 장면 재구성을 발전시켰으나, 세밀한(fine-grained) 프리미티브에 대한 의존은 확장성과 강건성을 제한한다.

### 3.2 일반화 향상 방향

#### (a) 카테고리 수준 일반화

기존 메시 변형 데이터셋을 활용한 데이터 기반 Gaussian 변형이 가능하다는 점은 이미 일반화의 잠재력을 보여준다. 이를 확장하면:

- **학습된 변형 필드**: 특정 객체 카테고리(예: 인체, 동물)에 대한 변형 사전(prior)을 학습하여 새로운 인스턴스에 전이 가능
- **템플릿 기반 접근**: GART는 canonical space에서 Gaussian mixture model을 사용하여 관절 주체를 모델링하며, 카테고리 특화 템플릿(예: SMPL/SMAL)과 학습 가능한 forward skinning을 활용한다.

#### (b) 크로스-도메인 일반화

향후 연구는 Gaussian을 영구적인 엔티티로 계층적으로 그룹화하는 객체 중심 프레임워크를 우선시하여, 내재적 모션 분리(동적 vs 정적)를 통해 효율적인 대규모 재구성을 가능하게 할 수 있다.

#### (c) 스파스 뷰 일반화

스파스 멀티뷰 이미지에서의 재구성은 제한된 입력에도 불구하고 뷰 일관성과 강한 일반화를 보장하는 기술이 필요하다.

#### (d) Mesh-Gaussian 하이브리드의 강화

다양한 장면 복잡도에 적응하기 위해:

$$\text{Generalization Score} = f(\text{mesh quality}, \text{Gaussian density}, \text{deformation prior})$$

이 세 요소의 균형을 통해 일반화 성능을 최적화할 수 있다.

### 3.3 구체적 일반화 향상 전략

1. **메시 재구성의 강건화**: 다양한 SDF 방법론 (NeuS, instant-nsr-pl 등)을 앙상블하여 초기 메시 품질 향상
2. **적응적 메시-가우시안 밀도 제어**: 장면 복잡도에 따라 자동으로 면 수와 가우시안 밀도를 조절
3. **전이 학습(Transfer Learning)**: 유사 카테고리 간 변형 패턴 공유를 통한 few-shot 변형 학습
4. **하이브리드 변형 모델**: 기하학적 ARAP/ACAP 변형과 신경망 기반 변형 필드의 결합

---

## 4. 연구 영향 및 향후 연구 시 고려할 점

### 4.1 연구에 미치는 영향

**① 새로운 표현 패러다임의 확립**

3D Gaussian Splatting의 명시적 표현은 동적 재구성, 기하학 편집, 물리 시뮬레이션 같은 하류 작업을 촉진한다. 본 논문의 mesh-Gaussian 하이브리드 표현은 이 방향의 핵심적 기반을 제공한다.

**② 인터랙티브 콘텐츠 생성 가속화**

실시간 65 FPS의 변형 가능한 렌더링은 VR/AR, 게임, 영화 제작 등에서 인터랙티브 콘텐츠 생성의 실용성을 크게 높였다.

**③ 메시-가우시안 결합 연구의 촉진**

Mani-GS는 삼각 메시를 추출하고, 3D Gaussians를 삼각형 프리미티브에 바인딩하여, 메시 연산을 통해 변형 및 소프트 바디 시뮬레이션을 가능하게 한다. 이처럼 mesh-Gaussian 결합 연구가 활발히 진행되고 있다.

**④ 디지털 휴먼 분야 발전**

SplattingAvatar는 메시 표면 위에서 Gaussian 파라미터와 메시 임베딩을 공동 최적화하며, GoMAvatar는 Gaussians-on-Mesh(GoM) 표현을 채택하여 splatting의 렌더링 속도와 메시 변형의 호환성을 결합한다. 이러한 후속 연구들이 본 논문의 아이디어를 확장하고 있다.

### 4.2 향후 연구 시 고려할 점

| 연구 방향 | 세부 내용 |
|----------|----------|
| **물리 기반 시뮬레이션 통합** | mesh 기반 GS 방법이 재질 속성이나 반투명성 같은 추가 특성을 지원하도록 적응 가능한지 연구 필요 |
| **토폴로지 변화 처리** | 절단, 병합 등의 토폴로지 변경을 지원하는 동적 메시-가우시안 표현 개발 |
| **대규모 장면 확장** | unbounded 장면에서 메시가 빈번하게 아티팩트를 보이므로, 3D Gaussian splats을 메시 표면에 효과적으로 바인딩하기 위한 더 나은 전략이 필요하다. |
| **효율성 최적화** | 3D Gaussian을 대안적 구조로 표현하면 기하학적 충실도가 향상되지만, 계산 효율성과 렌더링 성능이 저하될 수 있다. |
| **동적 장면 일관성** | 객체 수준의 모션 추론 부재는 아티팩트와 장기 시퀀스에 대한 불량한 일반화를 초래한다. |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 접근법 | 본 논문과의 차이 |
|------|------|------------|----------------|
| **3DGS** (Kerbl et al.) | 2023 | 3D Gaussian Splatting은 미분 가능한 래스터화로 최적화된 비등방 Gaussian으로 장면을 표현하는 돌파구적 기술이다. | 변형 기능 없음, 토폴로지 부재 |
| **D-NeRF** (Pumarola et al.) | 2021 | 동적 장면을 위한 Neural Radiance Fields | 암시적 표현으로 실시간 렌더링 어려움 |
| **Deformable 3DGS** (Yang et al.) | 2023 | 변형 가능한 3D Gaussians가 canonical space에서 학습되고, 공간-시간 동역학을 모델링하는 변형 필드(spatial MLP)와 결합된다. | MLP 기반 변형 필드 사용, 메시 제약 없음 |
| **4D Gaussian Splatting** (Wu et al.) | 2023 | 실시간 동적 장면 렌더링을 위한 4D Gaussian Splatting | 시간 차원 모델링에 초점, 사용자 인터랙션 미지원 |
| **SuGaR** (Guédon & Lepetit) | CVPR 2024 | 3D Gaussians가 표면에 잘 정렬되도록 정규화하고, Poisson 재구성으로 메시를 추출 | 메시 추출 후 편집 가능하나 대규모 변형에 최적화되지 않음 |
| **GaMeS** (Waczyńska et al.) | 2024 | Gaussian Mesh Splatting 모델을 도입하여 Gaussian 구성요소를 메시와 유사한 방식으로 수정할 수 있게 하며, 각 Gaussian을 메시 면의 꼭짓점으로 매개변수화한다. | 유사한 철학이나 적응적 분할 메커니즘 없음 |
| **E-D3DGS** (Bae et al.) | ECCV 2024 | per-Gaussian latent embedding을 사용하여 각 Gaussian의 변형을 예측하고 동적 모션의 더 명확한 표현을 달성한다. | 동적 장면 재구성에 초점, 인터랙티브 편집 미지원 |
| **MeshGS** (Choi et al.) | ACCV 2024 | unbounded 대규모 장면에서 메시와 3D Gaussian splats의 렌더링 및 기하학적 강점을 통합하는 새로운 접근법 | 렌더링 품질에 초점, 변형 기능 미지원 |
| **SC-GS** (Huang et al.) | 2024 | 희소 제어점 조정으로 모션 편집을 가능하게 하여, 고충실도 렌더링 품질을 보존하면서 강체 포즈 변형을 허용한다. | 제어점 기반 vs. 메시 기반 접근 |
| **Grid4D** | 2024 | Gaussian splatting 기반으로 4D 입력에 대한 새로운 명시적 인코딩 방법을 사용하며, low-rank 가정 없이 하나의 공간적 및 세 개의 시간적 3D 해시 인코딩으로 분해한다. | 해시 인코딩 기반 동적 장면, 메시 제약 없음 |

### 비교 차원 요약

```
                        렌더링 품질
                            ▲
                            │
        SuGaR ●             │         ● 본 논문 (Mesh-GS)
                            │
   MeshGS ●                 │
                            │      ● GaMeS
                            │
   3DGS ●                   │
                            │
    ──────────────────────────────────────► 변형 가능성
                            │
   D-NeRF ●                 │
                            │    ● Deformable 3DGS
                            │
            4DGS ●          │         ● SC-GS
                            │
                            │    ● E-D3DGS
```

---

## 참고 자료 및 출처

1. **Lin Gao et al.**, "Mesh-based Gaussian Splatting for Real-time Large-scale Deformation," arXiv:2402.04796 (2024). — https://arxiv.org/abs/2402.04796
2. **프로젝트 페이지**: http://geometrylearning.com/GaussianMesh/
3. **GitHub 구현 (IGLICT/GaussianMesh)**: https://github.com/IGLICT/GaussianMesh
4. **ACM TOG 게재본**: https://dl.acm.org/doi/10.1145/3687756
5. **SuGaR** (Guédon & Lepetit, CVPR 2024): https://github.com/Anttwo/SuGaR
6. **GaMeS** (Waczyńska et al., 2024): https://github.com/waczjoan/gaussian-mesh-splatting
7. **E-D3DGS** (Bae et al., ECCV 2024): https://github.com/JeongminB/E-D3DGS
8. **MeshGS** (Choi et al., ACCV 2024): https://openaccess.thecvf.com/content/ACCV2024/
9. **3DGS Survey** (Chen & Wang, 2024): https://arxiv.org/abs/2401.03890
10. **Recent advances in 3DGS** (Springer, 2024): https://link.springer.com/article/10.1007/s41095-024-0436-y
11. **3DGS: Survey, Technologies, Challenges** (IEEE TCSVT, 2025): https://www.tianyuding.com/papers/3DGS-survey.pdf
12. **Human reconstruction using 3DGS survey** (Frontiers in AI, 2025): https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1709229
13. **Surface reconstruction based on 3DGS survey** (PMC, 2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC12453780/
14. **Open3D ARAP documentation**: https://www.open3d.org/docs/latest/tutorial/geometry/mesh_deformation.html
15. **Semantic Scholar**: https://www.semanticscholar.org/paper/ea730b57613c0ab64d2c47fd0826d53620db2c61
16. **NASA/ADS**: https://ui.adsabs.harvard.edu/abs/2024arXiv240204796G/abstract
17. **EmergentMind**: https://www.emergentmind.com/papers/2402.04796
18. **Bytez**: https://bytez.com/docs/arxiv/2402.04796/paper

---

> **참고**: 본 분석은 논문의 공개된 초록, 프로젝트 페이지, 공식 코드 리포지토리, 그리고 관련 서베이 논문들을 기반으로 작성되었습니다. 논문 본문의 세부 수식(예: 정확한 정규화 항의 가중치, ablation study의 정량적 수치 등)은 전체 PDF에 기반한 것이 아니므로, 일부 수식은 해당 방법론의 원리에 기반한 재구성임을 밝힙니다. 정확한 수식과 정량적 결과는 원 논문(arXiv:2402.04796)의 전문을 참조해 주시기 바랍니다.
