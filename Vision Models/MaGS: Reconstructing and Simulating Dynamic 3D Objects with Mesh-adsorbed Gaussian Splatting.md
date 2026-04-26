
# MaGS: Reconstructing and Simulating Dynamic 3D Objects with Mesh-adsorbed Gaussian Splatting

> **논문 정보**
> - **제목**: MaGS: Reconstructing and Simulating Dynamic 3D Objects with Mesh-adsorbed Gaussian Splatting
> - **저자**: Shaojie Ma, Yawei Luo, Yi Yang 외
> - **arXiv**: [2406.01593](https://arxiv.org/abs/2406.01593) (2024년 6월 제출, 2024년 11월 업데이트)
> - **학술대회**: ICCV 2025 게재 확정

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 문제 인식

3D 재구성(Reconstruction)과 시뮬레이션(Simulation)은 상호 연관되어 있지만 목적이 서로 다릅니다. 재구성은 다양한 장면에 적응할 수 있는 유연한 3D 표현이 필요하고, 시뮬레이션은 운동 원리를 효과적으로 모델링하기 위한 구조화된 표현이 필요합니다. 이러한 이중 요구사항이 단일 통합 프레임워크 구성의 근본적인 도전 과제였습니다.

### 1.2 핵심 주장

이 논문은 Mesh-adsorbed Gaussian Splatting (MaGS) 방법을 제안합니다. MaGS는 3D Gaussian을 메쉬 근처에서 자유롭게 이동할 수 있도록 제약하여 상호 흡착된(mutually adsorbed) 메쉬-가우시안 3D 표현을 만들며, 3D Gaussian의 렌더링 유연성과 메쉬의 구조화된 특성을 동시에 활용합니다.

### 1.3 주요 기여 (3가지)

MaGS는 기존 방법 대비 세 가지 핵심 장점을 제공합니다:
1. **시뮬레이션에 직접 활용 가능한 연속 메쉬**를 내장 (SC-GS, D-Miso와 차별화)
2. **Gaussian이 메쉬 표면을 따라 이동 가능** (DG-Mesh, SplattingAvatar는 메쉬-가우시안 변위를 지원하지 않음)
3. **프레임 간 메쉬 연속성 보존** (DynaSurfGS 및 Dynamic 2DGS는 TSDF 방식으로 프레임별 독립 메쉬를 생성하여 교차 프레임 일관성 없음)

---

## 2. 문제 정의, 제안 방법 및 모델 구조

### 2.1 해결하고자 하는 문제

인간의 시각 시스템은 단안 비디오(monocular video)로부터 3D 외관(재구성)을 동시에 포착하고 동적 객체의 운동 가능성(시뮬레이션)을 추론할 수 있습니다. 반면 컴퓨터 비전 및 그래픽스 분야에서는 3D 재구성과 시뮬레이션을 별개의 작업으로 다루어 왔습니다.

기존 접근법의 문제:
- **순수 3D Gaussian Splatting**: 렌더링은 뛰어나지만 구조적 정보 부재로 물리 시뮬레이션 어려움
- **순수 메쉬 기반**: 물리 시뮬레이션에 적합하지만 렌더링 품질 한계
- **메쉬-가우시안 앵커 방식(DG-Mesh, SplattingAvatar)**: Gaussian이 메쉬에 고정되어 유연성 저하

### 2.2 Mesh-adsorbed Gaussian 표현

각 Gaussian $i$는 다음 속성으로 정의됩니다:

$$G_i = \{\mathbf{b}_i, h_i, \text{MeshId}_i, \boldsymbol{\mu}_i, \mathbf{r}_i, \mathbf{s}_i, \sigma_i, \mathbf{c}_i\}$$

여기서 $\mathbf{b} = (b_1, b_2, b_3)$는 메쉬 면(facet) 위의 무게중심 좌표(barycentric coordinates)로, $b_1 + b_2 + b_3 = 1$이고 $b_k \geq 0$ $(k=1,2,3)$을 만족하며, $h \in [0,1]$은 면의 법선 방향을 따른 오프셋을 나타냅니다. MeshId 속성은 해당 Gaussian이 위치하는 면(facet)을 기록합니다.

Gaussian의 전역 위치 $\boldsymbol{\mu}'_i$는 다음과 같이 계산됩니다:

$$\boldsymbol{\mu}'_i = b_1 \mathbf{v}_1 + b_2 \mathbf{v}_2 + b_3 \mathbf{v}_3 + h \cdot \mathbf{n}_f$$

여기서 $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$는 해당 facet의 세 꼭짓점, $\mathbf{n}_f$는 facet의 법선 벡터입니다.

### 2.3 파이프라인 개요 (2단계)

MaGS는 각 비디오 프레임에서 시간적으로 일관성 있는 거친 메쉬(coarse mesh)를 추출하는 것으로 시작합니다. 이 메쉬들은 Guide Mesh라 불리며 동적 재구성의 기반을 제공합니다. 재구성 과정에서 Guide Mesh의 포즈 정보가 MPE-Net에 의해 추출되어 RMD-Net과 RGD-Net으로 전달됩니다. RMD-Net과 RGD-Net은 각각 Guide Mesh와 Mesh-adsorbed Gaussian에 상대적 변형을 수행하여 정제된 메쉬(refined mesh)와 상대 변형된 Gaussian(relative deformed Gaussian)을 산출합니다. 이 두 요소가 결합하여 최종 변형된 Gaussian(Final Deformed Gaussians)을 생성합니다.

#### 단계 I: 재구성(Reconstruction)

Splatting 기반 렌더링이 적용되며, 렌더링 손실을 이용해 Gaussian, MPE-Net, RMD-Net, RGD-Net을 역전파(backpropagation)로 최적화합니다. 재구성 단계는 고정밀 메쉬와 Gaussian을 산출할 뿐만 아니라 네트워크가 비디오로부터 변형 원리를 학습하게 하여 시뮬레이션을 위한 준비도 효과적으로 수행합니다.

#### 단계 II: 시뮬레이션(Simulation)

시뮬레이션 단계에서는 소프트바디 시뮬레이션, ARAP, SMPL 등의 메쉬 기반 기법이 재구성된 메쉬를 변형하여 새로운 Guide Mesh를 생성합니다. Mesh-adsorbed Gaussian도 상속(해당 facet에 흡착된 채로)됩니다. 이후 과정은 재구성과 유사하게 MPE-Net, RMD-Net, RGD-Net이 다시 활용되어 최종 변형된 Gaussian을 산출하고, 이를 렌더링하여 최종 이미지를 생성합니다.

### 2.4 세 가지 핵심 네트워크

| 네트워크 | 역할 |
|---------|------|
| **MPE-Net** | 메쉬에서 포즈 임베딩(Pose Embedding) 추출 |
| **RMD-Net** | 메쉬의 상대적 변형 예측 (Guide Mesh 정제) |
| **RGD-Net** | Gaussian과 메쉬 간 상대 변위 모델링 (렌더링 품질 향상) |

RMD-Net은 비디오 데이터로부터 운동 사전(motion priors)을 학습하여 메쉬 변형을 정제하고, RGD-Net은 메쉬 제약 하에서 렌더링 충실도를 향상시키기 위해 메쉬와 Gaussian 간의 상대 변위를 모델링합니다.

### 2.5 손실 함수

최종 변형된 Gaussian을 렌더링한 후 실제값과 비교합니다. 손실은 역전파를 가능하게 하는 다음 수식으로 계산되며, $\mathcal{L}_\text{SSIM}$은 구조적 유사도 손실을 나타냅니다. 전체 과정이 미분 가능하기 때문에 MPE-Net, RMD-Net, RGD-Net 및 Mesh-adsorbed Gaussian을 렌더링 오류를 기반으로 역전파를 통해 공동 최적화할 수 있습니다.

총 손실 함수는 다음과 같이 구성됩니다:

$$\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{\text{SSIM}}$$

여기서 $\mathcal{L}\_1$은 픽셀 단위 L1 손실, $\mathcal{L}_{\text{SSIM}}$은 구조적 유사도 손실이며, $\lambda$는 가중치 하이퍼파라미터입니다.

전체 과정이 미분 가능하기 때문에 MPE-Net, RMD-Net, RGD-Net 및 Mesh-adsorbed Gaussian을 렌더링 오류 기반의 역전파를 통해 공동으로 최적화할 수 있습니다.

### 2.6 ARAP 기반 시뮬레이션 수식

MaGS는 사용자 상호작용 시뮬레이션(예: 드래깅)을 메쉬를 직접 수정함으로써 가능하게 하며, 이는 흡착된 3D Gaussian을 빠른 렌더링을 위해 업데이트합니다. 초기에는 ARAP 알고리즘이 사용자 정의 운동에 기반하여 메쉬를 변형하며, 로컬 강성을 보존하면서 왜곡을 최소화합니다.

ARAP 에너지 최소화 수식:

$$E_{\text{ARAP}} = \sum_{i}\sum_{j \in \mathcal{N}(i)} w_{ij} \left\| (\mathbf{p}'_i - \mathbf{p}'_j) - R_i(\mathbf{p}_i - \mathbf{p}_j) \right\|^2$$

여기서 $\mathbf{p}\_i, \mathbf{p}'\_i$는 각각 변형 전후 정점 위치, $R_i$는 정점 $i$의 국소 회전 행렬, $w_{ij}$는 에지 가중치입니다.

---

## 3. 일반화 성능 향상 가능성

### 3.1 MPE-Net의 역할: 시간 비의존적 일반화

시뮬레이션에서 입력 비디오를 넘어서는 새로운 사용자 정의 변형에 일반화하기 위해, 시간적 데이터에 의존하지 않고 메쉬의 고유 정보를 활용하여 RMD-Net과 RGD-Net을 부트스트랩하는 MPE-Net을 제안합니다.

> 이것이 MaGS 일반화 전략의 핵심입니다. 기존 방법들은 특정 타임스탬프(시간 정보)를 입력받아 변형을 예측하지만, MPE-Net은 **메쉬의 형태 자체**에서 포즈 임베딩을 추출하므로 학습 데이터에 없던 새로운 포즈/변형에도 대응 가능합니다.

### 3.2 다양한 물리 프레임워크와의 호환성

메쉬의 보편성으로 인해 MaGS는 ARAP, SMPL, 소프트바디 물리 시뮬레이션과 같은 다양한 변형 사전(deformation priors)과 호환됩니다.

이는 다음과 같은 일반화 시나리오를 가능하게 합니다:

| 시뮬레이션 유형 | 적용 예시 |
|--------------|---------|
| **ARAP** | 강체성 보존 탄성 변형, 사용자 드래깅 편집 |
| **SMPL** | 인체 포즈 전이(novel pose synthesis) |
| **소프트바디** | 물리 기반 천, 젤리, 유체 근사 변형 |

### 3.3 재구성에서 시뮬레이션으로의 지식 전이

재구성 단계는 고정밀 메쉬와 Gaussian을 산출할 뿐만 아니라, 네트워크가 비디오로부터 변형 원리를 학습하게 하여 시뮬레이션을 위해 효과적으로 준비시킵니다.

즉, MPE-Net, RMD-Net, RGD-Net이 재구성 단계에서 학습한 **변형 패턴(deformation principles)**은 시뮬레이션 단계에서도 재사용되어 out-of-distribution 변형에 대한 일반화를 지원합니다.

### 3.4 메쉬 연속성을 통한 시간적 일반화

MaGS는 프레임 간 메쉬 연속성을 보존하여 시간에 걸쳐 일관된 포인트-면(point-and-facet) 대응을 보장합니다. 이 연속성은 재구성에서 시뮬레이션으로 Mesh-adsorbed Gaussian을 상속하는 데 매우 중요하며, TSDF를 사용하여 프레임당 독립적인 메쉬를 생성하는 DynaSurfGS 및 Dynamic 2DGS와 비교하여 눈에 띄는 개선을 나타냅니다.

---

## 4. 성능 향상 및 한계

### 4.1 성능 향상

D-NeRF, DG-Mesh, PeopleSnapshot 데이터셋에 대한 광범위한 실험에서 MaGS는 재구성과 시뮬레이션 모두에서 최신 성능(state-of-the-art)을 달성했습니다.

- D-NeRF 데이터셋에서 동적 장면의 Novel View Synthesis에 대한 평균 PSNR/MS-SSIM/VGG-LPIPS 값이 보고되었습니다.
- PeopleSnapshot 데이터셋에서 Novel Pose Synthesis에 대한 평균 PSNR/SSIM/VGG-LPIPS 값이 비교되었습니다.

**Ablation Study 결과:**

MaGS의 두 가지 핵심 설계 결정(Mesh-adsorbed 3D Gaussian 표현, 비강체 변형 필드)에 대한 ablation을 수행했습니다. Mesh-adsorbed 3D Gaussian 표현에 대해 ablation 시 "hovering" 속성을 비활성화하여 mesh-adsorbed 패턴을 mesh-anchored 패턴으로 변환하면 렌더링 성능이 소폭 하락합니다.

상대 변형 필드를 ablation하여 순수 ARAP 사전만으로 시뮬레이션할 경우 렌더링 성능이 크게 하락합니다.

### 4.2 한계점 (공개된 내용 기반 추론)

| 한계 | 설명 |
|-----|------|
| **단안 비디오 의존성** | 단안 비디오(monocular video)로부터 동적 3D 객체를 재구성·시뮬레이션하는 구조로, 다시점 카메라 환경에서의 확장성은 별도 검증 필요 |
| **거친 메쉬 품질 의존성** | Guide Mesh의 초기 품질이 전체 파이프라인에 영향. 매우 복잡한 위상(topology)의 객체는 어려울 수 있음 |
| **물리 소재 다양성** | 현재 ARAP, SMPL, 소프트바디 시뮬레이션을 지원하지만, 유체·기체 등 비접촉 현상에는 적용 한계 |
| **실시간 처리** | 세 개의 신경망(MPE-Net, RMD-Net, RGD-Net)을 동시에 구동하는 계산 비용 |

---

## 5. 관련 최신 연구 비교 분석 (2020년 이후)

### 5.1 비교 대상 연구 정리

| 논문 | 방법 | 장점 | 단점 vs MaGS |
|-----|------|-----|-------------|
| **NeRF** (2020, Mildenhall et al.) | 암묵적 Neural Radiance Field | 고품질 뷰 합성 | 느린 렌더링, 시뮬레이션 불가 |
| **3D Gaussian Splatting** (2023, Kerbl et al.) | 명시적 3D Gaussian | 실시간 렌더링 | 구조 없음, 시뮬레이션 어려움 |
| **DG-Mesh** (2024) | 동적 메쉬 + Gaussian | 메쉬 재구성 | Gaussian이 메쉬에 고정, 유연성 부족 |
| **SC-GS** (2024) | 희소 제어점 기반 Gaussian | 편집 가능 | 연속 메쉬 없음, 시뮬레이션 제한 |
| **SplattingAvatar** (2024) | 인간 아바타용 GS + 메쉬 | 인체 특화 | 메쉬-Gaussian 변위 미지원 |
| **PhysGaussian** (2024) | 물리 기반 Gaussian | 다양한 재료 | 메쉬 구조 없이 Newtonian dynamics만 |
| **GaMeS** (2024) | 메쉬 기반 GS 편집 | 실시간 편집 | 동적 재구성과 시뮬레이션 통합 부재 |

GaMeS(Gaussian Mesh Splatting) 모델은 메쉬와 유사한 방식으로 Gaussian 구성 요소를 수정할 수 있게 하여 편집 가능한 GS의 실시간 렌더링을 가능하게 했습니다.

PhysGaussian은 3D Gaussian 내에 물리적으로 근거한 뉴턴 역학을 통합하여 고품질의 새로운 운동 합성을 달성하며 다양한 재료에서 뛰어난 다용성을 보여줍니다.

### 5.2 MaGS의 차별점 요약

```
기존 방법의 스펙트럼:
[렌더링 특화]──────────────[시뮬레이션 특화]
3DGS   SC-GS   GaMeS   DG-Mesh   물리엔진
                    ↑
              MaGS (통합)
```

MaGS는 렌더링 품질과 물리 시뮬레이션 호환성을 동시에 달성하는 **통합 프레임워크**를 최초로 제시했다는 점에서 기존 연구들과 근본적으로 차별화됩니다.

---

## 6. 향후 연구에 미치는 영향 및 고려 사항

### 6.1 연구에 미치는 영향

1. **재구성-시뮬레이션 통합 패러다임 제시**
   인간의 시각 시스템은 단안 비디오에서 3D 외관 포착과 운동 추론을 동시에 수행하지만, 컴퓨터 비전과 그래픽스는 이를 별개 작업으로 다루어 왔습니다. MaGS는 이 간극을 메운 첫 사례로서, 향후 "재구성=시뮬레이션 준비"라는 새로운 연구 방향을 열었습니다.

2. **메쉬-가우시안 하이브리드 표현의 기반 확립**
   학습 가능한 Relative Deformation Field(RDF)를 도입하여 메쉬와 3D Gaussian 간의 상대 변위를 모델링하고, ARAP 사전에만 의존하는 전통적 메쉬 구동 변형 패러다임을 확장하여 각 3D Gaussian의 운동을 더 정밀하게 포착합니다.

3. **디지털 트윈, XR, 로보틱스 응용 가능성**
   MaGS는 메쉬를 직접 수정함으로써 사용자 상호작용 시뮬레이션(예: 드래깅)을 가능하게 하며, 이는 흡착된 3D Gaussian을 빠른 렌더링을 위해 업데이트합니다. 이 특성은 VR/AR, 디지털 콘텐츠 제작, 로봇 매니퓰레이션 시뮬레이션에 직접 적용될 수 있습니다.

### 6.2 향후 연구 시 고려할 점

| 연구 방향 | 구체적 고려사항 |
|---------|-------------|
| **복잡한 위상 처리** | 옷, 머리카락, 얇은 구조물 등 복잡한 위상을 가진 객체에 대한 메쉬 추출 고도화 필요 |
| **다중 객체 상호작용** | 현재는 단일 동적 객체 중심 → 여러 객체 간의 충돌, 접촉 시뮬레이션으로 확장 |
| **재료 물성 추정** | 시뮬레이션의 물리 파라미터(탄성 계수, 마찰 계수 등)를 영상에서 자동 추정하는 역문제 연구 |
| **계산 효율화** | MPE-Net + RMD-Net + RGD-Net의 동시 추론 비용 감소 (경량화, 지식 증류) |
| **대규모 장면 확장** | 현재 단일 객체 중심에서 실외 장면 전체로의 확장 가능성 |
| **비감독 학습** | 레이블 없는 비디오로부터의 자기 지도 학습(self-supervised) 방식 도입 |
| **일반화 벤치마크** | Novel deformation 일반화 성능 평가를 위한 표준화된 벤치마크 데이터셋 구축 필요 |

---

## 참고자료 및 출처

1. **MaGS 공식 arXiv 논문**: Shaojie Ma et al., "MaGS: Reconstructing and Simulating Dynamic 3D Objects with Mesh-adsorbed Gaussian Splatting," arXiv:2406.01593, 2024. https://arxiv.org/abs/2406.01593

2. **MaGS 프로젝트 페이지 (공식)**: https://wcwac.github.io/MaGS-page/

3. **MaGS ICCV 2025 공개 논문 PDF**: https://openaccess.thecvf.com/content/ICCV2025/papers/Ma_MaGS_Reconstructing_and_Simulating_Dynamic_3D_Objects_with_Mesh-adsorbed_Gaussian_ICCV_2025_paper.pdf

4. **MaGS HTML 전체 버전 (v2)**: https://arxiv.org/html/2406.01593v2

5. **Semantic Scholar MaGS 인용 정보**: https://www.semanticscholar.org/paper/Reconstructing-and-Simulating-Dynamic-3D-Objects-Ma-Luo/0435c4a7862b1ad170a81e64481dbff4e8523b83

6. **DynaSurfGS (비교 논문)**: "DynaSurfGS: Dynamic Surface Reconstruction with Planar-based Gaussian Splatting," arXiv:2408.13972, 2024. https://arxiv.org/html/2408.13972v1

7. **비교 참고 논문들** (MaGS 논문 내 인용):
   - 3D Gaussian Splatting: Kerbl et al., ACM TOG 2023
   - D-NeRF: Pumarola et al., CVPR 2021
   - DG-Mesh: Liu et al., arXiv:2404.12379, 2024
   - PhysGaussian: arXiv 2024
   - GaMeS: Waczyńska et al., 2024
   - SC-GS: Huang et al., CVPR 2024

> **⚠️ 주의사항**: 본 논문의 구체적인 수식(특히 최종 변형된 Gaussian의 파라미터 업데이트 수식)의 일부 세부 사항은 ICCV 2025 공개 PDF와 arXiv HTML 버전에서 수식 렌더링이 불완전하게 공개되어 있어, 이 답변에서는 검증된 개념과 구조를 기반으로 설명하였습니다. 완전한 수식은 공식 논문 PDF를 직접 참조하시기 바랍니다.
