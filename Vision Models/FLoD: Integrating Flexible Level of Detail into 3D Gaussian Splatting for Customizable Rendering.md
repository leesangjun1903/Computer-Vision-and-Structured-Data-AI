
# FLoD: Integrating Flexible Level of Detail into 3D Gaussian Splatting for Customizable Rendering

---

## 1. 핵심 주장 및 주요 기여 요약

### 📌 핵심 주장

3D Gaussian Splatting(3DGS)은 고품질 3D 재구성과 빠른 렌더링 속도로 컴퓨터 그래픽스 분야를 크게 발전시켰으나, 3DGS 및 후속 연구들은 특정 하드웨어 설정에만 제한되어 있어 저사양 또는 고사양 구성 중 하나에만 적합하다.

메모리 사용을 줄이는 접근법은 저사양 GPU에서 렌더링을 가능하게 하지만 품질을 희생하며, 반대로 렌더링 품질을 향상시키는 방법은 대용량 VRAM을 가진 고사양 GPU를 필요로 한다. 결과적으로 3DGS 기반 연구들은 단일 하드웨어 설정을 가정하며 다양한 하드웨어 제약에 유연하게 적응하지 못한다.

이에 FLoD는 **"하나의 훈련된 모델로 다양한 하드웨어에서 품질과 메모리의 균형을 조절"** 한다는 핵심 주장을 내세웁니다.

### 📌 주요 기여

| 기여 항목 | 내용 |
|---|---|
| ① 다중 레벨 3DGS 표현 | 레벨별 독립적 장면 재구성 |
| ② 3D 스케일 제약 (Scale Constraint) | 레벨별 Gaussian 크기 제어 |
| ③ 레벨별 순차 학습 | 구조적 일관성 보장 |
| ④ 중복 가지치기 (Overlap Pruning) | Gaussian 중복 완화 |
| ⑤ Selective Rendering | 이미지 영역별 다른 레벨 적용 |
| ⑥ 타 프레임워크 일반화 | Scaffold-GS 등에 통합 가능 |

기존 3DGS에 LoD 개념을 통합한 선행 연구들 중, FLoD는 광범위한 GPU 설정에 맞는 조정 가능한 옵션을 제공함으로써 LoD의 핵심 원칙을 최초로 따른다.

---

## 2. 상세 분석: 문제 → 방법 → 구조 → 성능 → 한계

---

### 2-1. 해결하고자 하는 문제

3DGS는 수많은 작은 Gaussian을 사용하여 빠르고 고품질의 렌더링을 달성하지만, 이는 상당한 메모리 소비를 초래한다. 이러한 대량의 Gaussian에 대한 의존성은 메모리 제한으로 인해 저사양 장치에서의 3DGS 기반 모델 적용을 제한한다. 그러나 단순히 Gaussian 수를 줄이는 것은 고사양 하드웨어에서 달성 가능한 품질보다 낮은 품질로 이어진다.

즉, 기존 방법들은:
- **압축 방법**: 메모리는 줄이지만 품질 저하
- **품질 향상 방법**: 고성능 GPU 필수, 저사양에서는 불가
- **단일 하드웨어 가정**: 하드웨어 다양성에 대한 적응성 결여

---

### 2-2. 제안 방법 및 수식

#### (A) 3DGS의 기본 렌더링 수식

3DGS는 각 픽셀의 색상 $C$를 Gaussian 프리미티브들의 alpha-compositing으로 계산합니다:

$$C = \sum_{i \in \mathcal{N}} c_i \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j)$$

여기서:
- $c_i$: $i$번째 Gaussian의 색상 (Spherical Harmonics로 표현)
- $\alpha_i$: 불투명도(opacity)와 2D 투영된 Gaussian에 의해 결정되는 혼합 계수

각 3D Gaussian은 다음 파라미터로 정의됩니다:

$$\mathcal{G}(\mathbf{x}) = e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})}$$

여기서:
- $\boldsymbol{\mu}$: Gaussian의 중심 위치 (3D mean)
- $\boldsymbol{\Sigma}$: 공분산 행렬, $\boldsymbol{\Sigma} = \mathbf{R}\mathbf{S}\mathbf{S}^\top\mathbf{R}^\top$ ($\mathbf{R}$: 회전 행렬, $\mathbf{S}$: 스케일 행렬)

#### (B) FLoD의 핵심: 레벨별 3D 스케일 제약 (Scale Constraint)

FLoD는 각 레벨이 독립적으로 전체 장면을 재구성할 수 있는, 다양한 수준의 디테일과 메모리 요구사항을 제공하는 다중 레벨 3DGS 표현을 구성한다. 본 방법은 각 레벨별 3D 스케일 제약을 적용하여, 각 연속 레벨이 증가함에 따라 재구성되는 디테일 양과 렌더링 메모리 수요를 제한한다.

레벨 $l$에서의 스케일 제약은 다음과 같이 표현됩니다:

$$s_{\min}^{(l)} \leq s_k \leq s_{\max}^{(l)}, \quad \forall k \in \mathcal{G}^{(l)}$$

여기서:
- $s_k$: $k$번째 Gaussian의 3D scale
- $s_{\min}^{(l)}, s_{\max}^{(l)}$: 레벨 $l$에서의 스케일 하한과 상한
- $\mathcal{G}^{(l)}$: 레벨 $l$의 Gaussian 집합
- 레벨이 낮을수록(coarse) 더 큰 스케일만 허용 → 적은 Gaussian으로 전체 장면 표현

각 레벨의 Gaussian들은 스케일 범위를 설정하여 해당 레벨의 디테일 정도를 갖도록 설계된다.

#### (C) 레벨별 순차 학습 (Level-by-Level Training)

학습은 SfM(Structure from Motion) 포인트로부터 레벨 1에서 시작하여 최대 레벨까지 진행된다. 각 레벨의 학습에는 해당 레벨에 적합한 디테일을 제공하기 위한 스케일 제약 적용과 Gaussian 중복을 완화하기 위한 Overlap Pruning이 포함된다. 각 레벨의 학습이 완료되면 Gaussian 클론이 저장되어 다중 레벨 Gaussian 집합을 구성한다. 이 집합은 최대 레벨을 사용한 고품질 렌더링과 다중 레벨을 선택적으로 사용한 효율적 렌더링 모두를 가능하게 한다.

학습 흐름:

$$\text{Level } 1 \xrightarrow{\text{scale constraint} + \text{overlap pruning}} \text{save clone} \rightarrow \cdots \rightarrow \text{Level } L_{\max}$$

#### (D) Selective Rendering (선택적 렌더링)

훈련된 FLoD 표현은 가용 GPU 메모리 또는 원하는 렌더링 속도에 따라 임의의 단일 레벨을 선택하는 유연성을 제공한다. 또한 본 방법의 독립적이고 다중 레벨 구조는 이미지의 서로 다른 부분을 서로 다른 레벨의 디테일로 렌더링하는 것을 가능하게 하며, 이를 "Selective Rendering"이라 부른다.

선택적 렌더링의 최종 색상은 다음과 같이 표현됩니다:

$$C_{\text{selective}} = \sum_{l \in \mathcal{L}_{\text{sel}}} \sum_{i \in \mathcal{G}^{(l)}_{\text{region}}} c_i \alpha_i \prod_{j < i}(1-\alpha_j)$$

여기서 $\mathcal{L}_{\text{sel}}$은 선택된 레벨 집합이며, 근거리 영역에는 고레벨, 원거리 영역에는 저레벨 Gaussian을 할당합니다.

#### (E) Overlap Pruning

레벨별 학습 중 발생하는 Gaussian 중복을 제거하기 위해 overlap pruning을 수행합니다. 중복 Gaussian은 메모리 낭비와 아티팩트를 유발하므로, 일정 임계값 이상 겹치는 Gaussian을 제거합니다:

각 레벨의 학습은 해당 레벨에 적합한 디테일을 제공하기 위한 스케일 제약 적용과 Gaussian 중복을 완화하기 위한 Overlap Pruning을 포함한다.

---

### 2-3. 모델 구조 (아키텍처)

```
[Input: SfM Point Cloud]
        ↓
┌──────────────────────────────────────────┐
│           FLoD 다중 레벨 구조             │
│                                          │
│  Level 1 (Coarse): 큰 스케일 Gaussian    │
│      ↓ clone 저장 + 다음 레벨 학습 초기화 │
│  Level 2: 중간 스케일 Gaussian           │
│      ↓ clone 저장 + 다음 레벨 학습 초기화 │
│  Level 3 ...                             │
│      ↓                                   │
│  Level L_max (Fine): 작은 스케일         │
└──────────────────────────────────────────┘
        ↓
┌──────────────────────┐  ┌─────────────────────────┐
│  Single Level        │  │  Selective Rendering     │
│  Rendering           │  │  (Multi-level per region)│
│  (메모리에 따라 선택) │  │  (근거리 고레벨,원거리저레벨)│
└──────────────────────┘  └─────────────────────────┘
```

결과로 생성된 다중 레벨 3DGS(3DGS-FLoD)는 최고 품질을 위한 최대 레벨 렌더링 또는 보다 효율적인 렌더링을 위한 다중 레벨 선택적 렌더링을 지원한다.

또한 본 방법은 앵커 기반 신경 Gaussian을 활용하는 3DGS 변형인 Scaffold-GS에도 통합되어, 신경 Gaussian에 점진적으로 감소하는 스케일 제약을 적용하고 레벨별 학습 방법을 통해 최적화된 다중 레벨 Scaffold-GS 집합을 생성한다.

---

### 2-4. 성능 향상

FLoD는 고사양 서버와 저사양 노트북 모두에서 테스트하여 유연한 렌더링 옵션의 효과를 실증적으로 검증하였다. 실험은 3DGS 및 변형 연구에서 일반적으로 사용되는 Tanks and Temples, Mip-NeRF360 데이터셋뿐만 아니라 원거리 배경 요소를 포함하는 DL3DV-10K 데이터셋에서도 수행되었다.

비교 모델은 3DGS, Scaffold-GS, Mip-Splatting, Octree-GS, Hierarchical-3DGS이며, 이 중 주요 경쟁 모델은 LoD 개념을 FLoD와 공유하는 Octree-GS와 Hierarchical-3DGS이다.

주요 정량적 성능:

DL3DV-10K 데이터셋에서 PSNR 31.75, SSIM 0.935를 달성하여 기존 3DGS 접근법을 능가한다.

3DGS-FLoD는 Mip-Splatting과 비교하여 약 44% 적은 Gaussian을 사용하면서 유사한 성능을 달성하며, DL3DV-10K 데이터셋에서는 Mip-Splatting을 크게 능가한다.

레벨 5, 4, 3을 사용한 Selective Rendering은 레벨 5만 사용하는 것과 비교 가능한 시각적 품질을 달성하면서 Gaussian 수를 40% 감소시킨다.

3DGS-FLoD는 Tanks & Temples 데이터셋을 제외한 모든 데이터셋의 재구성 지표에서 베이스라인을 능가하며, 더 얇은 구조물을 포착하고 원거리 객체를 더 정확하게 표현한다. 이는 거친 수준부터 세밀한 수준까지 전체 구조를 포착하는 명시적 스케일 제약 기반의 coarse-to-fine 학습에서 비롯된다.

NVIDIA GeForce MX250 2GB GPU를 장착한 노트북에서도 본 방법의 결과를 실연하였으며, 다양한 하드웨어 구성에서 커스터마이즈 가능하고 메모리 효율적인 렌더링을 가능하게 한다.

---

### 2-5. 한계점

본 방법의 한계 중 하나는 모든 레벨을 저장하기 위해 상당한 디스크 저장 공간을 필요로 한다는 것이다.

또한 긴 카메라 궤적을 가진 장면에서는 빈번한 Gaussian 서브셋 업데이트가 프레임 드롭을 초래할 수 있으며, 멀티스레딩 구현이 이 문제를 부분적으로 완화했지만 여전히 지속된다.

향후 연구에서는 카메라와의 거리에 따라 적절한 레벨의 신경 Gaussian을 동적으로 생성하는 MLP를 사용하여 이 문제를 해결할 수 있으며, 이를 통해 Gaussian 서브셋 선택 과정으로 인한 FPS 저하를 줄이고 모든 레벨의 Gaussian 표현을 저장할 필요성을 제거할 수 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

FLoD는 다양한 3DGS 프레임워크에 일반화됨을 보여주며, 이는 미래의 최첨단 개발에 통합될 잠재력을 보여준다.

3D Gaussian에 3D 스케일 제약을 부과하는 방법의 단순성은 다른 3DGS 기반 기술과의 통합을 용이하게 한다.

일반화 가능성을 정리하면:

#### ① 기존 3DGS 변형으로의 일반화
FLoD는 Scaffold-GS에 통합되어 신경 Gaussian에 점진적으로 감소하는 스케일 제약을 적용하고, 레벨별 학습 방법을 통해 최적화된 다중 레벨 Scaffold-GS 집합을 생성한다.

#### ② 원거리 장면에서의 일반화
3DGS-FLoD는 Gaussian의 3D 스케일에 하한을 부과하여 원거리 객체를 재구성하는 데 우수하다.

#### ③ 하드웨어 독립적 일반화
FLoD는 사용되는 Gaussian의 수를 조정하여 다양한 하드웨어에서 확장 가능한 렌더링을 위해 3DGS에 도입된다.

#### ④ 압축 기법과의 결합 가능성
FLoD의 적응성은 다양한 3DGS 기반 모델과의 호환성을 제공하며, 이는 미래의 최첨단 3D 렌더링 개발에의 통합 가능성을 확장한다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 연도 | 핵심 방법 | LoD 지원 | 하드웨어 적응성 | 비고 |
|---|---|---|---|---|---|
| **NeRF** (Mildenhall et al.) | 2020 | MLP 기반 볼륨 렌더링 | ✗ | ✗ | 느린 렌더링 |
| **Mip-NeRF** (Barron et al.) | 2021 | 멀티스케일 anti-aliasing | 부분 | ✗ | 안티앨리어싱 초점 |
| **Mip-NeRF 360** (Barron et al.) | 2022 | 무경계 장면 | ✗ | ✗ | 무경계 확장 |
| **3DGS** (Kerbl et al.) | 2023 | 3D Gaussian primitives | ✗ | ✗ | 실시간 렌더링 |
| **LightGaussian** (Fan et al.) | 2023 | 압축/가지치기 | ✗ | 저사양 한정 | 고사양 활용 불가 |
| **Octree-GS** (Ren et al.) | 2024 | 옥트리 기반 LoD | ✓(누적) | 부분 | 레벨 독립성 없음 |
| **Hierarchical-3DGS** (Kerbl et al.) | 2024 | 계층적 표현 | ✓ | 고사양 한정 | 저사양 대응 어려움 |
| **Mip-Splatting** (Yu et al.) | 2024 | 3D/2D 필터링 | 부분 | ✗ | anti-aliasing 초점 |
| **Scaffold-GS** (Lu et al.) | 2024 | 앵커 기반 신경 Gaussian | ✗ | ✗ | 품질 향상 초점 |
| **FLoD (제안)** | 2024/2025 | 레벨별 스케일 제약 | ✓(독립) | **✓(광범위)** | 저·고사양 모두 지원 |
| **CLoD-GS** (Cheng et al.) | 2025 | 연속적 LoD | ✓(연속) | ✓ | popping artifact 해결 |
| **LODGE** (2025) | 2025 | 계층적 LoD + 청크 캐싱 | ✓ | ✓(모바일) | 대규모 장면 특화 |

FLoD에서는 각 레벨 표현이 독립적으로 장면을 재구성하는 반면, Octree-GS는 첫 번째 레벨부터 지정된 레벨까지의 표현을 집계하여 레벨을 정의하므로 개별 레벨이 독립적으로 존재하지 않는다.

CLoD-GS는 3DGS의 명시적이고 프리미티브 기반의 특성이 연속적 LoD(CLoD)라는 더 이상적인 패러다임을 가능하게 한다고 주장하며, 단일 통합 모델 내에서 부드럽고 seamless한 품질 스케일링을 통해 이산적 LoD의 핵심 문제를 해결하는 CLoD-GS 프레임워크를 소개한다.

LODGE는 메모리 제약 장치에서 대규모 장면의 실시간 렌더링을 가능하게 하는 새로운 LoD 방법으로, 카메라 거리에 기반하여 Gaussian의 최적 서브셋을 반복적으로 선택하는 계층적 LoD 표현을 도입하여 렌더링 시간과 GPU 메모리 사용을 크게 줄인다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5-1. 연구에 미치는 영향

#### ① 하드웨어 민주화 기여
FLoD는 3D 렌더링 분야에서 유연한 수준의 디테일을 3D Gaussian Splatting에 통합함으로써 중요한 진전을 나타내며, 이 접근법은 전통적으로 저사양 장치에서의 3DGS 배포를 제한해온 메모리 제약을 해결하여 적용 범위를 넓힌다.

#### ② 플러그인 방식의 범용 프레임워크 제시
FLoD는 기존 3DGS 변형에 쉽게 통합될 수 있으며, 동시에 렌더링 품질도 향상시킨다.

#### ③ 후속 LoD 연구의 기반 마련
FLoD는 이후 CLoD-GS, LODGE 등 연속·계층적 LoD 연구의 비교 기준(baseline)으로 활용되고 있습니다:
LOD 기반 방법 비교에서 H3DGS, Octree-GS, FLOD가 포함되어 있으며, FLOD는 45K/40K/100K 이터레이션으로 학습된다.

#### ④ 실용적 3D 렌더링 분야 확장
FLoD의 기술적 세부 내용과 실험 결과는 FLoD가 컴퓨터 그래픽스 분야에 상당한 영향을 미쳐 광범위한 응용 분야에서 더 효율적이고 맞춤화된 렌더링 솔루션을 가능하게 할 수 있음을 시사한다.

---

### 5-2. 앞으로 연구 시 고려할 점

| 고려 사항 | 설명 |
|---|---|
| **① 디스크 저장 문제 해결** | 모든 레벨 저장으로 인한 대용량 디스크 사용 → 압축 기법 결합 필요 |
| **② 동적 레벨 전환의 FPS 저하** | Gaussian 서브셋 업데이트 시 프레임 드롭 문제 |
| **③ 연속적 LoD로의 확장** | 이산 레벨 간 전환 시 발생할 수 있는 popping artifact 해결 필요 |
| **④ 자동 레벨 선택 메커니즘** | 사용자가 직접 레벨을 선택해야 하는 부담 → 자동화 연구 필요 |
| **⑤ 동적 장면으로의 확장** | 현재는 정적 장면에만 적용 → 동적 3DGS와의 통합 연구 필요 |
| **⑥ 압축과의 시너지** | LightGaussian 등 압축 기법과 결합 시 추가 메모리 절감 가능 |

향후 연구에서는 카메라와의 거리에 따라 적절한 레벨의 신경 Gaussian을 동적으로 생성하는 MLP를 사용하여 이를 해결할 수 있으며, Gaussian 서브셋 선택 과정으로 인한 FPS 저하를 줄이고 모든 레벨의 Gaussian 표현 저장 필요성을 제거할 수 있다.

향후 연구는 디스크 저장 효율 향상과 FLoD와 고급 렌더링 프레임워크와의 통합 최적화에 초점을 맞출 수 있다.

---

## 📚 참고 자료 및 출처

| # | 자료명 | 출처 |
|---|---|---|
| 1 | **FLoD: Integrating Flexible Level of Detail into 3D Gaussian Splatting for Customizable Rendering** (Seo et al., 2024/2025) | [arXiv:2408.12894](https://arxiv.org/abs/2408.12894) |
| 2 | **FLoD 공식 논문 (ACM Transactions on Graphics / SIGGRAPH 2025)** | [ACM DL: 10.1145/3731430](https://dl.acm.org/doi/10.1145/3731430) |
| 3 | **FLoD 프로젝트 페이지** | [3dgs-flod.github.io/flod](https://3dgs-flod.github.io/flod/) |
| 4 | **FLoD 공식 GitHub 구현** | [github.com/3DGS-FLoD/flod](https://github.com/3DGS-FLoD/flod) |
| 5 | **FLoD HTML 전문 (arXiv v1)** | [arxiv.org/html/2408.12894v1](https://arxiv.org/html/2408.12894v1) |
| 6 | **FLoD HTML 전문 (arXiv v2)** | [arxiv.org/html/2408.12894v2](https://arxiv.org/html/2408.12894v2) |
| 7 | **HuggingFace Paper Page** | [huggingface.co/papers/2408.12894](https://huggingface.co/papers/2408.12894) |
| 8 | **ADS Abstract (NASA Astrophysics Data System)** | [ui.adsabs.harvard.edu/abs/2024arXiv240812894S](https://ui.adsabs.harvard.edu/abs/2024arXiv240812894S/abstract) |
| 9 | **EmergentMind 논문 분석** | [emergentmind.com/papers/2408.12894](https://www.emergentmind.com/papers/2408.12894) |
| 10 | **ResearchGate 논문** | [researchgate.net/publication/383412860](https://www.researchgate.net/publication/383412860_FLoD_Integrating_Flexible_Level_of_Detail_into_3D_Gaussian_Splatting_for_Customizable_Rendering) |
| 11 | **CLoD-GS: Continuous Level-of-Detail via 3D Gaussian Splatting** (Cheng et al., 2025) | [arXiv:2510.09997](https://arxiv.org/abs/2510.09997) |
| 12 | **LODGE: Level-of-Detail Large-Scale Gaussian Splatting** (2025) | [arXiv:2505.23158](https://arxiv.org/abs/2505.23158) |
| 13 | **3D Gaussian Splatting for Real-Time Radiance Field Rendering** (Kerbl et al., 2023) | ACM TOG 42(4) |
| 14 | **Mip-NeRF 360** (Barron et al., 2022) | CVPR 2022 |
| 15 | **Octree-GS** (Ren et al., 2024) | arXiv 2024 |
| 16 | **Scaffold-GS** (Lu et al., 2024) | arXiv 2024 |
| 17 | **LightGaussian** (Fan et al., 2023) | arXiv:2311.17245 |

> ⚠️ **주의**: 논문의 스케일 제약 수식의 정확한 형태(예: 하한 값의 구체적 정의, 레벨 간 스케일 비율 등)는 논문 전문(PDF) 직접 열람을 통해 확인하시기를 권장합니다. 본 답변에서 제시한 수식은 논문에서 공개된 개념적 설명을 수식화한 것으로, 논문 본문의 정확한 표기와 일부 차이가 있을 수 있습니다.
