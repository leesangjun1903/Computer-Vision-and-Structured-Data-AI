
# GS2Mesh: Surface Reconstruction from Gaussian Splatting via Novel Stereo Views

> **논문 정보:**
> - **저자:** Yaniv Wolf, Amit Bracha, Ron Kimmel (Technion - Israel Institute of Technology)
> - **발표:** ECCV 2024
> - **arXiv:** [arXiv:2404.01810](https://arxiv.org/abs/2404.01810)
> - **프로젝트 페이지:** [gs2mesh.github.io](https://gs2mesh.github.io/)
> - **GitHub:** [yanivw12/gs2mesh](https://github.com/yanivw12/gs2mesh)
> - **Springer DOI:** [10.1007/978-3-031-73024-5_13](https://doi.org/10.1007/978-3-031-73024-5_13)

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

최근 3D Gaussian Splatting(3DGS)은 장면을 효율적으로 표현하는 방법으로 주목받고 있다. 그러나 뛰어난 Novel View Synthesis 능력에도 불구하고, Gaussian 속성으로부터 직접 기하 구조를 추출하는 것은 어려운 과제이다. 이는 Gaussian들이 Photometric Loss 기반으로 최적화되기 때문이다. 일부 동시대 모델들이 Gaussian 최적화 과정에 기하학적 제약을 추가하려 했지만, 여전히 노이즈가 많고 비현실적인 표면을 생성하는 문제가 남아 있다.

GS2Mesh는 이 문제를 해결하기 위해 전혀 다른 접근법을 제안한다. 즉, 노이즈가 많은 3DGS 표현과 매끄러운 3D Mesh 표현 사이의 간극을 메우기 위해 **깊이 추출 과정에 실세계 지식(real-world knowledge)을 주입**하는 새로운 방법을 제안한다. Gaussian 속성에서 직접 기하를 추출하는 대신, **사전 학습된 스테레오 매칭 모델**을 통해 기하를 추출한다.

### 1.2 주요 기여 (Contributions)

| 번호 | 기여 내용 |
|---|---|
| ① | 3DGS 기반 노이즈 Gaussian Point Cloud → 매끄러운 3D Mesh 변환 파이프라인 신규 제안 |
| ② | Novel View Synthesis 능력을 활용한 Stereo 이미지 쌍 렌더링 전략 |
| ③ | 사전 학습 Stereo 매칭 모델(DLNR 등)을 기하학적 Prior로 활용하는 방식 |
| ④ | Occlusion Mask를 활용한 기하학적 일관성 향상 |
| ⑤ | TSDF Fusion + Marching Cubes로 다중 뷰 깊이 맵 통합 |
| ⑥ | DTU, Tanks and Temples 벤치마크에서 SOTA 달성 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능, 한계

### 2.1 해결하고자 하는 문제

3DGS로부터 직접 표면을 복원하는 것에는 중요한 과제가 있다. 핵심 문제는 3D 공간에서 Gaussian 요소들의 위치가 기하학적으로 일관된 표면을 형성하지 않는다는 것인데, 이는 Gaussian들이 원래 이미지 평면으로 다시 투영될 때 입력 이미지에 가장 잘 매칭되도록 최적화되기 때문이다.

3DGS는 새로운 시점에서 이미지를 생성하는 데 탁월하지만, 최적화된 Gaussian 파라미터를 깨끗하고 정확한 메쉬로 직접 변환하는 것은 문제가 있었다. 핵심 문제는 3DGS에서 Gaussian 파라미터가 기하학적 정확도가 아닌 주로 Photometric 일관성(이미지 외관)을 위해 최적화되어 노이즈가 많거나 비현실적인 표면을 생성한다는 것이다.

### 2.2 제안하는 방법 및 파이프라인

GS2Mesh는 Gaussian 요소의 위치를 표면 복원의 Prior로 사용하는 대신, 3DGS의 뛰어난 Novel View Synthesis 능력을 활용한다. 이를 위해 3DGS 모델로 스테레오 보정된 Novel View 쌍을 렌더링하고 스테레오 매칭 방법을 사용하여 깊이 프로파일을 추출한다. 그런 다음 추출된 RGB-D 이미지를 기하학적으로 일관된 표면으로 결합한다.

**전체 파이프라인은 다음 4단계로 구성된다:**

```
입력 이미지 → [1단계] 3DGS 최적화
           → [2단계] 스테레오 이미지 쌍 렌더링
           → [3단계] 스테레오 매칭 → 깊이 맵(Disparity)
           → [4단계] TSDF Fusion + Marching Cubes → 최종 Mesh
```

#### Step 1: 3DGS 최적화

3DGS는 최근 Novel View Rendering 분야에서 큰 도약을 이루었으며, 이전 신경망 렌더링 방법들을 속도와 정확도 모두에서 능가했다. Gaussian 요소들의 분포, 크기, 색상, 불투명도를 최적화하고 이를 가상 카메라에 투영(Splatting)함으로써 3DGS는 실시간으로 복잡한 장면의 실제적인 이미지를 생성할 수 있다.

3DGS의 각 Gaussian은 다음 파라미터로 정의된다:

$$\mathcal{G}_i = \{\mu_i \in \mathbb{R}^3, \Sigma_i \in \mathbb{R}^{3\times3}, c_i, \alpha_i\}$$

여기서 $\mu_i$는 중심 위치, $\Sigma_i$는 공분산 행렬(형태/크기), $c_i$는 색상(SH 계수), $\alpha_i$는 불투명도다.

3DGS의 렌더링(알파 합성)은 다음과 같다:

$$\hat{C}(\mathbf{r}) = \sum_{i \in \mathcal{N}} c_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)$$

그리고 Photometric Loss로 학습된다:

$$\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}}$$

#### Step 2: 스테레오 이미지 쌍 생성

오른쪽 이미지의 포즈 $(R_R, T_R)$는 수평 베이스라인 $b$를 도입하여 계산된다: $R_R = R_L$, $T_R = T_L + (R_L \times [b, 0, 0])$. 이렇게 하면 왼쪽과 오른쪽 이미지가 스테레오 보정된다. 베이스라인 $b$는 매우 중요한데, 베이스라인이 클수록 스테레오 깊이 추정이 더 정확하지만, 너무 크면 원래 학습 뷰에서 너무 멀어져 렌더링 노이즈가 발생할 수 있다. 저자들은 실험적으로 **장면 반경의 7%의 베이스라인이 최적**임을 확인했다.

즉, 각 원래 학습 포즈 $(R_L, T_L)$에 대해 스테레오 우측 뷰 포즈를 다음과 같이 정의한다:

$$R_R = R_L, \quad T_R = T_L + R_L \cdot \begin{bmatrix} b \\ 0 \\ 0 \end{bmatrix}$$

#### Step 3: 스테레오 매칭 → 깊이 추정

렌더링된 스테레오 이미지 쌍은 사전 학습된 스테레오 매칭 모델에 입력된다. 이 논문에서는 Middlebury 사전 학습 가중치를 사용한 최신 신경망 스테레오 매칭 네트워크인 **DLNR**을 사용하며, 최상의 결과를 얻는다. RAFT-Stereo 등 다른 모델도 테스트되었다.

스테레오 시스템의 핵심 수식인 깊이-Disparity 변환은 다음과 같다:

$$Z = \frac{f \cdot b}{d}$$

여기서 $Z$는 깊이, $f$는 초점 거리(focal length), $b$는 베이스라인, $d$는 Disparity(좌우 이미지 간 픽셀 차이)이다.

기하학적 일관성을 높이기 위해 중요한 단계로 **Occlusion Masking**이 적용된다. Occlusion Mask는 좌→우 Disparity와 우→좌 Disparity를 비교하여 생성된다.

#### Step 4: TSDF Fusion + Marching Cubes

기하학적 일관성을 더욱 강화하고 개별 깊이 프로파일에서 발생할 수 있는 노이즈와 오류를 제거하기 위해, 추출된 모든 깊이를 **TSDF(Truncated Signed Distance Function) 알고리즘**으로 통합한 후, **Marching-Cubes 메시 알고리즘**을 적용한다.

TSDF는 각 복셀 $\mathbf{v}$에 대해 표면까지의 절단된 부호 거리를 계산한다:

$$\text{TSDF}(\mathbf{v}) = \text{trunc}\left(\frac{d(\mathbf{v})}{\tau}\right), \quad \text{trunc}(x) = \text{clip}(x, -1, 1)$$

여기서 $d(\mathbf{v})$는 복셀에서 가장 가까운 표면까지의 부호 거리, $\tau$는 절단 임계값이다. 가중 누적을 통해:

$$\text{TSDF}(\mathbf{v}) \leftarrow \frac{W(\mathbf{v}) \cdot \text{TSDF}(\mathbf{v}) + w_i \cdot d_i(\mathbf{v})}{W(\mathbf{v}) + w_i}$$

최종적으로 TSDF의 Zero-crossing에서 Marching Cubes로 메쉬를 추출한다.

### 2.3 모델 구조 요약

제안된 파이프라인은 다음과 같이 구성된다: 먼저 3DGS 모델을 적용하여 장면을 표현한다. 그런 다음 3DGS 모델을 사용하여 원래 뷰에 해당하는 스테레오 정렬 이미지 쌍을 렌더링한다. 각 쌍에 대해 스테레오 알고리즘을 사용하여 RGB-D 구조를 재구성하고, 이를 모든 뷰에서 TSDF를 사용하여 통합하여 장면의 삼각형 메쉬를 생성한다.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GS2Mesh 파이프라인                           │
├──────────────┬──────────────┬──────────────┬────────────────────────┤
│   입력 이미지  │   3DGS 최적화 │  Stereo 렌더링 │  깊이 → Mesh 생성       │
│   + COLMAP   │  (7k-60k     │  (수평 Baseline│  DLNR / RAFT-Stereo   │
│   SfM        │   반복)       │   b = 7% r)   │  → TSDF → Marching    │
│              │              │              │    Cubes              │
└──────────────┴──────────────┴──────────────┴────────────────────────┘
```

### 2.4 성능 향상

본 방법은 다른 Splatting 기반 방법들을 능가하며, 실행 시간이 훨씬 짧음에도 불구하고 Neuralangelo와 같은 최신 신경망 기반 방법들과 비교 가능한 성능을 보인다.

스마트폰으로 촬영된 실제 환경(In-the-Wild) 장면에서 광범위한 테스트를 수행하여 뛰어난 재구성 능력을 보여주었고, Tanks and Temples 및 DTU 벤치마크 테스트에서 State-of-the-Art 결과를 달성했다.

본 방법은 다른 재구성 방법들에 비해 훨씬 빠르게 실행되며, 파이프라인의 병목 지점은 3DGS 최적화 시간이다. 약 80개 이미지를 포함하는 일반적인 In-the-Wild 장면에서, Nvidia L40 GPU 기준 3DGS 최적화는 반복 횟수에 따라 5~30분 소요된다.

**정량적 결과 (DTU 데이터셋, Chamfer Distance 기준):**

본 방법은 다른 Splatting 기반 방법들을 능가하며, Neuralangelo 같은 최신 신경망 방법들과 비교 가능한 수준이면서 실행 시간은 훨씬 짧다.

| 방법 | 유형 | DTU (Chamfer ↓) | 속도 |
|---|---|---|---|
| NeuS | Neural Implicit | 낮음 | 느림(수 시간) |
| Neuralangelo | Neural Implicit | 매우 낮음 | 매우 느림 |
| SuGaR | 3DGS 기반 | 보통 | 보통 |
| 2DGS | 3DGS 기반 | 낮음 | 보통 |
| GOF | 3DGS 기반 | 낮음 | 빠름 |
| **GS2Mesh** | **3DGS + Stereo** | **매우 낮음** | **빠름** |

### 2.5 한계점

파이프라인은 3DGS, 스테레오를 통한 깊이 추출, TSDF Fusion으로 구성되어 있으며, 각 단계는 최종 재구성에 영향을 미칠 수 있는 한계가 있다. 3DGS는 원래 학습 이미지에 충분히 포함되지 않은 영역에서 노이즈가 많은 "Floater" Gaussian을 생성할 수 있다. 또한 스테레오 매칭 모델은 **투명한 표면**에서 어려움을 겪는 것으로 알려져 있다. 마지막으로 **TSDF Fusion은 대규모 장면**(예: TnT의 Meetingroom 및 Courthouse)에서 확장성이 떨어진다. 미래에 개선된 정확도와 견고성을 가진 3DGS 및 스테레오로 교체하고 대규모 장면에 더 적합한 Fusion 방법을 추가하면 이러한 한계를 완화하는 데 도움이 될 것이다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능의 핵심 원천: Pre-trained Stereo Prior

GS2Mesh는 제어되지 않은 비디오에서의 다중 뷰 3D 재구성을 위한 새로운 파이프라인을 도입하여, 노이즈가 많은 Gaussian Splatting Cloud에서 매끄러운 메쉬를 추출할 수 있다. 다른 기하 추출 방법들이 Gaussian 요소의 위치에 의존하고 기하학적 제약으로 이를 정규화하려는 것과 달리, **사전 학습된 스테레오 모델을 실세계 기하학적 Prior로 활용**하여 모든 뷰에서 정확한 깊이를 추출하고, 추가 최적화 없이 함께 Fusion하여 매끄러운 메쉬를 생성한다.

즉, 일반화 성능 향상의 핵심 메커니즘은 다음과 같다:

1. **Pre-trained Stereo Model의 일반화 능력 활용**: DLNR이나 RAFT-Stereo 같은 스테레오 매칭 모델은 대규모 다양한 데이터셋(Middlebury 등)에서 학습되어 다양한 장면 유형에 대한 강력한 Prior를 보유한다.

2. **Plug-and-Play 구조**: 미래에 정확도와 견고성이 향상된 3DGS 및 스테레오 버전으로 교체하면 이러한 한계를 완화하는 데 도움이 될 것이다. 이는 모델의 각 컴포넌트를 독립적으로 업그레이드할 수 있는 모듈형 설계임을 시사한다.

3. **In-the-Wild 장면 처리**: 스마트폰으로 촬영된 In-the-Wild 장면에서 광범위한 테스트를 수행하여 뛰어난 재구성 능력을 입증했다. 이는 실세계 다양한 환경에서의 강력한 일반화를 의미한다.

4. **SAM/SAM2와의 결합**: 세분화 방법과의 결합으로 객체 특정 표면 복원이 가능하며, 이는 방법론의 다재다능함과 효율성을 더욱 보여준다.

### 3.2 일반화 확장 가능성 분석

Novel Stereo View 활용의 이점을 강화하기 위해 두 가지 Ablation이 수행되었는데, 이는 딥 MVS 방법으로 파이프라인의 단계를 교체하는 것을 포함한다. 원래 이미지에서의 MVS: 방법은 각 원래 포즈에서 해당 포즈의 스테레오 정렬 뷰를 생성하고 사전 학습된 스테레오 매칭 모델을 적용하여 깊이를 추출한다. 첫 번째 Ablation으로 원래 이미지에서 사전 학습된 딥 MVS 모델을 실행하고 TSDF를 사용하여 결과 깊이를 Fusion했다.

이 Ablation 연구는 **렌더링된 스테레오 쌍이 원본 이미지 대비 노이즈 감소와 카메라 왜곡 제거에 효과적**임을 입증하며, 이는 다양한 촬영 조건에서의 일반화를 직접적으로 지원한다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 관련 연구 카테고리

#### 4.1.1 Neural Implicit Surface 방법

| 방법 | 연도 | 특징 |
|---|---|---|
| NeRF (Mildenhall et al.) | 2020 | 볼륨 렌더링 기반 뷰 합성 |
| NeuS (Wang et al.) | 2021 | SDF + 볼륨 렌더링으로 표면 복원 |
| VolSDF (Yariv et al.) | 2021 | 부호 거리 함수 기반 표면 복원 |
| BakedSDF (Yariv et al.) | 2023 | 실시간 뷰 합성을 위한 SDF 메쉬화 |
| Neuralangelo (Li et al.) | 2023 | 고해상도 SDF 표면 복원 |

Mip-NeRF360 데이터셋의 정성적 결과에서, 본 방법은 테이블의 작은 균열과 같은 세밀한 부분을 재구성할 수 있었으며, BakedSDF 방법과 유사한 수준의 결과를 얻으면서도 실행 시간은 현저히 짧았다.

#### 4.1.2 3DGS 기반 표면 복원 방법

SuGaR (Guédon & Lepetit, 2024)는 Gaussian Fitting을 장려하고 Poisson 재구성을 사용하여 메쉬를 추출하는 최초의 작업이었지만, 기하학적 정규화가 부족하여 제한된 정확도를 보였다. 3D Gaussian을 평면으로 근사하는 아이디어는 2D Gaussian Splatting(2DGS)과 PGSR로 확장되었는데, 전자는 3DGS를 2D 평면으로 압축하고, 후자는 교차 뷰 기하학적 일관성을 강조한다.

RaDe-GS는 래스터화 기술을 적용하여 깊이 및 법선 맵을 계산함으로써 3D Gaussian의 표현 능력을 향상시켰다. GS2Mesh는 스테레오 매칭 알고리즘을 통합하여 RGB-D 구조를 구성하고 TSDF를 사용하여 여러 뷰를 삼각형 메쉬로 통합함으로써 TSDF 메시화를 발전시켰다.

**주요 방법 비교:**

| 방법 | 연도 | 전략 | 장점 | 단점 |
|---|---|---|---|---|
| **SuGaR** | CVPR 2024 | Gaussian 정렬 + Poisson 재구성 | 직관적 | 낮은 정밀도 |
| **2DGS** | SIGGRAPH 2024 | 2D Gaussian Surfel | 얇은 표면 처리 우수 | 배경 재구성 취약 |
| **GOF** | SIGGRAPH Asia 2024 | Ray-tracing 기반 Opacity Field | 세밀한 기하 | 복잡한 구조 |
| **3DGSR/GSDF** | 2024 | 3DGS + SDF 공동 최적화 | 암묵 표면 강점 | 계산 비용 높음 |
| **GS2Mesh** | ECCV 2024 | 스테레오 Prior + TSDF | 매끄러운 메쉬, 빠른 속도 | 투명 표면, 대규모 장면 취약 |

Gaussian Opacity Fields(GOF)의 평가 결과, GOF는 표면 복원과 Novel View Synthesis 모두에서 기존 3DGS 기반 방법들을 능가하며, 품질과 속도 모두에서 신경망 암묵적 방법들과 비슷하거나 능가하는 성능을 보인다.

---

## 5. 향후 연구에 미치는 영향 및 고려 사항

### 5.1 향후 연구에 미치는 영향

#### 영향 1: "Pre-trained Geometric Prior" 패러다임의 확립

GS2Mesh는 제어되지 않은 비디오에서의 다중 뷰 3D 재구성을 위한 새로운 파이프라인을 제시하며, 다른 기하 추출 방법들이 Gaussian 요소의 위치에 의존하는 것과 달리, 사전 학습된 스테레오 모델을 실세계 기하학적 Prior로 단순 활용하는 방식을 도입했다. 이 패러다임은 기하학적 딥러닝 Prior를 다양한 3D 재구성 방법에 연결하는 후속 연구를 촉발할 수 있다.

#### 영향 2: 모듈형 파이프라인 연구의 촉진

미래에 정확도와 견고성이 향상된 3DGS 및 스테레오 버전으로 교체하고, 대규모 장면에 더 적합한 Fusion 방법을 추가하면 이러한 한계들을 완화하는 데 도움이 될 것이다. 이는 각 모듈이 독립적으로 발전할 수 있는 연구 방향을 제시한다.

#### 영향 3: 스마트폰 기반 고품질 3D 재구성

스마트폰 카메라를 사용하여 제어되지 않은 In-the-Wild 조건의 다양한 장면에 대한 광범위한 테스트를 통해 방법의 효과가 확인됐다. 이는 민주화된(democratized) 3D 재구성 도구 개발에 직접적인 영감을 제공한다.

#### 영향 4: 3DGS 확장 연구의 표준 기준점

SAM2를 활용한 자동 배경 제거, 추가 최신 GS 모델 및 gsplat 프레임워크 지원 추가를 포함한 지속적인 개발이 이루어지고 있어, 후속 연구의 기준 방법론으로 활용될 가능성이 높다.

### 5.2 향후 연구 시 고려할 점

#### 고려사항 1: 대규모 장면 확장성 문제

TSDF Fusion은 Tanks and Temples의 Meetingroom 및 Courthouse 장면과 같은 대규모 장면에서는 확장성이 떨어진다. 따라서 **Neural TSDF, OctoMap, 또는 Gaussian Splatting 기반 대규모 Fusion 방법** 연구가 필요하다.

#### 고려사항 2: 투명/반투명 표면 처리

스테레오 매칭 모델은 투명한 표면에서 어려움을 겪는 것으로 알려져 있다. 투명 표면 처리는 **깊이 추정의 근본적 한계**이므로, 물리 기반 렌더링(PBR)이나 편광 카메라 기반 접근 등과의 결합이 요구된다.

#### 고려사항 3: 스테레오 베이스라인 자동 최적화

베이스라인 $b$는 매우 중요한데, 너무 크면 렌더링 노이즈가 발생할 수 있으며, 저자들은 경험적으로 장면 반경의 7%가 최적임을 확인했다. 장면별 자동 베이스라인 조정 메커니즘 연구가 일반화를 더욱 향상시킬 수 있다.

#### 고려사항 4: 동적 장면 처리

현재 GS2Mesh는 정적 장면을 가정한다. 동적 객체나 시간에 따라 변하는 장면 처리를 위해서는 **Dynamic 3DGS + Temporal Stereo Matching** 통합 연구가 필요하다.

#### 고려사항 5: Foundation Model과의 통합

세분화 방법과의 결합으로 객체 특정 표면 복원이 가능하며 방법론의 다재다능함을 보여준다. 앞으로는 SAM2, DUSt3R, MASt3R 같은 Foundation Model과의 통합을 통한 Zero-shot 3D 재구성 파이프라인 연구가 유망하다.

#### 고려사항 6: 스파스 뷰 조건에서의 일반화

기존 방법들은 희소 시점 입력에서 고품질 메쉬를 생성하는 데 실패하고 교차 장면 일반화가 부족하다. GS2Mesh도 입력 이미지 수가 제한될 때 3DGS 품질 자체가 저하되므로, 스파스 뷰 조건에서의 견고성 향상이 주요 연구 과제가 된다.

---

## 참고 문헌 (References)

| # | 참고 자료 |
|---|---|
| 1 | Wolf, Y., Bracha, A., Kimmel, R. **"GS2Mesh: Surface Reconstruction from Gaussian Splatting via Novel Stereo Views."** ECCV 2024. arXiv:2404.01810 |
| 2 | **GS2Mesh 공식 프로젝트 페이지:** https://gs2mesh.github.io/ |
| 3 | **GS2Mesh GitHub (공식 구현):** https://github.com/yanivw12/gs2mesh |
| 4 | **Springer ECCV 2024 출판본:** https://doi.org/10.1007/978-3-031-73024-5_13 |
| 5 | **ECCV 2024 공식 PDF:** https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/12486.pdf |
| 6 | **OpenReview (Wild3D @ ECCV 2024):** https://openreview.net/forum?id=4WEtbfizmh |
| 7 | **Moonlight Literature Review:** https://www.themoonlight.io/en/review/gs2mesh-... |
| 8 | Kerbl, B. et al. **"3D Gaussian Splatting for Real-Time Radiance Field Rendering."** ACM TOG 42.4 (2023) |
| 9 | Guédon, A., Lepetit, V. **"SuGaR: Surface-Aligned Gaussian Splatting."** CVPR 2024 |
| 10 | Huang, B. et al. **"2D Gaussian Splatting for Geometrically Accurate Radiance Fields."** SIGGRAPH 2024 |
| 11 | Yu, Z. et al. **"Gaussian Opacity Fields."** SIGGRAPH Asia 2024 / ACM TOG 2024 |
| 12 | Wang, P. et al. **"NeuS: Learning Neural Implicit Surfaces by Volume Rendering."** NeurIPS 2021 |
| 13 | **PMC Survey:** "A survey on surface reconstruction based on 3D Gaussian splatting." https://pmc.ncbi.nlm.nih.gov/articles/PMC12453780/ |
| 14 | Knapitsch, A. et al. **"Tanks and Temples."** ACM TOG 36.4 (2017) |
| 15 | Jensen, R. et al. **"Large Scale Multi-View Stereopsis Evaluation (DTU)."** CVPR 2014 |
