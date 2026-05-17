
# GaussianVideo: Efficient Video Representation via Hierarchical Gaussian Splatting 

> **논문 정보**
> - **저자**: Andrew Bond, Jui-Hsien Wang, Long Mai, Erkut Erdem, Aykut Erdem
> - **소속**: Koç University, Adobe Research, Hacettepe University
> - **arXiv**: [arXiv:2501.04782](https://arxiv.org/abs/2501.04782) (2025년 1월 8일)
> - **학회**: ICCV 2025 (OpenAccess 확인)
> - **프로젝트 페이지**: https://cyberiada.github.io/GaussianVideo/

---

## 1. 🔑 핵심 주장 및 주요 기여 요약

### 핵심 주장

동적 비디오 장면을 위한 효율적인 신경 표현(Neural Representation)은 비디오 압축부터 인터랙티브 시뮬레이션까지 다양한 응용에서 매우 중요하다. 그러나 기존 방법들은 높은 메모리 사용량, 긴 학습 시간, 시간적 일관성(temporal consistency) 문제에 직면해 있다.

GaussianVideo는 이 세 가지 문제를 동시에 해결하는 새로운 신경 비디오 표현(neural video representation) 프레임워크로, in-the-wild 비디오를 효과적으로 모델링하면서 학습 효율성을 유지하고, 최소한의 감독(supervision)으로 의미론적 모션(semantic motions)을 포착하는 새로운 Gaussian Splatting 프레임워크를 제시한다.

### 주요 기여 (3가지)

논문의 핵심 기술 기여는 다음과 같다:
(1) **B-spline 기반 모션 표현**: 장면 요소의 부드러운 모션 궤적을 모델링하여, 시간적 일관성을 보장하면서 모션의 국소적 변화를 허용한다.
(2) **계층적 학습 전략(Hierarchical Learning Strategy)**: Gaussian 표현의 공간적·시간적 특징을 점진적으로 정제하여, 재구성 품질을 개선하고 학습 수렴을 가속화한다.

(3) 3D Gaussian Splatting과 연속 카메라 모션 모델링을 결합한 신규 신경 비디오 표현 도입 — Neural ODE를 활용하여 부드러운 카메라 궤적을 학습하면서 Gaussian을 통한 명시적 3D 장면 표현을 유지한다.

---

## 2. 🔬 상세 설명

### 2-1. 해결하고자 하는 문제

기존 방법(Splatter-a-Video 등)은 Gaussian의 의미론적 움직임을 보조 감독 신호(auxiliary supervisory signals)에 의해 강제하는 방식을 취한다. 그러나 이러한 감독 신호 자체가 계산 비용이 높고, 정확하게 추출하기 어려우며, 표현 자체의 설계 선택의 부정적 영향을 숨길 수 있다.

따라서 GaussianVideo는 보조 신호 없이 **의미론적 Gaussian 움직임이 자연스럽게 발현(emergent behavior)되도록** 설계하는 것을 목표로 한다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### 🔷 (A) 3D Gaussian Splatting 기반 비디오 표현

3D Gaussian Splatting(3DGS)은 원래 정적 3D 장면을 학습하기 위한 방법으로 도입되었으며, 이후 동적 장면으로의 확장이 활발히 연구되어 왔다.

각 Gaussian $\mathcal{G}$는 다음의 파라미터로 정의된다:

$$\mathcal{G} = \{\mu, \Sigma, \alpha, \mathbf{c}\}$$

- $\mu \in \mathbb{R}^3$: 위치 (mean)
- $\Sigma \in \mathbb{R}^{3 \times 3}$: 공분산 행렬 (크기·방향 인코딩)
- $\alpha \in [0,1]$: 불투명도(opacity)
- $\mathbf{c}$: Spherical Harmonics(SH) 계수로 인코딩된 색상

3D Gaussian의 공간 분포는 다음과 같이 표현된다:

$$G(\mathbf{x}) = \exp\left(-\frac{1}{2}(\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu)\right)$$

공분산 행렬은 학습 안정성을 위해 스케일 행렬 $S$와 회전 행렬 $R$로 분해된다:

$$\Sigma = R S S^T R^T$$

#### 🔷 (B) B-spline 기반 모션 표현

계층적 Gaussian Splatting 기반의 효율적인 신경 비디오 표현 방법은 B-spline 기반 모션 궤적과 Neural ODE 기반 카메라 모델링을 통합하여, 사전 카메라 파라미터나 무거운 감독 없이 정적·동적 요소를 모두 포착한다.

각 Gaussian의 시간 $t$에서의 위치는 B-spline 제어점(control points) $\{\mathbf{p}_k\}$로 표현된다:

$$\mu(t) = \sum_{k} B_{k,d}(t) \cdot \mathbf{p}_k$$

여기서 $B_{k,d}(t)$는 $d$차 B-spline 기저 함수(basis function)이다. B-spline의 특성상 **전체 궤적의 부드러움(global smoothness)**을 보장하면서도 **국소적 변화(local variation)**를 허용한다.

#### 🔷 (C) Neural ODE 기반 카메라 모델링

Neural ODE를 활용하여 부드러운 카메라 궤적을 학습하면서 Gaussian을 통한 명시적 3D 장면 표현을 유지한다.

카메라 파라미터의 연속 시간 진화는 다음 Neural ODE로 모델링된다:

$$\frac{d\mathbf{h}(t)}{dt} = f_\theta(\mathbf{h}(t), t)$$

$$\mathbf{h}(t) = \mathbf{h}(t_0) + \int_{t_0}^{t} f_\theta(\mathbf{h}(s), s) \, ds$$

여기서 $f_\theta$는 신경망으로 파라미터화된 동역학 함수이며, $\mathbf{h}(t)$는 카메라 상태(위치, 방향 등)를 나타내는 잠재 벡터이다. 학습된 카메라 표현이 있으면 어떤 형태든 성능 향상에 유리하며, Neural ODE로 카메라 모션을 모델링할 때 성능이 크게 향상된다.

#### 🔷 (D) 렌더링 (Splatting)

2D 투영된 Gaussian을 이용한 최종 픽셀 색상 렌더링은 alpha-blending으로 이루어진다:

$$C(\mathbf{r}) = \sum_{i \in \mathcal{N}} \mathbf{c}_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)$$

여기서 $\mathbf{r}$은 카메라 레이(ray), $\mathcal{N}$은 depth-sorted Gaussian 집합이다.

#### 🔷 (E) 손실 함수

논문의 공개 HTML에서 확인되는 손실은 기본적으로 픽셀 재구성 손실과 정규화 항으로 구성된다:

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_{\text{smooth}} \mathcal{L}_{\text{smooth}}$$

재구성 손실은 L1과 SSIM 조합으로 구성되며:

$$\mathcal{L}_{\text{recon}} = (1-\lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{\text{SSIM}}$$

$\mathcal{L}_{\text{smooth}}$는 B-spline 궤적의 시간적 부드러움을 강제하는 정규화 항이다.

> ⚠️ **주의**: 세부 손실 가중치 등 정확한 수식은 논문 전문(PDF) 기준이며, 위 수식은 공개된 arXiv HTML 및 관련 3DGS 표준 수식에 기반합니다.

---

### 2-3. 모델 구조

GaussianVideo 접근 방식의 파이프라인은 Neural ODE를 통한 연속 카메라 모션 모델링과 3D Gaussian Splatting을 결합하며, (a) 공간 및 (b) 시간 도메인 모두에 대한 계층적 학습 전략을 포함하여 세부 사항과 부드러운 모션을 포착하기 위해 Gaussian을 점진적으로 정제한다.

구체적으로 모델 구조는 다음으로 구성된다:

| 구성 요소 | 역할 |
|---|---|
| **Gaussian Primitives** | 장면의 명시적 3D 표현 (~400K Gaussians/video) |
| **B-spline 모션 모듈** | 각 Gaussian의 시간적 궤적 모델링 |
| **Neural ODE 카메라 모듈** | 연속 카메라 파라미터 학습 |
| **공간적 계층 학습** | 저해상도 → 고해상도로 점진적 Gaussian 정제 |
| **시간적 계층 학습** | 짧은 구간 → 긴 구간으로 점진적 학습 |
| **SH 계수** | 뷰 의존적(view-dependent) 색상 표현 |

이 방법은 장면 전반에 걸쳐 객체를 의미론적으로 추적(semantically track)하는 능력을 보여준다. 이를 시각화하기 위해 비디오당 사용되는 400K Gaussian 중 100K를 서브샘플링하여 반경 1의 구(sphere)로 축소하고 시간에 따른 움직임을 렌더링한다.

---

### 2-4. 성능 향상

GaussianVideo는 두 데이터셋(DL3DV, DAVIS Tap-Vid)에서 기존 방법들과 비교하여 세 가지 평가 지표(PSNR, SSIM, LPIPS) 모두에서 일관되게 최고 점수를 달성한다.

960×540 비디오를 NVIDIA A40 GPU에서 93 FPS로 렌더링하며, 해당 비디오에서의 재구성 PSNR은 44.21로 NeRV의 29.36 대비 약 50.6% 향상되었다.

DL3DV와 DAVIS Tap-Vid 벤치마크 데이터셋에서 PSNR, SSIM, LPIPS 지표로 검증되었다.

비교 대상 방법:
- **NeRV** (2021): HNeRV는 트랙을 따라 바위를 선명하게 렌더링하는 데 어려움을 겪어 흐릿한 표현을 보이는 반면, GaussianVideo는 날카롭고 세밀한 재구성을 제공한다.
- **GaussianImage**: DL3DV 데이터셋 비디오 비교에서, GaussianImage 비디오의 하늘 부분에 눈에 띄는 노이즈가 나타나는 반면, GaussianVideo는 더 부드럽고 균일한 하늘을 생성하여 더 높은 재구성 품질을 보인다.

추가 다운스트림 응용:
- **프레임 보간(Frame Interpolation)**: 원래 프레임레이트를 유지하면서 기존 프레임 사이를 매끄럽게 보간하여 프레임 수를 두 배로 늘리며, 부드러운 모션과 시간적 일관성을 보존한다.
- **비디오 스타일화(Video Stylization)**: 첫 프레임에 스타일을 적용하고, SH 계수만 업데이트하여 추가 감독 없이 비디오 전체에 스타일을 전파한다. 프레임 간의 시각적 일관성이 높은 품질의 스타일 전이를 유지한다.
- **공간 리샘플링(Spatial Resampling)**: 프레임 높이를 1.5배로 스케일링하는 등 학습된 카메라 파라미터의 스케일, 주점(principal point), 초점거리를 수정함으로써, 기존 리샘플링 방법을 넘어선 뷰포트 조정이 가능하다.

---

### 2-5. 한계

논문 및 관련 연구에서 확인되는 한계:

1. **per-video 최적화**: GaussianVideo는 각 비디오에 대해 개별 최적화를 수행하므로, **새로운 비디오에 대한 즉각적인 일반화(feed-forward inference)**가 불가능하다.

2. **장면 외삽(extrapolation) 불가**: GaussianVideo는 Neural ODE와 계층적 Gaussian Splatting을 결합하였으나, ODE를 장면 모션이 아닌 카메라 궤적 학습에 주로 활용하며, 보간(interpolation) 작업을 위해 설계되었다. 따라서 학습 범위 밖의 미래 시간으로의 외삽(extrapolation)에는 취약하다.

3. 명시적 시변 Gaussian 특징을 가진 기존 접근법들은 동적 장면 모델링에서 뛰어난 성능을 보이지만, 대부분의 3DGS 기반 접근법은 시간 조건부 변형 필드(time-conditioned deformation field)에 의존하기 때문에 보간에서는 뛰어나지만 미래의 미관측 시간으로의 외삽에는 한계를 보인다.

4. **고속 모션 장면**: 매우 빠른 모션이 있는 장면에서의 Gaussian 표현 충분성에 대한 검증이 제한적이다.

---

## 3. 🌐 모델의 일반화 성능 향상 가능성

GaussianVideo는 per-video 최적화(overfitting-to-scene) 패러다임에 속하므로, 일반화 측면에서 다음의 가능성과 한계가 공존한다.

### 3-1. 현재의 일반화 관련 강점

**① 다양한 도메인 적용 가능성**

계층적 학습과 강건한 카메라 모션 모델링의 조합으로 복잡한 동적 장면을 강한 시간적 일관성을 갖추어 포착하며, 고모션 및 저모션 시나리오 모두에서 다양한 비디오 데이터셋에 걸쳐 최첨단 성능을 달성한다.

**② 최소 감독 하의 의미론적 동작 창발(Emergent Semantic Motion)**

Gaussian들은 정적 객체를 표현할 때 효과적으로 정지 상태를 유지하고, 건물이 해당 경로를 따라 이동하는 것처럼 예상되는 의미론적 움직임을 보인다. 이는 외부 감독 신호 없이도 의미론적으로 타당한 모션 표현이 학습됨을 의미한다.

**③ 연속적 표현으로 인한 시공간 일반화**

Gaussian 비디오 표현의 이점 중 하나는 학습된 연속적(continuous) 특성이다. B-spline 및 Neural ODE 기반의 연속 시간 모델링은 이산적(discrete) 표현 대비 관측되지 않은 중간 타임스탬프로의 보간 능력을 제공한다.

### 3-2. 일반화 성능 향상을 위한 향후 방향

| 방향 | 설명 |
|---|---|
| **Feed-forward 예측 모델** | 대규모 비디오 데이터로 사전학습된 인코더를 통해 새로운 비디오에 대해 즉각적으로 Gaussian 파라미터를 예측 |
| **외삽(Extrapolation) 능력 강화** | 타임스탬프 입력에 의존하지 않고 관측된 궤적 이력(history)만을 기반으로 미래 Gaussian 파라미터를 예측하는 Transformer 기반 Latent ODE 모델을 결합하면 시간 의존 함수와 관련된 일반화 문제를 회피할 수 있다. |
| **도메인 적응** | 의료 영상, 위성 영상 등 특수 도메인으로의 전이학습 적용 가능성 |
| **멀티-비디오 공유 Gaussian Prior** | 여러 비디오에서 공유되는 장면 prior를 학습하여 새로운 비디오에서의 수렴 가속화 |

---

## 4. 🔍 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 연도 | 표현 방식 | 시간 모델링 | 카메라 필요 | 특징 |
|---|---|---|---|---|---|
| **NeRF** | 2020 | Implicit MLP | ✗ | ✓ | 정적 장면, 고품질 |
| **NeRV** | 2021 | Implicit NN | 인덱스 기반 | ✗ | 비디오 → 가중치 매핑 |
| **HNeRV** | 2023 | Hybrid NeRV | 인덱스 기반 | ✗ | 콘텐츠 어댑티브 |
| **3DGS** | 2023 | Explicit 3D Gaussian | ✗ | ✓ | 실시간 렌더링 |
| **Spacetime GS** | 2024 | 3D Gaussian | 다항식 | ✓ | 위치와 회전을 다항식으로 모델링하고 색상을 MLP로, 불투명도를 방사 기저 함수(RBF)로 표현 |
| **4DGS** | 2024 | 3D Gaussian | 변형 네트워크 | ✓ | MLP로 파라미터화된 학습된 변형 네트워크를 이용해 시간에 따른 위치·회전·스케일의 오프셋을 예측 |
| **GaussianVideo** | 2025 | 3D Gaussian | B-spline + Neural ODE | **✗** | 계층적 학습, 카메라 파라미터 불필요 |
| **ODE-GS** | 2025 | 3D Gaussian | Latent ODE | ✓ | 3D Gaussian Splatting과 잠재 신경 ODE를 통합하여 학습 중 관측된 시간 범위를 훨씬 넘어 동적 3D 장면을 예측(extrapolate)하는 신규 방법 |

**GaussianVideo의 차별점**:
GaussianVideo는 고재구성 품질과 계산 효율성의 균형을 맞추는 효율적인 계층적 Gaussian splatting 접근법을 통한 설득력 있는 발전을 제시한다. 카메라 모션 모델링, 정교한 모션 표현, 계층적 학습 전략의 원활한 통합이 다양한 다운스트림 응용에 적합하게 만들며, 비디오 표현 및 편집 기술의 미래 연구를 위한 기초를 마련한다.

---

## 5. 🚀 미래 연구에 미치는 영향 및 고려할 점

### 5-1. 미래 연구에 미치는 영향

**① 비디오 압축 및 스트리밍**

이 연구의 함의는 실시간 렌더링 및 동적 장면 분석과 같은 영역으로 확장되며, 높은 충실도와 강건한 비디오 처리 능력이 요구되는 분야에서 실질적 응용 가능성을 제시한다.

**② 비디오 편집 및 생성 기반**

비디오 스타일화 응용에서, 첫 프레임에 스타일을 적용하고 SH 계수만 업데이트하여 추가 감독 없이 비디오 전체에 일관된 스타일을 전파할 수 있다. 프레임 간 시각적 일관성은 구조적 세부 사항을 보존하면서 고품질 스타일 전이를 유지하는 GaussianVideo의 능력을 강조한다.

**③ 후속 연구의 직접적 기반**

GaussianVideo는 후속 연구인 ODE-GS(2025)에서 직접 인용되어, 계층적 Gaussian Splatting과 Neural ODE의 결합 방향성을 제시한 선행 연구로 자리매김하고 있다.

### 5-2. 향후 연구 시 고려할 점

1. **Per-scene 최적화의 일반화 한계 극복**: GaussianVideo는 장면별 최적화 방식으로, 새로운 비디오에 대한 즉각적 추론(feed-forward inference)이 불가능하다. 대규모 사전학습 기반의 feed-forward Gaussian 예측 모델로의 확장이 필요하다.

2. **외삽(Extrapolation) 연구**: 기존 방법들은 관측된 시간 프레임 내에서의 보간에는 뛰어나지만 미래의 미관측 타임스탬프로의 일반화에는 실패하는 경우가 많다. ODE-GS는 Transformer 기반 Latent ODE와 3D Gaussian Splatting을 통합하여 이 문제를 해결한다.

3. **더 효율적인 Gaussian 관리**: 비디오당 400K Gaussian 사용은 메모리 부담을 야기하므로, 적응적 Gaussian 밀도 제어(adaptive densification/pruning) 전략의 개선이 필요하다.

4. **멀티모달 확장**: 오디오-비디오 동기화, 텍스트 조건부 비디오 편집 등 멀티모달 신호와 결합한 확장 연구가 유망하다.

5. **평가 지표 다양화**: DL3DV와 DAVIS Tap-Vid 두 개 벤치마크에서 PSNR, SSIM, LPIPS로 검증되었으나, 더 다양한 도메인(의료, 자율주행 등)과 지표(FID, FVD 등 생성 품질 지표)에서의 검증이 필요하다.

6. **실시간 학습(Online Learning)**: 현재의 오프라인 최적화 방식에서 스트리밍 비디오에 대한 온라인 학습으로의 확장이 요구된다.

---

## 📚 참고 문헌 및 출처

| # | 제목 / 출처 | URL |
|---|---|---|
| 1 | **GaussianVideo: Efficient Video Representation via Hierarchical Gaussian Splatting** (arXiv:2501.04782) | https://arxiv.org/abs/2501.04782 |
| 2 | GaussianVideo HTML Full Paper (arXiv HTML) | https://arxiv.org/html/2501.04782v1 |
| 3 | GaussianVideo Official Project Page | https://cyberiada.github.io/GaussianVideo/ |
| 4 | GaussianVideo ICCV 2025 Paper (OpenAccess) | https://openaccess.thecvf.com/content/ICCV2025/papers/Bond_GaussianVideo... |
| 5 | GaussianVideo ResearchGate | https://www.researchgate.net/publication/387872940 |
| 6 | GaussianVideo Hugging Face Paper Page | https://huggingface.co/papers/2501.04782 |
| 7 | Moonlight Literature Review: GaussianVideo | https://www.themoonlight.io/en/review/gaussianvideo-... |
| 8 | **ODE-GS: Latent ODEs for Dynamic Scene Extrapolation with 3DGS** (arXiv:2506.05480) | https://arxiv.org/html/2506.05480v1 |
| 9 | 3DGS and Beyond Docs (GitHub, yangjiheng) | https://github.com/yangjiheng/3DGS_and_Beyond_Docs |
| 10 | AlphaXiv Overview: GaussianVideo | https://www.alphaxiv.org/overview/2501.04782v1 |

> ⚠️ **정확도 안내**: 본 답변은 공개된 arXiv 논문 HTML 및 초록, 관련 인용 연구를 기반으로 작성되었습니다. 세부 수식(특히 손실 함수 가중치, 정확한 하이퍼파라미터)은 논문 전문 PDF에서 직접 확인을 권장드립니다.
