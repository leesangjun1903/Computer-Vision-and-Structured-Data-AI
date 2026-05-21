
# RayZer: A Self-supervised Large View Synthesis Model

> **논문 정보**
> - **제목**: RayZer: A Self-supervised Large View Synthesis Model
> - **저자**: Hanwen Jiang, Hao Tan, Peng Wang, Haian Jin, Yue Zhao, Sai Bi, Kai Zhang, Fujun Luan, Kalyan Sunkavalli, Qixing Huang, Georgios Pavlakos
> - **arXiv**: [2505.00702](https://arxiv.org/abs/2505.00702) (2025년 5월 1일)
> - **학회**: ICCV 2025 **(Best Student Paper Honorable Mention)**
> - **코드**: [GitHub - hwjiang1510/RayZer](https://github.com/hwjiang1510/RayZer)
> - **프로젝트 페이지**: https://hwjiang1510.github.io/RayZer/

---

## 1. 핵심 주장 및 주요 기여 요약

RayZer는 카메라 포즈나 씬 기하(3D 지오메트리) 등 어떠한 3D 지도(Supervision)도 없이 학습되면서도 3D 인식 능력(emerging 3D awareness)이 나타나는 자기 지도(self-supervised) 멀티뷰 3D Vision 모델입니다.

구체적으로, RayZer는 포즈 및 캘리브레이션 정보가 없는(unposed & uncalibrated) 이미지들을 입력으로 받아, 카메라 파라미터를 복원하고, 씬 표현을 재구성하며, 새로운 시점(novel view)을 합성합니다.

RayZer는 포즈 어노테이션에 의존하는 "oracle" 방법들과 동등하거나 오히려 우월한 novel view synthesis 성능을 보입니다.

**핵심 기여 3가지:**

| 기여 | 내용 |
|------|------|
| ① Self-supervised Framework | 카메라와 씬 표현을 분리(disentangle)하는 3D-aware auto-encoding |
| ② Ray-구조 기반 Transformer | 3D prior를 ray 구조 하나만으로 제한하여 카메라·픽셀·씬을 동시에 연결 |
| ③ 어노테이션 불필요 | 학습 및 추론 시 어떠한 3D 레이블도 필요 없음 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

기존 대규모 Novel View Synthesis(NVS) 모델들(GS-LRM, LVSM 등)은 학습과 추론 모두에서 COLMAP 등으로 얻은 카메라 포즈 어노테이션을 필요로 합니다. 이는 두 가지 근본적 문제를 낳습니다.

1. **데이터 병목**: 포즈 레이블이 있는 데이터셋에만 학습이 제한됨
2. **COLMAP의 불완전성**: GS-LRM과 LVSM은 추론 시 일부 케이스에서 지속적으로 실패하는데, 흥미롭게도 이러한 케이스들은 COLMAP이 보통 실패하는 시나리오들입니다.

기존 연구들은 3D 표현과 물리 기반 렌더링 방정식을 잠재 표현(latent representations)과 학습된 렌더링 함수로 대체하여 성능·확장성을 높였지만, 지도 학습(supervised training) 및/또는 추론 시 정확한 카메라 어노테이션을 여전히 필요로 합니다.

---

### 2-2. 제안하는 방법 (Self-supervised Framework)

#### (A) 핵심 아이디어: 3D-aware Auto-encoding

RayZer는 자기 자신이 예측한 카메라 포즈를 학습에 사용하므로, 이 자기 지도 태스크는 3D-aware 이미지 오토인코딩으로 해석될 수 있습니다. 처음에 RayZer는 입력 이미지를 카메라 파라미터와 씬 표현으로 분리(reconstruction)하고, 이후 이 예측된 표현들을 다시 이미지로 합성(rendering)합니다.

#### (B) 이미지 분할 전략

RayZer는 이미지를 두 집합 $\mathcal{I}\_\mathcal{A}$와 $\mathcal{I}\_\mathcal{B}$로 분리합니다. $\mathcal{I}\_\mathcal{A}$로부터 씬 표현을 예측하고, $\mathcal{I}\_\mathcal{B}$의 예측된 카메라를 사용해 씬을 렌더링하여 $\hat{\mathcal{I}}\_\mathcal{B}$를 생성합니다. 따라서 RayZer는 오직 $\mathcal{I}\_\mathcal{B}$와 $\hat{\mathcal{I}}_\mathcal{B}$ 사이의 photometric loss만으로 학습되며, 카메라와 기하에 대한 3D 지도가 전혀 필요하지 않습니다.

모든 이미지를 두 파트로 나누어 하나는 씬 표현 예측(입력 뷰)에, 다른 하나는 photometric self-supervision(타겟 뷰)에 활용합니다. 이는 두 번째 집합의 예측 포즈로 첫 번째 집합의 씬 표현을 렌더링함으로써, 3D-aware하지 않은 trivial solution을 방지합니다.

#### (C) 학습 손실 함수

학습 목적 함수는 photometric loss와 perceptual loss의 합으로 구성됩니다:

$$\mathcal{L} = \frac{1}{K_\mathcal{B}} \sum_{I \in \mathcal{I}_\mathcal{B}} \left( \| I - \hat{I} \|_1 + \lambda \cdot \mathcal{L}_\text{perceptual}(I, \hat{I}) \right)$$

여기서:
- $K_\mathcal{B} = |\mathcal{I}_\mathcal{B}|$: 타겟 뷰 집합의 이미지 수
- $I \in \mathcal{I}_\mathcal{B}$: 실제 타겟 이미지
- $\hat{I}$: 모델이 렌더링한 예측 이미지
- $\lambda$: perceptual loss의 가중치

두 집합은 학습 중 무작위로 샘플링됩니다.

#### (D) Ray 기반 카메라 표현 (Plücker Coordinates)

RayZer는 ray 구조를 유일한 3D prior로 사용하는 transformer 기반 모델을 설계하여, 카메라·픽셀·씬을 동시에 연결합니다.

Ray는 Plücker 좌표계로 표현됩니다:

$$\mathbf{r} = (\mathbf{d}, \mathbf{m}) \in \mathbb{R}^6, \quad \mathbf{m} = \mathbf{o} \times \mathbf{d}$$

여기서:
- $\mathbf{d} \in \mathbb{R}^3$: 정규화된 ray 방향 벡터
- $\mathbf{o} \in \mathbb{R}^3$: 카메라 원점 (world 좌표계)
- $\mathbf{m} = \mathbf{o} \times \mathbf{d}$: moment 벡터

각 픽셀 $(u, v)$에 대해:

$$\mathbf{d}_{u,v} = \frac{\mathbf{K}^{-1} [u, v, 1]^\top}{\| \mathbf{K}^{-1} [u, v, 1]^\top \|}, \quad \mathbf{o} = \mathbf{R}^\top (-\mathbf{t})$$

이를 통해 모델은 이미지 패치마다 하나의 ray embedding을 생성하고, 이것이 transformer의 위치 인코딩 역할을 합니다.

---

### 2-3. 모델 구조

RayZer는 self-supervised 학습을 위해 오직 transformer만으로 구성되며, 별도의 3D 표현, 수작업 렌더링 방정식, 또는 3D-특화 아키텍처를 사용하지 않습니다.

RayZer는 포즈 없고 캘리브레이션되지 않은 입력 이미지에서 카메라 파라미터와 씬 표현 모두를 복원합니다. RayZer의 핵심 설계 요소는 카메라와 씬 표현의 **cascade prediction** 구조입니다.

이는 노이즈가 있는 카메라조차도 더 좋은 씬 재구성을 위한 강력한 조건이 될 수 있다는 사실에서 동기를 얻은 것으로, 전통적인 structure-from-motion 방식과 유사합니다.

전체 구조를 정리하면:

```
입력: {I_1, ..., I_N} (unposed & uncalibrated)
        ↓
[Camera Estimator] → 예측 카메라 파라미터 {R̂_i, t̂_i, K̂_i}
        ↓
[Ray Embedding] → Plücker 좌표 r̂ ∈ ℝ^6 (per pixel/patch)
        ↓
[Scene Reconstructor (Transformer)] → 씬 잠재 표현 F_scene
        ↓
[Renderer (Transformer-based)] → 렌더링된 타겟 뷰 Î_B
        ↓
Photometric Loss: L(I_B, Î_B)
```

PF-LRM(Pose-free LRM) 구조에서는 24개의 transformer 레이어를 사용합니다.

효율성을 위해 패치 크기 16을 사용하며, 이는 원래 LVSM 논문의 패치 크기 8과 다릅니다.

추론 시 RayZer는 카메라 및 씬 표현을 feed-forward 방식으로 예측하며, 씬별 최적화(per-scene optimization)를 필요로 하지 않습니다.

---

### 2-4. 성능 향상

놀랍게도, 학습 중 어떠한 3D 레이블(카메라 포즈 어노테이션 등)도 없이 RayZer는 최고의 "oracle" 모델인 LVSM과 비슷한 성능을 달성합니다. 실제로 RayZer는 DL3DV와 RealEstate10K에서 LVSM을 능가하지만, Objaverse에서는 약간 낮은 성능을 보입니다.

RayZer는 학습과 테스트 모두에서 COLMAP 포즈 어노테이션을 사용하는 "oracle" 방법인 GS-LRM 및 LVSM과 비교됩니다. 자기 지도 방식의 RayZer는 어떠한 포즈 어노테이션도 사용하지 않으며, COLMAP이 통상적으로 어려움을 겪는 케이스(예: 유리, 흰 벽)에서 oracle 방법들을 능가합니다.

| 데이터셋 | RayZer vs LVSM | RayZer vs GS-LRM |
|---------|---------------|-----------------|
| DL3DV | **RayZer 우세** | **RayZer 우세** |
| RealEstate10K | **RayZer 우세** | **RayZer 우세** |
| Objaverse | 약간 열세 | 비교 중 |

---

### 2-5. 한계점

모델은 이미지 인덱스 임베딩을 사용하기 때문에 **뷰의 수(number of views)에 민감**할 수 있습니다.

기존 연구들은 여전히 근본적인 가정, 즉 **3D 씬이 정적(static)** 이라는 가정에 의존합니다. 이 모델들은 학습과 추론 모두에서 정적 입력을 필요로 하지만, 실세계 3D 환경은 본질적으로 동적입니다. 결과적으로 기존 방법들은 RealEstate10K 같은 정적 씬 데이터셋에 의존하게 되어, 확장성이 제한되고 풍부한 동적 콘텐츠를 담은 실세계 영상을 완전히 활용하지 못합니다.

추가적으로 아래 한계도 있습니다:
- Objaverse(합성 오브젝트 데이터셋)에서는 oracle 모델 대비 약간 성능이 낮음
- 학습 시 8 GPU × 4 노드 (총 256 배치)라는 대규모 컴퓨팅 자원 필요

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 포즈 어노테이션 없는 학습 → 대규모 비레이블 데이터 활용 가능

RayZer는 카메라 포즈 레이블 등 어떠한 어노테이션도 없는 비레이블 데이터로 학습됩니다. 이는 곧 인터넷상의 수십억 개의 비디오 및 이미지를 학습 데이터로 활용할 수 있음을 의미하며, 데이터 규모 측면에서 기존 지도 학습 방식 대비 획기적인 확장 가능성을 열어줍니다.

### 3-2. COLMAP 노이즈 문제에 강한 일반화

RayZer(자기 지도 방식)는 DL3DV와 RealEstate에서 더 우수한 novel view synthesis 성능을 보이며, 이 결과는 RayZer의 강력한 능력을 보여줄 뿐만 아니라 COLMAP 어노테이션이 완벽하지 않음을 시사합니다.

### 3-3. 다양한 씬 유형에서의 강건성 (도메인 일반화)

RayZer는 COLMAP이 일반적으로 어려워하는 케이스, 예를 들어 유리(glasses)나 흰 벽(white walls)에서 oracle 방법들을 능가합니다. 이는 RayZer가 특정 씬 구조에 overfitting되지 않고 더 일반적인 3D 인식을 학습함을 시사합니다.

### 3-4. E-RayZer를 통한 일반화 확장

E-RayZer는 포즈 추정에서 RayZer를 크게 능가하고, VGGT 같은 완전 지도 학습 재구성 모델과 동등하거나 때로는 그 이상의 성능을 보입니다. 또한 E-RayZer의 학습된 표현은 DINOv3, CroCo v2, VideoMAE V2, RayZer 등 선도적인 시각적 사전 학습 모델들을 3D 다운스트림 태스크에서 능가하며, 공간적 시각 사전 학습의 유망한 패러다임으로 자리매김합니다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

| 모델 | 연도 | 포즈 필요 | 학습 방식 | 주요 특징 |
|------|------|----------|----------|----------|
| NeRF | 2020 | ✅ (학습+추론) | Per-scene 최적화 | 고품질 NVS, 느린 최적화 |
| 3DGS | 2023 | ✅ (학습+추론) | Per-scene 최적화 | 실시간 렌더링 |
| GS-LRM | 2024 | ✅ (학습+추론) | 지도 학습 | 3DGS 기반 피드포워드 |
| LVSM | 2024/2025 | ✅ (학습+추론) | 지도 학습 | 순수 Transformer, 고품질 |
| **RayZer** | **2025** | **❌ 불필요** | **자기 지도** | **Ray-구조 기반, 비레이블 학습** |
| WildRayZer | 2025 | ❌ 불필요 | 자기 지도 | 동적 씬으로 확장 |
| E-RayZer | 2025 | ❌ 불필요 | 자기 지도 | 공간 시각 사전 학습 |
| Efficient-LVSM | 2026 | ✅ | 지도 학습 | Decoupled Attention, 4.4× 빠른 추론 |

분야는 고전적인 포토그래메트리 시스템에서 NeRF, 3DGS 같은 씬별 최적화 신경 표현으로 발전해 왔으며, 고품질 재구성을 달성하지만 각 새 씬마다 밀집 입력과 비용이 큰 최적화가 필요합니다.

그러나 LRM들은 여전히 NeRF, 메시, 3DGS 등의 표현 수준 편향과 각각의 렌더링 방정식에 의존하여 일반화 및 확장성이 제한됩니다. LVSM 연구는 3D 귀납적 편향을 최소화하고 완전 데이터 기반 접근으로 novel view synthesis의 한계를 넓히는 것을 목표로 합니다.

WildRayZer는 동적 환경에서 희소하고 포즈 없는 뷰들로부터 novel view synthesis를 위한 자기 지도 학습 프레임워크를 제안하며, NeRF와 3DGS를 실세계 환경에 적응시킨 선행 연구들과 유사하게 RayZer를 확장합니다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려 사항

### 5-1. 연구에 미치는 영향

**① 자기 지도 3D Vision의 패러다임 전환**

RayZer는 3D Vision 모델이 지도 학습에서 벗어날 수 있는 가능성을 제시합니다. 이는 NLP 분야에서 BERT, GPT 등이 자기 지도 학습으로 패러다임을 바꾼 것처럼, 3D Vision에서도 유사한 혁신의 출발점이 될 수 있습니다.

**② 비레이블 데이터 활용 가능성 극대화**

자기 지도 사전 학습은 프론티어 모델들의 기반을 형성하며, 방대한 양의 비레이블 데이터에서 의미 있는 표현을 학습할 수 있게 합니다. RayZer는 이를 3D 공간 이해 영역으로 확장합니다.

**③ 공간 시각 사전 학습 연구 활성화**

E-RayZer의 학습된 표현은 DINOv3, CroCo v2, VideoMAE V2 등 선도적 시각 사전 학습 모델들을 3D 다운스트림 태스크에서 능가하며, 공간적 시각 사전 학습의 유망한 패러다임으로 자리매김합니다.

### 5-2. 앞으로 연구 시 고려할 점

**① 동적 씬(Dynamic Scene) 확장**

기존 연구들은 3D 씬이 정적이라는 근본적 가정에 의존하고, 모델들은 학습과 추론 모두에서 정적 입력을 필요로 합니다. 결과적으로 기존 방법들은 RealEstate10K 같은 정적 씬 데이터셋에 의존하여 확장성이 제한되고, 자연스러운 동적 콘텐츠를 가진 실세계 비디오를 충분히 활용하지 못합니다. 동적 씬 처리 능력이 필수적으로 연구되어야 합니다.

**② 뷰 수(Number of Views) 일반화**

모델이 이미지 인덱스 임베딩을 사용하기 때문에 뷰의 수에 민감할 수 있습니다. 임의의 뷰 수에 강건한 아키텍처 개발이 중요한 연구 과제입니다.

**③ 계산 효율성 개선**

LVSM과 같은 Transformer 기반 novel view synthesis 방법들은 최적화되지 않은 full self-attention으로 인해 입력 뷰의 수에 대해 이차(quadratic) 복잡도를 가집니다. RayZer 역시 동일 구조를 공유하므로 효율적인 어텐션 메커니즘 도입이 필요합니다.

**④ 대규모 인터넷 데이터를 활용한 Scaling Law 탐구**

포즈 어노테이션이 필요 없다는 특성상, YouTube 영상, 소셜 미디어 사진 등 대규모 비레이블 멀티뷰 데이터로의 확장이 가능합니다. 데이터 규모와 모델 성능 간의 Scaling Law를 정량적으로 탐구하는 연구가 요구됩니다.

**⑤ 다운스트림 태스크로의 전이 학습**

E-RayZer는 포즈 추정에서 RayZer를 크게 능가하고, 완전 지도 학습 재구성 모델과 동등하거나 우월한 성능을 보입니다. 자기 지도로 학습된 3D 표현을 로보틱스, AR/VR, 자율주행 등 다양한 다운스트림 태스크로 효과적으로 전이하는 방법론 연구가 중요합니다.

---

## 📚 참고 자료 (출처)

| # | 제목 / 출처 | URL |
|---|------------|-----|
| 1 | **RayZer: A Self-supervised Large View Synthesis Model** (arXiv:2505.00702) | https://arxiv.org/abs/2505.00702 |
| 2 | **RayZer arXiv HTML 전문** | https://arxiv.org/html/2505.00702v1 |
| 3 | **RayZer arXiv PDF** | https://arxiv.org/pdf/2505.00702 |
| 4 | **RayZer 공식 GitHub** (hwjiang1510/RayZer) | https://github.com/hwjiang1510/RayZer |
| 5 | **RayZer 프로젝트 페이지** | https://hwjiang1510.github.io/RayZer/ |
| 6 | **RayZer ICCV 2025 Open Access Paper** | https://openaccess.thecvf.com/content/ICCV2025/papers/Jiang_RayZer_A_Self-supervised_Large_View_Synthesis_Model_ICCV_2025_paper.pdf |
| 7 | **NSF Public Access Repository (RayZer)** | https://par.nsf.gov/biblio/10613245 |
| 8 | **E-RayZer: Self-supervised 3D Reconstruction as Spatial Visual Pre-training** (arXiv:2512.10950) | https://arxiv.org/html/2512.10950 |
| 9 | **WildRayZer: Self-supervised Large View Synthesis in Dynamic Environments** | https://wild-rayzer.cs.virginia.edu/static/pdf/wild-rayzer_arxiv.pdf |
| 10 | **LVSM: A Large View Synthesis Model** (ICLR 2025) | https://proceedings.iclr.cc/paper_files/paper/2025/file/9676c5283df26cabca412ca66b164a7d-Paper-Conference.pdf |
| 11 | **Efficient-LVSM** (arXiv:2602.06478) | https://arxiv.org/html/2602.06478v1 |
| 12 | **Cameras as Rays: Pose Estimation via Ray Diffusion** (arXiv:2402.14817) | https://arxiv.org/pdf/2402.14817 |
| 13 | **deeplearn.org RayZer 요약** | https://deeplearn.org/arxiv/600683/rayzer:-a-self-supervised-large-view-synthesis-model |

> ⚠️ **정확도 고지**: 본 답변은 공개된 arXiv 논문 전문, 공식 GitHub, 프로젝트 페이지 및 ICCV 2025 Open Access 논문을 기반으로 작성되었습니다. 수식의 세부 구현(예: PF-LRM 내 정확한 어텐션 레이어 수, 손실 함수의 정확한 계수 값 등)은 논문 전문 내 상세 기술을 직접 확인하시길 권장드립니다.
