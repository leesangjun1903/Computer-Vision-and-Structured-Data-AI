# COLMAP-Free 3D Gaussian Splatting (CF-3DGS)

---

## 1. 핵심 주장 및 주요 기여 요약

CF-3DGS는 알려진 카메라 파라미터 없이 새로운 시점 합성(Novel View Synthesis)을 수행하는 방법으로, 포즈 추정의 강건성과 뷰 합성의 품질 모두에서 기존 최신 방법을 능가한다.

**핵심 주장:**
이 논문은 명시적 기하학적 표현(explicit geometric representation)과 입력 비디오 스트림의 시간적 연속성(continuity)을 모두 활용하여, SfM 전처리 없이 새로운 시점 합성을 수행한다.

**주요 기여:**
1. 비디오의 시간적 연속성과 명시적 포인트 클라우드 표현이라는 두 가지 핵심 요소를 활용하며, 모든 프레임을 한 번에 최적화하는 대신 카메라가 이동하면서 한 프레임씩 "성장(growing)"하는 방식으로 장면의 3D Gaussians을 구축한다. 이 과정에서 각 프레임에 대한 **Local 3DGS**를 추출하고, 전체 장면의 **Global 3DGS**를 유지한다.
2. Nope-NeRF(25~30시간)에 비해 약 1.5시간의 훈련 시간으로 우수한 결과를 달성하는 효율성을 보여준다.
3. 이 방법은 큰 모션 변화(large motion changes) 환경에서 뷰 합성 및 카메라 포즈 추정 모두에서 이전 접근법 대비 크게 개선된 성능을 보인다.

---

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

NeRF로 대표되는 photo-realistic 장면 복원 및 뷰 합성 분야는 크게 발전했으나, NeRF를 훈련하기 위한 핵심 초기화 단계로 각 입력 이미지의 카메라 포즈를 사전에 준비해야 한다. 이는 보통 SfM 라이브러리인 COLMAP을 실행하여 달성하는데, 이 전처리는 시간이 많이 소요될 뿐만 아니라 특징 추출 오류에 대한 민감성과 텍스처가 없거나 반복적인 영역 처리의 어려움으로 인해 실패할 수 있다.

NeRF의 암묵적(implicit) 표현은 3D 구조와 카메라 포즈를 동시에 최적화하는 데 추가적인 어려움을 초래한다.

### 2.2 제안 방법 및 수식

CF-3DGS는 크게 **세 단계**로 구성된다:

#### (a) 단안 깊이 기반 초기화 (Initialization from a Single View)

프레임 $I_t$가 주어지면, 사전 학습된 단안 깊이 추정 네트워크 DPT를 사용하여 단안 깊이 $D_t$를 생성한다. 이 깊이 맵을 카메라 내부 파라미터를 이용해 3D 포인트 클라우드로 리프팅(lifting)한 후, 3D Gaussian 집합 $\mathcal{G}_t$를 초기화한다.

초기화 후, photometric loss를 최소화하여 Gaussian의 속성을 학습한다:

$$\mathcal{L} = (1 - \lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}}$$

여기서 $\lambda = 0.2$로 설정되며, 이 단계는 매우 경량화되어 약 5초 내에 수행된다.

#### (b) 3D Gaussian 변환을 통한 포즈 추정 (Pose Estimation by 3D Gaussian Transformation)

상대적 카메라 포즈를 추정하기 위해, 사전 학습된 Local 3DGS를 변환 $\mathcal{T}_t$를 통해 프레임 $t+1$로 변환한다:

$$\mathcal{G}_{t+1} = \mathcal{T}_t \odot \mathcal{G}_t$$

여기서 변환 $\mathcal{T}\_t$는 렌더링된 이미지와 다음 프레임 $I_{t+1}$ 사이의 photometric loss를 최소화하여 최적화된다:

$$\mathcal{T}_t^* = \arg\min_{\mathcal{T}_t} \mathcal{L}\bigl(\text{Render}(\mathcal{T}_t \odot \mathcal{G}_t),\; I_{t+1}\bigr)$$

카메라 포즈는 쿼터니언 회전(quaternion rotation) $q \in so(3)$과 병진 벡터(translation vector) $t \in \mathbb{R}^3$로 표현되어 최적화된다.

변환 학습 시에는 사전 학습된 Local 3DGS의 모든 속성(위치, SH 계수, 불투명도, 스케일, 회전)을 고정(freeze)하고, 포즈 파라미터만 학습한다. 이 최적화는 300 스텝으로 수행된다.

#### (c) 점진적 성장을 통한 전역 3DGS (Progressive Growing of Global 3DGS)

전체 파이프라인은 이미지 시퀀스를 입력으로 받아 장면을 나타내는 3D Gaussian 집합을 학습하며 카메라 포즈를 공동 추정한다. Local 3DGS가 인접 프레임 간의 상대 포즈를 추정한 후, Global 3DGS가 점진적으로 Gaussian 집합을 성장시켜 장면을 모델링한다.

$t$번째 프레임 $I_t$에서 시작하여 카메라 포즈를 직교(orthogonal)로 설정한 3D Gaussian 점 집합을 초기화한다. Local 3DGS를 활용하여 프레임 $I_t$와 $I_{t+1}$ 사이의 상대 카메라 포즈를 추정하고, Global 3DGS는 추정된 상대 포즈와 두 관찰 프레임을 입력으로 사용하여 $N$번 반복 동안 3D Gaussian 점 집합을 업데이트한다. 다음 프레임 $I_{t+2}$가 사용 가능해지면 이 과정이 반복된다.

**Densification 전략:**
새 프레임이 도착할 때 "미재구성(under-reconstruction)" Gaussian을 densify한다. 뷰 공간 위치 기울기의 평균 크기로 densification 후보를 결정하며, 새 프레임 추가 속도에 맞춰 $N$ 스텝마다 Global 3DGS를 densify한다. 또한 훈련 중간에 densification을 중단하지 않고 입력 시퀀스의 끝까지 3D Gaussian 점을 계속 성장시킨다.

새 프레임 추가 스텝 수를 포인트 densification 간격과 동일하게 설정하여 전체 장면의 점진적 성장을 달성하고, 훈련 과정이 끝날 때까지 불투명도를 계속 리셋하여 관찰된 프레임에서 구축된 Gaussian 모델에 새 프레임을 통합할 수 있게 한다.

### 2.3 모델 구조 개요

```
입력: 비디오 프레임 시퀀스 {I₁, I₂, ..., I_T}
                    │
    ┌───────────────┴───────────────┐
    │                               │
 [Local 3DGS]                  [Global 3DGS]
    │                               │
 • 단안 깊이(DPT)로              • 전체 장면의 3D Gaussian
   단일 뷰 초기화                   집합을 점진적으로 성장
 • Gaussian 변환 최적화로        • 모든 관찰된 프레임으로
   인접 프레임 간 상대             공동 최적화
   포즈 추정                     • 새 프레임 도착 시 
    │                              densification 수행
    └───────────────┬───────────────┘
                    │
    출력: 3D 장면 표현 + 카메라 포즈 궤적
```

### 2.4 성능

CF-3DGS는 Tanks and Temples 데이터셋에서 Nope-NeRF, BARF, NeRFmm, SC-NeRF 등 기존 포즈 미지 방법들을 novel view synthesis에서 크게 능가하며, 카메라 포즈 추정에서도 RPE_t 0.041, RPE_r 0.069, ATE 0.004의 경쟁력 있는 성능을 보여준다.

사진과 같은 사실적인 새로운 뷰 이미지를 효율적으로 합성하면서, 훈련 시간 단축 및 실시간 렌더링 능력을 제공하고 COLMAP 처리 의존성을 제거한다.

**평가 메트릭:**
PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index Measure), LPIPS (Learned Perceptual Image Patch Similarity)를 사용하여 novel view synthesis 품질을 측정한다.

### 2.5 한계

제안된 방법은 카메라 포즈와 3DGS를 순차적(sequential)으로 공동 최적화하므로, 주로 비디오 스트림이나 정렬된 이미지 컬렉션에만 적용이 제한된다. 비정렬 이미지 컬렉션에 대한 확장은 향후 흥미로운 연구 방향이다.

또한, 인접 카메라 뷰 간에 극적인 회전 및 병진이 특징인 복잡한 카메라 궤적이 있는 장면에서는 카메라 포즈 추정이 저하되고 카메라 포즈와 3D-GS의 공동 최적화에서 local minima에 빠질 수 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

CF-3DGS의 일반화 성능과 관련하여 다음과 같은 핵심 포인트를 분석할 수 있다:

### 3.1 단안 깊이 추정기의 영향

Local 3DGS 훈련 시 사전 학습된 단안 깊이 추정기(DPT, ZoeDepth)를 통해 입력 이미지의 단안 깊이 맵을 얻고, 카메라 내부 파라미터로 3D로 리프팅한다. 고해상도 입력 이미지에서 발생할 수 있는 대량의 포인트 클라우드를 다운샘플링한 후 Local 3DGS를 초기화하고 photometric loss로 500회 반복 최적화한다.

서로 다른 단안 깊이 추정 알고리즘에 대한 ablation study가 수행되었으며, 이는 깊이 추정기의 일반화 성능이 CF-3DGS의 전체 성능에 직접적으로 영향을 미침을 시사한다. 더 강력하고 범용적인 깊이 추정기를 사용하면 CF-3DGS의 일반화 성능을 향상시킬 수 있다.

### 3.2 카메라 내부 파라미터의 민감성

모든 장면의 FoV를 79°로 설정하고 주점을 이미지 중심으로 설정하는 휴리스틱 카메라 내부 파라미터를 사용했을 때, novel view synthesis와 카메라 포즈 추정 성능이 약간 저하되는 것이 확인되었다. 이는 내부 파라미터도 중요하며 카메라 외부 파라미터와 함께 추가로 최적화될 수 있음을 보여준다.

### 3.3 일반화를 위한 핵심 방향

| 측면 | 현재 한계 | 개선 가능성 |
|------|-----------|-------------|
| 입력 형태 | 정렬된 비디오만 가능 | 비정렬 이미지 컬렉션으로 확장 |
| 깊이 추정기 | DPT 등 특정 모델에 의존 | Foundation model 기반 범용 깊이 추정기 통합 |
| 카메라 내부 파라미터 | 알려진 내부 파라미터 필요 | 내부 파라미터 공동 최적화 |
| 장면 복잡도 | 극심한 카메라 모션에 취약 | 글로벌 기하 제약 조건 추가 |
| 동적 장면 | 정적 장면만 지원 | 4D Gaussian Splatting으로 확장 |

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 연구에 미치는 영향

CF-3DGS는 3DGS 기반 연구에서 **COLMAP 의존성 제거**라는 새로운 패러다임을 개척하였다. 이 논문의 영향으로 다수의 후속 연구가 등장하였다:

- **TrackGS** (2025): 기존 COLMAP-free 접근법이 복잡한 시나리오에서 실패하는 local 제약에 의존하는 문제를 해결하기 위해, 특징 트랙(feature tracks)을 활용한 글로벌 기하학적 제약을 도입하여 카메라 파라미터와 3D Gaussians을 동시 최적화한다.

- **PCR-GS** (2025): 인접 카메라 뷰 간 급격한 회전·병진 문제를 해결하기 위해, DINO 특징을 사용한 feature reprojection regularization 등 카메라 포즈 co-regularization 기법을 제안한다.

- **ZeroGS**: 수백 개의 비정렬·비정리 이미지에서 3DGS를 훈련하여 기존 pose-free 방법보다 정확한 카메라 포즈를 복원하고, COLMAP 포즈를 사용한 3DGS보다도 높은 품질의 이미지를 렌더링한다.

- **HT-3DGS**: 비디오 입력에 대한 SfM-Free 3DGS 방법으로, 장면 영역별로 최적화된 여러 3D Gaussian 표현을 훈련한 후 단일 통합 모델로 병합하는 계층적 훈련 전략을 도입한다.

### 4.2 향후 연구 시 고려할 점

1. **글로벌 일관성(Global Consistency) 확보**: 기존 COLMAP-free 접근법이 의존하는 local 제약은 복잡한 시나리오에서 실패한다. 글로벌 기하학적 제약(global geometric constraints)을 확립하여 카메라 파라미터와 3D Gaussians의 동시 최적화를 가능하게 하는 방향이 중요하다.

2. **비정렬 이미지 컬렉션 지원**: CF-3DGS의 순차적 처리 제약을 넘어, 정렬되지 않은 이미지 집합에서도 동작하는 방법이 필요하다.

3. **확장 가능한 장면 표현**: 대규모 장면이나 동적 장면에 대한 확장이 요구되며, 4D Gaussian Splatting과 같은 동적 장면의 holistic 표현이 실시간 렌더링과 함께 연구되고 있다.

4. **카메라 내부 파라미터의 공동 최적화**: 미분 가능한(differentiable) 카메라 내부 파라미터 최적화 공식의 개발이 필요하다.

5. **Foundation Model과의 통합**: DUSt3R, MASt3R 등 대규모 3D 사전학습 모델과의 결합을 통한 일반화 성능 향상이 유망한 방향이다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 연도 | 표현 방식 | COLMAP 필요 | 핵심 특징 |
|------|------|-----------|-------------|-----------|
| **NeRF** (Mildenhall et al.) | 2020 | Implicit (MLP) | ✅ | 볼류메트릭 렌더링 기반 NVS의 시작 |
| **NeRFmm** (Wang et al.) | 2021 | Implicit | ❌ | 카메라 포즈 임베딩 공동 최적화 (forward-facing 제한) |
| **BARF** (Lin et al.) | 2021 | Implicit | ❌ | coarse-to-fine 포즈 최적화, 기울기 불일치 완화 |
| **NoPe-NeRF** (Bian et al.) | 2023 | Implicit | ❌ | 단안 깊이 priors로 포즈 제약, 25~30시간 훈련 |
| **3DGS** (Kerbl et al.) | 2023 | Explicit (Point cloud) | ✅ | 실시간 래스터화, 명시적 포인트 클라우드 표현 |
| **CF-3DGS** (Fu et al.) | 2024 | Explicit | ❌ | Local/Global 3DGS 순차적 성장, ~1.5시간 훈련 |
| **InstantSplat** (Fan et al.) | 2024 | Explicit | ❌ | dense stereo 모델 통합, 포인트 기반 end-to-end |
| **TrackGS** (Shi et al.) | 2025 | Explicit | ❌ | 글로벌 특징 트랙 제약, 2D/3D track loss |
| **PCR-GS** (Wei et al.) | 2025 | Explicit | ❌ | DINO 기반 포즈 co-regularization |
| **ZeroGS** | 2024+ | Explicit | ❌ | 비정렬 이미지에서 학습, COLMAP 포즈보다 우수 |

### 핵심 비교 분석

**NeRF 계열 vs. 3DGS 계열:**
3D Gaussian Splatting의 도래는 NeRF의 볼류메트릭 렌더링을 포인트 클라우드로 확장하였으며, 원래 사전 계산된 카메라로 제안되었지만 SfM 전처리 없이 뷰 합성을 수행할 수 있는 새로운 기회를 제공한다. 명시적 표현의 장점은 기하 구조의 직접적 조작이 가능하여 포즈 최적화에 유리하다는 것이다.

**CF-3DGS vs. TrackGS:**
CF-3DGS와 같은 기존 COLMAP-free 접근법은 local 제약에 의존하여 복잡한 시나리오에서 실패하는 반면, TrackGS는 특징 트랙을 활용하여 글로벌 기하학적 제약을 확립한다. 도전적인 실제 및 합성 데이터셋에서 이전 방법보다 훨씬 낮은 포즈 오류와 우수한 렌더링 품질을 달성한다.

**CF-3DGS vs. PCR-GS:**
CF-3DGS가 인접 뷰 간 급격한 회전·병진 시 카메라 포즈 추정이 저하되고 local minima에 빠지는 문제를 PCR-GS는 카메라 포즈 co-regularization을 통해 해결하며, DINO 특징 기반 feature reprojection regularization으로 의미론적 정보를 정렬한다.

---

## 참고자료

1. **Fu, Y., Liu, S., Kulkarni, A., Kautz, J., Efros, A. A., & Wang, X.** (2024). "COLMAP-Free 3D Gaussian Splatting." *CVPR 2024*, pp. 20796-20805. [arXiv:2312.07504](https://arxiv.org/abs/2312.07504)
2. **CF-3DGS 프로젝트 페이지**: https://oasisyang.github.io/colmap-free-3dgs/
3. **CVPR 2024 공식 논문 PDF**: https://openaccess.thecvf.com/content/CVPR2024/papers/Fu_COLMAP-Free_3D_Gaussian_Splatting_CVPR_2024_paper.pdf
4. **CVPR 2024 보충 자료**: https://openaccess.thecvf.com/content/CVPR2024/supplemental/Fu_COLMAP-Free_3D_Gaussian_CVPR_2024_supplemental.pdf
5. **Shi, D. et al.** (2025). "TrackGS: Optimizing COLMAP-Free 3D Gaussian Splatting with Global Track Constraints." [arXiv:2502.19800](https://arxiv.org/abs/2502.19800)
6. **Wei, Y. et al.** (2025). "PCR-GS: COLMAP-Free 3D Gaussian Splatting via Pose Co-Regularizations." [arXiv:2507.13891](https://arxiv.org/abs/2507.13891)
7. **Bian, W. et al.** (2023). "NoPe-NeRF: Optimising Neural Radiance Field with No Pose Prior." *CVPR 2023*.
8. **Kerbl, B. et al.** (2023). "3D Gaussian Splatting for Real-Time Radiance Field Rendering." *ACM TOG*.
9. **Semantic Scholar**: https://www.semanticscholar.org/paper/COLMAP-Free-3D-Gaussian-Splatting-Fu-Liu/b8eb5493895c8a342cfb176e90f57bc5f483a07c
10. **Liner Quick Review**: https://liner.com/review/colmapfree-3d-gaussian-splatting
