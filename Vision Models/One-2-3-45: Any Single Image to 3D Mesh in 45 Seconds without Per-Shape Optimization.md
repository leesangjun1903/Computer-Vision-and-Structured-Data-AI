# One-2-3-45: Any Single Image to 3D Mesh in 45 Seconds without Per-Shape Optimization

## 논문 종합 분석 보고서

---

## 1. 핵심 주장 및 주요 기여 요약

Single image 3D reconstruction은 자연 세계에 대한 광범위한 지식을 요구하는 중요하지만 도전적인 과제입니다. 기존 많은 방법들은 2D 확산 모델의 가이던스 하에 Neural Radiance Field(NeRF)를 최적화하여 이 문제를 해결하지만, 긴 최적화 시간, 3D 불일치 결과, 그리고 낮은 기하학적 품질에 시달립니다.

**핵심 주장:** 본 논문은 임의의 객체에 대한 단일 이미지를 입력으로 받아, 단일 피드포워드 패스에서 360도 3D 텍스처 메시를 생성하는 새로운 방법을 제안합니다.

**주요 기여:**
1. 단일 이미지를 입력받아 view-conditioned 2D 확산 모델인 Zero123을 사용해 멀티뷰 이미지를 생성하고, 이를 3D 공간으로 리프팅하는 파이프라인을 제안합니다. 전통적 복원 방법이 불일치하는 멀티뷰 예측에 어려움을 겪기 때문에, SDF 기반 일반화 가능 신경 표면 복원 방법 위에 3D 복원 모듈을 구축하고 360도 메시 복원을 위한 핵심적 학습 전략들을 제안합니다.
2. 비용이 많이 드는 최적화 없이, 기존 방법보다 훨씬 짧은 시간에 3D 형상을 복원하며, 더 나은 기하학적 품질, 더 높은 3D 일관성, 그리고 입력 이미지에 대한 더 높은 충실도를 제공합니다.
3. 또한 기성(off-the-shelf) 텍스트-투-이미지 확산 모델과 통합하여 text-to-3D 작업도 원활하게 지원할 수 있습니다.

---

## 2. 상세 기술 분석

### 2.1 해결하고자 하는 문제

기존 최적화 기반 방법(DreamFusion, RealFusion 등)의 근본적 한계:

Per-shape optimization은 일반적으로 전체 이미지 볼륨 렌더링과 사전 모델 추론의 수만 번 반복을 포함하며, 형상당 일반적으로 수십 분이 소요됩니다.

구체적 문제점:
- 메모리 집약적: 2D 사전 모델에 전체 이미지가 필요하므로, 이미지 해상도가 올라갈수록 볼륨 렌더링이 메모리 집약적이 됩니다.
- 3D 불일치: 2D 사전 모델은 각 반복에서 단일 뷰만 보고 모든 뷰가 입력처럼 보이도록 만들려 하므로, 종종 3D 불일치 형상(예: 두 개의 얼굴, 또는 Janus 문제)을 생성합니다.
- 낮은 기하학적 품질: 많은 방법이 볼륨 렌더링에서 밀도 필드를 표현으로 사용하여, 좋은 RGB 렌더링을 생성하지만 고품질 메시 추출이 어렵습니다.

또한, 클래스 특정 사전을 이용한 기존 방법들은 보지 못한 카테고리에 대한 일반화에 실패하며, 복원 품질이 제한된 공공 3D 데이터셋 크기에 의해 제약됩니다.

### 2.2 제안하는 방법 (모델 구조)

멀티뷰 합성, 고도(elevation) 추정, 3D 복원의 세 모듈을 통합하여, 단일 이미지에서 피드포워드 방식으로 3D 메시를 복원합니다.

#### (a) Multi-view Synthesis (멀티뷰 합성)

view-conditioned 2D 확산 모델인 Zero123을 사용하여 2단계 방식으로 멀티뷰 이미지를 생성합니다. Zero123의 입력은 단일 이미지와 상대적 카메라 변환이며, 이는 상대적 구면 좌표 $(\Delta\theta, \Delta\phi, \Delta r)$로 매개변수화됩니다.

Zero123 모델의 뷰 조건 생성은 다음과 같이 공식화됩니다:

$$\hat{x}_{\text{novel}} = f_{\text{Zero123}}(x_{\text{input}}, \Delta\theta, \Delta\phi, \Delta r)$$

여기서:
- $x_{\text{input}}$: 입력 단일 이미지
- $(\Delta\theta, \Delta\phi, \Delta r)$: 상대적 고도, 방위각, 거리 변화
- $\hat{x}_{\text{novel}}$: 생성된 새로운 뷰 이미지

**2단계 생성 전략:**
- **1단계:** 입력 뷰에서 근접한 뷰들을 생성 (작은 $\Delta\theta, \Delta\phi$)
- **2단계:** 1단계에서 생성된 뷰들을 조건으로 사용하여 더 먼 뷰들을 생성 → 360도 커버리지 달성

#### (b) Pose Estimation (포즈 추정)

Zero123이 생성한 4개의 인접 뷰를 기반으로 입력 이미지의 고도각 $\theta$를 추정합니다. 그런 다음, 지정된 상대적 포즈와 추정된 입력 뷰의 포즈를 결합하여 멀티뷰 이미지의 포즈를 얻습니다.

고도각 추정 최적화:

$$\hat{\theta} = \arg\min_{\theta} \sum_{i=1}^{4} \mathcal{L}_{\text{reproj}}(I_i, \theta)$$

#### (c) 3D Reconstruction (3D 복원) — SDF 기반 일반화 가능 신경 표면 복원

멀티뷰 포즈 이미지를 SDF 기반 일반화 가능 신경 표면 복원 모듈에 입력하여 360도 메시를 복원합니다.

본 논문의 3D 복원 모듈은 **SparseNeuS** 아키텍처를 기반으로 구축됩니다. SDF(Signed Distance Function) 표현은 다음과 같이 정의됩니다:

$$S = \{\mathbf{x} \in \mathbb{R}^3 \mid f(\mathbf{x}) = 0\}$$

여기서 $f(\mathbf{x})$는 3D 공간 점 $\mathbf{x}$에서의 부호 거리 함수 값입니다.

**볼륨 렌더링 수식 (NeuS 기반):**

$$\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i \alpha_i c_i$$

여기서:
- $T_i = \prod_{j=1}^{i-1}(1 - \alpha_j)$: 누적 투과율
- $\alpha_i$: SDF 값으로부터 변환된 불투명도 (opacity)
- $c_i$: 샘플 포인트의 색상

불투명도 변환 함수:

$$\alpha_i = \max\left(\frac{\Phi_s(f(\mathbf{x}_i)) - \Phi_s(f(\mathbf{x}_{i+1}))}{\Phi_s(f(\mathbf{x}_i))}, 0\right)$$

여기서 $\Phi_s(x) = (1 + e^{-sx})^{-1}$는 시그모이드 함수이며, $s$는 학습 가능한 파라미터입니다.

**학습 손실 함수:**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{color}} + \lambda_1 \mathcal{L}_{\text{depth}} + \lambda_2 \mathcal{L}_{\text{normal}} + \lambda_3 \mathcal{L}_{\text{reg}}$$

- $\mathcal{L}\_{\text{color}} = \sum_{\mathbf{r}} \|\hat{C}(\mathbf{r}) - C_{\text{gt}}(\mathbf{r})\|_1$: 컬러 복원 손실
- $\mathcal{L}_{\text{depth}}$: 깊이 일관성 손실
- $\mathcal{L}_{\text{normal}}$: 법선 벡터 일관성 손실
- $\mathcal{L}\_{\text{reg}}$: Eikonal 정규화 항 $\mathcal{L}\_{\text{eik}} = \sum_{\mathbf{x}} (\|\nabla f(\mathbf{x})\| - 1)^2$

**핵심 학습 전략:**
1. **비용 볼륨 기반 특징 융합**: 멀티뷰 이미지에서 추출한 특징들을 비용 볼륨을 통해 융합
2. **360도 학습을 위한 데이터 증강**: Objaverse-LVIS 데이터셋에서 다양한 시점의 렌더링을 활용
3. **불일치 뷰 처리**: Zero123이 생성하는 불완전하고 불일치한 뷰에 대한 견고성 확보

### 2.3 성능 향상

비용이 많이 드는 최적화 없이, 단 45초 만에 3D 형상을 복원합니다.

| 방법 | 시간 | 3D 일관성 | 기하학적 품질 |
|------|------|-----------|-------------|
| DreamFusion | ~1.5시간 | 낮음 (Janus 문제) | 낮음 (밀도 필드) |
| RealFusion | ~1시간 | 낮음 | 낮음 |
| **One-2-3-45** | **~45초** | **높음** | **높음 (SDF)** |
| Point-E | ~1분 | 중간 | 중간 (포인트 클라우드) |
| Shap-E | ~1분 | 중간 | 중간 |

SDF 표현을 사용하므로 더 나은 기하학적 품질을 보이며, 카메라 조건부 멀티뷰 예측 덕분에 더 일관된 3D 메시를 생성합니다.

합성 데이터와 실제 이미지(in-the-wild) 모두에서 평가하여 메시 품질과 실행 시간 양면에서 우수성을 입증합니다.

### 2.4 한계

논문에서 인정하는 주요 한계점:

1. **Zero123의 품질 의존성**: 멀티뷰 이미지 생성 품질이 Zero123의 성능에 크게 의존하며, 복잡한 형상이나 텍스처에 대해 불일치가 발생할 수 있음
2. **고도각 추정 오차**: 입력 이미지의 포즈 추정이 정확하지 않으면 전체 복원 품질에 영향
3. **해상도 제한**: 생성되는 메시의 세부 디테일이 SparseNeuS 기반 복원의 해상도에 제약
4. **얇은 구조 복원 어려움**: 매우 얇거나 복잡한 토폴로지를 가진 구조물의 복원이 여전히 어려움
5. **카테고리 편향**: Objaverse 데이터셋으로 학습된 특성으로 인해 특정 카테고리에 편향 가능

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재의 일반화 전략

One-2-3-45는 2D 확산 모델을 3D AIGC에 활용하는 방법을 재고하여, 시간이 많이 소요되는 최적화를 피하는 새로운 전방향 전용(forward-only) 패러다임을 도입합니다.

**일반화의 핵심 원천:**

1. **2D 확산 모델의 사전 지식 활용**: Zero123은 대규모 이미지-3D 데이터 쌍으로 사전 학습되어 다양한 객체 카테고리에 대한 지식을 내재화
2. **SDF 기반 일반화 가능 복원**: 카테고리 특정이 아닌 범용 SDF 복원 모듈 사용으로 미지의 객체에 대한 일반화 가능
3. **Objaverse 대규모 데이터셋 학습**: Objaverse-LVIS 데이터셋을 학습에 사용하고, 선택된 형상(CC-BY 라이선스)을 Blender로 2D 이미지로 렌더링합니다.

### 3.2 일반화 성능 향상을 위한 방향

#### (1) 더 강력한 멀티뷰 생성 모델

$$\text{Generalization} \propto \text{Quality}(f_{\text{MV-Diffusion}}) \times \text{Diversity}(\mathcal{D}_{\text{train}})$$

- Zero123 → Zero123++ → SV3D 등 더 발전된 멀티뷰 확산 모델 적용
- 비디오 확산 모델의 시간적 사전(temporal prior) 활용

#### (2) 더 큰 3D 데이터셋 활용

$$\mathcal{L}_{\text{generalize}} = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} \left[\sum_{v \in \mathcal{V}} \|\hat{I}_v - I_v^{\text{gt}}\|^2 \right]$$

데이터셋 다양성 $p_{\text{data}}$의 범위를 확장하면 일반화 성능 직접 향상

#### (3) 스케일러블 아키텍처로의 전환
- SparseNeuS 기반 → Transformer 기반 Large Reconstruction Model(LRM)로 전환
- 더 많은 파라미터와 더 큰 데이터셋을 통한 스케일링 법칙 활용

#### (4) 도메인 적응 기법

$$\mathcal{L}_{\text{adapt}} = \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{domain}}$$

실제 환경(in-the-wild) 이미지와 합성 데이터 간의 도메인 갭을 줄이는 적응 학습

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 연구적 영향

본 논문은 일반적인 최적화 기반 패러다임을 따르는 대신, 3D 모델링을 위해 2D 사전 모델을 활용하는 새로운 접근법을 제안합니다.

**패러다임 전환의 영향:**
- **피드포워드 3D 생성의 실현 가능성 입증**: 형상별 최적화 없이도 합리적 품질의 3D 복원이 가능함을 최초로 체계적으로 보여줌
- **멀티뷰 확산 + 3D 복원 파이프라인의 표준화**: 이후 InstantMesh, TripoSR, One-2-3-45++ 등 후속 연구의 기본 프레임워크가 됨
- **Text-to-3D의 실용화**: 텍스트→이미지→3D 파이프라인의 가능성을 열어 실용적 응용 확대

### 4.2 향후 연구 시 고려할 점

1. **멀티뷰 일관성 강화**: Zero123 등 2D 확산 모델이 생성하는 멀티뷰 간의 3D 일관성 보장이 핵심 과제
2. **복원 해상도 및 디테일**: 고해상도 메시와 텍스처 생성을 위한 표현 방식 개선 필요
3. **실시간 응용**: 45초에서 더 빠른 추론 속도 달성을 위한 모델 경량화
4. **다양한 3D 표현 지원**: 메시 외에 가우시안 스플래팅(Gaussian Splatting) 등 새로운 표현 방식과의 통합
5. **평가 메트릭 표준화**: Chamfer Distance, F-Score, LPIPS, SSIM 등 통합적 벤치마크 필요

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 방법 | 속도 | 특징 |
|------|------|----------|------|------|
| **DreamFusion** (Poole et al.) | 2022 | SDS Loss + NeRF 최적화 | ~1.5시간 | Text-to-3D의 선구적 연구, per-shape 최적화 필요 |
| **Zero-1-to-3** (Liu et al.) | 2023 | View-conditioned Diffusion | - | 단일 이미지에서 새로운 뷰 합성 |
| **Point-E / Shap-E** (OpenAI) | 2022-23 | Transformer 기반 3D 생성 | ~1분 | 포인트 클라우드/암묵적 표현 직접 생성 |
| **One-2-3-45** (Liu et al.) | 2023 | Zero123 + SDF 복원 | **~45초** | 피드포워드, SDF 기반 메시 |
| **One-2-3-45++** (Liu et al.) | 2023 | 단일 이미지를 약 1분 만에 상세한 3D 텍스처 메시로 변환하며, 2D 확산 모델에 내재된 광범위한 지식과 제한적이지만 가치 있는 3D 데이터의 사전 지식을 완전히 활용합니다. 일관된 멀티뷰 이미지 생성을 위한 2D 확산 모델 미세 조정 후, 멀티뷰 조건부 3D 네이티브 확산 모델로 3D로 승격합니다. | ~1분 | 3D 네이티브 확산 모델 통합 |
| **TripoSR** (Tochilkin et al.) | 2024 | Transformer 아키텍처를 활용한 빠른 피드포워드 3D 생성 모델로, 단일 이미지에서 0.5초 이내에 3D 메시를 생성합니다. LRM 네트워크 아키텍처를 기반으로 데이터 처리, 모델 설계, 학습 기법에서 상당한 개선을 통합합니다. | **<0.5초** | Transformer/LRM 기반, 초고속 |
| **InstantMesh** (Xu et al.) | 2024 | 멀티뷰 확산 모델과 LRM 아키텍처 기반 sparse-view 복원 모델의 강점을 시너지화하는 피드포워드 프레임워크입니다. 입력 이미지로부터 멀티뷰 확산 모델이 3D 일관적 멀티뷰 이미지를 생성하고, 이를 sparse-view 복원 모델에 입력하여 고품질 3D 메시를 복원합니다. | ~10초 | FlexiCubes와 같은 등치면(iso-surface) 추출 모듈을 통합하여, 3D 기하학을 효율적으로 렌더링하고 깊이 및 법선과 같은 기하학적 감독을 메시 표현에 직접 적용합니다. |
| **SF3D** (Stability AI) | 2024 | TripoSR를 기반으로 구축되며, 더 높은 해상도의 triplane(384×384)과 강화된 transformer 아키텍처를 도입하여, TripoSR의 상대적으로 낮은 해상도(64×64) triplane의 한계를 정면으로 해결합니다. | ~2초 | 고해상도 triplane |

### 진화 흐름 요약

```
[2022] DreamFusion (최적화 기반, 느림)
    ↓ 패러다임 전환
[2023] Zero123 (뷰 조건부 확산) → One-2-3-45 (피드포워드 파이프라인)
    ↓ 스케일업
[2024] LRM/TripoSR (<0.5초) → InstantMesh (멀티뷰+LRM)
    ↓ 품질 향상
[2024] SF3D (고해상도 triplane) → MeshFormer (3D 네이티브 구조)
```

One-2-3-45의 가장 큰 공헌은 **"멀티뷰 확산 모델 + 일반화 가능 3D 복원"이라는 2단계 피드포워드 패러다임**을 확립한 것이며, 이는 이후 거의 모든 실시간 image-to-3D 연구의 기본 프레임워크가 되었습니다.

---

## 참고자료

1. **Liu, M., Xu, C., Jin, H., Chen, L., Varma T, M., Xu, Z., & Su, H.** (2024). "One-2-3-45: Any Single Image to 3D Mesh in 45 Seconds without Per-Shape Optimization." *Advances in Neural Information Processing Systems*, 36. — [arXiv:2306.16928](https://arxiv.org/abs/2306.16928)
2. **One-2-3-45 Project Page** — [https://one-2-3-45.github.io/](https://one-2-3-45.github.io/)
3. **One-2-3-45 Official GitHub Repository** — [https://github.com/One-2-3-45/One-2-3-45](https://github.com/One-2-3-45/One-2-3-45)
4. **NeurIPS 2023 Proceedings** — [https://proceedings.neurips.cc/paper_files/paper/2023/hash/4683beb6bab325650db13afd05d1a14a-Abstract-Conference.html](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4683beb6bab325650db13afd05d1a14a-Abstract-Conference.html)
5. **Liu, M., et al.** (2023). "One-2-3-45++: Fast Single Image to 3D Objects with Consistent Multi-View Generation and 3D Diffusion." — [arXiv:2311.07885](https://arxiv.org/abs/2311.07885)
6. **Tochilkin, D., et al.** (2024). "TripoSR: Fast 3D Object Reconstruction from a Single Image." — [arXiv:2403.02151](https://arxiv.org/abs/2403.02151)
7. **Xu, J., et al.** (2024). "InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models." — [arXiv:2404.07191](https://arxiv.org/abs/2404.07191)
8. **Semantic Scholar Paper Page** — [https://www.semanticscholar.org/paper/c22ef963b2388cdbbfcc7a00b24f68710a7febd2](https://www.semanticscholar.org/paper/c22ef963b2388cdbbfcc7a00b24f68710a7febd2)
9. **Hugging Face Paper Page** — [https://huggingface.co/papers/2306.16928](https://huggingface.co/papers/2306.16928)
10. **SF3D 분석 블로그** — [https://didyouknowbg8.wordpress.com/2024/08/02/sf3d-the-evolution-from-triposr/](https://didyouknowbg8.wordpress.com/2024/08/02/sf3d-the-evolution-from-triposr/)

> **참고:** 본 분석에서 수식 부분 중 논문 원문에 직접 명시되지 않은 일부 수식(예: 일반화 관련 개념적 수식)은 논문의 방법론을 기반으로 한 연구자의 해석적 정리임을 밝힙니다. SDF 볼륨 렌더링 관련 수식은 NeuS 프레임워크를 기반으로 하며, 논문에서 참조하는 선행 연구(SparseNeuS, NeuS)의 표준 공식을 따릅니다.
