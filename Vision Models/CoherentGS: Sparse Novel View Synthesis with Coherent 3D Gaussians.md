# CoherentGS: Sparse Novel View Synthesis with Coherent 3D Gaussians

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

CoherentGS는 **극단적으로 희소한 입력 이미지(2~4장)** 환경에서 3D Gaussian Splatting(3DGS)이 심각하게 과적합(overfitting)되는 문제를 해결하기 위해, **Gaussian들 사이에 구조적 일관성(coherency)을 부여**하는 정규화 최적화 프레임워크를 제안한다.

기존 3DGS는 dense input에서는 탁월하지만, sparse input(예: 3장)에서는 비구조적인 점군(point cloud) 특성 때문에 각 Gaussian이 독립적으로 이동하여 novel view에서 "바늘 더미(jumble of needles)"처럼 보이는 아티팩트가 발생한다.

### 주요 기여 3가지

1. **구조화된 Gaussian 표현(Structured Gaussian Representation)**: 각 픽셀에 단일 Gaussian을 할당하여 2D 이미지 공간에서 제약 가능한 구조를 형성
2. **다중 정규화를 통한 일관성 도입**: 암묵적 합성곱 디코더(implicit convolutional decoder)를 이용한 단일뷰 제약, Total Variation 손실을 이용한 다중뷰 제약, 광학 흐름(optical flow) 기반 정규화
3. **단안 깊이 기반 초기화(Monocular Depth-based Initialization)**: 뷰 간 깊이 정합(scale & offset 최적화)을 통해 일관된 초기 Gaussian 위치 제공

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

| 문제 | 설명 |
|---|---|
| 3DGS의 sparse 설정 취약성 | 비구조적 표현이 training view에 과적합, novel view에서 floater 및 아티팩트 발생 |
| NeRF 기반 방법의 한계 | implicit MLP의 coherency에 의존하는 정규화가 3DGS의 explicit 표현에 직접 적용 불가 |
| 단안 깊이의 비일관성 | 뷰 간 깊이 스케일 불일치로 인한 Gaussian 초기 위치 오정렬 |

### 2.2 제안하는 방법 (수식 포함)

#### 기본 3DGS 렌더링 모델

각 픽셀 $\mathbf{p}$에 대한 렌더링 색상:

$$R_{\Sigma, \alpha, \mathbf{x}, \mathbf{c}}(\mathbf{p}) = \sum_{i \in \mathcal{N}(\mathbf{p})} \mathbf{c}_i \gamma_i \prod_{j=1}^{i-1}(1 - \gamma_j), \quad \text{where} \quad \gamma_k = f(\Sigma_k, \alpha_k, \mathbf{x}_k, \mathbf{p}) $$

#### 표준 3DGS 최적화 목표

```math
\Sigma^*, \alpha^*, \mathbf{x}^*, \mathbf{c}^* = \arg\min_{\Sigma, \alpha, \mathbf{x}, \mathbf{c}} \sum_{\mathbf{p} \in \mathcal{P}} \mathcal{L}(R_{\Sigma, \alpha, \mathbf{x}, \mathbf{c}}(\mathbf{p}), R(\mathbf{p}))
```

#### 픽셀별 Gaussian 위치 제어 (잔차 깊이 기반)

각 픽셀 $\mathbf{p}$의 Gaussian 위치를 깊이로 제어:

$$\mathbf{x} = g(D_n^{\text{init}}[\mathbf{p}] + \Delta D_n[\mathbf{p}], \mathbf{p}) $$

여기서 $g(d, \mathbf{p})$는 픽셀 $\mathbf{p}$를 깊이 $d$에 따라 3D 공간으로 투영하는 함수이며, $D_n^{\text{init}}$은 초기 깊이 추정값, $\Delta D_n$은 implicit decoder $f_\phi(n)$이 예측하는 잔차 깊이.

#### 다중뷰 제약: Total Variation 손실

렌더링된 시차(disparity)에 대한 TV 손실:

$$\mathcal{L}_{\text{TV}} = \left\| \nabla \left( \frac{1}{1 + R_{\Sigma, \alpha, \mathbf{x}, d}} \right) \right\|_1, \quad \mathcal{L}_{\text{MTV}} = \left\| \nabla \left( \mathbf{S} \odot \left( \frac{1}{1 + R_{\Sigma, \alpha, \mathbf{x}, d}} \right) \right) \right\|_1 $$

점진적으로 조합되는 다중뷰 손실:

$$\mathcal{L}_{\text{multi}} = (1 - \lambda_s)\mathcal{L}_{\text{TV}} + \lambda_s \mathcal{L}_{\text{MTV}} $$

여기서 $\lambda_s$는 0에서 시작하여 최적화 종료 시 1에 도달하도록 점진적으로 증가.

#### 광학 흐름 기반 정규화 손실

뷰 $i$, $j$ 간 대응 픽셀 $\mathbf{p}$, $\mathbf{q}$의 3D 위치를 일치시키는 손실:

$$\mathcal{L}_{\text{flow}} = \sum_{(i,j)} \sum_{\mathbf{p}} \left\| M_{i \rightarrow j} \odot \left( g(D_i[\mathbf{p}], \mathbf{p}) - g(D_j[\mathbf{q}], \mathbf{q}) \right) \right\|_1 $$

여기서 $M_{i \rightarrow j}$는 forward-backward consistency check로 얻은 신뢰할 수 있는 대응점의 이진 마스크.

#### 최종 최적화 목표

```math
\Sigma^*, \phi^*, \mathbf{c}^* = \arg\min_{\Sigma, \phi, \mathbf{c}} \sum_{\mathbf{p} \in \mathcal{P}} \mathcal{L}(R_{\Sigma, \alpha, \mathbf{x}, \mathbf{c}}(\mathbf{p}), R(\mathbf{p})) + \beta_m \mathcal{L}_{\text{multi}} + \beta_f \mathcal{L}_{\text{flow}}
```

$\beta_m = 5$, $\beta_f = 0.1$로 설정. 불투명도 $\alpha$와 위치 $\mathbf{x}$는 디코더 파라미터 $\phi$ 업데이트를 통해 간접적으로 최적화됨.

#### 깊이 초기화: Scale & Offset 정렬

단안 깊이 $D_i^{\text{m}}$의 스케일 $s_i$와 오프셋 $o_i$를 최적화:

```math
\mathbf{s}^*, \mathbf{o}^* = \arg\min_{\mathbf{s}, \mathbf{o}} \sum_{(i,j)} \sum_{\mathbf{p}} \left\| M_{i \rightarrow j} \odot \left( g(s_i \cdot D_i^{\text{m}}[\mathbf{p}] + o_i, \mathbf{p}) - g(s_j \cdot D_j^{\text{m}}[\mathbf{q}] + o_j, \mathbf{q}) \right) \right\|_1 
```

최종 초기 깊이: $D^{\text{init}} = s \cdot D^{\text{m}} + o$

#### Gaussian 반경(스케일) 초기화

$$r = \frac{f \cdot D^{\text{init}}}{H} $$

여기서 $f$는 수직 초점 거리, $H$는 이미지 높이.

### 2.3 모델 구조

```
입력 이미지들 (N = 2~4장)
        ↓
  ┌─────────────────────────────────────────────┐
  │           Depth-based Initialization        │
  │  • Depth Anything [Yang et al., 2024]       │
  │  • FlowFormer++ [Shi et al., 2023]          │
  │  • Scale & Offset 최적화 (Eq. 8, 1000 iter)│
  └──────────────────┬──────────────────────────┘
                     ↓
         초기 Gaussian 집합 (픽셀별 1개)
         + 깊이 기반 세분화 마스크 S (5채널)
                     ↓
  ┌─────────────────────────────────────────────┐
  │        Regularized Optimization             │
  │  ┌─────────────────────────────────────┐   │
  │  │   Implicit Convolutional Decoder    │   │
  │  │   입력: 정규화된 뷰 인덱스 n        │   │
  │  │   출력: 잔차 깊이 ΔD_n (5채널)     │   │
  │  │         잔차 불투명도 Δα_n          │   │
  │  └─────────────────────────────────────┘   │
  │  • 단일뷰 제약: Decoder smooth deformation  │
  │  • 다중뷰 제약: TV Loss (Eq. 4, 5)         │
  │  • 흐름 정규화: Flow Loss (Eq. 6)          │
  │  • 멀티샘플링: 픽셀 내 다중 샘플           │
  │  총 13,000 iter (처음 8,000: 고정 회전/스케일│
  │              나머지 5,000: 자유 최적화)     │
  └──────────────────┬──────────────────────────┘
                     ↓
           재구성된 3D Gaussians
```

**Implicit Decoder 구조 (Supplementary 참고)**:
- CoordConv 레이어 + 정규화 뷰 인덱스 $n$ 입력
- 일련의 합성곱 + 이중선형 업샘플링 레이어
- 용량 인수(capacity factor): 깊이 디코더 [10, 15, 18], 불투명도 디코더 [6, 10, 12] (뷰 수 2, 3, 4에 대응)

### 2.4 성능 향상

#### LLFF 데이터셋 (2~4뷰)

| 방법 | PSNR (2/3/4) | SSIM (2/3/4) | LPIPS (2/3/4) |
|---|---|---|---|
| 3DGS | 12.83/14.99/17.31 | 0.311/0.483/0.584 | 0.470/0.362/0.297 |
| RegNeRF | 16.55/19.41/21.49 | 0.468/0.627/0.713 | 0.417/0.306/0.257 |
| FreeNeRF | 17.07/19.97/21.80 | 0.513/0.652/0.713 | 0.376/0.280/0.259 |
| SparseNeRF | 17.74/20.33/21.90 | 0.513/0.657/0.720 | 0.386/0.302/0.260 |
| **Ours** | **18.32/20.33/21.58** | **0.644/0.725/0.762** | **0.220/0.180/0.167** |

- **LPIPS에서 가장 큰 개선**: 2뷰 기준 SparseNeRF 대비 약 43% 향상 (0.386 → 0.220)
- **추론 속도**: LLFF 3뷰 기준 278 fps (NeRF 기반 방법 대비 약 3,475배 빠름, 0.08 fps 대비)

#### DNGaussian 대비 (LLFF 3뷰)

| 방법 | PSNR | SSIM | LPIPS |
|---|---|---|---|
| DNGaussian | 19.55 | 0.647 | 0.264 |
| **Ours** | **20.33** | **0.725** | **0.180** |

#### Ablation Study 결과 (LLFF 3뷰)

| 구성 | PSNR | SSIM | LPIPS |
|---|---|---|---|
| w/o alignment | 19.06 | 0.679 | 0.217 |
| **w/o implicit decoder** | **16.68** | **0.477** | **0.331** |
| w/o tv reg. | 20.20 | 0.724 | 0.186 |
| w/o flow reg. | 20.32 | 0.723 | 0.185 |
| w/o multisampling | 19.99 | 0.718 | 0.194 |
| **Ours (전체)** | **20.33** | **0.725** | **0.180** |

Implicit decoder 제거 시 성능 저하가 가장 심각 (PSNR 3.65 dB 하락, SSIM 0.248 하락).

### 2.5 한계

1. **투명/반사 물체 처리 어려움**: 픽셀당 단일 Gaussian 할당 구조로 인해 유리나 반사면처럼 여러 레이어가 존재하는 장면 재구성 한계 (Fig. 10 참조)
2. **단안 깊이 정확도 의존성**: 단안 깊이 추정이 부정확할 경우 결과 품질 저하
3. **폐색 영역 미복원**: 모든 입력 뷰에서 가려진 영역은 재구성하지 않음 (단, 이를 역으로 이용해 inpainting으로 보완 가능)
4. **per-scene 최적화**: Feed-forward 방식이 아닌 scene별 최적화 필요 (generalization 한계)

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화 구조와 한계

CoherentGS는 **per-scene 최적화(test-time optimization)** 방식이다. 즉, 새로운 장면마다 전체 최적화 과정(14,000 iterations)을 처음부터 수행해야 한다. 이는 다음과 같은 구조적 한계를 갖는다:

- 학습된 파라미터가 특정 장면에 특화되어 있어 다른 장면으로의 **직접적인 zero-shot 전이 불가**
- 최적화 비용이 장면마다 발생

### 3.2 일반화 잠재력이 있는 요소들

**① Implicit Decoder의 재사용 가능성**

암묵적 합성곱 디코더 $f_\phi$는 단순히 뷰 인덱스 $n$을 입력으로 받아 잔차 깊이를 예측한다. 이 구조는 다음과 같이 확장 가능하다:

$$\Delta D_n = f_\phi(\mathbf{F}_n, n)$$

여기서 $\mathbf{F}_n$은 이미지 특징 맵(feature map)으로, 장면에 무관한 방식으로 decoder를 조건화할 수 있다. 이는 **IBRNet**, **pixelNeRF**, **MVSNeRF**와 같은 generalizable NeRF의 접근 방식과 유사하다.

**② 단안 깊이 모델의 일반화 능력**

Depth Anything [Yang et al., 2024]은 대규모 unlabeled 데이터로 학습되어 높은 일반화 성능을 보인다. 이 강력한 prior가 CoherentGS 프레임워크에 내재되어 있어, 다양한 장면 유형에 대한 암묵적 일반화 기반을 제공한다.

**③ Flow 기반 대응 관계의 범용성**

FlowFormer++ 기반의 광학 흐름 추정은 장면 유형에 무관하게 동작하므로, 다양한 도메인(실내/실외, 다양한 물체 유형)에 적용 가능한 일반적인 제약 신호를 제공한다.

### 3.3 일반화 성능 향상을 위한 구체적 방향

**방향 1: Feed-forward 방식으로의 확장**

현재 per-scene 최적화를 feed-forward 네트워크로 대체:

$$(\Delta D, \Delta\alpha, \Sigma, \mathbf{c}) = \Phi_\theta(\{I_n, D_n^{\text{m}}\}_{n=1}^{N})$$

여기서 $\Phi_\theta$는 다중 뷰 입력으로부터 Gaussian 파라미터를 직접 예측하는 네트워크. pixelSplat [Chen et al., 2024], MVSplat [Chen et al., 2024] 등이 이러한 방향을 탐색하고 있다.

**방향 2: 메타 러닝(Meta-Learning) 적용**

MAML 또는 Reptile 등의 메타 러닝으로 decoder 파라미터 $\phi$를 초기화:

$$\phi_0 = \arg\min_\phi \mathbb{E}_{\text{scene}} \left[ \mathcal{L}_{\text{novel}}(\phi - \alpha \nabla_\phi \mathcal{L}_{\text{sparse}}(\phi)) \right]$$

좋은 초기값을 통해 few-shot adaptation이 빠르게 수렴 가능.

**방향 3: 세그멘테이션 마스크 생성의 자동화 및 일반화**

현재 고정된 채널 수($C=5$)의 깊이 기반 세분화를 적응적으로 결정하는 방법 연구:

$$C^* = \arg\max_C \text{성능}(C)$$

장면 복잡도에 따라 자동으로 $C$를 결정하는 메커니즘이 일반화에 기여할 수 있다.

**방향 4: 대규모 사전 학습된 특징과의 결합**

DINOv2, CLIP 등의 대규모 사전 학습된 시각 모델에서 추출된 특징을 decoder 조건으로 활용:

$$\Delta D_n = f_\phi(\text{DINO}(I_n), n)$$

이를 통해 의미론적 일관성(semantic coherency)을 추가로 확보.

---

## 4. 최신 연구 비교 분석 (2020년 이후)

### 4.1 Sparse View Synthesis 관련 방법 비교

| 방법 | 표현 | 정규화 전략 | Sparse 처리 | 추론 속도 | 일반화 |
|---|---|---|---|---|---|
| **NeRF** (Mildenhall et al., ECCV 2020) | Implicit MLP | 없음 | ❌ | 매우 느림 | ❌ |
| **RegNeRF** (Niemeyer et al., CVPR 2022) | Implicit MLP | 비관측 뷰 기하/색상 정규화 | ⭕ | 느림 | ❌ |
| **DS-NeRF** (Deng et al., CVPR 2022) | Implicit MLP | COLMAP 깊이 감독 | ⭕ | 느림 | ❌ |
| **FreeNeRF** (Yang et al., CVPR 2023) | Implicit MLP | Frequency 점진적 인코딩 | ⭕ | 느림 | ❌ |
| **SparseNeRF** (Wang et al., ICCV 2023) | Implicit MLP | 단안 깊이 순위 정규화 | ⭕ | 느림 | ❌ |
| **FSGS** (Zhu et al., 2023) | 3DGS | 초기화 개선 | ⭕ | **빠름** | ❌ |
| **DNGaussian** (Li et al., CVPR 2024) | 3DGS | 전역-지역 깊이 정규화 | ⭕ | **빠름** | ❌ |
| **SparseGS** (Xiong et al., 2023) | 3DGS | 마스크 기반 | ⭕ | **빠름** | ❌ |
| **CoherentGS** (Paliwal et al., 2024) | 3DGS | Implicit decoder + TV + Flow | ⭕ | **매우 빠름** | ❌ (향후 가능성 ⭕) |
| **pixelSplat** (Chen et al., 2024) | 3DGS | Feed-forward | ⭕ | **실시간** | ⭕ |
| **MVSplat** (Chen et al., 2024) | 3DGS | Cost volume | ⭕ | **실시간** | ⭕ |

> **참고**: pixelSplat 및 MVSplat은 본 논문에서 직접 비교되지 않았으며, 이들은 generalizable 방향으로의 발전을 보여주는 동시기/이후 연구들임.

### 4.2 CoherentGS의 차별화 포인트

1. **coherency 메커니즘**: FSGS, DNGaussian, SparseGS와 달리 인접 Gaussian 간 이동 일관성을 명시적으로 강제
2. **폐색 영역 처리**: 다른 방법들이 elongated Gaussian으로 폐색 영역을 흐릿하게 채우는 반면, CoherentGS는 폐색 영역을 인식하고 diffusion model 기반 inpainting으로 고품질 처리 가능
3. **추론 속도**: NeRF 기반 방법 대비 약 3,475배 빠른 278 fps

---

## 5. 향후 연구에 미치는 영향 및 고려할 점

### 5.1 향후 연구에 미치는 영향

**① 3DGS의 sparse 설정 연구 활성화**

CoherentGS는 3DGS를 sparse input에 적용하는 선구적 연구로, 이후 FSGS, DNGaussian, SparseGS, pixelSplat, MVSplat 등 관련 연구들의 토대를 마련했다. 특히 **픽셀당 Gaussian 할당** 아이디어는 구조적 표현의 효과를 입증하여 이후 연구들이 이 방향을 계승하고 있다.

**② 명시적 표현에서의 coherency 도입 패러다임**

NeRF의 implicit MLP가 제공하던 자연스러운 공간적 coherency를 3DGS 같은 explicit 표현에서 어떻게 인위적으로 부여할 것인가라는 질문에 대해 concrete한 방법론(decoder + TV loss)을 제시했다.

**③ 단안 깊이와 광학 흐름의 보조 신호 활용 프레임워크**

Depth Anything + FlowFormer++를 초기화 및 정규화에 통합하는 방식은 이후 sparse view synthesis 연구들이 외부 foundation model을 활용하는 방향을 제시했다.

**④ 3D inpainting과 생성 모델의 결합**

폐색 영역을 diffusion model로 inpainting한 후 3D로 투영하는 파이프라인은 생성 모델과 3D 재구성의 결합 가능성을 보여주었다.

### 5.2 앞으로 연구 시 고려할 점

**① Feed-forward Generalization 확장**

현재 per-scene 최적화 구조를 feed-forward 방식으로 전환하는 것이 중요한 과제다:
- 대규모 다중 뷰 데이터셋(예: RealEstate10K, CO3D)으로 decoder를 사전 학습
- 이미지 특징 조건부 decoder 설계: $\Delta D = f_\phi(\text{features}(I), n)$

**② 더 강건한 깊이 통합**

단안 깊이 모델의 오류가 전파되는 문제를 해결하기 위해:
- 깊이 불확실성(uncertainty)을 추정하여 신뢰도 가중 정규화 적용
- Depth Anything V2와 같은 더 강력한 foundation depth model 활용
- 깊이와 3DGS를 공동 학습하는 end-to-end 프레임워크 설계

**③ 투명/반사 물체 처리**

픽셀당 단일 Gaussian 할당이 투명 물체에서 실패하는 문제:
- 투명도가 높은 영역에 multiple Gaussian 할당 허용
- 물체 분할 정보를 활용한 적응적 Gaussian 수 결정

**④ 동적 장면으로의 확장**

현재는 정적 장면만 가정:
- 광학 흐름 기반 정규화를 동적 요소 감지에 활용
- 시간적 coherency 제약 추가

**⑤ 카메라 포즈 불확실성 처리**

현재는 정확한 카메라 포즈를 가정:
- BARF [Lin et al., ICCV 2021] 스타일의 포즈-장면 공동 최적화와 결합
- Pose-free sparse view synthesis로의 확장

**⑥ 평가 프로토콜의 표준화**

폐색 마스킹 기반 평가(occluded region masking)를 사용하는 CoherentGS와 전체 이미지 평가를 사용하는 다른 방법들의 비교가 완전히 공정하지 않을 수 있음. 향후 연구에서는 통일된 평가 기준 마련이 필요하다.

---

## 참고 자료

**주요 참고 논문 (본 PDF 논문 내 인용 기준)**:

1. **CoherentGS 원문**: Paliwal et al., "CoherentGS: Sparse Novel View Synthesis with Coherent 3D Gaussians," arXiv:2403.19495v2, 2024. [https://arxiv.org/abs/2403.19495](https://arxiv.org/abs/2403.19495)
2. Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering," ACM ToG, 2023.
3. Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis," ECCV, 2020.
4. Yang et al., "Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data," CVPR, 2024.
5. Shi et al., "FlowFormer++: Masked Cost Volume Autoencoding for Pretraining Optical Flow Estimation," CVPR, 2023.
6. Niemeyer et al., "RegNeRF: Regularizing Neural Radiance Fields for View Synthesis from Sparse Inputs," CVPR, 2022.
7. Wang et al., "SparseNeRF: Distilling Depth Ranking for Few-Shot Novel View Synthesis," ICCV, 2023.
8. Yang et al., "FreeNeRF: Improving Few-Shot Neural Rendering with Free Frequency Regularization," CVPR, 2023.
9. Li et al., "DNGaussian: Optimizing Sparse-View 3D Gaussian Radiance Fields with Global-Local Depth Normalization," CVPR, 2024.
10. Zhu et al., "FSGS: Real-Time Few-Shot View Synthesis using Gaussian Splatting," 2023.
11. Bemana et al., "X-Fields: Implicit Neural View-, Light- and Time-Image Interpolation," ACM SIGGRAPH Asia, 2020.
12. Luo et al., "Consistent Video Depth Estimation," ACM SIGGRAPH, 2020.

> **⚠️ 주의**: pixelSplat, MVSplat 등 2024년 이후 후속 연구들은 본 논문 원문에 포함되지 않으며, 일반화 방향 논의에서 맥락적 참조로만 언급하였습니다. 해당 연구들의 구체적 수치 비교는 원문을 직접 확인하시기 바랍니다.
