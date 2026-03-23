# FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting

---

## 1. 핵심 주장과 주요 기여 요약

FSGS는 **극소수의 학습 뷰(3~24장)**만으로 **실시간(200+ FPS) 포토리얼리스틱 노벨 뷰 합성(Novel View Synthesis)**을 달성하는, 3D Gaussian Splatting 기반 프레임워크이다. 기존 NeRF 기반 few-shot 방법들이 높은 렌더링 품질과 실시간 속도 사이에서 트레이드오프를 겪는 문제를 해결한다.

**주요 기여 3가지:**

1. **Proximity-guided Gaussian Unpooling**: 극히 희소한 SfM 초기 포인트 클라우드로부터 Gaussian 간 근접도(proximity score)를 활용하여 새로운 Gaussian을 전략적으로 삽입, 장면 커버리지를 효과적으로 확대한다.
2. **Pseudo-view 합성 + 기하학적 정규화**: 학습 중 가상 카메라 뷰를 생성하고, 단안 깊이(monocular depth) prior를 활용한 Pearson correlation 기반 상대 깊이 정규화를 적용하여 과적합을 방지한다.
3. **실시간 렌더링 + SOTA 품질**: SparseNeRF 대비 SSIM 0.624→0.652, 추론 속도 **2,180배 이상** 빠른 성능을 달성한다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

3D Gaussian Splatting(3D-GS)은 밀집 뷰(100장 이상)에서 실시간 렌더링을 달성하지만, **few-shot(3~24장) 시나리오**에서 두 가지 핵심 문제가 발생한다:

- **문제 1 — 초기 포인트의 극심한 희소성**: COLMAP SfM이 소수 뷰에서 생성하는 포인트가 매우 적어, 기존 gradient 기반 densification으로는 장면 전체를 커버하지 못한다.
- **문제 2 — 과적합 및 과도한 스무딩**: 소수 뷰의 photometric loss만으로 최적화하면 학습 뷰에 과적합되고, 미관측 영역에서는 블러링과 아티팩트가 심해진다.

### 2.2 제안하는 방법 (수식 포함)

#### (A) 3D Gaussian Splatting 기본 구조

각 3D Gaussian은 위치 $\boldsymbol{\mu} \in \mathbb{R}^3$, 공분산 $\Sigma \in \mathbb{R}^{3 \times 3}$으로 정의된다:

$$G(\boldsymbol{x}) = \frac{1}{(2\pi)^{3/2} |\Sigma|^{1/2}} e^{-\frac{1}{2}(\boldsymbol{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\boldsymbol{x} - \boldsymbol{\mu})} $$

공분산은 양의 준정치를 보장하기 위해 $\Sigma = R S S^T R^T$로 분해된다( $R$: 회전 quaternion, $S$: 스케일). 2D 렌더링은 알파 블렌딩으로 수행된다:

$$c = \sum_{i=1}^{n} c_i \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j) $$

#### (B) Proximity-guided Gaussian Unpooling

**근접도 그래프 구성**: 각 Gaussian $G_i$에 대해 $K$-최근접 이웃을 구하고, 근접도 점수를 계산한다:

$$D_i^K = K\text{-min}(d_{ij}), \quad \forall j \neq i $$

여기서 $d_{ij} = \|\boldsymbol{\mu}_i - \boldsymbol{\mu}_j\|$이다. Gaussian $G_i$의 근접도 점수(proximity score)는:

$$P_i = \frac{1}{K} \sum_{j=1}^{K} D_i^K $$

$P_i > t_{\text{prox}}$인 경우 (즉, 이웃 Gaussian과의 간격이 임계값을 초과하면), "source"와 "destination" Gaussian을 잇는 **간선의 중앙에 새 Gaussian을 삽입**한다. 새 Gaussian의 스케일과 불투명도는 destination Gaussian의 것을 복사하고, 회전과 SH 계수는 0으로 초기화한다. 이 전략은 메시 세분화(mesh subdivision) 알고리즘에서 영감을 받았다.

#### (C) Pseudo-view 합성

두 가장 가까운 학습 카메라의 위치를 보간하고 노이즈를 추가하여 가상 카메라를 생성한다:

$$\boldsymbol{P}' = (\boldsymbol{t} + \varepsilon, \boldsymbol{q}), \quad \varepsilon \sim \mathcal{N}(0, \delta) $$

여기서 $\boldsymbol{t}$는 카메라 위치, $\boldsymbol{q}$는 두 카메라의 평균 회전 quaternion이다.

#### (D) 기하학적 정규화 — 상대 깊이 대응

사전학습된 DPT(Dense Prediction Transformer)로 추정한 깊이 $\hat{\boldsymbol{D}}\_{\text{est}}$와 래스터화된 깊이 $\hat{\boldsymbol{D}}_{\text{ras}}$ 사이의 **Pearson 상관계수**를 사용하여 스케일 불확실성을 완화한다:

$$\text{Corr}(\hat{\boldsymbol{D}}\_{\text{ras}}, \hat{\boldsymbol{D}}\_{\text{est}}) = \frac{\text{Cov}(\hat{\boldsymbol{D}}_{\text{ras}}, \hat{\boldsymbol{D}}_{\text{est}})}{\sqrt{\text{Var}(\hat{\boldsymbol{D}}_{\text{ras}}) \text{Var}(\hat{\boldsymbol{D}}_{\text{est}})}} $$

깊이 래스터화는 알파 블렌딩 기반으로 미분 가능하게 구현된다:

$$d = \sum_{i=1}^{n} d_i \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j) $$

#### (E) 최종 손실 함수

$$\mathcal{L}(\boldsymbol{G}, \boldsymbol{C}) = \lambda_1 \underbrace{\|\boldsymbol{C} - \hat{\boldsymbol{C}}\|_1}_{\mathcal{L}_1} + \lambda_2 \underbrace{\text{D-SSIM}(\boldsymbol{C}, \hat{\boldsymbol{C}})}_{\mathcal{L}_{\text{ssim}}} + \lambda_3 \underbrace{\|\text{Corr}(\boldsymbol{D}_{\text{ras}}, \boldsymbol{D}_{\text{est}})\|_1}_{\mathcal{L}_{\text{regularization}}} $$

$\lambda_1 = 0.8$, $\lambda_2 = 0.2$, $\lambda_3 = 0.05$로 설정되며, $\mathcal{L}_{\text{regularization}}$은 학습 뷰와 pseudo 뷰 모두에 적용된다. Pseudo-view 샘플링은 2,000 iteration 이후 활성화된다.

### 2.3 모델 구조

FSGS의 파이프라인은 다음과 같다:

1. **COLMAP SfM** → 극소수 뷰에서 희소 포인트 클라우드 추출
2. **3D Gaussian 초기화** (SH degree 0, opacity 0.1)
3. **학습 루프** (총 10,000 iteration):
   - 매 100 iteration마다 densification 수행 (500 iteration 이후 시작)
   - Proximity-guided Gaussian Unpooling: $P_i > t_{\text{prox}}=10$이면 새 Gaussian 삽입
   - 기존 3D-GS densification (gradient 기반)도 병행
   - 2,000 iteration 이후 pseudo-view 샘플링 활성화
   - Photometric loss + Depth correlation loss 역전파
4. **추론**: 최적화된 Gaussian 집합으로 실시간 렌더링 (200+ FPS)

### 2.4 성능 향상

| 데이터셋 | 메트릭 | 3D-GS | SparseNeRF | **FSGS** | 향상 (vs. SparseNeRF) |
|---------|--------|-------|------------|---------|---------------------|
| LLFF (3-view, 1/8 res) | PSNR | 17.43 | 19.86 | **20.31** | +0.45 dB |
| LLFF (3-view, 1/8 res) | SSIM | 0.522 | 0.624 | **0.652** | +0.028 |
| LLFF (3-view, 1/8 res) | FPS | 385 | 0.21 | **458** | **2,180×** |
| Mip-NeRF360 (24-view, 1/8 res) | PSNR | 20.89 | 22.85 | **23.70** | +0.85 dB |
| Mip-NeRF360 (24-view, 1/8 res) | SSIM | 0.633 | 0.693 | **0.745** | +0.052 |
| Blender (8-view) | PSNR | 21.56 | 24.04 | **24.64** | +0.60 dB |
| Shiny (3-view) | PSNR | 17.83 | 18.81 | **19.63** | +0.82 dB |

**핵심 포인트**: FSGS는 3D-GS보다 **더 적은 Gaussian 수**(LLFF: 57,513 vs. 63,219)로 더 높은 품질과 빠른 FPS를 달성한다.

### 2.5 한계점

- **가려진 영역(occluded views)에 대한 일반화 불가**: 학습 시 전혀 관측되지 않은 영역은 복원할 수 없다.
- SfM(COLMAP)에 의존하므로, **카메라 포즈 추정 실패 시** 전체 파이프라인이 영향을 받는다.
- Pseudo-view 생성이 학습 카메라 사이의 보간에 한정되어, **큰 시점 변화**에 대한 커버리지가 부족하다.

---

## 3. 모델의 일반화 성능 향상 가능성

FSGS의 일반화 성능 향상과 관련된 핵심 메커니즘들을 심층 분석한다.

### 3.1 Pseudo-view 기반 정규화의 일반화 효과

FSGS의 가장 핵심적인 일반화 전략은 **학습 중 미관측 시점을 합성**하고, 해당 시점에서 깊이 정규화를 적용하는 것이다. 이는:

- 과적합 방지: 소수 학습 뷰에만 최적화되는 것을 방지
- 기하학적 일관성 강화: 학습 뷰 사이의 공간에서도 합리적인 기하학을 유지
- Ablation 결과(Table 4): Pseudo-view 추가 시 PSNR 19.83→20.31, SSIM 0.634→0.652

### 3.2 상대 깊이 정규화의 일반화 기여

Pearson 상관계수 기반 **상대적(relative) 깊이 정규화**는:
- 절대 깊이의 스케일 불일치를 우회하여 **다양한 장면 스케일에 강건**
- 다양한 depth estimator(MiDaS small, DPT Hybrid, DPT Large, DepthAnything)에 대해 강건함을 실험적으로 검증 (Table 5)
- 최소 PSNR 20.17(MiDaS small) ~ 최대 20.37(DepthAnything)으로, 모든 경우 baseline 대비 큰 폭 개선

### 3.3 다양한 데이터셋에 대한 교차 일반화

FSGS는 **4가지 이질적 데이터셋**에서 일관된 SOTA 성능을 보여, 높은 교차 일반화 능력을 입증한다:
- **NeRF-Synthetic (Blender)**: 객체 중심 합성 데이터
- **LLFF**: 전방향(forward-facing) 실세계 장면
- **Shiny**: 복잡한 반사/굴절 효과
- **Mip-NeRF360**: 비바운드(unbounded) 실내외 장면
- **자체 수집 모바일 데이터**: iPhone 15 Pro로 촬영한 실제 장면 (Table 7)

### 3.4 일반화 성능 향상을 위한 향후 방향

1. **Diffusion 기반 뷰 합성과의 결합**: ReconFusion처럼 diffusion model로 더 다양한 pseudo-view를 생성하되, 3D-GS의 실시간 특성을 유지하는 하이브리드 접근
2. **Feed-forward 일반화 모델 통합**: pixelSplat, MVSplat 등 피드포워드 Gaussian 예측 모델의 초기화와 결합
3. **더 강력한 foundation depth model 활용**: Depth Anything V2 등의 최신 깊이 추정 모델 적용
4. **가려진 영역 복원**: 생성 모델을 통한 가려진 영역의 hallucination과 기하학적 일관성의 균형

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 연구 영향

1. **3D-GS의 few-shot 확장 가능성 입증**: 이전까지 3D-GS는 밀집 뷰 전용으로 인식되었으나, FSGS는 적절한 densification 전략과 정규화만으로 소수 뷰에서도 경쟁력 있는 결과를 달성할 수 있음을 보여주었다.
2. **실시간성과 품질의 동시 달성**: NeRF 기반 few-shot 방법들의 가장 큰 단점인 느린 렌더링(0.07~0.21 FPS)을 완전히 극복하여, VR/AR, 자율주행 등 **실용적 응용**의 문을 열었다.
3. **Point-based representation의 few-shot 최적화 패러다임 확립**: Gaussian Unpooling이라는 새로운 densification 패러다임을 제시하여, 이후 DNGaussian, SparseGS 등 다수의 후속 연구에 영감을 제공하였다.

### 4.2 앞으로 연구 시 고려할 점

| 고려사항 | 설명 |
|---------|------|
| **COLMAP 의존성 탈피** | SfM 없이도 동작하는 포즈-프리 few-shot GS 연구 필요 |
| **단일 이미지 기반 3D 복원** | 1-shot이나 zero-shot 시나리오로의 확장 |
| **동적 장면 확장** | 정적 장면 가정을 넘어 동적 객체가 포함된 few-shot 합성 |
| **대규모 장면 스케일링** | 도시 규모 장면에서의 few-shot 3D-GS |
| **깊이 prior의 신뢰도** | 모노큘러 깊이 추정의 오류가 결과에 미치는 영향과 이를 보정하는 메커니즘 |
| **가려진 영역 처리** | 논문에서도 인정한 한계: 완전히 미관측된 영역에 대한 처리 전략 필요 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 표현 방식 | 핵심 전략 | Few-shot 성능 (LLFF 3-view PSNR) | 추론 속도 |
|------|------|---------|---------|------|---------|
| **DietNeRF** [Jain et al., ICCV 2021] | 2021 | NeRF (MLP) | CLIP 임베딩 기반 의미적 일관성 | 14.94 | ~0.14 FPS |
| **RegNeRF** [Niemeyer et al., CVPR 2022] | 2022 | NeRF (MLP) | 깊이 스무스니스 정규화 + 패치 기반 정규화 | 19.08 | ~0.21 FPS |
| **FreeNeRF** [Yang et al., CVPR 2023] | 2023 | NeRF (MLP) | 주파수 어닐링 (frequency regularization) | 19.63 | ~0.21 FPS |
| **SparseNeRF** [Wang et al., ICCV 2023] | 2023 | NeRF (MLP) | 단안 깊이의 공간 연속성 loss | 19.86 | ~0.21 FPS |
| **3D-GS** [Kerbl et al., SIGGRAPH 2023] | 2023 | 3D Gaussian | Gradient 기반 densification (밀집 뷰 대상) | 17.43 | ~385 FPS |
| **ReconFusion** [Wu et al., CVPR 2024] | 2023 | Zip-NeRF | Diffusion model로 추가 뷰 합성 | - | 매우 느림 |
| **FSGS** [Zhu et al., ECCV 2024] | 2024 | 3D Gaussian | Proximity Unpooling + Pseudo-view + Depth regularization | **20.31** | **~458 FPS** |
| **DNGaussian** [Li et al., CVPR 2024] | 2024 | 3D Gaussian | Hard/soft depth regularization | 비교 가능 수준 | 실시간 |
| **pixelSplat** [Charatan et al., CVPR 2024] | 2024 | 3D Gaussian | Feed-forward 방식, 2-view 입력 | cross-scene 일반화 | 실시간 |
| **MVSplat** [Chen et al., ECCV 2024] | 2024 | 3D Gaussian | Cost volume 기반 feed-forward Gaussian 예측 | cross-scene 일반화 | 실시간 |

### 주요 트렌드 분석

1. **NeRF → 3D Gaussian Splatting으로의 전환**: 2023년 3D-GS 등장 이후, few-shot NVS 분야도 급속히 Gaussian 기반으로 전환. FSGS는 이 전환의 선두 주자.

2. **Per-scene 최적화 vs. Feed-forward 일반화**: FSGS는 per-scene 최적화 방식이나, pixelSplat/MVSplat 등은 대규모 데이터로 사전학습 후 새로운 장면에 feed-forward로 Gaussian을 예측. 두 패러다임의 장단점이 존재:
   - Per-scene (FSGS): 높은 품질, 학습 시간 필요 (~10분)
   - Feed-forward: 즉각적 추론, 학습 데이터 분포에 의존

3. **Depth prior의 보편화**: DPT, DepthAnything 등 foundation depth model의 발전으로, 거의 모든 최신 few-shot 방법이 깊이 prior를 활용. FSGS의 Pearson correlation 기반 상대 깊이 loss는 이 흐름의 효과적 구현.

4. **Diffusion model과의 결합**: ReconFusion 등이 시도한 diffusion 기반 뷰 합성은 강력하나 느리고 뷰 일관성이 보장되지 않음. FSGS의 경량 pseudo-view 전략은 이에 대한 효율적 대안.

---

## 참고자료

1. Zhu, Z., Fan, Z., Jiang, Y., Wang, Z. "FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting." *arXiv:2312.00451v2*, ECCV 2024.
2. Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G. "3D Gaussian Splatting for Real-Time Radiance Field Rendering." *ACM Transactions on Graphics (SIGGRAPH)*, 2023.
3. Yang, J., Pavone, M., Wang, Y. "FreeNeRF: Improving Few-shot Neural Rendering with Free Frequency Regularization." *CVPR*, 2023.
4. Wang, G., Chen, Z., Loy, C.C., Liu, Z. "SparseNeRF: Distilling Depth Ranking for Few-shot Novel View Synthesis." *ICCV*, 2023.
5. Niemeyer, M., Barron, J.T., et al. "RegNeRF: Regularizing Neural Radiance Fields for View Synthesis from Sparse Inputs." *CVPR*, 2022.
6. Jain, A., Tancik, M., Abbeel, P. "Putting NeRF on a Diet: Semantically Consistent Few-shot View Synthesis." *ICCV*, 2021.
7. Ranftl, R., Bochkovskiy, A., Koltun, V. "Vision Transformers for Dense Prediction." *ICCV*, 2021.
8. Wu, R., Mildenhall, B., et al. "ReconFusion: 3D Reconstruction with Diffusion Priors." *CVPR*, 2024.
9. Yang, L., Kang, B., et al. "Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data." *CVPR*, 2024.
10. Charatan, D., Li, S., Tagliasacchi, A., Sitzmann, V. "pixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction." *CVPR*, 2024.
11. Chen, Y., et al. "MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images." *ECCV*, 2024.
12. FSGS 프로젝트 페이지: https://zehaozhu.github.io/FSGS/
