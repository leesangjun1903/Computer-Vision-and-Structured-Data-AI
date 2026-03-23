# Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering

---

## 1. 핵심 주장 및 주요 기여 요약

3D Gaussian Splatting(3D-GS)은 primitive 기반 표현과 volumetric 표현의 장점을 결합하여 최첨단 렌더링 품질과 속도를 달성했지만, 모든 학습 뷰에 맞추려는 과도하게 중복된 가우시안을 생성하여 장면 기하학을 무시하게 되며, 그 결과 뷰 변화, 텍스처 없는 영역, 조명 효과에 대해 로버스트하지 못한 모델이 된다.

이를 해결하기 위해, Scaffold-GS는 앵커 포인트를 사용하여 로컬 3D 가우시안을 분배하고, 뷰 프러스텀 내에서 시선 방향과 거리에 기반하여 속성을 즉석에서(on-the-fly) 예측한다. 이 방법은 더 빠르게 수렴하고, 더 적은 프리미티브를 사용하며, 더 나은 시각적 품질을 달성한다.

**주요 기여:**
- Scaffold-GS는 3D Gaussian Splatting에 대한 계층적이고 뷰 적응형(view-adaptive) 확장을 제시하여, 모델 저장 요구량을 크게 줄이면서 렌더링 품질을 향상시킨다.
- Neural Gaussian의 중요도에 기반한 앵커 성장(growing) 및 가지치기(pruning) 전략을 개발하여 장면 커버리지를 안정적으로 개선한다.
- Scaffold-GS는 CVPR 2024에서 Highlight 논문으로 선정되었다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 3D-GS는 모든 학습 뷰에 과적합(overfitting)하는 과다 중복 가우시안을 생성하여 장면의 기하학적 구조를 무시하게 되고, 그 결과 큰 뷰 변화, 텍스처가 없는 영역, 조명 효과에 취약하다.

구체적 한계:
- **뷰 의존적 아티팩트**: 반사, 투명도, 광택(specularity) 등의 표현 실패
- **중복 프리미티브**: 불필요한 가우시안이 과도하게 생성되어 메모리 낭비
- 다중 디테일 레벨에서 캡처된 장면에서 아티팩트 발생, 학습에 포함되지 않은 뷰잉 거리로의 외삽(extrapolation) 시 문제 발생

### 2.2 제안하는 방법 및 모델 구조

#### (a) 앵커 포인트 초기화 — Sparse Voxel Grid

SfM으로 도출된 점들로부터 희소 복셀 격자(sparse voxel grid)를 형성하고, 각 복셀의 중심에 학습 가능한 스케일(learnable scale)을 가진 앵커 포인트를 배치하여 장면의 점유(occupancy)를 대략적으로 표현한다.

각 앵커 $v$는 다음과 같은 속성으로 정의된다:

$$v = \{\mathbf{x}_v, \, \mathbf{f}_v, \, l_v, \, \mathbf{O}_k\}$$

여기서:
- $\mathbf{x}_v \in \mathbb{R}^3$: 앵커 위치 (복셀 중심)
- $\mathbf{f}_v \in \mathbb{R}^{d}$: 학습 가능한 앵커 피처 벡터
- $l_v \in \mathbb{R}^3$: 학습 가능한 스케일링 팩터
- $\mathbf{O}_k \in \mathbb{R}^{k \times 3}$: $k$개의 학습 가능한 오프셋

#### (b) Neural Gaussian 생성 — View-Adaptive Prediction

뷰 프러스텀 내에서 각 가시(visible) 앵커로부터 $k$개의 뉴럴 가우시안이 오프셋과 함께 생성되며, 이들의 속성(opacity, color, scale, quaternion)은 앵커 피처, 상대적 카메라-앵커 시선 방향, 거리를 사용하여 MLP로 디코딩된다.

각 앵커 $v$에서 $i$번째 뉴럴 가우시안의 중심 위치는 다음과 같이 결정된다:

$$\boldsymbol{\mu}_i = \mathbf{x}_v + \mathbf{O}_i \odot l_v, \quad i = 1, 2, \ldots, k$$

뷰 의존적 속성 예측을 위한 MLP 디코딩:

$$\{\alpha_i\} = F_{\alpha}\left(\mathbf{f}_v, \, \hat{\mathbf{d}}_v, \, \delta_v\right)$$

$$\{c_i\} = F_c\left(\mathbf{f}_v, \, \hat{\mathbf{d}}_v, \, \delta_v\right)$$

$$\{s_i\} = F_s\left(\mathbf{f}_v, \, \hat{\mathbf{d}}_v, \, \delta_v\right)$$

$$\{q_i\} = F_q\left(\mathbf{f}_v, \, \hat{\mathbf{d}}_v, \, \delta_v\right)$$

여기서:
- $F_{\alpha}, F_c, F_s, F_q$: 각각 opacity, color, scale, quaternion 예측 MLP
- $\hat{\mathbf{d}}\_v = \frac{\mathbf{x}\_v - \mathbf{x}\_{cam}}{\|\mathbf{x}\_v - \mathbf{x}_{cam}\|}$: 정규화된 시선 방향 벡터
- $\delta_v = \|\mathbf{x}\_v - \mathbf{x}_{cam}\|$: 카메라-앵커 간 거리

#### (c) 렌더링 — Alpha Blending

예측된 opacity가 임계값 $\tau_\alpha$보다 큰 뉴럴 가우시안만 래스터화에 사용되어 계산 효율을 높인다. 최종 픽셀 색상은 표준 alpha-blending으로 계산된다:

$$C(\mathbf{p}) = \sum_{i \in \mathcal{N}} c_i \, \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)$$

#### (d) 손실 함수 (Loss Function)

학습은 다음의 복합 손실 함수로 수행된다:

$$\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{\text{SSIM}} + \beta \mathcal{L}_{\text{vol}}$$

여기서:
- $\mathcal{L}_1$: 렌더링 이미지와 ground truth 간의 L1 손실
- $\mathcal{L}_{\text{SSIM}}$: 구조적 유사도 손실
- $\mathcal{L}_{\text{vol}}$: 볼륨 정규화 항 (앵커 스케일과 뉴럴 가우시안 크기를 억제)

#### (e) 앵커 정제 전략

**성장(Growing) 연산:**
오류 기반 앵커 성장 정책으로, 뉴럴 가우시안이 중요하다고 판단하는 곳에 새로운 앵커를 성장시킨다. 뉴럴 가우시안을 다중 해상도 복셀로 양자화하고, 레벨별 임계값보다 큰 그래디언트를 가진 복셀에 새 앵커를 추가한다.

다중 해상도 성장 임계값:

$$\epsilon_g^{(m)} = \frac{\epsilon_g}{4^{m-1}}, \quad \tau_g^{(m)} = \tau_g \cdot 2^{m-1}$$

여기서 $m$은 양자화 레벨을 나타낸다.

**가지치기(Pruning) 연산:**
랜덤 제거와 opacity 기반 가지치기가 앵커 성장 및 정제를 조절한다. $N$번의 학습 반복 동안 누적 opacity가 일정 수준 이하인 앵커는 제거된다.

### 2.3 정량적 성능 비교

NerfBaselines 벤치마크(Mip-NeRF 360 데이터셋)에서의 결과:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | 학습 시간 | GPU 메모리 |
|--------|-------|-------|--------|----------|-----------|
| **Zip-NeRF** | 28.553 | 0.829 | 0.218 | 5h 30m | 26.8 GB |
| **3DGS-MCMC** | 27.983 | 0.835 | 0.224 | 41m | 28.9 GB |
| **Scaffold-GS** | **27.714** | **0.813** | **0.262** | **23m 28s** | **8.7 GB** |
| Mip-NeRF 360 | 27.681 | 0.792 | 0.272 | 30h 14m | 33.6 GB |
| Gaussian Splatting | 27.434 | 0.814 | 0.257 | 23m 25s | 11.1 GB |

Scaffold-GS는 PSNR 27.714, SSIM 0.813을 달성하며, 학습 시간 23분 28초에 GPU 메모리 8.7 GB만을 사용한다.

3D-HGS 논문(CVPR 2025)에서 보고한 다중 데이터셋 비교:

| Dataset | Scaffold-GS PSNR | 3D-GS PSNR | Scaffold-GS 대비 |
|---------|------------------|------------|----------------|
| Mip-NeRF360 | 28.95 | 28.88 | +0.07 |
| Tanks&Temples | 23.96 | 23.60 | +0.36 |
| Deep Blending | 30.21 | 29.41 | +0.80 |

Scaffold-GS는 더 컴팩트한 모델로 3D-GS와 동등하거나 우수한 렌더링 품질과 속도를 달성한다.

### 2.4 한계점

1. **SfM 초기화 의존성**: SfM 포인트 클라우드로부터의 초기화는 빠르고 실용적이지만, 텍스처가 없는 넓은 영역이 지배적인 시나리오에서는 최적이 아닐 수 있다. 앵커 정제 전략이 이를 어느 정도 보완하지만, 극도로 희소한 포인트에서는 여전히 어려움이 있다.

2. **MLP 추론 오버헤드**: 뷰 의존적 속성을 MLP로 실시간 예측하므로, 매우 큰 장면에서는 추가 연산 비용이 발생한다.

3. **LPIPS 지표**: 벤치마크 결과에서 Scaffold-GS의 LPIPS는 다른 방법 대비 다소 높은 경향을 보여, perceptual quality 측면에서 개선 여지가 있다.

---

## 3. 일반화 성능 향상 가능성

Scaffold-GS의 일반화 성능 향상은 아래 핵심 메커니즘에 기인한다:

### 3.1 뷰-적응형 속성 예측
각 앵커는 학습 가능한 오프셋을 가진 뉴럴 가우시안 세트를 연결하고, 이들의 속성은 앵커 피처와 뷰잉 포지션에 기반하여 동적으로 예측된다. 이는 3D 가우시안이 자유롭게 표류하고 분할되는 기존 3D-GS와 달리 장면 구조를 활용하여 분포를 안내하고 제약하면서도 다양한 뷰 각도와 거리에 적응할 수 있게 한다.

이 메커니즘을 통해 학습 시 보지 못한 시점에서의 **보간(interpolation) 및 외삽(extrapolation)** 성능이 크게 향상된다:

$$\text{Attributes}_i = \text{MLP}\left(\mathbf{f}_v, \, \hat{\mathbf{d}}_v, \, \delta_v\right) \quad \forall \, (\hat{\mathbf{d}}_v, \, \delta_v) \in \text{novel views}$$

### 3.2 계층적 장면 표현
각 SfM 포인트를 개별 가우시안 중심으로 다루는 대신, 장면 공간을 복셀화하고 점유된 복셀 중심에 앵커를 생성하는 2단계 계층적 표현을 도입한다.

이 구조화된 접근은:
- 중복 가우시안을 효과적으로 감소
- 장면 기하학에 대한 더 강건한 이해를 제공
- 다양한 디테일 수준(levels-of-detail)과 뷰 의존적 관찰을 수용하는 향상된 능력을 보여준다.

### 3.3 다중 해상도 피처 뱅킹
다중 해상도 피처 뱅킹 시스템은 뷰 의존적 요소로 가중치가 부여된 여러 스케일의 앵커 피처를 혼합하여 다양한 뷰잉 거리에 적응할 수 있게 한다.

### 3.4 앵커 피처의 의미적 구조
클러스터링된 앵커 피처는 장면 콘텐츠의 단서를 보여주어, 3D-GS 모델의 해석 가능성(interpretability)을 향상시키고, 재사용 가능한 피처를 활용한 훨씬 더 큰 장면으로의 확장 가능성을 가진다.

### 3.5 도전적 시나리오에서의 우수성
Scaffold-GS는 반사(reflection), 그림자(shadowing) 등 뷰 의존적 효과에 더 로버스트하며, 중복 3D 가우시안으로 인한 아티팩트(floater, 구조 오류)를 완화한다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구적 영향

1. **구조화된 가우시안 표현의 패러다임 확립**: 구조화된 접근이 품질을 희생하지 않으면서 더 나은 효율성을 달성할 수 있음을 보여, 향후 연구가 계층적이고 적응적인 전략을 더 탐구하도록 방향을 제시한다.

2. **후속 연구의 기반 프레임워크**: Octree-AnyGS와 같은 일반적인 앵커 기반 프레임워크가 등장하여 명시적 가우시안(2D-GS, 3D-GS)과 뉴럴 가우시안(Scaffold-GS) 모두를 지원하며, Level-of-Detail 표현으로 대규모 장면을 처리한다.

3. **압축 연구의 기반**: HAC 프레임워크와 같은 압축 기법이 Scaffold-GS를 기반 모델로 사용하여, 앵커 위치로 해시 그리드를 쿼리해 효율적인 엔트로피 코딩을 수행한다.

4. **장면 이해와의 결합 가능성**: 앵커 피처에서 발견된 암묵적 의미 구조는 뉴럴 렌더링과 장면 이해 태스크를 결합할 가능성을 시사하며, 더 지능적이고 조작 가능한 3D 표현을 가능하게 할 수 있다.

### 4.2 향후 연구 시 고려할 점

| 고려 사항 | 상세 내용 |
|----------|----------|
| **초기화 전략 개선** | SfM 의존성을 줄이기 위한 depth prior, diffusion model 기반 초기화 탐구 |
| **대규모 장면 확장성** | LOD 전략(Octree-GS 등)과의 통합을 통한 도시 규모 장면 지원 |
| **동적 장면 확장** | 4D Gaussian Splatting과의 결합으로 시간 축 확장 |
| **모바일/VR 배포** | MLP 추론 경량화 및 양자화를 통한 엣지 디바이스 배포 |
| **Perceptual Quality** | LPIPS 개선을 위한 perceptual loss 및 topology-aware 정규화 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연도 | 방법 | 핵심 아이디어 | Scaffold-GS 대비 특징 |
|------|------|-------------|---------------------|
| 2020 | **NeRF** (Mildenhall et al.) | 암묵적 볼륨 렌더링 | 고품질이나 실시간 불가 |
| 2022 | **Instant-NGP** (Müller et al.) | 해시 인코딩으로 NeRF 가속 | 학습 속도 향상, 여전히 암묵적 |
| 2023 | **3D-GS** (Kerbl et al., SIGGRAPH) | 명시적 가우시안 스플래팅 | Scaffold-GS가 직접 개선하고자 하는 핵심 기반 방법 |
| 2023 | **Mip-Splatting** | 안티앨리어싱 가우시안 | Scaffold-GS 대비 유사 PSNR, 멀티스케일 처리 |
| 2024 | **Scaffold-GS** (CVPR Highlight) | 앵커 기반 구조화 + 뷰 적응형 MLP | 본 논문 |
| 2024 | **2D-GS** (SIGGRAPH) | 2D 가우시안으로 기하 정확도 향상 | 표면 재구성에 강점, 뷰 적응성 부족 |
| 2024 | **3DGS-MCMC** | MCMC 기반 최적화 | PSNR 27.983로 Scaffold-GS(27.714) 대비 우위, 그러나 GPU 메모리 28.9 GB로 3배 이상 |
| 2024 | **Octree-GS** | 명시적 LOD 표현으로 대규모 장면에서 더 빠르게 렌더링하면서 높은 품질 유지 | Scaffold-GS의 앵커 개념을 LOD로 확장 |
| 2024 | **HAC/HAC++** | 학습된 분해와 계층을 통해 최대 100배 압축 추구 | Scaffold-GS 기반 압축 프레임워크 |
| 2024 | **Topology-GS** | Topology-GS는 Scaffold-GS 등 기존 방법을 일관되게 능가하며, LPVI와 PersLoss를 통합한다. Mip-NeRF360에서 PSNR 29.50 달성 |
| 2025 | **3D-HGS** (CVPR) | Half-Gaussian Splatting을 도입, 각 가우시안을 두 반으로 분리하여 추론 시간 증가 없이 정밀한 렌더링 제어를 가능하게 하는 plug-and-play 방법 | Scaffold-HGS로 +0.30 PSNR 향상 |
| 2025 | **FlashGS** (CVPR) | 래스터화 파이프라인을 재설계하여 4K 대규모 장면의 실시간 렌더링 달성 |
| 2025 | **Perceptual-GS** | 장면 적응적 지각 밀집화(densification) | SSIM, LPIPS에서 우수한 품질-효율 트레이드오프 달성 |

---

## 참고 자료 및 출처

1. **[공식 프로젝트 페이지]** Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering — https://city-super.github.io/scaffold-gs/
2. **[논문 원문 — CVPR 2024]** Lu, T., Yu, M., Xu, L., Xiangli, Y., Wang, L., Lin, D., & Dai, B. (2024). *Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering.* CVPR 2024, pp. 20654–20664 — https://openaccess.thecvf.com/content/CVPR2024/papers/Lu_Scaffold-GS_Structured_3D_Gaussians_for_View-Adaptive_Rendering_CVPR_2024_paper.pdf
3. **[arXiv]** arXiv:2312.00109 — https://arxiv.org/abs/2312.00109
4. **[ar5iv HTML 버전]** https://ar5iv.labs.arxiv.org/html/2312.00109
5. **[GitHub 공식 리포지토리]** https://github.com/city-super/Scaffold-GS
6. **[alphaXiv 분석]** https://www.alphaxiv.org/overview/2312.00109v1
7. **[NerfBaselines 벤치마크]** https://nerfbaselines.github.io/ (https://github.com/nerfbaselines/nerfbaselines)
8. **[3DGS Compression Survey]** https://w-m.github.io/3dgs-compression-survey/ (arXiv: 2407.09510)
9. **[3D-HGS: 3D Half-Gaussian Splatting]** Li et al. (CVPR 2025) — https://arxiv.org/html/2406.02720v3
10. **[Topology-GS]** Topology-Aware 3D Gaussian Splatting — https://liner.com/review/topologyaware-3d-gaussian-splatting
11. **[Emergent Mind 분석]** https://www.emergentmind.com/papers/2312.00109
12. **[Semantic Scholar]** https://www.semanticscholar.org/paper/Scaffold-GS-Lu-Yu/a294b8632fed59e7079ef6187b0afa532c97ed7f
13. **[Hugging Face Paper Page]** https://huggingface.co/papers/2312.00109
14. **[3DGS 기술 서베이]** *A review of recent advances in 3D Gaussian Splatting* — ScienceDirect (2024)
15. **[The Impact and Outlook of 3D Gaussian Splatting]** arXiv:2510.26694 (2025)

---

> **참고**: 위 분석의 수식은 논문 원문과 공개된 프로젝트 자료를 바탕으로 재구성한 것입니다. 세부 하이퍼파라미터 값(예: $\lambda$, $\beta$, $\tau_\alpha$, $k$ 등)은 논문 원문 PDF를 직접 참조하시기 바랍니다. 벤치마크 수치는 측정 조건(장면 구성, 해상도, 하드웨어)에 따라 달라질 수 있으므로, 정확한 비교는 동일 실험 환경에서의 재현을 권장합니다.
