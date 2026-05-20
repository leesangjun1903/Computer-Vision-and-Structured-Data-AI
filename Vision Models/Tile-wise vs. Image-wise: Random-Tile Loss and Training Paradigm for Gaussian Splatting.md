# Tile-wise vs. Image-wise: Random-Tile Loss and Training Paradigm for Gaussian Splatting

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

기존 3D Gaussian Splatting(3DGS)의 **image-wise 학습 패러다임**은 단일 최적화 스텝에서 다중 시점(multi-view)의 제약을 통합하지 못한다는 근본적 한계를 가진다. 이를 해결하기 위해 **Tile 단위 무작위 샘플링 기반 학습 패러다임(RT-Loss)**을 제안하여 수렴 효율성과 재구성 품질을 동시에 향상시킬 수 있다.

### 주요 기여

1. **RT-Loss (Random-Tile Loss)**: 3DGS 최초로 image-wise 학습을 tile-wise 학습으로 대체하는 방법론 제안
2. **Tile-based 학습 패러다임**: 매 반복(iteration)마다 다중 시점의 tile을 동시에 샘플링하여 멀티뷰 제약 통합
3. **Tile-based Adaptive Density Control**: 멀티뷰 그래디언트를 고려한 Gaussian 분할/복제 전략 재설계
4. 다양한 벤치마크(정적/동적 장면 모두)에서 3DGS 대비 성능 향상을 모델 컴팩트성 유지와 함께 달성

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

**Image-wise 학습의 한계:**

기존 3DGS의 손실 함수:

$$\mathcal{L}(\Theta) = (1 - \lambda)L_{\text{MAE}}(\Theta, \mathcal{I}) + \lambda L_{\text{D-SSIM}}(\Theta, \mathcal{I}) \tag{1}$$

- 한 번의 최적화 스텝에서 **단일 이미지 $\mathcal{I}$만** 사용
- 서로 다른 시점의 그래디언트가 동시에 반영되지 않음
- 훈련 뷰 수가 많을수록(수백~수천 장) 비효율적 수렴

NeRF의 pixel-wise 손실과 비교:

$$\mathcal{L}(\Theta) = \frac{1}{\|\mathcal{R}\|} \sum_{r \in \mathcal{R}} L_{\text{MSE}}(\Theta, r) \tag{2}$$

- NeRF는 무작위 ray 샘플링으로 멀티뷰 제약을 통합하지만, 구조적 유사성(SSIM) 활용이 어려움
- 3DGS는 Tile 기반 래스터화를 사용하므로 ray-wise 최적화와 다른 접근이 필요

**핵심 문제:** 멀티뷰 제약 + 구조적 유사성 제약을 **동시에, 효율적으로** 통합하는 방법의 부재

---

### 2.2 제안하는 방법 (수식 포함)

#### (A) RT-Loss 기본 구조

$N$개의 tile을 $V$개의 서로 다른 시점에서 샘플링하여 배치 $\mathcal{T}$를 구성:

**RT-MAE 색상 손실:**

$$L_{\text{RT-MAE}} = \frac{1}{\|\mathcal{T}\|} \sum_{t \in \mathcal{T}} \|\hat{C}(t) - C(t)\|_1 \tag{8}$$

여기서 $\hat{C}(t)$는 렌더링된 tile, $C(t)$는 GT tile

**RT-SSIM 구조 손실:**

다중 시점의 독립적이고 비중첩 tile들에 대해 SSIM을 평균:

$$\text{RT-SSIM}(\hat{\mathcal{T}}, \mathcal{T}) = \frac{1}{N} \sum_{t \in \mathcal{T}, v \in \mathcal{V}} \text{SSIM}\left(t^{(v)}(\hat{C}),\ t^{(v)}(C)\right) \tag{9}$$

- $\mathcal{V}$: 샘플링된 시점의 집합 ($|\mathcal{V}| = V$)
- 커널 크기 $K \times K$ (전체 구조: $9 \times 9$, 세부 구조: $3 \times 3$)

$$L_{\text{RT-SSIM}}(\Theta, \mathcal{T}) = 1 - \text{RT-SSIM}(\hat{\mathcal{T}}, \mathcal{T}) \tag{10}$$

**최종 하이브리드 손실:**

$$\mathcal{L}_{\text{RT}}(\Theta, \mathcal{T}) = (1 - \lambda_s) L_{\text{RT-MAE}} + \lambda_s L_{\text{RT-SSIM}} \tag{11}$$

- $\lambda_s = 0.2$ (색상 손실과 구조 손실의 균형 조절)
- $V = 5$ (기본 설정: 5개 시점에서 tile 샘플링)

#### (B) 참고: 3DGS Gaussian 정의 및 렌더링

각 Gaussian의 공간 분포:

$$G(x) = \exp\left(-\frac{1}{2}(x - \mu)^T \Sigma^{-1}(x - \mu)\right) \tag{4}$$

$$\Sigma = RSS^T R^T$$

픽셀 색상 합성 (Alpha-blending):

$$C(x') = \sum_{i \in \mathcal{N}} c_i \sigma_i \prod_{j=1}^{i-1}(1 - \sigma_j), \quad \sigma_i = \alpha_i G'_i(x') \tag{5}$$

---

### 2.3 모델 구조

```
[학습 데이터셋 D]
        ↓
[Tile 단위로 데이터 로드]
        ↓
[V개의 시점에서 균등 무작위 샘플링 → N개의 tile 배치 T 구성]
        ↓
[각 tile에 대해 VisibilityCheck() → 병렬 Forward Splatting]
        ↓
[렌더링 결과 Ĉ(t) 및 GT C(t) 획득]
        ↓
[RT-MAE 계산: tile별 L1 손실 평균]
[RT-SSIM 계산: 시점별 tile 패치에 SSIM 적용 후 평균]
        ↓
[최종 RT-Loss = (1-λs)·RT-MAE + λs·RT-SSIM]
        ↓
[그래디언트 역전파 → Gaussian 파라미터 Θ 업데이트]
        ↓
[Tile-based Adaptive Density Control]
```

#### Tile-based Adaptive Density Control

**Tile-Count Densification:**

$$\frac{\sum_{k=1}^{\text{Iter}} \sum_{m=1}^{M_k} \|g_{km}\|}{\sum_{k=1}^{\text{Iter}} M_k} > \tau_{\text{densify}} \tag{12}$$

**Iteration-Count Densification (기본값):**

$$\frac{\sum_{k=1}^{\text{Iter}} \sum_{m=1}^{M_k} (\|g_{km}\| \times r_m)}{\sum_{k=1}^{\text{Iter}} 1} > \tau_{\text{densify}} \tag{13}$$

- $M_k$: 반복 $k$에서 Gaussian이 관찰된 tile 수
- $r_m$: Gaussian 투영 면적 비율 ($S_1/S_2$)
- Iteration-Count가 기본값인 이유: 원래 3DGS의 분할 파라미터($\tau_{\text{densify}} = 0.0002$)와의 일관성 유지

---

### 2.4 성능 향상

| 데이터셋 | 지표 | 3DGS | Scaffold-GS | **Ours** | **Ours+Scaffold-GS** |
|---------|------|------|-------------|----------|----------------------|
| Mip-NeRF360 | PSNR↑ | 28.69 | 28.84 | **29.66** | **29.86** |
| Mip-NeRF360 | SSIM↑ | 0.870 | 0.848 | **0.892** | 0.885 |
| Mip-NeRF360 | LPIPS↓ | 0.182 | 0.220 | **0.171** | 0.206 |
| Tanks&Temples | PSNR↑ | 24.36 | 24.81 | **25.13** | **25.32** |
| Deep Blending | PSNR↑ | 29.41 | 30.21 | 30.15 | **30.49** |

**추가 효율성 비교 (KITCHEN 씬):**
- 3DGS: PSNR 31.63, Mem 372MB, FPS 118
- **Ours**: PSNR 32.35, Mem **320MB**, FPS **135** (메모리 감소 + 속도 향상 + 품질 향상)

**대규모 씬 수렴 성능 (SMERF 데이터셋, 4k iter 기준):**
- Berlin(1511 views): 3DGS 22.63 → **Ours 24.80** (+2.17 dB)

**동적 장면 (Neural 3D Video Dataset):**
- 4D-GS: PSNR 31.15, FPS 30, Storage 90MB
- **Ours**: PSNR **31.74**, FPS **34**, Storage **78MB**

---

### 2.5 한계점

1. **Texture-less 영역**: 대규모 texture가 없는 영역에서 기하 구조 개선이 제한적
2. **하이퍼파라미터 의존성**: $\lambda_s$, $V$, 커널 크기 $K$ 등의 튜닝 필요
3. **멀티뷰 제약 할당의 비균일성**: 동일한 콘텐츠에 멀티뷰 제약을 더 정밀하게 배분하는 전략 미흡
4. **SfM 초기화 개선의 한계**: 랜덤 초기화 시 성능 향상에도 불구, 여전히 SfM 기반보다 열세

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 멀티뷰 제약을 통한 일반화

RT-Loss의 핵심 메커니즘은 각 최적화 스텝에서 **서로 다른 시점의 tile을 동시에 샘플링**하는 것이다. 이는 다음과 같은 일반화 효과를 가져온다:

**그래디언트 다양성 증가:**
- Image-wise 학습: 단일 시점의 그래디언트만 반영 → 특정 시점에 과적합(overfitting) 위험
- RT-Loss: $V$개 시점의 그래디언트가 동시에 반영 → 장면 전체에 대한 균형 잡힌 최적화

$$\nabla_\Theta \mathcal{L}_{\text{RT}} = \frac{1}{\|\mathcal{T}\|}\sum_{t \in \mathcal{T}} \nabla_\Theta \ell(t) \approx \mathbb{E}_{t \sim p(\mathcal{T})}[\nabla_\Theta \ell(t)]$$

이는 전체 훈련 분포에 대한 기댓값에 더 가까운 그래디언트 추정을 의미한다.

### 3.2 초기화 강건성 (Robustness to Initialization)

**실험 결과 (Tab. 8):**

| 조건 | 3DGS PSNR | Ours PSNR | 차이 |
|------|----------|-----------|------|
| SfM 초기화 (Kitchen) | 31.63 | 32.35 | +0.72 |
| **랜덤 초기화** (Kitchen) | 29.89 | **31.22** | **+1.33** |
| SfM 초기화 (Courthouse) | 23.13 | 24.14 | +1.01 |
| **랜덤 초기화** (Courthouse) | 21.77 | **23.21** | **+1.44** |

→ SfM 초기화가 없을 때 RT-Loss의 이점이 더욱 크게 나타남. 이는 **불균일하고 불충분한 훈련 제약**이 초기화 문제의 일부 원인임을 시사하며, RT-Loss가 이를 완화할 수 있음을 보여준다.

### 3.3 훈련 뷰 수 증가에 따른 스케일 일반화

대규모 장면(훈련 뷰 1000개 이상)에서 특히 뚜렷한 일반화 효과:

- Image-wise 학습: 훈련 뷰가 많을수록 한 번의 iteration에서 커버할 수 있는 시점 비율↓ → 수렴 저하
- RT-Loss: 뷰 수에 관계없이 $V$개 시점을 균등 샘플링 → **스케일에 무관한 안정적 수렴**

Fig. 4에서 Berlin(1511 views) 씬에서 수렴 gap이 Kitchen(279 views)보다 훨씬 크게 나타나는 것이 이를 뒷받침한다.

### 3.4 동적 장면으로의 일반화

4D 재구성(시공간 차원)에서도 RT-Loss는 효과적이며, 시간 축을 추가적인 "시점"으로 간주하여 멀티뷰+시간적 제약을 동시에 처리할 수 있음을 실험적으로 검증하였다.

### 3.5 타 모델 아키텍처로의 이전 가능성

Scaffold-GS에 tile-wise 방법을 적용했을 때도 일관된 성능 향상이 관찰됨 (Mip-NeRF360: 28.84 → 29.86 PSNR). 이는 RT-Loss가 특정 아키텍처에 국한되지 않고 **Gaussian Splatting 계열 전반에 적용 가능한 일반적 방법론**임을 시사한다.

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 연구에 미치는 영향

**① 학습 패러다임의 패러다임 전환**

기존 3DGS 계열 연구는 image-wise 학습을 당연한 기본값으로 수용해 왔다. 이 논문은 tile이 splatting 렌더링의 최소 단위라는 점에 착안하여, **학습 단위를 이미지에서 tile로 전환**하는 새로운 관점을 제시한다. 이는 이후 3DGS 기반 연구에서 학습 파이프라인 설계 시 tile-wise 접근법을 표준 고려사항으로 만들 가능성이 높다.

**② Sparse View 합성 연구에의 영향**

훈련 뷰가 적은 환경에서 멀티뷰 제약의 중요성은 더욱 크다. RT-Loss의 철학은 Sparse View 3DGS 연구(SparseGS, MVSplat 등)에서 멀티뷰 일관성을 강화하는 방향으로 자연스럽게 확장될 수 있다.

**③ 4D 및 동적 장면 재구성**

시간 축을 추가적 "시점"으로 해석하는 프레임워크는 동적 장면 재구성에서 시공간 일관성을 향상시키는 연구 방향을 제시한다.

**④ S3IM [42]과의 연계**

NeRF에서 랜덤 ray 패치에 SSIM을 적용한 S3IM의 아이디어를 3DGS의 tile 구조에 맞게 재해석한 것으로, **NeRF와 3DGS 사이의 학습 전략 교차 적용** 연구를 촉진할 수 있다.

### 4.2 앞으로 연구 시 고려할 점

**① Tile 샘플링 전략의 고도화**

현재는 균등 무작위 샘플링을 사용하지만, 다음을 고려할 수 있다:
- **불확실성 기반 샘플링**: 재구성 오차가 높은 영역의 tile을 우선 샘플링
- **콘텐츠 인식 샘플링**: 동일한 3D 콘텐츠를 다루는 tile들을 매칭하여 더 강한 멀티뷰 제약 부여 (논문에서도 한계로 언급)
- **커리큘럼 학습**: 초기에는 넓은 시점 분포, 후기에는 유사 시점 집중

**② Texture-less 영역 처리**

논문이 명시한 한계인 texture-less 영역에서의 기하 구조 개선을 위해:
- 기하학적 정규화 항 추가 (법선 일관성, 깊이 일관성)
- 깊이 추정 모델과의 결합

**③ 하이퍼파라미터 적응형 조절**

$\lambda_s$, $V$, 커널 크기 $K$를 훈련 진행도나 장면 복잡도에 따라 **동적으로 조절**하는 방법 연구가 필요하다.

**④ 메모리 효율성**

$V$개 시점에서 동시에 렌더링하면 GPU 메모리 요구량이 증가할 수 있다. 특히 고해상도 장면이나 제한된 GPU 환경에서 메모리 효율적인 tile 배치 방법 연구가 필요하다.

**⑤ 초기화 독립성과의 결합**

RT-Loss가 랜덤 초기화에 대한 강건성을 향상시킨다는 결과는, **SfM 없이 3DGS를 학습하는 연구** (예: RAIN-GS)와의 결합 가능성을 제시한다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 발표 | 핵심 기여 | 학습 패러다임 | 멀티뷰 제약 | 구조 손실 |
|------|------|----------|--------------|------------|---------|
| **NeRF** [33] | 2020 | MLP 기반 체적 렌더링 | Ray-wise | ✅ (암묵적) | ❌ |
| **Instant-NGP** [34] | 2022 | 해시 인코딩 기반 고속화 | Ray-wise | ✅ | ❌ |
| **Mip-NeRF360** [3] | 2022 | 무한 장면 anti-aliasing | Ray-wise | ✅ | ❌ |
| **3DGS** [19] | 2023 | Tile 기반 가우시안 래스터화 | Image-wise | ❌ | SSIM (단일뷰) |
| **S3IM** [42] | 2023 | NeRF에 랜덤 패치 SSIM 적용 | Patch-wise (ray) | ✅ | ✅ |
| **Scaffold-GS** [31] | 2024 | 신경망 필드 + 가우시안 결합 | Image-wise | ❌ | SSIM (단일뷰) |
| **4D-GS** [10] | 2024 | 동적 장면 4D 가우시안 | Image-wise (batch) | 부분적 | SSIM (단일뷰) |
| **MVGS** [8] | 2024 | 멀티뷰 배치 그래디언트 누적 | Image batch-wise | ✅ | SSIM (단일뷰) |
| **RT-Loss (본 논문)** | 2025 | Tile-wise 멀티뷰 학습 | **Tile-wise** | ✅ | **RT-SSIM (멀티뷰)** |

### 핵심 차별점 분석

**vs. S3IM [42]:**
- S3IM은 NeRF의 ray에 SSIM 패치를 적용하는 반면, RT-Loss는 3DGS의 tile 구조에 맞게 재설계
- 3DGS의 tile-based 래스터화는 ray marching보다 구조적 정보 포함에 더 적합한 최소 단위를 제공

**vs. MVGS [8]:**
- MVGS는 이미지 단위로 다중 이미지를 배치로 묶어 그래디언트 누적
- RT-Loss는 tile 단위 샘플링으로 **동일한 배치 크기에서** 더 다양한 시점 커버 가능
- 논문의 Tab. 7에서 image-wise batch($B=2,4$) 대비 tile-wise가 일관되게 우수함을 실증

**vs. 4D-GS [10]:**
- 4D-GS는 시간 차원 처리를 위해 배치 이미지 학습 + 그래디언트 누적 사용
- RT-Loss를 4D-GS에 적용 시 PSNR 31.15 → 31.74, FPS 30 → 34, Storage 90 → 78MB로 전방위 개선

---

## 참고 자료

**본 답변에서 직접 참고한 자료:**

- **Zhang, X., Pan, W., et al.** "Tile-wise vs. Image-wise: Random-Tile Loss and Training Paradigm for Gaussian Splatting." *ICCV 2025*. (Computer Vision Foundation Open Access 제공 PDF)

**논문 내에서 인용된 핵심 관련 연구:**

- [19] Kerbl, B., et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering." *ACM Trans. Graph.*, 2023.
- [33] Mildenhall, B., et al. "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis." *Communications of the ACM*, 2021.
- [42] Xie, Z., et al. "S3IM: Stochastic Structural Similarity and Its Unreasonable Effectiveness for Neural Fields." 2023.
- [31] Lu, T., et al. "Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering." *CVPR 2024*.
- [10] Duan, Y., et al. "4D Gaussian Splatting: Towards Efficient Novel View Synthesis for Dynamic Scenes." arXiv:2402.03307, 2024.
- [3] Barron, J.T., et al. "Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields." *CVPR 2022*.
- [8] Du, X., et al. "MVGS: Multi-View-Regulated Gaussian Splatting for Novel View Synthesis." 2024.
- [34] Müller, T., et al. "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding." *ACM TOG*, 2022.

> **정확도 관련 고지:** 본 답변은 제공된 논문 PDF를 직접 분석한 결과를 기반으로 하며, 논문에 명시된 수식, 수치, 실험 결과를 충실히 반영하였습니다. 논문에 명시되지 않은 내용에 대한 추론은 관련 연구와의 맥락 비교를 통해 제시하였으며, 이를 별도로 구분하였습니다.
