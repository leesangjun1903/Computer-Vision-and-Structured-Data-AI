
# 2D Gaussian Splatting for Geometrically Accurate Radiance Fields

> **논문 정보**
> - **제목**: 2D Gaussian Splatting for Geometrically Accurate Radiance Fields
> - **저자**: Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, Shenghua Gao
> - **학회**: ACM SIGGRAPH 2024 Conference Papers
> - **arXiv**: [arXiv:2403.17888](https://arxiv.org/abs/2403.17888)
> - **DOI**: [10.1145/3641519.3657428](https://dl.acm.org/doi/10.1145/3641519.3657428)
> - **공식 프로젝트 페이지**: [surfsplatting.github.io](https://surfsplatting.github.io/)
> - **GitHub**: [hbb1/2d-gaussian-splatting](https://github.com/hbb1/2d-gaussian-splatting)

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 배경 및 핵심 주장

3D Gaussian Splatting (3DGS)는 최근 Radiance Field 재구성 분야를 혁신하여 고품질 Novel View Synthesis(NVS)와 빠른 렌더링 속도를 달성했다. 그러나 3DGS는 3D Gaussian의 멀티뷰 비일관성(multi-view inconsistency) 문제로 인해 정확한 표면 표현에 실패한다.

이에 본 논문은 2D Gaussian Splatting(2DGS)이라는 새로운 접근법을 제안하며, 멀티뷰 이미지로부터 기하학적으로 정확한 Radiance Field를 모델링하고 재구성한다. 핵심 아이디어는 3D 볼륨을 2D 방향성 평면 Gaussian 디스크(oriented planar Gaussian disks)의 집합으로 압축하는 것이다. 3D Gaussian과 달리, 2D Gaussian은 표면을 내재적으로 모델링하면서 뷰 일관성 있는 기하학(view-consistent geometry)을 제공한다.

### 1.2 주요 기여 (3가지)

| 기여 | 내용 |
|------|------|
| **차별화된 2D Gaussian 렌더러** | Perspective-correct splatting, ray-splat intersection 기반 |
| **두 가지 정규화 손실** | Depth Distortion Loss + Normal Consistency Loss |
| **SOTA 성능** | 명시적 표현(explicit representation) 중 최고 수준의 기하학적 재구성 및 NVS 결과 달성 |

2DGS는 고효율 차분 가능 2D Gaussian 렌더러를 제시하여, 2D 표면 모델링·ray-splat 교차·체적 적분(volumetric integration)을 활용한 perspective-correct splatting을 가능하게 한다. 또한 향상된 노이즈 없는 표면 재구성을 위한 두 가지 정규화 손실을 도입하며, 다른 명시적 표현 방법들과 비교하여 최고 수준의 기하학적 재구성 및 NVS 결과를 달성한다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

3D Gaussian 모델링 및 splatting을 사용한 표면 재구성은 다음과 같은 세 가지 핵심 과제를 안고 있다. **첫째**, 3D Gaussian의 체적 Radiance 표현은 표면의 얇은 특성(thin nature)과 충돌한다. **둘째**, 3DGS는 고품질 표면 재구성에 필수적인 표면 법선(surface normal)을 기본적으로 모델링하지 않는다. **셋째**, 3DGS의 래스터화 프로세스는 멀티뷰 일관성이 없어, 서로 다른 시점에 대해 2D 교차 평면이 달라지는 문제가 발생한다.

또한, 3D Gaussian을 레이 공간으로 변환하는 데 affine 행렬을 사용하면 중앙 근처에서만 정확한 투영이 이루어지고, 주변 영역의 원근 정확도가 저하된다. 이로 인해 노이즈가 많은 재구성 결과가 초래된다.

### 2.2 제안 방법 및 수식

#### 2.2.1 2D Gaussian 디스크 표현

3DGS가 blob 형태로 전체 각도 복사량(angular radiance)을 모델링하는 것과 달리, 2DGS는 3D 공간에 임베딩된 "평면(flat)" 2D Gaussian을 채택하여 3차원 모델링을 단순화한다. 2D Gaussian 모델링에서 기본 요소는 평면 디스크 내에 밀도를 분포시키며, 밀도의 가장 가파른 변화 방향으로 법선(normal)을 정의한다. 이 특성은 얇은 표면과의 더 나은 정렬을 가능하게 한다.

각 2D Gaussian 디스크는 로컬 접선 공간(local tangent space)에서 다음과 같이 정의된다:

$$\mathcal{G}(u, v) = \exp\!\left(-\frac{u^2 + v^2}{2}\right)$$

여기서 중심점 $\mathbf{p}_k \in \mathbb{R}^3$, 두 개의 주 접선 벡터 $\mathbf{t}_u, \mathbf{t}_v$와 스케일링 벡터 $S = (s_u, s_v)$로 파라미터화된다. 법선 벡터는 $\mathbf{n} = \mathbf{t}_u \times \mathbf{t}_v$로 정의된다.

#### 2.2.2 Ray-Splat 교차 및 Perspective-correct 렌더링

기존의 표면 splatting 방법(3DGS 포함)은 affine 투영(affine projection)에 의존하여 디스크가 원근 하에서 어떻게 보이는지를 근사한다. 2DGS는 이를 정확한 ray-plane 교차 계산으로 대체한다.

각 이미지 픽셀은 위치와 방향으로 정의되는 카메라 레이에 대응한다. 각 Gaussian 디스크에 대해 해당 레이가 디스크 평면과 교차하는 지점을 계산한다. 접선 평면 위 교차점의 좌표는 방정식을 풀어서 구하며, 이 좌표를 사용하여 Gaussian 함수를 평가한다. 그 결과는 이 디스크가 픽셀 색상에 얼마나 기여하는지를 나타낸다. 이 정확한 ray-plane 교차는 perspective-correct 렌더링을 보장하여, 극단적인 시점 각도에서도 모든 디스크의 형태, 크기, 기여도가 기하학적으로 정확하게 유지된다.

볼륨 렌더링은 다음 방정식으로 수행된다:

$$\hat{C} = \sum_{i=1}^{N} c_i \cdot \alpha_i \cdot \hat{\mathcal{G}}_i(u_i, v_i) \cdot \prod_{j=1}^{i-1}\left(1 - \alpha_j \cdot \hat{\mathcal{G}}_j(u_j, v_j)\right)$$

여기서 $c_i$는 색상, $\alpha_i$는 불투명도, $\hat{\mathcal{G}}$는 low-pass 필터가 적용된 Gaussian 값이다.

#### 2.2.3 정규화 손실 (Regularization Losses)

2DGS는 더 매끄러운 표면을 위해 두 가지 정규화 항을 도입한다. **Depth Distortion Term**은 레이를 따라 좁은 범위 내에 분포된 2D 기본 요소의 집중을 유도하며, 렌더링 프로세스에서 Gaussian 간 거리가 무시되는 제한을 해결한다. **Normal Consistency Term**은 렌더링된 법선 맵과 렌더링된 깊이의 기울기(gradient) 간 불일치를 최소화하여, 깊이와 법선으로 정의된 기하학 간의 정렬을 보장한다.

**(1) Depth Distortion Loss** ($\mathcal{L}_{d}$):

$$\mathcal{L}_d = \sum_{i,j} \omega_i \omega_j \left| z_i - z_j \right|$$

여기서 $\omega_i = \alpha_i \prod_{j<i}(1-\alpha_j)$는 볼륨 렌더링 가중치, $z_i$는 ray-splat 교차의 깊이(depth)이다.

Mip-NeRF360의 distortion loss와 달리, 이 접근법은 교차 깊이 $z_i$를 직접 조정함으로써 splats의 집중을 장려한다.

**(2) Normal Consistency Loss** ($\mathcal{L}_{n}$):

법선 일관성 손실은 다음과 같이 정의된다:

$$\mathcal{L}_n = \sum_{i} \omega_i (1 - \mathbf{n}_i^\top \mathbf{N})$$

여기서 $\mathbf{n}_i$는 각 2D Gaussian splat의 법선 벡터, $\mathbf{N}$은 렌더링된 깊이 맵의 기울기에서 추정된 법선 벡터이다.

2D Gaussian 표면 요소 기반 표현이므로, 모든 2D splats이 실제 표면과 국소적으로 정렬되어야 한다. 여러 반투명 서펠(surfel)이 레이를 따라 존재하는 체적 렌더링의 맥락에서, 축적 불투명도가 0.5에 도달하는 교차점 $p_s$에서의 실제 표면을 고려한다.

**최종 학습 손실 함수**:

$$\mathcal{L} = \mathcal{L}_{\text{color}} + \lambda_d \mathcal{L}_d + \lambda_n \mathcal{L}_n$$

여기서 $\mathcal{L}_{\text{color}}$는 포토메트릭 손실(L1 + D-SSIM 결합), $\lambda_d, \lambda_n$은 각 정규화 항의 가중치이다.

### 2.3 모델 구조

2DGS는 장면을 2D 방향성 디스크(서페이스 요소, surfels)의 집합으로 표현하며, perspective-correct 차분 가능한 래스터화로 이 서펠들을 렌더링한다. 또한 재구성 품질을 향상시키는 정규화 기법을 개발한다.

**전체 파이프라인**은 다음과 같다:

```
[입력] 멀티뷰 RGB 이미지 + SfM Sparse Point Cloud
    ↓
[초기화] SfM 점군에서 2D Gaussian 디스크 초기화
    ↓
[최적화] Adam Optimizer로 반복 최적화
    - 포토메트릭 손실 (L1 + D-SSIM)
    - Depth Distortion Loss
    - Normal Consistency Loss
    - Gaussian 제거(Pruning) & 밀도 증가(Densification)
    ↓
[메쉬 추출] TSDF Fusion (중앙값 깊이 사용)
    ↓
[출력] 고품질 삼각형 메쉬 + 실시간 Novel View Rendering
```

2D Gaussian splatting을 통해 뷰 일관성 있는 법선 및 깊이 맵과 함께 고품질 Novel View 이미지의 실시간 렌더링이 가능하며, 최적화된 2D 디스크로부터 상세하고 노이즈 없는 삼각형 메쉬 재구성을 제공한다.

### 2.4 성능 향상

2DGS는 DTU 데이터셋에서 다른 방법들과 비교하여 최고의 재구성 정확도를 달성하며, SDF 기반 기준선(baselines) 대비 100배의 속도 향상을 제공한다.

2DGS의 핵심 강점은 NeRF 수준의 시각적 품질과 SDF 수준의 기하학을 3DGS 수준의 실시간 속도로 제공한다는 점이다.

이 차분 가능 렌더러는 경쟁력 있는 외관 품질, 빠른 학습 속도, 실시간 렌더링을 유지하면서 노이즈 없고 상세한 기하학적 재구성을 가능하게 한다는 것을 입증한다.

**벤치마크 성능 요약**:

| 데이터셋 | 지표 | 결과 |
|----------|------|------|
| DTU | Chamfer Distance | SOTA (명시적 표현 중 최저) |
| Tanks & Temples | F-score | SOTA |
| MipNeRF-360 | PSNR/SSIM/LPIPS | 3DGS와 경쟁적 수준 |
| DTU vs SDF 기준선 | 학습 속도 | ~100배 빠름 |

### 2.5 한계점

2DGS의 주요 한계는 반투명 표면(semi-transparent surfaces)의 정확한 재구성에 어려움을 겪는다는 점이다(예: 유리).

2DGS는 얇은 표면 근사를 위해 2D 서펠을 사용하여 3DGS보다 우수한 기하학 재구성 품질을 보이지만, 광택이 있는 표면(glossy surfaces)을 처리할 때는 부족함이 있어 해당 영역에 가시적인 구멍(holes)이 발생한다. 반사 불연속성(reflection discontinuity)이 이 문제의 원인이며, 다른 시점 각도에서 확산 반사에서 정반사로의 전환을 맞추기 위해 최적화 과정에서 실제 표면 뒤에 하이라이트 Gaussian을 생성한다.

2DGS는 효율성을 유지하면서 3DGS 대비 향상된 기하학적 재구성 성능을 제공하지만, 렌더링 품질 측면에서는 정성적·정량적 평가 모두에서 한계를 드러낸다.

추가적인 한계 사항:
- **대규모 장면**: 2DGS가 더 나은 일반화 능력을 보이지만, 흐릿한 Gaussian으로 인해 수렴이 방해받는 문제가 있다.
- **초기화 의존성**: SfM 기반 희소 점군(sparse point cloud)이 여전히 필요하여 pose-free 시나리오에는 직접 적용이 어렵다.

---

## 3. 모델 일반화 성능 향상 가능성

### 3.1 기존 2DGS의 일반화 특성

기존 방법들이 기하학 재구성을 위해 dense point cloud나 ground-truth 법선을 입력으로 요구하는 반면, 2DGS는 희소 캘리브레이션 점군과 포토메트릭 감독만으로 외관과 기하학을 동시에 재구성한다.

SDF 기반 재구성 방법들은 초기화를 위해 구형 크기를 사전 정의해야 하며, 이것이 SDF 재구성 성공에 결정적인 역할을 한다. 반면 2DGS의 방법은 Radiance Field 기반 기하학 모델링을 활용하여 초기화에 덜 민감하다.

### 3.2 Feed-forward 일반화 (Generalizable 2DGS)

MeshSplat은 2DGS를 브릿지로 활용하여 Novel View Synthesis와 학습된 기하학적 사전(geometric priors)을 연결하고, 표면 재구성으로 이전하는 generalizable sparse-view 표면 재구성 프레임워크를 제안한다. 구체적으로 feed-forward 네트워크를 통합하여 뷰별 픽셀 정렬 2DGS를 예측하며, 이를 통해 직접적인 3D ground-truth 감독 없이 Novel View 이미지를 합성할 수 있다.

Sparse2DGS는 희소 뷰에서의 표면 재구성을 위한 기하학 우선(geometry-prioritized) Gaussian Splatting으로, CVPR 2025에 발표되었으며, 2DGS 대비 재구성 정확도를 크게 향상시켰다.

### 3.3 대규모 장면에서의 일반화

CityGaussianV2는 2DGS를 그 유리한 일반화 능력으로 인해 기본 기본 요소(primitive)로 채택하며, Depth-Anything V2로 유도된 깊이 회귀와 Decomposed-Gradient-based Densification(DGD)을 사용하여 재구성을 가속화한다.

### 3.4 동적 장면 일반화

Dynamic 2D Gaussians(ACM MM 2025)은 2DGS를 동적 객체를 위한 기하학적으로 정확한 Radiance Field로 확장하는 연구로, 2DGS 프레임워크의 동적 장면 일반화 가능성을 보여준다.

### 3.5 반사 표면 일반화

GS-2DGS는 2DGS 기반의 반사 객체 재구성을 위한 새로운 방법을 제안하며, Gaussian Splatting의 빠른 렌더링 능력과 foundation model의 추가적인 기하학적 정보를 결합한다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

최근 3DGS는 암묵적(implicit) 표현이나 특징 그리드 기반 표현의 매력적인 대안으로 부상하였으며, 고해상도에서 실시간 포토리얼리스틱 NVS 결과를 달성한다.

3DGS는 안티-앨리어싱 렌더링, 재료 모델링, 동적 장면 재구성, 애니메이터블 아바타 생성 등 여러 도메인에서 빠르게 확장되었다.

### 비교표: 2020년 이후 주요 방법론

| 방법 | 연도 | 표현 방식 | 렌더링 속도 | 기하학 정확도 | 주요 특징 |
|------|------|-----------|------------|--------------|----------|
| **NeRF** | 2020 | 암묵적 MLP | 느림 | 보통 | 최초의 Neural Radiance Field |
| **NeuS** | 2021 | SDF + MLP | 느림 | 높음 | SDF 기반 표면 재구성 |
| **Instant-NGP** | 2022 | Hash Grid | 중간 | 보통 | 빠른 학습 속도 |
| **3DGS** | 2023 | 3D Gaussian | 실시간 | 낮음 | 실시간 렌더링 혁신 |
| **SuGaR** | 2023 | 3DGS + 정규화 | 빠름 | 중간 | 메쉬 추출 용이 |
| **2DGS (본 논문)** | 2024 | 2D Gaussian 디스크 | 실시간 | 높음 | 기하학 정확도와 속도 균형 |
| **GOF** | 2024 | 3DGS + Ray-tracing | 중간 | 높음 | Gaussian Opacity Fields |
| **PGSR** | 2024 | 평면 3DGS | 빠름 | 높음 | 편향 없는 깊이 렌더링 |
| **Mip-Splatting** | 2024 | 3DGS (anti-alias) | 실시간 | 보통 | 앨리어싱 제거 |

SDF 접근법은 NeuS에 의해 대중화되었으며, NeuS2(빠른 신경 표면 재구성, 멀티해상도 해시 인코딩 사용)와 Neuralangelo(고차 도함수 감독 사용) 등의 최근 발전이 있었다.

더 나은 기하학 재구성을 위해 SuGaR는 기하학 정규화를 사용하여 3D Gaussian을 표면에 정렬하도록 강제한다. 2DGS와 GaussianSurfels는 편평화된 기본 요소(flatten primitive)를 활용하여 표면에 맞춘다. 이러한 방법들이 기하학적 재구성 품질을 향상시키지만, 원본 대비 렌더링 품질이 약간 저하되는 경향이 있으며, 이는 신경 필드 기반 표현에서 렌더링 품질과 기하학적 정확도를 조화시키는 지속적인 과제를 반영한다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려 사항

### 5.1 연구에 미치는 영향

**① 표면 기반 명시적 표현의 패러다임 전환**
NeRF의 암묵적 Radiance Field에서 고속 3D Gaussian Splatting을 거쳐, 투영 드리프트를 수정하고 정확한 호모그래피를 강제하며 기하학을 distortion 및 normal 정규화로 다듬는 표면 충실도 높은 2DGS 프레임워크까지 발전해 왔다.

**② 후속 연구 활성화**
Sparse2DGS는 최고 수준의 표면 재구성 정확도를 달성하며 NeRF 기반 fine-tuning보다 훨씬 빠르다는 사례와 같이, 2DGS는 희소 뷰(sparse-view) 재구성, 동적 장면, 자율주행, SLAM 등 다양한 후속 연구를 촉발시켰다.

**③ 역 렌더링 및 물리 기반 표현으로의 확장**
각 2D Gaussian이 명시적 법선 벡터를 가진 표면 평면 위에 직접 정의되므로, 기하학이 부산물이 아닌 일급 요소(first-class element)가 된다. 이는 물리 기반 렌더링(PBR), 재조명(relighting), 재질 분해(material decomposition)로의 확장을 용이하게 한다.

**④ 실용적 응용 확대**
실시간 렌더링, 깔끔하고 삼각분할 가능한 기하학, 고급 신경 필드도 맞추기 어려운 멀티뷰 일관성을 제공한다. 이는 디지털 트윈, AR/VR, 로보틱스 등 실용적 분야로의 적용을 앞당긴다.

### 5.2 앞으로 연구 시 고려할 점

**① 렌더링 품질과 기하학적 정확도의 균형**
이 방법들이 기하학적 재구성 품질을 향상시키지만 렌더링 품질이 약간 저하되는 문제는, 신경 필드 기반 표현에서 렌더링 품질과 기하학적 정확도를 조화시키는 지속적인 과제를 반영한다. 향후 연구에서 이 trade-off를 해결할 필요가 있다.

**② 반투명 및 광택 표면 처리**
2DGS에 depth distortion loss가 있음에도 불구하고, transmittance 가중치 사용으로 인해 깊이 연속성 강제가 충분하지 않아 반사 영역에서 구멍이 발생한다. 이러한 특수 재질 처리가 중요한 연구 방향이다.

**③ Feed-forward 일반화 모델로의 발전**
도메인 일반화 및 pose-free 재구성에서의 지속적인 과제들과 3D 네이티브 생성적 사전(generative priors) 개발 및 실시간 제약 없는 희소 뷰 재구성 달성을 위한 미래 연구 방향이 필요하다.

**④ 대규모 장면 처리**
2DGS의 규모 확장 시 핵심 장애물은 병렬 학습 단계에서 특정 기본 요소의 과도한 증식이다. 2D Gaussian은 특히 극단적 늘어짐(extreme elongation)을 보이는 경우 거리에서 투영될 때 매우 작은 점으로 붕괴될 수 있다.

**⑤ 정규화 항의 세밀한 튜닝**
법선 일관성을 비활성화하면 잘못된 방향이 발생하고, depth distortion 없이는 노이즈 있는 표면이 발생한다는 ablation 결과는, 정규화 항의 가중치 조정이 성능에 매우 민감하므로 자동화된 하이퍼파라미터 탐색이나 adaptive 정규화 전략 개발이 필요함을 시사한다.

**⑥ 동적 장면 및 비정형 조건 대응**
희소 뷰 3D 재구성은 로보틱스, AR/VR, 자율 시스템과 같이 밀집 이미지 획득이 비현실적인 응용에 필수적이다. 이러한 환경에서 최소한의 이미지 겹침은 신뢰할 수 있는 대응점 매칭을 방해하여 전통적 방법들이 실패한다. 2DGS 기반 시스템이 이러한 도전적 조건에서도 강건하게 동작하도록 하는 연구가 필요하다.

---

## 참고 자료

1. **원 논문 (ACM SIGGRAPH 2024)**: Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, Shenghua Gao. "2D Gaussian Splatting for Geometrically Accurate Radiance Fields." *SIGGRAPH 2024 Conference Papers*. DOI: 10.1145/3641519.3657428.
2. **arXiv 논문**: [arXiv:2403.17888](https://arxiv.org/abs/2403.17888)
3. **공식 프로젝트 페이지**: [surfsplatting.github.io](https://surfsplatting.github.io/)
4. **GitHub 코드**: [github.com/hbb1/2d-gaussian-splatting](https://github.com/hbb1/2d-gaussian-splatting)
5. **LearnOpenCV 튜토리얼**: "2D Gaussian Splatting: Radiance Field Reconstruction." [learnopencv.com](https://learnopencv.com/2d-gaussian-splatting-2dgs/)
6. **Medium (Sergeodesico)**: "2D Gaussian Splatting: from pixels to geometry, part 1." [medium.com](https://medium.com/@sergio.deleon_41219/2d-gaussian-splatting-from-pixels-to-geometry-part-1-b08763fbfefe)
7. **2DGS-R (arXiv:2510.16837)**: "Revisiting the Normal Consistency Regularization in 2D Gaussian Splatting." [arxiv.org/html/2510.16837](https://arxiv.org/html/2510.16837)
8. **GS-2DGS (arXiv:2506.13110)**: "Geometrically Supervised 2DGS for Reflective Object Reconstruction." [arxiv.org/html/2506.13110](https://arxiv.org/html/2506.13110)
9. **Sparse2DGS (CVPR 2025)**: "Sparse2DGS: Geometry-Prioritized Gaussian Splatting for Surface Reconstruction from Sparse Views."
10. **CityGaussianV2**: "Efficient and Geometrically Accurate Large-scale Scene Reconstruction." [dekuliutesla.github.io](https://dekuliutesla.github.io/CityGaussianV2/)
11. **3DGS Survey (IEEE TCSVT 2025)**: "A Survey on 3D Gaussian Splatting." DOI: 10.1109/TCSVT.2025.3538684
12. **Sparse-View 3D Reconstruction Survey (arXiv:2507.16406)**: "Sparse-View 3D Reconstruction: Recent Advances and Open Challenges."
13. **Under the 3D (블로그)**: "Geometrically Accurate 2D Gaussian Splatting." [hwan-h-heo.github.io](https://hwan-h-heo.github.io/hwan-h-heo.io/blogs/posts/240602_2dgs/)
14. **ACM Full Paper**: [dl.acm.org/doi/fullHtml/10.1145/3641519.3657428](https://dl.acm.org/doi/fullHtml/10.1145/3641519.3657428)
