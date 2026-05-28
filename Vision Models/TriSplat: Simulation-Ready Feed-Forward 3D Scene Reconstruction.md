
# TriSplat: Simulation-Ready Feed-Forward 3D Scene Reconstruction

> **논문 정보**: Wang et al., arXiv:2605.26115, 25 May 2026
> **소속**: Zhejiang University, ETH Zurich, ETH AI Center, Microsoft, Monash University

---

## 1. 핵심 주장 및 주요 기여 요약

### 🔑 핵심 주장

Sparse-view 3D 재구성은 이미지로부터 explicit primitive를 직접 예측하는 feed-forward splatting 네트워크로 점점 해결되고 있지만, 대부분의 기존 방법은 Gaussian primitive에 집중하며 표면을 간접적으로만 노출한다. 시뮬레이션, 물리 추론, 또는 embodied interaction을 위해 사용 가능한 mesh를 추출하려면 여전히 비용이 큰 후처리(post-hoc) 단계가 필요하며, 이는 feed-forward의 장점을 깨뜨린다.

이 한계는 특히 pose-free 환경에서 두드러지는데, 장면 구조와 카메라 파라미터를 희소한 관측으로부터 동시에 추정해야 하기 때문이다.

**TriSplat의 핵심 주장**: TriSplat은 oriented triangle primitive로 장면을 표현하고, 단일 forward pass에서 직접 simulation-ready mesh 장면을 출력하는 feed-forward 재구성 네트워크이다.

### 📌 주요 기여

| 기여 | 내용 |
|------|------|
| 표현 방식 혁신 | Gaussian 대신 oriented triangle primitive 채택 |
| 직접 mesh 출력 | 후처리 없이 단일 forward pass에서 simulation-ready mesh 생성 |
| 훈련 안정화 | Mono-normal bootstrap schedule 및 opacity/blur scheduling |
| 경쟁력 있는 성능 | 기하학적 충실도 향상 + 경쟁력 있는 novel-view 렌더링 품질 유지 |

렌더링 primitive가 표면 삼각형 자체이므로, 출력은 어떠한 변환 없이 물리 엔진, 충돌 감지기, 표준 렌더링 파이프라인에서 직접 사용할 수 있어, feed-forward 3D 장면 재구성을 위한 실용적인 simulation-ready 솔루션이 된다.

---

## 2. 해결하고자 하는 문제 / 제안 방법 / 모델 구조 / 성능 및 한계

### 2.1 해결하고자 하는 문제

기존 feed-forward splatting 네트워크들은 Gaussian primitive로 표면을 간접적으로만 노출한다. 이를 mesh로 변환하려면 여전히 비용이 큰 TSDF fusion 또는 Poisson reconstruction이 필요하며, 이는 feed-forward의 장점을 깨뜨리고 다운스트림 시뮬레이션을 번거롭게 만든다.

NeRF와 3D Gaussian Splatting(3DGS)은 인상적인 렌더링 품질을 달성하지만, 시간 소모적인 scene별 최적화에 의존한다는 점에서 실시간 배포를 제한한다. 등장하고 있는 feed-forward Gaussian splatting 방법들은 효율성을 개선하지만 직접 시뮬레이션에 필요한 explicit, manifold 기하학이 종종 부족하다.

---

### 2.2 제안 방법

#### (A) 전체 파이프라인

주어진 unposed 이미지로부터, TriSplat은 단일 forward pass에서 local 3D point map, per-pixel triangle attribute, 카메라 포즈, 선택적 focal length를 동시에 예측한다.

Triangle orientation을 비제약 잠재 변수로 회귀하는 대신, 예측된 point map에서 geometry normal을 구성하고, image-conditioned normal head로 이를 정제한 뒤, triangle parameterization을 위한 안정적인 local frame으로 변환한다.

#### (B) Triangle Primitive 표현

각 primitive는 학습된 center, scale, rotation, appearance, opacity, blur를 가진 canonical triangle template에서 인스턴스화되고, 미분 가능한 triangle rasterizer로 렌더링되며, 부드러운 primitive에서 선명한 표면 요소로 점진적으로 sharpening된다.

각 triangle primitive는 다음과 같은 attribute들로 구성됩니다:

$$\mathbf{t}_i = \{\mathbf{c}_i, s_i, \mathbf{R}_i, \mathbf{a}_i, \alpha_i, \sigma_i\}$$

여기서:
- $\mathbf{c}_i \in \mathbb{R}^3$: triangle center (3D 좌표)
- $s_i \in \mathbb{R}$: scale
- $\mathbf{R}_i \in SO(3)$: rotation (normal-anchored local frame)
- $\mathbf{a}_i$: appearance (색상 등)
- $\alpha_i \in [0,1]$: opacity
- $\sigma_i$: edge blur (sharpening 스케줄에 의해 점진적으로 감소)

#### (C) Normal 기반 Triangle 방향 앵커링

예측된 point map의 geometry normal은 image-conditioned normal head에 의해 정제되고, monocular teacher로부터 warm-start되며, validity-aware masking으로 안정화된다.

Point map $\mathbf{P} \in \mathbb{R}^{H \times W \times 3}$으로부터 finite-difference로 geometry normal을 추정하면:

$$\hat{\mathbf{n}}_{ij} = \frac{(\mathbf{p}_{i+1,j} - \mathbf{p}_{i-1,j}) \times (\mathbf{p}_{i,j+1} - \mathbf{p}_{i,j-1})}{\|(\mathbf{p}_{i+1,j} - \mathbf{p}_{i-1,j}) \times (\mathbf{p}_{i,j+1} - \mathbf{p}_{i,j-1})\|}$$

이를 image-conditioned normal head $f_\phi$로 정제:

$$\mathbf{n}_{ij} = f_\phi(\hat{\mathbf{n}}_{ij},\ \mathbf{I})$$

> ⚠️ **주의**: 위 수식은 논문의 설계 원리("finite-difference geometry normals", "image-conditioned normal head")를 바탕으로 표준적인 표기법으로 재구성한 것이며, 논문 원문의 정확한 수식 기호와 다를 수 있습니다. 정확한 수식은 [arxiv.org/abs/2605.26115](https://arxiv.org/abs/2605.26115) 원문을 확인하세요.

#### (D) Mono-Normal Bootstrap Schedule

Mono-normal bootstrap schedule은 초기 학습을 안정화하고, opacity 및 blur scheduling은 직접 mesh 추출을 위해 학습된 표면 표현을 점진적으로 sharpening한다.

Opacity와 edge blur 스케줄은 관대한 soft footprint에서 시작하여 점진적으로 선명하고 mesh-ready한 표면 요소로 수렴한다.

학습 스케줄을 수식으로 표현하면:

$$\sigma(t) = \sigma_0 \cdot \exp\!\left(-\lambda_\sigma \cdot \frac{t}{T}\right), \quad \alpha_{\min}(t) = \alpha_0 + (\alpha_T - \alpha_0)\cdot\frac{t}{T}$$

여기서 $t$는 현재 학습 스텝, $T$는 총 스텝 수입니다.

> ⚠️ 위는 논문의 설계 원리를 일반적인 스케줄링 수식으로 표현한 것이며, 논문 원문의 정확한 파라미터와 다를 수 있습니다.

#### (E) Mesh 추출

낮은 opacity의 triangle들을 제거(filtering)하고, face winding을 보정하며, 인접 vertex를 병합함으로써, native triangle primitive들이 표준 mesh가 된다.

#### (F) 학습 손실 함수 (추론)

논문에서 밝히는 학습 구성 요소에 기반하면, 전체 손실 함수는 다음과 같은 형태를 가집니다:

$$\mathcal{L} = \lambda_\text{rgb}\,\mathcal{L}_\text{rgb} + \lambda_\text{normal}\,\mathcal{L}_\text{normal} + \lambda_\text{depth}\,\mathcal{L}_\text{depth} + \lambda_\text{pose}\,\mathcal{L}_\text{pose}$$

여기서:
- $\mathcal{L}_\text{rgb}$: 렌더링된 이미지와 GT 이미지 간의 photometric loss (LPIPS, L1/L2 등)
- $\mathcal{L}_\text{normal}$: geometry normal 감독 (monocular teacher 및 multi-view 일관성)
- $\mathcal{L}_\text{depth}$: point map depth 감독
- $\mathcal{L}_\text{pose}$: 카메라 포즈 회귀 손실

> ⚠️ 이 수식 구조는 논문의 설계 원리 및 유사 논문들을 기반으로 합리적으로 추론한 것입니다. 정확한 계수 및 손실 구성은 원문을 확인하세요.

---

### 2.3 모델 구조

DINOv2 기반의 transformer decoder가 dense local 3D point map, 상대적 카메라 포즈, 선택적 intrinsics, 그리고 per-pixel triangle attribute를 예측한다.

희소 입력 뷰가 주어지면, TriSplat은 dense local point map, triangle attribute, 카메라 포즈, 선택적 intrinsics를 예측한다. Point-map geometry는 geometry normal, learned normal refiner, monocular-normal bootstrap을 통해 triangle orientation을 앵커링한다. 미분 가능한 triangle rasterizer가 RGB, depth, normal을 렌더링하며, mesh 추출은 opacity filtering, winding correction, duplicate-vertex merging만 필요하다.

**모델 구조 도식:**

```
입력 이미지 (sparse, unposed)
        │
        ▼
[DINOv2 Backbone] ── 이미지 특징 추출
        │
        ▼
[Transformer Decoder]
        ├──► 3D Point Map (기하학적 구조)
        ├──► Camera Poses (상대적)
        ├──► Intrinsics (선택적)
        └──► Per-pixel Triangle Attributes (center, scale, R, color, opacity, blur)
                    │
                    ▼
        [Normal Refinement Head] ← Monocular Teacher (warm-start)
                    │
                    ▼
        [Differentiable Triangle Rasterizer]
                    │
                    ▼
        RGB / Depth / Normal 렌더링
                    │
                    ▼
        [Mesh 추출: opacity filter + winding correction + vertex merge]
                    │
                    ▼
        Simulation-Ready Triangle Mesh (.ply / .off)
```

---

### 2.4 성능 향상

RealEstate10K과 DL3DV에서의 실험은 TriSplat이 최신 Gaussian feed-forward baseline들을 능가하는 mesh 렌더링 품질을 제공하면서 표면 정확도 지표에서도 일관되게 더 나은 결과를 보임을 보여준다. 특히, 모든 방법이 표준 triangle 렌더링을 위해 mesh를 export할 때, Gaussian baseline들은 손실이 큰 TSDF fusion으로 인해 상당한 품질 저하를 겪는 반면, TriSplat은 렌더링 primitive 자체가 이미 mesh이기 때문에 최소한의 품질 저하만 발생한다.

Feed-forward 효율성 측면에서, TriSplat은 1.3초 이내에 simulation-ready triangle mesh를 출력하는 반면, Gaussian-to-mesh baseline들은 수십 초에서 수백 초가 필요하다.

TriSplat은 선명한 경계와 얇은 구조를 보존하는 반면, TSDF 기반 baseline들은 흐릿한 경계와 누락된 기하학을 보여준다.

| 측면 | TriSplat | Gaussian Baseline |
|------|---------|-------------------|
| Mesh 렌더링 품질 | ✅ 우수 | ❌ TSDF 손실로 저하 |
| 표면 정확도 | ✅ 일관되게 우수 | ❌ 열등 |
| 추론 속도 | ✅ < 1.3초 | ❌ 수십~수백 초 (mesh 변환 포함) |
| 후처리 필요 | ❌ 불필요 | ✅ TSDF/Poisson 필요 |
| 시뮬레이션 호환성 | ✅ 직접 호환 | ❌ 변환 필요 |

---

### 2.5 한계

현재 공개된 arXiv 논문 기준으로 확인된 한계는 다음과 같습니다:

1. **학습 데이터 편향**: RealEstate10K과 DL3DV에서 학습하고, RE10K 학습 모델로 ScanNet에서 zero-shot 평가를 수행한다. 즉, 실내 환경 위주의 데이터에 편향될 수 있으며, 야외나 동적 장면에 대한 일반화는 검증되지 않았습니다.

2. **Triangle Splatting의 표현력 한계**: 반투명 물체, 털(fur), 연기 등 비표면적(non-surface) 현상의 표현이 Gaussian에 비해 어려울 수 있습니다 (Gaussian 표현의 volumetric 장점 포기).

3. **입력 뷰 수의 제한**: DL3DV에서 6, 12, 24 입력 뷰, RE10K에서 6 뷰로 평가한다. 매우 극단적인 sparse 뷰(1~2장)에서의 성능은 불명확합니다.

4. **수식 및 구조 세부 공개**: 이 논문은 2026년 5월 25일 공개된 초판(v1)이므로, 일부 세부 수식 및 하이퍼파라미터 정보가 제한적입니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 Zero-Shot 일반화 검증

ScanNet에 대한 zero-shot 평가는 cross-dataset 일반화를 추가로 확인하며, ablation 연구들은 각 제안 구성 요소의 상보적인 기여를 검증한다.

Table 4는 RE10K 학습 모델을 fine-tuning 없이 zero-shot 설정에서 ScanNet의 depth 및 normal 정확도를 평가한다.

### 3.2 일반화를 돕는 핵심 설계 요소

**① DINOv2 기반 backbone의 강력한 사전 학습 표현**

DINOv2 기반의 transformer decoder가 dense local 3D point map, 상대적 카메라 포즈, 선택적 intrinsics, per-pixel triangle attribute를 예측한다.

DINOv2는 대규모 자기지도 학습으로 뛰어난 visual feature를 제공하여 미학습 장면에서도 의미 있는 특징을 추출할 수 있습니다.

**② Normal-Anchored Triangle Parameterization의 구조적 유도 편향(Inductive Bias)**

TriSplat은 point-map geometry, normal-anchored local frame, 점진적 surface sharpening을 결합하여 hard-edged triangle primitive들이 안정적으로 학습되고 깔끔하게 export될 수 있게 한다.

geometry normal에 orientation을 앵커링하는 것은 장면의 물리적 구조를 직접 반영하는 **강한 귀납적 편향**을 제공하여, 학습 데이터 분포 밖의 장면에도 안정적으로 작동할 가능성을 높입니다.

**③ Pose-Free 설계의 잠재력**

이 한계(포즈 추정의 어려움)는 특히 pose-free 환경에서 두드러지는데, 장면 구조와 카메라 파라미터를 희소한 관측으로부터 동시에 추정해야 하기 때문이다.

TriSplat이 카메라 포즈를 동시에 예측하므로, calibrated 이미지 없이도 다양한 환경에 적용 가능하며, 이는 실세계 일반화에 직접적으로 기여합니다.

**④ 표현 방식의 범용성**

시뮬레이션 준비성을 위해 렌더링 primitive 자체가 표면 요소여야 한다—삼각형은 구조적으로 이를 만족하며 중간 추출 없이 mesh로 export될 수 있다.

Triangle mesh는 표준 3D 형식으로, 물리 엔진, 렌더러 등 어디서나 호환되므로 다양한 도메인으로의 전이 가능성이 높습니다.

### 3.3 일반화 향상을 위한 잠재적 방향

- **다양한 데이터셋 혼합 학습**: 실외(outdoor), 동적(dynamic), 의료(medical) 등 다양한 도메인의 데이터 추가
- **Foundation Model과의 결합**: SAM, CLIP 등과의 통합으로 의미론적 이해를 통한 일반화 강화
- **Self-supervised / Unsupervised 학습**: GT mesh나 깊이 지도 없이 학습하는 방향으로 확장
- **도메인 적응(Domain Adaptation)**: 소수의 타겟 도메인 샘플만으로 fine-tuning하는 경량 적응 전략

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 향후 연구에 미치는 영향

**① Feed-Forward 3D 재구성 패러다임의 전환**

TriSplat은 oriented triangle primitive를 native 표현으로 사용하는 feed-forward 네트워크로, 단일 forward pass에서 희소하고 unposed된 이미지로부터 geometry, appearance, 카메라 포즈를 동시에 예측한다.

이는 "Gaussian Splatting이 표준"이었던 feed-forward 3D 재구성 분야에서 **triangle primitive 기반 접근이 실용적임을 증명**한 선구적 사례가 됩니다.

**② 로봇공학 및 시뮬레이션 AI와의 융합 가속화**

feed-forward 모델이 단일 pass에서 triangle mesh를 예측함으로써, 로코모션, 역학, 로봇 그래스핑을 위해 물리 엔진에서 직접 사용할 수 있게 한다.

이는 **Embodied AI, 자율주행, 로봇 매니퓰레이션** 분야의 실시간 3D 환경 이해 연구에 직접적인 영향을 미칩니다.

**③ 관련 최신 연구와의 비교**

| 방법 | 연도 | 표현 방식 | Mesh 직접 출력 | Pose-free | 실시간성 |
|------|------|-----------|--------------|-----------|---------|
| NeRF (Mildenhall et al.) | 2020 | Implicit (MLP) | ❌ | ❌ | ❌ |
| 3DGS (Kerbl et al.) | 2023 | 3D Gaussian | ❌ (후처리 필요) | ❌ | ✅ |
| pixelSplat | 2023 | 3D Gaussian | ❌ | ❌ | ✅ |
| MVSplat | 2024 | 3D Gaussian | ❌ | ❌ | ✅ |
| FTSplat (2026) | 2026 | Triangle | ✅ | ❌ (calibrated 필요) | ✅ |
| **TriSplat (2026)** | 2026 | Triangle | ✅ | ✅ | ✅ |

NeRF와 3DGS는 인상적인 렌더링 품질을 달성하지만, 시간 소모적인 scene별 최적화에 의존하며, 등장하는 feed-forward Gaussian splatting 방법들도 직접 시뮬레이션을 위한 explicit manifold geometry가 부족하다.

### 4.2 앞으로 연구 시 고려할 점

1. **동적 장면(Dynamic Scene) 확장**: TriSplat은 정적 장면을 가정합니다. 움직이는 물체가 있는 동적 장면으로의 확장은 중요한 연구 방향입니다.

2. **Primitive 해상도와 메모리 트레이드오프**: Triangle primitive 수가 많아질수록 메모리와 연산 비용이 증가합니다. 적응적(adaptive) triangle 밀도 제어 메커니즘 연구가 필요합니다.

3. **텍스처 및 재질(Material) 표현 강화**: 현재 appearance 표현이 기본적인 색상에 머무른다면, PBR(Physically Based Rendering) 재질 파라미터 예측으로의 확장이 시뮬레이션 현실성을 크게 높일 수 있습니다.

4. **훈련 데이터 다양성**: DL3DV와 RE10K 중심으로 평가가 이루어지며, ScanNet은 zero-shot 설정에서 평가된다. 더 다양한 실세계 환경(야외, 의료, 산업 현장 등)에서의 검증이 필요합니다.

5. **Foundation Model과의 통합**: DINOv2 외에 더 강력한 vision-language 모델(e.g., CLIP, SAM)을 통합하여 의미론적(semantic) 이해 기반 재구성으로 확장하는 방향이 유망합니다.

6. **물리 시뮬레이션 평가 지표 개발**: 현재 평가는 주로 렌더링 품질(PSNR, LPIPS)과 기하학적 정확도(depth, normal)에 집중됩니다. 실제 물리 시뮬레이션에서의 유용성을 측정하는 새로운 벤치마크 개발이 필요합니다.

7. **Triangle Topology 최적화**: 단순 per-pixel triangle 배치를 넘어, 장면의 기하학적 특성에 맞는 topology(위상 구조)를 학습하는 방향이 mesh 품질을 크게 향상시킬 수 있습니다.

---

## 📚 참고 자료 및 출처

| # | 자료 | 링크 |
|---|------|------|
| 1 | **TriSplat 논문 (arXiv:2605.26115)** | https://arxiv.org/abs/2605.26115 |
| 2 | **TriSplat HTML 전문** | https://arxiv.org/html/2605.26115v1 |
| 3 | **TriSplat 프로젝트 페이지** | https://lhmd.top/trisplat/ |
| 4 | **TriSplat GitHub** | https://github.com/ziplab/TriSplat |
| 5 | **AlphaXiv (TriSplat)** | https://www.alphaxiv.org/abs/2605.26115 |
| 6 | **FTSplat: Feed-forward Triangle Splatting Network (arXiv:2603.05932)** | https://arxiv.org/abs/2603.05932 |
| 7 | **G3Splat: Geometrically Consistent Generalizable Gaussian Splatting (arXiv:2512.17547)** | https://arxiv.org/abs/2512.17547 |

> ⚠️ **정확도 고지**: TriSplat은 2026년 5월 25일 공개된 초판(v1) 논문입니다. 본 답변의 수식 일부는 논문의 설계 원리와 유사 연구를 바탕으로 합리적으로 재구성한 것이며, 논문 원문의 정확한 수식과 다를 수 있습니다. 정확한 수식과 세부 내용은 반드시 원문 PDF(https://arxiv.org/pdf/2605.26115)를 확인하시기 바랍니다.
