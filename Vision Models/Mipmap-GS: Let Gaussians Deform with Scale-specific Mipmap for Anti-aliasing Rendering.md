# Mipmap-GS: Let Gaussians Deform with Scale-specific Mipmap for Anti-aliasing Rendering
---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

Mipmap-GS는 **3D Gaussian Splatting(3DGS)이 단일 스케일 학습으로 인해 줌-인/줌-아웃 시 심각한 앨리어싱(aliasing) 문제를 겪는다**는 점에 주목합니다. 기존 방법들이 후처리(post-processing) 필터링에 의존했다면, 이 논문은 **Gaussian 자체를 스케일에 맞게 변형(deform)** 시키는 근본적인 해결책을 제안합니다.

### 주요 기여 (4가지)

| 기여 | 설명 |
|------|------|
| ① Scale-adaptive 최적화 | Scale-aware guidance loss로 Gaussians를 줌-인/아웃 모두에 적응 |
| ② Mipmap 유사 Pseudo-GT | 테스트 시간에 동적으로 스케일별 pseudo ground-truth 생성 |
| ③ 빠른 수렴 | 기존 학습의 3%인 1K 이내 반복으로 수렴 (수십 초 단위) |
| ④ Plug-in 모듈 | 모든 3DGS 변형 모델에 적용 가능한 범용 모듈 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

3DGS는 단일 스케일로 학습된 후 **다양한 해상도(관찰 거리)에서 렌더링할 때** 두 가지 문제가 발생합니다:

#### 문제 1: 가변적인 샘플링 레이트 (Varying Sampling Rates)

3DGS는 3D Gaussian ellipsoid들을 2D 이미지 공간에 투영하여 픽셀 색상을 계산합니다:

$$c(x) = \sum_{k=1}^{K} c_k \alpha_k \prod_{i=1}^{k-1}(1 - \alpha_i) \tag{2}$$

줌-인 시 픽셀 그리드가 미세해져 각 픽셀에 기여하는 Gaussian이 줄어들고, 줌-아웃 시 하나의 픽셀에 과도하게 많은 Gaussian이 누적됩니다.

#### 문제 2: 고정된 2D Dilation 문제

3DGS는 투영된 2D Gaussian에 고정 dilation 상수를 적용합니다:

$$G_k^{2D}(x) = e^{-\frac{1}{2}(x - \mu_k)^\top (\Sigma_k^{2D} + sI)^{-1}(x - \mu_k)} \tag{3}$$

여기서 $s = 0.3$으로 **고정**되어 있어:
- **줌-인 시**: shrinkage bias를 충분히 보정하지 못해 바늘 형태(needle-like) 스파이크 발생
- **줌-아웃 시**: 과도한 dilation으로 두꺼운 구조물, 과도한 밝기 발생

```
줌-아웃: 하나의 픽셀에 너무 많은 Gaussian → 과도한 기여 → Dilation
줌-인:   하나의 픽셀에 너무 적은 Gaussian → 구조 결손 → Erosion
```

---

### 2.2 제안하는 방법

#### 전체 파이프라인

```
Base GS (×1)
    ↓ 렌더링 (기본 스케일)
Novel View (̂x)
    ↓ Mipmap 함수 r(·) 적용
Pseudo-GT: r(x̂)
    ↕ Loss 계산
Adaptive GS (렌더링 ×N 또는 ×1/N)
    → 최적화 → Optimized GS
```

#### Step 1: Mipmap 유사 Pseudo-GT 생성

기존 학습된 Base Gaussian $\mathbf{G}$로 기본 스케일($\times 1$) Novel View $\hat{x}$를 렌더링 후:

- **줌-인**: SwinIR(초해상도 모델)로 $\times N$ 업샘플링
  $$r(\hat{x}) = \text{SwinIR}(\hat{x}) \quad \text{(for zoom-in, } \times N\text{)}$$

- **줌-아웃**: 양선형 보간(bilinear interpolation)으로 $\times \frac{1}{N}$ 다운샘플링
  $$r(\hat{x}) = \text{Bilinear}(\hat{x}) \quad \text{(for zoom-out, } \times \frac{1}{N}\text{)}$$

#### Step 2: Scale-Adaptive Gaussian 최적화

기본 Gaussian $\mathbf{G}$를 최적화된 $\mathbf{G}^{\text{opt}}$로 변형:

$$\mathbf{G}^{\text{opt}} = \mathbf{G} - \beta \nabla \mathcal{L}(x, r(\hat{x})) \tag{4}$$

#### Step 3: Scale-aware Guidance Loss

$$\mathcal{L}(x, r(\hat{x})) = \|x - r(\hat{x})\|^2 \tag{5}$$

여기서 $x$는 새로운 스케일에서 렌더링된 이미지, $r(\hat{x})$는 pseudo-GT입니다.

#### Algorithm 1: 최적화 알고리즘 (원문)

```
Input: Base Gaussians {Gk | k=1,...,K}, viewpoints {Vj}, scale N, iteration S
Output: Optimized Gaussians {G_k^opt | k=1,...,K_opt}

for i = 0, ..., S do
    select vj ∈ Vj randomly
    render {Gk} → basic scale image x̂j  [Eqn. 2]
    construct mipmap r(x̂j): ×N 업샘플링(줌-인) or ×1/N 다운샘플링(줌-아웃)
    render {Gi_k} → new scale image xj  [Eqn. 2]
    optimize {Gi_k} = argmin L(xj, r(x̂j))
    if i mod 100 == 0: densify or prune
end
return {G^opt_k} = {G^(S+1)_k}
```

---

### 2.3 모델 구조

#### Mipmap-GS의 구성 요소

```
┌─────────────────────────────────────────────────────┐
│                   Mipmap-GS 구조                     │
├─────────────┬───────────────────────────────────────┤
│ Base GS     │ 기존 3DGS (사전 학습된 상태)           │
├─────────────┼───────────────────────────────────────┤
│ Mipmap      │ - Zoom-in: SwinIR (초해상도)           │
│ 제안 모듈   │ - Zoom-out: Bilinear Interpolation     │
├─────────────┼───────────────────────────────────────┤
│ 최적화      │ L2 loss 기반 Scale-aware guidance loss │
├─────────────┼───────────────────────────────────────┤
│ Active      │ 전체 최적화 과정에서 지속적 pruning    │
│ Pruning     │ (저불투명 Gaussian 제거)               │
└─────────────┴───────────────────────────────────────┘
```

#### 최적화 대상 파라미터

3DGS의 모든 학습 가능한 파라미터가 조정됩니다:
- **기하학 정보**: 평균 $\mu_k \in \mathbb{R}^{3 \times 1}$, 공분산 $\Sigma_k = RSS^\top R^\top$
- **색상 정보**: 구면 조화(SH) 계수, 불투명도(opacity) $\alpha_k$
- **분포**: Gaussian의 위치 및 밀도

---

### 2.4 성능 향상

#### NeRF Synthetic Dataset 결과

**줌-아웃 비교 (Table 1a)**

| Method | $\times 1/2$ PSNR | $\times 1/4$ PSNR | $\times 1/8$ PSNR |
|--------|-------------------|-------------------|-------------------|
| 3DGS | 27.14 | 21.39 | 17.59 |
| Mip-Splatting | 34.00 | 31.85 | 28.67 |
| **3DGS-Ours** | **34.18** | **32.51** | **30.64** |

→ 3DGS 대비 평균 **+10.40 dB** 향상

**줌-인 비교 (Table 1b)**

| Method | $\times 2$ PSNR | $\times 4$ PSNR | $\times 8$ PSNR |
|--------|-----------------|-----------------|-----------------|
| 3DGS | 23.38 | 19.93 | 18.52 |
| Mip-Splatting | 30.08 | 27.12 | 25.71 |
| **3DGS-Ours** | **31.23** | **28.29** | **26.43** |

→ 3DGS 대비 평균 **+9.25 dB** 향상

#### 학습 시간 비교 (Table 3, bicycle scene)

| Method | Zoom-out | Zoom-in |
|--------|----------|---------|
| 3DGS 전체 학습 | 2h 30m | 8m |
| **3DGS-Ours (추가)** | **+40s** | **+75s** |
| Mip-Splatting | 2h 40m | 8m |

---

### 2.5 한계점

논문에서 명시적·암묵적으로 언급된 한계:

1. **Pseudo-GT 품질 의존성**: 복잡한 배경이 있는 실세계 장면(Mip-NeRF 360)에서 줌-아웃 pseudo-GT 품질이 낮아 성능 향상이 제한됩니다. NeRF Synthetic(고립 객체)에서는 큰 성능 향상을 보이지만, Mip-NeRF 360에서는 Mip-Splatting 대비 소폭 우위에 그칩니다.

2. **줌-인 SR 모델 의존**: 줌-인 pseudo-GT 생성에 SwinIR이라는 외부 초해상도 모델이 필요하며, 이 모델의 성능 한계가 직접 영향을 미칩니다.

3. **테스트 시간 최적화의 뷰 제한**: 훈련 뷰 대신 테스트 뷰로 최적화할 때 성능이 가장 높지만 (+train 설정에서는 오히려 성능 저하), 실시간 렌더링에서 매번 최적화가 필요하다는 구조적 한계가 있습니다.

4. **LPIPS 지표에서의 제한**: 일부 Mip-NeRF 360 줌-인 결과에서 LPIPS 점수가 Mip-Splatting보다 약간 낮은 경우가 있어 지각 품질 측면에서 완전한 우위를 보이지 못합니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화 메커니즘

Mipmap-GS는 여러 측면에서 **out-of-distribution 일반화**를 직접적으로 다룹니다:

#### (1) Test-Time Adaptation (TTA) 패러다임

$$\mathbf{G}^{\text{opt}} = \arg\min_{\mathbf{G}} \mathcal{L}(x, r(\hat{x}))$$

테스트 시간에 대상 스케일에 맞게 Gaussian을 적응시키는 방식은, 고정된 모델로 다양한 분포를 처리하는 것보다 근본적으로 우수한 일반화 전략입니다. 특히 **임의의 줌 팩터 $N$에 대응**할 수 있다는 점에서 연속적 스케일 일반화가 가능합니다.

#### (2) Plug-in 특성에 의한 일반화 폭 확장

Scaffold-GS, Pixel-GS, 동적 장면 3DGS([61] Yang et al.) 등 다양한 3DGS 변형 모델에 적용 가능함을 실험적으로 검증했습니다. 이는 특정 아키텍처에 종속되지 않은 **범용적 일반화 향상 전략**임을 시사합니다.

#### (3) 자기지도(Self-supervised) 방식의 일반화

레이블된 다중 스케일 데이터 없이 기존 렌더링 결과로부터 pseudo-GT를 구성하는 방식은, 데이터 수집 비용 없이 새로운 스케일 도메인으로 일반화하는 혁신적 접근입니다.

### 3.2 일반화 성능 향상을 위한 추가 가능성

#### (A) 다양한 도메인으로의 확장

현재 논문은 관찰 거리(줌 팩터) 변화에 집중하지만, 동일한 프레임워크를 다음에 적용할 수 있습니다:
- **조명 변화**: 다른 조명 조건의 pseudo-GT를 생성하여 조명 일반화
- **카메라 파라미터 변화**: 초점 거리, 렌즈 왜곡 등에 대한 적응
- **날씨/시간 변화**: 동적 장면에서의 도메인 적응

#### (B) Continual Learning 관점에서의 일반화

현재는 특정 스케일 $N$에 대해 별도로 최적화하는데, 여러 스케일을 동시에 학습하거나 이전 스케일 적응 정보를 활용하는 **continual adaptation** 방식으로 발전시킬 수 있습니다:

$$\mathcal{L}_{\text{total}} = \sum_{n \in \mathcal{N}} \lambda_n \mathcal{L}(x_n, r_n(\hat{x}))$$

여기서 $\mathcal{N}$은 목표 스케일 집합입니다.

#### (C) Pseudo-GT 품질 개선을 통한 일반화

현재 줌-아웃에는 bilinear interpolation을, 줌-인에는 SwinIR을 사용하지만:
- **줌-아웃**: Perceptual loss를 활용한 더 정교한 다운샘플링
- **줌-인**: 장면 특화 초해상도(scene-specific SR) 모델 활용
- **양방향**: Diffusion 기반 생성 모델을 통한 더 현실적인 pseudo-GT 생성

#### (D) Meta-Learning 관점

테스트 시간 적응의 효율성을 더욱 높이기 위해, MAML(Model-Agnostic Meta-Learning) 류의 접근법으로 **빠른 스케일 적응을 위한 초기화**를 학습시킬 수 있습니다:

$$\mathbf{G}^* = \mathbf{G} - \alpha \nabla_{\mathbf{G}} \mathcal{L}_{\text{inner}}(\mathbf{G}, r(\hat{x}))$$

### 3.3 현재 일반화의 한계와 개선 방향

```
실세계 장면 (Mip-NeRF 360) → 제한된 성능 향상
원인: 복잡한 배경으로 인한 pseudo-GT 품질 저하

개선 방안:
1. 장면 복잡도에 따른 적응형 pseudo-GT 생성 전략
2. 신뢰도 가중(confidence-weighted) 손실 함수 도입
3. 배경/전경 분리를 통한 선택적 최적화
```

---

## 4. 최신 연구 비교 분석 (2020년 이후)

### 4.1 NeRF 기반 앤티앨리어싱 연구

| 연구 | 연도 | 핵심 방법 | 한계 |
|------|------|----------|------|
| **NeRF** (Mildenhall et al.) | ECCV 2020 | 포인트 기반 레이 샘플링 | 단일 스케일, 앨리어싱 취약 |
| **Mip-NeRF** (Barron et al.) | CVPR 2021 | 원추형 frustum + 사전 필터링 | 속도 느림, 3DGS 미적용 |
| **Mip-NeRF 360** (Barron et al.) | CVPR 2022 | 무경계 장면 + 앤티앨리어싱 | 렌더링 속도 한계 |
| **Zip-NeRF** (Barron et al.) | ICCV 2023 | 그리드 기반 + 앤티앨리어싱 | NeRF 계열, 실시간 불가 |
| **Tri-MipRF** (Hu et al.) | ICCV 2023 | 3단계 Mip 표현 | 복잡한 구조 |

### 4.2 3DGS 기반 앤티앨리어싱 연구

| 연구 | 연도 | 방법 | Mipmap-GS와의 차이 |
|------|------|------|-------------------|
| **3DGS** (Kerbl et al.) | SIGGRAPH 2023 | Gaussian splatting 기반 | 기준선 |
| **Multi-scale 3DGS** (Yan et al.) | CVPR 2024 | 선택적 렌더링, LoD | 줌-아웃만, 하이퍼파라미터 민감 |
| **Mip-Splatting** (Yu et al.) | CVPR 2024 | 3D smooth filter + 2D Mip filter | Gaussian 변형 없음, 세부 손실 |
| **Analytic-Splatting** (Liang et al.) | ECCV 2024 | 픽셀을 면적으로 처리 | 수식 복잡, 일부 스케일 제한 |
| **Octree-GS** (Ren et al.) | arXiv 2024 | LoD 계층 구조 | 레벨 선택 하이퍼파라미터 |
| **SAGS** (Song et al.) | arXiv 2024 | 스케일 적응형 2D 필터 | Gaussian 변형 없음 |
| **Mipmap-GS (본 논문)** | arXiv 2024 | Gaussian 직접 변형 + pseudo-GT | **근본적 해결, 양방향 적용** |

### 4.3 핵심 차별점 비교

```
필터 기반 방법 (Mip-Splatting, SAGS):
  장점: 학습 중 적용 가능, 추가 학습 불필요
  단점: Gaussian 자체는 변하지 않음, 세부 정보 손실

LoD 기반 방법 (Octree-GS, Multi-scale 3DGS):
  장점: 효율적인 렌더링
  단점: 하이퍼파라미터 민감, 줌-아웃에만 집중

Mipmap-GS:
  장점: Gaussian 자체 변형, 줌-인/아웃 모두 적용, plug-in
  단점: 테스트 시간 추가 최적화 필요, SR 모델 의존
```

---

## 5. 향후 연구에 미치는 영향과 고려사항

### 5.1 향후 연구에 미치는 영향

#### (1) Test-Time Adaptation 패러다임의 3DGS 확산

Mipmap-GS는 **학습 후 테스트 시간에 추가 최적화**를 수행하는 패러다임을 3DGS에 성공적으로 도입했습니다. 이는 향후:
- 조명 변화 적응 TTA
- 날씨/계절 변화 TTA
- 카메라 파라미터 불일치 TTA

등 다양한 도메인 적응 연구를 자극할 것입니다.

#### (2) Self-supervised Pseudo-GT 설계 방법론

레이블 데이터 없이 렌더링 결과로부터 pseudo-GT를 구성하는 방법론은, **데이터 효율적인 3DGS 개선 연구**의 새로운 방향을 제시합니다. 특히 다음 연구들을 촉발할 수 있습니다:
- Diffusion 기반 pseudo-GT 생성
- 멀티뷰 일관성을 고려한 pseudo-GT 설계

#### (3) Plug-in 모듈 설계 철학

기존 3DGS 변형 모델에 쉽게 적용 가능한 plug-in 방식은, **3DGS 생태계에서 모듈형 개선의 표준**이 될 가능성이 있습니다.

#### (4) 초해상도와 3DGS의 결합

SwinIR을 zoom-in pseudo-GT 생성에 활용한 접근은, **2D 이미지 처리 기술과 3D 장면 표현의 융합** 연구를 더욱 활성화시킬 것입니다.

---

### 5.2 향후 연구 시 고려할 점

#### ① Pseudo-GT 품질 제어

```
현재 문제: 실세계 복잡 장면에서 pseudo-GT 품질이 낮음
고려 사항:
  - 장면 복잡도 자동 측정 및 적응형 전략 선택
  - Perceptual quality metric 기반 pseudo-GT 필터링
  - Uncertainty-aware 가중 손실 함수 설계
```

#### ② 연속적 스케일 일반화

현재 방법은 특정 스케일 $N$에 대해 최적화합니다. 향후 연구에서는:

$$\mathcal{L}_{\text{continuous}} = \int_{N_{\min}}^{N_{\max}} w(N) \cdot \mathcal{L}(x_N, r_N(\hat{x})) \, dN$$

형태의 연속 스케일 최적화를 고려해야 합니다.

#### ③ 실시간성과 적응의 균형

테스트 시간 최적화는 수십 초가 추가되므로, **실시간 응용(VR/AR)** 에서는:
- 경량화된 적응 모듈 설계
- 미리 여러 스케일을 캐싱하는 전략
- 점진적(progressive) 적응 렌더링

을 고려해야 합니다.

#### ④ 다양한 분포 이동(Distribution Shift)으로의 확장

스케일 변화 외에도 3DGS의 out-of-distribution 문제는 다양합니다:

| 분포 이동 유형 | Mipmap-GS 적용 가능성 | 필요한 추가 연구 |
|--------------|---------------------|----------------|
| 스케일 변화 | ✅ 직접 적용 가능 | 완료 |
| 조명 변화 | ⚠️ 부분적 | 조명별 pseudo-GT 생성 |
| 뷰포인트 극단값 | ⚠️ 제한적 | 새로운 뷰 생성 방법 필요 |
| 동적 장면 | ✅ 증명됨 | 시간 차원 추가 고려 |

#### ⑤ 평가 지표의 다양화

PSNR, SSIM, LPIPS 외에:
- **FID(Fréchet Inception Distance)**: 생성 품질 전반
- **스케일별 주파수 분석**: 앨리어싱의 직접적 측정
- **사용자 연구**: 지각적 품질 검증

를 포함한 더 포괄적인 평가가 필요합니다.

#### ⑥ 메모리 및 저장 효율성

Active pruning으로 Gaussian 수를 줄이지만, 여러 스케일에 대응하기 위해 다수의 Gaussian 집합을 유지해야 할 경우 메모리 부담이 증가할 수 있습니다. **적응형 압축 기법**과의 결합을 고려해야 합니다.

---

## 참고자료

**본 논문:**
- Li, J., Shi, Y., Cao, J., Ni, B., Zhang, W., Zhang, K., & Van Gool, L. (2024). *Mipmap-GS: Let Gaussians Deform with Scale-specific Mipmap for Anti-aliasing Rendering.* arXiv:2408.06286v1.

**논문 내 주요 인용 문헌:**
- Kerbl, B., et al. (2023). *3D Gaussian Splatting for Real-Time Radiance Field Rendering.* ACM TOG. [3DGS 기준선]
- Barron, J.T., et al. (2021). *Mip-NeRF: A Multiscale Representation for Anti-aliasing Neural Radiance Fields.* CVPR. [Mip-NeRF]
- Barron, J.T., et al. (2022). *Mip-NeRF 360: Unbounded Anti-aliased Neural Radiance Fields.* CVPR. [Mip-NeRF 360]
- Yu, Z., et al. (2023). *Mip-Splatting: Alias-free 3D Gaussian Splatting.* arXiv:2311.16493. [주요 비교 방법]
- Liang, Z., et al. (2024). *Analytic-Splatting: Anti-aliased 3D Gaussian Splatting via Analytic Integration.* ECCV. [비교 방법]
- Lu, T., et al. (2024). *Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering.* CVPR. [비교 방법]
- Liang, J., et al. (2021). *SwinIR: Image Restoration Using Swin Transformer.* CVPR. [Zoom-in pseudo-GT 생성]
- Zwicker, M., et al. (2001). *EWA Volume Splatting.* IEEE Visualization. [EWA 필터]
- Ren, K., et al. (2024). *Octree-GS: Towards Consistent Real-time Rendering with LOD-structured 3D Gaussians.* arXiv:2403.17898. [비교 방법]
- Zhang, Z., et al. (2024). *Pixel-GS: Density Control with Pixel-aware Gradient for 3D Gaussian Splatting.* ECCV. [비교 방법]
- Yan, Z., et al. (2024). *Multi-scale 3D Gaussian Splatting for Anti-aliased Rendering.* CVPR. [비교 방법]
- Barron, J.T., et al. (2023). *Zip-NeRF: Anti-aliased Grid-based Neural Radiance Fields.* ICCV. [비교 방법]

> **⚠️ 주의**: 본 답변은 제공된 PDF 원문(arXiv:2408.06286v1)에 기반하여 작성되었습니다. 수식 번호, 실험 수치, 알고리즘 등은 원문을 직접 인용하였으며, 논문에 명시되지 않은 내용에 대해서는 명확히 "가능성" 또는 "추후 연구 방향"으로 구분하여 서술하였습니다.
