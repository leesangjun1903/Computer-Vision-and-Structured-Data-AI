# NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis

> **논문 정보**: Mildenhall, B., Srinivasan, P.P., Tancik, M., Barron, J.T., Ramamoorthi, R., Ng, R. — ECCV 2020

---

## 1. 핵심 주장 및 주요 기여 (요약)

NeRF는 sparse한 입력 뷰 집합으로부터 복잡한 장면의 새로운 시점(novel view)을 합성하는 데 최첨단 결과를 달성하는 방법을 제시한다. 이 알고리즘은 완전 연결(non-convolutional) 심층 네트워크를 사용하여 장면을 표현하며, 입력은 연속적인 5D 좌표 $(x, y, z, \theta, \phi)$이고, 출력은 해당 공간 위치에서의 체적 밀도(volume density)와 시점 의존적 방출 복사(view-dependent emitted radiance)이다.

원 논문은 NeRF를 뷰 합성의 핵심 방법으로 대중화하였으며, 사실적인(photorealistic) 출력을 생성하기 위한 3가지 주요 기여를 한다.

### 3대 핵심 기여:

| 기여 | 설명 |
|------|------|
| **(1) 5D Neural Radiance Field 표현** | 복잡한 기하와 외관을 가진 장면을 연속 함수로 표현 |
| **(2) Positional Encoding** | 고주파 세부사항 복원을 위한 위치 인코딩 기법 |
| **(3) Hierarchical Sampling** | 효율적 렌더링을 위한 계층적 샘플링 전략 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

2020년 3월, NeRF는 암시적(implicit) 신경망 기반 장면 표현과 새로운 뷰 합성을 가능하게 하여 컴퓨터 비전 분야에 혁명을 일으켰다. 기존 뷰 합성 방법들은 다음과 같은 한계가 있었다:

- **이산 표현의 한계**: 복셀(voxel) 기반 방법은 메모리 사용량이 $O(N^3)$으로 해상도에 제한됨
- **메쉬 기반의 한계**: 복잡한 반투명 재질이나 세밀한 기하 표현에 부적합
- **일반화 부재**: 장면별(per-scene) 최적화에 과도한 시간 소요

### 2.2 제안 방법 및 수식

#### (A) Neural Radiance Field 정의

NeRF는 장면을 연속 함수 $F_\Theta$로 모델링한다:

$$F_\Theta : (\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)$$

여기서:
- $\mathbf{x} = (x, y, z)$: 3D 공간 좌표
- $\mathbf{d} = (\theta, \phi)$: 시선 방향(viewing direction)
- $\mathbf{c} = (r, g, b)$: RGB 색상
- $\sigma$: 체적 밀도(volume density)

#### (B) Volume Rendering 수식

카메라 광선(camera ray)을 따라 5D 좌표를 쿼리하고, 고전적인 체적 렌더링(volume rendering) 기법을 사용하여 출력 색상과 밀도를 이미지로 투영한다. 카메라 광선 $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$를 따라 기대되는 색상 $C(\mathbf{r})$은 다음과 같이 계산된다:

$$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \cdot \sigma(\mathbf{r}(t)) \cdot \mathbf{c}(\mathbf{r}(t), \mathbf{d}) \, dt$$

여기서 누적 투과율(accumulated transmittance) $T(t)$는:

$$T(t) = \exp\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s)) \, ds\right)$$

실제 구현 시 **수치적 적분(quadrature)**을 사용한 이산화:

$$\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i \cdot \alpha_i \cdot \mathbf{c}_i$$

$$T_i = \prod_{j=1}^{i-1}(1 - \alpha_j), \quad \alpha_i = 1 - \exp(-\sigma_i \delta_i)$$

여기서 $\delta_i = t_{i+1} - t_i$는 인접 샘플 사이의 거리이다.

#### (C) Positional Encoding (위치 인코딩)

신경망이 고주파(high-frequency) 함수를 학습하도록 돕기 위해, 입력 좌표를 더 높은 차원으로 매핑하는 위치 인코딩 $\gamma$를 적용한다:

$$\gamma(p) = \left(\sin(2^0\pi p), \cos(2^0\pi p), \sin(2^1\pi p), \cos(2^1\pi p), \ldots, \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p)\right)$$

논문에서는 $\mathbf{x}$에 대해 $L=10$ (60차원), $\mathbf{d}$에 대해 $L=4$ (24차원)을 사용한다.

#### (D) 손실 함수

체적 렌더링은 자연스럽게 미분 가능(differentiable)하므로, 표현을 최적화하는 데 필요한 유일한 입력은 알려진 카메라 포즈를 가진 이미지 집합이다.

$$\mathcal{L} = \sum_{\mathbf{r} \in \mathcal{R}} \left[ \left\| \hat{C}_c(\mathbf{r}) - C_{gt}(\mathbf{r}) \right\|_2^2 + \left\| \hat{C}_f(\mathbf{r}) - C_{gt}(\mathbf{r}) \right\|_2^2 \right]$$

여기서 $\hat{C}\_c$는 coarse network, $\hat{C}\_f$는 fine network의 렌더링 결과이며, $C_{gt}$는 ground truth 색상이다.

### 2.3 모델 구조

```
입력 (x, y, z) → Positional Encoding (γ(x), 60D)
         ↓
   MLP Layer 1-4 (256 units, ReLU)
         ↓
   Skip Connection (Layer 5에서 입력 재결합)
         ↓
   MLP Layer 5-8 (256 units, ReLU)
         ↓
   σ 출력 (밀도, 시점 독립적)
         ↓
   + Viewing Direction (γ(d), 24D) 결합
         ↓
   MLP Layer 9 (128 units, ReLU)
         ↓
   RGB 색상 출력 (시점 의존적)
```

**Hierarchical Volume Sampling (계층적 샘플링):**
- **Coarse Network**: 광선을 따라 $N_c = 64$개 점을 균일하게 샘플링
- **Fine Network**: Coarse 결과의 밀도 분포에 기반하여 $N_f = 128$개 추가 점을 중요도 샘플링(importance sampling)

### 2.4 성능

NeRF는 복잡한 기하와 외관을 가진 장면의 사실적인 새로운 뷰를 효과적으로 최적화하여 렌더링하는 방법을 설명하며, 신경 렌더링 및 뷰 합성에 관한 이전 연구를 능가하는 결과를 입증했다.

| 메트릭 | NeRF | SRN | LLFF | NV |
|--------|------|-----|------|----|
| **PSNR ↑** | **31.01** | 22.26 | 24.88 | 26.05 |
| **SSIM ↑** | **0.947** | 0.846 | 0.911 | 0.893 |
| **LPIPS ↓** | **0.081** | 0.170 | 0.114 | 0.160 |

*(Blender Synthetic 데이터셋 기준)*

### 2.5 한계점

| 한계 | 설명 |
|------|------|
| **느린 학습/렌더링** | 장면당 1~2일 최적화, 렌더링에 수초 소요 |
| **장면 고유 모델** | 새 장면마다 처음부터 재학습 필요 (일반화 불가) |
| **정적 장면 한정** | 동적 객체나 시간적 변화 처리 불가 |
| **카메라 포즈 필요** | 정확한 카메라 외부/내부 파라미터 사전 필요 |
| **메모리 집약적** | 광선당 다수의 MLP 쿼리로 인한 연산량 |

---

## 3. 모델의 일반화 성능 향상 가능성

NeRF의 가장 큰 한계 중 하나는 **장면별 최적화(per-scene optimization)** 패러다임으로, 이는 본질적으로 일반화를 제한한다. 이를 개선하기 위한 연구 방향은 다음과 같다:

### 3.1 Generalizable NeRF의 핵심 전략

일반화 가능한(generalizable) NeRF는 소스 뷰(source views)에서 장면 특징(scene features)을 추출하여 바닐라 NeRF를 조건화(conditioning)함으로써, 장면별 최적화의 필요성을 제거하고 상용 AR/VR 응용에 가장 실현 가능한 NeRF 솔루션으로 자리잡고 있다.

#### (A) 이미지 조건화 접근법

PixelNeRF는 처음으로 소스 이미지의 픽셀 특징(pixel features)을 위치 정보만 사용하는 바닐라 NeRF에 통합했다. 이를 통해:

$$F_\Theta(\gamma(\mathbf{x}), \mathbf{d}, \mathbf{W}(\mathbf{x})) \rightarrow (\mathbf{c}, \sigma)$$

여기서 $\mathbf{W}(\mathbf{x})$는 CNN으로 추출한 이미지 특징이다.

이 모델은 단일 보정 이미지만으로도 학습 가능하지만, 단순한 기하에만 권장된다.

#### (B) Multi-View Stereo 기반 접근법

MVSNeRF는 장면별 최적화 대신, 단 세 장의 인접 입력 뷰로부터 빠른 네트워크 추론을 통해 복사 필드를 재구성하는 범용 심층 신경망을 제안한다. 이 접근법은 multi-view stereo에서 널리 사용되는 plane-swept cost volume을 활용하여 기하 인식 장면 추론을 수행한다.

이 접근법은 장면 간 일반화가 가능하며(심지어 학습 데이터와 완전히 다른 실내 장면에서도), 세 장의 입력 이미지만으로 사실적인 뷰 합성 결과를 생성할 수 있다. 또한 밀집 이미지가 캡처된 경우 추정된 복사 필드를 쉽게 미세 조정할 수 있어 NeRF보다 상당히 적은 최적화 시간으로 더 높은 렌더링 품질을 달성한다.

#### (C) Transformer 기반 일반화

IBRNet과 GNT는 MLP 또는 트랜스포머를 통해 이러한 관계를 암시적으로 포착한다. 샘플링-렌더링 과정에서, 대부분의 연구들은 트랜스포머 기반 아키텍처를 활용하여 샘플링 포인트 특징을 집계하고 전통적인 볼륨 렌더링을 학습 가능한 기법으로 대체한다.

#### (D) HyperNetwork 기반 접근법 (InsertNeRF, ICLR 2024)

InsertNeRF는 GNT 대비 일관되게 우수한 렌더링 성능과 추론 효율성을 보여주며, 이 성공은 새로운 일반화 패러다임과 HyperNet 모듈의 컴팩트한 구조에 기인한다.

#### (E) 확산 모델(Diffusion) 기반 접근법

ID-NeRF는 사전 학습된 지식(pre-trained knowledge)을 score-based distillation을 통해 상상적 잠재 공간(imaginative latent space)으로 증류하고, attention 기반 정제 모듈을 통해 sparse 입력에서 추출된 재투영 특징을 개선한다. 다양한 데이터셋에 대한 실험 결과, 특히 sparse 설정에서 일반화 가능한 방식으로 새로운 뷰를 합성하는 데 효과적임을 입증한다.

### 3.2 일반화 성능 향상 로드맵

```
[원본 NeRF: Per-Scene]
        ↓
[PixelNeRF/IBRNet: Image-Conditioned Generalization]
        ↓
[MVSNeRF: Geometry-Aware Generalization via Cost Volume]
        ↓
[GeoNeRF: Occlusion-Aware Geometry Priors]
        ↓
[InsertNeRF/GNT: Transformer + HyperNet]
        ↓
[ID-NeRF: Diffusion-Guided Generalization]
```

---

## 4. 앞으로의 연구에 미치는 영향과 고려할 점

### 4.1 NeRF가 후속 연구에 미친 영향

2020년 NeRF 프레임워크 개발 이래, 성능과 기능을 극적으로 개선하는 많은 변형과 확장이 이루어졌다. 이 모델의 최첨단 결과와 사실적 렌더링 달성 능력은 뷰 합성 분야 및 그 너머에서 많은 기회를 제공하며, NeRF는 그 자체로 연구 분야가 되어 지속적으로 중요한 발전이 이루어지고 있다. 응용 분야로는 영화 제작의 3D 장면 렌더링, 3D 그래픽 생성, 가상 렌더링과 사이트 워크스루 등이 있다.

NeRF 모델은 로보틱스, 도시 매핑, 자율 내비게이션, 가상 현실/증강 현실 등 다양한 분야에서 응용되고 있다.

### 4.2 향후 연구 시 고려할 점

최근 NeRF 연구는 렌더링 효율성 개선, 소수 뷰(few-view) 합성 최적화, 렌더링 품질 향상, 자기 지도 학습(self-supervised learning) 등 핵심 영역에 집중하고 있으며, 실시간 렌더링 요구 충족, 3D 재구성의 정확도와 일반화 능력 향상, 대규모 어노테이션 데이터에 대한 의존도 감소를 목표로 한다.

| 연구 방향 | 세부 내용 |
|-----------|----------|
| **실시간 렌더링** | Instant-NGP, Plenoxels 등 해시 인코딩 기반 가속 |
| **소수 뷰 합성** | RegNeRF, FreeNeRF, DietNeRF 등 정규화 기반 접근 |
| **동적 장면** | D-NeRF, Nerfies, HyperNeRF 등 시간축 확장 |
| **대규모 장면** | Block-NeRF, Mega-NeRF 등 분할 학습 전략 |
| **합성-실제 도메인 갭** | ContraNeRF 등 도메인 적응 기법 |
| **3DGS와의 통합** | NeRF-GS 등 하이브리드 접근법 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 주요 후속 연구 비교표

| 방법 | 연도 | 핵심 개선 | 학습 시간 | 일반화 | 실시간 |
|------|------|-----------|-----------|--------|--------|
| **Original NeRF** | 2020 | - (baseline) | ~1-2일 | ✗ | ✗ |
| **Mip-NeRF** | 2021 | 안티앨리어싱 (원뿔 추적) | ~1일 | ✗ | ✗ |
| **PixelNeRF** | 2021 | CNN 조건화, 소수 뷰 | 사전학습 | ✓ | ✗ |
| **IBRNet** | 2021 | Transformer 집계 | 사전학습 | ✓ | ✗ |
| **MVSNeRF** | 2021 | Cost Volume 기반 | 사전학습 + 15분 미세조정 | ✓ | ✗ |
| **Instant-NGP** | 2022 | 다중해상도 해시 인코딩 | ~5분 | ✗ | △ |
| **Mip-NeRF 360** | 2022 | 무한 장면, 수축 함수 | ~수시간 | ✗ | ✗ |
| **Zip-NeRF** | 2023 | Mip-NeRF 360 + Instant-NGP 결합 | 고속 | ✗ | △ |
| **3D Gaussian Splatting** | 2023 | 명시적 가우시안, 래스터화 | ~수십 분 | ✗ | ✓ |
| **InsertNeRF** | 2024 | HyperNet 기반 일반화 | 사전학습 | ✓ | △ |
| **GP-NeRF** | 2024 | 의미론적 장면 이해 + 일반화 | 사전학습 | ✓ | ✗ |

### 5.2 핵심 기술별 비교

#### (A) Instant-NGP (Müller et al., 2022)
Instant-NGP는 학습 가능한 다중 해상도 해시 인코딩(multi-resolution hash encoding)을 도입하여 장면을 효율적으로 적합시킨다. NeRF의 positional encoding을 해시 테이블 기반 인코딩으로 대체하여 학습 시간을 수 시간에서 수 분으로 단축했다.

#### (B) Mip-NeRF (Barron et al., 2021)
Mip-NeRF는 원뿔 추적(cone tracing)의 다중 스케일 속성과 자동 안티앨리어싱으로 NeRF를 향상시킨다. 광선을 원뿔로 확장하여 integrated positional encoding을 사용:

$$\text{IPE}(\mu, \Sigma) = \mathbb{E}_{\mathbf{x} \sim \mathcal{N}(\mu, \Sigma)}[\gamma(\mathbf{x})]$$

#### (C) Zip-NeRF (Barron et al., 2023)
Zip-NeRF는 Mip-NeRF 360의 스케일 인식 및 안티앨리어싱 속성과 Instant-NGP의 빠른 학습을 결합한 모델이다. 다중 샘플링과 사전 필터링 기법으로 공간 안티앨리어싱 문제를 해결하고, "z-aliasing" 문제를 해결하기 위한 새로운 손실 함수를 도입했다. 이 접근법은 렌더링 품질을 개선할 뿐만 아니라 학습 속도를 크게 가속하여, Mip-NeRF 360보다 24배 빠르다.

#### (D) 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023)

2023년 8월, NeRF 기반 프레임워크의 직접적인 경쟁자인 Gaussian Splatting이 제안되어 엄청난 모멘텀을 얻으며 새로운 뷰 합성의 지배적 프레임워크로서 NeRF 기반 연구를 추월하고 있다.

3DGS는 카메라 캘리브레이션 중 생성된 sparse 포인트로부터 시작하여 3D 가우시안으로 장면을 표현하며, 이방성 공분산(anisotropic covariance) 최적화를 통해 정확한 장면 표현을 달성하고, 이방성 스플래팅을 지원하는 빠른 가시성 인식 렌더링 알고리즘을 개발하여 학습 가속과 실시간 렌더링을 모두 가능하게 한다.

**NeRF vs. 3DGS 핵심 비교:**

| 특성 | NeRF | 3DGS |
|------|------|------|
| 표현 | 암시적(implicit): 가중치 파일이 사람이 해석하기 어려운 난수처럼 보임 | 명시적(explicit): 파일이 사람이 읽을 수 있음 |
| 렌더링 방식 | Ray Tracing (레이 마칭) | Rasterization (래스터화) |
| 메모리 | 수십 MB의 작은 신경망 | 수백만 가우시안으로 약 1GB (10-100배 차이) |
| 렌더링 속도 | 느린 레이 트레이싱 (광선당 다수의 느린 신경망 호출) | 복잡한 장면에서도 90fps 이상 |

#### (E) NeRF-GS 하이브리드 (2025)

NeRF-GS는 Neural Radiance Fields와 3D Gaussian Splatting을 공동 최적화하는 새로운 프레임워크로, NeRF의 고유한 연속 공간 표현을 활용하여 가우시안 초기화 민감도, 제한된 공간 인식, 약한 가우시안 간 상관관계 등 3DGS의 여러 한계를 완화한다.

### 5.3 일반화 관련 최신 연구 특별 비교

신경 장면 표현의 일반화 능력을 향상시키는 것은 도전적인 문제이다. pixelNeRF, IBRNet, MVSNeRF, GeoNeRF 등의 최근 연구들이 신경 복사 필드 기반의 일반화 가능한 새로운 뷰 합성을 달성하는 방법을 탐구하고 있다.

pixelNeRF, GRF, MINE, SRF, IBRNet, MVSNeRF 등의 접근법은 인접 뷰의 소스 이미지에서 추출한 특징으로 NeRF 렌더러를 조건화하는 것이 공통 동기이다. 그러나 이 모델들의 장면 기하와 가림(occlusion)에 대한 이해는 제한적이어서, 렌더링 출력에 원치 않는 아티팩트가 발생한다.

---

## 6. 결론 및 전망

NeRF는 신경 암시적 표현이라는 새로운 패러다임을 확립하여, 2020년 이후 컴퓨터 비전과 그래픽스 연구의 핵심 축이 되었다. Gaussian Splatting 이전 시대에는 NeRF가 새로운 뷰 합성과 3D 암시적/하이브리드 표현 학습 분야를 지배했으며, Gaussian Splatting 이후 시대에는 NeRF와 암시적/하이브리드 신경 필드가 더 전문화된(niche) 응용 분야를 찾고 있다.

향후 연구의 핵심 고려사항:
1. **일반화 vs. 품질의 트레이드오프**: 범용 모델은 장면 고유 최적화 대비 여전히 품질 격차 존재
2. **실시간 렌더링과 품질의 양립**: 3DGS의 도전에 대응하는 NeRF 기반 실시간 솔루션 필요
3. **소수 뷰 / 단일 뷰 재구성**: 실용적 배포를 위한 데이터 효율성 극대화
4. **다중 모달리티 융합**: LiDAR, 이벤트 카메라, 시맨틱 정보와의 통합
5. **NeRF-3DGS 하이브리드**: 양 패러다임의 장점을 결합하는 통합 프레임워크

---

### 참고 자료

1. Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis," ECCV 2020 — [arXiv:2003.08934](https://arxiv.org/abs/2003.08934)
2. NeRF Project Page — [matthewtancik.com/nerf](https://www.matthewtancik.com/nerf)
3. "Neural Radiance Fields (NeRFs): A Review and Some Recent Developments" — [arXiv:2305.00375](https://arxiv.org/abs/2305.00375)
4. Gao et al., "NeRF: Neural Radiance Field in 3D Vision: A Comprehensive Review (Updated Post-Gaussian Splatting)" — [arXiv:2210.00379](https://arxiv.org/abs/2210.00379)
5. "Neural Radiance Field-based Visual Rendering: A Comprehensive Review" — [arXiv:2404.00714](https://arxiv.org/abs/2404.00714)
6. Chen et al., "MVSNeRF: Fast Generalizable Radiance Field Reconstruction from Multi-View Stereo," ICCV 2021
7. Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering," SIGGRAPH 2023 — [repo-sam.inria.fr](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
8. InsertNeRF, ICLR 2024 — [proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2024/)
9. Li et al., "GP-NeRF: Generalized Perception NeRF for Context-Aware 3D Scene Understanding," CVPR 2024
10. Johari et al., "GeoNeRF: Generalizing NeRF with Geometry Priors," CVPR 2022
11. ID-NeRF: Indirect diffusion-guided neural radiance fields — ScienceDirect, 2024
12. Gen-NeRF: Efficient and Generalizable Neural Radiance Fields via Algorithm-Hardware Co-Design — ISCA 2023
13. ColNeRF: Collaboration for Generalizable Sparse Input Neural Radiance Field — [arXiv:2312.09095](https://arxiv.org/abs/2312.09095)
14. NeRF-GS: NeRF Is a Valuable Assistant for 3D Gaussian Splatting — [arXiv:2507.23374](https://arxiv.org/abs/2507.23374)
15. Mark Boss, "NeRF at CVPR 2023" — [markboss.me](https://markboss.me/post/nerf_at_cvpr23/)
16. awesome-NeRF Paper List — [github.com/awesome-NeRF](https://github.com/awesome-NeRF/awesome-NeRF)

# NeRF: Neural Radiance Fields for View Synthesis | 3D reconstruction, Neural rendering, Novel view synthesis

## 개요
NeRF(Neural Radiance Fields for View Synthesis)는 2020년 발표된 혁신적인 3D 장면 표현 및 새로운 시점 합성 기술로, 컴퓨터 비전과 3D 그래픽스 분야에 큰 패러다임 변화를 가져온 획기적인 연구이다[1]. 이 논문은 희소한 입력 뷰를 사용하여 복잡한 장면의 새로운 시점을 합성하는 최첨단 결과를 달성하는 방법을 제시한다[1].

## 핵심 주장과 주요 기여
### 주요 기여사항
NeRF의 핵심 기여는 다음과 같이 요약할 수 있다[1]:

- **5D 연속적 장면 표현**: 공간 위치(x, y, z)와 시점 방향(θ, φ)을 포함하는 5차원 좌표를 입력으로 받는 완전 연결 신경망을 통해 장면을 표현[1]
- **미분 가능한 볼륨 렌더링**: 고전적인 볼륨 렌더링 기법과 결합된 미분 가능한 렌더링 절차[1]
- **위치 인코딩(Positional Encoding)**: MLP가 고주파수 함수를 표현할 수 있도록 하는 입력 좌표의 변환 기법[1]
- **계층적 샘플링(Hierarchical Sampling)**: 효율적인 샘플링을 위한 coarse-to-fine 전략[1]

### 기술적 혁신성
NeRF는 기존의 3D 표현 방식과 달리 **연속적이고 암시적인 장면 표현**을 제공한다[1]. 이는 voxel grid나 mesh와 같은 이산적 표현의 한계를 극복하며, 훨씬 적은 저장 공간(5MB vs 15GB)으로 고품질의 3D 장면을 표현할 수 있다[1].

## 해결하고자 하는 문제와 제안 방법
### 문제 정의
NeRF가 해결하고자 하는 핵심 문제는 **희소한 입력 이미지들로부터 새로운 시점의 사진같은 이미지를 생성**하는 것이다[1]. 기존의 방법들은 다음과 같은 한계를 가지고 있었다:

- 메시 기반 방법의 지역 최소값 및 초기화 문제[1]
- 볼륨 표현의 높은 계산 복잡도와 메모리 요구사항[1]
- 고해상도 장면 표현 시의 확장성 문제[1]

### 제안 방법론
#### 1. 5D 신경 복사장(Neural Radiance Field)

NeRF는 장면을 5차원 벡터 함수로 표현한다[1]:

$$ F_\Theta : (x, d) \rightarrow (c, \sigma) $$

여기서:
- **입력**: 3D 위치 x = (x, y, z)와 2D 시점 방향 d = (θ, φ)
- **출력**: RGB 색상 c = (r, g, b)와 볼륨 밀도 σ

#### 2. 볼륨 렌더링 수식

카메라 광선 r(t) = o + td를 따라 색상을 렌더링하는 수식은 다음과 같다[1]:

$$ C(r) = \int_{t_n}^{t_f} T(t)\sigma(r(t))c(r(t), d)dt $$

여기서 T(t)는 누적 투과도이다:

$$ T(t) = \exp\left(-\int_{t_n}^{t} \sigma(r(s))ds\right) $$

#### 3. 위치 인코딩(Positional Encoding)

MLP가 고주파수 신호를 학습할 수 있도록 하기 위한 핵심 기법이다[1]:

$$ \gamma(p) = \left(\sin(2^0\pi p), \cos(2^0\pi p), \ldots, \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p)\right) $$

- 위치 좌표: L = 10
- 시점 방향: L = 4

이는 Transformer의 위치 인코딩과 유사하지만, 연속적인 입력 좌표를 고차원 공간으로 매핑하여 MLP가 고주파수 함수를 더 쉽게 근사할 수 있도록 한다는 차이가 있다[2][3].

#### 4. 계층적 볼륨 샘플링

효율적인 렌더링을 위해 두 개의 네트워크를 사용한다[1]:

- **Coarse 네트워크**: 광선을 따라 64개 지점을 균등 샘플링
- **Fine 네트워크**: coarse 네트워크의 밀도 예측을 바탕으로 128개 추가 지점 샘플링

이는 중요한 영역에 더 많은 샘플을 할당하여 계산 효율성을 높인다[1][4][5].

### 모델 구조
NeRF의 네트워크 구조는 다음과 같다[1]:

- **8개 완전 연결 레이어** (각 256 채널)
- **ReLU 활성화 함수**
- **5번째 레이어에 skip connection**
- **별도의 브랜치로 시점 의존적 색상 예측**

훈련 시 4,096개의 광선을 배치로 사용하며, 각 광선당 64개(coarse) + 128개(fine) = 192개의 샘플을 사용한다[1].

## 성능 향상 및 실험 결과
### 정량적 성능
NeRF는 다음 데이터셋에서 기존 방법들을 크게 앞서는 성능을 보였다[1]:

| 데이터셋 | PSNR | SSIM | LPIPS |
|---------|------|------|--------|
| Diffuse Synthetic 360° | 40.15 | 0.991 | 0.023 |
| Realistic Synthetic 360° | 31.01 | 0.947 | 0.081 |
| Real Forward-Facing | 26.50 | 0.811 | 0.250 |

### 비교 방법론

NeRF는 다음 최신 방법들과 비교되었다[1]:

- **Neural Volumes (NV)**: 128³ 복셀 그리드를 사용하는 3D CNN 기반 방법
- **Scene Representation Networks (SRN)**: 불투명 표면을 가정하는 MLP 기반 방법  
- **Local Light Field Fusion (LLFF)**: 멀티플레인 이미지를 사용하는 전방향 장면 특화 방법

### 정성적 결과
실험 결과, NeRF는 다음과 같은 우수한 성능을 보였다[1]:

- **복잡한 기하학적 구조** (예: 배의 밧줄, 레고의 기어)의 정밀한 재현
- **비-Lambertian 재질** (예: 반사 표면, 투명도)의 정확한 표현
- **시점 의존적 효과** (예: 정반사)의 자연스러운 렌더링

### 저장 효율성
NeRF의 주요 장점 중 하나는 **극도로 효율적인 저장**이다[1]:

- NeRF 모델 크기: **5MB**
- LLFF 출력 크기: **15GB 이상**
- **3,000배의 압축 효과**

## 한계점 분석
### 주요 한계사항
NeRF는 혁신적인 성과에도 불구하고 다음과 같은 한계점들을 가지고 있다[1][6][7][8]:

#### 1. 긴 훈련 시간
- **100-300k 반복 필요** (1-2일 소요)[1]
- 장면별 개별 최적화 필요[6]
- GPU 메모리 집약적 처리[7]

#### 2. 느린 렌더링 속도  
- **픽셀당 수백 번의 네트워크 평가** 필요[7]
- 800×800 이미지 렌더링에 **30초 소요**[1]
- 실시간 애플리케이션에 부적합[7]

#### 3. 많은 입력 이미지 필요
- 합성 데이터: 100개 이상의 이미지[1]
- 실제 장면: 20-62개 이미지[1]  
- 희소한 입력에서 성능 저하[9][10]

#### 4. 일반화 능력의 한계
- 장면별 개별 훈련 필요[6]
- 새로운 장면에 대한 일반화 어려움[11]
- 동적 장면 처리 불가[1]

### 기술적 제약사항
#### 1. 기하학적 제약
- 물체 경계의 모호성 문제[6]
- 복잡한 구조에서의 자기 가림 현상[12]
- 투명하거나 반사적인 표면 처리의 어려움[6]

#### 2. 데이터 품질 의존성
- 고품질 다양한 각도의 이미지 필요[6]
- 카메라 포즈 정보의 정확성에 민감[10]
- 조명 변화에 대한 취약성[6]

## 일반화 성능 향상 가능성
### 일반화 NeRF 연구 동향
NeRF의 일반화 성능 향상을 위한 여러 연구들이 활발히 진행되고 있다[13][11][14][15]:

#### 1. 사전 학습된 특징 활용

**PixelNeRF**[2]는 CNN을 통해 추출된 이미지 특징을 조건으로 사용하여 단일 이미지나 소수의 이미지만으로도 새로운 시점을 생성할 수 있다. 이는 장면 사전 정보(scene prior)를 학습하여 일반화 능력을 크게 향상시킨다[2].

**CP-NeRF**[16]는 Feature Pyramid Network(FPN)를 사용하여 서로 다른 스케일의 글로벌 및 로컬 정보를 추출하고, 이를 통해 모델 파라미터를 생성하는 방식으로 cross-scene 일반화를 달성한다.

#### 2. Transformer 기반 접근법

**TransNeRF**[15]는 Transformer의 어텐션 메커니즘을 활용하여 임의 개수의 소스 뷰들 간의 복잡한 관계를 학습한다. 이를 통해 소스 뷰와 타겟 뷰 간의 큰 차이가 있을 때도 우수한 성능을 보인다[15].

#### 3. 시맨틱 정보 통합

**GSNeRF**[14]는 semantic segmentation을 함께 수행하면서 깊이 맵 예측을 통해 효율적인 샘플링을 구현한다. 이는 geometric reasoning을 통해 일반화 성능을 향상시킨다[14].

### 일반화 성능 향상 전략
#### 1. 메타 학습 접근법
- 다양한 장면에서 학습된 사전 지식 활용
- Few-shot 학습 능력 향상
- 새로운 장면에 대한 빠른 적응

#### 2. 다중 모달리티 활용  
- RGB 이미지와 깊이 정보 결합[14]
- 시맨틱 세그멘테이션 정보 활용
- LiDAR 등 추가 센서 데이터 통합[17]

#### 3. 정규화 및 제약 조건
- 기하학적 일관성 강화[9]
- 멀티뷰 스테레오 제약 활용
- 광도학적 일관성 보장

## 앞으로의 연구에 미치는 영향
### 컴퓨터 비전 분야에 미친 영향
NeRF는 컴퓨터 비전 분야에 다음과 같은 광범위한 영향을 미쳤다[18][19]:

#### 1. 3D 표현 패러다임 전환
- **암시적 신경 표현**의 새로운 표준 확립[18]
- 연속적 장면 표현의 가능성 입증[19]
- 볼륨 렌더링과 딥러닝의 성공적 결합[20]

#### 2. 응용 분야 확산
NeRF 기술은 다음과 같은 다양한 분야로 확산되었다[18][21]:

- **로보틱스**: 환경 인식 및 상호작용[21]
- **의료 영상**: 3D 재건 및 진단[6][22]
- **문화유산 보존**: 문화재 디지털 보존[23][24]
- **자율주행**: 환경 모델링 및 시뮬레이션[25]
- **VR/AR**: 몰입형 경험 제공[26]

### 기술적 파급효과
#### 1. 신경 렌더링 생태계 구축
NeRF는 **신경 렌더링(Neural Rendering)** 분야의 기초를 다져 수많은 후속 연구를 촉발했다[18][27]:

- **속도 개선**: FastNeRF[28], Instant-NGP[29], TensoRF 등
- **품질 향상**: Mip-NeRF[30], NeRF++[31] 등  
- **일반화**: PixelNeRF[2], MVSNeRF[32] 등
- **편집 가능성**: NeRF-Editing, EditNeRF 등

#### 2. 하드웨어 가속 연구
NeRF의 계산 집약적 특성으로 인해 전용 하드웨어 가속기 연구가 활발해졌다[33][34]:

- **ICARUS**: NeRF 전용 가속기 아키텍처[33]
- GPU 최적화 및 병렬화 기법 개발
- 실시간 렌더링을 위한 하드웨어 솔루션

### 최신 연구 동향 (2024-2025)
#### 1. 3D Gaussian Splatting과의 경쟁
2023년 등장한 3D Gaussian Splatting은 NeRF의 강력한 경쟁자로 부상했다[18]. 이는 더 빠른 렌더링 속도와 유사한 품질을 제공하며, NeRF 기반 연구에 새로운 도전을 제시하고 있다.

#### 2. 특수 도메인 적용
최근 연구들은 NeRF를 특수한 도메인에 적용하는 데 초점을 맞추고 있다:

- **위성 이미지**: Sat-NeRF, Planet-NeRF[35]
- **의료 내시경**: Hybrid NeRF-Stereo Vision[36]
- **재해 대응**: 지질 재해 3D 모델링[37]

#### 3. 희소 입력 문제 해결

**AIM 2024 Sparse Neural Rendering Challenge**[38]에서는 매우 적은 입력(3-9개 뷰)으로도 고품질 렌더링을 달성하는 방법들이 제시되었다.

## 앞으로 연구 시 고려할 점
### 1. 효율성과 품질의 균형
미래 연구에서는 다음 사항들을 중점적으로 고려해야 한다[7][39][40]:

#### 계산 효율성 개선
- **샘플링 전략 최적화**: 중요한 영역에 집중한 적응적 샘플링[39]
- **네트워크 아키텍처 경량화**: 작은 MLP로도 고품질 결과 달성[40]
- **하드웨어 최적화**: GPU 메모리 사용량 최소화 및 병렬화[33]

#### 품질 유지 방법
- **고주파수 디테일 보존**: 향상된 위치 인코딩 기법
- **경계면 정의 개선**: 물체 간 경계의 명확한 분리
- **일관성 보장**: 다중 시점 간 기하학적 일관성

### 2. 실세계 적용 가능성 확대
#### 견고성(Robustness) 강화
실제 환경에서의 적용을 위해서는 다음과 같은 견고성이 필요하다[6][41]:

- **노이즈 내성**: 이미지 노이즈 및 아티팩트에 대한 강건성
- **조명 변화 적응**: 다양한 조명 조건에서의 일관된 성능
- **불완전한 데이터 처리**: 일부 가려진 영역이나 결측 데이터 처리

#### 도메인 특화 최적화
- **의료 영상**: X-ray, MRI 등 특수 영상 모달리티 지원[6][22]
- **산업 로봇**: 실시간 환경 인식 및 상호작용[21]
- **자율주행**: 동적 환경에서의 실시간 처리[25]

### 3. 새로운 연구 방향
#### Foundation Models와의 통합
- **대규모 사전 훈련**: 다양한 장면에서 학습된 기반 모델 활용
- **멀티모달 학습**: 텍스트, 이미지, 3D 정보의 통합 학습
- **제로샷 일반화**: 새로운 도메인에 대한 즉시 적용 능력

#### 강화 학습과의 결합
NeRF와 강화 학습의 통합은 다음과 같은 잠재력을 가진다[41]:

- **적응적 뷰 선택**: 최적의 카메라 위치 자동 결정
- **점진적 품질 향상**: 추가 데이터에 따른 지속적 개선
- **개인화된 최적화**: 사용자 선호도에 따른 맞춤형 렌더링

### 4. 윤리적 및 사회적 고려사항
#### 데이터 프라이버시
- 개인 정보가 포함된 3D 장면 처리 시 프라이버시 보호
- 의료 데이터 사용 시 환자 정보 보안

#### 계산 자원 소비
- 환경 친화적인 훈련 방법 개발
- 에너지 효율적인 알고리즘 설계

## 결론
NeRF는 2020년 발표 이후 3D 컴퓨터 비전 분야에 혁신적인 변화를 가져온 획기적인 연구이다[1]. 5D 연속 장면 표현, 위치 인코딩, 계층적 샘플링 등의 핵심 기술을 통해 기존 방법들을 크게 앞서는 성능을 달성했다[1].

비록 긴 훈련 시간, 느린 렌더링 속도, 제한된 일반화 능력 등의 한계점들이 있지만[6][7], 이러한 문제들을 해결하기 위한 후속 연구들이 활발히 진행되고 있다[18]. 특히 일반화 성능 향상을 위한 다양한 접근법들이 제시되고 있으며, 실시간 렌더링과 희소 입력 처리 등의 실용적 문제들도 점차 해결되고 있다[11][15][38].

NeRF의 영향은 단순히 새로운 시점 합성을 넘어서, 로보틱스, 의료 영상, 문화유산 보존, 자율주행 등 다양한 분야로 확산되고 있다[18][21]. 앞으로의 연구에서는 효율성과 품질의 균형, 실세계 적용 가능성 확대, 새로운 연구 방향 탐색 등을 중점적으로 고려해야 할 것이다.

NeRF는 암시적 신경 표현의 새로운 패러다임을 제시했으며, 이는 향후 3D 인공지능 발전의 중요한 기초가 될 것으로 전망된다[18][42]. 지속적인 기술 발전과 함께 더욱 실용적이고 강건한 3D 장면 이해 시스템의 구현이 기대된다.

https://www.liuxiao.org/ko/2021/11/%EB%85%BC%EB%AC%B8-%EB%85%B8%ED%8A%B8-nerf-%EC%8B%A0%EA%B2%BD-%EB%B0%A9%EC%82%AC-%ED%95%84%EB%93%9C%EB%A1%9C-%EC%9E%A5%EB%A9%B4%EC%9D%84-%ED%91%9C%ED%98%84%ED%95%98%EC%97%AC-%EB%B7%B0-%ED%95%A9/

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/22370781/d42e6539-f9fa-437b-a747-fe1ce75738ec/2003.08934v2.pdf
[2] https://linkinghub.elsevier.com/retrieve/pii/S026288562400177X
[3] https://dl.acm.org/doi/10.1145/3664647.3681482
[4] https://www.semanticscholar.org/paper/6caf3307096a15832ace34a0d54cd28413503f8b
[5] https://ieeexplore.ieee.org/document/9879876/
[6] https://arxiv.org/abs/2303.00749
[7] https://ieeexplore.ieee.org/document/9879664/
[8] https://link.springer.com/10.1007/s11390-024-4157-6
[9] https://onlinelibrary.wiley.com/doi/10.1111/cgf.14940
[10] https://www.matthewtancik.com/nerf
[11] https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Nerflets_Local_Radiance_Fields_for_Efficient_Structure-Aware_3D_Scene_Representation_CVPR_2023_paper.pdf
[12] https://keras.io/examples/vision/nerf/
[13] https://arxiv.org/abs/2003.08934
[14] https://arxiv.org/abs/2308.02751
[15] https://yconquesty.github.io/blog/ml/nerf/nerf_rendering.html
[16] https://velog.io/@minkyu4506/NeRF-Representing-Scenes-asNeural-Radiance-Fields-for-View-Synthesis-%EB%A6%AC%EB%B7%B0
[17] https://www.koreascience.kr/article/JAKO202403543202919.page
[18] https://hsejun07.tistory.com/78
[19] https://openreview.net/forum?id=gJHAT79cZU
[20] https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003163265
[21] https://arxiv.org/abs/2209.02417
[22] https://mole-starseeker.tistory.com/114
[23] https://blog.outta.ai/249
[24] https://www.bucketplace.com/post/2023-08-10-neural-rendering-%EA%B0%9C%EB%B0%9C-%EC%9D%B4%EC%95%BC%EA%B8%B0-1-%EA%B3%B5%EA%B0%84%EC%9D%98-%EA%B8%B0%EB%A1%9D/
[25] https://velog.io/@logger_j_k/NeRF
[26] https://arxiv.org/abs/2406.13251
[27] https://ieeexplore.ieee.org/document/10376997/
[28] https://dl.acm.org/doi/10.1145/3550454.3555505
[29] https://www.semanticscholar.org/paper/159673efb2549ecd5f264a8aacc8c7d391619c9a
[30] https://ojs.aaai.org/index.php/AAAI/article/view/28625
[31] https://linkinghub.elsevier.com/retrieve/pii/S0030401824001858
[32] https://dl.acm.org/doi/10.1145/3503161.3547953
[33] https://ieeexplore.ieee.org/document/10203381/
[34] https://gpuopen.com/download/paper1117_CRC.pdf
[35] https://arxiv.org/abs/2311.07044
[36] https://openreview.net/forum?id=YxLxrWkwsX
[37] https://dohyeon.tistory.com/92
[38] https://openaccess.thecvf.com/content/CVPR2024/papers/Chou_GSNeRF_Generalizable_Semantic_Neural_Radiance_Fields_with_Enhanced_3D_Scene_CVPR_2024_paper.pdf
[39] https://arxiv.org/pdf/2305.00375.pdf
[40] https://dhk1349.tistory.com/10
[41] https://arxiv.org/abs/2206.05375
[42] https://nuggy875.tistory.com/168
[43] https://187cm.tistory.com/51
[44] https://openreview.net/forum?id=q73FfT7yfp
[45] https://velog.io/@onground/%EB%85%BC%EB%AC%B8%EC%8A%A4%ED%84%B0%EB%94%94-NeRF-Representing-Scenes-as-Neural-Radiance-Fields-for-View-Synthesis
[46] https://jaeyeol816.github.io/neural_representation/nerf-nerf-basic-theory/
[47] https://arxiv.org/abs/2304.11842
[48] https://canvas4sh.tistory.com/313
[49] https://blog-ko.superb-ai.com/nerf-view-synthesis-for-representing-scenes/
[50] https://arxiv.org/abs/2402.17797
[51] https://www.semanticscholar.org/paper/9a0edc69ad2540c3f6a0c079b4974abac9663f43
[52] https://www.cambridge.org/core/product/identifier/S095442242400012X/type/journal_article
[53] https://link.springer.com/10.1007/s11356-024-34535-9
[54] https://link.springer.com/10.1007/s41666-025-00200-0
[55] https://ieeexplore.ieee.org/document/10625026/
[56] https://iptek.its.ac.id/index.php/ijmeir/article/view/21475
[57] https://ijsrem.com/download/integrating-machine-and-deep-learning-for-enhanced-security-in-cyber-physical-systems-challenges-and-future-research-agenda/
[58] https://arxiv.org/html/2402.17797v2
[59] https://openaccess.thecvf.com/content/CVPR2022/papers/Hu_EfficientNeRF__Efficient_Neural_Radiance_Fields_CVPR_2022_paper.pdf
[60] https://arxiv.org/html/2506.18208v1
[61] https://vds.sogang.ac.kr/wp-content/uploads/2023/01/2023_%EA%B2%A8%EC%9A%B8%EC%84%B8%EB%AF%B8%EB%82%98_%EC%86%90%ED%98%B8%EC%84%B1.pdf
[62] https://arxiv.org/html/2312.02255v3
[63] https://www.sciencedirect.com/science/article/abs/pii/S0736584524000978
[64] https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05300.pdf
[65] https://openaccess.thecvf.com/content/ICCV2023/papers/Wu_Efficient_View_Synthesis_with_Neural_Radiance_Distribution_Field_ICCV_2023_paper.pdf
[66] https://ojs.aaai.org/index.php/AAAI-SS/article/download/27473/27246/31524
[67] https://www.sciencedirect.com/science/article/pii/S2643651524002279
[68] https://glanceyes.com/entry/NeRF-2D-%EC%9D%B4%EB%AF%B8%EC%A7%80%EB%A5%BC-3D-%EC%9D%B4%EB%AF%B8%EC%A7%80%EB%A1%9C-Reconstruction%ED%95%98%EC%97%AC-Novel-View-Synthesis%EC%9D%B4-%EA%B0%80%EB%8A%A5%ED%95%9C-Neural-Radiance-Fields
[69] https://www.journalijar.com/article/52714/neural-radiance-fields-in-space-applications:-a-comprehensive-review/
[70] https://arxiv.org/html/2409.08056v1
[71] https://arxiv.org/html/2210.00379v6
[72] https://www.nature.com/articles/s41598-025-88614-z
[73] https://openaccess.thecvf.com/content/ICCV2023/papers/Rojas_Re-ReND_Real-Time_Rendering_of_NeRFs_across_Devices_ICCV_2023_paper.pdf
[74] https://arxiv.org/abs/2411.02972
[75] https://ieeexplore.ieee.org/document/9848697/
[76] https://ieeexplore.ieee.org/document/10350393/
[77] https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13071/3025454/Neural-implicit-surfaces-learning-for-multi-view-reconstruction/10.1117/12.3025454.full
[78] https://www.mdpi.com/1424-8220/24/7/2314
[79] https://arxiv.org/abs/2410.04041
[80] https://www.mdpi.com/1424-8220/24/23/7594
[81] https://www.mdpi.com/2072-4292/16/2/301
[82] https://papers.neurips.cc/paper_files/paper/2023/file/d705dd6e77decdc399162d6d5b92f6e8-Paper-Conference.pdf
[83] https://neural-rendering.com
[84] https://pub.towardsai.net/10-nerf-papers-you-should-follow-up-part-1-9566707b8f30
[85] https://ieiespc.org/ieiespc/XmlViewer/f436727
[86] https://www.mdpi.com/2220-9964/14/6/218
[87] https://arxiv.org/html/2409.15045v1
[88] https://www.mdpi.com/2072-4292/15/14/3585
[89] https://arxiv.org/html/2405.01333v1
[90] https://arxiv.org/abs/2402.00028
[91] https://pyimagesearch.com/2024/10/28/nerfs-explained-goodbye-photogrammetry/
[92] https://arxiv.org/html/2501.13104v1
[93] https://www.themoonlight.io/en/review/aim-2024-sparse-neural-rendering-challenge-methods-and-results
[94] https://www.tooli.qa/insights/neural-radiance-fields-nerf-a-breakthrough-in-3d-reconstruction
[95] https://modulabs.co.kr/blog/nerf-followup
[96] https://codalab.lisn.upsaclay.fr/competitions/19223
[97] https://drpress.org/ojs/index.php/jceim/article/view/28768
[98] https://linkinghub.elsevier.com/retrieve/pii/S095741742402935X
[99] https://ieeexplore.ieee.org/document/9878829/
[100] https://arxiv.org/pdf/2208.04717.pdf
[101] https://arxiv.org/html/2402.01217v3
[102] https://dl.acm.org/doi/pdf/10.1145/3588432.3591483
[103] https://arxiv.org/pdf/2003.08934.pdf
[104] https://arxiv.org/html/2312.05855v1
[105] https://arxiv.org/pdf/2301.00411.pdf
[106] https://arxiv.org/html/2401.11711v1
[107] https://arxiv.org/html/2403.01325v1
[108] http://arxiv.org/pdf/2010.07492.pdf
[109] https://arxiv.org/pdf/2304.05218.pdf
[110] https://velog.io/@gjaegal/NeRF
[111] https://github.com/KerasKorea/KEKOxTutorial/blob/master/204_3D%20volumetric%20rendering%20with%20NeRF.md
[112] https://gaussian37.github.io/vision-fusion-nerf/
[113] https://arxiv.org/abs/2401.01391
[114] https://arxiv.org/abs/2309.15101
[115] https://arxiv.org/html/2406.11840v2
[116] https://arxiv.org/pdf/2304.10075.pdf
[117] https://arxiv.org/html/2401.01391v1
[118] https://arxiv.org/pdf/2211.12285.pdf
[119] https://www.frontiersin.org/articles/10.3389/fnbot.2025.1558948/full
[120] https://arxiv.org/pdf/2103.13744.pdf
[121] https://arxiv.org/pdf/2203.17261.pdf
[122] https://arxiv.org/html/2404.04913v2
[123] http://arxiv.org/pdf/2406.07828.pdf
[124] https://arxiv.org/html/2410.17839
[125] https://bcommons.berkeley.edu/generalizing-neural-radiance-fields
[126] https://csm-kr.tistory.com/64
[127] https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_ContraNeRF_Generalizable_Neural_Radiance_Fields_for_Synthetic-to-Real_Novel_View_Synthesis_CVPR_2023_paper.pdf
[128] https://link.springer.com/10.1007/s10479-023-05251-3
[129] http://ijcrims.com/pdfcopy/2023/oct2023/ijcrims5.pdf
[130] https://arxiv.org/pdf/2402.17797.pdf
[131] https://arxiv.org/html/2501.13104v1?trk=public_post_comment-text
[132] http://arxiv.org/pdf/2405.05526.pdf
[133] http://arxiv.org/pdf/2405.18715.pdf
[134] https://pmc.ncbi.nlm.nih.gov/articles/PMC10974786/
[135] http://arxiv.org/pdf/2405.01333.pdf
[136] https://pmc.ncbi.nlm.nih.gov/articles/PMC11436004/
[137] https://arxiv.org/pdf/2308.11130.pdf
[138] https://arxiv.org/html/2401.03257
[139] https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/3dim/
[140] https://arxiv.org/html/2402.10344v3
[141] https://linkinghub.elsevier.com/retrieve/pii/S0926580523000705
[142] https://isprs-archives.copernicus.org/articles/XLVIII-M-2-2023/1113/2023/
[143] https://isprs-archives.copernicus.org/articles/XLVIII-4-W10-2024/199/2024/isprs-archives-XLVIII-4-W10-2024-199-2024.pdf
[144] http://arxiv.org/pdf/2404.00714.pdf
[145] https://arxiv.org/pdf/2210.00379v4.pdf
[146] https://arxiv.org/pdf/2209.13433.pdf
[147] http://arxiv.org/pdf/2301.04075.pdf
[148] https://arxiv.org/pdf/2112.05504.pdf
[149] https://arxiv.org/pdf/2308.07868.pdf
[150] https://arxiv.org/pdf/2304.10050.pdf
[151] https://arxiv.org/pdf/2304.11342.pdf
[152] https://arxiv.org/pdf/2401.12451.pdf
[153] https://www.reddit.com/r/computervision/comments/1fpynz8/how_do_i_learn_about_nerf_in_order_to_do_research/
[154] https://dl.acm.org/doi/10.1145/3728725.3728737
