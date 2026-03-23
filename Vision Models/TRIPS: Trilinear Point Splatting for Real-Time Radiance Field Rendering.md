# TRIPS: Trilinear Point Splatting for Real-Time Radiance Field Rendering

---

## 1. 핵심 주장 및 주요 기여 (Summary)

TRIPS(Trilinear Point Splatting)는 Gaussian Splatting과 ADOP의 아이디어를 결합한 접근법으로, 핵심 기법은 포인트를 스크린-스페이스 이미지 피라미드에 래스터화하며, 피라미드 레이어 선택은 투영된 포인트 크기에 의해 결정된다.

**주요 기여 3가지:**

1. TRIPS라는 새로운 삼선형(trilinear) 포인트 스플래팅 기법의 도입
2. 포인트 위치와 크기를 포함한 모든 입력 파라미터를 최적화할 수 있는 미분 가능한(differentiable) 파이프라인 설계
3. 다양한 촬영 조건에서 고품질 실시간 렌더링을 달성하는 구현체 공개

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 포인트 기반 radiance field 렌더링에는 두 가지 대표적 한계가 존재했다:

- **3D Gaussian Splatting (3DGS)의 문제:** 3D Gaussian Splatting은 고도로 디테일한 장면을 렌더링할 때 블러링(blurring)과 구름 같은 아티팩트(cloudy artifacts)로 인해 어려움을 겪는다.
- **ADOP의 문제:** ADOP는 더 선명한 이미지를 생성할 수 있지만, 신경망 재구성 네트워크로 인해 성능이 저하되며, 시간적 불안정성(temporal instability)과 포인트 클라우드의 큰 공백을 효과적으로 채우지 못하는 문제가 있다.

### 2.2 제안하는 방법 (수식 포함)

#### (a) Trilinear Point Splatting 핵심 메커니즘

3D Gaussian Splatting과 유사하게 TRIPS는 다양한 크기의 스플랫을 래스터화하지만, ADOP처럼 재구성 네트워크를 적용하여 빈 공간 없이 선명한 이미지를 생성한다. 구체적으로, 포인트 클라우드를 $2 \times 2 \times 2$ 삼선형 스플랫으로 이미지 피라미드에 래스터화하고 front-to-back 알파 블렌딩으로 혼합한다.

각 포인트 $p_i$는 월드 좌표 $\mathbf{x}_i \in \mathbb{R}^3$, 크기 $s_i \in \mathbb{R}$, 신경 색상 디스크립터 $\mathbf{d}_i \in \mathbb{R}^F$, 투명도 $\alpha_i$를 가진다. 투영된 포인트 크기에 따라 이미지 피라미드의 레이어 $l$이 결정된다:

$$l = \log_2\left(\frac{s_{\text{proj}}}{s_{\text{pixel}}}\right)$$

여기서 $s_{\text{proj}}$는 스크린 공간에 투영된 포인트의 크기, $s_{\text{pixel}}$은 기준 픽셀 크기이다.

다른 접근법과 달리 점진적으로 작은 해상도의 다중 렌더 패스를 사용하지 않는데, 이는 저해상도 레이어에서 심각한 오버드로를 유발하기 때문이다. 대신, 포인트의 투영 크기에 가장 적합한 두 레이어만 계산하고 $2 \times 2$ 스플랫으로만 렌더링한다.

#### (b) 삼선형 보간 (Trilinear Interpolation)

$2 \times 2 \times 2$ 삼선형 쓰기 연산에서 각 포인트의 기여는 세 축(screen-space $x$, screen-space $y$, 피라미드 level $l$)에 대해 선형 보간된다:

$$w(x, y, l) = w_x \cdot w_y \cdot w_l$$

여기서 $w_x, w_y$는 공간적 바이리니어 가중치, $w_l$은 레이어 간 선형 가중치이다.

#### (c) 알파 블렌딩

이미지 피라미드의 각 픽셀에는 깊이 정렬된 색상 및 알파 값의 목록이 저장된다. Front-to-back 알파 블렌딩은 다음과 같다:

$$C_{\text{pixel}} = \sum_{k=1}^{K} \mathbf{c}_k \cdot \alpha_k \cdot \prod_{j=1}^{k-1}(1 - \alpha_j)$$

여기서 $\mathbf{c}_k$는 $k$번째 포인트의 색상, $\alpha_k$는 투명도이다.

#### (d) 신경망 재구성 네트워크

레이어들은 이후 소형 신경 재구성 네트워크에서 최종 이미지로 병합되며, 이는 U-Net의 디코더 부분과 유사하다.

이 네트워크는 효율성을 위해 특별히 간소화되어 있으며, 게이트 컨볼루션(gated convolution)과 셀프 바이패스 연결(self-bypass connection)을 특징으로 한다. 이 아키텍처 덕분에 최소한의 연산 비용으로 복잡한 작업을 수행할 수 있다.

각 레이어별 게이트 컨볼루션은 다음과 같이 정의된다:

$$\mathbf{F}_{\text{out}} = \sigma(\mathbf{W}_g * \mathbf{F}_{\text{in}}) \odot (\mathbf{W}_f * \mathbf{F}_{\text{in}}) + \mathbf{F}_{\text{bypass}}$$

여기서 $\sigma$는 시그모이드 함수, $\mathbf{W}_g$와 $\mathbf{W}_f$는 게이트 및 피처 컨볼루션 커널, $\odot$는 요소별 곱셈이다.

#### (e) 구면 조화 함수 및 톤 매핑

특히 도전적인 입력 시나리오에서 높은 디테일 보존을 보장하기 위해, 구면 조화 함수(spherical harmonics)와 톤 매핑 모듈을 파이프라인에 통합한다.

$$\mathbf{c}(\mathbf{v}) = \sum_{l=0}^{L} \sum_{m=-l}^{l} c_{lm} \cdot Y_l^m(\mathbf{v})$$

여기서 $Y_l^m$은 구면 조화 기저 함수, $\mathbf{v}$는 뷰 방향이다.

#### (f) 손실 함수

전체 파이프라인은 미분 가능하며, 일반적으로 다음과 같은 결합 손실 함수로 최적화된다:

$$\mathcal{L} = \lambda_1 \cdot \mathcal{L}_{\text{L1}} + \lambda_2 \cdot \mathcal{L}_{\text{LPIPS}} + \lambda_3 \cdot \mathcal{L}_{\text{SSIM}}$$

여기서:
- $\mathcal{L}\_{\text{L1}} = \| I_{\text{pred}} - I_{\text{gt}} \|_1$ : 픽셀 단위 L1 손실
- $\mathcal{L}_{\text{LPIPS}}$ : 지각적(perceptual) 손실
- $\mathcal{L}\_{\text{SSIM}} = 1 - \text{SSIM}(I_{\text{pred}}, I_{\text{gt}})$ : 구조적 유사도 손실

### 2.3 모델 구조 (Pipeline Overview)

TRIPS는 포인트 클라우드를 $2 \times 2 \times 2$ 스플랫으로 삼선형 렌더링 및 블렌딩하여 다층 피처 맵에 기록하고, 결과를 레이어당 단일 게이트 컨볼루션만 포함하는 소형 신경망에 통과시킨다.

이후 선택적 구면 조화 모듈과 톤 매퍼가 최종 이미지를 생성한다. 이 파이프라인은 완전히 미분 가능하여 포인트 디스크립터(색상), 위치, 카메라 파라미터가 경사 하강법으로 최적화된다.

```
입력 포인트 클라우드 + 카메라 파라미터
        ↓
[Trilinear Point Splatting] → 이미지 피라미드 (다층 피처 맵)
        ↓
[Front-to-Back Alpha Blending]
        ↓
[경량 신경 재구성 네트워크 (Gated Conv per layer, U-Net Decoder 유사)]
        ↓
[Spherical Harmonics Module]
        ↓
[Tone Mapper]
        ↓
최종 RGB 이미지
```

### 2.4 성능 향상

TRIPS는 기존 최첨단 방법을 렌더링 품질 면에서 능가하면서도 일반적으로 사용 가능한 하드웨어에서 초당 60프레임의 실시간 프레임 레이트를 유지한다.

정량적 결과:

| 메트릭 | TRIPS (Mip-NeRF360) | 비고 |
|--------|---------------------|------|
| **LPIPS $^{\text{VGG}}$ ** | 평균 **0.176** | 최고 수준 |
| **PSNR** | 평균 **25.94 dB** | 경쟁력 있는 수준 |
| **SSIM** | 평균 **0.778** | 경쟁력 있는 수준 |

Tanks&Temples 데이터셋에서 TRIPS는 평균적으로 가장 우수한 LPIPS 점수를 달성하며 2위 대비 20% 개선되었다. PSNR과 SSIM에서는 최첨단 수준과 동등하다.

구체적 효율성:
- 학습은 Nvidia A100에서 장면당 약 2-4시간이 소요되며, RTX 4090에서 약 11ms의 새로운 뷰 렌더링 속도를 달성한다.
- TRIPS는 대규모 포인트 클라우드 처리에서 뛰어난 효율성을 보여, 12M 포인트를 11.1ms에 처리하는 반면 dense Gaussian Splatting은 8M 포인트를 11.4ms에 처리한다.

### 2.5 한계점

1. **깊이 정보 손실:** 삼선형 포인트 스플래팅은 포인트를 별도의 레이어로 분리하므로 깊이 정보를 잃게 된다. 이론적으로 재결합 시 고체 기하학에 구멍이 생길 수 있다. 실제로는 학습 데이터를 크게 벗어나는 극단적 줌인의 경우를 제외하고는 이러한 현상이 발견되지 않았다.

2. **학습 시간:** TRIPS는 3DGS나 Instant-NGP보다 학습에 다소 더 많은 시간이 소요된다.

3. **신경망 의존성:** ADOP보다 가볍지만, 경량 신경 재구성 네트워크에 여전히 의존하므로 완전한 explicit representation에 비해 해석 가능성이 제한된다.

4. **MipNeRF-360 대비 PSNR/SSIM:** MipNeRF-360 데이터셋에서 다시 최고의 LPIPS 점수를 얻지만, 볼류메트릭 방법과 Gaussian Splatting은 PSNR과 SSIM에서 더 우수하다. TRIPS 렌더링은 더 나은 선명도와 디테일을 제공하지만 MipNeRF-360과 Gaussian 출력은 전반적으로 더 깨끗하고 노이즈가 적다.

---

## 3. 모델의 일반화 성능 향상 가능성

TRIPS는 다음과 같은 측면에서 일반화 성능 향상의 잠재력을 보인다:

### 3.1 다양한 시나리오에 대한 로버스트성

이 성능은 복잡한 기하학을 가진 장면, 광활한 풍경, 자동 노출 영상 등 도전적인 시나리오에까지 확장된다.

### 3.2 적응적 포인트 크기 최적화

삼선형 스플래팅 기법으로 포인트 크기를 최적화하여 장면의 큰 빈 공간을 채울 수 있다. 이는 불완전한 포인트 클라우드(LiDAR 등으로 취득한 경우)에서도 효과적인 일반화를 가능하게 한다:

$$s_i^{*} = \arg\min_{s_i} \mathcal{L}(I_{\text{pred}}(s_i), I_{\text{gt}})$$

### 3.3 완전 미분 가능한 파이프라인

TRIPS는 포인트를 스크린 스페이스 이미지 피라미드로 래스터화하는 효율적 전략을 채택하여 대형 포인트의 효율적 렌더링을 가능하게 하며, 완전히 미분 가능하여 포인트 크기와 위치의 자동 최적화를 허용한다. 이를 통해 새로운 장면에 대한 fine-tuning이 용이하다.

### 3.4 시간적 안정성(Temporal Stability) 향상

TRIPS는 round-to-next-pixel 래스터화 대신 선형 보간을 사용하므로 ADOP보다 더 나은 시간적 안정성을 제공한다. 이는 동영상 기반 응용에서의 일반화에 중요하다.

### 3.5 향후 일반화 향상을 위한 방향

- **크로스 데이터셋 전이 학습**: 신경 재구성 네트워크를 여러 장면에서 사전 학습하여 zero-shot/few-shot 일반화 성능을 높일 수 있음
- **동적 장면 확장**: 시간 축을 추가하여 $2 \times 2 \times 2 \times T$ 스플래팅으로 확장 가능
- **포인트 밀도 적응 전략**: 장면 복잡도에 따라 포인트 밀도를 자동 조절하는 메커니즘 추가

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구 영향

1. **하이브리드 접근의 정당성 확립**: TRIPS는 explicit (포인트 기반)과 implicit (신경망 기반) 표현의 결합이 두 방법의 장점을 모두 취할 수 있음을 입증했다.

2. **이미지 피라미드 기반 멀티스케일 렌더링**: 스크린 스페이스 이미지 피라미드를 통한 멀티스케일 처리가 radiance field 렌더링에서 효과적인 패러다임임을 보여주었다.

3. TRIPS는 60fps의 일관된 성능과 시간적 일관성을 제공하여 게임, VR, 인터랙티브 미디어에 이상적이다.

4. **미분 가능 렌더링 파이프라인 설계의 방향성 제시**: 포인트 위치, 크기, 카메라 파라미터까지 모든 요소를 end-to-end로 최적화하는 접근은 향후 연구의 표준이 될 가능성이 있다.

### 4.2 향후 연구 시 고려할 점

| 항목 | 고려사항 |
|------|---------|
| **메모리 효율성** | 대규모 장면(수천만 포인트)에서의 GPU 메모리 최적화 필요 |
| **학습 시간 단축** | 2-4시간 학습 시간을 3DGS 수준(~0.75시간)으로 줄이는 연구 |
| **동적 장면 대응** | 시간적으로 변화하는 장면에 대한 확장 연구 필요 |
| **네트워크 의존성 감소** | 재구성 네트워크를 더 경량화하거나 제거하는 방향 탐색 |
| **극단적 뷰 일반화** | 학습 데이터에서 크게 벗어난 뷰에서의 성능 향상 |
| **다양한 입력 소스** | LiDAR, 스테레오 카메라 등 다양한 포인트 클라우드 소스에 대한 로버스트성 검증 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 연도 | 표현 방식 | 실시간 | 핵심 특징 | TRIPS 대비 |
|------|------|----------|--------|----------|-----------|
| **NeRF** (Mildenhall et al.) | 2020 | Implicit (MLP) | ✗ | NeRF는 명시적 3D 기하학을 구성하는 대신 신경망으로 정의된 기본 볼류메트릭 장면 함수를 학습한다. | TRIPS가 실시간, 더 빠른 렌더링 |
| **Plenoxels** | 2022 | Explicit (Voxel Grid) | △ | 신경망 없이 radiance field 구현 | TRIPS가 더 유연한 포인트 크기 적응 |
| **Instant-NGP** (Müller et al.) | 2022 | Hybrid (Hash Grid + MLP) | △ | 멀티 해상도 해시 그리드를 사용하여 렌더링 속도와 메모리 요구량을 크게 최적화한다. | TRIPS가 디테일 보존에서 우위 |
| **ADOP** (Rückert et al.) | 2022 | Point + Neural | △ | 1픽셀 포인트로 다중 해상도에서 깊이 테스트 후 신경망으로 공백 해소 및 텍스처 향상 | TRIPS가 시간적 안정성, 큰 공백 처리에서 우위 |
| **Mip-NeRF 360** (Barron et al.) | 2022 | Implicit (MLP) | ✗ | 안티앨리어싱된 무한 범위 radiance field | TRIPS가 실시간, LPIPS에서 우위; PSNR/SSIM에서 열위 |
| **3D Gaussian Splatting** (Kerbl et al.) | 2023 | Explicit (3D Gaussians) | ✓ | 3D 가우시안을 사용한 장면 표현, 비등방 공분산 최적화, 빠른 가시성 인식 렌더링 알고리즘 | TRIPS가 LPIPS에서 우위, 미세 디테일 보존에서 우위 |
| **TRIPS** (Franke et al.) | 2024 | Point + Lightweight Neural | ✓ | 삼선형 스플래팅 + 이미지 피라미드 + 경량 재구성 | — |

### 핵심 비교 분석

**NeRF 계열 대비:**
- NeRF에서는 3D 공간의 모든 샘플링 포인트의 체적 밀도를 신경망으로 추정하고, 레이 마칭 알고리즘을 활용하여 RGB 값을 결정한다. 이는 고품질이지만 매우 느린 렌더링을 초래한다.
- TRIPS는 explicit 포인트 기반이므로 빈 공간에서의 불필요한 연산을 피하고, 실시간 렌더링을 달성한다.

**3D Gaussian Splatting 대비:**
- 현 상태에서 3D Gaussian Splatting은 미세 디테일에서 다소 어려움을 겪지만, 또 다른 실시간 radiance field 방법인 TRIPS가 해결책을 제안했다.
- 밀집 포인트 클라우드에서도 Gaussian Splatting(LPIPS 0.283)은 PLAYGROUND 장면에서 TRIPS(LPIPS 0.229)의 전반적 품질에 미치지 못한다. 이는 TRIPS의 삼선형 포인트 스플래팅, 신경 재구성 네트워크, 대규모 포인트 클라우드의 효율적 처리의 조합이 더 많은 포인트를 유지하고 세밀한 정보를 인코딩할 수 있게 하여 더 선명한 결과를 가져온다는 것을 시사한다.

**3DGS vs. NeRF 최신 비교에서의 시사점:**
- 3DGS는 연산 효율성과 노이즈 감소 면에서 NeRF를 일관되게 능가한다.
- TRIPS는 이러한 3DGS의 효율성 장점을 유지하면서도 ADOP의 디테일 재구성 능력을 결합하여 **두 방향의 발전을 동시에** 추구한다.

---

## 참고 출처

1. **Franke, L., Rückert, D., Fink, L., & Stamminger, M.** (2024). "TRIPS: Trilinear Point Splatting for Real-Time Radiance Field Rendering." *Computer Graphics Forum*, 43(2). DOI: 10.1111/cgf.15012
2. **arXiv 프리프린트**: https://arxiv.org/abs/2401.06003
3. **프로젝트 페이지**: https://lfranke.github.io/trips/
4. **GitHub 리포지토리**: https://github.com/lfranke/TRIPS
5. **Magnopus 기술 블로그**: https://www.magnopus.com/blog/trilinear-point-splatting-trips-and-its-advantages-over-gaussian-splatting-and-adop
6. **Radiance Fields 분석**: https://radiancefields.com/trips-trilinear-point-splatting-for-real-time-radiance-field-rendering
7. **Emergent Mind 요약**: https://www.emergentmind.com/papers/2401.06003
8. **Liner Quick Review**: https://liner.com/review/trips-trilinear-point-splatting-for-realtime-radiance-field-rendering
9. **Wiley Online Library (CGF)**: https://onlinelibrary.wiley.com/doi/10.1111/cgf.15012
10. **Semantic Scholar**: https://www.semanticscholar.org/paper/efba025c6cf9c8cb57a7383efba7c4800329260d
11. **ResearchGate PDF**: https://www.researchgate.net/publication/380237030
12. **Kerbl, B. et al.** (2023). "3D Gaussian Splatting for Real-Time Radiance Field Rendering." *ACM Transactions on Graphics*, 42(4).
13. **Mildenhall, B. et al.** (2020). "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis." *ECCV 2020*.
14. **Barron, J.T. et al.** (2022). "Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields." *CVPR 2022*.
15. **Rückert, D. et al.** (2022). "ADOP: Approximate Differentiable One-Pixel Point Rendering." *ACM TOG*, 41(4).
16. **Müller, T. et al.** (2022). "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding." *ACM TOG*, 41(4).

> **참고**: 본 분석에서 논문의 세부 수식(삼선형 보간 가중치, 게이트 컨볼루션 정의 등)은 원 논문 및 공개된 기술 설명에 기반하여 재구성한 것이며, 논문 원문의 표기법과 미세한 차이가 있을 수 있습니다. 정확한 수식 세부사항은 원 논문(CGF 2024, DOI: 10.1111/cgf.15012)을 직접 참조하시기 바랍니다.
