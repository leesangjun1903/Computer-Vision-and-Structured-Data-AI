# 3D Gaussian Splatting for Real-Time Radiance Field Rendering

## 1. 핵심 주장과 주요 기여 요약

이 논문은 실시간 radiance field 렌더링을 위한 혁신적인 접근법을 제시하며, 다음과 같은 핵심 주장과 기여를 담고 있습니다:[1]

**핵심 주장**: 기존 NeRF 방식의 고품질 렌더링과 최신 고속 방법의 효율성을 동시에 달성할 수 있다는 것입니다. 이는 3D Gaussian을 장면 표현의 기본 단위로 사용하고, 타일 기반 래스터화를 통해 가능해졌습니다.[1]

**주요 기여**:

1. **비등방성 3D Gaussian 표현**: 고품질의 비구조화된 radiance field 표현으로 3D Gaussian을 도입했습니다. 이는 미분 가능한 볼륨 표현의 속성을 유지하면서도 빠른 GPU 래스터화가 가능합니다.[1]

2. **적응형 밀도 제어를 통한 최적화**: 3D Gaussian의 위치, 불투명도 $$\alpha$$, 비등방성 공분산, 구면 조화(SH) 계수를 최적화하는 방법을 제안했습니다. 이 과정은 Gaussian의 추가 및 제거를 통한 적응형 밀도 제어와 교차로 진행됩니다.[1]

3. **고속 미분 가능 렌더러**: 가시성을 인식하고 비등방성 스플래팅을 지원하는 GPU 기반 렌더링 방식을 개발했습니다. 이는 빠른 역전파를 가능하게 하여 고품질 novel view synthesis를 달성합니다.[1]

**성능 요약**: 6분의 학습으로 PSNR 23.6을 달성하며 135fps로 렌더링하고, 51분 학습으로 PSNR 25.2를 달성하며 93fps로 렌더링합니다. 이는 Mip-NeRF360(48시간 학습, 0.071 fps)와 동등하거나 더 나은 품질을 보입니다.[1]

## 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

**핵심 문제**: 기존 radiance field 방법들은 고품질과 실시간 렌더링 사이의 트레이드오프가 존재했습니다:[1]

- **NeRF 기반 방법**: Mip-NeRF360 같은 방법은 뛰어난 품질(PSNR 27.69)을 보이지만, 48시간의 학습 시간과 0.06 fps의 느린 렌더링 속도를 가집니다.[1]
- **고속 방법**: InstantNGP와 Plenoxels는 빠른 학습(5-7분)과 개선된 렌더링 속도(9-17 fps)를 제공하지만, 품질이 떨어지고(PSNR 21-25) 1080p 해상도에서 실시간 렌더링이 불가능합니다.[1]

**기술적 제약사항**:
- 볼륨 ray-marching은 많은 샘플링이 필요하여 계산 비용이 높고 노이즈를 발생시킵니다.[1]
- 구조화된 그리드 기반 가속 방법은 빈 공간 표현에 어려움을 겪습니다.[1]
- 기존 포인트 기반 방법은 MVS 데이터가 필요하여 over/under-reconstruction 문제를 상속받습니다.[1]

### 2.2 제안하는 방법과 수식

#### 3D Gaussian 표현

각 3D Gaussian은 평균 $$\mu$$와 공분산 행렬 $$\Sigma$$로 정의됩니다:[1]

$$
G(x) = e^{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}
$$

**공분산 행렬의 최적화 가능한 표현**: 직접 $$\Sigma$$를 최적화하면 양의 준정부호 제약이 위반될 수 있으므로, 스케일 행렬 $$S$$와 회전 행렬 $$R$$로 분해합니다:[1]

$$
\Sigma = RSS^TR^T
$$

여기서:
- $$s$$는 3D 스케일 벡터
- $$q$$는 회전을 나타내는 단위 쿼터니언

이 표현은 gradient descent에 적합하며 유효한 공분산 행렬을 보장합니다.[1]

#### 2D 투영

3D Gaussian을 화면 공간으로 투영하기 위해, viewing transformation $$W$$와 affine approximation의 Jacobian $$J$$를 사용합니다:[1]

$$
\Sigma' = JW\Sigma W^TJ^T
$$

여기서 $$\Sigma'$$는 카메라 좌표계에서의 공분산 행렬입니다. $$\Sigma'$$의 3번째 행과 열을 제거하면 2D 분산 행렬을 얻습니다.[1]

#### 렌더링 수식

픽셀 색상 $$C$$는 정렬된 $$N$$개의 Gaussian을 블렌딩하여 계산됩니다:[1]

$$
C = \sum_{i \in N} c_i\alpha_i \prod_{j=1}^{i-1}(1-\alpha_j)
$$

여기서:
- $$c_i$$는 각 포인트의 색상 (구면 조화 계수로 표현)
- $$\alpha_i$$는 2D Gaussian 평가와 학습된 불투명도의 곱

**손실 함수**: L1과 D-SSIM의 조합을 사용합니다:[1]

$$
\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{D-SSIM}
$$

여기서 $$\lambda = 0.2$$를 모든 실험에서 사용합니다.[1]

#### 적응형 밀도 제어

**Densification 전략**: view-space positional gradient가 임계값 $$\tau_{pos} = 0.0002$$를 초과하는 Gaussian을 대상으로 합니다:[1]

1. **Clone (복제)**: 작은 Gaussian은 복제하여 위치 gradient 방향으로 이동시킵니다. 이는 under-reconstruction 영역을 처리합니다.[1]

2. **Split (분할)**: 큰 Gaussian($$\|S\| > \tau_S$$)은 두 개로 분할하고, 스케일을 $$\phi = 1.6$$으로 나눕니다. 원래 Gaussian을 PDF로 사용하여 새 위치를 샘플링합니다[1].

**Pruning (가지치기)**: 100 iteration마다 densification을 수행하고, 불투명도가 $$\epsilon_\alpha$$ 미만인 Gaussian을 제거합니다. 또한 3000 iteration마다 $$\alpha$$를 0에 가깝게 설정하여 필요한 Gaussian만 유지합니다.[1]

### 2.3 모델 구조 및 파이프라인

#### 초기화

Structure-from-Motion (SfM)으로 생성된 sparse point cloud로 시작합니다. 각 포인트에서 3D Gaussian을 생성하며, 초기 공분산 행렬은 가장 가까운 3개 포인트까지의 평균 거리를 축으로 하는 등방성 Gaussian으로 설정합니다.[1]

#### 타일 기반 래스터화

**핵심 설계**:[1]

1. **화면 분할**: 화면을 16×16 픽셀 타일로 분할합니다.

2. **Frustum culling**: 99% 신뢰 구간이 view frustum과 교차하는 Gaussian만 유지합니다.

3. **키 할당**: 각 Gaussian 인스턴스에 view space depth와 tile ID를 결합한 키를 할당합니다.

4. **GPU Radix sort**: 단일 고속 GPU Radix sort로 모든 Gaussian을 정렬합니다.

5. **타일당 렌더링**: 각 타일에 대해 thread block을 실행하여 front-to-back으로 색상과 $$\alpha$$ 값을 누적합니다. 픽셀이 포화($$\alpha \rightarrow 1$$)에 도달하면 종료합니다.

**역전파**: forward pass에서 사용된 정렬된 배열과 타일 범위를 재사용하여 back-to-front로 순회합니다. 최종 누적 불투명도만 저장하고, 각 포인트의 $$\alpha$$로 나누어 중간 불투명도를 복원합니다.[1]

**차별점**: 이전 방법(Pulsar)과 달리, gradient를 받는 Gaussian 수에 제한이 없으며, 픽셀당 일정한 메모리 오버헤드만 필요합니다.[1]

#### 최적화 세부사항

- **Warm-up**: 4배 작은 해상도로 시작하여 250, 500 iteration 후 업샘플링합니다.[1]
- **구면 조화 최적화**: 0차 성분부터 시작하여 1000 iteration마다 한 밴드씩 추가하여 4개 밴드까지 확장합니다.[1]
- **활성화 함수**: $$\alpha$$는 sigmoid, 공분산 스케일은 exponential 활성화를 사용합니다.[1]
- **학습률 스케줄링**: 위치에 대해서만 exponential decay를 적용합니다.[1]

### 2.4 성능 향상

#### 정량적 결과

**Mip-NeRF360 데이터셋** (평균 30K iterations):[1]
- **SSIM**: 0.815 (Mip-NeRF360: 0.792, InstantNGP-Big: 0.699)
- **PSNR**: 27.21 dB (Mip-NeRF360: 27.69 dB, InstantNGP-Big: 25.59 dB)
- **LPIPS**: 0.214 (Mip-NeRF360: 0.237, InstantNGP-Big: 0.331)
- **학습 시간**: 41분 33초 (Mip-NeRF360: 48시간)
- **FPS**: 134 (Mip-NeRF360: 0.06)

**Tanks&Temples 데이터셋** (30K iterations):[1]
- **SSIM**: 0.841, **PSNR**: 23.14 dB, **LPIPS**: 0.183
- **학습 시간**: 26분 54초, **FPS**: 154

**Deep Blending 데이터셋** (30K iterations):[1]
- **SSIM**: 0.903, **PSNR**: 29.41 dB, **LPIPS**: 0.243
- **학습 시간**: 36분 2초, **FPS**: 137

#### Ablation Study 결과

**Anisotropic covariance의 중요성**: isotropic Gaussian을 사용하면 평균 PSNR이 26.05에서 25.23으로 감소합니다. 비등방성은 표면 정렬을 크게 개선하여 동일한 포인트 수로 더 높은 품질을 달성합니다.[1]

**Densification 전략**: split 없이는 평균 PSNR이 23.90으로 감소하고(특히 배경 재구성 저하), clone 없이는 25.91로 감소합니다(얇은 구조 처리 어려움).[1]

**Gradient 제한 제거**: 10개 Gaussian으로 gradient를 제한하면 Truck 장면에서 PSNR이 11dB 감소합니다(22.71 → 14.66 at 5K iterations).[1]

**구면 조화**: SH 없이는 평균 PSNR이 25.35로 감소하여, view-dependent 효과 보상의 중요성을 보여줍니다.[1]

### 2.5 한계점

논문에서 명시한 주요 한계:[1]

1. **관찰되지 않은 영역의 아티팩트**: 장면이 잘 관찰되지 않은 영역에서 아티팩트가 발생합니다. Mip-NeRF360도 유사한 문제를 겪습니다.

2. **Elongated/splotchy Gaussians**: 비등방성 Gaussian의 장점에도 불구하고, 때때로 길쭉하거나 얼룩진 Gaussian이 생성됩니다.

3. **Popping artifacts**: 큰 Gaussian 생성 시 popping 아티팩트가 발생할 수 있습니다. 이는:
   - 래스터화기의 trivial guard band rejection
   - 단순한 가시성 알고리즘으로 인한 갑작스러운 depth/blending order 변경 때문입니다.

4. **정규화 부재**: 현재 최적화에 정규화를 적용하지 않습니다. 정규화는 위의 문제들을 완화할 수 있습니다.

5. **대규모 장면의 하이퍼파라미터**: 동일한 하이퍼파라미터를 모든 평가에 사용했지만, 초기 실험에서 매우 큰 장면(예: 도시 데이터셋)에서는 위치 학습률 감소가 필요할 수 있습니다.

6. **높은 메모리 소비**: 이전 포인트 기반 방법보다는 compact하지만, NeRF 기반 솔루션보다 메모리 소비가 현저히 높습니다:
   - **학습 시**: 대규모 장면에서 20GB 이상의 GPU 메모리 필요
   - **렌더링 시**: 모델 저장에 수백 MB + 래스터화에 30-500 MB 필요
   - 그러나 point cloud 압축 기술을 적용하여 개선 가능성이 있습니다.[1]

## 3. 모델의 일반화 성능 향상

### 3.1 다양한 장면 타입에서의 일반화

논문은 다음과 같은 다양한 장면 타입에서 일관된 성능을 보여줍니다:[1]

**실내 bounded 장면**: Room, Counter, Kitchen 장면에서 PSNR 28.7-30.6 달성[1]

**실외 unbounded 장면**: Bicycle, Garden, Stump 장면에서 PSNR 21.5-27.4 달성[1]

**합성 장면**: 100K 랜덤 초기화로 시작하여 NeRF-synthetic 데이터셋에서 평균 PSNR 33.32 달성 (Point-NeRF: 33.30, Mip-NeRF: 33.09)[1]

### 3.2 초기화 유연성

**SfM 포인트 없이도 작동**: 랜덤 초기화 ablation에서, 방법이 완전 실패를 피하고 합리적인 성능을 유지합니다. SfM 초기화 대비 품질 저하가 있지만(평균 PSNR: 20.42 vs 26.05 at 30K), 주로 배경에서 발생합니다.[1]

**합성 장면에서의 강건성**: NeRF-synthetic 데이터셋에서 exact camera parameters와 exhaustive view set으로 인해 랜덤 초기화로도 state-of-the-art 결과 달성합니다.[1]

### 3.3 Compactness와 효율성

**적은 primitive 수로 고품질 달성**: Zhang et al. 의 highly compact point-based 모델과 비교하여, 약 1/4의 포인트 수로 동일한 PSNR 달성 (평균 모델 크기: 3.8 MB vs 9 MB).[1]

**장면당 Gaussian 수**: 모든 테스트 장면에서 1-5백만 Gaussian으로 합리적으로 compact한 표현 달성. 합성 장면은 30K iteration 후 200-500K Gaussian으로 수렴합니다.[1]

### 3.4 일반화 제한사항 및 개선 방향

**현재 제한사항**:

1. **Angular coverage 의존성**: 각도 정보가 부족한 캡처(예: 장면 코너, inside-out 캡처)에서 SH 0차 성분이 부정확할 수 있습니다. 이를 완화하기 위해 progressive SH band 최적화를 사용합니다.[1]

2. **MVS 데이터 불필요**: 대부분 포인트 기반 방법과 달리 MVS 데이터가 필요하지 않지만, SfM sparse points에 여전히 의존합니다.[1]

**일반화 향상 메커니즘**:

- **적응형 밀도 제어**: clone과 split 전략은 다양한 기하학적 복잡도에 자동으로 적응합니다.[1]
- **동일한 하이퍼파라미터**: 모든 평가에서 동일한 하이퍼파라미터 설정을 사용하여 강건성을 입증했습니다.[1]

## 4. 향후 연구에 미치는 영향과 고려사항 (최신 연구 기반)

### 4.1 후속 연구 동향 (2024-2025)

3D Gaussian Splatting은 2023년 발표 이후 폭발적인 연구 성장을 보였습니다. 최신 연구들은 다음 방향으로 발전하고 있습니다:

#### 동적 장면 확장

**BARD-GS (2025)**: 모션 블러와 부정확한 카메라 포즈를 처리하는 robust dynamic scene reconstruction을 제안합니다. 카메라 모션 블러와 객체 모션 블러를 명시적으로 분리하여 모델링합니다.[2]

**Sliding Windows for Dynamic 3DGS (ECCV)**: temporally-local dynamic MLP를 사용하여 동적 장면을 재구성합니다. 각 sliding window에 대해 별도의 canonical representation을 학습하여 significant geometric changes를 처리합니다.[3]

**Temporally Compressed 3DGS (TC3DGS, 2024)**: 동적 3D Gaussian 표현을 효과적으로 압축하여 AR/VR, 게임에 적합하도록 합니다. 최대 67배 압축을 달성하면서도 복잡한 동작을 고충실도로 표현합니다.[4]

#### 표현 방식 개선

**3D Convex Splatting (2024)**: Gaussian 대신 3D smooth convexes를 primitive로 사용합니다. Hard edges와 dense volumes를 더 적은 primitive로 표현할 수 있으며, 3DGS 대비 PSNR 0.81, LPIPS 0.026 개선을 달성합니다.[5]

**Deformable Beta Splatting (2025)**: Beta distribution을 사용한 새로운 radiance field 표현으로 3DGS보다 큰 장점을 제공합니다.[6]

**Triangle Splatting (2025)**: 실시간 radiance field 렌더링을 위한 또 다른 새로운 표현 방법입니다.[6]

#### 압축 및 효율성

**Compression in 3DGS Survey (2025)**: 3DGS의 압축 방법, 트렌드, 향후 방향을 종합적으로 조사합니다. 효율적인 NeRF 표현의 발전이 향후 3DGS 최적화에 영감을 줄 수 있다고 제안합니다.[7]

**DOGS (2024)**: 대규모 3D 재구성을 위한 distributed-oriented Gaussian Splatting을 제안합니다. Scene decomposition을 통해 학습 시간을 6배 이상 가속화합니다.[8]

#### 강건성 향상

**Robust Gaussian Splatting (2024)**: blur, 부정확한 카메라 포즈, 색상 불일치 등 일반적인 오류 소스를 해결합니다. Scannet++과 Deblur-NeRF 벤치마크에서 state-of-the-art 결과를 달성합니다.[9]

**GP-GS (2025)**: Gaussian Processes를 통합하여 sparse SfM point cloud의 한계를 극복합니다. Dense point clouds를 생성하여 고품질 초기 3D Gaussian을 제공합니다.[10]

**SelfSplat (2025)**: Pose-free와 3D prior-free generalizable 3D reconstruction을 수행합니다. Self-supervised depth와 pose estimation을 통합하여 상호 개선을 달성합니다.[11]

#### 응용 분야 확장

**Extended Reality (XR)**: 3DGS의 XR 적용에 대한 연구가 증가하고 있습니다. SAGE (2025)는 semantic-driven adaptive Gaussian Splatting을 XR에 적용합니다.[12][13]

**자율주행**: Dynamic radiance field framework가 자율주행 시나리오에 특화되어 개발되고 있습니다. SDF 기반 formulation을 통해 동적 객체와 정적 배경을 효과적으로 분리합니다.[14]

**물리 기반 시뮬레이션**: Physics-integrated Gaussian Splatting을 통한 editable dynamic scene modeling이 제안되었습니다. 물리적 속성을 임베딩하여 realistic, complex motion modeling을 가능하게 합니다.[15]

### 4.2 향후 연구 시 고려사항

#### 1. 메모리 효율성 개선

**압축 기술 통합**: Point cloud 압축은 잘 연구된 분야이며, 이러한 접근법을 3DGS 표현에 적용할 수 있는 많은 기회가 있습니다. 최신 연구들은 gradient-aware mixed-precision quantization 같은 기술을 탐구하고 있습니다.[4][1]

**적응형 해상도**: 장면의 복잡도에 따라 Gaussian 밀도를 동적으로 조절하는 방법이 필요합니다.

#### 2. 정규화 및 Prior 통합

**기하학적 prior**: 현재 방법은 정규화를 적용하지 않지만, depth maps나 다른 geometric priors를 통합하면 unseen regions와 popping artifacts를 완화할 수 있습니다.[10][1]

**Semantic information**: Semantic-driven approaches는 더 의미 있는 장면 분해를 가능하게 합니다.[13][16]

#### 3. 동적 장면 처리

**Temporal consistency**: 동적 장면에서 temporal flickering을 방지하고 일관된 motion modeling을 보장하는 것이 중요합니다.[2][3]

**Physics-based constraints**: 물리 법칙을 통합하면 더 현실적인 동작 예측과 편집이 가능합니다.[15]

#### 4. 일반화 능력 향상

**Pose estimation 통합**: Pose-free 방법은 실제 애플리케이션에서 매우 유용하며, self-supervised 기술과의 통합이 promising합니다.[11]

**Multi-scale representation**: Multi-scale Gaussian이나 hierarchical representations는 다양한 장면 스케일에 더 잘 적응할 수 있습니다.[17]

#### 5. 렌더링 품질 개선

**Antialiasing**: 간단한 visibility 알고리즘을 개선하고 antialiasing을 추가하면 popping artifacts를 완화할 수 있습니다.[1]

**Advanced rasterization**: 더 정교한 culling과 blending 전략이 시각적 품질을 향상시킬 수 있습니다.[5]

#### 6. 실용적 애플리케이션

**Real-time editing**: 실시간 장면 조작과 편집 기능은 XR과 게임 애플리케이션에 필수적입니다.[16][12]

**Hardware optimization**: Mobile devices와 저전력 플랫폼에서의 효율적인 실행을 위한 최적화가 필요합니다.[4]

### 4.3 연구 커뮤니티에 미친 영향

3D Gaussian Splatting은 radiance field 연구의 패러다임 전환을 촉발했습니다:[17]

1. **Explicit vs Implicit**: "연속 표현이 고품질 radiance field 학습에 필수적"이라는 통념을 깨뜨렸습니다.[1]

2. **Real-time rendering**: 처음으로 SOTA 품질과 실시간 렌더링을 동시에 달성했습니다.[1]

3. **Editability**: Explicit representation은 전례 없는 수준의 장면 제어와 편집 가능성을 제공합니다.[17]

4. **다양한 응용**: XR, 자율주행, 로봇공학 등 다양한 분야에서 실질적 영향을 미치고 있습니다.[12][13][14]

### 4.4 결론

3D Gaussian Splatting은 radiance field rendering의 혁신적 돌파구를 제공했으며, 실시간 고품질 렌더링이라는 오랜 과제를 해결했습니다. 2024-2025년 최신 연구들은 동적 장면, 압축, 강건성, XR 응용 등 다양한 방향으로 빠르게 발전하고 있습니다.[7][8][13][9][3][10][12][11][2][4][1]

향후 연구는 메모리 효율성, 일반화 능력, 동적 장면 처리, 실용적 애플리케이션 최적화에 집중해야 합니다. 특히 물리 기반 시뮬레이션, semantic information 통합, pose-free 방법, 그리고 다양한 표현 방식(convex, beta distribution 등)의 탐구가 promising한 방향입니다.[11][15][6][5]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/48b06790-56b2-490d-9764-e2a02a800fee/2308.04079v1.pdf)
[2](https://arxiv.org/abs/2503.15835)
[3](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07170.pdf)
[4](https://arxiv.org/abs/2412.05700)
[5](https://arxiv.org/abs/2411.14974)
[6](https://radiancefields.com/research)
[7](https://arxiv.org/html/2502.19457v1)
[8](https://arxiv.org/abs/2405.13943)
[9](https://arxiv.org/abs/2404.04211)
[10](https://arxiv.org/html/2502.02283v3)
[11](https://arxiv.org/html/2411.17190)
[12](http://arxiv.org/pdf/2412.06257.pdf)
[13](http://arxiv.org/pdf/2503.16747.pdf)
[14](https://www.sciencedirect.com/science/article/abs/pii/S0167865525002752)
[15](https://viplab.snu.ac.kr/viplab/courses/mlvu_2024_1/projects/11.pdf)
[16](https://arxiv.org/html/2503.11601v1)
[17](https://arxiv.org/html/2401.03890v8)
[18](https://github.com/Lee-JaeWon/2025-Arxiv-Paper-List-Gaussian-Splatting)
[19](https://www.sciencedirect.com/science/article/abs/pii/S092523122502301X)
[20](https://isprs-archives.copernicus.org/articles/XLVIII-G-2025/891/2025/isprs-archives-XLVIII-G-2025-891-2025.pdf)
[21](https://arxiv.org/html/2511.06408v1)
[22](https://mrnerf.github.io/awesome-3D-gaussian-splatting/)
[23](https://www.themoonlight.io/ko/review/temporally-compressed-3d-gaussian-splatting-for-dynamic-scenes)
[24](https://dl.acm.org/doi/10.1145/3687897)
[25](https://ieeexplore.ieee.org/iel8/6287639/10820123/10884729.pdf)
[26](https://github.com/hustvl/4DGaussians)
[27](https://ieeexplore.ieee.org/iel8/76/11223720/11016927.pdf)
[28](https://dl.acm.org/doi/10.1145/3728302)
[29](https://dynamic3dgaussians.github.io)
