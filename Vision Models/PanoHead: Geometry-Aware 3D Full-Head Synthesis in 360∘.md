# PanoHead: Geometry-Aware 3D Full-Head Synthesis in 360°

---

## 1. 핵심 주장 및 주요 기여 요약

**PanoHead**는 **in-the-wild 비정형 단일 뷰 이미지만으로 학습**하여, **360° 전 방위에서 뷰 일관적(view-consistent)이고 사실적인 풀헤드(full-head) 이미지와 정밀한 3D 기하를 합성**할 수 있는 최초의 3D-aware GAN 프레임워크이다.

### 주요 기여 4가지

| # | 기여 | 설명 |
|---|------|------|
| 1 | **Tri-Grid 표현** | tri-plane의 투영 모호성(projection ambiguity)을 해결하기 위해 깊이 차원 $D$를 추가한 tri-grid를 제안하여 전면/후면 특징 얽힘(entanglement) 제거 |
| 2 | **Foreground-Aware Tri-Discriminator** | 2D 세그멘테이션 사전 지식을 3D NeRF 밀도 분포 학습에 주입하여 전경(머리)과 배경을 완전히 분리 |
| 3 | **2단계 Self-Adaptive Camera Alignment** | 후두부 이미지의 부정확한 카메라 파라미터와 정렬 불일치를 자동 보정하는 카메라 자기 적응 모듈 도입 |
| 4 | **360° Full-Head 합성 및 단일 뷰 재구성** | 단일 입력 이미지로부터 GAN inversion을 통해 360° 풀헤드 3D 아바타 재구성 시연 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

기존 최신 3D GAN(특히 EG3D [5])은 다음 세 가지 한계로 인해 360° 풀헤드 합성에 실패한다:

1. **전경-배경 얽힘(Foreground-Background Entanglement):** Tri-plane 표현이 전경과 배경을 분리하지 못해 2.5D 형태의 기하가 생성되며, 대각도 렌더링 시 배경 구조물(벽 형태)이 머리와 엉김
2. **Tri-plane의 투영 모호성(Projection Ambiguity):** 전면 얼굴과 후두부의 3D 점이 동일한 2D 평면 좌표로 투영되어 "미러링 페이스(mirrored face)" 아티팩트 발생
3. **후두부 이미지의 카메라 정렬 불일치:** 얼굴 랜드마크가 검출되지 않는 후두부 이미지에서 카메라 외부 파라미터 추정과 크롭 정렬이 부정확

### 2.2 제안 방법 (수식 포함)

#### (A) Foreground-Aware Tri-Discrimination (Section 3.2)

볼륨 렌더링에서 raw feature 이미지 $I^r$과 전경 마스크 $I^m$을 다음과 같이 산출한다:

$$
I^r(\mathbf{r}) = \int_0^{\infty} w(t) \, f(\mathbf{r}(t)) \, dt, \qquad I^m(\mathbf{r}) = \int_0^{\infty} w(t) \, dt
$$

여기서 볼륨 렌더링 가중치 $w(t)$는:

$$
w(t) = \exp\!\left(-\int_0^{t} \sigma(\mathbf{r}(s)) \, ds\right) \sigma(\mathbf{r}(t))
$$

$\mathbf{r}(t)$는 카메라 중심에서 발사된 광선, $\sigma$는 밀도, $f$는 특징 함수이다.

별도의 StyleGAN2 네트워크가 저해상도 배경 $I^{bg}$를 생성하며, 최종 합성 이미지는:

$$
I^{gen} = (1 - I^m) \, I^{bg} + I^r
$$

**Tri-Discriminator**는 7채널 입력 $(I, I^+, I^{m+})$을 받아 RGB와 마스크를 동시에 판별하며, 이를 통해 2D 세그멘테이션 사전 지식이 Neural Radiance Field의 밀도 분포로 역전파된다.

#### (B) Tri-Grid 표현 (Section 3.3)

기존 tri-plane은 각 평면의 형태가 $H \times W \times C$이다. 예를 들어, 3D 점 $\mathbf{p}_1 = (x, y, z)$와 $\mathbf{p}_2 = (x, y, -z)$는 $P^{XY}$ 평면 위에서 동일한 좌표 $(x, y)$로 투영되어 **구별 불가능한 특징**을 공유한다.

**Tri-Grid**는 각 그리드의 형태를 $D \times H \times W \times C$로 확장하여, $Z$축을 따라 $D$개의 축 정렬 특징 평면 $P_i^{XY},\; i=1,\ldots,D$를 균등 배치한다. 임의의 3D 점에 대해 **삼선형 보간(trilinear interpolation)**으로 특징을 조회하므로, 동일한 $(x, y)$ 투영 좌표를 가지더라도 깊이 $z$가 다르면 **서로 다른 평면 조합**에서 보간되어 구별되는 특징 벡터를 얻는다.

StyleGAN2 생성기의 출력 채널 수를 $D$배로 증가시켜 $3 \times D$개의 특징 평면을 생성한다. 따라서 **tri-plane은 $D=1$인 tri-grid의 특수 경우**이다. 실험적으로 $D=3$ 정도의 작은 값으로 충분한 특징 분리가 가능하면서 효율성을 유지한다.

#### (C) Self-Adaptive Camera Alignment (Section 3.4)

**1단계:** 얼굴 랜드마크가 검출 가능한 이미지는 3DDFA [14]로 정렬, 검출 불가한 대각도 이미지는 WHENet [52] + YOLO [18]로 대략적 카메라 포즈와 바운딩 박스를 추정한 후 일정 오프셋으로 보정

**2단계:** 잔여 정렬 오차를 해결하기 위해, 각 학습 이미지에 대해 잠재 코드 $z$와 카메라 포즈 $c_{cam}$으로부터 **잔여 카메라 변환** $\Delta c_{cam}$을 공동 학습한다:

$$
c_{cam}^{\text{refined}} = c_{cam} + \Delta c_{cam}, \qquad \Delta c_{cam} = h(z, c_{cam})
$$

$\Delta c_{cam}$의 크기는 $L_2$ 정규화로 제어된다. 이 모듈은 3D-aware GAN이 다양한 카메라에서 뷰 일관적 이미지를 합성하는 특성을 활용하여 **동적으로 렌더링 프러스텀의 변환을 자기 보정**한다.

### 2.3 모델 구조

전체 파이프라인은 세 핵심 모듈로 구성된다 (Figure 2):

```
z ~ p_z(0,1), c ~ p_c
        │
        ▼
  ┌─────────────┐
  │ Mapping Net  │ ─── c_con ──→ Mapping Network M ──→ w
  └─────────────┘
        │ w
        ▼
  ┌─────────────┐
  │ StyleGAN2 G  │ ──→ Tri-Grid feature f (3×D planes, reshape)
  └─────────────┘
        │ f + c_cam (+ Δc_cam)
        ▼
  ┌──────────────────┐
  │ Neural Renderer R │
  │  - Trilinear int. │
  │  - MLP (color/σ)  │
  │  - Volume Render  │
  │  - BG generator   │
  │  - Super-res      │
  └──────────────────┘
        │
     (I⁺, I, I^{m+})
        │
        ▼
  ┌─────────────────────┐
  │ Tri-Discriminator D  │ ──→ real/fake (7-ch input)
  └─────────────────────┘
```

- **Generator G:** StyleGAN2 백본 → tri-grid ($3 \times D$ 특징 평면) 출력
- **Neural Renderer R:** Trilinear interpolation → MLP (밀도 $\sigma$ + 색상) → 볼륨 렌더링 → 배경 합성 → Super-resolution
- **Discriminator D:** 7채널 tri-discriminator $(I, I^+, I^{m+})$
- **Camera Self-Adaptation:** $(z, c_{cam}) \rightarrow \Delta c_{cam}$ 잔여 변환 학습

### 2.4 성능 향상

**Table 1: 전체 비교 (FFHQ-F 데이터셋)**

| 메트릭 | GRAF | GIRAFFEHD | StyleSDF | EG3D | **PanoHead** |
|--------|------|-----------|----------|------|------------|
| FID-all $\downarrow$ | 68.2 | 37.3 | 78.5 | 6.2 | **5.4** |
| MSE ($\times 10^{-2}$) $\downarrow$ | N/A | 42.6 | N/A | N/A | **9.1** |
| ID $\uparrow$ | N/A | 0.39 | 0.41 | 0.74 | **0.74** |

**Table 2: Ablation Study**

| 구성 | FID-back $\downarrow$ | FID-front $\downarrow$ | IS-back $\uparrow$ | Runtime |
|------|---------|-----------|---------|---------|
| EG3D (baseline) | 50.4 | 6.6 | 4.3 | 1× |
| +seg. (tri-plane) | 44.1 | 5.0 | 3.9 | 1.14× |
| +seg. (tri-grid) | 44.0 | 5.5 | 4.2 | 1.26× |
| **+seg.&self-adapt. (tri-grid)** | **40.9** | 5.4 | **4.4** | 1.28× |

핵심 관찰:
- Foreground-aware tri-discrimination 추가만으로 FID-back이 50.4 → 44.1로 크게 감소
- Tri-grid 교체 시 IS-back 향상 (3.9 → 4.2)
- Camera self-adaptation 추가 시 FID-back이 최저치 40.9 달성
- 전체 계산 오버헤드는 1.28배로 미미

### 2.5 한계점

1. **치아 영역 등 세부 아티팩트**: 여전히 미세한 아티팩트 존재
2. **텍스처 플리커링(Flickering)**: EG3D로부터 상속된 문제, StyleGAN3 전환 시 개선 가능
3. **고주파 기하 디테일 부족**: 머리카락 끝 등의 정밀 기하 표현 한계
4. **데이터 편향(Data Bias)**: FFHQ + K-hairstyle + 자체 수집 데이터의 조합에 의존하므로 인종·성별·스타일 다양성에서 여전히 편향 가능
5. **정량적 기하 평가 부재**: Depth map 등을 통한 기하 품질의 체계적 정량 평가 미실시
6. **조건부 카메라 포즈의 영향**: Conditioning 포즈와 렌더링 포즈가 다를 때 품질 저하 (FID-back 제안의 동기)

---

## 3. 모델의 일반화 성능 향상 가능성

PanoHead의 일반화 성능과 관련된 핵심 포인트를 다음과 같이 분석한다.

### 3.1 현재 일반화 강점

- **In-the-wild 비정형 이미지 학습**: 제어된 환경이 아닌 wild 이미지(FFHQ 70K + K-hairstyle 4K + 자체 15K)만으로 학습하여, 다양한 조명·배경·스타일에 대한 일반화를 보여줌
- **360° 전방위 합성**: 전면뿐 아니라 측면·후면을 포함한 완전한 포즈 공간에서의 일반화
- **다양한 외관(Diverse Appearance)**: 면도 머리, 안경, 긴 곱슬머리, 아프로 헤어 등 다양한 스타일에 대해 일관된 품질
- **단일 뷰 3D 재구성**: GAN inversion (L2 + LPIPS 최적화 → PTI fine-tuning)을 통해 임의의 단일 이미지에서 360° 풀헤드 복원이 가능하여, 학습 시 보지 못한 개인에 대한 일반화 시연

### 3.2 일반화 성능 향상을 위한 잠재적 방향

#### (1) 데이터 확장 및 다양성 증대
논문에서 명시적으로 "large-scale full-head annotated training image dataset is one of the most critical directions"라고 강조한다. 현재 FFHQ-F는 약 89K 이미지로 구성되며, 특히 후두부 이미지(19K)가 상대적으로 적어 **데이터 불균형**이 존재한다. 대규모 다인종·다스타일 풀헤드 데이터셋 구축이 일반화 향상의 가장 직접적 방법이다.

#### (2) 백본 아키텍처 업그레이드
- **StyleGAN3**: Alias-free 설계로 텍스처 플리커링 완화 및 고주파 디테일 보존
- **Diffusion Model 기반**: 최근 3D-aware diffusion 모델(예: DiffusionGAN3D, SiTH 등)로의 전환 시 mode collapse 완화와 더 안정적 학습으로 일반화 향상 기대

#### (3) Tri-Grid 깊이 $D$의 적응적 조절
현재 $D$는 고정값(예: 3)이지만, 입력의 복잡도에 따라 적응적으로 조절하거나, 계층적(multi-scale) tri-grid 구조를 도입하면 더 풍부한 표현력과 효율 사이의 최적 균형을 달성할 수 있다.

#### (4) Cross-Domain 일반화
현재 모델은 인간 머리에 특화되어 있으나, tri-grid 표현과 foreground-aware 학습 전략은 **동물 머리, 전신(full-body), 일반 물체** 등 다른 도메인의 360° 생성에도 적용 가능한 일반적 프레임워크이다.

#### (5) Camera Self-Adaptation의 일반화
잔여 카메라 변환 $\Delta c_{cam}$의 학습은 본질적으로 **noisy annotation에 대한 강건성**을 제공하며, 이는 더 다양하고 품질이 낮은 데이터셋(예: 소셜 미디어 이미지)에서의 학습을 가능하게 하여 일반화 범위를 확장한다.

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 연구 영향

1. **3D-aware GAN의 전방위 합성 가능성 입증**: PanoHead 이전에는 어떤 3D GAN도 360° 풀헤드 합성을 달성하지 못했으며, 이 연구는 디지털 아바타, 텔레프레즌스, 메타버스 응용의 실질적 가능성을 열었다.

2. **Tri-Grid 표현의 범용성**: Tri-plane의 투영 모호성 문제에 대한 근본적 해결책으로서, 풀헤드뿐 아니라 **전신 아바타, 실내 장면, 물체 생성** 등 다양한 3D 생성 과제에 적용 가능하다.

3. **In-the-wild 학습 파이프라인 확립**: 2단계 정렬 + 카메라 자기 적응은 비정형 데이터에서의 3D GAN 학습을 위한 표준적 전처리 방법론을 제시한다.

4. **단일 뷰 3D 재구성의 실용화**: GAN inversion + PTI를 통한 360° 재구성은 일반 사용자의 셀카 한 장으로 3D 아바타를 생성하는 실용적 파이프라인을 시연하였다.

### 4.2 향후 연구 시 고려할 점

| 고려 사항 | 구체적 내용 |
|-----------|------------|
| **기하 품질 정량 평가** | Depth map, Chamfer distance, normal consistency 등을 통한 체계적 3D 기하 평가 필요 |
| **고주파 디테일** | 머리카락 끝, 이어링 등 미세 구조의 정밀 모델링을 위해 multi-scale 표현 또는 point-based rendering 통합 검토 |
| **시간적 일관성** | 비디오 아바타 응용 시 프레임 간 텍스처 플리커링 해결이 필수적 (StyleGAN3 또는 temporal consistency loss) |
| **조건부 생성 및 편집** | 표정 제어, 헤어스타일 변경, 조명 재조명 등의 조건부 제어 통합 필요 |
| **윤리적 고려** | Deepfake 악용 방지를 위한 워터마킹, 탐지 기법과의 병행 연구 필수 |
| **계산 효율** | Tri-grid의 $D$ 증가에 따른 메모리·연산 오버헤드 최적화 (예: sparse grid, hash encoding) |
| **대규모 데이터셋 구축** | 후두부·측면 이미지를 포함한 대규모 다양성 데이터셋의 공개적 구축이 연구 커뮤니티에 필수 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

아래는 PanoHead와 관련된 2020년 이후 주요 3D-aware 생성 모델들을 비교 분석한 표이다.

| 연구 | 연도 | 3D 표현 | 360° 지원 | 핵심 차별점 | 한계 |
|------|------|---------|----------|------------|------|
| **GRAF** [37] | 2020 (NeurIPS) | NeRF | ✗ | 최초의 NeRF 기반 GAN | 저해상도, 카메라 분포 학습 실패 |
| **pi-GAN** [6] | 2021 (CVPR) | SIREN-based NeRF | ✗ | Periodic activation으로 세밀한 표현 | 연산 비용 과다, 저해상도 |
| **GIRAFFE** [29] | 2021 (CVPR) | Compositional NeRF | ✗ | 장면 구성적 분해 | 고해상도 합성 어려움 |
| **StyleNeRF** [13] | 2022 (CVPR) | NeRF + style modulation | 제한적 | Progressive volume rendering으로 효율 향상 | 대각도에서 품질 저하 |
| **EG3D** [5] | 2022 (CVPR) | **Tri-plane** | ✗ (정면 위주) | StyleGAN2 + tri-plane, dual discrimination | 미러링 페이스, 전경-배경 미분리, ~60° 이내 |
| **StyleSDF** [31] | 2022 (CVPR) | SDF + NeRF hybrid | ✗ | SDF 기반 기하 + volume rendering | 후두부 기하 깨짐, 대각도 부적합 |
| **GIRAFFEHD** [48] | 2022 (CVPR) | Compositional NeRF | ✗ | GIRAFFE의 고해상도 확장 | 360° 카메라 분포 해석 실패 |
| **VoxGRAF** [38] | 2022 | Sparse Voxel Grid | 제한적 | Voxel 기반 효율적 3D 표현 | 해상도 제약, 세부 표현 한계 |
| **GET3D** [11] | 2022 (NeurIPS) | DMTet mesh | ✗ (물체 중심) | 미분가능 래스터라이제이션, 텍스처 메쉬 직접 생성 | 인체 머리 특화 아님 |
| **ENARF-GAN** [30] | 2022 (ECCV) | Articulated NeRF | 제한적 | 관절 표현 + 마스크 기반 전경-배경 분리 | 단일 판별자로 고해상도 일관성 미보장 |
| **PanoHead** (본 논문) | 2023 | **Tri-Grid** | **✓ (360°)** | Tri-grid + tri-discriminator + self-adaptive camera alignment | 텍스처 플리커링, 고주파 기하 한계, 데이터 편향 |

### 비교 분석 핵심

1. **EG3D vs. PanoHead**: 동일한 StyleGAN2 백본을 공유하지만, PanoHead는 tri-plane을 tri-grid로 확장하여 투영 모호성을 해결하고, tri-discriminator로 전경-배경 분리를 달성하며, camera self-adaptation으로 noisy annotation 문제를 극복하여 **FID-back을 50.4에서 40.9로** 대폭 개선하였다.

2. **StyleSDF vs. PanoHead**: StyleSDF는 SDF 기반 기하를 도입했으나 360° 합성에서 기하가 깨지는 반면, PanoHead의 NeRF 기반 볼륨 렌더링 + tri-grid는 완전한 헤드 기하를 유지한다.

3. **Diffusion 기반 후속 연구와의 관계**: PanoHead 발표 이후 **Score Distillation Sampling (SDS)** 기반 3D 생성(DreamFusion, Magic3D 등)과 **3D Gaussian Splatting** 기반 접근이 급부상하였다. 이들은 GAN의 mode collapse 문제를 회피하면서 더 유연한 3D 생성이 가능하나, PanoHead의 tri-grid 표현과 foreground-aware 학습 전략은 이러한 후속 프레임워크에도 적용 가능한 범용적 기법이다.

---

## 참고자료

1. **An, S., Xu, H., Shi, Y., Song, G., Ogras, U. Y., & Luo, L.** (2023). "PanoHead: Geometry-Aware 3D Full-Head Synthesis in 360°." *arXiv preprint arXiv:2303.13071*. — 본 논문 원문
2. **Chan, E. R., et al.** (2022). "Efficient Geometry-Aware 3D Generative Adversarial Networks." *CVPR 2022*. — EG3D, 기반 프레임워크
3. **Schwarz, K., et al.** (2020). "GRAF: Generative Radiance Fields for 3D-Aware Image Synthesis." *NeurIPS 2020*. — 최초 NeRF 기반 GAN
4. **Or-El, R., et al.** (2022). "StyleSDF: High-Resolution 3D-Consistent Image and Geometry Generation." *CVPR 2022*. — SDF 기반 비교 대상
5. **Xue, Y., et al.** (2022). "GIRAFFE HD: A High-Resolution 3D-Aware Generative Model." *CVPR 2022*. — 비교 대상
6. **Karras, T., et al.** (2020). "Analyzing and Improving the Image Quality of StyleGAN." *CVPR 2020*. — StyleGAN2, 핵심 백본
7. **Karras, T., et al.** (2021). "Alias-Free Generative Adversarial Networks." *NeurIPS 2021*. — StyleGAN3, 한계 개선 제안
8. **Noguchi, A., et al.** (2022). "Unsupervised Learning of Efficient Geometry-Aware Neural Articulated Representations." *ECCV 2022*. — ENARF-GAN
9. **Schwarz, K., et al.** (2022). "VoxGRAF: Fast 3D-Aware Image Synthesis with Sparse Voxel Grids." *arXiv:2206.07695*. — Voxel 기반 비교 대상
10. **Mildenhall, B., et al.** (2020). "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis." *ECCV 2020*. — NeRF 기초
11. **Guo, J., et al.** (2020). "Towards Fast, Accurate and Stable 3D Dense Face Alignment." *ECCV 2020*. — 3DDFA, 카메라 추정
12. **Zhou, Y. & Gregson, J.** (2020). "WHENet: Real-Time Fine-Grained Estimation for Wide Range Head Pose." *BMVC 2020*. — 후두부 포즈 추정
13. **PanoHead 프로젝트 페이지**: https://sizhean.github.io/panohead
