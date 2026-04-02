# VastGaussian: Vast 3D Gaussians for Large Scene Reconstruction
---

## 1. 핵심 주장 및 주요 기여 요약

**VastGaussian**은 3D Gaussian Splatting(3DGS)을 대규모 장면(large-scale scene)으로 확장한 **최초의 방법**으로, 고품질 복원과 실시간 렌더링을 동시에 달성한다. 핵심 주장은 다음과 같다:

1. **Progressive Data Partitioning**: 대규모 장면을 여러 셀(cell)로 분할하고, airspace-aware visibility criterion을 활용하여 학습 카메라와 포인트 클라우드를 적절히 배분한 뒤, 병렬 최적화 후 seamless merging을 수행한다.
2. **Decoupled Appearance Modeling**: 학습 이미지 간 조명/노출 변화로 인한 appearance variation을 학습 과정에서만 사용되는 CNN 기반 변환 모듈로 분리하여, floater 아티팩트를 억제하고 일관된 렌더링을 달성한다. 이 모듈은 최적화 후 제거 가능하여 실시간 렌더링 속도에 영향을 주지 않는다.
3. **State-of-the-art 성능**: 기존 NeRF 기반 대규모 장면 복원 방법(Mega-NeRF, Switch-NeRF, Grid-NeRF)을 SSIM, LPIPS 등 주요 지표에서 압도하며, 훈련 시간은 약 3시간(8 GPU 병렬), 렌더링 속도는 1080p에서 약 140+ FPS를 달성한다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

3DGS는 소규모·객체 중심 장면에서 뛰어난 성능을 보이지만, 대규모 장면으로 확장 시 세 가지 핵심 문제가 발생한다:

| 문제 | 설명 |
|------|------|
| **비디오 메모리 제한** | 32GB GPU로 약 1,100만 개의 3D Gaussian만 최적화 가능. 대규모 장면은 수천만 개 이상 필요 |
| **긴 최적화 시간** | 전체 장면을 하나로 최적화하면 수렴이 느리고 불안정 |
| **외관 변동(Appearance Variation)** | 조명 불균일, 자동 노출 등으로 인해 학습 이미지 간 밝기 차이가 발생하고, 3DGS가 이를 보상하기 위해 저투명도의 큰 Gaussian(floater)을 생성 |

### 2.2 제안하는 방법

#### (A) 3DGS 기초 (Preliminaries)

3DGS는 3D Gaussian 집합 $\mathbf{G}$로 장면의 기하와 외관을 표현한다. 각 Gaussian은 위치, 이방성 공분산, 불투명도, 구면 조화(SH) 계수로 특성화된다. 카메라 $\mathcal{C}_i$에 대해 미분 가능한 래스터라이저 $\mathcal{R}$을 통해 렌더링 이미지를 얻는다:

$$\mathcal{I}_i^r = \mathcal{R}(\mathbf{G}, \mathcal{C}_i)$$

기본 손실 함수는 다음과 같다:

$$\mathcal{L} = (1 - \lambda)\mathcal{L}_1(\mathcal{I}_i^r, \mathcal{I}_i) + \lambda \mathcal{L}_{\text{D-SSIM}}(\mathcal{I}_i^r, \mathcal{I}_i) $$

여기서 $\lambda$는 하이퍼파라미터, $\mathcal{L}_{\text{D-SSIM}}$은 D-SSIM 손실이다.

#### (B) Progressive Data Partitioning (점진적 데이터 분할)

대규모 장면을 **분할-정복(divide-and-conquer)** 방식으로 처리한다. 파이프라인은 4단계로 구성된다:

**① Camera-position-based region division**: 카메라 위치를 지면 평면에 투영하여 $m \times n$ 그리드로 분할하되, 각 셀이 약 $|\mathbf{V}|/(m \times n)$개의 학습 뷰를 포함하도록 균등 분배한다.

**② Position-based data selection**: 각 셀의 경계를 20% 확장하여 확장된 영역 내의 카메라와 포인트를 할당한다. $j$번째 셀의 원래 크기가 $\ell_j^h \times \ell_j^w$일 때, 확장 후 크기는:

$$(\ell_j^h + 0.2\ell_j^h) \times (\ell_j^w + 0.2\ell_j^w)$$

**③ Visibility-based camera selection (핵심 기여)**: 아직 선택되지 않은 카메라 $\mathcal{C}_i$에 대해 $j$번째 셀의 가시성(visibility)을 정의한다:

$$\text{Visibility} = \frac{\Omega_{ij}}{\Omega_i}$$

여기서 $\Omega_i$는 이미지 $\mathcal{I}\_i$의 면적이다. 기존의 **airspace-agnostic** 방식은 표면 포인트의 투영 볼록 껍질 면적 $\Omega_{ij}^{\text{surf}}$를 사용하지만, 이는 공중 영역에 대한 감독이 부족하여 floater를 억제하지 못한다.

본 논문에서는 **airspace-aware** 방식을 제안한다: $j$번째 셀의 포인트 클라우드로 형성된 축 정렬 바운딩 박스(AABB)를 이미지 $\mathcal{I}\_i$에 투영하여 볼록 껍질 면적 $\Omega_{ij}^{\text{air}}$를 계산한다. 가시성이 임계값 $T_h$(논문에서 25%)를 초과하는 카메라를 추가 선택한다:

$$\frac{\Omega_{ij}^{\text{air}}}{\Omega_i} > T_h$$

**④ Coverage-based point selection**: 추가 선택된 카메라가 관측하는 포인트를 해당 셀의 포인트 집합 $\mathbf{P}_j$에 추가하여 초기화를 개선하고, 깊이 모호성으로 인한 floater를 방지한다.

#### (C) Decoupled Appearance Modeling (분리된 외관 모델링)

NeRF 기반 방법은 ray-marching 시 appearance embedding을 MLP에 입력하지만, 3DGS는 프레임 단위 래스터화를 사용하므로 이 방식이 부적합하다.

본 논문은 **최적화 과정에서만 적용되는** 외관 모델링을 제안한다:

1. 렌더링 이미지 $\mathcal{I}_i^r$을 32배 다운샘플링
2. 길이 $m$(논문에서 64)의 최적화 가능한 appearance embedding $\ell_i$를 픽셀별로 연결하여 $(3+m)$ 채널의 2D 맵 $\mathcal{D}_i$ 생성
3. CNN을 통해 $\mathcal{D}_i$를 원본 해상도의 transformation map $\mathcal{M}_i$로 업샘플링
4. 픽셀별 변환 $\mathcal{T}$를 적용하여 외관 조정 이미지 생성:

$$\mathcal{I}_i^a = \mathcal{T}(\mathcal{I}_i^r; \mathcal{M}_i) $$

실험에서는 간단한 **픽셀별 곱셈(pixel-wise multiplication)**이 효과적이었다.

수정된 손실 함수:

$$\mathcal{L} = (1 - \lambda)\mathcal{L}_1(\mathcal{I}_i^a, \mathcal{I}_i) + \lambda \mathcal{L}_{\text{D-SSIM}}(\mathcal{I}_i^r, \mathcal{I}_i) $$

- $\mathcal{L}_{\text{D-SSIM}}$: $\mathcal{I}_i^r$과 $\mathcal{I}_i$ 사이에 적용 → **구조적 일관성** 학습 (외관 변동 제외)
- $\mathcal{L}_1$: $\mathcal{I}_i^a$와 $\mathcal{I}_i$ 사이에 적용 → **외관 변동**을 embedding과 CNN이 흡수

**핵심 설계 철학**: D-SSIM은 구조적 비유사성을 주로 벌점하므로, 렌더링 이미지 $\mathcal{I}_i^r$이 일관된 외관과 올바른 기하를 학습하게 하고, 외관 변동은 별도 모듈이 담당한다. 최적화 후 이 모듈을 제거하면 **실시간 렌더링 속도가 유지**된다.

#### (D) Seamless Merging

각 셀 최적화 후, 원래 영역(확장 전) 외부의 3D Gaussian을 제거하고, 비중복 셀의 Gaussian을 단순 병합한다. 인접 셀 간 공유 카메라 덕분에 경계 아티팩트 없이 매끄러운 병합이 가능하다.

### 2.3 모델 구조

**전체 파이프라인**:

```
입력 (SfM 포인트 + 카메라) 
    → Progressive Data Partitioning (4단계)
        → 각 셀 독립 최적화 (3DGS + Decoupled Appearance Modeling)
            → Seamless Merging → 완전한 대규모 장면
```

**Decoupled Appearance Modeling CNN 구조** (Supplementary Material):
- 입력: $\frac{H}{32} \times \frac{W}{32} \times 67$ (다운샘플 이미지 3ch + embedding 64ch)
- Conv $3\times3$ → 256 채널
- 4× 업샘플링 블록 (Pixel Shuffle + Conv $3\times3$ + ReLU), 각 블록에서 해상도 2배, 채널 절반
- Bilinear interpolation → $H \times W \times 16$
- Conv $3\times3$ + ReLU + Conv $3\times3$ → $H \times W \times 3$ (Transformation Map)

### 2.4 성능 향상

**정량적 결과** (Table 1, 5개 대규모 장면 평균):

| 방법 | SSIM↑ | PSNR↑ | LPIPS↓ | FPS | Training |
|------|-------|-------|--------|-----|----------|
| Mega-NeRF | 0.624 | 23.84 | 0.378 | ~0.28 | ~29h |
| Switch-NeRF | 0.643 | 24.30 | 0.342 | ~0.14 | ~40h |
| Modified 3DGS | 0.790 | 24.66 | 0.184 | ~194 | ~20h |
| **VastGaussian** | **0.836** | **25.50** | **0.132** | **~149** | **~2.7h** |

- **SSIM, LPIPS에서 모든 장면에서 최고 성능**
- 훈련 시간 **약 10~15배 단축** (vs. NeRF 기반)
- 메모리 소비 약 10~12GB (vs. Modified 3DGS 31GB)
- Campus 장면: Modified 3DGS 890만 Gaussian → VastGaussian 2,740만 Gaussian

**Ablation Study 결과** (Table 3, Sci-Art 장면):

| 설정 | SSIM | PSNR | LPIPS |
|------|------|------|-------|
| w/o VisCam | 0.694 | 20.05 | 0.261 |
| w/o CovPoint | 0.874 | 26.14 | 0.128 |
| airspace-aware → agnostic | 0.855 | 24.54 | 0.128 |
| w/o Decoupled AM | 0.858 | 25.08 | 0.148 |
| **Full Model** | **0.885** | **26.81** | **0.121** |

### 2.5 한계

1. **최적 분할 전략 부재**: 장면 레이아웃, 셀 수, 카메라 분포를 고려한 최적 분할 솔루션을 제공하지 않음
2. **대규모 장면의 저장 공간 및 렌더링 속도**: 셀 수가 많아질수록 총 Gaussian 수가 크게 증가하여 저장 공간이 커지고 렌더링 속도가 저하될 수 있음
3. **셀 수 증가 시 PSNR 감소**: 셀이 16개 이상이면 원거리 셀 간 밝기 차이로 PSNR이 소폭 감소 (Table 4)
4. **Ground-level 장면 한정**: Manhattan world alignment을 가정하며, 주로 항공 촬영/도시 장면에 특화

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화 관련 설계

VastGaussian의 일반화 성능과 관련된 핵심 설계는 다음과 같다:

**Decoupled Appearance Modeling의 일반화 기여**:
- 학습 이미지 간 외관 변동을 분리하여 3D Gaussian이 **일관된 기하와 평균 외관**을 학습하도록 유도
- 이는 본질적으로 **도메인 불변(domain-invariant) 표현** 학습에 해당
- 손실 함수 Eq. (3)에서 $\mathcal{L}_{\text{D-SSIM}}(\mathcal{I}_i^r, \mathcal{I}_i)$는 구조적 일관성을, $\mathcal{L}_1(\mathcal{I}_i^a, \mathcal{I}_i)$는 외관 적합을 분리 담당

**Progressive Partitioning의 일반화 기여**:
- 그리드 기반 분할 외에도 부채꼴화(sectorization), 쿼드트리 등 다양한 지리 기반 분할에 적용 가능
- 셀 수 조절을 통한 유연한 스케일링

### 3.2 일반화 향상을 위한 잠재적 방향

1. **다양한 장면 유형으로의 확장**: 현재 항공 촬영 도시 장면에 특화되어 있으나, 실내 장면, 자연 환경, 동적 장면으로의 확장이 필요
2. **적응적 분할 전략**: 장면 복잡도에 따라 셀 크기와 수를 자동으로 결정하는 학습 기반 분할 방법
3. **외관 모델링 강화**: 단순 곱셈 변환을 넘어 비선형 변환, 시간 변화 조명 모델링, 날씨 변화 대응 등으로 확장
4. **Cross-scene 전이 학습**: 하나의 장면에서 학습한 appearance embedding이나 분할 전략을 새로운 장면에 전이
5. **Level-of-Detail (LoD) 통합**: 거리에 따른 Gaussian 밀도 조절로 다양한 스케일의 장면에 적응

### 3.3 일반화 성능의 잠재적 제약

- **Manhattan world alignment 가정**: 비정형 지형(산악, 동굴 등)에는 부적합
- **SfM 의존성**: 초기 포인트 클라우드 품질에 따라 성능 좌우
- **정적 장면 가정**: 동적 객체가 포함된 장면에서의 일반화 미검증
- **카메라 분포 의존성**: 균일하지 않은 카메라 분포 시 셀 간 불균형 발생 가능

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4.1 연구적 영향

1. **3DGS의 대규모 확장 패러다임 제시**: 분할-정복 + 외관 분리라는 프레임워크가 이후 대규모 3DGS 연구의 기본 패러다임으로 자리잡을 가능성
2. **실시간 렌더링과 고품질의 양립**: NeRF 기반 방법의 느린 렌더링 한계를 극복하면서도 품질을 유지하는 방향성 제시
3. **외관 변동 처리의 새로운 접근**: 기존 NeRF의 ray-marching 기반 appearance embedding과 달리, 래스터화 기반 렌더링에 적합한 이미지 공간 외관 모델링 제안
4. **산업 응용 가속화**: 자율주행, 항공 측량, VR/AR 등에서 대규모 장면의 실시간 렌더링 수요 충족

### 4.2 향후 연구 시 고려할 점

| 고려 사항 | 세부 내용 |
|-----------|----------|
| **최적 분할 전략** | 장면 구조, 카메라 궤적, 복잡도를 고려한 학습 기반 또는 적응적 분할 알고리즘 연구 |
| **Gaussian 압축** | 수천만 개 Gaussian의 저장/전송 효율화를 위한 압축 기법 필요 |
| **동적 장면 확장** | 4D Gaussian Splatting과의 결합으로 대규모 동적 장면 처리 |
| **품질-속도 트레이드오프** | 셀 수 증가에 따른 품질 포화와 속도 저하 간 최적점 탐색 |
| **평가 프로토콜** | 외관 변동이 있는 장면에서의 표준화된 평가 메트릭 필요 (color correction 의존도 감소) |
| **멀티 스케일 처리** | 극단적으로 다양한 스케일의 디테일이 공존하는 장면 처리 |
| **메모리 효율적 학습** | 단일 GPU에서도 대규모 장면 학습이 가능한 메모리 최적화 기법 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 대규모 장면 복원 방법 비교

| 방법 | 연도 | 표현 방식 | 분할 전략 | 외관 처리 | 실시간 렌더링 | 핵심 차별점 |
|------|------|----------|----------|----------|------------|-----------|
| **NeRF** [31] | 2020 | Implicit (MLP) | 없음 | 없음 | ✗ | 신경 복사 필드 기초 |
| **Mip-NeRF** [2] | 2021 | Implicit (MLP) | 없음 | 없음 | ✗ | 멀티스케일 앤티앨리어싱 |
| **NeRF-W** [29] | 2021 | Implicit (MLP) | 없음 | AE + MLP | ✗ | Wild 이미지에서의 외관 임베딩 |
| **Block-NeRF** [41] | 2022 | Implicit (MLP) | 위치 기반 블록 | AE + MLP | ✗ | 도시 스케일 NeRF, 블록 분할 |
| **Mega-NeRF** [44] | 2022 | Implicit (MLP) | 그리드 기반 (픽셀별 할당) | AE + MLP | ✗ | 픽셀 단위 그리드 할당 |
| **Instant-NGP** [32] | 2022 | Hybrid (Hash grid + MLP) | 없음 | 없음 | △ | 다해상도 해시 인코딩 |
| **Switch-NeRF** [61] | 2022 | Implicit (MoE) | 학습 기반 분해 | AE + MLP | ✗ | Mixture-of-Experts 장면 분해 |
| **Grid-NeRF** [53] | 2023 | Hybrid (Grid + NeRF) | 없음 (통합 방식) | 없음 | ✗ | NeRF + Grid 통합 |
| **3DGS** [21] | 2023 | Explicit (3D Gaussian) | 없음 | 없음 | ✓ | 명시적 Gaussian, 실시간 래스터화 |
| **VastGaussian** | 2024 | Explicit (3D Gaussian) | Progressive partitioning + airspace-aware visibility | Decoupled AM (CNN) | ✓ | 3DGS 최초 대규모 확장 |

### 5.2 핵심 기술별 심층 비교

#### (A) 장면 분할 전략

| 방법 | 분할 방식 | 장점 | 단점 |
|------|----------|------|------|
| Block-NeRF [41] | 위치 기반 블록 | 단순, 직관적 | 경계 아티팩트, 별도 외관 조정 필요 |
| Mega-NeRF [44] | 그리드 기반, 픽셀 단위 ray 할당 | 정밀한 데이터 분배 | 복잡한 구현, 픽셀 단위 처리 부담 |
| Switch-NeRF [61] | 학습 기반 MoE 분해 | 자동 분해 학습 | 학습 불안정, 긴 훈련 시간 |
| **VastGaussian** | Progressive partitioning (4단계) | 유연, 확장 가능, seamless merging | 최적 분할 미제공, Manhattan 가정 |

#### (B) 외관 변동 처리

| 방법 | 처리 방식 | 추론 시 오버헤드 | 3DGS 호환성 |
|------|----------|---------------|------------|
| NeRF-W [29] | Point AE + MLP | 있음 (AE 필요) | ✗ |
| Ha-NeRF [10] | 글로벌 AE + view-consistent loss | 있음 | ✗ |
| Block-NeRF [41] | AE + MLP + 후처리 외관 조정 | 있음 | ✗ |
| **VastGaussian** | 이미지 공간 CNN + AE (decoupled) | **없음** (모듈 제거 가능) | **✓** |

#### (C) 렌더링 속도 비교

| 방법 | FPS (1080p) | 실시간 여부 |
|------|------------|----------|
| Mega-NeRF | ~0.28 | ✗ |
| Switch-NeRF | ~0.14 | ✗ |
| Grid-NeRF | ~0.28 | ✗ |
| 3DGS (원본) | ~194 | ✓ |
| **VastGaussian** | **~149** | **✓** |

### 5.3 동시기 및 후속 관련 연구 동향

2023-2024년 사이 3DGS 기반 대규모 장면 연구가 활발히 진행되고 있으며, VastGaussian과 관련된 주요 동향은 다음과 같다:

1. **4D Gaussian Splatting** [51, 55, 56] (2023): 동적 장면으로의 확장. VastGaussian의 분할 전략과 결합 시 대규모 동적 장면 처리 가능성
2. **Gaussian 기반 3D 콘텐츠 생성** [12, 42, 59] (2023): Text-to-3D 생성에 Gaussian 활용. 대규모 장면 생성으로의 확장 가능
3. **Mip-NeRF 360** [3] (2022): 비제한 장면에서의 앤티앨리어싱. VastGaussian이 평가 시 color correction을 이 방법에서 차용
4. **Zip-NeRF** [4] (2023): Grid 기반 NeRF의 앤티앨리어싱 개선. 품질 측면에서 참고할 만한 기법
5. **BungeeNeRF** [52] (2022): 극단적 멀티스케일 렌더링. VastGaussian과 상호보완적 접근

---

## 참고자료

1. **Lin, J., Li, Z., Tang, X., et al.** "VastGaussian: Vast 3D Gaussians for Large Scene Reconstruction." *arXiv preprint arXiv:2402.17427v1*, 2024. (본 논문)
2. **Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G.** "3D Gaussian Splatting for Real-Time Radiance Field Rendering." *ACM Transactions on Graphics (ToG)*, 2023.
3. **Turki, H., Ramanan, D., & Satyanarayanan, M.** "Mega-NeRF: Scalable Construction of Large-Scale NeRFs for Virtual Fly-Throughs." *CVPR*, 2022.
4. **Tancik, M., Casser, V., Yan, X., et al.** "Block-NeRF: Scalable Large Scene Neural View Synthesis." *CVPR*, 2022.
5. **MI Zhenxing & Xu, D.** "Switch-NeRF: Learning Scene Decomposition with Mixture of Experts for Large-Scale Neural Radiance Fields." *ICLR*, 2022.
6. **Xu, L., Xiangli, Y., Peng, S., et al.** "Grid-Guided Neural Radiance Fields for Large Urban Scenes." *CVPR*, 2023.
7. **Martin-Brualla, R., et al.** "NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections." *CVPR*, 2021.
8. **Barron, J.T., et al.** "Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields." *CVPR*, 2022.
9. **Mildenhall, B., et al.** "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis." *ECCV*, 2020.
10. **Müller, T., Evans, A., Schied, C., & Keller, A.** "Instant Neural Graphics Primitives with a Multi-Resolution Hash Encoding." *ACM ToG*, 2022.
11. **Lin, L., Liu, Y., Hu, Y., et al.** "Capturing, Reconstructing, and Simulating: The UrbanScene3D Dataset." *ECCV*, 2022.
12. VastGaussian 프로젝트 페이지: https://vastgaussian.github.io
