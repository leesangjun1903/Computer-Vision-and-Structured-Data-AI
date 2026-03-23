# 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering

---

## 1. 핵심 주장 및 주요 기여 요약

본 논문은 동적 장면(dynamic scene)의 실시간 렌더링을 위해 **4D Gaussian Splatting (4D-GS)**이라는 통합적 표현 방식을 제안한다. 기존 3D Gaussian Splatting(3D-GS)이 정적 장면에 한정되었던 한계를 극복하고, 시간 축을 포함한 4D 표현을 구축하여 동적 장면에서도 **실시간 렌더링(82 FPS @ 800×800, RTX 3090)**과 **높은 렌더링 품질(PSNR 34.05 dB)**을 동시에 달성한다.

### 주요 기여:
1. **효율적인 4D Gaussian Splatting 프레임워크**: Gaussian 운동(motion)과 형상 변형(shape deformation)을 동시에 모델링하는 Gaussian Deformation Field 제안
2. **다중 해상도 시공간 구조 인코더(Spatial-Temporal Structure Encoder)**: HexPlane에서 영감을 받은 분해된 4D 뉴럴 복셀 인코딩으로 인접 3D Gaussian들의 풍부한 특징을 효율적으로 구축
3. **실시간 렌더링 달성**: 합성 데이터셋에서 82 FPS, 실제 데이터셋에서 30 FPS를 기록하며, 이전 SOTA 대비 비교 가능하거나 우수한 품질 유지

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

동적 장면의 Novel View Synthesis(NVS)는 시공간적으로 희소한 입력으로부터 복잡한 움직임을 모델링해야 하는 도전적인 과제이다. 기존 접근법들의 한계는 다음과 같다:

| 문제점 | 기존 방법 | 한계 |
|--------|----------|------|
| 렌더링 속도 | NeRF 기반 (D-NeRF, HyperNeRF 등) | 볼륨 렌더링으로 인해 실시간 렌더링 불가능 (< 3 FPS) |
| 메모리 효율 | Dynamic3DGS | 매 타임스탬프마다 개별 3D Gaussian 저장 → $O(tN)$ 메모리 복잡도 |
| 동적 모델링 | 3D-GS | 정적 장면에만 적용 가능, 동적 장면 재구성 실패 |
| 시간적 일관성 | 4D Gaussian (Yang et al.) | 각 Gaussian이 국소적 시간 공간에만 집중 |

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 3D Gaussian Splatting 기초

각 3D Gaussian은 공분산 행렬 $\Sigma$와 중심점 $\mathcal{X}$로 정의된다:

$$G(X) = e^{-\frac{1}{2}X^T \Sigma^{-1} X} \tag{1}$$

미분 가능한 최적화를 위해 공분산 행렬을 스케일링 행렬 $\mathbf{S}$와 회전 행렬 $\mathbf{R}$로 분해한다:

$$\Sigma = \mathbf{R}\mathbf{S}\mathbf{S}^T\mathbf{R}^T \tag{2}$$

카메라 좌표에서의 공분산 행렬은 뷰 변환 행렬 $W$와 야코비안 행렬 $J$를 사용하여 계산된다:

$$\Sigma' = JW\Sigma W^T J^T \tag{3}$$

픽셀 색상의 알파 블렌딩:

$$C = \sum_{i \in N} c_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_i) \tag{4}$$

#### 2.2.2 4D Gaussian Splatting 프레임워크

뷰 행렬 $M = [R, T]$와 타임스탬프 $t$가 주어지면, 새로운 뷰 이미지는 다음과 같이 렌더링된다:

$$\hat{I} = \mathcal{S}(M, \mathcal{G}'), \quad \text{where} \quad \mathcal{G}' = \Delta\mathcal{G} + \mathcal{G}$$

Gaussian 변형은 Gaussian Deformation Field Network에 의해 생성된다:

$$\Delta\mathcal{G} = \mathcal{F}(\mathcal{G}, t)$$

이는 기존 동적 NeRF의 **world-to-canonical 매핑**과 달리, **canonical-to-world 매핑**을 직접 수행한다는 점에서 핵심적 차이를 갖는다.

#### 2.2.3 Gaussian Deformation Field Network

**① Spatial-Temporal Structure Encoder $\mathcal{H}$:**

6개의 다중 해상도 평면 모듈 $R_l(i,j)$과 경량 MLP $\phi_d$로 구성된다:

$$\mathcal{H}(\mathcal{G}, t) = \{R_l(i,j), \phi_d \mid (i,j) \in \{(x,y),(x,z),(y,z),(x,t),(y,t),(z,t)\}, \; l \in \{1,2\}\}$$

각 복셀 모듈은 $R(i,j) \in \mathbb{R}^{h \times lN_i \times lN_j}$로 정의되며, 복셀 특징은 다음과 같이 계산된다:

$$f_h = \bigcup_l \prod \text{interp}(R_l(i,j)), \quad (i,j) \in \{(x,y),(x,z),(y,z),(x,t),(y,t),(z,t)\} \tag{7}$$

여기서 $f_h \in \mathbb{R}^{h \cdot l}$이며, 'interp'는 격자 4개 꼭짓점에서의 이중선형 보간(bilinear interpolation)을 의미한다. 이후 경량 MLP가 모든 특징을 병합한다:

$$f_d = \phi_d(f_h)$$

**② Multi-head Gaussian Deformation Decoder $\mathcal{D}$:**

$\mathcal{D} = \{\phi_x, \phi_r, \phi_s\}$로 구성되며, 각각 위치, 회전, 스케일링의 변형을 예측한다:

$$\Delta\mathcal{X} = \phi_x(f_d), \quad \Delta r = \phi_r(f_d), \quad \Delta s = \phi_s(f_d)$$

최종 변형된 Gaussian 속성:

$$(\mathcal{X}', r', s') = (\mathcal{X} + \Delta\mathcal{X}, \; r + \Delta r, \; s + \Delta s) \tag{8}$$

변형된 3D Gaussian: $\mathcal{G}' = \{\mathcal{X}', s', r', \sigma, \mathcal{C}\}$

#### 2.2.4 최적화

**손실 함수:**

$$\mathcal{L} = |\hat{I} - I| + \mathcal{L}_{tv} \tag{9}$$

여기서 $\mathcal{L}_{tv}$는 격자 기반 총변동(total variation) 손실이다.

**3D Gaussian 초기화**: 처음 3000 이터레이션에서 $\hat{I} = \mathcal{S}(M, \mathcal{G})$로 워밍업한 후, 4D Gaussian 공동 최적화 $\hat{I} = \mathcal{S}(M, \mathcal{G}')$로 전환한다.

### 2.3 모델 구조

전체 파이프라인은 다음과 같이 구성된다:

```
3D Gaussians G (위치 x,y,z) + Timestamp t
        ↓
Spatial-Temporal Structure Encoder (6개 HexPlane + MLP)
        ↓
    Feature f_d
        ↓
Multi-head Gaussian Deformation Decoder
    ├── Position Head φ_x → Δx, Δy, Δz
    ├── Rotation Head φ_r → Δr
    └── Scaling Head φ_s → Δs
        ↓
Deformed 3D Gaussians G'
        ↓
    Splatting → Rendered Image
```

**메모리 복잡도**: $O(N + \mathcal{F})$ (3D Gaussian 수 $N$ + 변형 필드 네트워크 파라미터 $\mathcal{F}$), Dynamic3DGS의 $O(tN)$에 비해 현저히 효율적.

### 2.4 성능 향상

#### 합성 데이터셋 (D-NeRF) 결과:

| 모델 | PSNR (dB)↑ | SSIM↑ | LPIPS↓ | 학습시간↓ | FPS↑ | 저장공간(MB)↓ |
|------|-----------|-------|--------|---------|------|------------|
| TiNeuVox-B | 32.67 | 0.97 | 0.04 | 28 min | 1.5 | 48 |
| V4D | 33.72 | 0.98 | 0.02 | 6.9 h | 2.08 | 377 |
| **4D-GS (Ours)** | **34.05** | **0.98** | **0.02** | **8 min** | **82** | **18** |

#### 실세계 데이터셋 결과:
- **HyperNeRF (960×540)**: PSNR 25.2 dB, 34 FPS (기존 SOTA 대비 PSNR +0.4 dB, FPS 34배 향상)
- **Neu3D (1352×1014)**: PSNR 31.15 dB, 30 FPS, 90 MB 저장공간

#### Ablation Study 핵심 결과:

| 구성 | PSNR | FPS | 분석 |
|------|------|-----|------|
| w/o HexPlane $R_l(i,j)$ | 27.05 | 140 | 공간-시간 구조 인코더 제거 시 품질 급감 (−7.0 dB) |
| w/o initialization | 31.91 | 79 | 워밍업 없이 직접 학습 시 수렴 어려움 (−2.14 dB) |
| w/o $\phi_x$ | 26.67 | 82 | 위치 변형 없이는 동적 장면 모델링 실패 (−7.38 dB) |
| w/o $\phi_r$ | 33.08 | 83 | 회전 변형 제거 시 세부 모델링 저하 (−0.97 dB) |
| w/o $\phi_s$ | 33.02 | 82 | 스케일 변형 제거 시 미세 움직임 모델링 저하 (−1.03 dB) |
| **Full model** | **34.05** | **82** | - |

### 2.5 한계점

1. **큰 움직임(Large Motion)**: 변형 필드 네트워크가 급격한 움직임이나 장면 변화를 모델링하는 데 실패할 수 있음
2. **배경 포인트 부재 및 부정확한 카메라 포즈**: 4D Gaussian 최적화에 어려움 초래
3. **단안(Monocular) 설정에서의 정적/동적 분리**: 추가 감독 신호 없이 정적·동적 Gaussian의 공동 움직임을 분리하기 어려움
4. **도시 규모(Urban-scale) 재구성**: 방대한 수의 3D Gaussian에 대한 변형 필드 쿼리가 비용이 큼
5. **단안 설정에서의 과적합(Overfitting)**: 카메라 포즈와 타임스탬프 차원에서 희소한 입력으로 인해 새로운 뷰 렌더링 시 과적합 발생 가능

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화 성능의 특성

4D-GS는 **장면별(scene-specific) 최적화** 방식으로, 한 장면에서 학습한 모델을 다른 장면에 직접 전이할 수 없다. 이는 3D-GS 및 대부분의 NeRF 기반 방법론과 동일한 한계이다.

그러나 4D-GS의 구조적 설계에는 일반화 성능 향상의 잠재력이 내재되어 있다:

#### (1) Spatial-Temporal Structure Encoder의 귀납적 편향(Inductive Bias)
- 인접 Gaussian들이 유사한 시공간 정보를 공유한다는 가정 하에, HexPlane 기반 인코더가 **공간적 연속성**과 **시간적 일관성**을 자연스럽게 부여한다.
- 이러한 구조적 사전 지식은 새로운 타임스탬프나 근접한 시점에서의 보간 성능을 향상시킨다.
- Fig. 13에서 보여주듯이, $R_1(x,t)$, $R_1(y,t)$, $R_1(z,t)$ 평면이 장면의 통합된 움직임 정보를 명시적으로 인코딩하여 **시간 축 보간**(temporal interpolation)의 일반화를 지원한다.

#### (2) Canonical-to-World 매핑의 장점
- 기존 deformation NeRF의 world-to-canonical 매핑과 달리, 4D-GS의 canonical-to-world 매핑은 **역방향 흐름(backward flow) 계산과 트래킹**이 가능하여, 시간 축에서의 일반화를 강화한다.

#### (3) 다중 해상도 표현
- $l \in \{1, 2\}$의 다중 해상도 평면을 사용하여, 서로 다른 스케일의 움직임을 계층적으로 포착함으로써 다양한 동적 패턴에 대한 적응력을 높인다.

### 3.2 일반화 성능 향상을 위한 향후 방향

1. **깊이(Depth) 감독 및 광학 흐름(Optical Flow) 활용**: 논문에서 언급한 바와 같이 (Sec. A.3), 단안 설정에서의 과적합 문제를 완화하기 위해 추가적인 기하학적 사전 지식이 필요하다.

2. **크로스 장면 일반화(Cross-scene Generalization)**: IBRNet이나 Generalizable Neural Voxels와 같은 일반화 가능한 프레임워크와의 통합을 통해, 학습되지 않은 새로운 장면에도 적용 가능한 모델로 확장할 수 있다.

3. **정적/동적 분리 메커니즘**: 추가적인 감독 신호(motion mask, semantic segmentation 등)를 활용하여 정적 배경과 동적 전경을 명시적으로 분리함으로써 일반화 성능을 향상시킬 수 있다.

4. **대규모 움직임에 대한 로버스트니스**: 온라인 학습(online training) 또는 다중 뷰 정보 활용으로 큰 움직임에 대한 대응력을 강화할 수 있다.

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 4.1 연구적 영향

#### (1) 패러다임 전환: NeRF에서 Gaussian Splatting으로
4D-GS는 동적 장면 렌더링에서 볼륨 렌더링 기반의 NeRF 패러다임을 **포인트 기반 래스터라이제이션** 패러다임으로 전환하는 핵심적 연구이다. 이는 실시간 응용(VR, AR, 영화 제작)에서의 실용적 배포 가능성을 크게 높인다.

#### (2) 명시적 표현의 장점 재확인
3D Gaussian의 명시적 특성은 장면 편집, 합성(composition), 트래킹과 같은 다운스트림 태스크를 자연스럽게 지원한다. 이는 암시적(implicit) 표현의 불투명성(opacity)에 대한 대안을 제시한다.

#### (3) 효율성과 품질의 균형
학습 시간 8분, 저장 공간 18MB로 SOTA 수준의 품질을 달성함으로써, 연구 및 산업 환경에서의 접근성을 대폭 향상시킨다.

#### (4) 분해된(Decomposed) 시공간 인코딩의 보편화
HexPlane/K-Planes에서 영감을 받은 6개 평면 분해 방식은 4D 뉴럴 복셀의 메모리 효율적 대안으로서, 향후 다양한 4D 표현 연구에 핵심 모듈로 채택될 것으로 예상된다.

### 4.2 향후 연구 시 고려할 점

| 연구 방향 | 세부 사항 |
|----------|----------|
| **대규모 동적 장면** | 도시 규모 장면에서의 효율적인 Gaussian 변형 필드 쿼리 알고리즘 설계 |
| **토폴로지 변화** | 물체의 등장/퇴장, 분리/결합 등 위상학적 변화를 모델링하기 위한 Gaussian 생성/소멸 메커니즘 |
| **일반화 모델** | 장면별 최적화 없이 새로운 장면에 즉시 적용 가능한 피드포워드 4D-GS 모델 |
| **물리적 사전 지식** | 물리적 시뮬레이션, 인체 스켈레톤 모델 등의 도메인 특화 사전 지식 통합 |
| **장기 시퀀스** | 수분~수시간에 걸친 장기 동적 장면에서의 효율적 표현 및 누적 오차 관리 |
| **품질 향상** | Anti-aliasing, 반사/투과 등 복잡한 광학적 효과의 정확한 모델링 |
| **스트리밍/온라인 학습** | 실시간으로 들어오는 프레임을 점진적으로 학습하는 온라인 4D-GS |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

### 5.1 동적 NeRF 계열

| 연구 | 연도 | 핵심 접근법 | PSNR (D-NeRF) | FPS | 한계 |
|------|------|-----------|---------------|-----|------|
| **D-NeRF** (Pumarola et al.) | CVPR 2021 | Deformation field + NeRF | ~29.7 | < 1 | 느린 학습/렌더링 |
| **Nerfies** (Park et al.) | ICCV 2021 | Elastic regularization + deformation | - | < 1 | 단안 동적 장면 한정 |
| **HyperNeRF** (Park et al.) | 2021 | 고차원 ambient space로 위상 변화 모델링 | - | < 1 | 매우 느린 학습(32시간) |
| **TiNeuVox** (Fang et al.) | SIGGRAPH Asia 2022 | 시간 인식 뉴럴 복셀 | 32.67 | 1.5 | 실시간 불가 |
| **K-Planes** (Fridovich-Keil et al.) | CVPR 2023 | 명시적 시공간 평면 분해 | 31.61 | 0.97 | 느린 렌더링 |
| **HexPlane** (Cao & Johnson) | CVPR 2023 | 6개 평면 기반 동적 장면 표현 | 31.04 | 2.5 | 실시간 불가 |
| **V4D** (Gan et al.) | TVCG 2023 | Voxel for 4D NVS | 33.72 | 2.08 | 6.9시간 학습, 느린 렌더링 |
| **MSTH** (Wang et al.) | NeurIPS 2023 | Masked space-time hash encoding | 31.34 | - | 실시간 자유 시점 불가 |

### 5.2 동적 Gaussian Splatting 계열

| 연구 | 연도 | 핵심 접근법 | 특징 | 4D-GS와의 비교 |
|------|------|-----------|------|--------------|
| **Dynamic3DGS** (Luiten et al.) | 3DV 2024 | 매 프레임 각 Gaussian 추적 | $O(tN)$ 메모리, 명시적 테이블 저장 | 메모리 비효율적 (장기 시퀀스에서 비용 선형 증가) |
| **4D Gaussian Splatting (Yang et al.)** | 2023 | 시간적 Gaussian 분포 추가 | 3D→4D Gaussian 확장 | 각 Gaussian이 국소 시간 공간에만 집중하는 한계 |
| **Deformable-3DGS** (Yang et al.) | 2023 | MLP 변형 네트워크 | 동시 연구(concurrent work) | 공간-시간 구조 인코더 없이 MLP만 사용 → 인접 Gaussian 간 연결성 부족 |
| **SpacetimeGS** (Li et al.) | 2023 | 각 Gaussian 개별 추적 | 시공간 Gaussian feature splatting | 개별 추적으로 인한 확장성 한계 |
| **4D-GS (본 논문)** | 2023 | HexPlane 인코더 + Multi-head 디코더 | $O(N+\mathcal{F})$ 메모리, 82 FPS | 인접 Gaussian 연결을 통한 일관된 변형 모델링 |

### 5.3 핵심 비교 분석

**렌더링 속도 측면**: 4D-GS는 동적 장면 렌더링에서 **최초로 실시간(≥30 FPS)을 달성**한 방법 중 하나이다. 기존 NeRF 기반 방법들이 1 FPS 미만이었던 것에 비해, 약 **30~80배의 속도 향상**을 이룩했다.

**품질 측면**: D-NeRF 합성 데이터셋에서 V4D(33.72 dB) 대비 +0.33 dB 향상된 34.05 dB를 달성하면서도, 학습 시간은 6.9시간에서 8분으로 **약 50배 단축**되었다.

**효율성 측면**: Dynamic3DGS의 $O(tN)$ 대비 $O(N + \mathcal{F})$의 메모리 복잡도로, 장기 시퀀스에서의 확장성이 현저히 우수하다.

**구조적 차별점**: 4D-GS의 HexPlane 기반 spatial-temporal encoder는 **인접 Gaussian 간의 공간-시간적 연결**을 구축하여, 개별 Gaussian을 독립적으로 처리하는 기존 방법들(Dynamic3DGS, SpacetimeGS)에 비해 **더 완전한 변형 기하학(deformed geometry)**을 학습하고 **분열(avulsion) 문제를 효과적으로 방지**한다.

---

## 참고자료

1. **Wu, G., Yi, T., Fang, J., et al.** "4D Gaussian Splatting for Real-Time Dynamic Scene Rendering." *arXiv preprint arXiv:2310.08528v3*, 2024. (본 논문)
2. **Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G.** "3D Gaussian Splatting for Real-Time Radiance Field Rendering." *ACM Transactions on Graphics (ToG)*, 42(4):1–14, 2023.
3. **Pumarola, A., Corona, E., Pons-Moll, G., & Moreno-Noguer, F.** "D-NeRF: Neural Radiance Fields for Dynamic Scenes." *CVPR*, 2021.
4. **Cao, A. & Johnson, J.** "HexPlane: A Fast Representation for Dynamic Scenes." *CVPR*, 2023.
5. **Fridovich-Keil, S., et al.** "K-Planes: Explicit Radiance Fields in Space, Time, and Appearance." *CVPR*, 2023.
6. **Fang, J., Yi, T., et al.** "Fast Dynamic Radiance Fields with Time-Aware Neural Voxels." *SIGGRAPH Asia*, 2022.
7. **Park, K., et al.** "HyperNeRF: A Higher-Dimensional Representation for Topologically Varying Neural Radiance Fields." *arXiv:2106.13228*, 2021.
8. **Luiten, J., Kopanas, G., Leibe, B., & Ramanan, D.** "Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis." *3DV*, 2024.
9. **Yang, Z., et al.** "Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction." *arXiv:2309.13101*, 2023.
10. **Yang, Z., et al.** "Real-Time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting." *arXiv:2310.10642*, 2023.
11. **Li, Z., et al.** "Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis." *arXiv:2312.16812*, 2023.
12. **Gan, W., et al.** "V4D: Voxel for 4D Novel View Synthesis." *IEEE TVCG*, 2023.
13. **Wang, F., et al.** "Masked Space-Time Hash Encoding for Efficient Dynamic Scene Reconstruction." *NeurIPS*, 2023.
14. **Mildenhall, B., et al.** "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis." *Communications of the ACM*, 65(1):99–106, 2021.
15. **Chen, G. & Wang, W.** "A Survey on 3D Gaussian Splatting." *arXiv:2401.03890*, 2024.
