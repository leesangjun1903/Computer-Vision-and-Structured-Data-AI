# GaussianPro: 3D Gaussian Splatting with Progressive Propagation

---

## 1. 핵심 주장 및 주요 기여 (요약)

3D Gaussian Splatting(3DGS)은 뉴럴 렌더링 분야에 혁명을 가져왔으나, Structure-from-Motion(SfM)으로 생성된 초기 포인트 클라우드에 크게 의존한다. 대규모 장면에서 텍스처 없는 표면이 존재할 경우, SfM은 충분한 포인트를 생성하지 못하여 3DGS의 최적화가 어렵고 렌더링 품질이 저하된다.

이를 해결하기 위해 저자들은 고전적 Multi-View Stereo(MVS) 기법에서 영감을 받아 **progressive propagation strategy**를 적용하여 3D Gaussian의 densification을 안내하는 **GaussianPro**를 제안했다. 기존 3DGS의 단순한 split/clone 전략과 달리, 이미 재구성된 기하 구조의 사전 정보(priors)와 패치 매칭(patch matching) 기법을 활용하여 정확한 위치와 방향을 가진 새로운 Gaussian을 생성한다.

**주요 기여:**
1. **Progressive propagation 기반 densification 전략** — MVS의 패치 매칭을 3DGS 훈련 루프에 통합
2. **Planar constraint loss** 도입 — Gaussian이 실제 표면 기하에 충실하도록 정규화
3. 대규모 및 소규모 장면 모두에서 유효성을 검증하였으며, Waymo 데이터셋에서 PSNR 기준 **1.15dB**의 향상을 달성하였다.

> **발표:** ICML 2024에 게재되었으며, 저자는 Kai Cheng, Xiaoxiao Long, Kaizhi Yang, Yao Yao, Wei Yin, Yuexin Ma, Wenping Wang, Xuejin Chen이다.

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

3DGS의 희소한 SfM 포인트와 제약이 부족한 densification 전략은 특히 텍스처 없는 영역에서 3D Gaussian 최적화에 어려움을 야기한다. 3DGS는 훈련 이미지에 과적합(overfit)되는 잘못된 Gaussian을 생성하여, 새로운 시점(novel view) 렌더링에서 뚜렷한 성능 저하와 기하 오류를 일으킨다.

기존 3DGS에서는 최적화가 기하학적 제약 없이 이미지 복원 손실만을 기반으로 수행되어, 최적화된 Gaussian의 형상이 실제 표면 기하로부터 크게 벗어날 수 있다. 이러한 편차는 특히 제한된 뷰의 대규모 장면에서 새로운 시점의 렌더링 품질 저하로 이어진다.

### 2.2 제안하는 방법 (수식 포함)

#### (A) 하이브리드 기하 표현 (Hybrid Geometric Representation)

GaussianPro는 3D Gaussian을 2D 뷰 의존적 깊이(depth) 맵과 법선(normal) 맵으로 매핑하는 하이브리드 기하 표현을 결합한다. 각 입력 이미지에 대해 기존 3D Gaussian으로부터 depth map과 normal map을 렌더링하며, 각 Gaussian의 깊이는 중심을 카메라 좌표계로 투영하여 계산한다. 각 Gaussian의 법선은 공분산 행렬(covariance matrix)의 최단 축(shortest axis) 방향으로 근사된다.

각 Gaussian $G_i$의 깊이와 법선:

$$z_i = (\mathbf{R} \cdot \boldsymbol{\mu}_i + \mathbf{t})_z$$

여기서 $\mathbf{R}$, $\mathbf{t}$는 카메라의 외부 파라미터, $\boldsymbol{\mu}_i$는 Gaussian의 중심 좌표이다.

법선 벡터 $\mathbf{n}_i$는 공분산 행렬 $\Sigma_i$의 고유값 분해에서 가장 작은 고유값에 대응하는 고유벡터로 정의된다:

$$\mathbf{n}_i = \text{eigvec}_{\min}(\Sigma_i)$$

#### (B) 평면 정의 (Plane Definition)

렌더링된 맵에서 각 픽셀의 depth, normal 정보를 3D 로컬 평면으로 변환한다. 이 평면은 normal 벡터와 카메라 원점으로부터 평면까지의 거리로 정의된다.

거리 $d$를 계산하는 공식:

$$d = z \cdot \mathbf{n}^T \mathbf{K}^{-1} \mathbf{e}_p$$

여기서:
- $z$: 픽셀의 렌더링된 깊이
- $\mathbf{n}$: 픽셀의 렌더링된 법선
- $\mathbf{K}$: 카메라 내부 파라미터 행렬 (intrinsic matrix)
- $\mathbf{e}_p$: 픽셀 $p$의 동차 좌표 (homogeneous coordinate)

#### (C) Progressive Propagation (점진적 전파)

먼저 3D Gaussian으로부터 depth, normal 맵을 렌더링한 후, 렌더링된 depth와 normal에 대해 반복적으로 전파(propagation) 연산을 수행하여 패치 매칭 기법을 통해 새로운 depth, normal 값(propagated depth/normal)을 생성한다.

각 픽셀에 대해 체커보드 패턴으로 이웃 픽셀을 선택하고, 이웃 픽셀에 정의된 평면을 현재 픽셀의 후보(candidate)로 간주하여 정보를 전파한다. 각 평면 후보에 대해 homography 변환 $\mathbf{H}$를 계산한다.

Homography 변환:

$$\mathbf{H} = \mathbf{K}_j \left( \mathbf{R}_{j} \mathbf{R}_{i}^{-1} + \frac{(\mathbf{t}_{j} - \mathbf{R}_{j}\mathbf{R}_{i}^{-1}\mathbf{t}_{i}) \cdot \mathbf{n}^T \mathbf{K}_i^{-1}}{d} \right)$$

여기서 $i$는 참조 뷰, $j$는 소스 뷰이며, 패치 매칭을 통해 multi-view photo-consistency를 평가한다.

#### (D) Planar Constraint Loss (평면 제약 손실)

전파된 2D normal 맵은 장면의 평면 방향을 나타낸다. Gaussian의 렌더링된 normal과 전파된 normal 간의 일관성을 명시적으로 강제한다.

$$\mathcal{L}_{\text{planar}} = \sum_{p} \left\| \mathbf{n}_{\text{rendered}}(p) - \mathbf{n}_{\text{propagated}}(p) \right\|_1$$

전파된 normal 맵은 planar constraint loss 계산에 저장되며, 최종 훈련 손실은 이미지 복원 손실(L1 및 D-SSIM)과 제안된 planar constraint loss의 결합이다.

최종 손실 함수:

$$\mathcal{L} = (1-\lambda_{\text{ssim}})\mathcal{L}_1 + \lambda_{\text{ssim}} \mathcal{L}_{\text{D-SSIM}} + \lambda_{\text{planar}} \mathcal{L}_{\text{planar}}$$

#### (E) 새로운 Gaussian 생성

전파된 depth로 선택된 픽셀을 3D 공간으로 역투영(back-projection)하여, 기존 모델이 부정확한 영역에 새로운 Gaussian을 초기화한다.

역투영 공식:

$$\mathbf{X} = z_{\text{propagated}} \cdot \mathbf{R}^{-1} (\mathbf{K}^{-1} \mathbf{e}_p - \mathbf{t})$$

### 2.3 모델 구조

progressive Gaussian propagation 전략은 3DGS 훈련 루프에 통합되어, 매 $m$ 반복마다 활성화된다.

전체 파이프라인은 다음과 같다:

```
[SfM 초기화] → [3DGS 훈련 시작]
    ↓
매 m iteration마다:
    (1) Depth/Normal Map 렌더링
    (2) Patch Matching을 통한 Progressive Propagation
    (3) Propagated Depth로 새 Gaussian 생성 (Back-projection)
    (4) Planar Constraint Loss 계산
    ↓
[L1 + D-SSIM + Planar Loss로 최적화]
    ↓
[기존 3DGS split/clone 전략과 병행]
```

이 프로젝트는 3D Gaussian Splatting과 ACMH/ACMM(고전적 MVS 패치 매칭 라이브러리)을 주요 참조로 삼고 있다.

### 2.4 성능 향상

| 데이터셋 | 주요 결과 |
|---------|---------|
| **Waymo** | 기존 split-and-clone 전략 대비 PSNR 1.15dB 향상, 텍스처 약한 영역의 기하 완전성과 렌더링 충실도 대폭 개선 |
| **MipNeRF360** | 소규모 텍스처 풍부한 장면에서 3DGS 대비 유사하거나 약간의 개선을 보임 |
| **FPS 비교** | 거리 장면에서 GaussianPro는 108 FPS, 3DGS는 119 FPS; 방 장면에서 GaussianPro는 113 FPS, 3DGS는 105 FPS로, 실시간 렌더링 수준을 유지 |

### 2.5 한계

GaussianPro는 동적 객체를 별도로 모델링하지 않으며, 모든 정적 Gaussian 방법과 마찬가지로 이러한 영역에서 아티팩트가 발생한다. 향후 최근의 동적 Gaussian 기술을 보완 요소로 통합할 수 있을 것이다.

현재 버전은 비정렬(unordered) 이미지 세트를 지원하지 않으며, 시간 순서로 정렬된 비디오 데이터가 필요하다.

SfM 초기 포인트 클라우드가 이미 높은 품질이고 텍스처가 풍부한 장면에서는 GaussianPro의 기하 정교화 효과가 덜 두드러진다.

---

## 3. 모델의 일반화 성능 향상 가능성

GaussianPro의 일반화 성능에 관한 핵심 논의:

### 3.1 텍스처 없는 영역에서의 일반화

GaussianPro는 progressive propagation 전략을 제안하여, 법선 일관성 제약(normal consistency constraints)과 평면 사전 정보(planar priors)를 공동으로 시행하는 반복 업데이트 메커니즘을 통해 3D Gaussian 포인트 클라우드의 적응적 밀도화를 주도한다.

이는 기존 3DGS가 실패하던 **도로, 벽, 건물 외벽** 등 텍스처 약한 표면에서의 일반화를 크게 향상시킨다.

### 3.2 장면 규모(Scale)에 대한 일반화

GaussianPro는 Waymo 및 MipNeRF360 데이터셋 모두에서 3DGS 대비 우수한 렌더링 결과를 보이며, 구조적 장면에서 유의미한 개선을 보이고 훈련 이미지 수의 변동에 대해 강건하다.

### 3.3 외부 사전 정보 없이 동작

GaussianPro에서는 외부 사전 정보(prior)를 사용하지 않는다. 이는 사전 학습된 깊이 추정 네트워크 등에 의존하는 다른 방법들과 차별화되며, **도메인 간 전이(domain transfer) 시 외부 모델의 일반화 한계에 영향받지 않는 장점**이 있다.

### 3.4 일반화 향상을 위한 향후 방향

1. **동적 장면 통합**: 동적 Gaussian 기법(4D Gaussian Splatting 등)과의 결합을 통해 동적 객체가 포함된 실세계 장면으로 일반화 확장
2. **비정렬 이미지 지원**: 현재 시간 순서 이미지만 지원하는 제약을 해결하면 더 다양한 캡처 환경으로 일반화 가능
3. **Feed-forward 일반화**: 제한된 뷰 수에서 정규화 기법의 효용이 감소하므로, 학습된 사전 정보를 활용하는 일반화 기반 방법(feed-forward Gaussian model)이 연구되고 있다.

---

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구 영향

1. **MVS-3DGS 융합 패러다임 확립**: GaussianPro는 고전적 MVS 기법(patch matching)과 현대적 3DGS를 결합하는 새로운 패러다임을 제시하였으며, 이후 여러 연구가 이를 계승하고 있다:
   - GausSurf는 패치 매칭 알고리즘을 실행하여 GausSurf가 생산한 깊이 값과 법선 맵을 정교화하고, 이를 통해 다중 뷰 이미지를 매칭하여 Gaussian 최적화 중 표면 위치를 정확히 찾는다.
   - PGSR은 고충실도 표면 재구성을 달성하면서 고품질 렌더링을 보장하는 평면 기반 Gaussian Splatting 표현을 제안한다.

2. **기하학적 정규화(Geometric Regularization)의 중요성 강조**: GaussianPro는 최적화된 depth/normal 맵을 활용하여 SfM으로 초기화된 영역의 갭을 채우는 방식으로 densification을 안내하며, 이러한 개선은 기존의 density control 위에 적용될 수 있다.

3. **대규모 실외 장면 연구 촉진**: Waymo 자율주행 데이터셋에서의 우수한 성능은 자율주행, 도시 매핑 등 산업 응용에서 3DGS 활용 가능성을 넓혔다.

### 4.2 향후 연구 시 고려할 점

| 고려사항 | 설명 |
|---------|------|
| **동적 장면** | 동적 객체 처리를 위한 temporal Gaussian 기법과의 통합 필요 |
| **메모리 효율성** | 대규모 시나리오에서 메모리 소비는 여전히 도전과제이며, 향후 경량 모델 설계, 효율적 포인트 클라우드 전처리, 동적 메모리 관리에 초점을 맞추어야 한다. |
| **깊이 모델 선택** | 입력 해상도와 깊이 모델 선택 간에 비자명적 트레이드오프가 존재하며, 공격적인 다운샘플링은 PSNR/SSIM을 부풀리지만 고주파 디테일 손실을 야기한다. |
| **비정렬 데이터** | 실세계 응용에서는 시간 순서가 보장되지 않는 이미지 세트 처리가 필수적 |
| **텍스처 풍부한 장면** | 이미 SfM이 잘 동작하는 장면에서는 추가 오버헤드 대비 이점이 제한적 |

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 접근법 | GaussianPro와의 관계 |
|------|------|-----------|---------------------|
| **NeRF** (Mildenhall et al.) | 2020 | 암시적 뉴럴 표현, 볼륨 렌더링 | 3DGS/GaussianPro의 전신; 렌더링 속도가 느림 |
| **3DGS** (Kerbl et al.) | 2023 | SfM 희소 포인트 기반 3D Gaussian, 이방성 공분산 최적화, 타일 기반 래스터라이저로 실시간 렌더링 | GaussianPro의 기반(baseline) |
| **2DGS** (Huang et al.) | 2024 | 3D 볼륨을 2D 평면형 Gaussian 디스크로 축소하여 뷰 일관적 기하 달성 | 평면 표현의 대안적 접근 |
| **GaussianPro** (Cheng et al.) | 2024 | Progressive propagation + Patch matching + Planar loss | **본 논문** |
| **PGSR** (2024) | 2024 | 비편향(unbiased) 깊이 렌더링, 평면 기반 Gaussian Splatting으로 고충실도 표면 재구성 | 평면 제약을 더욱 발전시킨 방법 |
| **GausSurf** (2024) | 2024 | Gaussian 최적화와 패치 매칭 정교화를 반복적으로 수행하여 상호 강화 | GaussianPro의 MVS 통합 아이디어를 계승·확장 |
| **Pixel-GS** (Zhang et al.) | 2024 | 픽셀 인식 그래디언트를 활용한 density control | Densification 전략의 다른 관점 |
| **D2GS** | 2024 | Depth-Anything V2를 활용한 깊이·밀도 기반 안정적 희소 뷰 재구성 | 외부 depth prior 활용 (GaussianPro는 미사용) |
| **FreGS** (Zhang et al.) | 2024 | 점진적 주파수 정규화를 통한 3D Gaussian Splatting | 주파수 도메인에서의 정규화 접근 |
| **RSGaussian** | 2024 | 패치 매칭으로 normal 맵 생성 + LiDAR 포인트에서 생성된 true depth/planar 맵에 Gaussian을 근접시키는 정규화 손실 | GaussianPro + LiDAR 확장 |
| **Compact3D** (Lee et al.) | 2024 | 벡터 양자화를 통한 Gaussian 압축 | 효율성 측면의 보완적 연구 |
| **Mip-Splatting** | 2024 | 3D 평활 필터 + 2D 밉맵 필터로 스케일링 시 앨리어싱 제거 | 안티앨리어싱 관점의 보완적 접근 |

### 핵심 비교 수식 — 3DGS vs. GaussianPro의 Densification

**기존 3DGS Densification:**

3DGS는 렌더링 손실에서 역전파된 그래디언트가 임계값을 초과하면 해당 3D 영역이 충분히 표현되지 않았다고 판단한다. Gaussian의 공분산이 크면 2개로 분할(split)하고, 작으면 복제(clone)한다.

$$\text{If } \|\nabla_{\boldsymbol{\mu}} \mathcal{L}\| > \tau : \begin{cases} \text{Split}(G_i) & \text{if } \max(\text{eigval}(\Sigma_i)) > \sigma_{\text{thresh}} \\ \text{Clone}(G_i) & \text{otherwise} \end{cases}$$

**GaussianPro의 추가적 Densification:**

$$\text{Propagation: } (\hat{d}_p, \hat{\mathbf{n}}_p) = \underset{(d_q, \mathbf{n}_q) \in \mathcal{N}(p)}{\arg\max} \; S_{\text{NCC}}(p, q; \mathbf{H}(d_q, \mathbf{n}_q))$$

여기서 $S_{\text{NCC}}$는 Normalized Cross-Correlation 패치 매칭 스코어이며, 전파된 depth-normal 쌍 중 가장 높은 multi-view photo-consistency를 갖는 것이 채택된다.

---

## 참고자료 및 출처

1. **GaussianPro 공식 프로젝트 페이지**: https://kcheng1021.github.io/gaussianpro.github.io/
2. **arXiv 원문**: Cheng et al., "GaussianPro: 3D Gaussian Splatting with Progressive Propagation," arXiv:2402.14650, 2024. (https://arxiv.org/abs/2402.14650)
3. **ICML 2024 Proceedings**: https://proceedings.mlr.press/v235/cheng24f.html
4. **GitHub 공식 코드**: https://github.com/kcheng1021/GaussianPro
5. **ACM Digital Library**: https://dl.acm.org/doi/abs/10.5555/3692070.3692390
6. **Moonlight Literature Review**: https://www.themoonlight.io/en/review/gaussianpro-3d-gaussian-splatting-with-progressive-propagation
7. **Liner Quick Review**: https://liner.com/review/gaussianpro-3d-gaussian-splatting-with-progressive-propagation
8. **arXiv HTML (Full Paper)**: https://arxiv.org/html/2402.14650
9. **PMC — Enhanced 3DGS via Depth Priors**: https://pmc.ncbi.nlm.nih.gov/articles/PMC12656154/
10. **RSGaussian (arXiv)**: https://arxiv.org/pdf/2412.18380
11. **GausSurf (arXiv)**: https://arxiv.org/html/2411.19454v1
12. **PGSR (arXiv)**: https://arxiv.org/html/2406.06521v1
13. **3DGS 원본 논문 (Kerbl et al., 2023)**: https://github.com/graphdeco-inria/gaussian-splatting
14. **Recent Advances in 3DGS (Springer)**: https://link.springer.com/article/10.1007/s41095-024-0436-y
15. **3DGS Survey — Technologies, Challenges**: https://www.tianyuding.com/papers/3DGS-survey.pdf
16. **A Survey on 3DGS (Chen & Wang, 2024)**: https://arxiv.org/html/2401.03890v8
17. **Wikipedia — Gaussian Splatting**: https://en.wikipedia.org/wiki/Gaussian_splatting

---

> **참고**: 본 분석에서 수식은 논문의 원문과 공개된 리뷰·해설에 기반하여 재구성한 것입니다. 특정 하이퍼파라미터 값($\lambda_{\text{ssim}}$, $\lambda_{\text{planar}}$, $m$ 등)의 정확한 수치는 원 논문 및 공식 코드를 직접 확인하시기 바랍니다.
