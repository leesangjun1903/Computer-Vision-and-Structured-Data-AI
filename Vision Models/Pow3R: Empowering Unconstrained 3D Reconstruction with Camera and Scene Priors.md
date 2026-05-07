# Pow3R: Empowering Unconstrained 3D Reconstruction with Camera and Scene Priors

## 1. 핵심 주장과 주요 기여 요약

**Pow3R**는 DUSt3R 기반의 3D 비전 회귀 모델로, 입력 모달리티에 대해 **유연한 조건부 입력(versatile conditioning)** 을 제공하는 것이 핵심입니다.

**주요 기여:**

- **단일 모델로 다중 모달리티 지원**: RGB 이미지 외에 카메라 내부 파라미터(intrinsics) $K_1, K_2$, 상대 카메라 자세 $P_{1,2}$, 희소/조밀 깊이맵 $D_1, D_2$ 의 임의의 부분집합을 받을 수 있음
- **추가 포인트맵 $X^{2,2}$ 예측**: 두 번째 이미지의 자체 좌표계에서의 포인트맵을 동시에 예측하여 단일 forward pass로 상대 자세 추정 가능 (Procrustes 정렬을 통해 PnP+RANSAC 대비 약 **10배 빠름**)
- **고해상도 처리(native resolution)**: 카메라 내부 파라미터 입력을 통해 비중심(non-centered) 크롭을 처리할 수 있어 sliding-window 방식의 native 해상도 추론이 가능
- **다양한 다운스트림 태스크에서 SOTA**: 깊이 완성(depth completion), MVS, 다중뷰 깊이/자세 추정에서 기존 방법 대비 큰 폭 향상

---

## 2. 상세 설명

### 2.1 해결하고자 하는 문제

기존 DUSt3R/MASt3R는 RGB 이미지만을 입력으로 받기 때문에, 실제 응용에서 자주 가용한 **사전 정보(priors)** — 보정된 카메라 내부 파라미터, RGB-D 센서/LiDAR의 깊이, SfM에서 얻은 카메라 자세 등 — 를 활용할 방법이 없습니다. Pow3R는 이 한계를 다음과 같이 정의합니다:

> 학습 기반의 단일 모델로 희소 깊이, 카메라 내부/외부 파라미터의 **임의의 부분집합**을 가이드로 받을 수 있는 모델은 현재까지 존재하지 않음.

### 2.2 제안 방법 (수식 포함)

#### 2.2.1 Pointmap 표현

각 픽셀 $(i,j)$ 에 3D 점 $X_{i,j}$ 를 대응시키며, 이미지 $I^n$ 의 포인트맵을 카메라 $I^m$ 좌표계에서 표현한 것을 $X^{n,m}$ 이라 표기합니다:

$$X_{i,j} = K^{-1}\big[i\, D_{i,j},\ j\, D_{i,j},\ D_{i,j}\big]^\top$$

좌표계 변환:

$$X^{n,k} = P_{m,k}\, X^{n,m},\quad P_{m,k} = P_k P_m^{-1}$$

#### 2.2.2 인코더–디코더 구성

ViT 기반 Siamese 인코더는 이미지와 보조 정보 $\Omega_n \in \sigma(\{K^n, D^n\})$ 를 함께 처리합니다:

$$F^1 = \text{Encoder}(I^1, \Omega_1),\qquad F^2 = \text{Encoder}(I^2, \Omega_2)$$

두 디코더는 cross-attention으로 통신하며 상대 자세 $\Omega_D \in \sigma(\{P_{12}\})$ 를 i번째 블록에서 받을 수 있습니다:

$$G^1_i = \text{DecoderBlock}^1_i\big(G^1_{i-1}, G^2_{i-1}, \Omega_D\big)$$

$$G^2_i = \text{DecoderBlock}^2_i\big(G^2_{i-1}, G^1_{i-1}, \Omega_D\big)$$

최종 헤드는 세 개의 포인트맵을 회귀합니다:

$$X^{1,1}, C^{1,1} = \text{Head}^1(G^1_B)$$

$$X^{2,1}, X^{2,2}, C^{2,1}, C^{2,2} = \text{Head}^2(G^2_B)$$

#### 2.2.3 손실 함수

스케일 불변 회귀 손실:

$$\mathcal{L}^{\text{regr}}_{i,j}(n,m) = \left\|\frac{X^{n,m}_{i,j}}{z_m} - \frac{\hat{X}^{n,m}_{i,j}}{\hat{z}_m}\right\|$$

여기서 $z_m = \text{norm}(X)$ 는 유효 3D 점들의 평균 노름. 신뢰도 가중 손실:

$$\mathcal{L}^{\text{conf}}(n,m) = \sum_{i,j \in \mathcal{D}} C^{n,m}_{i,j}\, \mathcal{L}^{\text{regr}}_{i,j}(n,m) - \alpha \log C^{n,m}_{i,j}$$

전체 손실 ($\alpha=0.2,\ \beta=1$):

$$\mathcal{L} = \mathcal{L}^{\text{conf}}(1,1) + \mathcal{L}^{\text{conf}}(2,1) + \beta\, \mathcal{L}^{\text{conf}}(2,2)$$

#### 2.2.4 보조 정보 임베딩 (Inject 전략)

- **내부 파라미터(Intrinsics)**: RayDiffusion 방식을 따라 각 픽셀에 대해 $K^{-1}[i,j,1]^\top$ 의 ray로 변환 후 패치화하여 인코더에 주입
- **깊이맵/포인트클라우드**: 마스크 $M$ 과 함께 $D' = D/\text{norm}(D)$ 로 정규화한 뒤 $[D', M] \in \mathbb{R}^{W\times H \times 2}$ 를 패치화
- **상대 자세**: 변환의 스케일을 $t'\_{12} = t_{12}/\|t_{12}\|$ 로 정규화 후, 디코더의 CLS 토큰에 임베딩 추가

저자들은 첫 번째 transformer 블록에만 주입하는 **inject-1** 방식이 가장 효율적이라고 보고합니다 (전체 파라미터의 +4%만 추가).

#### 2.2.5 상대 자세 추정 (Procrustes 정렬)

$X^{2,2}$ 와 $X^{2,1}$ 사이의 강체 변환을 풀어 얻습니다:

```math
R^*, t^* = \arg\min_{\sigma, R, t} \sum_{i,j} \sqrt{C^{2,2}_{i,j} C^{2,1}_{i,j}}\, \big\|\sigma(R\, X^{2,2}_{i,j} + t) - X^{2,1}_{i,j}\big\|^2
```

### 2.3 모델 구조 (시각화)

핵심 아키텍처 요약:
- 공유 ViT 인코더 (옵션: rays/depth 토큰 주입)
- 두 개의 cross-attention 디코더 (옵션: pose 토큰을 CLS에 추가)
- DUSt3R는 DPT 헤드를 사용했으나 Pow3R는 **선형(linear) 헤드**로 충분 (효율성 향상)

### 2.4 성능 및 향상

| 태스크 | 데이터셋 | 주요 결과 |
|---|---|---|
| Multi-view depth | KITTI/ScanNet/ETH3D/DTU/T&T 평균 | rel = 3.18 (DUSt3R: 3.73), $\tau$ = 73.3 (DUSt3R: 68.19) |
| MVS | DTU | Overall = 1.115mm (DUSt3R: 1.741mm) — **36% 상대 개선** |
| Pose estimation | Co3Dv2 | mAA(30) = 82.2 (DUSt3R: 77.2), 약 **10배 더 빠름** |
| Depth completion | NYUv2 (zero-shot) | LRRU/CompletionFormer 등 SOTA보다 우수 |

특히 표 1에서 모든 보조 정보를 제공했을 때 focal acc@1.015는 **39.4 → 99.3%**, RTA@2°는 **53.8 → 98.1%** 로 극적 향상을 보입니다.

### 2.5 한계

- **학습 파이프라인은 DUSt3R와 동일**한 8개 데이터셋 (8.5M 페어)에 의존 — 새로운 도메인에는 추가 학습 필요할 수 있음
- **Pairwise 처리에 머무름**: 한 번에 두 장의 이미지만 처리, $N$ 개 이미지 시 $O(N^2)$ pairwise 추론과 global alignment 필요
- **동적 장면 미지원**: 정적 장면 가정 (이는 후속 MonST3R가 다룸)
- **노이즈에 대한 견고성은 정성적으로만 평가** — 그림 6의 controllability 분석은 흥미롭지만 실제 노이즈 환경(예: 실측 LiDAR depth)에서의 정량적 평가는 제한적

---

## 3. 일반화 성능 향상 가능성

Pow3R는 일반화 측면에서 흥미로운 특성을 보입니다:

### 3.1 보조 정보를 통한 일반화

- **Zero-shot 깊이 완성**: NYUv2와 KITTI는 학습 데이터에 포함되지 않았음에도, 희소 깊이를 가이드로 제공하면 LRRU/CompletionFormer 같은 전용 모델을 능가합니다 (그림 5). 이는 보조 정보가 **도메인 갭을 메우는 다리** 역할을 한다는 의미입니다.
- **Infinigen에서의 일관된 트렌드**: 학습에 포함되지 않은 InfiniGen 데이터셋(부록 표 A)에서도 Habitat과 동일한 향상 패턴을 보여 — 이는 보조 정보 활용 메커니즘이 데이터 도메인에 종속되지 않음을 시사합니다.

### 3.2 Native 해상도 추론

학습 시 비중심 크롭과 함께 intrinsics를 주입했기 때문에, 학습 시 보지 못한 해상도/종횡비(aspect ratio)에서도 sliding-window 방식으로 처리할 수 있습니다. 표 2에 따르면 KITTI(370×1226의 비전형적 해상도)에서 native sliding-window 방식이 baseline 대비 rel을 5.3 → 4.6으로 개선합니다.

### 3.3 Controllability — "정보를 맹신하지 않는" 모델

그림 6에서 가장 흥미로운 발견은, 잘못된 focal/pose를 입력해도 모델이 **GT에서 너무 벗어나면 가이던스를 거부**한다는 점입니다. 동시에 신뢰도(confidence)는 가이던스와 GT의 일치 정도와 명확히 상관됩니다. 이는:
- 모델이 **단순 패스스루(pass-through)가 아닌 RGB 증거와 보조 정보를 결합**해 추론한다는 증거이며,
- 실제 환경에서 노이즈가 있는 센서 정보(예: 약간 부정확한 IMU pose, sparse LiDAR)에 대해 견고할 가능성이 높음을 시사합니다.

### 3.4 일반화 가능성의 한계

- 학습은 224px → 512px의 2단계로 진행되며, 8.5M 페어로는 VGGT(많은 데이터+더 큰 모델로 학습)에 비해 데이터/스케일이 제한적
- Dynamic scene(움직이는 객체), 비-사진학적 도메인(의료, 위성 등)에서의 일반화는 아직 검증되지 않음

---

## 4. 미래 연구 영향 및 고려사항

### 4.1 미치는 영향

1. **"옵셔널 가이던스(optional guidance)" 패러다임의 정착**: 단일 feed-forward 모델이 RGB만으로도, 또는 임의의 보조 정보로도 동작한다는 설계 원칙은 후속 3D 파운데이션 모델 설계의 표준이 될 가능성이 큼.
2. **로보틱스/AR/VR 응용에 직접 활용 가능**: IMU, 캘리브레이션된 카메라, RGB-D 센서 등이 표준 장비인 환경에서 실용성이 매우 큼.
3. **High-resolution 추론의 새 길**: 학습은 저해상도에서, 추론은 native 해상도에서 — 이는 Win-Win 등 기존 접근과 다른 새로운 방향성.
4. **단일 forward pose 추정**: $X^{2,2}$ 동시 예측 + Procrustes는 SLAM/SfM 파이프라인에 직접 통합하기에 매력적.

### 4.2 향후 연구 시 고려할 점

- **N-view 확장**: Pow3R는 여전히 pairwise이므로, Fast3R/VGGT처럼 다중 뷰 동시 처리로의 확장이 자연스러운 다음 단계
- **노이즈 모델링**: 보조 정보에 인위적 노이즈를 주입하여 학습하면 실제 센서 환경에서의 견고성이 개선될 가능성
- **동적 장면 + 가이던스**: MonST3R와 같이 시간적 변화가 있는 장면에서 보조 정보 활용
- **자기지도 학습/펄스 학습**: GT 깊이/자세가 없는 도메인에서 부분적인 보조 정보(예: SfM 출력)로 학습할 수 있는지
- **메모리/효율성**: 대규모 장면(수백~수천 이미지)에서는 pairwise + global alignment의 $O(N^2)$ 비용이 문제. 메모리 기반 또는 streaming 접근의 통합 필요
- **불확실성 정량화**: 그림 6의 controllability를 보다 정량적으로 측정하고, 베이지안 prior로 명시적으로 모델링할 가치

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 모델 (연도) | 입력 형태 | 다중 뷰 처리 | 보조 정보 입력 | 핵심 차별점 |
|---|---|---|---|---|
| **DUSt3R** (CVPR 2024) | RGB pair | ❌ pairwise + global alignment | ❌ 없음 | Pointmap 회귀 패러다임 도입 |
| **MASt3R** (ECCV 2024) | RGB pair | ❌ pairwise | ❌ 없음 | Metric pointmap + dense matching head 추가 |
| **Spann3R** (3DV 2025) | RGB sequence | ⭕ 순차적 + spatial memory | ❌ 없음 | 글로벌 좌표계에서 직접 예측, 메모리 모듈 |
| **MonST3R** (ICLR 2025) | RGB sequence | ⭕ video | ❌ 없음 | 동적 장면 처리 |
| **Fast3R** (CVPR 2025) | RGB N-view | ⭕ all-to-all attention, 1000+ 이미지 | ❌ 없음 | $O(N^2)$ 제거, 단일 forward로 다중 뷰 |
| **VGGT** (CVPR 2025 Best Paper) | RGB N-view | ⭕ feed-forward, 수백 뷰 | ❌ 없음 | 1.2B 파라미터, 카메라+depth+pointmap+track 통합 출력 |
| **Pow3R** (2025) | RGB pair + 임의 prior | ❌ pairwise | ⭕ K, RT, sparse depth | **유일하게 옵셔널 prior 입력 지원** |

### 비교 분석

**Spann3R, MonST3R, Fast3R, VGGT** 는 모두 "다중 뷰 확장"이라는 축에서 DUSt3R를 발전시켰습니다. 특히 VGGT는 단일 forward pass로 카메라 파라미터, 깊이맵, 포인트맵, 3D 포인트 트랙까지 한꺼번에 출력하며 BA(Bundle Adjustment) 같은 후처리를 거의 제거했습니다.

**Pow3R는 다른 축 — "옵셔널 prior 입력"** — 을 개척한 점에서 독창적입니다. 이는 상호보완적인 방향이며, 향후 두 축을 결합한 모델(예: VGGT + Pow3R-style conditioning)이 자연스러운 다음 연구 방향입니다. Naver Labs 자체 페이지에서도 "DUSt3R, MASt3R, Pow3R는 모두 페어 단위 처리에 머무르며, 페어 수가 quadratic하게 증가한다"는 한계를 명시하고 있습니다.

연구 관점에서 흥미로운 점은, Fast3R가 Train-Short/Test-Long 전략으로 20 view 학습 → 1000+ view 추론을 달성한 것처럼, Pow3R도 **"Train-with-priors / Test-with-or-without-priors"** 전략으로 학습 시 prior dropout을 통해 견고한 prior 활용 능력을 갖춘 것이 본질적으로 같은 철학이라는 점입니다.

---

## 참고 자료 (출처)

1. **원 논문**: Jang, Weinzaepfel, Leroy, Agapito, Revaud. "Pow3R: Empowering Unconstrained 3D Reconstruction with Camera and Scene Priors", arXiv:2503.17316v1, 2025년 3월. (사용자 업로드 PDF)

2. **NAVER Labs Europe — 3D Foundation Models**: https://europe.naverlabs.com/research/3d-foundation-models/ (DUSt3R/MASt3R/Pow3R 공식 소개 및 비교)

3. **NAVER Labs Europe — MASt3R 블로그**: https://europe.naverlabs.com/blog/mast3r-matching-and-stereo-3d-reconstruction/

4. **VGGT 공식 사이트**: https://vgg-t.github.io/ (Wang et al., CVPR 2025 Best Paper)

5. **VGGT 논문**: Wang et al., "VGGT: Visual Geometry Grounded Transformer", arXiv:2503.11651, CVPR 2025.

6. **Fast3R 공식 사이트 / 논문**: https://fast3r-3d.github.io/ ; Yang et al., arXiv:2501.13928, CVPR 2025.

7. **Awesome-DUSt3R 큐레이션 리포지토리**: https://github.com/ruili3/awesome-dust3r (DUSt3R 후속 연구 목록)

8. **LearnOpenCV — MASt3R & MASt3R-SfM 개관**: https://learnopencv.com/mast3r-sfm-grounding-image-matching-3d/

9. **MVS 비교 서베이**: "A Comparison of Multi-View Stereo Methods for Photogrammetric 3D Reconstruction" (DUSt3R/MASt3R/VGGT/Fast3R 정량 비교 포함), arXiv 게시.

---

**주의사항**: 이 분석에서 Pow3R 자체에 대한 모든 수식·표·정량 수치는 사용자가 업로드한 원 논문 PDF에서 직접 인용했습니다. 다른 모델(VGGT, Fast3R 등)과의 비교는 각 논문/공식 페이지의 보고된 수치에 기반하며, 동일 벤치마크에서의 직접 head-to-head 비교는 일부 항목에 한정됩니다 (Pow3R 논문은 동시기 발표된 VGGT/Fast3R와는 직접 비교하지 않음). 따라서 표 5(다중뷰 자세 추정)의 SOTA는 Pow3R 발표 시점 기준이며, 현재(2026년 5월) 시점에서는 VGGT 등이 일부 지표에서 추가 향상을 달성한 상태입니다.
