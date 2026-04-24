
# Global Structure-from-Motion Revisited

> **논문 정보**
> - **제목:** Global Structure-from-Motion Revisited
> - **저자:** Linfei Pan, Dániel Baráth, Marc Pollefeys, Johannes L. Schönberger (ETH Zürich / Microsoft)
> - **학회:** ECCV 2024 (European Conference on Computer Vision, 2024)
> - **arXiv:** [arXiv:2407.20219](https://arxiv.org/abs/2407.20219)
> - **프로젝트 페이지:** https://lpanaf.github.io/eccv24_glomap/
> - **공식 코드:** https://github.com/colmap/glomap

---

## 1. 📌 핵심 주장 및 주요 기여 요약

SfM 문제는 크게 두 가지 패러다임으로 나뉜다: **점진적(incremental) 방식**과 **전역적(global) 방식**. 지금까지 대부분의 최신 시스템은 정확도와 강인성이 뛰어난 점진적 방식을 따랐고, 전역적 방식은 훨씬 더 확장 가능하고 효율적이지만 정확도 면에서 뒤처졌다.

이 연구에서는 global SfM 문제를 재검토하고, **GLOMAP**이라는 새로운 범용 시스템을 제안한다. 정확도와 강인성 면에서 가장 널리 사용되는 점진적 SfM인 COLMAP과 동등하거나 그 이상의 결과를 달성하면서도, **수십 배~수백 배 빠른** 처리 속도를 실현한다.

### 핵심 통찰:

저자들은 문제의 핵심이 **최적화 과정에서 3D 포인트를 활용하는 것**에 있다고 결론지었다. 이에 따라 ill-posed 문제인 translation averaging으로 카메라 위치를 추정하고 point triangulation으로 3D 구조를 별도로 구하는 방식 대신, 이 두 단계를 **단일 전역 위치 추정(global positioning) 스텝**으로 통합했다.

---

## 2. 🔍 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능, 한계

### 2-1. 해결하고자 하는 문제

#### ① 기존 Global SfM의 Translation Averaging 취약성

정확도·강인성의 gap은 주로 **global translation averaging 단계**에서 비롯된다. Translation averaging은 view graph에서 상대적 포즈로부터 전역 카메라 위치를 추정하는 문제이며, 실제로 세 가지 주요 도전 과제가 있다. 첫째는 **스케일 모호성(scale ambiguity)**: 추정된 two-view geometry로부터 얻는 상대적 translation은 스케일 결정이 불가능하며, 전역 카메라 위치를 정확히 추정하기 위해서는 상대 방향의 삼중항(triplet)이 필요하다.

기존 global 방식의 핵심 문제는 **스케일 모호성, 카메라 내부 파라미터에 대한 의존성, 그리고 거의 선형에 가까운 카메라 움직임(co-linear motion)**에 의한 degenerate case이다.

#### ② Incremental SfM의 확장성 문제

점진적 방식은 절대 카메라 포즈 추정, triangulation, bundle adjustment를 반복적으로 교차 수행하는데, 높은 정확도와 강인성을 달성하지만 반복적인 bundle adjustment의 비용으로 인해 확장성이 크게 제한된다.

---

### 2-2. 제안 방법: GLOMAP 파이프라인

GLOMAP은 다른 global 방식들과 구별되는 점으로, **translation averaging과 triangulation 단계를 단일 global positioning 스텝으로 통합**한다는 점이 핵심이다.

#### GLOMAP 전체 파이프라인 (9단계)

전체 Global Mapping Pipeline은 아홉 단계로 구성된다.

| 단계 | 설명 |
|------|------|
| 1 | View graph 준비 및 상대 포즈 분해 |
| 2 | 카메라 내부 파라미터(intrinsics) 보정 |
| 3 | 이미지 쌍 간 상대 포즈 추정 |
| 4 | 전역 일관 카메라 회전 추정 (Rotation Averaging) |
| 5 | Feature track을 통한 3D 포인트 초기 triangulation |
| 6 | **전역 카메라 위치 및 3D 포인트 동시 추정 (Global Positioning)** |
| 7 | Bundle Adjustment를 통한 정제 |
| 8 | 추가 3D 포인트 복원 및 refinement |
| 9 | 약하게 연결된 이미지 정리 |

---

#### Step 1: 강인한 Feature Track 구성

Feature track 구성 단계에서는 two-view geometry 검증을 통해 얻은 inlier correspondence만을 사용하며, outlier 매칭을 제거하고 삼각측량 각도(triangulation angle)로 인한 singularity를 방지하는 robust pipeline을 적용한다.

---

#### Step 2: Rotation Averaging

전역 카메라 회전 $\{R_i\}$는 모든 상대 회전 $\{R_{ij}\}$로부터 동시에 추정된다:

$$\min_{\{R_i\}} \sum_{(i,j) \in \mathcal{E}} d(R_i^T R_j,\ \hat{R}_{ij})^2$$

여기서 $d(\cdot)$는 $SO(3)$ 위에서의 거리 함수(예: geodesic distance), $\hat{R}_{ij}$는 two-view geometry에서 추정된 상대 회전이다.

---

#### Step 3: Global Positioning (핵심 기여)

Translation averaging 후 global triangulation을 순차적으로 수행하는 대신, **joint global triangulation과 카메라 위치 추정**을 직접 동시에 수행한다. 기존 대부분의 연구와 달리, 이 목적 함수는 **초기값 없이도(initialization-free)** 실제로 일관되게 좋은 해로 수렴한다.

**Global Positioning 목적 함수:**

카메라 위치 $\mathbf{c}_i \in \mathbb{R}^3$와 3D 포인트 $\mathbf{X}_k \in \mathbb{R}^3$를 동시에 최적화:

$$\min_{\{\mathbf{c}_i\},\ \{\mathbf{X}_k\}} \sum_{(i,k) \in \mathcal{V}} \left\| \hat{\mathbf{d}}_{ik} \times \frac{R_i(\mathbf{X}_k - \mathbf{c}_i)}{\|R_i(\mathbf{X}_k - \mathbf{c}_i)\|} \right\|^2$$

여기서:
- $\hat{\mathbf{d}}_{ik}$: 카메라 $i$에서 feature $k$에 대한 **관측된 bearing vector** (normalized image ray)
- $R_i$: 이미 추정된 카메라 회전 (rotation averaging으로 고정)
- $\times$: 외적(cross product)으로, **angular reprojection error**를 측정
- 이 공식화는 scale-invariant하며, 절대 scale 정보 없이도 카메라 위치와 3D 점을 동시에 복원 가능

**핵심 차별점:** 핵심은 최적화에서 **포인트를 직접 활용**하는 것이다. ill-posed translation averaging으로 카메라 위치를 추정하고 point triangulation으로 3D 구조를 별도 획득하는 대신, 이 둘을 **단일 전역 위치 추정 스텝**으로 통합했다.

Ablation study 결과, **포인트 단독 제약조건**이 가장 좋은 성능을 보였다.

---

#### Step 4: Bundle Adjustment (최종 정제)

모든 카메라 파라미터와 3D 포인트를 함께 최적화하는 최종 단계:

$$\min_{\{P_i\},\ \{\mathbf{X}_k\}} \sum_{(i,k) \in \mathcal{V}} \rho\!\left(\left\| \pi(P_i,\ \mathbf{X}_k) - \mathbf{x}_{ik} \right\|^2\right)$$

여기서:
- $\pi(P_i, \mathbf{X}_k)$: 카메라 $P_i = (R_i, \mathbf{c}_i, K_i)$에 의한 투영 함수
- $\mathbf{x}_{ik}$: 카메라 $i$에서 포인트 $k$의 관측된 2D 좌표
- $\rho(\cdot)$: Cauchy/Huber 같은 robust loss function

Ceres 2.3 이상과 cuDSS가 설치된 경우, GLOMAP은 **GPU 가속 최적화**를 지원한다.

---

### 2-3. 모델 구조 (추가 특징)

대부분의 기존 global SfM 시스템과 달리, GLOMAP은 **unknown camera intrinsics**(인터넷 사진에서 흔히 발생)를 처리할 수 있으며, 순차적 이미지 데이터도 강인하게 다룬다.

또한 PixSfM과 같은 학습 기반 파이프라인과 결합하여 서브픽셀 수준의 정확도를 달성하는 joint refinement 메커니즘과도 통합할 수 있다.

---

### 2-4. 성능 향상

GLOMAP은 이미지 기반 sparse 재건을 위한 범용 global SfM 파이프라인으로서, COLMAP 대비 **통상 1~2 order of magnitude(10~100배) 빠른** 효율적이고 확장 가능한 재건 프로세스를 제공하며, 동등하거나 더 우수한 재건 품질을 달성한다.

OpenMVG, Theia, COLMAP을 포함한 최신 시스템들과의 광범위한 실험적 평가에서, GLOMAP은 정확도와 속도 모두에서 우월한 성능을 보였다. 여러 데이터셋(구조화·비구조화 이미지 모두)에서 recall과 AUC 점수에서 유의미한 우위를 보였다.

**Novel View Synthesis 평가 결과 (Mip-NeRF360 데이터셋, AUC ↑):**

| 방법 | bicycle | bonsai | counter | garden | kitchen | room | stump | 평균 |
|------|---------|--------|---------|--------|---------|------|-------|------|
| OpenMVG | 95.3 | 61.2 | 99.1 | 98.5 | 96.5 | 44.9 | 98.5 | 84.9 |
| Theia | 17.9 | 95.6 | 99.7 | 38.9 | 97.6 | 27.6 | 8.0 | 55.0 |
| **GLOMAP** | **98.7** | **99.5** | **99.8** | **99.2** | **98.4** | **99.3** | **99.7** | **99.2** |
| COLMAP | 98.7 | 97.8 | 99.8 | 99.2 | 98.5 | 98.9 | 99.7 | 98.9 |

*(출처: ResearchGate PDF / Table 7 in the paper)*

---

### 2-5. 한계

전반적으로 만족스러운 성능을 달성하지만, 여전히 일부 실패 케이스가 존재한다. 주요 원인은 **rotation averaging의 실패**인데, 예를 들어 대칭 구조(symmetric structures, 논문의 Exhibition_Hall 등)에서 발생한다. 이 경우 Doppelganger 같은 기존 방법과 결합할 수 있다.

또한, **전통적인 correspondence search에 의존**하기 때문에, 잘못 추정된 two-view geometry나 이미지 쌍을 매칭하지 못하는 경우(예: 극적인 외관 변화나 시점 변화)에는 결과가 저하되거나 최악의 경우 완전히 실패할 수 있다.

---

## 3. 🌐 모델의 일반화 성능 향상 가능성

### 3-1. 현재 일반화 한계

Co-linear motion (직선에 가까운 카메라 이동)은 degenerate reconstruction 문제를 초래하며, 이런 패턴은 특히 순차적(sequential) 데이터셋에서 흔히 발생한다. 이러한 문제들은 카메라 위치 추정의 불안정성에 집합적으로 기여하여, 기존 global SfM 시스템의 전반적인 정확도와 강인성에 심각한 영향을 미친다.

### 3-2. 일반화 향상 잠재력

새로운 joint 최적화 접근 방식은 global SfM 프레임워크에서 카메라와 포인트 추정의 통합에 대한 추가 탐구의 길을 열어준다. **하이브리드 접근 방식**(고대칭 장면이나 심한 occlusion과 같은 특정 시나리오에 맞게 조정된 incremental+global 혼합)에 대한 미래 개발이 기대된다. 또한, **학습 기반 방식과의 통합**이 어려운 시각적 조건과 데이터셋의 이상치에 대한 강인성을 더욱 향상시킬 수 있다.

### 3-3. 학습 기반 방식과의 결합 가능성

PixSfM은 서브픽셀 정확도의 재건을 달성하기 위해 features와 structure에 대한 joint refinement 메커니즘을 제안하며, GLOMAP 시스템과 결합할 수 있다.

이와 관련하여, 최근 연구들은 Transformer 기반 방식(DUSt3R, MASt3R, VGGT 등)이 전통적인 SfM 및 MVS 방법을 완전히 대체할 수는 없지만, 특히 어려운 저해상도·극단적 sparse 시나리오에서 **보완적 접근**으로서 잠재력을 가진다고 지적한다.

---

## 4. 🔮 앞으로의 연구에 미치는 영향 및 고려 사항

### 4-1. 연구에 미치는 영향

1. **Global SfM 패러다임의 재정립:** 이 방법은 traditional translation averaging의 단점을 완화하면서, 단일 통합 프레임워크에서 카메라 위치와 구조의 최적화를 강조하는 중요한 진전을 나타낸다.

2. **NeRF/3DGS와의 연계:** NeRF의 입력으로 필요한 sparse point cloud를 COLMAP 대신 global SfM으로 얻는 접근이 유망한 해결책으로 부상하고 있다.

3. **오픈소스 생태계 강화:** 다양한 데이터셋에 대한 광범위한 실험이 제안된 시스템이 점진적 방식과 비교하여 동등하거나 우수한 결과를 훨씬 빠른 속도로 달성함을 보여주었으며, 코드는 상업적으로 친화적인 라이선스 하에 오픈소스로 공개되었다.

---

## 5. 📊 2020년 이후 관련 최신 연구 비교 분석

DUSt3R, MASt3R, VGGT 같은 최신 end-to-end 방법들은 단일 또는 스테레오 이미지에서 포인트 클라우드를 직접 예측하는 패러다임을 따르며, 전통적인 두 단계(sparse → dense) 과정을 우회하여 occlusion에 강인함을 높인다. global motion averaging을 post-processing 단계로 활용하며, VGGT는 DUSt3R가 사용하는 비용이 큰 반복 최적화를 제거하는 feed-forward 신경망으로 파이프라인을 더욱 발전시켰다.

| 방법 | 연도 | 패러다임 | 속도 | 확장성 | 정확도 | 특징 |
|------|------|----------|------|--------|--------|------|
| **COLMAP** | 2016 | Incremental | 느림 | 낮음 | 매우 높음 | 기준선 |
| **PixSfM** | 2021 | Incremental+학습 | 느림 | 낮음 | 매우 높음 | 서브픽셀 정확도 |
| **DUSt3R** | 2024 | E2E 학습 | 중간 | 낮음 | 높음 | 쌍별 포인트맵 예측 |
| **VGGSfM** | 2024 | E2E 학습+BA | 중간 | 낮음\* | 높음 | 미분가능 BA |
| **MASt3R-SfM** | 2024 | E2E 학습 | 중간 | 중간 | 높음 | RANSAC-free |
| **GLOMAP** | 2024 | Global 기하 | **매우 빠름** | **높음** | 매우 높음 | 전통 방식 최고 수준 |
| **VGGT** | 2025 | Transformer E2E | **매우 빠름** | 중간 | 높음 | 단일 forward pass |

학습 기반 방법(VGGSfM 등)의 경우, 80GB GPU에서도 대규모 컬렉션 처리 시 메모리 부족으로 충돌이 발생하는 한계가 있다.

DUSt3R, MASt3R, VGGT 세 방법 모두 고해상도 이미지와 대규모 이미지 세트에서 한계를 보이며, 이미지 수와 장면의 기하학적 복잡성이 증가할수록 카메라 포즈 추정 신뢰도가 크게 저하된다.

Light3R-SfM과 같은 최신 연구에서도, GLOMAP과 COLMAP이 특히 200장 이상의 dense view 설정에서 학습 기반 방법들보다 더 높은 정확도를 보임을 보고하고 있다.

---

## 6. 🔑 연구 시 고려해야 할 점

### (1) Rotation Averaging의 취약점 보완
대칭 구조로 인한 rotation averaging 실패가 주요 약점으로, 이를 해결하기 위해 Doppelganger 같은 방법과의 결합이 필요하다. 향후 연구에서는 대칭 구조 감지 및 처리를 강화해야 한다.

### (2) 학습 기반 특징 매칭과의 통합
학습 기반 방식과의 통합이 어려운 시각적 조건과 데이터셋의 이상치에 대한 강인성을 더욱 향상시킬 수 있다. SuperGlue, LightGlue 등 신경망 기반 매칭기를 GLOMAP의 전처리로 활용하는 방향을 고려해야 한다.

### (3) Large-scale 및 Sequential 데이터
Co-linear motion 패턴이 흔한 순차적(sequential) 데이터셋에서는 degenerate reconstruction 문제가 발생할 수 있어, 드론 촬영이나 자율주행 데이터에 적용 시 추가적인 처리 전략이 필요하다.

### (4) 전통 방식 vs. 학습 기반 방식의 상호 보완
학습 기반 Transformer 방식은 전통적인 SfM 방법을 완전히 대체할 수 없으며, 특히 어렵고 저해상도의 극단적 sparse 시나리오에서 보완적 접근으로서 잠재력을 가진다. GLOMAP은 대규모·고해상도 재건에, 학습 기반 방식은 극단적 sparse·저해상도 환경에 활용하는 하이브리드 전략이 유망하다.

---

## 📚 참고 자료

1. **Pan, L., Baráth, D., Pollefeys, M., & Schönberger, J. L. (2024).** *Global Structure-from-Motion Revisited.* ECCV 2024. arXiv:2407.20219. https://arxiv.org/abs/2407.20219
2. **공식 논문 PDF:** https://demuc.de/papers/pan2024glomap.pdf
3. **ECCV 2024 공식 출판:** https://link.springer.com/chapter/10.1007/978-3-031-73661-2_4
4. **ECCV 2024 포스터:** https://eccv.ecva.net/virtual/2024/poster/1699
5. **Springer (ACM DL):** https://dl.acm.org/doi/10.1007/978-3-031-73661-2_4
6. **ResearchGate:** https://www.researchgate.net/publication/382654690
7. **Semantic Scholar:** https://www.semanticscholar.org/paper/Global-Structure-from-Motion-Revisited-Pan-Bar%C3%A1th/2abe2406539ca8b1f3e4e4f0fab2557dd15ce60f
8. **EmergentMind 분석:** https://www.emergentmind.com/papers/2407.20219
9. **The Moonlight 리뷰:** https://www.themoonlight.io/en/review/global-structure-from-motion-revisited
10. **DeepWiki 파이프라인 분석:** https://deepwiki.com/colmap/glomap/4-global-mapping-pipeline
11. **Rerun.io 예제:** https://rerun.io/examples/3d-reconstruction/glomap
12. **비교 최신 논문 - MASt3R-SfM (2024):** arXiv:2409.19152
13. **비교 최신 논문 - VGGT (2025):** arXiv:2503.11651
14. **비교 최신 논문 - Light3R-SfM (2025):** arXiv:2501.14914
15. **비교 평가 - DUSt3R/MASt3R/VGGT (2025):** Tandfonline, https://doi.org/10.1080/10095020.2025.2597491
16. **COLMAP 기반 논문:** Schönberger & Frahm, *Structure-from-Motion Revisited*, CVPR 2016.
