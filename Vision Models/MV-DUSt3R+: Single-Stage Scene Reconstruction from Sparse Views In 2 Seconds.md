
# MV-DUSt3R+: Single-Stage Scene Reconstruction from Sparse Views In 2 Seconds

> **논문 정보:**
> - **저자:** Zhenggang Tang, Yuchen Fan, Dilin Wang, Hongyu Xu, Rakesh Ranjan, Alexander Schwing, Zhicheng Yan (Meta Reality Labs)
> - **게재:** arXiv:2412.06974 (2024.12) → **CVPR 2025 Oral**
> - **GitHub:** https://github.com/facebookresearch/mvdust3r
> - **Project Page:** https://mv-dust3rp.github.io/

---

## 1. 핵심 주장 및 주요 기여 요약

### 🔑 핵심 주장

DUSt3R·MASt3R와 같은 최신 희소 다시점 재구성 방법들은 카메라 캘리브레이션과 포즈 추정 없이 동작하지만, 한 번에 한 쌍의 뷰(pair)만 처리하여 픽셀 정렬 포인트맵을 추론한다. 2개 이상의 뷰를 다룰 때, 조합론적으로 증가하는 오류가 발생하기 쉬운 쌍별 재구성이 수행된 후, 비용이 큰 전역 최적화(Global Optimization)가 뒤따르며, 이 과정이 쌍별 재구성 오류를 수정하지 못하는 경우가 많다.

이 문제를 해결하기 위해 MV-DUSt3R(+)를 제안한다.

### 🏆 주요 기여

**① MV-DUSt3R 제안:** 희소 다시점 입력으로부터 포즈 없이 씬을 재구성하는 새로운 피드포워드 네트워크로, DUSt3R 대비 4~24개 뷰에서 **48~78배 빠르게** 동작하면서 HM3D, ScanNet, MP3D 3개 평가 데이터셋에서 Chamfer Distance를 각각 감소시킨다.

**② MV-DUSt3R+ 제안:** 단일 레퍼런스 뷰를 통해 모든 입력 뷰 간의 관계를 추론할 때 발생하는 문제를 해결하기 위해 다중 레퍼런스 뷰를 사용하는 MV-DUSt3R+를 제안한다. 이는 모든 태스크, 뷰 수, 3개 데이터셋에서 우수한 성능을 보이며, 예를 들어 24-뷰 입력의 대형 씬에서 Chamfer Distance를 각각 2.6×, 1.6×, 1.8× 추가로 감소시키면서 DUSt3R보다 **14배 빠른** 추론 속도를 유지한다.

**③ Novel View Synthesis(NVS) 확장:** Gaussian Splatting 헤드를 추가하고 공동 학습함으로써 두 방법 모두 NVS로 확장하며, 다시점 스테레오 재구성, 다시점 포즈 추정, 신규 뷰 합성 실험에서 기존 방법들을 크게 능가한다.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능

### 2.1 해결하고자 하는 문제

기존 DUSt3R·MASt3R는 2개 이상의 뷰를 처리할 때 조합론적으로 많은 쌍별 재구성을 수행하고, 그 뒤에 비싼 전역 최적화가 따른다. 이 문제를 해결하기 위해 빠른 단일 단계 피드포워드 네트워크인 MV-DUSt3R를 제안한다.

특히 MV-DUSt3R의 재구성 품질은 공간적으로 불균일하다. 입력 소스 뷰에 대한 포인트맵 예측은 레퍼런스 뷰와의 시점 변화가 작을수록 좋고, 시점 변화가 커질수록 품질이 저하된다. 그러나 희소한 입력 뷰로 대형 씬을 재구성할 때, 모든 소스 뷰와 적당한 시점 변화만을 가지는 단일 레퍼런스 뷰가 존재하기 어렵다.

### 2.2 제안 방법

#### (A) MV-DUSt3R: 단일 레퍼런스 뷰 기반 멀티뷰 디코더

MV-DUSt3R는 대규모의 입력 뷰를 한 번의 피드포워드 패스로 공동 처리하고, 기존 방법들의 캐스케이드 전역 최적화를 완전히 제거한다. 이를 위해 다시점 디코더 블록(Multi-View Decoder Block)을 사용하며, 선택된 레퍼런스 뷰와 모든 소스 뷰 사이의 쌍별 관계뿐만 아니라 소스 뷰들 간의 쌍별 관계도 적절히 처리한다. 또한 학습 레시피를 통해 예측된 뷰별 포인트맵이 동일한 레퍼런스 카메라 좌표계를 따르도록 강제함으로써 후속 전역 최적화의 필요성을 제거한다.

**포인트맵 예측 공식** (DUSt3R 기반):

뷰 $v$에 대한 포인트맵 $X_{v,m}$과 신뢰도 맵 $C_{v,m}$을 레퍼런스 뷰 $r_m$의 카메라 좌표계에서 예측:

$$\{X_{v,m}, C_{v,m}\} = \text{MV-DUSt3R}(\{I_v\}_{v=1}^{N},\, r_m)$$

여기서 $I_v$는 $v$번째 입력 RGB 이미지, $r_m$은 선택된 레퍼런스 뷰이다.

**신뢰도 기반 손실 함수** (DUSt3R 방식 따름):

$$\mathcal{L}_{\text{conf}} = \sum_{v} \sum_{i \in \text{valid}} \left( \frac{\|\hat{X}_{v,m}^i - X_{v,m}^{*i}\|}{C_{v,m}^i} + \log C_{v,m}^i \right)$$

여기서 $\hat{X}\_{v,m}^i$는 예측 포인트맵, $X_{v,m}^{*i}$는 GT 포인트맵, $C_{v,m}^i$는 신뢰도 값이다.

#### (B) MV-DUSt3R+: Cross-Reference-View Block 도입

MV-DUSt3R+는 여러 뷰를 레퍼런스 뷰로 선택하고, 선택된 각 레퍼런스 뷰의 카메라 좌표계에서 모든 입력 뷰의 포인트맵을 공동으로 예측한다.

레퍼런스 뷰 집합 $R = \{r_m\}_{M}$에 대해, 각 디코더 블록 이후에 **Cross-Reference-View Block**을 추가하여 서로 다른 레퍼런스 뷰 선택 하에서 계산된 뷰별 토큰을 융합하고 업데이트한다.

**Multi-Path 모델 추론:**

$$\bar{X}_{v} = \text{Fusion}\left(\{X_{v,m}\}_{m=1}^{M}\right)$$

- 추론 시 $M$개의 레퍼런스 뷰가 균일하게 선택되고 (첫 번째 입력 뷰는 항상 포함), $M$-경로 모델이 사용되어 최종 뷰별 포인트맵 예측이 계산된다.

#### (C) Novel View Synthesis 확장: Gaussian Head

별도의 헤드를 추가하여 픽셀별 가우시안 파라미터를 예측한다: 스케일 팩터 $S_{v,m} \in \mathbb{R}^{H \times W \times 3}$, 회전 쿼터니언 $q_{v,m} \in \mathbb{R}^{H \times W \times 4}$, 불투명도 $\alpha_{v,m} \in \mathbb{R}^{H \times W}$. 다른 가우시안 파라미터의 경우, 예측된 포인트맵 $X_{v,m}$을 중심으로, 픽셀 색상 $I_v$를 색상으로 사용하고, 구면 조화 차수는 0으로 고정한다.

**전체 훈련 손실:**

$$\mathcal{L}_{\text{all}} = \mathcal{L}_{\text{conf}} + \delta \mathcal{L}_{\text{render}}$$

$\mathcal{L}\_{\text{conf}}$는 재구성 손실이고, $\mathcal{L}_{\text{render}}$는 뷰 렌더링 손실이며, $\delta$는 두 손실의 균형을 맞추는 가중치이다.

#### (D) 학습 전략 (2-Stage Training)

1단계(Stage 1)에서는 DUSt3R로부터 로드하여 8-뷰 입력으로 파인튜닝하고, 2단계(Stage 2)에서는 1단계 모델을 4~12개 뷰의 혼합 뷰로 추가 파인튜닝한다.

MP3D에서의 향상은 입력 뷰 수가 많을 때(예: 24 뷰) 더욱 두드러지며, 2단계 학습은 모델이 씬 크기와 입력 뷰 수에 더 강인하도록 만든다.

---

### 2.3 모델 구조 요약

| 구성 요소 | 설명 |
|---|---|
| **Encoder** | 공유 가중치 ViT 인코더 (DUSt3R 사전학습 모델 재사용) |
| **DecBlock (Multi-View)** | 레퍼런스 뷰와 모든 소스 뷰 간 정보 교환 (Self-Attention + Cross-Attention) |
| **CrossRefViewBlock** | 서로 다른 레퍼런스 뷰 경로 간 토큰 융합 (MV-DUSt3R+에만 존재) |
| **Pointmap Head** | 각 뷰에 대해 $X_{v,m}$, $C_{v,m}$ 예측 |
| **Gaussian Head** | NVS를 위한 $S_{v,m}$, $q_{v,m}$, $\alpha_{v,m}$ 예측 (경량 추가 헤드) |

---

### 2.4 성능 향상

HM3D (다중 방 씬)에서 MV-DUSt3R는 씬 크기와 입력 뷰 수가 증가할수록 DUSt3R을 지속적으로 능가한다. 예를 들어 4-뷰 입력에서 ND(정규화 거리)를 1.7배 감소, DAc(거리 정확도)를 1.2배 향상시키고, 24-뷰 입력에서는 ND를 2배 감소, DAc를 5.3배 향상시킨다.

MV-DUSt3R+는 다중 포즈 없는 RGB 뷰로부터 대형 씬을 재구성할 수 있으며, 12개 입력 뷰로는 단일 방 씬을 **0.89초**, 20개 입력 뷰로는 대형 다중 방 씬을 **1.54초** 만에 재구성한다.

MV-DUSt3R+는 특히 어려운 설정에서 재구성 품질을 향상시키면서 DUSt3R보다 **한 자릿수 빠르게** 추론한다.

---

### 2.5 한계점

Spann3R은 공간 메모리를 활용하여 순서 있는 이미지 세트를 처리할 수 있지만, 대형 씬에서 공간 메모리의 제한된 크기와 전역 정렬 부재로 인해 드리프트가 발생하기 쉽다. 반면 MV-DUSt3R+는 이를 극복하지만 다음과 같은 한계를 가진다:

- 대형 씬을 희소 다시점 이미지로 재구성할 때, 선택된 단일 레퍼런스 뷰와 특정 소스 뷰 간의 스테레오 단서가 불충분할 수 있다.
- MV-DUSt3R는 전반적으로 더 강인하지만 레퍼런스 뷰에서 먼 영역에서 정확한 기하학 재구성에 실패하는 경우가 있으며, MV-DUSt3R+는 공간 전체에서 더 균일하게 기하학을 예측한다.
- 학습에 사용된 실내 중심 데이터셋(ScanNet, HM3D 등)으로 인해 실외 또는 객체 중심 씬으로의 일반화에 제한이 있다.
- 훈련 시 최대 24-뷰 고정이므로 그 이상의 뷰 수 처리에 대한 추가 연구가 필요하다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 뷰 수 일반화

MV-DUSt3R+ 모델은 8-뷰 샘플로 훈련됨에도 불구하고, **100-뷰 입력**으로의 일반화가 놀랍도록 잘 이루어지며, 단일 방 씬에서 100-뷰 처리에 19.1초가 소요된다.

100-뷰 입력을 사용한 MV-DUSt3R+의 일반화 성능 검증 결과, 훈련 시 뷰 수가 고정되어 있음에도 불구하고 우수한 성능을 보였다.

### 3.2 2단계 학습을 통한 일반화

2단계 학습에서는 1단계에서 8-뷰 고정 입력으로 학습된 모델을 70 에폭 더 파인튜닝하는데, 4~12 뷰로 랜덤하게 샘플링된 입력을 사용한다. 2단계 학습 모델이 HM3D, MP3D를 포함한 대형 씬 데이터셋에서 1단계 학습 모델보다 일관되게 우수한 성능을 보였다.

### 3.3 Cross-Reference-View Block의 역할

4-뷰 입력에서 MV-DUSt3R+는 ND를 1.3배 향상, DAc를 1.2배 향상시키며, 24-뷰 입력에서는 ND가 1.6배, DAc가 1.8배 향상되어 더 큰 개선을 보인다. 다중 경로 아키텍처가 서로 다른 레퍼런스 뷰 선택에 걸쳐 멀티뷰 단서를 보다 효과적으로 융합할 수 있게 한다.

### 3.4 일반화 성능 향상 전략 정리

| 전략 | 내용 |
|---|---|
| **DUSt3R 가중치 초기화** | 대규모 데이터로 사전학습된 DUSt3R의 표현력을 상속 |
| **2단계 학습** | 고정 뷰 수 → 혼합 뷰 수로 확장하여 뷰 수 일반화 강화 |
| **Cross-Reference-View Block** | 여러 레퍼런스 뷰 경로 간 정보 융합으로 공간 일반화 향상 |
| **다양한 실내 데이터셋 활용** | ScanNet, ScanNet++, HM3D, Gibson, MP3D 5가지 데이터셋 활용 |
| **100-뷰 테스트** | 훈련 분포 밖 뷰 수에 대한 제로샷 일반화 검증 |

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4.1 앞으로의 연구에 미치는 영향

**① 단일 단계 멀티뷰 재구성 패러다임 정착**

MV-DUSt3R+는 모든 입력 뷰(실험에서 최대 24개)를 동시에 처리하는 오프라인 씬 재구성을 수행하며, DUSt3R와 달리 예측된 뷰별 포인트맵이 이미 전역 정렬되어 있어 전역 최적화가 필요하지 않다. 이는 향후 3D 재구성 연구의 표준 기준을 높인다.

**② 관련 후속 연구 방향**

- MUSt3R(2025)는 DUSt3R를 다시점으로 확장하는 연구로, DUSt3R 아키텍처를 대칭적으로 수정하여 공통 좌표 프레임에서 모든 뷰에 대한 3D 구조를 직접 예측한다.
- D2USt3R(2025)는 동적 씬 재구성 문제를 다루며, 정적 씬을 위해 설계된 DUSt3R 계열 방법들이 동적 움직임이 있는 경우 어려움을 겪는다는 점을 지적하고, 4D 포인트맵을 회귀하는 방법을 제안한다.

**③ 실시간 AR/VR, 로보틱스, 자율주행 응용**

MV-DUSt3R(+)는 포즈 없는 멀티뷰 RGB만으로 한 단계에 3D 재구성이 가능하며, 신규 뷰 합성과 상대적 포즈 추정도 지원한다. 이는 실시간 환경 인식이 필요한 다양한 응용 분야로의 확장 가능성을 열어준다.

---

### 4.2 향후 연구 시 고려할 점

#### 🔬 기술적 한계 극복
1. **실외 씬 일반화**: 훈련 데이터는 ScanNet, ScanNet++, HM3D, Gibson, MP3D의 5가지 데이터셋을 사용하며, 모두 실내 중심이므로 실외 씬 혹은 다양한 도메인으로의 일반화 연구가 필요하다.

2. **동적 씬 처리**: 현재 방법은 정적 씬을 가정하므로, 동적 요소를 포함하는 씬에서는 카메라 포즈 기반 정렬이 방해받아 잘못 정렬된 대응 관계와 부정확한 깊이 추정이 발생한다.

3. **메모리 효율 및 초고해상도 뷰 처리**: 뷰 수가 크게 늘어날 때의 Transformer 어텐션 복잡도( $O(N^2)$ ) 문제를 해결하기 위한 효율적 어텐션 메커니즘 연구가 필요하다.

#### 📊 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 연도 | 입력 뷰 | 포즈 필요 여부 | 전역 최적화 | 속도 | 특징 |
|---|---|---|---|---|---|---|
| **NeRF** (Mildenhall et al.) | 2020 | 다수 | ✅ 필요 | ❌ | 느림 | 씬별 최적화 |
| **COLMAP+MVS** | 전통 | 다수 | ✅ 필요 | ✅ BA | 느림 | SfM 기반 |
| **DUSt3R** (Wang et al.) | 2024 | **2뷰** | ❌ | ✅ GO | 기준 | 포즈프리 쌍별 |
| **MASt3R** (Leroy et al.) | 2024 | **2뷰** | ❌ | ✅ GO | 기준 | 메트릭 포인트맵 |
| **Spann3R** (Wang et al.) | 2024 | 순서 있음 | ❌ | ❌ | 빠름 | 공간 메모리, 드리프트 |
| **MV-DUSt3R** | 2024 | **N뷰** | ❌ | ❌ | DUSt3R 대비 48~78× | 단일 레퍼런스 |
| **MV-DUSt3R+** | 2024 | **N뷰** | ❌ | ❌ | DUSt3R 대비 14× | 다중 레퍼런스, CVPR'25 Oral |
| **MUSt3R** (Naver) | 2025 | **N뷰** | ❌ | ❌ | 빠름 | 대칭 아키텍처 |

NoPoSplat과 같이 포즈 없이 3D 가우시안을 예측하는 방법들은 주로 2개 입력 뷰에 초점을 맞추며, 희소 다시점 입력으로의 처리 성능이 불분명한 반면, MV-DUSt3R·MV-DUSt3R+는 카메라 포즈 없이 다중 뷰에서 대형 씬을 단일 피드포워드 패스로 재구성한다.

#### 💡 미래 연구 제언
1. **도메인 일반화**: 실내 외 혼합 및 항공 씬 등으로 사전학습 데이터 다각화
2. **증분적 재구성**: 온라인 환경에서의 점진적 씬 업데이트 메커니즘 연구 (Spann3R의 드리프트 문제 해결)
3. **의미 정보 통합**: 3D 재구성과 의미 분할을 결합한 통합 모델 연구 (LSM 방향)
4. **동적 씬 확장**: MV-DUSt3R+의 멀티뷰 프레임워크를 4D 재구성으로 확장
5. **경량화**: 엣지 디바이스 배포를 위한 모델 압축 및 양자화 연구

---

## 📚 참고 자료 (출처)

1. **논문 원문 (arXiv):** Tang et al., "MV-DUSt3R+: Single-Stage Scene Reconstruction from Sparse Views In 2 Seconds", arXiv:2412.06974, 2024. https://arxiv.org/abs/2412.06974
2. **CVPR 2025 공식 논문 PDF:** https://openaccess.thecvf.com/content/CVPR2025/papers/Tang_MV-DUSt3R_Single-Stage_Scene_Reconstruction_from_Sparse_Views_In_2_Seconds_CVPR_2025_paper.pdf
3. **CVPR 2025 Supplemental PDF:** https://openaccess.thecvf.com/content/CVPR2025/supplemental/Tang_MV-DUSt3R_Single-Stage_Scene_CVPR_2025_supplemental.pdf
4. **Project Page:** https://mv-dust3rp.github.io/
5. **GitHub (공식 구현):** https://github.com/facebookresearch/mvdust3r
6. **DUSt3R 원논문:** Wang et al., "DUSt3R: Geometric 3D Vision Made Easy", CVPR 2024. arXiv:2312.14132. https://arxiv.org/abs/2312.14132
7. **MUSt3R:** "MUSt3R: Multi-view Network for Stereo 3D Reconstruction", arXiv:2503.01661, 2025. https://arxiv.org/abs/2503.01661
8. **D2USt3R:** "D2USt3R: Enhancing 3D Reconstruction with 4D Pointmaps for Dynamic Scenes", arXiv:2504.06264, 2025. https://arxiv.org/html/2504.06264v1
9. **Easi3R:** "Easi3R: Estimating Disentangled Motion from DUSt3R Without Training", arXiv:2503.24391, 2025.
10. **DUSt3R 설명 블로그:** LearnOpenCV, "DUSt3R: Geometric 3D Vision Made Easy - Explanation & Results". https://learnopencv.com/dust3r-geometric-3d-vision/
11. **Semantic Scholar:** https://www.semanticscholar.org/paper/MV-DUSt3R+
