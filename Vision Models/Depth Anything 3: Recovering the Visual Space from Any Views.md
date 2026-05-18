
# Depth Anything 3: Recovering the Visual Space from Any Views

> **논문 정보**
> - **제목**: Depth Anything 3: Recovering the Visual Space from Any Views
> - **저자**: Haotong Lin, Sili Chen, Jun Hao Liew, Donny Y. Chen, Zhenyu Li, Guang Shi, Jiashi Feng, Bingyi Kang (ByteDance Seed)
> - **arXiv**: [2511.10647](https://arxiv.org/abs/2511.10647) (2025년 11월 13일)
> - **게재**: ICLR 2026 (Academic Paper)

---

## 1️⃣ 핵심 주장 및 주요 기여 요약

### 🔑 핵심 주장 (TL;DR)

"Depth Anything 3 recovers the space with superior geometry and 3DGS rendering from any visual inputs" — 이는 단일 plain transformer를 depth-ray representation으로 학습함으로써 달성된다.

이 연구는 기존의 확립된 3D 태스크 정의에서 한 발 물러서, 인간의 공간 지능에서 영감을 받은 보다 근본적인 목표로 돌아가고자 한다: 단일 이미지, 다중 뷰, 또는 비디오 스트림 등 임의의 시각적 입력으로부터 3D 구조를 복원하는 것이다. 이를 위해 복잡한 아키텍처 엔지니어링을 버리고 최소한의 모델링 전략을 추구한다.

### 📌 주요 기여 요약표

| 기여 항목 | 내용 |
|---|---|
| **통합 아키텍처** | 단일 plain transformer로 임의 뷰 입력 처리 |
| **Depth-Ray 표현** | 깊이와 포즈를 동시에 예측하는 단일 타겟 |
| **Teacher-Student 학습** | 실세계 데이터의 pseudo-label 생성으로 일반화 달성 |
| **새 벤치마크** | 카메라 포즈, 임의뷰 기하, 시각 렌더링 포함 |
| **SOTA 성능** | VGGT 대비 카메라 포즈 정확도 +44.3%, 기하 정확도 +25.1% |

이 벤치마크에서 DA3는 모든 작업에서 새로운 최첨단(SOTA)을 달성하며, 이전 SOTA인 VGGT를 카메라 포즈 정확도에서 평균 35.7%, 기하 정확도에서 23.6% 능가하며, DA2에 비해 단안 깊이 추정에서도 우월한 성능을 보인다.

---

## 2️⃣ 상세 설명: 문제 → 방법 → 구조 → 성능 → 한계

---

### 🔴 2-1. 해결하고자 하는 문제

기존 컴퓨터 비전에서는 단안 깊이 추정, 다중 뷰 스테레오, 카메라 포즈 추정 등의 작업에 각각 별도의 특화 모델이 필요했다. DA3는 이 모든 것을 단일 아키텍처로 해결하며, 이러한 태스크들을 하나의 근본적인 문제인 "시각적 관찰로부터 3D 구조 복원"으로 재구성한다.

구체적으로 해결하고자 하는 두 가지 핵심 질문은 다음과 같다:

① 최소한의 예측 타겟 집합이 존재하는가, 아니면 수많은 3D 태스크에 걸친 공동 모델링이 필요한가? ② 단일 plain transformer가 이 목표에 충분한가? 이 논문은 두 질문 모두에 긍정적인 답을 제공한다.

---

### 🟡 2-2. 제안하는 방법 및 수식

#### ① Depth-Ray 예측 타겟 (핵심 표현)

DA3는 새로운 depth-ray 예측 타겟을 활용하는데, 이는 dense depth 맵과 pixel-aligned ray 맵을 결합한 것으로, 장면 구조와 카메라 운동을 포착하기 위한 최소한이면서도 충분한 표현임이 입증되었다.

각 입력 이미지에 대해 DA3는 세 가지 보완적 출력(depth-ray-camera)을 생성한다: **Depth**: 카메라로부터 픽셀당 거리 정보를 나타내는 $H \times W$ 크기의 지수 깊이 맵; **Ray**: 각 픽셀의 기하를 인코딩하는 $H \times W \times 6$ 크기의 dense ray 맵 텐서.

$$\text{Model Input: } \{I_1, I_2, \ldots, I_N\} \quad \text{(임의 개수의 이미지, 포즈 선택적)}$$

$$\text{Model Output: } \{(D_i, R_i)\}_{i=1}^{N}$$

여기서:
- $D_i \in \mathbb{R}^{H \times W}$: $i$번째 뷰의 **지수 깊이 맵(exponential depth map)**
- $R_i \in \mathbb{R}^{H \times W \times 6}$: $i$번째 뷰의 **픽셀 정렬 ray 맵(dense ray map)**

메트릭 깊이 변환 수식은:

$$d_{\text{metric}} = \frac{f_{\text{pix}} \cdot d_{\text{net}}}{300}$$

여기서 $f_{\text{pix}}$는 픽셀 단위의 focal length (일반적으로 카메라 내부 행렬 $K$에서 $f_x$와 $f_y$의 평균)이다.

#### ② Teacher-Student 학습 패러다임

DA3는 다양한 학습 데이터를 통합하기 위해 teacher-student 패러다임으로 학습된다. 데이터 소스에는 실세계 깊이 카메라 캡처(ARKitScenes 등), 3D 재구성(Common Objects in 3D 등), 합성 데이터 등 다양한 형식이 포함된다. 이를 해결하기 위해 이전 연구들에서 영감을 받은 pseudo-labeling 전략을 채택한다.

구체적으로, 강력한 teacher 단안 깊이 모델을 합성 데이터로 학습시켜 모든 실세계 데이터에 대한 dense한 고품질 pseudo-depth를 생성한다. 기하 무결성 보존을 위해, 이 dense pseudo-depth 맵을 원래의 희소하거나 노이즈가 있는 깊이와 정렬한다. 이 접근법은 기하 정확도를 희생하지 않으면서 레이블의 세부 사항과 완성도를 크게 향상시키는 데 매우 효과적임이 입증되었다.

Teacher 손실 함수:

$$\mathcal{L}_{\text{teacher}} = \mathcal{L}_{\text{grad}} + \mathcal{L}_{\text{align}} + \mathcal{L}_{\text{normal}} + \mathcal{L}_{\text{semantic}}$$

Teacher(DA3-Teacher)는 학생과 동일한 아키텍처를 공유하지만 대규모 합성 깊이 데이터셋에서 독점적으로 학습되며, scale-shift-invariant exponential depth를 강조한다. Teacher 손실은 gradient, alignment, normal, semantic(하늘/객체) 항을 결합한다.

Pseudo-label RANSAC 정렬:

$$\hat{d}_{\text{pseudo}} = s \cdot d_{\text{teacher}} + t, \quad (s, t) = \arg\min_{s,t} \|\hat{d}_{\text{pseudo}} - d_{\text{sparse}}\|$$

Pseudo-depth는 RANSAC least squares를 사용하여 scale과 shift를 위해 이용 가능한 ground truth에 견고하게 정렬된다. 이는 teacher의 상세한 상대 기하를 기하학적으로 정확하지만 희소한 실세계 측정에 고정시킨다.

포즈 컨디셔닝은 다양한 사용 시나리오에 걸친 일반화를 촉진하기 위해 확률적으로 토글된다.

---

### 🟢 2-3. 모델 구조

아키텍처는 사전학습된 표준 비전 트랜스포머(DINOv2)를 백본으로 시작하며, 강력한 특징 추출 능력을 활용한다. 임의의 뷰 수를 처리하기 위해 핵심적인 수정을 도입한다: **input-adaptive cross-view self-attention 메커니즘**이다. 이 모듈은 forward pass 중 선택된 레이어에서 토큰을 동적으로 재배열하여 모든 뷰에 걸친 효율적인 정보 교환을 가능하게 한다.

최종 예측을 위해, 동일한 특징 집합을 별개의 fusion 파라미터로 처리함으로써 깊이와 ray 값을 동시에 출력하는 **새로운 dual DPT 헤드**를 제안한다. 유연성을 높이기 위해 모델은 간단한 카메라 인코더를 통해 알려진 카메라 포즈를 선택적으로 통합할 수 있어, 다양한 실용적 설정에 적응할 수 있다.

**모델 시리즈 구성:**

① **DA3 Main Series** (DA3-Giant, DA3-Large, DA3-Base, DA3-Small): unified depth-ray representation으로 학습된 flagship 기반 모델로, 단일 입력 구성 변경으로 Monocular Depth Estimation, Multi-View Depth Estimation, Pose-Conditioned Depth Estimation 등 광범위한 태스크 수행 가능.

② **DA3 Monocular Series** (DA3Mono-Large): 고품질 상대 단안 깊이 추정을 위한 전용 모델. DA2와 같은 disparity 기반 모델과 달리, 직접 깊이를 예측하여 우수한 기하 정확도를 달성.

③ **DA3 Nested Series** (DA3Nested-Giant-Large): Any-view giant 모델과 메트릭 모델을 결합하여 실세계 메트릭 스케일로 시각 기하를 복원.

**아키텍처 다이어그램 (개념):**

```
입력 이미지 {I_1,...,I_N} (포즈 선택적)
         ↓
[DINOv2 Vision Transformer Backbone]
         ↓
[Input-Adaptive Cross-View Self-Attention]  ← 뷰 간 정보 교환
         ↓
[Dual DPT Head]
   ↙           ↘
Depth Maps    Ray Maps
{D_i}         {R_i}
         ↓
[Point Cloud Fusion / 3DGS]
```

---

### 🔵 2-4. 성능 향상

새 벤치마크에서 DA3는 모든 태스크에서 새로운 SOTA를 달성하며, 이전 SOTA VGGT를 카메라 포즈 정확도에서 평균 44.3%, 기하 정확도에서 25.1% 능가한다. 또한 단안 깊이 추정에서도 DA2를 능가한다.

벤치마크는 5개의 서로 다른 데이터셋으로 구성되며, 총 89개 이상의 장면을 포함하고 객체 레벨부터 실내·실외 환경까지 다양하다.

DA3-Metric 모델은 ETH3D 벤치마크에서 delta 1 점수 0.917, AbsRel 0.104를 달성하여 2위인 UniDepthv2(delta 1: 0.863)를 크게 앞선다. 이는 다양한 실외 장면에서 우수한 정확도와 견고성을 나타낸다. DA3-metric은 SUN-RGBD 데이터셋에서 AbsRel(0.105) 최고 성능을, DIODE에서 2위(delta 1 = 0.838, AbsRel = 0.128)를 달성한다.

**성능 비교 요약표:**

| 벤치마크 | DA3 vs VGGT | DA3 vs DA2 |
|---|---|---|
| 카메라 포즈 정확도 | **+44.3%** | — |
| 기하 정확도 | **+25.1%** | — |
| ETH3D (delta 1) | — | **+>10% (0.917)** |
| SUN-RGBD AbsRel | — | **0.105 (SOTA)** |
| 20개 설정 SOTA 달성 | **18/20** | — |

정량적 결과에 따르면, VGGT-Long에서 VGGT를 DA3(DA3-Long)로 단순 교체하는 것만으로도 대규모 환경에서의 드리프트가 크게 줄어들며, 완료하는 데 48시간 이상 소요되는 COLMAP보다도 우수한 성능을 보인다.

---

### ⚠️ 2-5. 한계점

모든 모델이 DL3DV에서는 다른 데이터셋에 비해 훨씬 더 우수한 성능을 보여, 3DGS 기반 NVS가 장면 콘텐츠보다 DL3DV로 표준화된 궤적 및 포즈 분포에 민감하다는 점을 시사한다.

합성 데이터만으로 학습된 teacher를 사용할 경우, 실제 이미지에 대한 레이블 생성 시 도메인 갭이나 합성 세계 편향이 도입될 위험이 있다는 우려가 제기되고 있다.

저질감 텍스처 또는 동적 장면에서의 DA3 적응에 관한 과제가 여전히 남아있다.

추가로, 논문 자체에서 공개적으로 밝히지 않은 한계로서 다음을 고려해야 한다:
- **대용량 모델 크기**: DA3-Giant와 같은 대형 모델은 엣지 디바이스 배포에 어려움
- **실시간 처리 제한**: 대규모 멀티뷰 입력에서 추론 속도
- **특정 벤치마크 편향**: NYUv2, KITTI에서 UniDepthv1/v2가 더 우수할 수 있음

---

## 3️⃣ 일반화 성능 향상 가능성

UniDepthv1, UniDepthv2와 같은 다른 방법들이 NYUv2, KITTI와 같은 특정 벤치마크에서 뛰어날 수 있지만, DA3-metric은 광범위한 벤치마크에 걸쳐 강력한 일반화 성능을 보여주며 다재다능함을 강조한다.

DA3-metric의 학습 과정에는 Taskonomy, DIML (Outdoor), DDAD, Argoverse, Lyft, PandaSet, Waymo, ScanNet++, ARKitScenes, Map-free, DSEC, Driving Stereo, Cityscapes를 포함한 14개의 다양한 데이터셋이 활용된다. 이처럼 광범위하고 다양한 학습 데이터와 스테레오 데이터셋에 대한 FoundationStereo 예측의 사용이 다양한 환경과 시나리오에 걸친 모델의 강력한 일반화 및 경쟁력 있는 성능에 기여한다.

대규모 pretraining이 제공하는 이점 덕분에, epipolar transformer, cost volume, 또는 cascaded 모듈에 의존하는 접근법보다 더 나은 일반화 및 확장성이 가능하다. 이 그룹 내에서 NVS 성능은 기하 추정 능력과 상관관계가 있으므로 DA3가 가장 강력한 백본이 된다.

단일 트랜스포머가 하나의 이미지, 다중 뷰, 포즈 유무에 상관없이 모든 입력에서 작동하는 것은 학문적으로뿐 아니라 입력 제도가 변화하는 실제 시스템에서도 매력적이다.

### 📈 일반화 향상을 위한 핵심 메커니즘

1. **포즈 컨디셔닝의 확률적 토글**
포즈 컨디셔닝은 다양한 사용 시나리오에 걸친 일반화를 촉진하기 위해 확률적으로 토글된다.

2. **Teacher 생성 Pseudo-label의 RANSAC 정렬**
ablation 연구를 통해 teacher supervision이 결정적임이 확인되었다; 이를 제거하면 성능이 눈에 띄게 하락하며, 특히 세부 사항 포착에서 큰 차이가 난다.

3. **공개 학술 데이터셋만 사용**
모든 모델은 공개 학술 데이터셋만으로 학습된다. 이는 재현 가능성과 공정한 비교를 보장한다.

4. **Feed-Forward 3DGS에서의 확장**
전체 백본을 동결하고 3DGS 파라미터를 예측하는 DPT 헤드를 학습시킴으로써 매우 강력하고 일반화 가능한 novel view synthesis 능력을 달성한다.

---

## 4️⃣ 앞으로의 연구에 미치는 영향 및 고려 사항

### 🌍 연구에 미치는 영향

#### A. "Minimal Modeling" 패러다임의 확산

또 다른 시사점은 "foundation geometry"가 새로운 네트워크를 발명하는 것보다 대규모에서 supervision을 표준화하는 것에 관한 것일 수 있다는 점이다.

최소 모델링 원칙과 DA3에서 확립된 prompt-foundation fusion이 더욱 일반화되어, 단일 다재다능한 아키텍처를 사용하여 대부분의 하위 컴퓨터 비전 및 로보틱스 태스크에 걸쳐 통합 시각 기하 모델링을 가능하게 할 것으로 기대된다.

#### B. SLAM 및 로보틱스 분야 혁신

DA3는 깊이와 포즈를 공동 추정하고, 카메라 궤적을 추적하며, 정확한 3D 포인트 클라우드를 위한 dense하고 pixel-aligned된 깊이 및 ray 맵을 생성함으로써 로보틱스의 SLAM에 활용될 수 있다.

#### C. 3D Gaussian Splatting / Novel View Synthesis 파이프라인

앞으로 FF-NVS는 사전학습된 기하 백본을 활용하는 단순 아키텍처로 효과적으로 해결될 수 있을 것으로 기대되며, DA3의 강력한 공간 이해가 다른 3D 비전 태스크에도 도움이 될 것이다.

#### D. 새로운 벤치마크 기준 설정

모델을 보다 잘 평가하고 분야의 진보를 추적하기 위해, 기하 및 포즈 정확도를 평가하는 포괄적인 벤치마크를 구축하였다. 이 벤치마크는 총 89개 이상의 장면을 포함하는 5개의 서로 다른 데이터셋으로 구성되며, 객체 레벨부터 실내 및 실외 환경까지 다양하다.

---

### 🔬 2020년 이후 관련 최신 연구 비교 분석

| 연구명 | 연도 | 핵심 방법 | 한계 | DA3와의 비교 |
|---|---|---|---|---|
| **MiDaS** | 2020 | 다중 데이터셋 단안 깊이 | 단일 뷰만 지원 | DA3가 다중 뷰 확장 |
| **NeRF** | 2020 | 신경 복사 필드 | 장면별 최적화 필요 | DA3는 feed-forward |
| **DUSt3R** | 2023 | 포즈 없는 다중 뷰 재구성 | 별도 복잡한 헤드 필요 | DA3가 더 단순한 구조 |
| **Depth Anything v1** | 2024 | 대규모 단안 깊이 | 단일 뷰 한정 | DA3는 Any-view 확장 |
| **Depth Anything v2** | 2024 | 고품질 단안 깊이 | 다중 뷰 비일관성 | DA3가 일관성 추가 달성 |
| **VGGT** | 2025 | 다중 뷰 기하 추정 | 특화 아키텍처 필요 | DA3가 포즈 +44.3%, 기하 +25.1% 우세 |
| **3D Gaussian Splatting** | 2023 | 실시간 장면 렌더링 | COLMAP 사전 처리 필요 | DA3가 포즈 추정 내장 |

VGGT, DUSt3R과 같이 여러 3D 태스크를 한 번에 해결하려는 대규모 통합 모델의 추세가 있었다. ByteDance Seed 팀의 이 연구는 그 패러다임에 도전하며, "더 복잡한 아키텍처로 더 많이 하는 것이 아니라, 더 적은 것으로 더 많은 것을 달성할 수 있는가?"를 묻는다.

---

### 🗺️ 향후 연구 시 고려할 점

1. **동적 장면 및 저질감 텍스처 대응**
   저질감 텍스처 또는 동적 장면에서 DA3를 적응하는 과제가 남아있다. 이는 실세계 로보틱스 적용 시 중요한 연구 방향이다.

2. **합성-실세계 도메인 갭 최소화**
   Teacher 레이블을 단독으로 사용하는 것이 아니라, pseudo-depth를 RANSAC least squares로 scale과 shift를 위해 ground truth에 견고하게 정렬함으로써 이 문제를 완화한다. 향후 연구에서는 더 강건한 도메인 적응 기법이 필요하다.

3. **경량화 모델 연구**
   DA3-Streaming이 공개되어 슬라이딩 윈도우 스트리밍 추론을 통해 12GB 미만의 GPU 메모리로 초장기 비디오 시퀀스 추론이 가능하게 되었다. 그러나 모바일/엣지 환경을 위한 더욱 경량화된 버전 개발이 요구된다.

4. **Pose-Adaptive 설계의 확장**
   모든 입력 이미지가 비보정 상태라고 가정하는 대신, 포즈가 있거나 없는 입력을 모두 수용하는 pose-adaptive 설계를 채택하여 포즈 유무에 상관없이 작동하는 유연한 프레임워크를 제공한다. 이 설계 원칙을 더 많은 하위 태스크에 확장하는 연구가 필요하다.

5. **Teacher 다양화 및 앙상블**
   Teacher-student 파이프라인이 노이즈가 많은 실세계 깊이를 단일 모델이 일관성 있게 학습할 수 있는 형태로 변환하는 데 핵심적인 역할을 한다. 다양한 도메인에 특화된 teacher 앙상블을 활용하면 더 강력한 일반화가 가능할 것이다.

6. **Minimal Modeling 철학의 타 3D 태스크 적용**
   "최소 모델링, 최대 일반성"이 이 논문의 핵심 메시지다: 타겟 표현을 잘 선택하면 많은 "서로 다른" 3D 태스크가 동일한 문제처럼 보이기 시작한다. 이 철학을 semantic 3D 이해, 동적 장면 복원, 의료 영상 등에 적용하는 연구가 기대된다.

---

## 📚 참고 자료 및 출처

| # | 출처 | URL |
|---|---|---|
| 1 | **arXiv 논문 (v1)** | https://arxiv.org/abs/2511.10647 |
| 2 | **프로젝트 페이지** | https://depth-anything-3.github.io/ |
| 3 | **공식 GitHub (ByteDance-Seed)** | https://github.com/ByteDance-Seed/Depth-Anything-3 |
| 4 | **Hugging Face Papers** | https://huggingface.co/papers/2511.10647 |
| 5 | **arXiv HTML 전문** | https://arxiv.org/html/2511.10647v1 |
| 6 | **OpenReview (ICLR 2026)** | https://openreview.net/forum?id=yirunib8l8 |
| 7 | **ResearchGate** | https://www.researchgate.net/publication/397595945 |
| 8 | **Roboflow Blog** | https://blog.roboflow.com/depth-anything-3/ |
| 9 | **Medium (Jang_ai)** | https://medium.com/@Jang_ai/depth-anything-3-... |
| 10 | **liner.com Quick Review** | https://liner.com/review/depth-anything-3-... |
| 11 | **alphaXiv** | https://www.alphaxiv.org/resources/2511.10647v1 |
| 12 | **emergentmind.com** | https://www.emergentmind.com/topics/depth-anything-3-da3 |
| 13 | **DA3 기술 보고서 (PDF)** | https://depth-anything-3.github.io/assets/da3_tech_report_2025.pdf |
| 14 | **Hugging Face Model Card** | https://huggingface.co/depth-anything/DA3-LARGE |

> ⚠️ **정확도 주의 사항**: 본 답변은 공개된 arXiv 논문, 공식 프로젝트 페이지, GitHub 레포지토리, 기술 보고서 등 검증된 1차 출처를 기반으로 작성되었습니다. 수식 세부 사항(손실 함수의 가중치, 특정 하이퍼파라미터 등)은 전문 PDF를 직접 확인하시기를 권장합니다. 논문은 ICLR 2026에서 최종 버전이 공개될 예정입니다.
