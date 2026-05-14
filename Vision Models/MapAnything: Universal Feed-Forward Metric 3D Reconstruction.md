
# MapAnything: Universal Feed-Forward Metric 3D Reconstruction

> **논문 정보**
> - **제목**: MapAnything: Universal Feed-Forward Metric 3D Reconstruction
> - **저자**: Nikhil Keetha, Norman Müller, Johannes Schönberger, Lorenzo Porzi 외 13인 (Meta AI / CMU)
> - **arXiv**: [2509.13414](https://arxiv.org/abs/2509.13414) (2025년 9월 16일 초고 공개, 2026년 1월 최종 수정)
> - **발표**: IEEE 3DV 2026
> - **코드**: [github.com/facebookresearch/map-anything](https://github.com/facebookresearch/map-anything)
> - **모델**: [HuggingFace: facebook/map-anything](https://huggingface.co/facebook/map-anything)

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장 (TLDR)

MapAnything는 이미지, 카메라 보정(intrinsics), 포즈(poses), 깊이(depth) 등 다양한 종류의 입력을 받아 메트릭(metric) 3D 장면 기하구조와 카메라를 직접 회귀(regress)하는, 단순하고 end-to-end 학습된 통합 트랜스포머 기반 feed-forward 모델입니다.

MapAnything는 이미지, 카메라 내부 파라미터, 포즈, 깊이 맵, 또는 부분 재구성을 포함한 유연한 입력으로부터 단일 패스(single pass)에서 메트릭 3D 기하와 카메라 포즈를 직접 회귀하는 최초의 범용 트랜스포머 기반 백본입니다.

### 1.2 주요 기여 (4가지)

연구팀은 네 가지 주요 기여를 강조합니다: ① 12개 이상의 문제 설정(단안 깊이~SfM~스테레오)을 처리하는 **통합 feed-forward 모델**, ② 광선(rays), 깊이, 포즈, 메트릭 스케일의 명시적 분리를 가능하게 하는 **분인화 장면 표현(Factored Scene Representation)**, ③ 더 적은 중복성과 높은 확장성으로 다양한 벤치마크에서 **최첨단 성능(SoTA)**, ④ 데이터 처리, 훈련 스크립트, 벤치마크, 사전 학습 가중치를 포함한 **오픈소스 공개(Apache 2.0)**.

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

기존 접근법은 특징 감지/매칭, 2뷰 포즈 추정, 카메라 보정, 회전/이동 평균화, 번들 조정(BA), 다중 뷰 스테레오(MVS), 단안 표면 추정 등 **별개의 작업으로 문제를 분해**했습니다. 최근 feed-forward 아키텍처를 사용한 통합 접근이 주목받고 있지만, 기존 feed-forward 연구들은 **서로 다른 태스크를 분리해서 접근하거나 가용한 모든 입력 양식(modality)을 활용하지 않는 문제**가 있었습니다.

MapAnything는 가장 일반적인 비보정(uncalibrated) SfM 문제뿐만 아니라 보정된 SfM, 다중 뷰 스테레오, 단안 깊이 추정, 카메라 위치추정(localization), 메트릭 깊이 완성(depth completion) 등 다양한 하위 문제들의 조합 모두를 단일 모델로 해결하는 것을 목표로 합니다.

---

### 2.2 제안하는 방법: 분인화 표현(Factored Representation)

#### 핵심 표현 방식

MapAnything는 장면을 포인트맵(pointmap)의 집합으로 직접 표현하는 대신, **깊이 맵(depth maps), 로컬 레이맵(local ray maps), 카메라 포즈(camera poses), 메트릭 스케일 인자(metric scale factor)**의 집합으로 표현합니다. 이 분인화 표현은 출력과 (선택적) 입력 모두에 사용되어, 보조 기하 입력이 있을 때 이를 활용할 수 있습니다.

수학적으로, 각 뷰 $i$에 대한 3D 포인트 $\mathbf{P}_i^{(w)}$ (월드 좌표계)는 다음과 같이 분인화됩니다:

$$
\mathbf{P}_i^{(w)} = s \cdot \mathbf{T}_i \cdot \left( d_i \cdot \mathbf{r}_i \right)
$$

여기서:
- $\mathbf{r}_i \in \mathbb{R}^{H \times W \times 3}$: 로컬 레이 방향(카메라 내부 파라미터로부터 유도)
- $d_i \in \mathbb{R}^{H \times W}$: 레이 방향 기준 깊이 (up-to-scale)
- $\mathbf{T}_i \in SE(3)$: 카메라 포즈 (view 1 기준 프레임)
- $s \in \mathbb{R}^+$: 글로벌 메트릭 스케일 인자

이 분인화 표현의 중요한 이점은 MapAnything이 **부분적 어노테이션이 있는 다양한 데이터셋으로부터 효과적으로 학습**될 수 있다는 점입니다. 예를 들어, 비메트릭(up-to-scale) 기하만 어노테이션된 데이터셋도 활용 가능합니다.

#### 학습 방법: 입력 증강(Input Augmentation)

훈련 전략으로는 세 가지가 핵심입니다: **확률적 입력 드롭아웃(probabilistic input dropout)** — 훈련 중 광선, 깊이, 포즈 등 기하 입력을 다양한 확률로 제공하여 이질적인 구성에 대한 강건성 확보; **공시성 기반 샘플링(covisibility-based sampling)** — 입력 뷰가 의미 있는 중첩(overlap)을 갖도록 하여 100개 이상의 뷰도 지원; **로그 공간에서의 분인화 손실(factored losses in log-space)** — 깊이, 스케일, 포즈에 스케일 불변 및 강건 회귀 손실을 적용하여 안정성 향상.

손실 함수는 다음 세 가지 요소로 구성됩니다:

$$
\mathcal{L}_{\text{total}} = \lambda_d \mathcal{L}_{\text{depth}} + \lambda_p \mathcal{L}_{\text{pose}} + \lambda_s \mathcal{L}_{\text{scale}}
$$

- $\mathcal{L}_{\text{depth}}$: 로그 공간 스케일 불변 깊이 손실 (scale-invariant log depth loss)
- $\mathcal{L}_{\text{pose}}$: 쿼터니언 회전에 대한 지오데식 손실(Geodesic Loss) + 이동 손실
- $\mathcal{L}_{\text{scale}}$: 메트릭 스케일 회귀 손실

특히 포즈 손실에는 **쿼터니언 회전의 2-대-1 매핑을 처리하기 위한 지오데식 손실**이 사용됩니다.

로그 스케일 기반 설계의 중요성은 ablation으로 입증되었습니다:

Table S.3에서 보여지듯이, **로그 스케일링과 교차 어텐션(alternating attention)은 훈련 시 사용한 4개 뷰를 훨씬 초과한 50개 뷰로 평가할 때 강력한 성능을 위한 핵심 설계 선택**임이 확인되었습니다.

---

### 2.3 모델 구조

모델은 $N$개의 시각 및 선택적 기하 입력을 받아, 이미지와 기하 입력의 분인화 표현을 공통 잠재 공간으로 인코딩합니다. 패치 특징(이미지, 레이, 깊이)과 브로드캐스트 글로벌 특징(이동, 회전, 포즈 스케일 등)이 합산되고, 고정된 참조 뷰 임베딩이 첫 번째 뷰의 특징에 추가되며, 단일 학습 가능 스케일 토큰이 $N$개의 뷰 패치 토큰 집합에 추가됩니다. 이 토큰들은 교차-어텐션 트랜스포머(alternating-attention transformer)에 입력됩니다. 단일 DPT(Dense Prediction Transformer)가 $N$개의 밀도 출력을 디코딩합니다.

예측이 up-to-scale 공간에서 존재하는 동안, 모델은 스케일 토큰을 MLP를 통해 전달하여 **메트릭 스케일링 인자를 예측**하고, 이를 다른 예측과 결합하여 밀도 있는 메트릭 3D 재구성을 제공합니다.

구체적인 아키텍처 세부사항:

MapAnything는 병렬 이미지 및 기하 인코더를 갖는 모듈식 트랜스포머 백본 기반입니다. 시각 입력($N$개 이미지)은 사전 학습된 **DINOv2 ViT-L**로 인코딩되어 패치 단위 특징 맵을 생성합니다. 기하 보조 입력(가용 시)은 얕은 CNN 또는 MLP 브랜치를 통해 별도 인코딩됩니다. 입력들은 학습 가능한 스케일 토큰을 포함하여 토큰으로 융합됩니다. 메인 트랜스포머(24레이어, 교차 어텐션)가 모든 뷰와 보조 입력에 걸쳐 조인트 토큰을 처리합니다. 디코딩은 깊이와 로컬 레이를 위한 뷰별 밀도 헤드와, 쿼터니언 및 이동을 위한 포즈 헤드로 분인화됩니다. 별도의 MLP가 스케일 토큰을 처리하여 글로벌 메트릭 스케일을 회귀하고, 이를 통해 모든 up-to-scale 출력이 메트릭 재구성으로 업그레이드됩니다.

전체 구조를 다이어그램으로 나타내면:

```
입력: [이미지 × N] + [선택: 내부 파라미터, 포즈, 깊이]
        │                      │
   DINOv2 ViT-L           CNN/MLP 인코더
        │                      │
        └──────── 융합 (패치 토큰 + 글로벌 특징) ────────┐
                                                        + [스케일 토큰]
                                                              │
                              교차-어텐션 트랜스포머 (24레이어)
                              (뷰 내부: Self-Attn ↔ 뷰 간: Cross-Attn)
                                                              │
                    ┌──────────────────────────────────────────┐
                    │                                          │
              DPT 디코더 (× N뷰)                        MLP (스케일 토큰)
          깊이 맵 + 레이맵 + 신뢰도                     메트릭 스케일 s
                    │                                          │
                    └─────────────── 결합 ─────────────────────┘
                                      │
                       밀도 메트릭 3D 재구성 (pts3d, 카메라 포즈)
```

기하 입력을 6개 인자로 분인화함으로써 64가지 완전한 입력 조합을 지원합니다. 입력 양식이 모든 뷰에 대해 제공되는 경우를 주로 벤치마킹하지만, MapAnything는 입력 뷰의 일부에 대한 선택적 기하 입력도 지원합니다.

---

### 2.4 훈련 데이터

MapAnything는 BlendedMVS, Mapillary Planet-Scale Depth, ScanNet++, TartanAirV2 등을 포함하는 **13개의 다양한 데이터셋(실내, 실외, 합성 도메인)**에 걸쳐 훈련되었습니다. 두 가지 변형이 공개됩니다: 6개 데이터셋으로 훈련된 Apache 2.0 라이선스 모델과, 더 강력한 성능을 위해 13개 전체 데이터셋으로 훈련된 CC BY-NC 모델.

훈련은 혼합 정밀도(mixed precision), 그래디언트 체크포인팅, 4개에서 24개 입력 뷰로 스케일링하는 커리큘럼 스케줄링과 함께 **64개 H200 GPU**에서 수행되었습니다.

---

### 2.5 성능 향상

광범위한 실험에서 MapAnything는 다양한 태스크에서 전문 feed-forward 모델과 동등하거나 능가함을 보여줍니다: 다중 뷰 밀도 재구성에서 전용 스테레오 시스템보다 낮은 절대 상대 오차를 달성하고, 2뷰 설정에서 기하 보조 입력(내부 파라미터, 포즈, 깊이) 추가 시 오차가 추가로 감소하며, 단일 뷰 카메라 보정에서 SoTA 각도 예측 오차를 달성하고, ETH3D, ScanNet, KITTI 깊이 벤치마크에서 부분적 입력 양식으로도 강건한 성능을 보입니다.

MapAnything는 다중 뷰 메트릭 깊이 추정에서 새로운 SoTA를 설정합니다. 보조 입력을 사용하면 오차율이 MVSA 및 Metric3D v2 같은 전문 깊이 모델과 동등하거나 능가합니다. 전반적으로 벤치마크에서 많은 태스크에서 기존 SoTA 방법 대비 **2배 향상**을 확인하여, 통합 훈련의 이점을 검증합니다.

---

### 2.6 한계점

MapAnything의 주요 한계 및 미래 방향으로 논문은 다음을 언급합니다: **(a)** 기하 입력의 노이즈나 불확실성을 명시적으로 고려하지 않음; **(b)** 모든 입력 뷰에 이미지가 없는 태스크(예: 새로운 뷰 합성에서 타겟 뷰는 카메라만 입력으로 가짐)를 아직 지원하지 않지만, 아키텍처는 쉽게 확장 가능; **(c)** 반복적 추론(iterative inference)을 지원하는 설계이지만, 테스트 타임 연산 스케일링의 효과는 아직 탐구되지 않음; **(d)** 현재 다중 양식 특징은 입력 전 융합되지만, 서로 다른 양식을 트랜스포머에 직접 효율적으로 입력하는 방법은 탐구할 필요가 있음.

---

## 3. 일반화 성능 향상 가능성

MapAnything의 일반화 성능 향상은 다음 메커니즘에서 비롯됩니다:

### 3.1 분인화 표현의 일반화 이점

분인화 표현의 중요한 이점은 MapAnything가 부분 어노테이션이 있는 **다양한 데이터셋으로부터 효과적으로 학습**될 수 있다는 것입니다. 예를 들어, 비메트릭(up-to-scale) 기하만 어노테이션된 데이터셋도 활용 가능합니다.

이를 수식으로 표현하면, 스케일 인자 $s$가 알려지지 않은 경우(up-to-scale 데이터셋):

$$
\hat{d}_i = \frac{d_i}{\bar{d}}, \quad \hat{s} = 1 \quad \text{(up-to-scale 모드)}
$$

메트릭 스케일이 알려진 경우:

$$
\mathbf{P}^{(w)}_i = s \cdot \mathbf{T}_i \cdot (d_i \cdot \mathbf{r}_i), \quad s = \text{MLP}(z_{\text{scale}})
$$

### 3.2 유연한 입력 증강

MapAnything의 입력 구성의 유연성은 Table S.1에서 잘 드러나는데, **더 많은 모달리티가 제공될수록 성능이 향상**됩니다.

또한 객체 중심(object-centric) 데이터로 훈련되지 않았음에도, 제공된 입력에 따라 장면 기하와 카메라가 어떻게 변화하는지를 보여주는 **제로샷(zero-shot) 일반화 능력**을 시연합니다.

### 3.3 확장성 및 뷰 수에 대한 일반화

훈련 시에는 **공시성 기반 뷰 샘플링, 보조 인자에 대한 확률적 입력 선택, 공격적인 증강**을 사용하여 누락되거나 불완전한 입력에 대한 강건한 처리를 가능하게 합니다. 감독 신호는 모든 데이터셋에 걸쳐 up-to-scale, 깊이 전용, 포즈 전용 데이터 어노테이션을 수용하도록 조화되었습니다.

Apache 및 첫 번째 단계(최대 4뷰) 훈련 변형 모두 **2개에서 100개까지 다양한 입력 뷰 수**에 대해 강력한 밀도 다중 뷰 재구성을 보여줍니다.

### 3.4 모듈식 파인튜닝으로 일반화 확장

MapAnything 프레임워크의 모듈성을 활용하여 MoGe-2, VGGT, π³ 등 다른 기하 추정 모델의 파인튜닝을 지원하며, 이는 **다양한 도메인에 대한 특화 모델로의 확장성**을 보여줍니다.

### 3.5 우주 환경 등 극단적 도메인 이전(Domain Shift) 한계

실제 운용 시 고려할 점으로, MapAnything과 같은 범용 feed-forward 모델은 **심각한 도메인 이전(domain shift)에 직면할 때 기하 복원 능력이 크게 저하**됩니다. 이를 해결하기 위해 In-Orbit MapAnything과 같은 환경 특화 변형이 제안되기도 하였습니다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 Feed-Forward 3D 재구성 패러다임의 흐름

3D 재구성 분야는 '반복적 최적화'에서 'end-to-end 추론'으로의 전환이 이루어졌습니다. DUSt3R의 등장 이후 이 분야에서는 짧은 기간 안에 관련 연구가 폭발적으로 증가했습니다. 핵심 대응 품질 향상(MASt3R, VGGT), 다중 뷰 일관성 해결(Align3R, Pow3R), 실시간 SLAM(SLAM3R), 자율 주행(Driv3R), 시각적 재지역화(Reloc3r) 등을 대상으로 하는 연구들이 빠르게 기술 생태계를 형성하고 있습니다.

| 모델 | 연도 | 핵심 특징 | MapAnything와 차이점 |
|---|---|---|---|
| **DUSt3R** | CVPR 2024 | 최초 통합 feed-forward 재구성, 포인트맵 기반 | 비보정 2뷰 중심, 메트릭 스케일 미지원 |
| **MASt3R** | ECCV 2024 | DUSt3R + 매칭 품질 강화, 특징 헤드 추가 | SfM에 특화, 단일 태스크 |
| **MUSt3R** | CVPR 2025 | 다중 뷰 스테레오 특화 확장 | 다중 뷰 스테레오에 집중 |
| **VGGT** | CVPR 2025 | 단일 패스 다중 뷰 재구성, 계산 효율성 | 고정 입력 양식, 보조 기하 입력 미지원 |
| **Pow3R** | CVPR 2025 | 카메라/장면 사전 정보 통합 | 특정 사전 정보 타입에 제한적 |
| **MapAnything** | 3DV 2026 | 12+ 태스크 통합, 유연한 멀티모달 입력 | 위 모든 특성을 단일 모델로 통합 |

MASt3R와 VGGT는 더욱 정교한 멀티스케일 특징 융합 전략을 통해 와이드 베이스라인 및 다양한 해상도 시나리오에서 매칭 정확도를 향상시키는 데 집중합니다.

그러나 DUSt3R/MASt3R/VGGT 계열 모두 **고해상도 이미지와 대규모 세트에서 한계를 보이며, 이미지가 많아질수록 포즈 신뢰성이 저하**됩니다. 이러한 결과들은 트랜스포머 기반 방법이 전통적인 SfM과 MVS를 완전히 대체할 수 없지만, 특히 도전적이고 저해상도, 희소한 시나리오에서 상호 보완적 접근으로 큰 잠재력이 있음을 시사합니다.

MapAnything 프레임워크는 데이터 처리, 훈련, 추론, 프로파일링을 포함하는 **완전한 스택을 제공**하며, VGGT, DUSt3R, MASt3R, MUSt3R, Pi3-X 등 다양한 3D 재구성 모델을 통합 인터페이스를 통해 상호 교환하여 사용할 수 있는 모듈식 설계를 지원합니다.

---

## 5. 향후 연구에 미치는 영향 및 고려해야 할 점

### 5.1 향후 연구에 미치는 영향

MapAnything는 범용적이고 메트릭 3D 장면 재구성을 위한 공식적인 프레임워크를 확립하며, feed-forward 트랜스포머 아키텍처와 유연하고 표준화된 멀티모달 입출력 처리를 연결하여 **범용 3D 비전 시스템의 미래 발전을 위한 기반**을 마련합니다.

동적 장면(dynamic scenes), 불확실성 정량화(uncertainty quantification), 장면 이해(scene understanding)로의 미래 확장은 MapAnything의 능력과 강건성을 더욱 일반화할 수 있을 것으로 기대됩니다.

MapAnything는 범용 3D 재구성 백본 개발을 향한 중요한 발전을 나타냅니다. 태스크별 적응 없이 여러 3D 비전 태스크를 효율적으로 처리하며, 모델의 확장성은 동적 장면 재구성 및 향상된 불확실성 모델링을 포함한 미래 연구 방향을 지원합니다.

### 5.2 향후 연구 시 고려할 점

다음 네 가지를 중점적으로 고려해야 합니다:

**① 불확실성 정량화 (Uncertainty Quantification)**
MapAnything는 **기하 입력의 노이즈나 불확실성을 명시적으로 고려하지 않습니다.** 이는 실제 환경에서 센서 노이즈가 있는 깊이 데이터나 부정확한 포즈 정보가 입력될 때 성능 저하로 이어질 수 있어, 베이지안 프레임워크나 conformal prediction 등을 통한 불확실성 추정이 향후 중요한 연구 방향입니다.

**② 테스트 타임 연산 스케일링 (Test-Time Compute Scaling)**
MapAnything의 설계는 반복적 추론을 지원하지만, **3D 재구성에서 테스트 타임 연산 스케일링이 얼마나 효과적인지는 아직 탐구되지 않았습니다.** 이는 입력 노이즈를 효과적으로 처리하는 방법과도 연관됩니다.

**③ 멀티모달 직접 입력 (Direct Multi-Modal Input)**
현재 멀티모달 특징은 **입력 전에 융합되는데**, 서로 다른 모달리티를 트랜스포머에 효율적으로 직접 입력하는 방법을 탐구하면 모달리티 간 상호작용을 보다 세밀하게 제어할 수 있을 것입니다.

**④ 도메인 이전 강건성 (Domain Transfer Robustness)**
우주 환경 실험에서 보여지듯, **범용 feed-forward 모델은 심각한 도메인 이전에 직면할 때 기하 복원 능력이 크게 저하**됩니다. 따라서 훈련 데이터와 크게 다른 도메인(우주, 수중, 의료 영상 등)에서의 적용 시 파인튜닝 전략이나 도메인 적응(domain adaptation) 기법을 함께 고려해야 합니다.

---

## 📚 참고 자료 및 출처

| # | 자료명 | URL/출처 |
|---|---|---|
| 1 | MapAnything: Universal Feed-Forward Metric 3D Reconstruction (arXiv) | [arxiv.org/abs/2509.13414](https://arxiv.org/abs/2509.13414) |
| 2 | MapAnything 공식 프로젝트 페이지 | [map-anything.github.io](https://map-anything.github.io/) |
| 3 | MapAnything GitHub 공식 리포지토리 | [github.com/facebookresearch/map-anything](https://github.com/facebookresearch/map-anything) |
| 4 | MapAnything HuggingFace 모델 | [huggingface.co/facebook/map-anything](https://huggingface.co/facebook/map-anything) |
| 5 | MapAnything HTML 논문 (arXiv v1) | [arxiv.org/html/2509.13414v1](https://arxiv.org/html/2509.13414v1) |
| 6 | MapAnything ResearchGate (PDF) | [researchgate.net/publication/395583122](https://www.researchgate.net/publication/395583122_MapAnything_Universal_Feed-Forward_Metric_3D_Reconstruction) |
| 7 | MarkTechPost: Meta AI MapAnything 해설 | [marktechpost.com](https://www.marktechpost.com/2025/09/17/meta-ai-researchers-release-mapanything-an-end-to-end-transformer-architecture-that-directly-regresses-factored-metric-3d-scene-geometry/) |
| 8 | EmergentMind: MapAnything 요약 | [emergentmind.com/topics/mapanything](https://www.emergentmind.com/topics/mapanything) |
| 9 | Review of Feed-forward 3D Reconstruction: From DUSt3R to VGGT (arXiv:2507.08448) | [arxiv.org/abs/2507.08448](https://arxiv.org/abs/2507.08448) |
| 10 | An Evaluation of DUSt3R/MASt3R/VGGT 3D Reconstruction (arXiv:2507.14798) | [arxiv.org/abs/2507.14798](https://arxiv.org/abs/2507.14798) |
| 11 | In-Orbit MapAnything (MDPI Sensors, 2026) | [mdpi.com/1424-8220/26/7/2026](https://www.mdpi.com/1424-8220/26/7/2026) |
| 12 | Semantic Scholar: MapAnything 페이지 | [semanticscholar.org](https://www.semanticscholar.org/paper/MapAnything:-Universal-Feed-Forward-Metric-3D-Keetha-Muller/613d2e430e097e8c83f9b59bde5de9e2fc86a098) |
| 13 | OpenReview: MapAnything | [openreview.net/forum?id=h9J2UUVWat](https://openreview.net/forum?id=h9J2UUVWat) |

> ⚠️ **정확도 관련 주의사항**: 본 답변의 수식 세부 표기(특히 손실 함수 가중치 $\lambda$ 등)는 논문 원문의 공개된 arXiv HTML 및 GitHub 정보를 기반으로 재구성한 것으로, 논문 본문 내 정확한 수식 표기와 세부 계수는 원문([arxiv.org/pdf/2509.13414](https://arxiv.org/pdf/2509.13414))을 직접 확인하시길 권장합니다.
