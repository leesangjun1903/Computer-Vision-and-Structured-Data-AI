
# Video Depth without Video Models

> **논문 정보**
> - **제목**: Video Depth without Video Models
> - **모델명**: RollingDepth
> - **저자**: Bingxin Ke, Dominik Narnhofer, Shengyu Huang, Lei Ke, Torben Peters, Katerina Fragkiadaki, Anton Obukhov, Konrad Schindler
> - **학회**: CVPR 2025
> - **arXiv**: [2411.19189](https://arxiv.org/abs/2411.19189)
> - **프로젝트 페이지**: [rollingdepth.github.io](https://rollingdepth.github.io/)
> - **코드**: [github.com/prs-eth/RollingDepth](https://github.com/prs-eth/RollingDepth)

---

## 1. 핵심 주장 및 주요 기여 요약

### 🎯 핵심 주장

단일 이미지 Latent Diffusion Model(LDM)을 최첨단 비디오 깊이 추정기(video depth estimator)로 전환할 수 있음을 보인다.

즉, **비디오 모델 없이도 비디오 깊이 추정이 가능하다**는 것이 논문의 핵심 주장이다.

### 🏆 주요 기여

| 기여 항목 | 설명 |
|---|---|
| Multi-frame Snippet 추정기 | 단일 이미지 LDM을 확장하여 짧은 스니펫(frame triplet 등)에서 깊이 추정 |
| Depth Co-alignment 알고리즘 | 최적화 기반 등록(registration)으로 전역 일관성 확보 |
| 장거리 비디오 처리 | 수백 프레임의 긴 비디오도 효율적으로 처리 가능 |
| 제로샷 일반화 | 전용 비디오 모델보다 뛰어난 zero-shot 벤치마크 성능 달성 |

본 접근법은 복잡한 비디오 확산 모델에 의존하지 않고 정확하고 시간적으로 일관된 비디오 깊이를 추정한다.

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

비디오 깊이 추정은 단안 비디오 클립의 모든 프레임에서 밀집 깊이를 추론하여 3D로 리프팅하는 것이다. 대형 기반 모델과 합성 훈련 데이터의 활용으로 단일 이미지 깊이 추정이 크게 발전했지만, 단일 이미지 깊이 추정기를 비디오의 매 프레임에 단순 적용하면 시간적 연속성을 무시하여 플리커링이 발생하고, 카메라 움직임으로 인한 깊이 범위의 급격한 변화에도 취약하다.

비디오 기반 모델을 사용하는 것이 명백한 해결책이지만, 이러한 모델들은 고비용 학습/추론, 불완전한 3D 일관성, 고정 길이(짧은) 출력의 stitching 루틴 등의 한계를 지닌다.

비디오 LDM은 계산 비용이 클 뿐만 아니라 고정된 짧은 시퀀스 길이로 학습되어 다양한 길이의 영상에 직접 적용할 수 없으며, 비디오를 분할-처리-재결합하는 파이프라인은 저주파 플리커링과 점진적 드리프트를 유발한다.

---

### 2-2. 제안하는 방법

#### 전체 파이프라인 개요

RollingDepth는 두 가지 핵심 구성요소를 가진다: (i) 단일 이미지 LDM에서 파생된 멀티프레임 깊이 추정기로, 매우 짧은 비디오 스니펫(일반적으로 프레임 트리플렛)을 깊이 스니펫으로 매핑한다. (ii) 다양한 프레임 레이트로 샘플링된 깊이 스니펫을 일관된 비디오로 최적적으로 조합하는 강건한 최적화 기반 등록 알고리즘이다.

#### Step 1: Snippet 샘플링 (Dilated Rolling Kernel)

비디오 시퀀스 $\mathbf{x}$가 주어지면, 다양한 dilation rate를 가진 dilated rolling kernel을 사용해 $N_T$개의 겹치는 스니펫을 구성하고, 1-step 추론으로 초기 깊이 스니펫을 얻는다. 다음으로, depth co-alignment가 $N_T$쌍의 스케일과 시프트 값을 최적화하여 전체 비디오에 걸쳐 전역적으로 일관된 깊이를 달성한다.

즉, dilation rate $d$로 샘플링된 스니펫 $S_d$는 다음과 같이 표현된다:

$$S_d = \{x_{t}, x_{t+d}, x_{t+2d}, \ldots, x_{t+(n-1)d}\}$$

여기서 $n \ll N_F$ (전체 프레임 수), $d \in \{1, 10, 25\}$ (fast 설정 시 $\{1, 25\}$).

#### Step 2: 1-Step LDM 기반 깊이 추정

원래 Marigold 모델은 이미지별 near/far 플레인 사이의 (아핀-불변) 깊이를 예측했으나, 깊이 범위가 시간에 따라 변하는 비디오 깊이 추정에서의 문제를 해결하기 위해 역깊이(inverse depth) 예측 방식으로 재학습했다.

멀티프레임 처리를 위한 **Cross-Frame Self-Attention** 메커니즘이 핵심이다. 수정된 diffusers 라이브러리에 cross-frame self-attention이 적용되어 있다.

#### Step 3: Depth Co-Alignment (Scale-Shift 최적화)

스니펫들이 독립적으로 처리되므로 각각 고유의 스케일과 시프트를 가진다. 이 모호성을 해결하기 위해 다른 dilation rate의 겹치는 스니펫들을 구성한다.

전역 일관성을 위한 최적화 목표를 수식으로 나타내면:

$$\min_{\{a_t, b_t\}_{t=1}^{N_T}} \sum_{(i,j) \in \mathcal{O}} \left\| (a_i \cdot \hat{d}_i^{(f)} + b_i) - (a_j \cdot \hat{d}_j^{(f)} + b_j) \right\|^2$$

여기서:
- $\hat{d}_t^{(f)}$: 스니펫 $t$의 $f$번 프레임에서 예측된 깊이
- $a_t, b_t$: 각 스니펫의 스케일/시프트 파라미터
- $\mathcal{O}$: 겹치는 프레임 쌍의 집합

Depth co-alignment는 $N_T$쌍의 스케일과 시프트 값을 최적화하여 전체 비디오에 걸쳐 전역적으로 일관된 깊이를 달성한다.

#### Step 4: 선택적 정제(Optional Refinement)

선택적으로 결과 비디오에 적당한 임의 노이즈를 추가하고 동일한 per-snippet LDM으로 다시 디노이징하여 공간적 세부 정보를 더욱 정제한다.

---

### 2-3. 모델 구조

```
입력 비디오 (임의 길이 NF 프레임)
         │
         ▼
┌─────────────────────────────┐
│  Dilated Rolling Kernel      │
│  dilation rates {1, 10, 25} │
│  → NT개의 짧은 스니펫 구성   │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Snippet LDM (1-step 추론)  │
│  (Marigold 기반 역깊이 모델) │
│  + Cross-Frame Self-Attention│
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Depth Co-Alignment          │
│  (Scale & Shift 최적화)      │
│  → 전역 일관된 깊이 맵       │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Optional Refinement         │
│  (추가 디노이징 정제)        │
└─────────────────────────────┘
         │
         ▼
출력: 시간적으로 일관된 깊이 비디오
```

스니펫 LDM 파인튜닝에는 TartanAir 합성 비디오 데이터셋(다양한 실내외 장면, 스타일, 카메라 모션)이 사용된다.

---

### 2-4. 성능 향상

RollingDepth는 6개의 최신 zero-shot 단안 깊이 추정 방법(Marigold, DepthAnything, DepthAnythingV2, NVDS, ChronoDepth, DepthCrafter)과 비교하여 여러 데이터셋과 다양한 시퀀스 길이에 걸쳐 단일 프레임 및 비디오 기반 접근법 모두를 능가하며, 이미지 기반 모델의 정확성과 스니펫 추론 및 전역 깊이 co-alignment를 통한 시간적 일관성을 결합한 덕분이다.

특히 깊이 범위 변화가 심한 장면이 포함된 PointOdyssey에서 RollingDepth가 압도적인 최고 성능을 달성했으며, 비디오 모델 기반 방법들은 단일 프레임 방법의 성능조차 따라가지 못했다.

PointOdyssey, ScanNet, Bonn, DyDToF 등 정적·동적 장면을 포함한 데이터셋에서 실험이 수행되었으며, RollingDepth는 단일 프레임 모델과 다른 비디오 기반 기법 모두를 정확도와 시간적 일관성 측면에서 능가했다.

논문은 절대 상대 오차(AbsRel)와 $\delta_1$ 정확도 지표를 통한 정량적 비교를 제시한다.

---

### 2-5. 한계점

1. **추론 비용**: 다양한 dilation rate의 여러 스니펫을 처리하므로 단순 단일 프레임 추론보다 연산 비용이 높다.
2. **Scale-Shift 제약**: 스케일과 시프트만으로 표현되는 아핀-불변(affine-invariant) 정렬은 복잡한 조명 변화나 급격한 장면 전환에 취약할 수 있다.
3. **훈련 데이터 한계**: 합성 데이터(TartanAir)에 주로 의존하므로 실세계 다양성 커버리지에 제약이 있다.
4. **메트릭 깊이 불가**: 아핀-불변 역깊이를 출력하므로 절대적 미터 단위 깊이 추정에는 추가 처리가 필요하다.
5. 현재 LDM 기반 비디오 깊이 모델들은 원거리 장면 부분에서 정확도가 떨어지는 경향이 있다.

---

## 3. 일반화 성능 향상 가능성

### 3-1. 강력한 제로샷 일반화의 원천

최근 방법들은 DINOv2나 Stable Diffusion과 같은 인터넷 규모 데이터로 학습된 기반 모델에 의존하며, 대규모로 생성 가능한 합성 RGB+깊이 이미지 쌍으로 깊이 추정에 파인튜닝한다. 이러한 풍부한 시각 사전 정보는 이 깊이 추정기에 장면 유형, 이미징, 조명 조건 전반에 걸쳐 탁월한 제로샷 일반화를 제공한다.

Marigold(RollingDepth의 기반)의 핵심 원칙은 현대 생성 이미지 모델에 저장된 풍부한 시각 지식을 활용하는 것이며, Stable Diffusion에서 파생되고 합성 데이터로 파인튜닝된 이 모델은 미경험 데이터에 제로샷으로 전이할 수 있다.

### 3-2. 비디오 도메인에서의 일반화 우위

RollingDepth는 이미지 기반 모델임에도 불구하고 단/장 비디오 시퀀스 모두에서 뛰어난 성능을 보인다.

기존 비디오 모델 대비 RollingDepth의 일반화 강점은 다음과 같다:

1. **데이터 독립성**: 비디오 전용 학습 데이터 없이 이미지 LDM의 사전 정보 활용
2. **길이 독립성**: RollingDepth는 수백 프레임의 긴 비디오를 효율적으로 처리할 수 있다.
3. **깊이 범위 적응**: 역깊이(inverse depth) 예측으로 카메라 움직임에 따른 깊이 범위 변화에 강건

### 3-3. 일반화 향상을 위한 설계 원칙

다양한 dilation rate를 사용한 스니펫 샘플링:

$$d \in \{1, 10, 25\}$$

는 단기적(빠른 움직임)·장기적(느린 움직임) 시간 컨텍스트를 동시에 포착함으로써 다양한 비디오 유형에 대한 일반화를 가능하게 한다.

---

## 4. 최신 연구 비교 분석 (2020년 이후)

### 주요 비디오/이미지 깊이 추정 방법 비교표

| 방법 | 연도 | 유형 | 기반 모델 | 장점 | 단점 |
|---|---|---|---|---|---|
| **MiDaS** | 2020 | 이미지 | DPT | 다중 데이터셋 혼합 학습, 범용성 | 세부 묘사 부족 |
| **NVDS** | 2023 | 비디오 | DPT | 비디오 안정성 모듈 추가 | 정확도 한계 |
| **Marigold** | 2024 | 이미지 | StableDiffusion | 고품질 세부 묘사, 제로샷 우수 | 느린 추론 속도 |
| **DepthAnything V2** | 2024 | 이미지 | DINOv2 | 빠른 속도, 높은 정확도 | 시간 일관성 없음 |
| **ChronoDepth** | 2024 | 비디오 | SVD | 시간 일관성 우수 | 짧은 시퀀스 한계 |
| **DepthCrafter** | 2024 | 비디오 | SVD | 오픈월드 비디오 지원 | 고비용, 느린 속도 |
| **DepthAnyVideo** | 2024 | 비디오 | FlowMatch | 빠른 추론, 고정확 | 192프레임 한계 |
| **RollingDepth** | 2025 | 비디오 | StableDiffusion | 이미지 모델로 비디오 처리, 제한 없는 길이 | 스케일 절대값 불가 |
| **Video Depth Anything** | 2025 | 비디오 | DepthAnythingV2 | 초장거리 비디오, SOTA | 학습 데이터 의존 |

비디오 확산 모델인 ChronoDepth, DepthCrafter, DepthAnyVideo는 더 나은 세부 묘사와 시간적 일관성을 보인다.

그러나 이들은 느린 추론 속도와 방대한 비디오 깊이 학습 데이터를 요구하며, 훈련 시 최대 윈도우 길이 내에서만 테스트됨으로써 윈도우 간 플리커링과 성능 저하가 발생한다.

DepthAnyVideo는 단 3회 디노이징 스텝, 1422.8M 파라미터, 0.37초/추론으로 $\delta_1$ 96.1%를 달성한다.

---

## 5. 연구에 미치는 영향 및 향후 연구 고려점

### 5-1. 향후 연구에 미치는 영향

#### 🔬 패러다임 전환
이 논문은 **"비디오 처리에 비디오 모델이 반드시 필요한가?"** 라는 근본적 질문을 던진다. 이미지 기반 모델의 강력한 사전 정보를 최소한의 아키텍처 수정으로 비디오 도메인에 전이하는 접근법은 다른 비디오 이해 태스크(광학 흐름, 표면 법선, 세그멘테이션 등)에도 영감을 줄 수 있다.

#### 📐 스니펫 기반 처리의 일반화
Dilated rolling kernel + co-alignment 프레임워크는 시계열 데이터 처리의 새로운 패러다임으로, 강화학습, 의료 영상, 자율주행 시나리오에서도 적용 가능하다.

#### 🔗 기반 모델의 비디오 전이 가능성
기반 모델의 풍부한 시각 사전 정보가 탁월한 제로샷 일반화를 제공한다는 발견은, 향후 더 강력한 이미지 기반 모델(FLUX, SD3 등)을 활용한 비디오 깊이 추정 연구로 이어질 수 있다.

### 5-2. 향후 연구 시 고려할 점

#### ① 절대 메트릭 깊이 확장
현재 RollingDepth는 아핀-불변 역깊이를 출력하므로, 메트릭 깊이 모델은 카메라 파라미터를 포함한 데이터로 학습이 필요하여 가용 학습 데이터가 더 제한적이고 일반화 성능이 저하된다는 점을 고려해, 단안 카메라 캘리브레이션과의 결합 또는 학습 없는 스케일 추정 방법 개발이 필요하다.

#### ② 동적 객체 처리 강화
현재 co-alignment는 전역 스케일/시프트 최적화에 의존하므로, 카메라와 독립적으로 움직이는 동적 객체에 대한 국소적 일관성 처리가 약할 수 있다. 객체 단위 또는 영역 단위 정렬 전략 연구가 유망하다.

#### ③ 실시간 처리 최적화
빠른 추론 모드(dilation {1, 25}, fp16, 정제 없음)와 전체 정제 모드(dilation {1, 10, 25}, 10회 정제 스텝) 사이의 트레이드오프가 존재하므로, 경량화 및 지식 증류(knowledge distillation)를 통한 실시간 추론 연구가 필요하다.

#### ④ 다양한 합성 학습 데이터 확보
기존 합성 학습 데이터셋의 다양성 부족이 모델의 일반화 능력을 제한할 수 있다. 더 다양한 조명, 날씨, 동적 장면을 포함한 합성 데이터 생성 및 실세계 pseudo-label 기반 학습이 중요하다.

#### ⑤ 기반 모델 업그레이드
RollingDepth가 Marigold(Stable Diffusion 기반)를 사용하는 것처럼, 더 강력한 차세대 이미지 생성 기반 모델(예: SDXL, SD3)로의 교체를 통한 성능 향상 가능성을 탐색해야 한다.

#### ⑥ Video Depth Anything 등 후속 연구와의 비교
Video Depth Anything(VDA) 모델은 모든 장거리 비디오 데이터셋에서 SOTA 성능을 달성하고 있으며, RollingDepth와 같은 이미지 기반 접근 방식과 비디오 기반 discriminative 접근 방식의 융합이 향후 유망한 방향으로 보인다.

---

## 📚 참고자료 및 출처

| 구분 | 출처 |
|---|---|
| 논문 원문 | Ke, B. et al. "Video Depth without Video Models," CVPR 2025. arXiv:2411.19189 |
| 프로젝트 페이지 | https://rollingdepth.github.io/ |
| arXiv | https://arxiv.org/abs/2411.19189 |
| CVPR 2025 공식 페이지 | https://cvpr.thecvf.com/virtual/2025/poster/32677 |
| GitHub (공식 코드) | https://github.com/prs-eth/RollingDepth |
| HuggingFace 논문 페이지 | https://huggingface.co/papers/2411.19189 |
| IEEE Xplore | https://ieeexplore.ieee.org/document/11094885 |
| Moonlight 리뷰 | https://www.themoonlight.io/en/review/video-depth-without-video-models |
| ChronoDepth | Shao et al., "Learning Temporally Consistent Video Depth from Video Diffusion Priors," arXiv:2406.01493 |
| Depth Any Video | Yang et al., "Depth Any Video with Scalable Synthetic Data," arXiv:2410.10815 |
| Video Depth Anything | Chen et al., "Video Depth Anything: Consistent Depth Estimation for Super-Long Videos," CVPR 2025 |
| Marigold | Ke et al., "Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation," CVPR 2024. arXiv:2312.02145 |
| Survey on Monocular Metric Depth | arXiv:2501.11841 |
| Roboflow 깊이 추정 모델 비교 | https://blog.roboflow.com/depth-estimation-models/ |
