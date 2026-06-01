
# E³C: Video Generation with 3D Environmental Memory and Ego-Exo Human Pose Control 

> **⚠️ 주의**: 본 논문(arXiv:2605.26316)은 2025년 5월 말에 공개된 최신 논문으로, 검색으로 확보 가능한 정보(Abstract, HTML 전문 일부)를 최대한 활용하여 작성하였습니다. 수식 표현 중 일부는 Video Diffusion 분야의 일반적 관례를 기반으로 해석·서술하였음을 밝힙니다.

---

## 1. 핵심 주장 및 주요 기여 요약

E³C는 이고센트릭(egocentric) 비디오 생성을 위한 제어 가능한 비디오 디퓨전 프레임워크로, **지속적인 장면 구조(scene structure)** 와 **인간 행동(human dynamics)** 을 분리(disentangle)하는 구조화되고 압축된 조건(condition)을 구축합니다.

E³C의 핵심 아이디어는 **3D 환경 메모리(3D Environmental Memory)** 와 **Ego–Exo 인간 포즈 제어(Ego-Exo Human Pose Control)** 를 하나의 모델로 통합하여, 지속적인 장면 구조와 인간 행동을 분리하는 구조화된 조건을 만드는 것입니다.

### 주요 기여 요약

| 기여 항목 | 내용 |
|---|---|
| 3D 환경 메모리 | SLAM/SfM 기반 포인트 클라우드 + Video-VAE 외관 특징 |
| Ego 제어 | 카메라 착용자의 신체 포즈를 Ego Pose Encoding Token으로 제어 |
| Exo 제어 | 장면 내 타인의 동작을 3D 스켈레톤 렌더링으로 제어 |
| 구조 분리 | 정적 장면과 동적 인간 행동의 명시적 분리 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

이고센트릭 비디오 생성은 체화된 에이전트(embodied agent)가 자신과 타인의 행동이 세계에 어떻게 나타나고 변화하는지를 추론하는 데 필수적입니다. 특히, 카메라가 행동 주체와 밀접하게 연결되어 있어 급격한 시점 변화와 빈번한 자기-폐색(self-occlusion)이 발생하고, 사람과 장면 상태가 주어진 제어 신호에 일관되게 진화해야 하는 어려움이 있습니다.

특히 Nymeria와 같은 데이터셋은 빠른 이고센트릭 모션, 빈번한 자기-폐색, 복잡한 실내 장면으로 인해 일반 도메인 비디오 생성기에 심각한 분포 이동(distribution shift)을 야기합니다.

### 2-2. 제안 방법

#### (1) 3D 환경 메모리 구성

반밀집(semi-dense) 포인트 클라우드는 SLAM 또는 SfM(Structure-from-Motion) 알고리즘을 통해 이미지의 고-기울기(high-gradient) 영역에서 3D 포인트를 재구성하여 얻으며, 각 3D 포인트는 Video-VAE 특징에서 추출한 외관 디스크립터(appearance descriptor)로 보강됩니다.

수식으로 표현하면, 컨텍스트 프레임 $\{I_1, \ldots, I_T\}$로부터 다음과 같이 포인트 클라우드 메모리가 구성됩니다:

$$\mathcal{M} = \{(\mathbf{p}_i, \mathbf{f}_i)\}_{i=1}^{N}$$

여기서 $\mathbf{p}_i \in \mathbb{R}^3$은 3D 포인트 위치, $\mathbf{f}_i$는 Video-VAE 인코더에서 추출된 외관 특징 벡터입니다.

E³C는 컨텍스트 프레임에서 반밀집 포인트 클라우드 기반 3D 메모리를 구성하고 각 포인트에 Video-VAE 특징에서 추출한 외관 디스크립터를 부가합니다. 이 메모리를 목표 시점(target viewpoint)으로 렌더링하면 목표 프레임에 정렬된 조건(conditioning)이 생성됩니다.

#### (2) Ego-Exo 포즈 제어

실내 이고센트릭 장면에서의 역동성은 주로 인간에 의해 주도되므로, Ego–Exo 모션을 명시적으로 제어합니다. 타인의 동작(exo human dynamics)과 착용자 자신의 동작(ego human dynamics) 모두에 대해 동적 3D 스켈레톤을 조건화 카메라 시점으로 렌더링합니다. 착용자 자신의 신체는 이 시점에서 일부분만 관찰되기 때문에, **Ego Pose Encoding Token**을 도입하여 완전한(amodal) 이고센트릭 신체 동작을 포착합니다.

전체 조건화 신호를 수식으로 표현하면:

$$\mathbf{c} = \left[\mathbf{c}_{\text{scene}},\ \mathbf{c}_{\text{exo}},\ \mathbf{c}_{\text{ego}}\right]$$

- $\mathbf{c}_{\text{scene}}$: 포인트 클라우드를 렌더링한 장면 메모리 특징
- $\mathbf{c}_{\text{exo}}$: 타인의 3D 스켈레톤 렌더링
- $\mathbf{c}_{\text{ego}}$: Ego Pose Encoding Token (전신 amodal 포즈)

#### (3) 비디오 디퓨전 생성 과정

표준 비디오 디퓨전 모델 기반으로, 조건부 역방향 과정은 다음과 같이 정의됩니다:

$$p_\theta(\mathbf{x}_{0:T} \mid \mathbf{c}) = p(\mathbf{x}_T) \prod_{t=1}^{T} p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{c})$$

노이즈 제거 목표 함수 (DDPM/Score-matching):

$$\mathcal{L} = \mathbb{E}_{\mathbf{x}_0, t, \boldsymbol{\epsilon}} \left[\left\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\left(\mathbf{x}_t, t, \mathbf{c}\right)\right\|^2\right]$$

여기서 $\mathbf{c}$는 위에서 정의한 구조화된 조건 벡터이며, 장면 구조와 Ego-Exo 포즈가 모두 포함됩니다.

### 2-3. 모델 구조

파인튜닝된 VACE 모델은 포인트 클라우드 조건이 제공될 때 강력한 성능을 회복하며, Exo 스켈레톤을 추가하면 Exo 제어 지표가 더욱 향상됩니다. E³C는 최고의 파인튜닝 VACE 기준 모델(points+exo) 대비 포인트에 외관 특징을 추가하고, 특징 인코더 브랜치(feature encoder branch)를 추가하며, 지속적인 Ego 인간 제어를 위한 Ego Pose Encoder를 사용합니다.

모델 구조 요약:

```
[Context Frames]
    ↓ SLAM/SfM
[Semi-dense Point Cloud]
    ↓ Video-VAE Feature Augmentation
[3D Scene Memory]  +  [Exo Skeleton Rendering]  +  [Ego Pose Encoding Tokens]
    ↓ Feature Encoder Branch
[Structured Condition c]
    ↓
[Video Diffusion Model (VACE 기반)]
    ↓
[Generated Egocentric Video]
```

### 2-4. 성능 향상

E³C는 가장 강력한 전체 성능을 달성하여, 파인튜닝 VACE 기준 모델 대비 충실도(fidelity)와 객체 일관성(object consistency)을 향상시키고, 카메라 이동 오차(camera translation error)를 줄이며, Ego 핸드 지표(ego hand metrics)를 개선합니다.

비-디퓨전 기준 모델인 Splatfacto는 장면이 충분히 재구성된 경우 정확한 시점 렌더링을 제공하지만, 동적 콘텐츠 처리에 어려움을 겪습니다.

### 2-5. 한계점

현재 검색 가능한 정보로부터 식별된 주요 한계는 다음과 같습니다:

1. **SLAM/SfM 의존성**: 3D 장면 재구성(SLAM/SfM 등)에 의존하는 접근 방식은 확장성(scalability)과 일반화 능력(generalization)에 제약이 발생합니다.
2. **동적 콘텐츠 한계**: 장면이 충분히 재구성되지 않은 경우나 동적 콘텐츠가 많은 환경에서 성능이 저하될 수 있습니다.
3. **데이터 분포 편향**: 유사 연구(EgoControl)에서도 지적되듯, 훈련 데이터가 특정 수집 환경에 편향될 경우 일상적인 시점, 의복, 센서 장비 등에 대한 분포 외 일반화(out-of-distribution generalization)가 감소할 수 있습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 현재의 일반화 한계

Nymeria와 같은 데이터셋은 일반 도메인 비디오 생성기에 심각한 분포 이동을 야기하며, Zero-shot 모델이나 일부 3D 인식 기준 모델은 성능이 크게 저하됩니다.

### 3-2. 일반화 향상을 위한 설계적 강점

E³C의 설계는 다음과 같은 구조적 이유로 일반화에 유리합니다:

1. **장면-동작 분리(Disentanglement)**:
   장면의 지속적인 구조와 인간 행동을 분리하는 구조화된 조건을 구축함으로써, 새로운 환경에서도 동작 제어 신호가 독립적으로 작동할 수 있습니다.

2. **외관 기반 3D 메모리**:
   각 3D 포인트에 Video-VAE 특징에서 추출한 외관 디스크립터를 보강함으로써, 단순 기하학적 정보만을 사용하는 것보다 다양한 장면에 적응적으로 대응할 수 있습니다.

3. **Amodal Ego Pose Encoding**:
   착용자 자신의 신체가 시점에서 부분적으로만 관찰되는 경우를 위해 Ego Pose Encoding Token을 도입하여 전신(amodal) 이고센트릭 신체 동작을 포착합니다. 이 설계는 다양한 체형이나 카메라 구성에도 확장될 수 있는 잠재력이 있습니다.

### 3-3. 비교 연구 관점에서의 일반화 분석

| 방법 | 일반화 전략 | 한계 |
|---|---|---|
| **E³C** (2025) | Video-VAE 외관 기반 3D 메모리 + 구조 분리 | SLAM 의존, 동적 장면 약점 |
| **EgoControl** (2024) | 3D 전신 포즈 시퀀스 기반 조건화 | 특정 수집 환경 편향으로 분포 외 일반화 약점 |
| **EgoX** (2024) | 외부 시점 비디오 → 이고센트릭 변환 | 깊이 맵 기반 포인트 클라우드 리프팅, 시간 정렬 필요 |
| **EgoExo-Gen** (2025) | 교차 시점 마스크 예측 | 3D 재구성 또는 정밀 인간 어노테이션 의존 |

---

## 4. 향후 연구에 미치는 영향과 고려사항

### 4-1. 향후 연구에 미치는 영향

1. **체화 AI(Embodied AI) 시뮬레이션**:
   이고센트릭 비디오 생성을 제어 가능하고 물리적으로 정합된 방식으로 수행하는 것은 체화 에이전트가 자신과 타인의 행동이 세계에 어떻게 나타나는지를 추론하는 데 필수적입니다. E³C의 구조는 로봇 시뮬레이션, 자율 에이전트 훈련 데이터 생성에 직접 응용될 수 있습니다.

2. **AR/VR 및 Human-Computer Interaction**:
   E³C는 3D 장면 일관성 유지, Ego 신체 모션 추종, Exo 인간 모션 준수를 모두 달성한 이고센트릭 비디오를 생성합니다. 이 특성은 XR 환경에서의 사실적인 인터랙션 시뮬레이션에 직접 활용될 수 있습니다.

3. **비디오 예측 및 World Model**:
   구조화된 3D 메모리 기반 조건화 방식은 비디오 예측(video prediction) 및 World Model 연구에 새로운 패러다임을 제시합니다.

4. **포즈-비디오 생성 패러다임의 확장**:
   실내 이고센트릭 장면에서의 역동성이 주로 인간에 의해 주도된다는 관찰 하에, Ego-Exo 모션 모두를 동적 3D 스켈레톤 렌더링으로 제어하는 방식은 향후 다중 행위자(multi-agent) 영상 생성 연구로 확장될 수 있습니다.

### 4-2. 앞으로 연구 시 고려해야 할 점

#### (a) SLAM 의존성 탈피
- 3D 장면 재구성에 의존하는 방식은 확장성과 일반화에 한계를 줍니다. 따라서 SLAM 없이도 작동하는 **단안 깊이 추정** 또는 **암묵적 신경 표현(NeRF/Gaussian Splatting)** 기반 메모리 구성 연구가 필요합니다.

#### (b) 동적 객체 처리
- 현재 E³C는 인간이 아닌 동적 객체(움직이는 물체 등)를 포인트 클라우드 정적 메모리와 디퓨전 모델의 암묵적 학습에 의존합니다. 별도의 **동적 객체 추적 모듈** 통합이 후속 연구의 과제입니다.

#### (c) 데이터 다양성 확장
- 훈련 데이터가 특정 수집 환경에 편향되면 분포 외 일반화에 어려움이 생깁니다. 실외, 다양한 문화권, 다양한 센서(일반 스마트폰 카메라 등)를 포함한 대규모 데이터셋 구축이 중요합니다.

#### (d) Ego Pose 추정 정확도
- 착용자의 신체가 부분적으로만 관찰되는 문제를 해결하기 위해 Ego Pose Encoding Token을 도입하였으나, 이를 위한 정확한 전신 포즈 추정 자체가 여전히 어려운 문제입니다. **Amodal 포즈 추정** 모델과의 공동 학습 방향이 고려되어야 합니다.

#### (e) 평가 지표 표준화
- 이고센트릭 비디오 생성의 평가는 일반 비디오 품질 지표(FID, FVD 등) 외에 카메라 이동 오차, Ego 핸드 지표 등 도메인 특화 지표가 필요하며, 카메라 제어 가능 비디오 생성 평가를 위한 표준화된 평균 이동 오차(TransError) 및 회전 오차(RotError) 프로토콜의 공동체 표준 정립이 필요합니다.

---

## 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 핵심 방법 | 주요 특징 |
|---|---|---|---|
| **E³C** (arXiv:2605.26316) | 2025 | Video-VAE 기반 3D 메모리 + Ego-Exo 포즈 제어 | 장면-인간 동작 분리, 포인트 클라우드 외관 특징 |
| **EgoControl** (arXiv:2511.18173) | 2024 | 3D 전신 포즈 기반 비디오 디퓨전 | 전역 카메라 동역학과 관절 신체 움직임 모두 포착하는 포즈 표현 |
| **EgoExo-Gen** (arXiv:2504.11732) | 2025 | 교차 시점 마스크 예측 + 크로스뷰 정렬 | Ego-Exo4D에서 H2O로 재훈련 없이도 손-객체 움직임 모델링 |
| **EgoX** (arXiv:2512.08269) | 2024 | 외부→이고센트릭 시점 변환 + 이중 잠재 인코딩 | 채널 방향 연결(이고센트릭 prior)과 너비 방향 연결(외부 잠재)의 이중 조합으로 기하 충실도 확보 |
| **HMD2** (arXiv:2409.13426) | 2024 | HMD 센서 기반 환경 인식 모션 생성 | 이고센트릭 카메라 이미지 스트림과 SLAM 헤드 궤적, 특징 포인트 클라우드를 이용한 디퓨전 기반 전신 모션 생성 |

---

## 참고 자료 (출처)

1. **[주 논문]** Qiao Gu et al., *"E³C: Video Generation with 3D Environmental Memory and Ego-Exo Human Pose Control"*, arXiv:2605.26316, May 2025. https://arxiv.org/abs/2605.26316
2. **[HTML 전문]** arXiv HTML 버전: https://arxiv.org/html/2605.26316
3. **[비교 논문 1]** Pallotta et al., *"EgoControl: Controllable Egocentric Video Generation via 3D Full-Body Poses"*, arXiv:2511.18173, Nov 2024. https://arxiv.org/pdf/2511.18173
4. **[비교 논문 2]** *"EgoExo-Gen: Ego-centric Video Prediction by Watching Exo-centric Videos"*, arXiv:2504.11732. https://arxiv.org/html/2504.11732
5. **[비교 논문 3]** *"EgoX: Exocentric to Egocentric Video Generation"*, arXiv:2512.08269. https://www.emergentmind.com/papers/2512.08269
6. **[비교 논문 4]** *"HMD2: Environment-aware Motion Generation from Single Egocentric Head-Mounted Device"*, arXiv:2409.13426. https://arxiv.org/html/2409.13426v1
7. **[비교 논문 5]** *"Controllable Egocentric Video Generation via Occlusion-Aware Sparse 3D Hand Joints"*, arXiv:2603.11755. https://arxiv.org/pdf/2603.11755
8. **[관련 연구]** *"Robust Ego-Exo Correspondence with Long-Term Memory"*, arXiv:2510.11417. https://arxiv.org/html/2510.11417
9. **[프로젝트 페이지]** e3c-videogen.github.io
