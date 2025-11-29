# GigaWorld-0: World Models as Data Engine to Empower Embodied AI

### 1. 핵심 주장 및 주요 기여

GigaWorld-0는 세계 모델(World Model)을 Vision-Language-Action (VLA) 학습의 통합 데이터 엔진으로 명시적으로 설계한 프레임워크입니다. 이 논문의 핵심 주장은 실제 로봇 데이터 수집의 높은 비용 문제를 해결하기 위해 **사진 현실적이고 다양하며 제어 가능한 합성 데이터를 대규모로 생성할 수 있는 통합 파이프라인**이 필요하다는 것입니다.[1]

주요 기여는 다음과 같습니다:

**이중 시너지 아키텍처**: GigaWorld-0-Video와 GigaWorld-0-3D의 통합으로 시각적 풍부함과 물리적 타당성을 동시에 확보합니다. 비디오 생성 모듈은 텍스처와 조명의 다양성을 제공하고, 3D 모듈은 기하학적 일관성과 물리 정확성을 보장합니다.[1]

**세분화된 제어 가능성**: 외형(appearance), 카메라 시점(viewpoint), 행동 의미(action semantics)에 대한 독립적인 제어로 대규모 다양한 데이터 생성을 가능하게 합니다.[1]

**스케일 효율적 학습**: FP8 정밀도와 희소 주의(sparse attention), 혼합 전문가(Mixture-of-Experts) 아키텍처를 통해 메모리와 계산량을 대폭 감소시키면서도 고품질 데이터 생성이 가능합니다.[1]

***

### 2. 해결하는 문제 및 제안하는 방법

#### 해결하는 문제

**데이터 수집 병목**: 실제 로봇을 이용한 데이터 수집은 하드웨어 비용, 안전 제약, 노동력 비용으로 인해 확장성이 제한됩니다. 기존 시뮬레이션 데이터는 Sim2Real 간극(reality gap)으로 인해 현실 세계 성능이 떨어집니다.[1]

**일반화 능력 부족**: 제한된 시각(viewpoint), 외형 변화, 대상 객체 배치 다양성으로 인해 VLA 모델의 일반화 성능이 현저히 감소합니다.[1]

**물리적 타당성**: 순수 비디오 생성 모델은 물리 법칙을 보장하지 못하며, 로봇 시뮬레이션에 필요한 기하학적 일관성과 동역학 정보가 부족합니다.[1]

#### 제안하는 방법 - 수학적 공식화

**Flow Matching 기반 비디오 생성**:

$$\frac{dz_t}{dt} = v_\theta(z_t, t, c)$$

여기서 $z_t$는 시간 $t$의 잠재 표현, $c$는 텍스트와 이미지 조건, $v_\theta$는 모델이 학습하는 속도 벡터입니다.[1]

**혼합 전문가(MoE) 구조**:

$$h'_t = u_t + \sum_{i=1}^{N_r} g_{i,t} \text{FFN}_i(u_t)$$

여기서 전문가 선택은 다음과 같이 계산됩니다:

$$g'_{i,t} = \begin{cases} s_{i,t}, & \text{if } s_{i,t} \in \text{Topk}(\{s_{j,t}\}, K_r) \\ 0, & \text{otherwise} \end{cases}$$

$$s_{i,t} = \text{softmax}(u_t^\top e_i)$$

로드 밸런스 손실함수:

$$\mathcal{L}_{\text{Load}} = \alpha \sum_{i=1}^{N_r} f_i P_i$$

$$f_i = \frac{N_r}{K_r T} \sum_{t=1}^{T} \mathbb{1}(s_{i,t} \in \text{Topk}(\{s_{j,t}\}, K_r))$$

$$P_i = \frac{1}{T} \sum_{t=1}^{T} s'_{i,t}, \quad s'_{i,t} = \frac{s_{i,t}}{\sum_{j=1}^{N_r} s_{j,t}}$$

여기서 $\alpha = 0.01$은 밸런스 계수입니다.[1]

**시점 변환(View Transfer)**:

로봇이 월드 좌표계 $\mathcal{W}_A$에서 $\mathcal{W}_B$로 재배치될 때, 절대 엔드이펙터 위치를 보존하면서 새로운 관찰을 합성합니다:

$$T^{\text{ee} \to \mathcal{W}}_t = T^{\text{base} \to \mathcal{W}_A} \cdot T^{\text{ee} \to \text{base}}_t = T^{\text{base} \to \mathcal{W}_B} \cdot K_t$$

변환된 관찰에서의 엔드이펙터 포즈:

$$K_t = (T^{\text{base} \to \mathcal{W}_B})^{-1} \cdot T^{\text{base} \to \mathcal{W}_A} \cdot T^{\text{ee} \to \text{base}}_t$$

이를 통해 기하학적 일관성을 유지하면서 시점 변환을 수행합니다.[1]

**역 동역학 모델(Inverse Dynamics Model)**:

생성된 비디오 시퀀스 $V = \{v_1, v_2, \ldots, v_T\}$에서 로봇 행동을 추론합니다:

$$\theta_{1:T} = f_{\text{IDM}}(V)$$

여기서 $\theta_t = [\theta^{(1)}_t, \theta^{(2)}_t, \ldots, \theta^{(D)}_t]^\top \in \mathbb{R}^D$는 $D$개 관절의 회전각 궤적입니다. 마스킹된 학습을 통해 로봇 팔 영역만 처리하여 배경 잡음의 영향을 감소시킵니다.[1]

***

### 3. 모델 구조

#### GigaWorld-0-Video 구조

**기초 모델 - GigaWorld-0-Video-Dreamer**:

3D-VAE를 이용한 압축(공간-시간 압축 비율 4:8:8)으로 16채널 잠재 표현을 생성합니다. DiT(Diffusion Transformer) 백본에 MoE 아키텍처 ($N_r = 4$ 라우팅 전문가, $K_r = 2$ 활성화 전문가)를 적용하여 이미지-텍스트-비디오 생성을 수행합니다. 희소 주의(NATTEN)와 3D 회전 위치 임베딩(3D-RoPE)을 활용합니다.[1]

**적응형 제어 분지**:

세 가지 포스트트레이닝 모델은 공통된 제어 메커니즘을 사용합니다. 제어 신호(깊이맵, 표면 법선 등)를 3D-VAE로 인코딩하고, 채널 방향 연결 후 채널 압축 MLP를 거쳐 변환 블록에 입력됩니다. ControlNet과 달리 MoE 계층 복제로 인한 매개변수 증가를 피합니다.[1]

**멀티뷰 생성**:

다중 뷰 이미지를 파노라마 형식으로 연결하여 단일 입력으로 처리. 미세 조정 후 확산 모델의 문맥 학습 능력으로 뷰 간 일관성을 달성합니다.[1]

#### GigaWorld-0-3D 구조

**전경 생성(GigaWorld-0-3D-FG)**:

Trellis 기반 이미지-3D 변환에 품질 관리 게이트를 적용합니다. Aesthetic Checker로 텍스처 풍부도 평가, ImageSegChecker(GPT-4o)로 분할 신뢰도 확인, MeshGeoChecker로 기하학적 완성도 검증합니다. URDF 포맷으로 내보내집니다.[1]

**배경 재구성(GigaWorld-0-3D-BG)**:

3DGRUT를 이용한 초기 희소 뷰 재구성 후, 뷰 복원 모델로 중간 뷰를 생성하여 아티팩트를 완화합니다. 이후 밀집 3DGS 재구성으로 고충실도 표현을 달성하고, Poisson 표면 재구성으로 메시 변환합니다.[1]

**물리 모델링(GigaWorld-0-3D-Phys)**:

**미분 가능 물리**를 통한 시스템 식별:
1. 실제 궤적과 랜덤 물리 매개변수를 시뮬레이터에 입력하여 시뮬레이션 롤아웃 생성
2. 대리 모델 $\mathcal{M}_{f,p,d}$를 학습하여 시뮬레이터 동역학 근사 (MSE 최소화)
3. 대리 모델 고정 후 그래디언트 강하로 물리 매개변수 $(f^\*, p^\*, d^*)$ 최적화

로봇 팔의 경우 마찰(friction), 강성(stiffness), 감쇠(damping)를 식별하고, 조작 객체는 Qwen3-VL 기반 멀티모달 전문가로 질량, 마찰계수, 크기 등을 추정합니다.[1]

**동작 생성(GigaWorld-0-3D-Act)**:

단순 시나리오에서는 MimicGen으로 기본 시연을 확장하고, 복잡한 시나리오에서는 강화 학습(RLPD)으로 정책을 부트스트랩하여 다양한 조작 궤적을 생성합니다.[1]

***

### 4. 성능 향상

#### 벤치마크 성능

**PBench Robot Set 평가**: GigaWorld-0-Video-Dreamer는 2B 활성 매개변수로 82.07의 전체 점수를 달성하여, 14B 매개변수의 Cosmos-Predict2를 포함한 최신 모델들을 능가했습니다. 특히 대상 일관성(object consistency) 88.2점과 배경 일관성(background consistency) 66.8점에서 우수한 성능을 보였습니다.[2]

**DreamGen Bench 평가**: GR1 로봇 데이터셋에서 미세 조정 후, GigaWorld-0-Video-Dreamer는 GR1-Env, GR1-Object, GR1-Behavior 모두에서 2B 규모의 Cosmos-Predict2.5를 능가하는 지시 따르기 충실도(instruction-following fidelity)를 보여주었습니다.[2]

#### 계산 효율성

논문의 Table 2에 따르면, FSDP-2 프레임워크에서:
- **FP8 정밀도**: 메모리 사용량 25-30% 감소, 학습 시간 15% 단축
- **희소 주의 + FP8**: 메모리 73,131 MB, 시간당 33.38초 달성
- **MoE 활성화 시**: 활성화 체크포인트 필수, 메모리 73,997 MB 유지

단계 증류(step distillation)와 FP8 추론으로 기준 확산 모델 대비 **50배 이상의 속도 향상**을 달성합니다.[1]

***

### 5. 모델의 일반화 성능 향상 가능성

GigaWorld-0는 세 가지 핵심 방향에서 일반화 성능을 개선합니다:[1]

#### 외형 일반화(Appearance Generalization)

**기존 문제**: 제한된 색상, 재질, 조명 조건으로 수집된 실제 데이터는 다양한 환경에서 성능 저하
  
**GigaWorld-0-Video-AppearanceTransfer의 해결책**:
- 텍스트 프롬프트를 통해 실제 또는 시뮬레이션 영상의 텍스처, 재질, 조명을 독립적으로 편집
- VideoDepthAnything과 LOTUS로 추출한 깊이맵과 법선맵 활용으로 기하학적 구조 보존
- 시뮬레이션-현실 간 외형 간극(sim2real appearance gap) 축소

**효과**: VLA 모델이 다양한 시각적 조건에서 학습되므로, 배포 환경의 조명이나 색상 변화에 더욱 강건합니다.[1]

#### 시점 일반화(Viewpoint Generalization)

**기존 문제**: 단일 시각에서 수집된 데이터로 학습한 모델은 다른 카메라 위치에서 성능 저하[3]

**GigaWorld-0-Video-ViewTransfer의 해결책**:
- 이중 재투영 전략(double-reprojection strategy)으로 자기 감독 학습 쌍 구성
- 배경: MoGe 깊이 추정 → 목표 뷰로 워프 → 원본 뷰로 재투영
- 로봇 팔: 물리 시뮬레이터에서 변환된 동작 궤적 렌더링
- 임의의 카메라 시점에서 기하학적으로 일관성 있는 로봇 행동 생성

**효과**: 단일 실제 영상에서 다중 시점 데이터 생성으로 효과적 데이터셋 확대, 다양한 카메라 위치에서의 견고성 향상[1]

#### 교차 화신 일반화(Cross-embodiment Generalization)

**기존 문제**: 인간 시연 데이터와 로봇 동작 간 외형 간극으로 인해 직접 활용 어려움

**GigaWorld-0-Video-MimicTransfer의 해결책**:
- 첫 인칭 인간 시연 영상에서 인간 손 마스킹
- 인간 손의 주석된 엔드이펙터 포즈에서 역기구학 계산
- 로봇 팔 포즈를 물리 시뮬레이터에서 렌더링
- 배경과 로봇 팔 모션 조건을 통합하여 로봇 조작 영상 합성

**효과**: 저비용의 인간 시연 데이터를 로봇 학습에 직접 활용 가능, 데이터 수집 효율성 극대화[1]

#### 멀티뷰 일관성을 통한 공간 추론 강화

**효과** (Figure 7): 이동 로봇 팔 기반 IDM이 멀티뷰 입력에서 훨씬 정확한 그리퍼 상태 추정을 수행. 공간 추론이 필요한 downstream 작업에서 다중 관점 데이터의 이점을 활용합니다.[1]

***

### 6. 모델의 한계

#### 논문에서 명시된 한계

**생성 영상의 품질 관제**: 생성 과정에서 발생할 수 있는 환각(hallucination)이나 아티팩트가 정책 학습을 방해할 수 있으며, 이를 위해 다차원적 품질 평가 파이프라인(기하학적 일관성, 다중뷰 코히어런스, 텍스트-비디오 정렬, 물리 타당성)을 도입했습니다.[1]

**동적 객체 모델링**: 변형 가능한 객체(deformable objects)의 물리 시뮬레이션이 아직 진행 중(PhysTwin 영감의 스프링-질량계 방식이 피드포워드 접근으로 개발 중).[1]

**데이터 분포의 편향**: 학습 데이터가 GigaAI의 특정 로봇 플랫폼(Agilex Cobot Magic, AgiBot G1)과 수집 환경(산업, 상업, 사무실, 주거, 실험실 설정)에 편향될 수 있습니다.[1]

#### 최신 연구에서 식별된 일반적 한계

**물리 일관성 vs 픽셀 충실도의 트레이드오프**: 세계 모델의 평가에서 픽셀 수준의 충실도가 물리 법칙 준수성보다 우선되는 경향이 있으나, 실제 로봇 학습에는 물리 타당성이 더 중요합니다.[4]

**장기 시간 일관성 부족**: 시퀀셜 시뮬레이션에서 에러가 누적되어 장기(long-horizon) 작업 예측이 부정확해질 수 있습니다.[4]

**안전성 우려**: 예측된 세계 모델이 위험한 행동(collision, 비물리적 상태)을 생성할 가능성이 있으며, 이는 로봇의 안전-중심 제어와 충돌할 수 있습니다.[5]

**계산-성능 트레이드오프**: 실시간 제어가 필요한 로봇 응용에서 모델 크기와 추론 속도 간 균형이 중요한데, GigaTrain의 최적화에도 불구하고 엣지 디바이스 배포 시 도전이 남아있습니다.[6]

***

### 7. 논문이 앞으로의 연구에 미치는 영향

#### 즉각적 영향

**데이터 엔진 패러다임의 확산**: GigaWorld-0의 성공은 세계 모델을 단순 예측 도구에서 **확장 가능한 데이터 생성 플랫폼**으로 재정의합니다. 이는 자율주행과 로보틱스 커뮤니티에서 합성 데이터 생성의 중요성을 강조합니다.[7][2][4]

**다중 모달 기초 모델의 통합**: LLM(고수준 추론)과 World Model(저수준 동역학 예측)의 협력 아키텍처 필요성을 입증합니다. 향후 연구는 MLLM-WM 통합 프레임워크 개발에 집중할 것으로 예상됩니다.[8][7]

**VLA 모델의 확장성 향상**: 합성 데이터로 학습된 VLA가 실제 로봇에서 강한 성능을 보임으로써, 실제 데이터 수집의 필요성을 대폭 감소시킬 수 있음을 입증합니다.[8][1]

#### 기술적 파급 효과

**도메인 랜덤화의 고도화**: GigaWorld-0의 외형/시점 일반화 전략은 **합리적이고 제어 가능한 도메인 랜덤화**의 새로운 기준을 제시합니다. 기존의 무작위 랜덤화에서 벗어나 의미 있는 변화 공간을 탐색하는 방향으로 발전합니다.[9][10]

**미분 가능 물리의 실용화**: 시스템 식별에 미분 가능 물리를 적용하는 GigaWorld-0의 접근은 로봇팔의 정확한 시뮬레이션을 가능하게 하며, 향후 연구에서 더 복잡한 동역학(변형체 상호작용, 접촉 역학) 식별로 확대될 것입니다.[11][12][13]

**3D Gaussian Splatting의 로봇 응용 가속화**: 배경 재구성에 3DGS를 활용함으로써, 향후 **SLAM, 경로 계획, 조작 제어**에서 3DGS 기반 표현의 채택이 급증할 것으로 예상됩니다.[14][12]

#### 학술 커뮤니티에 미치는 영향

**Open-sourcing의 기대 효과**: 논문에서 명시한 모델, 데이터 생성 파이프라인, GigaTrain 프레임워크의 공개는 세계 모델 연구의 진입 장벽을 낮춥니다.[1]

**새로운 벤치마크와 평가 기준**: PBench와 DreamGen Bench의 성공은 향후 세계 모델 평가에서 **물리 타당성, 기하학적 일관성, 장기 안정성**을 중시하는 지표 개발을 가속화합니다.[4]

***

### 8. 향후 연구 시 고려할 점

#### 기술적 고려사항

**도메인 랜덤화의 최적화**: GigaWorld-0의 제어 가능한 다양성은 좋지만, "어느 정도의 다양성이 최적인가"에 대한 정량적 기준이 필요합니다. Entropy Maximization 기반 방법(DORAEMON)이나 Offline Domain Randomization(DROPO) 같은 적응형 랜덤화 기법과의 결합을 고려해야 합니다.[10][9]

**물리-기반 재구성의 정확도**: 미분 가능 물리 식별에서 수렴 보장과 매개변수 식별 정확도를 향상시키기 위해, **제약 조건이 있는 최적화**(예: 물리적 타당성 범위 내)와 **멀티태스크 학습**(여러 시나리오 동시 식별)의 통합을 검토해야 합니다.[12][11]

**장기 시간 일관성의 해결**: 시퀀셜 생성에서 에러 누적을 완화하기 위해, 자기 정정 메커니즘(self-correcting mechanism)이나 오류 피드백 루프(GigaWorld-0-3D의 "물리 정확성 검증"을 생성 단계로 역전)의 통합을 고려합니다.[15][16][4]

**안전성 검증 강화**: 생성된 데이터가 물리적으로 불가능한 상태(collision, 비인과성)를 포함하지 않도록, **안전성 제약이 있는 생성**(safety-constrained generation)을 구현합니다. 예를 들어, 접촉 감지 후 강제 중단 같은 보안 장치를 추가합니다.[5]

#### 실전 배포 고려사항

**Sim2Real 전이의 견고성**: GigaWorld-0는 높은 충실도 데이터를 생성하지만, 실제 로봇에서의 변동성(센서 노이즈, 동작 편차)에 대한 건강성을 검증해야 합니다. 실제 데이터의 작은 배치로 fine-tuning하는 **이중 강화(two-stage fine-tuning)** 전략이 효과적일 수 있습니다.[17][18][19]

**엣지 배포 최적화**: GigaTrain의 FP8 최적화는 유망하지만, 모바일 로봇이나 임베디드 엣지 디바이스에서의 추론 속도와 메모리 제약을 감안한 **더욱 가벼운 모델 변형**(예: 1-2B 매개변수 완전 버전)과 **적응형 계산(adaptive computation)** 메커니즘이 필요합니다.[20][6]

**연합 학습과의 통합**: 개인화되고 프라이버시 보호 환경에서 GigaWorld-0 같은 기초 모델을 지속적으로 개선하기 위해, 멀티모달 멀티태스크 연합 기초 모델(M3T-FFM) 패러다임과의 결합을 탐색합니다.[20]

#### 차세대 연구 방향 (최신 연구 기반)

**World Model을 정책 환경으로의 진화**: 현재 GigaWorld-0는 데이터 엔진이지만, 향후에는 이를 **모델 기반 강화 학습의 정책 환경**으로 활용하여, 로봇이 안전하게 시뮬레이션에서 "상상"하며 학습할 수 있게 합니다. 이를 "World Models as Policy Environments"라 부릅니다.[21][15][1]

**인간-로봇 협력의 향상**: 인간의 정신 세계 모델(mental world model)을 통합하여, 로봇이 인간의 의도를 더 정확히 이해하고 협력하도록 합니다.[21][15]

**자가 개선 루프의 구축**: 실제 로봇이 수행한 경험을 지속적으로 GigaWorld-0에 피드백하여, 모델이 자동으로 개선되는 **자기 개선 체계(self-improving system)**를 구축합니다. 이는 "로봇 경험 → 세계 모델 개선 → 더 나은 정책 → 더 좋은 경험" 사이클을 형성합니다.[16][22][1]

**멀티모달 기초 모델의 통합 아키텍처**: LLM의 고수준 추론, Vision-Language Model의 지각, World Model의 역학 예측, VLA의 행동 생성을 **단일 통합 프레임워크**로 통합하여, 진정한 의미의 구체화된 AGI(Embodied AGI) 추구합니다.[23][7]

**3D 기하학 인식 제어의 확대**: 3DGS 기반 재구성을 경로 계획, SLAM, 비주얼 서보 제어로 확장하여, 순수 픽셀 기반 제어에서 **3D 기하학 인식 제어**로의 패러다임 전환을 가속화합니다.[24][25][14]

***

## 결론

GigaWorld-0는 세계 모델을 **데이터 엔진**으로 재구성함으로써 구체화된 AI(Embodied AI) 학습의 확장성 문제를 혁신적으로 해결합니다. 이중 시너지 아키텍처(비디오 생성 + 3D 물리)와 세분화된 제어 가능성은 외형, 시점, 교차 화신 일반화를 동시에 달성합니다. 특히 FP8 정밀도와 희소 주의를 통한 계산 효율성 개선은 대규모 확장을 현실화합니다.

이 논문이 향후 연구에 미치는 영향은 **데이터 엔진 패러다임의 확산**, **LLM-World Model 통합**, **미분 가능 물리의 실용화**, **3DGS 로봇 응용의 가속화**로 요약됩니다. 동시에 물리 일관성, 장기 안정성, 안전성 검증, Sim2Real 견고성 같은 해결해야 할 과제들이 분명히 존재합니다.

향후 연구는 **자가 개선 루프를 통한 지속적 개선**, **멀티모달 기초 모델 통합**, **안전성 제약이 있는 생성**, **엣지 배포 최적화**에 집중할 것으로 예상되며, 궁극적으로 GigaWorld-0 같은 세계 모델이 로봇의 자율 학습과 진정한 구체화된 지능(Embodied Intelligence)의 실현을 가능하게 할 것입니다.

***

### 참고 자료

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/970a169b-2718-49a6-bf17-9bc6d968388a/2511.19861v1.pdf)
[2](https://ieeexplore.ieee.org/document/10972630/)
[3](https://arxiv.org/abs/2510.07067)
[4](https://arxiv.org/abs/2510.16732)
[5](https://arxiv.org/abs/2510.05865)
[6](https://www.ijcai.org/proceedings/2025/766)
[7](https://arxiv.org/abs/2509.20021)
[8](https://arxiv.org/abs/2505.20503)
[9](https://arxiv.org/abs/2311.01885)
[10](https://arxiv.org/abs/2201.08434)
[11](https://arxiv.org/abs/2511.06846)
[12](https://journals.sagepub.com/doi/abs/10.1177/02783649251334661)
[13](https://arxiv.org/abs/2411.00554)
[14](https://github.com/zstsandy/Awesome-3D-Gaussian-Splatting-in-Robotics)
[15](http://arxiv.org/pdf/2503.00727.pdf)
[16](https://arxiv.org/html/2501.01895v2)
[17](https://arxiv.org/abs/2406.15149)
[18](https://ieeexplore.ieee.org/document/8981654/)
[19](https://www.cambridge.org/core/product/identifier/S0263574722001230/type/journal_article)
[20](https://arxiv.org/abs/2505.11191)
[21](https://arxiv.org/abs/2506.22355)
[22](https://arxiv.org/html/2410.15461v1)
[23](http://arxiv.org/pdf/2502.15336.pdf)
[24](http://mcsl.skku.edu/MCSL/wp-content/uploads/2025/03/isyang_IPIU2025_3D-Gaussian-Splatting-based-Static-Scene-Volumetric-Video-Capturing-System-Using-Remote-Controlled-Movable-Robots_r1.pdf)
[25](https://www.themoonlight.io/ko/review/3d-gaussian-splatting-in-robotics-a-survey)
[26](https://www.semanticscholar.org/paper/078b530b3dd7b3dcd50b8764087453280fd7a2a6)
[27](http://arxiv.org/pdf/2406.18043.pdf)
[28](https://arxiv.org/pdf/2403.09631.pdf)
[29](https://arxiv.org/html/2409.16019v1)
[30](https://arxiv.org/pdf/2402.06665.pdf)
[31](https://neurips.cc/virtual/2025/workshop/109532)
[32](https://en.wikipedia.org/wiki/Vision-language-action_model)
[33](https://www.episodeyang.com/research_statement_geyang.pdf)
[34](https://learnopencv.com/vision-language-action-models-lerobot-policy/)
[35](https://pmc.ncbi.nlm.nih.gov/articles/PMC10959504/)
[36](https://github.com/Li-Zn-H/AwesomeWorldModels)
[37](https://arxiv.org/abs/2510.07077)
[38](https://www.softserveinc.com/en-us/blog/ai-powered-synthetic-data-for-robotics)
[39](https://mn.cs.tsinghua.edu.cn/xinwang/PDF/papers/2025_Embodied%20AI%20from%20LLMs%20to%20World%20Models.pdf)
[40](https://ieeexplore.ieee.org/document/10226252/)
[41](https://www.semanticscholar.org/paper/e1f6d4f020721052e0980c0f265f1415d1aec562)
[42](https://link.springer.com/10.1007/978-3-030-86486-6_16)
[43](https://www.semanticscholar.org/paper/d7bcfc6543eefc67083ef788fd007980e3302a12)
[44](https://ieeexplore.ieee.org/document/9093563/)
[45](https://ieeexplore.ieee.org/document/9263751/)
[46](https://arxiv.org/pdf/2110.03239.pdf)
[47](https://arxiv.org/pdf/1909.00889.pdf)
[48](http://arxiv.org/pdf/2106.03632.pdf)
[49](http://arxiv.org/pdf/2011.01891.pdf)
[50](https://arxiv.org/pdf/2311.08503.pdf)
[51](https://arxiv.org/pdf/2211.04393.pdf)
[52](https://arxiv.org/pdf/2012.02055.pdf)
[53](https://www.emergentmind.com/topics/domain-randomization-dr)
[54](https://arxiv.org/abs/2110.03239)
[55](https://openreview.net/pdf?id=T8vZHIRTrY)
[56](https://talkingaboutme.tistory.com/entry/RL-Domain-Randomization-for-Sim2Real-Transfer)
[57](https://arxiv.org/html/2510.23988v1)
[58](https://lilianweng.github.io/posts/2019-05-05-domain-randomization/)
