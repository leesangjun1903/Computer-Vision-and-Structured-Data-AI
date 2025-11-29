# SteadyDancer: Harmonized and Coherent Human Image Animation with First-Frame Preservation

### 1. 핵심 주장과 주요 기여도 요약[1]

**SteadyDancer**는 **Image-to-Video (I2V) 패러다임**을 기반으로 한 혁신적인 인간 이미지 애니메이션 프레임워크로, 기존의 Reference-to-Video (R2V) 패러다임이 간과한 **첫 프레임 보존(first-frame preservation)**을 처음으로 견고하게 달성한다.[1]

**핵심 주장:**
- R2V 패러다임은 참조 이미지를 주어진 포즈에 단순히 바인딩하는 것에 집중하며, 실제 애플리케이션에서 나타나는 **공간-시간적 정렬 오류(spatio-temporal misalignment)**를 무시한다.[1]
- I2V 패러다임은 첫 번째 프레임에서 시작하는 일관되고 조화로운 비디오를 생성함으로써 시각적 충실도를 극대화한다.[1]
- 정확한 모션 제어와 첫 프레임 충실도 사이의 근본적인 트레이드오프를 해결할 수 있다.[1]

**주요 기여:**
1. **첫 프레임 보존을 달성하는 첫 번째 I2V 기반 애니메이션 프레임워크**[1]
2. **Condition-Reconciliation Mechanism**: 충돌하는 두 조건(참조 이미지 및 포즈)을 조화시켜 충실도를 희생하지 않으면서 정확한 제어 달성[1]
3. **Synergistic Pose Modulation Modules**: 참조 이미지와 호환성 높은 적응형 포즈 표현 생성[1]
4. **Staged Decoupled-Objective Training Pipeline**: 계층적으로 모션 충실도, 시각 품질, 시간적 일관성을 최적화[1]
5. **X-Dance 벤치마크**: 실제 공간-시간적 정렬 오류를 평가하는 새로운 평가 데이터셋 제안[1]

***

### 2. 문제 정의, 제안 방법, 모델 구조, 성능 및 한계

#### 2.1 해결하려는 문제[1]

**주요 문제들:**

**공간적 정렬 오류 (Spatial Misalignment):**
- 원본 이미지와 구동 포즈 간의 신체 골격 비율, 팔다리 구조 등 정적 속성의 근본적인 차이[1]
- 이로 인해 구조적 변형, 신원 드리프트(identity drift), 세부 사항 저하 발생[1]

**시간적 정렬 오류 (Temporal Misalignment):**
1. **모션 지터(jitter)**: 복잡하거나 노이즈가 많은 포즈 시퀀스로 인한 시간적 불안정성[1]
2. **모션 불연속성(motion discontinuity)**: 원본 이미지와 초기 포즈 프레임 사이의 급격한 전환 (start-gap misalignment)[1]

**R2V 패러다임의 한계:**
- R2V는 이 정렬 제약을 무시하고 이미지를 포즈에 바인딩만 하므로, 정렬 오류 상황에서 심각한 아티팩트, 외형 왜곡, 시간적 불일치 발생[1]

#### 2.2 제안하는 방법[1]

##### **핵심 기술 1: Condition-Reconciliation Mechanism**

**문제:** 순진한 I2V 베이스라인의 단순 원소별 덧셈(element-wise addition)은 정적 외형 정보와 동적 포즈 신호를 혼동하여 충돌 야기[1]

**해결책:**

**1) Condition Fusion (채널별 연결):**

$$z_t = \text{ChannelConcat}(\hat{z}_t, m, z_c, z_p)$$

채널별 연결(channel-wise concatenation)을 통해 각 조건의 명확성 유지[1]

이전의 원소별 덧셈:

$$z_t = \text{ChannelConcat}(\hat{z}_t, m, z_c + z_p)$$

**2) Condition Injection (매개변수 효율적 전략):**
- LoRA 기반 미세 조정으로 매개변수 오버헤드 최소화[1]
- 추가 모듈 없이 포즈 조건 주입[1]

**3) Condition Augmentation (첫 프레임 강화):**

$$z_{\text{cond}} = \text{ChannelConcat}(\hat{z}_t, m, z_c, z_p)$$

$$z_t = \text{TemporalConcat}(z_{\text{fused}}, z_{c_0}, z_{p_0})$$

- 시간적 수준에서 첫 프레임 이미지 잠재(z_c0)와 첫 프레임 포즈 잠재(z_p0)를 연결[1]
- 글로벌 컨텍스트 augmentation: CLIP 이미지 특성을 첫 포즈 프레임의 CLIP 특성과 연결[1]

***

##### **핵심 기술 2: Synergistic Pose Modulation Modules**[1]

**공간적 정렬 해결:**

**Spatial Structure Adaptive Refiner (P_SSAR):**
- 동적 합성곱(dynamic convolution)으로 포즈 특성으로부터 커널을 적응적으로 생성[1]
- 입력 포즈 특성에 맞춰진 더 표현력 있는 표현 생성[1]

**시간적 정렬 해결:**

**Temporal Motion Coherence Module (P_TMCM):**
- 스택된 인수분해 합성곱 블록(factorized convolutional blocks) 활용[1]
  - 깊이별 공간 합성곱: 프레임 내 구조 포착
  - 지점별 시간 합성곱: 프레임 간 동역학 모델링[1]
- 불안정한 포즈로부터의 아티팩트 억제[1]

**Frame-wise Attention Alignment Unit (P_FAAU):**
- 크로스-어텐션 메커니즘으로 디노이징 잠재(ˆz_t)가 포즈 잠재에 어텐션[1]
- 외형-보정 포즈 표현 생성[1]

**계층적 집계:**

$$z_p^* = z_p + \Phi_{\text{SPAE}}(z_p) + \Phi_{\text{TMSM}}(z_p)$$

$$z_p^† = \Phi_{\text{FAAU}}(q = \hat{z}_t, kv = z_p^*)$$

$$z_{\text{cond}} = \text{ChannelConcat}(\hat{z}_t, m, z_c, z_p^*, z_p^†)$$

***

##### **핵심 기술 3: Staged Decoupled-Objective Training Pipeline**[1]

**Stage 1: Action Supervision (행동 감시)**
- 목표: 모션 제어 능력 신속하게 습득[1]
- 첫 프레임을 참조로 고정, 전체 비디오를 모션 조건 및 감시 대상으로 사용[1]
- LoRA 기반 학습으로 사전학습 모델의 생성 사전(generative priors) 보호[1]
- 12,000 스텝[1]

**Stage 2: Condition-Decoupled Distillation (조건 분리 증류)**

목표: 모션 제어 학습 중 생성 품질 손실 보상[1]

$$v_\phi(x_t, t, c) = \underbrace{v_\phi^u(x_t, t)}_{\text{unconditional component}} + \underbrace{v_\phi^c(x_t, t, c)}_{\text{conditional component}}$$

감시 목표:
- 무조건 성분: $L_{\text{distill}} = \|v_\phi^u(x_t, t) - v_\theta^u(x_t, t)\|_2$ (교사 모델과 정렬)
- 조건 성분: $L_{\text{fidelity}} = \|v_\phi(x_t, t, c) - v^*\|_2$ (진정한 속도장 회귀)[1]

이러한 분리를 통해 조건부 네트워크가 무조건 대상을 모방하도록 강제될 때 발생하는 분포 시프트 제거[1]
- 2,000 스텝[1]

**Stage 3: Motion Discontinuity Mitigation (모션 불연속성 완화)**

**Pose Simulation (포즈 시뮬레이션):**
부드러운 시퀀스 $\{p_0, p_1, \ldots, p_T\}$ 에 대해:
- 합성 쌍 $(p_0, p_{T^\*})$ 구성, $T^*$는 $\{2, 3, 4\}$에서 샘플링[1]
- 해당 쌍 사이의 포즈 보간으로 중간 시퀀스 $\{\tilde{p}\_1, \ldots, \tilde{p}_{T^*-1}\}$ 생성[1]
- 원본 $p_1$을 합성 $\tilde{p}_1$로 대체하여 의사 학습 샘플 $\{p_0, \tilde{p}_1, \ldots, p_T\}$ 생성[1]
- 확률 0.5로 원본 포즈와 합성 포즈 중 선택[1]

효과: 수백 스텝의 미세 조정으로 80% 이상의 극단적 점프 시나리오 해결[1]
- 500 스텝[1]

**추가 기술: Decoupled-Condition Classifier-Free Guidance (DC-CFG)**[1]

표준 CFG를 포즈 신호에 특화시킴:

$$\varepsilon_\theta(x_t, t, y, w_{\text{pose}}, w_{\text{txt}}) = \varepsilon_0(x_t, t, y_{\text{net}})$$

$$+ w_{\text{pose}} \cdot \Delta\varepsilon_\theta(x_t, t, y, y_{\text{neg pose}}) + w_{\text{txt}} \cdot \Delta\varepsilon_\theta(x_t, t, y, y_{\text{neg txt}})$$

시간적 스케줄링: 조기 디노이징 단계에서 강한 포즈 가이던스, 후기 단계에서 감소 (시간 간격 [0.1, 0.4])[1]

***

#### 2.3 모델 구조[1]

**기본 구조**: Wan-2.1 I2V 14B 사전학습 모델 기반[1]

**주요 구성 요소:**
1. **VAE 인코더 (E)**: 이미지와 포즈 시퀀스를 잠재 공간으로 인코딩[1]
2. **Diffusion Transformer (DiT)**: 메인 생성 모듈[1]
3. **Pose Modulation Module 집합**: 포즈 특성 정제[1]
4. **교사-학생 아키텍처** (Stage 2): 증류를 위한 고정 교사 모델 사용[1]

**입력 처리:**

$$z_t = \text{ChannelConcat}(\hat{z}_t, m, z_c, z_p)$$

여기서:
- $\hat{z}_t$: 현재 노이즈 잠재
- $m$: 이진 마스크
- $z_c$: 참조 이미지 잠재
- $z_p$: 포즈 시퀀스 잠재[1]

***

#### 2.4 성능 향상[1]

**정량적 결과:**

| 데이터셋 | 메트릭 | 성능 | 비고 |
|---------|--------|------|------|
| TikTok | FVD↓ | 451.3 | SOTA (RealisdancE-DiT: 458.8) |
| RealisDance-Val | FVD↓ | 326.49 | SOTA (Wan-Animate: 386.87) |
| RealisDance-Val | Motion Smoothness↑ | 99.02 | 상위 성능 |
| RealisDance-Val | Aesthetic Quality↑ | 56.80 | 상위 성능 |

**훈련 효율성:**
- **총 훈련 스텝**: 14,500 스텝 (12k + 2k + 0.5k)[1]
- **훈련 데이터**: 7,338개 비디오 클립 (10.2시간) - 비교 방법 대비 **훨씬 작은 데이터**[1]
- **계산 리소스**: 8개 NVIDIA H800 GPU[1]
- **비교**: 기존 DiT 기반 방법들은 20k-200k 스텝, 대규모 데이터셋 필요[1]

**정성적 개선:**

1. **첫 프레임 보존**: 기존 R2V 방법이 신원 드리프트를 보이는 동안 SteadyDancer는 완벽한 신원 보존[1]

2. **모션 정확성**: X-Dance 벤치마크에서 복잡한 모션, 블러, 폐색이 있는 환경에서 다른 방법들은 "재앙적 이중 실패"(catastrophic dual failure)를 보이나, SteadyDancer는 조화로운 결과 생성[1]

3. **Human-Object Interaction (HOI)**: 인간-물체 상호작용 장면에서:
   - 기존 모델: 정적 아티팩트 또는 심각한 형태 붕괴
   - SteadyDancer: 물체의 물리적으로 그럴듯한 모션과 변형 합성[1]

4. **모션 불연속성 완화**: Pose Simulation을 통해 원본 이미지와 초기 포즈 프레임 간의 극단적 차이에도 부드러운 과도 효과 생성[1]

***

#### 2.5 모델의 한계[1]

논문에서 명시된 한계:

1. **스타일화 이미지의 도메인 갭**
   - 애니메이션 참조 프레임에 대한 성능이 실제 이미지보다 약간 낮음[1]
   - 원인: 훈련 데이터에서 애니메이션 샘플 부족[1]
   - 해결 방향: 애니메이션 데이터셋으로 확충 필요[1]

2. **극단적 모션 불연속성**
   - 참조 프레임과 초기 포즈 프레임 간 극단적인 포즈 차이 상황에서 약간 부자연스러운 가속화된 전환 가능[1]
   - 해결 방향: 더 정교한 시간적 모델링 아키텍처, 훈련 데이터 확장[1]

3. **포즈 추정 오류의 누적**
   - 연속적인 포즈 추정 오류가 생성된 비디오에 되돌릴 수 없는 아티팩트 야기[1]
   - 현재: 정확한 제어성과 높은 오류 허용성 간의 트레이드오프 존재[1]

4. **고급 모션 표현 필요성**
   - 포즈 기반 표현만으로는 제한적[1]

***

### 3. 모델의 일반화 성능 향상 가능성[1]

#### 3.1 현재 일반화 성능 평가[1]

**X-Dance 벤치마크의 도입 배경:**
기존 TikTok, RealisDance 같은 같은-출처 벤치마크(same-source benchmark)는 참조 이미지와 포즈 시퀀스가 동일 비디오에서 추출되어 현실 세계의 공간-시간적 정렬 오류를 반영하지 못함[1]

**X-Dance 벤치마크 구성:**[1]
- 12개의 서로 다른 구동 비디오 (8개 복잡 고속 댄스 + 4개 저진폭 일상 활동)
- 비이상적 실제 요인: 모션 블러, 심각한 폐색, 급격한 포즈 변화
- 다양한 참조 이미지: 애니메이션 캐릭터, 반신 촬영, 성별/체형 차이, 구별되는 자세

**평가 결과:**[1]
- 경쟁 방법: 신원 보존 실패 + 모션 제어 실패 = 재앙적 이중 실패
- SteadyDancer: 첫 프레임 신원 근완벽 유지 + 정확한 모션 제어

#### 3.2 일반화 성능 향상 메커니즘[1]

**1) 조건 분리를 통한 일반화:**
- Condition-Reconciliation 메커니즘이 각 조건의 명확성 유지[1]
- 과도한 매개변수 주입 방지로 사전학습 모델의 일반화 능력 보존[1]

**2) 포즈 적응형 표현:**
- Spatial Structure Adaptive Refiner가 다양한 신체 구조에 대응[1]
- 동적 합성곱으로 다양한 포즈 형태에 동적으로 적응[1]

**3) 시간적 일관성 강화:**
- Temporal Motion Coherence Module의 스택된 인수분해 합성곱은 다양한 모션 패턴에 일관된 응답[1]

**4) 데이터 효율적 훈련:**
- 10.2시간의 제한된 고품질 데이터로도 강력한 일반화 달성[1]
- Pose Simulation으로 훈련 중 미본 시나리오 명시적으로 노출[1]

**5) 단계적 훈련 파이프라인:**
- Stage별 목표 분리로 각 측면(모션, 품질, 연속성)에 최적화 집중[1]

#### 3.3 일반화 성능 향상 가능성[1]

**1) 도메인 다양성 확대:**
- 현재: 주로 댄스 시퀀스 (약 70%) + 일부 다큐멘터리 촬영
- 제안: 다양한 신체 활동(스포츠, 일상 활동, 극단적 움직임) 데이터 확충[1]

**2) 스타일 일반화:**
- 애니메이션, 만화, 아트워크 등 다양한 스타일의 훈련 데이터 추가[1]
- 스타일 전이 기법과 결합 고려[1]

**3) 신체 다양성:**
- 다양한 체형, 연령, 성별, 인종의 데이터 수집[1]
- 장애인 포함 모든 신체 유형 대응[1]

**4) 포즈 추정 견고성:**
- 노이즈 포함 포즈 데이터에 대한 명시적 훈련[1]
- 포즈 정정 모듈 개발로 추정 오류 보상[1]

**5) 계층적 조건 학습:**
- 추상적 모션 표현 학습 추가[1]
- 세맨틱 모션 설명(예: "점프", "회전")과 포즈 시퀀스 간 연결[1]

***

### 4. 앞으로의 연구 영향과 고려사항

#### 4.1 학술적 영향[2][3][4][5][6]

**I2V 패러다임의 새로운 방향:**
- SteadyDancer는 R2V 패러다임의 주류성을 도전하며, I2V 패러다임을 인간 애니메이션 연구의 중심에 위치시킴[1]
- 첫 프레임 보존의 중요성을 강조하여 후속 연구의 기준점 제시[1]

**조건 통합 방법의 혁신:**
- 채널별 연결과 조건 분리 전략은 다중 조건 생성 모델 설계의 새로운 패러다임[4]
- Decoupled-Condition Distillation은 조건부 네트워크에서 무조건 사전(unconditional priors) 주입의 효과적 방법 제시[2]

**시간적 모델링 개선:**
- Pose Simulation과 같은 데이터 중심 접근은 훈련 리소스 제한 상황에서의 모델 개선 전략 제시[1]

**벤치마크 기여:**
- X-Dance 벤치마크는 실제 세계 시나리오를 반영한 평가의 중요성 강조[1]
- 향후 모델 평가의 새로운 기준 제시[1]

#### 4.2 기술 트렌드와의 연관성[3][5][7][6][4]

**최신 연구 방향:**

**1) 일반화 성능 강화 (Generalization Enhancement):**
- **Animate-X++ (2025)**: 다양한 캐릭터 타입(인간, 의인화 캐릭터) 및 동적 배경 지원[5]
- **EvAnimate (2025)**: 이벤트 카메라를 활용한 로버스트한 모션 큐[4]
- **OmniHuman-1 (2025)**: 대규모 일반 비디오 생성 모델로의 확장[8]

→ **SteadyDancer와의 연계**: 포즈 모듈레이션 기법을 다양한 입력 모달리티와 결합

**2) 강력한 신원 보존 (Identity Preservation):**
- **Identity-Preserving Reward-guided Optimization (IPRO, 2025)**: RL 기반 신원 일관성 강화[6]
- **HumanRAM (2025)**: 3D 인간 재구성과 애니메이션 통합[9]

→ **SteadyDancer와의 보완**: 첫 프레임 보존 메커니즘과 신원 보상 최적화의 결합

**3) 복잡한 모션 처리 (Complex Motion Handling):**
- **HyperMotion (2025)**: 저주파 공간 특성 강화로 복잡한 인간 모션 처리[7]
- **MotionLab (2025)**: 모션 생성 및 편집 통합 프레임워크[10]

→ **SteadyDancer와의 발전**: 복잡한 모션 시나리오에서의 안정성 개선

**4) Human-Object Interaction (HOI):**
- SteadyDancer는 HOI 장면에서 객체의 물리적으로 타당한 모션 합성 달성[1]
- 향후 명시적 객체 모델링과의 통합 가능성[1]

**5) 효율성과 확장성:**
- **FexiAct (SIGGRAPH 2025)**: 이질적 시나리오에서의 유연한 행동 제어
- **LivePortrait (2025)**: 69M 고품질 프레임으로 확장된 효율적 초상화 애니메이션[11]

→ **SteadyDancer의 강점**: 훈련 리소스 대비 성능 효율성이 향후 모델 확장의 기초

***

#### 4.3 앞으로 연구 시 고려할 점[3][5][7][6][4][1]

**1) 데이터 다양성 확보:**
- 현재 10.2시간의 제한된 데이터에서 벗어나, LivePortrait의 69M 프레임 수준으로 확장 고려[11]
- 다양한 신체 유형, 의상, 환경, 모션 스타일 포함[1]

**2) 도메인 간 전이 (Cross-Domain Transfer):**
- 애니메이션, 게임 캐릭터, 3D 모델 등 다양한 도메인으로의 확대[5]
- 스타일 보존 메커니즘 개발[1]

**3) 포즈 표현의 고급화:**
- SMPL-X, HaMeR 등 고급 신체 모델 통합 고려[7][1]
- 손 동작, 얼굴 표정 등 세세한 부위 제어[1]
- 의미론적 모션 설명(semantic motion descriptions) 추가[10]

**4) 모션 불연속성 해결의 고도화:**
- 현재 Pose Simulation은 80% 극단적 불연속성 해결[1]
- 나머지 20%에 대한 더 정교한 시간적 모델링 필요[1]
- 예측적 모션 보간 모듈 개발[1]

**5) 포즈 추정 견고성:**
- 노이즈 포함 포즈에 대한 명시적 견고성 훈련[1]
- 포즈 오류 정정(pose error correction) 메커니즘[1]

**6) 계산 효율성과 실시간 성능:**
- 추론 속도 최적화[1]
- 경량 모델 버전 개발[1]

**7) Human-Object/Environment Interaction:**
- 명시적 객체/환경 모델과의 통합[1]
- 물리적 제약(physics constraints) 추가[1]

**8) 평가 메트릭 개선:**
- X-Dance와 같은 현실 세계 기반 벤치마크 확장[1]
- 인간 평가 프로토콜 표준화[1]
- 세밀한 신원 보존 측정 지표[6]

**9) 멀티모달 입력:**
- 오디오 기반 애니메이션(오디오-구동 토킹 헤드)[8]
- 텍스트 기반 모션 설명[10]
- 이벤트 카메라 데이터[4]

**10) 이론적 이해:**
- 왜 I2V 패러다임이 R2V보다 우수한지에 대한 이론적 분석[1]
- 조건 분리의 최적성에 대한 수학적 증명[1]

***

### 결론

**SteadyDancer**는 인간 이미지 애니메이션 분야에서 **첫 프레임 보존의 중요성**을 확립하고, **I2V 패러다임의 우월성**을 입증하는 획기적인 작업이다.[1]

조건 통합, 포즈 모듈레이션, 단계별 훈련이라는 세 가지 핵심 기술을 통해 **공간-시간적 정렬 오류 문제를 해결**하면서도 **훈련 효율성**을 극적으로 개선했다.[1]

향후 연구는 **도메인 다양성 확대**, **모션 표현의 고도화**, **포즈 추정 견고성 강화**, **Human-Object Interaction 명시적 모델링**에 집중해야 하며, 이러한 방향의 발전은 영화 제작, 광고, 게임, 가상현실 등 다양한 산업 응용으로의 확대를 가능하게 할 것이다.[2][3][5][4][1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/a5bee3db-8f90-41c7-844d-1d59e5302e45/2511.19320v1-abcugdoem.pdf)
[2](https://arxiv.org/abs/2405.18156)
[3](https://arxiv.org/abs/2410.10306)
[4](https://arxiv.org/abs/2503.18552)
[5](https://arxiv.org/abs/2508.09454)
[6](https://arxiv.org/html/2510.14255v3)
[7](https://arxiv.org/html/2505.22977v1)
[8](https://arxiv.org/html/2502.01061v2)
[9](https://dl.acm.org/doi/10.1145/3721238.3730605)
[10](https://arxiv.org/html/2502.02358v1)
[11](https://arxiv.org/html/2407.03168)
[12](https://ieeexplore.ieee.org/document/10981068/)
[13](https://arxiv.org/abs/2504.14373)
[14](https://www.ewadirect.com/proceedings/ace/article/view/25643)
[15](http://medrxiv.org/lookup/doi/10.1101/2025.02.23.25322754)
[16](https://arxiv.org/abs/2501.03880)
[17](https://arxiv.org/html/2502.09617v1)
[18](https://arxiv.org/html/2410.10306)
[19](https://arxiv.org/html/2406.01188)
[20](http://arxiv.org/pdf/2304.05684v1.pdf)
[21](https://arxiv.org/html/2412.02684v1)
[22](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09947.pdf)
[23](https://arxiv.org/html/2502.12080v1)
[24](https://openaccess.thecvf.com/content/CVPR2023/papers/Ni_Conditional_Image-to-Video_Generation_With_Latent_Flow_Diffusion_Models_CVPR_2023_paper.pdf)
[25](https://arxiv.org/html/2408.16506v2)
[26](https://www.sciencedirect.com/science/article/abs/pii/S1566253525006311)
[27](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07921.pdf)
[28](https://www.sciencedirect.com/science/article/abs/pii/S0020025520303078)
[29](http://openaccess.thecvf.com/content/CVPR2025/papers/Chang_X-Dyna_Expressive_Dynamic_Human_Image_Animation_CVPR_2025_paper.pdf)
