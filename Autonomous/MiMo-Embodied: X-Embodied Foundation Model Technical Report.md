# MiMo-Embodied: X-Embodied Foundation Model Technical Report

### 1. 핵심 주장 및 주요 기여

**MiMo-Embodied**는 자율주행과 embodied AI를 통합하는 **최초의 크로스-embodied 파운데이션 모델**입니다. 핵심 주장은 다음과 같습니다:[1]

**도메인 통합의 시너지**: 자율주행과 embodied AI는 공간 추론, 감지, 예측, 계획 능력을 공유하므로, 통합 학습 시 상호 강화 효과가 발생합니다.

**긍정적 전이**: 다중 단계 학습(multi-stage learning), 큐레이션된 데이터, CoT/RL 미세조정을 통해 도메인 간 강력한 긍정적 전이가 가능합니다.

**주요 성과**:
- 자율주행 12개 벤치마크에서 SOTA 달성
- Embodied AI 17개 벤치마크에서 SOTA 달성
- 29개 통합 벤치마크에서 전문화된 모델 능가

***

### 2. 해결하고자 하는 문제 및 제안 방법론

#### 2.1 기존 접근법의 한계[1]

**1. 통합된 Embodied VLM 부재**
- RoboBrain, VeBrain 등은 단일 도메인 특화
- 실내(embodied AI)와 실외(자율주행) 간 도메인 격차로 인한 일반화 어려움

**2. 불완전한 평가 체계**
- 각 모델이 자율주행 또는 embodied AI 중 하나만 평가
- 크로스-embodiment 능력의 종합적 평가 부재

**3. 공간 이해 능력 한계**
- 3D 공간 관계, 거리 추정, 깊이 추론 약세
- 멀티뷰 입력 처리 미흡

#### 2.2 모델 아키텍처[1]

MiMo-Embodied는 세 가지 핵심 컴포넌트로 구성:

$$\text{MiMo-Embodied} = \text{ViT} \oplus \text{Projector(MLP)} \oplus \text{LLM}$$

**1. Vision Transformer (ViT)**:
- MiMo-VL의 비전 인코더 활용
- 단일 이미지, 다중 이미지, 비디오 처리 가능
- 자기-주의 메커니즘으로 복잡한 패턴 추출

**2. 프로젝터 (MLP)**:
$$\text{proj}(\mathbf{v}) = \text{MLP}(\mathbf{v}) \in \mathbb{R}^{d_{\text{LLM}}}$$

시각 토큰을 LLM 입력 공간으로 변환

**3. 대형 언어 모델 (LLM)**:
- 텍스트 이해 및 추론 담당
- MiMo-VL의 사전 학습된 가중치 상속

#### 2.3 데이터 구성 전략[1]

**1. 일반 데이터셋 (42.3%)**
- 시각 그라운딩: 객체 수준의 세밀한 지역화
- 문서/차트 이해: 구조적 텍스트, 테이블 처리
- 비디오 이해: 밀집 이벤트 캡셔닝
- 멀티모달 추론: 수학, 과학 논역

**2. Embodied AI 데이터셋 (42.6%)**

어포던스 예측:
$$\text{Affordance} = \text{Object-level} \cup \text{Scene-level}$$

- PixMo-Points, RoboAfford, RoboRefIt로부터 수집
- 객체별 상호작용 추론 및 장면 수준 능력

고수준 작업 계획:
$$P(\text{action}_t | \text{observation}_t, \text{goal}) \propto P(\text{observation}|\text{action}) \cdot P(\text{action}|\text{context})$$

- Cosmos-Reason1, EgoPlan-IT, RoboVQA 활용
- 장시간 비디오 이해 및 인과 추론

공간 이해:
- SQA3D, VLM-3R, RefSpatial, EmbSpatial 활용
- 3D 시각 그라운딩 및 에고센트릭 공간 추론

**3. 자율주행 데이터셋 (15.1%)**

환경 인식:
- CODA-LM, DriveLM, nuScenes-QA
- 장면 전체 이해 및 지역 객체 인식

상태 예측:

```math
\text{Intent}(t+\Delta t) = f(\text{history}, \text{context}, \text{agent\_state})
```

주행 계획:
- 행동 결정, 안전성 검증, 설명 가능한 추론

#### 2.4 4단계 훈련 전략[1]

**Stage 1: Embodied AI 감독 미세조정**

$$\mathcal{L}_{\text{S1}} = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{general}} \cup \mathcal{D}_{\text{embodied}}} [\text{CE}(f(x), y)]$$

일반 데이터와 embodied 데이터로 핵심 능력 구축

**Stage 2: 자율주행 감독 미세조정**
- 이전 단계에 자율주행 데이터 추가
- 크로스-도메인 혼합 감독 학습

**Stage 3: Chain-of-Thought 미세조정**

$$\mathcal{L}_{\text{CoT}} = -\sum_{i} \log P(a_i | o_i, r_1 \ldots r_{i-1})$$

명시적 추론 체인으로 복잡 추론 능력 강화

**Stage 4: 강화학습 미세조정 (GRPO)**

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}_{g \sim \mathcal{G}} \left[ \min \left( R(g) \frac{\pi_{\text{new}}(a|x)}{\pi_{\text{old}}(a|x)}, \text{clip}(R(g), 1-\epsilon, 1+\epsilon) \right) \right]$$

***

### 3. 성능 향상 및 일반화 능력 (중점)

#### 3.1 정량적 벤치마크 성능[1]

**Embodied AI 성능:**

| 능력 | 벤치마크 | MiMo-Embodied | RoboBrain-2.0 | 향상도 |
|------|---------|--------------|--------------|-------|
| 어포던스 예측 | VABench-Point | 46.93% | 26.67% | **+20.26%** |
| | RoboAfford-Eval | 69.81% | 51.46% | **+18.35%** |
| 작업 계획 | RoboVQA | 61.99% | 46.32% | **+15.67%** |
| 공간 이해 | CV-Bench | 88.82% | 85.75% | **+3.07%** |

**자율주행 성능:**
- CODA-LM: 58.55% (업계 표준 45.46% 대비 **+13.09%**)
- DRAMA: 76.14% (specialist 68.40% 대비 **+7.74%**)
- MAPLM: 74.52% (업계 표준 71.76% 대비 **+2.76%**)
- BDD-X: 52.18% (specialist 48.61% 대비 **+3.57%**)

#### 3.2 일반화 성능 향상의 핵심[1]

**절제 연구(Ablation Study) 결과:**

| 모델 구성 | Embodied 평균 | AD 성능 | 개선도 |
|---------|-------------|--------|-------|
| MiMo-VL (기준) | 46.2% | 32.2% | - |
| + Embodied만 | 56.9% | 57.6% | 제한적 |
| + AD만 | 43.2% | 57.5% | 역효과 |
| Embodied+AD (직접 혼합) | 58.4% | 55.2% | 간섭 발생 |
| **MiMo-Embodied (4단계)** | **62.4%** | **63.3%** | **+8.1%** |

**핵심 발견**: 멀티스테이지 전략이 직접 혼합보다 **4% 이상 우수**

#### 3.3 일반화 메커니즘 분석[1]

**1. 공간 추론의 공통 기반**

$$\text{Spatial Capability} = \alpha \cdot \text{Embodied}_{\text{spatial}} + \beta \cdot \text{Driving}_{\text{spatial}}$$

- RefSpatial-Bench에서 embodied 학습이 driving 성능 **+15.5%** 향상
- MAPLM에서 driving 학습이 embodied 네비게이션 **+8.2%** 향상

**2. 예측 능력의 전이**

- 자율주행의 다중 에이전트 예측(MME-RealWorld)이 embodied 작업 계획(RoboVQA)에 **+12.3%** 기여
- 역방향 전이: embodied 계획이 주행 추론에 **+7.8%** 기여

**3. 미학습 시나리오 성능 (NAVSIM)**

| 주행 상황 | 상대 개선도 |
|---------|----------|
| U-Turn | +8.1% |
| 교차로 좌회전 | +9.6% |
| 교차로 우회전 | +8.4% |
| 효율성 중시 차선 변경 | +9.9% |
| 안전성 중시 차선 변경 | +3.8% |

**평균**: **+7.7%** 상대 성능 개선

**특징**: 복잡한 상호작용 시나리오(우회, 차선 변경)에서 더 큰 개선

***

### 4. 주요 한계점

#### 4.1 모델 아키텍처 한계[1]

**1. 토큰 길이 제한**: 최대 32,768 토큰으로 매우 장기 비디오 시퀀스 처리 불가

**2. 실시간 성능**: 추론 시간이 실제 로봇 제어 또는 자동 주행 시스템 배포에 부족할 수 있음

**3. 3D 인식 부재**: RGB 기반 처리로 LiDAR, 레이더 등 다중 센서 정보 활용 불가

#### 4.2 데이터 관점 한계[1]

**1. Embodied AI 데이터 부족**: 시뮬레이션 데이터 > 90%
**2. 자율주행 시뮬레이션 의존성**: 실제 주행 데이터 제한
**3. 극단적 상황 데이터 부족**: 위험 상황에 대한 데이터 부재

#### 4.3 일반화 한계[1]

**1. 도메인 특수성 상충**: 두 도메인의 상충하는 요구사항이 학습 시 간섭 가능

**2. 미학습 환경 적응**: 완전히 새로운 환경(새로운 로봇 형태, 미지의 도로)에서의 성능 미검증

**3. 실제 배포 격차**: 벤치마크와 실제 물리 환경 간의 시뮬레이션 격차 존재

***

### 5. 앞으로의 영향 및 연구 시 고려사항

#### 5.1 학계에 미치는 영향

**새로운 패러다임 제시**:[2][3][1]
- 더 많은 도메인 통합 시도 촉발
- 크로스-도메인 전이 이론 발전
- 평가 메트릭의 표준화

**후속 연구 방향**:
- 의료 로봇 + 산업용 로봇 통합
- 해양 탐사 로봇 + 항공 드론 통합
- 부정적 전이 예방 메커니즘

#### 5.2 향후 연구 시 고려할 점

**1. 데이터 전략 개선**[4][1]

현재 한계:
- Embodied AI: 시뮬레이션 데이터 > 90%
- 자율주행: 시뮬레이션 + 합성 데이터 > 70%

개선 방향:
- 다양한 로봇 플랫폼의 실제 데이터 수집
- 극단적 주행 상황 데이터 확보
- 인간-로봇 상호작용 데이터 보강

**2. 모델 아키텍처 혁신**[5][6][1]

멀티모달 센서 통합:
$$\text{Multimodal Fusion} = \text{Attention}(\text{Vision}, \text{LiDAR}, \text{Radar}, \text{Audio})$$

- 3D 포인트 클라우드 처리 모듈 추가
- 센서 퓨전 시 정보 손실 최소화
- 센서 간 불확실성 모델링

계층적 추론 아키텍처:
- Level 1: 빠른 반응 (즉각적 인식 및 기본 계획)
- Level 2: 중간 추론 (상세 분석 및 상황 판단)
- Level 3: 심화 추론 (복합 계획 및 설명)

**3. 학습 전략 고도화**[7][8][1]

더욱 정교한 멀티스테이지 학습:
- S2.5: 교차 도메인 대비(Cross-Domain Contrastive) 학습
- S3.5: 도메인 특수 미세조정
- S5: 연속학습(Continual Learning) 능력 추가

강화학습 심화:
$$\text{Reward} = w_1 \cdot \text{Task Success} + w_2 \cdot \text{Safety} + w_3 \cdot \text{Efficiency}$$

- 다중 목표 최적화 (Multi-Objective RL)
- 안전 제약 조건 통합 (Constrained RL)
- 인간의 피드백 활용 (RLHF) 고도화

메타학습 기반 접근:[9][1]
$$\text{Meta-Model}: \theta^* = \arg\min_\theta \sum_{D} \mathcal{L}(D; \theta)$$

- 빠른 적응(Few-Shot Adaptation) 능력
- 새로운 도메인에 대한 빠른 수렴

**4. 일반화 능력 강화 전략**

도메인 일반화 이론 개발:[10][1]
$$\text{Generalization Error} = \mathbb{E}_{\text{new domain}}[\text{Loss}] \leq \text{Training Error} + \text{Complexity Penalty} + \text{Domain Gap}$$

물리적 제약 조건 통합:
$$\text{Action Space} = \{\mathbf{a} : g(\mathbf{a}) \leq 0, h(\mathbf{a}) = 0\}$$

- 로봇의 기구학적 제약 모델링
- 차량의 동역학 제약 반영
- 안전 제약 조건 학습

***

### 6. 2020년 이후 관련 최신 연구 탐색

#### 6.1 Vision-Language 모델 발전 경로[11][3][12]

**초기 단계 (2020-2022)**:
- **CLIP (2021)**: 제로샷 인식 능력으로 VLM 기초 정립[11]
- **BLIP (2022)**: 멀티모달 이해와 생성 능력 통합

**최근 발전 (2023-2025)**:
- **LLaVA 계열** (2023-2024): 비전 인코더 + LLM 결합으로 VQA 능력 향상
- **Qwen2.5-VL (2024-2025)**: 초고해상도 입력(13MP 이미지) 처리 및 멀티프레임 비디오 이해[13]
- **InternVL3.5 (2025)**: 8B 파라미터로 효율성 우수[8]

#### 6.2 Embodied AI 파운데이션 모델 진화[14][2]

**초기 연구 (2020-2022)**:
- **RT-1**: 로봇 조작에 Transformer 첫 적용
- **CLIPort**: CLIP 기반 포인팅 작업
- 주로 단일 작업 또는 단일 로봇 특화

**파운데이션 모델 시대 (2023-2024)**:
- **PaLM-E (2023)**: 대규모 LLM + 로봇 관찰 통합, 다양한 에구현에서 양성 전이 입증[12]
- **RT-2 (2023)**: VLM 활용한 로봇 제어로 제로샷 새 작업 학습 가능
- **Vision-Language-Action (VLA) 폭증 (2024-2025)**:
  - **Octo (2024)**: 다중 로봇, 다중 작업 학습
  - **OpenVLA (2024)**: 오픈소스 VLA 표준화
  - **GATO 파생 모델들**: 범용 에이전트 지향

**공간 추론 특화 (2024-2025)**:[6][15][5]
- **SpatialBot (2024)**: RGB + 깊이 이미지로 공간 QA 데이터셋 개발
- **SpatialRGPT (2024)**: 3D 공간 추론에 특화, 실내/실외/시뮬레이션 벤치마크
- **Seeing Across Views (MV-RoboBench, 2024)**: 멀티뷰 입력 처리의 중요성 강조

#### 6.3 자율주행 기술 발전[16][17][18]

**초기 시대 (2020-2022)**:
- **Transfuser (2021-2022)**: RGB + BEV 멀티모달 입력으로 Transformer 자율주행 적용

**파운데이션 모델 시대 (2023-2024)**:
- **DriveLM (2024)**: 대규모 주행 시나리오에서 VLM 평가, 환경 인식/상태 예측/계획 벤치마크[16]
- **RoboTron-Drive (2024)**: 전문화된 자율주행 VLM[19]
- **DriveLMM-o1 (2024-2025)**: 복잡한 추론 경로와 CoT 스타일 설명[17]

**End-to-End 자율주행 (2024-2025)**:
- **NAVSIM 기반 평가**: 궤적 계획 중심, 시뮬레이션 메트릭 표준화

#### 6.4 크로스-도메인 전이 연구[20][2][19]

**이론적 토대 (2020-2023)**:
- **Bridge Data (2022)**: 다중 도메인, 다중 작업 데이터의 중요성으로 도메인 격차 해소 입증

**최근 응용 (2024-2025)**:
- **GenRL (2024)**: 멀티모달 파운데이션 세계 모델로 VLM과 RL 통합[20]
- **Redefining Robot Generalization (2025)**: 상호작용형 멀티에이전트 관점 제시[9]

#### 6.5 Chain-of-Thought 및 강화학습[7][8]

**CoT 진화 (2022-2025)**:
- **초기 CoT (2022-2023)**: 언어 모델의 단계별 추론으로 수학 문제 해결 성능 향상
- **Embodied CoT (ECoT, 2024)**: 로봇 제어에 CoT 적용, 계획 → 부작업 → 움직임 → 시각 특징 추론[8]
- **Long CoT 연구 (2025)**: 장시간 추론 체인의 효과 분석, RL 통한 오류 수정 능력[7]

**RL 기법 진화**:
- **GRPO (2024)**: 그룹 상대 정규화로 보상 신호 노이즈 강건성[1]
- **DRL 기반 자율주행 (2024-2025)**: 깊은 강화학습으로 주행 정책 학습[17]

#### 6.6 데이터셋 및 벤치마크 발전[19][16][1]

**로봇 조작 데이터셋**:
- **BridgeData V2 (2023)**: 다중 로봇 조작 데이터
- **Open X-Embodiment (2023)**: 다양한 에구현 통합 데이터
- **Kaiwu (2025)**: 멀티모달 조립 작업 데이터[21]

**평가 표준화 (2024-2025)**:
- 단순 정확도에서 해석 가능성 평가로 확대
- 안전성(Safety) 중심 메트릭 추가
- 일반화 능력(Generalization) 평가 강조

***

### 7. 최종 평가 및 전망

MiMo-Embodied는 다음과 같은 중요한 지위를 차지합니다:

**기술적 마일스톤**: 최초의 통합 크로스-embodied 파운데이션 모델로 자율주행과 embodied AI의 수렴을 증명[1]

**방법론 기여**: 멀티스테이지 학습 프레임워크의 유효성을 실증적으로 입증하여, 도메인 간 간섭 완화 가능성 제시[1]

**산업 가능성**: 실제 로봇 조종 및 자동차 시스템 통합의 실현 가능성을 시연[1]

**일반화 성능**: 절제 연구를 통해 크로스-도메인 긍정적 전이가 구체적으로 입증되었으며, 미학습 시나리오에서도 평균 7.7% 성능 향상[1]

향후 연구는 실제 환경 배포 시 성능 격차 해소, 극단적 상황 대응 능력 강화, 안전성 검증 표준화에 집중해야 할 것으로 예상됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/481d2e16-c4d6-49dd-9546-b625bd4d7349/2511.16518v1.pdf)
[2](https://arxiv.org/abs/2402.02385)
[3](https://arxiv.org/pdf/2405.14093.pdf)
[4](https://arxiv.org/abs/2412.14989)
[5](https://ieeexplore.ieee.org/document/11128671/)
[6](https://arxiv.org/abs/2510.19400)
[7](https://huggingface.co/papers/2502.03373)
[8](https://arxiv.org/abs/2407.08693)
[9](https://arxiv.org/pdf/2502.05963.pdf)
[10](https://arxiv.org/html/2501.18592)
[11](https://arxiv.org/abs/2405.14093)
[12](http://arxiv.org/pdf/2303.03378.pdf)
[13](https://arxiv.org/abs/2506.10172)
[14](https://arxiv.org/abs/2311.14379)
[15](https://papers.nips.cc/paper_files/paper/2024/file/f38cb4cf9a5eaa92b3cfa481832719c6-Paper-Conference.pdf)
[16](https://ieeexplore.ieee.org/document/11093228/)
[17](https://www.sciencedirect.com/science/article/pii/S0957417425039533)
[18](http://arxiv.org/pdf/2310.17642v1.pdf)
[19](https://www.roboticsproceedings.org/rss18/p063.pdf)
[20](http://arxiv.org/pdf/2406.18043.pdf)
[21](https://arxiv.org/html/2503.05231v1)
[22](https://ieeexplore.ieee.org/document/11049053/)
[23](https://arxiv.org/abs/2509.06768)
[24](https://arxiv.org/abs/2501.01141)
[25](https://arxiv.org/abs/2509.14687)
[26](https://arxiv.org/html/2502.09560v1)
[27](https://arxiv.org/pdf/2403.09631.pdf)
[28](http://arxiv.org/pdf/2305.15021v2.pdf)
[29](https://pmc.ncbi.nlm.nih.gov/articles/PMC12088599/)
[30](https://arxiv.org/pdf/2311.04193.pdf)
[31](http://arxiv.org/pdf/2502.15336.pdf)
[32](https://pmc.ncbi.nlm.nih.gov/articles/PMC11249913/)
[33](https://arxiv.org/html/2507.00236v1)
[34](https://space-in-vision-language-embodied-ai.github.io)
[35](https://www.understandingai.org/p/how-transformer-based-networks-are)
[36](https://www.sciencedirect.com/science/article/abs/pii/S2452414X25001669)
[37](https://www.nature.com/articles/s41598-025-92701-6)
[38](https://github.com/jonyzhang2023/awesome-embodied-vla-va-vln)
[39](https://www.themoonlight.io/en/review/survey-of-vision-language-action-models-for-embodied-manipulation)
[40](https://arxiv.org/abs/2412.04368)
[41](https://www.semanticscholar.org/paper/b021962b5ecd1fe2d94b5488ec0ed99004b8585a)
[42](https://www.sciltp.com/journals/ijndi/2024/2/410)
[43](https://www.mdpi.com/2076-3417/14/19/8868)
[44](https://arxiv.org/abs/2404.01571)
[45](https://arxiv.org/abs/2407.15086)
[46](https://ieeexplore.ieee.org/document/10754333/)
[47](https://link.springer.com/10.1007/s11633-023-1431-y)
[48](https://arxiv.org/pdf/2306.05716.pdf)
[49](http://arxiv.org/pdf/2305.10455.pdf)
[50](https://arxiv.org/pdf/2305.11176.pdf)
[51](https://www.sciencedirect.com/science/article/abs/pii/S0925231225006356)
[52](https://openreview.net/pdf?id=T2IHuIib74)
[53](https://arxiv.org/pdf/2406.18043.pdf)
[54](https://www.arxiv.org/abs/2510.17057)
[55](https://github.com/JeffreyYH/Awesome-Generalist-Robots-via-Foundation-Models/blob/master/README.md)
[56](https://openreview.net/forum?id=Kzj3nBorD8)
[57](https://aclanthology.org/2025.emnlp-main.843.pdf)
[58](https://blog.iese.edu/artificial-intelligence-management/2024/chain-of-thought-reasoning-the-new-llm-breakthrough/)
