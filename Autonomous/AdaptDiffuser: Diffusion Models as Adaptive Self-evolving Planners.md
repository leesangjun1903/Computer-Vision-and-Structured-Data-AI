# AdaptDiffuser: Diffusion Models as Adaptive Self-evolving Planners

### 1. 핵심 주장 및 주요 기여 요약

**AdaptDiffuser**는 오프라인 강화학습에서 확산 모델의 기본적 한계인 훈련 데이터 부족으로 인한 궤적 품질 저하 문제를 해결합니다. 핵심 주장은 다음과 같습니다:[1]

- **기존 Diffuser의 한계**: 조건부 역확산 프로세스에서 생성된 궤적 품질이 학습된 평균 $$\mu_{\theta}$$와 분산 $$\Sigma$$에만 의존하여, 훈련 데이터 다양성 부족으로 제약됨

- **자기진화 메커니즘**: 보상 기울기를 이용한 합성 데이터 생성과 동역학 일관성 판별기를 통한 데이터 필터링으로 확산 모델을 반복 개선

- **일반화 능력 향상**: 추가 전문가 데이터 없이 새로운 과제에 27.9% 성능 향상 달성[1]

**주요 기여**:[1]
1. 보상 기울기 기반 합성 데이터와 판별기를 활용한 자기진화 확산 계획 프레임워크
2. 제로샷 적응을 통한 보지 못한 과제의 일반화
3. D4RL 벤치마크 및 신규 과제(KUKA, Maze2D)에서의 광범위 검증

***

### 2. 해결 문제, 제안 방법, 모델 구조 상세 분석

#### 2.1 문제 정의

오프라인 RL에서 확산 기반 계획의 조건부 역확산 프로세스:[1]

$$p_{\theta}(\tau^{i-1}|\tau^i, O_{1:T}) \approx \mathcal{N}(\tau^{i-1}; \mu_{\theta} + \alpha\Sigma g, \Sigma)$$

여기서 $$g = \nabla_{\tau}\mathcal{J}(\mu_{\theta}) = \sum_{t=0}^{T}\gamma^t \nabla_{s_t,a_t}R(s_t, a_t)$$

**핵심 문제**: 생성 궤적 품질이 $$\mu_{\theta}$$에 크게 의존하여, 훈련 데이터 부족 시 성능 저하[1]

#### 2.2 제안 방법론

**반복적 자기진화 프로세스**:[1]

$$\theta^*\_k = \arg\min_{\theta} -\mathbb{E}_{\hat{\tau}^k}[\log p_{\theta}(\hat{\tau}^k|y(\hat{\tau}^k))]$$

```math
\tau^{k+1} = \mathcal{G}(\mu_{\theta^*_k}, \Sigma, \nabla_{\tau}\mathcal{J}(\mu_{\theta^*_k}))
```

$$\hat{\tau}^{k+1} = [\hat{\tau}^k, \mathcal{D}(\hat{R}(\tau^{k+1}))]$$

**세 가지 보상 함수 유형**:[1]

1. **연속 보상**: Eq. (9) 직접 적용
2. **희소 보상**: 목표 조건 제약으로 변환
3. **조합**: 보조 보상 함수 정의 $$\mathcal{J}(\tau) = \sum_{t=0}^{T}\|s_t - s_c\|_p$$

**동역학 일관성 검증**:[1]
- 역동역학 모델: $$\tilde{a}\_t = I(s_t, s_{t+1})$$
- 순방향 동역학: $$\tilde{s}_{t+1} = T(\tilde{s}_t, \tilde{a}_t)$$
- 필터링 기준: $$d = \|\tilde{s}\_{t+1} - s_{t+1}\|_2 < \epsilon$$

#### 2.3 모델 구조

**네트워크 아키텍처**:[1]
- Temporal U-Net: 6개 반복 잔차 블록
- 각 블록: 시간 컨볼루션 + 그룹 정규화 + Mish 활성화
- 타임스텝 임베딩: 완전 연결층으로 생성

**훈련 설정**:[1]
- 최적화: Adam (학습률 $2 \times 10^{-4}$, 배치 크기 32)
- 확산 스텝: MuJoCo 100, Maze2D 64-256, KUKA 1000

***

### 3. 성능 향상 및 한계

#### 3.1 성능 개선 결과

**Maze2D 벤치마크**:[1]
- Large: 123.0 → 167.9 (+36.5%)
- Medium: 121.5 → 129.9 (+7.0%)
- **평균 개선**: 20.8%

**MuJoCo 환경**:[1]
- Hopper-Medium: 74.3 → 96.6 (+30.0%)
- Walker2d-Medium: 79.6 → 84.4 (+6.1%)
- **전체 평균**: 77.5 → 83.4 (+7.5%)

**KUKA Pick-and-Place (보지 못한 과제)**:[1]
- Setup 1: 28.16 → 36.03 (+27.9%)
- 추가 전문가 데이터 없음

**개선의 원인**:[1]
- 저품질 데이터("Medium")에서 특히 큰 개선
- 합성 데이터로 훈련 분포 개선
- 동역학 검증으로 궤적 실행 가능성 보장

#### 3.2 한계 분석

**계산 비용**:[1]
- 합성 데이터 생성: 3-16시간
- 세부조정: 6-7시간
- Diffuser 기본 훈련: 36-45시간
- **총 추가 시간**: ~13시간

**추론 시간**: 거의 동일 (1.4-1.6초, MuJoCo)[1]

**고차원 관찰 공간**:[1]
- 현재: 저차원 상태 공간만 평가
- 제안 해결책: 오토인코더 기반 잠재 공간 활용

***

### 4. 일반화 성능 향상 가능성

#### 4.1 일반화 개선 메커니즘

**다양한 목표 조건화**:[1]
- 훈련 중 여러 목표에 대한 궤적 생성
- Maze2D: 1개 훈련 목표 → 1M 합성 궤적
- 결과: 보지 못한 목표 조합에 강건성

**반복적 개선 효과**:[1]
- 저품질 데이터(Medium): 1차에서 2차로 1-2% 추가 개선
- 고품질 데이터(Medium-Expert): 포화로 미미한 개선

**판별기의 중요성**:[1]
동역학 일관성이 일반화의 핵심:

$$P(\text{성능}) \propto P(\|\tilde{s}_{t+1} - s_{t+1}\|_2 < \epsilon)$$

#### 4.2 보지 못한 과제 적응

**메커니즘**:[1]

$$\mathcal{J}_{\text{적응}} = \sum_{t=0}^{T}\|s_t - s_{\text{목표}}\|_p$$

훈련 중 다양한 거리에 대한 학습 → 새로운 목표에 용이한 적응

**극단적 사례 한계**:[1]
- 금화가 최적 경로에서 멀 때 성능 저하
- 해결책: 더 많은 합성 데이터 또는 다양한 초기화

***

### 5. 앞으로의 연구에 미치는 영향과 고려사항

#### 5.1 연구 영향

**새로운 패러다임**:[2][3]
- "사전 훈련 → 고정 모델"에서 "사전 훈련 → 반복적 개선"으로 전환
- 자기진화 개념의 일반화 가능성

**합성 데이터 품질 관리**:[4][5]
- 동역학 일관성 검증의 구체적 구현
- 다중 과제 환경으로 확장 가능

**OOD 일반화 평가**:[1]
- 의도적으로 설계된 평가 프레임워크
- 추가 데이터 없는 제로샷 적응 표준화

#### 5.2 향후 연구 시 고려 사항

**1. 고차원 관찰 공간**[1]
- 고려사항: 이미지 기반 과제 확장, VAE/Autoencoder 전처리
- 계산 비용 증가 관리

**2. 계산 효율성**[6][7]
- 병렬 데이터 생성: 현재 순차 → 10배 병렬 처리 가능
- 성능 포화 시점 조기 감지로 반복 중단

**3. 이론적 분석**
- 수렴성 증명: 충분한 다양성 조건 하에서 수렴성
- 일반화 경계: OOD 성능에 대한 상한 설정

**4. 도메인별 특화**[8][9][10]
- 로봇: 실제 로봇 검증, 심 투 리얼 전이
- 자율주행: nuPlan, CARLA 벤치마크 적용
- 안전성: 제약 조건 통합 (예: 속도, 토크 제한)

**5. 다중 작업 통합**[11]
- MultiTask 적응: 여러 과제 동시 학습
- Meta-Learning: MAML과의 결합 가능성

***

### 6. 2020년 이후 관련 최신 연구

#### 6.1 초기 단계 (2020-2022)

**Decision Transformer (2021)**:[12]
- 시퀀스 모델링 패러다임 도입
- 한계: 단계적 오류 누적

**Diffuser (2022)**:[13]
- 확산 모델을 궤적 생성에 적용
- AdaptDiffuser의 직접적 선행 연구

#### 6.2 확대 단계 (2023-2024)

**Decision Diffuser (2023)**[14]
- Classifier-free 지도
- 성능: 81.8 vs AdaptDiffuser 83.4 (+2.0%)

**DiffStitch (2024)**[12]
- 궤적 결합을 통한 데이터 증강
- 유사 동기, 다른 접근

**MODULI (2024)**[3]
- 다중 목표 오프라인 RL
- 분포 외 선호도 일반화

**MetaDiffuser (2023)**[14]
- 메타 오프라인 RL
- 보상/동역학 변화 적응

#### 6.3 효율성 개선 (2024-2025)

**DiffuserLite (2024)**[15]
- 계획 정제 프로세스(PRP)
- 추론 시간 단축

**Trajectory Diffuser (2024)**[16]
- 두 단계 분해
- 효율성과 품질 균형

#### 6.4 응용 확대 (2024-2025)

**로봇 공학**:[17][18]
- M2Diffuser: 모바일 조작
- DexHandDiff: 손가락 조작

**자율주행**:[19][20]
- 최적 가우시안 확산(OGD)
- 신호 없는 교차로 네비게이션

**이론 발전**:[21][22]
- Diffusion-DICE: 분포 정정 추정 관점
- Bellman Diffusion Models: 시간 차이 연결

#### 6.5 안전성 통합

**FISOR (2024)**[10]
- 하드 안전 제약 강제
- 도달 가능성 분석 활용

**Control Barrier Functions (2024)**[20]
- 동적 고차 제어 장벽
- 계획 중 안전성 보장

***

### 7. 결론

AdaptDiffuser는 확산 모델 기반 오프라인 RL의 혁신적 진전을 나타냅니다. **자기진화 메커니즘**을 통해 고정된 모델의 한계를 극복하며, **동역학 일관성 판별기**로 실행 가능성을 보장하고, **보지 못한 과제에 대한 제로샷 적응**으로 강력한 일반화 능력을 입증합니다.

최신 연구 동향은 **효율성**, **안전성**, **다중 목표**로의 확장을 보여주며, AdaptDiffuser의 자기진화 개념이 로봇공학, 자율주행, 게임 AI 등 다양한 분야에서 더욱 능력 있고 적응적인 시스템 개발의 기초가 될 것으로 기대됩니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/28ad83b5-cee7-448c-b77f-ded4434df94a/2302.01877v2.pdf)
[2](https://arxiv.org/abs/2405.19878)
[3](https://arxiv.org/abs/2408.15501)
[4](https://arxiv.org/abs/2405.19189)
[5](https://arxiv.org/abs/2407.00741)
[6](https://arxiv.org/abs/2407.20109)
[7](https://arxiv.org/abs/2405.20555)
[8](https://arxiv.org/abs/2405.19690)
[9](https://arxiv.org/abs/2406.12120)
[10](https://arxiv.org/abs/2401.10700)
[11](https://www.semanticscholar.org/paper/11afef7414d44476a335de9ea7e3400f30e49e05)
[12](https://arxiv.org/html/2402.02439v2)
[13](https://proceedings.mlr.press/v162/janner22a/janner22a.pdf)
[14](https://arxiv.org/pdf/2305.19923.pdf)
[15](https://arxiv.org/pdf/2401.15443.pdf)
[16](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06735.pdf)
[17](https://ieeexplore.ieee.org/document/10937276/)
[18](https://arxiv.org/html/2412.19500v1)
[19](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04239.pdf)
[20](https://ieeexplore.ieee.org/document/10995231/)
[21](https://papers.nips.cc/paper_files/paper/2024/file/b2fea79b1137d917e8b7cce9434ab5fa-Paper-Conference.pdf)
[22](https://openreview.net/pdf?id=1isC66Gozb)
[23](http://arxiv.org/pdf/2306.04875.pdf)
[24](http://arxiv.org/pdf/2405.19878.pdf)
[25](http://arxiv.org/pdf/2307.01472.pdf)
[26](http://arxiv.org/pdf/2406.09089.pdf)
[27](https://arxiv.org/html/2410.11338v1)
[28](https://arxiv.org/abs/2407.16142)
[29](https://arxiv.org/pdf/2509.19538.pdf)
[30](https://writer.com/engineering/self-evolving-models/)
[31](https://www.sciencedirect.com/science/article/abs/pii/S089360802500574X)
[32](https://openaccess.thecvf.com/content/CVPR2025/papers/Liang_DexHandDiff_Interaction-aware_Diffusion_Planning_for_Adaptive_Dexterous_Manipulation_CVPR_2025_paper.pdf)
[33](https://www.semanticscholar.org/paper/Guided-Flows-for-Generative-Modeling-and-Decision-Zheng-Le/b1d77c7921bcd79f520d252473225ac19d6b6289)
[34](https://arxiv.org/html/2403.09900v3)
[35](https://arxiv.org/abs/2502.17100)
[36](https://dergipark.org.tr/en/pub/anatomy/issue/73166/1194426)
[37](https://www.semanticscholar.org/paper/885e1b652813280a89faccfe59f1813d60f76d7e)
[38](https://link.aps.org/doi/10.1103/PhysRevE.107.014208)
[39](https://iopscience.iop.org/article/10.1088/1748-9326/acadf6)
[40](https://journals.lww.com/10.1097/HTR.0000000000000810)
[41](https://arxiv.org/abs/2303.12410)
[42](https://arxiv.org/abs/2509.11930)
[43](https://arxiv.org/html/2410.04261v1)
[44](https://arxiv.org/html/2409.16012v1)
[45](http://arxiv.org/pdf/2405.19232.pdf)
[46](https://arxiv.org/pdf/2401.02644.pdf)
[47](https://arxiv.org/pdf/2402.06559.pdf)
[48](https://arxiv.org/pdf/2405.01758.pdf)
[49](https://openreview.net/pdf?id=a7APmM4B9d)
[50](https://bimsa.net/doc/publication/4543.pdf)
[51](https://www.emergentmind.com/topics/diffusion-based-planning)
[52](https://hugrypiggykim.com/2022/03/29/decision-transformer-reinforcement-learning-via-sequence-modeling/)
[53](https://www.nature.com/articles/s41598-025-25949-7)
[54](https://velog.io/@dutch-tulip/decision-transformer)
[55](https://arxiv.org/abs/2401.02225)
[56](https://proceedings.iclr.cc/paper_files/paper/2025/file/f95606d8e870020085990d9650b4f2a1-Paper-Conference.pdf)
