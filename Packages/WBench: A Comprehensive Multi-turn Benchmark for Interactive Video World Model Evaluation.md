
# WBench: A Comprehensive Multi-turn Benchmark for Interactive Video World Model Evaluation

**논문 정보**: Ying et al. (2025), arXiv:2605.25874
**소속**: Meituan (LongCat 팀)
**GitHub**: https://github.com/meituan-longcat/WBench

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

WBench는 20개의 비디오 월드 모델을 **5개 차원, 22개 메트릭**으로 평가하며, 현재 세계 모델들이 고품질 렌더링과 신뢰 가능한 제어 가능성·일관성·물리 준수를 아직 통합하지 못하고 있음을 체계적으로 진단합니다.

### 주요 기여 (GitHub README 기반)

WBench는 **289개의 케이스**와 **1,058개의 인터랙션 턴**을 포함하는 포괄적 평가 프레임워크로, **내비게이션(navigation), 피사체 행동(subject action), 이벤트 편집(event editing), 시점 전환(perspective switching)**의 4가지 인터랙션 유형을 다양한 장면과 시점에 걸쳐 다룹니다.

**통합 내비게이션 프로토콜**을 제안하여 텍스트, 6-DoF 카메라 포즈, 이산 행동 인터페이스를 연결하며, 서로 다른 모델 패밀리 간의 공정한 비교를 가능하게 합니다.

**22개의 자동 메트릭**이 5개의 상호보완적 차원에 걸쳐 인간 판단과 검증되어, 대규모 자동 평가의 신뢰성을 보장합니다.

---

## 2. 문제 정의 · 제안 방법 · 모델 구조 · 성능 향상 및 한계

### 2-1. 해결하고자 하는 문제

기존 비디오 생성 평가 벤치마크들은 주로 단일 턴(single-turn) 평가에 집중하거나, 멀티 턴 인터랙티브 설정에서 모델을 체계적으로 비교할 공통 기준이 부재했습니다.

인터랙티브 비디오 생성 모델들은 각자의 전용 벤치마크와 비공개 장면·궤적으로 평가되어 공정한 모델 간 비교가 불가능했으며, 기존 공개 벤치마크들은 표준화된 테스트 조건—동일한 장면, 동일한 행동 시퀀스, 통합된 제어 인터페이스—을 제공하지 못했습니다.

구체적으로 WBench가 해결하는 문제:
- **멀티 턴 인터랙션**에서의 일관성·제어 가능성 평가 부재
- 이질적인 제어 인터페이스(텍스트 / 포즈 / 이산 액션)를 가진 모델들의 **공정 비교** 불가
- 물리적 타당성, 장면 일관성을 포함한 **다차원 자동 평가** 체계 미흡

---

### 2-2. 제안하는 방법 및 평가 체계

#### 📐 5개 평가 차원 및 22개 메트릭

GitHub README에서 공개된 메트릭 이름에 기반하면 아래와 같이 구성됩니다:

| 차원 (Dimension) | 대표 메트릭 (확인된 범위) |
|---|---|
| **Rendering Quality** | `hpsv3_quality`, 시각적 타당성(visual plausibility) |
| **Controllability** | 내비게이션 정확도, 카메라 제어 준수도 |
| **Consistency** | `background_consistency`, `subject_consistency`, `photometric_consistency`, `geometric_consistency`, `spatial_consistency`, `perspective_consistency`, `segment_continuity` |
| **Physics Compliance** | 물리 법칙 준수 (중력, 충돌 등) |
| **Scene Dynamics** | `gated_spatial_consistency` 등 장면 변화 관련 |

> ⚠️ 5개 차원의 정확한 공식 명칭 및 전체 22개 메트릭의 완전한 목록은 arXiv 전문에서 확인이 필요합니다.

#### 📐 통합 점수 산출 (공개 정보 기반 추정 표현)

WBench는 각 차원 점수를 집계하여 모델별 종합 순위를 산출합니다. GitHub에서 공개된 실행 방식에 따르면:

$$\text{WBench Score} = \frac{1}{|\mathcal{D}|} \sum_{d \in \mathcal{D}} w_d \cdot S_d$$

여기서:
- $\mathcal{D}$: 5개 평가 차원의 집합
- $w_d$: 각 차원 $d$의 가중치
- $S_d$: 차원 $d$에서의 정규화된 점수

> ⚠️ 위 수식은 GitHub의 `main.py` 및 평균 기반 집계 방식에 근거한 일반적 표현이며, 논문 본문의 정확한 수식과 다를 수 있습니다.

#### 📐 멀티 턴 일관성 측정

멀티 턴 인터랙션 $t = 1, 2, \ldots, T$에 걸쳐 시각적 일관성을 측정하는 지표는 다음과 같은 형태로 표현할 수 있습니다:

$$\text{Consistency}_{t} = \text{sim}(f(V_t),\ f(V_{t-1}))$$

여기서:
- $V_t$: $t$ 번째 턴에서 생성된 비디오 세그먼트
- $f(\cdot)$: 특징 추출 함수 (DreamSim, RAFT optical flow 등 활용)
- $\text{sim}(\cdot, \cdot)$: 코사인 유사도 또는 지각 유사도

WBench는 **WorldScore, VBench, SAM2, Depth-Anything-V3, MegaSAM, DreamSim, HPSv3, AMT, RAFT, TransNetV2** 등 다수의 오픈소스 프로젝트를 기반으로 구축되었습니다.

#### 📐 6-DoF 카메라 포즈 제어 프로토콜

텍스트, 6-DoF 카메라 포즈, 이산 행동 인터페이스를 연결하는 **통합 내비게이션 프로토콜**을 통해 모델 패밀리 간 공정한 비교가 가능합니다.

6-DoF 포즈는 다음과 같이 표현됩니다:

$$\mathbf{P} = [\mathbf{R} \mid \mathbf{t}] \in \mathbb{R}^{3 \times 4}$$

여기서 $\mathbf{R} \in SO(3)$은 회전 행렬, $\mathbf{t} \in \mathbb{R}^3$은 이동 벡터입니다.

---

### 2-3. 모델 구조

WBench 자체는 **평가 프레임워크(벤치마크)**이며, 특정 생성 모델 아키텍처를 제안하는 논문이 아닙니다. 평가 대상인 20개 모델은 다음 3가지 제어 인터페이스 유형으로 분류됩니다:

```
WBench 지원 모델 유형
├── Type 1: 텍스트 기반 제어 모델 (Text-conditioned)
├── Type 2: 6-DoF 카메라 포즈 제어 모델 (Pose-conditioned)
└── Type 3: 이산 행동 인터페이스 모델 (Discrete-action)
```

평가 파이프라인 구조:

```
입력 (케이스 초기 프레임 + 인터랙션 시퀀스)
        ↓
멀티 턴 비디오 생성 (각 모델)
        ↓
5개 차원 × 22개 자동 메트릭 계산
        ↓
인간 판단 검증 (Human Judgment Validation)
        ↓
모델 순위 및 진단 분석
```

---

### 2-4. 성능 향상 및 한계

#### 성능 진단 결과

체계적 진단 결과, **현재 세계 모델들은 고품질 렌더링과 신뢰 가능한 제어 가능성·일관성·물리 준수를 아직 통합하지 못하고 있음**이 밝혀졌습니다.

22개 자동 메트릭은 인간 판단과 비교하여 검증되었으므로, 대규모 신뢰 가능한 자동 평가가 가능합니다.

#### 한계

- **289개 케이스**는 다양성 측면에서 더 큰 규모의 확장이 필요
- 텍스트·포즈·이산 행동 인터페이스를 통합하는 평가에서 각 모달리티 간 평가 공정성 확보의 어려움
- 물리 준수 평가는 시뮬레이션 기반 GT(Ground Truth) 없이 자동 메트릭만으로 수행됨
- 일부 메트릭(visual plausibility)은 vLLM 환경이 별도로 필요하여 재현성에 제약

---

## 3. 모델의 일반화 성능 향상 가능성

WBench가 일반화 성능 향상에 기여하는 메커니즘은 다음과 같습니다:

### 3-1. 다양한 인터랙션 유형을 통한 일반화 평가

4가지 인터랙션 유형(내비게이션, 피사체 행동, 이벤트 편집, 시점 전환)을 다양한 장면과 시점에서 평가함으로써, 특정 시나리오에 과적합된 모델과 실제로 일반화된 세계 이해를 갖춘 모델을 구별할 수 있습니다.

### 3-2. 멀티 턴 평가를 통한 장기 일관성 검증

단일 턴 생성과 달리, **1,058개의 인터랙션 턴**에 걸친 멀티 턴 평가는 모델이 이전 맥락을 유지하며 새로운 인터랙션에 반응하는 능력, 즉 **시간적 일반화 능력**을 직접 측정합니다.

$$\text{Temporal Generalization} = \frac{1}{T-1}\sum_{t=2}^{T} \mathcal{C}(V_t, \text{context}_{1:t-1})$$

여기서 $\mathcal{C}$는 현재 생성 비디오와 이전 맥락 간의 일관성 함수입니다.

### 3-3. 통합 제어 프로토콜의 일반화 효과

텍스트·6-DoF 포즈·이산 행동을 통합하는 프로토콜은 이질적인 모델 간의 공정 비교를 가능하게 하며, 이를 통해 **특정 제어 인터페이스에 종속되지 않는 일반화된 세계 모델 능력**을 측정할 수 있습니다.

### 3-4. 비교 벤치마크와의 일반화 관점 비교

| 벤치마크 | 일반화 관점 |
|---|---|
| **WBench** | 멀티 턴 + 4종 인터랙션 + 통합 제어 인터페이스 |
| WorldModelBench | 7개 도메인 × 56 서브도메인에서 물리·명령 준수 |
| VBench-2.0 | 5개 차원(인간 충실도·제어·창의성·물리·상식) 일반화 |
| Omni-WorldBench | 4D 공간-시간 인터랙션 응답 일반화 |

기존 평가 벤치마크들은 생성 모델의 시각적 충실도와 텍스트-비디오 정렬에만 집중하거나 정적 3D 재구성 지표에 의존하여 시간적 역학을 무시했으며, 세계 모델링의 미래는 공간 구조와 시간적 진화를 함께 모델링하는 **4D 생성**에 있으며, 이 패러다임의 핵심 능력은 인터랙션 행동이 공간·시간에 걸쳐 상태 전이를 어떻게 유도하는지 충실히 반영하는 **인터랙티브 응답 능력**입니다.

---

## 4. 향후 연구에 미치는 영향 및 고려 사항

### 4-1. 향후 연구에 미치는 영향

#### (A) 표준화된 멀티 턴 평가 패러다임 확립

WBench는 인터랙티브 비디오 세계 모델 평가에서 **멀티 턴 표준 프로토콜**을 제시함으로써, 향후 모델 개발 시 단순한 시각적 품질을 넘어 제어 가능성·일관성·물리 준수를 동시에 고려해야 함을 연구 커뮤니티에 시사합니다.

#### (B) 세계 모델의 병목 진단 기여

현재 모델들이 렌더링 품질과 제어 가능성·일관성·물리 준수를 동시에 달성하지 못하고 있다는 체계적 진단은, 향후 연구가 **어느 능력을 우선 개선해야 하는지**에 대한 명확한 방향을 제공합니다.

#### (C) 구현 가능성과 재현성

MIT 라이선스 하에 코드와 데이터를 공개하여, 후속 연구자들이 동일한 조건에서 새로운 모델을 평가하고 비교하는 것이 가능해졌습니다.

#### (D) 멀티모달 제어 통합 연구 촉진

통합된 행동 어휘(WASD 스타일 등)와 공통 제어 인터페이스 연구가 다양한 모델들 간의 사과-대-사과(apples-to-apples) 비교를 가능하게 한다는 방향성과 맞물려, WBench의 6-DoF 통합 프로토콜은 **멀티모달 세계 모델 훈련** 연구에 직접적인 영향을 줄 것입니다.

### 4-2. 앞으로 연구 시 고려할 점

#### 🔬 평가 설계 측면

1. **케이스 규모 확장**: 289개 케이스는 다양한 도메인(실내/야외, 로봇/자율주행/게임)을 충분히 커버하기에 제한적이므로, 도메인별 특화 서브셋 확장이 필요합니다.

2. **물리 GT 기반 평가**: 현재 물리 준수 평가는 자동 메트릭에 의존하는데, 초기 프레임을 제공하고 지속 영상을 생성한 후 SAM2 세그멘테이션 마스크와 실제 물리 시뮬레이션 GT를 비교하는 방식처럼 시뮬레이션 기반 Ground Truth를 활용하면 더욱 엄밀한 물리 평가가 가능합니다.

3. **개방형 도메인 일반화 평가**: 보상 기반 파인튜닝과의 통합, 더 어려운 물리 시나리오, 개방 도메인 일반화와 도메인 특화 역량을 함께 테스트하는 통합 하이브리드 프로토콜로의 발전이 필요합니다.

#### 🔬 모델 개발 측면

4. **제어-품질 트레이드오프 연구**: WBench 진단 결과에 따라, 렌더링 품질과 제어 가능성을 동시에 높이는 것이 핵심 연구 과제입니다. 이를 위한 손실 함수 설계:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{render}} \mathcal{L}_{\text{render}} + \lambda_{\text{ctrl}} \mathcal{L}_{\text{ctrl}} + \lambda_{\text{phys}} \mathcal{L}_{\text{phys}} + \lambda_{\text{cons}} \mathcal{L}_{\text{cons}}$$

5. **인간 피드백 기반 정렬**: 대규모 인간 레이블(67K 이상)을 크라우드소싱하고 정밀한 판별 모델을 파인튜닝하여 자동화 평가를 수행하며, 판별 모델의 보상을 최대화하는 방향으로 훈련하면 세계 모델링 능력이 실질적으로 향상됨이 WorldModelBench에서 확인되었으므로, WBench에서도 유사한 RLHF 접근법을 적용할 수 있습니다.

6. **4D 시공간 모델링**: 세계 모델링의 미래는 4D 생성(공간 구조 + 시간 진화의 결합 모델링)에 있으며, 핵심 능력은 인터랙션 행동이 어떻게 공간·시간에 걸쳐 상태 전이를 유도하는지 충실히 반영하는 것이므로, WBench의 멀티 턴 설계는 4D 세계 모델 개발의 평가 기반으로 확장될 수 있습니다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 연도 | 핵심 특징 | WBench와의 차별점 |
|---|---|---|---|
| **VBench** | 2023 | 16개 차원 단일 턴 평가, 시각 품질 중심 | WBench는 멀티 턴 + 인터랙션 제어 |
| **WorldModelBench** | 2025.02 | 7개 도메인 56 서브도메인, 물리 준수 + 명령 이행 | WBench는 멀티 턴 연속 인터랙션 특화 |
| **VBench-2.0** | 2025.03 | 인간 충실도·제어 가능성·창의성·물리·상식 5개 차원 | WBench는 6-DoF 통합 제어 + 멀티 턴 |
| **WorldBench(물리)** | 2025 | 개념 특화, 분리된 물리 평가 | WBench는 물리를 포함한 종합 인터랙션 평가 |
| **Omni-WorldBench** | 2026.03 | 4D 설정에서의 인터랙티브 응답 평가, Omni-WorldSuite + Omni-Metrics | WBench의 멀티 턴 설계와 상호보완적 |
| **WorldMark** | 2026.04 | 통합 행동 매핑 레이어, 500개 케이스, 시각 품질·제어 정렬·세계 일관성 | WBench는 더 다양한 인터랙션 유형 |

---

## 참고 자료

1. **WBench 공식 GitHub**: https://github.com/meituan-longcat/WBench (meituan-longcat, 2025)
2. **WBench arXiv 논문**: Ying et al., "WBench: A Comprehensive Multi-turn Benchmark for Interactive Video World Model Evaluation," arXiv:2605.25874, 2025
3. **WorldModelBench**: Li et al., "WorldModelBench: Judging Video Generation Models As World Models," arXiv:2502.20694, 2025
4. **VBench-2.0**: "VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness," arXiv:2503.21755, 2025
5. **Omni-WorldBench**: Wu et al., "Omni-WorldBench: Towards a Comprehensive Interaction-Centric Evaluation for World Models," arXiv:2603.22212, 2026
6. **WorldMark**: "WorldMark: A Unified Benchmark Suite for Interactive Video World Models," arXiv:2604.21686, 2026
7. **WorldBench(물리)**: "WorldBench: Disambiguating Physics for Diagnostic Evaluation of World Models," arXiv:2601.21282, 2025
8. **WorldBench(물리 시뮬레이션)**: https://world-bench.github.io/
