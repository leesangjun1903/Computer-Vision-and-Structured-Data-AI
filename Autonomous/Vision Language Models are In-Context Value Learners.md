
# Vision Language Models are In-Context Value Learners

> **저자**: Yecheng Jason Ma et al. (Google DeepMind, UPenn 외)
> **발표**: arXiv:2411.04549 (2024.11.07), ICLR 2025 채택
> **공식 사이트**: [generative-value-learning.github.io](https://generative-value-learning.github.io)
> **논문 링크**: [arxiv.org/abs/2411.04549](https://arxiv.org/abs/2411.04549)

---

## 1. 핵심 주장 및 주요 기여 요약

이 논문은 **Generative Value Learning (GVL)**이라는 새로운 프레임워크를 제안하며, VLM을 활용하여 로봇 시스템의 태스크 **시간적 진행도(temporal progress)**를 예측한다.

GVL은 VLM에 내재된 세계 지식(world knowledge)을 활용하여 **태스크 진행도를 예측하는 범용 가치 함수 추정기(universal value function estimator)**이다.

### 핵심 주장:

VLM에게 단순히 비디오 시퀀스의 값을 예측하도록 요청하는 것은 연속 프레임 간의 강한 시간적 상관관계로 인해 성능이 낮다. 대신 GVL은 **셔플된(shuffled) 비디오 프레임에 대한 시간적 순서 문제**로 가치 추정을 재정의하여, VLM이 태스크 진행도를 기반으로 프레임을 구별할 수 있도록 의미적·시간적 이해 능력을 보다 충분히 활용하게 한다.

### 주요 기여:

| 기여 항목 | 내용 |
|---|---|
| GVL 프레임워크 | 로봇 학습 없이도 300+ 태스크에 적용 가능한 범용 가치 함수 |
| 프레임 셔플링 기법 | 시간적 편향 제거, VLM의 의미 추론 능력 극대화 |
| VOC 평가 지표 | 새로운 가치 함수 평가 메트릭 제안 |
| 다운스트림 응용 | 데이터셋 필터링, 성공 감지, 정책 학습 |

---

## 2. 해결하고자 하는 문제, 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

시각적 궤적에서 시간적 진행도를 예측하는 것은 학습하고 적응하며 개선할 수 있는 지능형 로봇에게 매우 중요하다. 그러나 다양한 태스크와 도메인에서 이러한 **진행도 추정기, 즉 시간적 가치 함수(temporal value function)**를 학습하려면 대량의 다양한 데이터와 스케일링·일반화가 가능한 방법이 모두 필요하다.

로봇 학습 문헌에서 in-context learning은 주로 액션 생성에 초점을 맞추었지만, 이러한 선행 연구들은 in-context learning 능력을 실현하기 위해 로봇 태스크에 대한 명시적이고 광범위한 학습이 필요하며, 좁은 태스크 분포에서만 일반화가 달성된다.

### 2.2 제안 방법 및 수식

#### 문제 형식화: Goal-Conditioned POMDP

GVL은 로봇 태스크를 **목표 조건부 부분 관찰 마르코프 결정 프로세스(goal-conditioned POMDP)**로 모델링한다.

수학적으로 POMDP는 다음과 같이 정의됩니다:

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{O}, T, R, \Omega, \gamma, \mathcal{G})$$

- $\mathcal{S}$: 상태 공간(state space)
- $\mathcal{A}$: 행동 공간(action space)
- $\mathcal{O}$: 관찰 공간(observation space)
- $T$: 전이 함수(transition function)
- $R$: 보상 함수(reward function)
- $\Omega$: 관찰 함수(observation function)
- $\gamma$: 할인 인자(discount factor)
- $\mathcal{G}$: 목표 공간(goal space)

#### 시간적 가치 함수 (Temporal Value Function)

목표 $g \in \mathcal{G}$에 대한 가치 함수 $V^*$는:

$$V^*(o_t, g) = \mathbb{E}\left[\sum_{k=t}^{T} \gamma^{k-t} R(o_k, g) \,\Big|\, o_t, g\right]$$

GVL은 이 $V^*(o_t, g)$를 VLM의 세계 지식을 통해 **추가적인 로봇 학습 없이** 근사합니다.

#### GVL 핵심: 자기회귀적 가치 예측 (Autoregressive Value Prediction)

GVL은 가치 추정을 **자기회귀적 next-token 예측 문제**로 프레임화하여, VLM이 셔플된 궤적 프레임 배치(batch)에 대한 태스크 진행도를 출력하도록 한다.

셔플된 프레임 집합 $\tilde{\mathcal{F}} = \{o_{\sigma(1)}, o_{\sigma(2)}, \ldots, o_{\sigma(N)}\}$ ($\sigma$: 랜덤 순열)에 대해:

$$\hat{V}(o_{\sigma(i)}, g) = \text{VLM}\bigl(\tilde{\mathcal{F}}, g, \mathcal{C}\bigr)_i, \quad i=1,\ldots,N$$

- $g$: 언어 또는 이미지 형태의 목표(goal)
- $\mathcal{C}$: in-context 예시 (zero-shot 시 $\mathcal{C} = \emptyset$)

GVL은 실제로 VLM이 0에서 100 사이의 **정수형 백분율 수치**를 출력하도록 요청한다.

실제 세계 로봇 비디오 데이터셋은 길이와 빈도가 다양하기 때문에, 입력 시퀀스가 **30 프레임**이 되도록 모든 비디오를 서브샘플링하여 데이터셋 간 비교 가능성을 보장한다.

#### 평가 지표: Value-Order Correlation (VOC)

GVL은 새로운 경량 평가 지표인 **Value-Order Correlation (VOC)**를 도입하며, 이는 예측된 가치와 입력 전문가 비디오의 시간 순서 간의 **순위 상관관계(rank correlation)**를 계산한다. VOC는 $-1$에서 $1$ 사이의 값이며, $1$은 두 순서가 완벽히 일치함을 나타낸다.

수식으로는 Spearman의 순위 상관계수를 활용:

$$\text{VOC} = \rho_s(\hat{V}, t) = 1 - \frac{6\sum_i d_i^2}{N(N^2-1)}$$

- $\hat{V} = (\hat{V}_1, \ldots, \hat{V}_N)$: 예측된 가치 시퀀스
- $t = (1, 2, \ldots, N)$: 실제 시간 순서
- $d_i$: $\hat{V}_i$의 순위와 $t_i$의 순위 차이

전문가 품질의 데모는 구조상 시간이 지남에 따라 단조 증가하는 가치를 가지므로, 좋은 가치 모델은 전문가 비디오에서 높은 VOC 점수를 가져야 한다. 반면 좋은 가치 모델을 고정했을 때, 낮은 품질의 궤적은 낮은 VOC 점수를 가져야 한다.

### 2.3 모델 구조

모든 실험에서 GVL의 백본 VLM으로 **Gemini-1.5-Pro**를 사용하였으며, 이 모델 선택에 대한 ablation을 통해 다른 VLM에서도 GVL이 효과적임을 확인하였다.

GVL의 전체 파이프라인:

```
[입력: 셔플된 N개의 프레임] + [목표 g (언어/이미지)] + [in-context 예시 C]
                    ↓
           Gemini-1.5-Pro (VLM)
                    ↓
     [각 프레임의 태스크 진행도 값 (0~100)]
                    ↓
        VOC 기반 평가 / 다운스트림 응용
```

### 2.4 성능 향상

로봇 또는 태스크 특정 학습 없이, GVL은 **300개 이상의 다양한 실제 태스크**에서 다양한 로봇 플랫폼(bimanual manipulation 태스크 포함)에 대해 in-context zero-shot 및 few-shot으로 효과적인 가치를 예측할 수 있다.

GVL의 성능은 언어 목표에서 LIV보다 현저히 우수하며, LIV의 예측은 임의 수준에 그쳐 임의의 미확인 로봇 비디오에 대한 밀도 있는 가치를 예측하기에 임베딩 공간의 지식이 충분하지 않음을 시사한다.

GVL은 60% 이상의 ALOHA bimanual 태스크에서 양의 상관관계를 가진 가치 예측을 생성하며, **중위 VOC 0.12**를 기록한다.

**다운스트림 응용 성능**:
VOC 점수를 성공 감지의 임계값 점수로 활용할 수 있으며, GVL-SD라는 성공 감지 방법은 동일한 VLM을 사용하는 SuccessVQA를 모든 평가 지표에서 크게 능가한다. 또한 GVL-SD를 이용한 필터링된 BC는 임계값과 무관하게 기본 모방 학습 알고리즘(ACT)을 항상 능가한다.

### 2.5 한계

이 논문은 이 접근법의 한계나 잠재적 편향을 깊이 탐구하지 않는다.

추가적으로 확인 가능한 한계점들:
- OXE 데이터셋은 주로 더 단순한, 단기 single-arm 태스크에 초점을 맞추고 있어, 장기 복잡 태스크에 대한 평가가 제한적이다.
- OpenGVL 후속 연구에서 평가한 결과, 오픈소스 모델 계열은 클로즈드소스 모델 대비 현저히 낮은 성능을 보여, GVL의 성능이 **Gemini-1.5-Pro와 같은 대형 상용 모델에 크게 의존**함을 시사한다.
- VLM은 프롬프트 변형에 대해 강건해야 하는데, 서로 다른 시스템 프롬프트 표현에 따른 VOC 점수 민감도 조사가 평가 프레임워크를 강화할 것이다.

---

## 3. 모델의 일반화 성능 향상 가능성

GVL의 일반화 능력은 세 가지 핵심 차원에서 분석됩니다.

### 3.1 Zero-Shot 일반화

GVL은 로봇 특정 파인튜닝 없이도 사전 학습된 VLM에서 유연한 멀티모달 in-context learning을 통해 시각적 가치 추정이 이미 가능함을 보여준다.

VLM은 가치 추정의 후보로 자주 고려되지 않지만, 핵심 도전을 잘 처리할 수 있다. 특히 최신 VLM은 다양한 비전 태스크에서 강력한 공간 추론 및 시간적 이해 능력을 보여주어, **새로운 시나리오로의 일반화**를 가능하게 한다.

### 3.2 In-Context Scaling (Few-Shot 일반화)

GVL은 매력적인 **in-context 스케일링**을 보여주며, in-context 예시 수가 증가할수록 평균 VOC 점수가 꾸준히 향상된다. 5개의 in-context 궤적(총 150개의 셔플된 이미지)으로도 GVL은 전체 컨텍스트를 활용하여 강력한 일반화를 보인다. 이는 **Gemini-1.5-Pro와 같은 최신 장문 컨텍스트 윈도우 VLM**이 범용 가치 함수로 재활용될 수 있음을 보여준다.

### 3.3 Cross-Embodiment 일반화 (다른 embodiment 간 전이)

In-context 예시는 로봇 데모에 국한되지 않으며, GVL의 장점 중 하나는 **다른 embodiment에서 온 데모로도 in-context learning의 혜택**을 받을 수 있다는 것이다. 구체적으로, 인간이 ALOHA 로봇과 동일한 태스크를 수행하는 것을 기록하여 이를 가치 예측의 in-context 예시로 활용한다.

GVL은 인간 비디오와 같은 **이종 태스크 및 embodiment의 예시를 통한 유연한 멀티모달 in-context learning**을 허용함이 실증되었다.

### 3.4 다운스트림 응용으로의 일반화

GVL은 이종 태스크와 embodiment의 예시를 통한 유연한 멀티모달 in-context learning을 허용하며, GVL의 범용성은 **데이터셋 필터링, 성공 감지, 어드밴티지 가중 회귀**를 포함하는 visuomotor 정책 학습과 관련된 다양한 다운스트림 응용을 가능하게 하며, 모두 모델 학습이나 파인튜닝 없이 동작한다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 연구 | 방법 | 일반화 방식 | GVL과의 차이 |
|---|---|---|---|
| **R3M** (2022) | 인간 비디오 자기지도 학습 | 제한적 도메인 | 로봇 특정 파인튜닝 필요 |
| **LIV** (2023, Ma et al.) | 대조 학습 + 가치 목적함수 | 인간 비디오 범위 내 | LIV는 in-the-wild 가치 추정을 위해 인간 비디오에서 파인튜닝된 대조 VLM이며, 목표 이미지/설명의 임베딩 거리로 가치 예측; GVL 대비 언어 목표 성능 열등 |
| **RT-2** (2023, Google) | VLA 모델, 행동 토큰 예측 | 웹 지식 전이 | 행동 생성 중심, 가치 추정 아님 |
| **SayCan** (2022, Ahn et al.) | LLM + 가치 함수 조합 | 언어 지시 기반 | 별도의 가치 함수 필요 |
| **GVL (본 논문, 2024)** | VLM 프레임 셔플링 | Zero/few-shot 범용 | 추가 학습 없이 300+ 태스크 적용 |
| **OpenGVL** (2025) | GVL 오픈소스 확장 | 오픈소스 VLM 평가 | GVL 접근법을 오픈소스 모델로 복제·확장한 OpenGVL 벤치마크를 개발 |
| **ROVER** (2025) | 재귀적 VLM 비디오 추론 | 서브태스크 분해 |  ROVER는 OpenX Embodiment 비디오에서 태스크 진행도 추정 등 세 가지 비디오 추론 태스크에서 강력한 베이스라인을 능가 |

### 비교 분석 요약

기존 LIV의 예측은 언어 목표에서 임의 수준이며, 이미지 목표에서도 단순 이미지 유사도 기반의 임베딩 공간이 타임스텝과 상관된 값을 만들 수 있어 더 단순한 문제임에도 불구하고 GVL이 더 높은 VOC를 기록한다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 연구에 미치는 영향

GVL은 태스크 진행도 예측을 위해 VLM을 효과적으로 활용하는 방향으로의 중요한 진전을 보여준다. 자기회귀적 예측, 입력 셔플링, in-context learning을 활용하여 태스크 특정 학습 없이 높은 수준의 일반화를 달성하며, 이 프레임워크는 다양한 실제 태스크와 환경이 제기하는 과제와 잘 부합하는 로봇 학습 시스템 강화의 새로운 방향을 열어준다.

구체적인 영향 방향:

1. **훈련-없는 보상 함수 설계**: GVL은 로봇 강화학습에서 별도의 보상 함수 설계 없이도 VLM의 사전 지식만으로 가치 함수를 근사하는 패러다임을 제시합니다.

2. **데이터 큐레이션 자동화**: 데이터 부족이 로봇공학의 가장 큰 제약 요인으로 남아 있는 가운데, 현장에서 이용 가능한 로봇 데이터의 양이 기하급수적으로 증가하고 있으며, 신뢰할 수 있는 시간적 태스크 완료 예측은 이 데이터를 대규모로 자동 주석 및 큐레이션하는 데 도움이 될 수 있다.

3. **오프라인 강화학습 통합**: GVL의 원시 가치 추정값은 실제 세계 오프라인 강화학습을 위한 **어드밴티지 가중 회귀(advantage-weighted regression)**에 활용될 수 있다.

### 5.2 앞으로 연구 시 고려할 점

1. **오픈소스 VLM 의존성 해소**: 오픈소스 모델 계열이 클로즈드소스 모델 대비 현저히 낮은 성능을 보이므로, 오픈소스 VLM에서도 GVL이 효과적으로 동작할 수 있도록 경량화 또는 특화 파인튜닝 전략이 필요합니다.

2. **장기 태스크 일반화**: NIST 보드에서 순차적으로 세 개의 기어를 제거하거나, 드레스를 8겹으로 접거나, 옷걸이에 티셔츠를 거는 등의 고난이도 장기 스킬에 대한 VOC 중위값(0.12)이 낮은 편으로, 장기·복합 태스크에 대한 추가 연구가 필요합니다.

3. **프롬프트 강건성**: 다양한 시스템 프롬프트 표현에 따른 VOC 점수 민감도를 조사하는 것이 평가 프레임워크를 강화할 것이다.

4. **멀티뷰 관찰 시스템 확장**: 향후 연구 방향은 멀티뷰 관찰 시스템과 태스크 진행도 예측 개선을 위한 VLM의 추가 최적화를 탐구할 수 있다.

5. **샘플링 전략 다양화**: 현재는 전문가 데모에서 균일하게 샘플링하고 있으나, 중요도 샘플링이나 계층화 샘플링과 같은 대안적 샘플링 전략에 따른 VOC 성능 변화를 탐구하면 모델 능력과 평가 강건성에 대한 더 깊은 통찰을 얻을 수 있다.

6. **실시간 적용 가능성**: 현재 GVL은 30프레임을 일괄 처리하므로, 온라인 정책 학습 및 실시간 로봇 제어에 적용하기 위한 스트리밍·온라인 추론 방법 개발이 필요합니다.

---

## 📚 참고 출처

1. **arXiv 원문**: Ma, Y.J. et al., "Vision Language Models are In-Context Value Learners," arXiv:2411.04549, 2024. https://arxiv.org/abs/2411.04549
2. **공식 프로젝트 페이지**: Generative Value Learning (GVL). https://generative-value-learning.github.io/
3. **OpenReview (ICLR 2025)**: https://openreview.net/forum?id=friHAl5ofG
4. **ICLR 2025 Poster**: https://iclr.cc/virtual/2025/poster/28853
5. **ResearchGate**: https://www.researchgate.net/publication/385630622
6. **Moonlight Literature Review**: https://www.themoonlight.io/en/review/vision-language-models-are-in-context-value-learners
7. **Dinesh Jayaraman 연구실 페이지**: https://www.seas.upenn.edu/~dineshj/publication/ma-2025-gvl/
8. **OpenGVL (후속 연구)**: arXiv:2509.17321, "OpenGVL - Benchmarking Visual Temporal Progress for Data Curation"
9. **ROVER (관련 후속 연구)**: arXiv:2508.01943, "ROVER: Recursive Reasoning Over Videos with Vision-Language Models for Embodied Tasks"
10. **LIV (비교 연구)**: Ma, Y.J. et al., "LIV: Language-Image Representations and Rewards for Robotic Control," 2023.
