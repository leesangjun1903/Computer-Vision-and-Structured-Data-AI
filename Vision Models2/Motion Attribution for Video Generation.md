
# Motion Attribution for Video Generation (Motive) 

> **논문 정보**
> - **제목**: Motion Attribution for Video Generation
> - **저자**: Xindi Wu 외 7인 (NVIDIA)
> - **arXiv**: [arxiv.org/abs/2601.08828](https://arxiv.org/abs/2601.08828) (2026년 1월 13일)
> - **학회**: **ICML 2026 Oral**
> - **공식 프로젝트 페이지**: [research.nvidia.com/labs/sil/projects/MOTIVE](https://research.nvidia.com/labs/sil/projects/MOTIVE/)

---

## 1. 핵심 주장 및 주요 기여 요약

### 🎯 핵심 주장

비디오 생성 모델의 급격한 발전에도 불구하고, 모션(motion)에 영향을 미치는 데이터의 역할은 거의 이해되지 않고 있었다. 이 논문은 **Motive (MOTIon attribution for Video gEneration)**를 제안하는데, 이는 현대의 대규모 고품질 비디오 데이터셋 및 모델로 확장 가능한 **모션 중심의 그래디언트 기반 데이터 어트리뷰션(data attribution) 프레임워크**이다.

### ✅ 주요 기여

1. **최초의 모션 어트리뷰션 프레임워크**: 기존 이미지 외관(visual appearance)이 아닌 **모션을 어트리뷰션하는 최초의 프레임워크**이며, 이를 파인튜닝 데이터 큐레이션에 활용한다.

2. **확장 가능한 그래디언트 기반 어트리뷰션**: 현대 대규모 고품질 데이터 및 대형 생성 모델 규모에서도 계산적으로 효율적인 **확장 가능한 그래디언트 기반 어트리뷰션 방법론**을 제안한다.

3. **비디오 고유의 편향 수정**: **프레임 길이 편향(frame-length bias)을 수정**하고, 다양한 비디오 길이와 대규모 모델 스케일을 처리하기 위한 **확장 가능한 그래디언트 계산 기법**을 통합한다.

4. **시공간 분리**: 모션 가중 그래디언트(motion-weighted gradients)를 이용해 **시간적 역학(temporal dynamics)을 정적 외관(static appearance)에서 분리**하여, 생성된 모션 패턴을 가장 영향력 있는 학습 클립으로 추적한다.

---

## 2. 문제 정의, 제안 방법, 모델 구조, 성능

### 🔴 2.1 해결하고자 하는 문제

기존의 Diffusion 모델 데이터 어트리뷰션 연구들은 이미지에 집중하여 **정적 컨텐츠(static content)를 설명**하는 데 그쳤다. 이 방법들을 비디오에 단순 확장하면 **모션이 외관(appearance)으로 붕괴(collapse)**되어, 비디오가 이미지와 구별되는 시간적 구조(temporal structure)를 놓치게 된다.

고품질 데이터는 특히 파인튜닝 단계에서 가장 중요하며, 여기서 대규모 사전학습 코퍼스는 접근 불가능하고, 신중하게 선택된 클립이 큰 영향을 미칠 수 있다. 따라서 모션 특화 어트리뷰션은 어떤 클립이 **시간적 일관성(temporal coherence)과 물리적 그럴듯함(physical plausibility)**에 가장 영향을 미치는지를 식별하는 파인튜닝 단계에서 특히 가치 있다.

---

### 🟢 2.2 제안하는 방법 (수식 포함)

#### ① 모션 감지 및 마스크 생성

모션 계산에는 **AllTracker**를 사용하여 픽셀 공간에서 모션 정보를 추출한다:

$$\mathbf{A} = \mathcal{A}(\mathbf{v}) \in \mathbb{R}^{F \times H \times W \times 4}$$

첫 두 채널에는 픽셀 변위(displacement)를 나타내는 **광학 흐름(optical flow)** 맵이 포함되며, 나머지 채널에는 가시성(visibility)과 신뢰도(confidence) 점수가 인코딩된다. 각 픽셀 위치에서 변위 벡터는 다음과 같이 추출된다:

$$\mathbf{D}_f(h, w) = (\mathbf{A}_{f,h,w,0},\ \mathbf{A}_{f,h,w,1}) = (d_w, d_h)$$

#### ② 모션 크기(Motion Magnitude) 계산

각 위치에서의 모션 크기(motion magnitude)는 다음과 같이 정의된다:

$$M_f(h, w) = \| \mathbf{D}_f(h, w) \|_2$$

프레임과 픽셀 간에 비교 가능한 모션 가중치를 얻기 위해 **min-max 정규화**를 수행한다:

$$\hat{M}_f(h, w) = \frac{M_f(h, w) - \min_{f', h', w'} M_{f'}(h', w')}{\max_{f', h', w'} M_{f'}(h', w') - \min_{f', h', w'} M_{f'}(h', w') + \zeta}$$

여기서 $\zeta = 10^{-6}$은 분모가 양수임을 보장한다.

#### ③ 모션 가중 그래디언트 (Motion-Weighted Gradient) 및 Influence Score

$\mathcal{L}_{\text{diff}}$를 diffusion loss로, 샘플링된 타임스텝-노이즈 집합 $\mathcal{T}$와 함께, **정규화된 테스트와 학습 그래디언트에 대한 코사인 스타일 점수(cosine-style score)**를 계산한다. 그래디언트를 $(t, \epsilon)$에 대해 평균화하면 추정치가 안정화되고, 정규화를 통해 타임스텝 유도 스케일 효과가 완화된다.

Influence score를 공식적으로 표현하면:

$$S(\mathbf{v}_{\text{train}}, \mathbf{v}_{\text{query}}) = \frac{1}{|\mathcal{T}|} \sum_{(t,\epsilon) \in \mathcal{T}} \frac{\nabla_\theta \mathcal{L}_{\text{diff}}^{\text{motion}}(\theta; \mathbf{v}_{\text{train}}, t, \epsilon)^\top \cdot \nabla_\theta \mathcal{L}_{\text{diff}}^{\text{motion}}(\theta; \mathbf{v}_{\text{query}}, t, \epsilon)}{\|\cdot\|_2 \|\cdot\|_2}$$

#### ④ 프레임 길이 편향 수정 (Frame-Length Bias Correction)

원시 그래디언트 크기는 비디오 $\mathbf{v}$의 프레임 수 $F$에 따라 달라져, **긴 비디오 쪽으로 점수가 편향**된다. 이를 수정하기 위해 프로젝션-정규화 단계 이전에 프레임 수에 대한 정규화를 수행한다:

$$\tilde{g} = \frac{1}{\sqrt{F}} \nabla_\theta \mathcal{L}_{\text{diff}}(\theta; \mathbf{v}, t_{\text{fix}}, \epsilon_{\text{fix}})$$

#### ⑤ 확장성(Scalability)

본 방법은 **공통 무작위성(common randomness)을 사용하는 단일 샘플 변형(single-sample variant)과 투영(projection)**을 통해 확장 가능하게 만들어지며, 각 학습-쿼리 데이터 쌍에 대해 계산되고, 최종 랭킹으로 집계되어 파인튜닝 서브셋을 선택한다.

Influence 계산은 $\mathbb{R}^{D'}$ 공간에서의 내적(inner product)이므로, 모든 학습 예제에 대한 평가 복잡도는 $\mathcal{O}(|\mathcal{D}| \cdot D')$이고, 정렬은 $\mathcal{O}(|\mathcal{D}| \log |\mathcal{D}|)$이다. 모션 특화 오버헤드는 주로 AllTracker 마스크 추출로부터 발생하며 복잡도는 $\mathcal{O}(|\mathcal{D}| \cdot H \cdot W \cdot F)$이지만, 마스크는 **한 번만 추출되어 캐싱**되므로 그래디언트 비용 대비 무시할 수 있는 수준이다.

---

### 🏗️ 2.3 모델 구조

모션 그래디언트 계산의 세 단계:
1. **AllTracker**로 모션 감지
2. **모션 크기 패치(motion-magnitude patches)** 계산
3. 동적 영역에 집중하도록 **손실 공간 모션 마스크(loss-space motion masks)** 적용

Motive는 **광학 흐름(optical flow)**을 사용하여 사람, 이동하는 물체, 카메라 모션과 같은 **동적 영역을 분리**하고 정적 배경을 필터링한다. 이러한 모션 인식 가중치 부여(motion-aware weighting)는 어트리뷰션을 외관이 아닌 **시간적 패턴에 집중**시켜 모션 품질을 향상시킨다.

실험에서는 **Wan2.1-T2V-1.3B**를 Motive가 선택한 고품질 비디오 데이터로 파인튜닝하며, 파인튜닝 동안 **DiT 백본(backbone)만 업데이트**하고 T5 텍스트 인코더와 VAE는 동결(freeze)한다.

어트리뷰션 프레임워크는 **VIDGEN-1M**과 **4DNeX-10M**이라는 두 개의 대규모 비디오 데이터셋에서 평가되며, 두 데이터셋 모두 다양한 모션 패턴, 풍부한 시간적 역학, 복잡한 장면을 제공한다.

---

### 📊 2.4 성능 향상

Motive가 선택한 고영향(high-influence) 데이터로 학습 시, **VBench**에서 모션 부드러움(motion smoothness)과 동적 정도(dynamic degree) 모두 향상되어, **사전학습 베이스 모델 대비 74.1%의 인간 선호도(human preference win rate)**를 달성한다.

실험에서 무작위 선택(random selection)과 Motive 모두 **학습 데이터의 10%만**을 사용하는 조건에서 비교되었으므로, 이는 데이터 선택의 질적 차이를 보여주는 결과이다.

추가로, **Wan2.1-T2V-1.3B 이상의 아키텍처 범용성**을 검증하기 위해 훨씬 더 큰 파라미터 수(5B vs 1.3B)와 새로운 고압축 VAE를 도입한 **Wan2.2-TI2V-5B**에서도 실험을 수행하여 일반화 성능을 검증하였다.

---

### ⚠️ 2.5 한계 (Limitations)

모션 학습은 특정 예시들로 추적 가능하여, 아티팩트(artifacts) 진단과 타겟 데이터 선택을 위한 정량적 도구를 제공하고 더욱 **제어 가능하고 해석 가능한 비디오 Diffusion 모델**을 가능하게 한다. 모델이 확장될수록, 이러한 데이터 수준의 이해가 **강건하고 신뢰할 수 있는 생성 시스템 구축에 필수적**이다.

공개된 자료를 통해 확인된 주요 한계:

- 모션 마스킹 없이는 어트리뷰션이 시간적 역학이 아닌 **정적 외관에 집중**하게 된다. 모션 인식 가중치가 시간적 패턴을 분리하는 핵심이다.
- AllTracker에 의존하므로, 광학 흐름 추정이 어려운 극단적인 조명 변화, 급격한 카메라 전환 등의 영상에서는 모션 마스크 품질이 저하될 수 있다.
- 현재는 주로 **파인튜닝 단계**에서 검증되었으며, 대규모 사전학습(pre-training) 데이터셋에의 전체 적용은 추가적인 확장성 검토가 필요하다.

---

## 3. 모델의 일반화 성능 향상 가능성

일반화 성능 평가를 위해 **10가지 물리적 모션 유형** (bounce, compress, explode, float, free fall, roll, slide, spin, stretch, swing)에 대해 각 유형별 5개의 프롬프트를 사용하여 특정 물리적 모션에 대한 프레임워크의 효과를 평가하였다.

**Specialist 모델**은 단일 모션 카테고리로 선택된 데이터에서 학습되고, **Generalist 모델**은 집계된 선택(aggregated selections)을 사용하여 학습된다.

Motive는 모션 특화 그래디언트를 분리함으로써 생성된 동역학(dynamics)을 영향력 있는 학습 클립으로 추적한다. 이미지 기반 어트리뷰션과 달리, **직접 시간적 역학을 타겟으로** 하여 데이터에서 일관성(coherence)과 물리적 그럴듯함이 어떻게 나타나는지를 밝힌다.

생성된 모션이 모델을 형성한 데이터 분포를 반영한다면, **모션을 영향력 있는 학습 클립에 어트리뷰션하는 것이 모델이 왜 특정 방식으로 움직이는지에 대한 직접적인 렌즈**를 제공하고, 원하는 역학을 위한 타겟 데이터 선택을 가능하게 한다.

**일반화 가능성 정리:**

| 측면 | 내용 |
|---|---|
| **다중 아키텍처** | Wan2.1-T2V-1.3B 및 Wan2.2-TI2V-5B 모두에서 검증 |
| **다중 데이터셋** | VIDGEN-1M, 4DNeX-10M 두 데이터셋에서 평가 |
| **모션 다양성** | 10가지 물리적 모션 유형에 대한 평가 |
| **Generalist 모델** | 집계된 데이터 선택으로 광범위한 모션 학습 가능 |
| **데이터 효율성** | 전체 데이터의 10%만으로 74.1% human preference 달성 |

---

## 4. 향후 연구에 미치는 영향 및 고려할 점

### 🔮 4.1 향후 연구에 미치는 영향

Motive 큐레이션 데이터는 **베이스라인 모델 대비 74.1%의 인간 선호도 향상**이라는 상당한 모션 품질 개선과 계산 효율성을 함께 달성한다. 본 연구는 연구자와 실무자들이 **원칙적인 데이터 큐레이션(principled data curation)**을 통해 더 나은 모델을 구축할 수 있도록 한다.

**구체적인 영향:**

1. **데이터 중심 AI(Data-Centric AI) 연구 촉진**: 모델 아키텍처 개선뿐 아니라 데이터 선택과 큐레이션이 모션 품질에 결정적임을 입증하여, 비디오 생성에서의 데이터 중심 접근법 연구를 가속화한다.

2. **Diffusion 모델 해석 가능성(Interpretability) 연구**: 어떤 학습 클립이 생성 비디오의 모션에 영향을 미치는지라는 **핵심적이고 탐구되지 않은 질문**에 답하며, Motive는 생성된 역학(dynamics)을 영향력 있는 학습 클립으로 추적한다.

3. **세계 모델(World Model) 연구**: 키워드로 제시된 **세계 모델(World model), 비디오 생성, 데이터 어트리뷰션**의 교차점에서, 물리적으로 그럴듯한 세계 모델 구축에 필요한 데이터 이해를 위한 기초를 제공한다.

4. **비디오 Diffusion 데이터 법적/윤리적 이슈**: 공정한 어트리뷰션 방법으로 훈련에 기여한 아티스트와 데이터 제공자의 공헌을 인정하고, 관련 법적·프라이버시 문제를 해결하는 데 기여할 수 있다.

---

### 📝 4.2 향후 연구 시 고려할 점

1. **사전학습(Pre-training) 단계로의 확장**: 현재는 파인튜닝 단계에서 주로 검증되었으므로, 수억-수십억 단위의 사전학습 코퍼스로 Motive를 확장하기 위한 추가적인 연산 최적화가 필요하다.

2. **AllTracker 의존성 탈피**: 현재 AllTracker를 사용하여 광학 흐름 기반 모션 정보를 추출하는 구조이므로, 향후에는 더 일반적이거나 학습 가능한 모션 추정기와의 통합을 고려해야 한다.

3. **어트리뷰션의 인과성(Causality) 검증**: 그래디언트 기반 Influence Score는 상관관계를 반영하지만, 실제 인과적 영향에 대한 추가 검증이 필요하다. Influence Function은 학습 샘플이 제거될 때 유도되는 효과를 추정하지만, Hessian의 역행렬 계산은 계산 비용이 높고 고도로 비볼록한(highly non-convex) 딥 뉴럴 네트워크에서 불안정할 수 있다.

4. **오디오-비주얼 통합 모션 어트리뷰션**: 현재는 시각적 모션에만 초점을 맞추고 있으나, 오디오 신호와 연계된 모션 어트리뷰션으로 확장하면 멀티모달 비디오 생성 연구에 기여할 수 있다.

5. **네거티브 영향 클립 식별(Negative Influence Mining)**: 어떤 파인튜닝 클립이 시간적 역학을 향상시키거나 저하시키는지를 연구하는 방향으로, 저품질 데이터 자동 필터링 파이프라인 구축에 활용할 수 있다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 방법 | 타겟 | 주요 특징 |
|---|---|---|---|
| **Koh & Liang (2017) — Influence Functions** | Hessian 기반 Influence | 이미지/분류 | 데이터 어트리뷰션의 기초; Hessian 행렬 계산 및 역행렬의 높은 계산 비용이 문제 |
| **TracIn (Pruthi et al., 2020)** | 1차 그래디언트 근사 | 분류 | 최적성 조건에 의존하지 않는 1차 그래디언트 근사 기반 Influence 측정 |
| **TRAK (Park et al., 2023)** | NTK 기반 근사 | 비전·언어 | Neural Tangent Kernel 인사이트를 활용하여 소수의 모델 체크포인트로 확장 가능하고 정확한 어트리뷰션 구현 |
| **Data Attribution for Diffusion (2024, arXiv:2401.09031)** | Timestep bias 수정 | 이미지 Diffusion | Diffusion 모델에서 Influence Function의 계산 비용 및 자연스러운 "최종 레이어" 부재 문제 지적 |
| **Data Attribution for T2I (2024, arXiv:2406.09408)** | 언러닝(unlearning) | Text-to-Image | Influence Function으로 학습 데이터 포인트 교란 시 테스트 데이터포인트의 목적 함수 변화를 근사; 가장 큰 변화를 유발하는 학습 포인트로 어트리뷰션 |
| **🌟 Motive (2026, arXiv:2601.08828)** | 모션 가중 그래디언트 | **비디오 모션** | 시각적 외관이 아닌 **모션을 어트리뷰션하는 최초의 프레임워크**; VBench 74.1% 인간 선호도 |

고전적인 Influence Function은 학습 예제를 무한히 업웨이팅(upweighting)할 때 테스트 손실 변화를 측정하는데, 이는 규모에서 비현실적인 **역 Hessian 벡터 곱(inverse-Hessian-vector products)**을 필요로 한다. 실용적인 대안으로 **TracIn 및 TRAK** 같은 그래디언트 유사도 기반 근사가 사용되어 왔다.

---

## 📌 참고 자료

1. **논문 원문**: Xindi Wu et al., "Motion Attribution for Video Generation," arXiv:2601.08828, January 2026. [https://arxiv.org/abs/2601.08828](https://arxiv.org/abs/2601.08828)
2. **NVIDIA 공식 프로젝트 페이지 (ICML 2026 Oral)**: [https://research.nvidia.com/labs/sil/projects/MOTIVE/](https://research.nvidia.com/labs/sil/projects/MOTIVE/)
3. **Hugging Face Papers 페이지**: [https://huggingface.co/papers/2601.08828](https://huggingface.co/papers/2601.08828)
4. **OpenReview (ICML 2026)**: [https://openreview.net/forum?id=fjO8QlDKSt](https://openreview.net/forum?id=fjO8QlDKSt)
5. **ResearchGate**: [https://www.researchgate.net/publication/399754836](https://www.researchgate.net/publication/399754836)
6. **Liner Quick Review**: [https://liner.com/review/motion-attribution-for-video-generation](https://liner.com/review/motion-attribution-for-video-generation)
7. **관련 선행 연구**: Koh & Liang, "Understanding Black-box Predictions via Influence Functions," ICML 2017.
8. **관련 선행 연구**: Pruthi et al., "Estimating Training Data Influence by Tracing Gradient Descent," NeurIPS 2020.
9. **관련 선행 연구**: Park et al., "TRAK: Attributing Model Behavior at Scale," ICML 2023.
10. **관련 선행 연구**: "Data Attribution for Diffusion Models: Timestep-induced Bias in Influence Estimation," arXiv:2401.09031, 2024.
11. **관련 선행 연구**: "Data Attribution for Text-to-Image Models by Unlearning Synthesized Images," arXiv:2406.09408, 2024.

> ⚠️ **정확도 주의 사항**: 본 답변의 수식 일부(특히 Influence Score 통합 수식)는 공개된 arXiv PDF 및 HTML에서 확인 가능한 내용을 기반으로 재구성한 것이며, 논문의 완전한 수식 체계는 원문 PDF를 직접 확인하시기 바랍니다.
