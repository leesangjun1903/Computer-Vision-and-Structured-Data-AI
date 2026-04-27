
# SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory

> **논문 정보**
> - **제목**: SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory
> - **저자**: Cheng-Yen Yang, Hsiang-Wei Huang, Wenhao Chai, Zhongyu Jiang, Jenq-Neng Hwang (University of Washington)
> - **arXiv**: [2411.11922](https://arxiv.org/abs/2411.11922) (v1: 2024.11.18, v2: 2024.11.30)
> - **프로젝트 페이지**: https://yangchris11.github.io/samurai/
> - **GitHub**: https://github.com/yangchris11/samurai

---

## 1. 핵심 주장과 주요 기여 요약

### 1.1 핵심 주장 (Core Claims)

SAM 2(Segment Anything Model 2)는 객체 분할(segmentation) 태스크에서 강력한 성능을 보이지만, 혼잡한 장면, 빠르게 움직이거나 자기 폐색(self-occlusion)이 일어나는 객체를 추적(tracking)할 때 어려움을 겪는다. 또한 원래 모델의 고정 윈도우(fixed-window) 메모리 방식은 다음 프레임의 이미지 특성을 조건화하기 위해 선택한 메모리의 품질을 고려하지 않아, 영상에서 오류가 전파(error propagation)되는 문제가 발생한다.

이에 대해 SAMURAI는 다음의 핵심 주장을 내세웁니다:

시간적 모션 단서(temporal motion cues)를 제안한 모션 인식 메모리 선택 메커니즘(motion-aware memory selection mechanism)과 결합함으로써, SAMURAI는 재학습(retraining)이나 파인튜닝(fine-tuning) 없이 객체 모션을 효과적으로 예측하고 마스크 선택을 정제(refine)하여 견고하고 정확한 추적을 달성한다. SAMURAI는 실시간(real-time)으로 작동하며, 다양한 벤치마크 데이터셋에서 파인튜닝 없이 강력한 제로샷(zero-shot) 성능을 보여 일반화 능력을 입증한다.

### 1.2 주요 기여 (Main Contributions)

| 기여 | 설명 |
|------|------|
| **① Motion Modeling** | 칼만 필터(Kalman Filter) 기반 모션 모델링으로 마스크 선택 정제 |
| **② Motion-Aware Memory Selection** | 어피니티 점수 + 모션 점수를 결합한 하이브리드 메모리 선택 |
| **③ Zero-Shot 추적** | 추가 학습 없이 SAM 2.1 가중치로 직접 VOT 수행 |
| **④ 실시간 온라인 추적** | 실시간 온라인 추론 가능 |

---

## 2. 상세 분석

### 2.1 해결하고자 하는 문제 (Problem Statement)

SAM 2의 대표적인 두 가지 실패 케이스가 존재한다: (1) 유사한 외형을 가진 배경 객체들이 존재하는 혼잡한 장면에서, SAM 2는 모션 단서를 무시하고 더 높은 IoU 점수를 가진 마스크를 예측하는 경향이 있다. (2) 기존 메모리 뱅크는 단순히 이전 $n$개 프레임을 선택·저장하여, 폐색(occlusion) 중에 잘못된 특징이 도입되는 결과를 낳는다.

구체적으로 두 가지 핵심 문제가 있습니다:

1. **마스크 선택의 모호성**: 시각적으로 유사한 객체들 사이에서 올바른 객체를 식별하지 못함
2. **메모리 품질 미고려**: 고정 윈도우 방식으로 인해 잘못된 프레임이 메모리에 저장되어 오류 전파 발생

---

### 2.2 제안 방법 (Proposed Methods with Formulas)

#### 2.2.1 칼만 필터 기반 모션 모델링 (Motion Modeling with Kalman Filter)

SAMURAI는 상기 문제들을 해결하기 위해, 멀티 마스크 선택 위에 칼만 필터(KF) 기반 모션 모델링을 적용하고, 어피니티 점수와 모션 점수를 결합한 하이브리드 스코어링 시스템에 기반한 향상된 메모리 선택 방식을 제안한다. 이 향상된 기법들은 복잡한 비디오 시나리오에서 객체를 정확하게 추적하는 모델의 능력을 강화하도록 설계되었다. 중요하게도, 이 방법은 파인튜닝이나 추가 학습이 필요하지 않으며, 기존 SAM 2 모델에 직접 통합될 수 있다.

SAMURAI는 모션 모델링을 위해 선형 칼만 필터를 사용한다. 이 공식화는 재학습 또는 파인튜닝이 필요하지 않으며, 칼만 필터의 예측-수정 사이클(prediction-correction cycle)로 작동하는데, 이는 객체의 위치, 크기 및 속도를 포함하는 내부 상태 벡터 $\mathbf{x}$를 업데이트한다. 구체적으로, 상태 벡터는 다음과 같이 표현된다:

$$\mathbf{x} = \begin{bmatrix} x \\ y \\ w \\ h \\ \dot{x} \\ \dot{y} \\ \dot{w} \\ \dot{h} \end{bmatrix}$$

여기서 $x$, $y$는 바운딩 박스의 중심 좌표이며, $w$, $h$는 너비와 높이를 나타낸다.

칼만 필터의 예측 단계는 다음과 같이 표현됩니다:

$$\hat{\mathbf{x}}_{t+1|t} = \mathbf{F} \mathbf{x}_t$$

$$\hat{\mathbf{P}}_{t+1|t} = \mathbf{F} \mathbf{P}_t \mathbf{F}^\top + \mathbf{Q}$$

수정(correction) 단계:

$$\mathbf{K}_t = \hat{\mathbf{P}}_{t|t-1} \mathbf{H}^\top (\mathbf{H} \hat{\mathbf{P}}_{t|t-1} \mathbf{H}^\top + \mathbf{R})^{-1}$$

$$\mathbf{x}_t = \hat{\mathbf{x}}_{t|t-1} + \mathbf{K}_t (\mathbf{z}_t - \mathbf{H} \hat{\mathbf{x}}_{t|t-1})$$

여기서 $\mathbf{F}$는 상태 전이 행렬(state transition matrix), $\mathbf{H}$는 관측 행렬(observation matrix), $\mathbf{z}_t$는 선택된 마스크로부터 도출된 바운딩 박스 측정값입니다.

#### 2.2.2 마스크 선택 (Mask Selection via KF-IoU Score)

모델은 예측된 상태와 마스크 사이의 IoU(Intersection over Union)를 계산하여 칼만 필터 IoU 점수 $s_{kf}$를 도출한다:

$$s_{kf} = \text{IoU}(\hat{\mathbf{x}}_{t+1|t}, M)$$

이 점수는 마스크 선택 시 원래의 마스크 어피니티 점수와 결합된다:

$$M^{*} = \arg\max_{M_i} \left( \alpha_{kf} \cdot s_{kf}(M_i) + (1 - \alpha_{kf}) \cdot s_{mask}(M_i) \right)$$

여기서 $\alpha_{kf}$는 KF-IoU 점수와 마스크 어피니티 점수 사이의 균형을 조절하는 가중치 하이퍼파라미터입니다.

#### 2.2.3 모션 인식 메모리 선택 (Motion-Aware Memory Selection)

SAMURAI는 SAM 2의 고정 윈도우 메모리 전략을 마스크 어피니티(mask affinity), 객체 존재 여부(object presence), 모션 점수(motion score)를 포함한 여러 스코어링 메커니즘을 결합하는 정교한 시스템으로 대체한다. 이 접근 방식은 중요한 프레임이 메모리에 유지되도록 하여, 추가 학습이나 파인튜닝 없이 일관성과 신뢰성을 향상시킨다.

메모리 선택에 사용되는 세 가지 점수를 수식으로 표현하면:

$$s_{mem} = \beta_1 \cdot s_{affinity} + \beta_2 \cdot s_{occurrence} + \beta_3 \cdot s_{motion}$$

최근 프레임을 유지하는 기존 방법 대신, SAMURAI는 다음 세 가지를 평가한다: **마스크 어피니티 점수**(예측 마스크와 객체 외형의 호환성), **객체 발생 점수**(추적 세그먼트에서 객체가 나타나는 빈도), **모션 점수**(과거 프레임을 기반으로 한 예측된 움직임의 신뢰도).

또한, 목표 객체가 재등장하거나 일정 기간 마스크 품질이 낮은 상황에서의 모션 모델링 견고성을 보장하기 위해, SAMURAI는 안정적인 모션 상태를 유지한다. 즉, 추적 객체가 과거 일정 기간 동안 성공적으로 업데이트된 경우에만 모션 모듈을 고려한다.

---

### 2.3 모델 구조 (Model Architecture)

SAMURAI는 입력 비디오 프레임을 이미지 인코더(image encoder)를 통해 처리하여 시각적 특징을 추출하고, 메모리 어텐션(memory attention)을 통해 이전 프레임들을 동적으로 참조한다. SAMURAI는 모션 점수를 기반으로 메모리 프레임을 평가하여 관련 메모리를 유지하고 무관한 메모리를 버리는 모션 인식 메모리 선택 메커니즘을 활용한다. 마스크 디코더(mask decoder)는 이러한 선택된 특징들을 결합하여 추적을 위한 예측 마스크를 생성한다. 또한 어피니티 헤드(affinity head)를 통해 마스크 품질을 평가하고, 객체 헤드(object head)를 통해 객체 존재 여부를 검증함으로써 복잡한 시나리오에서 견고한 추적 성능을 보장한다.

아키텍처를 도식화하면 다음과 같습니다:

```
입력 프레임 (Video Frame)
        │
        ▼
  ┌─────────────┐
  │ Image Encoder│  ← SAM 2 ViT 백본 (T/S/B/L)
  └──────┬──────┘
         │ Visual Features
         ▼
  ┌─────────────────┐
  │ Memory Attention │  ← Motion-Aware Memory Bank
  │  (KF + Hybrid   │     (Affinity + Occurrence
  │   Score Select) │      + Motion Score)
  └──────┬──────────┘
         │ Conditioned Features
         ▼
  ┌──────────────┐
  │ Mask Decoder  │
  │ (+ KF-IoU    │  → M* = argmax(α·s_kf + (1-α)·s_mask)
  │  Mask Select)│
  └──────┬───────┘
         │
         ▼
  최종 추적 마스크 & 바운딩 박스
```

실제 사무라이와 달리, 제안된 SAMURAI는 추가 학습이 필요하지 않다. 이는 제로샷 방법으로, SAM 2.1의 가중치를 직접 사용하여 VOT 실험을 수행한다. 칼만 필터는 시간이 지남에 따른 측정에 기반하여 움직이는 객체의 현재 및 미래 상태(바운딩 박스 위치와 크기)를 추정하는 데 사용되며, 이는 어떤 학습도 필요하지 않다.

---

### 2.4 성능 향상 (Performance Improvements)

평가에서 SAMURAI는 기존 트래커 대비 성공률과 정밀도에서 큰 개선을 달성했다: **LaSOT ${ext}$에서 7.1% AUC 향상**, **GOT-10k에서 3.5% AO 향상**. 또한 LaSOT에서 완전 지도(fully supervised) 방법들과 경쟁력 있는 결과를 달성하여, 복잡한 추적 시나리오에서의 견고성과 동적 환경에서의 실세계 응용 가능성을 강조한다.

SAMURAI-B는 SAM2.1-B 대비 AO에서 2.1%, OP0.5에서 2.9% 개선을 보였으며, SAMURAI-L은 AO에서 0.6%, OP0.5에서 0.7% 개선을 보였다. 모든 SAMURAI 모델은 GOT-10k의 모든 메트릭에서 최신 기술 수준(SOTA)을 능가하였다.

제로샷 SAMURAI-L 모델은 AUC에서 최신 지도 학습(supervised) 방법과 비슷하거나 능가하는 수준을 보여, 다양한 데이터셋에서의 모델 역량과 일반화 능력을 입증한다.

**벤치마크별 성능 요약 (SAMURAI vs. SAM 2 기준):**

| 데이터셋 | 지표 | 향상 폭 |
|----------|------|---------|
| **LaSOT** | AUC | 74.23% (SOTA 수준) |
| **LaSOT ${ext}$** | AUC | +7.1% |
| **GOT-10k** | AO | +3.5% |
| **GOT-10k (B)** | OP0.5 | +2.9% |

어블레이션 스터디에서, 제안된 각 모듈(모션 모델링 및 모션 인식 메모리 선택)의 기여도는 성능 지표에 긍정적인 영향을 미치는 것으로 나타났으며, 두 모듈의 시너지 관계를 강조했다.

---

### 2.5 한계 (Limitations)

칼만 필터는 강렬하거나 급격한 모션이 있는 시나리오에서 실패하는 경우가 있다. 표준 및 노이즈 스케일 적응(NSA) 칼만 필터 모두 많은 하이퍼파라미터를 수반하여, 특정 유형의 모션 시나리오에 대한 효과성을 제한할 수 있다.

더 극단적인 폐색이나 도전적인 시나리오(예: 극심한 시점 변화, 극단적인 조명 조건)를 처리하는 기술 탐구가 매우 가치 있을 것이다. 또한 SAMURAI를 다중 객체 추적(multi-object tracking)으로 확장하는 것은 중요한 발전을 의미하지만, 객체 연관(object association)과 ID 전환(ID switching)의 복잡성을 해결해야 한다.

현재 코드는 라이브/스트리밍 비디오를 지원하지 않는데, 이는 SAM 2의 코드베이스를 대부분 상속했기 때문이다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재의 일반화 성능

SAMURAI는 SAM2 자체에서 적응된 제로샷 시각 추적 모델이다. 제로샷 시각 추적은 특정 객체 클래스에 대한 사전 학습 없이 비디오 스트림에서 객체를 추적할 수 있게 하는 컴퓨터 비전의 고급 기술이다. 이 접근 방식은 제로샷 학습 원리를 활용하여, 훈련 중에 특정 객체를 본 적이 없더라도 시각적 특성과 맥락 정보를 기반으로 객체를 식별하고 추적할 수 있게 한다.

모션 모델링과 메모리 선택을 SAM 2에 통합하면 재학습 없이 시각 객체 추적이 향상된다. 이 접근 방식은 **모델 불가지론적(model-agnostic)**이므로, SAM 2 이외의 다른 추적 프레임워크에도 적용될 수 있다.

### 3.2 일반화를 강화하는 설계 요소

모션 인식 메모리 선택 메커니즘은 동등하게 중요하다. 이는 모션과 어피니티 점수를 모두 고려하는 하이브리드 스코어링 시스템을 기반으로 관련 과거 프레임들을 지능적으로 우선순위화한다. 덜 관련 있는, 잠재적으로 혼동을 주는 프레임들을 버림으로써, 트래커는 오류 전파를 방지하고 특히 혼잡한 장면에서 견고성을 크게 향상시킨다. 이 동적 메모리 관리는 SAM의 고정 윈도우 메모리 접근 방식의 한계를 해결하는 핵심 차별점이며, 정확도와 효율성을 향상시킨다.

### 3.3 향후 일반화 확장 가능성

미래 연구는 SAMURAI를 향상시키기 위한 여러 유망한 방향을 탐구할 수 있다. 단순한 칼만 필터 이상의 모션 모델을 개선하는 것, 예를 들어 딥러닝 기반 접근 방식을 사용하면 비선형 객체 움직임이 있는 복잡한 시나리오에서 더 견고한 추적이 가능할 것이다. 더 적응적인 메모리 선택 메커니즘을 개발하는 것이 핵심이다. 현재 하이브리드 스코어링 시스템은 잘 작동하지만, 객체 외형 변화나 객체 간 상호작용과 같은 추가적인 요소를 통합함으로써 이점을 얻을 수 있다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 기반 모델 계보

| 연구 | 연도 | 핵심 기여 | SAMURAI와의 관계 |
|------|------|-----------|----------------|
| **SAM (Segment Anything)** | 2023 | 프롬프트 기반 범용 분할 | SAMURAI의 조부 모델 |
| **SAM 2** | 2024 | 비디오 세그멘테이션 + 메모리 메커니즘 | SAMURAI의 직접 기반 모델 |
| **SAMURAI** | 2024 | 모션 인식 메모리 + KF 기반 마스크 선택 | **본 논문** |

SAM은 소개된 이후 상당한 후속 연구를 촉발했다. SAM은 사용자가 포인트, 바운딩 박스, 텍스트를 입력하여 이미지 내 어떤 객체든 분할하도록 안내하는 프롬프트 기반 분할 접근 방식을 도입했다. SAM 2는 모델의 기능을 비디오 분할로 확장하여, 동적 비디오 시퀀스에서 여러 프레임에 걸쳐 객체를 추적하기 위한 메모리 메커니즘을 통합했다.

### 4.2 동시 및 후속 연구들

최근 **SAM2Long**은 트리 기반 메모리를 사용하여 긴 비디오에서의 객체 분할을 향상시켰다.

**SAM2.1++(A Distractor-Aware Memory for Visual Object Tracking with SAM2)** 연구는 동시에 SAM2.1을 기반으로 하며, 방해 요소(distractor) 처리와 메모리 관리를 개선하기 위해 마스크 정제 과정에 모션 단서를 통합한다는 점에서 밀접하게 관련된다. 결과는 SAM2.1++가 DiDi 데이터셋에서 추적 품질 면에서 SAMURAI를 2% 능가함을 보여준다. 이는 주로 높은 견고성(robustness) 때문이며, SAM2.1++의 새로운 DAM 메모리와 관리 프로토콜의 우수성을 보여준다.

**HiM2SAM**은 두 가지 핵심 혁신을 통해 장기적이고 복잡한 추적 문제를 해결하는 향상된 SAM2 프레임워크를 소개한다: (1) 계층적 모션 추정(Hierarchical Motion Estimation) - 가벼운 칼만 기반 선형 모션 예측과 포인트 트래커를 통해 선택된 프레임에만 픽셀 수준 예측을 결합한다. (2) 최적화된 메모리 구조(Optimized Memory Structure) - 메모리 뱅크를 단기와 장기 구성 요소로 분할하고, 고신뢰도 및 독특한 프레임을 선택적으로 저장하는 모션 인식 필터링을 통합한다. HiM2SAM은 학습이 불필요하며, 최소한의 추가 비용을 발생시키고, 특히 LaSOT 시리즈 데이터셋에서 VOT 벤치마크 전반에 걸쳐 성능 향상을 달성한다.

### 4.3 연구 흐름 비교표

| 모델 | 기반 | 모션 모델링 | 메모리 전략 | 학습 필요 여부 | 주요 특징 |
|------|------|-----------|------------|-------------|----------|
| **SAM 2** | ViT | ✗ | Fixed-window | ✓ (대규모) | 범용 비디오 분할 |
| **SAM2Long** | SAM 2 | ✗ | Tree-based | ✗ | 장기 비디오 분할 |
| **SAMURAI** | SAM 2.1 | ✓ (KF) | Hybrid Score | ✗ | Zero-shot VOT |
| **SAM2.1++** | SAM 2.1 | ✓ | DAM Memory | ✗ | 방해요소 인식 추적 |
| **HiM2SAM** | SAM 2 | ✓ (KF + Point Tracker) | 단기/장기 분리 | ✗ | 장기 복잡 추적 |

---

## 5. 앞으로의 연구에 미치는 영향과 고려사항

### 5.1 앞으로의 연구에 미치는 영향

**① 제로샷 패러다임의 확산**

요약하면, SAMURAI는 세그먼트-애니씽 모델에 기반하여 혼잡한 장면에서의 자기 폐색 및 급격한 모션을 해결하기 위해 모션 기반 스코어링과 메모리 선택을 도입했다. 모션 인식 메모리와 향상된 마스크 선택의 결합은 지도 모델과 제로샷 모델 사이의 격차를 줄이는 실시간 추적에 대한 새로운 접근 방식을 보여준다. 제안된 모듈들은 재학습이나 파인튜닝 없이 여러 시각 객체 추적 벤치마크에서 지속적으로 성능을 향상시켰다.

**② 모델 불가지론적 방법론의 활용**

이 접근 방식은 모델 불가지론적(model-agnostic)이므로, SAM 2 이외의 다른 추적 프레임워크에도 적용될 수 있다. 효과적인 모션 모델링과 지능적인 메모리 선택을 결합함으로써, SAMURAI는 다양하고 복잡한 환경에서 추적 성능을 크게 향상시키는 능력을 보여준다.

**③ 의료 영상, 스포츠 분석 등 도메인 확장**

스포츠 분석에서 감시(surveillance)까지, SAMURAI의 견고한 성능은 정밀하고 신뢰할 수 있는 추적을 요구하는 모든 시나리오에 이상적이다.

### 5.2 향후 연구 시 고려할 사항

**① 더 고급화된 모션 모델 탐구**

단순한 칼만 필터 이상의 모션 모델을 개선하는 것 — 딥러닝 기반 접근 방식을 사용하는 것처럼 — 은 비선형 객체 움직임이 있는 복잡한 시나리오에서 더 견고한 추적을 가능하게 할 것이다.

**② 다중 객체 추적(MOT)으로의 확장**

SAMURAI를 다중 객체 추적으로 확장하는 것은 중요한 발전을 의미하지만, 객체 연관(object association)과 ID 전환(ID switching)의 복잡성을 해결해야 한다.

**③ 하이브리드 스코어링 시스템 개선**

더 적응적인 메모리 선택 메커니즘을 개발하는 것이 핵심이다. 현재의 하이브리드 스코어링 시스템은 잘 작동하지만, 객체 외형 변화나 객체 간 상호작용과 같은 추가적인 요소를 통합함으로써 이점을 얻을 수 있다.

**④ 극단적 시나리오 대응**

마스크 디코더를 위한 다양한 프롬프트 전략을 탐구하면 정확도와 효율성이 향상될 수 있다. 또한, 더 극단적인 폐색이나 도전적인 시나리오(예: 심각한 시점 변화, 극단적인 조명 조건)를 처리하는 기술을 탐구하는 것이 매우 가치 있을 것이다.

**⑤ 스트리밍/실시간 확장**

현재 코드는 라이브/스트리밍 비디오를 지원하지 않으므로, 실시간 스트리밍 입력(예: 웹캠) 지원을 구현하는 연구가 필요하다.

---

## 6. 결론

SAMURAI는 SAM 2에 대한 단순한 업그레이드 이상이며, 시각 추적 기술의 도약이다. 모션 인식 메모리(motion-aware memory)와 실시간 추적 기능을 도입함으로써, SAMURAI는 분할(segmentation)과 실세계 추적 도전 사이의 간극을 좁힌다. 특히 **재학습 없이(zero-shot)** 지도 학습 방법들과 경쟁력 있는 성능을 달성하고, 다양한 도메인과 추적 프레임워크에 적용 가능한 **모델 불가지론적** 설계를 보여준다는 점에서 향후 비디오 이해(video understanding) 연구 전반에 중요한 이정표가 될 것입니다.

---

## 📚 참고 자료 (References)

| # | 출처 | URL |
|---|------|-----|
| 1 | **arXiv 논문 페이지** (v1/v2) | https://arxiv.org/abs/2411.11922 |
| 2 | **공식 프로젝트 페이지** | https://yangchris11.github.io/samurai/ |
| 3 | **공식 GitHub 저장소** | https://github.com/yangchris11/samurai |
| 4 | **Hugging Face Papers** | https://huggingface.co/papers/2411.11922 |
| 5 | **arXiv HTML 풀텍스트** (v1) | https://arxiv.org/html/2411.11922v1 |
| 6 | **AZoAI 뉴스 분석** | https://www.azoai.com/news/20241201/SAMURAI-Enhances-Object-Tracking-with-Motion-Aware-Intelligence.aspx |
| 7 | **Emergent Mind 리뷰** | https://www.emergentmind.com/papers/2411.11922 |
| 8 | **Moonlight Literature Review** | https://www.themoonlight.io/en/review/samurai-adapting-segment-anything-model-for-zero-shot-visual-tracking-with-motion-aware-memory |
| 9 | **Medium (Data Science in Your Pocket)** | https://medium.com/data-science-in-your-pocket/samurai-enhanced-sam-2-for-visual-object-tracking-0cf9a649f517 |
| 10 | **AI Paper Reviews** | https://deep-diver.github.io/ai-paper-reviewer/paper-reviews/2411.11922/ |
| 11 | **A Distractor-Aware Memory for VOT with SAM2** (비교 논문) | https://arxiv.org/html/2411.17576v2 |
| 12 | **HiM2SAM** (비교 논문, Springer) | https://link.springer.com/chapter/10.1007/978-981-95-5755-4_19 |
| 13 | **NASA ADS Abstract** | https://ui.adsabs.harvard.edu/abs/2024arXiv241111922Y/abstract |

> ⚠️ **주의사항**: 본 논문은 arXiv에 게재된 프리프린트(preprint)로, arXiv는 동료 심사(peer-review)를 거치지 않은 예비 과학 보고서를 게시하므로, AI 연구 분야의 확정적인 정보로 취급하거나 개발 결정을 안내하는 데 사용하지 않도록 주의해야 한다.
