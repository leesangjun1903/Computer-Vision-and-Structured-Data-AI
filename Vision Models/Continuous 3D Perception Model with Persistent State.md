
# CUT3R: Continuous 3D Perception Model with Persistent State

> **논문 정보**
> - **제목:** Continuous 3D Perception Model with Persistent State
> - **저자:** Qianqian Wang\*, Yifei Zhang\*, Aleksander Holynski, Alexei A. Efros, Angjoo Kanazawa
> - **발표:** CVPR 2025 Oral
> - **arXiv:** [arXiv:2501.12387](https://arxiv.org/abs/2501.12387)
> - **프로젝트 페이지:** https://cut3r.github.io/
> - **GitHub:** https://github.com/CUT3R/CUT3R

---

## 1. 핵심 주장 및 주요 기여 요약

이 논문은 다양한 3D 태스크를 통합적으로 해결할 수 있는 프레임워크를 제시하며, 상태(State)를 유지하는 순환(Recurrent) 모델을 사용해 새로운 관측이 입력될 때마다 내부 표현을 지속적으로 업데이트한다. 주어진 이미지 스트림으로부터 각 입력에 대해 메트릭 스케일 포인트맵(픽셀당 3D 포인트)을 온라인 방식으로 생성한다.

이 모델은 **CUT3R (Continuous Updating Transformer for 3D Reconstruction)**로 명명되며, 실세계 장면의 풍부한 사전 지식(prior)을 내재화하여, 이미지 관측으로부터 정확한 포인트맵을 예측할 뿐만 아니라 가상의 미관측 뷰를 쿼리하여 장면의 미관측 영역을 추론할 수도 있다.

이 방법은 단순하면서도 높은 유연성을 가지며, 비디오 스트림이나 순서 없는 사진 모음을 모두 자연스럽게 처리하고, 정적 및 동적 콘텐츠를 모두 포함할 수 있다. 단안/비디오 깊이 추정, 카메라 추정, 다중 뷰 재구성 등 다양한 3D/4D 태스크에서 경쟁적이거나 SOTA 수준의 성능을 달성하였다.

해당 논문은 **CVPR 2025 Oral**로 채택되었다.

### ✅ 주요 기여 요약

| 기여 | 설명 |
|------|------|
| 통합 프레임워크 | 다양한 3D/4D 태스크를 단일 모델로 해결 |
| Persistent State | 새 입력마다 지속적으로 업데이트되는 내부 상태 |
| 온라인 재구성 | 비디오 스트림을 실시간으로 처리 |
| 미관측 뷰 추론 | 가상 카메라를 쿼리하여 미관측 영역 예측 |
| 동적 장면 처리 | 정적·동적 장면 모두 통합 처리 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능, 한계

### 2-1. 해결하고자 하는 문제

이 논문은 지속적으로 업데이트되는 영구 상태를 가진 **온라인 3D 인식 모델**을 제안한다. 이미지 스트림이 주어지면, 각 관측에 대해 상태-업데이트(State Update)와 상태-읽기(State Readout)를 동시에 수행한다.

데이터 기반 사전 지식(data-driven prior)은 전통적 방법이 겪는 어려움(예: 동적 객체, 희박한 관측, 퇴화된 카메라 움직임)을 해결하는 반면, 지속적 업데이트 능력은 새로운 관측을 온라인으로 처리하고 재구성을 점진적으로 개선하게 한다.

기존의 유사 접근들(3D-R2N2, Spann3R 등)은 객체 중심적이거나 포즈가 입력으로 요구되는 한계가 있었다. 동시에 개발된 Spann3R도 공간 메모리를 이용한 연속적 재구성을 보이지만, Spann3R의 메모리는 관측된 장면의 캐시 역할에 그치는 반면, CUT3R의 압축된 상태 표현은 관측된 장면뿐 아니라 미관측 구조의 추론까지 가능하게 한다.

---

### 2-2. 제안 방법 및 핵심 수식

#### 📐 Step 1: 이미지 인코딩

입력 이미지 $I_t$가 주어지면, Vision Transformer(ViT) 인코더를 통해 토큰 표현으로 인코딩된다:

$$F_t = \text{Encoder}_i(I_t)$$

상태 또한 토큰의 집합으로 표현되며, 어떤 이미지도 보기 전에는 모든 장면이 공유하는 학습 가능한(learnable) 토큰들로 초기화된다.

#### 📐 Step 2: 상태 업데이트 및 읽기 (State Update & Readout)

각 입력 이미지는 공유 가중치의 ViT 인코더를 통해 시각 토큰으로 인코딩된다. 이 토큰들은 상태 토큰과 상호작용하여, 상태 업데이트는 현재 이미지를 상태에 통합하고 상태 읽기는 예측을 위해 상태에 저장된 과거 컨텍스트를 불러온다. 두 과정은 두 개의 상호 연결된 ViT 디코더를 통해 동시에 발생한다.

이 상호작용의 핵심 수식은 다음과 같다:

$$[z'_t,\; F'_t],\; s_t = \text{Decoders}([z,\; F_t],\; s_{t-1})$$

여기서 $z$는 **자기 운동(ego-motion) 정보를 담는 특수 "포즈 토큰(pose token)"** 으로, $F_t$ 앞에 붙여진다. $s_{t-1}$은 이전 시간의 상태 표현이다.

#### 📐 Step 3: 3D 기하 출력

시스템은 각 관측 뷰에 대해 두 종류의 포인트맵과 신뢰도 점수를 예측한다:

- **자기 프레임(self frame)에서의 포인트맵 및 신뢰도:**

$$\hat{X}_{\text{self},t},\; C_{\text{self},t} = \text{Head}_{\text{self}}(F'_t)$$

- **월드 좌표계(world frame)에서의 포인트맵 및 신뢰도:**

$$\hat{X}_{\text{world},t},\; C_{\text{world},t} = \text{Head}_{\text{world}}(F'_t,\; z'_t)$$

- **카메라 포즈 예측:**

$$\hat{P}_t = \text{Head}_{\text{pose}}(z'_t)$$

#### 📐 학습 손실 함수

MASt3R와 CUT3R를 따라 포즈에는 $L_2$ 노름 손실을, 포인트맵에는 신뢰도 기반 손실을 사용한다. 예측 포즈 $\hat{T}_t$는 쿼터니언 $\hat{q}_t$와 이동 벡터 $\hat{\tau}_t$로 파라미터화된다.

$$\mathcal{L} = \mathcal{L}_{\text{pointmap}} + \lambda \mathcal{L}_{\text{pose}}$$

포인트맵 손실의 신뢰도 가중 형태:

$$\mathcal{L}_{\text{pointmap}} = \sum_{i} C_i \cdot \| \hat{X}_i - X_i \|_2 - \alpha \log C_i$$

---

### 2-3. 모델 구조

모델의 핵심은 새로운 관측이 입력될 때마다 지속적으로 업데이트되는 상태 표현이다. 현재 이미지가 주어지면, ViT 인코더로 토큰 표현으로 인코딩된다. 이미지 토큰은 상태-업데이트와 상태-읽기의 통합 프로세스를 통해 상태 토큰과 상호작용한다. 상태 읽기의 최종 출력은 현재 관측에 대한 포인트맵과 카메라 파라미터이다. 동시에, 상태-업데이트 모듈은 현재 관측을 반영하여 상태를 업데이트한다. 이 과정을 모든 이미지에 반복하여, 출력이 누적되어 밀도 높은 장면 재구성이 형성된다.

또한 입력 이미지로 관측되지 않은 새로운 구조를 추론할 수도 있다. 파란 카메라로 표시된 쿼리 카메라가 주어지면, 해당 레이맵(raymap)으로 현재 상태를 쿼리하여 해당 포인트맵을 읽어낸다.

```
입력 이미지 스트림
       ↓
 ViT Encoder (공유 가중치)
       ↓
  이미지 토큰 F_t
       ↓
 ┌─────────────────────────┐
 │  Interconnected Decoder  │
 │  State Update ↕ Readout │  ← s_{t-1} (이전 상태)
 └─────────────────────────┘
       ↓             ↓
   s_t (새 상태)    F'_t, z'_t
                     ↓
          Head_self / Head_world / Head_pose
                     ↓
         포인트맵 + 카메라 파라미터 출력
```

---

### 2-4. 성능 향상

CUT3R는 단안/비디오 깊이 추정, 카메라 추정, 다중 뷰 재구성 등 다양한 3D/4D 태스크에서 경쟁적이거나 최신 기술(state-of-the-art) 수준의 성능을 달성하였다.

예를 들어, 보다 광범위한 지도(supervision)로 훈련된 CUT3R는 여러 시나리오에서 MASt3R를 능가한다. 강한 일반화를 위해서는 다양하고 고품질의 데이터가 중요하다.

온라인 등록 방법인 Spann3R 및 CUT3R는 더 빠른 추론과 낮은 GPU 메모리 사용량을 보인다.

**모노큘러 깊이 추정 비교 (Abs Rel ↓, NYU-v2 기준):**

| 방법 | NYU-v2 (Abs Rel↓) | Bonn (Abs Rel↓) |
|------|-------------------|-----------------|
| DUSt3R-GA | 0.080 | 0.141 |
| MASt3R-GA | 0.129 | 0.142 |
| MonST3R-GA | 0.102 | 0.076 |
| Spann3R | 0.122 | 0.118 |
| **CUT3R** | **0.086** | **0.063** |

*(출처: Point3R 논문 내 Table 2 비교)*

---

### 2-5. 한계점

CUT3R는 스트리밍 시퀀스에 대한 온라인 재구성을 위해 지속적인 상태 토큰과 Transformer 기반 순환 업데이트를 사용하지만, 결정론적 추론으로 인한 극단적 뷰포인트 외삽(extrapolation) 능력이 제한되며, 긴 시퀀스에서 글로벌 정렬이 부재하여 누적 드리프트(drift accumulation)가 발생할 수 있다.

CUT3R는 고정 길이의 학습 가능한 토큰 특징을 메모리 모듈로 사용하여 순차적 처리 중 지속적으로 업데이트되지만, 제한된 용량으로 인해 정보 손실이 발생할 수 있다.

CUT3R는 RNN 스타일 아키텍처를 채택하여 비정형 입력을 점진적으로 처리하지만, 메모리 용량이 제한적이고 Flash-Attention과 같은 현대적 하드웨어 가속 기법과의 호환성이 낮다.

또한 CUT3R는 넓은 깊이 범위를 가진 드론 데이터셋이나 공간적 사전 지식을 수립하기 어려운 극단적인 희박-뷰(sparse-view) 시나리오에서 성능이 저하된다.

더불어, CUT3R는 주로 64프레임 시퀀스로 훈련되어 긴 시퀀스에 대한 일반화에 실패하는 문제가 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 리비지팅(Revisiting)을 통한 컨텍스트 확장

온라인 방식으로 실행할 때, 상태는 현재 시간까지의 컨텍스트만 접근 가능하고 미래 관측 정보는 없다. 추가 컨텍스트로 성능을 더 높일 수 있는지 탐구하기 위해, 논문은 **"revisiting"**이라는 새 실험 설정을 도입하였다: 모든 이미지를 먼저 한 번 실행하여 상태가 전체 컨텍스트를 볼 수 있게 한 뒤, 이 최종 상태를 고정(freeze)하고 동일한 이미지 세트를 다시 처리하여 예측을 생성한다. 이 연산은 최종 상태에 포착된 3D 이해도를 평가한다.

예를 들어, 첫 번째와 두 번째 이미지가 겹치지 않는(큰 시점 변화) 경우, 모델은 처음에 최적이 아닌 예측을 생성하여 TV와 커피 테이블이 겹쳐 보인다. 그러나 최종 상태(즉, 전체 컨텍스트 포함)로 장면을 revisit하면, TV와 커피 테이블 및 소파에 대한 예측이 더 정확해진다.

### 3-2. 동적 장면 일반화

다른 기존 방법들과 달리 정적 장면에만 국한되지 않고, CUT3R는 동적 장면도 원활하게 재구성할 수 있다.

### 3-3. 다양한 데이터를 통한 일반화

강한 일반화를 위해서는 다양하고 고품질의 데이터가 중요하며, 충분히 표현되지 않는 도메인에서 강건성을 높이기 위해 더 넓은 분포를 포함하는 데이터로 훈련해야 한다. 목표 지향적이고 다양한 데이터를 통한 훈련이 DUSt3R 계열의 도메인 일반화를 향상시킬 수 있음이 확인되었다.

### 3-4. 조인트 학습을 통한 일반화

다수의 기하 속성(포즈, 깊이, 매칭 등)의 조인트 예측이 최근 성능 향상의 기반이 될 수 있다. CUT3R, Fast3R, Geo4D와 같은 최신 GFM(Geometric Foundation Model)은 포인트맵과 카메라 포즈를 조인트로 예측한다. 이러한 조인트 학습 방식은 공간 기하와 뷰 간 관계를 모두 포착하는 더 풍부하고 구조화된 표현을 학습하게 하여, 일반화와 강건성이 향상된다.

---

## 4. 앞으로의 연구에 미치는 영향 및 고려할 점

### 4-1. 파생 연구 및 영향

CUT3R는 발표 이후 다양한 후속 연구에 직접적인 영향을 미쳤다:

| 후속 논문 | 핵심 아이디어 |
|-----------|--------------|
| **G-CUT3R** (ICLR 2026) | CUT3R에 카메라 및 깊이 사전 정보를 통합한 확장 |
| **TTT3R** | CUT3R의 긴 시퀀스 일반화를 테스트-타임 훈련으로 해결 |
| **Point3R** | CUT3R의 고정 메모리 한계를 3D 공간 포인터 메모리로 극복 |
| **STream3R** | CUT3R의 RNN 구조 대신 Decoder-only Transformer 사용 |
| **LONG3R** | 시공간 컨텍스트 활용으로 누적 드리프트 감소 |
| **VLM-3R** | CUT3R를 공간 인코더로 활용, VLM과 3D 인식 통합 |

G-CUT3R는 CUT3R 프레임워크에 대한 경량화 및 모달리티-불가지론적(modality-agnostic) 확장으로, 스트림라인된 인코딩 프로세스와 신중하게 설계된 융합 기법을 통해 기하 사전 정보를 통합한다.

TTT3R는 최근 3D 재구성 기반 모델에 대한 테스트-타임 훈련 관점을 제공하며, CUT3R에 대한 단순하고 효율적인 수정을 제안하여 긴 시퀀스 일반화를 향상시킨다.

### 4-2. 향후 연구 시 고려할 점

#### ① 긴 시퀀스 일반화 문제

모델이 짧은 컨텍스트로 훈련되면, 순환(recurrence)이 상태를 훈련 중에 만나지 않은 분포 밖(OOD) 영역으로 몰아내기 때문에 더 긴 시퀀스로의 일반화에 실패하는 **"미탐색 상태(unexplored states) 가설"** 이 있다.

최근 연구는 더 효과적이고 안정적이며 병렬화 가능한 순환 아키텍처를 개발할 거대한 기회를 강조하며, 향후 연구가 3D 재구성 모델의 기반을 재검토하고 재구성 정확도와 시퀀스 일반화를 더욱 향상시킬 것을 기대한다.

#### ② 고정 용량 메모리의 한계 극복

CUT3R는 고정 길이의 학습 가능한 토큰 특징을 메모리 모듈로 사용하여 순차적 처리 중 지속적으로 업데이트되지만, 그 제한된 용량이 정보 손실로 이어질 수 있다.

따라서 향후 연구에서는 **동적으로 확장 가능한 메모리 구조**(예: 공간 포인터 메모리, KV-Cache 기반 어텐션)를 도입해야 한다.

#### ③ 하드웨어 최적화 및 실시간성

CUT3R는 RNN 스타일 아키텍처로 인해 Flash-Attention 등 현대 하드웨어 가속 기법과의 호환성이 낮은 반면, Decoder-only Transformer 방식은 KV-Cache와 윈도우 어텐션 같은 기법을 통해 효율적인 인과적(causal) 추론을 가능하게 한다.

#### ④ 다양한 도메인 데이터 확보

메트릭 스케일 깊이 추정에도 한계가 있다. 메트릭 정확도가 있는 훈련 데이터의 부족으로 인해 GFM은 절대 깊이를 예측하는 데 어려움을 겪는 경우가 많다.

#### ⑤ 루프 클로저(Loop Closure) 부재

루프 클로저 감지(loop closure detection)와 사후 최적화(post-optimization)의 부재로 인한 누적 오차를 줄이기 위해, 훈련 및 추론 단계에서 시공간 맥락 정보를 활용하는 것이 중요하다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

최근 연구는 명시적 최적화 루프를 제거하고 단일 네트워크 패스로 장면 기하를 예측하는 방향으로 발전하였다. DUSt3R는 이 흐름의 선구자로, 두 개의 캘리브레이션되지 않은 이미지로부터 밀도 높은 포인트맵을 생성한다.

| 모델 | 연도 | 방식 | 스트리밍 | 동적 장면 | 주요 특징 |
|------|------|------|----------|-----------|-----------|
| **DUSt3R** | 2024 | 쌍방향 피드포워드 | ❌ | ❌ | 포인트맵 직접 예측 |
| **MASt3R** | 2024 | 쌍방향 피드포워드 | ❌ | ❌ | 매칭 헤드 추가 |
| **MonST3R** | 2024 | 쌍방향 피드포워드 | ❌ | ✅ | 동적 장면 파인튜닝 |
| **Spann3R** | 2024 | 메모리 기반 | ✅ | ❌ | 공간 메모리 캐시 |
| **CUT3R** | 2025 | RNN 기반 | ✅ | ✅ | 지속 상태 업데이트 |
| **Fast3R** | 2025 | 전역 어텐션 | ⚠️ | ❌ | 1000+ 이미지 단일 패스 |
| **VGGT** | 2025 | 전역 어텐션 | ⚠️ | ❌ | 매칭+깊이+포즈 통합 |
| **Point3R** | 2025 | 공간 포인터 메모리 | ✅ | ✅ | 확장 가능 3D 메모리 |
| **TTT3R** | 2025 | RNN+TTT | ✅ | ✅ | CUT3R의 긴 시퀀스 개선 |

Spann3R와 CUT3R는 더 긴 시퀀스를 위한 메모리와 순환(recurrence)을 도입한 반면, Fast3R와 VGGT는 전역 좌표에서 출력을 지원하는 확장 가능한 다중 뷰 추론을 지원한다.

CUT3R, Fast3R, Geo4D와 같은 최신 GFM은 포인트맵과 카메라 포즈를 조인트로 예측하며, 이러한 조인트 학습 방식은 공간 기하와 뷰 간 관계를 모두 포착하는 더 풍부하고 구조화된 표현을 학습하게 하여 일반화와 강건성을 향상시킨다.

---

## 📚 참고 자료 및 출처

1. **[주논문]** Wang, Q., Zhang, Y., Holynski, A., Efros, A.A., Kanazawa, A. (2025). *Continuous 3D Perception Model with Persistent State*. CVPR 2025. [arXiv:2501.12387](https://arxiv.org/abs/2501.12387)
2. **[프로젝트 페이지]** CUT3R Project Page: https://cut3r.github.io/
3. **[공식 구현]** CUT3R GitHub: https://github.com/CUT3R/CUT3R
4. **[CVPR 2025 Open Access]** https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Continuous_3D_Perception_Model_with_Persistent_State_CVPR_2025_paper.html
5. **[문헌 리뷰]** The Moonlight: [Literature Review – Continuous 3D Perception Model with Persistent State](https://www.themoonlight.io/en/review/continuous-3d-perception-model-with-persistent-state)
6. **[후속 연구: TTT3R]** arXiv:2509.26645 — TTT3R: 3D Reconstruction as Test-Time Training
7. **[후속 연구: G-CUT3R]** arXiv:2508.11379 — G-CUT3R: Guided 3D Reconstruction with Camera and Depth Prior Integration (ICLR 2026)
8. **[후속 연구: Point3R]** arXiv:2507.02863 — Point3R: Streaming 3D Reconstruction with Explicit Spatial Memory
9. **[후속 연구: STream3R]** arXiv:2508.10893 — STream3R: Scalable Sequential 3D Reconstruction with Causal Transformer
10. **[후속 연구: LONG3R]** arXiv:2507.18255 — LONG3R: Long Sequence Streaming 3D Reconstruction
11. **[후속 연구: VLM-3R]** arXiv:2505.20279 — VLM-3R: Vision-Language Models Augmented with 3D Spatial Features
12. **[벤치마크]** arXiv:2506.01933 — E3D-Bench: A Benchmark for End-to-End 3D Geometric Foundation Models
13. **[관련 비교]** arXiv:2511.22429 — Fin3R: Fine-tuning Feed-forward 3D Reconstruction Models via Monocular Knowledge Distillation
14. **[Awesome-DUSt3R GitHub]** https://github.com/Ethan-Lee-Sunghoon/Awesome-DUST3R
