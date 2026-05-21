
# ALTER: All-in-One Layer Pruning and Temporal Expert Routing for Efficient Diffusion Generation

> **논문 정보**
> - 제목: ALTER: All-in-One Layer Pruning and Temporal Expert Routing for Efficient Diffusion Generation
> - arXiv ID: 2505.21817 (2025년 5월 27일 제출)
> - 발표: NeurIPS 2025 (포스터)
> - 저자: Xiaomeng Yang 외 7인

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

Diffusion 모델은 고품질 이미지 생성에 뛰어난 성능을 보이지만, 반복적인 denoising 과정으로 인해 추론 시 막대한 계산 비용이 발생하며, 이는 자원 제한 환경에서의 실용적 배포를 어렵게 한다.

기존 가속화 방법들은 diffusion 생성 과정에서의 시간적 변화(temporal variation)를 포착하지 못하는 균일한(uniform) 전략을 채택하며, 일반적으로 사용되는 순차적 pruning-then-fine-tuning 전략은 사전학습된 가중치에서 이루어진 pruning 결정과 최종 모델 파라미터 간의 불일치로 인해 준최적(sub-optimal) 결과를 낳는다.

이를 해결하기 위해 ALTER는 diffusion 모델을 효율적인 시간적 전문가(temporal expert)들의 혼합으로 변환하는 통합 프레임워크를 제안하며, 훈련 가능한 하이퍼네트워크(hypernetwork)를 통해 레이어 pruning 결정을 동적으로 생성하고 전문화된 pruned 전문가 서브네트워크로의 timestep 라우팅을 관리하는 단일 단계 최적화를 달성한다.

### 1.2 주요 기여 (3가지)

| 기여 | 내용 |
|---|---|
| **단일 단계 통합 최적화** | pruning, expert routing, fine-tuning을 하나의 스테이지에서 공동 최적화 |
| **하이퍼네트워크 기반 동적 pruning** | UNet 가중치 업데이트에 따라 최적 pruning 결정을 지속적으로 생성 |
| **Temporal Expert Routing** | 각 denoising timestep을 특화된 pruned 서브네트워크로 동적 라우팅 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

Diffusion 모델은 반복적인 전체 네트워크 denoising으로 인한 지연 시간과 메모리 사용 문제를 안고 있으며, 이를 완화하기 위한 두 가지 주요 전략이 등장했다: (1) 시간적 차원에서 denoising 단계 수를 줄이는 방식과 (2) 단계별 계산 부담을 줄이기 위한 모델 수준 압축이다.

레이어 pruning은 기존 모델 실행 파이프라인의 최소한의 수정으로 전체 레이어를 제거하여 속도 향상을 달성하는 가장 실용적인 방식이지만, 이러한 거친 pruning은 모든 입력 프롬프트 및 denoising timestep에 걸쳐 정적으로 적용될 때 성능 저하를 야기한다.

유연성 향상을 위해 동적 pruning과 거친 구조적 제거를 결합하는 접근이 자연스러운 방향이며, 기존 연구들은 각 입력에 고유한 서브네트워크를 선택하는 샘플 단위 동적 전략을 탐구했다. 그러나 이는 파라미터 활용을 심각하게 제한한다.

---

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 전체 프레임워크 개요

ALTER는 레이어 pruning과 temporal expert routing, 모델 fine-tuning을 하나의 원스테이지 최적화 접근으로 통합하며, 표준 diffusion 모델을 효율적인 시간적 전문가들의 혼합으로 변환한다. 여기서 각 전문가는 생성 과정의 서로 다른 단계에 특화된 원본 모델의 pruned 서브네트워크이다. 이 동적 구성은 서로 다른 전문가 서브네트워크의 최적 부분 구조를 식별하고 전체 fine-tuning 단계에 걸쳐 denoising timestep을 지능적으로 라우팅함으로써 달성된다.

#### 2.2.2 Diffusion 모델의 기본 목적 함수

Diffusion 모델의 기본 훈련 목적 함수는 다음과 같다:

$$\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{x_0, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

여기서:
- $x_0$: 원본 데이터 샘플
- $\epsilon \sim \mathcal{N}(0, I)$: 가우시안 노이즈
- $t$: denoising timestep
- $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$: 노이즈가 추가된 샘플
- $\epsilon_\theta$: 노이즈 예측 네트워크 (UNet)

#### 2.2.3 하이퍼네트워크 기반 Layer Pruning

하이퍼네트워크는 업데이트된 모델 가중치를 바탕으로 지속적으로 레이어 pruning 결정을 생성하는 동시에, 적절한 전문가에게 timestep을 라우팅하는 역할을 수행한다.

하이퍼네트워크 $\mathcal{H}$는 현재 UNet 가중치 $\theta$를 입력으로 받아 각 레이어 $l$에 대한 이진 마스크(binary mask) $m_l \in \{0, 1\}$를 생성한다:

$$\mathbf{m} = \mathcal{H}(\theta) = [m_1, m_2, \ldots, m_L]$$

전문가 $k$에 대한 pruned 서브네트워크는 다음과 같이 정의된다:

$$\epsilon_{\theta}^{(k)}(x_t, t) = \epsilon_\theta(x_t, t; \mathbf{m}^{(k)})$$

#### 2.2.4 Temporal Expert Routing

timestep $t$를 $K$개의 전문가 구간(interval)으로 분할하는 라우팅 함수 $\mathcal{R}$:

$$k^* = \mathcal{R}(t) = \arg\min_{k} |t - c_k|$$

여기서 $c_k$는 $k$번째 전문가의 timestep 중심값이다.

실제 추론 시 전체 denoising 과정:

$$x_{t-1} = f\left(\epsilon_{\theta}^{(\mathcal{R}(t))}(x_t, t), t\right)$$

#### 2.2.5 통합 최적화 목적 함수

ALTER의 단일 단계 공동 최적화:

$$\min_{\theta, \phi} \sum_{k=1}^{K} \mathbb{E}_{t \in \mathcal{T}_k, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t; \mathbf{m}^{(k)}(\phi))\|^2\right] + \lambda \cdot \mathcal{R}_{\text{sparsity}}(\mathbf{m})$$

여기서:
- $\phi$: 하이퍼네트워크 파라미터
- $\mathcal{T}_k$: $k$번째 전문가에 할당된 timestep 집합
- $\mathcal{R}_{\text{sparsity}}$: sparsity 제약 항 (목표 pruning 비율 달성을 위한 정규화)
- $\lambda$: sparsity 제약 가중치

---

### 2.3 모델 구조

ALTER는 표준 diffusion UNet을 동적인 temporal expert 앙상블로 재구성하며, 각 전문가는 공유된 백본(backbone)에 적용된 서로 다른 레이어 단위 pruning 구성으로 정의된다. 즉, 각 전문가는 생성 과정의 서로 다른 단계에 특화된 원본 모델의 pruned 서브네트워크이다.

```
[입력: x_t, t]
       │
  Hypernetwork φ
  (가중치 θ 기반)
       │
  ┌────┴────┐
  │ Pruning │ → 각 전문가별 마스크 m^(k) 생성
  │ Decisions│
  └────┬────┘
       │
  Timestep Router R(t)
  ────────────────────
  t ∈ T_1 → Expert 1 (pruned UNet, m^(1))
  t ∈ T_2 → Expert 2 (pruned UNet, m^(2))
  ...
  t ∈ T_K → Expert K (pruned UNet, m^(K))
       │
  [출력: x_{t-1}]
```

pruning 시스템은 중요도 점수(importance score)를 기반으로 중복된 신경망 레이어를 식별하고 제거하며, expert routing 시스템은 diffusion 과정을 구간(interval)으로 나누고 각 단계마다 특화된 네트워크 구성을 적용한다.

시스템은 서로 다른 timestep에 걸친 레이어 중요도를 평가하는 새로운 점수 메커니즘을 활용하며, 이를 통해 생성 과정 전반에 걸쳐 계산 자원을 동적으로 할당한다. 라우팅 메커니즘은 노이즈에서 이미지로 이어지는 diffusion 과정의 자연적인 진행을 활용한다.

---

### 2.4 성능 향상

ALTER는 단 20번의 추론 단계와 35% sparsity를 통해 원본 50-step Stable Diffusion v2.1 모델의 총 MACs(Multiply-Accumulate Operations)의 25.9%만을 사용하면서 동일 수준의 시각적 품질을 달성하고 3.64배의 속도 향상을 제공한다.

ALTER는 동적이고 timestep 의존적인 레이어 단위 pruning 전략을 학습하고 timestep을 특화된 temporal expert로 라우팅하는 새로운 통합 프레임워크를 제안하며, 이는 하이퍼네트워크와 UNet의 공동 최적화를 통해 단일 훈련 단계에서 달성되어 높은 생성 품질을 유지하면서 상당한 계산 절감과 속도 향상을 이끌어낸다.

**성능 요약표:**

| 지표 | 결과 |
|---|---|
| 추론 속도 향상 | **3.64×** |
| MACs 사용 비율 | 전체의 **25.9%** |
| 추론 단계 수 | 50 → **20 steps** |
| 비교 기준 | Stable Diffusion v2.1 (50-step) |

---

### 2.5 한계

한계로 지적된 것은, 하이퍼네트워크 훈련 종료 후($T_{end}$) 동적으로 pruned된 UNet에 대한 최종 fine-tuning 단계가 성능 복원을 위해 여전히 유익하다는 점이다. 이는 완전한 단일 단계 최적화라는 주장과 다소 상충될 수 있는 부분이다.

또한 연구가 계산 효율성에 주로 집중되어 있으며, 속도 향상이 인상적임에도 특정 유형의 이미지나 생성 시나리오에 대한 미탐색된 영향이 존재할 수 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화 성능에 기여하는 핵심 메커니즘

ALTER의 설계에는 일반화 성능을 향상시킬 수 있는 여러 메커니즘이 내재되어 있다.

#### (1) 하이퍼네트워크의 동적 적응성
ALTER는 서로 다른 전문가 서브네트워크의 최적 부분 구조를 식별하고 전체 fine-tuning 단계에 걸쳐 denoising timestep을 지능적으로 라우팅하는 방식으로 동적 구성을 달성하며, 하이퍼네트워크는 업데이트된 모델 가중치를 기반으로 지속적으로 레이어 pruning 결정을 생성한다.

이는 훈련 중 가중치 변화에 따라 pruning 결정이 자동으로 조정되므로, **고정된 마스크에 의한 과적합(overfitting)을 방지**하는 데 기여한다.

#### (2) Temporal Expert의 전문화를 통한 표현 다양성
ALTER에서 각 전문가는 특정 timestep 범위를 위해 특화되며, timestep embedding 라우팅을 기반으로 동적으로 선택된다.

각 전문가가 denoising 과정의 서로 다른 단계(초기 노이즈 제거: 전반적 구조 파악 / 후기 단계: 세부 디테일 생성)에 특화됨으로써, 단일 네트워크보다 각 단계에서 더 전문화된 표현 학습이 가능하다. 이는 다양한 생성 시나리오에서의 **도메인 간 일반화**를 지원한다.

#### (3) 단일 단계 공동 최적화의 정렬 효과 (Alignment Effect)
기존의 순차적 pruning-then-fine-tuning은 사전학습된 가중치에서 이루어진 pruning 결정과 최종 모델 파라미터 간의 불일치로 인해 준최적 결과를 낳는다.

ALTER의 단일 단계 공동 최적화는 pruning과 fine-tuning이 동시에 진행되므로 **분포 이동(distribution shift) 문제를 완화**하고, 다양한 프롬프트 분포에 대해 더 강건한 모델을 만들 수 있다.

#### (4) 기존 모델과의 호환성
Stable Diffusion과 같은 인기 있는 모델과 호환되며, 재훈련 없이도 적용 가능하다.

이러한 플러그인(plug-in) 방식의 적용 가능성은 다양한 사전학습 모델에 범용적으로 적용 가능함을 시사하며, 이는 일반화 성능의 핵심 지표이다.

### 3.2 일반화 향상 관련 수식적 관점

Pruning과 일반화 성능의 관계는 이론적으로도 뒷받침된다:

훈련 손실을 0으로 수렴시킬 수 있는 경우, 네트워크는 좋은 일반화 성능을 보이며, 더욱 주목할 점은 pruning 비율이 클수록 일반화 경계(generalization bound)가 더 좋아진다는 것이다.

이를 수식으로 표현하면, Rademacher complexity 기반의 일반화 경계:

$$\mathcal{E}_{\text{gen}} \leq \hat{\mathcal{E}}_{\text{train}} + \mathcal{O}\left(\sqrt{\frac{d_{\text{eff}} \log n}{n}}\right)$$

여기서 $d_{\text{eff}}$는 유효 파라미터 수이며, pruning으로 인해 감소하므로, **적절한 pruning은 과적합을 줄이고 일반화를 향상**시킬 수 있다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

UNet 기반 텍스트-이미지 diffusion 모델 압축에 초점을 맞춘 주요 연구들로는 SnapFusion, MobileDiffusion, BK-SDM, LAPTOP-Diff, LD-Pruner 등이 있으며, 모두 UNet 아키텍처의 덜 중요한 구성 요소를 대상으로 한다.

| 연구 | 연도 | 방법론 | 주요 특징 | 한계 |
|---|---|---|---|---|
| **Diff-Pruning** (Fang et al.) | NeurIPS 2023 | 구조적 pruning (gradient 기반) | 50% FLOPs 감소 가능 | 소규모 DDPM에 한정, 재훈련 비용 높음 |
| **BK-SDM** (Kim et al.) | ECCV 2024 | UNet 블록 제거 (수동 설계) | 경량·빠른 Stable Diffusion | 수동 handcrafting, 일반화 부족 |
| **LAPTOP-Diff** (Zhang et al.) | 2024 | 레이어 pruning + 정규화 증류 | SDXL/SDM-v1.5 적용, 50% pruning에서 PickScore 4% 하락 | 순차적 pruning-then-distillation |
| **DeepCache** (Ma et al.) | CVPR 2024 | 캐싱 기반 가속 (훈련 불필요) | 훈련 없이 적용 가능 | 캐시 재사용으로 인한 품질 손실 |
| **DiP-GO** | NeurIPS 2024 | Few-step gradient 기반 pruning | 4.4× 속도 향상, 재훈련 불필요 | SD-2.1에 집중, 범용성 제한 |
| **TinyFusion** (Fang et al.) | 2025 | DiT 아키텍처 depth pruning | DiT 특화 | UNet 비적용 |
| **ALTER** (Yang et al.) | NeurIPS 2025 | 단일 단계 통합 최적화 + Temporal MoE | 3.64× 속도, 25.9% MACs, timestep 적응 | 최종 fine-tuning 단계 여전히 필요 |

현재 방법들은 종종 일반성이 부족하여, UNet과 같은 특정 아키텍처에 맞춰진 경향이 있으며 다양한 구조를 가진 대규모 텍스트-이미지 diffusion 모델(예: Multimodal Diffusion Transformer)에 쉽게 적용하기 어렵다.

이전의 효율적 아키텍처 설계 방식들은 SDM의 U-Net에서 중요하지 않은 레이어를 식별하고 제거하는 실험적 연구를 거치는 수작업 방식을 택했으며, 이러한 수작업 방식은 최적 성능을 달성하기 어렵고 확장성과 일반화 능력이 부족하다.

---

## 5. 향후 연구에 미치는 영향과 고려할 점

### 5.1 향후 연구에 미치는 영향

ALTER는 실제 응용을 위한 diffusion 모델의 실용화에 있어 중요한 진보를 나타내며, 레이어 pruning과 expert routing의 결합은 미래 최적화 연구의 유망한 방향을 제시한다. 이는 더 많은 사용자와 응용에 접근 가능한 더 빠르고 효율적인 AI 이미지 생성 도구로 이어질 수 있다.

구체적으로:

1. **MoE(Mixture of Experts) + Structural Pruning 융합 패러다임 확산**: ALTER가 제시한 "시간적 MoE + 구조적 pruning" 결합은 언어 모델, 비디오 생성 모델 등 다른 반복적 추론 모델에도 적용 가능한 설계 원칙을 제공한다.

2. **하이퍼네트워크 기반 동적 압축 연구 촉진**: 훈련 가능한 하이퍼네트워크가 레이어 pruning 결정을 동적으로 생성하고 timestep 라우팅을 관리하는 방식은 기존의 정적 압축 패러다임을 넘어서는 새로운 연구 방향을 제시한다.

3. **DiT(Diffusion Transformer) 아키텍처로의 확장 가능성**: TinyFusion이 DiT 아키텍처에 depth pruning을 도입한 것처럼, ALTER의 temporal routing 아이디어는 DiT 기반 모델(예: FLUX, SD3)에도 적용 가능한 연구 과제를 남긴다.

### 5.2 앞으로 연구 시 고려할 점

#### ① 아키텍처 범용성 확장
diffusion 모델의 빠른 발전은 기존 pruning 기법의 심각한 한계를 드러내며, 현재 방법들은 UNet과 같은 특정 아키텍처에 맞춰져 있어 다양한 구조의 대규모 텍스트-이미지 diffusion 모델에 쉽게 적용하기 어렵다. 따라서 ALTER의 프레임워크를 Transformer 기반 아키텍처(DiT)나 비디오 생성 모델(예: Sora 계열)로 확장하는 연구가 필요하다.

#### ② 최종 Fine-tuning 단계 제거를 위한 연구
하이퍼네트워크 훈련 종료 후 동적으로 pruned된 UNet에 대한 최종 fine-tuning 단계가 성능 복원에 여전히 유익하다는 한계를 극복하기 위해, 진정한 단일 단계 최적화(true one-shot optimization)를 달성하기 위한 연구가 필요하다. 예를 들어, 지식 증류(knowledge distillation)와의 통합을 통해 추가적인 fine-tuning 없이 성능을 복원하는 방향이 고려될 수 있다.

#### ③ 개인화(Personalization) 및 도메인 특화 시나리오 검증
연구가 계산 효율성에 주로 집중되어 있으며, 특정 유형의 이미지나 생성 시나리오에 대한 미탐색된 영향이 존재할 수 있다. LoRA, DreamBooth 등의 개인화 기법과 결합된 시나리오에서의 성능 검증이 필요하다.

#### ④ Expert 수($K$)와 Sparsity 비율의 최적화 이론 정립
전문가 수 $K$와 각 전문가의 sparsity 비율 $s_k$ 간의 관계에 대한 이론적 분석이 부재하다. 다음과 같은 최적화 문제가 열린 연구 주제로 남아있다:

$$\arg\min_{K, \{s_k\}_{k=1}^K} \mathcal{L}_{\text{gen}} \quad \text{s.t.} \quad \text{FLOPs} \leq \text{Budget}$$

#### ⑤ 비디오/멀티모달 생성으로의 확장
DiT 백본을 활용한 대규모 diffusion 기반 아키텍처의 비디오 생성 및 실시간 게임 렌더링에의 적용이 확대됨에 따라, ALTER의 temporal routing 개념은 프레임(frame) 단위 시간적 변화를 가진 비디오 생성 모델의 압축에 자연스럽게 확장 가능하다.

---

## 참고 자료

1. **[주 논문]** Yang, Xiaomeng et al. *"ALTER: All-in-One Layer Pruning and Temporal Expert Routing for Efficient Diffusion Generation"*, arXiv:2505.21817, NeurIPS 2025. https://arxiv.org/abs/2505.21817

2. **[관련 연구]** Zhang, Dingkun et al. *"LAPTOP-Diff: Layer Pruning and Normalized Distillation for Compressing Diffusion Models"*, arXiv:2404.11098, 2024. https://arxiv.org/abs/2404.11098

3. **[관련 연구]** Kim, Bo-Kyeong et al. *"BK-SDM: A Lightweight, Fast, and Cheap Version of Stable Diffusion"*, ECCV 2024. https://dl.acm.org/doi/10.1007/978-3-031-72949-2_22

4. **[관련 연구]** Fang, Gongfan, Ma, Xinyin, Wang, Xinchao. *"Structural Pruning for Diffusion Models (Diff-Pruning)"*, NeurIPS 2023. https://github.com/VainF/Diff-Pruning

5. **[관련 연구]** *"DiP-GO: A Diffusion Pruner via Few-step Gradient Optimization"*, NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/file/a845fdc3f87751710218718adb634fe7-Paper-Conference.pdf

6. **[관련 연구]** *"OBS-Diff: Accurate Pruning For Diffusion Models in One-Shot"*, arXiv:2510.06751. https://arxiv.org/pdf/2510.06751

7. **[일반화 이론]** *"Pruning Before Training May Improve Generalization, Provably"*, arXiv:2301.00335. https://arxiv.org/pdf/2301.00335

8. **[리뷰 사이트]** Moonlight AI Literature Review: https://www.themoonlight.io/en/review/alter-all-in-one-layer-pruning-and-temporal-expert-routing-for-efficient-diffusion-generation

9. **[NeurIPS 포스터]** https://neurips.cc/virtual/2025/loc/san-diego/poster/120357

---

> ⚠️ **정확도 관련 주의사항**: 본 논문(arXiv:2505.21817)은 2025년 5월 27일 제출된 최신 논문으로, 하이퍼네트워크의 세부 손실 함수 구성 요소, 레이어 중요도 점수 계산 방법, 전문가 수($K$) 설정 등의 구체적 수식은 전문(full paper) PDF의 세부 내용을 직접 확인하지 못한 부분이 있어, 프레임워크의 구조적 논리에 기반하여 표준적인 형태로 제시하였습니다. 정확한 수식은 arXiv 원문([https://arxiv.org/abs/2505.21817](https://arxiv.org/abs/2505.21817))을 직접 참조하시기를 강력히 권장합니다.
