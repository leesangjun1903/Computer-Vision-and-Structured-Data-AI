
# Diffusion Adversarial Post-Training for One-Step Video Generation (Seaweed-APT)

> **논문 정보**
> - **저자**: Shanchuan Lin, Xin Xia, Yuxi Ren, Ceyuan Yang, Xuefeng Xiao, Lu Jiang (ByteDance Seed)
> - **arXiv**: [2501.08316](https://arxiv.org/abs/2501.08316) (2025년 1월)
> - **학회**: ICML 2025 (Proceedings of the 42nd International Conference on Machine Learning, pp. 37959–37974)
> - **프로젝트 페이지**: https://seaweed-apt.com/

---

## 1. 핵심 주장 및 주요 기여 요약

확산 모델(Diffusion Models)은 이미지·비디오 생성에 널리 사용되지만, 반복적인 생성 프로세스가 느리고 비용이 많이 든다. 기존 증류(distillation) 방법들은 이미지 도메인에서 1-step 생성 가능성을 보였으나 심각한 품질 저하를 겪는다.

이 논문의 핵심 주장과 기여를 요약하면 다음과 같습니다:

| 항목 | 내용 |
|---|---|
| **핵심 주장** | Adversarial Post-Training(APT)으로 느린 확산 모델을 실시간 1-step 비디오 생성기로 변환 가능 |
| **핵심 기여 1** | 실제 데이터 기반 GAN 적대적 후처리 훈련 (teacher 불필요) |
| **핵심 기여 2** | 대규모 DiT(~16B) 기반 안정적 적대적 학습 기법 |
| **핵심 기여 3** | 근사 R1 정규화 목적 함수 도입 |
| **주요 모델명** | **Seaweed-APT** |

Seaweed-APT는 단일 순전파 평가 단계(single forward evaluation step)로 실시간 2초, 1280×720, 24fps 비디오를 생성하며, 1024px 이미지를 1-step으로 생성해 최신 기법에 필적하는 품질을 달성한다.

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

확산 모델의 반복적 생성 프로세스는 느리고 비용이 많이 들며, 기존 증류 방법들은 이미지 도메인에서 1-step 생성 가능성을 보였으나 심각한 품질 저하 문제를 여전히 겪는다.

구체적으로, 기존 방법들이 가진 세 가지 핵심 문제를 해결하고자 한다:

1. **다중 추론 단계 문제**: 수십~수백 번의 순전파 필요
2. **증류 기반 품질 저하**: teacher 기반 증류에서 발생하는 blurry 출력
3. **비디오 도메인 미적용**: 기존 1-step 방법이 주로 이미지에 국한

결정론적 방법들은 단순 회귀 손실로 학습이 용이하지만, 소수 단계 생성 결과가 최적화 부정확성과 학생 모델의 Lipschitz 상수 감소로 인해 매우 흐릿(blurry)해지는 문제가 있다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### **핵심 아이디어: Adversarial Post-Training (APT)**

APT는 새로운 1-step 이미지·비디오 생성 방법으로, 사전 훈련된 확산 모델, 특히 DiT(Diffusion Transformer)를 초기화로 활용하고, 실제 데이터에 대한 적대적 훈련 목적 함수로 DiT를 계속 훈련한다. 기존 확산 증류 방법들이 사전 훈련된 모델을 증류 teacher로 사용해 타겟을 생성하는 것과 대조적으로, APT는 사전 훈련 모델을 초기화에만 사용하고, 실제 데이터에 직접 DiT의 적대적 훈련을 수행한다.

#### **훈련 목적 함수: GAN Loss**

표준 GAN 목적 함수를 기반으로, Generator $G_\theta$와 Discriminator $D_\phi$에 대해:

**Generator Loss:**

$$\mathcal{L}_G = \mathbb{E}_{z \sim \mathcal{N}(0,I)} \left[ -\log D_\phi(G_\theta(z)) \right]$$

**Discriminator Loss:**

$$\mathcal{L}_D = \mathbb{E}_{x \sim p_{\text{real}}} \left[ -\log D_\phi(x) \right] + \mathbb{E}_{z \sim \mathcal{N}(0,I)} \left[ -\log(1 - D_\phi(G_\theta(z))) \right]$$

#### **근사 R1 정규화 (Approximated R1 Regularization)**

저자들은 근사 R1 정규화를 도입했는데, 이는 대규모 모델에서 고차 미분 계산 시 발생하는 불안정성을 줄이기 위한 것이다.

표준 R1 정규화는 실제 데이터 $x_{\text{real}}$에 대한 Discriminator 기울기 패널티로 정의된다:

$$\mathcal{R}_1 = \frac{\gamma}{2} \mathbb{E}_{x \sim p_{\text{real}}} \left[ \| \nabla_x D_\phi(x) \|^2 \right]$$

그러나 수십억 파라미터 규모의 모델에서는 $\nabla_x D_\phi(x)$ 계산이 메모리·연산 측면에서 매우 비싸기 때문에, 논문은 이를 근사화하는 **approximated R1 regularization**을 도입한다:

$$\hat{\mathcal{R}}_1 \approx \frac{\gamma}{2} \mathbb{E}_{x \sim p_{\text{real}}} \left[ \| D_\phi(x + \epsilon) - D_\phi(x) \|^2 \right], \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)$$

이는 유한 차분(finite difference)으로 gradient norm을 근사하여 계산 비용을 크게 낮추면서 훈련 안정성을 유지한다.

#### **결정론적 증류를 통한 Generator 초기화**

Generator는 결정론적 증류(deterministic distillation)를 통해 초기화되며, Discriminator에는 트랜스포머 기반 아키텍처 변경, 타임스텝에 대한 앙상블(ensemble across timesteps), 그리고 대규모 훈련을 위한 근사 R1 정규화 손실 등 여러 개선 사항이 도입된다.

---

### 2-3. 모델 구조

#### **전체 파이프라인**

```
[사전훈련된 DiT (Diffusion Transformer)]
        ↓ 결정론적 증류 초기화
[Generator G_θ (DiT 기반, ~8B params)]
        ↓ 적대적 후훈련 (APT)
[Discriminator D_φ (DiT 기반)]
        ↕
[실제 데이터 (Real Data)]
```

#### **Generator**

방법은 사전 훈련된 확산 모델, 특히 DiT(Diffusion Transformer)를 초기화로 활용하고, 실제 데이터에 대한 적대적 훈련 목적 함수로 DiT를 계속 훈련한다.

- 입력: 가우시안 노이즈 $z \sim \mathcal{N}(0, I)$ + 텍스트 조건
- 구조: DiT (Diffusion Transformer) 기반
- 출력: 단일 순전파로 생성된 비디오 latent

#### **Discriminator**

Discriminator는 다층 특징 추출(multi-layer feature extraction)과 서로 다른 타임스텝 앙상블을 입력으로 사용하는 개선 사항으로 재조정된다. 이러한 변경은 훈련을 안정화하고 구조적 무결성 및 세부 사항 포착을 향상시키는 것을 목표로 한다.

#### **모델 규모**

APT를 통해 현재까지 보고된 가장 큰 GAN 중 하나(~16B 파라미터)를 훈련했으며, 단일 순전파 평가로 이미지와 비디오 모두 생성 가능하다.

---

### 2-4. 성능 향상

#### **비디오 생성**

Seaweed-APT는 단일 순전파 평가 단계만으로 실시간 2초, 1280×720, 24fps 비디오를 생성할 수 있다.

실험에 따르면, 8B 모델은 단일 H100에서 736×416 해상도의 실시간 24fps 스트리밍 비디오 생성, 또는 8×H100에서 1280×720 해상도로 최대 1분(1440 프레임) 분량의 비디오를 생성한다.

#### **이미지 생성**

또한, 단일 단계에서 1024px 이미지를 생성하며 최신 방법들에 필적하는 품질을 달성한다.

#### **기존 방법 대비 차별성**

DMD2 등의 방법은 실제 데이터에 대한 적대적 학습과 teacher 모델의 score distillation을 함께 적용한다. 가장 유사한 선행 연구인 UFO-Gen도 실제 데이터에만 적대적 학습을 적용하지만, 그 Discriminator가 손상된(corrupted) 데이터를 입력으로 받는 DiffusionGAN 방식을 채택한 반면, APT의 방법은 Discriminator에 실제 손상되지 않은 데이터를 입력으로 사용하여 표준 GAN 적대적 학습에 더 가깝게 따른다.

또한 UFO-Gen의 이미지 Generator와 Discriminator는 1B 파라미터 이하의 합성곱 모델인 반면, APT의 모델은 8B 파라미터의 트랜스포머 모델로 이미지와 비디오를 모두 생성한다.

---

### 2-5. 한계

실제 데이터에 대한 적대적 후훈련은 생성 분포를 실제 분포에 가깝게 만들지만, 실제 분포 자체가 텍스트 정렬이 더 약한 경우가 많다. 훈련 데이터셋에 텍스트 정렬 개선을 위한 재캡션(re-caption)이 이미 적용되었음에도 불구하고, 1-step 생성 모델은 classifier-free guidance에 비해 텍스트 정렬이 약하다는 한계가 있다.

평균적인 케이스에서 적대적 후훈련은 구조와 텍스트 정렬에서 열화가 나타날 수 있고, 일부 프롬프트에서 실패 케이스도 발생한다.

직접적인 확산 모델에 대한 적대적 학습은 매우 불안정하고 붕괴(collapse)에 취약하며, 특히 Generator와 Discriminator 모두 수십억 개의 파라미터를 가진 대규모 트랜스포머 모델일 때 더욱 그렇다.

---

## 3. 모델의 일반화 성능 향상 가능성

APT의 일반화 성능 향상 가능성은 여러 측면에서 분석할 수 있다.

### 3-1. 실제 데이터 직접 학습의 일반화 이점

기존 확산 증류 방법들은 사전 훈련된 확산 모델을 증류 teacher로 사용해 목표를 생성하는 반면, APT는 실제 데이터에 직접 DiT의 적대적 훈련을 수행하며 초기화 목적으로만 사전 훈련 모델을 활용한다.

이는 teacher 모델의 편향(bias)에 종속되지 않고 실제 데이터 분포를 직접 학습함으로써, **더 다양한 분포**에서 일반화 성능을 높이는 잠재력을 가진다.

### 3-2. 이미지-비디오 통합 모델의 일반화

APT를 통해 훈련된 모델은 단일 순전파로 이미지와 비디오를 모두 생성할 수 있는 통합 모델로서, 이미지와 비디오 두 도메인에 걸친 광범위한 일반화 능력을 시사한다.

### 3-3. Discriminator 앙상블의 일반화 기여

Discriminator는 다층 특징 추출과 다양한 타임스텝을 입력으로 사용하는 앙상블 기법을 사용하며, 이는 다양한 노이즈 수준에서의 실제/생성 데이터 구분 능력을 강화하여 다양한 시각적 패턴에 대한 일반화를 돕는다.

### 3-4. 제로샷 외삽(Zero-Shot Extrapolation) 가능성

후속 연구인 Seaweed-APT2에서, 모델은 훈련 중 보지 못한 훨씬 긴 길이의 비디오를 제로샷으로 외삽(zero-shot extrapolate)하여 생성할 수 있으며, 최대 5분(7200 프레임) 영상을 실시간 스트리밍 방식으로 KV 캐시를 활용해 1NFE 연산으로 생성한다.

### 3-5. 일반화의 한계와 트레이드오프

적대적 후훈련이 생성 분포를 실제 분포에 가깝게 만드는 반면, 실제 분포 자체가 텍스트 정렬이 더 약한 경향이 있어, 텍스트 조건 일반화(text-conditioned generalization) 측면에서는 오히려 약화가 발생할 수 있다는 트레이드오프가 존재한다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 방법 | 연도 | 핵심 접근 | 최소 NFE | 도메인 | 한계 |
|---|---|---|---|---|---|
| **DDPM** (Ho et al.) | 2020 | 확산 모델 기초 | ~1000 | 이미지 | 매우 느림 |
| **Progressive Distillation** (Salimans & Ho) | 2022 | 단계 절반씩 감소 | ~4 | 이미지 | 단계 한계 존재 |
| **Consistency Model** (Song et al.) | 2023 | ODE trajectory 일관성 | 1 | 이미지 | blurry |
| **LCM** (Luo et al.) | 2023 | Consistency + LoRA | 2~4 | 이미지 | 1-step에서 품질 저하 |
| **DMD** (Yin et al.) | 2024 | Distribution Matching Distillation | 1 | 이미지 | mode collapse 위험 |
| **UFO-Gen** (Xu et al.) | 2023 | 확산 + GAN (corrupted input) | 1 | 이미지 | 소규모 CNN, 이미지만 |
| **DMD2** (Yin et al.) | 2024 | DMD + 적대적 학습 혼합 | 1 | 이미지 | teacher 의존성 |
| **Seaweed-APT (본 논문)** | 2025 | APT (real data GAN, teacher 불필요) | **1** | **이미지+비디오** | 텍스트 정렬 약화 |

결정론적 방법들은 확산 모델이 노이즈-샘플 매핑의 결정론적 확률 흐름(probability flow)을 학습한다는 사실을 활용하여 더 적은 단계로 정확한 teacher 출력을 예측하려 하며, 여기에는 Progressive Distillation, Consistency Distillation, Rectified Flow 등이 포함된다.

DMD2와 일부 후속 연구들은 실제 데이터에 대한 적대적 학습과 teacher 모델로부터의 score distillation을 모두 적용한다.

**APT의 결정적 차별점**은 teacher 모델 없이 실제 데이터에 대한 순수 GAN 학습만을 사용한다는 점이며, 이를 수십억 파라미터 트랜스포머로 확장하는 데 성공한 것이 핵심 기여이다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려 사항

### 5-1. 연구에 미치는 영향

#### ① 대규모 GAN의 부활 가능성
APT를 통해 현재까지 보고된 가장 큰 GAN 중 하나(~16B 파라미터)가 성공적으로 훈련됨으로써, 한동안 확산 모델에 밀려 주목받지 못했던 GAN 패러다임의 대규모 적용 가능성을 다시 열었다.

#### ② 실시간 비디오 생성의 새 기준
적대적 후훈련 방법이 고해상도 비디오·이미지 생성의 추론을 단일 단계로 줄임으로써, 실시간 비디오 생성이 가능한 새로운 벤치마크를 제시한다. 이는 인터랙티브 미디어, 게임, 실시간 창작 도구 등 응용 연구에 큰 영향을 미친다.

#### ③ APT의 다른 영역으로의 확장
후속 연구인 Seaweed-APT2는 실시간 인터랙티브 응용을 위한 스트리밍 비디오 생성 모델로, APT1을 기반으로 자기회귀적 적대적 후훈련 패러다임을 채택해 최소한의 지연 시간으로 연속 비디오 프레임을 생성한다.

#### ④ 인프라 효율화
단일 H100 GPU로 실시간 고해상도 비디오를 생성한다는 것은 추론 비용의 극적 감소를 의미하며, 상용화 연구에 중요한 이정표가 된다.

---

### 5-2. 앞으로 연구 시 고려할 점

#### ① 텍스트 정렬(Text Alignment) 개선
적대적 후훈련이 실제 분포에 가까워지게 하지만, 실제 분포 자체의 텍스트 정렬이 약할 수 있고, 재캡션 전처리를 해도 classifier-free guidance보다는 텍스트 정렬이 떨어진다. 따라서 **reward 학습 기반 텍스트 정렬 강화** 연구가 필요하다.

#### ② 훈련 안정성 확보 기법 연구
직접적인 확산 모델에 대한 적대적 학습은 매우 불안정하고 붕괴에 취약하며, 특히 수십억 파라미터를 가진 대규모 트랜스포머 모델에서 더욱 그렇기 때문에 훈련 안정화를 위한 핵심 설계들이 도입된다. 향후 **더 효율적인 정규화 기법**과 **훈련 안정화 알고리즘** 개발이 중요하다.

#### ③ 모션 일관성 및 장기 의존성
빠른 동작(fast-motion) 시나리오는 여전히 단일 평가 설계에 도전적이며, 슬라이딩 윈도우 어텐션이 매우 긴 거리 의존성에 어려움을 겪을 수 있다. 또한 장시간 스트리밍에서 간헐적인 물리 법칙 위반 및 피사체 드리프트가 나타난다.

#### ④ 데이터 효율성 및 다양성
실제 비디오 데이터셋에 대한 분포 매칭에 의존하는 방법들은 생성된 캡션이 세밀한 디테일이 부족하여 텍스트-비디오 정렬이 약화되고 제어 가능성이 감소할 수 있다. 고품질 훈련 데이터 구축이 APT 접근법에서 특히 중요하다.

#### ⑤ 개방형 생태계 구축
현재 Seaweed-APT는 ByteDance Seed 내부 모델로, 공개 재현 가능한 구현이 없다. 학술 커뮤니티의 발전을 위해 **오픈소스 재현 연구** 및 **경량화 버전** 개발이 중요한 연구 방향이 될 것이다.

#### ⑥ 인간 선호도 정렬
Seaweed 연구팀은 인간 선호도 정렬(human-preference alignment), 메모리 확장, 강인성 개선에 대한 추가 연구를 계획하고 있다. RLHF/RLAIF 기반 정렬 연구와 APT의 결합이 중요한 미래 방향이다.

---

## 📚 참고 자료 및 출처

| # | 출처 |
|---|---|
| 1 | **논문 원문 (arXiv)**: Shanchuan Lin et al., "Diffusion Adversarial Post-Training for One-Step Video Generation," arXiv:2501.08316, 2025. https://arxiv.org/abs/2501.08316 |
| 2 | **ICML 2025 공식 게재**: Proceedings of the 42nd International Conference on Machine Learning, PMLR v267, pp. 37959–37974. https://proceedings.mlr.press/v267/lin25m.html |
| 3 | **논문 HTML 버전**: https://arxiv.org/html/2501.08316v1 |
| 4 | **PDF 원문**: https://arxiv.org/pdf/2501.08316 |
| 5 | **HuggingFace Paper Page**: https://huggingface.co/papers/2501.08316 |
| 6 | **OpenReview**: https://openreview.net/forum?id=AAgzsnhc28 |
| 7 | **EmergentMind 분석**: https://www.emergentmind.com/papers/2501.08316 |
| 8 | **ResearchGate**: https://www.researchgate.net/publication/388029117 |
| 9 | **Seaweed-APT 프로젝트 페이지**: https://seaweed-apt.com/ |
| 10 | **Seaweed-APT2 (후속 연구)**: https://seaweed-apt.com/2 |
| 11 | **RITS NYU - APT2 분석**: https://rits.shanghai.nyu.edu/ai/seaweed-apt2-real-time-interactive-video-generation-with-autoregressive-adversarial-post-training/ |
| 12 | **Deeplearn.org**: https://deeplearn.org/arxiv/567311/diffusion-adversarial-post-training-for-one-step-video-generation |

> ⚠️ **정확도 안내**: 본 답변은 공개된 arXiv 논문 원문(PDF/HTML), ICML 2025 공식 게재본, 및 관련 분석 자료를 기반으로 작성되었습니다. 논문 내부 ablation 실험 수치, 세부 하이퍼파라미터 등 검색으로 확인되지 않은 정보는 포함하지 않았습니다.
