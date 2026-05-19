
# Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy

> **논문 정보**
> - **저자:** Zhi Hou, Tianyi Zhang, Yuwen Xiong, Haonan Duan, Hengjun Pu, Ronglei Tong, Chengyang Zhao, Xizhou Zhu, Yu Qiao, Jifeng Dai, Yuntao Chen
> - **arXiv:** [2503.19757](https://arxiv.org/abs/2503.19757) (2025년 3월 25일)
> - **학회:** ICCV 2025
> - **프로젝트 페이지:** https://robodita.github.io/

---

## 1. 🔑 핵심 주장과 주요 기여 요약

최근 다양한 로봇 데이터셋으로 학습된 Vision-Language-Action(VLA) 모델들은 제한된 인도메인 데이터에서 유망한 일반화 능력을 보이지만, 이산화된 혹은 연속적 행동을 예측하는 **소형 action head에 대한 의존성**이 이질적인 행동 공간에 대한 적응력을 제한한다는 문제가 있다.

이를 해결하기 위해:

Dita는 Transformer 아키텍처를 활용해 통합된 멀티모달 확산 프로세스를 통해 **연속적 행동 시퀀스를 직접 노이즈 제거(denoise)하는 확장 가능한 프레임워크**를 제시하며, 얕은 네트워크를 통한 융합 임베딩 조건화 방식에서 벗어나 **인컨텍스트 조건화(in-context conditioning)** 를 사용하여 노이즈 제거된 행동과 과거 관측의 원시 시각 토큰 간의 세밀한 정렬을 가능하게 한다.

**주요 기여 4가지:**

| 기여 | 내용 |
|------|------|
| ① In-context conditioning | Action denoising을 raw visual token에 직접 조건화 |
| ② Scalable DiT 기반 정책 | Transformer 확장성을 action denoiser에 통합 |
| ③ Cross-embodiment 통합 | 다양한 카메라/행동 공간/태스크 데이터 통합 |
| ④ 10-shot 실세계 적응 | 제3자 시점 카메라만으로 실세계 일반화 달성 |

이 아키텍처는 일반화 로봇 정책 학습을 위한 **다목적·경량·오픈소스 베이스라인**을 확립한다.

---

## 2. 📐 해결하고자 하는 문제, 제안 방법(수식 포함), 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

기존 방법들은 오토리그레시브 멀티모달 Transformer로부터 하나의 임베딩에 조건화된 소형 네트워크(MLP/DiT)를 diffusion head로 사용하여 행동을 노이즈 제거한다. 그러나 다양한 카메라 뷰와 행동 공간을 포함하는 대규모 cross-embodiment 데이터셋 내의 방대한 로봇 공간은 소형 diffusion head가 연속적 행동을 효과적으로 노이즈 제거하는 데 상당한 도전을 제기한다.

또한 일부 확산 정책들은 노이즈 제거 프로세스 이전에 과거 이미지 관측과 지시를 임베딩으로 통합하려 하지만, 이는 노이즈 제거 학습을 제한할 수 있다. 행동 예측은 초기 융합 임베딩보다 직관적인 과거 관측에 더 많이 의존하기 때문이다.

---

### 2-2. 제안하는 방법 및 수식

#### ① Diffusion Policy 기본 수식

표준 DDPM(Denoising Diffusion Probabilistic Model) 기반의 행동 생성:

**Forward process (노이즈 추가):**

$$q(\mathbf{a}^k | \mathbf{a}^0) = \mathcal{N}(\mathbf{a}^k; \sqrt{\bar{\alpha}_k}\,\mathbf{a}^0,\; (1-\bar{\alpha}_k)\mathbf{I})$$

여기서 $\mathbf{a}^0$는 원본 행동, $k$는 diffusion timestep, $\bar{\alpha}\_k = \prod_{i=1}^{k}(1-\beta_i)$이다.

**Reverse process (노이즈 제거 학습 목표):**

$$\mathcal{L} = \mathbb{E}_{\mathbf{a}^0,\,k,\,\boldsymbol{\epsilon}}\left[\left\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{a}^k,\, k,\, \mathcal{C})\right\|^2\right]$$

여기서 $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$는 추가된 노이즈, $\boldsymbol{\epsilon}_\theta$는 DiT 기반 노이즈 예측 네트워크, $\mathcal{C}$는 컨텍스트(언어 + 이미지 토큰)이다.

모델은 인과 Transformer 구조로 언어 지시와 이미지 관측에 조건화되어 연속 행동에 추가한 노이즈에 의해 학습된다.

#### ② In-context Conditioning (Dita 핵심)

언어 지시 토큰, 이미지 관측 특징, 타임스텝, 노이즈가 추가된 행동을 연결(concatenate)하여 토큰 시퀀스를 구성하고 Transformer 네트워크에 입력한다. 핵심 설계는 각 단일 행동 토큰이 아닌 **행동 토큰 청크(action token chunks)를 노이즈 제거**하는 Diffusion Transformer 구조로, 인과 Transformer를 통해 이미지 관측 및 지시 토큰에 직접 조건화된 in-context conditioning 방식을 사용한다.

입력 시퀀스를 수식으로 나타내면:

$$\mathbf{S} = \left[\underbrace{\mathbf{T}_\text{lang}}_{\text{언어 토큰}},\; \underbrace{\mathbf{T}_\text{img}^{1}, \ldots, \mathbf{T}_\text{img}^{H}}_{\text{과거 H 프레임 이미지 토큰}},\; \underbrace{\mathbf{e}_k}_{\text{timestep}},\; \underbrace{\mathbf{a}^k_{1:L}}_{\text{노이즈 행동 청크}}\right]$$

여기서 $H$는 과거 관측 프레임 수, $L$은 행동 청크 길이이다.

#### ③ Action Delta 모델링

Dita는 노이즈 제거된 행동과 과거 관측의 원시 시각 토큰 간의 세밀한 정렬을 가능하게 하며, 이 설계는 **행동 델타(action delta)와 환경 뉘앙스를 명시적으로 모델링**한다.

행동 델타는 다음과 같이 정의된다:

$$\Delta \mathbf{a}_t = \mathbf{a}_{t} - \mathbf{a}_{t-1}$$

이를 통해 절대 행동값이 아닌 변화량을 학습함으로써 cross-embodiment 환경에서의 일반화를 촉진한다.

---

### 2-3. 모델 구조

모델은 사전 학습된 **CLIP 네트워크**로 언어 지시 토큰을 추출하는 Transformer 기반 확산 아키텍처를 채용하며, **DINOv2** 모델이 이미지 관측을 인코딩하고, **Q-Former**가 각 이미지에 대한 특징을 쿼리한다. 지시 토큰, 이미지 특징, 타임스텝 임베딩, 노이즈가 추가된 행동이 연결되어 토큰 시퀀스를 구성하고, 이 시퀀스가 네트워크에 입력되어 원시 행동을 노이즈 제거한다.

```
입력: 언어 지시 + 이미지 관측 (제3자 시점 RGB)
    │
    ├─ CLIP → 언어 토큰 (T_lang)
    ├─ DINOv2 + Q-Former → 이미지 토큰 (T_img)
    │
    ↓
[T_lang | T_img^1 | ... | T_img^H | e_k | a^k_{1:L}]
    │
    └─ Causal Transformer (DiT) → ε_θ (노이즈 예측)
    │
    └─ Reverse diffusion → 행동 청크 a^0_{1:L}
```

모델은 Transformer 네트워크의 확장성을 유지하면서 디노이징이 이미지 패치에 직접 조건화될 수 있도록 하며, 연속 행동에 추가된 노이즈에 의해 학습된다. 즉 소형 diffusion action head 방식과 달리, **대형 Transformer 모델을 사용하여 행동 청크 공간에 diffusion 목표를 직접 적용**한다.

---

### 2-4. 성능 향상

ManiSkill2, LIBERO, Calvin, SimplerEnv 등의 광범위한 벤치마크에서 Dita의 효과성과 일반화가 입증되었으며, Real-to-Sim 벤치마크 SimplerEnv, 실세계 Franka Arm, LIBERO에서 OpenVLA 및 Octo 대비 일관되게 더 나은 성능을 달성했다.

특히 단일 제3자 카메라 스트림만으로 Calvin ABC→D 태스크에서 5개 중 연속 완료 작업 수를 **3.6으로 끌어올리는 최고 성능**을 달성했으며, 사전 학습 단계가 Calvin에서 성공 시퀀스 길이를 1.2 이상 향상시켰다.

실세계에서는 다양한 환경 변화와 복잡한 장기 과제에 대해 **10-shot 파인튜닝만으로 강건한 적응**을 달성했다.

| 벤치마크 | 주요 결과 |
|-----------|-----------|
| Calvin ABC→D | 평균 완료 태스크 수 **3.6** (단일 3인칭 카메라) |
| SimplerEnv | OpenVLA, Octo 대비 일관적으로 우수 |
| LIBERO | OpenVLA, Octo 대비 우수 |
| 실세계 (Franka) | 10-shot으로 환경 변화 및 장기 과제 적응 성공 |

---

### 2-5. 한계

논문에서 직접 언급된 주요 한계:

이질적인 로봇 데이터에 걸쳐 다양한 센서, 행동 공간, 과제, 카메라 뷰, 환경을 포함하는 대규모 cross-embodiment 데이터셋에서 정책을 학습시키는 것은 여전히 **열린 과제(open challenge)** 로 남아 있다.

- **추론 속도:** Diffusion 기반 모델은 반복적 디노이징으로 인해 RT 제어(3Hz 동작 확인)에 제약이 존재하며, 시스템이 **3Hz의 제어 주파수**로 작동한다.
- **하드웨어 의존:** RealSense D435i RGB-D 카메라와 1대의 NVIDIA A100 GPU를 활용하는 서버가 필요하다.
- **평가 벤치마크 신뢰성:** LIBERO-PRO 연구에 따르면 기존 LIBERO 벤치마크는 조작 객체, 초기 상태, 태스크 지시, 환경 등 다양한 변화에 대한 일반화 평가가 미흡하며, 기존 모델들이 표준 평가에서 90% 이상의 정확도를 보이더라도 일반화 설정에서는 성능이 급격히 하락하는 문제가 있다.

---

## 3. 🌐 일반화 성능 향상 가능성

### 3-1. 일반화의 핵심 메커니즘

Dita는 인컨텍스트 조건화 메커니즘을 통해 연속 행동 시퀀스를 디노이징하는 Transformer 기반 확산 모델을 활용하며, Transformer의 확장성을 활용하여 광범위한 cross-embodiment 데이터셋에서 다양한 로봇 행동을 효과적으로 모델링하고 여러 시뮬레이션 벤치마크에서 통합 프레임워크 내에서 강건한 일반화를 달성한다.

또한 Dita는 강력한 **few-shot 적응 능력**을 보여주며, 최소한의 인도메인 샘플로 새로운 실세계 로봇 설정과 장기 과제로 전이에 성공한다.

### 3-2. Cross-embodiment 일반화

Diffusion action denoiser를 Transformer의 확장성과 결합하여 스케일링함으로써, Dita는 다양한 카메라 시점, 관측 장면, 과제, 행동 공간에 걸친 cross-embodiment 데이터셋을 효과적으로 통합하며, 이러한 시너지는 다양한 분산에 대한 강건성을 향상시키고 장기 과제의 성공적 실행을 촉진한다.

### 3-3. 시각적 변화에 대한 강건성

Dita는 10-shot으로 새로운 로봇 설정의 복잡한 멀티태스크 장기 시나리오에 적응할 수 있으며, 복잡한 물체 배치와 정교한 3D pick-and-rotation 과제에서 도전적인 조명 조건에서도 **뛰어난 강건성**을 보인다.

### 3-4. 일반화 가능성의 수식적 해석

인컨텍스트 조건화를 통한 일반화 과정:

$$p_\theta(\mathbf{a}^0 | \mathcal{O}_{1:H}, \ell) = \int p_\theta(\mathbf{a}^0 | \mathbf{a}^K, \mathcal{O}_{1:H}, \ell) \prod_{k=1}^{K} p_\theta(\mathbf{a}^{k-1}|\mathbf{a}^k, \mathcal{O}_{1:H}, \ell)\, d\mathbf{a}^{1:K}$$

여기서 $\mathcal{O}\_{1:H}$는 과거 $H$ 프레임 이미지 관측, $\ell$은 언어 지시이다. 기존 방식이 $\mathcal{O}_{1:H}$를 하나의 압축 임베딩으로 변환한 뒤 조건화하는 데 반해, Dita는 **원시 토큰 수준에서 직접 조건화**하므로 환경 변화에 훨씬 민감하게 반응할 수 있다.

---

## 4. 🔬 2020년 이후 관련 최신 연구 비교 분석

### 4-1. 관련 연구 비교표

| 모델 | 연도 | 행동 표현 | 확장성 | Cross-emb. | Few-shot |
|------|------|-----------|--------|------------|----------|
| **RT-1** | 2022 | 이산화 토큰 | ✓ | △ | ✗ |
| **Octo** | 2024 | Diffusion head (소형) | △ | ✓ | △ |
| **OpenVLA** | 2024 | 이산화 토큰 (LLM) | ✓ | ✓ | △ |
| **π₀ (pi0)** | 2024 | Flow Matching | ✓ | ✓ | ✓ |
| **Dita** | 2025 | In-context Diffusion DiT | ✓✓ | ✓✓ | ✓✓ |

### 4-2. 기존 연구의 한계와 Dita의 차별점

Robot Transformer(RT 시리즈)는 광범위한 OXE 데이터셋 학습을 통해 강건한 일반화를 달성하는 정책 프레임워크를 제시했고, Octo는 오토리그레시브 Transformer 설계에 diffusion action head를 채용하며 OpenVLA는 행동 공간을 이산화하고 사전 학습된 시각-언어 모델을 활용해 VLA 모델을 구축했다.

그러나 대규모 cross-embodiment 데이터셋 내의 방대한 로봇 공간은 소형 diffusion head에 상당한 도전을 제기하며, 다른 확산 정책들은 노이즈 제거 과정 이전에 과거 이미지 관측과 지시를 임베딩으로 통합하려 해 노이즈 제거 학습을 제한할 수 있다.

Dita의 설계는 Transformer 네트워크의 확장성을 유지하면서 디노이징 학습이 **이미지 패치에 직접 조건화**되도록 하여 모델이 과거 관측에서 세밀한 행동 변화를 포착할 수 있게 한다.

### 4-3. 최신 연구 동향과 위치

- **RT-2 (2023):** 인터넷 규모 VLM을 로봇 정책으로 전이 → Dita는 이보다 확산 모델 표현력을 추가
- **Diffusion Policy (2023, Chi et al.):** 단일 환경 확산 정책의 성공 → Dita는 이를 cross-embodiment로 확장
- **$\pi_0$ (2024, Black et al.):** Flow Matching 기반 범용 정책 → Dita는 DiT 구조의 Transformer 확장성에 집중
- 최근 범용 로봇 정책의 발전은 탐색과 조작 모두에 걸쳐 기반 멀티모달 모델을 활용하며, 확장 가능한 VLA 모델이 지배적 프레임워크로 부상했다. 일부 접근법은 시간적 시각 추론 향상을 위해 인터넷 규모 데이터로 학습된 대규모 비디오 백본을 통합한다.

---

## 5. 🚀 향후 연구에 미치는 영향과 고려 사항

### 5-1. 향후 연구에 미치는 영향

**① 확장 가능한 로봇 정책 패러다임 제시**

Transformer의 확장 능력을 활용함으로써 대규모 다양한 로봇 데이터셋에 걸쳐 연속적 엔드이펙터 행동을 효과적으로 모델링하고 더 나은 일반화 성능을 달성할 수 있음을 보여준다. 이는 향후 연구에서 DiT 기반 구조가 로봇 정책의 표준 아키텍처로 자리잡을 가능성을 높인다.

**② In-context Conditioning의 일반화**

추가적인 관측 토큰과 입력이 Transformer 아키텍처에 원활하게 통합될 수 있다. 이는 깊이 센서, 고유 감각(proprioception), 촉각 센서 등 다양한 모달리티를 추가하는 멀티모달 확장 연구를 자극할 것이다.

**③ 오픈소스 베이스라인으로의 가치**

이 아키텍처는 범용 로봇 정책 학습을 위한 **다목적·경량·오픈소스 베이스라인**을 확립한다. 연구 커뮤니티가 이를 기반으로 다양한 개선 연구를 진행할 수 있는 플랫폼이 된다.

**④ Few-shot 적응 연구 촉진**

이질적인 로봇 데이터로 사전 학습되고 최소한의 감독으로 파인튜닝된 범용 로봇 정책이 VLA 모델 개발에서 **진정한 일반화 실현에 핵심적인 역할**을 할 수 있음을 시사한다.

---

### 5-2. 향후 연구 시 고려할 점

#### 🔴 기술적 과제

1. **추론 속도 개선:** Diffusion 모델의 반복적 디노이징은 3Hz 제어 주파수라는 실시간성 한계를 가진다. DDIM, Consistency Model, Flow Matching 등을 활용한 inference 가속화 연구가 필요하다.

2. **벤치마크 신뢰성:** LIBERO-PRO처럼 조작 객체, 초기 상태, 태스크 지시, 환경 등 다양한 차원의 변화를 체계적으로 평가하는 설정이 필요하며, 기존 표준 평가에서 90% 이상을 달성한 모델도 일반화 설정에서는 성능이 0%로 붕괴되는 현상이 관찰된다. 따라서 단순 성공률 보고를 넘어 robust한 평가 프로토콜 설계가 중요하다.

3. **데이터 다양성:** 다양한 센서, 행동 공간, 과제, 카메라 뷰, 환경을 포함하는 광범위한 cross-embodiment 데이터셋에서 정책 학습이 여전히 열린 과제로 남아 있다.

#### 🟡 연구 방향 제안

4. **멀티뷰·멀티모달 확장:** Dita는 현재 단일 제3자 시점 RGB 카메라만을 사용하므로, 손목 카메라·깊이 센서 통합이 조작 성능 향상에 기여할 수 있다.

5. **온라인 학습과의 결합:** 강화학습 또는 자기 개선(self-improving) 파이프라인과의 결합으로 데모 없이도 지속적 성능 향상이 가능하다.

6. **능동적 불확실성 추정:** 확산 모델의 확률적 특성을 활용하여 에피스테믹 불확실성을 추정하고, 안전-critical 로봇 제어에 활용하는 연구가 가치 있다.

7. **언어 기반 태스크 구성:** 복잡한 언어 지시로부터 하위 태스크를 분해하여 Dita의 장기 과제 수행 능력을 더욱 향상시킬 수 있다.

---

## 📚 참고 자료 및 출처

| # | 출처 | 링크 |
|---|------|-------|
| 1 | **Dita 공식 arXiv 논문** (arXiv:2503.19757) | https://arxiv.org/abs/2503.19757 |
| 2 | **Dita 프로젝트 페이지** | https://robodita.github.io/ |
| 3 | **ICCV 2025 논문 (OpenAccess)** | https://openaccess.thecvf.com/content/ICCV2025/papers/Hou_Dita_... |
| 4 | **Dita PDF 전문** | https://arxiv.org/pdf/2503.19757 |
| 5 | **Diffusion Transformer Policy (전신 논문)** arXiv:2410.15959 | https://arxiv.org/html/2410.15959v4 |
| 6 | **HuggingFace Paper Page** | https://huggingface.co/papers/2503.19757 |
| 7 | **LIBERO-PRO** (arXiv:2510.03827) - 벤치마크 신뢰성 관련 | https://arxiv.org/pdf/2510.03827 |
| 8 | **OpenVLA GitHub** | https://github.com/openvla/openvla |
| 9 | **Consensus Paper Summary** | https://consensus.app/papers/dita-scaling-diffusion-transformer... |

> ⚠️ **정확도 주의사항:** 본 답변의 수식 중 일부(Forward/Reverse Diffusion, Action Delta, 조건부 확률 분해)는 논문의 일반적인 DDPM/Diffusion Policy 방법론을 기반으로 표준적으로 기술한 것입니다. 논문 PDF의 구체적 수식 표기와 미세한 차이가 있을 수 있으므로, 정확한 수식은 반드시 원문 PDF(https://arxiv.org/pdf/2503.19757)를 직접 확인하시기 바랍니다.
