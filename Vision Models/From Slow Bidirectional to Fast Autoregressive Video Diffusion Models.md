
# From Slow Bidirectional to Fast Autoregressive Video Diffusion Models

> **논문 정보**
> - **저자:** Tianwei Yin*, Qiang Zhang*, Richard Zhang, William T. Freeman, Frédo Durand, Eli Shechtman, Xun Huang (*equal contribution)
> - **소속:** MIT, Adobe Research
> - **학회:** CVPR 2025 (arXiv: 2412.07772)
> - **프로젝트 페이지:** https://causvid.github.io/
> - **코드:** https://github.com/tianweiy/CausVid

---

## 1. 📌 핵심 주장 및 주요 기여 (요약)

이 논문은 **CausVid**를 소개하며, 빠르고 인터랙티브한 인과적(causal) 비디오 생성을 위한 모델로, 비디오 프레임 간의 인과적 의존성을 가진 자기회귀적(autoregressive) 확산 트랜스포머 아키텍처를 설계합니다.

### 🔑 세 가지 핵심 기여

| 기여 | 설명 |
|------|------|
| **① 양방향 → 인과적 변환** | 사전학습된 양방향 DiT를 자기회귀 트랜스포머로 적응 |
| **② DMD의 비디오 확장** | 50-step → 4-step 蒸류로 극단적 속도 향상 |
| **③ 비대칭 증류 전략** | 인과적 학생 모델을 양방향 교사로 학습, 오류 누적 방지 |

종합 실험 결과, 이 모델은 최첨단 양방향 확산 모델과 동등한 비디오 품질을 달성하면서도 향상된 인터랙티브성과 속도를 제공하며, 지식의 한에서 이는 품질 면에서 양방향 확산과 경쟁하는 최초의 자기회귀 비디오 생성 방법입니다.

---

## 2. 🔬 상세 분석

### 2.1 해결하고자 하는 문제

양방향 의존성은 단일 프레임 생성 시 전체 비디오를 처리해야 함을 의미하며, 이는 긴 지연 시간을 유발하고 인터랙티브 및 스트리밍 애플리케이션에 적용하는 것을 막습니다. 특히 현재 프레임 생성이 아직 존재하지 않는 미래 조건 입력에 의존한다는 문제가 있습니다.

또한, 현재 비디오 확산 모델은 속도 면에서도 제한적입니다. 연산과 메모리 비용이 프레임 수에 따라 이차적으로 증가하며, 추론 중 많은 수의 잡음 제거 단계와 결합되어 긴 비디오를 생성하는 것이 매우 느리고 비용이 많이 듭니다.

예를 들어, 기존 양방향 확산 모델은 128프레임 비디오 생성에 **219초**가 소요되며 전체 시퀀스가 완료될 때까지 기다려야 합니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### 🔹 Step 1: 비디오 확산 프로세스 기본 공식

비디오 확산 모델의 순방향(forward) 프로세스는 다음과 같이 정의됩니다:

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$$

여기서 $\mathbf{x}_0$는 원본 비디오, $\bar{\alpha}_t$는 노이즈 스케줄, $t$는 타임스텝입니다.

#### 🔹 Step 2: 자기회귀적 비디오 생성 구조

LLM의 decoder-only 구조와 유사하게, 모델은 각 반복에서 모든 입력 프레임의 supervision을 활용하여 샘플 효율적 학습을 달성하고, KV 캐싱을 통한 효율적인 자기회귀 추론을 수행합니다.

자기회귀 생성의 확률 분해:

$$p(\mathbf{x}^{1:T}) = \prod_{i=1}^{T} p(\mathbf{x}^i | \mathbf{x}^{1:i-1})$$

각 청크(chunk)는 이전에 생성된 클린(clean) 프레임에 조건화되어 순차적으로 잡음 제거됩니다:

$$\mathbf{x}^i \sim p_\theta(\mathbf{x}^i | \mathbf{x}^{1:i-1})$$

#### 🔹 Step 3: Distribution Matching Distillation (DMD) 비디오 확장

DMD는 원래 이미지 확산 모델을 위해 설계된 소수 단계 증류 접근법으로, 이를 비디오 데이터에 적용합니다. 자기회귀 확산 모델을 단순히 소수 단계 학생으로 증류하는 대신, 양방향 어텐션을 가진 사전 학습된 교사 확산 모델의 지식을 인과적 학생 모델로 증류하는 **비대칭 증류 전략**을 제안합니다.

DMD 목적 함수는 분포 매칭 손실 기반입니다:

$$\mathcal{L}_{\text{DMD}} = \mathbb{E}_{\mathbf{x} \sim G_\phi(\mathbf{z})} \left[ D_{\text{KL}}\left( p_{G_\phi} \| p_{\text{data}} \right) \right]$$

실용적으로는 다음 기울기 추정 방식을 사용합니다:

$$\nabla_\phi \mathcal{L}_{\text{DMD}} \approx \mathbb{E}\left[ \left( s_{\text{gen}}(\mathbf{x}_t, t) - s_{\text{data}}(\mathbf{x}_t, t) \right) \frac{\partial G_\phi(\mathbf{z})}{\partial \phi} \right]$$

여기서:
- $G_\phi$: 인과적 학생 생성기 (4-step 자기회귀 모델)
- $s_{\text{data}}$: 양방향 교사 모델 (frozen)
- $s_{\text{gen}}$: 학생 생성 분포를 추정하는 가짜 스코어 네트워크

#### 🔹 Step 4: 학생 초기화 (ODE 기반)

학생 초기화 단계에서, 양방향 교사가 생성한 ODE 솔루션 쌍의 소규모 데이터셋에 대해 인과적 학생을 사전학습합니다. 이 단계는 이후 증류 학습을 안정화하는 데 도움을 줍니다.

ODE 초기화의 회귀 손실:

$$\mathcal{L}_{\text{ODE}} = \mathbb{E}_{\mathbf{z}, \{t^i\}} \left\| G_\phi(\mathbf{z}, \{t^i\}) - \mathbf{x}_{t^i}^* \right\|_2^2$$

여기서 $\mathbf{x}_{t^i}^*$는 양방향 교사의 ODE 궤적에서 추출한 중간 상태입니다.

ODE 초기화는 계산 효율적으로, 상대적으로 적은 ODE 솔루션 쌍에 대한 소수의 학습 반복만을 필요로 합니다.

---

### 2.3 모델 구조 (Architecture)

교사 모델은 CogVideoX와 유사한 아키텍처를 가진 양방향 DiT(Diffusion Transformer)입니다.

```
[CausVid 아키텍처 구성]
┌─────────────────────────────────────────────────┐
│          Autoregressive DiT (CausVid)           │
│                                                 │
│  Frame_i  →  [Causal Attn] → Frame_i 생성       │
│               ↑                                 │
│  Frame_1,...,Frame_{i-1} (KV Cache로 저장)       │
│                                                 │
│  Teacher: Bidirectional DiT (frozen)            │
│  Student: Causal DiT (학습 대상)                │
└─────────────────────────────────────────────────┘
```

자기회귀(인과적) 모델은 각 프레임이 이전 프레임에만 어텐션을 주는 **인과적 어텐션**을 사용하여 스트리밍 애플리케이션을 위한 순차적 생성을 가능하게 합니다.

추론 효율성 향상을 위해, 이전에 계산된 정보를 재사용하는 **KV 캐싱 메커니즘**을 적용하여 동일한 어텐션 스코어의 반복 계산을 피함으로써 생성 속도를 크게 높입니다.

두 단계 학습 파이프라인:

| 단계 | 방법 | 목적 |
|------|------|------|
| **Stage 1: Student Init** | ODE 궤적 기반 회귀 학습 | 학습 안정화 |
| **Stage 2: Asymmetric Distillation** | DMD 손실 + 양방향 교사 | 품질 및 속도 최적화 |

---

### 2.4 성능 향상

모델은 VBench-Long 벤치마크에서 **84.27점**을 달성하여 모든 이전 비디오 생성 모델을 능가합니다.

CausVid는 초기 지연 시간 **1.3초**, 이후 단일 GPU에서 약 **9.4 FPS**의 스트리밍 방식으로 프레임을 연속 생성합니다.

인간 평가에서도 빠른 모델이 CogVideoX, PyramidFlow, MovieGen, 양방향 교사 모델보다 우수한 성능을 보였으며, 이 경쟁 모델들은 CausVid보다 수 배 느립니다.

단기 비디오 생성에서 CausVid는 10초 비디오 기준 시간적 품질 94.7, 프레임 품질 64.4, 텍스트 정렬 30.1로 기존 최첨단 방법을 크게 능가합니다.

특히 스트리밍 추론을 대화형 프레임레이트(9.4FPS)로 지원하는 유일한 방법이며, CogVideoX와 MovieGen 등 다른 두 방법은 10초짜리 비디오 생성에 최소 210초가 소요됩니다.

**속도 비교 요약:**

| 모델 | 생성 방식 | 초기 지연 | 처리량 |
|------|----------|----------|--------|
| **CausVid (Ours)** | 스트리밍 (자기회귀) | **1.3초** | **9.4 FPS** |
| CogVideoX | 양방향 (전체 처리) | 210초+ | 스트리밍 불가 |
| MovieGen | 양방향 (전체 처리) | 210초+ | 스트리밍 불가 |
| PyramidFlow | 피라미드 플로우 | 수 분 | 스트리밍 불가 |

---

### 2.5 한계점

증류된 인과적 모델이 프레임 단위 품질에서 양방향 확산 교사를 능가하는 반면, **약간 증가한 시간적 플리커링(flickering)과 감소된 다양성** 등 약간의 단점을 보이며, 이는 추가 개선이 필요한 영역입니다.

ablation 연구에 의하면, few-step 증류 과정, 특히 양방향 교사 모델과의 결합이 CausVid의 높은 성능에 핵심적이며, 증류 없이 양방향 모델을 인과적 모델로 직접 fine-tuning하는 경우 성능이 크게 저하되고 심각한 오류 누적이 발생합니다.

---

## 3. 🌐 모델의 일반화 성능 향상 가능성

### 3.1 Zero-Shot 태스크 일반화

텍스트-비디오 생성만을 대상으로 학습되었음에도 불구하고, CausVid는 자기회귀 설계 덕분에 이미지-비디오 생성 태스크에 **zero-shot**으로 적용될 수 있습니다.

이 접근법은 스트리밍 비디오-비디오 변환(video-to-video translation), 이미지-비디오(image-to-video), 동적 프롬프팅(dynamic prompting)을 **zero-shot** 방식으로 가능하게 합니다.

CausVid는 이미지-비디오 생성, 비디오-비디오 변환, 동적 프롬프팅 등 다양한 태스크에서 다용도성을 입증하며 낮은 지연 시간으로 최첨단 품질을 달성합니다.

### 3.2 긴 비디오 일반화 (길이 외삽)

비대칭 증류 접근법은 자기회귀 추론 중 오류 누적을 크게 줄이며, 이는 학습 중에 본 것보다 훨씬 긴 비디오를 자기회귀적으로 생성할 수 있게 합니다.

생성기가 짧은 비디오 클립으로 학습되었지만, 자기회귀적 특성 덕분에 슬라이딩 윈도우 추론을 통해 무한 길이 비디오를 생성할 수 있습니다.

이 접근의 일반화 메커니즘을 수식으로 표현하면:

$$p(\mathbf{x}^{1:T_{\text{long}}}) = \prod_{i=1}^{T_{\text{long}}} p_\theta(\mathbf{x}^i | \mathbf{x}^{\max(1, i-W):i-1})$$

여기서 $W$는 슬라이딩 윈도우 크기이며, $T_{\text{long}} \gg T_{\text{train}}$이어도 적용 가능합니다.

### 3.3 인터랙티브 애플리케이션으로의 일반화

인터랙티브 UI를 통해 텍스트-10초 비디오 생성, 슬라이딩 윈도우 추론을 통한 무한 비디오 생성, 이미지-비디오 생성 기능을 제공하며, 초기 지연 1.3초 이후 약 9.4 FPS로 프레임을 스트리밍 방식으로 지속 생성합니다.

---

## 4. 🔄 2020년 이후 관련 최신 연구 비교 분석

### 4.1 비디오 확산 모델 비교

| 모델 | 연도 | 어텐션 타입 | 속도 | 스트리밍 | 특징 |
|------|------|-----------|------|----------|------|
| **CausVid** | 2024 | 인과적(Causal) | 9.4 FPS | ✅ | DMD + 비대칭 증류 |
| CogVideoX | 2024 | 양방향 | ~210초/10s | ❌ | Expert Transformer |
| PyramidFlow | 2024 | 양방향 | 수 분 | ❌ | Pyramidal Flow Matching |
| MovieGen | 2024 | 양방향 | ~210초/10s | ❌ | LLaMA-like 구조 |
| Ca2-VDM | 2024 | 인과적 | 빠름 | ✅ | KV Cache 공유 |

CogVideoX는 확산 트랜스포머 기반의 대규모 텍스트-비디오 생성 모델로, 초당 16프레임, 768×1360 해상도로 텍스트 프롬프트와 정렬된 10초짜리 연속 비디오를 생성할 수 있습니다.

Ca2-VDM은 확산 모델의 발전으로 비디오 생성이 인상적인 품질을 달성하였으며, 이전에 생성된 클립의 마지막 프레임에 조건화하여 순차적으로 비디오를 생성하는 자기회귀 방식을 사용하나, 인접한 클립 간 겹치는 조건부 프레임을 재계산하는 비효율성이 있습니다.

### 4.2 증류 방법 비교

최근 연구자들은 프로그레시브 증류, 일관성 증류, 적대적 증류 등의 방법을 비디오 확산 모델에 적용하기 시작했지만, 대부분의 접근법은 2초 미만의 짧은 비디오 생성을 위한 모델 증류에 초점을 맞추고 있으며, 비인과적 교사를 비인과적 학생으로 증류하는 방식을 사용합니다. 이와 달리, CausVid의 방법은 비인과적 교사를 인과적 학생으로 증류하여 스트리밍 비디오 생성을 가능하게 합니다.

---

## 5. 🔮 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 연구에 미치는 영향

**① 패러다임 전환 가능성**

CausVid는 자기회귀 비디오 확산이 일반 텍스트-비디오 작업을 위해 효과적으로 확장될 수 있음을 보여주며, 양방향 확산 모델과 동등한 품질을 달성하면서, 증류 기술과 결합하면 수 배의 속도 향상을 제공합니다.

**② 새로운 애플리케이션 영역 개척**

기존 양방향 모델로는 불가능했던 사용자 입력이 시간에 따라 변할 수 있는 인터랙티브 및 스트리밍 애플리케이션에서의 지속적인 프레임 생성이 이제 가능해집니다.

**③ 지식 증류의 비대칭적 활용**

DMD는 교사와 학생 간에 서로 다른 아키텍처 구성을 허용하는 유일한 이점을 제공하며, 이를 통해 양방향 교사 확산 모델로부터 자기회귀 생성기를 학습하는 비대칭적 접근법이 가능합니다.

### 5.2 앞으로의 연구 시 고려할 점

**① 시간적 일관성 개선**

약간 증가한 시간적 플리커링과 감소된 다양성이 개선이 필요한 영역으로 지적되었습니다. 향후 연구는 인과적 아키텍처에서의 시간적 일관성 강화 메커니즘 개발에 집중해야 합니다.

**② 오류 누적의 이론적 한계 분석**

자기회귀 생성에서 오류 누적은 수식적으로 다음 문제입니다:

$$\epsilon_{\text{total}} = \sum_{i=1}^{T} \epsilon_i \cdot \prod_{j=i}^{T} \lambda_j$$

여기서 $\lambda_j$는 오류 전파 계수입니다. 비대칭 증류가 $\lambda_j$를 줄이는 원리의 이론적 증명이 필요합니다.

**③ 더 강력한 교사 모델 활용**

인과적 확산 모델을 fine-tune하여 교사로 사용하는 방식은 이론적으로는 유망하나, 초기 실험에서 이 단순한 접근은 최적이 아닌 결과를 보였습니다. 인과적 확산 모델은 일반적으로 양방향 모델보다 성능이 떨어지므로, 더 약한 인과적 교사로부터 학생을 학습시키면 학생의 능력이 본질적으로 제한됩니다.

더 강력한 양방향 교사 모델(예: Sora, HunyuanVideo 등)을 활용한 증류 연구가 주요 발전 방향입니다.

**④ 다양한 모달리티로의 확장**

본 논문의 비대칭 증류 전략은 오디오-비디오 공동 생성, 3D 영상 생성 등 다른 시퀀셜 모달리티로의 확장 가능성이 있습니다.

**⑤ 스케일링 법칙(Scaling Law) 검증**

CausVid의 핵심 혁신인 사전 학습된 양방향 확산 트랜스포머를 자기회귀 트랜스포머로 적응시키는 방식이 더 큰 모델 규모에서도 일관된 성능 향상을 보이는지 스케일링 법칙 분석이 필요합니다.

**⑥ 실시간 인터랙티브 제어**

현재 CausVid가 지원하는 동적 프롬프팅을 넘어, 실시간 사용자 제어(카메라 조작, 물리 시뮬레이션 등)와의 통합을 위한 조건화 메커니즘 연구가 필요합니다.

---

## 📚 참고 자료 (출처)

1. **arXiv 원문**: Yin, T. et al. (2024). *From Slow Bidirectional to Fast Autoregressive Video Diffusion Models*. arXiv:2412.07772. https://arxiv.org/abs/2412.07772

2. **CVPR 2025 공식 게재**: Yin, T. et al. (2025). *From Slow Bidirectional to Fast Autoregressive Video Diffusion Models*. CVPR 2025, pp. 22963–22974. https://openaccess.thecvf.com/content/CVPR2025/html/Yin_From_Slow_Bidirectional_to_Fast_Autoregressive_Video_Diffusion_Models_CVPR_2025_paper.html

3. **프로젝트 페이지 (CausVid)**: https://causvid.github.io/

4. **GitHub 코드**: https://github.com/tianweiy/CausVid

5. **Adobe Research 공식 페이지**: https://research.adobe.com/publication/causvid/

6. **IEEE Xplore**: https://ieeexplore.ieee.org/document/11092830

7. **관련 논문 - CogVideoX**: Yang, Z. et al. (2024). *CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer*. arXiv:2408.06072. https://arxiv.org/abs/2408.06072

8. **관련 논문 - Ca2-VDM**: *Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing*. https://openreview.net/forum?id=YbtH1aoE1V

9. **관련 논문 - PyramidFlow**: Jin, Y. et al. (2024). *Pyramidal Flow Matching for Efficient Video Generative Modeling*. ICLR 2025. https://github.com/jy0205/Pyramid-Flow

10. **Moonlight Literature Review**: https://www.themoonlight.io/en/review/from-slow-bidirectional-to-fast-autoregressive-video-diffusion-models

11. **Liner Quick Review**: https://liner.com/review/from-slow-bidirectional-to-fast-autoregressive-video-diffusion-models

12. **DeepWiki CausVid 분석**: https://deepwiki.com/tianweiy/CausVid
