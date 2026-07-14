# LingBot-Video: Scaling Mixture-of-Experts Video Pretraining for Embodied Intelligence 

---

## 1. 핵심 주장과 주요 기여 (요약)

이 논문은 **LingBot-Video**라는, embodied intelligence(로봇/구체화 지능)에 특화된 DiT(Diffusion Transformer) 기반 비디오 사전학습 모델을 제안합니다. 이 논문은 로봇 제어에서의 최근 가능성에도 불구하고, 비디오 생성 모델들이 콘텐츠 생성에 주력하기 때문에 도메인 불일치 문제를 겪고 있다고 지적하며, 대신 MoE(Mixture-of-Experts) 프레임워크를 채택합니다.

핵심 기여는 세 가지 관점으로 정리됩니다:
- **아키텍처**: 모델링 용량과 추론 효율성 사이의 더 나은 균형을 위해 dense 대신 MoE 프레임워크를 채택하고, 이를 처음부터(from scratch) 스케일업하는 데 성공했습니다.
- **데이터**: 표준 인터넷 비디오에 조작(manipulation), 내비게이션, 1인칭 시점을 포함하는 로봇 지향 영상을 대폭 보강하는 데이터 프로파일링 엔진을 구축하여, 기본 모델이 행동과 세계 동역학에 대한 본질적 이해를 갖추도록 했습니다.
- **학습**: 미학, 프롬프트 순응도, 모션 일관성 같은 표준 기준을 넘어 물리적 합리성과 과업 완수에 대한 정렬을 강제하는 다차원 보상 시스템을 개발했습니다.

결과적으로 LingBot-Video는 디지털 창작과 물리적 행동을 잇는 선구적 시도로서, 커뮤니티에 최초의 대규모 오픈소스 MoE 비디오 파운데이션 모델로 공개되었습니다.

---

## 2. 문제 정의, 방법론, 모델 구조, 성능, 한계

### (1) 해결하고자 하는 문제
기존 비디오 생성 모델은 시각적 충실도와 창의성을 계산 효율성 및 물리적 사실성보다 본질적으로 우선시하도록 설계되어 있습니다. 즉, 일반 영상 생성 모델은 "예쁘게 보이는" 영상을 만드는 데는 강하지만, 로봇이 실제 물리 법칙과 상호작용 결과를 예측/모사하는 데는 부적합하다는 domain mismatch가 핵심 문제입니다.

### (2) 모델 구조 및 방법(수식 포함)

**① MoE 기반 DiT 아키텍처**
공개된 정보에 따르면 LingBot-Video는 DiT 기반 아키텍처에 sparse MoE를 적용하여, 토큰당 3B 파라미터만 활성화하면서 총 30B의 용량을 유지합니다. 구체적으로 Robbyant는 300억 총 파라미터와 토큰당 30억 활성 파라미터를 가진 30B-A3B MoE 모델을 공개했으며, 이보다 작은 1.3B dense 변형도 함께 제공했습니다.

이러한 top- $k$ 라우팅 기반 MoE FFN 레이어는 일반적으로 다음과 같은 형태로 표현됩니다(DiT-MoE류 아키텍처의 표준 정식화이며, 본 논문도 이 계열을 따릅니다):

$$
\text{MoE}(x) = \sum_{i=1}^{N} g_i(x) \cdot E_i(x), \qquad g_i(x) = \begin{cases} \text{Softmax}(W_r x)_i & \text{if } i \in \text{Top-}k(W_r x) \\ 0 & \text{otherwise} \end{cases}
$$

여기서 $E_i$는 $i$번째 전문가(expert) FFN, $W_r$는 라우터(router) 가중치, $N$은 전체 전문가 수, $k$는 토큰당 활성화되는 전문가 수입니다. 이러한 라우팅 불균형을 막기 위해 통상 보조 손실(auxiliary load-balancing loss)이 추가됩니다:

$$
\mathcal{L}_{\text{balance}} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i
$$

($f_i$: 전문가 $i$에 라우팅된 토큰 비율, $P_i$: 라우터가 부여한 평균 확률, $\alpha$: 가중치 하이퍼파라미터). 이 구조를 통해 기존 dense 모델 대비 약 3배 빠른 추론 속도를 달성했습니다.

**② 데이터 엔진**
모델은 7만 시간 이상의 embodied 데이터를 포함하는 웹 비디오로 학습되었으며, 학습 시 보상 신호는 미학뿐 아니라 물리적 합리성과 과업 완수를 함께 가중했습니다.

**③ 사전학습 목적함수 (Flow-matching/Diffusion 손실)**
DiT 계열 비디오 생성모델(LingBot-Video가 속한 계열)이 일반적으로 사용하는 flow-matching 목적함수는 다음과 같은 형태입니다:

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t,\,x_0,\,x_1}\Big[\big\|\,v_\theta(x_t, t, c) - (x_1 - x_0)\,\big\|^2\Big], \qquad x_t = (1-t)x_0 + t x_1
$$

여기서 $x_0$는 노이즈, $x_1$은 실제 데이터(비디오 latent), $c$는 텍스트/이미지 조건, $v_\theta$는 MoE-DiT가 예측하는 속도장(velocity field)입니다.

**④ 다차원 보상 정렬**
학습 후반 단계에서는 여러 보상 항을 결합한 형태의 정렬 손실을 사용하는 것으로 알려져 있으며, 개념적으로는 다음처럼 나타낼 수 있습니다:

$$
\mathcal{L}_{\text{reward}} = -\left(\lambda_{\text{aes}} R_{\text{aesthetic}} + \lambda_{\text{phys}} R_{\text{physical}} + \lambda_{\text{task}} R_{\text{task}} + \lambda_{\text{motion}} R_{\text{motion}}\right)
$$

이는 미학, 프롬프트-순응도, 모션 일관성이라는 표준 기준을 넘어 물리적 합리성과 과업 완수에 대한 정렬을 강제한다는 설명과 부합하는 구조입니다.

> ⚠️ 위 수식들은 논문이 속한 DiT/MoE/flow-matching 계열 모델의 표준적 정식화를 재구성한 것이며, arXiv 원문의 정확한 수식 번호나 세부 하이퍼파라미터 표기는 검색 결과에서 직접 확인되지 않았습니다.

**⑤ 성능**
공개 벤치마크 결과는 다음과 같습니다:
- RBench는 로봇 중심 상호작용의 정확성을 목표로 하며, 650개의 텍스트-이미지 프롬프트가 5가지 상호작용 유형(조작, 공간관계, 다중개체 협업, 장기 계획, 시각적 추론)의 250개 과업 지향 시나리오와 4가지 로봇 형태(단일 암, 양팔, 인간형, 4족)의 400개 구체화 특정 시나리오로 구성됩니다.
- RBench 전체 평균은 0.620이며, 4족 보행 0.758, 인간형 0.689, 양팔 0.639, 장기 과업 0.634를 기록했으나, 추론(reasoning)은 0.505, 다중개체 장면은 0.444로 가장 취약했습니다.
- Veo 3(0.563), Seedance 1.5 Pro(0.584) 같은 상용 폐쇄형 모델들도 능가했습니다.
- Physics-IQ Verified는 영상이 시각적으로만 그럴듯한 움직임이 아니라 실제 물리 현상을 예측할 수 있는지 평가하는 벤치마크로, 고체/유체 동역학, 열역학, 광학, 자기학을 포함한 66개의 통제된 물리 실험, 3개 시점·2회 촬영으로 총 396개의 실제 영상으로 구성됩니다. 이 벤치마크에서 40.4점을 기록했습니다.
- RBench에서의 우수한 성능은 내부 벤치마크에서 관찰된 로봇 도메인 우위와 일치하며, LingBot-Video의 물리 세계 모델링 능력이 내부 평가 스위트를 넘어 일반화된다는 것을 보여줍니다.

**⑥ 한계**
- 벤치마크 결과의 신뢰성 측면에서 이 수치들은 모델 자체 페이지에서 나온 자체 보고 값이며 독립적인 검증은 아니라는 점이 정직한 유의사항입니다.
- 추론(reasoning)과 다중개체 시나리오는 상대적으로 취약하여, 복잡한 다중 객체 상호작용이나 고차원적 계획 추론에는 한계가 있습니다.
- 실행 측면에서 FSDP 추론 시에도 각 랭크가 샤딩 전에 호스트 메모리에서 트랜스포머 전체를 구성해야 하므로, 대형 MoE 체크포인트에는 충분한 시스템 RAM이 필요하다는 실용적 제약도 있습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

이 논문에서 일반화 성능과 가장 밀접하게 연관된 설계 요소는 다음 세 가지입니다.

1. **MoE 아키텍처를 통한 용량-효율 트레이드오프 개선**: 토큰별로 전문 네트워크의 부분집합만 활성화하는 방식은 30B의 표현력을 유지하면서 추론 연산량은 3B 모델에 가깝게 유지합니다. 이는 다양한 로봇 형태(단일 암, 양팔, 인간형, 4족)와 과업 유형에 대해 서로 다른 전문가가 특화(specialize)될 잠재력을 제공하며, 이것이 내부 평가를 넘어선 RBench에서의 일반화로 이어졌을 가능성이 있습니다.

2. **데이터 다양성을 통한 물리적 상식의 내재화**: 조작, 내비게이션, 1인칭 시점을 포함하는 로봇 지향 영상으로 표준 인터넷 영상을 보강함으로써, 모델이 특정 로봇 플랫폼이나 특정 과업에 국한되지 않는 "행동과 세계 동역학에 대한 본질적 이해"를 학습하도록 유도합니다. 이는 특정 도메인에 과적합되지 않고 여러 embodiment로 전이될 수 있는 기반이 됩니다.

3. **다차원 보상 정렬을 통한 물리적 일반화**: 미학, 프롬프트 순응, 모션 일관성 외에도 물리적 합리성과 과업 완수에 대한 보상을 부여함으로써, 생성된 움직임이 표면적으로 그럴듯해 보이는 것이 아니라 실제 물리 법칙에 가깝게 유도됩니다. 이 덕분에 단순히 시각적으로 그럴듯한 움직임이 아니라 실제 물리 현상을 예측하는 Physics-IQ 벤치마크에서도 유의미한 점수(40.4)를 얻을 수 있었던 것으로 보입니다.

다만, 다중개체(0.444)와 추론(0.505) 카테고리에서의 낮은 점수는 일반화의 한계를 동시에 보여줍니다. 즉, 단일 개체·단일 과업 중심의 물리적 사실성은 잘 일반화되지만, 여러 객체 간 복잡한 상호작용이나 고차 추론이 필요한 시나리오로의 일반화는 아직 제한적입니다. 이는 데이터 분포(70,000시간의 embodied 데이터가 조작/내비게이션/1인칭 중심으로 구성되어 다중개체 협업 시나리오 비중이 낮을 가능성)와 MoE 라우팅이 복잡한 조합적 추론까지는 자연스럽게 특화되지 않을 수 있다는 구조적 한계를 함께 시사합니다.

---

## 4. 향후 연구에 미치는 영향과 고려할 점

### 영향
- **오픈소스 생태계 기여**: 이 모델은 최초의 대규모 오픈소스 MoE 비디오 파운데이션 모델로서, 디지털 창작과 물리적 행동을 잇는 선구적 시도로 평가되며, 이후 world-model 기반 로봇 정책 연구(VLA, world model 등)의 기반 백본으로 활용될 가능성이 높습니다.
- **"비디오 생성 → 로봇 정책"의 연결고리 강화**: 비디오 우선 세계 모델이 로봇 팀들이 정책과 결합하는 공유된 상상력 계층이 되어가고 있으며, 강력한 모델 대부분이 폐쇄형이라는 흐름 속에서, 이 논문은 오픈소스로 이 격차를 줄이는 역할을 합니다.
- **MoE의 비디오 도메인 확장**: 기존 MoE 연구가 주로 LLM(Switch Transformer, DeepSeek-V3 등)이나 VLM에 집중되었던 것에서, 이 논문은 비디오 diffusion/DiT 도메인에 MoE를 처음부터(from scratch) 대규모로 스케일업한 사례로서, DiT-MoE와 같은 선행 연구(shared expert routing, load balance loss 설계)의 실전 적용 사례를 확장합니다.

### 2020년 이후 관련 연구와의 비교 맥락
- LLM MoE 스케일링 연구(Switch Transformer, DeepSeek-MoE/V3)의 라우팅·불균형 해소 기법이 비디오 도메인에도 이식되는 추세이며, 유사 연구로 비디오 전문가에 sparse MoE를, 액션 전문가는 dense로 유지하며 DeepSeekMoE/V3의 sparse scaling 원칙을 따르는 Native Video-Action Pretraining 접근도 등장했습니다.
- 로봇 도메인에 특화된 MoE 연구로 스킬 지향 라우팅이 자율주행·로봇 조작에서 우수한 성능을 보이며, 추가 데이터나 사전학습 없이 모델 용량을 확장하는 스케일업 전략을 제안한 MoSE도 유사한 방향성을 공유합니다.
- world model 계열에서는 대규모 비주석 비디오 데이터셋으로 사전학습하여 더 넓은 범위의 embodied 과업으로 zero-shot 전이를 가능케 하는 방향으로 확장하려는 MoWM과 같은 후속 연구가 병행되고 있어, "비디오 사전학습 → embodied 일반화"라는 연구 흐름이 2025~2026년 사이 급속히 확산되고 있음을 확인할 수 있습니다.

### 향후 연구 시 고려할 점
1. **독립적 검증 필요**: 현재 공개된 성능 지표가 자체 보고치이며 독립적 검증이 아니라는 점을 고려하여, 후속 연구에서는 제3자 벤치마크나 실제 로봇 배치 환경에서의 재현 검증이 필요합니다.
2. **다중개체·고차 추론 보강**: 다중개체 협업과 시각적 추론 카테고리의 약점을 해결하기 위한 데이터 구성(복잡한 다중 에이전트 시나리오 확충) 및 아키텍처적 보완(예: 계층적 라우팅, 명시적 추론 모듈 결합)이 후속 과제로 남습니다.
3. **MoE 라우팅의 해석 가능성**: 어떤 전문가가 어떤 embodiment/과업에 특화되는지에 대한 라우팅 분석이 부족하면, 실제 물리적 일반화가 라우팅 특화에서 오는 것인지 데이터 규모에서 오는 것인지 구분하기 어렵습니다. 향후 연구는 라우팅 패턴에 대한 심층 분석을 포함해야 합니다.
4. **연산·메모리 인프라 고려**: 대형 MoE 체크포인트를 다루기 위한 시스템 RAM 요구사항 등 실제 배치 환경에서의 엔지니어링 비용도 함께 고려되어야 합니다.
5. **비디오 생성과 실제 행동 정책(VLA) 간의 격차**: 이 논문 자체는 비디오 생성 모델이며, 실제 로봇 정책으로의 전이(action decoding) 성능은 별도로 검증되어야 합니다. 액션 전문가에도 MoE 설계를 적용해 스케일링하는 LingBot-VLA 2.0류의 후속 연구와의 결합이 중요한 다음 단계로 보입니다.

---

## 참고 자료 (출처)
1. arXiv:2607.07675, "Scaling Mixture-of-Experts Video Pretraining for Embodied Intelligence" (arxiv.org/abs/2607.07675, arxiv.org/pdf/2607.07675)
2. alphaXiv, "Scaling Mixture-of-Experts Video Pretraining for Embodied Intelligence" (alphaxiv.org/abs/2607.07675)
3. GitHub - Robbyant/lingbot-video (github.com/robbyant/lingbot-video)
4. Hugging Face - robbyant/lingbot-video-moe-30b-a3b, robbyant/lingbot-video-dense-1.3b
5. AI Weekly, "Robbyant open-sources LingBot-Video, an MoE model for robotics" (aiweekly.co)
6. AI TLDR, "LingBot-Video — Apache-2.0 30B-A3B MoE video model" (ai-tldr.dev)
7. ComfyUI Wiki, "LingBot-Video: First Open-Source MoE Video Foundation" (comfyui-wiki.com)
8. AIFilms Studio Blog, "LingBot Video: The First Open Source MoE Video Generation" (studio.aifilms.ai)
9. arXiv:2607.08639, "Native Video-Action Pretraining for Generalizable Robot Control"
10. arXiv:2507.07818, "MoSE: Skill-by-Skill Mixture-of-Experts Learning for Embodied Autonomous Machines"
11. arXiv:2509.21797, "MoWM: Mixture-of-World-Models for Embodied Planning"
12. arXiv:2407.11633, "Scaling Diffusion Transformers to 16 Billion Parameters"
13. MarkTechPost, "Robbyant Releases LingBot-VLA 2.0" (marktechpost.com)

---

**주의**: 본문 중 명시적으로 표기한 수식들(flow-matching 손실, MoE 라우팅 손실, 보상 결합 함수)은 논문이 속한 DiT/MoE/flow-matching 계열 모델의 일반적·표준적 정식화를 재구성하여 제시한 것으로, arXiv 원문에서 정확히 동일한 수식 번호·표기를 직접 확인한 것은 아닙니다. 원문의 정확한 수식이 필요하다면 arXiv:2607.07675 원문(PDF)을 직접 확인하시기 바랍니다.
