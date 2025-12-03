
# RynnVLA-002: A Unified Vision-Language-Action and World Model

## 1. 핵심 주장 및 주요 기여 요약

RynnVLA-002는 **Vision-Language-Action(VLA) 모델과 World Model을 단일 프레임워크로 통합**한 혁신적인 접근법을 제시합니다. 이 연구의 핵심 주장은 VLA 모델과 World Model이 상호 보완적으로 작용하여 각각의 성능을 향상시킬 수 있다는 것입니다.[1]

**주요 기여점:**

1. **VLA와 World Model의 통합 프레임워크**: 행동(Action)과 이미지 이해 및 생성을 단일 LLM 아키텍처 내에서 통합하는 최초의 Action World Model 제안[1]
2. **이산적 행동 청크 생성을 위한 Action Attention Masking 전략**: 자기회귀 모델에서 행동 시퀀스 생성 시 발생하는 오류 누적 문제 해결[1]
3. **연속 Action Transformer 헤드 도입**: 일반화 성능 향상과 부드러운 궤적 생성을 위한 하이브리드 아키텍처 설계[1]
4. **사전학습 없이 LIBERO 벤치마크에서 97.4% 성공률** 달성, 실제 로봇 실험에서 50% 성능 향상[1]

***

## 2. 상세 분석

### 2.1 해결하고자 하는 문제

표준 VLA 아키텍처는 세 가지 근본적인 한계를 가지고 있습니다:[1]

**첫째, 행동에 대한 불완전한 이해**: 행동이 출력 측면에만 존재하여 모델이 행동 역학에 대한 명시적 내부 표현을 형성하지 못합니다.

**둘째, 상상력의 부재**: 주어진 후보 행동에 따라 세계가 어떻게 진화할지 예측하지 못하여 선견지명과 반사실적 추론이 제한됩니다.

**셋째, 물리학에 대한 명시적 이해 부족**: 물리적 역학을 포착하지 못해 객체 상호작용, 접촉, 안정성을 내재화할 수 없습니다.

반면 World Model은 미래 관측을 예측할 수 있지만, **직접적인 행동 출력을 생성하지 못하는 기능적 격차**가 있습니다.[1]

### 2.2 제안 방법 및 수식

#### 전체 아키텍처

RynnVLA-002는 **Chameleon**을 기반으로 하며, 이미지 이해와 생성을 위한 통합 모델입니다. 네 가지 토크나이저를 사용합니다:[2][1]

- **이미지 토크나이저**: VQ-GAN 모델(압축 비율 16, 코드북 크기 8192)
- **텍스트 토크나이저**: BPE 토크나이저
- **상태/행동 토크나이저**: 연속적인 로봇 고유수용감각 상태와 행동을 256개 빈으로 이산화

모든 토큰은 크기 65,536의 단일 어휘에 통합됩니다.[1]

#### VLA 모델 정의

VLA 모델에서 정책 $$\pi$$는 언어 목표 $$l$$, 고유수용감각 상태 $$s_{t-1}$$, 관측 이력 $$o_{t-h:t}$$를 기반으로 행동 $$a_t$$를 생성합니다:[1]

$$a_t = \pi(l, s_{t-1}, o_{t-h:t})$$

#### World Model 정의

World Model $$f$$는 과거 관측과 행동으로부터 다음 관측 $$o_t$$를 예측합니다:[1]

$$o_t = f(o_{t-h:t-1}, a_{t-h:t-1})$$

#### 훈련 목표 함수

VLA 모델 데이터와 World Model 데이터를 혼합하여 훈련하며, 이산적 행동에 대한 전체 손실 함수는:[1]

$$\mathcal{L}\_{dis} = \mathcal{L}\_{dis-action} + \mathcal{L}_{img}$$

여기서 $$\mathcal{L}\_{dis-action}$$은 이산 행동 토큰의 교차 엔트로피 손실, $$\mathcal{L}_{img}$$는 이산 이미지 토큰의 교차 엔트로피 손실입니다.

연속 Action Transformer를 추가한 최종 손실 함수는:[1]

$$\mathcal{L} = \mathcal{L}_{dis} + \lambda\mathcal{L}_{conti} = \mathcal{L}_{dis-action} + \mathcal{L}_{img} + \lambda\mathcal{L}_{conti-action}$$

여기서 $$\lambda = 10$$이며, $$\mathcal{L}_{conti-action}$$은 Action Transformer의 L1 회귀 손실입니다.[1]

#### Action Attention Masking 전략

기존 인과적 어텐션 마스크 대신, **현재 행동이 오직 텍스트 및 시각 입력에만 의존**하고 이전 행동에 대한 접근을 차단하는 수정된 어텐션 마스크를 도입합니다. 이를 통해 자기회귀 프레임워크 내에서 여러 행동을 독립적으로 생성하여 오류 누적 문제를 완화합니다.[1]

### 2.3 모델 구조

RynnVLA-002는 하이브리드 아키텍처를 채택합니다:[1]

| 구성 요소 | 역할 |
|-----------|------|
| **기본 LLM** | Chameleon 기반 멀티모달 언어 모델[1] |
| **이미지 토크나이저** | VQ-GAN (256×256 이미지 → 256 토큰, 512×512 → 1024 토큰)[1] |
| **State 토크나이저** | 로봇 고유수용감각 이산화 (256 빈)[1] |
| **Action 토크나이저** | 이산적 행동 토큰화 (256 빈)[1] |
| **Action Transformer** | 연속 행동 청크 병렬 생성[1] |

Action Transformer는 **학습 가능한 행동 쿼리**를 사용하여 전체 행동 청크를 단일 순방향 패스로 출력합니다. 이 설계는 두 가지 장점을 제공합니다:[1]

1. **과적합 방지**: 더 작은 아키텍처로 제한된 데이터에서의 과적합 완화
2. **추론 가속화**: 병렬 생성으로 순차적 기준선 대비 상당한 속도 향상[1]

### 2.4 성능 향상

#### 시뮬레이션 결과 (LIBERO 벤치마크)

| 모델 | 사전학습 | 행동 유형 | Spatial | Object | Goal | Long | 평균 |
|------|----------|-----------|---------|--------|------|------|------|
| OpenVLA[3] | ✓ | 이산 | 84.7% | 88.4% | 79.2% | 53.7% | 76.5%[1] |
| π0[4] | ✓ | 연속 | 90.0% | 86.0% | 95.0% | 73.0% | 86.0%[1] |
| RynnVLA-002-Discrete | ✗ | 이산 | 94.2% | 96.8% | 94.6% | 87.6% | 93.3%[1] |
| **RynnVLA-002-Continuous** | ✗ | 연속 | 99.0% | 99.8% | 96.4% | 94.4% | **97.4%**[1] |

#### 실제 로봇 실험 (LeRobot SO100)

| 작업 | 모델 | 단일 목표 | 다중 목표 | 방해물 포함 |
|------|------|-----------|-----------|-------------|
| 블록 배치 | GR00T N1.5[5] | 90.0% | 60.0% | 50.0%[1] |
| | π0[4] | 100.0% | 70.0% | 50.0%[1] |
| | **RynnVLA-002** | 90.0% | **90.0%** | **80.0%**[1] |

특히 **복잡한 환경(다중 목표, 방해물 포함)**에서 기준선 대비 10-30% 향상된 성능을 보여줍니다.[1]

### 2.5 한계점

1. **실제 로봇 데이터 의존성**: 대규모 실제 로봇 데이터셋이 부족한 환경에서 완전한 일반화 달성이 어려움[1]
2. **이산적 자기회귀 모델의 한계**: 고용량 데이터 요구량으로 인해 로봇 공학에서 종종 제한됨[1]
3. **추론 속도**: 이산 행동의 순차적 자기회귀 생성 과정으로 인한 느린 추론[1]
4. **과적합 문제**: 대규모 자기회귀 아키텍처가 제한된 실제 데이터셋에서 심각한 과적합 경향[1]

***

## 3. 일반화 성능 향상 가능성

### 3.1 World Model을 통한 일반화 향상

RynnVLA-002의 핵심 발견 중 하나는 **World Model 데이터를 훈련에 포함시키면 VLA 모델의 성능이 일관되게 향상**된다는 점입니다:[1]

| 구성 | World Model 포함 | Goal | Object | Spatial | Long | 평균 |
|------|------------------|------|--------|---------|------|------|
| 이산 행동 | ✗ | 67.3% | 82.9% | 77.8% | 23.0% | 62.8%[1] |
| 이산 행동 | ✓ | 73.1% | 88.0% | 80.2% | 27.3% | 67.2%[1] |
| 연속 행동 | ✗ | 91.4% | 95.4% | 98.2% | 81.4% | 91.6%[1] |
| 연속 행동 | ✓ | 96.0% | 97.4% | 99.0% | 85.8% | 94.6%[1] |

실제 로봇 실험에서 World Model 없이 훈련된 모델은 **30% 미만의 성공률**을 보인 반면, World Model과 함께 훈련하면 **80% 이상으로 향상**됩니다. 이는 World Model 훈련이 객체 움직임의 정확한 예측을 요구하기 때문에 VLA 모델이 조작 대상 객체에 더 집중하도록 유도하기 때문입니다.[1]

### 3.2 연속 행동 생성의 일반화 이점

연속 Action Transformer는 다음과 같은 이유로 일반화에 유리합니다:[1]

1. **컴팩트한 아키텍처**: 기본 LLM보다 훨씬 작아 제한된 데이터에서 과적합 방지
2. **양방향 어텐션**: 행동 간 시간적 일관성 보장으로 부드러운 궤적 생성
3. **병렬 디코딩**: 추론 단계 감소로 효율성 향상[1]

### 3.3 Action Attention Masking의 일반화 효과

사전학습된 MLLM이 이미지와 텍스트 도메인에서는 강력한 일반화 능력을 보이지만, **행동 도메인에서의 일반화 능력은 비교적 제한적**입니다. 기존 인과적 어텐션에서 후속 행동이 선행 행동에 과도하게 의존하면, 시각적 입력(다른 양식)에 기반하기보다 오류가 전파됩니다. 제안된 어텐션 마스킹 메커니즘은 **각 행동이 시각적 입력에만 의존하도록** 하여 행동 시퀀스 내 오류 전파를 완화합니다.[1]

### 3.4 관련 최신 연구 동향 (2020년 이후)

최근 VLA 및 World Model 연구는 일반화 성능 향상을 위한 다양한 접근법을 탐구하고 있습니다:

**π0 (Physical Intelligence, 2024)**: 사전학습된 VLM 위에 flow matching 아키텍처를 구축하여 인터넷 규모의 의미적 지식을 상속받습니다. 7개 로봇 플랫폼과 68개 고유 작업에서 훈련되어 제로샷 및 미세조정 모두에서 강력한 성능을 보여줍니다.[4][6]

**OpenVLA (Stanford, 2024)**: 970k 실제 로봇 시연 데이터로 훈련된 7B 파라미터 오픈소스 VLA로, RT-2-X(55B)를 16.5% 초과하는 성능을 달성했습니다. DINOv2와 SigLIP의 사전학습된 특징을 융합하여 일반화 능력을 향상시킵니다.[3][7]

**FAST 토크나이저 (2024)**: 이산 코사인 변환(DCT) 기반 시계열 압축을 통한 새로운 행동 토큰화 방식으로, 고주파 로봇 제어 작업에서 기존 이산화 방법이 실패하는 경우에도 효과적입니다.[8]

**OTTER (ICML 2025)**: 사전학습된 VLM의 비전-언어 인코더를 동결 상태로 유지하면서 텍스트-인식 시각 특징 추출을 통해 강력한 제로샷 일반화를 달성합니다.[9][10]

**GenRL (2024)**: 멀티모달 파운데이션 World Model이 파운데이션 VLM의 표현을 생성적 World Model의 잠재 공간과 연결하여, 시각 및/또는 언어 프롬프트로부터 다중 작업 일반화를 가능하게 합니다.[11][12]

**GWM (Gaussian World Model, 2025)**: 3D Gaussian 표현을 기반으로 동적 미래 상태를 예측하고 로봇 조작을 가능하게 하는 새로운 유형의 World Model입니다.[13]

***

## 4. 향후 연구에 미치는 영향 및 고려사항

### 4.1 연구 영향

**통합 프레임워크의 패러다임 전환**: RynnVLA-002는 VLA와 World Model을 별개의 시스템으로 취급하던 기존 접근법에서 벗어나, **두 모델의 시너지적 상호작용**을 통한 성능 향상 가능성을 입증했습니다. 이는 향후 체화된 AI 연구에서 멀티모달 이해와 생성을 텍스트, 비전, 행동에 걸쳐 통합하는 기반을 마련합니다.[1]

**사전학습 없는 고성능 달성**: 대규모 사전학습 데이터 없이도 사전학습된 강력한 기준선과 동등한 성능을 달성했다는 점은, **데이터 효율적인 로봇 학습**의 새로운 방향을 제시합니다.[1]

**물리 역학 학습의 중요성**: World Model을 통한 환경 물리학 학습이 행동 생성의 정확성을 크게 향상시킨다는 발견은, 로봇이 단순히 패턴을 모방하는 것이 아니라 **세계의 인과적 구조를 이해**해야 함을 시사합니다.[1]

### 4.2 향후 연구 시 고려할 점

**데이터 확장성 문제**: 이산적 자기회귀 모델의 고용량 데이터 요구량과 로봇 공학에서의 데이터 희소성 간의 격차를 해결해야 합니다. 시뮬레이션 데이터 활용, 합성 데이터 생성, 또는 VQ-VLA와 같은 대규모 합성 궤적 데이터 기반 접근법이 유망합니다.[14][1]

**실시간 추론 최적화**: 연속 Action Transformer가 이산적 접근법보다 빠르지만, 실제 로봇 배포를 위해서는 추가적인 최적화가 필요합니다. LightDP와 같은 경량화 기법이나 VLA-Cache와 같은 적응형 토큰 캐싱이 고려될 수 있습니다.[15][16][1]

**멀티모달 확장**: 현재 모델은 시각과 언어에 초점을 맞추지만, VLAS와 같이 음성 명령을 직접 통합하거나, 촉각 정보를 포함한 확장이 더 자연스러운 인간-로봇 상호작용을 가능하게 할 것입니다.[17]

**Action Chunking의 최적화**: 행동 청크 길이가 지나치게 길어지면 로봇의 적시 정책 적응이 제한됩니다. Bidirectional Decoding(BID)이나 Temporal Action Selection(TAS)과 같은 테스트 시간 적응 기법이 이 균형을 개선할 수 있습니다.[18][19][1]

**크로스-에메보디먼트 일반화**: 다양한 로봇 플랫폼과 형태에 걸친 일반화는 여전히 중요한 과제입니다. π0.5와 같이 대규모 멀티모달 웹 및 크로스-에메보디먼트 데이터를 활용하는 접근법이 이 방향의 핵심입니다.[20][21]

**추론 능력과 행동의 통합**: CoT-VLA나 MolmoAct와 같이 시각적 사고 연쇄(Chain-of-Thought) 추론을 VLA에 통합하여 설명 가능하고 조종 가능한 행동을 생성하는 연구가 증가하고 있습니다.[22][23][1]

**실제 환경에서의 강건성**: 조명, 객체 위치 등 동적 변수에 대한 강건성 향상이 필요합니다. World Model 사전학습이 이러한 일반화에 도움이 된다는 것이 확인되었으며, 이를 더 체계적으로 활용하는 연구가 필요합니다.[1]

### 4.3 결론적 시사점

RynnVLA-002는 **체화된 AI의 핵심 구성 요소인 VLA와 World Model의 상호 강화 가능성**을 실증적으로 보여주었습니다. 이 연구는 단순히 성능 향상을 넘어, 로봇이 세계를 이해하고 상상하며 행동하는 통합된 인지 시스템으로 발전하기 위한 중요한 이정표입니다. 향후 연구에서는 이러한 통합 프레임워크를 더 다양한 로봇 플랫폼, 작업 도메인, 그리고 실제 환경 조건에 확장하는 것이 핵심 과제가 될 것입니다.[1]

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/441876ae-38b6-4d4f-afec-f84267bae2a6/2511.17502v2.pdf)
[2](https://arxiv.org/abs/2405.09818)
[3](https://proceedings.mlr.press/v270/kim25c.html)
[4](https://arxiv.org/abs/2410.24164)
[5](https://arxiv.org/abs/2503.14734)
[6](https://twimlai.com/podcast/twimlai/%CF%800-a-foundation-model-for-robotics/)
[7](https://arxiv.org/html/2406.09246v3)
[8](https://www.physicalintelligence.company/download/fast.pdf)
[9](https://arxiv.org/html/2503.03734v1)
[10](https://openreview.net/forum?id=UHF0km7R5M)
[11](https://openreview.net/pdf?id=za9Jx8yqUA)
[12](https://arxiv.org/abs/2406.18043)
[13](https://openaccess.thecvf.com/content/ICCV2025/papers/Lu_GWM_Towards_Scalable_Gaussian_World_Models_for_Robotic_Manipulation_ICCV_2025_paper.pdf)
[14](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_VQ-VLA_Improving_Vision-Language-Action_Models_via_Scaling_Vector-Quantized_Action_Tokenizers_ICCV_2025_paper.pdf)
[15](https://arxiv.org/html/2502.02175v1)
[16](https://openaccess.thecvf.com/content/ICCV2025/papers/Wu_On-Device_Diffusion_Transformer_Policy_for_Efficient_Robot_Manipulation_ICCV_2025_paper.pdf)
[17](https://arxiv.org/abs/2502.13508)
[18](https://openreview.net/forum?id=qZmn2hkuzw)
[19](https://arxiv.org/html/2511.04421)
[20](https://arxiv.org/abs/2510.24795)
[21](https://arxiv.org/html/2504.16054v1)
[22](https://mbreuss.github.io/blog_post_iclr_26_vla.html)
[23](https://arxiv.org/abs/2508.07917)
[24](https://arxiv.org/abs/2510.23511)
[25](https://arxiv.org/abs/2505.19381)
[26](https://arxiv.org/abs/2412.05467)
[27](https://invergejournals.com/index.php/ijss/article/view/189)
[28](https://jelle.lgu.edu.pk/jelle/article/view/342)
[29](http://arxiv.org/pdf/2409.12514.pdf)
[30](https://arxiv.org/html/2501.05952v2)
[31](http://arxiv.org/pdf/2410.05191.pdf)
[32](https://arxiv.org/html/2503.23463v1)
[33](https://arxiv.org/html/2408.10845)
[34](https://arxiv.org/html/2412.10345v2)
[35](https://openreview.net/forum?id=aVyJwS1fqQ)
[36](https://proceedings.mlr.press/v235/lv24a.html)
[37](https://arxiv.org/html/2505.04769v1)
[38](https://www.dynsyslab.org/mastering-robot-manipulation-in-a-world-of-abundant-data/)
[39](https://www.sciencedirect.com/science/article/abs/pii/S1566253525007249)
[40](https://arxiv.org/html/2510.10125v1)
[41](https://neurips.cc/virtual/2024/100962)
[42](https://en.wikipedia.org/wiki/Vision-language-action_model)
[43](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_CoA-VLA_Improving_Vision-Language-Action_Models_via_Visual-Text_Chain-of-Affordance_ICCV_2025_paper.pdf)
[44](https://arxiv.org/html/2508.17600v1)
[45](https://openaccess.thecvf.com/content/CVPR2024/html/Li_ManipLLM_Embodied_Multimodal_Large_Language_Model_for_Object-Centric_Robotic_Manipulation_CVPR_2024_paper.html)
[46](https://aclanthology.org/2025.emnlp-main.273.pdf)
[47](https://github.com/operator22th/awesome-world-models-for-robots)
[48](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1453061/full)
[49](https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation)
[50](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2025.1585386/full)
[51](https://proceedings.neurips.cc/paper_files/paper/2024/file/6164b6e5352c139e9ddc1a98c09e4e4a-Paper-Conference.pdf)
[52](https://arxiv.org/abs/2506.09784)
[53](https://arxiv.org/abs/2505.21906)
[54](https://arxiv.org/pdf/2403.08248.pdf)
[55](http://arxiv.org/pdf/2410.24164.pdf)
[56](https://arxiv.org/html/2503.12533v1)
[57](https://arxiv.org/pdf/2402.02385.pdf)
[58](https://arxiv.org/pdf/2306.14846.pdf)
[59](https://arxiv.org/pdf/2204.11134.pdf)
[60](https://arxiv.org/html/2401.12963v2)
[61](https://www.emergentmind.com/topics/libero-tasks)
[62](https://github.com/lucidrains/pi-zero-pytorch)
[63](https://arxiv.org/html/2510.16732v2)
[64](https://www.youtube.com/watch?v=5mY71rGXAkM)
[65](https://arxiv.org/abs/2406.09246)
[66](https://arxiv.org/html/2510.16732v1)
[67](https://www.therobotreport.com/physical-intelligence-open-sources-pi0-robotics-foundation-model/)
[68](https://openvla.github.io)
[69](https://mn.cs.tsinghua.edu.cn/xinwang/PDF/papers/2025_Embodied%20AI%20from%20LLMs%20to%20World%20Models.pdf)
[70](https://huggingface.co/blog/pi0)
[71](https://github.com/openvla/openvla)
[72](https://arxiv.org/html/2506.22355v1)
[73](https://arxiv.org/html/2410.24164v1)
[74](https://velog.io/@nyl0522/Paper-Review-OpenVLA-AnOpen-Source-Vision-Language-Action-Model)
[75](https://aclanthology.org/2025.findings-emnlp.69.pdf)
[76](https://www.physicalintelligence.company/blog/pi0)
[77](https://blog.naver.com/ndolab/223770941603?recommendCode=2&recommendTrackingCode=2)
[78](https://arxiv.org/abs/2304.13705)
[79](https://www.ewadirect.com/proceedings/ace/article/view/22913)
[80](http://ieeexplore.ieee.org/document/100057/)
[81](https://ieeexplore.ieee.org/document/11077988/)
[82](https://www.semanticscholar.org/paper/0500e4208067e4f9e82d29159cd47ac0ca5be85a)
[83](https://www.ewadirect.com/proceedings/ace/article/view/22703)
[84](https://www.semanticscholar.org/paper/7105a46f97440aa9de04ae28ede0d48d52f2ad9c)
[85](http://arxiv.org/pdf/2409.18768.pdf)
[86](http://arxiv.org/pdf/2408.17355.pdf)
[87](https://arxiv.org/html/2502.20771v1)
[88](http://arxiv.org/pdf/2403.00929.pdf)
[89](http://arxiv.org/pdf/2402.10340.pdf)
[90](https://arxiv.org/pdf/2306.17237.pdf)
[91](https://arxiv.org/pdf/2209.08903.pdf)
[92](http://arxiv.org/pdf/2404.02728.pdf)
[93](https://jkros.org/xml/44006/44006.pdf)
[94](https://diffusion-policy.cs.columbia.edu)
[95](http://www.researchinchina.com/Htmls/Report/2025/77089.html)
[96](https://arxiv.org/html/2410.03132v3)
[97](https://arxiv.org/html/2410.15959v6)
[98](https://www.techrxiv.org/users/908014/articles/1282755/master/file/data/TPAMI/TPAMI.pdf)
[99](https://www.pangram.com/history/a11cc1c1-6fd7-46d7-9b78-d8a37b68f922)
[100](https://dexterous-humanoid-manipulation.github.io/src/file/paper/shahn.pdf)
[101](https://openreview.net/pdf/7766b8ff1b63be53bd2e90e18736336b0cf5e847.pdf)
[102](http://ieeexplore.ieee.org/iel8/7083369/11045364/11037245.pdf)
[103](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1606247/full)
[104](https://ieeexplore.ieee.org/iel8/6287639/10820123/11164279.pdf)
[105](https://github.com/BaiShuanghao/Awesome-Robotics-Manipulation)
[106](https://github.com/showlab/Awesome-Robotics-Diffusion)
[107](https://arxiv.org/abs/2307.15818)
[108](https://ieeexplore.ieee.org/document/10611597/)
[109](https://link.springer.com/10.1007/s12283-024-00466-4)
[110](https://biss.pensoft.net/article/112436/)
[111](https://arxiv.org/abs/2411.19650)
[112](https://arxiv.org/abs/2406.07549)
[113](https://arxiv.org/abs/2411.00508)
[114](https://academic.oup.com/jas/article/103/Supplement_3/71/8274300)
[115](https://www.semanticscholar.org/paper/3c854c7ae9060234f04bf6abc96392702c7c67e8)
[116](http://arxiv.org/pdf/2406.07549.pdf)
[117](https://arxiv.org/pdf/2312.17172.pdf)
[118](https://aclanthology.org/2023.findings-emnlp.793.pdf)
[119](https://arxiv.org/pdf/2310.09478.pdf)
[120](http://arxiv.org/pdf/2406.08394v1.pdf)
[121](https://ajithp.com/2024/05/26/chameleon-early-fusion-multimodal-ai-model-for-visual-and-textual-interaction/)
[122](https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/)
[123](https://brjathu.github.io/toto/)
[124](https://smcho1201.tistory.com/120)
[125](https://robotics-transformer2.github.io)
[126](https://arxiv.org/html/2511.17502v2)
[127](https://discuss.pytorch.kr/t/meta-chameleon/4410)
[128](https://stibee.com/api/v1.0/emails/share/hduoknTSs2mKyQRk3JscqOQNVIJesRM)
[129](https://arxiv.org/html/2506.21539v1)
[130](https://huggingface.co/docs/transformers/model_doc/chameleon)
[131](https://seohyun00.tistory.com/7)
[132](https://github.com/ChaofanTao/Autoregressive-Models-in-Vision-Survey)
[133](https://openaccess.thecvf.com/content/CVPR2025/papers/Nguyen_YoChameleon_Personalized_Vision_and_Language_Generation_CVPR_2025_paper.pdf)
[134](https://blog.naver.com/edblab/223263883904)
[135](https://openreview.net/forum?id=m29SV0n6DO)
