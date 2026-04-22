
# Drag Your Noise: Interactive Point-based Editing via Diffusion Semantic Propagation

> **출처 및 참고자료**
> - Liu, H., Xu, C., Yang, Y., Zeng, L., He, S. (2024). *Drag Your Noise: Interactive Point-based Editing via Diffusion Semantic Propagation*. CVPR 2024, pp. 6743–6752. ([arXiv:2404.01050](https://arxiv.org/abs/2404.01050))
> - CVPR 2024 Open Access: [https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Drag_Your_Noise...](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Drag_Your_Noise_Interactive_Point-based_Editing_via_Diffusion_Semantic_Propagation_CVPR_2024_paper.html)
> - GitHub (공식 코드): [https://github.com/haofengl/DragNoise](https://github.com/haofengl/DragNoise)
> - arXiv HTML 전문: [https://arxiv.org/html/2404.01050v1](https://arxiv.org/html/2404.01050v1)
> - DragDiffusion (비교 논문): [arXiv:2306.14435](https://arxiv.org/abs/2306.14435)
> - AdaptiveDrag: [arXiv:2410.12696](https://arxiv.org/html/2410.12696v1)
> - FastDrag (NeurIPS 2024): [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2024/file/87ccd80753f787e81d4c8da135385b4e-Paper-Conference.pdf)
> - DirectDrag: [arXiv:2512.03981](https://arxiv.org/html/2512.03981)
> - awesome-drag-editing (GitHub): [https://github.com/frakw/awesome-drag-editing](https://github.com/frakw/awesome-drag-editing)

---

## 1. 핵심 주장 및 주요 기여 (간결 요약)

포인트 기반 인터랙티브 편집은 기존 생성 모델의 제어성을 보완하는 필수적인 도구입니다.

DragNoise는 latent map의 재추적(retracing) 없이 견고하고 가속화된 편집을 제공하며, 그 핵심 원리는 각 U-Net의 예측된 노이즈 출력을 시맨틱 에디터로 활용하는 것입니다.

이 접근법은 두 가지 핵심 관찰에 기반합니다: 첫째, U-Net의 보틀넥(bottleneck) 피처가 인터랙티브 편집에 이상적인 풍부한 시맨틱 정보를 내재하고 있다는 것; 둘째, 디노이징 초기 단계에서 확립된 고수준 시맨틱은 이후 단계에서 최소한의 변화만 보인다는 것입니다. 이러한 통찰을 활용하여 DragNoise는 단일 디노이징 스텝에서 디퓨전 시맨틱을 편집하고 변경 사항을 효율적으로 전파합니다.

### 핵심 기여 요약

| 기여 | 내용 |
|---|---|
| 새로운 패러다임 | 노이즈 맵을 시맨틱 에디터로 사용 |
| 단일 스텝 최적화 | 기존 다단계 반복 최적화 → 1 스텝으로 단축 |
| 시맨틱 전파(Propagation) | 최적화된 보틀넥 피처를 이후 모든 timestep에 전파 |
| 성능 향상 | DragDiffusion 대비 최적화 시간 50% 이상 단축 |

---

## 2. 상세 설명

### 2-1. 해결하고자 하는 문제

DragDiffusion과 같은 동시기 연구는 사용자 입력에 응답하여 디퓨전 latent map을 업데이트하는 방식으로, 이는 전역적인 latent map 변경을 초래합니다. 그 결과 원본 콘텐츠의 부정확한 보존과 gradient vanishing으로 인한 편집 실패를 야기합니다.

기존 방법들은 드래그 기반 이미지 편집을 위해 일반적으로 latent 시맨틱 최적화를 위한 n-step 반복을 채택하여 시간이 많이 걸리고 실용적인 응용을 제한합니다.

포인트 기반 이미지 편집 분야에서 DragGAN은 GAN을 활용한 중요한 이정표를 세웠으나, GAN의 고유한 제약으로 인해 고품질 편집 결과를 달성하는 데 한계가 있었습니다.

정리하면 기존 방법들의 문제점은 다음과 같습니다:
- **GAN 기반 방법 (DragGAN, FreeDrag)**: 생성 능력, 인버전 효율성, latent 코드의 시맨틱 다양성 제약
- **DragDiffusion**: 전역 latent map 변경으로 인한 콘텐츠 보존 부정확 및 gradient vanishing
- **공통 문제**: 다단계(n-step) 반복 최적화로 인한 높은 계산 비용

---

### 2-2. 제안하는 방법 (수식 포함)

#### (A) Latent Diffusion Model (LDM) 기초

LDM의 순방향 디퓨전 프로세스는 다음과 같이 정의됩니다:

$$q(\mathbf{z}_t | \mathbf{z}_0) = \mathcal{N}(\mathbf{z}_t; \sqrt{\bar{\alpha}_t}\mathbf{z}_0, (1-\bar{\alpha}_t)\mathbf{I})$$

역방향 디노이징 프로세스에서 U-Net $\epsilon_\theta$는 timestep $t$에서의 노이즈를 예측합니다:

$$\hat{\epsilon}_t = \epsilon_\theta(\mathbf{z}_t, t, c)$$

여기서 $c$는 텍스트 조건, $\mathbf{z}_t$는 timestep $t$에서의 latent입니다.

#### (B) Diffusion Semantic Optimization (디퓨전 시맨틱 최적화)

편집 프로세스는 고수준 시맨틱이 충분히 학습된 timestep(예: $t=35$)에서 시작됩니다. 이 단계에서 디퓨전 시맨틱 최적화는 사용자 편집을 반영하기 위해 U-Net의 보틀넥 피처에 대해 수행됩니다. 최적화된 보틀넥 피처는 의도된 드래그 효과를 학습하고 대응하는 조작 노이즈를 생성합니다.

Motion Supervision Loss (DragGAN/DragDiffusion에서 계승된 기본 개념):

$$\mathcal{L}_{motion} = \sum_{i} \left\| F(\mathbf{p}_i) - F(\mathbf{q}_i) \right\|_2$$

여기서:
- $\mathbf{p}_i$: handle point (드래그 시작점)
- $\mathbf{q}_i$: target point (드래그 목표점)
- $F(\cdot)$: U-Net **보틀넥 피처** 추출 함수

DragNoise의 핵심 차별점은 $F(\cdot)$를 latent map이 아닌 **보틀넥 피처**에 직접 적용한다는 점입니다.

#### (C) Diffusion Semantic Propagation (디퓨전 시맨틱 전파)

최적화된 보틀넥 피처는 목표 시맨틱을 포함하고 있으므로, 대응하는 보틀넥 피처를 대체함으로써 이후 모든 timestep에 전파되어 불필요한 피처 최적화를 피합니다. 이 대체는 안정적이고 효율적인 방식으로 조작 효과를 크게 증폭시킵니다.

전파 수식은 다음과 같이 표현할 수 있습니다:

```math
F^*_{t'} \leftarrow F^*_{t_{edit}}, \quad \forall t' < t_{edit}
```

여기서:
- $t_{edit}$: 편집이 수행되는 초기 timestep (예: $t=35$)
- $F^*\_{t_{edit}}$: 최적화된 보틀넥 피처
- $F^*_{t'}$: 이후 timestep $t'$에 전파되는 피처

전체 파이프라인은 다음과 같습니다:

```math
\mathbf{z}_0 = \text{DDIM\_Denoise}\left(\mathbf{z}_T,\; \{F^*_{t_{edit}} \rightarrow \text{propagate to all } t' < t_{edit}\}\right)
```

---

### 2-3. 모델 구조

DragNoise는 인터랙티브 포인트 기반 편집을 위해 설계되었으며, 디퓨전 모델과 LDM의 기본 사항을 기반으로 합니다. DragNoise는 예측된 노이즈를 조작하는 데 초점을 맞추어, 디퓨전 시맨틱 최적화와 디퓨전 시맨틱 전파를 포함합니다.

전체 구조는 아래와 같이 세 단계로 구성됩니다:

```
[입력 이미지]
     ↓
① LoRA Fine-tuning (이미지 identity 보존)
     ↓
② DDIM Inversion → z_T 획득
     ↓
③ t = t_edit (예: 35)에서
   U-Net Bottleneck Feature 최적화 (Motion Supervision Loss)
     ↓
④ 최적화된 Bottleneck Feature를 이후 모든 timestep에 전파
     ↓
⑤ DDIM Denoising → 최종 편집 이미지 z_0 생성
```

흥미롭게도 timestep 35의 보틀넥 피처를 대체하면 전체적인 구조가 보존되며, 이 초기 timestep의 시맨틱을 이후 단계에 전파해도 재구성 품질이 저하되지 않습니다. 이 발견은 보틀넥 피처가 효율적인 편집에 특히 적합한 최적의 디퓨전 시맨틱 표현임을 결론짓게 합니다.

초기 timestep에서 효과적으로 훈련될 수 있기 때문에, 보틀넥 피처를 조작하면 이후 디노이징 단계로의 부드러운 전파가 가능하여 완전한 디퓨전 시맨틱의 무결성이 유지됩니다. 또한 짧은 최적화 경로로 인해 gradient vanishing 문제도 효율적으로 방지됩니다.

---

### 2-4. 성능 향상

DragBench 데이터셋과 Mean Distance(MD), Image Fidelity(IF)를 정량적 분석에 사용합니다. IF는 편집 결과의 충실도 지표이며, MD는 드래그 효과의 정확도를 반영합니다.

DragBench 데이터셋에서 네 가지 방법의 성능을 비교한 결과, 디퓨전 기반 편집 방법이 GAN 기반 방법보다 일반적으로 우수합니다. 특히 DragNoise는 드래그 정확도와 이미지 충실도 측면에서 모든 기존 방법을 능가합니다.

비교 실험에서 DragNoise는 우수한 제어력과 시맨틱 보존을 달성하며, DragDiffusion 대비 최적화 시간을 50% 이상 단축합니다.

| 지표 | DragNoise 성능 |
|---|---|
| MD (Mean Distance) | 모든 비교 방법 중 최고 (낮을수록 좋음) |
| IF (Image Fidelity / 1-LPIPS) | 모든 비교 방법 중 최고 (높을수록 좋음) |
| 최적화 시간 | DragDiffusion 대비 **50% 이상 단축** |

---

### 2-5. 한계점

후속 연구들이 지적하는 DragNoise의 한계는 다음과 같습니다:

DragDiffusion과 DragNoise는 동물 얼굴 회전 등 복잡한 변환 작업에서 실패하는 경우가 있습니다.

DragDiffusion과 DragNoise는 소파 등 객체의 일부를 늘리는(stretching) 작업에서도 실패하는 경향이 있습니다.

DragNoise와 EasyDrag는 꽃의 상단을 높은 위치로 이동시키긴 하지만, 자연스러운 성장 패턴을 유지하지 못하고 handle point 주변의 영역만 변경하는 문제가 있습니다.

DragDiffusion과 DragNoise는 대형 차량 회전 등 대규모 조작 시 바퀴의 위치를 잘못 배치하는 오류를 보입니다.

DragDiffusion, FreeDrag, DragNoise와 같은 방법들은 정확하고 고품질의 결과를 산출하지만, 시맨틱 정보를 충분히 활용하지 못하여 덜 이상적인 결과를 낳는 경우가 있습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

DragGAN은 GAN에 의존하기 때문에, 사전 훈련된 GAN 모델의 용량에 의해 일반화 성능이 제한되었습니다. DragDiffusion은 이 편집 프레임워크를 디퓨전 모델로 확장하여, 대규모 사전 훈련된 디퓨전 모델을 활용함으로써 실제 이미지와 디퓨전 생성 이미지 모두에서 인터랙티브 포인트 기반 편집의 적용 가능성을 크게 향상시켰습니다.

DragNoise의 일반화 성능 향상 요인을 정리하면 다음과 같습니다:

### (1) Stable Diffusion 기반의 광범위한 도메인 커버

디퓨전 모델에 내재된 제한적인 제어성은 이미지 조작에서 인터랙티브 편집의 필요성을 강조합니다. 텍스트 가이드 편집, 스트로크 기반 편집, 예제 기반 방법 등 다양한 인터랙티브 접근법들이 발전해왔습니다. 드래그 앤 드롭 방식의 제어점 조작은 실제 응용 분야에서 직관적이고 효율적인 접근법으로 부상하고 있습니다.

### (2) LoRA 파인튜닝을 통한 Identity 보존

DragNoise는 입력 이미지에 대해 LoRA를 훈련시키고, 마스크로 편집 가능 영역을 지정하는 방식을 사용합니다. LoRA 파인튜닝은 특정 이미지의 identity를 보존하면서도 대규모 사전훈련 모델의 일반화 능력을 유지합니다.

### (3) 단일 타임스텝 최적화의 일반화 이점

timestep 35의 보틀넥 피처를 대체하면 전체 구조가 보존되며, 이 초기 timestep 시맨틱을 이후 단계에 전파해도 재구성 품질이 저하되지 않습니다. 보틀넥 피처는 특히 효율적인 편집에 적합한 최적의 디퓨전 시맨틱 표현입니다. 초기 타임스텝에서 훈련될 수 있기 때문에 보틀넥 피처 조작이 이후 디노이징 단계로 부드럽게 전파될 수 있습니다.

이는 특정 도메인에 과적합하지 않고도 다양한 이미지 유형에 적용 가능함을 의미합니다.

### (4) Gradient Vanishing 방지

짧은 최적화 경로로 인해 gradient vanishing 문제가 효율적으로 방지됩니다. 이는 곧 다양한 이미지에서 안정적으로 편집이 수행되므로 일반화 성능에 직접적으로 기여합니다.

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 방법론 계보 및 비교

| 논문 | 발표 | 기반 | 편집 방식 | 주요 특징 |
|---|---|---|---|---|
| **DragGAN** | SIGGRAPH 2023 | GAN | Motion Supervision + Point Tracking | 최초 포인트 기반 드래그 편집 |
| **DragDiffusion** | CVPR 2024 | Diffusion (LDM) | Latent Map 최적화 | LoRA + DDIM Inversion, 일반화↑ |
| **DragonDiffusion** | ICLR 2024 | Diffusion | Classifier Guidance + Feature Correspondence | 에너지 함수 기반 편집 |
| **FreeDrag** | CVPR 2024 | GAN/Diffusion | Template Feature + Line Search | Adaptive Updating, 안정적 드래그 |
| **DragNoise** | CVPR 2024 | Diffusion (LDM) | Bottleneck Feature + Semantic Propagation | 단일 스텝 최적화, 50% 시간↓ |
| **GoodDrag** | 2024 | Diffusion | Alternating Drag-Denoising (AlDD) | 오차 누적 방지 |
| **FastDrag** | NeurIPS 2024 | Diffusion | Latent Warpage Function (LWF) | 1-step 편집, 최고 속도 |
| **AdaptiveDrag** | 2024 | Diffusion | Semantic-Driven Optimization + SAM2 | 마스크 자동 생성, 미세 시맨틱 제어 |
| **DirectDrag** | 2025 | Diffusion | Readout-Guided Feature Alignment | Mask-free, Prompt-free |

드래그 기반 이미지 편집 방법은 사용자가 특정 포인트를 목표 위치로 드래그하여 이미지 구조를 제어할 수 있게 합니다. DragGAN은 GAN 기반의 latent code 최적화 프레임워크를 처음 제안했지만 실제 입력에 대한 일반화에 어려움을 겪었습니다. DragDiffusion과 DragonDiffusion은 이 패러다임을 디퓨전 모델로 확장하여 구조적 조작과 시맨틱 제어 가능성을 향상시켰습니다.

이후 방법들은 편집 품질과 견고성 향상을 목표로 했습니다. DragNoise는 U-Net 보틀넥 피처를 최적화하여 비용을 줄였습니다. GoodDrag는 드래그와 디노이징을 교대하여 오차 누적을 방지했습니다. GDrag는 훈련 없이 원자적 조작과 밀집 궤적으로 의도와 콘텐츠 모호성을 해결했습니다. FlowDrag는 3D 메시 가이드 플로우 필드로 기하학적 일관성을 개선했습니다.

이러한 방법들은 latent 최적화를 위해 n-step 반복이 필요하여 시간 소모가 크게 증가합니다.

FastDrag는 latent warpage function(LWF)을 핵심으로 하는 새로운 1-step 드래그 기반 이미지 편집 방법을 도입했습니다. LWF는 늘어난 재료의 동작을 시뮬레이션하여 latent 공간에서 개별 픽셀의 위치를 조정합니다. 이 혁신으로 1-step latent 시맨틱 최적화를 달성하여 편집 속도를 크게 향상시킵니다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5-1. 앞으로의 연구에 미치는 영향

**① 노이즈 맵의 시맨틱 에디터 활용 패러다임 확산**

DragNoise는 디퓨전 시맨틱 전파를 활용하는 새로운 인터랙티브 포인트 기반 이미지 편집 방법을 제안했으며, 역방향 디퓨전 프로세스에서 예측된 노이즈를 시맨틱 에디터로 간주합니다. 이 발견은 노이즈 예측값이 단순한 중간 결과물이 아니라 편집 가능한 풍부한 시맨틱 표현임을 입증하였으며, 이후 연구들의 피처 활용 방향에 영향을 주었습니다.

**② 단일 스텝 최적화 트렌드의 가속화**

GoodDrag는 드래그와 디노이징 작업을 교대하는 AlDD 프레임워크를 도입하여 축적된 perturbation과 왜곡 문제를 효과적으로 개선하고, 시작점의 원본 피처를 유지하는 정보 보존 motion supervision을 제안했습니다. 이는 DragNoise의 단일 스텝 접근법으로부터 영감을 받은 방향입니다.

**③ 마스크 자동화 연구 촉진**

많은 기존 방법들은 정확하고 시맨틱적으로 일관된 결과를 위해 편집 가능 영역 마스크 및 텍스트 프롬프트와 같은 추가 정보를 사용자가 제공할 것을 요구합니다. DragNoise 이후 이를 자동화하는 연구(AdaptiveDrag, DirectDrag 등)가 활발해졌습니다.

**④ 3D/비디오 편집으로의 확장**

SyncNoise는 2D 디퓨전 모델로 여러 뷰를 동기적으로 편집하면서 다중 뷰 노이즈 예측을 기하학적으로 일관되게 강제하여 시맨틱 구조와 저주파 외관 모두의 전역적 일관성을 보장합니다. 로컬 일관성을 더욱 향상시키기 위해 앵커 뷰를 설정하고 크로스-뷰 역투영을 통해 인접 프레임에 전파합니다.

### 5-2. 향후 연구 시 고려할 점

**① 대규모 조작의 한계 극복**
DragDiffusion과 DragNoise는 동물 얼굴 회전과 같은 작업에서 실패합니다. 대규모 회전, 변형 등에서의 안정성 향상이 필요합니다.

**② 세밀한 시맨틱 제어 부족 해결**
DragDiffusion, FreeDrag, DragNoise와 같은 방법들은 시맨틱 정보를 충분히 활용하지 못하여 덜 이상적인 결과를 낳는 경우가 있습니다. SAM2 등 세그멘테이션 모델과의 결합이 하나의 해결책이 될 수 있습니다.

**③ 마스크 없는(Mask-free) 편집**
DirectDrag는 마스크 없이(manual mask-free) 드래그 기반 이미지 편집을 위한 프레임워크를 제안하며, 이전 방법들과 달리 수작업 마스크나 프롬프트 없이 편집 품질을 유지합니다.

**④ 메모리 효율 및 실시간 처리**
EasyDrag의 경우처럼, 24GB 이상의 메모리(3090 GPU)를 요구하는 방법들은 광범위한 적용을 제한합니다. 경량화 및 LCM(Latency Consistency Model) 등과의 결합을 통한 실시간 편집 연구가 필요합니다.

**⑤ 다중 포인트 편집의 일관성**
DragNoise는 멀티 포인트 편집도 지원합니다. 그러나 다중 포인트 간의 상호 영향을 제어하는 더 정교한 메커니즘이 필요합니다.

**⑥ 3D 및 비디오 일관성**
단일 이미지 편집에서 나아가 멀티뷰/비디오 편집 시의 시간적(temporal) 일관성 유지가 중요한 연구 과제로 남아 있습니다.

---

> ⚠️ **정확도 유의사항**: 본 답변에서 제시된 수식의 일반 구조(LDM forward/reverse process 등)는 공개된 논문 전문(arXiv HTML)의 기술 내용과 관련 문헌을 종합하여 작성되었으나, 논문 원문 PDF의 정확한 표기법과 세부 notation에서 일부 차이가 있을 수 있습니다. 수식의 완전한 정확성이 필요할 경우 반드시 원문 PDF([CVPR 2024 Open Access](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Drag_Your_Noise_Interactive_Point-based_Editing_via_Diffusion_Semantic_Propagation_CVPR_2024_paper.pdf))를 직접 확인하시기 바랍니다.
