
# Cavia: Camera-controllable Multi-view Video Diffusion with View-Integrated Attention

> **논문 정보**
> - **제목**: Cavia: Camera-controllable Multi-view Video Diffusion with View-Integrated Attention
> - **저자**: Dejia Xu, Yifan Jiang, Chen Huang, Liangchen Song, Thorsten Gernoth, Liangliang Cao, Zhangyang Wang, Hao Tang
> - **arXiv**: [arXiv:2410.10774](https://arxiv.org/abs/2410.10774) (2024년 10월 14일)
> - **게재**: ICML 2025 (Proceedings of the 42nd International Conference on Machine Learning, PMLR 267:69293–69317)
> - **프로젝트 페이지**: https://ir1d.github.io/Cavia/

---

## 1. 핵심 주장 및 주요 기여 요약

### 핵심 주장

Cavia는 이미지로부터 카메라 제어 가능하고(camera-controllable), 다중 시점(multi-view)이면서 시공간적으로 일관된(spatiotemporally consistent) 비디오를 생성하기 위해 View-Integrated Attention 모듈을 통합한 새로운 프레임워크입니다.

Cavia는 카메라 모션을 정밀하게 제어하면서 동시에 객체 모션을 보존하여 동일한 장면의 다중 비디오를 생성할 수 있도록 하는 최초의 프레임워크입니다.

### 주요 기여 (4가지)

| 기여 | 설명 |
|------|------|
| **① View-Integrated Attention** | 기존 Spatial/Temporal Attention을 Cross-view + Cross-frame 3D Attention으로 확장 |
| **② 최초의 Multi-view + Camera Control 통합** | 카메라 모션 제어 + 객체 모션 보존을 동시에 달성한 최초 시도 |
| **③ 유연한 Joint Training 전략** | 다종 데이터(정적 장면, 합성 동적, 실세계 단안 비디오)를 동시 학습 |
| **④ 4-View 추론 외삽(extrapolation)** | 학습 시 2-view만 사용하더라도 추론 시 4-view로 확장 가능 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

생성된 프레임의 3D 일관성과 카메라 제어 가능성은 여전히 미해결 문제로 남아 있었으며, 기존 연구들은 카메라 제어를 생성 과정에 통합하려 했으나 단순한 카메라 경로에만 제한되거나, 동일한 장면에 대해 서로 다른 카메라 경로로부터 일관된 비디오를 생성하는 능력이 부족했습니다.

구체적으로 기존 방법들은 다음 두 가지 핵심 한계를 가졌습니다:

1. **단안 비디오 카메라 제어의 한계**: 단순 궤적에 제한, 복잡한 장면 적용 어려움
2. **멀티뷰 일관성의 결여**: 야외 멀티뷰 비디오 데이터 부족으로 인해 멀티뷰 생성 결과물이 비일관적인 준정적(near-static) 장면이나 합성 객체에 한정됨.

동시대 연구인 CVD(Kuang et al., 2024)는 멀티뷰 정적 비디오와 워핑-증강 단안 비디오를 활용하였지만, 제한된 기준선(limited baselines)을 가진 비디오만 생성 가능하며, 객체 모션이 존재할 때 비일관적인 결과를 생성합니다.

또 다른 동시대 연구인 Vivid-ZOO는 Objaverse 데이터셋의 동적 객체를 활용하여 멀티뷰 비디오 생성기를 학습시켰지만, 제한적인 데이터 소스로 인해 결과물이 고정 시점의 객체 중심 프레임에 머물러 실제 배경이 부족합니다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### (A) 기반 모델: Stable Video Diffusion (SVD)

Cavia의 모델은 사전 학습된 SVD(Stable Video Diffusion)를 기반으로 하며, SVD는 Stable Diffusion 2.1에 VideoLDM 아키텍처를 따라 시간적 합성곱 및 어텐션 레이어를 추가한 모델입니다. SVD는 연속 시간 노이즈 스케줄러(continuous-time noise scheduler)로 학습됩니다.

SVD의 확산 과정은 다음과 같이 표현됩니다:

$$\mathbf{x}_t = \mathbf{x}_0 + \mathbf{n}(t), \quad \mathbf{n}(t) \sim \mathcal{N}(0, \sigma^2(t)\mathbf{I})$$

이로부터 확률 흐름 ODE(Probability Flow ODE)를 통해 반복적 정제(iterative refinement)가 이루어집니다:

$$\frac{d\mathbf{x}}{dt} = -\dot{\sigma}(t)\sigma(t) \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$$

여기서 $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$는 학습된 스코어 함수(score function)이며, 확산 모델의 denoiser $D_\theta(\mathbf{x}_t, t)$로 근사됩니다.

#### (B) 카메라 조건화: Plücker 좌표

CameraCtrl, CamCo, VD3D 등은 ControlNet을 통해 Plücker 좌표를 비디오 모델에 도입하여 카메라 제어 정확도를 향상시켰으며, Cavia 역시 이 방식을 채택합니다.

Plücker 좌표는 3D 공간의 광선(ray)을 표현하는 6차원 벡터로, 각 픽셀의 카메라 광선을 다음과 같이 표현합니다:

$$\mathbf{p} = (\mathbf{d}, \mathbf{m}) \in \mathbb{R}^6, \quad \mathbf{m} = \mathbf{o} \times \mathbf{d}$$

여기서 $\mathbf{d} \in \mathbb{R}^3$는 광선 방향(ray direction), $\mathbf{o} \in \mathbb{R}^3$는 카메라 원점(camera origin), $\mathbf{m}$은 모멘트 벡터입니다.

Plücker 좌표 없이 학습된 모델 변형은 복잡한 카메라 시점 지시를 무시하고 단순화된 카메라 모션만을 생성합니다.

#### (C) View-Integrated Attention 모듈

Cavia는 뷰포인트와 프레임 간 일관성을 향상시키기 위해 **Cross-view Attention**과 **Cross-frame 3D Attention**으로 구성된 View-Integrated Attention을 도입합니다.

View-Integrated Attention 모듈은 Cross-view Attention과 Cross-frame Attention으로 구성되며, 각각 생성 프레임의 뷰포인트 일관성과 시간적 일관성을 강화합니다. 이 모듈은 어텐션 메커니즘에 추가적인 특징 차원을 통합하여 뷰와 프레임 간 일관성을 향상시킵니다.

**Cross-view Attention**: $V$개의 뷰를 동시에 처리할 때, 서로 다른 뷰의 특징을 어텐션 연산에 통합합니다. 단안 비디오의 기존 Spatial Attention이 단일 뷰 내 $H \times W$ 픽셀 간의 어텐션을 계산한다면, Cross-view Attention은 $V \times H \times W$ 특징 맵을 하나의 확장된 시퀀스로 처리합니다:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

여기서 $Q, K, V$는 $V$개 뷰의 특징들을 포함한 확장된 차원에서 계산됩니다.

**Cross-frame Attention**: 기존 1D Temporal Attention을 $V$개 뷰 × $T$개 프레임으로 확장하여 뷰 간 시간적 일관성을 유지합니다.

Cross-frame Attention 없이는 구부러진 벽과 같은 심각한 왜곡 아티팩트가 발생합니다.

Cross-view Attention 모듈을 제거하면 서로 다른 객체 모션을 포함하는 여러 개별 비디오 샘플이 생성되어 비일관성이 발생합니다. 예를 들어 펭귄이 첫 번째 경우에서 다르게 움직이거나, 불 속의 나무 막대기가 두 번째 경우에서 다르게 나타납니다. 이는 동일한 장면의 서로 다른 카메라 경로로부터 여러 비디오를 얻고자 하는 목표에 반합니다.

---

### 2-3. 모델 구조 (아키텍처 개요)

```
[입력 이미지 I]
        │
        ▼
[SVD 인코더 (VAE Encoder)]
        │
        ▼
[노이즈 잠재 변수 x_t + Plücker 좌표 임베딩 p]
        │
        ▼
┌─────────────────────────────────────┐
│        U-Net Denoiser (SVD 기반)     │
│                                     │
│  [Spatial Attention]                │
│       ↓ 확장                        │
│  [Cross-view Attention (새로 도입)] │
│                                     │
│  [Temporal Attention]               │
│       ↓ 확장                        │
│  [Cross-frame Attention (새로 도입)]│
└─────────────────────────────────────┘
        │
        ▼
[V개의 시공간 일관 비디오 출력]
(학습: 2-view → 추론 시 4-view 외삽 가능)
```

이 유연한 프레임워크는 추론 시 4개 뷰에서도 작동할 수 있으며, 향상된 뷰 일관성을 제공하고 생성된 프레임의 3D 재구성을 가능하게 합니다.

---

### 2-4. Joint Training 전략

Cavia는 정적(static), 단안 동적(monocular dynamic), 멀티뷰 동적(multi-view dynamic) 비디오의 큐레이션된 혼합을 활용하는 효과적인 Joint Training 전략을 도입하여, 생성 결과물에서 기하학적 일관성, 고품질 객체 모션, 배경 보존을 보장합니다.

학습 데이터는 세 가지 유형으로 구성됩니다:

| 데이터 유형 | 역할 |
|-------------|------|
| 장면 수준 정적 비디오 | 기하학적 일관성 학습 |
| 객체 수준 합성 멀티뷰 동적 비디오 | 다중 시점 일관성 학습 |
| 실세계 단안 동적 비디오 | 복잡한 실세계 장면 적용 능력 |

합성 데이터에 대한 과적합(overfitting)을 방지하기 위해, 복잡한 장면에서의 성능을 향상시키고자 포즈-주석(pose-annotated) 단안 비디오로 모델을 파인튜닝합니다.

---

### 2-5. 성능 향상

광범위한 실험을 통해 Cavia는 기하학적 일관성과 지각 품질 측면에서 최신 방법들을 능가함을 입증합니다.

Cavia의 프레임워크는 실제 이미지와 텍스트-이미지 생성 이미지에 대한 광범위한 평가에서 도전적인 실내, 실외, 객체 중심, 대규모 장면 사례들에 대한 적용 가능성을 보여줍니다.

비교 대상 방법들:
- **MotionCtrl** (Wang et al., 2023)
- **CameraCtrl** (He et al., 2024)
- **CVD** (Kuang et al., 2024) — Collaborative Video Diffusion
- **Vivid-ZOO** (Li et al., 2024)
- **VD3D** (Bahmani et al., 2024)

---

### 2-6. 한계

실세계의 멀티뷰 비디오 데이터가 극히 부족하다는 근본적인 문제로 인해, 현재 멀티뷰 생성은 비일관적인 준정적 장면이나 합성 객체에 제한되는 경향이 있습니다.

논문의 공개 자료와 연구 맥락에서 파악 가능한 추가적 한계:

1. **데이터 의존성**: 합성 데이터(Objaverse 등) 의존으로 실세계 분포와의 도메인 갭 존재
2. **계산 비용**: 멀티뷰 어텐션 확장은 싱글뷰 대비 메모리·연산 비용이 $V$배 증가
3. **뷰 수 제한**: 학습은 2-view 기준이며, 4-view 추론은 외삽(extrapolation)으로 일관성 저하 가능성

---

## 3. 모델의 일반화 성능 향상 가능성

Cavia의 일반화 성능 향상을 위한 핵심 설계 요소들은 다음과 같습니다:

### 3-1. 이종 데이터 Joint Training에 의한 일반화

이 유연한 설계는 장면 수준의 정적 비디오, 객체 수준의 합성 멀티뷰 동적 비디오, 실세계 단안 동적 비디오를 포함한 다양하고 큐레이션된 데이터 소스로의 Joint Training을 가능하게 합니다.

이 전략은 단일 도메인 학습의 한계를 극복하고 다음과 같은 일반화를 가능하게 합니다:
- **정적 장면 → 동적 장면**: 동적 데이터를 추가함으로써 객체 모션이 있는 실세계 장면으로 일반화
- **합성 → 실세계 도메인**: 단안 실세계 비디오 파인튜닝으로 도메인 갭 완화

합성 데이터에 대한 과적합을 방지하기 위해, 포즈-주석 단안 비디오로 파인튜닝하여 복잡한 장면에서의 성능을 향상시킵니다.

### 3-2. 뷰 수 외삽(View Extrapolation)에 의한 확장성

Cavia는 추론 시 4개의 뷰를 생성하도록 외삽할 수 있으며, 생성된 프레임의 3D 재구성을 가능하게 합니다.

이는 **학습 시 뷰 수에 구애받지 않고 추론 시 확장 가능한 아키텍처 설계**의 일반화 능력을 시사합니다.

### 3-3. Plücker 좌표 조건화에 의한 카메라 일반화

Plücker 좌표는 임의의 카메라 궤적을 균일하게 표현할 수 있어, 학습 시 보지 못한 카메라 경로에도 일반화됩니다. 이는 특정 카메라 모션 패턴의 LoRA 방식(AnimateDiff, SVD-LoRA)과 대비되는 장점입니다.

카메라 제어 향상에 초점을 맞춘 이 연구는 빠르게 성장하는 비디오 확산 프로세스 제어 분야에 기여합니다.

### 3-4. 일반화 성능의 현실적 제약

그러나 일반화 성능 향상에는 다음과 같은 제약이 여전히 존재합니다:
- **멀티뷰 동적 데이터 부족**: 실세계 멀티뷰 동적 데이터가 극히 드물어 합성 데이터에 의존해야 함
- **도메인 편향**: Objaverse 기반 합성 객체 비디오와 실세계 장면 간의 외관(appearance) 및 물리적 특성 차이

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 연도 | 기관 | 카메라 제어 | 멀티뷰 일관성 | 동적 장면 | 핵심 기법 |
|------|------|------|------------|--------------|-----------|-----------|
| **AnimateDiff** | 2023 | 개인 | 제한적(LoRA) | ✗ | △ | Motion LoRA |
| **MotionCtrl** | 2023 | 기타 | 카메라 행렬 | ✗ | ✓ | 카메라 행렬 조건화 |
| **CameraCtrl** | 2024 | 기타 | Plücker 좌표 | ✗ | ✓ | ControlNet + Plücker |
| **CVD** | 2024 | Stanford | 제한적 | △ | △ | 정적 멀티뷰 + 워핑 |
| **Vivid-ZOO** | 2024 | KAUST | ✗ | △ | ✓ | Objaverse 렌더링 |
| **VD3D** | 2024 | Snap | Plücker 좌표 | ✗ | ✓ | DiT 기반 |
| **Cavia** | 2024 | Apple+UT Austin | Plücker 좌표 | ✓ | ✓ | View-Integrated Attention |

CVD는 멀티뷰 정적 비디오와 워핑-증강 단안 비디오를 기반으로 하지만, 제한된 기준선을 가진 비디오만 생성 가능하며 객체 모션이 있을 때 비일관적인 결과를 생성합니다.

AnimateDiff와 SVD는 특정 카메라 모션을 위해 개별 카메라 LoRA 모델을 사용하며, MotionCtrl은 카메라 행렬을 도입하여 유연성을 향상시켰고, CameraCtrl, CamCo, VD3D는 ControlNet을 통해 Plücker 좌표를 비디오 모델에 도입하여 카메라 제어 정확도를 향상시켰습니다.

**Cavia의 핵심 차별점**:
Cavia는 카메라 모션에 대한 정밀한 제어를 가능하게 하면서 동시에 객체 모션을 보존하여 동일한 장면의 여러 비디오를 생성할 수 있는 최초의 프레임워크입니다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5-1. 앞으로의 연구에 미치는 영향

**① 4D 콘텐츠 생성의 새로운 패러다임**

Cavia는 동적 3D 장면(4D: 공간 3D + 시간 1D)을 비디오 확산 모델로부터 생성하는 연구의 핵심 기반이 됩니다. 실제 이미지와 텍스트-이미지 생성 이미지에 대한 광범위한 평가는 도전적인 실내, 실외, 객체 중심, 대규모 장면 케이스에 대한 적용 가능성을 보여줍니다.

**② 3D 재구성과의 통합**

Cavia의 유연한 프레임워크는 추론 시 4개 뷰에서 작동하여 생성된 프레임의 3D 재구성을 가능하게 합니다. 이는 향후 비디오 생성 → 3D Gaussian Splatting/NeRF 재구성으로 이어지는 파이프라인 연구에 큰 영향을 미칩니다.

**③ 자율주행 및 로보틱스 시뮬레이션**

관련 연구로는 드라이빙 비디오의 멀티뷰 제어를 다루는 MyGo, DriveScape 등이 있습니다. Cavia의 멀티뷰 + 카메라 제어 프레임워크는 자율주행 데이터 증강에 직접 활용 가능합니다.

**④ View-Integrated Attention의 범용화**

Cross-view + Cross-frame Attention의 설계 원리는 이미지→비디오 확산 모델(I2V) 전반에 적용 가능하며, SVD 이외의 대규모 비디오 생성 모델(예: CogVideoX, Sora 유사 아키텍처)로의 확장 가능성을 열어줍니다.

---

### 5-2. 앞으로의 연구에서 고려할 점

**① 실세계 멀티뷰 동적 데이터 확보**

실세계의 멀티뷰 비디오 데이터가 극히 부족하다는 근본적인 문제가 있으며, 이로 인해 멀티뷰 생성은 비일관적인 준정적 장면이나 합성 객체에 제한됩니다. 향후 연구는 실세계 멀티뷰 비디오 데이터셋 구축 또는 더 효과적인 합성-실세계 도메인 적응(domain adaptation) 기법을 고려해야 합니다.

**② 확장성(Scalability): 더 많은 뷰와 더 긴 시퀀스**

현재 Cavia는 주로 2-view 학습 기반이며, 4-view는 외삽입니다. 더 많은 뷰($V \gg 4$)와 더 긴 시퀀스($T \gg 25$)로의 확장은 메모리 복잡도 $O(V^2 T^2)$ 문제를 낳으므로, **효율적인 어텐션 근사**(예: Sparse Attention, Flash Attention) 연구가 필요합니다.

**③ 보다 정교한 카메라-객체 분리**

Cavia의 Plücker 좌표는 카메라 모션을 제어하지만, 복잡한 장면에서 카메라 모션과 객체 모션의 명시적 분리(disentanglement)는 여전히 도전적입니다. 이를 위한 더 정교한 조건화 메커니즘 연구가 필요합니다.

**④ 비디오 기반 세계 모델(World Model)로의 확장**

Cavia의 멀티뷰 일관 비디오 생성 능력은 향후 **비디오 기반 세계 모델** 연구와 자연스럽게 결합될 수 있습니다. 특히 물리 법칙을 암묵적으로 인코딩하는 세계 모델과의 통합은 로보틱스 및 시뮬레이션 분야에서 중요한 방향입니다.

**⑤ 추론 속도 최적화**

멀티뷰 + 멀티프레임 확산 모델은 추론 시 단안 비디오 대비 $V$배 이상의 계산이 필요합니다. Consistency 모델, 흐름 일치(flow matching), 또는 캐시 기반 어텐션 최적화를 통한 추론 가속화가 실용적 배포를 위해 중요합니다.

---

## 📚 참고 자료 (References)

1. **arXiv 원문**: Dejia Xu et al., "Cavia: Camera-controllable Multi-view Video Diffusion with View-Integrated Attention," arXiv:2410.10774, Oct. 2024. https://arxiv.org/abs/2410.10774
2. **ICML 2025 공식 게재**: Proceedings of the 42nd International Conference on Machine Learning, PMLR 267:69293–69317, 2025. https://proceedings.mlr.press/v267/xu25l.html
3. **프로젝트 페이지**: https://ir1d.github.io/Cavia/
4. **Apple Machine Learning Research 소개**: https://machinelearning.apple.com/research/cavia
5. **HuggingFace 논문 페이지**: https://huggingface.co/papers/2410.10774
6. **OpenReview (ICML 2025)**: https://openreview.net/forum?id=g5CijB2ERy
7. **ICML 2025 포스터**: https://icml.cc/virtual/2025/poster/44487
8. **ResearchGate**: https://www.researchgate.net/publication/384929932
9. **NASA ADS**: https://ui.adsabs.harvard.edu/abs/2024arXiv241010774X/abstract

### 비교 분석에서 언급된 관련 논문
- Kuang et al., "Collaborative Video Diffusion (CVD)," arXiv:2405.17414, 2024.
- Li et al., "Vivid-ZOO: Multi-view Video Generation with Diffusion Model," arXiv:2406.08659, 2024.
- Bahmani et al., "VD3D: Taming Large Video Diffusion Transformers for 3D Camera Control," arXiv:2407.12781, 2024.
- He et al., "CameraCtrl: Enabling Camera Control for Text-to-Video Generation," arXiv:2404.02101, 2024.
- Guo et al., "AnimateDiff," arXiv:2307.04725, 2023.
- Stability AI, "Stable Video Diffusion (SVD)," 2023.
