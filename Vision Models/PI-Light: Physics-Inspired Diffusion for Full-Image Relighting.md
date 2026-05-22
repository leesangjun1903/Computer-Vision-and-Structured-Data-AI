
# PI-Light: Physics-Inspired Diffusion for Full-Image Relighting

> **📌 논문 정보**
> - **제목**: PI-Light: Physics-Inspired Diffusion for Full-Image Relighting
> - **저자**: Zhexin Liang, Zhaoxi Chen, Yongwei Chen, Tianyi Wei, Tengfei Wang, Xingang Pan
> - **소속**: S-Lab, Nanyang Technological University & Tencent
> - **발표**: ICLR 2026
> - **arXiv**: [2601.22135](https://arxiv.org/abs/2601.22135)
> - **GitHub**: [ZhexinLiang/PI-Light](https://github.com/ZhexinLiang/PI-Light)

---

## 1. 핵심 주장 및 주요 기여 요약

### 1-1. 핵심 주장

Full-image relighting은 (1) 대규모 구조화된 쌍 데이터 수집의 어려움, (2) 물리적 타당성 유지의 난이도, (3) 데이터 기반 사전 지식(prior)에 의한 제한된 일반화 성능이라는 세 가지 근본적 문제로 인해 여전히 난제로 남아 있다. 합성 데이터와 실제 데이터 사이의 간극(synthetic-to-real gap)을 메우려는 기존 시도들은 아직도 최적의 성능에 미치지 못하고 있다.

PI-Light의 핵심 주장은 **물리 법칙을 diffusion 모델에 내재화함으로써 데이터 의존성을 낮추고, 실세계 이미지에 대한 일반화 성능을 획기적으로 향상**시킬 수 있다는 것이다.

### 1-2. 주요 기여

PI-Light는 physics-inspired diffusion 모델을 활용한 2단계(two-stage) 프레임워크로서, (i) **Batch-Aware Attention**: 이미지 컬렉션 전반에 걸쳐 intrinsic 예측의 일관성을 향상, (ii) **Physics-Guided Neural Rendering Module**: 물리적으로 타당한 광전달(light transport)을 강제, (iii) **Physics-Inspired Loss**: 학습 동역학을 물리적으로 의미 있는 공간으로 정규화하여 실세계 이미지 편집으로의 일반화 향상, (iv) **Curated Dataset**: 제어된 조명 조건 하에서 촬영된 다양한 객체 및 장면으로 구성된 데이터셋 제공. 이 구성 요소들이 함께 pretrained diffusion 모델의 효율적 파인튜닝을 가능하게 하며, 다운스트림 평가를 위한 견고한 벤치마크를 제공한다.

---

## 2. 해결 문제 · 제안 방법 · 모델 구조 · 성능 및 한계

### 2-1. 해결하고자 하는 문제

기존 scene-level 방식들(예: Choi et al., 2024; Kocsis et al., 2024a)은 full-image relighting을 시도하지만, 대부분 데이터 기반에 의존하여 학습 분포 밖으로의 일반화에 어려움을 겪으며, 자체 발광 객체(self-luminous objects)의 조명을 제대로 다루지 못한다. 최신 전경(foreground) relighting 방법들도 albedo 불일치와 부정확한 조명 제어 문제를 여전히 안고 있다.

구체적으로는 다음 3가지 문제를 표적으로 삼는다:

1. **데이터 희소성**: 조명 조건이 다양한 대규모 쌍 데이터 부재
2. **물리적 타당성 결여**: 순수 데이터 기반 모델의 비물리적 렌더링
3. **일반화 한계**: 학습 분포 외 장면에서의 성능 저하

---

### 2-2. 제안 방법 및 수식

#### (A) 전체 파이프라인: 2단계 프레임워크

PI-Light는 기존 PBR 방식과 유사하게 inverse neural rendering → neural forward rendering의 2단계 프레임워크를 채택하지만, 이전 연구와는 핵심적인 차별점을 가진다.

**[Stage 1] Inverse Neural Rendering (역신경렌더링)**

역신경 렌더링 파이프라인은 입력 RGB 이미지를 여러 intrinsic 구성 요소로 분해한다. 이 분해는 PBR(Physically-Based Rendering) 프레임워크를 기반으로 하며, 아래의 **Surface Reflection Model**을 따른다:

$$I_{\text{rendered}} = A \cdot D + S$$

여기서:
- $I_{\text{rendered}}$: 렌더링된 이미지
- $A$: Albedo (표면 반사색)
- $D$: Diffuse 성분
- $S$: Specular 성분

이 relighting 파이프라인은 Surface Reflection Model을 기반으로 하며, Physically-Based Rendering(PBR) 프레임워크 내에서 모델은 Principled BRDF를 따른다.

**[Stage 2] Neural Forward Rendering (신경 순방향 렌더링)**

이 단계에서는 이전 단계에서 분해된 intrinsic 구성 요소를 활용하여 Stable Diffusion을 파인튜닝하고, 사용자가 제공한 조명 조건에 따라 이미지를 relight한다.

relighting 모델의 입출력은 각각 세 가지 구성 요소로 이루어지며, 모델은 diffuse와 specular 성분을 별도로 출력하고, 최종 relit 이미지는 loss 제약을 통해 얻어진다.

따라서 최종 렌더링은 다음과 같이 표현될 수 있다:

$$I_{\text{relit}} = f_{\theta}(A, N, R, L_{\text{target}}) = D_{\text{pred}} + S_{\text{pred}}$$

여기서:
- $N$: 표면 법선(Normal)
- $R$: Roughness/Metallic 등 재질 파라미터
- $L_{\text{target}}$: 목표 조명 조건 (HDR 환경 맵 등)
- $f_{\theta}$: Physics-Guided Neural Renderer

---

#### (B) Batch-Aware Attention

Wonder3D에서 영감을 받아, inverse neural rendering 및 neural forward rendering 단계 모두에서 표준 self-attention 레이어를 전역적으로 인식하는(global-aware) 구조로 확장하여 배치 간 통신을 가능하게 하며, 이는 예측된 intrinsic 값들의 효율성과 일관성을 동시에 향상시킨다.

수식적으로 표준 Self-Attention을 배치 차원으로 확장하면:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right)V$$

Batch-Aware Attention에서는 동일 배치 내의 모든 이미지 토큰을 Key/Value로 포함:

$$\text{BA-Attention}(Q_i, K_{\text{all}}, V_{\text{all}}) = \text{softmax}\left(\frac{Q_i K_{\text{all}}^{\top}}{\sqrt{d_k}}\right)V_{\text{all}}$$

여기서 $K_{\text{all}}, V_{\text{all}}$은 배치 내 전체 이미지의 Key/Value를 concat한 것이다.

---

#### (C) Physics-Inspired Loss

physics-inspired neural forward rendering 모듈에서 물리 기반 loss는 효율적인 학습 메커니즘으로 작동한다. 이 physics-inspired loss 함수는 단순하지 않으며, 학습 동역학을 물리적으로 타당한 공간으로 정규화함으로써 수렴을 용이하게 하고, 더 적은 데이터와 연산으로도 올바른 light transport를 학습할 수 있도록 한다.

Loss는 크게 다음 세 항으로 구성됨 (논문 구조 기반 추정):

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda_{\text{pbr}} \mathcal{L}_{\text{PBR}} + \lambda_{\text{diff}} \mathcal{L}_{\text{diffusion}}$$

- $\mathcal{L}_{\text{recon}}$: 재구성 손실 (pixel-level 또는 perceptual loss)
- $\mathcal{L}_{\text{PBR}}$: PBR 렌더링 방정식 기반 물리적 정규화 항
- $\mathcal{L}_{\text{diffusion}}$: Diffusion 학습을 위한 denoising score matching loss

> ⚠️ **주의**: 위 수식은 논문의 개념적 구조를 바탕으로 표현한 것으로, 정확한 계수 및 세부 항목은 원문 PDF 직접 확인을 권장합니다.

---

### 2-3. 모델 구조

```
입력 이미지 (RGB)
        │
        ▼
┌─────────────────────────────────────────┐
│      Stage 1: Inverse Neural Rendering  │
│  (Diffusion 기반 Intrinsic Decompose)   │
│  - Batch-Aware Attention 적용           │
│  - 출력: Albedo(A), Normal(N),          │
│    Roughness(R), Depth(D), Mask(M)      │
└─────────────────────────────────────────┘
        │
        ▼  + Target Lighting L_target
┌─────────────────────────────────────────┐
│    Stage 2: Neural Forward Rendering    │
│  (Stable Diffusion 파인튜닝)            │
│  - Physics-Guided Neural Renderer       │
│  - Principled BRDF 기반                 │
│  - Diffuse + Specular 분리 출력         │
│  - Physics-Inspired Loss 적용           │
└─────────────────────────────────────────┘
        │
        ▼
   최종 Relit 이미지 (I_relit)
```

두 단계 모두에서 유리한 결과를 달성하며, 조명에 대한 정밀한 제어를 가능하게 한다.

---

### 2-4. 성능 향상

실험 결과, $\pi$-Light는 다양한 재질에 걸쳐 specular highlights와 diffuse reflections를 합성하며, 이전 방법들에 비해 실세계 장면에 대한 우월한 일반화 성능을 달성했다.

이러한 구성 요소들은 함께 pretrained diffusion 모델의 효율적인 파인튜닝을 가능하게 하며, 다운스트림 평가를 위한 견고한 벤치마크도 제공한다.

---

### 2-5. 한계점

논문 및 관련 연구에서 도출할 수 있는 한계는 다음과 같다:

1. **데이터셋의 범위**: 기존 데이터셋들은 주로 실내 장면에 집중하며, 반투명/투명 객체를 제외하기 위한 객체 마스크가 부족하고, 다양한 조명 조건 하에서의 이미지를 렌더링하지 않는 한계가 있다.
2. **Self-luminous 객체의 복잡성**: 자체 발광 객체의 조명 유지는 여전히 도전적인 과제로 남아있다.
3. **합성-실제 간극**: 합성 데이터와 실제 데이터 간의 간극을 완전히 해소하는 데에는 아직 한계가 존재한다.

---

## 3. 모델의 일반화 성능 향상 가능성 (심층 분석)

PI-Light의 일반화 성능 향상은 다음 네 가지 핵심 메커니즘에 의해 뒷받침된다:

### 3-1. Physics-Inspired Loss를 통한 일반화

Physics-inspired loss 함수는 학습 동역학을 물리적으로 타당한 공간으로 정규화하여 수렴을 용이하게 하고, **더 적은 데이터와 연산으로도** 올바른 light transport를 학습할 수 있도록 한다.

이는 데이터가 적은 실세계 환경에서도 물리 법칙이 **implicit regularizer** 역할을 하여 overfitting을 방지함을 의미한다.

### 3-2. Batch-Aware Attention을 통한 일관성

Batch-aware attention은 이미지 컬렉션 전반에 걸쳐 intrinsic 예측의 일관성을 향상시킨다. 이는 단일 이미지에서 발생하는 모호성(예: albedo-shading 분해의 모호성)을 다중 이미지 정보를 통해 해소하여, 보다 안정적이고 일반화된 표현을 학습하게 한다.

### 3-3. Pretrained Diffusion 모델 활용

Physics-inspired loss가 학습 동역학을 물리적으로 의미 있는 공간으로 정규화하여 실세계 이미지 편집으로의 일반화를 향상시키며, 이는 pretrained diffusion 모델의 효율적인 파인튜닝을 가능하게 한다.

기존에 대규모 데이터로 학습된 diffusion 모델의 강력한 이미지 prior를 활용함으로써, 제한된 조명 데이터만으로도 광범위한 장면 타입에 일반화할 수 있다.

### 3-4. 기존 방법과의 일반화 비교

기존의 대표적인 방법들과의 일반화 측면에서의 비교:

| 방법 | 일반화 전략 | 한계 |
|------|------------|------|
| **IC-Light** | 대규모 다양한 데이터로 학습, 유망한 결과를 보이나 파이프라인/데이터셋을 공개하지 않음 | 재현 불가 |
| **Neural Gaffer (NeurIPS 2024)** | 환경 맵 조건부 end-to-end diffusion, 합성 relighting 데이터셋으로 파인튜닝 | 전문적인 HDR 환경 맵 입력 필요 |
| **DiLightNet** | 조명 조건부 ControlNet 기반, 미리 정의된 재질/기하 추정 활용 | 전경 객체 중심, 전체 이미지 적용 어려움 |
| **PI-Light** | Physics-Inspired Loss + Batch-Aware Attention + Curated Data | 합성-실제 간극 완전 해소 어려움 |

---

## 4. 앞으로의 연구에 미치는 영향 및 고려 사항

### 4-1. 연구에 미치는 영향

#### (1) 물리 기반 정규화의 새로운 패러다임
PI-Light는 **물리 법칙을 loss 함수로 내재화**하는 것이 데이터 의존성을 낮추는 강력한 수단임을 입증했다. 이는 relighting을 넘어, 역광(backlighting) 제거, HDR 복원, 재질 추정 등 다양한 광학 관련 vision 태스크에서의 물리 기반 정규화 연구를 촉진할 것으로 기대된다.

#### (2) Diffusion 모델과 물리 시뮬레이션의 통합
Physics-inspired neural forward rendering 모듈에서 물리 기반 loss가 효율적인 학습 메커니즘으로 작동한다는 점은, 앞으로 Neural Radiance Field(NeRF), Gaussian Splatting 등 3D 표현 학습에도 유사한 접근법을 적용하는 연구의 기초가 될 수 있다.

#### (3) Batch-Aware Attention의 확장 가능성
Wonder3D에서 영감을 받은 배치 인식 전역 attention 메커니즘은 멀티뷰 일관성, 비디오 relighting, 장면 편집 등으로 자연스럽게 확장될 수 있다.

#### (4) 벤치마크 기여
제어된 조명 조건 하에서 다양한 객체와 장면을 포함하는 신중하게 구성된 데이터셋은 다운스트림 평가를 위한 견고한 벤치마크를 제공한다.

---

### 4-2. 향후 연구 시 고려할 점

#### ✅ 기술적 고려 사항

1. **투명/반투명 객체 처리**
   기존 데이터셋들은 반투명 또는 투명 객체를 제외하기 위한 객체 마스크가 부족하여 정확한 재질 추정을 방해한다. 투명 객체의 굴절, 반투과 등을 처리하는 확장 연구가 필요하다.

2. **비디오/시계열 relighting으로 확장**
   현재 PI-Light는 단일 이미지 또는 이미지 컬렉션을 대상으로 하지만, 비디오에서의 시간적 일관성(temporal consistency) 유지는 추가 연구가 필요하다.

3. **동적 조명 환경에서의 강건성**
   실세계의 복잡한 동적 조명 환경(예: 구름이 지나가는 야외, 점멸하는 실내 조명)에 대한 강건성 평가가 필요하다.

4. **자체 발광 객체(Self-Luminous Objects)**
   scene-level 방법들이 자체 발광 객체의 조명을 제대로 다루지 못하는 문제는 PI-Light에서도 여전한 도전 과제로, 에너지 보존 법칙을 명시적으로 모델링한 loss 설계가 필요하다.

5. **더 세밀한 조명 표현**
   현재 환경 맵(environment map) 기반의 조명 표현을 넘어, 지역적 조명 효과(예: 면 광원, 코스틱 효과)를 정밀하게 제어하는 연구가 요구된다.

#### ✅ 방법론적 고려 사항

6. **Few-shot/Zero-shot 일반화**
   물리 prior를 더욱 강화하여 완전히 새로운 도메인(예: 의료 이미지, 위성 이미지)에서도 few-shot으로 적용 가능한 방법 연구가 필요하다.

7. **Physics-Inspired Loss의 이론적 분석**
   physics-inspired loss가 학습 동역학에 미치는 영향이 non-trivial하며, 물리적으로 타당한 공간으로 수렴을 유도한다고 주장되지만, 이에 대한 이론적 수렴 보장 분석이 향후 연구 과제로 남는다.

---

## 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 접근법 | 일반화 | 전체 이미지 |
|------|------|--------|--------|------------|
| **IC-Light** | 2024 | End-to-end diffusion, 대규모 다양한 데이터 학습 | 높음 | ✅ |
| **Neural Gaffer** (NeurIPS 2024) | 2024 | 단일 이미지 + 환경 맵 조건, 명시적 분해 없이 end-to-end | 중간 | ❌ (전경 중심) |
| **DiLightNet** | 2024 | 조명 조건부 ControlNet, 미리 정의된 재질 활용 | 낮음~중간 | ❌ |
| **DifFRelight** (SIGGRAPH Asia 2024) | 2024 | 면 광원과 방향성 조명 통합, HDR 합성 | 중간 | ❌ (얼굴 특화) |
| **PI-Light** (ICLR 2026) | 2026 | 2단계 Physics-Inspired Diffusion | **높음** | ✅ |

최근 연구들은 명시적 장면 분해 없이 end-to-end로 relighting을 학습하기 위해 diffusion 모델을 활용하는 방향으로 발전하고 있다. PI-Light는 이 흐름에서 **물리 법칙을 학습 과정에 내재화**하는 것으로 차별화된다.

---

## 📚 참고 자료 (출처)

1. **[주 논문]** Zhexin Liang et al., *"PI-Light: Physics-Inspired Diffusion for Full-Image Relighting"*, arXiv:2601.22135, ICLR 2026. — https://arxiv.org/abs/2601.22135
2. **[논문 PDF]** https://arxiv.org/pdf/2601.22135
3. **[GitHub]** ZhexinLiang/PI-Light — https://github.com/ZhexinLiang/PI-Light
4. **[OpenReview]** PI-Light @ ICLR 2026 — https://openreview.net/forum?id=LWS5Gkx0mT
5. **[비교 연구 1]** Haian Jin et al., *"Neural Gaffer: Relighting Any Object via Diffusion"*, NeurIPS 2024. — https://arxiv.org/abs/2406.07520
6. **[비교 연구 2]** Zeng et al., *"DiLightNet: Fine-Grained Lighting Control for Diffusion-Based Image Generation"*, 2024.
7. **[비교 연구 3]** Zhang & Agrawala, *"IC-Light"*, GitHub, 2024.
8. **[비교 연구 4]** *"DifFRelight: Diffusion-Based Facial Performance Relighting"*, SIGGRAPH Asia 2024. — https://arxiv.org/abs/2410.08188
9. **[비교 연구 5]** *"PractiLight: Practical Light Control Using Foundational Diffusion Models"*, arXiv:2509.01837. — https://arxiv.org/pdf/2509.01837
10. **[비교 연구 6]** *"SpotLight: Shadow-Guided Object Relighting via Diffusion"*, arXiv:2411.18665. — https://arxiv.org/html/2411.18665v1

> ⚠️ **정확도 고지**: 본 분석은 공개된 arXiv abstract, OpenReview, PDF 일부에 기반합니다. 일부 수식(특히 loss 세부 항목)은 논문 전문 기반의 정확한 표현이 아닌 구조적 추론임을 밝히며, 정밀한 수식 확인은 원문 PDF를 직접 참조하시기 바랍니다.
