# Improving Sample Quality of Diffusion Models Using Self-Attention Guidance

### 1. 핵심 주장 및 주요 기여 요약

이 논문은 **Self-Attention Guidance (SAG)**라는 조건 및 학습 없이 확산 모델의 샘플 품질을 개선하는 혁신적인 방법을 제시합니다. 핵심 기여는 다음과 같습니다.

**주요 기여:**

- **조건-무관(condition-free) 가이던스의 일반화**: 기존의 분류기 가이던스(Classifier Guidance, CG)와 분류기-무관 가이던스(Classifier-Free Guidance, CFG)는 외부 조건(클래스 레이블, 텍스트)을 요구하지만, SAG는 diffusion 모델의 **내부 자기-주의(self-attention) 맵**을 활용하여 추가 학습이나 외부 조건 없이 작동합니다.[1]

- **blur guidance에서 SAG로의 진화**: 먼저 간단한 blur guidance를 제안한 후, 이의 한계(큰 guidance scale에서의 노이즈)를 극복하기 위해 자기-주의 기반 선택적 블러링으로 개선.[1]

- **광범위한 적용성**: ADM, IDDPM, Stable Diffusion, DiT 등 다양한 diffusion 모델에 적용 가능하며, 기존 가이던스 방법(CG, CFG)과의 직교성을 시연.[1]

- **추가 학습 불필요**: 사전 학습된 diffusion 모델에 직접 적용 가능.[1]

***

### 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 향상 및 한계

#### 2.1 해결하고자 하는 문제

기존 가이던스 방법들의 한계:

- **Classifier Guidance (CG)**: 추가 분류기 모델을 학습해야 함.[1]
- **Classifier-Free Guidance (CFG)**: 학습 단계에서 라벨 드롭핑이 필요하고, 조건 정보가 필수.[1]
- 두 방법 모두 **외부 조건에 의존**하므로 조건 없는 생성에는 적용 불가.[1]

SAG는 이러한 문제를 해결하고자 합니다.

#### 2.2 제안 방법 (수식 포함)

**일반화된 가이던스 공식:**

먼저 기존 가이던스를 일반화한 프레임워크를 제시합니다. 타임스텝 $$t$$에서 조건 $$\mathbf{h}_t$$와 교란된 샘플 $$\bar{\mathbf{x}}_t$$가 주어질 때:[1]

$$\tilde{\epsilon}(\bar{\mathbf{x}}_t, \mathbf{h}_t) = \epsilon_\theta(\bar{\mathbf{x}}_t, \mathbf{h}_t) - s\sigma_t\nabla_{\bar{\mathbf{x}}_t}\log p_{\textrm{im}}(\mathbf{h}_t|\bar{\mathbf{x}}_t)$$

여기서 $$s$$는 가이던스 강도(guidance scale)입니다.[1]

**Blur Guidance:**

$$\mathbf{h}_t = \mathbf{x}_t - \tilde{\mathbf{x}}_t$$로 설정하여, 가우시안 블러된 버전과의 차이를 가이던스로 사용:

$$\tilde{\mathbf{x}}_t = (1-M_t) \odot \mathbf{x}_t + M_t \odot \tilde{\mathbf{x}}_t$$

여기서 $$\tilde{\mathbf{x}}_t$$는 중간 재구성 값입니다.[1]

**Self-Attention Guidance (SAG):**

자기-주의 맵 $$A_t$$를 집계(global average pooling)하여 얻은 마스크 $$M_t$$를 사용:

$$M_t = \mathbf{1}(A_t > \psi)$$

여기서 $$\psi$$는 마스킹 임계값(보통 $$A_t$$의 평균값으로 설정):[1]

$$\hat{\mathbf{x}}_t = (1-M_t)\odot\mathbf{x}_t + M_t\odot\tilde{\mathbf{x}}_t$$

$$\tilde{\epsilon}(\mathbf{x}_t) = \epsilon_\theta(\hat{\mathbf{x}}_t) + (1+s)(\epsilon_\theta(\mathbf{x}_t) - \epsilon_\theta(\hat{\mathbf{x}}_t))$$

이 공식은 일반화된 가이던스 프레임워크의 특수한 경우:

$$\tilde{\epsilon}(\mathbf{x}_t) = \epsilon_\theta(\mathbf{x}_t) + (1+s)(\epsilon_\theta(\mathbf{x}_t, \mathbf{h}_t) - \epsilon_\theta(\mathbf{x}_t))$$

**자기-주의 맵 집계:**

다중 헤드 자기-주의 맵 $$A_t^S \in \mathbb{R}^{N \times (HW) \times (HW)}$$으로부터:[1]

$$A_t = \text{Upsample}(\text{Reshape}(\text{GAP}(A_t^S)))$$

여기서 GAP는 global average pooling입니다.[1]

#### 2.3 모델 구조

**Diffusion 모델의 구조:**

표준 U-Net 기반 architecture를 사용하며, 주요 특징:[1]

- 여러 해상도에서 자기-주의 레이어 포함
- Algorithm 1의 SAG 샘플링 알고리즘을 역과정에 통합:

```
Algorithm 1: Self-Attention Guidance (SAG) Sampling
입력: Diffusion 모델 Model(x_t), Gaussian-Blur 함수
초기화: x_T ~ N(0, I)

for t in T, T-1, ..., 1 do:
    ε_t, Σ_t, A_t ← Model(x_t)
    M_t ← 1(A_t > ψ)
    x̂_0 ← (x_t - √(1-ᾱ_t)ε_t)/√ᾱ_t     (Eq. 2)
    x̃_0 ← Gaussian-Blur(x̂_0)
    x̃_t ← √ᾱ_t x̃_0 + √(1-ᾱ_t)ε_t
    x̂_t ← (1-M_t)⊙x_t + M_t⊙x̃_t       (Eq. 15)
    ε̂_t ← Model(x̂_t)
    ε̃_t ← ε̂_t + (1+s)(ε_t - ε̂_t)        (Eq. 16)
    x_{t-1} ~ N(1/√ᾱ_t(x_t - (1-α_t)/√(1-ᾱ_t) ε̃_t), Σ_t)
end for
return x_0
```

#### 2.4 성능 향상

**정량적 결과:**

ImageNet 256×256 무조건 생성 (ADM 기반):[1]

| 메트릭 | 베이스라인 | SAG 적용 |
|--------|-----------|---------|
| FID ↓ | 26.21 | 20.08 |
| sFID ↓ | 6.35 | 5.77 |
| IS ↑ | 39.70 | 45.56 |
| Precision ↑ | 0.61 | 0.68 |

IDDPM (ImageNet 64×64):[1]
- FID: 19.2 → 18.0 (약 6% 개선)

LSUN Cat 256×256 (무조건):[1]
- FID: 7.03 → 6.87

LSUN Horse 256×256 (무조건):[1]
- FID: 3.45 → 3.43

**조건부 생성:**

ImageNet 256×256 조건부 (ADM):[1]
- FID: 10.94 → 9.41 (13.9% 개선)

**기존 가이던스와의 조합:**

Classifier Guidance와 결합 (ImageNet 128×128):[1]
- FID: 2.97 → 2.58 (SAG+CG)

DiT-XL/2 + CFG:[1]
- FID: 2.27 → 2.16 (SAG+CFG)

**마스킹 전략 비교 (ImageNet 128×128):**[1]

| 전략 | FID ↓ | IS ↑ |
|-----|-------|------|
| 베이스라인 | 5.98 | 141.72 |
| Blur (Global) | 5.82 | 143.15 |
| High-frequency | 5.74 | 148.87 |
| Random | 5.68 | 148.99 |
| **Self-attention** | **5.47** | **151.12** |
| DINO-attention | 5.63 | 146.18 |

**가우시안 블러 파라미터 분석:**[1]

$$\sigma$$ 값에 따른 성능 변화:

| σ | FID ↓ | IS ↑ |
|---|-------|------|
| 1 | 5.58 | 145.85 |
| 3 (최적) | 5.47 | 151.12 |
| 9 | 5.70 | 148.70 |
| 27 | 5.80 | 147.83 |

**가이던스 스케일 분석:**[1]

$$s = 0.1$$에서 최적의 FID, $$s = 0.3$$에서 최고 Precision 달성

**계산 비용:**

메모리 오버헤드: 12,167 MB → 12,209 MB (+0.3%)
런타임: 108.27s (기준) → 186.60s (약 72% 증가, CFG와 유사)[1]

#### 2.5 논문의 한계

**주요 한계:**

1. **다양성 감소**: Recall 메트릭에서 감소 관찰 (0.65 → 0.59). 논문은 이것이 sample fidelity와 diversity 간의 trade-off라고 설명하지만, 생성 다양성 감소는 여전히 중요한 문제입니다.[1]

2. **계산 오버헤드**: 추론 시간이 약 72% 증가하며, 이는 CFG와 유사하지만 실시간 애플리케이션에서는 제약이 될 수 있습니다.[1]

3. **자기-주의 맵 의존성**: 방법의 효과가 diffusion 모델의 자기-주의 메커니즘의 품질에 의존합니다. 자기-주의가 충분하지 않은 아키텍처에서는 성능 저하 가능.[1]

4. **하이퍼파라미터 민감도**: 

   - 마스킹 임계값 $$\psi$$: 0.7, 1.0, 1.3 테스트 시 1.0에서 최적[1]
   - 가우시안 블러 $$\sigma$$: 입력 해상도에 따라 다르게 설정 필요 (σ=1 for 64×64, σ=9 for 256×256)[1]
   - 가이던스 스케일 $$s$$: 음수값이나 너무 큰값(≥0.4)은 품질 저하[1]

5. **일반화 제한**: 
   
   - Stable Diffusion의 빈 프롬프트(" ")에서만 평가
   - 텍스트-이미지 생성에서의 효과가 제한적 (CFG와 결합할 때만 효과적)[1]

***

### 3. 모델의 일반화 성능 향상 가능성 및 분석

#### 3.1 일반화 성능 향상의 메커니즘

**내부 정보 활용의 이점:**

SAG는 diffusion 모델의 **내부 자기-주의 맵을 활용**하여 다음과 같은 일반화 이점을 제공합니다:[1]

1. **특성 기반 가이던스**: 자기-주의는 생성 과정에서 모델이 주목하는 영역을 반영하므로, 이를 기반으로 한 가이던스는 모델의 내재적 선호도와 정렬됩니다.[1]

2. **고주파 특성 강화**: 논문 Fig. 4는 자기-주의 맵이 고주파 디테일과 높은 상관성을 가지고 있음을 시연하였고, 이는 생성 품질 개선에 기여합니다.[1]

3. **의미론적 정렬**: Table 9의 의미론적 분석에서 자기-주의 맵이 객체 마스크와 높은 IoU(Intersection over Union)를 달성(0.16 대비 0.23-0.26)했습니다.[1]

**비분포 샘플에 대한 견고성:**

Blur guidance는 글로벌 블러로 인해 구조적 모호함을 야기하여 큰 guidance scale에서 노이즈가 증가합니다. 하지만 SAG는 자기-주의 마스크를 통해 **선택적 블러링**을 수행하므로:[1]

$$\hat{\mathbf{x}}_t = (1-M_t)\odot\mathbf{x}_t + M_t\odot\tilde{\mathbf{x}}_t$$

이 공식은 원본 정보의 일부(비마스크된 영역)를 유지하여 비분포 이탈을 방지합니다.[1]

#### 3.2 다양한 모델에 대한 일반화

**광범위한 아키텍처 지원:**

- **ADM (Ablative Diffusion Model)**: U-Net 기반, FID 26.21 → 20.08[1]
- **IDDPM (Improved DDPM)**: U-Net 기반 분산 예측 추가, FID 19.2 → 18.0[1]
- **Stable Diffusion**: Latent diffusion 모델, 가시적 품질 개선[1]
- **DiT (Diffusion Transformers)**: Transformer 기반 백본, FID 2.27 → 2.16[1]

이는 SAG가 **아키텍처-무관적 접근**임을 시사합니다. 자기-주의 메커니즘이 있는 모든 diffusion 모델에 적용 가능합니다.[1]

#### 3.3 조건 유무에 관계없는 적용 가능성

**조건-무관성 증명:**

무조건 생성, 조건부 생성, 기존 가이던스 방법과의 결합 모두에서 개선을 달성했습니다:[1]

- 무조건 생성: FID 26.21 → 20.08 (23% 개선)
- 조건부 생성: FID 10.94 → 9.41 (14% 개선)
- CG 결합: FID 2.97 → 2.58 (13% 개선)
- CFG 결합: FID 2.27 → 2.16 (5% 개선)

**직교성(Orthogonality):**

Table 3과 4는 SAG가 CG, CFG와 독립적으로 작동하며, 결합 시 시너지 효과를 제공함을 보여줍니다.[1]

#### 3.4 데이터셋 간 일반화

**다양한 데이터셋에서의 성능:**

1. **ImageNet**: 고수준 의미론적 특성, 다양한 객체
2. **LSUN Cat/Horse**: 도메인 특화 데이터, 제한된 다양성
3. **Stable Diffusion의 공개 이미지**: 실제 배포 환경

모든 데이터셋에서 일관된 개선이 관찰되어 **도메인-무관적 우수성**을 시사합니다.[1]

#### 3.5 일반화 한계

그러나 다음과 같은 일반화 제약이 존재합니다:[1]

1. **모델 용량 의존성**: 자기-주의가 불충분한 경량 모델에서는 효과가 제한적일 수 있습니다.

2. **Recall 감소**: 모든 벤치마크에서 recall이 감소(0.63-0.65 → 0.50-0.59)하여, 모드 커버리지가 감소할 수 있습니다.[1]

3. **하이퍼파라미터 비-일반성**: σ, s, ψ의 최적값이 **해상도와 모델에 따라 다르게** 설정되어야 합니다(Table 8).[1]

***

### 4. 논문이 앞으로의 연구에 미치는 영향 및 고려사항

#### 4.1 학술적 영향

**1. 조건-무관 가이던스의 새 패러다임:**

전통적으로 diffusion 가이던스는 외부 조건(클래스, 텍스트)에 의존했지만, SAG는 **내부 표현(self-attention)**을 활용하는 새로운 가능성을 열었습니다. 이는 다음 논문들에 영향을 미쳤습니다:[2]

- **Universal Guidance for Diffusion Models (2024)**: SAG의 아이디어를 확장하여 임의의 외부 differentiable predictor와 통합하는 통합 프레임워크 제시[2]
- **TFG (Unified Training-Free Guidance, 2024)**: 다양한 학습-무관 가이던스 방법을 통합하는 이론적 프레임워크 제공[3][4]

**2. 자기-주의 메커니즘의 중요성 재조명:**

최근 조사(2024-2025) "Attention in Diffusion Models: A Survey"는 SAG를 포함하여 diffusion 모델에서 주의 메커니즘의 역할을 체계적으로 분류했습니다. 구체적으로, SAG는 다음과 같이 분류됩니다:[5]

- **Attention Score-Driven Guidance**: 자기-주의 특성 맵을 손실 또는 제약으로 활용하여 생성 과정 전반에 걸쳐 일관성 확보
- **Attention-based Mask Guidance**: 자기-주의 맵으로부터 마스크를 생성하여 공간적 제어 수행

이는 diffusion 모델의 주의 메커니즘이 단순한 아키텍처 성분이 아니라 **생성 프로세스의 근본적인 제어기**임을 입증합니다.[5]

**3. 생성 모델의 해석 가능성:**

SAG는 diffusion 모델의 자기-주의 맵이 고주파 디테일과 의미론적 정보를 캡슐화하고 있음을 실증적으로 보여주었습니다. 이는 다음 연구에 기여:[1]

- Diffusion 모델의 내부 표현 이해
- 생성 과정의 단계별 특성 출현 메커니즘 규명

#### 4.2 실무적 영향

**1. 사전 학습 모델 개선의 새로운 방법:**

SAG는 **추가 학습 없이** 기존의 사전 학습된 diffusion 모델의 품질을 개선할 수 있는 "플러그인" 방식의 솔루션입니다. 이는 다음의 실무적 이점을 제공합니다:[1]

- 새로운 모델 학습의 계산 비용 회피
- 기존 배포된 모델의 품질 즉시 개선 가능
- 다양한 하류 모델에 통합 가능

**2. 조건 정보 부족 상황에서의 생성:**

Stable Diffusion의 빈 프롬프트 실험(Fig. 8)은 **조건 정보 없이도 생성 품질을 개선**할 수 있음을 시사합니다. 이는 다음 시나리오에 유용합니다:[1]

- 조건 정보가 불충분하거나 불명확한 경우
- 조건 없는 무조건 생성이 필요한 경우
- 기존 가이던스 방법이 작동하지 않는 도메인

#### 4.3 앞으로의 연구 방향 및 고려사항

**1. 다양성-충실도 트레이드오프 해결:**

논문이 언급한 recall 감소(0.65 → 0.59)는 **중요한 미해결 문제**입니다. 향후 연구는:[1]

- 동적 guidance scale 조정 메커니즘 개발
- 다양성을 보존하면서 충실도를 개선하는 하이브리드 접근
- 가이던스의 선택적 적용 (특정 타임스텝에만 적용)

**2. 계산 효율성 개선:**

72% 런타임 증가는 실시간 애플리케이션에서 제약입니다. 다음과 같은 개선 방향이 고려될 수 있습니다:[1]

- 자기-주의 맵의 경량 근사 계산
- 가우시안 블러의 고속 구현 (FFT 기반)
- 선택적 SAG 적용 (일부 레이어/타임스텝만)

관련 최신 연구: Latent Diffusion Model-Enabled Semantic Communication (2024)는 **일관성 증류(consistency distillation)**를 활용하여 diffusion 모델의 추론 비용을 극적으로 감소시키는 방법을 제시합니다.[6]

**3. 하이퍼파라미터 자동 최적화:**

Table 8에서 보듯이, σ, s, ψ의 최적값이 모델과 해상도에 따라 다릅니다. 향후 연구 방향:[1]

- 자동 하이퍼파라미터 튜닝 알고리즘 개발
- 모델-무관적 휴리스틱 설계

최신 연구 TFG는 이 문제를 부분적으로 해결하여, **효율적인 하이퍼파라미터 탐색 전략**을 제시했습니다.[4][3]

**4. 자기-주의 이외의 내부 표현 활용:**

SAG는 자기-주의에만 제한되지만, 다음과 같은 확장이 가능합니다:[1]

- **Cross-attention 맵** (조건부 모델): 텍스트-이미지 생성에서의 미세한 제어
- **Feature map 활용**: 중간 특성 맵의 직접 활용
- **Gradient 기반 정보**: 모델의 그래디언트 정보 활용

최신 연구 "Towards Understanding Cross and Self-Attention in Stable Diffusion" (CVPR 2024)는 Stable Diffusion에서 cross-attention과 self-attention의 역할을 분석하여, 이를 기반으로 한 이미지 편집 기법을 제시합니다.[7]

**5. 일반화 성능의 이론적 기초:**

논문은 주로 실증적 결과를 제시하지만, 다음과 같은 이론적 질문이 남아있습니다:[1]

- SAG가 비분포(out-of-distribution) 샘플에 얼마나 견고한가?
- 일반화 오차의 이론적 한계는?
- 자기-주의 기반 가이던스의 최적성은?

최신 연구들이 이 문제를 다루고 있습니다:

- **On the Generalization Properties of Diffusion Models (2025)**: 확산 모델의 일반화 간격에 대한 이론적 분석, $$O(n^{-2/5}+m^{-4/5})$$의 다항식 작은 일반화 오류 제시[8]
- **Towards a Mechanistic Explanation of Diffusion Model Generalization (2025)**: Diffusion 모델이 **국소화된 denoising 연산**을 통해 일반화하는 메커니즘 설명[9]

**6. Out-of-Distribution (OOD) 일반화:**

최근 관심 분야인 OOD 일반화와 SAG의 상관성:[1]

- GDA (Generalized Diffusion Adaptation, 2024): 테스트 시간 적응을 위해 diffusion 모델을 사용하여 분포 이탈(domain shift) 극복, 이미지넷-C에서 4.4-5.02% 개선[10]
- CausalDiffRec (2024): 구조적 인과 모델과 diffusion을 결합하여 OOD 추천에서 일반화 성능 개선[11]

이러한 연구들은 **diffusion 기반 내부 정보 활용(SAG의 핵심 아이디어)이 OOD 일반화에도 유효**함을 시사합니다.

***

### 5. 2020년 이후 관련 최신 연구 탐색

#### 5.1 가이던스 방법의 발전

**분류기-무관 가이던스 (CFG) [2021 이후]:**

- **Ho & Salimans (2021)**: Classifier-Free Guidance 제시 - diffusion 모델 학습 중 조건을 드롭하여 조건/무조건 모델 동시 학습[12]
- **Ye et al. (2024) - TFG**: 7개 모델, 16개 작업, 40개 대상에서 통합 학습-무관 가이던스 프레임워크 제시, 평균 8.5% 성능 개선[3][4]

**Universal Guidance (2024):**

임의의 differentiable 외부 predictor(분류기, loss 함수, CLIP 등)를 diffusion에 통합하여 task-specific 제어 가능, 40개 작업에서 8.5% 평균 개선 달성[2]

#### 5.2 주의 메커니즘 연구

**Attention in Diffusion Models: A Survey (2025):**

최신 조사는 diffusion 모델의 주의 메커니즘을 5단계로 분류:[5]

1. **구조적 레벨**: 레이어, 해상도, 타입(self/cross-attention)
2. **조직적 레벨**: 주의 활성화, 마스킹, 정규화
3. **시간적 레벨**: 타임스텝 내 동작
4. **상호작용 레벨**: 다양한 모달리티 간 상호작용

이 분류 프레임워크에서 SAG는 **Attention Score-Driven Guidance**의 대표 사례로 위치합니다.[5]

**Understanding Attention Mechanism in Video Diffusion (2024):**

U-Net의 다양한 레이어에서 주의 맵의 역할을 분석:
- 시작/최종 레이어: 에너지 함수 근처
- 중간 병목 레이어: 구조 정보 포함

SAG의 자기-주의 맵 집계 전략이 이러한 계층별 특성을 활용함을 시사합니다.[13]

#### 5.3 일반화 성능 분석

**Critical Windows 이론 (2024):**

Diffusion 모델이 특정 타임스텝 구간에서만 특정 특성(class, color)을 생성하는 현상을 이론화합니다:[14]

- 강하게 log-concave 데이터에서 critical window를 증명
- SAG의 타임스텝별 마스크 전략과 상호보완 가능

**The Emergence of Reproducibility and Generalizability (2024):**

Diffusion 모델이 학습 데이터셋 크기에 따라 2가지 정권 존재:[15]

1. **암기 정권** (작은 데이터셋): 학습 분포에 과적합
2. **일반화 정권** (큰 데이터셋): 기본 데이터 분포 학습

이는 SAG가 어떤 정권에서 더 효과적인지에 대한 연구 기회를 제시합니다.

#### 5.4 조건부 생성의 최신 동향

**Model-Guidance (2025):**

기존 CFG를 넘어선 새로운 학습 목표 제시 - 데이터 분포 모델링 + 조건의 사후 확률 통합:[16]

$$\text{목표: } p_\theta(x|c) \text{ 대신 조건의 사후 확률 포함}$$

이는 SAG의 조건-무관성과 상이한 접근이지만, 둘 다 기존 CFG의 한계를 극복하려는 시도입니다.

#### 5.5 특정 도메인에서의 확산 모델 활용

**구조 예측 (2024):**

- **Geometry-to-Flow Diffusion**: 복잡한 기하학적 조건에 적응하는 조건부 diffusion 모델 개발
- SAG의 내부 정보 활용 원리를 구조 예측 도메인에 확장 가능[17]

**강화학습 (2024):**

- **DIAR (Diffusion-model-guided Implicit Q-learning)**: Diffusion 모델을 정책 생성기로 사용하여 OOD RL 해결[18]
- **MODULI**: 다중-목표 RL에서 OOD 선호도 일반화, diffusion 기반 슬라이딩 가이던스 제안[19]

***

### 결론

**"Improving Sample Quality of Diffusion Models Using Self-Attention Guidance"**는 diffusion 모델의 생성 품질 개선에 있어 **패러다임 전환**을 이루었습니다. 기존의 외부 조건 의존적 가이던스에서 벗어나 **내부 자기-주의 정보**를 활용함으로써, 사전 학습 모델의 직접적 개선을 가능하게 했습니다.

**주요 성과:**
- 무조건 생성에서 23% FID 개선
- 다양한 아키텍처(ADM, IDDPM, Stable Diffusion, DiT)에서 일관된 효과
- 기존 가이던스 방법과의 직교성 및 시너지 효과

**현재 한계:**
- 모드 다양성 감소 (Recall)
- 계산 오버헤드 (72% 증가)
- 하이퍼파라미터 세팅의 복잡성

**미래 방향:**
최신 연구들(2024-2025)은 SAG의 아이디어를 확장하여:
1. 통합 학습-무관 가이던스 프레임워크 (TFG, Universal Guidance)
2. 주의 메커니즘의 심층 이론화
3. OOD 일반화 성능 분석
4. 다양한 도메인(RL, 구조 예측, 의미론적 통신)으로의 적용

이는 SAG가 단순한 기술적 개선을 넘어 **diffusion 모델 연구의 새로운 방향**을 제시했음을 의미합니다.

***

### 참고문헌 (인용 근거)

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/2f937194-cf87-4e26-a204-3eff870fee43/2210.00939v6.pdf)
[2](https://www.emergentmind.com/topics/universal-guidance-algorithm-for-diffusion-models)
[3](https://proceedings.neurips.cc/paper_files/paper/2024/file/2818054fc6de6dacdda0f142a3475933-Paper-Conference.pdf)
[4](https://arxiv.org/abs/2409.15761)
[5](https://arxiv.org/html/2504.03738v1)
[6](https://ieeexplore.ieee.org/document/10896580/)
[7](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Towards_Understanding_Cross_and_Self-Attention_in_Stable_Diffusion_for_Text-Guided_CVPR_2024_paper.pdf)
[8](https://arxiv.org/pdf/2311.01797.pdf)
[9](https://arxiv.org/html/2411.19339v2)
[10](https://ieeexplore.ieee.org/document/10657563/)
[11](https://dl.acm.org/doi/10.1145/3696410.3714849)
[12](https://theaisummer.com/classifier-free-guidance/)
[13](https://arxiv.org/html/2504.12027v1)
[14](https://arxiv.org/abs/2403.01633)
[15](https://arxiv.org/html/2310.05264v3)
[16](https://arxiv.org/pdf/2502.12154.pdf)
[17](https://link.aps.org/doi/10.1103/8flg-k6s5)
[18](https://arxiv.org/abs/2410.11338)
[19](https://arxiv.org/abs/2408.15501)
[20](https://link.springer.com/10.1007/s10461-025-04912-7)
[21](https://edu.pubmedia.id/index.php/ptk/article/view/1603)
[22](https://www.stratfordjournals.com/journals/index.php/journal-of-education/article/view/2486)
[23](http://journal.yiigle.com/LinkIn.do?linkin_type=DOI&DOI=10.3760/cma.j.cn112338-20250307-00142)
[24](https://jpfis.unram.ac.id/index.php/GeoScienceEdu/article/view/588)
[25](https://www.esri.ie/publications/projections-of-regional-demand-and-workforce-requirements-for-general-practice-in)
[26](https://ieeexplore.ieee.org/document/10655542/)
[27](https://ejournal.stiblambangan.ac.id/index.php/munaqosyah/article/view/181)
[28](https://rjor.ro/considerations-on-the-efficiency-of-time-series-analysis-in-forecasting-new-influenza-cases-in-the-2024-2025-season/)
[29](http://arxiv.org/pdf/2412.17162.pdf)
[30](http://arxiv.org/pdf/2404.14743.pdf)
[31](https://aclanthology.org/2023.acl-long.248.pdf)
[32](https://arxiv.org/pdf/2306.01984.pdf)
[33](http://arxiv.org/pdf/2410.11795.pdf)
[34](http://arxiv.org/pdf/2306.06874.pdf)
[35](https://arxiv.org/pdf/2211.01324.pdf)
[36](https://arxiv.org/abs/2210.00939)
[37](https://hyunsooworld.tistory.com/entry/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-Improving-Sample-Quality-of-Diffusion-Models-Using-Self-Attention-Guidance-ICCV-2023)
[38](https://www.sciencedirect.com/science/article/abs/pii/S0950705125015503)
[39](https://pure.kaist.ac.kr/en/publications/improving-sample-quality-of-diffusion-models-using-self-attention)
[40](https://dmqa.korea.ac.kr/uploads/seminar/Improving%20Sampling%20Speed%20of%20Diffusion%20Models.pdf)
[41](https://openaccess.thecvf.com/content/ICCV2023/papers/Hong_Improving_Sample_Quality_of_Diffusion_Models_Using_Self-Attention_Guidance_ICCV_2023_paper.pdf)
[42](https://openreview.net/forum?id=MKvQH1ekeY)
[43](https://arxiv.org/html/2506.06085v2)
[44](https://arxiv.org/abs/2411.06308)
[45](https://ieeexplore.ieee.org/document/11018297/)
[46](https://arxiv.org/abs/2404.10312)
[47](https://arxiv.org/abs/2406.03537)
[48](https://arxiv.org/pdf/2310.08337.pdf)
[49](https://arxiv.org/pdf/2108.13624.pdf)
[50](http://arxiv.org/pdf/2106.04496.pdf)
[51](http://arxiv.org/pdf/2409.10094.pdf)
[52](https://arxiv.org/pdf/2106.03721.pdf)
[53](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/papers/Le_Bellier_Detecting_Out-Of-Distribution_Earth_Observation_Images_with_Diffusion_Models_CVPRW_2024_paper.pdf)
[54](https://www.ijcai.org/proceedings/2025/0764.pdf)
[55](https://arxiv.org/abs/2307.04726)
[56](https://openaccess.thecvf.com/content/ICCV2025/papers/Gao_Frequency-Guided_Diffusion_for_Training-Free_Text-Driven_Image_Translation_ICCV_2025_paper.pdf)
[57](https://openreview.net/forum?id=tTnFH7D1h4)
