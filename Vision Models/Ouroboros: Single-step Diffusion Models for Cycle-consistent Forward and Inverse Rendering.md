
# Ouroboros: Single-step Diffusion Models for Cycle-consistent Forward and Inverse Rendering

> **논문 정보**
> - **제목**: Ouroboros: Single-step Diffusion Models for Cycle-consistent Forward and Inverse Rendering
> - **저자**: Shanlin Sun, Yifan Wang, Hanwen Zhang, Yifeng Xiong, Qin Ren, Ruogu Fang, Xiaohui Xie, Chenyu You
> - **소속**: UC Irvine, Stony Brook University, HUST, University of Florida
> - **게재**: ICCV 2025 (arXiv: 2508.14461)
> - **공식 페이지**: https://y-research-sbu.github.io/Ouroboros/
> - **GitHub**: https://github.com/Y-Research-SBU/Ouroboros

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

멀티-스텝 디퓨전 모델이 순방향(forward) 및 역방향(inverse) 렌더링을 각각 발전시켜 왔으나, 기존 접근법들은 이 두 문제를 독립적으로 다루어 **사이클 불일치(cycle inconsistency)**와 **느린 추론 속도**를 초래한다.

Ouroboros는 이 문제를 해결하기 위해 두 모델이 **서로를 강화**하는 공동 학습 패러다임을 제안합니다.

### 1.2 주요 기여 (세 가지 핵심 축)


논문의 주요 기여는 다음과 같습니다:
1. **최신 고속 디퓨전 기반 신경 렌더링 프레임워크**: 실내·외 장면 도메인 모두에 걸쳐 검증된 역방향/순방향 렌더링 프레임워크
2. **사이클 일관성 학습 방법론**: 이미지 분해(decomposition)와 합성(synthesis) 간의 일관성을 보장하며, 이질적인 합성 데이터셋과 비어노테이션 실세계 데이터 활용을 가능하게 하는 훈련 방법
3. **비디오 응용을 위한 훈련-프리 방법**: 이미지 데이터만으로 학습했음에도 비디오의 시간적 안정성을 달성하는 훈련 불필요 접근법


---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

#### 문제 1: 사이클 불일치 (Cycle Inconsistency)

독립적으로 학습된 순방향/역방향 모델의 중요한 한계는, 순차적으로 적용했을 때 분해된 속성들이 원본 이미지를 정확히 재구성하는 데 종종 실패한다는 점이다.

즉, $I \xrightarrow{\text{Inverse}} \{A, N, R, M, L\} \xrightarrow{\text{Forward}} \hat{I}$ 수행 시, $\hat{I} \neq I$ 가 되는 비일관성이 발생합니다.

#### 문제 2: 느린 추론 속도

멀티-스텝 디퓨전 모델은 수십~수백 번의 디노이징 단계를 필요로 하여 실시간 응용에 부적합합니다.

#### 문제 3: 실내/실외 도메인 간 일반화 부족

기존 방법들은 대부분 실내 혹은 실외 중 하나의 도메인에 특화되어 있었습니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### Step 1: 단일-스텝 디퓨전 파인튜닝

역방향 렌더링의 경우 모델은 이미지 $I$와 출력 intrinsic map을 나타내는 텍스트 프롬프트를 입력으로 받아 잠재 디퓨전 UNet을 파인튜닝합니다. 순방향 렌더링의 경우 연결된(concatenated) intrinsic map과 간단한 이미지 설명을 입력받아 원본 이미지를 추정합니다.

모델은 E2E 손실(E2E loss)에서 영감을 받은 단일-스텝 예측 접근법을 통해 파인튜닝됩니다. 디퓨전 모델의 UNet 컴포넌트만 훈련되어 고노이즈 입력에서 직접 디노이즈된 잠재 표현을 예측하도록 하며, 빠른 단일-스텝 추론을 보장합니다. 특정 손실 함수들이 각 intrinsic 속성에 적용되는데, 법선 맵을 위한 **각도 차이 손실(angular difference loss)**과 조도(irradiance)를 위한 **아핀-불변 손실(affine-invariant loss)**이 있습니다.

**역방향 렌더링의 태스크별 손실 함수:**

$$\mathcal{L}_{\text{normal}} = \frac{1}{N}\sum_{i=1}^{N}\arccos\left(\frac{\hat{\mathbf{n}}_i \cdot \mathbf{n}_i}{\|\hat{\mathbf{n}}_i\|\|\mathbf{n}_i\|}\right)$$

$$\mathcal{L}_{\text{irradiance}} = \min_{s, t} \|\hat{L} - (sL + t)\|_2^2$$

위 아핀-불변 손실은 조도 추정 시의 스케일/이동 모호성(scale-shift ambiguity)을 처리합니다.

**albedo, roughness, metallicity에는 일반적인 픽셀 재구성 손실:**

$$\mathcal{L}_{\text{pixel}} = \|\hat{X} - X\|_1$$

#### Step 2: 사이클 일관성 훈련 (Cycle Consistency Training)

ControlNet++와 유사하게, 조건부 이미지 이해 및 생성에서 사이클 일관성을 구현합니다. 단일-스텝 생성 프레임워크는 CycleGAN과 유사하게, 훈련 중 픽셀 공간에서 역방향/순방향 렌더링 간 사이클 일관성을 간단히 강제할 수 있게 합니다.

**사이클 일관성 손실 (End-to-End Loss):**

순방향 → 역방향 → 순방향 사이클:

$$\mathcal{L}_{\text{cycle}}^{fwd} = \|F_{\text{fwd}}\left(F_{\text{inv}}(I)\right) - I\|_1$$

역방향 → 순방향 → 역방향 사이클:

$$\mathcal{L}_{\text{cycle}}^{inv} = \|F_{\text{inv}}\left(F_{\text{fwd}}(X)\right) - X\|_1$$

전체 학습 목적함수:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{cycle}} \cdot \left(\mathcal{L}_{\text{cycle}}^{fwd} + \mathcal{L}_{\text{cycle}}^{inv}\right)$$

여기서:
- $F_{\text{inv}}$: 역방향 렌더링 모델 (이미지 → intrinsic maps)
- $F_{\text{fwd}}$: 순방향 렌더링 모델 (intrinsic maps → 이미지)
- $X = \{A, N, R, M, L\}$: albedo, normal, roughness, metallicity, irradiance

이 사이클 일관성 메커니즘은 자기지도(self-supervision)를 통해 비어노테이션 실세계 데이터를 훈련 과정에 통합하는 것을 용이하게 하여, 쌍(paired) 어노테이션이 있는 대규모 고품질 합성 렌더링에 대한 의존도를 낮춥니다.

#### Step 3: 비디오 추론 파이프라인

이미지 기반 신경 역방향 렌더링 방법을 일관성 있는 장편 비디오 intrinsic 분해로 확장하기 위해, 시공간 비디오 패치를 평탄화하고 2D 합성곱 커널을 의사 3D 커널(pseudo-3D kernels)로 확장하는 간단한 훈련-프리 접근법을 탐구합니다.

---

### 2.3 모델 구조

Ouroboros 프레임워크는 두 가지 주요 디퓨전 모델로 구성됩니다: 하나는 역방향 렌더링, 다른 하나는 순방향 렌더링을 위한 것입니다. 역방향 렌더링 모델은 이미지를 albedo, normal, roughness, metallicity, irradiance 등의 intrinsic map으로 분해합니다. 순방향 렌더링 모델은 이러한 intrinsic map과 텍스트 프롬프트로부터 이미지를 재구성합니다.

전체 구조를 도식화하면:

```
[Training Pipeline]
─────────────────────────────────────────────────────────
Annotated Data:
    I ──────────────────────────► F_inv ──► {A, N, R, M, L}
                                    │           │
                              L_task (per-channel losses)
                                    │           │
                              {A,N,R,M,L} ──► F_fwd ──► Î
                                                │
                                          L_cycle = ||Î - I||

Unannotated Real Data (MSCOCO, Flickr30k):
    I ──► F_inv ──► {Â, N̂, R̂, M̂, L̂} ──► F_fwd ──► Î
           └─────────────────────────────────────── L_cycle ──┘

[Inference Pipeline]
    RGB Image I ──► F_inv ──► {A, N, R, M, L}
                               └──────────────────► F_fwd ──► Î ≈ I
```

**베이스 아키텍처:**
- 기반: 사전학습된 잠재 디퓨전(Latent Diffusion) UNet (Stable Diffusion 계열)
- 단일-스텝 추론을 위해 E2E 파인튜닝 전략 적용
- 텍스트 프롬프트를 조건(conditioning)으로 활용

---

### 2.4 성능 향상

실험 결과 Ouroboros는 추론 시간을 크게 줄이면서 최신 기술 수준의 결과를 달성합니다. 실내·외 데이터셋에서 RGB↔X 등의 기준 모델과 비교하여 PSNR, SSIM, 각도 오차 등의 정량적 지표에서 우수한 성능을 보입니다.

시각적 비교에서 두 가지 핵심 관찰이 두드러집니다: 첫째, 사이클 일관성 단일-스텝 설계가 순방향 렌더링에서 RGB↔X보다 더 깔끔한 intrinsic 추정과 더 충실한 재구성을 생성하고, 둘째, 역방향 렌더링 모델이 그럴듯한 albedo, normal, roughness, metallicity, irradiance 예측을 유지하면서 다양한 장면 유형에 걸쳐 일반화됩니다.

사이클 훈련을 통해 irradiance의 세부적인 선명도가 향상되고, 재구성의 색상이 입력과 더 일치하게 됩니다.

---

### 2.5 한계점

공개된 정보를 기반으로 확인된 한계점:

1. **실내 특화 조도 훈련 데이터**: 모델이 Hypersim의 실내 장면에서만 irradiance를 추정하도록 훈련되었음에도, 사이클 기반 접근법이 새로운 환경으로 이해를 성공적으로 일반화했다는 것을 결과가 검증합니다. 즉, irradiance 훈련 데이터의 다양성은 여전히 제한적입니다.

2. **비디오 훈련 부재**: 비디오 디퓨전 모델을 훈련하는 것이 자연스럽지만, 이는 일반적으로 훨씬 더 큰 데이터셋, 높은 계산 비용, 긴 훈련 시간이 필요합니다. 대신, 추가 파인튜닝 없이 사전 훈련된 2D 디퓨전 모델을 활용하여 비디오 생성 능력을 달성합니다.

3. **고반사 영역 처리**: 창문, 거울 및 기타 고반사 영역에 대한 마스크를 생성하여 해당 영역이 훈련에 편향을 주지 않도록 합니다. 고반사 영역에 대한 완전한 처리는 여전히 도전 과제입니다.

---

## 3. 모델의 일반화 성능 향상 가능성

일반화 성능 향상은 Ouroboros의 가장 중요한 기여 중 하나입니다.

### 3.1 비어노테이션 실세계 데이터 활용

어노테이션된 데이터셋 외에, 사이클 훈련을 위해 MSCOCO와 Flickr30k 데이터셋에서 샘플링한 20,000개의 이미지를 활용합니다. 이 자연 이미지 컬렉션은 모델의 일반화 능력을 향상시키는 데 도움이 되는 다양한 시각적 콘텐츠를 제공합니다.

이는 **어노테이션 없는 실세계 데이터를 자기지도 방식으로 훈련에 활용**할 수 있다는 점에서 데이터 효율성과 범용성을 동시에 높입니다.

### 3.2 실내/실외 도메인 교차 일반화

모델이 Hypersim의 실내 장면에서만 irradiance를 추정하도록 훈련되었지만, 이 결과들은 사이클 기반 접근법이 새로운 환경으로의 이해를 성공적으로 일반화하였음을 검증합니다.

### 3.3 훈련-프리 비디오 일반화

Ouroboros는 훈련 없이(training-free manner) 비디오 분해로 전이될 수 있으며, 이미지 데이터로만 학습했음에도 비디오 시퀀스의 시간적 불일치를 줄이면서 고품질의 프레임별 역방향 렌더링을 유지합니다.

### 3.4 다운스트림 태스크로의 전이

단일-스텝 디퓨전 기반 오브젝트 제거 및 삽입을 위한 Ouroboros 파인튜닝에서 유망한 예비적 결과를 보여줍니다.

일반화 성능 향상을 도식화하면:

```
[일반화 메커니즘 요약]
┌─────────────────────────────────────────────────────┐
│  데이터 다양성: 합성(synthetic) + 실세계(MSCOCO, Flickr) │
│  → 도메인 갭 감소, 미보정 실세계 데이터에서도 작동     │
├─────────────────────────────────────────────────────┤
│  사이클 일관성 자기지도: 어노테이션 없는 데이터 활용   │
│  → 합성 데이터 의존도 감소                          │
├─────────────────────────────────────────────────────┤
│  훈련-프리 비디오 확장: 2D → 시공간 커널 확장         │
│  → 이미지 학습만으로 비디오 일반화 달성               │
└─────────────────────────────────────────────────────┘
```

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 논문 | 연도/학회 | 핵심 방법 | 속도 | 일관성 | 도메인 | 비고 |
|------|-----------|-----------|------|--------|--------|------|
| **NeRF** (Mildenhall et al.) | 2020/ECCV | Neural Radiance Field | 느림 | - | 단일 장면 | 역방향 렌더링의 전환점 |
| **RGB↔X** (Zeng et al.) | 2024/SIGGRAPH | 멀티-스텝 디퓨전, 분리 학습 | 느림 | ❌ | 실내/실외 | 기준 모델 |
| **IntrinsicDiffusion** (Luo et al.) | 2024/SIGGRAPH | 잠재 디퓨전 기반 intrinsic 분해 | 느림 | ❌ | 실내 중심 | 공동 추정 |
| **Uni-Renderer** | 2024→CVPR2025 | 듀얼 스트림 디퓨전, 공동 모델링 | 보통 | 부분적 | - | 단일 프레임워크 통합 |
| **DiffusionRenderer** (NVIDIA) | 2025/CVPR Oral | 비디오 디퓨전 모델 기반 | 보통 | 비디오 | 실세계 비디오 | 비디오 특화 |
| **Ouroboros (본 논문)** | 2025/ICCV | 단일-스텝 + 사이클 일관성 | **빠름** | ✅ | **실내+실외** | **비디오 훈련-프리** |

### 4.1 RGB↔X (SIGGRAPH 2024)와의 비교

Ouroboros는 속도와 정확도 모두에서 최신 기술인 RGB↔X를 능가하는 end-to-end 파인튜닝 기법을 확장합니다.

### 4.2 Uni-Renderer (CVPR 2025)와의 비교

Uni-Renderer는 렌더링과 역방향 렌더링을 단일 디퓨전 프레임워크 내에서 두 조건부 생성 태스크로 공동 모델링하는 데이터 기반 방법을 제안합니다. UniDiffuser에서 영감을 받아 두 가지 다른 시간 스케줄을 활용하고, 맞춤형 듀얼 스트리밍 모듈을 통해 두 사전 학습된 디퓨전 모델의 교차-조건화(cross-conditioning)를 달성합니다.

→ Ouroboros와 유사한 목표를 가지지만, Ouroboros는 **단일-스텝 추론**과 **사이클 일관성 손실**을 통해 더 빠른 추론 속도와 명시적인 일관성 보장을 제공합니다.

### 4.3 DiffusionRenderer (CVPR 2025 Oral, NVIDIA)와의 비교

DiffusionRenderer는 신경 역방향/순방향 렌더링을 위한 범용 방법입니다. 입력 이미지나 비디오에서 기하(geometry)와 재질(material) 버퍼를 정확하게 추정하고, 지정된 조명 조건에서 사실적인 이미지를 생성합니다.

→ DiffusionRenderer는 비디오 디퓨전 모델을 활용하지만 멀티-스텝 추론이 필요한 반면, Ouroboros는 **단일-스텝**으로 이미지 기반 추론 후 훈련-프리로 비디오에 확장합니다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5.1 앞으로의 연구에 미치는 영향

#### (1) 사이클 일관성을 렌더링 학습에 도입하는 패러다임 확립

단일-스텝 생성 프레임워크는 CycleGAN과 유사하게, 훈련 중 픽셀 공간에서 역방향/순방향 렌더링 간 사이클 일관성을 간단히 강제할 수 있습니다. 이 사이클 일관성 메커니즘은 자기지도를 통해 비어노테이션 실세계 데이터를 훈련 과정에 통합하는 것을 가능하게 하여, 쌍 어노테이션이 있는 대규모 고품질 합성 렌더링에 대한 의존도를 낮춥니다.

이는 앞으로 **합성 데이터 의존도를 낮추는 자기지도 역방향 렌더링 연구**의 기반이 될 것입니다.

#### (2) 단일-스텝 디퓨전의 렌더링 응용 가능성 제시

모델은 최신 E2E 파인튜닝 전략에서 영감을 받은 단일-스텝 예측 접근법으로 파인튜닝됩니다.

단일-스텝 디퓨전이 품질-속도 트레이드오프에서 멀티-스텝 방법을 대체할 수 있음을 실증적으로 증명하여, 렌더링 이외의 다른 밀집 예측(dense prediction) 태스크에도 적용 가능한 방향을 제시합니다.

#### (3) 훈련-프리 비디오 확장의 새로운 접근법

Ouroboros는 훈련 없이 비디오 분해로 전이될 수 있으며, 비디오 시퀀스의 시간적 불일치를 줄이면서 고품질의 프레임별 역방향 렌더링을 유지합니다.

이는 **이미지 모델을 비디오 모델로 확장하는 비용 효율적 방법론**의 새로운 방향을 제시하며, 관련 분야(3D 재구성, 비디오 편집 등)에도 적용 가능합니다.

#### (4) 다운스트림 응용 확장

실세계 데이터의 사이클 훈련 통합이 모델의 강건성을 더욱 향상시킵니다.

이 프레임워크는 장면 편집, 역조명(relighting), 소재 편집, AR/VR 등 다양한 응용 분야로 확장될 가능성이 높습니다.

---

### 5.2 향후 연구 시 고려할 점

#### 고려 사항 1: 고반사/투명 소재 처리

현재 창문, 거울 등 고반사 영역에 마스크를 적용하여 훈련 편향을 방지하지만, 이는 해당 영역에 대한 완전한 intrinsic 추정을 포기하는 것입니다. **고반사 소재 특화 모델** 또는 **물리 기반 반사 모델** 결합 연구가 필요합니다.

#### 고려 사항 2: 비디오 전용 파인튜닝 vs. 훈련-프리의 트레이드오프

비디오 디퓨전 모델을 훈련하는 것이 자연스럽지만, 이는 일반적으로 훨씬 더 큰 데이터셋, 높은 계산 비용, 긴 훈련 시간이 필요합니다.

미래 연구에서는 훈련-프리 방법의 한계를 뛰어넘는 **경량 비디오 파인튜닝** 방법이나 **시간 일관성 손실**을 추가하는 방향을 탐색할 수 있습니다.

#### 고려 사항 3: 사이클 일관성과 물리 정확도 간의 균형

사이클 일관성 손실

$$\mathcal{L}_{\text{cycle}} = \|F_{\text{fwd}}(F_{\text{inv}}(I)) - I\|_1$$

은 지각적 일관성을 높이지만, 물리적으로 정확한 재질 분리(예: 조도와 알베도의 완전한 분리)를 보장하지는 않습니다. **물리 기반 렌더링 방정식**을 명시적으로 손실 함수에 통합하는 연구가 필요합니다:

$$L_o(\omega_o) = \int_\Omega f_r(\omega_i, \omega_o) L_i(\omega_i) (\omega_i \cdot \mathbf{n}) \, d\omega_i$$

#### 고려 사항 4: 대규모 실세계 데이터 활용 확장

MSCOCO와 Flickr30k에서 20,000개의 이미지를 사이클 훈련에 활용합니다. 이 자연 이미지 컬렉션은 모델의 일반화 능력을 향상시키는 데 도움이 되는 다양한 시각적 콘텐츠를 제공합니다.

20,000장의 비어노테이션 이미지로도 효과가 검증되었으므로, **더 대규모의 웹 크롤링 데이터**나 **멀티모달 데이터**를 활용한 확장 연구가 기대됩니다.

#### 고려 사항 5: 3D 렌더링과의 통합

단일-스텝 디퓨전 기반 오브젝트 제거 및 삽입에서 유망한 예비적 결과를 보여줍니다.

이를 NeRF, 3D Gaussian Splatting 등과 결합하여 **3D 장면 이해 및 편집** 파이프라인으로 확장하는 것이 유망한 연구 방향입니다.

---

## 참고 자료

1. **arXiv 원문**: Shanlin Sun et al., "Ouroboros: Single-step Diffusion Models for Cycle-consistent Forward and Inverse Rendering," arXiv:2508.14461, 2025. https://arxiv.org/abs/2508.14461
2. **arXiv HTML 전문**: https://arxiv.org/html/2508.14461v1
3. **공식 프로젝트 페이지**: https://y-research-sbu.github.io/Ouroboros/
4. **공식 GitHub**: https://github.com/Y-Research-SBU/Ouroboros
5. **ICCV 2025 포스터**: https://iccv.thecvf.com/virtual/2025/poster/1412
6. **EmergentMind 분석**: https://www.emergentmind.com/papers/2508.14461
7. **비교 논문 – RGB↔X** (Zeng et al.), ACM SIGGRAPH 2024
8. **비교 논문 – Uni-Renderer** (Chen et al.), arXiv:2412.15050, CVPR 2025. https://arxiv.org/abs/2412.15050
9. **비교 논문 – DiffusionRenderer** (Liang et al.), CVPR 2025 Oral. https://research.nvidia.com/labs/toronto-ai/DiffusionRenderer/
10. **관련 논문 – IntrinsicDiffusion** (Luo et al.), ACM SIGGRAPH 2024
11. **CVPR 공식 페이지 – Uni-Renderer**: https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Uni-Renderer_...
