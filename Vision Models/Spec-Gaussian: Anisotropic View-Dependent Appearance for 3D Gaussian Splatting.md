
# Spec-Gaussian: Anisotropic View-Dependent Appearance for 3D Gaussian Splatting

> **논문 정보**
> - **저자**: Ziyi Yang, Xinyu Gao, Yangtian Sun, Yihua Huang, Xiaoyang Lyu, Wen Zhou, Shaohui Jiao, Xiaojuan Qi, Xiaogang Jin
> - **게재**: NeurIPS 2024
> - **arXiv**: [2402.15870](https://arxiv.org/abs/2402.15870)
> - **공식 코드**: [GitHub - ingra14m/Specular-Gaussians](https://github.com/ingra14m/Specular-Gaussians)

---

## 1. 핵심 주장과 주요 기여 요약

### 1.1 핵심 주장

3D Gaussian Splatting(3D-GS)은 실시간 렌더링 및 최첨단 렌더링 품질을 달성하였으나, 정반사(specular)와 이방성(anisotropic) 성분을 정확히 모델링하는 데 어려움을 겪는다. 이 문제의 근본 원인은 구면 조화 함수(Spherical Harmonics, SH)가 고주파 정보를 표현하는 데 제한적이기 때문이다.

이를 극복하기 위해 **Spec-Gaussian**을 제안하며, 이는 각 3D Gaussian의 뷰 의존적 외관을 모델링하기 위해 SH 대신 **이방성 구면 가우시안(Anisotropic Spherical Gaussian, ASG)** appearance field를 활용한다.

### 1.2 주요 기여 (3가지 핵심 설계)

Spec-Gaussian은 세 가지 핵심 설계를 결합한다: **(1)** ASG appearance field를 이용한 새로운 3D Gaussian 표현, **(2)** 앵커 기반 기하학적 인식 3D Gaussian, **(3)** floater 제거 및 학습 효율 향상을 위한 효과적인 훈련 메커니즘.

소수의 차수의 ASG만으로도 저차 SH가 처리하지 못하는 고주파 정보를 효과적으로 모델링할 수 있으며, 이 설계를 통해 3D-GS가 정적 장면에서 이방성 및 정반사 성분을 보다 효과적으로 모델링할 수 있게 된다.

앵커 기반 방법은 희소 앵커 포인트를 사용하여 자식 Gaussian의 위치와 표현을 제어하며, 이를 통해 계층적·기하학 인식 포인트 기반 장면 표현을 구성하고 앵커 Gaussian만 저장해 스토리지 요구량을 크게 줄인다.

Coarse-to-Fine 훈련 방식은 floater를 제거하고 학습 효율을 높이도록 설계되었으며, 초기 단계에서 저해상도 렌더링을 최적화하여 불필요한 기하 구조 생성을 방지한다.

---

## 2. 상세 분석: 문제 → 방법 → 구조 → 성능 → 한계

### 2.1 해결하고자 하는 문제

3D-GS는 장면 내 정반사 성분 모델링에 어려움을 겪으며, 이는 주로 저차 SH가 이러한 시나리오에서 요구되는 고주파 정보를 포착하기 어렵기 때문이다. 결과적으로 반사와 정반사 성분이 포함된 장면 모델링이 어렵다.

구체적으로 저차 SH는 뷰 의존적 컬러를 표현하는 데 다음 한계를 가진다:

$$
c(\mathbf{d}) = \sum_{l=0}^{L} \sum_{m=-l}^{l} k_l^m Y_l^m(\mathbf{d})
$$

여기서 $Y_l^m$은 구면 조화 함수이며, 낮은 차수 $L$에서는 고주파 정보(정반사 하이라이트, 이방성 반사 등)를 충분히 표현하지 못한다.

---

### 2.2 제안하는 방법과 수식

#### 2.2.1 이방성 구면 가우시안 (ASG) 기본 수식

ASG 함수는 다음과 같이 정의된다:

$$
G(\mathbf{v}; \mathbf{z}, \mathbf{x}, \mathbf{y}, \lambda, \mu) = \exp\!\left(-\lambda(\mathbf{v} \cdot \mathbf{x})^2 - \mu(\mathbf{v} \cdot \mathbf{y})^2\right) \cdot \max(\mathbf{v} \cdot \mathbf{z}, 0)
$$

여기서:
- $\mathbf{z}$: ASG 로브의 주 방향(lobe axis)
- $\mathbf{x}, \mathbf{y}$: $\mathbf{z}$에 수직인 접선 방향 두 축
- $\lambda, \mu$: 각 축 방향의 샤프니스(sharpness) 파라미터 (이 두 값이 다를 때 이방성 표현)
- $\mathbf{v}$: 입사 방향(viewing direction)

> **핵심**: $\lambda \neq \mu$ 이면 타원형 로브가 형성되어 이방성 하이라이트를 정확히 표현 가능

ASG는 단위 구 위에 정의된 파라미터화된 함수로, 두 직교 접선 축에서 서로 다른 대역폭을 사용하여 이방성의 타원형 하이라이트 형태를 포착하며, 샤프니스와 타원율 파라미터를 통해 형태와 이방성을 정밀하게 제어한다.

#### 2.2.2 ASG Appearance Field 구성

각 3D Gaussian에 대해 다음과 같이 로컬 특징 $\mathbf{f}$를 학습하고, 디커플링 MLP $\Psi$를 통해 확산(diffuse)과 정반사(specular) 컬러를 분리 모델링한다:

$$
\mathbf{c} = \mathbf{c}_{\text{diffuse}} + \mathbf{c}_{\text{specular}}(\omega_r)
$$

$$
\mathbf{c}_{\text{specular}} = \Psi\!\left(\mathbf{f},\; \sum_{k=1}^{K} G_k(\omega_r;\, \mathbf{z}_k, \mathbf{x}_k, \mathbf{y}_k, \lambda_k, \mu_k)\right)
$$

여기서:
- $\omega_r$: 반사 방향(reflected viewing direction)
- $K$: ASG 로브의 개수
- $\Psi$: 3층, 은닉 유닛 64개의 디커플링 MLP

ASG appearance field의 feature decoupling MLP $\Psi$는 3개 레이어, 각 64개의 은닉 유닛으로 구성되며, 뷰 방향에 대한 위치 인코딩(positional encoding)은 차수 2를 사용한다.

Scaffold-GS처럼 MLP로 직접 컬러를 모델링하는 것과 달리, 확산 컬러와 정반사 컬러를 별도로 모델링하면 고주파 정보 적합 능력이 향상된다. ASG는 고주파 이방성 특징을 인코딩할 수 있으며, 디커플링 MLP는 복잡한 광학 현상을 더 정확하게 렌더링할 수 있다.

#### 2.2.3 3D Gaussian Splatting 렌더링 파이프라인 (기존)

3D-GS의 알파 블렌딩 기반 렌더링:

$$
\mathbf{C} = \sum_{i=1}^{N} \mathbf{c}_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)
$$

여기서 $\alpha_i = o_i \cdot \exp\!\left(-\frac{1}{2}\Delta\mathbf{x}_i^\top \Sigma_i^{-1} \Delta\mathbf{x}_i\right)$, $o_i$는 불투명도, $\Sigma_i$는 공분산 행렬이다.

Spec-Gaussian은 위 $\mathbf{c}_i$를 SH가 아닌 ASG appearance field로 계산하는 것이 핵심이다.

#### 2.2.4 Coarse-to-Fine 훈련 전략

Coarse-to-fine 훈련 전략은 학습 효율을 향상시키고 실세계 장면에서 과적합(overfitting)으로 인한 floater를 제거한다.

훈련 초기에는 저해상도($\downarrow$ 해상도)로 렌더링하여 전역적 기하 구조를 학습하고, 점진적으로 고해상도로 해상도를 높이며 세부 텍스처와 정반사를 정밀화한다:

$$
\mathcal{L} = (1-\lambda_{\text{D-SSIM}})\mathcal{L}_1 + \lambda_{\text{D-SSIM}}\mathcal{L}_{\text{D-SSIM}}
$$

---

### 2.3 모델 구조

```
입력: 멀티뷰 이미지 + SfM 포인트 클라우드
        │
        ▼
┌─────────────────────────────────────────────┐
│           3D Gaussian 집합                  │
│  각 Gaussian: μ, Σ, α, f (로컬 피처)        │
└──────────────┬──────────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
  c_diffuse     ASG Appearance Field
  (MLP 또는       K개의 ASG 로브(z,x,y,λ,μ)
   직접 학습)     + 뷰 방향 ω_r 입력
                       │
                       ▼
              디커플링 MLP Ψ
              (3-layer, 64-hidden)
                       │
                       ▼
                 c_specular
                       │
               c = c_diff + c_spec
                       │
                       ▼
            Alpha-blending 렌더링
                       │
                       ▼
            최종 렌더링 이미지
```

앵커 기반 가속 구조도 병행 사용:

앵커 기반 Gaussian splatting을 채용하여 스토리지 오버헤드와 렌더링에 필요한 3D Gaussian 수를 줄여 렌더링을 가속화한다. 이 방법은 다른 3D-GS 기반 방법이 달성하지 못한 로컬 정반사 하이라이트 모델링을 달성하면서 빠른 렌더링 속도를 유지한다.

---

### 2.4 성능 향상

Spec-Gaussian은 도전적인 데이터셋에서 vanilla 3D-GS 대비 **0.5~3.6 dB의 PSNR 향상**을 달성하고, 기존 방법들이 놓친 하이라이트를 포착한다.

이 방법은 NeRF, NSVF, 그리고 자체 제작한 "Anisotropic Synthetic" 데이터셋에서의 비교를 통해 복잡한 정반사 및 이방성 특징 모델링에서 우수한 성능을 보인다. 또한 3D-GS의 모든 시나리오에서 성능을 비교하여 접근 방식의 강건성을 추가로 증명한다.

이 방법은 3D-GS보다 현저히 높은 렌더링 품질을 달성하였으며, NeRF 기반 방법도 능가하였다.

또한 고차 SH(6차)와 더 많은 MLP 레이어(4층)도 3D-GS나 Scaffold-GS에서 만족스러운 결과를 얻지 못한다는 점을 실험으로 입증하여 ASG의 중요성을 강조한다.

앵커 Gaussian 없이 사용하는 버전이 더 나은 렌더링 효과를, 앵커 Gaussian을 사용하는 버전이 더 빠른 훈련과 추론을 달성한다.

---

### 2.5 한계점

ASG appearance field는 3D-GS의 정반사 및 이방성 특징 모델링 능력을 크게 향상시키지만, 각 Gaussian에 추가적인 로컬 피처가 연결되어 순수 SH 대비 추가적인 스토리지 및 연산 오버헤드를 발생시킨다. 제한된 장면에서는 100 FPS 이상의 실시간 렌더링이 가능하지만, 실세계 비제한 장면에서 ASG로 인한 상당한 스토리지 증가와 렌더링 속도 저하는 수용 불가능하다.

추가적인 한계:
- **동적 장면 미지원**: 정적 장면에 특화되어 있으며 동적 물체나 변형 가능한 장면에 대한 확장이 부재
- **역 렌더링(Inverse Rendering) 부재**: 재조명(relighting)이나 재질 분리(material decomposition)를 직접 지원하지 않음
- **비제한(Unbounded) 장면에서의 속도 저하**: 앵커 기반 구조로 일부 완화되었지만 여전히 overhead 존재

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 일반화의 핵심 요인: ASG의 구조적 귀납적 편향(Inductive Bias)

Spec-Gaussian은 각 3D Gaussian의 뷰 의존적 외관 모델링에 SH 대신 ASG appearance field를 사용함으로써, **3D Gaussian의 수를 늘리지 않고도** 정반사 및 이방성 성분을 포함한 장면을 모델링하는 능력을 크게 향상시킨다.

이는 일반화에 두 가지 방식으로 기여한다:

1. **파라미터 효율성**: 동일한 Gaussian 수로 더 복잡한 외관을 표현 → 과적합(overfitting) 위험 감소
2. **물리 기반 귀납적 편향**: ASG 로브는 실제 정반사 BRDF 로브의 구조와 유사 → 물리적으로 타당한 외관 학습 유도

### 3.2 Coarse-to-Fine 전략과 일반화

Coarse-to-Fine 훈련 방식은 3D-GS에 맞춰 설계되어 floater를 제거하고 학습 효율을 높이며, 초기 단계에서 저해상도 렌더링을 최적화하여 3D Gaussian 수를 늘릴 필요 없이 불필요한 기하 구조 생성을 방지하고 학습 과정을 정규화한다.

이는 정규화(Regularization)의 역할을 하여 새로운 뷰에 대한 일반화 성능 향상에 기여한다.

### 3.3 앵커 기반 구조와 기하학적 일반화

희소 앵커 포인트를 사용하여 자식 Gaussian의 위치와 표현을 제어하는 하이브리드 방식은 계층적이고 기하학 인식적인 포인트 기반 장면 표현을 가능하게 하며, 앵커 Gaussian만 저장하여 스토리지 요구량을 크게 줄이고 기하 품질을 향상시킨다.

기하학적으로 구조화된 표현은 다음과 같은 일반화 이점을 제공한다:
- **위치 일반화**: 새로운 카메라 위치에서도 일관된 기하 구조 예측 가능
- **조명 조건 일반화**: 확산/정반사 분리를 통해 조명이 다른 환경에서도 어느 정도 robust한 특성

### 3.4 데이터셋 다양성과 일반화

Coarse-to-fine 훈련 방식은 실세계 장면에서 floater를 제거하고 3D-GS의 학습 효율을 향상시키며, 이방성 데이터셋도 공개하여 모델의 이방성 표현 능력을 평가한다.

- **NeRF Synthetic**: 합성 데이터
- **NSVF Synthetic**: 뷰 적응형 장면
- **Mip-NeRF 360**: 실세계 무제한 장면
- **Anisotropic Synthetic (자체 제작)**: 이방성 특화 벤치마크

이 다양한 벤치마크에서의 검증은 특정 도메인에 과적합되지 않음을 보여준다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 방법 | 연도 | 표현 방식 | 정반사/이방성 처리 | 렌더링 속도 | 역 렌더링 |
|------|------|----------|----------------|-----------|---------|
| **NeRF** | 2020 | MLP + 볼륨 렌더링 | SH (저차) | 느림 (수 시간) | ✗ |
| **Ref-NeRF** | 2022 | NeRF + 반사 방향 인코딩 | Integrated Directional Enc. | 느림 | ✗ |
| **3D-GS** | 2023 | 3D Gaussian + 래스터화 | SH (저차) | 실시간 | ✗ |
| **Scaffold-GS** | 2024 | 앵커 기반 3D-GS | MLP (고주파 취약) | 실시간 | ✗ |
| **GaussianShader** | 2024 | 3D-GS + 쉐이딩 함수 | PBR 근사 + SH 잔차 | 실시간(느린 편) | 부분 |
| **Spec-Gaussian** | 2024 | 3D-GS + ASG | **ASG (고주파 가능)** | 실시간 | ✗ |
| **3DGS-DR** | 2024 | 3D-GS + 지연 렌더링 | 지연 쉐이딩 | 실시간 | 부분 |
| **GS-IR** | 2023 | 3D-GS + SH feature | SH 기반 반사 수송 | 실시간 | ✓ |

GaussianShader와 Spec-Gaussian은 주로 원거리(far-field) 또는 직접 반사를 대상으로 Gaussian 프레임워크 내에 명시적 정반사 성분을 도입한다. 지연 렌더링(deferred rendering) 방식은 기하와 쉐이딩을 추가로 분리하여 복잡한 조명 환경에서 복원 강건성을 향상시킨다.

Ref-NeRF는 Integrated Directional Encoding을, GaussianShader는 specular GGX를, Ref-GS는 spherical-mip encoding을, 3DGS-DR은 deferred rendering을 채택한다. 이 다양한 공식들은 서로 다른 복원 품질을 보이지만, 공통적으로 역 렌더링 과정에서 동일한 외관을 만들 수 있는 여러 재질-조명 조합 사이의 모호성 문제를 가진다.

### 4.1 Ref-NeRF (2022) vs Spec-Gaussian

- **Ref-NeRF**: 반사 복사량(reflected radiance)을 암묵적으로 학습하는 단순화된 BRDF를 추가하지만, 속도가 느리고 명시적 재조명을 지원하지 않는다.
- **Spec-Gaussian**: 훨씬 빠른 렌더링(실시간)을 달성하면서 ASG로 더 정교한 정반사 표현

### 4.2 GaussianShader (CVPR 2024) vs Spec-Gaussian

GaussianShader는 3D Gaussian에 단순화된 쉐이딩 함수를 적용하여 반사 표면을 포함한 장면에서 신경 렌더링을 향상시키며, 이 과정에서 주요 과제는 이산적 3D Gaussian 위에서의 정확한 법선 추정이다. 3D Gaussian의 가장 짧은 축 방향에 기반한 새로운 법선 추정 프레임워크를 제안한다.

Spec-Gaussian과의 차이:
- GaussianShader: PBR 기반 물리 분해 + 법선 추정 → 재조명 부분 지원, 노이즈 있는 법선
- Spec-Gaussian: ASG 기반 순수 외관 모델링 → 고주파 이방성 표현 우수, 재조명 미지원

---

## 5. 앞으로의 연구에 미치는 영향 및 고려사항

### 5.1 앞으로의 연구에 미치는 영향

#### (1) ASG의 3DGS 적용 패러다임 정립

3D Gaussian Splatting에서 각 3D Gaussian의 뷰 의존적 채널을 학습된 ASG 로브의 합으로 파라미터화함으로써 SH의 저주파 한계를 극복하는 패러다임이 Spec-Gaussian과 GlossGau에서 제시되었다.

이는 향후 3DGS 기반 방법에서 **외관 표현의 기본 빌딩 블록**으로 ASG가 활용될 가능성을 열어준다.

#### (2) 동적 장면 확장 가능성

이 향상은 3D GS가 정반사 및 이방성 표면을 포함하는 복잡한 시나리오를 처리할 수 있는 적용 가능성을 확장한다.

정적 장면에서의 우수한 성능은 향후 동적 Gaussian 방법(4D-GS, Deformable 3D-GS 등)과의 결합 연구를 촉진할 것이다.

#### (3) 역 렌더링 연구에 대한 시사점

Gaussian 기반 표현에서 정확한 표면 재구성과 반사 효과의 충실한 모델링을 효율적 래스터화 기반 프레임워크 내에서 동시에 지원하는 것은 여전히 미해결 문제이며, 이 어려움의 핵심은 물리적 정반사 반사의 성질과 기존 Gaussian 기반 방법에서 사용되는 표현 사이의 불일치에 있다.

ASG 기반 외관 모델링은 재질-조명 분리를 위한 더 나은 초기화 또는 정규화 항목으로 활용될 수 있다.

### 5.2 앞으로의 연구 시 고려할 점

#### (a) 역 렌더링 및 재조명 통합
현재 Spec-Gaussian은 새로운 뷰 합성(Novel View Synthesis, NVS)에 초점이 맞춰져 있으며 재조명(relighting)이나 재질 편집을 직접 지원하지 않는다. ASG appearance field와 물리 기반 렌더링(PBR) 파이프라인을 결합하여 재조명 가능한 확장 연구가 유망하다.

#### (b) 동적 장면 및 변형 가능 장면으로의 확장
이 프로젝트는 정반사 하이라이트를 포함한 장면 모델링에서 3D Gaussian Splatting을 향상시키는 것을 목표로 한다. 동적 장면(시간적 변화)에 ASG를 결합하는 4D Spec-Gaussian 방향 연구가 필요하다.

#### (c) 소수 뷰(Few-Shot) 설정에서의 일반화
ASG appearance field는 3D-GS의 정반사 및 이방성 특징 모델링 능력을 크게 향상시키지만, 각 Gaussian에 연결된 추가 로컬 피처로 인해 순수 SH 대비 추가적인 스토리지 및 연산 오버헤드를 도입한다. 소수 뷰 환경에서 과적합 방지를 위해 ASG 로브의 수나 구조에 대한 추가 정규화 기법이 필요하다.

#### (d) 무제한 야외 장면에서의 효율성
실시간 렌더링이 제한된 장면에서는 100 FPS 이상으로 달성 가능하지만, 실세계 비제한 장면에서 ASG로 인한 상당한 스토리지 증가와 렌더링 속도 저하는 수용 불가능한 수준이다. 양자화(quantization), 가지치기(pruning), 또는 신경 압축(neural compression)과의 결합이 필요하다.

#### (e) 일반화 가능한(Generalizable) 3DGS와의 결합
현재 Spec-Gaussian은 장면별(per-scene) 최적화에 의존한다. 향후에는 단일 전방 패스로 새로운 장면에 일반화할 수 있는 피드포워드(feed-forward) 구조와 ASG appearance field를 결합하는 연구가 중요하다.

#### (f) 모호성(Ambiguity) 해소
다양한 재질-조명 조합이 동일한 외관을 생성할 수 있다는 근본적인 모호성이 존재한다. 결과적으로 재구성된 재질 속성과 복원된 조명이 실제 값에서 벗어나 새로운 조명 조건 하에서 렌더링 품질이 저하될 수 있다. 이 모호성을 줄이기 위한 물리적 제약 또는 사전 정보(prior) 활용이 필수적이다.

---

## 📚 참고 자료 및 출처

| # | 자료명 | 링크/식별자 |
|---|--------|------------|
| 1 | **Spec-Gaussian** (원논문, arXiv) | arXiv:2402.15870 |
| 2 | **Spec-Gaussian** (NeurIPS 2024 공식 포스터) | neurips.cc/virtual/2024/poster/93509 |
| 3 | **Spec-Gaussian** (OpenReview, NeurIPS 2024) | openreview.net/forum?id=qDfPSWXSLt |
| 4 | **Spec-Gaussian** (공식 GitHub) | github.com/ingra14m/Specular-Gaussians |
| 5 | **Spec-Gaussian** (프로젝트 페이지) | ingra14m.github.io/Spec-Gaussian-website/ |
| 6 | **Spec-Gaussian** (HTML 전문, arXiv v1) | arxiv.org/html/2402.15870v1 |
| 7 | **Spec-Gaussian** (HTML 전문, arXiv v2) | arxiv.org/html/2402.15870v2 |
| 8 | **Semantic Scholar** 분석 | semanticscholar.org/paper/835ee411... |
| 9 | **Scaffold-GS** (CVPR 2024 Highlight) | github.com/city-super/Scaffold-GS |
| 10 | **GaussianShader** (CVPR 2024) | openaccess.thecvf.com/content/CVPR2024/... |
| 11 | **3D Gaussian Splatting with Deferred Reflection** | arxiv.org/html/2404.18454v2 |
| 12 | **Ref-DGS: Reflective Dual Gaussian Splatting** | arxiv.org/html/2603.07664v1 |
| 13 | **Recent advances in 3D Gaussian splatting** (Springer) | link.springer.com/article/10.1007/s41095-024-0436-y |
| 14 | **Anisotropic Spherical Gaussian Distribution** (Emergent Mind) | emergentmind.com/topics/anisotropic-spherical-gaussian-asg-distribution |
| 15 | **Awesome-3DGS** (GitHub survey) | github.com/qqqqqqy0227/awesome-3DGS |
| 16 | **NeurIPS 2024 Proceedings** (공식 PDF) | proceedings.neurips.cc/paper_files/paper/2024/file/708e0d6... |
| 17 | **Ziyi Yang 개인 홈페이지** | ingra14m.github.io |
