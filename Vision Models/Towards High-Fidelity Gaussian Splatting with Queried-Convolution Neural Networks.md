
# Towards High-Fidelity Gaussian Splatting with Queried-Convolution Neural Networks (arXiv: 2512.12898)

> **저자:** Abhinav Kumar, Tristan Aumentado-Armstrong\*, Lazar Valkov\*, Gopal Sharma, Alex Levinshtein, Radek Grzeszczuk, Suren Kumar (Samsung Research America / Samsung Research Canada)
> **프로젝트 페이지:** https://abhi1kumar.github.io/qonvolution/
> **GitHub:** https://github.com/abhi1kumar/qonvolution

---

## 1. 핵심 주장 및 주요 기여 요약

Gaussian Splatting은 빠른 학습 속도와 실시간 렌더링으로 Novel View Synthesis(NVS) 분야에 혁신을 가져왔지만, Zip-NeRF 같은 강력한 Radiance Field 모델에 비해 복원 충실도(Fidelity)가 여전히 뒤처진다는 문제가 있다.

이 논문은 **쿼리(예: 좌표)와 이웃 정보(Neighborhood) 모두 고충실도 신호 학습에 중요하다**는 이론적 결과를 바탕으로, 합성곱(Convolution)의 이웃 속성을 활용한 단순하면서도 강력한 변형 기법인 **Queried-Convolutions(Qonvolutions)** 를 제안한다.

### 주요 기여 정리

| 기여 항목 | 내용 |
|---|---|
| 이론적 기여 | 쿼리 + 이웃 정보의 동시 활용 필요성에 대한 이론적 근거 제시 |
| 방법론적 기여 | Qonvolution 연산 및 QNN 아키텍처 제안 |
| NVS 성능 | 3DGS + QNN 조합으로 Zip-NeRF 능가 |
| 범용성 | 1D 회귀, 2D 회귀, 2D 초해상도 등 다양한 태스크에서 성능 향상 확인 |

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

신경망은 **스펙트럼 편향(Spectral Bias)** 또는 최적화 어려움으로 인해 고주파 신호 학습에 어려움을 겪는다. Fourier 인코딩 같은 기존 기법들이 이를 개선하는 데 기여했지만, 고주파 정보 처리에는 여전히 개선 여지가 있다.

구체적으로:
- 3DGS의 복원 충실도가 Zip-NeRF 같은 강력한 Radiance Field 모델에 비해 뒤처진다.
- 표준 MLP 기반 접근법은 좌표(쿼리)만 입력으로 사용하여 이웃 구조적 정보를 활용하지 못한다.

---

### 2-2. 제안하는 방법: Queried-Convolution (Qonvolution)

Qonvolution은 합성곱의 이웃 속성을 활용하는 단순하면서도 강력한 변형으로, **저주파 신호를 쿼리(좌표 등)와 합성곱하여 복잡한 고주파 신호 학습을 향상**시킨다.

Qonvolution은 저충실도 신호를 쿼리와 합성곱하여 **잔차(Residual)를 출력**하고, 이를 통해 고충실도 복원을 달성한다.

핵심 아이디어를 수식으로 표현하면 다음과 같습니다 (논문의 개념을 기반으로 정리):

#### 기존 MLP 기반 방식 (표준 NeRF 계열)

$$f_{\text{MLP}}(\mathbf{q}) = \text{MLP}(\gamma(\mathbf{q}))$$

여기서 $\mathbf{q}$는 쿼리(좌표), $\gamma$는 Positional Encoding (Fourier Feature 등)

#### Qonvolution 방식

$$y_{\text{high}} = y_{\text{low}} + \Delta y, \quad \Delta y = \text{Qonv}(y_{\text{low}}, \mathbf{q})$$

- $y_{\text{low}}$: 저충실도(저주파) 신호 (예: 3DGS 렌더링 결과)
- $\mathbf{q}$: 쿼리 (예: 픽셀 좌표)
- $\text{Qonv}(\cdot)$: Queried-Convolution 연산 → 이웃 구조를 활용하여 잔차 $\Delta y$ 예측
- $y_{\text{high}}$: 최종 고충실도 출력

#### Qonvolution 연산의 핵심

표준 MLP 기반 네트워크는 1D 좌표를 쿼리로 사용하는 반면, QNN은 **선형 레이어를 1D 합성곱 레이어로 교체**하고 쿼리에 더불어 **저주파 신호도 함께 입력**으로 받는다.

즉, 1D 기준 수식화:

$$\text{Qonv}(\mathbf{y}_{\text{low}}, \mathbf{q}) = \mathbf{W} * [\mathbf{y}_{\text{low}} \oplus \mathbf{q}]$$

- $*$: 합성곱 연산 (이웃 정보 활용)
- $\oplus$: 채널 방향 연결(concatenation) 또는 결합
- $\mathbf{W}$: 학습 가능한 합성곱 커널

> ⚠️ **주의:** 위 수식은 논문의 개념적 설명을 기반으로 재구성한 것입니다. 논문 내의 정확한 수식 표기와 세부 구현은 원문 PDF를 통해 확인하시기 바랍니다.

---

### 2-3. 모델 구조 (QNN: Qonvolution Neural Network)

QNN은 MLP 기반 아키텍처(Fourier 인코딩 포함)를 능가하여 고주파 신호를 회귀하며, 1D 쿼리와 저주파(LF) 신호를 입력으로 받아 고주파 1D 신호를 예측한다.

전체 파이프라인 구조 (개념적):

```
입력 이미지들 + 카메라 포즈
        ↓
   3D Gaussian Splatting
  (저충실도 렌더링 생성)
        ↓
   Qonvolution Neural Network (QNN)
   [저충실도 신호 + 쿼리(좌표) → 잔차 예측]
        ↓
   고충실도 최종 렌더링 출력
```

- 3DGS에 QNN을 추가함으로써 세부 사항을 충실하게 복원하고, 기존 3D Gaussian Splatting보다 높은 충실도의 합성을 달성한다.

---

### 2-4. 성능 향상

3DGS와 QNN 결합은 실제 세계 장면에서 최첨단 NVS 성능을 달성하며, **이미지 충실도에서 Zip-NeRF를 능가**하는 결과를 보인다.

또한 Qonvolution은 컴퓨터 비전 및 그래픽 커뮤니티에 중요한 다양한 고주파 학습 태스크, 즉 **1D 회귀, 2D 초해상도, 2D 이미지 회귀, NVS** 전반에서 성능을 향상시킨다.

| 태스크 | 비교 기준 | 결과 |
|---|---|---|
| NVS (실세계 장면) | Zip-NeRF | 이미지 충실도 능가 |
| NVS (실세계 장면) | 기존 3DGS | 명확한 향상 |
| 1D 회귀 | MLP + Fourier Encoding | QNN 우세 |
| 2D 회귀 | MLP 기반 | QNN 우세 |
| 2D 초해상도 | 기존 방법 | QNN 우세 |

---

### 2-5. 한계점

> ⚠️ 논문 전문에 명시된 한계 사항을 직접 확인하기 어려워, 방법론적 특성에서 추론 가능한 잠재적 한계를 기술합니다.

1. **계산 오버헤드:** QNN을 3DGS 렌더링 파이프라인 위에 추가함으로써 추가적인 네트워크 추론 비용이 발생하며, 순수 3DGS 대비 실시간성이 다소 저하될 수 있음
2. **두 단계 구조 의존성:** Qonvolution이 저충실도 신호(3DGS 출력)에 의존하므로 3DGS 초기화 품질에 민감할 수 있음
3. **평가 범위:** 주로 실세계 NVS 장면 벤치마크에서 평가되었으며, 동적 장면(Dynamic Scene)이나 희소 뷰(Sparse View) 설정에서의 성능은 추가 검증 필요

---

## 3. 모델의 일반화 성능 향상 가능성

Qonvolution은 컴퓨터 비전 및 그래픽 커뮤니티에 중요한 **다양한 고주파 학습 태스크 전반에 걸쳐** 성능을 향상시킴을 실증적으로 보여주며, 여기에는 1D 회귀, 2D 초해상도, 2D 이미지 회귀, NVS가 포함된다.

이는 일반화 성능 측면에서 매우 중요한 의미를 가집니다:

### 3-1. 태스크 범용성 (Task Generalization)

- **1D → 2D → 3D로의 확장 가능성:** 1D 회귀, 2D 이미지 처리, 3D NVS 등 차원을 넘나드는 고주파 신호 학습에서 일관된 성능 향상을 보임
- **플러그인 구조:** Qonvolution은 기존 MLP를 대체하는 방식으로, 다양한 신경 표현 학습 파이프라인에 적용 가능한 범용 모듈임

### 3-2. 이론적 일반화 기반

쿼리(좌표 등)와 이웃 정보 모두가 고충실도 신호 학습에 중요하다는 **이론적 결과**에 의해 동기 부여된 방법으로서, 이 이론이 성립하는 어떤 도메인에서든 일반화가 가능함을 시사한다.

### 3-3. 다운스트림 태스크 일반화

| 도메인 | 일반화 가능성 |
|---|---|
| 동적 장면(Dynamic NVS) | 저충실도 동적 GS 출력 위에 QNN 적용 가능 |
| 의료 영상 초해상도 | 2D 초해상도 실험에서 효과 확인 |
| 포인트 클라우드 처리 | 3D 좌표 쿼리 + 이웃 정보 조합 활용 가능 |
| 원격 탐사 / 위성 영상 | 고주파 텍스처 복원에 잠재적 적용 가능 |

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 논문 | 연도 | 방법 | 핵심 특징 | QNN과의 관계 |
|---|---|---|---|---|
| **NeRF** (Mildenhall et al.) | 2020 | MLP + Ray Marching | 최초 암묵적 표현, 고품질 but 느림 | QNN이 극복 대상 중 하나 |
| **Mip-NeRF 360** (Barron et al.) | 2022 | Multi-scale NeRF | 무경계 장면, 우수한 품질 | QNN의 비교 기준 |
| **Zip-NeRF** (Barron et al.) | 2023 | NeRF + 해시 인코딩 통합 | 매우 높은 충실도 | QNN 기반 3DGS가 Zip-NeRF를 능가하는 NVS를 달성 |
| **3DGS** (Kerbl et al.) | 2023 | 3D Gaussian + Rasterization | 실시간 렌더링, 빠른 학습 | QNN의 기반 모델 |
| **Mip-Splatting** | 2024 | 3DGS + Anti-aliasing | 다해상도 안티앨리어싱 | 고주파 측면에서 QNN과 상보적 |
| **NeuralGS** | 2025 | 3DGS + MLP 압축 | 경량화 중점 | 수백만 개 3D Gaussian 저장 비용 문제를 MLP로 압축하는 방향 |
| **Freq-Aware GS Decomp.** | 2025 | 주파수 분리 GS | 주파수 밴드별 분리 학습 | 각 서브밴드 내 일관성을 강화하고 잔차 세부 사항을 추가/제거 허용 → QNN의 잔차 학습과 유사한 동기 |
| **본 논문 (QNN)** | 2025 | 3DGS + Qonvolution | 쿼리+이웃 기반 잔차 보정 | — |

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5-1. 연구에 미치는 영향

1. **GS + 후처리 네트워크 패러다임 강화**
   - 3DGS를 저충실도 렌더러로 두고, 뒤에 경량 신경망을 붙이는 2단계 파이프라인의 효과를 입증함으로써 유사한 구조의 연구를 촉진

2. **합성곱의 재조명**
   - Fourier 인코딩 같은 기존 기법들이 고주파 정보 처리에서 개선 여지가 있음을 재확인하고, 합성곱의 이웃 활용 특성을 신경 표현 학습에 통합하는 방향성을 제시

3. **이론-실험 연계 연구 촉진**
   - 이론적 결과로 동기 부여한 후 실험적으로 검증하는 방식은 NVS 및 신경 표현 분야에서 보다 엄밀한 이론적 분석 연구를 유도할 가능성이 있음

4. **고주파 신호 학습의 통합 프레임워크 가능성**
   - 1D~3D에 이르는 다양한 차원의 고주파 신호 학습에 적용 가능한 통합 프레임워크로 발전할 가능성이 있음

### 5-2. 앞으로 연구 시 고려할 점

| 고려 사항 | 세부 내용 |
|---|---|
| **실시간성 유지** | QNN 추가에 따른 추론 지연(latency) 최소화 방안 연구 필요 (경량 Qonvolution 설계, 지식 증류 등) |
| **동적 장면 확장** | 시간 차원($t$)을 쿼리에 포함하는 Qonvolution 설계로 동적 NVS에 적용 가능성 탐구 |
| **희소 뷰 조건** | 소수의 입력 이미지만 존재할 때 과적합(overfitting) 없이 일반화하는 방안 연구 |
| **이론 확장** | 2D/3D 이웃 구조에서의 이론적 보장을 보다 엄밀하게 확장하는 연구 |
| **다른 표현과의 결합** | NeRF, 해시 인코딩, Tri-plane 등 다양한 장면 표현과 Qonvolution의 결합 효과 탐색 |
| **평가 프로토콜 표준화** | 공통 평가 프로토콜이 아직 확립되지 않은 상황에서, 공정한 비교를 위한 표준화된 벤치마킹 필요 |

---

## 📚 참고 자료 및 출처

1. **[주논문]** Kumar, A. et al., *"Towards High-Fidelity Gaussian Splatting with Queried-Convolution Neural Networks"*, arXiv:2512.12898 (2025) — https://arxiv.org/abs/2512.12898
2. **[프로젝트 페이지]** https://abhi1kumar.github.io/qonvolution/
3. **[GitHub]** https://github.com/abhi1kumar/qonvolution
4. **[OpenReview]** *"Qonvolution: Towards Learning of High-Frequency Signals with Queried Convolution"* — https://openreview.net/forum?id=j8Cz0jPvXW
5. **[관련 논문]** Barron et al., *"Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields"*, arXiv:2111.12077
6. **[관련 논문]** Chen et al., *"NeuralGS: Bridging Neural Fields and 3D Gaussian Splatting"*, arXiv:2503.23162
7. **[관련 논문]** *"Frequency-Aware Gaussian Splatting Decomposition"*, arXiv:2503.21226
8. **[Survey]** *"A Survey on 3D Gaussian Splatting"*, arXiv:2401.03890

> ⚠️ **정확도 안내:** 본 답변은 arXiv 초록, 프로젝트 페이지, OpenReview 공개 자료를 기반으로 작성되었습니다. 논문 내부의 세부 수식, 정확한 정량적 수치(PSNR, SSIM 등의 구체적 수치), 및 구현 세부 사항은 **원문 PDF** (https://arxiv.org/pdf/2512.12898) 를 직접 확인하시기 바랍니다.
