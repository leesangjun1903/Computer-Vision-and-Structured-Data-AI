
# Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis
**저자:** Zhan Li, Zhang Chen, Zhong Li, Yi Xu (OPPO US Research Center / Portland State University)
**학회:** CVPR 2024 (pp. 8508–8520)
**arXiv:** 2312.16812

---

## 1. 핵심 주장 및 주요 기여 요약

동적 장면의 새로운 시점 합성(Novel View Synthesis)은 매력적이면서도 도전적인 문제로, 최근의 발전에도 불구하고 고해상도 사실적 결과, 실시간 렌더링, 그리고 압축된 저장 공간을 동시에 달성하는 것은 여전히 어려운 과제로 남아있었습니다.

이 문제를 해결하기 위해 본 논문은 세 가지 핵심 구성 요소로 이루어진 새로운 동적 장면 표현 방식인 **Spacetime Gaussian Feature Splatting(STG)**을 제안합니다.

### 세 가지 주요 기여

| 번호 | 기여 | 핵심 내용 |
|---|---|---|
| ① | Spacetime Gaussians (STG) | 시간적 불투명도 + 다항식 모션/회전 |
| ② | Splatted Feature Rendering | 구면 조화 함수 → 신경망 피처 대체 |
| ③ | Guided Gaussian Sampling | 학습 오류 + 깊이 기반 샘플링 |

여러 실세계 데이터셋에서의 실험 결과, 본 방법은 최첨단 렌더링 품질과 속도를 달성하면서도 압축된 저장 공간을 유지합니다. Lite 버전 모델은 8K 해상도에서 Nvidia RTX 4090 GPU로 60 FPS 렌더링을 달성합니다.

---

## 2. 해결하고자 하는 문제, 제안 방법(수식 포함), 모델 구조, 성능

### 2.1 해결하고자 하는 문제

동적 장면의 새로운 시점 합성은 NeRF와 같은 신경 렌더링 기술의 최근 발전에도 불구하고 여전히 도전적입니다. 현재의 방법들은 고해상도 사실적 결과, 실시간 렌더링, 압축 저장을 동시에 달성하는 데 어려움을 겪고 있습니다. 본 연구는 이러한 한계를 해결하는 새로운 동적 장면 표현 방식을 개발하는 것을 목표로 합니다.

---

### 2.2 제안 방법 및 수식

#### ✅ 구성요소 1: Spacetime Gaussians (STG) 공식화

4D 다이나믹스를 표현하기 위해, 본 논문은 3D Gaussian과 시간적 구성요소를 결합하여 나타나거나 사라지는 콘텐츠 및 모션/변형을 모델링하는 Spacetime Gaussians(STG)를 제안합니다. 구체적으로, 씬 내에서 나타나거나 사라지는 콘텐츠를 효과적으로 모델링하기 위해 시간적 방사형 기저 함수(Temporal Radial Basis Function)를 도입하여 시간적 불투명도를 인코딩합니다. 또한, 씬 내 모션과 변형을 모델링하기 위해 3D Gaussian의 위치와 회전을 위해 시간 조건 기반 파라메트릭 함수를 활용합니다.

**시간적 불투명도 (Temporal Opacity)**

각 Spacetime Gaussian의 시간적 불투명도 $\alpha(t)$는 다음과 같이 정의됩니다:

$$\alpha(t) = \alpha_{\text{base}} \cdot \exp\!\left(-\frac{(t - \mu_t)^2}{2\sigma_t^2}\right)$$

여기서 $\mu_t$는 시간적 중심, $\sigma_t$는 시간적 크기를 의미하며, 이 함수는 Gaussian이 등장하고 소멸하는 과정을 표현합니다.

**다항식 모션/회전 (Polynomial Motion & Rotation)**

본 접근법의 핵심은 3D Gaussian을 4D 시공간 도메인으로 확장한 Spacetime Gaussian(STG)입니다. 3D Gaussian에 시간 의존적 불투명도, 다항식으로 파라미터화된 모션과 회전을 부여합니다.

각 STG의 위치는 시간 $t$에 대한 다항식으로 파라미터화됩니다:

$$\mathbf{p}(t) = \mathbf{p}_0 + \sum_{k=1}^{n_p} \mathbf{v}_k \cdot t^k$$

회전 $\mathbf{q}(t)$도 유사하게 다항식 형태로 파라미터화됩니다:

$$\mathbf{q}(t) = \mathbf{q}_0 + \sum_{k=1}^{n_q} \mathbf{r}_k \cdot t^k$$

여기서 $n_p$, $n_q$는 각각 위치와 회전에 대한 다항식의 차수(polynomial order)를 의미합니다.

그 결과, STG는 씬 내의 정적(static), 동적(dynamic), 그리고 일시적(transient, 즉 나타나거나 사라지는) 콘텐츠를 충실하게 모델링할 수 있습니다.

---

#### ✅ 구성요소 2: Splatted Feature Rendering

모델 압축성을 높이고 시간에 따라 변하는 외관을 설명하기 위해, 본 논문은 splatted feature rendering을 제안합니다. 구체적으로, 각 Spacetime Gaussian에 대해 구면 조화 함수 계수를 저장하는 대신, 기본 색상(base color), 시점 관련 정보(view-related), 시간 관련 정보(time-related)를 인코딩하는 피처를 저장합니다.

각 Gaussian $i$의 피처 벡터 $\mathbf{f}_i$는 다음 세 가지 부분으로 구성됩니다:

$$\mathbf{f}_i = [\mathbf{f}_i^{\text{base}},\ \mathbf{f}_i^{\text{view}},\ \mathbf{f}_i^{\text{time}}]$$

렌더링 파이프라인은 다음과 같습니다:

$$\hat{\mathbf{F}} = \sum_i \alpha_i(t) \cdot \mathbf{f}_i \cdot \prod_{j<i}(1 - \alpha_j(t))$$

$$\hat{\mathbf{c}} = \text{MLP}(\hat{\mathbf{F}},\ \mathbf{d},\ t)$$

여기서 $\mathbf{d}$는 카메라 시점 방향이고, $\hat{\mathbf{c}}$는 최종 렌더링 색상입니다.

이 피처 기반 접근법은 신경망 피처와 경량 MLP를 구면 조화 함수 대신 사용함으로써, STG당 파라미터 수를 3차 구면 조화 함수의 48개 대신 9개로 줄여 표현력을 희생하지 않으면서도 모델 크기를 크게 줄입니다.

---

#### ✅ 구성요소 3: Guided Gaussian Sampling

초기화 시 Gaussian이 희박한 영역은 높은 렌더링 품질로 수렴하기 어렵습니다. 따라서 본 논문은 학습 오류와 coarse depth의 가이던스를 이용하는 Gaussian 샘플링 전략을 도입합니다. 학습 중 큰 오류를 가진 픽셀의 광선을 따라 새로운 Gaussian을 샘플링합니다.

지나치게 큰 깊이 범위에서의 샘플링을 피하기 위해, Gaussian 중심의 coarse depth map을 활용하여 더 구체적인 깊이 범위를 결정합니다. 이 depth map은 feature splatting 중에 생성되어 계산 오버헤드가 거의 없습니다. 새로운 Gaussian은 광선을 따라 깊이 범위 내에서 균일하게 샘플링됩니다. 새로운 Gaussian의 중심에 작은 노이즈를 추가합니다. 불필요한 Gaussian은 학습 스텝 이후 낮은 불투명도를 가지게 되어 가지치기됩니다.

---

### 2.3 모델 구조 (파이프라인)

본 접근법은 멀티뷰 비디오를 입력으로 받아 새로운 시점에서 렌더링을 가능하게 하는 6-DoF 비디오를 생성합니다.

```
[Input] Multi-view Video Frames
         │
         ▼
[SfM Initialization] → Sparse Point Clouds (from all timestamps)
         │
         ▼
[Spacetime Gaussian (STG) 최적화]
  ├── 3D Gaussian Attributes: 위치 μ, 공분산 Σ, 불투명도 α_base
  ├── Temporal: μ_t, σ_t (temporal RBF)
  ├── Motion: polynomial coefficients v_1,...,v_{np}
  ├── Rotation: polynomial coefficients r_1,...,r_{nq}
  └── Feature: f_base, f_view, f_time
         │
         ▼
[Splatted Feature Rendering]
  ├── Feature map 생성 (α-blending)
  └── Lightweight MLP → RGB 이미지 생성
         │
         ▼
[Guided Gaussian Sampling] (학습 오류 기반 보완)
         │
         ▼
[Output] 고해상도 실시간 Novel View
```

---

### 2.4 성능 향상

제안된 STG 방법은 Neural 3D Video, Google Immersive, Technicolor 등의 실세계 데이터셋에서 렌더링 품질, 속도, 모델 크기 측면 모두에서 최첨단 방법들을 크게 능가합니다. 예를 들어, Neural 3D Video 데이터셋에서 STG는 140 FPS를 달성하여 대부분의 이전 방법들보다 훨씬 빠르면서도 우수한 LPIPS 점수와 경쟁력 있는 PSNR/DSSIM을 제공합니다.

핵심적인 발견은 Neural 3D Video 데이터셋에서 32.05 PSNR을 유지하면서 실시간 성능(140 FPS)과 압축된 모델 크기(300 프레임에 200 MB)를 동시에 달성한다는 점입니다.

특히 다항식 모션 모델은 일부 동시 연구에서 사용된 선형 모델보다 더 표현력이 뛰어남을 보여주며, 이것이 더 높은 렌더링 품질에 기여합니다.

**Ablation Study 핵심 결과:**

시간적 불투명도를 제거한("w/o Temporal Opacity") 변형은 학습 중에 temporal radial basis function의 중심과 크기를 고정시키며, 이 변형은 성능이 크게 저하되어 시간적 불투명도의 중요성이 드러납니다.

guided sampling 없이는 SfM 포인트가 잘 커버되지 않는 원거리 영역이 학습 뷰와 새로운 뷰 모두에서 매우 흐릿하게 렌더링됩니다. 이는 그래디언트 기반 최적화와 밀도 제어만으로는 Gaussian이 이 영역으로 이동하기 어렵기 때문입니다. 반면, guided sampling을 적용하면 이 영역의 렌더링이 훈련 뷰와 새로운 뷰 모두에서 훨씬 선명해집니다.

---

### 2.5 한계점

본 표현법은 빠른 렌더링 속도를 달성하지만, on-the-fly 학습(실시간 학습)은 불가능합니다. On-the-fly 학습 지원은 다양한 스트리밍 애플리케이션에 도움이 될 수 있습니다. 이를 위해, 고급 초기화 기술을 탐색하여 학습 과정을 가속화하거나 씬별 학습의 요구 사항을 완화할 수 있습니다.

또한, 현재 방법은 멀티뷰 비디오 입력에 집중되어 있습니다. 정규화(regularization) 또는 생성 사전(generative priors)을 결합하여 단안 카메라(monocular) 설정으로 접근 방식을 적용하는 것이 유망한 방향입니다.

명시적 모션 방법들은 각 Gaussian에 모션 계수와 시간적 불투명도를 파라미터화하여 복잡한 궤적을 모델링하고 Gaussian의 수명을 제어하지만, 이러한 기술들은 일반적으로 보다 정확한 초기 위치가 필요합니다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 현재 일반화 한계

현재 표현법은 멀티뷰 비디오 입력을 필요로 하며 on-the-fly 학습이 불가능하다는 한계가 있습니다. 즉, STG는 **씬별(per-scene) 최적화** 방식이며, 새로운 씬에 대해서는 처음부터 다시 학습해야 합니다.

### 3.2 일반화를 위한 잠재적 방향성

본 논문이 자체적으로 제시하거나 연구 커뮤니티에서 도출되는 일반화 관련 확장 방향은 다음과 같습니다:

#### (a) 단안 비디오(Monocular Setting)로의 확장
현재 방법은 멀티뷰 비디오 입력에 초점을 두고 있으며, 정규화 또는 생성적 사전(generative priors)과 결합하여 단안 설정으로 접근 방식을 적용하는 것이 유망한 방향입니다.

이는 다음과 같은 방향으로 실현 가능합니다:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{render}} + \lambda_{\text{depth}} \mathcal{L}_{\text{depth}} + \lambda_{\text{flow}} \mathcal{L}_{\text{flow}} + \lambda_{\text{reg}} \mathcal{L}_{\text{reg}}$$

- $\mathcal{L}_{\text{depth}}$: 단안 깊이 추정 모델(MiDaS 등)을 통한 깊이 정규화
- $\mathcal{L}_{\text{flow}}$: 옵티컬 플로우 일관성 손실
- $\mathcal{L}_{\text{reg}}$: 모션의 부드러움을 위한 정규화 항

#### (b) 피드포워드 일반화 모델(Feed-forward Generalization)
On-the-fly 학습 지원은 다양한 스트리밍 애플리케이션에 도움이 될 수 있으며, 이를 위해 고급 초기화 기술을 탐색하여 학습 과정을 가속화하거나 씬별 학습의 요구 사항을 완화할 수 있습니다.

이는 다음과 같은 구조를 통해 실현될 수 있습니다:

$$\{\mu_i, \Sigma_i, \alpha_i, \mathbf{f}_i, \mathbf{v}_i, \mathbf{r}_i\}_{i=1}^{N} = \text{Encoder}(\{I_c, P_c\}_{c=1}^{C})$$

즉, 인코더 네트워크가 멀티뷰 이미지 $\{I_c\}$와 카메라 파라미터 $\{P_c\}$로부터 STG 속성을 직접 예측하는 구조입니다.

#### (c) 긴 시퀀스 일반화
복잡한 장기 시퀀스 모션에 대해, 시간적 불투명도 공식화를 통해 여러 STG가 각각 더 짧고 덜 복잡한 모션 세그먼트만을 처리하도록 함으로써 복잡한 장기 모션을 표현할 수 있습니다.

#### (d) Splatted Feature의 범용성
방법의 우수한 성능은 시간적 불투명도와 파라메트릭 모션/회전을 가진 Spacetime Gaussians, 그리고 splatted feature rendering이라는 새로운 구성요소에서 비롯됩니다. 특히 다항식 모션 모델은 일부 동시 연구에서 사용된 선형 모델보다 더 표현력이 뛰어남을 보여주며, 이것이 더 높은 렌더링 품질에 기여합니다.

Splatted feature rendering의 구조는 신경 피처 기반이므로, 이를 **장면 간 공유 가능한 사전 학습(pre-training)** 구조로 확장하는 것이 가능합니다:

$$\mathbf{f}_i = \text{SharedEncoder}(\mathbf{x}_i, t) + \text{SceneSpecificBias}_i$$

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

### 4.1 주요 방법론별 비교

| 방법 | 표현 방식 | 렌더링 속도 | 품질 | 저장 | 특징 |
|---|---|---|---|---|---|
| **DyNeRF** (2022) | 암묵적 NeRF + 시간 잠재 코드 | 매우 느림 (8 GPU/주) | 높음 | 크지 않음 | 최초 고품질 동적 NeRF |
| **HexPlane** (2023) | 6개의 시공간 평면 분해 | 빠름 | 높음 | ~200MB | 학습 속도 100x 향상 |
| **K-Planes** (2023) | 6개의 특징 평면 | 빠름 | 높음 | 보통 | 다해상도 쿼리 |
| **Dynamic 3DGS** (2024) | 3DGS + 모션 추적 | 빠름 | 보통 | 큼 | 시간적 노이즈 존재 |
| **4D Gaussian Splatting** (2023) | 4D 시공간 Gaussian | 빠름 | 높음 | 보통 | STG와 동시 제안 |
| **STG (본 논문)** (2024) | STG + 피처 렌더링 | **실시간 (140 FPS)** | **최고** | **압축** | 세 가지 혁신 통합 |

DyNeRF는 NeRF와 시간 조건 잠재 코드를 결합하여 동적 씬을 압축적으로 표현하며, StreamRF는 연속 프레임 간의 차이를 모델링하여 동적 씬의 학습을 가속화합니다.

HexPlane은 동적 3D 씬을 6개의 학습된 피처 평면으로 명시적으로 표현하며, 각 평면에서 추출된 벡터를 융합하여 시공간의 포인트에 대한 피처를 계산하는 매우 효율적인 방법입니다. HexPlane에 작은 MLP를 결합하여 출력 색상을 회귀하고 볼륨 렌더링을 통해 학습하면 동적 씬의 새로운 시점 합성에서 인상적인 결과를 달성하며, 이전 연구 대비 학습 시간을 100배 이상 단축합니다.

변형 필드(deformation field) 방법들은 MLP 또는 그리드를 이용하여 시간에 따른 모션을 암묵적으로 모델링합니다. 각 Gaussian 프리미티브가 동적 씬에서 지속되지만, 잠재적 모션 피처를 디코딩하는 데 상당한 계산 비용이 발생하여 느린 렌더링 속도로 이어집니다.

STG의 결과는 Dynamic 3DGS보다 더 선명하고 시간적 노이즈가 적습니다.

---

## 5. 앞으로의 연구에 미치는 영향 및 고려 사항

### 5.1 연구에 미치는 영향

#### (a) 4D 표현의 패러다임 전환
본 접근법의 핵심은 3D Gaussian을 4D 시공간 도메인으로 확장한 Spacetime Gaussian(STG)입니다. 시간 의존적 불투명도와 다항식으로 파라미터화된 모션 및 회전을 3D Gaussian에 부여합니다. 그 결과, STG는 씬 내의 정적, 동적, 일시적 콘텐츠를 충실하게 모델링할 수 있습니다.

이 패러다임은 VR/AR, 방송, 교육, 스트리밍 등의 응용 분야에서 실시간 동적 씬 렌더링의 실용화 가능성을 크게 높입니다.

#### (b) Splatted Feature Rendering의 확장 가능성
모델 압축성을 높이고 시간에 따라 변하는 외관을 설명하기 위해 splatted feature rendering을 제안합니다. 구체적으로, 각 Spacetime Gaussian은 구면 조화 함수 계수 대신 기본 색상, 시점 관련 정보, 시간 관련 정보를 인코딩하는 피처를 저장합니다.

이 접근법은 이후 연구에서 더 강력한 피처(예: CLIP 임베딩, semantic feature)와 결합하여 씬 편집, 세그멘테이션, 다운스트림 작업으로 확장될 수 있습니다.

#### (c) 후속 연구에 직접적인 영향
이후 SaRO-GS와 같은 연구들이 STG에서 영감을 받아, 동적 씬을 실시간으로 렌더링하면서 시간적 복잡성을 효과적으로 처리할 수 있는 새로운 동적 씬 표현 방식을 제안하며, 동적 영역의 재구성을 가속화하기 위한 적응적 최적화 스케줄을 제안합니다.

### 5.2 앞으로의 연구 시 고려할 점

#### 🔑 고려점 1: 단안 비디오 및 희소 뷰 일반화
현재 STG는 밀집된 멀티뷰 카메라를 필요로 합니다. 현재 방법은 멀티뷰 비디오 입력에 초점을 두고 있으며, 정규화 또는 생성적 사전(generative priors)을 결합하여 단안 설정에 적용하는 것이 유망한 방향입니다. 따라서 단안 깊이 추정, 옵티컬 플로우, diffusion 기반 사전을 통합하는 연구가 필요합니다.

#### 🔑 고려점 2: On-the-fly 및 스트리밍 학습
본 표현법은 빠른 렌더링 속도를 달성하지만 on-the-fly 학습은 불가능합니다. On-the-fly 학습 지원은 다양한 스트리밍 애플리케이션에 도움이 될 수 있습니다. 실시간 학습이 가능한 온라인 최적화 방식의 개발이 중요한 연구 과제입니다.

#### 🔑 고려점 3: 다항식 모델의 표현력 한계
STG와 같은 일부 연구들은 3D Gaussian의 시간에 따른 위치와 회전을 모델링하기 위해 다항식 함수를 적용합니다. 하지만 다항식은 고차원으로 갈수록 과적합(Overfitting) 위험이 있습니다. 이를 완화하기 위해 다음과 같은 방향을 고려해야 합니다:
- **적응적 차수 선택**: 각 Gaussian의 모션 복잡도에 따라 $n_p$, $n_q$를 동적으로 결정
- **주기성 모델**: 주기적 모션에 대해 푸리에(Fourier) 기저 함수와의 결합

$$\mathbf{p}(t) = \mathbf{p}_0 + \sum_{k=1}^{K} \left[a_k \sin\!\left(\frac{2\pi k t}{T}\right) + b_k \cos\!\left(\frac{2\pi k t}{T}\right)\right]$$

#### 🔑 고려점 4: 대규모 동적 씬으로의 확장
Neural 3D 데이터셋 학습에 24GB GPU 메모리가 필요하고, Technicolor 데이터셋 학습에는 48GB GPU 메모리가 필요하며, 이는 학습 이미지가 GPU 메모리에 로드되기 때문입니다. 대규모 야외 동적 씬에 적용하기 위한 메모리 효율적인 학습 방법, 점진적 학습(Progressive Learning), 공간 분할 전략 등의 연구가 요구됩니다.

#### 🔑 고려점 5: 초기화의 정확성 의존성
명시적 모션 방법들은 복잡한 궤적을 모델링하고 Gaussian의 수명을 제어하는 데 유용하지만, 이러한 기술들은 일반적으로 보다 정확한 초기 위치가 필요합니다. 따라서 SfM에 의존하지 않거나 더 강건한 초기화 방법(예: MVSNet 기반 깊이 초기화)의 개발이 필요합니다.

#### 🔑 고려점 6: 시맨틱 및 다운스트림 작업 통합
Spacetime Gaussian Feature Splatting의 도입은 동적 시점 합성의 중요한 발전을 나타냅니다. 렌더링 품질, 속도, 모델 압축성의 핵심 과제를 해결함으로써, 이 기술은 여러 응용 분야에서 사용자 경험을 향상시킬 준비가 되어 있습니다. 특히, semantic feature와의 결합을 통해 동적 씬의 세그멘테이션, 편집, 물리 기반 시뮬레이션 등의 응용 연구로 확장하는 것이 유망한 방향입니다.

---

## 📚 참고 자료 및 출처

| # | 자료명 | 출처 |
|---|---|---|
| 1 | **Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis** (주 논문) | arXiv:2312.16812, CVPR 2024, pp.8508–8520 |
| 2 | arXiv 논문 페이지 (v1, v2) | https://arxiv.org/abs/2312.16812 |
| 3 | 공식 프로젝트 웹사이트 | https://oppo-us-research.github.io/SpacetimeGaussians-website/ |
| 4 | 공식 GitHub 코드 저장소 | https://github.com/oppo-us-research/SpacetimeGaussians |
| 5 | CVPR 2024 공식 논문 PDF | https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Spacetime_Gaussian_Feature_Splatting_for_Real-Time_Dynamic_View_Synthesis_CVPR_2024_paper.pdf |
| 6 | CVPR 2024 공식 포스터 페이지 | https://cvpr.thecvf.com/virtual/2024/poster/31791 |
| 7 | Hugging Face 논문 페이지 | https://huggingface.co/papers/2312.16812 |
| 8 | Liner Quick Review | https://liner.com/review/spacetime-gaussian-feature-splatting-for-realtime-dynamic-view-synthesis |
| 9 | CVPR 2024 Supplemental PDF | https://openaccess.thecvf.com/content/CVPR2024/supplemental/Li_Spacetime_Gaussian_Feature_CVPR_2024_supplemental.pdf |
| 10 | arXiv HTML 버전 (v1, v2) | https://arxiv.org/html/2312.16812v1, https://arxiv.org/html/2312.16812v2 |
| 11 | IEEE Xplore 논문 페이지 | https://ieeexplore.ieee.org/document/10657623/ |
| 12 | EmergentMind 논문 분석 | https://www.emergentmind.com/papers/2312.16812 |
| 13 | **HexPlane: A Fast Representation for Dynamic Scenes** (비교 연구) | CVPR 2023, arXiv:2301.09632 |
| 14 | **K-Planes: Explicit Radiance Fields in Space, Time, and Appearance** (비교 연구) | CVPR 2023 |
| 15 | **Time-Varying 3D Gaussian Splatting Representation for Dynamic Scenes** (후속 연구) | IET Image Processing, 2026 |
| 16 | Semantic Scholar 논문 그래프 | https://www.semanticscholar.org/paper/Spacetime-Gaussian-Feature-Splatting |
