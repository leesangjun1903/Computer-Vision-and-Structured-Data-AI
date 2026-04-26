
# NeRFLiX: High-Quality Neural View Synthesis by Learning a Degradation-Driven Inter-viewpoint MiXer

> **논문 정보**
> - **저자:** Kun Zhou, Wenbo Li, Yi Wang, Tao Hu, Nianjuan Jiang, Xiaoguang Han, Jiangbo Lu
> - **발표:** CVPR 2023 (pp. 12363–12374)
> - **arXiv:** [2303.06919](https://arxiv.org/abs/2303.06919)

---

## 1. 핵심 주장 및 주요 기여 요약

### 🔑 핵심 주장

실세계 장면에서는 잘못된 카메라 캘리브레이션 정보 및 장면 표현의 부정확성으로 인해 소스 이미지로부터 고품질 세부 정보를 복원하는 것이 여전히 어렵고, 고품질 학습 프레임이 주어지더라도 NeRF 모델이 생성한 합성 뷰는 노이즈, 블러 등의 눈에 띄는 렌더링 아티팩트에 시달린다.

이를 해결하기 위해, NeRF 기반 접근법의 합성 품질을 향상시키기 위해 NeRFLiX를 제안한다. 이는 degradation-driven inter-viewpoint mixer를 학습하는 **일반적인 NeRF-agnostic 복원 패러다임(restorer paradigm)**이다. 특히 NeRF 스타일의 degradation 모델링 접근법을 설계하고 대규모 학습 데이터를 구축하여 기존 심층 신경망이 NeRF 고유의 렌더링 아티팩트를 효과적으로 제거할 수 있는 가능성을 열었다. 또한 degradation 제거를 넘어, 고품질의 학습 이미지를 융합할 수 있는 inter-viewpoint 집계 프레임워크를 제안하여 최신 NeRF 모델의 성능을 전혀 새로운 수준으로 끌어올리고 사실적인 합성 뷰를 생성한다.

### 📌 주요 기여 요약

| 기여 | 설명 |
|---|---|
| NeRF-Agnostic 복원 | 특정 NeRF 구조에 종속되지 않는 범용 후처리 복원기 |
| NDS (NeRF-style Degradation Simulator) | NeRF 렌더링 아티팩트를 모사하는 시뮬레이터 |
| IVM (Inter-Viewpoint Mixer) | 고품질 참조 뷰에서 정보를 집계하는 네트워크 |
| 대규모 학습 데이터 구축 | LLFF-T + Vimeo90K 기반의 대규모 시뮬레이션 데이터 |

---

## 2. 해결하고자 하는 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2-1. 해결하고자 하는 문제

Neural Radiance Fields(NeRF)는 새로운 시점 합성에서 큰 성공을 보여주었으나, 실세계 장면에서 고품질 세부 정보를 복원하는 것은 여전히 어렵다. 그 이유는 잠재적으로 부정확한 캘리브레이션 정보와 장면 표현의 부정확성 때문이다. 고품질 학습 프레임이 있더라도 NeRF 모델이 합성한 새로운 시점들은 노이즈와 블러 등의 렌더링 아티팩트를 포함한다.

또한, 기존 이미지/비디오 복원 모델(BSRGAN 등)을 그대로 NeRF 렌더링 프레임에 적용하는 것은 효과가 없으며, NDS 시뮬레이션 데이터로 BSRGAN을 재학습하면 0.62dB의 성능 향상이 있었다. 이는 기존 이미지/비디오 복원 모델이 NeRF 렌더링 프레임을 향상시키지 못함을 나타내며, NeRFLiX의 필요성을 확인한다.

---

### 2-2. 제안하는 방법 (수식 포함)

NeRFLiX는 두 가지 핵심 구성 요소인 **NeRF-style 열화 시뮬레이터(NDS)**와 **inter-viewpoint mixer(IVM)**로 구성된다.

#### 🔬 (A) NeRF-style Degradation Simulator (NDS)

NDS는 다음 세 가지 주요 degradation 유형을 시뮬레이션한다:

**① Splatted Gaussian Noise (SGN)**

NeRF의 볼륨 렌더링 과정에서 발생하는 노이즈를 모방한다.

$$\hat{I}(r) = I(r) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

여기서 $I(r)$은 클린 픽셀값이고, $\sigma$는 가우시안 노이즈의 표준 편차이다.

**② Re-positioning (Ray Jittering 시뮬레이션)**

ray jittering을 시뮬레이션하기 위해 위치 재배치(re-positioning) 기법을 설계하였다. 픽셀 위치 $(i, j)$에 대해 확률 0.1로 랜덤 2D 오프셋 $\delta_i, \delta_j \in [-2, 2]$를 추가한다.

$$\hat{p}(i, j) = p(i + \delta_i, \; j + \delta_j), \quad \delta_i, \delta_j \in [-2, 2]$$

**③ Anisotropic Blur**

NeRF 합성 프레임에는 블러가 포함되어 있으며, 블러 패턴을 시뮬레이션하기 위해 비등방성 가우시안 커널을 사용한다.

$$k(\mathbf{x}; \Sigma) = \frac{1}{2\pi \sqrt{|\Sigma|}} \exp\left(-\frac{1}{2} \mathbf{x}^T \Sigma^{-1} \mathbf{x}\right)$$

여기서 $\Sigma$는 비등방성을 표현하는 $2 \times 2$ 공분산 행렬이다.

**④ Region-Adaptive Strategy (공간적 변이 적용)**

Neural Radiance Fields는 종종 불균형한 학습 뷰로 지도학습이 이루어진다. 새로운 뷰에 투영된 2D 영역들은 서로 다른 degradation 수준을 가지므로, 각 degradation을 공간적으로 변이하는 방식으로 적용한다. 이를 위해 2차원 oriented mask $M$을 정의한다.

$$\hat{D}(i, j) = M(i,j) \cdot D_\text{local}(i,j) + (1 - M(i,j)) \cdot D_\text{global}(i,j)$$

여기서 $D_\text{local}$과 $D_\text{global}$은 각각 국소 및 전역 열화 강도, $M$은 공간적 마스크를 나타낸다.

---

#### 🔬 (B) Inter-Viewpoint Mixer (IVM)

IVM은 렌더링된 뷰를 향상시키기 위해 고품질 참조 뷰(훈련 이미지)에서 관련 정보를 집계한다.

**① View Selection (뷰 선택 전략)**

NeRFLiX는 고품질 참조 뷰를 최대한 활용하기 위한 뷰 선택 전략을 개발하였다. 이 시스템은 무작위 선택에 비해 품질 향상에 가장 관련 있는 뷰를 식별할 수 있으며, 뷰 선택 전략이 유의미하게 향상된 결과를 달성함을 보여준다.

매칭 비용 $\mathcal{C}$는 다음과 같이 정의된다:

$$\mathcal{C}(V_\text{ref}, V_\text{target}) = \sum_{k} \left\| f(V_\text{ref}^{(k)}) - f(V_\text{target}^{(k)}) \right\|_2^2$$

여기서 $f(\cdot)$은 딥 피처 추출 함수이고, $V_\text{ref}^{(k)}$는 $k$번째 참조 후보 뷰이다.

**② Hybrid Recurrent Multi-view Aggregation**

대규모 시점 차이를 처리하기 위해 hybrid recurrent inter-viewpoint aggregation 네트워크를 개발하였다. pixel-wise 집계와 patch-wise 집계를 모두 사용하여 다양한 반복 횟수를 실험하였으며, 단일 집계 방식만 사용하는 모델들은 전체 구성보다 성능이 낮음을 확인하였다.

픽셀 레벨 집계 수식:

$$\hat{I}_t = \mathcal{F}_\theta\left(I_\text{render},\; \{W(I_\text{ref}^{(k)}, F_\text{flow}^{(k)})\}_{k=1}^{K}\right)$$

여기서 $W(\cdot)$은 optical flow 기반의 warping 함수, $F_\text{flow}^{(k)}$는 $k$번째 참조 뷰와 대상 뷰 사이의 flow 필드, $\mathcal{F}_\theta$는 복원 네트워크이다.

**③ 학습 손실 함수**

$$\mathcal{L} = \mathcal{L}_\text{rec} + \lambda_\text{per} \mathcal{L}_\text{per} + \lambda_\text{adv} \mathcal{L}_\text{adv}$$

여기서:
- $\mathcal{L}\_\text{rec} = \| \hat{I} - I_\text{GT} \|_1$ : L1 재구성 손실
- $\mathcal{L}_\text{per}$ : VGG 기반 perceptual loss
- $\mathcal{L}_\text{adv}$ : GAN adversarial loss

---

### 2-3. 모델 구조 (Overview)

```
NeRF 렌더링 → 렌더링된 뷰 (저품질)
                     ↓
         [NDS: NeRF-style 열화 시뮬레이터]
           ① SGN ② Re-positioning ③ A-Blur ④ RA
                     ↓ (훈련 데이터 구성)
         [IVM: Inter-Viewpoint Mixer]
           - View Selection (매칭 비용 기반)
           - Pixel-wise + Patch-wise Aggregation
           - Hybrid Recurrent Fusion
                     ↓
         향상된 고품질 합성 뷰 출력
```

학습 데이터는 LLFF-T와 Vimeo90K에서 수집하며, 인접 프레임을 원시 시퀀스로 처리한다.

---

### 2-4. 성능 향상

참조 뷰가 없는 모델(IVM-0V)이 다른 모델들에 비해 가장 낮은 PSNR 및 SSIM 값을 달성하였다. 반복 횟수를 1에서 3으로 점진적으로 증가시킴으로써 0.12dB와 0.06dB의 향상을 달성하였으나, 추가적인 66ms와 46ms의 집계 비용이 발생하였다. 또한 recurrent hybrid aggregation 전략 덕분에 IVM이 정량적 결과에서 모든 기존 모델들을 능가하여 집계 설계의 강점을 입증하였다.

NeRFLiX++로의 개선에서는, NeRFLiX++는 NeRFLiX와 비교하여 유사한 수준의 향상을 보이며, NeRF 모델의 성능을 전례 없는 수준으로 끌어올렸다. 예를 들어 Plenoxels 데이터셋에서 PSNR/SSIM/LPIPS 측면에서 0.61dB/0.025/0.054의 상당한 향상을 달성하였다.

| 모델 | 참조 뷰 수 | 성능 |
|---|---|---|
| IVM-0V | 0개 | 최저 PSNR/SSIM |
| IVM-1V | 1개 | 중간 |
| IVM-2V (기본) | 2개 | 기본 설정 |
| IVM-3V | 3개 | 최고 (비용↑) |

---

### 2-5. 한계

NeRFLiX는 기존 NeRF 모델에 대해 범용적인 향상을 달성하는 데 있어 유망한 진전을 이루었으나, 여전히 추가 탐구가 필요한 미래 방향이 있다. (1) NDS는 NeRF 열화 시뮬레이션을 위한 많은 가능한 솔루션 중 하나이다. (2) 실시간 inter-viewpoint mixer를 탐색하는 것은 흥미롭고 유용하다.

또한 고해상도 프레임을 처리하는 것은 높은 계산 비용으로 인해 여전히 비실용적이다. 이 한계를 극복하기 위해 효율적인 multi-view fusion 모듈을 갖춘 guided inter-viewpoint mixer(G-IVM)를 제안하였다.

---

## 3. 모델의 일반화 성능 향상 가능성

NeRFLiX의 가장 강력한 특징은 **NeRF-agnostic** 설계, 즉 특정 NeRF 아키텍처에 종속되지 않는 구조이다.

### 3-1. NeRF-Agnostic 설계

NeRFLiX의 방법은 다단계 scene-specific 학습이 필요한 기존 접근법들과 달리 단일 학습 프로세스를 사용한다. 새로운 장면에 직접 적용하면 학습 오버헤드가 크게 감소한다. 기존의 HR-NeRF에서는 초기 NeRF 모델과 정제 모델의 공동 최적화로 인해 두 모델이 강하게 결합되어 어느 하나를 교체할 경우 성능이 저하될 수 있다. 반면 NeRFLiX의 프레임워크는 NeRF 렌더링과 정제 단계를 효과적으로 분리(decouple)하여 기존 또는 미래의 NeRF에 대한 적응성을 높인다.

### 3-2. 대규모 데이터 기반 일반화

최종 성능은 학습 쌍의 수와 양의 상관관계를 가진다. LLFF-T 데이터만 또는 소수의 시뮬레이션 쌍(Vimeo90K의 10%)으로 학습한 IVM은 TensoRF 렌더링 결과를 향상시키지 못하였다. 이 실험은 NeRF 복원기 학습에 있어 충분한 크기의 학습 쌍의 중요성을 입증한다.

### 3-3. 일반화 성능의 정량적 검증

NeRFLiX++의 결과는 상당한 향상, 강력한 일반화 능력, 개선된 계산 효율성을 명확히 보여준다. NeRF-SR 및 제안된 NeRFLiX++의 일반화 분석에서, 도시 규모(city-scale) 장면에 대한 재학습 없이 PSNR(dB)/SSIM 지표를 사용하여 향상 능력을 직접 평가하였다.

### 3-4. 향후 일반화 확장 가능성

NeRFLiX는 후처리 방식으로 NeRF 스타일의 열화(splatted Gaussian noise, ray jittering, blur 등)를 수정함으로써 렌더링 품질을 향상시킨다. 이러한 후처리 접근법은 다음과 같은 일반화 확장 가능성을 가진다:

1. **3D Gaussian Splatting(3DGS)** 등 새로운 표현 방식으로의 확장 가능성
2. **도시 규모 장면**에서의 일반화 (재학습 없이도 가능)
3. NeRFLiX++는 더 강력한 2단계 NeRF 열화 시뮬레이터와 더 빠른 inter-viewpoint mixer를 갖추어, 계산 효율성을 크게 향상시키면서 우수한 성능을 달성한다. 특히 NeRFLiX++는 노이즈가 포함된 저해상도 NeRF 렌더링 뷰에서 photo-realistic한 초고해상도 출력을 복원할 수 있다.

---

## 4. 연구에 미치는 영향 및 앞으로 고려할 점

### 4-1. 해당 논문이 앞으로의 연구에 미치는 영향

#### ① 후처리 복원 패러다임의 확립

일부 연구는 NeRF의 효율성이나 렌더링 품질을 높이기 위한 NeRF-agnostic 플러그인 역할을 한다. 예컨대 NerfAcc는 NeRF에 효율적인 샘플링 방법을 결합하기 위해 CUDA 커널로 연산을 융합하는 유연한 API를 제공하고, NeRFLiX는 후처리 방식으로 splatted Gaussian noise, ray jittering, blur 등의 NeRF 스타일 열화를 수정하여 렌더링 품질을 향상시킨다.

#### ② NeRF 열화 모델링의 선구적 역할

NeRFLiX++(2023)는 새로운 시점 합성의 품질 향상을 위해 NeRFLiX와 NeRFLiX++를 제안했다. 최근 NeRF 연구는 렌더링 효율성 향상, 소수 뷰 합성 최적화, 렌더링 품질 향상, 자기지도 학습 등의 핵심 영역에 집중하며, 실시간 렌더링 요구 사항, 3D 재구성의 정확도와 일반화 능력 향상, 그리고 대량의 어노테이션 데이터에 대한 의존도 감소를 목표로 한다.

#### ③ 3DGS 시대에도 유효한 방향

NeRF에서 Gaussian Splatting으로 조합이 이동하고 있으나, 후처리 기반의 품질 향상 연구 방향은 지속적으로 유효하다. NeRFLiX의 degradation-driven 접근법은 3DGS 아티팩트 제거에도 응용 가능한 일반 원리를 제시한다.

#### ④ Diffusion 기반 NeRF 복원과의 시너지

Drantal-NeRF와 같은 후속 연구는 안티앨리어싱 NeRF를 위한 Diffusion 기반 복원 방법을 제시하며, 앨리어싱 아티팩트를 클린 ground truth에 추가된 열화 모델의 일종으로 간주한다. diffusion 모델에 내재된 강력한 사전 지식을 활용함으로써 저품질의 앨리어싱된 뷰를 조건으로 하여 고사실감의 안티앨리어싱 렌더링을 복원할 수 있다.

---

### 4-2. 향후 연구 시 고려할 점

#### ✅ 기술적 측면

1. **실시간 처리 가능한 경량 IVM 설계**: 실시간 inter-viewpoint mixer를 탐색하는 것은 흥미롭고 유용하다. 모바일/엣지 환경에서의 실용화를 위해 경량화 연구가 필요하다.

2. **더 정교한 NDS 설계**: NDS는 NeRF 열화 시뮬레이션을 위한 많은 가능한 솔루션 중 하나이다. 실제 렌더링 아티팩트와 시뮬레이션 간의 도메인 갭을 좁히는 연구가 필요하다.

3. **Diffusion 모델과의 결합**: ReconFusion은 희소 입력 뷰에서 고충실도 NeRF를 복원하기 위해 2D diffusion prior를 활용한다. DiffusionNeRF는 RGBD 패치 prior의 로그 기울기를 학습하기 위해 diffusion 모델을 활용한다. NeRFLiX의 inter-viewpoint mixing과 diffusion prior를 결합하는 연구가 유망하다.

4. **3DGS로의 확장**: 3D Gaussian Splatting(3DGS)은 수백만 개의 이방성 가우시안으로 장면을 표현하며, 미분 가능한 래스터라이제이션을 통해 30 FPS 이상의 사실적인 렌더링을 달성한다. NeRFLiX의 접근법을 3DGS 렌더링 아티팩트 제거에 응용하는 연구가 필요하다.

5. **동적 장면(Dynamic Scene)으로의 확장**: 현재 NeRFLiX는 정적 장면에 특화되어 있어, 시간 축이 추가된 동적 장면에서의 inter-viewpoint mixing 방법론 확장이 필요하다.

#### ✅ 일반화 측면

6. **Few-Shot 설정에서의 견고성 강화**: 입력 뷰가 극단적으로 적을 때 참조 뷰 선택 전략의 신뢰성이 저하될 수 있으며, ReconFusion은 희소 입력 뷰에서 고충실도 NeRF를 복원하기 위해 2D diffusion prior를 활용하는 반면, DiffusionNeRF는 diffusion 모델을 활용하여 장면의 기하학과 색상을 정규화한다. 이와 같은 사전 지식과 NeRFLiX를 결합하는 방향을 고려해야 한다.

7. **도메인 일반화**: 도시 규모 장면에 대한 재학습 없이도 향상 능력을 직접 평가할 수 있었지만, 의료 영상, 위성 이미지 등 완전히 다른 도메인에서의 일반화 가능성은 아직 미탐구 영역이다.

---

## 5. 2020년 이후 관련 최신 연구 비교 분석

| 연구 | 연도 | 접근 방식 | 특징 | NeRFLiX와의 비교 |
|---|---|---|---|---|
| **NeRF** (Mildenhall et al.) | 2020 | MLP 기반 암시적 표현 | 기본 모델 | 출발점 |
| **Mip-NeRF** | 2021 | 멀티스케일 표현 | 안티앨리어싱 | 아티팩트 자체를 줄이는 방향 |
| **Mip-NeRF 360** | 2022 | 비경계 장면 | 야외 장면 처리 | 내부 개선, 후처리 없음 |
| **TensoRF / Plenoxels** | 2022 | Grid 기반 표현 | 빠른 학습 | NeRFLiX의 Base 모델로 활용 |
| **NeRF-SR** | 2022 | 초해상도 | Scene-specific 학습 필요 | NeRFLiX는 재학습 불필요 |
| **NeRFLiX** | 2023 | 후처리 복원 | NeRF-agnostic | — |
| **GANeRF** | 2023 | GAN 기반 품질 향상 | GAN을 활용하여 NeRF로부터 3D 재구성의 사실감을 향상시키며, 장면의 패치 분포를 학습하는 적대적 판별기를 사용한다. | 3D 일관성은 있으나 NeRF-agnostic이 아님 |
| **3D Gaussian Splatting** | 2023 | 가우시안 기반 explicit 표현 | 실시간 렌더링 | 새로운 표현 방식, NeRFLiX 확장 대상 |
| **ReconFusion** | 2024 | Diffusion prior | 희소 입력 복원 | 2D diffusion prior를 활용하여 희소 입력 뷰에서 NeRF를 복원한다. |
| **NeRFLiX++** | 2023 | NeRFLiX 개선판 | 4K 초해상도 지원 | NeRFLiX의 직접적 후속 연구 |

---

## 📚 참고 자료 출처

1. **arXiv 원문**: Zhou et al., "NeRFLiX: High-Quality Neural View Synthesis by Learning a Degradation-Driven Inter-viewpoint MiXer," arXiv:2303.06919, March 2023. — https://arxiv.org/abs/2303.06919

2. **CVPR 2023 공식 논문**: Zhou, Kun, et al. "NeRFLix: High-Quality Neural View Synthesis by Learning a Degradation-Driven Inter-Viewpoint MiXer." *CVPR 2023*, pp. 12363-12374. — https://openaccess.thecvf.com/content/CVPR2023/html/Zhou_NeRFLix_High-Quality_Neural_View_Synthesis_by_Learning_a_Degradation-Driven_Inter-Viewpoint_CVPR_2023_paper.html

3. **IEEE Xplore**: "NeRFLiX: High-Quality Neural View Synthesis by Learning a Degradation-Driven Inter-viewpoint MiXer." — https://ieeexplore.ieee.org/document/10204371/

4. **공식 프로젝트 페이지**: https://redrock303.github.io/nerflix/

5. **공식 GitHub 구현**: https://github.com/redrock303/NeRFLiX_CVPR2023

6. **NeRFLiX++ 논문**: Zhou et al., "From NeRFLiX to NeRFLiX++: A General NeRF-Agnostic Restorer Paradigm," arXiv:2306.06388, 2023. — https://arxiv.org/abs/2306.06388

7. **ar5iv 상세 내용** (arXiv 2303.06919 HTML 버전): https://ar5iv.labs.arxiv.org/html/2303.06919

8. **ResearchGate**: "NeRFLiX: High-Quality Neural View Synthesis..." — https://www.researchgate.net/publication/369199420

9. **Neural Radiance Fields for the Real World: A Survey** (arXiv:2501.13104v1, 2025) — https://arxiv.org/html/2501.13104v1

10. **Neural Radiance Field-based Visual Rendering: A Comprehensive Review** (arXiv:2404.00714, 2024) — https://arxiv.org/html/2404.00714v1

11. **Methods and Strategies for Improving Novel View Synthesis Quality** (arXiv:2401.12451, 2024) — https://arxiv.org/html/2401.12451v1

12. **CVPR 2023 NeRF Papers 목록**: https://github.com/lif314/awesome-NeRF-papers/blob/main/NeRFs-CVPR2023.md
