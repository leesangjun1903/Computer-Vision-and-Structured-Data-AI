
# 3D Convex Splatting: Radiance Field Rendering with 3D Smooth Convexes

> **논문 정보**
> - **제목**: 3D Convex Splatting: Radiance Field Rendering with 3D Smooth Convexes
> - **저자**: Jan Held, Renaud Vandeghen, Abdullah Hamdi, Adrien Deliege, Anthony Cioppa, Silvio Giancola, Andrea Vedaldi, Bernard Ghanem, Marc Van Droogenbroeck
> - **발표**: CVPR 2025
> - **arXiv**: [2411.14974](https://arxiv.org/abs/2411.14974) (최초 제출: 2024.11.22, 최신 v3: 2025.05.25)
> - **프로젝트 페이지**: [convexsplatting.github.io](https://convexsplatting.github.io/)

---

## 1. 핵심 주장 및 주요 기여 (요약)

3D Gaussian Splatting(3DGS)과 같은 Radiance Field 재구성 기술은 가우시안 프리미티브의 합성으로 장면을 표현하여 고품질의 novel view synthesis와 빠른 렌더링을 달성했습니다. 그러나 이 논문은 가우시안 기반 방법의 근본적 한계를 지적하고 새로운 프리미티브 패러다임을 제시합니다.

### 🔑 핵심 주장

이 논문은 **3D Convex Splatting(3DCS)**라는 새로운 방법을 도입하며, 멀티뷰 이미지로부터 기하학적으로 의미 있는 Radiance Field를 모델링하기 위한 프리미티브로 **3D Smooth Convex(부드러운 볼록 도형)**를 활용합니다. Smooth Convex 형태는 가우시안보다 유연성이 높아 더 적은 프리미티브로 날카로운 엣지와 밀집 볼륨을 더 잘 표현할 수 있습니다.

### 📌 주요 기여 3가지


1. **3DCS 프리미티브 제안**: Gaussian 프리미티브의 밀집 볼륨 표현 한계를 극복하기 위해 3D Smooth Convex를 새로운 Radiance Field 표현 프리미티브로 도입.
2. **최적화 프레임워크 및 렌더링 파이프라인 개발**: 3D Smooth Convex에 대한 빠르고 미분 가능한 GPU 기반 렌더링 파이프라인을 개발하여 멀티뷰 이미지로부터 고품질 3D 장면 표현 및 빠른 렌더링 속도 달성.
3. **벤치마크 성능 초과 달성**: Mip-NeRF360, Tanks and Temples, Deep Blending 데이터셋에서 3DGS보다 적은 프리미티브로 더 높은 성능 달성.


---

## 2. 해결하고자 하는 문제, 제안 방법(수식 포함), 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

Gaussian 프리미티브의 두 가지 주요 한계가 있습니다.
1. **물리적 경계 부재**: 정의된 물리적 경계가 없어 평평한 표면 표현이나 물리적으로 의미 있는 장면 분해에 부적합합니다.
2. **날카로운 엣지 표현 불가**: 가우시안의 둥근 특성으로 인해 날카로운 엣지와 기하학적 구조 포착이 불충분하며, 각진 경계나 평평한 표면에 적응하지 못합니다.

구체적으로:
- 가우시안의 수를 크게 늘리지 않고 날카로운 엣지를 정확히 포착하기 어려워 메모리 사용량이 증가합니다.
- 평평한 표면 표현도 어려우며, 수작업 정규화기 없이는 실제 표면 주변에 불규칙하게 분산되는 경향이 있습니다.

---

### 2.2 제안 방법 및 수식

#### 2.2.1 Smooth Convex의 정의

3D Smooth Convex는 3D 공간의 점 집합 $\mathcal{P} = \{p_1, p_2, \ldots, p_K\} \subset \mathbb{R}^3$으로 정의됩니다. 각 프리미티브는 다음 파라미터들로 구성됩니다:

$$\theta = \{\mathcal{P}, \mathbf{c}, \alpha, \delta, \sigma, \mathbf{sh}\}$$

- $\mathcal{P}$: 볼록 도형을 정의하는 3D 점 집합 ($K=6$ 기본값)
- $\mathbf{c} \in \mathbb{R}^3$: 중심 위치
- $\alpha \in [0,1]$: 불투명도(opacity)
- $\delta \in \mathbb{R}^+$: **Smoothness 파라미터** (꼭짓점/엣지의 부드러움 제어)
- $\sigma \in \mathbb{R}^+$: **Sharpness 파라미터** (Radiance Field의 확산 제어)
- $\mathbf{sh}$: Spherical Harmonics 계수 (외관 표현)

#### 2.2.2 렌더링 파이프라인

3D Smooth Convex는 점 집합을 2D 카메라 평면에 투영하여 표현됩니다. 투영된 점들의 선분으로 구분된 볼록 껍질(convex hull)을 추출하고 각 선에 대한 부호 거리 함수(signed distance function)를 정의합니다.

각 선분 $l_i$에 대한 부호 거리 함수는:

$$d_i(\mathbf{x}) = \mathbf{n}_i^T \mathbf{x} - b_i$$

여기서 $\mathbf{n}_i$는 법선 벡터, $b_i$는 오프셋입니다.

이 선분들을 결합하여 smoothness $\delta$와 sharpness $\sigma$를 기반으로 각 픽셀에 대한 indicator function을 정의합니다.

Smooth Convex의 픽셀 $\mathbf{x}$에서의 opacity 함수:

$$\alpha(\mathbf{x}) = \alpha_0 \cdot \exp\left(-\sigma \cdot f_\delta\left(\min_i d_i(\mathbf{x})\right)\right)$$

여기서 $f_\delta$는 smoothness $\delta$에 의해 제어되는 부드러운 최소값 함수(softmin)입니다:

$$f_\delta\left(\{d_i\}\right) = -\delta \log\left(\sum_i \exp\left(-\frac{d_i}{\delta}\right)\right)$$

Smoothness와 Sharpness 파라미터를 통합함으로써 Smooth Convex의 곡률과 확산을 각각 제어할 수 있습니다. 이를 통해 딱딱하거나 부드럽고, 밀집하거나 확산적인 형태를 생성할 수 있습니다.

#### 2.2.3 Alpha-Compositing 기반 최종 렌더링

3DGS와 동일한 타일 기반 래스터라이저 구조를 채택하며, 깊이 기준 정렬 후 다음 alpha-compositing으로 최종 색상을 계산합니다:

$$C(\mathbf{x}) = \sum_{i=1}^{N} c_i \alpha_i(\mathbf{x}) \prod_{j=1}^{i-1} (1 - \alpha_j(\mathbf{x}))$$

여기서 $c_i$는 $i$번째 프리미티브의 색상(SH 기반), $\alpha_i(\mathbf{x})$는 픽셀 $\mathbf{x}$에서의 불투명도입니다.

#### 2.2.4 초기화 전략

각 Convex 형태는 피보나치 구 알고리즘(Fibonacci sphere algorithm)을 사용하여 포인트 클라우드의 점을 중심으로 구면 위에 균등하게 분포된 점 집합으로 초기화됩니다. 초기 구의 반지름은 포인트 클라우드에서 가장 가까운 3개의 이웃까지의 평균 거리의 1.2배로 설정됩니다.

이 적응형 초기화는 밀집한 3D 영역에는 많은 소형 Convex 형태가 포함되고, 희박한 영역은 더 큰 Convex로 표현되도록 합니다.

#### 2.2.5 손실 함수

3DGS와 동일한 포토메트릭 손실을 사용합니다:

$$\mathcal{L} = (1 - \lambda) \mathcal{L}_1 + \lambda \mathcal{L}_\text{D-SSIM}$$

여기서 $\lambda = 0.2$, $\mathcal{L}\_1$은 픽셀별 L1 손실, $\mathcal{L}_\text{D-SSIM}$은 구조적 유사성 손실입니다.

---

### 2.3 모델 구조

신경망으로 Radiance Field를 모델링하는 NeRF와 달리, 3D Smooth Convex를 직접 최적화하여 3D 장면에 효율적으로 피팅합니다. 그 결과 3DCS는 MipNeRF-360에 필적하는 시각적 충실도를 가지면서도 실시간 렌더링 속도를 달성합니다.

각 실험에서 Convex 형태 당 점의 수를 $K=6$, Spherical Harmonics 차수를 3으로 초기화하여 Convex 형태 당 총 69개의 파라미터를 사용합니다. 반면 3D Gaussian은 59개의 파라미터가 필요합니다.

3D Smooth Convex는 3D Gaussian의 렌더링 속도와 Smooth Convex의 유연한 표현력을 공유합니다.

파이프라인은 end-to-end로 미분 가능하여 렌더링된 이미지를 기반으로 Smooth Convex 프리미티브의 파라미터를 최적화할 수 있습니다.

---

### 2.4 성능 향상

효율적인 CUDA 기반 래스터라이저를 통해 3DCS는 Mip-NeRF360, Tanks and Temples, Deep Blending 벤치마크에서 3DGS보다 우수한 성능을 달성하며, 특히 **PSNR에서 최대 0.81 향상**, **LPIPS에서 0.026 향상**을 달성하면서도 높은 렌더링 속도를 유지하고 필요한 프리미티브 수를 줄입니다.

구조화된 평평한 표면과 날카로운 엣지로 구성된 실내 장면에서 볼록 형태가 구조적 이점을 가지며, 특히 실내 장면에서 3DCS는 PSNR 0.9, SSIM 0.007, LPIPS 0.023의 향상으로 3DGS를 크게 능가하며 다른 모든 가우시안 기반 방법을 초과합니다.

3DCS는 3DGS, 2DGS(2D Gaussian Splatting), GES(Generalized Exponential Splatting)와 같은 다른 렌더링 프리미티브보다 꾸준히 높은 렌더링 품질을 달성하며, 경량(lightweight) 버전은 3DGS에서 요구하는 메모리의 15%만 사용합니다.

**메모리 비교 (Tanks and Temples 기준)**:
3DCS는 3DGS가 요구하는 메모리의 약 70%만 사용합니다 (예: Tanks and Temples에서 282MB vs. 411MB).

---

### 2.5 한계

3DCS는 더 높은 시각적 품질을 제공하지만, 일반적으로 3DGS에 비해 훈련 시간이 약간 더 길고 렌더링 속도가 낮습니다. 예를 들어, Tanks and Temples 데이터셋에서 3DCS는 훈련에 60분이 소요되고 33 FPS로 렌더링하는 반면, 3DGS는 26분이 소요되고 154 FPS로 렌더링합니다. 이 트레이드오프는 볼록 형태의 향상된 표현력이 최적화와 렌더링 중 더 많은 계산 노력을 필요로 하지만, 많은 응용 분야에서 실용적인 수준임을 시사합니다.

- **동적 장면 미지원**: 현재 정적 장면에 초점을 맞추며 동적 장면 처리를 위한 확장 방법이 논의되지 않습니다.
- 실외 장면은 구조화되지 않은 표면이 많아 볼록 형태의 기하학적 이점이 실내 장면 대비 제한적입니다.
- **파라미터 수 증가**: Gaussian(59개)보다 Convex 형태 당 파라미터 수(69개)가 더 많아 프리미티브 당 계산 비용이 증가합니다.

---

## 3. 일반화 성능 향상 가능성

### 3.1 기하학적 표현의 우수성이 일반화에 미치는 영향

Smooth Convex 형태는 가우시안보다 높은 유연성을 제공하며, 더 적은 프리미티브로 날카로운 엣지와 정밀한 표면을 정확히 포착하는 밀집 볼륨을 형성할 수 있습니다. 이는 특정 데이터셋에 과적합되지 않는 일반화 능력을 향상시킵니다.

### 3.2 실내/실외 장면에서의 일반화

Mip-NeRF360 데이터셋의 실내 및 실외 장면의 비교 분석에서, 실내 장면은 구조화된 평평한 표면과 날카로운 엣지로 구성되어 있고 실외 장면은 더 비구조화된 표면을 가지는데, 이 구조적 차이가 실내 환경의 기하학적 특성을 더 잘 포착하는 볼록 형태에 유리합니다.

- **일반화 강점**: Smooth Convex는 물리적 경계를 명시적으로 표현할 수 있어 특정 장면 유형에 관계없이 구조화된 기하학을 포착하는 능력이 뛰어납니다.

### 3.3 Smoothness/Sharpness 파라미터의 적응성

Smoothness $\delta$는 꼭짓점과 엣지를 소프트에서 하드까지 특성화하고, Sharpness $\sigma$는 Radiance Field의 전환 특성을 제어합니다. 이 두 파라미터의 학습 가능한 특성은 다양한 장면 유형에 자동으로 적응하여 범용 일반화를 가능하게 합니다.

### 3.4 적응형 초기화와 일반화

적응형 초기화는 밀집한 3D 영역에는 많은 소형 Convex, 희박한 영역에는 더 큰 Convex를 배치하여 다양한 스케일과 밀도의 장면에서도 효과적으로 작동하도록 합니다.

### 3.5 경량 버전과 일반화

3DCS는 3DGS, 2DGS, GES 등의 다른 렌더링 프리미티브보다 꾸준히 높은 렌더링 품질을 달성하며, 경량 버전은 3DGS 대비 15%의 메모리만 사용하면서 비교 가능한 렌더링 품질을 유지합니다. 이 메모리 효율성은 제한된 자원 환경에서의 배포 일반화를 지원합니다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 방법 | 프리미티브 | 날카로운 엣지 | 메모리 효율 | 렌더링 속도 | 핵심 특징 |
|------|-----------|------------|------------|------------|----------|
| **NeRF** (Mildenhall 등, 2020) | Implicit MLP | ✅ 보통 | ❌ 낮음 | ❌ 느림 | 암묵적 볼륨 표현 |
| **Mip-NeRF 360** (Barron 등, 2022) | Implicit MLP | ✅ 높음 | ❌ 낮음 | ❌ 느림 | 무경계 장면 |
| **Instant-NGP** (Müller 등, 2022) | Hash Grid | ✅ 보통 | ✅ 높음 | ✅ 빠름 | 해시 인코딩 |
| **3DGS** (Kerbl 등, 2023) | 3D Gaussian | ❌ 약함 | ❌ 낮음 | ✅✅ 매우 빠름 | 타일 기반 래스터라이제이션 |
| **2DGS** (Huang 등, 2024) | 2D Gaussian | ✅ 보통 | ✅ 보통 | ✅ 빠름 | 표면 정렬 |
| **GES** (Hamdi 등, 2024) | Generalized Exponential | ✅ 보통 | ✅ 높음 | ✅ 빠름 | 일반화 지수 함수 |
| **3DCS (본 논문)** | 3D Smooth Convex | ✅✅ 높음 | ✅✅ 매우 높음 | ✅ 빠름 | 볼록 형태, $\delta$ $\sigma$ 제어 |

미분 가능한 렌더링 기술은 렌더링 파이프라인을 통한 그래디언트 계산을 가능하게 하여 이미지 관측으로부터 장면 파라미터를 최적화할 수 있게 합니다.

3DGS의 개선 방향에는 안티에일리어싱 기술, 타원체의 정확한 볼륨 렌더링, 동적 장면 모델링으로의 확장이 포함됩니다.

3D Smooth Convex의 유연한 표현이 특히 날카로운 엣지를 가진 구조화된 환경에서 Gaussian 프리미티브보다 장면 기하학과 세밀한 디테일을 더 효과적으로 포착합니다.

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5.1 연구에 미치는 영향

#### 🔷 프리미티브 패러다임의 확장
3D Convex Splatting이 고품질 장면 재구성 및 novel view synthesis의 새로운 표준이 될 수 있는 가능성을 결과가 보여줍니다. 이는 가우시안 기반 방법이 사실상 표준이었던 Splatting 분야에 새로운 연구 방향을 제시합니다.

#### 🔷 기하학적으로 의미 있는 표현으로의 이동
3DCS는 Gaussian 프리미티브의 밀집 볼륨 포착 한계를 해결하며 Radiance Field 표현을 위한 새로운 프리미티브로서의 가능성을 보여줍니다. 이는 장면 편집, 분할, 물리 시뮬레이션 등 하위 작업에 직접적인 영향을 미칩니다.

#### 🔷 경량화 연구 촉진
PSNR에서 최대 0.81, LPIPS에서 0.026의 향상을 유지하면서 필요한 프리미티브 수를 줄이는 것은 모바일/엣지 디바이스를 위한 경량화 연구에 중요한 기반을 제공합니다.

### 5.2 앞으로 연구 시 고려할 점

#### ⚠️ 1. 동적 장면으로의 확장
현재 정적 및 동적 장면 재구성 모두를 지원한다고 언급하고 있지만, 동적 장면에서 Convex 형태의 시간적 일관성을 보장하기 위한 명시적인 변형(deformation) 필드 설계가 필요합니다.

#### ⚠️ 2. 훈련 속도 트레이드오프 해결
3DCS는 훈련 시간이 3DGS에 비해 약 2배 이상 길고 렌더링 속도도 낮습니다 (예: T&T에서 33 FPS vs. 154 FPS). 향후 연구에서 더 효율적인 CUDA 커널 최적화나 적응적 점 집합 크기 조절을 통해 이를 개선해야 합니다.

#### ⚠️ 3. Convex Hull 계산의 확장성
3D Smooth Convex는 2D 카메라 평면에 투영된 점들의 볼록 껍질을 추출하고 각 선에 부호 거리 함수를 정의하는데, 점의 수 $K$가 증가할수록 볼록 껍질 계산 비용이 증가합니다. 더 복잡한 형태에 대한 확장성 연구가 필요합니다.

#### ⚠️ 4. 실외 장면에서의 일반화 개선
구조화되지 않은 표면을 가진 실외 장면에서는 볼록 형태의 구조적 이점이 상대적으로 줄어듭니다. 비구조화 장면에서도 효과적인 Convex 배치 전략 개발이 필요합니다.

#### ⚠️ 5. Sparse-View 및 Few-Shot 설정에서의 일반화
현재 논문은 충분한 멀티뷰 이미지를 기반으로 최적화를 수행하므로, 적은 수의 입력 뷰에서의 일반화 성능은 별도의 연구가 필요합니다.

#### ⚠️ 6. 장면 편집 및 의미론적 분해 활용
볼록 도형의 명확한 기하학적 경계는 장면 분할(segmentation), 객체 편집, AR/VR 응용에서 3DGS보다 우수한 기반을 제공합니다. 이 방향의 연구가 큰 기여를 할 수 있습니다.

---

## 📚 참고 자료 (출처)

| # | 출처 |
|---|------|
| 1 | **arXiv 논문 원문**: Held et al., "3D Convex Splatting: Radiance Field Rendering with 3D Smooth Convexes", arXiv:2411.14974, 2024. https://arxiv.org/abs/2411.14974 |
| 2 | **CVPR 2025 공식 논문**: openaccess.thecvf.com/content/CVPR2025/papers/Held_3D_Convex_Splatting... |
| 3 | **프로젝트 페이지**: https://convexsplatting.github.io/ |
| 4 | **HuggingFace Papers**: https://huggingface.co/papers/2411.14974 |
| 5 | **Semantic Scholar**: https://www.semanticscholar.org/paper/3D-Convex-Splatting |
| 6 | **AI Models FYI 리뷰**: https://www.aimodels.fyi/papers/arxiv/3d-convex-splatting-radiance-field-rendering-3d |
| 7 | **Liner Quick Review**: https://liner.com/review/3d-convex-splatting-radiance-field-rendering-with-3d-smooth-convexes |
| 8 | **ResearchGate**: https://www.researchgate.net/publication/386093946 |
| 9 | **CVPR 2025 포스터 페이지**: https://cvpr.thecvf.com/virtual/2025/poster/33359 |

> ⚠️ **정확도 관련 고지**: 본 답변에서 Smooth Convex opacity 수식의 일부 세부 형식은 공개된 논문 PDF 및 HTML 버전에서 확인된 파라미터 정의($\delta$, $\sigma$)를 기반으로 구성되었습니다. 완전한 수학적 유도는 CVPR 2025 공식 논문 전문(openaccess.thecvf.com)을 직접 확인하시길 권장합니다.
