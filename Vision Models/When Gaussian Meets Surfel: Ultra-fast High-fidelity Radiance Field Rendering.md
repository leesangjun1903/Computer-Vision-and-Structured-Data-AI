
# When Gaussian Meets Surfel: Ultra-fast High-fidelity Radiance Field Rendering

---

## 1. 핵심 주장 및 주요 기여 요약

이 논문은 **Gaussian-enhanced Surfels (GESs)** 라는 이중 스케일(bi-scale) 표현 방식을 소개하며, 2D 불투명 서펠(surfel)들이 뷰 의존적 색상을 통해 장면의 거친 스케일 기하와 외형을 표현하고, 서펠 주변의 소수 3D 가우시안들이 세밀한 스케일의 외형 디테일을 보완합니다.

기본 GES 렌더링은 1080p 해상도에서 평균 **675 fps**를 달성하며, 확장 모델인 Speedy-GES는 품질 손실을 최소화하면서 **1135 fps**까지 렌더링 성능을 끌어올립니다.

### 주요 기여 항목

논문의 핵심 기여는 다음과 같습니다:
- 서펠 방사장(radiance field)과 가우시안 방사장을 결합하여 뷰 일관성을 갖는 초고속 렌더링을 달성한 최초의 표현인 GES의 제안
- 멀티뷰 이미지로부터 불투명 서펠과 가우시안을 효과적으로 최적화하는 **coarse-to-fine 최적화 방법**

또한 GES의 기본 표현은 **Mip-GES**(안티앨리어싱), **Speedy-GES**(렌더링 가속), **Compact-GES**(저장 압축), **2D-GES**(기하 재구성 향상) 등 다양한 확장으로 쉽게 적용될 수 있습니다.

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

**① 3DGS의 포핑 아티팩트(Popping Artifacts) 문제**

$\alpha$-blending은 각 픽셀에 대해 3D 가우시안들의 깊이 정렬(depth sorting)을 요구하는데, 이를 픽셀마다 정확히 수행하면 계산 비용이 매우 높습니다. 3DGS는 전체 이미지에 대해 타일 기반의 사전 정렬로 이를 근사화하며, 이러한 근사화는 높은 렌더링 속도를 유지하면서도 뷰 변경 시 패치 형태의 색상이 나타나거나 사라지는 **포핑 아티팩트**라는 뷰 비일관적 이미지를 생성할 수 있습니다.

**② 속도-품질 트레이드오프 문제**

최신 기법들은 초당 20 fps 미만으로 동작하며, 이는 기하 중심의 접근 방식(예: KinectFusion, 수백 fps)에 크게 뒤처집니다. 이 한계는 무거운 계산 부담에서 비롯되는데, 장면 모델링에 다수의 가우시안과 복잡한 반복 최적화가 필요하며 가우시안 수나 최적화 반복 횟수가 부족하면 심각한 품질 저하가 발생합니다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### ① GES의 이중 스케일 표현

각 서펠은 위치, 회전, 스케일의 기하 속성과 구면 조화 함수(SH) 계수의 외형 속성을 가진 **2D 불투명 타원**입니다. 세밀한 스케일에서는 거친 스케일 서펠 주변의 소수 3D 가우시안들이 서펠 방사장에서 잘 표현되지 않는 장면 외형을 보완하는 체적 방사장을 형성합니다. 각 가우시안은 3DGS(Kerbl et al., 2023)와 동일한 기하 및 외형 속성을 가집니다.

**서펠의 불투명도(Opacity) 함수:**

서펠이 완전히 불투명할 때의 opacity는 이진적으로 표현되지만, 최적화 과정에서는 점진적으로 불투명해지도록 조절 파라미터 $\tau$를 도입합니다.

$$
\alpha_s(\mathbf{x}) =
\begin{cases}
\mathcal{G}(\mathbf{x}; \mu_s, \Sigma_s) & \text{if } \tau < 1 \\
\mathbf{1}[\|\mathbf{x} - \mu_s\|_{\Sigma_s} \leq 1] & \text{if } \tau = 1
\end{cases}
$$

불투명 서펠 최적화의 어려움을 해결하기 위해, 최적화 과정에서 **불투명도 조절 파라미터**를 도입하여 반투명 서펠을 점진적으로 불투명하게 진화시킵니다. 파라미터가 1 미만일 때는 전체 서펠이 가우시안 분포 형태의 불투명도를 가지며, 파라미터가 증가함에 따라 서펠은 중심에서 바깥쪽으로 점점 불투명해져 결국 전체 서펠 영역에서 불투명도 1에 도달합니다.

**3D 가우시안 Splatting의 $\alpha$-blending (기존 3DGS):**

$$
C(\mathbf{r}) = \sum_{i=1}^{N} c_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)
$$

여기서 $c_i$는 $i$번째 가우시안의 색상, $\alpha_i = o_i \cdot \exp\!\left(-\frac{1}{2}(\mathbf{x} - \mu_i)^T \Sigma_i^{-1} (\mathbf{x} - \mu_i)\right)$ 이며, $o_i$는 불투명도입니다.

**GES의 최종 픽셀 색상 (두 패스 결합):**

$$
C_{\text{GES}}(\mathbf{p}) = C_s(\mathbf{p}) + \sum_{k \in \mathcal{G}(\mathbf{p})} c_k \cdot \alpha_k
$$

- $C_s(\mathbf{p})$: 서펠 래스터라이제이션으로 생성된 픽셀 $\mathbf{p}$의 색상
- $\mathcal{G}(\mathbf{p})$: 서펄 깊이 맵의 depth testing을 통과한 가우시안 집합
- $c_k, \alpha_k$: 각 가우시안의 색상 및 불투명도

#### ② 렌더링 파이프라인 (2-Pass)

GES 방사장 렌더링은 두 패스로 구성되며, 완전히 정렬이 없습니다(sorting-free). 첫 번째로, 불투명 서펠이 표준 그래픽스 파이프라인을 통해 래스터라이즈되어 색상 및 깊이 맵을 생성합니다. 두 번째로, 가우시안을 화면에 splatting할 때 depth testing을 적용하여, 픽셀별로 순서 독립적(order-independent) 방식으로 서펠 색상 맵 위에 가우시안 색상을 불투명도 가중치로 누적합니다. 각 픽셀에서 서펄 깊이 맵과의 depth testing을 통과하지 못한 가우시안은 해당 픽셀에 색상을 누적하지 않으며, 이는 서펠이 표현하는 기하에 의해 가우시안이 가려짐을 의미합니다.

#### ③ Coarse-to-Fine 최적화

GES 표현은 멀티뷰 입력 이미지로부터 coarse-to-fine 절차를 통해 효율적으로 구성되며, 먼저 서펠을 최적화한 후 서펠과 가우시안을 공동 최적화(joint optimization)합니다.

**손실 함수 (Loss Function):**

최적화는 광도계 손실(photometric loss)과 기하 정규화 항으로 구성됩니다:

$$
\mathcal{L} = \mathcal{L}_{\text{photo}} + \lambda_n \mathcal{L}_n + \lambda_d \mathcal{L}_d
$$

- $\mathcal{L}\_{\text{photo}} = (1 - \lambda_{\text{SSIM}}) \cdot \mathcal{L}\_1 + \lambda_{\text{SSIM}} \cdot \mathcal{L}_{\text{SSIM}}$: 광도계 손실 (L1 + SSIM)
- $\mathcal{L}_n$: 법선 일관성 손실 (normal consistency loss)
- $\mathcal{L}_d$: 깊이-법선 일관성 손실 (depth-normal consistency loss)

깊이-법선 일관성 손실은 서펠 최적화 시 구멍(hole)이나 스파이크(spiking) 아티팩트를 방지하기 위해 적용됩니다.

---

### 2-3. 모델 구조

핵심 관찰은 서펠 방사장이 장면 외형의 주요 부분을 포착하며, 가우시안 방사장이 세밀한 디테일로 서펠 방사장을 효과적으로 강화한다는 점입니다. 거친 스케일과 세밀한 스케일 렌더링을 결합한 GES 방사장은 최신 방법과 경쟁적인 고품질 이미지를 합성할 수 있습니다.

| 구성 요소 | 역할 | 표현 방식 |
|---|---|---|
| **2D Surfel** | Coarse-scale 기하 + 외형 | 2D 불투명 타원 + SH 계수 |
| **3D Gaussian** | Fine-scale 디테일 보완 | 3DGS 방식의 3D 가우시안 |
| **Depth Map** | 가우시안 Depth Testing 기준 | 서펄 래스터라이제이션 출력 |

**확장 모델 구조:**

기본 GES 표현은 렌더링에서 안티앨리어싱을 달성하는 Mip-GES, 렌더링 속도를 향상시키는 Speedy-GES, 저장 공간을 압축하는 Compact-GES, 그리고 3D 가우시안을 2D 가우시안으로 교체하여 더 나은 장면 기하를 재구성하는 2D-GES로 쉽게 확장될 수 있습니다.

- **Mip-GES**: 표준 MSAA(Multi-Sample Anti-Aliasing)를 서펄 래스터라이제이션에 적용
- **Speedy-GES**: Hessian pruning score를 사용하여 가우시안 수를 크게 줄여 렌더링 속도를 더욱 가속
- **Compact-GES**: 서펄과 가우시안 양쪽의 SH 계수를 해시 그리드에서 색상을 쿼리하는 방식으로 대체하고, 서펄과 가우시안의 스케일링 및 회전을 양자화하여 압축 저장
- **2D-GES**: 3D 가우시안을 2D 가우시안(Huang et al., 2024)으로 교체하여 더 나은 장면 기하 재구성

---

### 2-4. 성능 향상 및 한계

#### 성능 향상

기본 GES 렌더링은 테스트된 모든 장면에서 평균 1080p 해상도 기준 **675 fps**를 달성하며, 뷰 변경 시 포핑 아티팩트 없이 경쟁력 있는 시각 품질을 보입니다. Speedy-GES는 품질 손실을 최소화하면서 렌더링 성능을 **1135 fps**까지 향상시킵니다.

비교 대상 메서드:
GES와 그 확장들은 3DGS (Kerbl et al., 2023), SpeedySplat (Hanson et al., 2024), SortFreeGS (Hou et al., 2024), StopThePop (Radl et al., 2024) 등 최신 방사장 방법들과 비교하여 우수한 성능을 보입니다.

#### 한계점

서펄 최적화는 **무작위성에 민감**한 특성을 보입니다. 이로 인해 다음과 같은 한계가 존재합니다:

1. **초기화 의존성**: Point Cloud 품질 및 초기 서펠 배치에 최적화 결과가 민감하게 반응
2. **불투명 표면 가정**: 반투명 객체(유리, 연기 등)는 불투명 서펄로 표현하기 어려움
3. **정적 장면 한정**: 현재 GES는 동적 장면(dynamic scene)을 직접 처리하지 않음
4. **복잡한 조명 환경**: SH 기반 뷰 의존적 색상 표현은 강한 반사/굴절 표현에 한계

---

## 3. 모델의 일반화 성능 향상 가능성

### 3-1. 다양한 데이터셋 기반 일반화

다양한 데이터셋에 대한 실험은 GES가 초고속 고충실도 방사장 렌더링을 위한 설득력 있는 표현으로서 최신 기술을 앞선다는 것을 보여줍니다. 이는 특정 도메인에 국한되지 않는 일반적인 표현 능력을 시사합니다.

### 3-2. 확장 구조를 통한 일반화

기본 GES 표현은 안티앨리어싱(Mip-GES), 속도 향상(Speedy-GES), 컴팩트 저장(Compact-GES), 그리고 더 나은 기하 재구성을 위한 2D 가우시안 교체(2D-GES) 등 다양한 목적으로 쉽게 확장될 수 있습니다. 이러한 모듈성(modularity)은 다양한 응용 환경에서의 일반화 가능성을 높입니다.

### 3-3. 일반화를 위한 핵심 메커니즘

| 메커니즘 | 일반화 기여 |
|---|---|
| **이중 스케일 표현** | 거친 기하 + 세밀한 디테일로 다양한 장면 복잡도 처리 |
| **Coarse-to-fine 최적화** | 다양한 장면 구조에 대한 안정적 수렴 |
| **Depth-Normal 정규화** | 기하 일관성 강화로 미관측 뷰에서도 안정적 |
| **모듈식 확장 구조** | 목적에 따라 선택적 확장 적용 가능 |

### 3-4. 한계 및 개선 방향

- **동적 장면 일반화**: DynNeRF, 4D Gaussian Splatting 등과 결합하여 동적 장면으로 확장 가능
- **미지 장면 일반화**: MVSplat, pixelSplat 등의 feed-forward 일반화 접근법을 GES 구조와 결합하면 단일 장면 최적화를 넘어선 일반화 가능
- **도메인 일반화**: Mip-GES를 통한 다중 스케일 처리는 드론 촬영, 실내/실외 장면 등 다양한 스케일에서 안정적인 표현 가능

---

## 4. 2020년 이후 관련 최신 연구 비교 분석

| 방법 | 연도 | 표현 방식 | 렌더링 속도 | 품질 | 뷰 일관성 |
|---|---|---|---|---|---|
| **NeRF** (Mildenhall et al.) | 2020 | MLP + Volume Rendering | 매우 느림 | 높음 | 높음 |
| **Instant-NGP** (Müller et al.) | 2022 | Hash Grid + MLP | 실시간 가능 | 높음 | 높음 |
| **3DGS** (Kerbl et al.) | 2023 | 3D 가우시안 Splatting | 빠름 (~100fps) | 높음 | **낮음** (포핑) |
| **2DGS** (Huang et al.) | 2024 | 2D 가우시안 Splatting | 중간 | 높음 (기하↑) | 중간 |
| **SpeedySplat** (Hanson et al.) | 2024 | 3DGS + Hessian Pruning | 매우 빠름 | 중간 | 낮음 |
| **SortFreeGS** (Hou et al.) | 2024 | 정렬 없는 GS | 빠름 | 중간 | 중간 |
| **StopThePop** (Radl et al.) | 2024 | 정확한 정렬 GS | 중간 | 높음 | 높음 |
| **GES (본 논문)** | 2025 | Surfel + 3D Gaussian | **675~1135fps** | 높음 | **높음** |

NeRF (Mildenhall et al., 2020)와 3D Gaussian Splatting (Kerbl et al., 2023)은 자유시점 합성 연구에서 주목할 만한 발전을 이룬 대표적인 장면 표현 기법들입니다.

GES는 기존 방법들의 한계를 다음과 같이 극복합니다:
- **vs. 3DGS**: 포핑 아티팩트 제거 + 6배 이상 빠른 속도
- **vs. StopThePop**: 뷰 일관성 유지하면서도 훨씬 빠른 속도
- **vs. SpeedySplat**: 속도 경쟁력 유지하면서 높은 품질과 뷰 일관성 달성

---

## 5. 향후 연구에 미치는 영향 및 고려할 점

### 5-1. 향후 연구에 미치는 영향

1. **하이브리드 표현의 새로운 패러다임**
GESs는 최신 렌더링 품질과 기존 솔루션보다 훨씬 빠른 속도를 달성하는 새로운 이중 스케일 표현을 제시하며, 이는 명시적 기하 표현(서펄)과 암시적 외형 표현(가우시안)의 결합이라는 새로운 패러다임을 제시합니다.

2. **실시간 응용의 확대**: 675~1135 fps의 속도는 VR/AR, 실시간 3D 스트리밍, 자율주행 시뮬레이션 등 실시간성이 중요한 분야에 직접 응용 가능성을 열어줍니다.

3. **Sorting-Free 렌더링의 보편화**
GES의 정렬 없는(sorting-free) 방식은 뷰 변경 시 포핑 아티팩트를 성공적으로 회피하며, 이후 연구들이 정렬 비용 없이 뷰 일관성을 확보하는 방향으로 발전하도록 촉진할 것입니다.

4. **확장성 있는 모듈 설계**: Mip-GES, Speedy-GES, Compact-GES 등의 확장 설계는 후속 연구에서 필요에 따라 선택적으로 조합할 수 있는 모듈화된 렌더링 프레임워크를 제시합니다.

### 5-2. 향후 연구 시 고려할 점

| 고려 사항 | 세부 내용 |
|---|---|
| **동적 장면 확장** | 서펄의 시간적 변형 표현 메커니즘 연구 필요 |
| **반투명/비람버트 재질** | 현재 불투명 가정을 완화하는 확장 연구 필요 |
| **Feed-forward 일반화** | 장면별 최적화 없이 일반화된 추론 가능한 구조로 발전 |
| **초기화 안정성** | 서펄 최적화의 무작위성 민감도 극복을 위한 강건한 초기화 전략 연구 |
| **3D 재구성 정확도** | 서펄 기반 기하 표현의 정확도를 NeRF 수준으로 끌어올리는 연구 |
| **장거리 장면 처리** | Mip-GES를 활용한 대규모·야외 장면에서의 일반화 검증 |
| **경량화 배포** | Compact-GES를 모바일/엣지 디바이스에 적용하는 추가 연구 |

---

## 📚 참고 자료 (출처)

1. **arXiv 원문**: [arXiv:2504.17545](https://arxiv.org/abs/2504.17545) — *When Gaussian Meets Surfel: Ultra-fast High-fidelity Radiance Field Rendering*, Keyang Ye, Tianjia Shao, Kun Zhou (2025)
2. **ACM Digital Library**: [DOI: 10.1145/3730925](https://dl.acm.org/doi/10.1145/3730925) — *ACM Transactions on Graphics*, Vol. 44, No. 4, August 2025
3. **HTML Full Paper**: [arxiv.org/html/2504.17545v1](https://arxiv.org/html/2504.17545v1)
4. **Official Project Page**: [yessioncc.github.io/ges_web](https://yessioncc.github.io/ges_web/)
5. **GitHub 코드**: [github.com/YessionCC/GES](https://github.com/YessionCC/GES)
6. **ResearchGate**: [researchgate.net — GES paper](https://www.researchgate.net/publication/394042849_When_Gaussian_Meets_Surfel_Ultra-fast_High-fidelity_Radiance_Field_Rendering)
7. **Cool Papers**: [papers.cool/arxiv/2504.17545](https://papers.cool/arxiv/2504.17545)

> ⚠️ **정확도 안내**: 본 분석은 공개된 논문 초록, HTML 전문 및 GitHub 코드 설명을 기반으로 작성되었습니다. 손실 함수의 구체적인 하이퍼파라미터 값, 일부 내부 수식의 세부 계수 등 논문 전체 본문의 특정 수치는 확인 가능한 공개 자료 내에서만 서술하였으며, 불확실한 내용은 포함하지 않았습니다.
