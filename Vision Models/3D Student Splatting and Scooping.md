
# 3D Student Splatting and Scooping (SSS)

> **참고 자료 출처**
> 1. Zhu, J., Yue, J., He, F., & Wang, H. (2025). *3D Student Splatting and Scooping*. CVPR 2025, pp. 21045–21054. [arXiv:2503.10148](https://arxiv.org/abs/2503.10148)
> 2. CVF Open Access: https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_3D_Student_Splatting_and_Scooping_CVPR_2025_paper.pdf
> 3. GitHub (Official): https://github.com/realcrane/3D-student-splatting-and-scooping
> 4. arXiv HTML: https://arxiv.org/html/2503.10148v1
> 5. ar5iv: https://ar5iv.labs.arxiv.org/html/2503.10148
> 6. Liner Quick Review: https://liner.com/review/3d-student-splatting-and-scooping
> 7. UCL Discovery: https://discovery.ucl.ac.uk/id/eprint/10219841/
> 8. IEEE Xplore: https://ieeexplore.ieee.org/document/11092303/

---

## 1. 핵심 주장 및 주요 기여 (간결 요약)

3D Gaussian Splatting(3DGS)은 새로운 시점 합성(Novel View Synthesis)을 위한 강력한 프레임워크로 자리 잡았으며, 수많은 후속 연구의 기반이 되고 있다. 저자들은 바로 이 3DGS 자체의 근본적인 패러다임과 수식을 개선하는 것을 목표로 한다.

논문의 핵심 주장은, 3DGS가 비정규화 혼합 모델(unnormalized mixture model)로서 반드시 가우시안(Gaussian)을 사용하거나 기존 방식의 splatting을 고수할 필요가 없다는 것이다. 저자들은 이를 바탕으로 유연한 Student's t 분포와 양수(splatting) 및 음수(scooping) 밀도를 모두 포함하는 새로운 혼합 모델, **SSS(Student Splatting and Scooping)**를 제안한다.

### 🔑 주요 기여 요약

공식적인 기여는 다음과 같다:
- **(1)** 고표현력(expressive)이면서 파라미터 효율적인 새로운 모델 **SSS** 제안
- **(2)** 뉴럴 렌더링을 위한 유연한 분포 패밀리로부터 학습되는 새로운 혼합 모델
- **(3)** 3D 공간에서 음수 밀도를 도입하는 음수 성분(negative components)을 가진 혼합 모델

또한 파라미터 결합(parameter coupling) 문제를 완화하기 위해 **SGHMC(Stochastic Gradient Hamiltonian Monte Carlo)** 기반의 원칙적 샘플링 최적화 방법을 도입하며, 낮은 불투명도 성분을 재배치하는 **컴포넌트 재활용(component recycling) 전략**을 통해 ADC(Adaptive Density Control) 없이도 파라미터 효율을 높인다.

---

## 2. 상세 분석

### 2-1. 해결하고자 하는 문제

기존 3DGS는 비정규화 혼합 모델임에도 불구하고 반드시 가우시안 분포를 사용하고, 양수 밀도 공간에서만 동작하는 splatting 방식을 고수했다. 저자들은 이것이 표현력의 본질적인 한계임을 지적한다.

구체적인 문제점은 아래와 같다:

| 문제 | 설명 |
|---|---|
| **고정된 분포 선택** | 3DGS는 가우시안만을 기본 프리미티브로 사용 → 표현력 제한 |
| **양수 밀도 공간 한정** | 기존 splatting은 음수 밀도(빼기 연산)를 활용 불가 |
| **파라미터 결합 문제** | t-분포의 자유도 $\nu$, 평균 $\mu$, 공분산 $\Sigma$ 간의 강한 결합으로 일반 SGD 최적화 불충분 |

특히, 모델 복잡도가 증가함에 따라 일반적인 확률적 경사하강법(SGD) 기반 최적화 방법이 파라미터 결합 문제로 인해 충분하지 않게 되며, 이를 해결하기 위해 SGHMC 기반의 원칙적 샘플링 방법을 제안한다.

---

### 2-2. 제안하는 방법 (수식 포함)

#### (A) Student's t 분포 기반 혼합 모델

Student's t 분포는 자유도 $\nu$에 따라 그 형태가 변화하는데, $\nu \to 1$이면 코시(Cauchy) 분포에 수렴하고, $\nu \to \infty$이면 가우시안 분포에 수렴한다. 즉, t-분포는 가우시안이 표현할 수 있는 모든 것을 포함하며 그 이상도 가능하다.

3차원 공간에서의 SSS 기본 분포 성분(component)은 다음과 같이 정의된다:

$$
T(\mathbf{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma}, \nu) = \left(1 + \frac{(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}{\nu}\right)^{-\frac{\nu + d}{2}}
$$

여기서:
- $\boldsymbol{\mu} \in \mathbb{R}^3$: 성분의 중심(위치)
- $\boldsymbol{\Sigma} \in \mathbb{R}^{3\times3}$: 공분산 행렬 (스케일 및 회전 표현)
- $\nu > 0$: 자유도(degree of freedom), **학습 가능한 파라미터** (꼬리 두꺼움 제어)
- $d$: 차원 수 (3D에서 $d=3$)

#### (B) 양수+음수 혼합 모델 (SSS 전체 밀도장)

저자들은 양수 밀도 공간에서만 동작하는 기존 splatting 방식을 확장하여, 양수 성분(밀도를 더하는 splatting)과 음수 성분(밀도를 빼는 scooping)을 모두 사용하는 비단조(non-monotonic) 혼합 모델을 구성한다. 이는 3DGS보다 복잡한 수식을 야기하지만, 저자들은 학습을 위한 폐쇄형(closed-form) 그래디언트를 유도한다.

전체 밀도장 $f(\mathbf{x})$는:

$$
f(\mathbf{x}) = \sum_{k \in \mathcal{P}} w_k^+ \cdot T(\mathbf{x}; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k, \nu_k) - \sum_{j \in \mathcal{N}} w_j^- \cdot T(\mathbf{x}; \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j, \nu_j)
$$

여기서:
- $\mathcal{P}$: 양수(positive/splatting) 성분 집합
- $\mathcal{N}$: 음수(negative/scooping) 성분 집합
- $w_k^+, w_j^- \geq 0$: 각 성분의 가중치(불투명도)

이 구조는 토러스(torus)와 같이 복잡한 위상 구조를 가진 장면을 **훨씬 적은 성분 수**로 표현할 수 있게 해준다:

예를 들어, 토러스(도넛 모양)를 표현할 때 양수 성분만으로는 5개 이상의 성분이 필요하거나, 단 2개로는 위상을 제대로 포착하지 못한다. 반면 SSS는 양수 성분 1개 + 음수 성분 1개, 총 2개만으로도 토러스의 위상 구조를 정확하게 표현할 수 있다.

#### (C) SGHMC 기반 최적화

SGHMC 기반 샘플링 방법은 파라미터 결합 문제를 완화하기 위해 제안된다. 사후 분포(posterior distribution)를 다음과 같이 매개변수화한다:

$$
P(\theta, r) \propto \exp\left(-\mathcal{L}_\theta(\mathbf{x}) - \frac{1}{2} r^\top I r\right)
$$

여기서:
- $\mathcal{L}_\theta(\mathbf{x})$: 손실 함수 (에너지 함수 역할)
- $r$: 모멘텀 보조 변수 (auxiliary momentum variable)
- $I$: 단위 행렬
- $\theta$: 학습 가능한 파라미터 전체 집합 ($\boldsymbol{\mu}, \boldsymbol{\Sigma}, \nu, w$ 포함)

SGHMC는 2차(second-order) 샘플러이므로, Adam(1차 최적화)과는 학습률 설정 방식이 다르다. 초기 학습률의 제곱이 실제 학습률로 적용된다.

렌더링(splatting) 시에는 3D→2D 투영 변환을 위해:

$$
\boldsymbol{\Sigma}_{2D} = J W \boldsymbol{\Sigma} W^\top J^\top
$$

여기서 $W$와 $J$는 각각 아핀 변환(스케일, 이동)과 (근사) 투영 변환 행렬이다.

---

### 2-3. 모델 구조

전체 SSS 파이프라인을 정리하면 다음과 같다:

```
입력: SfM Point Cloud (Structure from Motion)
    ↓
초기화: 각 점을 Student's t 성분으로 초기화
    ↓
성분 파라미터 학습 (SGHMC):
  - 위치 μ, 공분산 Σ, 자유도 ν, 불투명도 w, 구형 조화 함수(SH) 색상
  - 양수/음수 성분 모두 포함
    ↓
2D 투영(α-compositing):
  - 각 t-성분을 2D로 투영 → 렌더링
    ↓
컴포넌트 재활용 전략:
  - 낮은 불투명도 성분을 다른 위치로 재배치
    ↓
출력: 렌더링 이미지
```

SSS는 3DGS 및 그 변형들의 단순하지만 강력하고 비자명한(non-trivial) 일반화(generalization)를 포함한다.

---

### 2-4. 성능 향상

실험 결과, SSS는 더 적은 성분 수로도 더 높은 품질을 달성하며, 더 높은 표현력과 파라미터 효율성을 입증한다.

에블레이션 연구에 따르면, Gaussian을 t-분포로만 교체(SGD+t-dis)해도 이미 Mip-NeRF, 3DGS, GES를 능가하며, 여기에 SGHMC를 추가하면 최고 성능 방법이 된다. 음수 성분까지 추가하면 성능이 더욱 향상된다.

정성적 비교에서도 SSS는 미세한 텍스처, 반사, 균질/비균질 영역(하늘과 먼 언덕 등) 구분에서 시각적으로 우수한 결과를 보여준다. 다른 방법들이 적은 성분 수에서 흐릿함이나 아티팩트를 보이는 반면, SSS는 선명한 디테일과 정확한 색상 재현을 유지한다.

벤치마크 결과, SSS는 세 가지 주요 데이터셋 **Mip-NeRF 360, Tanks & Temples, Deep Blending**에서 SOTA 성능을 달성한다.

속도 측면에서는 바닐라 3DGS보다 느리지만, **실시간 렌더링(>70 FPS)**은 여전히 달성한다.

---

### 2-5. 한계점

SSS의 한계는 두 가지다: (1) 프리미티브가 대칭적이고 매끄러운(symmetric and smooth) t-분포로 제한되어 표현력에 한계가 있고, (2) 음수 성분의 비율 등 SGHMC의 하이퍼파라미터 튜닝이 필요하다.

---

## 3. 모델의 일반화 성능 향상 가능성 (중점 분석)

### 3-1. 분포 패밀리의 일반화

Student's t 분포는 $\nu \to \infty$이면 가우시안과 동일해지므로, **SSS는 기존 3DGS를 완전히 포함하는 상위 호환 모델**이다. 즉, 3DGS가 표현하는 모든 장면을 SSS도 표현할 수 있으며, 그 이상도 가능하다.

성능 우위는 성분 수가 심하게 제한된 경우에도 유지되며, 이는 제한된 리소스 환경에서의 견고성과 효율성을 시사한다. 성분 수가 감소함에 따라 SSS의 렌더링 품질 저하가 다른 방법에 비해 더 완만하게 진행된다.

### 3-2. 비대칭 장면 일반화 가능성

기존 3DGS는 가우시안의 대칭성으로 인해 비대칭적, 불규칙적 형태의 장면에서 많은 수의 프리미티브가 필요했다. SSS는 두 가지 메커니즘으로 이를 극복한다:

- **자유도 $\nu$의 학습**: 두꺼운 꼬리(fat-tail)를 가진 코시 분포에서 가우시안까지 각 성분이 장면에 맞게 자동으로 분포 형태를 조절
- **음수 성분의 도입**: 토러스, 오목 구조 등 위상 복잡도가 높은 장면에서 극소수의 성분만으로도 정확한 표현 가능

### 3-3. 미래 일반화 방향

저자들은 향후 라플라스(Laplace) 등 다른 분포 패밀리와 t-분포를 결합하여 표현력을 더욱 향상시키고, SGHMC를 자기적응형(self-adaptive)으로 만들어 양수/음수 성분 간의 균형을 자동으로 조절할 계획이다.

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

| 모델 | 연도 | 핵심 방법 | 비고 |
|---|---|---|---|
| **NeRF** | 2020 | MLP 기반 암시적 표현, 부피 렌더링 | 느린 학습/렌더링 |
| **Mip-NeRF 360** | 2022 | 무한 경계 장면 처리 | 느린 렌더링 |
| **3DGS** | 2023 | 3D 가우시안 명시적 표현, 실시간 렌더링 | 기존 SOTA |
| **Scaffold-GS** | 2023 | 앵커 기반 계층적 표현 | 대형 장면 확장성 |
| **2DGS** | 2024 | 3D→2D 가우시안으로 표면 재구성 향상 | 표면 정밀도↑ |
| **GES** | 2024 | 일반화 지수 커널 사용 | 메모리 절약 |
| **SSS (본 논문)** | 2025 | t-분포 + 음수 성분 + SGHMC | **CVPR 2025 Oral + Best Paper HM** |

2DGS는 3D 가우시안 프리미티브를 2D 가우시안으로 변경하여 더 나은 표면 재구성을 달성하고, GES는 일반화 지수 커널(generalized exponential kernel)을 사용하여 프리미티브의 표현 능력을 높이고 메모리 비용을 줄였다.

3D Gaussian Splatting은 실용적 응용에서 NeRF를 대체하는 추세이며, NeRF 대비 100+ FPS의 렌더링 속도와 동등하거나 더 높은 시각적 품질(25–33 dB PSNR)을 보여준다.

SSS는 기존 연구들과 비교하여 다음과 같은 차별점을 가진다:

| 비교 항목 | 기존 3DGS/변형 | SSS |
|---|---|---|
| 분포 | 고정 가우시안 | t-분포 (자유도 학습 가능) |
| 밀도 공간 | 양수만 | 양수 + 음수 |
| 최적화 | Adam(SGD 기반) | SGHMC(2차 샘플러) |
| 파라미터 효율 | 낮음 | 높음 (적은 성분으로 동등 품질) |
| 일반화 | 제한적 | 3DGS의 완전한 상위 호환 |

---

## 5. 앞으로의 연구에 미치는 영향 및 고려할 점

### 5-1. 연구에 미치는 영향

3DGS가 수많은 모델의 기반 컴포넌트로 자리 잡은 상황에서, SSS처럼 3DGS 자체를 개선하는 연구는 그 파급 효과가 매우 크다.

구체적인 영향은 다음과 같다:

1. **프리미티브 설계 패러다임 전환**: 단순히 가우시안을 사용하던 관행에서 벗어나, **분포 패밀리 자체를 학습 대상**으로 삼는 접근법의 가능성을 제시
2. **음수 밀도의 활용**: 물리적으로 비직관적이었던 음수 밀도 개념을 3D 렌더링에 성공적으로 적용하여, 위상 복잡도가 높은 장면(홀, 오목면 등) 표현에 새로운 방향 제시
3. **MCMC 기반 최적화의 확산**: 3DGS-MCMC에 이어 SGHMC를 도입함으로써, 뉴럴 렌더링 분야에서 **2차 최적화/샘플링 기반 학습**의 중요성 부각
4. **다운스트림 태스크 기여**: 동적 장면 표현, 아바타 생성, 자율주행 시뮬레이션 등 3DGS 기반 응용 연구 전반에 걸쳐 더 높은 품질과 효율을 기대할 수 있음

### 5-2. 향후 연구 시 고려할 점

1. **하이퍼파라미터 민감도**: 음수 성분의 비율 등 SGHMC 관련 하이퍼파라미터 튜닝이 필요하므로, 자동 조율(auto-tuning) 또는 메타 학습(meta-learning) 기반 접근법 연구가 필요하다.

2. **비대칭 분포 확장**: 현재 SSS의 프리미티브는 대칭적이고 매끄러운 t-분포로 제한되므로, GMM(Gaussian Mixture Model)의 비대칭 변형이나 스큐(skew) t-분포를 활용하는 방향을 고려할 수 있다.

3. **다중 분포 패밀리 결합**: 라플라스 등 다른 분포 패밀리와의 결합을 통해 더욱 다양한 기하 구조를 효율적으로 표현하는 연구가 기대된다.

4. **동적 장면 확장**: 현재 SSS는 정적 장면을 주로 다루고 있으므로, 시간 축 $t$를 포함한 4D 혼합 모델로의 확장, 또는 변형 가능한(deformable) t-분포 기반 동적 장면 표현 연구가 중요한 과제이다.

5. **시각화 도구 부재**: 가우시안에서 t-분포로 변경함에 따라 기존 3DGS의 SIBR 기반 뷰어를 직접 적용할 수 없어, 새로운 시각화 도구 개발이 필요하다.

6. **학습 비용 고려**: SGHMC는 2차 샘플러로 Adam에 비해 계산 비용이 높으므로, 경량화 또는 근사 SGHMC 설계가 실용화에 중요한 요소이다.

---

> ⚠️ **정확도 관련 안내**: 본 답변에서 제시된 수식($T(\mathbf{x})$의 구체적 형태, SGHMC 식 등)은 논문 원문에서 확인된 정보를 바탕으로 작성되었으나, PDF 원문의 일부 상세 수식(예: 렌더링 파이프라인의 전체 폐쇄형 그래디언트 유도 과정)은 공개 검색 결과의 한계상 완전히 재현하지 못했을 수 있습니다. 보다 정확한 수식은 [arXiv 원문](https://arxiv.org/abs/2503.10148) 또는 [CVF 공개 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_3D_Student_Splatting_and_Scooping_CVPR_2025_paper.pdf)를 직접 참고하시기 바랍니다.
