
# Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh

> **논문 정보**
> - **저자**: Xiangjun Gao, Xiaoyu Li, Yiyu Zhuang, Qi Zhang, Wenbo Hu, Chaopeng Zhang, Yao Yao, Ying Shan, Long Quan
> - **게재**: CVPR 2025 (pp. 21392–21402)
> - **arXiv**: [2405.17811](https://arxiv.org/abs/2405.17811) (v1: 2024.05.28, v2: 2025.03.24)
> - **공식 GitHub**: [gaoxiangjun/Mani-GS](https://github.com/gaoxiangjun/Mani-GS)
> - **프로젝트 페이지**: [gaoxiangjun.github.io/mani_gs](https://gaoxiangjun.github.io/mani_gs/)

---

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 배경과 핵심 주장

Neural 3D 표현 방식인 NeRF는 사실적인 렌더링 결과를 생성하는 데 탁월하지만, 콘텐츠 제작에 필수적인 조작 및 편집의 유연성이 부족하다. 이전 연구들은 정준 공간(canonical space)에서 NeRF를 변형하거나 명시적 메쉬 기반으로 방사장(radiance field)을 조작하는 방법을 시도했으나, NeRF 조작은 제어가 어렵고 훈련 및 추론 시간이 오래 걸린다는 한계가 있었다.

3D Gaussian Splatting(3DGS)의 등장으로 명시적인 포인트 기반 3D 표현을 활용해 훨씬 빠른 훈련·렌더링 속도로 고충실도 신규 시점 합성이 가능해졌다. 그러나 렌더링 품질을 유지하면서 3DGS를 자유롭게 조작할 수 있는 효과적인 수단은 여전히 부족하다.

**Mani-GS의 핵심 주장**: 삼각형 메쉬를 활용하여 자기 적응(self-adaptation) 방식으로 3DGS를 직접 조작하며, 이 접근법은 다양한 유형의 Gaussian 조작을 위한 별도 알고리즘 설계 필요성을 줄인다.

### 1.2 주요 기여 (Contributions)

논문의 세 가지 주요 기여는 다음과 같다:
1. 삼각형 메쉬 조작을 3DGS로 효과적으로 전달하면서 고품질 렌더링을 유지하는 3DGS 조작 방법 제안
2. 메쉬 정확도에 대한 높은 허용 오차를 가진 삼각형 형상 인식(triangle shape-aware) Gaussian 바인딩 전략 제안
3. 대규모 변형, 로컬 조작, 소프트 바디 시뮬레이션 등 다양한 3DGS 조작을 지원하는 SOTA 결과 달성

요약하면, Gaussian-Mesh 바인딩 전략(self-adaptation 포함)을 도입하여 고품질 렌더링 유지, 메쉬 정확도에 대한 높은 내성, 다양한 유형의 3DGS 조작 지원을 달성한다.

---

## 2. 해결 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

3D 콘텐츠 조작 및 편집은 영화, 게임, VR/AR 등 다양한 분야에서 필수적이다. 하지만 기존 사실적 렌더링을 위한 3D 자산 모델링 파이프라인(기하 모델링, 텍스처링, UV 매핑, 조명, 렌더링)은 복잡하고 시간이 많이 소요된다.

구체적으로 해결하고자 하는 두 가지 핵심 문제:

1. **NeRF 기반 조작의 한계**: 제어가 어렵고 훈련·추론 속도가 느리다.
2. **3DGS 조작 수단의 부재**: 3DGS는 명시적인 포인트 기반 3D 표현으로 고충실도 신규 시점 합성을 빠르게 달성할 수 있지만, 렌더링 품질을 유지하면서 3DGS를 자유롭게 조작할 효과적인 수단이 부족하다.

---

### 2.2 제안 방법 및 수식

#### 단계 1: 메쉬 추출 (Mesh Extraction)

방법의 첫 번째 단계는 메쉬 추출 단계로, NeuS 또는 3DGS(Screened Poisson 또는 Marching Cube)에서 삼각형 메쉬를 추출한다.

3D Gaussian은 재구성을 위한 법선 벡터가 없으므로, 최근 3DGS 역 렌더링 방법들에서 영감을 받아 3D Gaussian에 추가적인 법선 속성 $\boldsymbol{n}$을 할당하고, 이를 깊이 맵에서 유도된 의사 법선(pseudo normal)으로 감독한다.

#### 단계 2: Triangle Shape-Aware Gaussian Binding

각 삼각형의 로컬 삼각형 공간에 $N$개의 Gaussian을 바인딩하고, 로컬 Gaussian 속성 $\{u, R, s, o, c\}$를 최적화한다. 삼각형 속성 $\{u, R, e\}$는 삼각형 꼭짓점을 기반으로 계산된다.

각 3D Gaussian의 기본 수식은 다음과 같다:

$$G(\boldsymbol{x}) = \exp\left(-\frac{1}{2}(\boldsymbol{x} - \boldsymbol{\mu})^\top \Sigma^{-1}(\boldsymbol{x} - \boldsymbol{\mu})\right)$$

여기서 $\boldsymbol{\mu}$는 Gaussian의 평균(위치), $\Sigma$는 공분산 행렬이다.

각 Gaussian은 다음 속성으로 정의된다:
- $\boldsymbol{\mu}$: 위치 (position)
- $\boldsymbol{R}$: 회전 (rotation)
- $\boldsymbol{s}$: 스케일 (scale)
- $o$: 불투명도 (opacity)
- $c$: 색상/SH 계수 (color/SH coefficients)

**로컬 삼각형 좌표계**에서 각 삼각형은 다음 속성으로 정의된다:

$$\text{삼각형 속성: } \{u_t, R_t, e_t\}$$

- $u_t$: 삼각형의 중심 위치
- $R_t$: 삼각형의 회전 행렬 (법선 방향 등에서 계산)
- $e_t$: 삼각형의 엣지 길이 (edge length)

#### 단계 3: 조작 전달 (Self-Adaptive Manipulation Transfer)

삼각형의 회전, 위치, 엣지 길이는 즉시 계산될 수 있으며, 이를 통해 전역 Gaussian 위치, 스케일링, 회전이 제안된 공식에 따라 자기 적응적으로 조정된다.

메쉬 조작 중 로컬 삼각형 공간의 속성은 변경되지 않고, 전역 Gaussian 위치, 스케일링, 회전이 제안된 공식에 따라 자기 적응적으로 조정된다. 결과적으로 삼각형 메쉬를 사용하여 3DGS를 조작하면서 렌더링 품질을 유지할 수 있다.

전역 좌표에서 Gaussian 위치의 변환은 다음과 같이 나타낼 수 있다:

$$\boldsymbol{\mu}_{global}^{new} = u_t^{new} + R_t^{new} \cdot \boldsymbol{\mu}_{local}$$

$$R_{G}^{new} = R_t^{new} \cdot R_{local}$$

$$s_{G}^{new} = \frac{e_t^{new}}{e_t^{old}} \cdot s_{local}$$

여기서:
- $\boldsymbol{\mu}_{local}$: 로컬 좌표계에서의 Gaussian 위치
- $R_t^{new}$: 조작 후 삼각형 회전 행렬
- $e_t^{new} / e_t^{old}$: 엣지 길이 비율 (스케일 자기 적응)

#### 학습 손실 함수

각 3D Gaussian에 법선 속성 $\boldsymbol{n}$을 추가하고 의사 법선 제약으로 법선 속성을 최적화한다. $\mathcal{L}_n$ 외에도 일반 L1 손실과 구조적 유사도 지수(SSIM) 손실도 함께 통합된다.

전체 손실 함수:

$$\mathcal{L} = \mathcal{L}_{color} + \lambda_n \mathcal{L}_n$$

$$\mathcal{L}_{color} = (1 - \lambda_{ssim}) \mathcal{L}_1 + \lambda_{ssim} \mathcal{L}_{SSIM}$$

$$\mathcal{L}_n = \|\boldsymbol{n}_{pred} - \boldsymbol{n}_{pseudo}\|_1$$

---

### 2.3 모델 구조 (파이프라인)

전체 파이프라인은 2단계로 구성된다:

훈련은 두 단계로 나뉜다: (1) Screened Poisson 재구성 또는 NeuS를 사용하여 3DGS에서 메쉬를 추출하는 단계; (2) 주어진 삼각형 메쉬에 3D Gaussian을 바인딩하는 단계.

```
[Stage 1 - 선택적]
3DGS 학습 (기존 3DGS 방법)
      ↓
메쉬 추출 (NeuS or Screened Poisson or Marching Cube)
      ↓
[Stage 2 - 필수]
Triangle Shape-Aware Gaussian Binding
      ↓
로컬 속성 최적화 (u, R, s, o, c)
      ↓
[조작 단계 - Blender 활용]
메쉬 조작 (대규모 변형, 로컬 조작, 소프트 바디)
      ↓
Self-Adaptive 전역 속성 업데이트
      ↓
조작된 고품질 렌더링 출력
```

본 논문은 조작된 메쉬에 의해 구동되는 대규모 변형, 로컬 조작, 소프트 바디 시뮬레이션 등의 3DGS 조작 렌더링 결과를 제시하며, 실험에서는 메쉬 조작을 위해 Blender를 사용한다.

---

### 2.4 성능 향상

SuGaR 및 NeuMesh와의 편집 비교에서, Mani-GS는 SuGaR보다 아티팩트와 블러링 효과가 적고, NeuMesh보다 더 풍부하고 뚜렷한 디테일을 제공한다.

이 논문에서는 자기 적응(self-adaptation) 기능을 갖춘 삼각형 형상 인식 Gaussian 바인딩 전략을 소개하며, 다양한 3DGS 조작을 지원하고 렌더링 품질을 유지하며 메쉬 정확도에 높은 내성을 가진다. 합성 및 실제 데이터셋 모두에서 방법을 평가하여 SOTA 결과를 달성한다.

**지원하는 조작 유형**:
- 대규모 변형, 로컬 조작, 물리 시뮬레이션(소프트 바디 포함) 등 고품질 렌더링을 유지하면서 수행 가능하며, 3DGS에서 추출된 부정확한 메쉬에도 효과적이다.

**편집 속도**:
편집 시간은 주로 메쉬 편집의 비용에 의존하며, Blender를 사용한 로컬 조작과 대규모 변형은 즉시 달성 가능하다. 소프트 바디 시뮬레이션은 Blender의 시뮬레이션 알고리즘에 따라 더 많은 시간이 소요될 수 있다.

---

### 2.5 한계 (Limitations)

실험 중 일부 결과에서 여전히 왜곡(distortion)이 나타남을 확인하였다. 조작된 메쉬의 로컬 영역에 고도로 비강체적(non-rigid) 변형이 포함될 경우 렌더링 왜곡이 발생할 수 있다. 또한 시뮬레이션 데모에서 35K 개 이상의 삼각형을 가진 메쉬에서 물리 시뮬레이션을 수행하면 수 시간이 걸릴 수 있다.

추가적인 한계:
주로 정적 3D 형상에 초점을 맞추고 있어 동적 또는 변형 가능한 3D 객체로의 확장 가능성이 충분히 탐구되지 않았으며, 복잡한 3D 형상에서의 Mani-GS 접근법의 효율성과 계산 비용이 완전히 탐구되지 않았다. 확장성과 실시간 성능에 대한 추가 조사가 필요하다.

또한 3DGS 바인딩 훈련과 렌더링 속도의 효율성은 Gaussian의 수에 의존하며, 이는 삼각형 수의 곱으로 결정된다. 서로 다른 삼각형(270K, 150K, 70K)으로 기반 메쉬 해상도의 영향도 평가하였다.

---

## 3. 모델의 일반화 성능 향상 가능성

### 3.1 핵심 일반화 강점: 메쉬 정확도 내성

Mani-GS의 가장 두드러진 일반화 특성은 **부정확한 메쉬에서도 작동하는 내성(tolerance)**이다.

Gaussian이 삼각형 외부에서 자유롭게 설정되어 있어 Gaussian이 부정확한 메쉬에 바인딩된 경우에도 고충실도 조작을 지원할 수 있으며, 메쉬 정확도에 대한 높은 내성을 보인다.

Mesh-Gaussian 바인딩 전략을 통해 부정확한 메쉬에서도 고충실도 렌더링을 달성하고 부드러운 Gaussian 조작을 지원할 수 있다.

### 3.2 다양한 데이터셋에서의 일반화

실제 장면에 대한 추가 조작 결과를 제시하였으며, 상위 3개 이미지는 DTU 데이터셋에서, 하위 2개는 Tanks and Templates 데이터셋에서 가져왔다.

이 결과는 접근법이 메쉬 조작을 Gaussian Splatting으로 성공적으로 전달하여 정확하고 시각적으로 매력적인 결과를 제공할 수 있음을 보여준다.

### 3.3 다양한 메쉬 추출 방식에서의 일반화

접근법의 첫 번째 단계는 메쉬 추출로, NeuS 메쉬를 Gaussian 바인딩의 기반으로 활용하면서도 Gaussian Splatting에서 메쉬 추출도 탐색한다. Screened Poisson 표면 재구성 방법을 사용하여 학습된 Gaussian Splatting 모델에서 삼각형 메쉬 추출을 시도한다.

### 3.4 다양한 조작 유형에서의 범용성

삼각형 메쉬를 활용하여 자기 적응 방식으로 3DGS를 직접 조작하며, 이 접근법은 다양한 유형의 3DGS 조작을 위한 별도 알고리즘 설계 필요성을 줄인다.

즉, 대규모 변형(large deformation), 로컬 조작(local manipulation), 소프트 바디 시뮬레이션(soft body simulation) 모두 **동일한 바인딩 프레임워크** 위에서 동작하므로, 각 조작 유형별로 별도의 모델을 설계할 필요가 없다.

### 3.5 일반화 한계와 개선 방향

| 일반화 측면 | 현재 상태 | 개선 가능성 |
|---|---|---|
| 메쉬 정확도 | 부정확한 메쉬에도 내성 | ✅ 강점 |
| 데이터셋 범위 | 합성 + 실제(DTU, T&T) | 추가 실내외 장면 확장 필요 |
| 동적 장면 | 정적 장면 중심 | 4D-GS 등과의 결합 필요 |
| 비강체 변형 | 고도 비강체 시 왜곡 발생 | 변형 필드 학습 결합 가능 |
| 대규모 장면 | 삼각형 수 증가 시 속도 저하 | 계층적 메쉬 LOD 필요 |

---

## 4. 관련 최신 연구 비교 분석 (2020년 이후)

### 4.1 NeRF 기반 조작 연구

| 방법 | 연도 | 핵심 방식 | Mani-GS 대비 한계 |
|---|---|---|---|
| **NeRF** (Mildenhall et al.) | 2020 | MLP 기반 암시적 표현 | 조작 불가, 속도 느림 |
| **NeuS** (Wang et al.) | 2021 | 부호 거리 함수 + 볼륨 렌더링 | 조작 제어 어려움 |
| **NeRF-Editing** (Yuan et al.) | 2022 | 삼각형 메쉬를 활용한 최초의 암시적 방사장 편집; NeuS를 학습하고 삼각형 메쉬 추출, 사면체 격자 구성 후 볼륨 렌더링 수행 | 긴 훈련 시간, 느린 렌더링 |
| **NeuMesh** (Yang et al.) | 2022 | 분리된 신경 메쉬 기반 | 제한적 디테일 |

### 4.2 3DGS 기반 조작/편집 연구

최근 수 년간 3D 편집은 급성장 중인 연구 주제로, NeRF와 3D Gaussian Splatting과 같은 방사장 기반 방법의 등장이 3D 편집의 효과와 효율을 크게 향상시켰다.

| 방법 | 연도 | 핵심 방식 | Mani-GS 대비 특성 |
|---|---|---|---|
| **3DGS** (Kerbl et al.) | 2023 | 명시적 포인트 기반 표현 | 조작 수단 없음 |
| **SuGaR** (Guédon & Lepetit) | 2024 | 다층 메쉬로 장면 표현; 단일 메쉬만 필요로 하는 Mani-GS와 달리 멀티레이어 메쉬 필요 | 아티팩트 발생, 블러링 |
| **GaussianEditor** (Chen et al.) | 2024 | 텍스트 지시에서 RoI를 추출, 3D Gaussian에 정렬 후 편집 | 기하학적 로컬 편집 불가 |
| **GaMeS** (Waczynska et al.) | 2024 | 메쉬와 유사하게 Gaussian 구성요소를 수정할 수 있는 모델로 실시간 렌더링 지원 | Mani-GS보다 자기 적응 능력 부족 |
| **SC-GS** (Huang et al.) | 2024 | 희소 제어점(sparse control) 방식 | 연속 메쉬 기반 시뮬레이션 지원 미흡 |
| **MaGS** (related) | 2024 | Mani-GS와 함께 정적 장면을 위한 시뮬레이션 기능 통합; 동적 장면 지원 미흡 | 동적 장면 확장 시도 |
| **Mani-GS** (Gao et al.) | 2024/2025 | 삼각형 형상 인식 Gaussian 바인딩 + 자기 적응 | **본 논문** |

### 4.3 물리 기반 시뮬레이션 연구

| 방법 | 특성 |
|---|---|
| **PhysGaussian** (Xie et al., 2023) | 물리 통합 3D Gaussian으로 생성적 다이나믹스 |
| **VR-GS** (Jiang et al., 2024) | 물리 다이나믹스 인식 인터랙티브 Gaussian Splatting |

Mani-GS는 물리 시뮬레이션과의 연결성에서 Blender의 시뮬레이션 엔진을 활용하는 방식으로 차별화된다.

---

## 5. 향후 연구에 미치는 영향과 고려 사항

### 5.1 향후 연구에 미치는 영향

**① 통합 조작 프레임워크의 표준화**
메쉬를 활용하여 3DGS를 자기 적응 방식으로 직접 조작하는 이 접근법은 다양한 유형의 3DGS 조작을 위한 별도 알고리즘 설계 필요성을 줄임으로써, 통합된 단일 프레임워크로 다양한 조작을 처리하는 방향의 연구를 촉진할 것이다.

**② 콘텐츠 제작 파이프라인 혁신**
Mani-GS는 3D 모델링, 컴퓨터 그래픽스, 가상 현실 등 다양한 3D 컴퓨팅 응용 분야에 유용한 도구가 될 잠재력을 가진다.

**③ 3DGS-물리 시뮬레이션 결합 연구 촉진**

3DGS와 기하학적 사전 지식을 통합하려는 여러 시도들(SC-GS의 희소 제어점, D-Miso의 불연속 메쉬 표면)이 있었으며, DG-Mesh와 SplattingAvatar는 메쉬 표면에 Gaussian을 바인딩하여 메쉬 기반 동적 재구성을 가능하게 하였다. Mani-GS의 바인딩 전략은 이러한 흐름에서 물리 시뮬레이션과 3DGS를 더욱 긴밀하게 통합하는 연구의 기반이 된다.

**④ NeRF→3DGS 전환 연구의 교량 역할**
최신 연구 흐름은 3DGS 기반으로 이동하여 더 높은 편집 효율과 우수한 렌더링 품질을 달성하고 있으며, GaussianEditor는 텍스트 기반 로컬 편집 등 다양한 3DGS 편집 방법을 선보인다. Mani-GS는 텍스트 기반이 아닌 **메쉬 기반 기하학적 조작**이라는 측면에서 3D 편집 연구의 다양성을 확장한다.

### 5.2 향후 연구 시 고려할 점

**① 동적 장면으로의 확장**
주로 정적 3D 형상에 초점을 맞추고 있어, Mani-GS를 동적 또는 변형 가능한 3D 객체를 처리하도록 확장하는 방법을 탐구하는 것이 흥미로울 것이다. 4D Gaussian Splatting이나 Deformable 3DGS와의 결합 가능성을 검토해야 한다.

**② 고도 비강체 변형 처리**
일부 결과에서 여전히 왜곡이 나타나며, 조작된 메쉬의 로컬 영역에 고도로 비강체적 변형이 포함될 경우 렌더링 왜곡이 발생할 수 있다. 이를 해결하기 위해 국소 적응적(locally adaptive) 변형 필드 학습이나 추가적인 정규화 항 설계를 고려해야 한다.

**③ 대규모 메쉬 처리 효율화**
35K 개 이상의 삼각형을 가진 메쉬에서 물리 시뮬레이션을 수행하면 수 시간이 걸릴 수 있다. 계층적 메쉬(LOD, Level of Detail), 메쉬 단순화, 또는 병렬 시뮬레이션 전략을 연구할 필요가 있다.

**④ 텍스트/이미지 기반 조작과의 결합**
현재 Mani-GS는 Blender를 통한 수동 메쉬 조작에 의존하므로, GaussianEditor나 Instruct-NeRF2NeRF 등의 텍스트 기반 편집 방법과 결합하여 더 직관적인 3D 콘텐츠 편집 파이프라인 구축을 고려할 수 있다.

**⑤ 렌더링 품질 지표의 다양화**
다른 3D 표현 기법(복셀 그리드, 포인트 클라우드 등)과의 포괄적인 비교가 제공되지 않았으므로, Mani-GS가 이러한 다른 방법들과 비교하여 품질, 효율성, 사용 편의성 측면에서 어떤 성능을 보이는지 파악하는 것이 중요하다.

**⑥ 실시간 응용을 위한 최적화**
Mani-GS 접근법의 효율성과 계산 비용(특히 복잡한 3D 형상에 대한)이 완전히 탐구되지 않았으므로, 확장성과 실시간 성능에 대한 추가 조사가 필요하다.

---

## 📚 참고 자료 및 출처

| # | 자료명 | URL/출처 |
|---|---|---|
| 1 | **[주논문] Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh** (arXiv) | https://arxiv.org/abs/2405.17811 |
| 2 | **[주논문] CVPR 2025 Open Access** | https://openaccess.thecvf.com/content/CVPR2025/html/Gao_Mani-GS_Gaussian_Splatting_Manipulation_with_Triangular_Mesh_CVPR_2025_paper.html |
| 3 | **[주논문] IEEE Xplore** | https://ieeexplore.ieee.org/document/11092889/ |
| 4 | **[공식 구현] GitHub - gaoxiangjun/Mani-GS** | https://github.com/gaoxiangjun/Mani-GS |
| 5 | **[프로젝트 페이지] Mani-GS 공식 페이지** | https://gaoxiangjun.github.io/mani_gs/ |
| 6 | **[HTML 풀텍스트] arXiv HTML v2** | https://arxiv.org/html/2405.17811v2 |
| 7 | **[OpenReview] ICLR 2025 (Withdrawn)** | https://openreview.net/forum?id=0N8yq8QwkD |
| 8 | **[Semantic Scholar] 논문 상세** | https://www.semanticscholar.org/paper/Mani-GS:-Gaussian-Splatting-Manipulation-with-Mesh-Gao-Li/82b7b1e63a3c5558af26cd92aa53be25b917ed58 |
| 9 | **[Survey] A survey on 3D editing based on NeRF and 3DGS** (Frontiers of Computer Science, Springer, 2025) | https://link.springer.com/article/10.1007/s11704-025-41176-9 |
| 10 | **[관련연구] MaGS: Mesh-adsorbed Gaussian Splatting** (arXiv 2406.01593) | https://arxiv.org/html/2406.01593v2 |
| 11 | **[관련연구] SuGaR: Surface-Aligned Gaussian Splatting** (CVPR 2024, GitHub) | https://github.com/Anttwo/SuGaR |
| 12 | **[관련연구] GaussianEditor** (ResearchGate) | https://www.researchgate.net/publication/384144010 |
| 13 | **[관련연구 목록] Awesome NeRF Editing (GitHub)** | https://github.com/EricLee0224/awesome-nerf-editing |
