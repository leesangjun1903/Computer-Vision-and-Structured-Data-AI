
# A Unified Differentiable Boolean Operator with Fuzzy Logic

> **논문 정보**
> - **저자**: Hsueh-Ti Derek Liu, Maneesh Agrawala, Cem Yuksel, Tim Omernick, Vinith Misra, Stefano Corazza, Morgan McGuire, Victor B. Zordan
> - **발표**: SIGGRAPH 2024 (Special Interest Group on Computer Graphics and Interactive Techniques Conference)
> - **arXiv**: [2407.10954](https://arxiv.org/abs/2407.10954) (2024년 7월 15일)
> - **공식 PDF**: [dgp.toronto.edu](https://www.dgp.toronto.edu/~hsuehtil/pdf/fuzzyBoolean.pdf)
> - **코드**: [GitHub - HTDerekLiu/fuzzy-boolean](https://github.com/HTDerekLiu/fuzzy-boolean)

---

## 1. 📌 핵심 주장 및 주요 기여 요약

퍼지 논리에서 영감을 받아, **연속 함수를 출력하며 연산자 타입에 대해 미분 가능한 통합 불리언 연산자(Unified Boolean Operator)**를 제안한다. 이를 통해 CSG(Constructive Solid Geometry)에서 primitives와 boolean operations 모두를 최적화할 수 있게 된다.

### 🔑 주요 기여 4가지

| 기여 | 내용 |
|------|------|
| **연속 불리언 연산자** | min/max 불연속 연산자를 퍼지 논리 기반 연속 함수로 대체 |
| **미분 가능한 연산자 타입** | 연산자 종류(union/intersection/difference) 자체를 연속 최적화 변수로 처리 |
| **소실 기울기 방지** | 사면체 무게중심(barycentric) 보간으로 monotonicity 보장 |
| **이중 형상 표현** | 날카로운 기계적 물체 + 부드러운 유기적 형태를 동일 프레임워크로 표현 |

---

## 2. 🔬 상세 분석

### 2-1. 해결하고자 하는 문제

이 논문은 CSG(Constructive Solid Geometry)를 이용한 암묵적 고체 형상 모델링을 위한 통합 미분 가능 불리언 연산자를 제안한다. 전통적인 CSG는 암묵적 형상에 불리언 연산을 수행하기 위해 min, max 연산자에 의존한다. 그러나 이러한 불리언 연산자는 불연속적이고 연산의 선택이 이산적이기 때문에, CSG 표현에 대한 최적화가 매우 어렵다.

구체적으로 두 가지 핵심 문제가 존재한다:

1. **불연속성(Discontinuity)**: 전통적 `min`/`max` 연산자는 기울기가 존재하지 않아 경사하강법 적용 불가
2. **이산적 연산자 선택**: union/intersection/difference 중 선택이 이산적이라 연속 최적화 불가

전통적인 Gödel max 연산자를 사용하면 소실 기울기(vanishing gradient) 문제가 발생하여, primitive가 최적화 과정 내내 변경되지 않아 목표 형상 복원에 실패한다.

---

### 2-2. 제안 방법 및 수식

#### **Step 1: Soft Occupancy 함수 기반의 퍼지 불리언 연산**

Fuzzy Logic [Zadeh, 1965]에서 영감을 받아, 개별 퍼지 논리 연산(t-norms, t-conorms)을 **소프트 점유 함수(soft occupancy functions)**로 표현된 고체 형상의 불리언 연산에 적용한다. 퍼지 불리언 연산자는 signed distance function에서 동작하는 기존 min/max 연산자와 달리, 결과가 항상 soft occupancy 함수로 유지됨을 보장한다.

핵심 개별 퍼지 연산자 (논문 Eq. 8 기반):

$$f_{X \cup Y}(p) = f_X(p) + f_Y(p) - f_X(p) \cdot f_Y(p) \quad \text{(Union, t-conorm)}$$

$$f_{X \cap Y}(p) = f_X(p) \cdot f_Y(p) \quad \text{(Intersection, t-norm)}$$

$$f_{X \setminus Y}(p) = f_X(p) \cdot (1 - f_Y(p)) \quad \text{(Difference)}$$

$$f_{Y \setminus X}(p) = (1 - f_X(p)) \cdot f_Y(p) \quad \text{(Reverse Difference)}$$

여기서 $f_X(p), f_Y(p) \in [0, 1]$은 점 $p$에서 각 primitive의 소프트 점유 함수 값이다.

#### **Step 2: 통합 불리언 연산자 $B_c$ (논문 Eq. 12)**

개별 퍼지 불리언 연산들을 **사면체 무게중심 보간(tetrahedral barycentric interpolation)**으로 결합하여 통합 퍼지 불리언 연산자를 구성한다.

통합 연산자 $B_c$는 다음과 같이 정의된다:

$$\boxed{B_c(x, y) = (c_1 + c_2)\,x + (c_1 + c_3)\,y + (c_0 - c_1 - c_2 - c_3)\,x \cdot y}$$

- $x = f_X(p)$, $y = f_Y(p)$: 두 primitive의 soft occupancy 값
- $\mathbf{c} = (c_0, c_1, c_2, c_3)$: **무게중심 좌표** (barycentric coordinates)
  - 조건: $c_i \geq 0$, $\sum_{i=0}^{3} c_i = 1$
- $\mathbf{c}$의 각 꼭짓점은 union, intersection, difference, reverse-difference에 대응

#### **Step 3: 연속 최적화**

연산자 타입 $\mathbf{c}$를 연속 변수로 처리하면, inverse CSG에서 primitives의 형상 파라미터 $\theta$와 연산자 파라미터 $\mathbf{c}$ 모두에 대해 동시에 경사하강법 적용 가능:

$$\min_{\theta,\, \mathbf{c}} \mathcal{L}\bigl(\text{CSG}(\theta, \mathbf{c}),\, \text{target}\bigr) \quad \text{s.t.} \quad c_i \geq 0,\ \sum_i c_i = 1$$

---

### 2-3. 모델 구조

이 연산자를 활용하여, 각 CSG 내부 노드에서의 불리언 연산 선택을 이산 변수에서 연속 최적화 변수로 전환함으로써 Inverse CSG 최적화를 완화(relax)한다.

```
[CSG Tree 구조]
          B_c (root)
         /          \
      B_c            B_c
     /    \          /   \
prim1   prim2   prim3   prim4

각 B_c 노드: (c₀, c₁, c₂, c₃) 파라미터 → 연속 최적화
각 primitive: 형상 파라미터 θ (위치, 크기, 방향 등) → 연속 최적화
```

무게중심 보간(barycentric interpolation)을 사용하여 서로 다른 불리언 연산자 사이의 단조(monotonic) 보간을 생성한다. 이는 점유값 보간에서의 불필요한 지역 최솟값(local minima)을 방지한다.

단순한 이중선형(bilinear) 보간을 사용하면 union과 difference 사이를 보간할 때 점유값에 추가적인 지역 최솟값이 발생하는 문제가 있다. 이것이 사면체 무게중심 보간을 채택한 핵심 이유이다.

---

### 2-4. 성능 향상

Inverse CSG 최적화 과정에서, 무작위로 초기화된 primitives와 boolean operations(좌측 트리)에서 시작하여 목표 형상에 맞도록 연속 최적화를 진행한 결과, 전통적인 min/max 연산자(빨간색)보다 훨씬 높은 품질(파란색)을 달성한다.

제안된 통합 연산자는 미분 가능성을 보장하고, 소실 기울기를 방지하며, 단조성(monotonicity)을 가져 경사하강법 기반 최적화에 특히 적합함을 입증한다. Inverse CSG 최적화 결과에서 이전 방법 대비 정확도가 크게 향상된다.

Equation 8의 개별 연산자 선택이 날카로운 고체 물체부터 부드러운 유기적 형태 모델링으로의 자연스러운 일반화를 가능하게 한다. 통합 불리언 연산자(Equation 12)와 결합하면, 단일 형상 피팅과 형상 컬렉션 피팅 모두에서 Inverse CSG 작업의 성능이 크게 향상된다.

---

### 2-5. 한계점

현재 퍼지 CSG 시스템은 퍼지 논리 연산자의 최적화되지 않은 구현에 기반하고 있다. 그러나 여러 다른 분야에서 보여주듯, 퍼지 논리 연산자는 병렬 하드웨어 구현으로 크게 가속할 수 있다.

- **트리 구조 자체 최적화 부재**: 현재는 고정된 CSG 트리 구조(topology) 안에서 연산자 타입과 primitive만 최적화하며, 트리 구조 자체를 최적화하는 것은 미래 과제
- **실시간 처리 한계**: 하드웨어 가속 버전이 구현된다면 실시간으로 실행 가능하나, 현재는 아님
- **복잡한 형상 표현의 한계**: 극도로 복잡한 위상 구조를 가진 형상에 대해서는 CSG 트리의 표현력 자체 한계 존재

---

## 3. 🚀 모델의 일반화 성능 향상 가능성

### 3-1. 형상 다양성에 대한 일반화

연속 불리언 연산자를 통해 날카로운 기계적 물체와 부드러운 유기적 형태 모두를 동일한 프레임워크로 모델링할 수 있음을 추가로 입증한다. 즉, 하나의 연산자 수식으로 다양한 형상 클래스를 커버한다.

퍼지 불리언 연산자를 CSG에 사용하면 날카로운 경계를 가진 기계적 물체와 부드러운 유기적 형태를 모두 표현할 수 있다. 특히 기반 암묵적 형상이 이진 점유 함수인 경우, 제안 방법은 전통적인 CSG와 동일한 날카로운 결과를 도출한다.

### 3-2. 신경-기호 하이브리드 모델과의 결합

CSG 시스템을 미분 가능하게 만드는 것은 (블랙박스) 신경 기호 생성 모델이 (화이트박스) CSG 트리 파라미터를 출력하는 미래 연구에 매우 유익하다. 이는 본 논문의 기여와는 직교(orthogonal)하는 방향이므로, 논문에서는 오프-더-셸프 아키텍처 평가에 집중한다.

### 3-3. 컬렉션 수준 일반화

통합 불리언 연산자는 단일 형상 피팅뿐만 아니라 **형상 컬렉션(collection of shapes)**에 대한 inverse CSG 작업에서도 성능 향상을 보임으로써, 단일 인스턴스를 넘어선 범용 모델로의 가능성을 시사한다.

### 3-4. 소실 기울기 방지 → 최적화 안정성 → 일반화

$$\frac{\partial B_c}{\partial x} = (c_1 + c_2) + (c_0 - c_1 - c_2 - c_3) \cdot y \neq 0 \quad \text{(일반적으로 항상 유효)}$$

무게중심 좌표 조건 $c_i \geq 0, \sum c_i = 1$에 의해 기울기가 소실되지 않으므로, 다양한 초기화 조건에서도 안정적으로 최적화되어 일반화 성능이 향상된다.

---

## 4. 🔭 미래 연구에 미치는 영향 및 고려사항

### 4-1. 연구 영향

| 분야 | 영향 |
|------|------|
| **Inverse CSG** | 완전 연속 CSG 최적화를 향한 새로운 기반 제공 |
| **3D 생성 모델** | 뉴럴 네트워크와 CSG 트리의 end-to-end 학습 가능성 |
| **CAD/CAM** | 미분 가능 CAD 파라미터 최적화로의 확장 |
| **뉴로-심볼릭 AI** | 기호 표현(CSG)과 신경망의 미분 가능 연결 강화 |

제안된 불리언 연산자는 **완전 연속 CSG 최적화(fully continuous CSG optimization)** 를 향한 미래 연구의 새로운 가능성을 열어준다.

이러한 연산자의 추가는, 예를 들어 불리언 연산 수행 시 어떤 primitive를 선택할지 결정하는 **트리 구조 최적화**의 새로운 가능성을 열어준다.

### 4-2. 미래 연구 시 고려사항

1. **트리 위상(Topology) 최적화**: 현재는 고정 트리 구조에서 연산자만 최적화하므로, 트리의 깊이·분기 자체를 최적화하는 방법 연구 필요
2. **하드웨어 가속**: 퍼지 논리 연산자는 병렬 하드웨어로 크게 가속될 수 있으며, 하드웨어 가속 버전은 CSG 시스템을 실시간으로 실행할 수 있게 해준다.
3. **신경망 아키텍처 탐색**: CSG 시스템을 미분 가능하게 만드는 것은 CSG 트리 파라미터를 출력하는 신경-기호 생성 모델(neural symbolic generative models)에 대한 미래 탐색에 유익할 것이다.
4. **스케일 확장**: 대규모 형상 데이터셋에서의 일반화 성능 검증
5. **다른 표현과의 결합**: NeRF, 3D Gaussian Splatting 등 최신 암묵적 표현과의 통합

---

## 5. 📊 2020년 이후 관련 최신 연구 비교 분석

| 논문 | 연도 | 핵심 방법 | 차별점 vs. 본 논문 |
|------|------|-----------|-------------------|
| **UCSG-NET** | NeurIPS 2020 | 비지도 학습으로 CSG 파스 트리 추출; 미분 가능 indicator 함수를 통해 SDF를 이진화하고 불리언 연산자 트리 구조를 발견 | 연산자 타입 최적화가 이산적, 소실 기울기 문제 존재 |
| **Analyzing Differentiable Fuzzy Logic Operators** | AI Journal 2022 | 미분 가능 퍼지 논리 연산자 분석 (Artificial Intelligence 302, 2022) | 3D 형상이 아닌 일반 논리 추론에 집중 |
| **CSG on Neural SDF** | SIGGRAPH Asia 2023 | 신경 Signed Distance Field 위에서의 CSG 연산 (SIGGRAPH Asia 2023) | SDF 기반, 본 논문의 soft occupancy 기반과 상보적 |
| **DiffCSG** | SIGGRAPH Asia 2024 | 래스터라이제이션을 통한 미분 가능 CSG (SIGGRAPH Asia 2024, Tokyo) | 렌더링 기반 접근, 본 논문은 점유 함수 기반 최적화 |
| **본 논문 (Fuzzy Boolean)** | SIGGRAPH 2024 | 퍼지 논리 + 사면체 무게중심 보간으로 연속·미분 가능 통합 연산자 | **연산자 타입 자체를 연속 변수로 최적화, 소실기울기 방지, 두 형상 클래스 통합** |

---

## 📚 참고 자료 및 출처

1. **arXiv 원문**: *A Unified Differentiable Boolean Operator with Fuzzy Logic*, arXiv:2407.10954 (2024) — https://arxiv.org/abs/2407.10954
2. **공식 PDF**: https://www.dgp.toronto.edu/~hsuehtil/pdf/fuzzyBoolean.pdf
3. **ACM SIGGRAPH 2024 DOI**: https://dl.acm.org/doi/10.1145/3641519.3657484
4. **Roblox Research 게시**: https://research.roblox.com/publications/a-unified-differentiable-boolean-operator-with-fuzzy-logic
5. **GitHub 코드**: https://github.com/HTDerekLiu/fuzzy-boolean
6. **DBLP 서지정보**: https://dblp.org/rec/journals/corr/abs-2407-10954.html
7. **HTML 전문 (arXiv)**: https://arxiv.org/html/2407.10954v1
8. **관련 논문 - UCSG-NET** (NeurIPS 2020): *UCSG-NET - Unsupervised Discovering of Constructive Solid Geometry Tree*, https://papers.neurips.cc/paper_files/paper/2020/file/63d5fb54a858dd033fe90e6e4a74b0f0-Paper.pdf
9. **관련 논문 - Analyzing Differentiable Fuzzy Logic Operators** (AI Journal 2022): Emile van Krieken et al., *Analyzing differentiable fuzzy logic operators*, Artificial Intelligence 302 (2022), 103602
10. **관련 논문 - DiffCSG** (SIGGRAPH Asia 2024): *DiffCSG: Differentiable CSG via Rasterization*, arXiv:2409.01421
11. **관련 논문 - CSG on Neural SDF** (SIGGRAPH Asia 2023): Zoë Marschner et al., *Constructive Solid Geometry on Neural Signed Distance Fields*

> ⚠️ **정확도 주의**: 개별 수식(Eq. 8의 union/intersection/difference 전개)은 논문 PDF 원문 및 arXiv HTML에서 확인 가능한 내용을 기반으로 작성하였으나, 논문 내 일부 세부 실험 수치(IoU 등 정량 결과)는 원문 직접 확인을 권장합니다.
