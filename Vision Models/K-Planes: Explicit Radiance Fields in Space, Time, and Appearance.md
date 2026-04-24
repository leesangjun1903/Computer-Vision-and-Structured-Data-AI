
# K-Planes: Explicit Radiance Fields in Space, Time, and Appearance

> **논문 정보**
> - **제목:** K-Planes: Explicit Radiance Fields in Space, Time, and Appearance
> - **저자:** Sara Fridovich-Keil\*, Giacomo Meanti\*, Frederik Rahbæk Warburg, Benjamin Recht, Angjoo Kanazawa
> - **학회:** CVPR 2023 (pp. 12479–12488)
> - **arXiv:** [2301.10241](https://arxiv.org/abs/2301.10241)

---

## 1. 핵심 주장 및 주요 기여 요약

K-Planes는 임의의 차원에서 radiance field를 표현하는 **white-box 모델**입니다. 이 모델은 $d$차원 장면을 $\binom{d}{2}$개의 평면(plane)으로 표현하며, 정적 장면($d=3$)에서 동적 장면($d=4$)으로 자연스럽게 확장됩니다.

이 평면 분해(planar factorization) 방식은 시간적 부드러움(temporal smoothness)이나 다중 해상도 공간 구조(multi-resolution spatial structure)와 같은 차원별 사전 정보(dimension-specific priors)를 쉽게 추가하고, 장면의 정적·동적 구성 요소를 자연스럽게 분리합니다.

또한 학습된 색상 기저(learned color basis)와 선형 특징 디코더를 사용하여 비선형 블랙박스 MLP 디코더와 유사한 성능을 발휘하며, 합성·실제, 정적·동적, 고정·변화 외관 장면에 걸쳐 경쟁력 있거나 종종 최고 수준의 재구성 충실도를 달성합니다. 전체 4D 그리드 대비 **1000배 압축**을 달성하며 순수 PyTorch 구현으로 빠른 최적화를 지원합니다.

### 주요 기여 요약

| 기여 | 설명 |
|------|------|
| **White-box 모델** | MLP에 의존하지 않는 완전 명시적(explicit) 표현 |
| **임의 차원 확장** | $d=3$ (정적) → $d=4$ (동적)으로 seamless 전환 |
| **평면 분해 프레임워크** | $\binom{d}{2}$개의 평면으로 $d$차원 장면 표현 |
| **선형 디코더 + 학습된 색상 기저** | SH 기저의 한계 극복 |
| **1000x 압축** | 전체 4D 그리드 대비 메모리 절감 |
| **빠른 최적화** | 단일 GPU에서 수 시간 내 훈련 가능 |

---

## 2. 논문 상세 설명

### 2.1 해결하고자 하는 문제

**기존 방법의 한계:**

기존의 Plenoxels처럼 3D 볼륨을 명시적 그리드로 표현하는 방법과 DVGO는 차원이 늘어날수록 지수적으로 메모리가 증가하여, 고해상도로 확장하기 어렵고 4D 동적 볼륨에는 완전히 비실용적입니다.

하이브리드 방법들은 정적 장면에서 메모리 효율과 최적화 시간의 균형을 제공하지만, 이러한 분해 방식을 메모리 효율적으로 4D 볼륨으로 확장하는 방법은 명확하지 않습니다.

K-Planes는 이 문제를 **평면 분해(planar factorization)** 로 해결하려고 합니다.

---

### 2.2 제안하는 방법 (수식 포함)

#### (1) 핵심 아이디어: $\binom{d}{2}$-플레인 분해

$d$차원 공간에 대해 $\binom{d}{2}$개의 2D 평면을 생성합니다.

- 정적 장면 ($d = 3$): 3개의 평면 $P_{xy},\, P_{xz},\, P_{yz}$
- 동적 장면 ($d = 4$): 6개의 평면 $P_{xy},\, P_{xz},\, P_{yz},\, P_{xt},\, P_{yt},\, P_{zt}$

#### (2) 포인트 쿼리: 다중 스케일 쌍선형 보간 + Hadamard 곱

K-Planes의 표현은 4D 동적 볼륨을 6개의 평면으로 분해합니다(3개는 공간, 3개는 시공간). 4D 포인트 $\mathbf{q}=(x,y,z,t)$의 값을 얻기 위해 포인트를 각 평면에 투영하고, 다중 스케일 쌍선형 보간을 수행합니다. 보간된 값들은 $S$개 스케일에 걸쳐 원소별 곱셈(Hadamard product)된 후 연결(concatenate)됩니다. 이 특징들은 소형 MLP 또는 명시적 선형 디코더로 디코딩됩니다.

수식으로는:

$$
\mathbf{f}(\mathbf{q}) = \bigotimes_{(i,j) \in \binom{[d]}{2}} \text{interp}\!\left(P_{ij},\, \pi_{ij}(\mathbf{q})\right)
$$

여기서:
- $\pi_{ij}(\mathbf{q})$: 포인트 $\mathbf{q}$를 $(i,j)$ 평면에 투영
- $\text{interp}(\cdot)$: 다중 스케일 쌍선형 보간
- $\bigotimes$: Hadamard(원소별) 곱 후 concatenate

멀티스케일의 경우:

$$
\mathbf{f}^{\text{multi}}(\mathbf{q}) = \bigoplus_{s=1}^{S} \left( \bigotimes_{(i,j)} \text{interp}_s\!\left(P^{(s)}_{ij},\, \pi_{ij}(\mathbf{q})\right) \right)
$$

여기서 $\bigoplus$는 concatenation이고 $s$는 스케일 인덱스입니다.

#### (3) 왜 덧셈이 아닌 Hadamard 곱인가?

평면 특징의 원소별 덧셈과 곱셈(오른쪽)을 비교하면, triplane 예시에서 각 평면의 한 항목만 양수이고 나머지가 0인 경우, 곱셈은 단일 3D 포인트를 선택하지만 덧셈은 교차하는 선을 생성합니다. 이 곱셈의 선택 능력이 명시적 모델의 표현력을 향상시킵니다.

평면 특징의 곱셈은 명시적 모델에서 PSNR을 크게 향상시키는 반면, 하이브리드 모델은 MLP 디코더를 사용해 덜 표현력 있는 덧셈을 부분적으로 보완할 수 있습니다.

#### (4) 체적 렌더링 (Volumetric Rendering)

표준 NeRF 체적 렌더링 수식을 따릅니다:

$$
\hat{C}(\mathbf{r}) = \sum_{k=1}^{N} T_k \cdot \alpha_k \cdot \mathbf{c}_k
$$

$$
T_k = \prod_{j=1}^{k-1}(1-\alpha_j), \quad \alpha_k = 1 - \exp(-\sigma_k \delta_k)
$$

여기서 $\sigma_k$는 밀도, $\mathbf{c}_k$는 색상, $\delta_k$는 샘플 간격입니다.

표준 체적 렌더링 공식을 따라 색상과 밀도를 예측하며, 공간과 시간에 대한 단순한 정규화와 함께 재구성 손실을 최소화하여 모델을 최적화합니다.

---

### 2.3 정규화 (Regularization)

논문에서 제안하는 세 가지 핵심 정규화 기법입니다.

#### (a) 공간 총 변분 (TV in Space)

$$
\mathcal{L}_{\text{TV}}(\mathbf{P}) = \frac{1}{|C|n^2} \sum_{c,i,j} \left( \|\mathbf{P}_c^{i,j} - \mathbf{P}_c^{i-1,j}\|_2^2 + \|\mathbf{P}_c^{i,j} - \mathbf{P}_c^{i,j-1}\|_2^2 \right)
$$

#### (b) 시간 부드러움 (Smoothness in Time)

시간에 걸친 급격한 "가속도"를 패널티로 부과하기 위해 아래와 같은 정규화를 적용합니다.

$$
\mathcal{L}_{\text{time}}(\mathbf{P}) = \frac{1}{|C|n^2} \sum_{c,i,t} \|\mathbf{P}_c^{i,t-1} - 2\mathbf{P}_c^{i,t} + \mathbf{P}_c^{i,t+1}\|_2^2
$$

이 정규화는 시공간 평면($P_{xt},\, P_{yt},\, P_{zt}$)의 시간 차원에만 적용됩니다.

#### (c) 희소 과도 현상 (Sparse Transients)

$$
\mathcal{L}_{p}(\mathbf{P}) = \sum_c \|\mathbf{1} - \mathbf{P}_c\|_1, \quad c \in \{xt, yt, zt\}
$$

이는 시공간 평면이 대부분 항등(identity)이 되도록 강제하여, 동적 요소가 실제로 변화하는 부분에만 집중하도록 합니다.

---

### 2.4 모델 구조

```
입력: 3D/4D 포인트 q = (x, y, z) 또는 (x, y, z, t)
         ↓
[다중 스케일 평면 분해]
  - 정적 (d=3): P_xy, P_xz, P_yz (3개 평면)
  - 동적 (d=4): P_xy, P_xz, P_yz, P_xt, P_yt, P_zt (6개 평면)
         ↓
[다중 스케일 쌍선형 보간 (S scales)]
         ↓
[Hadamard 곱 → Concatenation]
         ↓
[디코더 선택]
  - Hybrid 모델: 소형 MLP 디코더
  - Explicit 모델: 선형 디코더 + 학습된 색상 기저
         ↓
[체적 렌더링 → 색상/밀도 예측]
         ↓
출력: Ray color C(r)
```

K-Planes는 장면을 공간 전용 평면과 시공간 평면으로 분해하는 최초의 white-box, 해석 가능한 radiance field 모델입니다. 학습된 색상 기저를 갖는 선형 특징 디코더를 사용하여 비선형 블랙박스 MLP 디코더와 유사한 성능을 내면서도 더 높은 투명성과 적응성을 제공합니다. 또한 훈련 이미지당 전역 외관 코드(global appearance code)를 통합하여 기하학에 영향을 주지 않고 조명 등 외관 변화를 처리합니다.

**외관 변화(Appearance Variation)** 처리:

NeRF-W처럼 외관 코드를 보간하여 시간대 변화와 같은 랜드마크의 시각적 외관을 변환할 수 있습니다. 외관 코드는 색상에 영향을 주지만 기하학에는 영향을 미치지 않습니다.

---

### 2.5 성능 향상

K-Planes 모델은 정적, 변화하는 외관, 동적 장면 등 다양한 장면과 작업에 적용됩니다. 메모리 사용량이 낮고(compact) 빠른 훈련·추론 시간을 갖습니다.

명시적 모델과 하이브리드 모델 모두 실제 멀티뷰 동적 장면에서 최고 수준의 재구성을 달성합니다.

**평가 데이터셋:**
- **정적 장면:** NeRF 합성 데이터셋(Lego 등), LLFF (forward-facing scenes)
- **동적 장면:** DyNeRF (실제 멀티뷰 동적), D-NeRF (단안 합성 동적)
- **변화 외관:** Phototourism 데이터셋

**압축 효율:**

합성·실제, 정적·동적, 고정·변화 외관 장면에서 경쟁력 있거나 최고 수준의 재구성 충실도를 낮은 메모리 사용량으로 달성하며, 전체 4D 그리드 대비 1000배 압축을 달성합니다.

---

### 2.6 한계 (Limitations)

기존 연구 및 후속 연구에서 지적된 K-Planes의 한계:

1. **씬별(per-scene) 최적화:** 새로운 장면마다 별도의 학습이 필요하며, 일반화된 피드포워드 추론이 불가능합니다.

2. **저주파 표현의 어려움:**
이러한 표현들은 특히 동적 장면에 적용할 때 다중 스케일 표현을 사용해도 저주파 세부 사항에 어려움을 겪고 고주파 신호에 과적합되는 경향이 있습니다. TV 페널티와 같은 노이즈 제거 패널티의 도움으로 부분적인 성공을 거두었지만, 저주파 스펙트럼 특징의 표현이 여전히 부족합니다.

3. **연속 시간 축 이산화:**
HexPlane/K-Planes와 같이 연속적인 시간 축을 유한한 구간(bin)으로 이산화하면, 훈련 포즈가 희소한 경우 시간에 따라 변하는 물체의 움직임에 덜 반응합니다.

4. **고차원 확장성의 제한:**
K-Planes와 HexPlane은 3D+time 이상의 고차원을 처리하도록 설계되지 않았습니다.

5. **스펙큘러(반사) 동적 표면에 취약:** 동적 반사 표면의 처리에 특화된 설계가 없습니다.

---

## 3. 모델의 일반화 성능 향상 가능성

K-Planes는 기본적으로 **씬별 최적화 모델**입니다. 그러나 다음과 같은 구조적 특성이 일반화 성능 향상의 잠재력을 갖습니다.

### 3.1 내재적 일반화 기제

**① 차원 독립적 평면 분해**

임의의 차원 공간으로 자연스럽게 확장되며, 차원에 따라 최적화 시간과 모델 크기가 부드럽게 스케일됩니다.

→ 이는 새로운 차원(예: 조명, 재질 등)으로의 일반화 가능성을 시사합니다.

**② 시공간 분리(Space-Time Decomposition)**

K-Planes는 4D 비디오를 정적 구성 요소와 동적 구성 요소로 자연스럽게 분리합니다. 시간 평면을 항등으로 설정함으로써 정적 부분을 렌더링하고, 나머지가 동적 부분이 됩니다.

→ 이 분리 구조는 부분 전이(partial transfer) 학습의 토대가 됩니다.

**③ 외관 코드(Appearance Code) 보간**

외관 코드를 보간하여 명시적 모델이나 하이브리드 모델에서 랜드마크의 시각적 외관(예: 시간대 변화)을 변환할 수 있습니다. 외관 코드는 색상에 영향을 미치지만 기하학에는 영향을 미치지 않습니다.

→ 이 매커니즘은 **도메인 전이**에 유리하며, 외관 공간을 통한 새로운 장면 적응을 가능하게 합니다.

**④ 선형 디코더의 해석 가능성**

선형 특징 디코더와 학습된 색상 기저를 사용하여 비선형 블랙박스 MLP 디코더와 유사한 성능을 달성합니다.

→ 선형 디코더는 특징 공간이 해석 가능하고 전이 가능성이 높습니다.

### 3.2 일반화 성능 향상을 위한 방향성

| 방향 | 설명 |
|------|------|
| **사전 훈련 + 미세 조정** | 다수의 장면에서 평면을 사전 학습 후 새 장면에 적응 |
| **좌표 네트워크 결합** | 연속적 좌표 기반 인코딩과 결합하여 저주파 표현 강화 |
| **희소 뷰(sparse-view) 정규화** | TV 정규화 외 추가적 공간적 일관성 제약 |
| **외관 공간 학습** | 다수의 장면에 걸쳐 공통 외관 공간 학습 |
| **4D GS와의 결합** | 3D Gaussian Splatting의 실시간 렌더링과 결합 |

희소 입력 NeRF에서는 NeRF가 일반적으로 특정 장면에 맞춤화된 하나의 네트워크를 사용하기 때문에 훈련 데이터에 과적합되지 않는 것이 중요합니다. 특히 그리드 구조의 국소적 업데이트에 기반한 명시적 표현들은 전역 맥락을 캡처하는 데 크게 어려움을 겪습니다.

---

## 4. 최신 연구 비교 분석 (2020년 이후)

### 4.1 관련 연구 계보

```
NeRF (2020)
    ↓
Plenoxels / DVGO (2022) — 명시적 그리드, 빠른 학습
    ↓
TensoRF (2022) — 텐서 분해 (CP/VM)
    ↓
K-Planes / HexPlane (CVPR 2023) — 평면 분해, 4D 확장
    ↓
4D Gaussian Splatting / WavePlanes (2023~2024) — GS 기반 동적 표현
    ↓
LVSM (2024~2025) — 대규모 피드포워드 NVS 모델
```

### 4.2 주요 논문 비교표

| 방법 | 표현 방식 | 4D 지원 | 메모리 | 훈련 속도 | 일반화 |
|------|-----------|---------|--------|-----------|--------|
| **NeRF** (2020) | 암시적 MLP | ✗ | 낮음 | 매우 느림 | ✗ |
| **TensoRF** (2022) | 텐서 분해 (VM) | ✗ | 낮음 | 빠름 | ✗ |
| **HexPlane** (2023) | 6-평면 | ✓ | 낮음 | 빠름 | ✗ |
| **K-Planes** (2023) | $\binom{d}{2}$-평면 | ✓ | 낮음 | 빠름 | ✗ |
| **4D-GS** (2023) | 4D Gaussian | ✓ | 중간 | 빠름 | ✗ |
| **WavePlanes** (2024) | 웨이블릿+평면 | ✓ | 매우 낮음 | 빠름 | ✗ |
| **LVSM** (2024) | 트랜스포머 | ✗ | 높음 | 빠름 | ✓ |

TensoRF는 복셀 그리드를 평면과 벡터의 텐서 분해로 대체하여 유사한 모델 압축과 속도를 달성했습니다.

K-Planes와 HexPlane은 TensoRF에서 영감을 받아 동적 장면을 재구성하기 위한 분해된 4D NeRF 표현으로 확장됩니다.

**HexPlane vs K-Planes:**

arXiv 공개 하루 전, 유사한 논문인 HexPlane이 arXiv에 공개되었습니다.

두 논문은 독립적으로 유사한 6-평면 분해 아이디어를 제안했습니다. K-Planes는 임의의 $d$차원으로의 일반화와 white-box 해석 가능성을 더 강조하며, 외관 코드를 통해 Phototourism 같은 데이터셋도 처리합니다.

**WavePlanes의 K-Planes 개선:**

WavePlanes는 NeRF와 GS 파이프라인 모두에 적용 가능하며, 낮은 계산 비용, 경쟁력 있는 품질로 다양한 장면을 합성합니다. 웨이블릿 표현의 특성을 활용하여 추가 훈련이나 매개변수 없이 품질 손실 없이 소형 동적 NVS 모델을 생성합니다.

**4D Gaussian Splatting의 등장:**

HexPlane과 K-Planes는 공간-시간 볼륨을 여러 평면으로 분해하여 NeRF를 가속화합니다. 이에 비해 4D-GS는 가우시안 프리미티브 기반의 실시간 렌더링을 제공합니다.

**LVSM의 새로운 방향:**

LVSM은 3D 표현(NeRF, 3DGS)에서 네트워크 설계(에피폴라 투영, 평면 스윕)에 이르는 이전 방법들에서 사용된 3D 귀납적 편향을 우회하며, 완전히 데이터 기반 접근법으로 새로운 뷰 합성을 다룹니다. 디코더 전용 LVSM은 우수한 품질, 확장성 및 제로샷 일반화를 달성합니다.

---

## 5. 향후 연구에 미치는 영향 및 고려할 점

### 5.1 연구에 미치는 영향

**① 평면 분해의 표준화**

K-Planes와 HexPlane이 CVPR 2023에서 동시에 발표되면서, **평면 기반 4D 표현**이 동적 NeRF 연구의 사실상 표준 접근법으로 자리잡았습니다.

동적 NVS를 달성하는 데 있어 평면 기반 표현은 4D 특징 표현을 저랭크(low-rank) 및 컴팩트한 2D 구성 요소로 명시적으로 디코딩하는 일반적인 방법이 되었으며, 이 혁신은 NeRF 모델링에서 K-Planes에 의해 대중화되었습니다.

**② Gaussian Splatting과의 결합**

K-Planes의 평면 분해 아이디어는 4D-GS 등의 후속 연구에서 Gaussian Splatting과 결합되어 더욱 빠른 동적 장면 렌더링을 가능하게 했습니다.

**③ 해석 가능성(Interpretability)의 재조명**

K-Planes는 장면을 공간 전용 평면과 시공간 평면으로 분해하는 최초의 white-box, 해석 가능한 radiance field 모델로, 해석 가능성을 높이고 차원별 사전 정보를 가능하게 합니다. 선형 특징 디코더와 학습된 색상 기저를 사용하여 비선형 MLP 디코더와 유사한 성능을 내면서 더 높은 투명성을 제공합니다.

**④ 다차원 확장의 청사진**

$\binom{d}{2}$ 평면 공식은 미래 연구에서 조명, 재질, 시점 등 추가 차원을 포함한 더 높은 차원의 표현을 탐구하는 이론적 토대를 제공합니다.

---

### 5.2 앞으로 연구 시 고려할 점

**① 희소 뷰(Sparse-View) 일반화 문제**

동적 장면에서 시간 희소성을 다루는 후속 연구가 등장한 이후, 데이터 희소성 처리가 이 분야에서 더 많은 주목을 받고 있습니다. NeRF 모델들은 3D 또는 4D 공간에 대한 일관된 데이터 부족으로 인해 과적합 문제를 흔히 겪기 때문입니다.

→ **연구 과제:** 사전 훈련된 이미지 인코더 활용 또는 깊이/색상 제약 추가를 통한 희소 데이터 처리 방안

**② 연속 시간 모델링**

K-Planes는 시간 축을 이산화(discretize)합니다. 이를 연속 시간으로 모델링하거나, 시간 좌표 기반 네트워크를 결합하면 더 자연스러운 동적 표현이 가능합니다.

**③ 피드포워드 일반화 모델로의 발전**

일반화 가능한 방법들은 장면 전반에 걸쳐 훈련된 신경망을 사용하여 피드포워드 방식으로 새로운 뷰 또는 기저 3D 표현을 예측함으로써 빠른 NVS 추론을 가능하게 합니다.

→ K-Planes의 평면 구조에 메타 학습 또는 대규모 사전 훈련을 결합하면 씬별 최적화 없는 일반화 모델로 발전 가능

**④ 대규모 동적 장면으로의 확장**

현재 K-Planes는 제한된 시간 범위의 동적 장면에 최적화되어 있습니다. 긴 시간 범위, 대규모 야외 장면, 심한 움직임이 있는 장면에서의 확장이 중요한 연구 과제입니다.

**⑤ 3D Gaussian Splatting과의 통합**

3D Gaussian Splatting은 신경망의 필요성을 제거하여 훈련 및 추론 속도를 크게 높이는 효율적인 명시적 접근 방식을 제공합니다.

→ K-Planes의 평면 기반 특징 표현을 3DGS의 실시간 렌더링과 결합하는 연구가 유망합니다.

**⑥ 물리 기반 렌더링(PBR)과의 결합**

반사, 굴절 등 물리 기반 재질 표현과 K-Planes 구조를 결합하면 특수 소재의 동적 장면 처리가 가능합니다.

---

## 📚 참고 자료 및 출처

| # | 자료 | 링크/출처 |
|---|------|-----------|
| 1 | **K-Planes 논문 (arXiv)** | https://arxiv.org/abs/2301.10241 |
| 2 | **K-Planes 논문 (CVPR 2023 Open Access)** | https://openaccess.thecvf.com/content/CVPR2023/papers/Fridovich-Keil_K-Planes_Explicit_Radiance_Fields_in_Space_Time_and_Appearance_CVPR_2023_paper.pdf |
| 3 | **K-Planes 공식 프로젝트 페이지** | https://sarafridov.github.io/K-Planes/ |
| 4 | **K-Planes GitHub 저장소** | https://github.com/sarafridov/K-Planes |
| 5 | **CVPR 2023 포스터 페이지** | https://cvpr2023.thecvf.com/virtual/2023/poster/21220 |
| 6 | **IEEE Xplore (CVPR 2023)** | https://ieeexplore.ieee.org/document/10204118 |
| 7 | **DTU Research Database** | https://orbit.dtu.dk/en/publications/ik-iplanes-explicit-radiance-fields-in-space-time-and-appearance/ |
| 8 | **ResearchGate** | https://www.researchgate.net/publication/367389203_K-Planes_Explicit_Radiance_Fields_in_Space_Time_and_Appearance |
| 9 | **WavePlanes (arXiv 2023)** | https://arxiv.org/html/2312.02218 |
| 10 | **Synergistic Integration of Coordinate Network and Tensorial Feature (arXiv 2024)** | https://arxiv.org/html/2405.07857 |
| 11 | **Dynamic Scene Reconstruction Survey (arXiv 2025)** | https://arxiv.org/pdf/2503.08166 |
| 12 | **LVSM: A Large View Synthesis Model (ICLR 2025)** | https://arxiv.org/html/2410.17242v1 |
| 13 | **Refined Tensorial Radiance Field (NeurIPS 2023 Workshop)** | https://openreview.net/pdf/300e49f7756eed9e0e5fce6cc61ad1e55bbd29fb.pdf |
| 14 | **HexPlane: A Fast Representation for Dynamic Scenes (CVPR 2023)** | https://caoang327.github.io/HexPlane/ |
| 15 | **ViSNeRF (arXiv 2025)** | https://arxiv.org/html/2502.16731v1 |

# K-Planes: Explicit Radiance Fields in Space, Time, and Appearance

### 핵심 주장과 주요 기여[1]

K-Planes는 **임의의 차원(arbitrary dimensions)에서 방사성 장(radiance fields)을 표현하는 화이트박스 모델**로 제안됩니다. 이 모델의 핵심 기여는 다음과 같습니다.

#### 1. 평면 인수분해(Planar Factorization) 전략[1]

K-Planes는 $$d$$-차원 장면을 $$\binom{d}{2}$$개의 2차원 평면으로 분해합니다. 구체적으로:

- **정적 3D 장면(d=3)**: 3개의 평면(**트라이플레인**)으로 표현 - xy, xz, yz 평면
- **동적 4D 장면(d=4)**: 6개의 평면(**헥스플레인**)으로 표현 - 공간 평면 3개(xy, xz, yz) + 시공간 평면 3개(xt, yt, zt)

이 설계는 **메모리 효율성**과 **해석 가능성**이라는 두 가지 장점을 제공합니다.

#### 2. 곱셈 기반 특징 결합[1]

K-Planes의 중요한 개선 사항은 **Hadamard 곱(elementwise multiplication)**을 통해 평면들의 특징을 결합한다는 점입니다. 4D 점 $$\mathbf{q}=(i,j,k,\tau)$$의 특징은 다음과 같이 계산됩니다:

$$
f(\mathbf{q}) = \prod_{c \in C} f(\mathbf{q})_c
$$

여기서 $$f(\mathbf{q})_c$$는 각 평면 $$c$$에 대한 이중선형 보간(bilinear interpolation) 결과입니다. **덧셈 대신 곱셈을 사용하는 이유**는 공간적으로 국소화된 신호를 생성하는 능력에 있습니다. Table 2에서 곱셈은 명시적 모델에서 PSNR 35.29 대비 덧셈은 28.78로 큰 성능 차이를 보입니다.[1]

#### 3. 화이트박스 설계[1]

K-Planes는 **해석 가능한 모델**을 추구하며, 이는 두 가지 선택으로 구현됩니다:

1. **선형 특징 디코더**: 구면 조화 함수(SH) 대신 **학습된 색상 기저(learned color basis)**를 사용하여 뷰 의존 색상을 모델링합니다. MLP가 뷰-의존 색상과 공간 구조 결정을 모두 수행하는 것을 완화합니다.

2. **명시적 정적-동적 분해**: 공간 전용 평면과 시공간 평면의 분리로 인해 동적 영역이 명확하게 시각화됩니다.

### 해결하는 문제 및 제안 방법[1]

#### 문제 정의

기존 접근 방식의 한계:

- **NeRF**: 느린 최적화(수 시간 ~ 수일), 블랙박스 MLP 사용
- **Plenoxels, DVGO**: 3D 그리드는 차원 증가에 따라 지수적으로 메모리 증가
- **Tensor4D**: 9개 평면 사용으로 중복성 존재 (yt 평면 2개)

#### 제안 방법의 수식

**투영(Projection)**:

$$
f(\mathbf{q})_c = \psi\left(\mathbf{P}_c, \pi_c(\mathbf{q})\right) \quad \text{(식 1)}
$$

**특징 결합**:

$$
f(\mathbf{q}) = \prod_{c \in C} f(\mathbf{q})_c \quad \text{(식 2)}
$$

**색상 디코딩**:

$$
\mathbf{c}(\mathbf{q}, \mathbf{d}) = \bigcup_{i \in \{R,G,B\}} f(\mathbf{q}) \cdot \mathbf{b}_i(\mathbf{d}) \quad \text{(식 6)}
$$

**밀도 디코딩**:

$$
\sigma(\mathbf{q}) = f(\mathbf{q}) \cdot \mathbf{b}_{\sigma} \quad \text{(식 7)}
$$

### 정규화 항들[1]

모델은 다음 정규화항들로 학습됩니다:

**공간 전체변분(Total Variation)**:

$$
\mathcal{L}_{TV}(\mathbf{P}) = \frac{1}{|C|n^2} \sum_{c,i,j} \left(\left\|\mathbf{P}_c^{i,j} - \mathbf{P}_c^{i-1,j}\right\|_2^2 + \left\|\mathbf{P}_c^{i,j} - \mathbf{P}_c^{i,j-1}\right\|_2^2\right) \quad \text{(식 3)}
$$

**시간 평활성(Temporal Smoothness)**:

$$
\mathcal{L}_{smooth}(\mathbf{P}) = \frac{1}{|C|n^2} \sum_{c,i,t} \left\|\mathbf{P}_c^{i,t-1} - 2\mathbf{P}_c^{i,t} + \mathbf{P}_c^{i,t+1}\right\|_2^2 \quad \text{(식 4)}
$$

**희소 일시적 변화(Sparse Transients)**:

$$
\mathcal{L}_{sep}(\mathbf{P}) = \sum_c \left\|\mathbf{1} - \mathbf{P}_c\right\|_1, \quad c \in \{xt, yt, zt\} \quad \text{(식 5)}
$$

### 모델 구조[1]

K-Planes는 **다중 스케일 평면(Multiscale Planes)** 구조를 채택합니다:

- 공간 해상도: 64, 128, 256, 512 (실험에서 사용)
- 각 스케일의 특징 길이: M=32 (기본값)
- 서로 다른 스케일의 M-차원 특징 벡터는 **연결(concatenation)**되어 디코더에 전달됩니다.

#### 두 가지 디코더 버전

1. **명시적 버전**: 선형 디코더 + 학습된 색상 기저 (MLP 기반 기저)
2. **하이브리드 버전**: 두 개의 작은 MLP (밀도용 $$g_{\sigma}$$, 색상용 $$g_{RGB}$$)

### 성능 향상[1]

#### 정량적 결과 (Table 3)

| 데이터셋 | 메트릭 | K-Planes (명시) | K-Planes (하이브리드) | 최고 기존 방법 |
|---------|---------|-----------------|----------------|---------|
| NeRF (합성, 정적) | PSNR | 32.21 | 32.36 | TensoRF: 33.14 |
| LLFF (실제, 정적) | PSNR | 26.78 | 26.92 | TensoRF: 26.73 |
| D-NeRF (합성, 동적) | PSNR | 31.05 | 31.61 | V4D: 33.72 |
| DyNeRF (실제, 동적) | PSNR | 30.88 | 31.63 | Mix Voxels: 30.80 |

#### 주요 성과

1. **메모리 효율성**: 4D 그리드 대비 **1000배 압축** (300GB → 200MB)[1]
2. **최적화 속도**: DyNeRF 대비 **약 370배 빠름** (1344시간 → 3.7시간)[1]
3. **해석 가능성**: 동적-정적 성분 자동 분해 가능[1]

### 모델의 한계[1]

1. **성능 격차**: 일부 데이터셋에서 V4D, TiNeuVox 등 최신 방법에 비해 낮은 성능
2. **Phototourism 성능**: NeRF-W (PSNR 27.00) 대비 K-Planes (PSNR 22.92) 성능 차이[1]
3. **고주파 세부사항**: 명시적 선형 디코더는 복잡한 반사 특성 모델링에 제한
4. **모노큘러 동영상 한계**: 중요 샘플링(importance sampling)을 사용할 수 없음[1]

### 모델의 일반화 성능[1]

K-Planes의 일반화 성능과 관련된 핵심 특징:

#### 1. 장면 특정 최적화(Scene-Specific Optimization)

K-Planes는 다른 대부분의 방사성 장 방법처럼 **각 장면별로 개별 최적화**를 수행합니다. 이는:
- 각 장면마다 새로운 모델 매개변수를 학습해야 함
- 기존 학습 없이 전이 학습(transfer learning) 불가능

#### 2. 일반화 성능 개선 가능성

논문은 직접적으로 다루지 않지만, **구조적 이점**으로 인한 개선 가능성이 있습니다:

**다중 스케일 표현**: 64, 128, 256, 512 해상도의 계층적 구조는 **저주파에서 고주파까지 점진적 학습**을 가능하게 하여, 적은 관찰 데이터에서도 견고한 기하학 학습을 촉진합니다.

**명시적 정규화 제약**: 시간 평활성, 공간 TV 정규화, 희소 일시적 변화 제약은 **과적합 방지**에 도움이 됩니다.

### 앞으로의 연구 영향과 고려사항

#### 1. 현재 연구 트렌드[2][3][4][5][6][7]

**일반화 가능한 NeRF(Generalizable NeRF)**: 2024-2025년 연구는 **모든 장면에 적용되는 단일 모델** 개발에 집중합니다:

- **MRVM-NeRF** (2024): 마스크 기반 모델링으로 장면 간 일반화 개선
- **GSNeRF** (2024): 의미론적 정보와 함께 일반화
- **ID-NeRF** (2025): 사전 훈련된 확산 모델 기반 가이던스로 제한된 데이터 환경에서 일반화 향상

#### 2. 소수 샷 학습(Few-Shot Learning) 분야[3][4][5][6][2]

K-Planes의 **계층적 다중 스케일 구조**는 소수 샷 설정에 이상적입니다:

- **DWTNeRF** (2025): Instant-NGP 기반으로 해시 인코딩과 이산 웨이블릿 손실 결합 (3샷 LLFF에서 PSNR 15.07% 개선)
- **SANeRF** (2024): 공간 어닐링으로 다중 스케일 표현 최적화
- **FrugalNeRF** (2024): 가중치 공유 복셀과 교차 스케일 기하학 적응

이러한 방법들은 K-Planes의 다중 스케일 개념을 활용하여 **제한된 관찰 데이터에서 수렴 가속화**를 달성합니다.

#### 3. 명시적 표현의 부상[8][9]

**RefinedFields** (2024): K-Planes를 사전 훈련된 모델로 정제하여 **약한 감독(weakly supervised)** 환경에서 성능 향상

**TK-Planes** (2024): K-Planes를 동적 UAV 장면으로 확장하여 고도가 높은 비디오 캡처에서 동적 객체 추적 개선

**X-NeRF** (2023): 다중 장면 360° 부족 뷰(insufficient view) 문제에 명시적 완성 매핑 적용

#### 4. 고려할 연구 방향

**크로스 스케일 기하학 적응**: K-Planes의 다중 스케일 특성을 활용하여 각 스케일에서 자동으로 주파수 조정[10]

**사전 훈련 및 미세조정**: 대규모 장면 데이터셋에서 하이퍼네트워크로 일반 맵 학습 후 빠른 미세조정[11]

**깊이 정규화**: 단안 깊이 추정 모델의 사전 지식을 활용한 규칙화로 희소 뷰 성능 개선[12][13]

**상호정보 이론**: K-Planes의 다중 평면 구조를 통해 여러 뷰 간 상호정보 최대화로 일반화 강화[14]

### 결론

K-Planes는 **간단하고 효율적이면서도 해석 가능한** 평면 인수분해 접근법으로 3D 정적 장면, 4D 동적 장면, 변하는 외형의 장면을 모두 다룰 수 있습니다. 특히 **Hadamard 곱을 통한 공간적 국소화**와 **다중 스케일 표현**은 이후 연구에서 소수 샷 학습과 일반화 개선의 기초가 되었습니다.

다만 K-Planes는 장면별 최적화를 요구하므로, **앞으로의 연구 방향**은 다음을 포함합니다:

1. **크로스 장면 일반화**: 사전 훈련된 인코더나 하이퍼네트워크를 통한 장면 간 지식 전이
2. **약한 감독 학습**: 사전 훈련된 확산 모델이나 깊이 추정기를 통한 감독 신호 강화
3. **효율적 미세조정**: 사전 훈련된 K-Planes 매개변수에서 빠른 적응
4. **다중 모달 입력**: RGB-D 입력이나 의미론적 정보 결합

이러한 방향들은 K-Planes의 명시적 표현과 해석 가능성의 이점을 유지하면서, 현실 세계의 제한된 데이터 환경에서 더 나은 성능을 달성하는 것을 목표로 합니다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ecb80336-6035-49c2-aebb-c2154d89eb7a/2301.10241v2.pdf)
[2](https://arxiv.org/html/2501.12637v2)
[3](https://arxiv.org/abs/2301.10941)
[4](https://arxiv.org/html/2402.14586v1)
[5](http://arxiv.org/pdf/2404.00992.pdf)
[6](http://arxiv.org/pdf/2406.07828.pdf)
[7](https://arxiv.org/html/2408.04803v1)
[8](https://arxiv.org/html/2312.00639v3)
[9](https://openaccess.thecvf.com/content/WACV2023/papers/Zhu_X-NeRF_Explicit_Neural_Radiance_Field_for_Multi-Scene_360deg_Insufficient_RGB-D_WACV_2023_paper.pdf)
[10](https://linjohnss.github.io/frugalnerf/)
[11](https://arxiv.org/html/2310.17075)
[12](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Li_DNGaussian_Optimizing_Sparse-View_CVPR_2024_supplemental.pdf)
[13](https://arxiv.org/html/2403.06912v1)
[14](https://openreview.net/forum?id=5RPpwW82vs)
[15](https://arxiv.org/pdf/2301.10241.pdf)
[16](http://arxiv.org/pdf/2311.18159.pdf)
[17](http://arxiv.org/pdf/2405.02762.pdf)
[18](https://onlinelibrary.wiley.com/doi/pdfdirect/10.1029/2023MS003932)
[19](https://arxiv.org/html/2405.07857v3)
[20](https://arxiv.org/html/2407.13185)
[21](http://arxiv.org/pdf/2112.01523.pdf)
[22](https://openaccess.thecvf.com/content/CVPR2023/papers/Fridovich-Keil_K-Planes_Explicit_Radiance_Fields_in_Space_Time_and_Appearance_CVPR_2023_paper.pdf)
[23](https://openaccess.thecvf.com/content/CVPR2024/papers/Chou_GSNeRF_Generalizable_Semantic_Neural_Radiance_Fields_with_Enhanced_3D_Scene_CVPR_2024_paper.pdf)
[24](https://openaccess.thecvf.com/content/WACV2021/papers/Bautista_On_the_Generalization_of_Learning-Based_3D_Reconstruction_WACV_2021_paper.pdf)
[25](https://arxiv.org/abs/2301.10241)
[26](https://www.sciencedirect.com/science/article/abs/pii/S095741742402935X)
[27](https://arxiv.org/html/2404.03421v1)
[28](https://kimjy99.github.io/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0/k-planes/)
[29](https://arxiv.org/html/2210.00379v6)
[30](https://oasisyang.github.io/neural-prior/)
[31](https://jseobyun.tistory.com/361)
[32](http://arxiv.org/pdf/2212.02280.pdf)
[33](https://eccv.ecva.net/virtual/2024/poster/2027)
[34](https://proceedings.iclr.cc/paper_files/paper/2024/file/8882d370cdafec9885b918a8cfac642e-Paper-Conference.pdf)
[35](https://arxiv.org/html/2501.12637v1)
[36](https://proceedings.mlr.press/v202/fu23g/fu23g.pdf)
[37](https://openaccess.thecvf.com/content/CVPR2024/papers/Shi_ZeroRF_Fast_Sparse_View_360deg_Reconstruction_with_Zero_Pretraining_CVPR_2024_paper.pdf)
