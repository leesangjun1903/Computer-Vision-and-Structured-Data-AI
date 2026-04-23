# CityGaussian: Real-time High-quality Large-Scale Scene Rendering with Gaussians

---

## 1. 핵심 주장과 주요 기여 요약

### 핵심 주장

CityGaussian(CityGS)은 3D Gaussian Splatting(3DGS)을 대규모 도시 장면($1.5\,km^2$ 이상)에 효율적으로 적용하기 위해, **분할 정복(Divide-and-Conquer) 훈련 전략**과 **블록 단위 Level-of-Detail(LoD) 렌더링 전략**을 결합하여, 기존 방법 대비 최고 수준의 렌더링 품질과 실시간 속도를 동시에 달성한다는 것이다.

### 세 가지 핵심 기여

| 기여 | 내용 |
|------|------|
| ① 분할 정복 훈련 | 대규모 3DGS를 병렬로 재구성하는 효율적인 블록 기반 훈련 전략 |
| ② 블록 단위 LoD 렌더링 | 다양한 스케일에서 실시간 렌더링을 달성하는 LoD 전략 |
| ③ SOTA 달성 | 공개 벤치마크에서 최고 성능 달성 |

---

## 2. 해결 문제, 제안 방법, 모델 구조, 성능 및 한계

### 2.1 해결하고자 하는 문제

3DGS를 대규모 장면에 직접 적용할 때 두 가지 핵심 문제가 발생한다.

**문제 1: GPU 메모리 한계 (훈련 시)**
- 24GB RTX 3090: Gaussian 수가 1,100만 개를 초과하면 Out-of-Memory(OOM) 발생
- 2.7 $km^2$ MatrixCity 장면 재구성에는 2,000만 개 이상의 Gaussian이 필요
- 40GB A100에서도 직접 훈련 불가

**문제 2: 실시간 렌더링 병목 (렌더링 시)**
- 렌더링 속도의 병목은 **깊이 정렬(depth sorting)**
- 1.1M Gaussian의 Train 장면: 103 FPS
- 23M Gaussian의 MatrixCity 장면: 21 FPS (평균 가시 Gaussian 수는 유사함에도)
- 불필요한 Gaussian을 래스터라이저에서 제거하는 것이 핵심 과제

---

### 2.2 제안하는 방법 (수식 포함)

#### 2.2.1 3DGS 기초 (Preliminary)

3DGS는 장면을 $\mathbf{G_K} = \{G_k \mid k = 1, \ldots, K\}$로 표현하며, 각 Gaussian은 다음 속성을 가진다:
- 3D 위치: $\boldsymbol{p_k} \in \mathbb{R}^{3\times1}$
- 불투명도: $\alpha_k \in [0, 1]$
- 기하(공분산): Scaling + Rotation
- Spherical Harmonics 특징: $\boldsymbol{f_k} \in \mathbb{R}^{3\times16}$, 색상 $\boldsymbol{c_k} \in \mathbb{R}^{3\times1}$

알파 블렌딩(Alpha Blending) 렌더링:

$$c_i(\boldsymbol{x}) = \sum_{k=1}^{K} \alpha_k \boldsymbol{c_k} G_k^{2D}(\boldsymbol{x}) \prod_{t=1}^{k-1}(1 - \alpha_t G_t^{2D}(\boldsymbol{x})), \quad G_k^{2D} = \mathbf{proj}(G_k, \kappa, \tau_i) \tag{1}$$

여기서 $\mathbf{proj}$는 투영 연산, $c_i(\boldsymbol{x})$는 픽셀 $\boldsymbol{x}$에서의 색상, $G_k^{2D}$는 2D로 투영된 Gaussian 분포이다.

---

#### 2.2.2 훈련 전략: CityGS 훈련 파이프라인

**Step 1: 전역(Coarse Global) Gaussian Prior 생성**

전체 COLMAP 포인트 클라우드를 모든 관측으로 30,000 iteration 훈련 → 전체 장면의 대략적인 기하 분포 $\mathbf{G_K} = \{G_k \mid k=1,\ldots,K\}$ 획득

이 Prior는 이후 블록별 fine-tuning 초기화에 사용되어 블록 간 간섭을 최소화하고 seamless fusion을 가능하게 한다.

**Step 2: Gaussian 및 데이터 분할 (Primitives & Data Division)**

현실 장면은 비경계(unbounded)이므로 Gaussian이 무한히 퍼질 수 있다. 이를 위해 **공간 수축(Space Contraction)**을 적용한다.

전경(foreground) 영역의 정규화:

$$\hat{\boldsymbol{p}}_k = \frac{2(\boldsymbol{p}_k - \boldsymbol{p}_{\min})}{\boldsymbol{p}_{\max} - \boldsymbol{p}_{\min}} - 1$$

수축 함수 (ScaNeRF [40] 기반):

$$\mathbf{contract}(\hat{\boldsymbol{p}}_k) = \begin{cases} \hat{\boldsymbol{p}}_k, & \text{if } \|\hat{\boldsymbol{p}}_k\|_\infty \leq 1, \\ \left(2 - \frac{1}{\|\hat{\boldsymbol{p}}_k\|_\infty}\right)\frac{\hat{\boldsymbol{p}}_k}{\|\hat{\boldsymbol{p}}_k\|_\infty}, & \text{if } \|\hat{\boldsymbol{p}}_k\|_\infty > 1. \end{cases} \tag{2}$$

수축 후 균등 격자 분할 → 블록 간 작업량 균형 달성

**Step 3: 적응형 데이터 할당 (Adaptive Data Assignment)**

$j$번째 블록 $\mathbf{G_{K_j}}$에 $i$번째 포즈 $\tau_i$를 할당하는 기준:

**원칙 1 (기여도 기반):** 블록 $j$가 렌더링 결과에 상당한 기여를 하는 경우

$$\boldsymbol{B}_1(\tau_i, \mathbf{G_{K_j}}) = \begin{cases} 1, & L_{\text{SSIM}}\left(I_{\mathbf{G_K}}(\tau_i),\, I_{\mathbf{G_K} \setminus \mathbf{G_{K_j}}}(\tau_i)\right) > \varepsilon, \\ 0, & \text{otherwise}. \end{cases} \tag{3}$$

SSIM Loss가 임계값 $\varepsilon$보다 크면 블록 $j$가 렌더링에 상당한 기여를 하므로 해당 포즈를 할당한다.

**원칙 2 (위치 기반):** 카메라 위치가 블록 내부에 있는 경우

$$\boldsymbol{B}_2(\tau_i, \mathbf{G_{K_j}}) = \begin{cases} 1, & \boldsymbol{b_{j,\min}} \leq \mathbf{contract}(\hat{\boldsymbol{p}}_{\tau_i}) < \boldsymbol{b_{j,\max}}, \\ 0, & \text{otherwise}. \end{cases} \tag{4}$$

**최종 할당 결정:**

$$\boldsymbol{B}(\tau_i, \mathbf{G_{K_j}}) = \boldsymbol{B}_1(\tau_i, \mathbf{G_{K_j}}) + \boldsymbol{B}_2(\tau_i, \mathbf{G_{K_j}}) \tag{5}$$

훈련 손실은 원본 3DGS를 따라 $L_1$과 SSIM loss의 가중합으로 구성된다.

---

#### 2.2.3 렌더링 전략: 블록 단위 Level-of-Detail (LoD)

**Detail Level 생성:**
- LightGaussian [9] 압축 전략을 활용하여 훈련된 Gaussian에서 직접 여러 detail level 생성
- MatrixCity 기준: LoD 2 (50%), LoD 1 (34%), LoD 0 (25%) 압축률

**블록 경계 추정 (Floater 제거):**

Median Absolute Deviation (MAD) 알고리즘으로 $j$번째 블록의 경계 추정:

$$MAD_j = \text{median}\left(\left|\boldsymbol{p}_k^j - \text{median}\left(\boldsymbol{p}_k^j\right)\right|\right)$$

$$\boldsymbol{p}_{\min}^j = \max\left(\min\left(\boldsymbol{p}_k^j\right),\, \text{median}\left(\boldsymbol{p}_k^j\right) - n_{MAD} \times MAD_j\right)$$

$$\boldsymbol{p}_{\max}^j = \min\left(\max\left(\boldsymbol{p}_k^j\right),\, \text{median}\left(\boldsymbol{p}_k^j\right) + n_{MAD} \times MAD_j\right) \tag{6}$$

**블록 단위 LoD 선택 및 Fusion:**
- 블록의 8개 코너에서 카메라까지 최소 거리로 detail level 결정
- 스크린 공간 투영 후 IoU로 frustum intersection 체크
- 서로 다른 detail level의 Gaussian을 직접 연결(concatenation)하여 최소한의 불연속성 유발

---

### 2.3 모델 구조 (전체 파이프라인)

```
[훈련 파이프라인]
COLMAP 포인트 클라우드
        ↓
전체 데이터로 30,000 iter 훈련 → Coarse Global Gaussian Prior (G_K)
        ↓
공간 수축(Space Contraction) + 균등 격자 분할
        ↓
블록별 Gaussian 분할 (G_K_j) + 적응형 데이터 할당 (B_1, B_2)
        ↓
병렬 Fine-tuning (각 블록, 30,000 iter, Global Prior로 초기화)
        ↓
블록 내 Gaussian 필터링 + 직접 Concatenation → 전체 장면 Gaussian

[렌더링 파이프라인]
전체 장면 Gaussian
        ↓
LightGaussian으로 LoD 2/1/0 생성 (압축률: 50%/34%/25%)
        ↓
렌더링 시: 블록별 frustum intersection 체크 (MAD 기반 경계 추정)
        ↓
카메라 거리에 따른 블록별 detail level 선택
        ↓
선택된 Gaussian Concatenation → 래스터라이저 → 출력 이미지
```

---

### 2.4 성능 향상

#### 정량적 결과 (Table 1, Table S1)

| 방법 | MatrixCity SSIM↑ | PSNR↑ | LPIPS↓ | FPS↑ |
|------|-----------------|-------|--------|------|
| MegaNeRF | - | - | - | <0.1 |
| Switch-NeRF | - | - | - | <0.1 |
| GP-NeRF | 0.611 | 23.56 | 0.630 | 0.15 |
| 3DGS† | 0.735 | 23.67 | 0.384 | 35.9 |
| **CityGS (no LoD)** | **0.865** | **27.46** | **0.204** | 21.6 |
| **CityGS (with LoD)** | **0.855** | **27.32** | **0.229** | **53.7** |

#### LoD 효과 (Table 2)

| 모델 | SSIM↑ | PSNR↑ | LPIPS↓ | FPS↑ |
|------|-------|-------|--------|------|
| no-LoD | 0.865 | 27.46 | 0.204 | 21.6 |
| Only LoD 2 | 0.863 | 27.54 | 0.215 | 45.6 |
| Only LoD 1 | 0.848 | 27.20 | 0.244 | 57.2 |
| Only LoD 0 | 0.825 | 26.57 | 0.279 | 69.4 |
| **LoD (통합)** | **0.855** | **27.32** | **0.229** | **53.7** |

LoD 적용 시: SSIM은 LoD 2 수준을 유지하면서 FPS는 21.6 → 53.7로 약 **2.5배** 향상

---

### 2.5 한계

논문에서 명시적으로 언급한 한계:

1. **정적 장면 가정(Static Scene Assumption):** 동적 객체(차량, 보행자 등)가 포함된 장면에 일반화하기 어렵다.
2. **이질적 뷰 혼합 문제:** 항공 뷰(aerial view)와 지상 뷰(street view)를 동시에 훈련할 때 성능이 향상되지 않고 오히려 저하되는 현상이 관찰된다. 내부 메커니즘은 추가 연구가 필요하다.
3. **외관 변화(Appearance Variation):** Sci-Art, Residence 데이터셋에서 PSNR이 상대적으로 낮은데, 이는 뷰에 따른 외관 변화 때문으로, 향후 과제로 남겨둔다.
4. **연산 비용:** Global Prior 생성 단계가 추가됨으로써 전체 훈련 시간이 늘어난다.
5. **블록 경계 아티팩트 가능성:** Eq.(4)가 없으면 블록 경계에서 floater가 발생할 수 있으며, 완전히 제거되지 않을 수 있다.

---

## 3. 모델의 일반화 성능 향상 가능성

논문에서 일반화와 관련하여 다음 내용이 확인된다.

### 3.1 논문 내 일반화 실험

논문은 CityGS의 일반화 능력을 검증하기 위해 **거리 뷰(Street View) 장면인 MatrixCity Block_A**에서 추가 실험을 수행하였다 (Table 3).

| 방법 | SSIM↑ | PSNR↑ | LPIPS↓ |
|------|-------|-------|--------|
| MipNeRF360 | 0.717 | 22.00 | 0.488 |
| 3DGS† | 0.701 | 21.14 | 0.441 |
| **CityGS** | **0.808** | **22.98** | **0.301** |

항공 뷰 중심으로 설계된 CityGS가 거리 뷰에서도 우수한 성능을 보임으로써 어느 정도의 일반화 능력을 확인하였다.

### 3.2 일반화를 제한하는 요인

1. **정적 장면 가정:** 동적 객체를 포함하는 실제 도시 장면에 직접 적용 불가
2. **이질적 카메라 뷰 문제:** 항공 뷰(150~500m 고도)와 거리 뷰를 동시에 학습할 때 성능 저하 발생 — 논문은 이를 명시적 한계로 언급
3. **하이퍼파라미터 의존성:** SSIM 임계값 $\varepsilon$, 블록 수, MAD 하이퍼파라미터 $n_{MAD}$, 거리 구간 등이 데이터셋에 따라 수동 조정 필요 (Table S3 참조)
4. **외관 변화 미처리:** 조명 변화, 날씨 변화 등에 대한 별도 메커니즘 부재

### 3.3 일반화 향상 가능성 (논문이 시사하는 방향)

**① 다양한 장면 스케일 지원**
- 현재 MatrixCity(합성, 항공), Rubble/Building/Residence/Sci-Art(실사, 항공) 등 다양한 실환경에서 검증됨
- 블록 단위 분할 + LoD 구조 자체가 스케일 독립적으로 설계되어 있어 다양한 도시 규모에 적용 가능

**② 장면 편집 가능성 (Appendix E 참조)**
- 명시적 Gaussian 표현이기 때문에 외관 재편집(repainting), 객체 교체(replacement), 교통 시뮬레이션 등 가능
- NeRF 기반 방법에서는 불가능한 장면 조작이 가능하여 시뮬레이션 도메인 일반화 가능성 존재

**③ 구조적 일반화 가능성**
- 분할 정복 + Global Prior 구조는 임의의 대규모 장면에 원칙적으로 적용 가능
- Adaptive data assignment(Eq. 3, 4, 5)는 데이터 분포의 불균형에 대응하는 메커니즘으로 일반화 기여

**④ 개선 방향 (논문이 암시)**
- 동적 장면 처리를 위한 시간축 모델링 통합 필요
- 외관 변화 처리를 위한 조명/날씨 모델링 추가 필요 (VastGaussian [17]이 이를 일부 다룸)
- 이질적 뷰(aerial + street)의 통합 학습 메커니즘 연구 필요

---

## 4. 최신 관련 연구 비교 분석 (2020년 이후)

### 4.1 NeRF 기반 대규모 장면 재구성

| 방법 | 연도 | 핵심 기법 | 실시간 여부 | 주요 한계 |
|------|------|-----------|------------|----------|
| NeRF [Mildenhall et al.] | 2020 | MLP + 볼류메트릭 렌더링 | ✗ | 느린 속도, 소규모 장면 |
| Mip-NeRF [Barron et al.] | 2021 | 멀티스케일 표현 | ✗ | 대규모 장면 미지원 |
| Block-NeRF [Tancik et al.] | 2022 | 블록 분할 MLP | ✗ | 낮은 품질, 느린 속도 |
| Mega-NeRF [Turki et al.] | 2022 | 분할 정복 MLP | ✗ | 세부 디테일 부족 |
| Switch-NeRF [Zhenxing et al.] | 2022 | 학습 가능한 장면 분해 | ✗ | 렌더링 속도 낮음 |
| BungeeNeRF [Xiangli et al.] | 2022 | Progressive LoD, 잔차 블록 | ✗ | 속도 한계 |
| GP-NeRF [Zhang et al.] | 2023 | 해시 그리드 + 멀티해상도 tri-plane | ✗ (0.15~0.42 FPS) | 실시간 불가 |
| ScaNeRF [Wu et al.] | 2023 | 번들 조정 NeRF, 부정확한 포즈 대응 | ✗ | 속도 한계 |

### 4.2 3DGS 기반 방법

| 방법 | 연도 | 핵심 기법 | 실시간 여부 | 주요 한계 |
|------|------|-----------|------------|----------|
| 3DGS [Kerbl et al.] | 2023 | 기본 3DGS | ✓ (소규모) | 대규모 장면 OOM |
| VastGaussian [Lin et al.] | 2024 | 3DGS + 외관 변화 처리 | △ (낮은 속도) | 대규모 실시간 미달 |
| LightGaussian [Fan et al.] | 2023 | Gaussian 압축 (15x) | ✓ | 소규모 장면 중심 |
| **CityGS (본 논문)** | **2024** | **분할정복 + LoD + Global Prior** | **✓ (대규모)** | **정적 장면, 이질 뷰 한계** |

### 4.3 비교 분석 요약

**CityGS의 차별점:**

$$\underbrace{\text{CityGS}}_{\text{본 논문}} = \underbrace{\text{분할정복 훈련}}_{\text{Mega-NeRF 영감}} + \underbrace{\text{Global Prior}}_{\text{신규}} + \underbrace{\text{블록 단위 LoD}}_{\text{신규}} + \underbrace{\text{3DGS 기반 실시간}}_{\text{3DGS 강점}}$$

- MegaNeRF, Switch-NeRF 대비: **렌더링 품질 및 속도** 대폭 향상 (FPS: <0.1 → 53.7)
- VastGaussian 대비: **실시간 렌더링** 달성 (VastGaussian은 대규모 장면에서 실시간 미달)
- 원본 3DGS† 대비: **훈련 효율 + 품질** 향상 (MatrixCity SSIM: 0.735 → 0.865)

---

## 5. 앞으로의 연구에 미치는 영향과 고려할 점

### 5.1 앞으로의 연구에 미치는 영향

**① 대규모 장면 재구성 패러다임 전환**
- 3DGS를 대규모 장면에 확장하는 실용적 프레임워크를 제시하여, NeRF 기반 대규모 재구성 연구에서 3DGS 기반으로의 전환을 가속화할 것으로 예상된다.
- 분할 정복 + Global Prior 패턴은 이후 연구의 기본 설계 원칙으로 채택될 가능성이 높다.

**② 실시간 대규모 렌더링의 실용화**
- AR/VR, 자율주행 시뮬레이션, 디지털 트윈, 스마트 시티 등 산업 분야에서 실시간 대규모 장면 렌더링의 실용화 기반을 제공한다.
- LoD 전략의 블록 단위 구현은 게임 엔진의 LoD 기법과 결합 가능한 연구 방향을 제시한다.

**③ 명시적 표현의 편집 가능성 확대**
- NeRF의 암묵적 표현과 달리 Gaussian의 명시적 표현이 장면 편집, 시뮬레이션, 콘텐츠 생성 분야로의 확장 가능성을 보여준다.

**④ Gaussian 압축 연구 촉진**
- LoD 생성에 LightGaussian을 활용한 사례는 Gaussian 압축 연구와 장면 재구성 연구의 결합을 촉진할 것이다.

---

### 5.2 앞으로 연구 시 고려할 점

**① 동적 장면 처리**
- CityGS는 정적 장면 가정을 기반으로 하므로, 동적 객체(차량, 보행자)를 포함한 실제 도시 장면에 대응하기 위해 시간축 모델링(temporal modeling)과의 통합 연구가 필요하다.
- SUDS [37], D-NeRF [26] 등의 동적 장면 처리 기법과의 결합을 고려해야 한다.

**② 이질적 뷰 통합 학습**
- 항공 뷰와 지상 뷰를 동시에 학습할 때 발생하는 성능 저하 문제는 해결되지 않은 핵심 과제이다.
- 뷰 유형별 적응적 스케일 분할 또는 뷰 분리 학습 전략 연구가 필요하다.

**③ 외관 변화(Appearance Variation) 처리**
- 조명 변화, 날씨 변화, 시간대 변화 등에 대한 robust한 모델링이 필요하다.
- VastGaussian [17]이 외관 변화를 일부 다루고 있으나, CityGS 수준의 품질과 속도를 유지하면서 외관 변화를 처리하는 방법 연구가 필요하다.

**④ 자동화된 하이퍼파라미터 설정**
- 현재 블록 수, SSIM 임계값 $\varepsilon$, 거리 구간 등을 데이터셋마다 수동 조정해야 한다 (Table S3).
- 학습 기반 또는 적응적 하이퍼파라미터 최적화 연구가 필요하다.

**⑤ 메모리 효율적 Gaussian 표현**
- Compact 3DGS [Lee et al., 2023], LightGaussian [Fan et al., 2023] 등의 압축 기법과 더 깊이 통합하여 스토리지 및 메모리 효율을 높이는 연구가 필요하다.

**⑥ 점진적(Progressive) 재구성 및 온라인 업데이트**
- 새로운 데이터가 추가될 때 전체 재훈련 없이 점진적으로 모델을 업데이트하는 메커니즘 연구가 실용적으로 중요하다.

**⑦ 멀티모달 데이터 활용**
- LiDAR, IMU 등 추가 센서 데이터를 활용하여 기하 정확도와 일반화 성능을 높이는 방향도 중요한 연구 주제이다 (SUDS [37], Urban Radiance Fields [28] 참조).

---

## 참고 자료

1. **Liu, Y. et al.** "CityGaussian: Real-time High-quality Large-Scale Scene Rendering with Gaussians." arXiv:2404.01133v3 (2024). **(본 논문)**
2. **Kerbl, B. et al.** "3D Gaussian Splatting for Real-Time Radiance Field Rendering." ACM Transactions on Graphics 42(4), 2023.
3. **Turki, H. et al.** "Mega-NeRF: Scalable Construction of Large-Scale NeRFs for Virtual Fly-Throughs." CVPR 2022.
4. **Tancik, M. et al.** "Block-NeRF: Scalable Large Scene Neural View Synthesis." CVPR 2022.
5. **Lin, J. et al.** "VastGaussian: Vast 3D Gaussians for Large Scene Reconstruction." CVPR 2024.
6. **Fan, Z. et al.** "LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction and 200+ FPS." arXiv:2311.17245, 2023.
7. **Zhenxing, M., Xu, D.** "Switch-NeRF: Learning Scene Decomposition with Mixture of Experts for Large-Scale Neural Radiance Fields." ICLR 2022.
8. **Zhang, Y. et al.** "GP-NeRF: Efficient Large-Scale Scene Representation with a Hybrid of High-Resolution Grid and Plane Features." arXiv:2303.03003, 2023.
9. **Xiangli, Y. et al.** "BungeeNeRF: Progressive Neural Radiance Field for Extreme Multi-Scale Scene Rendering." ECCV 2022.
10. **Wu, X. et al.** "ScaNeRF: Scalable Bundle-Adjusting Neural Radiance Fields for Large-Scale Scene Rendering." ACM TOG 42(6), 2023.
11. **Li, Y. et al.** "MatrixCity: A Large-Scale City Dataset for City-Scale Neural Rendering and Beyond." ICCV 2023.
12. **Mildenhall, B. et al.** "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis." Communications of the ACM 65(1), 2021.
13. **Barron, J.T. et al.** "Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields." CVPR 2022.
14. **Song, K., Zhang, J.** "City-on-Web: Real-Time Neural Rendering of Large-Scale Scenes on the Web." arXiv:2312.16457, 2023.
