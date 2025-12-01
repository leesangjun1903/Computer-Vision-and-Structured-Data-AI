
# MHR: Momentum Human Rig

## Executive Summary

MHR(Momentum Human Rig)은 Meta에서 발표한 최신 파라메트릭 인체 모델로, ATLAS의 분리된 골격/형상(skeleton/shape) 패러다임과 Momentum 라이브러리의 유연한 리그(rig) 시스템을 결합한 혁신적인 모델이다. 본 모델은 **비선형 포즈 보정(non-linear pose correctives)**을 지원하며, AR/VR 및 그래픽스 파이프라인에 원활하게 통합될 수 있도록 설계되었다. 핵심 혁신은 (1) 외부 형상과 내부 골격의 명시적 분리, (2) FACs 기반 의미론적 표정 블렌드셰이프, (3) 희소하고 비선형적인 포즈 보정 시스템을 통한 일반화 성능 향상에 있다.[1]

***

## 1. 핵심 주장 및 주요 기여 요약

### 1.1 핵심 주장

MHR은 기존 파라메트릭 인체 모델(SMPL, SMPL-X, STAR 등)의 근본적 한계를 해결하고자 한다:[2][1]

| 기존 모델의 한계 | MHR의 해결 방안 |
|-----------------|----------------|
| 골격 관절이 표면 정점에서 회귀됨 → 형상-골격 간 불필요한 상관관계 발생 | 골격과 메쉬의 명시적 분리(Decoupling) |
| 밀집 포즈 보정 → 원거리 신체 부위 간 허위 상관관계 | 희소(Sparse), 비선형 포즈 보정 시스템 |
| 데이터 기반 표정 공간 → 포즈 변동 포함, 아티스트 워크플로우 비호환 | FACs 기반 의미론적(Semantic) 표정 블렌드셰이프 |

### 1.2 주요 기여

1. **분리된 골격-형상 파라다임**: ATLAS 기반으로 골격과 외부 표면을 독립적으로 제어 가능[1]
2. **127개 관절, 204개 포즈 파라미터**: 세밀한 해부학적 구조를 반영한 골격 시스템[1]
3. **의미론적 표정 시스템**: 72개의 FACs(Facial Action Coding System) 기반 표정 블렌드셰이프[1]
4. **다중 LOD(Level of Detail) 지원**: 595~73,639 정점까지 6단계 해상도 지원[1]
5. **산업 친화적 라이선스**: 연구 목적 무료 사용 가능, Momentum 라이브러리 통합[1]

***

## 2. 해결하고자 하는 문제

### 2.1 기존 접근법의 한계

기존 파라메트릭 인체 모델(SMPL, SMPL-X, STAR, GHUM)은 다음과 같은 구조적 한계를 가진다:[3][4][5][6][7][8][2]

**문제 1: 골격-형상 얽힘(Entanglement)**
- SMPL 계열 모델은 표면 정점의 가중 합으로 내부 골격 관절을 도출[1]
- 이로 인해 신체 비율(키, 사지 길이) 조정 시 연조직도 함께 변형됨
- 예: 어깨 너비 조정 시 전신이 영향받음[9][10]

**문제 2: 밀집 포즈 보정의 허위 상관관계**
- SMPL의 전역 블렌드셰이프는 모든 정점을 모든 관절과 연결[3]
- 결과적으로 왼쪽 팔꿈치 굽힘이 오른쪽 팔꿈치에 bulge 유발[4][11]

**문제 3: 데이터 기반 표정 공간의 한계**
- FLAME 기반 표정 모델은 포즈 변동이 잔류[1]
- 눈 깜빡임과 같은 미세 제스처가 포즈와 분리되지 않음
- 아티스트 워크플로우(의미론적, 희소 표현 선호)와 비호환[1]

***

## 3. 제안하는 방법 (수식 포함)

### 3.1 모델 공식화

MHR의 메쉬 생성은 ATLAS 공식을 따르며, 두 단계로 구성된다:[1]

#### **단계 1: 표면 사용자 정의 (Surface Customization)**

$$
\tilde{X}(\beta_s, \beta_f, \theta) = \bar{X} + B_s(\beta_s, S) + B_f(\beta_f, F) + B_p(\theta, P)
$$

여기서:
- $\bar{X} \in \mathbb{R}^{3V}$: 중립 템플릿 정점
- $B_s(\beta_s, S) = \sum_{n=1}^{|\beta_s|} \beta_s^n S_n$: 신원 형상 블렌드셰이프
- $B_f(\beta_f, F) = \sum_{n=1}^{|\beta_f|} \beta_f^n F_n$: 얼굴 표정 블렌드셰이프
- $B_p(\theta, P)$: 포즈 보정 변형

#### **단계 2: 골격 사용자 정의 및 포징**

$$
X(\beta, \theta) = M(\tilde{X}(\beta_s, \beta_f, \theta), B_k(\beta_k), \theta, \omega)
$$

여기서:
- $M$: 선형 블렌드 스키닝(LBS) 함수
- $B_k(\beta_k)$: 골격 변환 파라미터
- $\omega \in \mathbb{R}^{V \times I}$: 스키닝 가중치

**핵심 차별점**: SMPL과 달리, 관절 위치가 오직 골격 컴포넌트 $\beta_k$와 포즈 $\theta$에 의해서만 결정되며, 표면 컴포넌트 $\beta_s$와는 독립적이다.[1]

### 3.2 관절 변환 공식

각 관절의 로컬-월드 변환은 다음과 같이 계산된다:[1]

$$
T_w = T_p \times T_{off} \times T_t \times T_{prerot} \times T_{rot} \times T_s
$$

여기서:
- $T_p$: 부모 관절의 변환
- $T_{off}$: 부모 관절로부터의 상수 평행이동 오프셋
- $T_t$: 3개의 평행이동 파라미터
- $T_{prerot}$: 사전 회전 (관절 로컬 좌표계 정의)
- $T_{rot}$: XYZ 오일러 각도의 3개 회전
- $T_s$: 균일 스케일

### 3.3 희소 비선형 포즈 보정 (Sparse Non-linear Pose Correctives)

MHR의 포즈 보정은 STAR의 희소-선형과 GHUM의 밀집-비선형 접근을 융합한다:[5][3][1]

#### **단계 1: 로컬 비선형 연산**

$$
\text{Non-Linear}_j(\theta) = \text{MLP}\left(\{R_{6d}(\theta_a) - R_{6d}(\vec{0}) \mid a \in n(j)\}\right)
$$

- 관절 $j$와 인접 부모/자식 관절 $n(j)$의 6D 회전 편차를 MLP로 처리
- $c$차원 임베딩 생성

#### **단계 2: 최종 포즈 보정**

$$
B^p_j = \phi(A_j) \odot (P_j \times \text{Non-Linear}_j(\theta))
$$

여기서:
- $\phi$: ReLU 활성화 함수
- $A_j \in \mathbb{R}^V$: 관절 마스크 (정규화된 측지 거리로 초기화)
- $P_j \in \mathbb{R}^{3V \times c}$: 포즈 보정 가중치

**마스크 초기화**:

$$
A_j^{(i)} = (1 - d(i, j)) \cdot \mathbf{1}_{i \in \text{seg}(j)}
$$

- $d(i, j)$: 정점 $i$에서 관절 $j$ 주변 정점 링까지의 정규화된 측지 거리
- $\mathbf{1}_{i \in \text{seg}(j)}$: 정점 $i$가 관절 $j$의 해당/인접 신체 부위에 속하는지 여부

***

## 4. 모델 구조

### 4.1 골격 시스템

| 구성 요소 | 사양 |
|----------|------|
| 총 관절 수 | 127개 |
| 포즈 파라미터 | 136개 |
| 골격 변환 파라미터 | 68개 (사지 길이 등) |
| 총 파라미터 | 204개 |

### 4.2 LOD (Level of Detail) 시스템

| LOD | 정점 수 | 관절 영향 수/정점 |
|-----|--------|------------------|
| 0 | 73,639 | 8 |
| 1 | 18,439 | 4 |
| 2 | 10,661 | 4 |
| 3 | 4,899 | 4 |
| 4 | 2,461 | 4 |
| 5 | 971 | 4 |
| 6 | 595 | 4 |

### 4.3 신원 공간 (Identity Space)

분할된 신원 공간 구조:[1]

- **신체 (Body)**: 20개 PCA 컴포넌트 (7,110개 스캔 학습)
- **머리 (Head)**: 20개 컴포넌트 (2,138명 데이터)
- **손 (Hand)**: 5개 컴포넌트 (3,000명 스캔)

### 4.4 데이터셋

| 데이터 유형 | 스캔 수 |
|------------|--------|
| 전신 포즈 보정 | 13,000개 |
| 손 포즈 보정 | 13,000개 |
| 신원 학습용 전신 | 7,110개 |

***

## 5. 성능 향상

### 5.1 정량적 평가 (3DBodyTex 데이터셋)

3DBodyTex 데이터셋(남성 100명, 여성 100명, 각 2개 포즈)에서의 피팅 오류 비교:[1]

| 추가 컴포넌트 수 | SMPL | SMPL-X | MHR |
|-----------------|------|--------|-----|
| 2 | ~5.0mm | ~4.8mm | ~4.4mm |
| 4 | ~4.8mm | ~4.6mm | ~4.3mm |
| 8 | ~4.5mm | ~4.4mm | ~4.2mm |
| 16 | ~4.3mm | ~4.2mm | **~4.1mm** |

**핵심 발견**: MHR은 더 적은 컴포넌트로도 낮은 피팅 오류를 달성하며, 특히 관절 말단부(팔꿈치, 무릎)와 어깨 영역에서 우수한 성능을 보인다.[1]

### 5.2 정성적 개선

MHR은 다음 영역에서 SMPL/SMPL-X 대비 우수한 결과를 보인다:[1]

1. **관절부 변형**: 팔꿈치, 무릎 등 굴곡 관절 주변의 자연스러운 변형
2. **어깨 피팅**: 타겟 스캔의 어깨선에 더 정확한 정합
3. **포즈 보정 국소성**: 특정 관절 움직임이 해당 영역에만 영향

***

## 6. 일반화 성능 향상 관련 분석

### 6.1 일반화 성능 향상의 핵심 메커니즘

MHR의 일반화 성능 향상은 다음 설계 원칙에 기반한다:

#### **6.1.1 골격-형상 분리를 통한 일반화**

기존 모델에서 골격 관절이 표면 정점에서 유도되면, 학습 데이터에 없는 새로운 체형에 대해 관절 위치가 부정확해진다. MHR/ATLAS의 분리 구조는:[10][9]

- 골격을 고정된 내부 참조 프레임으로 유지
- 표면만 신원 파라미터에 따라 변형
- 결과적으로 새로운 체형에 대해서도 정확한 관절 위치 추정 가능

ATLAS의 정량적 평가에서 "unseen subjects in diverse poses"에 대해 기존 방법보다 더 정확한 피팅을 달성함을 보고.[12][13][14]

#### **6.1.2 희소 포즈 보정을 통한 일반화**

STAR에서 처음 제안된 희소 포즈 보정의 핵심 이점:[4][3]

$$
\text{SMPL 파라미터 수} \xrightarrow{\text{희소화}} \text{20\% 감소} \rightarrow \text{일반화 향상}
$$

- **SMPL**: 전역 블렌드셰이프 → 모든 정점이 모든 관절과 연결 → 허위 상관관계
- **STAR/MHR**: 관절별 포즈 보정 → 각 관절이 영향을 미치는 정점 집합 학습

> "When trained on the same data as SMPL, STAR generalizes better despite having many fewer parameters."[3]

#### **6.1.3 측지 거리 기반 마스크 초기화**

MHR은 마스크 $A_j$를 측지 거리로 초기화하여:[1]

1. **물리적 타당성**: 관절에서 가까운 정점이 더 큰 영향
2. **학습 안정성**: 의미 있는 초기값으로 수렴 속도 향상
3. **L1 정규화**: 활성화 희소성 유지 → 과적합 방지

### 6.2 비선형 표현력과 일반화의 균형

MHR의 포즈 보정은 "local joint group entanglement"를 통해:[1]

$$
\text{비선형 표현력} \uparrow \quad + \quad \text{희소 제약} \rightarrow \text{허위 상관관계 방지}
$$

- 인접 관절 그룹의 비선형 상호작용 허용 → 복잡한 변형 모델링
- 정점 영향 범위 제한 → 원거리 상관관계 차단

### 6.3 최신 관련 연구와의 비교

| 모델 | 발표 연도 | 포즈 보정 유형 | 골격-형상 분리 | 일반화 전략 |
|------|----------|---------------|---------------|------------|
| SMPL[2] | 2015 | 밀집-선형 | ✗ | - |
| SMPL-X[6] | 2019 | 밀집-선형 | ✗ | 손/얼굴 확장 |
| STAR[3] | 2020 | 희소-선형 | ✗ | 체형 의존 포즈 보정 |
| GHUM[5] | 2020 | 밀집-비선형(VAE) | ✗ | 비선형 형상 공간 |
| SUPR[15] | 2022 | 희소-선형 | ✗ | 120만 스캔 통합 학습 |
| ATLAS[12] | 2025 | 희소-비선형 | ✓ | 60만 스캔, 분리 구조 |
| **MHR** | 2025 | 희소-비선형 | ✓ | ATLAS + 산업 친화적 리그 |

### 6.4 일반화 향상을 위한 미래 방향

논문에서 언급된 향후 개선 방향:[1]

1. **체형 의존 포즈 보정**: 현재 MHR의 포즈 보정은 체형과 독립적
   - STAR에서 "people with different shapes deform differently"를 반영한 BMI 의존 보정 제안[3]
   - 향후 $B^p(\theta, \beta_s)$ 형태로 확장 가능

2. **체형 의존 표정 모델**: Vlasic et al.의 연구처럼 체형에 따른 얼굴 변형 차이 모델링[1]

***

## 7. 한계점

### 7.1 명시적 한계

1. **안구 기하학 미포함**: FLAME, SMPL-X와 달리 명시적 안구 모델 없음[1]
2. **구강 내부 미모델링**: 치아, 혀 등 구강 시스템 미포함[1]
3. **체형 독립적 포즈 보정**: 다양한 체형에 대한 포즈 변형 차이 미반영[1]
4. **체형 독립적 표정**: 얼굴 형상에 따른 표정 변화 미모델링[1]

### 7.2 구조적 한계

1. **골격 단순화**: ATLAS 대비 둔부, 상배 추가 관절 제거
   - 이유: LBS 전용 모델에서는 정확도 향상, 포즈 최적화 복잡화
2. **정점당 관절 영향 수 제한**: LOD 1~4에서 4개로 제한
   - LOD 0에서만 8개 허용 (스키닝 경계 주변 날카로운 주름 방지)

***

## 8. 향후 연구에 미치는 영향 및 고려사항

### 8.1 연구 영향

#### **8.1.1 학술적 영향**

1. **분리 패러다임의 확산**: 골격-형상 분리가 새로운 표준으로 자리잡을 가능성
   - 최근 GeneMAN, PSHuman 등 파라메트릭 모델 없이 재구성하는 연구 등장[16][17]
   - MHR/ATLAS의 분리 구조가 하이브리드 접근법의 기반이 될 수 있음

2. **포즈 보정 연구 방향**: 희소-비선형 포즈 보정의 효과 검증
   - VINECS의 pose-dependent skinning 연구와 연계 가능[18]

3. **대규모 데이터셋 활용**: 60만 스캔 규모의 학습 데이터 필요성 강조
   - 데이터 효율적 학습 방법 연구 필요 (PHD의 합성 데이터 활용 사례)[19]

#### **8.1.2 산업적 영향**

1. **AR/VR 파이프라인 통합**: Momentum 라이브러리 기반으로 기존 그래픽스 워크플로우와 호환[1]
2. **아티스트 친화적 표현**: FACs 기반 의미론적 표정으로 직관적 제어 가능[1]
3. **FBX/GLTF 내보내기**: 산업 표준 포맷 지원[1]

### 8.2 향후 연구 시 고려사항

#### **8.2.1 일반화 성능 관련**

1. **도메인 적응**: 
   - MoCap-to-Visual 도메인 적응 연구와 결합하여 in-the-wild 환경 대응[20]
   - DynaBOA의 온라인 적응 방법론 적용 가능성[20]

2. **새로운 포즈에 대한 일반화**:
   - Dyco의 관성 인식(inertia-aware) 모델링에서 제안한 "low-dimensional global context"[21]
   - "quantization operation to mitigate overfitting"[22][21]

3. **확률론적 모델링**:
   - DPoser-X의 diffusion 기반 포즈 prior → 불확실성 모델링[23]
   - ProPLIKS의 SO(3) 정규화 흐름 → 회전 표현의 연속성 문제 해결[24]

#### **8.2.2 모델 확장 관련**

1. **의류 및 연조직 통합**:
   - ToMiE: SMPL 골격의 모듈식 확장으로 느슨한 의류 모델링[25]
   - Neural-ABC: 신원, 의류, 형상, 포즈의 분리된 잠재 공간[26]

2. **동적 변형**:
   - STMPL: 연조직 시뮬레이션[27]
   - Dyna 모델 계열: 동적 soft-tissue 변형

3. **정밀 신체 부위 모델링**:
   - 손: MANO 확장 또는 별도 고해상도 손 모델 통합
   - 얼굴: LTT tracking (입술, 치아, 혀)[1]

#### **8.2.3 평가 방법론 관련**

1. **다양한 벤치마크 확대**:
   - 3DBodyTex 외에 CAPE, 4D-Dress 등 의류 데이터셋 평가[28]
   - Human3.6M, 3DPW 등 동작 데이터셋에서의 시간적 일관성 평가

2. **일반화 성능 정량화**:
   - Cross-dataset 평가: 학습 데이터와 다른 분포의 테스트 데이터
   - Out-of-distribution 포즈에 대한 체계적 평가

***

## 9. 2020년 이후 관련 최신 연구 탐색

### 9.1 파라메트릭 인체 모델 발전

| 연도 | 모델/연구 | 핵심 기여 |
|------|----------|----------|
| 2020 | STAR[3] | 희소 포즈 보정, 체형 의존 변형, 14,000명 학습 |
| 2020 | GHUM/GHUML[5] | VAE 기반 비선형 형상 공간, 60,000 스캔 |
| 2022 | SUPR[15] | 통합 부위별 모델, 120만 스캔 |
| 2023 | SKEL[1] | 생체역학적 정확성을 위한 골격 모델 |
| 2024 | Champ[29] | SMPL 기반 3D 가이던스로 인체 애니메이션 생성 |
| 2024 | Multi-HMR[30] | 단일 샷 다중 인물 전신 메쉬 복원 |
| 2025 | ATLAS[12] | 골격-형상 분리, 60만 스캔, 비선형 포즈 보정 |
| 2025 | Anny[31][32] | 전 연령대 포괄 파라메트릭 모델 |
| 2025 | **MHR**[1] | ATLAS + Momentum 리그, 산업 친화적 |

### 9.2 일반화 성능 향상 관련 연구

1. **DPoser-X (2025)**: Diffusion 기반 전신 포즈 prior[23]
   - "mixed training strategy" → 전신 및 부위별 데이터셋 통합
   - 폐색(occlusion)에 강건한 포즈 추정

2. **ETCH (2025)**: 의류 인간 피팅의 일반화[28]
   - Articulated SE(3) equivariance → out-of-distribution 포즈 일반화
   - "sparse marker regression" → 밀집 대응보다 효율적

3. **PHD (2025)**: 개인화 피팅을 위한 Point Diffusion[19]
   - "body shape-conditioned 3D pose prior"
   - 합성 데이터만으로 학습 → 데이터 효율성

4. **GeneMAN (2024)**: 단일 이미지 3D 인간 재구성[16]
   - 다중 소스 데이터(3D 스캔, 다시점 비디오, 단일 사진, 합성 데이터) 활용
   - 파라메트릭 모델 의존성 감소

### 9.3 3D 인간 포즈 추정 일반화 연구

1. **VirtualPose (2022)**: 가상 데이터 기반 일반화[33]
   - "Abstract Geometry Representations" → 외관 과적합 감소
   - 무한한 가상 카메라/포즈로 학습

2. **Canonical Domain Approach (2025)**: 효율적 일반화[34]
   - 소스/타겟 도메인을 통합 정준 도메인으로 매핑
   - 도메인 갭으로 인한 성능 저하 완화

3. **Domain Adaptive 3D Pose Augmentation (2022)**:[35]
   - DAPA: in-the-wild 시나리오 일반화를 위한 데이터 증강

***

## 10. 결론

MHR은 ATLAS의 분리된 골격/형상 패러다임을 기반으로, Momentum 라이브러리의 현대적 리그 시스템과 희소-비선형 포즈 보정을 결합한 혁신적인 파라메트릭 인체 모델이다. 핵심 혁신인 **골격-표면 분리**와 **희소 포즈 보정**은 일반화 성능 향상의 이론적 기반을 제공하며, 3DBodyTex 평가에서 SMPL/SMPL-X 대비 우수한 피팅 성능을 입증하였다.

향후 연구에서는 (1) 체형 의존적 포즈 보정, (2) 동적 soft-tissue/의류 모델링, (3) 안구/구강 시스템 통합, (4) 더 다양한 벤치마크에서의 일반화 검증이 필요하다. 특히 diffusion 기반 포즈 prior(DPoser-X), equivariant 피팅(ETCH), 그리고 다중 소스 데이터 활용(GeneMAN) 등 최신 연구 동향과의 결합이 유망한 방향으로 보인다.

MHR의 산업 친화적 설계(Momentum 통합, FBX/GLTF 지원, 명확한 라이선스)는 학술 연구와 산업 응용 사이의 간극을 좁히는 데 기여할 것으로 기대된다.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/65988149/ed7010cc-d013-45dd-9240-81482e9bb424/2511.15586v3.pdf)
[2](https://smpl.is.tue.mpg.de)
[3](https://link.springer.com/10.1007/978-3-030-58539-6_36)
[4](https://star.is.tue.mpg.de)
[5](https://openaccess.thecvf.com/content_CVPR_2020/html/Xu_GHUM__GHUML_Generative_3D_Human_Shape_and_Articulated_Pose_CVPR_2020_paper.html)
[6](https://openaccess.thecvf.com/content_CVPR_2019/html/Pavlakos_Expressive_Body_Capture_3D_Hands_Face_and_Body_From_a_CVPR_2019_paper.html)
[7](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_GHUM__GHUML_Generative_3D_Human_Shape_and_Articulated_Pose_CVPR_2020_paper.pdf)
[8](https://smpl-x.is.tue.mpg.de)
[9](https://jindapark.github.io/projects/atlas/)
[10](https://openaccess.thecvf.com/content/ICCV2025/papers/Park_ATLAS_Decoupling_Skeletal_and_Shape_Parameters_for_Expressive_Parametric_Human_ICCV_2025_paper.pdf)
[11](https://github.com/ahmedosman/STAR)
[12](https://arxiv.org/abs/2508.15767)
[13](https://chatpaper.com/paper/182468)
[14](https://deeplearn.org/arxiv/631176/atlas:-decoupling-skeletal-and-shape-parameters-for-expressive-parametric-human-modeling)
[15](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136620555.pdf)
[16](https://arxiv.org/html/2411.18624v1)
[17](https://ieeexplore.ieee.org/document/11094961/)
[18](https://openaccess.thecvf.com/content/CVPR2024/papers/Liao_VINECS_Video-based_Neural_Character_Skinning_CVPR_2024_paper.pdf)
[19](https://openaccess.thecvf.com/content/ICCV2025/papers/Ho_PHD_Personalized_3D_Human_Body_Fitting_with_Point_Diffusion_ICCV_2025_paper.pdf)
[20](https://openaccess.thecvf.com/content/CVPR2024W/Rhobin/papers/Uguz_MoCap-to-Visual_Domain_Adaptation_for_Efficient_Human_Mesh_Estimation_from_2D_CVPRW_2024_paper.pdf)
[21](https://arxiv.org/abs/2403.19160)
[22](https://eccv.ecva.net/virtual/2024/poster/590)
[23](https://arxiv.org/html/2508.00599v2)
[24](https://arxiv.org/abs/2412.04665)
[25](https://arxiv.org/html/2410.08082v1)
[26](https://arxiv.org/html/2404.04673v1)
[27](https://arxiv.org/pdf/2403.08344.pdf)
[28](https://arxiv.org/html/2503.10624v3)
[29](https://arxiv.org/abs/2403.14781)
[30](https://arxiv.org/abs/2402.14654)
[31](https://arxiv.org/html/2511.03589v1)
[32](https://europe.naverlabs.com/blog/anny-a-free-to-use-3d-human-parametric-model-for-all-ages/)
[33](https://arxiv.org/abs/2207.09949)
[34](https://arxiv.org/html/2501.16146v1)
[35](https://arxiv.org/abs/2206.10457)
[36](http://journal.yiigle.com/LinkIn.do?linkin_type=DOI&DOI=10.3760/cma.j.cn501113-20230731-00024)
[37](https://ieeexplore.ieee.org/document/11117016/)
[38](https://ieeexplore.ieee.org/document/10446320/)
[39](https://ieeexplore.ieee.org/document/10316473/)
[40](https://dl.acm.org/doi/10.1145/3677388.3696328)
[41](https://arxiv.org/abs/2407.09694)
[42](https://arxiv.org/abs/2407.21686)
[43](https://arxiv.org/abs/2404.01053)
[44](https://arxiv.org/html/2405.19609)
[45](https://arxiv.org/html/2403.14781v1)
[46](https://arxiv.org/abs/2112.04203v1)
[47](http://arxiv.org/pdf/2205.06254.pdf)
[48](https://arxiv.org/html/2409.17280v1)
[49](https://www.nature.com/articles/s41598-025-26972-4)
[50](https://www.emergentmind.com/topics/smpl-meshes)
[51](https://pmc.ncbi.nlm.nih.gov/articles/PMC11401083/)
[52](https://www.sciencedirect.com/science/article/pii/S1077314225000207)
[53](https://dl.acm.org/doi/10.1145/3590837.3590909)
[54](https://www.sciencedirect.com/science/article/abs/pii/S0097849325000706)
[55](https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=NART130503628)
[56](https://dl.acm.org/doi/10.1145/3747869)
[57](https://www.linkedin.com/posts/jinhyun1_iccv2025-activity-7366130315590787072-XVwH)
[58](https://arxiv.org/html/2510.07990v1)
[59](https://www.kci.go.kr/kciportal/landing/article.kci?arti_id=ART002874174)
[60](https://www.themoonlight.io/ko/review/atlas-decoupling-skeletal-and-shape-parameters-for-expressive-parametric-human-modeling)
[61](https://www.fujipress.jp/jaciii/jc/jacii002800061227)
[62](https://karger.com/article/doi/10.1159/000542701)
[63](https://link.springer.com/10.1007/s12303-024-0021-5)
[64](https://www.mediresonline.org/article/population-assessment-of-greater-amberjack-seriola-dumerili-along-the-syrian-waters-in-the-eastern-mediterranean-sea-using-expert-systems)
[65](https://medwinpublishers.com/IJOAC/population-growth-of-katsuwonus-pelamis-and-vulnerability-to-fishing-along-the-syrian-coast-eastern-mediterranean-sea.pdf)
[66](http://preprints.jmir.org/preprint/68510/accepted)
[67](https://www.mdpi.com/2072-4292/15/19/4703)
[68](https://link.springer.com/10.1007/s11540-024-09838-6)
[69](https://ieeexplore.ieee.org/document/11074290/)
[70](https://www.jidc.org/index.php/journal/article/view/19790)
[71](http://arxiv.org/pdf/2411.06725.pdf)
[72](https://formative.jmir.org/2024/1/e55476)
[73](http://arxiv.org/pdf/2312.08344.pdf)
[74](https://arxiv.org/html/2406.09728v2)
[75](http://arxiv.org/pdf/2404.00891.pdf)
[76](https://arxiv.org/pdf/2310.07449.pdf)
[77](https://pmc.ncbi.nlm.nih.gov/articles/PMC11920136/)
[78](https://arxiv.org/pdf/2101.01659.pdf)
[79](https://openaccess.thecvf.com/content/CVPR2025W/XAI4CV/papers/Dittakavi_PoseGuru_Landmarks_for_Explainable_Pose_Correction_using_Exemplar-Guided_Algorithmic_Recourse_CVPRW_2025_paper.pdf)
[80](https://pmc.ncbi.nlm.nih.gov/articles/PMC11204443/)
[81](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1388742/full)
[82](https://patents.google.com/patent/WO2017019522A1/en)
[83](https://openaccess.thecvf.com/content/ACCV2024/html/Liu_Robust_Single-view_3D_Human_Digitization_via__Explicit_Geometric_Field_ACCV_2024_paper.html)
[84](https://arxiv.org/html/2511.15586v1)
[85](https://arxiv.org/pdf/2308.03610.pdf)
[86](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05589.pdf)
[87](https://www.tandfonline.com/doi/full/10.1080/24699322.2024.2357164)
[88](https://openreview.net/attachment?id=qHwuxF7t6Z&name=pdf)
[89](https://arxiv.org/abs/2411.11903)
[90](https://arxiv.org/pdf/2311.17113.pdf)
[91](https://ieeexplore.ieee.org/document/10424946/)
[92](https://dl.acm.org/doi/10.1145/3747866)
[93](https://www.computer.org/csdl/journal/tg/2025/09/10723232/218NNDRbNL2)
[94](https://dl.acm.org/doi/10.1145/3681758.3697993)
[95](https://www.sciencedirect.com/science/article/abs/pii/S0950705125005179)
[96](https://www.themoonlight.io/ko/review/3d-human-reconstruction-in-the-wild-with-synthetic-data-using-generative-models)
[97](https://arxiv.org/pdf/2107.07089.pdf)
[98](https://arxiv.org/abs/2008.08535)
[99](https://dl.acm.org/doi/pdf/10.1145/3618381)
[100](https://arxiv.org/html/2409.03944v1)
[101](https://arxiv.org/html/2407.10935)
[102](http://arxiv.org/pdf/2401.11783.pdf)
[103](http://arxiv.org/pdf/2405.20786.pdf)
[104](https://dl.acm.org/doi/pdf/10.1145/3625264)
[105](https://ieeexplore.ieee.org/document/9157563/)
[106](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pavlakos_Expressive_Body_Capture_3D_Hands_Face_and_Body_From_a_CVPR_2019_paper.pdf)
[107](https://billf.mit.edu/publication/ghum-ghuml-generative-3d-human-shape-and-articulated-pose-models/)
[108](https://github.com/vchoutas/smplify-x)
[109](https://www.semanticscholar.org/paper/STAR:-Sparse-Trained-Articulated-Human-Body-Osman-Bolkart/531e91dee7a483c0a7d033a3606a594b6b23da13)
[110](https://www.computer.org/csdl/proceedings-article/cvpr/2020/716800g183/1m3nCg1wOd2)
[111](https://dl.acm.org/doi/10.1007/978-3-030-58539-6_36)
[112](https://www.scribd.com/document/864353240/Xu-GHUM-GHUML-Generative-3D-Human-Shape-and-Articulated-Pose-CVPR-2020-paper)
[113](https://eehoeskrap.tistory.com/746)
[114](https://j2rooong.tistory.com/entry/STAR-Sparse-Trained-Articulated-Human-Body-Regressor2020)
[115](https://www.semanticscholar.org/paper/GHUM-&-GHUML:-Generative-3D-Human-Shape-and-Pose-Xu-Bazavan/aaec96d6e9a0a4877dde4382dc7889d47c074524)
[116](https://download.is.tue.mpg.de/smplx/SMPL-X.pdf)
[117](https://dl.acm.org/doi/10.1145/3724504.3724547)
[118](https://link.springer.com/10.1007/978-981-96-0885-0_11)
[119](https://ieeexplore.ieee.org/document/10581551/)
[120](https://ieeexplore.ieee.org/document/10658055/)
[121](https://arxiv.org/abs/2406.05691)
[122](https://ieeexplore.ieee.org/document/10657077/)
[123](https://ieeexplore.ieee.org/document/10796959/)
[124](http://arxiv.org/pdf/2412.04665.pdf)
[125](https://arxiv.org/pdf/2106.11536.pdf)
[126](https://arxiv.org/html/2407.10220v1)
[127](https://arxiv.org/html/2403.19160)
[128](https://arxiv.org/html/2406.18453v1)
[129](http://arxiv.org/pdf/2405.17016.pdf)
[130](https://www.oaepublish.com/articles/ais.2024.19)
[131](https://www.sciencedirect.com/science/article/abs/pii/S0925231224008208)
[132](https://pmc.ncbi.nlm.nih.gov/articles/PMC11888865/)
[133](https://openaccess.thecvf.com/content/CVPR2024/papers/Yan_Forecasting_of_3D_Whole-body_Human_Poses_with_Grasping_Objects_CVPR_2024_paper.pdf)
[134](https://www.semanticscholar.org/paper/SMPL:-A-Skinned-Multi-Person-Linear-Model-Loper-Mahmood/32d3048a4fe4becc7c4638afd05f2354b631cfca)
[135](https://arxiv.org/html/2510.26196v1)
[136](https://github.com/liuyangme/SOTA-3DHPE-HMR)
[137](https://www.sciencedirect.com/science/article/abs/pii/S1077314223000954)
[138](https://www.computer.org/csdl/journal/tg/2025/09/10706631/20QTqcSQZAk)
[139](https://papers.nips.cc/paper_files/paper/2024/file/158f036baa5b80a4fe2af094de8f7539-Paper-Conference.pdf)
[140](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1371385/pdf)
[141](https://arxiv.org/html/2508.21257v1)
[142](https://www.semanticscholar.org/paper/Domain-Adaptive-3D-Pose-Augmentation-for-Human-Mesh-Weng-Wang/418bea838e79a58ebc08f4af5760ef56ecec1b32)
[143](https://eehoeskrap.tistory.com/438)
